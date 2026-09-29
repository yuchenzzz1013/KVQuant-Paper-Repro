import torch

from .dense_sparse import extract_outliers
from ..kernels.cuda_ops import get_cuda_ops


def _lut_lookup(x_norm: torch.Tensor, codebook: torch.Tensor):
    """向量化 LUT 查找（CUDA 优先）。"""
    cuda_ops = get_cuda_ops()
    if cuda_ops is not None and x_norm.is_cuda:
        cb = codebook.to(x_norm.device, dtype=torch.float32).reshape(-1).contiguous()
        idx, dq = cuda_ops.lut_lookup(x_norm.reshape(-1).float(), cb)
        return idx.reshape(x_norm.shape), dq.reshape(x_norm.shape)

    cb = codebook.to(x_norm.device, dtype=torch.float32).reshape(-1)
    shape = x_norm.shape
    flat = x_norm.reshape(-1).float()
    dist = (flat.unsqueeze(1) - cb.unsqueeze(0)).abs()
    idx = dist.argmin(dim=-1)
    dequant = cb.gather(0, idx)
    return idx.reshape(shape), dequant.reshape(shape)


class KVQuantizer:
    def __init__(
        self,
        k_codebooks,
        v_codebooks,
        k_scales,
        k_zeros,
        k_bits: int = 3,
        v_bits: int = 3,
        layer_bits: dict = None,
        outlier_ratio: float = 0.01,
        sink_token: bool = True,
        use_dense_sparse: bool = True,
    ):
        self.k_codebooks = k_codebooks
        self.v_codebooks = v_codebooks
        self.k_scales = k_scales
        self.k_zeros = k_zeros
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.layer_bits = layer_bits
        self.outlier_ratio = outlier_ratio
        self.sink_token = sink_token
        self.use_dense_sparse = use_dense_sparse

    def _get_bits(self, layer_idx: int, is_key: bool) -> int:
        if self.layer_bits is not None and layer_idx in self.layer_bits:
            return self.layer_bits[layer_idx]
        return self.k_bits if is_key else self.v_bits

    @staticmethod
    def _split_sink(x, sink: bool):
        if sink and x.size(1) > 1:
            return x[:, :1, :].contiguous(), x[:, 1:, :].contiguous()
        return None, x

    def _split_outliers(self, x_norm_raw: torch.Tensor, dim: int):
        """先在原始归一化值上提取 outlier，再对 dense 部分 clamp。"""
        if self.use_dense_sparse and self.outlier_ratio > 0:
            dense_raw, outliers, _ = extract_outliers(
                x_norm_raw, outlier_ratio=self.outlier_ratio, dim=dim
            )
        else:
            dense_raw = x_norm_raw
            outliers = torch.zeros_like(x_norm_raw)
        dense = torch.clamp(dense_raw, -1.0, 1.0)
        return dense, outliers

    def quantize_k(self, layer_idx: int, k: torch.Tensor):
        codebook = self.k_codebooks[layer_idx]
        scale = self.k_scales[layer_idx].to(k.device).float()  # [C]
        zero = self.k_zeros[layer_idx].to(k.device).float()    # [C]

        k_sink, k_rest = self._split_sink(k, self.sink_token)
        if k_rest.numel() == 0:
            return k

        # 1) 用原始值做归一化，不 clamp
        k_norm_raw = (k_rest.float() - zero) / (scale + 1e-8)   # [B, S, C]

        # 2) outlier 提取（per-channel，dim=1 即 token 维）
        dense_k, outliers_k = self._split_outliers(k_norm_raw, dim=1)

        # 3) 查表反量化，再把 outlier 加回去
        _, q = _lut_lookup(dense_k, codebook)
        k_dequant = (q + outliers_k) * scale + zero

        k_dequant = k_dequant.to(k.dtype)
        if k_sink is not None:
            k_dequant = torch.cat([k_sink, k_dequant], dim=1)
        return k_dequant

    def quantize_v(self, layer_idx: int, v: torch.Tensor):
        codebook = self.v_codebooks[layer_idx]

        v_sink, v_rest = self._split_sink(v, self.sink_token)
        if v_rest.numel() == 0:
            return v

        vf = v_rest.float()
        v_min = vf.min(dim=-1, keepdim=True).values
        v_max = vf.max(dim=-1, keepdim=True).values
        scale = (v_max - v_min) / 2.0
        zero = (v_max + v_min) / 2.0

        # 1) 原始归一化，不 clamp
        v_norm_raw = (vf - zero) / (scale + 1e-8)               # [B, S, C]

        # 2) outlier 提取（per-token，dim=-1 即 channel 维）
        dense_v, outliers_v = self._split_outliers(v_norm_raw, dim=-1)

        # 3) 查表 + 加回 outlier
        _, q = _lut_lookup(dense_v, codebook)
        v_dequant = (q + outliers_v) * scale + zero

        v_dequant = v_dequant.to(v.dtype)
        if v_sink is not None:
            v_dequant = torch.cat([v_sink, v_dequant], dim=1)
        return v_dequant