import torch

from .dense_sparse import extract_outliers
from ..kernels.cuda_ops import get_cuda_ops


def _lut_lookup(x_norm: torch.Tensor, codebook: torch.Tensor):
    """向量化 LUT 查找（CUDA 优先）。

    x_norm:   [..., D]，已归一化到 [-1,1]
    codebook: [2^b]
    返回: idx [..., D] int64, dequant [..., D] fp32
    """
    cuda_ops = get_cuda_ops()
    if cuda_ops is not None and x_norm.is_cuda:
        cb = codebook.to(x_norm.device, dtype=torch.float32).reshape(-1).contiguous()
        idx, dq = cuda_ops.lut_lookup(x_norm.reshape(-1).float(), cb)
        return idx.reshape(x_norm.shape), dq.reshape(x_norm.shape)

    # PyTorch 回退
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
        self.outlier_ratio = outlier_ratio
        self.sink_token = sink_token
        self.use_dense_sparse = use_dense_sparse

    # ---------- 内部 ----------
    def _dense_sparse(self, x_norm, dim):
        if self.use_dense_sparse and self.outlier_ratio > 0:
            dense_x, outliers, _ = extract_outliers(
                x_norm, outlier_ratio=self.outlier_ratio, dim=dim
            )
        else:
            dense_x = x_norm
            outliers = torch.zeros_like(x_norm)
        return dense_x, outliers

    @staticmethod
    def _split_sink(x, sink: bool):
        if sink and x.size(1) > 1:
            return x[:, :1, :].contiguous(), x[:, 1:, :].contiguous()
        return None, x

    # ---------- Key: Per-Channel + Pre-RoPE ----------
    def quantize_k(self, layer_idx: int, k: torch.Tensor):
        """k: [B, S, H]，k_proj 之后、RoPE 之前。"""
        codebook = self.k_codebooks[layer_idx]
        scale = self.k_scales[layer_idx].to(k.device).float()
        zero = self.k_zeros[layer_idx].to(k.device).float()

        k_sink, k_rest = self._split_sink(k, self.sink_token)
        if k_rest.numel() == 0:
            return k

        k_norm = torch.clamp((k_rest.float() - zero) / (scale + 1e-8), -1, 1)
        # dim=1（token 维）挑 outlier，per-channel
        dense_k, outliers_k = self._dense_sparse(k_norm, dim=1)
        _, q = _lut_lookup(dense_k, codebook)
        k_dequant = (q + outliers_k) * scale + zero

        k_dequant = k_dequant.to(k.dtype)
        if k_sink is not None:
            k_dequant = torch.cat([k_sink, k_dequant], dim=1)
        return k_dequant

    # ---------- Value: Per-Token ----------
    def quantize_v(self, layer_idx: int, v: torch.Tensor):
        """v: [B, S, H]"""
        codebook = self.v_codebooks[layer_idx]

        v_sink, v_rest = self._split_sink(v, self.sink_token)
        if v_rest.numel() == 0:
            return v

        vf = v_rest.float()
        v_min = vf.min(dim=-1, keepdim=True).values
        v_max = vf.max(dim=-1, keepdim=True).values
        scale = (v_max - v_min) / 2.0
        zero = (v_max + v_min) / 2.0

        v_norm = torch.clamp((vf - zero) / (scale + 1e-8), -1, 1)
        # dim=-1（channel 维）挑 outlier，per-token
        dense_v, outliers_v = self._dense_sparse(v_norm, dim=-1)
        _, q = _lut_lookup(dense_v, codebook)
        v_dequant = (q + outliers_v) * scale + zero

        v_dequant = v_dequant.to(v.dtype)
        if v_sink is not None:
            v_dequant = torch.cat([v_sink, v_dequant], dim=1)
        return v_dequant

    # ---------- 压缩存储（用于真实内存节省 / kernel 复现） ----------
    def compress_k_to_csr(self, layer_idx: int, k: torch.Tensor):
        from .dense_sparse import pack_csr

        codebook = self.k_codebooks[layer_idx]
        scale = self.k_scales[layer_idx].to(k.device).float()
        zero = self.k_zeros[layer_idx].to(k.device).float()

        k_norm = torch.clamp((k.float() - zero) / (scale + 1e-8), -1, 1)
        dense_x, outliers, _ = extract_outliers(
            k_norm, outlier_ratio=self.outlier_ratio, dim=1
        )
        idx, _ = _lut_lookup(dense_x, codebook)

        B, S, H = outliers.shape
        # [B, S, H] -> [B*H, S]，沿 token 维稀疏（等价 CSC-on-K）
        outliers_2d = outliers.permute(0, 2, 1).reshape(B * H, S)
        values, col_indices, row_ptr = pack_csr(outliers_2d)

        return {
            "q_idx": idx.to(torch.uint8),
            "outlier_values": values,
            "outlier_cols": col_indices,
            "outlier_rowptr": row_ptr,
            "scale": scale,
            "zero": zero,
            "shape": (B, S, H),
            "codebook": codebook,
        }

    def compress_v_to_csr(self, layer_idx: int, v: torch.Tensor):
        from .dense_sparse import pack_csr

        codebook = self.v_codebooks[layer_idx]

        vf = v.float()
        v_min = vf.min(dim=-1, keepdim=True).values
        v_max = vf.max(dim=-1, keepdim=True).values
        scale = (v_max - v_min) / 2.0
        zero = (v_max + v_min) / 2.0

        v_norm = torch.clamp((vf - zero) / (scale + 1e-8), -1, 1)
        dense_x, outliers, _ = extract_outliers(
            v_norm, outlier_ratio=self.outlier_ratio, dim=-1
        )
        idx, _ = _lut_lookup(dense_x, codebook)

        B, S, H = outliers.shape
        outliers_2d = outliers.reshape(B * S, H)  # 每 token 一行
        values, col_indices, row_ptr = pack_csr(outliers_2d)

        return {
            "q_idx": idx.to(torch.uint8),
            "outlier_values": values,
            "outlier_cols": col_indices,
            "outlier_rowptr": row_ptr,
            "scale": scale,
            "zero": zero,
            "shape": (B, S, H),
            "codebook": codebook,
        }