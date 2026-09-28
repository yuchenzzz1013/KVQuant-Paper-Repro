import torch

from .dense_sparse import extract_outliers, pack_csr


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

    # ---------- 内部工具 ----------
    def _quantize_to_codebook(self, x_norm: torch.Tensor, codebook: torch.Tensor):
        codebook = codebook.to(x_norm.device).float()
        x_flat = x_norm.reshape(-1, 1).float()
        cb = codebook.reshape(1, -1)
        idx = (x_flat - cb).abs().argmin(dim=-1)
        return codebook[idx].reshape(x_norm.shape)

    def _dense_sparse(self, x_norm, dim):
        if self.use_dense_sparse and self.outlier_ratio > 0:
            dense_x, outliers, _ = extract_outliers(
                x_norm, outlier_ratio=self.outlier_ratio, dim=dim
            )
        else:
            dense_x = x_norm
            outliers = torch.zeros_like(x_norm)
        return dense_x, outliers

    # ---------- Key: Per-Channel + Pre-RoPE ----------
    def quantize_k(self, layer_idx: int, k: torch.Tensor):
        """
        k: [B, S, H]，来自 k_proj，RoPE 之前。
        """
        codebook = self.k_codebooks[layer_idx]
        scale = self.k_scales[layer_idx].to(k.device).float()   # [H]
        zero = self.k_zeros[layer_idx].to(k.device).float()      # [H]

        if self.sink_token:
            k_sink = k[:, :1, :]
            k_rest = k[:, 1:, :]
        else:
            k_sink, k_rest = None, k

        if k_rest.numel() == 0:
            return k

        # 逐通道归一化（channel 为最后一维）
        k_norm = torch.clamp((k_rest - zero) / (scale + 1e-8), -1, 1)

        # 逐通道 outlier：沿 token 维度（dim=1）
        dense_k, outliers_k = self._dense_sparse(k_norm, dim=1)

        q = self._quantize_to_codebook(dense_k, codebook)

        # 反量化: k ≈ (q + outliers) * scale + zero
        k_dequant = (q + outliers_k) * scale + zero

        if self.sink_token:
            k_dequant = torch.cat([k_sink, k_dequant], dim=1)
        return k_dequant

    # ---------- Value: Per-Token ----------
    def quantize_v(self, layer_idx: int, v: torch.Tensor):
        """
        v: [B, S, H]，来自 v_proj。
        """
        codebook = self.v_codebooks[layer_idx]

        if self.sink_token:
            v_sink = v[:, :1, :]
            v_rest = v[:, 1:, :]
        else:
            v_sink, v_rest = None, v

        if v_rest.numel() == 0:
            return v

        # 在线逐 token min/max
        v_min = v_rest.min(dim=-1, keepdim=True).values
        v_max = v_rest.max(dim=-1, keepdim=True).values
        scale = (v_max - v_min) / 2.0
        zero = (v_max + v_min) / 2.0

        v_norm = torch.clamp((v_rest - zero) / (scale + 1e-8), -1, 1)

        # 逐 token outlier：沿 channel 维度（dim=-1）
        dense_v, outliers_v = self._dense_sparse(v_norm, dim=-1)

        q = self._quantize_to_codebook(dense_v, codebook)
        v_dequant = (q + outliers_v) * scale + zero

        if self.sink_token:
            v_dequant = torch.cat([v_sink, v_dequant], dim=1)
        return v_dequant

    # ---------- 压缩存储接口 ----------
    def compress_k_to_csr(self, layer_idx: int, k: torch.Tensor):
        codebook = self.k_codebooks[layer_idx]
        scale = self.k_scales[layer_idx].to(k.device).float()
        zero = self.k_zeros[layer_idx].to(k.device).float()

        k_norm = torch.clamp((k - zero) / (scale + 1e-8), -1, 1)
        dense_x, outliers, _ = extract_outliers(
            k_norm, outlier_ratio=self.outlier_ratio, dim=1
        )

        codebook = codebook.to(k.device).float()
        flat = dense_x.reshape(-1, 1)
        cb = codebook.reshape(1, -1)
        q_idx = (flat - cb).abs().argmin(dim=-1).reshape(dense_x.shape).to(torch.uint8)

        B, S, H = outliers.shape
        outliers_2d = outliers.permute(0, 2, 1).reshape(B * H, S)
        values, col_indices, row_ptr = pack_csr(outliers_2d)

        return {
            "q_idx": q_idx,
            "outlier_values": values,
            "outlier_cols": col_indices,
            "outlier_rowptr": row_ptr,
            "scale": scale,
            "zero": zero,
            "shape": (B, S, H),
            "codebook": codebook,
        }