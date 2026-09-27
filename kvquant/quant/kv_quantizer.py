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
        quant = codebook[idx].reshape(x_norm.shape)
        return quant

    def _quantize_per_vector(self, x_norm, codebook, dim):
        """per-vector（per-channel 或 per-token）量化 + dense-and-sparse。"""
        if self.use_dense_sparse and self.outlier_ratio > 0:
            dense_x, outliers, _ = extract_outliers(
                x_norm, outlier_ratio=self.outlier_ratio, dim=dim
            )
        else:
            dense_x, outliers = x_norm, torch.zeros_like(x_norm)

        q = self._quantize_to_codebook(dense_x, codebook)
        return q, outliers

    # ---------- Key：Per-Channel + Pre-RoPE ----------

    def quantize_k(self, layer_idx: int, k: torch.Tensor):
        """
        k: [B, S, H]  来自 k_proj 输出，尚未应用 RoPE。
        Per-Channel Key Quantization（论文 3.1）。
        """
        codebook = self.k_codebooks[layer_idx]
        scale = self.k_scales[layer_idx].to(k.device).float()
        zero = self.k_zeros[layer_idx].to(k.device).float()

        if self.sink_token:
            k_sink = k[:, :1, :]
            k_rest = k[:, 1:, :]
        else:
            k_sink = None
            k_rest = k

        if k_rest.numel() == 0:
            return k

        # 归一化到 [-1, 1]，按 channel 广播
        k_norm = (k_rest - zero) / (scale + 1e-8)
        k_norm = torch.clamp(k_norm, -1, 1)

        # per-channel 的 dim 是 S 维度（dim=1）
        q, outliers = self._quantize_per_vector(
            k_norm, codebook, dim=1
        )

        k_dequant = q * scale + zero
        k_dequant = k_dequant + outliers * scale

        if self.sink_token:
            k_dequant = torch.cat([k_sink, k_dequant], dim=1)

        return k_dequant

    # ---------- Value：Per-Token ----------

    def quantize_v(self, layer_idx: int, v: torch.Tensor):
        """
        v: [B, S, H]  来自 v_proj 输出。
        Per-Token Value Quantization（论文 3.1 / 3.6）。
        """
        codebook = self.v_codebooks[layer_idx]

        if self.sink_token:
            v_sink = v[:, :1, :]
            v_rest = v[:, 1:, :]
        else:
            v_sink = None
            v_rest = v

        if v_rest.numel() == 0:
            return v

        # 在线计算 per-token min/max
        v_min = v_rest.min(dim=-1, keepdim=True).values
        v_max = v_rest.max(dim=-1, keepdim=True).values
        scale = (v_max - v_min) / 2.0
        zero = (v_max + v_min) / 2.0

        v_norm = (v_rest - zero) / (scale + 1e-8)
        v_norm = torch.clamp(v_norm, -1, 1)

        # per-token 的 dim 是 H 维度（dim=-1）
        q, outliers = self._quantize_per_vector(
            v_norm, codebook, dim=-1
        )

        v_dequant = q * scale + zero
        v_dequant = v_dequant + outliers * scale

        if self.sink_token:
            v_dequant = torch.cat([v_sink, v_dequant], dim=1)

        return v_dequant

    # ---------- 存储接口（可选，真省显存用） ----------

    def compress_k_to_csr(self, layer_idx: int, k: torch.Tensor):
        """
        返回 CSR 压缩表示，用于真正节省显存。
        """
        codebook = self.k_codebooks[layer_idx]
        scale = self.k_scales[layer_idx].to(k.device).float()
        zero = self.k_zeros[layer_idx].to(k.device).float()

        k_norm = torch.clamp((k - zero) / (scale + 1e-8), -1, 1)
        dense_x, outliers, mask = extract_outliers(
            k_norm, outlier_ratio=self.outlier_ratio, dim=1
        )

        # 量化 indices（真正的 4-bit 存储可换成 bit-pack）
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