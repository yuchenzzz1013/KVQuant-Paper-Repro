import torch

from .cuda_ops import get_cuda_ops


def _build_cos_sin(seq_len, dim, device, dtype=torch.float32, base=10000.0):
    half = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device,
                                             dtype=torch.float32) / half))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope_elementwise(x: torch.Tensor,
                           position_offset: int = 0,
                           base: float = 10000.0):
    """对 x 应用 RoPE（element-wise 形式，等价于论文式 (3)）。

    若 CUDA 可用且张量为 fp16，则走 CUDA kernel，否则回退 PyTorch。
    输入维度支持 [B, S, H] 或 [B, H, S, D]。
    """
    assert x.dim() in (3, 4), f"不支持的输入维度: {x.shape}"
    orig_dim = x.dim()
    x4 = x.unsqueeze(2) if orig_dim == 3 else x   # 统一成 [B, H, S, D]
    B, H, S, D = x4.shape
    half_d = D // 2

    cuda_ops = get_cuda_ops()
    if cuda_ops is not None and x4.is_cuda and x4.dtype == torch.float16:
        cos, sin = _build_cos_sin(S + position_offset, D, x4.device,
                                  dtype=torch.float32, base=base)
        out4 = cuda_ops.rope_apply(x4.contiguous(), cos.contiguous(),
                                   sin.contiguous(), int(position_offset))
    else:
        cos, sin = _build_cos_sin(S + position_offset, D, x4.device,
                                  dtype=x4.dtype, base=base)
        cos = cos[position_offset:position_offset + S].view(1, 1, S, half_d)
        sin = sin[position_offset:position_offset + S].view(1, 1, S, half_d)
        x1 = x4[..., :half_d]
        x2 = x4[..., half_d:]
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        out4 = torch.cat([out1, out2], dim=-1)

    return out4.squeeze(2) if orig_dim == 3 else out4


def apply_rope_inplace(x: torch.Tensor,
                       position_offset: int = 0,
                       base: float = 10000.0):
    out = apply_rope_elementwise(x, position_offset=position_offset, base=base)
    x.copy_(out)
    return x