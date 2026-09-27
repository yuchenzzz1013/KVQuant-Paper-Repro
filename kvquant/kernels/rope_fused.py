import torch


def _build_cos_sin(seq_len: int, dim: int, device, dtype=torch.float32, base: float = 10000.0):
    """
    生成 RoPE 的 cos / sin 表：
        cos: [seq_len, dim // 2]
        sin: [seq_len, dim // 2]
    """
    half = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=dtype) / half))
    t = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.outer(t, inv_freq)   # [seq_len, half]
    return freqs.cos(), freqs.sin()


def apply_rope_elementwise(x: torch.Tensor, position_offset: int = 0, base: float = 10000.0):
    """
    对 x 应用 RoPE（element-wise 形式，等价于论文式 (3)）。

    x: [B, S, H] 或 [B, H, S, D]
    返回同形状张量。
    """
    assert x.dim() in (3, 4), f"不支持的输入维度: {x.shape}"

    orig_dim = x.dim()
    if orig_dim == 3:
        # [B, S, H] -> [B, S, 1, H]，把 head_dim 当作最后一维
        x4 = x.unsqueeze(2)
    else:
        x4 = x

    B, H, S, D = x4.shape
    half = D // 2

    cos, sin = _build_cos_sin(
        seq_len=S + position_offset,
        dim=D,
        device=x.device,
        dtype=x.dtype,
        base=base,
    )
    cos = cos[position_offset:position_offset + S]   # [S, half]
    sin = sin[position_offset:position_offset + S]   # [S, half]

    cos = cos.view(1, 1, S, half)
    sin = sin.view(1, 1, S, half)

    x1 = x4[..., :half]
    x2 = x4[..., half:]

    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos

    out = torch.cat([out1, out2], dim=-1)

    if orig_dim == 3:
        out = out.squeeze(2)
    return out


def apply_rope_inplace(x: torch.Tensor, position_offset: int = 0, base: float = 10000.0):
    """inplace 版本，直接覆盖 x。"""
    out = apply_rope_elementwise(x, position_offset=position_offset, base=base)
    x.copy_(out)
    return x