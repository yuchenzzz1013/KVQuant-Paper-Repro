import torch


def extract_outliers(x: torch.Tensor, outlier_ratio: float = 0.01, dim: int = -1):
    """按 dim 提取 outlier（论文 §3.4 Per-Vector Dense-and-Sparse）。

    x: [..., D]
    outlier_ratio: 每个 vector 中 outlier 占比
    dim: 沿哪个维度挑 outlier
         - Key  per-channel: dim=1（token 维，[B,S,H] 中 S 是 token）
         - Value per-token : dim=-1（channel 维）

    返回:
        dense_x:  与 x 同形状，outlier 位置置 0
        outliers: 与 x 同形状，仅 outlier 位置非 0
        mask:     bool，True 表示该位置是 outlier
    """
    if outlier_ratio <= 0:
        mask = torch.zeros_like(x, dtype=torch.bool)
        return x.clone(), torch.zeros_like(x), mask

    n = x.shape[dim]
    k = max(1, min(int(n * outlier_ratio), n))

    # 用 topk.indices 生成 mask，避免阈值并列导致 > k 个 outlier
    abs_x = x.abs()
    topk_idx = torch.topk(abs_x, k, dim=dim).indices
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask.scatter_(dim, topk_idx, True)

    dense_x = x.masked_fill(mask, 0.0)
    outliers = x - dense_x  # 等价于 x * mask，但数值更稳
    return dense_x, outliers, mask


def pack_csr(x: torch.Tensor):
    """CSR 打包。x: [R, C] -> (values, col_indices, row_ptr)"""
    R, C = x.shape
    mask = x != 0
    values = x[mask]
    col_indices = mask.nonzero(as_tuple=False)[:, 1].to(torch.int32)
    row_counts = mask.sum(dim=1)
    row_ptr = torch.zeros(R + 1, dtype=torch.int32, device=x.device)
    row_ptr[1:] = row_counts.cumsum(0).to(torch.int32)
    return values, col_indices, row_ptr


def unpack_csr(values, col_indices, row_ptr, R: int, C: int,
               device=None, dtype=torch.float16):
    if device is None:
        device = values.device
    out = torch.zeros(R, C, device=device, dtype=dtype)
    row_ptr = row_ptr.to(torch.long)
    for r in range(R):
        s, e = int(row_ptr[r].item()), int(row_ptr[r + 1].item())
        if e > s:
            out[r, col_indices[s:e].long()] = values[s:e].to(dtype)
    return out