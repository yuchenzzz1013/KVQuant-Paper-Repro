import torch


def extract_outliers(x: torch.Tensor, outlier_ratio: float = 0.01, dim: int = -1):
    """
    按 dim 提取 outlier（论文 3.4 Per-Vector Dense-and-Sparse）。

    x: [..., D]
    返回:
        dense_x:  去掉 outlier 后，outlier 位置置 0
        outliers: 只保留 outlier 值，其余为 0
        mask:     bool 掩码，True 表示 outlier
    """
    if outlier_ratio <= 0:
        mask = torch.zeros_like(x, dtype=torch.bool)
        return x, torch.zeros_like(x), mask

    k = max(1, int(x.shape[dim] * outlier_ratio))
    abs_x = x.abs()
    topk_vals = torch.topk(abs_x, k, dim=dim).values
    threshold = topk_vals[..., -1:].expand_as(x)
    mask = abs_x >= threshold

    dense_x = x.clone()
    dense_x[mask] = 0.0
    outliers = x * mask
    return dense_x, outliers, mask


def pack_csr(x: torch.Tensor):
    """
    简单 CSR 打包（PyTorch 参考）。

    x: [R, C]
    返回: (values, col_indices, row_ptr)
    """
    R, C = x.shape
    mask = x != 0
    values = x[mask]
    col_indices = mask.nonzero(as_tuple=False)[:, 1]
    row_counts = mask.sum(dim=1)
    row_ptr = torch.zeros(R + 1, dtype=torch.long, device=x.device)
    row_ptr[1:] = row_counts.cumsum(0)
    return values, col_indices, row_ptr


def unpack_csr(values, col_indices, row_ptr, R: int, C: int, device=None, dtype=torch.float16):
    """把 CSR 还原成 dense [R, C]。"""
    if device is None:
        device = values.device
    out = torch.zeros(R, C, device=device, dtype=dtype)
    for r in range(R):
        s, e = row_ptr[r].item(), row_ptr[r + 1].item()
        if e > s:
            out[r, col_indices[s:e]] = values[s:e].to(dtype)
    return out