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

    abs_x = x.abs()
    topk_idx = torch.topk(abs_x, k, dim=dim).indices
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask.scatter_(dim, topk_idx, True)

    dense_x = x.masked_fill(mask, 0.0)
    outliers = x - dense_x
    return dense_x, outliers, mask


def extract_outliers_attention(
    attn_weights: torch.Tensor,
    k: torch.Tensor,
    outlier_ratio: float = 0.01,
):
    """基于注意力分数的 Key outlier 提取（论文 §3.4 的注意力感知版本）。

    attn_weights: [B, H, S_q, S_k] 注意力分数（softmax 后）
    k:            [B, S_k, H_kv]  Key 张量

    对每个 head，按注意力分数对 Key 位置加权，选出对输出影响最大的 outlier。

    返回: outlier_mask [B, S_k, H_kv] bool
    """
    B, H_q, S_q, S_k = attn_weights.shape
    H_kv = k.size(-1)

    # GQA/MQA: 将 query head 的注意力分数聚合到 kv head
    n_rep = H_q // H_kv
    if n_rep > 1:
        # [B, H_q, S_q, S_k] -> [B, H_kv, n_rep, S_q, S_k] -> mean
        attn_agg = attn_weights.reshape(B, H_kv, n_rep, S_q, S_k).mean(dim=2)
    else:
        attn_agg = attn_weights

    # 每个 kv head 对每个 key 位置的总注意力：对 query 维求和
    # [B, H_kv, S_k]
    key_importance = attn_agg.sum(dim=2)  # sum over S_q

    # 对每个 head，按注意力分数 top-k 选 outlier
    n = S_k
    k_out = max(1, min(int(n * outlier_ratio), n))
    topk_idx = torch.topk(key_importance, k_out, dim=-1).indices  # [B, H_kv, k]

    mask = torch.zeros(B, S_k, H_kv, dtype=torch.bool, device=k.device)
    for h in range(H_kv):
        mask.scatter_(
            1,
            topk_idx[:, h, :].unsqueeze(-1).expand(-1, -1, 1),
            True,
        )
    return mask


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