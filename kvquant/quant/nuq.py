import numpy as np
import torch
from sklearn.cluster import KMeans


def train_nuq_codebook(
    activations: torch.Tensor,
    fisher: torch.Tensor,
    n_bits: int = 3,
):
    """
    activations: [N] 已归一化到 [-1, 1]
    fisher:      [N] 敏感度权重
    返回:         [2^n_bits] 排序后的码本中心
    """
    n_clusters = 2 ** n_bits
    X = activations.reshape(-1).float().cpu().numpy()
    w = fisher.reshape(-1).float().cpu().numpy()

    if X.size == 0:
        return torch.zeros(n_clusters, dtype=torch.float32)

    w = w / (w.sum() + 1e-8)

    # -------- 边界情况 1：样本数不足 --------
    if X.size < n_clusters:
        centers = np.sort(X)
        pad = np.full(n_clusters - centers.size, centers[-1])
        return torch.tensor(np.sort(np.concatenate([centers, pad])),
                            dtype=torch.float32)

    # -------- 边界情况 2：唯一值不足 --------
    uniq = np.unique(X)
    if uniq.size < n_clusters:
        pad = np.full(n_clusters - uniq.size, uniq[-1])
        return torch.tensor(np.sort(np.concatenate([uniq, pad])),
                            dtype=torch.float32)

    kmeans = KMeans(
        n_clusters=n_clusters, n_init=10, max_iter=100, random_state=0
    )
    kmeans.fit(X.reshape(-1, 1), sample_weight=w)
    centers = torch.tensor(kmeans.cluster_centers_.flatten(), dtype=torch.float32)
    return torch.sort(centers).values