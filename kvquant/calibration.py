import torch
from tqdm import tqdm

from kvquant.quant.nuq import train_nuq_codebook
from kvquant.quant.dense_sparse import extract_outliers


def calibrate(
    model,
    tokenizer,
    calib_texts,
    n_bits=3,
    seq_len=2048,
    device="cuda",
    outlier_ratio=0.01,
    sink_token=True,
):
    """
    离线校准：
      - Key:   逐通道 scale/zero（离线）+ 逐层 NUQ codebook（Fisher 加权）
      - Value: 逐层 NUQ codebook（Fisher 加权），scale/zero 留给在线逐 token 计算
    论文 3.3 / 3.4 / 3.5 / 3.6。
    """
    model.eval()
    n_layers = len(model.model.layers)

    act_store, grad_store = {}, {}
    hooks = []

    for i, layer in enumerate(model.model.layers):
        k_proj = layer.self_attn.k_proj
        v_proj = layer.self_attn.v_proj

        def make_act_hook(name):
            def hook(module, inp, out):
                act_store[name] = out.detach()
            return hook

        def make_grad_hook(name):
            def hook(module, grad_in, grad_out):
                grad_store[name] = grad_out[0].detach()
            return hook

        hooks.append(k_proj.register_forward_hook(make_act_hook(f"k_{i}")))
        hooks.append(v_proj.register_forward_hook(make_act_hook(f"v_{i}")))
        hooks.append(k_proj.register_full_backward_hook(make_grad_hook(f"k_{i}")))
        hooks.append(v_proj.register_full_backward_hook(make_grad_hook(f"v_{i}")))

    fisher_k = {i: None for i in range(n_layers)}
    fisher_v = {i: None for i in range(n_layers)}
    k_acts = {i: [] for i in range(n_layers)}
    v_acts = {i: [] for i in range(n_layers)}

    for text in tqdm(calib_texts, desc="Calibrating"):
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=seq_len
        ).to(device)
        labels = inputs["input_ids"].clone()

        model.zero_grad()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()

        for i in range(n_layers):
            k_act = act_store.get(f"k_{i}")
            v_act = act_store.get(f"v_{i}")
            k_grad = grad_store.get(f"k_{i}")
            v_grad = grad_store.get(f"v_{i}")

            if k_act is not None and k_grad is not None:
                if sink_token and k_act.size(1) > 1:
                    k_act = k_act[:, 1:, :].contiguous()
                    k_grad = k_grad[:, 1:, :].contiguous()
                k_acts[i].append(k_act.float().cpu())
                # 逐通道 Fisher：对 B、S 求和 -> [H]
                g = (k_grad.float() ** 2).sum(dim=(0, 1)).cpu()
                fisher_k[i] = g if fisher_k[i] is None else fisher_k[i] + g

            if v_act is not None and v_grad is not None:
                if sink_token and v_act.size(1) > 1:
                    v_act = v_act[:, 1:, :].contiguous()
                    v_grad = v_grad[:, 1:, :].contiguous()
                v_acts[i].append(v_act.float().cpu())
                g = (v_grad.float() ** 2).sum(dim=(0, 1)).cpu()
                fisher_v[i] = g if fisher_v[i] is None else fisher_v[i] + g

        for k in list(act_store.keys()):
            del act_store[k]
        for k in list(grad_store.keys()):
            del grad_store[k]

    for h in hooks:
        h.remove()

    k_codebooks, v_codebooks = [], []
    k_scales, k_zeros = [], []

    for i in range(n_layers):
        # ---------- Key: Per-Channel (Pre-RoPE) ----------
        k_cat = torch.cat(k_acts[i], dim=0)              # [N, S, H]
        k_flat = k_cat.reshape(-1, k_cat.shape[-1])      # [N*S, H]

        k_min = k_flat.min(dim=0).values
        k_max = k_flat.max(dim=0).values
        scale = (k_max - k_min) / 2.0
        zero = (k_max + k_min) / 2.0

        k_norm = torch.clamp((k_flat - zero) / (scale + 1e-8), -1, 1)  # [N*S, H]

        # 逐通道剔除 outlier（dim=0：沿 token 维度取 top-k）
        k_dense, _, _ = extract_outliers(
            k_norm, outlier_ratio=outlier_ratio, dim=0
        )
        # 只对非 outlier 元素训练码本（论文 3.4）
        nonzero_mask = (k_dense != 0)
        k_train = k_dense[nonzero_mask]

        # 逐元素 Fisher 权重（通道广播）
        fisher = fisher_k[i].flatten()                                       # [H]
        weights = fisher.unsqueeze(0).expand(k_dense.shape[0], -1)           # [N*S, H]
        weights = weights[nonzero_mask]

        codebook_k = train_nuq_codebook(k_train, weights, n_bits=n_bits)
        k_codebooks.append(codebook_k)
        k_scales.append(scale)
        k_zeros.append(zero)

        # ---------- Value: Per-Token (共享 per-layer 码本) ----------
        v_cat = torch.cat(v_acts[i], dim=0)              # [N, S, H]
        v_flat = v_cat.reshape(-1, v_cat.shape[-1])      # [N*S, H]

        v_min = v_flat.min(dim=-1, keepdim=True).values
        v_max = v_flat.max(dim=-1, keepdim=True).values
        v_scale = (v_max - v_min) / 2.0
        v_zero = (v_max + v_min) / 2.0

        v_norm = torch.clamp((v_flat - v_zero) / (v_scale + 1e-8), -1, 1)

        # 逐 token 剔除 outlier（dim=-1：沿 channel 维度取 top-k）
        v_dense, _, _ = extract_outliers(
            v_norm, outlier_ratio=outlier_ratio, dim=-1
        )
        nonzero_mask_v = (v_dense != 0)
        v_train = v_dense[nonzero_mask_v]

        fisher_v_i = fisher_v[i].flatten()                                   # [H]
        weights_v = fisher_v_i.unsqueeze(0).expand(v_dense.shape[0], -1)
        weights_v = weights_v[nonzero_mask_v]

        codebook_v = train_nuq_codebook(v_train, weights_v, n_bits=n_bits)
        v_codebooks.append(codebook_v)

    return {
        "k_codebooks": k_codebooks,
        "v_codebooks": v_codebooks,
        "k_scales": k_scales,
        "k_zeros": k_zeros,
        "outlier_ratio": outlier_ratio,
        "sink_token": sink_token,
    }