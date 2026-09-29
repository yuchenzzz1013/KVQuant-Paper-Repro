import torch
from tqdm import tqdm

from kvquant.quant.nuq import train_nuq_codebook
from kvquant.quant.dense_sparse import extract_outliers


def calibrate(
    model,
    tokenizer,
    calib_texts,
    n_bits: int = 3,
    layer_bits: dict = None,
    seq_len: int = 2048,
    device: str = "cuda",
    outlier_ratio: float = 0.01,
    sink_token: bool = True,
):
    model.eval()
    n_layers = len(model.model.layers)

    if layer_bits is None:
        layer_bits = {i: n_bits for i in range(n_layers)}

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

        model.zero_grad(set_to_none=True)
        outputs = model(**inputs, labels=labels)
        outputs.loss.backward()

        for i in range(n_layers):
            k_act = act_store.get(f"k_{i}")
            v_act = act_store.get(f"v_{i}")
            k_grad = grad_store.get(f"k_{i}")
            v_grad = grad_store.get(f"v_{i}")

            if k_act is not None and k_grad is not None:
                if sink_token and k_act.size(1) > 1:
                    k_act = k_act[:, 1:, :]
                    k_grad = k_grad[:, 1:, :]
                k_acts[i].append(k_act.float().cpu())
                g = (k_grad.float() ** 2).sum(dim=(0, 1)).cpu()
                fisher_k[i] = g if fisher_k[i] is None else fisher_k[i] + g

            if v_act is not None and v_grad is not None:
                if sink_token and v_act.size(1) > 1:
                    v_act = v_act[:, 1:, :]
                    v_grad = v_grad[:, 1:, :]
                v_acts[i].append(v_act.float().cpu())
                g = (v_grad.float() ** 2).sum(dim=(0, 1)).cpu()
                fisher_v[i] = g if fisher_v[i] is None else fisher_v[i] + g

        act_store.clear()
        grad_store.clear()

    for h in hooks:
        h.remove()

    k_codebooks, v_codebooks = [], []
    k_scales, k_zeros = [], []

    for i in range(n_layers):
        bits_i = layer_bits[i]

        # ---------------- Key: Per-Channel Pre-RoPE ----------------
        # k_cat: [n_samples, S, C] -> reshape 成 [N, C]
        k_cat = torch.cat(k_acts[i], dim=0)
        C = k_cat.shape[-1]
        k_flat = k_cat.reshape(-1, C)

        k_min = k_flat.min(dim=0).values   # [C]
        k_max = k_flat.max(dim=0).values   # [C]
        scale = (k_max - k_min) / 2.0      # [C]
        zero = (k_max + k_min) / 2.0       # [C]

        # 不 clamp 先归一化
        k_norm_raw = (k_flat - zero) / (scale + 1e-8)
        dense_raw, _, k_mask = extract_outliers(
            k_norm_raw, outlier_ratio=outlier_ratio, dim=0
        )
        dense_k = torch.clamp(dense_raw, -1.0, 1.0)

        # Fisher 权重 × scale²（per-channel）
        fisher = fisher_k[i].flatten()                 # [C]
        w_channel = fisher * (scale ** 2)              # [C]
        weights = w_channel.unsqueeze(0).expand_as(dense_k)  # [N, C]

        k_train = dense_k[~k_mask]
        w_train = weights[~k_mask]
        if k_train.numel() == 0:
            k_train = dense_k.reshape(-1)
            w_train = weights.reshape(-1)

        k_codebooks.append(train_nuq_codebook(k_train, w_train, n_bits=bits_i))
        k_scales.append(scale)
        k_zeros.append(zero)

        # ---------------- Value: Per-Token ----------------
        v_cat = torch.cat(v_acts[i], dim=0)
        Cv = v_cat.shape[-1]
        v_flat = v_cat.reshape(-1, Cv)

        v_min = v_flat.min(dim=-1, keepdim=True).values   # [N, 1]
        v_max = v_flat.max(dim=-1, keepdim=True).values   # [N, 1]
        v_scale = (v_max - v_min) / 2.0
        v_zero = (v_max + v_min) / 2.0

        v_norm_raw = (v_flat - v_zero) / (v_scale + 1e-8)
        dense_raw_v, _, v_mask = extract_outliers(
            v_norm_raw, outlier_ratio=outlier_ratio, dim=-1
        )
        dense_v = torch.clamp(dense_raw_v, -1.0, 1.0)

        # Fisher 权重 × scale²（per-token）：[1, C] * [N, 1] -> [N, C]
        fisher_v_i = fisher_v[i].flatten()             # [C]
        weights_v = fisher_v_i.unsqueeze(0) * (v_scale ** 2)

        v_train = dense_v[~v_mask]
        wv_train = weights_v[~v_mask]
        if v_train.numel() == 0:
            v_train = dense_v.reshape(-1)
            wv_train = weights_v.reshape(-1)

        v_codebooks.append(train_nuq_codebook(v_train, wv_train, n_bits=bits_i))

    return {
        "k_codebooks": k_codebooks,
        "v_codebooks": v_codebooks,
        "k_scales": k_scales,
        "k_zeros": k_zeros,
        "layer_bits": layer_bits,
        "n_bits": n_bits,
        "outlier_ratio": outlier_ratio,
        "sink_token": sink_token,
    }