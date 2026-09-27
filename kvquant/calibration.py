import torch
from tqdm import tqdm

from kvquant.quant.nuq import train_nuq_codebook


def calibrate(model, tokenizer, calib_texts, n_bits=3, seq_len=2048, device="cuda"):
    model.eval()
    n_layers = len(model.model.layers)

    act_store = {}
    grad_store = {}
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
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss

        model.zero_grad()
        loss.backward()

        for i in range(n_layers):
            k_act = act_store.get(f"k_{i}")
            v_act = act_store.get(f"v_{i}")
            k_grad = grad_store.get(f"k_{i}")
            v_grad = grad_store.get(f"v_{i}")

            if k_act is not None and k_grad is not None:
                k_acts[i].append(k_act.float().cpu())
                # 关键修复：grad 形状是 [B, S, H]，对 B、S 求和得每通道敏感度
                g = (k_grad.float() ** 2).sum(dim=(0, 1)).cpu()
                fisher_k[i] = g if fisher_k[i] is None else fisher_k[i] + g

            if v_act is not None and v_grad is not None:
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
        # ---- Key：Per-Channel ----
        k_cat = torch.cat(k_acts[i], dim=0)          # [N, S, H]
        k_cat = k_cat.reshape(-1, k_cat.shape[-1])   # [N*S, H]

        k_min = k_cat.min(dim=0).values
        k_max = k_cat.max(dim=0).values
        scale = (k_max - k_min) / 2.0
        zero = (k_max + k_min) / 2.0

        k_norm = torch.clamp((k_cat - zero) / (scale + 1e-8), -1, 1)

        # 关键修复：Fisher 是 per-channel 的，需要按每个元素展开给 kmeans
        fisher = fisher_k[i].flatten()                     # [H]
        # 每个通道的权重复制到该通道所有元素
        fisher_weights = fisher.unsqueeze(0).expand(k_norm.shape[0], -1)  # [N*S, H]
        fisher_weights = fisher_weights.reshape(-1)        # [N*S*H]

        codebook_k = train_nuq_codebook(
            k_norm.flatten(), fisher_weights, n_bits=n_bits
        )
        k_codebooks.append(codebook_k)
        k_scales.append(scale)
        k_zeros.append(zero)

        # ---- Value：Per-Token，共享 per-layer 码本 ----
        v_cat = torch.cat(v_acts[i], dim=0)          # [N, S, H]
        v_min = v_cat.min(dim=-1, keepdim=True).values
        v_max = v_cat.max(dim=-1, keepdim=True).values
        v_scale = (v_max - v_min) / 2.0
        v_zero = (v_max + v_min) / 2.0

        v_norm = torch.clamp((v_cat - v_zero) / (v_scale + 1e-8), -1, 1)

        weights = torch.ones_like(v_norm).reshape(-1)
        codebook_v = train_nuq_codebook(v_norm.flatten(), weights, n_bits=n_bits)
        v_codebooks.append(codebook_v)

    return {
        "k_codebooks": k_codebooks,
        "v_codebooks": v_codebooks,
        "k_scales": k_scales,
        "k_zeros": k_zeros,
    }