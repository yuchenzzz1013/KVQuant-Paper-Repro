import math
import torch
from tqdm import tqdm
from datasets import load_dataset


@torch.no_grad()
def evaluate_perplexity(
    model,
    tokenizer,
    dataset_name="wikitext",
    dataset_config="wikitext-2-raw-v1",
    seq_len=2048,
    stride=None,
    device="cuda",
):
    """滑动窗口 PPL 评估。

    - 拼接全文后按 stride 切窗，避免 token 边界处 loss 异常
    - 最后一个不满窗的 chunk 若长度 < 2 直接跳过
    - PPL = exp(Σ NLL / Σ valid_tokens)
    """
    dataset = load_dataset(dataset_name, dataset_config, split="test")
    text = "\n\n".join(t for t in dataset["text"] if t.strip())
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    L = input_ids.size(1)

    if stride is None:
        stride = seq_len  # 非重叠

    total_nll = 0.0
    total_tokens = 0
    prev_end = 0

    for begin in tqdm(range(0, L, stride), desc="Evaluating PPL"):
        end = min(begin + seq_len, L)
        if end - begin < 2:
            break

        input_chunk = input_ids[:, begin:end]
        target_chunk = input_chunk.clone()

        if prev_end > begin:
            target_chunk[:, : prev_end - begin] = -100

        outputs = model(input_chunk, labels=target_chunk, use_cache=False)
        valid = (target_chunk != -100).sum().item()
        total_nll += outputs.loss.item() * valid
        total_tokens += valid
        prev_end = end

    return math.exp(total_nll / max(total_tokens, 1))