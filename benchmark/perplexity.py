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
    device="cuda",
):
    dataset = load_dataset(dataset_name, dataset_config, split="test")
    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)

    nlls = []
    for i in tqdm(range(0, input_ids.size(1), seq_len), desc="Evaluating PPL"):
        batch = input_ids[:, i : i + seq_len]
        if batch.size(1) < 2:
            continue
        outputs = model(batch, labels=batch)
        loss = outputs.loss
        nlls.append(loss * batch.size(1))

    ppl = torch.exp(torch.stack(nlls).sum() / input_ids.size(1))
    return ppl.item()