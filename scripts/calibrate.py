import argparse
import pickle

import torch
from datasets import load_dataset

from kvquant.model_loader import load_hf_model
from kvquant.calibration import calibrate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="model")
    parser.add_argument("--output", default="kvquant_calib.pkl")
    parser.add_argument("--n_bits", type=int, default=3)
    parser.add_argument("--n_samples", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--outlier_ratio", type=float, default=0.01)
    parser.add_argument("--no_sink", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_hf_model(
        args.model_path, device_map=None, torch_dtype=torch.float16
    )
    model.to(device)

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t for t in dataset["text"] if t.strip()][: args.n_samples]

    calib = calibrate(
        model,
        tokenizer,
        texts,
        n_bits=args.n_bits,
        seq_len=args.seq_len,
        device=device,
        outlier_ratio=args.outlier_ratio,
        sink_token=not args.no_sink,
    )

    with open(args.output, "wb") as f:
        pickle.dump(calib, f)
    print(f"Saved calibration to {args.output}")


if __name__ == "__main__":
    main()