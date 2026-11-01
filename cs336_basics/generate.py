from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .decoding import generate
from .model import TransformerLM
from .tokenizer import Tokenizer
from .train import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a trained Transformer LM")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--merges", required=True)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--special-token", action="append", default=["<|endoftext|>"])
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    with (run_dir / "config.json").open(encoding="utf-8") as file:
        config = json.load(file)
    device = resolve_device(args.device)
    model = TransformerLM(
        config["vocab_size"],
        config["context_length"],
        config["d_model"],
        config["num_layers"],
        config["num_heads"],
        config["d_ff"],
        config["rope_theta"],
        device=device,
    )
    checkpoint = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    tokenizer = Tokenizer.from_files(args.vocab, args.merges, args.special_token)
    text = generate(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(text)


if __name__ == "__main__":
    main()
