from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from cs336_basics.decoding import generate
from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.train import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive TinyStories generator")
    parser.add_argument("--prompt", help="Generate once instead of entering interactive mode")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    run_dir = ROOT_DIR / "runs/tinystories_baseline"
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
    tokenizer = Tokenizer.from_files(
        ROOT_DIR / "data/tinystories_10k/vocab.json",
        ROOT_DIR / "data/tinystories_10k/merges.txt",
        ["<|endoftext|>"],
    )
    print(f"Model loaded on {device}. Max new tokens: {args.max_new_tokens}")

    def generate_story(prompt: str) -> None:
        text = generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print("\n" + text + "\n")

    if args.prompt:
        generate_story(args.prompt)
        return

    print("Enter an English story opening. Type 'quit' or press Ctrl-C to exit.")
    while True:
        try:
            prompt = input("\nStory opening> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if prompt.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break
        if prompt:
            generate_story(prompt)


if __name__ == "__main__":
    main()
