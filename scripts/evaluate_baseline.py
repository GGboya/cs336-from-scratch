from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from cs336_basics.model import TransformerLM
from cs336_basics.train import evaluate, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on the validation set")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--valid-data", required=True)
    parser.add_argument("--eval-batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
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
    valid_data = np.memmap(args.valid_data, dtype=np.uint16, mode="r")
    eval_args = SimpleNamespace(
        eval_batches=args.eval_batches,
        batch_size=args.batch_size,
        context_length=config["context_length"],
        amp=device.type == "cuda",
    )
    np.random.seed(2026)
    valid_loss = evaluate(model, valid_data, eval_args, device)
    result = {
        "checkpoint_iteration": checkpoint["iteration"],
        "eval_batches": args.eval_batches,
        "batch_size": args.batch_size,
        "valid_loss": valid_loss,
        "valid_perplexity": float(np.exp(valid_loss)),
    }
    with (run_dir / "final_evaluation.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
