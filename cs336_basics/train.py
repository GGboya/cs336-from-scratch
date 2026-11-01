from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from .model import TransformerLM
from .training import (
    AdamW,
    cross_entropy,
    get_batch,
    get_lr_cosine_schedule,
    gradient_clipping,
    load_checkpoint,
    save_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Transformer language model")
    parser.add_argument("--train-data", required=True, help="Raw uint16 token IDs")
    parser.add_argument("--valid-data", required=True, help="Raw uint16 token IDs")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--rope-theta", type=float, default=10000.0)

    parser.add_argument("--max-iters", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-iters", type=int, default=500)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true", help="Use bfloat16 autocast on CUDA")

    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--wandb-project")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, dataset, args, device) -> float:
    was_training = model.training
    model.eval()
    losses = []
    for _ in range(args.eval_batches):
        inputs, targets = get_batch(dataset, args.batch_size, args.context_length, str(device))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.amp and device.type == "cuda"):
            logits = model(inputs)
            loss = cross_entropy(logits.flatten(0, -2), targets.flatten())
        losses.append(loss.float().item())
    if was_training:
        model.train()
    return sum(losses) / len(losses)


def main() -> None:
    args = parse_args()
    if args.eval_batches < 1:
        raise ValueError("eval_batches must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2)

    train_data = np.memmap(args.train_data, dtype=np.uint16, mode="r")
    valid_data = np.memmap(args.valid_data, dtype=np.uint16, mode="r")
    model = TransformerLM(
        args.vocab_size,
        args.context_length,
        args.d_model,
        args.num_layers,
        args.num_heads,
        args.d_ff,
        args.rope_theta,
        device=device,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )
    start_iteration = load_checkpoint(args.resume, model, optimizer) if args.resume else 0
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Device: {device}; parameters: {parameter_count:,}; resume iteration: {start_iteration}")

    wandb_run = None
    if args.wandb_project:
        import wandb

        wandb_run = wandb.init(project=args.wandb_project, config=vars(args))

    log_path = output_dir / "metrics.jsonl"
    log_file = log_path.open("a", encoding="utf-8")
    started_at = time.perf_counter()
    last_log_at = started_at
    accumulated_loss = 0.0
    logged_steps = 0

    def record(metrics: dict) -> None:
        record = {"wallclock_seconds": time.perf_counter() - started_at, **metrics}
        log_file.write(json.dumps(record) + "\n")
        log_file.flush()
        print(json.dumps(record))
        if wandb_run is not None:
            wandb_run.log(record)

    for iteration in range(start_iteration, args.max_iters):
        learning_rate = get_lr_cosine_schedule(
            iteration, args.max_lr, args.min_lr, args.warmup_iters, args.max_iters
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        inputs, targets = get_batch(train_data, args.batch_size, args.context_length, str(device))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.amp and device.type == "cuda"):
            logits = model(inputs)
            loss = cross_entropy(logits.flatten(0, -2), targets.flatten())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()
        accumulated_loss += loss.float().item()
        logged_steps += 1
        step = iteration + 1

        if step % args.log_interval == 0 or step == args.max_iters:
            now = time.perf_counter()
            tokens = logged_steps * args.batch_size * args.context_length
            metrics = {
                "step": step,
                "train_loss": accumulated_loss / logged_steps,
                "learning_rate": learning_rate,
                "tokens_per_second": tokens / (now - last_log_at),
            }
            if device.type == "cuda":
                metrics["gpu_max_memory_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
            record(metrics)
            accumulated_loss = 0.0
            logged_steps = 0
            last_log_at = now

        if step % args.eval_interval == 0 or step == args.max_iters:
            valid_loss = evaluate(model, valid_data, args, device)
            record({"step": step, "valid_loss": valid_loss, "valid_perplexity": float(np.exp(valid_loss))})

        if step % args.checkpoint_interval == 0 or step == args.max_iters:
            checkpoint_path = output_dir / "checkpoint.pt"
            temporary_path = output_dir / "checkpoint.pt.tmp"
            save_checkpoint(model, optimizer, step, temporary_path)
            os.replace(temporary_path, checkpoint_path)

    log_file.close()
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
