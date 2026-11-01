from __future__ import annotations

import math
import os
from collections.abc import Iterable
from typing import BinaryIO, IO

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, nn


def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    if inputs.dtype in (torch.float16, torch.bfloat16):
        inputs = inputs.float()
    max_logits = inputs.max(dim=-1, keepdim=True).values
    shifted = inputs - max_logits
    log_normalizers = shifted.exp().sum(dim=-1).log()
    target_logits = shifted.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return (log_normalizers - target_logits).mean()


def gradient_clipping(parameters: Iterable[nn.Parameter], max_l2_norm: float) -> None:
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return
    total_norm = torch.sqrt(torch.stack([gradient.square().sum() for gradient in gradients]).sum())
    clip_coefficient = max_l2_norm / (total_norm + 1e-6)
    if clip_coefficient < 1:
        for gradient in gradients:
            gradient.mul_(clip_coefficient)


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        if lr < 0 or eps < 0 or weight_decay < 0:
            raise ValueError("Learning rate, epsilon, and weight decay must be non-negative")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError("Adam beta parameters must be in [0, 1)")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)

                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(gradient, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)

                step_size = group["lr"] * math.sqrt(1 - beta2**step) / (1 - beta1**step)
                denominator = exp_avg_sq.sqrt().add_(group["eps"])
                parameter.addcdiv_(exp_avg, denominator, value=-step_size)
                parameter.add_(parameter, alpha=-group["lr"] * group["weight_decay"])
        return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    if it > cosine_cycle_iters or cosine_cycle_iters <= warmup_iters:
        return min_learning_rate
    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    coefficient = 0.5 * (1 + math.cos(math.pi * progress))
    return min_learning_rate + coefficient * (max_learning_rate - min_learning_rate)


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[Tensor, Tensor]:
    if len(dataset) <= context_length:
        raise ValueError("Dataset must contain more tokens than context_length")
    starting_indices = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    inputs = np.stack([dataset[start : start + context_length] for start in starting_indices])
    targets = np.stack([dataset[start + 1 : start + context_length + 1] for start in starting_indices])
    return (
        torch.as_tensor(inputs, dtype=torch.long, device=device),
        torch.as_tensor(targets, dtype=torch.long, device=device),
    )


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration},
        out,
    )


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(src, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]
