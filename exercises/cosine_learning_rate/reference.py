"""Warmup plus cosine learning-rate schedule using the standard library."""

from math import cos, pi


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
    coefficient = 0.5 * (1.0 + cos(pi * progress))
    return min_learning_rate + coefficient * (max_learning_rate - min_learning_rate)
