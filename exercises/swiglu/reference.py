"""Numerically stable SwiGLU core reference implementation using plain loops."""

from math import exp


def silu(x: float) -> float:
    if x >= 0.0:
        return x / (1.0 + exp(-x))

    exponential = exp(x)
    return x * exponential / (1.0 + exponential)


def swiglu(gate: list[float], value: list[float]) -> list[float]:
    output = []
    for index in range(len(gate)):
        output.append(silu(gate[index]) * value[index])
    return output
