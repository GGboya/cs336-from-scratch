"""Simplified adjacent-pair RoPE reference implementation using plain loops."""

from math import cos, sin


def rope(vector: list[float], position: int, theta: float) -> list[float]:
    dimensions = len(vector)
    output = vector.copy()

    for pair_index in range(dimensions // 2):
        angle = position * theta ** (-2.0 * pair_index / dimensions)
        cosine = cos(angle)
        sine = sin(angle)
        x = vector[2 * pair_index]
        y = vector[2 * pair_index + 1]
        output[2 * pair_index] = x * cosine - y * sine
        output[2 * pair_index + 1] = x * sine + y * cosine

    return output
