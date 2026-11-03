"""RMSNorm reference implementation using plain loops."""

from math import sqrt


def rmsnorm(vector: list[float], gain: list[float], eps: float = 1e-5) -> list[float]:
    squared_sum = 0.0
    for value in vector:
        squared_sum += value * value

    rms = sqrt(squared_sum / len(vector) + eps)
    output = []
    for index in range(len(vector)):
        output.append(vector[index] / rms * gain[index])
    return output
