"""L2 gradient clipping implemented with plain loops."""

from math import sqrt


def gradient_clipping(grads: list[float], max_norm: float) -> list[float]:
    squared_sum = 0.0
    for grad in grads:
        squared_sum += grad * grad
    norm = sqrt(squared_sum)

    scale = max_norm / (norm + 1e-6)
    if scale >= 1.0:
        return grads.copy()

    clipped = []
    for grad in grads:
        clipped.append(grad * scale)
    return clipped
