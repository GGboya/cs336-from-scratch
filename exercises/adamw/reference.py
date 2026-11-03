"""One A1 AdamW step implemented with plain loops."""

from math import sqrt


def adamw_step(
    params: list[float],
    grads: list[float],
    m: list[float],
    v: list[float],
    learning_rate: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    step: int,
) -> tuple[list[float], list[float], list[float]]:
    new_m = []
    new_v = []
    new_params = []

    step_size = learning_rate * sqrt(1.0 - beta2**step) / (1.0 - beta1**step)

    for param, grad, old_m, old_v in zip(params, grads, m, v):
        current_m = beta1 * old_m + (1.0 - beta1) * grad
        current_v = beta2 * old_v + (1.0 - beta2) * grad * grad
        new_m.append(current_m)
        new_v.append(current_v)

        denominator = sqrt(current_v) + eps
        updated_param = param - step_size * current_m / denominator
        updated_param *= 1.0 - learning_rate * weight_decay
        new_params.append(updated_param)

    return new_params, new_m, new_v
