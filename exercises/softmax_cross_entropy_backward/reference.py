"""Numerically stable Softmax cross-entropy backward using plain loops."""

from math import exp


def softmax_cross_entropy_backward(logits: list[float], target_id: int) -> list[float]:
    max_logit = logits[0]
    for logit in logits:
        if logit > max_logit:
            max_logit = logit

    exponentials = []
    for logit in logits:
        exponentials.append(exp(logit - max_logit))

    exp_sum = 0.0
    for value in exponentials:
        exp_sum += value

    gradients = []
    for index, value in enumerate(exponentials):
        probability = value / exp_sum
        if index == target_id:
            gradients.append(probability - 1.0)
        else:
            gradients.append(probability)
    return gradients
