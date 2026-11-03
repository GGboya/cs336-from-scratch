"""Numerically stable Softmax reference implementation using plain loops."""

from math import exp


def softmax(logits: list[float]) -> list[float]:
    max_logit = logits[0]
    for logit in logits:
        if logit > max_logit:
            max_logit = logit

    exponentials = []
    for logit in logits:
        exponentials.append(exp(logit - max_logit))

    total = 0.0
    for value in exponentials:
        total += value

    probabilities = []
    for value in exponentials:
        probabilities.append(value / total)
    return probabilities
