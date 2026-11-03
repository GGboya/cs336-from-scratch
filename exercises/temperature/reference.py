"""Temperature scaling and stable Softmax reference using plain loops."""

from math import exp


def temperature_softmax(logits: list[float], temperature: float) -> list[float]:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    scaled_logits = []
    for logit in logits:
        scaled_logits.append(logit / temperature)

    max_logit = scaled_logits[0]
    for logit in scaled_logits:
        if logit > max_logit:
            max_logit = logit

    exponentials = []
    for logit in scaled_logits:
        exponentials.append(exp(logit - max_logit))

    total = 0.0
    for value in exponentials:
        total += value

    probabilities = []
    for value in exponentials:
        probabilities.append(value / total)
    return probabilities
