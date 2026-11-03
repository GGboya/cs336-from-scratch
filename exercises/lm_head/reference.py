"""LM Head reference implementation using plain loops."""


def lm_head(hidden: list[float], weight: list[list[float]]) -> list[float]:
    logits = []
    for row in weight:
        logit = 0.0
        for index in range(len(hidden)):
            logit += row[index] * hidden[index]
        logits.append(logit)
    return logits
