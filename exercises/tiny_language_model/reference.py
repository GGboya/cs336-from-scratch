"""Tiny Bigram language-model training reference using plain loops."""

from math import exp, log


def probabilities_for(logits: list[float]) -> list[float]:
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


def train_bigram(
    input_ids: list[int],
    target_ids: list[int],
    vocab_size: int,
    learning_rate: float,
    steps: int,
) -> tuple[list[list[float]], list[float]]:
    weight = []
    for _ in range(vocab_size):
        weight.append([0.0] * vocab_size)

    losses = []
    sample_count = len(input_ids)

    for _ in range(steps):
        gradients = []
        for _ in range(vocab_size):
            gradients.append([0.0] * vocab_size)

        total_loss = 0.0
        for input_id, target_id in zip(input_ids, target_ids):
            logits = weight[input_id]
            probabilities = probabilities_for(logits)
            total_loss += -log(probabilities[target_id])

            for token_id in range(vocab_size):
                gradient = probabilities[token_id]
                if token_id == target_id:
                    gradient -= 1.0
                gradients[input_id][token_id] += gradient / sample_count

        losses.append(total_loss / sample_count)

        for input_id in range(vocab_size):
            for token_id in range(vocab_size):
                weight[input_id][token_id] -= (
                    learning_rate * gradients[input_id][token_id]
                )

    return weight, losses
