"""Embedding gradient accumulation and SGD reference using plain loops."""


def embedding_backward(
    token_ids: list[int],
    weight: list[list[float]],
    grad_output: list[list[float]],
    learning_rate: float,
) -> list[list[float]]:
    gradients = []
    for row in weight:
        gradients.append([0.0] * len(row))

    for position, token_id in enumerate(token_ids):
        for dimension in range(len(gradients[token_id])):
            gradients[token_id][dimension] += grad_output[position][dimension]

    updated_weight = []
    for token_id, row in enumerate(weight):
        updated_row = []
        for dimension, value in enumerate(row):
            updated_value = value - learning_rate * gradients[token_id][dimension]
            updated_row.append(updated_value)
        updated_weight.append(updated_row)

    return updated_weight
