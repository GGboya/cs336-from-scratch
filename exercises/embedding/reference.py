"""Embedding lookup reference implementation using plain loops."""


def embedding(token_ids: list[int], weight: list[list[float]]) -> list[list[float]]:
    output = []
    for token_id in token_ids:
        output.append(weight[token_id].copy())
    return output
