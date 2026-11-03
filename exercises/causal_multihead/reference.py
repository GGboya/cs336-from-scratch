"""Causal multihead attention reference implementation using plain loops."""

from math import cos, exp, sin, sqrt


def _project(inputs: list[list[float]], weight: list[list[float]]) -> list[list[float]]:
    outputs = []
    for input_vector in inputs:
        output_vector = []
        for weight_row in weight:
            value = 0.0
            for input_index in range(len(input_vector)):
                value += weight_row[input_index] * input_vector[input_index]
            output_vector.append(value)
        outputs.append(output_vector)
    return outputs


def _apply_rope(vector: list[float], position: int, theta: float) -> list[float]:
    rotated = vector.copy()
    head_dim = len(vector)
    for even_index in range(0, head_dim, 2):
        odd_index = even_index + 1
        frequency = theta ** (-even_index / head_dim)
        angle = position * frequency
        cosine = cos(angle)
        sine = sin(angle)
        even = vector[even_index]
        odd = vector[odd_index]
        rotated[even_index] = even * cosine - odd * sine
        rotated[odd_index] = even * sine + odd * cosine
    return rotated


def causal_multihead_attention(
    x: list[list[float]],
    wq: list[list[float]],
    wk: list[list[float]],
    wv: list[list[float]],
    wo: list[list[float]],
    num_heads: int,
    theta: float,
) -> list[list[float]]:
    if not x:
        return []

    d_model = len(x[0])
    if num_heads <= 0:
        raise ValueError("num_heads must be positive")
    if d_model % num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")
    head_dim = d_model // num_heads
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for RoPE")

    queries = _project(x, wq)
    keys = _project(x, wk)
    values = _project(x, wv)
    merged_outputs = []

    for query_position in range(len(x)):
        merged = []
        for head_index in range(num_heads):
            start = head_index * head_dim
            stop = start + head_dim
            query = _apply_rope(queries[query_position][start:stop], query_position, theta)

            scores = []
            for key_position in range(query_position + 1):
                key = _apply_rope(keys[key_position][start:stop], key_position, theta)
                dot_product = 0.0
                for dimension in range(head_dim):
                    dot_product += query[dimension] * key[dimension]
                scores.append(dot_product / sqrt(head_dim))

            max_score = scores[0]
            for score in scores:
                if score > max_score:
                    max_score = score

            exponentials = []
            for score in scores:
                exponentials.append(exp(score - max_score))
            total = 0.0
            for value in exponentials:
                total += value
            weights = []
            for value in exponentials:
                weights.append(value / total)

            for dimension in range(head_dim):
                weighted_value = 0.0
                for key_position in range(query_position + 1):
                    value = values[key_position][start + dimension]
                    weighted_value += weights[key_position] * value
                merged.append(weighted_value)

        merged_outputs.append(merged)

    return _project(merged_outputs, wo)
