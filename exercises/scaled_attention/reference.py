"""Numerically stable scaled attention reference implementation using plain loops."""

from math import exp, sqrt


def scaled_attention(
    queries: list[list[float]],
    keys: list[list[float]],
    values: list[list[float]],
) -> list[list[float]]:
    outputs = []
    scale = sqrt(len(queries[0]))

    for query in queries:
        scores = []
        for key in keys:
            dot_product = 0.0
            for dimension in range(len(query)):
                dot_product += query[dimension] * key[dimension]
            scores.append(dot_product / scale)

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

        output = []
        for value_dimension in range(len(values[0])):
            weighted_value = 0.0
            for key_index in range(len(keys)):
                weighted_value += weights[key_index] * values[key_index][value_dimension]
            output.append(weighted_value)
        outputs.append(output)

    return outputs
