"""Causal mask reference implementation using plain loops."""


def apply_causal_mask(scores: list[list[float]]) -> list[list[float]]:
    masked_scores = []
    for row_index in range(len(scores)):
        masked_row = []
        for column_index in range(len(scores[row_index])):
            if column_index <= row_index:
                masked_row.append(scores[row_index][column_index])
            else:
                masked_row.append(float("-inf"))
        masked_scores.append(masked_row)
    return masked_scores
