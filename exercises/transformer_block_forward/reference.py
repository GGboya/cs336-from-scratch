"""Pre-Norm Transformer Block composition reference using plain loops."""

from math import sqrt


def normalize_rows(x, gain, eps):
    output = []
    for row in x:
        squared_sum = 0.0
        for value in row:
            squared_sum += value * value
        rms = sqrt(squared_sum / len(row) + eps)
        normalized_row = []
        for index, value in enumerate(row):
            normalized_row.append(value / rms * gain[index])
        output.append(normalized_row)
    return output


def add_rows(left, right):
    output = []
    for left_row, right_row in zip(left, right):
        row = []
        for index in range(len(left_row)):
            row.append(left_row[index] + right_row[index])
        output.append(row)
    return output


def transformer_block_forward(x, gain1, gain2, attention, ffn, eps=1e-5):
    normalized = normalize_rows(x, gain1, eps)
    y = add_rows(x, attention(normalized))

    normalized = normalize_rows(y, gain2, eps)
    return add_rows(y, ffn(normalized))
