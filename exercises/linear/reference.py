"""Bias-free Linear reference implementation using plain loops."""


def linear(input_matrix: list[list[float]], weight: list[list[float]]) -> list[list[float]]:
    output = []
    for input_row in input_matrix:
        output_row = []
        for weight_row in weight:
            value = 0.0
            for index in range(len(input_row)):
                value += input_row[index] * weight_row[index]
            output_row.append(value)
        output.append(output_row)
    return output
