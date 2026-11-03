"""Residual addition reference implementation using a plain loop."""


def residual_add(input: list[float], sublayer_output: list[float]) -> list[float]:
    if len(input) != len(sublayer_output):
        raise ValueError("input and sublayer_output must have the same length")

    output = []
    for index in range(len(input)):
        output.append(input[index] + sublayer_output[index])
    return output
