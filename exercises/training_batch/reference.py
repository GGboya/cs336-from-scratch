"""Deterministic language-model batch construction using plain loops."""


def get_batch(
    tokens: list[int],
    start_indices: list[int],
    context_length: int,
) -> tuple[list[list[int]], list[list[int]]]:
    inputs = []
    targets = []

    for start in start_indices:
        input_row = []
        target_row = []
        for offset in range(context_length):
            input_row.append(tokens[start + offset])
            target_row.append(tokens[start + offset + 1])
        inputs.append(input_row)
        targets.append(target_row)

    return inputs, targets
