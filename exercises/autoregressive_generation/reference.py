"""Autoregressive greedy generation reference implementation using plain loops."""


def greedy_generate(
    prompt_ids: list[int],
    model,
    max_new_tokens: int,
    eos_id: int,
) -> list[int]:
    ids = prompt_ids.copy()

    for _ in range(max_new_tokens):
        probabilities = model(ids.copy())
        next_id = 0
        for token_id in range(1, len(probabilities)):
            if probabilities[token_id] > probabilities[next_id]:
                next_id = token_id

        if next_id == eos_id:
            break
        ids.append(next_id)

    return ids
