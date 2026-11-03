"""Single-step greedy decoding reference implementation using plain loops."""


def greedy_decode_step(
    probabilities: list[float],
    vocab: list[str],
) -> tuple[int, str]:
    best_id = 0
    for token_id in range(1, len(probabilities)):
        if probabilities[token_id] > probabilities[best_id]:
            best_id = token_id
    return best_id, vocab[best_id]
