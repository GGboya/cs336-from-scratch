"""Top-p filtering reference implementation using plain loops and the standard library."""


def top_p_filter(probabilities: list[float], p: float) -> list[tuple[int, float]]:
    ranked = []
    for token_id, probability in enumerate(probabilities):
        ranked.append((token_id, probability))
    ranked.sort(key=lambda item: (-item[1], item[0]))

    selected = []
    total = 0.0
    for token_id, probability in ranked:
        selected.append((token_id, probability))
        total += probability
        if total >= p:
            break

    normalized = []
    for token_id, probability in selected:
        normalized.append((token_id, probability / total))
    return normalized
