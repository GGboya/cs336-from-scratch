"""Special-token splitting reference implementation using plain loops."""


def split_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    if text == "":
        return []

    ordered_tokens = sorted(special_tokens, key=len, reverse=True)
    parts = []
    ordinary_chars = []
    index = 0

    while index < len(text):
        matched_token = None
        for token in ordered_tokens:
            if text.startswith(token, index):
                matched_token = token
                break

        if matched_token is None:
            ordinary_chars.append(text[index])
            index += 1
            continue

        if ordinary_chars:
            parts.append("".join(ordinary_chars))
            ordinary_chars = []
        parts.append(matched_token)
        index += len(matched_token)

    if ordinary_chars:
        parts.append("".join(ordinary_chars))
    return parts
