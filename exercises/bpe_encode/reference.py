"""Simplified BPE encoding reference implementation using plain loops."""


def _pretokenize(text: str) -> list[str]:
    if text == "":
        return []
    words = text.split(" ")
    pretokens = [words[0]]
    for word in words[1:]:
        pretokens.append(" " + word)
    return pretokens


def bpe_encode(text: str, merges: list[tuple[str, str]], vocab: list[str]) -> list[int]:
    token_ids = []
    for pretoken in _pretokenize(text):
        tokens = []
        for char in pretoken:
            tokens.append(char)

        for left, right in merges:
            merged_tokens = []
            index = 0
            while index < len(tokens):
                has_next_token = index + 1 < len(tokens)
                if has_next_token and tokens[index] == left and tokens[index + 1] == right:
                    merged_tokens.append(left + right)
                    index += 2
                else:
                    merged_tokens.append(tokens[index])
                    index += 1
            tokens = merged_tokens

        for token in tokens:
            token_ids.append(vocab.index(token))

    return token_ids
