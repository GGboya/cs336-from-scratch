"""Simplified BPE reference implementation using plain loops."""


def train_bpe(words: list[str], vocab_size: int) -> tuple[list[str], list[tuple[str, str]]]:
    # 1. 收集所有字符，构造初始词表。
    vocab = []
    for word in words:
        for char in word:
            if char not in vocab:
                vocab.append(char)
    vocab.sort()

    # 2. 把每个单词拆成 token 列表，例如 "abc" -> ["a", "b", "c"]。
    tokenized_words = []
    for word in words:
        tokens = []
        for char in word:
            tokens.append(char)
        tokenized_words.append(tokens)

    merges = []

    while len(vocab) < vocab_size:
        # 3. 统计所有单词中，每个相邻 pair 出现了多少次。
        pair_counts = {}
        for tokens in tokenized_words:
            for index in range(len(tokens) - 1):
                left = tokens[index]
                right = tokens[index + 1]
                pair = (left, right)
                if pair in pair_counts:
                    pair_counts[pair] = pair_counts[pair] + 1
                else:
                    pair_counts[pair] = 1

        if len(pair_counts) == 0:
            break

        # 4. 找出频次最高的 pair；平频时选择字典序更大的 pair。
        best_pair = None
        best_count = -1
        for pair, count in pair_counts.items():
            if count > best_count:
                best_pair = pair
                best_count = count
            elif count == best_count and pair > best_pair:
                best_pair = pair

        left = best_pair[0]
        right = best_pair[1]
        new_token = left + right
        merges.append(best_pair)
        vocab.append(new_token)

        # 5. 在每个单词中，从左到右、不重叠地合并选中的 pair。
        for word_index in range(len(tokenized_words)):
            tokens = tokenized_words[word_index]
            merged_tokens = []
            index = 0

            while index < len(tokens):
                has_next_token = index + 1 < len(tokens)
                if has_next_token and tokens[index] == left and tokens[index + 1] == right:
                    merged_tokens.append(new_token)
                    index = index + 2
                else:
                    merged_tokens.append(tokens[index])
                    index = index + 1

            tokenized_words[word_index] = merged_tokens

    return vocab, merges
