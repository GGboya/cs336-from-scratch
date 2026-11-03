"""完整训练闭环：极简 Bigram 语言模型。题目见同目录的 README.md。"""


def train_bigram(
    input_ids: list[int],
    target_ids: list[int],
    vocab_size: int,
    learning_rate: float,
    steps: int,
) -> tuple[list[list[float]], list[float]]:
    """训练 Bigram 参数表，并返回每一步更新前的平均 loss。"""
    raise NotImplementedError(
        "Implement train_bigram in exercises/tiny_language_model/solution.py"
    )
