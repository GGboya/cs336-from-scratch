"""单步贪心解码：概率 → token ID → 文本。题目见同目录的 README.md。"""


def greedy_decode_step(
    probabilities: list[float],
    vocab: list[str],
) -> tuple[int, str]:
    """选择概率最高的 token，并返回它的 ID 和文本片段。"""
