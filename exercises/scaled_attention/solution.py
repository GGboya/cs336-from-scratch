"""Scaled Attention：计算 softmax(QK^T/sqrt(d_k))V。题目见 README.md。"""


def scaled_attention(
    queries: list[list[float]],
    keys: list[list[float]],
    values: list[list[float]],
) -> list[list[float]]:
    """为每个 Query 返回所有 Values 的注意力加权和。"""
    raise NotImplementedError("Implement scaled_attention in exercises/scaled_attention/solution.py")
