"""Multihead Attention：把单个向量拆成连续的头。题目见 README.md。"""


def split_heads(vector: list[float], num_heads: int) -> list[list[float]]:
    """按原有顺序把 vector 拆成 num_heads 个等长向量。"""
    raise NotImplementedError("Implement split_heads in exercises/multihead_attention/solution.py")
