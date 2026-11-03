"""Causal Mask：屏蔽每个位置之后的未来分数。题目见 README.md。"""


def apply_causal_mask(scores: list[list[float]]) -> list[list[float]]:
    """保留 j <= i 的分数，并把 j > i 的分数替换为负无穷。"""
    raise NotImplementedError("Implement apply_causal_mask in exercises/causal_mask/solution.py")
