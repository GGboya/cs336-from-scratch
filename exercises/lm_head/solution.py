"""LM Head：向量 → logits。题目和示例见同目录的 README.md。"""


def lm_head(hidden: list[float], weight: list[list[float]]) -> list[float]:
    """为词表中的每个候选 token 计算一个 logit。"""
    raise NotImplementedError("Implement lm_head in exercises/lm_head/solution.py")
