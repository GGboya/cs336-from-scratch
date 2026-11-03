"""Top-p 候选集过滤与概率归一化。题目和示例见同目录的 README.md。"""


def top_p_filter(probabilities: list[float], p: float) -> list[tuple[int, float]]:
    """选择达到累计阈值的最小前缀，并返回 ID 与归一化概率。"""
    raise NotImplementedError("Implement top_p_filter in exercises/top_p_sampling/solution.py")
