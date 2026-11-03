"""Warmup 与 cosine 学习率。题目和公式见同目录的 README.md。"""


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """返回与真实 training.py 对齐的当前迭代学习率。"""
    raise NotImplementedError(
        "Implement get_lr_cosine_schedule in exercises/cosine_learning_rate/solution.py"
    )
