"""一步 AdamW 更新。题目、公式和示例见同目录的 README.md。"""


def adamw_step(
    params: list[float],
    grads: list[float],
    m: list[float],
    v: list[float],
    learning_rate: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    step: int,
) -> tuple[list[float], list[float], list[float]]:
    """返回一步 PyTorch 风格 AdamW 更新后的参数、一阶矩和二阶矩。"""
    raise NotImplementedError("Implement adamw_step in exercises/adamw/solution.py")
