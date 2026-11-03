"""按 L2 范数裁剪梯度。题目和示例见同目录的 README.md。"""


def gradient_clipping(grads: list[float], max_norm: float) -> list[float]:
    """仅当梯度 L2 范数超过 max_norm 时进行等比例缩放。"""
    raise NotImplementedError(
        "Implement gradient_clipping in exercises/gradient_clipping/solution.py"
    )
