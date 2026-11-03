"""完整 Pre-Norm Transformer Block 组合练习。"""


def transformer_block_forward(
    x: list[list[float]],
    gain1: list[float],
    gain2: list[float],
    attention,
    ffn,
    eps: float = 1e-5,
) -> list[list[float]]:
    """按 Pre-Norm 顺序组合 Attention、FFN 和两次残差。"""
    raise NotImplementedError(
        "Implement transformer_block_forward in exercises/transformer_block_forward/solution.py"
    )
