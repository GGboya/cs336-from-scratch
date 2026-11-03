"""Causal Multihead Attention：投影、RoPE、逐头因果注意力和输出投影。"""


def causal_multihead_attention(
    x: list[list[float]],
    wq: list[list[float]],
    wk: list[list[float]],
    wv: list[list[float]],
    wo: list[list[float]],
    num_heads: int,
    theta: float,
) -> list[list[float]]:
    """计算带相邻配对 RoPE 的完整因果多头自注意力。"""
    raise NotImplementedError(
        "Implement causal_multihead_attention in exercises/causal_multihead/solution.py"
    )
