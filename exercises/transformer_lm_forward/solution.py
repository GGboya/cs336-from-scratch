"""完整 TransformerLM 前向组合练习。"""


def transformer_lm_forward(
    token_ids: list[int],
    embedding_weight: list[list[float]],
    blocks: list,
    final_gain: list[float],
    lm_head_weight: list[list[float]],
    eps: float = 1e-5,
) -> list[list[float]]:
    """组合 Embedding、Blocks、最终 RMSNorm 和 LM Head。"""
    raise NotImplementedError(
        "Implement transformer_lm_forward in exercises/transformer_lm_forward/solution.py"
    )
