"""Transformer 参数量计算。题目见同目录的 README.md。"""


def count_transformer_parameters(
    vocab_size: int,
    d_model: int,
    num_layers: int,
    d_ff: int,
) -> int:
    """计算当前无 bias、Embedding 与 LM Head 独立的 A1 模型参数量。"""
    raise NotImplementedError(
        "Implement count_transformer_parameters in exercises/resource_accounting/solution.py"
    )
