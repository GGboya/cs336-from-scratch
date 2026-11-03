"""Softmax 与交叉熵 Loss。题目和示例见同目录的 README.md。"""


def language_model_loss(logits: list[float], target_id: int) -> float:
    """计算模型在一个预测位置上给正确 token 分配的交叉熵 loss。"""
    raise NotImplementedError(
        "Implement language_model_loss in exercises/language_model_loss/solution.py"
    )
