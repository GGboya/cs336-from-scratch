"""确定性构造语言模型训练 batch。题目和示例见同目录的 README.md。"""


def get_batch(
    tokens: list[int],
    start_indices: list[int],
    context_length: int,
) -> tuple[list[list[int]], list[list[int]]]:
    """按给定起点返回输入窗口及右移一位的目标窗口。"""
    raise NotImplementedError("Implement get_batch in exercises/training_batch/solution.py")
