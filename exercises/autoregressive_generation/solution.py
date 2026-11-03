"""自回归贪心生成：反复预测、选择并接回 token。题目见同目录 README.md。"""


def greedy_generate(
    prompt_ids: list[int],
    model,
    max_new_tokens: int,
    eos_id: int,
) -> list[int]:
    """返回包含 prompt 的完整 IDs，在达到长度限制或生成 EOS 后停止。"""
    raise NotImplementedError("Implement greedy_generate in exercises/autoregressive_generation/solution.py")
