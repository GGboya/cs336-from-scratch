"""Special token 切分：保留控制标记及其间的普通文本。题目见同目录 README.md。"""


def split_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    """切分文本并保留 special token；同一位置最长匹配优先。"""
    raise NotImplementedError("Implement split_special_tokens in exercises/special_tokens/solution.py")
