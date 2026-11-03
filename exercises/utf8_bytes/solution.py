"""UTF-8 字节编码：文本 → 字节 ID。题目和示例见同目录的 README.md。"""


def text_to_byte_ids(text: str) -> list[int]:
    """返回文本的 UTF-8 编码中各字节的整数值。"""
    raise NotImplementedError("Implement text_to_byte_ids in exercises/utf8_bytes/solution.py")


def byte_ids_to_text(byte_ids: list[int]) -> str:
    """把字节序列按 UTF-8 解码，非法序列用 U+FFFD 替换。"""
    raise NotImplementedError("Implement byte_ids_to_text in exercises/utf8_bytes/solution.py")
