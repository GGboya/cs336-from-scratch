from exercises.special_tokens.solution import split_special_tokens
from exercises.test_utils import check_case


def test_split_and_keep_special_token():
    check_case(
        "切分并保留 special token",
        split_special_tokens,
        ("hello<|end|>world", ["<|end|>"]),
        ["hello", "<|end|>", "world"],
    )


def test_longest_overlapping_token_wins():
    check_case(
        "重叠时最长匹配优先",
        split_special_tokens,
        ("x<|a|>longy<|a|>", ["<|a|>", "<|a|>long"]),
        ["x", "<|a|>long", "y", "<|a|>"],
    )


def test_adjacent_special_tokens():
    check_case(
        "相邻 special tokens 之间不产生空串",
        split_special_tokens,
        ("<|end|><|endoftext|>", ["<|end|>", "<|endoftext|>"]),
        ["<|end|>", "<|endoftext|>"],
    )


def test_ordinary_text_is_unchanged():
    check_case("普通文本保持原样", split_special_tokens, ("普通 文本\n", ["<|end|>"]), ["普通 文本\n"])


def test_partial_token_is_ordinary_text():
    check_case("不完整标记仍是普通文本", split_special_tokens, ("ab<|end", ["<|end|>"]), ["ab<|end"])


def test_empty_text():
    check_case("空字符串", split_special_tokens, ("", ["<|end|>"]), [])


def test_no_special_tokens():
    check_case("候选列表为空", split_special_tokens, ("hello", []), ["hello"])


def test_special_tokens_input_is_not_modified():
    special_tokens = ["<|end|>", "<|endoftext|>"]
    original = special_tokens.copy()
    split_special_tokens("<|endoftext|>", special_tokens)
    assert special_tokens == original, "用例：输入不可修改。special_tokens 被函数修改了"
