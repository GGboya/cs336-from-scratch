from exercises.greedy_decoding.solution import greedy_decode_step
from exercises.test_utils import check_case


VOCAB = ["there", ",", "."]


def test_select_highest_probability():
    check_case(
        "选择最大概率",
        greedy_decode_step,
        ([0.2, 0.7, 0.1], VOCAB),
        (1, ","),
    )


def test_first_token_can_win():
    check_case("第一个 token 概率最高", greedy_decode_step, ([0.8, 0.1, 0.1], VOCAB), (0, "there"))


def test_last_token_can_win():
    check_case("最后一个 token 概率最高", greedy_decode_step, ([0.1, 0.2, 0.7], VOCAB), (2, "."))


def test_tie_selects_smallest_id():
    check_case("并列最高选择最小 ID", greedy_decode_step, ([0.5, 0.5, 0.0], VOCAB), (0, "there"))


def test_single_token():
    check_case("只有一个候选 token", greedy_decode_step, ([1.0], ["你好"]), (0, "你好"))


def test_text_can_contain_space():
    check_case(
        "保留 token 中的空格",
        greedy_decode_step,
        ([0.1, 0.9], ["hello", " world"]),
        (1, " world"),
    )


def test_inputs_are_not_modified():
    probabilities = [0.2, 0.7, 0.1]
    vocab = VOCAB.copy()
    greedy_decode_step(probabilities, vocab)
    assert probabilities == [0.2, 0.7, 0.1], "用例：输入不可修改。probabilities 被函数修改了"
    assert vocab == VOCAB, "用例：输入不可修改。vocab 被函数修改了"
