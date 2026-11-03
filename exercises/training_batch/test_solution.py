from exercises.test_utils import check_case
from exercises.training_batch.solution import get_batch


def test_hand_constructed_batch():
    check_case(
        "手算两个起点的输入与目标",
        get_batch,
        ([10, 11, 12, 13, 14, 15], [0, 2], 3),
        (
            [[10, 11, 12], [12, 13, 14]],
            [[11, 12, 13], [13, 14, 15]],
        ),
    )


def test_targets_are_inputs_shifted_by_one():
    check_case(
        "目标恰好右移一位",
        get_batch,
        ([0, 1, 2, 3, 4], [1], 3),
        ([[1, 2, 3]], [[2, 3, 4]]),
    )


def test_last_valid_start():
    check_case(
        "最后一个合法起点",
        get_batch,
        ([0, 1, 2, 3, 4], [2], 2),
        ([[2, 3]], [[3, 4]]),
    )


def test_context_length_one():
    check_case(
        "上下文长度为一",
        get_batch,
        ([5, 6, 7], [1], 1),
        ([[6]], [[7]]),
    )


def test_repeated_start_is_deterministic():
    check_case(
        "重复起点得到重复样本",
        get_batch,
        ([0, 1, 2, 3, 4], [2, 0, 2], 2),
        (
            [[2, 3], [0, 1], [2, 3]],
            [[3, 4], [1, 2], [3, 4]],
        ),
    )


def test_empty_start_indices():
    check_case("空起点列表", get_batch, ([0, 1, 2], [], 2), ([], []))


def test_inputs_are_not_modified():
    tokens = [10, 11, 12, 13, 14]
    start_indices = [0, 1]
    get_batch(tokens, start_indices, 3)
    assert tokens == [10, 11, 12, 13, 14], "用例：输入不可修改。tokens 被函数修改了"
    assert start_indices == [0, 1], "用例：输入不可修改。start_indices 被函数修改了"


def test_output_rows_do_not_alias_tokens():
    tokens = [10, 11, 12, 13]
    inputs, targets = get_batch(tokens, [0], 3)
    inputs[0][0] = 99
    targets[0][0] = 99
    assert tokens == [10, 11, 12, 13], "用例：输出独立副本。修改输出后 tokens 也发生了变化"
