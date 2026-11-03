from math import exp

from exercises.scaled_attention.solution import scaled_attention
from exercises.test_utils import check_case


VALUES = [
    [2.0, 4.0],
    [6.0, 8.0],
]


def test_hand_calculated_attention():
    first_weight = exp(1.0) / (exp(1.0) + 1.0)
    second_weight = 1.0 / (exp(1.0) + 1.0)
    expected_first = [
        first_weight * 2.0 + second_weight * 6.0,
        first_weight * 4.0 + second_weight * 8.0,
    ]
    check_case(
        "手算两个 Query 的注意力",
        scaled_attention,
        ([[1.0], [0.0]], [[1.0], [0.0]], VALUES),
        [expected_first, [4.0, 6.0]],
    )


def test_scaling_by_key_dimension():
    weight = exp(1.0 / (2.0 ** 0.5)) / (exp(1.0 / (2.0 ** 0.5)) + 1.0)
    check_case(
        "点积除以 sqrt(d_k)",
        scaled_attention,
        ([[1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]], VALUES),
        [[weight * 2.0 + (1.0 - weight) * 6.0, weight * 4.0 + (1.0 - weight) * 8.0]],
    )


def test_large_scores_do_not_overflow():
    check_case(
        "大分数不溢出",
        scaled_attention,
        ([[1000.0], [0.0]], [[1000.0], [0.0]], [[1.0, 2.0], [3.0, 4.0]]),
        [[1.0, 2.0], [2.0, 3.0]],
    )


def test_single_key_and_independent_value_dimension():
    check_case(
        "单个 Key 且 Value 维度不同",
        scaled_attention,
        ([[2.0, -1.0]], [[3.0, 4.0]], [[5.0, -2.0, 1.0]]),
        [[5.0, -2.0, 1.0]],
    )


def test_inputs_are_not_modified():
    queries = [[1.0], [0.0]]
    keys = [[1.0], [0.0]]
    values = [row.copy() for row in VALUES]
    scaled_attention(queries, keys, values)
    assert queries == [[1.0], [0.0]], "用例：输入不可修改。queries 被函数修改了"
    assert keys == [[1.0], [0.0]], "用例：输入不可修改。keys 被函数修改了"
    assert values == VALUES, "用例：输入不可修改。values 被函数修改了"
