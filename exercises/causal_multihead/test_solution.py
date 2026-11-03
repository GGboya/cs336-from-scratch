from math import cos, exp, sin, sqrt

from exercises.causal_multihead.solution import causal_multihead_attention
from exercises.test_utils import check_case


IDENTITY_2 = [
    [1.0, 0.0],
    [0.0, 1.0],
]
ZERO_2 = [
    [0.0, 0.0],
    [0.0, 0.0],
]
IDENTITY_4 = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


def test_single_token_and_output_projection():
    check_case(
        "单 token 只读取自己并完成 V、O 投影",
        causal_multihead_attention,
        (
            [[1.0, 2.0]],
            IDENTITY_2,
            IDENTITY_2,
            [[2.0, 0.0], [0.0, 3.0]],
            [[1.0, 1.0], [0.0, 2.0]],
            1,
            10000.0,
        ),
        [[8.0, 12.0]],
    )


def test_zero_qk_gives_causal_prefix_average():
    check_case(
        "零 QK 时每个位置对可见 V 做因果平均",
        causal_multihead_attention,
        (
            [[1.0, 10.0], [3.0, 30.0], [7.0, 70.0]],
            ZERO_2,
            ZERO_2,
            IDENTITY_2,
            IDENTITY_2,
            1,
            10000.0,
        ),
        [[1.0, 10.0], [2.0, 20.0], [11.0 / 3.0, 110.0 / 3.0]],
    )


def test_two_heads_have_independent_attention():
    qk_weight = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    value_weight = [
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 10.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    past_score = cos(1.0) / sqrt(2.0)
    current_score = 1.0 / sqrt(2.0)
    total = exp(past_score) + exp(current_score)
    past_weight = exp(past_score) / total
    current_weight = exp(current_score) / total
    head_zero_output = past_weight * 10.0 + current_weight * 20.0
    check_case(
        "两个头独立拆分、旋转、注意力计算并拼头",
        causal_multihead_attention,
        (
            [[1.0, 0.0, 10.0, 0.0], [1.0, 0.0, 20.0, 0.0]],
            qk_weight,
            qk_weight,
            value_weight,
            IDENTITY_4,
            2,
            1.0,
        ),
        [[10.0, 0.0, 100.0, 0.0], [head_zero_output, 0.0, 150.0, 0.0]],
    )


def test_rope_rotates_qk_but_not_values():
    past_score = -sin(1.0) / sqrt(2.0)
    current_score = 1.0 / sqrt(2.0)
    total = exp(past_score) + exp(current_score)
    past_weight = exp(past_score) / total
    current_weight = exp(current_score) / total
    check_case(
        "手算 RoPE 只旋转 QK，V 保持原值",
        causal_multihead_attention,
        (
            [[1.0, 0.0], [0.0, 1.0]],
            IDENTITY_2,
            IDENTITY_2,
            IDENTITY_2,
            IDENTITY_2,
            1,
            1.0,
        ),
        [[1.0, 0.0], [past_weight, current_weight]],
    )


def test_large_scores_do_not_overflow():
    check_case(
        "大点积分数使用稳定 Softmax",
        causal_multihead_attention,
        (
            [[1000.0, 0.0], [1000.0, 0.0]],
            IDENTITY_2,
            IDENTITY_2,
            IDENTITY_2,
            IDENTITY_2,
            1,
            10000.0,
        ),
        [[1000.0, 0.0], [1000.0, 0.0]],
    )


def test_inputs_are_not_modified():
    x = [[1.0, 0.0], [0.0, 1.0]]
    wq = [row.copy() for row in IDENTITY_2]
    wk = [row.copy() for row in IDENTITY_2]
    wv = [row.copy() for row in IDENTITY_2]
    wo = [row.copy() for row in IDENTITY_2]
    original_x = [row.copy() for row in x]
    causal_multihead_attention(x, wq, wk, wv, wo, 1, 1.0)
    assert x == original_x, "用例：输入不可修改。x 被函数修改了"
    assert wq == IDENTITY_2, "用例：输入不可修改。wq 被函数修改了"
    assert wk == IDENTITY_2, "用例：输入不可修改。wk 被函数修改了"
    assert wv == IDENTITY_2, "用例：输入不可修改。wv 被函数修改了"
    assert wo == IDENTITY_2, "用例：输入不可修改。wo 被函数修改了"
