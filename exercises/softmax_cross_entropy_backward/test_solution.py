from math import exp

from exercises.softmax_cross_entropy_backward.solution import (
    softmax_cross_entropy_backward,
)
from exercises.test_utils import check_case


def test_equal_logits_hand_calculation():
    check_case(
        "手算两个相同 logits 的梯度",
        softmax_cross_entropy_backward,
        ([0.0, 0.0], 0),
        [-0.5, 0.5],
    )


def test_three_equal_logits():
    check_case(
        "三个相同 logits",
        softmax_cross_entropy_backward,
        ([0.0, 0.0, 0.0], 1),
        [1.0 / 3.0, -2.0 / 3.0, 1.0 / 3.0],
    )


def test_different_logits():
    total = 1.0 + exp(2.0)
    check_case(
        "不同 logits 先求概率再减 one_hot",
        softmax_cross_entropy_backward,
        ([0.0, 2.0], 1),
        [1.0 / total, exp(2.0) / total - 1.0],
    )


def test_single_token_has_zero_gradient():
    check_case(
        "单个 token 的概率必为一",
        softmax_cross_entropy_backward,
        ([12.5], 0),
        [0.0],
    )


def test_large_logits_do_not_overflow():
    check_case(
        "大 logits 不溢出",
        softmax_cross_entropy_backward,
        ([1000.0, 1001.0], 1),
        [0.2689414213699951, -0.2689414213699951],
    )


def test_extremely_low_target_probability():
    check_case(
        "正确 token 概率极低",
        softmax_cross_entropy_backward,
        ([0.0, -1000.0], 1),
        [1.0, -1.0],
    )


def test_constant_shift_does_not_change_gradient():
    check_case(
        "所有 logits 同时平移不改变梯度",
        softmax_cross_entropy_backward,
        ([-999.0, -1000.0, -1001.0], 0),
        [0.6652409557748219 - 1.0, 0.24472847105479764, 0.09003057317038046],
    )


def test_gradients_sum_to_zero():
    gradients = softmax_cross_entropy_backward([1.0, 2.0, 3.0, 4.0], 2)
    check_case("梯度总和为零", sum, (gradients,), 0.0)


def test_input_is_not_modified():
    logits = [1.0, 2.0, 3.0]
    softmax_cross_entropy_backward(logits, 1)
    assert logits == [1.0, 2.0, 3.0], "用例：输入不可修改。logits 被函数修改了"
