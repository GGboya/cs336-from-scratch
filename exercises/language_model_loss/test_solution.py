from math import exp, log

from exercises.language_model_loss.solution import language_model_loss
from exercises.test_utils import check_case


def test_two_equal_logits():
    check_case(
        "两个 token 概率相同",
        language_model_loss,
        ([0.0, 0.0], 0),
        log(2.0),
    )


def test_confident_correct_prediction():
    check_case(
        "正确 token 的概率较高",
        language_model_loss,
        ([0.0, 2.0], 1),
        log(1.0 + exp(-2.0)),
    )


def test_same_distribution_with_different_target():
    check_case(
        "相同概率分布但正确答案不同",
        language_model_loss,
        ([0.0, 2.0], 0),
        2.0 + log(1.0 + exp(-2.0)),
    )


def test_three_equal_logits():
    check_case(
        "三个 token 概率相同",
        language_model_loss,
        ([0.0, 0.0, 0.0], 1),
        log(3.0),
    )


def test_three_different_logits():
    check_case(
        "三个不同分数",
        language_model_loss,
        ([1.0, 2.0, 3.0], 2),
        0.4076059644443804,
    )


def test_single_token_has_zero_loss():
    check_case(
        "词表只有一个 token",
        language_model_loss,
        ([12.5], 0),
        0.0,
    )


def test_large_logits_do_not_overflow():
    check_case(
        "大 logits 不溢出",
        language_model_loss,
        ([1000.0, 1001.0], 1),
        log(1.0 + exp(-1.0)),
    )


def test_adding_a_constant_does_not_change_loss():
    check_case(
        "所有 logits 同时平移不改变结果",
        language_model_loss,
        ([-999.0, -1000.0, -1001.0], 0),
        0.4076059644443804,
    )


def test_very_low_target_probability_does_not_underflow():
    check_case(
        "正确 token 概率极低时仍能计算",
        language_model_loss,
        ([0.0, -1000.0], 1),
        1000.0,
    )


def test_input_is_not_modified():
    logits = [1.0, 2.0, 3.0]
    language_model_loss(logits, 1)
    assert logits == [1.0, 2.0, 3.0], "用例：输入不可修改。logits 被函数修改了"
