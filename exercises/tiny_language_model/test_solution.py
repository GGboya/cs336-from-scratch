from math import exp, log

from exercises.test_utils import check_case
from exercises.tiny_language_model.solution import train_bigram


def test_single_sample_single_step():
    check_case(
        "一个样本的一步训练",
        train_bigram,
        ([0], [1], 2, 0.1, 1),
        ([[-0.05, 0.05], [0.0, 0.0]], [log(2.0)]),
    )


def test_repeated_identical_samples_use_average():
    check_case(
        "相同样本求平均梯度",
        train_bigram,
        ([0, 0], [1, 1], 2, 0.1, 1),
        ([[-0.05, 0.05], [0.0, 0.0]], [log(2.0)]),
    )


def test_zero_steps_returns_initial_weight():
    check_case(
        "零训练步数",
        train_bigram,
        ([0], [1], 3, 0.1, 0),
        ([[0.0, 0.0, 0.0]] * 3, []),
    )


def test_loss_decreases():
    weight, losses = train_bigram([0, 0, 0], [1, 1, 1], 2, 0.5, 20)
    assert losses[-1] < losses[0], f"用例：训练降低 loss。初始 {losses[0]}，最终记录 {losses[-1]}"
    assert weight[0][1] > weight[0][0], "用例：正确 token 的 logit 应高于错误 token"


def test_correct_token_probability_increases():
    weight, losses = train_bigram([0, 0, 0], [1, 1, 1], 2, 0.5, 120)
    difference = weight[0][1] - weight[0][0]
    correct_probability = exp(difference) / (1.0 + exp(difference))
    assert correct_probability > 0.99, f"用例：学会简单规律。正确概率只有 {correct_probability}"


def test_unused_weight_rows_stay_zero():
    weight, _ = train_bigram([0], [1], 3, 0.1, 5)
    assert weight[2] == [0.0, 0.0, 0.0], "用例：未作为输入的行不更新"


def test_inputs_are_not_modified():
    input_ids = [0, 1]
    target_ids = [1, 0]
    train_bigram(input_ids, target_ids, 2, 0.1, 2)
    assert input_ids == [0, 1], "用例：输入不可修改。input_ids 被函数修改了"
    assert target_ids == [1, 0], "用例：输入不可修改。target_ids 被函数修改了"
