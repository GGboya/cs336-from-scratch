from math import exp

from exercises.softmax.solution import softmax
from exercises.test_utils import check_case


def test_equal_logits():
    check_case("相同分数得到相同概率", softmax, ([0.0, 0.0],), [0.5, 0.5])


def test_higher_logit_gets_higher_probability():
    total = 1.0 + exp(2.0)
    check_case("不同分数", softmax, ([0.0, 2.0],), [1.0 / total, exp(2.0) / total])


def test_three_logits():
    check_case(
        "三个候选 token",
        softmax,
        ([1.0, 2.0, 3.0],),
        [0.09003057317038046, 0.24472847105479764, 0.6652409557748219],
    )


def test_single_token():
    check_case("只有一个候选 token", softmax, ([12.5],), [1.0])


def test_large_logits_do_not_overflow():
    check_case(
        "大 logits 不溢出",
        softmax,
        ([1000.0, 1001.0],),
        [0.2689414213699951, 0.7310585786300049],
    )


def test_constant_shift_does_not_change_probabilities():
    check_case(
        "所有分数同时平移",
        softmax,
        ([-999.0, -1000.0, -1001.0],),
        [0.6652409557748219, 0.24472847105479764, 0.09003057317038046],
    )


def test_probabilities_sum_to_one():
    probabilities = softmax([1.0, 2.0, 3.0, 4.0])
    check_case("概率总和", sum, (probabilities,), 1.0)


def test_input_is_not_modified():
    logits = [1.0, 2.0, 3.0]
    softmax(logits)
    assert logits == [1.0, 2.0, 3.0], "用例：输入不可修改。logits 被函数修改了"
