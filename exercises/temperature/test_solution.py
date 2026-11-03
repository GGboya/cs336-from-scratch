from math import exp

from exercises.temperature.solution import temperature_softmax
from exercises.test_utils import check_case


def expected(logits, temperature):
    scaled = [logit / temperature for logit in logits]
    exponentials = [exp(value) for value in scaled]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def test_temperature_one_matches_softmax():
    check_case("温度为 1", temperature_softmax, ([0.0, 1.0], 1.0), expected([0.0, 1.0], 1.0))


def test_higher_temperature_flattens_distribution():
    check_case("温度大于 1 更平滑", temperature_softmax, ([0.0, 1.0], 2.0), expected([0.0, 1.0], 2.0))


def test_lower_temperature_sharpens_distribution():
    check_case("温度小于 1 更尖锐", temperature_softmax, ([0.0, 1.0], 0.5), expected([0.0, 1.0], 0.5))


def test_equal_logits_remain_equal():
    check_case("相同 logits", temperature_softmax, ([2.0, 2.0, 2.0], 3.0), [1.0 / 3.0] * 3)


def test_large_logits_do_not_overflow():
    check_case(
        "大 logits 不溢出",
        temperature_softmax,
        ([1000.0, 1001.0], 0.5),
        [0.11920292202211755, 0.8807970779778823],
    )


def test_invalid_temperature():
    try:
        temperature_softmax([1.0], 0.0)
    except ValueError:
        return
    raise AssertionError("用例：温度为 0 时应抛出 ValueError")


def test_input_is_not_modified():
    logits = [1.0, 2.0]
    temperature_softmax(logits, 2.0)
    assert logits == [1.0, 2.0], "用例：输入不可修改。logits 被函数修改了"
