from exercises.test_utils import check_case
from exercises.top_p_sampling.solution import top_p_filter


def test_minimum_prefix_and_renormalization():
    check_case(
        "选择最小累计前缀并重新归一化",
        top_p_filter,
        ([0.1, 0.5, 0.3, 0.1], 0.7),
        [(1, 0.625), (2, 0.375)],
    )


def test_probability_tie_uses_smaller_id():
    check_case(
        "平频时按 ID 升序选择",
        top_p_filter,
        ([0.25, 0.25, 0.25, 0.25], 0.5),
        [(0, 0.5), (1, 0.5)],
    )


def test_p_one_keeps_all_tokens():
    check_case(
        "p 等于 1 时保留全部 token",
        top_p_filter,
        ([0.2, 0.5, 0.3], 1.0),
        [(1, 0.5), (2, 0.3), (0, 0.2)],
    )


def test_single_token():
    check_case("只有一个 token", top_p_filter, ([1.0], 0.8), [(0, 1.0)])


def test_threshold_crossing_token_is_fully_kept():
    check_case(
        "完整保留跨过阈值的 token",
        top_p_filter,
        ([0.4, 0.4, 0.2], 0.5),
        [(0, 0.5), (1, 0.5)],
    )


def test_input_is_not_modified():
    probabilities = [0.1, 0.5, 0.3, 0.1]
    original = probabilities.copy()
    top_p_filter(probabilities, 0.7)
    assert probabilities == original, "用例：输入不可修改。probabilities 被函数修改了"
