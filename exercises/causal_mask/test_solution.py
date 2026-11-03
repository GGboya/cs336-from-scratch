from exercises.causal_mask.solution import apply_causal_mask
from exercises.test_utils import check_case


NEGATIVE_INFINITY = float("-inf")


def test_hand_calculated_mask():
    check_case(
        "手算三行因果掩码",
        apply_causal_mask,
        ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],),
        [
            [1.0, NEGATIVE_INFINITY, NEGATIVE_INFINITY],
            [4.0, 5.0, NEGATIVE_INFINITY],
            [7.0, 8.0, 9.0],
        ],
    )


def test_visible_negative_scores_are_preserved():
    check_case(
        "保留过去和当前位置的负分数",
        apply_causal_mask,
        ([[-3.0, 8.0], [-2.0, -1.0]],),
        [[-3.0, NEGATIVE_INFINITY], [-2.0, -1.0]],
    )


def test_single_position():
    check_case("单个位置可以看见自己", apply_causal_mask, ([[4.5]],), [[4.5]])


def test_input_is_not_modified():
    scores = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    original = [row.copy() for row in scores]
    apply_causal_mask(scores)
    assert scores == original, "用例：输入不可修改。scores 被函数修改了"
