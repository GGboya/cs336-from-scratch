from exercises.lm_head.solution import lm_head
from exercises.test_utils import check_case


WEIGHT = [
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
]


def test_basic_logits():
    check_case("基本点积", lm_head, ([1.0, 2.0], WEIGHT), [1.0, 2.0, 3.0])


def test_negative_values():
    check_case(
        "包含负数",
        lm_head,
        ([2.0, -1.0], [[1.0, 3.0], [-2.0, 4.0]]),
        [-1.0, -8.0],
    )


def test_zero_hidden():
    check_case("零上下文向量", lm_head, ([0.0, 0.0], WEIGHT), [0.0, 0.0, 0.0])


def test_single_token_vocabulary():
    check_case("词表只有一个 token", lm_head, ([2.0, 3.0], [[4.0, 5.0]]), [23.0])


def test_three_dimensional_hidden():
    check_case(
        "三维向量",
        lm_head,
        ([1.0, 2.0, 3.0], [[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]]),
        [7.0, 5.0],
    )


def test_inputs_are_not_modified():
    hidden = [1.0, 2.0]
    weight = [row.copy() for row in WEIGHT]
    lm_head(hidden, weight)
    assert hidden == [1.0, 2.0], "用例：输入不可修改。hidden 被函数修改了"
    assert weight == WEIGHT, "用例：输入不可修改。weight 被函数修改了"
