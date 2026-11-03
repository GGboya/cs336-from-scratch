from math import cos, sin

from exercises.rope.solution import rope
from exercises.test_utils import check_case


def test_basic_two_dimensional_rotation():
    check_case(
        "二维向量基本旋转",
        rope,
        ([1.0, 0.0], 1, 10000.0),
        [cos(1.0), sin(1.0)],
    )


def test_general_two_dimensional_vector():
    check_case(
        "一般二维向量",
        rope,
        ([1.0, 2.0], 1, 10000.0),
        [cos(1.0) - 2.0 * sin(1.0), sin(1.0) + 2.0 * cos(1.0)],
    )


def test_two_pairs_use_different_frequencies():
    check_case(
        "四维向量使用两个频率",
        rope,
        ([1.0, 0.0, 1.0, 0.0], 2, 100.0),
        [cos(2.0), sin(2.0), cos(0.2), sin(0.2)],
    )


def test_position_zero_keeps_vector_unchanged():
    check_case(
        "位置零是边界情况",
        rope,
        ([1.0, -2.0, 3.0, 4.0], 0, 10000.0),
        [1.0, -2.0, 3.0, 4.0],
    )


def test_empty_vector():
    check_case("空向量边界", rope, ([], 3, 10000.0), [])


def test_input_is_not_modified():
    vector = [1.0, 2.0, 3.0, 4.0]
    rope(vector, 2, 10000.0)
    assert vector == [1.0, 2.0, 3.0, 4.0], "用例：输入不可修改。vector 被函数修改了"
