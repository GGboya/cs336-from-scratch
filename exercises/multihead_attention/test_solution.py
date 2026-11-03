from exercises.multihead_attention.solution import split_heads
from exercises.test_utils import check_case


def test_hand_calculated_split():
    check_case(
        "手算六维向量拆成三个头",
        split_heads,
        ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3),
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
    )


def test_single_head():
    check_case("单个头保留全部维度", split_heads, ([1.5, -2.0, 3.0], 1), [[1.5, -2.0, 3.0]])


def test_one_dimension_per_head():
    check_case(
        "每个头只有一个维度",
        split_heads,
        ([2.0, 4.0, 6.0, 8.0], 4),
        [[2.0], [4.0], [6.0], [8.0]],
    )


def test_invalid_dimensions_raise_value_error():
    try:
        split_heads([1.0, 2.0, 3.0], 2)
    except ValueError:
        pass
    else:
        raise AssertionError("用例：维度必须整除。无法整除时应该抛出 ValueError")


def test_input_is_not_modified():
    vector = [1.0, 2.0, 3.0, 4.0]
    split_heads(vector, 2)
    assert vector == [1.0, 2.0, 3.0, 4.0], "用例：输入不可修改。vector 被函数修改了"
