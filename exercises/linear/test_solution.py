from exercises.linear.solution import linear
from exercises.test_utils import check_case


def test_basic_matrix_multiplication():
    input_matrix = [[1.0, 2.0], [3.0, 4.0]]
    weight = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    check_case(
        "基本矩阵乘法",
        linear,
        (input_matrix, weight),
        [[1.0, 2.0, 3.0], [3.0, 4.0, 7.0]],
    )


def test_rectangular_shapes_and_negative_values():
    check_case(
        "矩形形状与负数",
        linear,
        (
            [[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]],
            [[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]],
        ),
        [[7.0, 5.0], [1.0, 1.0]],
    )


def test_single_input_and_output_feature():
    check_case("单行且只有一个输出特征", linear, ([[2.0, 3.0]], [[4.0, 5.0]]), [[23.0]])


def test_empty_input_matrix():
    check_case("没有输入行", linear, ([], [[1.0, 2.0]]), [])


def test_no_output_features():
    check_case("没有输出特征", linear, ([[1.0, 2.0], [3.0, 4.0]], []), [[], []])


def test_inputs_are_not_modified():
    input_matrix = [[1.0, 2.0], [3.0, 4.0]]
    weight = [[1.0, 0.0], [0.0, 1.0]]
    original_input = [row.copy() for row in input_matrix]
    original_weight = [row.copy() for row in weight]
    linear(input_matrix, weight)
    assert input_matrix == original_input, "用例：输入不可修改。input_matrix 被函数修改了"
    assert weight == original_weight, "用例：输入不可修改。weight 被函数修改了"
