from exercises.test_utils import check_case
from exercises.transformer_block.solution import residual_add


def test_hand_calculated_residual_add():
    check_case(
        "手算逐元素残差相加",
        residual_add,
        ([1.0, 2.0, 3.0], [0.5, -1.0, 4.0]),
        [1.5, 1.0, 7.0],
    )


def test_zero_sublayer_output_preserves_input_values():
    check_case("零子层输出保留原值", residual_add, ([2.0, -3.0, 4.5], [0.0, 0.0, 0.0]), [2.0, -3.0, 4.5])


def test_negative_values():
    check_case("两个负向量相加", residual_add, ([-1.5, -2.0], [-2.5, 3.0]), [-4.0, 1.0])


def test_single_dimension():
    check_case("单向量维度", residual_add, ([2.5], [3.5]), [6.0])


def test_mismatched_lengths_raise_value_error():
    try:
        residual_add([1.0, 2.0], [3.0])
    except ValueError:
        pass
    else:
        raise AssertionError("用例：维度一致。长度不同时应该抛出 ValueError")


def test_inputs_are_not_modified():
    input_vector = [1.0, 2.0, 3.0]
    sublayer_output = [0.5, -1.0, 4.0]
    residual_add(input_vector, sublayer_output)
    assert input_vector == [1.0, 2.0, 3.0], "用例：输入不可修改。input 被函数修改了"
    assert sublayer_output == [0.5, -1.0, 4.0], "用例：输入不可修改。sublayer_output 被函数修改了"
