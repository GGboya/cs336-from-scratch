from exercises.swiglu.solution import silu, swiglu
from exercises.test_utils import check_case


def test_silu_basic_value():
    check_case("SiLU 基本值", silu, (1.0,), 0.7310585786300049)


def test_basic_elementwise_gating():
    check_case(
        "基本逐元素门控",
        swiglu,
        ([0.0, 1.0, -1.0], [2.0, 2.0, 2.0]),
        [0.0, 1.4621171572600098, -0.5378828427399902],
    )


def test_zero_gate_blocks_corresponding_value():
    check_case("零门控截断对应位置", swiglu, ([0.0, 1.0], [100.0, 0.0]), [0.0, 0.0])


def test_empty_vectors():
    check_case("空向量边界", swiglu, ([], []), [])


def test_large_negative_silu_does_not_overflow():
    check_case("大负数 SiLU 不溢出", silu, (-1000.0,), 0.0)
    check_case("大负数门控不溢出", swiglu, ([-1000.0, -1000.0], [1.0, 2.0]), [0.0, 0.0])


def test_large_positive_silu_does_not_overflow():
    check_case("大正数 SiLU", silu, (1000.0,), 1000.0)
    check_case("大正数门控", swiglu, ([1000.0], [2.0]), [2000.0])


def test_inputs_are_not_modified():
    gate = [0.0, 1.0, -1.0]
    value = [2.0, 3.0, 4.0]
    swiglu(gate, value)
    assert gate == [0.0, 1.0, -1.0], "用例：输入不可修改。gate 被函数修改了"
    assert value == [2.0, 3.0, 4.0], "用例：输入不可修改。value 被函数修改了"
