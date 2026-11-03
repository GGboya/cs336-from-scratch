from exercises.rmsnorm.solution import rmsnorm
from exercises.test_utils import check_case


def test_basic_example_with_float_tolerance():
    check_case(
        "基本数值示例（1e-9 容差）",
        rmsnorm,
        ([1.0, 2.0], [1.0, 0.5], 0.0),
        [0.632455532, 0.632455532],
    )


def test_gain_is_applied_per_dimension():
    check_case("逐维 gain", rmsnorm, ([2.0, -2.0], [0.5, 3.0], 0.0), [0.5, -3.0])


def test_eps_is_added_inside_square_root():
    check_case(
        "eps 加在平方根内部",
        rmsnorm,
        ([1.0, 0.0], [1.0, 1.0], 0.25),
        [1.1547005383792515, 0.0],
    )


def test_zero_vector_with_positive_eps():
    check_case("零向量不会除以零", rmsnorm, ([0.0, 0.0], [2.0, 3.0], 1e-5), [0.0, 0.0])


def test_single_element_vector():
    check_case("单元素边界", rmsnorm, ([-4.0], [2.0], 0.0), [-2.0])


def test_default_eps():
    check_case("默认 eps", rmsnorm, ([1.0], [1.0]), [0.9999950000374998])


def test_inputs_are_not_modified():
    vector = [1.0, 2.0, 3.0]
    gain = [1.0, 0.5, 2.0]
    rmsnorm(vector, gain, 1e-5)
    assert vector == [1.0, 2.0, 3.0], "用例：输入不可修改。vector 被函数修改了"
    assert gain == [1.0, 0.5, 2.0], "用例：输入不可修改。gain 被函数修改了"
