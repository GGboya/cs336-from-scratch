from exercises.gradient_descent.solution import gradient_descent
from exercises.test_utils import check_case


def test_first_step_hand_calculation():
    check_case("手算第一步", gradient_descent, (5.0, 0.1, 1), 4.6)


def test_second_step_uses_updated_weight():
    check_case("手算第二步", gradient_descent, (5.0, 0.1, 2), 4.28)


def test_below_minimum_moves_upward():
    check_case("最优点左侧向右更新", gradient_descent, (1.0, 0.25, 1), 2.0)


def test_zero_steps_keeps_initial_weight():
    check_case("零步更新", gradient_descent, (5.0, 0.1, 0), 5.0)


def test_zero_learning_rate_keeps_weight():
    check_case("零学习率", gradient_descent, (5.0, 0.0, 10), 5.0)


def test_minimum_is_fixed_point():
    check_case("最小值点梯度为零", gradient_descent, (3.0, 0.1, 10), 3.0)


def test_half_learning_rate_reaches_minimum_in_one_step():
    check_case("学习率一半时一步到达", gradient_descent, (9.0, 0.5, 1), 3.0)


def test_unit_learning_rate_oscillates():
    check_case("学习率一时来回振荡", gradient_descent, (5.0, 1.0, 2), 5.0)


def test_closed_form_after_many_steps():
    initial_w = 7.0
    learning_rate = 0.1
    steps = 20
    expected = 3.0 + (initial_w - 3.0) * (1.0 - 2.0 * learning_rate) ** steps
    check_case(
        "多步更新与递推公式一致",
        gradient_descent,
        (initial_w, learning_rate, steps),
        expected,
    )


def test_scalar_inputs_are_not_modified():
    initial_w = 5.0
    learning_rate = 0.1
    steps = 3
    inputs_before = (initial_w, learning_rate, steps)
    gradient_descent(initial_w, learning_rate, steps)
    assert (initial_w, learning_rate, steps) == inputs_before, "用例：输入不可修改。标量输入发生了变化"
