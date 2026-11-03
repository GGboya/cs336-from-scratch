from exercises.adamw.solution import adamw_step
from exercises.test_utils import check_case


def test_first_step_hand_calculation():
    check_case(
        "手算 AdamW 第一步",
        adamw_step,
        (
            [1.0, -2.0],
            [0.5, -1.0],
            [0.0, 0.0],
            [0.0, 0.0],
            0.1,
            0.9,
            0.999,
            0.0,
            0.01,
            1,
        ),
        ([0.8991, -1.8981], [0.05, -0.1], [0.00025, 0.001]),
    )


def test_second_step_hand_calculation():
    check_case(
        "手算连续更新的第二步",
        adamw_step,
        ([0.9], [1.0], [0.5], [0.5], 0.1, 0.5, 0.5, 0.0, 0.0, 2),
        ([0.8], [0.75], [0.75]),
    )


def test_first_step_of_consecutive_example():
    check_case(
        "连续更新示例的第一步",
        adamw_step,
        ([1.0], [1.0], [0.0], [0.0], 0.1, 0.5, 0.5, 0.0, 0.0, 1),
        ([0.9], [0.5], [0.5]),
    )


def test_weight_decay_zero():
    check_case(
        "weight_decay 为零时只自适应更新",
        adamw_step,
        ([1.0], [0.5], [0.0], [0.0], 0.1, 0.9, 0.999, 0.0, 0.0, 1),
        ([0.9], [0.05], [0.00025]),
    )


def test_nonzero_eps_and_decay_match_a1_order():
    check_case(
        "非零 eps 时先自适应更新再衰减",
        adamw_step,
        ([1.0], [1.0], [0.0], [0.0], 0.1, 0.9, 0.999, 0.01, 0.01, 1),
        ([0.923101282027869], [0.1], [0.001]),
    )


def test_zero_gradient_still_decays_weight():
    check_case(
        "零梯度仍执行解耦权重衰减",
        adamw_step,
        ([2.0], [0.0], [0.0], [0.0], 0.1, 0.9, 0.999, 1e-8, 0.01, 1),
        ([1.998], [0.0], [0.0]),
    )


def test_zero_learning_rate_only_updates_moments():
    check_case(
        "零学习率不改变参数",
        adamw_step,
        ([1.0], [2.0], [0.0], [0.0], 0.0, 0.9, 0.999, 1e-8, 0.01, 1),
        ([1.0], [0.2], [0.004]),
    )


def test_empty_parameter_vector():
    check_case(
        "空参数向量",
        adamw_step,
        ([], [], [], [], 0.001, 0.9, 0.999, 1e-8, 0.01, 1),
        ([], [], []),
    )


def test_inputs_are_not_modified():
    params = [1.0, -2.0]
    grads = [0.5, -1.0]
    m = [0.1, 0.2]
    v = [0.3, 0.4]
    snapshots = (params.copy(), grads.copy(), m.copy(), v.copy())
    result = adamw_step(params, grads, m, v, 0.01, 0.9, 0.999, 1e-8, 0.01, 3)
    assert (params, grads, m, v) == snapshots, "用例：输入不可修改。有输入列表被函数修改了"

    result[0][0] = 99.0
    result[1][0] = 99.0
    result[2][0] = 99.0
    assert (params, grads, m, v) == snapshots, "用例：输出独立副本。修改输出后输入也发生了变化"
