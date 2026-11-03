from exercises.embedding_backward.solution import embedding_backward
from exercises.test_utils import check_case


def test_basic_update():
    check_case(
        "基本 SGD 更新",
        embedding_backward,
        ([1], [[0.1, 0.2], [0.3, 0.4]], [[0.2, 0.4]], 0.1),
        [[0.1, 0.2], [0.28, 0.36]],
    )


def test_repeated_ids_accumulate_gradients():
    check_case(
        "重复 ID 累加梯度",
        embedding_backward,
        (
            [2, 0, 2],
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            [[1.0, 1.0], [0.5, 0.5], [1.0, 1.0]],
            0.1,
        ),
        [[0.05, 0.15], [0.3, 0.4], [0.3, 0.4]],
    )


def test_unused_rows_remain_unchanged():
    check_case(
        "未使用的行保持不变",
        embedding_backward,
        ([0], [[0.1], [0.2], [0.3]], [[1.0]], 0.1),
        [[0.0], [0.2], [0.3]],
    )


def test_gradient_order_matches_token_positions():
    check_case(
        "梯度位置与 token 位置对应",
        embedding_backward,
        ([1, 0], [[0.0], [0.0]], [[1.0], [2.0]], 0.1),
        [[-0.2], [-0.1]],
    )


def test_zero_learning_rate_returns_equal_values():
    check_case(
        "学习率为零",
        embedding_backward,
        ([0, 0], [[0.5, 0.6]], [[1.0, 1.0], [2.0, 2.0]], 0.0),
        [[0.5, 0.6]],
    )


def test_empty_ids_returns_unchanged_copy():
    check_case("空 ID 列表", embedding_backward, ([], [[0.1, 0.2]], [], 0.1), [[0.1, 0.2]])


def test_inputs_are_not_modified():
    token_ids = [0, 0]
    weight = [[0.5, 0.6]]
    grad_output = [[1.0, 1.0], [2.0, 2.0]]
    embedding_backward(token_ids, weight, grad_output, 0.1)
    assert token_ids == [0, 0], "用例：输入不可修改。token_ids 被函数修改了"
    assert weight == [[0.5, 0.6]], "用例：输入不可修改。weight 被函数修改了"
    assert grad_output == [[1.0, 1.0], [2.0, 2.0]], "用例：输入不可修改。grad_output 被函数修改了"


def test_modifying_output_does_not_modify_input_weight():
    weight = [[0.5, 0.6]]
    result = embedding_backward([0], weight, [[1.0, 1.0]], 0.1)
    result[0][0] = 99.0
    assert weight[0][0] == 0.5, "用例：输出独立副本。修改输出后，原始 weight 也被修改了"
