from exercises.test_utils import check_case
from exercises.transformer_block_forward.solution import transformer_block_forward


def zero_layer(x):
    return [[0.0] * len(row) for row in x]


def test_zero_sublayers_return_residual_input():
    check_case(
        "零子层输出保留残差输入",
        transformer_block_forward,
        ([[1.0, 2.0]], [1.0, 1.0], [1.0, 1.0], zero_layer, zero_layer, 0.0),
        [[1.0, 2.0]],
    )


def test_complete_pre_norm_block():
    def attention(x):
        return [[2.0 * value for value in row] for row in x]

    def ffn(x):
        return [row.copy() for row in x]

    check_case(
        "两次 Pre-Norm 与残差",
        transformer_block_forward,
        ([[3.0, 4.0]], [1.0, 1.0], [1.0, 1.0], attention, ffn, 0.0),
        [[5.545584412271571, 7.3941125496954285]],
    )


def test_second_norm_receives_first_residual():
    seen = []

    def attention(x):
        seen.append(x)
        return [[0.0] * len(row) for row in x]

    def ffn(x):
        seen.append(x)
        return [[0.0] * len(row) for row in x]

    transformer_block_forward([[3.0, 4.0]], [1.0, 1.0], [1.0, 1.0], attention, ffn, 0.0)
    expected = 3.0 / 3.5355339059327378, 4.0 / 3.5355339059327378
    assert seen[0] == [list(expected)]
    assert seen[1] == [list(expected)], "用例：零 Attention 时第二次归一化仍应使用第一次残差结果"


def test_input_is_not_modified():
    x = [[3.0, 4.0]]
    original = [row.copy() for row in x]
    transformer_block_forward(x, [1.0, 1.0], [1.0, 1.0], zero_layer, zero_layer)
    assert x == original, "用例：输入不可修改。x 被函数修改了"
