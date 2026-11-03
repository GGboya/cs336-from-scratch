from exercises.gradient_clipping.solution import gradient_clipping
from exercises.test_utils import check_case


def test_hand_calculated_clipping():
    check_case("手算三到四向量的裁剪", gradient_clipping, ([3.0, 4.0], 1.0), [0.599999880000024, 0.799999840000032])


def test_negative_components_keep_direction():
    check_case("负分量保持方向", gradient_clipping, ([-3.0, 4.0], 2.5), [-1.49999970000006, 1.99999960000008])


def test_equal_norm_is_slightly_scaled():
    check_case("范数等于上限时因 epsilon 略微缩放", gradient_clipping, ([3.0, 4.0], 5.0), [2.99999940000012, 3.99999920000016])


def test_smaller_norm_is_not_scaled():
    check_case("范数小于上限不缩放", gradient_clipping, ([3.0, 4.0], 10.0), [3.0, 4.0])


def test_zero_gradients():
    check_case("零梯度不做除法", gradient_clipping, ([0.0, 0.0], 1.0), [0.0, 0.0])


def test_empty_gradient_list():
    check_case("空梯度列表", gradient_clipping, ([], 1.0), [])


def test_zero_max_norm():
    check_case("零范数上限", gradient_clipping, ([3.0, 4.0], 0.0), [0.0, 0.0])


def test_input_is_not_modified():
    grads = [3.0, 4.0]
    result = gradient_clipping(grads, 1.0)
    assert grads == [3.0, 4.0], "用例：输入不可修改。grads 被函数修改了"

    result[0] = 99.0
    assert grads == [3.0, 4.0], "用例：输出独立副本。修改输出后 grads 也发生了变化"


def test_unchanged_result_is_also_a_new_list():
    grads = [1.0, 2.0]
    result = gradient_clipping(grads, 10.0)
    result[0] = 99.0
    assert grads == [1.0, 2.0], "用例：未缩放时仍返回新列表。修改输出后 grads 也发生了变化"
