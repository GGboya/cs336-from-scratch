from exercises.resource_accounting.solution import count_transformer_parameters
from exercises.test_utils import check_case


def test_small_model():
    check_case("手算小模型参数量", count_transformer_parameters, (3, 2, 1, 4), 58)


def test_zero_layers():
    check_case("没有 Block", count_transformer_parameters, (3, 2, 0, 4), 14)


def test_multiple_layers_are_independent():
    check_case("多层参数累加", count_transformer_parameters, (3, 2, 2, 4), 102)


def test_real_a1_baseline():
    check_case("真实 A1 baseline 参数量", count_transformer_parameters, (10000, 512, 4, 1344), 22696448)


def test_training_state_bytes():
    parameter_count = count_transformer_parameters(10000, 512, 4, 1344)
    check_case("参数梯度与 AdamW 状态总字节数", int, (4 * parameter_count * 4,), 363143168)
