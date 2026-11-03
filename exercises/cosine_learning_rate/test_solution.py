from exercises.cosine_learning_rate.solution import get_lr_cosine_schedule
from exercises.test_utils import check_case


MAX_LR = 1.0
MIN_LR = 0.1
WARMUP = 7
CYCLE = 21


def test_warmup_starts_at_zero():
    check_case(
        "warmup 从零开始",
        get_lr_cosine_schedule,
        (0, MAX_LR, MIN_LR, WARMUP, CYCLE),
        0.0,
    )


def test_warmup_hand_calculation():
    check_case(
        "手算 warmup 中间点",
        get_lr_cosine_schedule,
        (3, MAX_LR, MIN_LR, WARMUP, CYCLE),
        3.0 / 7.0,
    )


def test_peak_at_warmup_end():
    check_case(
        "warmup 结束达到峰值",
        get_lr_cosine_schedule,
        (7, MAX_LR, MIN_LR, WARMUP, CYCLE),
        1.0,
    )


def test_cosine_intermediate_point():
    check_case(
        "cosine 中间的一个点",
        get_lr_cosine_schedule,
        (10, MAX_LR, MIN_LR, WARMUP, CYCLE),
        0.9018241671106134,
    )


def test_cosine_midpoint_hand_calculation():
    check_case(
        "手算 cosine 正中间",
        get_lr_cosine_schedule,
        (14, MAX_LR, MIN_LR, WARMUP, CYCLE),
        0.55,
    )


def test_cycle_end_reaches_minimum():
    check_case(
        "cosine 周期结束",
        get_lr_cosine_schedule,
        (21, MAX_LR, MIN_LR, WARMUP, CYCLE),
        0.1,
    )


def test_after_cycle_stays_at_minimum():
    check_case(
        "周期结束后保持最小值",
        get_lr_cosine_schedule,
        (25, MAX_LR, MIN_LR, WARMUP, CYCLE),
        0.1,
    )


def test_zero_warmup_starts_at_peak():
    check_case(
        "没有 warmup 时从峰值开始",
        get_lr_cosine_schedule,
        (0, MAX_LR, MIN_LR, 0, CYCLE),
        1.0,
    )


def test_invalid_cycle_returns_minimum_after_warmup():
    check_case(
        "cosine 周期不大于 warmup",
        get_lr_cosine_schedule,
        (2, MAX_LR, MIN_LR, 2, 2),
        0.1,
    )


def test_scalar_inputs_are_not_modified():
    inputs = (3, MAX_LR, MIN_LR, WARMUP, CYCLE)
    inputs_before = tuple(inputs)
    get_lr_cosine_schedule(*inputs)
    assert inputs == inputs_before, "用例：输入不可修改。标量输入发生了变化"
