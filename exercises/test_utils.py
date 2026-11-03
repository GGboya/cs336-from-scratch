from collections.abc import Callable
from math import isclose
from pprint import pformat
from typing import Any


def values_equal(actual: Any, expected: Any) -> bool:
    if isinstance(actual, float) or isinstance(expected, float):
        return isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        return len(actual) == len(expected) and all(
            values_equal(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected)
        )
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            values_equal(actual[key], expected[key]) for key in actual
        )
    return actual == expected


def check_case(name: str, function: Callable, args: tuple, expected: Any) -> Any:
    """Run one exercise case and report inputs and outputs in a LeetCode-like format."""
    try:
        actual = function(*args)
    except Exception as error:
        message = (
            f"\n用例：{name}\n"
            f"输入：\n{pformat(args, width=100, sort_dicts=False)}\n"
            f"执行出错：{type(error).__name__}: {error}\n"
            "请根据上面的输入检查实现。"
        )
        raise AssertionError(message) from None

    if not values_equal(actual, expected):
        message = (
            f"\n用例：{name}\n"
            f"输入：\n{pformat(args, width=100, sort_dicts=False)}\n"
            f"期望输出：\n{pformat(expected, width=100, sort_dicts=False)}\n"
            f"实际输出：\n{pformat(actual, width=100, sort_dicts=False)}"
        )
        raise AssertionError(message)

    return actual
