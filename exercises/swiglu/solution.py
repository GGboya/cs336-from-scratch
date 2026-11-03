"""SwiGLU：SiLU 与逐元素门控。题目和示例见同目录的 README.md。"""


def silu(x: float) -> float:
    """稳定地计算一个标量的 SiLU。"""
    raise NotImplementedError("Implement silu in exercises/swiglu/solution.py")


def swiglu(gate: list[float], value: list[float]) -> list[float]:
    """返回 silu(gate) 与 value 的逐元素乘积。"""
    raise NotImplementedError("Implement swiglu in exercises/swiglu/solution.py")
