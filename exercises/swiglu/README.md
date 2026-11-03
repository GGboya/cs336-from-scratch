# 算法练习：SwiGLU——逐元素门控

## 概念

SwiGLU 的核心是：对 `gate` 的每个元素做 SiLU，再与 `value` 对应位置的元素相乘。`gate` 决定每个位置让多少信息通过，这不是矩阵乘法。

在 `solution.py` 中实现：

```python
def silu(x: float) -> float:
    ...


def swiglu(gate: list[float], value: list[float]) -> list[float]:
    ...
```

- `gate` 和 `value` 是等长向量。
- 返回等长的新列表，不修改输入。

## 公式与逐元素计算

```text
sigmoid(x) = 1 / (1 + exp(-x))
silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
output[i] = silu(gate[i]) * value[i]
```

为了避免大负数导致 `exp(-x)` 溢出，实现时按符号计算等价的 SiLU：

```text
x >= 0：silu(x) = x / (1 + exp(-x))
x < 0： silu(x) = x * exp(x) / (1 + exp(x))
```

## 示例

```python
gate = [0.0, 1.0, -1.0]
value = [2.0, 2.0, 2.0]
```

```text
silu(0) = 0
silu(1) ≈ 0.7310585786300049
silu(-1) ≈ -0.2689414213699951
```

逐元素乘以 `value`，输出：

```python
[0.0, 1.4621171572600098, -0.5378828427399902]
```

第一个位置的门输出为 0，因此对应信息被完全截断；后两个位置分别保留不同比例和符号的信息。

## 与真实项目的对应关系

本题直接给出已经算好的 `gate` 和 `value`，只练核心门控。真实项目中的 SwiGLU 还有 `w1/w2/w3` 三个矩阵：

```text
gate = w1(x)
value = w3(x)
output = w2(silu(gate) * value)
```

其中 `w1`、`w3` 把输入映射到隐藏维度，`w2` 再映射回模型维度。

## 运行 case

在仓库根目录运行：

```bash
.venv/bin/python exercises/run.py swiglu
```
