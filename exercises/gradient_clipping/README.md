# 算法练习：按 L2 范数裁剪梯度

## 题目描述

梯度过大时，参数更新可能不稳定。梯度裁剪保留梯度方向，只在整体 L2 范数超过上限时缩小它。

在 `solution.py` 中实现：

```python
def gradient_clipping(
    grads: list[float],
    max_norm: float,
) -> list[float]:
    ...
```

返回缩放后的新列表，不要修改输入。`max_norm` 是非负数。

## 公式

先计算所有梯度分量共同组成的 L2 范数：

```text
norm = sqrt(Σ_i grads[i]^2)
```

按照 A1 的真实实现，先计算缩放系数：

```text
scale = max_norm / (norm + 1e-6)
```

只有 `scale < 1` 时才缩放：

```text
clipped[i] = grads[i] × scale
```

裁剪后的范数会略小于 `max_norm`，方向保持不变。`1e-6` 用于避免除零，也意味着当 `norm == max_norm` 时，`scale` 仍会略小于 1，因此会发生非常小的缩放；当范数明显小于上限时则不缩放。

零梯度或空梯度会得到大于 1 的缩放系数（`max_norm > 0`），直接返回副本；`max_norm = 0` 时缩放结果为零。

## 手算示例

```python
grads = [3.0, 4.0]
max_norm = 1.0
```

```text
norm = sqrt(3^2 + 4^2) = 5
scale = 1 / (5 + 1e-6) ≈ 0.19999996
clipped ≈ [0.59999988, 0.79999984]
```

如果 `max_norm = 5.0`，原范数恰好等于上限，缩放系数仍略小于 1，因此结果略小于 `[3.0, 4.0]`。

本题只允许使用 Python 标准库。

## 运行 case

在仓库根目录运行：

```bash
.venv/bin/python exercises/run.py gradient_clipping
```
