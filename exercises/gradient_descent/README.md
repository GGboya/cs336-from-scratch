# 算法练习：一维梯度下降

## 题目描述

用一个标量参数 `w` 拟合最小值点。损失函数固定为：

```text
loss(w) = (w - 3)^2
```

在 `solution.py` 中实现：

```python
def gradient_descent(
    initial_w: float,
    learning_rate: float,
    steps: int,
) -> float:
    ...
```

从 `initial_w` 开始更新 `steps` 次，返回最后的 `w`。`steps` 是非负整数。

## 为什么减去梯度

对损失求导：

```text
dloss/dw = 2 × (w - 3)
```

梯度指向函数局部上升最快的方向，因此沿负梯度方向更新：

```text
w = w - learning_rate × 2 × (w - 3)
```

- 当 `w > 3` 时，梯度为正，减去梯度会让 `w` 变小。
- 当 `w < 3` 时，梯度为负，减去梯度会让 `w` 变大。
- 当 `w = 3` 时，梯度为零，参数已经到达最小值点。

学习率控制每一步走多远。对这个二次函数，`0 < learning_rate < 1` 时迭代会收敛到 3；学习率为 `0.5` 时一步到达最小值；学习率为 `1` 时会在最优点两侧来回跳动；学习率大于 `1` 时通常越发越远。学习率为 `0` 则不更新。

## 手算示例

```python
initial_w = 5.0
learning_rate = 0.1
steps = 2
```

第一步：

```text
梯度 = 2 × (5 - 3) = 4
w = 5 - 0.1 × 4 = 4.6
```

第二步：

```text
梯度 = 2 × (4.6 - 3) = 3.2
w = 4.6 - 0.1 × 3.2 = 4.6 - 0.32 = 4.28
```

所以返回 `4.28`。注意第二步要使用第一步更新后的参数重新计算梯度。

本题只允许使用 Python 标准库。

## 运行 case

在仓库根目录运行：

```bash
.venv/bin/python exercises/run.py gradient_descent
```
