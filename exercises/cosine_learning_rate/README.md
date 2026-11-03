# 算法练习：Warmup 与 Cosine 学习率

## 题目描述

训练初期常用 warmup 逐渐增大学习率，随后用 cosine schedule 平滑地降到最小值。本练习与 `cs336_basics/training.py` 中的 `get_lr_cosine_schedule` 使用相同公式和边界。

在 `solution.py` 中实现：

```python
def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    ...
```

`it` 是当前训练迭代数，从 0 开始。注意它与 AdamW 练习中从 1 开始的 `step` 不同。

## 公式与边界

### 1. Warmup

当 `it < warmup_iters` 时线性增大：

```text
lr = max_learning_rate × it / warmup_iters
```

因此 `it = 0` 时学习率为 0；`it = warmup_iters` 时不再走 warmup 分支，学习率达到峰值。

### 2. Cosine 衰减

当 `warmup_iters <= it <= cosine_cycle_iters` 且 `cosine_cycle_iters > warmup_iters` 时：

```text
progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
coefficient = 0.5 × (1 + cos(pi × progress))
lr = min_learning_rate + coefficient × (max_learning_rate - min_learning_rate)
```

`progress = 0` 时得到峰值，`progress = 1` 时得到最小值。`cos` 使用 `math.cos`，`pi` 使用 `math.pi`。

### 3. 周期结束与无效周期

当 `it > cosine_cycle_iters`，或者 `cosine_cycle_iters <= warmup_iters` 且 warmup 已结束时，返回 `min_learning_rate`。这些分支顺序与真实训练实现一致。

## 手算中间点

设峰值 `max_learning_rate = 1.0`、最小值 `min_learning_rate = 0.1`、`warmup_iters = 7`、`cosine_cycle_iters = 21`。

- `it = 3`：仍在 warmup，`lr = 1 × 3 / 7`。
- `it = 7`：cosine 的起点，`lr = 1.0`。
- `it = 14`：`progress = (14 - 7) / (21 - 7) = 0.5`，`cos(pi / 2) = 0`，所以 `lr = 0.1 + 0.5 × (1 - 0.1) = 0.55`。
- `it = 21`：周期结束，`lr = 0.1`。

本题只允许使用 Python 标准库。

## 运行 case

在仓库根目录运行：

```bash
.venv/bin/python exercises/run.py cosine_learning_rate
```
