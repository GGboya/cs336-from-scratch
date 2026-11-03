# 算法练习：一步 AdamW 更新

## 题目描述

AdamW 用梯度的一阶矩估计 `m` 和二阶矩估计 `v` 调整每个参数的更新幅度，并把权重衰减与梯度分开。本练习只处理一个参数向量的一步更新。

在 `solution.py` 中实现：

```python
def adamw_step(
    params: list[float],
    grads: list[float],
    m: list[float],
    v: list[float],
    learning_rate: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    step: int,
) -> tuple[list[float], list[float], list[float]]:
    ...
```

`m` 和 `v` 是本步之前的状态，`step` 是当前步数且从 1 开始。返回 `(new_params, new_m, new_v)`，三个返回值都应是新列表，不要修改任何输入列表。

## A1 AdamW 的公式

对向量中的每个位置分别计算。先使用原始梯度更新矩估计：

```text
m_t = beta1 × m_(t-1) + (1 - beta1) × g_t
v_t = beta2 × v_(t-1) + (1 - beta2) × g_t^2
```

由于 `m`、`v` 通常从零初始化，A1 在更新步长中进行 bias correction，并直接对未归一化的 `v_t` 开方后加 `eps`：

```text
step_size = learning_rate × sqrt(1 - beta2^step) / (1 - beta1^step)
updated = param - step_size × m_t / (sqrt(v_t) + eps)
new_param = (1 - learning_rate × weight_decay) × updated
```

注意顺序：先进行自适应更新，再对更新后的参数执行 decoupled weight decay。`eps` 加在未做 bias correction 的 `sqrt(v_t)` 后，用于避免除零。权重衰减不经过 `m`、`v`，所以即使梯度为零，参数仍会衰减；这与把 `weight_decay × param` 加进梯度的 L2 正则写法不同。

## 手算第一步

取向量：

```python
params = [1.0, -2.0]
grads = [0.5, -1.0]
m = [0.0, 0.0]
v = [0.0, 0.0]
learning_rate = 0.1
beta1 = 0.9
beta2 = 0.999
eps = 0.0
weight_decay = 0.01
step = 1
```

矩估计为：

```text
new_m = [0.05, -0.1]
new_v = [0.00025, 0.001]
```

第一步中：

```text
step_size = 0.1 × sqrt(1 - 0.999) / (1 - 0.9) = sqrt(0.001)
sqrt(v_t) = [0.015811..., 0.031622...]
```

由于 `eps = 0`，`step_size × m_t / sqrt(v_t)` 正好为 `[0.1, -0.1]`。

第一步的自适应更新为 `[0.1, -0.1]`，得到 `[0.9, -1.9]`；再乘以 `1 - 0.1×0.01 = 0.999`：

```text
new_params = [0.8991, -1.8981]
```

实际使用中 `eps` 通常是很小的正数，例如 `1e-8`；这里取零只是为了手算简洁。

## 边界与输入

四个向量长度相同，可以为空。`step >= 1`，`eps >= 0`。不要原地修改 `params`、`grads`、`m` 或 `v`。本题只允许使用 Python 标准库。

## 运行 case

在仓库根目录运行：

```bash
.venv/bin/python exercises/run.py adamw
```
