# 算法练习：Softmax 与交叉熵的反向传播

## 题目描述

语言模型在一个位置输出 logits `z`，正确 token 的下标是 `target_id`。Softmax 概率和交叉熵 loss 分别为：

```text
p_i = exp(z_i) / Σ_j exp(z_j)
loss = -log(p_target_id)
```

在 `solution.py` 中实现：

```python
def softmax_cross_entropy_backward(
    logits: list[float],
    target_id: int,
) -> list[float]:
    ...
```

返回 loss 对每个 logit 的梯度 `dloss/dlogits`，不返回 loss。

## 链式法则的结果

Softmax 的输出依赖所有 logits，交叉熵又依赖正确 token 的 Softmax 概率。把这两层复合函数一起应用链式法则，可得到：

```text
grad[i] = p_i - one_hot(target_id)[i]
```

也就是：

- 当 `i != target_id` 时，`grad[i] = p_i`。
- 当 `i == target_id` 时，`grad[i] = p_i - 1`。

所有梯度的总和为 0。这个式子已经把 Softmax 与交叉熵的求导合并，不需要再额外乘一次 Softmax 的导数。

## 手算示例

```python
logits = [0.0, 0.0]
target_id = 0
```

两个概率都为 `0.5`，所以：

```text
grad = [0.5 - 1, 0.5] = [-0.5, 0.5]
```

正确 token 的 logit 应增大，另一个 token 的 logit 应减小；后续参数更新会沿负梯度方向进行。

## 数值稳定性要求

先找出最大 logit `M`，再计算：

```text
shifted_i = z_i - M
p_i = exp(shifted_i) / Σ_j exp(shifted_j)
```

给所有 logits 同时减去 `M` 不改变 Softmax 概率，并可避免直接计算 `exp(10000)` 导致溢出。不要修改输入列表。本题只允许使用 Python 标准库。

## 运行 case

在仓库根目录运行：

```bash
.venv/bin/python exercises/run.py softmax_cross_entropy_backward
```
