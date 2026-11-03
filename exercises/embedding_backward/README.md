# 算法练习：Embedding 反向传播与 SGD 更新

## 题目描述

Embedding 前向传播时，根据 token ID 取出对应行向量。训练时，后续网络会计算 loss，并把每个输出位置的梯度反传回来。本练习要求手动完成 Embedding 的梯度累加和参数更新。

本题使用普通 SGD：

```text
新权重 = 旧权重 - learning_rate × 梯度
```

真实项目使用 AdamW，梯度计算由 PyTorch autograd 完成；本练习不需要推导后续网络的梯度。

在 `solution.py` 中实现：

```python
def embedding_backward(
    token_ids: list[int],
    weight: list[list[float]],
    grad_output: list[list[float]],
    learning_rate: float,
) -> list[list[float]]:
    ...
```

## 输入与输出

- `token_ids`：前向查表时使用的 token IDs，可以包含重复 ID。
- `weight`：原 Embedding 权重表。
- `grad_output`：后续网络反传回来的梯度。第 `i` 行对应 `token_ids[i]` 表示的 token，而不是固定对应 token `i`。
- `learning_rate`：SGD 学习率。
- 返回更新后的新权重表，不要修改原始输入。

## 示例 1：只有一个 token

输入：

```python
token_ids = [1]
weight = [
    [0.1, 0.2],  # token 0
    [0.3, 0.4],  # token 1
]
grad_output = [
    [0.2, 0.4],  # grad_output[0]，对应 token_ids[0]，也就是 token 1
]
learning_rate = 0.1
```

这次只使用了 token 1，所以只需要更新第 1 行：

```text
新值 = 旧值 - 学习率 × 梯度

第一个数：0.3 - 0.1 × 0.2 = 0.28
第二个数：0.4 - 0.1 × 0.4 = 0.36
```

token 0 没有被使用，保持原值。

输出：

```python
[
    [0.1, 0.2],
    [0.28, 0.36],
]
```

## 示例 2：同一个 token 出现两次

输入：

```python
token_ids = [2, 0, 2]
weight = [
    [0.1, 0.2],  # token 0
    [0.3, 0.4],  # token 1
    [0.5, 0.6],  # token 2
]
grad_output = [
    [1.0, 1.0],  # 第一个 token 2 的梯度
    [0.5, 0.5],  # token 0 的梯度
    [1.0, 1.0],  # 第二个 token 2 的梯度
]
learning_rate = 0.1
```

先把梯度放到对应的 token 行上：

```text
token 0：只有一份梯度 [0.5, 0.5]
token 1：没有使用，梯度为 [0.0, 0.0]
token 2：出现两次，梯度相加 [1.0, 1.0] + [1.0, 1.0] = [2.0, 2.0]
```

再逐行更新：

```text
token 0：
[0.1, 0.2] - 0.1 × [0.5, 0.5]
= [0.05, 0.15]

token 1：
梯度为 0，保持 [0.3, 0.4]

token 2：
[0.5, 0.6] - 0.1 × [2.0, 2.0]
= [0.3, 0.4]
```

输出：

```python
[
    [0.05, 0.15],
    [0.3, 0.4],
    [0.3, 0.4],
]
```

## 规则

1. `grad_output[i]` 要加到 `token_ids[i]` 对应的行上。
2. 同一个 token ID 出现多次时，梯度要累加。
3. 没有使用的 token 行保持原值。
4. 使用 `新值 = 旧值 - learning_rate × 梯度` 更新。
5. token IDs 为空时，返回与 weight 内容相同的新表。
6. 不要修改输入的 token_ids、weight 或 grad_output。

## 运行 case

在仓库根目录运行：

```bash
.venv/bin/python exercises/run.py embedding_backward
```
