# 算法练习：Embedding 查表

## 题目描述

token ID 只是 token 在词表中的编号。Embedding 根据编号，从权重表中取出对应向量，将离散的 token IDs 转换成神经网络可以计算的浮点向量。

在 `solution.py` 中实现：

```python
def embedding(
    token_ids: list[int],
    weight: list[list[float]],
) -> list[list[float]]:
    ...
```

## 输入

- `token_ids`：一组 token IDs，例如 `[2, 0]`。
- `weight`：Embedding 权重表，`weight[i]` 是 token `i` 对应的向量。所有向量长度相同，token ID 保证合法。

## 输出

按 `token_ids` 的顺序返回对应向量。

## 规则

1. 按 token IDs 的顺序查表，不要排序或去重。
2. 同一个 ID 出现多次时，每次都要返回对应向量。
3. token IDs 为空时返回 `[]`。
4. 返回新的向量列表，修改输出时不应改动原始 weight。

## 示例 1：基本查表

```python
token_ids = [2, 0]
weight = [
    [0.1, 0.2],  # token 0
    [0.3, 0.4],  # token 1
    [0.5, 0.6],  # token 2
]
```

输出：

```python
[
    [0.5, 0.6],
    [0.1, 0.2],
]
```

## 示例 2：保持顺序和重复项

```python
token_ids = [1, 0, 1]
weight = [
    [0.1, 0.2],
    [0.3, 0.4],
]
```

输出：

```python
[
    [0.3, 0.4],
    [0.1, 0.2],
    [0.3, 0.4],
]
```

## 形状直觉

3 个 token IDs、每个向量长度为 2：

```text
输入 shape：(3,)
输出 shape：(3, 2)
```

真实模型还包含 batch 和 sequence 维度：

```text
(batch, sequence) → (batch, sequence, d_model)
```

## 运行 case

在仓库根目录运行：

```bash
.venv/bin/python exercises/run.py embedding
```
