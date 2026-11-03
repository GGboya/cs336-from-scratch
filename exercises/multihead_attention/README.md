# Multihead Attention 练习：Split Heads——一个向量怎样拆成多个头？

## 概念

多头注意力不会让所有特征只参与同一组 Attention，而是把 `d_model` 维向量拆成 `num_heads` 个连续的小向量。每个头的维度为：

```text
head_dim = d_model / num_heads
```

Q、K、V 都分别进行同样的拆分。随后，**每个头独立计算自己的 Attention**：

```text
头 0：softmax(Q0 K0^T / sqrt(head_dim)) V0
头 1：softmax(Q1 K1^T / sqrt(head_dim)) V1
...
```

各头得到结果后，再按照原来的头顺序拼接回 `d_model` 维向量，最后经过输出投影。不同头可以学习关注不同的模式。

## 本题只实现拆头

为了避免把投影、Attention 和拼接混在一道题里，本题只实现一个核心函数：

```python
def split_heads(vector: list[float], num_heads: int) -> list[list[float]]:
    ...
```

- `vector`：一个 token 的 `d_model` 维向量。本题聚焦单个向量，不额外引入 batch 和序列维度。
- `num_heads`：头的数量，必须是正整数，且 `d_model` 必须能被它整除。
- 返回 `num_heads` 个连续切片，每个切片长度为 `head_dim`。
- 返回新列表，不修改输入；不实现 Attention 本身，也不实现拼头。

## 手算示例

输入：

```python
vector = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
num_heads = 3
```

这里 `head_dim = 6 / 3 = 2`，输出：

```python
[
    [1.0, 2.0],  # 头 0
    [3.0, 4.0],  # 头 1
    [5.0, 6.0],  # 头 2
]
```

注意是按连续区间拆分，不是交错取样。若之后要合并，按头 0、头 1、头 2 的顺序拼接即可还原原向量。

本题只允许使用 Python 标准库。

## 运行 case

```bash
.venv/bin/python exercises/run.py multihead_attention
```
