# Scaled Attention——Query 怎样从 Values 中读取信息？

## 概念

Attention 用 Query 和每个 Key 的相似度决定应该关注哪些 Value。缩放点积注意力（scaled dot-product attention）的公式是：

```text
softmax(QK^T / sqrt(d_k)) V
```

其中 `d_k` 是每个 Query 和 Key 向量的长度。除以 `sqrt(d_k)` 可以防止维度较大时点积分数过大。Softmax 在每个 Query 对应的一整行上计算，得到一组总和为 1 的注意力权重，再用这些权重对 Values 做加权和。

本题不加入 mask，也不实现多头拆分，只聚焦这一个核心公式。

## Q、K、V 从哪里来？

它们不是用户直接提供的三种输入，也不是 Tokenizer 生成的。对同一个输入矩阵 `X`，模型分别使用三张可训练权重矩阵做线性投影：

```text
Q = X × Wq
K = X × Wk
V = X × Wv
```

训练会学习这些投影参数。Attention 中的 Query 是“当前位置想查找什么信息”，Key 是“每个位置提供什么可匹配特征”，Value 是“匹配后实际读取的内容”。这里的 Query 与用户输入的一句 query 不是同一个概念。

## 输入与输出

```python
def scaled_attention(
    queries: list[list[float]],
    keys: list[list[float]],
    values: list[list[float]],
) -> list[list[float]]:
    ...
```

- `queries`：形状为 `(num_queries, d_k)`。
- `keys`：形状为 `(num_keys, d_k)`。
- `values`：形状为 `(num_keys, d_v)`；`d_v` 可以不同于 `d_k`。
- 返回形状为 `(num_queries, d_v)` 的矩阵，不修改任何输入。

## 手算示例

输入：

```python
queries = [
    [1.0],
    [0.0],
]
keys = [
    [1.0],
    [0.0],
]
values = [
    [2.0, 4.0],
    [6.0, 8.0],
]
```

这里 `d_k = 1`，所以缩放因子也是 1。第一个 Query 与两个 Key 的点积分数为 `[1.0, 0.0]`。Softmax 后：

```text
a = exp(1) / (exp(1) + exp(0)) ≈ 0.7310585786
b = exp(0) / (exp(1) + exp(0)) ≈ 0.2689414214
```

对 Values 加权求和：

```text
第一个输出维度：a×2 + b×6 ≈ 3.0757656855
第二个输出维度：a×4 + b×8 ≈ 5.0757656855
```

第二个 Query 的两个分数都是 0，因此两个权重都是 `0.5`，输出为 `[4.0, 6.0]`。完整输出为：

```python
[
    [3.07576568548004, 5.07576568548004],
    [4.0, 6.0],
]
```

## 数值稳定性要求

计算每一行 Softmax 时，先让该行的所有分数减去此行最大分数，再计算指数。否则，直接计算很大的 `exp(score)` 可能溢出。所有分数同时减去一个常数不会改变 Softmax 结果。

本题只允许使用 Python 标准库，不需要复刻项目中的高性能张量实现。

## 运行 case

```bash
.venv/bin/python exercises/run.py scaled_attention
```
