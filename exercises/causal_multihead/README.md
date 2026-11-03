# Causal Multihead Attention——完整因果多头注意力怎样计算？

## 本题范围

本题把线性投影、相邻配对 RoPE、逐头因果 Attention、拼头和输出投影串成一条完整数据流。输入和参数全部使用纯 Python 列表，不依赖 NumPy 或 PyTorch。

```python
def causal_multihead_attention(
    x: list[list[float]],
    wq: list[list[float]],
    wk: list[list[float]],
    wv: list[list[float]],
    wo: list[list[float]],
    num_heads: int,
    theta: float,
) -> list[list[float]]:
    ...
```

- `x`：形状为 `(sequence_length, d_model)`，每一行是一个 token 的输入向量。
- `wq`、`wk`、`wv`、`wo`：形状均为 `(d_model, d_model)`，采用 **`(out, in)`** 排列。
- `num_heads`：头的数量；`d_model` 必须能被它整除，且每头维度 `head_dim` 必须是偶数。
- `theta`：RoPE 的频率底数。
- 返回形状为 `(sequence_length, d_model)` 的新矩阵，不修改任何输入。

## 1. Q、K、V 线性投影

对 `x` 的每一行分别计算三组投影。以 Q 为例，由于权重按 `(out, in)` 排列：

```text
Q[token][out] = sum(x[token][in] * wq[out][in])
```

K、V 同理。投影后，每个 token 的 Q、K、V 都仍是 `d_model` 维向量。

## 2. 拆成连续的头

将每行的 `d_model` 维 Q、K、V 按连续区间拆成 `num_heads` 份：

```text
head_dim = d_model / num_heads
头 0：维度 [0, head_dim)
头 1：维度 [head_dim, 2 * head_dim)
...
```

每个头只使用自己的 Q、K、V 切片，独立计算 Attention。

## 3. 只对 Q、K 应用相邻配对 RoPE

RoPE 使用 token 在序列中的位置 `position`（从 0 开始）。对某个头中第 `p` 对相邻维度 `(2p, 2p+1)`：

```text
frequency = theta ** (-2p / head_dim)
angle = position * frequency
```

若旋转向量为 `[even, odd]`，则：

```text
rotated_even = even * cos(angle) - odd * sin(angle)
rotated_odd  = even * sin(angle) + odd * cos(angle)
```

Q 和 K 在每个头内分别进行上述旋转；**V 完全不旋转**。位置 0 的角度为 0，因此位置 0 的 Q、K 保持不变。

## 4. 每个头独立计算因果 Scaled Attention

对于 Query 位置 `i`，只允许 Key 位置 `j <= i`：

```text
score[i][j] = rotated_Q[i] · rotated_K[j] / sqrt(head_dim)
```

位置 `j > i` 属于未来，不参与计算。随后对当前 Query 的所有可见分数做数值稳定的 Softmax：先减去本行最大分数，再计算指数和归一化。

最后用注意力权重对**未旋转的 V** 加权求和。每个头得到一段长度为 `head_dim` 的输出。

## 5. 拼头与输出投影

对每个 token，按头 0、头 1……的顺序把各头输出拼回 `d_model` 维向量，再应用 `wo`：

```text
output[token][out] = sum(merged[token][in] * wo[out][in])
```

## 手算例：RoPE 作用于 Q/K，而 V 不旋转

取一个头、`d_model = head_dim = 2`、`theta = 1.0`，四个投影矩阵都设为单位矩阵：

```python
x = [
    [1.0, 0.0],
    [0.0, 1.0],
]
```

因此 Q、K、V 初始都等于 `x`。

- 位置 0：角度为 0，旋转后 `Q0 = K0 = [1, 0]`。
- 位置 1：角度为 1 弧度，旋转后 `Q1 = K1 = [-sin(1), cos(1)]`。
- V 不旋转，所以始终是 `V0 = [1, 0]`、`V1 = [0, 1]`。

位置 1 可以看到位置 0 和自己，两个缩放分数为：

```text
score[1][0] = -sin(1) / sqrt(2) ≈ -0.5950098395
score[1][1] = 1 / sqrt(2)       ≈  0.7071067812
```

Softmax 权重约为 `[0.2138090087, 0.7861909913]`，所以位置 1 输出：

```text
0.2138090087 * V0 + 0.7861909913 * V1
= [0.2138090087, 0.7861909913]
```

这个结果既能区分是否旋转了 Q/K，也能检查是否错误地旋转了 V：如果没有 RoPE，两个分数会相同；如果旋转 V，输出也不会是上述两个注意力权重。

## 运行 case

```bash
.venv/bin/python exercises/run.py causal_multihead
```
