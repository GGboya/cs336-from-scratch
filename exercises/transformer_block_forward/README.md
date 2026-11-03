# 综合练习：完整 Pre-Norm Transformer Block

## 概念

单独的 RMSNorm、Attention、SwiGLU 和残差相加还不能组成 Block。本题把它们按真实顺序组装：

```text
y = x + Attention(RMSNorm(x, gain1))
output = y + FFN(RMSNorm(y, gain2))
```

`attention` 和 `ffn` 是已经实现好的可调用子层，都接收并返回矩阵。本题专注 Pre-Norm 与两次残差的组合顺序。

## 输入与输出

```python
def transformer_block_forward(
    x: list[list[float]],
    gain1: list[float],
    gain2: list[float],
    attention,
    ffn,
    eps: float = 1e-5,
) -> list[list[float]]:
    ...
```

RMSNorm 对每一行独立计算。不要修改 `x`；调用子层时传入新矩阵。

## 手算示例

```python
x = [[3.0, 4.0]]
gain1 = gain2 = [1.0, 1.0]
eps = 0.0
```

假设 `attention(x) = 2x`，`ffn(x) = x`：

```text
RMSNorm(x) ≈ [0.848528, 1.131371]
Attention 输出 ≈ [1.697056, 2.262742]
第一次残差 y ≈ [4.697056, 6.262742]
```

再对 `y` 做 RMSNorm，经过 FFN 后与 `y` 相加，得到约：

```python
[[5.545584412271571, 7.3941125496954285]]
```

关键是第二次归一化使用第一次残差后的 `y`，残差相加也使用 `y`，不能回头使用原始 `x`。

## 运行 case

```bash
.venv/bin/python exercises/run.py transformer_block_forward
```
