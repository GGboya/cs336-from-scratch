# Transformer Block 练习：Residual Add——残差连接怎样保留原信息？

## 本题范围

一个完整的 pre-norm Transformer Block 还包含 RMSNorm、多头注意力、SwiGLU 和两次残差连接。本题**只聚焦其中的逐元素残差相加**，不要求用纯列表复刻整个 Block。

要实现的核心函数是：

```python
def residual_add(
    input: list[float],
    sublayer_output: list[float],
) -> list[float]:
    ...
```

两个输入必须等长。对每个位置 `i` 计算：

```text
output[i] = input[i] + sublayer_output[i]
```

返回新列表，不修改任何一个输入。

## 残差连接在完整 Block 中的位置

pre-norm 结构先归一化输入，再把归一化后的结果交给子层；子层输出不会替代原始 `x`，而是与原始 `x` 相加。完整数据流为：

```text
attention_output = MultiHeadAttention(RMSNorm(x))
x = residual_add(x, attention_output)       # 第一次残差连接

ffn_output = SwiGLU(RMSNorm(x))
output = residual_add(x, ffn_output)        # 第二次残差连接
```

也就是两次使用 `x + sublayer(norm(x))` 的形式。第一次相加使用 Attention 前的原始 `x`；第二次相加使用已经完成 Attention 残差连接后的新 `x`。残差通路保留了原始表示，并让子层只需学习要添加的变化。

## 手算示例

输入：

```python
input = [1.0, 2.0, 3.0]
sublayer_output = [0.5, -1.0, 4.0]
```

逐元素相加：

```text
1.0 + 0.5  = 1.5
2.0 + -1.0 = 1.0
3.0 + 4.0  = 7.0
```

输出：

```python
[1.5, 1.0, 7.0]
```

这里只做加法，不做 Softmax、归一化或激活。本题只允许使用 Python 标准库。

## 运行 case

```bash
.venv/bin/python exercises/run.py transformer_block
```
