# 算法练习：计算 Transformer 参数量与训练状态内存

## 概念

Resource accounting 帮助我们在训练前估计模型规模。当前 A1 模型没有 bias，Embedding 与 LM Head 参数独立，每个 Block 包含：

- Q/K/V/O 四张 `d_model × d_model` 矩阵；
- SwiGLU 的 w1/w2/w3 三张 `d_ff × d_model` 矩阵；
- 两个 RMSNorm，每个有 `d_model` 个 gain。

另外还有一层最终 RMSNorm。

## 要实现的公式

```text
Embedding + LM Head：2 × vocab_size × d_model
每个 Block：        4 × d_model² + 3 × d_ff × d_model + 2 × d_model
最终 RMSNorm：      d_model
```

实现：

```python
def count_transformer_parameters(
    vocab_size: int,
    d_model: int,
    num_layers: int,
    d_ff: int,
) -> int:
    ...
```

## 手算示例

```python
vocab_size = 3
d_model = 2
num_layers = 1
d_ff = 4
```

```text
Embedding + LM Head = 2×3×2 = 12
Block = 4×2² + 3×4×2 + 2×2 = 44
最终 RMSNorm = 2
总参数量 = 58
```

## 训练状态内存

若每个参数使用 4 字节，训练时通常还需要一份梯度以及 AdamW 的 `m`、`v` 两份状态：

```text
参数 + 梯度 + m + v = 4 × 参数量 × 4 字节
```

这还不包含激活、临时矩阵和 CUDA 工作区，因此实际显存会更高。

## 运行 case

```bash
.venv/bin/python exercises/run.py resource_accounting
```
