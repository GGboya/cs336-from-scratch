# 推理第 4 课：LM Head——向量怎样变成 logits？

## 概念

Transformer 最后会为当前位置输出一个上下文向量，但模型还需要回答：词表中的每个 token 有多适合成为下一个 token？

LM Head 是一个不带 bias 的线性层。它为每个候选 token 准备一行权重，分别与上下文向量做点积，得到一个分数。所有分数组成 logits：

```text
上下文向量 → LM Head → 每个候选 token 的 logits
```

logits 越大，表示模型越倾向于选择对应 token；它们还不是概率。

## 从 Query 矩阵到下一个 token

假设 Query 有 3 个 token，每个向量只有 2 个数。经过 Embedding 和 Transformer Blocks 后，得到一个 `(3, 2)` 的矩阵；真实模型还会先经过最终 RMSNorm，再把结果交给 LM Head：

```text
3 个 token IDs
→ Embedding
→ Transformer Blocks
→ 最终 RMSNorm
→ hidden_matrix = [
    [1.0, 0.0],  # 第 0 个位置的上下文向量
    [0.0, 1.0],  # 第 1 个位置的上下文向量
    [1.0, 2.0],  # 第 2 个位置，也就是最后一个位置的上下文向量
  ]
```

每一行都对应 Query 中的一个位置。推理要预测**整个 Query 后面的下一个 token**，所以取最后一行：

```text
hidden = hidden_matrix[-1] = [1.0, 2.0]
```

再把它交给 LM Head，得到词表中每个候选 token 的分数。

训练时会让矩阵中的每个位置都预测自己后面的 token；当前这道推理练习只计算最后一个位置。

## 要实现的公式

对于词表中的第 `i` 个 token：

```text
logits[i] = weight[i] 与 hidden 的点积
```

也就是把两个向量对应位置的数相乘，再全部相加。

## 输入与输出

```python
def lm_head(
    hidden: list[float],
    weight: list[list[float]],
) -> list[float]:
    ...
```

- `hidden`：**最后一个 Transformer Block 的输出经过最终 RMSNorm 后得到的上下文向量**，长度为 `d_model`。预测下一个 token 时，通常取序列最后一个位置的向量。它的来源是 `token IDs → Embedding → Transformer Blocks → 最终 RMSNorm → hidden`，不是原始 token ID，也不只是最初的 Embedding 查表结果。
- `weight`：形状为 `(vocab_size, d_model)`，`weight[i]` 是 token `i` 的权重行。
- 返回长度为 `vocab_size` 的 logits，不修改输入。

## weight 从哪里来？

`weight` 是 **LM Head 自己的参数矩阵**。词表中有多少个候选 token，它就有多少行；每一行负责给对应 token 计算分数。

它不是 Tokenizer 提供的 vocab 或 merges，也不是根据 `hidden` 临时生成的。模型初始化时，这张矩阵是随机数；训练时通过 loss 和反向传播不断更新，逐渐学会在什么上下文下给哪个 token 打更高的相对分数。

推理时直接使用 checkpoint 中训练好的矩阵。在当前项目中，它与 Embedding 权重表是两张独立的参数表：

```python
self.token_embeddings = Embedding(vocab_size, d_model)
self.lm_head = Linear(d_model, vocab_size)
```

本题直接给出训练好的 `weight`，只实现前向计算。

## 示例

输入：

```python
hidden = [1.0, 2.0]
weight = [
    [1.0, 0.0],  # token 0
    [0.0, 1.0],  # token 1
    [1.0, 1.0],  # token 2
]
```

逐行计算点积：

```text
token 0：1.0×1.0 + 2.0×0.0 = 1.0
token 1：1.0×0.0 + 2.0×1.0 = 2.0
token 2：1.0×1.0 + 2.0×1.0 = 3.0
```

输出：

```python
[1.0, 2.0, 3.0]
```

这里 token 2 的分数最高。下一步再由 Softmax 把这些分数转换成概率。

## 运行 case

```bash
.venv/bin/python exercises/run.py lm_head
```
