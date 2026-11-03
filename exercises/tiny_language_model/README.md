# 训练综合练习：从零训练一个极简 Bigram 语言模型

## 概念

前面的练习分别实现了 Softmax、交叉熵、梯度和参数更新。本题把它们串成一个完整训练闭环。

Bigram 模型只根据当前 token 预测下一个 token。它有一张可训练参数表 `weight[input_id]`，这一行直接作为所有候选 token 的 logits。它没有 Embedding、Transformer 或 LM Head 的复杂结构，但训练过程与语言模型一致：

```text
输入 token ID
→ 取出一行 logits
→ Softmax
→ 交叉熵 loss
→ 计算 logits 的梯度
→ 全 batch 累加并求平均
→ SGD 更新 weight
```

初始权重全部为 0，因此每个 token 的初始概率相同。`losses` 记录每一步更新前的平均 loss。

## 输入与输出

```python
def train_bigram(
    input_ids: list[int],
    target_ids: list[int],
    vocab_size: int,
    learning_rate: float,
    steps: int,
) -> tuple[list[list[float]], list[float]]:
    ...
```

- `input_ids[i]` 的正确下一个 token 是 `target_ids[i]`，两个列表非空且等长。
- 返回更新后的 `(vocab_size, vocab_size)` 权重表，以及长度为 `steps` 的 loss 列表。
- 每步使用全部样本，梯度与 loss 都取样本平均值。
- `steps = 0` 时返回全零初始表和空 loss 列表。
- 不修改输入列表。

## 一个样本的一步更新

```python
input_ids = [0]
target_ids = [1]
vocab_size = 2
learning_rate = 0.1
steps = 1
```

初始两行权重都是 `[0, 0]`，所以输入 token 0 时：

```text
logits = [0, 0]
概率 = [0.5, 0.5]
loss = -log(0.5) ≈ 0.693147
```

正确 token 是 1，因此 logits 梯度为：

```text
[p0, p1 - 1] = [0.5, -0.5]
```

SGD 更新第 0 行：

```text
[0, 0] - 0.1 × [0.5, -0.5] = [-0.05, 0.05]
```

输出：

```python
(
    [[-0.05, 0.05], [0.0, 0.0]],
    [0.6931471805599453],
)
```

正确 token 的 logit 从 0 上升到 0.05，错误 token 的 logit 从 0 下降到 -0.05。继续训练，正确 token 的概率会逐渐升高。

## 运行 case

```bash
.venv/bin/python exercises/run.py tiny_language_model
```
