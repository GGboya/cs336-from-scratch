# 综合练习：完整 TransformerLM 前向组合

## 概念

本题把 Embedding、多个 Transformer Block、最终 RMSNorm 和 LM Head 按真实顺序组装：

```text
token_ids
→ Embedding 查表
→ 依次经过所有 blocks
→ 对每一行做最终 RMSNorm
→ LM Head 计算所有位置的 logits
```

`blocks` 是可调用 Block 的列表，每个 Block 接收并返回 `(sequence, d_model)` 矩阵。返回 `(sequence, vocab_size)` 的 logits；推理下一个 token 时再取最后一行。

## 输入与输出

```python
def transformer_lm_forward(
    token_ids: list[int],
    embedding_weight: list[list[float]],
    blocks: list,
    final_gain: list[float],
    lm_head_weight: list[list[float]],
    eps: float = 1e-5,
) -> list[list[float]]:
    ...
```

不要修改任何输入矩阵。最终 RMSNorm 位于所有 Block 之后、LM Head 之前，不能省略。

## 手算示例

没有 Block，输入：

```python
token_ids = [0]
embedding_weight = [[3.0, 4.0]]
blocks = []
final_gain = [1.0, 1.0]
lm_head_weight = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
eps = 0.0
```

Embedding 得到 `[3, 4]`，最终 RMSNorm 得到约 `[0.848528, 1.131371]`，LM Head 输出：

```python
[[0.848528137423857, 1.131370849898476, 1.9798989873223332]]
```

## 运行 case

```bash
.venv/bin/python exercises/run.py transformer_lm_forward
```
