# 算法练习：自回归生成——怎样把单步预测串成完整序列？

## 概念

语言模型一次只预测下一个 token。自回归生成会把选中的 token 接回输入，再让模型根据更新后的完整序列预测下一步，如此循环。

```text
当前 IDs → 模型输出下一位置概率 → 贪心选择一个 ID → 接回当前 IDs → 下一轮
```

本题把“模型”简化为一个可调用对象：它接收当前完整 token ID 列表，直接返回下一位置的概率列表。因此不需要处理 logits、Softmax、Tokenizer 或文本解码。

## 输入与输出

在 `solution.py` 中实现：

```python
def greedy_generate(
    prompt_ids: list[int],
    model,
    max_new_tokens: int,
    eos_id: int,
) -> list[int]:
    ...
```

- `prompt_ids`：提示词的 token IDs。
- `model`：可调用对象，调用形式为 `model(current_ids)`；参数是当前完整 IDs，返回值是下一位置概率列表。
- `max_new_tokens`：最多新增的 token 数量，不包含 prompt 长度。
- `eos_id`：结束标记的 token ID。
- 返回包含原始 prompt 和新生成 token 的完整 ID 列表。
- 每轮选择概率最大的 token；并列最大时选择 ID 最小的 token。
- 如果选中 EOS，立即停止，不再调用模型，也不把 EOS 放入返回给用户的结果。
- 达到 `max_new_tokens` 时停止。`max_new_tokens = 0` 时不调用模型，直接返回 prompt 的副本。
- 不要修改输入的 `prompt_ids` 列表。

## 手算示例

```python
prompt_ids = [10]
max_new_tokens = 5
eos_id = 2
```

假设模型依次给出：

```text
第 1 轮：接收 [10]
概率 [0.1, 0.8, 0.1] → 选择 token 1 → IDs 变为 [10, 1]

第 2 轮：接收 [10, 1]
概率 [0.2, 0.2, 0.6] → 选择 token 2（EOS）→ IDs 变为 [10, 1, 2]
```

第 2 轮遇到 EOS，立即停止。即使 `max_new_tokens = 5`，最终用户可见的输出也只有：

```python
[10, 1]
```

这与真实生成器一致：EOS 是停止信号，不是需要展示给用户的文本内容。

如果没有遇到 EOS，则最多执行 5 轮，输出长度为 `1 + 5 = 6`。每轮传入模型的列表都必须包含此前已经生成的 token，不能一直只传原始 prompt。

真实模型的 context length 有限。输入过长时，通常只把最后 `context_length` 个 IDs 交给模型，但返回值仍保留完整 prompt 和全部生成结果。真实生成还会在每次选 token 前应用 Temperature 和 top-p；本题已经由 `model` 直接给出概率，只专注自回归循环。

## 实现提示

先复制 `prompt_ids`，之后只修改副本。单步选最大概率时，从下标 0 开始，并只在遇到**严格更大**的概率时更新最佳 ID，就能自然实现“平频选小 ID”。

## 运行 case

在仓库根目录运行：

```bash
.venv/bin/python exercises/run.py autoregressive_generation
```
