# 推理第 6 课：贪心解码——概率怎样变成文本？

## 概念

Softmax 输出的是每个候选 token 的概率。要生成文本，还需要先选一个 token，再把它的 ID 转换回对应的文本片段。

本题使用最简单的 **Greedy Decoding（贪心解码）**：每一步都选择概率最高的 token。如果多个 token 并列最高，选择 ID 最小的那个。

```text
概率列表 → 找到最大概率的下标 → token ID → vocab[ID] → 文本片段
```

真实模型还可以使用 Temperature、top-p 等采样方法，不一定选择概率最高的 token。本题只实现单步贪心解码。

## 输入与输出

```python
def greedy_decode_step(
    probabilities: list[float],
    vocab: list[str],
) -> tuple[int, str]:
    ...
```

- `probabilities[i]`：token `i` 成为下一个 token 的概率。
- `vocab[i]`：token `i` 对应的文本片段，列表下标就是 token ID。
- 返回 `(选中的 token ID, 对应文本)`。
- 输入列表非空且长度相同，不修改输入。

## 示例

输入：

```python
probabilities = [0.2, 0.7, 0.1]
vocab = ["there", ",", "."]
```

最大概率是 `0.7`，它的下标为 `1`，所以：

```text
选中的 token ID = 1
vocab[1] = ","
```

输出：

```python
(1, ",")
```

如果原输入是 `Once upon a time`，就把这个文本片段接到后面：

```text
Once upon a time,
```

然后把新 token ID 接回模型输入，再次执行“模型 → Softmax → 解码”，就能继续生成下一个 token。遇到 EOS 或达到最大生成长度时停止，最终得到的完整文本就是用户看到的输出。

## 运行 case

```bash
.venv/bin/python exercises/run.py greedy_decoding
```
