# 算法练习：Temperature——怎样调整概率分布的平滑程度？

## 概念

Temperature 在 Softmax 之前缩放 logits：

```text
scaled_logits[i] = logits[i] / temperature
probability = softmax(scaled_logits)
```

- `temperature = 1`：分布不变。
- `temperature > 1`：logits 差异缩小，概率更平滑，低概率 token 获得更多机会。
- `0 < temperature < 1`：logits 差异放大，分布更尖锐，高分 token 更占优势。

它只改变解码时的概率分布，不修改模型参数。

## 输入与输出

```python
def temperature_softmax(
    logits: list[float],
    temperature: float,
) -> list[float]:
    ...
```

`temperature` 必须大于 0，否则抛出 `ValueError`。返回概率列表，不修改输入。

## 手算示例

```python
logits = [0.0, 1.0]
```

当 `temperature = 2` 时，先得到 `[0, 0.5]`，所以：

```text
P(token 0) = exp(0) / (exp(0) + exp(0.5)) ≈ 0.377541
P(token 1) = exp(0.5) / (exp(0) + exp(0.5)) ≈ 0.622459
```

输出：

```python
[0.3775406687981454, 0.6224593312018546]
```

与温度为 1 时的 `[0.268941, 0.731059]` 相比，两个候选的概率更接近了。

实现时仍然要先减最大 scaled logit，避免 `exp` 溢出。

## 运行 case

```bash
.venv/bin/python exercises/run.py temperature
```
