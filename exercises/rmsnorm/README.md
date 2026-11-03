# 算法练习：RMSNorm——按均方根缩放一行向量

## 概念

RMSNorm 对单个 token 的一行向量做归一化：先计算这一行各元素的均方，再按均方根缩放，最后逐维乘以可学习的 `gain`。它不减均值。

在 `solution.py` 中实现：

```python
def rmsnorm(
    vector: list[float],
    gain: list[float],
    eps: float = 1e-5,
) -> list[float]:
    ...
```

## 输入、输出与公式

- `vector`：长度为 `d_model` 的非空向量。
- `gain`：长度也为 `d_model`，每一维都有独立、可学习的缩放系数。
- `eps`：加到均方上的小正数，避免均方接近零时除以零；手算时也可以设为 `0.0`。
- 返回同长度的新列表，不修改输入。

```text
mean_square = (vector[0]² + ... + vector[d_model-1]²) / d_model
rms = sqrt(mean_square + eps)
output[i] = vector[i] / rms * gain[i]
```

`eps` 在平方根内部。真实模型对矩阵逐行计算，但所有行共享同一组长度为 `d_model` 的 `gain`，不是每个元素各算一个 RMS。

## 数值示例

为方便手算，令 `eps = 0.0`：

```python
vector = [1.0, 2.0]
gain = [1.0, 0.5]
```

```text
mean_square = (1² + 2²) / 2 = 2.5
rms = sqrt(2.5) ≈ 1.5811388300841898
output[0] = 1 / rms × 1.0 ≈ 0.6324555320336759
output[1] = 2 / rms × 0.5 ≈ 0.6324555320336759
```

输出：

```python
[0.6324555320336759, 0.6324555320336759]
```

测试使用 `check_case` 比较浮点数，相对容差和绝对容差都是 `1e-9`，因此末尾少量浮点误差不会导致失败。

## 运行 case

在仓库根目录运行：

```bash
.venv/bin/python exercises/run.py rmsnorm
```
