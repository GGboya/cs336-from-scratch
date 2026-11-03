# 算法练习：RoPE——按位置旋转向量

## 概念

RoPE（旋转位置编码）把向量中的二维分量看成平面坐标，再根据 token 的位置旋转它们。旋转角度同时取决于位置和该二维分量所在的配对编号。

在 `solution.py` 中实现：

```python
def rope(
    vector: list[float],
    position: int,
    theta: float,
) -> list[float]:
    ...
```

- `vector`：一个位置的偶数维向量，长度为 `d`。
- `position`：该 token 的序列位置。
- `theta`：控制不同配对的频率，通常大于 `1.0`。
- 返回同长度的新列表，不修改输入。

## 相邻配对与公式

本题按 `(0, 1)`、`(2, 3)`……配对。对第 `j` 对（从 0 开始），令：

```text
x = vector[2j]
y = vector[2j+1]
angle = position * theta ** (-2j / d)
```

二维旋转公式为：

```text
output[2j]   = x * cos(angle) - y * sin(angle)
output[2j+1] = x * sin(angle) + y * cos(angle)
```

**这是教学简化版**：只处理一个位置的一个向量，并固定使用相邻两维配对。真实 RoPE 还常见**分半配对**，即把 `(0, d/2)`、`(1, d/2+1)` 等配对；两种索引布局不能混用，本题不声称与分半配对实现完全同构。当前仓库的 `cs336_basics/model.py` 恰好也采用相邻配对，但真实类还处理 batch、attention head、序列维度和预计算的 cos/sin 缓存，本题不包含这些工程部分。

## 具体例子

输入：

```python
vector = [1.0, 0.0]
position = 1
theta = 10000.0
```

因为 `d = 2`，唯一配对的频率指数为 0：

```text
angle = 1 * 10000 ** 0 = 1
output[0] = 1 * cos(1) - 0 * sin(1) ≈ 0.5403023058681398
output[1] = 1 * sin(1) + 0 * cos(1) ≈ 0.8414709848078965
```

输出：

```python
[0.5403023058681398, 0.8414709848078965]
```

若有 4 维，第 0 对的频率指数是 0，第 1 对是 `-2/4 = -0.5`，所以两对旋转角通常不同。`position = 0` 时所有角度都是 0，向量保持不变。

## 运行 case

在仓库根目录运行：

```bash
.venv/bin/python exercises/run.py rope
```
