# Causal Mask——怎样阻止模型看见未来？

## 概念

自回归语言模型在位置 `i` 只能利用位置 `0` 到位置 `i` 的信息，不能利用位置 `i+1` 及之后的未来 token。Causal mask 就是用来阻止未来信息进入 Attention 的。

本题聚焦一个独立函数：把因果掩码直接应用到 Attention 的分数矩阵上。

```python
def apply_causal_mask(scores: list[list[float]]) -> list[list[float]]:
    ...
```

对于第 `i` 行、第 `j` 列：

- 当 `j <= i` 时，位置 `i` 可以看见位置 `j`，保留原分数。
- 当 `j > i` 时，位置 `j` 属于未来，把该分数替换为 `float("-inf")`。

输入是方阵，返回一个同样形状的新矩阵，不修改输入。

## 为什么在 Softmax 前使用？

实际顺序是：

```text
QK^T / sqrt(d_k)
→ apply_causal_mask
→ 对每一行计算 Softmax
→ 乘以 V
```

被替换成 `-inf` 的未来位置，在 Softmax 中有 `exp(-inf - max_score) = 0`，因此注意力权重为 0，不会从未来位置的 Value 中读取信息。每一行的对角线位置 `j == i` 始终保留，所以每行至少有一个可见位置。

应当在完成点积缩放之后、Softmax 之前应用 mask；不要对 Softmax 后的概率直接填 `-inf`。

## 手算示例

输入：

```python
scores = [
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
]
```

输出：

```python
[
    [1.0, float("-inf"), float("-inf")],
    [4.0, 5.0, float("-inf")],
    [7.0, 8.0, 9.0],
]
```

第 0 行只能看见位置 0；第 1 行能看见位置 0、1；第 2 行能看见位置 0、1、2。

本题只允许使用 Python 标准库。

## 运行 case

```bash
.venv/bin/python exercises/run.py causal_mask
```
