# 算法练习：Linear——无 bias 线性变换

## 概念

Linear 用同一组权重变换输入矩阵的每一行。本题没有 bias，只做矩阵乘法：输入矩阵乘以**转置后的权重矩阵**。

在 `solution.py` 中实现：

```python
def linear(
    input_matrix: list[list[float]],
    weight: list[list[float]],
) -> list[list[float]]:
    ...
```

## 输入、输出与形状

- `input_matrix`：形状为 `(num_rows, in_features)`，每一行是一个输入向量。
- `weight`：形状为 `(out_features, in_features)`，`weight[o]` 是第 `o` 个输出的权重行。
- 输出：形状为 `(num_rows, out_features)`，不修改输入。

对第 `r` 行输入和第 `o` 个输出：

```text
output[r][o] = sum(input_matrix[r][k] * weight[o][k] for k in range(in_features))
```

因此矩阵形式是 `output = input_matrix @ weight.T`，没有额外的 `+ bias`。`weight` 不转置前的每一行正好对应一个输出，而不是一个输入维度。

## 手算示例

```python
input_matrix = [
    [1.0, 2.0],
    [3.0, 4.0],
]
weight = [
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
]
```

输入形状是 `(2, 2)`，权重形状是 `(3, 2)`，所以输出形状是 `(2, 3)`。逐行做点积：

```text
第 0 行：1×1 + 2×0 = 1，1×0 + 2×1 = 2，1×1 + 2×1 = 3
第 1 行：3×1 + 4×0 = 3，3×0 + 4×1 = 4，3×1 + 4×1 = 7
```

输出：

```python
[
    [1.0, 2.0, 3.0],
    [3.0, 4.0, 7.0],
]
```

没有输入行时返回 `[]`；没有输出特征时，每个输入行对应一个空列表。

## 运行 case

在仓库根目录运行：

```bash
.venv/bin/python exercises/run.py linear
```
