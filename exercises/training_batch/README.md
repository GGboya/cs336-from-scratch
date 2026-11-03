# 算法练习：构造语言模型训练 Batch

## 题目描述

语言模型学习“根据前面的 token 预测下一个 token”。连续 token 序列中，同一个起点可以构造一对输入和目标：目标就是输入右移一位后的序列。

为了让练习不依赖随机数，在 `solution.py` 中实现确定性的版本：

```python
def get_batch(
    tokens: list[int],
    start_indices: list[int],
    context_length: int,
) -> tuple[list[list[int]], list[list[int]]]:
    ...
```

`start_indices` 中的每个起点生成一行样本，起点顺序和重复项都要保留。返回 `(inputs, targets)`，两者形状相同，均为 `len(start_indices)` 行、`context_length` 列。

## 样本构造公式

对每个起点 `start`：

```text
inputs_row  = tokens[start : start + context_length]
targets_row = tokens[start + 1 : start + context_length + 1]
```

也就是：

```text
inputs[i]  = tokens[start + i]
targets[i] = tokens[start + i + 1]
```

模型在输入第 `i` 个位置看到上下文后，要学习预测 `targets[i]`。一个切片因此提供 `context_length` 个预测位置，而不是只预测窗口末尾的一个 token。

合法起点需要保证目标和切片末尾都存在：

```text
0 <= start <= len(tokens) - context_length - 1
```

测试保证起点合法，`context_length` 为正数。`start_indices` 为空时返回 `([], [])`。

## 示例

```python
tokens = [10, 11, 12, 13, 14, 15]
start_indices = [0, 2]
context_length = 3
```

起点 0 的输入是 `[10, 11, 12]`，目标是 `[11, 12, 13]`；起点 2 的输入是 `[12, 13, 14]`，目标是 `[13, 14, 15]`。

```python
inputs = [
    [10, 11, 12],
    [12, 13, 14],
]
targets = [
    [11, 12, 13],
    [13, 14, 15],
]
```

## 真实工程中的随机采样与 memmap

真实 `cs336_basics/training.py` 会随机生成起点，再取相同的输入、目标切片。本练习把起点作为参数，便于精确测试。

训练数据通常很大，不能总把整个 token 文件读进内存。NumPy memmap 把磁盘文件映射为可随机访问的数组：程序按起点切片时，由操作系统按需读取相关页面，不必一次性加载全部数据。memmap 不改变样本构造公式，它解决的是大规模数据的存储与按需读取问题。

本练习使用纯列表演示公式，只允许使用 Python 标准库，不要修改输入列表。

## 运行 case

在仓库根目录运行：

```bash
.venv/bin/python exercises/run.py training_batch
```
