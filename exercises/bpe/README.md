# 算法练习：训练一个简化版 BPE

## 题目描述

给定一组已经预分词好的单词，从单个字符开始，不断合并出现次数最多的相邻片段，直到达到目标词表大小。

你只需要实现 `solution.py` 中的 `train_bpe()`：

```python
def train_bpe(
    words: list[str],
    vocab_size: int,
) -> tuple[list[str], list[tuple[str, str]]]:
    ...
```

返回：

- 最终词表，列表下标就是 token ID。
- 按学习顺序排列的 merge 记录。

## 规则

1. 初始词表由所有输入单词中的单字符组成，按字典序排列。
2. 每轮统计所有单词中相邻 pair 的出现次数，同一个 pair 在同一单词中出现多次时要重复计数。
3. 合并频次最高的 pair；频次相同时，选择字典序更大的 pair。
4. 在同一单词中从左到右、不重叠地合并选中的 pair。
5. 将合并后的新 token 追加到词表，并将 pair 追加到 merges。
6. 达到 `vocab_size` 或不存在可合并 pair 时停止。
7. 不要修改输入的 `words` 列表。

本题暂时忽略 UTF-8、正则预分词和 special tokens，专注核心算法。

## 示例 1

输入：

```python
words = ["ab", "ab", "abc"]
vocab_size = 4
```

初始状态：

```text
词表：[a, b, c]
单词：[a, b]、[a, b]、[a, b, c]
```

pair 频次为 `(a, b)=3`、`(b, c)=1`，因此合并 `(a, b)`。

合并完成后，词表和单词分别变成：

```text
词表：[a, b, c, ab]
单词：[ab]、[ab]、[ab, c]
```

可以看到，`ab` 已经成为一个新的 token；第三个单词则从三个 token 变成了两个 token。

输出：

```python
vocab = ["a", "b", "c", "ab"]
merges = [("a", "b")]
```

## 示例 2

继续使用相同输入，但令 `vocab_size = 5`，还可以合并 `(ab, c)`：

```python
vocab = ["a", "b", "c", "ab", "abc"]
merges = [("a", "b"), ("ab", "c")]
```

## 示例 3：不重叠合并

输入：

```python
words = ["aaa"]
vocab_size = 2
```

合并 `(a, a)` 后，结果为 `[aa, a]`，而不是 `[a, aa]`：

```python
vocab = ["a", "aa"]
merges = [("a", "a")]
```

## 运行测试

在仓库根目录运行：

```bash
.venv/bin/python exercises/run.py bpe
```

测试未通过时，阅读失败信息并修改 `solution.py`。测试全部通过后，可以再思考一个进阶问题：如何避免每轮都重新扫描所有单词？
