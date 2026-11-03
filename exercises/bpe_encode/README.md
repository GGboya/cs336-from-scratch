# 算法练习：完整的简化版 BPE 编码

## 题目描述

给定一段文本、按学习顺序排列的 merges，以及 token 词表，将文本编码成 token IDs。

完整流程：

```text
文本
→ 预分词
→ 每个 pre-token 按 merges 顺序合并
→ 查询 vocab
→ 按原顺序拼接所有 token IDs
```

在 `solution.py` 中实现：

```python
def bpe_encode(
    text: str,
    merges: list[tuple[str, str]],
    vocab: list[str],
) -> list[int]:
    ...
```

预分词函数 `_pretokenize()` 已经提供。

## 输入

- `text`：待编码文本。本题只处理小写英文字母和单个空格，不包含标点、连续空格或首尾空格；空字符串返回 `[]`。
- `merges`：按学习顺序排列的合并规则。
- `vocab`：token 词表，列表下标就是 token ID。最终 token 保证能在词表中找到。

## 输出

返回 `list[int]`，即整段文本对应的 token IDs。

## 规则

1. 使用提供的 `_pretokenize()` 切分文本。除第一个单词外，后续 pre-token 包含前导空格。
2. 每个 pre-token 先拆成单字符 token。
3. 严格按照 merges 的学习顺序应用规则；每条规则在当前序列中从左到右、不重叠合并。
4. 新 token 可以继续参与后续 merge。
5. 将最终 token 转换为 vocab 下标，并按原顺序拼接。
6. 不要修改输入的 merges 或 vocab。

## 示例 1：一个 pre-token

```python
text = "abc"
merges = [(" ", "a"), (" a", "b"), ("a", "b"), ("ab", "c")]
vocab = ["a", "b", "c", " ", " a", " ab", "ab", "abc"]
```

编码过程：

```text
abc → [a, b, c] → [ab, c] → [abc] → [7]
```

输出：

```python
[7]
```

## 示例 2：包含空格的完整句子

输入相同，但 `text = "abc ab"`。

预分词：

```text
["abc", " ab"]
```

第一个 pre-token：

```text
[a, b, c] → [ab, c] → [abc] → 7
```

第二个 pre-token：

```text
[" ", "a", "b"]
→ 应用 (" ", "a")，得到 [" a", "b"]
→ 应用 (" a", "b")，得到 [" ab"]
→ 查询词表得到 5
```

输出：

```python
[7, 5]
```

## 示例 3：不重叠合并

```python
text = "aaaa"
merges = [("a", "a"), ("aa", "aa")]
vocab = ["a", "aa", "aaaa"]
```

过程：

```text
[a, a, a, a] → [aa, aa] → [aaaa] → [2]
```

输出：

```python
[2]
```

## 运行 case

在仓库根目录运行：

```bash
.venv/bin/python exercises/run.py bpe_encode
```
