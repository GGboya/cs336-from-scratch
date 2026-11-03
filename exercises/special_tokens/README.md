# 算法练习：Special Tokens——控制标记怎样保留下来？

## 概念

`<|endoftext|>` 等 special token 是模型词表中的完整控制标记，不能先当作普通字符做 BPE。编码前需要把文本切开：special token 单独成为一段，普通文本则保持原样，之后再分别处理。

```text
普通文本 + special token + 普通文本
→ [普通文本片段, special token, 普通文本片段]
```

本题只负责切分并保留 special token，不把它们转换成 ID。

## 输入与输出

在 `solution.py` 中实现：

```python
def split_special_tokens(
    text: str,
    special_tokens: list[str],
) -> list[str]:
    ...
```

- `text`：待切分文本。
- `special_tokens`：需要保留的特殊字符串，每个字符串非空。
- 按原顺序返回切分结果。命中的 special token 必须独立保留，普通文本片段不得改写。
- 不返回由切分产生的空字符串；输入 `text == ""` 时返回 `[]`。
- 如果没有 special token，非空文本应作为唯一元素原样返回。
- 同一位置有多个 special token 能匹配时，选择最长的一个。
- 不要修改输入的 `special_tokens` 列表。

## 手算示例

### 基本切分

```python
text = "hello<|end|>world"
special_tokens = ["<|end|>"]
```

从左到右扫描，在位置 5 命中 `<|end|>`。它前面的 `hello`、匹配到的标记和后面的 `world` 分别保留：

```python
["hello", "<|end|>", "world"]
```

### 重叠时最长匹配优先

```python
text = "x<|a|>longy<|a|>"
special_tokens = ["<|a|>", "<|a|>long"]
```

在 `x` 后面，`<|a|>` 和 `<|a|>long` 都能从当前位置开始匹配，但必须选择更长的 `<|a|>long`。输出：

```python
["x", "<|a|>long", "y", "<|a|>"]
```

如果只按输入列表顺序找到第一个匹配就停止，就会错误地先识别短标记 `<|a|>`，把剩余的 `long` 当作普通文本。

### 没有命中与空串

```python
split_special_tokens("普通文本", ["<|end|>"]) == ["普通文本"]
split_special_tokens("", ["<|end|>"]) == []
```

## 实现提示与真实工程

参考答案使用普通循环：先把候选 token 按长度从长到短排列，再逐位置用 `startswith` 尝试匹配；没有匹配时积累普通字符，匹配成功后先交出之前积累的普通片段。

真实 Tokenizer 通常把转义后的 special token 编译成正则，并按长度降序排列 alternation，一次扫描完成切分。处理流式输入或文件 chunk 时，还要注意标记可能横跨 chunk 边界：如果缓冲区末尾是某个 special token 的前缀，就需要保留这段后缀，等下一块数据到来后再判断。若短标记已经匹配、但更长的重叠标记仍可能随后续数据出现，也不能过早确认短标记。

## 运行 case

在仓库根目录运行：

```bash
.venv/bin/python exercises/run.py special_tokens
```
