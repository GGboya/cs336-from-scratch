# Tokenizer 第 1 课：UTF-8 字节——任意文本怎样变成基础 token？

## 概念

Python 的 `str` 保存的是 Unicode 文本，而计算机存储和传输的是字节。UTF-8 是一种编码规则：它把每个 Unicode 字符编码为 1～4 个字节。每个字节的数值都在 `0～255` 之间，因此可以直接把字节值当作基础 token ID。

```text
文本 → UTF-8 编码 → 字节序列 → 每个字节的整数值
```

本题只完成“文本到字节 ID”这一步，不训练 BPE，也不查询训练后的词表。

## 输入与输出

在 `solution.py` 中实现：

```python
def text_to_byte_ids(text: str) -> list[int]:
    ...
```

- `text`：任意有效 Unicode 文本，可以包含 ASCII、中文、emoji 和空字符串。
- 返回一个整数列表，每个整数依次表示 UTF-8 编码中的一个字节，取值范围为 `0～255`。
- 空字符串编码后没有字节，因此返回 `[]`。

## 手算示例

### ASCII

```python
text = "A!"
```

`A` 的 UTF-8 字节是 `0x41`，即十进制 `65`；`!` 是 `0x21`，即十进制 `33`。

```python
text_to_byte_ids("A!") == [65, 33]
```

### 中文

```python
text = "你"
```

`你` 的 UTF-8 编码是三个字节：

```text
E4 BD A0 → [228, 189, 160]
```

所以输出为 `[228, 189, 160]`，而不是一个“中文标点 ID”或 Unicode 码点。

### emoji

```python
text = "🙂"
```

它的 UTF-8 编码是四个字节：

```text
F0 9F 99 82 → [240, 159, 153, 130]
```

## 为什么 256 个基础字节足够？

一个字节只有 256 种可能的值。只要为基础词表准备好 `bytes([0])`、`bytes([1])`、……、`bytes([255])`，任何 UTF-8 字节就都有对应的基础 token。任意有效 Unicode 文本都能编码成这样的字节序列，因此不存在“词表中没有这个字，无法表示”的问题。

一个中文或 emoji 在基础阶段可能对应多个 token，这是正常的。BPE 后续会把常见字节组合合并成更长的 token，以缩短序列；合并不会破坏字节表示的完整性。解码时则先按顺序拼接所有 token 的字节，再统一进行 UTF-8 解码，不能把多字节字符的每个字节单独解码。

## 附加练习：非法 UTF-8 怎样解码？

再实现反方向函数：

```python
def byte_ids_to_text(byte_ids: list[int]) -> str:
    ...
```

使用 `bytes(byte_ids).decode("utf-8", errors="replace")`。合法字节恢复原文；非法或不完整序列用替换字符 `�`（U+FFFD），不要因为单个坏字节让整个生成过程崩溃。例如：

```text
[65, 255, 66] → "A�B"
[228, 189]    → "�"
```

## 运行 case

在仓库根目录运行：

```bash
.venv/bin/python exercises/run.py utf8_bytes
```
