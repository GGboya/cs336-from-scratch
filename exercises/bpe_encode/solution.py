"""算法练习：完整的简化版 BPE 编码

题目描述
========
给定一段文本、按学习顺序排列的 merges，以及 token 词表，将文本编码成 token IDs。

完整流程：
    文本
    -> 预分词
    -> 每个 pre-token 按 merges 顺序合并
    -> 查询 vocab
    -> 按原顺序拼接所有 token IDs

你只需要实现 bpe_encode()。预分词函数 _pretokenize() 已经提供。

输入
====
text: str
    待编码文本。为聚焦 BPE，本题只处理小写英文字母和单个空格，
    不包含标点、连续空格或首尾空格。空字符串应返回 []。
merges: list[tuple[str, str]]
    按学习顺序排列的合并规则。
vocab: list[str]
    token 词表，列表下标就是 token ID。题目保证最终 token 都能在词表中找到。

输出
====
list[int]
    整段文本对应的 token IDs。

规则
====
1. 使用提供的 _pretokenize() 切分文本。除第一个单词外，后续 pre-token 包含前导空格。
2. 每个 pre-token 先拆成单字符 token。
3. 严格按照 merges 的学习顺序应用规则；每条规则在当前序列中从左到右、不重叠合并。
4. 新 token 可以继续参与后续 merge。
5. 将每个 pre-token 的最终 token 转换为 vocab 下标，并按原顺序拼接。
6. 不要修改输入的 merges 或 vocab。

示例 1：一个 pre-token
=====================
输入：
    text = "abc"
    merges = [(" ", "a"), (" a", "b"), ("a", "b"), ("ab", "c")]
    vocab = ["a", "b", "c", " ", " a", " ab", "ab", "abc"]

编码过程：
    abc -> [a, b, c] -> [ab, c] -> [abc] -> [7]

输出：
    [7]

示例 2：包含空格的完整句子
==========================
输入与示例 1 相同，但 text = "abc ab"。

预分词：
    ["abc", " ab"]

第一个 pre-token：
    [a, b, c] -> [ab, c] -> [abc] -> 7

第二个 pre-token：
    [" ", "a", "b"]
    -> 应用 (" ", "a")，得到 [" a", "b"]
    -> 应用 (" a", "b")，得到 [" ab"]
    -> 查询词表得到 5

输出：
    [7, 5]

示例 3：不重叠合并
==================
输入：
    text = "aaaa"
    merges = [("a", "a"), ("aa", "aa")]
    vocab = ["a", "aa", "aaaa"]

编码过程：
    [a, a, a, a] -> [aa, aa] -> [aaaa] -> [2]

输出：
    [2]

运行测试
========
在仓库根目录运行：
    .venv/bin/python exercises/run.py bpe_encode
"""


def _pretokenize(text: str) -> list[str]:
    """Split on single spaces while attaching each space to the following word."""
    if text == "":
        return []
    words = text.split(" ")
    pretokens = [words[0]]
    for word in words[1:]:
        pretokens.append(" " + word)
    return pretokens


def bpe_encode(text: str, merges: list[tuple[str, str]], vocab: list[str]) -> list[int]:
    """根据本文件顶部的题目说明，将文本编码成 token IDs。"""
    raise NotImplementedError("Implement bpe_encode in exercises/bpe_encode/solution.py")
