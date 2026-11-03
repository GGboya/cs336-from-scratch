"""算法练习：Embedding 查表

题目描述
========
token ID 只是 token 在词表中的编号。Embedding 根据编号，从权重表中取出对应的向量，
将离散的 token IDs 转换成神经网络可以计算的浮点向量。

你只需要实现本文件中的 embedding()。

输入
====
token_ids: list[int]
    一组 token IDs。例如 [2, 0]。
weight: list[list[float]]
    Embedding 权重表。weight[i] 是 token i 对应的向量。
    所有向量长度相同，题目保证 token ID 合法。

输出
====
list[list[float]]
    按 token_ids 的顺序返回对应向量。

规则
====
1. 按 token_ids 的顺序查表，不要对 IDs 排序或去重。
2. 同一个 ID 出现多次时，每次都要返回对应向量。
3. token_ids 为空时返回 []。
4. 返回新的向量列表，不要因为修改输出而改动原始 weight。

示例 1：基本查表
================
输入：
    token_ids = [2, 0]
    weight = [
        [0.1, 0.2],  # token 0
        [0.3, 0.4],  # token 1
        [0.5, 0.6],  # token 2
    ]

token 2 对应 [0.5, 0.6]，token 0 对应 [0.1, 0.2]。

输出：
    [
        [0.5, 0.6],
        [0.1, 0.2],
    ]

示例 2：保持顺序和重复项
========================
输入：
    token_ids = [1, 0, 1]
    weight = [
        [0.1, 0.2],
        [0.3, 0.4],
    ]

输出：
    [
        [0.3, 0.4],
        [0.1, 0.2],
        [0.3, 0.4],
    ]

形状直觉
========
如果有 3 个 token IDs，每个 Embedding 向量长度为 2：
    输入 shape：(3,)
    输出 shape：(3, 2)

真实模型还会包含 batch 和 sequence 维度：
    (batch, sequence) -> (batch, sequence, d_model)

运行测试
========
在仓库根目录运行：
    .venv/bin/python exercises/run.py embedding
"""


def embedding(token_ids: list[int], weight: list[list[float]]) -> list[list[float]]:
    """根据本文件顶部的题目说明完成 Embedding 查表。"""
    res = []
    for token in token_ids:
        res.append(weight[token].copy())
    return res
