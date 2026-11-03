"""算法练习：Embedding 反向传播与 SGD 更新

题目描述
========
Embedding 前向传播时，根据 token ID 取出对应行向量。训练时，后续网络会计算 loss，
并把每个输出位置的梯度反传回来。本练习要求你手动完成 Embedding 的梯度累加和参数更新。

为聚焦核心逻辑，本题使用普通 SGD：
    新权重 = 旧权重 - learning_rate × 梯度

真实项目使用 AdamW，梯度的计算由 PyTorch autograd 完成，本练习不需要推导后续网络的梯度。

输入
====
token_ids: list[int]
    前向查表时使用的 token IDs，可以包含重复 ID。
weight: list[list[float]]
    原 Embedding 权重表，weight[i] 是 token i 的行向量。
grad_output: list[list[float]]
    后续网络反传回来的梯度。grad_output[i] 对应 token_ids[i] 的输出位置。
learning_rate: float
    SGD 学习率。

输出
====
list[list[float]]
    更新后的新权重表。

规则
====
1. 将 grad_output[i] 累加到 token_ids[i] 对应的权重行梯度上。
2. 同一个 token ID 出现多次时，梯度必须累加。
3. 没有被使用的 token 行保持原值。
4. 使用普通 SGD 更新：new = old - learning_rate * gradient。
5. token_ids 为空时，返回与 weight 内容相同的新表。
6. 不要修改输入的 token_ids、weight 或 grad_output。

示例 1：基本更新
================
输入：
    token_ids = [1]
    weight = [[0.1, 0.2], [0.3, 0.4]]
    grad_output = [[0.2, 0.4]]
    learning_rate = 0.1

token 1 的梯度为 [0.2, 0.4]，因此：
    第 1 行：[0.3 - 0.1×0.2, 0.4 - 0.1×0.4] = [0.28, 0.36]
    第 0 行未使用，保持不变。

输出：
    [[0.1, 0.2], [0.28, 0.36]]

示例 2：重复 ID 的梯度累加
==========================
输入：
    token_ids = [2, 0, 2]
    weight = [
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6],
    ]
    grad_output = [
        [1.0, 1.0],
        [0.5, 0.5],
        [1.0, 1.0],
    ]
    learning_rate = 0.1

token 2 出现两次，梯度为 [2.0, 2.0]；token 0 的梯度为 [0.5, 0.5]。
输出：
    [
        [0.05, 0.15],
        [0.3, 0.4],
        [0.3, 0.4],
    ]

运行测试
========
在仓库根目录运行：
    .venv/bin/python exercises/run.py embedding_backward
"""


def embedding_backward(
    token_ids: list[int],
    weight: list[list[float]],
    grad_output: list[list[float]],
    learning_rate: float,
) -> list[list[float]]:
    """根据本文件顶部的题目说明，完成梯度累加和 SGD 更新。"""
    # 先拿到每个 token 累加之后的梯度
    m = {}

    new_grad = [row.copy() for row in grad_output]

    for i, token in enumerate(token_ids):
        if token not in m:
            m[token] = new_grad[i]
        else:
            # 累加起来
            tmp = new_grad[i].copy()
            for j in range(len(m[token])):
                m[token][j] += tmp[j]
    # m 就是 token 对应的 累加的 梯度

    # 然后计算一下 weight 即可

    res = [row.copy() for row in weight]

    for i in range(len(res)):
        if i not in m:
            continue
        for j in range(len(res[i])):
            res[i][j] = res[i][j] - learning_rate*m[i][j]

    return res
