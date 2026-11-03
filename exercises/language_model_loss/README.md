# 算法练习：Softmax 与交叉熵 Loss

## 题目描述

语言模型会为词表中的每个 token 输出一个分数，这些分数叫作 **logits**。logits 还不是概率，需要先经过 Softmax 转成概率分布。

训练时，我们知道正确的下一个 token。交叉熵 loss 用来衡量模型这次预测得有多差：

```text
正确答案的概率越高，loss 越小
正确答案的概率越低，loss 越大
```

在 `solution.py` 中实现：

```python
def language_model_loss(
    logits: list[float],
    target_id: int,
) -> float:
    ...
```

只计算一个预测位置的 loss，不涉及反向传播和参数更新。

## 要实现的公式

先使用 Softmax 把 logits 转成概率。对于第 `i` 个 token：

```text
probability[i] = exp(logits[i]) / 所有 exp(logits[j]) 的和
```

然后取正确 token 的概率计算交叉熵：

```text
loss = -log(probability[target_id])
```

其中 `log` 是自然对数，Python 中可以使用 `math.log`。只需要返回 loss，不需要返回中间的概率列表。

## 输入与输出

- `logits`：模型对词表中所有 token 给出的分数，至少包含一个数。
- `target_id`：正确 token 的 ID，保证是有效下标。
- 返回一个浮点数，表示这次预测的交叉熵 loss。
- 不要修改输入的 `logits`。

## 示例 1：模型比较确信正确答案

输入：

```python
logits = [0.0, 2.0]
target_id = 1
```

先对 logits 做指数运算：

```text
exp(0) = 1
exp(2) ≈ 7.389056
总和 ≈ 8.389056
```

再用每个指数除以总和，得到 Softmax 概率：

```text
token 0：1 / 8.389056 ≈ 0.119203
token 1：7.389056 / 8.389056 ≈ 0.880797
```

正确答案是 token 1，因此只取它的概率计算 loss：

```text
loss = -log(0.880797) ≈ 0.126928
```

输出：

```python
0.12692801104297263
```

## 示例 2：同样的预测，但正确答案不同

输入：

```python
logits = [0.0, 2.0]
target_id = 0
```

模型的概率分布没有改变：

```text
token 0：0.119203
token 1：0.880797
```

但这次正确答案是 token 0，模型只给它分配了约 0.119203 的概率：

```text
loss = -log(0.119203) ≈ 2.126928
```

输出：

```python
2.126928011042973
```

对比两个示例可以看到：**不是最大概率的 token 决定 loss，而是模型给正确 token 分配了多少概率决定 loss。**

## 示例 3：完全不知道怎么预测

输入：

```python
logits = [0.0, 0.0, 0.0]
target_id = 1
```

三个 token 的概率都是 `1 / 3`，所以：

```text
loss = -log(1 / 3) = log(3) ≈ 1.098612
```

输出：

```python
1.0986122886681098
```

## 数值稳定性要求

直接计算 `exp(10000)` 会溢出。给所有 logits 同时减去同一个数，不会改变 Softmax 概率，因此通常先减去最大 logit，再计算指数。测试包含大数值 logits 和正确 token 概率极低的情况。

本题只允许使用 Python 标准库。

## 运行 case

在仓库根目录运行：

```bash
.venv/bin/python exercises/run.py language_model_loss
```
