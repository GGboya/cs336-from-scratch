# 推理第 5 课：Softmax——logits 怎样变成概率？

## 概念

LM Head 会给每个候选 token 一个分数，但分数可能是负数，而且总和不一定为 1。Softmax 把它们转换成概率分布，使每个概率都大于等于 0，并且全部相加等于 1。

分数越高的 token，转换后的概率也越高。

## 要实现的公式

对于第 `i` 个 token：

```text
probability[i] = exp(logits[i]) / 所有 exp(logits[j]) 的和
```

返回整个概率列表，不计算 loss，也不选择 token。

## 示例

输入：

```python
logits = [0.0, 2.0]
```

计算指数：

```text
exp(0) = 1
exp(2) ≈ 7.389056
总和 ≈ 8.389056
```

分别除以总和：

```text
token 0：1 / 8.389056 ≈ 0.119203
token 1：7.389056 / 8.389056 ≈ 0.880797
```

输出：

```python
[0.11920292202211755, 0.8807970779778823]
```

可以看到，token 1 的 logit 更高，因此概率也更高；两个概率相加为 1。

## 数值稳定性要求

直接计算 `exp(10000)` 会溢出。给所有 logits 同时减去最大 logit，不会改变 Softmax 的输出概率，因此实现时应该先减最大值，再计算指数。

不要修改输入的 `logits`。本题只允许使用 Python 标准库。

## 运行 case

```bash
.venv/bin/python exercises/run.py softmax
```
