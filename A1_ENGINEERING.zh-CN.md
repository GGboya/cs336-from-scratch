# CS336 A1：简化练习与真实工程对照

本项目采用两层学习结构：

```text
LeetCode 式练习：用 Python 列表看清核心算法
真实工程实现：用 PyTorch Tensor 处理 batch、GPU、大数据和 checkpoint
```

本文把 Assignment 1 的知识点映射到实际代码，避免只会做玩具题，却不知道它们在完整模型中的位置。

## 1. Byte-level BPE Tokenizer

对应实现：`cs336_basics/tokenizer.py`、`cs336_basics/train_tokenizer.py`。

| 工程问题 | 实际处理 |
|---|---|
| 任意 Unicode 文本 | UTF-8 编码为字节，从 256 个基础字节开始 |
| 单词、标点、空格 | 使用正则预分词，保留前导空格等文本特征 |
| special tokens | 单独保留，不与普通文本混合 merge；重叠时需要注意匹配顺序 |
| 数 GB 语料 | 按 chunk 处理，chunk 边界与 special token 对齐 |
| 多进程统计 | 分块预分词，再合并统计结果 |
| 反复统计 pair | 维护 `pair_counts` 和 `pair_locations`，merge 后增量更新 |
| 大文件编码 | `encode_iterable` 流式处理，避免一次性读入全部文本 |
| 跨 chunk 边界 | 延迟不完整的 pre-token 或 special token，保证与一次性编码一致 |
| 保存与加载 | `vocab.json` 保存 ID 与字节 token 的对应关系，`merges.txt` 保存 merge 顺序，`Tokenizer.from_files` 恢复它们 |
| 非法 UTF-8 | 拼接 token 字节后统一解码，`errors="replace"` 用 U+FFFD 替换非法序列 |

简化练习只保留核心算法；UTF-8、special tokens 和流式边界另有对应练习与真实测试。

## 2. Transformer 语言模型

对应实现：`cs336_basics/model.py`。

### 模型构造与参数初始化

- Linear 使用截断正态分布，`std = sqrt(2 / (in_features + out_features))`，截断范围为 `[-3std, 3std]`。
- Embedding 使用 `std = 1` 的截断正态分布，范围为 `[-3, 3]`。
- RMSNorm 的可学习 gain 初始化为全 1。
- SwiGLU 的 `d_ff` 取约 `8/3 × d_model`，并对齐到 64 的倍数；本项目 `d_model = 512`、`d_ff = 1344`。

完整前向路径：

```text
token_ids
→ token_embeddings
→ 多层 TransformerBlock
   → RMSNorm → 因果多头 Attention（含 RoPE）→ 残差
   → RMSNorm → SwiGLU 前馈网络 → 残差
→ 最后的 RMSNorm
→ lm_head
→ logits
```

练习中的向量通常是 `list[float]`，矩阵是 `list[list[float]]`。真实模型还包含 batch 维度：

```text
token_ids：       (batch, sequence)
Embedding 输出：  (batch, sequence, d_model)
logits：          (batch, sequence, vocab_size)
```

自回归预测下一个 token 时，使用序列最后一个位置的 logits。训练时则利用所有位置，并将目标整体右移一位。

## 3. 数值计算与训练

对应实现：`cs336_basics/training.py`、`cs336_basics/train.py`。

| 组件 | 作用 |
|---|---|
| 稳定 Softmax | 减去最大 logit，避免指数溢出 |
| 交叉熵 | 衡量正确 token 的概率是否足够高 |
| 反向传播 | 通过链式法则计算各参数梯度，由 PyTorch autograd 执行 |
| AdamW | 使用一阶矩、二阶矩和 decoupled weight decay 更新参数 |
| Gradient Clipping | 梯度全局范数过大时按比例缩小 |
| Warmup + Cosine LR | 开始逐渐提高学习率，随后按余弦曲线降低 |
| Mini-batch | 用一小批样本估计梯度，兼顾计算成本与稳定性 |

语言模型样本来自连续 token 序列：

```text
tokens： [t0, t1, t2, t3, t4]
输入：   [t0, t1, t2, t3]
目标：   [t1, t2, t3, t4]
```

真实数据预先编码为 `uint16` 文件，通过 `numpy.memmap` 按需读取，避免把整个数据集加载进内存。

## 4. Checkpoint、验证与恢复训练

`save_checkpoint` 保存模型参数、优化器状态和训练步数；`load_checkpoint` 恢复它们。

保存优化器状态很重要：AdamW 还包含一阶矩、二阶矩等内部状态，只保存模型参数并不等于完整恢复训练现场。

训练过程中需要区分：

```text
Training loss：当前训练 batch 上的表现
Validation loss：未参与参数更新的数据上的表现
```

训练 loss 下降不自动代表泛化变好，因此需要定期评估 validation loss，并保留可恢复的 checkpoint。

## 5. 解码与本地推理

对应实现：`cs336_basics/decoding.py`、`cs336_basics/generate.py`。

```text
模型输出最后一个位置的 logits
→ Temperature 调整分布
→ top-p 筛选并重新归一化候选集合
→ 采样一个 token ID
→ 接回输入
→ 重复直到 EOS 或最大生成长度
```

贪心解码是每次取最大概率的特例。Temperature、top-p 只改变选择过程，不会重新训练或修改模型参数。

## 6. Resource Accounting：参数量、计算量与训练状态

`exercises/resource_accounting` 给出了逐组件参数计数。当前 baseline 的参数构成是：

```text
Embedding + LM Head：2 × 10000 × 512 = 10,240,000
每个 Block：4 × 512² + 3 × 1344 × 512 + 2 × 512 = 3,113,984
4 个 Block + 最终 RMSNorm：4 × 3,113,984 + 512 = 12,456,448
总参数量：22,696,448
```

若参数、梯度、AdamW 的 `m` 和 `v` 都使用 float32，仅这四份状态就需要：

```text
22,696,448 × 4 份 × 4 bytes = 363,143,168 bytes ≈ 346.3 MiB
```

这不包含激活、Attention 分数、临时矩阵和 CUDA 工作区。反向传播还需要保存或重建中间激活，因此实际峰值显存远大于参数本身。

矩阵乘法的粗略 FLOPs 通常按 `2 × M × K × N` 估计（一次乘法加一次加法）。Attention 还需要计算 `QK^T` 和 `Attention × V`，其成本随序列长度平方增长；线性投影则随序列长度线性增长。完整分析时应分别统计各层矩阵乘法，再乘以训练步数，不能把参数量直接当作 FLOPs。

## 7. 真实实验对照

本仓库已完成：

- 从 256 个基础字节和 special token 开始训练 10,000 词表的 BPE。
- 处理约 2.23GB TinyStories 文本。
- 在单张 A100 40GB 上训练约 22.7M 参数的 Transformer。
- 训练 10,000 步，共处理 327.68M tokens。
- 独立 100-batch validation loss 为 1.38567，perplexity 为 3.9975。
- 下载 checkpoint 后在本地完成文本生成。

复现步骤见 `RUNNING.zh-CN.md`，实验数据见 `RESULTS.zh-CN.md`，本地生成见 `LOCAL_USAGE.zh-CN.md`。
