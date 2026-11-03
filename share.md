# 碎碎念
嘿嘿。其实我很喜欢写博文，所以大家能看到 ggboy 发帖频率还是很高的。今天这个博文，每天下了班或者周末就狠狠的在撰写。
旁白：ai 时代，你手撕搞鸡毛呀。我认为是有的，不过到底有什么意义，可能需要个人去手撕一遍才能体会。
课程原文：
https://cs336.stanford.edu/spring2025/index.html

# 一、从零实现并训练一个 Transformer 语言模型

> 用 LeetCode 的方式手撕大模型：从 BPE 到 Transformer，从算法题到真实训练。
> 文章讲人话，算法练真功，代码能跑通，模型真训练。

## 学习地图：先推理，再训练

这套课程按照两个大问题展开：

```text
推理：一句 Query 怎样变成用户看到的文本？
训练：模型参数为什么能给出有用的预测？
```

每课都包含简短概念、矩阵或数值示例、核心函数、独立 case 和普通循环版参考答案。先完成推理，再进入训练。

# 第一部分：推理——Query 怎样变成用户看到的文本？

```text
Query
→ UTF-8 / BPE Tokenizer
→ Token IDs
→ Embedding
→ 多层 Transformer Blocks
→ 最终 RMSNorm
→ LM Head
→ logits
→ Softmax
→ 解码选择 token
→ Tokenizer 转回文本
→ 自回归循环直到停止
```

## A. Tokenizer：文本 → Token IDs

### 第 1 课：UTF-8 字节——为什么任何文本都能表示？

[练习：文本 → UTF-8 byte IDs](exercises/utf8_bytes/README.md)

理解 256 个基础字节怎样覆盖中文、emoji 和未见过的文本。

### 第 2 课：BPE 训练——merges 和 vocab 怎样学出来？

[练习：BPE 训练](exercises/bpe/README.md)

统计相邻 pair，按频次和 tie 规则合并，制作 Tokenizer 的合并规则和词典。

### 第 3 课：Special Tokens——特殊标记怎样保留？

[练习：切分并保留 special tokens](exercises/special_tokens/README.md)

处理重叠 special token 的最长匹配，并了解真实流式编码中的 chunk 边界。

### 第 4 课：BPE 编码——Query 怎样变成模型输入？

[练习：完整 BPE 编码](exercises/bpe_encode/README.md)

按照 merges 的学习顺序切分新文本，再查询 vocab 得到 token IDs。

## B. Transformer：Token IDs → 上下文向量矩阵

### 第 5 课：Embedding——ID 怎样变成向量？

[练习：Embedding 查表](exercises/embedding/README.md)

把 `(sequence,)` 的 IDs 转换成 `(sequence, d_model)` 的矩阵。

### 第 6 课：Linear——矩阵怎样做可训练投影？

[练习：无 bias Linear](exercises/linear/README.md)

用矩阵乘法和点积理解 Q/K/V、前馈网络与 LM Head 共有的线性变换。

### 第 7 课：RMSNorm——为什么先调整向量尺度？

[练习：RMSNorm](exercises/rmsnorm/README.md)

根据均方根归一化向量，再应用可学习 gain。

### 第 8 课：RoPE——怎样加入相对位置信息？

[练习：旋转位置编码](exercises/rope/README.md)

按维度配对旋转向量，让位置差反映到 Attention 的内积中。

### 第 9 课：Scaled Attention——怎样按相关性读取信息？

[练习：softmax(QK^T / sqrt(d_k))V](exercises/scaled_attention/README.md)

理解 Q/K/V 的来源、注意力权重以及 Value 加权和。

### 第 10 课：Causal Mask——为什么不能偷看未来？

[练习：应用因果掩码](exercises/causal_mask/README.md)

在 Softmax 前屏蔽未来位置，保证位置 `i` 只能使用 `0...i` 的信息。

### 第 11 课：Multi-Head Attention——为什么拆成多个头？

[练习：拆分 Attention heads](exercises/multihead_attention/README.md)

把 `d_model` 维向量拆成多个头，分别 Attention，再拼接和输出投影。

### 第 12 课：完整 Causal Multi-Head Attention——怎样把各头组合起来？

[综合练习：Q/K/V 投影 + RoPE + 逐头因果 Attention + 输出投影](exercises/causal_multihead/README.md)

亲手完成完整 MHA，确认 RoPE 只作用于 Q/K，未来位置不参与 Attention，最后拼接各头并应用输出投影。

### 第 13 课：SwiGLU——前馈网络怎样做门控？

[练习：SiLU 与 SwiGLU 核心计算](exercises/swiglu/README.md)

理解 `SiLU(gate) × value`，以及真实实现中的 w1、w2、w3 投影。

### 第 14 课：Transformer Block——各组件怎样组合？

[练习：Pre-Norm 与残差相加](exercises/transformer_block/README.md)

串起 `x + Attention(RMSNorm(x))` 和 `x + SwiGLU(RMSNorm(x))` 两次残差连接。

## C. 模型输出：上下文向量 → 候选 token 概率

### 第 15 课：完整 Transformer Block——Pre-Norm 顺序怎样组合？

[综合练习：完整 Pre-Norm Block](exercises/transformer_block_forward/README.md)

实际组合两次归一化、Attention、FFN 与两次残差，第二次使用第一次残差后的表示。

### 第 16 课：完整 TransformerLM——模型前向怎样组装？

[综合练习：Embedding + Blocks + 最终 RMSNorm + LM Head](exercises/transformer_lm_forward/README.md)

把所有组件按真实顺序组合，输出每个位置的 logits，并明确最终 RMSNorm 的位置。

### 第 17 课：LM Head——最后一个位置怎样变成 logits？

[练习：LM Head](exercises/lm_head/README.md)

从 `hidden_matrix` 取最后一个位置，经过最终 RMSNorm，再与词表中每个 token 的权重行做点积。

### 第 18 课：Softmax——logits 怎样变成概率？

[练习：数值稳定的 Softmax](exercises/softmax/README.md)

把任意分数转换成总和为 1 的概率分布。

## D. 解码：概率 → 用户看到的文本

### 第 19 课：贪心解码——怎样选择并还原一个 token？

[练习：单步贪心解码](exercises/greedy_decoding/README.md)

选择最大概率的 token ID，再由 Tokenizer 根据 vocab 转回文本。

### 第 20 课：Temperature——怎样调整分布的平滑程度？

[练习：Temperature Scaling + Softmax](exercises/temperature/README.md)

在 Softmax 前用温度缩放 logits：高温更平滑，低温更尖锐。

### 第 21 课：Top-p——怎样限制采样候选集合？

[练习：Top-p 筛选与重新归一化](exercises/top_p_sampling/README.md)

按概率排序，选择累计概率达到阈值的最小候选集合。

### 第 22 课：自回归生成——完整回答怎样一步步产生？

[练习：自回归贪心生成](exercises/autoregressive_generation/README.md)

把新 token 接回输入，循环预测，直到 EOS 或最大生成长度。

## 推理部分回顾

到这里，推理不再留黑盒：

```text
文本 → bytes / token IDs → 向量矩阵
→ Attention 与前馈网络逐层更新
→ logits → 概率 → token ID → 文本
```

简化列表练习与真实 PyTorch 工程的对应关系见 [A1 工程对照](A1_ENGINEERING.zh-CN.md)。

# 第二部分：训练——模型参数从哪里来？

推理使用训练好的参数；训练则通过大量“预测下一个 token”的样本，反复执行：

```text
构造输入与右移目标
→ 前向预测
→ 计算 loss
→ 反向传播得到梯度
→ 调整梯度与学习率
→ 更新参数
→ 验证并保存 checkpoint
```

## E. Loss 与反向传播：预测错误怎样变成调整方向？

### 第 23 课：交叉熵——正确 token 的概率怎样变成 loss？

[练习：Softmax 与交叉熵 Loss](exercises/language_model_loss/README.md)

理解 `-log(正确 token 的概率)`，以及为什么正确答案概率越低，惩罚越大。

### 第 24 课：梯度下降——为什么减去梯度能降低 loss？

[练习：一维梯度下降](exercises/gradient_descent/README.md)

用 `loss = (w - 3)^2` 亲手验证梯度方向、学习率和多步更新。

### 第 25 课：Softmax 交叉熵反向传播——logits 梯度怎样产生？

[练习：Softmax + Cross Entropy Backward](exercises/softmax_cross_entropy_backward/README.md)

推导并实现 `gradient = probability - one_hot(target)`，看清链式法则的结果。

### 第 26 课：Embedding Backward——重复 token 为什么累加梯度？

[练习：Embedding 梯度累加与 SGD 更新](exercises/embedding_backward/README.md)

把每个输出位置的梯度送回原权重行，理解重复 ID 累加和未使用行不变。

## F. 优化器与训练稳定性：参数怎样更新？

### 第 27 课：AdamW——真实项目怎样更新参数？

[练习：一步 AdamW](exercises/adamw/README.md)

实现一阶矩、二阶矩、bias correction 和 decoupled weight decay。

### 第 28 课：Gradient Clipping——梯度爆炸时怎么办？

[练习：按 L2 范数裁剪梯度](exercises/gradient_clipping/README.md)

计算全局梯度范数，超过阈值时等比例缩小，不改变梯度方向。

### 第 29 课：学习率调度——为什么先 warmup 再 cosine decay？

[练习：Warmup + Cosine Learning Rate](exercises/cosine_learning_rate/README.md)

让训练初期逐步增大学习率，随后按余弦曲线减小。

## G. 数据与完整训练闭环

### 第 30 课：Training Batch——语言模型样本怎样构造？

[练习：输入与右移一位的目标](exercises/training_batch/README.md)

从连续 token 序列构造 inputs 和 targets，并了解真实工程中的 uint16 与 memmap。

### 第 31 课：Tiny Language Model——训练真的能让 loss 下降吗？

[综合练习：训练一个 Bigram 语言模型](exercises/tiny_language_model/README.md)

从全零参数开始，独立完成预测、loss、梯度、平均和 SGD 更新，观察正确 token 概率上升。

### 第 32 课：Resource Accounting——模型有多大，训练状态占多少内存？

[练习：Transformer 参数量与训练状态计算](exercises/resource_accounting/README.md)

逐组件计算参数量，估算参数、梯度与 AdamW 状态内存，并理解 FLOPs 与参数量的区别。

## H. 从算法练习到真实 A1 工程

完成核心算法后，还需要理解这些工程环节：

- **Checkpoint**：同时保存模型、优化器状态和训练步数，才能完整恢复训练。
- **Validation**：使用不参与参数更新的数据评估泛化，不能只看 training loss。
- **Mini-batch 与 memmap**：按需读取大批数据，控制内存和计算成本。
- **混合精度与 GPU**：真实训练使用 Tensor 和 CUDA/bf16，而不是 Python 列表。
- **流式 Tokenizer**：处理数 GB 文本和任意 chunk 边界，不能一次性载入全部语料。

这些内容与本仓库实际文件的对应关系见 [CS336 A1：简化练习与真实工程对照](A1_ENGINEERING.zh-CN.md)。

## 真实训练结果

本项目已经在 TinyStories 上完成真实训练：

```text
模型：约 22.7M 参数
硬件：单张 NVIDIA A100 40GB
训练：10,000 步，327.68M tokens
耗时：约 28.2 分钟
独立 100-batch validation loss：1.38567
Validation perplexity：3.9975
```

实验配置、曲线和生成结果见 [实验报告](RESULTS.zh-CN.md)，复现步骤见 [运行指南](RUNNING.zh-CN.md)。

到这里，课程覆盖了 A1 的 Tokenizer、Transformer、训练、解码和实验落地。可以继续阅读 `cs336_basics/` 中的真实实现，把每道简化练习对应回完整模型。
