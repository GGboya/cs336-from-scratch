# 碎碎念
嘿嘿。其实我很喜欢写博文，所以大家能看到 ggboy 发帖频率还是很高的。今天这个博文，每天下了班或者周末就狠狠的在撰写。
旁白：ai 时代，你手撕搞鸡毛呀。我认为是有的，不过到底有什么意义，可能需要个人去手撕一遍才能体会。
课程原文：
https://cs336.stanford.edu/spring2025/index.html

# 一、从零实现并训练一个 Transformer 语言模型

> 用 LeetCode 的方式手撕大模型：从 BPE 到 Transformer，从算法题到真实训练。
> 文章讲人话，算法练真功，代码能跑通，模型真训练。

## 先从一次提问看起：Query → 模型 → 输出

假设我们给模型输入一句 query：

```text
Once upon a time
```

它最后为什么会接着输出一个故事？我们先不讨论模型是怎样训练的，只看一句输入是怎样变成输出的。

### 第一步：Query 仍然只是文本

此时，输入还是一个普通字符串。模型不能直接拿这段文字做数学计算，需要先把它转换成数值输入。

### 第二步：Tokenizer 把文本转换成 token IDs

Tokenizer 先把文本切成一个个 token，再根据词典给每个 token 一个固定编号：

```text
"Once upon a time"
→ ["Once", " upon", " a", " time"]
→ [430, 439, 259, 398]
```

所以，**真正送进模型的不是原始字符串，而是一串 token IDs。** 至于文本为什么这样切、编号从哪里来，就是后面 BPE 章节要解决的问题。

### 第三步：模型接收 token IDs，预测下一个 token

我们现在可以先把模型当成一个黑盒：

```text
输入：[430, 439, 259, 398]
       ↓
     模型
       ↓
输出：词表中每个候选 token 成为下一个 token 的概率
```

例如，模型可能认为下一个 token 是 `","` 的概率最高。

这个黑盒内部怎样从 token IDs 算出这些概率，我们后面再逐层拆开：先经过 Embedding，再经过 Transformer Blocks，最后经过 LM Head。

### 第四步：把选中的 token 转换回文本

解码阶段会根据概率选出一个 token，再由 Tokenizer 把它转换回文本片段，接到原输入后面：

```text
Once upon a time,
```

然后把更长的输入重新交给模型，继续预测下一个 token：

```text
Once upon a time, there
Once upon a time, there was
Once upon a time, there was a
```

所以，大模型生成回答的基本方式就是：**一次预测一个 token，不断重复。**

整个过程可以先记成：

```text
Query
→ Tokenizer（BPE）
→ Token IDs
→ 模型
→ 下一个 token 的概率
→ 解码
→ 输出文本
```

接下来，我们先解决输入侧的问题：文本怎样经过 BPE 变成 token IDs。讲完 BPE，再打开模型这个黑盒，慢慢看清 Embedding、Transformer 和 LM Head 分别做了什么。理解这条生成流程之后，再讨论模型是怎样通过训练学会这些能力的。

## 一、Tokenizer（BPE）：文本 → Token IDs
文本 → Tokenizer → Token IDs
计算机并不懂咱们的文字，需要把常见片段切成 token，再编成数字。
"Once upon a time"
  → ["Once", " upon", " a", " time"]
  → [430, 439, 259, 398]

这里引入了一个新问题，文本是怎么被切成 token 的呢？

---

答案是：**我们使用了一种叫 BPE（Byte Pair Encoding，字节对编码）的算法。**

BPE 会做两件事：

1. **训练 Tokenizer 时**：从最小的字节片段开始，不断统计相邻片段的出现频次，把高频 pair 合并成新 token，逐渐形成一本“片段词典”。
2. **处理新文本时**：按照已经学到的合并规则，把文本切成 token，再查询词典，得到每个 token 对应的数字 ID。

举个极简例子。假设训练语料只有：

```text
he、he、her
```

训练 Tokenizer 时，先统计相邻片段：

```text
(h, e) 出现 3 次
(e, r) 出现 1 次
```

`(h, e)` 的频次最高，因此把它合并成 `he`。训练结束后，我们得到：

```text
合并规则：(h, e) → he

词典：
h  → 0
e  → 1
r  → 2
he → 3
```

接下来处理新文本 `her`。先按照合并规则切分：

```text
[h, e, r]
→ [he, r]
```

再查询词典：

```text
he → 3
r  → 2
```

最终得到：

```text
"her" → [3, 2]
```

这就是两个阶段的关系：**训练时制作合并规则和词典，处理新文本时使用它们。**

因此：

```text
文本是怎样被切成 token 的？
→ 由 BPE 学到的 merge 规则决定。

token 的数字从哪里来？
→ 由训练得到的 vocab 词典决定。
```

### 有了词典，怎么把文字变成数字？

Tokenizer 训练完成后，我们会保存两样东西：

```text
vocab.json   # 每个 token 对应哪个数字
merges.txt   # 片段应该按照什么顺序合并
```

处理新文本时，先按照 `merges.txt` 中的规则进行切分，再查询 `vocab.json`：

```text
"Once upon a time"
→ ["Once", " upon", " a", " time"]
→ [430, 439, 259, 398]
```

这里的数字不是随机分配的，也不是根据单词含义计算出来的。它们只是 token 在词典里的固定编号，类似字典中的页码。

### 为什么不是简单按照单词切分？

如果每个单词只对应一个 token，就会遇到几个问题：

- 新名字、拼写错误和新单词可能不在词典里；
- 词表需要覆盖大量单词，规模会越来越大；
- `cat` 和 `cats` 会被看成完全无关的两个 token。

BPE 则提供了一种折中：常见片段可以合并成一个 token，不常见的内容则继续拆成更小的片段。

例如：

```text
"i love ggboy"
→ ["i", " love", " g", "g", "b", "oy"]
```

`love` 是常见片段，所以只需要一个 token；`ggboy` 不是常见片段，只能由多个 token 拼出来。

因此，**token 不一定是一个完整单词，也不一定只是一个字符，而是 Tokenizer 学到的文本片段。**

我们这次使用了约 2.23GB 的 TinyStories 文本，从 256 个基础字节和 1 个特殊 token 开始，经过 9,743 次合并，最终得到一本包含 10,000 个 token 的“片段词典”。

这里的“训练”不是神经网络中的反向传播，而是**统计文本并学习合并规则**。

### 真正的 Tokenizer 还需要处理什么？

上面介绍的是 BPE 的核心思想。要把它变成能够处理真实语料的 Tokenizer，还需要处理：

- 中文、emoji 等 Unicode 文本；
- 标点、空格和特殊标记；
- 数 GB 语料的统计性能；
- 大文件的流式编码；
- 如何在不重新扫描全部语料的情况下更新 pair 频次。

这些不属于 BPE 的核心算法，却是处理大规模文本时不可缺少的工程细节。

### 动手练习：Tokenizer 与 BPE

如果想亲手实现这些过程，我准备了一组 LeetCode 式算法题。题目、输入输出和示例都放在各自文件中，只需要实现核心函数：

1. [BPE 训练：学习 merges 和 vocab](exercises/bpe/README.md)
2. [完整 BPE 编码：文本 → token IDs](exercises/bpe_encode/README.md)

分别运行测试：

```bash
.venv/bin/python exercises/run.py bpe
.venv/bin/python exercises/run.py bpe_encode
```

## 二、Embedding：Token IDs → 向量

现在我们已经知道，一句话是怎样变成 token IDs 的：

```text
"i love you"
→ ["i", " love", " you"]
→ [106, 867, 349]
```

但这三个数字只是词典编号，不能直接表达词语的特征。比如，ID 867 并不比 ID 106 更重要，它们之间的大小和距离也没有实际语义。

下一步，就是根据这些编号，到 **Embedding 权重表**中取出对应的向量：

```text
106 → Embedding 表第 106 行
867 → Embedding 表第 867 行
349 → Embedding 表第 349 行
```

一个向量里具体放多少个数，是模型设计时选择的超参数，叫作向量维度，并不是由 token ID 计算出来的。在我们的模型里，这个维度被设置为 512，也就是每个 token 都用一个 512 维向量表示：

```text
[106, 867, 349]
        ↓ Embedding
[
  [0.12, -0.37, ..., 0.08],
  [0.45,  0.03, ..., 0.21],
  [0.06, -0.18, ..., 0.33],
]
```

形状变化是：

```text
(3,) → (3, 512)
```

如果输入还包含 batch 和 sequence 维度，则是：

```text
(batch, sequence) → (batch, sequence, d_model)
```

所以，Embedding 本质上就是一次查表：**token ID 是下标，weight 是向量表，输出按照 ID 顺序排列的向量。**

### Embedding 权重表是哪里来的？

这张表不是 Tokenizer 提供的，而是语言模型自己创建并通过训练学出来的。

模型初始化时，我们先随机创建一张表。在我们的模型中：

```text
词表大小：10,000
每个向量的维度：512
权重表形状：(10000, 512)
```

对应代码在 `cs336_basics/model.py:26`：

```python
self.weight = nn.Parameter(
    torch.empty(vocab_size, d_model)
)

nn.init.trunc_normal_(
    self.weight,
    mean=0.0,
    std=1.0,
    a=-3.0,
    b=3.0,
)
```

初始时，这些向量只是随机数，并没有什么语义。

现在先记住：这张表由模型随机创建，里面的数值会在训练时不断更新。至于预测错误怎样变成梯度、梯度又怎样更新表中的行，我们放到第六章按顺序展开。

训练结束后，这张表会保存在 `checkpoint.pt` 的 `token_embeddings.weight` 中。推理时重新加载，再执行查表：

```python
self.weight[token_ids]
```

因此，Tokenizer 和 Embedding 学习的是两种不同东西：

> **Tokenizer 学的是“怎么切分文本”；Embedding 学的是“每个 token 用什么向量表示”。**

### 动手练习：Embedding 查表

[Embedding 查表：token IDs → 向量](exercises/embedding/README.md)

运行测试：

```bash
.venv/bin/python exercises/run.py embedding
```

## 三、Transformer Blocks：结合上下文更新向量

Embedding 之后，向量会经过多层 Transformer Blocks。这里会结合上下文更新每个位置的表示，让模型不仅看到当前 token，也能利用它周围的信息。

这一部分还可以继续拆成 Attention、RMSNorm、前馈网络等组件，我们后面再逐层介绍。

## 四、LM Head 与 Softmax：向量 → 下一个 token 的概率

LM Head 把向量转换成词表中每个 token 的分数，也就是 logits。比如词表中有 10,000 个 token，就会输出 10,000 个分数：

```text
上下文向量
→ LM Head
→ 每个 token 的 logits
→ Softmax
→ 每个 token 成为下一个 token 的概率
```

这些分数还不是概率，Softmax 会把它们转换成总和为 1 的概率分布。

## 五、解码：下一个 token 的概率 → 输出文本

模型会根据概率分布选出一个 token，把它接在输入后面，再重复整个过程，直到生成结束。推理时只需要使用训练好的参数，不需要更新模型。

## 六、训练：模型怎样学会给出正确的概率？

前面讲的是模型怎样生成文本。接下来才讨论训练：模型里的参数并不是人工指定的，而是通过大量“预测下一个 token”的练习学出来的。

训练数据里有正确的下一个 token，因此我们可以检查模型给它分配了多少概率。

例如，正确答案是 `time`：

```text
P(time) = 0.01 → loss ≈ 4.605
P(time) = 0.80 → loss ≈ 0.223
```

交叉熵 loss 的计算是：

```text
loss = -log(正确 token 的概率)
```

所以，正确答案的概率越高，loss 越小。训练的目标，就是通过调整参数来降低这个 loss。

### 动手练习：Softmax 与交叉熵

[Softmax 与交叉熵：把预测错误转换成 loss](exercises/language_model_loss/README.md)

运行测试：

```bash
.venv/bin/python exercises/run.py language_model_loss
```

### 反向传播：loss 怎样变成参数的修改方向？

loss 只是一个数，它告诉我们这次预测有多差，却没有直接告诉我们每个参数应该怎样修改。

反向传播会沿着前面的计算过程反向走一遍，利用链式法则计算：**如果某个参数发生一点变化，loss 会怎样变化？** 计算结果就是参数的梯度。

```text
前向：Token IDs → Embedding → Transformer → LM Head → 概率 → loss
反向：loss → 各层输出的梯度 → 各层参数的梯度
```

有了梯度，优化器就可以更新参数。用最简单的 SGD 表示就是：

```text
新参数 = 旧参数 - 学习率 × 梯度
```

我们的真实项目使用 AdamW，它是在这个基本思路上进一步改进的优化器。

### 以 Embedding 为例：查过的行怎样被训练？

假设一次输入是：

```python
token_ids = [2, 0, 2]
```

前向查表时，依次取出第 2、0、2 行。这些向量继续经过后面的网络，最终得到预测概率并计算出 loss。

随后，反向传播会把梯度传回这三个输出位置。假设得到：

```python
grad_output = [
    [1.0, 1.0],  # 第一个位置，对应 token 2
    [0.5, 0.5],  # 第二个位置，对应 token 0
    [1.0, 1.0],  # 第三个位置，对应 token 2
]
```

这些梯度不是人工随便指定的，而是从 loss 反向计算出来的。算法练习会直接给出它们，让我们专注于 Embedding 这一层怎样更新。

由于 token 2 出现了两次，它对 loss 的两份影响都要累加到第 2 行：

```text
token 0 的梯度：[0.5, 0.5]
token 2 的梯度：[1.0, 1.0] + [1.0, 1.0] = [2.0, 2.0]
token 1 没有被使用，不需要更新
```

假设学习率为 0.1，使用 SGD 更新：

```text
token 0：[0.1, 0.2] - 0.1 × [0.5, 0.5] = [0.05, 0.15]
token 2：[0.5, 0.6] - 0.1 × [2.0, 2.0] = [0.3, 0.4]
```

所以，Embedding 的训练并不只是“查表”。完整过程是：

```text
根据 token ID 查表
→ 使用这些向量预测下一个 token
→ 计算 loss
→ 反向传播得到梯度
→ 按 token ID 累加梯度
→ 更新被查到的权重行
```

经过大量文本和许多轮更新，这些向量会逐渐变成适合语言建模的特征表示。语义不是人工填进去的，而是从“预测下一个 token”的任务中学出来的。

### 动手练习：Embedding 反向传播与权重更新

[Embedding 反向传播：梯度累加与权重更新](exercises/embedding_backward/README.md)

运行测试：

```bash
.venv/bin/python exercises/run.py embedding_backward
```
