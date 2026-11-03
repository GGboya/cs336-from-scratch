# 用 LeetCode 的方式手撕大模型

> 从 BPE 到 Transformer，从算法题到真实训练。

**文章讲人话，算法练真功，代码能跑通，模型真训练。**

这是一个面向传统程序员的语言模型算法练习与全流程实践项目，也是 **Stanford CS336 Spring 2025 Assignment 1: Basics** 的非官方实现。

我们希望通过一条渐进路线，让有编程基础、第一次接触 LLM 的读者真正理解每个组件：

```text
通俗文章 → 简化算法题 → 独立测试 → 易读参考答案 → 真实工程实现 → GPU 训练 → 本地推理
```

原始作业脚手架、测试和讲义由 Stanford University 基于 MIT License 提供，详见 [LICENSE](LICENSE)。本仓库不是 Stanford 官方仓库。

## 按“Query → 模型 → 输出”学习与练习

```text
Query → Tokenizer（BPE）→ Token IDs → 模型
→ 下一个 token 的概率 → 解码 → 输出文本
```

先理解输入和输出，再逐层打开模型内部；训练细节放在生成流程之后继续展开。

每道题的说明、输入输出和示例都放在对应的 `README.md` 中。读者只需要实现 `solution.py` 里的核心函数，测试由项目提供；参考答案使用普通循环，重点展示逻辑而不是 Python 技巧。

### 1. Tokenizer 与 BPE：文本 → Token IDs

1. [BPE 训练：学习 merges 和 vocab](exercises/bpe/README.md)
2. [完整 BPE 编码：文本 → token IDs](exercises/bpe_encode/README.md)

### 2. Embedding：Token IDs → 向量

[Embedding 查表：token IDs → 向量](exercises/embedding/README.md)

### 3. 模型训练：预测误差 → loss → 反向传播与权重更新

1. [Softmax 与交叉熵：把预测错误转换成 loss](exercises/language_model_loss/README.md)
2. [Embedding 反向传播：梯度累加与权重更新](exercises/embedding_backward/README.md)

先用正确 token 的概率计算 loss，再通过反向传播得到梯度，最后更新模型参数。推理时只需要使用训练好的参数，不计算 loss，也不更新权重。

运行全部现有练习：

```bash
.venv/bin/python exercises/run.py bpe
.venv/bin/python exercises/run.py bpe_encode
.venv/bin/python exercises/run.py embedding
.venv/bin/python exercises/run.py embedding_backward
.venv/bin/python exercises/run.py language_model_loss
```

配套文章：[用 LeetCode 的方式手撕大模型](share.md)。文章按照同一流程展开，每个组件介绍完后都会附上对应练习。

## 已实现功能

- Byte-level BPE：多进程预分词、增量 merge 统计
- 支持 UTF-8、special tokens 和任意 chunk 边界的流式 Tokenizer
- Linear、Embedding、RMSNorm、SwiGLU、RoPE、因果多头注意力
- Pre-norm Transformer 语言模型
- 数值稳定的 softmax 和交叉熵
- AdamW、cosine 学习率调度、梯度裁剪
- memmap 数据加载、checkpoint、验证集评估、JSONL 训练日志
- Temperature 和 top-p 文本生成
- 支持 CPU、Apple MPS、CUDA/bf16

核心神经网络计算使用 PyTorch tensor 算子实现，没有使用 `nn.Linear`、`nn.Embedding`、`torch.nn.functional` 或 `torch.optim.AdamW` 等高层现成实现替代作业要求。

## 训练结果

使用单张 NVIDIA A100 40GB，将约 22.7M 参数的模型训练 10,000 步，共计 327.68M tokens：

| 指标 | 结果 |
|---|---:|
| 训练耗时 | 28.2 分钟 |
| 独立 100-batch validation loss | **1.38567** |
| Validation perplexity | **3.9975** |
| 峰值训练显存 | 约 11.6GiB |

Linux 测试结果：**59 passed、1 xpassed**。完整配置、验证曲线和生成样例见[实验报告](https://github.com/GGboya/cs336-from-scratch/blob/main/RESULTS.zh-CN.md)。

## 快速开始

需要 Python 3.11 和 [uv](https://docs.astral.sh/uv/)：

```bash
uv sync --python 3.11
uv run pytest -q
```

下载并处理 TinyStories：

```bash
scripts/download_tinystories.sh
PYTHON_BIN="$PWD/.venv/bin/python" scripts/prepare_tinystories.sh
```

运行 20 步 CUDA smoke test：

```bash
RUN_DIR="$PWD/runs/cloud_smoke" scripts/train_tinystories_baseline.sh \
  --max-iters 20 --batch-size 8 --warmup-iters 2 \
  --log-interval 5 --eval-interval 20 --eval-batches 2 \
  --checkpoint-interval 20
```

启动完整 baseline 训练：

```bash
scripts/train_tinystories_baseline.sh
```

## 本地生成故事

数据集和训练后的 checkpoint 不纳入 Git。完成训练，或者将兼容的 checkpoint 放到 `runs/tinystories_baseline/` 后，运行：

```bash
scripts/story.sh
```

看到提示后输入英文故事开头：

```text
Story opening> Once upon a time, a little rabbit found a red balloon.
```

模型会在终端中续写故事。交互模式、单次生成和参数说明见[本地使用文档](https://github.com/GGboya/cs336-from-scratch/blob/main/LOCAL_USAGE.zh-CN.md)。

## 目录结构

```text
cs336_basics/       Tokenizer、模型、训练和解码实现
scripts/            数据处理、训练、评估、云端和本地推理脚本
tests/              官方作业测试及补充回归测试
RUNNING.zh-CN.md    复现指南
RESULTS.zh-CN.md    训练配置与实验结果
LOCAL_USAGE.zh-CN.md 本地推理使用说明
```

## 数据与模型文件

仓库不包含 TinyStories 数据集和训练后的模型权重。可以通过脚本下载、预处理并训练。默认本地路径为：

```text
data/tinystories_10k/
data/tinystories_train.bin
data/tinystories_valid.bin
runs/tinystories_baseline/checkpoint.pt
```

## License 与版权

MIT License。原始作业材料 Copyright 2025 Stanford University；后续实现与修改 Copyright 2026 cs336-from-scratch contributors。
