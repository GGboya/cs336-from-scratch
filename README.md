# 用 LeetCode 的方式手撕大模型

> 从 BPE 到 Transformer，从算法题到真实训练。

**文章讲人话，算法练真功，代码能跑通，模型真训练。**

这是一个面向传统程序员的语言模型算法练习与全流程实践项目，也是 **Stanford CS336 Spring 2025 Assignment 1: Basics** 的非官方实现。

我们希望通过一条渐进路线，让有编程基础、第一次接触 LLM 的读者真正理解每个组件：

```text
通俗文章 → 简化算法题 → 独立测试 → 易读参考答案 → 真实工程实现 → GPU 训练 → 本地推理
```

原始作业脚手架、测试和讲义由 Stanford University 基于 MIT License 提供，详见 [LICENSE](LICENSE)。本仓库不是 Stanford 官方仓库。

## 学习路线：先推理，再训练

这套课程用 **32 道 LeetCode 式练习**覆盖 CS336 Assignment 1 的核心算法，并通过真实工程对照、GPU 训练和本地推理完成落地。

每道题的 `README.md` 都包含简短概念、输入输出和手算示例；读者只实现 `solution.py`，测试由项目提供，`reference.py` 使用普通循环展示参考答案。

完整学习地图和逐课讲解见 [share.md](share.md)。

### 第一部分：推理——Query → 模型 → 输出

| 模块 | 练习 |
|---|---|
| Tokenizer | [UTF-8](exercises/utf8_bytes/README.md)、[BPE 训练](exercises/bpe/README.md)、[Special Tokens](exercises/special_tokens/README.md)、[BPE 编码](exercises/bpe_encode/README.md) |
| Transformer 基础 | [Embedding](exercises/embedding/README.md)、[Linear](exercises/linear/README.md)、[RMSNorm](exercises/rmsnorm/README.md)、[RoPE](exercises/rope/README.md) |
| Attention | [Scaled Attention](exercises/scaled_attention/README.md)、[Causal Mask](exercises/causal_mask/README.md)、[Multi-Head](exercises/multihead_attention/README.md)、[完整 Causal MHA](exercises/causal_multihead/README.md) |
| Block 与输出 | [SwiGLU](exercises/swiglu/README.md)、[Residual / Pre-Norm](exercises/transformer_block/README.md)、[完整 Block](exercises/transformer_block_forward/README.md)、[完整 TransformerLM](exercises/transformer_lm_forward/README.md)、[LM Head](exercises/lm_head/README.md)、[Softmax](exercises/softmax/README.md) |
| 解码 | [Greedy Decoding](exercises/greedy_decoding/README.md)、[Temperature](exercises/temperature/README.md)、[Top-p](exercises/top_p_sampling/README.md)、[自回归生成](exercises/autoregressive_generation/README.md) |

### 第二部分：训练——预测误差 → loss → 参数更新

| 模块 | 练习 |
|---|---|
| Loss 与反向传播 | [交叉熵](exercises/language_model_loss/README.md)、[一维梯度下降](exercises/gradient_descent/README.md)、[Softmax CE Backward](exercises/softmax_cross_entropy_backward/README.md)、[Embedding Backward](exercises/embedding_backward/README.md) |
| 优化与稳定性 | [AdamW](exercises/adamw/README.md)、[Gradient Clipping](exercises/gradient_clipping/README.md)、[Warmup + Cosine LR](exercises/cosine_learning_rate/README.md) |
| 数据与综合训练 | [Training Batch](exercises/training_batch/README.md)、[Tiny Language Model](exercises/tiny_language_model/README.md)、[Resource Accounting](exercises/resource_accounting/README.md) |

简化练习与 memmap、checkpoint、混合精度、验证集和流式编码等真实工程的对应关系见 [A1 工程对照](A1_ENGINEERING.zh-CN.md)。

### 运行练习

以 BPE 为例：

```bash
.venv/bin/python exercises/run.py bpe
```

每道题的 README 都给出对应命令。开发时可以验证全部参考答案：

```bash
.venv/bin/python exercises/verify_references.py
```

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
