# CS336 Assignment 1：TinyStories Baseline 结果

## 结论

已于 2026-11-01 完成完整 baseline 训练。最终 checkpoint 为第 10,000 步，达到作业要求的 validation loss ≤ 1.45。

| 指标 | 结果 |
|---|---:|
| 总训练步数 | 10,000 |
| 总训练 tokens | 327,680,000 |
| 训练耗时 | 1,691.49 秒（约 28.2 分钟） |
| 最终 20-batch validation loss | 1.38058 |
| 独立 100-batch validation loss | **1.38567** |
| 独立 100-batch validation perplexity | **3.9975** |
| 峰值 GPU 显存（训练日志） | 约 11.6GiB |

## 配置

- GPU：矩池云单卡 NVIDIA A100-PCIE-40GB。
- PyTorch 2.6.0 + CUDA 12.4，bf16 autocast。
- 参数总量：22,696,448。
- vocab 10,000；context 256；d_model 512。
- 4 层；16 heads；d_ff 1,344；RoPE theta 10,000。
- batch 128；最大学习率 1e-3；最小学习率 1e-4。
- warmup 500 步；cosine schedule 终点 10,000 步。
- AdamW betas `(0.9, 0.95)`；weight decay 0.1；gradient clip 1.0。

## Validation loss 曲线

| Step | Loss | Perplexity |
|---:|---:|---:|
| 500 | 2.29196 | 9.8943 |
| 1,000 | 1.90580 | 6.7248 |
| 1,500 | 1.77232 | 5.8845 |
| 2,000 | 1.68194 | 5.3760 |
| 2,500 | 1.64410 | 5.1764 |
| 3,000 | 1.60687 | 4.9872 |
| 3,500 | 1.57697 | 4.8403 |
| 4,000 | 1.54447 | 4.6855 |
| 4,500 | 1.52098 | 4.5767 |
| 5,000 | 1.49848 | 4.4749 |
| 5,500 | 1.47529 | 4.3723 |
| 6,000 | 1.46648 | 4.3340 |
| 6,500 | 1.44849 | 4.2567 |
| 7,000 | 1.43280 | 4.1904 |
| 7,500 | 1.40752 | 4.0858 |
| 8,000 | 1.40856 | 4.0901 |
| 8,500 | 1.39426 | 4.0320 |
| 9,000 | 1.37068 | 3.9380 |
| 9,500 | 1.38928 | 4.0119 |
| 10,000 | 1.38058 | 3.9772 |

每 500 步的指标使用 20 个 validation batches，因此相邻点存在采样波动。最终另用固定随机种子和 100 个 batches 做独立评估，结果见 `final_evaluation.json`。

## 生成样例

Prompt：`Once upon a time`；temperature 0.8；top-p 0.9；最多生成 256 tokens。

```text
Once upon a time, there was a big, ugly fish named Bob. Bob lived in a reef with his friends. The reef was full of pretty fish, colorful fish, and big fish. Bob was sad because he had no friends.
One day, a little fish named Tim came to the reef. Tim saw Bob and said, "Hi, I'm Tim. I'm sorry. Can I play with you?" Bob was surprised. He had never talked to Tim before.
Tim said, "Yes, let's play together!" They swam and played in the reef. Bob was not sad anymore. He was happy to have a new friend. From that day on, Bob and Tim were best friends, and they played together every day.
<|endoftext|>
```

模型能够生成结构完整、语法基本正确的儿童故事，但仍存在人物关系或叙述逻辑不完全一致等小模型常见问题。

## 本地文件

已从云端下载到本地：

```text
runs/tinystories_baseline/
├── checkpoint.pt          # 约 260MiB，模型、优化器和 iteration
├── config.json
├── metrics.jsonl
├── console.log
├── final_evaluation.json
└── generated_sample.txt
```

## 本地 MPS 推理

已使用下载后的 checkpoint 在 Mac MPS 上实际生成文本，验证通过。

```bash
scripts/story.sh --prompt "Once upon a time" \
  --max-new-tokens 180 --temperature 0.8 --top-p 0.9
```
