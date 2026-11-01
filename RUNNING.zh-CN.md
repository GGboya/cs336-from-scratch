# CS336 Assignment 1：运行与云端复现

## 环境

推荐 Python 3.11 和 uv：

```bash
uv sync --python 3.11
uv run pytest -q
```

已在 Apple MPS 和 NVIDIA A100/CUDA 上验证。Linux 云端测试结果为 59 passed、1 xpassed。

## 数据与 tokenizer

原始数据放在 `data/raw/`。编码结果为 little-endian `uint16` token IDs，训练时通过 `np.memmap` 读取。

```bash
# 默认通过 hf-mirror.com 下载，可修改 HF_ENDPOINT
scripts/download_tinystories.sh

# 训练 10K BPE 并编码训练/验证集
PYTHON_BIN="$PWD/.venv/bin/python" scripts/prepare_tinystories.sh

# 校验 token 数量、ID 范围及首尾解码一致性
.venv/bin/python scripts/verify_tinystories.py
```

生成文件：

```text
data/tinystories_10k/vocab.json
data/tinystories_10k/merges.txt
data/tinystories_train.bin
data/tinystories_valid.bin
```

本次实测：全量 BPE 使用 4 进程训练 107.48 秒，峰值进程内存约 1.65GiB；训练集 541,229,347 tokens，验证集 5,465,883 tokens。

## 本机小规模训练

```bash
.venv/bin/python -m cs336_basics.train \
  --train-data data/smoke_train.bin \
  --valid-data data/smoke_valid.bin \
  --output-dir runs/smoke \
  --device mps \
  --vocab-size 1000 --context-length 64 \
  --d-model 64 --num-layers 2 --num-heads 4 --d-ff 128 \
  --batch-size 16 --max-iters 100 --warmup-iters 10 \
  --max-lr 0.001 --min-lr 0.0001 \
  --log-interval 20 --eval-interval 50 --eval-batches 4 \
  --checkpoint-interval 50
```

训练输出包括 `config.json`、`metrics.jsonl` 和 `checkpoint.pt`。恢复训练时添加 `--resume runs/smoke/checkpoint.pt` 并调整 `--max-iters`。

## CUDA smoke test 与正式训练

```bash
# 检查 CUDA 和 bf16
.venv/bin/python scripts/check_gpu.py

# 20 步 smoke test
RUN_DIR="$PWD/runs/cloud_smoke" scripts/train_tinystories_baseline.sh \
  --max-iters 20 --batch-size 8 --warmup-iters 2 \
  --log-interval 5 --eval-interval 20 --eval-batches 2 \
  --checkpoint-interval 20

# 在 tmux 中启动正式训练
scripts/start_cloud_training.sh
```

正式配置为 10,000 步、batch 128、context 256，总计 327,680,000 tokens。CUDA 下启用 bf16 autocast。训练控制台输出保存在 `runs/tinystories_baseline/console.log`。

重新进入 tmux：

```bash
tmux attach -t cs336
```

## 最终评估与生成

```bash
# 独立 100-batch 验证评估
.venv/bin/python scripts/evaluate_baseline.py \
  --run-dir runs/tinystories_baseline \
  --valid-data data/tinystories_valid.bin \
  --eval-batches 100 --device cuda

# 本地交互生成（也支持 MPS/CPU）
scripts/story.sh
```

训练结束后应下载整个 `runs/tinystories_baseline/`，保留配置、日志、checkpoint 和生成样例，再关闭付费实例。
