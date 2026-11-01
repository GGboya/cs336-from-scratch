# CS336 from Scratch

An unofficial, educational implementation of **Stanford CS336 Spring 2025 Assignment 1: Basics**. The project builds a byte-level BPE tokenizer and a Transformer language model from scratch, trains it on TinyStories, and supports local text generation.

The original assignment scaffold, tests, and handout are provided by Stanford University under the MIT License. See [LICENSE](LICENSE).

## Implemented

- Byte-level BPE training with parallel pretokenization and incremental merge statistics
- Streaming tokenizer with UTF-8 and special-token support
- Linear, Embedding, RMSNorm, SwiGLU, RoPE, causal multi-head attention
- Pre-norm Transformer language model
- Numerically stable softmax and cross-entropy
- AdamW, cosine learning-rate schedule, and gradient clipping
- Memory-mapped data loading, checkpoints, validation, and JSONL metrics
- Temperature and top-p decoding
- CPU, Apple MPS, and CUDA/bf16 execution

Core neural-network operations are implemented with PyTorch tensor operators rather than high-level `nn.Linear`, `nn.Embedding`, `torch.nn.functional`, or `torch.optim.AdamW` shortcuts.

## Verified result

A 22.7M-parameter model was trained for 10,000 steps (327.68M tokens) on a single NVIDIA A100 40GB:

| Metric | Result |
|---|---:|
| Training time | 28.2 minutes |
| Independent 100-batch validation loss | **1.38567** |
| Validation perplexity | **3.9975** |
| Peak training memory | approximately 11.6GiB |

Linux test result: **59 passed, 1 xpassed**. See [the experiment report](RESULTS.zh-CN.md) for configuration, validation curves, and a generated sample.

## Quick start

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --python 3.11
uv run pytest -q
```

Download and prepare TinyStories:

```bash
scripts/download_tinystories.sh
PYTHON_BIN="$PWD/.venv/bin/python" scripts/prepare_tinystories.sh
```

Run a short CUDA smoke test:

```bash
RUN_DIR="$PWD/runs/cloud_smoke" scripts/train_tinystories_baseline.sh \
  --max-iters 20 --batch-size 8 --warmup-iters 2 \
  --log-interval 5 --eval-interval 20 --eval-batches 2 \
  --checkpoint-interval 20
```

Train the full baseline:

```bash
scripts/train_tinystories_baseline.sh
```

## Local story generation

Datasets and trained checkpoints are intentionally not tracked by Git. After training, or after placing a compatible checkpoint in `runs/tinystories_baseline/`, run:

```bash
scripts/story.sh
```

Enter an English story opening at the prompt:

```text
Story opening> Once upon a time, a little rabbit found a red balloon.
```

See [LOCAL_USAGE.zh-CN.md](LOCAL_USAGE.zh-CN.md) for interactive and one-shot generation options.

## Repository layout

```text
cs336_basics/       Tokenizer, model, training, and decoding implementations
scripts/            Data, training, evaluation, cloud, and local inference tools
tests/              Official assignment tests and additional regression tests
RUNNING.zh-CN.md    Reproduction guide
RESULTS.zh-CN.md    Training configuration and experiment results
```

## Data and artifacts

The repository does not include the TinyStories dataset or trained model weights. Download and preprocessing scripts are provided. Expected local paths are:

```text
data/tinystories_10k/
data/tinystories_train.bin
data/tinystories_valid.bin
runs/tinystories_baseline/checkpoint.pt
```

## License and attribution

MIT License. Original assignment materials are Copyright 2025 Stanford University. This is an independent educational implementation, not an official Stanford repository.
