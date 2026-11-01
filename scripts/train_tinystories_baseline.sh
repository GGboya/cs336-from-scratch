#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/runs/tinystories_baseline}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
cd "$ROOT_DIR"

$PYTHON_BIN -m cs336_basics.train \
  --train-data "$ROOT_DIR/data/tinystories_train.bin" \
  --valid-data "$ROOT_DIR/data/tinystories_valid.bin" \
  --output-dir "$RUN_DIR" \
  --device cuda \
  --amp \
  --vocab-size 10000 \
  --context-length 256 \
  --d-model 512 \
  --num-layers 4 \
  --num-heads 16 \
  --d-ff 1344 \
  --batch-size 128 \
  --max-iters 10000 \
  --warmup-iters 500 \
  --max-lr 0.001 \
  --min-lr 0.0001 \
  --beta1 0.9 \
  --beta2 0.95 \
  --weight-decay 0.1 \
  --grad-clip 1.0 \
  --log-interval 50 \
  --eval-interval 500 \
  --eval-batches 20 \
  --checkpoint-interval 500 \
  "$@"
