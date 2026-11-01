#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_TEXT="$ROOT_DIR/data/raw/tinystories_train/TinyStoriesV2-GPT4-train.txt"
VALID_TEXT="$ROOT_DIR/data/raw/tinystories_valid/TinyStoriesV2-GPT4-valid.txt"
TOKENIZER_DIR="$ROOT_DIR/data/tinystories_10k"

cd "$ROOT_DIR"
"$PYTHON_BIN" -m cs336_basics.train_tokenizer \
  --input "$TRAIN_TEXT" \
  --output-dir "$TOKENIZER_DIR" \
  --vocab-size 10000
"$PYTHON_BIN" -m cs336_basics.encode_dataset \
  --input "$TRAIN_TEXT" \
  --output "$ROOT_DIR/data/tinystories_train.bin" \
  --vocab "$TOKENIZER_DIR/vocab.json" \
  --merges "$TOKENIZER_DIR/merges.txt"
"$PYTHON_BIN" -m cs336_basics.encode_dataset \
  --input "$VALID_TEXT" \
  --output "$ROOT_DIR/data/tinystories_valid.bin" \
  --vocab "$TOKENIZER_DIR/vocab.json" \
  --merges "$TOKENIZER_DIR/merges.txt"
