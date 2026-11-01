#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
TRAIN_DIR="$ROOT_DIR/data/raw/tinystories_train"
VALID_DIR="$ROOT_DIR/data/raw/tinystories_valid"
mkdir -p "$TRAIN_DIR" "$VALID_DIR"

curl -L --fail --retry 3 --retry-delay 5 -C - \
  -o "$TRAIN_DIR/TinyStoriesV2-GPT4-train.txt" \
  "$HF_ENDPOINT/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
curl -L --fail --retry 3 --retry-delay 5 -C - \
  -o "$VALID_DIR/TinyStoriesV2-GPT4-valid.txt" \
  "$HF_ENDPOINT/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"
