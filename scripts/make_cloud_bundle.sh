#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOKENIZER_DIR="$ROOT_DIR/data/tinystories_10k"
if [[ ! -f "$TOKENIZER_DIR/vocab.json" || ! -f "$TOKENIZER_DIR/merges.txt" ]]; then
  echo "Missing $TOKENIZER_DIR; run scripts/prepare_tinystories.sh first" >&2
  exit 1
fi

mkdir -p "$ROOT_DIR/dist"
tar -czf "$ROOT_DIR/dist/cs336_assignment1_cloud.tar.gz" \
  -C "$ROOT_DIR" \
  cs336_basics tests scripts pyproject.toml uv.lock .python-version README.md RUNNING.zh-CN.md RESULTS.zh-CN.md LOCAL_USAGE.zh-CN.md data/tinystories_10k
ls -lh "$ROOT_DIR/dist/cs336_assignment1_cloud.tar.gz"
