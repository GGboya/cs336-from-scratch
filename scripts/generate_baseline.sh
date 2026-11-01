#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT_DIR/runs/tinystories_baseline"
cd "$ROOT_DIR"

"$ROOT_DIR/.venv/bin/python" -m cs336_basics.generate \
  --run-dir "$RUN_DIR" \
  --vocab "$ROOT_DIR/data/tinystories_10k/vocab.json" \
  --merges "$ROOT_DIR/data/tinystories_10k/merges.txt" \
  --device cuda \
  --prompt "Once upon a time" \
  --max-new-tokens 256 --temperature 0.8 --top-p 0.9 \
  > "$RUN_DIR/generated_sample.txt"

cat "$RUN_DIR/generated_sample.txt"
