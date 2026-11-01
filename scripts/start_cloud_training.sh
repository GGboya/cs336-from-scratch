#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT_DIR/runs/tinystories_baseline"
mkdir -p "$RUN_DIR"

if tmux has-session -t cs336 2>/dev/null; then
  echo "tmux session 'cs336' already exists; attach with: tmux attach -t cs336" >&2
  exit 1
fi

tmux new-session -d -s cs336 -c "$ROOT_DIR" \
  "$ROOT_DIR/scripts/train_tinystories_baseline.sh > $RUN_DIR/console.log 2>&1"
tmux ls
