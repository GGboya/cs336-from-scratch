#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  if python3 -m pip --version >/dev/null 2>&1; then
    python3 -m pip install --user uv
  elif [[ -x "$HOME/miniconda3/bin/python" ]]; then
    "$HOME/miniconda3/bin/python" -m pip install --user uv
  else
    echo "No pip-enabled Python found; install pip or uv first" >&2
    exit 1
  fi
  export PATH="$HOME/.local/bin:$PATH"
fi

uv sync --python 3.11
uv run pytest -q
