from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from cs336_basics.tokenizer import Tokenizer


def verify_split(tokenizer: Tokenizer, text_path: Path, token_path: Path) -> None:
    ids = np.memmap(token_path, dtype=np.uint16, mode="r")
    assert len(ids) > 0
    max_id = int(ids.max())
    min_id = int(ids.min())
    assert min_id >= 0 and max_id < len(tokenizer.vocab)

    sample_size = min(2048, len(ids))
    prefix = tokenizer.decode(ids[:sample_size].tolist()).encode("utf-8")
    suffix = tokenizer.decode(ids[-sample_size:].tolist()).encode("utf-8")
    with text_path.open("rb") as text_file:
        raw_prefix = text_file.read(max(len(prefix), 65536))
        text_file.seek(-max(len(suffix), 65536), os.SEEK_END)
        raw_suffix = text_file.read()
    assert raw_prefix.startswith(prefix)
    assert raw_suffix.endswith(suffix)

    print(
        f"{token_path.name}: {len(ids):,} tokens, IDs [{min_id}, {max_id}], "
        f"{text_path.stat().st_size / len(ids):.3f} bytes/token; boundary decoding passed"
    )


def main() -> None:
    tokenizer = Tokenizer.from_files(
        ROOT_DIR / "data/tinystories_10k/vocab.json",
        ROOT_DIR / "data/tinystories_10k/merges.txt",
        ["<|endoftext|>"],
    )
    verify_split(
        tokenizer,
        ROOT_DIR / "data/raw/tinystories_train/TinyStoriesV2-GPT4-train.txt",
        ROOT_DIR / "data/tinystories_train.bin",
    )
    verify_split(
        tokenizer,
        ROOT_DIR / "data/raw/tinystories_valid/TinyStoriesV2-GPT4-valid.txt",
        ROOT_DIR / "data/tinystories_valid.bin",
    )


if __name__ == "__main__":
    main()
