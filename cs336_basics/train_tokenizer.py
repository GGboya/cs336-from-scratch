from __future__ import annotations

import argparse
import base64
import json
import resource
import sys
import time
from pathlib import Path

from .tokenizer import train_bpe


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--num-processes", type=int)
    parser.add_argument("--special-token", action="append", default=["<|endoftext|>"])
    args = parser.parse_args()

    start = time.perf_counter()
    vocab, merges = train_bpe(args.input, args.vocab_size, args.special_token, args.num_processes)
    elapsed = time.perf_counter() - start
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    serialized_vocab = {
        "format": "cs336-bpe-base64-v1",
        "tokens": {
            str(token_id): base64.b64encode(token).decode("ascii") for token_id, token in vocab.items()
        },
    }
    with (output_dir / "vocab.json").open("w", encoding="utf-8") as file:
        json.dump(serialized_vocab, file, indent=2)
    with (output_dir / "merges.txt").open("w", encoding="ascii") as file:
        for left, right in merges:
            file.write(
                f"{base64.b64encode(left).decode('ascii')} {base64.b64encode(right).decode('ascii')}\n"
            )

    longest = max(vocab.values(), key=len)
    rss_scale = 1024 * 1024 if sys.platform == "darwin" else 1024
    peak_memory_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / rss_scale
    print(f"Trained {len(vocab)} tokens and {len(merges)} merges in {elapsed:.2f}s")
    print(f"Peak RSS: {peak_memory_mb:.1f} MiB")
    print(f"Longest token ({len(longest)} bytes): {longest!r}")
    print(f"Saved tokenizer to {output_dir}")


if __name__ == "__main__":
    main()
