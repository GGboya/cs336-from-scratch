from __future__ import annotations

import argparse
import time
from array import array
from pathlib import Path

from .tokenizer import Tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode a text file as raw little-endian uint16 token IDs")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--merges", required=True)
    parser.add_argument("--special-token", action="append", default=["<|endoftext|>"])
    args = parser.parse_args()

    tokenizer = Tokenizer.from_files(args.vocab, args.merges, args.special_token)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    token_count = 0
    buffer = array("H")
    start = time.perf_counter()

    with open(args.input, encoding="utf-8") as source, output_path.open("wb") as output:
        for token_id in tokenizer.encode_iterable(source):
            buffer.append(token_id)
            token_count += 1
            if len(buffer) >= 1_000_000:
                buffer.tofile(output)
                del buffer[:]
        if buffer:
            buffer.tofile(output)

    elapsed = time.perf_counter() - start
    input_bytes = Path(args.input).stat().st_size
    print(f"Wrote {token_count:,} uint16 tokens to {output_path} in {elapsed:.2f}s")
    if token_count:
        print(f"Compression: {input_bytes / token_count:.3f} input bytes/token")


if __name__ == "__main__":
    main()
