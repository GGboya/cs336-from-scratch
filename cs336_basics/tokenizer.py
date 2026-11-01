from __future__ import annotations

import base64
import json
import multiprocessing
import os
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path

import regex

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PRETOKEN_PATTERN = regex.compile(PAT)


def _split_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    if not special_tokens:
        return [text]
    pattern = re.compile("(" + "|".join(re.escape(token) for token in special_tokens) + ")")
    return pattern.split(text)


def _count_pretokens(text: str, special_tokens: list[str]) -> Counter[tuple[bytes, ...]]:
    word_counts: Counter[tuple[bytes, ...]] = Counter()
    single_bytes = [bytes([value]) for value in range(256)]
    for segment in _split_special_tokens(text, special_tokens):
        if segment in special_tokens:
            continue
        for match in PRETOKEN_PATTERN.finditer(segment):
            raw = match.group().encode("utf-8")
            word_counts[tuple(single_bytes[value] for value in raw)] += 1
    return word_counts


def _count_pretokens_chunk(
    task: tuple[str, int, int, list[str]],
) -> Counter[tuple[bytes, ...]]:
    input_path, start, end, special_tokens = task
    with open(input_path, "rb") as file:
        file.seek(start)
        text = file.read(end - start).decode("utf-8")
    return _count_pretokens(text, special_tokens)


def _find_chunk_boundaries(input_path: str, num_chunks: int, special_token: bytes) -> list[int]:
    file_size = os.path.getsize(input_path)
    chunk_size = file_size // num_chunks
    boundaries = [index * chunk_size for index in range(num_chunks)] + [file_size]
    with open(input_path, "rb") as file:
        for index in range(1, len(boundaries) - 1):
            position = boundaries[index]
            file.seek(position)
            while True:
                chunk = file.read(4096)
                if not chunk:
                    boundaries[index] = file_size
                    break
                found_at = chunk.find(special_token)
                if found_at >= 0:
                    boundaries[index] = position + found_at
                    break
                position += len(chunk)
    return sorted(set(boundaries))


def train_bpe(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    special_tokens = list(special_tokens)
    split_tokens = sorted(special_tokens, key=len, reverse=True)
    vocab = {index: token.encode("utf-8") for index, token in enumerate(special_tokens)}
    next_id = len(vocab)
    for value in range(256):
        token = bytes([value])
        if token not in vocab.values():
            vocab[next_id] = token
            next_id += 1

    input_path = str(input_path)
    if num_processes is None:
        num_processes = min(4, os.cpu_count() or 1) if os.path.getsize(input_path) > 64 * 1024 * 1024 else 1
    word_counts: Counter[tuple[bytes, ...]] = Counter()
    if num_processes > 1 and special_tokens:
        boundaries = _find_chunk_boundaries(input_path, num_processes, special_tokens[0].encode("utf-8"))
        tasks = [
            (input_path, start, end, split_tokens)
            for start, end in zip(boundaries, boundaries[1:])
            if start < end
        ]
        if len(tasks) > 1:
            with multiprocessing.Pool(processes=len(tasks)) as pool:
                for chunk_counts in pool.imap(_count_pretokens_chunk, tasks):
                    word_counts.update(chunk_counts)
        else:
            for task in tasks:
                word_counts.update(_count_pretokens_chunk(task))
    else:
        text = Path(input_path).read_text(encoding="utf-8")
        word_counts = _count_pretokens(text, split_tokens)

    words = [list(word) for word in word_counts]
    counts = list(word_counts.values())
    pair_counts: dict[tuple[bytes, bytes], int] = {}
    pair_locations: dict[tuple[bytes, bytes], set[int]] = {}

    def add_word_pairs(word_index: int) -> None:
        word = words[word_index]
        count = counts[word_index]
        for pair in zip(word, word[1:]):
            pair_counts[pair] = pair_counts.get(pair, 0) + count
            pair_locations.setdefault(pair, set()).add(word_index)

    def remove_word_pairs(word_index: int) -> None:
        word = words[word_index]
        count = counts[word_index]
        pairs = list(zip(word, word[1:]))
        for pair in pairs:
            pair_counts[pair] -= count
        for pair in set(pairs):
            locations = pair_locations[pair]
            locations.discard(word_index)
            if not locations:
                del pair_locations[pair]
                del pair_counts[pair]

    for word_index in range(len(words)):
        add_word_pairs(word_index)

    merges: list[tuple[bytes, bytes]] = []
    while len(vocab) < vocab_size and pair_counts:
        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        merges.append(best_pair)
        vocab[next_id] = best_pair[0] + best_pair[1]
        next_id += 1

        affected_words = tuple(pair_locations[best_pair])
        for word_index in affected_words:
            remove_word_pairs(word_index)
            old_word = words[word_index]
            new_word: list[bytes] = []
            index = 0
            while index < len(old_word):
                if (
                    index + 1 < len(old_word)
                    and old_word[index] == best_pair[0]
                    and old_word[index + 1] == best_pair[1]
                ):
                    new_word.append(best_pair[0] + best_pair[1])
                    index += 2
                else:
                    new_word.append(old_word[index])
                    index += 1
            words[word_index] = new_word
            add_word_pairs(word_index)

    return vocab, merges


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = dict(vocab)
        self.token_to_id = {token: token_id for token_id, token in self.vocab.items()}
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
        self.special_tokens = sorted(special_tokens or [], key=len, reverse=True)

        next_id = max(self.vocab, default=-1) + 1
        for token in self.special_tokens:
            encoded = token.encode("utf-8")
            if encoded not in self.token_to_id:
                self.vocab[next_id] = encoded
                self.token_to_id[encoded] = next_id
                next_id += 1

        self.special_token_set = set(self.special_tokens)
        self.special_pattern = (
            re.compile("(" + "|".join(re.escape(token) for token in self.special_tokens) + ")")
            if self.special_tokens
            else None
        )
        self._cache: dict[bytes, tuple[int, ...]] = {}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | Path,
        merges_filepath: str | Path,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        with open(vocab_filepath, encoding="utf-8") as file:
            serialized_vocab = json.load(file)
        if serialized_vocab.get("format") == "cs336-bpe-base64-v1":
            vocab = {
                int(token_id): base64.b64decode(token)
                for token_id, token in serialized_vocab["tokens"].items()
            }
        else:
            vocab = {int(token_id): token.encode("utf-8") for token, token_id in serialized_vocab.items()}
        merges = []
        with open(merges_filepath, encoding="utf-8") as file:
            for line in file:
                line = line.rstrip("\n")
                if line:
                    left, right = line.split(" ")
                    if serialized_vocab.get("format") == "cs336-bpe-base64-v1":
                        merges.append((base64.b64decode(left), base64.b64decode(right)))
                    else:
                        merges.append((left.encode("utf-8"), right.encode("utf-8")))
        return cls(vocab, merges, special_tokens)

    def _encode_pretoken(self, pretoken: bytes, use_cache: bool = True) -> tuple[int, ...]:
        cached = self._cache.get(pretoken) if use_cache else None
        if cached is not None:
            return cached

        tokens = [pretoken[index : index + 1] for index in range(len(pretoken))]
        while len(tokens) > 1:
            candidates = [
                (rank, index)
                for index, pair in enumerate(zip(tokens, tokens[1:]))
                if (rank := self.merge_ranks.get(pair)) is not None
            ]
            if not candidates:
                break
            _, merge_index = min(candidates)
            tokens[merge_index : merge_index + 2] = [tokens[merge_index] + tokens[merge_index + 1]]

        encoded = tuple(self.token_to_id[token] for token in tokens)
        if use_cache and len(self._cache) < 1024:
            self._cache[pretoken] = encoded
        return encoded

    def _encode_ordinary(self, text: str, use_cache: bool = True) -> Iterator[int]:
        for match in PRETOKEN_PATTERN.finditer(text):
            yield from self._encode_pretoken(match.group().encode("utf-8"), use_cache=use_cache)

    def encode(self, text: str) -> list[int]:
        if self.special_pattern is None:
            return list(self._encode_ordinary(text))

        ids = []
        for segment in self.special_pattern.split(text):
            if segment in self.special_token_set:
                ids.append(self.token_to_id[segment.encode("utf-8")])
            elif segment:
                ids.extend(self._encode_ordinary(segment))
        return ids

    def _partial_special_suffix_length(self, text: str) -> int:
        max_length = 0
        for special_token in self.special_tokens:
            upper_bound = min(len(text), len(special_token) - 1)
            for length in range(upper_bound, max_length, -1):
                if text.endswith(special_token[:length]):
                    max_length = length
                    break
        return max_length

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        buffer = ""
        for chunk in iterable:
            buffer += chunk
            while buffer:
                special_match = self.special_pattern.search(buffer) if self.special_pattern else None
                if special_match is not None:
                    matched_token = special_match.group()
                    could_be_extended = any(
                        token != matched_token and token.startswith(buffer[special_match.start() :])
                        for token in self.special_tokens
                    )
                    if not could_be_extended:
                        yield from self._encode_ordinary(buffer[: special_match.start()])
                        yield self.token_to_id[matched_token.encode("utf-8")]
                        buffer = buffer[special_match.end() :]
                        continue

                safe_end = len(buffer) - self._partial_special_suffix_length(buffer)
                matches = list(PRETOKEN_PATTERN.finditer(buffer[:safe_end]))
                if len(matches) < 2:
                    break
                consumed_end = matches[-1].start()
                for match in matches[:-1]:
                    yield from self._encode_pretoken(match.group().encode("utf-8"))
                buffer = buffer[consumed_end:]

        if buffer:
            yield from self.encode(buffer)

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[token_id] for token_id in ids).decode("utf-8", errors="replace")
