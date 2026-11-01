from .adapters import run_train_bpe
from .common import FIXTURES_PATH


def test_parallel_pretokenization_matches_serial():
    input_path = FIXTURES_PATH / "tinystories_sample_5M.txt"
    serial_vocab, serial_merges = run_train_bpe(input_path, 1000, ["<|endoftext|>"], num_processes=1)
    parallel_vocab, parallel_merges = run_train_bpe(input_path, 1000, ["<|endoftext|>"], num_processes=4)
    assert parallel_vocab == serial_vocab
    assert parallel_merges == serial_merges
