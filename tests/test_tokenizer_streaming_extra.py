import pytest

from .test_tokenizer import MERGES_PATH, VOCAB_PATH, get_tokenizer_from_vocab_merges_path


@pytest.mark.parametrize("chunk_size", range(1, 12))
def test_encode_iterable_arbitrary_chunk_boundaries(chunk_size):
    specials = ["<|endoftext|>", "<|endoftext|><|endoftext|>"]
    tokenizer = get_tokenizer_from_vocab_merges_path(VOCAB_PATH, MERGES_PATH, specials)
    text = "Hello🙃 <|endoftext|><|endoftext|>\n\nnext <|endoftext|>!"
    chunks = [text[index : index + chunk_size] for index in range(0, len(text), chunk_size)]
    assert list(tokenizer.encode_iterable(chunks)) == tokenizer.encode(text)
