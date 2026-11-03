from exercises.bpe_encode.solution import bpe_encode
from exercises.test_utils import check_case


MERGES = [(" ", "a"), (" a", "b"), ("a", "b"), ("ab", "c")]
VOCAB = ["a", "b", "c", " ", " a", " ab", "ab", "abc"]


def test_single_pretoken():
    check_case("单个 pre-token", bpe_encode, ("abc", MERGES, VOCAB), [7])


def test_complete_sentence_with_leading_spaces():
    check_case("带前导空格的完整句子", bpe_encode, ("abc ab", MERGES, VOCAB), [7, 5])


def test_repeated_word_uses_same_merge_rules():
    check_case("重复单词使用相同 merge 规则", bpe_encode, ("ab ab", MERGES, VOCAB), [6, 5])


def test_token_without_merge():
    check_case("不需要 merge 的 token", bpe_encode, ("b", MERGES, VOCAB), [1])


def test_empty_text():
    check_case("空文本", bpe_encode, ("", MERGES, VOCAB), [])


def test_non_overlapping_merge():
    check_case("不重叠合并", bpe_encode, ("aaaa", [("a", "a"), ("aa", "aa")], ["a", "aa", "aaaa"]), [2])


def test_new_token_participates_in_next_merge():
    check_case("新 token 参与下一次 merge", bpe_encode, ("aaa", [("a", "a"), ("aa", "a")], ["a", "aa", "aaa"]), [2])


def test_inputs_are_not_modified():
    merges = MERGES.copy()
    vocab = VOCAB.copy()
    bpe_encode("abc ab", merges, vocab)
    assert merges == MERGES, "用例：输入不可修改。merges 被函数修改了"
    assert vocab == VOCAB, "用例：输入不可修改。vocab 被函数修改了"
