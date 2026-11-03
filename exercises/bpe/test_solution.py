from exercises.bpe.solution import train_bpe
from exercises.test_utils import check_case


def test_single_merge():
    check_case("单次 merge", train_bpe, (["ab", "ab", "abc"], 4), (["a", "b", "c", "ab"], [("a", "b")]))


def test_multiple_merges_in_learning_order():
    check_case(
        "按学习顺序执行多次 merge",
        train_bpe,
        (["ab", "ab", "abc"], 5),
        (["a", "b", "c", "ab", "abc"], [("a", "b"), ("ab", "c")]),
    )


def test_frequency_tie_uses_lexicographically_larger_pair():
    check_case(
        "平频时选择字典序更大的 pair",
        train_bpe,
        (["cat", "cat", "cats"], 5),
        (["a", "c", "s", "t", "ca"], [("c", "a")]),
    )


def test_merge_is_left_to_right_and_non_overlapping():
    check_case("从左到右、不重叠合并", train_bpe, (["aaa"], 2), (["a", "aa"], [("a", "a")]))


def test_new_tokens_can_be_merged_again():
    check_case(
        "新 token 再次参与 merge",
        train_bpe,
        (["aaaa"], 3),
        (["a", "aa", "aaaa"], [("a", "a"), ("aa", "aa")]),
    )


def test_pair_occurrences_are_counted_repeatedly():
    check_case("重复 pair 重复计数", train_bpe, (["aba", "ab"], 3), (["a", "b", "ab"], [("a", "b")]))


def test_stop_when_no_pairs_exist():
    check_case("没有 pair 时停止", train_bpe, (["a", "b"], 10), (["a", "b"], []))


def test_empty_input():
    check_case("空输入", train_bpe, ([], 10), ([], []))


def test_already_at_or_below_initial_vocabulary_size():
    check_case("目标小于初始词表大小", train_bpe, (["ab"], 1), (["a", "b"], []))


def test_input_is_not_modified():
    words = ["ab", "ab", "abc"]
    train_bpe(words, 5)
    assert words == ["ab", "ab", "abc"], "用例：输入不可修改。words 被函数修改了"
