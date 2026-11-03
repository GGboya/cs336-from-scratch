from exercises.embedding.solution import embedding
from exercises.test_utils import check_case


WEIGHT = [
    [0.1, 0.2],
    [0.3, 0.4],
    [0.5, 0.6],
]


def test_basic_lookup():
    check_case("基本查表", embedding, ([2, 0], WEIGHT), [[0.5, 0.6], [0.1, 0.2]])


def test_order_and_repeated_ids_are_preserved():
    check_case(
        "保持顺序和重复 ID",
        embedding,
        ([1, 0, 1], WEIGHT),
        [[0.3, 0.4], [0.1, 0.2], [0.3, 0.4]],
    )


def test_single_id():
    check_case("单个 ID", embedding, ([1], WEIGHT), [[0.3, 0.4]])


def test_empty_ids():
    check_case("空 ID 列表", embedding, ([], WEIGHT), [])


def test_different_vector_dimensions():
    weight = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    check_case("三维向量", embedding, ([1, 0], weight), [[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]])


def test_inputs_are_not_modified():
    token_ids = [2, 0]
    weight = [row.copy() for row in WEIGHT]
    original_weight = [row.copy() for row in weight]
    embedding(token_ids, weight)
    assert token_ids == [2, 0], "用例：输入不可修改。token_ids 被函数修改了"
    assert weight == original_weight, "用例：输入不可修改。weight 被函数修改了"


def test_modifying_output_does_not_modify_input_weight():
    weight = [row.copy() for row in WEIGHT]
    output = embedding([1], weight)
    output[0][0] = 99.0
    assert weight[1][0] == 0.3, "用例：输出独立副本。修改输出后，原始 weight 也被修改了"
