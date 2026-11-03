from exercises.test_utils import check_case
from exercises.transformer_lm_forward.solution import transformer_lm_forward


def test_embedding_final_norm_and_lm_head():
    check_case(
        "最终 RMSNorm 后计算 logits",
        transformer_lm_forward,
        ([0], [[3.0, 4.0]], [], [1.0, 1.0], [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], 0.0),
        [[0.848528137423857, 1.131370849898476, 1.9798989873223332]],
    )


def test_all_positions_produce_logits():
    check_case(
        "所有位置都产生 logits",
        transformer_lm_forward,
        ([0, 1], [[3.0, 4.0], [0.0, 2.0]], [], [1.0, 1.0], [[1.0, 0.0], [0.0, 1.0]], 0.0),
        [[0.848528137423857, 1.131370849898476], [0.0, 1.414213562373095]],
    )


def test_blocks_run_in_order_before_final_norm():
    def add_one(x):
        return [[value + 1.0 for value in row] for row in x]

    def double(x):
        return [[2.0 * value for value in row] for row in x]

    check_case(
        "Block 按顺序执行",
        transformer_lm_forward,
        ([0], [[3.0, 4.0]], [add_one, double], [1.0, 1.0], [[1.0, 0.0], [0.0, 1.0]], 0.0),
        [[8.0 / (82.0 ** 0.5), 10.0 / (82.0 ** 0.5)]],
    )


def test_final_gain_is_applied():
    check_case(
        "最终 RMSNorm 的 gain",
        transformer_lm_forward,
        ([0], [[3.0, 4.0]], [], [1.0, 2.0], [[1.0, 0.0], [0.0, 1.0]], 0.0),
        [[0.848528137423857, 2.262741699796952]],
    )


def test_inputs_are_not_modified():
    token_ids = [0]
    embedding = [[3.0, 4.0]]
    gain = [1.0, 1.0]
    head = [[1.0, 0.0], [0.0, 1.0]]
    transformer_lm_forward(token_ids, embedding, [], gain, head)
    assert token_ids == [0]
    assert embedding == [[3.0, 4.0]]
    assert gain == [1.0, 1.0]
    assert head == [[1.0, 0.0], [0.0, 1.0]]
