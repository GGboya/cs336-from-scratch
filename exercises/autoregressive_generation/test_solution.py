from exercises.autoregressive_generation.solution import greedy_generate
from exercises.test_utils import check_case


def test_max_new_tokens_limits_generated_count():
    def model(current_ids):
        return [0.1, 0.9, 0.0]

    check_case(
        "最大新增数量不包含 prompt",
        greedy_generate,
        ([1], model, 3, 2),
        [1, 1, 1, 1],
    )


def test_eos_stops_generation_without_being_returned():
    calls = []

    def model(current_ids):
        calls.append(current_ids)
        return [0.1, 0.2, 0.7]

    check_case("生成 EOS 后停止且不返回 EOS", greedy_generate, ([4], model, 5, 2), [4])
    assert len(calls) == 1, "用例：EOS 停止。生成 EOS 后仍调用了模型"


def test_next_round_receives_updated_ids():
    seen = []
    distributions = [
        [0.8, 0.1, 0.0],
        [0.0, 0.9, 0.1],
        [0.0, 0.0, 1.0],
    ]

    def model(current_ids):
        seen.append(current_ids.copy())
        return distributions[len(seen) - 1]

    check_case(
        "下一轮接收更新后的完整 IDs",
        greedy_generate,
        ([5, 7], model, 3, 2),
        [5, 7, 0, 1],
    )
    assert seen == [[5, 7], [5, 7, 0], [5, 7, 0, 1]], "用例：自回归输入。模型收到的 IDs 不正确"


def test_tie_selects_smallest_id():
    def model(current_ids):
        return [0.5, 0.5]

    check_case("并列最大选择最小 ID", greedy_generate, ([3], model, 1, 9), [3, 0])


def test_zero_new_tokens_does_not_call_model():
    def model(current_ids):
        raise AssertionError("max_new_tokens 为 0 时不应调用模型")

    check_case("新增数量为零", greedy_generate, ([8, 9], model, 0, 2), [8, 9])


def test_prompt_is_not_modified():
    prompt_ids = [5, 7]

    def model(current_ids):
        return [0.0, 1.0]

    greedy_generate(prompt_ids, model, 2, 9)
    assert prompt_ids == [5, 7], "用例：输入不可修改。prompt_ids 被函数修改了"
