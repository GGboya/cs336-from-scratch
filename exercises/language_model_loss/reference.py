"""Numerically stable reference implementation using plain loops."""

from math import exp, log


def language_model_loss(logits: list[float], target_id: int) -> float:
    # 同时减去最大 logit，不改变 Softmax 概率，并且避免 exp 溢出。
    max_logit = logits[0]
    for logit in logits:
        if logit > max_logit:
            max_logit = logit

    exp_sum = 0.0
    for logit in logits:
        exp_sum += exp(logit - max_logit)

    # log(sum(exp(logits))) 的稳定写法。
    log_exp_sum = max_logit + log(exp_sum)
    return log_exp_sum - logits[target_id]
