"""Run every available exercise test module against its reference implementation."""

from __future__ import annotations

import importlib


REFERENCES = {
    "bpe": "train_bpe",
    "bpe_encode": "bpe_encode",
    "embedding": "embedding",
    "lm_head": "lm_head",
    "softmax": "softmax",
    "greedy_decoding": "greedy_decode_step",
    "temperature": "temperature_softmax",
    "scaled_attention": "scaled_attention",
    "causal_mask": "apply_causal_mask",
    "multihead_attention": "split_heads",
    "causal_multihead": "causal_multihead_attention",
    "transformer_block": "residual_add",
    "transformer_block_forward": "transformer_block_forward",
    "transformer_lm_forward": "transformer_lm_forward",
    "linear": "linear",
    "rmsnorm": "rmsnorm",
    "rope": "rope",
    "swiglu": "swiglu",
    "utf8_bytes": "text_to_byte_ids",
    "special_tokens": "split_special_tokens",
    "top_p_sampling": "top_p_filter",
    "autoregressive_generation": "greedy_generate",
    "language_model_loss": "language_model_loss",
    "gradient_descent": "gradient_descent",
    "softmax_cross_entropy_backward": "softmax_cross_entropy_backward",
    "adamw": "adamw_step",
    "gradient_clipping": "gradient_clipping",
    "cosine_learning_rate": "get_lr_cosine_schedule",
    "training_batch": "get_batch",
    "embedding_backward": "embedding_backward",
    "tiny_language_model": "train_bigram",
    "resource_accounting": "count_transformer_parameters",
}


AUXILIARY_REFERENCES = {
    "swiglu": ["silu"],
    "utf8_bytes": ["byte_ids_to_text"],
}


def main() -> None:
    failed = []
    total_cases = 0
    for exercise, function_name in REFERENCES.items():
        reference = importlib.import_module(f"exercises.{exercise}.reference")
        tests = importlib.import_module(f"exercises.{exercise}.test_solution")
        setattr(tests, function_name, getattr(reference, function_name))
        for auxiliary_name in AUXILIARY_REFERENCES.get(exercise, []):
            setattr(tests, auxiliary_name, getattr(reference, auxiliary_name))
        cases = [
            (name, case)
            for name, case in vars(tests).items()
            if name.startswith("test_") and callable(case)
        ]
        for name, case in cases:
            try:
                case()
            except Exception as error:
                failed.append((exercise, name, error))
        total_cases += len(cases)
        print(f"{exercise}: {len(cases)} cases checked")

    if failed:
        for exercise, name, error in failed:
            print(f"FAIL | {exercise}.{name}: {type(error).__name__}: {error}")
        raise SystemExit(1)
    print(f"全部参考答案通过，共 {total_cases} 个 case！")


if __name__ == "__main__":
    main()
