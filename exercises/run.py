from __future__ import annotations

import argparse
import importlib


EXERCISES = [
    "bpe",
    "bpe_encode",
    "embedding",
    "embedding_backward",
    "language_model_loss",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run language-model exercise cases without pytest output")
    parser.add_argument("exercise", choices=EXERCISES)
    args = parser.parse_args()

    module = importlib.import_module(f"exercises.{args.exercise}.test_solution")
    cases = [
        (name, function)
        for name, function in module.__dict__.items()
        if name.startswith("test_") and callable(function)
    ]
    passed = 0
    print(f"练习：{args.exercise}\n")

    for name, case in cases:
        try:
            case()
            passed += 1
            print(f"PASS | {name}")
        except AssertionError as error:
            print(f"FAIL | {name}{error}\n")
        except Exception as error:
            print(f"FAIL | {name}\n执行出错：{type(error).__name__}: {error}\n")

    print(f"结果：{passed}/{len(cases)} 个 case 通过")
    if passed != len(cases):
        raise SystemExit(1)
    print("全部通过！")


if __name__ == "__main__":
    main()
