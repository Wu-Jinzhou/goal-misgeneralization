#!/usr/bin/env python
from __future__ import annotations

import argparse

from gm_nim.rl import RLConfig, run_rl_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RL fine-tuning for bounded Nim.")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="410m")
    parser.add_argument("--eval-files", nargs="*", default=[])
    parser.add_argument("--max-prompt-length", type=int, default=120)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--mini-batch-size", type=int, default=8)
    parser.add_argument("--samples-per-prompt", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--kl-coef", type=float, default=0.02)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--reward-exact", type=float, default=1.0)
    parser.add_argument("--reward-invalid", type=float, default=-0.25)
    parser.add_argument("--reward-wrong", type=float, default=0.0)
    parser.add_argument("--coarsened-factors", nargs="*", type=int, default=[])
    parser.add_argument("--coarsened-reward", type=float, default=0.25)
    parser.add_argument("--no-normalize-group-advantages", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    args = parser.parse_args()
    values = vars(args)
    values["normalize_group_advantages"] = not values.pop("no_normalize_group_advantages")
    run_rl_training(RLConfig(**values))


if __name__ == "__main__":
    main()

