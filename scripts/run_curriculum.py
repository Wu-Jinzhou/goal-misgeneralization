#!/usr/bin/env python
from __future__ import annotations

import argparse

from gm_nim.curriculum import CurriculumConfig, run_curriculum


def main() -> None:
    parser = argparse.ArgumentParser(description="Run two-phase curriculum training with replay.")
    parser.add_argument("--phase1-file", required=True)
    parser.add_argument("--phase2-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="410m")
    parser.add_argument("--eval-files", nargs="*", default=[])
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--phase-steps", type=int, default=75_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-ratio", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--save-steps", type=int, default=5_000)
    parser.add_argument("--eval-steps", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--factors", nargs="*", type=int, default=[2, 3, 4])
    args = parser.parse_args()
    run_curriculum(CurriculumConfig(**vars(args)))


if __name__ == "__main__":
    main()

