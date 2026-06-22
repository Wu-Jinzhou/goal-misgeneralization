#!/usr/bin/env python
from __future__ import annotations

import argparse

from gm_nim.training import TrainConfig, run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SFT, DANN, or contrastive training.")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="410m")
    parser.add_argument("--eval-file")
    parser.add_argument("--mode", choices=["sft", "dann", "contrastive"], default="sft")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--epochs", type=float, default=300.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument("--lambda-value", type=float, default=0.05)
    parser.add_argument("--discriminator-hidden", type=int, default=512)
    args = parser.parse_args()
    run_training(TrainConfig(**vars(args)))


if __name__ == "__main__":
    main()

