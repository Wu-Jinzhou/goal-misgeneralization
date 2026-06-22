#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import fields

from gm_nim.rl_play import GameRLConfig, run_game_rl_training


def add_game_rl_args(parser: argparse.ArgumentParser, *, require_output_dir: bool = True) -> None:
    parser.add_argument("--output-dir", required=require_output_dir, default="")
    parser.add_argument("--model", default="410m")
    parser.add_argument("--game", choices=["bounded", "multipile", "fibonacci", "wythoff"], default="bounded")
    parser.add_argument("--mr", type=int, default=5)
    parser.add_argument("--train-opponent", default="random")
    parser.add_argument("--eval-opponents", nargs="*", default=["random", "optimal"])
    parser.add_argument("--self-play", action="store_true")
    parser.add_argument("--no-randomize-player", action="store_true")
    parser.add_argument("--min-pile", type=int, default=20)
    parser.add_argument("--max-pile", type=int, default=400)
    parser.add_argument("--pile-count", type=int, default=3)
    parser.add_argument("--min-heap", type=int, default=1)
    parser.add_argument("--max-heap", type=int, default=120)
    parser.add_argument("--fib-min-pile", type=int, default=8)
    parser.add_argument("--fib-max-pile", type=int, default=160)
    parser.add_argument("--wythoff-max-heap", type=int, default=120)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--episodes-per-step", type=int, default=8)
    parser.add_argument("--update-batch-size", type=int, default=16)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--max-turns", type=int, default=128)
    parser.add_argument("--max-prompt-length", type=int, default=160)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--kl-coef", type=float, default=0.02)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--invalid-reward", type=float, default=-1.0)
    parser.add_argument("--win-reward", type=float, default=1.0)
    parser.add_argument("--loss-reward", type=float, default=-1.0)
    parser.add_argument("--discount", type=float, default=1.0)
    parser.add_argument("--no-normalize-advantages", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--proxy-metrics", nargs="*", default=[])


def config_from_args(args: argparse.Namespace) -> GameRLConfig:
    values = vars(args)
    values["randomize_player"] = not values.pop("no_randomize_player")
    values["normalize_advantages"] = not values.pop("no_normalize_advantages")
    allowed = {field.name for field in fields(GameRLConfig)}
    values = {key: value for key, value in values.items() if key in allowed}
    return GameRLConfig(**values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run game-play RL against fixed opponents or self-play.")
    add_game_rl_args(parser)
    run_game_rl_training(config_from_args(parser.parse_args()))


if __name__ == "__main__":
    main()
