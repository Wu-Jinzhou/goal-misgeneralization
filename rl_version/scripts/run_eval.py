#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from run_game_rl import add_game_rl_args, config_from_args  # noqa: E402
from gm_nim.hf import load_causal_lm, load_tokenizer  # noqa: E402
from gm_nim.rl_play import evaluate_game_policy  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a game-play RL checkpoint against opponents.")
    add_game_rl_args(parser, require_output_dir=False)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = config_from_args(args)
    tokenizer = load_tokenizer(args.checkpoint)
    model = load_causal_lm(args.checkpoint)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("a", encoding="utf-8") as handle:
        for offset, opponent in enumerate(config.eval_opponents):
            metrics = evaluate_game_policy(
                model,
                tokenizer,
                config,
                opponent=opponent,
                episodes=config.eval_episodes,
                seed=config.seed + offset,
            )
            handle.write(json.dumps({"checkpoint": args.checkpoint, **metrics}, sort_keys=True) + "\n")
