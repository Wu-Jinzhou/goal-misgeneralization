#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from run_game_rl import add_game_rl_args, config_from_args  # noqa: E402
from gm_nim.rl_play import run_game_rl_training  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run game-play RL for GMG experiments with opponent-distribution shift."
    )
    add_game_rl_args(parser)
    run_game_rl_training(config_from_args(parser.parse_args()))


if __name__ == "__main__":
    main()

