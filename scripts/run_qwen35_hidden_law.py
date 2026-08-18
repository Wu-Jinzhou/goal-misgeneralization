#!/usr/bin/env python3
"""Run one exact condition or one native shard of the hidden-law plan."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from goalzendo_hidden_law.runner import execute_plan


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/goalzendo/qwen35_hidden_law_finite_choice.yaml"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--condition",
        help="exact registered run_id or plan_key (mutually exclusive with native sharding)",
    )
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run the reserved one-block execution smoke; requires exactly one condition",
    )
    args = parser.parse_args(argv)
    if args.condition is not None and (args.shard_index is not None or args.num_shards is not None):
        parser.error("--condition is mutually exclusive with --shard-index/--num-shards")
    if (args.shard_index is None) != (args.num_shards is None):
        parser.error("--shard-index and --num-shards must be supplied together")
    results = execute_plan(
        args.config,
        repo_root=args.repo_root,
        condition_reference=args.condition,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        output_root=args.output_root,
        device=args.device,
        smoke=args.smoke,
    )
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
