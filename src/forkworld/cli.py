"""Command-line interface for reproducible ForkWorld experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .analysis import analyze_root
from .artifacts import json_safe
from .config import ConfigError, expand_sweep, get_path, load_config, set_path
from .navigation_support import pretrain_navigator_suite
from .protocols import resolve_device
from .runner import planned_runs, run_experiment, smoke_config


def _seed_list(value: str) -> list[int]:
    try:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not seeds or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must be non-empty and unique")
    return seeds


def _json(value: Any) -> str:
    return json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False, default=str)


def _load(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config, getattr(args, "overrides", ()) or ())
    if getattr(args, "output", None):
        set_path(config, "run.output_root", str(Path(args.output).resolve()))
    if getattr(args, "device", None):
        set_path(config, "run.device", args.device)
    return config


def command_validate(args: argparse.Namespace) -> int:
    config = _load(args)
    cells = expand_sweep(config)
    print(_json({"config": str(Path(args.config).resolve()), "valid": True, "cells": len(cells)}))
    return 0


def command_plan(args: argparse.Namespace) -> int:
    config = _load(args)
    pairs = planned_runs(
        config,
        smoke=args.smoke,
        seed_override=args.seeds,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        max_runs=args.max_runs,
    )
    preview = [
        {
            "seed": seed,
            "sweep": cell.get("_sweep_values", {}),
            "case": cell.get("_case_index"),
        }
        for cell, seed in pairs[: min(10, len(pairs))]
    ]
    print(_json({"runs": len(pairs), "preview": preview, "smoke": args.smoke}))
    return 0


def command_run(args: argparse.Namespace) -> int:
    config = _load(args)
    if args.dry_run:
        return command_plan(args)
    outcomes = run_experiment(
        config,
        output_root=args.output,
        device=args.device,
        jobs=args.jobs,
        smoke=args.smoke,
        seed_override=args.seeds,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        max_runs=args.max_runs,
        force=args.force,
        fail_fast=args.fail_fast,
    )
    counts = {status: sum(item["status"] == status for item in outcomes) for status in ("complete", "skipped", "failed")}
    failures = [item for item in outcomes if item["status"] == "failed"]
    print(_json({"counts": counts, "failures": failures, "output": str(args.output or get_path(config, "run.output_root"))}))
    return 1 if failures else 0


def command_analyze(args: argparse.Namespace) -> int:
    result = analyze_root(args.input, args.output)
    print(_json(result))
    return 0


def command_pretrain(args: argparse.Namespace) -> int:
    device = str(resolve_device(args.device))
    maps = 2 if args.smoke else args.navigation_maps
    epochs = min(args.epochs, 60) if args.smoke else args.epochs
    report = pretrain_navigator_suite(
        args.output,
        device=device,
        seed=args.seed,
        navigation_maps=maps,
        epochs=epochs,
        target_accuracy=0.9 if args.smoke else args.target_accuracy,
        target_patience=1 if args.smoke else args.target_patience,
        force=args.force,
    )
    print(_json(report))
    return 0


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="YAML experiment configuration")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="PATH=VALUE", help="dotted YAML override; repeatable")
    parser.add_argument("--output", help="override run.output_root")
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, cuda:N, or mps")


def _add_plan_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--smoke", action="store_true", help="run one tiny, scientifically valid integration cell")
    parser.add_argument("--seeds", type=_seed_list, help="comma-separated seed override")
    parser.add_argument("--shard-index", type=int, default=0, help="zero-based deterministic shard index")
    parser.add_argument("--shard-count", type=int, default=1, help="number of deterministic shards")
    parser.add_argument("--max-runs", type=int, help="cap runs after sharding (useful for pilots)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forkworld",
        description="Hypothesis-driven experiments for learned goal selection",
    )
    parser.add_argument("--version", action="version", version="forkworld 0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="validate a config and every expanded sweep cell")
    _add_config_arguments(validate)
    validate.set_defaults(function=command_validate)

    plan = commands.add_parser("plan", help="show the deterministic run plan without training")
    _add_config_arguments(plan)
    _add_plan_arguments(plan)
    plan.set_defaults(function=command_plan)

    run = commands.add_parser("run", help="execute a resumable experiment sweep")
    _add_config_arguments(run)
    _add_plan_arguments(run)
    run.add_argument("--jobs", type=int, default=1, help="parallel CPU worker processes")
    run.add_argument("--force", action="store_true", help="rerun completed run IDs")
    run.add_argument("--fail-fast", action="store_true", help="stop scheduling after the first failed run")
    run.add_argument("--dry-run", action="store_true", help="alias for plan using the same arguments")
    run.set_defaults(function=command_run)

    analyze = commands.add_parser("analyze", help="aggregate seed-level results, tests, and figures")
    analyze.add_argument("--input", required=True, help="artifact root")
    analyze.add_argument("--output", help="analysis directory (default: INPUT/analysis)")
    analyze.set_defaults(function=command_analyze)

    pretrain = commands.add_parser("pretrain-navigator", help="fit frozen fork and full-navigation policies")
    pretrain.add_argument("--output", required=True, help="navigator checkpoint directory")
    pretrain.add_argument("--device", default="auto")
    pretrain.add_argument("--seed", type=int, default=2027)
    pretrain.add_argument("--navigation-maps", type=int, default=32)
    pretrain.add_argument("--epochs", type=int, default=250)
    pretrain.add_argument("--target-accuracy", type=float, default=0.99)
    pretrain.add_argument("--target-patience", type=int, default=3)
    pretrain.add_argument("--smoke", action="store_true")
    pretrain.add_argument("--force", action="store_true")
    pretrain.set_defaults(function=command_pretrain)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.function(args))
    except (ConfigError, ValueError, FileNotFoundError, RuntimeError) as error:
        parser.exit(2, f"forkworld: {type(error).__name__}: {error}\n")


if __name__ == "__main__":
    sys.exit(main())
