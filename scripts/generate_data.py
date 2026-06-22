#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from gm_nim.data import (
    make_bounded_nim_dataset,
    make_cheat_pair_dataset,
    make_fibonacci_dataset,
    make_modular_reduction_dataset,
    make_multipile_dataset,
    make_multitask_bounded_dataset,
    make_wythoff_dataset,
    write_jsonl,
)


def cmd_bounded(args: argparse.Namespace) -> None:
    out = Path(args.out_dir)
    train = make_bounded_nim_dataset(
        mr=args.mr,
        size=args.train_size,
        seed=args.seed,
        split="train",
        current_min=args.current_min,
        current_max=args.current_max,
        history_len=args.history_len,
    )
    eval_set = make_bounded_nim_dataset(
        mr=args.mr,
        size=args.eval_size,
        seed=args.seed + 1,
        split="eval",
        current_min=args.current_min,
        current_max=args.current_max,
        history_len=args.history_len,
        exclude_prompts={example.prompt for example in train},
    )
    write_jsonl(out / f"bounded_mr{args.mr}_train.jsonl", train)
    write_jsonl(out / f"bounded_mr{args.mr}_eval.jsonl", eval_set)


def cmd_multitask(args: argparse.Namespace) -> None:
    out = Path(args.out_dir)
    train = make_multitask_bounded_dataset(
        mrs=args.mrs,
        size_per_task=args.train_size_per_task,
        seed=args.seed,
        split="train",
        history_len=args.history_len,
    )
    eval_set = make_multitask_bounded_dataset(
        mrs=args.mrs,
        size_per_task=args.eval_size_per_task,
        seed=args.seed + 1,
        split="eval",
        history_len=args.history_len,
    )
    suffix = "".join(str(mr) for mr in args.mrs)
    write_jsonl(out / f"bounded_multitask_{suffix}_train.jsonl", train)
    write_jsonl(out / f"bounded_multitask_{suffix}_eval.jsonl", eval_set)


def cmd_mod(args: argparse.Namespace) -> None:
    train, eval_set = make_modular_reduction_dataset(
        modulus=args.modulus,
        seed=args.seed,
        train_size=args.train_size,
        eval_size=args.eval_size,
        label_mode=args.label_mode,
    )
    out = Path(args.out_dir)
    tag = f"mod{args.modulus}_{args.label_mode}"
    write_jsonl(out / f"{tag}_train.jsonl", train)
    write_jsonl(out / f"{tag}_eval.jsonl", eval_set)


def cmd_shortcut(args: argparse.Namespace) -> None:
    datasets = make_cheat_pair_dataset(
        seed=args.seed,
        train_size=args.train_size,
        eval_size=args.eval_size,
        mr=args.mr,
        enforce_cheat_consistency=not args.literal_cheat_labels,
    )
    out = Path(args.out_dir)
    for name, examples in datasets.items():
        write_jsonl(out / f"shortcut_mr{args.mr}_{name}.jsonl", examples)


def cmd_extensions(args: argparse.Namespace) -> None:
    out = Path(args.out_dir)
    write_jsonl(
        out / "multipile_train.jsonl",
        make_multipile_dataset(size=args.size, seed=args.seed, split="train"),
    )
    write_jsonl(
        out / "fibonacci_train.jsonl",
        make_fibonacci_dataset(size=args.size, seed=args.seed + 1, split="train"),
    )
    write_jsonl(
        out / "wythoff_train.jsonl",
        make_wythoff_dataset(size=args.size, seed=args.seed + 2, split="train"),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate paper reproduction datasets.")
    sub = parser.add_subparsers(required=True)

    bounded = sub.add_parser("bounded")
    bounded.add_argument("--mr", type=int, required=True)
    bounded.add_argument("--out-dir", default="data")
    bounded.add_argument("--train-size", type=int, default=15_000)
    bounded.add_argument("--eval-size", type=int, default=2_000)
    bounded.add_argument("--seed", type=int, default=0)
    bounded.add_argument("--current-min", type=int, default=1)
    bounded.add_argument("--current-max", type=int, default=400)
    bounded.add_argument("--history-len", type=int, default=3)
    bounded.set_defaults(func=cmd_bounded)

    multitask = sub.add_parser("multitask")
    multitask.add_argument("--mrs", nargs="+", type=int, required=True)
    multitask.add_argument("--out-dir", default="data")
    multitask.add_argument("--train-size-per-task", type=int, default=15_000)
    multitask.add_argument("--eval-size-per-task", type=int, default=2_000)
    multitask.add_argument("--seed", type=int, default=0)
    multitask.add_argument("--history-len", type=int, default=3)
    multitask.set_defaults(func=cmd_multitask)

    mod = sub.add_parser("mod")
    mod.add_argument("--modulus", type=int, required=True)
    mod.add_argument("--out-dir", default="data")
    mod.add_argument("--train-size", type=int, default=9_000)
    mod.add_argument("--eval-size", type=int, default=1_000)
    mod.add_argument("--seed", type=int, default=0)
    mod.add_argument("--label-mode", choices=["standard", "reversed", "scrambled"], default="standard")
    mod.set_defaults(func=cmd_mod)

    shortcut = sub.add_parser("shortcut")
    shortcut.add_argument("--out-dir", default="data")
    shortcut.add_argument("--train-size", type=int, default=60_000)
    shortcut.add_argument("--eval-size", type=int, default=2_000)
    shortcut.add_argument("--mr", type=int, default=4)
    shortcut.add_argument("--seed", type=int, default=0)
    shortcut.add_argument("--literal-cheat-labels", action="store_true")
    shortcut.set_defaults(func=cmd_shortcut)

    extensions = sub.add_parser("extensions")
    extensions.add_argument("--out-dir", default="data")
    extensions.add_argument("--size", type=int, default=10_000)
    extensions.add_argument("--seed", type=int, default=0)
    extensions.set_defaults(func=cmd_extensions)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

