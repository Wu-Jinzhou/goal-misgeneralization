#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gm_nim.hf import evaluate_bounded_jsonl, load_causal_lm, load_tokenizer, parse_checkpoint_step


def main() -> None:
    parser = argparse.ArgumentParser(description="Generation-evaluate bounded-Nim checkpoints.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--eval-files", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--condition", default="")
    parser.add_argument("--mr", type=int)
    parser.add_argument("--factors", nargs="*", type=int, default=[])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model)
    model = load_causal_lm(args.model)
    model.to("cuda" if __import__("torch").cuda.is_available() else "cpu")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("a", encoding="utf-8") as handle:
        for eval_file in args.eval_files:
            metrics, _rows = evaluate_bounded_jsonl(
                model,
                tokenizer,
                eval_file,
                mr=args.mr,
                factors=args.factors,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
            )
            handle.write(
                json.dumps(
                    {
                        "model": args.model,
                        "step": parse_checkpoint_step(args.model),
                        "condition": args.condition,
                        "eval_file": eval_file,
                        **metrics,
                    },
                    sort_keys=True,
                )
                + "\n"
            )


if __name__ == "__main__":
    main()

