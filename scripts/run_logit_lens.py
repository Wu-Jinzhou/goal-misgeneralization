#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from gm_nim.logit_lens import run_logit_lens


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GPT-NeoX/Pythia logit-lens diagnostics.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-file")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--mr", type=int, required=True)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()
    prompt = args.prompt
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    if not prompt:
        raise SystemExit("provide --prompt or --prompt-file")
    run_logit_lens(
        model_path=args.model_path,
        prompt=prompt,
        output_csv=args.output_csv,
        mr=args.mr,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()

