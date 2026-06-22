#!/usr/bin/env python
from __future__ import annotations

import argparse

from gm_nim.causal import run_causal_trace


def main() -> None:
    parser = argparse.ArgumentParser(description="Run name-token or final-token causal trace.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--mr", type=int, default=4)
    parser.add_argument("--layers", nargs="*", type=int)
    parser.add_argument("--max-examples", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--position-mode", choices=["name", "final"], default="name")
    args = parser.parse_args()
    run_causal_trace(**vars(args))


if __name__ == "__main__":
    main()

