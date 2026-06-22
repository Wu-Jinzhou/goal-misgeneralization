#!/usr/bin/env python
from __future__ import annotations

import argparse

from gm_nim.probes import run_probe_grid


def main() -> None:
    parser = argparse.ArgumentParser(description="Run shortcut MLP probe grid.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--layers", nargs="*", type=int)
    parser.add_argument("--max-examples", type=int, default=5_000)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_probe_grid(**vars(args))


if __name__ == "__main__":
    main()

