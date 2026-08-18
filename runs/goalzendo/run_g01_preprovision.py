#!/usr/bin/env python3
"""Unconditionally refusing G01Q preprovision entrypoint."""

from __future__ import annotations

import sys

REFUSAL = "G01Q_REGISTRAR_NOT_DESIGNATED"


def main() -> int:
    print(f"run_g01_preprovision: error: {REFUSAL}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
