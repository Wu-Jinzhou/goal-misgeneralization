#!/usr/bin/env python3
"""Unconditionally refusing G01Q production entrypoint."""

from __future__ import annotations

import sys

REFUSAL = "G01Q_RUNTIME_PROVISION_NOT_FROZEN"


def main() -> int:
    print(f"run_g01_compute_qualification: error: {REFUSAL}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
