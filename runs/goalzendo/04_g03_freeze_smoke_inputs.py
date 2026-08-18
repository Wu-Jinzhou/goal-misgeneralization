#!/usr/bin/env python3
"""Freeze G03-G artifact/source inputs without loading the model."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from goalzendo_interactive.pinned_qwen_smoke_v1 import (
    build_pinned_qwen_smoke_preflight,
    validate_pinned_qwen_smoke_path_separation,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--episode-fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output: Path = args.output
    validate_pinned_qwen_smoke_path_separation(
        args.artifact_root,
        args.episode_fixture,
        output,
    )
    if output.exists() or output.is_symlink():
        raise FileExistsError("preflight output must be fresh and absent")
    rendered = build_pinned_qwen_smoke_preflight(
        args.artifact_root,
        args.episode_fixture,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(rendered + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
