#!/usr/bin/env python3
"""Run one frozen, nonauthorizing G03-G full-model smoke cell."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from goalzendo_interactive.pinned_qwen_smoke_v1 import (
    CellKind,
    PinnedQwenSmokeCellConfig,
    run_pinned_qwen_smoke_cell,
    validate_pinned_qwen_smoke_path_separation,
    write_failure_marker,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", choices=("sft", "rl"), required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--episode-fixture", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--artifact-manifest-digest", required=True)
    parser.add_argument("--source-fingerprint", required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    cell = cast(CellKind, args.cell)
    config = PinnedQwenSmokeCellConfig(
        cell_kind=cell,
        artifact_root=args.artifact_root,
        episode_fixture_path=args.episode_fixture,
        output_root=args.output_root,
        expected_artifact_manifest_digest=args.artifact_manifest_digest,
        expected_source_fingerprint=args.source_fingerprint,
        device=args.device,
    )
    try:
        completion_digest = run_pinned_qwen_smoke_cell(config)
    except BaseException as exc:
        try:
            validate_pinned_qwen_smoke_path_separation(
                config.artifact_root,
                config.episode_fixture_path,
                config.output_root,
            )
        except BaseException:
            pass
        else:
            write_failure_marker(config.output_root, cell, exc)
        raise
    print(completion_digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
