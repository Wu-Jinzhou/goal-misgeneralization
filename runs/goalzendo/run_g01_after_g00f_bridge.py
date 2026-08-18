#!/usr/bin/env python3
"""Checkpoint-A eligibility verifier that always refuses direct G01 execution."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from goalzendo_g00f_g01_bridge import (
    BridgeError,
    assert_thin_runtime_boundary,
    exact_g01_plan,
    require_entrypoint_file,
    verify_coordinator_token,
)

_COORDINATOR_TOKEN_FILENAME = "g00f-g01-coordinator-input.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run-g01-after-g00f-bridge",
        description=(
            "Verify checkpoint-A scientific eligibility and refuse direct execution. "
            "A separately frozen global coordinator checkpoint B is required."
        ),
    )
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--expected-coordinator-token-sha256", required=True)
    parser.add_argument("--expected-bridge-source-digest", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo = Path(args.repo).absolute()
    token = repo.parents[1] / _COORDINATOR_TOKEN_FILENAME
    try:
        assert_thin_runtime_boundary()
        require_entrypoint_file(repo, __file__, "runs/goalzendo/run_g01_after_g00f_bridge.py")
        verified = verify_coordinator_token(
            token,
            repo=repo,
            expected_token_sha256=args.expected_coordinator_token_sha256,
            expected_bridge_source_digest=args.expected_bridge_source_digest,
        )
        plan = exact_g01_plan(repo)
        if len(plan) != 120 or verified.g01_scientifically_eligible is not True:
            raise BridgeError("checkpoint-A eligibility or exact G01 membership changed")
        assert_thin_runtime_boundary()
        raise BridgeError(
            "direct G01 execution is prohibited: checkpoint A establishes scientific "
            "eligibility only; a separately frozen global coordinator checkpoint B is required"
        )
    except BridgeError as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
