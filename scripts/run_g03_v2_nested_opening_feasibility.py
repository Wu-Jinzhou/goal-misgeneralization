#!/usr/bin/env python3
# ruff: noqa: E402,I001
"""Prepare the prospective G03-v2 nested-opening feasibility plan pair.

Production execution and root verification are intentionally unavailable in
this tranche.  The refusal below runs before importing any project package.
"""

from __future__ import annotations

import sys


PRODUCTION_EXECUTION_REFUSAL_CODE = (
    "G03_V2_NESTED_RECONSTRUCTION_CONTRACT_NOT_VERIFIED"
)


if len(sys.argv) > 1 and sys.argv[1] in ("execute", "verify"):
    sys.stderr.write(f"error: {PRODUCTION_EXECUTION_REFUSAL_CODE}\n")
    raise SystemExit(2)


import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from goalzendo_interactive_v2.nested_opening_feasibility_execution import (
    NestedOpeningFeasibilityExecutionV1Error,
    prepare_production_nested_opening_feasibility_v1,
)


def _summary(value: dict[str, Any]) -> None:
    sys.stdout.write(
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the exact prospective G03-v2 nested-opening feasibility "
            "plan pair. Production execute/verify are held pending an audited "
            "transitive reconstruction contract and external registrar."
        )
    )
    subparsers = parser.add_subparsers(dest="action", required=True)
    prepare = subparsers.add_parser(
        "prepare",
        help="persist both exact plans and an outcome-free freeze request",
    )
    prepare.add_argument("--upstream-census-seed", required=True)
    prepare.add_argument("--nested-generator-seed", required=True)
    prepare.add_argument("--output-root", required=True, type=Path)
    prepare.add_argument("--source-archive", required=True, type=Path)
    prepare.add_argument("--constraints", required=True, type=Path)
    prepare.add_argument("--environment-lock", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw = tuple(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] in ("execute", "verify"):
        sys.stderr.write(f"error: {PRODUCTION_EXECUTION_REFUSAL_CODE}\n")
        return 2
    parser = _parser()
    args = parser.parse_args(raw)
    runner = Path(__file__).resolve(strict=True)
    try:
        prepared = prepare_production_nested_opening_feasibility_v1(
            args.upstream_census_seed,
            args.nested_generator_seed,
            args.output_root,
            runner_source_path=runner,
            source_archive_path=args.source_archive,
            constraints_path=args.constraints,
            environment_lock_path=args.environment_lock,
        )
        _summary(
            {
                "action": "prepare",
                "status": "prospective_outcome_free_nonauthorizing",
                "output_root": str(prepared.output_root),
                "upstream_evaluation_census_plan_digest": (
                    prepared.upstream_census_plan_digest
                ),
                "exact_upstream_plan_bytes_sha256": (
                    prepared.upstream_census_plan_bytes_sha256
                ),
                "exact_upstream_plan_byte_count": (
                    prepared.upstream_census_plan_byte_count
                ),
                "nested_opening_feasibility_plan_digest": (
                    prepared.nested_plan_digest
                ),
                "exact_nested_plan_bytes_sha256": (
                    prepared.nested_plan_bytes_sha256
                ),
                "exact_nested_plan_byte_count": prepared.nested_plan_byte_count,
                "freeze_request_digest": prepared.freeze_request_digest,
                "exact_freeze_request_bytes_sha256": (
                    prepared.freeze_request_bytes_sha256
                ),
                "exact_freeze_request_byte_count": (
                    prepared.freeze_request_byte_count
                ),
                "separate_external_registration_required": True,
                "production_execute_available": False,
                "production_execute_refusal_code": (
                    PRODUCTION_EXECUTION_REFUSAL_CODE
                ),
                "source_archive_reconstructability_verified_by_repository": False,
                "environment_reconstruction_verified_by_repository": False,
                "g01_authorized": False,
                "launch_authorized": False,
            }
        )
        return 0
    except NestedOpeningFeasibilityExecutionV1Error as exc:
        parser.exit(2, f"error: {exc}\n")
    raise AssertionError("unreachable action")


if __name__ == "__main__":
    raise SystemExit(main())
