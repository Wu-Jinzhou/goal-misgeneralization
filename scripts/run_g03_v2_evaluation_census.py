#!/usr/bin/env python3
"""Prepare or execute the registered G03-v2 evaluation census."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from goalzendo_interactive_v2.evaluation_census_execution import (
    EvaluationCensusExecutionV1Error,
    execute_production_evaluation_census_v1,
    prepare_production_evaluation_census_v1,
    verify_production_evaluation_census_execution_root_v1,
)


def _summary(value: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":")) + "\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Two-invocation, nonauthorizing lifecycle for the exact G03-v2 evaluation census. "
            "Prepare never executes; execute never prepares or registers."
        )
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="write an exact production plan and freeze request without executing it",
    )
    prepare.add_argument("--seed", required=True, help="exactly 64 lowercase hexadecimal characters")
    prepare.add_argument("--output-root", required=True, type=Path)
    prepare.add_argument("--source-archive", required=True, type=Path)
    prepare.add_argument("--constraints", required=True, type=Path)
    prepare.add_argument("--environment-lock", required=True, type=Path)

    execute = subparsers.add_parser(
        "execute",
        help="execute a separately prepared and externally registered production plan",
    )
    execute.add_argument("--plan", required=True, type=Path)
    execute.add_argument("--freeze-request", required=True, type=Path)
    execute.add_argument("--registration-receipt", required=True, type=Path)
    execute.add_argument("--expected-plan-digest", required=True)
    execute.add_argument("--expected-plan-sha256", required=True)
    execute.add_argument("--expected-registration-receipt-sha256", required=True)
    execute.add_argument("--expected-registration-reference", required=True)
    execute.add_argument("--expected-execution-uuid", required=True)
    execute.add_argument("--expected-execution-nonce", required=True)
    execute.add_argument("--output-root", required=True, type=Path)
    execute.add_argument(
        "--deadline-seconds",
        required=True,
        type=int,
        help="single wall-clock deadline shared by the fresh builder and verifier processes",
    )
    execute.add_argument("--source-archive", required=True, type=Path)
    execute.add_argument("--constraints", required=True, type=Path)
    execute.add_argument("--environment-lock", required=True, type=Path)

    verify = subparsers.add_parser(
        "verify",
        help="verify an externally pinned completed root and run a third fresh replay",
    )
    verify.add_argument("--output-root", required=True, type=Path)
    verify.add_argument("--plan", required=True, type=Path)
    verify.add_argument("--freeze-request", required=True, type=Path)
    verify.add_argument("--registration-receipt", required=True, type=Path)
    verify.add_argument("--expected-plan-digest", required=True)
    verify.add_argument("--expected-plan-sha256", required=True)
    verify.add_argument("--expected-registration-receipt-sha256", required=True)
    verify.add_argument("--expected-registration-reference", required=True)
    verify.add_argument("--expected-execution-uuid", required=True)
    verify.add_argument("--expected-execution-nonce", required=True)
    verify.add_argument("--expected-execution-deadline-seconds", required=True, type=int)
    verify.add_argument("--expected-execution-receipt-sha256", required=True)
    verify.add_argument("--expected-execution-receipt-digest", required=True)
    verify.add_argument("--verification-deadline-seconds", required=True, type=int)
    verify.add_argument("--source-archive", required=True, type=Path)
    verify.add_argument("--constraints", required=True, type=Path)
    verify.add_argument("--environment-lock", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    runner = Path(__file__).resolve(strict=True)
    try:
        if args.action == "prepare":
            prepared = prepare_production_evaluation_census_v1(
                args.seed,
                args.output_root,
                runner_source_path=runner,
                source_archive_path=args.source_archive,
                constraints_path=args.constraints,
                environment_lock_path=args.environment_lock,
            )
            plan_bytes = prepared.plan_path.read_bytes()
            request_bytes = prepared.freeze_request_path.read_bytes()
            _summary(
                {
                    "action": "prepare",
                    "status": "prospective_outcome_free_nonauthorizing",
                    "output_root": str(prepared.output_root),
                    "prospective_plan_digest": prepared.plan.digest,
                    "exact_plan_bytes_sha256": hashlib.sha256(plan_bytes).hexdigest(),
                    "exact_plan_byte_count": len(plan_bytes),
                    "freeze_request_digest": prepared.freeze_request.digest,
                    "exact_freeze_request_bytes_sha256": hashlib.sha256(request_bytes).hexdigest(),
                    "external_registration_required_before_execute": True,
                }
            )
            return 0
        if args.action == "verify":
            verified = verify_production_evaluation_census_execution_root_v1(
                output_root=args.output_root,
                plan_path=args.plan,
                freeze_request_path=args.freeze_request,
                registration_receipt_path=args.registration_receipt,
                expected_plan_digest=args.expected_plan_digest,
                expected_plan_bytes_sha256=args.expected_plan_sha256,
                expected_registration_receipt_sha256=(args.expected_registration_receipt_sha256),
                expected_registration_reference=args.expected_registration_reference,
                expected_execution_uuid=args.expected_execution_uuid,
                expected_execution_nonce=args.expected_execution_nonce,
                expected_execution_deadline_seconds=(args.expected_execution_deadline_seconds),
                expected_execution_receipt_sha256=(args.expected_execution_receipt_sha256),
                expected_execution_receipt_digest=(args.expected_execution_receipt_digest),
                runner_source_path=runner,
                verification_deadline_seconds=args.verification_deadline_seconds,
                source_archive_path=args.source_archive,
                constraints_path=args.constraints,
                environment_lock_path=args.environment_lock,
            )
            _summary(
                {
                    "action": "verify",
                    "status": "root_verified_with_third_fresh_replay_nonauthorizing",
                    "output_root": str(verified.output_root),
                    "observed_report_digest": verified.report_digest,
                    "exact_report_bytes_sha256": verified.report_bytes_sha256,
                    "execution_receipt_digest": verified.execution_receipt_digest,
                    "fresh_replay_terminal_digest": verified.fresh_replay_terminal_digest,
                    "fresh_replay_watchdog_terminal_digest": (verified.fresh_replay_watchdog_terminal_digest),
                    "verification_output_must_be_captured_and_externally_pinned": True,
                    "g01_authorized": False,
                    "launch_authorized": False,
                }
            )
            return 0
        if args.action == "execute":
            executed = execute_production_evaluation_census_v1(
                plan_path=args.plan,
                freeze_request_path=args.freeze_request,
                registration_receipt_path=args.registration_receipt,
                expected_plan_digest=args.expected_plan_digest,
                expected_plan_bytes_sha256=args.expected_plan_sha256,
                expected_registration_receipt_sha256=(args.expected_registration_receipt_sha256),
                expected_registration_reference=args.expected_registration_reference,
                expected_execution_uuid=args.expected_execution_uuid,
                expected_execution_nonce=args.expected_execution_nonce,
                output_root=args.output_root,
                runner_source_path=runner,
                deadline_seconds=args.deadline_seconds,
                controller_argv=tuple(sys.argv if argv is None else [sys.argv[0], *argv]),
                source_archive_path=args.source_archive,
                constraints_path=args.constraints,
                environment_lock_path=args.environment_lock,
            )
            _summary(
                {
                    "action": "execute",
                    "status": "complete_verified_nonauthorizing",
                    "output_root": str(executed.output_root),
                    "observed_report_digest": executed.report_digest,
                    "exact_report_bytes_sha256": executed.report_bytes_sha256,
                    "execution_receipt_digest": executed.execution_receipt_digest,
                    "registration_service_independently_verified_by_repository": False,
                    "g01_authorized": False,
                    "launch_authorized": False,
                }
            )
            return 0
    except EvaluationCensusExecutionV1Error as exc:
        parser.exit(2, f"error: {exc}\n")
    raise AssertionError("unreachable action")


if __name__ == "__main__":
    raise SystemExit(main())
