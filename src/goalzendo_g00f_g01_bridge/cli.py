"""Source-only CLI for checkpoint-A G00-F to G01 scientific eligibility."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .bridge import (
    BridgeError,
    calculate_bridge_source_binding,
    create_route_lock,
    execution_layout,
    produce_eligibility,
    project_coordinator_token,
    require_entrypoint_file,
    verify_coordinator_token,
    verify_eligibility,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="goalzendo-g00f-g01-eligibility")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest="command", required=True)

    source = commands.add_parser("source-binding")
    source.set_defaults(handler=_source_binding)

    lock = commands.add_parser("route-lock")
    lock.add_argument("--route", choices=("h100", "h200"), required=True)
    lock.add_argument("--execution-uuid", required=True)
    lock.add_argument("--expected-freeze-sha256", required=True)
    lock.add_argument("--expected-bridge-source-digest", required=True)
    lock.set_defaults(handler=_route_lock)

    assess = commands.add_parser("assess-eligibility")
    assess.add_argument("--route", choices=("h100", "h200"), required=True)
    assess.add_argument("--execution-uuid", required=True)
    assess.add_argument("--expected-route-lock-sha256", required=True)
    assess.add_argument("--expected-freeze-sha256", required=True)
    assess.add_argument("--expected-final-gate-sha256", required=True)
    assess.add_argument("--expected-provision-receipt-sha256", required=True)
    assess.add_argument("--expected-pod-id", required=True)
    assess.add_argument("--expected-profile-selection-sha256")
    assess.add_argument("--expected-bridge-source-digest", required=True)
    assess.set_defaults(handler=_assess_eligibility)

    verify = commands.add_parser("verify-eligibility")
    verify.add_argument("--route", choices=("h100", "h200"), required=True)
    verify.add_argument("--execution-uuid", required=True)
    verify.add_argument("--expected-eligibility-sha256", required=True)
    verify.add_argument("--expected-route-lock-sha256", required=True)
    verify.add_argument("--expected-bridge-source-digest", required=True)
    verify.add_argument("--evidence-level", choices=("full", "metadata"), default="full")
    verify.set_defaults(handler=_verify_eligibility)

    project = commands.add_parser("project-token")
    project.add_argument("--route", choices=("h100", "h200"), required=True)
    project.add_argument("--execution-uuid", required=True)
    project.add_argument("--expected-eligibility-sha256", required=True)
    project.add_argument("--expected-route-lock-sha256", required=True)
    project.add_argument("--expected-bridge-source-digest", required=True)
    project.set_defaults(handler=_project_token)

    token = commands.add_parser("verify-token")
    token.add_argument("--route", choices=("h100", "h200"), required=True)
    token.add_argument("--execution-uuid", required=True)
    token.add_argument("--expected-token-sha256", required=True)
    token.add_argument("--expected-bridge-source-digest", required=True)
    token.set_defaults(handler=_verify_token)
    return parser


def _layout(args: argparse.Namespace) -> dict[str, object]:
    return execution_layout(args.repo, args.execution_uuid, args.route)


def _source_binding(args: argparse.Namespace) -> dict[str, object]:
    return calculate_bridge_source_binding(args.repo)


def _route_lock(args: argparse.Namespace) -> dict[str, object]:
    layout = _layout(args)
    return create_route_lock(
        repo=args.repo,
        route=args.route,
        execution_uuid=args.execution_uuid,
        freeze_path=Path(str(layout["freeze"])),
        expected_freeze_sha256=args.expected_freeze_sha256,
        ledger_root=Path(str(layout["ledger_root"])),
        prospective_gate_output=Path(str(layout["final_gate"])),
        expected_bridge_source_digest=args.expected_bridge_source_digest,
        output=Path(str(layout["route_lock"])),
    )


def _assess_eligibility(args: argparse.Namespace) -> dict[str, object]:
    layout = _layout(args)
    profile = Path(str(layout["profile_selection"])) if args.route == "h200" else None
    if args.route == "h100" and args.expected_profile_selection_sha256 is not None:
        raise BridgeError("H100 eligibility forbids an H200 profile-selection digest")
    return produce_eligibility(
        repo=args.repo,
        route=args.route,
        route_lock_path=Path(str(layout["route_lock"])),
        expected_route_lock_sha256=args.expected_route_lock_sha256,
        freeze_path=Path(str(layout["freeze"])),
        expected_freeze_sha256=args.expected_freeze_sha256,
        final_gate_path=Path(str(layout["final_gate"])),
        expected_final_gate_sha256=args.expected_final_gate_sha256,
        artifact_roots=None,
        ledger_root=Path(str(layout["ledger_root"])),
        worker_result_receipts=None,
        expected_provision_receipt_sha256=args.expected_provision_receipt_sha256,
        expected_pod_id=args.expected_pod_id,
        profile_selection_path=profile,
        expected_profile_selection_sha256=args.expected_profile_selection_sha256,
        expected_bridge_source_digest=args.expected_bridge_source_digest,
        output=Path(str(layout["eligibility"])),
    )


def _verify_eligibility(args: argparse.Namespace) -> dict[str, object]:
    layout = _layout(args)
    verified = verify_eligibility(
        Path(str(layout["eligibility"])),
        repo=args.repo,
        expected_eligibility_sha256=args.expected_eligibility_sha256,
        expected_route_lock_sha256=args.expected_route_lock_sha256,
        expected_bridge_source_digest=args.expected_bridge_source_digest,
        evidence_level=args.evidence_level,
    )
    return {
        "path": str(verified.path),
        "file_sha256": verified.file_sha256,
        "route_lock_sha256": verified.route_lock_sha256,
        "bridge_source_digest": verified.bridge_source_digest,
        "route": verified.route,
        "execution_uuid": verified.execution_uuid,
        "selected_profile": verified.selected_profile,
        "plan_rows_digest": verified.plan_rows_digest,
        "plan_key_set_digest": verified.plan_key_set_digest,
        "target_binding_digest": verified.target_binding_digest,
        "model_name": verified.model_name,
        "model_revision": verified.model_revision,
        "scope": verified.scope,
        "g01_scientifically_eligible": verified.g01_scientifically_eligible,
        "direct_g01_launch_authorized": verified.direct_g01_launch_authorized,
        "dedicated_global_coordinator_required": (verified.dedicated_global_coordinator_required),
    }


def _project_token(args: argparse.Namespace) -> dict[str, object]:
    layout = _layout(args)
    return project_coordinator_token(
        Path(str(layout["eligibility"])),
        repo=args.repo,
        expected_eligibility_sha256=args.expected_eligibility_sha256,
        expected_route_lock_sha256=args.expected_route_lock_sha256,
        expected_bridge_source_digest=args.expected_bridge_source_digest,
        output=Path(str(layout["coordinator_token"])),
    )


def _verify_token(args: argparse.Namespace) -> dict[str, object]:
    layout = _layout(args)
    verified = verify_coordinator_token(
        Path(str(layout["coordinator_token"])),
        repo=args.repo,
        expected_token_sha256=args.expected_token_sha256,
        expected_bridge_source_digest=args.expected_bridge_source_digest,
    )
    return {
        "path": str(verified.path),
        "file_sha256": verified.file_sha256,
        "token_digest": verified.token_digest,
        "eligibility_file_sha256": verified.eligibility_file_sha256,
        "eligibility_digest": verified.eligibility_digest,
        "route_lock_sha256": verified.route_lock_sha256,
        "route_lock_digest": verified.route_lock_digest,
        "bridge_source_digest": verified.bridge_source_digest,
        "route": verified.route,
        "execution_uuid": verified.execution_uuid,
        "target_binding_digest": verified.target_binding_digest,
        "plan_rows_digest": verified.plan_rows_digest,
        "plan_key_set_digest": verified.plan_key_set_digest,
        "g01_scientifically_eligible": verified.g01_scientifically_eligible,
        "direct_g01_launch_authorized": verified.direct_g01_launch_authorized,
        "dedicated_global_coordinator_required": (verified.dedicated_global_coordinator_required),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        require_entrypoint_file(
            args.repo,
            __file__,
            "src/goalzendo_g00f_g01_bridge/cli.py",
        )
        result = args.handler(args)
    except BridgeError as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
