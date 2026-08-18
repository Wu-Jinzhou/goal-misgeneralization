"""Explicit command line for the additive, authenticated G00-F workflow."""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from goalzendo.artifacts import accelerator_metadata, package_manifest

from .evaluator import (
    WORKER_RESULT_SCHEMA,
    WORKER_RESULT_SCHEMA_VERSION,
    EvaluationError,
    create_gate,
    create_preexecution_gate,
    verify_gate_artifact,
)
from .freeze import (
    CONFIG_SPECS,
    LAUNCH_RECEIPT_SCHEMA,
    LAUNCH_RECEIPT_SCHEMA_VERSION,
    FreezeError,
    VerifiedFreeze,
    bind_runpod_provision_receipt,
    create_model_integration_audit,
    create_model_snapshot_receipt,
    create_profile_qualification_cleanup_receipt,
    create_profile_qualification_handoff,
    create_profile_selection_receipt,
    create_runpod_provision_receipt,
    create_runpod_ssh_identity_receipt,
    create_storage_preflight_receipt,
    exclusive_json,
    initialize_attempt_ledger,
    materialize_model_snapshot,
    reconcile_execution_failure,
    record_budget_timeout,
    run_worker,
    semantic_digest,
    sha256_file,
    strict_json,
    verify_attempt_ledger,
    verify_detached_supervisor_receipts,
    verify_freeze,
    verify_profile_qualification,
    verify_profile_qualification_cleanup_receipt,
    verify_profile_selection_receipt,
    verify_runpod_gpu_catalog_snapshot,
    verify_runpod_provision_receipt,
    verify_storage_preflight_receipt,
)


def _repo(start: str | Path | None) -> Path:
    current = Path(start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "goalzendo_g00f_h200").is_dir():
            return candidate
    raise FreezeError("could not locate the repository containing goalzendo_g00f_h200")


def _add_freeze(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo", type=Path, default=None)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--expected-freeze-sha256", required=True)


def _verified(args: argparse.Namespace) -> VerifiedFreeze:
    verified = verify_freeze(
        repo=_repo(args.repo),
        freeze_path=args.freeze,
        expected_freeze_sha256=args.expected_freeze_sha256,
    )
    selection = getattr(args, "profile_selection", None)
    if selection is not None:
        return verify_profile_selection_receipt(
            verified=verified,
            receipt_path=selection,
            expected_receipt_sha256=args.expected_profile_selection_sha256,
            expected_provision_receipt_sha256=args.expected_provision_receipt_sha256,
            expected_pod_id=args.expected_pod_id,
        )
    return verified


def _add_selected_freeze(parser: argparse.ArgumentParser) -> None:
    _add_freeze(parser)
    parser.add_argument("--profile-selection", type=Path, required=True)
    parser.add_argument("--expected-profile-selection-sha256", required=True)
    parser.add_argument("--expected-provision-receipt-sha256", required=True)
    parser.add_argument("--expected-pod-id", required=True)


def _print(value: Any) -> None:
    print(json.dumps(value, sort_keys=True, allow_nan=False))


def _package_versions(names: Sequence[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise FreezeError(f"frozen runtime package is absent: {name}") from error
    return versions


def _actual_visible_gpu() -> dict[str, str]:
    """Independently resolve the one CUDA-visible GPU through both stacks."""

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible or any(character.isspace() for character in visible):
        raise FreezeError("each G00-F worker must expose exactly one CUDA device token")
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--id",
            visible,
            "--query-gpu=name,uuid",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    rows = [row.strip() for row in completed.stdout.splitlines() if row.strip()]
    if completed.returncode != 0 or len(rows) != 1 or "," not in rows[0]:
        raise FreezeError("nvidia-smi did not resolve exactly one CUDA-visible GPU")
    gpu_name, gpu_uuid = (value.strip() for value in rows[0].split(",", maxsplit=1))
    try:
        import torch
    except ImportError as error:  # pragma: no cover - frozen GPU runtime dependency
        raise FreezeError("frozen torch dependency is absent") from error
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise FreezeError("torch does not see exactly one CUDA-visible GPU")
    torch_name = str(torch.cuda.get_device_name(0))
    if not gpu_name.startswith("NVIDIA H200") or "H200" not in torch_name:
        raise FreezeError("G00-F runtime is not backed by one NVIDIA H200")
    return {
        "cuda_visible_devices": visible,
        "gpu_name": gpu_name,
        "gpu_uuid": gpu_uuid,
        "torch_gpu_name": torch_name,
    }


def _create_launch_receipt(args: argparse.Namespace, verified: VerifiedFreeze) -> dict[str, Any]:
    worker_index = int(args.worker_index)
    if not 0 <= worker_index < 4:
        raise FreezeError("worker index must lie in [0,4)")
    ledger = verify_attempt_ledger(verified=verified, ledger_root=args.ledger_root)
    provision = verify_runpod_provision_receipt(
        verified=verified,
        receipt_path=args.provision_receipt,
        expected_receipt_sha256=args.expected_provision_receipt_sha256,
        expected_pod_id=args.expected_pod_id,
    )
    if ledger.get("runpod_provision") != provision:
        raise FreezeError("worker/ledger Runpod provision receipt binding changed")
    bundle = strict_json(args.bundle_receipt, "G00-F extracted source bundle receipt")
    bundle_body = {key: value for key, value in bundle.items() if key != "receipt_digest"}
    source_bundle = verified.payload["source_bundle"]
    execution_root = args.output.resolve().parent
    embedded_manifest = strict_json(
        verified.repo / "G00F-BUNDLE-MANIFEST.json",
        "embedded G00-F bundle manifest",
    )
    authenticated_runtime_files = bundle.get("authenticated_runtime_files")
    if (
        set(bundle_body)
        != {
            "archive_sha256",
            "authenticated_runtime_files",
            "extracted_root",
            "freeze_digest",
            "freeze_file_sha256",
            "g01_launch_authorized",
            "manifest_digest",
            "manifest_sha256",
            "member_count",
            "schema",
            "schema_version",
            "tar_safety",
        }
        or bundle.get("schema") != "goalzendo.g00f_h200_extracted_source_bundle_receipt"
        or bundle.get("schema_version") != 1
        or bundle.get("freeze_file_sha256") != verified.file_sha256
        or bundle.get("freeze_digest") != verified.digest
        or bundle.get("archive_sha256") != source_bundle["archive_sha256"]
        or bundle.get("manifest_sha256") != source_bundle["manifest_sha256"]
        or bundle.get("manifest_digest") != source_bundle["manifest_digest"]
        or bundle.get("extracted_root") != str(verified.repo)
        or Path(args.bundle_receipt).resolve() != execution_root / "source-bundle-receipt.json"
        or verified.repo != execution_root / "frozen-source"
        or bundle.get("member_count") != len(embedded_manifest.get("members", []))
        or bundle.get("tar_safety")
        != {
            "exact_bytes": True,
            "exact_member_set": True,
            "exact_modes": True,
            "no_absolute_or_parent_paths": True,
            "no_links": True,
            "only_regular_files": True,
        }
        or not isinstance(authenticated_runtime_files, Mapping)
        or set(authenticated_runtime_files)
        != {
            "bootstrap",
            "detached_supervisor",
            "launcher",
            "qualification_controller",
            "qualification_supervisor",
            "watchdog",
        }
        or bundle.get("g01_launch_authorized") is not False
        or bundle.get("receipt_digest") != semantic_digest(bundle_body)
    ):
        raise FreezeError("source bundle receipt is not bound to this execution freeze")
    if (
        args.output.resolve() != execution_root / f"worker-{worker_index}-launch.json"
        or Path(str(provision["path"])) != execution_root / "runpod-provision-receipt.json"
        or Path(str(ledger["ledger_root"])) != execution_root / "itt-ledger" / str(ledger["execution_uuid"])
    ):
        raise FreezeError("launch, provision, or ITT receipt lies outside the exact execution root")
    expected_packages = verified.payload["runtime"]["python_packages"]
    if not isinstance(expected_packages, Mapping):
        raise FreezeError("freeze runtime package map is malformed")
    packages = _package_versions(tuple(sorted(str(name) for name in expected_packages)))
    if packages != expected_packages:
        raise FreezeError("installed package versions differ from the frozen runtime")
    if args.image != verified.payload["runtime"]["image"]:
        raise FreezeError("runtime image literal differs from the freeze")
    if args.network_volume_id != verified.payload["runtime"]["network_volume_id"]:
        raise FreezeError("runtime network volume differs from the freeze")
    if args.data_center != verified.payload["runtime"]["data_center"]:
        raise FreezeError("runtime data center differs from the freeze")
    if platform.python_version() != verified.payload["runtime"]["python"]:
        raise FreezeError("runtime Python version differs from the frozen image stack")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise FreezeError("G00-F worker launch requires offline Hugging Face loading")
    actual_gpu = _actual_visible_gpu()
    if args.gpu_name != actual_gpu["gpu_name"] or args.gpu_uuid != actual_gpu["gpu_uuid"]:
        raise FreezeError("caller GPU name/UUID differs from the independently queried device")
    full_packages = package_manifest()
    accelerator = accelerator_metadata()
    devices = accelerator.get("cuda_devices")
    if (
        accelerator.get("torch_available") is not True
        or accelerator.get("cuda_available") is not True
        or accelerator.get("cuda_runtime") != "12.8"
        or accelerator.get("torch_version") != "2.8.0+cu128"
        or not isinstance(devices, Sequence)
        or len(devices) != 1
        or not isinstance(devices[0], Mapping)
        or "H200" not in str(devices[0].get("name", ""))
    ):
        raise FreezeError("pre-outcome accelerator inventory differs from the frozen CUDA stack")
    runtime_environment = {
        "accelerator": accelerator,
        "accelerator_digest": semantic_digest(accelerator),
        "installed_distributions": full_packages,
        "installed_distributions_digest": semantic_digest(full_packages),
        "lock_scope": "recorded_pre_outcome_runtime_evidence_not_complete_dependency_lock",
    }
    body = {
        "schema": LAUNCH_RECEIPT_SCHEMA,
        "schema_version": LAUNCH_RECEIPT_SCHEMA_VERSION,
        "execution_uuid": ledger["execution_uuid"],
        "worker_index": worker_index,
        "freeze_file_sha256": verified.file_sha256,
        "freeze_digest": verified.digest,
        "image": args.image,
        "network_volume_id": args.network_volume_id,
        "network_volume_mount": verified.payload["runtime"]["network_volume_mount"],
        "data_center": args.data_center,
        "gpu_family": "NVIDIA H200",
        "gpu_name": actual_gpu["gpu_name"],
        "gpu_uuid": actual_gpu["gpu_uuid"],
        "torch_gpu_name": actual_gpu["torch_gpu_name"],
        "visible_gpu_count": 1,
        "cuda_visible_devices": actual_gpu["cuda_visible_devices"],
        "concurrent_runs": 1,
        "packages": packages,
        "python": platform.python_version(),
        "runpod_provision": provision,
        "model_integration_audits": copy.deepcopy(dict(ledger["model_integration_audits"])),
        "profile_selection": copy.deepcopy(dict(ledger["profile_selection"])),
        "profile_qualification_cleanup": copy.deepcopy(dict(ledger["profile_qualification_cleanup"])),
        "selected_profile": ledger["selected_profile"],
        "runtime_environment": runtime_environment,
        "source_bundle": {
            "receipt_path": str(Path(args.bundle_receipt).resolve()),
            "receipt_file_sha256": sha256_file(args.bundle_receipt),
            "receipt_digest": bundle["receipt_digest"],
            "archive_sha256": bundle["archive_sha256"],
            "manifest_sha256": bundle["manifest_sha256"],
            "manifest_digest": bundle["manifest_digest"],
            "extracted_root": bundle["extracted_root"],
            "authenticated_runtime_files": bundle["authenticated_runtime_files"],
        },
        "ledger": {
            "ledger_root": ledger["ledger_root"],
            "path": ledger["path"],
            "file_sha256": ledger["file_sha256"],
            "ledger_digest": ledger["ledger_digest"],
            "budget_start_file_sha256": ledger["budget_start_file_sha256"],
        },
        "started_unix_ns": time.time_ns(),
        "started_monotonic_ns": time.monotonic_ns(),
        "offline_environment": {
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
        },
        "outcomes_seen": False,
        "g01_launch_authorized": False,
    }
    payload = {**body, "receipt_digest": semantic_digest(body)}
    exclusive_json(args.output, payload)
    return {**payload, "file_sha256": sha256_file(args.output), "path": str(args.output.resolve())}


def _worker_result(
    *,
    verified: VerifiedFreeze,
    worker_index: int,
    launch_path: Path,
    outcomes: Sequence[Mapping[str, Any]],
    state: str,
    exit_code: int,
    error_type: str | None,
    output: Path,
) -> dict[str, Any]:
    launch = strict_json(launch_path, "G00-F launch receipt for worker result")
    completed_monotonic_ns = time.monotonic_ns()
    elapsed_ns = completed_monotonic_ns - int(launch["started_monotonic_ns"])
    if elapsed_ns < 0:
        raise FreezeError("worker monotonic result time preceded its launch receipt")
    completed = sum(row.get("state") == "complete" for row in outcomes)
    failed = sum(row.get("state") in {"failed", "not_started_after_failure"} for row in outcomes)
    body = {
        "schema": WORKER_RESULT_SCHEMA,
        "schema_version": WORKER_RESULT_SCHEMA_VERSION,
        "execution_uuid": launch["execution_uuid"],
        "worker_index": worker_index,
        "freeze_file_sha256": verified.file_sha256,
        "freeze_digest": verified.digest,
        "profile_selection": copy.deepcopy(dict(verified.selection_receipt or {})),
        "profile_qualification_cleanup": copy.deepcopy(dict(launch.get("profile_qualification_cleanup", {}))),
        "selected_profile": verified.selected_profile,
        "state": state,
        "exit_code": exit_code,
        "error_type": error_type,
        "planned_runs": 40,
        "completed_runs": completed,
        "failed_runs": failed,
        "observed_outcome_rows": len(outcomes),
        "completed_monotonic_ns": completed_monotonic_ns,
        "wall_seconds": elapsed_ns / 1_000_000_000,
        "no_reassignment": True,
        "outcome_metrics_read": False,
        "predictions_read": False,
        "launch_receipt": {
            "path": str(launch_path.resolve()),
            "file_sha256": sha256_file(launch_path),
            "receipt_digest": launch["receipt_digest"],
        },
        "g01_launch_authorized": False,
    }
    payload = {**body, "receipt_digest": semantic_digest(body)}
    exclusive_json(output, payload)
    return {**payload, "file_sha256": sha256_file(output), "path": str(output.resolve())}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="goalzendo-g00f-h200")
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser("verify-freeze")
    _add_freeze(verify)

    plan = subparsers.add_parser("plan")
    _add_freeze(plan)
    plan.add_argument("--profile", choices=("baseline", "tuned"), default="baseline")
    plan.add_argument("--worker-index", type=int, default=None)

    ledger = subparsers.add_parser("initialize-ledger")
    _add_selected_freeze(ledger)
    ledger.add_argument("--execution-uuid", required=True)
    ledger.add_argument("--ledger-root", type=Path, required=True)
    ledger.add_argument("--provision-receipt", type=Path, required=True)
    ledger.add_argument("--model-integration-audit-0p5b", type=Path, required=True)
    ledger.add_argument("--model-integration-audit-1p5b", type=Path, required=True)
    ledger.add_argument("--profile-qualification-cleanup", type=Path, required=True)
    ledger.add_argument(
        "--expected-profile-qualification-cleanup-sha256",
        required=True,
    )

    provision = subparsers.add_parser("bind-provision-receipt")
    _add_freeze(provision)
    provision.add_argument("--input", type=Path, required=True)
    provision.add_argument("--output", type=Path, required=True)
    provision.add_argument("--expected-provision-receipt-sha256", required=True)
    provision.add_argument("--expected-pod-id", required=True)

    catalog_verify = subparsers.add_parser("verify-runpod-catalog")
    _add_freeze(catalog_verify)
    catalog_verify.add_argument("--raw-gpu-catalog", type=Path, required=True)
    catalog_verify.add_argument("--catalog-operator-capture-utc", required=True)

    ssh_probe = subparsers.add_parser("probe-runpod-ssh-identity")
    _add_freeze(ssh_probe)
    ssh_probe.add_argument("--raw-api-response", type=Path, required=True)
    ssh_probe.add_argument("--raw-ssh-info", type=Path, required=True)
    ssh_probe.add_argument("--expected-pod-id", required=True)
    ssh_probe.add_argument("--output", type=Path, required=True)

    provision_create = subparsers.add_parser("create-provision-receipt")
    _add_freeze(provision_create)
    provision_create.add_argument("--raw-runpodctl-version", type=Path, required=True)
    provision_create.add_argument("--raw-gpu-catalog", type=Path, required=True)
    provision_create.add_argument("--catalog-operator-capture-utc", required=True)
    provision_create.add_argument("--execution-uuid", required=True)
    provision_create.add_argument("--raw-create-response", type=Path, required=True)
    provision_create.add_argument("--raw-api-response", type=Path, required=True)
    provision_create.add_argument("--raw-ssh-info", type=Path, required=True)
    provision_create.add_argument("--ssh-identity-receipt", type=Path, required=True)
    provision_create.add_argument("--terminate-after-utc", required=True)
    provision_create.add_argument("--output", type=Path, required=True)

    qualification_bind = subparsers.add_parser("bind-profile-qualification")
    _add_freeze(qualification_bind)
    qualification_bind.add_argument("--execution-uuid", required=True)
    qualification_bind.add_argument("--producer-receipt", type=Path, required=True)
    qualification_bind.add_argument("--expected-producer-receipt-sha256", required=True)
    qualification_bind.add_argument("--expected-provision-receipt-sha256", required=True)
    qualification_bind.add_argument("--expected-pod-id", required=True)
    qualification_bind.add_argument("--gpu-uuid", action="append", required=True)
    qualification_bind.add_argument("--output", type=Path, required=True)

    qualification = subparsers.add_parser("verify-profile-qualification")
    _add_freeze(qualification)
    qualification.add_argument("--handoff", type=Path, required=True)
    qualification.add_argument("--expected-handoff-sha256", required=True)
    qualification.add_argument("--expected-provision-receipt-sha256", required=True)
    qualification.add_argument("--expected-pod-id", required=True)

    detached = subparsers.add_parser("verify-detached-supervisor")
    _add_freeze(detached)
    detached.add_argument("--execution-uuid", required=True)
    detached.add_argument("--execution-root", type=Path, required=True)
    detached.add_argument("--started-receipt", type=Path, required=True)
    detached.add_argument("--terminal-receipt", type=Path, required=True)
    detached.add_argument("--operator-handoff-sha256", required=True)
    detached.add_argument("--execution-handoff-sha256", required=True)
    detached.add_argument("--launcher", type=Path, required=True)
    detached.add_argument("--detached-supervisor", type=Path, required=True)

    storage = subparsers.add_parser("record-storage-preflight")
    _add_freeze(storage)
    storage.add_argument("--execution-uuid", required=True)
    storage.add_argument("--execution-root", type=Path, required=True)
    storage.add_argument("--phase", choices=("before_qualification", "before_itt"), required=True)
    storage.add_argument("--expected-provision-receipt-sha256", required=True)
    storage.add_argument("--expected-pod-id", required=True)
    storage.add_argument("--output", type=Path, required=True)

    storage_verify = subparsers.add_parser("verify-storage-preflight")
    _add_freeze(storage_verify)
    storage_verify.add_argument("--execution-uuid", required=True)
    storage_verify.add_argument("--execution-root", type=Path, required=True)
    storage_verify.add_argument("--phase", choices=("before_qualification", "before_itt"), required=True)
    storage_verify.add_argument("--receipt", type=Path, required=True)
    storage_verify.add_argument("--expected-receipt-sha256", required=True)
    storage_verify.add_argument("--expected-provision-receipt-sha256", required=True)
    storage_verify.add_argument("--expected-pod-id", required=True)

    selection = subparsers.add_parser("select-profile")
    _add_freeze(selection)
    selection.add_argument("--execution-uuid", required=True)
    selection.add_argument("--qualification-handoff", type=Path, required=True)
    selection.add_argument("--expected-qualification-handoff-sha256", required=True)
    selection.add_argument("--expected-provision-receipt-sha256", required=True)
    selection.add_argument("--expected-pod-id", required=True)
    selection.add_argument("--gpu-uuid", action="append", required=True)
    selection.add_argument("--output", type=Path, required=True)

    cleanup = subparsers.add_parser("record-profile-qualification-cleanup")
    _add_selected_freeze(cleanup)
    cleanup.add_argument("--output", type=Path, required=True)

    cleanup_verify = subparsers.add_parser("verify-profile-qualification-cleanup")
    _add_selected_freeze(cleanup_verify)
    cleanup_verify.add_argument("--cleanup", type=Path, required=True)
    cleanup_verify.add_argument("--expected-cleanup-sha256", required=True)

    model = subparsers.add_parser("model-receipt")
    _add_freeze(model)
    model.add_argument("--panel-id", choices=tuple(CONFIG_SPECS), required=True)
    model.add_argument("--snapshot-root", type=Path, required=True)
    model.add_argument("--output", type=Path, required=True)

    integration = subparsers.add_parser("model-integration-audit")
    _add_freeze(integration)
    integration.add_argument("--panel-id", choices=tuple(CONFIG_SPECS), required=True)
    integration.add_argument("--model-receipt", type=Path, required=True)
    integration.add_argument("--output", type=Path, required=True)

    materialize = subparsers.add_parser("materialize-model")
    _add_freeze(materialize)
    materialize.add_argument("--panel-id", choices=tuple(CONFIG_SPECS), required=True)
    materialize.add_argument("--output-root", type=Path, required=True)

    launch = subparsers.add_parser("launch-receipt")
    _add_selected_freeze(launch)
    launch.add_argument("--worker-index", type=int, required=True)
    launch.add_argument("--ledger-root", type=Path, required=True)
    launch.add_argument("--bundle-receipt", type=Path, required=True)
    launch.add_argument("--provision-receipt", type=Path, required=True)
    launch.add_argument("--image", required=True)
    launch.add_argument("--network-volume-id", required=True)
    launch.add_argument("--data-center", required=True)
    launch.add_argument("--gpu-name", required=True)
    launch.add_argument("--gpu-uuid", required=True)
    launch.add_argument("--output", type=Path, required=True)

    worker = subparsers.add_parser("run-worker")
    _add_selected_freeze(worker)
    worker.add_argument("--worker-index", type=int, required=True)
    worker.add_argument("--ledger-root", type=Path, required=True)
    worker.add_argument("--launch-receipt", type=Path, required=True)
    worker.add_argument("--model-receipt-0p5b", type=Path, required=True)
    worker.add_argument("--model-receipt-1p5b", type=Path, required=True)
    worker.add_argument("--result-output", type=Path, required=True)
    worker.add_argument("--dry-run", action="store_true")

    timeout = subparsers.add_parser("record-timeout")
    _add_selected_freeze(timeout)
    timeout.add_argument("--ledger-root", type=Path, required=True)

    reconcile = subparsers.add_parser("reconcile-failure")
    _add_selected_freeze(reconcile)
    reconcile.add_argument("--ledger-root", type=Path, required=True)
    reconcile.add_argument("--error-type", required=True)
    reconcile.add_argument(
        "--trigger",
        choices=(
            "coordinator_exit_trap",
            "launcher_partial_start",
            "launcher_signal",
            "watchdog_process_exit",
            "worker_nonzero_exit",
        ),
        required=True,
    )
    reconcile.add_argument("--cancel-receipt", type=Path, required=True)

    pre = subparsers.add_parser("preexecution-gate")
    _add_freeze(pre)
    pre.add_argument("--assessment-output", type=Path, required=True)
    pre.add_argument("--gate-output", type=Path, required=True)

    gate = subparsers.add_parser("gate")
    _add_selected_freeze(gate)
    gate.add_argument("--artifacts-0p5b", type=Path, required=True)
    gate.add_argument("--artifacts-1p5b", type=Path, required=True)
    gate.add_argument("--ledger-root", type=Path, required=True)
    for index in range(4):
        gate.add_argument(f"--worker-result-{index}", type=Path, required=True)
    gate.add_argument("--assessment-output", type=Path, required=True)
    gate.add_argument("--gate-output", type=Path, required=True)

    verify_gate = subparsers.add_parser("verify-gate")
    _add_selected_freeze(verify_gate)
    verify_gate.add_argument("--gate", type=Path, required=True)
    verify_gate.add_argument("--expected-gate-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        verified = _verified(args)
        if args.command == "verify-freeze":
            _print(
                {
                    "freeze_file_sha256": verified.file_sha256,
                    "freeze_digest": verified.digest,
                    "planned_runs": len(verified.all_rows),
                    "g01_launch_authorized": False,
                }
            )
            return 0
        if args.command == "plan":
            candidate_rows = tuple(
                sorted(
                    (row for panel in verified.candidate_plans[args.profile].values() for row in panel),
                    key=lambda row: (int(row["worker_index"]), int(row["worker_order"])),
                )
            )
            if args.worker_index is None:
                rows = candidate_rows
            else:
                if not 0 <= args.worker_index < 4:
                    raise FreezeError("worker index must lie in [0,4)")
                rows = tuple(row for row in candidate_rows if int(row["worker_index"]) == args.worker_index)
            for row in rows:
                _print(row)
            _print(
                {
                    "candidate_profile": args.profile,
                    "planned_runs": len(rows),
                    "selection_authorized": False,
                    "g01_launch_authorized": False,
                }
            )
            return 0
        if args.command == "initialize-ledger":
            _print(
                initialize_attempt_ledger(
                    verified=verified,
                    ledger_root=args.ledger_root,
                    execution_uuid=args.execution_uuid,
                    provision_receipt_path=args.provision_receipt,
                    expected_provision_receipt_sha256=args.expected_provision_receipt_sha256,
                    expected_pod_id=args.expected_pod_id,
                    model_integration_audit_paths={
                        "g00f-0p5b": args.model_integration_audit_0p5b,
                        "g00f-1p5b": args.model_integration_audit_1p5b,
                    },
                    profile_qualification_cleanup_path=args.profile_qualification_cleanup,
                    expected_profile_qualification_cleanup_sha256=(
                        args.expected_profile_qualification_cleanup_sha256
                    ),
                )
            )
            return 0
        if args.command == "bind-provision-receipt":
            _print(
                bind_runpod_provision_receipt(
                    verified=verified,
                    input_path=args.input,
                    output_path=args.output,
                    expected_receipt_sha256=args.expected_provision_receipt_sha256,
                    expected_pod_id=args.expected_pod_id,
                )
            )
            return 0
        if args.command == "verify-runpod-catalog":
            _print(
                verify_runpod_gpu_catalog_snapshot(
                    catalog_path=args.raw_gpu_catalog,
                    operator_capture_utc=args.catalog_operator_capture_utc,
                )
            )
            return 0
        if args.command == "probe-runpod-ssh-identity":
            _print(
                create_runpod_ssh_identity_receipt(
                    raw_api_response_path=args.raw_api_response,
                    raw_ssh_info_path=args.raw_ssh_info,
                    expected_pod_id=args.expected_pod_id,
                    output_path=args.output,
                )
            )
            return 0
        if args.command == "create-provision-receipt":
            _print(
                create_runpod_provision_receipt(
                    verified=verified,
                    raw_runpodctl_version_path=args.raw_runpodctl_version,
                    raw_gpu_catalog_path=args.raw_gpu_catalog,
                    catalog_operator_capture_utc=args.catalog_operator_capture_utc,
                    execution_uuid=args.execution_uuid,
                    raw_create_response_path=args.raw_create_response,
                    raw_api_response_path=args.raw_api_response,
                    raw_ssh_info_path=args.raw_ssh_info,
                    ssh_identity_receipt_path=args.ssh_identity_receipt,
                    terminate_after_utc=args.terminate_after_utc,
                    output_path=args.output,
                )
            )
            return 0
        if args.command == "bind-profile-qualification":
            _print(
                create_profile_qualification_handoff(
                    verified=verified,
                    execution_uuid=args.execution_uuid,
                    producer_receipt_path=args.producer_receipt,
                    expected_producer_receipt_sha256=args.expected_producer_receipt_sha256,
                    expected_provision_receipt_sha256=(args.expected_provision_receipt_sha256),
                    expected_pod_id=args.expected_pod_id,
                    observed_gpu_uuids=args.gpu_uuid,
                    output_path=args.output,
                )
            )
            return 0
        if args.command == "verify-profile-qualification":
            _print(
                verify_profile_qualification(
                    verified=verified,
                    handoff_path=args.handoff,
                    expected_handoff_sha256=args.expected_handoff_sha256,
                    expected_provision_receipt_sha256=args.expected_provision_receipt_sha256,
                    expected_pod_id=args.expected_pod_id,
                )
            )
            return 0
        if args.command == "verify-detached-supervisor":
            _print(
                verify_detached_supervisor_receipts(
                    verified=verified,
                    started_receipt_path=args.started_receipt,
                    terminal_receipt_path=args.terminal_receipt,
                    expected_execution_uuid=args.execution_uuid,
                    expected_execution_root=args.execution_root,
                    expected_operator_handoff_sha256=args.operator_handoff_sha256,
                    expected_execution_handoff_sha256=args.execution_handoff_sha256,
                    expected_launcher_path=args.launcher,
                    expected_supervisor_path=args.detached_supervisor,
                )
            )
            return 0
        if args.command == "record-storage-preflight":
            _print(
                create_storage_preflight_receipt(
                    verified=verified,
                    execution_uuid=args.execution_uuid,
                    execution_root=args.execution_root,
                    phase=args.phase,
                    expected_provision_receipt_sha256=args.expected_provision_receipt_sha256,
                    expected_pod_id=args.expected_pod_id,
                    output_path=args.output,
                )
            )
            return 0
        if args.command == "verify-storage-preflight":
            _print(
                verify_storage_preflight_receipt(
                    verified=verified,
                    execution_uuid=args.execution_uuid,
                    execution_root=args.execution_root,
                    phase=args.phase,
                    receipt_path=args.receipt,
                    expected_receipt_sha256=args.expected_receipt_sha256,
                    expected_provision_receipt_sha256=args.expected_provision_receipt_sha256,
                    expected_pod_id=args.expected_pod_id,
                )
            )
            return 0
        if args.command == "select-profile":
            selected = create_profile_selection_receipt(
                verified=verified,
                execution_uuid=args.execution_uuid,
                qualification_handoff_path=args.qualification_handoff,
                expected_qualification_handoff_sha256=(args.expected_qualification_handoff_sha256),
                expected_provision_receipt_sha256=args.expected_provision_receipt_sha256,
                expected_pod_id=args.expected_pod_id,
                observed_gpu_uuids=args.gpu_uuid,
                output_path=args.output,
            )
            _print(dict(selected.selection_receipt or {}))
            return 0
        if args.command == "record-profile-qualification-cleanup":
            _print(
                create_profile_qualification_cleanup_receipt(
                    verified=verified,
                    output_path=args.output,
                    expected_provision_receipt_sha256=args.expected_provision_receipt_sha256,
                    expected_pod_id=args.expected_pod_id,
                )
            )
            return 0
        if args.command == "verify-profile-qualification-cleanup":
            _print(
                verify_profile_qualification_cleanup_receipt(
                    verified=verified,
                    receipt_path=args.cleanup,
                    expected_receipt_sha256=args.expected_cleanup_sha256,
                    expected_provision_receipt_sha256=args.expected_provision_receipt_sha256,
                    expected_pod_id=args.expected_pod_id,
                )
            )
            return 0
        if args.command == "model-receipt":
            _print(
                create_model_snapshot_receipt(
                    verified=verified,
                    panel_id=args.panel_id,
                    snapshot_root=args.snapshot_root,
                    output=args.output,
                )
            )
            return 0
        if args.command == "model-integration-audit":
            _print(
                create_model_integration_audit(
                    verified=verified,
                    panel_id=args.panel_id,
                    model_receipt_path=args.model_receipt,
                    output=args.output,
                )
            )
            return 0
        if args.command == "materialize-model":
            _print(
                materialize_model_snapshot(
                    verified=verified,
                    panel_id=args.panel_id,
                    output_root=args.output_root,
                )
            )
            return 0
        if args.command == "launch-receipt":
            _print(_create_launch_receipt(args, verified))
            return 0
        if args.command == "run-worker":
            outcomes: tuple[Mapping[str, Any], ...] = ()
            try:
                outcomes = run_worker(
                    verified=verified,
                    worker_index=args.worker_index,
                    ledger_root=args.ledger_root,
                    launch_receipt_path=args.launch_receipt,
                    model_receipt_paths={
                        "g00f-0p5b": args.model_receipt_0p5b,
                        "g00f-1p5b": args.model_receipt_1p5b,
                    },
                    dry_run=args.dry_run,
                )
            except BaseException as error:
                if not args.dry_run:
                    result = _worker_result(
                        verified=verified,
                        worker_index=args.worker_index,
                        launch_path=args.launch_receipt,
                        outcomes=outcomes,
                        state="failed",
                        exit_code=1,
                        error_type=type(error).__name__,
                        output=args.result_output,
                    )
                    _print(result)
                raise
            if args.dry_run:
                for row in outcomes:
                    _print(row)
                return 0
            state = (
                "complete"
                if len(outcomes) == 40 and all(row["state"] == "complete" for row in outcomes)
                else "failed"
            )
            result = _worker_result(
                verified=verified,
                worker_index=args.worker_index,
                launch_path=args.launch_receipt,
                outcomes=outcomes,
                state=state,
                exit_code=0 if state == "complete" else 1,
                error_type=None,
                output=args.result_output,
            )
            _print(result)
            return 0 if state == "complete" else 1
        if args.command == "record-timeout":
            _print(record_budget_timeout(verified=verified, ledger_root=args.ledger_root))
            return 1
        if args.command == "reconcile-failure":
            _print(
                reconcile_execution_failure(
                    verified=verified,
                    ledger_root=args.ledger_root,
                    error_type=args.error_type,
                    trigger=args.trigger,
                    cancel_receipt=args.cancel_receipt,
                )
            )
            return 1
        if args.command == "preexecution-gate":
            result = create_preexecution_gate(
                verified=verified,
                assessment_output=args.assessment_output,
                gate_output=args.gate_output,
            )
            _print(result)
            return 1
        if args.command == "gate":
            result = create_gate(
                verified=verified,
                artifact_roots={
                    "g00f-0p5b": args.artifacts_0p5b,
                    "g00f-1p5b": args.artifacts_1p5b,
                },
                ledger_root=args.ledger_root,
                worker_result_receipts={index: getattr(args, f"worker_result_{index}") for index in range(4)},
                expected_provision_receipt_sha256=args.expected_provision_receipt_sha256,
                expected_pod_id=args.expected_pod_id,
                assessment_output=args.assessment_output,
                gate_output=args.gate_output,
            )
            _print(result)
            return 0 if result["gate"]["overall_passed"] else 1
        if args.command == "verify-gate":
            _print(
                verify_gate_artifact(
                    args.gate,
                    verified=verified,
                    expected_gate_sha256=args.expected_gate_sha256,
                )
            )
            return 0
        parser.error(f"unknown G00-F command: {args.command}")
    except (FreezeError, EvaluationError, OSError, ValueError) as error:
        print(f"goalzendo-g00f-h200: error: {error}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
