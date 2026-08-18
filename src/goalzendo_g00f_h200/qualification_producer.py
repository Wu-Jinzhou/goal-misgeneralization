"""Authenticated actual-model producer for the pre-ITT H200 qualification.

The controller accepts only freeze-bound inputs, authenticates them before any
model load, discovers exactly four physical H200 UUIDs, and launches one worker
per UUID.  Workers produce canonical process/comparison shards.  The
controller combines those shards into compact evidence, independently replays
it through :mod:`goalzendo_g00f_h200.qualification`, and writes immutable
evidence, report, transient-inventory, and producer-receipt artifacts.

The default worker is deliberately an actual PyTorch/Transformers execution
path.  Dependency injection exists only at the Python API boundary so CPU unit
tests can exercise orchestration and fail-closed validation without weakening
the CLI or admitting operator-supplied evidence JSON.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, cast

import torch
from torch import Tensor, nn

from goalzendo.artifacts import RunStore, verify_completion_attestation
from goalzendo.config import load_config
from goalzendo.experiment import (
    configure_numerical_execution,
    materialize_banks,
    render_experiment,
    render_prompt_view,
)
from goalzendo.modeling import TwoActionScorer, encode_action_continuations, format_chat_prompt
from goalzendo.runner import build_plan, derived_seeds
from goalzendo.training import (
    _step_rng,
    build_optimizer,
    constant_with_warmup_multiplier,
    deterministic_batch_indices,
    sft_action_loss,
    train_steps,
)

from .freeze import (
    PROFILE_CONFIG_SPECS,
    PROFILE_QUALIFICATION_PRODUCER_SCHEMA,
    PROFILE_QUALIFICATION_PRODUCER_SCHEMA_VERSION,
    VerifiedFreeze,
    verify_freeze,
    verify_model_integration_audit,
    verify_model_snapshot_receipt,
    verify_runpod_provision_receipt,
)
from .qualification import (
    ACTION_LABELS,
    COMPARISON_SCHEMA,
    COMPARISON_SCHEMA_VERSION,
    ENGINEERING_ROOT_PREFIX,
    EVIDENCE_BOUNDARY,
    EVIDENCE_SCHEMA,
    EVIDENCE_SCHEMA_VERSION,
    LAW_FAMILIES,
    PANELS,
    PROCESS_SCHEMA,
    PROCESS_SCHEMA_VERSION,
    PROFILE_CONTRACT,
    PROFILES,
    REPLICATES,
    TRAINABLE_NUMEL,
    TRAINING_VIEWS,
    TUNED_CAPACITY_DISQUALIFIERS,
    TUNED_CAPACITY_PROBE_SCHEMA,
    TUNED_CAPACITY_PROBE_SCHEMA_VERSION,
    UPDATES,
    VECTOR_KINDS,
    QualificationError,
    build_paired_execution_receipt,
    canonical_json_bytes,
    canonical_process_comparison_binding,
    seal_comparison_record,
    seal_evidence,
    seal_process_record,
    seal_tuned_capacity_probe,
    semantic_digest,
    streaming_metric_from_accumulator_rows,
    validate_qualification_evidence,
    validate_qualification_report,
    validate_tuned_capacity_probe,
    write_qualification_report,
)

PRODUCER_SCHEMA = PROFILE_QUALIFICATION_PRODUCER_SCHEMA
PRODUCER_SCHEMA_VERSION = PROFILE_QUALIFICATION_PRODUCER_SCHEMA_VERSION
WORKER_REQUEST_SCHEMA = "goalzendo.g00f_h200_profile_qualification_worker_request"
WORKER_REQUEST_SCHEMA_VERSION = 1
WORKER_SHARD_SCHEMA = "goalzendo.g00f_h200_profile_qualification_worker_shard"
WORKER_SHARD_SCHEMA_VERSION = 1
WORKER_RECEIPT_SCHEMA = "goalzendo.g00f_h200_profile_qualification_worker_receipt"
WORKER_RECEIPT_SCHEMA_VERSION = 1
CAPACITY_CELL_RESULT_SCHEMA = "goalzendo.g00f_h200_tuned_capacity_probe_cell_result"
CAPACITY_CELL_RESULT_SCHEMA_VERSION = 1
TRANSIENT_INVENTORY_SCHEMA = "goalzendo.g00f_h200_profile_qualification_transient_inventory"
TRANSIENT_INVENTORY_SCHEMA_VERSION = 1

CONTROLLER_RELATIVE = "runs/goalzendo/g00f_h200_qualification_controller.py"
IMPLEMENTATION_RELATIVE = "src/goalzendo_g00f_h200/qualification_producer.py"
EVIDENCE_NAME = "qualification-evidence.json"
REPORT_NAME = "profile-qualification.json"
PRODUCER_RECEIPT_NAME = "qualification-producer-receipt.json"
TRANSIENT_INVENTORY_NAME = "qualification-transient-inventory.json"
TRANSIENT_ROOT_NAME = "transient-raw"
TRANSIENT_CEILING_BYTES = 1024**4
PERSISTED_EVIDENCE_CEILING_BYTES = 512 * 1024**2
CORPUS_SIZE = 10_000
CORPUS_SEEDS: Mapping[str, int] = {"majority": 8_611_101, "parity": 8_611_103}
ENGINEERING_SEED = 8_611_107
EVALUATION_BOUNDARY_COUNT = 13
NATIVE_SCAN_CHUNK_ELEMENTS = 1_048_576
GPU_UUID_PATTERN = re.compile(r"^GPU-[A-Za-z0-9][A-Za-z0-9-]{7,}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
PRODUCER_CEILING_SECONDS = 6_300
POST_QUALIFICATION_RESERVE_SECONDS = 600
LEDGER_CEILING_SECONDS = 50_400
PROVISION_GRACE_SECONDS = 60
MINIMUM_PROVISION_REMAINING_AT_START_SECONDS = 57_360
PROVISION_START_GUARD_MESSAGE = "provision leaves less than 57,360 seconds at producer start"
HOST_ENVELOPE_ORDER = (
    "contextual_token_work_variability",
    "utf8_byte_work_variability",
    "contextual_token_work_dominance",
    "utf8_byte_work_dominance",
)


class ProducerError(RuntimeError):
    """Raised when producer authentication or actual execution fails closed."""


class NonFiniteCapacityError(ProducerError):
    """Exact recoverable tuned-capacity non-finite signal."""


def _canonical_utc_from_timestamp(value: float) -> str:
    return (
        datetime.fromtimestamp(value, tz=timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _utc_timestamp(value: str, label: str) -> float:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ProducerError(f"{label} is not canonical UTC") from error
    _require(parsed.tzinfo is not None, f"{label} is not timezone-aware")
    return parsed.timestamp()


def _deadline_binding(
    *,
    started_wall_seconds: float,
    started_monotonic_seconds: float,
    completed_wall_seconds: float,
    completed_monotonic_seconds: float,
    provision_terminate_after_utc: str,
) -> dict[str, Any]:
    remaining = _utc_timestamp(provision_terminate_after_utc, "provision termination") - float(
        started_wall_seconds
    )
    elapsed = float(completed_monotonic_seconds) - float(started_monotonic_seconds)
    body = {
        "producer_ceiling_seconds": PRODUCER_CEILING_SECONDS,
        "post_qualification_reserve_seconds": POST_QUALIFICATION_RESERVE_SECONDS,
        "ledger_ceiling_seconds": LEDGER_CEILING_SECONDS,
        "grace_seconds": PROVISION_GRACE_SECONDS,
        "minimum_provision_remaining_at_start_seconds": (MINIMUM_PROVISION_REMAINING_AT_START_SECONDS),
        "provision_terminate_after_utc": provision_terminate_after_utc,
        "producer_started_at_utc": _canonical_utc_from_timestamp(started_wall_seconds),
        "producer_completed_at_utc": _canonical_utc_from_timestamp(completed_wall_seconds),
        "producer_started_monotonic_seconds": float(started_monotonic_seconds),
        "producer_deadline_monotonic_seconds": (float(started_monotonic_seconds) + PRODUCER_CEILING_SECONDS),
        "producer_completed_monotonic_seconds": float(completed_monotonic_seconds),
        "provision_remaining_at_start_seconds": remaining,
        "producer_elapsed_seconds": elapsed,
        "completed_within_ceiling": elapsed <= PRODUCER_CEILING_SECONDS,
    }
    _validate_deadline_binding(body)
    return body


def _validate_deadline_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "producer_ceiling_seconds",
        "post_qualification_reserve_seconds",
        "ledger_ceiling_seconds",
        "grace_seconds",
        "minimum_provision_remaining_at_start_seconds",
        "provision_terminate_after_utc",
        "producer_started_at_utc",
        "producer_completed_at_utc",
        "producer_started_monotonic_seconds",
        "producer_deadline_monotonic_seconds",
        "producer_completed_monotonic_seconds",
        "provision_remaining_at_start_seconds",
        "producer_elapsed_seconds",
        "completed_within_ceiling",
    }
    _require(set(value) == expected, "producer deadline binding fields changed")
    _require(
        value["producer_ceiling_seconds"] == PRODUCER_CEILING_SECONDS
        and value["post_qualification_reserve_seconds"] == POST_QUALIFICATION_RESERVE_SECONDS
        and value["ledger_ceiling_seconds"] == LEDGER_CEILING_SECONDS
        and value["grace_seconds"] == PROVISION_GRACE_SECONDS
        and value["minimum_provision_remaining_at_start_seconds"]
        == MINIMUM_PROVISION_REMAINING_AT_START_SECONDS,
        "producer deadline constants changed",
    )
    started_wall = _utc_timestamp(str(value["producer_started_at_utc"]), "producer start UTC")
    completed_wall = _utc_timestamp(str(value["producer_completed_at_utc"]), "producer completion UTC")
    terminate_wall = _utc_timestamp(str(value["provision_terminate_after_utc"]), "provision termination UTC")
    started_monotonic = float(value["producer_started_monotonic_seconds"])
    deadline_monotonic = float(value["producer_deadline_monotonic_seconds"])
    completed_monotonic = float(value["producer_completed_monotonic_seconds"])
    elapsed = float(value["producer_elapsed_seconds"])
    for number in (
        started_wall,
        completed_wall,
        terminate_wall,
        started_monotonic,
        deadline_monotonic,
        completed_monotonic,
        elapsed,
    ):
        _require(math.isfinite(number), "producer deadline contains non-finite time")
    _require(
        math.isclose(
            deadline_monotonic,
            started_monotonic + PRODUCER_CEILING_SECONDS,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        and math.isclose(
            elapsed,
            completed_monotonic - started_monotonic,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        and elapsed >= 0.0
        and completed_wall >= started_wall,
        "producer deadline arithmetic is inconsistent",
    )
    remaining = float(value["provision_remaining_at_start_seconds"])
    _require(
        math.isclose(remaining, terminate_wall - started_wall, rel_tol=0.0, abs_tol=1e-3)
        and remaining >= MINIMUM_PROVISION_REMAINING_AT_START_SECONDS,
        "provision leaves insufficient qualification/handoff/ledger time",
    )
    completed = completed_monotonic <= deadline_monotonic and elapsed <= PRODUCER_CEILING_SECONDS
    _require(
        value["completed_within_ceiling"] is completed and completed,
        "producer exceeded its frozen 105-minute ceiling",
    )
    return copy.deepcopy(dict(value))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProducerError(message)


def _sha256(value: Any, label: str) -> str:
    normalized = str(value)
    if SHA256_PATTERN.fullmatch(normalized) is None:
        raise ProducerError(f"{label} must be a lowercase SHA-256 digest")
    return normalized


def _canonical_uuid(value: Any, label: str) -> str:
    try:
        parsed = uuid.UUID(str(value))
    except ValueError as error:
        raise ProducerError(f"{label} is not a UUID") from error
    normalized = str(parsed)
    _require(normalized == value, f"{label} is not canonical")
    return normalized


def sha256_file(path: str | Path) -> str:
    target = Path(path)
    hasher = hashlib.sha256()
    with target.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _strict_json(path: str | Path, label: str) -> dict[str, Any]:
    target = Path(path)
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProducerError(f"could not read strict {label} JSON") from error
    if not isinstance(raw, dict):
        raise ProducerError(f"{label} must contain one JSON object")
    try:
        canonical_json_bytes(raw)
    except QualificationError as error:
        raise ProducerError(f"{label} is not finite canonical JSON") from error
    return raw


def _exclusive_json(path: str | Path, value: Mapping[str, Any], *, mode: int = 0o400) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )
    try:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    except FileExistsError as error:
        raise ProducerError(f"append-only artifact already exists: {target}") from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(target, mode)
        directory_descriptor = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        # An interrupted append-only artifact remains evidence of failure.  Do
        # not unlink it and accidentally permit an ambiguous retry.
        raise


def _exclusive_jsonl(
    path: str | Path,
    records: Sequence[Mapping[str, Any]],
    *,
    mode: int = 0o400,
) -> None:
    """Create one production-shaped append stream with O_EXCL, fsync, and chmod."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _require(bool(records), "production-shaped JSONL benchmark cannot be empty")
    try:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    except FileExistsError as error:
        raise ProducerError(f"append-only artifact already exists: {target}") from error
    with os.fdopen(descriptor, "wb") as handle:
        for record in records:
            handle.write(canonical_json_bytes(dict(record)) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(target, mode)
    directory_descriptor = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def _seal(body: Mapping[str, Any], digest_field: str) -> dict[str, Any]:
    _require(digest_field not in body, f"unsealed body contains {digest_field}")
    result = copy.deepcopy(dict(body))
    result[digest_field] = semantic_digest(result)
    return result


def _verify_self_digest(value: Mapping[str, Any], field: str, label: str) -> None:
    _require(
        value.get(field) == semantic_digest({key: item for key, item in value.items() if key != field}),
        f"{label} self-digest mismatch",
    )


def _direct_file_binding(path: str | Path, *, kind: str, digest_field: str) -> dict[str, Any]:
    target = Path(path).resolve()
    _require(target.is_file() and not target.is_symlink(), f"{kind} artifact is not a direct file")
    payload = _strict_json(target, kind)
    digest = _sha256(payload.get(digest_field), f"{kind} semantic digest")
    return {
        "kind": kind,
        "path": str(target),
        "file_sha256": sha256_file(target),
        digest_field: digest,
        "bytes": target.stat().st_size,
        "mode": stat.S_IMODE(target.stat().st_mode),
    }


@dataclass(frozen=True)
class GPUDevice:
    ordinal: int
    uuid: str
    name: str

    def as_dict(self) -> dict[str, Any]:
        return {"host_ordinal": self.ordinal, "uuid": self.uuid, "name": self.name}


def construct_tuned_capacity_probe(
    *,
    devices: Sequence[GPUDevice],
    device_receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Derive and seal the branch from sixteen authenticated capacity cells."""

    gpu_uuids = [device.uuid for device in devices]
    _require(
        len(gpu_uuids) == 4 and gpu_uuids == sorted(set(gpu_uuids)),
        "capacity probe devices are not four sorted UUIDs",
    )
    _require(len(device_receipts) == 4, "capacity probe requires four device receipts")
    failed = [
        str(receipt["device_uuid"])
        for receipt in device_receipts
        if receipt.get("status") == "recoverable_capacity_failure"
    ]
    branch = "tuned_probe_passed_full" if not failed else "tuned_capacity_fallback_baseline"
    probe = seal_tuned_capacity_probe(
        {
            "schema": TUNED_CAPACITY_PROBE_SCHEMA,
            "schema_version": TUNED_CAPACITY_PROBE_SCHEMA_VERSION,
            "qualification_branch": branch,
            "status": "passed" if not failed else "recoverable_capacity_failure",
            "actual_model_execution": True,
            "profile_contract": copy.deepcopy(dict(PROFILE_CONTRACT["tuned"])),
            "workload_contract": {
                "panels": list(PANELS),
                "train_batch_size": 50,
                "gradient_accumulation_steps": 1,
                "evaluation_batch_size_examples": 128,
                "registered_evaluation_example_count": 128,
                "production_evaluation_partition": "example_chunks_then_flatten_prompt_views",
                "all_six_views": True,
                "both_laws": True,
                "corpus_max_train_sample_injected": True,
                "exact_worst_shape_batch_exercised": True,
            },
            "numerical_execution": {
                "configure_numerical_execution_called": True,
                "receipt": {
                    "deterministic_algorithms": True,
                    "deterministic_warn_only": False,
                    "cudnn_benchmark": False,
                    "cudnn_deterministic": True,
                    "cuda_matmul_allow_tf32": False,
                    "cudnn_allow_tf32": False,
                    "float32_matmul_precision": "highest",
                    "cublas_workspace_config": ":4096:8",
                },
            },
            "gpu_uuids": gpu_uuids,
            "allowed_disqualifiers": list(TUNED_CAPACITY_DISQUALIFIERS),
            "device_receipts": copy.deepcopy(list(device_receipts)),
            "failed_device_uuids": failed,
            "outcomes_seen": False,
            "itt_ledger_created": False,
            "g01_launch_authorized": False,
        }
    )
    return validate_tuned_capacity_probe(probe, expected_device_uuids=gpu_uuids)


@dataclass(frozen=True)
class ControllerOptions:
    repo: Path
    freeze_path: Path
    freeze_sha256: str
    execution_root: Path
    engineering_root: Path
    provision_receipt: Path
    provision_receipt_sha256: str
    pod_id: str
    model_receipts: Mapping[str, Path]
    integration_audits: Mapping[str, Path]
    execution_uuid: str


@dataclass(frozen=True)
class AuthenticatedInputs:
    verified: VerifiedFreeze
    freeze_binding: Mapping[str, Any]
    provision_binding: Mapping[str, Any]
    model_receipt_bindings: Mapping[str, Mapping[str, Any]]
    model_integration_audit_bindings: Mapping[str, Mapping[str, Any]]
    snapshot_roots: Mapping[str, str]
    provision_terminate_after_utc: str


def canonical_controller_argv(
    *,
    repo: str | Path,
    freeze_path: str | Path,
    expected_freeze_sha256: str,
    execution_root: str | Path,
    engineering_root: str | Path,
    provision_receipt_path: str | Path,
    expected_provision_receipt_sha256: str,
    expected_pod_id: str,
    model_receipt_0p5b: str | Path,
    model_receipt_1p5b: str | Path,
    integration_audit_0p5b: str | Path,
    integration_audit_1p5b: str | Path,
    execution_uuid: str,
) -> list[str]:
    """Return the only accepted controller invocation, excluding Python."""

    resolved_repo = Path(repo).resolve()
    _sha256(expected_freeze_sha256, "freeze SHA-256")
    _sha256(expected_provision_receipt_sha256, "provision receipt SHA-256")
    _canonical_uuid(execution_uuid, "qualification execution UUID")
    _require(
        bool(expected_pod_id) and not any(character.isspace() for character in expected_pod_id),
        "pod ID must be a nonempty token",
    )
    values = [
        str((resolved_repo / CONTROLLER_RELATIVE).resolve()),
        "--mode",
        "controller",
        "--repo",
        str(resolved_repo),
        "--freeze",
        str(Path(freeze_path).resolve()),
        "--freeze-sha256",
        expected_freeze_sha256,
        "--execution-root",
        str(Path(execution_root).resolve()),
        "--engineering-root",
        str(Path(engineering_root).resolve()),
        "--provision-receipt",
        str(Path(provision_receipt_path).resolve()),
        "--provision-receipt-sha256",
        expected_provision_receipt_sha256,
        "--pod-id",
        expected_pod_id,
        "--model-receipt-0p5b",
        str(Path(model_receipt_0p5b).resolve()),
        "--model-receipt-1p5b",
        str(Path(model_receipt_1p5b).resolve()),
        "--integration-audit-0p5b",
        str(Path(integration_audit_0p5b).resolve()),
        "--integration-audit-1p5b",
        str(Path(integration_audit_1p5b).resolve()),
        "--execution-uuid",
        execution_uuid,
    ]
    return values


def _options_argv(options: ControllerOptions) -> list[str]:
    return canonical_controller_argv(
        repo=options.repo,
        freeze_path=options.freeze_path,
        expected_freeze_sha256=options.freeze_sha256,
        execution_root=options.execution_root,
        engineering_root=options.engineering_root,
        provision_receipt_path=options.provision_receipt,
        expected_provision_receipt_sha256=options.provision_receipt_sha256,
        expected_pod_id=options.pod_id,
        model_receipt_0p5b=options.model_receipts["g00f-0p5b"],
        model_receipt_1p5b=options.model_receipts["g00f-1p5b"],
        integration_audit_0p5b=options.integration_audits["g00f-0p5b"],
        integration_audit_1p5b=options.integration_audits["g00f-1p5b"],
        execution_uuid=options.execution_uuid,
    )


def discover_h200_devices(
    *,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    require_torch_cuda: bool = True,
) -> tuple[GPUDevice, ...]:
    """Discover exactly four distinct physical H200 UUIDs, fail closed otherwise."""

    try:
        completed = command_runner(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ProducerError("could not authenticate H200 inventory with nvidia-smi") from error
    devices: list[GPUDevice] = []
    for raw_line in completed.stdout.splitlines():
        parts = [part.strip() for part in raw_line.split(",", maxsplit=2)]
        _require(len(parts) == 3, "nvidia-smi returned a malformed GPU row")
        try:
            ordinal = int(parts[0])
        except ValueError as error:
            raise ProducerError("nvidia-smi GPU ordinal is invalid") from error
        device = GPUDevice(ordinal=ordinal, uuid=parts[1], name=parts[2])
        _require(GPU_UUID_PATTERN.fullmatch(device.uuid) is not None, "GPU UUID is invalid")
        _require(device.name.startswith("NVIDIA H200"), "qualification requires actual NVIDIA H200 GPUs")
        devices.append(device)
    _require(len(devices) == 4, "qualification requires exactly four H200 GPUs")
    _require(len({device.uuid for device in devices}) == 4, "H200 GPU UUIDs are not distinct")
    _require(len({device.ordinal for device in devices}) == 4, "H200 GPU ordinals are not distinct")
    devices.sort(key=lambda device: device.uuid)
    if require_torch_cuda:
        _require(
            torch.cuda.is_available() and torch.cuda.device_count() == 4,
            "PyTorch does not expose exactly four CUDA devices",
        )
        observed_names = [torch.cuda.get_device_name(device.ordinal) for device in devices]
        _require(
            all(name.startswith("NVIDIA H200") for name in observed_names),
            "PyTorch CUDA inventory is not four H200s",
        )
    return tuple(devices)


def authenticate_inputs(options: ControllerOptions) -> AuthenticatedInputs:
    """Authenticate freeze, provision, model, and integration receipts pre-load."""

    verified = verify_freeze(
        repo=options.repo,
        freeze_path=options.freeze_path,
        expected_freeze_sha256=options.freeze_sha256,
    )
    _require(options.execution_root.is_absolute(), "execution root must be absolute")
    _require(options.engineering_root.is_absolute(), "engineering root must be absolute")
    _require(
        options.engineering_root.name.startswith(ENGINEERING_ROOT_PREFIX),
        "engineering root name is not dedicated to H200 qualification",
    )
    _require(
        options.engineering_root != options.execution_root, "engineering and execution roots must be distinct"
    )
    _require(
        not (options.execution_root / "itt-ledger").exists(), "qualification must precede ITT ledger creation"
    )
    provision = verify_runpod_provision_receipt(
        verified=verified,
        receipt_path=options.provision_receipt,
        expected_receipt_sha256=options.provision_receipt_sha256,
        expected_pod_id=options.pod_id,
    )
    model_bindings: dict[str, Mapping[str, Any]] = {}
    audit_bindings: dict[str, Mapping[str, Any]] = {}
    snapshot_roots: dict[str, str] = {}
    for panel in PANELS:
        model = verify_model_snapshot_receipt(
            verified=verified,
            panel_id=panel,
            receipt_path=options.model_receipts[panel],
        )
        audit = verify_model_integration_audit(
            verified=verified,
            panel_id=panel,
            audit_path=options.integration_audits[panel],
            replay_model_snapshot=True,
        )
        _require(
            audit["model_snapshot_receipt"] == model,
            f"{panel} integration audit does not bind the exact model receipt",
        )
        model_bindings[panel] = {
            "panel_id": panel,
            "file_sha256": model["file_sha256"],
            "receipt_digest": model["receipt_digest"],
        }
        audit_bindings[panel] = {
            "panel_id": panel,
            "file_sha256": audit["file_sha256"],
            "audit_digest": audit["audit_digest"],
            "report_digest": audit["report_digest"],
        }
        snapshot_roots[panel] = str(model["snapshot_root"])
    return AuthenticatedInputs(
        verified=verified,
        freeze_binding={
            "freeze_file_sha256": verified.file_sha256,
            "freeze_digest": verified.digest,
        },
        provision_binding={
            "file_sha256": provision["file_sha256"],
            "receipt_digest": provision["receipt_digest"],
            "pod_id": provision["pod_id"],
        },
        model_receipt_bindings=model_bindings,
        model_integration_audit_bindings=audit_bindings,
        snapshot_roots=snapshot_roots,
        provision_terminate_after_utc=str(provision["provisioning"]["terminate_after_utc"]),
    )


class TransientInventory:
    """Track every raw temporary file while enforcing the one-TiB ceiling."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        self.entries: list[dict[str, Any]] = []
        self.live_bytes = 0
        self.peak_bytes = 0

    def register(self, path: str | Path, *, lifecycle: str) -> None:
        target = Path(path).resolve()
        _require(self.root in target.parents, "transient artifact lies outside its raw root")
        _require(target.is_file() and not target.is_symlink(), "transient artifact is not direct")
        size = target.stat().st_size
        digest = sha256_file(target)
        self.live_bytes += size
        self.peak_bytes = max(self.peak_bytes, self.live_bytes)
        _require(self.peak_bytes <= TRANSIENT_CEILING_BYTES, "transient raw storage exceeded one TiB")
        self.entries.append(
            {
                "path": str(target),
                "bytes": size,
                "sha256": digest,
                "lifecycle": lifecycle,
                "deleted": False,
            }
        )

    def delete(self, path: str | Path) -> None:
        target = Path(path).resolve()
        matches = [entry for entry in self.entries if entry["path"] == str(target) and not entry["deleted"]]
        _require(len(matches) == 1, "transient deletion does not match one live inventory entry")
        entry = matches[0]
        _require(
            target.is_file() and sha256_file(target) == entry["sha256"],
            "transient artifact changed before deletion",
        )
        target.unlink()
        entry["deleted"] = True
        self.live_bytes -= int(entry["bytes"])
        parent = target.parent
        while parent != self.root and self.root in parent.parents:
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent

    def rename(self, source: str | Path, target: str | Path, *, lifecycle: str) -> None:
        source_path = Path(source).resolve()
        target_path = Path(target).resolve()
        matches = [
            entry for entry in self.entries if entry["path"] == str(source_path) and not entry["deleted"]
        ]
        _require(
            len(matches) == 1 and not target_path.exists(),
            "transient atomic rename source/target is ambiguous",
        )
        entry = matches[0]
        _require(sha256_file(source_path) == entry["sha256"], "transient atomic stage changed before rename")
        os.replace(source_path, target_path)
        entry["deleted"] = True
        self.live_bytes -= int(entry["bytes"])
        self.register(target_path, lifecycle=lifecycle)

    def final_body(self, *, execution_uuid: str) -> dict[str, Any]:
        final_inventory = sorted(str(path.relative_to(self.root)) for path in self.root.rglob("*"))
        _require(
            self.live_bytes == 0 and not final_inventory,
            "transient raw root is not empty at producer handoff",
        )
        _require(
            all(entry["deleted"] for entry in self.entries),
            "transient inventory contains an undeleted raw artifact",
        )
        return {
            "schema": TRANSIENT_INVENTORY_SCHEMA,
            "schema_version": TRANSIENT_INVENTORY_SCHEMA_VERSION,
            "execution_uuid": execution_uuid,
            "transient_root": str(self.root),
            "ceiling_bytes": TRANSIENT_CEILING_BYTES,
            "peak_bytes": self.peak_bytes,
            "entries": copy.deepcopy(self.entries),
            "final_inventory": final_inventory,
            "final_bytes": 0,
            "all_listed_paths_absent": all(not Path(entry["path"]).exists() for entry in self.entries),
            "compact_manifest_only": True,
            "raw_payloads_embedded": False,
            "outcomes_seen": False,
            "itt_ledger_created": False,
            "g01_launch_authorized": False,
        }


def _validate_transient_inventory(value: Mapping[str, Any]) -> dict[str, Any]:
    expected_keys = {
        "schema",
        "schema_version",
        "execution_uuid",
        "transient_root",
        "ceiling_bytes",
        "peak_bytes",
        "entries",
        "final_inventory",
        "final_bytes",
        "all_listed_paths_absent",
        "compact_manifest_only",
        "raw_payloads_embedded",
        "outcomes_seen",
        "itt_ledger_created",
        "g01_launch_authorized",
        "inventory_digest",
    }
    _require(set(value) == expected_keys, "transient inventory fields changed")
    _require(
        value.get("schema") == TRANSIENT_INVENTORY_SCHEMA
        and value.get("schema_version") == TRANSIENT_INVENTORY_SCHEMA_VERSION,
        "transient inventory schema changed",
    )
    _verify_self_digest(value, "inventory_digest", "transient inventory")
    _canonical_uuid(value.get("execution_uuid"), "transient inventory execution UUID")
    root = Path(str(value.get("transient_root", "")))
    _require(root.is_absolute(), "transient inventory root is not absolute")
    _require(value.get("ceiling_bytes") == TRANSIENT_CEILING_BYTES, "transient inventory ceiling changed")
    peak = value.get("peak_bytes")
    _require(
        type(peak) is int and 0 <= peak <= TRANSIENT_CEILING_BYTES, "transient inventory peak is invalid"
    )
    entries = value.get("entries")
    _require(isinstance(entries, list), "transient inventory entries are malformed")
    seen: set[str] = set()
    for raw in cast(list[Any], entries):
        _require(isinstance(raw, Mapping), "transient inventory entry is not an object")
        entry = dict(raw)
        _require(
            set(entry) == {"path", "bytes", "sha256", "lifecycle", "deleted"},
            "transient inventory entry fields changed",
        )
        path = Path(str(entry["path"]))
        _require(
            path.is_absolute() and root in path.parents and str(path) not in seen,
            "transient inventory path is unsafe or duplicated",
        )
        seen.add(str(path))
        _require(
            type(entry["bytes"]) is int and entry["bytes"] >= 0, "transient inventory byte count is invalid"
        )
        _sha256(entry["sha256"], "transient artifact SHA-256")
        _require(
            isinstance(entry["lifecycle"], str) and bool(entry["lifecycle"]), "transient lifecycle is empty"
        )
        _require(
            entry["deleted"] is True and not path.exists(), "listed transient raw artifact was not deleted"
        )
    _require(
        value.get("final_inventory") == [] and value.get("final_bytes") == 0,
        "transient root final inventory is not exactly empty",
    )
    _require(
        value.get("all_listed_paths_absent") is True
        and value.get("compact_manifest_only") is True
        and value.get("raw_payloads_embedded") is False,
        "transient inventory retained a raw payload",
    )
    _require(
        value.get("outcomes_seen") is False
        and value.get("itt_ledger_created") is False
        and value.get("g01_launch_authorized") is False,
        "transient inventory crossed a scientific boundary",
    )
    return copy.deepcopy(dict(value))


def _validate_artifact_binding(
    binding: Mapping[str, Any],
    *,
    expected_kind: str,
    digest_field: str,
    path_override: str | Path | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_keys = {"kind", "path", "file_sha256", digest_field, "bytes", "mode"}
    _require(set(binding) == expected_keys, f"{expected_kind} artifact fields changed")
    _require(binding.get("kind") == expected_kind, f"{expected_kind} artifact kind changed")
    recorded_path = Path(str(binding.get("path", "")))
    _require(recorded_path.is_absolute(), f"{expected_kind} recorded path is not absolute")
    target = recorded_path if path_override is None else Path(path_override).resolve()
    _require(target.is_file() and not target.is_symlink(), f"{expected_kind} artifact is absent")
    _require(
        sha256_file(target) == _sha256(binding.get("file_sha256"), f"{expected_kind} SHA-256"),
        f"{expected_kind} artifact bytes changed",
    )
    _require(target.stat().st_size == binding.get("bytes"), f"{expected_kind} artifact size changed")
    _require(
        binding.get("mode") == 0o400 and stat.S_IMODE(target.stat().st_mode) == 0o400,
        f"{expected_kind} artifact mode changed",
    )
    payload = _strict_json(target, expected_kind)
    _require(
        payload.get(digest_field) == _sha256(binding.get(digest_field), digest_field),
        f"{expected_kind} semantic digest changed",
    )
    replayed = {
        **copy.deepcopy(dict(binding)),
        "path": str(target),
        "recorded_source_path": str(recorded_path),
    }
    return replayed, payload


def _validate_worker_receipt_body(value: Mapping[str, Any]) -> dict[str, Any]:
    expected_keys = {
        "schema",
        "schema_version",
        "execution_uuid",
        "qualification_branch",
        "worker_index",
        "device",
        "actual_model_execution",
        "controller_binding",
        "implementation_binding",
        "freeze_binding",
        "provision_binding",
        "model_receipt_bindings",
        "model_integration_audit_bindings",
        "engineering_scope",
        "assignments",
        "process_record_count",
        "numeric_comparison_record_count",
        "shard",
        "transient_root",
        "transient_peak_bytes",
        "transient_entries",
        "transient_final_inventory",
        "outcomes_seen",
        "itt_ledger_created",
        "g01_launch_authorized",
        "receipt_digest",
    }
    _require(set(value) == expected_keys, "worker receipt fields changed")
    _require(
        value.get("schema") == WORKER_RECEIPT_SCHEMA
        and value.get("schema_version") == WORKER_RECEIPT_SCHEMA_VERSION,
        "worker receipt schema changed",
    )
    _verify_self_digest(value, "receipt_digest", "worker receipt")
    _canonical_uuid(value.get("execution_uuid"), "worker execution UUID")
    branch = value.get("qualification_branch")
    _require(
        branch in {"tuned_probe_passed_full", "tuned_capacity_fallback_baseline"},
        "worker qualification branch changed",
    )
    worker_index = value.get("worker_index")
    _require(type(worker_index) is int and 0 <= worker_index < 4, "worker index is invalid")
    device = value.get("device")
    _require(
        isinstance(device, Mapping) and set(device) == {"host_ordinal", "uuid", "name"},
        "worker device binding is malformed",
    )
    device = cast(Mapping[str, Any], device)
    _require(
        type(device.get("host_ordinal")) is int and int(device["host_ordinal"]) >= 0,
        "worker host GPU ordinal is invalid",
    )
    _require(
        GPU_UUID_PATTERN.fullmatch(str(device.get("uuid", ""))) is not None, "worker GPU UUID is invalid"
    )
    _require(str(device.get("name", "")).startswith("NVIDIA H200"), "worker did not execute on H200")
    _require(value.get("actual_model_execution") is True, "worker is not actual-model evidence")
    assignments = value.get("assignments")
    expected_profiles = PROFILES if branch == "tuned_probe_passed_full" else ("baseline",)
    expected_count = 16 if branch == "tuned_probe_passed_full" else 8
    _require(
        isinstance(assignments, list) and len(assignments) == expected_count,
        "worker logical process assignment cardinality changed",
    )
    assignments = cast(list[Any], assignments)
    expected_assignments = [
        {"profile": profile, "replicate": replicate, "panel_id": panel, "law_family": law}
        for profile in expected_profiles
        for replicate in REPLICATES[profile]
        for panel in PANELS
        for law in LAW_FAMILIES
    ]
    _require(assignments == expected_assignments, "worker assignment Cartesian product changed")
    _require(value.get("process_record_count") == expected_count, "worker process count changed")
    expected_numeric = 4 if worker_index == 0 and branch == "tuned_probe_passed_full" else 0
    _require(
        value.get("numeric_comparison_record_count") == expected_numeric,
        "worker numeric comparison count changed",
    )
    shard = value.get("shard")
    _require(isinstance(shard, Mapping), "worker shard binding is malformed")
    shard = cast(Mapping[str, Any], shard)
    _require(
        set(shard) == {"path", "file_sha256", "shard_digest", "bytes", "mode"},
        "worker shard binding fields changed",
    )
    _sha256(shard.get("file_sha256"), "worker shard SHA-256")
    _sha256(shard.get("shard_digest"), "worker shard digest")
    _require(shard.get("mode") == 0o400, "worker shard is not immutable")
    transient_root = Path(str(value.get("transient_root", "")))
    _require(transient_root.is_absolute(), "worker transient root is not absolute")
    peak = value.get("transient_peak_bytes")
    _require(type(peak) is int and 0 <= peak <= TRANSIENT_CEILING_BYTES, "worker transient peak is invalid")
    entries = value.get("transient_entries")
    _require(isinstance(entries, list), "worker transient entries are malformed")
    for entry in cast(list[Any], entries):
        _require(
            isinstance(entry, Mapping) and entry.get("deleted") is True,
            "worker retained a transient artifact",
        )
        _require(not Path(str(entry.get("path", ""))).exists(), "worker transient artifact still exists")
    _require(value.get("transient_final_inventory") == [], "worker transient root is not empty")
    _require(
        value.get("outcomes_seen") is False
        and value.get("itt_ledger_created") is False
        and value.get("g01_launch_authorized") is False,
        "worker crossed a scientific boundary",
    )
    return copy.deepcopy(dict(value))


def _validate_worker_launch_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "request_digest",
        "request_kind",
        "worker_index",
        "panel_id",
        "law_family",
        "device_uuid",
        "argv",
        "argv_sha256",
        "exit_code",
        "timed_out",
        "cancelled",
        "log_binding",
        "launch_digest",
    }
    _require(set(value) == expected, "qualification worker launch receipt fields changed")
    _verify_self_digest(value, "launch_digest", "qualification worker launch receipt")
    _sha256(value["request_digest"], "qualification worker request digest")
    _require(
        value["request_kind"] == "qualification"
        and type(value["worker_index"]) is int
        and 0 <= int(value["worker_index"]) < 4
        and value["panel_id"] is None
        and value["law_family"] is None
        and GPU_UUID_PATTERN.fullmatch(str(value["device_uuid"])) is not None,
        "qualification worker launch identity changed",
    )
    argv = value["argv"]
    _require(
        isinstance(argv, list)
        and value["argv_sha256"] == semantic_digest(argv)
        and value["exit_code"] == 0
        and value["timed_out"] is False
        and value["cancelled"] is False,
        "qualification worker did not exit cleanly before the deadline",
    )
    log = value["log_binding"]
    _require(
        isinstance(log, Mapping)
        and set(log) == {"path", "file_sha256", "bytes", "mode"}
        and Path(str(log["path"])).is_absolute()
        and type(log["bytes"]) is int
        and int(log["bytes"]) >= 0
        and log["mode"] == 0o400,
        "qualification worker log binding changed",
    )
    _sha256(log["file_sha256"], "qualification worker log SHA-256")
    return copy.deepcopy(dict(value))


def validate_producer_receipt_body(
    receipt: Mapping[str, Any],
    *,
    artifact_path_overrides: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Strictly replay a producer receipt and its three compact artifacts."""

    expected_keys = {
        "schema",
        "schema_version",
        "execution_uuid",
        "actual_model_execution",
        "qualification_branch",
        "tuned_capacity_probe",
        "deadline_binding",
        "controller_binding",
        "implementation_binding",
        "freeze_binding",
        "provision_binding",
        "model_receipt_bindings",
        "model_integration_audit_bindings",
        "gpu_uuids",
        "worker_receipts",
        "worker_launch_receipts",
        "artifacts",
        "engineering_scope",
        "transient_storage",
        "weight_updates_scope",
        "outcomes_seen",
        "itt_ledger_created",
        "g01_launch_authorized",
        "receipt_digest",
    }
    _require(set(receipt) == expected_keys, "producer receipt fields changed")
    _require(
        receipt.get("schema") == PRODUCER_SCHEMA and receipt.get("schema_version") == PRODUCER_SCHEMA_VERSION,
        "producer receipt schema changed",
    )
    _verify_self_digest(receipt, "receipt_digest", "producer receipt")
    _canonical_uuid(receipt.get("execution_uuid"), "producer execution UUID")
    _require(receipt.get("actual_model_execution") is True, "producer receipt is not actual-model execution")
    _validate_deadline_binding(receipt["deadline_binding"])
    controller = receipt.get("controller_binding")
    implementation = receipt.get("implementation_binding")
    _require(isinstance(controller, Mapping), "producer controller binding is malformed")
    controller = cast(Mapping[str, Any], controller)
    _require(
        set(controller) == {"path", "file_sha256", "argv", "argv_sha256"}
        and isinstance(controller.get("argv"), list)
        and controller.get("argv_sha256") == semantic_digest(controller.get("argv")),
        "producer controller argv binding changed",
    )
    _sha256(controller.get("file_sha256"), "producer controller SHA-256")
    _require(isinstance(implementation, Mapping), "producer implementation binding is malformed")
    implementation = cast(Mapping[str, Any], implementation)
    _require(set(implementation) == {"path", "file_sha256"}, "producer implementation binding fields changed")
    _sha256(implementation.get("file_sha256"), "producer implementation SHA-256")
    gpu_uuids = receipt.get("gpu_uuids")
    _require(
        isinstance(gpu_uuids, list)
        and len(gpu_uuids) == 4
        and gpu_uuids == sorted(set(str(item) for item in gpu_uuids))
        and all(GPU_UUID_PATTERN.fullmatch(str(item)) is not None for item in gpu_uuids),
        "producer does not bind four sorted distinct H200 UUIDs",
    )
    gpu_uuids = cast(list[Any], gpu_uuids)
    workers = receipt.get("worker_receipts")
    _require(
        isinstance(workers, list) and len(workers) == 4,
        "producer must embed exactly four worker receipt bodies",
    )
    validated_workers = [
        _validate_worker_receipt_body(cast(Mapping[str, Any], worker)) for worker in cast(list[Any], workers)
    ]
    _require(
        [worker["worker_index"] for worker in validated_workers] == list(range(4)),
        "producer worker receipts are not ordered 0 through 3",
    )
    _require(
        [worker["device"]["uuid"] for worker in validated_workers] == gpu_uuids,
        "producer GPU UUIDs differ from worker receipts",
    )
    launches = receipt.get("worker_launch_receipts")
    _require(
        isinstance(launches, list) and len(launches) == 4,
        "producer must embed exactly four qualification worker launch receipts",
    )
    validated_launches = [
        _validate_worker_launch_receipt(cast(Mapping[str, Any], item)) for item in cast(list[Any], launches)
    ]
    _require(
        [item["worker_index"] for item in validated_launches] == list(range(4))
        and [item["device_uuid"] for item in validated_launches] == gpu_uuids,
        "producer qualification worker launch order/devices changed",
    )
    for worker in validated_workers:
        _require(
            worker["execution_uuid"] == receipt["execution_uuid"]
            and worker["qualification_branch"] == receipt["qualification_branch"]
            and worker["controller_binding"] == controller
            and worker["implementation_binding"] == implementation
            and worker["freeze_binding"] == receipt["freeze_binding"]
            and worker["provision_binding"] == receipt["provision_binding"]
            and worker["model_receipt_bindings"] == receipt["model_receipt_bindings"]
            and worker["model_integration_audit_bindings"] == receipt["model_integration_audit_bindings"],
            "worker receipt authentication bindings differ from producer",
        )
    artifacts = receipt.get("artifacts")
    _require(
        isinstance(artifacts, Mapping) and set(artifacts) == {"evidence", "report", "transient_inventory"},
        "producer compact artifact set changed",
    )
    artifacts = cast(Mapping[str, Mapping[str, Any]], artifacts)
    overrides = {} if artifact_path_overrides is None else dict(artifact_path_overrides)
    _require(
        set(overrides) <= {"evidence", "report", "transient_inventory"},
        "unknown compact artifact path override",
    )
    replayed_artifacts: dict[str, Any] = {}
    evidence_binding, evidence = _validate_artifact_binding(
        artifacts["evidence"],
        expected_kind="evidence",
        digest_field="evidence_digest",
        path_override=overrides.get("evidence"),
    )
    report_binding, report = _validate_artifact_binding(
        artifacts["report"],
        expected_kind="report",
        digest_field="report_digest",
        path_override=overrides.get("report"),
    )
    transient_binding, transient = _validate_artifact_binding(
        artifacts["transient_inventory"],
        expected_kind="transient_inventory",
        digest_field="inventory_digest",
        path_override=overrides.get("transient_inventory"),
    )
    try:
        normalized_evidence = validate_qualification_evidence(evidence)
        normalized_report = validate_qualification_report(report)
    except QualificationError as error:
        raise ProducerError("producer compact qualification artifacts failed replay") from error
    _require(
        normalized_evidence == evidence and normalized_report == report,
        "producer compact qualification normalization changed",
    )
    _validate_transient_inventory(transient)
    _require(
        report["evidence_binding"]["evidence_digest"] == evidence["evidence_digest"]
        and receipt["qualification_branch"] == evidence["qualification_branch"]
        and receipt["qualification_branch"] == report["qualification_branch"]
        and receipt["tuned_capacity_probe"] == evidence["tuned_capacity_probe"]
        and receipt["tuned_capacity_probe"] == report["tuned_capacity_probe"]
        and report["freeze_binding"] == receipt["freeze_binding"]
        and report["provision_binding"] == receipt["provision_binding"]
        and report["model_receipt_bindings"] == receipt["model_receipt_bindings"]
        and report["model_integration_audit_bindings"] == receipt["model_integration_audit_bindings"]
        and report["gpu_uuids"] == gpu_uuids
        and transient["execution_uuid"] == receipt["execution_uuid"],
        "producer compact artifacts do not cross-bind",
    )
    replayed_artifacts.update(
        evidence=evidence_binding,
        report=report_binding,
        transient_inventory=transient_binding,
    )
    transient_storage = receipt.get("transient_storage")
    _require(isinstance(transient_storage, Mapping), "producer transient storage is malformed")
    _require(
        transient_storage
        == {
            "root": transient["transient_root"],
            "ceiling_bytes": TRANSIENT_CEILING_BYTES,
            "peak_bytes": transient["peak_bytes"],
            "final_inventory": [],
            "final_bytes": 0,
            "all_listed_paths_absent": True,
            "raw_vectors_persisted": False,
            "compact_evidence_retained_through_final_gate": True,
        },
        "producer transient-storage summary changed",
    )
    scope = receipt.get("engineering_scope")
    _require(
        isinstance(scope, Mapping) and scope == evidence["engineering_scope"],
        "producer engineering scope differs from evidence",
    )
    _require(
        receipt.get("weight_updates_scope") == "engineering_qualification_only"
        and receipt.get("outcomes_seen") is False
        and receipt.get("itt_ledger_created") is False
        and receipt.get("g01_launch_authorized") is False,
        "producer receipt crossed a scientific boundary",
    )
    normalized = copy.deepcopy(dict(receipt))
    normalized["replayed_artifacts"] = replayed_artifacts
    return normalized


def verify_producer_receipt(
    *,
    receipt_path: str | Path,
    expected_controller_sha256: str,
    expected_implementation_sha256: str,
    expected_controller_argv: Sequence[str],
    expected_freeze_binding: Mapping[str, Any],
    expected_provision_binding: Mapping[str, Any],
    expected_model_receipt_bindings: Mapping[str, Any],
    expected_integration_audit_bindings: Mapping[str, Any],
    expected_gpu_uuids: Sequence[str],
    artifact_path_overrides: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Authenticate a producer receipt and return four handoff bindings."""

    target = Path(receipt_path).resolve()
    _require(target.is_file() and not target.is_symlink(), "producer receipt is absent")
    _require(stat.S_IMODE(target.stat().st_mode) == 0o400, "producer receipt mode changed")
    receipt = _strict_json(target, "producer receipt")
    normalized = validate_producer_receipt_body(
        receipt,
        artifact_path_overrides=artifact_path_overrides,
    )
    _require(
        receipt["controller_binding"]["file_sha256"]
        == _sha256(expected_controller_sha256, "expected controller SHA-256")
        and receipt["implementation_binding"]["file_sha256"]
        == _sha256(expected_implementation_sha256, "expected implementation SHA-256")
        and receipt["controller_binding"]["argv"] == list(expected_controller_argv),
        "producer code or controller argv differs from the expected binding",
    )
    _require(
        receipt["freeze_binding"] == dict(expected_freeze_binding)
        and receipt["provision_binding"] == dict(expected_provision_binding)
        and receipt["model_receipt_bindings"] == dict(expected_model_receipt_bindings)
        and receipt["model_integration_audit_bindings"] == dict(expected_integration_audit_bindings)
        and receipt["gpu_uuids"] == list(expected_gpu_uuids),
        "producer receipt input bindings differ from expected values",
    )
    producer_binding = {
        "kind": "producer_receipt",
        "path": str(target),
        "file_sha256": sha256_file(target),
        "receipt_digest": receipt["receipt_digest"],
        "bytes": target.stat().st_size,
        "mode": stat.S_IMODE(target.stat().st_mode),
    }
    return {
        "receipt": copy.deepcopy(receipt),
        "artifacts": {
            **normalized["replayed_artifacts"],
            "producer_receipt": producer_binding,
        },
        "execution_uuid": receipt["execution_uuid"],
        "gpu_uuids": list(receipt["gpu_uuids"]),
        "eligibility": copy.deepcopy(
            _strict_json(normalized["replayed_artifacts"]["report"]["path"], "report")["eligibility"]
        ),
    }


def replay_producer_receipt(**kwargs: Any) -> dict[str, Any]:
    """Replay a copied producer receipt and compact artifacts without source roots."""

    return verify_producer_receipt(**kwargs)


def _tensor_contiguous_bytes_sha256(tensor: Tensor) -> str:
    contiguous = tensor.detach().contiguous().cpu()
    byte_view = contiguous.view(torch.uint8).reshape(-1)
    hasher = hashlib.sha256()
    for start in range(0, byte_view.numel(), NATIVE_SCAN_CHUNK_ELEMENTS):
        chunk = byte_view[start : start + NATIVE_SCAN_CHUNK_ELEMENTS]
        hasher.update(chunk.numpy().tobytes(order="C"))
    return hasher.hexdigest()


def _native_tensor_sha256(tensor: Tensor) -> str:
    return semantic_digest(
        {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "contiguous_bytes_sha256": _tensor_contiguous_bytes_sha256(tensor),
        }
    )


def _float64_squared_sum(tensor: Tensor) -> float:
    flat = tensor.detach().reshape(-1)
    partials: list[float] = []
    for start in range(0, flat.numel(), NATIVE_SCAN_CHUNK_ELEMENTS):
        chunk = flat[start : start + NATIVE_SCAN_CHUNK_ELEMENTS].to(dtype=torch.float64)
        value = float(torch.sum(chunk * chunk, dtype=torch.float64).cpu())
        _require(math.isfinite(value), "float64 tensor squared sum became non-finite")
        partials.append(value)
    total = math.fsum(partials)
    _require(math.isfinite(total) and total >= 0.0, "tensor norm accumulation is invalid")
    return total


def _float64_pair_sums(left: Tensor, right: Tensor) -> tuple[float, float, float, float]:
    _require(left.shape == right.shape, "paired vector tensor shapes differ")
    left_flat = left.detach().reshape(-1)
    right_flat = right.detach().reshape(-1)
    left_parts: list[float] = []
    right_parts: list[float] = []
    dot_parts: list[float] = []
    difference_parts: list[float] = []
    for start in range(0, left_flat.numel(), NATIVE_SCAN_CHUNK_ELEMENTS):
        lhs = left_flat[start : start + NATIVE_SCAN_CHUNK_ELEMENTS].to(dtype=torch.float64)
        rhs = right_flat[start : start + NATIVE_SCAN_CHUNK_ELEMENTS].to(dtype=torch.float64)
        difference = lhs - rhs
        values = (
            float(torch.sum(lhs * lhs, dtype=torch.float64).cpu()),
            float(torch.sum(rhs * rhs, dtype=torch.float64).cpu()),
            float(torch.sum(lhs * rhs, dtype=torch.float64).cpu()),
            float(torch.sum(difference * difference, dtype=torch.float64).cpu()),
        )
        _require(
            all(math.isfinite(value) for value in values), "paired float64 accumulator became non-finite"
        )
        left_parts.append(values[0])
        right_parts.append(values[1])
        dot_parts.append(values[2])
        difference_parts.append(values[3])
    return (
        math.fsum(left_parts),
        math.fsum(right_parts),
        math.fsum(difference_parts),
        math.fsum(dot_parts),
    )


def _named_trainable_parameters(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    try:
        iterator = model.named_parameters(remove_duplicate=True)
    except TypeError as error:  # pragma: no cover - frozen torch supports the argument
        raise ProducerError("torch lacks named_parameters(remove_duplicate=True)") from error
    rows = sorted(
        ((str(name), parameter) for name, parameter in iterator if parameter.requires_grad),
        key=lambda item: item[0],
    )
    _require(
        bool(rows) and len({name for name, _parameter in rows}) == len(rows),
        "trainable parameter enumeration is empty or duplicated",
    )
    return rows


def _trainable_parameter_manifest(
    panel_id: str,
    parameters: Sequence[tuple[str, nn.Parameter]],
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = [
        {
            "parameter_key": name,
            "dtype": str(parameter.dtype),
            "shape": list(parameter.shape),
            "numel": parameter.numel(),
        }
        for name, parameter in parameters
    ]
    total = sum(int(entry["numel"]) for entry in entries)
    _require(total == TRAINABLE_NUMEL[panel_id], f"{panel_id} trainable numel changed")
    parameter_binding = [
        {
            "order_index": index,
            "parameter_key": entry["parameter_key"],
            "element_count": entry["numel"],
        }
        for index, entry in enumerate(entries)
    ]
    return {
        "enumeration_api": "named_parameters(remove_duplicate=True)",
        "requires_grad_only": True,
        "order": "sorted_parameter_keys",
        "entries": entries,
        "trainable_numel": total,
        "parameter_keys_sha256": semantic_digest(parameter_binding),
        "manifest_sha256": semantic_digest(entries),
    }


def _zero_like(parameter: Tensor) -> Tensor:
    return torch.zeros_like(parameter, memory_format=torch.contiguous_format)


def _vector_descriptor(
    *,
    manifest: Mapping[str, Any],
    parameters: Sequence[tuple[str, nn.Parameter]],
    tensors: Mapping[str, Tensor | None],
) -> dict[str, Any]:
    chunks: list[dict[str, Any]] = []
    squared_sums: list[float] = []
    for order_index, (name, parameter) in enumerate(parameters):
        raw = tensors.get(name)
        tensor = _zero_like(parameter) if raw is None else raw.detach()
        _require(tensor.shape == parameter.shape, f"vector shape differs for {name}")
        chunks.append(
            {
                "order_index": order_index,
                "parameter_key": name,
                "element_count": parameter.numel(),
                "native_tensor_sha256": _native_tensor_sha256(tensor),
            }
        )
        squared_sums.append(_float64_squared_sum(tensor))
    return {
        "parameter_keys_sha256": manifest["parameter_keys_sha256"],
        "native_chunk_manifest_sha256": semantic_digest(chunks),
        "element_count": manifest["trainable_numel"],
        "float64_norm": math.sqrt(max(0.0, math.fsum(squared_sums))),
        "chunk_count": len(chunks),
        "trainable_parameter_manifest_sha256": manifest["manifest_sha256"],
    }


def _state_tensor_manifest(items: Sequence[tuple[str, Tensor]]) -> list[dict[str, Any]]:
    return [
        {
            "key": key,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "native_tensor_sha256": _native_tensor_sha256(tensor),
        }
        for key, tensor in sorted(items, key=lambda item: item[0])
    ]


def _model_state_sha256(model: nn.Module) -> str:
    state = model.state_dict()
    tensors = [(str(key), tensor) for key, tensor in state.items() if isinstance(tensor, Tensor)]
    _require(len(tensors) == len(state), "model state contains a non-tensor value")
    return semantic_digest(_state_tensor_manifest(tensors))


def _optimizer_state_sha256(
    optimizer: torch.optim.Optimizer,
    parameters: Sequence[tuple[str, nn.Parameter]],
) -> str:
    names_by_id = {id(parameter): name for name, parameter in parameters}
    state_rows: list[dict[str, Any]] = []
    for parameter, state in sorted(
        optimizer.state.items(),
        key=lambda item: names_by_id.get(id(item[0]), ""),
    ):
        name = names_by_id.get(id(parameter))
        _require(name is not None, "optimizer contains an unknown parameter")
        values: dict[str, Any] = {}
        for key, value in sorted(state.items(), key=lambda item: str(item[0])):
            if isinstance(value, Tensor):
                values[str(key)] = {
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                    "native_tensor_sha256": _native_tensor_sha256(value),
                }
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                number = float(value)
                _require(math.isfinite(number), "optimizer scalar state is non-finite")
                values[str(key)] = number
            else:
                raise ProducerError("optimizer state contains an unsupported value")
        state_rows.append({"parameter_key": name, "state": values})
    groups: list[dict[str, Any]] = []
    for group in optimizer.param_groups:
        normalized: dict[str, Any] = {}
        for key, value in sorted(group.items(), key=lambda item: str(item[0])):
            if key == "params":
                normalized["parameter_keys"] = [names_by_id[id(parameter)] for parameter in value]
            elif isinstance(value, (str, int, float, bool)) or value is None:
                normalized[str(key)] = value
            else:
                normalized[str(key)] = str(value)
        groups.append(normalized)
    return semantic_digest({"state": state_rows, "parameter_groups": groups})


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    return value


def _stochastic_module_audit(model: nn.Module) -> dict[str, Any]:
    modules: list[dict[str, Any]] = []
    for name, module in sorted(model.named_modules(), key=lambda item: item[0]):
        class_name = type(module).__name__
        if isinstance(module, nn.modules.dropout._DropoutNd):
            probability = float(module.p)
            _require(
                math.isfinite(probability) and 0.0 <= probability <= 1.0,
                "dropout module has an invalid probability",
            )
            modules.append(
                {
                    "name": name or "<root>",
                    "class_name": class_name,
                    "kind": "dropout",
                    "active": bool(module.training and probability > 0.0),
                }
            )
        elif "stochastic" in class_name.lower() or class_name == "RReLU":
            modules.append(
                {
                    "name": name or "<root>",
                    "class_name": class_name,
                    "kind": "stochastic",
                    "active": bool(module.training),
                }
            )
    config = getattr(model, "config", None)
    raw_config = config.to_dict() if config is not None and hasattr(config, "to_dict") else {}

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
                path = f"{prefix}.{key}" if prefix else str(key)
                visit(path, item)
            return
        leaf = prefix.rsplit(".", maxsplit=1)[-1].lower()
        if "dropout" not in leaf or isinstance(value, bool) or not isinstance(value, (int, float)):
            return
        probability = float(value)
        _require(
            math.isfinite(probability) and 0.0 <= probability <= 1.0,
            f"config stochastic probability {prefix} is invalid",
        )
        modules.append(
            {
                "name": f"config.{prefix}",
                "class_name": "ModelConfigProbability",
                "kind": "dropout",
                "active": bool(model.training and probability > 0.0),
            }
        )

    visit("", raw_config)
    modules.sort(key=lambda row: str(row["name"]))
    _require(
        len({str(row["name"]) for row in modules}) == len(modules), "stochastic audit names are duplicated"
    )
    _require(all(not row["active"] for row in modules), "a dropout/stochastic module is active")
    dropout_count = sum(row["kind"] == "dropout" for row in modules)
    config_probability_count = sum(str(row["name"]).startswith("config.") for row in modules)
    attention_dropout_count = sum("attention_dropout" in str(row["name"]).lower() for row in modules)
    _require(model.training, "stochastic audit must run in production training mode")
    _require(attention_dropout_count >= 1, "attention_dropout was not enumerated from model config")
    return {
        "enumeration_complete": True,
        "dropout_inactive": True,
        "all_stochastic_modules_inactive": True,
        "dropout_module_count": dropout_count,
        "stochastic_module_count": len(modules),
        "functional_stochasticity_audit": {
            "runtime_training_mode": True,
            "model_config_dropout_fields_enumerated": True,
            "config_probability_count": config_probability_count,
            "attention_dropout_fields_enumerated": attention_dropout_count,
            "effective_probability_zero": True,
        },
        "modules": modules,
        "modules_sha256": semantic_digest(modules),
    }


@dataclass(frozen=True)
class CorpusRow:
    sample_id: str
    view: str
    prompt: str
    action: int
    token_length: int


@dataclass(frozen=True)
class QualificationCorpus:
    panel_id: str
    law_family: str
    prompts: tuple[str, ...]
    actions: tuple[int, ...]
    candidate_choices: tuple[tuple[int, int, int], ...]
    sample_ids: tuple[str, ...]
    views: tuple[str, ...]
    evaluation_rows: Mapping[str, CorpusRow]
    worst: CorpusRow
    dataset_sha256: str
    view_counts: Mapping[str, int]
    worst_index: int
    semantic_scene_digests: frozenset[str]


@dataclass(frozen=True)
class EvaluationBenchmark:
    """Exact production-shaped prompt banks for all thirteen boundaries."""

    diagnostic_prompts: tuple[str, ...]
    final_prompts: tuple[str, ...]
    registered_prompts: tuple[str, ...]
    diagnostic_banks: tuple[tuple[tuple[str, ...], ...], ...]
    final_banks: tuple[tuple[tuple[str, ...], ...], ...]
    registered_groups: tuple[tuple[str, ...], ...]
    registered_row_ids: tuple[str, ...]
    registered_example_ids: tuple[str, ...]
    registered_token_shapes: tuple[tuple[int, ...], ...]
    registered_bank_name: str
    registered_start_index: int
    registered_example_ids_sha256: str
    registered_token_shape_sha256: str
    diagnostic_group_token_lengths: tuple[tuple[tuple[int, ...], ...], ...]
    final_group_token_lengths: tuple[tuple[tuple[int, ...], ...], ...]
    diagnostic_prompt_sha256: str
    final_prompt_sha256: str
    registered_prompt_sha256: str
    bank_counts: Mapping[str, int]
    maximum_token_length: int


@dataclass(frozen=True)
class GlobalTrainingMaximum:
    """Private prompt payload plus its compact exhaustive public proof."""

    prompt: str
    action: int
    candidate_choices: tuple[int, int, int]
    host_envelope_prompts: Mapping[str, tuple[str, ...]]
    proof: Mapping[str, Any]


def build_global_training_maximum(
    *,
    repo: str | Path,
    panel_id: str,
    tokenizer: Any,
    deadline_monotonic: float | None = None,
) -> GlobalTrainingMaximum:
    """Exhaustively bind the longest registered production training prompt."""

    scan_started = time.perf_counter()
    resolved_repo = Path(repo).resolve()
    _require(panel_id in PANELS, "unknown global-training-maximum panel")
    specifications_by_profile: dict[str, tuple[Any, ...]] = {}
    plan_bindings: dict[str, dict[str, Any]] = {}
    for profile in PROFILES:
        specification = PROFILE_CONFIG_SPECS[profile][panel_id]
        config = load_config(resolved_repo / str(specification["path"]))
        plans = build_plan(config)
        _require(len(plans) == 80, "registered panel plan does not contain exactly eighty runs")
        specifications_by_profile[profile] = plans
        plan_path = resolved_repo / str(specification["plan_path"])
        _require(plan_path.is_file() and not plan_path.is_symlink(), "registered H200 plan is absent")
        plan_bindings[profile] = {
            "path": str(specification["plan_path"]),
            "file_sha256": sha256_file(plan_path),
            "run_count": len(plans),
            "plan_key_sha256": semantic_digest([str(item.plan_key) for item in plans]),
        }
    baseline_plans = specifications_by_profile["baseline"]
    tuned_plans = specifications_by_profile["tuned"]
    _require(
        all(
            baseline.seed == tuned.seed
            and dict(baseline.seeds) == dict(tuned.seeds)
            and baseline.cell_index == tuned.cell_index
            and baseline.global_index == tuned.global_index
            and baseline.config["data"] == tuned.config["data"]
            and baseline.config["model"] == tuned.config["model"]
            for baseline, tuned in zip(baseline_plans, tuned_plans, strict=True)
        ),
        "baseline/tuned registered prompt-generation inputs differ",
    )

    run_receipts: list[dict[str, Any]] = []
    longest_rank: tuple[int, str, str, int] | None = None
    longest_prompt = ""
    longest_action = 0
    longest_candidates = (0, 0, 0)
    longest_prompt_token_length = 0
    longest_identity: dict[str, Any] | None = None
    host_candidates: dict[str, tuple[dict[str, Any], str]] = {}
    for specification in baseline_plans:
        banks = materialize_banks(specification.config, specification.seeds)
        rendered = render_experiment(
            specification.config,
            banks,
            tokenizer,
            int(specification.seeds["rendering"]),
        )
        _require(
            len(rendered.train_prompts)
            == len(rendered.train_actions)
            == len(rendered.train_candidate_choices)
            == CORPUS_SIZE,
            "registered production training prompt bank is not exactly 10,000 rows",
        )
        lengths: list[int] = []
        prompt_hasher = hashlib.sha256()
        contextual_count_hasher = hashlib.sha256()
        run_prompt_maximum = 0
        run_scorer_branch_maximum = 0
        run_contextual_work_maximum = 0
        run_utf8_work_maximum = 0
        for prompt_index, (prompt, action, candidates) in enumerate(
            zip(
                rendered.train_prompts,
                rendered.train_actions,
                rendered.train_candidate_choices,
                strict=True,
            )
        ):
            if prompt_index % 256 == 0 and deadline_monotonic is not None:
                _require(
                    time.monotonic() < deadline_monotonic,
                    "controller global-training-maximum scan exceeded the producer deadline",
                )
            token_ids = tokenizer.encode(prompt, add_special_tokens=False)
            _require(isinstance(token_ids, list) and bool(token_ids), "production prompt tokenization failed")
            token_length = len(token_ids)
            prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            contextual = encode_action_continuations(
                tokenizer,
                (prompt,),
                ACTION_LABELS,
                add_prompt_special_tokens=False,
            )
            _require(
                len(contextual.prompt_tokens) == 1 and len(contextual.prompt_tokens[0]) == token_length,
                "contextual prompt tokenization differs from the registered prompt encoding",
            )
            continuation_a, continuation_b = contextual.action_tokens[0]
            _require(
                len(continuation_a) == len(continuation_b) == 1,
                "registered production prompt does not have one-token A/B contextual continuations",
            )
            prompt_utf8_bytes = len(prompt.encode("utf-8"))
            prompt_a_utf8_bytes = len((prompt + ACTION_LABELS[0]).encode("utf-8"))
            prompt_b_utf8_bytes = len((prompt + ACTION_LABELS[1]).encode("utf-8"))
            prompt_a_token_length = token_length + len(continuation_a)
            prompt_b_token_length = token_length + len(continuation_b)
            host_row = {
                "baseline_plan_key": str(specification.plan_key),
                "global_index": int(specification.global_index),
                "prompt_index": prompt_index,
                "prompt_sha256": prompt_sha256,
                "prompt_utf8_bytes": prompt_utf8_bytes,
                "prompt_token_length": token_length,
                "prompt_a_utf8_bytes": prompt_a_utf8_bytes,
                "prompt_a_token_length": prompt_a_token_length,
                "prompt_b_utf8_bytes": prompt_b_utf8_bytes,
                "prompt_b_token_length": prompt_b_token_length,
                "continuation_a_token_count": len(continuation_a),
                "continuation_a_token_ids_sha256": semantic_digest(list(continuation_a)),
                "continuation_b_token_count": len(continuation_b),
                "continuation_b_token_ids_sha256": semantic_digest(list(continuation_b)),
            }
            total_contextual_token_work = token_length + prompt_a_token_length + prompt_b_token_length
            host_row["total_contextual_token_work"] = total_contextual_token_work
            host_row["maximum_contextual_token_length"] = max(
                token_length,
                prompt_a_token_length,
                prompt_b_token_length,
            )
            total_utf8_bytes = prompt_utf8_bytes + prompt_a_utf8_bytes + prompt_b_utf8_bytes
            host_row["total_utf8_bytes"] = total_utf8_bytes
            host_row["maximum_utf8_bytes"] = max(
                prompt_utf8_bytes,
                prompt_a_utf8_bytes,
                prompt_b_utf8_bytes,
            )
            # Duplicate rendered prompts have identical tokenizer/UTF-8 work.  The
            # first production-plan occurrence is the canonical identity.
            if prompt_sha256 not in host_candidates:
                host_candidates[prompt_sha256] = (host_row, str(prompt))
            run_contextual_work_maximum = max(
                run_contextual_work_maximum,
                total_contextual_token_work,
            )
            run_utf8_work_maximum = max(
                run_utf8_work_maximum,
                total_utf8_bytes,
            )
            lengths.append(token_length)
            scorer_branch_length = max(
                token_length,
                prompt_a_token_length,
                prompt_b_token_length,
            )
            run_prompt_maximum = max(run_prompt_maximum, token_length)
            run_scorer_branch_maximum = max(run_scorer_branch_maximum, scorer_branch_length)
            contextual_count_row = canonical_json_bytes(
                {
                    "prompt_index": prompt_index,
                    "prompt_sha256": prompt_sha256,
                    "continuation_a_token_count": len(continuation_a),
                    "continuation_a_token_ids_sha256": semantic_digest(list(continuation_a)),
                    "continuation_b_token_count": len(continuation_b),
                    "continuation_b_token_ids_sha256": semantic_digest(list(continuation_b)),
                    "maximum_scorer_branch_token_length": scorer_branch_length,
                }
            )
            contextual_count_hasher.update(len(contextual_count_row).to_bytes(8, "big"))
            contextual_count_hasher.update(contextual_count_row)
            row_bytes = canonical_json_bytes(
                {
                    "prompt_index": prompt_index,
                    "prompt_sha256": prompt_sha256,
                    **host_row,
                }
            )
            prompt_hasher.update(len(row_bytes).to_bytes(8, "big"))
            prompt_hasher.update(row_bytes)
            rank = (scorer_branch_length, prompt_sha256, str(specification.plan_key), prompt_index)
            if longest_rank is None or rank > longest_rank:
                longest_rank = rank
                longest_prompt = str(prompt)
                longest_action = int(action)
                longest_candidates = cast(
                    tuple[int, int, int],
                    tuple(int(value) for value in candidates),
                )
                longest_prompt_token_length = token_length
                longest_identity = {
                    "baseline_plan_key": str(specification.plan_key),
                    "global_index": int(specification.global_index),
                    "prompt_index": prompt_index,
                    "seed": int(specification.seed),
                    "law_family": str(specification.config["data"]["rule_family"]),
                    "training_view": str(specification.config["data"]["training_view"]),
                }
        renderer_counts = dict(rendered.metadata["training_renderer_counts"])
        run_receipts.append(
            {
                "global_index": int(specification.global_index),
                "baseline_plan_key": str(specification.plan_key),
                "seed": int(specification.seed),
                "derived_seeds_sha256": semantic_digest(dict(specification.seeds)),
                "law_family": str(specification.config["data"]["rule_family"]),
                "training_view": str(specification.config["data"]["training_view"]),
                "train_renderers": list(specification.config["data"]["train_renderers"]),
                "training_renderer_counts": renderer_counts,
                "prompt_count": len(lengths),
                "training_prompt_sha256": str(rendered.metadata["training_prompt_digest"]),
                "token_length_stream_sha256": prompt_hasher.hexdigest(),
                "contextual_continuation_stream_sha256": contextual_count_hasher.hexdigest(),
                "maximum_prompt_token_length": run_prompt_maximum,
                "maximum_token_length": run_scorer_branch_maximum,
                "maximum_contextual_token_work": run_contextual_work_maximum,
                "maximum_utf8_bytes_work": run_utf8_work_maximum,
            }
        )
    _require(longest_rank is not None and longest_identity is not None, "global longest prompt is absent")
    longest_rank = cast(tuple[int, str, str, int], longest_rank)
    _require(len(host_candidates) >= 10, "exhaustive scan did not produce ten distinct host rows")

    def host_envelope(
        metric: str,
        *,
        role: str,
    ) -> tuple[list[dict[str, Any]], tuple[str, ...], dict[str, Any]]:
        selected = sorted(
            host_candidates.values(),
            key=lambda item: (int(item[0][metric]), str(item[0]["prompt_sha256"])),
            reverse=True,
        )[: (10 if role == "variability_stress" else 1)]
        rows = [copy.deepcopy(item[0]) for item in selected]
        prompts = tuple(item[1] for item in selected)
        source_digests = [str(row["prompt_sha256"]) for row in rows]
        repetition_count = 5 if role == "variability_stress" else 50
        execution_digests = source_digests * repetition_count
        execution_rows = rows * repetition_count
        metric_values = [int(row[metric]) for row in rows]
        _require(
            len(execution_rows) == 50 and role in {"variability_stress", "hard_dominance"},
            "host envelope role or execution cardinality changed",
        )
        baseline_partitions = [execution_digests[index : index + 10] for index in range(0, 50, 10)]
        body = {
            "role": role,
            "selection_algorithm": (
                f"global_top10_distinct_by_{metric}_desc_prompt_sha256_tiebreak_v1"
                if role == "variability_stress"
                else f"global_maximum_by_{metric}_desc_prompt_sha256_tiebreak_v1"
            ),
            "metric_field": metric,
            "source_row_count": len(rows),
            "rows": rows,
            "rows_sha256": semantic_digest(rows),
            "selected_metric_sum": sum(metric_values),
            "selected_metric_maximum": max(metric_values),
            "repetition_count": repetition_count,
            "execution_row_count_per_update": len(execution_rows),
            "ordered_execution_prompt_sha256": semantic_digest(execution_digests),
            "ordered_execution_rows_sha256": semantic_digest(execution_rows),
            "baseline_call_partition_sizes": [10, 10, 10, 10, 10],
            "baseline_call_partitions_sha256": semantic_digest(baseline_partitions),
            "tuned_call_partition_sizes": [50],
            "tuned_call_partitions_sha256": semantic_digest([execution_digests]),
            "variability_stress_only": role == "variability_stress",
            "hard_dominance_tier": role == "hard_dominance",
            "each_baseline_call_dominates_any_registered_10_row_call": role == "hard_dominance",
            "tuned_ordered50_is_identical_to_concatenated_baseline_calls": True,
            "tuned_total_dominates_any_registered_50_row_batch": role == "hard_dominance",
            "global_metric_maximum_over_all_registered_occurrences": role == "hard_dominance",
        }
        return rows, prompts, body

    _token_rows, token_prompts, token_envelope = host_envelope(
        "total_contextual_token_work",
        role="variability_stress",
    )
    _utf8_rows, utf8_prompts, utf8_envelope = host_envelope(
        "total_utf8_bytes",
        role="variability_stress",
    )
    _token_dominance_rows, token_dominance_prompts, token_dominance_envelope = host_envelope(
        "total_contextual_token_work",
        role="hard_dominance",
    )
    _utf8_dominance_rows, utf8_dominance_prompts, utf8_dominance_envelope = host_envelope(
        "total_utf8_bytes",
        role="hard_dominance",
    )
    host_envelope_prompts = {
        "contextual_token_work_variability": token_prompts,
        "utf8_byte_work_variability": utf8_prompts,
        "contextual_token_work_dominance": token_dominance_prompts,
        "utf8_byte_work_dominance": utf8_dominance_prompts,
    }
    standalone_action_tokens = [
        list(tokenizer.encode(label, add_special_tokens=False)) for label in ACTION_LABELS
    ]
    public_body = {
        "algorithm": "exhaustive_registered_80_run_training_prompt_maximum_v1",
        "panel_id": panel_id,
        "profile_plan_bindings": plan_bindings,
        "prompt_generation_inputs_identical_across_profiles": True,
        "registered_run_count": len(run_receipts),
        "registered_training_prompt_count": sum(int(item["prompt_count"]) for item in run_receipts),
        "law_families": sorted({str(item["law_family"]) for item in run_receipts}),
        "training_views": sorted({str(item["training_view"]) for item in run_receipts}),
        "train_renderers": sorted(
            {str(renderer) for item in run_receipts for renderer in item["train_renderers"]}
        ),
        "run_receipts": run_receipts,
        "run_receipts_sha256": semantic_digest(run_receipts),
        "maximum_token_length": int(longest_rank[0]),
        "maximum_prompt_token_length": longest_prompt_token_length,
        "maximum_prompt_sha256": hashlib.sha256(longest_prompt.encode("utf-8")).hexdigest(),
        "maximum_prompt_identity": longest_identity,
        "tokenizer_host_envelopes": {
            "contextual_continuation_mode": (
                "goalzendo.modeling.encode_action_continuations:add_prompt_special_tokens=false"
            ),
            "action_labels": list(ACTION_LABELS),
            "standalone_action_token_ids": standalone_action_tokens,
            "standalone_action_token_ids_sha256": semantic_digest(standalone_action_tokens),
            "envelope_order": [
                "contextual_token_work_variability",
                "utf8_byte_work_variability",
                "contextual_token_work_dominance",
                "utf8_byte_work_dominance",
            ],
            "envelopes": {
                "contextual_token_work_variability": token_envelope,
                "utf8_byte_work_variability": utf8_envelope,
                "contextual_token_work_dominance": token_dominance_envelope,
                "utf8_byte_work_dominance": utf8_dominance_envelope,
            },
            "measured_seconds_aggregation": "sum_all_four_envelopes",
        },
        "contextual_scorer_branch_invariant": {
            "contextual_continuation_mode": (
                "goalzendo.modeling.encode_action_continuations:add_prompt_special_tokens=false"
            ),
            "action_labels": list(ACTION_LABELS),
            "required_token_count_per_action_continuation": 1,
            "registered_prompt_count": sum(int(item["prompt_count"]) for item in run_receipts),
            "per_run_stream_digest_field": "contextual_continuation_stream_sha256",
            "every_registered_a_and_b_continuation_exactly_one_token": True,
            "maximum_prompt_token_length": longest_prompt_token_length,
            "maximum_scorer_branch_token_length": int(longest_rank[0]),
            "maximum_scorer_branch_is_prompt_plus_one_token": True,
            "global_maximum_ranked_over_prompt_and_both_action_branches": True,
        },
        "global_maximum_dominates_every_registered_training_prompt": True,
        "stream_encoding": "uint64be_length_prefixed_canonical_json_rows_v1",
        "controller_scan_seconds": max(time.perf_counter() - scan_started, 1e-9),
    }
    proof = {**public_body, "proof_digest": semantic_digest(public_body)}
    return GlobalTrainingMaximum(
        prompt=longest_prompt,
        action=longest_action,
        candidate_choices=longest_candidates,
        host_envelope_prompts=host_envelope_prompts,
        proof=proof,
    )


def build_controller_training_maximums(
    *,
    repo: str | Path,
    snapshot_roots: Mapping[str, str],
    deadline_monotonic: float,
) -> dict[str, dict[str, Any]]:
    """Perform each exhaustive panel scan once in the authenticated controller."""

    _require(os.environ.get("HF_HUB_OFFLINE") == "1", "controller maximum scan requires HF_HUB_OFFLINE=1")
    _require(
        os.environ.get("TRANSFORMERS_OFFLINE") == "1",
        "controller maximum scan requires TRANSFORMERS_OFFLINE=1",
    )
    try:
        from transformers import AutoTokenizer  # type: ignore[import-not-found]
    except ImportError as error:  # pragma: no cover - actual image dependency
        raise ProducerError("actual qualification requires transformers") from error
    result: dict[str, dict[str, Any]] = {}
    for panel_id in PANELS:
        tokenizer = AutoTokenizer.from_pretrained(
            snapshot_roots[panel_id],
            local_files_only=True,
            trust_remote_code=False,
        )
        maximum = build_global_training_maximum(
            repo=repo,
            panel_id=panel_id,
            tokenizer=tokenizer,
            deadline_monotonic=deadline_monotonic,
        )
        result[panel_id] = {
            "prompt": maximum.prompt,
            "action": maximum.action,
            "candidate_choices": list(maximum.candidate_choices),
            "host_envelope_prompts": {
                name: list(prompts) for name, prompts in maximum.host_envelope_prompts.items()
            },
            "proof": copy.deepcopy(dict(maximum.proof)),
        }
        del tokenizer
    return result


def _evaluation_workload(
    benchmark: EvaluationBenchmark,
    example_batch_size: int,
) -> dict[str, Any]:
    padded = 0
    calls = 0
    maximum_flattened_prompts = 0
    maximum_padded_elements_per_call = 0
    call_shapes: list[dict[str, int | str]] = []
    for boundary_ordinal in range(EVALUATION_BOUNDARY_COUNT):
        kind = "final" if boundary_ordinal == EVALUATION_BOUNDARY_COUNT - 1 else "diagnostic"
        banks = (
            benchmark.final_group_token_lengths
            if kind == "final"
            else benchmark.diagnostic_group_token_lengths
        )
        for bank_index, bank in enumerate(banks):
            for chunk_index, start in enumerate(range(0, len(bank), example_batch_size)):
                chunk = bank[start : start + example_batch_size]
                lengths = [length for group in chunk for length in group]
                elements = len(lengths) * max(lengths)
                shape: dict[str, int | str] = {
                    "boundary_ordinal": boundary_ordinal,
                    "kind": kind,
                    "bank_index": bank_index,
                    "chunk_index": chunk_index,
                    "example_count": len(chunk),
                    "flattened_prompt_count": len(lengths),
                    "max_seq_len": max(lengths),
                    "padded_elements": elements,
                }
                call_shapes.append(shape)
                padded += elements
                calls += 1
                maximum_flattened_prompts = max(maximum_flattened_prompts, len(lengths))
                maximum_padded_elements_per_call = max(maximum_padded_elements_per_call, elements)
    return {
        "padded_token_elements": padded,
        "scorer_call_count": calls,
        "maximum_flattened_prompts_per_call": maximum_flattened_prompts,
        "maximum_padded_token_elements_per_call": maximum_padded_elements_per_call,
        "maximum_token_length": benchmark.maximum_token_length,
        "call_shapes": call_shapes,
        "call_shapes_sha256": semantic_digest(call_shapes),
    }


def _chat_prompt(tokenizer: Any, plain: str) -> str:
    system = (
        "Play GoalZendo. 'A' means Koan A and 'B' means Koan B. "
        "Reply with exactly one of those labels and no explanation."
    )
    return format_chat_prompt(
        tokenizer,
        plain,
        system_prompt=system,
        enable_thinking=False,
    )


def _qualification_config_and_banks(
    *, repo: str | Path, panel_id: str, law_family: str
) -> tuple[dict[str, Any], Any, dict[str, int]]:
    config_path = Path(repo).resolve() / str(PROFILE_CONFIG_SPECS["baseline"][panel_id]["path"])
    config = copy.deepcopy(load_config(config_path))
    config["data"]["rule_family"] = law_family
    seeds = derived_seeds(CORPUS_SEEDS[law_family])
    return config, materialize_banks(config, seeds), seeds


def build_qualification_corpus(
    *,
    repo: str | Path,
    panel_id: str,
    law_family: str,
    tokenizer: Any,
    prepared: tuple[dict[str, Any], Any, dict[str, int]] | None = None,
) -> QualificationCorpus:
    """Build the frozen disjoint 10k engineering corpus using GoalZendo core."""

    _require(panel_id in PANELS and law_family in LAW_FAMILIES, "unknown corpus panel or Law")
    config, banks, _seeds = (
        _qualification_config_and_banks(repo=repo, panel_id=panel_id, law_family=law_family)
        if prepared is None
        else prepared
    )
    dataset = banks.train
    _require(
        len(banks.effective_train_decisions) == CORPUS_SIZE,
        "qualification training bank is not exactly 10,000 rows",
    )
    data = config["data"]
    renderers = tuple(str(value) for value in data["train_renderers"])
    _require(len(renderers) == 4, "qualification corpus requires four frozen renderers")
    rows: list[CorpusRow] = []
    for index, decision in enumerate(banks.effective_train_decisions):
        view = TRAINING_VIEWS[index % len(TRAINING_VIEWS)]
        renderer = renderers[index % len(renderers)]
        plain = render_prompt_view(
            decision,
            dataset.feature_names,
            renderer_id=renderer,
            prompt_view=view,
        )
        prompt = _chat_prompt(tokenizer, plain)
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        _require(isinstance(tokens, list) and bool(tokens), "qualification prompt tokenization failed")
        rows.append(
            CorpusRow(
                sample_id=f"{decision.sample_id}:{view}",
                view=view,
                prompt=prompt,
                action=int(decision.choice_y),
                token_length=len(tokens),
            )
        )
    _require(
        len(rows) == CORPUS_SIZE and len({row.sample_id for row in rows}) == CORPUS_SIZE,
        "qualification corpus is incomplete or duplicated",
    )
    evaluation_rows = {
        view: max(
            (row for row in rows if row.view == view),
            key=lambda row: (row.token_length, row.sample_id),
        )
        for view in TRAINING_VIEWS
    }
    worst = max(rows, key=lambda row: (row.token_length, row.sample_id))
    worst_index = rows.index(worst)
    view_counts = {view: sum(row.view == view for row in rows) for view in TRAINING_VIEWS}
    _require(
        max(view_counts.values()) - min(view_counts.values()) <= 1,
        "qualification corpus view coverage is not balanced",
    )
    dataset_sha256 = semantic_digest(
        [
            {
                "sample_id": row.sample_id,
                "view": row.view,
                "prompt_sha256": hashlib.sha256(row.prompt.encode("utf-8")).hexdigest(),
                "action": row.action,
                "token_length": row.token_length,
            }
            for row in rows
        ]
    )
    return QualificationCorpus(
        panel_id=panel_id,
        law_family=law_family,
        prompts=tuple(row.prompt for row in rows),
        actions=tuple(row.action for row in rows),
        candidate_choices=tuple(
            (int(decision.choice_y), int(decision.choice_p), int(decision.choice_q))
            for decision in banks.effective_train_decisions
        ),
        sample_ids=tuple(row.sample_id for row in rows),
        views=tuple(row.view for row in rows),
        evaluation_rows=evaluation_rows,
        worst=worst,
        dataset_sha256=dataset_sha256,
        view_counts=view_counts,
        worst_index=worst_index,
        semantic_scene_digests=frozenset(
            koan.scene.semantic_digest
            for decision in banks.effective_train_decisions
            for koan in decision.koans
        ),
    )


def build_evaluation_benchmark(
    *,
    repo: str | Path,
    panel_id: str,
    law_family: str,
    tokenizer: Any,
    prepared: tuple[dict[str, Any], Any, dict[str, int]] | None = None,
) -> EvaluationBenchmark:
    """Render production banks and one registered production-shaped parity chunk."""

    _require(panel_id in PANELS and law_family in LAW_FAMILIES, "unknown benchmark panel or Law")
    config, banks, seeds = (
        _qualification_config_and_banks(repo=repo, panel_id=panel_id, law_family=law_family)
        if prepared is None
        else prepared
    )
    rendered = render_experiment(config, banks, tokenizer, int(seeds["rendering"]))

    def groups(examples: Sequence[Mapping[str, Any]]) -> tuple[tuple[str, ...], ...]:
        return tuple(
            tuple(str(prompt) for _view, prompt in sorted(dict(example["prompt_views"]).items()))
            for example in examples
        )

    def group_lengths(examples: Sequence[Mapping[str, Any]]) -> tuple[tuple[int, ...], ...]:
        return tuple(
            tuple(
                len(tokenizer.encode(str(prompt), add_special_tokens=False))
                for _view, prompt in sorted(dict(example["prompt_views"]).items())
            )
            for example in examples
        )

    def flatten(examples: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
        return tuple(
            str(prompt)
            for example in examples
            for _view, prompt in sorted(dict(example["prompt_views"]).items())
        )

    validation = flatten(rendered.validation_examples)
    diagnostic_factorial = flatten(rendered.diagnostic_factorial_examples)
    final_factorial = flatten(rendered.final_factorial_examples)
    diagnostic_causal = flatten(rendered.diagnostic_causal_examples)
    final_causal = flatten(rendered.final_causal_examples)
    diagnostic = (*validation, *diagnostic_factorial, *diagnostic_causal)
    final = (*validation, *final_factorial, *final_causal)
    _require(bool(diagnostic) and bool(final), "production-shaped evaluation banks are empty")
    named_banks = (
        ("validation", rendered.validation_examples),
        ("diagnostic_factorial", rendered.diagnostic_factorial_examples),
        ("diagnostic_causal", rendered.diagnostic_causal_examples),
        ("final_factorial", rendered.final_factorial_examples),
        ("final_causal", rendered.final_causal_examples),
    )
    registered_candidate: tuple[tuple[int, int, str, int], str, int, Sequence[Mapping[str, Any]]] | None = (
        None
    )
    for bank_name, examples in named_banks:
        if len(examples) < 128:
            continue
        lengths = group_lengths(examples)
        for start in range(0, len(examples) - 128 + 1):
            window_lengths = lengths[start : start + 128]
            if not all(len(group) == len(TRAINING_VIEWS) for group in window_lengths):
                continue
            maximum = max(length for group in window_lengths for length in group)
            padded_elements = len(window_lengths) * len(TRAINING_VIEWS) * maximum
            identity = semantic_digest(
                [str(example["sample_id"]) for example in examples[start : start + 128]]
            )
            candidate = (
                (padded_elements, maximum, identity, -start),
                bank_name,
                start,
                examples[start : start + 128],
            )
            if registered_candidate is None or candidate[0] > registered_candidate[0]:
                registered_candidate = candidate
    if registered_candidate is None:
        raise ProducerError("no contiguous 128-example six-view production chunk exists")
    _rank, registered_bank_name, registered_start_index, registered_examples = registered_candidate
    registered_groups = groups(registered_examples)
    registered = tuple(prompt for group in registered_groups for prompt in group)
    registered_row_ids = tuple(
        f"{example['sample_id']}:{view}"
        for example in registered_examples
        for view, _prompt in sorted(dict(example["prompt_views"]).items())
    )
    _require(
        len(registered_groups) == 128
        and all(len(group) == len(TRAINING_VIEWS) for group in registered_groups)
        and len(registered) == 128 * len(TRAINING_VIEWS),
        "production-shaped worst probe is not one contiguous 128-example six-view chunk",
    )
    registered_example_ids = [str(example["sample_id"]) for example in registered_examples]
    registered_lengths = group_lengths(registered_examples)
    return EvaluationBenchmark(
        diagnostic_prompts=tuple(diagnostic),
        final_prompts=tuple(final),
        registered_prompts=registered,
        diagnostic_banks=(
            groups(rendered.validation_examples),
            groups(rendered.diagnostic_factorial_examples),
            groups(rendered.diagnostic_causal_examples),
        ),
        final_banks=(
            groups(rendered.validation_examples),
            groups(rendered.final_factorial_examples),
            groups(rendered.final_causal_examples),
        ),
        registered_groups=registered_groups,
        registered_row_ids=registered_row_ids,
        registered_example_ids=tuple(registered_example_ids),
        registered_token_shapes=registered_lengths,
        registered_bank_name=registered_bank_name,
        registered_start_index=registered_start_index,
        registered_example_ids_sha256=semantic_digest(registered_example_ids),
        registered_token_shape_sha256=semantic_digest([list(group) for group in registered_lengths]),
        diagnostic_group_token_lengths=(
            group_lengths(rendered.validation_examples),
            group_lengths(rendered.diagnostic_factorial_examples),
            group_lengths(rendered.diagnostic_causal_examples),
        ),
        final_group_token_lengths=(
            group_lengths(rendered.validation_examples),
            group_lengths(rendered.final_factorial_examples),
            group_lengths(rendered.final_causal_examples),
        ),
        diagnostic_prompt_sha256=semantic_digest(list(diagnostic)),
        final_prompt_sha256=semantic_digest(list(final)),
        registered_prompt_sha256=semantic_digest(list(registered)),
        bank_counts={
            "validation": len(validation),
            "diagnostic_factorial": len(diagnostic_factorial),
            "diagnostic_causal": len(diagnostic_causal),
            "final_factorial": len(final_factorial),
            "final_causal": len(final_causal),
            "diagnostic_boundary_total": len(diagnostic),
            "final_boundary_total": len(final),
            "registered_worst_length": len(registered),
            "registered_worst_length_examples": len(registered_groups),
        },
        maximum_token_length=max(
            len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in (*diagnostic, *final)
        ),
    )


@dataclass(frozen=True)
class CapturedProcess:
    record: Mapping[str, Any]
    vector_files: Mapping[tuple[int, str], Path]


class WorkerEngine(Protocol):
    def prepare_corpora(self) -> None: ...

    def timing_law(self, profile: str, panel_id: str) -> str: ...

    def run(
        self,
        *,
        profile: str,
        replicate: str,
        panel_id: str,
        law_family: str,
        process_uuid: str,
        persist_vectors_under: Path | None,
        full_timing: bool,
    ) -> CapturedProcess: ...


def _qualification_update_order(
    corpus: QualificationCorpus,
    update: int,
) -> tuple[list[int], dict[str, Any]]:
    """Return the shared effective-50 order with an honest worst-row injection."""

    base = [
        int(index)
        for index in deterministic_batch_indices(
            CORPUS_SIZE,
            50,
            update - 1,
            seed=ENGINEERING_SEED,
        ).tolist()
    ]
    _require(len(base) == 50 and len(set(base)) == 50, "base effective-50 order is invalid")
    final = list(base)
    inserted = corpus.worst_index not in final
    replaced_sample_id: str | None = None
    if inserted:
        replaced_sample_id = corpus.sample_ids[final[-1]]
        final[-1] = corpus.worst_index
    position = final.index(corpus.worst_index)
    _require(
        len(set(final)) == 50 and corpus.worst_index in final, "worst-shaped training order injection failed"
    )
    base_ids = [corpus.sample_ids[index] for index in base]
    final_ids = [corpus.sample_ids[index] for index in final]
    return final, {
        "algorithm": "deterministic_effective_50_replace_final_with_corpus_max_if_absent_v1",
        "engineering_seed": ENGINEERING_SEED,
        "base_order_sha256": semantic_digest(base_ids),
        "inserted": inserted,
        "position": position,
        "worst_sample_id": corpus.worst.sample_id,
        "replaced_sample_id": replaced_sample_id,
        "final_order_sha256": semantic_digest(final_ids),
    }


class ActualModelWorkerEngine:
    """Fresh-model AdamW qualification execution on one isolated H200."""

    def __init__(
        self,
        *,
        repo: Path,
        snapshot_roots: Mapping[str, str],
        model_snapshot_sha256: Mapping[str, str],
        device: GPUDevice,
        engineering_scope: Mapping[str, Any],
        inventory: TransientInventory,
        profile_timing_order: Sequence[str],
        capacity_probe_identity: tuple[str, str] | None = None,
        training_maximum_bindings: Mapping[str, Any] | None = None,
    ) -> None:
        self.repo = repo
        self.snapshot_roots = dict(snapshot_roots)
        self.model_snapshot_sha256 = dict(model_snapshot_sha256)
        self.device_binding = device
        self.scope = copy.deepcopy(dict(engineering_scope))
        self.inventory = inventory
        self.profile_timing_order = tuple(profile_timing_order)
        _require(
            self.profile_timing_order in (PROFILES, tuple(reversed(PROFILES)), ("baseline",)),
            "worker profile timing order is invalid",
        )
        self.device = torch.device("cuda:0")
        self.tokenizers: dict[str, Any] = {}
        self.corpora: dict[tuple[str, str], QualificationCorpus] = {}
        self.evaluation_benchmarks: dict[tuple[str, str], EvaluationBenchmark] = {}
        self.preparation_seconds: dict[tuple[str, str], float] = {}
        self.tokenizer_initialization_seconds: dict[str, float] = {}
        self.training_maximums: dict[str, GlobalTrainingMaximum] = {}
        self.numerical_execution_receipt: Mapping[str, Any] | None = None
        self.capacity_probe_identity = capacity_probe_identity
        self.controller_training_maximum_bindings = (
            None if training_maximum_bindings is None else copy.deepcopy(dict(training_maximum_bindings))
        )
        _require(
            (capacity_probe_identity is not None and training_maximum_bindings is None)
            or (
                capacity_probe_identity is None
                and isinstance(training_maximum_bindings, Mapping)
                and set(training_maximum_bindings) == set(PANELS)
            ),
            "controller training-maximum handoff differs from worker kind",
        )
        self.capture_index = 0

    def _require_actual_runtime(self) -> None:
        _require(os.environ.get("HF_HUB_OFFLINE") == "1", "HF_HUB_OFFLINE=1 is required")
        _require(os.environ.get("TRANSFORMERS_OFFLINE") == "1", "TRANSFORMERS_OFFLINE=1 is required")
        _require(
            torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "worker must expose exactly one isolated CUDA device",
        )
        _require(
            torch.cuda.get_device_name(0).startswith("NVIDIA H200"),
            "worker CUDA device is not an actual H200",
        )
        _require(
            os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8",
            "deterministic CUBLAS workspace guard changed",
        )

    def prepare_corpora(self) -> None:
        self._require_actual_runtime()
        try:
            from transformers import AutoTokenizer
        except ImportError as error:  # pragma: no cover - actual image dependency
            raise ProducerError("actual qualification requires transformers") from error
        panels = PANELS if self.capacity_probe_identity is None else (self.capacity_probe_identity[0],)
        for panel_id in panels:
            snapshot_root = self.snapshot_roots[panel_id]
            tokenizer_started = time.perf_counter()
            tokenizer = AutoTokenizer.from_pretrained(
                snapshot_root,
                local_files_only=True,
                trust_remote_code=False,
            )
            if tokenizer.pad_token_id is None:
                _require(tokenizer.eos_token_id is not None, "tokenizer has no padding or EOS token")
                tokenizer.pad_token = tokenizer.eos_token
            self.tokenizer_initialization_seconds[panel_id] = max(
                time.perf_counter() - tokenizer_started,
                1e-9,
            )
            self.tokenizers[panel_id] = tokenizer
            if self.capacity_probe_identity is None:
                _require(
                    self.controller_training_maximum_bindings is not None,
                    "qualification worker lacks controller training-maximum evidence",
                )
                maximum_bindings = cast(Mapping[str, Any], self.controller_training_maximum_bindings)
                raw_maximum = maximum_bindings[panel_id]
                _require(
                    isinstance(raw_maximum, Mapping)
                    and set(raw_maximum)
                    == {"prompt", "action", "candidate_choices", "host_envelope_prompts", "proof"},
                    "controller training-maximum handoff is malformed",
                )
                prompt = str(raw_maximum["prompt"])
                candidates = tuple(int(value) for value in raw_maximum["candidate_choices"])
                proof = copy.deepcopy(dict(raw_maximum["proof"]))
                raw_host_prompts = raw_maximum["host_envelope_prompts"]
                _require(
                    isinstance(raw_host_prompts, Mapping)
                    and set(raw_host_prompts) == set(HOST_ENVELOPE_ORDER),
                    "controller tokenizer-host envelope prompt handoff is malformed",
                )
                host_prompts = {
                    name: tuple(str(value) for value in raw_host_prompts[name])
                    for name in HOST_ENVELOPE_ORDER
                }
                host_envelopes = proof["tokenizer_host_envelopes"]["envelopes"]
                maximum_contextual = encode_action_continuations(
                    tokenizer,
                    (prompt,),
                    ACTION_LABELS,
                    add_prompt_special_tokens=False,
                )
                maximum_a, maximum_b = maximum_contextual.action_tokens[0]
                _require(
                    len(candidates) == 3
                    and all(value in (0, 1) for value in candidates)
                    and int(raw_maximum["action"]) == candidates[0]
                    and hashlib.sha256(prompt.encode("utf-8")).hexdigest() == proof["maximum_prompt_sha256"]
                    and len(tokenizer.encode(prompt, add_special_tokens=False))
                    == proof["maximum_prompt_token_length"]
                    and len(maximum_contextual.prompt_tokens) == 1
                    and len(maximum_contextual.prompt_tokens[0]) == proof["maximum_prompt_token_length"]
                    and len(maximum_a) == len(maximum_b) == 1
                    and max(
                        len(maximum_contextual.prompt_tokens[0]),
                        len(maximum_contextual.prompt_tokens[0]) + len(maximum_a),
                        len(maximum_contextual.prompt_tokens[0]) + len(maximum_b),
                    )
                    == proof["maximum_token_length"],
                    "worker could not replay the controller global-maximum prompt binding",
                )
                _require(
                    all(
                        len(host_prompts[name])
                        == len(host_envelopes[name]["rows"])
                        == host_envelopes[name]["source_row_count"]
                        and all(
                            hashlib.sha256(host_prompt.encode("utf-8")).hexdigest()
                            == host_row["prompt_sha256"]
                            for host_prompt, host_row in zip(
                                host_prompts[name],
                                host_envelopes[name]["rows"],
                                strict=True,
                            )
                        )
                        for name in HOST_ENVELOPE_ORDER
                    ),
                    "worker could not replay the controller tokenizer-host envelope",
                )
                self.training_maximums[panel_id] = GlobalTrainingMaximum(
                    prompt=prompt,
                    action=int(raw_maximum["action"]),
                    candidate_choices=cast(tuple[int, int, int], candidates),
                    host_envelope_prompts=host_prompts,
                    proof=proof,
                )
            laws = (
                LAW_FAMILIES if self.capacity_probe_identity is None else (self.capacity_probe_identity[1],)
            )
            for law_family in laws:
                started = time.perf_counter()
                prepared = _qualification_config_and_banks(
                    repo=self.repo,
                    panel_id=panel_id,
                    law_family=law_family,
                )
                if self.numerical_execution_receipt is None:
                    self.numerical_execution_receipt = configure_numerical_execution(prepared[0])
                self.corpora[(panel_id, law_family)] = build_qualification_corpus(
                    repo=self.repo,
                    panel_id=panel_id,
                    law_family=law_family,
                    tokenizer=tokenizer,
                    prepared=prepared,
                )
                self.evaluation_benchmarks[(panel_id, law_family)] = build_evaluation_benchmark(
                    repo=self.repo,
                    panel_id=panel_id,
                    law_family=law_family,
                    tokenizer=tokenizer,
                    prepared=prepared,
                )
                self.preparation_seconds[(panel_id, law_family)] = max(
                    time.perf_counter() - started,
                    1e-9,
                )

    def timing_law(self, profile: str, panel_id: str) -> str:
        _require(profile in PROFILES, "unknown timing profile")
        _require(
            all((panel_id, law) in self.evaluation_benchmarks for law in LAW_FAMILIES),
            "evaluation benchmarks are not prepared",
        )
        batch_size = int(PROFILE_CONTRACT[profile]["evaluation_batch_size"])
        workloads = {
            law: _evaluation_workload(self.evaluation_benchmarks[(panel_id, law)], batch_size)
            for law in LAW_FAMILIES
        }
        fields = (
            "padded_token_elements",
            "maximum_padded_token_elements_per_call",
            "maximum_token_length",
            "maximum_flattened_prompts_per_call",
            "scorer_call_count",
        )

        def dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
            left_shapes = left["call_shapes"]
            right_shapes = right["call_shapes"]
            return bool(
                all(int(left[field]) >= int(right[field]) for field in fields)
                and len(left_shapes) == len(right_shapes)
                and all(
                    tuple(
                        left_row[field] for field in ("boundary_ordinal", "kind", "bank_index", "chunk_index")
                    )
                    == tuple(
                        right_row[field]
                        for field in ("boundary_ordinal", "kind", "bank_index", "chunk_index")
                    )
                    and all(
                        int(left_row[field]) >= int(right_row[field])
                        for field in (
                            "example_count",
                            "flattened_prompt_count",
                            "max_seq_len",
                            "padded_elements",
                        )
                    )
                    for left_row, right_row in zip(left_shapes, right_shapes, strict=True)
                )
            )

        dominating = [
            law
            for law in LAW_FAMILIES
            if all(dominates(workloads[law], workloads[other]) for other in LAW_FAMILIES if other != law)
        ]
        _require(
            bool(dominating),
            "neither Law componentwise dominates the exact tokenized production workload",
        )
        return sorted(dominating)[-1]

    def _load_fresh_model(self, panel_id: str, profile: str) -> tuple[nn.Module, Any, float]:
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as error:  # pragma: no cover - actual image dependency
            raise ProducerError("actual qualification requires transformers") from error
        torch.manual_seed(ENGINEERING_SEED)
        torch.cuda.manual_seed_all(ENGINEERING_SEED)
        started = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            self.snapshot_roots[panel_id],
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.bfloat16,
        )
        model.to(self.device)
        model.eval()
        runtime_config = getattr(model, "config", None)
        if profile == "baseline":
            enable = getattr(model, "gradient_checkpointing_enable", None)
            _require(callable(enable), "baseline requires supported gradient checkpointing")
            cast(Callable[[], Any], enable)()
            _require(
                getattr(model, "is_gradient_checkpointing", False) is True,
                "baseline did not enable the checkpointed training graph",
            )
            if runtime_config is not None and hasattr(runtime_config, "use_cache"):
                runtime_config.use_cache = False
        else:
            disable = getattr(model, "gradient_checkpointing_disable", None)
            if callable(disable):
                disable()
            _require(
                getattr(model, "is_gradient_checkpointing", False) is False,
                "tuned profile retained gradient checkpointing",
            )
            if runtime_config is not None and hasattr(runtime_config, "use_cache"):
                runtime_config.use_cache = False
        torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        _require(elapsed > 0.0, "fresh model load timing is not positive")
        return model, self.tokenizers[panel_id], elapsed

    def _clean_timing_only_updates(
        self,
        *,
        profile: str,
        panel_id: str,
    ) -> tuple[float, float, float, dict[str, Any]]:
        """Measure eight shared-helper updates on the global worst prompt shape."""

        config_path = self.repo / str(PROFILE_CONFIG_SPECS[profile][panel_id]["path"])
        resolved = load_config(config_path)
        train_config = resolved["train"]
        contract = PROFILE_CONTRACT[profile]
        _require(
            train_config["algorithm"] == "sft"
            and int(train_config["steps"]) == 1_000
            and int(train_config["batch_size"]) == int(contract["train_batch_size"])
            and int(train_config["gradient_accumulation_steps"])
            == int(contract["gradient_accumulation_steps"])
            and float(train_config["learning_rate"]) == 1e-5
            and float(train_config["weight_decay"]) == 0.0
            and float(train_config["warmup_ratio"]) == 0.01171875
            and float(train_config["grad_clip"]) == 1.0
            and int(train_config["parameter_finite_check_interval"]) == 0,
            "resolved H200 optimizer/timing contract drifted",
        )
        warmup_steps = round(int(train_config["steps"]) * float(train_config["warmup_ratio"]))
        maximum = self.training_maximums[panel_id]
        maximum_length = int(maximum.proof["maximum_prompt_token_length"])
        timing_prompts = (maximum.prompt,) * CORPUS_SIZE
        timing_actions = (maximum.action,) * CORPUS_SIZE
        timing_candidates = (maximum.candidate_choices,) * CORPUS_SIZE
        accumulation_steps = int(train_config["gradient_accumulation_steps"])
        batch_size = int(train_config["batch_size"])
        timed_call_shapes = [
            {
                "update": update,
                "micro_step": (update - 1) * accumulation_steps + accumulation_index,
                "example_count": batch_size,
                "max_seq_len": maximum_length,
                "padded_elements": batch_size * maximum_length,
            }
            for update in UPDATES
            for accumulation_index in range(accumulation_steps)
        ]
        _require(
            sum(int(item["example_count"]) for item in timed_call_shapes)
            == len(UPDATES) * int(contract["effective_batch_size"]),
            "standardized timing call shapes changed effective-batch geometry",
        )

        host_proof = maximum.proof["tokenizer_host_envelopes"]
        host_envelope_order = HOST_ENVELOPE_ORDER
        _require(
            host_proof["envelope_order"] == list(host_envelope_order)
            and host_proof["measured_seconds_aggregation"] == "sum_all_four_envelopes",
            "tokenizer host-envelope aggregation contract changed",
        )
        host_execution_prompts: dict[str, tuple[str, ...]] = {}
        host_contracts: dict[str, dict[str, Any]] = {}
        for envelope_name in host_envelope_order:
            source_prompts = maximum.host_envelope_prompts[envelope_name]
            envelope_proof = host_proof["envelopes"][envelope_name]
            rows = envelope_proof["rows"]
            _require(
                len(source_prompts) == len(rows) == envelope_proof["source_row_count"],
                "tokenizer host envelope source-row cardinality changed",
            )
            contextual = encode_action_continuations(
                self.tokenizers[panel_id],
                source_prompts,
                ACTION_LABELS,
                add_prompt_special_tokens=False,
            )
            for prompt, prompt_tokens, continuations, row in zip(
                source_prompts,
                contextual.prompt_tokens,
                contextual.action_tokens,
                rows,
                strict=True,
            ):
                continuation_a, continuation_b = continuations
                _require(
                    hashlib.sha256(prompt.encode("utf-8")).hexdigest() == row["prompt_sha256"]
                    and len(prompt.encode("utf-8")) == row["prompt_utf8_bytes"]
                    and len(prompt_tokens) == row["prompt_token_length"]
                    and len((prompt + ACTION_LABELS[0]).encode("utf-8")) == row["prompt_a_utf8_bytes"]
                    and len(prompt_tokens) + len(continuation_a) == row["prompt_a_token_length"]
                    and len((prompt + ACTION_LABELS[1]).encode("utf-8")) == row["prompt_b_utf8_bytes"]
                    and len(prompt_tokens) + len(continuation_b) == row["prompt_b_token_length"]
                    and len(continuation_a) == row["continuation_a_token_count"]
                    and semantic_digest(list(continuation_a)) == row["continuation_a_token_ids_sha256"]
                    and len(continuation_b) == row["continuation_b_token_count"]
                    and semantic_digest(list(continuation_b)) == row["continuation_b_token_ids_sha256"],
                    "worker tokenizer-host envelope differs from controller contextual proof",
                )
            execution_prompts = source_prompts * int(envelope_proof["repetition_count"])
            execution_digests = [
                hashlib.sha256(prompt.encode("utf-8")).hexdigest() for prompt in execution_prompts
            ]
            partitions = [
                execution_digests[start : start + batch_size]
                for start in range(0, len(execution_digests), batch_size)
            ]
            _require(
                len(execution_prompts) == envelope_proof["execution_row_count_per_update"] == 50
                and semantic_digest(execution_digests) == envelope_proof["ordered_execution_prompt_sha256"]
                and [len(partition) for partition in partitions]
                == envelope_proof[f"{profile}_call_partition_sizes"]
                and semantic_digest(partitions) == envelope_proof[f"{profile}_call_partitions_sha256"],
                "worker tokenizer-host envelope partition replay failed",
            )
            host_execution_prompts[envelope_name] = execution_prompts
            host_contracts[envelope_name] = {
                "metric_field": envelope_proof["metric_field"],
                "role": envelope_proof["role"],
                "source_row_count": envelope_proof["source_row_count"],
                "rows_sha256": envelope_proof["rows_sha256"],
                "selected_metric_sum": envelope_proof["selected_metric_sum"],
                "selected_metric_maximum": envelope_proof["selected_metric_maximum"],
                "repetition_count": envelope_proof["repetition_count"],
                "ordered_execution_prompt_sha256": envelope_proof["ordered_execution_prompt_sha256"],
                "timed_updates": len(UPDATES),
                "profile_batch_size": batch_size,
                "profile_call_partition_sizes": [len(partition) for partition in partitions],
                "profile_call_partitions_sha256": semantic_digest(partitions),
                "profile_scorer_call_count": len(UPDATES) * len(partitions),
                "total_examples_tokenized": len(UPDATES) * len(execution_prompts),
                "warmup_horizon_excluded": True,
                "worker_contextual_rows_replayed": True,
            }

        def execute_host_envelope(prompts: tuple[str, ...]) -> None:
            for _update in UPDATES:
                for start in range(0, len(prompts), batch_size):
                    encoded = encode_action_continuations(
                        self.tokenizers[panel_id],
                        prompts[start : start + batch_size],
                        ACTION_LABELS,
                        add_prompt_special_tokens=False,
                    )
                    _require(
                        len(encoded.prompt_tokens) == min(batch_size, len(prompts) - start),
                        "tokenizer host envelope returned the wrong batch cardinality",
                    )

        host_envelope_seconds_by_kind: dict[str, float] = {}
        for envelope_name in host_envelope_order:
            prompts = host_execution_prompts[envelope_name]
            execute_host_envelope(prompts)
            host_started = time.perf_counter()
            execute_host_envelope(prompts)
            host_envelope_seconds_by_kind[envelope_name] = max(
                time.perf_counter() - host_started,
                1e-9,
            )
        host_envelopes_seconds = sum(host_envelope_seconds_by_kind.values())

        def execute(model: nn.Module, tokenizer: Any, *, measure: bool) -> tuple[float, float]:
            scorer = TwoActionScorer(
                model,
                tokenizer,
                ACTION_LABELS,
                add_prompt_special_tokens=False,
            )
            scorer.train()
            native_parameters = [parameter for parameter in scorer.parameters() if parameter.requires_grad]
            optimizer = build_optimizer(
                scorer,
                learning_rate=float(train_config["learning_rate"]),
                weight_decay=float(train_config["weight_decay"]),
            )
            _require(
                [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
                == [id(parameter) for parameter in native_parameters],
                "clean timing optimizer changed native production parameter order",
            )
            event_started = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
            event_completed = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
            if measure:
                torch.cuda.synchronize(self.device)
                wall_started = time.perf_counter()
                event_started.record()
            result = train_steps(
                scorer,
                optimizer,
                timing_prompts,
                timing_actions,
                total_steps=8 if measure else 1,
                batch_size=int(train_config["batch_size"]),
                algorithm="sft",
                gradient_accumulation_steps=int(train_config["gradient_accumulation_steps"]),
                warmup_steps=warmup_steps,
                max_grad_norm=float(train_config["grad_clip"]),
                parameter_finite_check_interval=int(train_config["parameter_finite_check_interval"]),
                candidate_choices=timing_candidates,
                seed=ENGINEERING_SEED,
                hooks=None,
            )
            _require(
                result.state.global_step == (8 if measure else 1)
                and result.state.micro_step
                == (8 if measure else 1) * int(train_config["gradient_accumulation_steps"])
                and len(result.metrics) == (8 if measure else 1),
                "shared train_steps timing path did not complete the exact update horizon",
            )
            wall_seconds = 0.0
            cuda_seconds = 0.0
            if measure:
                event_completed.record()
                torch.cuda.synchronize(self.device)
                wall_seconds = max(time.perf_counter() - wall_started, 1e-9)
                cuda_seconds = max(event_started.elapsed_time(event_completed) / 1_000.0, 1e-9)
            del optimizer, scorer
            return wall_seconds, cuda_seconds

        # A fully representative warmup includes forward/backward/clip/AdamW, then
        # its entire model and optimizer are discarded.  The measured state is fresh.
        warmup_model, tokenizer, _ = self._load_fresh_model(panel_id, profile)
        execute(warmup_model, tokenizer, measure=False)
        del warmup_model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(self.device)

        timed_model, tokenizer, _ = self._load_fresh_model(panel_id, profile)
        wall_seconds, cuda_seconds = execute(timed_model, tokenizer, measure=True)
        del timed_model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(self.device)
        _require(wall_seconds > 0.0 and cuda_seconds > 0.0, "clean update timing is empty")
        return (
            wall_seconds,
            cuda_seconds,
            host_envelopes_seconds,
            {
                "shared_helper": "goalzendo.training.train_steps",
                "algorithm": "sft",
                "total_steps": 8,
                "batch_size": int(train_config["batch_size"]),
                "gradient_accumulation_steps": int(train_config["gradient_accumulation_steps"]),
                "warmup_steps": warmup_steps,
                "max_grad_norm": float(train_config["grad_clip"]),
                "parameter_finite_check_interval": int(train_config["parameter_finite_check_interval"]),
                "seed": ENGINEERING_SEED,
                "candidate_choice_geometry_included": True,
                "hooks": None,
                "fresh_state_final_global_step": 8,
                "fresh_state_final_micro_step": 8 * int(train_config["gradient_accumulation_steps"]),
                "standardized_worst_shape_training": {
                    "global_maximum_proof": copy.deepcopy(dict(maximum.proof)),
                    "timed_corpus_size": CORPUS_SIZE,
                    "all_timed_prompts_equal_global_maximum": True,
                    "timed_call_shapes": timed_call_shapes,
                    "timed_call_shapes_sha256": semantic_digest(timed_call_shapes),
                    "timed_padded_token_elements": sum(
                        int(item["padded_elements"]) for item in timed_call_shapes
                    ),
                    "production_updates_per_run": 1_000,
                    "runs_per_panel_per_worker": 20,
                    "runs_per_worker": 40,
                    "conservative_scaling_ratio": 1.0,
                    "standardized_workload_dominates_every_registered_training_call": True,
                    "tokenizer_host_envelope_contracts": {
                        "contextual_continuation_mode": host_proof["contextual_continuation_mode"],
                        "envelope_order": list(host_envelope_order),
                        "measured_seconds_aggregation": "sum_all_four_envelopes",
                        "measured_seconds_by_envelope": host_envelope_seconds_by_kind,
                        "measured_seconds_sum": host_envelopes_seconds,
                        "envelopes": host_contracts,
                    },
                },
            },
        )

    def _save_vector_bundle(
        self,
        *,
        path: Path,
        parameters: Sequence[tuple[str, nn.Parameter]],
        tensors: Mapping[str, Tensor | None],
        lifecycle: str,
    ) -> None:
        try:
            from safetensors.torch import save_file  # type: ignore[import-not-found]
        except ImportError as error:  # pragma: no cover - actual image dependency
            raise ProducerError("actual qualification requires safetensors") from error
        _require(not path.exists(), "transient safetensors path already exists")
        path.parent.mkdir(parents=True, exist_ok=True)
        cpu_tensors: dict[str, Tensor] = {}
        for name, parameter in parameters:
            value = tensors.get(name)
            source = _zero_like(parameter) if value is None else value.detach()
            cpu_tensors[name] = source.contiguous().cpu()
        save_file(cpu_tensors, str(path), metadata={"qualification_lifecycle": lifecycle})
        self.inventory.register(path, lifecycle=lifecycle)

    def _capture_vector(
        self,
        *,
        kind: str,
        update: int,
        manifest: Mapping[str, Any],
        parameters: Sequence[tuple[str, nn.Parameter]],
        tensors: Mapping[str, Tensor | None],
        persist_vectors_under: Path | None,
    ) -> tuple[dict[str, Any], Path | None]:
        descriptor = _vector_descriptor(
            manifest=manifest,
            parameters=parameters,
            tensors=tensors,
        )
        path: Path | None = None
        if persist_vectors_under is not None:
            path = persist_vectors_under / f"update-{update:02d}-{kind}.safetensors"
            self._save_vector_bundle(
                path=path,
                parameters=parameters,
                tensors=tensors,
                lifecycle=f"paired_comparison:update={update}:vector={kind}",
            )
        return descriptor, path

    def _evaluate(
        self,
        scorer: TwoActionScorer,
        corpus: QualificationCorpus,
        *,
        batch_size: int,
    ) -> list[dict[str, Any]]:
        was_training = scorer.training
        scorer.eval()
        rows = [corpus.evaluation_rows[view] for view in TRAINING_VIEWS]
        all_scores: list[Tensor] = []
        with torch.no_grad():
            for start in range(0, len(rows), batch_size):
                prompts = [row.prompt for row in rows[start : start + batch_size]]
                scores = scorer(prompts)
                _require(
                    isinstance(scores, Tensor) and scores.shape == (len(prompts), 2),
                    "evaluation scorer returned an invalid shape",
                )
                all_scores.append(scores.detach().float().cpu())
        matrix = torch.cat(all_scores, dim=0)
        normalized = matrix.log_softmax(dim=-1)
        probabilities = normalized.exp()
        _require(
            bool(torch.isfinite(normalized).all() and torch.isfinite(probabilities).all()),
            "evaluation outputs are non-finite",
        )
        outputs: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            action_index = int(torch.argmax(probabilities[index]).item())
            outputs.append(
                {
                    "view": row.view,
                    "sample_id": row.sample_id,
                    "token_length": row.token_length,
                    "worst_case_token_sample": row.sample_id == corpus.worst.sample_id,
                    "normalized_log_scores": [float(value) for value in normalized[index].tolist()],
                    "probabilities": [float(value) for value in probabilities[index].tolist()],
                    "action_index": action_index,
                    "action_label": ACTION_LABELS[action_index],
                }
            )
        scorer.train(was_training)
        return outputs

    def _score_group_banks(
        self,
        scorer: TwoActionScorer,
        banks: Sequence[Sequence[Sequence[str]]],
        *,
        example_batch_size: int,
    ) -> dict[str, Any]:
        """Mirror evaluate_batches example chunking and prompt-view flattening."""

        example_count = sum(len(bank) for bank in banks)
        prompt_count = sum(len(group) for bank in banks for group in bank)
        _require(prompt_count >= 128, "evaluation benchmark must contain at least 128 prompts")
        was_training = scorer.training
        scorer.eval()
        normalized_rows: list[list[float]] = []
        scorer_call_count = 0
        with torch.no_grad():
            for bank in banks:
                for start in range(0, len(bank), example_batch_size):
                    example_chunk = bank[start : start + example_batch_size]
                    prompts = [prompt for group in example_chunk for prompt in group]
                    scores = scorer(prompts)
                    scorer_call_count += 1
                    _require(
                        isinstance(scores, Tensor) and scores.shape == (len(prompts), 2),
                        "production-shaped evaluator returned an invalid shape",
                    )
                    normalized = scores.detach().float().log_softmax(dim=-1).cpu()
                    _require(
                        bool(torch.isfinite(normalized).all()),
                        "production-shaped evaluator produced non-finite scores",
                    )
                    normalized_rows.extend([float(value) for value in row] for row in normalized.tolist())
        scorer.train(was_training)
        return {
            "example_count": example_count,
            "prompt_count": prompt_count,
            "batch_size": example_batch_size,
            "batch_count": scorer_call_count,
            "scorer_call_count": scorer_call_count,
            "normalized_outputs_sha256": semantic_digest(normalized_rows),
            "normalized_outputs": normalized_rows,
        }

    def _registered_parity_chunk(
        self,
        scorer: TwoActionScorer,
        benchmark: EvaluationBenchmark,
        *,
        example_batch_size: int,
        measurement_state: str,
    ) -> dict[str, Any]:
        """Score the exact contiguous 128-example/six-view registered partition."""

        _require(
            measurement_state == "initial"
            or measurement_state in {f"after_update_{update}" for update in (1, 2, 4, 8)},
            "registered parity measurement state is not frozen",
        )
        scored = self._score_group_banks(
            scorer,
            (benchmark.registered_groups,),
            example_batch_size=example_batch_size,
        )
        normalized_values = scored.pop("normalized_outputs")
        _require(
            len(normalized_values) == len(benchmark.registered_row_ids),
            "registered parity row IDs/output cardinality changed",
        )
        rows: list[dict[str, Any]] = []
        for row_id, scores in zip(
            benchmark.registered_row_ids,
            normalized_values,
            strict=True,
        ):
            probabilities = [math.exp(float(value)) for value in scores]
            action_index = int(probabilities[1] > probabilities[0])
            rows.append(
                {
                    "row_id": row_id,
                    "normalized_log_scores": [float(value) for value in scores],
                    "probabilities": probabilities,
                    "action_index": action_index,
                    "action_label": ACTION_LABELS[action_index],
                }
            )
        scored.update(
            {
                "measurement_state": measurement_state,
                "after_optimizer_step": measurement_state != "initial",
                "production_example_partition_count": 1,
                "rows": rows,
                "rows_sha256": semantic_digest(rows),
                "prompt_sha256": benchmark.registered_prompt_sha256,
                "contiguous_example_chunk": True,
                "prompt_views_per_example": len(TRAINING_VIEWS),
                "production_bank_name": benchmark.registered_bank_name,
                "production_bank_start_index": benchmark.registered_start_index,
                "example_ids": list(benchmark.registered_example_ids),
                "example_ids_sha256": benchmark.registered_example_ids_sha256,
                "token_shapes": [list(shape) for shape in benchmark.registered_token_shapes],
                "token_shape_sha256": benchmark.registered_token_shape_sha256,
            }
        )
        return scored

    def _evaluation_callback_io(
        self,
        *,
        capture_root: Path,
        ordinal: int,
        kind: str,
        scored: Mapping[str, Any],
    ) -> tuple[dict[str, Any], float]:
        started = time.perf_counter()
        root = capture_root / "evaluation-callbacks" / f"boundary-{ordinal:02d}-{kind}"
        step = 1000 if kind == "final" else ordinal
        normalized_outputs = list(scored["normalized_outputs"])
        _require(
            len(normalized_outputs) == int(scored["prompt_count"]),
            "callback output count differs from the scored production-shaped bank",
        )
        prediction_rows = [
            {
                "record_id": f"prediction:{step}:{index}",
                "kind": "prediction",
                "step": step,
                "split": "final_factorial_causal" if kind == "final" else "diagnostic_factorial_causal",
                "sample_id": f"engineering-evaluation-{index:06d}",
                "prompt_view": TRAINING_VIEWS[index % len(TRAINING_VIEWS)],
                "normalized_log_scores": values,
                "probability_a": math.exp(float(values[0])),
                "probability_b": math.exp(float(values[1])),
                "predicted_action": ACTION_LABELS[int(float(values[1]) > float(values[0]))],
                "engineering_qualification_only": True,
            }
            for index, values in enumerate(normalized_outputs)
        ]
        # This one-row-per-prompt metrics envelope is conservative relative to
        # production's aggregated behavior/intervention/wide-checkpoint rows.
        metric_rows = [
            {
                "record_id": f"metric:{step}:{index}",
                "kind": "behavior",
                "step": step,
                "split": row["split"],
                "prompt_view": row["prompt_view"],
                "metric": "choice_probability_b",
                "value": row["probability_b"],
                "engineering_qualification_only": True,
            }
            for index, row in enumerate(prediction_rows)
        ]
        metrics_path = root / "metrics.jsonl"
        predictions_path = root / "predictions.jsonl"
        progress_path = root / "status.json"
        _exclusive_jsonl(metrics_path, metric_rows, mode=0o600)
        _exclusive_jsonl(predictions_path, prediction_rows, mode=0o600)
        _exclusive_json(
            progress_path,
            {
                "state": "running",
                "last_step": step,
                "phase": "evaluated",
                "evaluation_bank": ("final_factorial" if kind == "final" else "diagnostic_factorial"),
                "engineering_qualification_only": True,
            },
            mode=0o600,
        )
        paths = (metrics_path, predictions_path, progress_path)
        for path in paths:
            self.inventory.register(
                path,
                lifecycle="production_evaluation_callback_append_fsync_progress",
            )
        receipt = {
            "sha256": semantic_digest([{"name": path.name, "sha256": sha256_file(path)} for path in paths]),
            "bytes": sum(path.stat().st_size for path in paths),
            "artifact_count": len(paths),
            "metrics_row_count": len(metric_rows),
            "prediction_row_count": len(prediction_rows),
            "progress_record_count": 1,
            "schema": "production_metrics_predictions_progress_envelope_v1",
        }
        for path in paths:
            self.inventory.delete(path)
        return receipt, max(time.perf_counter() - started, 1e-9)

    def _checkpoint_io(
        self,
        *,
        capture_root: Path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        benchmark_identity: str,
    ) -> tuple[dict[str, Any], float]:
        started = time.perf_counter()
        directory = capture_root / "checkpoint"
        directory.mkdir(parents=True, exist_ok=True)
        stage = directory / ".resume-step-00001000.pt.stage"
        target = directory / "resume-step-00001000.pt"
        _require(not stage.exists() and not target.exists(), "checkpoint benchmark path already exists")
        descriptor = os.open(stage, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(
                {
                    "schema_version": 2,
                    "artifact_kind": "resumable_checkpoint",
                    "step": 1000,
                    "binding": {"qualification_benchmark_identity": benchmark_identity},
                    "model_state_kind": "full",
                    "model_state": _cpu_tree(model.state_dict()),
                    "optimizer_state": _cpu_tree(optimizer.state_dict()),
                    "train_state": {"global_step": 1000},
                    "hook_state": {"emitted": [["checkpoint", 1000]]},
                },
                handle,
            )
            handle.flush()
            os.fsync(handle.fileno())
        self.inventory.register(stage, lifecycle="production_checkpoint_atomic_stage")
        self.inventory.rename(stage, target, lifecycle="production_checkpoint_authoritative")
        checkpoint_sha = sha256_file(target)
        pointer = directory / "latest.json"
        _exclusive_json(
            pointer,
            {
                "schema_version": 2,
                "artifact_kind": "resumable_checkpoint_pointer",
                "step": 1000,
                "file": target.name,
                "sha256": checkpoint_sha,
                "binding": {"qualification_benchmark_identity": benchmark_identity},
            },
            mode=0o600,
        )
        self.inventory.register(pointer, lifecycle="production_checkpoint_pointer_fsync")
        total_bytes = target.stat().st_size + pointer.stat().st_size
        receipt_sha = semantic_digest(
            {
                "checkpoint_sha256": checkpoint_sha,
                "pointer_sha256": sha256_file(pointer),
            }
        )
        self.inventory.delete(pointer)
        self.inventory.delete(target)
        return {
            "executed": True,
            "artifact_count": 2,
            "bytes": total_bytes,
            "sha256": receipt_sha,
        }, max(time.perf_counter() - started, 1e-9)

    def _final_seal_io(
        self,
        *,
        capture_root: Path,
        benchmark_identity: str,
        profile: str,
        panel_id: str,
        steps: Sequence[Mapping[str, Any]],
        boundaries: Sequence[Mapping[str, Any]],
    ) -> tuple[dict[str, Any], float]:
        total_prediction_rows = sum(int(item["prompt_count"]) for item in boundaries)
        _require(
            len(boundaries) == EVALUATION_BOUNDARY_COUNT and total_prediction_rows > 0,
            "final-seal benchmark lacks thirteen production boundaries",
        )
        prediction_rows = [
            {
                "record_id": f"sealed-prediction:{index}",
                "kind": "prediction",
                "step": 1000 if index >= total_prediction_rows - int(boundaries[-1]["prompt_count"]) else 999,
                "split": "production_evaluation_envelope",
                "sample_id": f"engineering-sealed-{index:07d}",
                "prompt_view": TRAINING_VIEWS[index % len(TRAINING_VIEWS)],
                "normalized_log_scores": [-0.5108256237659907, -0.916290731874155],
                "probability_a": 0.6,
                "probability_b": 0.4,
                "predicted_action": "A",
                "engineering_qualification_only": True,
            }
            for index in range(total_prediction_rows)
        ]
        metric_rows = [
            {
                "record_id": f"sealed-metric:{index}",
                "kind": "behavior",
                "step": row["step"],
                "split": row["split"],
                "prompt_view": row["prompt_view"],
                "metric": "choice_probability_b",
                "value": row["probability_b"],
                "engineering_qualification_only": True,
            }
            for index, row in enumerate(prediction_rows)
        ]
        config = copy.deepcopy(load_config(self.repo / str(PROFILE_CONFIG_SPECS[profile][panel_id]["path"])))
        store = RunStore(
            capture_root / "final-seal-store",
            config,
            ENGINEERING_SEED,
            self.repo,
        )
        _require(store.initialize(resume=False) == "new", "isolated final-seal RunStore was not fresh")
        store.record_dataset_metadata(
            {
                "qualification_benchmark_identity": benchmark_identity,
                "data_class": "synthetic_engineering_only",
            }
        )
        store.record_model_metadata(
            {
                "qualification_benchmark_identity": benchmark_identity,
                "panel_id": panel_id,
                "model_name": PANELS[panel_id]["model_name"],
                "model_revision": PANELS[panel_id]["model_revision"],
            }
        )
        store.record_tokenizer_metadata(
            {
                "qualification_benchmark_identity": benchmark_identity,
                "offline_local_snapshot": True,
            }
        )
        store.append_metrics(metric_rows)
        store.append_predictions(prediction_rows)
        started = time.perf_counter()
        store.finalize(
            {
                "benchmark_identity": benchmark_identity,
                "final_step": 1000,
                "completion": "engineering_qualification_only",
                "metrics_record_count": len(metric_rows),
                "prediction_record_count": len(prediction_rows),
                "steps_sha256": semantic_digest(list(steps)),
                "boundary_outputs": [item["normalized_outputs_sha256"] for item in boundaries],
                "outcomes_seen": False,
                "g01_launch_authorized": False,
            }
        )
        completion = verify_completion_attestation(store.path)
        sealed_names = ("metrics.jsonl", "predictions.jsonl", "summary.json")
        sealed_hashes = {name: sha256_file(store.path / name) for name in sealed_names}
        for name in sealed_names:
            os.chmod(store.path / name, 0o000)
        sealed_modes = {name: stat.S_IMODE((store.path / name).stat().st_mode) for name in sealed_names}
        _require(
            all(mode == 0 for mode in sealed_modes.values()),
            "production outcome-file seal did not chmod every outcome file to mode 000",
        )
        elapsed = max(time.perf_counter() - started, 1e-9)
        # The only post-seal mutation is the authenticated transient teardown:
        # restore owner access, verify the pre-chmod digest, inventory, and delete.
        for name in sealed_names:
            target = store.path / name
            os.chmod(target, 0o600)
            _require(
                sha256_file(target) == sealed_hashes[name],
                "sealed outcome file changed before authenticated transient cleanup",
            )
        files = sorted(path for path in store.path.rglob("*") if path.is_file())
        complete_marker = store.path / "COMPLETE"
        _require(
            complete_marker.read_text(encoding="utf-8") == "complete\n"
            and complete_marker.stat().st_mtime_ns
            >= max(
                (store.path / name).stat().st_mtime_ns
                for name in ("summary.json", "status.json", "completion.json")
            ),
            "production COMPLETE marker was not the final seal write",
        )
        for target in files:
            self.inventory.register(
                target,
                lifecycle="production_runstore_finalize_completion_attestation",
            )
        total_bytes = sum(path.stat().st_size for path in files)
        digest = semantic_digest(
            [
                {
                    "path": str(path.relative_to(store.path)),
                    "sha256": sha256_file(path),
                }
                for path in files
            ]
        )
        for path in files:
            self.inventory.delete(path)
        return {
            "executed": True,
            "artifact_count": len(files),
            "bytes": total_bytes,
            "sha256": digest,
            "file_names": [
                "metrics.jsonl",
                "predictions.jsonl",
                "summary.json",
                "status.json",
                "completion.json",
                "COMPLETE",
            ],
            "metrics_row_count": len(metric_rows),
            "prediction_row_count": len(prediction_rows),
            "summary_schema": "production_run_summary_envelope_v1",
            "attested_file_count": len(completion["files"]),
            "completion_attestation_verified": True,
            "complete_marker_final_write": True,
            "run_store_finalize_used": True,
            "outcome_file_seal_hashes": sealed_hashes,
            "outcome_file_seal_modes": sealed_modes,
            "outcome_file_seal_file_count": len(sealed_names),
            "outcome_file_seal_scan_and_chmod_completed": True,
            "seal_restore_only_for_authenticated_transient_cleanup": True,
        }, elapsed

    def run_capacity_probe_cell(self, *, panel_id: str, law_family: str) -> dict[str, Any]:
        """Exercise one fresh tuned train+eval cell; caller isolates process lifetime."""

        _require(panel_id in PANELS and law_family in LAW_FAMILIES, "unknown capacity probe cell")
        corpus = self.corpora[(panel_id, law_family)]
        benchmark = self.evaluation_benchmarks[(panel_id, law_family)]
        torch.cuda.reset_peak_memory_stats(self.device)
        model, tokenizer, _model_load_seconds = self._load_fresh_model(panel_id, "tuned")
        scorer = TwoActionScorer(model, tokenizer, ACTION_LABELS, add_prompt_special_tokens=False)
        scorer.train()
        optimizer = build_optimizer(model, learning_rate=1e-5, weight_decay=0.0)
        native_parameters = [parameter for parameter in scorer.parameters() if parameter.requires_grad]
        ordered_indices, insertion = _qualification_update_order(corpus, 1)
        _require(
            len(ordered_indices) == 50
            and insertion["worst_sample_id"] in [corpus.sample_ids[index] for index in ordered_indices],
            "capacity train batch omitted the corpus-max sample",
        )
        prompts = [corpus.prompts[index] for index in ordered_indices]
        targets = torch.tensor(
            [corpus.actions[index] for index in ordered_indices],
            dtype=torch.long,
            device=self.device,
        )
        optimizer.zero_grad(set_to_none=True)
        with _step_rng(ENGINEERING_SEED, 0, self.device):
            scores = scorer(prompts)
        _require(isinstance(scores, Tensor), "capacity scorer did not return a tensor")
        loss = sft_action_loss(scores, targets).loss
        if not bool(torch.isfinite(loss).detach().cpu()):
            raise NonFiniteCapacityError("capacity loss is non-finite")
        loss.backward()  # type: ignore[no-untyped-call]
        if not all(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all().detach().cpu())
            for parameter in native_parameters
        ):
            raise NonFiniteCapacityError("capacity gradient is non-finite or absent")
        try:
            torch.nn.utils.clip_grad_norm_(
                native_parameters,
                max_norm=1.0,
                error_if_nonfinite=True,
            )
        except RuntimeError as error:
            if "non-finite" in str(error).lower() or "nonfinite" in str(error).lower():
                raise NonFiniteCapacityError("capacity clipped gradient is non-finite") from error
            raise
        optimizer.step()
        _require(
            all(
                isinstance(optimizer.state.get(parameter, {}).get("exp_avg"), Tensor)
                and isinstance(optimizer.state.get(parameter, {}).get("exp_avg_sq"), Tensor)
                for parameter in native_parameters
            ),
            "capacity optimizer moments are not resident",
        )
        scored = self._score_group_banks(
            scorer,
            (benchmark.registered_groups,),
            example_batch_size=128,
        )
        _require(
            scored["example_count"] == 128
            and scored["prompt_count"] == 128 * len(TRAINING_VIEWS)
            and scored["scorer_call_count"] == 1,
            "capacity evaluation did not execute one contiguous 128-example six-view call",
        )
        torch.cuda.synchronize(self.device)
        peak_allocated = int(torch.cuda.max_memory_allocated(self.device))
        peak_reserved = int(torch.cuda.max_memory_reserved(self.device))
        bindings = self.capacity_probe_bindings(panel_id=panel_id, law_family=law_family)
        return {
            "actual_model_execution": True,
            "fresh_model_and_optimizer": True,
            "train_batch_exercised": True,
            "gradient_clip_exercised": True,
            "adamw_step_exercised": True,
            "optimizer_moments_resident_during_evaluation": True,
            "contiguous_evaluation_example_count": 128,
            "flattened_evaluation_prompt_count": scored["prompt_count"],
            "cuda_synchronized_before_memory_read": True,
            "finite_execution": True,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            **bindings,
        }

    def capacity_probe_bindings(self, *, panel_id: str, law_family: str) -> dict[str, str]:
        corpus = self.corpora[(panel_id, law_family)]
        benchmark = self.evaluation_benchmarks[(panel_id, law_family)]
        ordered_indices, _insertion = _qualification_update_order(corpus, 1)
        tuned_config_path = self.repo / str(PROFILE_CONFIG_SPECS["tuned"][panel_id]["path"])
        tuned_config = copy.deepcopy(load_config(tuned_config_path))
        tuned_config["data"]["rule_family"] = law_family
        expected_numerical = {
            "deterministic_algorithms": True,
            "deterministic_warn_only": False,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "float32_matmul_precision": "highest",
            "cublas_workspace_config": ":4096:8",
        }
        _require(
            self.numerical_execution_receipt == expected_numerical,
            "production numerical execution receipt changed",
        )
        return {
            "resolved_config_sha256": semantic_digest(tuned_config),
            "numerical_execution_sha256": semantic_digest(expected_numerical),
            "workload_sha256": semantic_digest(
                {
                    "ordered_example_ids": [corpus.sample_ids[index] for index in ordered_indices],
                    "registered_prompt_sha256": benchmark.registered_prompt_sha256,
                    "registered_example_ids_sha256": benchmark.registered_example_ids_sha256,
                    "registered_token_shape_sha256": benchmark.registered_token_shape_sha256,
                }
            ),
        }

    def run(
        self,
        *,
        profile: str,
        replicate: str,
        panel_id: str,
        law_family: str,
        process_uuid: str,
        persist_vectors_under: Path | None,
        full_timing: bool,
    ) -> CapturedProcess:
        _require(
            profile in PROFILES and replicate in REPLICATES[profile], "unknown logical profile/replicate"
        )
        _require(panel_id in PANELS and law_family in LAW_FAMILIES, "unknown logical panel/Law")
        _canonical_uuid(process_uuid, "logical process UUID")
        _require(
            (panel_id, law_family) in self.corpora,
            "qualification corpora were not prepared before model load",
        )
        _require(
            not full_timing or (replicate == "primary" and law_family == self.timing_law(profile, panel_id)),
            "full production timing is not the primary worst-Law representative",
        )
        corpus = self.corpora[(panel_id, law_family)]
        benchmark = self.evaluation_benchmarks[(panel_id, law_family)]
        clean_update_wall_seconds = 0.0
        clean_update_cuda_seconds = 0.0
        tokenizer_host_envelopes_seconds = 0.0
        clean_update_contract: dict[str, Any] | None = None
        if full_timing:
            (
                clean_update_wall_seconds,
                clean_update_cuda_seconds,
                tokenizer_host_envelopes_seconds,
                clean_update_contract,
            ) = self._clean_timing_only_updates(profile=profile, panel_id=panel_id)
        self.capture_index += 1
        capture_root = self.inventory.root / f"capture-{self.capture_index:04d}"
        torch.cuda.reset_peak_memory_stats(self.device)
        model, tokenizer, model_load_seconds = self._load_fresh_model(panel_id, profile)
        scorer = TwoActionScorer(
            model,
            tokenizer,
            ACTION_LABELS,
            add_prompt_special_tokens=False,
        )
        scorer.train()
        parameters = _named_trainable_parameters(model)
        manifest = _trainable_parameter_manifest(panel_id, parameters)
        native_parameters = [parameter for parameter in scorer.parameters() if parameter.requires_grad]
        _require(
            len(native_parameters) == len(parameters)
            and {id(parameter) for parameter in native_parameters}
            == {id(parameter) for _name, parameter in parameters},
            "native scorer.parameters traversal differs from the evidence manifest set",
        )
        optimizer = build_optimizer(model, learning_rate=1e-5, weight_decay=0.0)
        _require(
            [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
            == [id(parameter) for parameter in native_parameters],
            "shared optimizer did not preserve native scorer.parameters traversal",
        )
        initial_model = _model_state_sha256(model)
        initial_optimizer = _optimizer_state_sha256(optimizer, parameters)
        stochastic = _stochastic_module_audit(model)
        contract = PROFILE_CONTRACT[profile]
        batch_size = int(contract["train_batch_size"])
        accumulation_steps = int(contract["gradient_accumulation_steps"])
        evaluation_batch_size = int(contract["evaluation_batch_size"])
        gradient_checkpointing = bool(contract["gradient_checkpointing"])
        _require(
            gradient_checkpointing == (profile == "baseline"),
            "profile gradient-checkpointing contract changed",
        )
        warmup_indices, _warmup_insertion = _qualification_update_order(corpus, 1)
        warmup_batch = warmup_indices[:batch_size]
        warmup_prompts = [corpus.prompts[index] for index in warmup_batch]
        warmup_targets = torch.tensor(
            [corpus.actions[index] for index in warmup_batch],
            dtype=torch.long,
            device=self.device,
        )
        optimizer.zero_grad(set_to_none=True)
        with _step_rng(ENGINEERING_SEED, 0, self.device):
            warmup_scores = scorer(warmup_prompts)
        _require(isinstance(warmup_scores, Tensor), "warmup scorer did not return a tensor")
        warmup_loss = sft_action_loss(warmup_scores, warmup_targets).loss
        _require(bool(torch.isfinite(warmup_loss).detach().cpu()), "warmup loss is non-finite")
        warmup_loss.backward()  # type: ignore[no-untyped-call]
        torch.cuda.synchronize(self.device)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(self.device)

        evaluation_seconds = 0.0
        evaluation_callback_io_seconds = 0.0
        evaluation_calls = 0
        boundaries: list[dict[str, Any]] = []

        registered_parity_chunk = self._registered_parity_chunk(
            scorer,
            benchmark,
            example_batch_size=evaluation_batch_size,
            measurement_state="initial",
        )

        def execute_boundary(kind: str) -> None:
            nonlocal evaluation_seconds, evaluation_callback_io_seconds, evaluation_calls
            banks = benchmark.final_banks if kind == "final" else benchmark.diagnostic_banks
            prompt_sha = (
                benchmark.final_prompt_sha256 if kind == "final" else benchmark.diagnostic_prompt_sha256
            )
            evaluation_started = time.perf_counter()
            scored = self._score_group_banks(
                scorer,
                banks,
                example_batch_size=evaluation_batch_size,
            )
            torch.cuda.synchronize(self.device)
            evaluation_seconds += max(time.perf_counter() - evaluation_started, 1e-9)
            callback, callback_seconds = self._evaluation_callback_io(
                capture_root=capture_root,
                ordinal=evaluation_calls,
                kind=kind,
                scored=scored,
            )
            evaluation_callback_io_seconds += callback_seconds
            scored.pop("normalized_outputs")
            boundaries.append(
                {
                    "ordinal": evaluation_calls,
                    "kind": kind,
                    **scored,
                    "prompt_sha256": prompt_sha,
                    "callback_artifact_sha256": callback["sha256"],
                    "callback_artifact_bytes": callback["bytes"],
                    "callback_artifact_count": callback["artifact_count"],
                    "callback_metrics_row_count": callback["metrics_row_count"],
                    "callback_prediction_row_count": callback["prediction_row_count"],
                    "callback_progress_record_count": callback["progress_record_count"],
                    "callback_schema": callback["schema"],
                }
            )
            evaluation_calls += 1
            _require(scorer.training, "evaluation boundary did not restore scorer.train()")

        if full_timing:
            benchmark_identity = semantic_digest(
                {
                    "profile": profile,
                    "replicate": replicate,
                    "panel_id": panel_id,
                    "law_family": law_family,
                }
            )
            execute_boundary("diagnostic")
        capture_path_update_cuda_seconds = 0.0
        vector_files: dict[tuple[int, str], Path] = {}
        steps: list[dict[str, Any]] = []
        warmup_steps = round(1_000 * 0.01171875)
        for update in UPDATES:
            _require(scorer.training, "training update began with scorer outside train mode")
            learning_rate = 1e-5 * constant_with_warmup_multiplier(update, warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            ordered_indices, insertion = _qualification_update_order(corpus, update)
            ordered_ids = [corpus.sample_ids[index] for index in ordered_indices]
            production_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
            backward_start = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
            backward_end = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
            backward_start.record()
            micro_steps: list[int] = []
            for accumulation_index in range(accumulation_steps):
                start = accumulation_index * batch_size
                observed = ordered_indices[start : start + batch_size]
                _require(len(observed) == batch_size, "qualification micro-batch is incomplete")
                prompts = [corpus.prompts[index] for index in observed]
                targets = torch.tensor(
                    [corpus.actions[index] for index in observed],
                    dtype=torch.long,
                    device=self.device,
                )
                micro_step = (update - 1) * accumulation_steps + accumulation_index
                micro_steps.append(micro_step)
                with _step_rng(ENGINEERING_SEED, micro_step, self.device):
                    scores = scorer(prompts)
                _require(isinstance(scores, Tensor), "training scorer did not return a tensor")
                loss = sft_action_loss(scores, targets).loss / accumulation_steps
                _require(bool(torch.isfinite(loss).detach().cpu()), "training loss is non-finite")
                loss.backward()  # type: ignore[no-untyped-call]
            backward_end.record()
            production_events.append((backward_start, backward_end))
            _require(
                len(ordered_indices) == 50 and len(set(ordered_indices)) == 50,
                "optimizer update does not contain 50 exact ordered examples",
            )
            raw_gradients = {name: parameter.grad for name, parameter in parameters}
            raw_descriptor, raw_path = self._capture_vector(
                kind="raw_preclip_gradient",
                update=update,
                manifest=manifest,
                parameters=parameters,
                tensors=raw_gradients,
                persist_vectors_under=persist_vectors_under,
            )
            clip_start = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
            clip_end = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
            clip_start.record()
            torch.nn.utils.clip_grad_norm_(
                native_parameters,
                max_norm=1.0,
                error_if_nonfinite=True,
            )
            clip_end.record()
            production_events.append((clip_start, clip_end))
            post_gradients = {name: parameter.grad for name, parameter in parameters}
            post_descriptor, post_path = self._capture_vector(
                kind="postclip_gradient",
                update=update,
                manifest=manifest,
                parameters=parameters,
                tensors=post_gradients,
                persist_vectors_under=persist_vectors_under,
            )
            before = {name: parameter.detach().clone() for name, parameter in parameters}
            optimizer_start = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
            optimizer_end = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
            optimizer_start.record()
            optimizer.step()
            optimizer_end.record()
            torch.cuda.synchronize(self.device)
            production_events.append((optimizer_start, optimizer_end))
            capture_path_update_cuda_seconds += max(
                sum(start.elapsed_time(end) for start, end in production_events) / 1_000.0,
                1e-9,
            )
            deltas = {name: parameter.detach() - before[name] for name, parameter in parameters}
            delta_descriptor, delta_path = self._capture_vector(
                kind="parameter_delta",
                update=update,
                manifest=manifest,
                parameters=parameters,
                tensors=deltas,
                persist_vectors_under=persist_vectors_under,
            )
            first_moments: dict[str, Tensor | None] = {}
            second_moments: dict[str, Tensor | None] = {}
            for name, parameter in parameters:
                state = optimizer.state.get(parameter, {})
                first = state.get("exp_avg")
                second = state.get("exp_avg_sq")
                _require(
                    isinstance(first, Tensor) and isinstance(second, Tensor),
                    "AdamW moments are absent after optimizer step",
                )
                first_moments[name] = first
                second_moments[name] = second
            first_descriptor, first_path = self._capture_vector(
                kind="optimizer_first_moment",
                update=update,
                manifest=manifest,
                parameters=parameters,
                tensors=first_moments,
                persist_vectors_under=persist_vectors_under,
            )
            second_descriptor, second_path = self._capture_vector(
                kind="optimizer_second_moment",
                update=update,
                manifest=manifest,
                parameters=parameters,
                tensors=second_moments,
                persist_vectors_under=persist_vectors_under,
            )
            for kind, path in zip(
                VECTOR_KINDS,
                (raw_path, post_path, delta_path, first_path, second_path),
                strict=True,
            ):
                if path is not None:
                    vector_files[(update, kind)] = path
            model_state = _model_state_sha256(model)
            optimizer_state = _optimizer_state_sha256(optimizer, parameters)

            outputs = self._evaluate(scorer, corpus, batch_size=evaluation_batch_size)
            _require(scorer.training, "six-row registered evaluation did not restore train mode")
            outputs_digest = semantic_digest(outputs)
            step_parity_chunk = (
                self._registered_parity_chunk(
                    scorer,
                    benchmark,
                    example_batch_size=evaluation_batch_size,
                    measurement_state=f"after_update_{update}",
                )
                if update in (1, 2, 4, 8)
                else None
            )
            _require(scorer.training, "registered parity chunk did not restore train mode")
            steps.append(
                {
                    "update": update,
                    "ordered_example_count": 50,
                    "ordered_example_ids": ordered_ids,
                    "ordered_example_ids_sha256": semantic_digest(ordered_ids),
                    "worst_sample_insertion": insertion,
                    "rng_receipt": {
                        "helper": "goalzendo.training._step_rng",
                        "engineering_seed": ENGINEERING_SEED,
                        "micro_steps": micro_steps,
                        "scorer_calls_guarded": True,
                    },
                    "outputs": outputs,
                    "outputs_sha256": outputs_digest,
                    "registered_production_shaped_parity_chunk": step_parity_chunk,
                    "vectors": {
                        "raw_preclip_gradient": raw_descriptor,
                        "postclip_gradient": post_descriptor,
                        "parameter_delta": delta_descriptor,
                        "optimizer_first_moment": first_descriptor,
                        "optimizer_second_moment": second_descriptor,
                    },
                    "state_hashes": {
                        "model_state_sha256": model_state,
                        "optimizer_state_sha256": optimizer_state,
                        "combined_state_sha256": semantic_digest(
                            {
                                "model_state_sha256": model_state,
                                "optimizer_state_sha256": optimizer_state,
                            }
                        ),
                        "outputs_sha256": outputs_digest,
                    },
                }
            )
            del before, deltas, raw_gradients, post_gradients, first_moments, second_moments
            if full_timing:
                execute_boundary("diagnostic")

        if full_timing:
            while evaluation_calls < EVALUATION_BOUNDARY_COUNT - 1:
                execute_boundary("diagnostic")
            execute_boundary("final")
            _require(
                evaluation_calls == EVALUATION_BOUNDARY_COUNT,
                "qualification did not execute exactly 13 evaluation boundaries",
            )
            checkpoint_receipt, final_checkpoint_io_seconds = self._checkpoint_io(
                capture_root=capture_root,
                model=model,
                optimizer=optimizer,
                benchmark_identity=benchmark_identity,
            )
            seal_receipt, outcome_seal_seconds = self._final_seal_io(
                capture_root=capture_root,
                benchmark_identity=benchmark_identity,
                profile=profile,
                panel_id=panel_id,
                steps=steps,
                boundaries=boundaries,
            )
        else:
            checkpoint_receipt = {
                "executed": False,
                "artifact_count": 0,
                "bytes": 0,
                "sha256": semantic_digest({"not_executed": "registered_numeric_only"}),
            }
            seal_receipt = {
                **checkpoint_receipt,
                "file_names": [],
                "metrics_row_count": 0,
                "prediction_row_count": 0,
                "summary_schema": "not_executed_registered_numeric_only",
                "attested_file_count": 0,
                "completion_attestation_verified": False,
                "complete_marker_final_write": False,
                "run_store_finalize_used": False,
                "outcome_file_seal_hashes": {},
                "outcome_file_seal_modes": {},
                "outcome_file_seal_file_count": 0,
                "outcome_file_seal_scan_and_chmod_completed": False,
                "seal_restore_only_for_authenticated_transient_cleanup": False,
            }
            final_checkpoint_io_seconds = 0.0
            outcome_seal_seconds = 0.0
        torch.cuda.synchronize(self.device)
        peak_allocated = int(torch.cuda.max_memory_allocated(self.device))
        peak_reserved = int(torch.cuda.max_memory_reserved(self.device))
        _require(peak_allocated <= peak_reserved, "CUDA allocated memory exceeds reserved memory")
        data_seconds = (
            max(self.preparation_seconds[(panel_id, law)] for law in LAW_FAMILIES) if full_timing else 0.0
        )
        other_law = next(law for law in LAW_FAMILIES if law != law_family)
        record_workload = _evaluation_workload(benchmark, evaluation_batch_size)
        other_workload = _evaluation_workload(
            self.evaluation_benchmarks[(panel_id, other_law)],
            evaluation_batch_size,
        )
        timing = {
            "tokenizer_load_initialization_seconds": (
                self.tokenizer_initialization_seconds[panel_id] if full_timing else 0.0
            ),
            "data_bank_render_tokenization_seconds": data_seconds,
            "model_load_seconds": model_load_seconds,
            "updates_1_to_8_seconds": clean_update_wall_seconds,
            "updates_1_to_8_cuda_event_seconds_diagnostic": clean_update_cuda_seconds,
            "tokenizer_host_envelopes_8_updates_seconds": tokenizer_host_envelopes_seconds,
            "update_timing_contract": clean_update_contract,
            "evaluation_13_boundaries_seconds": evaluation_seconds,
            "evaluation_callback_io_seconds": evaluation_callback_io_seconds,
            "final_checkpoint_io_seconds": final_checkpoint_io_seconds,
            "outcome_seal_seconds": outcome_seal_seconds,
            "update_timing_method": (
                "shared_train_steps_8_update_wall_clock_single_pre_post_cuda_sync_horizon"
                if full_timing
                else "not_measured_registered_numeric_only"
            ),
            "fresh_timing_only_model_optimizer": full_timing,
            "vector_state_capture_excluded_from_update_timing": full_timing,
            "warmup_model_discarded_before_measurement": full_timing,
            "capture_path_cuda_event_seconds_diagnostic_only": capture_path_update_cuda_seconds,
            "warmup_excluded": True,
            "profile_timing_order": list(self.profile_timing_order),
            "measured_end_to_end_seconds": (
                (self.tokenizer_initialization_seconds[panel_id] if full_timing else 0.0)
                + data_seconds
                + model_load_seconds
                + clean_update_wall_seconds
                + tokenizer_host_envelopes_seconds
                + evaluation_seconds
                + evaluation_callback_io_seconds
                + final_checkpoint_io_seconds
                + outcome_seal_seconds
            ),
            "evaluation_boundary_count": EVALUATION_BOUNDARY_COUNT if full_timing else 0,
            "projection_training_updates": 1_000,
            "worker_mix": {
                "runs_per_worker": 40,
                "panel_runs": {panel: 20 for panel in PANELS},
            },
            "projection_safety_multiplier": 1.20,
        }
        body = {
            "schema": PROCESS_SCHEMA,
            "schema_version": PROCESS_SCHEMA_VERSION,
            "identity": {
                "profile": profile,
                "replicate": replicate,
                "panel_id": panel_id,
                "model_name": PANELS[panel_id]["model_name"],
                "model_revision": PANELS[panel_id]["model_revision"],
                "law_family": law_family,
                "device_uuid": self.device_binding.uuid,
                "device_name": self.device_binding.name,
                "process_uuid": process_uuid,
            },
            "actual_model_execution": True,
            "profile_contract": copy.deepcopy(dict(contract)),
            "engineering_scope": copy.deepcopy(dict(self.scope)),
            "coverage": {
                "updates": list(UPDATES),
                "training_views": list(TRAINING_VIEWS),
                "action_labels": list(ACTION_LABELS),
                "dataset_sha256": corpus.dataset_sha256,
                "engineering_seed": ENGINEERING_SEED,
                "ordered_corpus_sample_ids_sha256": semantic_digest(list(corpus.sample_ids)),
                "order_algorithm": ("deterministic_effective_50_replace_final_with_corpus_max_if_absent_v1"),
                "worst_case_token_sample": {
                    "sample_id": corpus.worst.sample_id,
                    "view": corpus.worst.view,
                    "token_length": corpus.worst.token_length,
                    "corpus_max_token_length": corpus.worst.token_length,
                    "included_in_every_update": True,
                },
            },
            "trainable_parameters": manifest,
            "initial_state": {
                "fresh_model_instance": True,
                "fresh_optimizer_instance": True,
                "model_snapshot_sha256": _sha256(
                    self.model_snapshot_sha256[panel_id],
                    f"{panel_id} model snapshot digest",
                ),
                "model_state_sha256": initial_model,
                "optimizer_state_sha256": initial_optimizer,
            },
            "stochastic_modules": stochastic,
            "execution_benchmark": _seal(
                {
                    "tier": (
                        "full_production_timing_representative" if full_timing else "registered_numeric_only"
                    ),
                    "timing_representative": full_timing,
                    "worst_law_proof": {
                        "selected_law": self.timing_law(profile, panel_id),
                        "record_law": law_family,
                        "record_is_selected_worst": law_family == self.timing_law(profile, panel_id),
                        "law_max_token_length": benchmark.maximum_token_length,
                        "other_law_max_token_length": self.evaluation_benchmarks[
                            (panel_id, other_law)
                        ].maximum_token_length,
                        "record_padded_token_elements": record_workload["padded_token_elements"],
                        "other_law_padded_token_elements": other_workload["padded_token_elements"],
                        "record_maximum_padded_token_elements_per_call": record_workload[
                            "maximum_padded_token_elements_per_call"
                        ],
                        "other_law_maximum_padded_token_elements_per_call": other_workload[
                            "maximum_padded_token_elements_per_call"
                        ],
                        "record_maximum_flattened_prompts_per_call": record_workload[
                            "maximum_flattened_prompts_per_call"
                        ],
                        "other_law_maximum_flattened_prompts_per_call": other_workload[
                            "maximum_flattened_prompts_per_call"
                        ],
                        "record_scorer_call_count": record_workload["scorer_call_count"],
                        "other_law_scorer_call_count": other_workload["scorer_call_count"],
                        "record_call_shapes": record_workload["call_shapes"],
                        "other_law_call_shapes": other_workload["call_shapes"],
                        "record_call_shapes_sha256": record_workload["call_shapes_sha256"],
                        "other_law_call_shapes_sha256": other_workload["call_shapes_sha256"],
                    },
                    "data_bank_render_tokenization": {
                        "corpus_size": CORPUS_SIZE,
                        "law_family": law_family,
                        "six_training_views_rendered": set(corpus.views) == set(TRAINING_VIEWS),
                        "diagnostic_prompt_count": len(benchmark.diagnostic_prompts),
                        "final_prompt_count": len(benchmark.final_prompts),
                        "diagnostic_bank_example_counts": [len(bank) for bank in benchmark.diagnostic_banks],
                        "final_bank_example_counts": [len(bank) for bank in benchmark.final_banks],
                        "diagnostic_prompt_sha256": benchmark.diagnostic_prompt_sha256,
                        "final_prompt_sha256": benchmark.final_prompt_sha256,
                    },
                    "mode_transitions": {
                        "scorer_train_for_every_forward_backward": True,
                        "eval_only_at_boundaries_and_registered_parity_chunk": True,
                        "train_mode_restored_after_every_evaluation": True,
                        "dropout_effectively_zero_while_training": True,
                    },
                    "gradient_checkpointing": {
                        "configured": gradient_checkpointing,
                        "runtime_enabled": bool(getattr(model, "is_gradient_checkpointing", False)),
                        "training_graph_exercised": gradient_checkpointing,
                    },
                    "registered_production_shaped_parity_chunk": registered_parity_chunk,
                    "boundaries": boundaries,
                    "checkpoint_io": checkpoint_receipt,
                    "final_seal_io": seal_receipt,
                },
                "benchmark_digest",
            ),
            "steps": steps,
            "memory": {
                "cuda_synchronized_before_read": True,
                "peak_stats_reset_before_run": True,
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
            },
            "timing": timing,
        }
        record = seal_process_record(body)
        del optimizer, model, parameters, native_parameters
        torch.cuda.empty_cache()
        return CapturedProcess(record=record, vector_files=vector_files)


def _output_comparison(
    left_outputs: Sequence[Mapping[str, Any]],
    right_outputs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    _require(
        len(left_outputs) == len(right_outputs) == len(TRAINING_VIEWS), "paired output cardinality changed"
    )
    score_difference = 0.0
    probability_difference = 0.0
    sample_identity_exact = True
    actions_identical = True
    for left, right in zip(left_outputs, right_outputs, strict=True):
        sample_identity_exact = sample_identity_exact and (
            left["view"],
            left["sample_id"],
            left["token_length"],
        ) == (right["view"], right["sample_id"], right["token_length"])
        score_difference = max(
            score_difference,
            *(
                abs(float(lhs) - float(rhs))
                for lhs, rhs in zip(
                    left["normalized_log_scores"],
                    right["normalized_log_scores"],
                    strict=True,
                )
            ),
        )
        probability_difference = max(
            probability_difference,
            *(
                abs(float(lhs) - float(rhs))
                for lhs, rhs in zip(
                    left["probabilities"],
                    right["probabilities"],
                    strict=True,
                )
            ),
        )
        actions_identical = actions_identical and (left["action_index"], left["action_label"]) == (
            right["action_index"],
            right["action_label"],
        )
    return {
        "left_outputs_sha256": semantic_digest(list(left_outputs)),
        "right_outputs_sha256": semantic_digest(list(right_outputs)),
        "sample_identity_exact": sample_identity_exact,
        "maximum_normalized_score_difference": score_difference,
        "maximum_probability_difference": probability_difference,
        "actions_identical": actions_identical,
    }


def _metric_from_safetensors(left_path: Path, right_path: Path) -> dict[str, Any]:
    try:
        from safetensors import safe_open  # type: ignore[import-not-found]
    except ImportError as error:  # pragma: no cover - actual image dependency
        raise ProducerError("actual qualification requires safetensors") from error
    rows: list[dict[str, Any]] = []
    with (
        safe_open(str(left_path), framework="pt", device="cpu") as left_handle,
        safe_open(str(right_path), framework="pt", device="cpu") as right_handle,
    ):
        left_keys = sorted(left_handle.keys())
        right_keys = sorted(right_handle.keys())
        _require(left_keys == right_keys and bool(left_keys), "paired safetensors parameter keys differ")
        for order_index, key in enumerate(left_keys):
            left = left_handle.get_tensor(key)
            right = right_handle.get_tensor(key)
            _require(
                bool(torch.isfinite(left).all()) and bool(torch.isfinite(right).all()),
                f"paired vector {key} is non-finite",
            )
            left_squared, right_squared, difference_squared, dot = _float64_pair_sums(left, right)
            rows.append(
                {
                    "order_index": order_index,
                    "parameter_key": key,
                    "element_count": left.numel(),
                    "finite": True,
                    "left_squared_sum": left_squared,
                    "right_squared_sum": right_squared,
                    "difference_squared_sum": difference_squared,
                    "dot": dot,
                    "left_native_tensor_sha256": _native_tensor_sha256(left),
                    "right_native_tensor_sha256": _native_tensor_sha256(right),
                }
            )
    try:
        return streaming_metric_from_accumulator_rows(rows)
    except QualificationError as error:
        raise ProducerError("paired accumulator rows failed strict metric construction") from error


def _assert_logical_replay_exact(
    canonical: Mapping[str, Any],
    replay: Mapping[str, Any],
) -> None:
    _require(canonical["identity"] == replay["identity"], "logical replay identity changed")
    _require(
        canonical["profile_contract"] == replay["profile_contract"], "logical replay profile contract changed"
    )
    _require(canonical["coverage"] == replay["coverage"], "logical replay corpus binding changed")
    _require(
        canonical["trainable_parameters"] == replay["trainable_parameters"],
        "logical replay parameter manifest changed",
    )
    _require(
        canonical["initial_state"] == replay["initial_state"],
        "logical replay fresh model/optimizer state changed",
    )
    _require(
        canonical["stochastic_modules"] == replay["stochastic_modules"],
        "logical replay stochastic-module audit changed",
    )
    _require(
        canonical["execution_benchmark"]["registered_production_shaped_parity_chunk"]
        == replay["execution_benchmark"]["registered_production_shaped_parity_chunk"],
        "logical replay registered production-shaped parity chunk changed",
    )
    left_binding = canonical_process_comparison_binding(canonical)
    replay_binding = canonical_process_comparison_binding(replay)
    _require(
        left_binding["process_uuid"] == replay_binding["process_uuid"], "logical replay process UUID changed"
    )
    _require(
        left_binding["steps"] == replay_binding["steps"],
        "logical replay ordered IDs/outputs/state/vector manifests changed",
    )


def _comparison_process_keys(
    kind: str,
    panel_id: str,
    law_family: str,
    device_uuid: str,
) -> tuple[tuple[str, str, str, str, str], tuple[str, str, str, str, str]]:
    suffix = (panel_id, law_family, device_uuid)
    _require(kind == "baseline_vs_tuned", "only baseline/tuned numeric comparisons are retained")
    return ("baseline", "primary", *suffix), ("tuned", "primary", *suffix)


def _build_actual_comparison(
    *,
    kind: str,
    panel_id: str,
    law_family: str,
    device: GPUDevice,
    canonical_left: Mapping[str, Any],
    canonical_right: Mapping[str, Any],
    replay_left: CapturedProcess,
    replay_right: CapturedProcess,
    inventory: TransientInventory,
) -> dict[str, Any]:
    _assert_logical_replay_exact(canonical_left, replay_left.record)
    _assert_logical_replay_exact(canonical_right, replay_right.record)
    comparison_steps: list[dict[str, Any]] = []
    compared_file_digests: list[dict[str, Any]] = []
    for update, left_step, right_step in zip(
        UPDATES,
        replay_left.record["steps"],
        replay_right.record["steps"],
        strict=True,
    ):
        vector_metrics: dict[str, Any] = {}
        for vector_kind in VECTOR_KINDS:
            left_path = replay_left.vector_files[(update, vector_kind)]
            right_path = replay_right.vector_files[(update, vector_kind)]
            vector_metrics[vector_kind] = _metric_from_safetensors(left_path, right_path)
            compared_file_digests.append(
                {
                    "update": update,
                    "vector_kind": vector_kind,
                    "left_sha256": sha256_file(left_path),
                    "right_sha256": sha256_file(right_path),
                }
            )
            inventory.delete(left_path)
            inventory.delete(right_path)
        comparison_steps.append(
            {
                "update": update,
                "outputs": _output_comparison(left_step["outputs"], right_step["outputs"]),
                "vectors": vector_metrics,
            }
        )
    left_binding = canonical_process_comparison_binding(canonical_left)
    right_binding = canonical_process_comparison_binding(canonical_right)
    pair_capture_digest = semantic_digest(
        {
            "kind": kind,
            "panel_id": panel_id,
            "law_family": law_family,
            "device_uuid": device.uuid,
            "left_replay_steps_sha256": semantic_digest(replay_left.record["steps"]),
            "right_replay_steps_sha256": semantic_digest(replay_right.record["steps"]),
            "compared_temporary_safetensors": compared_file_digests,
            "temporary_files_deleted": True,
        }
    )
    paired_receipt = build_paired_execution_receipt(
        left_record=canonical_left,
        right_record=canonical_right,
        left_observed_binding=left_binding,
        right_observed_binding=right_binding,
        pair_capture_receipt_sha256=pair_capture_digest,
        left_capture_mode="exact_logical_identity_replay",
        right_capture_mode="exact_logical_identity_replay",
    )
    return seal_comparison_record(
        {
            "schema": COMPARISON_SCHEMA,
            "schema_version": COMPARISON_SCHEMA_VERSION,
            "kind": kind,
            "panel_id": panel_id,
            "law_family": law_family,
            "device_uuid": device.uuid,
            "left_record_digest": canonical_left["record_digest"],
            "right_record_digest": canonical_right["record_digest"],
            "paired_execution_receipt": paired_receipt,
            "steps": comparison_steps,
        }
    )


def _options_body(options: ControllerOptions) -> dict[str, Any]:
    return {
        "repo": str(options.repo.resolve()),
        "freeze_path": str(options.freeze_path.resolve()),
        "freeze_sha256": options.freeze_sha256,
        "execution_root": str(options.execution_root.resolve()),
        "engineering_root": str(options.engineering_root.resolve()),
        "provision_receipt": str(options.provision_receipt.resolve()),
        "provision_receipt_sha256": options.provision_receipt_sha256,
        "pod_id": options.pod_id,
        "model_receipts": {panel: str(options.model_receipts[panel].resolve()) for panel in PANELS},
        "integration_audits": {panel: str(options.integration_audits[panel].resolve()) for panel in PANELS},
        "execution_uuid": options.execution_uuid,
    }


def _options_from_body(value: Mapping[str, Any]) -> ControllerOptions:
    expected = {
        "repo",
        "freeze_path",
        "freeze_sha256",
        "execution_root",
        "engineering_root",
        "provision_receipt",
        "provision_receipt_sha256",
        "pod_id",
        "model_receipts",
        "integration_audits",
        "execution_uuid",
    }
    _require(set(value) == expected, "worker controller-option fields changed")
    model_receipts = value["model_receipts"]
    audits = value["integration_audits"]
    _require(
        isinstance(model_receipts, Mapping) and set(model_receipts) == set(PANELS),
        "worker model receipt paths changed",
    )
    _require(
        isinstance(audits, Mapping) and set(audits) == set(PANELS), "worker integration audit paths changed"
    )
    return ControllerOptions(
        repo=Path(str(value["repo"])).resolve(),
        freeze_path=Path(str(value["freeze_path"])).resolve(),
        freeze_sha256=_sha256(value["freeze_sha256"], "worker freeze SHA-256"),
        execution_root=Path(str(value["execution_root"])).resolve(),
        engineering_root=Path(str(value["engineering_root"])).resolve(),
        provision_receipt=Path(str(value["provision_receipt"])).resolve(),
        provision_receipt_sha256=_sha256(
            value["provision_receipt_sha256"],
            "worker provision receipt SHA-256",
        ),
        pod_id=str(value["pod_id"]),
        model_receipts={panel: Path(str(model_receipts[panel])).resolve() for panel in PANELS},
        integration_audits={panel: Path(str(audits[panel])).resolve() for panel in PANELS},
        execution_uuid=_canonical_uuid(value["execution_uuid"], "worker execution UUID"),
    )


def _validate_worker_request(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema",
        "schema_version",
        "execution_uuid",
        "request_kind",
        "qualification_branch",
        "probe_panel_id",
        "probe_law_family",
        "deadline_monotonic_seconds",
        "worker_index",
        "device",
        "controller_options",
        "controller_binding",
        "implementation_binding",
        "freeze_binding",
        "provision_binding",
        "model_receipt_bindings",
        "model_integration_audit_bindings",
        "snapshot_roots",
        "training_maximums",
        "engineering_scope",
        "raw_root",
        "shard_path",
        "receipt_path",
        "request_digest",
    }
    _require(set(value) == expected, "worker request fields changed")
    _require(
        value.get("schema") == WORKER_REQUEST_SCHEMA
        and value.get("schema_version") == WORKER_REQUEST_SCHEMA_VERSION,
        "worker request schema changed",
    )
    _verify_self_digest(value, "request_digest", "worker request")
    execution_uuid = _canonical_uuid(value.get("execution_uuid"), "worker request execution UUID")
    request_kind = str(value["request_kind"])
    _require(request_kind in {"capacity_probe_cell", "qualification"}, "worker request kind changed")
    if request_kind == "capacity_probe_cell":
        _require(
            value["qualification_branch"] is None
            and value["probe_panel_id"] in PANELS
            and value["probe_law_family"] in LAW_FAMILIES,
            "capacity probe worker identity changed",
        )
        _require(value["training_maximums"] is None, "capacity worker received timing-only maximums")
    else:
        _require(
            value["qualification_branch"] in {"tuned_probe_passed_full", "tuned_capacity_fallback_baseline"}
            and value["probe_panel_id"] is None
            and value["probe_law_family"] is None,
            "qualification worker branch/probe identity changed",
        )
        maxima = value["training_maximums"]
        _require(
            isinstance(maxima, Mapping)
            and set(maxima) == set(PANELS)
            and all(
                isinstance(maxima[panel], Mapping)
                and set(maxima[panel])
                == {"prompt", "action", "candidate_choices", "host_envelope_prompts", "proof"}
                for panel in PANELS
            ),
            "qualification worker training-maximum handoff changed",
        )
    deadline = value["deadline_monotonic_seconds"]
    _require(
        isinstance(deadline, (int, float))
        and not isinstance(deadline, bool)
        and math.isfinite(float(deadline)),
        "worker deadline is invalid",
    )
    options = _options_from_body(value["controller_options"])
    _require(options.execution_uuid == execution_uuid, "worker request/options UUID differs")
    index = value.get("worker_index")
    _require(type(index) is int and 0 <= index < 4, "worker request index is invalid")
    device = value.get("device")
    _require(
        isinstance(device, Mapping) and set(device) == {"host_ordinal", "uuid", "name"},
        "worker request device is malformed",
    )
    device = cast(Mapping[str, Any], device)
    _require(
        type(device["host_ordinal"]) is int and int(device["host_ordinal"]) >= 0,
        "worker request host ordinal is invalid",
    )
    _require(
        GPU_UUID_PATTERN.fullmatch(str(device["uuid"])) is not None
        and str(device["name"]).startswith("NVIDIA H200"),
        "worker request is not bound to H200",
    )
    for field in ("raw_root", "shard_path", "receipt_path"):
        _require(Path(str(value[field])).is_absolute(), f"worker request {field} is not absolute")
    return copy.deepcopy(dict(value))


def _authenticate_isolated_worker_device(device: GPUDevice) -> None:
    _require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == device.uuid,
        "worker CUDA visibility is not bound to its assigned UUID",
    )
    _require(
        torch.cuda.is_available() and torch.cuda.device_count() == 1,
        "worker does not expose exactly one logical CUDA device",
    )
    _require(
        torch.cuda.get_device_name(0).startswith("NVIDIA H200"), "worker logical CUDA device is not H200"
    )
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device.uuid}",
                "--query-gpu=uuid,name",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ProducerError("worker could not independently query its assigned UUID") from error
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    parts = [part.strip() for part in rows[0].split(",", maxsplit=1)] if len(rows) == 1 else []
    _require(
        parts == [device.uuid, device.name],
        "worker logical CUDA ordinal zero does not resolve to the assigned host UUID/name",
    )


def _logical_process_uuid(
    execution_uuid: str,
    *,
    profile: str,
    replicate: str,
    panel_id: str,
    law_family: str,
    device_uuid: str,
) -> str:
    namespace = uuid.UUID(execution_uuid)
    name = ":".join((profile, replicate, panel_id, law_family, device_uuid))
    return str(uuid.uuid5(namespace, name))


def _validate_capacity_cell_result(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema",
        "schema_version",
        "execution_uuid",
        "worker_index",
        "device",
        "panel_id",
        "law_family",
        "actual_model_execution",
        "model_receipt_digest",
        "resolved_config_sha256",
        "workload_sha256",
        "numerical_execution_sha256",
        "fresh_model_and_optimizer",
        "train_batch_exercised",
        "gradient_clip_exercised",
        "adamw_step_exercised",
        "optimizer_moments_resident_during_evaluation",
        "contiguous_evaluation_example_count",
        "flattened_evaluation_prompt_count",
        "cuda_synchronized_before_memory_read",
        "finite_execution",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "status",
        "disqualifier",
        "failure_type_sha256",
        "failure_message_sha256",
        "transient_peak_bytes",
        "transient_entries",
        "transient_final_inventory",
        "outcomes_seen",
        "itt_ledger_created",
        "g01_launch_authorized",
        "result_digest",
    }
    _require(set(value) == expected, "capacity cell result fields changed")
    _require(
        value["schema"] == CAPACITY_CELL_RESULT_SCHEMA
        and value["schema_version"] == CAPACITY_CELL_RESULT_SCHEMA_VERSION,
        "capacity cell result schema changed",
    )
    _verify_self_digest(value, "result_digest", "capacity cell result")
    _canonical_uuid(value["execution_uuid"], "capacity cell execution UUID")
    _require(
        type(value["worker_index"]) is int
        and 0 <= int(value["worker_index"]) < 4
        and value["panel_id"] in PANELS
        and value["law_family"] in LAW_FAMILIES,
        "capacity cell identity changed",
    )
    _sha256(value["model_receipt_digest"], "capacity cell model receipt digest")
    _sha256(value["resolved_config_sha256"], "capacity cell config digest")
    _sha256(value["workload_sha256"], "capacity cell workload digest")
    _sha256(value["numerical_execution_sha256"], "capacity cell numerical-execution digest")
    _require(
        value["actual_model_execution"] is True
        and value["status"] in {"passed", "recoverable_capacity_failure"},
        "capacity cell execution/status changed",
    )
    if value["status"] == "passed":
        _require(
            value["disqualifier"] is None
            and value["failure_type_sha256"] is None
            and value["failure_message_sha256"] is None,
            "passing capacity result contains a failure",
        )
    else:
        _require(value["disqualifier"] in TUNED_CAPACITY_DISQUALIFIERS, "capacity disqualifier changed")
        _sha256(value["failure_type_sha256"], "capacity failure type digest")
        _sha256(value["failure_message_sha256"], "capacity failure message digest")
    _require(
        isinstance(value["transient_entries"], list)
        and value["transient_final_inventory"] == []
        and value["outcomes_seen"] is False
        and value["itt_ledger_created"] is False
        and value["g01_launch_authorized"] is False,
        "capacity cell retained transient data or crossed a scientific boundary",
    )
    return copy.deepcopy(dict(value))


def run_worker(
    request_path: str | Path,
    *,
    engine_factory: Callable[..., WorkerEngine] = ActualModelWorkerEngine,
    authenticate_device: Callable[[GPUDevice], None] = _authenticate_isolated_worker_device,
) -> dict[str, Any]:
    """Execute one authenticated H200 worker request and write its immutable shard."""

    request = _validate_worker_request(_strict_json(request_path, "worker request"))
    options = _options_from_body(request["controller_options"])
    authenticated = authenticate_inputs(options)
    _require(
        request["freeze_binding"] == authenticated.freeze_binding
        and request["provision_binding"] == authenticated.provision_binding
        and request["model_receipt_bindings"] == authenticated.model_receipt_bindings
        and request["model_integration_audit_bindings"] == authenticated.model_integration_audit_bindings
        and request["snapshot_roots"] == authenticated.snapshot_roots,
        "worker independently authenticated different frozen inputs",
    )
    controller_binding = request["controller_binding"]
    implementation_binding = request["implementation_binding"]
    _require(
        sha256_file(controller_binding["path"]) == controller_binding["file_sha256"]
        and sha256_file(implementation_binding["path"]) == implementation_binding["file_sha256"],
        "worker controller/implementation bytes changed",
    )
    raw_device = request["device"]
    device = GPUDevice(
        ordinal=int(raw_device["host_ordinal"]),
        uuid=str(raw_device["uuid"]),
        name=str(raw_device["name"]),
    )
    authenticate_device(device)
    raw_root = Path(str(request["raw_root"])).resolve()
    inventory = TransientInventory(raw_root)
    model_snapshot_sha256 = {
        panel: semantic_digest(authenticated.model_receipt_bindings[panel]) for panel in PANELS
    }
    engine = engine_factory(
        repo=options.repo,
        snapshot_roots=authenticated.snapshot_roots,
        model_snapshot_sha256=model_snapshot_sha256,
        device=device,
        engineering_scope=request["engineering_scope"],
        inventory=inventory,
        profile_timing_order=(
            (PROFILES if int(request["worker_index"]) % 2 == 0 else tuple(reversed(PROFILES)))
            if request["qualification_branch"] == "tuned_probe_passed_full"
            else ("baseline",)
        ),
        capacity_probe_identity=(
            (str(request["probe_panel_id"]), str(request["probe_law_family"]))
            if request["request_kind"] == "capacity_probe_cell"
            else None
        ),
        training_maximum_bindings=request["training_maximums"],
    )
    prepare = getattr(engine, "prepare_corpora", None)
    _require(callable(prepare), "worker engine lacks authenticated corpus preparation")
    cast(Callable[[], Any], prepare)()
    _require(
        time.monotonic() < float(request["deadline_monotonic_seconds"]),
        "worker reached the producer deadline before execution",
    )
    if request["request_kind"] == "capacity_probe_cell":
        panel_id = str(request["probe_panel_id"])
        law_family = str(request["probe_law_family"])
        binding_builder = getattr(engine, "capacity_probe_bindings", None)
        cell_runner = getattr(engine, "run_capacity_probe_cell", None)
        _require(
            callable(binding_builder) and callable(cell_runner),
            "actual worker engine lacks capacity probe APIs",
        )
        bindings = cast(Callable[..., Mapping[str, Any]], binding_builder)(
            panel_id=panel_id,
            law_family=law_family,
        )
        base: dict[str, Any] = {
            "actual_model_execution": True,
            "fresh_model_and_optimizer": False,
            "train_batch_exercised": False,
            "gradient_clip_exercised": False,
            "adamw_step_exercised": False,
            "optimizer_moments_resident_during_evaluation": False,
            "contiguous_evaluation_example_count": 0,
            "flattened_evaluation_prompt_count": 0,
            "cuda_synchronized_before_memory_read": False,
            "finite_execution": True,
            "peak_allocated_bytes": 0,
            "peak_reserved_bytes": 0,
            **bindings,
        }
        disqualifier: str | None = None
        failure: BaseException | None = None
        try:
            base.update(
                cast(Callable[..., Mapping[str, Any]], cell_runner)(
                    panel_id=panel_id,
                    law_family=law_family,
                )
            )
            if int(base["peak_reserved_bytes"]) > 120 * 1024**3:
                disqualifier = "reserved_memory_ceiling_exceeded"
                failure = ProducerError("tuned capacity peak reserved bytes exceeded 120 GiB")
        except torch.cuda.OutOfMemoryError as error:
            disqualifier, failure = "cuda_out_of_memory", error
            base["finite_execution"] = True
        except NonFiniteCapacityError as error:
            disqualifier, failure = "nonfinite_capacity_execution", error
            base["finite_execution"] = False
        status = "passed" if disqualifier is None else "recoverable_capacity_failure"
        transient = inventory.final_body(execution_uuid=options.execution_uuid)
        result = _seal(
            {
                "schema": CAPACITY_CELL_RESULT_SCHEMA,
                "schema_version": CAPACITY_CELL_RESULT_SCHEMA_VERSION,
                "execution_uuid": options.execution_uuid,
                "worker_index": request["worker_index"],
                "device": device.as_dict(),
                "panel_id": panel_id,
                "law_family": law_family,
                **base,
                "model_receipt_digest": authenticated.model_receipt_bindings[panel_id]["receipt_digest"],
                "status": status,
                "disqualifier": disqualifier,
                "failure_type_sha256": (
                    None if failure is None else hashlib.sha256(type(failure).__name__.encode()).hexdigest()
                ),
                "failure_message_sha256": (
                    None if failure is None else hashlib.sha256(str(failure).encode()).hexdigest()
                ),
                "transient_peak_bytes": transient["peak_bytes"],
                "transient_entries": transient["entries"],
                "transient_final_inventory": transient["final_inventory"],
                "outcomes_seen": False,
                "itt_ledger_created": False,
                "g01_launch_authorized": False,
            },
            "result_digest",
        )
        _require(
            time.monotonic() < float(request["deadline_monotonic_seconds"]),
            "capacity worker exceeded the producer deadline",
        )
        result_path = Path(str(request["shard_path"])).resolve()
        _exclusive_json(result_path, result, mode=0o400)
        return _validate_capacity_cell_result(result)
    active_profiles = (
        PROFILES if request["qualification_branch"] == "tuned_probe_passed_full" else ("baseline",)
    )
    assignments = [
        {"profile": profile, "replicate": replicate, "panel_id": panel, "law_family": law}
        for profile in active_profiles
        for replicate in REPLICATES[profile]
        for panel in PANELS
        for law in LAW_FAMILIES
    ]
    records: list[Mapping[str, Any]] = []
    by_key: dict[tuple[str, str, str, str, str], Mapping[str, Any]] = {}
    timing_laws = {
        (profile, panel): engine.timing_law(profile, panel) for profile in active_profiles for panel in PANELS
    }
    representatives = [
        assignment
        for assignment in assignments
        if assignment["replicate"] == "primary"
        and assignment["law_family"] == timing_laws[(assignment["profile"], assignment["panel_id"])]
    ]
    profile_order = (
        (PROFILES if int(request["worker_index"]) % 2 == 0 else tuple(reversed(PROFILES)))
        if request["qualification_branch"] == "tuned_probe_passed_full"
        else ("baseline",)
    )
    execution_order = sorted(
        assignments,
        key=lambda assignment: (
            0 if assignment in representatives else 1,
            profile_order.index(str(assignment["profile"])),
            assignments.index(assignment),
        ),
    )
    for assignment in execution_order:
        process_uuid = _logical_process_uuid(
            options.execution_uuid,
            profile=assignment["profile"],
            replicate=assignment["replicate"],
            panel_id=assignment["panel_id"],
            law_family=assignment["law_family"],
            device_uuid=device.uuid,
        )
        captured = engine.run(
            **assignment,
            process_uuid=process_uuid,
            persist_vectors_under=None,
            full_timing=assignment in representatives,
        )
        _require(not captured.vector_files, "canonical process unexpectedly persisted raw vectors")
        record = captured.record
        key = (
            assignment["profile"],
            assignment["replicate"],
            assignment["panel_id"],
            assignment["law_family"],
            device.uuid,
        )
        by_key[key] = record
    records = [
        by_key[
            (
                assignment["profile"],
                assignment["replicate"],
                assignment["panel_id"],
                assignment["law_family"],
                device.uuid,
            )
        ]
        for assignment in assignments
    ]

    comparisons: list[Mapping[str, Any]] = []
    if int(request["worker_index"]) == 0 and request["qualification_branch"] == "tuned_probe_passed_full":
        for panel_id in PANELS:
            for law_family in LAW_FAMILIES:
                left_key, right_key = _comparison_process_keys(
                    "baseline_vs_tuned",
                    panel_id,
                    law_family,
                    device.uuid,
                )
                pair_root = raw_root / "numeric-pairs" / f"{panel_id}-{law_family}"
                left_record = by_key[left_key]
                right_record = by_key[right_key]
                replay_left = engine.run(
                    profile="baseline",
                    replicate="primary",
                    panel_id=panel_id,
                    law_family=law_family,
                    process_uuid=str(left_record["identity"]["process_uuid"]),
                    persist_vectors_under=pair_root / "left",
                    full_timing=False,
                )
                replay_right = engine.run(
                    profile="tuned",
                    replicate="primary",
                    panel_id=panel_id,
                    law_family=law_family,
                    process_uuid=str(right_record["identity"]["process_uuid"]),
                    persist_vectors_under=pair_root / "right",
                    full_timing=False,
                )
                comparisons.append(
                    _build_actual_comparison(
                        kind="baseline_vs_tuned",
                        panel_id=panel_id,
                        law_family=law_family,
                        device=device,
                        canonical_left=left_record,
                        canonical_right=right_record,
                        replay_left=replay_left,
                        replay_right=replay_right,
                        inventory=inventory,
                    )
                )

    transient = inventory.final_body(execution_uuid=options.execution_uuid)
    shard = _seal(
        {
            "schema": WORKER_SHARD_SCHEMA,
            "schema_version": WORKER_SHARD_SCHEMA_VERSION,
            "execution_uuid": options.execution_uuid,
            "qualification_branch": request["qualification_branch"],
            "worker_index": request["worker_index"],
            "device": device.as_dict(),
            "process_records": records,
            "comparison_records": comparisons,
            "actual_model_execution": True,
            "outcomes_seen": False,
            "itt_ledger_created": False,
            "g01_launch_authorized": False,
        },
        "shard_digest",
    )
    shard_path = Path(str(request["shard_path"])).resolve()
    _exclusive_json(shard_path, shard, mode=0o400)
    shard_binding = {
        "path": str(shard_path),
        "file_sha256": sha256_file(shard_path),
        "shard_digest": shard["shard_digest"],
        "bytes": shard_path.stat().st_size,
        "mode": stat.S_IMODE(shard_path.stat().st_mode),
    }
    receipt = _seal(
        {
            "schema": WORKER_RECEIPT_SCHEMA,
            "schema_version": WORKER_RECEIPT_SCHEMA_VERSION,
            "execution_uuid": options.execution_uuid,
            "qualification_branch": request["qualification_branch"],
            "worker_index": request["worker_index"],
            "device": device.as_dict(),
            "actual_model_execution": True,
            "controller_binding": copy.deepcopy(controller_binding),
            "implementation_binding": copy.deepcopy(implementation_binding),
            "freeze_binding": copy.deepcopy(authenticated.freeze_binding),
            "provision_binding": copy.deepcopy(authenticated.provision_binding),
            "model_receipt_bindings": copy.deepcopy(authenticated.model_receipt_bindings),
            "model_integration_audit_bindings": copy.deepcopy(authenticated.model_integration_audit_bindings),
            "engineering_scope": copy.deepcopy(request["engineering_scope"]),
            "assignments": assignments,
            "process_record_count": len(records),
            "numeric_comparison_record_count": len(comparisons),
            "shard": shard_binding,
            "transient_root": str(raw_root),
            "transient_peak_bytes": transient["peak_bytes"],
            "transient_entries": transient["entries"],
            "transient_final_inventory": transient["final_inventory"],
            "outcomes_seen": False,
            "itt_ledger_created": False,
            "g01_launch_authorized": False,
        },
        "receipt_digest",
    )
    receipt_path = Path(str(request["receipt_path"])).resolve()
    _exclusive_json(receipt_path, receipt, mode=0o400)
    _validate_worker_receipt_body(receipt)
    return receipt


WorkerLauncher = Callable[..., Sequence[Mapping[str, Any]]]


def _launch_workers(
    request_paths: Sequence[Path],
    *,
    deadline_monotonic: float,
) -> list[dict[str, Any]]:
    processes: list[dict[str, Any]] = []
    pending_log: Path | None = None
    try:
        for request_path in request_paths:
            request = _validate_worker_request(_strict_json(request_path, "worker request"))
            environment = dict(os.environ)
            environment.update(
                {
                    "CUDA_VISIBLE_DEVICES": str(request["device"]["uuid"]),
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                    "TOKENIZERS_PARALLELISM": "false",
                }
            )
            command = [
                sys.executable,
                str(request["controller_binding"]["path"]),
                "--mode",
                "worker",
                "--request",
                str(request_path),
            ]
            log_path = request_path.with_name(f"{request_path.stem}-worker.log")
            pending_log = log_path
            try:
                log_descriptor = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError as error:
                raise ProducerError("worker log already exists") from error
            log_handle = os.fdopen(log_descriptor, "wb", buffering=0)
            try:
                process = subprocess.Popen(
                    command,
                    cwd=str(request["controller_options"]["repo"]),
                    env=environment,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                )
            finally:
                log_handle.close()
            processes.append(
                {
                    "request_path": request_path,
                    "request": request,
                    "command": command,
                    "process": process,
                    "log_path": log_path,
                    "timed_out": False,
                    "cancelled": False,
                }
            )
            pending_log = None
    except BaseException as error:
        for item in processes:
            process = item["process"]
            if process.poll() is None:
                item["cancelled"] = True
                process.terminate()
        for item in processes:
            process = item["process"]
            try:
                process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        failed_logs = [item["log_path"] for item in processes]
        if pending_log is not None and pending_log.exists():
            failed_logs.append(pending_log)
        for log_path in failed_logs:
            with log_path.open("r+b") as log_handle:
                os.fsync(log_handle.fileno())
            os.chmod(log_path, 0o400)
        for item in processes:
            failure_path = item["request_path"].with_name(
                f"{item['request_path'].stem}-launch-failure-receipt.json"
            )
            _exclusive_json(
                failure_path,
                _seal(
                    {
                        "request_digest": item["request"]["request_digest"],
                        "worker_index": item["request"]["worker_index"],
                        "device_uuid": item["request"]["device"]["uuid"],
                        "argv_sha256": semantic_digest(item["command"]),
                        "exit_code": item["process"].returncode,
                        "cancelled": True,
                        "launch_error_type_sha256": hashlib.sha256(type(error).__name__.encode()).hexdigest(),
                        "launch_error_message_sha256": hashlib.sha256(str(error).encode()).hexdigest(),
                        "log_sha256": sha256_file(item["log_path"]),
                    },
                    "failure_digest",
                ),
                mode=0o400,
            )
        raise ProducerError("authenticated H200 worker launch failed and children were reaped") from error
    while any(item["process"].poll() is None for item in processes):
        if time.monotonic() >= deadline_monotonic:
            for item in processes:
                process = item["process"]
                if process.poll() is None:
                    item["timed_out"] = True
                    item["cancelled"] = True
                    process.terminate()
            termination_deadline = time.monotonic() + 10.0
            while (
                any(item["process"].poll() is None for item in processes)
                and time.monotonic() < termination_deadline
            ):
                time.sleep(0.05)
            for item in processes:
                process = item["process"]
                if process.poll() is None:
                    process.kill()
            break
        time.sleep(0.05)
    launch_receipts: list[dict[str, Any]] = []
    failures: list[str] = []
    for item in processes:
        process = item["process"]
        return_code = process.wait()
        log_path = item["log_path"]
        with log_path.open("r+b") as log_handle:
            os.fsync(log_handle.fileno())
        os.chmod(log_path, 0o400)
        launch_body = {
            "request_digest": item["request"]["request_digest"],
            "request_kind": item["request"]["request_kind"],
            "worker_index": item["request"]["worker_index"],
            "panel_id": item["request"]["probe_panel_id"],
            "law_family": item["request"]["probe_law_family"],
            "device_uuid": item["request"]["device"]["uuid"],
            "argv": item["command"],
            "argv_sha256": semantic_digest(item["command"]),
            "exit_code": return_code,
            "timed_out": item["timed_out"],
            "cancelled": item["cancelled"],
            "log_binding": {
                "path": str(log_path.resolve()),
                "file_sha256": sha256_file(log_path),
                "bytes": log_path.stat().st_size,
                "mode": stat.S_IMODE(log_path.stat().st_mode),
            },
        }
        launch_receipt = _seal(launch_body, "launch_digest")
        launch_path = item["request_path"].with_name(f"{item['request_path'].stem}-launch-receipt.json")
        _exclusive_json(launch_path, launch_receipt, mode=0o400)
        launch_receipts.append(
            {
                **launch_receipt,
                "path": str(launch_path.resolve()),
                "file_sha256": sha256_file(launch_path),
                "bytes": launch_path.stat().st_size,
                "mode": stat.S_IMODE(launch_path.stat().st_mode),
            }
        )
        if return_code != 0 or item["timed_out"]:
            failures.append(f"{item['request_path'].name}: exit={return_code}; timed_out={item['timed_out']}")
    _require(not failures, "authenticated H200 worker failure: " + " | ".join(failures))
    return launch_receipts


def _checkpoint_inventory_entry(path: Path, *, lifecycle: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), "controller checkpoint artifact is absent")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "lifecycle": lifecycle,
        "deleted": False,
    }


def _worker_request(
    *,
    options: ControllerOptions,
    authenticated: AuthenticatedInputs,
    controller_binding: Mapping[str, Any],
    implementation_binding: Mapping[str, Any],
    engineering_scope: Mapping[str, Any],
    device: GPUDevice,
    worker_index: int,
    request_kind: str,
    qualification_branch: str | None,
    probe_panel_id: str | None,
    probe_law_family: str | None,
    raw_root: Path,
    shard_path: Path,
    receipt_path: Path,
    deadline_monotonic_seconds: float,
    training_maximums: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return _seal(
        {
            "schema": WORKER_REQUEST_SCHEMA,
            "schema_version": WORKER_REQUEST_SCHEMA_VERSION,
            "execution_uuid": options.execution_uuid,
            "request_kind": request_kind,
            "qualification_branch": qualification_branch,
            "probe_panel_id": probe_panel_id,
            "probe_law_family": probe_law_family,
            "deadline_monotonic_seconds": deadline_monotonic_seconds,
            "worker_index": worker_index,
            "device": device.as_dict(),
            "controller_options": _options_body(options),
            "controller_binding": copy.deepcopy(dict(controller_binding)),
            "implementation_binding": copy.deepcopy(dict(implementation_binding)),
            "freeze_binding": copy.deepcopy(authenticated.freeze_binding),
            "provision_binding": copy.deepcopy(authenticated.provision_binding),
            "model_receipt_bindings": copy.deepcopy(authenticated.model_receipt_bindings),
            "model_integration_audit_bindings": copy.deepcopy(authenticated.model_integration_audit_bindings),
            "snapshot_roots": copy.deepcopy(authenticated.snapshot_roots),
            "training_maximums": (
                None if training_maximums is None else copy.deepcopy(dict(training_maximums))
            ),
            "engineering_scope": copy.deepcopy(dict(engineering_scope)),
            "raw_root": str(raw_root.resolve()),
            "shard_path": str(shard_path.resolve()),
            "receipt_path": str(receipt_path.resolve()),
        },
        "request_digest",
    )


def _run_capacity_probe_phase(
    *,
    options: ControllerOptions,
    authenticated: AuthenticatedInputs,
    devices: Sequence[GPUDevice],
    controller_binding: Mapping[str, Any],
    implementation_binding: Mapping[str, Any],
    engineering_scope: Mapping[str, Any],
    transient_root: Path,
    checkpoint_root: Path,
    deadline_monotonic: float,
    worker_launcher: WorkerLauncher,
) -> dict[str, Any]:
    cells_by_device: dict[str, list[dict[str, Any]]] = {device.uuid: [] for device in devices}
    checkpoint_entries: list[dict[str, Any]] = []
    raw_entries: list[dict[str, Any]] = []
    raw_roots: list[Path] = []
    launch_receipts_all: list[dict[str, Any]] = []
    peak_sum = 0
    for panel_id in PANELS:
        for law_family in LAW_FAMILIES:
            wave_peak = 0
            request_paths: list[Path] = []
            for worker_index, device in enumerate(devices):
                stem = f"capacity-{panel_id}-{law_family}-worker-{worker_index}"
                raw_root = transient_root / "capacity" / panel_id / law_family / f"worker-{worker_index}"
                request_path = checkpoint_root / f"{stem}-request.json"
                result_path = checkpoint_root / f"{stem}-result.json"
                unused_receipt_path = checkpoint_root / f"{stem}-unused-receipt.json"
                request = _worker_request(
                    options=options,
                    authenticated=authenticated,
                    controller_binding=controller_binding,
                    implementation_binding=implementation_binding,
                    engineering_scope=engineering_scope,
                    device=device,
                    worker_index=worker_index,
                    request_kind="capacity_probe_cell",
                    qualification_branch=None,
                    probe_panel_id=panel_id,
                    probe_law_family=law_family,
                    raw_root=raw_root,
                    shard_path=result_path,
                    receipt_path=unused_receipt_path,
                    deadline_monotonic_seconds=deadline_monotonic,
                    training_maximums=None,
                )
                _exclusive_json(request_path, request, mode=0o400)
                checkpoint_entries.append(
                    _checkpoint_inventory_entry(
                        request_path,
                        lifecycle="authenticated_capacity_probe_cell_request",
                    )
                )
                request_paths.append(request_path)
                raw_roots.append(raw_root)
            launch_receipts = list(
                worker_launcher(tuple(request_paths), deadline_monotonic=deadline_monotonic)
            )
            _require(len(launch_receipts) == 4, "capacity probe launch receipt count changed")
            launch_receipts_all.extend(copy.deepcopy(dict(item)) for item in launch_receipts)
            launches_by_worker = {int(item["worker_index"]): item for item in launch_receipts}
            for launch in launch_receipts:
                for path, lifecycle in (
                    (Path(str(launch["path"])), "capacity_probe_launch_receipt"),
                    (Path(str(launch["log_binding"]["path"])), "capacity_probe_worker_log"),
                ):
                    checkpoint_entries.append(_checkpoint_inventory_entry(path, lifecycle=lifecycle))
            for worker_index, device in enumerate(devices):
                stem = f"capacity-{panel_id}-{law_family}-worker-{worker_index}"
                result_path = checkpoint_root / f"{stem}-result.json"
                result = _validate_capacity_cell_result(_strict_json(result_path, "capacity cell result"))
                _require(
                    result["worker_index"] == worker_index
                    and result["device"] == device.as_dict()
                    and result["panel_id"] == panel_id
                    and result["law_family"] == law_family,
                    "capacity result identity differs from request",
                )
                checkpoint_entries.append(
                    _checkpoint_inventory_entry(
                        result_path,
                        lifecycle="authenticated_capacity_probe_cell_result",
                    )
                )
                launch = launches_by_worker[worker_index]
                log = launch["log_binding"]
                cell = _seal(
                    {
                        "panel_id": panel_id,
                        "law_family": law_family,
                        "device_uuid": device.uuid,
                        "actual_model_execution": True,
                        "isolated_subprocess": True,
                        "subprocess_argv": copy.deepcopy(launch["argv"]),
                        "subprocess_argv_sha256": launch["argv_sha256"],
                        "subprocess_exit_code": launch["exit_code"],
                        "process_exit_resets_device": True,
                        "model_receipt_digest": result["model_receipt_digest"],
                        "resolved_config_sha256": result["resolved_config_sha256"],
                        "workload_sha256": result["workload_sha256"],
                        "numerical_execution_sha256": result["numerical_execution_sha256"],
                        "fresh_model_and_optimizer": result["fresh_model_and_optimizer"],
                        "train_batch_exercised": result["train_batch_exercised"],
                        "gradient_clip_exercised": result["gradient_clip_exercised"],
                        "adamw_step_exercised": result["adamw_step_exercised"],
                        "optimizer_moments_resident_during_evaluation": result[
                            "optimizer_moments_resident_during_evaluation"
                        ],
                        "contiguous_evaluation_example_count": result["contiguous_evaluation_example_count"],
                        "flattened_evaluation_prompt_count": result["flattened_evaluation_prompt_count"],
                        "cuda_synchronized_before_memory_read": result[
                            "cuda_synchronized_before_memory_read"
                        ],
                        "finite_execution": result["finite_execution"],
                        "peak_allocated_bytes": result["peak_allocated_bytes"],
                        "peak_reserved_bytes": result["peak_reserved_bytes"],
                        "status": result["status"],
                        "disqualifier": result["disqualifier"],
                        "failure_type_sha256": result["failure_type_sha256"],
                        "failure_message_sha256": result["failure_message_sha256"],
                        "log_binding": {
                            "file_sha256": log["file_sha256"],
                            "bytes": log["bytes"],
                            "mode": log["mode"],
                            "retention": "compact_digest_only_raw_log_deleted_after_hash",
                        },
                    },
                    "cell_digest",
                )
                cells_by_device[device.uuid].append(cell)
                wave_peak += int(result["transient_peak_bytes"])
                raw_entries.extend(copy.deepcopy(result["transient_entries"]))
            peak_sum = max(peak_sum, wave_peak)
    device_receipts = []
    for device in devices:
        cells = cells_by_device[device.uuid]
        _require(len(cells) == 4, "capacity device did not produce four canonical cells")
        disqualifiers = sorted(
            {str(cell["disqualifier"]) for cell in cells if cell["status"] == "recoverable_capacity_failure"}
        )
        device_receipts.append(
            _seal(
                {
                    "device_uuid": device.uuid,
                    "device_name": device.name,
                    "host_ordinal": device.ordinal,
                    "probe_cells": cells,
                    "peak_allocated_bytes": max(int(cell["peak_allocated_bytes"]) for cell in cells),
                    "peak_reserved_bytes": max(int(cell["peak_reserved_bytes"]) for cell in cells),
                    "status": "passed" if not disqualifiers else "recoverable_capacity_failure",
                    "disqualifiers": disqualifiers,
                },
                "receipt_digest",
            )
        )
    probe = construct_tuned_capacity_probe(devices=devices, device_receipts=device_receipts)
    return {
        "probe": probe,
        "checkpoint_entries": checkpoint_entries,
        "raw_entries": raw_entries,
        "raw_roots": raw_roots,
        "transient_peak_sum": peak_sum,
        "launch_receipts": launch_receipts_all,
    }


def _delete_checkpoint_inventory_entry(entry: dict[str, Any]) -> None:
    path = Path(str(entry["path"]))
    _require(
        path.is_file() and sha256_file(path) == entry["sha256"],
        "controller checkpoint changed before deletion",
    )
    path.unlink()
    entry["deleted"] = True


def _evidence_with_observed_size(body: Mapping[str, Any]) -> dict[str, Any]:
    observed = 1
    evidence: dict[str, Any] = {}
    for _iteration in range(8):
        candidate = copy.deepcopy(dict(body))
        candidate["storage_contract"]["observed_persisted_evidence_bytes"] = observed
        evidence = seal_evidence(candidate)
        updated = (
            len(
                json.dumps(
                    evidence,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                ).encode("ascii")
            )
            + 1
        )
        if updated == observed:
            break
        observed = updated
    _require(observed <= PERSISTED_EVIDENCE_CEILING_BYTES, "compact qualification evidence exceeds 512 MiB")
    evidence["storage_contract"]["observed_persisted_evidence_bytes"] = observed
    evidence = seal_evidence({key: value for key, value in evidence.items() if key != "evidence_digest"})
    _require(
        len(canonical_json_bytes(evidence)) + 1 == observed,
        "qualification evidence byte-size fixed point failed",
    )
    return evidence


def run_controller(
    options: ControllerOptions,
    *,
    input_authenticator: Callable[[ControllerOptions], AuthenticatedInputs] = authenticate_inputs,
    device_discoverer: Callable[[], tuple[GPUDevice, ...]] = discover_h200_devices,
    worker_launcher: WorkerLauncher = _launch_workers,
    wall_clock: Callable[[], float] = time.time,
    monotonic_clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Authenticate, execute four H200 workers, aggregate, replay, and seal."""

    started_wall_seconds = wall_clock()
    started_monotonic_seconds = monotonic_clock()
    authenticated = input_authenticator(options)
    provision_remaining_at_start = (
        _utc_timestamp(
            authenticated.provision_terminate_after_utc,
            "provision termination",
        )
        - started_wall_seconds
    )
    _require(
        provision_remaining_at_start >= MINIMUM_PROVISION_REMAINING_AT_START_SECONDS,
        PROVISION_START_GUARD_MESSAGE,
    )
    producer_deadline_monotonic = started_monotonic_seconds + PRODUCER_CEILING_SECONDS
    devices = device_discoverer()
    _require(
        len(devices) == 4
        and [device.uuid for device in devices] == sorted({device.uuid for device in devices}),
        "controller GPU inventory is not exactly four sorted UUIDs",
    )
    _require(not options.engineering_root.exists(), "qualification engineering root already exists")
    options.engineering_root.mkdir(parents=True, mode=0o700)
    transient_root = options.engineering_root / TRANSIENT_ROOT_NAME
    transient_root.mkdir(mode=0o700)
    checkpoint_root = transient_root / "controller-checkpoints"
    checkpoint_root.mkdir(mode=0o700)
    controller_path = (options.repo / CONTROLLER_RELATIVE).resolve()
    implementation_path = (options.repo / IMPLEMENTATION_RELATIVE).resolve()
    _require(
        controller_path.is_file() and implementation_path.is_file(),
        "qualification controller or implementation is absent",
    )
    controller_argv = _options_argv(options)
    controller_binding: dict[str, Any] = {
        "path": str(controller_path),
        "file_sha256": sha256_file(controller_path),
        "argv": controller_argv,
        "argv_sha256": semantic_digest(controller_argv),
    }
    implementation_binding: dict[str, Any] = {
        "path": str(implementation_path),
        "file_sha256": sha256_file(implementation_path),
    }
    engineering_scope = {
        "root": str(options.engineering_root.resolve()),
        "root_class": "engineering_only",
        "data_class": "synthetic_engineering_only",
        "outcomes_seen": False,
        "itt_ledger_created": False,
        "production_artifacts_read": False,
        "production_artifacts_written": False,
    }
    capacity_phase = _run_capacity_probe_phase(
        options=options,
        authenticated=authenticated,
        devices=devices,
        controller_binding=controller_binding,
        implementation_binding=implementation_binding,
        engineering_scope=engineering_scope,
        transient_root=transient_root,
        checkpoint_root=checkpoint_root,
        deadline_monotonic=producer_deadline_monotonic,
        worker_launcher=worker_launcher,
    )
    tuned_capacity_probe = capacity_phase["probe"]
    qualification_branch = str(tuned_capacity_probe["qualification_branch"])
    training_maximums = build_controller_training_maximums(
        repo=options.repo,
        snapshot_roots=authenticated.snapshot_roots,
        deadline_monotonic=producer_deadline_monotonic,
    )
    _require(
        monotonic_clock() < producer_deadline_monotonic,
        "controller training-maximum scan exceeded the producer deadline",
    )
    request_paths: list[Path] = []
    checkpoint_entries = list(capacity_phase["checkpoint_entries"])
    for worker_index, device in enumerate(devices):
        worker_root = transient_root / f"worker-{worker_index}"
        request_path = checkpoint_root / f"worker-{worker_index}-request.json"
        shard_path = checkpoint_root / f"worker-{worker_index}-shard.json"
        receipt_path = checkpoint_root / f"worker-{worker_index}-receipt.json"
        request = _worker_request(
            options=options,
            authenticated=authenticated,
            controller_binding=controller_binding,
            implementation_binding=implementation_binding,
            engineering_scope=engineering_scope,
            device=device,
            worker_index=worker_index,
            request_kind="qualification",
            qualification_branch=qualification_branch,
            probe_panel_id=None,
            probe_law_family=None,
            raw_root=worker_root / "raw",
            shard_path=shard_path,
            receipt_path=receipt_path,
            deadline_monotonic_seconds=producer_deadline_monotonic,
            training_maximums=training_maximums,
        )
        _exclusive_json(request_path, request, mode=0o400)
        checkpoint_entries.append(
            _checkpoint_inventory_entry(request_path, lifecycle="authenticated_worker_request")
        )
        request_paths.append(request_path)
    qualification_launch_receipts = list(
        worker_launcher(tuple(request_paths), deadline_monotonic=producer_deadline_monotonic)
    )
    _require(len(qualification_launch_receipts) == 4, "qualification launch receipt count changed")
    for launch in qualification_launch_receipts:
        for path, lifecycle in (
            (Path(str(launch["path"])), "qualification_launch_receipt"),
            (Path(str(launch["log_binding"]["path"])), "qualification_worker_log"),
        ):
            checkpoint_entries.append(_checkpoint_inventory_entry(path, lifecycle=lifecycle))

    worker_receipts: list[dict[str, Any]] = []
    process_records: list[Mapping[str, Any]] = []
    comparison_records: list[Mapping[str, Any]] = []
    worker_peak_sum = int(capacity_phase["transient_peak_sum"])
    raw_entries: list[dict[str, Any]] = list(capacity_phase["raw_entries"])
    for worker_index, request_path in enumerate(request_paths):
        request = _validate_worker_request(_strict_json(request_path, "worker request"))
        receipt_path = Path(str(request["receipt_path"]))
        shard_path = Path(str(request["shard_path"]))
        receipt = _validate_worker_receipt_body(_strict_json(receipt_path, "worker receipt"))
        _require(receipt["worker_index"] == worker_index, "worker receipt index changed")
        _require(
            receipt["shard"]
            == {
                "path": str(shard_path),
                "file_sha256": sha256_file(shard_path),
                "shard_digest": _strict_json(shard_path, "worker shard")["shard_digest"],
                "bytes": shard_path.stat().st_size,
                "mode": stat.S_IMODE(shard_path.stat().st_mode),
            },
            "worker receipt shard binding changed",
        )
        shard = _strict_json(shard_path, "worker shard")
        _verify_self_digest(shard, "shard_digest", "worker shard")
        _require(
            shard["execution_uuid"] == options.execution_uuid
            and shard["qualification_branch"] == qualification_branch
            and shard["worker_index"] == worker_index
            and shard["device"] == devices[worker_index].as_dict()
            and shard["actual_model_execution"] is True
            and shard["outcomes_seen"] is False
            and shard["itt_ledger_created"] is False
            and shard["g01_launch_authorized"] is False,
            "worker shard authentication boundary changed",
        )
        expected_worker_process_count = 16 if qualification_branch == "tuned_probe_passed_full" else 8
        _require(
            len(shard["process_records"]) == expected_worker_process_count,
            "worker shard process count changed",
        )
        _require(
            len(shard["comparison_records"])
            == (4 if worker_index == 0 and qualification_branch == "tuned_probe_passed_full" else 0),
            "worker shard numeric comparison count changed",
        )
        worker_receipts.append(receipt)
        process_records.extend(shard["process_records"])
        comparison_records.extend(shard["comparison_records"])
        worker_peak_sum += int(receipt["transient_peak_bytes"])
        raw_entries.extend(copy.deepcopy(receipt["transient_entries"]))
        for path, lifecycle in (
            (shard_path, "authenticated_worker_shard_checkpoint"),
            (receipt_path, "authenticated_worker_receipt_checkpoint"),
        ):
            checkpoint_entries.append(_checkpoint_inventory_entry(path, lifecycle=lifecycle))

    expected_process_count = 64 if qualification_branch == "tuned_probe_passed_full" else 32
    expected_comparison_count = 4 if qualification_branch == "tuned_probe_passed_full" else 0
    _require(
        len(process_records) == expected_process_count
        and len(comparison_records) == expected_comparison_count,
        "controller aggregate cardinality changed",
    )
    records_manifest = semantic_digest(
        {
            "qualification_branch": qualification_branch,
            "tuned_capacity_probe_digest": tuned_capacity_probe["probe_digest"],
            "process_record_digests": sorted(str(item["record_digest"]) for item in process_records),
            "comparison_record_digests": sorted(
                str(item["comparison_digest"]) for item in comparison_records
            ),
        }
    )
    accumulator_row_count = sum(
        len(metric["accumulator_rows"])
        for comparison in comparison_records
        for step in comparison["steps"]
        for metric in step["vectors"].values()
    )
    evidence_body: dict[str, Any] = {
        "schema": EVIDENCE_SCHEMA,
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "actual_model_execution": True,
        "qualification_branch": qualification_branch,
        "tuned_capacity_probe": copy.deepcopy(dict(tuned_capacity_probe)),
        "evidence_source": {
            **copy.deepcopy(dict(EVIDENCE_BOUNDARY)),
            "capture_command_sha256": controller_binding["argv_sha256"],
            "capture_implementation_sha256": implementation_binding["file_sha256"],
            "records_manifest_sha256": records_manifest,
        },
        "engineering_scope": engineering_scope,
        "freeze_binding": copy.deepcopy(authenticated.freeze_binding),
        "provision_binding": copy.deepcopy(authenticated.provision_binding),
        "model_receipt_bindings": copy.deepcopy(authenticated.model_receipt_bindings),
        "model_integration_audit_bindings": copy.deepcopy(authenticated.model_integration_audit_bindings),
        "storage_contract": {
            "maximum_persisted_evidence_bytes": PERSISTED_EVIDENCE_CEILING_BYTES,
            "observed_persisted_evidence_bytes": 1,
            "accumulator_row_count": accumulator_row_count,
            "raw_vector_bytes_persisted": 0,
            "vector_persistence": "bounded_inline_per_parameter_accumulator_rows_only",
            "transient_raw_cleanup_before_itt_required": True,
            "transient_raw_cleanup_receipt_required": True,
            "transient_raw_cleanup_timing": "during_producer_before_compact_evidence_seal",
            "compact_evidence_retained_through_final_gate": True,
        },
        "expected_device_uuids": [device.uuid for device in devices],
        "process_records": process_records,
        "comparison_records": comparison_records,
    }
    evidence = _evidence_with_observed_size(evidence_body)
    validate_qualification_evidence(evidence)
    evidence_path = options.engineering_root / EVIDENCE_NAME
    _exclusive_json(evidence_path, evidence, mode=0o400)
    _require(
        evidence_path.stat().st_size <= PERSISTED_EVIDENCE_CEILING_BYTES,
        "persisted qualification evidence exceeds 512 MiB",
    )
    report_path = options.engineering_root / REPORT_NAME
    report = write_qualification_report(evidence=evidence, output=report_path)
    os.chmod(report_path, 0o400)
    validate_qualification_report(report)

    for entry in checkpoint_entries:
        _delete_checkpoint_inventory_entry(entry)
    for worker_index in range(4):
        raw = transient_root / f"worker-{worker_index}" / "raw"
        worker = raw.parent
        _require(raw.is_dir() and not any(raw.iterdir()), "worker raw root is not empty")
        raw.rmdir()
        worker.rmdir()
    for raw_root in capacity_phase["raw_roots"]:
        raw = Path(raw_root)
        _require(raw.is_dir() and not any(raw.iterdir()), "capacity raw root is not empty")
        raw.rmdir()
    capacity_root = transient_root / "capacity"
    for directory in sorted(
        (path for path in capacity_root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        directory.rmdir()
    capacity_root.rmdir()
    checkpoint_root.rmdir()
    _require(not any(transient_root.iterdir()), "global transient root is not exactly empty")
    all_entries = [*raw_entries, *checkpoint_entries]
    _require(
        len({str(entry["path"]) for entry in all_entries}) == len(all_entries),
        "global transient inventory paths are duplicated",
    )
    _require(
        all(entry["deleted"] is True and not Path(str(entry["path"])).exists() for entry in all_entries),
        "global transient inventory retained an artifact",
    )
    peak_bytes = max(worker_peak_sum, max((int(entry["bytes"]) for entry in checkpoint_entries), default=0))
    _require(peak_bytes <= TRANSIENT_CEILING_BYTES, "global transient peak exceeded one TiB")
    transient = _seal(
        {
            "schema": TRANSIENT_INVENTORY_SCHEMA,
            "schema_version": TRANSIENT_INVENTORY_SCHEMA_VERSION,
            "execution_uuid": options.execution_uuid,
            "transient_root": str(transient_root.resolve()),
            "ceiling_bytes": TRANSIENT_CEILING_BYTES,
            "peak_bytes": peak_bytes,
            "entries": all_entries,
            "final_inventory": [],
            "final_bytes": 0,
            "all_listed_paths_absent": True,
            "compact_manifest_only": True,
            "raw_payloads_embedded": False,
            "outcomes_seen": False,
            "itt_ledger_created": False,
            "g01_launch_authorized": False,
        },
        "inventory_digest",
    )
    _validate_transient_inventory(transient)
    transient_path = options.engineering_root / TRANSIENT_INVENTORY_NAME
    _exclusive_json(transient_path, transient, mode=0o400)
    artifacts = {
        "evidence": _direct_file_binding(
            evidence_path,
            kind="evidence",
            digest_field="evidence_digest",
        ),
        "report": _direct_file_binding(report_path, kind="report", digest_field="report_digest"),
        "transient_inventory": _direct_file_binding(
            transient_path,
            kind="transient_inventory",
            digest_field="inventory_digest",
        ),
    }
    completed_wall_seconds = wall_clock()
    completed_monotonic_seconds = monotonic_clock()
    _require(
        completed_monotonic_seconds <= producer_deadline_monotonic,
        "qualification producer exceeded its frozen 105-minute ceiling",
    )
    deadline_binding = _deadline_binding(
        started_wall_seconds=started_wall_seconds,
        started_monotonic_seconds=started_monotonic_seconds,
        completed_wall_seconds=completed_wall_seconds,
        completed_monotonic_seconds=completed_monotonic_seconds,
        provision_terminate_after_utc=authenticated.provision_terminate_after_utc,
    )
    receipt = _seal(
        {
            "schema": PRODUCER_SCHEMA,
            "schema_version": PRODUCER_SCHEMA_VERSION,
            "execution_uuid": options.execution_uuid,
            "actual_model_execution": True,
            "qualification_branch": evidence["qualification_branch"],
            "tuned_capacity_probe": copy.deepcopy(dict(evidence["tuned_capacity_probe"])),
            "deadline_binding": deadline_binding,
            "controller_binding": controller_binding,
            "implementation_binding": implementation_binding,
            "freeze_binding": copy.deepcopy(authenticated.freeze_binding),
            "provision_binding": copy.deepcopy(authenticated.provision_binding),
            "model_receipt_bindings": copy.deepcopy(authenticated.model_receipt_bindings),
            "model_integration_audit_bindings": copy.deepcopy(authenticated.model_integration_audit_bindings),
            "gpu_uuids": [device.uuid for device in devices],
            "worker_receipts": worker_receipts,
            "worker_launch_receipts": [
                {
                    key: copy.deepcopy(value)
                    for key, value in launch.items()
                    if key not in {"path", "file_sha256", "bytes", "mode"}
                }
                for launch in qualification_launch_receipts
            ],
            "artifacts": artifacts,
            "engineering_scope": engineering_scope,
            "transient_storage": {
                "root": transient["transient_root"],
                "ceiling_bytes": TRANSIENT_CEILING_BYTES,
                "peak_bytes": peak_bytes,
                "final_inventory": [],
                "final_bytes": 0,
                "all_listed_paths_absent": True,
                "raw_vectors_persisted": False,
                "compact_evidence_retained_through_final_gate": True,
            },
            "weight_updates_scope": "engineering_qualification_only",
            "outcomes_seen": False,
            "itt_ledger_created": False,
            "g01_launch_authorized": False,
        },
        "receipt_digest",
    )
    receipt_path = options.engineering_root / PRODUCER_RECEIPT_NAME
    _exclusive_json(receipt_path, receipt, mode=0o400)
    return verify_producer_receipt(
        receipt_path=receipt_path,
        expected_controller_sha256=controller_binding["file_sha256"],
        expected_implementation_sha256=implementation_binding["file_sha256"],
        expected_controller_argv=controller_argv,
        expected_freeze_binding=authenticated.freeze_binding,
        expected_provision_binding=authenticated.provision_binding,
        expected_model_receipt_bindings=authenticated.model_receipt_bindings,
        expected_integration_audit_bindings=authenticated.model_integration_audit_bindings,
        expected_gpu_uuids=[device.uuid for device in devices],
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("controller", "worker"))
    parser.add_argument("--request")
    parser.add_argument("--repo")
    parser.add_argument("--freeze")
    parser.add_argument("--freeze-sha256")
    parser.add_argument("--execution-root")
    parser.add_argument("--engineering-root")
    parser.add_argument("--provision-receipt")
    parser.add_argument("--provision-receipt-sha256")
    parser.add_argument("--pod-id")
    parser.add_argument("--model-receipt-0p5b")
    parser.add_argument("--model-receipt-1p5b")
    parser.add_argument("--integration-audit-0p5b")
    parser.add_argument("--integration-audit-1p5b")
    parser.add_argument("--execution-uuid")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    namespace = _parser().parse_args(arguments)
    if namespace.mode == "worker":
        _require(namespace.request is not None, "worker mode requires --request")
        _require(
            arguments == ["--mode", "worker", "--request", str(Path(namespace.request).resolve())],
            "worker invocation is not canonical",
        )
        run_worker(Path(namespace.request).resolve())
        return 0
    required = (
        "repo",
        "freeze",
        "freeze_sha256",
        "execution_root",
        "engineering_root",
        "provision_receipt",
        "provision_receipt_sha256",
        "pod_id",
        "model_receipt_0p5b",
        "model_receipt_1p5b",
        "integration_audit_0p5b",
        "integration_audit_1p5b",
        "execution_uuid",
    )
    _require(
        all(getattr(namespace, field) is not None for field in required),
        "controller invocation is incomplete",
    )
    options = ControllerOptions(
        repo=Path(namespace.repo).resolve(),
        freeze_path=Path(namespace.freeze).resolve(),
        freeze_sha256=str(namespace.freeze_sha256),
        execution_root=Path(namespace.execution_root).resolve(),
        engineering_root=Path(namespace.engineering_root).resolve(),
        provision_receipt=Path(namespace.provision_receipt).resolve(),
        provision_receipt_sha256=str(namespace.provision_receipt_sha256),
        pod_id=str(namespace.pod_id),
        model_receipts={
            "g00f-0p5b": Path(namespace.model_receipt_0p5b).resolve(),
            "g00f-1p5b": Path(namespace.model_receipt_1p5b).resolve(),
        },
        integration_audits={
            "g00f-0p5b": Path(namespace.integration_audit_0p5b).resolve(),
            "g00f-1p5b": Path(namespace.integration_audit_1p5b).resolve(),
        },
        execution_uuid=str(namespace.execution_uuid),
    )
    expected = _options_argv(options)[1:]
    _require(arguments == expected, "controller invocation is not canonical")
    run_controller(options)
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI
    raise SystemExit(main())
