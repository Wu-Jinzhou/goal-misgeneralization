"""Pure, outcome-blind contracts for a prospective G01 qualification.

This module is an engineering source checkpoint, not an executor.  It exposes
only deterministic scheduling, strict control-evidence validation, and integer
feasibility arithmetic.  Every high-level execution entry refuses before any
runtime, provider, model, filesystem, or scientific action.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from types import MappingProxyType
from typing import Any, ClassVar, Literal, NoReturn

REFUSAL = "G01Q_RUNTIME_PROVISION_NOT_FROZEN"
STUDY_ID = "g01_compute_qualification"
REVIEW_SCHEMA = "goalzendo.g01q_source_review_contract"
RUN_RECEIPT_SCHEMA = "goalzendo.g01q_run_receipt"
REPORT_SCHEMA = "goalzendo.g01q_report"
SCHEMA_VERSION = 1

SCIENTIFIC_PAIR_COUNT = 60
SCIENTIFIC_RUN_COUNT = 120
ENGINEERING_PAIR_COUNT = 8
ENGINEERING_RUN_COUNT = 16
WORKER_COUNTS = (4, 8)
DEADLINE_MENU_SECONDS = (172_800, 259_200, 345_600, 432_000)
ENGINEERING_SEEDS = tuple(range(8_611_107, 8_611_115))
ENGINEERING_CELLS = (
    ("parity", 8_000),
    ("parity", 9_500),
    ("parity", 10_000),
    ("majority", 8_000),
    ("majority", 9_500),
    ("majority", 10_000),
)
ALGORITHMS = ("sft", "outcome_rl")
GIB = 1 << 30
TIB = 1 << 40

G01_TARGET: Mapping[str, Any] = MappingProxyType(
    {
        "config_file_sha256": "ee6d53556189a6b6e25cfbfb204325017a058c63a605653df4c175463e00ce18",
        "canonical_config_digest": "f9f91978a446a6e750172e0e377bc1b738ccef64b5238992555e1afb6e6bd110",
        "target_binding_digest": "3315d20a6f9bdae3c5fdaf9567c5bce7b592890d0ebd26e10682002d816bf0c6",
        "guard_signature": "9feaae82edf801aad4bd4a5b16f633be8dfa2dbd41b29601e7b61b564c763464",
        "protocol_file_sha256": "428243c3da271bc2d79e16c0fde20d47076eb3837c6740ab733e278cddcdbce9",
        "runner_file_sha256": "46b55ad4bdd08073e5f89ae101e0862372817b74b590f07ddb8f8c4331e5b9e1",
        "source_fingerprint": "1a8146377b9a9620690025671614edb2dd20d214f4e195da3cf3528809b2c694",
        "plan_rows_digest": "f51f6b6f574295433dacec8fafe20508526036ef15dce220a1d8c24e8cd6e55f",
        "plan_key_set_digest": "7fb1bc870c3b93d1d6a5ae6148b83b460c9ee6d246b592f67f908e8080fa8b91",
        "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "model_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        "planned_runs": SCIENTIFIC_RUN_COUNT,
        "planned_pairs": SCIENTIFIC_PAIR_COUNT,
        "cell_count": 12,
    }
)

# Every value is immutable.  Serialization materializes fresh lists, so a
# caller cannot mutate later contracts or their digest through a shallow copy.
PRODUCTION_SHAPE: Mapping[str, Any] = MappingProxyType(
    {
        "model_dtype": "bfloat16",
        "update_method": "full",
        "optimizer": "AdamW",
        "weight_decay_decimal": "0.0",
        "warmup_fraction_numerator": 5,
        "warmup_fraction_denominator": 100,
        "gradient_clip_decimal": "1.0",
        "sft_learning_rate_decimal": "0.000003",
        "outcome_rl_learning_rate_decimal": "0.000003",
        "sft_entropy_coefficient_decimal": "0.0",
        "outcome_rl_entropy_coefficient_decimal": "0.01",
        "updates": 1_000,
        "micro_batch": 10,
        "gradient_accumulation": 5,
        "effective_batch": 50,
        "samples_per_prompt": 4,
        "max_length": 640,
        "train_examples": 10_000,
        "validation_examples": 1_000,
        "train_presentations": 50_000,
        "eval_steps": (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768, 1_000),
        "factorial_cells": 8,
        "prompt_views": (
            "full",
            "audit_law_full",
            "audit_law_matched",
            "no_herald",
            "no_sage",
            "law_only",
        ),
        "causal_prompt_views": ("full",),
        "causal_per_cell": 16,
        "final_causal_per_cell": 64,
        "intermediate_eval_per_cell": 64,
        "final_eval_per_cell": 512,
        "eval_batch": 32,
        "checkpoint_steps": (256, 512, 768, 1_000),
        "final_weight_snapshot_steps": (1_000,),
        "bf16": True,
        "full_finetune": True,
        "gradient_checkpointing": True,
        "deterministic_algorithms": True,
        "allow_tf32": False,
        "cublas_workspace_config": ":4096:8",
        "overlength_policy": "fail_without_truncation",
        "save_predictions": True,
        "checkpoint_write_policy": "atomic_hash_fsync_syncfs",
        "checkpoint_predecessor_retirement": True,
        "optimizer_state_retired_after_completion": True,
        "final_weights_persisted": True,
        "completion_marker_after_durable_rows": True,
        "cuda_synchronize_before_pair_end": True,
        "model_teardown_before_pair_end": True,
    }
)

_CANDIDATE_PROFILES = (
    ("h200x8", "NVIDIA H200", 8, 79_200),
    ("h100-hbm3x8", "NVIDIA H100 80GB HBM3", 8, 79_200),
    ("h200x4", "NVIDIA H200", 4, 151_200),
    ("h100-hbm3x4", "NVIDIA H100 80GB HBM3", 4, 151_200),
)

_PHASE_KEYS = frozenset(
    {
        "model_and_data_materialization",
        "optimization",
        "evaluation_and_serialization",
        "cuda_sync_and_teardown",
    }
)
_RUN_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "schema_version",
        "study_id",
        "campaign_uuid",
        "qualification_uuid",
        "candidate_id",
        "candidate_profile_id",
        "candidate_priority",
        "candidate_policy_digest",
        "prior_candidate_dispositions_digest",
        "gpu_id",
        "cloud_type",
        "data_center_id",
        "network_volume_id",
        "network_volume_type",
        "network_volume_mount",
        "qualification_ceiling_seconds",
        "wave_index",
        "wave_release_monotonic_ns",
        "compute_freeze_sha256",
        "campaign_intent_sha256",
        "provision_receipt_sha256",
        "source_runtime_model_bindings_digest",
        "fixed_control_envelope_manifest_digest",
        "worker_count",
        "schedule_digest",
        "row_index",
        "pair_index",
        "worker_index",
        "algorithm_order_index",
        "gpu_uuid",
        "algorithm",
        "rule_family",
        "q_p_basis_points",
        "engineering_seed",
        "production_shape_digest",
        "started_monotonic_ns",
        "completed_monotonic_ns",
        "wall_ns",
        "phase_wall_ns",
        "maximum_reserved_gpu_bytes",
        "total_gpu_bytes",
        "maximum_peak_rss_bytes",
        "maximum_live_bytes",
        "final_bytes",
        "maximum_live_inodes",
        "final_inodes",
        "bytes_written",
        "completion_file_count",
        "state",
        "retry",
        "resume",
        "scientific_values_persisted",
        "scientific_values_read_by_control",
        "scientific_value_branching",
        "metric_prediction_stream_padded_to_frozen_bytes",
        "model_weight_serialization_uncompressed",
        "control_log_fixed_schema",
        "raw_scientific_tree_absent",
        "forbidden_field_scan_passed",
        "receipt_digest",
    }
)

_GPU_UUID = re.compile(r"GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_CANDIDATE_ID = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?")
_DATA_CENTER_ID = re.compile(r"[A-Z0-9]+(?:-[A-Z0-9]+){1,7}")
_NETWORK_VOLUME_ID = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9_-]{0,126}[A-Za-z0-9])?")


class QualificationError(RuntimeError):
    """A pure qualification contract or refusal invariant failed."""


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if type(key) is not str:
                raise QualificationError("canonical JSON object keys must be exact text")
            result[key] = _plain(child)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain(child) for child in value]
    if value is None or type(value) in {str, int, bool}:
        return value
    raise QualificationError(f"unsupported canonical JSON type: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Return the sole compact canonical JSON representation used here."""

    return json.dumps(
        _plain(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


CANDIDATE_POLICY_DIGEST = digest(
    [
        {
            "profile_id": profile_id,
            "gpu_id": gpu_id,
            "gpu_count": gpu_count,
            "qualification_ceiling_seconds": ceiling,
        }
        for profile_id, gpu_id, gpu_count, ceiling in _CANDIDATE_PROFILES
    ]
)


def strict_canonical_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    """Parse one duplicate-free, integer-only, byte-canonical JSON object."""

    if type(payload) is not bytes:
        raise QualificationError(f"{label} must be exact bytes")

    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, child in pairs:
            if key in result:
                raise QualificationError(f"{label} contains duplicate JSON key {key!r}")
            result[key] = child
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_float=lambda _value: (_ for _ in ()).throw(
                QualificationError(f"{label} contains a floating-point number")
            ),
            parse_constant=lambda constant: (_ for _ in ()).throw(
                QualificationError(f"{label} contains non-finite {constant}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise QualificationError(f"{label} is not strict UTF-8 JSON") from error
    if type(value) is not dict:
        raise QualificationError(f"{label} must contain one exact object")
    if canonical_json_bytes(value) != payload:
        raise QualificationError(f"{label} is not in the exact canonical byte encoding")
    return value


def _exact_equal(observed: Any, expected: Any, label: str) -> None:
    if type(observed) is not type(expected):
        raise QualificationError(f"{label} JSON type changed")
    if type(expected) is dict:
        if set(observed) != set(expected):
            raise QualificationError(f"{label} JSON field set changed")
        for key in expected:
            _exact_equal(observed[key], expected[key], f"{label}.{key}")
    elif type(expected) is list:
        if len(observed) != len(expected):
            raise QualificationError(f"{label} JSON list length changed")
        for index, (child, expected_child) in enumerate(zip(observed, expected, strict=True)):
            _exact_equal(child, expected_child, f"{label}[{index}]")
    elif observed != expected:
        raise QualificationError(f"{label} JSON value changed")


def _exact_object(value: Any, fields: frozenset[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise QualificationError(f"{label} exact field set changed")
    return value


def _exact_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise QualificationError(f"{label} must be an exact integer >= {minimum}")
    return value


def _exact_bool(value: Any, expected: bool, label: str) -> None:
    if type(value) is not bool or value is not expected:
        raise QualificationError(f"{label} must be exact boolean {expected}")


def _exact_text(value: Any, expected: str, label: str) -> None:
    if type(value) is not str or value != expected:
        raise QualificationError(f"{label} changed")


def _sha256(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise QualificationError(f"{label} must be one lowercase SHA-256")
    return value


def _uuid4(value: Any, label: str) -> str:
    if type(value) is not str:
        raise QualificationError(f"{label} must be one canonical UUIDv4")
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, ValueError) as error:
        raise QualificationError(f"{label} must be one canonical UUIDv4") from error
    if parsed.version != 4 or str(parsed) != value:
        raise QualificationError(f"{label} must be one canonical UUIDv4")
    return value


def _gpu_uuid(value: Any, label: str) -> str:
    if type(value) is not str or _GPU_UUID.fullmatch(value) is None:
        raise QualificationError(f"{label} must be one canonical lowercase NVIDIA GPU UUID")
    return value


def _candidate_id(value: Any, label: str) -> str:
    if type(value) is not str or _CANDIDATE_ID.fullmatch(value) is None:
        raise QualificationError(f"{label} must be one canonical candidate identifier")
    return value


def _data_center_id(value: Any, label: str) -> str:
    if type(value) is not str or _DATA_CENTER_ID.fullmatch(value) is None:
        raise QualificationError(f"{label} must be one exact provider data-center identifier")
    return value


def _network_volume_id(value: Any, label: str) -> str:
    if type(value) is not str or _NETWORK_VOLUME_ID.fullmatch(value) is None:
        raise QualificationError(f"{label} must be one exact provider network-volume identifier")
    return value


def _ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise QualificationError("ceil-div denominator must be positive")
    return -(-numerator // denominator)


@dataclass(frozen=True)
class EngineeringRow:
    row_index: int
    pair_index: int
    worker_index: int
    algorithm_order_index: int
    algorithm: str
    rule_family: str
    q_p_basis_points: int
    engineering_seed: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "row_index": self.row_index,
            "pair_index": self.pair_index,
            "worker_index": self.worker_index,
            "algorithm_order_index": self.algorithm_order_index,
            "algorithm": self.algorithm,
            "rule_family": self.rule_family,
            "q_p_basis_points": self.q_p_basis_points,
            "engineering_seed": self.engineering_seed,
        }


@dataclass(frozen=True)
class EngineeringSchedule:
    worker_count: int
    rows: tuple[EngineeringRow, ...]
    schedule_digest: str

    def for_worker(self, worker_index: int) -> tuple[EngineeringRow, ...]:
        _exact_int(worker_index, "worker_index")
        if worker_index >= self.worker_count:
            raise QualificationError("worker_index is outside the schedule")
        return tuple(row for row in self.rows if row.worker_index == worker_index)


def build_engineering_schedule(worker_count: int) -> EngineeringSchedule:
    """Build the exact structural 16-run engineering schedule."""

    _exact_int(worker_count, "worker_count")
    if worker_count not in WORKER_COUNTS:
        raise QualificationError("worker_count must be exactly 4 or 8")
    rows: list[EngineeringRow] = []
    for pair_index, engineering_seed in enumerate(ENGINEERING_SEEDS):
        rule_family, q_p_basis_points = ENGINEERING_CELLS[pair_index % len(ENGINEERING_CELLS)]
        worker_index = pair_index % worker_count
        algorithm_order = ALGORITHMS if worker_index % 2 == 0 else tuple(reversed(ALGORITHMS))
        for order_index, algorithm in enumerate(algorithm_order):
            rows.append(
                EngineeringRow(
                    row_index=len(rows),
                    pair_index=pair_index,
                    worker_index=worker_index,
                    algorithm_order_index=order_index,
                    algorithm=algorithm,
                    rule_family=rule_family,
                    q_p_basis_points=q_p_basis_points,
                    engineering_seed=engineering_seed,
                )
            )
    payload = [row.as_dict() for row in rows]
    if (
        len(rows) != ENGINEERING_RUN_COUNT
        or len({row.pair_index for row in rows}) != ENGINEERING_PAIR_COUNT
        or {row.algorithm for row in rows} != set(ALGORITHMS)
        or {(row.rule_family, row.q_p_basis_points) for row in rows} != set(ENGINEERING_CELLS)
    ):
        raise QualificationError("internal engineering schedule invariant failed")
    return EngineeringSchedule(worker_count, tuple(rows), digest(payload))


@dataclass(frozen=True)
class ComputeProjection:
    worker_count: int
    maximum_pairs_per_worker: int
    maximum_pair_wall_seconds: int
    raw_projected_study_wall_seconds: int
    qualified_projected_study_wall_seconds: int
    selected_deadline_seconds: int | None
    projected_study_gpu_seconds: int | None
    projected_cost_micro_usd: int | None
    gates: Mapping[str, bool]
    caps_satisfied: bool
    g01_launch_authorized: ClassVar[Literal[False]] = False


def project_compute(
    *,
    worker_count: int,
    maximum_pair_wall_seconds: int,
    price_micro_usd_per_gpu_hour: int,
    operator_maximum_wall_seconds: int,
    operator_maximum_gpu_seconds: int,
    operator_maximum_cost_micro_usd: int,
    candidate_maximum_wall_seconds: int,
    candidate_maximum_gpu_seconds: int,
    candidate_maximum_cost_micro_usd: int,
) -> ComputeProjection:
    """Apply the preregistered integer-only wall/GPU-hour/cost arithmetic.

    This helper reports cap arithmetic only.  It cannot qualify a route; only
    :func:`validate_campaign` can combine all 16 bound receipts and gates.
    """

    build_engineering_schedule(worker_count)
    qmax = _exact_int(maximum_pair_wall_seconds, "maximum_pair_wall_seconds", minimum=1)
    price = _exact_int(
        price_micro_usd_per_gpu_hour,
        "price_micro_usd_per_gpu_hour",
        minimum=1,
    )
    operator_wall = _exact_int(
        operator_maximum_wall_seconds,
        "operator_maximum_wall_seconds",
        minimum=1,
    )
    operator_gpu = _exact_int(
        operator_maximum_gpu_seconds,
        "operator_maximum_gpu_seconds",
        minimum=1,
    )
    operator_cost = _exact_int(
        operator_maximum_cost_micro_usd,
        "operator_maximum_cost_micro_usd",
        minimum=1,
    )
    candidate_wall = _exact_int(
        candidate_maximum_wall_seconds,
        "candidate_maximum_wall_seconds",
        minimum=1,
    )
    candidate_gpu = _exact_int(
        candidate_maximum_gpu_seconds,
        "candidate_maximum_gpu_seconds",
        minimum=1,
    )
    candidate_cost = _exact_int(
        candidate_maximum_cost_micro_usd,
        "candidate_maximum_cost_micro_usd",
        minimum=1,
    )
    pairs_per_worker = _ceil_div(SCIENTIFIC_PAIR_COUNT, worker_count)
    raw = pairs_per_worker * qmax
    qualified = _ceil_div(135 * raw + 100 * 7_200, 100 * 3_600) * 3_600
    selected = next((deadline for deadline in DEADLINE_MENU_SECONDS if deadline >= qualified), None)
    projected_gpu_seconds = worker_count * selected if selected is not None else None
    projected_cost = (
        _ceil_div(price * projected_gpu_seconds, 3_600) if projected_gpu_seconds is not None else None
    )
    gates = {
        "deadline_menu": selected is not None,
        "operator_wall": selected is not None and selected <= operator_wall,
        "candidate_wall": selected is not None and selected <= candidate_wall,
        "operator_gpu": projected_gpu_seconds is not None and projected_gpu_seconds <= operator_gpu,
        "candidate_gpu": projected_gpu_seconds is not None and projected_gpu_seconds <= candidate_gpu,
        "operator_cost": projected_cost is not None and projected_cost <= operator_cost,
        "candidate_cost": projected_cost is not None and projected_cost <= candidate_cost,
    }
    return ComputeProjection(
        worker_count=worker_count,
        maximum_pairs_per_worker=pairs_per_worker,
        maximum_pair_wall_seconds=qmax,
        raw_projected_study_wall_seconds=raw,
        qualified_projected_study_wall_seconds=qualified,
        selected_deadline_seconds=selected,
        projected_study_gpu_seconds=projected_gpu_seconds,
        projected_cost_micro_usd=projected_cost,
        gates=MappingProxyType(gates),
        caps_satisfied=all(gates.values()),
    )


def memory_gate(*, total_gpu_bytes: int, maximum_reserved_gpu_bytes: int) -> bool:
    total = _exact_int(total_gpu_bytes, "total_gpu_bytes", minimum=1)
    reserved = _exact_int(maximum_reserved_gpu_bytes, "maximum_reserved_gpu_bytes")
    safety = max(8 * GIB, _ceil_div(total, 10))
    return total >= safety and reserved <= total - safety


@dataclass(frozen=True)
class StorageProjection:
    required_free_bytes: int
    required_free_inodes: int
    projected_aggregate_write_bytes: int


def project_storage(
    *,
    worker_count: int,
    final_bytes_per_run: int,
    maximum_live_bytes_per_run: int,
    fixed_bytes: int,
    final_inodes_per_run: int,
    maximum_live_inodes_per_run: int,
    fixed_inodes: int,
    maximum_bytes_written_per_run: int,
) -> StorageProjection:
    build_engineering_schedule(worker_count)
    final_bytes = _exact_int(final_bytes_per_run, "final_bytes_per_run")
    live_bytes = _exact_int(maximum_live_bytes_per_run, "maximum_live_bytes_per_run")
    fixed_byte_count = _exact_int(fixed_bytes, "fixed_bytes")
    final_inodes = _exact_int(final_inodes_per_run, "final_inodes_per_run")
    live_inodes = _exact_int(maximum_live_inodes_per_run, "maximum_live_inodes_per_run")
    fixed_inode_count = _exact_int(fixed_inodes, "fixed_inodes")
    bytes_written = _exact_int(maximum_bytes_written_per_run, "maximum_bytes_written_per_run")
    if live_bytes < final_bytes or live_inodes < final_inodes or bytes_written < final_bytes:
        raise QualificationError("storage projection envelope is internally inconsistent")
    byte_base = (
        SCIENTIFIC_RUN_COUNT * final_bytes
        + worker_count * max(0, live_bytes - final_bytes)
        + fixed_byte_count
    )
    inode_base = (
        SCIENTIFIC_RUN_COUNT * final_inodes
        + worker_count * max(0, live_inodes - final_inodes)
        + fixed_inode_count
    )
    return StorageProjection(
        required_free_bytes=max(TIB, _ceil_div(5 * byte_base, 4)),
        required_free_inodes=max(1_000_000, _ceil_div(5 * inode_base, 4)),
        projected_aggregate_write_bytes=_ceil_div(
            5 * SCIENTIFIC_RUN_COUNT * bytes_written,
            4,
        ),
    )


PRODUCTION_SHAPE_DIGEST = digest(PRODUCTION_SHAPE)


def validate_run_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one schedule-bound, control-only G01Q run receipt."""

    receipt = _exact_object(value, _RUN_RECEIPT_KEYS, "G01Q run receipt")
    _exact_text(receipt["schema"], RUN_RECEIPT_SCHEMA, "receipt.schema")
    _exact_text(receipt["study_id"], STUDY_ID, "receipt.study_id")
    if type(receipt["schema_version"]) is not int or receipt["schema_version"] != SCHEMA_VERSION:
        raise QualificationError("G01Q run receipt schema_version changed")
    campaign_uuid = _uuid4(receipt["campaign_uuid"], "campaign_uuid")
    qualification_uuid = _uuid4(receipt["qualification_uuid"], "qualification_uuid")
    if campaign_uuid == qualification_uuid:
        raise QualificationError("campaign and qualification UUIDs must differ")
    _candidate_id(receipt["candidate_id"], "candidate_id")
    for field in (
        "compute_freeze_sha256",
        "campaign_intent_sha256",
        "provision_receipt_sha256",
        "source_runtime_model_bindings_digest",
        "fixed_control_envelope_manifest_digest",
    ):
        _sha256(receipt[field], field)

    worker_count = _exact_int(receipt["worker_count"], "worker_count")
    schedule = build_engineering_schedule(worker_count)
    if type(receipt["candidate_profile_id"]) is not str:
        raise QualificationError("candidate_profile_id must be exact text")
    matching_profiles = [
        profile
        for profile in _CANDIDATE_PROFILES
        if profile[0] == receipt["candidate_profile_id"] and profile[2] == worker_count
    ]
    if len(matching_profiles) != 1:
        raise QualificationError("candidate profile is not compatible with worker_count")
    profile = matching_profiles[0]
    priority = _exact_int(receipt["candidate_priority"], "candidate_priority")
    if priority >= len(_CANDIDATE_PROFILES) or _CANDIDATE_PROFILES[priority] != profile:
        raise QualificationError("candidate priority/profile binding changed")
    _exact_text(receipt["candidate_policy_digest"], CANDIDATE_POLICY_DIGEST, "candidate_policy_digest")
    _sha256(receipt["prior_candidate_dispositions_digest"], "prior_candidate_dispositions_digest")
    _exact_text(receipt["gpu_id"], profile[1], "gpu_id")
    _exact_text(receipt["cloud_type"], "SECURE", "cloud_type")
    _data_center_id(receipt["data_center_id"], "data_center_id")
    _network_volume_id(receipt["network_volume_id"], "network_volume_id")
    _exact_text(
        receipt["network_volume_type"],
        "HIGH_PERFORMANCE",
        "network_volume_type",
    )
    _exact_text(receipt["network_volume_mount"], "/workspace", "network_volume_mount")
    ceiling = _exact_int(
        receipt["qualification_ceiling_seconds"],
        "qualification_ceiling_seconds",
        minimum=1,
    )
    if ceiling != profile[3]:
        raise QualificationError("qualification ceiling changed from the candidate profile")
    _exact_text(receipt["schedule_digest"], schedule.schedule_digest, "schedule_digest")
    row_index = _exact_int(receipt["row_index"], "row_index")
    if row_index >= ENGINEERING_RUN_COUNT:
        raise QualificationError("row_index is outside the exact engineering schedule")
    row = schedule.rows[row_index]
    expected_wave_index = row.pair_index // worker_count
    wave_index = _exact_int(receipt["wave_index"], "wave_index")
    if wave_index != expected_wave_index:
        raise QualificationError("receipt wave_index is not bound to its schedule row")
    for field in (
        "pair_index",
        "worker_index",
        "algorithm_order_index",
        "q_p_basis_points",
        "engineering_seed",
    ):
        observed = _exact_int(receipt[field], field)
        if observed != getattr(row, field):
            raise QualificationError(f"receipt {field} is not bound to its exact schedule row")
    for field in ("algorithm", "rule_family"):
        _exact_text(receipt[field], getattr(row, field), field)
    _gpu_uuid(receipt["gpu_uuid"], "gpu_uuid")
    _exact_text(
        receipt["production_shape_digest"],
        PRODUCTION_SHAPE_DIGEST,
        "production_shape_digest",
    )

    started = _exact_int(receipt["started_monotonic_ns"], "started_monotonic_ns")
    wave_release = _exact_int(
        receipt["wave_release_monotonic_ns"],
        "wave_release_monotonic_ns",
    )
    completed = _exact_int(receipt["completed_monotonic_ns"], "completed_monotonic_ns", minimum=1)
    wall = _exact_int(receipt["wall_ns"], "wall_ns", minimum=1)
    if completed <= started or wall != completed - started:
        raise QualificationError("G01Q receipt monotonic interval is inconsistent")
    if started < wave_release:
        raise QualificationError("run started before its wave release")
    phase = _exact_object(receipt["phase_wall_ns"], _PHASE_KEYS, "phase_wall_ns")
    phase_values = {key: _exact_int(phase[key], f"phase_wall_ns.{key}") for key in _PHASE_KEYS}
    if phase_values["optimization"] < 1 or phase_values["evaluation_and_serialization"] < 1:
        raise QualificationError("optimization and evaluation/serialization phases must be nonzero")
    if any(child < 1 for child in phase_values.values()):
        raise QualificationError("every required full-shape phase must be nonzero")
    if sum(phase_values.values()) > wall:
        raise QualificationError("phase wall times cannot exceed complete run wall time")

    reserved = _exact_int(
        receipt["maximum_reserved_gpu_bytes"],
        "maximum_reserved_gpu_bytes",
    )
    total = _exact_int(receipt["total_gpu_bytes"], "total_gpu_bytes", minimum=1)
    _exact_int(receipt["maximum_peak_rss_bytes"], "maximum_peak_rss_bytes")
    live_bytes = _exact_int(receipt["maximum_live_bytes"], "maximum_live_bytes")
    final_bytes = _exact_int(receipt["final_bytes"], "final_bytes")
    live_inodes = _exact_int(receipt["maximum_live_inodes"], "maximum_live_inodes")
    final_inodes = _exact_int(receipt["final_inodes"], "final_inodes")
    bytes_written = _exact_int(receipt["bytes_written"], "bytes_written")
    _exact_int(receipt["completion_file_count"], "completion_file_count", minimum=1)
    if reserved < 1 or receipt["maximum_peak_rss_bytes"] < 1:
        raise QualificationError("GPU and host resource measurements must be nonzero")
    if reserved > total:
        raise QualificationError("reserved GPU bytes exceed total GPU bytes")
    if final_bytes < 1 or live_bytes < 1 or final_inodes < 1 or live_inodes < 1 or bytes_written < 1:
        raise QualificationError("full-shape storage measurements must be nonzero")
    if final_bytes > live_bytes or final_inodes > live_inodes or bytes_written < final_bytes:
        raise QualificationError("receipt storage envelope is internally inconsistent")

    _exact_text(receipt["state"], "complete", "state")
    for field in (
        "retry",
        "resume",
        "scientific_values_persisted",
        "scientific_values_read_by_control",
        "scientific_value_branching",
        "metric_prediction_stream_padded_to_frozen_bytes",
        "model_weight_serialization_uncompressed",
        "control_log_fixed_schema",
        "raw_scientific_tree_absent",
        "forbidden_field_scan_passed",
    ):
        expected = field in {
            "metric_prediction_stream_padded_to_frozen_bytes",
            "model_weight_serialization_uncompressed",
            "control_log_fixed_schema",
            "raw_scientific_tree_absent",
            "forbidden_field_scan_passed",
        }
        _exact_bool(receipt[field], expected, field)
    body = {key: child for key, child in receipt.items() if key != "receipt_digest"}
    _exact_text(receipt["receipt_digest"], digest(body), "receipt_digest")
    return dict(receipt)


def validate_run_receipt_bytes(payload: bytes) -> dict[str, Any]:
    """Validate one receipt directly from its sole accepted byte encoding."""

    return validate_run_receipt(strict_canonical_json_bytes(payload, "G01Q run receipt"))


@dataclass(frozen=True)
class QualificationReport:
    worker_count: int
    schedule_digest: str
    campaign_uuid: str
    qualification_uuid: str
    candidate_id: str
    candidate_profile_id: str
    candidate_priority: int
    candidate_policy_digest: str
    prior_candidate_dispositions_digest: str
    gpu_id: str
    cloud_type: str
    data_center_id: str
    network_volume_id: str
    network_volume_type: str
    network_volume_mount: str
    qualification_ceiling_seconds: int
    run_receipts_digest: str
    input_bindings: Mapping[str, Any]
    maximum_pair_wall_seconds: int
    maximum_reserved_gpu_bytes: int
    minimum_gpu_headroom_bytes: int
    maximum_peak_rss_bytes: int
    maximum_live_bytes_per_run: int
    maximum_final_bytes_per_run: int
    maximum_live_inodes_per_run: int
    maximum_final_inodes_per_run: int
    maximum_bytes_written_per_run: int
    required_free_bytes: int
    required_free_inodes: int
    projected_aggregate_write_bytes: int
    raw_projected_study_wall_seconds: int
    qualified_projected_study_wall_seconds: int
    selected_deadline_seconds: int | None
    projected_study_gpu_seconds: int | None
    projected_cost_micro_usd: int | None
    gates: Mapping[str, bool]
    supplied_evidence_gates_satisfied: bool
    report_digest: str
    evidence_externally_authenticated: ClassVar[Literal[False]] = False
    same_final_pod_verified: ClassVar[Literal[False]] = False
    compute_route_qualified: ClassVar[Literal[False]] = False
    g01_launch_authorized: ClassVar[Literal[False]] = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": REPORT_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "study_id": STUDY_ID,
            "worker_count": self.worker_count,
            "schedule_digest": self.schedule_digest,
            "campaign_uuid": self.campaign_uuid,
            "qualification_uuid": self.qualification_uuid,
            "candidate_id": self.candidate_id,
            "candidate_profile_id": self.candidate_profile_id,
            "candidate_priority": self.candidate_priority,
            "candidate_policy_digest": self.candidate_policy_digest,
            "prior_candidate_dispositions_digest": self.prior_candidate_dispositions_digest,
            "gpu_id": self.gpu_id,
            "cloud_type": self.cloud_type,
            "data_center_id": self.data_center_id,
            "network_volume_id": self.network_volume_id,
            "network_volume_type": self.network_volume_type,
            "network_volume_mount": self.network_volume_mount,
            "qualification_ceiling_seconds": self.qualification_ceiling_seconds,
            "run_receipts_digest": self.run_receipts_digest,
            "input_bindings": _plain(self.input_bindings),
            "pair_count": ENGINEERING_PAIR_COUNT,
            "run_count": ENGINEERING_RUN_COUNT,
            "maximum_pair_wall_seconds": self.maximum_pair_wall_seconds,
            "maximum_reserved_gpu_bytes": self.maximum_reserved_gpu_bytes,
            "minimum_gpu_headroom_bytes": self.minimum_gpu_headroom_bytes,
            "maximum_peak_rss_bytes": self.maximum_peak_rss_bytes,
            "maximum_live_bytes_per_run": self.maximum_live_bytes_per_run,
            "maximum_final_bytes_per_run": self.maximum_final_bytes_per_run,
            "maximum_live_inodes_per_run": self.maximum_live_inodes_per_run,
            "maximum_final_inodes_per_run": self.maximum_final_inodes_per_run,
            "maximum_bytes_written_per_run": self.maximum_bytes_written_per_run,
            "required_free_bytes": self.required_free_bytes,
            "required_free_inodes": self.required_free_inodes,
            "projected_aggregate_write_bytes": self.projected_aggregate_write_bytes,
            "raw_projected_study_wall_seconds": self.raw_projected_study_wall_seconds,
            "qualified_projected_study_wall_seconds": (self.qualified_projected_study_wall_seconds),
            "selected_deadline_seconds": self.selected_deadline_seconds,
            "projected_study_gpu_seconds": self.projected_study_gpu_seconds,
            "projected_cost_micro_usd": self.projected_cost_micro_usd,
            "gates": dict(self.gates),
            "supplied_evidence_gates_satisfied": self.supplied_evidence_gates_satisfied,
            "evidence_externally_authenticated": False,
            "same_final_pod_verified": False,
            "compute_route_qualified": False,
            "g01_launch_authorized": False,
            "report_digest": self.report_digest,
        }


def validate_campaign(
    receipts: Sequence[Mapping[str, Any]],
    *,
    worker_count: int,
    sorted_gpu_uuids: Sequence[str],
    expected_campaign_uuid: str,
    expected_qualification_uuid: str,
    expected_candidate_id: str,
    candidate_profile_id: str,
    candidate_priority: int,
    gpu_id: str,
    cloud_type: str,
    data_center_id: str,
    network_volume_id: str,
    network_volume_type: str,
    network_volume_mount: str,
    qualification_ceiling_seconds: int,
    prior_candidate_dispositions: Sequence[Mapping[str, Any]],
    expected_compute_freeze_sha256: str,
    expected_campaign_intent_sha256: str,
    expected_provision_receipt_sha256: str,
    expected_source_runtime_model_bindings_digest: str,
    expected_fixed_control_envelope_manifest_digest: str,
    expected_final_bytes_per_run: int,
    expected_maximum_live_bytes_per_run: int,
    expected_final_inodes_per_run: int,
    expected_maximum_live_inodes_per_run: int,
    expected_bytes_written_per_run: int,
    expected_completion_file_count: int,
    price_micro_usd_per_gpu_hour: int,
    operator_maximum_wall_seconds: int,
    operator_maximum_gpu_seconds: int,
    operator_maximum_cost_micro_usd: int,
    candidate_maximum_wall_seconds: int,
    candidate_maximum_gpu_seconds: int,
    candidate_maximum_cost_micro_usd: int,
    fixed_bytes: int,
    fixed_inodes: int,
    available_free_bytes: int,
    available_free_inodes: int,
) -> QualificationReport:
    """Validate all 16 receipts and derive every feasibility input.

    The expected identities and limits must eventually come from externally
    registered compute-freeze, campaign-intent, and provision receipts.  This
    pure source milestone validates their cross-bindings but does not create or
    authenticate those future artifacts.
    """

    schedule = build_engineering_schedule(worker_count)
    if type(receipts) not in {tuple, list} or len(receipts) != ENGINEERING_RUN_COUNT:
        raise QualificationError("campaign must contain exactly 16 ordered run receipts")
    if type(sorted_gpu_uuids) not in {tuple, list} or len(sorted_gpu_uuids) != worker_count:
        raise QualificationError("campaign must bind one GPU UUID per worker")
    gpu_uuids = tuple(_gpu_uuid(item, "sorted_gpu_uuids") for item in sorted_gpu_uuids)
    if tuple(sorted(gpu_uuids)) != gpu_uuids or len(set(gpu_uuids)) != worker_count:
        raise QualificationError("GPU UUIDs must be distinct and lexically sorted")

    campaign_uuid = _uuid4(expected_campaign_uuid, "expected_campaign_uuid")
    qualification_uuid = _uuid4(expected_qualification_uuid, "expected_qualification_uuid")
    if campaign_uuid == qualification_uuid:
        raise QualificationError("campaign and qualification UUIDs must differ")
    candidate_id = _candidate_id(expected_candidate_id, "expected_candidate_id")
    priority = _exact_int(candidate_priority, "candidate_priority")
    if priority >= len(_CANDIDATE_PROFILES):
        raise QualificationError("candidate_priority is outside the exact profile order")
    profile_id, expected_gpu_id, expected_workers, expected_ceiling = _CANDIDATE_PROFILES[priority]
    _exact_text(candidate_profile_id, profile_id, "candidate_profile_id")
    _exact_text(gpu_id, expected_gpu_id, "gpu_id")
    if worker_count != expected_workers:
        raise QualificationError("worker_count changed from the selected candidate profile")
    _exact_text(cloud_type, "SECURE", "cloud_type")
    _exact_text(network_volume_type, "HIGH_PERFORMANCE", "network_volume_type")
    _exact_text(network_volume_mount, "/workspace", "network_volume_mount")
    data_center_id = _data_center_id(data_center_id, "data_center_id")
    network_volume_id = _network_volume_id(network_volume_id, "network_volume_id")
    ceiling = _exact_int(
        qualification_ceiling_seconds,
        "qualification_ceiling_seconds",
        minimum=1,
    )
    if ceiling != expected_ceiling:
        raise QualificationError("qualification ceiling changed from the exact candidate profile")
    if (
        type(prior_candidate_dispositions) not in {tuple, list}
        or len(prior_candidate_dispositions) != priority
    ):
        raise QualificationError("all earlier candidate profiles must have exact dispositions")
    disposition_fields = frozenset({"profile_id", "state", "registrar_record_sha256"})
    for index, disposition_value in enumerate(prior_candidate_dispositions):
        disposition = _exact_object(
            disposition_value,
            disposition_fields,
            f"prior_candidate_dispositions[{index}]",
        )
        _exact_text(
            disposition["profile_id"],
            _CANDIDATE_PROFILES[index][0],
            f"prior_candidate_dispositions[{index}].profile_id",
        )
        if type(disposition["state"]) is not str or disposition["state"] not in {
            "consumed_failure",
            "skipped_no_stock",
        }:
            raise QualificationError("prior candidate disposition state changed")
        _sha256(
            disposition["registrar_record_sha256"],
            f"prior_candidate_dispositions[{index}].registrar_record_sha256",
        )
    prior_dispositions_digest = digest(prior_candidate_dispositions)
    expected_hashes = {
        "compute_freeze_sha256": _sha256(
            expected_compute_freeze_sha256,
            "expected_compute_freeze_sha256",
        ),
        "campaign_intent_sha256": _sha256(
            expected_campaign_intent_sha256,
            "expected_campaign_intent_sha256",
        ),
        "provision_receipt_sha256": _sha256(
            expected_provision_receipt_sha256,
            "expected_provision_receipt_sha256",
        ),
        "source_runtime_model_bindings_digest": _sha256(
            expected_source_runtime_model_bindings_digest,
            "expected_source_runtime_model_bindings_digest",
        ),
        "fixed_control_envelope_manifest_digest": _sha256(
            expected_fixed_control_envelope_manifest_digest,
            "expected_fixed_control_envelope_manifest_digest",
        ),
    }

    validated = tuple(validate_run_receipt(receipt) for receipt in receipts)
    if tuple(receipt["row_index"] for receipt in validated) != tuple(range(ENGINEERING_RUN_COUNT)):
        raise QualificationError("campaign receipts must be in exact schedule-row order")
    for row, receipt in zip(schedule.rows, validated, strict=True):
        expected_identity: dict[str, Any] = {
            "worker_count": worker_count,
            "schedule_digest": schedule.schedule_digest,
            "campaign_uuid": campaign_uuid,
            "qualification_uuid": qualification_uuid,
            "candidate_id": candidate_id,
            "candidate_profile_id": profile_id,
            "candidate_priority": priority,
            "candidate_policy_digest": CANDIDATE_POLICY_DIGEST,
            "prior_candidate_dispositions_digest": prior_dispositions_digest,
            "gpu_id": expected_gpu_id,
            "cloud_type": "SECURE",
            "data_center_id": data_center_id,
            "network_volume_id": network_volume_id,
            "network_volume_type": "HIGH_PERFORMANCE",
            "network_volume_mount": "/workspace",
            "qualification_ceiling_seconds": ceiling,
            "gpu_uuid": gpu_uuids[row.worker_index],
            **expected_hashes,
        }
        for field, expected in expected_identity.items():
            if type(receipt[field]) is not type(expected) or receipt[field] != expected:
                raise QualificationError(f"campaign receipt {field} cross-binding changed")
    fixed_envelope_body = {
        "final_bytes": _exact_int(
            expected_final_bytes_per_run,
            "expected_final_bytes_per_run",
            minimum=1,
        ),
        "maximum_live_bytes": _exact_int(
            expected_maximum_live_bytes_per_run,
            "expected_maximum_live_bytes_per_run",
            minimum=1,
        ),
        "final_inodes": _exact_int(
            expected_final_inodes_per_run,
            "expected_final_inodes_per_run",
            minimum=1,
        ),
        "maximum_live_inodes": _exact_int(
            expected_maximum_live_inodes_per_run,
            "expected_maximum_live_inodes_per_run",
            minimum=1,
        ),
        "bytes_written": _exact_int(
            expected_bytes_written_per_run,
            "expected_bytes_written_per_run",
            minimum=1,
        ),
        "completion_file_count": _exact_int(
            expected_completion_file_count,
            "expected_completion_file_count",
            minimum=1,
        ),
    }
    if (
        fixed_envelope_body["maximum_live_bytes"] < fixed_envelope_body["final_bytes"]
        or fixed_envelope_body["maximum_live_inodes"] < fixed_envelope_body["final_inodes"]
        or fixed_envelope_body["bytes_written"] < fixed_envelope_body["final_bytes"]
    ):
        raise QualificationError("fixed control envelope is internally inconsistent")
    fixed_envelope: Mapping[str, int] = MappingProxyType(fixed_envelope_body)
    for receipt in validated:
        for field, expected in fixed_envelope.items():
            if type(receipt[field]) is not int or receipt[field] != expected:
                raise QualificationError(f"receipt {field} changed from the fixed control envelope")

    pair_walls: list[int] = []
    for pair_index in range(ENGINEERING_PAIR_COUNT):
        first, second = (receipt for receipt in validated if receipt["pair_index"] == pair_index)
        if second["started_monotonic_ns"] != first["completed_monotonic_ns"]:
            raise QualificationError("a qualification pair must use one exact gapless algorithm handoff")
        pair_walls.append(
            _ceil_div(
                second["completed_monotonic_ns"] - first["wave_release_monotonic_ns"],
                1_000_000_000,
            )
        )
    for worker_index in range(worker_count):
        worker_receipts = [receipt for receipt in validated if receipt["worker_index"] == worker_index]
        for previous, following in pairwise(worker_receipts):
            if following["started_monotonic_ns"] < previous["completed_monotonic_ns"]:
                raise QualificationError("one GPU worker has overlapping qualification runs")
    for wave_start in range(worker_count, ENGINEERING_PAIR_COUNT, worker_count):
        prior_wave = [
            receipt
            for receipt in validated
            if wave_start - worker_count <= receipt["pair_index"] < wave_start
        ]
        next_wave = [
            receipt
            for receipt in validated
            if wave_start <= receipt["pair_index"] < wave_start + worker_count
        ]
        if next_wave and min(receipt["started_monotonic_ns"] for receipt in next_wave) < max(
            receipt["completed_monotonic_ns"] for receipt in prior_wave
        ):
            raise QualificationError("qualification workers did not rendezvous before the next wave")
    for wave_start in range(0, ENGINEERING_PAIR_COUNT, worker_count):
        wave_pair_indices = range(
            wave_start,
            min(wave_start + worker_count, ENGINEERING_PAIR_COUNT),
        )
        pair_intervals = []
        wave_receipts = [
            receipt
            for receipt in validated
            if wave_start <= receipt["pair_index"] < wave_start + worker_count
        ]
        release_values = {receipt["wave_release_monotonic_ns"] for receipt in wave_receipts}
        if len(release_values) != 1:
            raise QualificationError("all rows in one wave must bind one common release")
        release = next(iter(release_values))
        for pair_index in wave_pair_indices:
            pair_receipts = [receipt for receipt in validated if receipt["pair_index"] == pair_index]
            pair_intervals.append(
                (
                    pair_receipts[0]["started_monotonic_ns"],
                    pair_receipts[-1]["completed_monotonic_ns"],
                )
            )
        if (
            len(pair_intervals) != worker_count
            or max(start for start, _end in pair_intervals) - release > 1_000_000_000
            or min(end for _start, end in pair_intervals) <= max(start for start, _end in pair_intervals)
        ):
            raise QualificationError(
                "qualification wave must start within one second of its common release "
                "and overlap at W-way load"
            )
    maximum_pair_wall_seconds = max(pair_walls)
    qualification_wall_seconds = _ceil_div(
        max(receipt["completed_monotonic_ns"] for receipt in validated)
        - min(receipt["wave_release_monotonic_ns"] for receipt in validated),
        1_000_000_000,
    )

    totals_by_worker: dict[int, int] = {}
    for receipt in validated:
        worker_index = receipt["worker_index"]
        total = receipt["total_gpu_bytes"]
        prior = totals_by_worker.setdefault(worker_index, total)
        if prior != total:
            raise QualificationError("one worker reported inconsistent GPU capacity")
    if len(set(totals_by_worker.values())) != 1:
        raise QualificationError("qualification candidate GPUs must be homogeneous")
    memory_ok = all(
        memory_gate(
            total_gpu_bytes=receipt["total_gpu_bytes"],
            maximum_reserved_gpu_bytes=receipt["maximum_reserved_gpu_bytes"],
        )
        for receipt in validated
    )

    compute = project_compute(
        worker_count=worker_count,
        maximum_pair_wall_seconds=maximum_pair_wall_seconds,
        price_micro_usd_per_gpu_hour=price_micro_usd_per_gpu_hour,
        operator_maximum_wall_seconds=operator_maximum_wall_seconds,
        operator_maximum_gpu_seconds=operator_maximum_gpu_seconds,
        operator_maximum_cost_micro_usd=operator_maximum_cost_micro_usd,
        candidate_maximum_wall_seconds=candidate_maximum_wall_seconds,
        candidate_maximum_gpu_seconds=candidate_maximum_gpu_seconds,
        candidate_maximum_cost_micro_usd=candidate_maximum_cost_micro_usd,
    )
    maximum_final_bytes = fixed_envelope["final_bytes"]
    maximum_transient_bytes = fixed_envelope["maximum_live_bytes"] - maximum_final_bytes
    maximum_final_inodes = fixed_envelope["final_inodes"]
    maximum_transient_inodes = fixed_envelope["maximum_live_inodes"] - maximum_final_inodes
    storage = project_storage(
        worker_count=worker_count,
        final_bytes_per_run=maximum_final_bytes,
        maximum_live_bytes_per_run=maximum_final_bytes + maximum_transient_bytes,
        fixed_bytes=_exact_int(fixed_bytes, "fixed_bytes"),
        final_inodes_per_run=maximum_final_inodes,
        maximum_live_inodes_per_run=maximum_final_inodes + maximum_transient_inodes,
        fixed_inodes=_exact_int(fixed_inodes, "fixed_inodes"),
        maximum_bytes_written_per_run=fixed_envelope["bytes_written"],
    )
    free_bytes = _exact_int(available_free_bytes, "available_free_bytes")
    free_inodes = _exact_int(available_free_inodes, "available_free_inodes")
    gates = {
        **{f"compute_{key}": child for key, child in compute.gates.items()},
        "memory": memory_ok,
        "storage_free_bytes": free_bytes >= storage.required_free_bytes,
        "storage_free_inodes": free_inodes >= storage.required_free_inodes,
        "qualification_ceiling": qualification_wall_seconds <= ceiling,
    }
    supplied_evidence_gates_satisfied = all(gates.values())
    run_receipts_digest = digest([receipt["receipt_digest"] for receipt in validated])
    input_bindings: dict[str, Any] = {
        "compute_freeze_sha256": expected_hashes["compute_freeze_sha256"],
        "campaign_intent_sha256": expected_hashes["campaign_intent_sha256"],
        "provision_receipt_sha256": expected_hashes["provision_receipt_sha256"],
        "source_runtime_model_bindings_digest": expected_hashes["source_runtime_model_bindings_digest"],
        "fixed_control_envelope_manifest_digest": expected_hashes["fixed_control_envelope_manifest_digest"],
        "sorted_gpu_uuids": gpu_uuids,
        "price_micro_usd_per_gpu_hour": _exact_int(
            price_micro_usd_per_gpu_hour,
            "price_micro_usd_per_gpu_hour",
            minimum=1,
        ),
        "operator_maximum_wall_seconds": _exact_int(
            operator_maximum_wall_seconds,
            "operator_maximum_wall_seconds",
            minimum=1,
        ),
        "operator_maximum_gpu_seconds": _exact_int(
            operator_maximum_gpu_seconds,
            "operator_maximum_gpu_seconds",
            minimum=1,
        ),
        "operator_maximum_cost_micro_usd": _exact_int(
            operator_maximum_cost_micro_usd,
            "operator_maximum_cost_micro_usd",
            minimum=1,
        ),
        "candidate_maximum_wall_seconds": _exact_int(
            candidate_maximum_wall_seconds,
            "candidate_maximum_wall_seconds",
            minimum=1,
        ),
        "candidate_maximum_gpu_seconds": _exact_int(
            candidate_maximum_gpu_seconds,
            "candidate_maximum_gpu_seconds",
            minimum=1,
        ),
        "candidate_maximum_cost_micro_usd": _exact_int(
            candidate_maximum_cost_micro_usd,
            "candidate_maximum_cost_micro_usd",
            minimum=1,
        ),
        "fixed_bytes": _exact_int(fixed_bytes, "fixed_bytes"),
        "fixed_inodes": _exact_int(fixed_inodes, "fixed_inodes"),
        "available_free_bytes": free_bytes,
        "available_free_inodes": free_inodes,
        "fixed_control_envelope": fixed_envelope,
    }
    total_gpu_bytes = next(iter(totals_by_worker.values()))
    minimum_gpu_headroom_bytes = max(8 * GIB, _ceil_div(total_gpu_bytes, 10))
    body = {
        "schema": REPORT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "worker_count": worker_count,
        "schedule_digest": schedule.schedule_digest,
        "campaign_uuid": campaign_uuid,
        "qualification_uuid": qualification_uuid,
        "candidate_id": candidate_id,
        "candidate_profile_id": profile_id,
        "candidate_priority": priority,
        "candidate_policy_digest": CANDIDATE_POLICY_DIGEST,
        "prior_candidate_dispositions_digest": prior_dispositions_digest,
        "gpu_id": expected_gpu_id,
        "cloud_type": "SECURE",
        "data_center_id": data_center_id,
        "network_volume_id": network_volume_id,
        "network_volume_type": "HIGH_PERFORMANCE",
        "network_volume_mount": "/workspace",
        "qualification_ceiling_seconds": ceiling,
        "run_receipts_digest": run_receipts_digest,
        "input_bindings": input_bindings,
        "pair_count": ENGINEERING_PAIR_COUNT,
        "run_count": ENGINEERING_RUN_COUNT,
        "maximum_pair_wall_seconds": maximum_pair_wall_seconds,
        "maximum_reserved_gpu_bytes": max(receipt["maximum_reserved_gpu_bytes"] for receipt in validated),
        "minimum_gpu_headroom_bytes": minimum_gpu_headroom_bytes,
        "maximum_peak_rss_bytes": max(receipt["maximum_peak_rss_bytes"] for receipt in validated),
        "maximum_live_bytes_per_run": max(receipt["maximum_live_bytes"] for receipt in validated),
        "maximum_final_bytes_per_run": max(receipt["final_bytes"] for receipt in validated),
        "maximum_live_inodes_per_run": max(receipt["maximum_live_inodes"] for receipt in validated),
        "maximum_final_inodes_per_run": max(receipt["final_inodes"] for receipt in validated),
        "maximum_bytes_written_per_run": max(receipt["bytes_written"] for receipt in validated),
        "required_free_bytes": storage.required_free_bytes,
        "required_free_inodes": storage.required_free_inodes,
        "projected_aggregate_write_bytes": storage.projected_aggregate_write_bytes,
        "raw_projected_study_wall_seconds": compute.raw_projected_study_wall_seconds,
        "qualified_projected_study_wall_seconds": (compute.qualified_projected_study_wall_seconds),
        "selected_deadline_seconds": compute.selected_deadline_seconds,
        "projected_study_gpu_seconds": compute.projected_study_gpu_seconds,
        "projected_cost_micro_usd": compute.projected_cost_micro_usd,
        "gates": gates,
        "supplied_evidence_gates_satisfied": supplied_evidence_gates_satisfied,
        "evidence_externally_authenticated": False,
        "same_final_pod_verified": False,
        "compute_route_qualified": False,
        "g01_launch_authorized": False,
    }
    return QualificationReport(
        worker_count=worker_count,
        schedule_digest=schedule.schedule_digest,
        campaign_uuid=campaign_uuid,
        qualification_uuid=qualification_uuid,
        candidate_id=candidate_id,
        candidate_profile_id=profile_id,
        candidate_priority=priority,
        candidate_policy_digest=CANDIDATE_POLICY_DIGEST,
        prior_candidate_dispositions_digest=prior_dispositions_digest,
        gpu_id=expected_gpu_id,
        cloud_type="SECURE",
        data_center_id=data_center_id,
        network_volume_id=network_volume_id,
        network_volume_type="HIGH_PERFORMANCE",
        network_volume_mount="/workspace",
        qualification_ceiling_seconds=ceiling,
        run_receipts_digest=run_receipts_digest,
        input_bindings=MappingProxyType(input_bindings),
        maximum_pair_wall_seconds=maximum_pair_wall_seconds,
        maximum_reserved_gpu_bytes=body["maximum_reserved_gpu_bytes"],
        minimum_gpu_headroom_bytes=minimum_gpu_headroom_bytes,
        maximum_peak_rss_bytes=body["maximum_peak_rss_bytes"],
        maximum_live_bytes_per_run=body["maximum_live_bytes_per_run"],
        maximum_final_bytes_per_run=body["maximum_final_bytes_per_run"],
        maximum_live_inodes_per_run=body["maximum_live_inodes_per_run"],
        maximum_final_inodes_per_run=body["maximum_final_inodes_per_run"],
        maximum_bytes_written_per_run=body["maximum_bytes_written_per_run"],
        required_free_bytes=storage.required_free_bytes,
        required_free_inodes=storage.required_free_inodes,
        projected_aggregate_write_bytes=storage.projected_aggregate_write_bytes,
        raw_projected_study_wall_seconds=compute.raw_projected_study_wall_seconds,
        qualified_projected_study_wall_seconds=compute.qualified_projected_study_wall_seconds,
        selected_deadline_seconds=compute.selected_deadline_seconds,
        projected_study_gpu_seconds=compute.projected_study_gpu_seconds,
        projected_cost_micro_usd=compute.projected_cost_micro_usd,
        gates=MappingProxyType(gates),
        supplied_evidence_gates_satisfied=supplied_evidence_gates_satisfied,
        report_digest=digest(body),
    )


def validate_report_bytes(
    payload: bytes,
    *,
    expected_report: QualificationReport,
) -> dict[str, Any]:
    """Replay one canonical report against a freshly derived expected report."""

    if type(expected_report) is not QualificationReport:
        raise QualificationError("expected_report must be one exact QualificationReport")
    value = strict_canonical_json_bytes(payload, "G01Q report")
    expected = expected_report.as_dict()
    _exact_equal(value, expected, "G01Q report")
    body = {key: child for key, child in value.items() if key != "report_digest"}
    _exact_text(value["report_digest"], digest(body), "report_digest")
    for field in (
        "evidence_externally_authenticated",
        "same_final_pod_verified",
        "compute_route_qualified",
        "g01_launch_authorized",
    ):
        _exact_bool(value[field], False, field)
    return value


def build_review_contract() -> dict[str, Any]:
    """Return the exact nonauthorizing source-review contract."""

    body = {
        "schema": REVIEW_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "target": _plain(G01_TARGET),
        "production_shape": _plain(PRODUCTION_SHAPE),
        "production_shape_digest": PRODUCTION_SHAPE_DIGEST,
        "engineering": {
            "pair_count": ENGINEERING_PAIR_COUNT,
            "run_count": ENGINEERING_RUN_COUNT,
            "seeds": list(ENGINEERING_SEEDS),
            "cells": [{"rule_family": family, "q_p_basis_points": q_p} for family, q_p in ENGINEERING_CELLS],
            "worker_counts": list(WORKER_COUNTS),
            "schedule_digests": {
                str(worker_count): build_engineering_schedule(worker_count).schedule_digest
                for worker_count in WORKER_COUNTS
            },
        },
        "candidate_policy": {
            "ordered_profiles": [
                {
                    "profile_id": profile_id,
                    "gpu_id": gpu_id,
                    "gpu_count": gpu_count,
                    "qualification_ceiling_seconds": ceiling,
                }
                for profile_id, gpu_id, gpu_count, ceiling in _CANDIDATE_PROFILES
            ],
            "cloud_type": "SECURE",
            "one_homogeneous_single_pod": True,
            "one_process_per_gpu": True,
            "mixed_gpu_forbidden": True,
            "multi_pod_forbidden": True,
            "pool_fallback_forbidden": True,
            "network_volume_type": "HIGH_PERFORMANCE",
            "network_volume_mount": "/workspace",
            "same_data_center_required": True,
            "single_pod_mount_required": True,
            "campaign_must_preregister_concrete_dc_volume_tuple": True,
            "metric_prediction_stream_must_be_fixed_padded_envelope": True,
            "model_weight_serialization_must_be_uncompressed": True,
            "raw_scientific_tree_must_be_absent_before_control_handoff": True,
        },
        "projection": {
            "scientific_pairs": SCIENTIFIC_PAIR_COUNT,
            "wall_multiplier_numerator": 135,
            "wall_multiplier_denominator": 100,
            "terminal_reserve_seconds": 7_200,
            "deadline_menu_seconds": list(DEADLINE_MENU_SECONDS),
            "storage_multiplier_numerator": 5,
            "storage_multiplier_denominator": 4,
            "minimum_free_bytes": TIB,
            "minimum_free_inodes": 1_000_000,
            "minimum_gpu_headroom_bytes": 8 * GIB,
            "minimum_gpu_headroom_fraction_numerator": 1,
            "minimum_gpu_headroom_fraction_denominator": 10,
        },
        "authorization": {
            "qualification_only": True,
            "g01_scientific_outcomes_seen": False,
            "g01_itt_created": False,
            "g01_training_authorized": False,
            "checkpoint_b_launch_authorized": False,
            "runtime_provision_frozen": False,
        },
        "missing_prerequisites": [
            "checkpoint_a_token",
            "durable_external_registrar",
            "operator_cost_and_wall_caps",
            "runtime_native_and_model_freeze",
            "same_final_pod_provision_receipt",
            "checkpoint_b_lifecycle_audit",
            "launch_grant",
        ],
    }
    return {**body, "contract_digest": digest(body)}


def validate_review_contract_bytes(payload: bytes) -> dict[str, Any]:
    value = strict_canonical_json_bytes(payload, "G01Q source review contract")
    expected = build_review_contract()
    _exact_equal(value, expected, "G01Q source review contract")
    return value


def run_qualification(*_args: Any, **_kwargs: Any) -> NoReturn:
    raise QualificationError(REFUSAL)


def execute_qualification(*_args: Any, **_kwargs: Any) -> NoReturn:
    raise QualificationError(REFUSAL)
