"""Fail-closed prepare/execute lifecycle for the G03-v2 evaluation census.

This module adds durable engineering controls around :mod:`evaluation_census`.
It does not implement a second census generator.  A preparation invocation
materializes an outcome-free plan and freeze request.  A later execution
invocation requires an externally supplied, content-pinned registration
receipt before it starts two fresh Python processes: one constructs the report
and the other rederives and verifies it.

Every artifact remains nonauthorizing.  In particular, validating the local
shape and content pins of a registration receipt is not independent evidence
that the named registration service exists, issued the receipt, or establishes
temporal priority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import resource
import select
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, NoReturn, cast

from .evaluation_census import (
    DEFAULT_CANDIDATE_POOL_SIZE,
    PRODUCTION_MIRROR_ATTEMPTS_PER_STRATUM,
    EvaluationCensusPlanV1,
    EvaluationCensusReportV1,
    EvaluationCensusV2Error,
    build_evaluation_census_plan_v1,
    build_evaluation_census_report_v1,
    parse_evaluation_census_plan_v1,
    parse_evaluation_census_report_v1,
    serialize_evaluation_census_plan_v1,
    serialize_evaluation_census_report_v1,
)

EVALUATION_CENSUS_EXECUTION_SCHEMA_VERSION = 1

PLAN_FILENAME = "prospective-plan.json"
FREEZE_REQUEST_FILENAME = "freeze-request.json"
STARTED_RECEIPT_FILENAME = "execution-started.json"
BUILDER_TERMINAL_FILENAME = "builder-terminal.json"
VERIFIER_TERMINAL_FILENAME = "verifier-terminal.json"
BUILDER_GUARDIAN_TERMINAL_FILENAME = "builder-guardian-terminal.json"
VERIFIER_GUARDIAN_TERMINAL_FILENAME = "verifier-guardian-terminal.json"
REPORT_FILENAME = "observed-report.json"
EXECUTION_RECEIPT_FILENAME = "execution-receipt.json"
FAILURE_RECEIPT_FILENAME = "execution-failure.json"

_FREEZE_REQUEST_KIND = "g03-v2-evaluation-census-freeze-request-v1"
_EXTERNAL_REGISTRATION_KIND = "g03-v2-evaluation-census-external-registration-receipt-v1"
_STARTED_KIND = "g03-v2-evaluation-census-execution-started-v1"
_BUILDER_TERMINAL_KIND = "g03-v2-evaluation-census-builder-terminal-v1"
_VERIFIER_TERMINAL_KIND = "g03-v2-evaluation-census-verifier-terminal-v1"
_EXECUTION_RECEIPT_KIND = "g03-v2-evaluation-census-execution-receipt-v1"
_FAILURE_RECEIPT_KIND = "g03-v2-evaluation-census-execution-failure-v1"
_WATCHDOG_TERMINAL_KIND = "g03-v2-evaluation-census-watchdog-terminal-v1"

_FREEZE_REQUEST_DOMAIN = "goalzendo-interactive-v2-evaluation-census-freeze-request-v1"
_EXTERNAL_REGISTRATION_DOMAIN = "goalzendo-interactive-v2-evaluation-census-external-registration-receipt-v1"
_STARTED_DOMAIN = "goalzendo-interactive-v2-evaluation-census-execution-started-v1"
_WORKER_TERMINAL_DOMAIN = "goalzendo-interactive-v2-evaluation-census-worker-terminal-v1"
_WATCHDOG_TERMINAL_DOMAIN = "goalzendo-interactive-v2-evaluation-census-watchdog-terminal-v1"
_EXECUTION_RECEIPT_DOMAIN = "goalzendo-interactive-v2-evaluation-census-execution-receipt-v1"
_FAILURE_RECEIPT_DOMAIN = "goalzendo-interactive-v2-evaluation-census-execution-failure-v1"
_CELL_COORDINATE_DOMAIN = "goalzendo-interactive-v2-evaluation-census-cell-coordinates-v1"

_RUNNER_RELATIVE_PATH = "scripts/run_g03_v2_evaluation_census.py"
_MODULE_RELATIVE_PATH = "src/goalzendo_interactive_v2/evaluation_census_execution.py"
_PlanClass = Literal["production_9x16_pool32", "engineering_test_fixture"]
_PRODUCTION_PLAN_CLASS: _PlanClass = "production_9x16_pool32"
_ENGINEERING_PLAN_CLASS: _PlanClass = "engineering_test_fixture"

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}")
_UTC_TIMESTAMP = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}")
_NONCE = re.compile(r"[0-9a-f]{64}")

_AUTHORIZATION: dict[str, str | bool] = {
    "scope": "cpu_only_evaluation_census_artifact_lifecycle",
    "g01_authorized": False,
    "g03_capability_launch_authorized": False,
    "g03_scientific_launch_authorized": False,
    "model_execution_authorized": False,
    "weight_updates_authorized": False,
}

_REGISTRATION_CLAIM_BOUNDARY: dict[str, bool] = {
    "receipt_shape_and_content_pins_validated": True,
    "registration_service_identity_independently_verified_by_repository": False,
    "receipt_issuance_independently_verified_by_repository": False,
    "temporal_priority_independently_verified_by_repository": False,
    "external_timestamp_independently_verified_by_repository": False,
}

_SCIENTIFIC_CLAIM_BOUNDARY: dict[str, bool] = {
    "positive_m_q_quota_selected": False,
    "matched_bank_size_selected": False,
    "difficulty_matcher_run": False,
    "evaluation_quartets_materialized": False,
    "nested_opening_unions_materialized": False,
    "challenge_panels_materialized": False,
    "render_balance_audited": False,
    "model_outcomes_present": False,
    "g01_authorized": False,
    "launch_authorized": False,
}

_EXECUTION_POLICY: dict[str, int | str | bool] = {
    "worker_process_count": 2,
    "watchdog_process_count": 2,
    "worker_descendant_process_creation_allowed": False,
    "fresh_process_verification_required": True,
    "shared_in_process_census_cache_allowed": False,
    "retry_within_execution_uuid_allowed": False,
    "cross_parent_or_cross_host_one_shot_enforced_by_repository": False,
    "early_stop_allowed": False,
    "interrupted_output_root_reusable": False,
    "parent_death_guard": (
        "controller_to_watchdog_and_watchdog_to_worker_pipes_plus_external_"
        "guardian_ready_ACK_worker_lifetime_EOF_watchdog_TERM_5s_KILL_reap_"
        "and_worker_self_group_SIGKILL_on_watchdog_EOF"
    ),
    "controller_signal_teardown": (
        "TERM_INT_HUP_close_watchdog_guard_wait_reap_and_exact_worker_lifetime_EOF"
    ),
    "controller_deadline_guard": "ITIMER_REAL_plus_monotonic_post_fsync_fail_closed_check",
    "report_commit_method": "same_filesystem_exclusive_hard_link_after_verification",
    "completion_receipt_written_last": True,
}

_PROSPECTIVE_FORBIDDEN_KEYS = frozenset(
    {
        "attempt_accounting",
        "attempts_observed",
        "builder_terminal",
        "cell_summary",
        "constructed_opening_count",
        "disposition",
        "execution_receipt",
        "failure",
        "failures",
        "model_output",
        "observed",
        "observed_m_q_summary",
        "observed_report_digest",
        "openings",
        "preopening_failure_count",
        "ranking_tables",
        "reason_codes",
        "report_binding",
        "result",
        "results",
        "runtime_evidence",
        "scene_indices",
        "verifier_terminal",
    }
)

_EXPECTED_CELL_COORDINATES = tuple((m, q) for m in range(8, 17) for q in range(1, 5))


class EvaluationCensusExecutionV1Error(ValueError):
    """Raised when an execution artifact or lifecycle transition is unsafe."""


def _dump_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise EvaluationCensusExecutionV1Error(f"value is not canonical JSON: {exc}") from exc


def _canonical_bytes(value: Any) -> bytes:
    return (_dump_json(value) + "\n").encode("ascii")


def _load_json(text: str) -> Any:
    if type(text) is not str or not text:
        raise EvaluationCensusExecutionV1Error("JSON input must be nonempty text")

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise EvaluationCensusExecutionV1Error(f"duplicate JSON object key: {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> NoReturn:
        raise EvaluationCensusExecutionV1Error(f"non-finite JSON constant is forbidden: {value}")

    try:
        return json.loads(text, object_pairs_hook=no_duplicates, parse_constant=reject_constant)
    except EvaluationCensusExecutionV1Error:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise EvaluationCensusExecutionV1Error(f"invalid JSON: {exc}") from exc


def _digest(value: Any, *, domain: str) -> str:
    return hashlib.sha256(domain.encode("ascii") + b"\0" + _dump_json(value).encode("ascii")).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: object, *, name: str) -> str:
    if not _is_sha256(value):
        raise EvaluationCensusExecutionV1Error(f"{name} must be a lowercase SHA-256")
    return cast(str, value)


def _require_integer(
    value: object,
    *,
    name: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise EvaluationCensusExecutionV1Error(f"{name} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise EvaluationCensusExecutionV1Error(f"{name} must be an integer <= {maximum}")
    return value


def _require_boolean(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise EvaluationCensusExecutionV1Error(f"{name} must be a Boolean")
    return value


def _require_mapping(
    value: object,
    fields: tuple[str, ...],
    *,
    name: str,
) -> Mapping[str, Any]:
    if type(value) is not dict or tuple(value) != fields:
        raise EvaluationCensusExecutionV1Error(
            f"{name} has noncanonical, missing, extra, or reordered fields"
        )
    return cast(Mapping[str, Any], value)


def _require_identifier(value: object, *, name: str) -> str:
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        raise EvaluationCensusExecutionV1Error(f"{name} is not a canonical identifier")
    return value


def _require_timestamp(value: object) -> str:
    if type(value) is not str or _UTC_TIMESTAMP.fullmatch(value) is None:
        raise EvaluationCensusExecutionV1Error("registration timestamp must be YYYY-MM-DDTHH:MM:SSZ")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise EvaluationCensusExecutionV1Error("registration timestamp is invalid") from exc
    return value


def _registration_datetime(value: str) -> datetime:
    return datetime.strptime(_require_timestamp(value), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _wall_datetime(value: object, *, name: str) -> datetime:
    if type(value) is not str or not value.endswith("Z"):
        raise EvaluationCensusExecutionV1Error(f"{name} must be an exact UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise EvaluationCensusExecutionV1Error(f"{name} is invalid") from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise EvaluationCensusExecutionV1Error(f"{name} must use UTC")
    return parsed


def _require_uuid(value: object) -> str:
    if type(value) is not str or _UUID.fullmatch(value) is None:
        raise EvaluationCensusExecutionV1Error("execution UUID is not canonical UUIDv4 text")
    return value


def _require_nonce(value: object) -> str:
    if type(value) is not str or _NONCE.fullmatch(value) is None:
        raise EvaluationCensusExecutionV1Error("execution nonce must be 64 lowercase hex")
    return value


def _require_exact_constant(value: object, expected: Mapping[str, Any], *, name: str) -> None:
    obj = _require_mapping(value, tuple(expected), name=name)
    if dict(obj) != dict(expected):
        raise EvaluationCensusExecutionV1Error(f"{name} differs from the frozen constant")


def _reject_prospective_outcomes(value: object, *, path: str = "freeze_request") -> None:
    if type(value) is dict:
        for key, child in cast(dict[str, Any], value).items():
            normalized = key.lower().replace("-", "_")
            if normalized in _PROSPECTIVE_FORBIDDEN_KEYS or normalized.startswith("observed_"):
                raise EvaluationCensusExecutionV1Error(
                    f"prospective freeze request contains forbidden outcome field at {path}.{key}"
                )
            _reject_prospective_outcomes(child, path=f"{path}.{key}")
    elif type(value) is list:
        for position, child in enumerate(value):
            _reject_prospective_outcomes(child, path=f"{path}[{position}]")


def _repository_root() -> Path:
    path = Path(__file__)
    if path.is_symlink() or not path.is_file():
        raise EvaluationCensusExecutionV1Error("execution module source must be one ordinary file")
    root = path.resolve().parents[2]
    if not (root / _RUNNER_RELATIVE_PATH).is_file():
        raise EvaluationCensusExecutionV1Error("canonical census runner is missing")
    return root


def canonical_evaluation_census_runner_path_v1() -> Path:
    """Return the sole runner path accepted by preparation and execution."""

    return _repository_root() / _RUNNER_RELATIVE_PATH


def _ordinary_file_bytes(path: Path, *, name: str, maximum_bytes: int | None = None) -> bytes:
    path = path.resolve(strict=True)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise EvaluationCensusExecutionV1Error(f"{name} must be one ordinary file")
        if maximum_bytes is not None and metadata.st_size > maximum_bytes:
            raise EvaluationCensusExecutionV1Error(f"{name} exceeds the maximum byte count")
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise EvaluationCensusExecutionV1Error(f"{name} changed while it was read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise EvaluationCensusExecutionV1Error(f"{name} grew while it was read")
        current = os.fstat(descriptor)
        if (metadata.st_dev, metadata.st_ino, metadata.st_size) != (
            current.st_dev,
            current.st_ino,
            current.st_size,
        ):
            raise EvaluationCensusExecutionV1Error(f"{name} changed while it was read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _source_binding(path: Path, *, relative_path: str) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    payload = _ordinary_file_bytes(resolved, name=relative_path, maximum_bytes=8 * 1024 * 1024)
    return {
        "relative_path": relative_path,
        "sha256": _sha256_bytes(payload),
        "byte_count": len(payload),
    }


def _execution_code_binding(runner_source_path: Path) -> dict[str, Any]:
    root = _repository_root()
    runner = runner_source_path.resolve(strict=True)
    expected_runner = (root / _RUNNER_RELATIVE_PATH).resolve(strict=True)
    if runner != expected_runner:
        raise EvaluationCensusExecutionV1Error(
            "runner source path must resolve to the canonical production runner"
        )
    return {
        "execution_module": _source_binding(
            root / _MODULE_RELATIVE_PATH,
            relative_path=_MODULE_RELATIVE_PATH,
        ),
        "runner": _source_binding(runner, relative_path=_RUNNER_RELATIVE_PATH),
        "python": _python_identity(),
        "worker_environment": _safe_environment(),
    }


def _optional_input_binding(path: Path | None, *, label: str) -> dict[str, Any]:
    if path is None:
        return {"label": label, "supplied": False, "sha256": None, "byte_count": None}
    payload = _ordinary_file_bytes(path, name=label)
    return {
        "label": label,
        "supplied": True,
        "sha256": _sha256_bytes(payload),
        "byte_count": len(payload),
    }


def _environment_input_bindings(
    *,
    source_archive_path: Path | None,
    constraints_path: Path | None,
    environment_lock_path: Path | None,
) -> dict[str, Any]:
    return {
        "source_archive": _optional_input_binding(source_archive_path, label="source_archive"),
        "constraints": _optional_input_binding(constraints_path, label="constraints"),
        "environment_lock": _optional_input_binding(environment_lock_path, label="environment_lock"),
    }


def _assert_production_environment_files(
    *,
    source_archive_path: Path | None,
    constraints_path: Path | None,
    environment_lock_path: Path | None,
) -> None:
    paths = {
        "source archive": source_archive_path,
        "constraints": constraints_path,
        "environment lock": environment_lock_path,
    }
    if any(path is None for path in paths.values()):
        raise EvaluationCensusExecutionV1Error(
            "production requires source archive, constraints, and environment lock files"
        )
    for label, optional_path in paths.items():
        if optional_path is None:
            raise AssertionError("production environment path narrowing failed")
        payload = _ordinary_file_bytes(optional_path, name=label)
        if not payload:
            raise EvaluationCensusExecutionV1Error(f"production {label} must be nonempty")
        if label != "source archive":
            if b"\0" in payload:
                raise EvaluationCensusExecutionV1Error(f"production {label} contains NUL bytes")
            try:
                payload.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise EvaluationCensusExecutionV1Error(f"production {label} must be UTF-8 text") from exc
            if not payload.endswith(b"\n"):
                raise EvaluationCensusExecutionV1Error(f"production {label} must be newline terminated")


def _assert_production_plan(plan: EvaluationCensusPlanV1) -> None:
    if not plan.uses_production_attempt_budget:
        raise EvaluationCensusExecutionV1Error(
            "production execution requires exactly 16 attempts per stratum and pool size 32"
        )
    if plan.attempts_per_formula_stratum != PRODUCTION_MIRROR_ATTEMPTS_PER_STRATUM:
        raise EvaluationCensusExecutionV1Error("production attempt budget changed")
    if plan.candidate_pool_size != DEFAULT_CANDIDATE_POOL_SIZE:
        raise EvaluationCensusExecutionV1Error("production candidate pool changed")
    if len(plan.attempts) != 144:
        raise EvaluationCensusExecutionV1Error("production plan must contain exactly 144 attempts")
    composed = tuple(member.composed_rule_id for attempt in plan.attempts for member in attempt.members)
    if len(composed) != 288 or len(set(composed)) != 288:
        raise EvaluationCensusExecutionV1Error(
            "production plan must contain 288 distinct evaluation C identities"
        )
    if len(plan.attempts) * 8 != 1_152:
        raise EvaluationCensusExecutionV1Error("production plan must contain exactly 1152 draws")


def _plan_binding(plan: EvaluationCensusPlanV1, plan_bytes: bytes) -> dict[str, Any]:
    return {
        "prospective_plan_digest": plan.digest,
        "exact_plan_bytes_sha256": _sha256_bytes(plan_bytes),
        "exact_plan_byte_count": len(plan_bytes),
    }


@dataclass(frozen=True, slots=True)
class EvaluationCensusFreezeRequestV1:
    """Outcome-free request to register exact prospective census bytes."""

    plan_class: _PlanClass
    plan_binding: Mapping[str, Any]
    source_binding: Mapping[str, Any]
    catalog_binding: Mapping[str, Any]
    generator_binding: Mapping[str, Any]
    execution_code_binding: Mapping[str, Any]
    environment_input_bindings: Mapping[str, Any]

    def _unsigned_obj(self) -> dict[str, Any]:
        value = {
            "schema_version": EVALUATION_CENSUS_EXECUTION_SCHEMA_VERSION,
            "request_kind": _FREEZE_REQUEST_KIND,
            "status": "prospective_outcome_free_registration_requested_nonauthorizing",
            "authorization": dict(_AUTHORIZATION),
            "plan_class": self.plan_class,
            "plan_binding": dict(self.plan_binding),
            "source_binding": dict(self.source_binding),
            "catalog_binding": dict(self.catalog_binding),
            "generator_binding": dict(self.generator_binding),
            "execution_code_binding": dict(self.execution_code_binding),
            "environment_input_bindings": dict(self.environment_input_bindings),
            "fixed_budget": {
                "formula_stratum_count": 9,
                "attempts_per_formula_stratum": (16 if self.plan_class == _PRODUCTION_PLAN_CLASS else None),
                "mirror_attempt_count": 144 if self.plan_class == _PRODUCTION_PLAN_CLASS else None,
                "draws_per_mirror_attempt": 8,
                "planned_draw_count": 1_152 if self.plan_class == _PRODUCTION_PLAN_CLASS else None,
                "candidate_pool_size": 32 if self.plan_class == _PRODUCTION_PLAN_CLASS else None,
                "distinct_evaluation_c_identity_count": (
                    288 if self.plan_class == _PRODUCTION_PLAN_CLASS else None
                ),
                "early_stop_allowed": False,
            },
            "registration_requirement": {
                "separate_external_receipt_required_before_execution": True,
                "receipt_content_pin_required": True,
                "receipt_reference_pin_required": True,
                "repository_can_independently_verify_registration_service": False,
            },
            "prospective_payload_only": True,
            "scientific_claim_boundary": dict(_SCIENTIFIC_CLAIM_BOUNDARY),
        }
        _reject_prospective_outcomes(value)
        return value

    @property
    def digest(self) -> str:
        return _digest(self._unsigned_obj(), domain=_FREEZE_REQUEST_DOMAIN)

    def as_obj(self) -> dict[str, Any]:
        return {**self._unsigned_obj(), "freeze_request_digest": self.digest}


def serialize_evaluation_census_freeze_request_v1(
    request: EvaluationCensusFreezeRequestV1,
) -> str:
    if type(request) is not EvaluationCensusFreezeRequestV1:
        raise TypeError("request must be an EvaluationCensusFreezeRequestV1")
    return _canonical_bytes(request.as_obj()).decode("ascii")


def _build_freeze_request(
    plan: EvaluationCensusPlanV1,
    plan_bytes: bytes,
    *,
    plan_class: _PlanClass,
    runner_source_path: Path,
    source_archive_path: Path | None,
    constraints_path: Path | None,
    environment_lock_path: Path | None,
) -> EvaluationCensusFreezeRequestV1:
    if plan_class == _PRODUCTION_PLAN_CLASS:
        _assert_production_plan(plan)
    elif plan.uses_production_attempt_budget:
        raise EvaluationCensusExecutionV1Error(
            "a production plan cannot be labelled as an engineering fixture"
        )
    request = EvaluationCensusFreezeRequestV1(
        plan_class,
        _plan_binding(plan, plan_bytes),
        plan.source_binding.as_obj(),
        plan.catalog_binding.as_obj(),
        plan.generator_binding.as_obj(),
        _execution_code_binding(runner_source_path),
        _environment_input_bindings(
            source_archive_path=source_archive_path,
            constraints_path=constraints_path,
            environment_lock_path=environment_lock_path,
        ),
    )
    _reject_prospective_outcomes(request.as_obj())
    return request


def parse_evaluation_census_freeze_request_v1(
    text: str,
    *,
    plan: EvaluationCensusPlanV1,
    plan_text: str,
    runner_source_path: Path,
    source_archive_path: Path | None = None,
    constraints_path: Path | None = None,
    environment_lock_path: Path | None = None,
    expected_digest: str | None = None,
) -> EvaluationCensusFreezeRequestV1:
    value = _load_json(text)
    obj = _require_mapping(
        value,
        (
            "schema_version",
            "request_kind",
            "status",
            "authorization",
            "plan_class",
            "plan_binding",
            "source_binding",
            "catalog_binding",
            "generator_binding",
            "execution_code_binding",
            "environment_input_bindings",
            "fixed_budget",
            "registration_requirement",
            "prospective_payload_only",
            "scientific_claim_boundary",
            "freeze_request_digest",
        ),
        name="freeze request",
    )
    if (
        obj["schema_version"] != EVALUATION_CENSUS_EXECUTION_SCHEMA_VERSION
        or obj["request_kind"] != _FREEZE_REQUEST_KIND
    ):
        raise EvaluationCensusExecutionV1Error("unknown freeze-request schema or kind")
    if obj["status"] != "prospective_outcome_free_registration_requested_nonauthorizing":
        raise EvaluationCensusExecutionV1Error("freeze-request status changed")
    _require_exact_constant(obj["authorization"], _AUTHORIZATION, name="authorization")
    if obj["plan_class"] not in (_PRODUCTION_PLAN_CLASS, _ENGINEERING_PLAN_CLASS):
        raise EvaluationCensusExecutionV1Error("unknown plan class")
    plan_class = cast(_PlanClass, obj["plan_class"])
    expected = _build_freeze_request(
        plan,
        plan_text.encode("ascii"),
        plan_class=plan_class,
        runner_source_path=runner_source_path,
        source_archive_path=source_archive_path,
        constraints_path=constraints_path,
        environment_lock_path=environment_lock_path,
    )
    if _dump_json(value) != _dump_json(expected.as_obj()):
        raise EvaluationCensusExecutionV1Error(
            "freeze request differs from exact plan, source, catalog, generator, or code rederivation"
        )
    if expected_digest is not None and expected.digest != _require_sha256(
        expected_digest, name="expected freeze-request digest"
    ):
        raise EvaluationCensusExecutionV1Error("freeze-request digest differs from expected")
    if serialize_evaluation_census_freeze_request_v1(expected) != text:
        raise EvaluationCensusExecutionV1Error("freeze request is not canonical newline-terminated JSON")
    return expected


@dataclass(frozen=True, slots=True)
class ExternalEvaluationCensusRegistrationReceiptV1:
    """Locally checkable shape for a receipt supplied by an external registrar."""

    registration_service: str
    registration_reference: str
    registered_at_utc: str
    execution_uuid: str
    execution_nonce: str
    execution_deadline_seconds: int
    freeze_request_digest: str
    exact_freeze_request_bytes_sha256: str
    exact_freeze_request_byte_count: int
    prospective_plan_digest: str
    exact_plan_bytes_sha256: str
    exact_plan_byte_count: int

    def _unsigned_obj(self) -> dict[str, Any]:
        return {
            "schema_version": EVALUATION_CENSUS_EXECUTION_SCHEMA_VERSION,
            "receipt_kind": _EXTERNAL_REGISTRATION_KIND,
            "registration_service": self.registration_service,
            "registration_reference": self.registration_reference,
            "registered_at_utc": self.registered_at_utc,
            "registered_execution": {
                "execution_uuid": self.execution_uuid,
                "execution_nonce": self.execution_nonce,
                "execution_deadline_seconds": self.execution_deadline_seconds,
                "attempt_identity_precommitted": True,
                "global_one_shot_consumption_enforced_by_repository": False,
            },
            "freeze_request_binding": {
                "freeze_request_digest": self.freeze_request_digest,
                "exact_freeze_request_bytes_sha256": self.exact_freeze_request_bytes_sha256,
                "exact_freeze_request_byte_count": self.exact_freeze_request_byte_count,
            },
            "prospective_plan_binding": {
                "prospective_plan_digest": self.prospective_plan_digest,
                "exact_plan_bytes_sha256": self.exact_plan_bytes_sha256,
                "exact_plan_byte_count": self.exact_plan_byte_count,
            },
            "claim_boundary": {
                "receipt_is_supplied_to_repository_as_external_input": True,
                "repository_validates_only_schema_and_content_pins": True,
                "registration_service_identity_independently_verified_by_repository": False,
                "receipt_issuance_independently_verified_by_repository": False,
                "temporal_priority_independently_verified_by_repository": False,
            },
            "authorization": dict(_AUTHORIZATION),
        }

    @property
    def digest(self) -> str:
        return _digest(self._unsigned_obj(), domain=_EXTERNAL_REGISTRATION_DOMAIN)

    def as_obj(self) -> dict[str, Any]:
        return {**self._unsigned_obj(), "registration_receipt_digest": self.digest}


def build_external_evaluation_census_registration_receipt_v1(
    freeze_request: EvaluationCensusFreezeRequestV1,
    freeze_request_text: str,
    *,
    registration_service: str,
    registration_reference: str,
    registered_at_utc: str,
    execution_uuid: str,
    execution_nonce: str,
    execution_deadline_seconds: int,
) -> ExternalEvaluationCensusRegistrationReceiptV1:
    """Build the interchange shape an external registrar must return.

    Calling this helper does not register anything and cannot authenticate a
    service.  It exists so registrars and engineering tests can emit the exact
    schema accepted by :func:`execute_production_evaluation_census_v1`.
    """

    if type(freeze_request) is not EvaluationCensusFreezeRequestV1:
        raise TypeError("freeze_request must be an EvaluationCensusFreezeRequestV1")
    if serialize_evaluation_census_freeze_request_v1(freeze_request) != freeze_request_text:
        raise EvaluationCensusExecutionV1Error("freeze-request bytes differ from the request")
    service = _require_identifier(registration_service, name="registration service")
    reference = _require_identifier(registration_reference, name="registration reference")
    timestamp = _require_timestamp(registered_at_utc)
    registered_uuid = _require_uuid(execution_uuid)
    registered_nonce = _require_nonce(execution_nonce)
    deadline = _require_integer(
        execution_deadline_seconds,
        name="execution deadline seconds",
        minimum=1,
        maximum=86_400,
    )
    binding = freeze_request.plan_binding
    return ExternalEvaluationCensusRegistrationReceiptV1(
        service,
        reference,
        timestamp,
        registered_uuid,
        registered_nonce,
        deadline,
        freeze_request.digest,
        _sha256_bytes(freeze_request_text.encode("ascii")),
        len(freeze_request_text.encode("ascii")),
        _require_sha256(binding["prospective_plan_digest"], name="plan digest"),
        _require_sha256(binding["exact_plan_bytes_sha256"], name="plan sha256"),
        _require_integer(binding["exact_plan_byte_count"], name="plan byte count", minimum=1),
    )


def serialize_external_evaluation_census_registration_receipt_v1(
    receipt: ExternalEvaluationCensusRegistrationReceiptV1,
) -> str:
    if type(receipt) is not ExternalEvaluationCensusRegistrationReceiptV1:
        raise TypeError("receipt must be an ExternalEvaluationCensusRegistrationReceiptV1")
    return _canonical_bytes(receipt.as_obj()).decode("ascii")


def parse_external_evaluation_census_registration_receipt_v1(
    text: str,
    *,
    freeze_request: EvaluationCensusFreezeRequestV1,
    freeze_request_text: str,
    expected_bytes_sha256: str,
    expected_registration_reference: str,
    expected_execution_uuid: str,
    expected_execution_nonce: str,
    expected_execution_deadline_seconds: int,
) -> ExternalEvaluationCensusRegistrationReceiptV1:
    expected_sha = _require_sha256(expected_bytes_sha256, name="expected registration-receipt sha256")
    reference = _require_identifier(expected_registration_reference, name="expected registration reference")
    execution_uuid = _require_uuid(expected_execution_uuid)
    execution_nonce = _require_nonce(expected_execution_nonce)
    execution_deadline = _require_integer(
        expected_execution_deadline_seconds,
        name="expected execution deadline seconds",
        minimum=1,
        maximum=86_400,
    )
    raw = text.encode("ascii")
    if _sha256_bytes(raw) != expected_sha:
        raise EvaluationCensusExecutionV1Error(
            "external registration-receipt bytes differ from the required SHA-256"
        )
    value = _load_json(text)
    obj = _require_mapping(
        value,
        (
            "schema_version",
            "receipt_kind",
            "registration_service",
            "registration_reference",
            "registered_at_utc",
            "registered_execution",
            "freeze_request_binding",
            "prospective_plan_binding",
            "claim_boundary",
            "authorization",
            "registration_receipt_digest",
        ),
        name="external registration receipt",
    )
    if (
        obj["schema_version"] != EVALUATION_CENSUS_EXECUTION_SCHEMA_VERSION
        or obj["receipt_kind"] != _EXTERNAL_REGISTRATION_KIND
    ):
        raise EvaluationCensusExecutionV1Error("unknown external registration-receipt schema")
    service = _require_identifier(obj["registration_service"], name="registration service")
    supplied_reference = _require_identifier(obj["registration_reference"], name="registration reference")
    if supplied_reference != reference:
        raise EvaluationCensusExecutionV1Error("registration reference differs from expected")
    timestamp = _require_timestamp(obj["registered_at_utc"])
    registered_execution = _require_mapping(
        obj["registered_execution"],
        (
            "execution_uuid",
            "execution_nonce",
            "execution_deadline_seconds",
            "attempt_identity_precommitted",
            "global_one_shot_consumption_enforced_by_repository",
        ),
        name="registered execution",
    )
    if (
        _require_uuid(registered_execution["execution_uuid"]) != execution_uuid
        or _require_nonce(registered_execution["execution_nonce"]) != execution_nonce
        or _require_integer(
            registered_execution["execution_deadline_seconds"],
            name="registered execution deadline seconds",
            minimum=1,
            maximum=86_400,
        )
        != execution_deadline
        or _require_boolean(
            registered_execution["attempt_identity_precommitted"],
            name="attempt identity precommitted",
        )
        is not True
        or _require_boolean(
            registered_execution["global_one_shot_consumption_enforced_by_repository"],
            name="global one-shot enforcement",
        )
        is not False
    ):
        raise EvaluationCensusExecutionV1Error("registered execution identity differs from expected")
    freeze_binding = _require_mapping(
        obj["freeze_request_binding"],
        (
            "freeze_request_digest",
            "exact_freeze_request_bytes_sha256",
            "exact_freeze_request_byte_count",
        ),
        name="registered freeze-request binding",
    )
    plan_binding = _require_mapping(
        obj["prospective_plan_binding"],
        ("prospective_plan_digest", "exact_plan_bytes_sha256", "exact_plan_byte_count"),
        name="registered plan binding",
    )
    expected = ExternalEvaluationCensusRegistrationReceiptV1(
        service,
        supplied_reference,
        timestamp,
        execution_uuid,
        execution_nonce,
        execution_deadline,
        freeze_request.digest,
        _sha256_bytes(freeze_request_text.encode("ascii")),
        len(freeze_request_text.encode("ascii")),
        _require_sha256(freeze_request.plan_binding["prospective_plan_digest"], name="plan digest"),
        _require_sha256(freeze_request.plan_binding["exact_plan_bytes_sha256"], name="plan sha256"),
        _require_integer(
            freeze_request.plan_binding["exact_plan_byte_count"],
            name="plan byte count",
            minimum=1,
        ),
    )
    if dict(freeze_binding) != {
        "freeze_request_digest": expected.freeze_request_digest,
        "exact_freeze_request_bytes_sha256": expected.exact_freeze_request_bytes_sha256,
        "exact_freeze_request_byte_count": expected.exact_freeze_request_byte_count,
    }:
        raise EvaluationCensusExecutionV1Error("registered freeze-request pins differ")
    if dict(plan_binding) != {
        "prospective_plan_digest": expected.prospective_plan_digest,
        "exact_plan_bytes_sha256": expected.exact_plan_bytes_sha256,
        "exact_plan_byte_count": expected.exact_plan_byte_count,
    }:
        raise EvaluationCensusExecutionV1Error("registered prospective-plan pins differ")
    _require_exact_constant(
        obj["claim_boundary"], expected._unsigned_obj()["claim_boundary"], name="claim boundary"
    )
    _require_exact_constant(obj["authorization"], _AUTHORIZATION, name="authorization")
    if _dump_json(value) != _dump_json(expected.as_obj()):
        raise EvaluationCensusExecutionV1Error("external registration receipt is inconsistent")
    if serialize_external_evaluation_census_registration_receipt_v1(expected) != text:
        raise EvaluationCensusExecutionV1Error(
            "external registration receipt is not canonical newline-terminated JSON"
        )
    return expected


@dataclass(frozen=True, slots=True)
class PreparedEvaluationCensusArtifactsV1:
    output_root: Path
    plan_path: Path
    freeze_request_path: Path
    plan: EvaluationCensusPlanV1
    freeze_request: EvaluationCensusFreezeRequestV1


def _secure_absent_root(path: Path) -> tuple[Path, int]:
    if not path.name or path.name in (".", ".."):
        raise EvaluationCensusExecutionV1Error("output root must name one new child directory")
    parent = path.parent.resolve(strict=True)
    parent_metadata = parent.stat()
    if not stat.S_ISDIR(parent_metadata.st_mode):
        raise EvaluationCensusExecutionV1Error("output parent must be a directory")
    target = parent / path.name
    try:
        os.mkdir(target, 0o700)
    except FileExistsError as exc:
        raise EvaluationCensusExecutionV1Error(
            "output root already exists; overwrite and partial-root reuse are forbidden"
        ) from exc
    descriptor_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        descriptor_flags |= os.O_NOFOLLOW
    descriptor = os.open(target, descriptor_flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise EvaluationCensusExecutionV1Error("created output root is not a directory")
        os.chmod(target, 0o700, follow_symlinks=False)
        parent_descriptor = os.open(parent, descriptor_flags)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except BaseException:
        os.close(descriptor)
        raise
    return target, descriptor


def _write_exclusive_at(directory_fd: int, name: str, payload: bytes, *, mode: int = 0o400) -> None:
    if "/" in name or name in ("", ".", ".."):
        raise EvaluationCensusExecutionV1Error("artifact name must be one safe path component")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(name, flags, mode, dir_fd=directory_fd)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise EvaluationCensusExecutionV1Error("short artifact write")
            view = view[written:]
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.fsync(directory_fd)


def _prepare(
    generator_seed: str,
    output_root: Path,
    *,
    runner_source_path: Path,
    plan_class: _PlanClass,
    attempts_per_formula_stratum: int,
    candidate_pool_size: int,
    source_archive_path: Path | None,
    constraints_path: Path | None,
    environment_lock_path: Path | None,
) -> PreparedEvaluationCensusArtifactsV1:
    if not _is_sha256(generator_seed):
        raise EvaluationCensusExecutionV1Error("generator seed must be exactly 64 lowercase hex")
    if plan_class == _PRODUCTION_PLAN_CLASS:
        _assert_production_environment_files(
            source_archive_path=source_archive_path,
            constraints_path=constraints_path,
            environment_lock_path=environment_lock_path,
        )
    plan = build_evaluation_census_plan_v1(
        generator_seed,
        attempts_per_formula_stratum=attempts_per_formula_stratum,
        candidate_pool_size=candidate_pool_size,
    )
    if plan_class == _PRODUCTION_PLAN_CLASS:
        _assert_production_plan(plan)
    plan_text = serialize_evaluation_census_plan_v1(plan)
    plan_bytes = plan_text.encode("ascii")
    request = _build_freeze_request(
        plan,
        plan_bytes,
        plan_class=plan_class,
        runner_source_path=runner_source_path,
        source_archive_path=source_archive_path,
        constraints_path=constraints_path,
        environment_lock_path=environment_lock_path,
    )
    request_bytes = serialize_evaluation_census_freeze_request_v1(request).encode("ascii")
    root, directory_fd = _secure_absent_root(output_root)
    try:
        _write_exclusive_at(directory_fd, PLAN_FILENAME, plan_bytes)
        # The request is the preparation completion marker and is durable last.
        _write_exclusive_at(directory_fd, FREEZE_REQUEST_FILENAME, request_bytes)
    finally:
        os.close(directory_fd)
    return PreparedEvaluationCensusArtifactsV1(
        root,
        root / PLAN_FILENAME,
        root / FREEZE_REQUEST_FILENAME,
        plan,
        request,
    )


def prepare_production_evaluation_census_v1(
    generator_seed: str,
    output_root: Path,
    *,
    runner_source_path: Path,
    source_archive_path: Path,
    constraints_path: Path,
    environment_lock_path: Path,
) -> PreparedEvaluationCensusArtifactsV1:
    """Prepare the one exact 9x16, pool-32 prospective production census."""

    return _prepare(
        generator_seed,
        output_root,
        runner_source_path=runner_source_path,
        plan_class=_PRODUCTION_PLAN_CLASS,
        attempts_per_formula_stratum=PRODUCTION_MIRROR_ATTEMPTS_PER_STRATUM,
        candidate_pool_size=DEFAULT_CANDIDATE_POOL_SIZE,
        source_archive_path=source_archive_path,
        constraints_path=constraints_path,
        environment_lock_path=environment_lock_path,
    )


def prepare_engineering_evaluation_census_fixture_for_testing_v1(
    generator_seed: str,
    output_root: Path,
    *,
    runner_source_path: Path,
    attempts_per_formula_stratum: int = 1,
    candidate_pool_size: int = 1,
) -> PreparedEvaluationCensusArtifactsV1:
    """Prepare a reduced fixture; deliberately unavailable from the production CLI."""

    if (
        attempts_per_formula_stratum == PRODUCTION_MIRROR_ATTEMPTS_PER_STRATUM
        and candidate_pool_size == DEFAULT_CANDIDATE_POOL_SIZE
    ):
        raise EvaluationCensusExecutionV1Error("the engineering-test API must not emit the production plan")
    return _prepare(
        generator_seed,
        output_root,
        runner_source_path=runner_source_path,
        plan_class=_ENGINEERING_PLAN_CLASS,
        attempts_per_formula_stratum=attempts_per_formula_stratum,
        candidate_pool_size=candidate_pool_size,
        source_archive_path=None,
        constraints_path=None,
        environment_lock_path=None,
    )


def _read_ascii(path: Path, *, name: str, maximum_bytes: int | None = None) -> tuple[str, bytes]:
    raw = _ordinary_file_bytes(path, name=name, maximum_bytes=maximum_bytes)
    try:
        return raw.decode("ascii"), raw
    except UnicodeDecodeError as exc:
        raise EvaluationCensusExecutionV1Error(f"{name} must be ASCII") from exc


def _safe_environment() -> dict[str, str]:
    root = _repository_root()
    return {
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": str(root / "src"),
    }


def _python_identity() -> dict[str, Any]:
    executable = Path(sys.executable).resolve(strict=True)
    payload = _ordinary_file_bytes(executable, name="Python executable")
    return {
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
        "executable_sha256": _sha256_bytes(payload),
        "executable_byte_count": len(payload),
    }


def _host_identity() -> dict[str, str]:
    return {
        "hostname": socket.gethostname(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
    }


def _process_max_rss() -> dict[str, int | str]:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        byte_count = raw
        raw_unit = "bytes"
    else:
        byte_count = raw * 1024
        raw_unit = "kibibytes"
    return {
        "measurement_api": "resource.getrusage(RUSAGE_SELF).ru_maxrss",
        "raw_value": raw,
        "raw_unit": raw_unit,
        "normalized_bytes": byte_count,
    }


def _worker_revalidate_frozen_bytes(args: argparse.Namespace) -> dict[str, Any]:
    expected_code = _load_json(cast(str, args.expected_execution_code_binding_json))
    expected_inputs = _load_json(cast(str, args.expected_environment_input_bindings_json))
    if _dump_json(expected_code) != cast(str, args.expected_execution_code_binding_json):
        raise EvaluationCensusExecutionV1Error("expected execution-code binding is not canonical JSON")
    if _dump_json(expected_inputs) != cast(str, args.expected_environment_input_bindings_json):
        raise EvaluationCensusExecutionV1Error("expected environment-input binding is not canonical JSON")
    current_code = _execution_code_binding(Path(cast(str, args.runner_source)))
    current_inputs = _environment_input_bindings(
        source_archive_path=(None if args.source_archive is None else Path(cast(str, args.source_archive))),
        constraints_path=None if args.constraints is None else Path(cast(str, args.constraints)),
        environment_lock_path=(
            None if args.environment_lock is None else Path(cast(str, args.environment_lock))
        ),
    )
    if expected_code != current_code:
        raise EvaluationCensusExecutionV1Error("worker execution code differs from frozen bytes")
    if expected_inputs != current_inputs:
        raise EvaluationCensusExecutionV1Error("worker environment inputs differ from frozen bytes")
    if args.require_production:
        _assert_production_environment_files(
            source_archive_path=(
                None if args.source_archive is None else Path(cast(str, args.source_archive))
            ),
            constraints_path=(None if args.constraints is None else Path(cast(str, args.constraints))),
            environment_lock_path=(
                None if args.environment_lock is None else Path(cast(str, args.environment_lock))
            ),
        )
    return {
        "execution_code_binding": current_code,
        "environment_input_bindings": current_inputs,
    }


def _report_accounting(report: EvaluationCensusReportV1) -> dict[str, Any]:
    coordinates = tuple((row.m, row.q) for row in report.cell_summary)
    if coordinates != _EXPECTED_CELL_COORDINATES:
        raise EvaluationCensusExecutionV1Error("report does not preserve all 36 canonical cells")
    opening_count = sum(len(member.openings) for attempt in report.attempts for member in attempt.members)
    composed = tuple(
        member.member_plan.composed_rule_id for attempt in report.attempts for member in attempt.members
    )
    return {
        "mirror_attempt_count": len(report.attempts),
        "opening_record_count": opening_count,
        "constructed_opening_count": report.constructed_opening_count,
        "preopening_failure_count": report.preopening_failure_count,
        "candidate_group_count": report.candidate_group_count,
        "complete_census_pair_count": report.complete_census_pair_count,
        "candidate_pool_size": report.prospective_candidate_pool_size,
        "distinct_evaluation_c_identity_count": len(set(composed)),
        "observed_cell_count": len(coordinates),
        "observed_cell_coordinates_digest": _digest(
            [[m, q] for m, q in coordinates], domain=_CELL_COORDINATE_DOMAIN
        ),
        "production_attempt_budget_complete": report.production_attempt_budget_complete,
        "all_planned_positions_preserved": opening_count == len(report.attempts) * 8,
        "all_36_cells_preserved": coordinates == _EXPECTED_CELL_COORDINATES,
        "early_stop_used": False,
    }


def _validate_report_claim_boundary(report: EvaluationCensusReportV1) -> None:
    value = report.as_obj()
    authorization = value.get("authorization")
    if type(authorization) is not dict or any(
        item is not False for key, item in authorization.items() if key != "scope"
    ):
        raise EvaluationCensusExecutionV1Error("report authorization must remain false")
    selection = value.get("selection_boundary")
    if type(selection) is not dict or any(item is not None for item in selection.values()):
        raise EvaluationCensusExecutionV1Error("report cannot contain a selection or matcher choice")
    summaries = value.get("observed_m_q_summary")
    if type(summaries) is not list or len(summaries) != 36:
        raise EvaluationCensusExecutionV1Error("report must contain all 36 cell rows")
    if any(type(row) is not dict or row.get("positive_quota") is not None for row in summaries):
        raise EvaluationCensusExecutionV1Error("report cannot contain a positive-cell quota")
    boundary = value.get("claim_boundary")
    if type(boundary) is not dict:
        raise EvaluationCensusExecutionV1Error("report claim boundary is missing")
    for field in (
        "positive_m_q_quota_selected",
        "matched_bank_size_selected",
        "difficulty_matcher_run",
        "evaluation_quartets_materialized",
        "nested_opening_unions_materialized",
        "challenge_panels_materialized",
        "render_balance_audited",
        "model_outcomes_present",
        "g01_authorized",
        "launch_authorized",
    ):
        if boundary.get(field) is not False:
            raise EvaluationCensusExecutionV1Error(f"report overclaims {field}")


def _worker_terminal(
    *,
    kind: str,
    started_monotonic_ns: int,
    deadline_monotonic_ns: int,
    plan_digest: str,
    plan_sha256: str,
    report: EvaluationCensusReportV1,
    report_bytes: bytes,
    frozen_byte_revalidation: Mapping[str, Any],
    started_wall_utc: str,
) -> dict[str, Any]:
    ended = time.monotonic_ns()
    ended_wall_utc = _utc_now()
    unsigned = {
        "schema_version": EVALUATION_CENSUS_EXECUTION_SCHEMA_VERSION,
        "terminal_kind": kind,
        "status": "success",
        "process_identity": {
            "pid": os.getpid(),
            "argv": list(sys.argv),
            "working_directory": str(Path.cwd()),
            "host": _host_identity(),
            "python": _python_identity(),
            "environment": _safe_environment(),
        },
        "deadline_monotonic_ns": deadline_monotonic_ns,
        "timing": {
            "started_monotonic_ns": started_monotonic_ns,
            "ended_monotonic_ns": ended,
            "elapsed_ns": ended - started_monotonic_ns,
            "started_wall_utc": started_wall_utc,
            "ended_wall_utc": ended_wall_utc,
        },
        "maximum_resident_set_size": _process_max_rss(),
        "input_plan_binding": {
            "prospective_plan_digest": plan_digest,
            "exact_plan_bytes_sha256": plan_sha256,
        },
        "frozen_byte_revalidation": dict(frozen_byte_revalidation),
        "report_binding": {
            "observed_report_digest": report.digest,
            "exact_report_bytes_sha256": _sha256_bytes(report_bytes),
            "exact_report_byte_count": len(report_bytes),
        },
        "accounting": _report_accounting(report),
        "authorization": dict(_AUTHORIZATION),
    }
    return {
        **unsigned,
        "worker_terminal_digest": _digest(unsigned, domain=_WORKER_TERMINAL_DOMAIN),
    }


def _start_parent_guardian(parent_guard_fd: int, guardian_ready_write_fd: int) -> None:
    """Kill this worker group when its watchdog's private pipe closes."""

    def watch() -> None:
        try:
            if os.write(guardian_ready_write_fd, b"R") != 1:
                raise OSError("short guardian-ready acknowledgement")
        except OSError:
            with suppress(OSError):
                os.close(guardian_ready_write_fd)
            with suppress(ProcessLookupError):
                os.killpg(os.getpgrp(), signal.SIGKILL)
            return
        os.close(guardian_ready_write_fd)
        try:
            while os.read(parent_guard_fd, 1):
                pass
        except OSError:
            pass
        finally:
            with suppress(OSError):
                os.close(parent_guard_fd)
        with suppress(ProcessLookupError):
            os.killpg(os.getpgrp(), signal.SIGKILL)

    threading.Thread(target=watch, name="census-watchdog-death-guard", daemon=True).start()


def _install_worker_lifetime_capability(worker_lifetime_write_fd: int) -> None:
    """Keep an exact-worker lifetime writer open, without inheriting it further."""

    os.set_inheritable(worker_lifetime_write_fd, False)


def _worker_build(args: argparse.Namespace) -> int:
    started = time.monotonic_ns()
    started_wall = _utc_now()
    _install_worker_lifetime_capability(
        _require_integer(args.worker_lifetime_write_fd, name="worker lifetime fd", minimum=0)
    )
    _start_parent_guardian(
        _require_integer(args.parent_guard_fd, name="parent guard fd", minimum=0),
        _require_integer(args.guardian_ready_write_fd, name="guardian ready fd", minimum=0),
    )
    frozen_bytes = _worker_revalidate_frozen_bytes(args)
    plan_path = Path(cast(str, args.plan))
    spool_path = Path(cast(str, args.output_spool))
    expected_digest = _require_sha256(args.expected_plan_digest, name="expected plan digest")
    expected_sha = _require_sha256(args.expected_plan_sha256, name="expected plan sha256")
    deadline = _require_integer(args.deadline_monotonic_ns, name="deadline monotonic ns", minimum=1)
    plan_text, plan_bytes = _read_ascii(plan_path, name="prospective plan", maximum_bytes=8 * 1024 * 1024)
    if _sha256_bytes(plan_bytes) != expected_sha:
        raise EvaluationCensusExecutionV1Error("worker plan bytes differ from the required SHA-256")
    plan = parse_evaluation_census_plan_v1(plan_text, expected_digest=expected_digest)
    if args.require_production:
        _assert_production_plan(plan)
    report = build_evaluation_census_report_v1(plan_text, expected_plan_digest=expected_digest)
    if args.require_production and not report.production_attempt_budget_complete:
        raise EvaluationCensusExecutionV1Error("worker produced a reduced report")
    _validate_report_claim_boundary(report)
    report_bytes = serialize_evaluation_census_report_v1(report).encode("ascii")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(spool_path, flags, 0o400)
    try:
        view = memoryview(report_bytes)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise EvaluationCensusExecutionV1Error("short report-spool write")
            view = view[written:]
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    parent_descriptor = os.open(spool_path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)
    terminal = _worker_terminal(
        kind=_BUILDER_TERMINAL_KIND,
        started_monotonic_ns=started,
        deadline_monotonic_ns=deadline,
        plan_digest=expected_digest,
        plan_sha256=expected_sha,
        report=report,
        report_bytes=report_bytes,
        frozen_byte_revalidation=frozen_bytes,
        started_wall_utc=started_wall,
    )
    sys.stdout.buffer.write(_canonical_bytes(terminal))
    sys.stdout.buffer.flush()
    return 0


def _worker_verify(args: argparse.Namespace) -> int:
    started = time.monotonic_ns()
    started_wall = _utc_now()
    _install_worker_lifetime_capability(
        _require_integer(args.worker_lifetime_write_fd, name="worker lifetime fd", minimum=0)
    )
    _start_parent_guardian(
        _require_integer(args.parent_guard_fd, name="parent guard fd", minimum=0),
        _require_integer(args.guardian_ready_write_fd, name="guardian ready fd", minimum=0),
    )
    frozen_bytes = _worker_revalidate_frozen_bytes(args)
    plan_path = Path(cast(str, args.plan))
    report_path = Path(cast(str, args.report_spool))
    plan_digest = _require_sha256(args.expected_plan_digest, name="expected plan digest")
    plan_sha = _require_sha256(args.expected_plan_sha256, name="expected plan sha256")
    report_digest = _require_sha256(args.expected_report_digest, name="expected report digest")
    report_sha = _require_sha256(args.expected_report_sha256, name="expected report sha256")
    deadline = _require_integer(args.deadline_monotonic_ns, name="deadline monotonic ns", minimum=1)
    plan_text, plan_bytes = _read_ascii(plan_path, name="prospective plan", maximum_bytes=8 * 1024 * 1024)
    report_text, report_bytes = _read_ascii(report_path, name="report spool", maximum_bytes=256 * 1024 * 1024)
    if _sha256_bytes(plan_bytes) != plan_sha or _sha256_bytes(report_bytes) != report_sha:
        raise EvaluationCensusExecutionV1Error("verifier input bytes differ from required pins")
    plan = parse_evaluation_census_plan_v1(plan_text, expected_digest=plan_digest)
    if args.require_production:
        _assert_production_plan(plan)
    # This process has not built a report.  Parsing re-executes the entire
    # census in this fresh process and compares every canonical field.
    report = parse_evaluation_census_report_v1(
        report_text,
        plan_text=plan_text,
        expected_plan_digest=plan_digest,
        expected_report_digest=report_digest,
    )
    if args.require_production and not report.production_attempt_budget_complete:
        raise EvaluationCensusExecutionV1Error("verifier received a reduced report")
    _validate_report_claim_boundary(report)
    terminal = _worker_terminal(
        kind=_VERIFIER_TERMINAL_KIND,
        started_monotonic_ns=started,
        deadline_monotonic_ns=deadline,
        plan_digest=plan_digest,
        plan_sha256=plan_sha,
        report=report,
        report_bytes=report_bytes,
        frozen_byte_revalidation=frozen_bytes,
        started_wall_utc=started_wall,
    )
    sys.stdout.buffer.write(_canonical_bytes(terminal))
    sys.stdout.buffer.flush()
    return 0


def _watchdog_worker(args: argparse.Namespace) -> int:
    """Own, bound, terminate, and reap exactly one census worker process."""

    started_ns = time.monotonic_ns()
    started_wall = _utc_now()
    parent_fd = _require_integer(args.parent_death_fd, name="parent death fd", minimum=0)
    worker_death_read_fd = _require_integer(args.worker_death_read_fd, name="worker death read fd", minimum=0)
    worker_death_write_fd = _require_integer(
        args.worker_death_write_fd, name="worker death write fd", minimum=0
    )
    worker_pid_report_fd = _require_integer(args.worker_pid_report_fd, name="worker pid report fd", minimum=0)
    guardian_ready_read_fd = _require_integer(
        args.guardian_ready_read_fd, name="guardian ready read fd", minimum=0
    )
    guardian_ready_write_fd = _require_integer(
        args.guardian_ready_write_fd, name="guardian ready write fd", minimum=0
    )
    worker_lifetime_write_fd = _require_integer(
        args.worker_lifetime_write_fd, name="worker lifetime write fd", minimum=0
    )
    worker_lifetime_read_fd = _require_integer(
        args.worker_lifetime_read_fd, name="worker lifetime read fd", minimum=0
    )
    deadline_ns = _require_integer(args.deadline_monotonic_ns, name="watchdog deadline", minimum=1)
    worker_argv_value = _load_json(cast(str, args.worker_argv_json))
    if (
        type(worker_argv_value) is not list
        or not worker_argv_value
        or any(type(item) is not str or "\0" in item for item in worker_argv_value)
        or _dump_json(worker_argv_value) != cast(str, args.worker_argv_json)
    ):
        raise EvaluationCensusExecutionV1Error("watchdog worker argv is not canonical")
    worker_argv = cast(list[str], worker_argv_value)
    if "--parent-guard-fd" not in worker_argv or worker_argv[
        worker_argv.index("--parent-guard-fd") + 1
    ] != str(worker_death_read_fd):
        raise EvaluationCensusExecutionV1Error("worker argv lacks the watchdog-death pipe")
    for option, expected in (
        ("--guardian-ready-write-fd", guardian_ready_write_fd),
        ("--worker-lifetime-write-fd", worker_lifetime_write_fd),
    ):
        if option not in worker_argv or worker_argv[worker_argv.index(option) + 1] != str(expected):
            raise EvaluationCensusExecutionV1Error(f"worker argv lacks the exact {option} capability")
    worker = subprocess.Popen(
        worker_argv,
        cwd=_repository_root(),
        env=_safe_environment(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        pass_fds=(worker_death_read_fd, guardian_ready_write_fd, worker_lifetime_write_fd),
    )
    os.close(worker_death_read_fd)
    worker_death_read_fd = -1
    os.close(guardian_ready_write_fd)
    guardian_ready_write_fd = -1
    os.close(worker_lifetime_write_fd)
    worker_lifetime_write_fd = -1
    guardian_ready = False
    while not guardian_ready:
        remaining_ns = deadline_ns - time.monotonic_ns()
        if remaining_ns <= 0:
            break
        readable, _, _ = select.select(
            [parent_fd, guardian_ready_read_fd], [], [], min(0.1, remaining_ns / 1_000_000_000)
        )
        if parent_fd in readable and os.read(parent_fd, 1) == b"":
            break
        if guardian_ready_read_fd in readable:
            guardian_ready = os.read(guardian_ready_read_fd, 2) == b"R"
            break
        if worker.poll() is not None:
            break
    os.close(guardian_ready_read_fd)
    guardian_ready_read_fd = -1
    if not guardian_ready:
        if worker.poll() is None:
            with suppress(ProcessLookupError):
                os.killpg(worker.pid, signal.SIGTERM)
            try:
                worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    os.killpg(worker.pid, signal.SIGKILL)
                worker.wait(timeout=5)
        raise EvaluationCensusExecutionV1Error("worker guardian did not acknowledge readiness")
    pid_payload = f"{worker.pid}\n".encode("ascii")
    if os.write(worker_pid_report_fd, pid_payload) != len(pid_payload):
        raise EvaluationCensusExecutionV1Error("short watchdog worker-PID report")
    os.close(worker_pid_report_fd)
    worker_pid_report_fd = -1
    trigger = "worker_exited"
    term_sent = False
    kill_sent = False
    stdout = b""
    stderr = b""
    try:
        while True:
            readable, _, _ = select.select([parent_fd], [], [], 0)
            if readable and os.read(parent_fd, 1) == b"":
                trigger = "controller_pipe_eof"
                break
            remaining_ns = deadline_ns - time.monotonic_ns()
            if remaining_ns <= 0:
                trigger = "registered_deadline"
                break
            try:
                stdout, stderr = worker.communicate(timeout=min(0.1, remaining_ns / 1_000_000_000))
                break
            except subprocess.TimeoutExpired:
                continue
        if worker.poll() is None:
            term_sent = True
            with suppress(ProcessLookupError):
                os.killpg(worker.pid, signal.SIGTERM)
            try:
                stdout, stderr = worker.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                kill_sent = True
                with suppress(ProcessLookupError):
                    os.killpg(worker.pid, signal.SIGKILL)
                stdout, stderr = worker.communicate(timeout=5)
    except BaseException:
        if worker.poll() is None:
            with suppress(ProcessLookupError):
                os.killpg(worker.pid, signal.SIGTERM)
            try:
                worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    os.killpg(worker.pid, signal.SIGKILL)
                worker.wait(timeout=5)
        raise
    finally:
        with suppress(OSError):
            os.close(parent_fd)
        with suppress(OSError):
            os.close(worker_death_read_fd)
        with suppress(OSError):
            os.close(worker_death_write_fd)
        with suppress(OSError):
            os.close(worker_pid_report_fd)
    lifetime_readable, _, _ = select.select([worker_lifetime_read_fd], [], [], 1)
    worker_lifetime_pipe_eof = bool(lifetime_readable) and os.read(worker_lifetime_read_fd, 1) == b""
    os.close(worker_lifetime_read_fd)
    worker_lifetime_read_fd = -1
    ended_ns = time.monotonic_ns()
    worker_terminal: Any = None
    if worker.returncode == 0 and not stderr:
        try:
            worker_terminal = _load_json(stdout.decode("ascii"))
        except (UnicodeDecodeError, EvaluationCensusExecutionV1Error):
            worker_terminal = None
    success = (
        trigger == "worker_exited"
        and worker.returncode == 0
        and not stderr
        and type(worker_terminal) is dict
        and ended_ns <= deadline_ns
        and worker_lifetime_pipe_eof
    )
    unsigned = {
        "schema_version": EVALUATION_CENSUS_EXECUTION_SCHEMA_VERSION,
        "terminal_kind": _WATCHDOG_TERMINAL_KIND,
        "status": "success" if success else "failed_worker_contained_and_reaped",
        "watchdog_process_identity": {
            "pid": os.getpid(),
            "argv": list(sys.argv),
            "working_directory": str(Path.cwd()),
            "host": _host_identity(),
            "python": _python_identity(),
            "environment": _safe_environment(),
        },
        "worker_process": {
            "pid": worker.pid,
            "argv": worker_argv,
            "returncode": worker.returncode,
            "stdout_sha256": _sha256_bytes(stdout),
            "stderr_sha256": _sha256_bytes(stderr),
            "stderr_byte_count": len(stderr),
            "reaped": worker.poll() is not None,
        },
        "containment": {
            "trigger": trigger,
            "term_sent": term_sent,
            "kill_sent": kill_sent,
            "guardian_ready_before_pid_report": guardian_ready,
            "worker_lifetime_pipe_eof_observed": worker_lifetime_pipe_eof,
            "worker_reaped_by_owning_watchdog": worker.poll() is not None,
            "watchdog_outside_worker_process_group": os.getpgrp() != worker.pid,
        },
        "deadline_monotonic_ns": deadline_ns,
        "timing": {
            "started_monotonic_ns": started_ns,
            "ended_monotonic_ns": ended_ns,
            "elapsed_ns": ended_ns - started_ns,
            "started_wall_utc": started_wall,
            "ended_wall_utc": _utc_now(),
        },
        "worker_terminal": worker_terminal,
        "authorization": dict(_AUTHORIZATION),
    }
    terminal = {
        **unsigned,
        "watchdog_terminal_digest": _digest(unsigned, domain=_WATCHDOG_TERMINAL_DOMAIN),
    }
    sys.stdout.buffer.write(_canonical_bytes(terminal))
    sys.stdout.buffer.flush()
    return 0 if success else 2


def _containment_probe_worker(args: argparse.Namespace) -> int:
    """Private test probe used only to exercise watchdog-death containment."""

    self_deadline_ns = _require_integer(
        args.self_deadline_monotonic_ns, name="containment probe self deadline", minimum=1
    )
    _install_worker_lifetime_capability(
        _require_integer(args.worker_lifetime_write_fd, name="worker lifetime fd", minimum=0)
    )
    _start_parent_guardian(
        _require_integer(args.parent_guard_fd, name="parent guard fd", minimum=0),
        _require_integer(args.guardian_ready_write_fd, name="guardian ready fd", minimum=0),
    )
    while time.monotonic_ns() < self_deadline_ns:
        time.sleep(0.05)
    return 2


def _worker_parser() -> argparse.ArgumentParser:
    def add_common(worker: argparse.ArgumentParser) -> None:
        worker.add_argument("--plan", required=True)
        worker.add_argument("--expected-plan-digest", required=True)
        worker.add_argument("--expected-plan-sha256", required=True)
        worker.add_argument("--deadline-monotonic-ns", required=True, type=int)
        worker.add_argument("--parent-guard-fd", required=True, type=int)
        worker.add_argument("--guardian-ready-write-fd", required=True, type=int)
        worker.add_argument("--worker-lifetime-write-fd", required=True, type=int)
        worker.add_argument("--runner-source", required=True)
        worker.add_argument("--expected-execution-code-binding-json", required=True)
        worker.add_argument("--expected-environment-input-bindings-json", required=True)
        worker.add_argument("--source-archive")
        worker.add_argument("--constraints")
        worker.add_argument("--environment-lock")
        worker.add_argument("--require-production", action="store_true")

    parser = argparse.ArgumentParser(add_help=False)
    subparsers = parser.add_subparsers(dest="worker_action", required=True)
    build = subparsers.add_parser("__build-worker", add_help=False)
    add_common(build)
    build.add_argument("--output-spool", required=True)
    verify = subparsers.add_parser("__verify-worker", add_help=False)
    add_common(verify)
    verify.add_argument("--report-spool", required=True)
    verify.add_argument("--expected-report-digest", required=True)
    verify.add_argument("--expected-report-sha256", required=True)
    watchdog = subparsers.add_parser("__watchdog-worker", add_help=False)
    watchdog.add_argument("--parent-death-fd", required=True, type=int)
    watchdog.add_argument("--worker-death-read-fd", required=True, type=int)
    watchdog.add_argument("--worker-death-write-fd", required=True, type=int)
    watchdog.add_argument("--worker-pid-report-fd", required=True, type=int)
    watchdog.add_argument("--guardian-ready-read-fd", required=True, type=int)
    watchdog.add_argument("--guardian-ready-write-fd", required=True, type=int)
    watchdog.add_argument("--worker-lifetime-read-fd", required=True, type=int)
    watchdog.add_argument("--worker-lifetime-write-fd", required=True, type=int)
    watchdog.add_argument("--deadline-monotonic-ns", required=True, type=int)
    watchdog.add_argument("--worker-argv-json", required=True)
    probe = subparsers.add_parser("__containment-probe-worker", add_help=False)
    probe.add_argument("--parent-guard-fd", required=True, type=int)
    probe.add_argument("--guardian-ready-write-fd", required=True, type=int)
    probe.add_argument("--worker-lifetime-write-fd", required=True, type=int)
    probe.add_argument("--self-deadline-monotonic-ns", required=True, type=int)
    return parser


def _validate_accounting(value: object, *, require_production: bool) -> Mapping[str, Any]:
    accounting = _require_mapping(
        value,
        (
            "mirror_attempt_count",
            "opening_record_count",
            "constructed_opening_count",
            "preopening_failure_count",
            "candidate_group_count",
            "complete_census_pair_count",
            "candidate_pool_size",
            "distinct_evaluation_c_identity_count",
            "observed_cell_count",
            "observed_cell_coordinates_digest",
            "production_attempt_budget_complete",
            "all_planned_positions_preserved",
            "all_36_cells_preserved",
            "early_stop_used",
        ),
        name="census accounting",
    )
    attempts = _require_integer(accounting["mirror_attempt_count"], name="attempt count", minimum=1)
    openings = _require_integer(accounting["opening_record_count"], name="opening count", minimum=1)
    constructed = _require_integer(accounting["constructed_opening_count"], name="constructed count")
    failures = _require_integer(accounting["preopening_failure_count"], name="failure count")
    groups = _require_integer(accounting["candidate_group_count"], name="group count")
    pairs = _require_integer(accounting["complete_census_pair_count"], name="pair count")
    pool = _require_integer(accounting["candidate_pool_size"], name="candidate pool", minimum=1)
    distinct_c = _require_integer(
        accounting["distinct_evaluation_c_identity_count"], name="distinct C count", minimum=2
    )
    if (
        openings != attempts * 8
        or constructed + failures != openings
        or groups > attempts * 2
        or pairs > attempts
        or distinct_c != attempts * 2
    ):
        raise EvaluationCensusExecutionV1Error("census accounting is arithmetically inconsistent")
    if _require_integer(accounting["observed_cell_count"], name="cell count") != 36:
        raise EvaluationCensusExecutionV1Error("census accounting must preserve 36 cells")
    expected_cell_digest = _digest(
        [[m, q] for m, q in _EXPECTED_CELL_COORDINATES], domain=_CELL_COORDINATE_DOMAIN
    )
    if (
        _require_sha256(accounting["observed_cell_coordinates_digest"], name="cell coordinate digest")
        != expected_cell_digest
    ):
        raise EvaluationCensusExecutionV1Error("cell coordinate digest is inconsistent")
    production = _require_boolean(
        accounting["production_attempt_budget_complete"], name="production completion"
    )
    if not _require_boolean(accounting["all_planned_positions_preserved"], name="all positions preserved"):
        raise EvaluationCensusExecutionV1Error("not all planned positions were preserved")
    if not _require_boolean(accounting["all_36_cells_preserved"], name="all cells preserved"):
        raise EvaluationCensusExecutionV1Error("not all 36 cells were preserved")
    if _require_boolean(accounting["early_stop_used"], name="early stop"):
        raise EvaluationCensusExecutionV1Error("early stop is forbidden")
    expected_production = attempts == 144 and openings == 1_152 and pool == 32 and distinct_c == 288
    if production != expected_production:
        raise EvaluationCensusExecutionV1Error("production completion flag is inconsistent")
    if require_production and not expected_production:
        raise EvaluationCensusExecutionV1Error("production execution has reduced accounting")
    if not require_production and expected_production:
        raise EvaluationCensusExecutionV1Error("engineering execution has production accounting")
    return accounting


def _validate_worker_timing(value: object, *, deadline_monotonic_ns: int) -> None:
    timing = _require_mapping(
        value,
        (
            "started_monotonic_ns",
            "ended_monotonic_ns",
            "elapsed_ns",
            "started_wall_utc",
            "ended_wall_utc",
        ),
        name="worker timing",
    )
    started = _require_integer(timing["started_monotonic_ns"], name="worker start", minimum=1)
    ended = _require_integer(timing["ended_monotonic_ns"], name="worker end", minimum=1)
    elapsed = _require_integer(timing["elapsed_ns"], name="worker elapsed")
    if ended < started or elapsed != ended - started or ended > deadline_monotonic_ns:
        raise EvaluationCensusExecutionV1Error("worker timing or deadline arithmetic is inconsistent")
    if _wall_datetime(timing["ended_wall_utc"], name="worker ended wall UTC") < _wall_datetime(
        timing["started_wall_utc"], name="worker started wall UTC"
    ):
        raise EvaluationCensusExecutionV1Error("worker wall-clock interval is reversed")


def _validate_worker_rss(value: object) -> None:
    rss = _require_mapping(
        value,
        ("measurement_api", "raw_value", "raw_unit", "normalized_bytes"),
        name="worker RSS",
    )
    if rss["measurement_api"] != "resource.getrusage(RUSAGE_SELF).ru_maxrss":
        raise EvaluationCensusExecutionV1Error("worker RSS measurement API changed")
    raw = _require_integer(rss["raw_value"], name="raw RSS", minimum=1)
    normalized = _require_integer(rss["normalized_bytes"], name="normalized RSS", minimum=1)
    if rss["raw_unit"] == "bytes":
        expected = raw
    elif rss["raw_unit"] == "kibibytes":
        expected = raw * 1024
    else:
        raise EvaluationCensusExecutionV1Error("worker RSS unit is unknown")
    if normalized != expected:
        raise EvaluationCensusExecutionV1Error("worker RSS normalization is inconsistent")


def _parse_worker_terminal(
    payload: bytes,
    *,
    expected_kind: str,
    expected_pid: int,
    expected_argv: Sequence[str],
    expected_deadline_monotonic_ns: int,
    expected_plan_digest: str,
    expected_plan_sha256: str,
    expected_frozen_byte_revalidation: Mapping[str, Any],
    require_production: bool,
    expected_report_digest: str | None = None,
    expected_report_sha256: str | None = None,
) -> Mapping[str, Any]:
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise EvaluationCensusExecutionV1Error("worker terminal output is not ASCII") from exc
    value = _load_json(text)
    obj = _require_mapping(
        value,
        (
            "schema_version",
            "terminal_kind",
            "status",
            "process_identity",
            "deadline_monotonic_ns",
            "timing",
            "maximum_resident_set_size",
            "input_plan_binding",
            "frozen_byte_revalidation",
            "report_binding",
            "accounting",
            "authorization",
            "worker_terminal_digest",
        ),
        name="worker terminal",
    )
    if obj["schema_version"] != EVALUATION_CENSUS_EXECUTION_SCHEMA_VERSION:
        raise EvaluationCensusExecutionV1Error("worker terminal schema changed")
    if obj["terminal_kind"] != expected_kind or obj["status"] != "success":
        raise EvaluationCensusExecutionV1Error("worker did not emit the expected success terminal")
    process = _require_mapping(
        obj["process_identity"],
        ("pid", "argv", "working_directory", "host", "python", "environment"),
        name="worker process identity",
    )
    if _require_integer(process["pid"], name="worker pid", minimum=1) != expected_pid:
        raise EvaluationCensusExecutionV1Error("worker PID differs from the launched process")
    expected_process_argv = [str(Path(__file__).resolve()), *expected_argv[3:]]
    if process["argv"] != expected_process_argv:
        raise EvaluationCensusExecutionV1Error("worker argv differs from the launched command")
    _require_exact_constant(process["host"], _host_identity(), name="worker host identity")
    _require_exact_constant(process["python"], _python_identity(), name="worker Python identity")
    _require_exact_constant(process["environment"], _safe_environment(), name="worker environment")
    if (
        _require_integer(obj["deadline_monotonic_ns"], name="worker deadline", minimum=1)
        != expected_deadline_monotonic_ns
    ):
        raise EvaluationCensusExecutionV1Error("worker deadline differs from the controller")
    _validate_worker_timing(obj["timing"], deadline_monotonic_ns=expected_deadline_monotonic_ns)
    _validate_worker_rss(obj["maximum_resident_set_size"])
    _require_exact_constant(
        obj["input_plan_binding"],
        {
            "prospective_plan_digest": expected_plan_digest,
            "exact_plan_bytes_sha256": expected_plan_sha256,
        },
        name="worker plan binding",
    )
    _require_exact_constant(
        obj["frozen_byte_revalidation"],
        expected_frozen_byte_revalidation,
        name="worker frozen-byte revalidation",
    )
    report_binding = _require_mapping(
        obj["report_binding"],
        ("observed_report_digest", "exact_report_bytes_sha256", "exact_report_byte_count"),
        name="worker report binding",
    )
    report_digest = _require_sha256(report_binding["observed_report_digest"], name="worker report digest")
    report_sha = _require_sha256(report_binding["exact_report_bytes_sha256"], name="worker report sha")
    _require_integer(report_binding["exact_report_byte_count"], name="worker report byte count", minimum=1)
    if expected_report_digest is not None and report_digest != expected_report_digest:
        raise EvaluationCensusExecutionV1Error("worker report digest differs from expected")
    if expected_report_sha256 is not None and report_sha != expected_report_sha256:
        raise EvaluationCensusExecutionV1Error("worker report SHA-256 differs from expected")
    _validate_accounting(obj["accounting"], require_production=require_production)
    _require_exact_constant(obj["authorization"], _AUTHORIZATION, name="authorization")
    unsigned = {key: item for key, item in obj.items() if key != "worker_terminal_digest"}
    if _require_sha256(obj["worker_terminal_digest"], name="worker terminal digest") != _digest(
        unsigned, domain=_WORKER_TERMINAL_DOMAIN
    ):
        raise EvaluationCensusExecutionV1Error("worker terminal digest is inconsistent")
    if _canonical_bytes(value) != payload:
        raise EvaluationCensusExecutionV1Error("worker terminal is not canonical JSON")
    return obj


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait(timeout=5)
        return
    try:
        process.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        process.wait(timeout=5)
        return
    process.wait(timeout=5)


def _require_worker_lifetime_eof(parent_death_pipe: _ParentDeathPipe, *, timeout: float) -> None:
    descriptor = parent_death_pipe.worker_lifetime_read_fd
    if descriptor < 0:
        raise EvaluationCensusExecutionV1Error("worker lifetime capability is unavailable")
    readable, _, _ = select.select([descriptor], [], [], timeout)
    if not readable or os.read(descriptor, 1) != b"":
        raise EvaluationCensusExecutionV1Error("worker lifetime capability did not reach exact EOF")
    os.close(descriptor)
    parent_death_pipe.worker_lifetime_read_fd = -1


@dataclass(slots=True)
class _ParentDeathPipe:
    read_fd: int
    write_fd: int
    worker_read_fd: int
    worker_write_fd: int
    pid_read_fd: int
    pid_write_fd: int
    guardian_ready_read_fd: int
    guardian_ready_write_fd: int
    worker_lifetime_read_fd: int
    worker_lifetime_write_fd: int

    @classmethod
    def create(cls) -> _ParentDeathPipe:
        read_fd, write_fd = os.pipe()
        worker_read_fd, worker_write_fd = os.pipe()
        pid_read_fd, pid_write_fd = os.pipe()
        guardian_ready_read_fd, guardian_ready_write_fd = os.pipe()
        worker_lifetime_read_fd, worker_lifetime_write_fd = os.pipe()
        return cls(
            read_fd,
            write_fd,
            worker_read_fd,
            worker_write_fd,
            pid_read_fd,
            pid_write_fd,
            guardian_ready_read_fd,
            guardian_ready_write_fd,
            worker_lifetime_read_fd,
            worker_lifetime_write_fd,
        )

    def close(self) -> None:
        for field in (
            "read_fd",
            "write_fd",
            "worker_read_fd",
            "worker_write_fd",
            "pid_read_fd",
            "pid_write_fd",
            "guardian_ready_read_fd",
            "guardian_ready_write_fd",
            "worker_lifetime_read_fd",
            "worker_lifetime_write_fd",
        ):
            descriptor = getattr(self, field)
            if descriptor >= 0:
                with suppress(OSError):
                    os.close(descriptor)
                setattr(self, field, -1)


def _parse_watchdog_terminal(
    payload: bytes,
    *,
    expected_pid: int,
    expected_watchdog_argv: Sequence[str],
    expected_worker_argv: Sequence[str],
    expected_deadline_monotonic_ns: int,
) -> tuple[bytes, int, Mapping[str, Any]]:
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise EvaluationCensusExecutionV1Error("watchdog terminal output is not ASCII") from exc
    value = _load_json(text)
    obj = _require_mapping(
        value,
        (
            "schema_version",
            "terminal_kind",
            "status",
            "watchdog_process_identity",
            "worker_process",
            "containment",
            "deadline_monotonic_ns",
            "timing",
            "worker_terminal",
            "authorization",
            "watchdog_terminal_digest",
        ),
        name="watchdog terminal",
    )
    if (
        obj["schema_version"] != EVALUATION_CENSUS_EXECUTION_SCHEMA_VERSION
        or obj["terminal_kind"] != _WATCHDOG_TERMINAL_KIND
        or obj["status"] != "success"
    ):
        raise EvaluationCensusExecutionV1Error("watchdog did not report successful containment")
    watchdog = _require_mapping(
        obj["watchdog_process_identity"],
        ("pid", "argv", "working_directory", "host", "python", "environment"),
        name="watchdog process identity",
    )
    if _require_integer(watchdog["pid"], name="watchdog pid", minimum=1) != expected_pid:
        raise EvaluationCensusExecutionV1Error("watchdog PID differs from launched process")
    expected_process_argv = [str(Path(__file__).resolve()), *expected_watchdog_argv[3:]]
    if watchdog["argv"] != expected_process_argv:
        raise EvaluationCensusExecutionV1Error("watchdog argv differs from launched command")
    _require_exact_constant(watchdog["host"], _host_identity(), name="watchdog host")
    _require_exact_constant(watchdog["python"], _python_identity(), name="watchdog Python")
    _require_exact_constant(watchdog["environment"], _safe_environment(), name="watchdog environment")
    worker = _require_mapping(
        obj["worker_process"],
        (
            "pid",
            "argv",
            "returncode",
            "stdout_sha256",
            "stderr_sha256",
            "stderr_byte_count",
            "reaped",
        ),
        name="watchdog worker process",
    )
    worker_pid = _require_integer(worker["pid"], name="guarded worker pid", minimum=1)
    if worker["argv"] != list(expected_worker_argv) or worker["returncode"] != 0:
        raise EvaluationCensusExecutionV1Error("guarded worker command or return code changed")
    worker_terminal = obj["worker_terminal"]
    if type(worker_terminal) is not dict:
        raise EvaluationCensusExecutionV1Error("watchdog lacks canonical worker terminal")
    worker_payload = _canonical_bytes(worker_terminal)
    if _require_sha256(worker["stdout_sha256"], name="worker stdout sha256") != _sha256_bytes(worker_payload):
        raise EvaluationCensusExecutionV1Error("watchdog worker stdout binding is inconsistent")
    if (
        _require_sha256(worker["stderr_sha256"], name="worker stderr sha256") != _sha256_bytes(b"")
        or _require_integer(worker["stderr_byte_count"], name="worker stderr count") != 0
    ):
        raise EvaluationCensusExecutionV1Error("guarded worker emitted stderr")
    if not _require_boolean(worker["reaped"], name="worker reaped"):
        raise EvaluationCensusExecutionV1Error("guarded worker was not reaped")
    containment = _require_mapping(
        obj["containment"],
        (
            "trigger",
            "term_sent",
            "kill_sent",
            "guardian_ready_before_pid_report",
            "worker_lifetime_pipe_eof_observed",
            "worker_reaped_by_owning_watchdog",
            "watchdog_outside_worker_process_group",
        ),
        name="watchdog containment",
    )
    if containment["trigger"] != "worker_exited":
        raise EvaluationCensusExecutionV1Error("watchdog success has a non-normal trigger")
    if _require_boolean(containment["term_sent"], name="watchdog TERM sent") or _require_boolean(
        containment["kill_sent"], name="watchdog KILL sent"
    ):
        raise EvaluationCensusExecutionV1Error("successful worker required guardian termination")
    if not all(
        _require_boolean(containment[key], name=f"watchdog {key}")
        for key in (
            "guardian_ready_before_pid_report",
            "worker_lifetime_pipe_eof_observed",
            "worker_reaped_by_owning_watchdog",
            "watchdog_outside_worker_process_group",
        )
    ):
        raise EvaluationCensusExecutionV1Error("watchdog containment invariants are false")
    if (
        _require_integer(obj["deadline_monotonic_ns"], name="watchdog deadline", minimum=1)
        != expected_deadline_monotonic_ns
    ):
        raise EvaluationCensusExecutionV1Error("watchdog deadline differs from expected")
    _validate_worker_timing(obj["timing"], deadline_monotonic_ns=expected_deadline_monotonic_ns)
    _require_exact_constant(obj["authorization"], _AUTHORIZATION, name="authorization")
    unsigned = {key: item for key, item in obj.items() if key != "watchdog_terminal_digest"}
    if _require_sha256(obj["watchdog_terminal_digest"], name="watchdog terminal digest") != _digest(
        unsigned, domain=_WATCHDOG_TERMINAL_DOMAIN
    ):
        raise EvaluationCensusExecutionV1Error("watchdog terminal digest is inconsistent")
    if _canonical_bytes(value) != payload:
        raise EvaluationCensusExecutionV1Error("watchdog terminal is not canonical JSON")
    return worker_payload, worker_pid, obj


def _run_worker(
    watchdog_argv: list[str],
    *,
    deadline_monotonic_ns: int,
    parent_death_pipe: _ParentDeathPipe,
) -> tuple[bytes, int, int]:
    remaining_ns = deadline_monotonic_ns - time.monotonic_ns()
    if remaining_ns <= 0:
        raise EvaluationCensusExecutionV1Error("execution deadline expired before worker launch")
    process: subprocess.Popen[bytes] | None = None
    reported_worker_pid: int | None = None
    try:
        process = subprocess.Popen(
            watchdog_argv,
            cwd=_repository_root(),
            env=_safe_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            pass_fds=(
                parent_death_pipe.read_fd,
                parent_death_pipe.worker_read_fd,
                parent_death_pipe.worker_write_fd,
                parent_death_pipe.pid_write_fd,
                parent_death_pipe.guardian_ready_read_fd,
                parent_death_pipe.guardian_ready_write_fd,
                parent_death_pipe.worker_lifetime_read_fd,
                parent_death_pipe.worker_lifetime_write_fd,
            ),
        )
        for field in (
            "read_fd",
            "worker_read_fd",
            "worker_write_fd",
            "pid_write_fd",
            "guardian_ready_read_fd",
            "guardian_ready_write_fd",
            "worker_lifetime_write_fd",
        ):
            descriptor = getattr(parent_death_pipe, field)
            os.close(descriptor)
            setattr(parent_death_pipe, field, -1)
        pid_wait_seconds = min(30.0, max(0.001, remaining_ns / 1_000_000_000))
        readable, _, _ = select.select([parent_death_pipe.pid_read_fd], [], [], pid_wait_seconds)
        if not readable:
            raise EvaluationCensusExecutionV1Error("watchdog did not report its worker PID")
        pid_bytes = os.read(parent_death_pipe.pid_read_fd, 32)
        os.close(parent_death_pipe.pid_read_fd)
        parent_death_pipe.pid_read_fd = -1
        if not pid_bytes.endswith(b"\n") or not pid_bytes[:-1].isdigit():
            raise EvaluationCensusExecutionV1Error("watchdog worker-PID report is malformed")
        reported_worker_pid = _require_integer(int(pid_bytes[:-1]), name="reported worker pid", minimum=1)
        try:
            stdout, stderr = process.communicate(timeout=remaining_ns / 1_000_000_000 + 7)
        except subprocess.TimeoutExpired as exc:
            with suppress(OSError):
                os.close(parent_death_pipe.write_fd)
            parent_death_pipe.write_fd = -1
            try:
                stdout, stderr = process.communicate(timeout=7)
            except subprocess.TimeoutExpired:
                _terminate_process_group(process)
            raise EvaluationCensusExecutionV1Error(
                "census watchdog exceeded deadline containment grace"
            ) from exc
        if process.returncode != 0:
            stderr_digest = _sha256_bytes(stderr)
            raise EvaluationCensusExecutionV1Error(
                f"census watchdog failed with code {process.returncode}; stderr sha256={stderr_digest}; "
                f"stdout sha256={_sha256_bytes(stdout)}"
            )
        if stderr:
            raise EvaluationCensusExecutionV1Error(
                f"census watchdog emitted unexpected stderr sha256={_sha256_bytes(stderr)}"
            )
        _require_worker_lifetime_eof(parent_death_pipe, timeout=7)
        return stdout, process.pid, reported_worker_pid
    except BaseException as exc:
        if process is not None:
            with suppress(OSError):
                os.close(parent_death_pipe.write_fd)
            parent_death_pipe.write_fd = -1
            try:
                process.wait(timeout=7)
            except subprocess.TimeoutExpired:
                _terminate_process_group(process)
        if parent_death_pipe.worker_lifetime_read_fd >= 0:
            try:
                _require_worker_lifetime_eof(parent_death_pipe, timeout=7)
            except EvaluationCensusExecutionV1Error as lifetime_exc:
                raise lifetime_exc from exc
        raise
    finally:
        parent_death_pipe.close()


def _install_controller_signal_handlers() -> dict[int, Any]:
    previous: dict[int, Any] = {}

    def interrupted(signum: int, _frame: Any) -> NoReturn:
        raise EvaluationCensusExecutionV1Error(
            f"controller interrupted by signal {signal.Signals(signum).name}"
        )

    for candidate in (
        signal.SIGTERM,
        signal.SIGINT,
        getattr(signal, "SIGHUP", None),
        getattr(signal, "SIGALRM", None),
    ):
        if candidate is None:
            continue
        signum = int(candidate)
        previous[signum] = signal.getsignal(candidate)
        signal.signal(candidate, interrupted)
    return previous


def _restore_controller_signal_handlers(previous: Mapping[int, Any]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def _started_obj(
    *,
    execution_uuid: str,
    execution_nonce: str,
    registered_at_utc: str,
    started_wall_utc: str,
    controller_argv: Sequence[str],
    builder_argv: Sequence[str],
    verifier_argv_prefix: Sequence[str],
    builder_watchdog_argv: Sequence[str],
    verifier_watchdog_argv_prefix: Sequence[str],
    deadline_seconds: int,
    started_monotonic_ns: int,
    deadline_monotonic_ns: int,
    input_binding: Mapping[str, Any],
) -> dict[str, Any]:
    unsigned = {
        "schema_version": EVALUATION_CENSUS_EXECUTION_SCHEMA_VERSION,
        "receipt_kind": _STARTED_KIND,
        "status": "started_no_retry",
        "execution_uuid": execution_uuid,
        "execution_nonce": execution_nonce,
        "wall_clock": {
            "registered_at_utc": registered_at_utc,
            "controller_started_at_utc": started_wall_utc,
            "registered_not_after_controller_start_by_supplied_and_local_clocks": True,
            "clock_or_service_authenticity_independently_verified": False,
        },
        "controller_process_identity": {
            "pid": os.getpid(),
            "argv": list(controller_argv),
            "working_directory": str(Path.cwd()),
            "host": _host_identity(),
            "python": _python_identity(),
            "worker_environment": _safe_environment(),
        },
        "worker_commands": {
            "builder_argv": list(builder_argv),
            "verifier_argv_prefix": list(verifier_argv_prefix),
            "builder_watchdog_argv": list(builder_watchdog_argv),
            "verifier_watchdog_argv_prefix": list(verifier_watchdog_argv_prefix),
        },
        "deadline": {
            "deadline_seconds": deadline_seconds,
            "started_monotonic_ns": started_monotonic_ns,
            "deadline_monotonic_ns": deadline_monotonic_ns,
        },
        "input_binding": dict(input_binding),
        "execution_policy": dict(_EXECUTION_POLICY),
        "authorization": dict(_AUTHORIZATION),
    }
    return {**unsigned, "started_receipt_digest": _digest(unsigned, domain=_STARTED_DOMAIN)}


def _failure_obj(
    *,
    execution_uuid: str,
    failed_stage: str,
    exc: BaseException,
    started_receipt_sha256: str,
) -> dict[str, Any]:
    unsigned = {
        "schema_version": EVALUATION_CENSUS_EXECUTION_SCHEMA_VERSION,
        "receipt_kind": _FAILURE_RECEIPT_KIND,
        "status": "failed_incomplete_root_do_not_reuse",
        "execution_uuid": execution_uuid,
        "failed_stage": failed_stage,
        "exception_type": type(exc).__name__,
        "exception_message_sha256": _sha256_bytes(str(exc).encode("utf-8")),
        "started_receipt_sha256": started_receipt_sha256,
        "completion_receipt_present": False,
        "retry_within_execution_uuid_allowed": False,
        "authorization": dict(_AUTHORIZATION),
    }
    return {**unsigned, "failure_receipt_digest": _digest(unsigned, domain=_FAILURE_RECEIPT_DOMAIN)}


def _commit_report(directory_fd: int, spool_name: str) -> None:
    try:
        os.link(
            spool_name,
            REPORT_FILENAME,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
    except FileExistsError as exc:
        raise EvaluationCensusExecutionV1Error("final report path already exists") from exc
    os.fsync(directory_fd)
    os.unlink(spool_name, dir_fd=directory_fd)
    os.fsync(directory_fd)


@dataclass(frozen=True, slots=True)
class ExecutedEvaluationCensusArtifactsV1:
    output_root: Path
    report_path: Path
    execution_receipt_path: Path
    report_digest: str
    report_bytes_sha256: str
    execution_receipt_digest: str


def _execute(
    *,
    plan_path: Path,
    freeze_request_path: Path,
    registration_receipt_path: Path,
    expected_plan_digest: str,
    expected_plan_bytes_sha256: str,
    expected_registration_receipt_sha256: str,
    expected_registration_reference: str,
    expected_execution_uuid: str,
    expected_execution_nonce: str,
    output_root: Path,
    runner_source_path: Path,
    deadline_seconds: int,
    controller_argv: Sequence[str],
    require_production: bool,
    source_archive_path: Path | None,
    constraints_path: Path | None,
    environment_lock_path: Path | None,
) -> ExecutedEvaluationCensusArtifactsV1:
    plan_digest = _require_sha256(expected_plan_digest, name="expected plan digest")
    plan_sha = _require_sha256(expected_plan_bytes_sha256, name="expected plan sha256")
    registration_sha = _require_sha256(
        expected_registration_receipt_sha256, name="expected registration receipt sha256"
    )
    registration_reference = _require_identifier(
        expected_registration_reference, name="expected registration reference"
    )
    execution_uuid = _require_uuid(expected_execution_uuid)
    execution_nonce = _require_nonce(expected_execution_nonce)
    deadline_value = _require_integer(deadline_seconds, name="deadline seconds", minimum=1, maximum=86_400)
    if output_root.name != execution_uuid:
        raise EvaluationCensusExecutionV1Error(
            "output-root basename must equal the externally registered execution UUID"
        )
    if require_production:
        _assert_production_environment_files(
            source_archive_path=source_archive_path,
            constraints_path=constraints_path,
            environment_lock_path=environment_lock_path,
        )
    if (
        type(controller_argv) not in (tuple, list)
        or not controller_argv
        or any(type(item) is not str or "\0" in item for item in controller_argv)
    ):
        raise EvaluationCensusExecutionV1Error("controller argv must be nonempty exact strings")

    # Validate every input and current source byte before claiming an execution
    # UUID or creating an output directory.
    plan_text, plan_bytes = _read_ascii(plan_path, name="prospective plan", maximum_bytes=8 * 1024 * 1024)
    if _sha256_bytes(plan_bytes) != plan_sha:
        raise EvaluationCensusExecutionV1Error("prospective-plan bytes differ from expected SHA-256")
    try:
        plan = parse_evaluation_census_plan_v1(plan_text, expected_digest=plan_digest)
    except EvaluationCensusV2Error as exc:
        raise EvaluationCensusExecutionV1Error(f"prospective plan rejected: {exc}") from exc
    if require_production:
        _assert_production_plan(plan)
    elif plan.uses_production_attempt_budget:
        raise EvaluationCensusExecutionV1Error("engineering-test execution cannot consume a production plan")
    freeze_text, freeze_bytes = _read_ascii(
        freeze_request_path, name="freeze request", maximum_bytes=2 * 1024 * 1024
    )
    request = parse_evaluation_census_freeze_request_v1(
        freeze_text,
        plan=plan,
        plan_text=plan_text,
        runner_source_path=runner_source_path,
        source_archive_path=source_archive_path,
        constraints_path=constraints_path,
        environment_lock_path=environment_lock_path,
    )
    expected_class = _PRODUCTION_PLAN_CLASS if require_production else _ENGINEERING_PLAN_CLASS
    if request.plan_class != expected_class:
        raise EvaluationCensusExecutionV1Error("freeze-request plan class differs from execution API")
    registration_text, registration_bytes = _read_ascii(
        registration_receipt_path,
        name="external registration receipt",
        maximum_bytes=2 * 1024 * 1024,
    )
    registration = parse_external_evaluation_census_registration_receipt_v1(
        registration_text,
        freeze_request=request,
        freeze_request_text=freeze_text,
        expected_bytes_sha256=registration_sha,
        expected_registration_reference=registration_reference,
        expected_execution_uuid=execution_uuid,
        expected_execution_nonce=execution_nonce,
        expected_execution_deadline_seconds=deadline_value,
    )
    controller_started_wall_utc = _utc_now()
    if _registration_datetime(registration.registered_at_utc) > _wall_datetime(
        controller_started_wall_utc, name="controller started wall UTC"
    ):
        raise EvaluationCensusExecutionV1Error(
            "registered-at time is later than the controller's local UTC start"
        )
    if not hasattr(signal, "setitimer") or signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
        raise EvaluationCensusExecutionV1Error(
            "production controller requires one unused POSIX real-time interval timer"
        )
    root, directory_fd = _secure_absent_root(output_root)
    started_ns = time.monotonic_ns()
    deadline_ns = started_ns + deadline_value * 1_000_000_000
    builder_parent_pipe = _ParentDeathPipe.create()
    verifier_parent_pipe = _ParentDeathPipe.create()
    spool_name = f".observed-report.{execution_uuid}.spool"
    spool_path = root / spool_name
    module_name = "goalzendo_interactive_v2.evaluation_census_execution"
    common = [
        str(Path(sys.executable).resolve(strict=True)),
        "-m",
        module_name,
    ]
    frozen_worker_args = [
        "--runner-source",
        str(runner_source_path.resolve(strict=True)),
        "--expected-execution-code-binding-json",
        _dump_json(dict(request.execution_code_binding)),
        "--expected-environment-input-bindings-json",
        _dump_json(dict(request.environment_input_bindings)),
    ]
    for option, optional_path in (
        ("--source-archive", source_archive_path),
        ("--constraints", constraints_path),
        ("--environment-lock", environment_lock_path),
    ):
        if optional_path is not None:
            frozen_worker_args.extend((option, str(optional_path.resolve(strict=True))))
    builder_argv = [
        *common,
        "__build-worker",
        "--plan",
        str(plan_path.resolve(strict=True)),
        "--expected-plan-digest",
        plan_digest,
        "--expected-plan-sha256",
        plan_sha,
        "--output-spool",
        str(spool_path),
        "--deadline-monotonic-ns",
        str(deadline_ns),
        "--parent-guard-fd",
        str(builder_parent_pipe.worker_read_fd),
        "--guardian-ready-write-fd",
        str(builder_parent_pipe.guardian_ready_write_fd),
        "--worker-lifetime-write-fd",
        str(builder_parent_pipe.worker_lifetime_write_fd),
        *frozen_worker_args,
    ]
    verifier_prefix = [
        *common,
        "__verify-worker",
        "--plan",
        str(plan_path.resolve(strict=True)),
        "--expected-plan-digest",
        plan_digest,
        "--expected-plan-sha256",
        plan_sha,
        "--report-spool",
        str(spool_path),
        "--parent-guard-fd",
        str(verifier_parent_pipe.worker_read_fd),
        "--guardian-ready-write-fd",
        str(verifier_parent_pipe.guardian_ready_write_fd),
        "--worker-lifetime-write-fd",
        str(verifier_parent_pipe.worker_lifetime_write_fd),
        *frozen_worker_args,
    ]
    if require_production:
        builder_argv.append("--require-production")
    builder_watchdog_argv = [
        *common,
        "__watchdog-worker",
        "--parent-death-fd",
        str(builder_parent_pipe.read_fd),
        "--worker-death-read-fd",
        str(builder_parent_pipe.worker_read_fd),
        "--worker-death-write-fd",
        str(builder_parent_pipe.worker_write_fd),
        "--worker-pid-report-fd",
        str(builder_parent_pipe.pid_write_fd),
        "--guardian-ready-read-fd",
        str(builder_parent_pipe.guardian_ready_read_fd),
        "--guardian-ready-write-fd",
        str(builder_parent_pipe.guardian_ready_write_fd),
        "--worker-lifetime-read-fd",
        str(builder_parent_pipe.worker_lifetime_read_fd),
        "--worker-lifetime-write-fd",
        str(builder_parent_pipe.worker_lifetime_write_fd),
        "--deadline-monotonic-ns",
        str(deadline_ns),
        "--worker-argv-json",
        _dump_json(builder_argv),
    ]
    verifier_watchdog_prefix = [
        *common,
        "__watchdog-worker",
        "--parent-death-fd",
        str(verifier_parent_pipe.read_fd),
        "--worker-death-read-fd",
        str(verifier_parent_pipe.worker_read_fd),
        "--worker-death-write-fd",
        str(verifier_parent_pipe.worker_write_fd),
        "--worker-pid-report-fd",
        str(verifier_parent_pipe.pid_write_fd),
        "--guardian-ready-read-fd",
        str(verifier_parent_pipe.guardian_ready_read_fd),
        "--guardian-ready-write-fd",
        str(verifier_parent_pipe.guardian_ready_write_fd),
        "--worker-lifetime-read-fd",
        str(verifier_parent_pipe.worker_lifetime_read_fd),
        "--worker-lifetime-write-fd",
        str(verifier_parent_pipe.worker_lifetime_write_fd),
        "--deadline-monotonic-ns",
        str(deadline_ns),
        "--worker-argv-json",
    ]
    input_binding = {
        "prospective_plan": _plan_binding(plan, plan_bytes),
        "freeze_request": {
            "freeze_request_digest": request.digest,
            "exact_freeze_request_bytes_sha256": _sha256_bytes(freeze_bytes),
            "exact_freeze_request_byte_count": len(freeze_bytes),
        },
        "external_registration_receipt": {
            "registration_reference": registration.registration_reference,
            "registration_receipt_digest": registration.digest,
            "exact_registration_receipt_bytes_sha256": _sha256_bytes(registration_bytes),
            "exact_registration_receipt_byte_count": len(registration_bytes),
        },
        "execution_code_binding": request.execution_code_binding,
        "environment_input_bindings": request.environment_input_bindings,
    }
    started = _started_obj(
        execution_uuid=execution_uuid,
        execution_nonce=execution_nonce,
        registered_at_utc=registration.registered_at_utc,
        started_wall_utc=controller_started_wall_utc,
        controller_argv=controller_argv,
        builder_argv=builder_argv,
        verifier_argv_prefix=verifier_prefix,
        builder_watchdog_argv=builder_watchdog_argv,
        verifier_watchdog_argv_prefix=verifier_watchdog_prefix,
        deadline_seconds=deadline_value,
        started_monotonic_ns=started_ns,
        deadline_monotonic_ns=deadline_ns,
        input_binding=input_binding,
    )
    started_bytes = _canonical_bytes(started)
    failed_stage = "write_started_receipt"
    previous_signal_handlers = _install_controller_signal_handlers()
    signal.setitimer(
        signal.ITIMER_REAL,
        max(0.000001, (deadline_ns - time.monotonic_ns()) / 1_000_000_000),
    )
    try:
        _write_exclusive_at(directory_fd, STARTED_RECEIPT_FILENAME, started_bytes)
        failed_stage = "builder"
        builder_watchdog_stdout, builder_watchdog_pid, reported_builder_pid = _run_worker(
            builder_watchdog_argv,
            deadline_monotonic_ns=deadline_ns,
            parent_death_pipe=builder_parent_pipe,
        )
        builder_stdout, builder_pid, builder_watchdog_terminal = _parse_watchdog_terminal(
            builder_watchdog_stdout,
            expected_pid=builder_watchdog_pid,
            expected_watchdog_argv=builder_watchdog_argv,
            expected_worker_argv=builder_argv,
            expected_deadline_monotonic_ns=deadline_ns,
        )
        if builder_pid != reported_builder_pid:
            raise EvaluationCensusExecutionV1Error("builder PID differs from early watchdog report")
        builder_terminal = _parse_worker_terminal(
            builder_stdout,
            expected_kind=_BUILDER_TERMINAL_KIND,
            expected_pid=builder_pid,
            expected_argv=builder_argv,
            expected_deadline_monotonic_ns=deadline_ns,
            expected_plan_digest=plan_digest,
            expected_plan_sha256=plan_sha,
            expected_frozen_byte_revalidation={
                "execution_code_binding": request.execution_code_binding,
                "environment_input_bindings": request.environment_input_bindings,
            },
            require_production=require_production,
        )
        builder_watchdog_bytes = _canonical_bytes(builder_watchdog_terminal)
        _write_exclusive_at(
            directory_fd,
            BUILDER_GUARDIAN_TERMINAL_FILENAME,
            builder_watchdog_bytes,
        )
        builder_bytes = _canonical_bytes(builder_terminal)
        _write_exclusive_at(directory_fd, BUILDER_TERMINAL_FILENAME, builder_bytes)
        report_binding = _require_mapping(
            builder_terminal["report_binding"],
            ("observed_report_digest", "exact_report_bytes_sha256", "exact_report_byte_count"),
            name="builder report binding",
        )
        report_digest = _require_sha256(
            report_binding["observed_report_digest"], name="observed report digest"
        )
        report_sha = _require_sha256(report_binding["exact_report_bytes_sha256"], name="report sha256")
        report_byte_count = _require_integer(
            report_binding["exact_report_byte_count"], name="report byte count", minimum=1
        )
        spool_bytes = _ordinary_file_bytes(spool_path, name="report spool", maximum_bytes=256 * 1024 * 1024)
        if _sha256_bytes(spool_bytes) != report_sha or len(spool_bytes) != report_byte_count:
            raise EvaluationCensusExecutionV1Error("report spool differs from builder terminal")

        failed_stage = "verifier"
        verifier_argv = [
            *verifier_prefix,
            "--expected-report-digest",
            report_digest,
            "--expected-report-sha256",
            report_sha,
            "--deadline-monotonic-ns",
            str(deadline_ns),
        ]
        if require_production:
            verifier_argv.append("--require-production")
        verifier_watchdog_argv = [*verifier_watchdog_prefix, _dump_json(verifier_argv)]
        verifier_watchdog_stdout, verifier_watchdog_pid, reported_verifier_pid = _run_worker(
            verifier_watchdog_argv,
            deadline_monotonic_ns=deadline_ns,
            parent_death_pipe=verifier_parent_pipe,
        )
        verifier_stdout, verifier_pid, verifier_watchdog_terminal = _parse_watchdog_terminal(
            verifier_watchdog_stdout,
            expected_pid=verifier_watchdog_pid,
            expected_watchdog_argv=verifier_watchdog_argv,
            expected_worker_argv=verifier_argv,
            expected_deadline_monotonic_ns=deadline_ns,
        )
        if verifier_pid != reported_verifier_pid:
            raise EvaluationCensusExecutionV1Error("verifier PID differs from early watchdog report")
        verifier_terminal = _parse_worker_terminal(
            verifier_stdout,
            expected_kind=_VERIFIER_TERMINAL_KIND,
            expected_pid=verifier_pid,
            expected_argv=verifier_argv,
            expected_deadline_monotonic_ns=deadline_ns,
            expected_plan_digest=plan_digest,
            expected_plan_sha256=plan_sha,
            expected_frozen_byte_revalidation={
                "execution_code_binding": request.execution_code_binding,
                "environment_input_bindings": request.environment_input_bindings,
            },
            require_production=require_production,
            expected_report_digest=report_digest,
            expected_report_sha256=report_sha,
        )
        verifier_watchdog_bytes = _canonical_bytes(verifier_watchdog_terminal)
        _write_exclusive_at(
            directory_fd,
            VERIFIER_GUARDIAN_TERMINAL_FILENAME,
            verifier_watchdog_bytes,
        )
        verifier_bytes = _canonical_bytes(verifier_terminal)
        _write_exclusive_at(directory_fd, VERIFIER_TERMINAL_FILENAME, verifier_bytes)
        if verifier_terminal["report_binding"] != builder_terminal["report_binding"]:
            raise EvaluationCensusExecutionV1Error("fresh verifier report binding differs from builder")
        if verifier_terminal["accounting"] != builder_terminal["accounting"]:
            raise EvaluationCensusExecutionV1Error("fresh verifier accounting differs from builder")
        if _execution_code_binding(runner_source_path) != request.execution_code_binding:
            raise EvaluationCensusExecutionV1Error("execution code changed after fresh-process verification")
        if (
            _environment_input_bindings(
                source_archive_path=source_archive_path,
                constraints_path=constraints_path,
                environment_lock_path=environment_lock_path,
            )
            != request.environment_input_bindings
        ):
            raise EvaluationCensusExecutionV1Error(
                "frozen environment inputs changed after fresh-process verification"
            )
        if time.monotonic_ns() > deadline_ns:
            raise EvaluationCensusExecutionV1Error(
                "registered execution deadline expired before report commit"
            )

        failed_stage = "commit_report"
        _commit_report(directory_fd, spool_name)
        committed_bytes = _ordinary_file_bytes(
            root / REPORT_FILENAME, name="committed report", maximum_bytes=256 * 1024 * 1024
        )
        if committed_bytes != spool_bytes:
            raise EvaluationCensusExecutionV1Error("committed report differs from verified spool")
        if time.monotonic_ns() > deadline_ns:
            raise EvaluationCensusExecutionV1Error(
                "registered execution deadline expired after report commit"
            )

        failed_stage = "write_completion_receipt"
        ended_ns = time.monotonic_ns()
        if ended_ns > deadline_ns:
            raise EvaluationCensusExecutionV1Error(
                "registered execution deadline expired before completion receipt"
            )
        ended_wall_utc = _utc_now()
        unsigned_receipt = {
            "schema_version": EVALUATION_CENSUS_EXECUTION_SCHEMA_VERSION,
            "receipt_kind": _EXECUTION_RECEIPT_KIND,
            "status": "complete_verified_nonauthorizing",
            "execution_uuid": execution_uuid,
            "execution_nonce": execution_nonce,
            "input_binding": input_binding,
            "external_registration_validation": {
                "expected_registration_reference": registration_reference,
                "expected_registration_receipt_sha256": registration_sha,
                "registered_at_utc": registration.registered_at_utc,
                "controller_started_at_utc": controller_started_wall_utc,
                "registered_execution_deadline_seconds": deadline_value,
                "registered_not_after_controller_start_by_supplied_and_local_clocks": True,
                **_REGISTRATION_CLAIM_BOUNDARY,
            },
            "stage_receipts": {
                "started": {
                    "filename": STARTED_RECEIPT_FILENAME,
                    "sha256": _sha256_bytes(started_bytes),
                    "byte_count": len(started_bytes),
                },
                "builder_guardian_terminal": {
                    "filename": BUILDER_GUARDIAN_TERMINAL_FILENAME,
                    "sha256": _sha256_bytes(builder_watchdog_bytes),
                    "byte_count": len(builder_watchdog_bytes),
                    "watchdog_terminal_digest": builder_watchdog_terminal["watchdog_terminal_digest"],
                },
                "builder_terminal": {
                    "filename": BUILDER_TERMINAL_FILENAME,
                    "sha256": _sha256_bytes(builder_bytes),
                    "byte_count": len(builder_bytes),
                    "worker_terminal_digest": builder_terminal["worker_terminal_digest"],
                },
                "verifier_guardian_terminal": {
                    "filename": VERIFIER_GUARDIAN_TERMINAL_FILENAME,
                    "sha256": _sha256_bytes(verifier_watchdog_bytes),
                    "byte_count": len(verifier_watchdog_bytes),
                    "watchdog_terminal_digest": verifier_watchdog_terminal["watchdog_terminal_digest"],
                },
                "verifier_terminal": {
                    "filename": VERIFIER_TERMINAL_FILENAME,
                    "sha256": _sha256_bytes(verifier_bytes),
                    "byte_count": len(verifier_bytes),
                    "worker_terminal_digest": verifier_terminal["worker_terminal_digest"],
                },
            },
            "report_binding": {
                "filename": REPORT_FILENAME,
                "observed_report_digest": report_digest,
                "exact_report_bytes_sha256": report_sha,
                "exact_report_byte_count": report_byte_count,
                "committed_after_fresh_process_verification": True,
            },
            "accounting": builder_terminal["accounting"],
            "runtime_evidence": {
                "contract": (
                    "per-worker monotonic elapsed_ns and process-self ru_maxrss normalized "
                    "to bytes; descriptive engineering evidence, not a scientific gate"
                ),
                "builder": {
                    "timing": builder_terminal["timing"],
                    "maximum_resident_set_size": builder_terminal["maximum_resident_set_size"],
                    "process_identity": builder_terminal["process_identity"],
                },
                "builder_watchdog": {
                    "timing": builder_watchdog_terminal["timing"],
                    "process_identity": builder_watchdog_terminal["watchdog_process_identity"],
                    "containment": builder_watchdog_terminal["containment"],
                },
                "verifier": {
                    "timing": verifier_terminal["timing"],
                    "maximum_resident_set_size": verifier_terminal["maximum_resident_set_size"],
                    "process_identity": verifier_terminal["process_identity"],
                },
                "verifier_watchdog": {
                    "timing": verifier_watchdog_terminal["timing"],
                    "process_identity": verifier_watchdog_terminal["watchdog_process_identity"],
                    "containment": verifier_watchdog_terminal["containment"],
                },
                "controller": {
                    "pid": os.getpid(),
                    "started_monotonic_ns": started_ns,
                    "ended_monotonic_ns": ended_ns,
                    "elapsed_ns": ended_ns - started_ns,
                    "deadline_seconds": deadline_value,
                    "deadline_monotonic_ns": deadline_ns,
                    "started_wall_utc": controller_started_wall_utc,
                    "ended_wall_utc": ended_wall_utc,
                },
            },
            "execution_policy": started["execution_policy"],
            "scientific_claim_boundary": dict(_SCIENTIFIC_CLAIM_BOUNDARY),
            "authorization": dict(_AUTHORIZATION),
            "completion": {
                "started_receipt_written_first": True,
                "fresh_builder_succeeded": True,
                "fresh_verifier_succeeded": True,
                "report_committed_before_completion_receipt": True,
                "completion_receipt_written_last": True,
                "failure_receipt_present": False,
            },
        }
        receipt_obj = {
            **unsigned_receipt,
            "execution_receipt_digest": _digest(unsigned_receipt, domain=_EXECUTION_RECEIPT_DOMAIN),
        }
        receipt_bytes = _canonical_bytes(receipt_obj)
        _write_exclusive_at(directory_fd, EXECUTION_RECEIPT_FILENAME, receipt_bytes)
        durable_completion_ns = time.monotonic_ns()
        if durable_completion_ns > deadline_ns:
            raise EvaluationCensusExecutionV1Error(
                "registered execution deadline expired before durable completion"
            )
        return ExecutedEvaluationCensusArtifactsV1(
            root,
            root / REPORT_FILENAME,
            root / EXECUTION_RECEIPT_FILENAME,
            report_digest,
            report_sha,
            cast(str, receipt_obj["execution_receipt_digest"]),
        )
    except BaseException as exc:
        try:
            failure = _failure_obj(
                execution_uuid=execution_uuid,
                failed_stage=failed_stage,
                exc=exc,
                started_receipt_sha256=_sha256_bytes(started_bytes),
            )
            _write_exclusive_at(directory_fd, FAILURE_RECEIPT_FILENAME, _canonical_bytes(failure))
        except BaseException:
            pass
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        builder_parent_pipe.close()
        verifier_parent_pipe.close()
        _restore_controller_signal_handlers(previous_signal_handlers)
        os.close(directory_fd)


def execute_production_evaluation_census_v1(
    *,
    plan_path: Path,
    freeze_request_path: Path,
    registration_receipt_path: Path,
    expected_plan_digest: str,
    expected_plan_bytes_sha256: str,
    expected_registration_receipt_sha256: str,
    expected_registration_reference: str,
    expected_execution_uuid: str,
    expected_execution_nonce: str,
    output_root: Path,
    runner_source_path: Path,
    deadline_seconds: int,
    controller_argv: Sequence[str],
    source_archive_path: Path,
    constraints_path: Path,
    environment_lock_path: Path,
) -> ExecutedEvaluationCensusArtifactsV1:
    """Execute only a registered exact production plan in two fresh processes."""

    if tuple(controller_argv) != tuple(sys.argv):
        raise EvaluationCensusExecutionV1Error("production controller argv must equal the live process argv")
    live_entrypoint = (Path.cwd() / sys.argv[0]).resolve(strict=True)
    if live_entrypoint != runner_source_path.resolve(strict=True):
        raise EvaluationCensusExecutionV1Error("production execution must be launched by the bound runner")
    return _execute(
        plan_path=plan_path,
        freeze_request_path=freeze_request_path,
        registration_receipt_path=registration_receipt_path,
        expected_plan_digest=expected_plan_digest,
        expected_plan_bytes_sha256=expected_plan_bytes_sha256,
        expected_registration_receipt_sha256=expected_registration_receipt_sha256,
        expected_registration_reference=expected_registration_reference,
        expected_execution_uuid=expected_execution_uuid,
        expected_execution_nonce=expected_execution_nonce,
        output_root=output_root,
        runner_source_path=runner_source_path,
        deadline_seconds=deadline_seconds,
        controller_argv=controller_argv,
        require_production=True,
        source_archive_path=source_archive_path,
        constraints_path=constraints_path,
        environment_lock_path=environment_lock_path,
    )


def execute_engineering_evaluation_census_fixture_for_testing_v1(
    *,
    plan_path: Path,
    freeze_request_path: Path,
    registration_receipt_path: Path,
    expected_plan_digest: str,
    expected_plan_bytes_sha256: str,
    expected_registration_receipt_sha256: str,
    expected_registration_reference: str,
    expected_execution_uuid: str,
    expected_execution_nonce: str,
    output_root: Path,
    runner_source_path: Path,
    deadline_seconds: int = 600,
) -> ExecutedEvaluationCensusArtifactsV1:
    """Execute a reduced fixture; deliberately unavailable from the production CLI."""

    return _execute(
        plan_path=plan_path,
        freeze_request_path=freeze_request_path,
        registration_receipt_path=registration_receipt_path,
        expected_plan_digest=expected_plan_digest,
        expected_plan_bytes_sha256=expected_plan_bytes_sha256,
        expected_registration_receipt_sha256=expected_registration_receipt_sha256,
        expected_registration_reference=expected_registration_reference,
        expected_execution_uuid=expected_execution_uuid,
        expected_execution_nonce=expected_execution_nonce,
        output_root=output_root,
        runner_source_path=runner_source_path,
        deadline_seconds=deadline_seconds,
        controller_argv=("engineering-test-api", "execute"),
        require_production=False,
        source_archive_path=None,
        constraints_path=None,
        environment_lock_path=None,
    )


def _validate_source_file_binding(value: object, *, name: str) -> None:
    binding = _require_mapping(
        value, ("relative_path", "sha256", "byte_count"), name=f"{name} source binding"
    )
    if type(binding["relative_path"]) is not str or not binding["relative_path"]:
        raise EvaluationCensusExecutionV1Error(f"{name} relative path is invalid")
    _require_sha256(binding["sha256"], name=f"{name} sha256")
    _require_integer(binding["byte_count"], name=f"{name} byte count", minimum=1)


def _validate_python_identity_obj(value: object) -> None:
    identity = _require_mapping(
        value,
        ("implementation", "version", "executable_sha256", "executable_byte_count"),
        name="Python identity",
    )
    if type(identity["implementation"]) is not str or type(identity["version"]) is not str:
        raise EvaluationCensusExecutionV1Error("Python identity strings are invalid")
    _require_sha256(identity["executable_sha256"], name="Python executable sha256")
    _require_integer(identity["executable_byte_count"], name="Python byte count", minimum=1)


def _validate_execution_code_binding_obj(value: object) -> None:
    binding = _require_mapping(
        value,
        ("execution_module", "runner", "python", "worker_environment"),
        name="execution code binding",
    )
    _validate_source_file_binding(binding["execution_module"], name="execution module")
    _validate_source_file_binding(binding["runner"], name="runner")
    _validate_python_identity_obj(binding["python"])
    _require_exact_constant(binding["worker_environment"], _safe_environment(), name="worker environment")


def _validate_environment_input_bindings_obj(value: object, *, require_production: bool) -> None:
    bindings = _require_mapping(
        value,
        ("source_archive", "constraints", "environment_lock"),
        name="environment input bindings",
    )
    for field, label in (
        ("source_archive", "source_archive"),
        ("constraints", "constraints"),
        ("environment_lock", "environment_lock"),
    ):
        binding = _require_mapping(
            bindings[field],
            ("label", "supplied", "sha256", "byte_count"),
            name=f"{field} binding",
        )
        if binding["label"] != label:
            raise EvaluationCensusExecutionV1Error(f"{field} label changed")
        supplied = _require_boolean(binding["supplied"], name=f"{field} supplied")
        if supplied:
            _require_sha256(binding["sha256"], name=f"{field} sha256")
            _require_integer(binding["byte_count"], name=f"{field} byte count", minimum=1)
        elif binding["sha256"] is not None or binding["byte_count"] is not None:
            raise EvaluationCensusExecutionV1Error(f"absent {field} has content pins")
        if require_production and not supplied:
            raise EvaluationCensusExecutionV1Error(f"production receipt lacks {field}")


def _validate_process_identity_obj(value: object) -> None:
    process = _require_mapping(
        value,
        ("pid", "argv", "working_directory", "host", "python", "environment"),
        name="process identity",
    )
    _require_integer(process["pid"], name="process pid", minimum=1)
    if (
        type(process["argv"]) is not list
        or not process["argv"]
        or any(type(item) is not str or "\0" in item for item in process["argv"])
    ):
        raise EvaluationCensusExecutionV1Error("process argv is invalid")
    if type(process["working_directory"]) is not str or not Path(process["working_directory"]).is_absolute():
        raise EvaluationCensusExecutionV1Error("process working directory is not absolute")
    host = _require_mapping(
        process["host"], ("hostname", "system", "release", "machine"), name="host identity"
    )
    if any(type(item) is not str or not item for item in host.values()):
        raise EvaluationCensusExecutionV1Error("host identity is invalid")
    _validate_python_identity_obj(process["python"])
    _require_exact_constant(process["environment"], _safe_environment(), name="process environment")


def _validate_execution_receipt_input(value: object, *, require_production: bool) -> None:
    inputs = _require_mapping(
        value,
        (
            "prospective_plan",
            "freeze_request",
            "external_registration_receipt",
            "execution_code_binding",
            "environment_input_bindings",
        ),
        name="execution input binding",
    )
    plan = _require_mapping(
        inputs["prospective_plan"],
        ("prospective_plan_digest", "exact_plan_bytes_sha256", "exact_plan_byte_count"),
        name="receipt plan binding",
    )
    _require_sha256(plan["prospective_plan_digest"], name="plan digest")
    _require_sha256(plan["exact_plan_bytes_sha256"], name="plan sha256")
    _require_integer(plan["exact_plan_byte_count"], name="plan byte count", minimum=1)
    freeze = _require_mapping(
        inputs["freeze_request"],
        (
            "freeze_request_digest",
            "exact_freeze_request_bytes_sha256",
            "exact_freeze_request_byte_count",
        ),
        name="receipt freeze-request binding",
    )
    _require_sha256(freeze["freeze_request_digest"], name="freeze-request digest")
    _require_sha256(freeze["exact_freeze_request_bytes_sha256"], name="freeze-request sha256")
    _require_integer(freeze["exact_freeze_request_byte_count"], name="freeze byte count", minimum=1)
    registration = _require_mapping(
        inputs["external_registration_receipt"],
        (
            "registration_reference",
            "registration_receipt_digest",
            "exact_registration_receipt_bytes_sha256",
            "exact_registration_receipt_byte_count",
        ),
        name="receipt registration binding",
    )
    _require_identifier(registration["registration_reference"], name="registration reference")
    _require_sha256(registration["registration_receipt_digest"], name="registration digest")
    _require_sha256(registration["exact_registration_receipt_bytes_sha256"], name="registration sha256")
    _require_integer(
        registration["exact_registration_receipt_byte_count"],
        name="registration byte count",
        minimum=1,
    )
    _validate_execution_code_binding_obj(inputs["execution_code_binding"])
    _validate_environment_input_bindings_obj(
        inputs["environment_input_bindings"], require_production=require_production
    )


def parse_evaluation_census_execution_receipt_v1(text: str) -> Mapping[str, Any]:
    """Strictly parse a receipt; this is not a substitute for full-root verification."""

    value = _load_json(text)
    obj = _require_mapping(
        value,
        (
            "schema_version",
            "receipt_kind",
            "status",
            "execution_uuid",
            "execution_nonce",
            "input_binding",
            "external_registration_validation",
            "stage_receipts",
            "report_binding",
            "accounting",
            "runtime_evidence",
            "execution_policy",
            "scientific_claim_boundary",
            "authorization",
            "completion",
            "execution_receipt_digest",
        ),
        name="execution completion receipt",
    )
    if (
        obj["schema_version"] != EVALUATION_CENSUS_EXECUTION_SCHEMA_VERSION
        or obj["receipt_kind"] != _EXECUTION_RECEIPT_KIND
        or obj["status"] != "complete_verified_nonauthorizing"
    ):
        raise EvaluationCensusExecutionV1Error("unknown or incomplete execution receipt")
    _require_uuid(obj["execution_uuid"])
    _require_nonce(obj["execution_nonce"])
    accounting_untyped = _require_mapping(
        obj["accounting"],
        (
            "mirror_attempt_count",
            "opening_record_count",
            "constructed_opening_count",
            "preopening_failure_count",
            "candidate_group_count",
            "complete_census_pair_count",
            "candidate_pool_size",
            "distinct_evaluation_c_identity_count",
            "observed_cell_count",
            "observed_cell_coordinates_digest",
            "production_attempt_budget_complete",
            "all_planned_positions_preserved",
            "all_36_cells_preserved",
            "early_stop_used",
        ),
        name="receipt accounting",
    )
    require_production = _require_boolean(
        accounting_untyped["production_attempt_budget_complete"], name="production completion"
    )
    accounting = _validate_accounting(obj["accounting"], require_production=require_production)
    _validate_execution_receipt_input(obj["input_binding"], require_production=require_production)

    external = _require_mapping(
        obj["external_registration_validation"],
        (
            "expected_registration_reference",
            "expected_registration_receipt_sha256",
            "registered_at_utc",
            "controller_started_at_utc",
            "registered_execution_deadline_seconds",
            "registered_not_after_controller_start_by_supplied_and_local_clocks",
            *_REGISTRATION_CLAIM_BOUNDARY,
        ),
        name="external registration validation",
    )
    _require_identifier(external["expected_registration_reference"], name="registration reference")
    _require_sha256(external["expected_registration_receipt_sha256"], name="registration sha256")
    registered_at = _registration_datetime(_require_timestamp(external["registered_at_utc"]))
    controller_started_at = _wall_datetime(
        external["controller_started_at_utc"], name="controller started wall UTC"
    )
    _require_integer(
        external["registered_execution_deadline_seconds"],
        name="registered deadline seconds",
        minimum=1,
        maximum=86_400,
    )
    if registered_at > controller_started_at or not _require_boolean(
        external["registered_not_after_controller_start_by_supplied_and_local_clocks"],
        name="registration wall-clock ordering",
    ):
        raise EvaluationCensusExecutionV1Error("registration/controller wall-clock order is invalid")
    for key, expected in _REGISTRATION_CLAIM_BOUNDARY.items():
        if _require_boolean(external[key], name=key) is not expected:
            raise EvaluationCensusExecutionV1Error(f"registration claim boundary changed: {key}")

    stages = _require_mapping(
        obj["stage_receipts"],
        (
            "started",
            "builder_guardian_terminal",
            "builder_terminal",
            "verifier_guardian_terminal",
            "verifier_terminal",
        ),
        name="stage receipts",
    )
    for field, filename, digest_field in (
        ("started", STARTED_RECEIPT_FILENAME, None),
        (
            "builder_guardian_terminal",
            BUILDER_GUARDIAN_TERMINAL_FILENAME,
            "watchdog_terminal_digest",
        ),
        ("builder_terminal", BUILDER_TERMINAL_FILENAME, "worker_terminal_digest"),
        (
            "verifier_guardian_terminal",
            VERIFIER_GUARDIAN_TERMINAL_FILENAME,
            "watchdog_terminal_digest",
        ),
        ("verifier_terminal", VERIFIER_TERMINAL_FILENAME, "worker_terminal_digest"),
    ):
        fields = (
            ("filename", "sha256", "byte_count", digest_field)
            if digest_field is not None
            else ("filename", "sha256", "byte_count")
        )
        stage = _require_mapping(stages[field], fields, name=f"{field} stage binding")
        if stage["filename"] != filename:
            raise EvaluationCensusExecutionV1Error(f"{field} filename changed")
        _require_sha256(stage["sha256"], name=f"{field} sha256")
        _require_integer(stage["byte_count"], name=f"{field} byte count", minimum=1)
        if digest_field is not None:
            _require_sha256(stage[digest_field], name=f"{field} terminal digest")

    report = _require_mapping(
        obj["report_binding"],
        (
            "filename",
            "observed_report_digest",
            "exact_report_bytes_sha256",
            "exact_report_byte_count",
            "committed_after_fresh_process_verification",
        ),
        name="receipt report binding",
    )
    if report["filename"] != REPORT_FILENAME:
        raise EvaluationCensusExecutionV1Error("report filename changed")
    _require_sha256(report["observed_report_digest"], name="report digest")
    _require_sha256(report["exact_report_bytes_sha256"], name="report sha256")
    _require_integer(report["exact_report_byte_count"], name="report byte count", minimum=1)
    if not _require_boolean(
        report["committed_after_fresh_process_verification"], name="verified report commit"
    ):
        raise EvaluationCensusExecutionV1Error("report was not committed after verification")

    runtime = _require_mapping(
        obj["runtime_evidence"],
        (
            "contract",
            "builder",
            "builder_watchdog",
            "verifier",
            "verifier_watchdog",
            "controller",
        ),
        name="runtime evidence",
    )
    if runtime["contract"] != (
        "per-worker monotonic elapsed_ns and process-self ru_maxrss normalized "
        "to bytes; descriptive engineering evidence, not a scientific gate"
    ):
        raise EvaluationCensusExecutionV1Error("runtime evidence contract changed")
    controller = _require_mapping(
        runtime["controller"],
        (
            "pid",
            "started_monotonic_ns",
            "ended_monotonic_ns",
            "elapsed_ns",
            "deadline_seconds",
            "deadline_monotonic_ns",
            "started_wall_utc",
            "ended_wall_utc",
        ),
        name="controller runtime",
    )
    _require_integer(controller["pid"], name="controller pid", minimum=1)
    controller_start = _require_integer(
        controller["started_monotonic_ns"], name="controller start", minimum=1
    )
    controller_end = _require_integer(controller["ended_monotonic_ns"], name="controller end", minimum=1)
    controller_elapsed = _require_integer(controller["elapsed_ns"], name="controller elapsed")
    deadline_seconds = _require_integer(
        controller["deadline_seconds"], name="controller deadline seconds", minimum=1, maximum=86_400
    )
    deadline_ns = _require_integer(controller["deadline_monotonic_ns"], name="controller deadline", minimum=1)
    if (
        controller_end < controller_start
        or controller_elapsed != controller_end - controller_start
        or deadline_ns != controller_start + deadline_seconds * 1_000_000_000
        or controller_end > deadline_ns
        or deadline_seconds != external["registered_execution_deadline_seconds"]
    ):
        raise EvaluationCensusExecutionV1Error("controller runtime/deadline arithmetic is inconsistent")
    if _wall_datetime(controller["started_wall_utc"], name="controller started wall UTC") != (
        controller_started_at
    ) or _wall_datetime(controller["ended_wall_utc"], name="controller ended wall UTC") < (
        controller_started_at
    ):
        raise EvaluationCensusExecutionV1Error("controller wall-clock interval is inconsistent")
    for field in ("builder", "verifier"):
        worker = _require_mapping(
            runtime[field],
            ("timing", "maximum_resident_set_size", "process_identity"),
            name=f"{field} runtime",
        )
        _validate_worker_timing(worker["timing"], deadline_monotonic_ns=deadline_ns)
        _validate_worker_rss(worker["maximum_resident_set_size"])
        _validate_process_identity_obj(worker["process_identity"])
    for field in ("builder_watchdog", "verifier_watchdog"):
        watchdog = _require_mapping(
            runtime[field],
            ("timing", "process_identity", "containment"),
            name=f"{field} runtime",
        )
        _validate_worker_timing(watchdog["timing"], deadline_monotonic_ns=deadline_ns)
        _validate_process_identity_obj(watchdog["process_identity"])
        containment = _require_mapping(
            watchdog["containment"],
            (
                "trigger",
                "term_sent",
                "kill_sent",
                "guardian_ready_before_pid_report",
                "worker_lifetime_pipe_eof_observed",
                "worker_reaped_by_owning_watchdog",
                "watchdog_outside_worker_process_group",
            ),
            name=f"{field} containment",
        )
        if containment["trigger"] != "worker_exited" or any(
            _require_boolean(containment[key], name=f"{field} {key}") for key in ("term_sent", "kill_sent")
        ):
            raise EvaluationCensusExecutionV1Error(f"{field} success containment changed")
        if not all(
            _require_boolean(containment[key], name=f"{field} {key}")
            for key in (
                "guardian_ready_before_pid_report",
                "worker_lifetime_pipe_eof_observed",
                "worker_reaped_by_owning_watchdog",
                "watchdog_outside_worker_process_group",
            )
        ):
            raise EvaluationCensusExecutionV1Error(f"{field} containment is incomplete")

    _require_exact_constant(obj["execution_policy"], _EXECUTION_POLICY, name="execution policy")
    _require_exact_constant(
        obj["scientific_claim_boundary"],
        _SCIENTIFIC_CLAIM_BOUNDARY,
        name="scientific claim boundary",
    )
    _require_exact_constant(obj["authorization"], _AUTHORIZATION, name="authorization")
    _require_exact_constant(
        obj["completion"],
        {
            "started_receipt_written_first": True,
            "fresh_builder_succeeded": True,
            "fresh_verifier_succeeded": True,
            "report_committed_before_completion_receipt": True,
            "completion_receipt_written_last": True,
            "failure_receipt_present": False,
        },
        name="completion",
    )
    if accounting != obj["accounting"]:
        raise EvaluationCensusExecutionV1Error("receipt accounting normalization changed")
    unsigned = {key: item for key, item in obj.items() if key != "execution_receipt_digest"}
    if _require_sha256(obj["execution_receipt_digest"], name="execution receipt digest") != _digest(
        unsigned, domain=_EXECUTION_RECEIPT_DOMAIN
    ):
        raise EvaluationCensusExecutionV1Error("execution receipt digest is inconsistent")
    if _canonical_bytes(value).decode("ascii") != text:
        raise EvaluationCensusExecutionV1Error("execution receipt is not canonical newline-terminated JSON")
    return obj


_COMPLETE_ROOT_FILENAMES = (
    STARTED_RECEIPT_FILENAME,
    BUILDER_GUARDIAN_TERMINAL_FILENAME,
    BUILDER_TERMINAL_FILENAME,
    VERIFIER_GUARDIAN_TERMINAL_FILENAME,
    VERIFIER_TERMINAL_FILENAME,
    REPORT_FILENAME,
    EXECUTION_RECEIPT_FILENAME,
)


def _read_complete_execution_root(path: Path) -> tuple[Path, dict[str, bytes]]:
    try:
        root_lstat = path.lstat()
    except FileNotFoundError as exc:
        raise EvaluationCensusExecutionV1Error("execution root is missing") from exc
    if stat.S_ISLNK(root_lstat.st_mode) or not stat.S_ISDIR(root_lstat.st_mode):
        raise EvaluationCensusExecutionV1Error("execution root must be one nonsymlink directory")
    if stat.S_IMODE(root_lstat.st_mode) != 0o700:
        raise EvaluationCensusExecutionV1Error("execution root mode must be exactly 0700")
    root = path.resolve(strict=True)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    directory_fd = os.open(root, flags)
    try:
        names = tuple(sorted(os.listdir(directory_fd)))
        if names != tuple(sorted(_COMPLETE_ROOT_FILENAMES)):
            raise EvaluationCensusExecutionV1Error(
                "execution root has missing, extra, failure, or partial-spool artifacts"
            )
        payloads: dict[str, bytes] = {}
        inodes: set[tuple[int, int]] = set()
        mtimes: dict[str, int] = {}
        for name in _COMPLETE_ROOT_FILENAMES:
            metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or stat.S_IMODE(metadata.st_mode) != 0o400
                or metadata.st_nlink != 1
            ):
                raise EvaluationCensusExecutionV1Error(
                    f"execution artifact {name} must be regular, mode 0400, and singly linked"
                )
            identity = (metadata.st_dev, metadata.st_ino)
            if identity in inodes:
                raise EvaluationCensusExecutionV1Error("execution artifacts share an inode")
            inodes.add(identity)
            entry_flags = os.O_RDONLY
            if hasattr(os, "O_NOFOLLOW"):
                entry_flags |= os.O_NOFOLLOW
            descriptor = os.open(name, entry_flags, dir_fd=directory_fd)
            try:
                opened = os.fstat(descriptor)
                if (opened.st_dev, opened.st_ino, opened.st_size) != (
                    metadata.st_dev,
                    metadata.st_ino,
                    metadata.st_size,
                ):
                    raise EvaluationCensusExecutionV1Error(f"execution artifact {name} changed while opened")
                maximum = 256 * 1024 * 1024 if name == REPORT_FILENAME else 8 * 1024 * 1024
                if opened.st_size <= 0 or opened.st_size > maximum:
                    raise EvaluationCensusExecutionV1Error(
                        f"execution artifact {name} has an invalid byte count"
                    )
                chunks: list[bytes] = []
                remaining = opened.st_size
                while remaining:
                    chunk = os.read(descriptor, min(remaining, 1024 * 1024))
                    if not chunk:
                        raise EvaluationCensusExecutionV1Error(
                            f"execution artifact {name} changed during read"
                        )
                    chunks.append(chunk)
                    remaining -= len(chunk)
                if os.read(descriptor, 1):
                    raise EvaluationCensusExecutionV1Error(f"execution artifact {name} grew during read")
                payloads[name] = b"".join(chunks)
                mtimes[name] = opened.st_mtime_ns
            finally:
                os.close(descriptor)
        completion_mtime = mtimes[EXECUTION_RECEIPT_FILENAME]
        if any(
            completion_mtime < mtime for name, mtime in mtimes.items() if name != EXECUTION_RECEIPT_FILENAME
        ):
            raise EvaluationCensusExecutionV1Error(
                "completion receipt metadata predates another execution artifact"
            )
        return root, payloads
    finally:
        os.close(directory_fd)


def _ascii_payload(payload: bytes, *, name: str) -> str:
    try:
        return payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise EvaluationCensusExecutionV1Error(f"{name} is not ASCII") from exc


def _parse_started_receipt_for_root(
    payload: bytes,
    *,
    expected_execution_uuid: str,
    expected_execution_nonce: str,
    expected_input_binding: Mapping[str, Any],
    expected_registration_timestamp: str,
    expected_deadline_seconds: int,
    expected_controller_start_utc: str,
    runner_source_path: Path,
) -> Mapping[str, Any]:
    text = _ascii_payload(payload, name="started receipt")
    value = _load_json(text)
    obj = _require_mapping(
        value,
        (
            "schema_version",
            "receipt_kind",
            "status",
            "execution_uuid",
            "execution_nonce",
            "wall_clock",
            "controller_process_identity",
            "worker_commands",
            "deadline",
            "input_binding",
            "execution_policy",
            "authorization",
            "started_receipt_digest",
        ),
        name="started receipt",
    )
    if (
        obj["schema_version"] != EVALUATION_CENSUS_EXECUTION_SCHEMA_VERSION
        or obj["receipt_kind"] != _STARTED_KIND
        or obj["status"] != "started_no_retry"
        or _require_uuid(obj["execution_uuid"]) != expected_execution_uuid
        or _require_nonce(obj["execution_nonce"]) != expected_execution_nonce
    ):
        raise EvaluationCensusExecutionV1Error("started receipt identity changed")
    wall = _require_mapping(
        obj["wall_clock"],
        (
            "registered_at_utc",
            "controller_started_at_utc",
            "registered_not_after_controller_start_by_supplied_and_local_clocks",
            "clock_or_service_authenticity_independently_verified",
        ),
        name="started wall clock",
    )
    if (
        wall["registered_at_utc"] != expected_registration_timestamp
        or wall["controller_started_at_utc"] != expected_controller_start_utc
        or not _require_boolean(
            wall["registered_not_after_controller_start_by_supplied_and_local_clocks"],
            name="started wall ordering",
        )
        or _require_boolean(
            wall["clock_or_service_authenticity_independently_verified"],
            name="started clock authenticity",
        )
    ):
        raise EvaluationCensusExecutionV1Error("started wall-clock boundary changed")
    controller = _require_mapping(
        obj["controller_process_identity"],
        ("pid", "argv", "working_directory", "host", "python", "worker_environment"),
        name="started controller identity",
    )
    _require_integer(controller["pid"], name="started controller pid", minimum=1)
    if type(controller["argv"]) is not list or not controller["argv"]:
        raise EvaluationCensusExecutionV1Error("started controller argv is invalid")
    entrypoint = (
        Path(cast(str, controller["working_directory"])) / cast(list[str], controller["argv"])[0]
    ).resolve(strict=False)
    if (
        entrypoint != runner_source_path.resolve(strict=True)
        and cast(list[str], controller["argv"])[0] != "engineering-test-api"
    ):
        raise EvaluationCensusExecutionV1Error("started controller is not the bound runner")
    _require_exact_constant(controller["host"], _host_identity(), name="started host")
    _require_exact_constant(controller["python"], _python_identity(), name="started Python")
    _require_exact_constant(
        controller["worker_environment"], _safe_environment(), name="started worker environment"
    )
    _require_mapping(
        obj["worker_commands"],
        (
            "builder_argv",
            "verifier_argv_prefix",
            "builder_watchdog_argv",
            "verifier_watchdog_argv_prefix",
        ),
        name="started worker commands",
    )
    deadline = _require_mapping(
        obj["deadline"],
        ("deadline_seconds", "started_monotonic_ns", "deadline_monotonic_ns"),
        name="started deadline",
    )
    started_ns = _require_integer(deadline["started_monotonic_ns"], name="started monotonic", minimum=1)
    if (
        _require_integer(deadline["deadline_seconds"], name="started deadline seconds")
        != expected_deadline_seconds
        or _require_integer(deadline["deadline_monotonic_ns"], name="started deadline", minimum=1)
        != started_ns + expected_deadline_seconds * 1_000_000_000
    ):
        raise EvaluationCensusExecutionV1Error("started deadline is inconsistent")
    if obj["input_binding"] != expected_input_binding:
        raise EvaluationCensusExecutionV1Error("started input binding differs")
    _require_exact_constant(obj["execution_policy"], _EXECUTION_POLICY, name="execution policy")
    _require_exact_constant(obj["authorization"], _AUTHORIZATION, name="authorization")
    unsigned = {key: item for key, item in obj.items() if key != "started_receipt_digest"}
    if (
        _require_sha256(obj["started_receipt_digest"], name="started digest")
        != _digest(unsigned, domain=_STARTED_DOMAIN)
        or _canonical_bytes(value) != payload
    ):
        raise EvaluationCensusExecutionV1Error("started receipt digest or canonical bytes differ")
    return obj


@dataclass(frozen=True, slots=True)
class VerifiedEvaluationCensusExecutionRootV1:
    output_root: Path
    report_digest: str
    report_bytes_sha256: str
    execution_receipt_digest: str
    fresh_replay_terminal_digest: str
    fresh_replay_watchdog_terminal_digest: str


def _verify_execution_root(
    *,
    output_root: Path,
    plan_path: Path,
    freeze_request_path: Path,
    registration_receipt_path: Path,
    expected_plan_digest: str,
    expected_plan_bytes_sha256: str,
    expected_registration_receipt_sha256: str,
    expected_registration_reference: str,
    expected_execution_uuid: str,
    expected_execution_nonce: str,
    expected_execution_deadline_seconds: int,
    expected_execution_receipt_sha256: str,
    expected_execution_receipt_digest: str,
    runner_source_path: Path,
    require_production: bool,
    verification_deadline_seconds: int,
    source_archive_path: Path | None,
    constraints_path: Path | None,
    environment_lock_path: Path | None,
) -> VerifiedEvaluationCensusExecutionRootV1:
    plan_digest = _require_sha256(expected_plan_digest, name="expected plan digest")
    plan_sha = _require_sha256(expected_plan_bytes_sha256, name="expected plan sha256")
    registration_sha = _require_sha256(
        expected_registration_receipt_sha256, name="expected registration sha256"
    )
    completion_sha = _require_sha256(
        expected_execution_receipt_sha256, name="expected execution receipt sha256"
    )
    completion_digest = _require_sha256(
        expected_execution_receipt_digest, name="expected execution receipt digest"
    )
    execution_uuid = _require_uuid(expected_execution_uuid)
    execution_nonce = _require_nonce(expected_execution_nonce)
    registered_deadline = _require_integer(
        expected_execution_deadline_seconds,
        name="registered execution deadline seconds",
        minimum=1,
        maximum=86_400,
    )
    verification_deadline = _require_integer(
        verification_deadline_seconds,
        name="verification deadline seconds",
        minimum=1,
        maximum=86_400,
    )
    if require_production:
        _assert_production_environment_files(
            source_archive_path=source_archive_path,
            constraints_path=constraints_path,
            environment_lock_path=environment_lock_path,
        )
    root, payloads = _read_complete_execution_root(output_root)
    if root.name != execution_uuid:
        raise EvaluationCensusExecutionV1Error("execution-root basename differs from registered UUID")
    receipt_bytes = payloads[EXECUTION_RECEIPT_FILENAME]
    if _sha256_bytes(receipt_bytes) != completion_sha:
        raise EvaluationCensusExecutionV1Error(
            "completion-receipt bytes differ from the externally expected SHA-256"
        )
    receipt = parse_evaluation_census_execution_receipt_v1(
        _ascii_payload(receipt_bytes, name="execution receipt")
    )
    if (
        receipt["execution_receipt_digest"] != completion_digest
        or receipt["execution_uuid"] != execution_uuid
        or receipt["execution_nonce"] != execution_nonce
    ):
        raise EvaluationCensusExecutionV1Error("completion receipt identity differs from expected")

    plan_text, plan_bytes = _read_ascii(plan_path, name="prospective plan", maximum_bytes=8 * 1024 * 1024)
    if _sha256_bytes(plan_bytes) != plan_sha:
        raise EvaluationCensusExecutionV1Error("root verifier plan SHA differs")
    plan = parse_evaluation_census_plan_v1(plan_text, expected_digest=plan_digest)
    if require_production:
        _assert_production_plan(plan)
    elif plan.uses_production_attempt_budget:
        raise EvaluationCensusExecutionV1Error("engineering verifier received production plan")
    freeze_text, freeze_bytes = _read_ascii(
        freeze_request_path, name="freeze request", maximum_bytes=2 * 1024 * 1024
    )
    request = parse_evaluation_census_freeze_request_v1(
        freeze_text,
        plan=plan,
        plan_text=plan_text,
        runner_source_path=runner_source_path,
        source_archive_path=source_archive_path,
        constraints_path=constraints_path,
        environment_lock_path=environment_lock_path,
    )
    expected_class = _PRODUCTION_PLAN_CLASS if require_production else _ENGINEERING_PLAN_CLASS
    if request.plan_class != expected_class:
        raise EvaluationCensusExecutionV1Error("root verifier request class differs")
    registration_text, registration_bytes = _read_ascii(
        registration_receipt_path,
        name="external registration receipt",
        maximum_bytes=2 * 1024 * 1024,
    )
    registration = parse_external_evaluation_census_registration_receipt_v1(
        registration_text,
        freeze_request=request,
        freeze_request_text=freeze_text,
        expected_bytes_sha256=registration_sha,
        expected_registration_reference=expected_registration_reference,
        expected_execution_uuid=execution_uuid,
        expected_execution_nonce=execution_nonce,
        expected_execution_deadline_seconds=registered_deadline,
    )
    input_binding = {
        "prospective_plan": _plan_binding(plan, plan_bytes),
        "freeze_request": {
            "freeze_request_digest": request.digest,
            "exact_freeze_request_bytes_sha256": _sha256_bytes(freeze_bytes),
            "exact_freeze_request_byte_count": len(freeze_bytes),
        },
        "external_registration_receipt": {
            "registration_reference": registration.registration_reference,
            "registration_receipt_digest": registration.digest,
            "exact_registration_receipt_bytes_sha256": _sha256_bytes(registration_bytes),
            "exact_registration_receipt_byte_count": len(registration_bytes),
        },
        "execution_code_binding": request.execution_code_binding,
        "environment_input_bindings": request.environment_input_bindings,
    }
    if receipt["input_binding"] != input_binding:
        raise EvaluationCensusExecutionV1Error("completion input binding differs from rederivation")
    external = cast(Mapping[str, Any], receipt["external_registration_validation"])
    if (
        external["expected_registration_reference"] != expected_registration_reference
        or external["expected_registration_receipt_sha256"] != registration_sha
        or external["registered_at_utc"] != registration.registered_at_utc
        or external["registered_execution_deadline_seconds"] != registered_deadline
    ):
        raise EvaluationCensusExecutionV1Error("completion registration validation differs")
    started = _parse_started_receipt_for_root(
        payloads[STARTED_RECEIPT_FILENAME],
        expected_execution_uuid=execution_uuid,
        expected_execution_nonce=execution_nonce,
        expected_input_binding=input_binding,
        expected_registration_timestamp=registration.registered_at_utc,
        expected_deadline_seconds=registered_deadline,
        expected_controller_start_utc=cast(str, external["controller_started_at_utc"]),
        runner_source_path=runner_source_path,
    )
    stages = cast(Mapping[str, Mapping[str, Any]], receipt["stage_receipts"])
    for field, filename in (
        ("started", STARTED_RECEIPT_FILENAME),
        ("builder_guardian_terminal", BUILDER_GUARDIAN_TERMINAL_FILENAME),
        ("builder_terminal", BUILDER_TERMINAL_FILENAME),
        ("verifier_guardian_terminal", VERIFIER_GUARDIAN_TERMINAL_FILENAME),
        ("verifier_terminal", VERIFIER_TERMINAL_FILENAME),
    ):
        if stages[field]["sha256"] != _sha256_bytes(payloads[filename]) or stages[field]["byte_count"] != len(
            payloads[filename]
        ):
            raise EvaluationCensusExecutionV1Error(f"{field} file differs from completion binding")

    commands = cast(Mapping[str, Any], started["worker_commands"])
    builder_argv = cast(list[str], commands["builder_argv"])
    builder_watchdog_argv = cast(list[str], commands["builder_watchdog_argv"])
    builder_watchdog_value = _load_json(
        _ascii_payload(
            payloads[BUILDER_GUARDIAN_TERMINAL_FILENAME],
            name="builder guardian terminal",
        )
    )
    builder_watchdog_obj = cast(Mapping[str, Any], builder_watchdog_value)
    builder_watchdog_process = cast(Mapping[str, Any], builder_watchdog_obj["watchdog_process_identity"])
    builder_worker_payload, builder_pid, parsed_builder_watchdog = _parse_watchdog_terminal(
        payloads[BUILDER_GUARDIAN_TERMINAL_FILENAME],
        expected_pid=_require_integer(
            builder_watchdog_process["pid"], name="stored builder watchdog pid", minimum=1
        ),
        expected_watchdog_argv=builder_watchdog_argv,
        expected_worker_argv=builder_argv,
        expected_deadline_monotonic_ns=cast(Mapping[str, Any], started["deadline"])["deadline_monotonic_ns"],
    )
    if builder_worker_payload != payloads[BUILDER_TERMINAL_FILENAME]:
        raise EvaluationCensusExecutionV1Error("builder terminal differs from watchdog-bound bytes")
    builder_terminal = _parse_worker_terminal(
        builder_worker_payload,
        expected_kind=_BUILDER_TERMINAL_KIND,
        expected_pid=builder_pid,
        expected_argv=builder_argv,
        expected_deadline_monotonic_ns=cast(Mapping[str, Any], started["deadline"])["deadline_monotonic_ns"],
        expected_plan_digest=plan_digest,
        expected_plan_sha256=plan_sha,
        expected_frozen_byte_revalidation={
            "execution_code_binding": request.execution_code_binding,
            "environment_input_bindings": request.environment_input_bindings,
        },
        require_production=require_production,
    )
    verifier_watchdog_value = _load_json(
        _ascii_payload(
            payloads[VERIFIER_GUARDIAN_TERMINAL_FILENAME],
            name="verifier guardian terminal",
        )
    )
    verifier_watchdog_obj = cast(Mapping[str, Any], verifier_watchdog_value)
    verifier_worker_argv = cast(
        list[str], cast(Mapping[str, Any], verifier_watchdog_obj["worker_process"])["argv"]
    )
    verifier_watchdog_argv = [
        *cast(list[str], commands["verifier_watchdog_argv_prefix"]),
        _dump_json(verifier_worker_argv),
    ]
    verifier_watchdog_process = cast(Mapping[str, Any], verifier_watchdog_obj["watchdog_process_identity"])
    verifier_worker_payload, verifier_pid, parsed_verifier_watchdog = _parse_watchdog_terminal(
        payloads[VERIFIER_GUARDIAN_TERMINAL_FILENAME],
        expected_pid=_require_integer(
            verifier_watchdog_process["pid"], name="stored verifier watchdog pid", minimum=1
        ),
        expected_watchdog_argv=verifier_watchdog_argv,
        expected_worker_argv=verifier_worker_argv,
        expected_deadline_monotonic_ns=cast(Mapping[str, Any], started["deadline"])["deadline_monotonic_ns"],
    )
    if verifier_worker_payload != payloads[VERIFIER_TERMINAL_FILENAME]:
        raise EvaluationCensusExecutionV1Error("verifier terminal differs from watchdog-bound bytes")
    report_binding = cast(Mapping[str, Any], receipt["report_binding"])
    report_digest = _require_sha256(report_binding["observed_report_digest"], name="report digest")
    report_sha = _require_sha256(report_binding["exact_report_bytes_sha256"], name="report sha256")
    report_bytes = payloads[REPORT_FILENAME]
    if (
        _sha256_bytes(report_bytes) != report_sha
        or len(report_bytes) != report_binding["exact_report_byte_count"]
    ):
        raise EvaluationCensusExecutionV1Error("committed report differs from completion binding")
    verifier_terminal = _parse_worker_terminal(
        verifier_worker_payload,
        expected_kind=_VERIFIER_TERMINAL_KIND,
        expected_pid=verifier_pid,
        expected_argv=verifier_worker_argv,
        expected_deadline_monotonic_ns=cast(Mapping[str, Any], started["deadline"])["deadline_monotonic_ns"],
        expected_plan_digest=plan_digest,
        expected_plan_sha256=plan_sha,
        expected_frozen_byte_revalidation={
            "execution_code_binding": request.execution_code_binding,
            "environment_input_bindings": request.environment_input_bindings,
        },
        require_production=require_production,
        expected_report_digest=report_digest,
        expected_report_sha256=report_sha,
    )
    if (
        builder_terminal["report_binding"] != verifier_terminal["report_binding"]
        or builder_terminal["accounting"] != verifier_terminal["accounting"]
        or receipt["accounting"] != builder_terminal["accounting"]
        or stages["builder_guardian_terminal"]["watchdog_terminal_digest"]
        != parsed_builder_watchdog["watchdog_terminal_digest"]
        or stages["builder_terminal"]["worker_terminal_digest"] != builder_terminal["worker_terminal_digest"]
        or stages["verifier_guardian_terminal"]["watchdog_terminal_digest"]
        != parsed_verifier_watchdog["watchdog_terminal_digest"]
        or stages["verifier_terminal"]["worker_terminal_digest"]
        != verifier_terminal["worker_terminal_digest"]
    ):
        raise EvaluationCensusExecutionV1Error("stage/report/accounting cross-bindings differ")
    runtime = cast(Mapping[str, Any], receipt["runtime_evidence"])
    if runtime["builder"] != {
        "timing": builder_terminal["timing"],
        "maximum_resident_set_size": builder_terminal["maximum_resident_set_size"],
        "process_identity": builder_terminal["process_identity"],
    } or runtime["verifier"] != {
        "timing": verifier_terminal["timing"],
        "maximum_resident_set_size": verifier_terminal["maximum_resident_set_size"],
        "process_identity": verifier_terminal["process_identity"],
    }:
        raise EvaluationCensusExecutionV1Error("receipt worker runtime differs from stage terminals")
    if runtime["builder_watchdog"] != {
        "timing": parsed_builder_watchdog["timing"],
        "process_identity": parsed_builder_watchdog["watchdog_process_identity"],
        "containment": parsed_builder_watchdog["containment"],
    } or runtime["verifier_watchdog"] != {
        "timing": parsed_verifier_watchdog["timing"],
        "process_identity": parsed_verifier_watchdog["watchdog_process_identity"],
        "containment": parsed_verifier_watchdog["containment"],
    }:
        raise EvaluationCensusExecutionV1Error("receipt watchdog runtime differs from guardian terminals")
    controller_runtime = cast(Mapping[str, Any], runtime["controller"])
    started_controller = cast(Mapping[str, Any], started["controller_process_identity"])
    started_deadline = cast(Mapping[str, Any], started["deadline"])
    if (
        controller_runtime["pid"] != started_controller["pid"]
        or controller_runtime["started_monotonic_ns"] != started_deadline["started_monotonic_ns"]
        or controller_runtime["deadline_seconds"] != started_deadline["deadline_seconds"]
        or controller_runtime["deadline_monotonic_ns"] != started_deadline["deadline_monotonic_ns"]
        or controller_runtime["started_wall_utc"]
        != cast(Mapping[str, Any], started["wall_clock"])["controller_started_at_utc"]
    ):
        raise EvaluationCensusExecutionV1Error("receipt controller runtime differs from started receipt")

    # A third process, created by a separate watchdog, replays the report after
    # every durable artifact and external completion pin have been checked.
    verification_started = time.monotonic_ns()
    verification_deadline_ns = verification_started + verification_deadline * 1_000_000_000
    common = [
        str(Path(sys.executable).resolve(strict=True)),
        "-m",
        "goalzendo_interactive_v2.evaluation_census_execution",
    ]
    replay_parent_pipe = _ParentDeathPipe.create()
    replay_argv = [
        *common,
        "__verify-worker",
        "--plan",
        str(plan_path.resolve(strict=True)),
        "--expected-plan-digest",
        plan_digest,
        "--expected-plan-sha256",
        plan_sha,
        "--report-spool",
        str(root / REPORT_FILENAME),
        "--runner-source",
        str(runner_source_path.resolve(strict=True)),
        "--expected-execution-code-binding-json",
        _dump_json(dict(request.execution_code_binding)),
        "--expected-environment-input-bindings-json",
        _dump_json(dict(request.environment_input_bindings)),
        "--expected-report-digest",
        report_digest,
        "--expected-report-sha256",
        report_sha,
        "--deadline-monotonic-ns",
        str(verification_deadline_ns),
        "--parent-guard-fd",
        str(replay_parent_pipe.worker_read_fd),
        "--guardian-ready-write-fd",
        str(replay_parent_pipe.guardian_ready_write_fd),
        "--worker-lifetime-write-fd",
        str(replay_parent_pipe.worker_lifetime_write_fd),
    ]
    for option, optional_path in (
        ("--source-archive", source_archive_path),
        ("--constraints", constraints_path),
        ("--environment-lock", environment_lock_path),
    ):
        if optional_path is not None:
            replay_argv.extend((option, str(optional_path.resolve(strict=True))))
    if require_production:
        replay_argv.append("--require-production")
    replay_watchdog_argv = [
        *common,
        "__watchdog-worker",
        "--parent-death-fd",
        str(replay_parent_pipe.read_fd),
        "--worker-death-read-fd",
        str(replay_parent_pipe.worker_read_fd),
        "--worker-death-write-fd",
        str(replay_parent_pipe.worker_write_fd),
        "--worker-pid-report-fd",
        str(replay_parent_pipe.pid_write_fd),
        "--guardian-ready-read-fd",
        str(replay_parent_pipe.guardian_ready_read_fd),
        "--guardian-ready-write-fd",
        str(replay_parent_pipe.guardian_ready_write_fd),
        "--worker-lifetime-read-fd",
        str(replay_parent_pipe.worker_lifetime_read_fd),
        "--worker-lifetime-write-fd",
        str(replay_parent_pipe.worker_lifetime_write_fd),
        "--deadline-monotonic-ns",
        str(verification_deadline_ns),
        "--worker-argv-json",
        _dump_json(replay_argv),
    ]
    replay_watchdog_payload, replay_watchdog_pid, reported_replay_pid = _run_worker(
        replay_watchdog_argv,
        deadline_monotonic_ns=verification_deadline_ns,
        parent_death_pipe=replay_parent_pipe,
    )
    replay_payload, replay_pid, replay_watchdog = _parse_watchdog_terminal(
        replay_watchdog_payload,
        expected_pid=replay_watchdog_pid,
        expected_watchdog_argv=replay_watchdog_argv,
        expected_worker_argv=replay_argv,
        expected_deadline_monotonic_ns=verification_deadline_ns,
    )
    if replay_pid != reported_replay_pid:
        raise EvaluationCensusExecutionV1Error("replay PID differs from early watchdog report")
    replay_terminal = _parse_worker_terminal(
        replay_payload,
        expected_kind=_VERIFIER_TERMINAL_KIND,
        expected_pid=replay_pid,
        expected_argv=replay_argv,
        expected_deadline_monotonic_ns=verification_deadline_ns,
        expected_plan_digest=plan_digest,
        expected_plan_sha256=plan_sha,
        expected_frozen_byte_revalidation={
            "execution_code_binding": request.execution_code_binding,
            "environment_input_bindings": request.environment_input_bindings,
        },
        require_production=require_production,
        expected_report_digest=report_digest,
        expected_report_sha256=report_sha,
    )
    if replay_terminal["accounting"] != receipt["accounting"]:
        raise EvaluationCensusExecutionV1Error("third fresh replay accounting differs")
    root_after_replay, payloads_after_replay = _read_complete_execution_root(root)
    if root_after_replay != root or payloads_after_replay != payloads:
        raise EvaluationCensusExecutionV1Error("execution root changed during the third fresh replay")
    return VerifiedEvaluationCensusExecutionRootV1(
        root,
        report_digest,
        report_sha,
        completion_digest,
        cast(str, replay_terminal["worker_terminal_digest"]),
        cast(str, replay_watchdog["watchdog_terminal_digest"]),
    )


def verify_production_evaluation_census_execution_root_v1(
    *,
    output_root: Path,
    plan_path: Path,
    freeze_request_path: Path,
    registration_receipt_path: Path,
    expected_plan_digest: str,
    expected_plan_bytes_sha256: str,
    expected_registration_receipt_sha256: str,
    expected_registration_reference: str,
    expected_execution_uuid: str,
    expected_execution_nonce: str,
    expected_execution_deadline_seconds: int,
    expected_execution_receipt_sha256: str,
    expected_execution_receipt_digest: str,
    runner_source_path: Path,
    verification_deadline_seconds: int,
    source_archive_path: Path,
    constraints_path: Path,
    environment_lock_path: Path,
) -> VerifiedEvaluationCensusExecutionRootV1:
    """Verify an externally pinned production root and replay it in a third process."""

    return _verify_execution_root(
        output_root=output_root,
        plan_path=plan_path,
        freeze_request_path=freeze_request_path,
        registration_receipt_path=registration_receipt_path,
        expected_plan_digest=expected_plan_digest,
        expected_plan_bytes_sha256=expected_plan_bytes_sha256,
        expected_registration_receipt_sha256=expected_registration_receipt_sha256,
        expected_registration_reference=expected_registration_reference,
        expected_execution_uuid=expected_execution_uuid,
        expected_execution_nonce=expected_execution_nonce,
        expected_execution_deadline_seconds=expected_execution_deadline_seconds,
        expected_execution_receipt_sha256=expected_execution_receipt_sha256,
        expected_execution_receipt_digest=expected_execution_receipt_digest,
        runner_source_path=runner_source_path,
        require_production=True,
        verification_deadline_seconds=verification_deadline_seconds,
        source_archive_path=source_archive_path,
        constraints_path=constraints_path,
        environment_lock_path=environment_lock_path,
    )


def verify_engineering_evaluation_census_execution_root_for_testing_v1(
    *,
    output_root: Path,
    plan_path: Path,
    freeze_request_path: Path,
    registration_receipt_path: Path,
    expected_plan_digest: str,
    expected_plan_bytes_sha256: str,
    expected_registration_receipt_sha256: str,
    expected_registration_reference: str,
    expected_execution_uuid: str,
    expected_execution_nonce: str,
    expected_execution_deadline_seconds: int,
    expected_execution_receipt_sha256: str,
    expected_execution_receipt_digest: str,
    runner_source_path: Path,
    verification_deadline_seconds: int = 600,
) -> VerifiedEvaluationCensusExecutionRootV1:
    """Verify a reduced test root; deliberately unavailable from the production CLI."""

    return _verify_execution_root(
        output_root=output_root,
        plan_path=plan_path,
        freeze_request_path=freeze_request_path,
        registration_receipt_path=registration_receipt_path,
        expected_plan_digest=expected_plan_digest,
        expected_plan_bytes_sha256=expected_plan_bytes_sha256,
        expected_registration_receipt_sha256=expected_registration_receipt_sha256,
        expected_registration_reference=expected_registration_reference,
        expected_execution_uuid=expected_execution_uuid,
        expected_execution_nonce=expected_execution_nonce,
        expected_execution_deadline_seconds=expected_execution_deadline_seconds,
        expected_execution_receipt_sha256=expected_execution_receipt_sha256,
        expected_execution_receipt_digest=expected_execution_receipt_digest,
        runner_source_path=runner_source_path,
        require_production=False,
        verification_deadline_seconds=verification_deadline_seconds,
        source_archive_path=None,
        constraints_path=None,
        environment_lock_path=None,
    )


def _module_main(argv: Sequence[str] | None = None) -> int:
    args = _worker_parser().parse_args(argv)
    try:
        if args.worker_action == "__build-worker":
            return _worker_build(args)
        if args.worker_action == "__verify-worker":
            return _worker_verify(args)
        if args.worker_action == "__watchdog-worker":
            return _watchdog_worker(args)
        if args.worker_action == "__containment-probe-worker":
            return _containment_probe_worker(args)
    except (EvaluationCensusExecutionV1Error, EvaluationCensusV2Error) as exc:
        sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
        return 2
    raise AssertionError("unreachable worker action")


if __name__ == "__main__":
    raise SystemExit(_module_main())
