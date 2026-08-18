"""Fail-closed lifecycle for the G03-v2 nested-opening feasibility report.

Preparation persists the exact upstream identity plan, the exact nested plan,
and an outcome-free freeze request.  Execution is a later, distinct action
which requires an externally supplied content-pinned registration receipt.
Every artifact is CPU-only and nonauthorizing: no bank, quartet, matcher,
runtime, model, or launch authority is created here.

The registration receipt schema is locally checkable interchange only.  This
repository cannot authenticate the named registrar, its issuance, or temporal
priority.  Test-only reduced artifacts use nominally distinct public APIs and
are deliberately unavailable from the production runner.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
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
    build_evaluation_census_plan_v1,
    parse_evaluation_census_plan_v1,
    serialize_evaluation_census_plan_v1,
)
from .nested_opening_feasibility import (
    CANDIDATE_WINDOW_K,
    COMMON_UNION_SLOT_COUNT,
    DERIVED_OPENINGS_PER_MEMBER,
    MEMBERS_PER_ATTEMPT,
    PRODUCTION_MEMBER_CONSTRUCTION_COUNT,
    PRODUCTION_MIRROR_ATTEMPT_COUNT,
    PRODUCTION_OPENING_ASSESSMENT_COUNT,
    SUPPORTED_VERSION_SPACE_FLOOR,
    NestedOpeningFeasibilityPlanV1,
    NestedOpeningFeasibilityReportV1,
    build_nested_opening_construction_feasibility_plan_for_testing_v1,
    build_nested_opening_construction_feasibility_plan_v1,
    build_nested_opening_construction_feasibility_report_for_testing_v1,
    build_nested_opening_construction_feasibility_report_v1,
    derive_nested_opening_construction_feasibility_plan_digest_for_testing_v1,
    derive_nested_opening_construction_feasibility_plan_digest_v1,
    derive_nested_opening_construction_feasibility_report_digest_for_testing_v1,
    derive_nested_opening_construction_feasibility_report_digest_v1,
    parse_nested_opening_construction_feasibility_plan_for_testing_v1,
    parse_nested_opening_construction_feasibility_plan_v1,
    parse_nested_opening_construction_feasibility_report_for_testing_v1,
    parse_nested_opening_construction_feasibility_report_v1,
    serialize_nested_opening_construction_feasibility_plan_for_testing_v1,
    serialize_nested_opening_construction_feasibility_plan_v1,
    serialize_nested_opening_construction_feasibility_report_for_testing_v1,
    serialize_nested_opening_construction_feasibility_report_v1,
)

NESTED_OPENING_FEASIBILITY_EXECUTION_SCHEMA_VERSION = 1

UPSTREAM_CENSUS_PLAN_FILENAME = "upstream-evaluation-census-plan.json"
NESTED_PLAN_FILENAME = "nested-opening-feasibility-plan.json"
FREEZE_REQUEST_FILENAME = "freeze-request.json"
STARTED_RECEIPT_FILENAME = "execution-started.json"
BUILDER_TERMINAL_FILENAME = "builder-terminal.json"
VERIFIER_TERMINAL_FILENAME = "verifier-terminal.json"
BUILDER_GUARDIAN_TERMINAL_FILENAME = "builder-guardian-terminal.json"
VERIFIER_GUARDIAN_TERMINAL_FILENAME = "verifier-guardian-terminal.json"
REPORT_FILENAME = "observed-report.json"
EXECUTION_RECEIPT_FILENAME = "execution-receipt.json"
FAILURE_RECEIPT_FILENAME = "execution-failure.json"

_PREFIX = "g03-v2-nested-opening-construction-feasibility"
_DOMAIN_PREFIX = "goalzendo-interactive-v2-nested-opening-construction-feasibility"
_FREEZE_REQUEST_KIND = f"{_PREFIX}-freeze-request-v1"
_EXTERNAL_REGISTRATION_KIND = f"{_PREFIX}-external-registration-receipt-v1"
_STARTED_KIND = f"{_PREFIX}-execution-started-v1"
_BUILDER_TERMINAL_KIND = f"{_PREFIX}-builder-terminal-v1"
_VERIFIER_TERMINAL_KIND = f"{_PREFIX}-verifier-terminal-v1"
_WATCHDOG_TERMINAL_KIND = f"{_PREFIX}-watchdog-terminal-v1"
_EXECUTION_RECEIPT_KIND = f"{_PREFIX}-execution-receipt-v1"
_FAILURE_RECEIPT_KIND = f"{_PREFIX}-execution-failure-v1"
_ROOT_VERIFICATION_KIND = f"{_PREFIX}-engineering-root-verification-v1"

_FREEZE_REQUEST_DOMAIN = f"{_DOMAIN_PREFIX}-freeze-request-v1"
_EXTERNAL_REGISTRATION_DOMAIN = f"{_DOMAIN_PREFIX}-external-registration-receipt-v1"
_STARTED_DOMAIN = f"{_DOMAIN_PREFIX}-execution-started-v1"
_WORKER_TERMINAL_DOMAIN = f"{_DOMAIN_PREFIX}-worker-terminal-v1"
_WATCHDOG_TERMINAL_DOMAIN = f"{_DOMAIN_PREFIX}-watchdog-terminal-v1"
_EXECUTION_RECEIPT_DOMAIN = f"{_DOMAIN_PREFIX}-execution-receipt-v1"
_FAILURE_RECEIPT_DOMAIN = f"{_DOMAIN_PREFIX}-execution-failure-v1"
_ROOT_VERIFICATION_DOMAIN = f"{_DOMAIN_PREFIX}-engineering-root-verification-v1"
_CELL_COORDINATE_DOMAIN = f"{_DOMAIN_PREFIX}-cell-coordinates-v1"

PRODUCTION_EXECUTION_REFUSAL_CODE = (
    "G03_V2_NESTED_RECONSTRUCTION_CONTRACT_NOT_VERIFIED"
)

_RUNNER_RELATIVE_PATH = "scripts/run_g03_v2_nested_opening_feasibility.py"
_MODULE_RELATIVE_PATH = (
    "src/goalzendo_interactive_v2/nested_opening_feasibility_execution.py"
)
_UPSTREAM_MODULE_RELATIVE_PATH = "src/goalzendo_interactive_v2/evaluation_census.py"
_NESTED_MODULE_RELATIVE_PATH = "src/goalzendo_interactive_v2/nested_opening_feasibility.py"
_PACKAGE_INIT_RELATIVE_PATH = "src/goalzendo_interactive_v2/__init__.py"

_BOOTSTRAP_MARKER = "G03_V2_NESTED_GUARDIAN_BOOTSTRAPPED"
_DISABLED_PYCACHE_PREFIX = (
    "/dev/null/g03-v2-nested-opening-feasibility-pycache-disabled"
)
_BOOTSTRAP_SOURCE = r'''import json,os,runpy,signal,sys,threading
role=sys.argv[1]
if role not in ("--g03-bootstrap-watchdog","--g03-bootstrap-worker"):
    raise SystemExit(97)
signal.signal(signal.SIGCHLD,signal.SIG_DFL)
if signal.getsignal(signal.SIGCHLD)!=signal.SIG_DFL:
    raise SystemExit(106)
def value(name):
    if sys.argv.count(name)!=1:
        raise SystemExit(98)
    i=sys.argv.index(name)
    if i+1>=len(sys.argv):
        raise SystemExit(99)
    return sys.argv[i+1]
def fd(name):
    text=value(name)
    if not text.isdigit() or str(int(text))!=text:
        raise SystemExit(100)
    return int(text)
action=sys.argv[2] if len(sys.argv)>2 else ""
inherited=set()
if role.endswith("watchdog"):
    if action!="__watchdog-worker": raise SystemExit(101)
    names=("--parent-death-fd","--worker-guard-read-fd","--worker-guard-write-fd","--worker-pid-report-fd","--guardian-ready-read-fd","--guardian-ready-write-fd","--worker-lifetime-read-fd","--worker-lifetime-write-fd")
    inherited.update(fd(name) for name in names)
    forwarded=json.loads(value("--forward-fds-json"))
    if type(forwarded) is not list or any(type(item) is not int or item<0 for item in forwarded): raise SystemExit(102)
    inherited.update(forwarded)
    guard=fd("--parent-death-fd")
    ready=None
else:
    if action not in ("__build-worker","__verify-worker"): raise SystemExit(103)
    held=json.loads(value("--held-input-fds-json"))
    if type(held) is not dict or any(type(item) is not int or item<0 for item in held.values()): raise SystemExit(104)
    inherited.update(held.values())
    names=("--parent-guard-fd","--guardian-ready-write-fd","--worker-lifetime-write-fd")
    inherited.update(fd(name) for name in names)
    inherited.add(fd("--output-root-fd" if action=="__build-worker" else "--report-fd"))
    guard=fd("--parent-guard-fd")
    ready=fd("--guardian-ready-write-fd")
if len(inherited)<4: raise SystemExit(105)
for descriptor in inherited: os.set_inheritable(descriptor,False)
def watch():
    if ready is not None:
        try:
            if os.write(ready,b"R")!=1: raise OSError("short ready")
        except OSError:
            os.killpg(os.getpgrp(),signal.SIGKILL)
    try:
        while os.read(guard,1): pass
    except OSError:
        pass
    os.killpg(os.getpgrp(),signal.SIGKILL)
threading.Thread(target=watch,name="g03-bootstrap-guardian",daemon=True).start()
os.environ["G03_V2_NESTED_GUARDIAN_BOOTSTRAPPED"]=role
sys.argv=["goalzendo_interactive_v2.nested_opening_feasibility_execution",*sys.argv[2:]]
runpy.run_module("goalzendo_interactive_v2.nested_opening_feasibility_execution",run_name="__main__",alter_sys=True)
'''

_PlanClass = Literal[
    "production_9x16x2_nested_k32_floor8",
    "engineering_test_fixture",
]
_PRODUCTION_PLAN_CLASS: _PlanClass = "production_9x16x2_nested_k32_floor8"
_ENGINEERING_PLAN_CLASS: _PlanClass = "engineering_test_fixture"

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}")
_UTC_TIMESTAMP = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z"
)
_UUID = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
)
_NONCE = re.compile(r"[0-9a-f]{64}")

_AUTHORIZATION: dict[str, str | bool] = {
    "scope": "cpu_only_nested_opening_construction_feasibility_artifact_lifecycle",
    "g01_authorized": False,
    "g03_capability_launch_authorized": False,
    "g03_scientific_launch_authorized": False,
    "production_bank_authorized": False,
    "model_execution_authorized": False,
    "weight_updates_authorized": False,
    "launch_authorized": False,
}

_SCIENTIFIC_CLAIM_BOUNDARY: dict[str, bool] = {
    "cpu_nested_opening_construction_feasibility_report_present": False,
    "positive_m_q_quota_selected": False,
    "production_nested_opening_pool_frozen": False,
    "matched_bank_size_selected": False,
    "difficulty_matcher_run": False,
    "evaluation_quartets_materialized": False,
    "challenge_panels_materialized": False,
    "intervention_bank_materialized": False,
    "render_balance_audited": False,
    "tokenizer_artifact_bound": False,
    "runtime_bridge_invoked": False,
    "model_calls_observed": False,
    "model_outcomes_present": False,
    "production_bank_authorized": False,
    "g01_authorized": False,
    "g03_capability_launch_authorized": False,
    "g03_scientific_launch_authorized": False,
    "launch_authorized": False,
}

_EXECUTED_SCIENTIFIC_CLAIM_BOUNDARY: dict[str, bool] = {
    **_SCIENTIFIC_CLAIM_BOUNDARY,
    "cpu_nested_opening_construction_feasibility_report_present": True,
}

_RECONSTRUCTION_CLAIM_BOUNDARY: dict[str, bool] = {
    "source_archive_reconstructability_verified_by_repository": False,
    "constraints_installability_verified_by_repository": False,
    "environment_reconstruction_verified_by_repository": False,
    "reproducibility_or_reconstructability_claimed": False,
}

_ENGINEERING_EXECUTION_SCOPE: dict[str, bool] = {
    "engineering_fixture_only": True,
    "production_evidence": False,
    "study_evidence": False,
}

_REGISTRATION_CLAIM_BOUNDARY: dict[str, bool] = {
    "receipt_shape_and_content_pins_validated": True,
    "registration_service_identity_independently_verified_by_repository": False,
    "receipt_issuance_independently_verified_by_repository": False,
    "temporal_priority_independently_verified_by_repository": False,
    "external_timestamp_independently_verified_by_repository": False,
}

_EXECUTION_POLICY: dict[str, int | str | bool] = {
    "worker_process_count": 2,
    "watchdog_process_count": 2,
    "worker_descendant_process_creation_requested": False,
    "worker_descendant_process_creation_os_enforced": False,
    "same_process_group_cleanup_attempted_before_direct_worker_reap": True,
    "detached_descendant_absence_os_proven": False,
    "single_threaded_controller_host_required_for_signal_atomicity": True,
    "default_non_auto_reaping_sigchld_required": True,
    "fresh_process_verification_required": True,
    "shared_in_process_construction_cache_allowed": False,
    "retry_within_execution_uuid_allowed": False,
    "cross_parent_or_cross_host_one_shot_enforced_by_repository": False,
    "pre_root_failure_consumption_enforced_by_repository": False,
    "local_one_shot_survives_root_removal_or_rename": False,
    "early_stop_allowed": False,
    "identity_replacement_allowed": False,
    "interrupted_output_root_reusable": False,
    "seven_or_four_held_data_artifacts_held_by_descriptor": True,
    "held_data_artifacts_read_with_pread": True,
    "canonical_code_paths_reopened_for_hash_revalidation": True,
    "all_code_or_source_files_held_by_descriptor": False,
    "report_commit_method": "same_filesystem_exclusive_hard_link_after_verification",
    "completion_receipt_written_last": True,
}

_RUNTIME_EVIDENCE_CONTRACT = (
    "descriptive action-entry through terminal-object-admission monotonic timing "
    "and process-self ru_maxrss; excludes terminal serialization and stdout or "
    "filesystem durability and is not a scientific runtime, performance, or "
    "resource-sufficiency claim"
)

_WORKER_COMMON_LAUNCH_OPTIONS = (
    "--held-input-fds-json",
    "--held-input-bindings-json",
    "--expected-upstream-plan-digest",
    "--expected-upstream-plan-sha256",
    "--expected-nested-plan-digest",
    "--expected-nested-plan-sha256",
    "--expected-freeze-request-digest",
    "--expected-registration-receipt-sha256",
    "--expected-registration-reference",
    "--expected-execution-uuid",
    "--expected-execution-nonce",
    "--expected-execution-deadline-seconds",
    "--controller-started-wall-utc",
    "--deadline-monotonic-ns",
    "--runner-source",
    "--parent-guard-fd",
    "--guardian-ready-write-fd",
    "--worker-lifetime-write-fd",
)

_PROSPECTIVE_FORBIDDEN_KEYS = frozenset(
    {
        "accounting",
        "attempts_observed",
        "builder_terminal",
        "cell_summary",
        "completed_opening_assessment_count",
        "construction_status",
        "disposition",
        "execution_receipt",
        "exact_match",
        "failure",
        "failures",
        "model_output",
        "observed",
        "observed_m_q_summary",
        "observed_report_digest",
        "openings",
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
_HELD_INPUT_MAXIMUM_BYTES: dict[str, int] = {
    "upstream_census_plan": 32 * 1024 * 1024,
    "nested_plan": 256 * 1024 * 1024,
    "freeze_request": 4 * 1024 * 1024,
    "registration_receipt": 1 * 1024 * 1024,
    "source_archive": 2 * 1024 * 1024 * 1024,
    "constraints": 16 * 1024 * 1024,
    "environment_lock": 16 * 1024 * 1024,
}


class NestedOpeningFeasibilityExecutionV1Error(ValueError):
    """Raised when an execution artifact or lifecycle transition is unsafe."""


class _OutputRootAlreadyExistsV1(NestedOpeningFeasibilityExecutionV1Error):
    """Distinguish an unowned pre-existing root from post-mkdir failure."""


def _lexical_absolute(path: Path) -> Path:
    """Return one normalized absolute path without resolving any symlink."""

    candidate = path if path.is_absolute() else Path.cwd() / path
    if ".." in candidate.parts:
        raise NestedOpeningFeasibilityExecutionV1Error(
            "file path must not contain parent traversal components"
        )
    normalized = Path(os.path.normpath(os.fspath(candidate)))
    if not normalized.is_absolute():
        raise NestedOpeningFeasibilityExecutionV1Error(
            "file path must be an absolute normalized lexical path"
        )
    return normalized


def _open_directory_chain_nofollow(path: Path) -> tuple[int, tuple[tuple[int, int, str], ...]]:
    """Open every absolute directory component without following symlinks."""

    absolute = _lexical_absolute(path)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(os.path.sep, flags)
    chain: list[tuple[int, int, str]] = []
    try:
        root_stat = os.fstat(descriptor)
        chain.append((root_stat.st_dev, root_stat.st_ino, os.path.sep))
        for component in absolute.parts[1:]:
            try:
                next_descriptor = os.open(component, flags, dir_fd=descriptor)
            except OSError as exc:
                if exc.errno in (errno.ELOOP, errno.ENOTDIR):
                    raise NestedOpeningFeasibilityExecutionV1Error(
                        f"path component {component!r} is a symlink or not a directory"
                    ) from exc
                raise
            next_stat = os.fstat(next_descriptor)
            if not stat.S_ISDIR(next_stat.st_mode):
                os.close(next_descriptor)
                raise NestedOpeningFeasibilityExecutionV1Error(
                    f"path component {component!r} is not a directory"
                )
            os.close(descriptor)
            descriptor = next_descriptor
            chain.append((next_stat.st_dev, next_stat.st_ino, component))
        return descriptor, tuple(chain)
    except BaseException:
        os.close(descriptor)
        raise


def _replay_directory_chain_nofollow(
    path: Path, expected: tuple[tuple[int, int, str], ...]
) -> int:
    descriptor, observed = _open_directory_chain_nofollow(path)
    if observed != expected:
        os.close(descriptor)
        raise NestedOpeningFeasibilityExecutionV1Error(
            "lexical directory chain changed or names a different inode"
        )
    return descriptor


def _dump_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise NestedOpeningFeasibilityExecutionV1Error(
            f"value is not canonical JSON: {exc}"
        ) from exc


def _canonical_bytes(value: Any) -> bytes:
    return (_dump_json(value) + "\n").encode("ascii")


def _load_json(text: str) -> Any:
    if type(text) is not str or not text:
        raise NestedOpeningFeasibilityExecutionV1Error("JSON input must be nonempty text")

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise NestedOpeningFeasibilityExecutionV1Error(
                    f"duplicate JSON object key: {key!r}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> NoReturn:
        raise NestedOpeningFeasibilityExecutionV1Error(
            f"non-finite JSON constant is forbidden: {value}"
        )

    try:
        return json.loads(text, object_pairs_hook=no_duplicates, parse_constant=reject_constant)
    except NestedOpeningFeasibilityExecutionV1Error:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise NestedOpeningFeasibilityExecutionV1Error(f"invalid JSON: {exc}") from exc


def _digest(value: Any, *, domain: str) -> str:
    payload = domain.encode("ascii") + b"\0" + _dump_json(value).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


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
        raise NestedOpeningFeasibilityExecutionV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return cast(str, value)


def _require_integer(
    value: object,
    *,
    name: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise NestedOpeningFeasibilityExecutionV1Error(
            f"{name} must be an integer >= {minimum}"
        )
    if maximum is not None and value > maximum:
        raise NestedOpeningFeasibilityExecutionV1Error(
            f"{name} must be an integer <= {maximum}"
        )
    return value


def _require_boolean(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise NestedOpeningFeasibilityExecutionV1Error(f"{name} must be a Boolean")
    return value


def _require_mapping(
    value: object,
    fields: tuple[str, ...],
    *,
    name: str,
) -> Mapping[str, Any]:
    if type(value) is not dict or tuple(value) != fields:
        raise NestedOpeningFeasibilityExecutionV1Error(
            f"{name} has noncanonical, missing, extra, or reordered fields"
        )
    return cast(Mapping[str, Any], value)


def _require_identifier(value: object, *, name: str) -> str:
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        raise NestedOpeningFeasibilityExecutionV1Error(
            f"{name} is not a canonical identifier"
        )
    return value


def _require_timestamp(value: object) -> str:
    if type(value) is not str or _UTC_TIMESTAMP.fullmatch(value) is None:
        raise NestedOpeningFeasibilityExecutionV1Error(
            "registration timestamp must be YYYY-MM-DDTHH:MM:SSZ"
        )
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise NestedOpeningFeasibilityExecutionV1Error(
            "registration timestamp is invalid"
        ) from exc
    return value


def _registration_datetime(value: str) -> datetime:
    return datetime.strptime(
        _require_timestamp(value), "%Y-%m-%dT%H:%M:%SZ"
    ).replace(tzinfo=timezone.utc)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _wall_datetime(value: object, *, name: str) -> datetime:
    if type(value) is not str or not value.endswith("Z"):
        raise NestedOpeningFeasibilityExecutionV1Error(
            f"{name} must be an exact UTC timestamp"
        )
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise NestedOpeningFeasibilityExecutionV1Error(f"{name} is invalid") from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise NestedOpeningFeasibilityExecutionV1Error(f"{name} must use UTC")
    return parsed


def _require_uuid(value: object) -> str:
    if type(value) is not str or _UUID.fullmatch(value) is None:
        raise NestedOpeningFeasibilityExecutionV1Error(
            "execution UUID is not canonical UUIDv4 text"
        )
    return value


def _require_nonce(value: object) -> str:
    if type(value) is not str or _NONCE.fullmatch(value) is None:
        raise NestedOpeningFeasibilityExecutionV1Error(
            "execution nonce must be 64 lowercase hex"
        )
    return value


def _require_exact_constant(
    value: object, expected: Mapping[str, Any], *, name: str
) -> None:
    obj = _require_mapping(value, tuple(expected), name=name)
    if _dump_json(obj) != _dump_json(expected):
        raise NestedOpeningFeasibilityExecutionV1Error(
            f"{name} differs from the frozen constant"
        )


def _exact_json_equal(left: object, right: object) -> bool:
    """Compare JSON-shaped values without Python's bool/int aliasing."""

    return _dump_json(left) == _dump_json(right)


def _reject_prospective_outcomes(
    value: object, *, path: str = "freeze_request"
) -> None:
    if type(value) is dict:
        for key, child in cast(dict[str, Any], value).items():
            normalized = key.lower().replace("-", "_")
            if normalized in _PROSPECTIVE_FORBIDDEN_KEYS or normalized.startswith(
                "observed_"
            ):
                raise NestedOpeningFeasibilityExecutionV1Error(
                    "prospective freeze request contains forbidden outcome field "
                    f"at {path}.{key}"
                )
            _reject_prospective_outcomes(child, path=f"{path}.{key}")
    elif type(value) is list:
        for position, child in enumerate(value):
            _reject_prospective_outcomes(child, path=f"{path}[{position}]")


def _repository_root() -> Path:
    path = Path(__file__)
    if path.is_symlink() or not path.is_file():
        raise NestedOpeningFeasibilityExecutionV1Error(
            "execution module source must be one ordinary file"
        )
    root = path.resolve().parents[2]
    if not (root / _RUNNER_RELATIVE_PATH).is_file():
        raise NestedOpeningFeasibilityExecutionV1Error(
            "canonical nested-opening runner is missing"
        )
    return root


def canonical_nested_opening_feasibility_runner_path_v1() -> Path:
    """Return the sole production runner path accepted by this lifecycle."""

    return _repository_root() / _RUNNER_RELATIVE_PATH


def _ordinary_file_bytes(
    path: Path,
    *,
    name: str,
    maximum_bytes: int | None = None,
    expected_modes: frozenset[int] | None = None,
) -> bytes:
    held = _HeldOrdinaryFileV1.open(
        path,
        label=name,
        maximum_bytes=maximum_bytes,
        expected_modes=expected_modes,
    )
    try:
        payload = held.read_bytes(maximum_bytes=maximum_bytes)
        held.replay_external_basename()
        return payload
    finally:
        held.close()


def _pread_exact(descriptor: int, byte_count: int, *, name: str) -> bytes:
    chunks: list[bytes] = []
    offset = 0
    while offset < byte_count:
        chunk = os.pread(descriptor, min(byte_count - offset, 1024 * 1024), offset)
        if not chunk:
            raise NestedOpeningFeasibilityExecutionV1Error(
                f"{name} changed while held bytes were read"
            )
        chunks.append(chunk)
        offset += len(chunk)
    if os.pread(descriptor, 1, byte_count):
        raise NestedOpeningFeasibilityExecutionV1Error(
            f"{name} grew while held bytes were read"
        )
    return b"".join(chunks)


def _source_binding(path: Path, *, relative_path: str) -> dict[str, Any]:
    payload = _ordinary_file_bytes(
        path,
        name=relative_path,
        maximum_bytes=8 * 1024 * 1024,
        expected_modes=frozenset({0o644}),
    )
    return {
        "relative_path": relative_path,
        "sha256": _sha256_bytes(payload),
        "byte_count": len(payload),
        "mode": "0644",
    }


def _safe_environment(*, bootstrap_role: str = "") -> dict[str, str]:
    root = _repository_root()
    return {
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        _BOOTSTRAP_MARKER: bootstrap_role,
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPYCACHEPREFIX": _DISABLED_PYCACHE_PREFIX,
        "PYTHONSAFEPATH": "1",
        "PYTHONPATH": str(root / "src"),
    }


def _observed_safe_environment(*, bootstrap_role: str) -> dict[str, str]:
    expected = _safe_environment(bootstrap_role=bootstrap_role)
    observed = dict(os.environ)
    if observed != expected:
        raise NestedOpeningFeasibilityExecutionV1Error(
            "fresh process environment differs from the exact safe launch environment"
        )
    return observed


def _safe_bootstrap_launch_prefix(*, role: str) -> list[str]:
    if role not in (
        "--g03-bootstrap-watchdog",
        "--g03-bootstrap-worker",
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "unknown safe-bootstrap process role"
        )
    return [
        str(Path(sys.executable).resolve(strict=True)),
        "-S",
        "-P",
        "-B",
        "-c",
        _BOOTSTRAP_SOURCE,
        role,
    ]


def _worker_launch_options(action: str) -> tuple[str, ...]:
    if action == "__build-worker":
        return (*_WORKER_COMMON_LAUNCH_OPTIONS, "--output-root-fd")
    if action == "__verify-worker":
        return (
            *_WORKER_COMMON_LAUNCH_OPTIONS,
            "--report-fd",
            "--expected-report-digest",
            "--expected-report-sha256",
            "--expected-report-byte-count",
        )
    raise NestedOpeningFeasibilityExecutionV1Error(
        "worker action is not a supported engineering action"
    )


def _exact_worker_launch_option_map(
    worker_argv: object,
    *,
    expected_action: str,
) -> dict[str, str]:
    if (
        type(worker_argv) is not list
        or any(type(item) is not str or "\0" in item for item in worker_argv)
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "worker launch argv is not a string list"
        )
    argv = cast(list[str], worker_argv)
    prefix = [
        *_safe_bootstrap_launch_prefix(role="--g03-bootstrap-worker"),
        expected_action,
    ]
    options = _worker_launch_options(expected_action)
    tail = argv[len(prefix) :]
    if (
        argv[: len(prefix)] != prefix
        or len(tail) != 2 * len(options)
        or tuple(tail[::2]) != options
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "worker launch argv differs from the exact ordered action grammar"
        )
    values = dict(zip(options, tail[1::2], strict=True))
    descriptor_options = (
        "--parent-guard-fd",
        "--guardian-ready-write-fd",
        "--worker-lifetime-write-fd",
        "--output-root-fd" if expected_action == "__build-worker" else "--report-fd",
    )
    for option in descriptor_options:
        value = values[option]
        if not value.isdigit() or str(int(value)) != value:
            raise NestedOpeningFeasibilityExecutionV1Error(
                f"worker launch {option} is not a canonical descriptor"
            )
    for option in (
        "--expected-execution-deadline-seconds",
        "--deadline-monotonic-ns",
        *( 
            ("--expected-report-byte-count",)
            if expected_action == "__verify-worker"
            else ()
        ),
    ):
        value = values[option]
        if not value.isdigit() or str(int(value)) != value or int(value) < 1:
            raise NestedOpeningFeasibilityExecutionV1Error(
                f"worker launch {option} is not a canonical positive integer"
            )
    return values


def _full_bootstrap_launch_from_process_argv(
    process_argv: object,
    *,
    role: str,
    expected_action: str,
) -> list[str]:
    if (
        type(process_argv) is not list
        or len(process_argv) < 2
        or any(type(item) is not str or "\0" in item for item in process_argv)
        or process_argv[0] != str(Path(__file__).resolve())
        or process_argv[1] != expected_action
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "persisted process argv is not the exact transformed bootstrap argv"
        )
    return [
        *_safe_bootstrap_launch_prefix(role=role),
        *cast(list[str], process_argv)[1:],
    ]


def _exact_watchdog_launch_option_map(
    watchdog_argv: object,
    *,
    expected_worker_argv: Sequence[str],
) -> dict[str, str]:
    if (
        type(watchdog_argv) is not list
        or any(type(item) is not str or "\0" in item for item in watchdog_argv)
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "watchdog launch argv is not a string list"
        )
    argv = cast(list[str], watchdog_argv)
    prefix = [
        *_safe_bootstrap_launch_prefix(role="--g03-bootstrap-watchdog"),
        "__watchdog-worker",
    ]
    options = (
        "--parent-death-fd",
        "--worker-guard-read-fd",
        "--worker-guard-write-fd",
        "--worker-pid-report-fd",
        "--guardian-ready-read-fd",
        "--guardian-ready-write-fd",
        "--worker-lifetime-read-fd",
        "--worker-lifetime-write-fd",
        "--deadline-monotonic-ns",
        "--forward-fds-json",
        "--worker-argv-json",
    )
    tail = argv[len(prefix) :]
    if (
        argv[: len(prefix)] != prefix
        or len(tail) != 2 * len(options)
        or tuple(tail[::2]) != options
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "watchdog launch argv differs from the exact ordered grammar"
        )
    values = dict(zip(options, tail[1::2], strict=True))
    if values["--worker-argv-json"] != _dump_json(list(expected_worker_argv)):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "watchdog launch does not bind the exact worker launch argv"
        )
    descriptor_options = options[:8]
    descriptors: list[int] = []
    for option in descriptor_options:
        value = values[option]
        if not value.isdigit() or str(int(value)) != value:
            raise NestedOpeningFeasibilityExecutionV1Error(
                f"watchdog launch {option} is not a canonical descriptor"
            )
        descriptors.append(int(value))
    forward_value = _load_json(values["--forward-fds-json"])
    if (
        type(forward_value) is not list
        or not forward_value
        or any(type(item) is not int or item < 0 for item in forward_value)
        or len(set(forward_value)) != len(forward_value)
        or _dump_json(forward_value) != values["--forward-fds-json"]
        or len(set((*descriptors, *forward_value)))
        != len(descriptors) + len(forward_value)
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "watchdog launch descriptor closure is not canonical and disjoint"
        )
    deadline_text = values["--deadline-monotonic-ns"]
    if not deadline_text.isdigit() or str(int(deadline_text)) != deadline_text:
        raise NestedOpeningFeasibilityExecutionV1Error(
            "watchdog launch deadline is not a canonical integer"
        )
    _require_integer(
        int(deadline_text),
        name="watchdog launch deadline",
        minimum=1,
    )
    return values


def _python_identity() -> dict[str, Any]:
    executable = Path(sys.executable).resolve(strict=True)
    held = _HeldOrdinaryFileV1.open(
        executable,
        label="Python executable",
    )
    try:
        mode = stat.S_IMODE(held.file_mode)
        if mode & 0o111 == 0:
            raise NestedOpeningFeasibilityExecutionV1Error(
                "Python executable must retain at least one execute bit"
            )
        if mode & 0o002:
            raise NestedOpeningFeasibilityExecutionV1Error(
                "Python executable must not be world-writable"
            )
        payload = held.read_bytes()
        held.replay_external_basename()
        return {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable_mode": f"{mode:04o}",
            "executable_sha256": _sha256_bytes(payload),
            "executable_byte_count": len(payload),
        }
    finally:
        held.close()


def _execution_code_binding(runner_source_path: Path) -> dict[str, Any]:
    root = _repository_root()
    runner = _lexical_absolute(runner_source_path)
    expected_runner = _lexical_absolute(root / _RUNNER_RELATIVE_PATH)
    if runner != expected_runner:
        raise NestedOpeningFeasibilityExecutionV1Error(
            "runner source path must resolve to the canonical production runner"
        )
    return {
        "execution_module": _source_binding(
            root / _MODULE_RELATIVE_PATH, relative_path=_MODULE_RELATIVE_PATH
        ),
        "package_initializer": _source_binding(
            root / _PACKAGE_INIT_RELATIVE_PATH,
            relative_path=_PACKAGE_INIT_RELATIVE_PATH,
        ),
        "runner": _source_binding(runner, relative_path=_RUNNER_RELATIVE_PATH),
        "python": _python_identity(),
        "fresh_process_bootstrap": {
            "source_sha256": _sha256_bytes(_BOOTSTRAP_SOURCE.encode("ascii")),
            "interpreter_flags": ["-S", "-P", "-B", "-c"],
            "pycache_prefix": _DISABLED_PYCACHE_PREFIX,
            "package_initializer_bound": True,
        },
        "worker_environment": _safe_environment(),
    }


def _require_safe_bootstrap_runtime() -> None:
    required_os_features = (
        "P_PID",
        "WEXITED",
        "WNOHANG",
        "WNOWAIT",
    )
    if (
        sys.version_info < (3, 11)
        or not hasattr(sys.flags, "safe_path")
        or any(not hasattr(os, name) for name in required_os_features)
        or not hasattr(signal, "pthread_sigmask")
        or not _unreaped_child_observer_available()
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "engineering lifecycle requires Python >=3.11 safe-path, waitid "
            "WNOWAIT, and pthread signal-mask support"
        )


def _require_single_threaded_controller() -> None:
    current = threading.current_thread()
    live = tuple(threading.enumerate())
    if (
        current is not threading.main_thread()
        or len(live) != 1
        or live[0] is not current
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "engineering controller requires an explicitly single-threaded "
            "host for process-directed signal atomicity"
        )


def _require_default_sigchld() -> None:
    if signal.getsignal(signal.SIGCHLD) != signal.SIG_DFL:
        raise NestedOpeningFeasibilityExecutionV1Error(
            "engineering lifecycle requires default non-auto-reaping SIGCHLD"
        )


def _optional_input_binding(path: Path | None, *, label: str) -> dict[str, Any]:
    if path is None:
        return {
            "label": label,
            "supplied": False,
            "sha256": None,
            "byte_count": None,
            "mode": None,
        }
    payload = _ordinary_file_bytes(
        path,
        name=label,
        expected_modes=frozenset({0o400}),
        maximum_bytes=_HELD_INPUT_MAXIMUM_BYTES[label],
    )
    if not payload:
        raise NestedOpeningFeasibilityExecutionV1Error(
            f"production {label} must be nonempty"
        )
    if label in ("constraints", "environment_lock"):
        if b"\0" in payload:
            raise NestedOpeningFeasibilityExecutionV1Error(
                f"production {label} contains NUL bytes"
            )
        try:
            payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise NestedOpeningFeasibilityExecutionV1Error(
                f"production {label} must be UTF-8 text"
            ) from exc
        if not payload.endswith(b"\n"):
            raise NestedOpeningFeasibilityExecutionV1Error(
                f"production {label} must be newline terminated"
            )
    return {
        "label": label,
        "supplied": True,
        "sha256": _sha256_bytes(payload),
        "byte_count": len(payload),
        "mode": "0400",
    }


def _environment_input_bindings(
    *,
    source_archive_path: Path | None,
    constraints_path: Path | None,
    environment_lock_path: Path | None,
) -> dict[str, Any]:
    specs = (
        ("source_archive", source_archive_path),
        ("constraints", constraints_path),
        ("environment_lock", environment_lock_path),
    )
    supplied = tuple(path is not None for _, path in specs)
    if not any(supplied):
        return _environment_input_bindings_from_payloads({})
    if not all(supplied):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "sealed environment inputs must be supplied as one exact triple"
        )
    held: dict[str, _HeldOrdinaryFileV1] = {}
    try:
        for label, optional_path in specs:
            if optional_path is None:
                raise AssertionError("environment path narrowing failed")
            held[label] = _HeldOrdinaryFileV1.open(
                optional_path,
                label=label,
                maximum_bytes=_HELD_INPUT_MAXIMUM_BYTES[label],
                expected_modes=frozenset({0o400}),
            )
        _require_distinct_held_inodes(held)
        payloads = {label: item.read_bytes() for label, item in held.items()}
        for item in held.values():
            item.replay_external_basename()
        return _environment_input_bindings_from_payloads(payloads)
    finally:
        _close_held_files(held)


def _environment_input_bindings_from_payloads(
    payloads: Mapping[str, bytes],
) -> dict[str, Any]:
    expected_labels = ("source_archive", "constraints", "environment_lock")
    if payloads and tuple(payloads) != expected_labels:
        raise NestedOpeningFeasibilityExecutionV1Error(
            "environment payload labels are noncanonical"
        )
    bindings: dict[str, Any] = {}
    for label in expected_labels:
        payload = payloads.get(label)
        if payload is None:
            bindings[label] = {
                "label": label,
                "supplied": False,
                "sha256": None,
                "byte_count": None,
                "mode": None,
            }
            continue
        if not payload:
            raise NestedOpeningFeasibilityExecutionV1Error(
                f"production {label} must be nonempty"
            )
        if len(payload) > _HELD_INPUT_MAXIMUM_BYTES[label]:
            raise NestedOpeningFeasibilityExecutionV1Error(
                f"production {label} exceeds its lifecycle byte cap"
            )
        if label != "source_archive":
            if b"\0" in payload:
                raise NestedOpeningFeasibilityExecutionV1Error(
                    f"production {label} contains NUL bytes"
                )
            try:
                payload.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise NestedOpeningFeasibilityExecutionV1Error(
                    f"production {label} must be UTF-8 text"
                ) from exc
            if not payload.endswith(b"\n"):
                raise NestedOpeningFeasibilityExecutionV1Error(
                    f"production {label} must be newline terminated"
                )
        bindings[label] = {
            "label": label,
            "supplied": True,
            "sha256": _sha256_bytes(payload),
            "byte_count": len(payload),
            "mode": "0400",
        }
    bindings["verification_boundary"] = dict(_RECONSTRUCTION_CLAIM_BOUNDARY)
    return bindings


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
        raise NestedOpeningFeasibilityExecutionV1Error(
            "production requires source archive, constraints, and environment lock files"
        )
    # Validation and binding intentionally share one read per file.  Calling
    # this function never authenticates an archive format or an installation.
    _environment_input_bindings(
        source_archive_path=source_archive_path,
        constraints_path=constraints_path,
        environment_lock_path=environment_lock_path,
    )


def _assert_production_plans(
    upstream: EvaluationCensusPlanV1,
    nested: NestedOpeningFeasibilityPlanV1,
) -> None:
    if (
        not upstream.uses_production_attempt_budget
        or upstream.attempts_per_formula_stratum
        != PRODUCTION_MIRROR_ATTEMPTS_PER_STRATUM
        or upstream.candidate_pool_size != DEFAULT_CANDIDATE_POOL_SIZE
        or len(upstream.attempts) != PRODUCTION_MIRROR_ATTEMPT_COUNT
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "production requires the exact 9x16 pool-32 upstream plan"
        )
    composed = tuple(
        member.composed_rule_id
        for attempt in upstream.attempts
        for member in attempt.members
    )
    if len(composed) != PRODUCTION_MEMBER_CONSTRUCTION_COUNT or len(set(composed)) != len(
        composed
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "production upstream plan must contain 288 distinct identities"
        )
    if (
        nested.engineering_budget_override
        or not nested.exact_144x2_budget_complete
        or len(nested.attempts) != PRODUCTION_MIRROR_ATTEMPT_COUNT
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "production requires the exact 144-attempt nested plan"
        )


def _upstream_plan_binding(
    plan: EvaluationCensusPlanV1, plan_bytes: bytes
) -> dict[str, Any]:
    return {
        "evaluation_census_plan_digest": plan.digest,
        "canonical_census_plan_bytes_sha256": _sha256_bytes(plan_bytes),
        "canonical_census_plan_byte_count": len(plan_bytes),
    }


def _nested_plan_binding(
    plan: NestedOpeningFeasibilityPlanV1,
    plan_digest: str,
    plan_bytes: bytes,
) -> dict[str, Any]:
    return {
        "prospective_plan_digest": _require_sha256(
            plan_digest, name="nested plan digest"
        ),
        "canonical_plan_bytes_sha256": _sha256_bytes(plan_bytes),
        "canonical_plan_byte_count": len(plan_bytes),
        "upstream_census_plan_digest": (
            plan.upstream_identity_plan_binding.evaluation_census_plan_digest
        ),
        "upstream_census_plan_bytes_sha256": (
            plan.upstream_identity_plan_binding.canonical_census_plan_bytes_sha256
        ),
        "identity_schedule_digest": (
            plan.upstream_identity_plan_binding.identity_schedule_digest
        ),
        "generator_contract_digest": plan.generator_binding.contract_digest,
    }


@dataclass(frozen=True, slots=True)
class NestedOpeningFeasibilityFreezeRequestV1:
    """Outcome-free request to register both exact prospective plan byte strings."""

    plan_class: _PlanClass
    upstream_evaluation_census_plan_binding: Mapping[str, Any]
    nested_opening_feasibility_plan_binding: Mapping[str, Any]
    upstream_evaluation_census_source_binding: Mapping[str, Any]
    upstream_evaluation_census_catalog_binding: Mapping[str, Any]
    upstream_evaluation_census_generator_binding: Mapping[str, Any]
    nested_source_binding: Mapping[str, Any]
    nested_upstream_identity_plan_binding: Mapping[str, Any]
    nested_generator_binding: Mapping[str, Any]
    execution_code_binding: Mapping[str, Any]
    environment_input_bindings: Mapping[str, Any]
    fixed_budget: Mapping[str, Any]

    def _unsigned_obj(self) -> dict[str, Any]:
        value = {
            "schema_version": NESTED_OPENING_FEASIBILITY_EXECUTION_SCHEMA_VERSION,
            "request_kind": _FREEZE_REQUEST_KIND,
            "status": "prospective_outcome_free_registration_requested_nonauthorizing",
            "authorization": dict(_AUTHORIZATION),
            "plan_class": self.plan_class,
            "upstream_evaluation_census_plan_binding": dict(
                self.upstream_evaluation_census_plan_binding
            ),
            "nested_opening_feasibility_plan_binding": dict(
                self.nested_opening_feasibility_plan_binding
            ),
            "upstream_evaluation_census_source_binding": dict(
                self.upstream_evaluation_census_source_binding
            ),
            "upstream_evaluation_census_catalog_binding": dict(
                self.upstream_evaluation_census_catalog_binding
            ),
            "upstream_evaluation_census_generator_binding": dict(
                self.upstream_evaluation_census_generator_binding
            ),
            "nested_source_binding": dict(self.nested_source_binding),
            "nested_upstream_identity_plan_binding": dict(
                self.nested_upstream_identity_plan_binding
            ),
            "nested_generator_binding": dict(self.nested_generator_binding),
            "execution_code_binding": dict(self.execution_code_binding),
            "environment_input_bindings": dict(self.environment_input_bindings),
            "fixed_budget": dict(self.fixed_budget),
            "registration_requirement": {
                "separate_external_receipt_required_before_execution": True,
                "receipt_content_pin_required": True,
                "receipt_reference_pin_required": True,
                "both_plan_byte_strings_precommitted": True,
                "repository_can_independently_verify_registration_service": False,
            },
            "prospective_payload_only": True,
            "scientific_claim_boundary": dict(_SCIENTIFIC_CLAIM_BOUNDARY),
        }
        _reject_prospective_outcomes(value)
        return value

    def _digest_unverified(self) -> str:
        return _digest(self._unsigned_obj(), domain=_FREEZE_REQUEST_DOMAIN)

    def _artifact_obj_unverified(self) -> dict[str, Any]:
        return {
            **self._unsigned_obj(),
            "freeze_request_digest": self._digest_unverified(),
        }


def _serialize_freeze_request_after_replay(
    request: NestedOpeningFeasibilityFreezeRequestV1,
) -> str:
    if type(request) is not NestedOpeningFeasibilityFreezeRequestV1:
        raise TypeError("request must be a NestedOpeningFeasibilityFreezeRequestV1")
    return _canonical_bytes(request._artifact_obj_unverified()).decode("ascii")


def _fixed_budget(plan: NestedOpeningFeasibilityPlanV1) -> dict[str, Any]:
    production = not plan.engineering_budget_override
    attempts = len(plan.attempts)
    members = attempts * MEMBERS_PER_ATTEMPT
    return {
        "formula_stratum_count": 9,
        "attempts_per_formula_stratum": (
            PRODUCTION_MIRROR_ATTEMPTS_PER_STRATUM if production else None
        ),
        "mirror_attempt_count": PRODUCTION_MIRROR_ATTEMPT_COUNT if production else attempts,
        "members_per_attempt": MEMBERS_PER_ATTEMPT,
        "planned_member_construction_count": (
            PRODUCTION_MEMBER_CONSTRUCTION_COUNT if production else members
        ),
        "derived_openings_per_member": DERIVED_OPENINGS_PER_MEMBER,
        "planned_opening_assessment_count": (
            PRODUCTION_OPENING_ASSESSMENT_COUNT
            if production
            else members * DERIVED_OPENINGS_PER_MEMBER
        ),
        "common_union_slots_per_member": COMMON_UNION_SLOT_COUNT,
        "candidate_window_k": CANDIDATE_WINDOW_K,
        "supported_version_space_floor": SUPPORTED_VERSION_SPACE_FLOOR,
        "upstream_candidate_pool_size": (
            DEFAULT_CANDIDATE_POOL_SIZE if production else None
        ),
        "no_early_stop": True,
        "no_identity_replacement": True,
        "no_retry": True,
        "uses_exact_144x2_budget": production,
        "engineering_budget_override": not production,
    }


def _build_freeze_request(
    upstream: EvaluationCensusPlanV1,
    upstream_bytes: bytes,
    nested: NestedOpeningFeasibilityPlanV1,
    nested_digest: str,
    nested_bytes: bytes,
    *,
    plan_class: _PlanClass,
    runner_source_path: Path,
    source_archive_path: Path | None,
    constraints_path: Path | None,
    environment_lock_path: Path | None,
) -> NestedOpeningFeasibilityFreezeRequestV1:
    if plan_class == _PRODUCTION_PLAN_CLASS:
        _assert_production_plans(upstream, nested)
        if any(
            path is None
            for path in (
                source_archive_path,
                constraints_path,
                environment_lock_path,
            )
        ):
            raise NestedOpeningFeasibilityExecutionV1Error(
                "production freeze requires all three sealed environment inputs"
            )
    elif not nested.engineering_budget_override:
        raise NestedOpeningFeasibilityExecutionV1Error(
            "a production plan cannot be labelled as an engineering fixture"
        )
    environment_bindings = _environment_input_bindings(
        source_archive_path=source_archive_path,
        constraints_path=constraints_path,
        environment_lock_path=environment_lock_path,
    )
    request = NestedOpeningFeasibilityFreezeRequestV1(
        plan_class,
        _upstream_plan_binding(upstream, upstream_bytes),
        _nested_plan_binding(nested, nested_digest, nested_bytes),
        upstream.source_binding.as_obj(),
        upstream.catalog_binding.as_obj(),
        upstream.generator_binding.as_obj(),
        nested.source_binding.as_obj(),
        nested.upstream_identity_plan_binding.as_obj(),
        nested.generator_binding.as_obj(),
        _execution_code_binding(runner_source_path),
        environment_bindings,
        _fixed_budget(nested),
    )
    _reject_prospective_outcomes(request._artifact_obj_unverified())
    return request


def _parse_nested_opening_feasibility_freeze_request(
    text: str,
    *,
    upstream_census_plan: EvaluationCensusPlanV1,
    upstream_census_plan_text: str,
    nested_plan: NestedOpeningFeasibilityPlanV1,
    nested_plan_text: str,
    expected_upstream_census_plan_digest: str,
    expected_upstream_census_plan_bytes_sha256: str,
    expected_nested_plan_digest: str,
    expected_nested_plan_bytes_sha256: str,
    runner_source_path: Path,
    source_archive_path: Path | None = None,
    constraints_path: Path | None = None,
    environment_lock_path: Path | None = None,
    expected_digest: str | None = None,
    for_testing: bool = False,
) -> NestedOpeningFeasibilityFreezeRequestV1:
    value = _load_json(text)
    obj = _require_mapping(
        value,
        (
            "schema_version",
            "request_kind",
            "status",
            "authorization",
            "plan_class",
            "upstream_evaluation_census_plan_binding",
            "nested_opening_feasibility_plan_binding",
            "upstream_evaluation_census_source_binding",
            "upstream_evaluation_census_catalog_binding",
            "upstream_evaluation_census_generator_binding",
            "nested_source_binding",
            "nested_upstream_identity_plan_binding",
            "nested_generator_binding",
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
        _require_integer(obj["schema_version"], name="freeze schema version")
        != NESTED_OPENING_FEASIBILITY_EXECUTION_SCHEMA_VERSION
        or obj["request_kind"] != _FREEZE_REQUEST_KIND
        or obj["status"]
        != "prospective_outcome_free_registration_requested_nonauthorizing"
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "unknown freeze-request schema, kind, or status"
        )
    _require_exact_constant(obj["authorization"], _AUTHORIZATION, name="authorization")
    expected_plan_class = _ENGINEERING_PLAN_CLASS if for_testing else _PRODUCTION_PLAN_CLASS
    if obj["plan_class"] != expected_plan_class:
        raise NestedOpeningFeasibilityExecutionV1Error(
            "freeze-request plan class differs from the caller's nominal API boundary"
        )
    try:
        upstream_bytes = upstream_census_plan_text.encode("ascii")
        nested_bytes = nested_plan_text.encode("ascii")
    except UnicodeEncodeError as exc:
        raise NestedOpeningFeasibilityExecutionV1Error(
            "both prospective plans must be ASCII"
        ) from exc
    upstream_digest = _require_sha256(
        expected_upstream_census_plan_digest,
        name="expected upstream census plan digest",
    )
    upstream_sha = _require_sha256(
        expected_upstream_census_plan_bytes_sha256,
        name="expected upstream census plan sha256",
    )
    nested_sha = _require_sha256(
        expected_nested_plan_bytes_sha256,
        name="expected nested plan sha256",
    )
    if _sha256_bytes(upstream_bytes) != upstream_sha:
        raise NestedOpeningFeasibilityExecutionV1Error(
            "upstream census plan bytes differ from external SHA-256"
        )
    if _sha256_bytes(nested_bytes) != nested_sha:
        raise NestedOpeningFeasibilityExecutionV1Error(
            "nested plan bytes differ from external SHA-256"
        )
    replayed_upstream = parse_evaluation_census_plan_v1(
        upstream_census_plan_text,
        expected_digest=upstream_digest,
    )
    if replayed_upstream != upstream_census_plan:
        raise NestedOpeningFeasibilityExecutionV1Error(
            "supplied upstream plan object differs from exact canonical bytes"
        )
    if for_testing:
        replayed_nested = (
            parse_nested_opening_construction_feasibility_plan_for_testing_v1(
                nested_plan_text,
                census_plan_text=upstream_census_plan_text,
                expected_census_plan_digest=replayed_upstream.digest,
                expected_census_plan_bytes_sha256=upstream_sha,
                expected_plan_digest=expected_nested_plan_digest,
            )
        )
    else:
        replayed_nested = parse_nested_opening_construction_feasibility_plan_v1(
            nested_plan_text,
            census_plan_text=upstream_census_plan_text,
            expected_census_plan_digest=replayed_upstream.digest,
            expected_census_plan_bytes_sha256=upstream_sha,
            expected_plan_digest=expected_nested_plan_digest,
        )
    if replayed_nested != nested_plan:
        raise NestedOpeningFeasibilityExecutionV1Error(
            "supplied nested plan object differs from exact parent-bound canonical bytes"
        )
    expected = _build_freeze_request(
        replayed_upstream,
        upstream_bytes,
        replayed_nested,
        expected_nested_plan_digest,
        nested_bytes,
        plan_class=expected_plan_class,
        runner_source_path=runner_source_path,
        source_archive_path=source_archive_path,
        constraints_path=constraints_path,
        environment_lock_path=environment_lock_path,
    )
    if _dump_json(value) != _dump_json(expected._artifact_obj_unverified()):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "freeze request differs from both exact plans, sources, generator, code, or inputs"
        )
    if expected_digest is not None and expected._digest_unverified() != _require_sha256(
        expected_digest, name="expected freeze-request digest"
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "freeze-request digest differs from expected"
        )
    if _serialize_freeze_request_after_replay(expected) != text:
        raise NestedOpeningFeasibilityExecutionV1Error(
            "freeze request is not canonical newline-terminated JSON"
        )
    return expected


def parse_nested_opening_feasibility_freeze_request_v1(
    text: str,
    *,
    upstream_census_plan: EvaluationCensusPlanV1,
    upstream_census_plan_text: str,
    nested_plan: NestedOpeningFeasibilityPlanV1,
    nested_plan_text: str,
    expected_upstream_census_plan_digest: str,
    expected_upstream_census_plan_bytes_sha256: str,
    expected_nested_plan_digest: str,
    expected_nested_plan_bytes_sha256: str,
    runner_source_path: Path,
    source_archive_path: Path,
    constraints_path: Path,
    environment_lock_path: Path,
    expected_digest: str | None = None,
) -> NestedOpeningFeasibilityFreezeRequestV1:
    """Parse only a production freeze request after both exact plan replays."""

    return _parse_nested_opening_feasibility_freeze_request(
        text,
        upstream_census_plan=upstream_census_plan,
        upstream_census_plan_text=upstream_census_plan_text,
        nested_plan=nested_plan,
        nested_plan_text=nested_plan_text,
        expected_upstream_census_plan_digest=expected_upstream_census_plan_digest,
        expected_upstream_census_plan_bytes_sha256=(
            expected_upstream_census_plan_bytes_sha256
        ),
        expected_nested_plan_digest=expected_nested_plan_digest,
        expected_nested_plan_bytes_sha256=expected_nested_plan_bytes_sha256,
        runner_source_path=runner_source_path,
        source_archive_path=source_archive_path,
        constraints_path=constraints_path,
        environment_lock_path=environment_lock_path,
        expected_digest=expected_digest,
        for_testing=False,
    )


def parse_nested_opening_feasibility_freeze_request_for_testing_v1(
    text: str,
    *,
    upstream_census_plan: EvaluationCensusPlanV1,
    upstream_census_plan_text: str,
    nested_plan: NestedOpeningFeasibilityPlanV1,
    nested_plan_text: str,
    expected_upstream_census_plan_digest: str,
    expected_upstream_census_plan_bytes_sha256: str,
    expected_nested_plan_digest: str,
    expected_nested_plan_bytes_sha256: str,
    runner_source_path: Path,
    expected_digest: str | None = None,
) -> NestedOpeningFeasibilityFreezeRequestV1:
    """Parse only the nominally distinct reduced engineering freeze request."""

    return _parse_nested_opening_feasibility_freeze_request(
        text,
        upstream_census_plan=upstream_census_plan,
        upstream_census_plan_text=upstream_census_plan_text,
        nested_plan=nested_plan,
        nested_plan_text=nested_plan_text,
        expected_upstream_census_plan_digest=expected_upstream_census_plan_digest,
        expected_upstream_census_plan_bytes_sha256=(
            expected_upstream_census_plan_bytes_sha256
        ),
        expected_nested_plan_digest=expected_nested_plan_digest,
        expected_nested_plan_bytes_sha256=expected_nested_plan_bytes_sha256,
        runner_source_path=runner_source_path,
        expected_digest=expected_digest,
        for_testing=True,
    )


def serialize_nested_opening_feasibility_freeze_request_v1(
    request: NestedOpeningFeasibilityFreezeRequestV1,
    *,
    upstream_census_plan: EvaluationCensusPlanV1,
    upstream_census_plan_text: str,
    nested_plan: NestedOpeningFeasibilityPlanV1,
    nested_plan_text: str,
    expected_upstream_census_plan_digest: str,
    expected_upstream_census_plan_bytes_sha256: str,
    expected_nested_plan_digest: str,
    expected_nested_plan_bytes_sha256: str,
    runner_source_path: Path,
    source_archive_path: Path,
    constraints_path: Path,
    environment_lock_path: Path,
) -> str:
    """Serialize a production request only after both exact plan replays."""

    candidate = _serialize_freeze_request_after_replay(request)
    parsed = parse_nested_opening_feasibility_freeze_request_v1(
        candidate,
        upstream_census_plan=upstream_census_plan,
        upstream_census_plan_text=upstream_census_plan_text,
        nested_plan=nested_plan,
        nested_plan_text=nested_plan_text,
        expected_upstream_census_plan_digest=expected_upstream_census_plan_digest,
        expected_upstream_census_plan_bytes_sha256=(
            expected_upstream_census_plan_bytes_sha256
        ),
        expected_nested_plan_digest=expected_nested_plan_digest,
        expected_nested_plan_bytes_sha256=expected_nested_plan_bytes_sha256,
        runner_source_path=runner_source_path,
        source_archive_path=source_archive_path,
        constraints_path=constraints_path,
        environment_lock_path=environment_lock_path,
    )
    if parsed != request:
        raise NestedOpeningFeasibilityExecutionV1Error(
            "freeze request differs from exact replay"
        )
    return candidate


def serialize_nested_opening_feasibility_freeze_request_for_testing_v1(
    request: NestedOpeningFeasibilityFreezeRequestV1,
    *,
    upstream_census_plan: EvaluationCensusPlanV1,
    upstream_census_plan_text: str,
    nested_plan: NestedOpeningFeasibilityPlanV1,
    nested_plan_text: str,
    expected_upstream_census_plan_digest: str,
    expected_upstream_census_plan_bytes_sha256: str,
    expected_nested_plan_digest: str,
    expected_nested_plan_bytes_sha256: str,
    runner_source_path: Path,
) -> str:
    """Serialize only the nominally distinct reduced engineering request."""

    candidate = _serialize_freeze_request_after_replay(request)
    parsed = parse_nested_opening_feasibility_freeze_request_for_testing_v1(
        candidate,
        upstream_census_plan=upstream_census_plan,
        upstream_census_plan_text=upstream_census_plan_text,
        nested_plan=nested_plan,
        nested_plan_text=nested_plan_text,
        expected_upstream_census_plan_digest=expected_upstream_census_plan_digest,
        expected_upstream_census_plan_bytes_sha256=(
            expected_upstream_census_plan_bytes_sha256
        ),
        expected_nested_plan_digest=expected_nested_plan_digest,
        expected_nested_plan_bytes_sha256=expected_nested_plan_bytes_sha256,
        runner_source_path=runner_source_path,
    )
    if parsed != request:
        raise NestedOpeningFeasibilityExecutionV1Error(
            "test freeze request differs from exact replay"
        )
    return candidate


@dataclass(frozen=True, slots=True)
class ExternalNestedOpeningFeasibilityRegistrationReceiptV1:
    """Locally checkable shape for a receipt supplied by an external registrar."""

    registration_service: str
    registration_reference: str
    registered_at_utc: str
    execution_uuid: str
    execution_nonce: str
    execution_deadline_seconds: int
    freeze_request_digest: str
    freeze_request_bytes_sha256: str
    freeze_request_byte_count: int
    upstream_census_plan_digest: str
    upstream_census_plan_bytes_sha256: str
    upstream_census_plan_byte_count: int
    nested_plan_digest: str
    nested_plan_bytes_sha256: str
    nested_plan_byte_count: int

    def _unsigned_obj(self) -> dict[str, Any]:
        return {
            "schema_version": NESTED_OPENING_FEASIBILITY_EXECUTION_SCHEMA_VERSION,
            "receipt_kind": _EXTERNAL_REGISTRATION_KIND,
            "registration_service": self.registration_service,
            "registration_reference": self.registration_reference,
            "registered_at_utc": self.registered_at_utc,
            "registered_execution": {
                "execution_uuid": self.execution_uuid,
                "execution_nonce": self.execution_nonce,
                "execution_deadline_seconds": self.execution_deadline_seconds,
                "upstream_identity_schedule_precommitted": True,
                "nested_construction_plan_precommitted": True,
                "global_one_shot_consumption_enforced_by_repository": False,
            },
            "freeze_request_binding": {
                "freeze_request_digest": self.freeze_request_digest,
                "canonical_freeze_request_bytes_sha256": self.freeze_request_bytes_sha256,
                "canonical_freeze_request_byte_count": self.freeze_request_byte_count,
            },
            "upstream_evaluation_census_plan_binding": {
                "evaluation_census_plan_digest": self.upstream_census_plan_digest,
                "canonical_census_plan_bytes_sha256": self.upstream_census_plan_bytes_sha256,
                "canonical_census_plan_byte_count": self.upstream_census_plan_byte_count,
            },
            "nested_opening_feasibility_plan_binding": {
                "prospective_plan_digest": self.nested_plan_digest,
                "canonical_plan_bytes_sha256": self.nested_plan_bytes_sha256,
                "canonical_plan_byte_count": self.nested_plan_byte_count,
            },
            "claim_boundary": {
                "receipt_schema_requires_separate_external_input_at_execution": True,
                "repository_validates_only_schema_and_content_pins": True,
                "registration_service_identity_independently_verified_by_repository": False,
                "receipt_issuance_independently_verified_by_repository": False,
                "temporal_priority_independently_verified_by_repository": False,
                **_RECONSTRUCTION_CLAIM_BOUNDARY,
            },
            "authorization": dict(_AUTHORIZATION),
        }

    def _digest_unverified(self) -> str:
        return _digest(self._unsigned_obj(), domain=_EXTERNAL_REGISTRATION_DOMAIN)

    def _artifact_obj_unverified(self) -> dict[str, Any]:
        return {
            **self._unsigned_obj(),
            "registration_receipt_digest": self._digest_unverified(),
        }


def _build_external_registration_receipt_after_replay(
    freeze_request: NestedOpeningFeasibilityFreezeRequestV1,
    freeze_request_text: str,
    *,
    registration_service: str,
    registration_reference: str,
    registered_at_utc: str,
    execution_uuid: str,
    execution_nonce: str,
    execution_deadline_seconds: int,
) -> ExternalNestedOpeningFeasibilityRegistrationReceiptV1:
    """Build interchange bytes; this helper does not perform registration."""

    if type(freeze_request) is not NestedOpeningFeasibilityFreezeRequestV1:
        raise TypeError(
            "freeze_request must be a NestedOpeningFeasibilityFreezeRequestV1"
        )
    if _serialize_freeze_request_after_replay(freeze_request) != freeze_request_text:
        raise NestedOpeningFeasibilityExecutionV1Error(
            "freeze-request bytes differ from the request"
        )
    service = _require_identifier(registration_service, name="registration service")
    reference = _require_identifier(
        registration_reference, name="registration reference"
    )
    timestamp = _require_timestamp(registered_at_utc)
    registered_uuid = _require_uuid(execution_uuid)
    registered_nonce = _require_nonce(execution_nonce)
    deadline = _require_integer(
        execution_deadline_seconds,
        name="execution deadline seconds",
        minimum=1,
        maximum=86_400,
    )
    upstream = freeze_request.upstream_evaluation_census_plan_binding
    nested = freeze_request.nested_opening_feasibility_plan_binding
    freeze_bytes = freeze_request_text.encode("ascii")
    return ExternalNestedOpeningFeasibilityRegistrationReceiptV1(
        service,
        reference,
        timestamp,
        registered_uuid,
        registered_nonce,
        deadline,
        freeze_request._digest_unverified(),
        _sha256_bytes(freeze_bytes),
        len(freeze_bytes),
        _require_sha256(
            upstream["evaluation_census_plan_digest"], name="upstream plan digest"
        ),
        _require_sha256(
            upstream["canonical_census_plan_bytes_sha256"],
            name="upstream plan sha256",
        ),
        _require_integer(
            upstream["canonical_census_plan_byte_count"],
            name="upstream plan byte count",
            minimum=1,
        ),
        _require_sha256(nested["prospective_plan_digest"], name="nested plan digest"),
        _require_sha256(
            nested["canonical_plan_bytes_sha256"], name="nested plan sha256"
        ),
        _require_integer(
            nested["canonical_plan_byte_count"],
            name="nested plan byte count",
            minimum=1,
        ),
    )


def _serialize_external_registration_receipt_after_replay(
    receipt: ExternalNestedOpeningFeasibilityRegistrationReceiptV1,
) -> str:
    if type(receipt) is not ExternalNestedOpeningFeasibilityRegistrationReceiptV1:
        raise TypeError(
            "receipt must be an ExternalNestedOpeningFeasibilityRegistrationReceiptV1"
        )
    return _canonical_bytes(receipt._artifact_obj_unverified()).decode("ascii")


def build_external_nested_opening_feasibility_registration_receipt_for_testing_v1(
    freeze_request: NestedOpeningFeasibilityFreezeRequestV1,
    freeze_request_text: str,
    *,
    upstream_census_plan: EvaluationCensusPlanV1,
    upstream_census_plan_text: str,
    nested_plan: NestedOpeningFeasibilityPlanV1,
    nested_plan_text: str,
    expected_upstream_census_plan_digest: str,
    expected_upstream_census_plan_bytes_sha256: str,
    expected_nested_plan_digest: str,
    expected_nested_plan_bytes_sha256: str,
    runner_source_path: Path,
    registration_service: str,
    registration_reference: str,
    registered_at_utc: str,
    execution_uuid: str,
    execution_nonce: str,
    execution_deadline_seconds: int,
) -> ExternalNestedOpeningFeasibilityRegistrationReceiptV1:
    """Build only the nominally distinct reduced engineering interchange."""

    replayed = parse_nested_opening_feasibility_freeze_request_for_testing_v1(
        freeze_request_text,
        upstream_census_plan=upstream_census_plan,
        upstream_census_plan_text=upstream_census_plan_text,
        nested_plan=nested_plan,
        nested_plan_text=nested_plan_text,
        expected_upstream_census_plan_digest=expected_upstream_census_plan_digest,
        expected_upstream_census_plan_bytes_sha256=(
            expected_upstream_census_plan_bytes_sha256
        ),
        expected_nested_plan_digest=expected_nested_plan_digest,
        expected_nested_plan_bytes_sha256=expected_nested_plan_bytes_sha256,
        runner_source_path=runner_source_path,
    )
    if replayed != freeze_request:
        raise NestedOpeningFeasibilityExecutionV1Error(
            "test freeze request object differs from full exact replay"
        )
    return _build_external_registration_receipt_after_replay(
        replayed,
        freeze_request_text,
        registration_service=registration_service,
        registration_reference=registration_reference,
        registered_at_utc=registered_at_utc,
        execution_uuid=execution_uuid,
        execution_nonce=execution_nonce,
        execution_deadline_seconds=execution_deadline_seconds,
    )


def serialize_external_nested_opening_feasibility_registration_receipt_for_testing_v1(
    receipt: ExternalNestedOpeningFeasibilityRegistrationReceiptV1,
    freeze_request: NestedOpeningFeasibilityFreezeRequestV1,
    freeze_request_text: str,
    *,
    upstream_census_plan: EvaluationCensusPlanV1,
    upstream_census_plan_text: str,
    nested_plan: NestedOpeningFeasibilityPlanV1,
    nested_plan_text: str,
    expected_upstream_census_plan_digest: str,
    expected_upstream_census_plan_bytes_sha256: str,
    expected_nested_plan_digest: str,
    expected_nested_plan_bytes_sha256: str,
    runner_source_path: Path,
) -> str:
    """Serialize only the nominally distinct reduced engineering receipt."""

    expected = (
        build_external_nested_opening_feasibility_registration_receipt_for_testing_v1(
            freeze_request,
            freeze_request_text,
            upstream_census_plan=upstream_census_plan,
            upstream_census_plan_text=upstream_census_plan_text,
            nested_plan=nested_plan,
            nested_plan_text=nested_plan_text,
            expected_upstream_census_plan_digest=expected_upstream_census_plan_digest,
            expected_upstream_census_plan_bytes_sha256=(
                expected_upstream_census_plan_bytes_sha256
            ),
            expected_nested_plan_digest=expected_nested_plan_digest,
            expected_nested_plan_bytes_sha256=expected_nested_plan_bytes_sha256,
            runner_source_path=runner_source_path,
            registration_service=receipt.registration_service,
            registration_reference=receipt.registration_reference,
            registered_at_utc=receipt.registered_at_utc,
            execution_uuid=receipt.execution_uuid,
            execution_nonce=receipt.execution_nonce,
            execution_deadline_seconds=receipt.execution_deadline_seconds,
        )
    )
    if expected != receipt:
        raise NestedOpeningFeasibilityExecutionV1Error(
            "registration receipt differs from exact test replay"
        )
    return _serialize_external_registration_receipt_after_replay(expected)


def parse_external_nested_opening_feasibility_registration_receipt_v1(
    text: str,
    *,
    freeze_request: NestedOpeningFeasibilityFreezeRequestV1,
    freeze_request_text: str,
    expected_bytes_sha256: str,
    expected_registration_reference: str,
    expected_execution_uuid: str,
    expected_execution_nonce: str,
    expected_execution_deadline_seconds: int,
) -> ExternalNestedOpeningFeasibilityRegistrationReceiptV1:
    expected_sha = _require_sha256(
        expected_bytes_sha256, name="expected registration-receipt sha256"
    )
    raw = text.encode("ascii")
    if _sha256_bytes(raw) != expected_sha:
        raise NestedOpeningFeasibilityExecutionV1Error(
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
            "upstream_evaluation_census_plan_binding",
            "nested_opening_feasibility_plan_binding",
            "claim_boundary",
            "authorization",
            "registration_receipt_digest",
        ),
        name="external registration receipt",
    )
    if (
        _require_integer(obj["schema_version"], name="registration schema version")
        != NESTED_OPENING_FEASIBILITY_EXECUTION_SCHEMA_VERSION
        or obj["receipt_kind"] != _EXTERNAL_REGISTRATION_KIND
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "unknown external registration-receipt schema"
        )
    service = _require_identifier(obj["registration_service"], name="registration service")
    reference = _require_identifier(
        obj["registration_reference"], name="registration reference"
    )
    if reference != _require_identifier(
        expected_registration_reference, name="expected registration reference"
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "registration reference differs from expected"
        )
    timestamp = _require_timestamp(obj["registered_at_utc"])
    registered = _require_mapping(
        obj["registered_execution"],
        (
            "execution_uuid",
            "execution_nonce",
            "execution_deadline_seconds",
            "upstream_identity_schedule_precommitted",
            "nested_construction_plan_precommitted",
            "global_one_shot_consumption_enforced_by_repository",
        ),
        name="registered execution",
    )
    expected_uuid = _require_uuid(expected_execution_uuid)
    expected_nonce = _require_nonce(expected_execution_nonce)
    expected_deadline = _require_integer(
        expected_execution_deadline_seconds,
        name="expected execution deadline seconds",
        minimum=1,
        maximum=86_400,
    )
    if (
        _require_uuid(registered["execution_uuid"]) != expected_uuid
        or _require_nonce(registered["execution_nonce"]) != expected_nonce
        or _require_integer(
            registered["execution_deadline_seconds"],
            name="registered execution deadline seconds",
            minimum=1,
            maximum=86_400,
        )
        != expected_deadline
        or _require_boolean(
            registered["upstream_identity_schedule_precommitted"],
            name="upstream identity schedule precommitted",
        )
        is not True
        or _require_boolean(
            registered["nested_construction_plan_precommitted"],
            name="nested construction plan precommitted",
        )
        is not True
        or _require_boolean(
            registered["global_one_shot_consumption_enforced_by_repository"],
            name="global one-shot enforcement",
        )
        is not False
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "registered execution identity differs from expected"
        )
    expected = _build_external_registration_receipt_after_replay(
        freeze_request,
        freeze_request_text,
        registration_service=service,
        registration_reference=reference,
        registered_at_utc=timestamp,
        execution_uuid=expected_uuid,
        execution_nonce=expected_nonce,
        execution_deadline_seconds=expected_deadline,
    )
    _require_exact_constant(
        obj["claim_boundary"], expected._unsigned_obj()["claim_boundary"], name="claim boundary"
    )
    _require_exact_constant(obj["authorization"], _AUTHORIZATION, name="authorization")
    if _dump_json(value) != _dump_json(expected._artifact_obj_unverified()):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "external registration receipt is inconsistent"
        )
    if _serialize_external_registration_receipt_after_replay(expected) != text:
        raise NestedOpeningFeasibilityExecutionV1Error(
            "external registration receipt is not canonical newline-terminated JSON"
        )
    return expected


@dataclass(frozen=True, slots=True)
class PreparedNestedOpeningFeasibilityArtifactsV1:
    output_root: Path
    upstream_census_plan_path: Path
    nested_plan_path: Path
    freeze_request_path: Path
    upstream_census_plan: EvaluationCensusPlanV1
    nested_plan: NestedOpeningFeasibilityPlanV1
    upstream_census_plan_digest: str
    nested_plan_digest: str
    freeze_request: NestedOpeningFeasibilityFreezeRequestV1
    upstream_census_plan_bytes_sha256: str
    upstream_census_plan_byte_count: int
    nested_plan_bytes_sha256: str
    nested_plan_byte_count: int
    freeze_request_digest: str
    freeze_request_bytes_sha256: str
    freeze_request_byte_count: int


def _secure_absent_root(path: Path) -> tuple[Path, int]:
    if not path.name or path.name in (".", ".."):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "output root must name one new child directory"
        )
    absolute = _lexical_absolute(path)
    parent = absolute.parent
    parent_fd, parent_chain = _open_directory_chain_nofollow(parent)
    try:
        parent_stat = os.fstat(parent_fd)
        if (
            not stat.S_ISDIR(parent_stat.st_mode)
            or parent_stat.st_uid != os.geteuid()
            or stat.S_IMODE(parent_stat.st_mode) & 0o022
        ):
            raise NestedOpeningFeasibilityExecutionV1Error(
                "output-root parent must be owned by this effective user and not group/other writable"
            )
        try:
            os.mkdir(absolute.name, 0o700, dir_fd=parent_fd)
        except FileExistsError as exc:
            raise _OutputRootAlreadyExistsV1(
                "output root already exists; overwrite and partial-root reuse are forbidden"
            ) from exc
        root_flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        if hasattr(os, "O_NOFOLLOW"):
            root_flags |= os.O_NOFOLLOW
        descriptor = os.open(absolute.name, root_flags, dir_fd=parent_fd)
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid():
                raise NestedOpeningFeasibilityExecutionV1Error(
                    "created output root is not a directory"
                )
            os.fchmod(descriptor, 0o700)
            os.fsync(descriptor)
            os.fsync(parent_fd)
            replay_parent_fd = _replay_directory_chain_nofollow(parent, parent_chain)
            try:
                replay_descriptor = os.open(
                    absolute.name, root_flags, dir_fd=replay_parent_fd
                )
                try:
                    replay_stat = os.fstat(replay_descriptor)
                    if (replay_stat.st_dev, replay_stat.st_ino) != (
                        metadata.st_dev,
                        metadata.st_ino,
                    ):
                        raise NestedOpeningFeasibilityExecutionV1Error(
                            "returned output-root pathname names a different inode"
                        )
                finally:
                    os.close(replay_descriptor)
            finally:
                os.close(replay_parent_fd)
        except BaseException:
            os.close(descriptor)
            raise
    finally:
        os.close(parent_fd)
    return absolute, descriptor


def _secure_absent_execution_root_with_failure_fallback(
    path: Path,
    *,
    execution_uuid: str,
) -> tuple[Path, int]:
    absolute = _lexical_absolute(path)
    existed_before = False
    try:
        os.lstat(absolute)
        existed_before = True
    except FileNotFoundError:
        pass
    try:
        return _secure_absent_root(absolute)
    except BaseException as exc:
        # If mkdirat succeeded but a later open/fsync/path replay failed, make
        # one best-effort no-follow recovery of that exact new private root and
        # durably stamp it non-reusable.  This closes the post-mkdir/pre-return
        # ownership gap without ever treating a pre-existing root as ours.
        if existed_before or isinstance(exc, _OutputRootAlreadyExistsV1):
            raise
        parent_fd = -1
        recovered_fd = -1
        try:
            parent_fd, _ = _open_directory_chain_nofollow(absolute.parent)
            flags = (
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0)
            )
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            recovered_fd = os.open(absolute.name, flags, dir_fd=parent_fd)
            recovered = os.fstat(recovered_fd)
            if (
                not stat.S_ISDIR(recovered.st_mode)
                or recovered.st_uid != os.geteuid()
                or stat.S_IMODE(recovered.st_mode) != 0o700
                or os.listdir(recovered_fd)
            ):
                raise NestedOpeningFeasibilityExecutionV1Error(
                    "post-mkdir execution-root recovery found unsafe state"
                )
            failure = _failure_obj(
                execution_uuid=execution_uuid,
                failed_stage="exclusive_root_creation_after_mkdir",
                exc=exc,
                started_receipt_basename_present=False,
                started_receipt_sha256=None,
                root_inventory_at_failure=(),
            )
            _write_exclusive_at(
                recovered_fd, FAILURE_RECEIPT_FILENAME, _canonical_bytes(failure)
            )
        except BaseException:
            pass
        finally:
            if recovered_fd >= 0:
                with suppress(OSError):
                    os.close(recovered_fd)
            if parent_fd >= 0:
                with suppress(OSError):
                    os.close(parent_fd)
        raise


def _verify_prepared_root(
    root: Path,
    directory_fd: int,
    expected_payloads: Mapping[str, bytes],
) -> None:
    root_stat = os.fstat(directory_fd)
    if not stat.S_ISDIR(root_stat.st_mode) or stat.S_IMODE(root_stat.st_mode) != 0o700:
        raise NestedOpeningFeasibilityExecutionV1Error(
            "prepared output root must remain an exact 0700 directory"
        )
    if tuple(sorted(os.listdir(directory_fd))) != tuple(sorted(expected_payloads)):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "prepared output root inventory differs from the exact three artifacts"
        )
    observed_inodes: set[tuple[int, int]] = set()
    observed_mtimes: dict[str, int] = {}
    held_descriptors: dict[str, tuple[int, os.stat_result, bytes]] = {}
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        for name, expected_payload in expected_payloads.items():
            descriptor = os.open(name, flags, dir_fd=directory_fd)
            try:
                before = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(before.st_mode)
                    or stat.S_IMODE(before.st_mode) != 0o400
                    or before.st_nlink != 1
                    or before.st_size != len(expected_payload)
                ):
                    raise NestedOpeningFeasibilityExecutionV1Error(
                        f"prepared artifact {name} has unsafe metadata"
                    )
                payload = _pread_exact(descriptor, before.st_size, name=name)
                after = os.fstat(descriptor)
                if (
                    before.st_dev,
                    before.st_ino,
                    before.st_mode,
                    before.st_nlink,
                    before.st_size,
                    before.st_mtime_ns,
                    before.st_ctime_ns,
                ) != (
                    after.st_dev,
                    after.st_ino,
                    after.st_mode,
                    after.st_nlink,
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                ) or payload != expected_payload:
                    raise NestedOpeningFeasibilityExecutionV1Error(
                        f"prepared artifact {name} changed during read-back"
                    )
                inode = (before.st_dev, before.st_ino)
                if inode in observed_inodes:
                    raise NestedOpeningFeasibilityExecutionV1Error(
                        "prepared artifacts must use distinct inodes"
                    )
                observed_inodes.add(inode)
                observed_mtimes[name] = before.st_mtime_ns
                held_descriptors[name] = (descriptor, before, expected_payload)
            except BaseException:
                os.close(descriptor)
                raise
        if observed_mtimes[FREEZE_REQUEST_FILENAME] < max(
            observed_mtimes[UPSTREAM_CENSUS_PLAN_FILENAME],
            observed_mtimes[NESTED_PLAN_FILENAME],
        ):
            raise NestedOpeningFeasibilityExecutionV1Error(
                "freeze request was not the durable preparation completion marker"
            )
        if tuple(sorted(os.listdir(directory_fd))) != tuple(sorted(expected_payloads)):
            raise NestedOpeningFeasibilityExecutionV1Error(
                "prepared inventory changed after artifact read-back"
            )
        final_root_stat = os.fstat(directory_fd)
        if (
            final_root_stat.st_dev,
            final_root_stat.st_ino,
            final_root_stat.st_mode,
            final_root_stat.st_nlink,
            final_root_stat.st_mtime_ns,
            final_root_stat.st_ctime_ns,
        ) != (
            root_stat.st_dev,
            root_stat.st_ino,
            root_stat.st_mode,
            root_stat.st_nlink,
            root_stat.st_mtime_ns,
            root_stat.st_ctime_ns,
        ):
            raise NestedOpeningFeasibilityExecutionV1Error(
                "prepared root metadata changed during read-back"
            )
        for name, (descriptor, before, expected_payload) in held_descriptors.items():
            basename = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            after = os.fstat(descriptor)
            for current in (basename, after):
                if (
                    current.st_dev,
                    current.st_ino,
                    current.st_mode,
                    current.st_nlink,
                    current.st_size,
                    current.st_mtime_ns,
                    current.st_ctime_ns,
                ) != (
                    before.st_dev,
                    before.st_ino,
                    before.st_mode,
                    before.st_nlink,
                    before.st_size,
                    before.st_mtime_ns,
                    before.st_ctime_ns,
                ):
                    raise NestedOpeningFeasibilityExecutionV1Error(
                        f"prepared artifact {name} changed before return"
                    )
            if _pread_exact(descriptor, before.st_size, name=name) != expected_payload:
                raise NestedOpeningFeasibilityExecutionV1Error(
                    f"prepared artifact {name} bytes changed before return"
                )
        replay_fd, _ = _open_directory_chain_nofollow(root)
        try:
            replay_stat = os.fstat(replay_fd)
            if (
                replay_stat.st_dev,
                replay_stat.st_ino,
                replay_stat.st_mode,
                replay_stat.st_nlink,
                replay_stat.st_mtime_ns,
                replay_stat.st_ctime_ns,
            ) != (
                root_stat.st_dev,
                root_stat.st_ino,
                root_stat.st_mode,
                root_stat.st_nlink,
                root_stat.st_mtime_ns,
                root_stat.st_ctime_ns,
            ):
                raise NestedOpeningFeasibilityExecutionV1Error(
                    "prepared root pathname no longer names the held root"
                )
        finally:
            os.close(replay_fd)
        if tuple(sorted(os.listdir(directory_fd))) != tuple(sorted(expected_payloads)):
            raise NestedOpeningFeasibilityExecutionV1Error(
                "prepared inventory changed during final pathname replay"
            )
    finally:
        for descriptor, _, _ in held_descriptors.values():
            os.close(descriptor)


def _write_exclusive_at(
    directory_fd: int, name: str, payload: bytes, *, mode: int = 0o400
) -> None:
    if "/" in name or name in ("", ".", ".."):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "artifact name must be one safe path component"
        )
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(name, flags, mode, dir_fd=directory_fd)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise NestedOpeningFeasibilityExecutionV1Error("short artifact write")
            view = view[written:]
        os.fchmod(descriptor, mode)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != mode
            or metadata.st_nlink != 1
            or metadata.st_size != len(payload)
            or _pread_exact(descriptor, len(payload), name=name) != payload
        ):
            raise NestedOpeningFeasibilityExecutionV1Error(
                f"artifact {name} failed exact descriptor read-back"
            )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.fsync(directory_fd)


def _require_exact_artifact_at(
    directory_fd: int, name: str, payload: bytes
) -> os.stat_result:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(name, flags, dir_fd=directory_fd)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o400
            or before.st_nlink != 1
            or before.st_size != len(payload)
            or _pread_exact(descriptor, before.st_size, name=name) != payload
        ):
            raise NestedOpeningFeasibilityExecutionV1Error(
                f"artifact {name} differs from its durable exact payload"
            )
        after = os.fstat(descriptor)
        if (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) != (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ):
            raise NestedOpeningFeasibilityExecutionV1Error(
                f"artifact {name} metadata changed during durable read-back"
            )
        return after
    finally:
        os.close(descriptor)


def _durably_require_exact_artifact_at(
    directory_fd: int, name: str, payload: bytes
) -> os.stat_result:
    metadata = _require_exact_artifact_at(directory_fd, name, payload)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(name, flags, dir_fd=directory_fd)
    try:
        replay = os.fstat(descriptor)
        if (replay.st_dev, replay.st_ino) != (metadata.st_dev, metadata.st_ino):
            raise NestedOpeningFeasibilityExecutionV1Error(
                f"artifact {name} changed before durability recovery"
            )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.fsync(directory_fd)
    return _require_exact_artifact_at(directory_fd, name, payload)


def _replay_held_root_path(root: Path, directory_fd: int) -> os.stat_result:
    held = os.fstat(directory_fd)
    if (
        not stat.S_ISDIR(held.st_mode)
        or stat.S_IMODE(held.st_mode) != 0o700
        or held.st_uid != os.geteuid()
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "held execution root is not the exact private directory"
        )
    replay_fd, _ = _open_directory_chain_nofollow(root)
    try:
        replay = os.fstat(replay_fd)
        if (
            replay.st_dev,
            replay.st_ino,
            replay.st_mode,
            replay.st_nlink,
            replay.st_uid,
            replay.st_mtime_ns,
            replay.st_ctime_ns,
        ) != (
            held.st_dev,
            held.st_ino,
            held.st_mode,
            held.st_nlink,
            held.st_uid,
            held.st_mtime_ns,
            held.st_ctime_ns,
        ):
            raise NestedOpeningFeasibilityExecutionV1Error(
                "registered root pathname no longer names the exact held directory"
            )
    finally:
        os.close(replay_fd)
    return held


@dataclass(slots=True)
class _HeldExecutionRootSnapshotV1:
    root: Path
    directory_fd: int
    root_stat: os.stat_result
    expected_payloads: Mapping[str, bytes]
    files: dict[str, tuple[int, os.stat_result]]

    @classmethod
    def acquire(
        cls,
        root: Path,
        directory_fd: int,
        expected_payloads: Mapping[str, bytes],
    ) -> _HeldExecutionRootSnapshotV1:
        expected_names = tuple(sorted(expected_payloads))
        if tuple(sorted(os.listdir(directory_fd))) != expected_names:
            raise NestedOpeningFeasibilityExecutionV1Error(
                "execution-root inventory differs from exact precompletion artifacts"
            )
        root_stat = _replay_held_root_path(root, directory_fd)
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        files: dict[str, tuple[int, os.stat_result]] = {}
        inodes: set[tuple[int, int]] = set()
        try:
            for name in expected_names:
                payload = expected_payloads[name]
                descriptor = os.open(name, flags, dir_fd=directory_fd)
                try:
                    before = os.fstat(descriptor)
                    if (
                        not stat.S_ISREG(before.st_mode)
                        or stat.S_IMODE(before.st_mode) != 0o400
                        or before.st_nlink != 1
                        or before.st_size != len(payload)
                        or (before.st_dev, before.st_ino) in inodes
                    ):
                        raise NestedOpeningFeasibilityExecutionV1Error(
                            f"execution-root artifact {name} has unsafe metadata or inode alias"
                        )
                    inodes.add((before.st_dev, before.st_ino))
                    if _pread_exact(
                        descriptor, before.st_size, name=f"root artifact {name}"
                    ) != payload:
                        raise NestedOpeningFeasibilityExecutionV1Error(
                            f"execution-root artifact {name} bytes differ"
                        )
                    files[name] = (descriptor, before)
                except BaseException:
                    os.close(descriptor)
                    raise
            result = cls(root, directory_fd, root_stat, dict(expected_payloads), files)
            result.replay()
            return result
        except BaseException:
            for descriptor, _ in files.values():
                with suppress(OSError):
                    os.close(descriptor)
            raise

    @classmethod
    def acquire_existing_complete_root(
        cls,
        root: Path,
        directory_fd: int,
        *,
        input_inodes: frozenset[tuple[int, int]],
    ) -> _HeldExecutionRootSnapshotV1:
        maximum_bytes = {
            STARTED_RECEIPT_FILENAME: 64 * 1024 * 1024,
            BUILDER_GUARDIAN_TERMINAL_FILENAME: 64 * 1024 * 1024,
            BUILDER_TERMINAL_FILENAME: 64 * 1024 * 1024,
            VERIFIER_GUARDIAN_TERMINAL_FILENAME: 64 * 1024 * 1024,
            VERIFIER_TERMINAL_FILENAME: 64 * 1024 * 1024,
            REPORT_FILENAME: 1024 * 1024 * 1024,
            EXECUTION_RECEIPT_FILENAME: 64 * 1024 * 1024,
        }
        expected_names = tuple(sorted(maximum_bytes))
        if tuple(sorted(os.listdir(directory_fd))) != expected_names:
            raise NestedOpeningFeasibilityExecutionV1Error(
                "completed execution root must contain exactly seven artifacts"
            )
        root_stat = _replay_held_root_path(root, directory_fd)
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        files: dict[str, tuple[int, os.stat_result]] = {}
        payloads: dict[str, bytes] = {}
        inodes = set(input_inodes)
        try:
            for name in expected_names:
                descriptor = os.open(name, flags, dir_fd=directory_fd)
                try:
                    before = os.fstat(descriptor)
                    inode = (before.st_dev, before.st_ino)
                    if (
                        not stat.S_ISREG(before.st_mode)
                        or stat.S_IMODE(before.st_mode) != 0o400
                        or before.st_nlink != 1
                        or before.st_size < 1
                        or before.st_size > maximum_bytes[name]
                        or inode in inodes
                    ):
                        raise NestedOpeningFeasibilityExecutionV1Error(
                            f"completed-root artifact {name} has unsafe metadata or inode alias"
                        )
                    payload = _pread_exact(
                        descriptor, before.st_size, name=f"completed-root {name}"
                    )
                    after = os.fstat(descriptor)
                    if (
                        after.st_dev,
                        after.st_ino,
                        after.st_mode,
                        after.st_nlink,
                        after.st_size,
                        after.st_mtime_ns,
                        after.st_ctime_ns,
                    ) != (
                        before.st_dev,
                        before.st_ino,
                        before.st_mode,
                        before.st_nlink,
                        before.st_size,
                        before.st_mtime_ns,
                        before.st_ctime_ns,
                    ):
                        raise NestedOpeningFeasibilityExecutionV1Error(
                            f"completed-root artifact {name} changed during acquisition"
                        )
                    inodes.add(inode)
                    payloads[name] = payload
                    files[name] = (descriptor, before)
                except BaseException:
                    os.close(descriptor)
                    raise
            result = cls(root, directory_fd, root_stat, payloads, files)
            result.replay()
            return result
        except BaseException:
            for descriptor, _ in files.values():
                with suppress(OSError):
                    os.close(descriptor)
            raise

    def replay(
        self,
        *,
        additional_names: Sequence[str] = (),
        require_root_timestamps_stable: bool = True,
    ) -> None:
        if tuple(sorted(os.listdir(self.directory_fd))) != tuple(
            sorted((*self.expected_payloads, *additional_names))
        ):
            raise NestedOpeningFeasibilityExecutionV1Error(
                "execution-root inventory changed during held snapshot"
            )
        current_root = _replay_held_root_path(self.root, self.directory_fd)
        current_root_identity = (
            current_root.st_dev,
            current_root.st_ino,
            current_root.st_mode,
            current_root.st_nlink,
            current_root.st_uid,
        )
        initial_root_identity = (
            self.root_stat.st_dev,
            self.root_stat.st_ino,
            self.root_stat.st_mode,
            self.root_stat.st_nlink,
            self.root_stat.st_uid,
        )
        if current_root_identity != initial_root_identity or (
            require_root_timestamps_stable
            and (
                current_root.st_mtime_ns,
                current_root.st_ctime_ns,
            )
            != (self.root_stat.st_mtime_ns, self.root_stat.st_ctime_ns)
        ):
            raise NestedOpeningFeasibilityExecutionV1Error(
                "execution-root metadata changed during held snapshot"
            )
        for name, (descriptor, initial) in self.files.items():
            basename = os.stat(name, dir_fd=self.directory_fd, follow_symlinks=False)
            before = os.fstat(descriptor)
            payload = self.expected_payloads[name]
            observed = _pread_exact(descriptor, len(payload), name=f"root artifact {name}")
            after = os.fstat(descriptor)
            expected_stat = (
                initial.st_dev,
                initial.st_ino,
                initial.st_mode,
                initial.st_nlink,
                initial.st_size,
                initial.st_uid,
                initial.st_mtime_ns,
                initial.st_ctime_ns,
            )
            for current in (basename, before, after):
                if (
                    current.st_dev,
                    current.st_ino,
                    current.st_mode,
                    current.st_nlink,
                    current.st_size,
                    current.st_uid,
                    current.st_mtime_ns,
                    current.st_ctime_ns,
                ) != expected_stat:
                    raise NestedOpeningFeasibilityExecutionV1Error(
                        f"execution-root artifact {name} changed or was replaced"
                    )
            if observed != payload:
                raise NestedOpeningFeasibilityExecutionV1Error(
                    f"execution-root artifact {name} bytes changed"
                )

    def add_exact_file(self, name: str, payload: bytes) -> None:
        if name in self.files or name in self.expected_payloads:
            raise NestedOpeningFeasibilityExecutionV1Error(
                f"execution-root artifact {name} was already held"
            )
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(name, flags, dir_fd=self.directory_fd)
        try:
            before = os.fstat(descriptor)
            held_inodes = {
                (metadata.st_dev, metadata.st_ino)
                for _, metadata in self.files.values()
            }
            if (
                not stat.S_ISREG(before.st_mode)
                or stat.S_IMODE(before.st_mode) != 0o400
                or before.st_nlink != 1
                or before.st_size != len(payload)
                or (before.st_dev, before.st_ino) in held_inodes
                or _pread_exact(descriptor, len(payload), name=name) != payload
            ):
                raise NestedOpeningFeasibilityExecutionV1Error(
                    f"execution-root terminal artifact {name} is unsafe"
                )
            self.files[name] = (descriptor, before)
            self.expected_payloads = {**self.expected_payloads, name: payload}
        except BaseException:
            os.close(descriptor)
            raise
        self.replay(require_root_timestamps_stable=False)

    def require_declared_terminal_order_metadata(self) -> None:
        if (
            STARTED_RECEIPT_FILENAME not in self.files
            or EXECUTION_RECEIPT_FILENAME not in self.files
        ):
            raise NestedOpeningFeasibilityExecutionV1Error(
                "terminal-order replay requires held started and completion receipts"
            )
        started = self.files[STARTED_RECEIPT_FILENAME][1]
        completed = self.files[EXECUTION_RECEIPT_FILENAME][1]
        for name, (_, metadata) in self.files.items():
            if name in (STARTED_RECEIPT_FILENAME, EXECUTION_RECEIPT_FILENAME):
                continue
            if (
                started.st_mtime_ns > metadata.st_mtime_ns
                or started.st_ctime_ns > metadata.st_ctime_ns
                or completed.st_mtime_ns < metadata.st_mtime_ns
                or completed.st_ctime_ns < metadata.st_ctime_ns
            ):
                raise NestedOpeningFeasibilityExecutionV1Error(
                    "held filesystem metadata contradicts controller-declared "
                    "started-first/completion-last order"
                )

    def close(self) -> None:
        for descriptor, _ in self.files.values():
            with suppress(OSError):
                os.close(descriptor)
        self.files.clear()

    def security_snapshot_obj(self) -> dict[str, Any]:
        root_stat = os.fstat(self.directory_fd)
        artifacts: dict[str, Any] = {}
        for name in sorted(self.files):
            descriptor, _ = self.files[name]
            metadata = os.fstat(descriptor)
            payload = self.expected_payloads[name]
            artifacts[name] = {
                "dev": metadata.st_dev,
                "ino": metadata.st_ino,
                "mode": f"{stat.S_IMODE(metadata.st_mode):04o}",
                "nlink": metadata.st_nlink,
                "uid": metadata.st_uid,
                "size": metadata.st_size,
                "mtime_ns": metadata.st_mtime_ns,
                "ctime_ns": metadata.st_ctime_ns,
                "sha256": _sha256_bytes(payload),
                "byte_count": len(payload),
            }
        return {
            "root": {
                "dev": root_stat.st_dev,
                "ino": root_stat.st_ino,
                "mode": f"{stat.S_IMODE(root_stat.st_mode):04o}",
                "nlink": root_stat.st_nlink,
                "uid": root_stat.st_uid,
                "mtime_ns": root_stat.st_mtime_ns,
                "ctime_ns": root_stat.st_ctime_ns,
            },
            "inventory": list(sorted(self.files)),
            "artifacts": artifacts,
            "filesystem_metadata_consistent_with_controller_declared_terminal_order": (
                STARTED_RECEIPT_FILENAME in self.files
                and EXECUTION_RECEIPT_FILENAME in self.files
            ),
            "historical_write_order_independently_verified_from_external_clock": False,
        }


def _prepare(
    upstream_census_seed: str,
    nested_generator_seed: str,
    output_root: Path,
    *,
    runner_source_path: Path,
    plan_class: _PlanClass,
    attempts_per_formula_stratum: int,
    candidate_pool_size: int,
    source_archive_path: Path | None,
    constraints_path: Path | None,
    environment_lock_path: Path | None,
) -> PreparedNestedOpeningFeasibilityArtifactsV1:
    if not _is_sha256(upstream_census_seed):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "upstream census seed must be exactly 64 lowercase hex"
        )
    if not _is_sha256(nested_generator_seed):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "nested generator seed must be exactly 64 lowercase hex"
        )
    if nested_generator_seed == upstream_census_seed:
        raise NestedOpeningFeasibilityExecutionV1Error(
            "upstream census and nested generator seeds must be independently chosen"
        )
    if plan_class == _PRODUCTION_PLAN_CLASS and any(
        path is None
        for path in (
            source_archive_path,
            constraints_path,
            environment_lock_path,
        )
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "production requires source archive, constraints, and environment lock files"
        )
    upstream = build_evaluation_census_plan_v1(
        upstream_census_seed,
        attempts_per_formula_stratum=attempts_per_formula_stratum,
        candidate_pool_size=candidate_pool_size,
    )
    upstream_text = serialize_evaluation_census_plan_v1(upstream)
    upstream_bytes = upstream_text.encode("ascii")
    upstream_sha = _sha256_bytes(upstream_bytes)
    if plan_class == _PRODUCTION_PLAN_CLASS:
        nested = build_nested_opening_construction_feasibility_plan_v1(
            upstream_text,
            expected_census_plan_digest=upstream.digest,
            expected_census_plan_bytes_sha256=upstream_sha,
            generator_seed=nested_generator_seed,
        )
        nested_digest = derive_nested_opening_construction_feasibility_plan_digest_v1(
            nested,
            upstream_text,
            expected_census_plan_digest=upstream.digest,
            expected_census_plan_bytes_sha256=upstream_sha,
        )
        nested_text = serialize_nested_opening_construction_feasibility_plan_v1(
            nested,
            upstream_text,
            expected_census_plan_digest=upstream.digest,
            expected_census_plan_bytes_sha256=upstream_sha,
            expected_plan_digest=nested_digest,
        )
        _assert_production_plans(upstream, nested)
    else:
        nested = build_nested_opening_construction_feasibility_plan_for_testing_v1(
            upstream_text,
            expected_census_plan_digest=upstream.digest,
            expected_census_plan_bytes_sha256=upstream_sha,
            generator_seed=nested_generator_seed,
        )
        nested_digest = (
            derive_nested_opening_construction_feasibility_plan_digest_for_testing_v1(
                nested,
                upstream_text,
                expected_census_plan_digest=upstream.digest,
                expected_census_plan_bytes_sha256=upstream_sha,
            )
        )
        nested_text = serialize_nested_opening_construction_feasibility_plan_for_testing_v1(
            nested,
            upstream_text,
            expected_census_plan_digest=upstream.digest,
            expected_census_plan_bytes_sha256=upstream_sha,
            expected_plan_digest=nested_digest,
        )
    nested_bytes = nested_text.encode("ascii")
    request = _build_freeze_request(
        upstream,
        upstream_bytes,
        nested,
        nested_digest,
        nested_bytes,
        plan_class=plan_class,
        runner_source_path=runner_source_path,
        source_archive_path=source_archive_path,
        constraints_path=constraints_path,
        environment_lock_path=environment_lock_path,
    )
    if plan_class == _PRODUCTION_PLAN_CLASS:
        if source_archive_path is None or constraints_path is None or environment_lock_path is None:
            raise AssertionError("production environment path narrowing failed")
        request_text = serialize_nested_opening_feasibility_freeze_request_v1(
            request,
            upstream_census_plan=upstream,
            upstream_census_plan_text=upstream_text,
            nested_plan=nested,
            nested_plan_text=nested_text,
            expected_upstream_census_plan_digest=upstream.digest,
            expected_upstream_census_plan_bytes_sha256=upstream_sha,
            expected_nested_plan_digest=nested_digest,
            expected_nested_plan_bytes_sha256=_sha256_bytes(nested_bytes),
            runner_source_path=runner_source_path,
            source_archive_path=source_archive_path,
            constraints_path=constraints_path,
            environment_lock_path=environment_lock_path,
        )
    else:
        request_text = serialize_nested_opening_feasibility_freeze_request_for_testing_v1(
            request,
            upstream_census_plan=upstream,
            upstream_census_plan_text=upstream_text,
            nested_plan=nested,
            nested_plan_text=nested_text,
            expected_upstream_census_plan_digest=upstream.digest,
            expected_upstream_census_plan_bytes_sha256=upstream_sha,
            expected_nested_plan_digest=nested_digest,
            expected_nested_plan_bytes_sha256=_sha256_bytes(nested_bytes),
            runner_source_path=runner_source_path,
        )
    request_bytes = request_text.encode("ascii")
    root, directory_fd = _secure_absent_root(output_root)
    try:
        _write_exclusive_at(directory_fd, UPSTREAM_CENSUS_PLAN_FILENAME, upstream_bytes)
        _write_exclusive_at(directory_fd, NESTED_PLAN_FILENAME, nested_bytes)
        _write_exclusive_at(directory_fd, FREEZE_REQUEST_FILENAME, request_bytes)
        _verify_prepared_root(
            root,
            directory_fd,
            {
                UPSTREAM_CENSUS_PLAN_FILENAME: upstream_bytes,
                NESTED_PLAN_FILENAME: nested_bytes,
                FREEZE_REQUEST_FILENAME: request_bytes,
            },
        )
    finally:
        os.close(directory_fd)
    return PreparedNestedOpeningFeasibilityArtifactsV1(
        root,
        root / UPSTREAM_CENSUS_PLAN_FILENAME,
        root / NESTED_PLAN_FILENAME,
        root / FREEZE_REQUEST_FILENAME,
        upstream,
        nested,
        upstream.digest,
        nested_digest,
        request,
        _sha256_bytes(upstream_bytes),
        len(upstream_bytes),
        _sha256_bytes(nested_bytes),
        len(nested_bytes),
        request._digest_unverified(),
        _sha256_bytes(request_bytes),
        len(request_bytes),
    )


def prepare_production_nested_opening_feasibility_v1(
    upstream_census_seed: str,
    nested_generator_seed: str,
    output_root: Path,
    *,
    runner_source_path: Path,
    source_archive_path: Path,
    constraints_path: Path,
    environment_lock_path: Path,
) -> PreparedNestedOpeningFeasibilityArtifactsV1:
    """Persist—but do not externally freeze or execute—the exact production plans."""

    return _prepare(
        upstream_census_seed,
        nested_generator_seed,
        output_root,
        runner_source_path=runner_source_path,
        plan_class=_PRODUCTION_PLAN_CLASS,
        attempts_per_formula_stratum=PRODUCTION_MIRROR_ATTEMPTS_PER_STRATUM,
        candidate_pool_size=DEFAULT_CANDIDATE_POOL_SIZE,
        source_archive_path=source_archive_path,
        constraints_path=constraints_path,
        environment_lock_path=environment_lock_path,
    )


def prepare_engineering_nested_opening_feasibility_fixture_for_testing_v1(
    upstream_census_seed: str,
    nested_generator_seed: str,
    output_root: Path,
    *,
    runner_source_path: Path,
    attempts_per_formula_stratum: int = 1,
    candidate_pool_size: int = 1,
) -> PreparedNestedOpeningFeasibilityArtifactsV1:
    """Prepare a reduced fixture; deliberately unavailable from the production CLI."""

    if (
        attempts_per_formula_stratum == PRODUCTION_MIRROR_ATTEMPTS_PER_STRATUM
        and candidate_pool_size == DEFAULT_CANDIDATE_POOL_SIZE
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "the engineering-test API must not emit the production plan"
        )
    return _prepare(
        upstream_census_seed,
        nested_generator_seed,
        output_root,
        runner_source_path=runner_source_path,
        plan_class=_ENGINEERING_PLAN_CLASS,
        attempts_per_formula_stratum=attempts_per_formula_stratum,
        candidate_pool_size=candidate_pool_size,
        source_archive_path=None,
        constraints_path=None,
        environment_lock_path=None,
    )


# Execution and complete-root verification are implemented below.  They use
# controller-owned held descriptors and pass only explicit duplicates through
# each watchdog to its one worker; no worker reopens a held data-input pathname.
# Code and imported source identity are separately revalidated by canonical
# pathname and hash, with the documented pre-first-byte/same-UID limitation.


@dataclass(slots=True)
class _HeldOrdinaryFileV1:
    label: str
    supplied_path: Path
    resolved_parent: Path
    basename: str
    parent_directory_fd: int
    parent_chain: tuple[tuple[int, int, str], ...]
    file_fd: int
    parent_dev: int
    parent_ino: int
    file_dev: int
    file_ino: int
    file_mode: int
    file_nlink: int
    file_mtime_ns: int
    file_ctime_ns: int
    payload_sha256: str
    payload_byte_count: int

    @classmethod
    def open(
        cls,
        path: Path,
        *,
        label: str,
        maximum_bytes: int | None = None,
        expected_modes: frozenset[int] | None = None,
    ) -> _HeldOrdinaryFileV1:
        if not path.name or path.name in (".", ".."):
            raise NestedOpeningFeasibilityExecutionV1Error(
                f"{label} must name one file"
            )
        absolute = _lexical_absolute(path)
        parent = absolute.parent
        parent_fd, parent_chain = _open_directory_chain_nofollow(parent)
        file_fd = -1
        try:
            parent_stat = os.fstat(parent_fd)
            if not stat.S_ISDIR(parent_stat.st_mode):
                raise NestedOpeningFeasibilityExecutionV1Error(
                    f"{label} parent must be a directory"
                )
            file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            if hasattr(os, "O_NOFOLLOW"):
                file_flags |= os.O_NOFOLLOW
            file_fd = os.open(absolute.name, file_flags, dir_fd=parent_fd)
            file_stat = os.fstat(file_fd)
            if not stat.S_ISREG(file_stat.st_mode):
                raise NestedOpeningFeasibilityExecutionV1Error(
                    f"{label} must be one ordinary file"
                )
            if file_stat.st_nlink != 1:
                raise NestedOpeningFeasibilityExecutionV1Error(
                    f"{label} must have exactly one hard link"
                )
            observed_mode = stat.S_IMODE(file_stat.st_mode)
            if expected_modes is not None and observed_mode not in expected_modes:
                expected_text = ", ".join(f"{mode:04o}" for mode in sorted(expected_modes))
                raise NestedOpeningFeasibilityExecutionV1Error(
                    f"{label} mode must be exactly one of {expected_text}"
                )
            if maximum_bytes is not None and file_stat.st_size > maximum_bytes:
                raise NestedOpeningFeasibilityExecutionV1Error(
                    f"{label} exceeds the maximum byte count"
                )
            payload = _pread_exact(file_fd, file_stat.st_size, name=label)
            final_stat = os.fstat(file_fd)
            if (
                final_stat.st_dev,
                final_stat.st_ino,
                stat.S_IFMT(final_stat.st_mode),
                final_stat.st_nlink,
                final_stat.st_size,
                final_stat.st_mtime_ns,
                final_stat.st_ctime_ns,
            ) != (
                file_stat.st_dev,
                file_stat.st_ino,
                stat.S_IFMT(file_stat.st_mode),
                file_stat.st_nlink,
                file_stat.st_size,
                file_stat.st_mtime_ns,
                file_stat.st_ctime_ns,
            ):
                raise NestedOpeningFeasibilityExecutionV1Error(
                    f"{label} changed while initially held"
                )
            return cls(
                label,
                absolute,
                parent,
                absolute.name,
                parent_fd,
                parent_chain,
                file_fd,
                parent_stat.st_dev,
                parent_stat.st_ino,
                file_stat.st_dev,
                file_stat.st_ino,
                file_stat.st_mode,
                file_stat.st_nlink,
                file_stat.st_mtime_ns,
                file_stat.st_ctime_ns,
                _sha256_bytes(payload),
                len(payload),
            )
        except BaseException:
            if file_fd >= 0:
                os.close(file_fd)
            os.close(parent_fd)
            raise

    def read_bytes(self, *, maximum_bytes: int | None = None) -> bytes:
        metadata = os.fstat(self.file_fd)
        if maximum_bytes is not None and metadata.st_size > maximum_bytes:
            raise NestedOpeningFeasibilityExecutionV1Error(
                f"{self.label} exceeds the maximum byte count"
            )
        self._require_same_held_file(metadata)
        payload = _pread_exact(self.file_fd, self.payload_byte_count, name=self.label)
        self._require_same_held_file(os.fstat(self.file_fd))
        if _sha256_bytes(payload) != self.payload_sha256:
            raise NestedOpeningFeasibilityExecutionV1Error(
                f"{self.label} held bytes changed"
            )
        return payload

    def read_ascii(self, *, maximum_bytes: int | None = None) -> tuple[str, bytes]:
        payload = self.read_bytes(maximum_bytes=maximum_bytes)
        try:
            return payload.decode("ascii"), payload
        except UnicodeDecodeError as exc:
            raise NestedOpeningFeasibilityExecutionV1Error(
                f"{self.label} must be ASCII"
            ) from exc

    def _require_same_held_file(self, metadata: os.stat_result) -> None:
        if (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_nlink,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        ) != (
            self.file_dev,
            self.file_ino,
            self.file_mode,
            self.file_nlink,
            self.payload_byte_count,
            self.file_mtime_ns,
            self.file_ctime_ns,
        ):
            raise NestedOpeningFeasibilityExecutionV1Error(
                f"{self.label} held file identity or metadata changed"
            )

    def replay_external_basename(self) -> None:
        replay_fd = _replay_directory_chain_nofollow(
            self.resolved_parent, self.parent_chain
        )
        os.close(replay_fd)
        parent_now = os.fstat(self.parent_directory_fd)
        if (parent_now.st_dev, parent_now.st_ino) != (
            self.parent_dev,
            self.parent_ino,
        ):
            raise NestedOpeningFeasibilityExecutionV1Error(
                f"{self.label} parent directory changed"
            )
        try:
            basename_stat = os.stat(
                self.basename,
                dir_fd=self.parent_directory_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError as exc:
            raise NestedOpeningFeasibilityExecutionV1Error(
                f"{self.label} external basename disappeared"
            ) from exc
        self._require_same_held_file(basename_stat)
        self.read_bytes()

    def binding(self) -> dict[str, Any]:
        return {
            "sha256": self.payload_sha256,
            "byte_count": self.payload_byte_count,
            "mode": f"{stat.S_IMODE(self.file_mode):04o}",
        }

    def close(self) -> None:
        for field in ("file_fd", "parent_directory_fd"):
            descriptor = getattr(self, field)
            if descriptor >= 0:
                with suppress(OSError):
                    os.close(descriptor)
                setattr(self, field, -1)


def _close_held_files(files: Mapping[str, _HeldOrdinaryFileV1]) -> None:
    for held in files.values():
        held.close()


def _held_input_bindings(
    files: Mapping[str, _HeldOrdinaryFileV1],
) -> dict[str, Any]:
    return {label: held.binding() for label, held in files.items()}


def _worker_read_held_inputs(args: argparse.Namespace) -> dict[str, bytes]:
    fds_value = _load_json(cast(str, args.held_input_fds_json))
    bindings_value = _load_json(cast(str, args.held_input_bindings_json))
    if (
        type(fds_value) is not dict
        or not fds_value
        or any(
            type(label) is not str
            or isinstance(descriptor, bool)
            or not isinstance(descriptor, int)
            or descriptor < 0
            for label, descriptor in fds_value.items()
        )
        or _dump_json(fds_value) != cast(str, args.held_input_fds_json)
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "held-input descriptor map is not canonical"
        )
    if (
        type(bindings_value) is not dict
        or tuple(bindings_value) != tuple(fds_value)
        or _dump_json(bindings_value) != cast(str, args.held_input_bindings_json)
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "held-input binding labels differ from descriptor labels"
        )
    expected_labels = (
        (
            "upstream_census_plan",
            "nested_plan",
            "freeze_request",
            "registration_receipt",
            "source_archive",
            "constraints",
            "environment_lock",
        )
        if args.require_production
        else (
            "upstream_census_plan",
            "nested_plan",
            "freeze_request",
            "registration_receipt",
        )
    )
    if tuple(fds_value) != expected_labels:
        raise NestedOpeningFeasibilityExecutionV1Error(
            "worker held-input labels differ from the exact nominal API boundary"
        )
    descriptor_values = tuple(cast(dict[str, int], fds_value).values())
    if len(set(descriptor_values)) != len(descriptor_values):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "worker held-input descriptors must be pairwise distinct"
        )
    result: dict[str, bytes] = {}
    for label, descriptor_value in cast(dict[str, int], fds_value).items():
        binding = _require_mapping(
            cast(dict[str, Any], bindings_value)[label],
            ("sha256", "byte_count", "mode"),
            name=f"{label} held-input binding",
        )
        descriptor = _require_integer(
            descriptor_value, name=f"{label} descriptor", minimum=0
        )
        expected_sha = _require_sha256(binding["sha256"], name=f"{label} sha256")
        expected_count = _require_integer(
            binding["byte_count"], name=f"{label} byte count", minimum=1
        )
        if expected_count > _HELD_INPUT_MAXIMUM_BYTES[label]:
            raise NestedOpeningFeasibilityExecutionV1Error(
                f"{label} exceeds its lifecycle byte cap"
            )
        if binding["mode"] != "0400":
            raise NestedOpeningFeasibilityExecutionV1Error(
                f"{label} held-input mode binding must be exact 0400"
            )
        os.set_inheritable(descriptor, False)
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o400
            or before.st_nlink != 1
        ):
            raise NestedOpeningFeasibilityExecutionV1Error(
                f"{label} inherited descriptor is not one-link ordinary file"
            )
        payload = _pread_exact(descriptor, expected_count, name=label)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise NestedOpeningFeasibilityExecutionV1Error(
                f"{label} changed while worker read held bytes"
            )
        if len(payload) != expected_count or _sha256_bytes(payload) != expected_sha:
            raise NestedOpeningFeasibilityExecutionV1Error(
                f"{label} held bytes differ from controller binding"
            )
        result[label] = payload
    return result


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


def _report_accounting(report: NestedOpeningFeasibilityReportV1) -> dict[str, Any]:
    members = tuple(
        member for attempt in report.attempts for member in attempt.members
    )
    coordinates = tuple((row.m, row.q) for row in report.observed_m_q_summary)
    if coordinates != _EXPECTED_CELL_COORDINATES:
        raise NestedOpeningFeasibilityExecutionV1Error(
            "report does not preserve all 36 canonical cells"
        )
    completed_assessments = sum(len(member.openings or ()) for member in members)
    common_union_completed = sum(member.status == "completed" for member in members)
    member_exact = sum(
        member.exact_match is not None and member.exact_match.passed
        for member in members
    )
    pair_disjoint = sum(attempt.mirror_pair_disjoint for attempt in report.attempts)
    pair_full_common_key = 0
    rederived_pair_witnesses = 0
    for attempt in report.attempts:
        left, right = attempt.members
        full_key_match = (
            left.exact_match is not None
            and right.exact_match is not None
            and left.exact_match.passed
            and right.exact_match.passed
            and left.exact_match.common_key is not None
            and right.exact_match.common_key is not None
            and left.exact_match.common_key == right.exact_match.common_key
        )
        if full_key_match:
            pair_full_common_key += 1
        rederived = bool(
            left.member_feasibility_witness
            and right.member_feasibility_witness
            and attempt.mirror_pair_disjoint
            and full_key_match
        )
        if attempt.mirror_pair_feasibility_witness is not rederived:
            raise NestedOpeningFeasibilityExecutionV1Error(
                "stored mirror-pair witness differs from per-attempt full common-key replay"
            )
        rederived_pair_witnesses += rederived
    composed = tuple(member.member_plan.composed_rule_id for member in members)
    return {
        "mirror_attempt_count": len(report.attempts),
        "member_record_count": len(members),
        "distinct_evaluation_c_identity_count": len(set(composed)),
        "planned_opening_assessment_count": (
            len(members) * DERIVED_OPENINGS_PER_MEMBER
        ),
        "completed_opening_assessment_count": completed_assessments,
        "common_union_completed_count": common_union_completed,
        "bounded_slot_failure_count": len(members) - common_union_completed,
        "member_exact_match_witness_count": member_exact,
        "nested_opening_feasibility_witness_count": sum(
            member.member_feasibility_witness for member in members
        ),
        "mirror_pair_disjoint_count": pair_disjoint,
        "mirror_pair_full_common_key_match_count": pair_full_common_key,
        "mirror_pair_feasibility_witness_count": rederived_pair_witnesses,
        "protocol_quartet_candidate_count": 0,
        "observed_cell_count": len(coordinates),
        "observed_cell_coordinates_digest": _digest(
            [[m, q] for m, q in coordinates], domain=_CELL_COORDINATE_DOMAIN
        ),
        "all_planned_attempts_preserved": True,
        "all_36_cells_preserved": True,
        "early_stop_used": False,
        "identity_replacement_used": False,
        "exact_144x2_budget_complete": report.exact_144x2_budget_complete,
    }


def _validate_report_claim_boundary(
    report: NestedOpeningFeasibilityReportV1,
    report_text: str,
) -> None:
    value = _load_json(report_text)
    if type(value) is not dict:
        raise NestedOpeningFeasibilityExecutionV1Error("report must be one JSON object")
    authorization = value.get("authorization")
    if type(authorization) is not dict or any(
        item is not False for key, item in authorization.items() if key != "scope"
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "report authorization must remain false"
        )
    if value.get("protocol_quartet_candidate") is not False:
        raise NestedOpeningFeasibilityExecutionV1Error(
            "report cannot be a protocol-quartet candidate"
        )
    selection = value.get("selection_boundary")
    if type(selection) is not dict or any(item is not None for item in selection.values()):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "report cannot contain a selection, bank, or matcher choice"
        )
    boundary = value.get("construction_claim_boundary")
    if type(boundary) is not dict:
        raise NestedOpeningFeasibilityExecutionV1Error(
            "report construction claim boundary is missing"
        )
    for field in (
        "protocol_quartet_candidate",
        "production_nested_opening_pool_frozen",
        "positive_m_q_quota_selected",
        "matched_bank_size_selected",
        "difficulty_matcher_run",
        "model_calls_observed",
        "model_outcomes_present",
        "runtime_measurements_present",
        "production_bank_authorized",
        "g01_authorized",
        "g03_capability_launch_authorized",
        "g03_scientific_launch_authorized",
        "launch_authorized",
    ):
        if boundary.get(field) is not False:
            raise NestedOpeningFeasibilityExecutionV1Error(f"report overclaims {field}")
    _report_accounting(report)


def _build_or_parse_report(
    *,
    for_testing: bool,
    upstream_text: str,
    upstream_digest: str,
    upstream_sha256: str,
    nested_text: str,
    nested_digest: str,
    report_text: str | None,
    report_digest: str | None,
) -> tuple[NestedOpeningFeasibilityReportV1, str, bytes]:
    if report_text is None:
        if for_testing:
            report = build_nested_opening_construction_feasibility_report_for_testing_v1(
                nested_text,
                census_plan_text=upstream_text,
                expected_census_plan_digest=upstream_digest,
                expected_census_plan_bytes_sha256=upstream_sha256,
                expected_plan_digest=nested_digest,
            )
            derived = (
                derive_nested_opening_construction_feasibility_report_digest_for_testing_v1(
                    report,
                    nested_text,
                    census_plan_text=upstream_text,
                    expected_census_plan_digest=upstream_digest,
                    expected_census_plan_bytes_sha256=upstream_sha256,
                    expected_plan_digest=nested_digest,
                )
            )
            serialized = (
                serialize_nested_opening_construction_feasibility_report_for_testing_v1(
                    report,
                    nested_text,
                    census_plan_text=upstream_text,
                    expected_census_plan_digest=upstream_digest,
                    expected_census_plan_bytes_sha256=upstream_sha256,
                    expected_plan_digest=nested_digest,
                    expected_report_digest=derived,
                )
            )
        else:
            report = build_nested_opening_construction_feasibility_report_v1(
                nested_text,
                census_plan_text=upstream_text,
                expected_census_plan_digest=upstream_digest,
                expected_census_plan_bytes_sha256=upstream_sha256,
                expected_plan_digest=nested_digest,
            )
            derived = derive_nested_opening_construction_feasibility_report_digest_v1(
                report,
                nested_text,
                census_plan_text=upstream_text,
                expected_census_plan_digest=upstream_digest,
                expected_census_plan_bytes_sha256=upstream_sha256,
                expected_plan_digest=nested_digest,
            )
            serialized = serialize_nested_opening_construction_feasibility_report_v1(
                report,
                nested_text,
                census_plan_text=upstream_text,
                expected_census_plan_digest=upstream_digest,
                expected_census_plan_bytes_sha256=upstream_sha256,
                expected_plan_digest=nested_digest,
                expected_report_digest=derived,
            )
    else:
        expected_report_digest = _require_sha256(
            report_digest, name="expected report digest"
        )
        if for_testing:
            report = parse_nested_opening_construction_feasibility_report_for_testing_v1(
                report_text,
                plan_text=nested_text,
                census_plan_text=upstream_text,
                expected_census_plan_digest=upstream_digest,
                expected_census_plan_bytes_sha256=upstream_sha256,
                expected_plan_digest=nested_digest,
                expected_report_digest=expected_report_digest,
            )
        else:
            report = parse_nested_opening_construction_feasibility_report_v1(
                report_text,
                plan_text=nested_text,
                census_plan_text=upstream_text,
                expected_census_plan_digest=upstream_digest,
                expected_census_plan_bytes_sha256=upstream_sha256,
                expected_plan_digest=nested_digest,
                expected_report_digest=expected_report_digest,
            )
        derived = expected_report_digest
        serialized = report_text
    _validate_report_claim_boundary(report, serialized)
    return report, derived, serialized.encode("ascii")


def _worker_terminal(
    *,
    kind: str,
    started_monotonic_ns: int,
    deadline_monotonic_ns: int,
    upstream_digest: str,
    upstream_sha256: str,
    upstream_byte_count: int,
    nested_digest: str,
    nested_sha256: str,
    nested_byte_count: int,
    report: NestedOpeningFeasibilityReportV1,
    report_digest: str,
    report_bytes: bytes,
    frozen_byte_revalidation: Mapping[str, Any],
    started_wall_utc: str,
    observed_environment: Mapping[str, str],
) -> dict[str, Any]:
    accounting = _report_accounting(report)
    process_identity = {
        "pid": os.getpid(),
        "argv": list(sys.argv),
        "working_directory": str(Path.cwd()),
        "host": _host_identity(),
        "python": _python_identity(),
        "environment": dict(observed_environment),
    }
    maximum_resident_set_size = _process_max_rss()
    if dict(observed_environment) != _observed_safe_environment(
        bootstrap_role="--g03-bootstrap-worker"
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "worker environment changed before terminal admission"
        )
    admitted = time.monotonic_ns()
    admitted_wall = _utc_now()
    unsigned = {
        "schema_version": NESTED_OPENING_FEASIBILITY_EXECUTION_SCHEMA_VERSION,
        "terminal_kind": kind,
        "status": "success",
        "process_identity": process_identity,
        "deadline_monotonic_ns": deadline_monotonic_ns,
        "timing": {
            "started_monotonic_ns": started_monotonic_ns,
            "terminal_admitted_monotonic_ns": admitted,
            "elapsed_to_terminal_admission_ns": admitted - started_monotonic_ns,
            "started_wall_utc": started_wall_utc,
            "terminal_admitted_wall_utc": admitted_wall,
        },
        "maximum_resident_set_size": maximum_resident_set_size,
        "input_plan_bindings": {
            "upstream_evaluation_census_plan": {
                "evaluation_census_plan_digest": upstream_digest,
                "canonical_census_plan_bytes_sha256": upstream_sha256,
                "canonical_census_plan_byte_count": upstream_byte_count,
            },
            "nested_opening_feasibility_plan": {
                "prospective_plan_digest": nested_digest,
                "canonical_plan_bytes_sha256": nested_sha256,
                "canonical_plan_byte_count": nested_byte_count,
            },
        },
        "frozen_byte_revalidation": dict(frozen_byte_revalidation),
        "report_binding": {
            "observed_report_digest": report_digest,
            "exact_report_bytes_sha256": _sha256_bytes(report_bytes),
            "exact_report_byte_count": len(report_bytes),
        },
        "accounting": accounting,
        "execution_scope": dict(_ENGINEERING_EXECUTION_SCOPE),
        "authorization": dict(_AUTHORIZATION),
    }
    return {
        **unsigned,
        "worker_terminal_digest": _digest(unsigned, domain=_WORKER_TERMINAL_DOMAIN),
    }


def _start_parent_guardian(
    parent_guard_fd: int, guardian_ready_write_fd: int
) -> None:
    """Kill this worker group if its owning watchdog disappears."""

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

    threading.Thread(
        target=watch, name="nested-opening-watchdog-death-guard", daemon=True
    ).start()


def _worker_common(args: argparse.Namespace) -> tuple[
    int,
    str,
    dict[str, bytes],
    str,
    bytes,
    str,
    bytes,
    str,
    str,
    bool,
]:
    if bool(args.require_production):
        raise NestedOpeningFeasibilityExecutionV1Error(
            PRODUCTION_EXECUTION_REFUSAL_CODE
        )
    if os.environ.get(_BOOTSTRAP_MARKER) != "--g03-bootstrap-worker":
        raise NestedOpeningFeasibilityExecutionV1Error(
            "worker was not launched through the bound early-guardian bootstrap"
        )
    fds_value = _load_json(cast(str, args.held_input_fds_json))
    if type(fds_value) is not dict:
        raise NestedOpeningFeasibilityExecutionV1Error(
            "held-input descriptor map must be an object"
        )
    capability_fds = [
        _require_integer(
            args.worker_lifetime_write_fd, name="worker lifetime fd", minimum=0
        ),
        _require_integer(args.parent_guard_fd, name="parent guard fd", minimum=0),
        _require_integer(
            args.guardian_ready_write_fd, name="guardian ready fd", minimum=0
        ),
        *[
            _require_integer(value, name=f"{label} fd", minimum=0)
            for label, value in cast(dict[str, object], fds_value).items()
        ],
    ]
    for optional_name in ("output_root_fd", "report_fd"):
        value = getattr(args, optional_name, None)
        if value is not None:
            capability_fds.append(
                _require_integer(value, name=optional_name.replace("_", " "), minimum=0)
            )
    if len(set(capability_fds)) != len(capability_fds):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "all inherited worker capabilities must be pairwise distinct"
        )
    for descriptor in capability_fds:
        os.set_inheritable(descriptor, False)
    semantic_fds = {
        **{
            label: _require_integer(value, name=f"{label} fd", minimum=0)
            for label, value in cast(dict[str, object], fds_value).items()
        },
        **{
            name: _require_integer(
                getattr(args, name), name=name.replace("_", " "), minimum=0
            )
            for name in ("output_root_fd", "report_fd")
            if getattr(args, name, None) is not None
        },
    }
    semantic_inodes: dict[tuple[int, int], str] = {}
    for label, descriptor in semantic_fds.items():
        metadata = os.fstat(descriptor)
        inode = (metadata.st_dev, metadata.st_ino)
        if inode in semantic_inodes:
            raise NestedOpeningFeasibilityExecutionV1Error(
                f"semantic capabilities {semantic_inodes[inode]} and {label} alias one inode"
            )
        semantic_inodes[inode] = label
    # The stdlib-only bootstrap started the guardian before importing this
    # package and already sent the sole ready byte.  This module must not
    # acknowledge readiness a second time.
    held = _worker_read_held_inputs(args)
    try:
        upstream_text = held["upstream_census_plan"].decode("ascii")
        nested_text = held["nested_plan"].decode("ascii")
        freeze_text = held["freeze_request"].decode("ascii")
        registration_text = held["registration_receipt"].decode("ascii")
    except UnicodeDecodeError as exc:
        raise NestedOpeningFeasibilityExecutionV1Error(
            "held JSON artifact is not ASCII"
        ) from exc
    upstream_digest = _require_sha256(
        args.expected_upstream_plan_digest, name="expected upstream plan digest"
    )
    upstream_sha = _require_sha256(
        args.expected_upstream_plan_sha256, name="expected upstream plan sha256"
    )
    nested_digest = _require_sha256(
        args.expected_nested_plan_digest, name="expected nested plan digest"
    )
    nested_sha = _require_sha256(
        args.expected_nested_plan_sha256, name="expected nested plan sha256"
    )
    if (
        _sha256_bytes(held["upstream_census_plan"]) != upstream_sha
        or _sha256_bytes(held["nested_plan"]) != nested_sha
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "worker plan bytes differ from exact pins"
        )
    for_testing = not bool(args.require_production)
    upstream = parse_evaluation_census_plan_v1(
        upstream_text, expected_digest=upstream_digest
    )
    if for_testing:
        nested = parse_nested_opening_construction_feasibility_plan_for_testing_v1(
            nested_text,
            census_plan_text=upstream_text,
            expected_census_plan_digest=upstream_digest,
            expected_census_plan_bytes_sha256=upstream_sha,
            expected_plan_digest=nested_digest,
        )
    else:
        nested = parse_nested_opening_construction_feasibility_plan_v1(
            nested_text,
            census_plan_text=upstream_text,
            expected_census_plan_digest=upstream_digest,
            expected_census_plan_bytes_sha256=upstream_sha,
            expected_plan_digest=nested_digest,
        )
        _assert_production_plans(upstream, nested)
    current_code_binding = _execution_code_binding(Path(cast(str, args.runner_source)))
    if for_testing:
        environment_bindings = _environment_input_bindings_from_payloads({})
        plan_class = _ENGINEERING_PLAN_CLASS
    else:
        environment_bindings = _environment_input_bindings_from_payloads(
            {
                "source_archive": held["source_archive"],
                "constraints": held["constraints"],
                "environment_lock": held["environment_lock"],
            }
        )
        plan_class = _PRODUCTION_PLAN_CLASS
    expected_freeze = NestedOpeningFeasibilityFreezeRequestV1(
        plan_class,
        _upstream_plan_binding(upstream, held["upstream_census_plan"]),
        _nested_plan_binding(nested, nested_digest, held["nested_plan"]),
        upstream.source_binding.as_obj(),
        upstream.catalog_binding.as_obj(),
        upstream.generator_binding.as_obj(),
        nested.source_binding.as_obj(),
        nested.upstream_identity_plan_binding.as_obj(),
        nested.generator_binding.as_obj(),
        current_code_binding,
        environment_bindings,
        _fixed_budget(nested),
    )
    expected_freeze_digest = _require_sha256(
        args.expected_freeze_request_digest,
        name="expected freeze-request digest",
    )
    if (
        expected_freeze._digest_unverified() != expected_freeze_digest
        or _serialize_freeze_request_after_replay(expected_freeze)
        != freeze_text
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "worker freeze request differs from full held-chain replay"
        )
    registration = (
        parse_external_nested_opening_feasibility_registration_receipt_v1(
            registration_text,
            freeze_request=expected_freeze,
            freeze_request_text=freeze_text,
            expected_bytes_sha256=cast(
                str, args.expected_registration_receipt_sha256
            ),
            expected_registration_reference=cast(
                str, args.expected_registration_reference
            ),
            expected_execution_uuid=cast(str, args.expected_execution_uuid),
            expected_execution_nonce=cast(str, args.expected_execution_nonce),
            expected_execution_deadline_seconds=_require_integer(
                args.expected_execution_deadline_seconds,
                name="expected execution deadline seconds",
                minimum=1,
                maximum=86_400,
            ),
        )
    )
    if _registration_datetime(registration.registered_at_utc) > _wall_datetime(
        cast(str, args.controller_started_wall_utc),
        name="controller started wall UTC",
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "held registration timestamp is after controller start"
        )
    return (
        _require_integer(
            args.deadline_monotonic_ns, name="deadline monotonic ns", minimum=1
        ),
        _utc_now(),
        held,
        upstream_text,
        held["upstream_census_plan"],
        nested_text,
        held["nested_plan"],
        upstream_digest,
        nested_digest,
        for_testing,
    )


def _worker_build(args: argparse.Namespace) -> int:
    started = time.monotonic_ns()
    started_wall = _utc_now()
    observed_environment = _observed_safe_environment(
        bootstrap_role="--g03-bootstrap-worker"
    )
    (
        deadline,
        _controller_replay_wall,
        held,
        upstream_text,
        upstream_bytes,
        nested_text,
        nested_bytes,
        upstream_digest,
        nested_digest,
        for_testing,
    ) = _worker_common(args)
    if time.monotonic_ns() > deadline:
        raise NestedOpeningFeasibilityExecutionV1Error(
            "deadline expired before nested report construction"
        )
    report, report_digest, report_bytes = _build_or_parse_report(
        for_testing=for_testing,
        upstream_text=upstream_text,
        upstream_digest=upstream_digest,
        upstream_sha256=_sha256_bytes(upstream_bytes),
        nested_text=nested_text,
        nested_digest=nested_digest,
        report_text=None,
        report_digest=None,
    )
    root_fd = _require_integer(
        args.output_root_fd, name="output root directory fd", minimum=0
    )
    os.set_inheritable(root_fd, False)
    root_stat = os.fstat(root_fd)
    if not stat.S_ISDIR(root_stat.st_mode) or stat.S_IMODE(root_stat.st_mode) != 0o700:
        raise NestedOpeningFeasibilityExecutionV1Error(
            "builder output-root descriptor is not the exact private directory"
        )
    _write_exclusive_at(root_fd, ".observed-report.spool", report_bytes)
    terminal = _worker_terminal(
        kind=_BUILDER_TERMINAL_KIND,
        started_monotonic_ns=started,
        deadline_monotonic_ns=deadline,
        upstream_digest=upstream_digest,
        upstream_sha256=_sha256_bytes(upstream_bytes),
        upstream_byte_count=len(upstream_bytes),
        nested_digest=nested_digest,
        nested_sha256=_sha256_bytes(nested_bytes),
        nested_byte_count=len(nested_bytes),
        report=report,
        report_digest=report_digest,
        report_bytes=report_bytes,
        frozen_byte_revalidation={
            "held_input_bindings": {
                label: {
                    "sha256": _sha256_bytes(payload),
                    "byte_count": len(payload),
                }
                for label, payload in held.items()
            },
            "all_held_data_inputs_read_by_pread": True,
            "no_held_data_input_path_reopened": True,
            "canonical_code_paths_reopened_for_hash_revalidation": True,
        },
        started_wall_utc=started_wall,
        observed_environment=observed_environment,
    )
    sys.stdout.buffer.write(_canonical_bytes(terminal))
    sys.stdout.buffer.flush()
    return 0


def _worker_verify(args: argparse.Namespace) -> int:
    started = time.monotonic_ns()
    started_wall = _utc_now()
    observed_environment = _observed_safe_environment(
        bootstrap_role="--g03-bootstrap-worker"
    )
    (
        deadline,
        _controller_replay_wall,
        held,
        upstream_text,
        upstream_bytes,
        nested_text,
        nested_bytes,
        upstream_digest,
        nested_digest,
        for_testing,
    ) = _worker_common(args)
    report_fd = _require_integer(args.report_fd, name="held report fd", minimum=0)
    os.set_inheritable(report_fd, False)
    report_stat = os.fstat(report_fd)
    if (
        not stat.S_ISREG(report_stat.st_mode)
        or stat.S_IMODE(report_stat.st_mode) != 0o400
        or report_stat.st_nlink != 1
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "held report descriptor is not an ordinary file"
        )
    expected_report_sha = _require_sha256(
        args.expected_report_sha256, name="expected report sha256"
    )
    expected_report_count = _require_integer(
        args.expected_report_byte_count, name="expected report byte count", minimum=1
    )
    report_bytes = _pread_exact(report_fd, expected_report_count, name="held report")
    report_after = os.fstat(report_fd)
    if (
        report_after.st_dev,
        report_after.st_ino,
        report_after.st_mode,
        report_after.st_nlink,
        report_after.st_size,
        report_after.st_mtime_ns,
        report_after.st_ctime_ns,
    ) != (
        report_stat.st_dev,
        report_stat.st_ino,
        report_stat.st_mode,
        report_stat.st_nlink,
        report_stat.st_size,
        report_stat.st_mtime_ns,
        report_stat.st_ctime_ns,
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "held report metadata changed during verifier pread"
        )
    if _sha256_bytes(report_bytes) != expected_report_sha:
        raise NestedOpeningFeasibilityExecutionV1Error(
            "held report bytes differ from exact pin"
        )
    try:
        report_text = report_bytes.decode("ascii")
    except UnicodeDecodeError as exc:
        raise NestedOpeningFeasibilityExecutionV1Error(
            "held report must be ASCII"
        ) from exc
    report, report_digest, replayed_bytes = _build_or_parse_report(
        for_testing=for_testing,
        upstream_text=upstream_text,
        upstream_digest=upstream_digest,
        upstream_sha256=_sha256_bytes(upstream_bytes),
        nested_text=nested_text,
        nested_digest=nested_digest,
        report_text=report_text,
        report_digest=cast(str, args.expected_report_digest),
    )
    if replayed_bytes != report_bytes:
        raise NestedOpeningFeasibilityExecutionV1Error(
            "fresh verifier report bytes differ"
        )
    terminal = _worker_terminal(
        kind=_VERIFIER_TERMINAL_KIND,
        started_monotonic_ns=started,
        deadline_monotonic_ns=deadline,
        upstream_digest=upstream_digest,
        upstream_sha256=_sha256_bytes(upstream_bytes),
        upstream_byte_count=len(upstream_bytes),
        nested_digest=nested_digest,
        nested_sha256=_sha256_bytes(nested_bytes),
        nested_byte_count=len(nested_bytes),
        report=report,
        report_digest=report_digest,
        report_bytes=report_bytes,
        frozen_byte_revalidation={
            "held_input_bindings": {
                label: {
                    "sha256": _sha256_bytes(payload),
                    "byte_count": len(payload),
                }
                for label, payload in held.items()
            },
            "all_held_data_inputs_read_by_pread": True,
            "no_held_data_input_path_reopened": True,
            "canonical_code_paths_reopened_for_hash_revalidation": True,
            "report_read_by_pread": True,
        },
        started_wall_utc=started_wall,
        observed_environment=observed_environment,
    )
    sys.stdout.buffer.write(_canonical_bytes(terminal))
    sys.stdout.buffer.flush()
    return 0


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
        descriptors: list[int] = []
        try:
            for _ in range(5):
                read_fd, write_fd = os.pipe()
                descriptors.extend((read_fd, write_fd))
            return cls(*descriptors)
        except BaseException:
            for descriptor in descriptors:
                with suppress(OSError):
                    os.close(descriptor)
            raise

    def close(self) -> None:
        for field in self.__dataclass_fields__:
            descriptor = getattr(self, field)
            if descriptor >= 0:
                with suppress(OSError):
                    os.close(descriptor)
                setattr(self, field, -1)


def _terminate_owned_process_group(process: subprocess.Popen[bytes]) -> None:
    """Signal only a still-live process group owned by this Popen object."""

    if process.poll() is not None:
        return
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass
    if process.poll() is not None:
        return
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=5)


def _validate_watchdog_worker_launch_argv(
    worker_argv: Sequence[str],
    *,
    worker_guard_read_fd: int,
    ready_write_fd: int,
    lifetime_write_fd: int,
    forward_fds: Sequence[int],
) -> None:
    prefix = tuple(
        _safe_bootstrap_launch_prefix(role="--g03-bootstrap-worker")
    )
    if tuple(worker_argv[:7]) != prefix or len(worker_argv) < 8:
        raise NestedOpeningFeasibilityExecutionV1Error(
            "watchdog worker argv lacks the exact safe bootstrap prefix"
        )
    action = worker_argv[7]
    expected_options = _worker_launch_options(action)
    tail = worker_argv[8:]
    if len(tail) != 2 * len(expected_options) or tuple(tail[::2]) != expected_options:
        raise NestedOpeningFeasibilityExecutionV1Error(
            "watchdog worker argv does not match the exact ordered action grammar"
        )
    values = dict(zip(expected_options, tail[1::2], strict=True))
    for option, expected in (
        ("--parent-guard-fd", worker_guard_read_fd),
        ("--guardian-ready-write-fd", ready_write_fd),
        ("--worker-lifetime-write-fd", lifetime_write_fd),
    ):
        if values[option] != str(expected):
            raise NestedOpeningFeasibilityExecutionV1Error(
                f"watchdog worker argv changes {option}"
            )
    held_map = _load_json(values["--held-input-fds-json"])
    if (
        type(held_map) is not dict
        or _dump_json(held_map) != values["--held-input-fds-json"]
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in cast(dict[str, object], held_map).values()
        )
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "watchdog worker held-FD map is not canonical"
        )
    expected_forward = list(cast(dict[str, int], held_map).values())
    semantic_option = "--output-root-fd" if action == "__build-worker" else "--report-fd"
    semantic_value = values[semantic_option]
    if not semantic_value.isdigit():
        raise NestedOpeningFeasibilityExecutionV1Error(
            "watchdog worker semantic output FD is not canonical"
        )
    expected_forward.append(int(semantic_value))
    if list(forward_fds) != expected_forward:
        raise NestedOpeningFeasibilityExecutionV1Error(
            "watchdog forward-FD list differs from exact worker semantic capabilities"
        )


def _acquire_watchdog_worker_inputs(
    args: argparse.Namespace,
) -> tuple[int, int, int, int, int, int, int, int, int, list[str], list[int]]:
    names = (
        "parent_death_fd",
        "worker_guard_read_fd",
        "worker_guard_write_fd",
        "worker_pid_report_fd",
        "guardian_ready_read_fd",
        "guardian_ready_write_fd",
        "worker_lifetime_read_fd",
        "worker_lifetime_write_fd",
    )
    descriptors = [
        _require_integer(getattr(args, name), name=name.replace("_", " "), minimum=0)
        for name in names
    ]
    forward_fds: list[int] = []
    try:
        deadline_ns = _require_integer(
            args.deadline_monotonic_ns, name="watchdog deadline", minimum=1
        )
        worker_argv_value = _load_json(cast(str, args.worker_argv_json))
        forward_value = _load_json(cast(str, args.forward_fds_json))
        if (
            type(worker_argv_value) is not list
            or not worker_argv_value
            or any(
                type(item) is not str or "\0" in item for item in worker_argv_value
            )
            or _dump_json(worker_argv_value) != cast(str, args.worker_argv_json)
        ):
            raise NestedOpeningFeasibilityExecutionV1Error(
                "watchdog worker argv is not canonical"
            )
        if (
            type(forward_value) is not list
            or not forward_value
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in forward_value
            )
            or len(set(forward_value)) != len(forward_value)
            or _dump_json(forward_value) != cast(str, args.forward_fds_json)
        ):
            raise NestedOpeningFeasibilityExecutionV1Error(
                "watchdog forward-FD list is not canonical and unique"
            )
        worker_argv = cast(list[str], worker_argv_value)
        forward_fds = cast(list[int], forward_value)
        _validate_watchdog_worker_launch_argv(
            worker_argv,
            worker_guard_read_fd=descriptors[1],
            ready_write_fd=descriptors[5],
            lifetime_write_fd=descriptors[7],
            forward_fds=forward_fds,
        )
        all_inherited = (*descriptors, *forward_fds)
        if len(set(all_inherited)) != len(all_inherited):
            raise NestedOpeningFeasibilityExecutionV1Error(
                "watchdog inherited capabilities must be pairwise distinct"
            )
        for descriptor in all_inherited:
            os.set_inheritable(descriptor, False)
        return cast(
            tuple[
                int,
                int,
                int,
                int,
                int,
                int,
                int,
                int,
                int,
                list[str],
                list[int],
            ],
            (*descriptors, deadline_ns, worker_argv, forward_fds),
        )
    except BaseException:
        for descriptor in (*descriptors, *forward_fds):
            with suppress(OSError):
                os.close(descriptor)
        raise


def _unreaped_child_observer_available() -> bool:
    if hasattr(os, "waitid"):
        return True
    if sys.platform != "darwin":
        return False
    try:
        waitid = ctypes.CDLL(None, use_errno=True).waitid
    except (AttributeError, OSError):
        return False
    return waitid is not None


def _child_exited_without_reap(pid: int) -> bool:
    options = os.WEXITED | os.WNOHANG | os.WNOWAIT
    if hasattr(os, "waitid"):
        result = os.waitid(os.P_PID, pid, options)
        return result is not None
    if sys.platform != "darwin":
        raise NestedOpeningFeasibilityExecutionV1Error(
            "no unreaped-child observer is available on this platform"
        )
    # CPython on Darwin exposes the waitid constants but not os.waitid.  Call
    # the same POSIX primitive through libc, using a generously sized zeroed
    # siginfo buffer.  WNOHANG leaves the zero buffer unchanged when the child
    # is not waitable; WNOWAIT keeps an exited leader unreaped so its PID/PGID
    # cannot be reused before same-group cleanup signals are sent.
    libc = ctypes.CDLL(None, use_errno=True)
    waitid = libc.waitid
    waitid.argtypes = (
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.c_int,
    )
    waitid.restype = ctypes.c_int
    class DarwinSiginfo(ctypes.Structure):
        _fields_ = (
            ("si_signo", ctypes.c_int),
            ("si_errno", ctypes.c_int),
            ("si_code", ctypes.c_int),
            ("si_pid", ctypes.c_int),
            ("si_uid", ctypes.c_uint),
            ("si_status", ctypes.c_int),
            ("si_addr", ctypes.c_void_p),
            ("si_value", ctypes.c_void_p),
            ("si_band", ctypes.c_long),
            ("reserved", ctypes.c_ulong * 7),
        )

    information = DarwinSiginfo()
    ctypes.set_errno(0)
    result = waitid(
        int(os.P_PID),
        pid,
        ctypes.byref(information),
        options,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.ECHILD:
            raise NestedOpeningFeasibilityExecutionV1Error(
                "watchdog lost ownership of its direct child before reap"
            )
        raise OSError(error_number, os.strerror(error_number))
    return information.si_pid != 0


def _cleanup_worker_group_before_direct_reap(
    worker: subprocess.Popen[bytes],
) -> tuple[bytes, bytes]:
    def attempt(signum: signal.Signals) -> None:
        try:
            os.killpg(worker.pid, signum)
        except ProcessLookupError:
            pass
        except PermissionError:
            if sys.platform != "darwin":
                raise
            # Darwin reports EPERM when the only same-group member is the
            # WNOWAIT zombie session leader.  There is no signalable member in
            # that case; retaining the zombie still prevents PGID reuse until
            # communicate() performs the owned direct-child reap below.

    attempt(signal.SIGTERM)
    select.select([], [], [], 0.05)
    attempt(signal.SIGKILL)
    return worker.communicate(timeout=5)


def _watchdog_worker(args: argparse.Namespace) -> int:
    """Own, bound, terminate, and reap one worker with an exact FD closure."""

    if os.environ.get(_BOOTSTRAP_MARKER) != "--g03-bootstrap-watchdog":
        raise NestedOpeningFeasibilityExecutionV1Error(
            "watchdog was not launched through the bound early-guardian bootstrap"
        )
    _require_default_sigchld()
    started_ns = time.monotonic_ns()
    started_wall = _utc_now()
    observed_environment = _observed_safe_environment(
        bootstrap_role="--g03-bootstrap-watchdog"
    )
    (
        parent_fd,
        worker_guard_read_fd,
        worker_guard_write_fd,
        pid_report_fd,
        ready_read_fd,
        ready_write_fd,
        lifetime_read_fd,
        lifetime_write_fd,
        deadline_ns,
        worker_argv,
        forward_fds,
    ) = _acquire_watchdog_worker_inputs(args)
    worker: subprocess.Popen[bytes] | None = None
    worker_reaped = False
    try:
        worker = subprocess.Popen(
            worker_argv,
            cwd=_repository_root(),
            env=_safe_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            pass_fds=(
                worker_guard_read_fd,
                ready_write_fd,
                lifetime_write_fd,
                *forward_fds,
            ),
        )
        for descriptor in (worker_guard_read_fd, ready_write_fd, lifetime_write_fd, *forward_fds):
            os.close(descriptor)
        worker_guard_read_fd = ready_write_fd = lifetime_write_fd = -1
        forward_fds = []
        guardian_ready = False
        while not guardian_ready:
            remaining_ns = deadline_ns - time.monotonic_ns()
            if remaining_ns <= 0:
                break
            readable, _, _ = select.select(
                [parent_fd, ready_read_fd],
                [],
                [],
                min(0.1, remaining_ns / 1_000_000_000),
            )
            if parent_fd in readable and os.read(parent_fd, 1) == b"":
                break
            if ready_read_fd in readable:
                guardian_ready = os.read(ready_read_fd, 2) == b"R"
                break
            if _child_exited_without_reap(worker.pid):
                break
        os.close(ready_read_fd)
        ready_read_fd = -1
        if not guardian_ready:
            _cleanup_worker_group_before_direct_reap(worker)
            worker_reaped = True
            raise NestedOpeningFeasibilityExecutionV1Error(
                "worker guardian did not acknowledge readiness"
            )
        pid_payload = f"{worker.pid}\n".encode("ascii")
        if os.write(pid_report_fd, pid_payload) != len(pid_payload):
            _cleanup_worker_group_before_direct_reap(worker)
            worker_reaped = True
            raise NestedOpeningFeasibilityExecutionV1Error(
                "short watchdog worker-PID report"
            )
        os.close(pid_report_fd)
        pid_report_fd = -1
        trigger = "worker_exited"
        term_signal_attempted = False
        kill_signal_attempted = False
        stdout = b""
        stderr = b""
        while True:
            readable, _, _ = select.select([parent_fd], [], [], 0)
            if readable and os.read(parent_fd, 1) == b"":
                trigger = "controller_pipe_eof"
                break
            remaining_ns = deadline_ns - time.monotonic_ns()
            if remaining_ns <= 0:
                trigger = "registered_deadline"
                break
            if _child_exited_without_reap(worker.pid):
                break
            select.select([], [], [], min(0.05, remaining_ns / 1_000_000_000))
        # The direct worker remains unreaped here, including on ordinary exit.
        # Its zombie PID therefore still owns the process-group identity while
        # the watchdog terminates any same-group fork/exec descendants.  Only
        # after both signals does communicate() reap the owned direct child.
        term_signal_attempted = True
        kill_signal_attempted = True
        stdout, stderr = _cleanup_worker_group_before_direct_reap(worker)
        worker_reaped = True
        lifetime_ready, _, _ = select.select([lifetime_read_fd], [], [], 1)
        lifetime_eof = bool(lifetime_ready) and os.read(lifetime_read_fd, 1) == b""
        worker_terminal: Any = None
        if worker.returncode == 0 and not stderr:
            try:
                worker_terminal = _load_json(stdout.decode("ascii"))
            except (UnicodeDecodeError, NestedOpeningFeasibilityExecutionV1Error):
                worker_terminal = None
        if dict(observed_environment) != _observed_safe_environment(
            bootstrap_role="--g03-bootstrap-watchdog"
        ):
            raise NestedOpeningFeasibilityExecutionV1Error(
                "watchdog environment changed before terminal admission"
            )
        admitted_ns = time.monotonic_ns()
        admitted_wall = _utc_now()
        success = (
            trigger == "worker_exited"
            and worker.returncode == 0
            and not stderr
            and type(worker_terminal) is dict
            and admitted_ns <= deadline_ns
            and lifetime_eof
        )
        unsigned = {
            "schema_version": NESTED_OPENING_FEASIBILITY_EXECUTION_SCHEMA_VERSION,
            "terminal_kind": _WATCHDOG_TERMINAL_KIND,
            "status": (
                "success"
                if success
                else "failed_direct_worker_reaped_same_group_cleanup_attempted"
            ),
            "watchdog_process_identity": {
                "pid": os.getpid(),
                "argv": list(sys.argv),
                "working_directory": str(Path.cwd()),
                "host": _host_identity(),
                "python": _python_identity(),
                "environment": dict(observed_environment),
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
                "term_signal_attempted_before_direct_worker_reap": (
                    term_signal_attempted
                ),
                "kill_signal_attempted_before_direct_worker_reap": (
                    kill_signal_attempted
                ),
                "guardian_ready_before_pid_report": guardian_ready,
                "worker_lifetime_pipe_eof_observed": lifetime_eof,
                "worker_reaped_by_owning_watchdog": worker.poll() is not None,
                "watchdog_outside_worker_process_group": os.getpgrp() != worker.pid,
                "forwarded_data_fd_copies_closed_immediately_after_spawn": True,
                "direct_worker_observed_before_reap": True,
                "same_group_cleanup_signals_attempted_before_direct_worker_reap": True,
                "detached_descendant_absence_os_proven": False,
                "post_reap_process_group_signal_attempted": False,
            },
            "deadline_monotonic_ns": deadline_ns,
            "timing": {
                "started_monotonic_ns": started_ns,
                "terminal_admitted_monotonic_ns": admitted_ns,
                "elapsed_to_terminal_admission_ns": admitted_ns - started_ns,
                "started_wall_utc": started_wall,
                "terminal_admitted_wall_utc": admitted_wall,
            },
            "worker_terminal": worker_terminal,
            "execution_scope": dict(_ENGINEERING_EXECUTION_SCOPE),
            "authorization": dict(_AUTHORIZATION),
        }
        terminal = {
            **unsigned,
            "watchdog_terminal_digest": _digest(
                unsigned, domain=_WATCHDOG_TERMINAL_DOMAIN
            ),
        }
        sys.stdout.buffer.write(_canonical_bytes(terminal))
        sys.stdout.buffer.flush()
        return 0 if success else 2
    except BaseException:
        if worker is not None and not worker_reaped:
            with suppress(BaseException):
                _cleanup_worker_group_before_direct_reap(worker)
                worker_reaped = True
        raise
    finally:
        for descriptor in (
            parent_fd,
            worker_guard_read_fd,
            worker_guard_write_fd,
            pid_report_fd,
            ready_read_fd,
            ready_write_fd,
            lifetime_read_fd,
            lifetime_write_fd,
            *forward_fds,
        ):
            if descriptor >= 0:
                with suppress(OSError):
                    os.close(descriptor)


def _worker_parser() -> argparse.ArgumentParser:
    def add_common(worker: argparse.ArgumentParser) -> None:
        worker.set_defaults(require_production=False)
        worker.add_argument("--held-input-fds-json", required=True)
        worker.add_argument("--held-input-bindings-json", required=True)
        worker.add_argument("--expected-upstream-plan-digest", required=True)
        worker.add_argument("--expected-upstream-plan-sha256", required=True)
        worker.add_argument("--expected-nested-plan-digest", required=True)
        worker.add_argument("--expected-nested-plan-sha256", required=True)
        worker.add_argument("--expected-freeze-request-digest", required=True)
        worker.add_argument("--expected-registration-receipt-sha256", required=True)
        worker.add_argument("--expected-registration-reference", required=True)
        worker.add_argument("--expected-execution-uuid", required=True)
        worker.add_argument("--expected-execution-nonce", required=True)
        worker.add_argument("--expected-execution-deadline-seconds", required=True, type=int)
        worker.add_argument("--controller-started-wall-utc", required=True)
        worker.add_argument("--deadline-monotonic-ns", required=True, type=int)
        worker.add_argument("--parent-guard-fd", required=True, type=int)
        worker.add_argument("--guardian-ready-write-fd", required=True, type=int)
        worker.add_argument("--worker-lifetime-write-fd", required=True, type=int)
        worker.add_argument("--runner-source", required=True)

    parser = argparse.ArgumentParser(add_help=False)
    subparsers = parser.add_subparsers(dest="worker_action", required=True)
    build = subparsers.add_parser("__build-worker", add_help=False)
    add_common(build)
    build.add_argument("--output-root-fd", required=True, type=int)
    verify = subparsers.add_parser("__verify-worker", add_help=False)
    add_common(verify)
    verify.add_argument("--report-fd", required=True, type=int)
    verify.add_argument("--expected-report-digest", required=True)
    verify.add_argument("--expected-report-sha256", required=True)
    verify.add_argument("--expected-report-byte-count", required=True, type=int)
    watchdog = subparsers.add_parser("__watchdog-worker", add_help=False)
    watchdog.add_argument("--parent-death-fd", required=True, type=int)
    watchdog.add_argument("--worker-guard-read-fd", required=True, type=int)
    watchdog.add_argument("--worker-guard-write-fd", required=True, type=int)
    watchdog.add_argument("--worker-pid-report-fd", required=True, type=int)
    watchdog.add_argument("--guardian-ready-read-fd", required=True, type=int)
    watchdog.add_argument("--guardian-ready-write-fd", required=True, type=int)
    watchdog.add_argument("--worker-lifetime-read-fd", required=True, type=int)
    watchdog.add_argument("--worker-lifetime-write-fd", required=True, type=int)
    watchdog.add_argument("--deadline-monotonic-ns", required=True, type=int)
    watchdog.add_argument("--forward-fds-json", required=True)
    watchdog.add_argument("--worker-argv-json", required=True)
    return parser


def _require_worker_lifetime_eof(channels: _ParentDeathPipe, *, timeout: float) -> None:
    descriptor = channels.worker_lifetime_read_fd
    if descriptor < 0:
        raise NestedOpeningFeasibilityExecutionV1Error(
            "worker lifetime capability is unavailable"
        )
    readable, _, _ = select.select([descriptor], [], [], timeout)
    if not readable or os.read(descriptor, 1) != b"":
        raise NestedOpeningFeasibilityExecutionV1Error(
            "worker lifetime capability did not reach exact EOF"
        )
    os.close(descriptor)
    channels.worker_lifetime_read_fd = -1


def _run_watchdog(
    watchdog_argv: list[str],
    *,
    deadline_monotonic_ns: int,
    channels: _ParentDeathPipe,
    forward_fds: Sequence[int],
) -> tuple[Mapping[str, Any], int, int]:
    process: subprocess.Popen[bytes] | None = None
    owned_forward_fds = set(forward_fds)
    try:
        remaining_ns = deadline_monotonic_ns - time.monotonic_ns()
        if remaining_ns <= 0:
            raise NestedOpeningFeasibilityExecutionV1Error(
                "execution deadline expired before worker launch"
            )
        pass_fds = (
            channels.read_fd,
            channels.worker_read_fd,
            channels.worker_write_fd,
            channels.pid_write_fd,
            channels.guardian_ready_read_fd,
            channels.guardian_ready_write_fd,
            channels.worker_lifetime_read_fd,
            channels.worker_lifetime_write_fd,
            *forward_fds,
        )
        if len(set(pass_fds)) != len(pass_fds):
            raise NestedOpeningFeasibilityExecutionV1Error(
                "controller watchdog capabilities are not distinct"
            )
        # This is the final controller-side boundary before Popen.  Repeat the
        # ownership prerequisites here, after every channel/FD/argv setup
        # operation, so no setup hook can install an auto-reaper or a
        # concurrent host thread between the public preflight and spawn.
        _require_single_threaded_controller()
        _require_default_sigchld()
        process = subprocess.Popen(
            watchdog_argv,
            cwd=_repository_root(),
            env=_safe_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            pass_fds=pass_fds,
        )
        for descriptor in forward_fds:
            os.close(descriptor)
            owned_forward_fds.discard(descriptor)
        for field in (
            "read_fd",
            "worker_read_fd",
            "worker_write_fd",
            "pid_write_fd",
            "guardian_ready_read_fd",
            "guardian_ready_write_fd",
            "worker_lifetime_write_fd",
        ):
            os.close(getattr(channels, field))
            setattr(channels, field, -1)
        wait_seconds = min(30.0, max(0.001, remaining_ns / 1_000_000_000))
        readable, _, _ = select.select([channels.pid_read_fd], [], [], wait_seconds)
        if not readable:
            raise NestedOpeningFeasibilityExecutionV1Error(
                "watchdog did not report its worker PID"
            )
        pid_bytes = os.read(channels.pid_read_fd, 32)
        os.close(channels.pid_read_fd)
        channels.pid_read_fd = -1
        if not pid_bytes.endswith(b"\n") or not pid_bytes[:-1].isdigit():
            raise NestedOpeningFeasibilityExecutionV1Error(
                "watchdog worker-PID report is malformed"
            )
        reported_pid = int(pid_bytes[:-1])
        try:
            stdout, stderr = process.communicate(
                timeout=remaining_ns / 1_000_000_000 + 7
            )
        except subprocess.TimeoutExpired as exc:
            with suppress(OSError):
                os.close(channels.write_fd)
            channels.write_fd = -1
            _terminate_owned_process_group(process)
            raise NestedOpeningFeasibilityExecutionV1Error(
                "watchdog exceeded deadline containment grace"
            ) from exc
        if process.returncode != 0 or stderr:
            raise NestedOpeningFeasibilityExecutionV1Error(
                "watchdog failed; "
                f"stdout_sha256={_sha256_bytes(stdout)}; "
                f"stderr_sha256={_sha256_bytes(stderr)}"
            )
        _require_worker_lifetime_eof(channels, timeout=7)
        try:
            terminal = _load_json(stdout.decode("ascii"))
        except UnicodeDecodeError as exc:
            raise NestedOpeningFeasibilityExecutionV1Error(
                "watchdog terminal is not ASCII"
            ) from exc
        if type(terminal) is not dict:
            raise NestedOpeningFeasibilityExecutionV1Error(
                "watchdog terminal is not an object"
            )
        if _canonical_bytes(terminal) != stdout:
            raise NestedOpeningFeasibilityExecutionV1Error(
                "watchdog terminal stdout is not canonical newline-terminated JSON"
            )
        unsigned = {
            key: value
            for key, value in cast(dict[str, Any], terminal).items()
            if key != "watchdog_terminal_digest"
        }
        if (
            terminal.get("schema_version")
            != NESTED_OPENING_FEASIBILITY_EXECUTION_SCHEMA_VERSION
            or terminal.get("terminal_kind") != _WATCHDOG_TERMINAL_KIND
            or terminal.get("status") != "success"
            or terminal.get("deadline_monotonic_ns") != deadline_monotonic_ns
            or terminal.get("authorization") != _AUTHORIZATION
            or terminal.get("watchdog_terminal_digest")
            != _digest(unsigned, domain=_WATCHDOG_TERMINAL_DOMAIN)
        ):
            raise NestedOpeningFeasibilityExecutionV1Error(
                "watchdog terminal failed exact validation"
            )
        worker = cast(dict[str, Any], terminal["worker_process"])
        if (
            _require_integer(worker.get("pid"), name="reported worker pid", minimum=1)
            != reported_pid
            or worker.get("reaped") is not True
        ):
            raise NestedOpeningFeasibilityExecutionV1Error(
                "watchdog terminal worker identity differs from early report"
            )
        return cast(Mapping[str, Any], terminal), process.pid, reported_pid
    except BaseException:
        if process is not None:
            with suppress(OSError):
                os.close(channels.write_fd)
            channels.write_fd = -1
            _terminate_owned_process_group(process)
        if channels.worker_lifetime_read_fd >= 0:
            with suppress(NestedOpeningFeasibilityExecutionV1Error):
                _require_worker_lifetime_eof(channels, timeout=7)
        raise
    finally:
        for descriptor in owned_forward_fds:
            with suppress(OSError):
                os.close(descriptor)
        channels.close()


def _validate_timing(value: object, *, deadline_monotonic_ns: int) -> None:
    timing = _require_mapping(
        value,
        (
            "started_monotonic_ns",
            "terminal_admitted_monotonic_ns",
            "elapsed_to_terminal_admission_ns",
            "started_wall_utc",
            "terminal_admitted_wall_utc",
        ),
        name="process timing",
    )
    started = _require_integer(
        timing["started_monotonic_ns"], name="started monotonic ns", minimum=1
    )
    admitted = _require_integer(
        timing["terminal_admitted_monotonic_ns"],
        name="terminal-admitted monotonic ns",
        minimum=started,
    )
    if (
        _require_integer(
            timing["elapsed_to_terminal_admission_ns"],
            name="elapsed to terminal admission ns",
        )
        != admitted - started
        or admitted > deadline_monotonic_ns
        or _wall_datetime(
            timing["terminal_admitted_wall_utc"],
            name="terminal-admitted wall UTC",
        )
        < _wall_datetime(timing["started_wall_utc"], name="started wall UTC")
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "process timing is inconsistent or exceeds deadline"
        )


def _validate_rss(value: object) -> None:
    rss = _require_mapping(
        value,
        ("measurement_api", "raw_value", "raw_unit", "normalized_bytes"),
        name="maximum resident set size",
    )
    raw = _require_integer(rss["raw_value"], name="RSS raw value")
    normalized = _require_integer(rss["normalized_bytes"], name="RSS normalized bytes")
    if rss["measurement_api"] != "resource.getrusage(RUSAGE_SELF).ru_maxrss":
        raise NestedOpeningFeasibilityExecutionV1Error("unknown RSS measurement API")
    if rss["raw_unit"] == "bytes":
        expected = raw
    elif rss["raw_unit"] == "kibibytes":
        expected = raw * 1024
    else:
        raise NestedOpeningFeasibilityExecutionV1Error("unknown RSS raw unit")
    if normalized != expected:
        raise NestedOpeningFeasibilityExecutionV1Error("RSS normalization is inconsistent")


def _parse_worker_terminal(
    value: object,
    *,
    expected_kind: str,
    expected_pid: int,
    expected_argv: Sequence[str],
    expected_deadline_monotonic_ns: int,
    expected_plan_bindings: Mapping[str, Any],
    expected_held_bindings: Mapping[str, Any],
    require_production: bool,
) -> Mapping[str, Any]:
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
            "input_plan_bindings",
            "frozen_byte_revalidation",
            "report_binding",
            "accounting",
            "execution_scope",
            "authorization",
            "worker_terminal_digest",
        ),
        name="worker terminal",
    )
    if (
        _require_integer(obj["schema_version"], name="worker schema version")
        != NESTED_OPENING_FEASIBILITY_EXECUTION_SCHEMA_VERSION
        or obj["terminal_kind"] != expected_kind
        or obj["status"] != "success"
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "worker terminal schema, kind, or status changed"
        )
    process = _require_mapping(
        obj["process_identity"],
        ("pid", "argv", "working_directory", "host", "python", "environment"),
        name="worker process identity",
    )
    expected_process_argv = [str(Path(__file__).resolve()), *expected_argv[7:]]
    if (
        _require_integer(process["pid"], name="worker pid", minimum=1)
        != expected_pid
        or process["argv"] != expected_process_argv
        or process["working_directory"] != str(_repository_root())
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "worker process identity differs from launch"
        )
    _require_exact_constant(process["host"], _host_identity(), name="worker host")
    _require_exact_constant(process["python"], _python_identity(), name="worker Python")
    _require_exact_constant(
        process["environment"],
        _safe_environment(bootstrap_role="--g03-bootstrap-worker"),
        name="worker environment",
    )
    if (
        _require_integer(
            obj["deadline_monotonic_ns"], name="worker deadline", minimum=1
        )
        != expected_deadline_monotonic_ns
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "worker deadline differs from controller"
        )
    _validate_timing(obj["timing"], deadline_monotonic_ns=expected_deadline_monotonic_ns)
    _validate_rss(obj["maximum_resident_set_size"])
    if not _exact_json_equal(obj["input_plan_bindings"], expected_plan_bindings):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "worker input-plan bindings differ from controller"
        )
    revalidation = _require_mapping(
        obj["frozen_byte_revalidation"],
        (
            "held_input_bindings",
            "all_held_data_inputs_read_by_pread",
            "no_held_data_input_path_reopened",
            "canonical_code_paths_reopened_for_hash_revalidation",
        )
        + (("report_read_by_pread",) if expected_kind == _VERIFIER_TERMINAL_KIND else ()),
        name="worker frozen-byte revalidation",
    )
    expected_worker_held = {
        label: {"sha256": binding["sha256"], "byte_count": binding["byte_count"]}
        for label, binding in expected_held_bindings.items()
    }
    if not _exact_json_equal(
        revalidation["held_input_bindings"], expected_worker_held
    ) or any(
        revalidation[key] is not True
        for key in revalidation
        if key != "held_input_bindings"
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "worker did not revalidate exact held data bytes"
        )
    report_binding = _require_mapping(
        obj["report_binding"],
        (
            "observed_report_digest",
            "exact_report_bytes_sha256",
            "exact_report_byte_count",
        ),
        name="worker report binding",
    )
    _require_sha256(report_binding["observed_report_digest"], name="report digest")
    _require_sha256(report_binding["exact_report_bytes_sha256"], name="report sha256")
    _require_integer(report_binding["exact_report_byte_count"], name="report byte count", minimum=1)
    accounting = cast(Mapping[str, Any], obj["accounting"])
    if type(accounting) is not dict or tuple(accounting) != (
        "mirror_attempt_count",
        "member_record_count",
        "distinct_evaluation_c_identity_count",
        "planned_opening_assessment_count",
        "completed_opening_assessment_count",
        "common_union_completed_count",
        "bounded_slot_failure_count",
        "member_exact_match_witness_count",
        "nested_opening_feasibility_witness_count",
        "mirror_pair_disjoint_count",
        "mirror_pair_full_common_key_match_count",
        "mirror_pair_feasibility_witness_count",
        "protocol_quartet_candidate_count",
        "observed_cell_count",
        "observed_cell_coordinates_digest",
        "all_planned_attempts_preserved",
        "all_36_cells_preserved",
        "early_stop_used",
        "identity_replacement_used",
        "exact_144x2_budget_complete",
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "worker accounting fields are noncanonical"
        )
    for key, item in accounting.items():
        if key.endswith("_count"):
            _require_integer(item, name=key)
    for key in (
        "all_planned_attempts_preserved",
        "all_36_cells_preserved",
        "early_stop_used",
        "identity_replacement_used",
        "exact_144x2_budget_complete",
    ):
        _require_boolean(accounting[key], name=key)
    expected_coordinates_digest = _digest(
        [[m, q] for m, q in _EXPECTED_CELL_COORDINATES],
        domain=_CELL_COORDINATE_DOMAIN,
    )
    if _require_sha256(
        accounting["observed_cell_coordinates_digest"],
        name="observed cell coordinates digest",
    ) != expected_coordinates_digest:
        raise NestedOpeningFeasibilityExecutionV1Error(
            "worker observed-cell coordinate digest is not the canonical 36-cell grid"
        )
    expected_budget = require_production
    if (
        accounting["protocol_quartet_candidate_count"] != 0
        or accounting["observed_cell_count"] != 36
        or accounting["all_planned_attempts_preserved"] is not True
        or accounting["all_36_cells_preserved"] is not True
        or accounting["early_stop_used"] is not False
        or accounting["identity_replacement_used"] is not False
        or accounting["exact_144x2_budget_complete"] is not expected_budget
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "worker accounting violates fixed nonauthorizing lifecycle"
        )
    if require_production and (
        accounting["mirror_attempt_count"] != 144
        or accounting["member_record_count"] != 288
        or accounting["distinct_evaluation_c_identity_count"] != 288
        or accounting["planned_opening_assessment_count"] != 1_152
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "production worker accounting differs from exact 144/288/1152 budget"
        )
    _require_exact_constant(obj["authorization"], _AUTHORIZATION, name="authorization")
    _require_exact_constant(
        obj["execution_scope"],
        _ENGINEERING_EXECUTION_SCOPE,
        name="worker execution scope",
    )
    unsigned = {key: item for key, item in obj.items() if key != "worker_terminal_digest"}
    if _require_sha256(
        obj["worker_terminal_digest"], name="worker terminal digest"
    ) != _digest(unsigned, domain=_WORKER_TERMINAL_DOMAIN):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "worker terminal digest is inconsistent"
        )
    return obj


def _parse_watchdog_and_worker_terminal(
    terminal: Mapping[str, Any],
    *,
    expected_watchdog_pid: int,
    expected_watchdog_argv: Sequence[str],
    expected_worker_pid: int,
    expected_worker_argv: Sequence[str],
    expected_worker_kind: str,
    expected_deadline_monotonic_ns: int,
    expected_plan_bindings: Mapping[str, Any],
    expected_held_bindings: Mapping[str, Any],
    require_production: bool,
) -> Mapping[str, Any]:
    obj = _require_mapping(
        terminal,
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
            "execution_scope",
            "authorization",
            "watchdog_terminal_digest",
        ),
        name="watchdog terminal",
    )
    if (
        _require_integer(obj["schema_version"], name="watchdog schema version")
        != NESTED_OPENING_FEASIBILITY_EXECUTION_SCHEMA_VERSION
        or obj["terminal_kind"] != _WATCHDOG_TERMINAL_KIND
        or obj["status"] != "success"
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "watchdog terminal schema, kind, or status changed"
        )
    process = _require_mapping(
        obj["watchdog_process_identity"],
        ("pid", "argv", "working_directory", "host", "python", "environment"),
        name="watchdog process identity",
    )
    expected_process_argv = [
        str(Path(__file__).resolve()), *expected_watchdog_argv[7:]
    ]
    if (
        _require_integer(process["pid"], name="watchdog pid", minimum=1)
        != expected_watchdog_pid
        or process["argv"] != expected_process_argv
        or process["working_directory"] != str(_repository_root())
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "watchdog process identity differs from launch"
        )
    _require_exact_constant(process["host"], _host_identity(), name="watchdog host")
    _require_exact_constant(process["python"], _python_identity(), name="watchdog Python")
    _require_exact_constant(
        process["environment"],
        _safe_environment(bootstrap_role="--g03-bootstrap-watchdog"),
        name="watchdog environment",
    )
    worker_process = _require_mapping(
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
    worker_terminal = _parse_worker_terminal(
        obj["worker_terminal"],
        expected_kind=expected_worker_kind,
        expected_pid=expected_worker_pid,
        expected_argv=expected_worker_argv,
        expected_deadline_monotonic_ns=expected_deadline_monotonic_ns,
        expected_plan_bindings=expected_plan_bindings,
        expected_held_bindings=expected_held_bindings,
        require_production=require_production,
    )
    worker_bytes = _canonical_bytes(worker_terminal)
    if (
        _require_integer(worker_process["pid"], name="worker pid", minimum=1)
        != expected_worker_pid
        or worker_process["argv"] != list(expected_worker_argv)
        or _require_integer(worker_process["returncode"], name="worker returncode") != 0
        or worker_process["stdout_sha256"] != _sha256_bytes(worker_bytes)
        or worker_process["stderr_sha256"] != _sha256_bytes(b"")
        or _require_integer(
            worker_process["stderr_byte_count"], name="worker stderr byte count"
        )
        != 0
        or worker_process["reaped"] is not True
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "watchdog worker binding differs from exact launch and stdout"
        )
    containment = _require_mapping(
        obj["containment"],
        (
            "trigger",
            "term_signal_attempted_before_direct_worker_reap",
            "kill_signal_attempted_before_direct_worker_reap",
            "guardian_ready_before_pid_report",
            "worker_lifetime_pipe_eof_observed",
            "worker_reaped_by_owning_watchdog",
            "watchdog_outside_worker_process_group",
            "forwarded_data_fd_copies_closed_immediately_after_spawn",
            "direct_worker_observed_before_reap",
            "same_group_cleanup_signals_attempted_before_direct_worker_reap",
            "detached_descendant_absence_os_proven",
            "post_reap_process_group_signal_attempted",
        ),
        name="watchdog containment",
    )
    if (
        containment["trigger"] != "worker_exited"
        or containment["term_signal_attempted_before_direct_worker_reap"] is not True
        or containment["kill_signal_attempted_before_direct_worker_reap"] is not True
        or containment["detached_descendant_absence_os_proven"] is not False
        or containment["post_reap_process_group_signal_attempted"] is not False
        or any(
            containment[key] is not True
            for key in (
                "guardian_ready_before_pid_report",
                "worker_lifetime_pipe_eof_observed",
                "worker_reaped_by_owning_watchdog",
                "watchdog_outside_worker_process_group",
                "forwarded_data_fd_copies_closed_immediately_after_spawn",
                "direct_worker_observed_before_reap",
                "same_group_cleanup_signals_attempted_before_direct_worker_reap",
            )
        )
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "watchdog containment invariants differ from success contract"
        )
    if (
        _require_integer(
            obj["deadline_monotonic_ns"], name="watchdog deadline", minimum=1
        )
        != expected_deadline_monotonic_ns
    ):
        raise NestedOpeningFeasibilityExecutionV1Error("watchdog deadline changed")
    _validate_timing(obj["timing"], deadline_monotonic_ns=expected_deadline_monotonic_ns)
    _require_exact_constant(obj["authorization"], _AUTHORIZATION, name="authorization")
    _require_exact_constant(
        obj["execution_scope"],
        _ENGINEERING_EXECUTION_SCOPE,
        name="watchdog execution scope",
    )
    unsigned = {key: item for key, item in obj.items() if key != "watchdog_terminal_digest"}
    if obj["watchdog_terminal_digest"] != _digest(
        unsigned, domain=_WATCHDOG_TERMINAL_DOMAIN
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "watchdog terminal digest is inconsistent"
        )
    return worker_terminal


def _managed_controller_signals() -> tuple[signal.Signals, ...]:
    managed: list[signal.Signals] = []
    for candidate in (
        signal.SIGTERM,
        signal.SIGINT,
        getattr(signal, "SIGHUP", None),
        getattr(signal, "SIGALRM", None),
    ):
        if candidate is not None and candidate not in managed:
            managed.append(candidate)
    return tuple(managed)


def _install_controller_signal_handlers() -> dict[int, Any]:
    previous: dict[int, Any] = {}

    def interrupted(signum: int, _frame: Any) -> NoReturn:
        raise NestedOpeningFeasibilityExecutionV1Error(
            f"controller interrupted by signal {signal.Signals(signum).name}"
        )

    try:
        for candidate in _managed_controller_signals():
            signum = int(candidate)
            previous[signum] = signal.getsignal(candidate)
            signal.signal(candidate, interrupted)
        return previous
    except BaseException:
        for signum, handler in reversed(tuple(previous.items())):
            with suppress(BaseException):
                signal.signal(signum, handler)
        raise


def _restore_controller_signal_handlers(previous: Mapping[int, Any]) -> None:
    first_error: BaseException | None = None
    for signum, handler in reversed(tuple(previous.items())):
        try:
            signal.signal(signum, handler)
        except BaseException as exc:
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error


def _block_terminal_signals_and_disarm_alarm(
    *, timer_armed: bool
) -> set[signal.Signals]:
    if not hasattr(signal, "pthread_sigmask"):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "terminal signal deferral requires pthread_sigmask"
        )
    managed = set(_managed_controller_signals())
    previous_mask = cast(
        set[signal.Signals],
        signal.pthread_sigmask(signal.SIG_BLOCK, managed),
    )
    try:
        alarm = getattr(signal, "SIGALRM", None)
        if timer_armed:
            signal.setitimer(signal.ITIMER_REAL, 0)
        if (
            alarm is not None
            and hasattr(signal, "sigpending")
            and hasattr(signal, "sigwait")
            and alarm in signal.sigpending()
        ):
            signal.sigwait({alarm})
        return previous_mask
    except BaseException:
        with suppress(BaseException):
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        raise


def _restore_controller_signal_mask(previous: set[signal.Signals] | None) -> None:
    if previous is not None:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def _require_distinct_held_inodes(
    files: Mapping[str, _HeldOrdinaryFileV1],
) -> None:
    seen: dict[tuple[int, int], str] = {}
    for label, held in files.items():
        inode = (held.file_dev, held.file_ino)
        if inode in seen:
            raise NestedOpeningFeasibilityExecutionV1Error(
                f"held inputs {seen[inode]} and {label} alias one inode"
            )
        seen[inode] = label


def _open_execution_inputs(
    *,
    upstream_census_plan_path: Path,
    nested_plan_path: Path,
    freeze_request_path: Path,
    registration_receipt_path: Path,
    source_archive_path: Path | None,
    constraints_path: Path | None,
    environment_lock_path: Path | None,
    require_production: bool,
) -> dict[str, _HeldOrdinaryFileV1]:
    specs: list[tuple[str, Path]] = [
        ("upstream_census_plan", upstream_census_plan_path),
        ("nested_plan", nested_plan_path),
        ("freeze_request", freeze_request_path),
        ("registration_receipt", registration_receipt_path),
    ]
    if require_production:
        if source_archive_path is None or constraints_path is None or environment_lock_path is None:
            raise NestedOpeningFeasibilityExecutionV1Error(
                "production execution requires three sealed environment inputs"
            )
        specs.extend(
            (
                ("source_archive", source_archive_path),
                ("constraints", constraints_path),
                ("environment_lock", environment_lock_path),
            )
        )
    files: dict[str, _HeldOrdinaryFileV1] = {}
    try:
        for label, path in specs:
            files[label] = _HeldOrdinaryFileV1.open(
                path,
                label=label,
                maximum_bytes=_HELD_INPUT_MAXIMUM_BYTES[label],
                expected_modes=frozenset({0o400}),
            )
        _require_distinct_held_inodes(files)
        return files
    except BaseException:
        _close_held_files(files)
        raise


def _controller_replay_held_chain(
    files: Mapping[str, _HeldOrdinaryFileV1],
    *,
    expected_upstream_plan_digest: str,
    expected_upstream_plan_sha256: str,
    expected_nested_plan_digest: str,
    expected_nested_plan_sha256: str,
    expected_registration_receipt_sha256: str,
    expected_registration_reference: str,
    expected_execution_uuid: str,
    expected_execution_nonce: str,
    expected_execution_deadline_seconds: int,
    runner_source_path: Path,
    controller_started_wall_utc: str,
    require_production: bool,
) -> tuple[
    EvaluationCensusPlanV1,
    NestedOpeningFeasibilityPlanV1,
    NestedOpeningFeasibilityFreezeRequestV1,
    ExternalNestedOpeningFeasibilityRegistrationReceiptV1,
    dict[str, bytes],
]:
    payloads = {label: held.read_bytes() for label, held in files.items()}
    upstream_digest = _require_sha256(
        expected_upstream_plan_digest, name="expected upstream plan digest"
    )
    upstream_sha = _require_sha256(
        expected_upstream_plan_sha256, name="expected upstream plan sha256"
    )
    nested_digest = _require_sha256(
        expected_nested_plan_digest, name="expected nested plan digest"
    )
    nested_sha = _require_sha256(
        expected_nested_plan_sha256, name="expected nested plan sha256"
    )
    registration_sha = _require_sha256(
        expected_registration_receipt_sha256,
        name="expected registration receipt sha256",
    )
    if (
        _sha256_bytes(payloads["upstream_census_plan"]) != upstream_sha
        or _sha256_bytes(payloads["nested_plan"]) != nested_sha
        or _sha256_bytes(payloads["registration_receipt"]) != registration_sha
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "held plan or registration bytes differ from exact external pins"
        )
    try:
        upstream_text = payloads["upstream_census_plan"].decode("ascii")
        nested_text = payloads["nested_plan"].decode("ascii")
        freeze_text = payloads["freeze_request"].decode("ascii")
        registration_text = payloads["registration_receipt"].decode("ascii")
    except UnicodeDecodeError as exc:
        raise NestedOpeningFeasibilityExecutionV1Error(
            "held prospective or registration artifact is not ASCII"
        ) from exc
    upstream = parse_evaluation_census_plan_v1(
        upstream_text, expected_digest=upstream_digest
    )
    if require_production:
        nested = parse_nested_opening_construction_feasibility_plan_v1(
            nested_text,
            census_plan_text=upstream_text,
            expected_census_plan_digest=upstream_digest,
            expected_census_plan_bytes_sha256=upstream_sha,
            expected_plan_digest=nested_digest,
        )
        _assert_production_plans(upstream, nested)
        environment = _environment_input_bindings_from_payloads(
            {
                "source_archive": payloads["source_archive"],
                "constraints": payloads["constraints"],
                "environment_lock": payloads["environment_lock"],
            }
        )
        plan_class = _PRODUCTION_PLAN_CLASS
    else:
        nested = parse_nested_opening_construction_feasibility_plan_for_testing_v1(
            nested_text,
            census_plan_text=upstream_text,
            expected_census_plan_digest=upstream_digest,
            expected_census_plan_bytes_sha256=upstream_sha,
            expected_plan_digest=nested_digest,
        )
        environment = _environment_input_bindings_from_payloads({})
        plan_class = _ENGINEERING_PLAN_CLASS
    request = NestedOpeningFeasibilityFreezeRequestV1(
        plan_class,
        _upstream_plan_binding(upstream, payloads["upstream_census_plan"]),
        _nested_plan_binding(nested, nested_digest, payloads["nested_plan"]),
        upstream.source_binding.as_obj(),
        upstream.catalog_binding.as_obj(),
        upstream.generator_binding.as_obj(),
        nested.source_binding.as_obj(),
        nested.upstream_identity_plan_binding.as_obj(),
        nested.generator_binding.as_obj(),
        _execution_code_binding(runner_source_path),
        environment,
        _fixed_budget(nested),
    )
    if _serialize_freeze_request_after_replay(request) != freeze_text:
        raise NestedOpeningFeasibilityExecutionV1Error(
            "held freeze request differs from both plans, code, or environment replay"
        )
    registration = parse_external_nested_opening_feasibility_registration_receipt_v1(
        registration_text,
        freeze_request=request,
        freeze_request_text=freeze_text,
        expected_bytes_sha256=registration_sha,
        expected_registration_reference=expected_registration_reference,
        expected_execution_uuid=expected_execution_uuid,
        expected_execution_nonce=expected_execution_nonce,
        expected_execution_deadline_seconds=expected_execution_deadline_seconds,
    )
    if _registration_datetime(registration.registered_at_utc) > _wall_datetime(
        controller_started_wall_utc, name="controller started wall UTC"
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "registration timestamp is after controller start by supplied/local clocks"
        )
    return upstream, nested, request, registration, payloads


def _started_obj(
    *,
    execution_uuid: str,
    execution_nonce: str,
    registration: ExternalNestedOpeningFeasibilityRegistrationReceiptV1,
    controller_started_wall_utc: str,
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
        "schema_version": NESTED_OPENING_FEASIBILITY_EXECUTION_SCHEMA_VERSION,
        "receipt_kind": _STARTED_KIND,
        "status": "started_no_retry",
        "execution_uuid": execution_uuid,
        "execution_nonce": execution_nonce,
        "wall_clock": {
            "registered_at_utc": registration.registered_at_utc,
            "controller_started_at_utc": controller_started_wall_utc,
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
            "covers_held_input_acquisition_and_hashing": True,
        },
        "input_binding": dict(input_binding),
        "execution_policy": dict(_EXECUTION_POLICY),
        "execution_scope": dict(_ENGINEERING_EXECUTION_SCOPE),
        "reconstruction_claim_boundary": dict(_RECONSTRUCTION_CLAIM_BOUNDARY),
        "authorization": dict(_AUTHORIZATION),
    }
    return {
        **unsigned,
        "started_receipt_digest": _digest(unsigned, domain=_STARTED_DOMAIN),
    }


def _failure_obj(
    *,
    execution_uuid: str,
    failed_stage: str,
    exc: BaseException,
    started_receipt_basename_present: bool,
    started_receipt_sha256: str | None,
    root_inventory_at_failure: Sequence[str],
) -> dict[str, Any]:
    unsigned = {
        "schema_version": NESTED_OPENING_FEASIBILITY_EXECUTION_SCHEMA_VERSION,
        "receipt_kind": _FAILURE_RECEIPT_KIND,
        "status": "failed_incomplete_root_do_not_reuse",
        "execution_uuid": execution_uuid,
        "failed_stage": failed_stage,
        "exception_type": type(exc).__name__,
        "exception_message_sha256": _sha256_bytes(str(exc).encode("utf-8")),
        "started_receipt_basename_present": started_receipt_basename_present,
        "started_receipt_exact_and_canonical": started_receipt_sha256 is not None,
        "started_receipt_sha256": started_receipt_sha256,
        "root_inventory_at_failure": list(root_inventory_at_failure),
        "completion_receipt_present": False,
        "retry_within_execution_uuid_allowed": False,
        "execution_scope": dict(_ENGINEERING_EXECUTION_SCOPE),
        "reconstruction_claim_boundary": dict(_RECONSTRUCTION_CLAIM_BOUNDARY),
        "authorization": dict(_AUTHORIZATION),
    }
    return {
        **unsigned,
        "failure_receipt_digest": _digest(unsigned, domain=_FAILURE_RECEIPT_DOMAIN),
    }


def _report_fd_snapshot(
    report_fd: int,
    *,
    expected_byte_count: int,
    expected_sha256: str,
    expected_nlink: int,
    name: str,
) -> tuple[os.stat_result, bytes]:
    before = os.fstat(report_fd)
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) != 0o400
        or before.st_nlink != expected_nlink
        or before.st_size != expected_byte_count
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            f"{name} has unsafe metadata"
        )
    payload = _pread_exact(report_fd, expected_byte_count, name=name)
    after = os.fstat(report_fd)
    if (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ) != (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) or _sha256_bytes(payload) != expected_sha256:
        raise NestedOpeningFeasibilityExecutionV1Error(
            f"{name} changed during held-descriptor replay"
        )
    return after, payload


def _require_report_basename_matches_fd(
    directory_fd: int,
    basename: str,
    report_stat: os.stat_result,
    *,
    expected_nlink: int,
    name: str,
) -> None:
    path_stat = os.stat(basename, dir_fd=directory_fd, follow_symlinks=False)
    if (
        not stat.S_ISREG(path_stat.st_mode)
        or stat.S_IMODE(path_stat.st_mode) != 0o400
        or path_stat.st_nlink != expected_nlink
        or (
            path_stat.st_dev,
            path_stat.st_ino,
            path_stat.st_mode,
            path_stat.st_nlink,
            path_stat.st_size,
            path_stat.st_mtime_ns,
            path_stat.st_ctime_ns,
        )
        != (
            report_stat.st_dev,
            report_stat.st_ino,
            report_stat.st_mode,
            report_stat.st_nlink,
            report_stat.st_size,
            report_stat.st_mtime_ns,
            report_stat.st_ctime_ns,
        )
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            f"{name} does not name the exact held report inode"
        )


def _commit_report(
    directory_fd: int,
    report_fd: int,
    *,
    expected_byte_count: int,
    expected_sha256: str,
) -> bytes:
    pre_link, payload = _report_fd_snapshot(
        report_fd,
        expected_byte_count=expected_byte_count,
        expected_sha256=expected_sha256,
        expected_nlink=1,
        name="verified report spool",
    )
    _require_report_basename_matches_fd(
        directory_fd,
        ".observed-report.spool",
        pre_link,
        expected_nlink=1,
        name="verified report spool basename",
    )
    try:
        os.link(
            ".observed-report.spool",
            REPORT_FILENAME,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
    except FileExistsError as exc:
        raise NestedOpeningFeasibilityExecutionV1Error(
            "final report path already exists"
        ) from exc
    os.fsync(directory_fd)
    linked, linked_payload = _report_fd_snapshot(
        report_fd,
        expected_byte_count=expected_byte_count,
        expected_sha256=expected_sha256,
        expected_nlink=2,
        name="linked report",
    )
    _require_report_basename_matches_fd(
        directory_fd,
        ".observed-report.spool",
        linked,
        expected_nlink=2,
        name="linked report spool basename",
    )
    _require_report_basename_matches_fd(
        directory_fd,
        REPORT_FILENAME,
        linked,
        expected_nlink=2,
        name="linked final report basename",
    )
    if linked_payload != payload or linked.st_mtime_ns != pre_link.st_mtime_ns:
        raise NestedOpeningFeasibilityExecutionV1Error(
            "report changed during hard-link transition"
        )
    os.unlink(".observed-report.spool", dir_fd=directory_fd)
    os.fsync(directory_fd)
    committed, committed_payload = _report_fd_snapshot(
        report_fd,
        expected_byte_count=expected_byte_count,
        expected_sha256=expected_sha256,
        expected_nlink=1,
        name="committed report",
    )
    _require_report_basename_matches_fd(
        directory_fd,
        REPORT_FILENAME,
        committed,
        expected_nlink=1,
        name="committed final report basename",
    )
    if committed_payload != payload or committed.st_mtime_ns != pre_link.st_mtime_ns:
        raise NestedOpeningFeasibilityExecutionV1Error(
            "committed report differs from the verified held spool"
        )
    return payload


@dataclass(frozen=True, slots=True)
class ExecutedNestedOpeningFeasibilityArtifactsV1:
    output_root: Path
    report_path: Path
    execution_receipt_path: Path
    report_digest: str
    report_bytes_sha256: str
    execution_receipt_digest: str


def _duplicate_worker_semantic_fds(
    files: Mapping[str, _HeldOrdinaryFileV1],
    *,
    extra_label: str,
    extra_fd: int,
) -> tuple[dict[str, int], list[int]]:
    held_map: dict[str, int] = {}
    duplicates: list[int] = []
    try:
        for label, held in files.items():
            duplicate = os.dup(held.file_fd)
            os.set_inheritable(duplicate, False)
            held_map[label] = duplicate
            duplicates.append(duplicate)
        extra_duplicate = os.dup(extra_fd)
        os.set_inheritable(extra_duplicate, False)
        duplicates.append(extra_duplicate)
        if len(set(duplicates)) != len(duplicates):
            raise NestedOpeningFeasibilityExecutionV1Error(
                "duplicated worker semantic descriptors are not unique"
            )
        return held_map, duplicates
    except BaseException:
        for descriptor in duplicates:
            with suppress(OSError):
                os.close(descriptor)
        raise


def _watchdog_argv(
    *,
    channels: _ParentDeathPipe,
    deadline_monotonic_ns: int,
    forward_fds: Sequence[int],
    worker_argv: Sequence[str],
) -> list[str]:
    return [
        *_safe_bootstrap_launch_prefix(role="--g03-bootstrap-watchdog"),
        "__watchdog-worker",
        "--parent-death-fd",
        str(channels.read_fd),
        "--worker-guard-read-fd",
        str(channels.worker_read_fd),
        "--worker-guard-write-fd",
        str(channels.worker_write_fd),
        "--worker-pid-report-fd",
        str(channels.pid_write_fd),
        "--guardian-ready-read-fd",
        str(channels.guardian_ready_read_fd),
        "--guardian-ready-write-fd",
        str(channels.guardian_ready_write_fd),
        "--worker-lifetime-read-fd",
        str(channels.worker_lifetime_read_fd),
        "--worker-lifetime-write-fd",
        str(channels.worker_lifetime_write_fd),
        "--deadline-monotonic-ns",
        str(deadline_monotonic_ns),
        "--forward-fds-json",
        _dump_json(list(forward_fds)),
        "--worker-argv-json",
        _dump_json(list(worker_argv)),
    ]


def _execute(
    *,
    upstream_census_plan_path: Path,
    nested_plan_path: Path,
    freeze_request_path: Path,
    registration_receipt_path: Path,
    expected_upstream_census_plan_digest: str,
    expected_upstream_census_plan_bytes_sha256: str,
    expected_nested_plan_digest: str,
    expected_nested_plan_bytes_sha256: str,
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
    entry_started_monotonic_ns: int,
    entry_started_wall_utc: str,
) -> ExecutedNestedOpeningFeasibilityArtifactsV1:
    if require_production:
        raise NestedOpeningFeasibilityExecutionV1Error(
            PRODUCTION_EXECUTION_REFUSAL_CODE
        )
    _require_safe_bootstrap_runtime()
    _require_single_threaded_controller()
    _require_default_sigchld()
    initial_signal_mask = cast(
        set[signal.Signals], signal.pthread_sigmask(signal.SIG_BLOCK, set())
    )
    if initial_signal_mask.intersection(_managed_controller_signals()):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "engineering controller requires all managed signals initially unblocked"
        )
    upstream_digest = _require_sha256(
        expected_upstream_census_plan_digest,
        name="expected upstream census plan digest",
    )
    upstream_sha = _require_sha256(
        expected_upstream_census_plan_bytes_sha256,
        name="expected upstream census plan sha256",
    )
    nested_digest = _require_sha256(
        expected_nested_plan_digest, name="expected nested plan digest"
    )
    nested_sha = _require_sha256(
        expected_nested_plan_bytes_sha256, name="expected nested plan sha256"
    )
    registration_sha = _require_sha256(
        expected_registration_receipt_sha256,
        name="expected registration receipt sha256",
    )
    registration_reference = _require_identifier(
        expected_registration_reference, name="expected registration reference"
    )
    execution_uuid = _require_uuid(expected_execution_uuid)
    execution_nonce = _require_nonce(expected_execution_nonce)
    deadline_value = _require_integer(
        deadline_seconds, name="deadline seconds", minimum=1, maximum=86_400
    )
    if _lexical_absolute(output_root).name != execution_uuid:
        raise NestedOpeningFeasibilityExecutionV1Error(
            "output-root basename must equal the externally registered execution UUID"
        )
    if (
        type(controller_argv) not in (tuple, list)
        or not controller_argv
        or any(type(item) is not str or "\0" in item for item in controller_argv)
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "controller argv must be nonempty exact strings"
        )
    if not hasattr(signal, "setitimer") or signal.getitimer(signal.ITIMER_REAL) != (
        0.0,
        0.0,
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "execution requires one unused POSIX real-time interval timer"
        )

    controller_started_wall_utc = entry_started_wall_utc
    _wall_datetime(
        controller_started_wall_utc, name="controller public-entry wall UTC"
    )
    started_ns = _require_integer(
        entry_started_monotonic_ns,
        name="controller public-entry monotonic ns",
        minimum=1,
    )
    deadline_ns = started_ns + deadline_value * 1_000_000_000
    remaining_ns = deadline_ns - time.monotonic_ns()
    if remaining_ns <= 0:
        raise NestedOpeningFeasibilityExecutionV1Error(
            "execution deadline expired before exclusive root creation"
        )
    previous_signal_handlers: dict[int, Any] = {}
    timer_armed = False
    inherited_signal_mask: set[signal.Signals] | None = None
    files: dict[str, _HeldOrdinaryFileV1] = {}
    directory_fd = -1
    report_fd = -1
    root: Path | None = None
    started_bytes = b""
    started_committed = False
    receipt_bytes = b""
    builder_channels: _ParentDeathPipe | None = None
    verifier_channels: _ParentDeathPipe | None = None
    builder_forward: list[int] = []
    verifier_forward: list[int] = []
    root_snapshot: _HeldExecutionRootSnapshotV1 | None = None
    failed_stage = "exclusive_root_creation"
    inherited_signal_mask = cast(
        set[signal.Signals],
        signal.pthread_sigmask(
            signal.SIG_BLOCK, set(_managed_controller_signals())
        ),
    )
    try:
        # Every managed signal is blocked before handler installation and
        # remains blocked until both handler and timer ownership are published
        # inside this outer cleanup try.
        previous_signal_handlers = _install_controller_signal_handlers()
        arm_remaining_ns = deadline_ns - time.monotonic_ns()
        if arm_remaining_ns <= 0:
            raise NestedOpeningFeasibilityExecutionV1Error(
                "execution deadline expired before interval timer arm"
            )
        signal.setitimer(signal.ITIMER_REAL, arm_remaining_ns / 1_000_000_000)
        timer_armed = True
        root, directory_fd = _secure_absent_execution_root_with_failure_fallback(
            output_root,
            execution_uuid=execution_uuid,
        )
        # Publish exact root/FD ownership to outer cleanup before any managed
        # signal can run a Python handler between CALL return and tuple stores.
        signal.pthread_sigmask(signal.SIG_SETMASK, inherited_signal_mask)
        failed_stage = "held_input_preflight"
        files = _open_execution_inputs(
            upstream_census_plan_path=upstream_census_plan_path,
            nested_plan_path=nested_plan_path,
            freeze_request_path=freeze_request_path,
            registration_receipt_path=registration_receipt_path,
            source_archive_path=source_archive_path,
            constraints_path=constraints_path,
            environment_lock_path=environment_lock_path,
            require_production=require_production,
        )
        upstream, nested, request, registration, payloads = _controller_replay_held_chain(
            files,
            expected_upstream_plan_digest=upstream_digest,
            expected_upstream_plan_sha256=upstream_sha,
            expected_nested_plan_digest=nested_digest,
            expected_nested_plan_sha256=nested_sha,
            expected_registration_receipt_sha256=registration_sha,
            expected_registration_reference=registration_reference,
            expected_execution_uuid=execution_uuid,
            expected_execution_nonce=execution_nonce,
            expected_execution_deadline_seconds=deadline_value,
            runner_source_path=runner_source_path,
            controller_started_wall_utc=controller_started_wall_utc,
            require_production=require_production,
        )
        if time.monotonic_ns() > deadline_ns:
            raise NestedOpeningFeasibilityExecutionV1Error(
                "registered deadline expired during held-input preflight"
            )
        root_stat = os.fstat(directory_fd)
        if (root_stat.st_dev, root_stat.st_ino) in {
            (held.file_dev, held.file_ino) for held in files.values()
        }:
            raise NestedOpeningFeasibilityExecutionV1Error(
                "output root aliases a held input inode"
            )
        held_bindings = _held_input_bindings(files)
        plan_bindings = {
            "upstream_evaluation_census_plan": _upstream_plan_binding(
                upstream, payloads["upstream_census_plan"]
            ),
            "nested_opening_feasibility_plan": {
                "prospective_plan_digest": nested_digest,
                "canonical_plan_bytes_sha256": nested_sha,
                "canonical_plan_byte_count": len(payloads["nested_plan"]),
            },
        }
        input_binding = {
            "upstream_evaluation_census_plan": plan_bindings[
                "upstream_evaluation_census_plan"
            ],
            "nested_opening_feasibility_plan": _nested_plan_binding(
                nested, nested_digest, payloads["nested_plan"]
            ),
            "freeze_request": {
                "freeze_request_digest": request._digest_unverified(),
                "canonical_freeze_request_bytes_sha256": _sha256_bytes(
                    payloads["freeze_request"]
                ),
                "canonical_freeze_request_byte_count": len(payloads["freeze_request"]),
            },
            "external_registration_receipt": {
                "registration_reference": registration.registration_reference,
                "registration_receipt_digest": registration._digest_unverified(),
                "exact_registration_receipt_bytes_sha256": registration_sha,
                "exact_registration_receipt_byte_count": len(
                    payloads["registration_receipt"]
                ),
            },
            "execution_code_binding": dict(request.execution_code_binding),
            "environment_input_bindings": dict(request.environment_input_bindings),
        }

        module_argv = _safe_bootstrap_launch_prefix(
            role="--g03-bootstrap-worker"
        )
        common_args = [
            "--expected-upstream-plan-digest",
            upstream_digest,
            "--expected-upstream-plan-sha256",
            upstream_sha,
            "--expected-nested-plan-digest",
            nested_digest,
            "--expected-nested-plan-sha256",
            nested_sha,
            "--expected-freeze-request-digest",
            request._digest_unverified(),
            "--expected-registration-receipt-sha256",
            registration_sha,
            "--expected-registration-reference",
            registration_reference,
            "--expected-execution-uuid",
            execution_uuid,
            "--expected-execution-nonce",
            execution_nonce,
            "--expected-execution-deadline-seconds",
            str(deadline_value),
            "--controller-started-wall-utc",
            controller_started_wall_utc,
            "--deadline-monotonic-ns",
            str(deadline_ns),
            "--runner-source",
            str(_lexical_absolute(runner_source_path)),
        ]

        builder_channels = _ParentDeathPipe.create()
        builder_map, builder_forward = _duplicate_worker_semantic_fds(
            files, extra_label="output_root", extra_fd=directory_fd
        )
        builder_root_fd = builder_forward[-1]
        builder_argv = [
            *module_argv,
            "__build-worker",
            "--held-input-fds-json",
            _dump_json(builder_map),
            "--held-input-bindings-json",
            _dump_json(held_bindings),
            *common_args,
            "--parent-guard-fd",
            str(builder_channels.worker_read_fd),
            "--guardian-ready-write-fd",
            str(builder_channels.guardian_ready_write_fd),
            "--worker-lifetime-write-fd",
            str(builder_channels.worker_lifetime_write_fd),
            "--output-root-fd",
            str(builder_root_fd),
        ]
        if require_production:
            builder_argv.append("--require-production")
        builder_watchdog_argv = _watchdog_argv(
            channels=builder_channels,
            deadline_monotonic_ns=deadline_ns,
            forward_fds=builder_forward,
            worker_argv=builder_argv,
        )

        # The verifier's exact command is completed only after the builder
        # supplies the independently replayed report digest and whole-file pin.
        verifier_argv_prefix = [*module_argv, "__verify-worker"]
        verifier_watchdog_argv_prefix = [
            *_safe_bootstrap_launch_prefix(role="--g03-bootstrap-watchdog"),
            "__watchdog-worker",
        ]
        started = _started_obj(
            execution_uuid=execution_uuid,
            execution_nonce=execution_nonce,
            registration=registration,
            controller_started_wall_utc=controller_started_wall_utc,
            controller_argv=controller_argv,
            builder_argv=builder_argv,
            verifier_argv_prefix=verifier_argv_prefix,
            builder_watchdog_argv=builder_watchdog_argv,
            verifier_watchdog_argv_prefix=verifier_watchdog_argv_prefix,
            deadline_seconds=deadline_value,
            started_monotonic_ns=started_ns,
            deadline_monotonic_ns=deadline_ns,
            input_binding=input_binding,
        )
        started_bytes = _canonical_bytes(started)
        failed_stage = "write_started_receipt"
        _write_exclusive_at(directory_fd, STARTED_RECEIPT_FILENAME, started_bytes)
        _require_exact_artifact_at(
            directory_fd, STARTED_RECEIPT_FILENAME, started_bytes
        )
        started_committed = True

        failed_stage = "builder"
        try:
            builder_watchdog, builder_watchdog_pid, builder_pid = _run_watchdog(
                builder_watchdog_argv,
                deadline_monotonic_ns=deadline_ns,
                channels=builder_channels,
                forward_fds=builder_forward,
            )
        finally:
            # _run_watchdog owns and closes these copies on every path.  Clear
            # controller ownership immediately so a later FD-number reuse can
            # never be double-closed by the outer cleanup.
            builder_forward = []
            builder_channels = None
        builder_terminal = _parse_watchdog_and_worker_terminal(
            builder_watchdog,
            expected_watchdog_pid=builder_watchdog_pid,
            expected_watchdog_argv=builder_watchdog_argv,
            expected_worker_pid=builder_pid,
            expected_worker_argv=builder_argv,
            expected_worker_kind=_BUILDER_TERMINAL_KIND,
            expected_deadline_monotonic_ns=deadline_ns,
            expected_plan_bindings=plan_bindings,
            expected_held_bindings=held_bindings,
            require_production=require_production,
        )
        builder_watchdog_bytes = _canonical_bytes(builder_watchdog)
        builder_terminal_bytes = _canonical_bytes(builder_terminal)
        _write_exclusive_at(
            directory_fd,
            BUILDER_GUARDIAN_TERMINAL_FILENAME,
            builder_watchdog_bytes,
        )
        _write_exclusive_at(
            directory_fd, BUILDER_TERMINAL_FILENAME, builder_terminal_bytes
        )
        report_binding = cast(Mapping[str, Any], builder_terminal["report_binding"])
        report_digest = _require_sha256(
            report_binding["observed_report_digest"], name="report digest"
        )
        report_sha = _require_sha256(
            report_binding["exact_report_bytes_sha256"], name="report sha256"
        )
        report_count = _require_integer(
            report_binding["exact_report_byte_count"],
            name="report byte count",
            minimum=1,
            maximum=1024 * 1024 * 1024,
        )
        report_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            report_flags |= os.O_NOFOLLOW
        report_fd = os.open(
            ".observed-report.spool", report_flags, dir_fd=directory_fd
        )
        report_stat = os.fstat(report_fd)
        if (
            not stat.S_ISREG(report_stat.st_mode)
            or stat.S_IMODE(report_stat.st_mode) != 0o400
            or report_stat.st_nlink != 1
            or report_stat.st_size != report_count
            or (report_stat.st_dev, report_stat.st_ino)
            in {(held.file_dev, held.file_ino) for held in files.values()}
        ):
            raise NestedOpeningFeasibilityExecutionV1Error(
                "builder report spool has unsafe metadata or inode alias"
            )
        report_bytes = _pread_exact(report_fd, report_count, name="report spool")
        report_after = os.fstat(report_fd)
        if (
            report_after.st_dev,
            report_after.st_ino,
            report_after.st_mode,
            report_after.st_nlink,
            report_after.st_size,
            report_after.st_mtime_ns,
            report_after.st_ctime_ns,
        ) != (
            report_stat.st_dev,
            report_stat.st_ino,
            report_stat.st_mode,
            report_stat.st_nlink,
            report_stat.st_size,
            report_stat.st_mtime_ns,
            report_stat.st_ctime_ns,
        ) or _sha256_bytes(report_bytes) != report_sha:
            raise NestedOpeningFeasibilityExecutionV1Error(
                "builder report spool differs from terminal pin"
            )

        failed_stage = "verifier"
        verifier_channels = _ParentDeathPipe.create()
        verifier_map, verifier_forward = _duplicate_worker_semantic_fds(
            files, extra_label="report", extra_fd=report_fd
        )
        verifier_report_fd = verifier_forward[-1]
        verifier_argv = [
            *module_argv,
            "__verify-worker",
            "--held-input-fds-json",
            _dump_json(verifier_map),
            "--held-input-bindings-json",
            _dump_json(held_bindings),
            *common_args,
            "--parent-guard-fd",
            str(verifier_channels.worker_read_fd),
            "--guardian-ready-write-fd",
            str(verifier_channels.guardian_ready_write_fd),
            "--worker-lifetime-write-fd",
            str(verifier_channels.worker_lifetime_write_fd),
            "--report-fd",
            str(verifier_report_fd),
            "--expected-report-digest",
            report_digest,
            "--expected-report-sha256",
            report_sha,
            "--expected-report-byte-count",
            str(report_count),
        ]
        if require_production:
            verifier_argv.append("--require-production")
        verifier_watchdog_argv = _watchdog_argv(
            channels=verifier_channels,
            deadline_monotonic_ns=deadline_ns,
            forward_fds=verifier_forward,
            worker_argv=verifier_argv,
        )
        try:
            verifier_watchdog, verifier_watchdog_pid, verifier_pid = _run_watchdog(
                verifier_watchdog_argv,
                deadline_monotonic_ns=deadline_ns,
                channels=verifier_channels,
                forward_fds=verifier_forward,
            )
        finally:
            verifier_forward = []
            verifier_channels = None
        verifier_terminal = _parse_watchdog_and_worker_terminal(
            verifier_watchdog,
            expected_watchdog_pid=verifier_watchdog_pid,
            expected_watchdog_argv=verifier_watchdog_argv,
            expected_worker_pid=verifier_pid,
            expected_worker_argv=verifier_argv,
            expected_worker_kind=_VERIFIER_TERMINAL_KIND,
            expected_deadline_monotonic_ns=deadline_ns,
            expected_plan_bindings=plan_bindings,
            expected_held_bindings=held_bindings,
            require_production=require_production,
        )
        if (
            not _exact_json_equal(
                verifier_terminal["report_binding"],
                builder_terminal["report_binding"],
            )
            or not _exact_json_equal(
                verifier_terminal["accounting"], builder_terminal["accounting"]
            )
        ):
            raise NestedOpeningFeasibilityExecutionV1Error(
                "fresh verifier differs from builder report binding or accounting"
            )
        verifier_watchdog_bytes = _canonical_bytes(verifier_watchdog)
        verifier_terminal_bytes = _canonical_bytes(verifier_terminal)
        _write_exclusive_at(
            directory_fd,
            VERIFIER_GUARDIAN_TERMINAL_FILENAME,
            verifier_watchdog_bytes,
        )
        _write_exclusive_at(
            directory_fd, VERIFIER_TERMINAL_FILENAME, verifier_terminal_bytes
        )

        for held in files.values():
            held.replay_external_basename()
            held.read_bytes()
        if _execution_code_binding(runner_source_path) != request.execution_code_binding:
            raise NestedOpeningFeasibilityExecutionV1Error(
                "execution code changed after fresh-process verification"
            )
        if time.monotonic_ns() > deadline_ns:
            raise NestedOpeningFeasibilityExecutionV1Error(
                "registered deadline expired before report commit"
            )
        failed_stage = "commit_report"
        committed_report_bytes = _commit_report(
            directory_fd,
            report_fd,
            expected_byte_count=report_count,
            expected_sha256=report_sha,
        )
        if committed_report_bytes != report_bytes:
            raise NestedOpeningFeasibilityExecutionV1Error(
                "committed report differs from held verified spool"
            )

        failed_stage = "write_completion_receipt"
        unsigned_receipt = {
            "schema_version": NESTED_OPENING_FEASIBILITY_EXECUTION_SCHEMA_VERSION,
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
                "external_origin_proven_by_expected_pin": False,
                **_REGISTRATION_CLAIM_BOUNDARY,
                **_RECONSTRUCTION_CLAIM_BOUNDARY,
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
                    "watchdog_terminal_digest": builder_watchdog[
                        "watchdog_terminal_digest"
                    ],
                },
                "builder_terminal": {
                    "filename": BUILDER_TERMINAL_FILENAME,
                    "sha256": _sha256_bytes(builder_terminal_bytes),
                    "byte_count": len(builder_terminal_bytes),
                    "worker_terminal_digest": builder_terminal[
                        "worker_terminal_digest"
                    ],
                },
                "verifier_guardian_terminal": {
                    "filename": VERIFIER_GUARDIAN_TERMINAL_FILENAME,
                    "sha256": _sha256_bytes(verifier_watchdog_bytes),
                    "byte_count": len(verifier_watchdog_bytes),
                    "watchdog_terminal_digest": verifier_watchdog[
                        "watchdog_terminal_digest"
                    ],
                },
                "verifier_terminal": {
                    "filename": VERIFIER_TERMINAL_FILENAME,
                    "sha256": _sha256_bytes(verifier_terminal_bytes),
                    "byte_count": len(verifier_terminal_bytes),
                    "worker_terminal_digest": verifier_terminal[
                        "worker_terminal_digest"
                    ],
                },
            },
            "report_binding": {
                "filename": REPORT_FILENAME,
                "observed_report_digest": report_digest,
                "exact_report_bytes_sha256": report_sha,
                "exact_report_byte_count": report_count,
                "committed_after_fresh_process_verification": True,
            },
            "accounting": builder_terminal["accounting"],
            "runtime_evidence": {
                "contract": _RUNTIME_EVIDENCE_CONTRACT,
                "builder": {
                    "timing": builder_terminal["timing"],
                    "maximum_resident_set_size": builder_terminal[
                        "maximum_resident_set_size"
                    ],
                    "process_identity": builder_terminal["process_identity"],
                },
                "builder_watchdog": {
                    "timing": builder_watchdog["timing"],
                    "process_identity": builder_watchdog[
                        "watchdog_process_identity"
                    ],
                    "containment": builder_watchdog["containment"],
                },
                "verifier": {
                    "timing": verifier_terminal["timing"],
                    "maximum_resident_set_size": verifier_terminal[
                        "maximum_resident_set_size"
                    ],
                    "process_identity": verifier_terminal["process_identity"],
                },
                "verifier_watchdog": {
                    "timing": verifier_watchdog["timing"],
                    "process_identity": verifier_watchdog[
                        "watchdog_process_identity"
                    ],
                    "containment": verifier_watchdog["containment"],
                },
                "controller_terminal_commit_admission": None,
            },
            "execution_policy": dict(_EXECUTION_POLICY),
            "execution_scope": dict(_ENGINEERING_EXECUTION_SCOPE),
            "scientific_claim_boundary": dict(
                _EXECUTED_SCIENTIFIC_CLAIM_BOUNDARY
            ),
            "reconstruction_claim_boundary": dict(_RECONSTRUCTION_CLAIM_BOUNDARY),
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
        if time.monotonic_ns() > deadline_ns:
            raise NestedOpeningFeasibilityExecutionV1Error(
                "registered deadline expired before durable completion"
            )
        for held in files.values():
            held.replay_external_basename()
            held.read_bytes()
        if _execution_code_binding(runner_source_path) != request.execution_code_binding:
            raise NestedOpeningFeasibilityExecutionV1Error(
                "execution code changed before durable completion"
            )
        precompletion_payloads = {
            STARTED_RECEIPT_FILENAME: started_bytes,
            BUILDER_GUARDIAN_TERMINAL_FILENAME: builder_watchdog_bytes,
            BUILDER_TERMINAL_FILENAME: builder_terminal_bytes,
            VERIFIER_GUARDIAN_TERMINAL_FILENAME: verifier_watchdog_bytes,
            VERIFIER_TERMINAL_FILENAME: verifier_terminal_bytes,
            REPORT_FILENAME: report_bytes,
        }
        root_snapshot = _HeldExecutionRootSnapshotV1.acquire(
            root, directory_fd, precompletion_payloads
        )
        root_snapshot.replay()
        if time.monotonic_ns() > deadline_ns:
            raise NestedOpeningFeasibilityExecutionV1Error(
                "registered deadline expired after final input replay"
            )
        # Block every managed asynchronous signal and disarm the owned alarm
        # before the final held replay.  The following monotonic stamp is the
        # exact admission boundary: all inputs/root bytes were replayed and no
        # asynchronous handler can interrupt the terminal mutation.  Durable
        # write completion is deliberately not claimed to precede the
        # registered admission deadline.
        _require_single_threaded_controller()
        _block_terminal_signals_and_disarm_alarm(
            timer_armed=timer_armed
        )
        timer_armed = False
        for held in files.values():
            held.replay_external_basename()
            held.read_bytes()
        if _execution_code_binding(runner_source_path) != request.execution_code_binding:
            raise NestedOpeningFeasibilityExecutionV1Error(
                "execution code changed at terminal commit admission"
            )
        root_snapshot.replay()
        admitted_ns = time.monotonic_ns()
        if admitted_ns > deadline_ns:
            raise NestedOpeningFeasibilityExecutionV1Error(
                "registered deadline expired before terminal commit admission"
            )
        admitted_wall_utc = _utc_now()
        runtime_evidence = cast(dict[str, Any], unsigned_receipt["runtime_evidence"])
        runtime_evidence["controller_terminal_commit_admission"] = {
            "pid": os.getpid(),
            "started_monotonic_ns": started_ns,
            "admitted_monotonic_ns": admitted_ns,
            "elapsed_to_admission_ns": admitted_ns - started_ns,
            "deadline_seconds": deadline_value,
            "deadline_monotonic_ns": deadline_ns,
            "started_wall_utc": controller_started_wall_utc,
            "admitted_wall_utc": admitted_wall_utc,
            "durable_write_completion_timestamp_recorded_in_same_receipt": False,
        }
        receipt_obj = {
            **unsigned_receipt,
            "execution_receipt_digest": _digest(
                unsigned_receipt, domain=_EXECUTION_RECEIPT_DOMAIN
            ),
        }
        receipt_bytes = _canonical_bytes(receipt_obj)
        # This exclusive durable write is the terminal lifecycle mutation.
        _write_exclusive_at(directory_fd, EXECUTION_RECEIPT_FILENAME, receipt_bytes)
        _require_exact_artifact_at(
            directory_fd, EXECUTION_RECEIPT_FILENAME, receipt_bytes
        )
        root_snapshot.add_exact_file(EXECUTION_RECEIPT_FILENAME, receipt_bytes)
        root_snapshot.replay(require_root_timestamps_stable=False)
        return ExecutedNestedOpeningFeasibilityArtifactsV1(
            root,
            root / REPORT_FILENAME,
            root / EXECUTION_RECEIPT_FILENAME,
            report_digest,
            report_sha,
            cast(str, receipt_obj["execution_receipt_digest"]),
        )
    except BaseException as exc:
        # Re-block unconditionally.  A pending signal can raise during the
        # setup unmask itself, so no Python sentinel can safely represent the
        # current OS mask across that boundary.
        with suppress(BaseException):
            signal.pthread_sigmask(
                signal.SIG_BLOCK, set(_managed_controller_signals())
            )
        if timer_armed:
            with suppress(BaseException):
                signal.setitimer(signal.ITIMER_REAL, 0)
            timer_armed = False
        if directory_fd >= 0 and root is not None:
            try:
                inventory = tuple(sorted(os.listdir(directory_fd)))
                canonical_completion_present = False
                if EXECUTION_RECEIPT_FILENAME in inventory and receipt_bytes:
                    try:
                        _durably_require_exact_artifact_at(
                            directory_fd,
                            EXECUTION_RECEIPT_FILENAME,
                            receipt_bytes,
                        )
                        canonical_completion_present = True
                    except BaseException:
                        # A terminal write that failed before exact read-back is
                        # not a completion artifact.  Remove only this basename
                        # in the newly created private root, then durably stamp
                        # the root as failed and non-reusable.
                        os.unlink(EXECUTION_RECEIPT_FILENAME, dir_fd=directory_fd)
                        os.fsync(directory_fd)
                        inventory = tuple(sorted(os.listdir(directory_fd)))
                if canonical_completion_present:
                    # A canonical terminal receipt is the root's terminal
                    # truth even if cleanup or return-path mechanics failed.
                    raise NestedOpeningFeasibilityExecutionV1Error(
                        "canonical completion receipt exists; contradictory failure receipt forbidden"
                    )
                if STARTED_RECEIPT_FILENAME in inventory and not started_committed:
                    try:
                        _require_exact_artifact_at(
                            directory_fd, STARTED_RECEIPT_FILENAME, started_bytes
                        )
                        started_committed = True
                    except BaseException:
                        # Keep the partial basename visible in the failure
                        # inventory, but do not call it the exact started receipt.
                        pass
                failure = _failure_obj(
                    execution_uuid=execution_uuid,
                    failed_stage=failed_stage,
                    exc=exc,
                    started_receipt_basename_present=(
                        STARTED_RECEIPT_FILENAME in inventory
                    ),
                    started_receipt_sha256=(
                        _sha256_bytes(started_bytes) if started_committed else None
                    ),
                    root_inventory_at_failure=inventory,
                )
                _write_exclusive_at(
                    directory_fd, FAILURE_RECEIPT_FILENAME, _canonical_bytes(failure)
                )
            except BaseException:
                pass
        raise
    finally:
        with suppress(BaseException):
            if timer_armed:
                signal.setitimer(signal.ITIMER_REAL, 0)
                timer_armed = False
        signal_restore_error: BaseException | None = None
        try:
            _restore_controller_signal_handlers(previous_signal_handlers)
        except BaseException as restore_exc:
            signal_restore_error = restore_exc
        if builder_channels is not None:
            builder_channels.close()
        if verifier_channels is not None:
            verifier_channels.close()
        for descriptor in (*builder_forward, *verifier_forward):
            with suppress(OSError):
                os.close(descriptor)
        if root_snapshot is not None:
            root_snapshot.close()
        if report_fd >= 0:
            with suppress(OSError):
                os.close(report_fd)
        _close_held_files(files)
        if directory_fd >= 0:
            with suppress(OSError):
                os.close(directory_fd)
        try:
            _restore_controller_signal_mask(inherited_signal_mask)
        finally:
            if signal_restore_error is not None:
                raise signal_restore_error


def execute_production_nested_opening_feasibility_v1(
    *,
    upstream_census_plan_path: Path,
    nested_plan_path: Path,
    freeze_request_path: Path,
    registration_receipt_path: Path,
    expected_upstream_census_plan_digest: str,
    expected_upstream_census_plan_bytes_sha256: str,
    expected_nested_plan_digest: str,
    expected_nested_plan_bytes_sha256: str,
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
) -> ExecutedNestedOpeningFeasibilityArtifactsV1:
    """Refuse production until the transitive reconstruction contract is audited."""

    raise NestedOpeningFeasibilityExecutionV1Error(
        PRODUCTION_EXECUTION_REFUSAL_CODE
    )


def execute_engineering_nested_opening_feasibility_fixture_for_testing_v1(
    *,
    upstream_census_plan_path: Path,
    nested_plan_path: Path,
    freeze_request_path: Path,
    registration_receipt_path: Path,
    expected_upstream_census_plan_digest: str,
    expected_upstream_census_plan_bytes_sha256: str,
    expected_nested_plan_digest: str,
    expected_nested_plan_bytes_sha256: str,
    expected_registration_receipt_sha256: str,
    expected_registration_reference: str,
    expected_execution_uuid: str,
    expected_execution_nonce: str,
    output_root: Path,
    runner_source_path: Path,
    deadline_seconds: int = 600,
) -> ExecutedNestedOpeningFeasibilityArtifactsV1:
    """Execute a reduced fixture; unavailable from the production CLI."""

    entry_started_monotonic_ns = time.monotonic_ns()
    entry_started_wall_utc = _utc_now()
    _require_safe_bootstrap_runtime()
    return _execute(
        upstream_census_plan_path=upstream_census_plan_path,
        nested_plan_path=nested_plan_path,
        freeze_request_path=freeze_request_path,
        registration_receipt_path=registration_receipt_path,
        expected_upstream_census_plan_digest=expected_upstream_census_plan_digest,
        expected_upstream_census_plan_bytes_sha256=(
            expected_upstream_census_plan_bytes_sha256
        ),
        expected_nested_plan_digest=expected_nested_plan_digest,
        expected_nested_plan_bytes_sha256=expected_nested_plan_bytes_sha256,
        expected_registration_receipt_sha256=(
            expected_registration_receipt_sha256
        ),
        expected_registration_reference=expected_registration_reference,
        expected_execution_uuid=expected_execution_uuid,
        expected_execution_nonce=expected_execution_nonce,
        output_root=output_root,
        runner_source_path=runner_source_path,
        deadline_seconds=deadline_seconds,
        controller_argv=tuple(sys.argv),
        require_production=False,
        source_archive_path=None,
        constraints_path=None,
        environment_lock_path=None,
        entry_started_monotonic_ns=entry_started_monotonic_ns,
        entry_started_wall_utc=entry_started_wall_utc,
    )


def _strict_started_receipt(
    payload: bytes,
    *,
    expected_execution_uuid: str,
    expected_execution_nonce: str,
    expected_input_binding: Mapping[str, Any],
    expected_registered_at_utc: str,
    expected_registered_deadline_seconds: int,
) -> Mapping[str, Any]:
    try:
        value = _load_json(payload.decode("ascii"))
    except UnicodeDecodeError as exc:
        raise NestedOpeningFeasibilityExecutionV1Error(
            "started receipt is not ASCII"
        ) from exc
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
            "execution_scope",
            "reconstruction_claim_boundary",
            "authorization",
            "started_receipt_digest",
        ),
        name="started receipt",
    )
    if (
        _require_integer(obj["schema_version"], name="started schema version")
        != NESTED_OPENING_FEASIBILITY_EXECUTION_SCHEMA_VERSION
        or obj["receipt_kind"] != _STARTED_KIND
        or obj["status"] != "started_no_retry"
        or _require_uuid(obj["execution_uuid"]) != expected_execution_uuid
        or _require_nonce(obj["execution_nonce"]) != expected_execution_nonce
        or not _exact_json_equal(obj["input_binding"], expected_input_binding)
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "started receipt differs from the registered execution"
        )
    _require_exact_constant(
        obj["execution_policy"], _EXECUTION_POLICY, name="started execution policy"
    )
    _require_exact_constant(
        obj["execution_scope"],
        _ENGINEERING_EXECUTION_SCOPE,
        name="started execution scope",
    )
    _require_exact_constant(
        obj["reconstruction_claim_boundary"],
        _RECONSTRUCTION_CLAIM_BOUNDARY,
        name="started reconstruction boundary",
    )
    _require_exact_constant(obj["authorization"], _AUTHORIZATION, name="authorization")
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
        _require_timestamp(wall["registered_at_utc"])
        != _require_timestamp(expected_registered_at_utc)
        or _registration_datetime(wall["registered_at_utc"])
        > _wall_datetime(wall["controller_started_at_utc"], name="controller start")
        or wall["registered_not_after_controller_start_by_supplied_and_local_clocks"]
        is not True
        or wall["clock_or_service_authenticity_independently_verified"] is not False
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "started receipt wall-clock boundary is inconsistent"
        )
    identity = _require_mapping(
        obj["controller_process_identity"],
        ("pid", "argv", "working_directory", "host", "python", "worker_environment"),
        name="started controller identity",
    )
    _require_integer(identity["pid"], name="controller pid", minimum=1)
    if (
        type(identity["argv"]) is not list
        or not identity["argv"]
        or any(type(item) is not str for item in identity["argv"])
        or type(identity["working_directory"]) is not str
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "started controller argv or working directory is invalid"
        )
    _require_exact_constant(identity["host"], _host_identity(), name="controller host")
    _require_exact_constant(identity["python"], _python_identity(), name="controller Python")
    _require_exact_constant(
        identity["worker_environment"],
        _safe_environment(),
        name="controller worker environment",
    )
    deadline = _require_mapping(
        obj["deadline"],
        (
            "deadline_seconds",
            "started_monotonic_ns",
            "deadline_monotonic_ns",
            "covers_held_input_acquisition_and_hashing",
        ),
        name="started deadline",
    )
    seconds = _require_integer(
        deadline["deadline_seconds"], name="execution deadline seconds", minimum=1
    )
    registered_seconds = _require_integer(
        expected_registered_deadline_seconds,
        name="registered execution deadline seconds",
        minimum=1,
    )
    began = _require_integer(
        deadline["started_monotonic_ns"], name="execution start monotonic", minimum=1
    )
    if (
        seconds != registered_seconds
        or _require_integer(
            deadline["deadline_monotonic_ns"], name="execution deadline monotonic", minimum=1
        )
        != began + seconds * 1_000_000_000
        or deadline["covers_held_input_acquisition_and_hashing"] is not True
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "started deadline is inconsistent"
        )
    commands = _require_mapping(
        obj["worker_commands"],
        (
            "builder_argv",
            "verifier_argv_prefix",
            "builder_watchdog_argv",
            "verifier_watchdog_argv_prefix",
        ),
        name="started worker commands",
    )
    for key, item in commands.items():
        if type(item) is not list or any(type(part) is not str for part in item):
            raise NestedOpeningFeasibilityExecutionV1Error(
                f"started worker command {key} is invalid"
            )
    unsigned = {key: item for key, item in obj.items() if key != "started_receipt_digest"}
    if _require_sha256(
        obj["started_receipt_digest"], name="started receipt digest"
    ) != _digest(unsigned, domain=_STARTED_DOMAIN) or _canonical_bytes(obj) != payload:
        raise NestedOpeningFeasibilityExecutionV1Error(
            "started receipt digest or canonical bytes are inconsistent"
        )
    return obj


def _strict_execution_receipt(
    payload: bytes,
    *,
    expected_sha256: str,
    expected_digest: str,
    expected_execution_uuid: str,
    expected_execution_nonce: str,
    expected_input_binding: Mapping[str, Any],
    root_payloads: Mapping[str, bytes],
    builder_watchdog: Mapping[str, Any],
    builder_terminal: Mapping[str, Any],
    verifier_watchdog: Mapping[str, Any],
    verifier_terminal: Mapping[str, Any],
    started: Mapping[str, Any],
    registration: ExternalNestedOpeningFeasibilityRegistrationReceiptV1,
    expected_registration_reference: str,
    expected_registration_receipt_sha256: str,
) -> Mapping[str, Any]:
    expected_sha = _require_sha256(
        expected_sha256, name="expected execution-receipt sha256"
    )
    if _sha256_bytes(payload) != expected_sha:
        raise NestedOpeningFeasibilityExecutionV1Error(
            "execution-receipt bytes differ from external pin"
        )
    try:
        value = _load_json(payload.decode("ascii"))
    except UnicodeDecodeError as exc:
        raise NestedOpeningFeasibilityExecutionV1Error(
            "execution receipt is not ASCII"
        ) from exc
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
            "execution_scope",
            "scientific_claim_boundary",
            "reconstruction_claim_boundary",
            "authorization",
            "completion",
            "execution_receipt_digest",
        ),
        name="execution receipt",
    )
    if (
        _require_integer(obj["schema_version"], name="execution schema version")
        != NESTED_OPENING_FEASIBILITY_EXECUTION_SCHEMA_VERSION
        or obj["receipt_kind"] != _EXECUTION_RECEIPT_KIND
        or obj["status"] != "complete_verified_nonauthorizing"
        or _require_uuid(obj["execution_uuid"]) != expected_execution_uuid
        or _require_nonce(obj["execution_nonce"]) != expected_execution_nonce
        or not _exact_json_equal(obj["input_binding"], expected_input_binding)
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "execution receipt identity or input binding is inconsistent"
        )
    _require_exact_constant(obj["execution_policy"], _EXECUTION_POLICY, name="execution policy")
    _require_exact_constant(
        obj["execution_scope"],
        _ENGINEERING_EXECUTION_SCOPE,
        name="execution scope",
    )
    _require_exact_constant(
        obj["scientific_claim_boundary"],
        _EXECUTED_SCIENTIFIC_CLAIM_BOUNDARY,
        name="executed scientific boundary",
    )
    _require_exact_constant(
        obj["reconstruction_claim_boundary"],
        _RECONSTRUCTION_CLAIM_BOUNDARY,
        name="execution reconstruction boundary",
    )
    _require_exact_constant(obj["authorization"], _AUTHORIZATION, name="authorization")
    expected_stages = {
        "started": {
            "filename": STARTED_RECEIPT_FILENAME,
            "sha256": _sha256_bytes(root_payloads[STARTED_RECEIPT_FILENAME]),
            "byte_count": len(root_payloads[STARTED_RECEIPT_FILENAME]),
        },
        "builder_guardian_terminal": {
            "filename": BUILDER_GUARDIAN_TERMINAL_FILENAME,
            "sha256": _sha256_bytes(root_payloads[BUILDER_GUARDIAN_TERMINAL_FILENAME]),
            "byte_count": len(root_payloads[BUILDER_GUARDIAN_TERMINAL_FILENAME]),
            "watchdog_terminal_digest": builder_watchdog["watchdog_terminal_digest"],
        },
        "builder_terminal": {
            "filename": BUILDER_TERMINAL_FILENAME,
            "sha256": _sha256_bytes(root_payloads[BUILDER_TERMINAL_FILENAME]),
            "byte_count": len(root_payloads[BUILDER_TERMINAL_FILENAME]),
            "worker_terminal_digest": builder_terminal["worker_terminal_digest"],
        },
        "verifier_guardian_terminal": {
            "filename": VERIFIER_GUARDIAN_TERMINAL_FILENAME,
            "sha256": _sha256_bytes(root_payloads[VERIFIER_GUARDIAN_TERMINAL_FILENAME]),
            "byte_count": len(root_payloads[VERIFIER_GUARDIAN_TERMINAL_FILENAME]),
            "watchdog_terminal_digest": verifier_watchdog["watchdog_terminal_digest"],
        },
        "verifier_terminal": {
            "filename": VERIFIER_TERMINAL_FILENAME,
            "sha256": _sha256_bytes(root_payloads[VERIFIER_TERMINAL_FILENAME]),
            "byte_count": len(root_payloads[VERIFIER_TERMINAL_FILENAME]),
            "worker_terminal_digest": verifier_terminal["worker_terminal_digest"],
        },
    }
    if not _exact_json_equal(obj["stage_receipts"], expected_stages):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "execution receipt stage bindings differ from exact root bytes"
        )
    report_bytes = root_payloads[REPORT_FILENAME]
    expected_report_binding = {
        "filename": REPORT_FILENAME,
        "observed_report_digest": builder_terminal["report_binding"]["observed_report_digest"],
        "exact_report_bytes_sha256": _sha256_bytes(report_bytes),
        "exact_report_byte_count": len(report_bytes),
        "committed_after_fresh_process_verification": True,
    }
    expected_worker_report_binding = {
        "observed_report_digest": expected_report_binding[
            "observed_report_digest"
        ],
        "exact_report_bytes_sha256": expected_report_binding[
            "exact_report_bytes_sha256"
        ],
        "exact_report_byte_count": expected_report_binding[
            "exact_report_byte_count"
        ],
    }
    if (
        not _exact_json_equal(obj["report_binding"], expected_report_binding)
        or not _exact_json_equal(obj["accounting"], builder_terminal["accounting"])
        or not _exact_json_equal(
            builder_terminal["report_binding"], expected_worker_report_binding
        )
        or not _exact_json_equal(
            verifier_terminal["report_binding"], expected_worker_report_binding
        )
        or not _exact_json_equal(builder_terminal["accounting"], verifier_terminal["accounting"])
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "execution receipt report/accounting differs from both fresh stages"
        )
    registration_validation = _require_mapping(
        obj["external_registration_validation"],
        (
            "expected_registration_reference",
            "expected_registration_receipt_sha256",
            "registered_at_utc",
            "controller_started_at_utc",
            "registered_execution_deadline_seconds",
            "registered_not_after_controller_start_by_supplied_and_local_clocks",
            "external_origin_proven_by_expected_pin",
            *_REGISTRATION_CLAIM_BOUNDARY,
            *_RECONSTRUCTION_CLAIM_BOUNDARY,
        ),
        name="execution registration validation",
    )
    expected_registration_validation = {
        "expected_registration_reference": expected_registration_reference,
        "expected_registration_receipt_sha256": expected_registration_receipt_sha256,
        "registered_at_utc": registration.registered_at_utc,
        "controller_started_at_utc": started["wall_clock"]["controller_started_at_utc"],
        "registered_execution_deadline_seconds": registration.execution_deadline_seconds,
        "registered_not_after_controller_start_by_supplied_and_local_clocks": True,
        "external_origin_proven_by_expected_pin": False,
        **_REGISTRATION_CLAIM_BOUNDARY,
        **_RECONSTRUCTION_CLAIM_BOUNDARY,
    }
    if not _exact_json_equal(
        registration_validation, expected_registration_validation
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "execution receipt registration validation differs from held registrar bytes"
        )
    completion = _require_mapping(
        obj["completion"],
        (
            "started_receipt_written_first",
            "fresh_builder_succeeded",
            "fresh_verifier_succeeded",
            "report_committed_before_completion_receipt",
            "completion_receipt_written_last",
            "failure_receipt_present",
        ),
        name="execution completion",
    )
    if any(completion[key] is not True for key in tuple(completion)[:-1]) or completion[
        "failure_receipt_present"
    ] is not False:
        raise NestedOpeningFeasibilityExecutionV1Error(
            "execution completion flags are inconsistent"
        )
    runtime = _require_mapping(
        obj["runtime_evidence"],
        (
            "contract",
            "builder",
            "builder_watchdog",
            "verifier",
            "verifier_watchdog",
            "controller_terminal_commit_admission",
        ),
        name="runtime evidence",
    )
    expected_runtime_prefix = {
        "builder": {
            "timing": builder_terminal["timing"],
            "maximum_resident_set_size": builder_terminal[
                "maximum_resident_set_size"
            ],
            "process_identity": builder_terminal["process_identity"],
        },
        "builder_watchdog": {
            "timing": builder_watchdog["timing"],
            "process_identity": builder_watchdog["watchdog_process_identity"],
            "containment": builder_watchdog["containment"],
        },
        "verifier": {
            "timing": verifier_terminal["timing"],
            "maximum_resident_set_size": verifier_terminal[
                "maximum_resident_set_size"
            ],
            "process_identity": verifier_terminal["process_identity"],
        },
        "verifier_watchdog": {
            "timing": verifier_watchdog["timing"],
            "process_identity": verifier_watchdog["watchdog_process_identity"],
            "containment": verifier_watchdog["containment"],
        },
    }
    if runtime["contract"] != _RUNTIME_EVIDENCE_CONTRACT or any(
        not _exact_json_equal(runtime[key], expected)
        for key, expected in expected_runtime_prefix.items()
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "execution receipt runtime evidence differs from exact stage terminals"
        )
    admission = _require_mapping(
        runtime["controller_terminal_commit_admission"],
        (
            "pid",
            "started_monotonic_ns",
            "admitted_monotonic_ns",
            "elapsed_to_admission_ns",
            "deadline_seconds",
            "deadline_monotonic_ns",
            "started_wall_utc",
            "admitted_wall_utc",
            "durable_write_completion_timestamp_recorded_in_same_receipt",
        ),
        name="controller terminal-commit admission",
    )
    started_deadline = started["deadline"]
    if _require_integer(
        started_deadline["deadline_seconds"],
        name="started registered deadline seconds",
        minimum=1,
    ) != _require_integer(
        registration.execution_deadline_seconds,
        name="external registration deadline seconds",
        minimum=1,
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "started deadline duration differs from held external registration"
        )
    admission_started = _require_integer(
        admission["started_monotonic_ns"], name="controller admission start", minimum=1
    )
    admission_ended = _require_integer(
        admission["admitted_monotonic_ns"],
        name="controller admission end",
        minimum=admission_started,
    )
    admission_seconds = _require_integer(
        admission["deadline_seconds"],
        name="controller admission deadline seconds",
        minimum=1,
    )
    admission_deadline = _require_integer(
        admission["deadline_monotonic_ns"],
        name="controller admission deadline monotonic",
        minimum=1,
    )
    if (
        type(admission["admitted_wall_utc"]) is not str
        or type(admission["started_wall_utc"]) is not str
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "controller admission wall-clock fields are not strings"
        )
    admitted_wall = cast(str, admission["admitted_wall_utc"])
    started_wall = cast(str, admission["started_wall_utc"])
    if (
        _require_integer(admission["pid"], name="controller pid", minimum=1)
        != _require_integer(
            started["controller_process_identity"]["pid"],
            name="started controller pid",
            minimum=1,
        )
        or admission_started != started_deadline["started_monotonic_ns"]
        or admission_seconds
        != _require_integer(
            started_deadline["deadline_seconds"],
            name="started deadline seconds",
            minimum=1,
        )
        or admission_deadline
        != _require_integer(
            started_deadline["deadline_monotonic_ns"],
            name="started deadline monotonic",
            minimum=1,
        )
        or admission_ended > admission_deadline
        or _require_integer(
            admission["elapsed_to_admission_ns"], name="elapsed to admission"
        )
        != admission_ended - admission_started
        or started_wall != started["wall_clock"]["controller_started_at_utc"]
        or _wall_datetime(admitted_wall, name="controller admission wall UTC")
        < _wall_datetime(started_wall, name="controller start wall UTC")
        or admission["durable_write_completion_timestamp_recorded_in_same_receipt"]
        is not False
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "controller terminal-commit admission differs from started deadline"
        )
    unsigned = {key: item for key, item in obj.items() if key != "execution_receipt_digest"}
    digest = _require_sha256(
        obj["execution_receipt_digest"], name="execution receipt digest"
    )
    if (
        digest != _require_sha256(expected_digest, name="expected execution receipt digest")
        or digest != _digest(unsigned, domain=_EXECUTION_RECEIPT_DOMAIN)
        or _canonical_bytes(obj) != payload
    ):
        raise NestedOpeningFeasibilityExecutionV1Error(
            "execution receipt digest or canonical bytes are inconsistent"
        )
    return obj


@dataclass(frozen=True, slots=True)
class VerifiedNestedOpeningFeasibilityRootV1:
    output_root: Path
    report_digest: str
    report_bytes_sha256: str
    execution_receipt_digest: str
    fresh_replay_terminal_digest: str
    fresh_replay_watchdog_terminal_digest: str
    verification_receipt: Mapping[str, Any]


def verify_production_nested_opening_feasibility_execution_root_v1(
    **_kwargs: Any,
) -> NoReturn:
    """Refuse production verification under the same reconstruction boundary."""

    raise NestedOpeningFeasibilityExecutionV1Error(
        PRODUCTION_EXECUTION_REFUSAL_CODE
    )


def verify_engineering_nested_opening_feasibility_execution_root_for_testing_v1(
    *,
    output_root: Path,
    upstream_census_plan_path: Path,
    nested_plan_path: Path,
    freeze_request_path: Path,
    registration_receipt_path: Path,
    expected_upstream_census_plan_digest: str,
    expected_upstream_census_plan_bytes_sha256: str,
    expected_nested_plan_digest: str,
    expected_nested_plan_bytes_sha256: str,
    expected_registration_receipt_sha256: str,
    expected_registration_reference: str,
    expected_execution_uuid: str,
    expected_execution_nonce: str,
    expected_execution_deadline_seconds: int,
    expected_execution_receipt_sha256: str,
    expected_execution_receipt_digest: str,
    runner_source_path: Path,
    verification_deadline_seconds: int = 600,
) -> VerifiedNestedOpeningFeasibilityRootV1:
    """Strictly replay a reduced completed root in a third fresh process."""

    verification_started_ns = time.monotonic_ns()
    verification_started_wall = _utc_now()
    _require_safe_bootstrap_runtime()
    _require_single_threaded_controller()
    _require_default_sigchld()
    execution_uuid = _require_uuid(expected_execution_uuid)
    execution_nonce = _require_nonce(expected_execution_nonce)
    execution_deadline = _require_integer(
        expected_execution_deadline_seconds,
        name="expected execution deadline seconds",
        minimum=1,
        maximum=86_400,
    )
    verification_seconds = _require_integer(
        verification_deadline_seconds,
        name="verification deadline seconds",
        minimum=1,
        maximum=86_400,
    )
    verification_deadline_ns = (
        verification_started_ns + verification_seconds * 1_000_000_000
    )
    if _lexical_absolute(output_root).name != execution_uuid:
        raise NestedOpeningFeasibilityExecutionV1Error(
            "completed-root basename differs from registered execution UUID"
        )
    files: dict[str, _HeldOrdinaryFileV1] = {}
    directory_fd = -1
    root_snapshot: _HeldExecutionRootSnapshotV1 | None = None
    report_duplicate = -1
    channels: _ParentDeathPipe | None = None
    forward: list[int] = []
    try:
        files = _open_execution_inputs(
            upstream_census_plan_path=upstream_census_plan_path,
            nested_plan_path=nested_plan_path,
            freeze_request_path=freeze_request_path,
            registration_receipt_path=registration_receipt_path,
            source_archive_path=None,
            constraints_path=None,
            environment_lock_path=None,
            require_production=False,
        )
        upstream, nested, request, registration, held_payloads = (
            _controller_replay_held_chain(
                files,
                expected_upstream_plan_digest=expected_upstream_census_plan_digest,
                expected_upstream_plan_sha256=(
                    expected_upstream_census_plan_bytes_sha256
                ),
                expected_nested_plan_digest=expected_nested_plan_digest,
                expected_nested_plan_sha256=expected_nested_plan_bytes_sha256,
                expected_registration_receipt_sha256=(
                    expected_registration_receipt_sha256
                ),
                expected_registration_reference=expected_registration_reference,
                expected_execution_uuid=execution_uuid,
                expected_execution_nonce=execution_nonce,
                expected_execution_deadline_seconds=execution_deadline,
                runner_source_path=runner_source_path,
                controller_started_wall_utc=verification_started_wall,
                require_production=False,
            )
        )
        root = _lexical_absolute(output_root)
        directory_fd, _ = _open_directory_chain_nofollow(root)
        if (os.fstat(directory_fd).st_dev, os.fstat(directory_fd).st_ino) in {
            (held.file_dev, held.file_ino) for held in files.values()
        }:
            raise NestedOpeningFeasibilityExecutionV1Error(
                "completed root aliases a frozen input inode"
            )
        root_snapshot = _HeldExecutionRootSnapshotV1.acquire_existing_complete_root(
            root,
            directory_fd,
            input_inodes=frozenset(
                (held.file_dev, held.file_ino) for held in files.values()
            ),
        )
        root_snapshot.require_declared_terminal_order_metadata()
        root_payloads = root_snapshot.expected_payloads
        input_binding = {
            "upstream_evaluation_census_plan": _upstream_plan_binding(
                upstream, held_payloads["upstream_census_plan"]
            ),
            "nested_opening_feasibility_plan": _nested_plan_binding(
                nested,
                expected_nested_plan_digest,
                held_payloads["nested_plan"],
            ),
            "freeze_request": {
                "freeze_request_digest": request._digest_unverified(),
                "canonical_freeze_request_bytes_sha256": _sha256_bytes(
                    held_payloads["freeze_request"]
                ),
                "canonical_freeze_request_byte_count": len(
                    held_payloads["freeze_request"]
                ),
            },
            "external_registration_receipt": {
                "registration_reference": registration.registration_reference,
                "registration_receipt_digest": registration._digest_unverified(),
                "exact_registration_receipt_bytes_sha256": (
                    expected_registration_receipt_sha256
                ),
                "exact_registration_receipt_byte_count": len(
                    held_payloads["registration_receipt"]
                ),
            },
            "execution_code_binding": dict(request.execution_code_binding),
            "environment_input_bindings": dict(request.environment_input_bindings),
        }
        started = _strict_started_receipt(
            root_payloads[STARTED_RECEIPT_FILENAME],
            expected_execution_uuid=execution_uuid,
            expected_execution_nonce=execution_nonce,
            expected_input_binding=input_binding,
            expected_registered_at_utc=registration.registered_at_utc,
            expected_registered_deadline_seconds=(
                registration.execution_deadline_seconds
            ),
        )
        plan_bindings = {
            "upstream_evaluation_census_plan": _upstream_plan_binding(
                upstream, held_payloads["upstream_census_plan"]
            ),
            "nested_opening_feasibility_plan": {
                "prospective_plan_digest": expected_nested_plan_digest,
                "canonical_plan_bytes_sha256": (
                    expected_nested_plan_bytes_sha256
                ),
                "canonical_plan_byte_count": len(held_payloads["nested_plan"]),
            },
        }
        held_bindings = _held_input_bindings(files)

        def parse_stage_watchdog(
            watchdog_filename: str,
            worker_filename: str,
            *,
            kind: str,
        ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
            try:
                watchdog_value = _load_json(
                    root_payloads[watchdog_filename].decode("ascii")
                )
                worker_value = _load_json(root_payloads[worker_filename].decode("ascii"))
            except UnicodeDecodeError as exc:
                raise NestedOpeningFeasibilityExecutionV1Error(
                    "persisted terminal is not ASCII"
                ) from exc
            if type(watchdog_value) is not dict or type(worker_value) is not dict:
                raise NestedOpeningFeasibilityExecutionV1Error(
                    "persisted terminal is not an object"
                )
            watchdog_obj = cast(Mapping[str, Any], watchdog_value)
            watchdog_process = _require_mapping(
                watchdog_obj.get("watchdog_process_identity"),
                ("pid", "argv", "working_directory", "host", "python", "environment"),
                name="persisted watchdog process identity",
            )
            worker_process = _require_mapping(
                watchdog_obj.get("worker_process"),
                (
                    "pid",
                    "argv",
                    "returncode",
                    "stdout_sha256",
                    "stderr_sha256",
                    "stderr_byte_count",
                    "reaped",
                ),
                name="persisted watchdog worker process",
            )
            action = (
                "__build-worker"
                if kind == _BUILDER_TERMINAL_KIND
                else "__verify-worker"
            )
            worker_argv = worker_process["argv"]
            worker_options = _exact_worker_launch_option_map(
                worker_argv, expected_action=action
            )
            watchdog_argv = _full_bootstrap_launch_from_process_argv(
                watchdog_process["argv"],
                role="--g03-bootstrap-watchdog",
                expected_action="__watchdog-worker",
            )
            watchdog_options = _exact_watchdog_launch_option_map(
                watchdog_argv,
                expected_worker_argv=cast(list[str], worker_argv),
            )
            expected_common = {
                "--held-input-bindings-json": _dump_json(held_bindings),
                "--expected-upstream-plan-digest": (
                    expected_upstream_census_plan_digest
                ),
                "--expected-upstream-plan-sha256": (
                    expected_upstream_census_plan_bytes_sha256
                ),
                "--expected-nested-plan-digest": expected_nested_plan_digest,
                "--expected-nested-plan-sha256": (
                    expected_nested_plan_bytes_sha256
                ),
                "--expected-freeze-request-digest": request._digest_unverified(),
                "--expected-registration-receipt-sha256": (
                    expected_registration_receipt_sha256
                ),
                "--expected-registration-reference": (
                    expected_registration_reference
                ),
                "--expected-execution-uuid": execution_uuid,
                "--expected-execution-nonce": execution_nonce,
                "--expected-execution-deadline-seconds": str(execution_deadline),
                "--controller-started-wall-utc": started["wall_clock"][
                    "controller_started_at_utc"
                ],
                "--deadline-monotonic-ns": str(
                    started["deadline"]["deadline_monotonic_ns"]
                ),
                "--runner-source": str(_lexical_absolute(runner_source_path)),
            }
            if any(
                worker_options[option] != expected
                for option, expected in expected_common.items()
            ):
                raise NestedOpeningFeasibilityExecutionV1Error(
                    "persisted worker launch differs from registered exact semantic pins"
                )
            if watchdog_options["--deadline-monotonic-ns"] != str(
                started["deadline"]["deadline_monotonic_ns"]
            ):
                raise NestedOpeningFeasibilityExecutionV1Error(
                    "persisted watchdog launch differs from the registered deadline"
                )
            held_fd_value = _load_json(worker_options["--held-input-fds-json"])
            if (
                type(held_fd_value) is not dict
                or tuple(held_fd_value) != tuple(held_bindings)
                or any(type(item) is not int or item < 0 for item in held_fd_value.values())
                or len(set(held_fd_value.values())) != len(held_fd_value)
                or _dump_json(held_fd_value)
                != worker_options["--held-input-fds-json"]
            ):
                raise NestedOpeningFeasibilityExecutionV1Error(
                    "persisted worker held descriptor map is not canonical"
                )
            semantic_fd_option = (
                "--output-root-fd" if action == "__build-worker" else "--report-fd"
            )
            semantic_fd_text = worker_options[semantic_fd_option]
            if not semantic_fd_text.isdigit():
                raise NestedOpeningFeasibilityExecutionV1Error(
                    "persisted worker semantic descriptor is not canonical"
                )
            expected_forward = [
                *cast(dict[str, int], held_fd_value).values(),
                int(semantic_fd_text),
            ]
            if watchdog_options["--forward-fds-json"] != _dump_json(expected_forward):
                raise NestedOpeningFeasibilityExecutionV1Error(
                    "persisted watchdog forwarding differs from worker capabilities"
                )
            for worker_option, watchdog_option in (
                ("--parent-guard-fd", "--worker-guard-read-fd"),
                ("--guardian-ready-write-fd", "--guardian-ready-write-fd"),
                ("--worker-lifetime-write-fd", "--worker-lifetime-write-fd"),
            ):
                if worker_options[worker_option] != watchdog_options[watchdog_option]:
                    raise NestedOpeningFeasibilityExecutionV1Error(
                        "persisted worker control capability differs from watchdog launch"
                    )
            parsed_worker = _parse_watchdog_and_worker_terminal(
                watchdog_obj,
                expected_watchdog_pid=_require_integer(
                    watchdog_process["pid"],
                    name="persisted watchdog pid",
                    minimum=1,
                ),
                expected_watchdog_argv=watchdog_argv,
                expected_worker_pid=_require_integer(
                    worker_process["pid"], name="persisted worker pid", minimum=1
                ),
                expected_worker_argv=cast(list[str], worker_argv),
                expected_worker_kind=kind,
                expected_deadline_monotonic_ns=_require_integer(
                    started["deadline"]["deadline_monotonic_ns"],
                    name="persisted execution deadline",
                    minimum=1,
                ),
                expected_plan_bindings=plan_bindings,
                expected_held_bindings=held_bindings,
                require_production=False,
            )
            if action == "__verify-worker":
                root_report = root_payloads[REPORT_FILENAME]
                worker_report = cast(Mapping[str, Any], parsed_worker["report_binding"])
                if (
                    worker_options["--expected-report-digest"]
                    != worker_report["observed_report_digest"]
                    or worker_options["--expected-report-sha256"]
                    != _sha256_bytes(root_report)
                    or worker_options["--expected-report-byte-count"]
                    != str(len(root_report))
                ):
                    raise NestedOpeningFeasibilityExecutionV1Error(
                        "persisted verifier launch differs from exact held report"
                    )
            if (
                _canonical_bytes(watchdog_obj) != root_payloads[watchdog_filename]
                or _canonical_bytes(parsed_worker) != root_payloads[worker_filename]
                or not _exact_json_equal(parsed_worker, worker_value)
            ):
                raise NestedOpeningFeasibilityExecutionV1Error(
                    "persisted watchdog/worker terminal bytes disagree"
                )
            return watchdog_obj, parsed_worker

        builder_watchdog, builder_terminal = parse_stage_watchdog(
            BUILDER_GUARDIAN_TERMINAL_FILENAME,
            BUILDER_TERMINAL_FILENAME,
            kind=_BUILDER_TERMINAL_KIND,
        )
        verifier_watchdog, verifier_terminal = parse_stage_watchdog(
            VERIFIER_GUARDIAN_TERMINAL_FILENAME,
            VERIFIER_TERMINAL_FILENAME,
            kind=_VERIFIER_TERMINAL_KIND,
        )
        commands = started["worker_commands"]
        builder_watchdog_launch = _full_bootstrap_launch_from_process_argv(
            builder_watchdog["watchdog_process_identity"]["argv"],
            role="--g03-bootstrap-watchdog",
            expected_action="__watchdog-worker",
        )
        verifier_watchdog_launch = _full_bootstrap_launch_from_process_argv(
            verifier_watchdog["watchdog_process_identity"]["argv"],
            role="--g03-bootstrap-watchdog",
            expected_action="__watchdog-worker",
        )
        exact_verifier_prefix = [
            *_safe_bootstrap_launch_prefix(role="--g03-bootstrap-worker"),
            "__verify-worker",
        ]
        exact_watchdog_prefix = [
            *_safe_bootstrap_launch_prefix(role="--g03-bootstrap-watchdog"),
            "__watchdog-worker",
        ]
        if (
            not _exact_json_equal(
                commands["builder_argv"], builder_watchdog["worker_process"]["argv"]
            )
            or not _exact_json_equal(
                commands["builder_watchdog_argv"],
                builder_watchdog_launch,
            )
            or commands["verifier_argv_prefix"] != exact_verifier_prefix
            or verifier_watchdog["worker_process"]["argv"][:8]
            != exact_verifier_prefix
            or commands["verifier_watchdog_argv_prefix"] != exact_watchdog_prefix
            or verifier_watchdog_launch[:8] != exact_watchdog_prefix
        ):
            raise NestedOpeningFeasibilityExecutionV1Error(
                "started worker commands differ from persisted exact terminals"
            )
        execution_receipt = _strict_execution_receipt(
            root_payloads[EXECUTION_RECEIPT_FILENAME],
            expected_sha256=expected_execution_receipt_sha256,
            expected_digest=expected_execution_receipt_digest,
            expected_execution_uuid=execution_uuid,
            expected_execution_nonce=execution_nonce,
            expected_input_binding=input_binding,
            root_payloads=root_payloads,
            builder_watchdog=builder_watchdog,
            builder_terminal=builder_terminal,
            verifier_watchdog=verifier_watchdog,
            verifier_terminal=verifier_terminal,
            started=started,
            registration=registration,
            expected_registration_reference=expected_registration_reference,
            expected_registration_receipt_sha256=(
                expected_registration_receipt_sha256
            ),
        )
        report_binding = cast(Mapping[str, Any], execution_receipt["report_binding"])
        report_digest = _require_sha256(
            report_binding["observed_report_digest"], name="root report digest"
        )
        report_sha = _require_sha256(
            report_binding["exact_report_bytes_sha256"], name="root report sha256"
        )
        report_count = _require_integer(
            report_binding["exact_report_byte_count"],
            name="root report byte count",
            minimum=1,
        )
        # Preserve exact Popen ownership through the third fresh replay.  A
        # concurrent host reaper or an auto-reaping SIGCHLD disposition can
        # invalidate the watchdog PID before the held WNOWAIT/reap sequence.
        _require_single_threaded_controller()
        _require_default_sigchld()
        report_fd = root_snapshot.files[REPORT_FILENAME][0]
        channels = _ParentDeathPipe.create()
        verifier_map, forward = _duplicate_worker_semantic_fds(
            files, extra_label="report", extra_fd=report_fd
        )
        report_duplicate = forward[-1]
        module_argv = _safe_bootstrap_launch_prefix(
            role="--g03-bootstrap-worker"
        )
        worker_argv = [
            *module_argv,
            "__verify-worker",
            "--held-input-fds-json",
            _dump_json(verifier_map),
            "--held-input-bindings-json",
            _dump_json(held_bindings),
            "--expected-upstream-plan-digest",
            expected_upstream_census_plan_digest,
            "--expected-upstream-plan-sha256",
            expected_upstream_census_plan_bytes_sha256,
            "--expected-nested-plan-digest",
            expected_nested_plan_digest,
            "--expected-nested-plan-sha256",
            expected_nested_plan_bytes_sha256,
            "--expected-freeze-request-digest",
            request._digest_unverified(),
            "--expected-registration-receipt-sha256",
            expected_registration_receipt_sha256,
            "--expected-registration-reference",
            expected_registration_reference,
            "--expected-execution-uuid",
            execution_uuid,
            "--expected-execution-nonce",
            execution_nonce,
            "--expected-execution-deadline-seconds",
            str(execution_deadline),
            "--controller-started-wall-utc",
            verification_started_wall,
            "--deadline-monotonic-ns",
            str(verification_deadline_ns),
            "--runner-source",
            str(_lexical_absolute(runner_source_path)),
            "--parent-guard-fd",
            str(channels.worker_read_fd),
            "--guardian-ready-write-fd",
            str(channels.guardian_ready_write_fd),
            "--worker-lifetime-write-fd",
            str(channels.worker_lifetime_write_fd),
            "--report-fd",
            str(report_duplicate),
            "--expected-report-digest",
            report_digest,
            "--expected-report-sha256",
            report_sha,
            "--expected-report-byte-count",
            str(report_count),
        ]
        watchdog_argv = _watchdog_argv(
            channels=channels,
            deadline_monotonic_ns=verification_deadline_ns,
            forward_fds=forward,
            worker_argv=worker_argv,
        )
        try:
            fresh_watchdog, fresh_watchdog_pid, fresh_worker_pid = _run_watchdog(
                watchdog_argv,
                deadline_monotonic_ns=verification_deadline_ns,
                channels=channels,
                forward_fds=forward,
            )
        finally:
            channels = None
            forward = []
            report_duplicate = -1
        fresh_terminal = _parse_watchdog_and_worker_terminal(
            fresh_watchdog,
            expected_watchdog_pid=fresh_watchdog_pid,
            expected_watchdog_argv=watchdog_argv,
            expected_worker_pid=fresh_worker_pid,
            expected_worker_argv=worker_argv,
            expected_worker_kind=_VERIFIER_TERMINAL_KIND,
            expected_deadline_monotonic_ns=verification_deadline_ns,
            expected_plan_bindings=plan_bindings,
            expected_held_bindings=held_bindings,
            require_production=False,
        )
        expected_worker_report_binding = {
            "observed_report_digest": report_digest,
            "exact_report_bytes_sha256": report_sha,
            "exact_report_byte_count": report_count,
        }
        if (
            not _exact_json_equal(
                fresh_terminal["report_binding"], expected_worker_report_binding
            )
            or not _exact_json_equal(
                fresh_terminal["accounting"], execution_receipt["accounting"]
            )
        ):
            raise NestedOpeningFeasibilityExecutionV1Error(
                "third fresh replay differs from persisted report or accounting"
            )
        for held in files.values():
            held.replay_external_basename()
        if _execution_code_binding(runner_source_path) != request.execution_code_binding:
            raise NestedOpeningFeasibilityExecutionV1Error(
                "execution code changed during complete-root verification"
            )
        root_snapshot.replay()
        post_snapshot = root_snapshot.security_snapshot_obj()
        unsigned_verification = {
            "schema_version": NESTED_OPENING_FEASIBILITY_EXECUTION_SCHEMA_VERSION,
            "receipt_kind": _ROOT_VERIFICATION_KIND,
            "status": "engineering_fixture_root_verified_with_third_fresh_replay_nonauthorizing",
            "execution_uuid": execution_uuid,
            "execution_receipt_digest": execution_receipt["execution_receipt_digest"],
            "report_binding": expected_worker_report_binding,
            "accounting": execution_receipt["accounting"],
            "root_security_snapshot": post_snapshot,
            "fresh_replay_watchdog_terminal_digest": fresh_watchdog[
                "watchdog_terminal_digest"
            ],
            "fresh_replay_terminal_digest": fresh_terminal["worker_terminal_digest"],
            "fresh_replay_completed_before_deadline": (
                time.monotonic_ns() <= verification_deadline_ns
            ),
            "execution_scope": dict(_ENGINEERING_EXECUTION_SCOPE),
            "reconstruction_claim_boundary": dict(_RECONSTRUCTION_CLAIM_BOUNDARY),
            "scientific_claim_boundary": dict(_EXECUTED_SCIENTIFIC_CLAIM_BOUNDARY),
            "authorization": dict(_AUTHORIZATION),
        }
        if unsigned_verification["fresh_replay_completed_before_deadline"] is not True:
            raise NestedOpeningFeasibilityExecutionV1Error(
                "third fresh replay exceeded verification deadline"
            )
        verification = {
            **unsigned_verification,
            "root_verification_digest": _digest(
                unsigned_verification, domain=_ROOT_VERIFICATION_DOMAIN
            ),
        }
        root_snapshot.replay()
        return VerifiedNestedOpeningFeasibilityRootV1(
            root,
            report_digest,
            report_sha,
            cast(str, execution_receipt["execution_receipt_digest"]),
            cast(str, fresh_terminal["worker_terminal_digest"]),
            cast(str, fresh_watchdog["watchdog_terminal_digest"]),
            verification,
        )
    finally:
        if channels is not None:
            channels.close()
        for descriptor in forward:
            with suppress(OSError):
                os.close(descriptor)
        if report_duplicate >= 0:
            with suppress(OSError):
                os.close(report_duplicate)
        if root_snapshot is not None:
            root_snapshot.close()
        _close_held_files(files)
        if directory_fd >= 0:
            with suppress(OSError):
                os.close(directory_fd)


def _module_worker_main(argv: Sequence[str] | None = None) -> int:
    raw = tuple(sys.argv[1:] if argv is None else argv)
    if "--require-production" in raw:
        raise NestedOpeningFeasibilityExecutionV1Error(
            PRODUCTION_EXECUTION_REFUSAL_CODE
        )
    parser = _worker_parser()
    args = parser.parse_args(raw)
    if args.worker_action == "__watchdog-worker":
        return _watchdog_worker(args)
    if args.worker_action == "__build-worker":
        return _worker_build(args)
    if args.worker_action == "__verify-worker":
        return _worker_verify(args)
    raise AssertionError("unreachable worker action")


if __name__ == "__main__":
    try:
        raise SystemExit(_module_worker_main())
    except NestedOpeningFeasibilityExecutionV1Error as exc:
        sys.stderr.write(f"error: {exc}\n")
        raise SystemExit(2) from None
