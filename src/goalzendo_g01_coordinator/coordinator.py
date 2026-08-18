"""Prospective global coordination design for the exact 120-row G01 plan.

This source checkpoint is deliberately non-executable.  Its pure schedule and
accounting-verifier helpers are available for review, but every lifecycle entry
refuses until a separately frozen B overlay/runtime/provision transaction is
implemented and this source is revised and re-audited.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import select
import signal
import stat
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

SCHEMA_VERSION = 1
WORKER_COUNT = 4
ROWS_PER_WORKER = 30
PAIRS_PER_WORKER = 15
PLANNED_ROWS = 120
PAIR_COUNT = 60
WALL_CEILING_SECONDS = 48 * 60 * 60
TERM_GRACE_SECONDS = 90

CANONICAL_PROGRAM_ROOT = Path("/workspace/status-goalzendo/g01-executions")
CANONICAL_G00F_PROGRAM_ROOT = Path("/workspace/status-goalzendo/g00f-executions")
CANONICAL_ARTIFACT_ROOT = Path("/workspace/artifacts-goalzendo/g01-known-law")
RUNTIME_OVERLAY_REFUSAL = "B_RUNTIME_OVERLAY_NOT_FROZEN"
GLOBAL_CLAIM_NAME = "g01-global-study-claim.json"
THIN_TOKEN_NAME = "g00f-g01-coordinator-input.json"
DEFAULT_BACKEND = "goalzendo.experiment:run_experiment"
RUNNER_RELATIVE = "src/goalzendo/runner.py"
RUNNER_SHA256 = "46b55ad4bdd08073e5f89ae101e0862372817b74b590f07ddb8f8c4331e5b9e1"
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
G01_CONFIG_SHA256 = "ee6d53556189a6b6e25cfbfb204325017a058c63a605653df4c175463e00ce18"
G01_CANONICAL_CONFIG_DIGEST = "f9f91978a446a6e750172e0e377bc1b738ccef64b5238992555e1afb6e6bd110"
G01_PROTOCOL_SHA256 = "428243c3da271bc2d79e16c0fde20d47076eb3837c6740ab733e278cddcdbce9"
G01_SOURCE_FINGERPRINT = "1a8146377b9a9620690025671614edb2dd20d214f4e195da3cf3528809b2c694"
G01_GUARD_SIGNATURE = "9feaae82edf801aad4bd4a5b16f633be8dfa2dbd41b29601e7b61b564c763464"
G01_TARGET_BINDING_DIGEST = "3315d20a6f9bdae3c5fdaf9567c5bce7b592890d0ebd26e10682002d816bf0c6"
G01_PLAN_ROWS_DIGEST = "f51f6b6f574295433dacec8fafe20508526036ef15dce220a1d8c24e8cd6e55f"
G01_PLAN_KEY_SET_DIGEST = "7fb1bc870c3b93d1d6a5ae6148b83b460c9ee6d246b592f67f908e8080fa8b91"
SOURCE_PATHS = (
    "src/goalzendo_g01_coordinator/__init__.py",
    "src/goalzendo_g01_coordinator/coordinator.py",
    "runs/goalzendo/run_g01_global_coordinator.py",
)

# Test roots are private, unavailable to the CLI, and honored only while a
# pytest test marker is present.  Production accepts the literals above.
_TEST_ONLY_PROGRAM_ROOT: Path | None = None
_TEST_ONLY_G00F_PROGRAM_ROOT: Path | None = None
_TEST_ONLY_ARTIFACT_ROOT: Path | None = None

_FORBIDDEN_BEFORE_RENDEZVOUS = (
    "torch",
    "transformers",
    "goalzendo.experiment",
    "goalzendo.hf",
    "goalzendo.modeling",
    "goalzendo.training",
)


class CoordinatorError(RuntimeError):
    """A global invariant failed; the one-shot study must stop."""


class _CoordinatorSignal(CoordinatorError):
    """A handled operator signal that must enter global cleanup."""


def _require_isolated_runtime() -> None:
    flags = sys.flags
    if not (
        flags.isolated
        and flags.ignore_environment
        and flags.no_user_site
        and getattr(flags, "safe_path", False)
        and flags.no_site
    ):
        raise CoordinatorError("checkpoint B must be invoked with exact isolated mode -I -S")


def _signal_as_exception(number: int, _frame: Any) -> NoReturn:
    try:
        name = signal.Signals(number).name
    except ValueError:
        name = str(number)
    raise _CoordinatorSignal(f"coordinator received {name}")


@dataclass(frozen=True)
class ScheduleRow:
    """Operational assignment of one unchanged frozen RunSpec."""

    worker_index: int
    pair_index: int
    pair_key: tuple[str, str, int]
    algorithm: str
    plan_key: str
    run_id: str
    output_path: str
    spec: Any

    def public(self) -> dict[str, Any]:
        return {
            "worker_index": self.worker_index,
            "pair_index": self.pair_index,
            "pair_key": {
                "law_family": self.pair_key[0],
                "q_p": self.pair_key[1],
                "seed": self.pair_key[2],
            },
            "algorithm": self.algorithm,
            "plan_key": self.plan_key,
            "run_id": self.run_id,
            "output_path": self.output_path,
        }


@dataclass(frozen=True)
class Schedule:
    """The exact 4x30 schedule, retaining the original RunSpec objects."""

    rows: tuple[ScheduleRow, ...]
    plan_rows_digest: str
    plan_key_set_digest: str

    @property
    def digest(self) -> str:
        return _digest(self.body())

    def body(self) -> dict[str, Any]:
        return {
            "schema": "goalzendo.g01_checkpoint_b_schedule",
            "schema_version": SCHEMA_VERSION,
            "study_id": "g01",
            "worker_count": WORKER_COUNT,
            "rows_per_worker": ROWS_PER_WORKER,
            "pairs_per_worker": PAIRS_PER_WORKER,
            "pairing": "shared_law_family_q_p_seed_then_lexical_round_robin_v1",
            "plan_rows_digest": self.plan_rows_digest,
            "plan_key_set_digest": self.plan_key_set_digest,
            "rows": [row.public() for row in self.rows],
            "original_runspecs_unchanged": True,
            "work_stealing": False,
            "retry": False,
        }

    def for_worker(self, worker_index: int) -> tuple[ScheduleRow, ...]:
        return tuple(row for row in self.rows if row.worker_index == worker_index)


@dataclass
class _Reservation:
    """Held exact run directory created after the global claim."""

    plan_key: str
    path: Path
    descriptor: int


@dataclass(frozen=True)
class _ClaimTransaction:
    deadline_ns: int
    claim: Mapping[str, Any]
    root_descriptor: int
    execution_descriptor: int
    rows_descriptor: int


@dataclass(frozen=True)
class _ThinToken:
    token_digest: str
    bridge_source_digest: str
    target_binding_digest: str
    plan_rows_digest: str
    plan_key_set_digest: str
    execution_uuid: str
    route: str


_RUN_RELATIVE_COMPONENTS = 3


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _pretty(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    normalized = str(value)
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise CoordinatorError(f"{label} must be one lowercase SHA-256 digest")
    return normalized


def _is_exact_int(value: Any, expected: int) -> bool:
    """Reject bool/float JSON values that compare equal to an integer."""

    return type(value) is int and value == expected


def _strict_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    def reject(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CoordinatorError(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject,
            parse_constant=lambda constant: _raise(f"non-finite JSON constant {constant}"),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise CoordinatorError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise CoordinatorError(f"{label} must contain one JSON object")
    return value


def _token_from_snapshot(
    payload: bytes, *, expected_sha256: str, expected_bridge_source_digest: str
) -> _ThinToken:
    """Strictly project only the captured checkpoint-A thin-token identities."""

    if _sha256_bytes(payload) != expected_sha256:
        raise CoordinatorError("captured thin-token SHA-256 changed")
    token = _strict_json_bytes(payload, "captured checkpoint-A thin token")
    if payload != _pretty(token):
        raise CoordinatorError("captured checkpoint-A thin token is not canonical JSON")
    body = {key: value for key, value in token.items() if key != "token_digest"}
    exact_body_keys = {
        "bridge_source_digest",
        "eligibility",
        "eligibility_sidecar",
        "g00f_execution",
        "g01_identity",
        "produced_at_utc",
        "route_lock",
        "schema",
        "schema_version",
        "study_id",
    }
    eligibility = token.get("eligibility")
    execution = token.get("g00f_execution")
    sidecar = token.get("eligibility_sidecar")
    route_lock = token.get("route_lock")
    identity = token.get("g01_identity")
    expected_identity = {
        "config_file_sha256": G01_CONFIG_SHA256,
        "canonical_config_digest": G01_CANONICAL_CONFIG_DIGEST,
        "protocol_file_sha256": G01_PROTOCOL_SHA256,
        "runner_file_sha256": RUNNER_SHA256,
        "source_fingerprint": G01_SOURCE_FINGERPRINT,
        "guard_signature": G01_GUARD_SIGNATURE,
        "target_binding_digest": G01_TARGET_BINDING_DIGEST,
        "plan_rows_digest": G01_PLAN_ROWS_DIGEST,
        "plan_key_set_digest": G01_PLAN_KEY_SET_DIGEST,
        "planned_runs": PLANNED_ROWS,
        "cell_count": 12,
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
    }
    expected_eligibility = {
        "g01_scientifically_eligible": True,
        "direct_g01_launch_authorized": False,
        "dedicated_global_coordinator_required": True,
        "scope": "exact_unchanged_g01_primary_120_run_plan_only",
    }
    if (
        set(body) != exact_body_keys
        or token.get("schema") != "goalzendo.g00f_g01_coordinator_input"
        or not _is_exact_int(token.get("schema_version"), 1)
        or token.get("study_id") != "g01"
        or token.get("token_digest") != _digest(body)
        or token.get("bridge_source_digest") != expected_bridge_source_digest
        or eligibility != expected_eligibility
        or identity != expected_identity
        or not _is_exact_int(identity.get("planned_runs") if isinstance(identity, Mapping) else None, 120)
        or not _is_exact_int(identity.get("cell_count") if isinstance(identity, Mapping) else None, 12)
        or not isinstance(execution, Mapping)
        or set(execution) != {"execution_uuid", "route"}
        or execution.get("route") not in {"h100", "h200"}
        or not isinstance(sidecar, Mapping)
        or set(sidecar) != {"eligibility_digest", "file_sha256"}
        or not isinstance(route_lock, Mapping)
        or set(route_lock) != {"file_sha256", "route_lock_digest"}
    ):
        raise CoordinatorError("captured thin-token schema or false-launch boundary changed")
    for container, names in (
        (sidecar, ("eligibility_digest", "file_sha256")),
        (route_lock, ("file_sha256", "route_lock_digest")),
    ):
        for name in names:
            _require_sha256(container[name], f"thin-token {name}")
    produced = token.get("produced_at_utc")
    if (
        not isinstance(produced, str)
        or len(produced) != 20
        or produced[10] != "T"
        or not produced.endswith("Z")
    ):
        raise CoordinatorError("thin-token production timestamp is not second-resolution UTC")
    return _ThinToken(
        token_digest=_require_sha256(token["token_digest"], "thin-token semantic digest"),
        bridge_source_digest=expected_bridge_source_digest,
        target_binding_digest=G01_TARGET_BINDING_DIGEST,
        plan_rows_digest=G01_PLAN_ROWS_DIGEST,
        plan_key_set_digest=G01_PLAN_KEY_SET_DIGEST,
        execution_uuid=_uuid4(execution["execution_uuid"], "checkpoint-A execution UUID"),
        route=str(execution["route"]),
    )


def _require_token_repo_identity(token: _ThinToken, repo: Path) -> None:
    if repo.name != "frozen-source" or token.execution_uuid != repo.parent.name:
        raise CoordinatorError("thin-token execution UUID differs from its canonical frozen source")


def _raise(message: str) -> NoReturn:
    raise CoordinatorError(message)


def _test_roots_active() -> bool:
    return "PYTEST_CURRENT_TEST" in os.environ


def _program_root() -> Path:
    if _TEST_ONLY_PROGRAM_ROOT is not None:
        if not _test_roots_active():
            raise CoordinatorError("test-only checkpoint-B root is forbidden outside pytest")
        return Path(_TEST_ONLY_PROGRAM_ROOT).absolute()
    return CANONICAL_PROGRAM_ROOT


def _g00f_program_root() -> Path:
    if _TEST_ONLY_G00F_PROGRAM_ROOT is not None:
        if not _test_roots_active():
            raise CoordinatorError("test-only checkpoint-A root is forbidden outside pytest")
        return Path(_TEST_ONLY_G00F_PROGRAM_ROOT).absolute()
    return CANONICAL_G00F_PROGRAM_ROOT


def _artifact_root() -> Path:
    if _TEST_ONLY_ARTIFACT_ROOT is not None:
        if not _test_roots_active():
            raise CoordinatorError("test-only artifact root is forbidden outside pytest")
        return Path(_TEST_ONLY_ARTIFACT_ROOT).absolute()
    return CANONICAL_ARTIFACT_ROOT


def _logical_absolute(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else Path.cwd() / value


def _require_no_link_components(path: Path, label: str, *, allow_absent_leaf: bool = False) -> None:
    absolute = _logical_absolute(path)
    if not absolute.is_absolute() or ".." in absolute.parts:
        raise CoordinatorError(f"{label} is not a safe absolute path")
    current = Path(absolute.anchor)
    for index, part in enumerate(absolute.parts[1:]):
        current /= part
        leaf = index == len(absolute.parts[1:]) - 1
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            if leaf and allow_absent_leaf:
                return
            if allow_absent_leaf:
                return
            raise CoordinatorError(f"{label} component is absent: {current}") from None
        if stat.S_ISLNK(metadata.st_mode):
            raise CoordinatorError(f"{label} contains symlink indirection: {current}")


def _open_directory_chain(path: Path, label: str) -> int:
    absolute = _logical_absolute(path)
    if not absolute.is_absolute() or ".." in absolute.parts:
        raise CoordinatorError(f"{label} is not a safe absolute directory")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    current: int | None = None
    try:
        current = os.open(absolute.anchor, flags)
        for part in absolute.parts[1:]:
            child = os.open(part, flags, dir_fd=current)
            os.close(current)
            current = child
        return current
    except OSError as error:
        if current is not None:
            os.close(current)
        raise CoordinatorError(f"{label} cannot be opened component-wise without links") from error


def _mkdir_at(parent: int, name: str, *, mode: int = 0o755) -> int:
    if not name or Path(name).name != name or name in {".", ".."}:
        raise CoordinatorError("directory leaf is unsafe")
    try:
        os.mkdir(name, mode=mode, dir_fd=parent)
    except FileExistsError as error:
        raise CoordinatorError(f"refusing to reuse directory leaf: {name}") from error
    os.fsync(parent)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        child = os.open(name, flags, dir_fd=parent)
    except OSError as error:
        raise CoordinatorError(f"new directory leaf cannot be held: {name}") from error
    metadata = os.fstat(child)
    if stat.S_IMODE(metadata.st_mode) != mode:
        os.close(child)
        raise CoordinatorError(f"new directory leaf has unexpected mode: {name}")
    return child


def _ensure_directory_at(parent: int, name: str, *, mode: int = 0o755) -> int:
    if not name or Path(name).name != name or name in {".", ".."}:
        raise CoordinatorError("directory leaf is unsafe")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        return os.open(name, flags, dir_fd=parent)
    except FileNotFoundError:
        return _mkdir_at(parent, name, mode=mode)
    except OSError as error:
        raise CoordinatorError(f"directory leaf cannot be opened without links: {name}") from error


def _create_program_root() -> int:
    root = _program_root()
    parent = _open_directory_chain(root.parent, "checkpoint-B program parent")
    try:
        child = _ensure_directory_at(parent, root.name)
        os.fsync(parent)
        return child
    finally:
        os.close(parent)


def _require_regular(path: Path, label: str, *, mode: int | None = None) -> bytes:
    parent = _open_directory_chain(path.parent, f"{label} parent")
    descriptor: int | None = None
    try:
        descriptor = os.open(path.name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise CoordinatorError(f"{label} is not a single-link regular file")
        if mode is not None and stat.S_IMODE(metadata.st_mode) != mode:
            raise CoordinatorError(f"{label} mode must be {mode:04o}")
        with os.fdopen(os.dup(descriptor), "rb") as handle:
            payload = handle.read()
        observed = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        if (metadata.st_dev, metadata.st_ino) != (observed.st_dev, observed.st_ino):
            raise CoordinatorError(f"{label} changed while it was authenticated")
        return payload
    except OSError as error:
        raise CoordinatorError(f"{label} cannot be opened without link traversal") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent)


def _require_regular_at(parent: int, name: str, label: str, *, mode: int | None = None) -> bytes:
    descriptor: int | None = None
    try:
        descriptor = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise CoordinatorError(f"{label} is not a single-link regular file")
        if mode is not None and stat.S_IMODE(metadata.st_mode) != mode:
            raise CoordinatorError(f"{label} mode must be {mode:04o}")
        payload = os.pread(descriptor, metadata.st_size, 0)
        observed = os.stat(name, dir_fd=parent, follow_symlinks=False)
        if (metadata.st_dev, metadata.st_ino) != (observed.st_dev, observed.st_ino):
            raise CoordinatorError(f"{label} changed while it was authenticated")
        return payload
    except OSError as error:
        raise CoordinatorError(f"{label} cannot be opened without link traversal") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _repo_root(repo: str | Path) -> Path:
    resolved = _logical_absolute(repo)
    _require_no_link_components(resolved, "frozen source")
    try:
        _require_regular(resolved / "pyproject.toml", "repository pyproject")
        goalzendo_fd = _open_directory_chain(resolved / "src/goalzendo", "GoalZendo source package")
        os.close(goalzendo_fd)
    except CoordinatorError as error:
        raise CoordinatorError("repo is not one GoalZendo frozen-source tree") from error
    if not _test_roots_active():
        try:
            relative = resolved.relative_to(CANONICAL_G00F_PROGRAM_ROOT)
        except ValueError as error:
            raise CoordinatorError("repo is outside the canonical checkpoint-A program root") from error
        if len(relative.parts) != 2 or relative.parts[1] != "frozen-source":
            raise CoordinatorError("repo is not an exact checkpoint-A UUID frozen-source path")
        _uuid4(relative.parts[0], "checkpoint-A execution UUID")
    return resolved


def calculate_source_binding(repo: str | Path) -> dict[str, Any]:
    """Return the externally pinnable checkpoint-B source binding."""

    resolved = _repo_root(repo)
    files: dict[str, str] = {}
    for relative in SOURCE_PATHS:
        target = resolved / relative
        files[relative] = _sha256_bytes(_require_regular(target, f"coordinator source {relative}"))
    return {"source_files": files, "source_digest": _digest(files)}


def _require_live_module(repo: Path, module_name: str, relative: str) -> None:
    module = sys.modules.get(module_name)
    if module is None:
        raise CoordinatorError(f"required live module is absent: {module_name}")
    expected = repo / relative
    observed = _logical_absolute(str(getattr(module, "__file__", "")))
    spec = getattr(module, "__spec__", None)
    origin = _logical_absolute(str(getattr(spec, "origin", "")))
    if observed != expected or origin != expected:
        raise CoordinatorError(f"live module is not from the frozen source: {module_name}")
    expected_payload = _require_regular(expected, f"frozen module {module_name}")
    if _require_regular(observed, f"live module {module_name}") != expected_payload:
        raise CoordinatorError(f"live module bytes differ from frozen source: {module_name}")


def _verify_entrypoint(repo: Path, observed: str | Path, expected_source_digest: str) -> dict[str, Any]:
    source = calculate_source_binding(repo)
    expected = _require_sha256(expected_source_digest, "expected coordinator source digest")
    entrypoint = repo / SOURCE_PATHS[-1]
    observed_path = _logical_absolute(observed)
    if source["source_digest"] != expected or observed_path != entrypoint or observed_path.is_symlink():
        raise CoordinatorError("running entrypoint/source differs from the externally pinned coordinator")
    if _require_regular(observed_path, "coordinator entrypoint") != _require_regular(
        entrypoint, "frozen coordinator entrypoint"
    ):
        raise CoordinatorError("running entrypoint bytes differ from frozen source")
    _require_live_module(repo, "goalzendo_g01_coordinator", SOURCE_PATHS[0])
    _require_live_module(repo, "goalzendo_g01_coordinator.coordinator", SOURCE_PATHS[1])
    return source


def _assert_pre_rendezvous_import_boundary() -> None:
    present = sorted(
        name
        for name in sys.modules
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in _FORBIDDEN_BEFORE_RENDEZVOUS)
    )
    if present:
        raise CoordinatorError(
            "backend/model import occurred before global rendezvous: " + ", ".join(present)
        )


def _uuid4(value: Any, label: str) -> str:
    try:
        parsed = uuid.UUID(str(value))
    except (ValueError, TypeError, AttributeError) as error:
        raise CoordinatorError(f"{label} must be a canonical UUIDv4") from error
    if parsed.version != 4 or str(parsed) != str(value):
        raise CoordinatorError(f"{label} must be a canonical UUIDv4")
    return str(parsed)


def _exclusive_bytes(path: Path, payload: bytes, *, mode: int = 0o400) -> None:
    parent = _open_directory_chain(path.parent, f"{path.name} parent")
    try:
        _exclusive_bytes_at(parent, path.name, payload, mode=mode)
    finally:
        os.close(parent)


def _exclusive_bytes_at(parent: int, name: str, payload: bytes, *, mode: int = 0o400) -> None:
    if not name or Path(name).name != name or name in {".", ".."}:
        raise CoordinatorError("append-only receipt leaf is unsafe")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, 0o600, dir_fd=parent)
    except FileExistsError as error:
        raise CoordinatorError(f"refusing to reuse append-only receipt leaf: {name}") from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fchmod(handle.fileno(), mode)
            os.fsync(handle.fileno())
        os.fsync(parent)
    except BaseException:
        # The append-only partial file remains visible and makes reuse fail.
        raise


def _exclusive_receipt(path: Path, body: Mapping[str, Any]) -> dict[str, Any]:
    value = {**dict(body), "receipt_digest": _digest(body)}
    _exclusive_bytes(path, _pretty(value), mode=0o400)
    return value


def _exclusive_receipt_at(parent: int, name: str, body: Mapping[str, Any]) -> dict[str, Any]:
    value = {**dict(body), "receipt_digest": _digest(body)}
    _exclusive_bytes_at(parent, name, _pretty(value), mode=0o400)
    return value


def _algorithm(spec: Any) -> str:
    config = getattr(spec, "config", {})
    train = config.get("train", {}) if isinstance(config, Mapping) else {}
    return str(train.get("algorithm", "")) if isinstance(train, Mapping) else ""


def _pair_key(spec: Any) -> tuple[str, str, int]:
    config = getattr(spec, "config", {})
    data = config.get("data", {}) if isinstance(config, Mapping) else {}
    if not isinstance(data, Mapping):
        raise CoordinatorError("G01 RunSpec data section is malformed")
    law_family = str(data.get("rule_family", ""))
    raw_q_p = data.get("q_p")
    if law_family not in {"majority", "parity"} or type(raw_q_p) not in {int, float}:
        raise CoordinatorError("G01 pair key has an unexpected law family or q_p")
    try:
        numeric_q_p = float(raw_q_p)  # type: ignore[arg-type]
        if not math.isfinite(numeric_q_p):
            raise ValueError("non-finite")
        q_p = format(numeric_q_p, ".17g")
    except (TypeError, ValueError) as error:
        raise CoordinatorError("G01 pair key q_p is not numeric") from error
    seed = getattr(spec, "seed", None)
    if type(seed) is not int:
        raise CoordinatorError("G01 pair key seed is not one exact integer")
    return law_family, q_p, seed


def build_balanced_schedule(
    plan: Sequence[Any],
    *,
    output_paths: Mapping[str, tuple[str, str]],
    plan_rows_digest: str,
    plan_key_set_digest: str,
) -> Schedule:
    """Structurally schedule an already-authenticated 120-row plan.

    This helper validates pairing/cardinality/scalar shape only.  Its digest
    arguments are authenticated labels supplied by the caller; the prospective
    high-level lifecycle must independently rebuild and authenticate the exact
    frozen G01 membership before calling it.
    """

    if len(plan) != PLANNED_ROWS:
        raise CoordinatorError("checkpoint B requires exactly 120 original RunSpecs")
    by_pair: dict[tuple[str, str, int], dict[str, Any]] = {}
    plan_keys: set[str] = set()
    for spec in plan:
        plan_key = str(getattr(spec, "plan_key", ""))
        algorithm = _algorithm(spec)
        if algorithm not in {"sft", "outcome_rl"} or not plan_key or plan_key in plan_keys:
            raise CoordinatorError("G01 plan contains a duplicate key or unexpected algorithm")
        plan_keys.add(plan_key)
        pair = by_pair.setdefault(_pair_key(spec), {})
        if algorithm in pair:
            raise CoordinatorError("G01 plan has multiple algorithms for one shared pair key")
        pair[algorithm] = spec
    if len(by_pair) != PAIR_COUNT or any(set(rows) != {"sft", "outcome_rl"} for rows in by_pair.values()):
        raise CoordinatorError("G01 plan is not exactly 60 paired SFT/outcome-RL comparisons")
    if set(output_paths) != plan_keys:
        raise CoordinatorError("precomputed output paths do not cover the exact G01 membership")

    rows: list[ScheduleRow] = []
    for pair_index, pair_key in enumerate(sorted(by_pair)):
        worker_index = pair_index % WORKER_COUNT
        for algorithm in ("sft", "outcome_rl"):
            spec = by_pair[pair_key][algorithm]
            plan_key = str(spec.plan_key)
            run_id, output_path = output_paths[plan_key]
            rows.append(
                ScheduleRow(
                    worker_index=worker_index,
                    pair_index=pair_index,
                    pair_key=pair_key,
                    algorithm=algorithm,
                    plan_key=plan_key,
                    run_id=run_id,
                    output_path=output_path,
                    spec=spec,
                )
            )
    schedule = Schedule(
        rows=tuple(rows),
        plan_rows_digest=_require_sha256(plan_rows_digest, "plan rows digest"),
        plan_key_set_digest=_require_sha256(plan_key_set_digest, "plan key-set digest"),
    )
    if any(len(schedule.for_worker(index)) != ROWS_PER_WORKER for index in range(WORKER_COUNT)):
        raise CoordinatorError("balanced schedule is not exactly 4x30")
    if any(
        len({row.pair_index for row in schedule.for_worker(index)}) != PAIRS_PER_WORKER
        for index in range(WORKER_COUNT)
    ):
        raise CoordinatorError("worker schedule is not exactly 15 complete pairs")
    # Identity-bearing RunSpec dictionaries are never rewritten for scheduling.
    expected = {str(spec.plan_key): spec for spec in plan}
    if any(row.spec is not expected[row.plan_key] for row in schedule.rows):
        raise CoordinatorError("coordinator replaced an original RunSpec")
    return schedule


def _preflight_schedule(repo: Path, plan: Sequence[Any], token: Any) -> Schedule:
    from goalzendo.artifacts import RunStore

    _assert_pre_rendezvous_import_boundary()
    artifact_root = _artifact_root()
    if not _test_roots_active() and artifact_root != CANONICAL_ARTIFACT_ROOT:
        raise CoordinatorError("G01 artifact root differs from its frozen config")
    _require_no_link_components(artifact_root, "G01 artifact root", allow_absent_leaf=True)
    parent = _open_directory_chain(artifact_root.parent, "G01 artifact parent")
    try:
        if not _is_absent_at(parent, artifact_root.name):
            raise CoordinatorError("G01 artifact root must be absent before the global claim")
    finally:
        os.close(parent)
    output_paths: dict[str, tuple[str, str]] = {}
    for spec in plan:
        store = RunStore(artifact_root, spec.config, spec.seed, repo)
        if store.implementation.get("implementation_fingerprint") != G01_SOURCE_FINGERPRINT:
            raise CoordinatorError("RunStore implementation identity changed during global preflight")
        path = _logical_absolute(store.path)
        try:
            path.relative_to(artifact_root)
        except ValueError as error:
            raise CoordinatorError("G01 RunStore path escapes the canonical artifact root") from error
        _require_no_link_components(path, f"G01 output {spec.plan_key}", allow_absent_leaf=True)
        if path.exists() or path.is_symlink():
            raise CoordinatorError(f"G01 output already exists before global claim: {spec.plan_key}")
        output_paths[str(spec.plan_key)] = (str(store.run_id), str(path))
    if len(output_paths) != PLANNED_ROWS or len(set(output_paths.values())) != PLANNED_ROWS:
        raise CoordinatorError("G01 output preflight aliases or omits a RunSpec")
    return build_balanced_schedule(
        plan,
        output_paths=output_paths,
        plan_rows_digest=str(token.plan_rows_digest),
        plan_key_set_digest=str(token.plan_key_set_digest),
    )


def _reserve_all_outputs(schedule: Schedule) -> dict[str, _Reservation]:
    """Create and retain all exact run roots after the global claim.

    The artifact root itself must be absent.  Every component is created and
    opened fd-relative with O_NOFOLLOW.  Workers inherit only their held run
    FDs and re-anchor all stock RunStore/backend I/O through those descriptors.
    """

    artifact_root = _artifact_root()
    parent = _open_directory_chain(artifact_root.parent, "G01 artifact parent")
    held: dict[str, _Reservation] = {}
    try:
        artifact_fd = _mkdir_at(parent, artifact_root.name)
        try:
            for row in schedule.rows:
                path = Path(row.output_path)
                try:
                    relative = path.relative_to(artifact_root)
                except ValueError as error:
                    raise CoordinatorError("reserved output escaped artifact root") from error
                if len(relative.parts) != _RUN_RELATIVE_COMPONENTS or relative.parts[-1] != row.run_id:
                    raise CoordinatorError("reserved output has an unexpected RunStore layout")
                current = os.dup(artifact_fd)
                try:
                    for component in relative.parts[:-1]:
                        child = _ensure_directory_at(current, component)
                        os.close(current)
                        current = child
                    run_fd = _mkdir_at(current, relative.parts[-1])
                    held[row.plan_key] = _Reservation(row.plan_key, path, run_fd)
                finally:
                    os.close(current)
            if len(held) != PLANNED_ROWS:
                raise CoordinatorError("output reservation did not create exactly 120 run roots")
            os.fsync(artifact_fd)
        finally:
            os.close(artifact_fd)
    except BaseException:
        for reservation in held.values():
            os.close(reservation.descriptor)
        raise
    finally:
        os.close(parent)
    return held


def _verify_reservation(reservation: _Reservation, *, require_empty: bool = True) -> None:
    parent = _open_directory_chain(reservation.path.parent, "reserved run parent")
    try:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        observed = os.open(reservation.path.name, flags, dir_fd=parent)
        try:
            held = os.fstat(reservation.descriptor)
            current = os.fstat(observed)
            if (
                (held.st_dev, held.st_ino) != (current.st_dev, current.st_ino)
                or stat.S_IMODE(held.st_mode) != 0o755
                or (require_empty and os.listdir(reservation.descriptor))
            ):
                raise CoordinatorError("reserved run directory changed before initialization")
        finally:
            os.close(observed)
    finally:
        os.close(parent)


def _held_run_path(reservation: _Reservation) -> Path:
    """Return a kernel fd path that remains bound to the reserved inode."""

    if sys.platform != "linux":
        raise CoordinatorError("held artifact I/O requires Linux procfs")
    path = Path(f"/proc/self/fd/{reservation.descriptor}")
    try:
        held = os.fstat(reservation.descriptor)
        observed = os.stat(path)
    except OSError as error:
        raise CoordinatorError("held artifact descriptor is unavailable through procfs") from error
    if not stat.S_ISDIR(held.st_mode) or (held.st_dev, held.st_ino) != (observed.st_dev, observed.st_ino):
        raise CoordinatorError("procfs artifact descriptor does not identify the held run")
    return path


def _close_reservations(reservations: Mapping[str, _Reservation]) -> None:
    for reservation in reservations.values():
        with suppress(OSError):
            os.close(reservation.descriptor)


def _is_absent_at(parent: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        return True
    return False


def _layout(execution_uuid: str) -> dict[str, Path]:
    execution = _uuid4(execution_uuid, "checkpoint-B execution UUID")
    root = _program_root()
    return {
        "program_root": root,
        "claim": root / GLOBAL_CLAIM_NAME,
        "execution_root": root / execution,
        "schedule": root / execution / "schedule.json",
        "rendezvous": root / execution / "rendezvous.json",
        "rows": root / execution / "row-receipts",
        "coordinator_terminal": root / execution / "coordinator-terminal.json",
        "global_failure": root / execution / "global-failure.json",
        "accounting_gate": root / execution / "final-accounting-gate.json",
    }


def _acquire_global_claim(
    *,
    layout: Mapping[str, Any],
    execution_uuid: str,
    token: Any,
    source: Mapping[str, Any],
    schedule: Schedule,
    expected_token_sha256: str,
) -> _ClaimTransaction:
    """Prospective only; overlay audit must add partial-failure/held-claim replay."""

    raise CoordinatorError(RUNTIME_OVERLAY_REFUSAL)

    root_fd = _create_program_root()
    started = time.monotonic_ns()
    deadline = started + WALL_CEILING_SECONDS * 1_000_000_000
    body = {
        "schema": "goalzendo.g01_checkpoint_b_global_claim",
        "schema_version": SCHEMA_VERSION,
        "study_id": "g01",
        "execution_uuid": execution_uuid,
        "execution_root": str(layout["execution_root"]),
        "thin_token_sha256": expected_token_sha256,
        "thin_token_digest": token.token_digest,
        "bridge_source_digest": token.bridge_source_digest,
        "coordinator_source_digest": source["source_digest"],
        "target_binding_digest": token.target_binding_digest,
        "plan_rows_digest": token.plan_rows_digest,
        "plan_key_set_digest": token.plan_key_set_digest,
        "schedule_digest": schedule.digest,
        "started_monotonic_ns": started,
        "deadline_monotonic_ns": deadline,
        "wall_ceiling_seconds": WALL_CEILING_SECONDS,
        "worker_count": WORKER_COUNT,
        "rows_per_worker": ROWS_PER_WORKER,
        "global_one_shot": True,
        "work_stealing": False,
        "retry": False,
        "control_plane_outcome_based_branching": False,
    }
    execution_fd: int | None = None
    rows_fd: int | None = None
    try:
        if not _is_absent_at(root_fd, layout["claim"].name):
            raise CoordinatorError("the global one-shot G01 claim already exists")
        if not _is_absent_at(root_fd, layout["execution_root"].name):
            raise CoordinatorError("checkpoint-B UUID root already exists before global claim")
        claim = _exclusive_receipt_at(root_fd, layout["claim"].name, body)
        execution_fd = _mkdir_at(root_fd, layout["execution_root"].name)
        rows_fd = _mkdir_at(execution_fd, layout["rows"].name)
        os.fsync(execution_fd)
        claim_payload = _require_regular_at(root_fd, layout["claim"].name, "global claim", mode=0o400)
        _exclusive_receipt_at(
            execution_fd,
            layout["schedule"].name,
            {
                **schedule.body(),
                "execution_uuid": execution_uuid,
                "global_claim_sha256": _sha256_bytes(claim_payload),
            },
        )
    except BaseException:
        # The claim is deliberately permanent even if initialization fails.
        if rows_fd is not None:
            os.close(rows_fd)
        if execution_fd is not None:
            os.close(execution_fd)
        os.close(root_fd)
        raise
    return _ClaimTransaction(deadline, claim, root_fd, execution_fd, rows_fd)


def _worker_path(layout: Mapping[str, Any], worker_index: int, kind: str) -> Path:
    return Path(layout["execution_root"]) / f"worker-{worker_index}-{kind}.json"


def _row_receipt_path(layout: Mapping[str, Any], plan_key: str) -> Path:
    if len(plan_key) != 20 or any(character not in "0123456789abcdef" for character in plan_key):
        raise CoordinatorError("RunSpec plan key is malformed")
    return Path(layout["rows"]) / f"{plan_key}.json"


def _verify_runner_source(repo: Path) -> None:
    target = repo / RUNNER_RELATIVE
    if _sha256_bytes(_require_regular(target, "frozen G01 runner")) != RUNNER_SHA256:
        raise CoordinatorError("frozen G01 runner bytes changed")


def _verify_loaded_goalzendo_closure(repo: Path) -> None:
    required = {
        "goalzendo": "src/goalzendo/__init__.py",
        "goalzendo.artifacts": "src/goalzendo/artifacts.py",
        "goalzendo.config": "src/goalzendo/config.py",
        "goalzendo.runner": RUNNER_RELATIVE,
    }
    if "goalzendo.experiment" in sys.modules:
        required["goalzendo.experiment"] = "src/goalzendo/experiment.py"
    for name, required_relative in required.items():
        _require_live_module(repo, name, required_relative)
    for name in sorted(sys.modules):
        if name != "goalzendo" and not name.startswith("goalzendo."):
            continue
        module = sys.modules[name]
        raw = getattr(module, "__file__", None)
        if raw is None:
            raise CoordinatorError(f"loaded GoalZendo module has no source path: {name}")
        observed = _logical_absolute(str(raw))
        try:
            relative = observed.relative_to(repo)
        except ValueError as error:
            raise CoordinatorError(f"loaded GoalZendo module is outside frozen source: {name}") from error
        if not relative.parts or relative.parts[0] != "src":
            raise CoordinatorError(f"loaded GoalZendo module is outside src: {name}")
        _require_live_module(repo, name, relative.as_posix())


def _require_nested_regular_at(root: int, relative: str, label: str) -> bytes:
    parts = Path(relative).parts
    if not parts or Path(relative).is_absolute() or ".." in parts:
        raise CoordinatorError(f"{label} has an unsafe relative path")
    current = os.dup(root)
    try:
        for component in parts[:-1]:
            child = os.open(
                component,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=current,
            )
            os.close(current)
            current = child
        return _require_regular_at(current, parts[-1], label)
    except OSError as error:
        raise CoordinatorError(f"{label} cannot be opened fd-relative without links") from error
    finally:
        os.close(current)


def _read_manifest_at(root: int, relative: str, kind: str) -> dict[str, Any]:
    payload = _strict_json_bytes(
        _require_nested_regular_at(root, relative, f"{kind} manifest"), f"{kind} manifest"
    )
    metadata = payload.get("metadata")
    if (
        set(payload) != {"schema_version", "kind", "digest", "metadata"}
        or not _is_exact_int(payload.get("schema_version"), 1)
        or payload.get("kind") != kind
        or not isinstance(metadata, Mapping)
        or payload.get("digest") != _digest(metadata)
    ):
        raise CoordinatorError(f"{kind} manifest is not authentic")
    return payload


def _verify_materialized_identities(store: Any, run_descriptor: int) -> None:
    dataset = _read_manifest_at(run_descriptor, "manifests/dataset.json", "dataset")
    model = _read_manifest_at(run_descriptor, "manifests/model.json", "model")
    _read_manifest_at(run_descriptor, "manifests/tokenizer.json", "tokenizer")
    model_metadata = model["metadata"]
    if (
        model_metadata.get("requested_model") != MODEL_NAME
        or model_metadata.get("requested_revision") != MODEL_REVISION
        or not dataset["metadata"].get("dataset_binding_digest")
    ):
        raise CoordinatorError("materialized model or dataset identity differs from frozen G01")


def _outcome_blind_success_seal(store: Any, run_descriptor: int) -> dict[str, Any]:
    """Bind only identity/control files, never metrics, predictions or summary."""

    control_files = (
        "identity.json",
        "resolved_config.yaml",
        "implementation.json",
        "environment.json",
        "manifests/dataset.json",
        "manifests/model.json",
        "manifests/tokenizer.json",
        "status.json",
        "COMPLETE",
    )
    rows: list[dict[str, Any]] = []
    for relative in control_files:
        payload = _require_nested_regular_at(run_descriptor, relative, f"successful-row control {relative}")
        rows.append({"path": relative, "sha256": _sha256_bytes(payload), "bytes": len(payload)})
    status = _strict_json_bytes(
        _require_nested_regular_at(run_descriptor, "status.json", "successful-row status"),
        "successful-row status",
    )
    # completion.json is deliberately excluded: even its opaque file hash
    # transitively binds metrics, predictions, and summary.  The worker already
    # observed finalize() return; B seals only the binary state and control
    # identities needed for outcome-independent accounting.
    if (
        status.get("state") != "complete"
        or status.get("run_id") != store.run_id
        or _require_nested_regular_at(run_descriptor, "COMPLETE", "successful-row COMPLETE") != b"complete\n"
    ):
        raise CoordinatorError("successful row lacks an outcome-blind terminal control seal")
    body = {
        "schema": "goalzendo.g01_checkpoint_b_outcome_blind_success_seal",
        "schema_version": SCHEMA_VERSION,
        "run_id": store.run_id,
        "control_files": rows,
        "seal_metrics_file_read": False,
        "seal_predictions_file_read": False,
        "seal_summary_file_read": False,
        "seal_completion_envelope_read": False,
    }
    return {**body, "seal_digest": _digest(body)}


def _replay_outcome_blind_success_seal(run_descriptor: int, expected: Mapping[str, Any], run_id: str) -> None:
    allowed = (
        "identity.json",
        "resolved_config.yaml",
        "implementation.json",
        "environment.json",
        "manifests/dataset.json",
        "manifests/model.json",
        "manifests/tokenizer.json",
        "status.json",
        "COMPLETE",
    )
    rows = expected.get("control_files")
    if not isinstance(rows, list) or len(rows) != len(allowed):
        raise CoordinatorError("outcome-blind success seal has the wrong control inventory")
    current_rows: list[dict[str, Any]] = []
    for relative in allowed:
        payload = _require_nested_regular_at(run_descriptor, relative, f"successful-row replay {relative}")
        current_rows.append({"path": relative, "sha256": _sha256_bytes(payload), "bytes": len(payload)})
    status = _strict_json_bytes(
        _require_nested_regular_at(run_descriptor, "status.json", "successful-row replay status"),
        "successful-row replay status",
    )
    body = {key: value for key, value in expected.items() if key != "seal_digest"}
    exact_keys = {
        "schema",
        "schema_version",
        "run_id",
        "control_files",
        "seal_metrics_file_read",
        "seal_predictions_file_read",
        "seal_summary_file_read",
        "seal_completion_envelope_read",
        "seal_digest",
    }
    if (
        set(expected) != exact_keys
        or expected.get("seal_digest") != _digest(body)
        or expected.get("schema") != "goalzendo.g01_checkpoint_b_outcome_blind_success_seal"
        or not _is_exact_int(expected.get("schema_version"), SCHEMA_VERSION)
        or expected.get("run_id") != run_id
        or rows != current_rows
        or status.get("state") != "complete"
        or status.get("run_id") != run_id
        or _require_nested_regular_at(run_descriptor, "COMPLETE", "successful-row replay COMPLETE")
        != b"complete\n"
        or expected.get("seal_metrics_file_read") is not False
        or expected.get("seal_predictions_file_read") is not False
        or expected.get("seal_summary_file_read") is not False
        or expected.get("seal_completion_envelope_read") is not False
    ):
        raise CoordinatorError("successful row outcome-blind control seal changed")


def _execute_row(
    row: ScheduleRow,
    *,
    repo: Path,
    layout: Mapping[str, Any],
    backend: Any,
    reservation: _Reservation,
    authority: object,
) -> str:
    """Execute one row; every backend failure is global in checkpoint B v1."""

    raise CoordinatorError(RUNTIME_OVERLAY_REFUSAL)

    from goalzendo.artifacts import RunStore
    from goalzendo.runner import (
        RunContext,
        _execute_backend,
        _record_result,
        _require_manifests,
    )

    store = RunStore(_artifact_root(), row.spec.config, row.spec.seed, repo)
    if store.implementation.get("implementation_fingerprint") != G01_SOURCE_FINGERPRINT:
        raise CoordinatorError("RunStore implementation identity changed before initialization")
    if str(store.run_id) != row.run_id or str(_logical_absolute(store.path)) != row.output_path:
        raise CoordinatorError("RunStore identity/path changed after global preflight")
    _verify_reservation(reservation)
    held_path = _held_run_path(reservation)
    store.path = held_path
    store.metrics_path = held_path / "metrics.jsonl"
    store.predictions_path = held_path / "predictions.jsonl"
    initialization = store.initialize(resume=False)
    if store.implementation.get("implementation_fingerprint") != G01_SOURCE_FINGERPRINT:
        raise CoordinatorError("RunStore implementation identity changed after initialization")
    if initialization != "new":
        raise CoordinatorError("checkpoint B forbids resume, skip, steal, or retry")
    context = RunContext(spec=row.spec, store=store, resumed=False)
    try:
        result = _execute_backend(backend, context)
    except BaseException as error:
        # The accepted backend does not yet emit an authenticated phase marker
        # after scorer/optimizer construction.  Exact FloatingPointError and
        # torch.cuda.OutOfMemoryError therefore remain global: exception type
        # plus early manifests cannot prove the error arose inside optimization
        # or evaluation.  This deliberately narrower fail-closed policy can be
        # relaxed only by a separately frozen backend/marker revision.
        with suppress(BaseException):
            store.fail(error)
        raise CoordinatorError(f"global backend failure: {type(error).__name__}") from error
    try:
        _verify_materialized_identities(store, reservation.descriptor)
        _record_result(context, result)
        _require_manifests(context)
        summary = {
            **dict(result.summary),
            "run_id": store.run_id,
            "seed": row.spec.seed,
            "plan_key": row.spec.plan_key,
            "derived_seeds": dict(row.spec.seeds),
        }
        store.finalize(summary)
        success_seal = _outcome_blind_success_seal(store, reservation.descriptor)
        _verify_reservation(reservation, require_empty=False)
        _exclusive_receipt_at(
            int(layout["rows_descriptor"]),
            f"{row.plan_key}.json",
            {
                "schema": "goalzendo.g01_checkpoint_b_row_terminal",
                "schema_version": SCHEMA_VERSION,
                "study_id": "g01",
                "plan_key": row.plan_key,
                "run_id": row.run_id,
                "worker_index": row.worker_index,
                "pair_index": row.pair_index,
                "algorithm": row.algorithm,
                "state": "complete",
                "outcome_blind_success_seal": success_seal,
                "control_plane_outcome_based_branching": False,
            },
        )
    except BaseException as error:
        with suppress(BaseException):
            store.fail(error)
        raise CoordinatorError(f"global storage/finalization failure: {type(error).__name__}") from error
    return "complete"


def _worker_main(
    worker_index: int,
    rows: Sequence[ScheduleRow],
    *,
    repo: Path,
    layout: Mapping[str, Any],
    gate_read_fd: int,
    ready_write_fd: int,
    deadline_ns: int,
    reservations: Mapping[str, _Reservation],
    authority: object,
) -> int:
    raise CoordinatorError(RUNTIME_OVERLAY_REFUSAL)

    assigned_keys = {row.plan_key for row in rows}
    owned_reservations: dict[str, _Reservation] = {}
    for plan_key, reservation in reservations.items():
        if plan_key in assigned_keys:
            owned_reservations[plan_key] = reservation
        else:
            with suppress(OSError):
                os.close(reservation.descriptor)
    try:
        os.setsid()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_index)
        _assert_pre_rendezvous_import_boundary()
        _exclusive_receipt_at(
            int(layout["execution_descriptor"]),
            f"worker-{worker_index}-started.json",
            {
                "schema": "goalzendo.g01_checkpoint_b_worker_started",
                "schema_version": SCHEMA_VERSION,
                "worker_index": worker_index,
                "worker_pid": os.getpid(),
                "assigned_rows": len(rows),
                "assigned_plan_keys_digest": _digest([row.plan_key for row in rows]),
                "backend_imported": False,
                "control_plane_outcome_based_branching": False,
            },
        )
        os.write(ready_write_fd, b"R")
        os.close(ready_write_fd)
        release = os.read(gate_read_fd, 1)
        os.close(gate_read_fd)
        if release != b"G":
            raise CoordinatorError("worker rendezvous gate was not globally released")
        if time.monotonic_ns() >= deadline_ns:
            raise CoordinatorError("global deadline expired at worker release")
        _verify_runner_source(repo)
        from goalzendo.runner import load_backend

        backend = load_backend(DEFAULT_BACKEND)
        _verify_loaded_goalzendo_closure(repo)
        torch_module = importlib.import_module("torch")
        if not torch_module.cuda.is_available() or torch_module.cuda.device_count() != 1:
            raise CoordinatorError("worker does not own exactly one visible CUDA device")
        complete_rows = 0
        for row in rows:
            if time.monotonic_ns() >= deadline_ns:
                raise CoordinatorError("global deadline expired before assigned row")
            if row.plan_key not in owned_reservations:
                raise CoordinatorError("worker lacks its inherited exact run reservation")
            reservation = owned_reservations.pop(row.plan_key)
            try:
                state = _execute_row(
                    row,
                    repo=repo,
                    layout=layout,
                    backend=backend,
                    reservation=reservation,
                    authority=authority,
                )
            finally:
                os.close(reservation.descriptor)
            if state != "complete":
                raise CoordinatorError("checkpoint B v1 accepts only complete rows")
            complete_rows += 1
        _exclusive_receipt_at(
            int(layout["execution_descriptor"]),
            f"worker-{worker_index}-terminal.json",
            {
                "schema": "goalzendo.g01_checkpoint_b_worker_terminal",
                "schema_version": SCHEMA_VERSION,
                "worker_index": worker_index,
                "state": "terminal",
                "assigned_rows": len(rows),
                "complete_rows": complete_rows,
                "terminal_rows": complete_rows,
                "control_plane_outcome_based_branching": False,
            },
        )
        return 0
    except BaseException as error:
        with suppress(BaseException):
            _exclusive_receipt_at(
                int(layout["execution_descriptor"]),
                f"worker-{worker_index}-terminal.json",
                {
                    "schema": "goalzendo.g01_checkpoint_b_worker_terminal",
                    "schema_version": SCHEMA_VERSION,
                    "worker_index": worker_index,
                    "state": "global_fatal",
                    "error_type": type(error).__name__,
                    "control_plane_outcome_based_branching": False,
                },
            )
        return 1
    finally:
        _close_reservations(owned_reservations)


def _group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_groups(pgids: Sequence[int]) -> bool:
    for pgid in pgids:
        if _group_exists(pgid):
            with suppress(ProcessLookupError):
                os.killpg(pgid, signal.SIGTERM)
    deadline = time.monotonic_ns() + TERM_GRACE_SECONDS * 1_000_000_000
    while any(_group_exists(pgid) for pgid in pgids) and time.monotonic_ns() < deadline:
        time.sleep(0.05)
    for pgid in pgids:
        if _group_exists(pgid):
            with suppress(ProcessLookupError):
                os.killpg(pgid, signal.SIGKILL)
    clear_deadline = time.monotonic_ns() + 5 * 1_000_000_000
    while any(_group_exists(pgid) for pgid in pgids) and time.monotonic_ns() < clear_deadline:
        time.sleep(0.05)
    return not any(_group_exists(pgid) for pgid in pgids)


def _guardian_main(read_fd: int, pgids: Sequence[int], deadline_ns: int) -> int:
    for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(number, signal.SIG_IGN)
    while True:
        remaining = max(0.0, (deadline_ns - time.monotonic_ns()) / 1_000_000_000)
        readable, _, _ = select.select([read_fd], [], [], min(0.2, remaining))
        if readable:
            message = os.read(read_fd, 1)
            os.close(read_fd)
            if message == b"N":
                if not any(_group_exists(pgid) for pgid in pgids):
                    return 0
                return 1 if _terminate_groups(pgids) else 125
            return 1 if _terminate_groups(pgids) else 125
        if time.monotonic_ns() >= deadline_ns:
            return 124 if _terminate_groups(pgids) else 125


def _launch_guardian(
    pgids: Sequence[int],
    deadline_ns: int,
    close_fds: Sequence[int],
) -> tuple[int, int, int]:
    read_fd, write_fd = os.pipe()
    ready_read, ready_write = os.pipe()
    pid = os.fork()
    if pid == 0:  # pragma: no cover - subprocess lifecycle contract
        os.close(write_fd)
        os.close(ready_read)
        for descriptor in close_fds:
            if descriptor not in {read_fd, ready_write}:
                with suppress(OSError):
                    os.close(descriptor)
        try:
            os.setsid()
            os.write(ready_write, b"R")
            os.close(ready_write)
            code = _guardian_main(read_fd, pgids, deadline_ns)
        except BaseException:
            code = 125
        os._exit(code)
    os.close(read_fd)
    os.close(ready_write)
    return pid, write_fd, ready_read


def _wait_ready(descriptors: Sequence[int], deadline_ns: int) -> None:
    pending = set(descriptors)
    while pending:
        if time.monotonic_ns() >= deadline_ns:
            raise CoordinatorError("global rendezvous missed its monotonic deadline")
        readable, _, _ = select.select(list(pending), [], [], 0.2)
        for descriptor in readable:
            message = os.read(descriptor, 1)
            os.close(descriptor)
            pending.remove(descriptor)
            if message != b"R":
                raise CoordinatorError("worker or guardian did not acknowledge rendezvous")


def _bounded_reap(pids: Sequence[int], *, timeout_seconds: float = 5.0) -> dict[int, int]:
    """Prospective only; blocking post-SIGKILL waitpid remains a launch blocker."""

    pending = set(pids)
    statuses: dict[int, int] = {}
    deadline_ns = time.monotonic_ns() + int(timeout_seconds * 1_000_000_000)
    while pending and time.monotonic_ns() < deadline_ns:
        for pid in tuple(pending):
            try:
                observed, status = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                pending.remove(pid)
                continue
            if observed == pid:
                statuses[pid] = os.waitstatus_to_exitcode(status)
                pending.remove(pid)
        if pending:
            time.sleep(0.05)
    for pid in pending:
        with suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)
    for pid in tuple(pending):
        try:
            observed, status = os.waitpid(pid, 0)
        except ChildProcessError:
            pending.remove(pid)
            continue
        if observed == pid:
            statuses[pid] = os.waitstatus_to_exitcode(status)
            pending.remove(pid)
    if pending:
        raise CoordinatorError("child process could not be boundedly reaped")
    return statuses


def _expected_control_files(schedule: Schedule, execution_uuid: str) -> set[str]:
    files = {
        "schedule.json",
        "rendezvous.json",
        "coordinator-terminal.json",
        "final-accounting-gate.json",
    }
    files.update(f"worker-{index}-started.json" for index in range(WORKER_COUNT))
    files.update(f"worker-{index}-terminal.json" for index in range(WORKER_COUNT))
    files.update(f"row-receipts/{row.plan_key}.json" for row in schedule.rows)
    return files


def _control_inventory_fd(root_fd_source: int) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = set()
    root_fd = os.dup(root_fd_source)

    def walk(directory_fd: int, prefix: str) -> None:
        for name in sorted(os.listdir(directory_fd)):
            if not name or "/" in name or name in {".", ".."}:
                raise CoordinatorError("checkpoint-B control inventory contains an unsafe leaf")
            relative = f"{prefix}/{name}" if prefix else name
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            child = os.open(name, flags, dir_fd=directory_fd)
            try:
                metadata = os.fstat(child)
                if stat.S_ISDIR(metadata.st_mode):
                    directories.add(relative)
                    walk(child, relative)
                elif (
                    stat.S_ISREG(metadata.st_mode)
                    and metadata.st_nlink == 1
                    and stat.S_IMODE(metadata.st_mode) == 0o400
                ):
                    files.add(relative)
                else:
                    raise CoordinatorError(
                        "checkpoint-B control tree contains a link, special, or wrong-mode entry"
                    )
            finally:
                os.close(child)

    try:
        walk(root_fd, "")
    except OSError as error:
        raise CoordinatorError("checkpoint-B control inventory cannot be opened link-free") from error
    finally:
        os.close(root_fd)
    return files, directories


def _control_inventory(root: Path) -> tuple[set[str], set[str]]:
    root_fd = _open_directory_chain(root, "checkpoint-B execution root")
    try:
        return _control_inventory_fd(root_fd)
    finally:
        os.close(root_fd)


def _build_accounting_gate(
    layout: Mapping[str, Any],
    schedule: Schedule,
    execution_uuid: str,
    claim: Mapping[str, Any],
    reservations: Mapping[str, _Reservation],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for scheduled in schedule.rows:
        _row_receipt_path(layout, scheduled.plan_key)
        receipt_payload = _require_regular_at(
            int(layout["rows_descriptor"]), f"{scheduled.plan_key}.json", "row receipt", mode=0o400
        )
        receipt = _strict_json_bytes(receipt_payload, "row receipt")
        body = {key: value for key, value in receipt.items() if key != "receipt_digest"}
        exact_row_keys = {
            "schema",
            "schema_version",
            "study_id",
            "plan_key",
            "run_id",
            "worker_index",
            "pair_index",
            "algorithm",
            "state",
            "outcome_blind_success_seal",
            "control_plane_outcome_based_branching",
            "receipt_digest",
        }
        if (
            set(receipt) != exact_row_keys
            or receipt.get("receipt_digest") != _digest(body)
            or receipt.get("schema") != "goalzendo.g01_checkpoint_b_row_terminal"
            or not _is_exact_int(receipt.get("schema_version"), SCHEMA_VERSION)
            or receipt.get("study_id") != "g01"
            or receipt.get("plan_key") != scheduled.plan_key
            or receipt.get("run_id") != scheduled.run_id
            or not _is_exact_int(receipt.get("worker_index"), scheduled.worker_index)
            or not _is_exact_int(receipt.get("pair_index"), scheduled.pair_index)
            or receipt.get("algorithm") != scheduled.algorithm
            or receipt.get("state") != "complete"
            or receipt.get("control_plane_outcome_based_branching") is not False
        ):
            raise CoordinatorError("row terminal receipt changed or is not outcome-blind")
        reservation = reservations.get(scheduled.plan_key)
        if reservation is None:
            raise CoordinatorError("parent lost a held run reservation before accounting")
        _verify_reservation(reservation, require_empty=False)
        success_seal = receipt.get("outcome_blind_success_seal")
        if not isinstance(success_seal, Mapping):
            raise CoordinatorError("successful row omits its outcome-blind control seal")
        _replay_outcome_blind_success_seal(reservation.descriptor, success_seal, scheduled.run_id)
        rows.append(
            {
                "plan_key": scheduled.plan_key,
                "run_id": scheduled.run_id,
                "worker_index": scheduled.worker_index,
                "state": receipt["state"],
                "receipt_sha256": _sha256_bytes(receipt_payload),
            }
        )
    counts = {"complete": sum(row["state"] == "complete" for row in rows)}
    if counts != {"complete": PLANNED_ROWS}:
        raise CoordinatorError("accounting does not contain exactly 120 complete rows")
    for worker_index in range(WORKER_COUNT):
        terminal_path = _worker_path(layout, worker_index, "terminal")
        terminal = _strict_json_bytes(
            _require_regular_at(
                int(layout["execution_descriptor"]), terminal_path.name, "worker terminal", mode=0o400
            ),
            "worker terminal",
        )
        body = {key: value for key, value in terminal.items() if key != "receipt_digest"}
        exact_worker_keys = {
            "schema",
            "schema_version",
            "worker_index",
            "state",
            "assigned_rows",
            "complete_rows",
            "terminal_rows",
            "control_plane_outcome_based_branching",
            "receipt_digest",
        }
        if (
            set(terminal) != exact_worker_keys
            or terminal.get("receipt_digest") != _digest(body)
            or terminal.get("schema") != "goalzendo.g01_checkpoint_b_worker_terminal"
            or not _is_exact_int(terminal.get("schema_version"), SCHEMA_VERSION)
            or not _is_exact_int(terminal.get("worker_index"), worker_index)
            or terminal.get("state") != "terminal"
            or not _is_exact_int(terminal.get("assigned_rows"), ROWS_PER_WORKER)
            or not _is_exact_int(terminal.get("complete_rows"), ROWS_PER_WORKER)
            or not _is_exact_int(terminal.get("terminal_rows"), ROWS_PER_WORKER)
            or terminal.get("control_plane_outcome_based_branching") is not False
        ):
            raise CoordinatorError("worker terminal receipt is incomplete")
    return {
        "schema": "goalzendo.g01_checkpoint_b_final_accounting",
        "schema_version": SCHEMA_VERSION,
        "study_id": "g01",
        "execution_uuid": execution_uuid,
        "global_claim_sha256": _sha256_bytes(
            _require_regular_at(int(layout["root_descriptor"]), GLOBAL_CLAIM_NAME, "global claim", mode=0o400)
        ),
        "global_claim_digest": claim["receipt_digest"],
        "schedule_digest": schedule.digest,
        "plan_rows_digest": schedule.plan_rows_digest,
        "plan_key_set_digest": schedule.plan_key_set_digest,
        "planned_rows": PLANNED_ROWS,
        "terminal_rows": PLANNED_ROWS,
        "counts": counts,
        "row_receipts_digest": _digest(rows),
        "worker_count": WORKER_COUNT,
        "parent_accounting_outcome_independent": True,
        "parent_accounting_metrics_file_read": False,
        "parent_accounting_predictions_file_read": False,
        "parent_accounting_summary_file_read": False,
        "retry": False,
    }


def verify_accounting_gate(
    gate_path: str | Path,
    *,
    expected_gate_sha256: str,
    expected_claim_sha256: str,
    expected_execution_uuid: str,
    expected_global_claim_digest: str,
    expected_schedule_digest: str,
    expected_plan_rows_digest: str = G01_PLAN_ROWS_DIGEST,
    expected_plan_key_set_digest: str = G01_PLAN_KEY_SET_DIGEST,
    expected_row_receipts_digest: str,
) -> dict[str, Any]:
    """Read-only whole-file/semantic replay of a completed accounting gate."""

    path = _logical_absolute(gate_path)
    payload = _require_regular(path, "checkpoint-B accounting gate", mode=0o400)
    if _sha256_bytes(payload) != _require_sha256(expected_gate_sha256, "accounting gate SHA-256"):
        raise CoordinatorError("checkpoint-B accounting gate whole-file SHA changed")
    gate = _strict_json_bytes(payload, "checkpoint-B accounting gate")
    body = {key: value for key, value in gate.items() if key != "receipt_digest"}
    exact_keys = {
        "schema",
        "schema_version",
        "study_id",
        "execution_uuid",
        "global_claim_sha256",
        "global_claim_digest",
        "schedule_digest",
        "plan_rows_digest",
        "plan_key_set_digest",
        "planned_rows",
        "terminal_rows",
        "counts",
        "row_receipts_digest",
        "worker_count",
        "parent_accounting_outcome_independent",
        "parent_accounting_metrics_file_read",
        "parent_accounting_predictions_file_read",
        "parent_accounting_summary_file_read",
        "retry",
        "receipt_digest",
    }
    if (
        set(gate) != exact_keys
        or gate.get("receipt_digest") != _digest(body)
        or gate.get("schema") != "goalzendo.g01_checkpoint_b_final_accounting"
        or not _is_exact_int(gate.get("schema_version"), SCHEMA_VERSION)
        or gate.get("study_id") != "g01"
        or gate.get("execution_uuid") != _uuid4(expected_execution_uuid, "accounting execution UUID")
        or gate.get("global_claim_sha256") != _require_sha256(expected_claim_sha256, "claim SHA-256")
        or gate.get("global_claim_digest")
        != _require_sha256(expected_global_claim_digest, "global claim digest")
        or gate.get("schedule_digest") != _require_sha256(expected_schedule_digest, "schedule digest")
        or gate.get("plan_rows_digest") != _require_sha256(expected_plan_rows_digest, "plan rows digest")
        or gate.get("plan_key_set_digest")
        != _require_sha256(expected_plan_key_set_digest, "plan key-set digest")
        or not _is_exact_int(gate.get("planned_rows"), PLANNED_ROWS)
        or not _is_exact_int(gate.get("terminal_rows"), PLANNED_ROWS)
        or not isinstance(gate.get("counts"), Mapping)
        or set(gate["counts"]) != {"complete"}
        or not _is_exact_int(gate["counts"].get("complete"), PLANNED_ROWS)
        or gate.get("row_receipts_digest")
        != _require_sha256(expected_row_receipts_digest, "row receipts digest")
        or not _is_exact_int(gate.get("worker_count"), WORKER_COUNT)
        or gate.get("parent_accounting_outcome_independent") is not True
        or gate.get("parent_accounting_metrics_file_read") is not False
        or gate.get("parent_accounting_predictions_file_read") is not False
        or gate.get("parent_accounting_summary_file_read") is not False
        or gate.get("retry") is not False
    ):
        raise CoordinatorError("checkpoint-B accounting gate schema or outcome boundary changed")
    return gate


def _coordinate(
    *,
    repo: Path,
    execution_uuid: str,
    expected_token_sha256: str,
    expected_bridge_source_digest: str,
    source: Mapping[str, Any],
    authority: object,
) -> dict[str, Any]:
    """Prospective, unexercised lifecycle; unconditionally blocked in this revision."""

    raise CoordinatorError(RUNTIME_OVERLAY_REFUSAL)

    _assert_pre_rendezvous_import_boundary()
    token_payload = getattr(authority, "token_payload", None)
    if not isinstance(token_payload, bytes):
        raise CoordinatorError("bootstrap authority omits its captured thin token")
    token = _token_from_snapshot(
        token_payload,
        expected_sha256=expected_token_sha256,
        expected_bridge_source_digest=expected_bridge_source_digest,
    )
    _require_token_repo_identity(token, repo)
    import yaml  # type: ignore[import-untyped]

    from goalzendo.config import (
        DEFAULT_CONFIG,
        canonical_config,
        deep_merge,
        protected_guard_signature,
        validate_config,
    )
    from goalzendo.runner import build_plan, target_gate_binding

    read_snapshot = getattr(authority, "bytes", None)
    if not callable(read_snapshot):
        raise CoordinatorError("bootstrap authority omits its captured source reader")
    try:
        base = yaml.safe_load(read_snapshot("configs/goalzendo/base.yaml").decode("utf-8")) or {}
        leaf = yaml.safe_load(read_snapshot("configs/goalzendo/g01_known_law.yaml").decode("utf-8")) or {}
    except (UnicodeError, yaml.YAMLError) as error:
        raise CoordinatorError("captured G01 configuration is not exact YAML") from error
    if not isinstance(base, Mapping) or not isinstance(leaf, Mapping) or leaf.get("extends") != "base.yaml":
        raise CoordinatorError("captured G01 config/base relation changed")
    leaf_without_parent = dict(leaf)
    leaf_without_parent.pop("extends")
    config = deep_merge(DEFAULT_CONFIG, deep_merge(base, leaf_without_parent))
    config["_config_path"] = str(repo / "configs/goalzendo/g01_known_law.yaml")
    config["_declared_experiment_id"] = "g01"
    config["_declared_launch_guard"] = "G00_NOT_PASSED__LEARNING_RATES_NOT_FROZEN"
    validate_config(config)
    if (
        _digest(canonical_config(config)) != G01_CANONICAL_CONFIG_DIGEST
        or protected_guard_signature(config) != G01_GUARD_SIGNATURE
    ):
        raise CoordinatorError("captured G01 canonical config or guard identity changed")
    plan = build_plan(config)
    plan_rows = [spec.as_dict() for spec in plan]
    if (
        _digest(plan_rows) != token.plan_rows_digest
        or _digest(sorted(str(spec.plan_key) for spec in plan)) != token.plan_key_set_digest
    ):
        raise CoordinatorError("captured G01 configuration did not rebuild the token-bound plan")
    target = target_gate_binding(config, repo)
    if (
        target.get("source_fingerprint") != G01_SOURCE_FINGERPRINT
        or target.get("planned_run_count") != PLANNED_ROWS
        or target.get("model_identity")
        != {"requested_model": MODEL_NAME, "requested_revision": MODEL_REVISION}
        or len(target.get("authorized_cell_config_digests", ())) != 12
        or _digest(target) != G01_TARGET_BINDING_DIGEST
    ):
        raise CoordinatorError("captured G01 target binding changed")
    _verify_loaded_goalzendo_closure(repo)
    _assert_pre_rendezvous_import_boundary()
    schedule = _preflight_schedule(repo, plan, token)
    _assert_pre_rendezvous_import_boundary()
    layout = _layout(execution_uuid)
    previous_signal_handlers = {
        number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
    }
    for number in previous_signal_handlers:
        signal.signal(number, _signal_as_exception)
    try:
        transaction = _acquire_global_claim(
            layout=layout,
            execution_uuid=execution_uuid,
            token=token,
            source=source,
            schedule=schedule,
            expected_token_sha256=expected_token_sha256,
        )
    except BaseException:
        for number, handler in previous_signal_handlers.items():
            signal.signal(number, handler)
        raise
    deadline_ns = transaction.deadline_ns
    claim = transaction.claim
    layout = {
        **layout,
        "root_descriptor": transaction.root_descriptor,
        "execution_descriptor": transaction.execution_descriptor,
        "rows_descriptor": transaction.rows_descriptor,
    }
    gate_pipes: list[tuple[int, int]] = []
    ready_pipes: list[tuple[int, int]] = []
    worker_pids: list[int] = []
    guardian_rows: list[tuple[int, int, int]] = []
    all_pipe_fds: list[int] = []
    reservations: dict[str, _Reservation] = {}
    try:
        for _index in range(WORKER_COUNT):
            gate_pipes.append(os.pipe())
            ready_pipes.append(os.pipe())
        all_pipe_fds = [descriptor for pair in (*gate_pipes, *ready_pipes) for descriptor in pair]
        reservations = _reserve_all_outputs(schedule)
        for worker_index in range(WORKER_COUNT):
            pid = os.fork()
            if pid == 0:  # pragma: no cover - production subprocess lifecycle
                # Each child gets fresh duplicates of only its assigned run
                # capabilities; parent originals remain held through final replay.
                child_reservations: dict[str, _Reservation] = {}
                assigned = {row.plan_key for row in schedule.for_worker(worker_index)}
                for plan_key, reservation in reservations.items():
                    if plan_key in assigned:
                        child_reservations[plan_key] = _Reservation(
                            plan_key, reservation.path, os.dup(reservation.descriptor)
                        )
                    os.close(reservation.descriptor)
                for index, (read_fd, write_fd) in enumerate(gate_pipes):
                    if index == worker_index:
                        os.close(write_fd)
                    else:
                        os.close(read_fd)
                        os.close(write_fd)
                for index, (read_fd, write_fd) in enumerate(ready_pipes):
                    if index == worker_index:
                        os.close(read_fd)
                    else:
                        os.close(read_fd)
                        os.close(write_fd)
                code = _worker_main(
                    worker_index,
                    schedule.for_worker(worker_index),
                    repo=repo,
                    layout=layout,
                    gate_read_fd=gate_pipes[worker_index][0],
                    ready_write_fd=ready_pipes[worker_index][1],
                    deadline_ns=deadline_ns,
                    reservations=child_reservations,
                    authority=authority,
                )
                os._exit(code)
            worker_pids.append(pid)
        # Parent retains every original reservation through outcome-blind
        # final replay and canonical-path inode validation.
        for read_fd, _ in gate_pipes:
            os.close(read_fd)
        for _, write_fd in ready_pipes:
            os.close(write_fd)

        # A worker acknowledges only after setsid() and its pre-import started
        # receipt.  Waiting here closes the PID/PGID race before a guardian is
        # asked to supervise that process group.
        _wait_ready([read_fd for read_fd, _ in ready_pipes], deadline_ns)
        for index in range(WORKER_COUNT):
            inherited_guardian_fds = [fd for row in guardian_rows for fd in row[1:]]
            guardian_rows.append(
                _launch_guardian(
                    [worker_pids[index]],
                    deadline_ns,
                    [*all_pipe_fds, *inherited_guardian_fds],
                )
            )
        inherited_guardian_fds = [fd for row in guardian_rows for fd in row[1:]]
        guardian_rows.append(
            _launch_guardian(
                worker_pids,
                deadline_ns,
                [*all_pipe_fds, *inherited_guardian_fds],
            )
        )
        _wait_ready([ready_fd for _pid, _control, ready_fd in guardian_rows], deadline_ns)
        _assert_pre_rendezvous_import_boundary()
        _exclusive_receipt_at(
            transaction.execution_descriptor,
            layout["rendezvous"].name,
            {
                "schema": "goalzendo.g01_checkpoint_b_rendezvous",
                "schema_version": SCHEMA_VERSION,
                "execution_uuid": execution_uuid,
                "worker_pids": worker_pids,
                "worker_guardian_pids": [row[0] for row in guardian_rows[:WORKER_COUNT]],
                "outer_guardian_pid": guardian_rows[-1][0],
                "worker_count": WORKER_COUNT,
                "schedule_digest": schedule.digest,
                "backend_imported_before_release": False,
                "all_workers_and_guardians_ready": True,
                "control_plane_outcome_based_branching": False,
            },
        )
        for _read_fd, write_fd in gate_pipes:
            os.write(write_fd, b"G")
            os.close(write_fd)

        statuses: dict[int, int] = {}
        guardian_pids = [row[0] for row in guardian_rows]
        while len(statuses) < WORKER_COUNT:
            if time.monotonic_ns() >= deadline_ns:
                raise CoordinatorError("checkpoint-B global monotonic deadline expired")
            for guardian_pid in guardian_pids:
                observed, status_value = os.waitpid(guardian_pid, os.WNOHANG)
                if observed == guardian_pid:
                    exit_code = os.waitstatus_to_exitcode(status_value)
                    raise CoordinatorError(
                        f"guardian {guardian_pid} exited during worker execution with {exit_code}"
                    )
            for index, pid in enumerate(worker_pids):
                if index in statuses:
                    continue
                observed, status_value = os.waitpid(pid, os.WNOHANG)
                if observed == pid:
                    statuses[index] = os.waitstatus_to_exitcode(status_value)
                    if statuses[index] != 0:
                        raise CoordinatorError(f"worker {index} reported a global failure")
            time.sleep(0.05)
        # Every worker leader was reaped above.  Guardians independently prove
        # that no descendant remains in any supervised process group before N
        # can be accepted as a clean stop.
        if any(_group_exists(pid) for pid in worker_pids):
            raise CoordinatorError("worker descendants remain after every leader exited")
        for _pid, control_fd, _ready_fd in guardian_rows:
            with suppress(BrokenPipeError):
                os.write(control_fd, b"N")
            os.close(control_fd)
        guardian_pids = [row[0] for row in guardian_rows]
        remaining_seconds = max(0.0, (deadline_ns - time.monotonic_ns()) / 1_000_000_000)
        guardian_results = _bounded_reap(guardian_pids, timeout_seconds=min(5.0, remaining_seconds))
        guardian_statuses = [guardian_results.get(pid, 125) for pid in guardian_pids]
        if guardian_statuses != [0] * (WORKER_COUNT + 1):
            raise CoordinatorError("a worker or outer guardian failed its clean-stop contract")
        _exclusive_receipt_at(
            transaction.execution_descriptor,
            layout["coordinator_terminal"].name,
            {
                "schema": "goalzendo.g01_checkpoint_b_coordinator_terminal",
                "schema_version": SCHEMA_VERSION,
                "execution_uuid": execution_uuid,
                "state": "terminal",
                "worker_exit_codes": statuses,
                "guardian_exit_codes": guardian_statuses,
                "deadline_exceeded": False,
                "control_plane_outcome_based_branching": False,
            },
        )
        gate_body = _build_accounting_gate(layout, schedule, execution_uuid, claim, reservations)
        gate = _exclusive_receipt_at(
            transaction.execution_descriptor, layout["accounting_gate"].name, gate_body
        )
        files, directories = _control_inventory_fd(transaction.execution_descriptor)
        if files != _expected_control_files(schedule, execution_uuid) or directories != {"row-receipts"}:
            raise CoordinatorError("checkpoint-B control inventory is not exact after finalization")
        return {
            "execution_uuid": execution_uuid,
            "global_claim_sha256": _sha256_bytes(
                _require_regular_at(
                    transaction.root_descriptor, GLOBAL_CLAIM_NAME, "global claim", mode=0o400
                )
            ),
            "accounting_gate_sha256": _sha256_bytes(
                _require_regular_at(
                    transaction.execution_descriptor,
                    layout["accounting_gate"].name,
                    "accounting gate",
                    mode=0o400,
                )
            ),
            "accounting_gate_digest": gate["receipt_digest"],
            "terminal_rows": PLANNED_ROWS,
            "retry": False,
        }
    except BaseException as error:
        _terminate_groups(worker_pids)
        with suppress(CoordinatorError):
            _bounded_reap(worker_pids, timeout_seconds=5.0)
        for _pid, control_fd, ready_fd in guardian_rows:
            with suppress(OSError):
                os.close(ready_fd)
            with suppress(BrokenPipeError, OSError):
                os.write(control_fd, b"X")
            with suppress(OSError):
                os.close(control_fd)
        with suppress(CoordinatorError):
            _bounded_reap([row[0] for row in guardian_rows], timeout_seconds=5.0)
        with suppress(BaseException):
            _exclusive_receipt_at(
                transaction.execution_descriptor,
                layout["global_failure"].name,
                {
                    "schema": "goalzendo.g01_checkpoint_b_global_failure",
                    "schema_version": SCHEMA_VERSION,
                    "execution_uuid": execution_uuid,
                    "error_type": type(error).__name__,
                    "retry": False,
                    "control_plane_outcome_based_branching": False,
                },
            )
        for descriptor in all_pipe_fds:
            with suppress(OSError):
                os.close(descriptor)
        raise
    finally:
        _close_reservations(reservations)
        for number, handler in previous_signal_handlers.items():
            signal.signal(number, handler)
        os.close(transaction.rows_descriptor)
        os.close(transaction.execution_descriptor)
        os.close(transaction.root_descriptor)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run-g01-global-coordinator")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--execution-uuid", required=True)
    parser.add_argument("--expected-coordinator-token-sha256", required=True)
    parser.add_argument("--expected-bridge-source-digest", required=True)
    parser.add_argument("--expected-coordinator-source-digest", required=True)
    return parser


def main_from_entrypoint(
    observed_entrypoint: str | Path,
    argv: Sequence[str] | None = None,
    *,
    _bootstrap_snapshot: object | None = None,
) -> int:
    """Refuse until an independently frozen B overlay replaces this source."""

    del observed_entrypoint, argv, _bootstrap_snapshot
    print(f"run_g01_global_coordinator: error: {RUNTIME_OVERLAY_REFUSAL}", file=sys.stderr)
    return 2
