"""Prospective, fail-closed eligibility bridge from G00-F to exact G01.

The immutable G00-F evaluators deliberately never authorize G01.  This module
is the separately reviewed bridge they require.  It authenticates one
precommitted hardware route, independently replays that route's evaluator,
and binds passing evidence to the unchanged 120-run G01 plan.  Its sidecar is
scientific eligibility evidence only: direct execution remains prohibited
until a separately frozen global coordinator exists.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import stat
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

Route = Literal["h100", "h200"]
EvidenceLevel = Literal["full", "metadata"]

ROUTE_LOCK_SCHEMA = "goalzendo.g00f_g01_eligibility_route_lock"
ROUTE_LOCK_SCHEMA_VERSION = 1
ELIGIBILITY_SCHEMA = "goalzendo.g00f_g01_scientific_eligibility"
ELIGIBILITY_SCHEMA_VERSION = 1
COORDINATOR_TOKEN_SCHEMA = "goalzendo.g00f_g01_coordinator_input"
COORDINATOR_TOKEN_SCHEMA_VERSION = 1

_CANONICAL_G00F_PROGRAM_ROOT = Path("/workspace/status-goalzendo/g00f-executions")
# Test isolation is deliberately private, requires pytest's per-test marker, and
# is not accepted by any public API or CLI argument.  Production always takes
# the literal canonical root above.
_TEST_ONLY_G00F_PROGRAM_ROOT: Path | None = None
FROZEN_SOURCE_DIRECTORY = "frozen-source"
ROUTE_LOCK_FILENAME = "g00f-g01-route-lock.json"
ELIGIBILITY_FILENAME = "g00f-g01-scientific-eligibility.json"
COORDINATOR_TOKEN_FILENAME = "g00f-g01-coordinator-input.json"
FINAL_GATE_FILENAME = "g00f-final-gate.json"

G01_CONFIG_RELATIVE_PATH = "configs/goalzendo/g01_known_law.yaml"
G01_PROTOCOL_RELATIVE_PATH = "docs/goalzendo/protocols/g01-known-law.md"
G01_RUNNER_RELATIVE_PATH = "src/goalzendo/runner.py"
G01_WRAPPER_RELATIVE_PATH = "runs/goalzendo/run_g01_after_g00f_bridge.py"
G00E_GATE_RELATIVE_PATH = "reproducibility/goalzendo/g00d-gate-20260811/g00e-gate-v3.json"

G01_CONFIG_FILE_SHA256 = "ee6d53556189a6b6e25cfbfb204325017a058c63a605653df4c175463e00ce18"
G01_CANONICAL_CONFIG_DIGEST = "f9f91978a446a6e750172e0e377bc1b738ccef64b5238992555e1afb6e6bd110"
G01_TARGET_BINDING_DIGEST = "3315d20a6f9bdae3c5fdaf9567c5bce7b592890d0ebd26e10682002d816bf0c6"
G01_GUARD_SIGNATURE = "9feaae82edf801aad4bd4a5b16f633be8dfa2dbd41b29601e7b61b564c763464"
G01_LAUNCH_GUARD = "G00_NOT_PASSED__LEARNING_RATES_NOT_FROZEN"
G01_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
G01_MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
G01_PROTOCOL_FILE_SHA256 = "428243c3da271bc2d79e16c0fde20d47076eb3837c6740ab733e278cddcdbce9"
G01_RUNNER_FILE_SHA256 = "46b55ad4bdd08073e5f89ae101e0862372817b74b590f07ddb8f8c4331e5b9e1"
G01_SOURCE_FINGERPRINT = "1a8146377b9a9620690025671614edb2dd20d214f4e195da3cf3528809b2c694"
G01_PLAN_ROWS_DIGEST = "f51f6b6f574295433dacec8fafe20508526036ef15dce220a1d8c24e8cd6e55f"
G01_PLAN_KEY_SET_DIGEST = "7fb1bc870c3b93d1d6a5ae6148b83b460c9ee6d246b592f67f908e8080fa8b91"
G01_PLANNED_RUNS = 120
G01_CELL_COUNT = 12
G01_SEEDS = (1103, 1129, 1151, 1171, 1201, 1217, 1231, 1277, 1291, 1301)

G00E_GATE_FILE_SHA256 = "39eb64e2f049f9cbbf818851afec7e17324173b7f87a2c0df6df8398171e9e68"
G00E_GATE_DIGEST = "38f71b1ec096d6c5758bbc933e8afcfd258b0e3707f4a2bfd311499f87a7985d"
G00E_CHECK_STATUS: Mapping[str, bool] = {
    "constrained_scorer": False,
    "dataset_integrity": True,
    "no_signal_chance": True,
    "optimizer_stability": True,
    "rule_adapters": False,
    "surface_leakage": True,
}

H100_FREEZE_FILE_SHA256 = "b2e385488ea7eb7c6f7bfc834c32ff2707aa9f96b718ea62b7ae92c9a4b8df38"
H100_FREEZE_DIGEST = "2b04534b74873a88c73791a17a5c5a7b66a9ddd472b11168034f11517a8d5047"
H100_LAUNCHER_FILE_SHA256 = "576f7dc82f6c4842af872067f72bbe05b55667e4705b9255efbdb29b35493bbe"
H200_FREEZE_FILE_SHA256 = "fd9cf73d124ec566e0589ec0b2e5e4d48a172bca2d04aff51bbb75a58b76b0f9"
H200_FREEZE_DIGEST = "537282f70f7425855807db31681d13a7c07c9402c31170e4189f6fdf5889f52e"
H200_LAUNCHER_FILE_SHA256 = "5bc6dfcab92f9632a1fc3f4a29815d9c6298ee2fc86f47690ce6d1b371ad39f2"

BRIDGE_SOURCE_RELATIVE_PATHS = (
    "src/goalzendo_g00f_g01_bridge/__init__.py",
    "src/goalzendo_g00f_g01_bridge/bridge.py",
    "src/goalzendo_g00f_g01_bridge/cli.py",
    G01_WRAPPER_RELATIVE_PATH,
)

_EXPECTED_SELECTED_SETTINGS: Mapping[str, Mapping[str, Any]] = {
    "outcome_rl": {
        "effective_batch": 50,
        "entropy_coefficient": 0.01,
        "law_families": ["majority", "parity"],
        "learning_rate": 3e-6,
        "model_identity": {
            "requested_model": G01_MODEL_NAME,
            "requested_revision": G01_MODEL_REVISION,
        },
        "proxy_accuracies": [0.95, 1.0],
        "seeds": [9001, 9002, 9003],
        "stability_window_steps": [128, 256],
        "update_method": "full",
    },
    "sft": {
        "effective_batch": 50,
        "entropy_coefficient": 0.0,
        "law_families": ["majority", "parity"],
        "learning_rate": 3e-6,
        "model_identity": {
            "requested_model": G01_MODEL_NAME,
            "requested_revision": G01_MODEL_REVISION,
        },
        "proxy_accuracies": [0.95, 1.0],
        "seeds": [9001, 9002, 9003],
        "stability_window_steps": [128, 256],
        "update_method": "full",
    },
}


class BridgeError(RuntimeError):
    """Raised before eligibility can be established or any direct execution."""


@dataclass(frozen=True)
class Eligibility:
    """Verified exact-G01 scientific eligibility, never direct permission."""

    path: Path
    file_sha256: str
    eligibility_digest: str
    route_lock_sha256: str
    route_lock_digest: str
    bridge_source_digest: str
    route: Route
    execution_uuid: str
    selected_profile: str | None
    plan_rows_digest: str
    plan_key_set_digest: str
    target_binding_digest: str
    model_name: str
    model_revision: str
    scope: str
    g01_scientifically_eligible: bool
    direct_g01_launch_authorized: bool
    dedicated_global_coordinator_required: bool


@dataclass(frozen=True)
class CoordinatorToken:
    """Narrow verified projection safe for a future global coordinator."""

    path: Path
    file_sha256: str
    token_digest: str
    eligibility_file_sha256: str
    eligibility_digest: str
    route_lock_sha256: str
    route_lock_digest: str
    bridge_source_digest: str
    route: Route
    execution_uuid: str
    target_binding_digest: str
    plan_rows_digest: str
    plan_key_set_digest: str
    g01_scientifically_eligible: bool
    direct_g01_launch_authorized: bool
    dedicated_global_coordinator_required: bool


@dataclass(frozen=True)
class _ExecutionLayout:
    program_root: Path
    execution_root: Path
    frozen_source: Path
    freeze: Path
    ledger_root: Path
    route_lock: Path
    final_gate: Path
    eligibility: Path
    coordinator_token: Path
    profile_selection: Path | None


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def semantic_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def require_sha256(value: Any, label: str) -> str:
    normalized = str(value)
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise BridgeError(f"{label} must be a lowercase SHA-256 digest")
    return normalized


def _reject_constant(value: str) -> Any:
    raise BridgeError(f"strict JSON contains the non-finite constant {value}")


def strict_json(path: str | Path, label: str) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise BridgeError(f"{label} does not exist: {target}")

    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise BridgeError(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        parsed = json.loads(
            target.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BridgeError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(parsed, dict):
        raise BridgeError(f"{label} must contain exactly one JSON object")
    return parsed


def _exclusive_json(path: str | Path, value: Mapping[str, Any]) -> None:
    target = _logical_absolute(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    except FileExistsError as error:
        raise BridgeError(f"refusing to overwrite append-only artifact: {target}") from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(pretty_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(target, 0o400)
        directory_descriptor = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        # Never replace or silently reuse a partially created eligibility artifact.
        raise


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or len(value) != 20 or value[10] != "T" or not value.endswith("Z"):
        raise BridgeError(f"{label} must be a second-resolution UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise BridgeError(f"{label} is not a valid UTC timestamp") from error
    if parsed.tzinfo != timezone.utc or parsed.microsecond != 0:
        raise BridgeError(f"{label} must be a second-resolution UTC timestamp")
    return parsed


def _uuid4(value: Any, label: str) -> str:
    normalized = str(value)
    try:
        parsed = uuid.UUID(normalized)
    except (ValueError, AttributeError) as error:
        raise BridgeError(f"{label} must be a canonical UUID4") from error
    if parsed.version != 4 or str(parsed) != normalized:
        raise BridgeError(f"{label} must be a canonical UUID4")
    return normalized


def _route(value: Any) -> Route:
    if value == "h100":
        return "h100"
    if value == "h200":
        return "h200"
    raise BridgeError("route must be exactly one of h100 or h200")


def _logical_absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _test_root_active() -> bool:
    return _TEST_ONLY_G00F_PROGRAM_ROOT is not None and "PYTEST_CURRENT_TEST" in os.environ


def _require_no_symlink_components(
    path: str | Path,
    label: str,
    *,
    require_exists: bool = True,
) -> Path:
    """Return a lexical absolute path after lstat-checking every component."""

    logical = _logical_absolute(path)
    current = Path(logical.anchor)
    for part in logical.parts[1:]:
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError as error:
            if require_exists:
                raise BridgeError(f"{label} path component is absent: {current}") from error
            break
        if stat.S_ISLNK(metadata.st_mode):
            test_program = (
                _logical_absolute(_TEST_ONLY_G00F_PROGRAM_ROOT)
                if _test_root_active() and _TEST_ONLY_G00F_PROGRAM_ROOT is not None
                else None
            )
            if (
                test_program is not None
                and current.name == FROZEN_SOURCE_DIRECTORY
                and current.parent.parent == test_program
            ):
                continue
            raise BridgeError(f"{label} path contains a symlink component: {current}")
    return logical


def _require_repo_regular_file(repo: Path, relative: str, label: str) -> Path:
    target = _logical_absolute(repo / relative)
    try:
        target.relative_to(repo)
    except ValueError as error:  # pragma: no cover - relative paths are source constants
        raise BridgeError(f"{label} escaped the supplied repo") from error
    current = repo
    for part in Path(relative).parts:
        current /= part
        if current.is_symlink():
            raise BridgeError(f"{label} contains a symlink below the supplied repo")
    if not target.is_file() or not stat.S_ISREG(target.stat().st_mode):
        raise BridgeError(f"{label} is not one regular file under the supplied repo")
    return target


def _module_relative_path(repo: Path, module_name: str) -> str:
    stem = Path("src", *module_name.split("."))
    module_file = repo / stem.with_suffix(".py")
    package_file = repo / stem / "__init__.py"
    if module_file.is_file():
        return module_file.relative_to(repo).as_posix()
    if package_file.is_file():
        return package_file.relative_to(repo).as_posix()
    raise BridgeError(f"loaded project module {module_name} has no exact source file in the supplied repo")


def _observed_module_is_exact(observed: Path, expected: Path) -> bool:
    observed_logical = _logical_absolute(observed)
    if _test_root_active():
        return observed_logical.resolve() == expected.resolve()
    return observed_logical == expected


def _require_module_file(module_name: str, repo: Path, relative: str) -> None:
    module = importlib.import_module(module_name)
    raw_file = getattr(module, "__file__", None)
    module_spec = getattr(module, "__spec__", None)
    spec_origin = getattr(module_spec, "origin", None)
    loader = getattr(module_spec, "loader", None)
    get_filename = getattr(loader, "get_filename", None)
    try:
        loader_file = get_filename(module_name) if callable(get_filename) else None
    except (ImportError, AttributeError, TypeError, ValueError):
        loader_file = None
    expected = _require_repo_regular_file(repo, relative, f"module {module_name}")
    observed = Path(str(raw_file)) if raw_file is not None else None
    if (
        observed is None
        or spec_origin is None
        or loader_file is None
        or observed.is_symlink()
        or not observed.is_file()
        or not stat.S_ISREG(observed.stat().st_mode)
        or not _observed_module_is_exact(observed, expected)
        or not _observed_module_is_exact(Path(str(spec_origin)), expected)
        or not _observed_module_is_exact(Path(str(loader_file)), expected)
    ):
        raise BridgeError(f"imported {module_name} is not the exact module under the supplied frozen source")


def _require_loaded_project_module_closure(repo: Path, route_package: str | None = None) -> None:
    prefixes = ["goalzendo_g00f_g01_bridge"]
    if route_package is not None:
        prefixes.append(route_package)
    for name in sorted(sys.modules):
        if not (
            name == "goalzendo"
            or name.startswith("goalzendo.")
            or any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)
        ):
            continue
        module = sys.modules.get(name)
        raw_file = getattr(module, "__file__", None)
        if raw_file is None:
            raise BridgeError(f"loaded project module {name} has no auditable source file")
        _require_module_file(name, repo, _module_relative_path(repo, name))


def require_entrypoint_file(repo: str | Path, observed_file: str | Path, relative: str) -> None:
    """Prove a live CLI/stub entrypoint is the source-bound file in ``repo``."""

    resolved = _repo_root(repo)
    allowed = {
        "src/goalzendo_g00f_g01_bridge/cli.py",
        G01_WRAPPER_RELATIVE_PATH,
    }
    if relative not in allowed:
        raise BridgeError("executing entrypoint is not an allowed checkpoint-A entrypoint")
    expected = _require_repo_regular_file(resolved, relative, "checkpoint-A entrypoint")
    observed = Path(observed_file)
    if observed.is_symlink() or not observed.is_file() or not _observed_module_is_exact(observed, expected):
        raise BridgeError("executing entrypoint is not the exact source-bound file under the supplied repo")


def _repo_root(repo: str | Path) -> Path:
    logical = _logical_absolute(repo)
    if not _test_root_active() and (logical.is_symlink() or logical.resolve() != logical):
        raise BridgeError("supplied repo may not contain symlink indirection")
    if not (logical / "pyproject.toml").is_file() or not (logical / "src" / "goalzendo").is_dir():
        raise BridgeError("repo does not contain the GoalZendo source tree")
    expected_bridge = _require_repo_regular_file(
        logical,
        "src/goalzendo_g00f_g01_bridge/bridge.py",
        "bridge module",
    )
    observed_bridge = Path(__file__)
    if (
        observed_bridge.is_symlink()
        or not observed_bridge.is_file()
        or not _observed_module_is_exact(observed_bridge, expected_bridge)
    ):
        raise BridgeError("imported bridge is not the exact bridge under the supplied repo")
    for module_name, relative in (
        ("goalzendo_g00f_g01_bridge", "src/goalzendo_g00f_g01_bridge/__init__.py"),
        ("goalzendo", "src/goalzendo/__init__.py"),
        ("goalzendo.artifacts", "src/goalzendo/artifacts.py"),
        ("goalzendo.config", "src/goalzendo/config.py"),
        ("goalzendo.runner", "src/goalzendo/runner.py"),
    ):
        _require_module_file(module_name, logical, relative)
    _require_loaded_project_module_closure(logical)
    return logical


def _require_route_modules(repo: Path, route: Route) -> None:
    package = "goalzendo_g00f" if route == "h100" else "goalzendo_g00f_h200"
    if route == "h200":
        importlib.import_module("goalzendo_g00f_h200.qualification")
        importlib.import_module("goalzendo_g00f_h200.qualification_producer")
    _require_module_file(package, repo, f"src/{package}/__init__.py")
    for module in ("freeze", "evaluator"):
        _require_module_file(
            f"{package}.{module}",
            repo,
            f"src/{package}/{module}.py",
        )
    _require_loaded_project_module_closure(repo, package)


def _program_root() -> Path:
    override = _TEST_ONLY_G00F_PROGRAM_ROOT
    if override is not None:
        if "PYTEST_CURRENT_TEST" not in os.environ:
            raise BridgeError("test-only program-root override is forbidden outside pytest")
        return _logical_absolute(override)
    return _CANONICAL_G00F_PROGRAM_ROOT


def _execution_layout(repo: str | Path, execution_uuid: str, route: Route) -> _ExecutionLayout:
    resolved = _repo_root(repo)
    execution = _uuid4(execution_uuid, "execution UUID")
    selected_route = _route(route)
    program = _program_root()
    execution_root = program / execution
    analysis_root = program.parents[1] / "analysis-goalzendo" / "g00f-executions"
    frozen_source = execution_root / FROZEN_SOURCE_DIRECTORY
    if resolved != frozen_source:
        raise BridgeError("repo must be the exact /workspace status execution UUID frozen-source path")
    if program == _CANONICAL_G00F_PROGRAM_ROOT and resolved.resolve() != resolved:
        raise BridgeError("canonical frozen-source path may not contain symlink indirection")
    freeze_relative = (
        "reproducibility/goalzendo/g00f-execution-freeze-20260811/execution-freeze.json"
        if selected_route == "h100"
        else "reproducibility/goalzendo/g00f-h200-execution-freeze-20260811/execution-freeze.json"
    )
    return _ExecutionLayout(
        program_root=program,
        execution_root=execution_root,
        frozen_source=frozen_source,
        freeze=frozen_source / freeze_relative,
        ledger_root=execution_root / "itt-ledger" / execution,
        route_lock=program / ROUTE_LOCK_FILENAME,
        final_gate=analysis_root / execution / FINAL_GATE_FILENAME,
        eligibility=program / ELIGIBILITY_FILENAME,
        coordinator_token=program / COORDINATOR_TOKEN_FILENAME,
        profile_selection=(
            execution_root / "h200-profile-selection.json" if selected_route == "h200" else None
        ),
    )


def execution_layout(repo: str | Path, execution_uuid: str, route: Route) -> dict[str, Any]:
    """Return the source-enforced canonical paths; no caller may override them."""

    layout = _execution_layout(repo, execution_uuid, route)
    return {
        "program_root": str(layout.program_root),
        "execution_root": str(layout.execution_root),
        "frozen_source": str(layout.frozen_source),
        "freeze": str(layout.freeze),
        "ledger_root": str(layout.ledger_root),
        "route_lock": str(layout.route_lock),
        "final_gate": str(layout.final_gate),
        "eligibility": str(layout.eligibility),
        "coordinator_token": str(layout.coordinator_token),
        "profile_selection": (
            str(layout.profile_selection) if layout.profile_selection is not None else None
        ),
    }


def calculate_bridge_source_binding(repo: str | Path) -> dict[str, Any]:
    """Calculate the code binding that an independent operator must pin."""

    resolved = _repo_root(repo)
    files: dict[str, str] = {}
    for relative in BRIDGE_SOURCE_RELATIVE_PATHS:
        target = _require_repo_regular_file(resolved, relative, "bridge source file")
        files[relative] = sha256_file(target)
    observed = semantic_digest(files)
    return {"source_files": files, "source_digest": observed}


def bridge_source_binding(repo: str | Path, expected_source_digest: str) -> dict[str, Any]:
    """Authenticate the additive bridge and its non-executing checkpoint-A stub."""

    expected = require_sha256(expected_source_digest, "externally expected bridge source digest")
    binding = calculate_bridge_source_binding(repo)
    observed = str(binding["source_digest"])
    if observed != expected:
        raise BridgeError("bridge source bytes differ from the externally expected digest")
    return binding


def _g01_plan_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    from goalzendo.runner import build_plan

    return [spec.as_dict() for spec in build_plan(config)]


def g01_binding(repo: str | Path) -> dict[str, Any]:
    """Rebuild and authenticate the unchanged exact G01 scientific target."""

    from goalzendo.artifacts import implementation_provenance, stable_hash
    from goalzendo.config import canonical_config, get_path, load_config, protected_guard_signature
    from goalzendo.runner import target_gate_binding

    resolved = _repo_root(repo)
    config_path = _require_repo_regular_file(
        resolved,
        G01_CONFIG_RELATIVE_PATH,
        "G01 config",
    )
    protocol_path = _require_repo_regular_file(
        resolved,
        G01_PROTOCOL_RELATIVE_PATH,
        "G01 protocol",
    )
    runner_path = _require_repo_regular_file(
        resolved,
        G01_RUNNER_RELATIVE_PATH,
        "GoalZendo runner",
    )
    if (
        sha256_file(config_path) != G01_CONFIG_FILE_SHA256
        or sha256_file(protocol_path) != G01_PROTOCOL_FILE_SHA256
        or sha256_file(runner_path) != G01_RUNNER_FILE_SHA256
    ):
        raise BridgeError("G01 config, protocol, or frozen runner bytes changed")
    config = load_config(config_path)
    canonical_digest = stable_hash(canonical_config(config), 64)
    target = target_gate_binding(config, resolved)
    target_digest = stable_hash(target, 64)
    guard_signature = protected_guard_signature(config)
    plan_rows = _g01_plan_rows(config)
    plan_keys = [str(row["plan_key"]) for row in plan_rows]
    plan_rows_digest = stable_hash(plan_rows, 64)
    plan_key_set_digest = stable_hash(sorted(plan_keys), 64)
    provenance = implementation_provenance(resolved)
    sweep_values = [row.get("sweep_values") for row in plan_rows]
    if (
        canonical_digest != G01_CANONICAL_CONFIG_DIGEST
        or target_digest != G01_TARGET_BINDING_DIGEST
        or guard_signature != G01_GUARD_SIGNATURE
        or provenance.get("implementation_fingerprint") != G01_SOURCE_FINGERPRINT
        or get_path(config, "experiment.id") != "g01"
        or get_path(config, "experiment.status") != "prospective"
        or get_path(config, "run.launch_guard") != G01_LAUNCH_GUARD
        or tuple(int(seed) for seed in get_path(config, "run.seeds")) != G01_SEEDS
        or get_path(config, "model.name") != G01_MODEL_NAME
        or get_path(config, "model.revision") != G01_MODEL_REVISION
        or len(plan_rows) != G01_PLANNED_RUNS
        or len(set(plan_keys)) != G01_PLANNED_RUNS
        or len({str(row["cell_id"]) for row in plan_rows}) != G01_CELL_COUNT
        or plan_rows_digest != G01_PLAN_ROWS_DIGEST
        or plan_key_set_digest != G01_PLAN_KEY_SET_DIGEST
        or {str(values["train.algorithm"]) for values in sweep_values if isinstance(values, Mapping)}
        != {"sft", "outcome_rl"}
        or {str(values["data.rule_family"]) for values in sweep_values if isinstance(values, Mapping)}
        != {"parity", "majority"}
        or {float(values["data.q_p"]) for values in sweep_values if isinstance(values, Mapping)}
        != {0.8, 0.95, 1.0}
        or target.get("planned_run_count") != G01_PLANNED_RUNS
        or target.get("model_identity")
        != {"requested_model": G01_MODEL_NAME, "requested_revision": G01_MODEL_REVISION}
    ):
        raise BridgeError("G01 config/source/model/guard/seed/plan binding changed")
    return {
        "config": {
            "path": str(config_path),
            "file_sha256": G01_CONFIG_FILE_SHA256,
            "canonical_digest": canonical_digest,
        },
        "protocol": {"path": str(protocol_path), "file_sha256": G01_PROTOCOL_FILE_SHA256},
        "runner": {"path": str(runner_path), "file_sha256": G01_RUNNER_FILE_SHA256},
        "source_fingerprint": G01_SOURCE_FINGERPRINT,
        "guard": G01_LAUNCH_GUARD,
        "guard_signature": guard_signature,
        "model_identity": {
            "requested_model": G01_MODEL_NAME,
            "requested_revision": G01_MODEL_REVISION,
        },
        "target_binding": copy.deepcopy(target),
        "target_binding_digest": target_digest,
        "plan": {
            "planned_runs": G01_PLANNED_RUNS,
            "cell_count": G01_CELL_COUNT,
            "seed_count": len(G01_SEEDS),
            "seeds": list(G01_SEEDS),
            "rows_digest": plan_rows_digest,
            "plan_key_set_digest": plan_key_set_digest,
        },
        "scope": {
            "eligible": "g01_primary_full_model_known_law_only",
            "excluded_experiment_ids": ["g01a", "g01l", "g02"],
        },
    }


def historical_g00e_binding(repo: str | Path, current_g01: Mapping[str, Any]) -> dict[str, Any]:
    """Authenticate selected settings while affirming that G00-E failed."""

    resolved = _repo_root(repo)
    path = _require_repo_regular_file(
        resolved,
        G00E_GATE_RELATIVE_PATH,
        "historical G00-E gate",
    )
    if sha256_file(path) != G00E_GATE_FILE_SHA256:
        raise BridgeError("immutable historical G00-E gate bytes changed")
    gate = strict_json(path, "historical G00-E gate")
    body = {key: value for key, value in gate.items() if key != "gate_digest"}
    assessment = gate.get("assessment")
    assessment_body = (
        {key: value for key, value in assessment.items() if key != "assessment_digest"}
        if isinstance(assessment, Mapping)
        else None
    )
    checks = gate.get("checks")
    observed_status = (
        {str(name): value.get("passed") for name, value in checks.items()}
        if isinstance(checks, Mapping) and all(isinstance(value, Mapping) for value in checks.values())
        else None
    )
    target = (
        gate.get("authorized_targets", {}).get("g01")
        if isinstance(gate.get("authorized_targets"), Mapping)
        else None
    )
    if (
        gate.get("schema") != "goalzendo.g00_gate"
        or gate.get("schema_version") != 2
        or gate.get("gate_digest") != G00E_GATE_DIGEST
        or gate.get("gate_digest") != semantic_digest(body)
        or not isinstance(assessment, Mapping)
        or assessment_body is None
        or assessment.get("assessment_digest") != semantic_digest(assessment_body)
        or gate.get("overall_passed") is not False
        or observed_status != dict(G00E_CHECK_STATUS)
        or gate.get("selected_optimizer_settings") != _EXPECTED_SELECTED_SETTINGS
        or target != current_g01.get("target_binding")
        or semantic_digest(target) != G01_TARGET_BINDING_DIGEST
    ):
        raise BridgeError("historical G00-E failure/settings/target binding changed")
    return {
        "path": str(path),
        "file_sha256": G00E_GATE_FILE_SHA256,
        "gate_digest": G00E_GATE_DIGEST,
        "overall_passed": False,
        "check_status": dict(G00E_CHECK_STATUS),
        "failed_checks": ["constrained_scorer", "rule_adapters"],
        "selected_optimizer_settings": copy.deepcopy(dict(_EXPECTED_SELECTED_SETTINGS)),
        "selected_settings_digest": semantic_digest(_EXPECTED_SELECTED_SETTINGS),
        "target_binding_digest": G01_TARGET_BINDING_DIGEST,
        "g01_scientifically_eligible": False,
        "direct_g01_launch_authorized": False,
        "dedicated_global_coordinator_required": True,
    }


def _route_static_binding(repo: Path, route: Route) -> dict[str, Any]:
    if route == "h100":
        launcher_relative = "runs/goalzendo/run_g00f_frozen_4h100.sh"
        launcher_sha = H100_LAUNCHER_FILE_SHA256
        freeze_sha = H100_FREEZE_FILE_SHA256
        freeze_digest = H100_FREEZE_DIGEST
    else:
        launcher_relative = "runs/goalzendo/run_g00f_frozen_4h200.sh"
        launcher_sha = H200_LAUNCHER_FILE_SHA256
        freeze_sha = H200_FREEZE_FILE_SHA256
        freeze_digest = H200_FREEZE_DIGEST
    launcher = _require_repo_regular_file(repo, launcher_relative, "frozen route launcher")
    if sha256_file(launcher) != launcher_sha:
        raise BridgeError(f"frozen G00-F {route.upper()} launcher bytes changed")
    return {
        "route": route,
        "freeze_file_sha256": freeze_sha,
        "freeze_digest": freeze_digest,
        "launcher": {"path": str(launcher), "file_sha256": launcher_sha},
    }


def _verify_route_freeze(
    *,
    repo: Path,
    route: Route,
    freeze_path: str | Path,
    expected_freeze_sha256: str,
    profile_selection_path: str | Path | None,
    expected_profile_selection_sha256: str | None,
    expected_provision_receipt_sha256: str,
    expected_pod_id: str,
) -> Any:
    _require_route_modules(repo, route)
    if _logical_absolute(freeze_path).is_symlink():
        raise BridgeError("selected route freeze may not be a symlink")
    if profile_selection_path is not None and _logical_absolute(profile_selection_path).is_symlink():
        raise BridgeError("H200 profile-selection evidence may not be a symlink")
    static = _route_static_binding(repo, route)
    expected = require_sha256(expected_freeze_sha256, "externally expected G00-F freeze digest")
    if expected != static["freeze_file_sha256"]:
        raise BridgeError("selected route does not use its one canonical G00-F freeze")
    if route == "h100":
        if profile_selection_path is not None or expected_profile_selection_sha256 is not None:
            raise BridgeError("H100 route forbids H200 profile-selection evidence")
        from goalzendo_g00f.freeze import verify_freeze

        h100_verified = verify_freeze(
            repo=repo,
            freeze_path=freeze_path,
            expected_freeze_sha256=expected,
        )
        if (
            h100_verified.file_sha256 != static["freeze_file_sha256"]
            or h100_verified.digest != static["freeze_digest"]
        ):
            raise BridgeError("verified G00-F freeze differs from the selected route binding")
        return h100_verified
    else:
        if profile_selection_path is None or expected_profile_selection_sha256 is None:
            raise BridgeError("H200 route requires its externally pinned profile selection")
        from goalzendo_g00f_h200.freeze import verify_freeze as verify_h200_freeze
        from goalzendo_g00f_h200.freeze import verify_profile_selection_receipt

        h200_verified = verify_h200_freeze(
            repo=repo,
            freeze_path=freeze_path,
            expected_freeze_sha256=expected,
        )
        h200_verified = verify_profile_selection_receipt(
            verified=h200_verified,
            receipt_path=profile_selection_path,
            expected_receipt_sha256=require_sha256(
                expected_profile_selection_sha256,
                "externally expected H200 profile-selection digest",
            ),
            expected_provision_receipt_sha256=require_sha256(
                expected_provision_receipt_sha256,
                "externally expected provision-receipt digest",
            ),
            expected_pod_id=str(expected_pod_id),
        )
        if (
            h200_verified.file_sha256 != static["freeze_file_sha256"]
            or h200_verified.digest != static["freeze_digest"]
        ):
            raise BridgeError("verified G00-F freeze differs from the selected route binding")
        return h200_verified


def create_route_lock(
    *,
    repo: str | Path,
    route: Route,
    execution_uuid: str,
    freeze_path: str | Path,
    expected_freeze_sha256: str,
    ledger_root: str | Path,
    prospective_gate_output: str | Path,
    expected_bridge_source_digest: str,
    output: str | Path,
) -> dict[str, Any]:
    """Precommit one hardware route and execution UUID before any ITT row."""

    resolved = _repo_root(repo)
    selected_route = _route(route)
    execution = _uuid4(execution_uuid, "route-lock execution UUID")
    layout = _execution_layout(resolved, execution, selected_route)
    source = bridge_source_binding(resolved, expected_bridge_source_digest)
    g01 = g01_binding(resolved)
    g00e = historical_g00e_binding(resolved, g01)
    static = _route_static_binding(resolved, selected_route)
    expected_freeze = require_sha256(expected_freeze_sha256, "route-lock freeze SHA-256")
    freeze = _logical_absolute(freeze_path)
    _require_no_symlink_components(freeze, "route-lock G00-F freeze")
    ledger = _logical_absolute(ledger_root)
    gate = _logical_absolute(prospective_gate_output)
    target = _logical_absolute(output)
    if (
        expected_freeze != static["freeze_file_sha256"]
        or not freeze.is_file()
        or freeze.is_symlink()
        or sha256_file(freeze) != expected_freeze
        or freeze != layout.freeze
    ):
        raise BridgeError("route lock does not bind the canonical selected-route freeze")
    freeze_payload = strict_json(freeze, "route-lock G00-F freeze")
    if freeze_payload.get("freeze_digest") != static["freeze_digest"]:
        raise BridgeError("route-lock freeze semantic identity changed")
    if ledger != layout.ledger_root or gate != layout.final_gate or target != layout.route_lock:
        raise BridgeError("route lock inputs differ from the source-enforced execution layout")
    if ledger.exists():
        raise BridgeError("route lock must be created before the ITT ledger exists")
    if gate.exists():
        raise BridgeError("route lock must be created before a final G00-F gate exists")
    if layout.eligibility.exists() or layout.eligibility.is_symlink():
        raise BridgeError("route lock requires an absent canonical eligibility sidecar")
    if layout.coordinator_token.exists() or layout.coordinator_token.is_symlink():
        raise BridgeError("route lock requires an absent canonical coordinator token")
    body = {
        "schema": ROUTE_LOCK_SCHEMA,
        "schema_version": ROUTE_LOCK_SCHEMA_VERSION,
        "study_id": "g00f_to_g01",
        "created_at_utc": _utc_now(),
        "route": selected_route,
        "execution_uuid": execution,
        "freeze": {
            "path": str(freeze),
            "file_sha256": expected_freeze,
            "freeze_digest": static["freeze_digest"],
        },
        "route_launcher": static["launcher"],
        "ledger_root": str(ledger),
        "prospective_gate_output": str(gate),
        "prospective_eligibility_output": str(layout.eligibility),
        "prospective_coordinator_token_output": str(layout.coordinator_token),
        "bridge_source": source,
        "g01": g01,
        "historical_g00e": g00e,
        "precommitment": {
            "exactly_one_route": True,
            "ledger_absent_at_creation": True,
            "final_gate_absent_at_creation": True,
            "eligibility_absent_at_creation": True,
            "coordinator_token_absent_at_creation": True,
            "g00f_outcomes_seen": False,
            "post_itt_route_fallback": False,
            "g01_scientifically_eligible": False,
            "direct_g01_launch_authorized": False,
            "dedicated_global_coordinator_required": True,
        },
    }
    payload = {**body, "route_lock_digest": semantic_digest(body)}
    _exclusive_json(target, payload)
    return {
        "path": str(target),
        "file_sha256": sha256_file(target),
        "route_lock_digest": payload["route_lock_digest"],
        "route": selected_route,
        "execution_uuid": execution,
        "g01_scientifically_eligible": False,
        "direct_g01_launch_authorized": False,
        "dedicated_global_coordinator_required": True,
    }


def verify_route_lock(
    path: str | Path,
    *,
    repo: str | Path,
    expected_route_lock_sha256: str,
    expected_bridge_source_digest: str,
) -> dict[str, Any]:
    """Authenticate the independently pinned, append-only pre-ITT route lock."""

    resolved = _repo_root(repo)
    target = _logical_absolute(path)
    expected_sha = require_sha256(expected_route_lock_sha256, "route-lock SHA-256")
    if (
        not target.is_file()
        or target.is_symlink()
        or target.stat().st_nlink != 1
        or stat.S_IMODE(target.stat().st_mode) != 0o400
        or sha256_file(target) != expected_sha
    ):
        raise BridgeError("route-lock bytes, type, link count, or mode changed")
    lock = strict_json(target, "G00-F/G01 route lock")
    if target.read_bytes() != pretty_json_bytes(lock):
        raise BridgeError("route lock is not in the one canonical byte encoding")
    body = {key: value for key, value in lock.items() if key != "route_lock_digest"}
    exact_keys = {
        "bridge_source",
        "created_at_utc",
        "execution_uuid",
        "freeze",
        "g01",
        "historical_g00e",
        "ledger_root",
        "precommitment",
        "prospective_eligibility_output",
        "prospective_coordinator_token_output",
        "prospective_gate_output",
        "route",
        "route_launcher",
        "schema",
        "schema_version",
        "study_id",
    }
    route = _route(lock.get("route"))
    execution = _uuid4(lock.get("execution_uuid"), "route-lock execution UUID")
    layout = _execution_layout(resolved, execution, route)
    _parse_utc(lock.get("created_at_utc"), "route-lock creation time")
    current_source = bridge_source_binding(resolved, expected_bridge_source_digest)
    current_g01 = g01_binding(resolved)
    current_g00e = historical_g00e_binding(resolved, current_g01)
    static = _route_static_binding(resolved, route)
    _require_no_symlink_components(layout.freeze, "route-lock G00-F freeze")
    freeze = lock.get("freeze")
    ledger = _logical_absolute(str(lock.get("ledger_root", "")))
    gate = _logical_absolute(str(lock.get("prospective_gate_output", "")))
    eligibility = _logical_absolute(str(lock.get("prospective_eligibility_output", "")))
    coordinator_token = _logical_absolute(str(lock.get("prospective_coordinator_token_output", "")))
    if (
        set(body) != exact_keys
        or lock.get("schema") != ROUTE_LOCK_SCHEMA
        or lock.get("schema_version") != ROUTE_LOCK_SCHEMA_VERSION
        or lock.get("study_id") != "g00f_to_g01"
        or lock.get("route_lock_digest") != semantic_digest(body)
        or lock.get("bridge_source") != current_source
        or lock.get("g01") != current_g01
        or lock.get("historical_g00e") != current_g00e
        or lock.get("route_launcher") != static["launcher"]
        or not isinstance(freeze, Mapping)
        or set(freeze) != {"file_sha256", "freeze_digest", "path"}
        or freeze.get("file_sha256") != static["freeze_file_sha256"]
        or freeze.get("freeze_digest") != static["freeze_digest"]
        or _logical_absolute(str(freeze.get("path", ""))) != layout.freeze
        or layout.freeze.is_symlink()
        or sha256_file(layout.freeze) != static["freeze_file_sha256"]
        or ledger != layout.ledger_root
        or target != layout.route_lock
        or gate != layout.final_gate
        or eligibility != layout.eligibility
        or coordinator_token != layout.coordinator_token
        or lock.get("precommitment")
        != {
            "exactly_one_route": True,
            "ledger_absent_at_creation": True,
            "final_gate_absent_at_creation": True,
            "eligibility_absent_at_creation": True,
            "coordinator_token_absent_at_creation": True,
            "g00f_outcomes_seen": False,
            "post_itt_route_fallback": False,
            "g01_scientifically_eligible": False,
            "direct_g01_launch_authorized": False,
            "dedicated_global_coordinator_required": True,
        }
    ):
        raise BridgeError("route lock schema, source, route, execution, or noneligibility changed")
    return {
        "path": str(target),
        "file_sha256": expected_sha,
        "route_lock_digest": lock["route_lock_digest"],
        "created_at_utc": lock["created_at_utc"],
        "route": route,
        "execution_uuid": execution,
        "freeze": copy.deepcopy(dict(freeze)),
        "ledger_root": str(ledger),
        "prospective_gate_output": str(gate),
        "prospective_eligibility_output": str(eligibility),
        "prospective_coordinator_token_output": str(coordinator_token),
    }


def _validate_passing_assessment(
    *,
    route: Route,
    verified: Any,
    assessment: Mapping[str, Any],
    execution_uuid: str,
) -> dict[str, Any]:
    if assessment.get("evidence_status") != "complete" or assessment.get("overall_passed") is not True:
        raise BridgeError("G00-F assessment is not complete and passing")
    checks = assessment.get("checks")
    if (
        not isinstance(checks, Mapping)
        or not checks
        or any(not isinstance(value, Mapping) or value.get("passed") is not True for value in checks.values())
    ):
        raise BridgeError("one or more G00-F checks did not pass")
    itt = assessment.get("intention_to_train")
    audit = assessment.get("attempt_ledger")
    if not isinstance(itt, list) or not isinstance(audit, Mapping):
        raise BridgeError("passing G00-F assessment lacks ITT or attempt-ledger evidence")
    attempt_rows = audit.get("rows")
    ledger = audit.get("ledger")
    if not isinstance(attempt_rows, list) or not isinstance(ledger, Mapping):
        raise BridgeError("passing G00-F assessment has malformed attempt evidence")
    itt_plan_keys = [str(row.get("plan_key", "")) for row in itt if isinstance(row, Mapping)]
    itt_run_ids = [str(row.get("run_id", "")) for row in itt if isinstance(row, Mapping)]
    attempt_plan_keys = [str(row.get("plan_key", "")) for row in attempt_rows if isinstance(row, Mapping)]
    if (
        len(itt) != 160
        or len(itt_plan_keys) != 160
        or len(set(itt_plan_keys)) != 160
        or len(set(itt_run_ids)) != 160
        or any(
            not isinstance(row, Mapping)
            or row.get("state") != "complete"
            or row.get("attempt_state") != "complete"
            or row.get("intention_to_train") is not True
            for row in itt
        )
        or len(attempt_rows) != 160
        or len(attempt_plan_keys) != 160
        or len(set(attempt_plan_keys)) != 160
        or set(attempt_plan_keys) != set(itt_plan_keys)
        or any(not isinstance(row, Mapping) or row.get("state") != "complete" for row in attempt_rows)
        or audit.get("all_complete") is not True
        or audit.get("global_stop") is not None
        or ledger.get("execution_uuid") != execution_uuid
        or Path(str(ledger.get("ledger_root", ""))).name != execution_uuid
    ):
        raise BridgeError("G00-F does not contain exactly 160 successful ITT attempts")
    seal_rows: list[dict[str, Any]] = []
    for row in attempt_rows:
        assert isinstance(row, Mapping)
        seals = row.get("outcome_file_seals")
        if not isinstance(seals, Mapping) or set(seals) != {
            "metrics.jsonl",
            "predictions.jsonl",
            "summary.json",
        }:
            raise BridgeError("complete G00-F attempt lacks its exact three outcome seals")
        for name in ("metrics.jsonl", "predictions.jsonl", "summary.json"):
            seal = seals[name]
            if (
                not isinstance(seal, Mapping)
                or set(seal) != {"bytes", "path", "sealed_mode", "sha256"}
                or seal.get("sealed_mode") != 0
            ):
                raise BridgeError("G00-F outcome seal schema changed")
            seal_rows.append(
                {
                    "plan_key": row["plan_key"],
                    "name": name,
                    "bytes": seal["bytes"],
                    "path": seal["path"],
                    "sha256": seal["sha256"],
                    "unsealed_mode": 0o400,
                }
            )
    if len(seal_rows) != 480 or len({str(row["path"]) for row in seal_rows}) != 480:
        raise BridgeError("G00-F outcome seal inventory is not exactly 480 unique files")
    panel_unseal = audit.get("panel_unseal")
    if not isinstance(panel_unseal, Mapping):
        raise BridgeError("passing G00-F assessment lacks the all-160 panel unseal")
    unseal = strict_json(Path(str(panel_unseal.get("path", ""))), "G00-F panel unseal")
    unseal_rows = unseal.get("outcome_files")
    if not isinstance(unseal_rows, list) or len(unseal_rows) != 480:
        raise BridgeError("G00-F panel unseal inventory is not exactly 480 rows")
    expected_unseal = [
        {
            "plan_key": sealed["plan_key"],
            "run_id": next(
                str(row["run_id"])
                for row in itt
                if isinstance(row, Mapping) and row.get("plan_key") == sealed["plan_key"]
            ),
            "name": sealed["name"],
            "bytes": sealed["bytes"],
            "sha256": sealed["sha256"],
            "unsealed_mode": 0o400,
        }
        for sealed in seal_rows
    ]
    if unseal_rows != expected_unseal:
        raise BridgeError("G00-F 480-row unseal differs from the terminal outcome seals")
    selected_profile: str | None
    if route == "h100":
        if (
            getattr(verified, "selected_profile", None) is not None
            or "selected_profile" in assessment
            or "profile_selection" in assessment
            or "selected_profile" in ledger
            or "profile_selection" in ledger
        ):
            raise BridgeError("H100 eligibility contains forbidden H200 profile evidence")
        selected_profile = None
    else:
        selected_profile = getattr(verified, "selected_profile", None)
        selection = getattr(verified, "selection_receipt", None)
        if (
            selected_profile not in {"baseline", "tuned"}
            or not isinstance(selection, Mapping)
            or assessment.get("selected_profile") != selected_profile
            or assessment.get("profile_selection") != selection
            or ledger.get("selected_profile") != selected_profile
            or ledger.get("profile_selection") != selection
            or selection.get("execution_uuid") != execution_uuid
        ):
            raise BridgeError("H200 assessment does not bind one consistent pre-ITT profile")
    return {
        "itt_rows": itt,
        "attempt_rows": attempt_rows,
        "ledger": ledger,
        "seal_rows": seal_rows,
        "panel_unseal": panel_unseal,
        "selected_profile": selected_profile,
    }


def _file_binding(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    target = _logical_absolute(path)
    _require_no_symlink_components(target, "bound evidence file")
    if not target.is_file() or target.is_symlink() or target.stat().st_nlink != 1:
        raise BridgeError(f"bound evidence file is absent, symlinked, or multiply linked: {target}")
    digest = sha256_file(target)
    if expected_sha256 is not None and digest != require_sha256(expected_sha256, f"evidence {target}"):
        raise BridgeError(f"bound evidence file bytes changed: {target}")
    metadata = target.stat()
    return {
        "path": str(target),
        "file_sha256": digest,
        "bytes": metadata.st_size,
        "mode": stat.S_IMODE(metadata.st_mode),
    }


def _dedupe_bindings(bindings: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in bindings:
        binding = dict(raw)
        path = str(binding.get("path", ""))
        previous = result.get(path)
        if previous is not None and previous != binding:
            raise BridgeError(f"the same evidence path has conflicting bindings: {path}")
        result[path] = binding
    return [result[path] for path in sorted(result)]


def _collect_evidence_bindings(
    *,
    repo: Path,
    verified: Any,
    route_lock: Mapping[str, Any],
    freeze_path: str | Path,
    final_gate_path: str | Path,
    worker_result_receipts: Mapping[int, str | Path],
    validated: Mapping[str, Any],
    profile_selection_path: str | Path | None,
) -> dict[str, Any]:
    ledger = validated["ledger"]
    attempt_rows = validated["attempt_rows"]
    seal_rows = validated["seal_rows"]
    ledger_root = Path(str(ledger["ledger_root"]))
    bindings: list[dict[str, Any]] = [
        _file_binding(route_lock["path"], expected_sha256=str(route_lock["file_sha256"])),
        _file_binding(
            freeze_path,
            expected_sha256=str(route_lock["freeze"]["file_sha256"]),
        ),
        _file_binding(final_gate_path),
        _file_binding(
            repo / G00E_GATE_RELATIVE_PATH,
            expected_sha256=G00E_GATE_FILE_SHA256,
        ),
        _file_binding(ledger["path"], expected_sha256=str(ledger["file_sha256"])),
        _file_binding(
            ledger_root / "budget-start.json",
            expected_sha256=str(ledger["budget_start_file_sha256"]),
        ),
        _file_binding(
            validated["panel_unseal"]["path"],
            expected_sha256=str(validated["panel_unseal"]["file_sha256"]),
        ),
    ]
    provision = ledger.get("runpod_provision")
    if not isinstance(provision, Mapping):
        raise BridgeError("verified G00-F ledger omits provision evidence")
    bindings.append(_file_binding(provision["path"], expected_sha256=str(provision["file_sha256"])))
    if set(worker_result_receipts) != set(range(4)):
        raise BridgeError("eligibility requires exactly worker result receipts 0,1,2,3")
    bindings.extend(_file_binding(worker_result_receipts[index]) for index in range(4))
    if profile_selection_path is not None:
        selection = ledger.get("profile_selection")
        if not isinstance(selection, Mapping):
            raise BridgeError("H200 ledger omits its selected-profile receipt binding")
        bindings.append(_file_binding(profile_selection_path, expected_sha256=str(selection["file_sha256"])))
    for row in attempt_rows:
        if not isinstance(row, Mapping):
            raise BridgeError("attempt row is not an object")
        key = str(row["plan_key"])
        for directory, field in (
            ("starts", "start_file_sha256"),
            ("terminals", "terminal_file_sha256"),
            ("leases", "lease_file_sha256"),
        ):
            bindings.append(
                _file_binding(ledger_root / directory / f"{key}.json", expected_sha256=str(row[field]))
            )
    for row in seal_rows:
        bindings.append(_file_binding(row["path"], expected_sha256=str(row["sha256"])))

    run_input_names = {
        "COMPLETE",
        "attempts/attempt-0001.json",
        "attempts/resolved-config-0001.yaml",
        "completion.json",
        "environment.json",
        "g00f-freeze-binding.json",
        "identity.json",
        "implementation.json",
        "manifests/dataset.json",
        "manifests/model.json",
        "manifests/tokenizer.json",
        "metrics.jsonl",
        "predictions.jsonl",
        "resolved_config.yaml",
        "status.json",
        "summary.json",
    }
    run_input_count = 0
    for frozen in verified.all_rows:
        run_path = _require_no_symlink_components(
            frozen["artifact_path"],
            "G00-F artifact run root",
        )
        if not run_path.is_dir():
            raise BridgeError("G00-F artifact run root is not a directory")
        candidates = list(run_path.rglob("*"))
        observed = {
            candidate.relative_to(run_path).as_posix() for candidate in candidates if candidate.is_file()
        }
        if observed != run_input_names or any(candidate.is_symlink() for candidate in candidates):
            raise BridgeError("completed G00-F run input inventory has an extra or missing file")
        for relative in sorted(run_input_names):
            bindings.append(_file_binding(run_path / relative))
            run_input_count += 1

    execution_root = ledger_root.parents[1]
    direct_execution_files = sorted(
        (candidate for candidate in execution_root.iterdir() if candidate.is_file()),
        key=lambda candidate: candidate.name,
    )
    if any(candidate.is_symlink() for candidate in direct_execution_files):
        raise BridgeError("G00-F execution-root input is symlinked")
    bindings.extend(_file_binding(candidate) for candidate in direct_execution_files)

    model_snapshot_paths: set[str] = set()
    audits = ledger.get("model_integration_audits")
    if not isinstance(audits, Mapping) or set(audits) != {"g00f-0p5b", "g00f-1p5b"}:
        raise BridgeError("G00-F ledger omits the two model-snapshot audit bindings")
    for audit in audits.values():
        if not isinstance(audit, Mapping):
            raise BridgeError("G00-F model-integration audit binding is malformed")
        model_receipt = audit.get("model_snapshot_receipt")
        if not isinstance(model_receipt, Mapping):
            raise BridgeError("G00-F model-integration audit omits its snapshot receipt")
        bindings.append(
            _file_binding(model_receipt["path"], expected_sha256=str(model_receipt["file_sha256"]))
        )
        snapshot_root = _require_no_symlink_components(
            model_receipt["snapshot_root"],
            "G00-F model snapshot root",
        )
        if not snapshot_root.is_dir():
            raise BridgeError("G00-F model snapshot root is not a directory")
        snapshot_files = sorted(snapshot_root.iterdir(), key=lambda candidate: candidate.name)
        if not snapshot_files or any(
            candidate.is_symlink() or not candidate.is_file() for candidate in snapshot_files
        ):
            raise BridgeError("G00-F model snapshot contains a missing, nested, or symlinked leaf")
        for candidate in snapshot_files:
            bindings.append(_file_binding(candidate))
            model_snapshot_paths.add(str(candidate.resolve()))

    embedded_bundle_manifest = Path(verified.repo) / "G00F-BUNDLE-MANIFEST.json"
    if embedded_bundle_manifest.is_file():
        bindings.append(_file_binding(embedded_bundle_manifest))
    deduplicated = _dedupe_bindings(bindings)
    return {
        "files": deduplicated,
        "file_count": len(deduplicated),
        "files_digest": semantic_digest(deduplicated),
        "outcome_seal_count": len(seal_rows),
        "attempt_receipt_count": len(attempt_rows) * 3,
        "worker_result_receipt_count": 4,
        "run_evaluator_input_count": run_input_count,
        "execution_root_direct_file_count": len(direct_execution_files),
        "model_snapshot_leaf_count": len(model_snapshot_paths),
    }


def _verify_evidence_bindings(binding: Mapping[str, Any], *, level: EvidenceLevel) -> None:
    if level not in {"full", "metadata"}:
        raise BridgeError("evidence level must be full or metadata")
    files = binding.get("files")
    if (
        not isinstance(files, list)
        or binding.get("file_count") != len(files)
        or binding.get("files_digest") != semantic_digest(files)
        or binding.get("outcome_seal_count") != 480
        or binding.get("attempt_receipt_count") != 480
        or binding.get("worker_result_receipt_count") != 4
        or binding.get("run_evaluator_input_count") != 2_560
        or not isinstance(binding.get("execution_root_direct_file_count"), int)
        or int(binding["execution_root_direct_file_count"]) < 1
        or not isinstance(binding.get("model_snapshot_leaf_count"), int)
        or int(binding["model_snapshot_leaf_count"]) < 1
    ):
        raise BridgeError("eligibility evidence-file manifest is malformed")
    seen: set[str] = set()
    for raw in files:
        if not isinstance(raw, Mapping) or set(raw) != {
            "bytes",
            "file_sha256",
            "mode",
            "path",
        }:
            raise BridgeError("eligibility evidence-file row is malformed")
        path = str(raw["path"])
        if path in seen:
            raise BridgeError("eligibility evidence-file manifest repeats a path")
        seen.add(path)
        target = _logical_absolute(path)
        _require_no_symlink_components(target, "eligibility evidence file")
        if not target.is_file() or target.is_symlink() or target.stat().st_nlink != 1:
            raise BridgeError(f"eligibility evidence file disappeared or changed type: {target}")
        metadata = target.stat()
        if (
            metadata.st_size != raw["bytes"]
            or stat.S_IMODE(metadata.st_mode) != raw["mode"]
            or (level == "full" and sha256_file(target) != raw["file_sha256"])
        ):
            raise BridgeError(f"eligibility evidence file changed after replay: {target}")


def _derive_route_assessment(
    *,
    route: Route,
    verified: Any,
    artifact_roots: Mapping[str, str | Path],
    ledger_root: str | Path,
    worker_result_receipts: Mapping[int, str | Path],
    expected_provision_receipt_sha256: str,
    expected_pod_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require_route_modules(Path(verified.repo), route)
    if route == "h100":
        from goalzendo_g00f.evaluator import derive_assessment as derive_h100_assessment
        from goalzendo_g00f.evaluator import gate_from_assessment as h100_gate_from_assessment

        assessment = derive_h100_assessment(
            verified=verified,
            artifact_roots=artifact_roots,
            ledger_root=ledger_root,
            worker_result_receipts=worker_result_receipts,
            expected_provision_receipt_sha256=expected_provision_receipt_sha256,
            expected_pod_id=expected_pod_id,
        )
        return assessment, h100_gate_from_assessment(verified, assessment)
    else:
        from goalzendo_g00f_h200.evaluator import derive_assessment as derive_h200_assessment
        from goalzendo_g00f_h200.evaluator import (
            gate_from_assessment as h200_gate_from_assessment,
        )

    assessment = derive_h200_assessment(
        verified=verified,
        artifact_roots=artifact_roots,
        ledger_root=ledger_root,
        worker_result_receipts=worker_result_receipts,
        expected_provision_receipt_sha256=expected_provision_receipt_sha256,
        expected_pod_id=expected_pod_id,
    )
    return assessment, h200_gate_from_assessment(verified, assessment)


def _verify_route_gate(
    *,
    route: Route,
    path: str | Path,
    verified: Any,
    expected_gate_sha256: str,
) -> dict[str, Any]:
    _require_route_modules(Path(verified.repo), route)
    if _logical_absolute(path).is_symlink():
        raise BridgeError("final G00-F gate may not be a symlink")
    if route == "h100":
        from goalzendo_g00f.evaluator import verify_gate_artifact as verify_h100_gate

        return verify_h100_gate(
            path,
            verified=verified,
            expected_gate_sha256=expected_gate_sha256,
        )
    from goalzendo_g00f_h200.evaluator import verify_gate_artifact as verify_h200_gate

    return verify_h200_gate(
        path,
        verified=verified,
        expected_gate_sha256=expected_gate_sha256,
    )


def _artifact_roots_from_verified(verified: Any) -> dict[str, str]:
    expected: dict[str, str] = {}
    if set(verified.plans) != {"g00f-0p5b", "g00f-1p5b"}:
        raise BridgeError("verified G00-F freeze does not contain the exact two panels")
    for panel_id, rows in verified.plans.items():
        roots: set[str] = set()
        for row in rows:
            run_path = _require_no_symlink_components(
                row["artifact_path"],
                "frozen G00-F artifact path",
                require_exists=False,
            )
            roots.add(str(run_path.parents[2]))
        if len(roots) != 1:
            raise BridgeError("verified G00-F panel does not have one frozen artifact root")
        expected[str(panel_id)] = next(iter(roots))
    return expected


def produce_eligibility(
    *,
    repo: str | Path,
    route: Route,
    route_lock_path: str | Path,
    expected_route_lock_sha256: str,
    freeze_path: str | Path,
    expected_freeze_sha256: str,
    final_gate_path: str | Path,
    expected_final_gate_sha256: str,
    artifact_roots: Mapping[str, str | Path] | None,
    ledger_root: str | Path,
    worker_result_receipts: Mapping[int, str | Path] | None,
    expected_provision_receipt_sha256: str,
    expected_pod_id: str,
    expected_bridge_source_digest: str,
    output: str | Path,
    profile_selection_path: str | Path | None = None,
    expected_profile_selection_sha256: str | None = None,
) -> dict[str, Any]:
    """Replay one complete G00-F route and emit eligibility, never permission."""

    resolved = _repo_root(repo)
    selected_route = _route(route)
    output_path = _logical_absolute(output)
    if output_path.exists():
        raise BridgeError(f"refusing to overwrite eligibility sidecar: {output_path}")
    source = bridge_source_binding(resolved, expected_bridge_source_digest)
    g01 = g01_binding(resolved)
    g00e = historical_g00e_binding(resolved, g01)
    lock = verify_route_lock(
        route_lock_path,
        repo=resolved,
        expected_route_lock_sha256=expected_route_lock_sha256,
        expected_bridge_source_digest=expected_bridge_source_digest,
    )
    if selected_route != lock["route"]:
        raise BridgeError("requested eligibility route differs from the pre-ITT route lock")
    layout = _execution_layout(resolved, str(lock["execution_uuid"]), selected_route)
    if _logical_absolute(freeze_path) != layout.freeze:
        raise BridgeError("eligibility freeze path differs from the source-enforced layout")
    if require_sha256(expected_freeze_sha256, "G00-F freeze SHA-256") != lock["freeze"]["file_sha256"]:
        raise BridgeError("eligibility freeze digest differs from the pre-ITT route lock")
    if _logical_absolute(ledger_root) != layout.ledger_root:
        raise BridgeError("eligibility ITT ledger differs from the source-enforced layout")
    if _logical_absolute(final_gate_path) != layout.final_gate:
        raise BridgeError("submitted final gate path differs from the pre-ITT route lock")
    if output_path != layout.eligibility:
        raise BridgeError("eligibility output path differs from the source-enforced sidecar")
    canonical_worker_receipts = {
        index: layout.execution_root / f"worker-{index}-result.json" for index in range(4)
    }
    supplied_worker_receipts = (
        worker_result_receipts if worker_result_receipts is not None else canonical_worker_receipts
    )
    if set(supplied_worker_receipts) != set(range(4)) or any(
        _logical_absolute(supplied_worker_receipts[index]) != canonical_worker_receipts[index]
        for index in range(4)
    ):
        raise BridgeError("worker receipts differ from the four canonical execution paths")
    if selected_route == "h100":
        if profile_selection_path is not None or expected_profile_selection_sha256 is not None:
            raise BridgeError("H100 eligibility forbids H200 profile evidence")
    elif (
        layout.profile_selection is None
        or profile_selection_path is None
        or _logical_absolute(profile_selection_path) != layout.profile_selection
    ):
        raise BridgeError("H200 profile selection differs from its canonical execution path")
    verified = _verify_route_freeze(
        repo=resolved,
        route=selected_route,
        freeze_path=freeze_path,
        expected_freeze_sha256=expected_freeze_sha256,
        profile_selection_path=profile_selection_path,
        expected_profile_selection_sha256=expected_profile_selection_sha256,
        expected_provision_receipt_sha256=expected_provision_receipt_sha256,
        expected_pod_id=expected_pod_id,
    )
    frozen_artifact_roots = _artifact_roots_from_verified(verified)
    normalized_artifact_roots = (
        {str(key): str(Path(value).resolve()) for key, value in artifact_roots.items()}
        if artifact_roots is not None
        else frozen_artifact_roots
    )
    if normalized_artifact_roots != frozen_artifact_roots:
        raise BridgeError("artifact roots differ from the exact frozen G00-F panels")
    expected_gate_sha = require_sha256(expected_final_gate_sha256, "final G00-F gate SHA-256")
    submitted_summary = _verify_route_gate(
        route=selected_route,
        path=final_gate_path,
        verified=verified,
        expected_gate_sha256=expected_gate_sha,
    )
    assessment, regenerated_gate = _derive_route_assessment(
        route=selected_route,
        verified=verified,
        artifact_roots=normalized_artifact_roots,
        ledger_root=ledger_root,
        worker_result_receipts=supplied_worker_receipts,
        expected_provision_receipt_sha256=expected_provision_receipt_sha256,
        expected_pod_id=expected_pod_id,
    )
    submitted_gate = strict_json(final_gate_path, "submitted final G00-F gate")
    if (
        submitted_gate != regenerated_gate
        or Path(final_gate_path).read_bytes() != pretty_json_bytes(regenerated_gate)
        or sha256_file(final_gate_path) != expected_gate_sha
        or submitted_summary.get("overall_passed") is not True
        or regenerated_gate.get("overall_passed") is not True
    ):
        raise BridgeError("submitted final G00-F gate is not byte/object-identical to independent replay")
    validated = _validate_passing_assessment(
        route=selected_route,
        verified=verified,
        assessment=assessment,
        execution_uuid=str(lock["execution_uuid"]),
    )
    if _logical_absolute(str(validated["ledger"].get("ledger_root", ""))) != layout.ledger_root:
        raise BridgeError("replayed G00-F ledger escaped the source-enforced execution layout")
    budget_start = validated["ledger"].get("budget_start")
    if not isinstance(budget_start, Mapping):
        raise BridgeError("G00-F ledger omits its pre-ITT wall-budget timestamp")
    started_unix_ns = budget_start.get("started_unix_ns")
    lock_created_ns = int(
        _parse_utc(lock["created_at_utc"], "route-lock creation time").timestamp() * 1_000_000_000
    )
    if (
        isinstance(started_unix_ns, bool)
        or not isinstance(started_unix_ns, int)
        or lock_created_ns > started_unix_ns
    ):
        raise BridgeError("route lock was not created before the G00-F ITT ledger")
    evidence_files = _collect_evidence_bindings(
        repo=resolved,
        verified=verified,
        route_lock=lock,
        freeze_path=freeze_path,
        final_gate_path=final_gate_path,
        worker_result_receipts=supplied_worker_receipts,
        validated=validated,
        profile_selection_path=profile_selection_path,
    )
    # A second pass closes the ordinary read/verify/write race for every bound
    # file.  Full eligibility verification repeats these checks again.
    _verify_evidence_bindings(evidence_files, level="full")
    body = {
        "schema": ELIGIBILITY_SCHEMA,
        "schema_version": ELIGIBILITY_SCHEMA_VERSION,
        "study_id": "g01",
        "produced_at_utc": _utc_now(),
        "bridge_source": source,
        "route_lock": lock,
        "historical_g00e": g00e,
        "g00f": {
            "route": selected_route,
            "execution_uuid": lock["execution_uuid"],
            "freeze_file_sha256": verified.file_sha256,
            "freeze_digest": verified.digest,
            "final_gate": {
                "path": str(Path(final_gate_path).resolve()),
                "file_sha256": expected_gate_sha,
                "gate_digest": regenerated_gate["gate_digest"],
                "assessment_digest": assessment["assessment_digest"],
                "evidence_digest": assessment["evidence_digest"],
            },
            "artifact_roots": {str(key): value for key, value in sorted(normalized_artifact_roots.items())},
            "ledger_root": str(Path(ledger_root).resolve()),
            "worker_result_receipts": {
                str(index): str(Path(supplied_worker_receipts[index]).resolve()) for index in range(4)
            },
            "provision_receipt_sha256": require_sha256(
                expected_provision_receipt_sha256,
                "provision receipt SHA-256",
            ),
            "pod_id": str(expected_pod_id),
            "selected_profile": validated["selected_profile"],
            "profile_selection": (
                {
                    "path": str(Path(profile_selection_path).resolve()),
                    "file_sha256": require_sha256(
                        expected_profile_selection_sha256,
                        "H200 profile-selection SHA-256",
                    ),
                }
                if profile_selection_path is not None and expected_profile_selection_sha256 is not None
                else None
            ),
            "inventory": {
                "intention_to_train_rows": 160,
                "attempt_rows": 160,
                "outcome_seals": 480,
                "panel_unseal_rows": 480,
                "all_checks_passed": True,
            },
        },
        "evidence_files": evidence_files,
        "g01": g01,
        "eligibility": {
            "g00f_remediation_passed": True,
            "g01_scientifically_eligible": True,
            "direct_g01_launch_authorized": False,
            "dedicated_global_coordinator_required": True,
            "scope": "exact_unchanged_g01_primary_120_run_plan_only",
            "excluded_experiment_ids": ["g01a", "g01l", "g02"],
            "config_override_eligible": False,
            "backend_override_eligible": False,
            "smoke_or_partial_plan_eligible": False,
        },
    }
    payload = {**body, "eligibility_digest": semantic_digest(body)}
    _exclusive_json(output_path, payload)
    return {
        "path": str(output_path),
        "file_sha256": sha256_file(output_path),
        "eligibility_digest": payload["eligibility_digest"],
        "route": selected_route,
        "execution_uuid": lock["execution_uuid"],
        "selected_profile": validated["selected_profile"],
        "g01_scientifically_eligible": True,
        "direct_g01_launch_authorized": False,
        "dedicated_global_coordinator_required": True,
        "planned_runs": G01_PLANNED_RUNS,
    }


def verify_eligibility(
    path: str | Path,
    *,
    repo: str | Path,
    expected_eligibility_sha256: str,
    expected_route_lock_sha256: str,
    expected_bridge_source_digest: str,
    evidence_level: EvidenceLevel = "full",
) -> Eligibility:
    """Verify scientific eligibility while keeping direct execution false."""

    resolved = _repo_root(repo)
    target = _logical_absolute(path)
    expected_sha = require_sha256(expected_eligibility_sha256, "eligibility sidecar SHA-256")
    if (
        not target.is_file()
        or target.is_symlink()
        or target.stat().st_nlink != 1
        or stat.S_IMODE(target.stat().st_mode) != 0o400
        or sha256_file(target) != expected_sha
    ):
        raise BridgeError("eligibility sidecar bytes, type, link count, or mode changed")
    sidecar = strict_json(target, "G00-F/G01 eligibility sidecar")
    if target.read_bytes() != pretty_json_bytes(sidecar):
        raise BridgeError("eligibility sidecar is not in the one canonical byte encoding")
    body = {key: value for key, value in sidecar.items() if key != "eligibility_digest"}
    exact_keys = {
        "bridge_source",
        "evidence_files",
        "eligibility",
        "g00f",
        "g01",
        "historical_g00e",
        "produced_at_utc",
        "route_lock",
        "schema",
        "schema_version",
        "study_id",
    }
    current_source = bridge_source_binding(resolved, expected_bridge_source_digest)
    current_g01 = g01_binding(resolved)
    current_g00e = historical_g00e_binding(resolved, current_g01)
    route_lock_raw = sidecar.get("route_lock")
    if not isinstance(route_lock_raw, Mapping):
        raise BridgeError("eligibility sidecar omits its route lock")
    route_lock = verify_route_lock(
        route_lock_raw.get("path", ""),
        repo=resolved,
        expected_route_lock_sha256=expected_route_lock_sha256,
        expected_bridge_source_digest=expected_bridge_source_digest,
    )
    if target != _logical_absolute(str(route_lock["prospective_eligibility_output"])):
        raise BridgeError("eligibility is not at its pre-ITT canonical sidecar path")
    g00f = sidecar.get("g00f")
    eligibility = sidecar.get("eligibility")
    if not isinstance(g00f, Mapping):
        raise BridgeError("eligibility sidecar omits its G00-F evidence binding")
    route = _route(g00f.get("route"))
    execution = _uuid4(g00f.get("execution_uuid"), "eligibility execution UUID")
    layout = _execution_layout(resolved, execution, route)
    produced_at = _parse_utc(sidecar.get("produced_at_utc"), "eligibility production time")
    expected_eligibility = {
        "g00f_remediation_passed": True,
        "g01_scientifically_eligible": True,
        "direct_g01_launch_authorized": False,
        "dedicated_global_coordinator_required": True,
        "scope": "exact_unchanged_g01_primary_120_run_plan_only",
        "excluded_experiment_ids": ["g01a", "g01l", "g02"],
        "config_override_eligible": False,
        "backend_override_eligible": False,
        "smoke_or_partial_plan_eligible": False,
    }
    inventory = g00f.get("inventory")
    final_gate = g00f.get("final_gate")
    profile = g00f.get("profile_selection")
    selected_profile = g00f.get("selected_profile")
    exact_g00f_keys = {
        "artifact_roots",
        "execution_uuid",
        "final_gate",
        "freeze_digest",
        "freeze_file_sha256",
        "inventory",
        "ledger_root",
        "pod_id",
        "profile_selection",
        "provision_receipt_sha256",
        "route",
        "selected_profile",
        "worker_result_receipts",
    }
    if (
        set(body) != exact_keys
        or sidecar.get("schema") != ELIGIBILITY_SCHEMA
        or sidecar.get("schema_version") != ELIGIBILITY_SCHEMA_VERSION
        or sidecar.get("study_id") != "g01"
        or sidecar.get("eligibility_digest") != semantic_digest(body)
        or sidecar.get("bridge_source") != current_source
        or sidecar.get("g01") != current_g01
        or sidecar.get("historical_g00e") != current_g00e
        or set(g00f) != exact_g00f_keys
        or route_lock_raw != route_lock
        or route != route_lock["route"]
        or execution != route_lock["execution_uuid"]
        or g00f.get("freeze_file_sha256") != route_lock["freeze"]["file_sha256"]
        or g00f.get("freeze_digest") != route_lock["freeze"]["freeze_digest"]
        or _logical_absolute(str(g00f.get("ledger_root", ""))) != layout.ledger_root
        or eligibility != expected_eligibility
        or inventory
        != {
            "intention_to_train_rows": 160,
            "attempt_rows": 160,
            "outcome_seals": 480,
            "panel_unseal_rows": 480,
            "all_checks_passed": True,
        }
        or not isinstance(final_gate, Mapping)
        or set(final_gate) != {"assessment_digest", "evidence_digest", "file_sha256", "gate_digest", "path"}
        or _logical_absolute(str(final_gate.get("path", ""))) != layout.final_gate
    ):
        raise BridgeError("eligibility schema, route, inventory, or exact scope changed")
    if route == "h100":
        if selected_profile is not None or profile is not None:
            raise BridgeError("H100 eligibility contains H200 profile evidence")
    elif (
        selected_profile not in {"baseline", "tuned"}
        or not isinstance(profile, Mapping)
        or set(profile) != {"file_sha256", "path"}
        or layout.profile_selection is None
        or _logical_absolute(str(profile.get("path", ""))) != layout.profile_selection
    ):
        raise BridgeError("H200 eligibility lacks one selected profile")
    verified = _verify_route_freeze(
        repo=resolved,
        route=route,
        freeze_path=route_lock["freeze"]["path"],
        expected_freeze_sha256=route_lock["freeze"]["file_sha256"],
        profile_selection_path=profile.get("path") if isinstance(profile, Mapping) else None,
        expected_profile_selection_sha256=(
            profile.get("file_sha256") if isinstance(profile, Mapping) else None
        ),
        expected_provision_receipt_sha256=str(g00f.get("provision_receipt_sha256", "")),
        expected_pod_id=str(g00f.get("pod_id", "")),
    )
    if route == "h200" and getattr(verified, "selected_profile", None) != selected_profile:
        raise BridgeError("H200 selected profile changed since eligibility was produced")
    artifact_roots_raw = g00f.get("artifact_roots")
    worker_receipts_raw = g00f.get("worker_result_receipts")
    if (
        not isinstance(artifact_roots_raw, Mapping)
        or set(artifact_roots_raw) != {"g00f-0p5b", "g00f-1p5b"}
        or not isinstance(worker_receipts_raw, Mapping)
        or set(worker_receipts_raw) != {"0", "1", "2", "3"}
    ):
        raise BridgeError("eligibility route inputs are not the exact two panels/four workers")
    expected_artifact_roots = _artifact_roots_from_verified(verified)
    artifact_roots = {str(key): str(Path(str(value)).resolve()) for key, value in artifact_roots_raw.items()}
    worker_receipts = {
        int(key): str(Path(str(value)).resolve()) for key, value in worker_receipts_raw.items()
    }
    if artifact_roots != expected_artifact_roots or any(
        _logical_absolute(worker_receipts[index]) != layout.execution_root / f"worker-{index}-result.json"
        for index in range(4)
    ):
        raise BridgeError("eligibility artifact roots or worker receipts differ from frozen routes")
    gate_summary = _verify_route_gate(
        route=route,
        path=final_gate["path"],
        verified=verified,
        expected_gate_sha256=str(final_gate["file_sha256"]),
    )
    gate = strict_json(final_gate["path"], "eligibility-bound final G00-F gate")
    assessment = gate.get("assessment")
    if (
        gate_summary.get("overall_passed") is not True
        or gate.get("gate_digest") != final_gate.get("gate_digest")
        or not isinstance(assessment, Mapping)
        or assessment.get("assessment_digest") != final_gate.get("assessment_digest")
        or assessment.get("evidence_digest") != final_gate.get("evidence_digest")
    ):
        raise BridgeError("eligibility-bound final G00-F gate no longer proves a pass")
    if evidence_level == "full":
        replayed_assessment, replayed_gate = _derive_route_assessment(
            route=route,
            verified=verified,
            artifact_roots=artifact_roots,
            ledger_root=str(g00f["ledger_root"]),
            worker_result_receipts=worker_receipts,
            expected_provision_receipt_sha256=str(g00f["provision_receipt_sha256"]),
            expected_pod_id=str(g00f["pod_id"]),
        )
        if (
            replayed_gate != gate
            or Path(str(final_gate["path"])).read_bytes() != pretty_json_bytes(replayed_gate)
            or replayed_assessment != assessment
        ):
            raise BridgeError("eligibility-bound G00-F gate does not replay from route evidence")
        replayed_validation = _validate_passing_assessment(
            route=route,
            verified=verified,
            assessment=replayed_assessment,
            execution_uuid=execution,
        )
        if _logical_absolute(str(replayed_validation["ledger"].get("ledger_root", ""))) != layout.ledger_root:
            raise BridgeError("replayed G00-F ledger escaped the source-enforced layout")
        replayed_budget = replayed_validation["ledger"].get("budget_start")
        replayed_started_unix_ns = (
            replayed_budget.get("started_unix_ns") if isinstance(replayed_budget, Mapping) else None
        )
        lock_created_ns = int(
            _parse_utc(route_lock["created_at_utc"], "route-lock creation time").timestamp() * 1_000_000_000
        )
        if (
            isinstance(replayed_started_unix_ns, bool)
            or not isinstance(replayed_started_unix_ns, int)
            or lock_created_ns > replayed_started_unix_ns
        ):
            raise BridgeError("route lock does not precede the replayed G00-F ITT ledger")
        if int(produced_at.timestamp()) < replayed_started_unix_ns // 1_000_000_000:
            raise BridgeError("eligibility production time predates the replayed G00-F ledger")
    evidence = sidecar.get("evidence_files")
    if not isinstance(evidence, Mapping):
        raise BridgeError("eligibility sidecar lacks its evidence-file manifest")
    if evidence_level == "full":
        recomputed_evidence = _collect_evidence_bindings(
            repo=resolved,
            verified=verified,
            route_lock=route_lock,
            freeze_path=route_lock["freeze"]["path"],
            final_gate_path=final_gate["path"],
            worker_result_receipts=worker_receipts,
            validated=replayed_validation,
            profile_selection_path=(profile.get("path") if isinstance(profile, Mapping) else None),
        )
        if evidence != recomputed_evidence:
            raise BridgeError("eligibility evidence manifest is not the complete replayed dependency closure")
    if not isinstance(evidence.get("file_count"), int) or int(evidence["file_count"]) < 3_000:
        raise BridgeError("eligibility lacks the exact route-specific evidence-file inventory")
    _verify_evidence_bindings(evidence, level=evidence_level)
    return Eligibility(
        path=target,
        file_sha256=expected_sha,
        eligibility_digest=str(sidecar["eligibility_digest"]),
        route_lock_sha256=require_sha256(
            expected_route_lock_sha256,
            "route-lock SHA-256",
        ),
        route_lock_digest=str(route_lock["route_lock_digest"]),
        bridge_source_digest=require_sha256(
            expected_bridge_source_digest,
            "bridge source digest",
        ),
        route=route,
        execution_uuid=execution,
        selected_profile=str(selected_profile) if selected_profile is not None else None,
        plan_rows_digest=str(current_g01["plan"]["rows_digest"]),
        plan_key_set_digest=str(current_g01["plan"]["plan_key_set_digest"]),
        target_binding_digest=str(current_g01["target_binding_digest"]),
        model_name=str(current_g01["model_identity"]["requested_model"]),
        model_revision=str(current_g01["model_identity"]["requested_revision"]),
        scope=str(expected_eligibility["scope"]),
        g01_scientifically_eligible=True,
        direct_g01_launch_authorized=False,
        dedicated_global_coordinator_required=True,
    )


def _thin_g01_identity(binding: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "config_file_sha256": binding["config"]["file_sha256"],
        "canonical_config_digest": binding["config"]["canonical_digest"],
        "protocol_file_sha256": binding["protocol"]["file_sha256"],
        "runner_file_sha256": binding["runner"]["file_sha256"],
        "source_fingerprint": binding["source_fingerprint"],
        "guard_signature": binding["guard_signature"],
        "target_binding_digest": binding["target_binding_digest"],
        "plan_rows_digest": binding["plan"]["rows_digest"],
        "plan_key_set_digest": binding["plan"]["plan_key_set_digest"],
        "planned_runs": binding["plan"]["planned_runs"],
        "cell_count": binding["plan"]["cell_count"],
        "model_name": binding["model_identity"]["requested_model"],
        "model_revision": binding["model_identity"]["requested_revision"],
    }


_TOKEN_FORBIDDEN_KEYS = frozenset(
    {
        "assessment",
        "checks",
        "evidence",
        "evidence_files",
        "final_gate",
        "historical_g00e",
        "inventory",
        "outcome_files",
        "profile",
        "profile_selection",
        "selected_profile",
    }
)
_TOKEN_FORBIDDEN_KEY_FRAGMENTS = (
    "accuracy",
    "artifact",
    "assessment",
    "attempt",
    "check",
    "evidence",
    "final_gate",
    "inventory",
    "ledger",
    "loss",
    "metric",
    "outcome",
    "prediction",
    "profile",
    "provision",
    "reward",
    "score",
    "seal",
    "summary",
    "unseal",
    "worker",
)
_TOKEN_ALLOWED_NUMERIC_KEYS = frozenset({"cell_count", "planned_runs", "schema_version"})
_THIN_FORBIDDEN_PROJECT_MODULES = frozenset(
    {
        "goalzendo.experiment",
        "goalzendo.hf",
        "goalzendo.modeling",
        "goalzendo.training",
    }
)


def assert_thin_runtime_boundary() -> None:
    """Require that no GoalZendo model/backend module has entered the process."""

    present = sorted(
        name
        for name in sys.modules
        if any(
            name == forbidden or name.startswith(f"{forbidden}.")
            for forbidden in _THIN_FORBIDDEN_PROJECT_MODULES
        )
    )
    if present:
        raise BridgeError(
            "thin coordinator verification must precede every GoalZendo backend/model import: "
            + ", ".join(present)
        )


def _assert_thin_token(value: Any, *, parent_key: str | None = None) -> None:
    if isinstance(value, Mapping):
        for raw_key, nested in value.items():
            key = str(raw_key)
            normalized = key.casefold()
            if key in _TOKEN_FORBIDDEN_KEYS or any(
                fragment in normalized for fragment in _TOKEN_FORBIDDEN_KEY_FRAGMENTS
            ):
                raise BridgeError(f"coordinator token contains forbidden detailed field {key!r}")
            _assert_thin_token(nested, parent_key=key)
    elif isinstance(value, list):
        for nested in value:
            _assert_thin_token(nested, parent_key=parent_key)
    elif (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and parent_key not in _TOKEN_ALLOWED_NUMERIC_KEYS
    ):
        raise BridgeError(f"coordinator token contains a forbidden numerical value under {parent_key!r}")


def project_coordinator_token(
    eligibility_path: str | Path,
    *,
    repo: str | Path,
    expected_eligibility_sha256: str,
    expected_route_lock_sha256: str,
    expected_bridge_source_digest: str,
    output: str | Path,
) -> dict[str, Any]:
    """Project fully verified detailed evidence into one outcome-free token."""

    verified = verify_eligibility(
        eligibility_path,
        repo=repo,
        expected_eligibility_sha256=expected_eligibility_sha256,
        expected_route_lock_sha256=expected_route_lock_sha256,
        expected_bridge_source_digest=expected_bridge_source_digest,
        evidence_level="full",
    )
    resolved = _repo_root(repo)
    layout = _execution_layout(resolved, verified.execution_uuid, verified.route)
    target = _logical_absolute(output)
    if target != layout.coordinator_token:
        raise BridgeError("coordinator token output differs from the source-enforced path")
    if target.exists() or target.is_symlink():
        raise BridgeError(f"refusing to overwrite coordinator token: {target}")
    current_g01 = g01_binding(resolved)
    body = {
        "schema": COORDINATOR_TOKEN_SCHEMA,
        "schema_version": COORDINATOR_TOKEN_SCHEMA_VERSION,
        "study_id": "g01",
        "produced_at_utc": _utc_now(),
        "eligibility": {
            "g01_scientifically_eligible": True,
            "direct_g01_launch_authorized": False,
            "dedicated_global_coordinator_required": True,
            "scope": verified.scope,
        },
        "eligibility_sidecar": {
            "file_sha256": verified.file_sha256,
            "eligibility_digest": verified.eligibility_digest,
        },
        "route_lock": {
            "file_sha256": verified.route_lock_sha256,
            "route_lock_digest": verified.route_lock_digest,
        },
        "bridge_source_digest": verified.bridge_source_digest,
        "g00f_execution": {
            "route": verified.route,
            "execution_uuid": verified.execution_uuid,
        },
        "g01_identity": _thin_g01_identity(current_g01),
    }
    _assert_thin_token(body)
    payload = {**body, "token_digest": semantic_digest(body)}
    _exclusive_json(target, payload)
    return {
        "path": str(target),
        "file_sha256": sha256_file(target),
        "token_digest": payload["token_digest"],
        "g01_scientifically_eligible": True,
        "direct_g01_launch_authorized": False,
        "dedicated_global_coordinator_required": True,
    }


def verify_coordinator_token(
    path: str | Path,
    *,
    repo: str | Path,
    expected_token_sha256: str,
    expected_bridge_source_digest: str,
) -> CoordinatorToken:
    """Verify only the thin token; never open detailed G00-F evidence."""

    assert_thin_runtime_boundary()
    resolved = _repo_root(repo)
    target = _logical_absolute(path)
    expected_sha = require_sha256(expected_token_sha256, "coordinator token SHA-256")
    if (
        not target.is_file()
        or target.is_symlink()
        or target.stat().st_nlink != 1
        or stat.S_IMODE(target.stat().st_mode) != 0o400
        or sha256_file(target) != expected_sha
    ):
        raise BridgeError("coordinator token bytes, type, link count, or mode changed")
    token = strict_json(target, "G01 coordinator-input token")
    if target.read_bytes() != pretty_json_bytes(token):
        raise BridgeError("coordinator token is not in the one canonical byte encoding")
    _assert_thin_token(token)
    body = {key: value for key, value in token.items() if key != "token_digest"}
    exact_keys = {
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
    execution_raw = token.get("g00f_execution")
    if not isinstance(execution_raw, Mapping):
        raise BridgeError("coordinator token omits its execution identity")
    route = _route(execution_raw.get("route"))
    execution = _uuid4(execution_raw.get("execution_uuid"), "coordinator-token execution UUID")
    layout = _execution_layout(resolved, execution, route)
    source = bridge_source_binding(resolved, expected_bridge_source_digest)
    g01 = g01_binding(resolved)
    eligibility_raw = token.get("eligibility")
    sidecar_raw = token.get("eligibility_sidecar")
    route_lock_raw = token.get("route_lock")
    expected_eligibility = {
        "g01_scientifically_eligible": True,
        "direct_g01_launch_authorized": False,
        "dedicated_global_coordinator_required": True,
        "scope": "exact_unchanged_g01_primary_120_run_plan_only",
    }
    if (
        set(body) != exact_keys
        or token.get("schema") != COORDINATOR_TOKEN_SCHEMA
        or token.get("schema_version") != COORDINATOR_TOKEN_SCHEMA_VERSION
        or token.get("study_id") != "g01"
        or token.get("token_digest") != semantic_digest(body)
        or target != layout.coordinator_token
        or token.get("bridge_source_digest") != source["source_digest"]
        or eligibility_raw != expected_eligibility
        or token.get("g01_identity") != _thin_g01_identity(g01)
        or set(execution_raw) != {"execution_uuid", "route"}
        or not isinstance(sidecar_raw, Mapping)
        or set(sidecar_raw) != {"eligibility_digest", "file_sha256"}
        or not isinstance(route_lock_raw, Mapping)
        or set(route_lock_raw) != {"file_sha256", "route_lock_digest"}
    ):
        raise BridgeError("coordinator token schema, identities, or false-launch boundary changed")
    _parse_utc(token.get("produced_at_utc"), "coordinator token production time")
    eligibility_sha = require_sha256(sidecar_raw["file_sha256"], "eligibility sidecar SHA-256")
    eligibility_digest = require_sha256(sidecar_raw["eligibility_digest"], "eligibility digest")
    route_lock_sha = require_sha256(route_lock_raw["file_sha256"], "route-lock SHA-256")
    route_lock_digest = require_sha256(route_lock_raw["route_lock_digest"], "route-lock digest")
    result = CoordinatorToken(
        path=target,
        file_sha256=expected_sha,
        token_digest=str(token["token_digest"]),
        eligibility_file_sha256=eligibility_sha,
        eligibility_digest=eligibility_digest,
        route_lock_sha256=route_lock_sha,
        route_lock_digest=route_lock_digest,
        bridge_source_digest=str(source["source_digest"]),
        route=route,
        execution_uuid=execution,
        target_binding_digest=str(g01["target_binding_digest"]),
        plan_rows_digest=str(g01["plan"]["rows_digest"]),
        plan_key_set_digest=str(g01["plan"]["plan_key_set_digest"]),
        g01_scientifically_eligible=True,
        direct_g01_launch_authorized=False,
        dedicated_global_coordinator_required=True,
    )
    assert_thin_runtime_boundary()
    return result


def exact_g01_plan(repo: str | Path) -> tuple[Any, ...]:
    """Return the one exact 120-row plan after all immutable binding checks."""

    assert_thin_runtime_boundary()
    from goalzendo.config import load_config
    from goalzendo.runner import build_plan

    resolved = _repo_root(repo)
    binding = g01_binding(resolved)
    config = load_config(resolved / G01_CONFIG_RELATIVE_PATH)
    plan = build_plan(config)
    rows = [spec.as_dict() for spec in plan]
    if len(plan) != G01_PLANNED_RUNS or semantic_digest(rows) != binding["plan"]["rows_digest"]:
        raise BridgeError("rebuilt G01 plan is not the exact eligible 120-row membership")
    result = tuple(plan)
    assert_thin_runtime_boundary()
    return result


def exact_g01_shard(repo: str | Path, *, shard_index: int, num_shards: int) -> tuple[Any, ...]:
    """Select one deterministic shard while proving the union remains all 120 rows."""

    if isinstance(num_shards, bool) or not isinstance(num_shards, int) or num_shards < 1:
        raise BridgeError("num_shards must be a positive integer")
    if isinstance(shard_index, bool) or not isinstance(shard_index, int) or not 0 <= shard_index < num_shards:
        raise BridgeError("shard_index must lie in [0, num_shards)")
    full = exact_g01_plan(repo)
    shards = [
        tuple(spec for spec in full if int(str(spec.plan_key)[:16], 16) % num_shards == index)
        for index in range(num_shards)
    ]
    union = [spec for shard in shards for spec in shard]
    if (
        len(union) != G01_PLANNED_RUNS
        or len({str(spec.plan_key) for spec in union}) != G01_PLANNED_RUNS
        or {str(spec.plan_key) for spec in union} != {str(spec.plan_key) for spec in full}
    ):
        raise BridgeError("G01 shard union duplicates, drops, or adds plan membership")
    return shards[shard_index]


def verify_exact_g01_spec(
    coordinator_token_path: str | Path,
    *,
    repo: str | Path,
    spec: Any,
    expected_coordinator_token_sha256: str,
    expected_bridge_source_digest: str,
) -> CoordinatorToken:
    """Verify the thin token and one exact plan member without detailed evidence.

    This is a future checkpoint-B helper only.  It never opens the eligibility
    sidecar, route lock, final gate, or detailed G00-F evidence.
    """

    eligible = verify_coordinator_token(
        coordinator_token_path,
        repo=repo,
        expected_token_sha256=expected_coordinator_token_sha256,
        expected_bridge_source_digest=expected_bridge_source_digest,
    )
    assert_exact_g01_spec(repo, spec)
    return eligible


def assert_exact_g01_spec(repo: str | Path, spec: Any) -> None:
    """Reject a fabricated, mutated, smoke, or out-of-plan RunSpec."""

    from goalzendo.config import canonical_config

    expected_by_key = {str(row.plan_key): row for row in exact_g01_plan(repo)}
    plan_key = str(getattr(spec, "plan_key", ""))
    expected = expected_by_key.get(plan_key)
    if (
        expected is None
        or spec.as_dict() != expected.as_dict()
        or canonical_config(spec.config) != canonical_config(expected.config)
        or int(spec.seed) != int(expected.seed)
    ):
        raise BridgeError("requested RunSpec is not one exact member of the frozen G01 plan")


__all__ = [
    "BRIDGE_SOURCE_RELATIVE_PATHS",
    "COORDINATOR_TOKEN_SCHEMA",
    "ELIGIBILITY_SCHEMA",
    "G01_PLANNED_RUNS",
    "ROUTE_LOCK_SCHEMA",
    "BridgeError",
    "CoordinatorToken",
    "Eligibility",
    "assert_exact_g01_spec",
    "assert_thin_runtime_boundary",
    "bridge_source_binding",
    "calculate_bridge_source_binding",
    "create_route_lock",
    "exact_g01_plan",
    "exact_g01_shard",
    "execution_layout",
    "g01_binding",
    "historical_g00e_binding",
    "produce_eligibility",
    "project_coordinator_token",
    "require_entrypoint_file",
    "sha256_file",
    "verify_coordinator_token",
    "verify_eligibility",
    "verify_exact_g01_spec",
    "verify_route_lock",
]
