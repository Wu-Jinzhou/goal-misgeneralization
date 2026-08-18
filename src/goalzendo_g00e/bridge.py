"""Fail-closed numerical bridge for the frozen G00-D launch gate.

The G00-D source is immutable.  Its exact-binomial acceptance helper first
materializes floating-point probability masses and raises ``OverflowError``
for the completed capability-panel sizes.  This module authenticates that
exact frozen source and then replaces only that private helper, in-process,
with the algebraically identical integer calculation for the preregistered
``Binomial(n, 1/2)`` central 95% acceptance interval.

The legacy gate creator and verifier remain authoritative.  In particular,
this bridge does not replace, catch, or weaken any of the six gate checks, the
evidence replay, target binding, optimizer selection, or G01 launch guard.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

FROZEN_GOALZENDO_IMPLEMENTATION_FINGERPRINT = (
    "1a8146377b9a9620690025671614edb2dd20d214f4e195da3cf3528809b2c694"
)
FROZEN_RUNNER_FILE_SHA256 = (
    "46b55ad4bdd08073e5f89ae101e0862372817b74b590f07ddb8f8c4331e5b9e1"
)
FROZEN_BINOMIAL_CALLABLE_SOURCE_SHA256 = (
    "35f8c9318d5c41d5cc539ef76825cb01951c1c7e6a26dc6b34aed91a6150fa5f"
)
FROZEN_BINOMIAL_CALLABLE_MODULE = "goalzendo.runner"
FROZEN_BINOMIAL_CALLABLE_NAME = "_binomial_acceptance_interval"
FROZEN_BINOMIAL_CALLABLE_SIGNATURE = (
    "(trials: 'int', *, probability: 'float', alpha: 'float') -> 'tuple[int, int]'"
)
FROZEN_GATE_SCHEMA = "goalzendo.g00_gate"
FROZEN_GATE_SCHEMA_VERSION = 2
FROZEN_ASSESSMENT_SCHEMA = "goalzendo.g00_assessment"
FROZEN_ASSESSMENT_SCHEMA_VERSION = 2
BRIDGE_SIDECAR_SCHEMA = "goalzendo.g00e_numerical_gate_bridge_sidecar"
BRIDGE_SIDECAR_SCHEMA_VERSION = 3
BRIDGE_MANIFEST_SCHEMA = "goalzendo.g00e_numerical_gate_bridge_manifest"
BRIDGE_MANIFEST_SCHEMA_VERSION = 1
NON_OVERFLOW_EQUIVALENCE_MIN_TRIALS = 1
NON_OVERFLOW_EQUIVALENCE_MAX_TRIALS = 1029
FIRST_LEGACY_OVERFLOW_TRIALS = 1030
EQUIVALENCE_AUDIT_DIGEST = "e80049fbd121092dc95ef12faf5103e7ddaa9c35f3d4ec03b19b963ab37bf88b"
PINNED_ACTUAL_INTERVALS: Mapping[int, tuple[int, int]] = {
    1536: (730, 806),
    3072: (1482, 1590),
}
REQUIRED_GATE_CHECKS = frozenset(
    {
        "rule_adapters",
        "constrained_scorer",
        "no_signal_chance",
        "surface_leakage",
        "optimizer_stability",
        "dataset_integrity",
    }
)
FROZEN_THRESHOLDS: Mapping[str, Any] = {
    "minimum_rule_adapter_position_agreement": 0.95,
    "minimum_finite_probability_fraction": 1.0,
    "maximum_probability_sum_error": 1e-6,
    "maximum_absolute_position_bias": 0.02,
    "binomial_alpha": 0.05,
    "chance_probability": 0.50,
    "minimum_iid_improvement": 0.10,
    "minimum_terminal_iid_accuracy": 0.90,
    "maximum_high_baseline_regression": 0.02,
    "maximum_terminal_accuracy_drop": 0.05,
    "maximum_terminal_action_frequency": 0.95,
    "minimum_rl_early_both_actions_fraction": 0.25,
    "maximum_rl_early_all_zero_advantages_fraction": 0.75,
    "maximum_integrity_mismatches": 0,
}
FROZEN_OPTIMIZER_POLICY: Mapping[str, Any] = {
    "baseline_step": 0,
    "terminal_steps": [128, 256],
    "rl_sampling_steps": list(range(1, 33)),
    "required_law_families": ["majority", "parity"],
    "required_proxy_accuracies": [0.95, 1.0],
    "independent_seed_count": 3,
    "selection_rule": "minimum_learning_rate_then_entropy",
}


class BridgeError(RuntimeError):
    """Raised before a bridge can alter or authorize a legacy gate process."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _json_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _require_sha256(value: str, label: str) -> str:
    normalized = str(value)
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise BridgeError(f"{label} must be a lowercase SHA-256 digest")
    return normalized


def _strict_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise BridgeError(f"{label} does not exist: {path}")

    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise BridgeError(f"{label} contains duplicate JSON key {key!r}")
            value[key] = item
        return value

    try:
        parsed = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BridgeError(f"{label} is not strict UTF-8 JSON: {path}") from error
    if not isinstance(parsed, dict):
        raise BridgeError(f"{label} must contain one JSON object")
    return parsed


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def exact_central_binomial_interval(
    trials: int,
    *,
    probability: float,
    alpha: float,
) -> tuple[int, int]:
    """Compute the frozen central Binomial(n, .5), alpha=.05 interval exactly."""

    if type(trials) is not int or trials < 1:
        raise BridgeError("G00-E exact binomial checks require a positive integer trial count")
    if probability != 0.5 or alpha != 0.05:
        raise BridgeError(
            "G00-E is defined only for the frozen chance_probability=.5 and binomial_alpha=.05"
        )

    # For X ~ Binomial(n, 1/2), P(X <= k) has integer numerator
    # sum_{i=0}^k C(n,i) and denominator 2**n.  The two frozen tail
    # thresholds are exactly 1/40 and 39/40, so no float is needed.
    denominator = 1 << trials
    coefficient = 1
    cumulative = 0
    lower: int | None = None
    for successes in range(trials + 1):
        cumulative += coefficient
        if lower is None and 40 * cumulative >= denominator:
            lower = successes
        if 40 * cumulative >= 39 * denominator:
            if lower is None:  # pragma: no cover - impossible for a valid distribution
                raise AssertionError("upper exact-binomial boundary preceded lower boundary")
            return lower, successes
        if successes < trials:
            coefficient = coefficient * (trials - successes) // (successes + 1)
    raise AssertionError("exact binomial cumulative mass did not reach one")


def _equivalence_digest(
    old_callable: Callable[..., tuple[int, int]],
) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"goalzendo-g00e-exhaustive-equivalence-v1\0")
    for trials in range(
        NON_OVERFLOW_EQUIVALENCE_MIN_TRIALS,
        NON_OVERFLOW_EQUIVALENCE_MAX_TRIALS + 1,
    ):
        old_interval = old_callable(trials, probability=0.5, alpha=0.05)
        new_interval = exact_central_binomial_interval(trials, probability=0.5, alpha=0.05)
        if old_interval != new_interval:
            raise BridgeError(
                f"legacy/exact interval mismatch at n={trials}: {old_interval!r} != {new_interval!r}"
            )
        hasher.update(_canonical_json([trials, old_interval[0], old_interval[1]]))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _bridge_package_root() -> Path:
    return Path(__file__).resolve().parent


def verify_bridge_source_manifest(expected_manifest_sha256: str) -> dict[str, Any]:
    """Authenticate every bridge-package source byte against its shipped manifest."""

    expected_digest = _require_sha256(
        expected_manifest_sha256,
        "expected G00-E bridge manifest digest",
    )
    package_root = _bridge_package_root()
    manifest_path = package_root / "bridge_manifest.json"
    manifest_bytes = manifest_path.read_bytes() if manifest_path.is_file() else b""
    observed_manifest_sha256 = _sha256_bytes(manifest_bytes)
    if observed_manifest_sha256 != expected_digest:
        raise BridgeError("G00-E bridge manifest bytes do not match the externally expected digest")
    manifest = _strict_json(manifest_path, "G00-E bridge manifest")
    if (
        manifest.get("schema") != BRIDGE_MANIFEST_SCHEMA
        or manifest.get("schema_version") != BRIDGE_MANIFEST_SCHEMA_VERSION
    ):
        raise BridgeError("G00-E bridge manifest schema is not recognized")
    raw_files = manifest.get("source_files")
    if not isinstance(raw_files, Mapping) or not raw_files:
        raise BridgeError("G00-E bridge manifest has no source-file map")
    expected_names = {"__init__.py", "bridge.py", "cli.py", "py.typed"}
    if set(raw_files) != expected_names:
        raise BridgeError("G00-E bridge manifest source-file set changed")
    source_files: dict[str, str] = {}
    for raw_name, raw_digest in sorted(raw_files.items()):
        name = str(raw_name)
        if Path(name).name != name:
            raise BridgeError("G00-E bridge manifest contains a non-local source path")
        digest = _require_sha256(str(raw_digest), f"G00-E bridge source digest for {name}")
        source_path = package_root / name
        if not source_path.is_file() or _sha256_file(source_path) != digest:
            raise BridgeError(f"G00-E bridge source bytes changed: {name}")
        source_files[name] = digest
    source_digest = _json_digest(source_files)
    if manifest.get("source_digest") != source_digest:
        raise BridgeError("G00-E bridge source aggregate digest mismatch")
    return {
        "manifest": manifest,
        "manifest_sha256": observed_manifest_sha256,
        "source_digest": source_digest,
        "source_files": source_files,
    }


def _resolve_repo(repo: str | Path | None, runner_module: ModuleType) -> Path:
    imported_source = Path(str(runner_module.__file__)).resolve()
    imported_repo = imported_source.parents[2]
    target = imported_repo if repo is None else Path(repo).resolve()
    expected_source = target / "src" / "goalzendo" / "runner.py"
    if expected_source.resolve() != imported_source:
        raise BridgeError("requested repository does not contain the imported frozen GoalZendo runner")
    return target


def authenticate_legacy_and_bridge(
    *,
    repo: str | Path | None,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    """Verify the exact old source/callable before any in-process correction."""

    bridge_identity = verify_bridge_source_manifest(expected_manifest_sha256)

    import goalzendo.artifacts as artifacts_module
    import goalzendo.runner as runner_module

    resolved_repo = _resolve_repo(repo, runner_module)
    provenance = artifacts_module.implementation_provenance(resolved_repo)
    if provenance.get("implementation_fingerprint") != FROZEN_GOALZENDO_IMPLEMENTATION_FINGERPRINT:
        raise BridgeError("imported GoalZendo implementation fingerprint is not frozen G00-D")
    runner_path = Path(str(runner_module.__file__)).resolve()
    if _sha256_file(runner_path) != FROZEN_RUNNER_FILE_SHA256:
        raise BridgeError("frozen GoalZendo runner.py byte digest changed")

    old_callable = getattr(runner_module, FROZEN_BINOMIAL_CALLABLE_NAME, None)
    if (
        not inspect.isfunction(old_callable)
        or old_callable.__module__ != FROZEN_BINOMIAL_CALLABLE_MODULE
        or old_callable.__name__ != FROZEN_BINOMIAL_CALLABLE_NAME
        or old_callable.__qualname__ != FROZEN_BINOMIAL_CALLABLE_NAME
        or str(inspect.signature(old_callable)) != FROZEN_BINOMIAL_CALLABLE_SIGNATURE
    ):
        raise BridgeError("frozen binomial helper callable identity changed")
    try:
        callable_source = inspect.getsource(old_callable)
    except (OSError, TypeError) as error:
        raise BridgeError("could not recover the frozen binomial helper source") from error
    if _sha256_bytes(callable_source.encode("utf-8")) != FROZEN_BINOMIAL_CALLABLE_SOURCE_SHA256:
        raise BridgeError("frozen binomial helper source digest changed")

    if runner_module.G00_GATE_SCHEMA_VERSION != FROZEN_GATE_SCHEMA_VERSION:
        raise BridgeError("legacy G00 gate schema version changed")
    thresholds = runner_module.G00_GATE_THRESHOLDS.as_dict()
    policy = runner_module.G00_OPTIMIZER_STABILITY_POLICY.as_dict()
    if thresholds != FROZEN_THRESHOLDS:
        raise BridgeError("legacy G00 thresholds differ from the frozen preregistration")
    if policy != FROZEN_OPTIMIZER_POLICY:
        raise BridgeError("legacy G00 optimizer policy differs from the frozen preregistration")

    equivalence_digest = _equivalence_digest(old_callable)
    if equivalence_digest != EQUIVALENCE_AUDIT_DIGEST:
        raise BridgeError("exhaustive non-overflow equivalence digest changed")
    try:
        old_callable(FIRST_LEGACY_OVERFLOW_TRIALS, probability=0.5, alpha=0.05)
    except OverflowError:
        pass
    else:
        raise BridgeError("legacy binomial helper no longer has the pinned first-overflow boundary")
    observed_actual = {
        trials: exact_central_binomial_interval(trials, probability=0.5, alpha=0.05)
        for trials in PINNED_ACTUAL_INTERVALS
    }
    if observed_actual != PINNED_ACTUAL_INTERVALS:
        raise BridgeError("exact G00-D capability-panel intervals changed")

    return {
        "repo": str(resolved_repo),
        "legacy": {
            "implementation_fingerprint": FROZEN_GOALZENDO_IMPLEMENTATION_FINGERPRINT,
            "runner_file_sha256": FROZEN_RUNNER_FILE_SHA256,
            "binomial_callable_module": FROZEN_BINOMIAL_CALLABLE_MODULE,
            "binomial_callable_name": FROZEN_BINOMIAL_CALLABLE_NAME,
            "binomial_callable_signature": FROZEN_BINOMIAL_CALLABLE_SIGNATURE,
            "binomial_callable_source_sha256": FROZEN_BINOMIAL_CALLABLE_SOURCE_SHA256,
            "gate_schema_version": FROZEN_GATE_SCHEMA_VERSION,
            "thresholds": dict(FROZEN_THRESHOLDS),
            "threshold_digest": _json_digest(FROZEN_THRESHOLDS),
            "optimizer_policy": dict(FROZEN_OPTIMIZER_POLICY),
            "optimizer_policy_digest": _json_digest(FROZEN_OPTIMIZER_POLICY),
        },
        "numerical_audit": {
            "distribution": "Binomial(n, 1/2)",
            "alpha_fraction": "1/20",
            "lower_tail_fraction": "1/40",
            "upper_cdf_fraction": "39/40",
            "integer_acceptance_test": "40*cumulative_binomial_coefficient >= {1,39}*2**n",
            "equivalence_trials_inclusive": [
                NON_OVERFLOW_EQUIVALENCE_MIN_TRIALS,
                NON_OVERFLOW_EQUIVALENCE_MAX_TRIALS,
            ],
            "equivalence_digest": equivalence_digest,
            "first_legacy_overflow_trials": FIRST_LEGACY_OVERFLOW_TRIALS,
            "pinned_actual_intervals": {
                str(trials): list(interval) for trials, interval in PINNED_ACTUAL_INTERVALS.items()
            },
        },
        "bridge": bridge_identity,
    }


def install_exact_interval_correction(
    authenticated_identity: Mapping[str, Any],
) -> Callable[..., tuple[int, int]]:
    """Replace only the authenticated private interval helper in this process."""

    import goalzendo.runner as runner_module

    if authenticated_identity.get("legacy", {}).get(
        "implementation_fingerprint"
    ) != FROZEN_GOALZENDO_IMPLEMENTATION_FINGERPRINT:
        raise BridgeError("cannot install G00-E from an unauthenticated identity")
    current = getattr(runner_module, FROZEN_BINOMIAL_CALLABLE_NAME, None)
    if not inspect.isfunction(current):
        raise BridgeError("legacy binomial helper is absent before G00-E installation")
    try:
        current_source_sha256 = _sha256_bytes(inspect.getsource(current).encode("utf-8"))
    except (OSError, TypeError) as error:
        raise BridgeError("legacy binomial helper cannot be authenticated at installation") from error
    if current_source_sha256 != FROZEN_BINOMIAL_CALLABLE_SOURCE_SHA256:
        raise BridgeError("legacy binomial helper changed between authentication and installation")
    setattr(runner_module, FROZEN_BINOMIAL_CALLABLE_NAME, exact_central_binomial_interval)
    return current


def _validated_assessment(path: Path) -> tuple[dict[str, Any], str]:
    assessment = _strict_json(path, "G00-D assessment artifact")
    body = {key: value for key, value in assessment.items() if key != "assessment_digest"}
    if (
        assessment.get("schema") != FROZEN_ASSESSMENT_SCHEMA
        or assessment.get("schema_version") != FROZEN_ASSESSMENT_SCHEMA_VERSION
        or assessment.get("assessment_digest") != _json_digest(body)
    ):
        raise BridgeError("G00-D assessment schema or semantic digest is invalid")
    return assessment, _sha256_file(path)


def _validated_gate(
    path: Path,
    *,
    assessment: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    gate = _strict_json(path, "G00-D gate artifact")
    body = {key: value for key, value in gate.items() if key != "gate_digest"}
    checks = gate.get("checks")
    if (
        gate.get("schema") != FROZEN_GATE_SCHEMA
        or gate.get("schema_version") != FROZEN_GATE_SCHEMA_VERSION
        or gate.get("gate_digest") != _json_digest(body)
        or gate.get("thresholds") != FROZEN_THRESHOLDS
        or gate.get("threshold_digest") != _json_digest(FROZEN_THRESHOLDS)
        or gate.get("optimizer_stability_policy") != FROZEN_OPTIMIZER_POLICY
        or gate.get("optimizer_stability_policy_digest") != _json_digest(FROZEN_OPTIMIZER_POLICY)
        or gate.get("assessment") != assessment
        or gate.get("overall_passed") is not True
        or not isinstance(checks, Mapping)
        or set(checks) != REQUIRED_GATE_CHECKS
        or any(
            not isinstance(checks[name], Mapping) or checks[name].get("passed") is not True
            for name in REQUIRED_GATE_CHECKS
        )
    ):
        raise BridgeError(
            "G00-D gate is not a passing, exact six-check artifact bound to the assessment"
        )
    return gate, _sha256_file(path)


def _script_identities(repo: Path) -> dict[str, str]:
    paths = {
        "bridge_cli": repo / "src" / "goalzendo_g00e" / "cli.py",
        "runpod_entrypoint": repo / "runs" / "goalzendo" / "runpod_entrypoint.sh",
        "runpod_dispatch": repo / "runs" / "goalzendo" / "runpod_dispatch.sh",
    }
    if any(not path.is_file() for path in paths.values()):
        raise BridgeError("G00-E CLI/Runpod script identity inputs are incomplete")
    return {name: _sha256_file(path) for name, path in paths.items()}


def create_v3_sidecar(
    *,
    gate_path: str | Path,
    assessment_path: str | Path,
    output_path: str | Path,
    authenticated_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind successful legacy gate and assessment bytes to bridge source identity."""

    output = Path(output_path).resolve()
    if output.exists():
        raise BridgeError(f"refusing to overwrite an existing G00-E v3 sidecar: {output}")
    gate_target = Path(gate_path).resolve()
    assessment_target = Path(assessment_path).resolve()
    assessment, assessment_sha256 = _validated_assessment(assessment_target)
    gate, gate_sha256 = _validated_gate(gate_target, assessment=assessment)
    repo = Path(str(authenticated_identity.get("repo", ""))).resolve()
    bridge = authenticated_identity.get("bridge")
    legacy = authenticated_identity.get("legacy")
    numerical = authenticated_identity.get("numerical_audit")
    if not isinstance(bridge, Mapping) or not isinstance(legacy, Mapping) or not isinstance(
        numerical, Mapping
    ):
        raise BridgeError("authenticated G00-E identity is incomplete")
    body: dict[str, Any] = {
        "schema": BRIDGE_SIDECAR_SCHEMA,
        "schema_version": BRIDGE_SIDECAR_SCHEMA_VERSION,
        "legacy_identity": dict(legacy),
        "numerical_audit": dict(numerical),
        "bridge_source": {
            "manifest": bridge.get("manifest"),
            "manifest_sha256": bridge.get("manifest_sha256"),
            "source_digest": bridge.get("source_digest"),
            "source_files": bridge.get("source_files"),
        },
        "execution_scripts": _script_identities(repo),
        "assessment_artifact": {
            "path": str(assessment_target),
            "file_sha256": assessment_sha256,
            "assessment_digest": assessment.get("assessment_digest"),
        },
        "gate_artifact": {
            "path": str(gate_target),
            "file_sha256": gate_sha256,
            "gate_digest": gate.get("gate_digest"),
            "embedded_assessment_digest": gate.get("assessment", {}).get("assessment_digest"),
            "required_checks": sorted(REQUIRED_GATE_CHECKS),
            "overall_passed": gate.get("overall_passed"),
        },
    }
    payload = {**body, "sidecar_digest": _json_digest(body)}
    _atomic_json(output, payload)
    return {
        "path": str(output),
        "file_sha256": _sha256_file(output),
        "sidecar_digest": payload["sidecar_digest"],
        "gate_digest": gate["gate_digest"],
        "assessment_digest": assessment["assessment_digest"],
        "bridge_manifest_sha256": bridge.get("manifest_sha256"),
    }


def verify_v3_sidecar(
    *,
    sidecar_path: str | Path,
    expected_sidecar_sha256: str,
    gate_path: str | Path,
    repo: str | Path,
    expected_manifest_sha256: str,
    authenticated_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify external trust anchor, source, scripts, assessment, and gate bytes."""

    sidecar_target = Path(sidecar_path).resolve()
    expected_sidecar = _require_sha256(
        expected_sidecar_sha256,
        "externally expected G00-E v3 sidecar digest",
    )
    if not sidecar_target.is_file() or _sha256_file(sidecar_target) != expected_sidecar:
        raise BridgeError("G00-E v3 sidecar is absent or differs from its external digest")
    sidecar = _strict_json(sidecar_target, "G00-E v3 sidecar")
    sidecar_body = {key: value for key, value in sidecar.items() if key != "sidecar_digest"}
    if (
        sidecar.get("schema") != BRIDGE_SIDECAR_SCHEMA
        or sidecar.get("schema_version") != BRIDGE_SIDECAR_SCHEMA_VERSION
        or sidecar.get("sidecar_digest") != _json_digest(sidecar_body)
    ):
        raise BridgeError("G00-E v3 sidecar schema or semantic digest is invalid")

    identity = (
        authenticate_legacy_and_bridge(
            repo=repo,
            expected_manifest_sha256=expected_manifest_sha256,
        )
        if authenticated_identity is None
        else dict(authenticated_identity)
    )
    bridge = identity.get("bridge")
    if not isinstance(bridge, Mapping):
        raise BridgeError("authenticated G00-E bridge identity is incomplete")
    observed_bridge_source = {
        "manifest": bridge.get("manifest"),
        "manifest_sha256": bridge.get("manifest_sha256"),
        "source_digest": bridge.get("source_digest"),
        "source_files": bridge.get("source_files"),
    }
    if sidecar.get("legacy_identity") != identity.get("legacy"):
        raise BridgeError("G00-E v3 sidecar legacy identity differs from the current source")
    if sidecar.get("numerical_audit") != identity.get("numerical_audit"):
        raise BridgeError("G00-E v3 sidecar numerical audit differs from the authenticated bridge")
    if sidecar.get("bridge_source") != observed_bridge_source:
        raise BridgeError("G00-E v3 sidecar bridge-source binding differs from current bytes")
    resolved_repo = Path(str(identity.get("repo", repo))).resolve()
    if sidecar.get("execution_scripts") != _script_identities(resolved_repo):
        raise BridgeError("G00-E v3 sidecar CLI/Runpod script binding differs from current bytes")

    assessment_binding = sidecar.get("assessment_artifact")
    gate_binding = sidecar.get("gate_artifact")
    if not isinstance(assessment_binding, Mapping) or not isinstance(gate_binding, Mapping):
        raise BridgeError("G00-E v3 sidecar artifact bindings are malformed")
    assessment_path = Path(str(assessment_binding.get("path", ""))).resolve()
    assessment, assessment_sha256 = _validated_assessment(assessment_path)
    supplied_gate_path = Path(gate_path).resolve()
    if supplied_gate_path != Path(str(gate_binding.get("path", ""))).resolve():
        raise BridgeError("worker gate path is not the exact v3-sidecar-bound gate path")
    gate, gate_sha256 = _validated_gate(supplied_gate_path, assessment=assessment)
    if (
        assessment_binding.get("file_sha256") != assessment_sha256
        or assessment_binding.get("assessment_digest") != assessment.get("assessment_digest")
        or gate_binding.get("file_sha256") != gate_sha256
        or gate_binding.get("gate_digest") != gate.get("gate_digest")
        or gate_binding.get("embedded_assessment_digest") != assessment.get("assessment_digest")
        or gate_binding.get("required_checks") != sorted(REQUIRED_GATE_CHECKS)
        or gate_binding.get("overall_passed") is not True
    ):
        raise BridgeError("G00-E v3 sidecar no longer matches its assessment/gate artifacts")
    return {
        "path": str(sidecar_target),
        "file_sha256": expected_sidecar,
        "sidecar_digest": sidecar["sidecar_digest"],
        "gate_digest": gate["gate_digest"],
        "assessment_digest": assessment["assessment_digest"],
        "bridge_manifest_sha256": bridge["manifest_sha256"],
        "authenticated_identity": identity,
    }


def install_worker_bridge(
    *,
    sidecar_path: str | Path,
    expected_sidecar_sha256: str,
    gate_path: str | Path,
    repo: str | Path,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    """Require the v3 sidecar at each verifier call, then delegate to legacy."""

    import goalzendo.runner as runner_module

    identity = authenticate_legacy_and_bridge(
        repo=repo,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    initial = verify_v3_sidecar(
        sidecar_path=sidecar_path,
        expected_sidecar_sha256=expected_sidecar_sha256,
        gate_path=gate_path,
        repo=repo,
        expected_manifest_sha256=expected_manifest_sha256,
        authenticated_identity=identity,
    )
    original_verifier = runner_module.verify_g00_gate_artifact
    if getattr(original_verifier, "__g00e_v3_wrapper__", False):
        raise BridgeError("G00-E worker verifier is already installed")
    install_exact_interval_correction(identity)

    def verified_legacy_gate(
        path: str | Path,
        *,
        config: Mapping[str, Any],
        repo: str | Path,
    ) -> Any:
        try:
            current = verify_v3_sidecar(
                sidecar_path=sidecar_path,
                expected_sidecar_sha256=expected_sidecar_sha256,
                gate_path=path,
                repo=repo,
                expected_manifest_sha256=expected_manifest_sha256,
                authenticated_identity=identity,
            )
        except BridgeError as error:
            raise runner_module.LaunchGuardError(f"G00-E v3 sidecar verification failed: {error}") from error
        result = original_verifier(path, config=config, repo=repo)
        if result.digest != current["gate_digest"]:
            raise runner_module.LaunchGuardError(
                "legacy G00 verifier returned a gate other than the v3-sidecar-bound gate"
            )
        return result

    verified_legacy_gate.__g00e_v3_wrapper__ = True  # type: ignore[attr-defined]
    runner_module.verify_g00_gate_artifact = verified_legacy_gate
    return {key: value for key, value in initial.items() if key != "authenticated_identity"}


__all__ = [
    "BRIDGE_MANIFEST_SCHEMA",
    "BRIDGE_MANIFEST_SCHEMA_VERSION",
    "BRIDGE_SIDECAR_SCHEMA",
    "BRIDGE_SIDECAR_SCHEMA_VERSION",
    "EQUIVALENCE_AUDIT_DIGEST",
    "FIRST_LEGACY_OVERFLOW_TRIALS",
    "FROZEN_BINOMIAL_CALLABLE_SOURCE_SHA256",
    "FROZEN_GOALZENDO_IMPLEMENTATION_FINGERPRINT",
    "FROZEN_RUNNER_FILE_SHA256",
    "NON_OVERFLOW_EQUIVALENCE_MAX_TRIALS",
    "NON_OVERFLOW_EQUIVALENCE_MIN_TRIALS",
    "PINNED_ACTUAL_INTERVALS",
    "BridgeError",
    "authenticate_legacy_and_bridge",
    "create_v3_sidecar",
    "exact_central_binomial_interval",
    "install_exact_interval_correction",
    "install_worker_bridge",
    "verify_bridge_source_manifest",
    "verify_v3_sidecar",
]
