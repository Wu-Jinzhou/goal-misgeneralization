from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

import goalzendo.artifacts as artifacts_module
import goalzendo.runner as runner_module
from goalzendo.artifacts import stable_hash
from goalzendo_g00e import cli as bridge_cli
from goalzendo_g00e.bridge import (
    BRIDGE_SIDECAR_SCHEMA,
    BRIDGE_SIDECAR_SCHEMA_VERSION,
    EQUIVALENCE_AUDIT_DIGEST,
    FIRST_LEGACY_OVERFLOW_TRIALS,
    FROZEN_GOALZENDO_IMPLEMENTATION_FINGERPRINT,
    NON_OVERFLOW_EQUIVALENCE_MAX_TRIALS,
    PINNED_ACTUAL_INTERVALS,
    REQUIRED_GATE_CHECKS,
    BridgeError,
    _equivalence_digest,
    authenticate_legacy_and_bridge,
    create_v3_sidecar,
    exact_central_binomial_interval,
    install_worker_bridge,
    verify_bridge_source_manifest,
    verify_v3_sidecar,
)

ROOT = Path(__file__).resolve().parents[2]
BRIDGE_MANIFEST = ROOT / "src" / "goalzendo_g00e" / "bridge_manifest.json"
BRIDGE_MANIFEST_SHA256 = hashlib.sha256(BRIDGE_MANIFEST.read_bytes()).hexdigest()
G01_CONFIG = ROOT / "configs" / "goalzendo" / "g01_known_law.yaml"


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _artifact_pair(tmp_path: Path) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    assessment_body: dict[str, Any] = {
        "schema": "goalzendo.g00_assessment",
        "schema_version": 2,
        "evidence_run_binding_digest": "a" * 64,
        "evidence_source_fingerprint": FROZEN_GOALZENDO_IMPLEMENTATION_FINGERPRINT,
        "evidence_config_digest": "b" * 64,
        "optimizer_stability_policy": runner_module.G00_OPTIMIZER_STABILITY_POLICY.as_dict(),
        "optimizer_stability_policy_digest": stable_hash(
            runner_module.G00_OPTIMIZER_STABILITY_POLICY.as_dict(), 64
        ),
        "measurements": {},
        "derivation": {"source": "completed_metrics_predictions_and_manifests"},
    }
    assessment = {
        **assessment_body,
        "assessment_digest": stable_hash(assessment_body, 64),
    }
    checks = {name: {"passed": True} for name in REQUIRED_GATE_CHECKS}
    gate_body: dict[str, Any] = {
        "schema": "goalzendo.g00_gate",
        "schema_version": 2,
        "created_at": "2026-08-11T00:00:00+00:00",
        "thresholds": runner_module.G00_GATE_THRESHOLDS.as_dict(),
        "threshold_digest": stable_hash(runner_module.G00_GATE_THRESHOLDS.as_dict(), 64),
        "optimizer_stability_policy": runner_module.G00_OPTIMIZER_STABILITY_POLICY.as_dict(),
        "optimizer_stability_policy_digest": stable_hash(
            runner_module.G00_OPTIMIZER_STABILITY_POLICY.as_dict(), 64
        ),
        "verification_inputs": {},
        "verification_inputs_digest": stable_hash({}, 64),
        "g00_evidence": {},
        "assessment": assessment,
        "checks": checks,
        "selected_optimizer_settings": {},
        "authorized_targets": {},
        "overall_passed": True,
    }
    gate = {**gate_body, "gate_digest": stable_hash(gate_body, 64)}
    assessment_path = tmp_path / "assessment.json"
    gate_path = tmp_path / "gate.json"
    _write_json(assessment_path, assessment)
    _write_json(gate_path, gate)
    return assessment_path, gate_path, assessment, gate


@pytest.fixture(scope="module")
def authenticated_identity() -> dict[str, Any]:
    return authenticate_legacy_and_bridge(
        repo=ROOT,
        expected_manifest_sha256=BRIDGE_MANIFEST_SHA256,
    )


def test_exact_integer_interval_matches_every_legacy_non_overflow_fixture() -> None:
    assert NON_OVERFLOW_EQUIVALENCE_MAX_TRIALS == 1029
    assert _equivalence_digest(runner_module._binomial_acceptance_interval) == (
        EQUIVALENCE_AUDIT_DIGEST
    )
    with pytest.raises(OverflowError):
        runner_module._binomial_acceptance_interval(
            FIRST_LEGACY_OVERFLOW_TRIALS,
            probability=0.5,
            alpha=0.05,
        )


def test_actual_capability_trial_intervals_are_pinned() -> None:
    assert PINNED_ACTUAL_INTERVALS == {1536: (730, 806), 3072: (1482, 1590)}
    for trials, expected in PINNED_ACTUAL_INTERVALS.items():
        assert exact_central_binomial_interval(
            trials,
            probability=0.5,
            alpha=0.05,
        ) == expected


@pytest.mark.parametrize(
    ("probability", "alpha"),
    [(0.49, 0.05), (0.5, 0.051), (0.5, 0.049)],
)
def test_bridge_rejects_any_threshold_drift(probability: float, alpha: float) -> None:
    with pytest.raises(BridgeError, match="frozen"):
        exact_central_binomial_interval(100, probability=probability, alpha=alpha)


def test_bridge_authentication_rejects_wrong_manifest_source_and_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(BridgeError, match="manifest bytes"):
        verify_bridge_source_manifest("0" * 64)

    original_provenance = artifacts_module.implementation_provenance
    monkeypatch.setattr(
        artifacts_module,
        "implementation_provenance",
        lambda *_args, **_kwargs: {"implementation_fingerprint": "0" * 64},
    )
    with pytest.raises(BridgeError, match="implementation fingerprint"):
        authenticate_legacy_and_bridge(
            repo=ROOT,
            expected_manifest_sha256=BRIDGE_MANIFEST_SHA256,
        )
    monkeypatch.setattr(artifacts_module, "implementation_provenance", original_provenance)

    original_callable = runner_module._binomial_acceptance_interval
    monkeypatch.setattr(
        runner_module,
        "_binomial_acceptance_interval",
        lambda trials, *, probability, alpha: (0, trials),
    )
    with pytest.raises(BridgeError, match="callable identity"):
        authenticate_legacy_and_bridge(
            repo=ROOT,
            expected_manifest_sha256=BRIDGE_MANIFEST_SHA256,
        )
    monkeypatch.setattr(runner_module, "_binomial_acceptance_interval", original_callable)


def test_bridge_authentication_rejects_altered_frozen_thresholds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed = dataclasses.replace(runner_module.G00_GATE_THRESHOLDS, binomial_alpha=0.051)
    monkeypatch.setattr(runner_module, "G00_GATE_THRESHOLDS", changed)
    with pytest.raises(BridgeError, match="thresholds"):
        authenticate_legacy_and_bridge(
            repo=ROOT,
            expected_manifest_sha256=BRIDGE_MANIFEST_SHA256,
        )


def test_v3_sidecar_binds_exact_assessment_gate_and_bridge_bytes(
    tmp_path: Path,
    authenticated_identity: Mapping[str, Any],
) -> None:
    assessment_path, gate_path, _assessment, gate = _artifact_pair(tmp_path)
    sidecar_path = tmp_path / "sidecar.json"
    created = create_v3_sidecar(
        gate_path=gate_path,
        assessment_path=assessment_path,
        output_path=sidecar_path,
        authenticated_identity=authenticated_identity,
    )
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert payload["schema"] == BRIDGE_SIDECAR_SCHEMA
    assert payload["schema_version"] == BRIDGE_SIDECAR_SCHEMA_VERSION == 3
    assert payload["gate_artifact"]["gate_digest"] == gate["gate_digest"]
    verified = verify_v3_sidecar(
        sidecar_path=sidecar_path,
        expected_sidecar_sha256=created["file_sha256"],
        gate_path=gate_path,
        repo=ROOT,
        expected_manifest_sha256=BRIDGE_MANIFEST_SHA256,
        authenticated_identity=authenticated_identity,
    )
    assert verified["gate_digest"] == gate["gate_digest"]

    with pytest.raises(BridgeError, match="absent"):
        verify_v3_sidecar(
            sidecar_path=tmp_path / "absent.json",
            expected_sidecar_sha256="0" * 64,
            gate_path=gate_path,
            repo=ROOT,
            expected_manifest_sha256=BRIDGE_MANIFEST_SHA256,
            authenticated_identity=authenticated_identity,
        )

    payload["bridge_source"]["source_digest"] = "0" * 64
    body = {key: value for key, value in payload.items() if key != "sidecar_digest"}
    payload["sidecar_digest"] = stable_hash(body, 64)
    _write_json(sidecar_path, payload)
    tampered_file_digest = hashlib.sha256(sidecar_path.read_bytes()).hexdigest()
    with pytest.raises(BridgeError, match="bridge-source"):
        verify_v3_sidecar(
            sidecar_path=sidecar_path,
            expected_sidecar_sha256=tampered_file_digest,
            gate_path=gate_path,
            repo=ROOT,
            expected_manifest_sha256=BRIDGE_MANIFEST_SHA256,
            authenticated_identity=authenticated_identity,
        )


def test_v3_sidecar_rejects_gate_assessment_mismatch(
    tmp_path: Path,
    authenticated_identity: Mapping[str, Any],
) -> None:
    assessment_path, gate_path, assessment, _gate = _artifact_pair(tmp_path)
    changed_body = {
        key: value for key, value in assessment.items() if key != "assessment_digest"
    }
    changed_body["independent_edit"] = True
    changed = {**changed_body, "assessment_digest": stable_hash(changed_body, 64)}
    _write_json(assessment_path, changed)
    with pytest.raises(BridgeError, match="bound to the assessment"):
        create_v3_sidecar(
            gate_path=gate_path,
            assessment_path=assessment_path,
            output_path=tmp_path / "sidecar.json",
            authenticated_identity=authenticated_identity,
        )


def test_v3_sidecar_refuses_any_failed_or_missing_original_check(
    tmp_path: Path,
    authenticated_identity: Mapping[str, Any],
) -> None:
    assessment_path, gate_path, assessment, gate = _artifact_pair(tmp_path)
    gate_body = {key: value for key, value in gate.items() if key != "gate_digest"}
    gate_body["checks"]["dataset_integrity"]["passed"] = False
    gate_body["overall_passed"] = False
    failed_gate = {**gate_body, "gate_digest": stable_hash(gate_body, 64)}
    _write_json(gate_path, failed_gate)
    with pytest.raises(BridgeError, match="six-check"):
        create_v3_sidecar(
            gate_path=gate_path,
            assessment_path=assessment_path,
            output_path=tmp_path / "sidecar.json",
            authenticated_identity=authenticated_identity,
        )
    assert gate["assessment"] == assessment


def test_worker_bridge_requires_sidecar_then_delegates_to_original_verifier(
    tmp_path: Path,
    authenticated_identity: Mapping[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assessment_path, gate_path, _assessment, gate = _artifact_pair(tmp_path)
    sidecar_path = tmp_path / "sidecar.json"
    created = create_v3_sidecar(
        gate_path=gate_path,
        assessment_path=assessment_path,
        output_path=sidecar_path,
        authenticated_identity=authenticated_identity,
    )
    original_interval = runner_module._binomial_acceptance_interval
    monkeypatch.setattr(runner_module, "_binomial_acceptance_interval", original_interval)
    calls: list[Path] = []

    class Verified:
        digest = gate["gate_digest"]

    def original_verifier(
        path: str | Path,
        *,
        config: Mapping[str, Any],
        repo: str | Path,
    ) -> Verified:
        del config, repo
        calls.append(Path(path).resolve())
        return Verified()

    monkeypatch.setattr(runner_module, "verify_g00_gate_artifact", original_verifier)
    install_worker_bridge(
        sidecar_path=sidecar_path,
        expected_sidecar_sha256=created["file_sha256"],
        gate_path=gate_path,
        repo=ROOT,
        expected_manifest_sha256=BRIDGE_MANIFEST_SHA256,
    )
    observed = runner_module.verify_g00_gate_artifact(gate_path, config={}, repo=ROOT)
    assert observed.digest == gate["gate_digest"]
    assert calls == [gate_path.resolve()]

    sidecar_path.unlink()
    with pytest.raises(runner_module.LaunchGuardError, match="sidecar verification failed"):
        runner_module.verify_g00_gate_artifact(gate_path, config={}, repo=ROOT)


def test_worker_cli_fails_before_legacy_run_when_sidecar_is_not_explicit(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = bridge_cli.main(
        [
            "--repo",
            str(ROOT),
            "run",
            str(G01_CONFIG),
            "--gate-artifact",
            str(ROOT / "does-not-exist-gate.json"),
        ]
    )
    captured = capsys.readouterr()
    assert result == 2
    assert "requires explicit --g00e-sidecar" in captured.err


def test_failed_legacy_gate_artifacts_are_preserved_without_authorizing_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assessment_path, gate_path, _assessment, gate = _artifact_pair(tmp_path)
    failed_body = {key: value for key, value in gate.items() if key != "gate_digest"}
    failed_body["checks"]["rule_adapters"]["passed"] = False
    failed_body["checks"]["constrained_scorer"]["passed"] = False
    failed_body["overall_passed"] = False
    failed_gate = {**failed_body, "gate_digest": stable_hash(failed_body, 64)}
    _write_json(gate_path, failed_gate)
    assessment_bytes = assessment_path.read_bytes()
    gate_bytes = gate_path.read_bytes()
    assessment_path.unlink()
    gate_path.unlink()
    sidecar_path = tmp_path / "must-not-exist-sidecar.json"
    sidecar_called = False

    def failed_legacy_gate(_arguments: Any) -> int:
        assessment_path.write_bytes(assessment_bytes)
        gate_path.write_bytes(gate_bytes)
        return 1

    def forbidden_sidecar(**_arguments: Any) -> dict[str, Any]:
        nonlocal sidecar_called
        sidecar_called = True
        raise AssertionError("failed legacy gate must not create a v3 sidecar")

    import goalzendo.cli as legacy_cli

    monkeypatch.setattr(legacy_cli, "main", failed_legacy_gate)
    monkeypatch.setattr(
        bridge_cli,
        "authenticate_legacy_and_bridge",
        lambda **_arguments: {"authenticated": True},
    )
    monkeypatch.setattr(
        bridge_cli,
        "install_exact_interval_correction",
        lambda _identity: runner_module._binomial_acceptance_interval,
    )
    monkeypatch.setattr(bridge_cli, "create_v3_sidecar", forbidden_sidecar)
    result = bridge_cli.main(
        [
            "--repo",
            str(ROOT),
            "gate",
            "--g00-artifacts",
            str(tmp_path / "evidence"),
            "--g00-config",
            str(ROOT / "configs" / "goalzendo" / "g00d_fixed_window_engineering_0p5b.yaml"),
            "--target-config",
            str(G01_CONFIG),
            "--select-sft-learning-rate",
            "0.00001",
            "--select-rl-learning-rate",
            "0.000003",
            "--assessment-output",
            str(assessment_path),
            "--output",
            str(gate_path),
            "--g00e-sidecar-output",
            str(sidecar_path),
            "--g00e-manifest-sha256",
            BRIDGE_MANIFEST_SHA256,
        ]
    )
    assert result == 1
    assert assessment_path.read_bytes() == assessment_bytes
    assert gate_path.read_bytes() == gate_bytes
    assert json.loads(gate_path.read_text(encoding="utf-8"))["overall_passed"] is False
    assert sidecar_called is False
    assert not sidecar_path.exists()
