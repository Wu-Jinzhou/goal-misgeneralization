from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "paper" / "forkworld-current-results" / "e16_analysis.py"


def _load_e16_analysis():
    spec = importlib.util.spec_from_file_location("e16_analysis_for_test", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


E16_ANALYSIS = _load_e16_analysis()


def test_uniform_artifact_fingerprint_drift_is_reported_not_rejected() -> None:
    artifact_fingerprint = "a" * 64
    current_fingerprint = "b" * 64

    status = E16_ANALYSIS._source_fingerprint_status(
        [artifact_fingerprint, artifact_fingerprint], current_fingerprint
    )

    assert status["artifact_implementation_fingerprint"] == artifact_fingerprint
    assert status["current_source_fingerprint"] == current_fingerprint
    assert status["current_source_matches_artifact"] is False
    assert status["current_worktree_drift_detected"] is True


def test_multiple_artifact_fingerprints_remain_rejected() -> None:
    with pytest.raises(ValueError, match="expected one implementation fingerprint"):
        E16_ANALYSIS._source_fingerprint_status(["a" * 64, "b" * 64], "c" * 64)


def test_drift_disposition_requires_all_exact_audits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_fingerprint = "a" * 64
    current_fingerprint = "b" * 64
    panel_audit = {
        "implementation_fingerprints": [artifact_fingerprint],
        "current_source_fingerprint": current_fingerprint,
        "strict_grid_and_measurement_audit_passed": True,
    }
    pairing_audit = {
        "current_source_regeneration_audit_passed": True,
        "unchanged_array_and_panel_checks": 1,
        "exact_design_and_panel_metric_checks": 1,
        "exact_exposure_metric_checks": 1,
    }
    bridge_audit = {
        "m0_matches_archived_e15_nested_exactly": True,
        "m0_exact_metric_bridges": E16_ANALYSIS.EXPECTED_E15_BRIDGES,
    }
    monkeypatch.setattr(
        E16_ANALYSIS,
        "implementation_provenance",
        lambda _root: {"implementation_fingerprint": current_fingerprint},
    )

    audit = E16_ANALYSIS._final_source_fingerprint_audit(
        panel_audit, panel_audit, pairing_audit, bridge_audit
    )

    assert audit["current_worktree_drift_detected"] is True
    assert audit["current_worktree_drift_fatal"] is False
    assert all(audit["required_exact_audits"].values())

    pairing_audit["exact_exposure_metric_checks"] = 0
    with pytest.raises(RuntimeError, match="required exact audit did not pass"):
        E16_ANALYSIS._final_source_fingerprint_audit(panel_audit, panel_audit, pairing_audit, bridge_audit)
