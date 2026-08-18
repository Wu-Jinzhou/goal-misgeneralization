from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "runs" / "goalzendo" / "postcomplete_qwen35_evidence_geometry.py"


def _load_script() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "postcomplete_qwen35_evidence_geometry_for_test", SCRIPT
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


workflow = _load_script()


def _receipt(index: int) -> dict[str, object]:
    return {
        "schema": "goalzendo.qwen35_evidence_geometry_shard_status",
        "schema_version": 1,
        "job_id": workflow.JOB_ID,
        "shard_index": index,
        "num_shards": workflow.NUM_SHARDS,
        "state": "complete",
        "runner_rc": 0,
        "archive_sha256": workflow.ARCHIVE_SHA256,
        "config_sha256": workflow.CONFIG_SHA256,
        "implementation_fingerprint": workflow.IMPLEMENTATION_FINGERPRINT,
        "started_at": "2026-08-15T13:21:28Z",
        "finished_at": "2026-08-15T14:21:28Z",
    }


def _write_receipts(root: Path) -> None:
    status = root / "status"
    status.mkdir(parents=True)
    for index in range(workflow.NUM_SHARDS):
        target = status / f"shard-{index:02d}.complete.json"
        target.write_text(json.dumps(_receipt(index)) + "\n", encoding="utf-8")


def _analysis_report() -> dict[str, object]:
    return {
        "schema": "goalzendo.qwen35_evidence_geometry_analysis",
        "schema_version": 1,
        "panel": {
            "completed_only": True,
            "expected_run_count": 36,
            "observed_run_count": 36,
            "expected_config_sha256": workflow.CONFIG_SHA256,
            "registered_config_identity_sha256": workflow.CONFIG_IDENTITY_SHA256,
            "implementation_fingerprint": workflow.IMPLEMENTATION_FINGERPRINT,
        },
        "seed_endpoints": [{"row": index} for index in range(36)],
        "seed_trajectories": [{"row": index} for index in range(36)],
        "primary_contrast": {"label": "registered-primary"},
        "registered_secondary_contrast": {"label": "registered-secondary"},
    }


def test_receipts_require_exact_eight_rc0_frozen_bindings(tmp_path: Path) -> None:
    _write_receipts(tmp_path)
    assert workflow._verify_receipts(tmp_path) == 8

    bad = tmp_path / "status" / "shard-03.complete.json"
    payload = json.loads(bad.read_text(encoding="utf-8"))
    payload["runner_rc"] = 1
    bad.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(workflow.PostcompletionError, match="rc0"):
        workflow._verify_receipts(tmp_path)


def test_analysis_validation_is_canonical_and_exact() -> None:
    report = _analysis_report()
    canonical = workflow._canonical_bytes(report)
    assert workflow._validate_analysis(canonical) == report

    with pytest.raises(workflow.PostcompletionError, match="one-line"):
        workflow._validate_analysis(json.dumps(report, indent=2).encode() + b"\n")

    report["unregistered_contrast"] = {}
    with pytest.raises(workflow.PostcompletionError, match="contrast inventory"):
        workflow._validate_analysis(workflow._canonical_bytes(report))


def test_publication_is_atomic_and_never_overwrites(tmp_path: Path) -> None:
    analysis = workflow._canonical_bytes(_analysis_report())
    expected_sha = workflow.hashlib.sha256(analysis).hexdigest()
    destination = tmp_path / "analysis"
    status = {
        "schema": "goalzendo.qwen35_evidence_geometry_postcompletion_status",
        "schema_version": 1,
        "status_scope": "postlaunch_operational",
        "state": "complete",
    }

    assert workflow._publish_bundle(destination, analysis, status) == expected_sha
    assert (destination / workflow.ANALYSIS_NAME).read_bytes() == analysis
    assert (destination / "SHA256SUMS").read_text(encoding="utf-8") == (
        f"{expected_sha}  {workflow.ANALYSIS_NAME}\n"
    )
    assert json.loads((destination / "status.json").read_text(encoding="utf-8")) == status
    assert not list(tmp_path.glob(".analysis.staging-*"))

    with pytest.raises(workflow.PostcompletionError, match="refusing to overwrite"):
        workflow._publish_bundle(destination, analysis, status)


def test_embedded_preflight_is_valid_python() -> None:
    compile(workflow._PANEL_PREFLIGHT, "<qwen35-geometry-panel-preflight>", "exec")
