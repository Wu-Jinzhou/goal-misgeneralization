from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXPORTER_PATH = (
    ROOT
    / "paper"
    / "goalzendo-current-results"
    / "analysis"
    / "make_qwen35_evidence_geometry_assets.py"
)
ANALYZER_TEST_PATH = ROOT / "tests" / "goalzendo" / "test_analyze_qwen35_evidence_geometry.py"


def _load(path: Path, name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


exporter = _load(EXPORTER_PATH, "make_qwen35_evidence_geometry_assets_for_test")
analyzer_test = _load(ANALYZER_TEST_PATH, "analyzer_geometry_synthetic_fixture")


def _synthetic_report() -> dict[str, object]:
    panel, config = analyzer_test._synthetic_panel()
    panel = replace(panel, audit={"implementation_fingerprint": exporter.IMPLEMENTATION_FINGERPRINT})
    return analyzer_test.analyzer.analyze_panel(
        panel,
        config,
        config_sha256=exporter.CONFIG_SHA256,
    )


def _canonical_write(path: Path, report: dict[str, object]) -> None:
    path.write_text(
        json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_synthetic_export_is_complete_and_byte_deterministic(tmp_path: Path) -> None:
    report = _synthetic_report()
    exporter.validate_report(report)
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_outputs = exporter.export_assets(report, first)
    exporter.export_assets(report, second)

    assert [path.relative_to(first) for path in first_outputs] == [
        Path("generated/qwen35_evidence_geometry_endpoints.csv"),
        Path("generated/qwen35_evidence_geometry_trajectories.csv"),
        Path("figures/qwen35_evidence_geometry_endpoints.pdf"),
        Path("figures/qwen35_evidence_geometry_trajectories.pdf"),
    ]
    assert _tree_hashes(first) == _tree_hashes(second)

    with first_outputs[0].open(newline="", encoding="utf-8") as handle:
        endpoints = list(csv.DictReader(handle))
    with first_outputs[1].open(newline="", encoding="utf-8") as handle:
        trajectories = list(csv.DictReader(handle))
    assert len(endpoints) == 36
    assert len(trajectories) == 36 * len(exporter.EVAL_STEPS)
    assert {int(row["seed"]) for row in endpoints} == set(exporter.SEEDS)
    assert {int(row["step"]) for row in trajectories} == set(exporter.EVAL_STEPS)
    for path in first_outputs[2:]:
        assert path.read_bytes().startswith(b"%PDF-")
        assert path.stat().st_size > 10_000


@pytest.mark.parametrize(
    "mutation",
    (
        "schema",
        "config",
        "implementation",
        "run_count",
        "seed_blocks",
        "missing_cell",
        "trajectory_step",
        "causal_definition",
    ),
)
def test_registered_contract_mutations_fail_closed(mutation: str) -> None:
    report = copy.deepcopy(_synthetic_report())
    if mutation == "schema":
        report["schema"] = "unexpected"
    elif mutation == "config":
        report["panel"]["expected_config_sha256"] = "0" * 64
    elif mutation == "implementation":
        report["panel"]["implementation_fingerprint"] = "0" * 64
    elif mutation == "run_count":
        report["panel"]["observed_run_count"] = 35
    elif mutation == "seed_blocks":
        report["panel"]["seed_blocks"][-1] = -1
    elif mutation == "missing_cell":
        report["seed_endpoints"].pop()
    elif mutation == "trajectory_step":
        report["seed_trajectories"][0]["checkpoints"][1]["step"] = 2
    else:
        report["seed_endpoints"][0]["causal_companion_c"] += 0.01
    with pytest.raises(exporter.AssetError):
        exporter.validate_report(report)


def test_loader_requires_one_line_canonical_json(tmp_path: Path) -> None:
    report = _synthetic_report()
    canonical = tmp_path / "canonical.json"
    _canonical_write(canonical, report)
    loaded, digest = exporter.load_canonical_report(canonical)
    assert loaded == report
    assert digest == hashlib.sha256(canonical.read_bytes()).hexdigest()

    pretty = tmp_path / "pretty.json"
    pretty.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(exporter.AssetError, match="one-line canonical"):
        exporter.load_canonical_report(pretty)


def test_exported_interval_matches_the_registered_exhaustive_bootstrap() -> None:
    values = (0.10, 0.21, 0.32, 0.43, 0.54, 0.65)
    mean, lower, upper = exporter.seed_summary(values)
    expected_lower, expected_upper = analyzer_test.analyzer._bootstrap_interval(values)
    assert mean == pytest.approx(sum(values) / 6, abs=1e-15)
    assert lower == pytest.approx(expected_lower, abs=1e-15)
    assert upper == pytest.approx(expected_upper, abs=1e-15)
