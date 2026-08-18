from __future__ import annotations

import copy
import csv
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXPORTER = (
    ROOT
    / "paper"
    / "goalzendo-current-results"
    / "analysis"
    / "make_qwen35_hidden_law_assets.py"
)
ANALYZER_TEST = ROOT / "tests" / "goalzendo_hidden_law" / "test_analyze_hidden_law.py"


def _load_module(name: str, path: Path) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


assets = _load_module("make_qwen35_hidden_law_assets_for_test", EXPORTER)
synthetic = _load_module("qwen35_hidden_law_analyzer_fixture_for_assets", ANALYZER_TEST)


@pytest.fixture(scope="module")
def report() -> dict[str, Any]:
    runs, config = synthetic._synthetic_runs()
    value = synthetic.analyzer.analyze_runs(
        runs,
        config,
        config_file_sha256=assets.CONFIG_SHA256,
        implementation_fingerprint=synthetic.FINGERPRINT,
    )
    value["panel"]["implementation_fingerprint"] = assets.IMPLEMENTATION_FINGERPRINT
    return value


def test_fixed_tables_preserve_every_registered_axis(report: dict[str, Any]) -> None:
    endpoints, trajectories = assets.validate_report(report)
    endpoint_rows = assets.make_endpoint_rows(endpoints)
    trajectory_rows = assets.make_trajectory_rows(trajectories)
    contrast_rows = assets.make_contrast_rows(endpoints)
    view_rows = assets.make_view_rows(endpoints)

    assert len(endpoints) == 24
    assert len(trajectories) == 24
    assert len(endpoint_rows) == 24 * 4
    assert len(trajectory_rows) == 24 * 4 * 3
    assert len(contrast_rows) == 3 * 4 * 6
    assert len(view_rows) == 24 * 3 * 4
    assert Counter(row["metric"] for row in endpoint_rows) == Counter(
        {metric: 24 for metric in assets.METRIC_KEYS}
    )
    assert Counter(row["step"] for row in trajectory_rows) == Counter(
        {step: 24 * 3 for step in assets.EVAL_STEPS}
    )
    assert {row["view"] for row in trajectory_rows} == {"active"}
    assert Counter(row["view"] for row in view_rows) == Counter(
        {view: 24 * 4 for view in assets.FINAL_VIEWS}
    )
    assert Counter(row["scope"] for row in contrast_rows) == Counter(
        {"model": 2 * 4 * 6, "pooled_equal_weight": 4 * 6}
    )
    assert all(
        row["sft_minus_outcome_rl"]
        == pytest.approx(row["process_sft_value"] - row["outcome_rl_value"])
        for row in contrast_rows
    )


def test_export_is_byte_deterministic_and_writes_only_fixed_assets(
    report: dict[str, Any],
    tmp_path: Path,
) -> None:
    first = assets.export_assets(report, tmp_path / "first")
    second = assets.export_assets(report, tmp_path / "second")
    assert [path.relative_to(tmp_path / "first") for path in first] == [
        path.relative_to(tmp_path / "second") for path in second
    ]
    assert [path.read_bytes() for path in first] == [path.read_bytes() for path in second]
    assert len(first) == 6
    assert sum(path.suffix == ".pdf" for path in first) == 2
    assert sum(path.suffix == ".csv" for path in first) == 4
    assert all(path.read_bytes().startswith(b"%PDF-") for path in first if path.suffix == ".pdf")

    expected_rows = {
        "qwen35_hidden_law_endpoints.csv": 24 * 4,
        "qwen35_hidden_law_active_trajectories.csv": 24 * 4 * 3,
        "qwen35_hidden_law_registered_contrasts.csv": 3 * 4 * 6,
        "qwen35_hidden_law_view_decomposition.csv": 24 * 3 * 4,
    }
    for path in first:
        if path.suffix != ".csv":
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == expected_rows[path.name]


@pytest.mark.parametrize(
    "mutation",
    [
        "schema",
        "config",
        "scientific_digest",
        "implementation",
        "run_count",
        "seed",
        "model",
        "algorithm",
        "steps",
        "views",
        "evidence_cells",
        "metric_definition",
        "registered_contrast",
    ],
)
def test_exporter_fails_closed_on_frozen_contract_mutations(
    report: dict[str, Any],
    mutation: str,
) -> None:
    changed = copy.deepcopy(report)
    if mutation == "schema":
        changed["schema_version"] = 2
    elif mutation == "config":
        changed["panel"]["config_file_sha256"] = "0" * 64
    elif mutation == "scientific_digest":
        changed["panel"]["scientific_config_digest"] = "0" * 64
    elif mutation == "implementation":
        changed["panel"]["implementation_fingerprint"] = "0" * 64
    elif mutation == "run_count":
        changed["seed_endpoints"].pop()
    elif mutation == "seed":
        changed["seed_endpoints"][0]["seed"] = 999
    elif mutation == "model":
        changed["seed_endpoints"][0]["model"] = "unregistered"
    elif mutation == "algorithm":
        changed["seed_endpoints"][0]["algorithm"] = "unregistered"
    elif mutation == "steps":
        changed["seed_trajectories"][0]["checkpoints"][1]["step"] = 9
    elif mutation == "views":
        changed["seed_endpoints"][0]["final_views"].pop("no_query")
    elif mutation == "evidence_cells":
        changed["seed_endpoints"][0]["condition_cells"].pop("both_noisy")
    elif mutation == "metric_definition":
        changed["seed_endpoints"][0]["active"]["law_control_margin"] += 0.125
    else:
        changed["primary_sft_minus_outcome_rl"]["per_model"][0]["sft_minus_outcome_rl"][
            "information_fraction"
        ]["mean"] += 0.125
    with pytest.raises(assets.AssetError):
        assets.validate_report(changed)


def test_canonical_input_is_required(report: dict[str, Any], tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.json"
    canonical.write_text(
        json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )
    loaded, digest = assets.load_canonical_report(canonical)
    assert loaded == report
    assert len(digest) == 64

    pretty = tmp_path / "pretty.json"
    pretty.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with pytest.raises(assets.AssetError, match="canonical"):
        assets.load_canonical_report(pretty)
