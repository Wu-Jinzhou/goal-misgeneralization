from __future__ import annotations

import copy
import importlib.util
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd
import pytest

from goalzendo.analysis import AnalysisPanel
from goalzendo.config import get_path, load_config
from goalzendo.runner import build_plan

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "analyze_qwen35_known_law.py"
CONFIG = ROOT / "configs" / "goalzendo" / "qwen35_known_law_main.yaml"


def _load_script() -> ModuleType:
    specification = importlib.util.spec_from_file_location("analyze_qwen35_known_law_for_test", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


analyzer = _load_script()


def _synthetic_panel() -> tuple[AnalysisPanel, dict[str, Any]]:
    config = load_config(CONFIG)
    run_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    causal_rows: list[dict[str, Any]] = []
    for index, spec in enumerate(build_plan(config)):
        resolved = copy.deepcopy(spec.config)
        resolved["seed"] = spec.seed
        model = str(get_path(resolved, "model.name"))
        algorithm = str(get_path(resolved, "train.algorithm"))
        law = str(get_path(resolved, "data.rule_family"))
        q_p = float(get_path(resolved, "data.q_p"))
        seed_index = analyzer.SEEDS.index(spec.seed)
        rho_y = (
            0.35
            + 0.005 * seed_index
            + 0.04 * (model == analyzer.MODELS[1])
            + 0.03 * (algorithm == "outcome_rl")
            + 0.02 * (law == "majority")
            + 0.08 * (q_p == 0.95)
        )
        run_id = f"synthetic-{index:03d}"
        run_rows.append(
            {
                "run_id": run_id,
                "plan_key": spec.plan_key,
                "cell_id": spec.cell_id,
                "seed": spec.seed,
                "config": resolved,
            }
        )
        trajectory_rows.append(
            {
                "run_id": run_id,
                "step": 1000,
                "prompt_view": "full",
                "split": "final_factorial",
                "panel": "conflict",
                "rho_y": rho_y,
                "rho_p": rho_y + 0.20,
                "rho_q": 0.40,
            }
        )
        for target, action_flip_rate in (("y", 0.10), ("p", 0.70), ("q", 0.20), ("d", 0.05)):
            causal_rows.append(
                {
                    "run_id": run_id,
                    "step": 1000,
                    "prompt_view": "full",
                    "split": "final_factorial",
                    "target": target,
                    "action_flip_rate": action_flip_rate,
                }
            )
    panel = AnalysisPanel(
        runs=pd.DataFrame(run_rows),
        raw_metrics=pd.DataFrame(),
        predictions=pd.DataFrame(),
        trajectory=pd.DataFrame(trajectory_rows),
        causal_effects=pd.DataFrame(causal_rows),
        factorial=pd.DataFrame(),
        audit={"implementation_fingerprint": "synthetic-fingerprint"},
    )
    return panel, config


def _strata(report: dict[str, Any]) -> list[dict[str, Any]]:
    keys = [key for key in report if key.startswith("stratified_by_")]
    assert len(keys) == 1
    return report[keys[0]]


def test_exact_six_block_inference_is_deterministic_and_two_sided() -> None:
    values = {seed: 0.25 for seed in analyzer.SEEDS}
    first = analyzer._summarize_seed_blocks(values)
    second = analyzer._summarize_seed_blocks(values)
    assert first == second
    assert first["mean"] == 0.25
    assert first["bootstrap_95_percentile_interval"] == {"lower": 0.25, "upper": 0.25}
    assert first["descriptive_unadjusted_two_sided_sign_flip_p_value"] == 2 / 64
    assert analyzer._sign_flip_p_value([0.0] * 6) == 1.0
    assert analyzer.BOOTSTRAP_RESAMPLES == 6**6
    assert analyzer.SIGN_FLIP_ASSIGNMENTS == 2**6


def test_synthetic_panel_reports_all_registered_endpoints_and_contrasts() -> None:
    panel, config = _synthetic_panel()
    report = analyzer.analyze_panel(panel, config, config_sha256="frozen-config")

    assert report["panel"] == {
        "completed_only": True,
        "expected_run_count": 96,
        "observed_run_count": 96,
        "seed_blocks": list(analyzer.SEEDS),
        "expected_config_sha256": "frozen-config",
        "implementation_fingerprint": "synthetic-fingerprint",
    }
    assert len(report["seed_endpoints"]) == 96
    assert Counter(row["seed"] for row in report["seed_endpoints"]) == Counter(
        {seed: 16 for seed in analyzer.SEEDS}
    )

    primary = report["primary"]
    assert len(primary["stratified_by_model_algorithm_law"]) == 8
    for row in primary["stratified_by_model_algorithm_law"]:
        assert row["d_rho_p_minus_rho_y"]["mean"] == pytest.approx(0.20)
        assert row["causal_companion_flip_p_minus_flip_y"]["mean"] == pytest.approx(0.60)
        assert len(row["d_rho_p_minus_rho_y"]["per_seed"]) == 6
    assert primary["pooled_equal_weight_within_seed"]["d_rho_p_minus_rho_y"]["mean"] == pytest.approx(0.20)

    expected = {"evidence": 0.08, "algorithm": 0.03, "scale": 0.04}
    for name, effect in expected.items():
        contrast = report["secondary_contrasts"][name]
        assert len(_strata(contrast)) == 8
        assert contrast["pooled_equal_weight_within_seed"]["mean"] == pytest.approx(effect)
        assert all(row["effect"]["mean"] == pytest.approx(effect) for row in _strata(contrast))

    assert report["inference"]["p_values"] == {
        "method": "exact two-sided paired sign-flip over six seed blocks",
        "assignments": 64,
        "status": "descriptive_unadjusted",
        "binary_claims": False,
    }
    serialized = analyzer.canonical_json(report)
    assert serialized.endswith("\n") and serialized.count("\n") == 1
    assert json.loads(serialized) == report
    assert "significant" not in serialized and "reject_null" not in serialized


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate"])
def test_plan_row_mutations_fail_closed(mutation: str) -> None:
    panel, config = _synthetic_panel()
    rows = panel.runs.copy()
    if mutation == "missing":
        rows = rows.iloc[:-1].copy()
    elif mutation == "extra":
        extra = rows.iloc[[-1]].copy()
        extra.loc[:, "run_id"] = "unexpected-run"
        extra.loc[:, "plan_key"] = "unexpected-plan"
        rows = pd.concat([rows, extra], ignore_index=True)
    else:
        rows.loc[rows.index[-1], "plan_key"] = rows.loc[rows.index[0], "plan_key"]
    with pytest.raises(analyzer.Qwen35AnalysisError):
        analyzer.analyze_panel(replace(panel, runs=rows), config)


def test_duplicate_endpoint_and_missing_causal_target_fail_closed() -> None:
    panel, config = _synthetic_panel()
    duplicate = pd.concat([panel.trajectory, panel.trajectory.iloc[[0]]], ignore_index=True)
    with pytest.raises(analyzer.Qwen35AnalysisError, match="conflict rows; expected one"):
        analyzer.analyze_panel(replace(panel, trajectory=duplicate), config)

    first_run = str(panel.runs.iloc[0]["run_id"])
    causal = panel.causal_effects[
        ~((panel.causal_effects["run_id"] == first_run) & (panel.causal_effects["target"] == "d"))
    ].copy()
    with pytest.raises(analyzer.Qwen35AnalysisError, match="causal rows; expected four"):
        analyzer.analyze_panel(replace(panel, causal_effects=causal), config)


def test_loader_rejects_incomplete_directories_before_completed_only_load(
    tmp_path: Path,
) -> None:
    config = load_config(CONFIG)
    run = tmp_path / "experiment" / "run"
    run.mkdir(parents=True)
    (run / "identity.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(analyzer.Qwen35AnalysisError, match="incomplete run directory"):
        analyzer._load_exact_panel(tmp_path, config)


def test_cli_stdout_is_only_canonical_json(monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    monkeypatch.setattr(analyzer, "analyze_main", lambda _path: {"z": 0, "a": 1})
    assert analyzer.main(["unused"]) == 0
    captured = capsys.readouterr()
    assert captured.out == '{"a":1,"z":0}\n'
    assert captured.err == ""
