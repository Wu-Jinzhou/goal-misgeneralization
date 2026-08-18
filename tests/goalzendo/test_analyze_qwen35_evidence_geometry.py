from __future__ import annotations

import copy
import importlib.util
import json
from collections import Counter
from dataclasses import replace
from itertools import product
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd
import pytest

from goalzendo.analysis import AnalysisPanel
from goalzendo.config import get_path, load_config
from goalzendo.runner import build_plan

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "analyze_qwen35_evidence_geometry.py"
CONFIG = ROOT / "configs" / "goalzendo" / "qwen35_evidence_geometry.yaml"


def _load_script() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "analyze_qwen35_evidence_geometry_for_test", SCRIPT
    )
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
    iid_rows: list[dict[str, Any]] = []
    factorial_rows: list[dict[str, Any]] = []
    geometry_effects = {
        (0.005, "diverse"): 0.00,
        (0.040, "diverse"): 0.12,
        (0.040, "concentrated"): 0.05,
    }
    for index, spec in enumerate(build_plan(config)):
        resolved = copy.deepcopy(spec.config)
        resolved["seed"] = spec.seed
        model = str(get_path(resolved, "model.name"))
        joint_error_rate = float(get_path(resolved, "data.joint_error_rate"))
        diversity = str(get_path(resolved, "data.conflict_diversity"))
        geometry_effect = geometry_effects[(joint_error_rate, diversity)]
        model_effect = 0.02 if model == analyzer.MODELS[1] else 0.0
        seed_effect = 0.002 * analyzer.SEEDS.index(spec.seed)
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
        for step in analyzer.EVAL_STEPS:
            progress = step / analyzer.FINAL_STEP
            rho_y = 0.25 + model_effect + seed_effect + geometry_effect * progress
            split = "final_factorial" if step == analyzer.FINAL_STEP else "diagnostic_factorial"
            trajectory_rows.extend(
                [
                    {
                        "run_id": run_id,
                        "step": step,
                        "prompt_view": "full",
                        "split": split,
                        "panel": "conflict",
                        "rho_y": rho_y,
                        "rho_p": 0.75 - 0.10 * geometry_effect * progress,
                        "rho_q": 0.55 + 0.05 * geometry_effect * progress,
                    },
                    {
                        "run_id": run_id,
                        "step": step,
                        "prompt_view": "audit_law_matched",
                        "split": split,
                        "panel": "conflict",
                        "rho_y": 0.70 + 0.10 * progress,
                        "rho_p": 0.50,
                        "rho_q": 0.50,
                    },
                    {
                        "run_id": run_id,
                        "step": step,
                        "prompt_view": "full",
                        "split": "iid_validation",
                        "panel": "conflict",
                        "rho_y": 0.90 + 0.08 * progress,
                        "rho_p": 0.90 + 0.08 * progress,
                        "rho_q": 0.90 + 0.08 * progress,
                    },
                ]
            )
            for target, flip_rate in (
                ("y", 0.15 + geometry_effect * progress),
                ("p", 0.65),
                ("q", 0.45),
                ("d", 0.05),
            ):
                causal_rows.append(
                    {
                        "run_id": run_id,
                        "step": step,
                        "prompt_view": "full",
                        "split": split,
                        "target": target,
                        "action_flip_rate": flip_rate,
                        "n_pairs": 128 if step == analyzer.FINAL_STEP else 64,
                    }
                )
            conflict_n = 384 if step == analyzer.FINAL_STEP else 96
            for prompt_view in ("full", "audit_law_matched"):
                iid_rows.append(
                    {
                        "run_id": run_id,
                        "step": step,
                        "prompt_view": prompt_view,
                        "split": split,
                        "kind": "behavioral_agreement",
                        "panel": "conflict",
                        "rho_y": rho_y,
                        "rho_p": 0.75,
                        "rho_q": 0.55,
                        "n": conflict_n,
                    }
                )
            iid_rows.append(
                {
                    "run_id": run_id,
                    "step": step,
                    "prompt_view": "full",
                    "split": "iid_validation",
                    "kind": "behavioral_agreement",
                    "panel": "all",
                    "rho_y": 0.91 + 0.08 * progress,
                    "n": 1000,
                }
            )
            iid_rows.append(
                {
                    "run_id": run_id,
                    "step": step,
                    "prompt_view": "audit_law_matched",
                    "split": "iid_validation",
                    "kind": "behavioral_agreement",
                    "panel": "all",
                    "rho_y": 0.92 + 0.07 * progress,
                    "n": 1000,
                }
            )
        for choice_y, choice_p, choice_q in product((0, 1), repeat=3):
            factorial_rows.append(
                {
                    "run_id": run_id,
                    "step": analyzer.FINAL_STEP,
                    "prompt_view": "full",
                    "split": "final_factorial",
                    "choice_y": choice_y,
                    "choice_p": choice_p,
                    "choice_q": choice_q,
                    "action_b_rate": 0.1 + 0.1 * (4 * choice_y + 2 * choice_p + choice_q),
                    "n": 64,
                }
            )
    panel = AnalysisPanel(
        runs=pd.DataFrame(run_rows),
        raw_metrics=pd.DataFrame(iid_rows),
        predictions=pd.DataFrame(),
        trajectory=pd.DataFrame(trajectory_rows),
        causal_effects=pd.DataFrame(causal_rows),
        factorial=pd.DataFrame(factorial_rows),
        audit={"implementation_fingerprint": "synthetic-fingerprint"},
    )
    return panel, config


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


def test_synthetic_panel_reports_raw_endpoints_trajectories_and_registered_contrasts() -> None:
    panel, config = _synthetic_panel()
    report = analyzer.analyze_panel(panel, config, config_sha256="frozen-config")

    assert report["panel"] == {
        "completed_only": True,
        "expected_run_count": 36,
        "observed_run_count": 36,
        "seed_blocks": list(analyzer.SEEDS),
        "models": list(analyzer.MODELS),
        "registered_eval_steps": list(analyzer.EVAL_STEPS),
        "expected_config_sha256": "frozen-config",
        "registered_config_identity_sha256": analyzer.REGISTERED_CONFIG_IDENTITY_SHA256,
        "implementation_fingerprint": "synthetic-fingerprint",
    }
    assert len(report["seed_endpoints"]) == 36
    assert len(report["seed_trajectories"]) == 36
    assert Counter(row["seed"] for row in report["seed_endpoints"]) == Counter(
        {seed: 6 for seed in analyzer.SEEDS}
    )
    for trajectory in report["seed_trajectories"]:
        assert [row["step"] for row in trajectory["checkpoints"]] == list(analyzer.EVAL_STEPS)
        assert len(trajectory["final_full_truth_table"]) == 8
        assert {
            (row["choice_y"], row["choice_p"], row["choice_q"])
            for row in trajectory["final_full_truth_table"]
        } == set(product((0, 1), repeat=3))
        for checkpoint in trajectory["checkpoints"]:
            assert set(checkpoint["full_conflict_agreement"]) == {"rho_y", "rho_p", "rho_q"}
            assert set(checkpoint["full_causal_action_flip"]) == {
                "flip_y",
                "flip_p",
                "flip_q",
                "flip_d",
            }
            assert "iid_full_rho_y" in checkpoint
            assert "audit_law_matched_conflict_rho_y" in checkpoint
            final = checkpoint["step"] == analyzer.FINAL_STEP
            assert checkpoint["full_conflict_n"] == (384 if final else 96)
            assert checkpoint["audit_law_matched_conflict_n"] == (384 if final else 96)
            assert checkpoint["full_causal_n_pairs_per_target"] == (128 if final else 64)
            assert checkpoint["iid_full_n"] == 1000

    expected_effects = {"primary_contrast": 0.12, "registered_secondary_contrast": 0.07}
    for name, expected_effect in expected_effects.items():
        contrast = report[name]
        assert len(contrast["per_model"]) == 2
        for model_report in contrast["per_model"]:
            assert len(model_report["raw_paired_endpoints"]) == 6
            assert model_report["rho_y_effect"]["mean"] == pytest.approx(expected_effect)
            assert model_report["causal_companion_c_effect"]["mean"] == pytest.approx(expected_effect)
            raw = model_report["raw_paired_endpoints"][0]["comparison"]
            assert {
                "rho_y",
                "rho_p",
                "rho_q",
                "flip_y",
                "flip_p",
                "flip_q",
                "flip_d",
                "iid_full_rho_y",
                "audit_law_matched_conflict_rho_y",
                "full_truth_table",
                "full_conflict_n",
                "full_causal_n_pairs_per_target",
                "iid_full_n",
                "audit_law_matched_conflict_n",
            }.issubset(raw)
            assert len(raw["full_truth_table"]) == 8
        pooled = contrast["pooled_equal_weight_across_models_within_seed"]
        assert pooled["rho_y_effect"]["mean"] == pytest.approx(expected_effect)
        assert pooled["causal_companion_c_effect"]["mean"] == pytest.approx(expected_effect)

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
    with pytest.raises(analyzer.Qwen35EvidenceGeometryAnalysisError):
        analyzer.analyze_panel(replace(panel, runs=rows), config)


@pytest.mark.parametrize("mutation", ["condition", "seed", "cell_id"])
def test_run_identity_mutations_fail_closed(mutation: str) -> None:
    panel, config = _synthetic_panel()
    rows = panel.runs.copy(deep=True)
    if mutation == "condition":
        resolved = copy.deepcopy(rows.at[0, "config"])
        resolved["model"]["name"] = analyzer.MODELS[1]
        rows.at[0, "config"] = resolved
    elif mutation == "seed":
        rows.loc[rows.index[0], "seed"] = -1
    else:
        rows.loc[rows.index[0], "cell_id"] = "unexpected-cell"
    with pytest.raises(analyzer.Qwen35EvidenceGeometryAnalysisError):
        analyzer.analyze_panel(replace(panel, runs=rows), config)


@pytest.mark.parametrize(
    ("table", "mutation"),
    [
        ("trajectory", "missing_audit"),
        ("trajectory", "duplicate_full"),
        ("trajectory", "unexpected_step"),
        ("causal_effects", "missing_d"),
        ("causal_effects", "unexpected_causal_view"),
        ("causal_effects", "wrong_causal_count"),
        ("raw_metrics", "missing_iid"),
        ("raw_metrics", "duplicate_iid"),
        ("raw_metrics", "wrong_iid_count"),
        ("raw_metrics", "wrong_conflict_count"),
        ("factorial", "missing_truth_cell"),
        ("factorial", "duplicate_truth_cell"),
        ("factorial", "invalid_truth_count"),
    ],
)
def test_incomplete_or_unregistered_metric_axes_fail_closed(table: str, mutation: str) -> None:
    panel, config = _synthetic_panel()
    frame = getattr(panel, table).copy()
    first_run = str(panel.runs.iloc[0]["run_id"])
    if mutation == "missing_audit":
        frame = frame[
            ~(
                (frame["run_id"] == first_run)
                & (frame["step"] == analyzer.EVAL_STEPS[0])
                & (frame["prompt_view"] == "audit_law_matched")
            )
        ].copy()
    elif mutation == "duplicate_full":
        duplicate = frame[
            (frame["run_id"] == first_run)
            & (frame["step"] == analyzer.EVAL_STEPS[0])
            & (frame["prompt_view"] == "full")
        ].iloc[[0]]
        frame = pd.concat([frame, duplicate], ignore_index=True)
    elif mutation == "unexpected_step":
        extra = frame[(frame["run_id"] == first_run) & (frame["step"] == analyzer.EVAL_STEPS[0])].copy()
        extra.loc[:, "step"] = 999
        frame = pd.concat([frame, extra], ignore_index=True)
    elif mutation == "missing_d":
        frame = frame[
            ~(
                (frame["run_id"] == first_run)
                & (frame["step"] == analyzer.EVAL_STEPS[0])
                & (frame["target"] == "d")
            )
        ].copy()
    elif mutation == "unexpected_causal_view":
        extra = (
            frame[(frame["run_id"] == first_run) & (frame["step"] == analyzer.EVAL_STEPS[0])].iloc[[0]].copy()
        )
        extra.loc[:, "prompt_view"] = "audit_law_matched"
        frame = pd.concat([frame, extra], ignore_index=True)
    elif mutation == "wrong_causal_count":
        index = frame[(frame["run_id"] == first_run) & (frame["step"] == analyzer.EVAL_STEPS[0])].index[0]
        frame.loc[index, "n_pairs"] = 1
    elif mutation == "missing_iid":
        frame = frame[
            ~(
                (frame["run_id"] == first_run)
                & (frame["step"] == analyzer.EVAL_STEPS[0])
                & (frame["split"] == "iid_validation")
            )
        ].copy()
    elif mutation == "duplicate_iid":
        duplicate = frame[
            (frame["run_id"] == first_run)
            & (frame["step"] == analyzer.EVAL_STEPS[0])
            & (frame["prompt_view"] == "full")
            & (frame["split"] == "iid_validation")
            & (frame["panel"] == "all")
        ].iloc[[0]]
        frame = pd.concat([frame, duplicate], ignore_index=True)
    elif mutation == "wrong_iid_count":
        index = frame[
            (frame["run_id"] == first_run)
            & (frame["step"] == analyzer.EVAL_STEPS[0])
            & (frame["prompt_view"] == "full")
            & (frame["split"] == "iid_validation")
            & (frame["panel"] == "all")
        ].index[0]
        frame.loc[index, "n"] = 1
    elif mutation == "wrong_conflict_count":
        index = frame[
            (frame["run_id"] == first_run)
            & (frame["step"] == analyzer.EVAL_STEPS[0])
            & (frame["prompt_view"] == "full")
            & (frame["split"] == "diagnostic_factorial")
            & (frame["panel"] == "conflict")
        ].index[0]
        frame.loc[index, "n"] = 1
    elif mutation == "missing_truth_cell":
        frame = frame.drop(frame[frame["run_id"] == first_run].index[0]).copy()
    elif mutation == "duplicate_truth_cell":
        duplicate = frame[frame["run_id"] == first_run].iloc[[0]]
        frame = pd.concat([frame, duplicate], ignore_index=True)
    else:
        index = frame[frame["run_id"] == first_run].index[0]
        frame.loc[index, "n"] = 1
    with pytest.raises(analyzer.Qwen35EvidenceGeometryAnalysisError):
        analyzer.analyze_panel(replace(panel, **{table: frame}), config)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("experiment", "id"), "changed"),
        (("run", "seeds"), [22011, 22013, 22017, 22019, 22023]),
        (("train", "algorithm"), "sft"),
        (("train", "eval_steps"), [0, 1, 4, 16, 64, 256, 1000]),
        (("evaluation", "prompt_views"), ["full"]),
    ],
)
def test_registered_config_mutations_fail_closed(path: tuple[str, str], value: Any) -> None:
    config = load_config(CONFIG)
    config[path[0]][path[1]] = value
    with pytest.raises(analyzer.Qwen35EvidenceGeometryAnalysisError):
        analyzer._expected_plan(config)


def test_nonfinite_or_out_of_range_endpoint_fails_closed() -> None:
    panel, config = _synthetic_panel()
    trajectory = panel.trajectory.copy()
    trajectory.loc[trajectory.index[0], "rho_y"] = float("nan")
    with pytest.raises(analyzer.Qwen35EvidenceGeometryAnalysisError, match="finite"):
        analyzer.analyze_panel(replace(panel, trajectory=trajectory), config)


def test_loader_rejects_incomplete_directories_before_completed_only_load(
    tmp_path: Path,
) -> None:
    config = load_config(CONFIG)
    run = tmp_path / "experiment" / "run"
    run.mkdir(parents=True)
    (run / "identity.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(analyzer.Qwen35EvidenceGeometryAnalysisError, match="incomplete run directory"):
        analyzer._load_exact_panel(tmp_path, config)


def test_cli_stdout_is_only_canonical_json(monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    monkeypatch.setattr(analyzer, "analyze_main", lambda _path: {"z": 0, "a": 1})
    assert analyzer.main(["unused"]) == 0
    captured = capsys.readouterr()
    assert captured.out == '{"a":1,"z":0}\n'
    assert captured.err == ""
