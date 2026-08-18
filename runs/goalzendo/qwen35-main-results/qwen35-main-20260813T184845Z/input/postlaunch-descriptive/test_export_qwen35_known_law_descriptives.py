from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from dataclasses import replace
from itertools import product
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd
import pytest

from goalzendo.analysis import AnalysisPanel
from goalzendo.config import load_config
from goalzendo.runner import build_plan

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "export_qwen35_known_law_descriptives.py"
CONFIG = ROOT / "configs" / "goalzendo" / "qwen35_known_law_main.yaml"
FROZEN_ANALYZER = ROOT / "scripts" / "analyze_qwen35_known_law.py"


def _load_script() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "export_qwen35_known_law_descriptives_for_test",
        SCRIPT,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


exporter = _load_script()


def _synthetic_panel() -> tuple[AnalysisPanel, dict[str, Any]]:
    config = load_config(CONFIG)
    run_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    causal_rows: list[dict[str, Any]] = []
    factorial_rows: list[dict[str, Any]] = []
    for index, spec in enumerate(build_plan(config)):
        resolved = copy.deepcopy(spec.config)
        resolved["seed"] = spec.seed
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
        for step in exporter.EXPECTED_STEPS:
            split = "final_factorial" if step == 1000 else "diagnostic_factorial"
            trajectory_rows.extend(
                (
                    {
                        "run_id": run_id,
                        "cell_id": spec.cell_id,
                        "seed": spec.seed,
                        "step": step,
                        "prompt_view": "full",
                        "split": split,
                        "panel": "conflict",
                        "rho_y": 0.0,
                        "rho_p": 1.0,
                        "rho_q": 0.0,
                    },
                    {
                        "run_id": run_id,
                        "cell_id": spec.cell_id,
                        "seed": spec.seed,
                        "step": step,
                        "prompt_view": "audit_law_matched",
                        "split": split,
                        "panel": "conflict",
                        "rho_y": 0.75,
                        "rho_p": 0.25,
                        "rho_q": 0.25,
                    },
                )
            )
            for target, flip_rate in (("y", 0.10), ("p", 0.80), ("q", 0.10), ("d", 0.05)):
                causal_rows.append(
                    {
                        "run_id": run_id,
                        "cell_id": spec.cell_id,
                        "seed": spec.seed,
                        "step": step,
                        "prompt_view": "full",
                        "split": split,
                        "target": target,
                        "action_flip_rate": flip_rate,
                    }
                )
            for prompt_view in exporter.PROMPT_VIEWS:
                for choice_y, choice_p, choice_q in product((0, 1), repeat=3):
                    factorial_rows.append(
                        {
                            "run_id": run_id,
                            "cell_id": spec.cell_id,
                            "seed": spec.seed,
                            "step": step,
                            "prompt_view": prompt_view,
                            "split": split,
                            "choice_y": choice_y,
                            "choice_p": choice_p,
                            "choice_q": choice_q,
                            "action_b_rate": float(choice_p),
                            "n": 64 if step == 1000 else 16,
                        }
                    )
        trajectory_rows.append(
            {
                "run_id": run_id,
                "cell_id": spec.cell_id,
                "seed": spec.seed,
                "step": 1000,
                "prompt_view": "full",
                "split": "iid_validation",
                "panel": "all",
                "rho_y": 0.99,
                "rho_p": 0.95,
                "rho_q": 0.90,
            }
        )
    return (
        AnalysisPanel(
            runs=pd.DataFrame(run_rows),
            raw_metrics=pd.DataFrame(),
            predictions=pd.DataFrame(),
            trajectory=pd.DataFrame(trajectory_rows),
            causal_effects=pd.DataFrame(causal_rows),
            factorial=pd.DataFrame(factorial_rows),
            audit={"implementation_fingerprint": "synthetic-fingerprint"},
        ),
        config,
    )


def _export(panel: AnalysisPanel, config: dict[str, Any]) -> dict[str, Any]:
    return exporter.export_descriptives(
        panel,
        config,
        config_sha256=exporter.MAIN_CONFIG_SHA256,
        analyzer_sha256=exporter.FROZEN_ANALYZER_SHA256,
        expected_implementation_fingerprint="synthetic-fingerprint",
    )


def test_companion_exports_every_promised_descriptive_without_inference() -> None:
    panel, config = _synthetic_panel()
    report = _export(panel, config)

    assert report["epistemic_status"] == {
        "analysis_role": "descriptive_companion_to_prelaunch_frozen_primary_analysis",
        "materializes_prelaunch_defined_measurements_from_registered_checkpoints": True,
        "adds_inferential_hypotheses": False,
        "computes_p_values_or_confidence_intervals": False,
    }
    assert report["panel"] == {
        "completed_only": True,
        "expected_run_count": 96,
        "observed_run_count": 96,
        "evaluation_steps": list(exporter.EXPECTED_STEPS),
        "prompt_views": list(exporter.PROMPT_VIEWS),
        "full_view_causal_targets": list(exporter.CAUSAL_TARGETS),
        "expected_config_sha256": exporter.MAIN_CONFIG_SHA256,
        "frozen_primary_analyzer_sha256": exporter.FROZEN_ANALYZER_SHA256,
        "expected_implementation_fingerprint": "synthetic-fingerprint",
        "implementation_fingerprint": "synthetic-fingerprint",
    }
    assert len(report["runs"]) == 96
    first = report["runs"][0]
    assert first["final_full_iid_law_accuracy"] == 0.99
    assert first["final_audit_law_matched_conflict_rho_y"] == 0.75
    assert len(first["conflict_agreement_trajectories"]) == 8
    assert len(first["full_view_causal_action_flip_trajectories"]) == 8
    assert len(first["full_view_controller_trajectory"]) == 8
    assert first["full_view_controller_trajectory"][0] == {
        "step": 0,
        "raw_controller": "P",
        "persistent_controller": None,
    }
    assert first["full_view_controller_trajectory"][1] == {
        "step": 1,
        "raw_controller": "P",
        "persistent_controller": "P",
    }
    assert first["full_view_acquisition_intervals"] == [
        {
            "candidate": "Y",
            "left_step": 1000,
            "right_step": None,
            "confirmation_step": None,
            "censoring": "right",
        },
        {
            "candidate": "P",
            "left_step": None,
            "right_step": 0,
            "confirmation_step": 1,
            "censoring": "left",
        },
        {
            "candidate": "Q",
            "left_step": 1000,
            "right_step": None,
            "confirmation_step": None,
            "censoring": "right",
        },
    ]
    serialized = exporter.canonical_json(report)
    assert serialized.endswith("\n") and serialized.count("\n") == 1
    assert json.loads(serialized) == report
    assert '"p_value":' not in serialized and '"confidence_interval":' not in serialized


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_behavior",
        "duplicate_audit",
        "missing_iid",
        "missing_causal",
        "missing_factorial",
        "missing_matched_factorial",
    ),
)
def test_companion_fails_closed_on_incomplete_rows(mutation: str) -> None:
    panel, config = _synthetic_panel()
    run_id = str(panel.runs.iloc[0]["run_id"])
    if mutation == "missing_behavior":
        frame = panel.trajectory[
            ~(
                (panel.trajectory["run_id"] == run_id)
                & (panel.trajectory["step"] == 0)
                & (panel.trajectory["prompt_view"] == "full")
                & (panel.trajectory["panel"] == "conflict")
            )
        ].copy()
        panel = replace(panel, trajectory=frame)
    elif mutation == "duplicate_audit":
        row = panel.trajectory[
            (panel.trajectory["run_id"] == run_id)
            & (panel.trajectory["step"] == 0)
            & (panel.trajectory["prompt_view"] == "audit_law_matched")
            & (panel.trajectory["panel"] == "conflict")
        ].iloc[[0]]
        panel = replace(panel, trajectory=pd.concat([panel.trajectory, row], ignore_index=True))
    elif mutation == "missing_iid":
        frame = panel.trajectory[
            ~((panel.trajectory["run_id"] == run_id) & (panel.trajectory["split"] == "iid_validation"))
        ].copy()
        panel = replace(panel, trajectory=frame)
    elif mutation == "missing_causal":
        frame = panel.causal_effects[
            ~(
                (panel.causal_effects["run_id"] == run_id)
                & (panel.causal_effects["step"] == 0)
                & (panel.causal_effects["target"] == "d")
            )
        ].copy()
        panel = replace(panel, causal_effects=frame)
    elif mutation == "missing_factorial":
        index = panel.factorial[(panel.factorial["run_id"] == run_id) & (panel.factorial["step"] == 0)].index[
            0
        ]
        panel = replace(panel, factorial=panel.factorial.drop(index=index).copy())
    else:
        index = panel.factorial[
            (panel.factorial["run_id"] == run_id)
            & (panel.factorial["step"] == 0)
            & (panel.factorial["prompt_view"] == "audit_law_matched")
        ].index[0]
        panel = replace(panel, factorial=panel.factorial.drop(index=index).copy())

    with pytest.raises(exporter.Qwen35DescriptiveError):
        _export(panel, config)


@pytest.mark.parametrize(
    ("table", "column"),
    (("trajectory", "rho_y"), ("causal_effects", "action_flip_rate"), ("factorial", "action_b_rate")),
)
def test_companion_rejects_nonfinite_or_out_of_range_values(table: str, column: str) -> None:
    panel, config = _synthetic_panel()
    frame = getattr(panel, table).copy()
    frame.loc[frame.index[0], column] = 1.1
    panel = replace(panel, **{table: frame})
    with pytest.raises(exporter.Qwen35DescriptiveError, match=r"\[0, 1\]"):
        _export(panel, config)


def test_companion_binds_the_unchanged_primary_analyzer_and_config() -> None:
    assert hashlib.sha256(FROZEN_ANALYZER.read_bytes()).hexdigest() == (exporter.FROZEN_ANALYZER_SHA256)
    assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == exporter.MAIN_CONFIG_SHA256
    exporter._verify_frozen_bindings()


def test_companion_rejects_a_panel_from_a_different_implementation() -> None:
    panel, config = _synthetic_panel()
    panel = replace(panel, audit={"implementation_fingerprint": "different-source"})
    with pytest.raises(exporter.Qwen35DescriptiveError, match="implementation fingerprint"):
        _export(panel, config)


def test_cli_stdout_is_only_canonical_json(monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    monkeypatch.setattr(exporter, "analyze_main", lambda _path: {"z": 0, "a": 1})
    assert exporter.main(["unused"]) == 0
    captured = capsys.readouterr()
    assert captured.out == '{"a":1,"z":0}\n'
    assert captured.err == ""
