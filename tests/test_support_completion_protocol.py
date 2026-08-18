"""Integration checks for the adaptive support-completion panel."""

from __future__ import annotations

from pathlib import Path

from forkworld.config import expand_sweep, load_config
from forkworld.protocols import run_protocol
from forkworld.runner import planned_runs, smoke_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_support_completion_plan_has_fixed_seven_by_twenty_grid() -> None:
    config = load_config(REPO_ROOT / "configs/e16_support_completion.yaml")
    cells = expand_sweep(config)

    assert len(cells) == 7
    assert [cell["h13"]["q_only_error_count"] for cell in cells] == [
        0,
        2,
        10,
        50,
        100,
        250,
        450,
    ]
    assert len(planned_runs(config)) == 140


def test_support_completion_smoke_is_instrumented_and_h13_labeled() -> None:
    config = smoke_config(load_config(REPO_ROOT / "configs/e16_support_completion.yaml"))
    config["run"]["device"] = "cpu"
    result = run_protocol(config, seed=0)

    assert result.summary["hypothesis"] == "h13"
    assert result.summary["design_status"] == "adaptive_posthoc_support_completion"
    assert result.summary["data"]["support_intervention_scope"] == "training_only"
    assert result.summary["data"]["q_only_error_count"] == 0
    assert result.summary["data"]["q_only_codeword_coverage"] == {
        "training_rows": 0,
        "seen_active_codewords": 0,
        "possible_active_codewords": 8,
        "factorial_seen_rows": 0,
        "factorial_unseen_rows": 32,
    }
    assert result.summary["training"]["q_only_presentations_by_checkpoint"]["16"] == 0
    assert result.summary["training"]["all_unique_q_only_seen_step"] == 0

    stages = {record["stage"] for record in result.metrics}
    assert {
        "support_completion_design",
        "support_completion_exposure",
        "support_completion_generalization",
        "competition_probe",
        "competition_truth_table",
        "competition_causal",
    } <= stages
    assert {record["experiment"] for record in result.metrics} == {"h13"}
