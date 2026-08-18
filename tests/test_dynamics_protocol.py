"""Integration checks for the adaptive multi-goal temporal bridge."""

from __future__ import annotations

from pathlib import Path

from forkworld.config import load_config
from forkworld.protocols import run_protocol
from forkworld.runner import smoke_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_multigoal_dynamics_smoke_is_self_contained_and_instrumented() -> None:
    config = smoke_config(load_config(REPO_ROOT / "configs/e15_multigoal_dynamics.yaml"))
    config["run"]["device"] = "cpu"
    result = run_protocol(config, seed=0)

    assert result.summary["hypothesis"] == "h12"
    assert result.summary["design_status"] == "adaptive_posthoc_with_fresh_seed_replication"
    assert result.summary["data"]["factorial_interface_matches"] is True
    assert result.summary["data"]["probe_splits_disjoint"] is True
    assert result.summary["bridge"]["available"] is True
    assert result.summary["bridge"]["step"] == 16
    assert result.summary["final"]["dynamics"] is not None
    assert result.summary["model"]["competition"]["input_dim"] == 19
    assert set(result.summary["final"]["calibration"]) == {"P", "Q", "Y"}

    stages = {record["stage"] for record in result.metrics}
    assert {
        "competition_probe",
        "competition_behavior",
        "competition_truth_table",
        "competition_causal",
        "competition_candidates",
        "competition_bridge",
    } <= stages
    assert result.evaluation_batch is not None
