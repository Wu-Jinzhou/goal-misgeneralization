"""End-to-end protocol checks for the preregistered follow-up families."""

from __future__ import annotations

from pathlib import Path

from forkworld.config import load_config
from forkworld.protocols import run_protocol
from forkworld.runner import smoke_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_three_goal_protocol_calibrates_matched_interfaces_and_reports_identifying_panels() -> None:
    config = smoke_config(load_config(REPO_ROOT / "configs/e12_competing_goals.yaml"))
    config["run"]["device"] = "cpu"
    result = run_protocol(config, seed=0)

    assert result.summary["hypothesis"] == "h10"
    assert result.summary["data"]["calibration_interface_matches"] is True
    assert result.summary["data"]["calibration_evaluation_uses_masked_interface"] is True
    reports = result.summary["model"]
    input_dims = {
        reports["competition"]["input_dim"],
        *(item["input_dim"] for item in reports["calibration"].values()),
    }
    parameter_counts = {
        reports["competition"]["total_parameters"],
        *(item["total_parameters"] for item in reports["calibration"].values()),
    }
    assert len(input_dims) == 1
    assert len(parameter_counts) == 1
    assert set(result.summary["final"]["competition"]) == {
        "iid",
        "both_wrong",
        "p_wrong",
        "q_wrong",
    }
    for panel in ("both_wrong", "p_wrong", "q_wrong"):
        assert {"flip_P", "Q_mean", "Y_mean"} <= set(
            result.summary["final"]["interventions"][panel]
        )
    for name, metric in (("P", "rho_p"), ("Q", "rho_q"), ("Y", "rho_y_code")):
        values = result.summary["final"]["calibration"][name]["iid"]
        assert values["decoder_accuracy"] == values[metric]
    assert result.metrics
    assert result.evaluation_batch is not None


def test_routeworld_protocol_compounds_choices_and_confirms_physical_rollouts() -> None:
    config = smoke_config(load_config(REPO_ROOT / "configs/e13_routeworld.yaml"))
    config["run"]["device"] = "cpu"
    config["h11"]["route_depth"] = 2
    config["h11"]["max_depth"] = 2
    config["h11"]["evidence_regime"] = "per_fork_matched"
    result = run_protocol(config, seed=0)

    assert result.summary["hypothesis"] == "h11"
    assert result.summary["training"]["steps"] == 32
    assert result.summary["data"]["route_depth"] == 2
    final = result.summary["final"]
    assert set(final) >= {
        "iid",
        "all_conflict",
        "single_conflict_0",
        "single_conflict_1",
        "all_conflict_address_reversed",
        "all_conflict_address_removed",
        "interventions",
        "physical_rollouts",
    }
    assert final["physical_rollouts"]["matches_vectorized_success"] is True
    assert final["physical_rollouts"]["collision_rate"] == 0.0
    assert final["all_conflict"]["depth"] == 2
    assert {
        "flip_P_stage_local_mean",
        "flip_R1_stage_local_mean",
        "flip_P_other_stage_spillover_mean",
        "flip_R1_other_stage_spillover_mean",
        "stage_local",
    } <= set(final["interventions"])
    assert result.metrics
    # RouteWorld has its own vectorized decision batch, so the legacy one-step
    # navigation wrapper is intentionally not invoked by the artifact runner.
    assert result.evaluation_batch is None
