"""Design, continuation, and artifact-contract tests for exploratory E14."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import torch

from forkworld.config import ConfigError, load_config, validate_config
from forkworld.protocols import run_protocol
from forkworld.runner import planned_runs, smoke_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def _smoke_timing(schedule: str = "zero_zero") -> dict[str, Any]:
    config = smoke_config(load_config(REPO_ROOT / "configs/e14_entropy_timing.yaml"))
    # The frozen ForkWorld implementation predates MPS generator support and
    # its official deterministic smoke path is CPU.  Keep this test on that
    # registered backend instead of letting the shared ``auto`` default select
    # Apple Silicon's MPS device.
    config["run"]["device"] = "cpu"
    config["h5"]["timing_schedule"] = schedule
    validate_config(config)
    return config


def test_e14_fixed_grid_has_exactly_120_runs() -> None:
    config = load_config(REPO_ROOT / "configs/e14_entropy_timing.yaml")
    runs = planned_runs(config)

    assert len(runs) == 120
    assert {seed for _, seed in runs} == {11, 23, 37, 41, 53, 67, 71, 83, 97, 101}
    assert {
        (float(cell["data"]["q"]), int(cell["data"]["k"])) for cell, _ in runs
    } == {(0.75, 4), (0.90, 3)}
    assert {str(cell["h5"]["timing_schedule"]) for cell, _ in runs} == {
        "zero_zero",
        "high_high",
        "early_only",
        "delayed_carry",
        "delayed_actor_reset",
        "delayed_critic_reset",
    }
    cautions = validate_config(config)
    assert any("exploratory/post-hoc" in caution for caution in cautions)


def test_e14_rejects_an_unregistered_schedule() -> None:
    config = load_config(REPO_ROOT / "configs/e14_entropy_timing.yaml")
    config["h5"]["timing_schedule"] = "chosen_after_results"
    with pytest.raises(ConfigError, match="timing_schedule"):
        validate_config(config)


@pytest.mark.parametrize(("schedule", "beta"), [("zero_zero", 0.0), ("high_high", 0.30)])
def test_continuous_e14_schedule_matches_uninterrupted_h5_exactly(
    schedule: str,
    beta: float,
) -> None:
    timing_config = _smoke_timing(schedule)
    # Three 32-row updates leave the 64-row semantic sampler midway through an
    # epoch, so equality also verifies cursor/order continuation rather than
    # only generator continuation at a convenient epoch boundary.
    timing_config["h5"]["phase_a_steps"] = 3
    timing_config["h5"]["phase_b_steps"] = 13
    validate_config(timing_config)
    uninterrupted_config = copy.deepcopy(timing_config)
    uninterrupted_config["experiment"]["mode"] = "exploration_entropy"
    uninterrupted_config["h5"]["entropy_coefficient"] = beta
    validate_config(uninterrupted_config)

    timing = run_protocol(timing_config, seed=0)
    uninterrupted = run_protocol(uninterrupted_config, seed=0)

    for left, right in zip(
        timing.model.state_dict().values(),
        uninterrupted.model.state_dict().values(),
        strict=True,
    ):
        assert torch.equal(left, right)
    assert {
        name: value for name, value in timing.summary["final"].items() if name != "policy_entropy"
    } == uninterrupted.summary["final"]
    assert timing.summary["iid"] == uninterrupted.summary["iid"]


@pytest.mark.parametrize(
    ("schedule", "actor_carried", "critic_carried"),
    [
        ("delayed_actor_reset", False, True),
        ("delayed_critic_reset", True, False),
    ],
)
def test_e14_reset_schedules_are_isolated_and_audited(
    schedule: str,
    actor_carried: bool,
    critic_carried: bool,
) -> None:
    result = run_protocol(_smoke_timing(schedule), seed=0)
    audit = result.summary["schedule"]["reset_audit"]

    assert audit["actor_weights_reset"] is False
    assert audit["actor_optimizer_carried_observed"] is actor_carried
    assert audit["critic_optimizer_carried_observed"] is critic_carried
    assert audit["critic_object_reused_observed"] is True
    assert audit["data_generator_state_continued"] is True
    assert audit["action_generator_state_continued"] is True


def test_e14_smoke_emits_post_hoc_summary_and_phase_metrics() -> None:
    result = run_protocol(_smoke_timing("early_only"), seed=0)

    assert result.summary["design"]["registration_status"] == (
        "exploratory_post_hoc_after_complete_E10"
    )
    assert result.summary["design"]["confirmatory"] is False
    assert result.summary["schedule"]["phase_boundary_global_step"] == 4
    assert result.summary["costs"]["optimizer_steps"] == 16
    assert result.summary["costs"]["episode_or_example_presentations"] == 16 * 32
    assert {record["stage"] for record in result.metrics} >= {
        "phase_a",
        "phase_b",
        "phase_boundary",
        "final",
    }
    assert max(int(record["global_step"]) for record in result.metrics) == 16
    assert all(record["condition"] == "early_only" for record in result.metrics)
    assert len(result.predictions) == 128
    assert not result.checkpoints
