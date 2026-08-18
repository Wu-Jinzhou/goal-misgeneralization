"""Frozen-design and integration checks for the winner-knockout handoff."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from forkworld.config import ConfigError, expand_sweep, load_config, validate_config
from forkworld.protocols import run_protocol
from forkworld.runner import planned_runs, smoke_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/e17_winner_knockout.yaml"
TRAJECTORIES = [
    "independent_carry",
    "independent_reset",
    "nested_carry",
    "nested_reset",
    "scratch",
    "sham",
]
SEEDS = [
    157,
    163,
    167,
    173,
    179,
    181,
    191,
    193,
    197,
    199,
    211,
    223,
    227,
    229,
    233,
    239,
    241,
    251,
    257,
    263,
]


def test_winner_knockout_plan_is_frozen_six_by_twenty_grid() -> None:
    config = load_config(CONFIG_PATH)
    cells = expand_sweep(config)

    assert config["run"]["seeds"] == SEEDS
    assert len(cells) == 6
    assert [cell["h14"]["trajectory"] for cell in cells] == TRAJECTORIES
    assert len(planned_runs(config)) == 120
    assert "sweep" not in config or config["sweep"] == {}

    for cell in cells:
        assert cell["h14"]["phase_a_steps"] == 45
        assert cell["h14"]["phase_b_steps"] == 1024
        assert cell["train"]["steps"] == 1024
        assert cell["h14"]["eligibility_steps"] == [33, 45]
        assert cell["h14"]["auc_horizon"] == 128
        assert 128 in cell["h14"]["phase_b_checkpoints"]


def test_h14_validation_requires_direct_auc_checkpoint_and_matching_horizon() -> None:
    config = load_config(CONFIG_PATH)

    missing_auc_checkpoint = copy.deepcopy(config)
    missing_auc_checkpoint["h14"]["phase_b_checkpoints"].remove(128)
    with pytest.raises(ConfigError, match="auc_horizon"):
        validate_config(missing_auc_checkpoint)

    wrong_auc_horizon = copy.deepcopy(config)
    wrong_auc_horizon["h14"]["auc_horizon"] = 117
    with pytest.raises(ConfigError, match="directly observed local checkpoint 128"):
        validate_config(wrong_auc_horizon)

    mismatched_horizon = copy.deepcopy(config)
    mismatched_horizon["train"]["steps"] = 1069
    with pytest.raises(ConfigError, match=r"phase_b_steps must equal train\.steps"):
        validate_config(mismatched_horizon)

    unknown_arm = copy.deepcopy(config)
    unknown_arm["h14"]["trajectory"] = "independent_warm_start"
    with pytest.raises(ConfigError, match="six frozen handoff trajectories"):
        validate_config(unknown_arm)


def test_winner_knockout_smoke_preserves_scientific_structure() -> None:
    config = smoke_config(load_config(CONFIG_PATH))

    assert {key: config["data"][key] for key in ("n_train", "n_validation", "n_eval")} == {
        "n_train": 256,
        "n_validation": 128,
        "n_eval": 256,
    }
    assert config["train"]["steps"] == 16
    assert config["train"]["batch_size"] == 64
    expected_h14 = {
        "q_p": 0.75,
        "q_q": 0.75,
        "k_q": 2,
        "k_y": 2,
        "phase_a_steps": 4,
        "phase_b_steps": 16,
        "eligibility_steps": [3, 4],
        "probe_train_n": 128,
        "probe_eval_n": 256,
        "sham_source_repeats": 4,
        "auc_horizon": 8,
    }
    assert {key: config["h14"][key] for key in expected_h14} == expected_h14
    assert config["h14"]["phase_b_checkpoints"] == [
        0,
        1,
        2,
        3,
        4,
        5,
        7,
        8,
        9,
        13,
        16,
    ]
    validate_config(config)


@pytest.mark.parametrize(
    ("trajectory", "phase_a_stage"),
    [
        ("independent_carry", "phase_a_optimization"),
        ("sham", "phase_a_sham_optimization"),
    ],
)
def test_winner_knockout_smoke_audits_handoff_contract(trajectory: str, phase_a_stage: str) -> None:
    config = smoke_config(load_config(CONFIG_PATH))
    config["h14"]["trajectory"] = trajectory
    config["run"]["device"] = "cpu"
    validate_config(config)

    result = run_protocol(config, seed=0)
    summary = result.summary

    assert summary["hypothesis"] == "h14"
    assert summary["trajectory"] == trajectory
    assert summary["data"]["phase_b_batch_digest"]
    assert summary["data"]["phase_b_sampler_digest"]
    assert summary["data"]["phase_b_pairing_verified"] is True
    assert summary["data"]["probe_splits_disjoint"] is True
    assert summary["measurement"]["phase_b_checkpoints"] == [
        0,
        1,
        2,
        3,
        4,
        5,
        7,
        8,
        9,
        13,
        16,
    ]
    assert summary["measurement"]["direct_checkpoint_128"] is False
    assert summary["training"]["all_minibatches_full"] is True
    assert summary["training"]["phase_a_steps"] == 4
    assert summary["training"]["phase_b_steps"] == 16
    assert all(summary["replay"]["checks"].values())

    stages = {record["stage"] for record in result.metrics}
    assert {
        phase_a_stage,
        "phase_b_behavior",
        "phase_b_truth_table",
        "phase_b_probe",
        "phase_b_causal",
        "phase_b_optimization",
    } <= stages
    assert {record["experiment"] for record in result.metrics} == {"h14"}
    assert result.evaluation_batch is not None
