"""Frozen-grid and reduced-plan checks for identical-evidence ordering."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from forkworld.config import ConfigError, expand_sweep, load_config, validate_config
from forkworld.runner import planned_runs, smoke_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/e18_identical_evidence_order.yaml"
SCHEDULES = ["b_then_d", "d_then_b", "interleave"]
CONFIRMATORY_SEEDS = [
    269,
    271,
    277,
    281,
    283,
    293,
    307,
    311,
    313,
    317,
    331,
    337,
    347,
    349,
    353,
    359,
    367,
    373,
    379,
    383,
]
PILOT_SEEDS = [389, 397, 401]


def test_identical_evidence_plan_is_frozen_three_by_twenty_grid() -> None:
    config = load_config(CONFIG_PATH)
    cells = expand_sweep(config)

    assert config["run"]["seeds"] == CONFIRMATORY_SEEDS
    assert config["h15"]["pilot_seeds"] == PILOT_SEEDS
    assert not set(CONFIRMATORY_SEEDS) & set(PILOT_SEEDS)
    assert len(cells) == 3
    assert [cell["h15"]["schedule"] for cell in cells] == SCHEDULES
    assert len(planned_runs(config)) == 60
    assert len(planned_runs(config, seed_override=PILOT_SEEDS)) == 9

    for cell in cells:
        h15 = cell["h15"]
        assert h15["prefix_a_repetitions"] == 90
        assert h15["washout_a_repetitions"] == 10
        assert h15["diagnostic_repetitions"] == 100
        assert h15["prefix_steps"] == 3240
        assert h15["block_steps"] == 200
        assert h15["washout_steps"] == 360
        assert h15["total_steps"] == cell["train"]["steps"] == 4000
        assert cell["train"]["shuffle"] is False
        assert h15["auc_horizon"] == 128
        assert 128 in h15["washout_checkpoints"]
        assert h15["late_stability_steps"] == [222, 304, 360]
        assert h15["primary_effect_threshold"] == 0.10
        assert h15["minimum_sign_count"] == 15


def test_h15_validation_rejects_order_confound_and_frozen_design_drift() -> None:
    config = load_config(CONFIG_PATH)

    shuffled = copy.deepcopy(config)
    shuffled["train"]["shuffle"] = True
    with pytest.raises(ConfigError, match="atomic-batch order"):
        validate_config(shuffled)

    unequal_exposure = copy.deepcopy(config)
    unequal_exposure["h15"]["diagnostic_repetitions"] = 99
    with pytest.raises(ConfigError, match="same total presentations"):
        validate_config(unequal_exposure)

    missing_direct_auc = copy.deepcopy(config)
    missing_direct_auc["h15"]["washout_checkpoints"].remove(128)
    with pytest.raises(ConfigError, match="auc_horizon"):
        validate_config(missing_direct_auc)

    reused_pilot = copy.deepcopy(config)
    reused_pilot["h15"]["pilot_seeds"][0] = CONFIRMATORY_SEEDS[0]
    with pytest.raises(ConfigError, match="disjoint"):
        validate_config(reused_pilot)

    bad_schedule = copy.deepcopy(config)
    bad_schedule["h15"]["schedule"] = "random_shuffle"
    with pytest.raises(ConfigError, match=r"h15\.schedule"):
        validate_config(bad_schedule)


def test_identical_evidence_smoke_plan_preserves_causal_structure() -> None:
    base = load_config(CONFIG_PATH)
    config = smoke_config(base)

    assert len(planned_runs(base, smoke=True)) == 1
    assert config["h15"]["schedule"] == "b_then_d"
    assert {key: config["data"][key] for key in ("n_train", "n_validation", "n_eval")} == {
        "n_train": 256,
        "n_validation": 128,
        "n_eval": 256,
    }
    assert config["train"]["batch_size"] == 32
    assert config["train"]["steps"] == 24
    assert config["train"]["shuffle"] is False
    assert config["h15"]["q_p"] == 0.75
    assert config["h15"]["q_q"] == 0.875
    assert config["h15"]["prefix_a_repetitions"] == 2
    assert config["h15"]["washout_a_repetitions"] == 1
    assert config["h15"]["diagnostic_repetitions"] == 3
    assert config["h15"]["prefix_steps"] == 12
    assert config["h15"]["block_steps"] == 3
    assert config["h15"]["washout_steps"] == 6
    assert config["h15"]["total_steps"] == 24
    assert config["h15"]["washout_checkpoints"] == [0, 1, 2, 3, 4, 5, 6]
    assert config["h15"]["auc_horizon"] == 3
    assert config["h15"]["late_stability_steps"] == [3, 5, 6]
    validate_config(config)
