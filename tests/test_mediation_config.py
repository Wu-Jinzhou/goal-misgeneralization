"""Frozen-design checks for the active-Q input-pathway intervention."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from forkworld.config import ConfigError, expand_sweep, load_config, validate_config
from forkworld.protocols import run_protocol
from forkworld.runner import planned_runs, smoke_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/e19_q_pathway_mediation.yaml"
LAUNCHER_PATH = REPO_ROOT / "runs/20_q_pathway_mediation.sh"
BRANCHES = [
    "independent_noop",
    "independent_q_restore",
    "independent_padding_sham",
    "nested_noop",
    "nested_q_transplant",
    "nested_padding_sham",
]
CONFIRMATORY_SEEDS = [
    409,
    419,
    421,
    431,
    433,
    439,
    443,
    449,
    457,
    461,
    463,
    467,
    479,
    487,
    491,
    499,
    503,
    509,
    521,
    523,
]
PILOT_SEEDS = [541, 547, 557]
PHASE_B_CHECKPOINTS = [
    0,
    1,
    2,
    3,
    4,
    5,
    7,
    9,
    13,
    17,
    24,
    33,
    45,
    62,
    85,
    117,
    128,
    161,
    222,
    304,
    418,
    575,
    790,
    1024,
]


def test_active_q_pathway_plan_is_frozen_six_by_twenty_grid() -> None:
    config = load_config(CONFIG_PATH)
    cells = expand_sweep(config)

    assert config["run"]["seeds"] == CONFIRMATORY_SEEDS
    assert config["h16"]["pilot_seeds"] == PILOT_SEEDS
    assert not set(CONFIRMATORY_SEEDS) & set(PILOT_SEEDS)
    assert config["h16"]["pilot_only"] is False
    assert len(cells) == 6
    assert [cell["h16"]["branch"] for cell in cells] == BRANCHES
    assert len(planned_runs(config)) == 120

    pilot = copy.deepcopy(config)
    pilot["h16"]["pilot_only"] = True
    validate_config(pilot)
    pilot_plan = planned_runs(pilot, seed_override=PILOT_SEEDS)
    assert len(pilot_plan) == 18
    assert {seed for _, seed in pilot_plan} == set(PILOT_SEEDS)
    assert all(cell["h16"]["pilot_only"] is True for cell, _ in pilot_plan)
    with pytest.raises(ConfigError, match="pilot_only runs"):
        planned_runs(pilot)

    for cell in cells:
        h16 = cell["h16"]
        assert h16["phase_a_steps"] == 45
        assert h16["phase_b_steps"] == cell["train"]["steps"] == 1024
        assert h16["eligibility_steps"] == [33, 45]
        assert h16["phase_b_checkpoints"] == PHASE_B_CHECKPOINTS
        assert h16["auc_horizon"] == 128
        assert h16["active_q_columns"] == [8, 9]
        assert h16["sham_padding_columns"] == [5, 6]
        assert h16["expected_input_dim"] == 19
        assert h16["expected_edited_scalars"] == 128
        assert h16["primary_effect_threshold"] == 0.02
        assert h16["sham_auc_equivalence_margin"] == 0.01
        assert h16["minimum_sign_count"] == 15
        assert h16["minimum_eligible_seeds"] == 15


def test_h16_validation_rejects_seed_arm_cell_checkpoint_and_threshold_drift() -> None:
    config = load_config(CONFIG_PATH)

    reused_pilot = copy.deepcopy(config)
    reused_pilot["h16"]["pilot_seeds"][0] = CONFIRMATORY_SEEDS[0]
    with pytest.raises(ConfigError, match="disjoint"):
        validate_config(reused_pilot)

    changed_full_seed = copy.deepcopy(config)
    changed_full_seed["run"]["seeds"][-1] = 527
    with pytest.raises(ConfigError, match="frozen 20-seed"):
        validate_config(changed_full_seed)

    unknown_branch = copy.deepcopy(config)
    unknown_branch["h16"]["branch"] = "independent_random_edit"
    with pytest.raises(ConfigError, match="six frozen Q-pathway branches"):
        validate_config(unknown_branch)

    incomplete_grid = copy.deepcopy(config)
    incomplete_grid["cases"].pop()
    with pytest.raises(ConfigError, match="exact ordered six-branch"):
        validate_config(incomplete_grid)

    changed_cell = copy.deepcopy(config)
    changed_cell["train"]["learning_rate"] = 0.004
    with pytest.raises(ConfigError, match=r"train\.learning_rate"):
        validate_config(changed_cell)

    missing_auc = copy.deepcopy(config)
    missing_auc["h16"]["phase_b_checkpoints"].remove(128)
    with pytest.raises(ConfigError, match="auc_horizon"):
        validate_config(missing_auc)

    changed_threshold = copy.deepcopy(config)
    changed_threshold["h16"]["primary_effect_threshold"] = 0.021
    with pytest.raises(ConfigError, match=r"h16\.primary_effect_threshold"):
        validate_config(changed_threshold)

    changed_columns = copy.deepcopy(config)
    changed_columns["h16"]["active_q_columns"] = [8, 10]
    with pytest.raises(ConfigError, match="active Q and sham padding columns"):
        validate_config(changed_columns)


def test_active_q_pathway_smoke_preserves_intervention_surface() -> None:
    base = load_config(CONFIG_PATH)
    config = smoke_config(base)

    assert len(planned_runs(base, smoke=True)) == 1
    assert config["h16"]["branch"] == "independent_noop"
    assert {key: config["data"][key] for key in ("n_train", "n_validation", "n_eval")} == {
        "n_train": 256,
        "n_validation": 128,
        "n_eval": 256,
    }
    assert config["data"]["state_dim"] == 8
    assert config["data"]["max_k"] == 5
    assert config["model"]["width"] == 64
    assert config["model"]["depth"] == 2
    assert config["train"]["steps"] == 16
    assert config["train"]["batch_size"] == 64
    assert config["h16"]["q_p"] == config["h16"]["q_q"] == 0.75
    assert config["h16"]["k_q"] == config["h16"]["k_y"] == 2
    assert config["h16"]["phase_a_steps"] == 4
    assert config["h16"]["phase_b_steps"] == 16
    assert config["h16"]["eligibility_steps"] == [3, 4]
    assert config["h16"]["phase_b_checkpoints"] == [
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
    assert config["h16"]["auc_horizon"] == 8
    assert config["h16"]["feature_names"] == [
        "P",
        "P_present",
        "R_1",
        "R_2",
        "R_3",
        "R_4",
        "R_5",
        "Q_present",
        "Q_1",
        "Q_2",
        "Q_3",
        "state_0",
        "state_1",
        "state_2",
        "state_3",
        "state_4",
        "state_5",
        "state_6",
        "state_7",
    ]
    assert config["h16"]["active_q_columns"] == [8, 9]
    assert config["h16"]["sham_padding_columns"] == [5, 6]
    assert config["h16"]["expected_edited_scalars"] == 128
    validate_config(config)


def test_launcher_forces_distinct_pilot_identity() -> None:
    launcher = LAUNCHER_PATH.read_text(encoding="utf-8")

    assert "--seeds 541,547,557" in launcher
    assert "--set h16.pilot_only=true" in launcher
    assert "--set h16.pilot_only=false" in launcher
    assert launcher.index('"$@"') < launcher.index("--seeds 541,547,557")


def test_h16_dispatches_to_active_q_input_pathway_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import forkworld.protocols_mediation as protocol

    sentinel = object()
    dispatch_config = {"experiment": {"hypothesis": "h16"}}

    def fake_run_h16(config: object, seed: int) -> object:
        assert config is dispatch_config
        assert seed == 557
        return sentinel

    monkeypatch.setattr(protocol, "run_h16", fake_run_h16)
    result = run_protocol(dispatch_config, seed=557)
    assert result is sentinel
