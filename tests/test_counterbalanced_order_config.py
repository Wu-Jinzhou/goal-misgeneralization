"""Frozen configuration and dispatch checks for E20/H17."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from forkworld.config import ConfigError, expand_sweep, load_config, validate_config
from forkworld.protocols import run_protocol
from forkworld.runner import planned_runs, smoke_config

REPO_ROOT = Path(__file__).resolve().parents[1]
FULL_CONFIG = REPO_ROOT / "configs/e20_counterbalanced_order.yaml"
PILOT_CONFIG = REPO_ROOT / "configs/e20_counterbalanced_order_pilot.yaml"
LAUNCHER = REPO_ROOT / "runs/21_counterbalanced_order.sh"
GOALS = ["P", "Q", "Y"]
SCHEDULES = ["p_q_y", "p_y_q", "q_p_y", "q_y_p", "y_p_q", "y_q_p"]
PILOT_SEEDS = [563, 569, 571]
FULL_SEEDS = [
    577,
    587,
    593,
    599,
    601,
    607,
    613,
    617,
    619,
    631,
    641,
    643,
    647,
    653,
    659,
    661,
    673,
    677,
    683,
    691,
]
CHECKPOINTS = [
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
    256,
]
FEATURE_NAMES = [
    "P",
    "P_present",
    "R_1",
    "R_2",
    "R_3",
    "Q_present",
    "Q_1",
    "Q_2",
]
DATA_SEED = 171_000_001
STREAM_SEED = 171_000_002
PHASE_SEEDS = {
    "component_P": 171_000_101,
    "component_Q": 171_000_102,
    "component_Y": 171_000_103,
    "washout": 171_000_104,
}


def test_h17_frozen_pilot_and_full_case_expansion() -> None:
    pilot = load_config(PILOT_CONFIG)
    pilot_cells = expand_sweep(pilot)
    assert pilot["experiment"]["name"] == "counterbalanced_identical_evidence_order_pilot"
    assert pilot["run"]["seeds"] == PILOT_SEEDS
    assert pilot["h17"]["full_seeds"] == FULL_SEEDS
    assert [cell["h17"]["isolated_goal"] for cell in pilot_cells] == GOALS
    assert all("schedule" not in cell["h17"] for cell in pilot_cells)
    assert len(planned_runs(pilot)) == 9

    full = load_config(FULL_CONFIG)
    full_cells = expand_sweep(full)
    assert full["experiment"]["name"] == "counterbalanced_identical_evidence_order"
    assert full["experiment"]["mode"] == "frozen_adaptive_posthoc_full_panel"
    assert full["run"]["seeds"] == FULL_SEEDS
    assert full["h17"]["pilot_seeds"] == PILOT_SEEDS
    assert [cell["h17"]["schedule"] for cell in full_cells] == SCHEDULES
    assert all("isolated_goal" not in cell["h17"] for cell in full_cells)
    assert len(planned_runs(full)) == 120

    for config in (pilot, full):
        assert not set(config["h17"]["pilot_seeds"]) & set(config["h17"]["full_seeds"])
        assert config["run"]["device"] == "cpu"
        assert config["run"]["save_checkpoints"] is False
        assert config["evaluation"]["save_predictions"] is False
        assert config["h17"]["feature_names"] == FEATURE_NAMES
        assert config["h17"]["component_checkpoints"] == CHECKPOINTS
        assert config["h17"]["washout_checkpoints"] == CHECKPOINTS
        assert config["h17"]["pilot_late_steps"] == [161, 222, 256]
        assert config["h17"]["primary_auc_window"] == [33, 128]
        assert config["h17"]["primary_ci_lower_boundary"] == 0.05
        assert config["h17"]["data_seed"] == DATA_SEED
        assert config["h17"]["stream_seed"] == STREAM_SEED
        assert config["h17"]["phase_seeds"] == PHASE_SEEDS
        assert config["model"]["bias"] is True
        assert config["model"]["nuisance_bits"] == 0
        assert config["train"]["grad_clip"] == 1.0
        assert config["train"]["label_smoothing"] == 0.0


def test_h17_validation_rejects_frozen_design_drift() -> None:
    full = load_config(FULL_CONFIG)

    bad_seed = copy.deepcopy(full)
    bad_seed["run"]["seeds"][-1] = 701
    with pytest.raises(ConfigError, match="frozen pilot or full panel"):
        validate_config(bad_seed)

    bad_device = copy.deepcopy(full)
    bad_device["run"]["device"] = "mps"
    with pytest.raises(ConfigError, match="CPU-only"):
        validate_config(bad_device)

    bad_interface = copy.deepcopy(full)
    bad_interface["h17"]["feature_names"].remove("Q_present")
    with pytest.raises(ConfigError, match="eight-coordinate interface"):
        validate_config(bad_interface)

    shuffled = copy.deepcopy(full)
    shuffled["train"]["shuffle"] = True
    with pytest.raises(ConfigError, match="deterministic unshuffled"):
        validate_config(shuffled)

    changed_data_seed = copy.deepcopy(full)
    changed_data_seed["h17"]["data_seed"] += 1
    with pytest.raises(ConfigError, match="run-seed-independent constants"):
        validate_config(changed_data_seed)

    changed_phase_seed = copy.deepcopy(full)
    changed_phase_seed["h17"]["phase_seeds"]["washout"] += 1
    with pytest.raises(ConfigError, match="run-seed-independent constants"):
        validate_config(changed_phase_seed)

    unclipped = copy.deepcopy(full)
    unclipped["train"]["grad_clip"] = 2.0
    with pytest.raises(ConfigError, match="gradient clipping"):
        validate_config(unclipped)

    smoothed = copy.deepcopy(full)
    smoothed["train"]["label_smoothing"] = 0.1
    with pytest.raises(ConfigError, match="zero label smoothing"):
        validate_config(smoothed)

    auxiliary = copy.deepcopy(full)
    auxiliary["model"]["nuisance_bits"] = 1
    with pytest.raises(ConfigError, match="no auxiliary heads"):
        validate_config(auxiliary)

    missing_checkpoint = copy.deepcopy(full)
    missing_checkpoint["h17"]["washout_checkpoints"].remove(128)
    with pytest.raises(ConfigError, match="primary_auc_window"):
        validate_config(missing_checkpoint)

    incomplete_grid = copy.deepcopy(full)
    incomplete_grid["cases"].pop()
    with pytest.raises(ConfigError, match="exact ordered pilot-goal or six-schedule"):
        validate_config(incomplete_grid)

    pilot = load_config(PILOT_CONFIG)
    leaked_schedule = copy.deepcopy(pilot)
    leaked_schedule["h17"]["schedule"] = "p_q_y"
    with pytest.raises(ConfigError, match="must not define or construct"):
        validate_config(leaked_schedule)


def test_h17_planned_runs_reject_seed_panel_overrides() -> None:
    pilot = load_config(PILOT_CONFIG)
    full = load_config(FULL_CONFIG)
    with pytest.raises(ConfigError, match="frozen h17 pilot seed panel"):
        planned_runs(pilot, seed_override=[0])
    with pytest.raises(ConfigError, match="frozen h17 full seed panel"):
        planned_runs(full, seed_override=PILOT_SEEDS)


def test_h17_smoke_preserves_strata_degree_and_interface() -> None:
    full = load_config(FULL_CONFIG)
    smoke = smoke_config(full)
    assert len(planned_runs(full, smoke=True)) == 1
    assert smoke["h17"]["schedule"] == "p_q_y"
    assert smoke["data"]["n_train"] == 288
    assert smoke["data"]["k"] == smoke["data"]["max_k"] == 3
    assert smoke["data"]["state_dim"] == 0
    assert smoke["train"]["batch_size"] == 288
    assert smoke["train"]["steps"] == 8
    assert smoke["train"]["shuffle"] is False
    assert smoke["h17"]["rows_per_weight_unit"] == 3
    assert smoke["h17"]["weighted_strata"] == 96
    assert smoke["h17"]["batches_per_presentation"] == 1
    assert smoke["h17"]["presentations"] == 2
    assert smoke["h17"]["component_steps"] == 2
    assert smoke["h17"]["washout_steps"] == 2
    assert smoke["h17"]["component_checkpoints"] == [0, 1, 2]
    assert smoke["h17"]["washout_checkpoints"] == [0, 1, 2]
    assert smoke["h17"]["feature_names"] == FEATURE_NAMES
    validate_config(smoke)


def test_h17_launcher_selects_separate_frozen_configs_and_forwards_arguments() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    assert "PILOT" in launcher
    assert "configs/e20_counterbalanced_order_pilot.yaml" in launcher
    assert "configs/e20_counterbalanced_order.yaml" in launcher
    assert launcher.count('"$@"') == 2
    assert 'export DEVICE="cpu"' in launcher
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        assert f"export {variable}=1" in launcher
    assert "artifacts-e20-pilot" in launcher
    assert "artifacts-e20" in launcher
    assert "forkworld.e20_launch_guard" in launcher
    assert "--expected-gate-analyzer-sha256" in launcher
    assert "faaafa98f08eed830cfed4ec1c126d8e4f9c3079f91f8ab0cd27a99eaded3feb" in launcher
    assert launcher.index("forkworld.e20_launch_guard") < launcher.index(
        "run_config configs/e20_counterbalanced_order.yaml"
    )


def test_h17_dispatches_to_counterbalanced_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    import forkworld.protocols_counterbalanced as protocol

    sentinel = object()
    dispatch_config = {"experiment": {"hypothesis": "h17"}}

    def fake_run_h17(config: object, seed: int) -> object:
        assert config is dispatch_config
        assert seed == 571
        return sentinel

    monkeypatch.setattr(protocol, "run_h17", fake_run_h17)
    assert run_protocol(dispatch_config, seed=571) is sentinel
