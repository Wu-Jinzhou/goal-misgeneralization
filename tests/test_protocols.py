"""Fast integration coverage for every preregistered hypothesis protocol."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from forkworld.config import load_config
from forkworld.data import SemanticBatch
from forkworld.models import GoalMLP
from forkworld.noise import NoiseConfig, noise_array
from forkworld.protocols import run_protocol
from forkworld.protocols_algorithms import (
    _h6_training_plan,
    _make_recurring_batch,
    _NoisyTransform,
    _rl_reward,
    _SemanticSampler,
)
from forkworld.protocols_persistence import (
    _h8_batch,
    _h8_perturbation,
    _h8_stage1_eligible,
)
from forkworld.runner import smoke_config
from forkworld.training import SFTConfig, train_clean_sft

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [
    ("h1", "h01_simplicity.yaml"),
    ("h2", "h02_complexity.yaml"),
    ("h3", "h03_dynamics.yaml"),
    ("h4", "h04_conflict_diversity.yaml"),
    ("h5", "h05_update_capacity.yaml"),
    ("h6", "h06_noise.yaml"),
    ("h7", "h07_unlearning.yaml"),
    ("h8", "h08_hysteresis.yaml"),
    ("h9", "h09_multiplicity.yaml"),
]


@pytest.mark.parametrize(("hypothesis", "filename"), CONFIGS)
def test_hypothesis_protocol_smoke(hypothesis: str, filename: str) -> None:
    config = smoke_config(load_config(REPO_ROOT / "configs" / filename))
    config["run"]["device"] = "cpu"
    result = run_protocol(config, seed=0)

    assert result.summary["hypothesis"] == hypothesis
    assert result.summary["seed"] == 0
    assert result.summary["model"]
    assert result.summary["final"]
    assert result.metrics
    assert result.predictions
    assert result.checkpoints
    assert result.evaluation_batch is not None
    if hypothesis == "h2":
        data = result.summary["data"]
        models = result.summary["model"]
        assert data["calibration_interface"] == "competition_full_width"
        assert data["calibration_state_is_neutralized"] is True
        assert data["calibration_input_dim_matches_competition"] is True
        assert data["calibration_parameter_count_matches_competition"] is True
        assert data["calibration_trainable_count_matches_competition"] is True
        assert {
            models["proxy_calibration"]["input_dim"],
            models["exact_calibration"]["input_dim"],
            models["competition"]["input_dim"],
        } == {models["competition"]["input_dim"]}
    if hypothesis == "h3":
        events = result.summary["events"]
        assert "intended_dominance_crossing_time" in events
        assert "sequential_proxy_to_intended_replacement_time" in events
        assert events["replacement_definition"].startswith("sustained intended")
    if hypothesis == "h5":
        assert result.summary["iid"]["n"] == config["data"]["n_validation"]


def test_h8_behavior_matching_stops_at_a_variable_alignment_volume() -> None:
    config = smoke_config(load_config(REPO_ROOT / "configs" / "h08_hysteresis.yaml"))
    config["run"]["device"] = "cpu"
    config["h8"].update(
        {
            "stage1_mode": "behavior_matched",
            "stage1_match_band": [0.0, 1.0],
            "stage1_match_patience": 1,
        }
    )
    result = run_protocol(config, seed=0)

    assert result.summary["data"]["stage1_mode"] == "behavior_matched"
    assert result.summary["training"]["stage1_examples"] < result.summary["data"]["requested_n1"]
    assert result.summary["data"]["realized_n1"] == result.summary["training"]["stage1_examples"]


def test_h4_fraction_floor_is_target_stratified_and_logged() -> None:
    config = smoke_config(load_config(REPO_ROOT / "configs" / "h04_conflict_diversity.yaml"))
    config["run"]["device"] = "cpu"
    config["h4"].update(
        {"condition": "concentrated", "n_conflict": 8, "unique_fraction": 0.02}
    )
    result = run_protocol(config, seed=0)
    data = result.summary["data"]

    assert data["u_conflict_source"] == "fraction"
    assert data["requested_unique_fraction"] == 0.02
    assert data["fraction_implied_u_conflict"] == 0
    assert data["minimum_target_stratified_u_conflict"] == 2
    assert data["realized_u_conflict"] == 2
    assert result.summary["final"]["train_eval_ids_disjoint"] is True


def test_h6_recurring_rows_form_fixed_horizon_terminal_reward_episodes() -> None:
    batch = _make_recurring_batch(
        n=80,
        q=0.9,
        k=2,
        max_k=3,
        state_dim=2,
        visits=2,
        nuisance_bits=0,
        nuisance_entropy=0.0,
        seed=17,
        reward_mode="terminal",
        fixed_horizon=4,
    )
    for episode in set(batch.episode_id.tolist()):
        rows = batch.episode_id == episode
        assert len(set(batch.state_id[rows].tolist())) == 1
        assert len(set(batch.y[rows].tolist())) == 1
        assert batch.step_id[rows].tolist() == [0, 1, 2, 3]
        assert batch.reward[rows][:-1].tolist() == [0.0, 0.0, 0.0]
        assert batch.reward[rows][-1] == batch.y[rows][-1]

    episode_noise = noise_array(batch, NoiseConfig("episode", scale=0.3), draw=2)
    state_noise = noise_array(batch, NoiseConfig("state", scale=0.3), draw=2)
    step_noise = noise_array(batch, NoiseConfig("step", scale=0.3), draw=2)
    assert all(
        np.ptp(episode_noise[batch.episode_id == episode]) == 0
        for episode in np.unique(batch.episode_id)
    )
    assert all(
        np.ptp(state_noise[batch.state_id == state]) == 0
        for state in np.unique(batch.state_id)
    )
    assert any(
        np.ptp(step_noise[batch.episode_id == episode]) > 0
        for episode in np.unique(batch.episode_id)
    )


def test_h6_fixed_epoch_presentations_scale_with_dataset_size() -> None:
    config = {"h6": {"training_epochs": 8}, "train": {"batch_size": 256}}
    small = _h6_training_plan(config, 2_560)
    large = _h6_training_plan(config, 10_240)

    assert small["requested_presentations"] == 20_480
    assert large["requested_presentations"] == 81_920
    assert small["optimizer_steps"] == 80
    assert large["optimizer_steps"] == 320


def test_h6_recurrence_levels_are_exact_and_distinct() -> None:
    realized_states = []
    for visits in (1, 4, 16, 64):
        batch = _make_recurring_batch(
            n=2_560,
            q=0.9,
            k=2,
            max_k=3,
            state_dim=2,
            visits=visits,
            nuisance_bits=0,
            nuisance_entropy=0.0,
            seed=29,
            reward_mode="dense_fixed_horizon",
            fixed_horizon=4,
        )
        assert batch.metadata["requested_visits_per_state"] == visits
        assert batch.metadata["realized_visits_per_state"] == visits
        assert batch.metadata["requested_episode_visits_per_state"] == visits
        assert batch.metadata["realized_episode_visits_per_state"] == visits
        assert batch.metadata["minimum_episode_visits_per_state"] == visits
        assert batch.metadata["maximum_episode_visits_per_state"] == visits
        assert batch.metadata["recurrence_exact"] is True
        realized_states.append(batch.metadata["n_unique_states"])

    assert realized_states == [640, 160, 40, 10]
    assert len(set(realized_states)) == 4


def test_h6_label_noise_target_reaches_dynamic_sft_source() -> None:
    y = np.tile(np.asarray([-1, 1], dtype=np.int8), 32)
    clean = SemanticBatch(y=y, channels={"P": y})
    transform = _NoisyTransform(
        NoiseConfig(
            regime="biased",
            location="label",
            scale=2.0,
            bias=-2.0,
            seed=19,
        ),
        label_threshold=True,
    )
    source = _SemanticSampler(clean, transform)
    sampled = source.sample_batch(64, torch.Generator().manual_seed(7), 1)
    assert np.array_equal(np.asarray(sampled.target), -np.asarray(sampled.y))

    model = GoalMLP(2, width=4, depth=0)
    train_clean_sft(
        model,
        source,
        SFTConfig(
            steps=40,
            batch_size=64,
            learning_rate=0.1,
            seed=7,
            save_checkpoints=False,
        ),
    )
    with torch.no_grad():
        prediction = np.where(model(torch.as_tensor(clean.features())).numpy() >= 0, 1, -1)
    assert float(np.mean(prediction == -y)) > 0.95
    assert float(np.mean(prediction == y)) < 0.05


def test_h6_nuisance_labels_are_hidden_and_freshly_resampled() -> None:
    batch = _make_recurring_batch(
        n=80,
        q=0.9,
        k=2,
        max_k=3,
        state_dim=2,
        visits=2,
        nuisance_bits=2,
        nuisance_entropy=1.0,
        seed=23,
        reward_mode="dense_fixed_horizon",
        fixed_horizon=4,
    )
    assert batch.metadata["algorithm_abstraction"] == (
        "fixed_horizon_repeated_contextual_decisions"
    )
    assert batch.metadata["full_episode_policy"] is False
    assert batch.metadata["nuisance_labels_visible_in_observation"] is False
    assert np.asarray(batch.nuisance_targets).shape == (80, 2)
    assert all(not name.startswith("N_") for name in batch.feature_names(max_k=3))

    transform = _NoisyTransform(
        NoiseConfig("step", location="observation", scale=0.0, channels=("P",)),
        label_threshold=True,
        nuisance_positive_probabilities=np.asarray([0.5, 0.5]),
        nuisance_seed=41,
    )
    first = transform(batch, 1)
    second = transform(batch, 2)
    assert np.array_equal(first.state_id, second.state_id)
    assert np.array_equal(first.episode_id, second.episode_id)
    assert not np.array_equal(first.nuisance_targets, second.nuisance_targets)
    repeated_state = np.asarray(batch.state_id) == np.asarray(batch.state_id)[0]
    repeated_draws = np.concatenate(
        (
            np.asarray(first.nuisance_targets)[repeated_state],
            np.asarray(second.nuisance_targets)[repeated_state],
        ),
        axis=0,
    )
    assert np.unique(repeated_draws, axis=0).shape[0] > 1
    assert transform.nuisance_diagnostics()["draws"] == 2


def test_h6_rl_reward_consumes_dense_and_terminal_per_row_signals() -> None:
    dense = _make_recurring_batch(
        n=80,
        q=0.9,
        k=2,
        max_k=3,
        state_dim=2,
        visits=2,
        nuisance_bits=0,
        nuisance_entropy=0.0,
        seed=43,
        reward_mode="dense_fixed_horizon",
        fixed_horizon=4,
    )
    terminal = _make_recurring_batch(
        n=80,
        q=0.9,
        k=2,
        max_k=3,
        state_dim=2,
        visits=2,
        nuisance_bits=0,
        nuisance_entropy=0.0,
        seed=43,
        reward_mode="terminal",
        fixed_horizon=4,
    )
    intended_actions = torch.as_tensor((np.asarray(dense.y) > 0).astype(np.int64))

    assert torch.equal(_rl_reward(dense, intended_actions), torch.ones(len(dense)))
    terminal_rewards = _rl_reward(terminal, intended_actions).numpy()
    terminal_rows = np.asarray(terminal.step_id) == 3
    assert np.all(terminal_rewards[~terminal_rows] == 0.0)
    assert np.all(terminal_rewards[terminal_rows] == 1.0)


def test_h8_match_band_only_gates_behavior_matched_runs() -> None:
    assert _h8_stage1_eligible("fixed", gate_passed=True, in_match_band=False)
    assert not _h8_stage1_eligible(
        "behavior_matched", gate_passed=True, in_match_band=False
    )
    assert _h8_stage1_eligible(
        "behavior_matched", gate_passed=True, in_match_band=True
    )
    assert not _h8_stage1_eligible("fixed", gate_passed=False, in_match_band=True)


def test_h8_intended_removal_contains_no_anti_old_goal_evidence() -> None:
    config = {"data": {"k": 2, "max_k": 3, "state_dim": 0}}
    base = _h8_batch(config, 64, 31, split="test", id_offset=0)
    assert np.array_equal(base.channels["P"], -np.asarray(base.y))

    removed, details = _h8_perturbation(
        base, config, 37, "intended_removal"
    )
    assert np.array_equal(removed.channels["P"], np.asarray(removed.y))
    assert np.array_equal(np.asarray(removed.target), removed.channels["P"])
    assert all(np.all(removed.channels[f"R_{index}"] == 0) for index in (1, 2))
    assert details["intended_channels_present"] is False
    assert details["old_goal_counterevidence_rate"] == 0.0
    assert details["goal_identifying_disagreement_examples"] == 0
