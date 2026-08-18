"""Scientific invariants for the four H5 learning-algorithm arms."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch

from forkworld.config import load_config
from forkworld.data import SemanticBatch
from forkworld.envs import Goal, NeutralChoiceSimulator
from forkworld.models import GoalMLP, GoalModelOutput
from forkworld.protocols import run_protocol
from forkworld.protocols_algorithms import (
    _NeutralEpisodeSampler,
    _NeutralTrajectoryReward,
    _PolicyOccupancyCollector,
    _SemanticSampler,
    _make_h5_episode_dataset,
    _terminal_success_reward,
)
from forkworld.runner import smoke_config
from forkworld.training import BanditConfig, _auxiliary_loss, train_contextual_bandit


REPO_ROOT = Path(__file__).resolve().parents[1]


def _episode_batch() -> SemanticBatch:
    return _make_h5_episode_dataset(
        64,
        0.75,
        2,
        17,
        context_bits=1,
        nuisance_bits=3,
        nuisance_entropy=1.0,
        max_k=5,
        state_dim=4,
    )


def test_trajectory_demonstrations_are_successful_and_hide_neutral_labels() -> None:
    batch = _episode_batch()
    assert batch.metadata["trajectory_semantics"] == "NeutralChoiceSimulator"
    assert batch.metadata["trajectory_horizon"] == 7
    assert batch.metadata["learned_action_factors_per_episode"] == 4
    assert batch.metadata["neutral_actions_visible_in_observation"] is False
    assert all(not name.startswith("N_") for name in batch.feature_names(max_k=5))

    choices = np.asarray(batch.nuisance_targets)
    assert choices.shape == (64, 3)
    assert set(np.unique(choices)) == {0, 1}
    for row in range(8):
        goal = Goal.LEFT if batch.y[row] < 0 else Goal.RIGHT
        trajectory = NeutralChoiceSimulator(3, goal, seed=row).rollout(
            neutral_choices=choices[row].tolist()
        )
        assert trajectory.success
        assert trajectory.total_reward == 1.0
        assert trajectory.horizon == 7
        assert tuple(trajectory.neutral_choices) == tuple(choices[row])


def test_terminal_rl_reward_ignores_neutral_demonstration_actions() -> None:
    batch = _episode_batch()
    actions = torch.as_tensor((np.asarray(batch.y) > 0).astype(np.int64))
    rewards = _terminal_success_reward(batch, actions)
    flipped_nuisance = 1 - np.asarray(batch.nuisance_targets)
    changed = batch.with_updates(nuisance_targets=flipped_nuisance)
    changed_rewards = _terminal_success_reward(changed, actions)

    assert torch.equal(rewards, torch.ones_like(rewards))
    assert torch.equal(changed_rewards, rewards)
    assert _terminal_success_reward(batch, 1 - actions).sum().item() == 0


def test_factorized_rl_executes_policy_sampled_reward_invariant_branches() -> None:
    batch = _episode_batch()
    positive_index = int(np.flatnonzero(np.asarray(batch.y) > 0)[0])
    context = batch.select([positive_index]).with_updates(
        nuisance_targets=np.zeros((1, 3), dtype=np.int64)
    )
    features = torch.as_tensor(context.features(max_k=5), dtype=torch.float32)
    model = GoalMLP(features.shape[1], width=8, depth=1, nuisance_heads=3)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.goal_head.bias.fill_(100.0)  # The sole +1 goal always succeeds.

    reward = _NeutralTrajectoryReward(3, seed=47)
    result = train_contextual_bandit(
        model,
        (features, torch.ones(1), torch.zeros((1, 3), dtype=torch.long)),
        BanditConfig(
            steps=1,
            batch_size=1,
            learning_rate=0.01,
            algorithm="reinforce",
            center_reinforce_rewards=False,
            normalize_advantages=False,
            seed=23,
            save_checkpoints=False,
        ),
        reward_fn=lambda actions, branch_actions, **_: reward(
            context, actions, branch_actions
        ),
        factorized_actions=True,
    )

    assert reward.last_branch_actions is not None
    assert reward.last_branch_actions.shape == (1, 3)
    assert not np.array_equal(reward.last_branch_actions, np.zeros((1, 3), dtype=np.int64))
    assert reward.diagnostics()["branch_action_source"] == "actor_nuisance_heads"
    assert reward.diagnostics()["branch_supervision"] is False
    assert result.history[-1].metrics["learned_action_factors"] == 4
    # With reward one and p(branch)=0.5, each sampled branch has a nonzero
    # score-function gradient even though the batch's nuisance targets are zero.
    for head in model.nuisance_heads.values():
        assert head.bias.grad is not None
        assert torch.count_nonzero(head.bias.grad).item() == 1

    selected_goal = torch.ones(1, dtype=torch.long)
    zeros = torch.zeros((1, 3), dtype=torch.long)
    ones = torch.ones((1, 3), dtype=torch.long)
    left_reward = _NeutralTrajectoryReward(3, seed=47)(context, selected_goal, zeros)
    right_reward = _NeutralTrajectoryReward(3, seed=47)(context, selected_goal, ones)
    assert torch.equal(left_reward, right_reward)


def test_active_trajectory_heads_have_unit_loss_and_forced_heads_have_no_gradient() -> None:
    branch_logits = {
        f"nuisance_{index}": torch.zeros(1, requires_grad=True) for index in range(4)
    }
    output = GoalModelOutput(
        goal_logits=torch.zeros(1, requires_grad=True),
        nuisance_logits=branch_logits,
    )
    targets = torch.ones((1, 4), dtype=torch.long)
    active_weights = {
        "nuisance_0": 1.0,
        "nuisance_1": 1.0,
        "nuisance_2": 0.0,
        "nuisance_3": 0.0,
    }

    # _auxiliary_loss averages active heads; multiplying by H=2 recovers the
    # intended sum of one NLL per stochastic branch action.
    loss = 2.0 * _auxiliary_loss(output, targets, active_weights)
    expected = sum(
        torch.nn.functional.binary_cross_entropy_with_logits(
            branch_logits[f"nuisance_{index}"], torch.ones(1)
        )
        for index in range(2)
    )
    assert torch.allclose(loss, expected)
    loss.backward()

    for index in range(2):
        gradient = branch_logits[f"nuisance_{index}"].grad
        assert gradient is not None and torch.count_nonzero(gradient).item() == 1
    for index in range(2, 4):
        assert branch_logits[f"nuisance_{index}"].grad is None


def test_neutral_actions_are_resampled_for_repeated_episode_contexts() -> None:
    batch = _episode_batch()
    sampler = _NeutralEpisodeSampler(
        _SemanticSampler(batch),
        positive_probabilities=np.full(3, 0.5),
        seed=31,
    )
    generator = torch.Generator().manual_seed(12)
    first = sampler.sample_batch(64, generator, 1)
    second = sampler.sample_batch(64, generator, 2)
    first_order = np.argsort(first.sample_id)
    second_order = np.argsort(second.sample_id)

    assert np.array_equal(first.sample_id[first_order], second.sample_id[second_order])
    assert not np.array_equal(
        first.nuisance_targets[first_order], second.nuisance_targets[second_order]
    )
    assert np.all(np.sum(first.nuisance_targets, axis=0) == 32)
    assert sampler.diagnostics()["draws"] == 2
    assert sampler.diagnostics()["rollouts"] == 128


def test_on_policy_collector_visits_complementary_successors_and_queries_canonical_labels() -> None:
    original = _episode_batch()
    # Even a contaminated root target must not change the clean H5 oracle.
    batch = original.with_updates(target=-np.asarray(original.y))
    config = {
        "data": {"max_k": 5},
    }

    def constant_policy(logit: float) -> GoalMLP:
        model = GoalMLP(batch.features(max_k=5).shape[1], width=8, depth=1, nuisance_heads=3)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.goal_head.bias.fill_(logit)
        return model

    right = _PolicyOccupancyCollector(_SemanticSampler(batch), config)
    left = _PolicyOccupancyCollector(_SemanticSampler(batch), config)
    right_batch = right.sample_batch(
        constant_policy(100.0), 16, torch.Generator().manual_seed(9), 1
    )
    left_batch = left.sample_batch(
        constant_policy(-100.0), 16, torch.Generator().manual_seed(9), 1
    )

    right_base_ids = np.asarray(right_batch.latents["occupancy_base_state_id"])
    left_base_ids = np.asarray(left_batch.latents["occupancy_base_state_id"])
    assert np.array_equal(right_base_ids, left_base_ids)
    assert np.array_equal(right_batch.state_id, left_batch.state_id + 3)
    source_state_ids = set(np.asarray(batch.state_id).tolist())
    assert source_state_ids.isdisjoint(np.asarray(right_batch.state_id).tolist())
    assert source_state_ids.isdisjoint(np.asarray(left_batch.state_id).tolist())

    # Opposite policies see the same exogenous semantic contexts.  Only the
    # reserved model-visible fork-successor coordinate differs.
    assert np.array_equal(right_batch.y, left_batch.y)
    assert np.array_equal(right_batch.target, right_batch.y)
    assert np.array_equal(left_batch.target, left_batch.y)
    assert right_batch.nuisance_targets is None
    assert left_batch.nuisance_targets is None
    for name in right_batch.channels:
        assert np.array_equal(right_batch.channels[name], left_batch.channels[name])
    right_features = right_batch.features(max_k=5)
    left_features = left_batch.features(max_k=5)
    visitation_name = str(batch.metadata["policy_visitation_channel"])
    visitation_column = right_batch.feature_names(max_k=5).index(visitation_name)
    # At depth two, all-left has heap path code 3 and all-right has code 6.
    assert np.all(right_features[:, visitation_column] == 6.0)
    assert np.all(left_features[:, visitation_column] == 3.0)
    assert np.array_equal(
        np.delete(right_features, visitation_column, axis=1),
        np.delete(left_features, visitation_column, axis=1),
    )

    right_diagnostics = right.diagnostics()
    left_diagnostics = left.diagnostics()
    assert right_diagnostics["root_rollouts"] == 16
    assert right_diagnostics["transition_rollouts"] == 32
    assert right_diagnostics["queried_visited_states"] == 16
    assert right_diagnostics["policy_action_rate"] == {"left": 0.0, "right": 1.0}
    assert right_diagnostics["policy_action_right_rate_by_depth"] == [1.0, 1.0]
    assert right_diagnostics["path_occupancy_rate"] == {"11": 1.0}
    assert left_diagnostics["final_successor_occupancy_rate"] == {
        "left": 1.0,
        "right": 0.0,
    }
    assert left_diagnostics["path_occupancy_rate"] == {"00": 1.0}
    assert right_diagnostics["query_priority"] == "none"


def test_on_policy_tree_ids_are_unique_across_roots_depths_and_paths() -> None:
    batch = _episode_batch()
    collector = _PolicyOccupancyCollector(
        _SemanticSampler(batch), {"data": {"max_k": 5}}, rollout_depth=3
    )
    root_ids = np.asarray(batch.state_id)[:4]
    generated = []
    for depth in range(1, 4):
        paths = np.arange(1 << depth, dtype=np.int64)
        repeated_roots = np.repeat(root_ids, len(paths))
        repeated_paths = np.tile(paths, len(root_ids))
        generated.append(
            collector._tree_ids(
                repeated_roots,
                depth,
                repeated_paths,
                ordinals=collector._state_id_ordinals,
                namespace=collector._state_id_namespace,
                name="state_id",
            )
        )
    all_nodes = np.concatenate(generated)
    assert len(np.unique(all_nodes)) == len(all_nodes)
    assert set(np.asarray(batch.state_id).tolist()).isdisjoint(all_nodes.tolist())


def test_h5_all_algorithms_have_distinct_exact_action_costs() -> None:
    base = smoke_config(load_config(REPO_ROOT / "configs" / "h05_update_capacity.yaml"))
    base["run"].update(device="cpu", task_levels=["choice"], save_checkpoints=False)
    # H5 adds the same one-coordinate visitation channel to every arm even when
    # the underlying dataset has no generic state features.
    base["data"]["state_dim"] = 0
    base["train"].update(steps=2, batch_size=16, eval_steps=[1, 2])
    base["h5"].update(
        nuisance_bits=2,
        nuisance_entropy=2,
    )

    expected = {
        "clean_sft": (32, 32, 0),
        "trajectory_sft": (160, 160, 0),
        "on_policy_imitation": (64, 32, 0),
        "rl": (160, 0, 32),
    }
    actor_sizes = set()
    for algorithm, counts in expected.items():
        config = copy.deepcopy(base)
        config["h5"]["algorithm"] = algorithm
        result = run_protocol(config, seed=0)
        summary = result.summary
        costs = summary["costs"]
        assert summary["demonstration"]["trajectory_horizon"] == 5
        assert (
            costs["environment_interactions"],
            costs["labeled_actions"],
            costs["terminal_outcomes"],
        ) == counts
        assert summary["nuisance"]["applied_auxiliary_weight"] == (
            2.0 if algorithm == "trajectory_sft" else 0.0
        )
        assert bool(summary["on_policy_collection"]) == (
            algorithm == "on_policy_imitation"
        )
        assert summary["occupancy_control"]["model_state_dim"] == 1
        assert summary["occupancy_control"]["fixed_width_across_h5_arms"] is True
        if algorithm == "on_policy_imitation":
            collection = summary["on_policy_collection"]
            assert collection["root_rollouts"] == 32
            assert collection["transition_rollouts"] == 64
            assert collection["queried_visited_states"] == 32
        sampling = summary["demonstration"]["neutral_choice_sampling"]
        assert sampling["draws"] == (
            2 if algorithm in {"trajectory_sft", "rl"} else 0
        )
        actor_sizes.add(costs["trainable_scalars"])
    assert len(actor_sizes) == 1


def test_h5_entropy_counts_policy_sampled_branch_factors() -> None:
    base = smoke_config(load_config(REPO_ROOT / "configs" / "h05_update_capacity.yaml"))
    base["run"].update(device="cpu", task_levels=["choice"], save_checkpoints=False)
    base["train"].update(steps=1, batch_size=8, eval_steps=[1])
    base["h5"].update(algorithm="rl", nuisance_bits=2)

    summaries = {}
    for entropy in (0, 2):
        config = copy.deepcopy(base)
        config["h5"]["nuisance_entropy"] = entropy
        summaries[entropy] = run_protocol(config, seed=0).summary

    zero_sampling = summaries[0]["demonstration"]["neutral_choice_sampling"]
    rich_sampling = summaries[2]["demonstration"]["neutral_choice_sampling"]
    assert zero_sampling["stochastic_branch_actions_per_episode"] == 0
    assert zero_sampling["forced_branch_actions_per_episode"] == 2
    assert zero_sampling["branch_right_rate"] == []
    assert zero_sampling["realized_total_branch_entropy_bits"] == 0.0
    assert summaries[0]["costs"]["learned_action_factors_per_presentation"] == 1
    assert summaries[0]["nuisance"]["realized_entropy_per_bit"] == [0.0, 0.0]

    assert rich_sampling["stochastic_branch_actions_per_episode"] == 2
    assert rich_sampling["forced_branch_actions_per_episode"] == 0
    assert len(rich_sampling["branch_right_rate"]) == 2
    assert 0.0 <= rich_sampling["realized_total_branch_entropy_bits"] <= 2.0
    assert summaries[2]["costs"]["learned_action_factors_per_presentation"] == 3
    assert summaries[2]["nuisance"]["realized_entropy_per_bit"] == [1.0, 1.0]
    assert (
        summaries[0]["costs"]["trainable_scalars"]
        == summaries[2]["costs"]["trainable_scalars"]
    )

    trajectory_config = copy.deepcopy(base)
    trajectory_config["h5"].update(algorithm="trajectory_sft", nuisance_entropy=1)
    trajectory = run_protocol(trajectory_config, seed=0)
    assert isinstance(trajectory.model, GoalMLP)
    active_head = trajectory.model.nuisance_heads["nuisance_0"]
    forced_head = trajectory.model.nuisance_heads["nuisance_1"]
    assert any(parameter.grad is not None for parameter in active_head.parameters())
    assert all(parameter.grad is None for parameter in forced_head.parameters())
    assert trajectory.summary["nuisance"]["applied_auxiliary_weight"] == 1.0
