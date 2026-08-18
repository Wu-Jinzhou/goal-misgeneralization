from __future__ import annotations

import pytest
import torch

from forkworld.envs import (
    ForkGridWorld,
    Goal,
    GridAction,
    NeutralChoiceSimulator,
    OneStepChoiceEnv,
    RandomObstacleNavigationEnv,
    coerce_goal,
)
from forkworld.navigation import (
    NavigatorMLP,
    NavigatorTrainingConfig,
    evaluate_navigation,
    load_navigator_checkpoint,
    train_bfs_navigator,
)
from forkworld.navigation_support import _navigator_metadata_compatible, _navigator_request


def test_goal_ids_and_one_step_choice_are_explicit() -> None:
    with pytest.raises(ValueError):
        coerce_goal(0)
    with pytest.raises(ValueError):
        coerce_goal(True)

    env = OneStepChoiceEnv((1.0, -1.0, 1.0), Goal.LEFT, seed=4)
    semantic_id = env.semantic_id
    result = env.step(Goal.RIGHT)
    assert result.terminated and not result.truncated
    assert result.reward == 0.0
    assert result.info["selected_goal"] == int(Goal.RIGHT)
    assert env.reset().semantic_id == semantic_id


def test_fork_has_one_symmetric_goal_decision() -> None:
    env = ForkGridWorld(Goal.LEFT, seed=0)
    left_path = env.shortest_path(Goal.LEFT)
    right_path = env.shortest_path(Goal.RIGHT)
    assert left_path is not None and right_path is not None
    assert len(left_path) == len(right_path) == env.route_length == 7
    assert left_path[:3] == right_path[:3] == (GridAction.UP,) * 3
    assert env.decision_action(Goal.LEFT) is GridAction.LEFT
    assert env.decision_action(Goal.RIGHT) is GridAction.RIGHT

    for goal in Goal:
        rollout_env = env.clone(goal_id=goal)
        while not rollout_env.done:
            action = rollout_env.expert_action(goal)
            assert action is not None
            rollout_env.step(action)
        assert rollout_env.reached_goal is goal
        assert rollout_env.steps == 7


def test_random_navigation_is_reproducible_and_both_goals_are_safe() -> None:
    first = RandomObstacleNavigationEnv(Goal.LEFT, seed=17)
    repeated = RandomObstacleNavigationEnv(Goal.RIGHT, seed=17)
    different = RandomObstacleNavigationEnv(Goal.LEFT, seed=18)
    assert first.semantic_id == repeated.semantic_id
    assert first.start == repeated.start
    assert first.targets == repeated.targets
    assert first.obstacles == repeated.obstacles
    assert first.semantic_id != different.semantic_id

    for goal in Goal:
        other = Goal.RIGHT if goal is Goal.LEFT else Goal.LEFT
        path = first.shortest_path(goal)
        assert path
        position = first.start
        visited = {position}
        for action in path:
            delta = {
                GridAction.UP: (-1, 0),
                GridAction.RIGHT: (0, 1),
                GridAction.DOWN: (1, 0),
                GridAction.LEFT: (0, -1),
            }[action]
            position = position[0] + delta[0], position[1] + delta[1]
            visited.add(position)
        assert position == first.targets[goal]
        assert first.targets[other] not in visited


def test_neutral_gadgets_preserve_reward_timing_and_horizon() -> None:
    simulator = NeutralChoiceSimulator(4, Goal.RIGHT, seed=9)
    trajectories = simulator.equivalent_successful_trajectories()
    assert len(trajectories) == 16
    assert {trajectory.horizon for trajectory in trajectories} == {9}
    assert {len(trajectory.steps) for trajectory in trajectories} == {9}
    assert {trajectory.total_reward for trajectory in trajectories} == {1.0}
    assert {
        tuple(step.reward for step in trajectory.steps) for trajectory in trajectories
    } == {(0.0,) * 8 + (1.0,)}


def test_bfs_navigator_training_checkpoint_and_capability_metrics(tmp_path) -> None:
    env = ForkGridWorld(Goal.LEFT, seed=2)
    result = train_bfs_navigator(
        [env],
        hidden_sizes=(64, 64),
        config=NavigatorTrainingConfig(
            epochs=150,
            batch_size=64,
            learning_rate=1e-2,
            seed=3,
            target_patience=2,
        ),
    )
    assert result.final_accuracy == 1.0

    metrics = evaluate_navigation(
        result.model,
        [env],
        [Goal.RIGHT],
        intended_goals=[Goal.LEFT],
    ).summary()
    assert metrics["selection_accuracy"] == 0.0
    assert metrics["selected_goal_success_rate"] == 1.0
    assert metrics["intended_goal_success_rate"] == 0.0
    assert metrics["oracle_goal_success_rate"] == 1.0
    assert metrics["clamped_goal_success_rate"] == 1.0

    checkpoint = tmp_path / "navigator.pt"
    result.model.save_checkpoint(checkpoint, metadata={"semantic_id": env.semantic_id})
    restored, metadata = load_navigator_checkpoint(checkpoint)
    assert metadata == {"semantic_id": env.semantic_id}
    assert all(not parameter.requires_grad for parameter in restored.parameters())
    for left, right in zip(result.model.parameters(), restored.parameters()):
        assert torch.equal(left.cpu(), right.cpu())


def test_navigator_cache_key_covers_the_pretraining_request() -> None:
    request = _navigator_request(
        "navigation",
        seed=7,
        maps=4,
        epochs=100,
        target_accuracy=0.95,
        target_patience=2,
    )
    metadata = {
        **request,
        "final_accuracy": 0.96,
        "target_patience_achieved": 2,
        "frozen": True,
    }
    assert _navigator_metadata_compatible(metadata, request)
    assert not _navigator_metadata_compatible(
        {key: value for key, value in metadata.items() if key != "implementation_fingerprint"},
        request,
    )
    assert not _navigator_metadata_compatible(
        {**metadata, "navigator_cache_schema_version": 0}, request
    )
    assert not _navigator_metadata_compatible(
        {**metadata, "source_fingerprint_schema_version": 0}, request
    )
    assert not _navigator_metadata_compatible(
        {**metadata, "implementation_fingerprint": "obsolete-source"}, request
    )
    assert not _navigator_metadata_compatible({**metadata, "maps": 2}, request)
    assert not _navigator_metadata_compatible({**metadata, "seed": 8}, request)
    assert not _navigator_metadata_compatible({**metadata, "final_accuracy": 0.9}, request)
    assert not _navigator_metadata_compatible(
        {**metadata, "target_patience_achieved": 1}, request
    )
    assert not _navigator_metadata_compatible({**metadata, "final_accuracy": "invalid"}, request)


def test_navigator_checkpoint_overwrite_is_atomic(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "navigator.pt"
    checkpoint.write_bytes(b"previous-valid-checkpoint")
    model = NavigatorMLP(3, hidden_sizes=(4,))

    def interrupted_save(payload, path) -> None:
        del payload
        with open(path, "wb") as handle:
            handle.write(b"partial")
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(torch, "save", interrupted_save)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        model.save_checkpoint(checkpoint)

    assert checkpoint.read_bytes() == b"previous-valid-checkpoint"
    assert not list(tmp_path.glob(".navigator.pt.*.tmp"))
