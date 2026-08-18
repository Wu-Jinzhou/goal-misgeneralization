"""Invariant and hand-calculation tests for the multi-fork RouteWorld substrate."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from forkworld.envs import GridAction
from forkworld.routeworld import (
    BinaryTreeRouteMaze,
    flip_route_code,
    flip_route_proxy,
    intervene_route_batch,
    make_route_dataset,
    make_route_panels,
    mask_route_code,
    mask_route_proxy,
    permute_route_addresses,
    rollout_route,
    route_metrics,
)


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_binary_tree_has_equal_unique_routes_and_real_walls(depth: int) -> None:
    routes = tuple(itertools.product((-1, 1), repeat=depth))
    env = BinaryTreeRouteMaze(depth, routes[0])

    assert len(env.leaves) == 2**depth
    assert len(set(env.leaves)) == len(env.leaves)
    assert env.walls
    assert "#" in env.render_ascii()
    assert "T" in env.render_ascii()
    assert env.route_length == 2**depth - 1 + 3 * depth

    endpoints = set()
    for route in routes:
        actions = env.actions_for_route(route)
        assert len(actions) == env.route_length
        rollout = rollout_route(BinaryTreeRouteMaze(depth, route), route)
        assert rollout.success
        assert rollout.total_reward == 1.0
        assert rollout.collisions == 0
        assert rollout.terminated and not rollout.truncated
        assert len(rollout.actions) == env.route_length
        assert rollout.positions[-1] == env.route_to_leaf(route)
        endpoints.add(rollout.positions[-1])
    assert endpoints == set(env.leaves)


def test_layout_is_deterministic_and_observation_does_not_reveal_target() -> None:
    left = BinaryTreeRouteMaze(4, (-1, -1, -1, -1))
    right = BinaryTreeRouteMaze(4, (1, 1, 1, 1))

    assert left.semantic_id == right.semantic_id
    assert left.free_cells == right.free_cells
    assert left.walls == right.walls
    assert left.leaves == right.leaves
    assert left.render_ascii() == right.render_ascii()
    assert left.observation() == right.observation()
    assert left.observation().vector() == right.observation().vector()
    assert "target" not in left.observation().as_dict()
    assert left.target_leaf != right.target_leaf

    # Environment feedback also withholds the intended route before termination.
    left_result = left.step(GridAction.DOWN)
    right_result = right.step(GridAction.DOWN)
    assert left_result.observation == right_result.observation
    assert left_result.reward == right_result.reward == 0.0
    assert "target_route" not in left_result.info


def test_one_wrong_choice_reaches_a_decoy_and_cannot_be_repaired_within_horizon() -> None:
    target = (-1, -1, -1, -1)
    selected = (1, -1, -1, -1)
    env = BinaryTreeRouteMaze(4, target)
    rollout = rollout_route(env, selected)
    assert rollout.reached_route == selected
    assert not rollout.success
    assert rollout.total_reward == 0.0
    assert rollout.terminated and not rollout.truncated
    assert len(rollout.actions) == env.route_length

    env.reset()
    env.step(GridAction.DOWN)
    env.step(GridAction.RIGHT)
    correction_distance = env.distance_to_route(target)
    assert correction_distance is not None
    assert correction_distance > env.max_steps - env.steps


def test_controller_queries_policy_only_at_real_forks() -> None:
    env = BinaryTreeRouteMaze(4, (-1, 1, -1, 1))
    calls = []

    def policy(decision) -> int:
        calls.append(
            (
                decision.stage,
                decision.node_index,
                decision.selected_prefix,
                decision.observation.position,
            )
        )
        return env.target_route[decision.stage]

    rollout = rollout_route(env, policy)
    assert rollout.success
    assert len(calls) == env.depth
    assert [call[0] for call in calls] == list(range(env.depth))
    for stage, node_index, _, position in calls:
        assert position == env.decision_positions[(stage, node_index)]


def test_route_dataset_has_exact_factorial_balance_proxy_accuracy_and_codes() -> None:
    batch = make_route_dataset(
        400,
        depth=4,
        q=0.95,
        k=3,
        max_depth=4,
        max_k=5,
        seed=17,
    )
    repeated = make_route_dataset(
        400,
        depth=4,
        q=0.95,
        k=3,
        max_depth=4,
        max_k=5,
        seed=17,
    )
    different = make_route_dataset(
        400,
        depth=4,
        q=0.95,
        k=3,
        max_depth=4,
        max_k=5,
        seed=18,
    )

    patterns, counts = np.unique(batch.targets[:, :4], axis=0, return_counts=True)
    assert len(patterns) == 16
    assert np.all(counts == 25)
    for stage in range(batch.depth):
        for sign in (-1, 1):
            rows = batch.targets[:, stage] == sign
            assert int(np.sum(rows)) == 200
            assert float(np.mean(batch.proxy[rows, stage] == sign)) == 0.95
    assert np.array_equal(np.prod(batch.code[:, :4, :3], axis=2), batch.targets[:, :4])
    assert np.all(batch.proxy_present[:, :4])
    assert np.all(batch.code_present[:, :4, :3])
    assert np.all(batch.code[:, :, 3:] == 0)
    assert np.all(~batch.code_present[:, :, 3:])
    assert batch.x.shape == (1600, 4 * (3 + 2 * 5))
    assert len(batch.feature_names) == batch.input_dim
    assert np.array_equal(batch.targets, repeated.targets)
    assert np.array_equal(batch.proxy, repeated.proxy)
    assert np.array_equal(batch.code, repeated.code)
    assert not np.array_equal(batch.targets, different.targets)
    assert not batch.targets.flags.writeable
    assert not batch.code.flags.writeable


def test_padding_keeps_feature_width_fixed_across_depth_and_degree() -> None:
    shallow = make_route_dataset(
        400, depth=1, q=0.95, k=1, max_depth=4, max_k=5, seed=9
    )
    deep = make_route_dataset(
        400, depth=4, q=0.95, k=5, max_depth=4, max_k=5, seed=9
    )
    assert shallow.input_dim == deep.input_dim
    assert shallow.feature_names == deep.feature_names
    assert shallow.x.shape[1] == deep.x.shape[1]
    assert np.all(shallow.targets[:, 1:] == 0)
    assert np.all(shallow.proxy[:, 1:] == 0)
    assert np.all(~shallow.proxy_present[:, 1:])
    assert np.all(shallow.code[:, 1:, :] == 0)
    assert np.all(shallow.code[:, :, 1:] == 0)
    assert np.all(~shallow.code_present[:, 1:, :])
    assert np.all(~shallow.code_present[:, :, 1:])


def test_evaluation_panels_are_semantically_matched_and_exactly_conflicted() -> None:
    panels = make_route_panels(
        400, depth=4, q=0.95, k=3, max_depth=4, max_k=5, seed=23
    )
    for compared in (panels.all_conflict, *panels.single_conflict):
        assert np.array_equal(compared.targets, panels.iid.targets)
        assert np.array_equal(compared.code, panels.iid.code)
        assert np.array_equal(compared.stage_features, panels.iid.stage_features)
    assert np.array_equal(
        panels.all_conflict.proxy[:, :4], -panels.all_conflict.targets[:, :4]
    )
    for conflict_stage, panel in enumerate(panels.single_conflict):
        agreement = panel.proxy[:, :4] == panel.targets[:, :4]
        assert np.all(~agreement[:, conflict_stage])
        assert np.all(np.delete(agreement, conflict_stage, axis=1))
    assert set(panels.as_dict()) == {
        "iid",
        "all_conflict",
        "single_conflict_0",
        "single_conflict_1",
        "single_conflict_2",
        "single_conflict_3",
    }


def test_causal_interventions_are_scoped_paired_and_nonmutating() -> None:
    batch = make_route_dataset(
        400, depth=4, q=0.95, k=3, max_depth=4, max_k=5, seed=31
    )
    original_x = batch.x.copy()

    proxy_flip = flip_route_proxy(batch, [2])
    assert np.array_equal(proxy_flip.proxy[:, 2], -batch.proxy[:, 2])
    assert np.array_equal(proxy_flip.proxy[:, [0, 1, 3]], batch.proxy[:, [0, 1, 3]])

    code_flip = flip_route_code(batch, stages=[1], channels=[2])
    assert np.array_equal(code_flip.code[:, 1, 2], -batch.code[:, 1, 2])
    unchanged = np.ones_like(batch.code, dtype=bool)
    unchanged[:, 1, 2] = False
    assert np.array_equal(code_flip.code[unchanged], batch.code[unchanged])

    proxy_mask = mask_route_proxy(batch, [0])
    assert np.all(proxy_mask.proxy[:, 0] == 0)
    assert np.all(~proxy_mask.proxy_present[:, 0])
    visible_proxy_mask = mask_route_proxy(batch, [0], hide_presence=False)
    assert np.all(visible_proxy_mask.proxy[:, 0] == 0)
    assert np.all(visible_proxy_mask.proxy_present[:, 0])
    code_mask = mask_route_code(batch, stages=[3], channels=[0, 1])
    assert np.all(code_mask.code[:, 3, :2] == 0)
    assert np.all(~code_mask.code_present[:, 3, :2])
    visible_code_mask = mask_route_code(
        batch, stages=[3], channels=[0, 1], hide_presence=False
    )
    assert np.all(visible_code_mask.code[:, 3, :2] == 0)
    assert np.all(visible_code_mask.code_present[:, 3, :2])

    permuted = permute_route_addresses(batch, (3, 2, 1, 0))
    assert np.all(permuted.stage_features[:, 0, 3] == 1)
    assert np.all(permuted.stage_features[:, 3, 0] == 1)

    combined = intervene_route_batch(
        batch,
        flip_proxy_stages=[1],
        flip_code_coordinates=[(2, 0)],
        mask_proxy_stages=[3],
        mask_code_coordinates=[(0, 1)],
    )
    assert np.array_equal(combined.proxy[:, 1], -batch.proxy[:, 1])
    assert np.all(combined.proxy[:, 3] == 0)
    assert np.array_equal(combined.code[:, 2, 0], -batch.code[:, 2, 0])
    assert np.all(combined.code[:, 0, 1] == 0)
    assert np.array_equal(batch.x, original_x)
    assert not np.array_equal(combined.x, original_x)


def test_route_metrics_match_a_hand_constructed_error_pattern() -> None:
    batch = make_route_dataset(
        4, depth=2, q=0.5, k=1, max_depth=4, max_k=3, seed=5
    )
    target = batch.targets[:, :2].copy()
    predicted = target.copy()
    predicted[1, 0] *= -1
    predicted[2, 1] *= -1
    predicted[3, :] *= -1
    metrics = route_metrics(predicted, batch)

    assert metrics["n_episodes"] == 4
    assert metrics["n_branches"] == 8
    assert metrics["branch_accuracy"] == 0.5
    assert metrics["stage_branch_accuracy"] == [0.5, 0.5]
    assert metrics["full_route_success"] == 0.25
    assert metrics["effective_per_fork_success"] == 0.5
    assert metrics["independent_compounding_prediction"] == 0.25
    assert metrics["pooled_independent_prediction"] == 0.25
    assert metrics["compounding_gap"] == 0.0
    assert metrics["first_divergence_rate"] == [0.5, 0.25, 0.25]
    assert metrics["first_divergence_mean"] == 0.75


def test_route_metrics_break_exact_zero_logits_toward_positive_action() -> None:
    batch = make_route_dataset(
        4, depth=1, q=0.5, k=1, max_depth=1, max_k=1, seed=7
    )
    metrics = route_metrics(np.zeros(len(batch), dtype=np.float64), batch)

    assert metrics["branch_accuracy"] == float(np.mean(batch.y == 1))
    assert metrics["branch_accuracy"] == 0.5


def test_dataset_rejects_rounded_proxy_counts_and_unbalanced_route_counts() -> None:
    with pytest.raises(ValueError, match="exactly realizable"):
        make_route_dataset(16, depth=4, q=0.95, k=2)
    with pytest.raises(ValueError, match="divisible"):
        make_route_dataset(18, depth=4, q=0.5, k=2)
