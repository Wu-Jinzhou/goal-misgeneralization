from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from goalzendo_hidden_law.training import (
    RescoredTrajectory,
    categorical_sft_loss,
    collect_categorical_actions,
    grouped_leave_one_out_trajectory_loss,
    official_role_block_loss,
    rescore_trajectory,
    sample_categorical_actions,
)


@pytest.mark.parametrize("action_count", [2, 4, 9])
def test_categorical_sft_is_exact_ordinary_cross_entropy(action_count: int) -> None:
    generator = torch.Generator().manual_seed(3 + action_count)
    scores = torch.randn(2, 3, action_count, generator=generator, requires_grad=True)
    targets = torch.randint(action_count, (2, 3), generator=generator)
    result = categorical_sft_loss(scores, targets)
    expected = F.cross_entropy(scores.reshape(-1, action_count), targets.reshape(-1))
    assert torch.allclose(result.loss, expected)

    expected_gradient = torch.autograd.grad(expected, scores, retain_graph=True)[0]
    observed_gradient = torch.autograd.grad(result.loss, scores)[0]
    assert torch.allclose(observed_gradient, expected_gradient)
    assert not result.accuracy.requires_grad
    assert not result.mean_correct_probability.requires_grad


@pytest.mark.parametrize("action_count", [2, 4, 9])
def test_categorical_sampling_is_replayable_with_an_explicit_generator(
    action_count: int,
) -> None:
    scores = torch.linspace(-1.0, 1.0, 3 * action_count).reshape(3, action_count)
    first = sample_categorical_actions(
        scores,
        20,
        generator=torch.Generator().manual_seed(101),
    )
    second = sample_categorical_actions(
        scores,
        20,
        generator=torch.Generator().manual_seed(101),
    )
    assert first.shape == (3, 20)
    assert torch.equal(first, second)
    assert bool(((first >= 0) & (first < action_count)).all())

    with pytest.raises(TypeError, match="Generator"):
        sample_categorical_actions(scores, 1, generator=None)  # type: ignore[arg-type]


def test_collection_callback_runs_entirely_outside_autograd() -> None:
    parameter = torch.nn.Parameter(torch.tensor([[0.5, -0.5, 1.0, -1.0]]))
    grad_modes: list[bool] = []

    def score() -> torch.Tensor:
        grad_modes.append(torch.is_grad_enabled())
        return parameter * 2

    collected = collect_categorical_actions(
        score,
        8,
        generator=torch.Generator().manual_seed(11),
    )
    assert grad_modes == [False]
    assert collected.actions.shape == (1, 8)
    assert not collected.log_scores.requires_grad
    assert not collected.probabilities.requires_grad
    assert parameter.grad is None


def _make_variable_trajectories() -> tuple[
    list[torch.Tensor],
    list[RescoredTrajectory],
]:
    generator = torch.Generator().manual_seed(29)
    leaves: list[torch.Tensor] = []
    trajectories: list[RescoredTrajectory] = []
    specifications = (
        ((2,), (0,)),
        ((4, 9), (2, 8)),
        ((9, 2, 4), (3, 1, 0)),
        ((4,), (1,)),
        ((2, 9), (0, 7)),
    )
    for action_counts, actions in specifications:
        decisions: list[torch.Tensor] = []
        for action_count in action_counts:
            leaf = torch.randn(
                action_count,
                dtype=torch.float64,
                generator=generator,
                requires_grad=True,
            )
            leaves.append(leaf)
            decisions.append(leaf)
        trajectories.append(rescore_trajectory(decisions, actions))
    return leaves, trajectories


def test_variable_length_grouped_reinforce_is_exact_ordinary_loo_and_has_gradients() -> None:
    leaves, trajectories = _make_variable_trajectories()
    rewards = torch.tensor(
        [1.0, 0.0, 0.0, 1.0, 0.5],
        dtype=torch.float64,
        requires_grad=True,
    )
    result = grouped_leave_one_out_trajectory_loss(
        trajectories,
        rewards,
        ("first", "first", "second", "second", "second"),
        entropy_coefficient=0.07,
    )

    expected_baselines = torch.tensor([0.0, 1.0, 0.75, 0.25, 0.5], dtype=torch.float64)
    expected_advantages = rewards.detach() - expected_baselines
    expected_scores = torch.stack([trajectory.log_probability_sum for trajectory in trajectories])
    expected_entropy = torch.stack(
        [torch.stack(trajectory.decision_entropies).sum() for trajectory in trajectories]
    ).mean()
    expected_policy = -(expected_advantages * expected_scores).mean()
    expected_loss = expected_policy - 0.07 * expected_entropy

    assert torch.allclose(result.baselines, expected_baselines)
    assert torch.allclose(result.advantages, expected_advantages)
    assert torch.allclose(result.policy_loss, expected_policy)
    assert torch.allclose(result.mean_trajectory_entropy_sum, expected_entropy.detach())
    assert torch.allclose(result.loss, expected_loss)
    assert torch.equal(result.trajectory_lengths, torch.tensor([1, 2, 3, 1, 2]))
    assert not result.baselines.requires_grad
    assert not result.advantages.requires_grad

    expected_gradients = torch.autograd.grad(
        expected_loss,
        leaves,
        retain_graph=True,
    )
    observed_gradients = torch.autograd.grad(
        result.loss,
        leaves,
        retain_graph=True,
    )
    assert all(
        torch.allclose(observed, expected)
        for observed, expected in zip(
            observed_gradients,
            expected_gradients,
            strict=True,
        )
    )
    result.loss.backward()
    assert rewards.grad is None
    assert all(leaf.grad is not None for leaf in leaves)
    assert all(torch.count_nonzero(leaf.grad) > 0 for leaf in leaves if leaf.grad is not None)


def test_rescoring_sums_selected_log_probabilities_for_mixed_action_counts() -> None:
    first = torch.tensor([0.2, -0.3], requires_grad=True)
    second = torch.tensor([1.0, 0.0, -1.0, 0.5], requires_grad=True)
    third = torch.linspace(-1.0, 1.0, 9, requires_grad=True)
    result = rescore_trajectory((first, second, third), (1, 3, 8))
    expected = first.log_softmax(dim=-1)[1] + second.log_softmax(dim=-1)[3] + third.log_softmax(dim=-1)[8]
    assert torch.allclose(result.log_probability_sum, expected)
    assert result.decision_count == 3


def test_role_block_is_exact_mean_of_four_rotations_before_one_update() -> None:
    parameter = torch.nn.Parameter(torch.tensor(0.25))
    losses = tuple((parameter - target).square() for target in (1.0, 2.0, 3.0, 4.0))
    result = official_role_block_loss(losses)
    expected = torch.stack(losses).mean()
    assert torch.allclose(result.loss, expected)
    assert torch.allclose(result.rotation_losses, torch.stack(losses).detach())

    expected_gradient = torch.autograd.grad(expected, parameter, retain_graph=True)[0]
    observed_gradient = torch.autograd.grad(result.loss, parameter)[0]
    assert torch.allclose(observed_gradient, expected_gradient)


def test_grouped_reinforce_requires_true_leave_one_out_groups() -> None:
    trajectory = rescore_trajectory((torch.zeros(2, requires_grad=True),), (0,))
    with pytest.raises(ValueError, match="at least two"):
        grouped_leave_one_out_trajectory_loss((trajectory,), (1.0,), ("only",))


def test_training_primitives_reject_nonfinite_scores_and_rewards() -> None:
    with pytest.raises(FloatingPointError, match="log_scores"):
        categorical_sft_loss(torch.tensor([[0.0, torch.nan]]), torch.tensor([0]))
    trajectory = rescore_trajectory((torch.zeros(2, requires_grad=True),), (0,))
    with pytest.raises(FloatingPointError, match="rewards"):
        grouped_leave_one_out_trajectory_loss(
            (trajectory, trajectory),
            (1.0, float("inf")),
            (0, 0),
        )
