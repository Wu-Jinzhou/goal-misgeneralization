from __future__ import annotations

import pytest
import torch

from goalzendo_interactive.objectives import (
    TRAJECTORY_OBJECTIVE_SCHEMA_VERSION,
    TrajectoryObjectiveError,
    trajectory_objective_digest,
    trajectory_objective_manifest,
    trajectory_policy_gradient_objective,
    trajectory_sft_objective,
)
from goalzendo_interactive.trajectory_encoding import IGNORE_INDEX


def test_sft_objective_masks_prompts_and_shifts_next_token_targets() -> None:
    logits = torch.full((1, 5, 7), -3.0, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 3, 4, IGNORE_INDEX]])
    with torch.no_grad():
        logits[0, 1, 3] = 5.0
        logits[0, 2, 4] = 5.0
    result = trajectory_sft_objective(logits, labels)
    assert result.supervised_token_count == 2
    assert result.mean_token_accuracy.item() == 1.0
    assert result.loss.item() < 0.01
    result.loss.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[0, 0]).item() == 0
    assert torch.count_nonzero(logits.grad[0, 1]).item() > 0
    assert torch.count_nonzero(logits.grad[0, 2]).item() > 0
    assert torch.count_nonzero(logits.grad[0, 3]).item() == 0


def test_sft_objective_rejects_empty_nonfinite_and_out_of_vocabulary_targets() -> None:
    logits = torch.zeros((1, 3, 4))
    with pytest.raises(TrajectoryObjectiveError, match="no supervised"):
        trajectory_sft_objective(logits, torch.full((1, 3), IGNORE_INDEX))
    bad = logits.clone()
    bad[0, 0, 0] = torch.nan
    with pytest.raises(TrajectoryObjectiveError, match="non-finite"):
        trajectory_sft_objective(bad, torch.tensor([[IGNORE_INDEX, 1, IGNORE_INDEX]]))
    with pytest.raises(TrajectoryObjectiveError, match="out-of-vocabulary"):
        trajectory_sft_objective(logits, torch.tensor([[IGNORE_INDEX, 8, IGNORE_INDEX]]))


def _policy_inputs() -> tuple[torch.Tensor, ...]:
    token_log_probs = torch.tensor(
        [[-0.1, -9.0], [-0.2, -9.0], [-0.3, -9.0], [-0.4, -9.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    mask = torch.tensor([[True, False]] * 4)
    rewards = torch.tensor([1.0, 0.0, 0.25, 0.25], dtype=torch.float64)
    groups = torch.tensor([0, 0, 1, 1])
    entropies = torch.tensor([[0.5, 999.0]] * 4, dtype=torch.float64)
    return token_log_probs, mask, rewards, groups, entropies


def _digest(value: int) -> str:
    return f"{value:064x}"


def test_policy_gradient_uses_exact_leave_one_out_sequence_advantages() -> None:
    log_probs, mask, rewards, groups, entropies = _policy_inputs()
    result = trajectory_policy_gradient_objective(
        log_probs,
        mask,
        rewards,
        groups,
        (_digest(1), _digest(2), _digest(3), _digest(3)),
        episode_digests=(_digest(101), _digest(101), _digest(202), _digest(202)),
        token_entropies=entropies,
        entropy_coefficient=0.1,
        expected_rollouts_per_group=2,
    )
    assert result.advantages.tolist() == [1.0, -1.0, 0.0, 0.0]
    assert result.sequence_log_probabilities.tolist() == pytest.approx([-0.1, -0.2, -0.3, -0.4])
    assert result.policy_loss.item() == pytest.approx(-0.025)
    assert result.mean_action_token_entropy.item() == 0.5
    assert result.loss.item() == pytest.approx(-0.075)
    assert result.diagnostics.distinct_trajectory_group_fraction == 0.5
    assert result.diagnostics.all_zero_advantage_group_fraction == 0.5
    result.loss.backward()
    assert log_probs.grad is not None
    assert log_probs.grad[:, 1].tolist() == [0.0] * 4
    assert log_probs.grad[:, 0].tolist() == pytest.approx([-0.25, 0.25, 0.0, 0.0])


def test_policy_gradient_eight_rollout_groups_and_permutation_are_exact() -> None:
    log_probs = torch.linspace(-0.1, -1.6, 32, dtype=torch.float64).reshape(16, 2)
    mask = torch.ones_like(log_probs, dtype=torch.bool)
    rewards = torch.tensor([0.0, 1.0] * 8, dtype=torch.float64)
    groups = torch.tensor([7] * 8 + [3] * 8)
    digests = tuple(_digest(index % 3) for index in range(16))
    first = trajectory_policy_gradient_objective(
        log_probs,
        mask,
        rewards,
        groups,
        digests,
        episode_digests=(_digest(700),) * 8 + (_digest(300),) * 8,
    )
    permutation = torch.tensor([8, 0, 9, 1, 10, 2, 11, 3, 12, 4, 13, 5, 14, 6, 15, 7])
    second = trajectory_policy_gradient_objective(
        log_probs[permutation],
        mask[permutation],
        rewards[permutation],
        groups[permutation],
        tuple(digests[index] for index in permutation.tolist()),
        episode_digests=tuple(
            ((_digest(700),) * 8 + (_digest(300),) * 8)[index] for index in permutation.tolist()
        ),
    )
    assert first.loss.item() == pytest.approx(second.loss.item())
    assert first.diagnostics.group_count == 2
    assert first.diagnostics.rollouts_per_group == 8
    assert first.diagnostics.distinct_trajectory_group_fraction == 1.0


def test_authenticated_zero_action_abort_stays_in_leave_one_out_group() -> None:
    log_probs = torch.tensor(
        [[0.0, 0.0], [-0.2, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    mask = torch.tensor([[False, False], [True, False]])
    rewards = torch.tensor([0.0, 1.0], dtype=torch.float64)
    result = trajectory_policy_gradient_objective(
        log_probs,
        mask,
        rewards,
        torch.tensor([0, 0]),
        (_digest(1), _digest(2)),
        episode_digests=(_digest(101), _digest(101)),
        authenticated_zero_action_abort_mask=torch.tensor([True, False]),
        expected_rollouts_per_group=2,
    )
    assert result.advantages.tolist() == [-1.0, 1.0]
    assert result.sequence_log_probabilities.tolist() == [0.0, -0.2]
    assert result.policy_loss.item() == pytest.approx(0.1)
    assert result.diagnostics.authenticated_zero_action_abort_fraction == 0.5
    result.loss.backward()
    assert log_probs.grad is not None
    assert log_probs.grad[0].tolist() == [0.0, 0.0]
    assert log_probs.grad[1].tolist() == pytest.approx([-0.5, 0.0])

    all_empty_log_probs = torch.zeros((2, 1), dtype=torch.float64, requires_grad=True)
    all_empty = trajectory_policy_gradient_objective(
        all_empty_log_probs,
        torch.zeros((2, 1), dtype=torch.bool),
        torch.zeros(2, dtype=torch.float64),
        torch.tensor([0, 0]),
        (_digest(3), _digest(4)),
        episode_digests=(_digest(202), _digest(202)),
        authenticated_zero_action_abort_mask=torch.ones(2, dtype=torch.bool),
        token_entropies=torch.zeros((2, 1), dtype=torch.float64),
        entropy_coefficient=0.01,
        expected_rollouts_per_group=2,
    )
    assert all_empty.loss.item() == 0.0
    assert all_empty.mean_action_token_entropy.item() == 0.0
    all_empty.loss.backward()
    assert all_empty_log_probs.grad is not None
    assert torch.count_nonzero(all_empty_log_probs.grad).item() == 0


def test_policy_gradient_fails_closed_on_bad_groups_masks_entropy_and_numerics() -> None:
    log_probs, mask, rewards, groups, entropies = _policy_inputs()
    with pytest.raises(TrajectoryObjectiveError, match="exactly one hidden episode"):
        trajectory_policy_gradient_objective(
            log_probs,
            mask,
            rewards,
            groups,
            tuple(_digest(index) for index in range(4)),
            episode_digests=(_digest(1), _digest(2), _digest(3), _digest(3)),
            expected_rollouts_per_group=2,
        )
    with pytest.raises(TrajectoryObjectiveError, match="split across"):
        trajectory_policy_gradient_objective(
            log_probs,
            mask,
            rewards,
            groups,
            tuple(_digest(index) for index in range(4)),
            episode_digests=(_digest(1),) * 4,
            expected_rollouts_per_group=2,
        )
    with pytest.raises(TrajectoryObjectiveError, match="SHA-256"):
        trajectory_policy_gradient_objective(
            log_probs,
            mask,
            rewards,
            groups,
            ("a", "b", "c", "d"),
            episode_digests=(_digest(1), _digest(1), _digest(2), _digest(2)),
            expected_rollouts_per_group=2,
        )
    with pytest.raises(TrajectoryObjectiveError, match="exactly 8"):
        trajectory_policy_gradient_objective(
            log_probs,
            mask,
            rewards,
            groups,
            tuple(_digest(index) for index in range(4)),
            episode_digests=(_digest(1), _digest(1), _digest(2), _digest(2)),
        )
    empty_mask = mask.clone()
    empty_mask[0] = False
    with pytest.raises(TrajectoryObjectiveError, match="authenticated zero-action"):
        trajectory_policy_gradient_objective(
            log_probs,
            empty_mask,
            rewards,
            groups,
            tuple(_digest(index) for index in range(4)),
            episode_digests=(_digest(1), _digest(1), _digest(2), _digest(2)),
            expected_rollouts_per_group=2,
        )
    with pytest.raises(TrajectoryObjectiveError, match="exactly zero reward"):
        trajectory_policy_gradient_objective(
            log_probs,
            empty_mask,
            rewards,
            groups,
            tuple(_digest(index) for index in range(4)),
            episode_digests=(_digest(1), _digest(1), _digest(2), _digest(2)),
            authenticated_zero_action_abort_mask=torch.tensor([True, False, False, False]),
            expected_rollouts_per_group=2,
        )
    with pytest.raises(TrajectoryObjectiveError, match="required"):
        trajectory_policy_gradient_objective(
            log_probs,
            mask,
            rewards,
            groups,
            tuple(_digest(index) for index in range(4)),
            episode_digests=(_digest(1), _digest(1), _digest(2), _digest(2)),
            entropy_coefficient=0.01,
            expected_rollouts_per_group=2,
        )
    bad_entropy = entropies.clone()
    bad_entropy[0, 0] = -0.1
    with pytest.raises(TrajectoryObjectiveError, match="non-negative"):
        trajectory_policy_gradient_objective(
            log_probs,
            mask,
            rewards,
            groups,
            tuple(_digest(index) for index in range(4)),
            episode_digests=(_digest(1), _digest(1), _digest(2), _digest(2)),
            token_entropies=bad_entropy,
            expected_rollouts_per_group=2,
        )


def test_objective_manifest_is_explicit_and_digest_stable() -> None:
    manifest = trajectory_objective_manifest()
    assert TRAJECTORY_OBJECTIVE_SCHEMA_VERSION == 3
    assert manifest["outcome_rl"]["rollouts_per_episode"] == 8
    assert manifest["outcome_rl"]["sequence_reduction"] == ("sum_action_token_log_probabilities")
    assert manifest["outcome_rl"]["group_binding"].startswith("one verified hidden-episode")
    assert "empty sequence" in manifest["outcome_rl"]["zero_action_abort_rule"]
    assert len(trajectory_objective_digest()) == 64
