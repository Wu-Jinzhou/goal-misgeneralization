"""Small, architecture-agnostic learning primitives for hidden-law games."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


def _require_finite(value: Tensor, name: str) -> Tensor:
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if not bool(torch.isfinite(value).all().detach().cpu()):
        raise FloatingPointError(f"{name} contains NaN or infinity")
    return value


def _validate_categorical_scores(log_scores: Tensor, name: str = "log_scores") -> None:
    if log_scores.ndim < 1 or log_scores.shape[-1] < 2:
        raise ValueError(f"{name} must end in an action dimension of at least two")
    _require_finite(log_scores, name)


@dataclass(frozen=True)
class CategoricalSFTLoss:
    """Categorical cross-entropy and detached descriptive statistics."""

    loss: Tensor
    accuracy: Tensor
    mean_correct_probability: Tensor


def categorical_sft_loss(log_scores: Tensor, correct_actions: Tensor) -> CategoricalSFTLoss:
    """Ordinary categorical cross-entropy for any finite action count."""

    if log_scores.ndim < 2:
        raise ValueError("SFT log_scores must have shape [..., actions]")
    _validate_categorical_scores(log_scores)
    targets = correct_actions.to(device=log_scores.device, dtype=torch.long)
    if targets.shape != log_scores.shape[:-1]:
        raise ValueError("correct_actions must match every non-action dimension")
    if bool(((targets < 0) | (targets >= log_scores.shape[-1])).any().detach().cpu()):
        raise ValueError("correct_actions contains an out-of-range action index")
    flat_scores = log_scores.reshape(-1, log_scores.shape[-1])
    flat_targets = targets.reshape(-1)
    loss = _require_finite(
        F.cross_entropy(flat_scores, flat_targets),
        "categorical SFT loss",
    )
    probabilities = _require_finite(
        flat_scores.softmax(dim=-1),
        "categorical SFT probabilities",
    )
    correct_probabilities = probabilities.gather(-1, flat_targets[:, None]).squeeze(-1)
    accuracy = (flat_scores.argmax(dim=-1) == flat_targets).float().mean()
    return CategoricalSFTLoss(
        loss=loss,
        accuracy=accuracy.detach(),
        mean_correct_probability=correct_probabilities.mean().detach(),
    )


def sample_categorical_actions(
    log_scores: Tensor,
    sample_count: int,
    *,
    generator: torch.Generator,
) -> Tensor:
    """Draw replayable categorical actions using the caller's generator."""

    _validate_categorical_scores(log_scores)
    if not isinstance(sample_count, int) or isinstance(sample_count, bool) or sample_count < 1:
        raise ValueError("sample_count must be a positive integer")
    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be an explicit torch.Generator")
    probabilities = _require_finite(
        log_scores.softmax(dim=-1),
        "categorical sampling probabilities",
    )
    flat = probabilities.reshape(-1, probabilities.shape[-1])
    samples = torch.multinomial(
        flat,
        sample_count,
        replacement=True,
        generator=generator,
    )
    return samples.reshape(*log_scores.shape[:-1], sample_count)


@dataclass(frozen=True)
class CollectedCategoricalActions:
    """Detached policy snapshot and actions from the collection boundary."""

    log_scores: Tensor
    probabilities: Tensor
    actions: Tensor


@torch.no_grad()
def collect_categorical_actions(
    score_fn: Callable[[], Tensor],
    sample_count: int,
    *,
    generator: torch.Generator,
) -> CollectedCategoricalActions:
    """Run a scorer and sample actions with gradient recording disabled."""

    if not callable(score_fn):
        raise TypeError("score_fn must be callable")
    log_scores = score_fn()
    _validate_categorical_scores(log_scores, "collected log_scores")
    probabilities = _require_finite(
        log_scores.softmax(dim=-1),
        "collected categorical probabilities",
    )
    actions = sample_categorical_actions(
        log_scores,
        sample_count,
        generator=generator,
    )
    return CollectedCategoricalActions(
        log_scores=log_scores.detach(),
        probabilities=probabilities.detach(),
        actions=actions.detach(),
    )


@dataclass(frozen=True)
class RescoredTrajectory:
    """Differentiable scores for one fixed, previously collected trajectory."""

    log_probability_sum: Tensor
    decision_entropies: tuple[Tensor, ...]

    @property
    def decision_count(self) -> int:
        return len(self.decision_entropies)


def rescore_trajectory(
    decision_log_scores: Sequence[Tensor],
    sampled_actions: Sequence[int],
) -> RescoredTrajectory:
    """Sum chosen categorical log-probabilities across a variable-length path."""

    if not decision_log_scores:
        raise ValueError("a trajectory must contain at least one decision")
    if len(decision_log_scores) != len(sampled_actions):
        raise ValueError("sampled_actions must contain one index per decision")
    first = decision_log_scores[0]
    selected_log_probabilities: list[Tensor] = []
    entropies: list[Tensor] = []
    for step, (scores, action) in enumerate(zip(decision_log_scores, sampled_actions, strict=True)):
        if scores.ndim != 1:
            raise ValueError(f"decision {step} log_scores must have shape [actions]")
        _validate_categorical_scores(scores, f"decision {step} log_scores")
        if scores.device != first.device or scores.dtype != first.dtype:
            raise ValueError("all decisions in a trajectory must share device and dtype")
        if not isinstance(action, int) or isinstance(action, bool):
            raise TypeError("sampled action indices must be integers")
        if not 0 <= action < scores.shape[0]:
            raise ValueError(f"decision {step} has an out-of-range sampled action")
        log_probabilities = _require_finite(
            scores.log_softmax(dim=-1),
            f"decision {step} log probabilities",
        )
        probabilities = _require_finite(
            log_probabilities.exp(),
            f"decision {step} probabilities",
        )
        selected_log_probabilities.append(log_probabilities[action])
        entropies.append(-(probabilities * log_probabilities).sum())
    return RescoredTrajectory(
        log_probability_sum=_require_finite(
            torch.stack(selected_log_probabilities).sum(),
            "trajectory log-probability sum",
        ),
        decision_entropies=tuple(entropies),
    )


@dataclass(frozen=True)
class GroupedTrajectoryRLLoss:
    """Grouped leave-one-out episodic REINFORCE components."""

    loss: Tensor
    policy_loss: Tensor
    mean_trajectory_entropy_sum: Tensor
    mean_reward: Tensor
    baselines: Tensor
    advantages: Tensor
    trajectory_log_probability_sums: Tensor
    trajectory_lengths: Tensor


def grouped_leave_one_out_trajectory_loss(
    trajectories: Sequence[RescoredTrajectory],
    rewards: Tensor | Sequence[float],
    group_ids: Sequence[Hashable],
    *,
    entropy_coefficient: float = 0.0,
) -> GroupedTrajectoryRLLoss:
    """Apply leave-one-out REINFORCE to differentiably rescored trajectories.

    Rewards and within-group baselines are detached unconditionally.  The policy
    term gives each trajectory equal weight and uses its sum of chosen decision
    log-probabilities.  The entropy bonus is likewise the mean, across
    trajectories, of each trajectory's sum of exact categorical entropies.
    Every group must contain at least two independently collected rollouts.
    """

    if not trajectories:
        raise ValueError("trajectories must be non-empty")
    if len(trajectories) != len(group_ids):
        raise ValueError("group_ids must contain one value per trajectory")
    if not math.isfinite(float(entropy_coefficient)) or entropy_coefficient < 0:
        raise ValueError("entropy_coefficient must be finite and non-negative")

    first_score = trajectories[0].log_probability_sum
    if first_score.ndim != 0:
        raise ValueError("trajectory log_probability_sum values must be scalar")
    score_values: list[Tensor] = []
    entropy_sums: list[Tensor] = []
    lengths: list[int] = []
    for index, trajectory in enumerate(trajectories):
        score = trajectory.log_probability_sum
        if score.ndim != 0:
            raise ValueError("trajectory log_probability_sum values must be scalar")
        if score.device != first_score.device or score.dtype != first_score.dtype:
            raise ValueError("all trajectory scores must share device and dtype")
        _require_finite(score, f"trajectory {index} log-probability sum")
        if not trajectory.decision_entropies:
            raise ValueError("every trajectory must contain at least one decision entropy")
        for entropy in trajectory.decision_entropies:
            if entropy.ndim != 0 or entropy.device != score.device or entropy.dtype != score.dtype:
                raise ValueError("decision entropies must be scalar and match trajectory scores")
            _require_finite(entropy, f"trajectory {index} decision entropy")
        score_values.append(score)
        entropy_sums.append(torch.stack(trajectory.decision_entropies).sum())
        lengths.append(trajectory.decision_count)
    trajectory_scores = torch.stack(score_values)
    trajectory_entropy_sums = torch.stack(entropy_sums)

    observed_rewards = torch.as_tensor(
        rewards,
        dtype=first_score.dtype,
        device=first_score.device,
    ).detach()
    if observed_rewards.shape != (len(trajectories),):
        raise ValueError("rewards must have shape [trajectories]")
    _require_finite(observed_rewards, "trajectory rewards")

    members: dict[Hashable, list[int]] = defaultdict(list)
    for index, group_id in enumerate(group_ids):
        try:
            members[group_id].append(index)
        except TypeError as exc:
            raise TypeError("group_ids must be hashable") from exc
    if any(len(indices) < 2 for indices in members.values()):
        raise ValueError("each leave-one-out group requires at least two trajectories")
    baselines = torch.stack(
        [
            (observed_rewards[members[group_ids[index]]].sum() - observed_rewards[index])
            / (len(members[group_ids[index]]) - 1)
            for index in range(len(trajectories))
        ]
    ).detach()
    advantages = _require_finite(
        (observed_rewards - baselines).detach(),
        "trajectory advantages",
    )
    policy_loss = _require_finite(
        -(advantages * trajectory_scores).mean(),
        "trajectory policy loss",
    )
    entropy = _require_finite(
        trajectory_entropy_sums.mean(),
        "mean trajectory entropy sum",
    )
    loss = _require_finite(
        policy_loss - float(entropy_coefficient) * entropy,
        "grouped trajectory RL loss",
    )
    return GroupedTrajectoryRLLoss(
        loss=loss,
        policy_loss=policy_loss,
        mean_trajectory_entropy_sum=entropy.detach(),
        mean_reward=observed_rewards.mean().detach(),
        baselines=baselines,
        advantages=advantages,
        trajectory_log_probability_sums=trajectory_scores.detach(),
        trajectory_lengths=torch.tensor(
            lengths,
            dtype=torch.long,
            device=first_score.device,
        ),
    )


@dataclass(frozen=True)
class OfficialRoleBlockLoss:
    """Mean loss across the four Official-role rotations in one update block."""

    loss: Tensor
    rotation_losses: Tensor


def official_role_block_loss(rotation_losses: Sequence[Tensor]) -> OfficialRoleBlockLoss:
    """Average exactly four scalar losses before the caller performs one update."""

    if len(rotation_losses) != 4:
        raise ValueError("an Official-role block must contain exactly four rotations")
    first = rotation_losses[0]
    values: list[Tensor] = []
    for loss in rotation_losses:
        if loss.ndim != 0 or not loss.is_floating_point():
            raise ValueError("each role-rotation loss must be a floating-point scalar")
        if loss.device != first.device or loss.dtype != first.dtype:
            raise ValueError("role-rotation losses must share device and dtype")
        values.append(_require_finite(loss, "role-rotation loss"))
    stacked = torch.stack(values)
    return OfficialRoleBlockLoss(
        loss=_require_finite(stacked.mean(), "Official-role block loss"),
        rotation_losses=stacked.detach(),
    )
