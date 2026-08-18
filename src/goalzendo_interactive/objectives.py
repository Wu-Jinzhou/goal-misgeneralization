"""Training objectives for complete G03 interactive trajectories.

The functions in this module operate on already-tokenized trajectories and
already-sampled on-policy actions.  They deliberately know nothing about the
legacy two-choice GoalZendo scorer.  Environment validity and reward are
computed before this boundary; invalid or incomplete rollouts simply enter
with their registered zero reward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from ._json import json_digest
from .trajectory_encoding import IGNORE_INDEX

TRAJECTORY_OBJECTIVE_SCHEMA_VERSION = 3


class TrajectoryObjectiveError(ValueError):
    """Raised when an objective input is malformed or numerically unsafe."""


def _require_tensor(name: str, value: object, *, dimensions: int) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != dimensions:
        raise TrajectoryObjectiveError(f"{name} must be a rank-{dimensions} tensor")
    return value


def _require_finite(name: str, value: Tensor) -> Tensor:
    if not bool(torch.isfinite(value).all()):
        raise TrajectoryObjectiveError(f"{name} contains non-finite values")
    return value


@dataclass(frozen=True, slots=True)
class SFTObjective:
    """A differentiable assistant-token cross-entropy objective."""

    loss: Tensor
    supervised_token_count: int
    mean_token_accuracy: Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.loss, Tensor) or self.loss.ndim != 0:
            raise TrajectoryObjectiveError("SFT loss must be a scalar tensor")
        if (
            isinstance(self.supervised_token_count, bool)
            or not isinstance(self.supervised_token_count, int)
            or self.supervised_token_count < 1
        ):
            raise TrajectoryObjectiveError("SFT requires at least one supervised token")
        if not isinstance(self.mean_token_accuracy, Tensor) or self.mean_token_accuracy.ndim != 0:
            raise TrajectoryObjectiveError("SFT token accuracy must be a scalar tensor")


def trajectory_sft_objective(
    logits: Tensor,
    labels: Tensor,
    *,
    ignore_index: int = IGNORE_INDEX,
) -> SFTObjective:
    """Compute next-token CE only where assistant-action labels are present."""

    logits = _require_tensor("logits", logits, dimensions=3)
    labels = _require_tensor("labels", labels, dimensions=2)
    if logits.shape[:2] != labels.shape:
        raise TrajectoryObjectiveError("logits and labels must share batch and sequence shapes")
    if logits.shape[1] < 2 or logits.shape[2] < 2:
        raise TrajectoryObjectiveError("SFT logits need sequence length and vocabulary >= 2")
    if labels.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise TrajectoryObjectiveError("SFT labels must have an integer dtype")
    if isinstance(ignore_index, bool) or not isinstance(ignore_index, int):
        raise TrajectoryObjectiveError("ignore_index must be an integer")
    _require_finite("SFT logits", logits)

    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous().to(dtype=torch.long)
    mask = shifted_labels.ne(ignore_index)
    count = int(mask.sum().item())
    if count < 1:
        raise TrajectoryObjectiveError("SFT batch has no supervised next-token targets")
    selected_labels = shifted_labels[mask]
    if bool((selected_labels < 0).any()) or bool((selected_labels >= logits.shape[-1]).any()):
        raise TrajectoryObjectiveError("SFT labels contain an out-of-vocabulary token")

    loss = (
        F.cross_entropy(
            shifted_logits.reshape(-1, shifted_logits.shape[-1]),
            shifted_labels.reshape(-1),
            ignore_index=ignore_index,
            reduction="sum",
        )
        / count
    )
    accuracy = shifted_logits.argmax(dim=-1)[mask].eq(selected_labels).to(logits.dtype).mean()
    return SFTObjective(
        _require_finite("SFT loss", loss),
        count,
        _require_finite("SFT token accuracy", accuracy),
    )


@dataclass(frozen=True, slots=True)
class RolloutGroupDiagnostics:
    group_count: int
    rollout_count: int
    rollouts_per_group: int
    distinct_trajectory_group_fraction: float
    all_zero_advantage_group_fraction: float
    authenticated_zero_action_abort_fraction: float

    def __post_init__(self) -> None:
        for name in ("group_count", "rollout_count", "rollouts_per_group"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise TrajectoryObjectiveError(f"{name} must be a positive integer")
        for name in (
            "distinct_trajectory_group_fraction",
            "all_zero_advantage_group_fraction",
            "authenticated_zero_action_abort_fraction",
        ):
            value = getattr(self, name)
            if not isinstance(value, float) or not 0.0 <= value <= 1.0:
                raise TrajectoryObjectiveError(f"{name} must lie in [0, 1]")

    def as_obj(self) -> dict[str, int | float]:
        return {
            "group_count": self.group_count,
            "rollout_count": self.rollout_count,
            "rollouts_per_group": self.rollouts_per_group,
            "distinct_trajectory_group_fraction": self.distinct_trajectory_group_fraction,
            "all_zero_advantage_group_fraction": self.all_zero_advantage_group_fraction,
            "authenticated_zero_action_abort_fraction": (self.authenticated_zero_action_abort_fraction),
        }


@dataclass(frozen=True, slots=True)
class PolicyGradientObjective:
    """A differentiable sequence-level leave-one-out policy objective."""

    loss: Tensor
    policy_loss: Tensor
    mean_action_token_entropy: Tensor
    advantages: Tensor
    sequence_log_probabilities: Tensor
    diagnostics: RolloutGroupDiagnostics

    def __post_init__(self) -> None:
        for name in ("loss", "policy_loss", "mean_action_token_entropy"):
            value = getattr(self, name)
            if not isinstance(value, Tensor) or value.ndim != 0:
                raise TrajectoryObjectiveError(f"{name} must be a scalar tensor")
        for name in ("advantages", "sequence_log_probabilities"):
            value = getattr(self, name)
            if not isinstance(value, Tensor) or value.ndim != 1:
                raise TrajectoryObjectiveError(f"{name} must be a rank-1 tensor")
        if self.advantages.shape != self.sequence_log_probabilities.shape:
            raise TrajectoryObjectiveError("advantages and sequence log probabilities disagree")
        if type(self.diagnostics) is not RolloutGroupDiagnostics:
            raise TrajectoryObjectiveError("policy diagnostics have the wrong type")


def _validated_group_rows(
    group_ids: Tensor,
    *,
    rollout_count: int,
    expected_rollouts_per_group: int,
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    group_ids = _require_tensor("group_ids", group_ids, dimensions=1)
    if len(group_ids) != rollout_count:
        raise TrajectoryObjectiveError("group_ids must contain one id per rollout")
    if group_ids.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
        raise TrajectoryObjectiveError("group_ids must have an integer dtype")
    rows: dict[int, list[int]] = {}
    for row, raw_group in enumerate(group_ids.detach().cpu().tolist()):
        group = int(raw_group)
        if group < 0:
            raise TrajectoryObjectiveError("group ids must be non-negative")
        rows.setdefault(group, []).append(row)
    ordered = tuple((group, tuple(indices)) for group, indices in sorted(rows.items()))
    if not ordered or any(len(indices) != expected_rollouts_per_group for _, indices in ordered):
        raise TrajectoryObjectiveError(
            f"every group must contain exactly {expected_rollouts_per_group} rollouts"
        )
    return ordered


def _rollout_diagnostics(
    rows: tuple[tuple[int, tuple[int, ...]], ...],
    *,
    rewards: Tensor,
    trajectory_digests: tuple[str, ...],
    authenticated_zero_action_abort_mask: Tensor,
    expected_rollouts_per_group: int,
) -> RolloutGroupDiagnostics:
    if len(trajectory_digests) != len(rewards):
        raise TrajectoryObjectiveError("trajectory_digests must contain one digest per rollout")
    if any(
        type(digest) is not str
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        for digest in trajectory_digests
    ):
        raise TrajectoryObjectiveError("trajectory digests must be lowercase SHA-256 values")
    distinct_groups = 0
    zero_groups = 0
    detached_rewards = rewards.detach().cpu().tolist()
    for _, indices in rows:
        if len({trajectory_digests[index] for index in indices}) >= 2:
            distinct_groups += 1
        values = [float(detached_rewards[index]) for index in indices]
        if all(value == values[0] for value in values[1:]):
            zero_groups += 1
    group_count = len(rows)
    return RolloutGroupDiagnostics(
        group_count=group_count,
        rollout_count=len(rewards),
        rollouts_per_group=expected_rollouts_per_group,
        distinct_trajectory_group_fraction=distinct_groups / group_count,
        all_zero_advantage_group_fraction=zero_groups / group_count,
        authenticated_zero_action_abort_fraction=(
            int(authenticated_zero_action_abort_mask.sum().item()) / len(rewards)
        ),
    )


def _validate_episode_group_binding(
    rows: tuple[tuple[int, tuple[int, ...]], ...],
    *,
    episode_digests: tuple[str, ...],
    rollout_count: int,
) -> None:
    if len(episode_digests) != rollout_count:
        raise TrajectoryObjectiveError("episode_digests must contain one digest per rollout")
    if any(
        type(digest) is not str
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        for digest in episode_digests
    ):
        raise TrajectoryObjectiveError("episode digests must be lowercase SHA-256 values")
    episode_to_group: dict[str, int] = {}
    for group, indices in rows:
        group_episodes = {episode_digests[index] for index in indices}
        if len(group_episodes) != 1:
            raise TrajectoryObjectiveError("every rollout group must belong to exactly one hidden episode")
        episode = next(iter(group_episodes))
        prior = episode_to_group.setdefault(episode, group)
        if prior != group:
            raise TrajectoryObjectiveError("one hidden episode cannot be split across rollout groups")


def trajectory_policy_gradient_objective(
    token_log_probabilities: Tensor,
    action_token_mask: Tensor,
    rewards: Tensor,
    group_ids: Tensor,
    trajectory_digests: tuple[str, ...],
    *,
    episode_digests: tuple[str, ...],
    authenticated_zero_action_abort_mask: Tensor | None = None,
    token_entropies: Tensor | None = None,
    entropy_coefficient: float = 0.0,
    expected_rollouts_per_group: int = 8,
) -> PolicyGradientObjective:
    """Compute sequence-level REINFORCE with an exact leave-one-out baseline.

    ``token_log_probabilities`` contains the log probability of every sampled
    token.  Only positions selected by ``action_token_mask`` enter the policy
    gradient, thereby excluding prompts and oracle feedback. A rollout with
    no sampled action tokens is retained only when the caller supplies an
    exactly aligned authenticated-zero-action-abort mask and its reward is
    zero. Its sequence log probability is then the empty sum, zero, so it
    remains in the leave-one-out baseline without receiving a direct policy
    gradient. Advantages are detached and each rollout is compared with the
    other trajectories from the same hidden episode.
    """

    log_probs = _require_tensor("token_log_probabilities", token_log_probabilities, dimensions=2)
    mask = _require_tensor("action_token_mask", action_token_mask, dimensions=2)
    rewards = _require_tensor("rewards", rewards, dimensions=1)
    if log_probs.shape != mask.shape:
        raise TrajectoryObjectiveError("token log probabilities and action mask disagree")
    if len(rewards) != log_probs.shape[0]:
        raise TrajectoryObjectiveError("rewards must contain one scalar per rollout")
    if mask.dtype != torch.bool:
        raise TrajectoryObjectiveError("action_token_mask must be Boolean")
    empty_action_rows = mask.sum(dim=1).eq(0)
    if authenticated_zero_action_abort_mask is None:
        authenticated_empty = torch.zeros_like(empty_action_rows)
    else:
        authenticated_empty = _require_tensor(
            "authenticated_zero_action_abort_mask",
            authenticated_zero_action_abort_mask,
            dimensions=1,
        )
        if authenticated_empty.dtype != torch.bool:
            raise TrajectoryObjectiveError("authenticated_zero_action_abort_mask must be Boolean")
        if authenticated_empty.shape != empty_action_rows.shape:
            raise TrajectoryObjectiveError(
                "authenticated_zero_action_abort_mask must contain one value per rollout"
            )
        authenticated_empty = authenticated_empty.to(device=mask.device)
    if not bool(torch.equal(empty_action_rows, authenticated_empty)):
        raise TrajectoryObjectiveError(
            "zero-action rows must match authenticated zero-action abort evidence exactly"
        )
    if bool(rewards[empty_action_rows.to(device=rewards.device)].ne(0).any()):
        raise TrajectoryObjectiveError("an authenticated zero-action abort must have exactly zero reward")
    if (
        isinstance(expected_rollouts_per_group, bool)
        or not isinstance(expected_rollouts_per_group, int)
        or expected_rollouts_per_group < 2
    ):
        raise TrajectoryObjectiveError("expected_rollouts_per_group must be an integer >= 2")
    if isinstance(entropy_coefficient, bool) or not isinstance(entropy_coefficient, (int, float)):
        raise TrajectoryObjectiveError("entropy_coefficient must be a finite non-negative number")
    entropy_value = float(entropy_coefficient)
    if not torch.isfinite(torch.tensor(entropy_value)) or entropy_value < 0:
        raise TrajectoryObjectiveError("entropy_coefficient must be finite and non-negative")
    _require_finite("sampled token log probabilities", log_probs[mask])
    _require_finite("rollout rewards", rewards)

    rows = _validated_group_rows(
        group_ids,
        rollout_count=len(rewards),
        expected_rollouts_per_group=expected_rollouts_per_group,
    )
    _validate_episode_group_binding(
        rows,
        episode_digests=tuple(episode_digests),
        rollout_count=len(rewards),
    )
    advantages = torch.empty_like(rewards)
    for _, indices in rows:
        index = torch.tensor(indices, dtype=torch.long, device=rewards.device)
        group_rewards = rewards.index_select(0, index)
        baselines = (group_rewards.sum() - group_rewards) / (len(indices) - 1)
        advantages.index_copy_(0, index, group_rewards - baselines)
    advantages = advantages.detach()
    sequence_log_probs = log_probs.masked_fill(~mask, 0).sum(dim=1)
    policy_loss = -(advantages * sequence_log_probs).mean()

    if token_entropies is None:
        if entropy_value != 0.0:
            raise TrajectoryObjectiveError("token_entropies are required when entropy_coefficient is nonzero")
        mean_entropy = log_probs.new_zeros(())
    else:
        entropies = _require_tensor("token_entropies", token_entropies, dimensions=2)
        if entropies.shape != log_probs.shape:
            raise TrajectoryObjectiveError("token entropies and log probabilities disagree")
        _require_finite("action-token entropies", entropies[mask])
        if bool((entropies[mask] < 0).any()):
            raise TrajectoryObjectiveError("token entropies must be non-negative")
        mean_entropy = entropies[mask].mean() if bool(mask.any()) else sequence_log_probs.sum() * 0.0
    loss = policy_loss - entropy_value * mean_entropy
    diagnostics = _rollout_diagnostics(
        rows,
        rewards=rewards,
        trajectory_digests=trajectory_digests,
        authenticated_zero_action_abort_mask=authenticated_empty,
        expected_rollouts_per_group=expected_rollouts_per_group,
    )
    return PolicyGradientObjective(
        _require_finite("policy-gradient loss", loss),
        _require_finite("policy loss", policy_loss),
        _require_finite("mean token entropy", mean_entropy),
        _require_finite("leave-one-out advantages", advantages),
        _require_finite("sequence log probabilities", sequence_log_probs),
        diagnostics,
    )


def trajectory_objective_manifest() -> dict[str, Any]:
    """Return the immutable mathematical choices needed by a run manifest."""

    return {
        "schema_version": TRAJECTORY_OBJECTIVE_SCHEMA_VERSION,
        "sft": {
            "loss": "mean_next_token_cross_entropy",
            "mask": "assistant_action_content_only",
            "ignore_index": IGNORE_INDEX,
        },
        "outcome_rl": {
            "estimator": "sequence_reinforce_leave_one_out",
            "sequence_reduction": "sum_action_token_log_probabilities",
            "rollouts_per_episode": 8,
            "group_binding": "one verified hidden-episode SHA-256 digest per group",
            "advantage_normalization": "none",
            "prompt_oracle_feedback_tokens": "excluded",
            "entropy_reduction": "mean_action_token_entropy",
            "zero_action_abort_rule": (
                "retain only replay-authenticated turn-zero aborts with reward zero; "
                "use the empty sequence log-probability sum"
            ),
        },
    }


def trajectory_objective_digest() -> str:
    return json_digest(
        trajectory_objective_manifest(),
        domain="goalzendo-interactive-trajectory-objectives-v3",
    )
