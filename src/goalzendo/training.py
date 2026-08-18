"""Auditable training objectives and scheduling primitives for GoalZendo."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _require_finite(value: Tensor, name: str) -> Tensor:
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if not bool(torch.isfinite(value).all().detach().cpu()):
        raise FloatingPointError(f"{name} contains NaN or infinity")
    return value


def _validate_binary_logits(action_logits: Tensor, name: str = "action_logits") -> None:
    if action_logits.ndim < 2 or action_logits.shape[-1] != 2:
        raise ValueError(f"{name} must end in a two-action dimension")
    if not action_logits.is_floating_point():
        raise ValueError(f"{name} must be floating point")
    _require_finite(action_logits, name)


@dataclass(frozen=True)
class SFTLoss:
    """Supervised two-action loss and detached descriptive statistics."""

    loss: Tensor
    accuracy: Tensor
    mean_correct_probability: Tensor


def sft_action_loss(action_logits: Tensor, correct_actions: Tensor) -> SFTLoss:
    """Cross-entropy on the exact correct GoalZendo action."""

    _validate_binary_logits(action_logits)
    if action_logits.ndim != 2:
        raise ValueError("SFT action_logits must have shape [batch, 2]")
    targets = correct_actions.to(device=action_logits.device, dtype=torch.long)
    if targets.shape != action_logits.shape[:-1]:
        raise ValueError("correct_actions must have shape [batch]")
    if torch.any((targets < 0) | (targets > 1)):
        raise ValueError("correct_actions must contain only 0 (A) or 1 (B)")
    loss = F.cross_entropy(action_logits, targets)
    _require_finite(loss, "SFT loss")
    probabilities = _require_finite(
        action_logits.softmax(dim=-1),
        "SFT action probabilities",
    )
    correct_probabilities = probabilities.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    accuracy = (action_logits.argmax(dim=-1) == targets).float().mean()
    return SFTLoss(
        loss=loss,
        accuracy=accuracy.detach(),
        mean_correct_probability=correct_probabilities.mean().detach(),
    )


@dataclass(frozen=True)
class OutcomeRLLoss:
    """Components of a leave-one-out REINFORCE objective."""

    loss: Tensor
    policy_loss: Tensor
    entropy: Tensor
    kl: Tensor
    mean_reward: Tensor
    baselines: Tensor
    advantages: Tensor


@dataclass(frozen=True)
class ExpectedOutcomeRLLoss:
    """Full-information two-action objective and reward-gradient diagnostics."""

    loss: Tensor
    policy_loss: Tensor
    entropy: Tensor
    kl: Tensor
    mean_reward: Tensor
    gradient_exponent: float
    mean_correct_probability: Tensor
    mean_expected_reward_logit_gradient_magnitude: Tensor
    mean_logprob_logit_gradient_magnitude: Tensor
    mean_policy_logit_gradient_magnitude: Tensor


def leave_one_out_baselines(rewards: Tensor) -> Tensor:
    """Use the other rollouts for the same prompt as each rollout's baseline."""

    if rewards.ndim != 2:
        raise ValueError("rewards must have shape [prompts, samples_per_prompt]")
    sample_count = rewards.shape[1]
    if sample_count < 2:
        raise ValueError("leave-one-out baselines require at least two samples per prompt")
    rewards_float = rewards.float()
    _require_finite(rewards_float, "rewards")
    return _require_finite(
        (rewards_float.sum(dim=1, keepdim=True) - rewards_float) / (sample_count - 1),
        "leave-one-out baselines",
    )


def outcome_rl_loss(
    action_logits: Tensor,
    sampled_actions: Tensor,
    rewards: Tensor,
    *,
    entropy_coefficient: float = 0.0,
    reference_logits: Tensor | None = None,
    kl_coefficient: float = 0.0,
) -> OutcomeRLLoss:
    """One-step outcome RL with group leave-one-out advantages.

    ``action_logits`` has shape ``[prompts, samples, 2]``.  Each group consists
    of independent actions sampled for the same prompt.  The objective uses an
    exact categorical entropy and, when supplied, an exact forward KL to a
    frozen reference distribution.  Rewards and baselines are detached from
    the policy by construction.
    """

    _validate_binary_logits(action_logits)
    if action_logits.ndim != 3:
        raise ValueError("RL action_logits must have shape [prompts, samples, 2]")
    actions = sampled_actions.to(device=action_logits.device, dtype=torch.long)
    observed_rewards = rewards.to(device=action_logits.device, dtype=torch.float32)
    expected_shape = action_logits.shape[:-1]
    if actions.shape != expected_shape or observed_rewards.shape != expected_shape:
        raise ValueError("sampled_actions and rewards must have shape [prompts, samples]")
    if torch.any((actions < 0) | (actions > 1)):
        raise ValueError("sampled_actions must contain only 0 (A) or 1 (B)")
    if (
        not math.isfinite(float(entropy_coefficient))
        or not math.isfinite(float(kl_coefficient))
        or entropy_coefficient < 0
        or kl_coefficient < 0
    ):
        raise ValueError("entropy and KL coefficients must be non-negative")

    baselines = leave_one_out_baselines(observed_rewards)
    advantages = _require_finite(
        (observed_rewards - baselines).detach(),
        "RL advantages",
    )
    log_probabilities = _require_finite(
        action_logits.log_softmax(dim=-1),
        "RL log probabilities",
    )
    probabilities = _require_finite(log_probabilities.exp(), "RL probabilities")
    selected_log_probabilities = log_probabilities.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    policy_loss = -(advantages * selected_log_probabilities).mean()
    entropy = -(probabilities * log_probabilities).sum(dim=-1).mean()
    _require_finite(policy_loss, "RL policy loss")
    _require_finite(entropy, "RL entropy")

    if reference_logits is None:
        kl = action_logits.new_zeros(())
    else:
        _validate_binary_logits(reference_logits, "reference_logits")
        if reference_logits.shape != action_logits.shape:
            raise ValueError("reference_logits must have the same shape as action_logits")
        reference_log_probabilities = reference_logits.detach().log_softmax(dim=-1)
        _require_finite(reference_log_probabilities, "reference log probabilities")
        kl = (probabilities * (log_probabilities - reference_log_probabilities)).sum(dim=-1).mean()
        _require_finite(kl, "RL KL")

    loss = policy_loss + float(kl_coefficient) * kl - float(entropy_coefficient) * entropy
    _require_finite(loss, "RL loss")
    return OutcomeRLLoss(
        loss=loss,
        policy_loss=policy_loss,
        entropy=entropy.detach(),
        kl=kl.detach(),
        mean_reward=observed_rewards.mean().detach(),
        baselines=baselines.detach(),
        advantages=advantages,
    )


def expected_outcome_rl_loss(
    action_logits: Tensor,
    correct_actions: Tensor,
    *,
    entropy_coefficient: float = 0.0,
    reference_logits: Tensor | None = None,
    kl_coefficient: float = 0.0,
    reward_correct: float = 1.0,
    reward_incorrect: float = 0.0,
) -> ExpectedOutcomeRLLoss:
    """Enumerate the exact expected reward of both GoalZendo actions.

    Unlike :func:`outcome_rl_loss`, this objective does not sample an action
    or estimate a leave-one-out advantage.  GoalZendo has a deterministic
    binary reward, so the expectation can be evaluated exactly.  This is a
    full-information estimator/feedback control, not an information-matched
    replacement for sampled outcome RL.
    """

    return power_outcome_control_loss(
        action_logits,
        correct_actions,
        gradient_exponent=1.0,
        entropy_coefficient=entropy_coefficient,
        reference_logits=reference_logits,
        kl_coefficient=kl_coefficient,
        reward_correct=reward_correct,
        reward_incorrect=reward_incorrect,
    )


def power_outcome_control_loss(
    action_logits: Tensor,
    correct_actions: Tensor,
    *,
    gradient_exponent: float,
    entropy_coefficient: float = 0.0,
    reference_logits: Tensor | None = None,
    kl_coefficient: float = 0.0,
    reward_correct: float = 1.0,
    reward_incorrect: float = 0.0,
) -> ExpectedOutcomeRLLoss:
    r"""Shape the exact full-information reward gradient without changing its direction.

    Let :math:`p_y` be the probability of the rewarded action and let
    :math:`\Delta r=r_y-r_{\neg y}`.  The policy term is chosen so that

    .. math::

       \left|\frac{\partial L_\gamma}{\partial z_y}\right|
       = |\Delta r| p_y^\gamma(1-p_y), \qquad 0\leq\gamma\leq1.

    ``gradient_exponent=1`` is the ordinary exact expected-reward objective,
    whose gradient vanishes linearly as a policy becomes confidently wrong.
    ``gradient_exponent=0`` is exact log-probability control (cross entropy),
    and an exponent strictly between zero and one is a preregisterable
    intermediate.  With ``reward_correct > reward_incorrect``, every member of
    this family has the same unique probability optimum, :math:`p_y=1`, before
    shared entropy or KL regularization.  This is a full-information mechanism
    intervention, not an information-matched on-policy RL estimator.
    """

    _validate_binary_logits(action_logits)
    if action_logits.ndim != 2:
        raise ValueError("power-outcome action_logits must have shape [prompts, 2]")
    targets = correct_actions.to(device=action_logits.device, dtype=torch.long)
    if targets.shape != action_logits.shape[:-1]:
        raise ValueError("correct_actions must have shape [prompts]")
    if torch.any((targets < 0) | (targets > 1)):
        raise ValueError("correct_actions must contain only 0 (A) or 1 (B)")
    coefficients = (entropy_coefficient, kl_coefficient)
    if any(not math.isfinite(float(value)) or value < 0 for value in coefficients):
        raise ValueError("entropy and KL coefficients must be non-negative")
    exponent = float(gradient_exponent)
    if not math.isfinite(exponent) or not 0.0 <= exponent <= 1.0:
        raise ValueError("gradient_exponent must be finite and lie in [0, 1]")
    if not math.isfinite(float(reward_correct)) or not math.isfinite(float(reward_incorrect)):
        raise ValueError("reward values must be finite")

    log_probabilities = _require_finite(
        action_logits.log_softmax(dim=-1),
        "power-outcome log probabilities",
    )
    probabilities = _require_finite(
        log_probabilities.exp(),
        "power-outcome probabilities",
    )
    correct_log_probabilities = log_probabilities.gather(
        -1,
        targets.unsqueeze(-1),
    ).squeeze(-1)
    correct_probabilities = probabilities.gather(
        -1,
        targets.unsqueeze(-1),
    ).squeeze(-1)
    reward_gap = float(reward_correct) - float(reward_incorrect)
    expected_rewards = _require_finite(
        float(reward_incorrect) + reward_gap * correct_probabilities,
        "expected rewards",
    )

    if exponent == 0.0:
        # This explicit limit remains stable even when exp(log p_y) underflows.
        shaped_returns = (
            float(reward_incorrect) + reward_gap * correct_log_probabilities
        )
    else:
        shaped_returns = float(reward_incorrect) + (
            reward_gap * torch.exp(exponent * correct_log_probabilities) / exponent
        )
    policy_loss = -shaped_returns.mean()
    entropy = -(probabilities * log_probabilities).sum(dim=-1).mean()
    _require_finite(policy_loss, "power-outcome policy loss")
    _require_finite(entropy, "power-outcome entropy")

    if reference_logits is None:
        kl = action_logits.new_zeros(())
    else:
        _validate_binary_logits(reference_logits, "reference_logits")
        if reference_logits.shape != action_logits.shape:
            raise ValueError("reference_logits must have the same shape as action_logits")
        reference_log_probabilities = reference_logits.detach().log_softmax(dim=-1)
        _require_finite(reference_log_probabilities, "reference log probabilities")
        kl = (
            probabilities * (log_probabilities - reference_log_probabilities)
        ).sum(dim=-1).mean()
        _require_finite(kl, "power-outcome KL")

    loss = policy_loss + float(kl_coefficient) * kl - float(entropy_coefficient) * entropy
    _require_finite(loss, "power-outcome loss")
    absolute_reward_gap = abs(reward_gap)
    logprob_gradient_magnitudes = absolute_reward_gap * (1.0 - correct_probabilities)
    expected_reward_gradient_magnitudes = (
        correct_probabilities * logprob_gradient_magnitudes
    )
    active_gradient_magnitudes = (
        torch.exp(exponent * correct_log_probabilities)
        * logprob_gradient_magnitudes
    )
    return ExpectedOutcomeRLLoss(
        loss=loss,
        policy_loss=policy_loss,
        entropy=entropy.detach(),
        kl=kl.detach(),
        mean_reward=expected_rewards.mean().detach(),
        gradient_exponent=exponent,
        mean_correct_probability=correct_probabilities.mean().detach(),
        mean_expected_reward_logit_gradient_magnitude=(
            expected_reward_gradient_magnitudes.mean().detach()
        ),
        mean_logprob_logit_gradient_magnitude=(
            logprob_gradient_magnitudes.mean().detach()
        ),
        mean_policy_logit_gradient_magnitude=(
            active_gradient_magnitudes.mean().detach()
        ),
    )


def sample_binary_actions(
    action_logits: Tensor,
    samples_per_prompt: int,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample independent actions reproducibly from each prompt distribution."""

    _validate_binary_logits(action_logits)
    if action_logits.ndim != 2:
        raise ValueError("action_logits must have shape [prompts, 2]")
    if (
        isinstance(samples_per_prompt, bool)
        or not isinstance(samples_per_prompt, int)
        or samples_per_prompt < 2
    ):
        raise ValueError("samples_per_prompt must be an integer of at least two")
    probabilities = _require_finite(
        action_logits.softmax(dim=-1),
        "sampling probabilities",
    )
    row_sums = probabilities.sum(dim=-1)
    if not bool(
        torch.isclose(
            row_sums,
            torch.ones_like(row_sums),
            rtol=5e-3,
            atol=5e-3,
        ).all()
        .detach()
        .cpu()
    ):
        raise FloatingPointError("sampling probabilities are not normalized")
    return torch.multinomial(
        probabilities,
        num_samples=int(samples_per_prompt),
        replacement=True,
        generator=generator,
    )


def make_outcome_rl_tensors(
    action_logits: Tensor,
    correct_actions: Tensor,
    samples_per_prompt: int,
    *,
    generator: torch.Generator | None = None,
    reward_correct: float = 1.0,
    reward_incorrect: float = 0.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Sample actions and compute exact environment rewards for one-step play."""

    _validate_binary_logits(action_logits)
    if not math.isfinite(float(reward_correct)) or not math.isfinite(float(reward_incorrect)):
        raise ValueError("reward values must be finite")
    if action_logits.ndim != 2:
        raise ValueError("action_logits must have shape [prompts, 2]")
    targets = correct_actions.to(device=action_logits.device, dtype=torch.long)
    if targets.shape != action_logits.shape[:-1]:
        raise ValueError("correct_actions must have shape [prompts]")
    actions = sample_binary_actions(
        action_logits,
        samples_per_prompt,
        generator=generator,
    )
    rewards = torch.where(
        actions == targets[:, None],
        torch.as_tensor(reward_correct, device=action_logits.device, dtype=torch.float32),
        torch.as_tensor(reward_incorrect, device=action_logits.device, dtype=torch.float32),
    )
    expanded_logits = action_logits[:, None, :].expand(-1, int(samples_per_prompt), -1)
    return expanded_logits, actions, rewards


def seeded_generator(seed: int, *, device: str | torch.device = "cpu") -> torch.Generator:
    """Return an isolated RNG for replayable action sampling."""

    generator = torch.Generator(device=torch.device(device))
    generator.manual_seed(int(seed))
    return generator


HookKind = Literal["evaluate", "checkpoint"]
HookCallback = Callable[[int, nn.Module, Mapping[str, Any]], Any]


@dataclass(frozen=True)
class HookEvent:
    """A scheduled side effect and the callback value it produced."""

    kind: HookKind
    step: int
    value: Any


class ScheduledHooks:
    """Emit checkpoint and evaluation callbacks at deterministic steps.

    Calls are idempotent, and evaluation always precedes checkpointing when
    both are scheduled for the same step.  This fixes the meaning of a saved
    checkpoint: it is the exact parameter state that was just evaluated.
    """

    def __init__(
        self,
        *,
        eval_steps: Sequence[int] = (),
        checkpoint_steps: Sequence[int] = (),
        evaluate: HookCallback | None = None,
        checkpoint: HookCallback | None = None,
    ) -> None:
        self.eval_steps = self._validate_steps(eval_steps, "eval_steps")
        self.checkpoint_steps = self._validate_steps(checkpoint_steps, "checkpoint_steps")
        self.evaluate = evaluate
        self.checkpoint = checkpoint
        self._emitted: set[tuple[HookKind, int]] = set()

    @staticmethod
    def _validate_steps(steps: Sequence[int], name: str) -> tuple[int, ...]:
        values = tuple(int(step) for step in steps)
        if any(step < 0 for step in values):
            raise ValueError(f"{name} cannot contain negative steps")
        if values != tuple(sorted(set(values))):
            raise ValueError(f"{name} must be sorted and unique")
        return values

    def emit(
        self,
        step: int,
        model: nn.Module,
        state: Mapping[str, Any] | None = None,
    ) -> tuple[HookEvent, ...]:
        if isinstance(step, bool) or step < 0:
            raise ValueError("step must be a non-negative integer")
        snapshot = state or {}
        events: list[HookEvent] = []
        schedule: tuple[tuple[HookKind, tuple[int, ...], HookCallback | None], ...] = (
            ("evaluate", self.eval_steps, self.evaluate),
            ("checkpoint", self.checkpoint_steps, self.checkpoint),
        )
        for kind, steps, callback in schedule:
            key = (kind, int(step))
            if step in steps and key not in self._emitted:
                value = callback(int(step), model, snapshot) if callback is not None else None
                events.append(HookEvent(kind=kind, step=int(step), value=value))
                self._emitted.add(key)
        return tuple(events)

    @property
    def emitted(self) -> tuple[tuple[HookKind, int], ...]:
        order = {"evaluate": 0, "checkpoint": 1}
        return tuple(sorted(self._emitted, key=lambda item: (item[1], order[item[0]])))

    def state_dict(self) -> dict[str, list[list[str | int]]]:
        """Serialize emitted events so a resumed run does not emit them twice."""

        return {"emitted": [[kind, step] for kind, step in self.emitted]}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        emitted = state.get("emitted", [])
        restored: set[tuple[HookKind, int]] = set()
        if not isinstance(emitted, Sequence):
            raise ValueError("hook state 'emitted' must be a sequence")
        for item in emitted:
            if not isinstance(item, Sequence) or len(item) != 2:
                raise ValueError("each emitted hook entry must contain kind and step")
            kind, step = item
            if kind not in {"evaluate", "checkpoint"}:
                raise ValueError(f"invalid hook kind in saved state: {kind!r}")
            numeric_step = int(step)
            if numeric_step < 0:
                raise ValueError("saved hook steps cannot be negative")
            restored.add((kind, numeric_step))
        self._emitted = restored


@dataclass
class TrainState:
    """Minimal serializable state at an optimizer-step boundary."""

    global_step: int = 0
    micro_step: int = 0
    examples_seen: int = 0
    base_learning_rates: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.global_step < 0 or self.micro_step < 0 or self.examples_seen < 0:
            raise ValueError("training counters cannot be negative")
        if any(not math.isfinite(rate) or rate <= 0 for rate in self.base_learning_rates):
            raise ValueError("base learning rates must be finite and positive")

    def state_dict(self) -> dict[str, Any]:
        return {
            "global_step": self.global_step,
            "micro_step": self.micro_step,
            "examples_seen": self.examples_seen,
            "base_learning_rates": list(self.base_learning_rates),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> TrainState:
        return cls(
            global_step=int(state.get("global_step", 0)),
            micro_step=int(state.get("micro_step", 0)),
            examples_seen=int(state.get("examples_seen", 0)),
            base_learning_rates=tuple(float(value) for value in state.get("base_learning_rates", ())),
        )


@dataclass(frozen=True)
class StepMetrics:
    """Aggregated measurements for one completed optimizer update."""

    step: int
    micro_step: int
    loss: float
    learning_rate: float
    gradient_norm: float
    accuracy: float | None
    mean_reward: float | None
    entropy: float | None
    kl: float | None
    sampled_b_rate: float | None = None
    both_actions_sampled_fraction: float | None = None
    all_zero_loo_advantages_fraction: float | None = None
    loss_geometry: Mapping[str, Any] | None = None
    objective_gradient_diagnostics: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "micro_step": self.micro_step,
            "loss": self.loss,
            "learning_rate": self.learning_rate,
            "gradient_norm": self.gradient_norm,
            "accuracy": self.accuracy,
            "mean_reward": self.mean_reward,
            "entropy": self.entropy,
            "kl": self.kl,
            "sampled_b_rate": self.sampled_b_rate,
            "both_actions_sampled_fraction": self.both_actions_sampled_fraction,
            "all_zero_loo_advantages_fraction": self.all_zero_loo_advantages_fraction,
            "loss_geometry": (
                None if self.loss_geometry is None else dict(self.loss_geometry)
            ),
            "objective_gradient_diagnostics": (
                None
                if self.objective_gradient_diagnostics is None
                else dict(self.objective_gradient_diagnostics)
            ),
        }


@dataclass(frozen=True)
class TrainResult:
    """State, dense metrics, and hook outputs produced by :func:`train_steps`."""

    state: TrainState
    metrics: tuple[StepMetrics, ...]
    hook_events: tuple[HookEvent, ...]


def build_optimizer(
    model: nn.Module,
    *,
    learning_rate: float,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    """Build AdamW over trainable parameters only."""

    if (
        not math.isfinite(float(learning_rate))
        or not math.isfinite(float(weight_decay))
        or learning_rate <= 0
        or weight_decay < 0
    ):
        raise ValueError(
            "learning_rate must be finite and positive; weight_decay finite and non-negative"
        )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("model has no trainable parameters")
    return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)


def constant_with_warmup_multiplier(update_step: int, warmup_steps: int) -> float:
    """Linear warmup followed by a constant learning rate."""

    if update_step < 1:
        raise ValueError("update_step is one-indexed and must be positive")
    if warmup_steps < 0:
        raise ValueError("warmup_steps cannot be negative")
    if warmup_steps == 0:
        return 1.0
    return min(1.0, update_step / warmup_steps)


def _derived_seed(seed: int, index: int, lane: int) -> int:
    # Bounded arithmetic is stable across Python versions and process restarts.
    modulus = 2**63 - 1
    return (int(seed) + 1_000_003 * int(index) + 97_409 * int(lane)) % modulus


def deterministic_batch_indices(
    dataset_size: int,
    batch_size: int,
    micro_step: int,
    *,
    seed: int,
) -> Tensor:
    """Return a replayable shuffled batch determined only by its micro-step."""

    if dataset_size < 1 or batch_size < 1 or micro_step < 0:
        raise ValueError("dataset_size and batch_size must be positive; micro_step non-negative")
    batches_per_epoch = (dataset_size + batch_size - 1) // batch_size
    epoch, batch_in_epoch = divmod(micro_step, batches_per_epoch)
    generator = seeded_generator(_derived_seed(seed, epoch, lane=0))
    permutation = torch.randperm(dataset_size, generator=generator)
    start = batch_in_epoch * batch_size
    return permutation[start : min(start + batch_size, dataset_size)]


@contextmanager
def _step_rng(seed: int, micro_step: int, device: torch.device) -> Iterator[None]:
    """Make dropout replayable without permanently mutating the caller's RNG."""

    cuda_devices: list[int] = []
    if device.type == "cuda":
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(_derived_seed(seed, micro_step, lane=1))
        yield


def _base_learning_rates(
    optimizer: torch.optim.Optimizer,
    state: TrainState,
) -> tuple[float, ...]:
    if state.base_learning_rates:
        if len(state.base_learning_rates) != len(optimizer.param_groups):
            raise ValueError("saved base learning rates do not match optimizer parameter groups")
        return state.base_learning_rates
    rates = tuple(float(group.get("initial_lr", group["lr"])) for group in optimizer.param_groups)
    if any(not math.isfinite(rate) or rate <= 0 for rate in rates):
        raise ValueError("optimizer learning rates must be finite and positive")
    state.base_learning_rates = rates
    return rates


def _set_learning_rate(
    optimizer: torch.optim.Optimizer,
    base_rates: Sequence[float],
    multiplier: float,
) -> float:
    if not math.isfinite(float(multiplier)) or multiplier < 0:
        raise ValueError("learning-rate multiplier must be finite and non-negative")
    for group, base_rate in zip(optimizer.param_groups, base_rates, strict=True):
        group["lr"] = float(base_rate) * multiplier
    return float(base_rates[0]) * multiplier


def _mean_metric(values: Mapping[str, Sequence[float]], name: str) -> float | None:
    observed = values.get(name, ())
    return None if not observed else sum(observed) / len(observed)


_GEOMETRY_PREFIX = "loss_geometry:"


def _append_loss_geometry(
    accumulated: dict[str, list[float]],
    action_logits: Tensor,
    targets: Tensor,
    candidate_choices: Tensor | None,
    *,
    gradient_exponent: float,
) -> None:
    """Record per-example probability/gradient geometry before backpropagation."""

    exponent = float(gradient_exponent)
    if not math.isfinite(exponent) or not 0.0 <= exponent <= 1.0:
        raise ValueError("loss-geometry gradient exponent must lie in [0, 1]")
    log_probabilities = _require_finite(
        action_logits.detach().float().log_softmax(dim=-1),
        "loss-geometry log probabilities",
    )
    correct_log_probabilities = log_probabilities.gather(
        -1,
        targets.unsqueeze(-1),
    ).squeeze(-1)
    correct_probabilities = _require_finite(
        correct_log_probabilities.exp(),
        "loss-geometry correct probabilities",
    )
    incorrect_targets = 1 - targets
    incorrect_log_probabilities = log_probabilities.gather(
        -1,
        incorrect_targets.unsqueeze(-1),
    ).squeeze(-1)
    correct_margins = _require_finite(
        correct_log_probabilities - incorrect_log_probabilities,
        "loss-geometry correct margins",
    )
    logprob_gradients = 1.0 - correct_probabilities
    expected_reward_gradients = correct_probabilities * logprob_gradients
    active_gradients = (
        torch.exp(exponent * correct_log_probabilities) * logprob_gradients
    )

    strata: dict[str, Tensor] = {
        "all": torch.ones_like(targets, dtype=torch.bool),
    }
    if candidate_choices is not None:
        y = candidate_choices[:, 0]
        p = candidate_choices[:, 1]
        q = candidate_choices[:, 2]
        strata.update(
            {
                "herald_agreement": p == y,
                "herald_conflict": p != y,
                "sage_agreement": q == y,
                "sage_conflict": q != y,
                "joint_proxy_conflict": (p != y) & (q != y),
                "either_proxy_conflict": (p != y) | (q != y),
            }
        )

    per_example = {
        "correct_probability": correct_probabilities,
        "correct_log_probability": correct_log_probabilities,
        "correct_logit_margin": correct_margins,
        "expected_reward_gradient_magnitude": expected_reward_gradients,
        "logprob_gradient_magnitude": logprob_gradients,
        "active_policy_gradient_magnitude": active_gradients,
        "correct_probability_below_0_01": (correct_probabilities < 0.01).float(),
        "correct_probability_below_0_10": (correct_probabilities < 0.10).float(),
        "correct_probability_above_0_90": (correct_probabilities > 0.90).float(),
        "correct_probability_above_0_99": (correct_probabilities > 0.99).float(),
    }
    for stratum, mask in strata.items():
        if not bool(mask.any()):
            continue
        for metric, values in per_example.items():
            accumulated[f"{_GEOMETRY_PREFIX}{stratum}:{metric}"].extend(
                values[mask].cpu().tolist()
            )


def _summarize_loss_geometry(
    accumulated: Mapping[str, Sequence[float]],
    *,
    gradient_exponent: float,
) -> dict[str, Any]:
    """Convert per-example geometry streams into one auditable update record."""

    metric_names = {
        key.removeprefix(_GEOMETRY_PREFIX).split(":", maxsplit=1)[1]
        for key in accumulated
        if key.startswith(_GEOMETRY_PREFIX)
    }
    stratum_names = {
        key.removeprefix(_GEOMETRY_PREFIX).split(":", maxsplit=1)[0]
        for key in accumulated
        if key.startswith(_GEOMETRY_PREFIX)
    }
    strata: dict[str, dict[str, Any]] = {}
    for stratum in sorted(stratum_names):
        probability_key = f"{_GEOMETRY_PREFIX}{stratum}:correct_probability"
        probabilities = accumulated.get(probability_key, ())
        if not probabilities:
            continue
        summary: dict[str, Any] = {"n": len(probabilities)}
        for metric in sorted(metric_names):
            values = accumulated.get(f"{_GEOMETRY_PREFIX}{stratum}:{metric}", ())
            if not values:
                continue
            if metric == "correct_probability":
                summary["minimum_correct_probability"] = min(values)
            if metric.startswith("correct_probability_below_") or metric.startswith(
                "correct_probability_above_"
            ):
                summary[f"{metric}_fraction"] = sum(values) / len(values)
            else:
                summary[f"mean_{metric}"] = sum(values) / len(values)
        exact = float(summary["mean_expected_reward_gradient_magnitude"])
        active = float(summary["mean_active_policy_gradient_magnitude"])
        summary["active_to_expected_gradient_ratio"] = (
            active / exact if exact > 0.0 else None
        )
        strata[stratum] = summary
    return {
        "schema_version": 1,
        "gradient_exponent": float(gradient_exponent),
        "gradient_definition": "abs(d policy_loss / d correct_action_logit)",
        "strata": strata,
    }


def train_steps(
    scorer: nn.Module,
    optimizer: torch.optim.Optimizer,
    prompts: Sequence[str],
    correct_actions: Sequence[int] | Tensor,
    *,
    total_steps: int,
    batch_size: int,
    algorithm: Literal[
        "sft",
        "trajectory_sft",
        "outcome_rl",
        "expected_outcome_rl",
        "tempered_outcome_control",
        "logprob_outcome_control",
    ] = "sft",
    gradient_accumulation_steps: int = 1,
    warmup_steps: int = 0,
    max_grad_norm: float = 1.0,
    parameter_finite_check_interval: int = 1,
    samples_per_prompt: int = 4,
    entropy_coefficient: float = 0.0,
    kl_coefficient: float = 0.0,
    reference_scorer: nn.Module | None = None,
    reward_gradient_exponent: float | None = None,
    candidate_choices: Sequence[Sequence[int]] | Tensor | None = None,
    seed: int = 0,
    action_sampling_seed: int | None = None,
    state: TrainState | None = None,
    hooks: ScheduledHooks | None = None,
) -> TrainResult:
    """Run a deterministic SFT or one-step outcome-RL optimizer loop.

    ``total_steps`` is the final optimizer step, not an increment.  Passing a
    restored model, optimizer, :class:`TrainState`, and hook state resumes at
    the next update.  Batches, dropout seeds, and RL samples are functions of
    the saved micro-step, so a completed-step checkpoint has no hidden data
    loader cursor. ``action_sampling_seed`` can isolate policy sampling from
    data-order randomness. Each accumulated micro-batch is scored and
    backpropagated separately, avoiding reuse of an already-freed computation
    graph.
    """

    if (
        total_steps < 0
        or batch_size < 1
        or gradient_accumulation_steps < 1
        or parameter_finite_check_interval < 0
    ):
        raise ValueError("total_steps must be non-negative and batch sizes positive")
    if (
        warmup_steps < 0
        or not math.isfinite(float(max_grad_norm))
        or max_grad_norm <= 0
    ):
        raise ValueError(
            "warmup_steps must be non-negative and max_grad_norm finite and positive"
        )
    if algorithm not in {
        "sft",
        "trajectory_sft",
        "outcome_rl",
        "expected_outcome_rl",
        "tempered_outcome_control",
        "logprob_outcome_control",
    }:
        raise ValueError(f"unsupported training algorithm: {algorithm!r}")
    if not prompts or len(prompts) != len(correct_actions):
        raise ValueError("prompts and correct_actions must have the same non-zero length")
    targets_cpu = torch.as_tensor(correct_actions, dtype=torch.long, device="cpu")
    if targets_cpu.ndim != 1 or torch.any((targets_cpu < 0) | (targets_cpu > 1)):
        raise ValueError("correct_actions must be a one-dimensional sequence of 0/1 choices")
    candidate_choices_cpu: Tensor | None = None
    if candidate_choices is not None:
        candidate_choices_cpu = torch.as_tensor(
            candidate_choices,
            dtype=torch.long,
            device="cpu",
        )
        if candidate_choices_cpu.shape != (len(prompts), 3):
            raise ValueError("candidate_choices must have shape [examples, 3] for Y/P/Q")
        if torch.any((candidate_choices_cpu < 0) | (candidate_choices_cpu > 1)):
            raise ValueError("candidate_choices must contain only binary A/B choices")
        if not torch.equal(candidate_choices_cpu[:, 0], targets_cpu):
            raise ValueError("candidate_choices Y column must equal correct_actions")

    if algorithm == "tempered_outcome_control":
        if reward_gradient_exponent is None:
            raise ValueError("tempered outcome control requires reward_gradient_exponent")
        active_gradient_exponent = float(reward_gradient_exponent)
        if not math.isfinite(active_gradient_exponent) or not 0.0 < active_gradient_exponent < 1.0:
            raise ValueError("tempered reward_gradient_exponent must lie strictly between 0 and 1")
    elif algorithm == "logprob_outcome_control":
        if reward_gradient_exponent is None or float(reward_gradient_exponent) != 0.0:
            raise ValueError("logprob outcome control requires reward_gradient_exponent=0")
        active_gradient_exponent = 0.0
    else:
        if reward_gradient_exponent is not None:
            raise ValueError(
                "reward_gradient_exponent is only valid for tempered or logprob outcome control"
            )
        active_gradient_exponent = (
            0.0 if algorithm in {"sft", "trajectory_sft"} else 1.0
        )
    current = state if state is not None else TrainState()
    if current.global_step > total_steps:
        raise ValueError("restored global_step exceeds total_steps")
    expected_micro_step = current.global_step * gradient_accumulation_steps
    if current.micro_step != expected_micro_step:
        raise ValueError(
            "resume state must be at a completed optimizer boundary with "
            "micro_step == global_step * gradient_accumulation_steps"
        )
    if algorithm == "outcome_rl" and samples_per_prompt < 2:
        raise ValueError("outcome RL requires at least two samples per prompt")
    if reference_scorer is None and kl_coefficient > 0:
        raise ValueError("positive kl_coefficient requires reference_scorer")

    base_rates = _base_learning_rates(optimizer, current)
    sampling_seed = int(seed if action_sampling_seed is None else action_sampling_seed)
    trainable_parameters = [parameter for parameter in scorer.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("scorer has no trainable parameters")
    previous_reference_mode = None
    if reference_scorer is not None:
        previous_reference_mode = reference_scorer.training
        reference_scorer.eval()

    metrics: list[StepMetrics] = []
    hook_events: list[HookEvent] = []
    if hooks is not None:
        hook_events.extend(hooks.emit(current.global_step, scorer, {"train_state": current.state_dict()}))

    scorer.train()
    try:
        while current.global_step < total_steps:
            next_step = current.global_step + 1
            learning_rate = _set_learning_rate(
                optimizer,
                base_rates,
                constant_with_warmup_multiplier(next_step, warmup_steps),
            )
            optimizer.zero_grad(set_to_none=True)
            accumulated: dict[str, list[float]] = defaultdict(list)
            for _accumulation_index in range(gradient_accumulation_steps):
                indices = deterministic_batch_indices(
                    len(prompts),
                    batch_size,
                    current.micro_step,
                    seed=seed,
                )
                batch_prompts = [prompts[int(index)] for index in indices]
                batch_targets_cpu = targets_cpu[indices]
                batch_candidate_choices = (
                    None
                    if candidate_choices_cpu is None
                    else candidate_choices_cpu[indices]
                )
                with _step_rng(seed, current.micro_step, _model_device(scorer)):
                    action_logits = scorer(batch_prompts)
                if not isinstance(action_logits, Tensor):
                    raise TypeError("scorer must return a Tensor with shape [batch, 2]")
                batch_targets = batch_targets_cpu.to(action_logits.device)
                batch_candidates = (
                    None
                    if batch_candidate_choices is None
                    else batch_candidate_choices.to(action_logits.device)
                )
                _append_loss_geometry(
                    accumulated,
                    action_logits,
                    batch_targets,
                    batch_candidates,
                    gradient_exponent=active_gradient_exponent,
                )

                if algorithm in {"sft", "trajectory_sft"}:
                    objective = sft_action_loss(action_logits, batch_targets)
                    loss = objective.loss
                    accumulated["accuracy"].append(float(objective.accuracy.cpu()))
                elif algorithm == "outcome_rl":
                    sample_generator = seeded_generator(
                        _derived_seed(sampling_seed, current.micro_step, lane=2),
                        device=action_logits.device,
                    )
                    expanded, sampled, rewards = make_outcome_rl_tensors(
                        action_logits,
                        batch_targets,
                        samples_per_prompt,
                        generator=sample_generator,
                    )
                    reference_logits = None
                    if reference_scorer is not None:
                        with (
                            torch.no_grad(),
                            _step_rng(
                                seed,
                                current.micro_step,
                                _model_device(reference_scorer),
                            ),
                        ):
                            reference_base = reference_scorer(batch_prompts)
                        if not isinstance(reference_base, Tensor):
                            raise TypeError("reference_scorer must return a Tensor")
                        reference_logits = reference_base.to(action_logits.device)[:, None, :].expand_as(
                            expanded
                        )
                    objective_rl = outcome_rl_loss(
                        expanded,
                        sampled,
                        rewards,
                        entropy_coefficient=entropy_coefficient,
                        reference_logits=reference_logits,
                        kl_coefficient=kl_coefficient,
                    )
                    loss = objective_rl.loss
                    accumulated["mean_reward"].append(float(objective_rl.mean_reward.cpu()))
                    accumulated["entropy"].append(float(objective_rl.entropy.cpu()))
                    accumulated["kl"].append(float(objective_rl.kl.cpu()))
                    accumulated["sampled_b"].extend(
                        (sampled == 1).float().detach().cpu().flatten().tolist()
                    )
                    accumulated["both_actions_sampled"].extend(
                        (
                            (sampled == 0).any(dim=1)
                            & (sampled == 1).any(dim=1)
                        )
                        .float()
                        .detach()
                        .cpu()
                        .tolist()
                    )
                    accumulated["all_zero_loo_advantages"].extend(
                        (objective_rl.advantages == 0)
                        .all(dim=1)
                        .float()
                        .detach()
                        .cpu()
                        .tolist()
                    )
                elif algorithm in {
                    "expected_outcome_rl",
                    "tempered_outcome_control",
                    "logprob_outcome_control",
                }:
                    reference_logits = None
                    if reference_scorer is not None:
                        with (
                            torch.no_grad(),
                            _step_rng(
                                seed,
                                current.micro_step,
                                _model_device(reference_scorer),
                            ),
                        ):
                            reference_logits = reference_scorer(batch_prompts)
                        if not isinstance(reference_logits, Tensor):
                            raise TypeError("reference_scorer must return a Tensor")
                        reference_logits = reference_logits.to(action_logits.device)
                    if algorithm == "expected_outcome_rl":
                        objective_expected = expected_outcome_rl_loss(
                            action_logits,
                            batch_targets,
                            entropy_coefficient=entropy_coefficient,
                            reference_logits=reference_logits,
                            kl_coefficient=kl_coefficient,
                        )
                    else:
                        objective_expected = power_outcome_control_loss(
                            action_logits,
                            batch_targets,
                            gradient_exponent=active_gradient_exponent,
                            entropy_coefficient=entropy_coefficient,
                            reference_logits=reference_logits,
                            kl_coefficient=kl_coefficient,
                        )
                    loss = objective_expected.loss
                    accumulated["mean_reward"].append(
                        float(objective_expected.mean_reward.cpu())
                    )
                    accumulated["entropy"].append(float(objective_expected.entropy.cpu()))
                    accumulated["kl"].append(float(objective_expected.kl.cpu()))
                    accumulated["objective_correct_probability"].append(
                        float(objective_expected.mean_correct_probability.cpu())
                    )
                    accumulated["objective_expected_reward_gradient_magnitude"].append(
                        float(
                            objective_expected.mean_expected_reward_logit_gradient_magnitude.cpu()
                        )
                    )
                    accumulated["objective_logprob_gradient_magnitude"].append(
                        float(objective_expected.mean_logprob_logit_gradient_magnitude.cpu())
                    )
                    accumulated["objective_policy_gradient_magnitude"].append(
                        float(objective_expected.mean_policy_logit_gradient_magnitude.cpu())
                    )
                else:  # pragma: no cover - guarded before optimization begins
                    raise AssertionError(f"unhandled training algorithm: {algorithm}")

                _require_finite(loss, "training loss")
                (loss / gradient_accumulation_steps).backward()
                accumulated["loss"].append(float(loss.detach().cpu()))
                current.micro_step += 1
                current.examples_seen += len(batch_prompts)

            detailed_finite_check = (
                parameter_finite_check_interval > 0
                and next_step % parameter_finite_check_interval == 0
            )
            if detailed_finite_check:
                for parameter_index, parameter in enumerate(trainable_parameters):
                    if parameter.grad is not None:
                        _require_finite(
                            parameter.grad,
                            f"gradient for trainable parameter {parameter_index}",
                        )
            gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                max_norm=max_grad_norm,
                error_if_nonfinite=True,
            )
            _require_finite(gradient_norm_tensor, "gradient norm")
            optimizer.step()
            if detailed_finite_check:
                for parameter_index, parameter in enumerate(trainable_parameters):
                    _require_finite(
                        parameter,
                        f"trainable parameter {parameter_index} after optimizer step",
                    )
            current.global_step = next_step

            loss_value = _mean_metric(accumulated, "loss")
            assert loss_value is not None
            step_metrics = StepMetrics(
                step=current.global_step,
                micro_step=current.micro_step,
                loss=loss_value,
                learning_rate=learning_rate,
                gradient_norm=float(gradient_norm_tensor.detach().cpu()),
                accuracy=_mean_metric(accumulated, "accuracy"),
                mean_reward=_mean_metric(accumulated, "mean_reward"),
                entropy=_mean_metric(accumulated, "entropy"),
                kl=_mean_metric(accumulated, "kl"),
                sampled_b_rate=_mean_metric(accumulated, "sampled_b"),
                both_actions_sampled_fraction=_mean_metric(
                    accumulated,
                    "both_actions_sampled",
                ),
                all_zero_loo_advantages_fraction=_mean_metric(
                    accumulated,
                    "all_zero_loo_advantages",
                ),
                loss_geometry=_summarize_loss_geometry(
                    accumulated,
                    gradient_exponent=active_gradient_exponent,
                ),
                objective_gradient_diagnostics=(
                    {
                        "schema_version": 1,
                        "gradient_exponent": active_gradient_exponent,
                        "mean_correct_probability": _mean_metric(
                            accumulated,
                            "objective_correct_probability",
                        ),
                        "mean_expected_reward_gradient_magnitude": _mean_metric(
                            accumulated,
                            "objective_expected_reward_gradient_magnitude",
                        ),
                        "mean_logprob_gradient_magnitude": _mean_metric(
                            accumulated,
                            "objective_logprob_gradient_magnitude",
                        ),
                        "mean_policy_gradient_magnitude": _mean_metric(
                            accumulated,
                            "objective_policy_gradient_magnitude",
                        ),
                    }
                    if algorithm
                    in {
                        "expected_outcome_rl",
                        "tempered_outcome_control",
                        "logprob_outcome_control",
                    }
                    else None
                ),
            )
            metrics.append(step_metrics)
            if hooks is not None:
                hook_events.extend(
                    hooks.emit(
                        current.global_step,
                        scorer,
                        {
                            "train_state": current.state_dict(),
                            "step_metrics": step_metrics.as_dict(),
                        },
                    )
                )
                scorer.train()
    finally:
        if reference_scorer is not None and previous_reference_mode is not None:
            reference_scorer.train(previous_reference_mode)

    return TrainResult(
        state=current,
        metrics=tuple(metrics),
        hook_events=tuple(hook_events),
    )


def _model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")
