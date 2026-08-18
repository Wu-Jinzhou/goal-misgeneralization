from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from goalzendo.training import (
    ScheduledHooks,
    TrainState,
    build_optimizer,
    constant_with_warmup_multiplier,
    deterministic_batch_indices,
    expected_outcome_rl_loss,
    leave_one_out_baselines,
    make_outcome_rl_tensors,
    outcome_rl_loss,
    power_outcome_control_loss,
    sample_binary_actions,
    seeded_generator,
    sft_action_loss,
    train_steps,
)


def test_sft_loss_is_exact_binary_action_cross_entropy() -> None:
    logits = torch.tensor([[2.0, -1.0], [-0.5, 1.5]], requires_grad=True)
    targets = torch.tensor([0, 1])
    result = sft_action_loss(logits, targets)
    assert torch.allclose(result.loss, F.cross_entropy(logits, targets))
    assert result.accuracy.item() == 1.0
    result.loss.backward()
    assert logits.grad is not None


def test_leave_one_out_reinforce_components_are_hand_checkable() -> None:
    base_logits = torch.tensor([[1.2, -0.4], [-0.3, 0.8]], requires_grad=True)
    logits = base_logits[:, None, :].expand(-1, 2, -1)
    actions = torch.tensor([[0, 1], [1, 0]])
    rewards = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    result = outcome_rl_loss(logits, actions, rewards, entropy_coefficient=0.1)
    assert torch.equal(result.baselines, torch.tensor([[0.0, 1.0], [0.0, 1.0]]))
    assert torch.equal(result.advantages, torch.tensor([[1.0, -1.0], [1.0, -1.0]]))
    expected_policy = -(
        result.advantages * logits.log_softmax(dim=-1).gather(-1, actions[..., None]).squeeze(-1)
    ).mean()
    assert torch.allclose(result.policy_loss, expected_policy)
    assert torch.allclose(result.loss, result.policy_loss - 0.1 * result.entropy)
    result.loss.backward()
    assert base_logits.grad is not None
    assert torch.count_nonzero(base_logits.grad) > 0


def test_outcome_rl_exact_kl_and_shape_validation() -> None:
    logits = torch.tensor([[[2.0, 0.0], [2.0, 0.0]]], requires_grad=True)
    reference = torch.zeros_like(logits)
    actions = torch.tensor([[0, 1]])
    rewards = torch.tensor([[1.0, 0.0]])
    result = outcome_rl_loss(
        logits,
        actions,
        rewards,
        reference_logits=reference,
        kl_coefficient=0.5,
    )
    assert result.kl.item() > 0
    assert torch.allclose(result.loss, result.policy_loss + 0.5 * result.kl)
    with pytest.raises(ValueError, match="at least two"):
        leave_one_out_baselines(torch.ones(3, 1))


def test_expected_outcome_rl_is_the_exact_two_action_reward_gradient() -> None:
    logits = torch.tensor([[1.2, -0.4], [-0.3, 0.8]], requires_grad=True)
    targets = torch.tensor([0, 1])
    reference = torch.zeros_like(logits)
    result = expected_outcome_rl_loss(
        logits,
        targets,
        entropy_coefficient=0.1,
        reference_logits=reference,
        kl_coefficient=0.2,
    )
    probabilities = logits.softmax(dim=-1)
    exact_reward = probabilities.gather(-1, targets[:, None]).squeeze(-1).mean()
    exact_entropy = -(probabilities * logits.log_softmax(dim=-1)).sum(dim=-1).mean()
    reference_log_probabilities = reference.log_softmax(dim=-1)
    exact_kl = (
        probabilities * (logits.log_softmax(dim=-1) - reference_log_probabilities)
    ).sum(dim=-1).mean()
    expected_loss = -exact_reward + 0.2 * exact_kl - 0.1 * exact_entropy
    assert torch.allclose(result.mean_reward, exact_reward.detach())
    assert torch.allclose(result.policy_loss, -exact_reward)
    assert torch.allclose(result.loss, expected_loss)

    expected_gradient = torch.autograd.grad(expected_loss, logits, retain_graph=True)[0]
    observed_gradient = torch.autograd.grad(result.loss, logits)[0]
    assert torch.allclose(observed_gradient, expected_gradient, rtol=1e-6, atol=1e-7)


def test_expected_outcome_rl_rejects_invalid_rewards_and_targets() -> None:
    with pytest.raises(ValueError, match="only 0"):
        expected_outcome_rl_loss(torch.zeros(1, 2), torch.tensor([2]))
    with pytest.raises(ValueError, match="finite"):
        expected_outcome_rl_loss(
            torch.zeros(1, 2),
            torch.tensor([0]),
            reward_correct=float("nan"),
        )


def test_power_outcome_control_interpolates_the_registered_gradient_shapes() -> None:
    logits = torch.tensor(
        [[-8.0, 8.0], [0.3, -0.2]],
        dtype=torch.float64,
        requires_grad=True,
    )
    targets = torch.tensor([0, 0])
    correct_probabilities = logits.softmax(dim=-1).gather(-1, targets[:, None]).squeeze(-1)
    logprob_gradient = 1.0 - correct_probabilities

    observed: dict[float, torch.Tensor] = {}
    for exponent in (1.0, 0.5, 0.0):
        result = power_outcome_control_loss(
            logits,
            targets,
            gradient_exponent=exponent,
        )
        gradient = torch.autograd.grad(result.loss, logits, retain_graph=True)[0]
        # The batch mean contributes a common 1 / n factor.
        observed[exponent] = gradient[:, 0].abs() * len(targets)
        expected = correct_probabilities.pow(exponent) * logprob_gradient
        assert torch.allclose(observed[exponent], expected, rtol=1e-9, atol=1e-12)
        assert result.gradient_exponent == exponent
        assert torch.allclose(
            result.mean_policy_logit_gradient_magnitude,
            expected.mean().detach(),
        )

    assert torch.allclose(
        observed[1.0] / observed[0.0],
        correct_probabilities,
        rtol=1e-9,
        atol=1e-12,
    )
    assert torch.allclose(
        observed[0.5] / observed[0.0],
        correct_probabilities.sqrt(),
        rtol=1e-9,
        atol=1e-12,
    )
    # The confidently wrong example is effectively ignored by exact reward
    # but receives an order-one log-probability correction.
    assert observed[1.0][0] < 1e-6
    assert observed[0.0][0] > 0.99


def test_power_outcome_control_endpoints_match_expected_reward_and_sft() -> None:
    logits = torch.tensor([[1.2, -0.4], [-0.3, 0.8]], requires_grad=True)
    targets = torch.tensor([0, 1])
    exact = expected_outcome_rl_loss(logits, targets)
    power_exact = power_outcome_control_loss(
        logits,
        targets,
        gradient_exponent=1.0,
    )
    logprob = power_outcome_control_loss(
        logits,
        targets,
        gradient_exponent=0.0,
    )
    supervised = sft_action_loss(logits, targets)
    assert torch.allclose(power_exact.loss, exact.loss)
    assert torch.allclose(logprob.loss, supervised.loss)
    assert torch.allclose(
        torch.autograd.grad(logprob.loss, logits, retain_graph=True)[0],
        torch.autograd.grad(supervised.loss, logits)[0],
    )


@pytest.mark.parametrize("exponent", [-0.1, 1.1, float("nan")])
def test_power_outcome_control_rejects_invalid_gradient_exponents(exponent: float) -> None:
    with pytest.raises(ValueError, match="gradient_exponent"):
        power_outcome_control_loss(
            torch.zeros(1, 2),
            torch.tensor([0]),
            gradient_exponent=exponent,
        )


def test_objectives_and_sampling_reject_non_finite_values() -> None:
    with pytest.raises(FloatingPointError, match="action_logits"):
        sft_action_loss(torch.tensor([[0.0, torch.nan]]), torch.tensor([0]))
    with pytest.raises(FloatingPointError, match="rewards"):
        leave_one_out_baselines(torch.tensor([[1.0, torch.inf]]))
    with pytest.raises(FloatingPointError, match="action_logits"):
        sample_binary_actions(torch.tensor([[0.0, torch.inf]]), 2)
    with pytest.raises(FloatingPointError, match="reference_logits"):
        outcome_rl_loss(
            torch.zeros(1, 2, 2),
            torch.tensor([[0, 1]]),
            torch.tensor([[1.0, 0.0]]),
            reference_logits=torch.tensor([[[0.0, torch.nan], [0.0, 0.0]]]),
        )


def test_sampling_and_reward_construction_are_replayable() -> None:
    logits = torch.tensor([[0.1, 0.2], [1.0, -1.0]])
    first = sample_binary_actions(logits, 8, generator=seeded_generator(42))
    second = sample_binary_actions(logits, 8, generator=seeded_generator(42))
    assert torch.equal(first, second)

    expanded, actions, rewards = make_outcome_rl_tensors(
        logits,
        torch.tensor([1, 0]),
        8,
        generator=seeded_generator(9),
    )
    assert expanded.shape == (2, 8, 2)
    assert actions.shape == rewards.shape == (2, 8)
    assert torch.equal(rewards, (actions == torch.tensor([[1], [0]])).float())


def test_scheduled_hooks_have_fixed_order_and_are_idempotent() -> None:
    calls: list[tuple[str, int]] = []

    def evaluate(step: int, model: nn.Module, state: object) -> str:
        del model, state
        calls.append(("evaluate", step))
        return "metrics"

    def checkpoint(step: int, model: nn.Module, state: object) -> str:
        del model, state
        calls.append(("checkpoint", step))
        return "path"

    hooks = ScheduledHooks(
        eval_steps=(0, 2),
        checkpoint_steps=(2,),
        evaluate=evaluate,
        checkpoint=checkpoint,
    )
    model = nn.Linear(1, 1)
    assert [event.kind for event in hooks.emit(0, model)] == ["evaluate"]
    assert [event.kind for event in hooks.emit(2, model)] == ["evaluate", "checkpoint"]
    assert hooks.emit(2, model) == ()
    assert calls == [("evaluate", 0), ("evaluate", 2), ("checkpoint", 2)]
    assert hooks.emitted == (("evaluate", 0), ("evaluate", 2), ("checkpoint", 2))

    restored = ScheduledHooks(eval_steps=(0, 2), checkpoint_steps=(2,))
    restored.load_state_dict(hooks.state_dict())
    assert restored.emit(2, model) == ()


class LookupScorer(nn.Module):
    def __init__(self, initial: torch.Tensor) -> None:
        super().__init__()
        self.logits = nn.Parameter(initial.clone())

    def forward(self, prompts: list[str]) -> torch.Tensor:
        indices = torch.tensor([int(prompt.removeprefix("item-")) for prompt in prompts])
        return self.logits[indices]


def _make_scorer() -> LookupScorer:
    return LookupScorer(
        torch.tensor(
            [
                [0.2, -0.1],
                [-0.3, 0.1],
                [0.4, -0.2],
                [-0.1, 0.3],
                [0.1, -0.4],
            ]
        )
    )


def test_deterministic_batches_and_warmup_schedule() -> None:
    assert constant_with_warmup_multiplier(1, 4) == 0.25
    assert constant_with_warmup_multiplier(4, 4) == 1.0
    assert constant_with_warmup_multiplier(7, 4) == 1.0
    first = deterministic_batch_indices(5, 2, 3, seed=17)
    second = deterministic_batch_indices(5, 2, 3, seed=17)
    assert torch.equal(first, second)
    with pytest.raises(ValueError, match="finite and positive"):
        build_optimizer(_make_scorer(), learning_rate=float("nan"))
    with pytest.raises(ValueError, match="finite and positive"):
        TrainState(base_learning_rates=(float("inf"),))


def test_sft_loop_supports_accumulation_hooks_and_exact_resume() -> None:
    prompts = [f"item-{index}" for index in range(5)]
    targets = [0, 1, 0, 1, 1]

    uninterrupted = _make_scorer()
    uninterrupted_optimizer = build_optimizer(uninterrupted, learning_rate=0.1)
    full = train_steps(
        uninterrupted,
        uninterrupted_optimizer,
        prompts,
        targets,
        total_steps=4,
        batch_size=2,
        gradient_accumulation_steps=2,
        warmup_steps=2,
        seed=81,
    )

    resumed = _make_scorer()
    resumed_optimizer = build_optimizer(resumed, learning_rate=0.1)
    hooks = ScheduledHooks(eval_steps=(0, 1, 2, 3, 4), checkpoint_steps=(2, 4))
    first_half = train_steps(
        resumed,
        resumed_optimizer,
        prompts,
        targets,
        total_steps=2,
        batch_size=2,
        gradient_accumulation_steps=2,
        warmup_steps=2,
        seed=81,
        hooks=hooks,
    )
    saved_state = TrainState.from_state_dict(first_half.state.state_dict())
    hook_state = hooks.state_dict()
    resumed_hooks = ScheduledHooks(eval_steps=(0, 1, 2, 3, 4), checkpoint_steps=(2, 4))
    resumed_hooks.load_state_dict(hook_state)
    second_half = train_steps(
        resumed,
        resumed_optimizer,
        prompts,
        targets,
        total_steps=4,
        batch_size=2,
        gradient_accumulation_steps=2,
        warmup_steps=2,
        seed=81,
        state=saved_state,
        hooks=resumed_hooks,
    )
    assert second_half.state.global_step == 4
    assert second_half.state.micro_step == 8
    assert second_half.state.examples_seen == full.state.examples_seen
    assert torch.equal(resumed.logits, uninterrupted.logits)
    assert [metric.learning_rate for metric in (*first_half.metrics, *second_half.metrics)] == [
        0.05,
        0.1,
        0.1,
        0.1,
    ]
    assert [event.step for event in second_half.hook_events] == [3, 4, 4]
    for metric in (*first_half.metrics, *second_half.metrics):
        assert metric.sampled_b_rate is None
        assert metric.both_actions_sampled_fraction is None
        assert metric.all_zero_loo_advantages_fraction is None


def test_outcome_rl_loop_runs_fresh_graphs_with_accumulation_and_reference_kl() -> None:
    scorer = _make_scorer()
    reference = _make_scorer()
    reference.requires_grad_(False)
    optimizer = build_optimizer(scorer, learning_rate=0.03)
    result = train_steps(
        scorer,
        optimizer,
        [f"item-{index}" for index in range(5)],
        [0, 1, 0, 1, 1],
        total_steps=3,
        batch_size=3,
        algorithm="outcome_rl",
        gradient_accumulation_steps=2,
        samples_per_prompt=4,
        entropy_coefficient=0.01,
        kl_coefficient=0.1,
        reference_scorer=reference,
        seed=93,
    )
    assert result.state.global_step == 3
    assert len(result.metrics) == 3
    assert all(metric.mean_reward is not None for metric in result.metrics)
    assert all(metric.entropy is not None for metric in result.metrics)
    assert all(metric.kl is not None and metric.kl >= 0 for metric in result.metrics)
    for metric in result.metrics:
        assert metric.sampled_b_rate is not None
        assert 0.0 <= metric.sampled_b_rate <= 1.0
        assert metric.both_actions_sampled_fraction is not None
        assert metric.all_zero_loo_advantages_fraction is not None
        assert metric.both_actions_sampled_fraction + metric.all_zero_loo_advantages_fraction == (
            pytest.approx(1.0)
        )


def test_outcome_rl_sampling_collapse_diagnostics_resume_exactly() -> None:
    prompts = [f"item-{index}" for index in range(5)]
    targets = [0, 1, 0, 1, 1]
    kwargs = {
        "batch_size": 3,
        "algorithm": "outcome_rl",
        "gradient_accumulation_steps": 2,
        "samples_per_prompt": 5,
        "seed": 731,
        "action_sampling_seed": 991,
    }

    uninterrupted = _make_scorer()
    uninterrupted_optimizer = build_optimizer(uninterrupted, learning_rate=0.03)
    full = train_steps(
        uninterrupted,
        uninterrupted_optimizer,
        prompts,
        targets,
        total_steps=4,
        **kwargs,  # type: ignore[arg-type]
    )

    resumed = _make_scorer()
    resumed_optimizer = build_optimizer(resumed, learning_rate=0.03)
    first = train_steps(
        resumed,
        resumed_optimizer,
        prompts,
        targets,
        total_steps=2,
        **kwargs,  # type: ignore[arg-type]
    )
    second = train_steps(
        resumed,
        resumed_optimizer,
        prompts,
        targets,
        total_steps=4,
        state=TrainState.from_state_dict(first.state.state_dict()),
        **kwargs,  # type: ignore[arg-type]
    )

    assert torch.equal(resumed.logits, uninterrupted.logits)
    assert [metric.as_dict() for metric in (*first.metrics, *second.metrics)] == [
        metric.as_dict() for metric in full.metrics
    ]


def test_expected_outcome_rl_loop_resumes_exactly_without_sampling_diagnostics() -> None:
    prompts = [f"item-{index}" for index in range(5)]
    targets = [0, 1, 0, 1, 1]
    kwargs = {
        "batch_size": 3,
        "algorithm": "expected_outcome_rl",
        "gradient_accumulation_steps": 2,
        "entropy_coefficient": 0.01,
        "seed": 811,
    }

    uninterrupted = _make_scorer()
    uninterrupted_optimizer = build_optimizer(uninterrupted, learning_rate=0.03)
    full = train_steps(
        uninterrupted,
        uninterrupted_optimizer,
        prompts,
        targets,
        total_steps=4,
        **kwargs,  # type: ignore[arg-type]
    )

    resumed = _make_scorer()
    resumed_optimizer = build_optimizer(resumed, learning_rate=0.03)
    first = train_steps(
        resumed,
        resumed_optimizer,
        prompts,
        targets,
        total_steps=2,
        **kwargs,  # type: ignore[arg-type]
    )
    second = train_steps(
        resumed,
        resumed_optimizer,
        prompts,
        targets,
        total_steps=4,
        state=TrainState.from_state_dict(first.state.state_dict()),
        **kwargs,  # type: ignore[arg-type]
    )

    assert torch.equal(resumed.logits, uninterrupted.logits)
    assert [metric.as_dict() for metric in (*first.metrics, *second.metrics)] == [
        metric.as_dict() for metric in full.metrics
    ]
    for metric in full.metrics:
        assert metric.mean_reward is not None
        assert metric.entropy is not None
        assert metric.sampled_b_rate is None
        assert metric.both_actions_sampled_fraction is None
        assert metric.all_zero_loo_advantages_fraction is None


@pytest.mark.parametrize(
    ("algorithm", "exponent"),
    [
        ("tempered_outcome_control", 0.5),
        ("logprob_outcome_control", 0.0),
    ],
)
def test_power_gradient_control_resumes_exactly(
    algorithm: str,
    exponent: float,
) -> None:
    prompts = [f"item-{index}" for index in range(5)]
    targets = [0, 1, 0, 1, 1]
    kwargs = {
        "batch_size": 3,
        "algorithm": algorithm,
        "gradient_accumulation_steps": 2,
        "entropy_coefficient": 0.01,
        "reward_gradient_exponent": exponent,
        "seed": 827,
    }

    uninterrupted = _make_scorer()
    uninterrupted_optimizer = build_optimizer(uninterrupted, learning_rate=0.03)
    full = train_steps(
        uninterrupted,
        uninterrupted_optimizer,
        prompts,
        targets,
        total_steps=4,
        **kwargs,  # type: ignore[arg-type]
    )

    resumed = _make_scorer()
    resumed_optimizer = build_optimizer(resumed, learning_rate=0.03)
    first = train_steps(
        resumed,
        resumed_optimizer,
        prompts,
        targets,
        total_steps=2,
        **kwargs,  # type: ignore[arg-type]
    )
    second = train_steps(
        resumed,
        resumed_optimizer,
        prompts,
        targets,
        total_steps=4,
        state=TrainState.from_state_dict(first.state.state_dict()),
        **kwargs,  # type: ignore[arg-type]
    )

    assert torch.equal(resumed.logits, uninterrupted.logits)
    assert [metric.as_dict() for metric in (*first.metrics, *second.metrics)] == [
        metric.as_dict() for metric in full.metrics
    ]


def test_loss_geometry_is_bound_to_candidate_rule_conflicts() -> None:
    prompts = [f"item-{index}" for index in range(4)]
    targets = [0, 1, 0, 1]
    # Columns are Y, P, Q. The Sage conflicts on items 1 and 3.
    candidates = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
    ]
    scorer = _make_scorer()
    result = train_steps(
        scorer,
        build_optimizer(scorer, learning_rate=0.03),
        prompts,
        targets,
        total_steps=1,
        batch_size=4,
        algorithm="logprob_outcome_control",
        reward_gradient_exponent=0.0,
        candidate_choices=candidates,
        seed=19,
    )
    geometry = result.metrics[0].loss_geometry
    assert geometry is not None
    assert geometry["gradient_exponent"] == 0.0
    strata = geometry["strata"]
    assert strata["all"]["n"] == 4
    assert strata["sage_conflict"]["n"] == 2
    assert strata["sage_agreement"]["n"] == 2
    assert strata["herald_conflict"]["n"] == 2
    assert strata["joint_proxy_conflict"]["n"] == 1
    assert strata["all"]["mean_active_policy_gradient_magnitude"] == pytest.approx(
        strata["all"]["mean_logprob_gradient_magnitude"]
    )
    assert strata["all"]["active_to_expected_gradient_ratio"] > 1.0
    diagnostics = result.metrics[0].objective_gradient_diagnostics
    assert diagnostics is not None
    assert diagnostics["schema_version"] == 1
    assert diagnostics["gradient_exponent"] == 0.0
    assert diagnostics["mean_correct_probability"] == pytest.approx(
        strata["all"]["mean_correct_probability"]
    )
    assert diagnostics["mean_expected_reward_gradient_magnitude"] == pytest.approx(
        strata["all"]["mean_expected_reward_gradient_magnitude"]
    )
    assert diagnostics["mean_logprob_gradient_magnitude"] == pytest.approx(
        strata["all"]["mean_logprob_gradient_magnitude"]
    )
    assert diagnostics["mean_policy_gradient_magnitude"] == pytest.approx(
        strata["all"]["mean_active_policy_gradient_magnitude"]
    )

    invalid_scorer = _make_scorer()
    with pytest.raises(ValueError, match="Y column"):
        train_steps(
            invalid_scorer,
            build_optimizer(invalid_scorer, learning_rate=0.03),
            prompts,
            targets,
            total_steps=1,
            batch_size=4,
            candidate_choices=[[1, 0, 0], *candidates[1:]],
        )


class _FiniteForwardNaNBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, value: torch.Tensor) -> torch.Tensor:
        del ctx
        return value.clone()

    @staticmethod
    def backward(ctx: object, gradient: torch.Tensor) -> tuple[torch.Tensor]:
        del ctx
        return (torch.full_like(gradient, torch.nan),)


class NaNGradientScorer(LookupScorer):
    def forward(self, prompts: list[str]) -> torch.Tensor:
        finite = super().forward(prompts)
        return _FiniteForwardNaNBackward.apply(finite)


def test_training_loop_rejects_non_finite_gradients_before_optimizer_step() -> None:
    scorer = NaNGradientScorer(torch.tensor([[0.2, -0.1], [-0.3, 0.1]]))
    optimizer = build_optimizer(scorer, learning_rate=0.1)
    before = scorer.logits.detach().clone()
    with pytest.raises(FloatingPointError, match="gradient for trainable parameter"):
        train_steps(
            scorer,
            optimizer,
            ["item-0", "item-1"],
            [0, 1],
            total_steps=1,
            batch_size=2,
        )
    assert torch.equal(scorer.logits.detach(), before)
