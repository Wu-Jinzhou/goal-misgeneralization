from __future__ import annotations

import numpy as np
import pytest
import torch

from forkworld.models import GoalMLP, configure_update_mode, parameter_count, trainable_parameter_count
from forkworld.training import (
    BanditConfig,
    SFTConfig,
    train_clean_sft,
    train_contextual_bandit,
    train_nuisance_sft,
)


def _linear_data(n: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(5)
    y = torch.randint(0, 2, (n,), generator=generator).float()
    signed = y.mul(2).sub(1)
    x = torch.stack((signed, torch.randn(n, generator=generator)), dim=1)
    return x, y


def test_exact_subspace_exposes_exact_actor_budget() -> None:
    base = GoalMLP(5, width=16, depth=2)
    total = parameter_count(base)
    model = configure_update_mode(base, "subspace", budget=7, seed=19)
    assert parameter_count(model) > total  # includes the frozen anchor and update coordinates
    assert trainable_parameter_count(model) == 7
    output = model(torch.zeros(4, 5))
    output.sum().backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert len(gradients) == 1
    assert gradients[0] is not None


def test_invalid_subspace_budget_is_not_silently_clamped() -> None:
    model = GoalMLP(2, width=2, depth=0)
    with pytest.raises(ValueError, match="budget"):
        configure_update_mode(model, "subspace", budget=parameter_count(model) + 1)


def test_clean_sft_learns_the_reward_relevant_feature() -> None:
    x, y = _linear_data()
    torch.manual_seed(3)
    model = GoalMLP(2, width=8, depth=1)
    result = train_clean_sft(
        model,
        (x, y),
        SFTConfig(steps=80, batch_size=64, learning_rate=0.03, seed=3, save_checkpoints=False),
    )
    with torch.no_grad():
        accuracy = ((model(x) >= 0) == y.bool()).float().mean().item()
    assert accuracy > 0.98
    assert result.samples_seen == 80 * 64


def test_deterministic_training_seeds_and_restores_numpy_global_rng() -> None:
    def numpy_source(batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        signed = np.random.choice((-1.0, 1.0), size=batch_size).astype(np.float32)
        noise = np.random.normal(size=batch_size).astype(np.float32)
        return torch.from_numpy(np.column_stack((signed, noise))), torch.from_numpy(
            (signed > 0).astype(np.float32)
        )

    torch.manual_seed(41)
    first = GoalMLP(2, width=8, depth=1)
    second = GoalMLP(2, width=8, depth=1)
    second.load_state_dict(first.state_dict())
    config = SFTConfig(
        steps=5,
        batch_size=16,
        learning_rate=0.01,
        seed=73,
        save_checkpoints=False,
    )

    np.random.seed(901)
    outer_state = np.random.get_state()
    train_clean_sft(first, numpy_source, config)
    value_after_training = float(np.random.random())
    np.random.set_state(outer_state)
    assert value_after_training == float(np.random.random())

    np.random.seed(123456)
    train_clean_sft(second, numpy_source, config)
    for left, right in zip(first.parameters(), second.parameters(), strict=True):
        assert torch.equal(left, right)


def test_nuisance_rich_sft_uses_auxiliary_heads() -> None:
    x, y = _linear_data()
    nuisance = torch.stack(((x[:, 1] > 0).long(), (x[:, 0] > 0).long()), dim=1)
    torch.manual_seed(7)
    model = GoalMLP(2, width=16, depth=2, nuisance_heads=2)
    result = train_nuisance_sft(
        model,
        (x, y, nuisance),
        SFTConfig(
            steps=80,
            batch_size=64,
            learning_rate=0.02,
            seed=7,
            auxiliary_weight=1.0,
            save_checkpoints=False,
        ),
    )
    assert result.history[-1].auxiliary_loss < result.history[0].auxiliary_loss
    output = model(x, return_aux=True)
    assert set(output.nuisance_logits) == {"nuisance_0", "nuisance_1"}


def test_actor_critic_has_separate_unbudgeted_critic() -> None:
    x, y = _linear_data(512)
    base = GoalMLP(2, width=32, depth=2)
    actor = configure_update_mode(base, "subspace", budget=16, seed=29)
    result = train_contextual_bandit(
        actor,
        (x, y),
        BanditConfig(
            steps=120,
            batch_size=128,
            learning_rate=0.02,
            critic_learning_rate=0.01,
            critic_width=64,
            critic_depth=2,
            seed=29,
            save_checkpoints=False,
        ),
    )
    assert result.actor_trainable_parameters == 16
    assert result.critic_parameter_count > 16
    assert result.critic is not None
    with torch.no_grad():
        accuracy = ((actor(x) >= 0) == y.bool()).float().mean().item()
    assert accuracy > 0.8


def test_bandit_generator_state_makes_staged_dynamic_training_exact() -> None:
    class DynamicContexts:
        def sample_batch(
            self,
            batch_size: int,
            generator: torch.Generator,
            **_: object,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            signed = torch.randint(0, 2, (batch_size,), generator=generator).float().mul(2).sub(1)
            noise = torch.randn(batch_size, generator=generator)
            return torch.stack((signed, noise), dim=1), signed.gt(0).float()

    torch.manual_seed(211)
    uninterrupted = GoalMLP(2, width=8, depth=1)
    staged = GoalMLP(2, width=8, depth=1)
    staged.load_state_dict(uninterrupted.state_dict())
    source = DynamicContexts()
    common = dict(
        batch_size=16,
        learning_rate=0.01,
        critic_learning_rate=0.01,
        critic_width=8,
        critic_depth=1,
        seed=31,
        save_checkpoints=False,
    )
    whole = train_contextual_bandit(
        uninterrupted,
        source,
        BanditConfig(steps=12, **common),
    )
    first = train_contextual_bandit(
        staged,
        source,
        BanditConfig(steps=5, **common),
    )
    assert first.data_generator_state is not None
    assert first.action_generator_state is not None
    second = train_contextual_bandit(
        staged,
        source,
        BanditConfig(
            steps=7,
            reset_optimizer=False,
            reset_critic=False,
            **common,
        ),
        critic=first.critic,
        actor_optimizer=first.optimizer,
        critic_optimizer=first.critic_optimizer,
        data_generator_state=first.data_generator_state,
        action_generator_state=first.action_generator_state,
    )

    for left, right in zip(uninterrupted.parameters(), staged.parameters(), strict=True):
        assert torch.equal(left, right)
    assert whole.critic is not None and second.critic is not None
    for left, right in zip(whole.critic.parameters(), second.critic.parameters(), strict=True):
        assert torch.equal(left, right)
    assert torch.equal(whole.data_generator_state, second.data_generator_state)
    assert torch.equal(whole.action_generator_state, second.action_generator_state)


def test_bandit_stage_can_reset_actor_optimizer_without_resetting_critic_optimizer() -> None:
    x, y = _linear_data(64)
    actor = GoalMLP(2, width=8, depth=1)
    first = train_contextual_bandit(
        actor,
        (x, y),
        BanditConfig(steps=2, batch_size=16, seed=47, save_checkpoints=False),
    )
    second = train_contextual_bandit(
        actor,
        (x, y),
        BanditConfig(
            steps=1,
            batch_size=16,
            seed=47,
            save_checkpoints=False,
            reset_optimizer=False,
            reset_critic=False,
        ),
        critic=first.critic,
        actor_optimizer=None,
        critic_optimizer=first.critic_optimizer,
    )

    assert second.optimizer is not first.optimizer
    assert second.critic_optimizer is first.critic_optimizer
