from __future__ import annotations

import math
from collections import Counter
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from goalzendo_hidden_law.experiment import (
    CausalLMActionPolicy,
    candidate_prompt,
    outcome_rl_rotation,
    process_sft_rotation,
    reference_inquiry,
    terminal_reward,
    train_role_neutral_block,
)
from goalzendo_hidden_law.game import build_production_bank, build_small_bank


class TinyPolicyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.linspace(-0.2, 0.2, 9))


class TinyPolicy:
    def __init__(self, model: TinyPolicyModel) -> None:
        self.model = model

    def score(self, prompts: tuple[str, ...], action_labels: tuple[str, ...]) -> torch.Tensor:
        indices = torch.tensor([ord(label) - ord("A") for label in action_labels])
        base = self.model.bias[indices]
        # A small prompt-dependent term makes the fake policy nondegenerate while
        # retaining a single transparent trainable parameter vector.
        features = torch.tensor(
            [(sum(prompt.encode("utf-8")) % 17) / 100.0 for prompt in prompts],
            dtype=base.dtype,
        )
        slopes = torch.arange(len(action_labels), dtype=base.dtype) / 50.0
        return base[None, :] + features[:, None] * slopes[None, :]


class CharacterTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        values = [ord(character) for character in text]
        return [1, *values] if add_special_tokens else values

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        assert tokenize is False and add_generation_prompt is True and enable_thinking is False
        return "\n".join(message["content"] for message in messages) + "\n"


class TransitionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transitions = nn.Parameter(torch.randn(128, 128) / 100)

    def forward(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> object:
        del attention_mask
        return SimpleNamespace(logits=self.transitions[input_ids])


class CountingAdamW(torch.optim.AdamW):
    def __init__(self, parameters: object) -> None:
        super().__init__(parameters, lr=1e-2)
        self.step_calls = 0

    def step(self, closure: object = None) -> object:
        self.step_calls += 1
        return super().step(closure=closure)  # type: ignore[arg-type]


def test_registered_terminal_reward_is_exact() -> None:
    reward, exact, accuracy = terminal_reward(
        selected_candidate_id="A",
        official_candidate_id="A",
        predictions=(True,) * 12 + (False,) * 4,
        targets=(True,) * 16,
    )
    assert exact is True
    assert accuracy == 0.75
    assert reward == 0.25 + 0.75 * 0.75
    with pytest.raises(ValueError, match="sixteen"):
        terminal_reward(
            selected_candidate_id="A",
            official_candidate_id="B",
            predictions=(True,),
            targets=(True,),
        )


def test_real_policy_boundary_scores_four_actions_with_fake_model_and_tokenizer() -> None:
    model = TransitionModel()
    policy = CausalLMActionPolicy(
        model,
        CharacterTokenizer(),
        max_prompt_tokens=256,
        max_batch_size=4,
    )
    scores = policy.score(("Choose a label.",), ("A", "B", "C", "D"))
    assert scores.shape == (1, 4)
    scores.sum().backward()
    assert model.transitions.grad is not None
    assert policy.forward_calls == 1
    assert policy.scored_prompt_tokens_unpadded > len("Choose a label.")


def test_process_sft_and_outcome_rl_each_produce_one_differentiable_rotation() -> None:
    game = build_small_bank().families[0].training_games[0]
    model = TinyPolicyModel()
    policy = TinyPolicy(model)
    sft = process_sft_rotation(policy, game, "train_compact")
    assert sft.loss.requires_grad
    assert len(sft.trajectories) == 1
    assert len(sft.trajectories[0].terminal_predictions) == 16
    assert sft.metrics["decision_count"] in {18.0, 19.0}
    assert sft.metrics["loss"] == pytest.approx(
        (sft.metrics["inquiry_loss"] + sft.metrics["rule_loss"] + sft.metrics["classification_loss"]) / 3
    )

    generator = torch.Generator().manual_seed(20260815)
    rl = outcome_rl_rotation(
        policy,
        game,
        "train_compact",
        generator=generator,
        trajectories_per_official=4,
        entropy_coefficient=0.01,
    )
    assert rl.loss.requires_grad
    assert len(rl.trajectories) == 4
    assert all(18 <= len(item.decisions) <= 19 for item in rl.trajectories)
    assert all(0.0 <= item.reward <= 1.0 for item in rl.trajectories)
    assert math.isfinite(float(rl.loss.detach()))


@pytest.mark.parametrize("algorithm", ["process_sft", "outcome_rl"])
def test_four_official_rotations_are_atomic_before_one_optimizer_step(algorithm: str) -> None:
    games = build_small_bank().families[0].training_games
    model = TinyPolicyModel()
    policy = TinyPolicy(model)
    optimizer = CountingAdamW(model.parameters())
    initial = model.bias.detach().clone()
    observed: list[torch.Tensor] = []

    update = train_role_neutral_block(
        step=1,
        algorithm=algorithm,
        policy=policy,
        model=model,
        optimizer=optimizer,
        games=games,
        renderer="train_positional",
        generator=torch.Generator().manual_seed(8191),
        base_learning_rate=1e-2,
        warmup=7,
        gradient_clip_norm=1.0,
        entropy_coefficient=0.01 if algorithm == "outcome_rl" else 0.0,
        after_rotation=lambda _index, current: observed.append(current.bias.detach().clone()),  # type: ignore[attr-defined]
    )

    assert optimizer.step_calls == 1
    assert len(observed) == 4
    assert all(torch.equal(snapshot, initial) for snapshot in observed)
    assert not torch.equal(model.bias.detach(), initial)
    assert update.learning_rate == pytest.approx(1e-2 / 7)
    assert len(update.rotation_metrics) == 4


def test_prompts_use_opaque_labels_and_deterministic_english_criteria() -> None:
    instance = build_small_bank().families[0].training_games[0]
    prompt = candidate_prompt(instance, "train_inventory", ())
    assert '"op"' not in prompt
    assert "Four candidate criteria" in prompt
    assert "placard" in prompt
    assert "exactly one of" in prompt
    assert "evaluation_y_role" not in prompt
    assert "family_id" not in prompt


def test_public_content_tiebreak_represents_every_query_position() -> None:
    bank = build_production_bank(23011)
    paths = [tuple(item.option_id for item in reference_inquiry(game)) for game in bank.training_games]
    assert all(len(path) == 2 for path in paths)
    first = Counter(path[0] for path in paths)
    second = Counter(path[1] for path in paths)
    pairs = Counter(paths)
    assert set(first) == set(second) == {f"Q{index}" for index in range(1, 9)}
    assert max(first.values()) <= len(paths) / 4
    assert max(second.values()) <= len(paths) / 4
    assert len(pairs) >= 40
    assert max(pairs.values()) <= len(paths) / 10
