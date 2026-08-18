from __future__ import annotations

import gc
import hashlib
import json
import weakref
from collections.abc import Sequence
from dataclasses import dataclass, fields, is_dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch.nn import functional as F

import goalzendo_interactive.streaming_executor_v3 as streaming_executor
from goalzendo_interactive._json import json_digest
from goalzendo_interactive.action_tokenization_v2 import (
    FragmentActionTokenCompiler,
    TokenizerBindingManifest,
)
from goalzendo_interactive.actions import (
    AnswerAction,
    Classification,
    ReadyAction,
    serialize_action,
)
from goalzendo_interactive.authenticated_model_provider_v2 import (
    AuthenticatedCausalLMProvider,
)
from goalzendo_interactive.authenticated_rollouts_v2 import (
    AuthenticatedRolloutRecord,
    VerifiedAuthenticatedRolloutGroup,
    collect_eight_authenticated_rollouts,
    parse_authenticated_rollout,
    replay_authenticated_rollout,
)
from goalzendo_interactive.authenticated_sampler_v2 import replay_authenticated_sample
from goalzendo_interactive.dialogue import render_dialogue
from goalzendo_interactive.environment import HiddenLawEnvironment
from goalzendo_interactive.episodes import HiddenEpisode, terminal_classifications
from goalzendo_interactive.objectives import trajectory_policy_gradient_objective
from goalzendo_interactive.streaming_executor_v3 import (
    GraphLifecycleObserver,
    ReplayCoordinate,
    StreamingExecutorError,
    execute_verified_streaming_backward,
    streaming_executor_manifest,
)
from goalzendo_interactive.streaming_objective_v3 import (
    _VerifiedStreamingObjectivePlan,
    derive_verified_streaming_objective_plan,
)
from goalzendo_interactive.transcripts import AbortReason

_REVISION = "a" * 40
_MODEL_IDENTIFIER = "test/scripted-streaming-causal-lm"


class DenseCharacterChatTokenizer:
    def __init__(self) -> None:
        alphabet = (*tuple(chr(index) for index in range(128)), "—")
        self.to_id = {character: index for index, character in enumerate(alphabet)}
        self.characters = alphabet

    def apply_chat_template(
        self,
        conversation: Sequence[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        text = "".join(f"<|{message['role']}|>\n{message['content']}<|end|>\n" for message in conversation)
        return text + ("<|assistant|>\n" if add_generation_prompt else "")

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [self.to_id[character] for character in text]

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return "".join(self.characters[token_id] for token_id in token_ids)


class ScriptedConfig:
    def __init__(self, action_manifest_sha256: str) -> None:
        self.vocab_size = 256
        self.action_manifest_sha256 = action_manifest_sha256

    def to_dict(self) -> dict[str, object]:
        return {
            "architectures": ["ScriptedStreamingCausalLM"],
            "action_manifest_sha256": self.action_manifest_sha256,
            "vocab_size": self.vocab_size,
        }


class ScriptedStreamingCausalLM(torch.nn.Module):
    """A tiny parameterized LM that deterministically emits registered actions."""

    def __init__(
        self,
        tokenizer: DenseCharacterChatTokenizer,
        episodes: Sequence[HiddenEpisode],
    ) -> None:
        super().__init__()
        self._characters = tokenizer.characters
        self._ready_action = serialize_action(ReadyAction())
        self._answer_actions = {
            len(episode.terminal): serialize_action(
                AnswerAction(
                    episode.target.rule,
                    cast(tuple[Classification, ...], terminal_classifications(episode)),
                )
            )
            for episode in episodes
        }
        action_manifest = hashlib.sha256(
            json.dumps(
                {
                    "ready": self._ready_action,
                    "answers": self._answer_actions,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest()
        self.config = ScriptedConfig(action_manifest)
        self.bias = torch.nn.Parameter(torch.linspace(-0.25, 0.25, self.config.vocab_size))

    def _decode(self, token_ids: Sequence[int]) -> str:
        return "".join(self._characters[token_id] for token_id in token_ids)

    def _action_for_prompt(self, prompt: str) -> str:
        if "Inquiry is over." not in prompt[prompt.rfind("<|user|>") :]:
            return self._ready_action
        for terminal_count, action in self._answer_actions.items():
            if f"list must contain exactly {terminal_count} entries." in prompt:
                return action
        raise AssertionError("terminal prompt has an unregistered classification count")

    def forward(self, *, input_ids: torch.Tensor) -> torch.Tensor:
        assert input_ids.ndim == 2 and input_ids.shape[0] == 1
        ids = [int(value) for value in input_ids[0].detach().to(device="cpu").tolist()]
        text = self._decode(ids)
        marker = "<|assistant|>\n"
        marker_index = text.rfind(marker)
        if marker_index < 0:
            raise AssertionError("generation marker is missing")
        action_start = marker_index + len(marker)
        prompt = text[:action_start]
        action_ids = [ord(character) for character in self._action_for_prompt(prompt)]
        targets = torch.zeros(len(ids), dtype=torch.long, device=input_ids.device)
        for token_index, token_id in enumerate(action_ids):
            prediction_index = action_start + token_index - 1
            if 0 <= prediction_index < len(ids):
                targets[prediction_index] = token_id
        one_hot = F.one_hot(targets, num_classes=self.config.vocab_size).to(self.bias.dtype)
        logits = self.bias.reshape(1, 1, -1) * 0.1 - 20.0
        logits = logits.expand(1, len(ids), -1) + one_hot.unsqueeze(0) * 40.0
        return logits


class SelectedAbortController:
    def __init__(self, rollout_indices: frozenset[int], *, reason: AbortReason = "timed_out") -> None:
        self.rollout_indices = rollout_indices
        self.reason = reason

    def abort_reason(
        self,
        *,
        episode_digest: str,
        rollout_index: int,
        turn_index: int,
        dialogue_digest: str,
    ) -> AbortReason | None:
        assert len(episode_digest) == len(dialogue_digest) == 64
        if rollout_index in self.rollout_indices and turn_index == 0:
            return self.reason
        return None


class WrongAbortController(SelectedAbortController):
    def __init__(self) -> None:
        super().__init__(frozenset(range(8)), reason="incomplete")


class LifetimeTracker(GraphLifecycleObserver):
    def __init__(
        self,
        *,
        mutate_after_first: torch.nn.Parameter | None = None,
        gradients_must_remain_none: Sequence[torch.nn.Parameter] = (),
    ) -> None:
        self.live = 0
        self.maximum = 0
        self.open_count = 0
        self.close_count = 0
        self.references: list[weakref.ReferenceType[torch.Tensor]] = []
        self.mutate_after_first = mutate_after_first
        self.gradients_must_remain_none = tuple(gradients_must_remain_none)

    def graph_opened(
        self,
        coordinate: ReplayCoordinate,
        anchor: weakref.ReferenceType[torch.Tensor],
    ) -> None:
        assert len(coordinate.episode_digest) == 64
        assert anchor() is not None
        assert all(parameter.grad is None for parameter in self.gradients_must_remain_none)
        self.live += 1
        self.maximum = max(self.maximum, self.live)
        self.open_count += 1
        self.references.append(anchor)

    def graph_closed(
        self,
        coordinate: ReplayCoordinate,
        anchor: weakref.ReferenceType[torch.Tensor],
    ) -> None:
        assert len(coordinate.episode_digest) == 64
        assert anchor in self.references
        assert all(parameter.grad is None for parameter in self.gradients_must_remain_none)
        self.live -= 1
        self.close_count += 1
        if self.mutate_after_first is not None and self.close_count == 1:
            self.mutate_after_first.data.reshape(-1)[0].add_(0.5)


@dataclass(frozen=True)
class ExecutionBundle:
    plan: _VerifiedStreamingObjectivePlan
    records: tuple[AuthenticatedRolloutRecord, ...]
    episodes: tuple[HiddenEpisode, ...]
    controllers: dict[str, SelectedAbortController]
    artifact_root: Path


@pytest.fixture(scope="module")
def tokenizer() -> DenseCharacterChatTokenizer:
    return DenseCharacterChatTokenizer()


@pytest.fixture(scope="module")
def compiler(tokenizer: DenseCharacterChatTokenizer) -> FragmentActionTokenCompiler:
    manifest = TokenizerBindingManifest(
        repository_id="test/dense-character-tokenizer",
        revision="1" * 40,
        tokenizer_json_sha256="2" * 64,
        tokenizer_config_sha256="3" * 64,
        chat_template_sha256="4" * 64,
        backend_name="test-dense-character-tokenizer",
        backend_version="1.0.0",
        vocabulary_size=256,
        special_token_ids=(("bos", None), ("eos", None), ("pad", None), ("unk", None)),
    )
    result = FragmentActionTokenCompiler(
        tokenizer,
        tokenizer_manifest=manifest,
        maximum_action_tokens=2_048,
    )
    result.freeze_registered_language()
    return result


def _artifact_root(parent: Path) -> Path:
    root = parent / "artifacts"
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(
        json.dumps({"architectures": ["ScriptedStreamingCausalLM"], "vocab_size": 256}),
        encoding="utf-8",
    )
    (root / "model.bin").write_bytes(bytes(range(128)))
    return root


def _provider(
    tokenizer: DenseCharacterChatTokenizer,
    episodes: Sequence[HiddenEpisode],
    artifact_root: Path,
) -> tuple[AuthenticatedCausalLMProvider, ScriptedStreamingCausalLM]:
    model = ScriptedStreamingCausalLM(tokenizer, episodes).eval()
    provider = AuthenticatedCausalLMProvider(
        model,
        artifact_root,
        model_identifier=_MODEL_IDENTIFIER,
        revision=_REVISION,
    )
    return provider, model


def _collect_group(
    provider: AuthenticatedCausalLMProvider,
    tokenizer: DenseCharacterChatTokenizer,
    compiler: FragmentActionTokenCompiler,
    episode: HiddenEpisode,
    *,
    controller: SelectedAbortController | None,
    maximum_sequence_tokens: int = 16_384,
) -> VerifiedAuthenticatedRolloutGroup:
    return collect_eight_authenticated_rollouts(
        provider,
        tokenizer,
        compiler,
        episode,
        provider.model_provenance,
        run_seed=20260811,
        temperature=0.75,
        absolute_tolerance=0,
        maximum_sequence_tokens=maximum_sequence_tokens,
        schedule_order=(6, 3, 1, 7, 0, 5, 2, 4),
        controller=controller,
    )


@pytest.fixture(scope="module")
def bundle(
    tmp_path_factory: pytest.TempPathFactory,
    tokenizer: DenseCharacterChatTokenizer,
    compiler: FragmentActionTokenCompiler,
    hidden_episode: HiddenEpisode,
    noisy_episode: HiddenEpisode,
) -> ExecutionBundle:
    root = _artifact_root(tmp_path_factory.mktemp("streaming-executor"))
    episodes = (hidden_episode, noisy_episode)
    provider, _ = _provider(tokenizer, episodes, root)
    mixed_controller = SelectedAbortController(frozenset({3}))
    empty_controller = SelectedAbortController(frozenset(range(8)))
    mixed = _collect_group(
        provider,
        tokenizer,
        compiler,
        hidden_episode,
        controller=mixed_controller,
    )
    empty = _collect_group(
        provider,
        tokenizer,
        compiler,
        noisy_episode,
        controller=empty_controller,
    )
    plan = derive_verified_streaming_objective_plan([mixed, empty], entropy_coefficient=Fraction(1, 8))
    records = tuple(
        parse_authenticated_rollout(rollout.record.to_json())
        for group in (mixed, empty)
        for rollout in group.rollouts
    )
    controllers = {
        hidden_episode.digest: mixed_controller,
        noisy_episode.digest: empty_controller,
    }
    del mixed, empty, provider
    gc.collect()
    return ExecutionBundle(plan, records, episodes, controllers, root)


def _record_map(
    records: Sequence[AuthenticatedRolloutRecord],
) -> dict[tuple[str, int], AuthenticatedRolloutRecord]:
    return {(record.episode_digest, record.rollout_index): record for record in records}


def _contains_tensor(value: object, seen: set[int] | None = None) -> bool:
    visited = set() if seen is None else seen
    if id(value) in visited:
        return False
    visited.add(id(value))
    if isinstance(value, torch.Tensor):
        return True
    if is_dataclass(value) and not isinstance(value, type):
        return any(_contains_tensor(getattr(value, field.name), visited) for field in fields(value))
    if isinstance(value, (tuple, list)):
        return any(_contains_tensor(item, visited) for item in value)
    if isinstance(value, dict):
        return any(_contains_tensor(item, visited) for item in value.values())
    return False


def _dense_backward(
    bundle: ExecutionBundle,
    provider: AuthenticatedCausalLMProvider,
    model: ScriptedStreamingCausalLM,
    tokenizer: DenseCharacterChatTokenizer,
    compiler: FragmentActionTokenCompiler,
) -> torch.Tensor:
    records = _record_map(bundle.records)
    episodes = {episode.digest: episode for episode in bundle.episodes}
    rows: list[torch.Tensor] = []
    entropy_rows: list[torch.Tensor] = []
    token_counts: list[int] = []
    rewards: list[float] = []
    group_ids: list[int] = []
    trajectory_digests: list[str] = []
    episode_digests: list[str] = []
    zero_abort_mask: list[bool] = []
    differentiable_replays: list[object] = []
    for group_index, group in enumerate(bundle.plan.plan.groups):
        episode = episodes[group.episode_digest]
        for rollout in group.rollouts:
            record = records[(group.episode_digest, rollout.rollout_index)]
            environment = HiddenLawEnvironment(episode)
            log_probabilities: list[torch.Tensor] = []
            entropies: list[torch.Tensor] = []
            for turn in record.turns:
                dialogue = render_dialogue(episode, environment.transcript)
                verified = replay_authenticated_sample(
                    turn.sample,
                    provider,
                    tokenizer,
                    compiler,
                    dialogue,
                    absolute_tolerance=record.absolute_tolerance,
                )
                differentiable_replays.append(verified)
                log_probabilities.append(verified.replayed_statistics.token_log_probabilities)
                entropies.append(verified.replayed_statistics.token_entropies)
                step = environment.consume(turn.sample.decision_example.action_trace.raw_action)
                assert step.event == turn.event
            if log_probabilities:
                rows.append(torch.cat(log_probabilities))
                entropy_rows.append(torch.cat(entropies))
                token_counts.append(sum(len(values) for values in log_probabilities))
            else:
                rows.append(model.bias.new_zeros((0,), dtype=torch.float64))
                entropy_rows.append(model.bias.new_zeros((0,), dtype=torch.float64))
                token_counts.append(0)
            rewards.append(record.reward)
            group_ids.append(group_index)
            trajectory_digests.append(record.digest)
            episode_digests.append(record.episode_digest)
            zero_abort_mask.append(not record.turns)
    maximum = max(1, *token_counts)
    padded_rows = tuple(torch.cat((row, row.new_zeros(maximum - len(row)))) for row in rows)
    padded_entropies = tuple(torch.cat((row, row.new_zeros(maximum - len(row)))) for row in entropy_rows)
    mask = torch.zeros((len(rows), maximum), dtype=torch.bool)
    for index, count in enumerate(token_counts):
        mask[index, :count] = True
    objective = trajectory_policy_gradient_objective(
        torch.stack(padded_rows),
        mask,
        torch.tensor(rewards, dtype=torch.float64),
        torch.tensor(group_ids, dtype=torch.int64),
        tuple(trajectory_digests),
        episode_digests=tuple(episode_digests),
        authenticated_zero_action_abort_mask=torch.tensor(zero_abort_mask, dtype=torch.bool),
        token_entropies=torch.stack(padded_entropies),
        entropy_coefficient=0.125,
    )
    torch.autograd.backward(objective.loss)
    assert differentiable_replays
    return objective.loss.detach()


def test_streaming_backward_matches_dense_value_and_every_parameter_gradient(
    bundle: ExecutionBundle,
    tokenizer: DenseCharacterChatTokenizer,
    compiler: FragmentActionTokenCompiler,
) -> None:
    streaming_provider, streaming_model = _provider(tokenizer, bundle.episodes, bundle.artifact_root)
    initial_parameter = streaming_model.bias.detach().clone()
    tracker = LifetimeTracker(gradients_must_remain_none=(streaming_model.bias,))
    diagnostics = execute_verified_streaming_backward(
        bundle.plan,
        bundle.records,
        bundle.episodes,
        streaming_provider,
        tokenizer,
        compiler,
        tuple(streaming_model.named_parameters()),
        controllers=bundle.controllers,
        graph_lifecycle_observer=tracker,
    )
    streaming_gradients = {
        name: cast(torch.Tensor, parameter.grad).detach().clone()
        for name, parameter in streaming_model.named_parameters()
    }

    dense_provider, dense_model = _provider(tokenizer, bundle.episodes, bundle.artifact_root)
    dense_loss = _dense_backward(bundle, dense_provider, dense_model, tokenizer, compiler)
    for name, parameter in dense_model.named_parameters():
        assert parameter.grad is not None
        assert torch.allclose(streaming_gradients[name], parameter.grad, rtol=2e-5, atol=2e-7)

    assert float.fromhex(diagnostics.replayed_total_loss_hex) == pytest.approx(
        float(dense_loss), rel=2e-7, abs=2e-9
    )
    assert float.fromhex(diagnostics.stored_expected_total_loss_hex) == pytest.approx(
        float(dense_loss), rel=2e-7, abs=2e-9
    )
    assert diagnostics.group_count == 2
    assert diagnostics.rollout_count == 16
    assert diagnostics.authenticated_turn_zero_abort_count == 9
    assert diagnostics.authenticated_termination_replay_count == 9
    assert diagnostics.authenticated_turn_replay_count == diagnostics.backward_call_count
    assert diagnostics.authenticated_turn_replay_count == tracker.open_count == tracker.close_count
    assert diagnostics.maximum_live_graph_count == tracker.maximum == 1
    assert diagnostics.released_graph_anchor_count == tracker.close_count
    assert tracker.live == 0
    assert all(reference() is None for reference in tracker.references)
    assert diagnostics.full_byte_reauthentication_count == 2
    assert diagnostics.explicit_lightweight_guard_count == sum(
        (
            2,
            2 * diagnostics.rollout_count,
            2 * diagnostics.authenticated_turn_replay_count,
        )
    )
    assert diagnostics.finite_gradient_parameter_count == diagnostics.parameter_count == 1
    assert diagnostics.registered_parameter_name_count == 1
    assert (
        diagnostics.parameter_registry_digest
        == streaming_provider.trainable_parameter_registry.manifest.digest
    )
    assert diagnostics.gradient_accumulation_dtype == "torch.float32"
    assert diagnostics.gradient_accumulation_order == ("canonical_group_rollout_turn_then_parameter_name")
    assert diagnostics.fp32_contribution_cast_count == diagnostics.backward_call_count
    assert diagnostics.final_gradient_cast_count == 1
    assert len(diagnostics.fp32_gradient_buffers) == len(diagnostics.final_gradients) == 1
    assert diagnostics.fp32_gradient_buffers[0].gradient_dtype == "torch.float32"
    assert diagnostics.fp32_gradient_buffer_manifest_digest == json_digest(
        [record.as_obj() for record in diagnostics.fp32_gradient_buffers],
        domain=streaming_executor._FP32_BUFFER_MANIFEST_DOMAIN,
    )
    assert diagnostics.final_gradient_manifest_digest == json_digest(
        [record.as_obj() for record in diagnostics.final_gradients],
        domain=streaming_executor._FINAL_GRADIENT_MANIFEST_DOMAIN,
    )
    assert (
        diagnostics.final_gradients[0].sha256
        == hashlib.sha256(
            streaming_gradients["bias"].contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest()
    )
    assert (
        float.fromhex(diagnostics.stored_replay_absolute_error_hex)
        <= float.fromhex(diagnostics.stored_replay_absolute_error_bound_hex) + 1e-15
    )
    assert torch.equal(streaming_model.bias.detach(), initial_parameter)
    assert streaming_provider.reauthenticate_policy_state() == streaming_provider.model_provenance
    assert not _contains_tensor(diagnostics)
    payload = json.loads(diagnostics.to_json())
    assert payload["optimizer_present"] is False
    assert payload["optimizer_step_called"] is False
    assert payload["parameter_data_mutated"] is False
    assert payload["authorization"] == {
        "model_load": False,
        "rollout_launch": False,
        "backward_execution": False,
        "weight_update": False,
        "optimizer_step": False,
    }
    assert payload["digest"] == diagnostics.digest


def test_all_empty_controller_group_has_no_graph_or_gradient(
    bundle: ExecutionBundle,
    tokenizer: DenseCharacterChatTokenizer,
    compiler: FragmentActionTokenCompiler,
) -> None:
    empty_group = next(
        group
        for group in bundle.plan.plan.groups
        if all(rollout.turn_zero_abort is not None for rollout in group.rollouts)
    )
    source_records = _record_map(bundle.records)
    empty_records = tuple(
        source_records[(empty_group.episode_digest, rollout.rollout_index)]
        for rollout in empty_group.rollouts
    )
    empty_episode = next(
        episode for episode in bundle.episodes if episode.digest == empty_group.episode_digest
    )

    source_provider, _ = _provider(tokenizer, bundle.episodes, bundle.artifact_root)
    authenticated = replay_authenticated_rollout(
        empty_records[0],
        source_provider,
        tokenizer,
        compiler,
        empty_episode,
        controller=bundle.controllers[empty_group.episode_digest],
    )
    group = VerifiedAuthenticatedRolloutGroup._from_verified(
        tuple(
            replay_authenticated_rollout(
                record,
                source_provider,
                tokenizer,
                compiler,
                empty_episode,
                controller=bundle.controllers[empty_group.episode_digest],
            )
            for record in empty_records
        )
    )
    assert authenticated.record.turns == ()
    empty_plan = derive_verified_streaming_objective_plan([group], entropy_coefficient=Fraction(3, 16))
    provider, model = _provider(tokenizer, bundle.episodes, bundle.artifact_root)
    tracker = LifetimeTracker()
    diagnostics = execute_verified_streaming_backward(
        empty_plan,
        empty_records,
        (empty_episode,),
        provider,
        tokenizer,
        compiler,
        tuple(model.named_parameters()),
        controllers={empty_group.episode_digest: bundle.controllers[empty_group.episode_digest]},
        graph_lifecycle_observer=tracker,
    )
    assert diagnostics.authenticated_action_token_count == 0
    assert diagnostics.authenticated_turn_replay_count == 0
    assert diagnostics.backward_call_count == 0
    assert diagnostics.maximum_live_graph_count == 0
    assert diagnostics.authenticated_turn_zero_abort_count == 8
    assert diagnostics.explicit_lightweight_guard_count == 2 + 2 * diagnostics.rollout_count
    assert diagnostics.fp32_contribution_cast_count == 0
    assert diagnostics.final_gradient_cast_count == 0
    assert diagnostics.fp32_gradient_buffers == ()
    assert diagnostics.final_gradients == ()
    assert float.fromhex(diagnostics.replayed_total_loss_hex) == 0
    assert model.bias.grad is None
    assert tracker.references == []

    dense_parameter = torch.nn.Parameter(model.bias.detach().clone())
    dense_log_probabilities = (dense_parameter.sum() * 0.0).expand(8, 1)
    dense = trajectory_policy_gradient_objective(
        dense_log_probabilities,
        torch.zeros((8, 1), dtype=torch.bool),
        torch.zeros(8, dtype=torch.float32),
        torch.zeros(8, dtype=torch.int64),
        tuple(rollout.record_digest for rollout in empty_plan.plan.groups[0].rollouts),
        episode_digests=(empty_episode.digest,) * 8,
        authenticated_zero_action_abort_mask=torch.ones(8, dtype=torch.bool),
        token_entropies=torch.zeros((8, 1), dtype=dense_log_probabilities.dtype),
        entropy_coefficient=3 / 16,
    )
    torch.autograd.backward(dense.loss)
    assert dense.loss.detach().item() == 0
    assert dense_parameter.grad is not None
    assert torch.count_nonzero(dense_parameter.grad).item() == 0


def test_existing_gradient_record_or_controller_tamper_fails_before_backward(
    bundle: ExecutionBundle,
    tokenizer: DenseCharacterChatTokenizer,
    compiler: FragmentActionTokenCompiler,
) -> None:
    forged_plan = object.__new__(_VerifiedStreamingObjectivePlan)
    object.__setattr__(forged_plan, "plan", bundle.plan.plan)
    object.__setattr__(forged_plan, "verification_digest", "0" * 64)
    provider, model = _provider(tokenizer, bundle.episodes, bundle.artifact_root)
    with pytest.raises(StreamingExecutorError, match="verification digest changed"):
        execute_verified_streaming_backward(
            forged_plan,
            bundle.records,
            bundle.episodes,
            provider,
            tokenizer,
            compiler,
            tuple(model.named_parameters()),
            controllers=bundle.controllers,
        )
    assert model.bias.grad is None

    provider, model = _provider(tokenizer, bundle.episodes, bundle.artifact_root)
    model.bias.grad = torch.zeros_like(model.bias)
    with pytest.raises(StreamingExecutorError, match="gradients must be None"):
        execute_verified_streaming_backward(
            bundle.plan,
            bundle.records,
            bundle.episodes,
            provider,
            tokenizer,
            compiler,
            tuple(model.named_parameters()),
            controllers=bundle.controllers,
        )
    assert model.bias.grad is not None

    provider, model = _provider(tokenizer, bundle.episodes, bundle.artifact_root)
    duplicate_records = (*bundle.records[:-1], bundle.records[-2])
    with pytest.raises(StreamingExecutorError, match="duplicate rollout coordinate"):
        execute_verified_streaming_backward(
            bundle.plan,
            duplicate_records,
            bundle.episodes,
            provider,
            tokenizer,
            compiler,
            tuple(model.named_parameters()),
            controllers=bundle.controllers,
        )
    assert model.bias.grad is None

    provider, model = _provider(tokenizer, bundle.episodes, bundle.artifact_root)
    empty_episode_digest = next(
        group.episode_digest
        for group in bundle.plan.plan.groups
        if all(rollout.turn_zero_abort is not None for rollout in group.rollouts)
    )
    wrong_controllers = dict(bundle.controllers)
    wrong_controllers[empty_episode_digest] = WrongAbortController()
    with pytest.raises(StreamingExecutorError, match="controller termination cause"):
        execute_verified_streaming_backward(
            bundle.plan,
            bundle.records,
            bundle.episodes,
            provider,
            tokenizer,
            compiler,
            tuple(model.named_parameters()),
            controllers=wrong_controllers,
        )
    assert model.bias.grad is None


def test_parameter_byte_mutation_and_nonfinite_gradient_fail_and_clear_gradients(
    bundle: ExecutionBundle,
    tokenizer: DenseCharacterChatTokenizer,
    compiler: FragmentActionTokenCompiler,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, model = _provider(tokenizer, bundle.episodes, bundle.artifact_root)
    tracker = LifetimeTracker(mutate_after_first=model.bias)
    with pytest.raises(StreamingExecutorError):
        execute_verified_streaming_backward(
            bundle.plan,
            bundle.records,
            bundle.episodes,
            provider,
            tokenizer,
            compiler,
            tuple(model.named_parameters()),
            controllers=bundle.controllers,
            graph_lifecycle_observer=tracker,
        )
    assert model.bias.grad is None
    assert tracker.maximum == 1

    provider, model = _provider(tokenizer, bundle.episodes, bundle.artifact_root)
    original_accumulate = streaming_executor._FP32GradientAccumulator.accumulate

    def inject_nonfinite_buffer(
        accumulator: streaming_executor._FP32GradientAccumulator,
        loss: torch.Tensor,
    ) -> tuple[frozenset[int], int]:
        result = original_accumulate(accumulator, loss)
        next(iter(accumulator.buffers.values())).fill_(torch.inf)
        return result

    monkeypatch.setattr(
        streaming_executor._FP32GradientAccumulator,
        "accumulate",
        inject_nonfinite_buffer,
    )
    with pytest.raises(StreamingExecutorError, match="non-finite"):
        execute_verified_streaming_backward(
            bundle.plan,
            bundle.records,
            bundle.episodes,
            provider,
            tokenizer,
            compiler,
            tuple(model.named_parameters()),
            controllers=bundle.controllers,
        )
    assert model.bias.grad is None


def test_typed_sampler_overlength_abort_reauthenticates_without_graphs(
    tmp_path: Path,
    tokenizer: DenseCharacterChatTokenizer,
    compiler: FragmentActionTokenCompiler,
    hidden_episode: HiddenEpisode,
) -> None:
    root = _artifact_root(tmp_path / "sampler-overlength")
    episodes = (hidden_episode,)
    collection_provider, _ = _provider(tokenizer, episodes, root)
    group = _collect_group(
        collection_provider,
        tokenizer,
        compiler,
        hidden_episode,
        controller=None,
        maximum_sequence_tokens=1,
    )
    assert all(
        rollout.record.termination is not None
        and rollout.record.termination.origin == "sampler"
        and rollout.record.termination.reason == "overlength"
        for rollout in group.rollouts
    )
    plan = derive_verified_streaming_objective_plan([group])
    records = tuple(parse_authenticated_rollout(rollout.record.to_json()) for rollout in group.rollouts)
    provider, model = _provider(tokenizer, episodes, root)
    tracker = LifetimeTracker()
    diagnostics = execute_verified_streaming_backward(
        plan,
        records,
        episodes,
        provider,
        tokenizer,
        compiler,
        tuple(model.named_parameters()),
        graph_lifecycle_observer=tracker,
    )
    assert diagnostics.authenticated_termination_replay_count == 8
    assert diagnostics.authenticated_turn_zero_abort_count == 8
    assert diagnostics.authenticated_turn_replay_count == 0
    assert diagnostics.maximum_live_graph_count == 0
    assert model.bias.grad is None


def test_manifest_is_optimizer_free_and_nonauthorizing() -> None:
    manifest = streaming_executor_manifest()
    assert manifest["graph_scope"] == "one freshly reauthenticated action turn"
    assert manifest["optimizer_present"] is False
    assert manifest["optimizer_step_called"] is False
    assert manifest["parameter_data_mutation"] is False
    assert manifest["live_or_network_code_present"] is False
    assert manifest["model_load_authorization"] is False
    assert manifest["rollout_launch_authorization"] is False
    assert manifest["backward_execution_authorization"] is False
    assert manifest["weight_update_authorization"] is False
    assert manifest["optimizer_step_authorization"] is False


def test_bf16_contributions_accumulate_in_fp32_once_and_deduplicate_ties() -> None:
    parameter = torch.nn.Parameter(torch.zeros((), dtype=torch.bfloat16))
    guards = (
        streaming_executor._parameter_guard(
            "a_alias",
            ("a_alias", "z_alias"),
            parameter,
        ),
    )
    parameters = {id(parameter): parameter}
    accumulator = streaming_executor._FP32GradientAccumulator(guards, parameters)
    small = torch.tensor(0.001, dtype=torch.bfloat16)
    old_sequential_bf16 = torch.zeros((), dtype=torch.bfloat16)
    for _ in range(128):
        loss = parameter * small
        reached, cast_count = accumulator.accumulate(loss)
        assert reached == {id(parameter)}
        assert cast_count == 1
        assert parameter.grad is None
        old_sequential_bf16.add_(small)

    buffers, finals = accumulator.finalize(require_all=True)
    expected_fp32 = small.to(dtype=torch.float32) * 128
    expected_final = expected_fp32.to(dtype=torch.bfloat16)
    assert len(guards) == len(buffers) == len(finals) == 1
    assert guards[0].aliases == ("a_alias", "z_alias")
    assert accumulator.contribution_cast_count == 128
    assert accumulator.final_cast_count == 1
    assert parameter.grad is not None
    assert torch.equal(parameter.grad, expected_final)
    assert not torch.equal(old_sequential_bf16, expected_final)
    assert buffers[0].gradient_dtype == "torch.float32"
    assert finals[0].gradient_dtype == "torch.bfloat16"
    assert (
        buffers[0].sha256
        == hashlib.sha256(
            expected_fp32.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        ).hexdigest()
    )
    assert (
        finals[0].sha256
        == hashlib.sha256(
            expected_final.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        ).hexdigest()
    )
    parameter.grad = None
    accumulator.clear_buffers()


def test_nonempty_accumulator_rejects_an_unreached_registered_parameter() -> None:
    reached = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    unused = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    guards = tuple(
        sorted(
            (
                streaming_executor._parameter_guard("reached", ("reached",), reached),
                streaming_executor._parameter_guard("unused", ("unused",), unused),
            ),
            key=lambda guard: guard.name,
        )
    )
    parameters = {id(reached): reached, id(unused): unused}
    accumulator = streaming_executor._FP32GradientAccumulator(guards, parameters)
    reached_ids, cast_count = accumulator.accumulate(reached.square())
    assert reached_ids == {id(reached)}
    assert cast_count == 1
    assert reached.grad is unused.grad is None
    with pytest.raises(StreamingExecutorError, match="unused"):
        accumulator.finalize(require_all=True)
    assert reached.grad is unused.grad is None
    accumulator.clear_buffers()


class _FatalGradientManifestSignal(BaseException):
    pass


def test_base_exception_after_final_commit_preserves_signal_and_clears_created_gradients(
    monkeypatch: pytest.MonkeyPatch,
    bundle: ExecutionBundle,
    tokenizer: DenseCharacterChatTokenizer,
    compiler: FragmentActionTokenCompiler,
) -> None:
    provider, model = _provider(tokenizer, bundle.episodes, bundle.artifact_root)
    original = streaming_executor._gradient_byte_records
    signal = _FatalGradientManifestSignal()
    calls = 0

    def interrupt_second_manifest(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise signal
        return original(*args, **kwargs)

    monkeypatch.setattr(streaming_executor, "_gradient_byte_records", interrupt_second_manifest)
    with pytest.raises(_FatalGradientManifestSignal) as caught:
        execute_verified_streaming_backward(
            bundle.plan,
            bundle.records,
            bundle.episodes,
            provider,
            tokenizer,
            compiler,
            tuple(model.named_parameters()),
            controllers=bundle.controllers,
        )
    assert caught.value is signal
    assert calls == 2
    assert all(parameter.grad is None for parameter in model.parameters())
