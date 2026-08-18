from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import fields, is_dataclass
from fractions import Fraction
from typing import cast

import pytest
import torch

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
from goalzendo_interactive.authenticated_rollouts_v2 import (
    ModelPolicyProvenance,
    VerifiedAuthenticatedRolloutGroup,
    collect_eight_authenticated_rollouts,
)
from goalzendo_interactive.episodes import HiddenEpisode, terminal_classifications
from goalzendo_interactive.objectives import trajectory_policy_gradient_objective
from goalzendo_interactive.streaming_objective_v3 import (
    STREAMING_OBJECTIVE_CONTRACT_ID,
    STREAMING_OBJECTIVE_SCHEMA_VERSION,
    ExactScalar,
    StreamingObjectivePlan,
    StreamingObjectivePlanError,
    derive_verified_streaming_objective_plan,
    parse_streaming_objective_plan,
    streaming_objective_manifest,
)
from goalzendo_interactive.transcripts import AbortReason


def _digest(label: str) -> str:
    import hashlib

    return hashlib.sha256(label.encode("ascii")).hexdigest()


class DenseCharacterChatTokenizer:
    """Small exact-decode tokenizer for authenticated integration tests."""

    def __init__(self) -> None:
        alphabet = (*tuple(chr(index) for index in range(128)), "—")
        self._to_id = {character: index for index, character in enumerate(alphabet)}
        self._to_character = alphabet

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
        return [self._to_id[character] for character in text]

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return "".join(self._to_character[token_id] for token_id in token_ids)


class ScriptedProvenancedProvider:
    """Force ready/answer actions while exposing immutable provenance."""

    def __init__(
        self,
        tokenizer: DenseCharacterChatTokenizer,
        episode: HiddenEpisode,
        provenance: ModelPolicyProvenance,
    ) -> None:
        self._tokenizer = tokenizer
        self._episode = episode
        self._policy_state_digest = provenance.policy_state_digest
        self._model_provenance_digest = provenance.digest

    @property
    def policy_state_digest(self) -> str:
        return self._policy_state_digest

    @property
    def model_provenance_digest(self) -> str:
        return self._model_provenance_digest

    def _action_for_prompt(self, prompt: str) -> str:
        if "Inquiry is over." in prompt[prompt.rfind("<|user|>") :]:
            return serialize_action(
                AnswerAction(
                    self._episode.target.rule,
                    cast(tuple[Classification, ...], terminal_classifications(self._episode)),
                )
            )
        return serialize_action(ReadyAction())

    def _next_action_token(self, input_ids: tuple[int, ...]) -> int:
        text = self._tokenizer.decode(input_ids)
        marker = "<|assistant|>\n"
        marker_index = text.rfind(marker)
        if marker_index < 0:
            return 0
        action_start = marker_index + len(marker)
        prompt = text[:action_start]
        partial = text[action_start:]
        action_ids = self._tokenizer.encode(self._action_for_prompt(prompt))
        return 0 if len(partial) >= len(action_ids) else action_ids[len(partial)]

    def next_token_logits(self, input_ids: tuple[int, ...]) -> torch.Tensor:
        result = torch.full((256,), -80.0, dtype=torch.float32)
        result[self._next_action_token(input_ids)] = 80.0
        return result

    def full_forward_logits(self, input_ids: tuple[int, ...]) -> torch.Tensor:
        result = torch.full((len(input_ids), 256), -80.0, dtype=torch.float32)
        result[:, 0] = 80.0
        text = self._tokenizer.decode(input_ids)
        marker = "<|assistant|>\n"
        marker_index = text.rfind(marker)
        if marker_index < 0:
            raise AssertionError("generation prompt marker is missing")
        action_start = marker_index + len(marker)
        prompt = text[:action_start]
        action_ids = self._tokenizer.encode(self._action_for_prompt(prompt))
        for token_index, token_id in enumerate(action_ids):
            prediction_index = action_start + token_index - 1
            result[prediction_index].fill_(-80.0)
            result[prediction_index, token_id] = 80.0
        result.requires_grad_(True)
        return result


class SelectedAbortController:
    def __init__(self, rollout_indices: frozenset[int]) -> None:
        self._rollout_indices = rollout_indices

    def abort_reason(
        self,
        *,
        episode_digest: str,
        rollout_index: int,
        turn_index: int,
        dialogue_digest: str,
    ) -> AbortReason | None:
        assert len(episode_digest) == len(dialogue_digest) == 64
        if rollout_index in self._rollout_indices and turn_index == 0:
            return "timed_out"
        return None


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


@pytest.fixture(scope="module")
def provenance() -> ModelPolicyProvenance:
    return ModelPolicyProvenance(
        model_identifier="test/scripted-policy",
        revision="5" * 40,
        artifact_manifest_sha256="6" * 64,
        runtime_stack_sha256="7" * 64,
        policy_state_digest=_digest("stable-scripted-policy"),
    )


def _collect_group(
    tokenizer: DenseCharacterChatTokenizer,
    compiler: FragmentActionTokenCompiler,
    episode: HiddenEpisode,
    provenance: ModelPolicyProvenance,
    *,
    aborted: frozenset[int],
    temperature: float = 0.75,
) -> VerifiedAuthenticatedRolloutGroup:
    return collect_eight_authenticated_rollouts(
        ScriptedProvenancedProvider(tokenizer, episode, provenance),
        tokenizer,
        compiler,
        episode,
        provenance,
        run_seed=20260811,
        temperature=temperature,
        absolute_tolerance=0,
        maximum_sequence_tokens=16_384,
        schedule_order=(6, 3, 1, 7, 0, 5, 2, 4),
        controller=SelectedAbortController(aborted),
    )


@pytest.fixture(scope="module")
def mixed_group(
    tokenizer: DenseCharacterChatTokenizer,
    compiler: FragmentActionTokenCompiler,
    hidden_episode: HiddenEpisode,
    provenance: ModelPolicyProvenance,
) -> VerifiedAuthenticatedRolloutGroup:
    return _collect_group(
        tokenizer,
        compiler,
        hidden_episode,
        provenance,
        aborted=frozenset({3}),
    )


@pytest.fixture(scope="module")
def all_empty_group(
    tokenizer: DenseCharacterChatTokenizer,
    compiler: FragmentActionTokenCompiler,
    hidden_episode: HiddenEpisode,
    provenance: ModelPolicyProvenance,
) -> VerifiedAuthenticatedRolloutGroup:
    return _collect_group(
        tokenizer,
        compiler,
        hidden_episode,
        provenance,
        aborted=frozenset(range(8)),
    )


@pytest.fixture(scope="module")
def second_empty_group(
    tokenizer: DenseCharacterChatTokenizer,
    compiler: FragmentActionTokenCompiler,
    noisy_episode: HiddenEpisode,
    provenance: ModelPolicyProvenance,
) -> VerifiedAuthenticatedRolloutGroup:
    return _collect_group(
        tokenizer,
        compiler,
        noisy_episode,
        provenance,
        aborted=frozenset(range(8)),
    )


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


def _as_float(value: ExactScalar) -> float:
    return value.numerator / value.denominator


def _dense_inputs(
    plan: StreamingObjectivePlan,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[str, ...],
    tuple[str, ...],
    torch.Tensor,
    torch.Tensor,
]:
    rollouts = tuple(rollout for group in plan.groups for rollout in group.rollouts)
    maximum = max(1, *(rollout.action_token_count for rollout in rollouts))
    raw = torch.arange(1, len(rollouts) * maximum + 1, dtype=torch.float64).reshape(len(rollouts), maximum)
    log_probabilities = (-raw / 997.0).requires_grad_()
    entropies = (raw / 991.0).requires_grad_()
    mask = torch.zeros_like(log_probabilities, dtype=torch.bool)
    for row, rollout in enumerate(rollouts):
        mask[row, : rollout.action_token_count] = True
    rewards = torch.tensor([_as_float(rollout.reward) for rollout in rollouts], dtype=torch.float64)
    group_ids = torch.tensor(
        [group_index for group_index in range(plan.group_count) for _ in range(8)],
        dtype=torch.int64,
    )
    trajectory_digests = tuple(rollout.record_digest for rollout in rollouts)
    episode_digests = tuple(group.episode_digest for group in plan.groups for _ in range(8))
    zero_abort_mask = torch.tensor(
        [rollout.turn_zero_abort is not None for rollout in rollouts], dtype=torch.bool
    )
    return (
        log_probabilities,
        mask,
        rewards,
        group_ids,
        trajectory_digests,
        episode_digests,
        zero_abort_mask,
        entropies,
    )


def _streaming_loss(
    plan: StreamingObjectivePlan,
    log_probabilities: torch.Tensor,
    entropies: torch.Tensor,
) -> torch.Tensor:
    result = log_probabilities.sum() * 0.0
    row = 0
    for group in plan.groups:
        for rollout in group.rollouts:
            offset = 0
            for turn in rollout.turns:
                end = offset + turn.action_token_count
                result = (
                    result + _as_float(turn.policy_coefficient) * log_probabilities[row, offset:end].sum()
                )
                result = result + _as_float(turn.entropy_token_coefficient) * entropies[row, offset:end].sum()
                offset = end
            assert offset == rollout.action_token_count
            row += 1
    return result


def test_exact_mixed_group_plan_retains_abort_and_discards_graphs(
    mixed_group: VerifiedAuthenticatedRolloutGroup,
) -> None:
    verified = derive_verified_streaming_objective_plan([mixed_group], entropy_coefficient=Fraction(1, 8))
    plan = verified.plan
    aborted = plan.groups[0].rollouts[3]

    assert plan.schema_version == STREAMING_OBJECTIVE_SCHEMA_VERSION == 3
    assert plan.contract_id == STREAMING_OBJECTIVE_CONTRACT_ID
    assert plan.group_count == 1
    assert plan.rollout_count == 8
    assert plan.optimizer_step_eligible is True
    assert aborted.turn_zero_abort is not None
    assert aborted.turn_zero_abort.digest
    assert aborted.turns == ()
    assert aborted.action_token_count == 0
    assert aborted.reward.fraction == 0
    assert aborted.leave_one_out_advantage.fraction == -1
    assert aborted.policy_coefficient.fraction == Fraction(1, 8)
    for rollout in plan.groups[0].rollouts:
        if rollout.rollout_index != 3:
            assert rollout.reward.fraction == 1
            assert rollout.leave_one_out_advantage.fraction == Fraction(1, 7)
            assert rollout.policy_coefficient.fraction == Fraction(-1, 56)
    assert plan.entropy_token_coefficient.fraction == Fraction(
        -1, 8 * plan.total_authenticated_action_token_count
    )
    assert plan.groups[0].group_digest == mixed_group.digest
    assert plan.groups[0].action_evidence_group_digest == mixed_group.action_evidence_group_digest
    assert verified.verification_digest
    assert not _contains_tensor(verified)


def test_multigroup_streaming_scalar_replay_matches_dense_value_and_gradients(
    mixed_group: VerifiedAuthenticatedRolloutGroup,
    second_empty_group: VerifiedAuthenticatedRolloutGroup,
) -> None:
    plan = derive_verified_streaming_objective_plan(
        [mixed_group, second_empty_group], entropy_coefficient=Fraction(1, 8)
    ).plan
    (
        log_probabilities,
        mask,
        rewards,
        group_ids,
        trajectory_digests,
        episode_digests,
        zero_abort_mask,
        entropies,
    ) = _dense_inputs(plan)
    mixed_instruction = next(group for group in plan.groups if group.group_digest == mixed_group.digest)
    assert mixed_instruction.rollouts[3].policy_coefficient.fraction == Fraction(1, 16)
    assert all(
        rollout.policy_coefficient.fraction == Fraction(-1, 112)
        for rollout in mixed_instruction.rollouts
        if rollout.rollout_index != 3
    )
    dense = trajectory_policy_gradient_objective(
        log_probabilities,
        mask,
        rewards,
        group_ids,
        trajectory_digests,
        episode_digests=episode_digests,
        authenticated_zero_action_abort_mask=zero_abort_mask,
        token_entropies=entropies,
        entropy_coefficient=0.125,
    )
    streaming = _streaming_loss(plan, log_probabilities, entropies)

    expected_advantages = torch.tensor(
        [_as_float(rollout.leave_one_out_advantage) for group in plan.groups for rollout in group.rollouts],
        dtype=torch.float64,
    )
    assert torch.allclose(dense.advantages, expected_advantages, rtol=1e-15, atol=1e-15)
    assert torch.allclose(streaming, dense.loss, rtol=1e-13, atol=1e-13)
    dense_gradients = torch.autograd.grad(dense.loss, (log_probabilities, entropies), retain_graph=True)
    streaming_gradients = torch.autograd.grad(streaming, (log_probabilities, entropies))
    for dense_gradient, streaming_gradient in zip(dense_gradients, streaming_gradients, strict=True):
        assert torch.allclose(dense_gradient, streaming_gradient, rtol=1e-13, atol=1e-13)


def test_all_empty_group_is_finite_ineligible_and_matches_dense_zero_gradient(
    all_empty_group: VerifiedAuthenticatedRolloutGroup,
) -> None:
    plan = derive_verified_streaming_objective_plan(
        [all_empty_group], entropy_coefficient=Fraction(3, 16)
    ).plan
    assert plan.total_authenticated_action_token_count == 0
    assert plan.entropy_token_coefficient.fraction == 0
    assert plan.optimizer_step_eligible is False
    assert all(
        rollout.turn_zero_abort is not None
        and rollout.reward.fraction == 0
        and rollout.leave_one_out_advantage.fraction == 0
        and rollout.policy_coefficient.fraction == 0
        and rollout.action_token_count == 0
        for rollout in plan.groups[0].rollouts
    )

    (
        log_probabilities,
        mask,
        rewards,
        group_ids,
        trajectory_digests,
        episode_digests,
        zero_abort_mask,
        entropies,
    ) = _dense_inputs(plan)
    dense = trajectory_policy_gradient_objective(
        log_probabilities,
        mask,
        rewards,
        group_ids,
        trajectory_digests,
        episode_digests=episode_digests,
        authenticated_zero_action_abort_mask=zero_abort_mask,
        token_entropies=entropies,
        entropy_coefficient=3 / 16,
    )
    streaming = _streaming_loss(plan, log_probabilities, entropies)
    assert torch.isfinite(dense.loss)
    assert torch.equal(streaming, dense.loss)
    dense_gradients = torch.autograd.grad(
        dense.loss,
        (log_probabilities, entropies),
        allow_unused=True,
        retain_graph=True,
    )
    streaming_gradients = torch.autograd.grad(
        streaming,
        (log_probabilities, entropies),
        allow_unused=True,
    )
    assert dense_gradients[0] is not None
    assert streaming_gradients[0] is not None
    assert torch.equal(dense_gradients[0], streaming_gradients[0])
    assert torch.count_nonzero(dense_gradients[0]).item() == 0
    assert dense_gradients[1] is streaming_gradients[1] is None


def test_canonical_round_trip_is_structural_only_and_tampering_fails(
    mixed_group: VerifiedAuthenticatedRolloutGroup,
) -> None:
    verified = derive_verified_streaming_objective_plan([mixed_group])
    text = verified.plan.to_json()
    parsed = parse_streaming_objective_plan(text)

    assert parsed == verified.plan
    assert type(parsed) is StreamingObjectivePlan
    assert not hasattr(parsed, "verification_digest")
    canonical_payload = json.loads(text)
    assert canonical_payload["execution_contract"] == {
        "reauthenticate_each_stored_action": True,
        "bounded_graph_scope": "one_authenticated_turn",
        "parameter_mutation_between_replays": False,
        "optimizer_state_mutation_between_replays": False,
        "optimizer_step_after_all_entries_only": True,
    }
    reordered = json.dumps(json.loads(text), sort_keys=True, separators=(",", ":"))
    assert reordered != text
    with pytest.raises(StreamingObjectivePlanError, match="canonical byte representation"):
        parse_streaming_objective_plan(reordered)
    duplicate_digest = text[:-1] + f',"digest":"{"0" * 64}"}}'
    with pytest.raises(StreamingObjectivePlanError, match="strict JSON"):
        parse_streaming_objective_plan(duplicate_digest)
    payload = json.loads(text)
    payload["authorization"]["optimizer_step"] = True
    with pytest.raises(StreamingObjectivePlanError, match="cannot carry authorization"):
        parse_streaming_objective_plan(json.dumps(payload, separators=(",", ":")))
    payload = json.loads(text)
    payload["execution_contract"]["parameter_mutation_between_replays"] = True
    with pytest.raises(StreamingObjectivePlanError, match="execution contract changed"):
        parse_streaming_objective_plan(json.dumps(payload, separators=(",", ":")))

    wrong_nominal_type = cast(VerifiedAuthenticatedRolloutGroup, mixed_group.rollouts[0])
    with pytest.raises(StreamingObjectivePlanError, match="nominal wrappers"):
        derive_verified_streaming_objective_plan([wrong_nominal_type])


def test_group_order_is_canonical_and_duplicate_episode_or_drift_fails(
    tokenizer: DenseCharacterChatTokenizer,
    compiler: FragmentActionTokenCompiler,
    noisy_episode: HiddenEpisode,
    provenance: ModelPolicyProvenance,
    all_empty_group: VerifiedAuthenticatedRolloutGroup,
    second_empty_group: VerifiedAuthenticatedRolloutGroup,
) -> None:
    forward = derive_verified_streaming_objective_plan(
        [all_empty_group, second_empty_group], entropy_coefficient=0.25
    ).plan
    reverse = derive_verified_streaming_objective_plan(
        [second_empty_group, all_empty_group], entropy_coefficient=0.25
    ).plan
    assert forward.to_json() == reverse.to_json()

    with pytest.raises(StreamingObjectivePlanError, match="same hidden episode"):
        derive_verified_streaming_objective_plan([all_empty_group, all_empty_group])

    drifted = _collect_group(
        tokenizer,
        compiler,
        noisy_episode,
        provenance,
        aborted=frozenset(range(8)),
        temperature=0.5,
    )
    with pytest.raises(StreamingObjectivePlanError, match="share policy, provenance, and runtime"):
        derive_verified_streaming_objective_plan([all_empty_group, drifted])


def test_manifest_is_graph_free_and_nonauthorizing() -> None:
    manifest = streaming_objective_manifest()
    assert manifest["schema_version"] == 3
    assert manifest["retained_collection_graphs"] is False
    assert manifest["parsed_plan_is_verified"] is False
    assert manifest["production_executor_present"] is False
    assert manifest["model_load_authorization"] is False
    assert manifest["rollout_launch_authorization"] is False
    assert manifest["weight_update_authorization"] is False
    assert manifest["optimizer_step_authorization"] is False
