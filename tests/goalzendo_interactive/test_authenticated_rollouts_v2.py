from __future__ import annotations

import gc
import hashlib
import json
import weakref
from collections.abc import Sequence
from dataclasses import fields, is_dataclass, replace
from typing import cast

import pytest
import torch

import goalzendo_interactive.authenticated_rollouts_v2 as authenticated_rollouts
from goalzendo_interactive.action_tokenization_v2 import (
    FragmentActionTokenCompiler,
    TokenizerBindingManifest,
)
from goalzendo_interactive.actions import (
    AnswerAction,
    Classification,
    ReadyAction,
    TestAction,
    serialize_action,
)
from goalzendo_interactive.authenticated_rollouts_v2 import (
    AUTHENTICATED_ROLLOUT_CONTRACT_ID,
    AUTHENTICATED_ROLLOUT_SCHEMA_VERSION,
    AuthenticatedRolloutError,
    AuthenticatedRolloutRecord,
    ModelPolicyProvenance,
    VerifiedAuthenticatedRollout,
    authenticated_rollout_manifest,
    collect_authenticated_rollout,
    collect_eight_authenticated_rollouts,
    parse_authenticated_rollout,
    replay_authenticated_rollout,
    verify_eight_authenticated_rollout_group,
)
from goalzendo_interactive.authenticated_sampler_v2 import (
    GraphFreeVerifiedAuthenticatedActionSample,
    VerifiedAuthenticatedActionSample,
)
from goalzendo_interactive.episodes import HiddenEpisode, terminal_classifications
from goalzendo_interactive.schema import scene_at
from goalzendo_interactive.transcripts import (
    AbortEvent,
    AbortReason,
    AnswerEvent,
    ReadyEvent,
    TestEvent,
    Transcript,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


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


class DenseCharacterChatTokenizer:
    """One token per character with a compact test-only vocabulary."""

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
    """Deterministically forces a canonical action while retaining finite logits."""

    def __init__(
        self,
        tokenizer: DenseCharacterChatTokenizer,
        episode: HiddenEpisode,
        provenance: ModelPolicyProvenance,
        *,
        strategy: str = "ready",
    ) -> None:
        self._tokenizer = tokenizer
        self._episode = episode
        self._policy_state_digest = provenance.policy_state_digest
        self._model_provenance_digest = provenance.digest
        self._strategy = strategy
        self.next_call_count = 0

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
        if self._strategy == "ready":
            return serialize_action(ReadyAction())
        completed_tests = prompt.count('<|assistant|>\n{"move":"test"')
        return serialize_action(TestAction(scene_at(200 + completed_tests)))

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
        if len(partial) >= len(action_ids):
            return 0
        return action_ids[len(partial)]

    def next_token_logits(self, input_ids: tuple[int, ...]) -> torch.Tensor:
        self.next_call_count += 1
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


class MutatingPolicyProvider(ScriptedProvenancedProvider):
    def next_token_logits(self, input_ids: tuple[int, ...]) -> torch.Tensor:
        result = super().next_token_logits(input_ids)
        self._policy_state_digest = _digest("mutated-policy")
        return result


class FixedAbortController:
    def __init__(self, reason: AbortReason, *, at_turn: int = 0) -> None:
        self._reason = reason
        self._at_turn = at_turn

    def abort_reason(
        self,
        *,
        episode_digest: str,
        rollout_index: int,
        turn_index: int,
        dialogue_digest: str,
    ) -> AbortReason | None:
        assert len(episode_digest) == 64
        assert 0 <= rollout_index < 8
        assert len(dialogue_digest) == 64
        return self._reason if turn_index == self._at_turn else None


class OneRolloutAbortController:
    def __init__(self, rollout_index: int, reason: AbortReason = "timed_out") -> None:
        self._rollout_index = rollout_index
        self._reason = reason

    def abort_reason(
        self,
        *,
        episode_digest: str,
        rollout_index: int,
        turn_index: int,
        dialogue_digest: str,
    ) -> AbortReason | None:
        assert len(episode_digest) == len(dialogue_digest) == 64
        if rollout_index == self._rollout_index and turn_index == 0:
            return self._reason
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
        special_token_ids=(
            ("bos", None),
            ("eos", None),
            ("pad", None),
            ("unk", None),
        ),
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


def _collect(
    tokenizer: DenseCharacterChatTokenizer,
    compiler: FragmentActionTokenCompiler,
    hidden_episode: HiddenEpisode,
    provenance: ModelPolicyProvenance,
    *,
    strategy: str = "ready",
    rollout_index: int = 0,
    maximum_sequence_tokens: int = 16_384,
    maximum_turns: int = 8,
    controller: FixedAbortController | None = None,
) -> tuple[VerifiedAuthenticatedRollout, ScriptedProvenancedProvider]:
    provider = ScriptedProvenancedProvider(
        tokenizer,
        hidden_episode,
        provenance,
        strategy=strategy,
    )
    rollout = collect_authenticated_rollout(
        provider,
        tokenizer,
        compiler,
        hidden_episode,
        provenance,
        run_seed=20260811,
        rollout_index=rollout_index,
        temperature=0.75,
        absolute_tolerance=0,
        maximum_sequence_tokens=maximum_sequence_tokens,
        maximum_turns=maximum_turns,
        controller=controller,
    )
    return rollout, provider


def test_ready_answer_completion_round_trip_and_live_replay(
    tokenizer: DenseCharacterChatTokenizer,
    compiler: FragmentActionTokenCompiler,
    hidden_episode: HiddenEpisode,
    provenance: ModelPolicyProvenance,
) -> None:
    verified, _ = _collect(tokenizer, compiler, hidden_episode, provenance)
    record = verified.record

    assert record.schema_version == AUTHENTICATED_ROLLOUT_SCHEMA_VERSION == 2
    assert record.contract_id == AUTHENTICATED_ROLLOUT_CONTRACT_ID
    assert [type(turn.event) for turn in record.turns] == [ReadyEvent, AnswerEvent]
    assert [turn.turn_index for turn in record.turns] == [0, 1]
    assert [turn.sample.turn_seed.turn_index for turn in record.turns] == [0, 1]
    assert all(
        turn.nominal_verification_digest == evidence.verification_digest
        for turn, evidence in zip(record.turns, verified.verified_samples, strict=True)
    )
    assert record.transcript.state == "complete"
    assert record.termination is None
    assert record.reward == 1.0

    parsed = parse_authenticated_rollout(record.to_json())
    assert parsed == record
    replay_provider = ScriptedProvenancedProvider(tokenizer, hidden_episode, provenance)
    replayed = replay_authenticated_rollout(
        parsed,
        replay_provider,
        tokenizer,
        compiler,
        hidden_episode,
    )
    assert replayed.verification_digest == verified.verification_digest

    aborting_controller = FixedAbortController("timed_out")
    with pytest.raises(AuthenticatedRolloutError, match="before a recorded policy action"):
        replay_authenticated_rollout(
            parsed,
            ScriptedProvenancedProvider(tokenizer, hidden_episode, provenance),
            tokenizer,
            compiler,
            hidden_episode,
            controller=aborting_controller,
        )


def test_six_tests_force_budget_exhaustion_then_answer(
    tokenizer: DenseCharacterChatTokenizer,
    compiler: FragmentActionTokenCompiler,
    hidden_episode: HiddenEpisode,
    provenance: ModelPolicyProvenance,
) -> None:
    verified, _ = _collect(
        tokenizer,
        compiler,
        hidden_episode,
        provenance,
        strategy="tests",
    )
    events = verified.record.transcript.events
    assert len(verified.record.turns) == 7
    assert all(type(event) is TestEvent for event in events[:6])
    budget_event = events[5]
    assert type(budget_event) is TestEvent
    assert budget_event.outcome == "budget_exhausted"
    assert type(events[-1]) is AnswerEvent
    assert verified.record.transcript.state == "complete"
    assert verified.record.reward == 0.95


@pytest.mark.parametrize(
    ("reason", "expected_origin"),
    (("timed_out", "controller"), ("incomplete", "controller")),
)
def test_controller_timeout_and_cancellation_are_replayable_zero_reward_aborts(
    tokenizer: DenseCharacterChatTokenizer,
    compiler: FragmentActionTokenCompiler,
    hidden_episode: HiddenEpisode,
    provenance: ModelPolicyProvenance,
    reason: AbortReason,
    expected_origin: str,
) -> None:
    verified, provider = _collect(
        tokenizer,
        compiler,
        hidden_episode,
        provenance,
        controller=FixedAbortController(reason),
    )
    record = verified.record
    assert provider.next_call_count == 0
    assert record.turns == ()
    assert record.termination is not None
    assert record.termination.origin == expected_origin
    assert record.termination.reason == reason
    assert record.transcript.state == "aborted"
    assert record.reward_numerator == 0
    assert record.reward_denominator == 1

    replay_provider = ScriptedProvenancedProvider(tokenizer, hidden_episode, provenance)
    with pytest.raises(AuthenticatedRolloutError, match="requires its live controller"):
        replay_authenticated_rollout(
            AuthenticatedRolloutRecord.from_json(record.to_json()),
            replay_provider,
            tokenizer,
            compiler,
            hidden_episode,
        )
    replayed = replay_authenticated_rollout(
        AuthenticatedRolloutRecord.from_json(record.to_json()),
        replay_provider,
        tokenizer,
        compiler,
        hidden_episode,
        controller=FixedAbortController(reason),
    )
    assert replayed.record == record
    assert replayed.verified_samples == ()

    if reason == "timed_out":
        assert record.termination is not None
        forged_termination = replace(record.termination, reason="incomplete")
        forged_transcript = Transcript(
            episode_digest=record.episode_digest,
            opening=record.transcript.opening,
            events=(AbortEvent("incomplete"),),
            state="aborted",
        )
        forged_record = replace(
            record,
            termination=forged_termination,
            transcript=forged_transcript,
            transcript_digest=forged_transcript.digest,
        )
        with pytest.raises(AuthenticatedRolloutError, match="does not regenerate"):
            replay_authenticated_rollout(
                AuthenticatedRolloutRecord.from_json(forged_record.to_json()),
                ScriptedProvenancedProvider(tokenizer, hidden_episode, provenance),
                tokenizer,
                compiler,
                hidden_episode,
                controller=FixedAbortController("timed_out"),
            )


def test_sampler_overlength_and_turn_budget_incomplete_abort_without_repair(
    tokenizer: DenseCharacterChatTokenizer,
    compiler: FragmentActionTokenCompiler,
    hidden_episode: HiddenEpisode,
    provenance: ModelPolicyProvenance,
) -> None:
    overlength, _ = _collect(
        tokenizer,
        compiler,
        hidden_episode,
        provenance,
        maximum_sequence_tokens=1,
    )
    assert overlength.record.termination is not None
    assert overlength.record.termination.origin == "sampler"
    assert overlength.record.termination.reason == "overlength"
    assert overlength.record.turns == ()
    assert overlength.record.reward == 0
    replay_provider = ScriptedProvenancedProvider(tokenizer, hidden_episode, provenance)
    assert (
        replay_authenticated_rollout(
            AuthenticatedRolloutRecord.from_json(overlength.record.to_json()),
            replay_provider,
            tokenizer,
            compiler,
            hidden_episode,
        ).record
        == overlength.record
    )

    forged_overlength = replace(overlength.record, maximum_sequence_tokens=16_384)
    forged_provider = ScriptedProvenancedProvider(tokenizer, hidden_episode, provenance)
    with pytest.raises(AuthenticatedRolloutError, match="did not reproduce"):
        replay_authenticated_rollout(
            AuthenticatedRolloutRecord.from_json(forged_overlength.to_json()),
            forged_provider,
            tokenizer,
            compiler,
            hidden_episode,
        )

    incomplete, _ = _collect(
        tokenizer,
        compiler,
        hidden_episode,
        provenance,
        strategy="tests",
        maximum_turns=1,
    )
    assert len(incomplete.record.turns) == 1
    assert incomplete.record.termination is not None
    assert incomplete.record.termination.origin == "turn_budget"
    assert incomplete.record.termination.reason == "incomplete"
    assert incomplete.record.termination.turn_index == 1
    assert incomplete.record.reward == 0
    assert (
        replay_authenticated_rollout(
            AuthenticatedRolloutRecord.from_json(incomplete.record.to_json()),
            ScriptedProvenancedProvider(
                tokenizer,
                hidden_episode,
                provenance,
                strategy="tests",
            ),
            tokenizer,
            compiler,
            hidden_episode,
        ).record
        == incomplete.record
    )

    with pytest.raises(AuthenticatedRolloutError, match="unregistered abort reason"):
        _collect(
            tokenizer,
            compiler,
            hidden_episode,
            provenance,
            controller=FixedAbortController("overlength"),
        )


def test_tamper_and_policy_mutation_fail_closed(
    tokenizer: DenseCharacterChatTokenizer,
    compiler: FragmentActionTokenCompiler,
    hidden_episode: HiddenEpisode,
    provenance: ModelPolicyProvenance,
) -> None:
    verified, _ = _collect(tokenizer, compiler, hidden_episode, provenance)
    record = verified.record
    payload = json.loads(record.to_json())
    payload["turns"][0]["sample"]["selected_token_ids"][0] ^= 1
    tampered = json.dumps(payload, separators=(",", ":"))
    with pytest.raises(AuthenticatedRolloutError):
        parse_authenticated_rollout(tampered)

    reordered = json.dumps(json.loads(record.to_json()), sort_keys=True, separators=(",", ":"))
    assert reordered != record.to_json()
    with pytest.raises(AuthenticatedRolloutError):
        parse_authenticated_rollout(reordered)

    changed_turn = replace(record.turns[0], nominal_verification_digest="f" * 64)
    structurally_valid_forgery = replace(record, turns=(changed_turn, *record.turns[1:]))
    replay_provider = ScriptedProvenancedProvider(tokenizer, hidden_episode, provenance)
    with pytest.raises(AuthenticatedRolloutError, match="verification digest"):
        replay_authenticated_rollout(
            structurally_valid_forgery,
            replay_provider,
            tokenizer,
            compiler,
            hidden_episode,
        )

    mutating = MutatingPolicyProvider(tokenizer, hidden_episode, provenance)
    with pytest.raises(AuthenticatedRolloutError, match="fatal authenticated sampling"):
        collect_authenticated_rollout(
            mutating,
            tokenizer,
            compiler,
            hidden_episode,
            provenance,
            run_seed=20260811,
            rollout_index=0,
            temperature=0.75,
            absolute_tolerance=0,
            maximum_sequence_tokens=16_384,
        )


def test_eight_rollout_group_uses_verified_evidence_and_is_schedule_invariant(
    monkeypatch: pytest.MonkeyPatch,
    tokenizer: DenseCharacterChatTokenizer,
    compiler: FragmentActionTokenCompiler,
    hidden_episode: HiddenEpisode,
    provenance: ModelPolicyProvenance,
) -> None:
    replay_root_references: list[weakref.ReferenceType[torch.Tensor]] = []
    compact = authenticated_rollouts._compact_verified_action_evidence

    def observe_compaction(
        verified: VerifiedAuthenticatedActionSample,
    ) -> tuple[
        GraphFreeVerifiedAuthenticatedActionSample,
        tuple[weakref.ReferenceType[torch.Tensor], ...],
    ]:
        evidence, references = compact(verified)
        replay_root_references.extend(references)
        return evidence, references

    monkeypatch.setattr(authenticated_rollouts, "_compact_verified_action_evidence", observe_compaction)
    provider = ScriptedProvenancedProvider(tokenizer, hidden_episode, provenance)
    schedule = (5, 1, 7, 0, 3, 6, 2, 4)
    group = collect_eight_authenticated_rollouts(
        provider,
        tokenizer,
        compiler,
        hidden_episode,
        provenance,
        run_seed=20260811,
        temperature=0.75,
        absolute_tolerance=0,
        maximum_sequence_tokens=16_384,
        schedule_order=schedule,
    )
    assert tuple(rollout.record.rollout_index for rollout in group.rollouts) == tuple(range(8))
    assert verify_eight_authenticated_rollout_group(tuple(reversed(group.rollouts))).digest == group.digest
    assert len(group.action_evidence_group_digest) == 64
    assert all(len(rollout.verified_samples) == 2 for rollout in group.rollouts)
    assert all(
        type(evidence) is GraphFreeVerifiedAuthenticatedActionSample
        for rollout in group.rollouts
        for evidence in rollout.verified_samples
    )
    assert len(replay_root_references) == 5 * sum(len(rollout.record.turns) for rollout in group.rollouts)
    gc.collect()
    assert all(reference() is None for reference in replay_root_references)
    assert not _contains_tensor(group)

    raw_records = cast(
        tuple[VerifiedAuthenticatedRollout, ...],
        tuple(rollout.record for rollout in group.rollouts),
    )
    with pytest.raises(AuthenticatedRolloutError, match="replay-verified"):
        verify_eight_authenticated_rollout_group(raw_records)
    with pytest.raises(AuthenticatedRolloutError, match="indices zero through seven"):
        verify_eight_authenticated_rollout_group((*group.rollouts[:-1], group.rollouts[6]))


def test_eight_rollout_group_retains_authenticated_zero_turn_abort(
    tokenizer: DenseCharacterChatTokenizer,
    compiler: FragmentActionTokenCompiler,
    hidden_episode: HiddenEpisode,
    provenance: ModelPolicyProvenance,
) -> None:
    provider = ScriptedProvenancedProvider(tokenizer, hidden_episode, provenance)
    controller = OneRolloutAbortController(3)
    group = collect_eight_authenticated_rollouts(
        provider,
        tokenizer,
        compiler,
        hidden_episode,
        provenance,
        run_seed=20260811,
        temperature=0.75,
        absolute_tolerance=0,
        maximum_sequence_tokens=16_384,
        schedule_order=(6, 3, 1, 7, 0, 5, 2, 4),
        controller=controller,
    )
    aborted = group.rollouts[3].record
    assert aborted.rollout_index == 3
    assert aborted.turns == ()
    assert aborted.termination is not None
    assert aborted.termination.reason == "timed_out"
    assert aborted.reward == 0
    assert sum(rollout.record.reward for rollout in group.rollouts) == 7.0
    assert verify_eight_authenticated_rollout_group(tuple(reversed(group.rollouts))).digest == group.digest


def test_manifest_keeps_every_authorization_false() -> None:
    manifest = authenticated_rollout_manifest()
    assert manifest["schema_version"] == 2
    assert manifest["accepted_turn_evidence"] == "GraphFreeVerifiedAuthenticatedActionSample only"
    assert manifest["retained_differentiable_graphs"] is False
    assert manifest["accepted_empty_evidence"] == "live-verified turn-zero registered abort only"
    assert manifest["schedule_order_dependent"] is False
    assert manifest["optimizer_step_hook_present"] is False
    assert manifest["live_model_authorization"] is False
    assert manifest["rollout_launch_authorization"] is False
    assert manifest["weight_update_authorization"] is False
