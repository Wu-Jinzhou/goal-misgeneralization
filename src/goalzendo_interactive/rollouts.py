"""Exact controller records for sampled G03 interactive trajectories."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal, Protocol, TypeAlias, cast, runtime_checkable

from ._json import CanonicalJSONError, dump_json, json_digest, load_json
from .actions import serialize_action
from .dialogue import Dialogue, dialogue_as_obj, render_dialogue
from .environment import HiddenLawEnvironment, replay_transcript
from .episodes import HiddenEpisode
from .transcripts import (
    AbortEvent,
    AbortReason,
    AnswerEvent,
    InvalidEvent,
    ReadyEvent,
    TestEvent,
    Transcript,
    parse_transcript,
)

ROLLOUT_RECORD_SCHEMA_VERSION = 1
PolicyPhase = Literal["inquiry", "answer"]


class RolloutValidationError(ValueError):
    """Raised when a sampled rollout cannot be reconstructed exactly."""


def _valid_digest(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True, slots=True)
class PolicyTurn:
    """One model-emitted action and its on-policy token statistics."""

    raw_action: str
    token_ids: tuple[int, ...]
    token_log_probabilities: tuple[float, ...]
    token_entropies: tuple[float, ...]
    prompt_digest: str

    def __post_init__(self) -> None:
        if type(self.raw_action) is not str or not self.raw_action:
            raise RolloutValidationError("policy action text cannot be empty")
        token_ids = tuple(self.token_ids)
        log_probs = tuple(self.token_log_probabilities)
        entropies = tuple(self.token_entropies)
        object.__setattr__(self, "token_ids", token_ids)
        object.__setattr__(self, "token_log_probabilities", log_probs)
        object.__setattr__(self, "token_entropies", entropies)
        if not token_ids or len(token_ids) != len(log_probs) or len(token_ids) != len(entropies):
            raise RolloutValidationError(
                "policy token ids, log probabilities, and entropies must have equal positive length"
            )
        if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in token_ids):
            raise RolloutValidationError("policy token ids must be non-negative integers")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) > 1e-7
            for value in log_probs
        ):
            raise RolloutValidationError("policy token log probabilities must be finite and <= 0")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
            for value in entropies
        ):
            raise RolloutValidationError("policy token entropies must be finite and non-negative")
        if not _valid_digest(self.prompt_digest):
            raise RolloutValidationError("policy prompt digest must be a SHA-256 value")

    @property
    def digest(self) -> str:
        return json_digest(
            self.as_obj(), domain="goalzendo-interactive-policy-turn-v1"
        )

    def as_obj(self) -> dict[str, object]:
        return {
            "raw_action": self.raw_action,
            "token_ids": list(self.token_ids),
            "token_log_probabilities": list(self.token_log_probabilities),
            "token_entropies": list(self.token_entropies),
            "prompt_digest": self.prompt_digest,
        }


def policy_turn_from_obj(value: object) -> PolicyTurn:
    expected = {
        "raw_action",
        "token_ids",
        "token_log_probabilities",
        "token_entropies",
        "prompt_digest",
    }
    if type(value) is not dict or set(value) != expected or len(value) != len(expected):
        raise RolloutValidationError("policy-turn object has noncanonical fields")
    if any(
        type(value[name]) is not list
        for name in ("token_ids", "token_log_probabilities", "token_entropies")
    ):
        raise RolloutValidationError("policy-turn token fields must be arrays")
    result = PolicyTurn(
        raw_action=value["raw_action"],
        token_ids=tuple(value["token_ids"]),
        token_log_probabilities=tuple(value["token_log_probabilities"]),
        token_entropies=tuple(value["token_entropies"]),
        prompt_digest=value["prompt_digest"],
    )
    if result.as_obj() != value:
        raise RolloutValidationError("policy-turn object is valid but not canonical")
    return result


@dataclass(frozen=True, slots=True)
class PolicyAbort:
    """A controller-side incomplete, overlength, or timeout outcome."""

    reason: AbortReason

    def __post_init__(self) -> None:
        if self.reason not in {"incomplete", "overlength", "timed_out"}:
            raise RolloutValidationError(f"unknown policy abort reason: {self.reason!r}")


PolicyDecision: TypeAlias = PolicyTurn | PolicyAbort


@runtime_checkable
class InteractivePolicyProtocol(Protocol):
    def sample_action(
        self,
        dialogue: Dialogue,
        *,
        phase: PolicyPhase,
        terminal_count: int | None,
    ) -> PolicyDecision: ...


def dialogue_prompt_digest(dialogue: Dialogue) -> str:
    return json_digest(
        dialogue_as_obj(dialogue),
        domain="goalzendo-interactive-rollout-policy-prompt-v1",
    )


def _terminal_reward_fraction(transcript: Transcript) -> Fraction:
    final = transcript.events[-1]
    if type(final) is not AnswerEvent:
        return Fraction(0, 1)
    score = final.score
    return (
        Fraction(7 * score.classification_correct, 10 * score.classification_total)
        + Fraction(int(score.rule_equivalent), 4)
        + Fraction(1, 20) * (1 - Fraction(score.query_count, 6))
    )


@dataclass(frozen=True, slots=True)
class RolloutRecord:
    """A replayable trajectory plus every sampled action-token statistic."""

    episode_id: str
    episode_digest: str
    transcript: Transcript
    transcript_digest: str
    turns: tuple[PolicyTurn, ...]
    reward_numerator: int
    reward_denominator: int

    def __post_init__(self) -> None:
        if type(self.episode_id) is not str or not self.episode_id:
            raise RolloutValidationError("rollout episode id cannot be empty")
        if not _valid_digest(self.episode_digest) or not _valid_digest(self.transcript_digest):
            raise RolloutValidationError("rollout episode/transcript digests must be SHA-256 values")
        if type(self.transcript) is not Transcript or self.transcript.digest != self.transcript_digest:
            raise RolloutValidationError("rollout transcript digest is inconsistent")
        if self.transcript.episode_digest != self.episode_digest:
            raise RolloutValidationError("rollout transcript belongs to a different episode")
        turns = tuple(self.turns)
        object.__setattr__(self, "turns", turns)
        if any(type(turn) is not PolicyTurn for turn in turns):
            raise RolloutValidationError("rollout contains a non-policy turn")
        emitted_events = tuple(
            event
            for event in self.transcript.events
            if type(event) is not AbortEvent
        )
        if len(turns) != len(emitted_events):
            raise RolloutValidationError("policy turns do not align with transcript actions")
        for turn, event in zip(turns, emitted_events, strict=True):
            if type(event) is InvalidEvent:
                raw_action = event.raw_action
            else:
                action_event = cast(TestEvent | ReadyEvent | AnswerEvent, event)
                raw_action = serialize_action(action_event.action)
            if turn.raw_action != raw_action:
                raise RolloutValidationError("policy action text differs from transcript event")
        if (
            isinstance(self.reward_numerator, bool)
            or not isinstance(self.reward_numerator, int)
            or isinstance(self.reward_denominator, bool)
            or not isinstance(self.reward_denominator, int)
            or self.reward_denominator < 1
            or self.reward_numerator < 0
            or self.reward_numerator > self.reward_denominator
        ):
            raise RolloutValidationError("rollout reward fraction is invalid")
        expected = _terminal_reward_fraction(self.transcript)
        if (self.reward_numerator, self.reward_denominator) != (
            expected.numerator,
            expected.denominator,
        ):
            raise RolloutValidationError("rollout reward does not reconstruct from transcript")

    @property
    def reward(self) -> float:
        return self.reward_numerator / self.reward_denominator

    @property
    def token_count(self) -> int:
        return sum(len(turn.token_ids) for turn in self.turns)

    @property
    def digest(self) -> str:
        return json_digest(self.as_obj(), domain="goalzendo-interactive-rollout-record-v1")

    def as_obj(self) -> dict[str, object]:
        return {
            "schema_version": ROLLOUT_RECORD_SCHEMA_VERSION,
            "episode_id": self.episode_id,
            "episode_digest": self.episode_digest,
            "transcript": self.transcript.as_obj(),
            "transcript_digest": self.transcript_digest,
            "turns": [turn.as_obj() for turn in self.turns],
            "reward_numerator": self.reward_numerator,
            "reward_denominator": self.reward_denominator,
            "token_count": self.token_count,
        }


def rollout_record_from_obj(value: object) -> RolloutRecord:
    expected = {
        "schema_version",
        "episode_id",
        "episode_digest",
        "transcript",
        "transcript_digest",
        "turns",
        "reward_numerator",
        "reward_denominator",
        "token_count",
    }
    if type(value) is not dict or set(value) != expected or len(value) != len(expected):
        raise RolloutValidationError("rollout-record object has noncanonical fields")
    if value["schema_version"] != ROLLOUT_RECORD_SCHEMA_VERSION:
        raise RolloutValidationError("unsupported rollout-record schema version")
    if type(value["transcript"]) is not dict or type(value["turns"]) is not list:
        raise RolloutValidationError("rollout transcript/turns have invalid container types")
    try:
        transcript = parse_transcript(dump_json(value["transcript"]))
    except (CanonicalJSONError, ValueError) as exc:
        raise RolloutValidationError(f"invalid rollout transcript: {exc}") from exc
    result = RolloutRecord(
        episode_id=value["episode_id"],
        episode_digest=value["episode_digest"],
        transcript=transcript,
        transcript_digest=value["transcript_digest"],
        turns=tuple(policy_turn_from_obj(item) for item in value["turns"]),
        reward_numerator=value["reward_numerator"],
        reward_denominator=value["reward_denominator"],
    )
    if result.as_obj() != value:
        raise RolloutValidationError("rollout-record derived fields are inconsistent")
    return result


def serialize_rollout_record(record: RolloutRecord) -> str:
    if type(record) is not RolloutRecord:
        raise TypeError("serialize_rollout_record requires a RolloutRecord")
    return dump_json(record.as_obj())


def parse_rollout_record(text: str, *, require_canonical: bool = True) -> RolloutRecord:
    try:
        value = load_json(text)
    except CanonicalJSONError as exc:
        raise RolloutValidationError(str(exc)) from exc
    result = rollout_record_from_obj(value)
    if require_canonical and serialize_rollout_record(result) != text:
        raise RolloutValidationError("rollout JSON is valid but not canonical")
    return result


def verify_rollout_record(record: RolloutRecord, episode: HiddenEpisode) -> RolloutRecord:
    """Replay actions and prompt bindings against the authoritative episode."""

    if type(record) is not RolloutRecord or type(episode) is not HiddenEpisode:
        raise TypeError("verify_rollout_record requires a RolloutRecord and HiddenEpisode")
    if (record.episode_id, record.episode_digest) != (episode.episode_id, episode.digest):
        raise RolloutValidationError("rollout is bound to a different episode")
    environment = HiddenLawEnvironment(episode)
    turn_index = 0
    for expected_event in record.transcript.events:
        if type(expected_event) is AbortEvent:
            result = environment.abort(expected_event.reason)
        else:
            if turn_index >= len(record.turns):
                raise RolloutValidationError("rollout is missing a policy turn")
            turn = record.turns[turn_index]
            prompt = render_dialogue(episode, environment.transcript)
            if turn.prompt_digest != dialogue_prompt_digest(prompt):
                raise RolloutValidationError("rollout policy prompt binding does not reconstruct")
            result = environment.consume(turn.raw_action)
            turn_index += 1
        if result.event != expected_event:
            raise RolloutValidationError("rollout event differs under exact replay")
    if turn_index != len(record.turns) or environment.transcript != record.transcript:
        raise RolloutValidationError("rollout turns/transcript do not reconstruct exactly")
    return record


def collect_rollout(
    policy: InteractivePolicyProtocol,
    episode: HiddenEpisode,
) -> RolloutRecord:
    """Run one policy until a terminal result, recording no repaired actions."""

    if not isinstance(policy, InteractivePolicyProtocol):
        raise TypeError("policy does not implement InteractivePolicyProtocol")
    if type(episode) is not HiddenEpisode:
        raise TypeError("collect_rollout requires a HiddenEpisode")
    environment = HiddenLawEnvironment(episode)
    turns: list[PolicyTurn] = []
    # Six tests plus either ready and answer, or the budget-forced answer.
    for _ in range(8):
        if environment.state in {"complete", "invalid", "aborted"}:
            break
        dialogue = render_dialogue(episode, environment.transcript)
        phase: PolicyPhase = "inquiry" if environment.state == "inquiry" else "answer"
        terminal_count = None if phase == "inquiry" else len(episode.terminal)
        decision = policy.sample_action(
            dialogue,
            phase=phase,
            terminal_count=terminal_count,
        )
        if type(decision) is PolicyAbort:
            environment.abort(decision.reason)
            break
        if type(decision) is not PolicyTurn:
            raise RolloutValidationError("policy returned an unsupported decision")
        if decision.prompt_digest != dialogue_prompt_digest(dialogue):
            raise RolloutValidationError("policy turn is bound to the wrong dialogue prompt")
        turns.append(decision)
        environment.consume(decision.raw_action)
    else:  # pragma: no cover - the environment budget makes this unreachable for valid control
        environment.abort("incomplete")

    if environment.state not in {"complete", "invalid", "aborted"}:
        environment.abort("incomplete")
    replay_transcript(episode, environment.transcript)
    reward = _terminal_reward_fraction(environment.transcript)
    record = RolloutRecord(
        episode_id=episode.episode_id,
        episode_digest=episode.digest,
        transcript=environment.transcript,
        transcript_digest=environment.transcript.digest,
        turns=tuple(turns),
        reward_numerator=reward.numerator,
        reward_denominator=reward.denominator,
    )
    return verify_rollout_record(record, episode)
