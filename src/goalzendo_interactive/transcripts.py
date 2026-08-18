"""Canonical append-only records for interactive G03 games."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Literal, TypeAlias, cast

from ._json import CanonicalJSONError, dump_json, json_digest, load_json
from .actions import (
    AnswerAction,
    InvalidActionError,
    ReadyAction,
    TestAction,
    action_from_obj,
)
from .episodes import HiddenEpisode, Observation, observation_from_obj
from .query import QueryMetrics, query_metrics_from_obj
from .schema import scene_index

AbortReason = Literal["incomplete", "overlength", "timed_out"]
ABORT_REASONS: tuple[AbortReason, ...] = ("incomplete", "overlength", "timed_out")
EnvironmentState = Literal["inquiry", "awaiting_answer", "complete", "invalid", "aborted"]
TRANSCRIPT_SCHEMA_VERSION = 2
TERMINAL_SCORE_SCHEMA_VERSION = 2
MAX_TEST_MOVES = 6


@dataclass(frozen=True, slots=True)
class TerminalScore:
    classification_correct: int
    classification_total: int
    rule_equivalent: bool
    query_count: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.classification_correct, bool)
            or not isinstance(self.classification_correct, int)
            or isinstance(self.classification_total, bool)
            or not isinstance(self.classification_total, int)
            or self.classification_total < 1
            or not 0 <= self.classification_correct <= self.classification_total
        ):
            raise ValueError("classification counts must satisfy 0 <= correct <= positive total")
        if type(self.rule_equivalent) is not bool:
            raise ValueError("rule_equivalent must be Boolean")
        if (
            isinstance(self.query_count, bool)
            or not isinstance(self.query_count, int)
            or not 0 <= self.query_count <= MAX_TEST_MOVES
        ):
            raise ValueError(f"query_count must lie in [0, {MAX_TEST_MOVES}]")

    @property
    def classification_accuracy(self) -> float:
        return self.classification_correct / self.classification_total

    @property
    def rule_equivalence_credit(self) -> float:
        return float(self.rule_equivalent)

    @property
    def query_efficiency(self) -> float:
        return 1.0 - self.query_count / MAX_TEST_MOVES

    @property
    def reward(self) -> float:
        return (
            0.70 * self.classification_accuracy
            + 0.25 * self.rule_equivalence_credit
            + 0.05 * self.query_efficiency
        )

    def as_obj(self) -> dict[str, int | bool]:
        """Return exact sufficient statistics; floating scores are derived."""

        return {
            "schema_version": TERMINAL_SCORE_SCHEMA_VERSION,
            "classification_correct": self.classification_correct,
            "classification_total": self.classification_total,
            "rule_equivalent": self.rule_equivalent,
            "query_count": self.query_count,
        }


def terminal_score_from_obj(value: Any) -> TerminalScore:
    expected = {
        "schema_version",
        "classification_correct",
        "classification_total",
        "rule_equivalent",
        "query_count",
    }
    if type(value) is not dict or set(value) != expected or len(value) != len(expected):
        raise ValueError("terminal score object has noncanonical fields")
    if value["schema_version"] != TERMINAL_SCORE_SCHEMA_VERSION:
        raise ValueError("unsupported terminal score schema version")
    return TerminalScore(
        classification_correct=value["classification_correct"],
        classification_total=value["classification_total"],
        rule_equivalent=value["rule_equivalent"],
        query_count=value["query_count"],
    )


@dataclass(frozen=True, slots=True)
class TestEvent:
    __test__: ClassVar[bool] = False

    action: TestAction
    observation: Observation
    metrics: QueryMetrics
    outcome: Literal["observation", "budget_exhausted"]

    def __post_init__(self) -> None:
        if type(self.action) is not TestAction:
            raise ValueError("test event requires a TestAction")
        if type(self.observation) is not Observation:
            raise ValueError("test event requires an Observation")
        if scene_index(self.action.koan) != self.observation.scene_index:
            raise ValueError("test action and oracle observation refer to different scenes")
        if type(self.metrics) is not QueryMetrics or self.metrics.scene_index != self.observation.scene_index:
            raise ValueError("test event metrics refer to a different scene")
        if self.outcome not in {"observation", "budget_exhausted"}:
            raise ValueError(f"unknown test outcome: {self.outcome!r}")

    def as_obj(self) -> dict[str, Any]:
        return {
            "event": "test",
            "action": self.action.as_obj(),
            "observation": self.observation.as_obj(),
            "metrics": self.metrics.as_obj(),
            "outcome": self.outcome,
        }


@dataclass(frozen=True, slots=True)
class ReadyEvent:
    action: ReadyAction
    premature: bool
    outcome: Literal["ready", "premature_ready"]

    def __post_init__(self) -> None:
        if type(self.action) is not ReadyAction:
            raise ValueError("ready event requires a ReadyAction")
        if type(self.premature) is not bool:
            raise ValueError("premature flag must be Boolean")
        expected = "premature_ready" if self.premature else "ready"
        if self.outcome != expected:
            raise ValueError(f"ready outcome must be {expected!r}")

    def as_obj(self) -> dict[str, Any]:
        return {
            "event": "ready",
            "action": self.action.as_obj(),
            "premature": self.premature,
            "outcome": self.outcome,
        }


@dataclass(frozen=True, slots=True)
class AnswerEvent:
    action: AnswerAction
    score: TerminalScore
    outcome: Literal["complete"] = "complete"

    def __post_init__(self) -> None:
        if type(self.action) is not AnswerAction:
            raise ValueError("answer event requires an AnswerAction")
        if type(self.score) is not TerminalScore:
            raise ValueError("answer event requires a TerminalScore")
        if self.outcome != "complete":
            raise ValueError("answer event outcome must be 'complete'")

    def as_obj(self) -> dict[str, Any]:
        return {
            "event": "answer",
            "action": self.action.as_obj(),
            "score": self.score.as_obj(),
            "outcome": self.outcome,
        }


@dataclass(frozen=True, slots=True)
class InvalidEvent:
    raw_action: str
    code: str
    detail: str
    outcome: Literal["invalid"] = "invalid"

    def __post_init__(self) -> None:
        if type(self.raw_action) is not str:
            raise ValueError("invalid event raw_action must be a string")
        if type(self.code) is not str or not self.code:
            raise ValueError("invalid event code cannot be empty")
        if type(self.detail) is not str or not self.detail:
            raise ValueError("invalid event detail cannot be empty")
        if self.outcome != "invalid":
            raise ValueError("invalid event outcome must be 'invalid'")

    def as_obj(self) -> dict[str, str]:
        return {
            "event": "invalid",
            "raw_action": self.raw_action,
            "code": self.code,
            "detail": self.detail,
            "outcome": self.outcome,
        }


@dataclass(frozen=True, slots=True)
class AbortEvent:
    """A controller termination that makes an incomplete trajectory score zero."""

    reason: AbortReason
    outcome: Literal["aborted"] = "aborted"

    def __post_init__(self) -> None:
        if self.reason not in ABORT_REASONS:
            raise ValueError(f"unknown abort reason: {self.reason!r}")
        if self.outcome != "aborted":
            raise ValueError("abort event outcome must be 'aborted'")

    def as_obj(self) -> dict[str, str]:
        return {
            "event": "abort",
            "reason": self.reason,
            "outcome": self.outcome,
        }


TranscriptEvent: TypeAlias = TestEvent | ReadyEvent | AnswerEvent | InvalidEvent | AbortEvent


def event_from_obj(value: Any) -> TranscriptEvent:
    if type(value) is not dict or type(value.get("event")) is not str:
        raise ValueError("transcript event must be a tagged JSON object")
    event = value["event"]
    try:
        if event == "test":
            expected = {"event", "action", "observation", "metrics", "outcome"}
            if set(value) != expected or len(value) != len(expected):
                raise ValueError("test event has noncanonical fields")
            action = action_from_obj(value["action"])
            if type(action) is not TestAction:
                raise ValueError("test event contains a non-test action")
            return TestEvent(
                action,
                observation_from_obj(value["observation"]),
                query_metrics_from_obj(value["metrics"]),
                value["outcome"],
            )
        if event == "ready":
            expected = {"event", "action", "premature", "outcome"}
            if set(value) != expected or len(value) != len(expected):
                raise ValueError("ready event has noncanonical fields")
            action = action_from_obj(value["action"])
            if type(action) is not ReadyAction:
                raise ValueError("ready event contains a non-ready action")
            return ReadyEvent(action, value["premature"], value["outcome"])
        if event == "answer":
            expected = {"event", "action", "score", "outcome"}
            if set(value) != expected or len(value) != len(expected):
                raise ValueError("answer event has noncanonical fields")
            action = action_from_obj(value["action"])
            if type(action) is not AnswerAction:
                raise ValueError("answer event contains a non-answer action")
            return AnswerEvent(action, terminal_score_from_obj(value["score"]), value["outcome"])
        if event == "invalid":
            expected = {"event", "raw_action", "code", "detail", "outcome"}
            if set(value) != expected or len(value) != len(expected):
                raise ValueError("invalid event has noncanonical fields")
            return InvalidEvent(
                value["raw_action"], value["code"], value["detail"], value["outcome"]
            )
        if event == "abort":
            expected = {"event", "reason", "outcome"}
            if set(value) != expected or len(value) != len(expected):
                raise ValueError("abort event has noncanonical fields")
            return AbortEvent(value["reason"], value["outcome"])
    except InvalidActionError as exc:
        raise ValueError(f"transcript contains an invalid canonical action: {exc}") from exc
    raise ValueError(f"unknown transcript event type: {event!r}")


@dataclass(frozen=True, slots=True)
class Transcript:
    episode_digest: str
    opening: tuple[Observation, ...]
    events: tuple[TranscriptEvent, ...] = ()
    state: EnvironmentState = "inquiry"

    def __post_init__(self) -> None:
        if (
            type(self.episode_digest) is not str
            or len(self.episode_digest) != 64
            or any(character not in "0123456789abcdef" for character in self.episode_digest)
        ):
            raise ValueError("episode_digest must be a 64-character hexadecimal digest")
        opening = tuple(self.opening)
        events = tuple(self.events)
        object.__setattr__(self, "opening", opening)
        object.__setattr__(self, "events", events)
        if len(opening) != 10 or any(type(item) is not Observation for item in opening):
            raise ValueError("transcript opening must contain exactly ten observations")
        if self.state not in {"inquiry", "awaiting_answer", "complete", "invalid", "aborted"}:
            raise ValueError(f"unknown transcript state: {self.state!r}")
        self._validate_sequence()

    def _validate_sequence(self) -> None:
        state: EnvironmentState = "inquiry"
        test_count = 0
        seen = {observation.scene_index for observation in self.opening}
        for event in self.events:
            if state in {"complete", "invalid", "aborted"}:
                raise ValueError("transcript contains an event after a terminal event")
            if type(event) is InvalidEvent:
                state = "invalid"
                continue
            if type(event) is AbortEvent:
                state = "aborted"
                continue
            if state == "inquiry" and type(event) is TestEvent:
                test_count += 1
                expected_outcome = "budget_exhausted" if test_count == MAX_TEST_MOVES else "observation"
                if event.outcome != expected_outcome:
                    raise ValueError("test event outcome is inconsistent with the six-test budget")
                expected_duplicate = event.observation.scene_index in seen
                if event.metrics.duplicate is not expected_duplicate:
                    raise ValueError("test event duplicate flag is inconsistent with prior evidence")
                seen.add(event.observation.scene_index)
                if test_count == MAX_TEST_MOVES:
                    state = "awaiting_answer"
                continue
            if state == "inquiry" and type(event) is ReadyEvent:
                state = "awaiting_answer"
                continue
            if state == "awaiting_answer" and type(event) is AnswerEvent:
                if event.score.query_count != test_count:
                    raise ValueError("answer score query count disagrees with transcript")
                state = "complete"
                continue
            raise ValueError("event type is invalid for its transcript state")
        if state != self.state:
            raise ValueError(f"transcript events imply state {state!r}, not stored state {self.state!r}")

    @classmethod
    def for_episode(cls, episode: HiddenEpisode) -> Transcript:
        return cls(episode_digest=episode.digest, opening=episode.opening)

    def append(self, event: TranscriptEvent, *, state: EnvironmentState) -> Transcript:
        return Transcript(self.episode_digest, self.opening, (*self.events, event), state)

    def as_obj(self) -> dict[str, Any]:
        return {
            "schema_version": TRANSCRIPT_SCHEMA_VERSION,
            "episode_digest": self.episode_digest,
            "opening": [observation.as_obj() for observation in self.opening],
            "events": [event.as_obj() for event in self.events],
            "state": self.state,
        }

    @property
    def digest(self) -> str:
        return json_digest(self.as_obj(), domain="goalzendo-interactive-transcript-v2")


def serialize_transcript(transcript: Transcript) -> str:
    if type(transcript) is not Transcript:
        raise TypeError("serialize_transcript requires a Transcript")
    return dump_json(transcript.as_obj())


def parse_transcript(text: str, *, require_canonical: bool = True) -> Transcript:
    try:
        value = load_json(text)
    except CanonicalJSONError as exc:
        raise ValueError(str(exc)) from exc
    expected = {"schema_version", "episode_digest", "opening", "events", "state"}
    if type(value) is not dict or set(value) != expected or len(value) != len(expected):
        raise ValueError("transcript object has noncanonical fields")
    if value["schema_version"] != TRANSCRIPT_SCHEMA_VERSION:
        raise ValueError("unsupported transcript schema version")
    if type(value["opening"]) is not list or type(value["events"]) is not list:
        raise ValueError("transcript opening and events must be arrays")
    result = Transcript(
        episode_digest=value["episode_digest"],
        opening=tuple(observation_from_obj(item) for item in value["opening"]),
        events=tuple(event_from_obj(item) for item in value["events"]),
        state=cast(EnvironmentState, value["state"]),
    )
    if require_canonical and serialize_transcript(result) != text:
        raise ValueError("transcript JSON is valid but not in canonical serialized form")
    return result
