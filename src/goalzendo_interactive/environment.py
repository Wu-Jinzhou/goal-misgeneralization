"""Six-test stateful hidden-law G03 environment and exact transcript replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

from .actions import (
    Action,
    AnswerAction,
    InvalidActionError,
    ReadyAction,
    TestAction,
    parse_action,
    serialize_action,
)
from .catalog import VersionSpace, truth_vector
from .episodes import HiddenEpisode, terminal_classifications
from .query import optimal_legal_query, query_metrics
from .schema import Scene, scene_at, scene_index
from .transcripts import (
    ABORT_REASONS,
    AbortEvent,
    AbortReason,
    AnswerEvent,
    EnvironmentState,
    InvalidEvent,
    ReadyEvent,
    TerminalScore,
    TestEvent,
    Transcript,
    TranscriptEvent,
)

StepOutcome = Literal[
    "observation",
    "budget_exhausted",
    "ready",
    "premature_ready",
    "complete",
    "invalid",
    "aborted",
    "already_terminal",
]


@dataclass(frozen=True, slots=True)
class StepResult:
    outcome: StepOutcome
    state: EnvironmentState
    event: TranscriptEvent | None = None
    error_code: str | None = None

    @property
    def duplicate(self) -> bool:
        return type(self.event) is TestEvent and self.event.metrics.duplicate

    @property
    def reward(self) -> float | None:
        if type(self.event) is AnswerEvent:
            return self.event.score.reward
        if self.outcome in {"invalid", "aborted", "already_terminal"}:
            return 0.0
        return None


class HiddenLawEnvironment:
    """Consume canonical actions without repairing any scientific action."""

    __slots__ = ("_episode", "_space", "_transcript")

    def __init__(self, episode: HiddenEpisode) -> None:
        if type(episode) is not HiddenEpisode:
            raise TypeError("HiddenLawEnvironment requires a HiddenEpisode")
        self._episode = episode
        self._space = episode.opening_version_space()
        self._transcript = Transcript.for_episode(episode)

    @property
    def episode(self) -> HiddenEpisode:
        return self._episode

    @property
    def transcript(self) -> Transcript:
        return self._transcript

    @property
    def state(self) -> EnvironmentState:
        return self._transcript.state

    @property
    def version_space(self) -> VersionSpace:
        return self._space

    @property
    def query_count(self) -> int:
        return sum(type(event) is TestEvent for event in self._transcript.events)

    @property
    def terminal_score(self) -> TerminalScore | None:
        if self.state != "complete":
            return None
        final = self._transcript.events[-1]
        return final.score if type(final) is AnswerEvent else None

    @property
    def final_reward(self) -> float | None:
        if self.state in {"invalid", "aborted"}:
            return 0.0
        score = self.terminal_score
        return None if score is None else score.reward

    def terminal_scenes(self) -> tuple[Scene, ...]:
        if self.state == "inquiry":
            raise RuntimeError("terminal scenes remain withheld during inquiry")
        if self.state in {"invalid", "aborted"}:
            raise RuntimeError("terminal scenes are unavailable after a zero-reward trajectory")
        return tuple(observation.scene for observation in self.episode.terminal)

    def consume(self, action: Action | str) -> StepResult:
        if type(action) not in {str, TestAction, ReadyAction, AnswerAction}:
            raise TypeError("consume requires a canonical Action or JSON string")
        if self.state in {"complete", "invalid", "aborted"}:
            return StepResult("already_terminal", self.state, error_code="already_terminal")

        raw_action: str
        if type(action) is str:
            raw_action = action
            try:
                parsed = parse_action(
                    action,
                    terminal_count=(len(self.episode.terminal) if self.state == "awaiting_answer" else None),
                )
            except InvalidActionError as exc:
                return self._invalidate(raw_action, exc.code, exc.detail)
        elif type(action) in {TestAction, ReadyAction, AnswerAction}:
            parsed = cast(Action, action)
            raw_action = serialize_action(parsed)
        else:  # pragma: no cover - guarded by the exact-type check above
            raise AssertionError("unreachable action type")

        if self.state == "inquiry":
            if type(parsed) is TestAction:
                return self._consume_test(parsed)
            if type(parsed) is ReadyAction:
                premature = len(self._space) > 1
                ready_outcome: Literal["ready", "premature_ready"] = (
                    "premature_ready" if premature else "ready"
                )
                ready_event = ReadyEvent(parsed, premature, ready_outcome)
                self._transcript = self._transcript.append(ready_event, state="awaiting_answer")
                return StepResult(ready_outcome, self.state, ready_event)
            return self._invalidate(
                raw_action,
                "unexpected_answer",
                "answer action is invalid before ready or budget exhaustion",
            )

        if type(parsed) is not AnswerAction:
            return self._invalidate(
                raw_action,
                "expected_answer",
                "only an answer action is valid after ready or budget exhaustion",
            )
        if len(parsed.classifications) != len(self.episode.terminal):
            return self._invalidate(
                raw_action,
                "wrong_classification_count",
                f"expected {len(self.episode.terminal)}, received {len(parsed.classifications)}",
            )
        correct = sum(
            classification == expected
            for classification, expected in zip(
                parsed.classifications,
                terminal_classifications(self.episode),
                strict=True,
            )
        )
        score = TerminalScore(
            classification_correct=correct,
            classification_total=len(self.episode.terminal),
            rule_equivalent=truth_vector(parsed.rule).bits == self.episode.target.truth.bits,
            query_count=self.query_count,
        )
        answer_event = AnswerEvent(parsed, score)
        self._transcript = self._transcript.append(answer_event, state="complete")
        return StepResult("complete", self.state, answer_event)

    def abort(self, reason: AbortReason) -> StepResult:
        """Record an external incomplete/overlength/timeout termination exactly."""

        if reason not in ABORT_REASONS:
            raise ValueError(f"unknown abort reason: {reason!r}")
        if self.state in {"complete", "invalid", "aborted"}:
            return StepResult("already_terminal", self.state, error_code="already_terminal")
        event = AbortEvent(reason)
        self._transcript = self._transcript.append(event, state="aborted")
        return StepResult("aborted", self.state, event, error_code=reason)

    def _consume_test(self, action: TestAction) -> StepResult:
        observed = frozenset(
            observation.scene_index
            for observation in self.episode.opening
        ) | frozenset(
            event.observation.scene_index
            for event in self._transcript.events
            if type(event) is TestEvent
        )
        query_scene_index = scene_index(action.koan)
        metrics, after, observation = query_metrics(
            self._space,
            query_scene_index,
            target=self.episode.target,
            shadow=self.episode.shadow,
            observed_scene_indices=observed,
        )
        self._space = after
        next_count = self.query_count + 1
        test_outcome: Literal["observation", "budget_exhausted"] = (
            "budget_exhausted" if next_count == 6 else "observation"
        )
        event = TestEvent(action, observation, metrics, test_outcome)
        next_state: EnvironmentState = (
            "awaiting_answer" if test_outcome == "budget_exhausted" else "inquiry"
        )
        self._transcript = self._transcript.append(event, state=next_state)
        return StepResult(test_outcome, self.state, event)

    def _invalidate(self, raw_action: str, code: str, detail: str) -> StepResult:
        event = InvalidEvent(raw_action, code, detail)
        self._transcript = self._transcript.append(event, state="invalid")
        return StepResult("invalid", self.state, event, error_code=code)


class TranscriptReplayError(ValueError):
    pass


def replay_transcript(episode: HiddenEpisode, transcript: Transcript) -> HiddenLawEnvironment:
    """Replay every recorded action and require byte-identical derived state."""

    if transcript.episode_digest != episode.digest or transcript.opening != episode.opening:
        raise TranscriptReplayError("transcript is not bound to the supplied episode")
    environment = HiddenLawEnvironment(episode)
    for expected_event in transcript.events:
        if type(expected_event) is InvalidEvent:
            result = environment.consume(expected_event.raw_action)
        elif type(expected_event) is AbortEvent:
            result = environment.abort(expected_event.reason)
        else:
            action_event = cast(TestEvent | ReadyEvent | AnswerEvent, expected_event)
            result = environment.consume(action_event.action)
        if result.event != expected_event:
            raise TranscriptReplayError("replayed event differs from the recorded event")
    if environment.transcript != transcript or environment.transcript.digest != transcript.digest:
        raise TranscriptReplayError("replayed transcript or digest differs from the original")
    return environment


def play_reference_episode(episode: HiddenEpisode) -> Transcript:
    """Play an exact-query, exact-answer calibration trajectory."""

    environment = HiddenLawEnvironment(episode)
    observed = {observation.scene_index for observation in episode.opening}
    while environment.state == "inquiry" and len(environment.version_space) > 1:
        choice = optimal_legal_query(
            environment.version_space,
            legal_scene_indices=tuple(
                scene_index
                for scene_index in range(len(episode.target.truth))
                if scene_index not in observed
            ),
        )
        environment.consume(TestAction(scene_at(choice.scene_index)))
        observed.add(choice.scene_index)
    if environment.state == "inquiry":
        environment.consume(ReadyAction())
    answer = AnswerAction(
        episode.target.rule,
        cast(tuple, terminal_classifications(episode)),
    )
    result = environment.consume(answer)
    if result.outcome != "complete":
        raise RuntimeError("reference trajectory did not reach a complete answer")
    return environment.transcript
