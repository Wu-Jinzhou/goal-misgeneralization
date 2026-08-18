"""Dependency-bound reference trajectory banks for G03 SFT and calibration."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from ._json import CanonicalJSONError, dump_json, json_digest, load_json
from .action_language import action_language_digest, build_answer_action, build_inquiry_action
from .dialogue import dialogue_as_obj, dialogue_digest, render_dialogue
from .environment import play_reference_episode, replay_transcript
from .generation import EpisodeBank
from .transcripts import (
    AnswerEvent,
    TestEvent,
    Transcript,
    event_from_obj,
    observation_from_obj,
)

REFERENCE_TRAJECTORY_BANK_SCHEMA_VERSION = 1
REFERENCE_TRAJECTORY_GENERATOR_SCHEMA_VERSION = 1


class TrajectoryBankValidationError(ValueError):
    """Raised when a trajectory bank is not exact or dependency-bound."""


def _valid_digest(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _dialogue_manifest_digest(transcript: Transcript, episode: Any) -> str:
    return json_digest(
        dialogue_as_obj(render_dialogue(episode, transcript)),
        domain="goalzendo-interactive-reference-dialogue-v1",
    )


@dataclass(frozen=True, slots=True)
class ReferenceTrajectoryRecord:
    episode_id: str
    episode_digest: str
    transcript: Transcript
    transcript_digest: str
    dialogue_manifest_digest: str
    message_count: int
    assistant_action_count: int
    query_count: int
    terminal_reward_numerator: int
    terminal_reward_denominator: int

    def __post_init__(self) -> None:
        if type(self.episode_id) is not str or not self.episode_id:
            raise TrajectoryBankValidationError("trajectory episode id cannot be empty")
        for name in ("episode_digest", "transcript_digest", "dialogue_manifest_digest"):
            if not _valid_digest(getattr(self, name)):
                raise TrajectoryBankValidationError(f"{name} must be a SHA-256 digest")
        if type(self.transcript) is not Transcript or self.transcript.state != "complete":
            raise TrajectoryBankValidationError("reference trajectory must be a complete Transcript")
        if self.transcript.digest != self.transcript_digest:
            raise TrajectoryBankValidationError("transcript digest is inconsistent")
        for name in ("message_count", "assistant_action_count", "query_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TrajectoryBankValidationError(f"{name} must be a non-negative integer")
        if self.assistant_action_count < 2 or self.message_count < self.assistant_action_count:
            raise TrajectoryBankValidationError("reference trajectory has impossible message counts")
        if not 0 <= self.query_count <= 6:
            raise TrajectoryBankValidationError("reference query count must lie in [0, 6]")
        if (
            isinstance(self.terminal_reward_numerator, bool)
            or not isinstance(self.terminal_reward_numerator, int)
            or isinstance(self.terminal_reward_denominator, bool)
            or not isinstance(self.terminal_reward_denominator, int)
            or self.terminal_reward_denominator <= 0
            or not 0 <= self.terminal_reward_numerator <= self.terminal_reward_denominator
        ):
            raise TrajectoryBankValidationError("terminal reward fraction is invalid")

    @property
    def terminal_reward(self) -> float:
        return self.terminal_reward_numerator / self.terminal_reward_denominator

    def as_obj(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "episode_digest": self.episode_digest,
            "transcript": self.transcript.as_obj(),
            "transcript_digest": self.transcript_digest,
            "dialogue_manifest_digest": self.dialogue_manifest_digest,
            "message_count": self.message_count,
            "assistant_action_count": self.assistant_action_count,
            "query_count": self.query_count,
            "terminal_reward_numerator": self.terminal_reward_numerator,
            "terminal_reward_denominator": self.terminal_reward_denominator,
        }


def _exact_reward_fraction(query_count: int) -> tuple[int, int]:
    # Perfect classification and rule recovery earn .95 + .05(1-q/6)
    # = (120 - q) / 120 exactly.
    return 120 - query_count, 120


def _record_for_episode(episode: Any) -> ReferenceTrajectoryRecord:
    transcript = play_reference_episode(episode)
    replayed = replay_transcript(episode, transcript)
    final = transcript.events[-1]
    if (
        type(final) is not AnswerEvent
        or final.score.classification_correct != final.score.classification_total
        or not final.score.rule_equivalent
    ):
        raise TrajectoryBankValidationError("reference policy failed perfect terminal behavior")
    dialogue = render_dialogue(episode, transcript)
    assistant_count = sum(message.role == "assistant" for message in dialogue)
    query_count = sum(type(event) is TestEvent for event in transcript.events)
    if replayed.query_count != query_count or final.score.query_count != query_count:
        raise TrajectoryBankValidationError("reference trajectory query counts disagree")
    for event in transcript.events:
        action = getattr(event, "action", None)
        if action is None:
            continue
        if type(event) is AnswerEvent:
            build_answer_action(event.action)
        else:
            build_inquiry_action(action)
    numerator, denominator = _exact_reward_fraction(query_count)
    if replayed.final_reward != numerator / denominator:
        raise TrajectoryBankValidationError("reference terminal reward differs from exact fraction")
    return ReferenceTrajectoryRecord(
        episode_id=episode.episode_id,
        episode_digest=episode.digest,
        transcript=transcript,
        transcript_digest=transcript.digest,
        dialogue_manifest_digest=_dialogue_manifest_digest(transcript, episode),
        message_count=len(dialogue),
        assistant_action_count=assistant_count,
        query_count=query_count,
        terminal_reward_numerator=numerator,
        terminal_reward_denominator=denominator,
    )


@dataclass(frozen=True, slots=True)
class ReferenceTrajectoryBank:
    bank_id: str
    source_episode_bank: EpisodeBank
    records: tuple[ReferenceTrajectoryRecord, ...]

    def __post_init__(self) -> None:
        if type(self.bank_id) is not str or not self.bank_id:
            raise TrajectoryBankValidationError("trajectory bank id cannot be empty")
        if type(self.source_episode_bank) is not EpisodeBank:
            raise TrajectoryBankValidationError("trajectory bank requires an EpisodeBank")
        records = tuple(self.records)
        object.__setattr__(self, "records", records)
        if len(records) != len(self.source_episode_bank.episodes):
            raise TrajectoryBankValidationError("trajectory records must align one-to-one with episodes")
        if any(type(record) is not ReferenceTrajectoryRecord for record in records):
            raise TrajectoryBankValidationError("trajectory bank contains a non-record")
        expected = tuple(
            (episode.episode_id, episode.digest) for episode in self.source_episode_bank.episodes
        )
        observed = tuple((record.episode_id, record.episode_digest) for record in records)
        if observed != expected:
            raise TrajectoryBankValidationError("trajectory records are not in source episode order")
        for episode, record in zip(self.source_episode_bank.episodes, records, strict=True):
            environment = replay_transcript(episode, record.transcript)
            if record.transcript.episode_digest != episode.digest:
                raise TrajectoryBankValidationError("reference transcript episode binding mismatch")
            query_count = sum(type(event) is TestEvent for event in record.transcript.events)
            final = record.transcript.events[-1]
            if (
                type(final) is not AnswerEvent
                or final.score.classification_correct != final.score.classification_total
                or not final.score.rule_equivalent
                or final.score.query_count != query_count
                or environment.query_count != query_count
            ):
                raise TrajectoryBankValidationError("reference terminal outcome is not perfect")
            numerator, denominator = _exact_reward_fraction(query_count)
            if (
                record.query_count != query_count
                or record.terminal_reward_numerator != numerator
                or record.terminal_reward_denominator != denominator
                or record.terminal_reward != environment.final_reward
            ):
                raise TrajectoryBankValidationError("reference query/reward fields do not reconstruct")
            dialogue = render_dialogue(episode, record.transcript)
            if record.dialogue_manifest_digest != _dialogue_manifest_digest(
                record.transcript, episode
            ):
                raise TrajectoryBankValidationError("reference dialogue digest does not reconstruct")
            if record.message_count != len(dialogue) or record.assistant_action_count != sum(
                message.role == "assistant" for message in dialogue
            ):
                raise TrajectoryBankValidationError("reference dialogue counts do not reconstruct")

    @property
    def digest(self) -> str:
        return json_digest(
            self.as_obj(), domain="goalzendo-interactive-reference-trajectory-bank-v1"
        )

    def as_obj(self) -> dict[str, Any]:
        return {
            "schema_version": REFERENCE_TRAJECTORY_BANK_SCHEMA_VERSION,
            "generator_schema_version": REFERENCE_TRAJECTORY_GENERATOR_SCHEMA_VERSION,
            "bank_id": self.bank_id,
            "source_episode_bank_id": self.source_episode_bank.spec.bank_id,
            "source_episode_bank_digest": self.source_episode_bank.digest,
            "dialogue_registry_digest": dialogue_digest(),
            "action_language_digest": action_language_digest(),
            "record_count": len(self.records),
            "records": [record.as_obj() for record in self.records],
        }


@lru_cache(maxsize=8)
def generate_reference_trajectory_bank(
    source_episode_bank: EpisodeBank,
    bank_id: str | None = None,
) -> ReferenceTrajectoryBank:
    if type(source_episode_bank) is not EpisodeBank:
        raise TypeError("generate_reference_trajectory_bank requires an EpisodeBank")
    selected_id = (
        f"{source_episode_bank.spec.bank_id}-reference-trajectories-v1"
        if bank_id is None
        else bank_id
    )
    return ReferenceTrajectoryBank(
        selected_id,
        source_episode_bank,
        tuple(_record_for_episode(episode) for episode in source_episode_bank.episodes),
    )


def serialize_reference_trajectory_bank(bank: ReferenceTrajectoryBank) -> str:
    if type(bank) is not ReferenceTrajectoryBank:
        raise TypeError("serialize_reference_trajectory_bank requires a ReferenceTrajectoryBank")
    return dump_json(bank.as_obj())


def _transcript_from_obj(value: Any) -> Transcript:
    expected = {"schema_version", "episode_digest", "opening", "events", "state"}
    if type(value) is not dict or set(value) != expected or len(value) != len(expected):
        raise TrajectoryBankValidationError("trajectory transcript has noncanonical fields")
    if value["schema_version"] != 2:
        raise TrajectoryBankValidationError("unsupported embedded transcript schema")
    if type(value["opening"]) is not list or type(value["events"]) is not list:
        raise TrajectoryBankValidationError("embedded transcript arrays are invalid")
    return Transcript(
        episode_digest=value["episode_digest"],
        opening=tuple(observation_from_obj(item) for item in value["opening"]),
        events=tuple(event_from_obj(item) for item in value["events"]),
        state=value["state"],
    )


def reference_trajectory_record_from_obj(value: Any) -> ReferenceTrajectoryRecord:
    expected = {
        "episode_id",
        "episode_digest",
        "transcript",
        "transcript_digest",
        "dialogue_manifest_digest",
        "message_count",
        "assistant_action_count",
        "query_count",
        "terminal_reward_numerator",
        "terminal_reward_denominator",
    }
    if type(value) is not dict or set(value) != expected or len(value) != len(expected):
        raise TrajectoryBankValidationError("trajectory record has noncanonical fields")
    result = ReferenceTrajectoryRecord(
        episode_id=value["episode_id"],
        episode_digest=value["episode_digest"],
        transcript=_transcript_from_obj(value["transcript"]),
        transcript_digest=value["transcript_digest"],
        dialogue_manifest_digest=value["dialogue_manifest_digest"],
        message_count=value["message_count"],
        assistant_action_count=value["assistant_action_count"],
        query_count=value["query_count"],
        terminal_reward_numerator=value["terminal_reward_numerator"],
        terminal_reward_denominator=value["terminal_reward_denominator"],
    )
    if result.as_obj() != value:
        raise TrajectoryBankValidationError("trajectory record is valid but not canonical")
    return result


def parse_reference_trajectory_bank(
    text: str,
    *,
    source_episode_bank: EpisodeBank,
    require_canonical: bool = True,
) -> ReferenceTrajectoryBank:
    try:
        value = load_json(text)
    except CanonicalJSONError as exc:
        raise TrajectoryBankValidationError(str(exc)) from exc
    expected = {
        "schema_version",
        "generator_schema_version",
        "bank_id",
        "source_episode_bank_id",
        "source_episode_bank_digest",
        "dialogue_registry_digest",
        "action_language_digest",
        "record_count",
        "records",
    }
    if type(value) is not dict or set(value) != expected or len(value) != len(expected):
        raise TrajectoryBankValidationError("trajectory bank has noncanonical fields")
    if value["schema_version"] != REFERENCE_TRAJECTORY_BANK_SCHEMA_VERSION:
        raise TrajectoryBankValidationError("unsupported trajectory bank schema")
    if value["generator_schema_version"] != REFERENCE_TRAJECTORY_GENERATOR_SCHEMA_VERSION:
        raise TrajectoryBankValidationError("unsupported trajectory generator schema")
    if value["source_episode_bank_digest"] != source_episode_bank.digest:
        raise TrajectoryBankValidationError("trajectory source episode-bank digest mismatch")
    if type(value["records"]) is not list:
        raise TrajectoryBankValidationError("trajectory records must be an array")
    result = ReferenceTrajectoryBank(
        value["bank_id"],
        source_episode_bank,
        tuple(reference_trajectory_record_from_obj(item) for item in value["records"]),
    )
    if result.as_obj() != value:
        raise TrajectoryBankValidationError("trajectory bank derived fields are inconsistent")
    if require_canonical and serialize_reference_trajectory_bank(result) != text:
        raise TrajectoryBankValidationError("trajectory bank JSON is valid but noncanonical")
    return verify_reference_trajectory_bank(result)


def verify_reference_trajectory_bank(
    bank: ReferenceTrajectoryBank,
) -> ReferenceTrajectoryBank:
    if type(bank) is not ReferenceTrajectoryBank:
        raise TypeError("verify_reference_trajectory_bank requires a ReferenceTrajectoryBank")
    regenerated = generate_reference_trajectory_bank(bank.source_episode_bank, bank.bank_id)
    if serialize_reference_trajectory_bank(regenerated) != serialize_reference_trajectory_bank(bank):
        raise TrajectoryBankValidationError(
            "reference trajectory bank does not regenerate byte-for-byte"
        )
    return bank
