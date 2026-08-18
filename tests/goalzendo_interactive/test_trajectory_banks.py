from __future__ import annotations

import json

import pytest

from goalzendo_interactive.environment import replay_transcript
from goalzendo_interactive.generation import generate_episode_bank, small_fixture_bank_spec
from goalzendo_interactive.trajectory_banks import (
    TrajectoryBankValidationError,
    generate_reference_trajectory_bank,
    parse_reference_trajectory_bank,
    serialize_reference_trajectory_bank,
    verify_reference_trajectory_bank,
)


@pytest.fixture(scope="module")
def trajectory_bank():  # type: ignore[no-untyped-def]
    source = generate_episode_bank(small_fixture_bank_spec())
    return generate_reference_trajectory_bank(source)


def test_reference_bank_is_complete_perfect_and_replayable(trajectory_bank) -> None:  # type: ignore[no-untyped-def]
    source = trajectory_bank.source_episode_bank
    assert len(trajectory_bank.records) == len(source.episodes) == 12
    for episode, record in zip(source.episodes, trajectory_bank.records, strict=True):
        environment = replay_transcript(episode, record.transcript)
        assert environment.state == "complete"
        assert environment.terminal_score is not None
        assert environment.terminal_score.classification_accuracy == 1.0
        assert environment.terminal_score.rule_equivalent
        assert record.terminal_reward == environment.final_reward
        assert 3 <= record.query_count <= 4
        assert record.assistant_action_count == record.query_count + 2


def test_reference_bank_round_trips_and_regenerates(trajectory_bank) -> None:  # type: ignore[no-untyped-def]
    text = serialize_reference_trajectory_bank(trajectory_bank)
    parsed = parse_reference_trajectory_bank(
        text,
        source_episode_bank=trajectory_bank.source_episode_bank,
    )
    assert parsed == trajectory_bank
    assert parsed.digest == trajectory_bank.digest
    assert verify_reference_trajectory_bank(parsed) == parsed


def test_reference_bank_tamper_is_rejected(trajectory_bank) -> None:  # type: ignore[no-untyped-def]
    value = json.loads(serialize_reference_trajectory_bank(trajectory_bank))
    value["records"][0]["query_count"] += 1
    tampered = json.dumps(value, separators=(",", ":"))
    with pytest.raises((TrajectoryBankValidationError, ValueError)):
        parse_reference_trajectory_bank(
            tampered,
            source_episode_bank=trajectory_bank.source_episode_bank,
        )
