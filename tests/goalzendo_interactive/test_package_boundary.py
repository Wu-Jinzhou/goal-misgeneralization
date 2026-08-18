from __future__ import annotations

import pytest

import goalzendo.modeling
from goalzendo_interactive.actions import TestAction, parse_action, serialize_action
from goalzendo_interactive.environment import play_reference_episode, replay_transcript
from goalzendo_interactive.episodes import HiddenEpisode
from goalzendo_interactive.schema import scene_at


def test_complete_interactive_episode_never_calls_legacy_two_action_scorer(
    monkeypatch,
    hidden_episode: HiddenEpisode,
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("legacy A/B scorer was called by the interactive engine")

    monkeypatch.setattr(goalzendo.modeling, "score_action_sequences", forbidden)
    transcript = play_reference_episode(hidden_episode)
    replayed = replay_transcript(hidden_episode, transcript)
    assert replayed.state == "complete"
    assert replayed.final_reward == pytest.approx(1.0 - 0.05 * replayed.query_count / 6)


def test_fixed_inquiry_state_has_at_least_one_thousand_distinct_generated_tests() -> None:
    actions = tuple(serialize_action(TestAction(scene_at(index))) for index in range(1_000))
    assert len(actions) == len(set(actions)) == 1_000
    assert all(
        parse_action(action) == TestAction(scene_at(index))
        for index, action in enumerate(actions)
    )
