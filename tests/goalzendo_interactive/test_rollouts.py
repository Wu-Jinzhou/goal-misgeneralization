from __future__ import annotations

import json
from collections import deque

import pytest

from goalzendo_interactive.actions import serialize_action
from goalzendo_interactive.dialogue import Dialogue
from goalzendo_interactive.environment import play_reference_episode
from goalzendo_interactive.episodes import HiddenEpisode
from goalzendo_interactive.rollouts import (
    PolicyAbort,
    PolicyPhase,
    PolicyTurn,
    RolloutValidationError,
    collect_rollout,
    dialogue_prompt_digest,
    parse_rollout_record,
    serialize_rollout_record,
    verify_rollout_record,
)


class ScriptedPolicy:
    def __init__(self, actions: list[str | PolicyAbort]) -> None:
        self.actions = deque(actions)

    def sample_action(
        self,
        dialogue: Dialogue,
        *,
        phase: PolicyPhase,
        terminal_count: int | None,
    ) -> PolicyTurn | PolicyAbort:
        assert (phase == "inquiry") is (terminal_count is None)
        selected = self.actions.popleft()
        if type(selected) is PolicyAbort:
            return selected
        tokens = tuple(selected.encode("ascii"))
        return PolicyTurn(
            selected,
            tokens,
            (-0.5,) * len(tokens),
            (0.25,) * len(tokens),
            dialogue_prompt_digest(dialogue),
        )


def test_reference_rollout_is_exact_replayable_and_reward_bound(
    hidden_episode: HiddenEpisode,
) -> None:
    reference = play_reference_episode(hidden_episode)
    actions = [serialize_action(event.action) for event in reference.events]
    record = collect_rollout(ScriptedPolicy(actions), hidden_episode)
    assert record.transcript == reference
    assert record.reward == pytest.approx(reference.events[-1].score.reward)  # type: ignore[union-attr]
    assert record.reward_numerator == 23
    assert record.reward_denominator == 24
    assert len(record.turns) == len(reference.events)
    assert record.token_count == sum(len(action.encode("ascii")) for action in actions)
    assert len(record.digest) == 64


def test_invalid_action_is_retained_and_scores_zero(hidden_episode: HiddenEpisode) -> None:
    record = collect_rollout(ScriptedPolicy(["not json"]), hidden_episode)
    assert record.transcript.state == "invalid"
    assert record.reward_numerator == 0
    assert record.reward_denominator == 1
    assert record.turns[0].raw_action == "not json"


def test_rollout_serialization_is_canonical_and_episode_verified(
    hidden_episode: HiddenEpisode,
) -> None:
    reference = play_reference_episode(hidden_episode)
    actions = [serialize_action(event.action) for event in reference.events]
    record = collect_rollout(ScriptedPolicy(actions), hidden_episode)
    encoded = serialize_rollout_record(record)
    parsed = parse_rollout_record(encoded)
    assert parsed == record
    assert verify_rollout_record(parsed, hidden_episode) is parsed
    with pytest.raises(RolloutValidationError, match="not canonical"):
        parse_rollout_record(json.dumps(json.loads(encoded), indent=2))

    tampered_encoded = encoded.replace(record.turns[0].prompt_digest, "0" * 64, 1)
    parsed_tamper = parse_rollout_record(tampered_encoded)
    with pytest.raises(RolloutValidationError, match="prompt binding"):
        verify_rollout_record(parsed_tamper, hidden_episode)


@pytest.mark.parametrize("reason", ["incomplete", "overlength", "timed_out"])
def test_controller_abort_is_replayed_and_scores_zero(
    hidden_episode: HiddenEpisode,
    reason: str,
) -> None:
    record = collect_rollout(
        ScriptedPolicy([PolicyAbort(reason)]),  # type: ignore[arg-type]
        hidden_episode,
    )
    assert record.transcript.state == "aborted"
    assert record.reward == 0.0
    assert record.turns == ()
    assert record.transcript.events[-1].reason == reason  # type: ignore[union-attr]


def test_policy_prompt_mismatch_and_bad_token_statistics_fail_closed(
    hidden_episode: HiddenEpisode,
) -> None:
    class WrongPromptPolicy(ScriptedPolicy):
        def sample_action(
            self,
            dialogue: Dialogue,
            *,
            phase: PolicyPhase,
            terminal_count: int | None,
        ) -> PolicyTurn:
            return PolicyTurn(
                '{"move":"ready"}',
                (1,),
                (-0.1,),
                (0.1,),
                "0" * 64,
            )

    with pytest.raises(RolloutValidationError, match="wrong dialogue"):
        collect_rollout(WrongPromptPolicy([]), hidden_episode)
    with pytest.raises(RolloutValidationError, match="<= 0"):
        PolicyTurn("x", (1,), (0.1,), (0.1,), "0" * 64)
