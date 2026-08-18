from __future__ import annotations

import pytest

from goalzendo_interactive.actions import AnswerAction, ReadyAction, TestAction, serialize_action
from goalzendo_interactive.dialogue import (
    DIALOGUE_SCHEMA_VERSION,
    DialogueMessage,
    DialogueRenderError,
    dialogue_as_obj,
    dialogue_digest,
    dialogue_to_chat,
    render_dialogue,
    render_opening_prompt,
    render_terminal_prompt,
)
from goalzendo_interactive.environment import HiddenLawEnvironment, play_reference_episode
from goalzendo_interactive.episodes import HiddenEpisode, terminal_classifications
from goalzendo_interactive.rendering import render_scene
from goalzendo_interactive.schema import scene_at
from goalzendo_interactive.transcripts import Transcript


def test_initial_dialogue_withholds_terminal_and_evaluator_identities(
    hidden_episode: HiddenEpisode,
) -> None:
    dialogue = render_dialogue(hidden_episode)
    assert [(message.role, message.phase) for message in dialogue] == [
        ("system", "contract"),
        ("user", "opening"),
    ]
    opening = dialogue[1].content
    assert opening == render_opening_prompt(hidden_episode)
    assert opening.count("Fits —") == 5
    assert opening.count("Does not fit —") == 5
    assert hidden_episode.episode_id not in opening
    assert hidden_episode.target.rule_id not in opening
    assert hidden_episode.shadow.rule_id not in opening
    assert hidden_episode.target.truth_digest not in opening
    assert hidden_episode.shadow.truth_digest not in opening
    for observation in hidden_episode.terminal:
        assert render_scene(observation.scene, hidden_episode.renderer) not in opening


def test_test_feedback_and_ready_reveal_terminal_at_the_right_time(
    hidden_episode: HiddenEpisode,
) -> None:
    environment = HiddenLawEnvironment(hidden_episode)
    first_scene = scene_at(123)
    first = environment.consume(TestAction(first_scene))
    assert first.outcome == "observation"
    inquiry = render_dialogue(hidden_episode, environment.transcript)
    assert inquiry[-2].content == serialize_action(TestAction(first_scene))
    assert inquiry[-1].phase == "feedback"
    assert "5 test moves remaining" in inquiry[-1].content
    assert "Terminal koans:" not in inquiry[-1].content

    environment.consume(ReadyAction())
    ready = render_dialogue(hidden_episode, environment.transcript)
    assert ready[-2].content == '{"move":"ready"}'
    assert ready[-1].phase == "terminal"
    assert ready[-1].content == render_terminal_prompt(hidden_episode)
    assert ready[-1].content.count("\n") >= len(hidden_episode.terminal)
    assert hidden_episode.target.rule_id not in ready[-1].content
    assert hidden_episode.shadow.rule_id not in ready[-1].content


def test_budget_exhaustion_shows_feedback_and_terminal_once(hidden_episode: HiddenEpisode) -> None:
    environment = HiddenLawEnvironment(hidden_episode)
    for index in range(6):
        result = environment.consume(TestAction(scene_at(1_000 + index)))
    assert result.outcome == "budget_exhausted"
    dialogue = render_dialogue(hidden_episode, environment.transcript)
    assert dialogue[-1].phase == "terminal"
    assert dialogue[-1].content.count("Terminal koans:") == 1
    assert dialogue[-1].content.startswith("Master:")
    assert sum(message.phase == "terminal" for message in dialogue) == 1


def test_complete_reference_dialogue_contains_only_actions_not_scores(
    hidden_episode: HiddenEpisode,
) -> None:
    transcript = play_reference_episode(hidden_episode)
    dialogue = render_dialogue(hidden_episode, transcript)
    assert dialogue[-1].role == "assistant"
    assert dialogue[-1].content.startswith('{"move":"answer"')
    joined = "\n".join(message.content for message in dialogue)
    assert "classification_correct" not in joined
    assert "rule_equivalent" not in joined
    assert "query_efficiency" not in joined
    assert "target_rule_id" not in joined
    assert "shadow_rule_id" not in joined
    assert dialogue_to_chat(dialogue)[-1] == {
        "role": "assistant",
        "content": dialogue[-1].content,
    }
    assert dialogue_as_obj(dialogue)[-1]["phase"] == "action"


def test_abort_is_controller_only_and_invalid_action_is_model_text(
    hidden_episode: HiddenEpisode,
) -> None:
    aborted = HiddenLawEnvironment(hidden_episode)
    aborted.abort("timed_out")
    assert render_dialogue(hidden_episode, aborted.transcript) == render_dialogue(hidden_episode)

    invalid = HiddenLawEnvironment(hidden_episode)
    invalid.consume("not-json")
    rendered = render_dialogue(hidden_episode, invalid.transcript)
    assert rendered[-1] == DialogueMessage("assistant", "not-json", "action")


def test_forged_or_mismatched_transcript_is_rejected(
    hidden_episode: HiddenEpisode,
    noisy_episode: HiddenEpisode,
) -> None:
    with pytest.raises(DialogueRenderError, match="not bound"):
        render_dialogue(hidden_episode, Transcript.for_episode(noisy_episode))


def test_terminal_answer_contract_has_exact_count(hidden_episode: HiddenEpisode) -> None:
    prompt = render_terminal_prompt(hidden_episode)
    assert f"exactly {len(hidden_episode.terminal)} entries" in prompt
    action = AnswerAction(hidden_episode.target.rule, terminal_classifications(hidden_episode))
    environment = HiddenLawEnvironment(hidden_episode)
    environment.consume(ReadyAction())
    assert environment.consume(action).outcome == "complete"


def test_dialogue_registry_digest_is_stable_shape() -> None:
    assert DIALOGUE_SCHEMA_VERSION == 1
    assert len(dialogue_digest()) == 64
    assert dialogue_digest() == dialogue_digest()
