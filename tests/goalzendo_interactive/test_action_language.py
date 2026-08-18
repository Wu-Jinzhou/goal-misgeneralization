from __future__ import annotations

import pytest

from goalzendo_interactive.action_language import (
    ACTION_LANGUAGE_SCHEMA_VERSION,
    ActionLanguageError,
    AnswerActionState,
    InquiryActionState,
    action_from_state,
    action_language_digest,
    action_language_manifest,
    build_answer_action,
    build_inquiry_action,
)
from goalzendo_interactive.actions import AnswerAction, ReadyAction, TestAction, serialize_action
from goalzendo_interactive.episodes import HiddenEpisode, terminal_classifications
from goalzendo_interactive.rules import SYNTACTIC_RULE_COUNT
from goalzendo_interactive.schema import SCENE_COUNT, iter_scenes, scene_at


def test_ready_path_is_exact_canonical_json() -> None:
    state = InquiryActionState()
    assert state.next_segment().field_id == "move"
    complete = state.choose("ready")
    assert complete.complete
    assert complete.text == serialize_action(ReadyAction()) == '{"move":"ready"}'
    assert action_from_state(complete) == ReadyAction()
    with pytest.raises(ActionLanguageError, match="already complete"):
        complete.next_segment()


def test_every_scene_has_one_exact_test_action_path() -> None:
    texts: set[str] = set()
    for scene in iter_scenes():
        action = TestAction(scene)
        state = build_inquiry_action(action)
        assert state.complete
        assert state.action == action
        texts.add(state.text)
    assert len(texts) == SCENE_COUNT == 13_716


def test_all_empty_scene_is_unrepresentable() -> None:
    state = InquiryActionState().choose("test")
    state = state.choose("empty")
    state = state.choose("empty")
    right = state.next_segment()
    assert right.field_id == "right.occupied"
    assert [option.key for option in right.options] == ["piece"]
    with pytest.raises(ActionLanguageError, match="invalid choice"):
        state.choose("empty")


def test_piece_field_order_and_text_match_action_serializer() -> None:
    action = TestAction(scene_at(13_715))
    state = build_inquiry_action(action)
    fields = [field for field, _ in state.selections]
    assert fields[:5] == ["move", "left.occupied", "left.size", "left.color", "left.shape"]
    assert state.text == serialize_action(action)


def test_terminal_answer_uses_full_rule_grammar_and_exact_label_count(
    hidden_episode: HiddenEpisode,
) -> None:
    state = AnswerActionState(len(hidden_episode.terminal))
    rule_segment = state.next_segment()
    assert rule_segment.field_id == "rule"
    assert len(rule_segment.options) == SYNTACTIC_RULE_COUNT == 18_760
    assert len({option.key for option in rule_segment.options}) == SYNTACTIC_RULE_COUNT

    action = AnswerAction(hidden_episode.target.rule, terminal_classifications(hidden_episode))
    complete = build_answer_action(action)
    assert complete.complete
    assert complete.classification_count == len(hidden_episode.terminal)
    assert complete.text == serialize_action(action)
    assert action_from_state(complete) == action


def test_action_language_rejects_wrong_field_choice_and_incomplete_finish() -> None:
    with pytest.raises(ActionLanguageError, match="invalid choice"):
        InquiryActionState().choose("answer")
    with pytest.raises(ActionLanguageError, match="incomplete"):
        _ = InquiryActionState().text
    with pytest.raises(ActionLanguageError, match="incomplete"):
        _ = AnswerActionState(16).action


def test_action_language_manifest_is_explicit_and_stable_shape() -> None:
    manifest = action_language_manifest()
    assert ACTION_LANGUAGE_SCHEMA_VERSION == 1
    assert manifest["inquiry_action_count"] == SCENE_COUNT + 1
    assert manifest["test_action_count"] == SCENE_COUNT
    assert manifest["syntactic_rule_count"] == SYNTACTIC_RULE_COUNT
    assert manifest["candidate_koan_list_presented_to_model"] is False
    assert manifest["digest"] == action_language_digest()
    assert len(action_language_digest()) == 64
