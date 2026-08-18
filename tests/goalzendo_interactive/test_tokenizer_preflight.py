from __future__ import annotations

from collections.abc import Sequence

import pytest

from goalzendo_interactive.dialogue import DialogueMessage, render_dialogue
from goalzendo_interactive.environment import play_reference_episode
from goalzendo_interactive.episodes import HiddenEpisode
from goalzendo_interactive.rules import SYNTACTIC_RULE_COUNT
from goalzendo_interactive.schema import SCENE_COUNT
from goalzendo_interactive.tokenizer_preflight import (
    TOKENIZER_PREFLIGHT_CHECK_IDS,
    TOKENIZER_PREFLIGHT_SCHEMA_VERSION,
    TokenizerPreflightError,
    TokenizerPreflightReport,
    run_tokenizer_preflight,
)


class CharacterChatTokenizer:
    """A deterministic tokenizer with deliberately transparent boundaries."""

    def apply_chat_template(
        self,
        conversation: Sequence[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        text = "".join(
            f"<turn:{message['role']}>\n{message['content']}<end>\n"
            for message in conversation
        )
        if add_generation_prompt:
            text += "<turn:assistant>\n"
        return text

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [ord(character) for character in text]


class MissingGenerationBoundaryTokenizer(CharacterChatTokenizer):
    def apply_chat_template(
        self,
        conversation: Sequence[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        return super().apply_chat_template(
            conversation,
            tokenize=tokenize,
            add_generation_prompt=False,
        )


class BoundaryMergingTokenizer(CharacterChatTokenizer):
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        values = super().encode(text, add_special_tokens=add_special_tokens)
        if text.endswith("}") and len(values) >= 2:
            return [*values[:-2], values[-2] * 256 + values[-1]]
        return values


@pytest.fixture(scope="module")
def reference_dialogue(hidden_episode: HiddenEpisode) -> tuple[DialogueMessage, ...]:
    return render_dialogue(hidden_episode, play_reference_episode(hidden_episode))


@pytest.fixture(scope="module")
def preflight_report(
    reference_dialogue: tuple[DialogueMessage, ...],
) -> TokenizerPreflightReport:
    return run_tokenizer_preflight(
        CharacterChatTokenizer(),
        reference_dialogue,
        tokenizer_identifier="deterministic-character-chat-v1",
        maximum_tokens=250_000,
    )


def test_complete_preflight_is_pass_only_scoped_and_deterministic(
    preflight_report: TokenizerPreflightReport,
) -> None:
    assert TOKENIZER_PREFLIGHT_SCHEMA_VERSION == 1
    assert tuple(check.check_id for check in preflight_report.checks) == (
        TOKENIZER_PREFLIGHT_CHECK_IDS
    )
    assert preflight_report.passed
    assert not preflight_report.weight_updates_authorized
    assert not preflight_report.full_model_smoke_completed
    assert preflight_report.as_obj()["authorization"] == {
        "weight_updates_authorized": False,
        "full_model_smoke_completed": False,
        "reason": "local tokenizer-format evidence only",
    }
    assert preflight_report.digest == (
        "607b0cf9736da2c086f232e5c7047e776aaafc2ea0b795404fb4e73be65cb757"
    )


def test_finite_languages_are_exhaustive_and_token_samples_have_registered_coverage(
    preflight_report: TokenizerPreflightReport,
) -> None:
    roundtrips = preflight_report.check("canonical_action_roundtrips")
    coverage = preflight_report.check("enumeration_and_sample_coverage")
    prefixes = preflight_report.check("action_prefix_stability")
    assert roundtrips.exhaustive
    assert roundtrips.evidence["test_action_count"] == SCENE_COUNT == 13_716
    assert roundtrips.evidence["ready_action_count"] == 1
    assert roundtrips.evidence["syntactic_rule_count"] == SYNTACTIC_RULE_COUNT == 18_760
    assert roundtrips.evidence["action_inventory_digest"] == (
        "cb5f6c1fdb4324842a2b2852d820c9fe25fb680078f8d88c461b04ae828dbe96"
    )
    assert coverage.evidence["enumerated_test_action_count"] == SCENE_COUNT
    assert coverage.evidence["enumerated_rule_count"] == SYNTACTIC_RULE_COUNT
    assert coverage.evidence["tokenized_inquiry_action_count"] == 65
    assert coverage.evidence["tokenized_answer_action_count"] == 64
    assert len(coverage.evidence["inquiry_option_pairs"]) == coverage.evidence[
        "inquiry_option_pair_count"
    ]
    assert prefixes.item_count == 129
    assert not prefixes.exhaustive


def test_complete_dialogue_masks_only_actions_and_proves_no_truncation(
    preflight_report: TokenizerPreflightReport,
) -> None:
    masking = preflight_report.check("assistant_only_masking").evidence
    no_truncation = preflight_report.check("no_truncation").evidence
    assert masking["assistant_action_count"] >= 2
    assert masking["supervised_token_count"] > 0
    assert masking["ignored_token_count"] > masking["supervised_token_count"]
    assert masking["deterministic_repeat_equal"] is True
    assert no_truncation["full_token_count"] == masking["full_token_count"]
    assert no_truncation["one_token_short_limit_rejected"] is True
    assert no_truncation["truncation_performed"] is False


def test_missing_generation_boundary_fails_closed(
    reference_dialogue: tuple[DialogueMessage, ...],
) -> None:
    with pytest.raises(TokenizerPreflightError, match="strict extension"):
        run_tokenizer_preflight(
            MissingGenerationBoundaryTokenizer(),
            reference_dialogue,
            tokenizer_identifier="broken-no-generation-boundary",
            maximum_tokens=250_000,
        )


def test_token_boundary_merge_fails_closed(
    reference_dialogue: tuple[DialogueMessage, ...],
) -> None:
    with pytest.raises(TokenizerPreflightError, match="terminator changes"):
        run_tokenizer_preflight(
            BoundaryMergingTokenizer(),
            reference_dialogue,
            tokenizer_identifier="broken-boundary-merger",
            maximum_tokens=250_000,
        )


def test_too_small_declared_context_fails_without_truncation(
    reference_dialogue: tuple[DialogueMessage, ...],
) -> None:
    with pytest.raises(TokenizerPreflightError, match="truncation is forbidden"):
        run_tokenizer_preflight(
            CharacterChatTokenizer(),
            reference_dialogue,
            tokenizer_identifier="deterministic-character-chat-v1",
            maximum_tokens=1,
        )


def test_noncanonical_dialogue_action_fails_before_tokenizer_audit(
    reference_dialogue: tuple[DialogueMessage, ...],
) -> None:
    messages = list(reference_dialogue)
    assistant_index = next(
        index for index, message in enumerate(messages) if message.role == "assistant"
    )
    original = messages[assistant_index]
    messages[assistant_index] = DialogueMessage(
        original.role,
        original.content + "\n",
        original.phase,
    )
    with pytest.raises(TokenizerPreflightError, match="not a canonical G03 action"):
        run_tokenizer_preflight(
            CharacterChatTokenizer(),
            tuple(messages),
            tokenizer_identifier="deterministic-character-chat-v1",
            maximum_tokens=250_000,
        )
