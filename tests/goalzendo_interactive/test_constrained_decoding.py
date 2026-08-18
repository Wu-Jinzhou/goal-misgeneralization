from __future__ import annotations

import math
from collections.abc import Sequence

import pytest

import goalzendo_interactive
from goalzendo_interactive.actions import AnswerAction, ReadyAction, TestAction, parse_action
from goalzendo_interactive.constrained_decoding import (
    CONSTRAINED_DECODING_SCHEMA_VERSION,
    ActionOverlengthError,
    GrammarConstrainedPolicy,
    NoLegalTokenError,
    TokenizerPrefixStabilityError,
    TokenSamplingStep,
    constrained_decoder_manifest,
)
from goalzendo_interactive.dialogue import Dialogue, DialogueMessage
from goalzendo_interactive.rollouts import PolicyTurn, dialogue_prompt_digest
from goalzendo_interactive.rules import SYNTACTIC_RULE_COUNT


class CharacterChatTokenizer:
    """Transparent deterministic tokenizer used only for local unit tests."""

    def __init__(self) -> None:
        self.encode_calls = 0

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
        self.encode_calls += 1
        return [ord(character) for character in text]


class BoundaryMergingTokenizer(CharacterChatTokenizer):
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        values = super().encode(text, add_special_tokens=add_special_tokens)
        boundary = "<turn:assistant>\n{"
        if boundary in text:
            brace = text.index(boundary) + len(boundary) - 1
            return [
                *values[: brace - 1],
                values[brace - 1] * 256 + values[brace],
                *values[brace + 1 :],
            ]
        return values


class FieldBoundaryMergingTokenizer(CharacterChatTokenizer):
    """Mimic the pinned Qwen merge of the move quote with the following comma."""

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        values = super().encode(text, add_special_tokens=add_special_tokens)
        boundary = '"test",'
        if boundary in text:
            quote = text.index(boundary) + len('"test')
            return [*values[:quote], 400, *values[quote + 2 :]]
        return values


class FixedLogits:
    def __init__(self, values: tuple[float, ...]) -> None:
        self.values = values
        self.seen_contexts: list[tuple[int, ...]] = []

    def __call__(self, input_ids: tuple[int, ...]) -> tuple[float, ...]:
        self.seen_contexts.append(input_ids)
        return self.values


class PreferMoveLogits:
    def __init__(self, move: str) -> None:
        assert move in {"test", "ready"}
        self._ordinary = (0.0,) * 128
        values = list(self._ordinary)
        values[ord("r" if move == "test" else "t")] = -math.inf
        self._move_choice = tuple(values)

    def __call__(self, input_ids: tuple[int, ...]) -> tuple[float, ...]:
        if bytes(input_ids).endswith(b'{"move":"'):
            return self._move_choice
        return self._ordinary


@pytest.fixture
def short_dialogue() -> Dialogue:
    return (
        DialogueMessage("system", "Hidden-law contract.", "contract"),
        DialogueMessage("user", "Return one action.", "opening"),
    )


def test_seeded_field_grammar_generates_at_least_one_thousand_distinct_test_actions(
    short_dialogue: Dialogue,
) -> None:
    tokenizer = CharacterChatTokenizer()
    policy = GrammarConstrainedPolicy(
        tokenizer,
        PreferMoveLogits("test"),
        seed=314159,
        maximum_action_tokens=512,
    )
    actions: set[str] = set()
    for _ in range(3_000):
        turn = policy.sample_action(short_dialogue, phase="inquiry", terminal_count=None)
        parsed = parse_action(turn.raw_action, expected_move="test")
        assert type(parsed) is TestAction
        actions.add(turn.raw_action)
        if len(actions) >= 1_000:
            break
    assert len(actions) >= 1_000


def test_ready_and_terminal_answer_are_canonical_policy_turns(
    short_dialogue: Dialogue,
) -> None:
    ready = GrammarConstrainedPolicy(
        CharacterChatTokenizer(),
        PreferMoveLogits("ready"),
        seed=11,
    ).sample_action(short_dialogue, phase="inquiry", terminal_count=None)
    assert type(ready) is PolicyTurn
    assert parse_action(ready.raw_action, expected_move="ready") == ReadyAction()
    assert ready.prompt_digest == dialogue_prompt_digest(short_dialogue)

    tokenizer = CharacterChatTokenizer()
    answer = GrammarConstrainedPolicy(
        tokenizer,
        FixedLogits((0.0,) * 128),
        seed=12,
        maximum_action_tokens=2_048,
    ).sample_action(short_dialogue, phase="answer", terminal_count=12)
    parsed = parse_action(answer.raw_action, expected_move="answer", terminal_count=12)
    assert type(parsed) is AnswerAction
    assert len(parsed.classifications) == 12
    assert len(answer.token_ids) == len(answer.token_log_probabilities) == len(answer.token_entropies)
    # One rule trie plus two choices per label: no 18,760 x 2^12 answer table.
    assert SYNTACTIC_RULE_COUNT == 18_760
    assert tokenizer.encode_calls < 20_000


def test_exact_masks_and_recorded_statistics_are_the_distribution_used_for_sampling(
    short_dialogue: Dialogue,
) -> None:
    temperature = 0.7
    raw_logits = tuple((token % 17 - 8) / 3.0 for token in range(128))
    callback = FixedLogits(raw_logits)
    steps: list[TokenSamplingStep] = []
    turn = GrammarConstrainedPolicy(
        CharacterChatTokenizer(),
        callback,
        seed=2718,
        temperature=temperature,
        observer=steps.append,
    ).sample_action(short_dialogue, phase="inquiry", terminal_count=None)

    assert turn.token_ids == tuple(step.selected_token_id for step in steps)
    assert turn.token_log_probabilities == tuple(
        step.selected_log_probability for step in steps
    )
    assert turn.token_entropies == tuple(step.entropy for step in steps)
    assert any(step.mask.allowed_token_ids == (ord("r"), ord("t")) for step in steps)

    for step in steps:
        assert sum(step.mask.dense) == len(step.mask.allowed_token_ids)
        masked = step.mask.mask_logits(raw_logits)
        assert all(
            masked[token] == -math.inf
            for token in range(len(masked))
            if token not in step.mask.allowed_token_ids
        )
        scaled = [raw_logits[token] / temperature for token in step.mask.allowed_token_ids]
        maximum = max(scaled)
        normalizer = maximum + math.log(sum(math.exp(value - maximum) for value in scaled))
        expected = tuple(value - normalizer for value in scaled)
        assert tuple(value for _, value in step.allowed_token_log_probabilities) == pytest.approx(
            expected
        )
        assert step.selected_log_probability == pytest.approx(
            dict(step.allowed_token_log_probabilities)[step.selected_token_id]
        )
        assert step.selected_interval_lower <= step.sampling_draw < step.selected_interval_upper


def test_boundary_merge_no_legal_token_and_overlength_fail_closed(
    short_dialogue: Dialogue,
) -> None:
    with pytest.raises(TokenizerPrefixStabilityError, match="boundary merge"):
        GrammarConstrainedPolicy(
            BoundaryMergingTokenizer(),
            FixedLogits((0.0,) * 512),
            seed=1,
        ).sample_action(short_dialogue, phase="inquiry", terminal_count=None)

    with pytest.raises(TokenizerPrefixStabilityError, match="boundary merge"):
        GrammarConstrainedPolicy(
            FieldBoundaryMergingTokenizer(),
            PreferMoveLogits("test"),
            seed=1,
        ).sample_action(short_dialogue, phase="inquiry", terminal_count=None)

    with pytest.raises(NoLegalTokenError, match="no finite logit"):
        GrammarConstrainedPolicy(
            CharacterChatTokenizer(),
            FixedLogits((-math.inf,) * 128),
            seed=1,
        ).sample_action(short_dialogue, phase="inquiry", terminal_count=None)

    with pytest.raises(ActionOverlengthError, match="truncation is forbidden"):
        GrammarConstrainedPolicy(
            CharacterChatTokenizer(),
            FixedLogits((0.0,) * 128),
            seed=1,
            maximum_action_tokens=1,
        ).sample_action(short_dialogue, phase="inquiry", terminal_count=None)


def test_manifest_states_non_enumerated_model_surface_and_factorized_answer() -> None:
    manifest = constrained_decoder_manifest()
    assert CONSTRAINED_DECODING_SCHEMA_VERSION == 1
    assert manifest["status"] == "rejected_pinned_qwen_bpe_boundary_failure"
    assert manifest["rollout_authorized"] is False
    assert manifest["weight_update_authorized"] is False
    assert manifest["candidate_koan_list_presented_to_model"] is False
    assert "without rule-by-label-list enumeration" in str(
        manifest["classification_factorization"]
    )
    assert manifest["overlength_behavior"] == "raise without truncation or repair"
    assert not hasattr(goalzendo_interactive, "GrammarConstrainedPolicy")
