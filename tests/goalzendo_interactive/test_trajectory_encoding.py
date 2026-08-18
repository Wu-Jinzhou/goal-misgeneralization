from __future__ import annotations

from collections.abc import Sequence

import pytest

from goalzendo_interactive.dialogue import DialogueMessage, render_dialogue
from goalzendo_interactive.environment import play_reference_episode
from goalzendo_interactive.episodes import HiddenEpisode
from goalzendo_interactive.trajectory_encoding import (
    IGNORE_INDEX,
    TRAJECTORY_ENCODING_SCHEMA_VERSION,
    EncodedTrajectory,
    TokenSpan,
    TrajectoryEncodingError,
    encode_sft_dialogue,
    render_generation_prefix,
)


class CharacterChatTokenizer:
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


class BoundaryMergingTokenizer(CharacterChatTokenizer):
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        values = super().encode(text, add_special_tokens=add_special_tokens)
        if text.endswith("}") and len(values) >= 2:
            return [*values[:-2], values[-2] * 256 + values[-1]]
        return values


def test_reference_trajectory_masks_only_assistant_action_content(
    hidden_episode: HiddenEpisode,
) -> None:
    transcript = play_reference_episode(hidden_episode)
    dialogue = render_dialogue(hidden_episode, transcript)
    encoded = encode_sft_dialogue(CharacterChatTokenizer(), dialogue)
    assistant_indices = [index for index, message in enumerate(dialogue) if message.role == "assistant"]
    assert [span.message_index for span in encoded.action_spans] == assistant_indices
    assert encoded.supervised_token_count == sum(
        len(dialogue[index].content) for index in assistant_indices
    )
    supervised = [label for label in encoded.labels if label != IGNORE_INDEX]
    expected = [
        ord(character)
        for index in assistant_indices
        for character in dialogue[index].content
    ]
    assert supervised == expected
    assert len(encoded.input_ids) == len(encoded.labels) == len(encoded.attention_mask)
    assert len(encoded.digest) == 64


def test_generation_prefix_ends_at_empty_assistant_content(hidden_episode: HiddenEpisode) -> None:
    initial = render_dialogue(hidden_episode)
    text, token_ids = render_generation_prefix(CharacterChatTokenizer(), initial)
    assert text.endswith("<turn:assistant>\n")
    assert token_ids == tuple(ord(character) for character in text)


def test_token_limit_fails_without_truncation(hidden_episode: HiddenEpisode) -> None:
    dialogue = render_dialogue(hidden_episode, play_reference_episode(hidden_episode))
    encoded = encode_sft_dialogue(CharacterChatTokenizer(), dialogue)
    with pytest.raises(TrajectoryEncodingError, match="truncation is forbidden"):
        encode_sft_dialogue(
            CharacterChatTokenizer(),
            dialogue,
            maximum_tokens=len(encoded.input_ids) - 1,
        )


def test_boundary_merge_fails_closed() -> None:
    dialogue = (
        DialogueMessage("system", "contract", "contract"),
        DialogueMessage("user", "act", "opening"),
        DialogueMessage("assistant", '{"move":"ready"}', "action"),
    )
    with pytest.raises(TrajectoryEncodingError, match="terminator changes"):
        encode_sft_dialogue(BoundaryMergingTokenizer(), dialogue)


def test_encoded_trajectory_rejects_bad_mask() -> None:
    with pytest.raises(TrajectoryEncodingError, match="mask"):
        EncodedTrajectory(
            input_ids=(1, 2, 3),
            labels=(IGNORE_INDEX, 999, IGNORE_INDEX),
            attention_mask=(1, 1, 1),
            action_spans=(
                # Only index one is supervised, so its label must equal token 2.
                TokenSpan(1, 2, 2),
            ),
            rendered_text_sha256="0" * 64,
            tokenizer_template_sha256="1" * 64,
        )


def test_trajectory_encoding_schema_is_explicit() -> None:
    assert TRAJECTORY_ENCODING_SCHEMA_VERSION == 1
