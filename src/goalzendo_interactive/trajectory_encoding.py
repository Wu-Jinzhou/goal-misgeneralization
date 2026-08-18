"""Exact assistant-action masking for G03 trajectory SFT and policy prompts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ._json import json_digest
from .dialogue import Dialogue, DialogueMessage, dialogue_to_chat

TRAJECTORY_ENCODING_SCHEMA_VERSION = 1
IGNORE_INDEX = -100


class TrajectoryEncodingError(ValueError):
    """Raised when a tokenizer/template cannot prove exact action boundaries."""


@runtime_checkable
class ChatTokenizerProtocol(Protocol):
    def apply_chat_template(
        self,
        conversation: Sequence[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> Any: ...

    def encode(self, text: str, *, add_special_tokens: bool = False) -> Sequence[int]: ...


@dataclass(frozen=True, slots=True)
class TokenSpan:
    start: int
    end: int
    message_index: int

    def __post_init__(self) -> None:
        for name in ("start", "end", "message_index"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TrajectoryEncodingError(f"{name} must be a non-negative integer")
        if self.end <= self.start:
            raise TrajectoryEncodingError("token span must be nonempty")

    def as_obj(self) -> dict[str, int]:
        return {"start": self.start, "end": self.end, "message_index": self.message_index}


@dataclass(frozen=True, slots=True)
class EncodedTrajectory:
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    attention_mask: tuple[int, ...]
    action_spans: tuple[TokenSpan, ...]
    rendered_text_sha256: str
    tokenizer_template_sha256: str

    def __post_init__(self) -> None:
        input_ids = tuple(self.input_ids)
        labels = tuple(self.labels)
        attention = tuple(self.attention_mask)
        spans = tuple(self.action_spans)
        object.__setattr__(self, "input_ids", input_ids)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "attention_mask", attention)
        object.__setattr__(self, "action_spans", spans)
        if not input_ids or len(input_ids) != len(labels) or len(input_ids) != len(attention):
            raise TrajectoryEncodingError("encoded trajectory arrays must have one equal positive length")
        if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in input_ids):
            raise TrajectoryEncodingError("input token ids must be non-negative integers")
        if any(type(value) is not int or value != 1 for value in attention):
            raise TrajectoryEncodingError("an unpadded trajectory must have an all-one attention mask")
        if not spans or any(type(span) is not TokenSpan for span in spans):
            raise TrajectoryEncodingError("trajectory must supervise at least one assistant action")
        supervised = {index for span in spans for index in range(span.start, span.end)}
        if any(span.end > len(input_ids) for span in spans):
            raise TrajectoryEncodingError("action span lies beyond input token sequence")
        if len(supervised) != sum(span.end - span.start for span in spans):
            raise TrajectoryEncodingError("assistant action spans may not overlap")
        for index, (token, label) in enumerate(zip(input_ids, labels, strict=True)):
            if isinstance(label, bool) or not isinstance(label, int):
                raise TrajectoryEncodingError("labels must be integer token ids or IGNORE_INDEX")
            expected = token if index in supervised else IGNORE_INDEX
            if label != expected:
                raise TrajectoryEncodingError("labels do not exactly mask non-action tokens")
        for name in ("rendered_text_sha256", "tokenizer_template_sha256"):
            value = getattr(self, name)
            if (
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise TrajectoryEncodingError(f"{name} must be a lowercase SHA-256 digest")

    @property
    def supervised_token_count(self) -> int:
        return sum(span.end - span.start for span in self.action_spans)

    @property
    def digest(self) -> str:
        return json_digest(
            {
                "schema_version": TRAJECTORY_ENCODING_SCHEMA_VERSION,
                "input_ids": list(self.input_ids),
                "labels": list(self.labels),
                "attention_mask": list(self.attention_mask),
                "action_spans": [span.as_obj() for span in self.action_spans],
                "rendered_text_sha256": self.rendered_text_sha256,
                "tokenizer_template_sha256": self.tokenizer_template_sha256,
            },
            domain="goalzendo-interactive-encoded-trajectory-v1",
        )


def _token_ids(tokenizer: ChatTokenizerProtocol, text: str) -> tuple[int, ...]:
    raw = tokenizer.encode(text, add_special_tokens=False)
    values = tuple(raw)
    if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in values):
        raise TrajectoryEncodingError("tokenizer returned an invalid token id")
    return values


def _render_template(
    tokenizer: ChatTokenizerProtocol,
    messages: Sequence[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> str:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    if type(rendered) is not str or not rendered:
        raise TrajectoryEncodingError("chat template must return nonempty text when tokenize=False")
    return rendered


def _sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _is_prefix(prefix: tuple[int, ...], complete: tuple[int, ...]) -> bool:
    return len(prefix) <= len(complete) and complete[: len(prefix)] == prefix


def render_generation_prefix(
    tokenizer: ChatTokenizerProtocol,
    messages: Sequence[DialogueMessage],
) -> tuple[str, tuple[int, ...]]:
    """Render a stable assistant-generation boundary for the next action."""

    if not isinstance(tokenizer, ChatTokenizerProtocol):
        raise TypeError("tokenizer does not implement the required chat-tokenizer protocol")
    if not messages or any(type(message) is not DialogueMessage for message in messages):
        raise TypeError("messages must be a nonempty sequence of DialogueMessage objects")
    if messages[-1].role != "user":
        raise TrajectoryEncodingError("a generation prefix must end after a user message")
    text = _render_template(
        tokenizer,
        [message.as_chat_obj() for message in messages],
        add_generation_prompt=True,
    )
    return text, _token_ids(tokenizer, text)


def encode_sft_dialogue(
    tokenizer: ChatTokenizerProtocol,
    dialogue: Dialogue,
    *,
    maximum_tokens: int | None = None,
) -> EncodedTrajectory:
    """Tokenize a complete dialogue and supervise only assistant JSON content.

    Each assistant span is derived by rendering the exact conversation prefix
    with ``add_generation_prompt=True``, appending the canonical action text,
    and proving that both this content boundary and the completed assistant
    message remain exact token prefixes of the full dialogue.  Any tokenizer
    boundary merge or template rewrite fails closed; no truncation is allowed.
    """

    if not isinstance(tokenizer, ChatTokenizerProtocol):
        raise TypeError("tokenizer does not implement the required chat-tokenizer protocol")
    if type(dialogue) is not tuple or not dialogue:
        raise TypeError("dialogue must be a nonempty Dialogue tuple")
    if any(type(message) is not DialogueMessage for message in dialogue):
        raise TypeError("dialogue contains a non-DialogueMessage")
    if maximum_tokens is not None and (
        isinstance(maximum_tokens, bool)
        or not isinstance(maximum_tokens, int)
        or maximum_tokens < 1
    ):
        raise ValueError("maximum_tokens must be a positive integer or None")

    chat = dialogue_to_chat(dialogue)
    full_text = _render_template(tokenizer, chat, add_generation_prompt=False)
    full_ids = _token_ids(tokenizer, full_text)
    if maximum_tokens is not None and len(full_ids) > maximum_tokens:
        raise TrajectoryEncodingError(
            f"trajectory has {len(full_ids)} tokens, exceeding maximum {maximum_tokens}; truncation is forbidden"
        )

    spans: list[TokenSpan] = []
    template_probes: list[dict[str, object]] = []
    for index, message in enumerate(dialogue):
        if message.role != "assistant":
            continue
        if index == 0 or dialogue[index - 1].role != "user":
            raise TrajectoryEncodingError("assistant action must immediately follow a user message")
        prior_chat = chat[:index]
        generation_text = _render_template(
            tokenizer,
            prior_chat,
            add_generation_prompt=True,
        )
        generation_ids = _token_ids(tokenizer, generation_text)
        content_text = generation_text + message.content
        content_ids = _token_ids(tokenizer, content_text)
        complete_turn_text = _render_template(
            tokenizer,
            chat[: index + 1],
            add_generation_prompt=False,
        )
        complete_turn_ids = _token_ids(tokenizer, complete_turn_text)
        if not _is_prefix(generation_ids, content_ids):
            raise TrajectoryEncodingError(
                f"assistant message {index} action text changes its generation-prefix tokenization"
            )
        if not _is_prefix(content_ids, complete_turn_ids):
            raise TrajectoryEncodingError(
                f"assistant message {index} terminator changes its action-content tokenization"
            )
        if not _is_prefix(complete_turn_ids, full_ids):
            raise TrajectoryEncodingError(
                f"assistant message {index} is not an exact prefix of the full dialogue"
            )
        if len(content_ids) == len(generation_ids):
            raise TrajectoryEncodingError(f"assistant message {index} action has zero tokens")
        spans.append(TokenSpan(len(generation_ids), len(content_ids), index))
        template_probes.append(
            {
                "message_index": index,
                "generation_text_sha256": _sha256_text(generation_text),
                "complete_turn_text_sha256": _sha256_text(complete_turn_text),
                "generation_token_count": len(generation_ids),
                "content_token_count": len(content_ids) - len(generation_ids),
                "complete_turn_token_count": len(complete_turn_ids),
            }
        )

    supervised = {token_index for span in spans for token_index in range(span.start, span.end)}
    labels = tuple(
        token if index in supervised else IGNORE_INDEX
        for index, token in enumerate(full_ids)
    )
    template_digest = json_digest(
        {
            "schema_version": TRAJECTORY_ENCODING_SCHEMA_VERSION,
            "probes": template_probes,
        },
        domain="goalzendo-interactive-chat-template-boundaries-v1",
    )
    return EncodedTrajectory(
        input_ids=full_ids,
        labels=labels,
        attention_mask=(1,) * len(full_ids),
        action_spans=tuple(spans),
        rendered_text_sha256=_sha256_text(full_text),
        tokenizer_template_sha256=template_digest,
    )
