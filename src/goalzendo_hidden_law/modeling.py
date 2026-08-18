"""Finite-action scoring for the hidden-law experiments.

This module deliberately delegates causal sequence scoring and model-family chat
rendering to the already tested GoalZendo implementations.  It adds only the
small amount of bookkeeping needed when a decision has more than two actions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import torch
from torch import Tensor, nn

from goalzendo.modeling import (
    CausalLMProtocol,
    TokenizerProtocol,
    _explicit_forward_parameter,
    encode_action_labels,
    format_chat_prompt,
)


def _require_finite(value: Tensor, name: str) -> Tensor:
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if not bool(torch.isfinite(value).all().detach().cpu()):
        raise FloatingPointError(f"{name} contains NaN or infinity")
    return value


def validate_action_labels(
    action_labels: Sequence[str],
    *,
    tokenizer: TokenizerProtocol | None = None,
) -> tuple[str, ...]:
    """Validate a finite action alphabet without silently normalizing labels.

    Labels must be non-empty, unique strings, and no label may be a textual
    prefix of another.  When a tokenizer is supplied, every pair is also
    checked with GoalZendo's proven token-level prefix test.
    """

    if isinstance(action_labels, (str, bytes)):
        raise TypeError("action_labels must be a sequence of label strings")
    labels = tuple(action_labels)
    if len(labels) < 2:
        raise ValueError("action_labels must contain at least two labels")
    if any(not isinstance(label, str) for label in labels):
        raise TypeError("each action label must be a string")
    if any(not label or not label.strip() for label in labels):
        raise ValueError("each action label must be non-empty text")
    if len(set(labels)) != len(labels):
        raise ValueError("action_labels must be unique")
    for first, second in combinations(labels, 2):
        if first.startswith(second) or second.startswith(first):
            raise ValueError("action_labels must be prefix-free")
        if tokenizer is not None:
            encode_action_labels(tokenizer, (first, second))
    return labels


def format_chat_prompts(
    tokenizer: Any,
    user_prompts: Sequence[str],
    *,
    system_prompt: str | None = None,
    enable_thinking: bool = False,
) -> tuple[str, ...]:
    """Render non-empty user prompts with the shared model-family template."""

    if isinstance(user_prompts, (str, bytes)) or not user_prompts:
        raise ValueError("user_prompts must be a non-empty sequence of strings")
    if any(not isinstance(prompt, str) or not prompt for prompt in user_prompts):
        raise ValueError("each user prompt must be non-empty text")
    return tuple(
        format_chat_prompt(
            tokenizer,
            prompt,
            system_prompt=system_prompt,
            enable_thinking=enable_thinking,
        )
        for prompt in user_prompts
    )


@dataclass(frozen=True)
class FiniteActionScores:
    """Differentiable sequence scores for a common finite action alphabet."""

    log_scores: Tensor
    action_labels: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.log_scores.ndim != 2:
            raise ValueError("log_scores must have shape [batch, actions]")
        if self.log_scores.shape[1] != len(self.action_labels):
            raise ValueError("log_scores action dimension must match action_labels")
        validate_action_labels(self.action_labels)
        _require_finite(self.log_scores, "finite action scores")

    @property
    def probabilities(self) -> Tensor:
        return _require_finite(
            self.log_scores.softmax(dim=-1),
            "finite action probabilities",
        )

    @property
    def predicted_actions(self) -> Tensor:
        return self.log_scores.argmax(dim=-1)


def _encode(tokenizer: TokenizerProtocol, text: str, *, add_special_tokens: bool) -> tuple[int, ...]:
    return tuple(int(token) for token in tokenizer.encode(text, add_special_tokens=add_special_tokens))


def _model_device(model: CausalLMProtocol) -> torch.device:
    try:
        return next(model.parameters()).device  # type: ignore[attr-defined]
    except (AttributeError, StopIteration):
        return torch.device("cpu")


def _padding_id(tokenizer: TokenizerProtocol) -> int:
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is None:
        pad = getattr(tokenizer, "eos_token_id", None)
    if isinstance(pad, bool) or not isinstance(pad, int) or pad < 0:
        raise ValueError("tokenizer must define a non-negative pad or EOS token id")
    return pad


def _contextual_encodings(
    tokenizer: TokenizerProtocol,
    prompts: Sequence[str],
    labels: tuple[str, ...],
    *,
    add_special_tokens: bool,
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[tuple[int, ...], ...], ...]]:
    if isinstance(prompts, (str, bytes)) or not prompts:
        raise ValueError("prompts must be a non-empty sequence")
    encoded_prompts: list[tuple[int, ...]] = []
    encoded_actions: list[tuple[tuple[int, ...], ...]] = []
    for row, prompt in enumerate(prompts):
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("each prompt must be non-empty text")
        prompt_tokens = _encode(tokenizer, prompt, add_special_tokens=add_special_tokens)
        if not prompt_tokens:
            raise ValueError("each prompt must encode to at least one token")
        actions: list[tuple[int, ...]] = []
        for label in labels:
            complete = _encode(tokenizer, prompt + label, add_special_tokens=add_special_tokens)
            if complete[: len(prompt_tokens)] != prompt_tokens:
                raise ValueError(
                    "tokenizer is not prefix-stable at the prompt/action boundary "
                    f"for prompt row {row} and action {label!r}"
                )
            continuation = complete[len(prompt_tokens) :]
            if not continuation:
                raise ValueError(f"action {label!r} has an empty contextual continuation")
            actions.append(continuation)
        for first, second in combinations(actions, 2):
            if first == second[: len(first)] or second == first[: len(second)]:
                raise ValueError("contextual action token sequences must be prefix-free")
        encoded_prompts.append(prompt_tokens)
        encoded_actions.append(tuple(actions))
    return tuple(encoded_prompts), tuple(encoded_actions)


def _score_single_token_actions(
    model: CausalLMProtocol,
    prompt_tokens: Sequence[tuple[int, ...]],
    action_tokens: Sequence[tuple[tuple[int, ...], ...]],
    *,
    pad_id: int,
    device: torch.device,
) -> Tensor:
    """Score K next-token actions without materializing earlier vocabulary logits."""

    batch_size = len(prompt_tokens)
    max_length = max(len(tokens) for tokens in prompt_tokens)
    position_parameter = _explicit_forward_parameter(model, ("position_ids",))
    logits_parameter = _explicit_forward_parameter(
        model,
        ("logits_to_keep", "num_logits_to_keep"),
    )
    left_padding = position_parameter is not None
    input_ids = torch.full(
        (batch_size, max_length),
        pad_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    lengths = torch.tensor([len(tokens) for tokens in prompt_tokens], device=device)
    for row, tokens in enumerate(prompt_tokens):
        length = len(tokens)
        start = max_length - length if left_padding else 0
        input_ids[row, start : start + length] = torch.tensor(
            tokens,
            dtype=torch.long,
            device=device,
        )
        attention_mask[row, start : start + length] = 1

    kwargs: dict[str, Any] = {"input_ids": input_ids, "attention_mask": attention_mask}
    if left_padding:
        position_ids = attention_mask.cumsum(dim=-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        assert position_parameter is not None
        kwargs[position_parameter] = position_ids
        if logits_parameter is not None:
            kwargs[logits_parameter] = 1
    output = model(**kwargs)
    logits = output.logits if hasattr(output, "logits") else output[0]
    if logits.ndim != 3 or logits.shape[0] != batch_size:
        raise ValueError("causal model must return three-dimensional vocabulary logits")
    _require_finite(logits, "causal model logits")
    if logits.shape[1] == 1:
        next_token_logits = logits[:, 0, :]
    elif logits.shape[1] == max_length:
        end_positions = torch.full_like(lengths, max_length - 1) if left_padding else lengths - 1
        next_token_logits = logits[
            torch.arange(batch_size, device=device),
            end_positions,
        ]
    else:
        raise ValueError("causal model returned an unexpected sequence dimension")
    action_ids = torch.tensor(
        [[action[0] for action in actions] for actions in action_tokens],
        dtype=torch.long,
        device=device,
    )
    if torch.any(action_ids < 0) or torch.any(action_ids >= next_token_logits.shape[-1]):
        raise ValueError("an action continuation token is outside the model vocabulary")
    return _require_finite(
        next_token_logits.gather(-1, action_ids),
        "selected action logits",
    )


def score_finite_action_sequences(
    model: CausalLMProtocol,
    tokenizer: TokenizerProtocol,
    prompts: Sequence[str],
    action_labels: Sequence[str],
    *,
    device: str | torch.device | None = None,
    add_prompt_special_tokens: bool = True,
) -> FiniteActionScores:
    """Score all K continuations in one causal-model forward pass.

    One-token actions use their exact next-token logits. Multi-token actions use
    the sum of continuation-token log probabilities. Prompt and complete-string
    tokenization are compared exactly at the rendered answer boundary, and all
    contextual continuations must be pairwise prefix-free.
    """

    labels = validate_action_labels(action_labels, tokenizer=tokenizer)
    prompt_tokens, action_tokens = _contextual_encodings(
        tokenizer,
        prompts,
        labels,
        add_special_tokens=add_prompt_special_tokens,
    )
    target_device = torch.device(device) if device is not None else _model_device(model)
    pad_id = _padding_id(tokenizer)

    if all(len(action) == 1 for row in action_tokens for action in row):
        matrix = _score_single_token_actions(
            model,
            prompt_tokens,
            action_tokens,
            pad_id=pad_id,
            device=target_device,
        )
        return FiniteActionScores(log_scores=matrix, action_labels=labels)

    sequences: list[tuple[int, ...]] = []
    starts: list[int] = []
    lengths: list[int] = []
    for prompt, actions in zip(prompt_tokens, action_tokens, strict=True):
        for action in actions:
            sequences.append((*prompt, *action))
            starts.append(len(prompt))
            lengths.append(len(action))
    max_length = max(len(sequence) for sequence in sequences)
    input_ids = torch.full(
        (len(sequences), max_length),
        pad_id,
        dtype=torch.long,
        device=target_device,
    )
    attention_mask = torch.zeros_like(input_ids)
    for row, sequence in enumerate(sequences):
        input_ids[row, : len(sequence)] = torch.tensor(sequence, device=target_device)
        attention_mask[row, : len(sequence)] = 1
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = output.logits if hasattr(output, "logits") else output[0]
    _require_finite(logits, "causal model logits")
    log_probabilities = _require_finite(
        logits.log_softmax(dim=-1),
        "causal model log probabilities",
    )
    scores: list[Tensor] = []
    for row, (start, length) in enumerate(zip(starts, lengths, strict=True)):
        positions = torch.arange(start - 1, start + length - 1, device=target_device)
        target_tokens = input_ids[row, start : start + length]
        scores.append(log_probabilities[row, positions, target_tokens].sum())
    matrix = _require_finite(
        torch.stack(scores).reshape(len(prompt_tokens), len(labels)),
        "finite action sequence scores",
    )
    return FiniteActionScores(log_scores=matrix, action_labels=labels)


class FiniteActionScorer(nn.Module):
    """Thin module wrapper for raw or shared-chat-template finite decisions."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: TokenizerProtocol,
        action_labels: Sequence[str],
        *,
        format_as_chat: bool = False,
        system_prompt: str | None = None,
        enable_thinking: bool = False,
        add_prompt_special_tokens: bool = True,
    ) -> None:
        super().__init__()
        if format_as_chat and add_prompt_special_tokens:
            raise ValueError(
                "chat-rendered prompts already contain special tokens; set add_prompt_special_tokens=False"
            )
        self.model = model
        self.tokenizer = tokenizer
        self.action_labels = validate_action_labels(action_labels, tokenizer=tokenizer)
        self.format_as_chat = bool(format_as_chat)
        self.system_prompt = system_prompt
        self.enable_thinking = bool(enable_thinking)
        self.add_prompt_special_tokens = bool(add_prompt_special_tokens)

    def forward(self, prompts: Sequence[str]) -> Tensor:
        rendered = (
            format_chat_prompts(
                self.tokenizer,
                prompts,
                system_prompt=self.system_prompt,
                enable_thinking=self.enable_thinking,
            )
            if self.format_as_chat
            else tuple(prompts)
        )
        return score_finite_action_sequences(
            self.model,
            self.tokenizer,
            rendered,
            self.action_labels,
            add_prompt_special_tokens=self.add_prompt_special_tokens,
        ).log_scores
