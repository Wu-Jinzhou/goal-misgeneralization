"""Rejected schema-v1 constrained-sampling prototype for G03 actions.

This module is retained to preserve the exact engineering failure and its
tests.  A pinned-Qwen audit showed that its cumulative-text prefix assumption
is false under BPE boundary merges.  It is deliberately not re-exported from
the package root and cannot authorize a rollout or weight update.

The decoder walks :class:`InquiryActionState` or :class:`AnswerActionState`
one public grammar field at a time.  A model callback receives only the token
history.  It is never given a list of scenes, rules, or completed actions.

Every field is compiled into a token-prefix trie in its actual chat-template
context.  Tokenization must extend the already-emitted prefix exactly; a
boundary merge fails closed.  The answer rule field therefore needs one trie
over the 18,760 rule strings, but classifications are decoded afterward and
are never multiplied into a rule-by-classification action table.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias, cast, runtime_checkable

from .action_language import AnswerActionState, GrammarSegment, InquiryActionState
from .actions import AnswerAction, ReadyAction, TestAction, serialize_action
from .dialogue import Dialogue, DialogueMessage
from .rollouts import PolicyPhase, PolicyTurn, dialogue_prompt_digest
from .trajectory_encoding import ChatTokenizerProtocol, render_generation_prefix

CONSTRAINED_DECODING_SCHEMA_VERSION = 1


class ConstrainedDecodingError(RuntimeError):
    """Base class for fail-closed constrained-decoding errors."""


class TokenizerPrefixStabilityError(ConstrainedDecodingError):
    """Raised when an extension changes already-committed token IDs."""


class TokenGrammarError(ConstrainedDecodingError):
    """Raised when tokenization cannot represent an exact grammar branch."""


class NoLegalTokenError(ConstrainedDecodingError):
    """Raised when a model distribution assigns no mass to a legal token."""


class ActionOverlengthError(ConstrainedDecodingError):
    """Raised instead of truncating an action beyond its registered bound."""


@runtime_checkable
class LogitsCallbackProtocol(Protocol):
    """Return one vocabulary logit for the next token after ``input_ids``."""

    def __call__(self, input_ids: tuple[int, ...]) -> Sequence[float]: ...


@dataclass(frozen=True, slots=True)
class AllowedTokenMask:
    """A sparse exact Boolean vocabulary mask for one decoding step."""

    vocabulary_size: int
    allowed_token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.vocabulary_size, bool)
            or not isinstance(self.vocabulary_size, int)
            or self.vocabulary_size < 1
        ):
            raise ValueError("vocabulary_size must be a positive integer")
        values = tuple(self.allowed_token_ids)
        object.__setattr__(self, "allowed_token_ids", values)
        if not values:
            raise ValueError("an allowed-token mask cannot be empty")
        if values != tuple(sorted(set(values))):
            raise ValueError("allowed token ids must be unique and increasing")
        if any(
            isinstance(token, bool)
            or not isinstance(token, int)
            or token < 0
            or token >= self.vocabulary_size
            for token in values
        ):
            raise ValueError("allowed token id lies outside the vocabulary")

    def allows(self, token_id: int) -> bool:
        return token_id in self.allowed_token_ids

    @property
    def dense(self) -> tuple[bool, ...]:
        allowed = set(self.allowed_token_ids)
        return tuple(token in allowed for token in range(self.vocabulary_size))

    def mask_logits(self, logits: Sequence[float]) -> tuple[float, ...]:
        """Return logits with every grammar-illegal vocabulary entry at ``-inf``."""

        if len(logits) != self.vocabulary_size:
            raise ValueError("logit vector length differs from mask vocabulary size")
        allowed = set(self.allowed_token_ids)
        return tuple(
            _coerce_logit(logits[token], token_id=token) if token in allowed else -math.inf
            for token in range(self.vocabulary_size)
        )


@dataclass(frozen=True, slots=True)
class TokenSamplingStep:
    """Auditable evidence for one sample from the masked next-token policy."""

    action_token_index: int
    context_token_count: int
    mask: AllowedTokenMask
    allowed_token_log_probabilities: tuple[tuple[int, float], ...]
    selected_token_id: int
    selected_log_probability: float
    entropy: float
    sampling_draw: float
    selected_interval_lower: float
    selected_interval_upper: float

    def __post_init__(self) -> None:
        for name in ("action_token_index", "context_token_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if type(self.mask) is not AllowedTokenMask:
            raise ValueError("sampling step requires an AllowedTokenMask")
        pairs = tuple(self.allowed_token_log_probabilities)
        object.__setattr__(self, "allowed_token_log_probabilities", pairs)
        if tuple(token for token, _ in pairs) != self.mask.allowed_token_ids:
            raise ValueError("logged alternatives must exactly match the allowed-token mask")
        if not self.mask.allows(self.selected_token_id):
            raise ValueError("selected token is not grammar-legal")
        selected_values = [value for token, value in pairs if token == self.selected_token_id]
        if len(selected_values) != 1 or selected_values[0] != self.selected_log_probability:
            raise ValueError("selected log probability differs from the masked distribution")
        if not math.isfinite(self.selected_log_probability) or self.selected_log_probability > 1e-7:
            raise ValueError("selected log probability must be finite and <= 0")
        if not math.isfinite(self.entropy) or self.entropy < 0:
            raise ValueError("token entropy must be finite and non-negative")
        if not 0.0 <= self.sampling_draw < 1.0:
            raise ValueError("sampling draw must lie in [0, 1)")
        if not (
            0.0 <= self.selected_interval_lower
            <= self.sampling_draw
            < self.selected_interval_upper
            <= 1.0
        ):
            raise ValueError("sampling draw does not select the recorded probability interval")


TokenStepObserver: TypeAlias = Callable[[TokenSamplingStep], None]


@dataclass(slots=True)
class _TrieNode:
    children: dict[int, _TrieNode]
    option_key: str | None = None

    @classmethod
    def empty(cls) -> _TrieNode:
        return cls({})


def _token_ids(tokenizer: ChatTokenizerProtocol, text: str) -> tuple[int, ...]:
    raw = tokenizer.encode(text, add_special_tokens=False)
    values = tuple(raw)
    if any(
        isinstance(token, bool) or not isinstance(token, int) or token < 0
        for token in values
    ):
        raise TokenGrammarError("tokenizer returned an invalid token id")
    return values


def _is_prefix(prefix: tuple[int, ...], complete: tuple[int, ...]) -> bool:
    return len(prefix) <= len(complete) and complete[: len(prefix)] == prefix


def _extension_tokens(
    tokenizer: ChatTokenizerProtocol,
    *,
    base_text: str,
    base_ids: tuple[int, ...],
    extension: str,
) -> tuple[int, ...]:
    if type(extension) is not str or not extension:
        raise TokenGrammarError("grammar extension must be nonempty text")
    complete_ids = _token_ids(tokenizer, base_text + extension)
    if not _is_prefix(base_ids, complete_ids):
        raise TokenizerPrefixStabilityError(
            "tokenizer boundary merge changes an already-committed action prefix"
        )
    suffix = complete_ids[len(base_ids) :]
    if not suffix:
        raise TokenGrammarError("nonempty grammar extension emitted no token")
    return suffix


def _build_option_trie(option_sequences: Sequence[tuple[str, tuple[int, ...]]]) -> _TrieNode:
    root = _TrieNode.empty()
    seen_keys: set[str] = set()
    for option_key, tokens in option_sequences:
        if type(option_key) is not str or not option_key or option_key in seen_keys:
            raise TokenGrammarError("token-trie option keys must be unique nonempty strings")
        seen_keys.add(option_key)
        if not tokens:
            raise TokenGrammarError("token-trie branch cannot be empty")
        node = root
        for token in tokens:
            if node.option_key is not None:
                raise TokenGrammarError("one grammar option tokenization prefixes another")
            node = node.children.setdefault(token, _TrieNode.empty())
        if node.option_key is not None:
            raise TokenGrammarError("two grammar options have the same tokenization")
        if node.children:
            raise TokenGrammarError("one grammar option tokenization prefixes another")
        node.option_key = option_key
    return root


def _coerce_logit(value: object, *, token_id: int) -> float:
    if isinstance(value, bool):
        raise NoLegalTokenError(f"model returned a Boolean logit for token {token_id}")
    try:
        result = float(cast(Any, value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise NoLegalTokenError(f"model returned a nonnumeric logit for token {token_id}") from exc
    if math.isnan(result) or result == math.inf:
        raise NoLegalTokenError(f"model returned an invalid logit for token {token_id}")
    return result


def _masked_log_distribution(
    logits: Sequence[float],
    mask: AllowedTokenMask,
    *,
    temperature: float,
) -> tuple[tuple[tuple[int, float], ...], tuple[float, ...], float]:
    scaled = tuple(
        _coerce_logit(logits[token], token_id=token) / temperature
        for token in mask.allowed_token_ids
    )
    finite = [value for value in scaled if math.isfinite(value)]
    if not finite:
        raise NoLegalTokenError("model assigns no finite logit to any grammar-legal token")
    maximum = max(finite)
    weights = tuple(0.0 if value == -math.inf else math.exp(value - maximum) for value in scaled)
    total = math.fsum(weights)
    if not math.isfinite(total) or total <= 0.0:
        raise NoLegalTokenError("masked next-token distribution has zero probability mass")
    probabilities = tuple(weight / total for weight in weights)
    log_normalizer = maximum + math.log(total)
    log_probabilities = tuple(
        -math.inf if value == -math.inf else value - log_normalizer for value in scaled
    )
    entropy = max(
        0.0,
        -math.fsum(
            probability * log_probability
            for probability, log_probability in zip(
                probabilities, log_probabilities, strict=True
            )
            if probability > 0.0
        ),
    )
    return (
        tuple(zip(mask.allowed_token_ids, log_probabilities, strict=True)),
        probabilities,
        entropy,
    )


def _sample_index(
    probabilities: tuple[float, ...],
    *,
    draw: float,
) -> tuple[int, float, float]:
    cumulative = 0.0
    positive_indices = [index for index, probability in enumerate(probabilities) if probability > 0.0]
    if not positive_indices:
        raise NoLegalTokenError("masked next-token distribution has no positive probability")
    final_positive = positive_indices[-1]
    for index, probability in enumerate(probabilities):
        lower = cumulative
        cumulative += probability
        upper = 1.0 if index == final_positive else min(1.0, cumulative)
        if probability > 0.0 and lower <= draw < upper:
            return index, lower, upper
    raise NoLegalTokenError("floating-point categorical sampler could not select a token")


ActionState: TypeAlias = InquiryActionState | AnswerActionState


class GrammarConstrainedPolicy:
    """Stateful seeded policy implementing the G03 rollout sampling contract."""

    def __init__(
        self,
        tokenizer: ChatTokenizerProtocol,
        logits_callback: LogitsCallbackProtocol,
        *,
        seed: int,
        maximum_action_tokens: int = 2_048,
        temperature: float = 1.0,
        observer: TokenStepObserver | None = None,
    ) -> None:
        if not isinstance(tokenizer, ChatTokenizerProtocol):
            raise TypeError("tokenizer does not implement ChatTokenizerProtocol")
        if not callable(logits_callback):
            raise TypeError("logits_callback must be callable")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        if (
            isinstance(maximum_action_tokens, bool)
            or not isinstance(maximum_action_tokens, int)
            or maximum_action_tokens < 1
        ):
            raise ValueError("maximum_action_tokens must be a positive integer")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(float(temperature))
            or float(temperature) <= 0.0
        ):
            raise ValueError("temperature must be a finite positive number")
        if observer is not None and not callable(observer):
            raise TypeError("observer must be callable or None")
        self._tokenizer = tokenizer
        self._logits_callback = logits_callback
        self._rng = random.Random(seed)
        self._maximum_action_tokens = maximum_action_tokens
        self._temperature = float(temperature)
        self._observer = observer

    def _sample_token(
        self,
        *,
        context_ids: tuple[int, ...],
        allowed_token_ids: tuple[int, ...],
        action_token_index: int,
    ) -> tuple[int, float, float]:
        if action_token_index >= self._maximum_action_tokens:
            raise ActionOverlengthError(
                f"action exceeds maximum_action_tokens={self._maximum_action_tokens}; truncation is forbidden"
            )
        logits = self._logits_callback(context_ids)
        try:
            vocabulary_size = len(logits)
        except TypeError as exc:
            raise NoLegalTokenError("model callback did not return a sized logit vector") from exc
        if vocabulary_size < 1:
            raise NoLegalTokenError("model callback returned an empty logit vector")
        if any(token >= vocabulary_size for token in context_ids):
            raise NoLegalTokenError("model vocabulary cannot represent a context token id")
        try:
            mask = AllowedTokenMask(vocabulary_size, allowed_token_ids)
        except ValueError as exc:
            raise NoLegalTokenError(
                "model vocabulary cannot represent every grammar-legal next token"
            ) from exc
        logged, probabilities, entropy = _masked_log_distribution(
            logits,
            mask,
            temperature=self._temperature,
        )
        draw = self._rng.random()
        selected_index, lower, upper = _sample_index(probabilities, draw=draw)
        selected_token = mask.allowed_token_ids[selected_index]
        selected_log_probability = logged[selected_index][1]
        step = TokenSamplingStep(
            action_token_index=action_token_index,
            context_token_count=len(context_ids),
            mask=mask,
            allowed_token_log_probabilities=logged,
            selected_token_id=selected_token,
            selected_log_probability=selected_log_probability,
            entropy=entropy,
            sampling_draw=draw,
            selected_interval_lower=lower,
            selected_interval_upper=upper,
        )
        if self._observer is not None:
            self._observer(step)
        return selected_token, selected_log_probability, entropy

    def _sample_choice(
        self,
        *,
        base_text: str,
        base_ids: tuple[int, ...],
        choices: Sequence[tuple[str, str]],
        action_ids: list[int],
        log_probabilities: list[float],
        entropies: list[float],
    ) -> tuple[str, str, tuple[int, ...]]:
        sequences = tuple(
            (
                key,
                _extension_tokens(
                    self._tokenizer,
                    base_text=base_text,
                    base_ids=base_ids,
                    extension=extension,
                ),
            )
            for key, extension in choices
        )
        trie = _build_option_trie(sequences)
        node = trie
        selected_suffix: list[int] = []
        current_ids = list(base_ids)
        while node.option_key is None:
            allowed = tuple(sorted(node.children))
            if not allowed:
                raise TokenGrammarError("token trie reached a dead nonterminal state")
            selected, log_probability, entropy = self._sample_token(
                context_ids=tuple(current_ids),
                allowed_token_ids=allowed,
                action_token_index=len(action_ids),
            )
            current_ids.append(selected)
            selected_suffix.append(selected)
            action_ids.append(selected)
            log_probabilities.append(log_probability)
            entropies.append(entropy)
            node = node.children[selected]
        selected_key = node.option_key
        extension_by_key = dict(choices)
        if len(extension_by_key) != len(choices):
            raise TokenGrammarError("grammar choice keys must be unique")
        extension = extension_by_key[selected_key]
        expected_suffix = dict(sequences)[selected_key]
        if tuple(selected_suffix) != expected_suffix:
            raise TokenGrammarError("sampled token path differs from its grammar option")
        return selected_key, base_text + extension, tuple(current_ids)

    def _sample_segment(
        self,
        segment: GrammarSegment,
        *,
        base_text: str,
        base_ids: tuple[int, ...],
        action_ids: list[int],
        log_probabilities: list[float],
        entropies: list[float],
    ) -> tuple[str, str, tuple[int, ...]]:
        return self._sample_choice(
            base_text=base_text,
            base_ids=base_ids,
            choices=tuple(
                (option.key, segment.prefix + option.text) for option in segment.options
            ),
            action_ids=action_ids,
            log_probabilities=log_probabilities,
            entropies=entropies,
        )

    def _prove_assistant_terminator(
        self,
        dialogue: Dialogue,
        *,
        content_text: str,
        content_ids: tuple[int, ...],
        raw_action: str,
    ) -> None:
        complete_dialogue = (
            *dialogue,
            DialogueMessage("assistant", raw_action, "action"),
        )
        rendered = self._tokenizer.apply_chat_template(
            [message.as_chat_obj() for message in complete_dialogue],
            tokenize=False,
            add_generation_prompt=False,
        )
        if type(rendered) is not str or not rendered:
            raise TokenizerPrefixStabilityError(
                "chat template returned no text for the completed assistant turn"
            )
        if not rendered.startswith(content_text):
            raise TokenizerPrefixStabilityError(
                "completed assistant template changes its generation/content text prefix"
            )
        complete_ids = _token_ids(self._tokenizer, rendered)
        if not _is_prefix(content_ids, complete_ids):
            raise TokenizerPrefixStabilityError(
                "assistant terminator changes the emitted action tokenization"
            )

    def sample_action(
        self,
        dialogue: Dialogue,
        *,
        phase: PolicyPhase,
        terminal_count: int | None,
    ) -> PolicyTurn:
        """Sample exactly one canonical action under token-level grammar masks."""

        if type(dialogue) is not tuple or not dialogue:
            raise TypeError("dialogue must be a nonempty Dialogue tuple")
        if phase == "inquiry":
            if terminal_count is not None:
                raise ValueError("inquiry decoding requires terminal_count=None")
            state: ActionState = InquiryActionState()
        elif phase == "answer":
            if (
                isinstance(terminal_count, bool)
                or not isinstance(terminal_count, int)
                or terminal_count < 1
            ):
                raise ValueError("answer decoding requires a positive terminal_count")
            state = AnswerActionState(terminal_count)
        else:
            raise ValueError(f"unknown policy phase: {phase!r}")

        prompt_text, prompt_ids = render_generation_prefix(self._tokenizer, dialogue)
        base_text = prompt_text
        base_ids = prompt_ids
        action_ids: list[int] = []
        log_probabilities: list[float] = []
        entropies: list[float] = []

        while not state.complete:
            segment = state.next_segment()
            selected_key, base_text, base_ids = self._sample_segment(
                segment,
                base_text=base_text,
                base_ids=base_ids,
                action_ids=action_ids,
                log_probabilities=log_probabilities,
                entropies=entropies,
            )
            state = state.choose(selected_key)
            if not base_text.endswith(state.emitted):
                raise TokenGrammarError("grammar state text differs from sampled token branch")

        _, base_text, base_ids = self._sample_choice(
            base_text=base_text,
            base_ids=base_ids,
            choices=(("completion", state.completion_suffix),),
            action_ids=action_ids,
            log_probabilities=log_probabilities,
            entropies=entropies,
        )
        raw_action = state.text
        if base_text != prompt_text + raw_action:
            raise TokenGrammarError("sampled text differs from the completed grammar state")
        if _token_ids(self._tokenizer, base_text) != base_ids:
            raise TokenizerPrefixStabilityError(
                "completed action does not reproduce the incrementally committed token IDs"
            )

        action = state.action
        if type(action) not in {TestAction, ReadyAction, AnswerAction}:
            raise TokenGrammarError("grammar constructed an unsupported action type")
        if serialize_action(action) != raw_action:
            raise TokenGrammarError("grammar output is not canonical action JSON")
        self._prove_assistant_terminator(
            dialogue,
            content_text=base_text,
            content_ids=base_ids,
            raw_action=raw_action,
        )
        return PolicyTurn(
            raw_action=raw_action,
            token_ids=tuple(action_ids),
            token_log_probabilities=tuple(log_probabilities),
            token_entropies=tuple(entropies),
            prompt_digest=dialogue_prompt_digest(dialogue),
        )


def constrained_decoder_manifest() -> dict[str, object]:
    """Describe the bounded decoder's fixed scientific interface."""

    return {
        "schema_version": CONSTRAINED_DECODING_SCHEMA_VERSION,
        "status": "rejected_pinned_qwen_bpe_boundary_failure",
        "rollout_authorized": False,
        "weight_update_authorized": False,
        "known_blocker": (
            "cumulative field retokenization is not prefix-stable under the "
            "pinned Qwen BPE tokenizer"
        ),
        "state_machines": ["InquiryActionState", "AnswerActionState"],
        "mask_semantics": "softmax over exact token-trie children; all other vocabulary logits are -inf",
        "prefix_stability": "proved after every grammar field and at the assistant terminator",
        "rule_grammar": "one prefix trie over 18760 rules",
        "classification_factorization": "decoded after rule selection without rule-by-label-list enumeration",
        "candidate_koan_list_presented_to_model": False,
        "overlength_behavior": "raise without truncation or repair",
        "recorded_statistics": [
            "selected_token_ids",
            "selected_masked_log_probabilities",
            "masked_entropies",
        ],
    }
