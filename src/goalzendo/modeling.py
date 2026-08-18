"""Model loading and constrained action scoring for GoalZendo.

The symbolic environment does not depend on Hugging Face.  This module keeps
that property by importing ``transformers`` and ``peft`` only inside the
functions that need them.  The core scorer itself works with any causal model
and tokenizer implementing the small protocols below, which also makes the
scientific scoring rule straightforward to test.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import platform
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class PinnedModelSpec:
    """Immutable model identifier used by the optional real-model audit."""

    name: str
    revision: str
    dtype: str = "bfloat16"

    def as_config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "revision": self.revision,
            "dtype": self.dtype,
            "trust_remote_code": False,
        }


PINNED_QWEN_MODELS: tuple[PinnedModelSpec, ...] = (
    PinnedModelSpec(
        name="Qwen/Qwen2.5-0.5B-Instruct",
        revision="7ae557604adf67be50417f59c2c2f167def9a775",
    ),
    PinnedModelSpec(
        name="Qwen/Qwen2.5-1.5B-Instruct",
        revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    ),
)

PINNED_QWEN35_MODELS: tuple[PinnedModelSpec, ...] = (
    PinnedModelSpec(
        name="Qwen/Qwen3.5-0.8B",
        revision="2fc06364715b967f1860aea9cf38778875588b17",
    ),
    PinnedModelSpec(
        name="Qwen/Qwen3.5-2B",
        revision="15852e8c16360a2fea060d615a32b45270f8a8fc",
    ),
)

DEFAULT_INTEGRATION_PROMPTS: tuple[str, ...] = (
    "Choose the koan that follows the Law.\nA: one red pyramid.\n"
    "B: one blue wedge.\nReply with exactly A or B.\nAnswer: ",
    "The Law accepts exactly one blue object touching a pyramid. The Herald says A; "
    "the Sage says B.\nA: a blue wedge touches a red pyramid.\n"
    "B: a red wedge is separate from a blue pyramid.\nReply with exactly A or B.\nAnswer: ",
)


class OptionalDependencyError(ImportError):
    """Raised when an explicitly requested model backend is not installed."""


@runtime_checkable
class TokenizerProtocol(Protocol):
    """The tokenizer surface needed by :func:`score_action_sequences`."""

    pad_token_id: int | None
    eos_token_id: int | None

    def encode(self, text: str, *, add_special_tokens: bool = ...) -> list[int]: ...


@runtime_checkable
class CausalLMProtocol(Protocol):
    """A structural protocol for causal models returning an object with logits."""

    def __call__(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
        **kwargs: Any,
    ) -> Any: ...


def _require_finite(value: Tensor, name: str) -> Tensor:
    """Return ``value`` or fail before a non-finite quantity can propagate."""

    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if not bool(torch.isfinite(value).all().detach().cpu()):
        raise FloatingPointError(f"{name} contains NaN or infinity")
    return value


@dataclass(frozen=True)
class EncodedActionContinuations:
    """Contextual action encodings at each exact prompt boundary.

    Tokenizers are not generally prefix-stable: encoding ``prompt`` and
    ``prompt + action`` can merge tokens at their boundary.  A causal score is
    scientifically meaningful only when the already-tokenized prompt is an
    exact prefix of the complete rendered sequence.  We therefore derive the
    continuation from the complete string and fail closed if that invariant
    does not hold.
    """

    prompt_tokens: tuple[tuple[int, ...], ...]
    action_tokens: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    action_labels: tuple[str, str]


@dataclass(frozen=True)
class ActionScores:
    """Scores for a constrained two-action decision.

    Multi-token values are sequence log-probabilities. Single-token values are
    raw vocabulary logits, which differ from log-probabilities only by one
    prompt-specific additive constant. Their two-way probabilities, margins,
    supervised loss, and policy gradients are therefore exactly unchanged.
    """

    log_scores: Tensor
    action_labels: tuple[str, str]
    token_lengths: tuple[int, int]
    continuation_token_ids: tuple[
        tuple[tuple[int, ...], tuple[int, ...]], ...
    ] = ()

    def __post_init__(self) -> None:
        if self.log_scores.ndim != 2 or self.log_scores.shape[1] != 2:
            raise ValueError("log_scores must have shape [batch, 2]")
        _require_finite(self.log_scores, "action scores")
        if self.continuation_token_ids and len(self.continuation_token_ids) != len(
            self.log_scores
        ):
            raise ValueError("continuation_token_ids must have one pair per prompt")

    @property
    def probabilities(self) -> Tensor:
        probabilities = _require_finite(
            self.log_scores.softmax(dim=-1),
            "constrained action probabilities",
        )
        row_sums = probabilities.sum(dim=-1)
        if not bool(
            torch.isclose(
                row_sums,
                torch.ones_like(row_sums),
                rtol=5e-3,
                atol=5e-3,
            ).all()
            .detach()
            .cpu()
        ):
            raise FloatingPointError("constrained action probabilities are not normalized")
        return probabilities

    @property
    def predicted_actions(self) -> Tensor:
        return self.log_scores.argmax(dim=-1)

    @property
    def margin_b_minus_a(self) -> Tensor:
        return _require_finite(
            self.log_scores[:, 1] - self.log_scores[:, 0],
            "action score margins",
        )


def _parameter_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return torch.device("cpu")


def _padding_id(tokenizer: TokenizerProtocol) -> int:
    if tokenizer.pad_token_id is not None:
        return int(tokenizer.pad_token_id)
    if tokenizer.eos_token_id is not None:
        return int(tokenizer.eos_token_id)
    return 0


def _explicit_forward_parameter(model: Any, names: Sequence[str]) -> str | None:
    """Find an explicitly supported kwarg, looking through common PEFT wrappers."""

    candidates = [model]
    get_base_model = getattr(model, "get_base_model", None)
    if callable(get_base_model):
        with suppress(AttributeError, TypeError):
            candidates.append(get_base_model())
    base_model = getattr(model, "base_model", None)
    if base_model is not None:
        candidates.append(base_model)
        nested = getattr(base_model, "model", None)
        if nested is not None:
            candidates.append(nested)
    seen: set[int] = set()
    for candidate in candidates:
        if id(candidate) in seen:
            continue
        seen.add(id(candidate))
        try:
            parameters = inspect.signature(candidate.forward).parameters
        except (AttributeError, TypeError, ValueError):
            continue
        for name in names:
            if name in parameters:
                return name
    return None


def _score_single_token_actions(
    model: CausalLMProtocol,
    prompt_tokens: Sequence[tuple[int, ...]],
    action_tokens: Sequence[tuple[tuple[int, ...], tuple[int, ...]]],
    *,
    pad_id: int,
    device: torch.device,
) -> Tensor:
    """Score A/B from one prompt forward, with an optional last-logit fast path."""

    batch_size = len(prompt_tokens)
    max_length = max(len(tokens) for tokens in prompt_tokens)
    position_parameter = _explicit_forward_parameter(model, ("position_ids",))
    logits_parameter = _explicit_forward_parameter(
        model,
        ("logits_to_keep", "num_logits_to_keep"),
    )
    # Left padding puts every prediction at the final position, enabling model
    # implementations that avoid materializing vocabulary logits for earlier
    # tokens. Explicit position IDs prevent pad width from changing positions.
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
        input_ids[row, start : start + length] = torch.tensor(tokens, dtype=torch.long, device=device)
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
        [[pair[0][0], pair[1][0]] for pair in action_tokens],
        dtype=torch.long,
        device=device,
    )
    # The full-vocabulary log-normalizer is common to A and B. Avoiding it saves
    # another vocabulary-sized allocation while preserving every constrained
    # two-action estimand exactly.
    if torch.any(action_ids < 0) or torch.any(action_ids >= next_token_logits.shape[-1]):
        raise ValueError("an action continuation token is outside the model vocabulary")
    return _require_finite(
        next_token_logits.gather(-1, action_ids),
        "selected action logits",
    )


def _validate_action_labels(action_labels: Sequence[str]) -> tuple[str, str]:
    labels = tuple(str(label) for label in action_labels)
    if len(labels) != 2 or labels[0] == labels[1]:
        raise ValueError("action_labels must contain two distinct strings")
    if any(not label for label in labels):
        raise ValueError("each action label must encode to at least one token")
    return labels[0], labels[1]


def encode_action_labels(
    tokenizer: TokenizerProtocol,
    action_labels: Sequence[str] = ("A", "B"),
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Tokenize exactly two action strings without adding BOS or EOS tokens."""

    labels = _validate_action_labels(action_labels)
    encoded = tuple(
        tuple(int(token) for token in tokenizer.encode(label, add_special_tokens=False))
        for label in labels
    )
    if any(not tokens for tokens in encoded):
        raise ValueError("each action label must encode to at least one token")
    if encoded[0] == encoded[1][: len(encoded[0])] or encoded[1] == encoded[0][: len(encoded[1])]:
        raise ValueError(
            "action token sequences must be prefix-free; otherwise the shorter action "
            "needs an explicit terminator"
        )
    return encoded  # type: ignore[return-value]


def _encode_text_batch(
    tokenizer: TokenizerProtocol,
    texts: Sequence[str],
    *,
    add_special_tokens: bool,
) -> tuple[tuple[int, ...], ...]:
    """Use a fast tokenizer batch call when available, with an exact fallback."""

    tokenizer_call = tokenizer if callable(tokenizer) else None
    if tokenizer_call is not None:
        try:
            encoded_batch = tokenizer_call(
                list(texts),
                add_special_tokens=add_special_tokens,
                padding=False,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
        except TypeError:
            # The intentionally small protocol only requires ``encode``; dummy
            # and non-Hugging-Face tokenizers may not implement a batch call.
            encoded_batch = None
        if encoded_batch is not None:
            if not isinstance(encoded_batch, Mapping) or "input_ids" not in encoded_batch:
                raise TypeError("tokenizer batch call must return a mapping with input_ids")
            raw_ids = encoded_batch["input_ids"]
            if not isinstance(raw_ids, Sequence) or len(raw_ids) != len(texts):
                raise ValueError("tokenizer batch call returned the wrong number of sequences")
            return tuple(tuple(int(token) for token in row) for row in raw_ids)
    return tuple(
        tuple(
            int(token)
            for token in tokenizer.encode(text, add_special_tokens=add_special_tokens)
        )
        for text in texts
    )


def encode_action_continuations(
    tokenizer: TokenizerProtocol,
    prompts: Sequence[str],
    action_labels: Sequence[str] = ("A", "B"),
    *,
    add_prompt_special_tokens: bool = True,
) -> EncodedActionContinuations:
    """Encode actions as continuations of the exact rendered prompt strings.

    Complete sequences use the same ``add_special_tokens`` setting as their
    prompt.  The prompt tokens must be an exact prefix of both complete
    sequences.  A failure usually means the rendered answer boundary permits a
    BPE merge or the tokenizer appends a terminal special token; callers must
    fix the template rather than accepting a subtly different action score.
    """

    if not prompts:
        raise ValueError("prompts must be non-empty")
    labels = _validate_action_labels(action_labels)
    normalized_prompts: list[str] = []
    for prompt in prompts:
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("each prompt must be non-empty text")
        normalized_prompts.append(prompt)
    all_texts = [*normalized_prompts]
    all_texts.extend(prompt + labels[0] for prompt in normalized_prompts)
    all_texts.extend(prompt + labels[1] for prompt in normalized_prompts)
    all_encodings = _encode_text_batch(
        tokenizer,
        all_texts,
        add_special_tokens=add_prompt_special_tokens,
    )
    count = len(normalized_prompts)
    encoded_prompts = list(all_encodings[:count])
    complete_a = all_encodings[count : 2 * count]
    complete_b = all_encodings[2 * count :]
    contextual_actions: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for row, prompt_tokens in enumerate(encoded_prompts):
        if not prompt_tokens:
            raise ValueError("each prompt must encode to at least one token")
        action_pair: list[tuple[int, ...]] = []
        for label, complete_tokens in zip(
            labels,
            (complete_a[row], complete_b[row]),
            strict=True,
        ):
            if complete_tokens[: len(prompt_tokens)] != prompt_tokens:
                raise ValueError(
                    "tokenizer is not prefix-stable at the prompt/action boundary "
                    f"for prompt row {row} and action {label!r}; adjust the rendered "
                    "separator or chat template"
                )
            continuation = complete_tokens[len(prompt_tokens) :]
            if not continuation:
                raise ValueError(
                    f"action {label!r} has an empty contextual continuation at prompt row {row}"
                )
            action_pair.append(continuation)
        first, second = action_pair
        if first == second[: len(first)] or second == first[: len(second)]:
            raise ValueError(
                "contextual action token sequences must be prefix-free; otherwise "
                "the shorter action needs an explicit terminator"
            )
        contextual_actions.append((first, second))
    return EncodedActionContinuations(
        prompt_tokens=tuple(encoded_prompts),
        action_tokens=tuple(contextual_actions),
        action_labels=labels,
    )


def score_action_sequences(
    model: CausalLMProtocol,
    tokenizer: TokenizerProtocol,
    prompts: Sequence[str],
    action_labels: Sequence[str] = ("A", "B"),
    *,
    device: str | torch.device | None = None,
    add_prompt_special_tokens: bool = True,
) -> ActionScores:
    """Score two complete action strings under a causal language model.

    Multi-token scores sum token log-probabilities in ``p(action | prompt)``.
    When both actions are one token, the mathematically equivalent raw logits
    avoid duplicating prompts and materializing a full-vocabulary log-softmax.
    No generated text is parsed, and no probability from prompt or padding
    tokens enters the score.

    Action tokens are derived by tokenizing each complete ``prompt + action``
    string and subtracting the exact prompt prefix.  This verifies the actual
    chat-template continuation boundary rather than assuming that standalone
    action tokenization is context-independent. Set
    ``add_prompt_special_tokens=False`` when ``prompt`` was already rendered by
    a chat template that includes its own control tokens.
    """

    encoded = encode_action_continuations(
        tokenizer,
        prompts,
        action_labels,
        add_prompt_special_tokens=add_prompt_special_tokens,
    )
    labels = encoded.action_labels
    prompt_tokens = encoded.prompt_tokens
    action_tokens = encoded.action_tokens

    target_device = torch.device(device) if device is not None else _parameter_device(model)
    pad_id = _padding_id(tokenizer)
    if all(len(tokens) == 1 for pair in action_tokens for tokens in pair):
        matrix = _score_single_token_actions(
            model,
            prompt_tokens,
            action_tokens,
            pad_id=pad_id,
            device=target_device,
        )
        return ActionScores(
            log_scores=matrix,
            action_labels=(labels[0], labels[1]),
            token_lengths=(1, 1),
            continuation_token_ids=action_tokens,
        )

    sequences: list[tuple[int, ...]] = []
    action_starts: list[int] = []
    action_lengths: list[int] = []
    for prompt, actions_for_prompt in zip(prompt_tokens, action_tokens, strict=True):
        for action in actions_for_prompt:
            sequences.append((*prompt, *action))
            action_starts.append(len(prompt))
            action_lengths.append(len(action))

    max_length = max(len(sequence) for sequence in sequences)
    input_ids = torch.full(
        (len(sequences), max_length),
        pad_id,
        dtype=torch.long,
        device=target_device,
    )
    attention_mask = torch.zeros_like(input_ids)
    for row, sequence in enumerate(sequences):
        length = len(sequence)
        input_ids[row, :length] = torch.tensor(sequence, dtype=torch.long, device=target_device)
        attention_mask[row, :length] = 1

    output = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = output.logits if hasattr(output, "logits") else output[0]
    if logits.ndim != 3 or logits.shape[:2] != input_ids.shape:
        raise ValueError("causal model must return logits with shape [batch, sequence, vocabulary]")
    _require_finite(logits, "causal model logits")
    log_probabilities = _require_finite(
        logits.log_softmax(dim=-1),
        "causal model log probabilities",
    )

    scores: list[Tensor] = []
    for row, (start, length) in enumerate(zip(action_starts, action_lengths, strict=True)):
        # Token at position `position` is predicted by logits at position - 1.
        prediction_positions = torch.arange(
            start - 1,
            start + length - 1,
            device=target_device,
        )
        target_tokens = input_ids[row, start : start + length]
        token_scores = log_probabilities[row, prediction_positions, target_tokens]
        scores.append(token_scores.sum())

    matrix = _require_finite(
        torch.stack(scores).reshape(len(prompts), 2),
        "action sequence scores",
    )
    return ActionScores(
        log_scores=matrix,
        action_labels=(labels[0], labels[1]),
        token_lengths=(
            max(len(pair[0]) for pair in action_tokens),
            max(len(pair[1]) for pair in action_tokens),
        ),
        continuation_token_ids=action_tokens,
    )


class TwoActionScorer(nn.Module):
    """Thin module wrapper around constrained sequence scoring."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: TokenizerProtocol,
        action_labels: Sequence[str] = ("A", "B"),
        *,
        add_prompt_special_tokens: bool = True,
    ) -> None:
        super().__init__()
        if len(action_labels) != 2:
            raise ValueError("GoalZendo requires exactly two actions")
        self.model = model
        self.tokenizer = tokenizer
        self.action_labels = (str(action_labels[0]), str(action_labels[1]))
        self.add_prompt_special_tokens = bool(add_prompt_special_tokens)
        # Fail early if a label becomes empty under a particular tokenizer.
        encode_action_labels(tokenizer, self.action_labels)

    def forward(self, prompts: Sequence[str]) -> Tensor:
        return score_action_sequences(
            self.model,
            self.tokenizer,
            prompts,
            self.action_labels,
            add_prompt_special_tokens=self.add_prompt_special_tokens,
        ).log_scores


def resolve_torch_dtype(name: str | torch.dtype) -> torch.dtype:
    """Resolve a portable dtype name used in experiment configurations."""

    if isinstance(name, torch.dtype):
        return name
    normalized = str(name).lower().replace("torch.", "")
    aliases = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in aliases:
        raise ValueError(f"unsupported model dtype: {name!r}")
    return aliases[normalized]


def format_chat_prompt(
    tokenizer: Any,
    user_prompt: str,
    *,
    system_prompt: str | None = None,
    enable_thinking: bool = False,
) -> str:
    """Render a model-family chat prompt while leaving the answer slot open."""

    if not user_prompt:
        raise ValueError("user_prompt cannot be empty")
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if apply_template is None:
        raise ValueError("chat-template mode requires tokenizer.apply_chat_template")
    messages: list[dict[str, str]] = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    try:
        rendered = apply_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError as exc:
        if enable_thinking:
            raise ValueError("this tokenizer cannot explicitly enable or disable thinking") from exc
        # Older non-reasoning tokenizers do not expose the Qwen-style keyword.
        rendered = apply_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    if not isinstance(rendered, str) or not rendered:
        raise ValueError("chat template did not return non-empty text")
    return rendered


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime_dependency_versions() -> dict[str, str | None]:
    """Versions that can change Hugging Face loading or numerical behavior."""

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": _package_version("transformers"),
        "tokenizers": _package_version("tokenizers"),
        "peft": _package_version("peft"),
        "accelerate": _package_version("accelerate"),
        "huggingface_hub": _package_version("huggingface-hub"),
        "safetensors": _package_version("safetensors"),
    }


def model_provenance(
    model: nn.Module,
    tokenizer: TokenizerProtocol,
    model_config: Mapping[str, Any],
    action_labels: Sequence[str] = ("A", "B"),
) -> dict[str, Any]:
    """Return JSON-safe model, tokenizer, revision, and action-token metadata."""

    encoded_actions = encode_action_labels(tokenizer, action_labels)
    chat_template = getattr(tokenizer, "chat_template", None)
    model_runtime_config = getattr(model, "config", None)
    resolved_revision = getattr(model_runtime_config, "_commit_hash", None)
    tokenizer_init = getattr(tokenizer, "init_kwargs", {})
    tokenizer_revision = tokenizer_init.get("_commit_hash") if isinstance(tokenizer_init, Mapping) else None
    parameters = list(model.parameters())
    return {
        "requested_model": str(model_config.get("name", "")),
        "requested_revision": str(model_config.get("revision", "main")),
        "resolved_revision": None if resolved_revision is None else str(resolved_revision),
        "tokenizer_resolved_revision": (None if tokenizer_revision is None else str(tokenizer_revision)),
        "requested_dtype": str(model_config.get("dtype", "")),
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "tokenizer_class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "tokenizer_name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "vocabulary_size": getattr(tokenizer, "vocab_size", None),
        "chat_template_sha256": (
            None
            if not isinstance(chat_template, str)
            else hashlib.sha256(chat_template.encode("utf-8")).hexdigest()
        ),
        "action_labels": [str(label) for label in action_labels],
        "action_token_ids": [list(tokens) for tokens in encoded_actions],
        "parameter_count": sum(parameter.numel() for parameter in parameters),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in parameters if parameter.requires_grad
        ),
        "dependency_versions": runtime_dependency_versions(),
        # Retain these flat keys for compatibility with earlier manifests.
        "torch_version": torch.__version__,
        "transformers_version": _package_version("transformers"),
        "peft_version": _package_version("peft"),
    }


def build_lora_config(update_config: Mapping[str, Any]) -> Any:
    """Construct a PEFT LoRA configuration, importing PEFT only on demand."""

    try:
        from peft import LoraConfig, TaskType  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised without optional environment
        raise OptionalDependencyError("LoRA updates require the optional 'peft' package") from exc
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(update_config.get("rank", 8)),
        lora_alpha=int(update_config.get("alpha", 16)),
        lora_dropout=float(update_config.get("dropout", 0.0)),
        target_modules=list(update_config.get("target_modules", ())) or None,
        bias=str(update_config.get("bias", "none")),
    )


def apply_update_method(model: nn.Module, update_config: Mapping[str, Any]) -> nn.Module:
    """Apply the configured trainable-parameter scheme to a causal model."""

    method = str(update_config.get("method", "lora")).lower()
    if method == "full":
        model.requires_grad_(True)
        return model
    if method == "frozen":
        model.requires_grad_(False)
        return model
    if method != "lora":
        raise ValueError(f"unsupported update method: {method!r}")
    try:
        from peft import get_peft_model
    except ImportError as exc:  # pragma: no cover - exercised without optional environment
        raise OptionalDependencyError("LoRA updates require the optional 'peft' package") from exc
    return get_peft_model(model, build_lora_config(update_config))


def load_model_and_tokenizer(
    model_config: Mapping[str, Any],
    update_config: Mapping[str, Any] | None = None,
    *,
    device_map: str | Mapping[str, Any] | None = None,
) -> tuple[nn.Module, Any]:
    """Load a Hugging Face causal LM and optionally attach LoRA adapters."""

    try:
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForCausalLM,
            AutoTokenizer,
        )
    except ImportError as exc:  # pragma: no cover - exercised without optional environment
        raise OptionalDependencyError("model loading requires the optional 'transformers' package") from exc

    name = str(model_config["name"])
    revision = str(model_config.get("revision", "main"))
    trust_remote_code = bool(model_config.get("trust_remote_code", False))
    tokenizer = AutoTokenizer.from_pretrained(
        name,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer defines neither pad_token_id nor eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token
    kwargs: dict[str, Any] = {
        "revision": revision,
        "dtype": resolve_torch_dtype(model_config.get("dtype", "bfloat16")),
        "trust_remote_code": trust_remote_code,
    }
    if device_map is not None:
        kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    if update_config is not None:
        model = apply_update_method(model, update_config)
    return model, tokenizer


IntegrationModelLoader = Callable[
    [Mapping[str, Any], Mapping[str, Any] | None],
    tuple[nn.Module, Any],
]


def run_model_integration_check(
    model_config: Mapping[str, Any],
    *,
    prompts: Sequence[str] = DEFAULT_INTEGRATION_PROMPTS,
    action_labels: Sequence[str] = ("A", "B"),
    device: str | torch.device | None = None,
    max_prompt_tokens: int | None = None,
    system_prompt: str = (
        "Follow the official Law. Reply with exactly one answer label and no explanation."
    ),
    model_loader: IntegrationModelLoader | None = None,
    strict_revision: bool = True,
    swap_atol: float = 2e-3,
    swap_rtol: float = 2e-3,
) -> dict[str, Any]:
    """Run a small, fail-closed audit at the real chat-model boundary.

    The function is dependency-injectable for offline tests. With the default
    loader it downloads/loads the exact requested Hugging Face revision. It
    checks contextual A/B tokenization, normalized finite constrained scores,
    invariance to swapping the action-column order, revision provenance, and
    prompt lengths. It performs no training and never broadens the model set.
    """

    if not prompts:
        raise ValueError("integration prompts must be non-empty")
    if max_prompt_tokens is not None and max_prompt_tokens < 1:
        raise ValueError("max_prompt_tokens must be positive when supplied")
    if swap_atol < 0 or swap_rtol < 0:
        raise ValueError("swap tolerances must be non-negative")
    target_device = (
        torch.device(device)
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    loader = load_model_and_tokenizer if model_loader is None else model_loader
    model, tokenizer = loader(model_config, {"method": "frozen"})
    model.to(target_device)
    model.eval()

    rendered_prompts = tuple(
        format_chat_prompt(
            tokenizer,
            prompt,
            system_prompt=system_prompt,
            enable_thinking=False,
        )
        for prompt in prompts
    )
    contextual = encode_action_continuations(
        tokenizer,
        rendered_prompts,
        action_labels,
        add_prompt_special_tokens=False,
    )
    prompt_counts = [len(tokens) for tokens in contextual.prompt_tokens]
    observed_max = max(prompt_counts)
    if max_prompt_tokens is not None and observed_max > max_prompt_tokens:
        raise ValueError(
            f"integration prompt length {observed_max} exceeds limit {max_prompt_tokens}"
        )

    with torch.no_grad():
        scores_ab = score_action_sequences(
            model,
            tokenizer,
            rendered_prompts,
            action_labels,
            device=target_device,
            add_prompt_special_tokens=False,
        )
        scores_ba = score_action_sequences(
            model,
            tokenizer,
            rendered_prompts,
            tuple(reversed(tuple(action_labels))),
            device=target_device,
            add_prompt_special_tokens=False,
        )
        probabilities_ab = scores_ab.probabilities
        probabilities_ba = scores_ba.probabilities
        normalized_log_scores_ab = _require_finite(
            scores_ab.log_scores.float().log_softmax(dim=-1),
            "normalized integration action scores",
        )

    expected_swapped_scores = scores_ab.log_scores[:, [1, 0]]
    expected_swapped_probabilities = probabilities_ab[:, [1, 0]]
    score_swap_error = float(
        (scores_ba.log_scores - expected_swapped_scores).abs().max().detach().cpu()
    )
    probability_swap_error = float(
        (probabilities_ba - expected_swapped_probabilities).abs().max().detach().cpu()
    )
    score_swap_ok = torch.allclose(
        scores_ba.log_scores,
        expected_swapped_scores,
        atol=swap_atol,
        rtol=swap_rtol,
    )
    probability_swap_ok = torch.allclose(
        probabilities_ba,
        expected_swapped_probabilities,
        atol=swap_atol,
        rtol=swap_rtol,
    )
    if not score_swap_ok or not probability_swap_ok:
        raise AssertionError(
            "A/B action-order swap changed constrained scores beyond tolerance: "
            f"score_error={score_swap_error}, probability_error={probability_swap_error}"
        )

    provenance = model_provenance(model, tokenizer, model_config, action_labels)
    requested_revision = str(model_config.get("revision", "main"))
    if strict_revision and requested_revision != "main":
        for key in ("resolved_revision", "tokenizer_resolved_revision"):
            resolved = provenance.get(key)
            if resolved is not None and str(resolved) != requested_revision:
                raise ValueError(
                    f"{key}={resolved!r} does not match requested revision "
                    f"{requested_revision!r}"
                )

    parameters = list(model.parameters())
    parameter_devices = sorted({str(parameter.device) for parameter in parameters})
    parameter_dtypes = sorted({str(parameter.dtype) for parameter in parameters})
    row_sums = probabilities_ab.sum(dim=-1)
    normalized_error = float((row_sums - 1).abs().max().detach().cpu())
    return {
        "passed": True,
        "model": provenance,
        "runtime": {
            "requested_device": str(target_device),
            "parameter_devices": parameter_devices,
            "parameter_dtypes": parameter_dtypes,
            "dependencies": runtime_dependency_versions(),
            "last_logit_parameter": _explicit_forward_parameter(
                model,
                ("logits_to_keep", "num_logits_to_keep"),
            ),
        },
        "action_boundary": {
            "labels": list(contextual.action_labels),
            "standalone_token_ids": [
                list(tokens) for tokens in encode_action_labels(tokenizer, action_labels)
            ],
            "continuation_token_ids_by_prompt": [
                [list(pair[0]), list(pair[1])] for pair in contextual.action_tokens
            ],
            "continuation_token_lengths_by_prompt": [
                [len(pair[0]), len(pair[1])] for pair in contextual.action_tokens
            ],
            "prefix_stable": True,
        },
        "prompt_tokens": {
            "counts": prompt_counts,
            "maximum": observed_max,
            "configured_maximum": max_prompt_tokens,
        },
        "scores": {
            "finite": True,
            "normalized_log_scores": normalized_log_scores_ab.detach().cpu().tolist(),
            "probabilities": probabilities_ab.detach().float().cpu().tolist(),
            "maximum_normalization_error": normalized_error,
            "swap_invariant": True,
            "maximum_score_swap_error": score_swap_error,
            "maximum_probability_swap_error": probability_swap_error,
            "swap_atol": swap_atol,
            "swap_rtol": swap_rtol,
        },
    }


def _integration_check_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit pinned GoalZendo Qwen model boundaries")
    parser.add_argument(
        "--integration-check-qwen",
        action="store_true",
        help="load and check both exact Qwen revisions used by GoalZendo",
    )
    parser.add_argument(
        "--model-family",
        choices=("historical", "qwen35"),
        default="historical",
        help="select the exact pinned model pair to check",
    )
    parser.add_argument("--device", default=None, help="torch device (default: CUDA if available)")
    parser.add_argument("--max-prompt-tokens", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None, help="optional JSON report path")
    args = parser.parse_args(argv)
    if not args.integration_check_qwen:
        parser.error("pass --integration-check-qwen to run the optional network/model audit")

    specs = PINNED_QWEN35_MODELS if args.model_family == "qwen35" else PINNED_QWEN_MODELS
    reports: list[dict[str, Any]] = []
    for spec in specs:
        reports.append(
            run_model_integration_check(
                spec.as_config(),
                device=args.device,
                max_prompt_tokens=args.max_prompt_tokens,
            )
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    payload = {
        "model_family": args.model_family,
        "pinned_models": [spec.as_config() for spec in specs],
        "reports": reports,
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised explicitly on GPU hosts
    raise SystemExit(_integration_check_main())
