"""Decision-level SFT encoding and differentiable G03 action rescoring.

The first G03 trajectory encoder tokenized a complete multi-turn chat.  That
is not compatible with the version-2 fragment action contract: earlier
assistant messages are canonically rendered by the chat template, while the
action being predicted must use the exact standalone-fragment token trace used
by the behavior policy.  This module therefore emits one training example per
assistant decision.

Every example consists of a canonically rendered history prompt followed by a
verified :class:`~goalzendo_interactive.action_tokenization_v2.ActionTokenTrace`.
Prompt tokens are ignored by SFT.  The same trace supplies the sparse legal
token masks used to recompute differentiable outcome-RL log probabilities.
Nothing in this module loads a model or authorizes a weight update.
"""

from __future__ import annotations

import hashlib
import hmac
import math
from dataclasses import dataclass
from typing import cast

import torch

from ._json import CanonicalJSONError, dump_json, json_digest, load_json
from .action_tokenization_v2 import (
    ActionTokenTrace,
    ExactDecodeTokenizerProtocol,
    FragmentActionTokenCompiler,
    FragmentActionTokenizationError,
)
from .actions import Action, parse_action
from .dialogue import Dialogue, dialogue_as_obj
from .trajectory_encoding import IGNORE_INDEX, render_generation_prefix

DECISION_ENCODING_SCHEMA_VERSION = 1
DECISION_ENCODING_CONTRACT_ID = "goalzendo-decision-fragment-target-v1"
DETACHED_MASKED_STATISTICS_SCHEMA_VERSION = 1

_EXAMPLE_DIGEST_DOMAIN = "goalzendo-interactive-decision-token-example-v1"
_PROMPT_TOKEN_DIGEST_DOMAIN = "goalzendo-interactive-decision-prompt-tokens-v1"
_DETACHED_STATISTICS_DOMAIN = "goalzendo-interactive-detached-masked-statistics-v1"
_VERIFIED_EXAMPLE_DOMAIN = "goalzendo-interactive-verified-decision-example-v1"


class DecisionEncodingError(ValueError):
    """Raised when a decision example or masked replay is not exact."""


class DecisionOverlengthError(DecisionEncodingError):
    """Raised only when an otherwise valid decision exceeds a frozen bound."""


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_token_ids(values: tuple[int, ...], *, name: str) -> None:
    if not values:
        raise DecisionEncodingError(f"{name} must be nonempty")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise DecisionEncodingError(f"{name} must contain non-negative integer token ids")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _prompt_token_digest(token_ids: tuple[int, ...]) -> str:
    return json_digest(list(token_ids), domain=_PROMPT_TOKEN_DIGEST_DOMAIN)


def _require_exact_object(
    value: object,
    fields: tuple[str, ...],
    *,
    name: str,
) -> dict[str, object]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise DecisionEncodingError(f"{name} must be a JSON object")
    expected = set(fields)
    actual = set(value)
    if actual != expected or len(value) != len(fields):
        missing = sorted(expected.difference(actual))
        extra = sorted(actual.difference(expected))
        raise DecisionEncodingError(f"{name} has noncanonical fields; missing={missing}, extra={extra}")
    return cast(dict[str, object], value)


def _require_integer_tuple(
    value: object,
    *,
    name: str,
    nonnegative: bool,
) -> tuple[int, ...]:
    if type(value) is not list or not value:
        raise DecisionEncodingError(f"{name} must be a nonempty JSON integer array")
    if any(
        isinstance(item, bool) or not isinstance(item, int) or (nonnegative and item < 0) for item in value
    ):
        qualifier = "non-negative " if nonnegative else ""
        raise DecisionEncodingError(f"{name} must contain only {qualifier}integers")
    return tuple(cast(list[int], value))


def _decode_exact(
    tokenizer: ExactDecodeTokenizerProtocol,
    token_ids: tuple[int, ...],
) -> str:
    try:
        value = tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise DecisionEncodingError("tokenizer could not decode decision token ids") from exc
    if type(value) is not str:
        raise DecisionEncodingError("tokenizer decode must return exact text")
    return value


@dataclass(frozen=True, slots=True)
class DecisionTokenExample:
    """One canonical history prompt and one fragment-token action target."""

    tokenizer_binding_digest: str
    maximum_sequence_tokens: int
    dialogue_digest: str
    prompt_text_sha256: str
    prompt_token_digest: str
    prompt_token_ids: tuple[int, ...]
    action_trace: ActionTokenTrace
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    attention_mask: tuple[int, ...]
    action_start: int
    schema_version: int = DECISION_ENCODING_SCHEMA_VERSION
    contract_id: str = DECISION_ENCODING_CONTRACT_ID

    def __post_init__(self) -> None:
        if self.schema_version != DECISION_ENCODING_SCHEMA_VERSION:
            raise DecisionEncodingError("unexpected decision-encoding schema version")
        if self.contract_id != DECISION_ENCODING_CONTRACT_ID:
            raise DecisionEncodingError("unexpected decision-encoding contract id")
        for name in (
            "tokenizer_binding_digest",
            "dialogue_digest",
            "prompt_text_sha256",
            "prompt_token_digest",
        ):
            if not _is_sha256(getattr(self, name)):
                raise DecisionEncodingError(f"{name} must be a lowercase SHA-256 digest")
        if (
            isinstance(self.maximum_sequence_tokens, bool)
            or not isinstance(self.maximum_sequence_tokens, int)
            or self.maximum_sequence_tokens < 1
        ):
            raise DecisionEncodingError("maximum_sequence_tokens must be positive")
        if type(self.action_trace) is not ActionTokenTrace:
            raise DecisionEncodingError("action_trace must be an ActionTokenTrace")
        if self.action_trace.tokenizer_identifier != self.tokenizer_binding_digest:
            raise DecisionEncodingError("action trace is bound to a different tokenizer manifest")

        prompt_ids = tuple(self.prompt_token_ids)
        input_ids = tuple(self.input_ids)
        labels = tuple(self.labels)
        attention = tuple(self.attention_mask)
        object.__setattr__(self, "prompt_token_ids", prompt_ids)
        object.__setattr__(self, "input_ids", input_ids)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "attention_mask", attention)
        _require_token_ids(prompt_ids, name="prompt_token_ids")
        _require_token_ids(input_ids, name="input_ids")
        if _prompt_token_digest(prompt_ids) != self.prompt_token_digest:
            raise DecisionEncodingError("prompt token digest is inconsistent")
        if (
            isinstance(self.action_start, bool)
            or not isinstance(self.action_start, int)
            or self.action_start != len(prompt_ids)
        ):
            raise DecisionEncodingError("action_start must equal the prompt token count")
        expected_input = prompt_ids + self.action_trace.action_token_ids
        if input_ids != expected_input:
            raise DecisionEncodingError(
                "decision input must be prompt tokens followed by the exact action trace"
            )
        if len(input_ids) > self.maximum_sequence_tokens:
            raise DecisionOverlengthError(
                "decision example exceeds maximum_sequence_tokens; truncation is forbidden"
            )
        expected_labels = (IGNORE_INDEX,) * len(prompt_ids) + self.action_trace.action_token_ids
        if labels != expected_labels:
            raise DecisionEncodingError("decision labels must mask exactly the prompt tokens")
        if attention != (1,) * len(input_ids):
            raise DecisionEncodingError("an unpadded decision must have an all-one attention mask")

    @property
    def action_token_count(self) -> int:
        return len(self.action_trace.action_token_ids)

    def as_obj(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "tokenizer_binding_digest": self.tokenizer_binding_digest,
            "maximum_sequence_tokens": self.maximum_sequence_tokens,
            "dialogue_digest": self.dialogue_digest,
            "prompt_text_sha256": self.prompt_text_sha256,
            "prompt_token_digest": self.prompt_token_digest,
            "prompt_token_ids": list(self.prompt_token_ids),
            "action_start": self.action_start,
            "action_trace_digest": self.action_trace.digest,
            "action_trace": {
                **self.action_trace.as_obj(),
                "digest": self.action_trace.digest,
            },
            "input_ids": list(self.input_ids),
            "labels": list(self.labels),
            "attention_mask": list(self.attention_mask),
        }

    def to_json(self) -> str:
        """Return the unique compact JSON representation including its digest."""

        return dump_json({**self.as_obj(), "digest": self.digest})

    @classmethod
    def from_obj(cls, value: object) -> DecisionTokenExample:
        """Parse a strict JSON-shaped example and verify every supplied digest."""

        fields = (
            "schema_version",
            "contract_id",
            "tokenizer_binding_digest",
            "maximum_sequence_tokens",
            "dialogue_digest",
            "prompt_text_sha256",
            "prompt_token_digest",
            "prompt_token_ids",
            "action_start",
            "action_trace_digest",
            "action_trace",
            "input_ids",
            "labels",
            "attention_mask",
            "digest",
        )
        obj = _require_exact_object(value, fields, name="decision_token_example")
        try:
            trace = ActionTokenTrace.from_obj(obj["action_trace"])
        except (FragmentActionTokenizationError, TypeError, ValueError) as exc:
            raise DecisionEncodingError("decision action trace is invalid") from exc
        supplied_trace_digest = obj["action_trace_digest"]
        if not _is_sha256(supplied_trace_digest) or not hmac.compare_digest(
            cast(str, supplied_trace_digest), trace.digest
        ):
            raise DecisionEncodingError("decision action-trace digest check failed")
        result = cls(
            schema_version=cast(int, obj["schema_version"]),
            contract_id=cast(str, obj["contract_id"]),
            tokenizer_binding_digest=cast(str, obj["tokenizer_binding_digest"]),
            maximum_sequence_tokens=cast(int, obj["maximum_sequence_tokens"]),
            dialogue_digest=cast(str, obj["dialogue_digest"]),
            prompt_text_sha256=cast(str, obj["prompt_text_sha256"]),
            prompt_token_digest=cast(str, obj["prompt_token_digest"]),
            prompt_token_ids=_require_integer_tuple(
                obj["prompt_token_ids"],
                name="prompt_token_ids",
                nonnegative=True,
            ),
            action_start=cast(int, obj["action_start"]),
            action_trace=trace,
            input_ids=_require_integer_tuple(obj["input_ids"], name="input_ids", nonnegative=True),
            labels=_require_integer_tuple(obj["labels"], name="labels", nonnegative=False),
            attention_mask=_require_integer_tuple(
                obj["attention_mask"],
                name="attention_mask",
                nonnegative=True,
            ),
        )
        supplied_digest = obj["digest"]
        if not _is_sha256(supplied_digest) or not hmac.compare_digest(
            cast(str, supplied_digest), result.digest
        ):
            raise DecisionEncodingError("decision-example digest check failed")
        return result

    @classmethod
    def from_json(cls, text: str) -> DecisionTokenExample:
        """Parse only the unique JSON bytes emitted by :meth:`to_json`."""

        try:
            value = load_json(text)
        except CanonicalJSONError as exc:
            raise DecisionEncodingError("decision example is not strict JSON") from exc
        result = cls.from_obj(value)
        if not hmac.compare_digest(text, result.to_json()):
            raise DecisionEncodingError(
                "decision example JSON is not in the unique canonical byte representation"
            )
        return result

    @property
    def digest(self) -> str:
        return json_digest(self.as_obj(), domain=_EXAMPLE_DIGEST_DOMAIN)


@dataclass(frozen=True, slots=True, init=False)
class VerifiedDecisionTokenExample:
    """Nominal evidence that source contracts regenerated an example exactly.

    The constructor is intentionally unavailable.  Production SFT/RL bridges
    should accept this wrapper rather than a merely well-shaped
    :class:`DecisionTokenExample`.
    """

    example: DecisionTokenExample
    verification_digest: str

    @classmethod
    def _from_verified(
        cls,
        example: DecisionTokenExample,
        *,
        compiler_manifest_digest: str,
    ) -> VerifiedDecisionTokenExample:
        result = object.__new__(cls)
        object.__setattr__(result, "example", example)
        object.__setattr__(
            result,
            "verification_digest",
            json_digest(
                {
                    "example_digest": example.digest,
                    "tokenizer_binding_digest": example.tokenizer_binding_digest,
                    "compiler_manifest_digest": compiler_manifest_digest,
                    "dialogue_digest": example.dialogue_digest,
                },
                domain=_VERIFIED_EXAMPLE_DOMAIN,
            ),
        )
        return result

    @property
    def digest(self) -> str:
        return self.example.digest


def encode_decision_example(
    tokenizer: ExactDecodeTokenizerProtocol,
    compiler: FragmentActionTokenCompiler,
    dialogue: Dialogue,
    action: Action,
    *,
    tokenizer_binding_digest: str,
    maximum_sequence_tokens: int,
) -> DecisionTokenExample:
    """Encode one assistant decision under the shared fragment-token contract."""

    if not isinstance(tokenizer, ExactDecodeTokenizerProtocol):
        raise TypeError("tokenizer does not implement ExactDecodeTokenizerProtocol")
    if type(compiler) is not FragmentActionTokenCompiler:
        raise TypeError("compiler must be a FragmentActionTokenCompiler")
    if not _is_sha256(tokenizer_binding_digest):
        raise ValueError("tokenizer_binding_digest must be a lowercase SHA-256")
    if (
        isinstance(maximum_sequence_tokens, bool)
        or not isinstance(maximum_sequence_tokens, int)
        or maximum_sequence_tokens < 1
    ):
        raise ValueError("maximum_sequence_tokens must be positive")

    prompt_text, prompt_ids = render_generation_prefix(tokenizer, dialogue)
    trace = compiler.trace_action(action)
    if trace.tokenizer_identifier != tokenizer_binding_digest:
        raise DecisionEncodingError(
            "compiler tokenizer identifier differs from the registered tokenizer binding"
        )
    input_ids = prompt_ids + trace.action_token_ids
    if len(input_ids) > maximum_sequence_tokens:
        raise DecisionOverlengthError(
            f"decision has {len(input_ids)} tokens, exceeding maximum "
            f"{maximum_sequence_tokens}; truncation is forbidden"
        )
    if _decode_exact(tokenizer, prompt_ids) != prompt_text:
        raise DecisionEncodingError("prompt token ids do not exactly decode to the chat template")
    if _decode_exact(tokenizer, input_ids) != prompt_text + trace.raw_action:
        raise DecisionEncodingError(
            "prompt plus fragment-token action does not exactly decode to rendered text"
        )
    result = DecisionTokenExample(
        tokenizer_binding_digest=tokenizer_binding_digest,
        maximum_sequence_tokens=maximum_sequence_tokens,
        dialogue_digest=json_digest(
            dialogue_as_obj(dialogue),
            domain="goalzendo-interactive-decision-dialogue-v1",
        ),
        prompt_text_sha256=_sha256_text(prompt_text),
        prompt_token_digest=_prompt_token_digest(prompt_ids),
        prompt_token_ids=prompt_ids,
        action_trace=trace,
        input_ids=input_ids,
        labels=(IGNORE_INDEX,) * len(prompt_ids) + trace.action_token_ids,
        attention_mask=(1,) * len(input_ids),
        action_start=len(prompt_ids),
    )
    return result


def verify_decision_example(
    example: DecisionTokenExample,
    tokenizer: ExactDecodeTokenizerProtocol,
    compiler: FragmentActionTokenCompiler,
    dialogue: Dialogue,
) -> VerifiedDecisionTokenExample:
    """Recompile every derived token, mask, prompt, and digest byte-for-byte."""

    if type(example) is not DecisionTokenExample:
        raise TypeError("example must be a DecisionTokenExample")
    try:
        compiler.verify_trace(example.action_trace)
    except FragmentActionTokenizationError as exc:
        raise DecisionEncodingError(
            "decision action trace differs from exact tokenizer/compiler regeneration"
        ) from exc
    try:
        action = parse_action(example.action_trace.raw_action)
    except ValueError as exc:
        raise DecisionEncodingError("decision action no longer parses canonically") from exc
    rebuilt = encode_decision_example(
        tokenizer,
        compiler,
        dialogue,
        action,
        tokenizer_binding_digest=example.tokenizer_binding_digest,
        maximum_sequence_tokens=example.maximum_sequence_tokens,
    )
    if rebuilt.as_obj() != example.as_obj() or rebuilt.digest != example.digest:
        raise DecisionEncodingError("decision example differs from exact tokenizer/compiler regeneration")
    return VerifiedDecisionTokenExample._from_verified(
        example,
        compiler_manifest_digest=compiler.manifest.digest,
    )


@dataclass(frozen=True, slots=True)
class MaskedActionStatistics:
    """Differentiable masked token statistics for one verified action."""

    token_log_probabilities: torch.Tensor
    token_entropies: torch.Tensor
    sequence_log_probability: torch.Tensor
    mean_token_entropy: torch.Tensor
    action_token_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.token_log_probabilities, torch.Tensor) or not isinstance(
            self.token_entropies, torch.Tensor
        ):
            raise DecisionEncodingError("masked token statistics must be tensors")
        if not isinstance(self.sequence_log_probability, torch.Tensor) or not isinstance(
            self.mean_token_entropy, torch.Tensor
        ):
            raise DecisionEncodingError("masked statistic reductions must be tensors")
        if (
            isinstance(self.action_token_count, bool)
            or not isinstance(self.action_token_count, int)
            or self.action_token_count < 1
        ):
            raise DecisionEncodingError("action_token_count must be a positive integer")
        if self.token_log_probabilities.ndim != 1 or self.token_entropies.ndim != 1:
            raise DecisionEncodingError("masked token statistics must be one-dimensional")
        if self.token_log_probabilities.shape != self.token_entropies.shape:
            raise DecisionEncodingError("masked log probabilities and entropies must align")
        if len(self.token_log_probabilities) != self.action_token_count:
            raise DecisionEncodingError("masked statistic count differs from action length")
        if self.sequence_log_probability.ndim != 0 or self.mean_token_entropy.ndim != 0:
            raise DecisionEncodingError("masked statistic reductions must be scalar tensors")


def masked_action_statistics(
    logits: torch.Tensor,
    verified_example: VerifiedDecisionTokenExample,
    *,
    temperature: float,
) -> MaskedActionStatistics:
    """Recompute differentiable log probabilities under exact sparse masks.

    ``logits`` must contain the ordinary causal-model output for
    ``example.input_ids`` with shape ``[sequence, vocabulary]``.  The logit at
    the final prompt position predicts the first fragment token.
    """

    if type(verified_example) is not VerifiedDecisionTokenExample:
        raise TypeError("verified_example must be a VerifiedDecisionTokenExample")
    example = verified_example.example
    if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
        raise TypeError("logits must be a [sequence, vocabulary] Tensor")
    if not logits.is_floating_point():
        raise DecisionEncodingError("logits must have a floating dtype")
    if logits.shape[0] != len(example.input_ids):
        raise DecisionEncodingError("logit sequence length differs from decision input")
    if example.action_start < 1:
        raise DecisionEncodingError("the first action token requires a preceding prompt token")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature))
        or float(temperature) <= 0
    ):
        raise ValueError("temperature must be finite and positive")
    if not bool(torch.isfinite(logits).all().item()):
        raise DecisionEncodingError("model logits contain a non-finite value")

    vocabulary = logits.shape[1]
    log_probabilities: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    scale = float(temperature)
    for token_index, step in enumerate(example.action_trace.steps):
        if step.action_token_index != token_index:
            raise DecisionEncodingError("action trace token indices are not contiguous")
        if step.selected_token_id >= vocabulary or step.allowed_token_ids[-1] >= vocabulary:
            raise DecisionEncodingError("action trace references an out-of-vocabulary token")
        prediction_index = example.action_start + token_index - 1
        row = logits[prediction_index]
        allowed = torch.tensor(step.allowed_token_ids, dtype=torch.long, device=row.device)
        selected_position = step.allowed_token_ids.index(step.selected_token_id)
        allowed_logits = row.index_select(0, allowed) / scale
        log_distribution = allowed_logits - torch.logsumexp(allowed_logits, dim=0)
        probabilities = torch.exp(log_distribution)
        selected = log_distribution[selected_position]
        entropy = -(probabilities * log_distribution).sum()
        log_probabilities.append(selected)
        entropies.append(entropy)

    token_log_probabilities = torch.stack(log_probabilities)
    token_entropies = torch.stack(entropies)
    if not bool(torch.isfinite(token_log_probabilities).all().item()):
        raise DecisionEncodingError("masked token log probabilities are non-finite")
    if not bool(torch.isfinite(token_entropies).all().item()):
        raise DecisionEncodingError("masked token entropies are non-finite")
    return MaskedActionStatistics(
        token_log_probabilities=token_log_probabilities,
        token_entropies=token_entropies,
        sequence_log_probability=token_log_probabilities.sum(),
        mean_token_entropy=token_entropies.mean(),
        action_token_count=len(log_probabilities),
    )


def verified_masked_action_statistics(
    logits: torch.Tensor,
    example: DecisionTokenExample,
    tokenizer: ExactDecodeTokenizerProtocol,
    compiler: FragmentActionTokenCompiler,
    dialogue: Dialogue,
    *,
    temperature: float,
) -> MaskedActionStatistics:
    """Verify the example against source contracts before differentiable replay."""

    verified = verify_decision_example(example, tokenizer, compiler, dialogue)
    return masked_action_statistics(logits, verified, temperature=temperature)


def _float_to_hex(value: float, *, name: str, nonnegative: bool) -> str:
    selected = float(value)
    if not math.isfinite(selected):
        raise DecisionEncodingError(f"{name} must be finite")
    if nonnegative and selected < 0:
        raise DecisionEncodingError(f"{name} must be non-negative")
    if not nonnegative and selected > 1e-7:
        raise DecisionEncodingError(f"{name} must be <= 0")
    return selected.hex()


def _hex_to_float(value: str, *, name: str, nonnegative: bool) -> float:
    if type(value) is not str or not value or not value.isascii():
        raise DecisionEncodingError(f"{name} must be an ASCII hexadecimal float")
    try:
        selected = float.fromhex(value)
    except ValueError as exc:
        raise DecisionEncodingError(f"{name} is not a hexadecimal float") from exc
    if selected.hex() != value:
        raise DecisionEncodingError(f"{name} is not in canonical float.hex form")
    _float_to_hex(selected, name=name, nonnegative=nonnegative)
    return selected


@dataclass(frozen=True, slots=True)
class DetachedMaskedActionStatistics:
    """Canonical detached behavior-policy values, never a gradient source."""

    temperature_hex: str
    token_log_probability_hex: tuple[str, ...]
    token_entropy_hex: tuple[str, ...]
    example_digest: str
    schema_version: int = DETACHED_MASKED_STATISTICS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DETACHED_MASKED_STATISTICS_SCHEMA_VERSION:
            raise DecisionEncodingError("unexpected detached-statistics schema version")
        if not _is_sha256(self.example_digest):
            raise DecisionEncodingError("example_digest must be a SHA-256")
        temperature = _hex_to_float(
            self.temperature_hex,
            name="temperature_hex",
            nonnegative=True,
        )
        if temperature <= 0:
            raise DecisionEncodingError("detached temperature must be positive")
        log_probs = tuple(self.token_log_probability_hex)
        entropies = tuple(self.token_entropy_hex)
        object.__setattr__(self, "token_log_probability_hex", log_probs)
        object.__setattr__(self, "token_entropy_hex", entropies)
        if not log_probs or len(log_probs) != len(entropies):
            raise DecisionEncodingError("detached token statistics must align and be nonempty")
        for value in log_probs:
            _hex_to_float(value, name="token_log_probability_hex", nonnegative=False)
        for value in entropies:
            _hex_to_float(value, name="token_entropy_hex", nonnegative=True)

    @classmethod
    def from_replay(
        cls,
        statistics: MaskedActionStatistics,
        verified_example: VerifiedDecisionTokenExample,
        *,
        temperature: float,
    ) -> DetachedMaskedActionStatistics:
        if type(statistics) is not MaskedActionStatistics:
            raise TypeError("statistics must be MaskedActionStatistics")
        if type(verified_example) is not VerifiedDecisionTokenExample:
            raise TypeError("verified_example must be a VerifiedDecisionTokenExample")
        example = verified_example.example
        return cls(
            temperature_hex=_float_to_hex(float(temperature), name="temperature", nonnegative=True),
            token_log_probability_hex=tuple(
                _float_to_hex(float(value), name="token_log_probability", nonnegative=False)
                for value in statistics.token_log_probabilities.detach().to(device="cpu", dtype=torch.float64)
            ),
            token_entropy_hex=tuple(
                _float_to_hex(float(value), name="token_entropy", nonnegative=True)
                for value in statistics.token_entropies.detach().to(device="cpu", dtype=torch.float64)
            ),
            example_digest=example.digest,
        )

    @property
    def token_log_probabilities(self) -> tuple[float, ...]:
        return tuple(
            _hex_to_float(value, name="token_log_probability_hex", nonnegative=False)
            for value in self.token_log_probability_hex
        )

    @property
    def token_entropies(self) -> tuple[float, ...]:
        return tuple(
            _hex_to_float(value, name="token_entropy_hex", nonnegative=True)
            for value in self.token_entropy_hex
        )

    @property
    def temperature(self) -> float:
        return _hex_to_float(
            self.temperature_hex,
            name="temperature_hex",
            nonnegative=True,
        )

    def as_obj(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "temperature_hex": self.temperature_hex,
            "token_log_probability_hex": list(self.token_log_probability_hex),
            "token_entropy_hex": list(self.token_entropy_hex),
            "example_digest": self.example_digest,
        }

    @property
    def digest(self) -> str:
        return json_digest(self.as_obj(), domain=_DETACHED_STATISTICS_DOMAIN)


def compare_detached_masked_statistics(
    detached: DetachedMaskedActionStatistics,
    replayed: MaskedActionStatistics,
    verified_example: VerifiedDecisionTokenExample,
    *,
    temperature: float,
    absolute_tolerance: float,
) -> None:
    """Fail closed unless stored behavior values match differentiable replay."""

    if type(detached) is not DetachedMaskedActionStatistics:
        raise TypeError("detached must be DetachedMaskedActionStatistics")
    if type(replayed) is not MaskedActionStatistics:
        raise TypeError("replayed must be MaskedActionStatistics")
    if type(verified_example) is not VerifiedDecisionTokenExample:
        raise TypeError("verified_example must be a VerifiedDecisionTokenExample")
    example = verified_example.example
    if detached.example_digest != example.digest:
        raise DecisionEncodingError("detached statistics belong to another decision example")
    if float(temperature).hex() != detached.temperature_hex:
        raise DecisionEncodingError("detached statistics use a different temperature")
    if (
        isinstance(absolute_tolerance, bool)
        or not isinstance(absolute_tolerance, (int, float))
        or not math.isfinite(float(absolute_tolerance))
        or float(absolute_tolerance) < 0
    ):
        raise ValueError("absolute_tolerance must be finite and non-negative")
    if replayed.action_token_count != len(detached.token_log_probability_hex):
        raise DecisionEncodingError("detached/replayed action lengths differ")

    stored_log_probs = torch.tensor(
        detached.token_log_probabilities,
        dtype=torch.float64,
    )
    stored_entropies = torch.tensor(detached.token_entropies, dtype=torch.float64)
    replayed_log_probs = replayed.token_log_probabilities.detach().to(device="cpu", dtype=torch.float64)
    replayed_entropies = replayed.token_entropies.detach().to(device="cpu", dtype=torch.float64)
    tolerance = float(absolute_tolerance)
    log_prob_error = float(torch.max(torch.abs(stored_log_probs - replayed_log_probs)).item())
    entropy_error = float(torch.max(torch.abs(stored_entropies - replayed_entropies)).item())
    if log_prob_error > tolerance or entropy_error > tolerance:
        raise DecisionEncodingError(
            "detached masked statistics differ from differentiable replay: "
            f"max_log_prob_error={log_prob_error:.9g}, "
            f"max_entropy_error={entropy_error:.9g}, tolerance={tolerance:.9g}"
        )


def decision_encoding_manifest() -> dict[str, object]:
    """Return the fixed semantics of the local, nonauthorizing bridge."""

    return {
        "schema_version": DECISION_ENCODING_SCHEMA_VERSION,
        "contract_id": DECISION_ENCODING_CONTRACT_ID,
        "example_unit": "one canonical history prompt plus one fragment-token action",
        "sft_labels": "all prompt tokens ignored; every fragment action token supervised",
        "outcome_rl": (
            "selected log probabilities and entropies recomputed with torch.logsumexp "
            "over each exact sparse trie-child mask"
        ),
        "causal_alignment": (
            "the final prompt logit predicts action token zero; each action token predicts the next"
        ),
        "detached_values": "canonical float.hex audit values; never used as a gradient source",
        "verification": (
            "strict canonical JSON plus prompt, action trace, masks, and derived arrays "
            "regenerate exactly; gradient replay requires a nominal verified wrapper"
        ),
        "truncation": "forbidden",
        "live_model_authorization": False,
        "weight_update_authorization": False,
    }
