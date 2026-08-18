"""Authenticated, stateless token sampling for the G03 action language.

This module is an additive, nonauthorizing bridge between a frozen fragment
compiler and a model-like logits provider.  It samples the public finite-state
grammar one token at a time, using independently addressed policy draws and
the exact trie-child masks implied by each compiled fragment.  A sampled
record is not accepted for training until it has been regenerated from the
same policy state and compared with a differentiable full-forward replay.

No model is loaded here and no weight update is authorized.
"""

from __future__ import annotations

import hmac
import math
import weakref
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal, Protocol, cast, runtime_checkable

import torch

from ._json import CanonicalJSONError, dump_json, json_digest, load_json
from .action_language import (
    AnswerActionState,
    GrammarOption,
    GrammarSegment,
    InquiryActionState,
    action_from_state,
)
from .action_tokenization_v2 import (
    CompiledFragmentTrie,
    ExactDecodeTokenizerProtocol,
    FragmentActionTokenCompiler,
    FragmentActionTokenizationError,
)
from .actions import AnswerAction, ReadyAction, TestAction, parse_action
from .decision_encoding_v2 import (
    DecisionEncodingError,
    DecisionOverlengthError,
    DecisionTokenExample,
    DetachedMaskedActionStatistics,
    MaskedActionStatistics,
    VerifiedDecisionTokenExample,
    compare_detached_masked_statistics,
    encode_decision_example,
    verify_decision_example,
)
from .dialogue import Dialogue, DialogueMessage
from .policy_randomness_v2 import (
    ROLLOUTS_PER_EPISODE,
    PolicyRandomnessError,
    PolicyTokenDraw,
    PolicyTurnSeed,
    policy_token_draw_from_obj,
    policy_turn_seed_from_obj,
    verify_policy_token_draw,
)
from .trajectory_encoding import render_generation_prefix

AUTHENTICATED_SAMPLER_SCHEMA_VERSION = 1
AUTHENTICATED_SAMPLER_CONTRACT_ID = "goalzendo-authenticated-token-sampler-v1"

_SAMPLE_DIGEST_DOMAIN = "goalzendo-interactive-authenticated-token-sample-v1"
_VERIFICATION_DIGEST_DOMAIN = "goalzendo-interactive-authenticated-token-verification-v1"
_ROLLOUT_GROUP_DIGEST_DOMAIN = "goalzendo-interactive-unchanged-policy-rollout-group-v1"

SamplingMode = Literal["inquiry", "answer"]
_GrammarState = InquiryActionState | AnswerActionState


class AuthenticatedSamplingError(ValueError):
    """Raised when sampling provenance or exact replay fails closed."""


class AuthenticatedSamplingOverlengthError(AuthenticatedSamplingError):
    """A narrow, replayable controller outcome; every other error is fatal."""


@runtime_checkable
class NextTokenLogitsProvider(Protocol):
    """Minimal behavior-policy surface used during constrained sampling."""

    @property
    def policy_state_digest(self) -> str: ...

    def next_token_logits(self, input_ids: tuple[int, ...]) -> torch.Tensor: ...


@runtime_checkable
class FullForwardLogitsProvider(NextTokenLogitsProvider, Protocol):
    """Same-policy provider capable of differentiable full-sequence replay."""

    def full_forward_logits(self, input_ids: tuple[int, ...]) -> torch.Tensor: ...


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: object, *, name: str) -> str:
    if not _is_sha256(value):
        raise AuthenticatedSamplingError(f"{name} must be a lowercase SHA-256")
    return cast(str, value)


def _require_exact_object(
    value: object,
    fields: tuple[str, ...],
    *,
    name: str,
) -> dict[str, object]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise AuthenticatedSamplingError(f"{name} must be a JSON object")
    result = cast(dict[str, object], value)
    if set(result) != set(fields) or len(result) != len(fields):
        raise AuthenticatedSamplingError(f"{name} has noncanonical fields")
    return result


def _require_list(value: object, *, name: str) -> list[object]:
    if type(value) is not list:
        raise AuthenticatedSamplingError(f"{name} must be a JSON array")
    return cast(list[object], value)


def _require_integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AuthenticatedSamplingError(f"{name} must be an integer greater than or equal to {minimum}")
    return value


def _require_optional_integer(value: object, *, name: str) -> int | None:
    if value is None:
        return None
    return _require_integer(value, name=name, minimum=1)


def _require_text(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise AuthenticatedSamplingError(f"{name} must be nonempty text")
    return value


def _require_temperature(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise AuthenticatedSamplingError("temperature must be finite and positive")
    return float(value)


def _canonical_temperature_hex(value: object) -> str:
    return _require_temperature(value).hex()


def _validate_dialogue_mode(dialogue: Dialogue, mode: SamplingMode) -> None:
    if (
        type(dialogue) is not tuple
        or not dialogue
        or any(type(message) is not DialogueMessage for message in dialogue)
    ):
        raise AuthenticatedSamplingError("sampling requires a nonempty typed model-facing dialogue")
    final_message = dialogue[-1]
    if final_message.role != "user":
        raise AuthenticatedSamplingError("sampling dialogue must end with a user generation request")
    expected_mode: SamplingMode
    if final_message.phase == "terminal":
        expected_mode = "answer"
    elif final_message.phase in {"opening", "feedback"}:
        expected_mode = "inquiry"
    else:
        raise AuthenticatedSamplingError("dialogue does not end at an actionable game phase")
    if mode != expected_mode:
        raise AuthenticatedSamplingError("sampling mode differs from the final public dialogue phase")


def _provider_policy_state(provider: object) -> str:
    try:
        value = provider.policy_state_digest  # type: ignore[attr-defined]
    except (AttributeError, TypeError, ValueError) as exc:
        raise AuthenticatedSamplingError("logits provider does not expose a policy_state_digest") from exc
    return _require_sha256(value, name="provider.policy_state_digest")


def _validate_next_logits(
    logits: object,
    *,
    vocabulary_size: int,
) -> torch.Tensor:
    if not isinstance(logits, torch.Tensor) or logits.ndim != 1:
        raise AuthenticatedSamplingError("next_token_logits must return a one-dimensional Tensor")
    if not logits.is_floating_point():
        raise AuthenticatedSamplingError("next-token logits must have a floating dtype")
    if logits.shape[0] < vocabulary_size:
        raise AuthenticatedSamplingError(
            "next-token logits do not cover the tokenizer vocabulary"
        )
    try:
        detached = logits.detach().to(device="cpu", dtype=torch.float64)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise AuthenticatedSamplingError("next-token logits could not be materialized as float64") from exc
    if not bool(torch.isfinite(detached).all().item()):
        raise AuthenticatedSamplingError("next-token logits contain a non-finite value")
    return detached


def _next_logits(
    provider: NextTokenLogitsProvider,
    input_ids: tuple[int, ...],
    *,
    expected_policy_state_digest: str,
    vocabulary_size: int,
) -> torch.Tensor:
    if _provider_policy_state(provider) != expected_policy_state_digest:
        raise AuthenticatedSamplingError("logits provider policy state differs from the registered turn seed")
    try:
        raw_logits = provider.next_token_logits(input_ids)
    except Exception as exc:
        raise AuthenticatedSamplingError("next-token logits provider failed") from exc
    if _provider_policy_state(provider) != expected_policy_state_digest:
        raise AuthenticatedSamplingError("policy state changed during token sampling")
    return _validate_next_logits(raw_logits, vocabulary_size=vocabulary_size)


def _validate_allowed_token_ids(
    allowed_token_ids: tuple[int, ...],
    *,
    vocabulary_size: int,
) -> None:
    if not allowed_token_ids or allowed_token_ids != tuple(sorted(set(allowed_token_ids))):
        raise AuthenticatedSamplingError("legal token mask must be nonempty, unique, and increasing")
    if any(
        isinstance(token_id, bool) or not isinstance(token_id, int) or not 0 <= token_id < vocabulary_size
        for token_id in allowed_token_ids
    ):
        raise AuthenticatedSamplingError("legal token mask contains an out-of-vocabulary token")


def _select_from_sorted_mask(
    logits: torch.Tensor,
    allowed_token_ids: tuple[int, ...],
    draw: PolicyTokenDraw,
    *,
    temperature: float,
) -> tuple[int, float, float]:
    """Use an exact rational draw against float64 softmax weights.

    ``torch.exp`` defines each finite IEEE-754 weight.  Converting those
    binary64 values with :meth:`Fraction.from_float` makes accumulation and
    the inverse-CDF boundary comparison exact.  Token IDs define bin order.
    """

    vocabulary_size = logits.shape[0]
    _validate_allowed_token_ids(
        allowed_token_ids,
        vocabulary_size=vocabulary_size,
    )
    selected_temperature = _require_temperature(temperature)
    if len(allowed_token_ids) == 1:
        return allowed_token_ids[0], 0.0, 0.0

    allowed = torch.tensor(allowed_token_ids, dtype=torch.long)
    scaled = logits.index_select(0, allowed) / selected_temperature
    shifted = scaled - torch.max(scaled)
    weight_tensor = torch.exp(shifted)
    weights = tuple(Fraction.from_float(float(value)) for value in weight_tensor)
    total = sum(weights, start=Fraction(0, 1))
    if total <= 0:
        raise AuthenticatedSamplingError("masked softmax has no positive weight")
    threshold = draw.open_unit_interval * total
    cumulative = Fraction(0, 1)
    selected_position = len(weights) - 1
    for position, weight in enumerate(weights):
        cumulative += weight
        if threshold <= cumulative:
            selected_position = position
            break

    log_distribution = scaled - torch.logsumexp(scaled, dim=0)
    probabilities = torch.exp(log_distribution)
    entropy = -(probabilities * log_distribution).sum()
    log_probability_value = float(log_distribution[selected_position])
    entropy_value = float(entropy)
    if not math.isfinite(log_probability_value) or not math.isfinite(entropy_value):
        raise AuthenticatedSamplingError("masked token statistics are non-finite")
    if log_probability_value > 1e-12 or entropy_value < -1e-12:
        raise AuthenticatedSamplingError("masked token statistics violate probability bounds")
    return (
        allowed_token_ids[selected_position],
        log_probability_value,
        max(entropy_value, 0.0),
    )


@dataclass(frozen=True, slots=True)
class _SampledFragment:
    selected_key: str
    token_ids: tuple[int, ...]
    allowed_token_ids: tuple[tuple[int, ...], ...]
    draws: tuple[PolicyTokenDraw, ...]
    token_log_probabilities: tuple[float, ...]
    token_entropies: tuple[float, ...]


def _sample_compiled_fragment(
    provider: NextTokenLogitsProvider,
    compiled: CompiledFragmentTrie,
    prompt_token_ids: tuple[int, ...],
    preceding_action_ids: tuple[int, ...],
    turn_seed: PolicyTurnSeed,
    *,
    temperature: float,
    maximum_action_tokens: int,
) -> _SampledFragment:
    """Traverse a compiled prefix-free option set without private trie access."""

    if type(compiled) is not CompiledFragmentTrie:
        raise TypeError("compiled must be a CompiledFragmentTrie")
    candidates = list(compiled.option_token_ids)
    if not candidates:
        raise AuthenticatedSamplingError("compiled fragment contains no options")

    selected_ids: list[int] = []
    masks: list[tuple[int, ...]] = []
    draws: list[PolicyTokenDraw] = []
    log_probabilities: list[float] = []
    entropies: list[float] = []
    local_index = 0
    while True:
        if any(len(token_ids) <= local_index for _, _, token_ids in candidates):
            raise AuthenticatedSamplingError("compiled fragment is not a prefix-free token language")
        allowed = tuple(sorted({token_ids[local_index] for _, _, token_ids in candidates}))
        action_token_index = len(preceding_action_ids) + len(selected_ids)
        if action_token_index >= maximum_action_tokens:
            raise AuthenticatedSamplingOverlengthError(
                "sampled action exceeds maximum_action_tokens; truncation is forbidden"
            )
        draw = turn_seed.draw(action_token_index)
        row = _next_logits(
            provider,
            prompt_token_ids + preceding_action_ids + tuple(selected_ids),
            expected_policy_state_digest=turn_seed.policy_state_digest,
            vocabulary_size=compiled.vocabulary_size,
        )
        selected, log_probability, entropy = _select_from_sorted_mask(
            row,
            allowed,
            draw,
            temperature=temperature,
        )
        selected_ids.append(selected)
        masks.append(allowed)
        draws.append(draw)
        log_probabilities.append(log_probability)
        entropies.append(entropy)
        candidates = [candidate for candidate in candidates if candidate[2][local_index] == selected]
        local_index += 1
        completed = [candidate for candidate in candidates if len(candidate[2]) == local_index]
        if completed:
            if len(completed) != 1 or len(candidates) != 1:
                raise AuthenticatedSamplingError("compiled fragment has a terminal token-prefix collision")
            option_key, _, exact_ids = completed[0]
            if exact_ids != tuple(selected_ids):
                raise AuthenticatedSamplingError("sampled fragment differs from its compiled option sequence")
            return _SampledFragment(
                selected_key=option_key,
                token_ids=tuple(selected_ids),
                allowed_token_ids=tuple(masks),
                draws=tuple(draws),
                token_log_probabilities=tuple(log_probabilities),
                token_entropies=tuple(entropies),
            )


def _detached_statistics_from_sampling(
    verified_example: VerifiedDecisionTokenExample,
    *,
    temperature: float,
    log_probabilities: tuple[float, ...],
    entropies: tuple[float, ...],
) -> DetachedMaskedActionStatistics:
    if not log_probabilities or len(log_probabilities) != len(entropies):
        raise AuthenticatedSamplingError("sampled token statistics must align")
    try:
        statistics = MaskedActionStatistics(
            token_log_probabilities=torch.tensor(log_probabilities, dtype=torch.float64),
            token_entropies=torch.tensor(entropies, dtype=torch.float64),
            sequence_log_probability=torch.tensor(log_probabilities, dtype=torch.float64).sum(),
            mean_token_entropy=torch.tensor(entropies, dtype=torch.float64).mean(),
            action_token_count=len(log_probabilities),
        )
        return DetachedMaskedActionStatistics.from_replay(
            statistics,
            verified_example,
            temperature=temperature,
        )
    except (DecisionEncodingError, RuntimeError, TypeError, ValueError) as exc:
        raise AuthenticatedSamplingError(
            "sampled masked statistics could not be recorded canonically"
        ) from exc


def _detached_statistics_from_obj(value: object) -> DetachedMaskedActionStatistics:
    obj = _require_exact_object(
        value,
        (
            "schema_version",
            "temperature_hex",
            "token_log_probability_hex",
            "token_entropy_hex",
            "example_digest",
            "digest",
        ),
        name="detached_statistics",
    )
    raw_log_probabilities = _require_list(
        obj["token_log_probability_hex"],
        name="detached_statistics.token_log_probability_hex",
    )
    raw_entropies = _require_list(
        obj["token_entropy_hex"],
        name="detached_statistics.token_entropy_hex",
    )
    try:
        result = DetachedMaskedActionStatistics(
            schema_version=_require_integer(
                obj["schema_version"],
                name="detached_statistics.schema_version",
                minimum=1,
            ),
            temperature_hex=_require_text(obj["temperature_hex"], name="detached_statistics.temperature_hex"),
            token_log_probability_hex=tuple(
                _require_text(item, name="detached token log probability") for item in raw_log_probabilities
            ),
            token_entropy_hex=tuple(
                _require_text(item, name="detached token entropy") for item in raw_entropies
            ),
            example_digest=_require_sha256(obj["example_digest"], name="detached_statistics.example_digest"),
        )
    except (DecisionEncodingError, TypeError, ValueError) as exc:
        raise AuthenticatedSamplingError("detached statistics are invalid") from exc
    supplied_digest = _require_sha256(obj["digest"], name="detached_statistics.digest")
    if not hmac.compare_digest(supplied_digest, result.digest):
        raise AuthenticatedSamplingError("detached-statistics digest check failed")
    return result


def _differentiable_float64_masked_statistics(
    logits: torch.Tensor,
    verified_example: VerifiedDecisionTokenExample,
    *,
    temperature: float,
) -> MaskedActionStatistics:
    """Gather exact legal slices before differentiably promoting to float64.

    Casting a full ``[sequence, vocabulary]`` model output to float64 would
    multiply peak memory for a tensor whose overwhelming majority is masked
    away. This is algebraically the same sparse masked replay, but promotes
    only each action position's legal trie children. Gradients still flow to
    the original full-forward logits.
    """

    if type(verified_example) is not VerifiedDecisionTokenExample:
        raise TypeError("verified_example must be a VerifiedDecisionTokenExample")
    if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
        raise AuthenticatedSamplingError("full-forward logits must be a [sequence, vocabulary] Tensor")
    if not logits.is_floating_point():
        raise AuthenticatedSamplingError("full-forward logits must have a floating dtype")
    example = verified_example.example
    if logits.shape[0] != len(example.input_ids):
        raise AuthenticatedSamplingError("full-forward logit sequence differs from the decision input")
    if example.action_start < 1:
        raise AuthenticatedSamplingError("the first action token requires a preceding prompt token")
    selected_temperature = _require_temperature(temperature)
    if not bool(torch.isfinite(logits).all().item()):
        raise AuthenticatedSamplingError("full-forward logits contain a non-finite value")

    vocabulary_size = logits.shape[1]
    log_probabilities: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    for token_index, step in enumerate(example.action_trace.steps):
        if step.action_token_index != token_index:
            raise AuthenticatedSamplingError("action trace token indices are not contiguous")
        if step.selected_token_id >= vocabulary_size or step.allowed_token_ids[-1] >= vocabulary_size:
            raise AuthenticatedSamplingError("action trace references an out-of-vocabulary token")
        prediction_index = example.action_start + token_index - 1
        row = logits[prediction_index]
        allowed = torch.tensor(
            step.allowed_token_ids,
            dtype=torch.long,
            device=row.device,
        )
        selected_position = step.allowed_token_ids.index(step.selected_token_id)
        allowed_logits = row.index_select(0, allowed).to(dtype=torch.float64)
        allowed_logits = allowed_logits / selected_temperature
        log_distribution = allowed_logits - torch.logsumexp(allowed_logits, dim=0)
        probabilities = torch.exp(log_distribution)
        log_probabilities.append(log_distribution[selected_position])
        entropies.append(-(probabilities * log_distribution).sum())

    token_log_probabilities = torch.stack(log_probabilities)
    token_entropies = torch.stack(entropies)
    if not bool(torch.isfinite(token_log_probabilities).all().item()):
        raise AuthenticatedSamplingError("full-forward masked token log probabilities are non-finite")
    if not bool(torch.isfinite(token_entropies).all().item()):
        raise AuthenticatedSamplingError("full-forward masked token entropies are non-finite")
    return MaskedActionStatistics(
        token_log_probabilities=token_log_probabilities,
        token_entropies=token_entropies,
        sequence_log_probability=token_log_probabilities.sum(),
        mean_token_entropy=token_entropies.mean(),
        action_token_count=len(log_probabilities),
    )


@dataclass(frozen=True, slots=True)
class AuthenticatedActionSample:
    """Canonical audit record for one sampled model decision.

    Construction verifies all self-contained invariants.  Acceptance still
    requires :func:`replay_authenticated_sample`, which has access to the
    tokenizer, compiler, dialogue, and same-policy logits provider.
    """

    mode: SamplingMode
    terminal_count: int | None
    policy_state_digest: str
    tokenizer_binding_digest: str
    compiler_manifest_digest: str
    turn_seed: PolicyTurnSeed
    temperature_hex: str
    selected_token_ids: tuple[int, ...]
    draws: tuple[PolicyTokenDraw, ...]
    decision_verification_digest: str
    decision_example: DecisionTokenExample
    detached_statistics: DetachedMaskedActionStatistics
    schema_version: int = AUTHENTICATED_SAMPLER_SCHEMA_VERSION
    contract_id: str = AUTHENTICATED_SAMPLER_CONTRACT_ID

    def __post_init__(self) -> None:
        if self.schema_version != AUTHENTICATED_SAMPLER_SCHEMA_VERSION:
            raise AuthenticatedSamplingError("unexpected authenticated-sampler schema version")
        if self.contract_id != AUTHENTICATED_SAMPLER_CONTRACT_ID:
            raise AuthenticatedSamplingError("unexpected authenticated-sampler contract id")
        if self.mode not in {"inquiry", "answer"}:
            raise AuthenticatedSamplingError("sample mode must be inquiry or answer")
        if self.mode == "inquiry" and self.terminal_count is not None:
            raise AuthenticatedSamplingError("inquiry samples cannot have terminal_count")
        if self.mode == "answer":
            _require_integer(self.terminal_count, name="terminal_count", minimum=1)
        for name in (
            "policy_state_digest",
            "tokenizer_binding_digest",
            "compiler_manifest_digest",
            "decision_verification_digest",
        ):
            _require_sha256(getattr(self, name), name=name)
        if type(self.turn_seed) is not PolicyTurnSeed:
            raise AuthenticatedSamplingError("turn_seed must be a PolicyTurnSeed")
        if type(self.decision_example) is not DecisionTokenExample:
            raise AuthenticatedSamplingError("decision_example must be a DecisionTokenExample")
        if type(self.detached_statistics) is not DetachedMaskedActionStatistics:
            raise AuthenticatedSamplingError("detached_statistics must be DetachedMaskedActionStatistics")
        if self.turn_seed.policy_state_digest != self.policy_state_digest:
            raise AuthenticatedSamplingError("sample policy state differs from its turn seed")
        if self.temperature_hex != self.detached_statistics.temperature_hex:
            raise AuthenticatedSamplingError("sample temperature differs from detached statistics")
        if _canonical_temperature_hex(self.detached_statistics.temperature) != self.temperature_hex:
            raise AuthenticatedSamplingError("sample temperature is not canonical")

        selected = tuple(self.selected_token_ids)
        draws = tuple(self.draws)
        object.__setattr__(self, "selected_token_ids", selected)
        object.__setattr__(self, "draws", draws)
        if not selected or any(
            isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
            for token_id in selected
        ):
            raise AuthenticatedSamplingError("selected_token_ids must be nonempty non-negative integers")
        if len(draws) != len(selected) or any(type(draw) is not PolicyTokenDraw for draw in draws):
            raise AuthenticatedSamplingError("sample requires one typed policy draw per selected token")
        for token_index, draw in enumerate(draws):
            if draw.action_token_index != token_index:
                raise AuthenticatedSamplingError("sample draw indices must be contiguous")
            try:
                verify_policy_token_draw(draw, self.turn_seed)
            except PolicyRandomnessError as exc:
                raise AuthenticatedSamplingError("sample draw does not rederive from its turn seed") from exc

        example = self.decision_example
        trace = example.action_trace
        if example.tokenizer_binding_digest != self.tokenizer_binding_digest:
            raise AuthenticatedSamplingError("sample tokenizer differs from its decision example")
        if trace.compiler_manifest_digest != self.compiler_manifest_digest:
            raise AuthenticatedSamplingError("sample compiler differs from its decision trace")
        if selected != trace.action_token_ids:
            raise AuthenticatedSamplingError("selected token ids differ from the exact decision trace")
        detached = self.detached_statistics
        if detached.example_digest != example.digest:
            raise AuthenticatedSamplingError("detached statistics belong to another decision example")
        if len(detached.token_log_probability_hex) != len(selected):
            raise AuthenticatedSamplingError("detached statistics differ from the sampled action length")
        zero_hex = (0.0).hex()
        for step, log_probability_hex, entropy_hex in zip(
            trace.steps,
            detached.token_log_probability_hex,
            detached.token_entropy_hex,
            strict=True,
        ):
            if len(step.allowed_token_ids) == 1 and (
                log_probability_hex != zero_hex or entropy_hex != zero_hex
            ):
                raise AuthenticatedSamplingError(
                    "forced tokens must record exactly zero log probability and entropy"
                )

        try:
            action = parse_action(trace.raw_action)
        except ValueError as exc:
            raise AuthenticatedSamplingError("sampled action is not canonical") from exc
        if self.mode == "inquiry" and type(action) not in {ReadyAction, TestAction}:
            raise AuthenticatedSamplingError("inquiry sample contains a terminal answer")
        if self.mode == "answer":
            if type(action) is not AnswerAction:
                raise AuthenticatedSamplingError("answer sample contains an inquiry action")
            if len(action.classifications) != self.terminal_count:
                raise AuthenticatedSamplingError("answer classification count differs from terminal_count")

    @property
    def temperature(self) -> float:
        return self.detached_statistics.temperature

    def as_obj(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "mode": self.mode,
            "terminal_count": self.terminal_count,
            "policy_state_digest": self.policy_state_digest,
            "tokenizer_binding_digest": self.tokenizer_binding_digest,
            "compiler_manifest_digest": self.compiler_manifest_digest,
            "turn_seed": {**self.turn_seed.as_obj(), "digest": self.turn_seed.digest},
            "temperature_hex": self.temperature_hex,
            "selected_token_ids": list(self.selected_token_ids),
            "draws": [{**draw.as_obj(), "digest": draw.digest} for draw in self.draws],
            "decision_verification_digest": self.decision_verification_digest,
            "decision_example": {
                **self.decision_example.as_obj(),
                "digest": self.decision_example.digest,
            },
            "detached_statistics": {
                **self.detached_statistics.as_obj(),
                "digest": self.detached_statistics.digest,
            },
        }

    @property
    def digest(self) -> str:
        return json_digest(self.as_obj(), domain=_SAMPLE_DIGEST_DOMAIN)

    def to_json(self) -> str:
        return dump_json({**self.as_obj(), "digest": self.digest})

    @classmethod
    def from_obj(cls, value: object) -> AuthenticatedActionSample:
        obj = _require_exact_object(
            value,
            (
                "schema_version",
                "contract_id",
                "mode",
                "terminal_count",
                "policy_state_digest",
                "tokenizer_binding_digest",
                "compiler_manifest_digest",
                "turn_seed",
                "temperature_hex",
                "selected_token_ids",
                "draws",
                "decision_verification_digest",
                "decision_example",
                "detached_statistics",
                "digest",
            ),
            name="authenticated_action_sample",
        )
        turn_seed_obj = _require_exact_object(
            obj["turn_seed"],
            (
                "schema_version",
                "contract_id",
                "run_seed",
                "episode_digest",
                "rollout_index",
                "turn_index",
                "policy_state_digest",
                "digest",
            ),
            name="turn_seed",
        )
        supplied_turn_seed_digest = _require_sha256(turn_seed_obj["digest"], name="turn_seed.digest")
        raw_turn_seed = {key: item for key, item in turn_seed_obj.items() if key != "digest"}
        try:
            turn_seed = policy_turn_seed_from_obj(raw_turn_seed)
        except (PolicyRandomnessError, TypeError, ValueError) as exc:
            raise AuthenticatedSamplingError("turn seed is invalid") from exc
        if not hmac.compare_digest(supplied_turn_seed_digest, turn_seed.digest):
            raise AuthenticatedSamplingError("turn-seed digest check failed")

        parsed_draws: list[PolicyTokenDraw] = []
        for index, raw_draw in enumerate(_require_list(obj["draws"], name="draws")):
            draw_obj = _require_exact_object(
                raw_draw,
                (
                    "schema_version",
                    "turn_seed_digest",
                    "action_token_index",
                    "word_u53",
                    "digest",
                ),
                name=f"draws[{index}]",
            )
            supplied_draw_digest = _require_sha256(draw_obj["digest"], name=f"draws[{index}].digest")
            raw_draw_obj = {key: item for key, item in draw_obj.items() if key != "digest"}
            try:
                draw = policy_token_draw_from_obj(raw_draw_obj)
            except (PolicyRandomnessError, TypeError, ValueError) as exc:
                raise AuthenticatedSamplingError(f"draws[{index}] is invalid") from exc
            if not hmac.compare_digest(supplied_draw_digest, draw.digest):
                raise AuthenticatedSamplingError(f"draws[{index}] digest check failed")
            parsed_draws.append(draw)

        try:
            decision_example = DecisionTokenExample.from_obj(obj["decision_example"])
        except (DecisionEncodingError, TypeError, ValueError) as exc:
            raise AuthenticatedSamplingError("decision example is invalid") from exc
        detached = _detached_statistics_from_obj(obj["detached_statistics"])
        selected_items = _require_list(obj["selected_token_ids"], name="selected_token_ids")
        result = cls(
            schema_version=_require_integer(obj["schema_version"], name="schema_version", minimum=1),
            contract_id=_require_text(obj["contract_id"], name="contract_id"),
            mode=cast(SamplingMode, _require_text(obj["mode"], name="mode")),
            terminal_count=_require_optional_integer(obj["terminal_count"], name="terminal_count"),
            policy_state_digest=_require_sha256(obj["policy_state_digest"], name="policy_state_digest"),
            tokenizer_binding_digest=_require_sha256(
                obj["tokenizer_binding_digest"], name="tokenizer_binding_digest"
            ),
            compiler_manifest_digest=_require_sha256(
                obj["compiler_manifest_digest"], name="compiler_manifest_digest"
            ),
            turn_seed=turn_seed,
            temperature_hex=_require_text(obj["temperature_hex"], name="temperature_hex"),
            selected_token_ids=tuple(
                _require_integer(item, name="selected token id") for item in selected_items
            ),
            draws=tuple(parsed_draws),
            decision_verification_digest=_require_sha256(
                obj["decision_verification_digest"],
                name="decision_verification_digest",
            ),
            decision_example=decision_example,
            detached_statistics=detached,
        )
        supplied_digest = _require_sha256(obj["digest"], name="sample.digest")
        if not hmac.compare_digest(supplied_digest, result.digest):
            raise AuthenticatedSamplingError("authenticated-sample digest check failed")
        return result

    @classmethod
    def from_json(cls, text: str) -> AuthenticatedActionSample:
        if type(text) is not str:
            raise AuthenticatedSamplingError("authenticated sample JSON must be text")
        try:
            value = load_json(text)
        except CanonicalJSONError as exc:
            raise AuthenticatedSamplingError("authenticated sample is not strict JSON") from exc
        result = cls.from_obj(value)
        if not hmac.compare_digest(text.encode("utf-8"), result.to_json().encode("utf-8")):
            raise AuthenticatedSamplingError(
                "authenticated sample JSON is not in the unique canonical byte representation"
            )
        return result


@dataclass(frozen=True, slots=True, init=False)
class VerifiedAuthenticatedActionSample:
    """Nominal evidence of exact source, policy, and differentiable replay."""

    sample: AuthenticatedActionSample
    verified_decision: VerifiedDecisionTokenExample
    replayed_statistics: MaskedActionStatistics
    full_forward_logits_reference: weakref.ReferenceType[torch.Tensor]
    absolute_tolerance_hex: str
    verification_digest: str

    @classmethod
    def _from_verified(
        cls,
        sample: AuthenticatedActionSample,
        verified_decision: VerifiedDecisionTokenExample,
        replayed_statistics: MaskedActionStatistics,
        full_forward_logits: torch.Tensor,
        *,
        absolute_tolerance: float,
    ) -> VerifiedAuthenticatedActionSample:
        result = object.__new__(cls)
        object.__setattr__(result, "sample", sample)
        object.__setattr__(result, "verified_decision", verified_decision)
        object.__setattr__(result, "replayed_statistics", replayed_statistics)
        object.__setattr__(
            result,
            "full_forward_logits_reference",
            weakref.ref(full_forward_logits),
        )
        object.__setattr__(
            result,
            "absolute_tolerance_hex",
            float(absolute_tolerance).hex(),
        )
        object.__setattr__(
            result,
            "verification_digest",
            json_digest(
                {
                    "sample_digest": sample.digest,
                    "decision_verification_digest": verified_decision.verification_digest,
                    "policy_state_digest": sample.policy_state_digest,
                    "compiler_manifest_digest": sample.compiler_manifest_digest,
                    "absolute_tolerance_hex": result.absolute_tolerance_hex,
                },
                domain=_VERIFICATION_DIGEST_DOMAIN,
            ),
        )
        return result

    @property
    def digest(self) -> str:
        return self.sample.digest


@dataclass(frozen=True, slots=True, init=False)
class GraphFreeVerifiedAuthenticatedActionSample:
    """Nominal graph-free evidence copied from one successful live replay.

    ``VerifiedAuthenticatedActionSample`` deliberately remains the ephemeral
    graph-bearing result used by direct replay and backward execution.  This
    compact wrapper retains only canonical structural evidence and the same
    verification digest; it cannot carry replay tensors into a rollout.
    """

    sample: AuthenticatedActionSample
    verified_decision: VerifiedDecisionTokenExample
    absolute_tolerance_hex: str
    verification_digest: str

    @classmethod
    def _from_verified(
        cls,
        verified: VerifiedAuthenticatedActionSample,
    ) -> GraphFreeVerifiedAuthenticatedActionSample:
        if type(verified) is not VerifiedAuthenticatedActionSample:
            raise TypeError("verified must be a VerifiedAuthenticatedActionSample")
        sample = verified.sample
        verified_decision = verified.verified_decision
        if type(sample) is not AuthenticatedActionSample:
            raise AuthenticatedSamplingError("verified replay contains an invalid sample")
        if type(verified_decision) is not VerifiedDecisionTokenExample:
            raise AuthenticatedSamplingError("verified replay contains invalid decision evidence")
        if verified_decision.example != sample.decision_example:
            raise AuthenticatedSamplingError("verified decision belongs to another sample")
        if verified_decision.verification_digest != sample.decision_verification_digest:
            raise AuthenticatedSamplingError("verified decision digest differs from the sample")
        expected_digest = json_digest(
            {
                "sample_digest": sample.digest,
                "decision_verification_digest": verified_decision.verification_digest,
                "policy_state_digest": sample.policy_state_digest,
                "compiler_manifest_digest": sample.compiler_manifest_digest,
                "absolute_tolerance_hex": verified.absolute_tolerance_hex,
            },
            domain=_VERIFICATION_DIGEST_DOMAIN,
        )
        if not hmac.compare_digest(expected_digest, verified.verification_digest):
            raise AuthenticatedSamplingError("verified action digest does not regenerate")
        result = object.__new__(cls)
        object.__setattr__(result, "sample", sample)
        object.__setattr__(result, "verified_decision", verified_decision)
        object.__setattr__(result, "absolute_tolerance_hex", verified.absolute_tolerance_hex)
        object.__setattr__(result, "verification_digest", verified.verification_digest)
        return result

    @property
    def digest(self) -> str:
        return self.sample.digest


def sample_authenticated_action(
    provider: NextTokenLogitsProvider,
    tokenizer: ExactDecodeTokenizerProtocol,
    compiler: FragmentActionTokenCompiler,
    dialogue: Dialogue,
    turn_seed: PolicyTurnSeed,
    *,
    mode: SamplingMode,
    terminal_count: int | None,
    temperature: float,
    maximum_sequence_tokens: int,
) -> AuthenticatedActionSample:
    """Sample one exact grammar action under a frozen, unchanged policy state."""

    if not isinstance(provider, NextTokenLogitsProvider):
        raise TypeError("provider must implement NextTokenLogitsProvider")
    if type(compiler) is not FragmentActionTokenCompiler:
        raise TypeError("compiler must be a FragmentActionTokenCompiler")
    if type(turn_seed) is not PolicyTurnSeed:
        raise TypeError("turn_seed must be a PolicyTurnSeed")
    selected_temperature = _require_temperature(temperature)
    _validate_dialogue_mode(dialogue, mode)
    if mode == "inquiry":
        if terminal_count is not None:
            raise AuthenticatedSamplingError("inquiry sampling cannot set terminal_count")
        state: _GrammarState = InquiryActionState()
    elif mode == "answer":
        terminal = _require_integer(terminal_count, name="terminal_count", minimum=1)
        state = AnswerActionState(terminal)
    else:
        raise AuthenticatedSamplingError("mode must be inquiry or answer")
    if _provider_policy_state(provider) != turn_seed.policy_state_digest:
        raise AuthenticatedSamplingError("logits provider policy state differs from the registered turn seed")

    manifest = compiler.manifest
    prompt_text, prompt_token_ids = render_generation_prefix(tokenizer, dialogue)
    if not prompt_text or not prompt_token_ids:
        raise AuthenticatedSamplingError("decision generation prompt must be nonempty")

    action_ids: list[int] = []
    masks: list[tuple[int, ...]] = []
    draws: list[PolicyTokenDraw] = []
    log_probabilities: list[float] = []
    entropies: list[float] = []

    def sample_segment(segment: GrammarSegment) -> str:
        compiled = compiler.compile_segment(segment)
        sampled = _sample_compiled_fragment(
            provider,
            compiled,
            prompt_token_ids,
            tuple(action_ids),
            turn_seed,
            temperature=selected_temperature,
            maximum_action_tokens=manifest.maximum_action_tokens,
        )
        action_ids.extend(sampled.token_ids)
        masks.extend(sampled.allowed_token_ids)
        draws.extend(sampled.draws)
        log_probabilities.extend(sampled.token_log_probabilities)
        entropies.extend(sampled.token_entropies)
        return sampled.selected_key

    while not state.complete:
        segment = state.next_segment()
        state = state.choose(sample_segment(segment))
    completion = GrammarSegment(
        "$completion",
        "",
        (GrammarOption("completion", state.completion_suffix),),
    )
    if sample_segment(completion) != "completion":
        raise AuthenticatedSamplingError("completion fragment selected an invalid option")

    action = action_from_state(state)
    try:
        trace = compiler.trace_action(action)
    except FragmentActionTokenizationError as exc:
        raise AuthenticatedSamplingError("sampled action failed exact compiler regeneration") from exc
    if trace.action_token_ids != tuple(action_ids):
        raise AuthenticatedSamplingError("sampled token ids differ from exact compiler regeneration")
    if trace.allowed_token_ids != tuple(masks):
        raise AuthenticatedSamplingError("sampled legal masks differ from exact compiler regeneration")
    if tuple(draw.action_token_index for draw in draws) != tuple(range(len(action_ids))):
        raise AuthenticatedSamplingError("sampled policy draws are not contiguous")

    try:
        example = encode_decision_example(
            tokenizer,
            compiler,
            dialogue,
            action,
            tokenizer_binding_digest=manifest.tokenizer_identifier,
            maximum_sequence_tokens=maximum_sequence_tokens,
        )
        verified = verify_decision_example(example, tokenizer, compiler, dialogue)
    except DecisionOverlengthError as exc:
        raise AuthenticatedSamplingOverlengthError(
            "sampled decision exceeds maximum_sequence_tokens; truncation is forbidden"
        ) from exc
    except DecisionEncodingError as exc:
        raise AuthenticatedSamplingError("sampled decision failed exact prompt/action verification") from exc
    if example.action_trace.to_json() != trace.to_json():
        raise AuthenticatedSamplingError("decision example trace differs from sampled compiler trace")
    detached = _detached_statistics_from_sampling(
        verified,
        temperature=selected_temperature,
        log_probabilities=tuple(log_probabilities),
        entropies=tuple(entropies),
    )
    if _provider_policy_state(provider) != turn_seed.policy_state_digest:
        raise AuthenticatedSamplingError("policy state changed while sampling the action")
    return AuthenticatedActionSample(
        mode=mode,
        terminal_count=terminal_count,
        policy_state_digest=turn_seed.policy_state_digest,
        tokenizer_binding_digest=manifest.tokenizer_identifier,
        compiler_manifest_digest=manifest.digest,
        turn_seed=turn_seed,
        temperature_hex=selected_temperature.hex(),
        selected_token_ids=tuple(action_ids),
        draws=tuple(draws),
        decision_verification_digest=verified.verification_digest,
        decision_example=example,
        detached_statistics=detached,
    )


def replay_authenticated_sample(
    sample: AuthenticatedActionSample,
    provider: FullForwardLogitsProvider,
    tokenizer: ExactDecodeTokenizerProtocol,
    compiler: FragmentActionTokenCompiler,
    dialogue: Dialogue,
    *,
    absolute_tolerance: float,
) -> VerifiedAuthenticatedActionSample:
    """Regenerate sampling and compare same-policy differentiable replay."""

    if type(sample) is not AuthenticatedActionSample:
        raise TypeError("sample must be an AuthenticatedActionSample")
    if not isinstance(provider, FullForwardLogitsProvider):
        raise TypeError("provider must implement FullForwardLogitsProvider")
    if (
        isinstance(absolute_tolerance, bool)
        or not isinstance(absolute_tolerance, (int, float))
        or not math.isfinite(float(absolute_tolerance))
        or float(absolute_tolerance) < 0
    ):
        raise AuthenticatedSamplingError("absolute_tolerance must be finite and non-negative")
    if _provider_policy_state(provider) != sample.policy_state_digest:
        raise AuthenticatedSamplingError("replay provider policy state differs from the sampled policy")
    manifest = compiler.manifest
    if manifest.digest != sample.compiler_manifest_digest:
        raise AuthenticatedSamplingError("replay compiler differs from the sampled compiler manifest")
    if manifest.tokenizer_identifier != sample.tokenizer_binding_digest:
        raise AuthenticatedSamplingError("replay tokenizer binding differs from the sampled tokenizer")
    try:
        verified_decision = verify_decision_example(
            sample.decision_example,
            tokenizer,
            compiler,
            dialogue,
        )
    except DecisionEncodingError as exc:
        raise AuthenticatedSamplingError("sample decision failed exact source regeneration") from exc
    if verified_decision.verification_digest != sample.decision_verification_digest:
        raise AuthenticatedSamplingError("decision verification digest differs from source regeneration")

    regenerated = sample_authenticated_action(
        provider,
        tokenizer,
        compiler,
        dialogue,
        sample.turn_seed,
        mode=sample.mode,
        terminal_count=sample.terminal_count,
        temperature=sample.temperature,
        maximum_sequence_tokens=sample.decision_example.maximum_sequence_tokens,
    )
    if not hmac.compare_digest(regenerated.to_json().encode("utf-8"), sample.to_json().encode("utf-8")):
        raise AuthenticatedSamplingError("sample differs from stateless same-policy token regeneration")

    if _provider_policy_state(provider) != sample.policy_state_digest:
        raise AuthenticatedSamplingError("policy state changed before full-forward replay")
    try:
        full_logits = provider.full_forward_logits(sample.decision_example.input_ids)
    except Exception as exc:
        raise AuthenticatedSamplingError("full-forward logits provider failed") from exc
    if _provider_policy_state(provider) != sample.policy_state_digest:
        raise AuthenticatedSamplingError("policy state changed during full-forward replay")
    if not isinstance(full_logits, torch.Tensor):
        raise AuthenticatedSamplingError("full_forward_logits must return a Tensor")
    if not full_logits.requires_grad:
        raise AuthenticatedSamplingError("full-forward logits must retain a differentiable computation graph")
    try:
        replayed = _differentiable_float64_masked_statistics(
            full_logits,
            verified_decision,
            temperature=sample.temperature,
        )
        compare_detached_masked_statistics(
            sample.detached_statistics,
            replayed,
            verified_decision,
            temperature=sample.temperature,
            absolute_tolerance=float(absolute_tolerance),
        )
        if not replayed.sequence_log_probability.requires_grad:
            raise AuthenticatedSamplingError("masked full-forward replay is not differentiable")
    except (DecisionEncodingError, RuntimeError, TypeError, ValueError) as exc:
        raise AuthenticatedSamplingError(
            "detached sampling statistics differ from differentiable full-forward replay"
        ) from exc
    return VerifiedAuthenticatedActionSample._from_verified(
        sample,
        verified_decision,
        replayed,
        full_logits,
        absolute_tolerance=float(absolute_tolerance),
    )


def verify_eight_rollout_policy_group(
    verified_samples: Sequence[GraphFreeVerifiedAuthenticatedActionSample],
) -> str:
    """Bind eight graph-free, replay-verified rollouts to one behavior policy.

    The input may contain multiple turns per rollout and may arrive in any
    scheduling order. Every rollout index must occur, its turn indices must be
    unique and contiguous from zero, and temperature is frozen with policy
    state. Raw, merely structural records are deliberately rejected.
    """

    verified_records = tuple(verified_samples)
    if not verified_records or any(
        type(verified) is not GraphFreeVerifiedAuthenticatedActionSample for verified in verified_records
    ):
        raise AuthenticatedSamplingError(
            "rollout group must contain nominal graph-free replay-verified action evidence"
        )
    records = tuple(verified.sample for verified in verified_records)
    rollout_indices = {sample.turn_seed.rollout_index for sample in records}
    if rollout_indices != set(range(ROLLOUTS_PER_EPISODE)):
        raise AuthenticatedSamplingError("rollout group must contain all eight rollout indices")
    coordinates = [(sample.turn_seed.rollout_index, sample.turn_seed.turn_index) for sample in records]
    if len(set(coordinates)) != len(coordinates):
        raise AuthenticatedSamplingError("rollout group contains duplicate turn coordinates")
    for rollout_index in range(ROLLOUTS_PER_EPISODE):
        turn_indices = sorted(
            turn_index for selected_rollout, turn_index in coordinates if selected_rollout == rollout_index
        )
        if turn_indices != list(range(turn_indices[-1] + 1)):
            raise AuthenticatedSamplingError(
                "each rollout must contain contiguous turn indices starting at zero"
            )

    reference = records[0]
    invariant_fields = (
        "run_seed",
        "episode_digest",
        "policy_state_digest",
    )
    for sample in records[1:]:
        for name in invariant_fields:
            if getattr(sample.turn_seed, name) != getattr(reference.turn_seed, name):
                raise AuthenticatedSamplingError(f"eight-rollout group changed invariant {name}")
        if sample.tokenizer_binding_digest != reference.tokenizer_binding_digest:
            raise AuthenticatedSamplingError("eight-rollout group changed tokenizer binding")
        if sample.compiler_manifest_digest != reference.compiler_manifest_digest:
            raise AuthenticatedSamplingError("eight-rollout group changed compiler manifest")
        if sample.temperature_hex != reference.temperature_hex:
            raise AuthenticatedSamplingError("eight-rollout group changed behavior-policy temperature")

    answer_terminal_counts = {sample.terminal_count for sample in records if sample.mode == "answer"}
    if len(answer_terminal_counts) > 1:
        raise AuthenticatedSamplingError("eight-rollout group changed answer terminal_count")
    verification_tolerances = {verified.absolute_tolerance_hex for verified in verified_records}
    if len(verification_tolerances) != 1:
        raise AuthenticatedSamplingError("eight-rollout group changed full-forward replay tolerance")

    ordered = sorted(
        verified_records,
        key=lambda verified: (
            verified.sample.turn_seed.rollout_index,
            verified.sample.turn_seed.turn_index,
        ),
    )
    return json_digest(
        {
            "run_seed": reference.turn_seed.run_seed,
            "episode_digest": reference.turn_seed.episode_digest,
            "policy_state_digest": reference.policy_state_digest,
            "tokenizer_binding_digest": reference.tokenizer_binding_digest,
            "compiler_manifest_digest": reference.compiler_manifest_digest,
            "temperature_hex": reference.temperature_hex,
            "answer_terminal_count": (
                None if not answer_terminal_counts else next(iter(answer_terminal_counts))
            ),
            "absolute_tolerance_hex": next(iter(verification_tolerances)),
            "verified_samples": [
                {
                    "sample_digest": verified.sample.digest,
                    "verification_digest": verified.verification_digest,
                }
                for verified in ordered
            ],
        },
        domain=_ROLLOUT_GROUP_DIGEST_DOMAIN,
    )


def authenticated_sampler_manifest() -> dict[str, object]:
    """Describe the fixed, local-only semantics of this sampler slice."""

    return {
        "schema_version": AUTHENTICATED_SAMPLER_SCHEMA_VERSION,
        "contract_id": AUTHENTICATED_SAMPLER_CONTRACT_ID,
        "legal_masks": "sorted exact trie children from frozen compiled fragments",
        "sampling": (
            "float64 masked softmax weights accumulated as exact Fractions against "
            "independently addressed open-unit policy draws"
        ),
        "forced_tokens": "exactly zero log probability and entropy",
        "record": (
            "policy state, turn seed/draws, tokenizer/compiler, temperature, selected "
            "tokens, verified decision evidence, and detached masked statistics"
        ),
        "acceptance": (
            "strict source regeneration, stateless same-policy resampling, and "
            "differentiable same-policy full-forward masked replay"
        ),
        "live_replay_evidence": "ephemeral VerifiedAuthenticatedActionSample",
        "retained_action_evidence": "GraphFreeVerifiedAuthenticatedActionSample",
        "retained_differentiable_graphs": False,
        "rollout_group_size": ROLLOUTS_PER_EPISODE,
        "stateful_rng_present": False,
        "live_model_authorization": False,
        "weight_update_authorization": False,
    }
