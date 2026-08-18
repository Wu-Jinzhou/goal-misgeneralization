"""Schema-v2 authenticated collection and replay for G03 rollouts.

This local-only module composes the hidden-law environment with the
authenticated token sampler.  Every model action is sampled from an exactly
rendered public dialogue, replay-verified under the unchanged behavior policy,
and only then consumed by the environment.  Operational controller stops are
recorded as zero-reward aborts.  Scientific or provenance failures fail closed
and return no acceptable rollout.

Nothing in this module authorizes a live model, a launch, or a weight update.
"""

from __future__ import annotations

import gc
import hmac
import math
import weakref
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal, Protocol, cast, runtime_checkable

import torch

from ._json import CanonicalJSONError, dump_json, json_digest, load_json
from .action_tokenization_v2 import (
    ExactDecodeTokenizerProtocol,
    FragmentActionTokenCompiler,
    FragmentActionTokenizationError,
)
from .actions import serialize_action
from .authenticated_sampler_v2 import (
    AuthenticatedActionSample,
    AuthenticatedSamplingError,
    AuthenticatedSamplingOverlengthError,
    FullForwardLogitsProvider,
    GraphFreeVerifiedAuthenticatedActionSample,
    VerifiedAuthenticatedActionSample,
    replay_authenticated_sample,
    sample_authenticated_action,
    verify_eight_rollout_policy_group,
)
from .dialogue import Dialogue, dialogue_as_obj, render_dialogue
from .environment import HiddenLawEnvironment, TranscriptReplayError, replay_transcript
from .episodes import HiddenEpisode
from .policy_randomness_v2 import (
    MAXIMUM_TURNS_PER_ROLLOUT,
    ROLLOUTS_PER_EPISODE,
    PolicyRandomnessError,
    PolicyTurnSeed,
)
from .transcripts import (
    ABORT_REASONS,
    AbortEvent,
    AbortReason,
    AnswerEvent,
    ReadyEvent,
    TestEvent,
    Transcript,
    TranscriptEvent,
    event_from_obj,
    parse_transcript,
)

AUTHENTICATED_ROLLOUT_SCHEMA_VERSION = 2
AUTHENTICATED_ROLLOUT_CONTRACT_ID = "goalzendo-authenticated-rollout-v2"
MODEL_POLICY_PROVENANCE_SCHEMA_VERSION = 1
MODEL_POLICY_PROVENANCE_CONTRACT_ID = "goalzendo-model-policy-provenance-v1"

_MODEL_PROVENANCE_DIGEST_DOMAIN = "goalzendo-interactive-model-policy-provenance-v1"
_ROLLOUT_TURN_DIGEST_DOMAIN = "goalzendo-interactive-authenticated-rollout-turn-v2"
_ROLLOUT_DIGEST_DOMAIN = "goalzendo-interactive-authenticated-rollout-record-v2"
_ROLLOUT_VERIFICATION_DIGEST_DOMAIN = "goalzendo-interactive-authenticated-rollout-replay-v2"
_ROLLOUT_GROUP_DIGEST_DOMAIN = "goalzendo-interactive-authenticated-rollout-group-v2"
_ROLLOUT_ACTION_EVIDENCE_GROUP_DIGEST_DOMAIN = (
    "goalzendo-interactive-authenticated-rollout-action-evidence-group-v2"
)
_DIALOGUE_DIGEST_DOMAIN = "goalzendo-interactive-authenticated-rollout-dialogue-v2"

TerminationOrigin = Literal["controller", "sampler", "turn_budget"]
_TERMINATION_ORIGINS: tuple[TerminationOrigin, ...] = (
    "controller",
    "sampler",
    "turn_budget",
)


class AuthenticatedRolloutError(ValueError):
    """Raised when an authenticated rollout cannot be accepted exactly."""


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: object, *, name: str) -> str:
    if not _is_sha256(value):
        raise AuthenticatedRolloutError(f"{name} must be a lowercase SHA-256")
    return cast(str, value)


def _require_text(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise AuthenticatedRolloutError(f"{name} must be nonempty text")
    return value


def _require_ascii(value: object, *, name: str) -> str:
    result = _require_text(value, name=name)
    try:
        result.encode("ascii")
    except UnicodeEncodeError as exc:
        raise AuthenticatedRolloutError(f"{name} must be ASCII") from exc
    return result


def _require_integer(
    value: object,
    *,
    name: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AuthenticatedRolloutError(f"{name} must be an integer greater than or equal to {minimum}")
    if maximum is not None and value > maximum:
        raise AuthenticatedRolloutError(f"{name} must be at most {maximum}")
    return value


def _require_exact_object(
    value: object,
    fields: tuple[str, ...],
    *,
    name: str,
) -> dict[str, object]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise AuthenticatedRolloutError(f"{name} must be a JSON object")
    result = cast(dict[str, object], value)
    if set(result) != set(fields) or len(result) != len(fields):
        raise AuthenticatedRolloutError(f"{name} has noncanonical fields")
    return result


def _require_list(value: object, *, name: str) -> list[object]:
    if type(value) is not list:
        raise AuthenticatedRolloutError(f"{name} must be a JSON array")
    return cast(list[object], value)


def _canonical_positive_float_hex(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise AuthenticatedRolloutError(f"{name} must be canonical float.hex text")
    try:
        parsed = float.fromhex(value)
    except ValueError as exc:
        raise AuthenticatedRolloutError(f"{name} is not valid float.hex text") from exc
    if not math.isfinite(parsed) or parsed <= 0 or parsed.hex() != value:
        raise AuthenticatedRolloutError(f"{name} must encode a finite positive float canonically")
    return value


def _canonical_nonnegative_float_hex(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise AuthenticatedRolloutError(f"{name} must be canonical float.hex text")
    try:
        parsed = float.fromhex(value)
    except ValueError as exc:
        raise AuthenticatedRolloutError(f"{name} is not valid float.hex text") from exc
    if not math.isfinite(parsed) or parsed < 0 or parsed.hex() != value:
        raise AuthenticatedRolloutError(f"{name} must encode a finite non-negative float canonically")
    return value


def _is_immutable_revision(value: object) -> bool:
    return (
        type(value) is str
        and len(value) in {40, 64}
        and all(character in "0123456789abcdef" for character in value)
    )


def _dialogue_digest(dialogue: Dialogue) -> str:
    return json_digest(dialogue_as_obj(dialogue), domain=_DIALOGUE_DIGEST_DOMAIN)


def _event_raw_action(event: TranscriptEvent) -> str | None:
    if type(event) in {TestEvent, ReadyEvent, AnswerEvent}:
        action_event = cast(TestEvent | ReadyEvent | AnswerEvent, event)
        return serialize_action(action_event.action)
    return None


def _terminal_reward_fraction(transcript: Transcript) -> Fraction:
    if not transcript.events or type(transcript.events[-1]) is not AnswerEvent:
        return Fraction(0, 1)
    score = transcript.events[-1].score
    return (
        Fraction(7 * score.classification_correct, 10 * score.classification_total)
        + Fraction(int(score.rule_equivalent), 4)
        + Fraction(1, 20) * (1 - Fraction(score.query_count, 6))
    )


@dataclass(frozen=True, slots=True)
class ModelPolicyProvenance:
    """Immutable model artifacts and the exact behavior-policy state."""

    model_identifier: str
    revision: str
    artifact_manifest_sha256: str
    runtime_stack_sha256: str
    policy_state_digest: str
    schema_version: int = MODEL_POLICY_PROVENANCE_SCHEMA_VERSION
    contract_id: str = MODEL_POLICY_PROVENANCE_CONTRACT_ID

    def __post_init__(self) -> None:
        if self.schema_version != MODEL_POLICY_PROVENANCE_SCHEMA_VERSION:
            raise AuthenticatedRolloutError("unexpected model-provenance schema version")
        if self.contract_id != MODEL_POLICY_PROVENANCE_CONTRACT_ID:
            raise AuthenticatedRolloutError("unexpected model-provenance contract id")
        _require_ascii(self.model_identifier, name="model_identifier")
        if any(character.isspace() for character in self.model_identifier):
            raise AuthenticatedRolloutError("model_identifier may not contain whitespace")
        if not _is_immutable_revision(self.revision):
            raise AuthenticatedRolloutError(
                "model revision must be a lowercase 40- or 64-hex immutable commit"
            )
        for name in (
            "artifact_manifest_sha256",
            "runtime_stack_sha256",
            "policy_state_digest",
        ):
            _require_sha256(getattr(self, name), name=name)

    def as_obj(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "model_identifier": self.model_identifier,
            "revision": self.revision,
            "artifact_manifest_sha256": self.artifact_manifest_sha256,
            "runtime_stack_sha256": self.runtime_stack_sha256,
            "policy_state_digest": self.policy_state_digest,
        }

    @property
    def digest(self) -> str:
        return json_digest(self.as_obj(), domain=_MODEL_PROVENANCE_DIGEST_DOMAIN)

    @classmethod
    def from_obj(cls, value: object) -> ModelPolicyProvenance:
        obj = _require_exact_object(
            value,
            (
                "schema_version",
                "contract_id",
                "model_identifier",
                "revision",
                "artifact_manifest_sha256",
                "runtime_stack_sha256",
                "policy_state_digest",
                "digest",
            ),
            name="model_provenance",
        )
        result = cls(
            model_identifier=_require_text(obj["model_identifier"], name="model_identifier"),
            revision=_require_text(obj["revision"], name="model revision"),
            artifact_manifest_sha256=_require_sha256(
                obj["artifact_manifest_sha256"], name="artifact_manifest_sha256"
            ),
            runtime_stack_sha256=_require_sha256(obj["runtime_stack_sha256"], name="runtime_stack_sha256"),
            policy_state_digest=_require_sha256(obj["policy_state_digest"], name="policy_state_digest"),
            schema_version=_require_integer(
                obj["schema_version"], name="model_provenance.schema_version", minimum=1
            ),
            contract_id=_require_text(obj["contract_id"], name="model_provenance.contract_id"),
        )
        supplied = _require_sha256(obj["digest"], name="model_provenance.digest")
        if not hmac.compare_digest(supplied, result.digest):
            raise AuthenticatedRolloutError("model-provenance digest check failed")
        return result


@runtime_checkable
class ProvenancedFullForwardLogitsProvider(FullForwardLogitsProvider, Protocol):
    """A sampler provider bound to an independently computed model manifest."""

    @property
    def model_provenance_digest(self) -> str: ...


def _provider_policy_state(provider: object) -> str:
    try:
        value = provider.policy_state_digest  # type: ignore[attr-defined]
    except (AttributeError, TypeError, ValueError) as exc:
        raise AuthenticatedRolloutError("provider does not expose policy_state_digest") from exc
    return _require_sha256(value, name="provider.policy_state_digest")


def _provider_model_provenance(provider: object) -> str:
    try:
        value = provider.model_provenance_digest  # type: ignore[attr-defined]
    except (AttributeError, TypeError, ValueError) as exc:
        raise AuthenticatedRolloutError("provider does not expose model_provenance_digest") from exc
    return _require_sha256(value, name="provider.model_provenance_digest")


class _ProvenanceGuardProvider:
    """Check both policy and model identity around every provider invocation."""

    __slots__ = ("_expected_model", "_expected_policy", "_provider")

    def __init__(
        self,
        provider: ProvenancedFullForwardLogitsProvider,
        provenance: ModelPolicyProvenance,
    ) -> None:
        self._provider = provider
        self._expected_policy = provenance.policy_state_digest
        self._expected_model = provenance.digest
        self._check()

    @property
    def policy_state_digest(self) -> str:
        self._check()
        return self._expected_policy

    @property
    def model_provenance_digest(self) -> str:
        self._check()
        return self._expected_model

    def _check(self) -> None:
        if _provider_policy_state(self._provider) != self._expected_policy:
            raise AuthenticatedRolloutError("provider policy state differs from model provenance")
        if _provider_model_provenance(self._provider) != self._expected_model:
            raise AuthenticatedRolloutError("provider model provenance changed or does not match")

    def next_token_logits(self, input_ids: tuple[int, ...]) -> torch.Tensor:
        self._check()
        result = self._provider.next_token_logits(input_ids)
        self._check()
        return result

    def full_forward_logits(self, input_ids: tuple[int, ...]) -> torch.Tensor:
        self._check()
        result = self._provider.full_forward_logits(input_ids)
        self._check()
        return result


@runtime_checkable
class RolloutControllerProtocol(Protocol):
    """Operational stop polling; ``incomplete`` is explicit cancellation."""

    def abort_reason(
        self,
        *,
        episode_digest: str,
        rollout_index: int,
        turn_index: int,
        dialogue_digest: str,
    ) -> AbortReason | None: ...


@dataclass(frozen=True, slots=True)
class RolloutTermination:
    """One controller-side stop at the next unsampled policy coordinate."""

    origin: TerminationOrigin
    reason: AbortReason
    turn_index: int
    dialogue_digest: str

    def __post_init__(self) -> None:
        if self.origin not in _TERMINATION_ORIGINS:
            raise AuthenticatedRolloutError("unknown rollout termination origin")
        if self.reason not in ABORT_REASONS:
            raise AuthenticatedRolloutError("unknown rollout termination reason")
        _require_integer(
            self.turn_index,
            name="termination.turn_index",
            maximum=MAXIMUM_TURNS_PER_ROLLOUT,
        )
        _require_sha256(self.dialogue_digest, name="termination.dialogue_digest")
        if self.origin == "turn_budget" and self.reason != "incomplete":
            raise AuthenticatedRolloutError("turn-budget termination must be incomplete")
        if self.origin == "sampler" and self.reason != "overlength":
            raise AuthenticatedRolloutError("sampler termination may only record typed overlength")
        if self.origin == "controller" and self.reason not in {"incomplete", "timed_out"}:
            raise AuthenticatedRolloutError("controller termination may only record cancellation or timeout")

    def as_obj(self) -> dict[str, object]:
        return {
            "origin": self.origin,
            "reason": self.reason,
            "turn_index": self.turn_index,
            "dialogue_digest": self.dialogue_digest,
        }

    @classmethod
    def from_obj(cls, value: object) -> RolloutTermination:
        obj = _require_exact_object(
            value,
            ("origin", "reason", "turn_index", "dialogue_digest"),
            name="termination",
        )
        return cls(
            origin=cast(TerminationOrigin, _require_text(obj["origin"], name="termination.origin")),
            reason=cast(AbortReason, _require_text(obj["reason"], name="termination.reason")),
            turn_index=_require_integer(
                obj["turn_index"],
                name="termination.turn_index",
                maximum=MAXIMUM_TURNS_PER_ROLLOUT,
            ),
            dialogue_digest=_require_sha256(obj["dialogue_digest"], name="termination.dialogue_digest"),
        )


@dataclass(frozen=True, slots=True)
class AuthenticatedRolloutTurn:
    """One accepted action sample and the exact event it caused."""

    turn_index: int
    dialogue_digest: str
    sample: AuthenticatedActionSample
    nominal_verification_digest: str
    event: TestEvent | ReadyEvent | AnswerEvent

    def __post_init__(self) -> None:
        _require_integer(
            self.turn_index,
            name="turn_index",
            maximum=MAXIMUM_TURNS_PER_ROLLOUT - 1,
        )
        _require_sha256(self.dialogue_digest, name="dialogue_digest")
        if type(self.sample) is not AuthenticatedActionSample:
            raise AuthenticatedRolloutError("rollout turn requires an AuthenticatedActionSample")
        _require_sha256(
            self.nominal_verification_digest,
            name="nominal_verification_digest",
        )
        if type(self.event) not in {TestEvent, ReadyEvent, AnswerEvent}:
            raise AuthenticatedRolloutError("authenticated turn requires a canonical action event")
        if self.sample.turn_seed.turn_index != self.turn_index:
            raise AuthenticatedRolloutError("turn index differs from the sample turn seed")
        raw_action = _event_raw_action(self.event)
        if raw_action != self.sample.decision_example.action_trace.raw_action:
            raise AuthenticatedRolloutError("sampled action differs from the recorded event")
        if type(self.event) in {TestEvent, ReadyEvent} and self.sample.mode != "inquiry":
            raise AuthenticatedRolloutError("inquiry event has a non-inquiry sample")
        if type(self.event) is AnswerEvent and self.sample.mode != "answer":
            raise AuthenticatedRolloutError("answer event has a non-answer sample")

    def as_obj(self) -> dict[str, object]:
        return {
            "turn_index": self.turn_index,
            "dialogue_digest": self.dialogue_digest,
            "sample": {**self.sample.as_obj(), "digest": self.sample.digest},
            "nominal_verification_digest": self.nominal_verification_digest,
            "event": self.event.as_obj(),
        }

    @property
    def digest(self) -> str:
        return json_digest(self.as_obj(), domain=_ROLLOUT_TURN_DIGEST_DOMAIN)

    @classmethod
    def from_obj(cls, value: object) -> AuthenticatedRolloutTurn:
        obj = _require_exact_object(
            value,
            (
                "turn_index",
                "dialogue_digest",
                "sample",
                "nominal_verification_digest",
                "event",
                "digest",
            ),
            name="rollout_turn",
        )
        try:
            sample = AuthenticatedActionSample.from_obj(obj["sample"])
            event = event_from_obj(obj["event"])
        except (AuthenticatedSamplingError, TypeError, ValueError) as exc:
            raise AuthenticatedRolloutError("rollout turn contains invalid nested evidence") from exc
        if type(event) not in {TestEvent, ReadyEvent, AnswerEvent}:
            raise AuthenticatedRolloutError("rollout turn event is not a policy action event")
        result = cls(
            turn_index=_require_integer(
                obj["turn_index"],
                name="turn_index",
                maximum=MAXIMUM_TURNS_PER_ROLLOUT - 1,
            ),
            dialogue_digest=_require_sha256(obj["dialogue_digest"], name="dialogue_digest"),
            sample=sample,
            nominal_verification_digest=_require_sha256(
                obj["nominal_verification_digest"],
                name="nominal_verification_digest",
            ),
            event=cast(TestEvent | ReadyEvent | AnswerEvent, event),
        )
        supplied = _require_sha256(obj["digest"], name="rollout_turn.digest")
        if not hmac.compare_digest(supplied, result.digest):
            raise AuthenticatedRolloutError("rollout-turn digest check failed")
        return result


@dataclass(frozen=True, slots=True)
class AuthenticatedRolloutRecord:
    """Canonical structural record; live replay is still required for acceptance."""

    episode_id: str
    episode_digest: str
    run_seed: int
    rollout_index: int
    model_provenance: ModelPolicyProvenance
    model_provenance_digest: str
    policy_state_digest: str
    tokenizer_binding_digest: str
    compiler_manifest_digest: str
    temperature_hex: str
    absolute_tolerance_hex: str
    maximum_sequence_tokens: int
    maximum_turns: int
    turns: tuple[AuthenticatedRolloutTurn, ...]
    termination: RolloutTermination | None
    transcript: Transcript
    transcript_digest: str
    reward_numerator: int
    reward_denominator: int
    schema_version: int = AUTHENTICATED_ROLLOUT_SCHEMA_VERSION
    contract_id: str = AUTHENTICATED_ROLLOUT_CONTRACT_ID

    def __post_init__(self) -> None:
        if self.schema_version != AUTHENTICATED_ROLLOUT_SCHEMA_VERSION:
            raise AuthenticatedRolloutError("unexpected authenticated-rollout schema version")
        if self.contract_id != AUTHENTICATED_ROLLOUT_CONTRACT_ID:
            raise AuthenticatedRolloutError("unexpected authenticated-rollout contract id")
        _require_text(self.episode_id, name="episode_id")
        _require_sha256(self.episode_digest, name="episode_digest")
        _require_integer(self.run_seed, name="run_seed", maximum=2**63 - 1)
        _require_integer(
            self.rollout_index,
            name="rollout_index",
            maximum=ROLLOUTS_PER_EPISODE - 1,
        )
        if type(self.model_provenance) is not ModelPolicyProvenance:
            raise AuthenticatedRolloutError("model_provenance must be typed")
        for name in (
            "model_provenance_digest",
            "policy_state_digest",
            "tokenizer_binding_digest",
            "compiler_manifest_digest",
            "transcript_digest",
        ):
            _require_sha256(getattr(self, name), name=name)
        if self.model_provenance.digest != self.model_provenance_digest:
            raise AuthenticatedRolloutError("model-provenance digest is inconsistent")
        if self.model_provenance.policy_state_digest != self.policy_state_digest:
            raise AuthenticatedRolloutError("policy state differs from model provenance")
        _canonical_positive_float_hex(self.temperature_hex, name="temperature_hex")
        _canonical_nonnegative_float_hex(
            self.absolute_tolerance_hex,
            name="absolute_tolerance_hex",
        )
        _require_integer(
            self.maximum_sequence_tokens,
            name="maximum_sequence_tokens",
            minimum=1,
        )
        _require_integer(
            self.maximum_turns,
            name="maximum_turns",
            minimum=1,
            maximum=MAXIMUM_TURNS_PER_ROLLOUT,
        )
        turns = tuple(self.turns)
        object.__setattr__(self, "turns", turns)
        if len(turns) > self.maximum_turns or any(
            type(turn) is not AuthenticatedRolloutTurn for turn in turns
        ):
            raise AuthenticatedRolloutError("rollout turns violate the registered turn budget")
        if tuple(turn.turn_index for turn in turns) != tuple(range(len(turns))):
            raise AuthenticatedRolloutError("rollout turn indices must be exact and contiguous")
        for turn in turns:
            sample = turn.sample
            seed = sample.turn_seed
            if (
                seed.run_seed,
                seed.episode_digest,
                seed.rollout_index,
                seed.turn_index,
                seed.policy_state_digest,
            ) != (
                self.run_seed,
                self.episode_digest,
                self.rollout_index,
                turn.turn_index,
                self.policy_state_digest,
            ):
                raise AuthenticatedRolloutError("rollout sample seed coordinates do not match")
            if sample.tokenizer_binding_digest != self.tokenizer_binding_digest:
                raise AuthenticatedRolloutError("rollout sample changed tokenizer binding")
            if sample.compiler_manifest_digest != self.compiler_manifest_digest:
                raise AuthenticatedRolloutError("rollout sample changed compiler manifest")
            if sample.temperature_hex != self.temperature_hex:
                raise AuthenticatedRolloutError("rollout sample changed behavior temperature")
            if sample.decision_example.maximum_sequence_tokens != self.maximum_sequence_tokens:
                raise AuthenticatedRolloutError("rollout sample changed maximum sequence length")

        if self.termination is not None and type(self.termination) is not RolloutTermination:
            raise AuthenticatedRolloutError("termination must be a RolloutTermination or None")
        expected_events: tuple[TranscriptEvent, ...] = tuple(turn.event for turn in turns)
        if self.termination is None:
            if self.transcript.state != "complete":
                raise AuthenticatedRolloutError("unterminated authenticated rollout must be complete")
        else:
            if self.termination.turn_index != len(turns):
                raise AuthenticatedRolloutError("termination must address the next unsampled turn")
            if self.termination.origin == "turn_budget" and len(turns) != self.maximum_turns:
                raise AuthenticatedRolloutError("turn-budget termination occurred before the turn limit")
            if self.termination.origin != "turn_budget" and len(turns) >= self.maximum_turns:
                raise AuthenticatedRolloutError(
                    "controller or sampler termination occurred outside the turn loop"
                )
            expected_events = (*expected_events, AbortEvent(self.termination.reason))
            if self.transcript.state != "aborted":
                raise AuthenticatedRolloutError("terminated authenticated rollout must be aborted")
        if type(self.transcript) is not Transcript or self.transcript.events != expected_events:
            raise AuthenticatedRolloutError("rollout turns/termination differ from transcript events")
        if self.transcript.episode_digest != self.episode_digest:
            raise AuthenticatedRolloutError("rollout transcript belongs to another episode")
        if self.transcript.digest != self.transcript_digest:
            raise AuthenticatedRolloutError("rollout transcript digest is inconsistent")

        if (
            isinstance(self.reward_numerator, bool)
            or not isinstance(self.reward_numerator, int)
            or isinstance(self.reward_denominator, bool)
            or not isinstance(self.reward_denominator, int)
            or self.reward_denominator < 1
            or not 0 <= self.reward_numerator <= self.reward_denominator
        ):
            raise AuthenticatedRolloutError("rollout reward fraction is invalid")
        expected_reward = _terminal_reward_fraction(self.transcript)
        if (self.reward_numerator, self.reward_denominator) != (
            expected_reward.numerator,
            expected_reward.denominator,
        ):
            raise AuthenticatedRolloutError("rollout reward does not derive from its transcript")
        if self.termination is not None and expected_reward != 0:
            raise AuthenticatedRolloutError("an aborted rollout must have exactly zero reward")

    @property
    def temperature(self) -> float:
        return float.fromhex(self.temperature_hex)

    @property
    def absolute_tolerance(self) -> float:
        return float.fromhex(self.absolute_tolerance_hex)

    @property
    def reward(self) -> float:
        return self.reward_numerator / self.reward_denominator

    def as_obj(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "episode_id": self.episode_id,
            "episode_digest": self.episode_digest,
            "run_seed": self.run_seed,
            "rollout_index": self.rollout_index,
            "model_provenance": {
                **self.model_provenance.as_obj(),
                "digest": self.model_provenance.digest,
            },
            "model_provenance_digest": self.model_provenance_digest,
            "policy_state_digest": self.policy_state_digest,
            "tokenizer_binding_digest": self.tokenizer_binding_digest,
            "compiler_manifest_digest": self.compiler_manifest_digest,
            "temperature_hex": self.temperature_hex,
            "absolute_tolerance_hex": self.absolute_tolerance_hex,
            "maximum_sequence_tokens": self.maximum_sequence_tokens,
            "maximum_turns": self.maximum_turns,
            "turns": [{**turn.as_obj(), "digest": turn.digest} for turn in self.turns],
            "termination": None if self.termination is None else self.termination.as_obj(),
            "transcript": self.transcript.as_obj(),
            "transcript_digest": self.transcript_digest,
            "reward": {
                "numerator": self.reward_numerator,
                "denominator": self.reward_denominator,
            },
            "turn_count": len(self.turns),
            "authorization": {
                "live_model": False,
                "rollout_launch": False,
                "weight_update": False,
                "optimizer_step_hook_present": False,
            },
        }

    @property
    def digest(self) -> str:
        return json_digest(self.as_obj(), domain=_ROLLOUT_DIGEST_DOMAIN)

    def to_json(self) -> str:
        return dump_json({**self.as_obj(), "digest": self.digest})

    @classmethod
    def from_obj(cls, value: object) -> AuthenticatedRolloutRecord:
        obj = _require_exact_object(
            value,
            (
                "schema_version",
                "contract_id",
                "episode_id",
                "episode_digest",
                "run_seed",
                "rollout_index",
                "model_provenance",
                "model_provenance_digest",
                "policy_state_digest",
                "tokenizer_binding_digest",
                "compiler_manifest_digest",
                "temperature_hex",
                "absolute_tolerance_hex",
                "maximum_sequence_tokens",
                "maximum_turns",
                "turns",
                "termination",
                "transcript",
                "transcript_digest",
                "reward",
                "turn_count",
                "authorization",
                "digest",
            ),
            name="authenticated_rollout",
        )
        reward = _require_exact_object(
            obj["reward"],
            ("numerator", "denominator"),
            name="reward",
        )
        authorization = _require_exact_object(
            obj["authorization"],
            (
                "live_model",
                "rollout_launch",
                "weight_update",
                "optimizer_step_hook_present",
            ),
            name="authorization",
        )
        if authorization != {
            "live_model": False,
            "rollout_launch": False,
            "weight_update": False,
            "optimizer_step_hook_present": False,
        }:
            raise AuthenticatedRolloutError("authenticated rollout cannot carry authorization")
        raw_turns = _require_list(obj["turns"], name="turns")
        turns = tuple(AuthenticatedRolloutTurn.from_obj(item) for item in raw_turns)
        if _require_integer(obj["turn_count"], name="turn_count") != len(turns):
            raise AuthenticatedRolloutError("turn_count is inconsistent")
        try:
            transcript = parse_transcript(dump_json(obj["transcript"]))
        except (CanonicalJSONError, TypeError, ValueError) as exc:
            raise AuthenticatedRolloutError("authenticated rollout transcript is invalid") from exc
        termination = None if obj["termination"] is None else RolloutTermination.from_obj(obj["termination"])
        result = cls(
            episode_id=_require_text(obj["episode_id"], name="episode_id"),
            episode_digest=_require_sha256(obj["episode_digest"], name="episode_digest"),
            run_seed=_require_integer(obj["run_seed"], name="run_seed", maximum=2**63 - 1),
            rollout_index=_require_integer(
                obj["rollout_index"],
                name="rollout_index",
                maximum=ROLLOUTS_PER_EPISODE - 1,
            ),
            model_provenance=ModelPolicyProvenance.from_obj(obj["model_provenance"]),
            model_provenance_digest=_require_sha256(
                obj["model_provenance_digest"], name="model_provenance_digest"
            ),
            policy_state_digest=_require_sha256(obj["policy_state_digest"], name="policy_state_digest"),
            tokenizer_binding_digest=_require_sha256(
                obj["tokenizer_binding_digest"], name="tokenizer_binding_digest"
            ),
            compiler_manifest_digest=_require_sha256(
                obj["compiler_manifest_digest"], name="compiler_manifest_digest"
            ),
            temperature_hex=_canonical_positive_float_hex(obj["temperature_hex"], name="temperature_hex"),
            absolute_tolerance_hex=_canonical_nonnegative_float_hex(
                obj["absolute_tolerance_hex"], name="absolute_tolerance_hex"
            ),
            maximum_sequence_tokens=_require_integer(
                obj["maximum_sequence_tokens"], name="maximum_sequence_tokens", minimum=1
            ),
            maximum_turns=_require_integer(
                obj["maximum_turns"],
                name="maximum_turns",
                minimum=1,
                maximum=MAXIMUM_TURNS_PER_ROLLOUT,
            ),
            turns=turns,
            termination=termination,
            transcript=transcript,
            transcript_digest=_require_sha256(obj["transcript_digest"], name="transcript_digest"),
            reward_numerator=_require_integer(reward["numerator"], name="reward.numerator"),
            reward_denominator=_require_integer(reward["denominator"], name="reward.denominator", minimum=1),
            schema_version=_require_integer(obj["schema_version"], name="schema_version", minimum=1),
            contract_id=_require_text(obj["contract_id"], name="contract_id"),
        )
        supplied = _require_sha256(obj["digest"], name="authenticated_rollout.digest")
        if not hmac.compare_digest(supplied, result.digest):
            raise AuthenticatedRolloutError("authenticated-rollout digest check failed")
        return result

    @classmethod
    def from_json(cls, text: str) -> AuthenticatedRolloutRecord:
        if type(text) is not str:
            raise AuthenticatedRolloutError("authenticated rollout JSON must be text")
        try:
            value = load_json(text)
        except CanonicalJSONError as exc:
            raise AuthenticatedRolloutError("authenticated rollout is not strict JSON") from exc
        result = cls.from_obj(value)
        if not hmac.compare_digest(text.encode("utf-8"), result.to_json().encode("utf-8")):
            raise AuthenticatedRolloutError(
                "authenticated rollout JSON is not in the unique canonical byte representation"
            )
        return result


@dataclass(frozen=True, slots=True, init=False)
class VerifiedAuthenticatedRollout:
    """Nominal graph-free evidence that every recorded action passed live replay."""

    record: AuthenticatedRolloutRecord
    verified_samples: tuple[GraphFreeVerifiedAuthenticatedActionSample, ...]
    verification_digest: str

    @classmethod
    def _from_verified(
        cls,
        record: AuthenticatedRolloutRecord,
        verified_samples: tuple[GraphFreeVerifiedAuthenticatedActionSample, ...],
    ) -> VerifiedAuthenticatedRollout:
        if len(verified_samples) != len(record.turns) or any(
            type(item) is not GraphFreeVerifiedAuthenticatedActionSample for item in verified_samples
        ):
            raise AuthenticatedRolloutError(
                "graph-free verified action evidence does not align with rollout turns"
            )
        for turn, verified in zip(record.turns, verified_samples, strict=True):
            if verified.sample != turn.sample:
                raise AuthenticatedRolloutError("verified sample belongs to another rollout turn")
            if verified.verification_digest != turn.nominal_verification_digest:
                raise AuthenticatedRolloutError("nominal action verification digest differs")
            if verified.absolute_tolerance_hex != record.absolute_tolerance_hex:
                raise AuthenticatedRolloutError("verified sample changed replay tolerance")
        result = object.__new__(cls)
        object.__setattr__(result, "record", record)
        object.__setattr__(result, "verified_samples", tuple(verified_samples))
        object.__setattr__(
            result,
            "verification_digest",
            json_digest(
                {
                    "record_digest": record.digest,
                    "verified_samples": [
                        {
                            "sample_digest": verified.sample.digest,
                            "verification_digest": verified.verification_digest,
                        }
                        for verified in verified_samples
                    ],
                },
                domain=_ROLLOUT_VERIFICATION_DIGEST_DOMAIN,
            ),
        )
        return result

    @property
    def digest(self) -> str:
        return self.record.digest


def _compact_verified_action_evidence(
    verified: VerifiedAuthenticatedActionSample,
) -> tuple[
    GraphFreeVerifiedAuthenticatedActionSample,
    tuple[weakref.ReferenceType[torch.Tensor], ...],
]:
    """Copy nominal evidence and weakly observe every differentiable replay root."""

    if type(verified) is not VerifiedAuthenticatedActionSample:
        raise AuthenticatedRolloutError("live replay returned invalid verified action evidence")
    statistics = verified.replayed_statistics
    roots = (
        statistics.token_log_probabilities,
        statistics.token_entropies,
        statistics.sequence_log_probability,
        statistics.mean_token_entropy,
    )
    if any(
        not isinstance(root, torch.Tensor) or not root.requires_grad or root.grad_fn is None for root in roots
    ):
        raise AuthenticatedRolloutError("live replay omitted a differentiable statistics root")
    try:
        evidence = GraphFreeVerifiedAuthenticatedActionSample._from_verified(verified)
        references = (
            verified.full_forward_logits_reference,
            *(weakref.ref(root) for root in roots),
        )
    except (AuthenticatedSamplingError, TypeError) as exc:
        raise AuthenticatedRolloutError("live replay could not be compacted safely") from exc
    return evidence, references


def _require_replay_roots_released(
    references: tuple[weakref.ReferenceType[torch.Tensor], ...],
) -> None:
    """Fail closed unless graph-bearing replay roots died before retention."""

    gc.collect()
    if any(reference() is not None for reference in references):
        raise AuthenticatedRolloutError(
            "differentiable replay graph remained live after action-evidence compaction"
        )


def _abort(
    environment: HiddenLawEnvironment,
    dialogue: Dialogue,
    *,
    origin: TerminationOrigin,
    reason: AbortReason,
    turn_index: int,
) -> RolloutTermination:
    termination = RolloutTermination(
        origin=origin,
        reason=reason,
        turn_index=turn_index,
        dialogue_digest=_dialogue_digest(dialogue),
    )
    result = environment.abort(reason)
    if result.outcome != "aborted" or type(result.event) is not AbortEvent:
        raise AuthenticatedRolloutError("controller abort did not produce the registered abort event")
    return termination


def _poll_controller(
    controller: RolloutControllerProtocol | None,
    *,
    episode_digest: str,
    rollout_index: int,
    turn_index: int,
    dialogue_digest: str,
) -> AbortReason | None:
    if controller is None:
        return None
    if not isinstance(controller, RolloutControllerProtocol):
        raise TypeError("controller must implement RolloutControllerProtocol")
    try:
        reason = controller.abort_reason(
            episode_digest=episode_digest,
            rollout_index=rollout_index,
            turn_index=turn_index,
            dialogue_digest=dialogue_digest,
        )
    except TimeoutError:
        return "timed_out"
    except Exception as exc:
        raise AuthenticatedRolloutError("rollout controller failed") from exc
    if reason is not None and reason not in {"incomplete", "timed_out"}:
        raise AuthenticatedRolloutError("rollout controller returned an unregistered abort reason")
    return reason


def _require_collection_inputs(
    provider: ProvenancedFullForwardLogitsProvider,
    compiler: FragmentActionTokenCompiler,
    episode: HiddenEpisode,
    model_provenance: ModelPolicyProvenance,
    *,
    run_seed: int,
    rollout_index: int,
    temperature: float,
    absolute_tolerance: float,
    maximum_sequence_tokens: int,
    maximum_turns: int,
) -> _ProvenanceGuardProvider:
    if not isinstance(provider, ProvenancedFullForwardLogitsProvider):
        raise TypeError("provider must implement ProvenancedFullForwardLogitsProvider")
    if type(compiler) is not FragmentActionTokenCompiler:
        raise TypeError("compiler must be a FragmentActionTokenCompiler")
    if type(episode) is not HiddenEpisode:
        raise TypeError("episode must be a HiddenEpisode")
    if type(model_provenance) is not ModelPolicyProvenance:
        raise TypeError("model_provenance must be a ModelPolicyProvenance")
    _require_integer(run_seed, name="run_seed", maximum=2**63 - 1)
    _require_integer(
        rollout_index,
        name="rollout_index",
        maximum=ROLLOUTS_PER_EPISODE - 1,
    )
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature))
        or float(temperature) <= 0
    ):
        raise AuthenticatedRolloutError("temperature must be finite and positive")
    if (
        isinstance(absolute_tolerance, bool)
        or not isinstance(absolute_tolerance, (int, float))
        or not math.isfinite(float(absolute_tolerance))
        or float(absolute_tolerance) < 0
    ):
        raise AuthenticatedRolloutError("absolute_tolerance must be finite and non-negative")
    _require_integer(maximum_sequence_tokens, name="maximum_sequence_tokens", minimum=1)
    _require_integer(
        maximum_turns,
        name="maximum_turns",
        minimum=1,
        maximum=MAXIMUM_TURNS_PER_ROLLOUT,
    )
    try:
        _ = compiler.manifest
    except FragmentActionTokenizationError as exc:
        raise AuthenticatedRolloutError("authenticated rollout requires a frozen compiler") from exc
    return _ProvenanceGuardProvider(provider, model_provenance)


def collect_authenticated_rollout(
    provider: ProvenancedFullForwardLogitsProvider,
    tokenizer: ExactDecodeTokenizerProtocol,
    compiler: FragmentActionTokenCompiler,
    episode: HiddenEpisode,
    model_provenance: ModelPolicyProvenance,
    *,
    run_seed: int,
    rollout_index: int,
    temperature: float,
    absolute_tolerance: float,
    maximum_sequence_tokens: int,
    maximum_turns: int = MAXIMUM_TURNS_PER_ROLLOUT,
    controller: RolloutControllerProtocol | None = None,
) -> VerifiedAuthenticatedRollout:
    """Collect one rollout without accepting any unverified policy turn."""

    guarded = _require_collection_inputs(
        provider,
        compiler,
        episode,
        model_provenance,
        run_seed=run_seed,
        rollout_index=rollout_index,
        temperature=temperature,
        absolute_tolerance=absolute_tolerance,
        maximum_sequence_tokens=maximum_sequence_tokens,
        maximum_turns=maximum_turns,
    )
    manifest = compiler.manifest
    environment = HiddenLawEnvironment(episode)
    turns: list[AuthenticatedRolloutTurn] = []
    verified_samples: list[GraphFreeVerifiedAuthenticatedActionSample] = []
    termination: RolloutTermination | None = None

    for turn_index in range(maximum_turns):
        if environment.state in {"complete", "invalid", "aborted"}:
            break
        dialogue = render_dialogue(episode, environment.transcript)
        dialogue_digest = _dialogue_digest(dialogue)
        reason = _poll_controller(
            controller,
            episode_digest=episode.digest,
            rollout_index=rollout_index,
            turn_index=turn_index,
            dialogue_digest=dialogue_digest,
        )
        if reason is not None:
            termination = _abort(
                environment,
                dialogue,
                origin="controller",
                reason=reason,
                turn_index=turn_index,
            )
            break

        mode: Literal["inquiry", "answer"] = "inquiry" if environment.state == "inquiry" else "answer"
        terminal_count = None if mode == "inquiry" else len(episode.terminal)
        try:
            turn_seed = PolicyTurnSeed(
                run_seed=run_seed,
                episode_digest=episode.digest,
                rollout_index=rollout_index,
                turn_index=turn_index,
                policy_state_digest=model_provenance.policy_state_digest,
            )
            sample = sample_authenticated_action(
                guarded,
                tokenizer,
                compiler,
                dialogue,
                turn_seed,
                mode=mode,
                terminal_count=terminal_count,
                temperature=float(temperature),
                maximum_sequence_tokens=maximum_sequence_tokens,
            )
        except AuthenticatedSamplingOverlengthError:
            termination = _abort(
                environment,
                dialogue,
                origin="sampler",
                reason="overlength",
                turn_index=turn_index,
            )
            break
        except (AuthenticatedSamplingError, PolicyRandomnessError) as exc:
            raise AuthenticatedRolloutError(
                "fatal authenticated sampling invariant; no rollout record emitted"
            ) from exc
        try:
            verified = replay_authenticated_sample(
                sample,
                guarded,
                tokenizer,
                compiler,
                dialogue,
                absolute_tolerance=float(absolute_tolerance),
            )
        except AuthenticatedSamplingError as exc:
            raise AuthenticatedRolloutError(
                "fatal authenticated replay invariant; no rollout record emitted"
            ) from exc
        evidence, replay_root_references = _compact_verified_action_evidence(verified)
        verification_digest = verified.verification_digest
        del verified
        _require_replay_roots_released(replay_root_references)

        raw_action = sample.decision_example.action_trace.raw_action
        step = environment.consume(raw_action)
        if step.outcome in {"invalid", "already_terminal", "aborted"} or step.event is None:
            raise AuthenticatedRolloutError(
                "verified canonical action did not produce a valid environment event"
            )
        if type(step.event) not in {TestEvent, ReadyEvent, AnswerEvent}:
            raise AuthenticatedRolloutError("verified policy action produced an unsupported event")
        turn = AuthenticatedRolloutTurn(
            turn_index=turn_index,
            dialogue_digest=dialogue_digest,
            sample=sample,
            nominal_verification_digest=verification_digest,
            event=cast(TestEvent | ReadyEvent | AnswerEvent, step.event),
        )
        turns.append(turn)
        verified_samples.append(evidence)
    else:
        if environment.state not in {"complete", "invalid", "aborted"}:
            dialogue = render_dialogue(episode, environment.transcript)
            termination = _abort(
                environment,
                dialogue,
                origin="turn_budget",
                reason="incomplete",
                turn_index=maximum_turns,
            )

    if environment.state not in {"complete", "aborted"}:
        raise AuthenticatedRolloutError("collector ended in a non-recordable environment state")
    try:
        replay_transcript(episode, environment.transcript)
    except TranscriptReplayError as exc:
        raise AuthenticatedRolloutError("collected transcript failed exact environment replay") from exc
    guarded._check()
    reward = _terminal_reward_fraction(environment.transcript)
    record = AuthenticatedRolloutRecord(
        episode_id=episode.episode_id,
        episode_digest=episode.digest,
        run_seed=run_seed,
        rollout_index=rollout_index,
        model_provenance=model_provenance,
        model_provenance_digest=model_provenance.digest,
        policy_state_digest=model_provenance.policy_state_digest,
        tokenizer_binding_digest=manifest.tokenizer_identifier,
        compiler_manifest_digest=manifest.digest,
        temperature_hex=float(temperature).hex(),
        absolute_tolerance_hex=float(absolute_tolerance).hex(),
        maximum_sequence_tokens=maximum_sequence_tokens,
        maximum_turns=maximum_turns,
        turns=tuple(turns),
        termination=termination,
        transcript=environment.transcript,
        transcript_digest=environment.transcript.digest,
        reward_numerator=reward.numerator,
        reward_denominator=reward.denominator,
    )
    return VerifiedAuthenticatedRollout._from_verified(record, tuple(verified_samples))


def replay_authenticated_rollout(
    record: AuthenticatedRolloutRecord,
    provider: ProvenancedFullForwardLogitsProvider,
    tokenizer: ExactDecodeTokenizerProtocol,
    compiler: FragmentActionTokenCompiler,
    episode: HiddenEpisode,
    *,
    controller: RolloutControllerProtocol | None = None,
) -> VerifiedAuthenticatedRollout:
    """Rebuild every public prompt, authenticated sample, event, and reward."""

    if type(record) is not AuthenticatedRolloutRecord:
        raise TypeError("record must be an AuthenticatedRolloutRecord")
    guarded = _require_collection_inputs(
        provider,
        compiler,
        episode,
        record.model_provenance,
        run_seed=record.run_seed,
        rollout_index=record.rollout_index,
        temperature=record.temperature,
        absolute_tolerance=record.absolute_tolerance,
        maximum_sequence_tokens=record.maximum_sequence_tokens,
        maximum_turns=record.maximum_turns,
    )
    if (record.episode_id, record.episode_digest) != (episode.episode_id, episode.digest):
        raise AuthenticatedRolloutError("rollout record is bound to another hidden episode")
    manifest = compiler.manifest
    if manifest.tokenizer_identifier != record.tokenizer_binding_digest:
        raise AuthenticatedRolloutError("live tokenizer differs from rollout binding")
    if manifest.digest != record.compiler_manifest_digest:
        raise AuthenticatedRolloutError("live compiler differs from rollout binding")

    environment = HiddenLawEnvironment(episode)
    verified_samples: list[GraphFreeVerifiedAuthenticatedActionSample] = []
    for expected_turn_index, turn in enumerate(record.turns):
        if turn.turn_index != expected_turn_index:
            raise AuthenticatedRolloutError("rollout replay found a noncontiguous turn index")
        dialogue = render_dialogue(episode, environment.transcript)
        if _dialogue_digest(dialogue) != turn.dialogue_digest:
            raise AuthenticatedRolloutError("rollout dialogue does not regenerate exactly")
        if (
            _poll_controller(
                controller,
                episode_digest=episode.digest,
                rollout_index=record.rollout_index,
                turn_index=expected_turn_index,
                dialogue_digest=turn.dialogue_digest,
            )
            is not None
        ):
            raise AuthenticatedRolloutError("controller requested an abort before a recorded policy action")
        try:
            verified = replay_authenticated_sample(
                turn.sample,
                guarded,
                tokenizer,
                compiler,
                dialogue,
                absolute_tolerance=record.absolute_tolerance,
            )
        except AuthenticatedSamplingError as exc:
            raise AuthenticatedRolloutError("rollout action failed live authenticated replay") from exc
        evidence, replay_root_references = _compact_verified_action_evidence(verified)
        verification_digest = verified.verification_digest
        del verified
        _require_replay_roots_released(replay_root_references)
        if verification_digest != turn.nominal_verification_digest:
            raise AuthenticatedRolloutError("rollout nominal verification digest does not regenerate")
        result = environment.consume(turn.sample.decision_example.action_trace.raw_action)
        if result.event != turn.event:
            raise AuthenticatedRolloutError("rollout event differs under exact environment replay")
        verified_samples.append(evidence)

    if record.termination is not None:
        dialogue = render_dialogue(episode, environment.transcript)
        if _dialogue_digest(dialogue) != record.termination.dialogue_digest:
            raise AuthenticatedRolloutError("rollout termination dialogue does not regenerate")
        controller_reason = _poll_controller(
            controller,
            episode_digest=episode.digest,
            rollout_index=record.rollout_index,
            turn_index=record.termination.turn_index,
            dialogue_digest=record.termination.dialogue_digest,
        )
        if record.termination.origin == "controller":
            if controller is None:
                raise AuthenticatedRolloutError("controller-origin termination requires its live controller")
            if controller_reason != record.termination.reason:
                raise AuthenticatedRolloutError("controller termination reason does not regenerate exactly")
        elif controller_reason is not None:
            raise AuthenticatedRolloutError("controller requested an abort omitted by the rollout record")
        if record.termination.origin == "sampler":
            mode: Literal["inquiry", "answer"] = "inquiry" if environment.state == "inquiry" else "answer"
            terminal_count = None if mode == "inquiry" else len(episode.terminal)
            turn_seed = PolicyTurnSeed(
                run_seed=record.run_seed,
                episode_digest=record.episode_digest,
                rollout_index=record.rollout_index,
                turn_index=record.termination.turn_index,
                policy_state_digest=record.policy_state_digest,
            )
            try:
                sample_authenticated_action(
                    guarded,
                    tokenizer,
                    compiler,
                    dialogue,
                    turn_seed,
                    mode=mode,
                    terminal_count=terminal_count,
                    temperature=record.temperature,
                    maximum_sequence_tokens=record.maximum_sequence_tokens,
                )
            except AuthenticatedSamplingOverlengthError:
                pass
            except AuthenticatedSamplingError as exc:
                raise AuthenticatedRolloutError(
                    "sampler termination reproduced as a generic fatal failure"
                ) from exc
            else:
                raise AuthenticatedRolloutError(
                    "recorded sampler overlength did not reproduce at the termination coordinate"
                )
        result = environment.abort(record.termination.reason)
        expected_abort = AbortEvent(record.termination.reason)
        if result.event != expected_abort:
            raise AuthenticatedRolloutError("rollout abort event differs under exact replay")
    if environment.transcript != record.transcript:
        raise AuthenticatedRolloutError("rollout transcript differs under live replay")
    reward = _terminal_reward_fraction(environment.transcript)
    if (reward.numerator, reward.denominator) != (
        record.reward_numerator,
        record.reward_denominator,
    ):
        raise AuthenticatedRolloutError("rollout reward differs under live replay")
    guarded._check()
    return VerifiedAuthenticatedRollout._from_verified(record, tuple(verified_samples))


def parse_authenticated_rollout(text: str) -> AuthenticatedRolloutRecord:
    """Parse only the unique canonical schema-v2 byte representation."""

    return AuthenticatedRolloutRecord.from_json(text)


@dataclass(frozen=True, slots=True, init=False)
class VerifiedAuthenticatedRolloutGroup:
    """Eight complete rollout records accepted under one unchanged policy."""

    rollouts: tuple[VerifiedAuthenticatedRollout, ...]
    action_evidence_group_digest: str
    digest: str

    @classmethod
    def _from_verified(
        cls,
        rollouts: Sequence[VerifiedAuthenticatedRollout],
    ) -> VerifiedAuthenticatedRolloutGroup:
        selected = tuple(rollouts)
        if len(selected) != ROLLOUTS_PER_EPISODE or any(
            type(rollout) is not VerifiedAuthenticatedRollout for rollout in selected
        ):
            raise AuthenticatedRolloutError("group requires eight replay-verified rollouts")
        indices = {rollout.record.rollout_index for rollout in selected}
        if indices != set(range(ROLLOUTS_PER_EPISODE)):
            raise AuthenticatedRolloutError("group must contain rollout indices zero through seven")
        reference = selected[0].record
        answer_terminal_counts: set[int | None] = set()
        for rollout in selected:
            record = rollout.record
            for field in (
                "episode_digest",
                "run_seed",
                "model_provenance_digest",
                "policy_state_digest",
                "tokenizer_binding_digest",
                "compiler_manifest_digest",
                "temperature_hex",
                "absolute_tolerance_hex",
                "maximum_sequence_tokens",
                "maximum_turns",
            ):
                if getattr(record, field) != getattr(reference, field):
                    raise AuthenticatedRolloutError(f"eight-rollout collection changed invariant {field}")
            if any(
                type(sample) is not GraphFreeVerifiedAuthenticatedActionSample
                for sample in rollout.verified_samples
            ):
                raise AuthenticatedRolloutError(
                    "rollout group may use only GraphFreeVerifiedAuthenticatedActionSample evidence"
                )
            if len(rollout.verified_samples) != len(record.turns):
                raise AuthenticatedRolloutError("rollout action evidence does not align with recorded turns")
            if tuple(sample.sample.turn_seed.turn_index for sample in rollout.verified_samples) != tuple(
                range(len(rollout.verified_samples))
            ):
                raise AuthenticatedRolloutError("rollout action-evidence turn coordinates are not contiguous")
            for sample in rollout.verified_samples:
                if sample.sample.turn_seed.rollout_index != record.rollout_index:
                    raise AuthenticatedRolloutError(
                        "rollout action evidence belongs to another rollout coordinate"
                    )
                if sample.sample.mode == "answer":
                    answer_terminal_counts.add(sample.sample.terminal_count)
        if len(answer_terminal_counts) > 1:
            raise AuthenticatedRolloutError("eight-rollout collection changed answer terminal_count")
        ordered = tuple(sorted(selected, key=lambda item: item.record.rollout_index))
        flattened = tuple(sample for rollout in ordered for sample in rollout.verified_samples)
        empty_rollouts = tuple(rollout for rollout in ordered if not rollout.verified_samples)
        for rollout in empty_rollouts:
            record = rollout.record
            termination = record.termination
            if (
                record.turns
                or termination is None
                or termination.turn_index != 0
                or record.transcript.state != "aborted"
                or (record.reward_numerator, record.reward_denominator) != (0, 1)
            ):
                raise AuthenticatedRolloutError(
                    "empty action evidence is allowed only for a verified turn-zero abort"
                )
        sampler_group_digest: str | None = None
        if not empty_rollouts:
            try:
                sampler_group_digest = verify_eight_rollout_policy_group(flattened)
            except AuthenticatedSamplingError as exc:
                raise AuthenticatedRolloutError("authenticated action evidence failed group gate") from exc
        action_group_digest = json_digest(
            {
                "episode_digest": reference.episode_digest,
                "run_seed": reference.run_seed,
                "model_provenance_digest": reference.model_provenance_digest,
                "policy_state_digest": reference.policy_state_digest,
                "tokenizer_binding_digest": reference.tokenizer_binding_digest,
                "compiler_manifest_digest": reference.compiler_manifest_digest,
                "temperature_hex": reference.temperature_hex,
                "absolute_tolerance_hex": reference.absolute_tolerance_hex,
                "maximum_sequence_tokens": reference.maximum_sequence_tokens,
                "maximum_turns": reference.maximum_turns,
                "sampler_group_digest": sampler_group_digest,
                "rollouts": [
                    (
                        {
                            "kind": "verified_action_samples",
                            "rollout_index": rollout.record.rollout_index,
                            "samples": [
                                {
                                    "turn_index": sample.sample.turn_seed.turn_index,
                                    "sample_digest": sample.sample.digest,
                                    "verification_digest": sample.verification_digest,
                                }
                                for sample in rollout.verified_samples
                            ],
                        }
                        if rollout.verified_samples
                        else {
                            "kind": "verified_zero_turn_abort",
                            "rollout_index": rollout.record.rollout_index,
                            "record_digest": rollout.record.digest,
                            "rollout_verification_digest": rollout.verification_digest,
                            "termination": cast(
                                RolloutTermination,
                                rollout.record.termination,
                            ).as_obj(),
                            "reward": {"numerator": 0, "denominator": 1},
                        }
                    )
                    for rollout in ordered
                ],
            },
            domain=_ROLLOUT_ACTION_EVIDENCE_GROUP_DIGEST_DOMAIN,
        )
        result = object.__new__(cls)
        object.__setattr__(result, "rollouts", ordered)
        object.__setattr__(result, "action_evidence_group_digest", action_group_digest)
        object.__setattr__(
            result,
            "digest",
            json_digest(
                {
                    "action_evidence_group_digest": action_group_digest,
                    "rollouts": [
                        {
                            "rollout_index": rollout.record.rollout_index,
                            "record_digest": rollout.record.digest,
                            "verification_digest": rollout.verification_digest,
                        }
                        for rollout in ordered
                    ],
                },
                domain=_ROLLOUT_GROUP_DIGEST_DOMAIN,
            ),
        )
        return result


def verify_eight_authenticated_rollout_group(
    rollouts: Sequence[VerifiedAuthenticatedRollout],
) -> VerifiedAuthenticatedRolloutGroup:
    """Apply the unchanged-policy gate without accepting structural records."""

    return VerifiedAuthenticatedRolloutGroup._from_verified(rollouts)


def collect_eight_authenticated_rollouts(
    provider: ProvenancedFullForwardLogitsProvider,
    tokenizer: ExactDecodeTokenizerProtocol,
    compiler: FragmentActionTokenCompiler,
    episode: HiddenEpisode,
    model_provenance: ModelPolicyProvenance,
    *,
    run_seed: int,
    temperature: float,
    absolute_tolerance: float,
    maximum_sequence_tokens: int,
    maximum_turns: int = MAXIMUM_TURNS_PER_ROLLOUT,
    schedule_order: Sequence[int] = tuple(range(ROLLOUTS_PER_EPISODE)),
    controller: RolloutControllerProtocol | None = None,
) -> VerifiedAuthenticatedRolloutGroup:
    """Collect eight rollouts in any registered scheduling permutation.

    There is intentionally no optimizer or weight-update callback in this API.
    """

    order = tuple(schedule_order)
    if len(order) != ROLLOUTS_PER_EPISODE or any(
        isinstance(index, bool) or not isinstance(index, int) for index in order
    ):
        raise AuthenticatedRolloutError("schedule_order must contain eight integer indices")
    if set(order) != set(range(ROLLOUTS_PER_EPISODE)):
        raise AuthenticatedRolloutError("schedule_order must be a permutation of zero through seven")
    collected: list[VerifiedAuthenticatedRollout] = []
    for rollout_index in order:
        collected.append(
            collect_authenticated_rollout(
                provider,
                tokenizer,
                compiler,
                episode,
                model_provenance,
                run_seed=run_seed,
                rollout_index=rollout_index,
                temperature=temperature,
                absolute_tolerance=absolute_tolerance,
                maximum_sequence_tokens=maximum_sequence_tokens,
                maximum_turns=maximum_turns,
                controller=controller,
            )
        )
    return verify_eight_authenticated_rollout_group(collected)


def authenticated_rollout_manifest() -> dict[str, object]:
    """Describe this nonauthorizing local integration slice."""

    return {
        "schema_version": AUTHENTICATED_ROLLOUT_SCHEMA_VERSION,
        "contract_id": AUTHENTICATED_ROLLOUT_CONTRACT_ID,
        "rollouts_per_policy_group": ROLLOUTS_PER_EPISODE,
        "maximum_turns_per_rollout": MAXIMUM_TURNS_PER_ROLLOUT,
        "accepted_turn_evidence": "GraphFreeVerifiedAuthenticatedActionSample only",
        "live_replay_evidence": "ephemeral VerifiedAuthenticatedActionSample",
        "retained_differentiable_graphs": False,
        "replay_graph_release_check": "weak references dead before evidence append",
        "accepted_empty_evidence": "live-verified turn-zero registered abort only",
        "operational_abort_reasons": list(ABORT_REASONS),
        "fatal_failures": [
            "source_regeneration",
            "model_or_policy_provenance",
            "legal_mask",
            "nonfinite_logits_or_statistics",
            "stateless_resampling",
            "differentiable_full_forward_replay",
        ],
        "schedule_order_dependent": False,
        "optimizer_step_hook_present": False,
        "live_model_authorization": False,
        "rollout_launch_authorization": False,
        "weight_update_authorization": False,
    }
