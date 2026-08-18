"""Canonical bounded-memory plans for authenticated G03 outcome RL.

This module is a planning bridge, not an executor.  It accepts only nominal
``VerifiedAuthenticatedRolloutGroup`` values produced by live replay and
immediately reduces them to immutable scalar instructions and evidence
digests.  In particular, no tensor or collection-time autograd graph is
retained by the returned wrapper.

Parsed plans are structural records only.  A later executor must start from a
freshly derived nominal wrapper, reauthenticate every action under the frozen
provider, and accumulate the instructions one bounded graph at a time before
performing any optimizer step.
"""

from __future__ import annotations

import hmac
import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal, cast

from ._json import CanonicalJSONError, dump_json, json_digest, load_json
from .authenticated_rollouts_v2 import (
    AuthenticatedRolloutRecord,
    AuthenticatedRolloutTurn,
    ModelPolicyProvenance,
    VerifiedAuthenticatedRollout,
    VerifiedAuthenticatedRolloutGroup,
    verify_eight_authenticated_rollout_group,
)
from .authenticated_sampler_v2 import GraphFreeVerifiedAuthenticatedActionSample
from .objectives import trajectory_objective_digest
from .policy_randomness_v2 import MAXIMUM_TURNS_PER_ROLLOUT, ROLLOUTS_PER_EPISODE

STREAMING_OBJECTIVE_SCHEMA_VERSION = 3
STREAMING_OBJECTIVE_CONTRACT_ID = "goalzendo-streaming-objective-v3"

_PLAN_DIGEST_DOMAIN = "goalzendo-interactive-streaming-objective-plan-v3"
_VERIFIED_PLAN_DIGEST_DOMAIN = "goalzendo-interactive-verified-streaming-objective-plan-v3"
_TURN_ZERO_ABORT_DIGEST_DOMAIN = "goalzendo-interactive-streaming-turn-zero-abort-v3"

AbortOrigin = Literal["controller", "sampler", "turn_budget"]
AbortReason = Literal["incomplete", "overlength", "timed_out"]
_ABORT_ORIGINS: tuple[AbortOrigin, ...] = ("controller", "sampler", "turn_budget")
_ABORT_REASONS: tuple[AbortReason, ...] = ("incomplete", "overlength", "timed_out")


class StreamingObjectivePlanError(ValueError):
    """Raised when a streaming objective plan is malformed or unauthenticated."""


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: object, *, name: str) -> str:
    if not _is_sha256(value):
        raise StreamingObjectivePlanError(f"{name} must be a lowercase SHA-256")
    return cast(str, value)


def _require_text(value: object, *, name: str, ascii_only: bool = False) -> str:
    if type(value) is not str or not value:
        raise StreamingObjectivePlanError(f"{name} must be nonempty text")
    if ascii_only and not value.isascii():
        raise StreamingObjectivePlanError(f"{name} must contain only ASCII characters")
    return value


def _require_integer(
    value: object,
    *,
    name: str,
    minimum: int | None = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise StreamingObjectivePlanError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise StreamingObjectivePlanError(f"{name} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise StreamingObjectivePlanError(f"{name} must be an integer <= {maximum}")
    return value


def _require_boolean(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise StreamingObjectivePlanError(f"{name} must be Boolean")
    return value


def _require_exact_object(
    value: object,
    fields: tuple[str, ...],
    *,
    name: str,
) -> dict[str, object]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise StreamingObjectivePlanError(f"{name} must be a JSON object")
    result = cast(dict[str, object], value)
    if set(result) != set(fields) or len(result) != len(fields):
        raise StreamingObjectivePlanError(f"{name} has noncanonical fields")
    return result


def _require_list(value: object, *, name: str) -> list[object]:
    if type(value) is not list:
        raise StreamingObjectivePlanError(f"{name} must be a JSON array")
    return cast(list[object], value)


def _require_canonical_float_hex(
    value: object,
    *,
    name: str,
    positive: bool,
) -> str:
    if type(value) is not str:
        raise StreamingObjectivePlanError(f"{name} must be canonical float.hex text")
    try:
        parsed = float.fromhex(value)
    except ValueError as exc:
        raise StreamingObjectivePlanError(f"{name} is not valid float.hex text") from exc
    if not math.isfinite(parsed) or (parsed <= 0 if positive else parsed < 0) or parsed.hex() != value:
        qualifier = "positive" if positive else "non-negative"
        raise StreamingObjectivePlanError(f"{name} must encode a finite {qualifier} float canonically")
    return value


def _fraction_from_input(value: Fraction | int | float, *, name: str) -> Fraction:
    if isinstance(value, bool):
        raise StreamingObjectivePlanError(f"{name} must be an exact finite non-negative scalar")
    if type(value) is Fraction:
        selected = value
    elif type(value) is int:
        selected = Fraction(value, 1)
    elif type(value) is float:
        if not math.isfinite(value):
            raise StreamingObjectivePlanError(f"{name} must be finite")
        selected = Fraction(*value.as_integer_ratio())
    else:
        raise StreamingObjectivePlanError(f"{name} must be a Fraction, int, or finite float")
    if selected < 0:
        raise StreamingObjectivePlanError(f"{name} must be non-negative")
    return selected


def _execution_contract_obj() -> dict[str, object]:
    return {
        "reauthenticate_each_stored_action": True,
        "bounded_graph_scope": "one_authenticated_turn",
        "parameter_mutation_between_replays": False,
        "optimizer_state_mutation_between_replays": False,
        "optimizer_step_after_all_entries_only": True,
    }


@dataclass(frozen=True, slots=True)
class ExactScalar:
    """One reduced rational scalar with a unique JSON representation."""

    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if isinstance(self.numerator, bool) or not isinstance(self.numerator, int):
            raise StreamingObjectivePlanError("scalar numerator must be an integer")
        if (
            isinstance(self.denominator, bool)
            or not isinstance(self.denominator, int)
            or self.denominator < 1
        ):
            raise StreamingObjectivePlanError("scalar denominator must be a positive integer")
        reduced = Fraction(self.numerator, self.denominator)
        if (reduced.numerator, reduced.denominator) != (self.numerator, self.denominator):
            raise StreamingObjectivePlanError("scalar fraction must be reduced with a positive denominator")

    @classmethod
    def from_fraction(cls, value: Fraction) -> ExactScalar:
        if type(value) is not Fraction:
            raise TypeError("value must be a Fraction")
        return cls(value.numerator, value.denominator)

    @property
    def fraction(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)

    def as_obj(self) -> dict[str, int]:
        return {"numerator": self.numerator, "denominator": self.denominator}

    @classmethod
    def from_obj(cls, value: object, *, name: str) -> ExactScalar:
        obj = _require_exact_object(value, ("numerator", "denominator"), name=name)
        return cls(
            numerator=_require_integer(obj["numerator"], name=f"{name}.numerator", minimum=None),
            denominator=_require_integer(obj["denominator"], name=f"{name}.denominator", minimum=1),
        )


@dataclass(frozen=True, slots=True)
class StreamingUpdateInvariants:
    """Policy, artifact, and replay invariants shared by the entire update."""

    model_identifier: str
    model_revision: str
    artifact_manifest_sha256: str
    runtime_stack_sha256: str
    model_provenance_digest: str
    policy_state_digest: str
    tokenizer_binding_digest: str
    compiler_manifest_digest: str
    temperature_hex: str
    absolute_tolerance_hex: str
    maximum_sequence_tokens: int
    maximum_turns: int

    def __post_init__(self) -> None:
        _require_text(self.model_identifier, name="model_identifier", ascii_only=True)
        if any(character.isspace() for character in self.model_identifier):
            raise StreamingObjectivePlanError("model_identifier may not contain whitespace")
        if (
            type(self.model_revision) is not str
            or len(self.model_revision) not in {40, 64}
            or any(character not in "0123456789abcdef" for character in self.model_revision)
        ):
            raise StreamingObjectivePlanError(
                "model_revision must be an immutable lowercase 40- or 64-hex revision"
            )
        for name in (
            "artifact_manifest_sha256",
            "runtime_stack_sha256",
            "model_provenance_digest",
            "policy_state_digest",
            "tokenizer_binding_digest",
            "compiler_manifest_digest",
        ):
            _require_sha256(getattr(self, name), name=name)
        _require_canonical_float_hex(self.temperature_hex, name="temperature_hex", positive=True)
        _require_canonical_float_hex(
            self.absolute_tolerance_hex,
            name="absolute_tolerance_hex",
            positive=False,
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
        provenance = ModelPolicyProvenance(
            model_identifier=self.model_identifier,
            revision=self.model_revision,
            artifact_manifest_sha256=self.artifact_manifest_sha256,
            runtime_stack_sha256=self.runtime_stack_sha256,
            policy_state_digest=self.policy_state_digest,
        )
        if provenance.digest != self.model_provenance_digest:
            raise StreamingObjectivePlanError("model-provenance fields and digest disagree")

    def as_obj(self) -> dict[str, object]:
        return {
            "model_identifier": self.model_identifier,
            "model_revision": self.model_revision,
            "artifact_manifest_sha256": self.artifact_manifest_sha256,
            "runtime_stack_sha256": self.runtime_stack_sha256,
            "model_provenance_digest": self.model_provenance_digest,
            "policy_state_digest": self.policy_state_digest,
            "tokenizer_binding_digest": self.tokenizer_binding_digest,
            "compiler_manifest_digest": self.compiler_manifest_digest,
            "temperature_hex": self.temperature_hex,
            "absolute_tolerance_hex": self.absolute_tolerance_hex,
            "maximum_sequence_tokens": self.maximum_sequence_tokens,
            "maximum_turns": self.maximum_turns,
        }

    @classmethod
    def from_obj(cls, value: object) -> StreamingUpdateInvariants:
        fields = (
            "model_identifier",
            "model_revision",
            "artifact_manifest_sha256",
            "runtime_stack_sha256",
            "model_provenance_digest",
            "policy_state_digest",
            "tokenizer_binding_digest",
            "compiler_manifest_digest",
            "temperature_hex",
            "absolute_tolerance_hex",
            "maximum_sequence_tokens",
            "maximum_turns",
        )
        obj = _require_exact_object(value, fields, name="update_invariants")
        return cls(
            model_identifier=_require_text(obj["model_identifier"], name="model_identifier", ascii_only=True),
            model_revision=_require_text(obj["model_revision"], name="model_revision"),
            artifact_manifest_sha256=_require_sha256(
                obj["artifact_manifest_sha256"], name="artifact_manifest_sha256"
            ),
            runtime_stack_sha256=_require_sha256(obj["runtime_stack_sha256"], name="runtime_stack_sha256"),
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
            temperature_hex=_require_canonical_float_hex(
                obj["temperature_hex"], name="temperature_hex", positive=True
            ),
            absolute_tolerance_hex=_require_canonical_float_hex(
                obj["absolute_tolerance_hex"],
                name="absolute_tolerance_hex",
                positive=False,
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
        )


@dataclass(frozen=True, slots=True)
class StreamingTurnInstruction:
    """Scalar instructions and evidence keys for one later action replay."""

    turn_index: int
    rollout_turn_digest: str
    dialogue_digest: str
    sample_digest: str
    sample_verification_digest: str
    decision_example_digest: str
    decision_verification_digest: str
    action_trace_digest: str
    detached_statistics_digest: str
    action_token_count: int
    policy_coefficient: ExactScalar
    entropy_token_coefficient: ExactScalar

    def __post_init__(self) -> None:
        _require_integer(self.turn_index, name="turn_index")
        for name in (
            "rollout_turn_digest",
            "dialogue_digest",
            "sample_digest",
            "sample_verification_digest",
            "decision_example_digest",
            "decision_verification_digest",
            "action_trace_digest",
            "detached_statistics_digest",
        ):
            _require_sha256(getattr(self, name), name=name)
        _require_integer(self.action_token_count, name="action_token_count", minimum=1)
        if type(self.policy_coefficient) is not ExactScalar:
            raise StreamingObjectivePlanError("turn policy_coefficient must be an ExactScalar")
        if type(self.entropy_token_coefficient) is not ExactScalar:
            raise StreamingObjectivePlanError("turn entropy_token_coefficient must be an ExactScalar")

    def as_obj(self) -> dict[str, object]:
        return {
            "turn_index": self.turn_index,
            "rollout_turn_digest": self.rollout_turn_digest,
            "dialogue_digest": self.dialogue_digest,
            "sample_digest": self.sample_digest,
            "sample_verification_digest": self.sample_verification_digest,
            "decision_example_digest": self.decision_example_digest,
            "decision_verification_digest": self.decision_verification_digest,
            "action_trace_digest": self.action_trace_digest,
            "detached_statistics_digest": self.detached_statistics_digest,
            "action_token_count": self.action_token_count,
            "policy_coefficient": self.policy_coefficient.as_obj(),
            "entropy_token_coefficient": self.entropy_token_coefficient.as_obj(),
        }

    @classmethod
    def from_obj(cls, value: object, *, name: str) -> StreamingTurnInstruction:
        fields = (
            "turn_index",
            "rollout_turn_digest",
            "dialogue_digest",
            "sample_digest",
            "sample_verification_digest",
            "decision_example_digest",
            "decision_verification_digest",
            "action_trace_digest",
            "detached_statistics_digest",
            "action_token_count",
            "policy_coefficient",
            "entropy_token_coefficient",
        )
        obj = _require_exact_object(value, fields, name=name)
        return cls(
            turn_index=_require_integer(obj["turn_index"], name=f"{name}.turn_index"),
            rollout_turn_digest=_require_sha256(
                obj["rollout_turn_digest"], name=f"{name}.rollout_turn_digest"
            ),
            dialogue_digest=_require_sha256(obj["dialogue_digest"], name=f"{name}.dialogue_digest"),
            sample_digest=_require_sha256(obj["sample_digest"], name=f"{name}.sample_digest"),
            sample_verification_digest=_require_sha256(
                obj["sample_verification_digest"], name=f"{name}.sample_verification_digest"
            ),
            decision_example_digest=_require_sha256(
                obj["decision_example_digest"], name=f"{name}.decision_example_digest"
            ),
            decision_verification_digest=_require_sha256(
                obj["decision_verification_digest"],
                name=f"{name}.decision_verification_digest",
            ),
            action_trace_digest=_require_sha256(
                obj["action_trace_digest"], name=f"{name}.action_trace_digest"
            ),
            detached_statistics_digest=_require_sha256(
                obj["detached_statistics_digest"],
                name=f"{name}.detached_statistics_digest",
            ),
            action_token_count=_require_integer(
                obj["action_token_count"], name=f"{name}.action_token_count", minimum=1
            ),
            policy_coefficient=ExactScalar.from_obj(
                obj["policy_coefficient"], name=f"{name}.policy_coefficient"
            ),
            entropy_token_coefficient=ExactScalar.from_obj(
                obj["entropy_token_coefficient"],
                name=f"{name}.entropy_token_coefficient",
            ),
        )


@dataclass(frozen=True, slots=True)
class TurnZeroAbortEvidence:
    """Explicit evidence key for an authenticated empty rollout."""

    origin: AbortOrigin
    reason: AbortReason
    dialogue_digest: str
    record_digest: str
    rollout_verification_digest: str

    def __post_init__(self) -> None:
        if self.origin not in _ABORT_ORIGINS:
            raise StreamingObjectivePlanError("turn-zero abort has an unknown origin")
        if self.reason not in _ABORT_REASONS:
            raise StreamingObjectivePlanError("turn-zero abort has an unknown reason")
        for name in ("dialogue_digest", "record_digest", "rollout_verification_digest"):
            _require_sha256(getattr(self, name), name=name)

    def as_obj(self) -> dict[str, object]:
        return {
            "origin": self.origin,
            "reason": self.reason,
            "turn_index": 0,
            "dialogue_digest": self.dialogue_digest,
            "record_digest": self.record_digest,
            "rollout_verification_digest": self.rollout_verification_digest,
            "reward": {"numerator": 0, "denominator": 1},
        }

    @property
    def digest(self) -> str:
        return json_digest(self.as_obj(), domain=_TURN_ZERO_ABORT_DIGEST_DOMAIN)

    def as_bound_obj(self) -> dict[str, object]:
        return {**self.as_obj(), "digest": self.digest}

    @classmethod
    def from_obj(cls, value: object, *, name: str) -> TurnZeroAbortEvidence:
        fields = (
            "origin",
            "reason",
            "turn_index",
            "dialogue_digest",
            "record_digest",
            "rollout_verification_digest",
            "reward",
            "digest",
        )
        obj = _require_exact_object(value, fields, name=name)
        if _require_integer(obj["turn_index"], name=f"{name}.turn_index") != 0:
            raise StreamingObjectivePlanError("turn-zero abort evidence must address turn zero")
        reward = ExactScalar.from_obj(obj["reward"], name=f"{name}.reward")
        if reward.fraction != 0:
            raise StreamingObjectivePlanError("turn-zero abort reward must be exactly 0/1")
        result = cls(
            origin=cast(AbortOrigin, _require_text(obj["origin"], name=f"{name}.origin")),
            reason=cast(AbortReason, _require_text(obj["reason"], name=f"{name}.reason")),
            dialogue_digest=_require_sha256(obj["dialogue_digest"], name=f"{name}.dialogue_digest"),
            record_digest=_require_sha256(obj["record_digest"], name=f"{name}.record_digest"),
            rollout_verification_digest=_require_sha256(
                obj["rollout_verification_digest"],
                name=f"{name}.rollout_verification_digest",
            ),
        )
        supplied = _require_sha256(obj["digest"], name=f"{name}.digest")
        if not hmac.compare_digest(supplied, result.digest):
            raise StreamingObjectivePlanError("turn-zero abort evidence digest check failed")
        return result


@dataclass(frozen=True, slots=True)
class StreamingRolloutInstruction:
    """Exact sequence coefficient and bounded turn replay schedule for one rollout."""

    rollout_index: int
    rollout_digest: str
    record_digest: str
    rollout_verification_digest: str
    transcript_digest: str
    reward: ExactScalar
    leave_one_out_advantage: ExactScalar
    policy_coefficient: ExactScalar
    action_token_count: int
    turn_zero_abort: TurnZeroAbortEvidence | None
    turns: tuple[StreamingTurnInstruction, ...]

    def __post_init__(self) -> None:
        _require_integer(
            self.rollout_index,
            name="rollout_index",
            maximum=ROLLOUTS_PER_EPISODE - 1,
        )
        for name in (
            "rollout_digest",
            "record_digest",
            "rollout_verification_digest",
            "transcript_digest",
        ):
            _require_sha256(getattr(self, name), name=name)
        if self.rollout_digest != self.record_digest:
            raise StreamingObjectivePlanError("rollout and record digests must agree")
        for name in ("reward", "leave_one_out_advantage", "policy_coefficient"):
            if type(getattr(self, name)) is not ExactScalar:
                raise StreamingObjectivePlanError(f"{name} must be an ExactScalar")
        if not 0 <= self.reward.fraction <= 1:
            raise StreamingObjectivePlanError("rollout reward must lie in [0, 1]")
        turns = tuple(self.turns)
        object.__setattr__(self, "turns", turns)
        if any(type(turn) is not StreamingTurnInstruction for turn in turns):
            raise StreamingObjectivePlanError("rollout turns must be streaming turn instructions")
        if tuple(turn.turn_index for turn in turns) != tuple(range(len(turns))):
            raise StreamingObjectivePlanError("streaming turn indices must be contiguous from zero")
        total = sum(turn.action_token_count for turn in turns)
        if _require_integer(self.action_token_count, name="action_token_count") != total:
            raise StreamingObjectivePlanError("rollout action-token count differs from its turns")
        if self.turn_zero_abort is None:
            if not turns:
                raise StreamingObjectivePlanError(
                    "an empty rollout must retain authenticated turn-zero abort evidence"
                )
        else:
            if type(self.turn_zero_abort) is not TurnZeroAbortEvidence:
                raise StreamingObjectivePlanError("turn_zero_abort evidence has the wrong type")
            if turns or total != 0 or self.reward.fraction != 0:
                raise StreamingObjectivePlanError(
                    "turn-zero abort must have no turns, no action tokens, and reward zero"
                )
            if (
                self.turn_zero_abort.record_digest != self.record_digest
                or self.turn_zero_abort.rollout_verification_digest != self.rollout_verification_digest
            ):
                raise StreamingObjectivePlanError("turn-zero abort evidence belongs to another rollout")
        if any(turn.policy_coefficient != self.policy_coefficient for turn in turns):
            raise StreamingObjectivePlanError("turn and rollout policy coefficients differ")

    def as_obj(self) -> dict[str, object]:
        return {
            "rollout_index": self.rollout_index,
            "rollout_digest": self.rollout_digest,
            "record_digest": self.record_digest,
            "rollout_verification_digest": self.rollout_verification_digest,
            "transcript_digest": self.transcript_digest,
            "reward": self.reward.as_obj(),
            "leave_one_out_advantage": self.leave_one_out_advantage.as_obj(),
            "policy_coefficient": self.policy_coefficient.as_obj(),
            "action_token_count": self.action_token_count,
            "turn_zero_abort": (
                None if self.turn_zero_abort is None else self.turn_zero_abort.as_bound_obj()
            ),
            "turns": [turn.as_obj() for turn in self.turns],
        }

    @classmethod
    def from_obj(cls, value: object, *, name: str) -> StreamingRolloutInstruction:
        fields = (
            "rollout_index",
            "rollout_digest",
            "record_digest",
            "rollout_verification_digest",
            "transcript_digest",
            "reward",
            "leave_one_out_advantage",
            "policy_coefficient",
            "action_token_count",
            "turn_zero_abort",
            "turns",
        )
        obj = _require_exact_object(value, fields, name=name)
        raw_turns = _require_list(obj["turns"], name=f"{name}.turns")
        return cls(
            rollout_index=_require_integer(
                obj["rollout_index"],
                name=f"{name}.rollout_index",
                maximum=ROLLOUTS_PER_EPISODE - 1,
            ),
            rollout_digest=_require_sha256(obj["rollout_digest"], name=f"{name}.rollout_digest"),
            record_digest=_require_sha256(obj["record_digest"], name=f"{name}.record_digest"),
            rollout_verification_digest=_require_sha256(
                obj["rollout_verification_digest"],
                name=f"{name}.rollout_verification_digest",
            ),
            transcript_digest=_require_sha256(obj["transcript_digest"], name=f"{name}.transcript_digest"),
            reward=ExactScalar.from_obj(obj["reward"], name=f"{name}.reward"),
            leave_one_out_advantage=ExactScalar.from_obj(
                obj["leave_one_out_advantage"],
                name=f"{name}.leave_one_out_advantage",
            ),
            policy_coefficient=ExactScalar.from_obj(
                obj["policy_coefficient"], name=f"{name}.policy_coefficient"
            ),
            action_token_count=_require_integer(obj["action_token_count"], name=f"{name}.action_token_count"),
            turn_zero_abort=(
                None
                if obj["turn_zero_abort"] is None
                else TurnZeroAbortEvidence.from_obj(obj["turn_zero_abort"], name=f"{name}.turn_zero_abort")
            ),
            turns=tuple(
                StreamingTurnInstruction.from_obj(item, name=f"{name}.turns[{index}]")
                for index, item in enumerate(raw_turns)
            ),
        )


@dataclass(frozen=True, slots=True)
class StreamingGroupInstruction:
    """Eight rollout instructions for one unique hidden episode."""

    episode_id: str
    episode_digest: str
    run_seed: int
    group_digest: str
    action_evidence_group_digest: str
    action_token_count: int
    rollouts: tuple[StreamingRolloutInstruction, ...]

    def __post_init__(self) -> None:
        _require_text(self.episode_id, name="episode_id")
        _require_sha256(self.episode_digest, name="episode_digest")
        _require_integer(self.run_seed, name="run_seed", maximum=2**63 - 1)
        _require_sha256(self.group_digest, name="group_digest")
        _require_sha256(self.action_evidence_group_digest, name="action_evidence_group_digest")
        rollouts = tuple(self.rollouts)
        object.__setattr__(self, "rollouts", rollouts)
        if len(rollouts) != ROLLOUTS_PER_EPISODE or any(
            type(rollout) is not StreamingRolloutInstruction for rollout in rollouts
        ):
            raise StreamingObjectivePlanError("each streaming group must contain eight rollouts")
        if tuple(rollout.rollout_index for rollout in rollouts) != tuple(range(ROLLOUTS_PER_EPISODE)):
            raise StreamingObjectivePlanError("group rollout indices must be zero through seven")
        total = sum(rollout.action_token_count for rollout in rollouts)
        if _require_integer(self.action_token_count, name="action_token_count") != total:
            raise StreamingObjectivePlanError("group action-token count differs from its rollouts")

    def as_obj(self) -> dict[str, object]:
        return {
            "episode_id": self.episode_id,
            "episode_digest": self.episode_digest,
            "run_seed": self.run_seed,
            "group_digest": self.group_digest,
            "action_evidence_group_digest": self.action_evidence_group_digest,
            "action_token_count": self.action_token_count,
            "rollouts": [rollout.as_obj() for rollout in self.rollouts],
        }

    @classmethod
    def from_obj(cls, value: object, *, name: str) -> StreamingGroupInstruction:
        fields = (
            "episode_id",
            "episode_digest",
            "run_seed",
            "group_digest",
            "action_evidence_group_digest",
            "action_token_count",
            "rollouts",
        )
        obj = _require_exact_object(value, fields, name=name)
        raw_rollouts = _require_list(obj["rollouts"], name=f"{name}.rollouts")
        return cls(
            episode_id=_require_text(obj["episode_id"], name=f"{name}.episode_id"),
            episode_digest=_require_sha256(obj["episode_digest"], name=f"{name}.episode_digest"),
            run_seed=_require_integer(obj["run_seed"], name=f"{name}.run_seed", maximum=2**63 - 1),
            group_digest=_require_sha256(obj["group_digest"], name=f"{name}.group_digest"),
            action_evidence_group_digest=_require_sha256(
                obj["action_evidence_group_digest"],
                name=f"{name}.action_evidence_group_digest",
            ),
            action_token_count=_require_integer(obj["action_token_count"], name=f"{name}.action_token_count"),
            rollouts=tuple(
                StreamingRolloutInstruction.from_obj(item, name=f"{name}.rollouts[{index}]")
                for index, item in enumerate(raw_rollouts)
            ),
        )


@dataclass(frozen=True, slots=True)
class StreamingObjectivePlan:
    """Canonical structural plan; construction alone is not authentication."""

    update_invariants: StreamingUpdateInvariants
    trajectory_objective_digest: str
    entropy_coefficient: ExactScalar
    entropy_token_coefficient: ExactScalar
    group_count: int
    rollout_count: int
    total_authenticated_action_token_count: int
    optimizer_step_eligible: bool
    groups: tuple[StreamingGroupInstruction, ...]
    schema_version: int = STREAMING_OBJECTIVE_SCHEMA_VERSION
    contract_id: str = STREAMING_OBJECTIVE_CONTRACT_ID

    def __post_init__(self) -> None:
        if self.schema_version != STREAMING_OBJECTIVE_SCHEMA_VERSION:
            raise StreamingObjectivePlanError("unexpected streaming-objective schema version")
        if self.contract_id != STREAMING_OBJECTIVE_CONTRACT_ID:
            raise StreamingObjectivePlanError("unexpected streaming-objective contract id")
        if type(self.update_invariants) is not StreamingUpdateInvariants:
            raise StreamingObjectivePlanError("update_invariants has the wrong type")
        _require_sha256(self.trajectory_objective_digest, name="trajectory_objective_digest")
        if self.trajectory_objective_digest != trajectory_objective_digest():
            raise StreamingObjectivePlanError("trajectory-objective contract digest changed")
        if type(self.entropy_coefficient) is not ExactScalar:
            raise StreamingObjectivePlanError("entropy_coefficient must be an ExactScalar")
        if self.entropy_coefficient.fraction < 0:
            raise StreamingObjectivePlanError("entropy_coefficient must be non-negative")
        if type(self.entropy_token_coefficient) is not ExactScalar:
            raise StreamingObjectivePlanError("entropy_token_coefficient must be an ExactScalar")
        groups = tuple(self.groups)
        object.__setattr__(self, "groups", groups)
        if not groups or any(type(group) is not StreamingGroupInstruction for group in groups):
            raise StreamingObjectivePlanError("streaming plan requires at least one typed group")
        expected_order = tuple(sorted(groups, key=lambda group: (group.episode_digest, group.group_digest)))
        if groups != expected_order:
            raise StreamingObjectivePlanError("streaming groups are not in canonical schedule order")
        episodes = tuple(group.episode_digest for group in groups)
        if len(set(episodes)) != len(episodes):
            raise StreamingObjectivePlanError("the same hidden episode cannot appear twice")
        if _require_integer(self.group_count, name="group_count", minimum=1) != len(groups):
            raise StreamingObjectivePlanError("group_count is inconsistent")
        expected_rollouts = len(groups) * ROLLOUTS_PER_EPISODE
        if _require_integer(self.rollout_count, name="rollout_count", minimum=1) != expected_rollouts:
            raise StreamingObjectivePlanError("rollout_count is inconsistent")
        total_tokens = sum(group.action_token_count for group in groups)
        if (
            _require_integer(
                self.total_authenticated_action_token_count,
                name="total_authenticated_action_token_count",
            )
            != total_tokens
        ):
            raise StreamingObjectivePlanError("total authenticated action-token count is inconsistent")
        expected_entropy_scale = (
            Fraction(0, 1) if total_tokens == 0 else -self.entropy_coefficient.fraction / total_tokens
        )
        if self.entropy_token_coefficient.fraction != expected_entropy_scale:
            raise StreamingObjectivePlanError("entropy token scaling is inconsistent")
        if _require_boolean(
            self.optimizer_step_eligible,
            name="optimizer_step_eligible",
        ) is not (total_tokens > 0):
            raise StreamingObjectivePlanError("optimizer-step eligibility differs from token evidence")

        expected_policy_denominator = ROLLOUTS_PER_EPISODE * len(groups)
        for group in groups:
            rewards = tuple(rollout.reward.fraction for rollout in group.rollouts)
            reward_total = sum(rewards, Fraction(0, 1))
            for rollout in group.rollouts:
                expected_advantage = rollout.reward.fraction - (reward_total - rollout.reward.fraction) / (
                    ROLLOUTS_PER_EPISODE - 1
                )
                if rollout.leave_one_out_advantage.fraction != expected_advantage:
                    raise StreamingObjectivePlanError("leave-one-out advantage is inconsistent")
                expected_policy = -expected_advantage / expected_policy_denominator
                if rollout.policy_coefficient.fraction != expected_policy:
                    raise StreamingObjectivePlanError("rollout policy coefficient is inconsistent")
                for turn in rollout.turns:
                    if turn.entropy_token_coefficient != self.entropy_token_coefficient:
                        raise StreamingObjectivePlanError("turn entropy scaling differs from the plan")

    def as_obj(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "update_invariants": self.update_invariants.as_obj(),
            "trajectory_objective_digest": self.trajectory_objective_digest,
            "entropy_coefficient": self.entropy_coefficient.as_obj(),
            "entropy_token_coefficient": self.entropy_token_coefficient.as_obj(),
            "group_count": self.group_count,
            "rollout_count": self.rollout_count,
            "total_authenticated_action_token_count": (self.total_authenticated_action_token_count),
            "optimizer_step_eligible": self.optimizer_step_eligible,
            "groups": [group.as_obj() for group in self.groups],
            "execution_contract": _execution_contract_obj(),
            "authorization": {
                "model_load": False,
                "rollout_launch": False,
                "production_executor_present": False,
                "weight_update": False,
                "optimizer_step": False,
            },
        }

    @property
    def digest(self) -> str:
        return json_digest(self.as_obj(), domain=_PLAN_DIGEST_DOMAIN)

    def to_json(self) -> str:
        return dump_json({**self.as_obj(), "digest": self.digest})

    @classmethod
    def from_obj(cls, value: object) -> StreamingObjectivePlan:
        fields = (
            "schema_version",
            "contract_id",
            "update_invariants",
            "trajectory_objective_digest",
            "entropy_coefficient",
            "entropy_token_coefficient",
            "group_count",
            "rollout_count",
            "total_authenticated_action_token_count",
            "optimizer_step_eligible",
            "groups",
            "execution_contract",
            "authorization",
            "digest",
        )
        obj = _require_exact_object(value, fields, name="streaming_objective_plan")
        execution_contract = _require_exact_object(
            obj["execution_contract"],
            (
                "reauthenticate_each_stored_action",
                "bounded_graph_scope",
                "parameter_mutation_between_replays",
                "optimizer_state_mutation_between_replays",
                "optimizer_step_after_all_entries_only",
            ),
            name="execution_contract",
        )
        if execution_contract != _execution_contract_obj():
            raise StreamingObjectivePlanError("streaming execution contract changed")
        authorization = _require_exact_object(
            obj["authorization"],
            (
                "model_load",
                "rollout_launch",
                "production_executor_present",
                "weight_update",
                "optimizer_step",
            ),
            name="authorization",
        )
        if authorization != {
            "model_load": False,
            "rollout_launch": False,
            "production_executor_present": False,
            "weight_update": False,
            "optimizer_step": False,
        }:
            raise StreamingObjectivePlanError("streaming objective plan cannot carry authorization")
        raw_groups = _require_list(obj["groups"], name="groups")
        result = cls(
            update_invariants=StreamingUpdateInvariants.from_obj(obj["update_invariants"]),
            trajectory_objective_digest=_require_sha256(
                obj["trajectory_objective_digest"], name="trajectory_objective_digest"
            ),
            entropy_coefficient=ExactScalar.from_obj(obj["entropy_coefficient"], name="entropy_coefficient"),
            entropy_token_coefficient=ExactScalar.from_obj(
                obj["entropy_token_coefficient"], name="entropy_token_coefficient"
            ),
            group_count=_require_integer(obj["group_count"], name="group_count", minimum=1),
            rollout_count=_require_integer(obj["rollout_count"], name="rollout_count", minimum=1),
            total_authenticated_action_token_count=_require_integer(
                obj["total_authenticated_action_token_count"],
                name="total_authenticated_action_token_count",
            ),
            optimizer_step_eligible=_require_boolean(
                obj["optimizer_step_eligible"], name="optimizer_step_eligible"
            ),
            groups=tuple(
                StreamingGroupInstruction.from_obj(item, name=f"groups[{index}]")
                for index, item in enumerate(raw_groups)
            ),
            schema_version=_require_integer(obj["schema_version"], name="schema_version", minimum=1),
            contract_id=_require_text(obj["contract_id"], name="contract_id"),
        )
        supplied = _require_sha256(obj["digest"], name="streaming_objective_plan.digest")
        if not hmac.compare_digest(supplied, result.digest):
            raise StreamingObjectivePlanError("streaming-objective plan digest check failed")
        return result

    @classmethod
    def from_json(cls, text: str) -> StreamingObjectivePlan:
        if type(text) is not str:
            raise StreamingObjectivePlanError("streaming objective plan JSON must be text")
        try:
            value = load_json(text)
        except CanonicalJSONError as exc:
            raise StreamingObjectivePlanError("streaming objective plan is not strict JSON") from exc
        result = cls.from_obj(value)
        if not hmac.compare_digest(text.encode("utf-8"), result.to_json().encode("utf-8")):
            raise StreamingObjectivePlanError(
                "streaming objective plan JSON is not in the unique canonical byte representation"
            )
        return result


@dataclass(frozen=True, slots=True, init=False)
class _VerifiedStreamingObjectivePlan:
    """Nominal evidence of fresh reduction from verified rollout-group wrappers."""

    plan: StreamingObjectivePlan
    verification_digest: str

    @classmethod
    def _from_derived(cls, plan: StreamingObjectivePlan) -> _VerifiedStreamingObjectivePlan:
        if type(plan) is not StreamingObjectivePlan:
            raise TypeError("plan must be a StreamingObjectivePlan")
        result = object.__new__(cls)
        object.__setattr__(result, "plan", plan)
        object.__setattr__(
            result,
            "verification_digest",
            json_digest(
                {
                    "plan_digest": plan.digest,
                    "group_digests": [group.group_digest for group in plan.groups],
                    "action_evidence_group_digests": [
                        group.action_evidence_group_digest for group in plan.groups
                    ],
                    "source": "fresh_verified_authenticated_rollout_groups",
                },
                domain=_VERIFIED_PLAN_DIGEST_DOMAIN,
            ),
        )
        return result


def _update_invariants(record: AuthenticatedRolloutRecord) -> StreamingUpdateInvariants:
    provenance = record.model_provenance
    return StreamingUpdateInvariants(
        model_identifier=provenance.model_identifier,
        model_revision=provenance.revision,
        artifact_manifest_sha256=provenance.artifact_manifest_sha256,
        runtime_stack_sha256=provenance.runtime_stack_sha256,
        model_provenance_digest=record.model_provenance_digest,
        policy_state_digest=record.policy_state_digest,
        tokenizer_binding_digest=record.tokenizer_binding_digest,
        compiler_manifest_digest=record.compiler_manifest_digest,
        temperature_hex=record.temperature_hex,
        absolute_tolerance_hex=record.absolute_tolerance_hex,
        maximum_sequence_tokens=record.maximum_sequence_tokens,
        maximum_turns=record.maximum_turns,
    )


def _turn_instruction(
    turn: AuthenticatedRolloutTurn,
    verified: GraphFreeVerifiedAuthenticatedActionSample,
    *,
    policy_coefficient: ExactScalar,
    entropy_token_coefficient: ExactScalar,
) -> StreamingTurnInstruction:
    sample = verified.sample
    count = len(sample.selected_token_ids)
    if count != len(sample.decision_example.action_trace.steps):
        raise StreamingObjectivePlanError("authenticated action-token evidence is inconsistent")
    if turn.nominal_verification_digest != verified.verification_digest:
        raise StreamingObjectivePlanError("turn nominal verification evidence changed")
    if sample.decision_verification_digest != verified.verified_decision.verification_digest:
        raise StreamingObjectivePlanError("decision verification evidence changed")
    return StreamingTurnInstruction(
        turn_index=turn.turn_index,
        rollout_turn_digest=turn.digest,
        dialogue_digest=turn.dialogue_digest,
        sample_digest=sample.digest,
        sample_verification_digest=verified.verification_digest,
        decision_example_digest=sample.decision_example.digest,
        decision_verification_digest=sample.decision_verification_digest,
        action_trace_digest=sample.decision_example.action_trace.digest,
        detached_statistics_digest=sample.detached_statistics.digest,
        action_token_count=count,
        policy_coefficient=policy_coefficient,
        entropy_token_coefficient=entropy_token_coefficient,
    )


def _rollout_instruction(
    rollout: VerifiedAuthenticatedRollout,
    *,
    advantage: Fraction,
    policy_coefficient: Fraction,
    entropy_token_coefficient: ExactScalar,
) -> StreamingRolloutInstruction:
    if type(rollout) is not VerifiedAuthenticatedRollout:
        raise StreamingObjectivePlanError("streaming plans require VerifiedAuthenticatedRollout evidence")
    record = rollout.record
    if type(record) is not AuthenticatedRolloutRecord:
        raise StreamingObjectivePlanError("verified rollout record has the wrong type")
    if len(record.turns) != len(rollout.verified_samples):
        raise StreamingObjectivePlanError("rollout turns and verification evidence differ")
    policy_scalar = ExactScalar.from_fraction(policy_coefficient)
    turns = tuple(
        _turn_instruction(
            turn,
            verified,
            policy_coefficient=policy_scalar,
            entropy_token_coefficient=entropy_token_coefficient,
        )
        for turn, verified in zip(record.turns, rollout.verified_samples, strict=True)
    )
    abort: TurnZeroAbortEvidence | None = None
    if not turns:
        termination = record.termination
        if (
            termination is None
            or termination.turn_index != 0
            or record.reward_numerator != 0
            or record.reward_denominator != 1
        ):
            raise StreamingObjectivePlanError(
                "empty verified rollout is not a registered turn-zero zero-reward abort"
            )
        abort = TurnZeroAbortEvidence(
            origin=termination.origin,
            reason=termination.reason,
            dialogue_digest=termination.dialogue_digest,
            record_digest=record.digest,
            rollout_verification_digest=rollout.verification_digest,
        )
    return StreamingRolloutInstruction(
        rollout_index=record.rollout_index,
        rollout_digest=rollout.digest,
        record_digest=record.digest,
        rollout_verification_digest=rollout.verification_digest,
        transcript_digest=record.transcript_digest,
        reward=ExactScalar(record.reward_numerator, record.reward_denominator),
        leave_one_out_advantage=ExactScalar.from_fraction(advantage),
        policy_coefficient=policy_scalar,
        action_token_count=sum(turn.action_token_count for turn in turns),
        turn_zero_abort=abort,
        turns=turns,
    )


def derive_verified_streaming_objective_plan(
    groups: Sequence[VerifiedAuthenticatedRolloutGroup],
    *,
    entropy_coefficient: Fraction | int | float = Fraction(0, 1),
) -> _VerifiedStreamingObjectivePlan:
    """Reduce authenticated groups to a graph-free, schedule-invariant plan.

    This is the only authentication-producing entry point.  It deliberately
    accepts no raw rollout record and returns a private nominal wrapper.  The
    wrapper retains the canonical plan only, never the supplied rollout
    wrappers or their differentiable replay tensors.
    """

    selected = tuple(groups)
    if not selected or any(type(group) is not VerifiedAuthenticatedRolloutGroup for group in selected):
        raise StreamingObjectivePlanError(
            "streaming plans require VerifiedAuthenticatedRolloutGroup nominal wrappers"
        )
    entropy = _fraction_from_input(entropy_coefficient, name="entropy_coefficient")

    reverified: list[VerifiedAuthenticatedRolloutGroup] = []
    for supplied_group in selected:
        regenerated = verify_eight_authenticated_rollout_group(supplied_group.rollouts)
        if (
            regenerated.digest != supplied_group.digest
            or regenerated.action_evidence_group_digest != supplied_group.action_evidence_group_digest
        ):
            raise StreamingObjectivePlanError("authenticated rollout-group evidence changed")
        reverified.append(regenerated)
    ordered = tuple(
        sorted(reverified, key=lambda group: (group.rollouts[0].record.episode_digest, group.digest))
    )
    episode_digests = tuple(group.rollouts[0].record.episode_digest for group in ordered)
    if len(set(episode_digests)) != len(episode_digests):
        raise StreamingObjectivePlanError("the same hidden episode cannot appear twice")

    reference_record = ordered[0].rollouts[0].record
    invariants = _update_invariants(reference_record)
    reference_run_seed = reference_record.run_seed
    for group in ordered:
        record = group.rollouts[0].record
        if _update_invariants(record) != invariants:
            raise StreamingObjectivePlanError(
                "all groups in one update must share policy, provenance, and runtime invariants"
            )
        if record.run_seed != reference_run_seed:
            raise StreamingObjectivePlanError("all groups in one update must share the run seed")

    total_tokens = sum(
        len(verified.sample.selected_token_ids)
        for group in ordered
        for rollout in group.rollouts
        for verified in rollout.verified_samples
    )
    entropy_scale = Fraction(0, 1) if total_tokens == 0 else -entropy / total_tokens
    entropy_scalar = ExactScalar.from_fraction(entropy_scale)
    group_count = len(ordered)
    policy_denominator = ROLLOUTS_PER_EPISODE * group_count

    group_instructions: list[StreamingGroupInstruction] = []
    for group in ordered:
        records = tuple(rollout.record for rollout in group.rollouts)
        rewards = tuple(Fraction(record.reward_numerator, record.reward_denominator) for record in records)
        reward_total = sum(rewards, Fraction(0, 1))
        rollout_instructions: list[StreamingRolloutInstruction] = []
        for rollout, reward in zip(group.rollouts, rewards, strict=True):
            advantage = reward - (reward_total - reward) / (ROLLOUTS_PER_EPISODE - 1)
            policy_coefficient = -advantage / policy_denominator
            rollout_instructions.append(
                _rollout_instruction(
                    rollout,
                    advantage=advantage,
                    policy_coefficient=policy_coefficient,
                    entropy_token_coefficient=entropy_scalar,
                )
            )
        reference = records[0]
        group_instructions.append(
            StreamingGroupInstruction(
                episode_id=reference.episode_id,
                episode_digest=reference.episode_digest,
                run_seed=reference.run_seed,
                group_digest=group.digest,
                action_evidence_group_digest=group.action_evidence_group_digest,
                action_token_count=sum(rollout.action_token_count for rollout in rollout_instructions),
                rollouts=tuple(rollout_instructions),
            )
        )

    plan = StreamingObjectivePlan(
        update_invariants=invariants,
        trajectory_objective_digest=trajectory_objective_digest(),
        entropy_coefficient=ExactScalar.from_fraction(entropy),
        entropy_token_coefficient=entropy_scalar,
        group_count=group_count,
        rollout_count=ROLLOUTS_PER_EPISODE * group_count,
        total_authenticated_action_token_count=total_tokens,
        optimizer_step_eligible=total_tokens > 0,
        groups=tuple(group_instructions),
    )
    return _VerifiedStreamingObjectivePlan._from_derived(plan)


def parse_streaming_objective_plan(text: str) -> StreamingObjectivePlan:
    """Parse a unique canonical plan as structural, non-nominal evidence."""

    return StreamingObjectivePlan.from_json(text)


def streaming_objective_manifest() -> dict[str, object]:
    """Describe this nonauthorizing planning bridge."""

    return {
        "schema_version": STREAMING_OBJECTIVE_SCHEMA_VERSION,
        "contract_id": STREAMING_OBJECTIVE_CONTRACT_ID,
        "accepted_source": "VerifiedAuthenticatedRolloutGroup nominal wrappers only",
        "rollouts_per_group": ROLLOUTS_PER_EPISODE,
        "policy_coefficient": "-exact_leave_one_out_advantage/(8*group_count)",
        "entropy_token_coefficient": "-exact_entropy_coefficient/total_action_tokens",
        "turn_zero_abort": "retained as an explicit zero-token exact-zero-reward entry",
        "retained_collection_graphs": False,
        "parsed_plan_is_verified": False,
        "sequential_replay_parameter_mutation_allowed": False,
        "sequential_replay_optimizer_mutation_allowed": False,
        "optimizer_step_timing": "only after every authenticated instruction has accumulated",
        "production_executor_present": False,
        "model_load_authorization": False,
        "rollout_launch_authorization": False,
        "weight_update_authorization": False,
        "optimizer_step_authorization": False,
    }
