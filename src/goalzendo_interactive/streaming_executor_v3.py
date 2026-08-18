"""Bounded-memory authenticated backward execution for G03 outcome RL.

This module consumes a freshly nominal streaming plan and structural rollout
records, then reauthenticates and backpropagates one action turn at a time.
It never owns an optimizer, never calls ``step``, and never mutates parameter
data.  A successful call leaves only accumulated first-order gradients and a
graph-free, nonauthorizing diagnostic record.

The executor is intentionally additive.  It does not authorize model loading,
rollout collection, backward execution, or a later optimizer step.
"""

from __future__ import annotations

import gc
import hashlib
import hmac
import math
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Literal, Protocol, cast, runtime_checkable

import torch

from ._json import dump_json, json_digest
from .action_tokenization_v2 import ExactDecodeTokenizerProtocol, FragmentActionTokenCompiler
from .authenticated_model_provider_v2 import TrainableParameterRegistry
from .authenticated_rollouts_v2 import (
    AuthenticatedRolloutRecord,
    ModelPolicyProvenance,
    ProvenancedFullForwardLogitsProvider,
    RolloutControllerProtocol,
    parse_authenticated_rollout,
)
from .authenticated_sampler_v2 import (
    AuthenticatedSamplingError,
    AuthenticatedSamplingOverlengthError,
    replay_authenticated_sample,
    sample_authenticated_action,
)
from .dialogue import Dialogue, dialogue_as_obj, render_dialogue
from .environment import HiddenLawEnvironment
from .episodes import HiddenEpisode
from .policy_randomness_v2 import PolicyTurnSeed
from .streaming_objective_v3 import (
    ExactScalar,
    StreamingGroupInstruction,
    StreamingObjectivePlan,
    StreamingTurnInstruction,
    _VerifiedStreamingObjectivePlan,
    parse_streaming_objective_plan,
)
from .transcripts import AbortEvent, AbortReason, AnswerEvent, Transcript

STREAMING_EXECUTOR_SCHEMA_VERSION = 4
STREAMING_EXECUTOR_CONTRACT_ID = "goalzendo-streaming-backward-executor-v4"

_DIAGNOSTIC_DIGEST_DOMAIN = "goalzendo-interactive-streaming-backward-diagnostics-v4"
_DIALOGUE_DIGEST_DOMAIN = "goalzendo-interactive-authenticated-rollout-dialogue-v2"
_ROLLOUT_VERIFICATION_DIGEST_DOMAIN = "goalzendo-interactive-authenticated-rollout-replay-v2"
_ROLLOUT_GROUP_DIGEST_DOMAIN = "goalzendo-interactive-authenticated-rollout-group-v2"
_ROLLOUT_ACTION_EVIDENCE_GROUP_DIGEST_DOMAIN = (
    "goalzendo-interactive-authenticated-rollout-action-evidence-group-v2"
)
_SAMPLER_GROUP_DIGEST_DOMAIN = "goalzendo-interactive-unchanged-policy-rollout-group-v1"
_FP32_BUFFER_MANIFEST_DOMAIN = "goalzendo-interactive-streaming-fp32-gradient-buffers-v4"
_FINAL_GRADIENT_MANIFEST_DOMAIN = "goalzendo-interactive-streaming-final-gradients-v4"


class StreamingExecutorError(RuntimeError):
    """Raised when authenticated streaming backward execution fails closed."""


@runtime_checkable
class FullByteReauthenticatingProvider(ProvenancedFullForwardLogitsProvider, Protocol):
    """A provenanced logits provider with independent full-byte reauthentication."""

    def reauthenticate_policy_state(self) -> ModelPolicyProvenance: ...

    @property
    def trainable_parameter_registry(self) -> TrainableParameterRegistry: ...


@dataclass(frozen=True, slots=True)
class ReplayCoordinate:
    """One canonical graph-lifetime coordinate."""

    episode_digest: str
    rollout_index: int
    turn_index: int

    def as_obj(self) -> dict[str, object]:
        return {
            "episode_digest": self.episode_digest,
            "rollout_index": self.rollout_index,
            "turn_index": self.turn_index,
        }


@runtime_checkable
class GraphLifecycleObserver(Protocol):
    """Optional proof hook receiving weak references, never live graph ownership."""

    def graph_opened(
        self,
        coordinate: ReplayCoordinate,
        anchor: weakref.ReferenceType[torch.Tensor],
    ) -> None: ...

    def graph_closed(
        self,
        coordinate: ReplayCoordinate,
        anchor: weakref.ReferenceType[torch.Tensor],
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class ParameterGradientByteRecord:
    """Canonical bytes for one deduplicated parameter gradient representation."""

    name: str
    aliases: tuple[str, ...]
    parameter_dtype: str
    parameter_device: str
    parameter_shape: tuple[int, ...]
    gradient_dtype: str
    gradient_device: str
    sha256: str

    def __post_init__(self) -> None:
        aliases = tuple(self.aliases)
        object.__setattr__(self, "aliases", aliases)
        if (
            type(self.name) is not str
            or not self.name
            or tuple(sorted(set(aliases))) != aliases
            or not aliases
            or aliases[0] != self.name
        ):
            raise StreamingExecutorError("gradient record aliases are not canonical")
        for value in (
            self.parameter_dtype,
            self.parameter_device,
            self.gradient_dtype,
            self.gradient_device,
        ):
            if type(value) is not str or not value:
                raise StreamingExecutorError("gradient record metadata must be nonempty text")
        if any(
            isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0
            for dimension in self.parameter_shape
        ):
            raise StreamingExecutorError("gradient record parameter shape is invalid")
        if (
            type(self.sha256) is not str
            or len(self.sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.sha256)
        ):
            raise StreamingExecutorError("gradient record SHA-256 is invalid")

    def as_obj(self) -> dict[str, object]:
        return {
            "name": self.name,
            "aliases": list(self.aliases),
            "parameter_dtype": self.parameter_dtype,
            "parameter_device": self.parameter_device,
            "parameter_shape": list(self.parameter_shape),
            "gradient_dtype": self.gradient_dtype,
            "gradient_device": self.gradient_device,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True, init=False)
class StreamingExecutionDiagnostics:
    """Graph-free evidence about one completed, optimizer-free backward pass."""

    plan_digest: str
    plan_verification_digest: str
    policy_state_digest: str
    model_provenance_digest: str
    parameter_registry_digest: str
    registered_parameter_name_count: int
    group_count: int
    rollout_count: int
    authenticated_turn_replay_count: int
    authenticated_termination_replay_count: int
    authenticated_turn_zero_abort_count: int
    authenticated_action_token_count: int
    backward_call_count: int
    parameter_count: int
    finite_gradient_parameter_count: int
    gradient_accumulation_dtype: str
    gradient_accumulation_order: str
    fp32_contribution_cast_count: int
    final_gradient_cast_count: int
    fp32_gradient_buffers: tuple[ParameterGradientByteRecord, ...]
    fp32_gradient_buffer_manifest_digest: str
    final_gradients: tuple[ParameterGradientByteRecord, ...]
    final_gradient_manifest_digest: str
    full_byte_reauthentication_count: int
    explicit_lightweight_guard_count: int
    maximum_live_graph_count: int
    released_graph_anchor_count: int
    stored_expected_policy_loss_hex: str
    stored_expected_entropy_loss_hex: str
    stored_expected_total_loss_hex: str
    replayed_policy_loss_hex: str
    replayed_entropy_loss_hex: str
    replayed_total_loss_hex: str
    stored_replay_absolute_error_hex: str
    stored_replay_absolute_error_bound_hex: str
    schema_version: int
    contract_id: str

    @classmethod
    def _from_execution(
        cls,
        *,
        plan: StreamingObjectivePlan,
        plan_verification_digest: str,
        parameter_registry_digest: str,
        registered_parameter_name_count: int,
        authenticated_turn_replay_count: int,
        authenticated_termination_replay_count: int,
        authenticated_turn_zero_abort_count: int,
        backward_call_count: int,
        parameter_count: int,
        finite_gradient_parameter_count: int,
        fp32_contribution_cast_count: int,
        final_gradient_cast_count: int,
        fp32_gradient_buffers: tuple[ParameterGradientByteRecord, ...],
        fp32_gradient_buffer_manifest_digest: str,
        final_gradients: tuple[ParameterGradientByteRecord, ...],
        final_gradient_manifest_digest: str,
        full_byte_reauthentication_count: int,
        explicit_lightweight_guard_count: int,
        maximum_live_graph_count: int,
        released_graph_anchor_count: int,
        stored_policy_loss: float,
        stored_entropy_loss: float,
        replayed_policy_loss: float,
        replayed_entropy_loss: float,
        stored_replay_absolute_error_bound: float,
    ) -> StreamingExecutionDiagnostics:
        result = object.__new__(cls)
        values: dict[str, object] = {
            "plan_digest": plan.digest,
            "plan_verification_digest": plan_verification_digest,
            "policy_state_digest": plan.update_invariants.policy_state_digest,
            "model_provenance_digest": plan.update_invariants.model_provenance_digest,
            "parameter_registry_digest": parameter_registry_digest,
            "registered_parameter_name_count": registered_parameter_name_count,
            "group_count": plan.group_count,
            "rollout_count": plan.rollout_count,
            "authenticated_turn_replay_count": authenticated_turn_replay_count,
            "authenticated_termination_replay_count": authenticated_termination_replay_count,
            "authenticated_turn_zero_abort_count": authenticated_turn_zero_abort_count,
            "authenticated_action_token_count": plan.total_authenticated_action_token_count,
            "backward_call_count": backward_call_count,
            "parameter_count": parameter_count,
            "finite_gradient_parameter_count": finite_gradient_parameter_count,
            "gradient_accumulation_dtype": "torch.float32",
            "gradient_accumulation_order": "canonical_group_rollout_turn_then_parameter_name",
            "fp32_contribution_cast_count": fp32_contribution_cast_count,
            "final_gradient_cast_count": final_gradient_cast_count,
            "fp32_gradient_buffers": fp32_gradient_buffers,
            "fp32_gradient_buffer_manifest_digest": fp32_gradient_buffer_manifest_digest,
            "final_gradients": final_gradients,
            "final_gradient_manifest_digest": final_gradient_manifest_digest,
            "full_byte_reauthentication_count": full_byte_reauthentication_count,
            "explicit_lightweight_guard_count": explicit_lightweight_guard_count,
            "maximum_live_graph_count": maximum_live_graph_count,
            "released_graph_anchor_count": released_graph_anchor_count,
            "stored_expected_policy_loss_hex": _finite_float_hex(
                stored_policy_loss, name="stored expected policy loss"
            ),
            "stored_expected_entropy_loss_hex": _finite_float_hex(
                stored_entropy_loss, name="stored expected entropy loss"
            ),
            "stored_expected_total_loss_hex": _finite_float_hex(
                math.fsum((stored_policy_loss, stored_entropy_loss)),
                name="stored expected total loss",
            ),
            "replayed_policy_loss_hex": _finite_float_hex(replayed_policy_loss, name="replayed policy loss"),
            "replayed_entropy_loss_hex": _finite_float_hex(
                replayed_entropy_loss, name="replayed entropy loss"
            ),
            "replayed_total_loss_hex": _finite_float_hex(
                math.fsum((replayed_policy_loss, replayed_entropy_loss)),
                name="replayed total loss",
            ),
            "stored_replay_absolute_error_hex": _finite_float_hex(
                abs(
                    math.fsum((stored_policy_loss, stored_entropy_loss))
                    - math.fsum((replayed_policy_loss, replayed_entropy_loss))
                ),
                name="stored/replayed aggregate absolute error",
            ),
            "stored_replay_absolute_error_bound_hex": _finite_float_hex(
                stored_replay_absolute_error_bound,
                name="stored/replayed aggregate absolute error bound",
            ),
            "schema_version": STREAMING_EXECUTOR_SCHEMA_VERSION,
            "contract_id": STREAMING_EXECUTOR_CONTRACT_ID,
        }
        for name, value in values.items():
            object.__setattr__(result, name, value)
        return result

    def as_obj(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "plan_digest": self.plan_digest,
            "plan_verification_digest": self.plan_verification_digest,
            "policy_state_digest": self.policy_state_digest,
            "model_provenance_digest": self.model_provenance_digest,
            "parameter_registry_digest": self.parameter_registry_digest,
            "registered_parameter_name_count": self.registered_parameter_name_count,
            "group_count": self.group_count,
            "rollout_count": self.rollout_count,
            "authenticated_turn_replay_count": self.authenticated_turn_replay_count,
            "authenticated_termination_replay_count": (self.authenticated_termination_replay_count),
            "authenticated_turn_zero_abort_count": self.authenticated_turn_zero_abort_count,
            "authenticated_action_token_count": self.authenticated_action_token_count,
            "backward_call_count": self.backward_call_count,
            "parameter_count": self.parameter_count,
            "finite_gradient_parameter_count": self.finite_gradient_parameter_count,
            "gradient_accumulation_dtype": self.gradient_accumulation_dtype,
            "gradient_accumulation_order": self.gradient_accumulation_order,
            "fp32_contribution_cast_count": self.fp32_contribution_cast_count,
            "final_gradient_cast_count": self.final_gradient_cast_count,
            "fp32_gradient_buffer_manifest_digest": self.fp32_gradient_buffer_manifest_digest,
            "fp32_gradient_buffers": [record.as_obj() for record in self.fp32_gradient_buffers],
            "final_gradient_manifest_digest": self.final_gradient_manifest_digest,
            "final_gradients": [record.as_obj() for record in self.final_gradients],
            "full_byte_reauthentication_count": self.full_byte_reauthentication_count,
            "explicit_lightweight_guard_count": self.explicit_lightweight_guard_count,
            "maximum_live_graph_count": self.maximum_live_graph_count,
            "released_graph_anchor_count": self.released_graph_anchor_count,
            "stored_expected_policy_loss_hex": self.stored_expected_policy_loss_hex,
            "stored_expected_entropy_loss_hex": self.stored_expected_entropy_loss_hex,
            "stored_expected_total_loss_hex": self.stored_expected_total_loss_hex,
            "replayed_policy_loss_hex": self.replayed_policy_loss_hex,
            "replayed_entropy_loss_hex": self.replayed_entropy_loss_hex,
            "replayed_total_loss_hex": self.replayed_total_loss_hex,
            "stored_replay_absolute_error_hex": self.stored_replay_absolute_error_hex,
            "stored_replay_absolute_error_bound_hex": (self.stored_replay_absolute_error_bound_hex),
            "initial_gradient_contract": "all_registered_parameter_grads_are_none",
            "gradient_accumulation": "autograd_grad_to_fp32_buffers_then_one_final_cast",
            "parameter_data_mutated": False,
            "optimizer_present": False,
            "optimizer_step_called": False,
            "graphs_retained": False,
            "authorization": {
                "model_load": False,
                "rollout_launch": False,
                "backward_execution": False,
                "weight_update": False,
                "optimizer_step": False,
            },
        }

    @property
    def digest(self) -> str:
        return json_digest(self.as_obj(), domain=_DIAGNOSTIC_DIGEST_DOMAIN)

    def to_json(self) -> str:
        return dump_json({**self.as_obj(), "digest": self.digest})


@dataclass(frozen=True, slots=True)
class _ParameterGuard:
    name: str
    aliases: tuple[str, ...]
    identity: int
    version: int
    dtype: str
    device: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int
    data_pointer: int
    requires_grad: bool

    def reproducible_obj(self) -> dict[str, object]:
        return {
            "name": self.name,
            "aliases": list(self.aliases),
            "dtype": self.dtype,
            "device": self.device,
            "shape": list(self.shape),
            "stride": list(self.stride),
            "storage_offset": self.storage_offset,
            "requires_grad": self.requires_grad,
        }


@dataclass(frozen=True, slots=True)
class _TurnBackwardResult:
    verification_digest: str
    token_count: int
    stored_policy_loss: float
    stored_entropy_loss: float
    replayed_policy_loss: float
    replayed_entropy_loss: float
    stored_replay_absolute_error_bound: float
    graph_leaf_parameter_ids: frozenset[int]
    fp32_contribution_cast_count: int


class _GraphCounter:
    __slots__ = ("live", "maximum", "observer", "released")

    def __init__(self, observer: GraphLifecycleObserver | None) -> None:
        self.live = 0
        self.maximum = 0
        self.released = 0
        self.observer = observer

    def open(
        self,
        coordinate: ReplayCoordinate,
        anchor: torch.Tensor,
    ) -> weakref.ReferenceType[torch.Tensor]:
        if self.live != 0:
            raise StreamingExecutorError("more than one differentiable turn graph became live")
        reference = weakref.ref(anchor)
        self.live = 1
        self.maximum = max(self.maximum, self.live)
        if self.observer is not None:
            self.observer.graph_opened(coordinate, reference)
        return reference

    def close(
        self,
        coordinate: ReplayCoordinate,
        reference: weakref.ReferenceType[torch.Tensor],
    ) -> None:
        try:
            if self.observer is not None:
                self.observer.graph_closed(coordinate, reference)
            gc.collect()
            if reference() is not None:
                raise StreamingExecutorError(
                    "a differentiable turn graph anchor remained strongly referenced"
                )
            self.released += 1
        finally:
            self.live -= 1
            if self.live < 0:
                raise StreamingExecutorError("graph-lifetime counter underflow")


def _finite_float_hex(value: float, *, name: str) -> str:
    selected = float(value)
    if not math.isfinite(selected):
        raise StreamingExecutorError(f"{name} is non-finite")
    return selected.hex()


def _scalar_float(value: ExactScalar) -> float:
    return value.numerator / value.denominator


def _dialogue_digest(episode: HiddenEpisode, transcript: Transcript) -> tuple[Dialogue, str]:
    dialogue = render_dialogue(episode, transcript)
    return dialogue, json_digest(dialogue_as_obj(dialogue), domain=_DIALOGUE_DIGEST_DOMAIN)


def _terminal_reward(transcript: Transcript) -> Fraction:
    if not transcript.events or type(transcript.events[-1]) is not AnswerEvent:
        return Fraction(0, 1)
    score = transcript.events[-1].score
    return (
        Fraction(7 * score.classification_correct, 10 * score.classification_total)
        + Fraction(int(score.rule_equivalent), 4)
        + Fraction(1, 20) * (1 - Fraction(score.query_count, 6))
    )


def _parameter_guard(
    name: str,
    aliases: tuple[str, ...],
    parameter: torch.nn.Parameter,
) -> _ParameterGuard:
    try:
        return _ParameterGuard(
            name=name,
            aliases=aliases,
            identity=id(parameter),
            version=int(cast(Any, parameter)._version),
            dtype=str(parameter.dtype),
            device=str(parameter.device),
            shape=tuple(parameter.shape),
            stride=tuple(parameter.stride()),
            storage_offset=int(parameter.storage_offset()),
            data_pointer=int(parameter.data_ptr()),
            requires_grad=parameter.requires_grad,
        )
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise StreamingExecutorError("parameter metadata could not be authenticated") from exc


def _prepare_parameter_registry(
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
    authenticated_registry: TrainableParameterRegistry,
) -> tuple[tuple[_ParameterGuard, ...], dict[int, torch.nn.Parameter], str, int]:
    if type(authenticated_registry) is not TrainableParameterRegistry:
        raise StreamingExecutorError("provider trainable-parameter registry lacks exact nominal evidence")
    trainable_records = tuple(
        record for record in authenticated_registry.manifest.records if record.requires_grad
    )
    authenticated_parameters = authenticated_registry.parameters
    if (
        not authenticated_parameters
        or len(trainable_records) != len(authenticated_parameters)
        or authenticated_registry.manifest.trainable_parameter_count != len(authenticated_parameters)
    ):
        raise StreamingExecutorError(
            "provider trainable-parameter registry is empty or internally inconsistent"
        )
    parameters: dict[int, torch.nn.Parameter] = {}
    guards: list[_ParameterGuard] = []
    aliases_by_identity: dict[int, tuple[str, ...]] = {}
    for record, parameter in zip(
        trainable_records,
        authenticated_parameters,
        strict=True,
    ):
        aliases = tuple(record.aliases)
        if (
            not aliases
            or tuple(sorted(set(aliases))) != aliases
            or record.canonical_name != aliases[0]
            or any(not alias or not alias.isascii() for alias in aliases)
            or id(parameter) in parameters
            or not parameter.requires_grad
            or not parameter.is_floating_point()
            or str(parameter.dtype) != record.dtype
            or str(parameter.device) != record.device
            or str(parameter.layout) != record.layout
            or tuple(parameter.shape) != record.shape
            or tuple(parameter.stride()) != record.stride
            or int(parameter.storage_offset()) != record.storage_offset
            or int(parameter.numel()) != record.numel
        ):
            raise StreamingExecutorError(
                "provider trainable-parameter registry differs from its live parameters"
            )
        if parameter.grad is not None:
            raise StreamingExecutorError(
                "all registered parameter gradients must be None before streaming backward"
            )
        identity = id(parameter)
        parameters[identity] = parameter
        aliases_by_identity[identity] = aliases
        guards.append(_parameter_guard(record.canonical_name, aliases, parameter))

    selected = tuple(named_parameters)
    if not selected:
        raise StreamingExecutorError("streaming backward requires a nonempty parameter registry")
    names: set[str] = set()
    supplied_parameter_ids: set[int] = set()
    for item in selected:
        if type(item) is not tuple or len(item) != 2:
            raise StreamingExecutorError("parameter registry entries must be (name, Parameter) tuples")
        name, parameter = item
        if type(name) is not str or not name or not name.isascii():
            raise StreamingExecutorError("parameter names must be nonempty ASCII text")
        if not isinstance(parameter, torch.nn.Parameter):
            raise StreamingExecutorError("parameter registry values must be torch.nn.Parameter objects")
        if name in names:
            raise StreamingExecutorError("parameter registry names must be unique")
        identity = id(parameter)
        expected_aliases = aliases_by_identity.get(identity)
        if expected_aliases is None or name not in expected_aliases:
            raise StreamingExecutorError(
                "caller parameter registry differs from the provider-authenticated trainable set"
            )
        names.add(name)
        supplied_parameter_ids.add(identity)
    if supplied_parameter_ids != set(parameters):
        raise StreamingExecutorError(
            "caller parameter registry omitted a provider-authenticated trainable parameter"
        )
    ordered = tuple(sorted(guards, key=lambda guard: guard.name))
    parameters = {guard.identity: parameters[guard.identity] for guard in ordered}
    return (
        ordered,
        parameters,
        authenticated_registry.manifest.digest,
        sum(len(guard.aliases) for guard in ordered),
    )


def _assert_parameter_metadata(
    guards: tuple[_ParameterGuard, ...],
    parameters: Mapping[int, torch.nn.Parameter],
) -> None:
    for expected in guards:
        parameter = parameters.get(expected.identity)
        if parameter is None or _parameter_guard(expected.name, expected.aliases, parameter) != expected:
            raise StreamingExecutorError("registered parameter identity or metadata changed")


def _gradient_is_finite(gradient: torch.Tensor) -> bool:
    try:
        selected = gradient.coalesce().values() if gradient.is_sparse else gradient
        return bool(torch.isfinite(selected.detach()).all().item())
    except (RuntimeError, TypeError, ValueError) as exc:
        raise StreamingExecutorError("an accumulated gradient could not be checked") from exc


def _require_parameter_grads_none(
    guards: tuple[_ParameterGuard, ...],
    parameters: Mapping[int, torch.nn.Parameter],
) -> None:
    for guard in guards:
        if parameters[guard.identity].grad is not None:
            raise StreamingExecutorError("parameter gradients must remain strictly None until final commit")


def _raw_tensor_bytes(tensor: torch.Tensor) -> bytes:
    try:
        return (
            tensor.detach()
            .to(device="cpu")
            .contiguous()
            .reshape(-1)
            .view(torch.uint8)
            .numpy()
            .tobytes(order="C")
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise StreamingExecutorError("gradient bytes could not be materialized") from exc


def _gradient_byte_records(
    guards: tuple[_ParameterGuard, ...],
    parameters: Mapping[int, torch.nn.Parameter],
    tensors: Mapping[int, torch.Tensor],
) -> tuple[ParameterGradientByteRecord, ...]:
    records: list[ParameterGradientByteRecord] = []
    for guard in guards:
        tensor = tensors.get(guard.identity)
        if tensor is None:
            raise StreamingExecutorError("gradient byte manifest omitted a canonical parameter")
        if tensor.requires_grad or tensor.grad_fn is not None or not _gradient_is_finite(tensor):
            raise StreamingExecutorError("gradient byte manifest contains invalid tensor evidence")
        parameter = parameters[guard.identity]
        records.append(
            ParameterGradientByteRecord(
                name=guard.name,
                aliases=guard.aliases,
                parameter_dtype=str(parameter.dtype),
                parameter_device=str(parameter.device),
                parameter_shape=tuple(parameter.shape),
                gradient_dtype=str(tensor.dtype),
                gradient_device=str(tensor.device),
                sha256=hashlib.sha256(_raw_tensor_bytes(tensor)).hexdigest(),
            )
        )
    return tuple(records)


class _FP32GradientAccumulator:
    """Canonical first-order accumulation without touching ``Parameter.grad``."""

    __slots__ = (
        "buffers",
        "committed_gradients",
        "contribution_cast_count",
        "final_cast_count",
        "guards",
        "parameters",
        "reached_parameter_ids",
    )

    def __init__(
        self,
        guards: tuple[_ParameterGuard, ...],
        parameters: Mapping[int, torch.nn.Parameter],
    ) -> None:
        self.guards = guards
        self.parameters = parameters
        self.buffers = {
            guard.identity: torch.zeros_like(
                parameters[guard.identity],
                dtype=torch.float32,
                memory_format=torch.preserve_format,
            )
            for guard in guards
        }
        self.reached_parameter_ids: set[int] = set()
        self.contribution_cast_count = 0
        self.final_cast_count = 0
        self.committed_gradients: dict[int, torch.Tensor] = {}

    @property
    def parameter_ids(self) -> frozenset[int]:
        return frozenset(guard.identity for guard in self.guards)

    def accumulate(self, loss: torch.Tensor) -> tuple[frozenset[int], int]:
        _require_parameter_grads_none(self.guards, self.parameters)
        ordered_parameters = tuple(self.parameters[guard.identity] for guard in self.guards)
        contributions = cast(
            tuple[torch.Tensor | None, ...],
            torch.autograd.grad(
                loss,
                ordered_parameters,
                allow_unused=True,
                create_graph=False,
                retain_graph=False,
            ),
        )
        reached: set[int] = set()
        cast_count = 0
        try:
            for guard, parameter, contribution in zip(
                self.guards,
                ordered_parameters,
                contributions,
                strict=True,
            ):
                if contribution is None:
                    continue
                if (
                    contribution.requires_grad
                    or contribution.grad_fn is not None
                    or tuple(contribution.shape) != tuple(parameter.shape)
                ):
                    raise StreamingExecutorError("per-turn parameter gradient contribution is invalid")
                if not _gradient_is_finite(contribution):
                    raise StreamingExecutorError("per-turn parameter gradient is non-finite")
                detached = contribution.detach()
                if detached.is_sparse:
                    detached = detached.coalesce().to_dense()
                fp32 = detached.to(device=parameter.device, dtype=torch.float32)
                if not _gradient_is_finite(fp32):
                    raise StreamingExecutorError("per-turn FP32 gradient contribution is non-finite")
                with torch.no_grad():
                    self.buffers[guard.identity].add_(fp32)
                reached.add(guard.identity)
                cast_count += 1
        finally:
            contributions = ()
        self.reached_parameter_ids.update(reached)
        self.contribution_cast_count += cast_count
        _require_parameter_grads_none(self.guards, self.parameters)
        return frozenset(reached), cast_count

    def finalize(
        self,
        *,
        require_all: bool,
    ) -> tuple[
        tuple[ParameterGradientByteRecord, ...],
        tuple[ParameterGradientByteRecord, ...],
    ]:
        _require_parameter_grads_none(self.guards, self.parameters)
        if not require_all:
            if self.reached_parameter_ids or self.contribution_cast_count:
                raise StreamingExecutorError(
                    "token-empty plan unexpectedly accumulated gradient contributions"
                )
            return (), ()
        missing = self.parameter_ids - self.reached_parameter_ids
        if missing:
            names = tuple(guard.name for guard in self.guards if guard.identity in missing)
            raise StreamingExecutorError(f"registered trainable parameters were unused: {names!r}")
        for buffer in self.buffers.values():
            if (
                buffer.dtype != torch.float32
                or buffer.requires_grad
                or buffer.grad_fn is not None
                or not _gradient_is_finite(buffer)
            ):
                raise StreamingExecutorError("an accumulated FP32 gradient buffer is non-finite or invalid")
        buffer_records = _gradient_byte_records(
            self.guards,
            self.parameters,
            self.buffers,
        )
        finals: dict[int, torch.Tensor] = {}
        for guard in self.guards:
            parameter = self.parameters[guard.identity]
            final = (
                self.buffers[guard.identity]
                .to(device=parameter.device, dtype=parameter.dtype)
                .detach()
                .clone(memory_format=torch.preserve_format)
            )
            if final.requires_grad or final.grad_fn is not None or not _gradient_is_finite(final):
                raise StreamingExecutorError("a final cast parameter gradient is invalid")
            finals[guard.identity] = final
        _require_parameter_grads_none(self.guards, self.parameters)
        for guard in self.guards:
            parameter = self.parameters[guard.identity]
            parameter.grad = finals[guard.identity]
            committed = parameter.grad
            if committed is None:
                raise StreamingExecutorError("final parameter gradient assignment failed")
            self.committed_gradients[guard.identity] = committed
            self.final_cast_count += 1
        final_records = _gradient_byte_records(
            self.guards,
            self.parameters,
            self.committed_gradients,
        )
        return buffer_records, final_records

    def clear_created_final_gradients(self) -> None:
        for identity, gradient in self.committed_gradients.items():
            parameter = self.parameters[identity]
            if parameter.grad is gradient:
                parameter.grad = None
        self.committed_gradients.clear()

    def clear_buffers(self) -> None:
        self.buffers.clear()


def _graph_leaf_parameters(loss: torch.Tensor) -> dict[int, torch.Tensor]:
    root = loss.grad_fn
    if root is None:
        raise StreamingExecutorError("turn loss has no differentiable autograd root")
    pending: list[object] = [root]
    visited: set[int] = set()
    leaves: dict[int, torch.Tensor] = {}
    while pending:
        node = pending.pop()
        if id(node) in visited:
            continue
        visited.add(id(node))
        variable = getattr(node, "variable", None)
        if isinstance(variable, torch.Tensor) and variable.requires_grad:
            leaves[id(variable)] = variable
        next_functions = getattr(node, "next_functions", ())
        for next_function, _ in next_functions:
            if next_function is not None:
                pending.append(next_function)
    if not leaves:
        raise StreamingExecutorError("turn graph does not reach any trainable leaf tensor")
    return leaves


def _provider_lightweight_guard(
    provider: FullByteReauthenticatingProvider,
    plan: StreamingObjectivePlan,
) -> None:
    if provider.policy_state_digest != plan.update_invariants.policy_state_digest:
        raise StreamingExecutorError("provider policy state differs from the streaming plan")
    if provider.model_provenance_digest != plan.update_invariants.model_provenance_digest:
        raise StreamingExecutorError("provider model provenance differs from the streaming plan")


def _validate_full_provenance(
    provenance: ModelPolicyProvenance,
    plan: StreamingObjectivePlan,
) -> None:
    invariants = plan.update_invariants
    if type(provenance) is not ModelPolicyProvenance:
        raise StreamingExecutorError("full-byte reauthentication returned invalid provenance")
    if (
        provenance.model_identifier,
        provenance.revision,
        provenance.artifact_manifest_sha256,
        provenance.runtime_stack_sha256,
        provenance.policy_state_digest,
        provenance.digest,
    ) != (
        invariants.model_identifier,
        invariants.model_revision,
        invariants.artifact_manifest_sha256,
        invariants.runtime_stack_sha256,
        invariants.policy_state_digest,
        invariants.model_provenance_digest,
    ):
        raise StreamingExecutorError("full-byte provider provenance differs from the plan")


def _controller_reason(
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
        raise StreamingExecutorError("controller evidence does not implement the registered protocol")
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
        raise StreamingExecutorError("controller evidence failed during replay") from exc
    if reason is not None and reason not in {"incomplete", "timed_out"}:
        raise StreamingExecutorError("controller returned an unregistered replay reason")
    return reason


def _validate_plan_nominality(
    verified_plan: _VerifiedStreamingObjectivePlan,
) -> StreamingObjectivePlan:
    if type(verified_plan) is not _VerifiedStreamingObjectivePlan:
        raise StreamingExecutorError("executor requires a freshly nominal verified streaming plan")
    plan = verified_plan.plan
    if parse_streaming_objective_plan(plan.to_json()) != plan:
        raise StreamingExecutorError("nominal streaming plan is not structurally canonical")
    regenerated = _VerifiedStreamingObjectivePlan._from_derived(plan)
    if not hmac.compare_digest(regenerated.verification_digest, verified_plan.verification_digest):
        raise StreamingExecutorError("nominal streaming-plan verification digest changed")
    return plan


def _validate_structural_inputs(
    plan: StreamingObjectivePlan,
    records: Sequence[AuthenticatedRolloutRecord],
    episodes: Sequence[HiddenEpisode],
    compiler: FragmentActionTokenCompiler,
    controllers: Mapping[str, RolloutControllerProtocol] | None,
) -> tuple[
    dict[tuple[str, int], AuthenticatedRolloutRecord],
    dict[str, HiddenEpisode],
    dict[str, RolloutControllerProtocol],
]:
    if type(compiler) is not FragmentActionTokenCompiler:
        raise StreamingExecutorError("compiler must be a frozen FragmentActionTokenCompiler")
    manifest = compiler.manifest
    invariants = plan.update_invariants
    if manifest.digest != invariants.compiler_manifest_digest:
        raise StreamingExecutorError("compiler manifest differs from the streaming plan")
    if manifest.tokenizer_identifier != invariants.tokenizer_binding_digest:
        raise StreamingExecutorError("tokenizer binding differs from the streaming plan")

    selected_episodes = tuple(episodes)
    if any(type(episode) is not HiddenEpisode for episode in selected_episodes):
        raise StreamingExecutorError("executor episodes must be exact HiddenEpisode objects")
    episode_map = {episode.digest: episode for episode in selected_episodes}
    if len(episode_map) != len(selected_episodes):
        raise StreamingExecutorError("executor episodes must have unique digests")
    planned_episodes = {group.episode_digest for group in plan.groups}
    if set(episode_map) != planned_episodes:
        raise StreamingExecutorError("executor episodes differ from the complete plan episode set")
    for group in plan.groups:
        episode = episode_map[group.episode_digest]
        if episode.episode_id != group.episode_id:
            raise StreamingExecutorError("hidden episode identity differs from its plan group")

    selected_records = tuple(records)
    if len(selected_records) != plan.rollout_count or any(
        type(record) is not AuthenticatedRolloutRecord for record in selected_records
    ):
        raise StreamingExecutorError("executor requires exactly one structural record per rollout")
    record_map: dict[tuple[str, int], AuthenticatedRolloutRecord] = {}
    for record in selected_records:
        reparsed = parse_authenticated_rollout(record.to_json())
        if reparsed != record:
            raise StreamingExecutorError("rollout record is not uniquely canonical structural JSON")
        coordinate = (record.episode_digest, record.rollout_index)
        if coordinate in record_map:
            raise StreamingExecutorError("executor records contain a duplicate rollout coordinate")
        record_map[coordinate] = record

    expected_coordinates = {
        (group.episode_digest, rollout.rollout_index) for group in plan.groups for rollout in group.rollouts
    }
    if set(record_map) != expected_coordinates:
        raise StreamingExecutorError("executor rollout coordinates differ from the complete plan")

    for group in plan.groups:
        for rollout in group.rollouts:
            record = record_map[(group.episode_digest, rollout.rollout_index)]
            if (
                record.episode_id,
                record.episode_digest,
                record.run_seed,
                record.model_provenance_digest,
                record.policy_state_digest,
                record.tokenizer_binding_digest,
                record.compiler_manifest_digest,
                record.temperature_hex,
                record.absolute_tolerance_hex,
                record.maximum_sequence_tokens,
                record.maximum_turns,
            ) != (
                group.episode_id,
                group.episode_digest,
                group.run_seed,
                invariants.model_provenance_digest,
                invariants.policy_state_digest,
                invariants.tokenizer_binding_digest,
                invariants.compiler_manifest_digest,
                invariants.temperature_hex,
                invariants.absolute_tolerance_hex,
                invariants.maximum_sequence_tokens,
                invariants.maximum_turns,
            ):
                raise StreamingExecutorError("rollout record changed a plan invariant")
            if (
                rollout.record_digest != record.digest
                or rollout.rollout_digest != record.digest
                or rollout.transcript_digest != record.transcript_digest
                or rollout.reward.fraction != Fraction(record.reward_numerator, record.reward_denominator)
                or rollout.action_token_count
                != sum(len(turn.sample.selected_token_ids) for turn in record.turns)
                or len(rollout.turns) != len(record.turns)
            ):
                raise StreamingExecutorError("rollout plan instruction differs from its record")
            if (rollout.turn_zero_abort is not None) is not (not record.turns):
                raise StreamingExecutorError("turn-zero abort evidence differs from the record")
            for instruction, turn in zip(rollout.turns, record.turns, strict=True):
                _validate_turn_structural_binding(instruction, turn)

    controller_map = {} if controllers is None else dict(controllers)
    if any(type(key) is not str or key not in planned_episodes for key in controller_map):
        raise StreamingExecutorError("controller evidence contains an unknown episode digest")
    if any(not isinstance(controller, RolloutControllerProtocol) for controller in controller_map.values()):
        raise StreamingExecutorError("controller evidence has an invalid implementation")
    for record in selected_records:
        termination = record.termination
        if (
            termination is not None
            and termination.origin == "controller"
            and record.episode_digest not in controller_map
        ):
            raise StreamingExecutorError("controller-origin abort lacks required replay evidence")
    return record_map, episode_map, controller_map


def _validate_turn_structural_binding(
    instruction: StreamingTurnInstruction,
    turn: object,
) -> None:
    sample = getattr(turn, "sample", None)
    if sample is None:
        raise StreamingExecutorError("rollout turn omitted its authenticated sample")
    if (
        instruction.turn_index != getattr(turn, "turn_index", None)
        or instruction.rollout_turn_digest != getattr(turn, "digest", None)
        or instruction.dialogue_digest != getattr(turn, "dialogue_digest", None)
        or instruction.sample_digest != sample.digest
        or instruction.sample_verification_digest != getattr(turn, "nominal_verification_digest", None)
        or instruction.decision_example_digest != sample.decision_example.digest
        or instruction.decision_verification_digest != sample.decision_verification_digest
        or instruction.action_trace_digest != sample.decision_example.action_trace.digest
        or instruction.detached_statistics_digest != sample.detached_statistics.digest
        or instruction.action_token_count != len(sample.selected_token_ids)
    ):
        raise StreamingExecutorError("turn plan instruction differs from structural action evidence")


def _rollout_verification_digest(
    record: AuthenticatedRolloutRecord,
    verification_digests: tuple[str, ...],
) -> str:
    if len(verification_digests) != len(record.turns):
        raise StreamingExecutorError("fresh rollout verification evidence does not align")
    return json_digest(
        {
            "record_digest": record.digest,
            "verified_samples": [
                {
                    "sample_digest": turn.sample.digest,
                    "verification_digest": verification_digest,
                }
                for turn, verification_digest in zip(record.turns, verification_digests, strict=True)
            ],
        },
        domain=_ROLLOUT_VERIFICATION_DIGEST_DOMAIN,
    )


def _sampler_group_digest(
    records: tuple[AuthenticatedRolloutRecord, ...],
    fresh_verifications: Mapping[tuple[int, int], str],
) -> str:
    samples = [(record, turn) for record in records for turn in record.turns]
    if not samples:
        raise StreamingExecutorError("nonempty sampler group unexpectedly has no samples")
    reference = samples[0][1].sample
    answer_terminal_counts = {
        turn.sample.terminal_count for _, turn in samples if turn.sample.mode == "answer"
    }
    if len(answer_terminal_counts) > 1:
        raise StreamingExecutorError("fresh sampler group changed answer terminal count")
    ordered = sorted(
        samples,
        key=lambda pair: (pair[0].rollout_index, pair[1].turn_index),
    )
    return json_digest(
        {
            "run_seed": records[0].run_seed,
            "episode_digest": records[0].episode_digest,
            "policy_state_digest": reference.policy_state_digest,
            "tokenizer_binding_digest": reference.tokenizer_binding_digest,
            "compiler_manifest_digest": reference.compiler_manifest_digest,
            "temperature_hex": reference.temperature_hex,
            "answer_terminal_count": (
                None if not answer_terminal_counts else next(iter(answer_terminal_counts))
            ),
            "absolute_tolerance_hex": records[0].absolute_tolerance_hex,
            "verified_samples": [
                {
                    "sample_digest": turn.sample.digest,
                    "verification_digest": fresh_verifications[(record.rollout_index, turn.turn_index)],
                }
                for record, turn in ordered
            ],
        },
        domain=_SAMPLER_GROUP_DIGEST_DOMAIN,
    )


def _fresh_group_digests(
    group: StreamingGroupInstruction,
    records: tuple[AuthenticatedRolloutRecord, ...],
    fresh_verifications: Mapping[tuple[int, int], str],
    rollout_verifications: Mapping[int, str],
) -> tuple[str, str]:
    reference = records[0]
    has_empty_rollout = any(not record.turns for record in records)
    sampler_digest = None if has_empty_rollout else _sampler_group_digest(records, fresh_verifications)
    action_digest = json_digest(
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
            "sampler_group_digest": sampler_digest,
            "rollouts": [
                (
                    {
                        "kind": "verified_action_samples",
                        "rollout_index": record.rollout_index,
                        "samples": [
                            {
                                "turn_index": turn.turn_index,
                                "sample_digest": turn.sample.digest,
                                "verification_digest": fresh_verifications[
                                    (record.rollout_index, turn.turn_index)
                                ],
                            }
                            for turn in record.turns
                        ],
                    }
                    if record.turns
                    else {
                        "kind": "verified_zero_turn_abort",
                        "rollout_index": record.rollout_index,
                        "record_digest": record.digest,
                        "rollout_verification_digest": rollout_verifications[record.rollout_index],
                        "termination": cast(Any, record.termination).as_obj(),
                        "reward": {"numerator": 0, "denominator": 1},
                    }
                )
                for record in records
            ],
        },
        domain=_ROLLOUT_ACTION_EVIDENCE_GROUP_DIGEST_DOMAIN,
    )
    group_digest = json_digest(
        {
            "action_evidence_group_digest": action_digest,
            "rollouts": [
                {
                    "rollout_index": record.rollout_index,
                    "record_digest": record.digest,
                    "verification_digest": rollout_verifications[record.rollout_index],
                }
                for record in records
            ],
        },
        domain=_ROLLOUT_GROUP_DIGEST_DOMAIN,
    )
    return action_digest, group_digest


def _execute_turn_backward(
    *,
    coordinate: ReplayCoordinate,
    instruction: StreamingTurnInstruction,
    record: AuthenticatedRolloutRecord,
    turn: Any,
    environment: HiddenLawEnvironment,
    dialogue: Dialogue,
    provider: FullByteReauthenticatingProvider,
    tokenizer: ExactDecodeTokenizerProtocol,
    compiler: FragmentActionTokenCompiler,
    gradient_accumulator: _FP32GradientAccumulator,
    graph_counter: _GraphCounter,
) -> _TurnBackwardResult:
    verified: Any = None
    statistics: Any = None
    policy_loss: torch.Tensor | None = None
    entropy_loss: torch.Tensor | None = None
    turn_loss: torch.Tensor | None = None
    anchor_reference: weakref.ReferenceType[torch.Tensor] | None = None
    graph_opened = False
    try:
        verified = replay_authenticated_sample(
            turn.sample,
            provider,
            tokenizer,
            compiler,
            dialogue,
            absolute_tolerance=record.absolute_tolerance,
        )
        if verified.verification_digest != instruction.sample_verification_digest:
            raise StreamingExecutorError("fresh action verification digest differs from the plan")
        if verified.verification_digest != turn.nominal_verification_digest:
            raise StreamingExecutorError("fresh action verification digest differs from the record")
        _validate_turn_structural_binding(instruction, turn)
        statistics = verified.replayed_statistics
        if statistics.action_token_count != instruction.action_token_count:
            raise StreamingExecutorError("fresh replay action-token count differs from the plan")
        result = environment.consume(turn.sample.decision_example.action_trace.raw_action)
        if result.event != turn.event:
            raise StreamingExecutorError("fresh action produced a different environment event")

        policy_coefficient = _scalar_float(instruction.policy_coefficient)
        entropy_coefficient = _scalar_float(instruction.entropy_token_coefficient)
        policy_loss = statistics.sequence_log_probability * policy_coefficient
        entropy_loss = statistics.token_entropies.sum() * entropy_coefficient
        turn_loss = policy_loss + entropy_loss
        if turn_loss.ndim != 0 or not turn_loss.requires_grad:
            raise StreamingExecutorError("authenticated turn loss is not a differentiable scalar")
        if not bool(torch.isfinite(turn_loss.detach()).item()):
            raise StreamingExecutorError("authenticated turn loss is non-finite")
        leaves = _graph_leaf_parameters(turn_loss)
        unexpected_leaves = set(leaves) - gradient_accumulator.parameter_ids
        if unexpected_leaves:
            raise StreamingExecutorError("turn graph reaches an unregistered trainable leaf")

        anchor_reference = graph_counter.open(coordinate, turn_loss)
        graph_opened = True
        detached = turn.sample.detached_statistics
        replayed_log_probabilities = tuple(
            float(value)
            for value in statistics.token_log_probabilities.detach().to(device="cpu", dtype=torch.float64)
        )
        replayed_entropies = tuple(
            float(value)
            for value in statistics.token_entropies.detach().to(device="cpu", dtype=torch.float64)
        )
        replayed_policy = policy_coefficient * math.fsum(replayed_log_probabilities)
        replayed_entropy = entropy_coefficient * math.fsum(replayed_entropies)
        stored_policy = policy_coefficient * math.fsum(detached.token_log_probabilities)
        stored_entropy = entropy_coefficient * math.fsum(detached.token_entropies)
        comparison_bound = (
            record.absolute_tolerance
            * instruction.action_token_count
            * (abs(policy_coefficient) + abs(entropy_coefficient))
        )
        comparison_error = abs(
            math.fsum((stored_policy, stored_entropy)) - math.fsum((replayed_policy, replayed_entropy))
        )
        comparison_slack = 16 * math.ulp(
            max(
                1.0,
                abs(stored_policy),
                abs(stored_entropy),
                abs(replayed_policy),
                abs(replayed_entropy),
                abs(comparison_bound),
            )
        )
        if comparison_error > comparison_bound + comparison_slack:
            raise StreamingExecutorError(
                "aggregate stored/replayed turn loss exceeds the per-token tolerance bound"
            )
        reached_parameters, cast_count = gradient_accumulator.accumulate(turn_loss)
        if reached_parameters != frozenset(leaves):
            raise StreamingExecutorError(
                "autograd contributions differ from authenticated trainable graph leaves"
            )
        return _TurnBackwardResult(
            verification_digest=verified.verification_digest,
            token_count=instruction.action_token_count,
            stored_policy_loss=stored_policy,
            stored_entropy_loss=stored_entropy,
            replayed_policy_loss=replayed_policy,
            replayed_entropy_loss=replayed_entropy,
            stored_replay_absolute_error_bound=comparison_bound,
            graph_leaf_parameter_ids=frozenset(leaves),
            fp32_contribution_cast_count=cast_count,
        )
    except AuthenticatedSamplingError as exc:
        raise StreamingExecutorError("fresh authenticated action replay failed") from exc
    finally:
        verified = None
        statistics = None
        policy_loss = None
        entropy_loss = None
        turn_loss = None
        gc.collect()
        if graph_opened and anchor_reference is not None:
            graph_counter.close(coordinate, anchor_reference)


def _authenticate_termination(
    *,
    record: AuthenticatedRolloutRecord,
    environment: HiddenLawEnvironment,
    episode: HiddenEpisode,
    provider: FullByteReauthenticatingProvider,
    tokenizer: ExactDecodeTokenizerProtocol,
    compiler: FragmentActionTokenCompiler,
    controller: RolloutControllerProtocol | None,
) -> None:
    termination = record.termination
    if termination is None:
        return
    dialogue, dialogue_digest = _dialogue_digest(episode, environment.transcript)
    if dialogue_digest != termination.dialogue_digest:
        raise StreamingExecutorError("termination dialogue does not regenerate exactly")
    controller_reason = _controller_reason(
        controller,
        episode_digest=episode.digest,
        rollout_index=record.rollout_index,
        turn_index=termination.turn_index,
        dialogue_digest=dialogue_digest,
    )
    if termination.origin == "controller":
        if controller is None or controller_reason != termination.reason:
            raise StreamingExecutorError("controller termination cause does not reauthenticate")
    elif controller_reason is not None:
        raise StreamingExecutorError("controller evidence reports an abort omitted by the record")
    if termination.origin == "sampler":
        mode: Literal["inquiry", "answer"] = "inquiry" if environment.state == "inquiry" else "answer"
        terminal_count = None if mode == "inquiry" else len(episode.terminal)
        seed = PolicyTurnSeed(
            run_seed=record.run_seed,
            episode_digest=record.episode_digest,
            rollout_index=record.rollout_index,
            turn_index=termination.turn_index,
            policy_state_digest=record.policy_state_digest,
        )
        try:
            sample_authenticated_action(
                provider,
                tokenizer,
                compiler,
                dialogue,
                seed,
                mode=mode,
                terminal_count=terminal_count,
                temperature=record.temperature,
                maximum_sequence_tokens=record.maximum_sequence_tokens,
            )
        except AuthenticatedSamplingOverlengthError:
            pass
        except AuthenticatedSamplingError as exc:
            raise StreamingExecutorError("sampler termination reproduced as a generic fatal failure") from exc
        else:
            raise StreamingExecutorError("sampler overlength termination did not reproduce")
    result = environment.abort(termination.reason)
    if result.event != AbortEvent(termination.reason):
        raise StreamingExecutorError("termination produced a different abort event")


def execute_verified_streaming_backward(
    verified_plan: _VerifiedStreamingObjectivePlan,
    records: Sequence[AuthenticatedRolloutRecord],
    episodes: Sequence[HiddenEpisode],
    provider: FullByteReauthenticatingProvider,
    tokenizer: ExactDecodeTokenizerProtocol,
    compiler: FragmentActionTokenCompiler,
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
    *,
    controllers: Mapping[str, RolloutControllerProtocol] | None = None,
    graph_lifecycle_observer: GraphLifecycleObserver | None = None,
) -> StreamingExecutionDiagnostics:
    """Freshly reauthenticate and backpropagate one canonical turn at a time.

    Registered gradients must all be ``None`` on entry.  Success leaves finite
    accumulated gradients but does not change parameter bytes or optimizer
    state.  Any failure after execution begins clears only gradients created by
    this call and emits no diagnostic record.
    """

    plan = _validate_plan_nominality(verified_plan)
    if not isinstance(provider, FullByteReauthenticatingProvider):
        raise StreamingExecutorError("provider lacks provenanced full-byte reauthentication")
    if not isinstance(tokenizer, ExactDecodeTokenizerProtocol):
        raise StreamingExecutorError("tokenizer does not implement exact decode")
    if graph_lifecycle_observer is not None and not isinstance(
        graph_lifecycle_observer, GraphLifecycleObserver
    ):
        raise StreamingExecutorError("graph lifecycle observer has an invalid implementation")
    record_map, episode_map, controller_map = _validate_structural_inputs(
        plan, records, episodes, compiler, controllers
    )
    (
        parameter_guards,
        parameter_map,
        parameter_registry_digest,
        registered_parameter_name_count,
    ) = _prepare_parameter_registry(
        named_parameters,
        provider.trainable_parameter_registry,
    )
    gradient_accumulator = _FP32GradientAccumulator(parameter_guards, parameter_map)
    parameter_ids = gradient_accumulator.parameter_ids
    graph_counter = _GraphCounter(graph_lifecycle_observer)

    turn_replays = 0
    termination_replays = 0
    turn_zero_aborts = 0
    backward_calls = 0
    lightweight_guards = 0
    seen_graph_parameters: set[int] = set()
    stored_policy_terms: list[float] = []
    stored_entropy_terms: list[float] = []
    replayed_policy_terms: list[float] = []
    replayed_entropy_terms: list[float] = []
    stored_replay_error_bounds: list[float] = []
    execution_started = False
    success = False
    try:
        execution_started = True
        pre_provenance = provider.reauthenticate_policy_state()
        _validate_full_provenance(pre_provenance, plan)
        _provider_lightweight_guard(provider, plan)
        lightweight_guards += 1
        _assert_parameter_metadata(parameter_guards, parameter_map)

        for group in plan.groups:
            episode = episode_map[group.episode_digest]
            group_records = tuple(
                record_map[(group.episode_digest, rollout.rollout_index)] for rollout in group.rollouts
            )
            fresh_group_verifications: dict[tuple[int, int], str] = {}
            fresh_rollout_verifications: dict[int, str] = {}
            for rollout_instruction, record in zip(group.rollouts, group_records, strict=True):
                environment = HiddenLawEnvironment(episode)
                controller = controller_map.get(group.episode_digest)
                verification_digests: list[str] = []
                _provider_lightweight_guard(provider, plan)
                lightweight_guards += 1
                for turn_instruction, turn in zip(rollout_instruction.turns, record.turns, strict=True):
                    dialogue, dialogue_digest = _dialogue_digest(episode, environment.transcript)
                    if (
                        dialogue_digest != turn.dialogue_digest
                        or dialogue_digest != turn_instruction.dialogue_digest
                    ):
                        raise StreamingExecutorError("turn dialogue does not regenerate exactly")
                    if (
                        _controller_reason(
                            controller,
                            episode_digest=episode.digest,
                            rollout_index=record.rollout_index,
                            turn_index=turn.turn_index,
                            dialogue_digest=dialogue_digest,
                        )
                        is not None
                    ):
                        raise StreamingExecutorError(
                            "controller evidence requested an abort before a recorded action"
                        )
                    _provider_lightweight_guard(provider, plan)
                    lightweight_guards += 1
                    coordinate = ReplayCoordinate(
                        episode_digest=episode.digest,
                        rollout_index=record.rollout_index,
                        turn_index=turn.turn_index,
                    )
                    result = _execute_turn_backward(
                        coordinate=coordinate,
                        instruction=turn_instruction,
                        record=record,
                        turn=turn,
                        environment=environment,
                        dialogue=dialogue,
                        provider=provider,
                        tokenizer=tokenizer,
                        compiler=compiler,
                        gradient_accumulator=gradient_accumulator,
                        graph_counter=graph_counter,
                    )
                    verification_digests.append(result.verification_digest)
                    fresh_group_verifications[(record.rollout_index, turn.turn_index)] = (
                        result.verification_digest
                    )
                    seen_graph_parameters.update(result.graph_leaf_parameter_ids)
                    stored_policy_terms.append(result.stored_policy_loss)
                    stored_entropy_terms.append(result.stored_entropy_loss)
                    replayed_policy_terms.append(result.replayed_policy_loss)
                    replayed_entropy_terms.append(result.replayed_entropy_loss)
                    stored_replay_error_bounds.append(result.stored_replay_absolute_error_bound)
                    turn_replays += 1
                    backward_calls += 1
                    _assert_parameter_metadata(parameter_guards, parameter_map)
                    _require_parameter_grads_none(parameter_guards, parameter_map)
                    _provider_lightweight_guard(provider, plan)
                    lightweight_guards += 1

                if record.termination is not None:
                    _authenticate_termination(
                        record=record,
                        environment=environment,
                        episode=episode,
                        provider=provider,
                        tokenizer=tokenizer,
                        compiler=compiler,
                        controller=controller,
                    )
                    termination_replays += 1
                    if not record.turns:
                        turn_zero_aborts += 1
                if environment.transcript != record.transcript:
                    raise StreamingExecutorError("fresh rollout transcript differs from the record")
                reward = _terminal_reward(environment.transcript)
                if (reward.numerator, reward.denominator) != (
                    record.reward_numerator,
                    record.reward_denominator,
                ):
                    raise StreamingExecutorError("fresh rollout reward differs from the record")
                fresh_rollout_digest = _rollout_verification_digest(record, tuple(verification_digests))
                if fresh_rollout_digest != rollout_instruction.rollout_verification_digest:
                    raise StreamingExecutorError("fresh rollout verification digest differs from the plan")
                fresh_rollout_verifications[record.rollout_index] = fresh_rollout_digest
                _provider_lightweight_guard(provider, plan)
                lightweight_guards += 1

            action_digest, group_digest = _fresh_group_digests(
                group,
                group_records,
                fresh_group_verifications,
                fresh_rollout_verifications,
            )
            if action_digest != group.action_evidence_group_digest:
                raise StreamingExecutorError("fresh group action-evidence digest differs from the plan")
            if group_digest != group.group_digest:
                raise StreamingExecutorError("fresh verified group digest differs from the plan")

        if turn_replays != sum(len(rollout.turns) for group in plan.groups for rollout in group.rollouts):
            raise StreamingExecutorError("authenticated turn replay count differs from the plan")
        if (
            sum(
                turn.action_token_count
                for group in plan.groups
                for rollout in group.rollouts
                for turn in rollout.turns
            )
            != plan.total_authenticated_action_token_count
        ):
            raise StreamingExecutorError("authenticated action-token count differs from the plan")
        if backward_calls != turn_replays or graph_counter.live != 0:
            raise StreamingExecutorError("bounded graph/backward accounting is inconsistent")
        if plan.optimizer_step_eligible:
            if seen_graph_parameters != parameter_ids or gradient_accumulator.reached_parameter_ids != set(
                parameter_ids
            ):
                raise StreamingExecutorError(
                    "not every registered parameter was reached by authenticated turn graphs"
                )
        else:
            if (
                seen_graph_parameters
                or backward_calls
                or gradient_accumulator.reached_parameter_ids
                or gradient_accumulator.contribution_cast_count
            ):
                raise StreamingExecutorError("token-empty plan unexpectedly created a graph")
        _require_parameter_grads_none(parameter_guards, parameter_map)

        _assert_parameter_metadata(parameter_guards, parameter_map)
        post_provenance = provider.reauthenticate_policy_state()
        _validate_full_provenance(post_provenance, plan)
        if post_provenance != pre_provenance:
            raise StreamingExecutorError("provider provenance changed across the whole update")
        _provider_lightweight_guard(provider, plan)
        lightweight_guards += 1
        _assert_parameter_metadata(parameter_guards, parameter_map)
        _require_parameter_grads_none(parameter_guards, parameter_map)

        stored_policy_loss = math.fsum(stored_policy_terms)
        stored_entropy_loss = math.fsum(stored_entropy_terms)
        replayed_policy_loss = math.fsum(replayed_policy_terms)
        replayed_entropy_loss = math.fsum(replayed_entropy_terms)
        aggregate_error_bound = math.fsum(stored_replay_error_bounds)
        aggregate_error = abs(
            math.fsum((stored_policy_loss, stored_entropy_loss))
            - math.fsum((replayed_policy_loss, replayed_entropy_loss))
        )
        aggregate_slack = 32 * math.ulp(
            max(
                1.0,
                abs(stored_policy_loss),
                abs(stored_entropy_loss),
                abs(replayed_policy_loss),
                abs(replayed_entropy_loss),
                abs(aggregate_error_bound),
            )
        )
        if aggregate_error > aggregate_error_bound + aggregate_slack:
            raise StreamingExecutorError(
                "aggregate stored/replayed update loss exceeds the per-token tolerance bound"
            )

        fp32_gradient_buffers, final_gradients = gradient_accumulator.finalize(
            require_all=plan.optimizer_step_eligible
        )
        finite_gradient_count = len(final_gradients)
        fp32_gradient_buffer_manifest_digest = json_digest(
            [record.as_obj() for record in fp32_gradient_buffers],
            domain=_FP32_BUFFER_MANIFEST_DOMAIN,
        )
        final_gradient_manifest_digest = json_digest(
            [record.as_obj() for record in final_gradients],
            domain=_FINAL_GRADIENT_MANIFEST_DOMAIN,
        )

        diagnostics = StreamingExecutionDiagnostics._from_execution(
            plan=plan,
            plan_verification_digest=verified_plan.verification_digest,
            parameter_registry_digest=parameter_registry_digest,
            registered_parameter_name_count=registered_parameter_name_count,
            authenticated_turn_replay_count=turn_replays,
            authenticated_termination_replay_count=termination_replays,
            authenticated_turn_zero_abort_count=turn_zero_aborts,
            backward_call_count=backward_calls,
            parameter_count=len(parameter_map),
            finite_gradient_parameter_count=finite_gradient_count,
            fp32_contribution_cast_count=gradient_accumulator.contribution_cast_count,
            final_gradient_cast_count=gradient_accumulator.final_cast_count,
            fp32_gradient_buffers=fp32_gradient_buffers,
            fp32_gradient_buffer_manifest_digest=(fp32_gradient_buffer_manifest_digest),
            final_gradients=final_gradients,
            final_gradient_manifest_digest=final_gradient_manifest_digest,
            full_byte_reauthentication_count=2,
            explicit_lightweight_guard_count=lightweight_guards,
            maximum_live_graph_count=graph_counter.maximum,
            released_graph_anchor_count=graph_counter.released,
            stored_policy_loss=stored_policy_loss,
            stored_entropy_loss=stored_entropy_loss,
            replayed_policy_loss=replayed_policy_loss,
            replayed_entropy_loss=replayed_entropy_loss,
            stored_replay_absolute_error_bound=aggregate_error_bound,
        )
        success = True
        return diagnostics
    except Exception as exc:
        if isinstance(exc, StreamingExecutorError):
            raise
        raise StreamingExecutorError("authenticated streaming backward failed closed") from exc
    finally:
        if execution_started and not success:
            gradient_accumulator.clear_created_final_gradients()
        gradient_accumulator.clear_buffers()


def streaming_executor_manifest() -> dict[str, object]:
    """Describe the fixed nonauthorizing semantics of this executor slice."""

    return {
        "schema_version": STREAMING_EXECUTOR_SCHEMA_VERSION,
        "contract_id": STREAMING_EXECUTOR_CONTRACT_ID,
        "accepted_plan": "fresh _VerifiedStreamingObjectivePlan nominal wrapper only",
        "accepted_rollouts": "canonical structural AuthenticatedRolloutRecord values",
        "provider_authentication": "full bytes before and after; lightweight guards throughout",
        "parameter_registry": (
            "provider-issued full trainable set; tied aliases canonical; caller identities exact"
        ),
        "gradient_entry_contract": "all registered parameter grads are None",
        "graph_scope": "one freshly reauthenticated action turn",
        "backward_primitive": "torch.autograd.grad",
        "backward_order": "canonical group/rollout/turn then deduplicated parameter-name order",
        "accumulation_dtype": "torch.float32",
        "parameter_grad_during_replay": "strictly None",
        "final_gradient_assignment": "one cast and assignment per reached parameter after all replay",
        "gradient_byte_manifests": "FP32 buffers and final parameter gradients",
        "numerical_authentication": "per-token replay comparison plus derived aggregate bound",
        "termination_authentication": "controller, typed sampler overlength, or turn budget",
        "turn_zero_abort_graph_count": 0,
        "failure_gradient_action": "clear only gradients created by this call",
        "optimizer_present": False,
        "optimizer_step_called": False,
        "parameter_data_mutation": False,
        "live_or_network_code_present": False,
        "model_load_authorization": False,
        "rollout_launch_authorization": False,
        "backward_execution_authorization": False,
        "weight_update_authorization": False,
        "optimizer_step_authorization": False,
    }
