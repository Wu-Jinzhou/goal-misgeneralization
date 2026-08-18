"""Smoke-only atomic optimizer boundary for authenticated G03 backward passes.

This module is deliberately nonauthorizing.  It can execute exactly one
internally owned AdamW step only while holding a private-issuance sealed-Qwen
update lease.  A successful call returns a canonical record after a fresh
post-update provider has been prepared and the record has been fully
serialized.  Every failure restores byte-exact parameter snapshots or marks
the runtime corrupt before releasing the lease.
"""

from __future__ import annotations

import hashlib
import hmac
import inspect
import json
import math
import textwrap
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

import torch

from ._json import dump_json, json_digest, load_json
from .authenticated_model_provider_v2 import TensorStateManifest, TrainableParameterRegistry
from .authenticated_rollouts_v2 import AuthenticatedRolloutRecord
from .episodes import HiddenEpisode
from .sealed_runtime_v1 import (
    SealedQwenPreparedUpdate,
    SealedQwenRuntime,
    SealedQwenRuntimeError,
    SealedQwenRuntimeManifest,
    SealedQwenUpdateSession,
    require_exact_sealed_qwen_runtime,
)
from .streaming_executor_v3 import (
    StreamingExecutionDiagnostics,
    execute_verified_streaming_backward,
)
from .streaming_objective_v3 import _VerifiedStreamingObjectivePlan
from .streaming_sft_v3 import (
    ReferenceTrajectorySFTSource,
    StreamingSFTExecution,
    _VerifiedStreamingSFTPlan,
    execute_streaming_sft_backward,
)

AUTHENTICATED_STEP_SCHEMA_VERSION = 1
AUTHENTICATED_STEP_CONTRACT_ID = "goalzendo-authenticated-atomic-step-v1"
AUTHENTICATED_STEP_AUTHORIZES_EXECUTION = False

_OPTIMIZER_SPEC_DOMAIN = "goalzendo-interactive-optimizer-spec-v1"
_OPTIMIZER_STATE_DOMAIN = "goalzendo-interactive-optimizer-state-v1"
_PARAMETER_BYTES_DOMAIN = "goalzendo-interactive-step-parameter-bytes-v1"
_SOURCE_DOMAIN = "goalzendo-interactive-step-sources-v1"
_COORDINATE_DOMAIN = "goalzendo-interactive-step-coordinate-v1"
_STEP_RECORD_DOMAIN = "goalzendo-interactive-atomic-step-record-v1"

ObjectiveKind: TypeAlias = Literal["sft", "rl"]


class AuthenticatedStepError(RuntimeError):
    """Raised when an atomic smoke step fails closed."""


def _sha256(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise AuthenticatedStepError(f"{name} must be a lowercase SHA-256")
    return value


def _finite_float(value: object, *, name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AuthenticatedStepError(f"{name} must be a finite number")
    selected = float(value)
    if not math.isfinite(selected) or (minimum is not None and selected < minimum):
        raise AuthenticatedStepError(f"{name} is outside its registered finite domain")
    return selected


@dataclass(frozen=True, slots=True)
class AuthenticatedUpdateCoordinate:
    study_id: str
    run_id: str
    update_index: int
    objective_kind: ObjectiveKind

    def __post_init__(self) -> None:
        for value in (self.study_id, self.run_id):
            if type(value) is not str or not value or not value.isascii():
                raise AuthenticatedStepError("update coordinate labels must be nonempty ASCII")
        if (
            isinstance(self.update_index, bool)
            or not isinstance(self.update_index, int)
            or self.update_index < 0
        ):
            raise AuthenticatedStepError("update_index must be non-negative")
        if self.objective_kind not in {"sft", "rl"}:
            raise AuthenticatedStepError("objective_kind must be sft or rl")

    def as_obj(self) -> dict[str, object]:
        return {
            "study_id": self.study_id,
            "run_id": self.run_id,
            "update_index": self.update_index,
            "objective_kind": self.objective_kind,
        }

    @property
    def digest(self) -> str:
        return json_digest(self.as_obj(), domain=_COORDINATE_DOMAIN)


@dataclass(frozen=True, slots=True)
class AdamWOptimizerSpec:
    learning_rate: float
    beta1: float
    beta2: float
    epsilon: float
    weight_decay: float
    gradient_clip_norm: None = None
    scheduler: Literal["none"] = "none"
    scaler: Literal["none"] = "none"
    accumulation_steps: int = 1
    foreach: bool = False
    fused: bool = False

    def __post_init__(self) -> None:
        for name in ("learning_rate", "beta1", "beta2", "epsilon", "weight_decay"):
            if type(getattr(self, name)) is not float:
                raise AuthenticatedStepError(f"{name} must have exact type float")
        learning_rate = _finite_float(self.learning_rate, name="learning_rate", minimum=0.0)
        beta1 = _finite_float(self.beta1, name="beta1", minimum=0.0)
        beta2 = _finite_float(self.beta2, name="beta2", minimum=0.0)
        epsilon = _finite_float(self.epsilon, name="epsilon", minimum=0.0)
        _finite_float(self.weight_decay, name="weight_decay", minimum=0.0)
        if learning_rate <= 0 or epsilon <= 0 or not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise AuthenticatedStepError("AdamW scalar configuration is invalid")
        if self.gradient_clip_norm is not None:
            raise AuthenticatedStepError(
                "atomic-step v1 requires gradient_clip_norm to be explicitly None"
            )
        if (
            type(self.scheduler) is not str
            or type(self.scaler) is not str
            or self.scheduler != "none"
            or self.scaler != "none"
        ):
            raise AuthenticatedStepError("atomic-step v1 requires no scheduler and no scaler")
        if type(self.accumulation_steps) is not int or self.accumulation_steps != 1:
            raise AuthenticatedStepError("atomic-step v1 requires one accumulation unit")
        if type(self.foreach) is not bool or self.foreach:
            raise AuthenticatedStepError("atomic-step v1 requires foreach=False")
        if type(self.fused) is not bool or self.fused:
            raise AuthenticatedStepError("atomic-step v1 requires fused=False")

    def as_obj(self) -> dict[str, object]:
        return {
            "optimizer_class": "torch.optim.AdamW",
            "learning_rate_hex": float(self.learning_rate).hex(),
            "betas_hex": [float(self.beta1).hex(), float(self.beta2).hex()],
            "epsilon_hex": float(self.epsilon).hex(),
            "weight_decay_hex": float(self.weight_decay).hex(),
            "gradient_clip_norm": None,
            "scheduler": self.scheduler,
            "scaler": self.scaler,
            "accumulation_steps": self.accumulation_steps,
            "foreach": self.foreach,
            "fused": self.fused,
            "amsgrad": False,
            "maximize": False,
            "capturable": False,
            "differentiable": False,
        }

    @property
    def digest(self) -> str:
        return json_digest(self.as_obj(), domain=_OPTIMIZER_SPEC_DOMAIN)


@dataclass(frozen=True, slots=True)
class ParameterByteRecord:
    name: str
    aliases: tuple[str, ...]
    dtype: str
    device: str
    shape: tuple[int, ...]
    byte_length: int
    sha256: str

    def __post_init__(self) -> None:
        aliases = tuple(self.aliases)
        object.__setattr__(self, "aliases", aliases)
        if (
            type(self.name) is not str
            or not self.name
            or not aliases
            or any(type(alias) is not str or not alias for alias in aliases)
            or aliases != tuple(sorted(set(aliases)))
            or self.name not in aliases
            or any(type(value) is not str or not value for value in (self.dtype, self.device))
        ):
            raise AuthenticatedStepError("parameter byte-record identity is not canonical")
        shape = tuple(self.shape)
        object.__setattr__(self, "shape", shape)
        if any(type(value) is not int or value < 0 for value in shape):
            raise AuthenticatedStepError("parameter byte-record shape is invalid")
        if type(self.byte_length) is not int or self.byte_length < 0:
            raise AuthenticatedStepError("parameter byte-record length is invalid")
        _sha256(self.sha256, name="parameter byte-record sha256")

    def as_obj(self) -> dict[str, object]:
        return {
            "name": self.name,
            "aliases": list(self.aliases),
            "dtype": self.dtype,
            "device": self.device,
            "shape": list(self.shape),
            "byte_length": self.byte_length,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class OptimizerStateEntry:
    parameter_name: str
    state_key: str
    value_manifest_json: str

    def __post_init__(self) -> None:
        if any(
            type(value) is not str or not value
            for value in (self.parameter_name, self.state_key, self.value_manifest_json)
        ):
            raise AuthenticatedStepError("optimizer state-entry identity is invalid")
        try:
            parsed = load_json(self.value_manifest_json)
        except ValueError as exc:
            raise AuthenticatedStepError("optimizer state-entry JSON is invalid") from exc
        if dump_json(parsed) != self.value_manifest_json:
            raise AuthenticatedStepError("optimizer state-entry JSON is not canonical")

    def as_obj(self) -> dict[str, object]:
        return {
            "parameter_name": self.parameter_name,
            "state_key": self.state_key,
            "value_manifest": json.loads(self.value_manifest_json),
        }


@dataclass(frozen=True, slots=True)
class OptimizerStateManifest:
    optimizer_class: str
    optimizer_spec_digest: str
    optimizer_implementation_digest: str
    resolved_configuration_json: str
    parameter_group_names: tuple[str, ...]
    state_entries: tuple[OptimizerStateEntry, ...]
    update_count: int

    def __post_init__(self) -> None:
        if self.optimizer_class != "torch.optim.AdamW":
            raise AuthenticatedStepError("optimizer-state class is invalid")
        _sha256(self.optimizer_spec_digest, name="optimizer-state spec digest")
        _sha256(
            self.optimizer_implementation_digest,
            name="optimizer-state implementation digest",
        )
        try:
            configuration = load_json(self.resolved_configuration_json)
        except ValueError as exc:
            raise AuthenticatedStepError("optimizer configuration JSON is invalid") from exc
        if dump_json(configuration) != self.resolved_configuration_json:
            raise AuthenticatedStepError("optimizer configuration JSON is not canonical")
        group_names = tuple(self.parameter_group_names)
        entries = tuple(self.state_entries)
        object.__setattr__(self, "parameter_group_names", group_names)
        object.__setattr__(self, "state_entries", entries)
        if (
            not group_names
            or group_names != tuple(sorted(set(group_names)))
            or any(type(name) is not str or not name for name in group_names)
            or any(type(entry) is not OptimizerStateEntry for entry in entries)
            or tuple((entry.parameter_name, entry.state_key) for entry in entries)
            != tuple(sorted((entry.parameter_name, entry.state_key) for entry in entries))
            or any(entry.parameter_name not in group_names for entry in entries)
        ):
            raise AuthenticatedStepError("optimizer state inventory is not canonical")
        if type(self.update_count) is not int or self.update_count < 0:
            raise AuthenticatedStepError("optimizer update count is invalid")

    def as_obj(self) -> dict[str, object]:
        return {
            "optimizer_class": self.optimizer_class,
            "optimizer_spec_digest": self.optimizer_spec_digest,
            "optimizer_implementation_digest": self.optimizer_implementation_digest,
            "resolved_configuration": json.loads(self.resolved_configuration_json),
            "parameter_group_names": list(self.parameter_group_names),
            "state_entries": [entry.as_obj() for entry in self.state_entries],
            "update_count": self.update_count,
            "scheduler": None,
            "scaler": None,
        }

    @property
    def digest(self) -> str:
        return json_digest(self.as_obj(), domain=_OPTIMIZER_STATE_DOMAIN)


@dataclass(frozen=True, slots=True)
class AuthenticatedStepRecord:
    coordinate: AuthenticatedUpdateCoordinate
    source_digest: str
    plan_digest: str
    backward_execution_digest: str
    objective_value_hex: str
    optimizer_spec_digest: str
    optimizer_implementation_digest: str
    pre_optimizer_state_digest: str
    post_optimizer_state_digest: str
    parameter_registry_digest: str
    fp32_gradient_buffer_manifest_digest: str
    final_gradient_manifest_digest: str
    pre_runtime_manifest_digest: str
    post_runtime_manifest_digest: str
    pre_policy_state_digest: str
    post_policy_state_digest: str
    pre_model_provenance_digest: str
    post_model_provenance_digest: str
    pre_tensor_state_manifest_digest: str
    post_tensor_state_manifest_digest: str
    pre_parameter_manifest_digest: str
    post_parameter_manifest_digest: str
    changed_parameter_count: int
    gradient_global_norm_hex: str
    clipping_coefficient_hex: str
    forward_graph_count: int
    backward_call_count: int
    fp32_contribution_cast_count: int
    final_gradient_cast_count: int
    optimizer_step_call_count: int
    cuda_peak_allocated_bytes: int
    cuda_peak_reserved_bytes: int
    authorizes_execution: bool = False
    schema_version: int = AUTHENTICATED_STEP_SCHEMA_VERSION
    contract_id: str = AUTHENTICATED_STEP_CONTRACT_ID

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != AUTHENTICATED_STEP_SCHEMA_VERSION
            or type(self.contract_id) is not str
            or self.contract_id != AUTHENTICATED_STEP_CONTRACT_ID
        ):
            raise AuthenticatedStepError("step record schema or contract is invalid")
        if type(self.coordinate) is not AuthenticatedUpdateCoordinate:
            raise AuthenticatedStepError("step record coordinate is not nominal")
        for field_name in (
            "source_digest",
            "plan_digest",
            "backward_execution_digest",
            "optimizer_spec_digest",
            "optimizer_implementation_digest",
            "pre_optimizer_state_digest",
            "post_optimizer_state_digest",
            "parameter_registry_digest",
            "fp32_gradient_buffer_manifest_digest",
            "final_gradient_manifest_digest",
            "pre_runtime_manifest_digest",
            "post_runtime_manifest_digest",
            "pre_policy_state_digest",
            "post_policy_state_digest",
            "pre_model_provenance_digest",
            "post_model_provenance_digest",
            "pre_tensor_state_manifest_digest",
            "post_tensor_state_manifest_digest",
            "pre_parameter_manifest_digest",
            "post_parameter_manifest_digest",
        ):
            _sha256(getattr(self, field_name), name=field_name)
        if self.authorizes_execution is not False:
            raise AuthenticatedStepError("atomic smoke record may not authorize execution")
        count_fields = (
            "changed_parameter_count",
            "forward_graph_count",
            "backward_call_count",
            "fp32_contribution_cast_count",
            "final_gradient_cast_count",
            "optimizer_step_call_count",
            "cuda_peak_allocated_bytes",
            "cuda_peak_reserved_bytes",
        )
        if any(
            type(getattr(self, name)) is not int or getattr(self, name) < 0
            for name in count_fields
        ):
            raise AuthenticatedStepError("step record count evidence is invalid")
        if (
            self.optimizer_step_call_count != 1
            or self.changed_parameter_count < 1
            or self.forward_graph_count < 1
            or self.backward_call_count < 1
            or self.fp32_contribution_cast_count < 1
            or self.final_gradient_cast_count < 1
        ):
            raise AuthenticatedStepError("successful step record requires one nonzero update")
        if (
            self.pre_policy_state_digest == self.post_policy_state_digest
            or self.pre_model_provenance_digest == self.post_model_provenance_digest
            or self.pre_runtime_manifest_digest == self.post_runtime_manifest_digest
            or self.pre_tensor_state_manifest_digest
            == self.post_tensor_state_manifest_digest
            or self.pre_parameter_manifest_digest == self.post_parameter_manifest_digest
            or self.pre_optimizer_state_digest == self.post_optimizer_state_digest
        ):
            raise AuthenticatedStepError("successful step record requires a fresh policy digest")
        for name, value in (
            ("objective_value_hex", self.objective_value_hex),
            ("gradient_global_norm_hex", self.gradient_global_norm_hex),
        ):
            if type(value) is not str:
                raise AuthenticatedStepError(f"{name} is not canonical float.hex text")
            try:
                parsed = float.fromhex(value)
            except ValueError as exc:
                raise AuthenticatedStepError("step record float evidence is invalid") from exc
            if (
                not math.isfinite(parsed)
                or parsed.hex() != value
                or (name == "gradient_global_norm_hex" and parsed < 0)
            ):
                raise AuthenticatedStepError("step record float evidence is non-finite")
        if self.clipping_coefficient_hex != 1.0.hex():
            raise AuthenticatedStepError("v1 clipping coefficient must be exactly one")

    def as_obj(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "authorizes_execution": self.authorizes_execution,
            "coordinate": {**self.coordinate.as_obj(), "digest": self.coordinate.digest},
            "source_digest": self.source_digest,
            "plan_digest": self.plan_digest,
            "backward_execution_digest": self.backward_execution_digest,
            "objective_value_hex": self.objective_value_hex,
            "optimizer_spec_digest": self.optimizer_spec_digest,
            "optimizer_implementation_digest": self.optimizer_implementation_digest,
            "pre_optimizer_state_digest": self.pre_optimizer_state_digest,
            "post_optimizer_state_digest": self.post_optimizer_state_digest,
            "parameter_registry_digest": self.parameter_registry_digest,
            "fp32_gradient_buffer_manifest_digest": (
                self.fp32_gradient_buffer_manifest_digest
            ),
            "final_gradient_manifest_digest": self.final_gradient_manifest_digest,
            "pre_runtime_manifest_digest": self.pre_runtime_manifest_digest,
            "post_runtime_manifest_digest": self.post_runtime_manifest_digest,
            "pre_policy_state_digest": self.pre_policy_state_digest,
            "post_policy_state_digest": self.post_policy_state_digest,
            "pre_model_provenance_digest": self.pre_model_provenance_digest,
            "post_model_provenance_digest": self.post_model_provenance_digest,
            "pre_tensor_state_manifest_digest": self.pre_tensor_state_manifest_digest,
            "post_tensor_state_manifest_digest": self.post_tensor_state_manifest_digest,
            "pre_parameter_manifest_digest": self.pre_parameter_manifest_digest,
            "post_parameter_manifest_digest": self.post_parameter_manifest_digest,
            "changed_parameter_count": self.changed_parameter_count,
            "gradient_global_norm_hex": self.gradient_global_norm_hex,
            "clipping_coefficient_hex": self.clipping_coefficient_hex,
            "call_counts": {
                "forward_graph": self.forward_graph_count,
                "backward": self.backward_call_count,
                "fp32_contribution_cast": self.fp32_contribution_cast_count,
                "final_gradient_cast": self.final_gradient_cast_count,
                "optimizer_step": self.optimizer_step_call_count,
            },
            "cuda_peak_allocated_bytes": self.cuda_peak_allocated_bytes,
            "cuda_peak_reserved_bytes": self.cuda_peak_reserved_bytes,
        }

    @property
    def digest(self) -> str:
        return json_digest(self.as_obj(), domain=_STEP_RECORD_DOMAIN)

    def to_json(self) -> str:
        return dump_json({**self.as_obj(), "digest": self.digest})


@dataclass(frozen=True, slots=True)
class AuthenticatedStepEvidenceBundle:
    """Complete archivable nominal evidence published with one step record."""

    record: AuthenticatedStepRecord
    backward_execution: StreamingSFTExecution | StreamingExecutionDiagnostics
    optimizer_spec: AdamWOptimizerSpec
    pre_optimizer_state: OptimizerStateManifest
    post_optimizer_state: OptimizerStateManifest
    pre_runtime_manifest: SealedQwenRuntimeManifest
    post_runtime_manifest: SealedQwenRuntimeManifest
    pre_tensor_state_manifest: TensorStateManifest
    post_tensor_state_manifest: TensorStateManifest
    pre_parameters: tuple[ParameterByteRecord, ...]
    post_parameters: tuple[ParameterByteRecord, ...]

    def __post_init__(self) -> None:
        if type(self.record) is not AuthenticatedStepRecord:
            raise AuthenticatedStepError("evidence bundle record is not nominal")
        expected_execution_type: type[StreamingSFTExecution] | type[
            StreamingExecutionDiagnostics
        ] = (
            StreamingSFTExecution
            if self.record.coordinate.objective_kind == "sft"
            else StreamingExecutionDiagnostics
        )
        if type(self.backward_execution) is not expected_execution_type:
            raise AuthenticatedStepError("evidence bundle backward execution type differs")
        if self.backward_execution.digest != self.record.backward_execution_digest:
            raise AuthenticatedStepError("backward execution digest differs from step record")
        execution_evidence = (
            _sft_evidence(self.backward_execution)
            if type(self.backward_execution) is StreamingSFTExecution
            else _rl_evidence(
                cast(StreamingExecutionDiagnostics, self.backward_execution)
            )
        )
        if (
            execution_evidence.plan_digest != self.record.plan_digest
            or execution_evidence.objective_value_hex != self.record.objective_value_hex
            or execution_evidence.policy_state_digest != self.record.pre_policy_state_digest
            or execution_evidence.model_provenance_digest
            != self.record.pre_model_provenance_digest
            or execution_evidence.parameter_registry_digest
            != self.record.parameter_registry_digest
            or execution_evidence.fp32_manifest_digest
            != self.record.fp32_gradient_buffer_manifest_digest
            or execution_evidence.final_manifest_digest
            != self.record.final_gradient_manifest_digest
            or execution_evidence.forward_graph_count != self.record.forward_graph_count
            or execution_evidence.backward_call_count != self.record.backward_call_count
            or execution_evidence.fp32_cast_count
            != self.record.fp32_contribution_cast_count
            or execution_evidence.final_cast_count != self.record.final_gradient_cast_count
        ):
            raise AuthenticatedStepError("backward evidence fields differ from step record")
        if (
            type(self.optimizer_spec) is not AdamWOptimizerSpec
            or self.optimizer_spec.digest != self.record.optimizer_spec_digest
        ):
            raise AuthenticatedStepError("optimizer specification differs from step record")
        if (
            type(self.pre_optimizer_state) is not OptimizerStateManifest
            or type(self.post_optimizer_state) is not OptimizerStateManifest
            or self.pre_optimizer_state.digest != self.record.pre_optimizer_state_digest
            or self.post_optimizer_state.digest != self.record.post_optimizer_state_digest
        ):
            raise AuthenticatedStepError("optimizer manifests differ from step record")
        if (
            self.pre_optimizer_state.optimizer_spec_digest
            != self.record.optimizer_spec_digest
            or self.post_optimizer_state.optimizer_spec_digest
            != self.record.optimizer_spec_digest
            or self.pre_optimizer_state.optimizer_implementation_digest
            != self.record.optimizer_implementation_digest
            or self.post_optimizer_state.optimizer_implementation_digest
            != self.record.optimizer_implementation_digest
        ):
            raise AuthenticatedStepError("optimizer cross-bindings differ from step record")
        if (
            type(self.pre_runtime_manifest) is not SealedQwenRuntimeManifest
            or type(self.post_runtime_manifest) is not SealedQwenRuntimeManifest
            or self.pre_runtime_manifest.digest != self.record.pre_runtime_manifest_digest
            or self.post_runtime_manifest.digest != self.record.post_runtime_manifest_digest
        ):
            raise AuthenticatedStepError("runtime manifests differ from step record")
        if (
            self.pre_runtime_manifest.provider_policy_state_digest
            != self.record.pre_policy_state_digest
            or self.post_runtime_manifest.provider_policy_state_digest
            != self.record.post_policy_state_digest
            or self.pre_runtime_manifest.model_provenance_digest
            != self.record.pre_model_provenance_digest
            or self.post_runtime_manifest.model_provenance_digest
            != self.record.post_model_provenance_digest
            or self.pre_runtime_manifest.trainable_parameter_registry_digest
            != self.record.parameter_registry_digest
            or self.post_runtime_manifest.trainable_parameter_registry_digest
            != self.record.parameter_registry_digest
        ):
            raise AuthenticatedStepError("runtime cross-bindings differ from step record")
        if (
            type(self.pre_tensor_state_manifest) is not TensorStateManifest
            or type(self.post_tensor_state_manifest) is not TensorStateManifest
            or self.pre_tensor_state_manifest.digest
            != self.record.pre_tensor_state_manifest_digest
            or self.post_tensor_state_manifest.digest
            != self.record.post_tensor_state_manifest_digest
        ):
            raise AuthenticatedStepError("tensor-state manifests differ from step record")
        pre_parameters = tuple(self.pre_parameters)
        post_parameters = tuple(self.post_parameters)
        object.__setattr__(self, "pre_parameters", pre_parameters)
        object.__setattr__(self, "post_parameters", post_parameters)
        if (
            not pre_parameters
            or len(pre_parameters) != len(post_parameters)
            or any(type(value) is not ParameterByteRecord for value in pre_parameters)
            or any(type(value) is not ParameterByteRecord for value in post_parameters)
            or _parameter_manifest_digest(pre_parameters)
            != self.record.pre_parameter_manifest_digest
            or _parameter_manifest_digest(post_parameters)
            != self.record.post_parameter_manifest_digest
        ):
            raise AuthenticatedStepError("parameter manifests differ from step record")
        changed_parameter_count = 0
        for before, after in zip(pre_parameters, post_parameters, strict=True):
            if (
                before.name != after.name
                or before.aliases != after.aliases
                or before.dtype != after.dtype
                or before.device != after.device
                or before.shape != after.shape
                or before.byte_length != after.byte_length
            ):
                raise AuthenticatedStepError("parameter identity changed across the step")
            changed_parameter_count += before.sha256 != after.sha256
        if changed_parameter_count != self.record.changed_parameter_count:
            raise AuthenticatedStepError("changed-parameter count differs from byte evidence")

    def as_obj(self) -> dict[str, object]:
        return {
            "authorizes_execution": False,
            "record": {**self.record.as_obj(), "digest": self.record.digest},
            "backward_execution": {
                **self.backward_execution.as_obj(),
                "digest": self.backward_execution.digest,
            },
            "optimizer_spec": {
                **self.optimizer_spec.as_obj(),
                "digest": self.optimizer_spec.digest,
            },
            "pre_optimizer_state": {
                **self.pre_optimizer_state.as_obj(),
                "digest": self.pre_optimizer_state.digest,
            },
            "post_optimizer_state": {
                **self.post_optimizer_state.as_obj(),
                "digest": self.post_optimizer_state.digest,
            },
            "pre_runtime_manifest": {
                **self.pre_runtime_manifest.as_obj(),
                "digest": self.pre_runtime_manifest.digest,
            },
            "post_runtime_manifest": {
                **self.post_runtime_manifest.as_obj(),
                "digest": self.post_runtime_manifest.digest,
            },
            "pre_tensor_state_manifest": {
                **self.pre_tensor_state_manifest.as_obj(),
                "digest": self.pre_tensor_state_manifest.digest,
            },
            "post_tensor_state_manifest": {
                **self.post_tensor_state_manifest.as_obj(),
                "digest": self.post_tensor_state_manifest.digest,
            },
            "pre_parameters": [record.as_obj() for record in self.pre_parameters],
            "post_parameters": [record.as_obj() for record in self.post_parameters],
        }

    @property
    def digest(self) -> str:
        return json_digest(
            self.as_obj(),
            domain="goalzendo-interactive-atomic-step-evidence-bundle-v1",
        )

    def to_json(self) -> str:
        return dump_json({**self.as_obj(), "digest": self.digest})


def _raw_tensor_bytes(tensor: torch.Tensor) -> bytes:
    if tensor.layout is not torch.strided or tensor.is_quantized or tensor.device.type == "meta":
        raise AuthenticatedStepError("step tensors must be materialized nonquantized strided tensors")
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
        raise AuthenticatedStepError("step tensor bytes could not be materialized") from exc


def _parameter_records(
    registry: TrainableParameterRegistry,
) -> tuple[tuple[ParameterByteRecord, ...], str]:
    records: list[ParameterByteRecord] = []
    trainable = tuple(record for record in registry.manifest.records if record.requires_grad)
    if len(trainable) != len(registry.parameters):
        raise AuthenticatedStepError("trainable registry is internally inconsistent")
    for metadata, parameter in zip(trainable, registry.parameters, strict=True):
        if (parameter.is_floating_point() or parameter.is_complex()) and not bool(
            torch.isfinite(parameter.detach()).all().item()
        ):
            raise AuthenticatedStepError("registered parameter bytes are non-finite")
        raw = _raw_tensor_bytes(parameter)
        records.append(
            ParameterByteRecord(
                name=metadata.canonical_name,
                aliases=metadata.aliases,
                dtype=str(parameter.dtype),
                device=str(parameter.device),
                shape=tuple(parameter.shape),
                byte_length=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
            )
        )
    selected = tuple(records)
    return selected, _parameter_manifest_digest(selected)


def _parameter_manifest_digest(records: tuple[ParameterByteRecord, ...]) -> str:
    return json_digest(
        [record.as_obj() for record in records],
        domain=_PARAMETER_BYTES_DOMAIN,
    )


def _canonical_optimizer_value(value: object) -> object:
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise AuthenticatedStepError("optimizer scalar state is non-finite")
        return {"kind": "float", "hex": value.hex()}
    if type(value) is tuple:
        return {
            "kind": "tuple",
            "items": [
                _canonical_optimizer_value(item) for item in cast(tuple[object, ...], value)
            ],
        }
    if isinstance(value, torch.Tensor):
        raw = _raw_tensor_bytes(value)
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value.detach()).all().item()
        ):
            raise AuthenticatedStepError("optimizer tensor state is non-finite")
        return {
            "kind": "tensor",
            "dtype": str(value.dtype),
            "device": str(value.device),
            "shape": list(value.shape),
            "stride": list(value.stride()),
            "byte_length": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    raise AuthenticatedStepError(
        f"unsupported optimizer state value {type(value).__module__}.{type(value).__qualname__}"
    )


@dataclass(frozen=True, slots=True)
class _OptimizerCallableGuard:
    role: str
    identity: int
    code_identity: int
    module: str
    qualname: str
    source_sha256: str
    bytecode_sha256: str
    defaults_sha256: str
    closure_sha256: str
    closure_identities: tuple[int, ...]

    def semantic_obj(self) -> dict[str, object]:
        return {
            "role": self.role,
            "module": self.module,
            "qualname": self.qualname,
            "source_sha256": self.source_sha256,
            "bytecode_sha256": self.bytecode_sha256,
            "defaults_sha256": self.defaults_sha256,
            "closure_sha256": self.closure_sha256,
        }


@dataclass(frozen=True, slots=True)
class _OptimizerImplementationGuard:
    torch_version: str
    optimizer_class_identity: int
    optimizer_class: str
    parameter_group_template_json: str
    callables: tuple[_OptimizerCallableGuard, ...]

    @property
    def digest(self) -> str:
        return json_digest(
            {
                "torch_version": self.torch_version,
                "optimizer_class": self.optimizer_class,
                "parameter_group_template": load_json(
                    self.parameter_group_template_json
                ),
                "callables": [record.semantic_obj() for record in self.callables],
                "foreach": False,
                "fused": False,
            },
            domain="goalzendo-interactive-adamw-implementation-v1",
        )


def _optimizer_function_payload(
    function: object,
    *,
    path: str,
    seen: frozenset[int],
) -> tuple[dict[str, object], tuple[int, ...]]:
    if not inspect.isfunction(function):
        raise AuthenticatedStepError(f"AdamW {path} must be an inspectable Python function")
    dynamic = cast(object, function)
    code = getattr(dynamic, "__code__", None)
    module = getattr(dynamic, "__module__", None)
    qualname = getattr(dynamic, "__qualname__", None)
    if code is None or type(module) is not str or type(qualname) is not str:
        raise AuthenticatedStepError(f"AdamW {path} identity is not canonical")
    try:
        source = textwrap.dedent(inspect.getsource(function)).replace("\r\n", "\n")
    except (OSError, TypeError) as exc:
        raise AuthenticatedStepError(f"AdamW {path} source is not inspectable") from exc
    defaults = getattr(dynamic, "__defaults__", None) or ()
    kwdefaults = getattr(dynamic, "__kwdefaults__", None) or {}
    if type(defaults) is not tuple or type(kwdefaults) is not dict:
        raise AuthenticatedStepError(f"AdamW {path} defaults are not canonical")
    defaults_sha256 = json_digest(
        {
            "defaults": [_canonical_optimizer_value(value) for value in defaults],
            "kwdefaults": {
                key: _canonical_optimizer_value(kwdefaults[key])
                for key in sorted(kwdefaults)
            },
        },
        domain="goalzendo-interactive-adamw-callable-defaults-v1",
    )
    closure_values: list[object] = []
    closure_identities: list[int] = []
    closure = getattr(dynamic, "__closure__", None) or ()
    if type(closure) is not tuple:
        raise AuthenticatedStepError(f"AdamW {path} closure is not canonical")
    nested_seen = seen | {id(function)}
    for index, cell in enumerate(closure):
        try:
            value = cell.cell_contents
        except ValueError as exc:
            raise AuthenticatedStepError(f"AdamW {path} closure contains an empty cell") from exc
        semantic, identities = _optimizer_closure_value(
            value,
            path=f"{path}.closure[{index}]",
            seen=nested_seen,
        )
        closure_values.append(semantic)
        closure_identities.extend(identities)
    return (
        {
            "module": module,
            "qualname": qualname,
            "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "bytecode_sha256": hashlib.sha256(code.co_code).hexdigest(),
            "defaults_sha256": defaults_sha256,
            "closure": closure_values,
        },
        (id(function), id(code), *closure_identities),
    )


def _optimizer_closure_value(
    value: object,
    *,
    path: str,
    seen: frozenset[int],
) -> tuple[object, tuple[int, ...]]:
    if inspect.isfunction(value):
        dynamic = cast(object, value)
        code = getattr(dynamic, "__code__", None)
        module = getattr(dynamic, "__module__", None)
        qualname = getattr(dynamic, "__qualname__", None)
        if code is None or type(module) is not str or type(qualname) is not str:
            raise AuthenticatedStepError(f"AdamW {path} function is not canonical")
        if id(value) in seen:
            return (
                {"kind": "function_cycle", "module": module, "qualname": qualname},
                (id(value), id(code)),
            )
        payload, identities = _optimizer_function_payload(
            value,
            path=path,
            seen=seen,
        )
        return {"kind": "function", "value": payload}, identities
    if isinstance(value, type):
        module = getattr(value, "__module__", None)
        qualname = getattr(value, "__qualname__", None)
        if type(module) is not str or type(qualname) is not str:
            raise AuthenticatedStepError(f"AdamW {path} type is not canonical")
        return (
            {"kind": "type", "module": module, "qualname": qualname},
            (id(value),),
        )
    if type(value) is tuple:
        items: list[object] = []
        tuple_identities: list[int] = []
        for index, item in enumerate(cast(tuple[object, ...], value)):
            semantic, nested = _optimizer_closure_value(
                item,
                path=f"{path}[{index}]",
                seen=seen,
            )
            items.append(semantic)
            tuple_identities.extend(nested)
        return {"kind": "tuple", "items": items}, tuple(tuple_identities)
    if isinstance(value, torch.Tensor):
        return _canonical_optimizer_value(value), (id(value),)
    if value is None or type(value) in {bool, int, float, str}:
        return _canonical_optimizer_value(value), ()
    raise AuthenticatedStepError(
        f"unsupported AdamW closure value {path}: "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _optimizer_callable_guard(role: str, function: object) -> _OptimizerCallableGuard:
    payload, identities = _optimizer_function_payload(
        function,
        path=role,
        seen=frozenset(),
    )
    if len(identities) < 2:
        raise AuthenticatedStepError(f"AdamW {role} identity inventory is incomplete")
    return _OptimizerCallableGuard(
        role=role,
        identity=identities[0],
        code_identity=identities[1],
        module=cast(str, payload["module"]),
        qualname=cast(str, payload["qualname"]),
        source_sha256=cast(str, payload["source_sha256"]),
        bytecode_sha256=cast(str, payload["bytecode_sha256"]),
        defaults_sha256=cast(str, payload["defaults_sha256"]),
        closure_sha256=json_digest(
            payload["closure"],
            domain="goalzendo-interactive-adamw-callable-closure-v1",
        ),
        closure_identities=identities[2:],
    )


def _optimizer_implementation_guard() -> _OptimizerImplementationGuard:
    import torch.optim.adam as adam_module
    import torch.optim.adamw as adamw_module

    selected_class = torch.optim.AdamW
    class_name = f"{selected_class.__module__}.{selected_class.__qualname__}"
    if class_name != "torch.optim.adamw.AdamW":
        raise AuthenticatedStepError("torch.optim.AdamW class identity label changed")
    return _OptimizerImplementationGuard(
        torch_version=str(torch.__version__),
        optimizer_class_identity=id(selected_class),
        optimizer_class=class_name,
        parameter_group_template_json=_TRUSTED_ADAMW_GROUP_TEMPLATE_JSON,
        callables=(
            _optimizer_callable_guard("AdamW.__init__", selected_class.__init__),
            _optimizer_callable_guard("AdamW.step", selected_class.step),
            _optimizer_callable_guard("AdamW.zero_grad", selected_class.zero_grad),
            _optimizer_callable_guard(
                "AdamW._init_group",
                getattr(selected_class, "_init_group", None),
            ),
            _optimizer_callable_guard(
                "Optimizer._cuda_graph_capture_health_check",
                getattr(selected_class, "_cuda_graph_capture_health_check", None),
            ),
            _optimizer_callable_guard("adamw.adam", getattr(adamw_module, "adam", None)),
            _optimizer_callable_guard(
                "adam._single_tensor_adam",
                getattr(adam_module, "_single_tensor_adam", None),
            ),
        ),
    )


def _prime_torch_optimizer_step_wrapper() -> str:
    """Force PyTorch's one-time class step wrapper before fixing identities."""

    parameter = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
    optimizer = torch.optim.AdamW(
        (parameter,),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        amsgrad=False,
        maximize=False,
        foreach=False,
        capturable=False,
        differentiable=False,
        fused=False,
    )
    if len(optimizer.param_groups) != 1 or type(optimizer.param_groups[0]) is not dict:
        raise AuthenticatedStepError("AdamW baseline parameter group is not canonical")
    group = optimizer.param_groups[0]
    if tuple(group.get("params", ())) != (parameter,):
        raise AuthenticatedStepError("AdamW baseline parameter group changed its parameter")
    return dump_json(
        {
            key: _canonical_optimizer_value(value)
            for key, value in sorted(group.items())
            if key != "params"
        }
    )


_TRUSTED_ADAMW_GROUP_TEMPLATE_JSON = _prime_torch_optimizer_step_wrapper()
_TRUSTED_ADAMW_IMPLEMENTATION = _optimizer_implementation_guard()


def _require_trusted_optimizer_implementation() -> _OptimizerImplementationGuard:
    current = _optimizer_implementation_guard()
    if current != _TRUSTED_ADAMW_IMPLEMENTATION:
        raise AuthenticatedStepError("AdamW executable implementation changed after import")
    return current


def _assert_no_optimizer_hooks(optimizer: torch.optim.AdamW) -> None:
    import torch.optim.optimizer as optimizer_module

    if type(optimizer) is not torch.optim.AdamW:
        raise AuthenticatedStepError("optimizer must have exact type torch.optim.AdamW")
    for method_name in ("step", "zero_grad"):
        if method_name in vars(optimizer):
            raise AuthenticatedStepError(
                f"optimizer instance {method_name} override is forbidden"
            )
        bound = getattr(optimizer, method_name, None)
        if (
            getattr(bound, "__self__", None) is not optimizer
            or getattr(bound, "__func__", None) is not getattr(torch.optim.AdamW, method_name)
        ):
            raise AuthenticatedStepError(
                f"optimizer {method_name} does not resolve to the trusted class method"
            )
    if optimizer_module._global_optimizer_pre_hooks or optimizer_module._global_optimizer_post_hooks:
        raise AuthenticatedStepError("global optimizer step hooks are forbidden")
    for name, value in vars(optimizer).items():
        if "hook" in name and value:
            raise AuthenticatedStepError("optimizer instance hooks are forbidden")


def _optimizer_manifest(
    optimizer: torch.optim.AdamW,
    spec: AdamWOptimizerSpec,
    registry: TrainableParameterRegistry,
) -> OptimizerStateManifest:
    implementation = _require_trusted_optimizer_implementation()
    if type(optimizer) is not torch.optim.AdamW:
        raise AuthenticatedStepError("optimizer must have exact type torch.optim.AdamW")
    _assert_no_optimizer_hooks(optimizer)
    if len(optimizer.param_groups) != 1:
        raise AuthenticatedStepError("atomic AdamW must have exactly one parameter group")
    group = optimizer.param_groups[0]
    if type(group) is not dict or tuple(group.get("params", ())) != registry.parameters:
        raise AuthenticatedStepError("optimizer parameter group differs from the registry")
    resolved = {
        key: _canonical_optimizer_value(value)
        for key, value in sorted(group.items())
        if key != "params"
    }
    template = load_json(_TRUSTED_ADAMW_GROUP_TEMPLATE_JSON)
    if type(template) is not dict or any(type(key) is not str for key in template):
        raise AuthenticatedStepError("trusted optimizer group template is invalid")
    expected_resolved = dict(cast(dict[str, object], template))
    expected_resolved.update(
        {
            "lr": _canonical_optimizer_value(float(spec.learning_rate)),
            "betas": _canonical_optimizer_value(
                (float(spec.beta1), float(spec.beta2))
            ),
            "eps": _canonical_optimizer_value(float(spec.epsilon)),
            "weight_decay": _canonical_optimizer_value(float(spec.weight_decay)),
        }
    )
    if resolved != expected_resolved:
        raise AuthenticatedStepError("optimizer resolved configuration differs from its spec")
    entries: list[OptimizerStateEntry] = []
    update_counts: list[int] = []
    trainable_records = tuple(
        record for record in registry.manifest.records if record.requires_grad
    )
    for metadata, parameter in zip(trainable_records, registry.parameters, strict=True):
        state = optimizer.state.get(parameter, {})
        if type(state) is not dict or any(type(key) is not str for key in state):
            raise AuthenticatedStepError("optimizer state mapping is not canonical")
        for key in sorted(state):
            value = state[key]
            entries.append(
                OptimizerStateEntry(
                    parameter_name=metadata.canonical_name,
                    state_key=key,
                    value_manifest_json=dump_json(_canonical_optimizer_value(value)),
                )
            )
        step = state.get("step")
        if step is not None:
            if not isinstance(step, torch.Tensor) or step.numel() != 1:
                raise AuthenticatedStepError("AdamW step state must be one tensor scalar")
            step_value = float(step.detach().to(device="cpu", dtype=torch.float64).item())
            if not step_value.is_integer() or step_value < 0:
                raise AuthenticatedStepError("AdamW step state is invalid")
            update_counts.append(int(step_value))
    update_count = 0
    if update_counts:
        if len(update_counts) != len(registry.parameters) or len(set(update_counts)) != 1:
            raise AuthenticatedStepError("AdamW parameter update counts diverged")
        update_count = update_counts[0]
    manifest = OptimizerStateManifest(
        optimizer_class="torch.optim.AdamW",
        optimizer_spec_digest=spec.digest,
        optimizer_implementation_digest=implementation.digest,
        resolved_configuration_json=dump_json(resolved),
        parameter_group_names=tuple(record.canonical_name for record in trainable_records),
        state_entries=tuple(entries),
        update_count=update_count,
    )
    _ = manifest.digest
    return manifest


def _construct_optimizer(
    registry: TrainableParameterRegistry,
    spec: AdamWOptimizerSpec,
) -> torch.optim.AdamW:
    if type(spec) is not AdamWOptimizerSpec:
        raise TypeError("optimizer_spec must have exact type AdamWOptimizerSpec")
    _require_trusted_optimizer_implementation()
    optimizer = torch.optim.AdamW(
        registry.parameters,
        lr=float(spec.learning_rate),
        betas=(float(spec.beta1), float(spec.beta2)),
        eps=float(spec.epsilon),
        weight_decay=float(spec.weight_decay),
        amsgrad=False,
        maximize=False,
        foreach=False,
        capturable=False,
        differentiable=False,
        fused=False,
    )
    _assert_no_optimizer_hooks(optimizer)
    return optimizer


def _validate_live_gradients(
    registry: TrainableParameterRegistry,
    expected_records: Sequence[object],
) -> float:
    trainable_records = tuple(
        record for record in registry.manifest.records if record.requires_grad
    )
    expected = tuple(expected_records)
    if len(expected) != len(trainable_records):
        raise AuthenticatedStepError("final-gradient evidence is incomplete")
    squared_norm = 0.0
    for metadata, parameter, evidence in zip(
        trainable_records,
        registry.parameters,
        expected,
        strict=True,
    ):
        gradient = parameter.grad
        if gradient is None or gradient.is_sparse or gradient.layout is not torch.strided:
            raise AuthenticatedStepError("live final gradient is absent or unsupported")
        if not bool(torch.isfinite(gradient.detach()).all().item()):
            raise AuthenticatedStepError("live final gradient is non-finite")
        if (
            getattr(evidence, "name", None) != metadata.canonical_name
            or tuple(getattr(evidence, "aliases", ())) != metadata.aliases
            or getattr(evidence, "gradient_dtype", None) != str(gradient.dtype)
            or getattr(evidence, "gradient_device", None) != str(gradient.device)
            or tuple(getattr(evidence, "parameter_shape", ())) != tuple(parameter.shape)
        ):
            raise AuthenticatedStepError("live gradient metadata differs from executor evidence")
        raw = _raw_tensor_bytes(gradient)
        if not hmac.compare_digest(
            hashlib.sha256(raw).hexdigest(),
            cast(str, getattr(evidence, "sha256", "")),
        ):
            raise AuthenticatedStepError("live gradient bytes differ from executor evidence")
        squared_norm = math.fsum(
            (
                squared_norm,
                float(
                    gradient.detach()
                    .to(device="cpu", dtype=torch.float64)
                    .square()
                    .sum()
                    .item()
                ),
            )
        )
    norm = math.sqrt(squared_norm)
    if not math.isfinite(norm):
        raise AuthenticatedStepError("gradient global norm is non-finite")
    return norm


def _clear_gradients(registry: TrainableParameterRegistry) -> None:
    for parameter in registry.parameters:
        parameter.grad = None


def _snapshot_parameters(
    registry: TrainableParameterRegistry,
) -> tuple[torch.Tensor, ...]:
    return tuple(parameter.detach().to(device="cpu").clone() for parameter in registry.parameters)


def _restore_parameters(
    registry: TrainableParameterRegistry,
    snapshots: tuple[torch.Tensor, ...],
) -> None:
    if len(snapshots) != len(registry.parameters):
        raise AuthenticatedStepError("recovery snapshot is incomplete")
    with torch.no_grad():
        for parameter, snapshot in zip(registry.parameters, snapshots, strict=True):
            parameter.copy_(snapshot.to(device=parameter.device, dtype=parameter.dtype))


@dataclass(frozen=True, slots=True)
class _BackwardEvidence:
    execution_digest: str
    objective_value_hex: str
    plan_digest: str
    policy_state_digest: str
    model_provenance_digest: str
    parameter_registry_digest: str
    fp32_manifest_digest: str
    final_manifest_digest: str
    final_records: tuple[object, ...]
    forward_graph_count: int
    backward_call_count: int
    fp32_cast_count: int
    final_cast_count: int


def _sft_evidence(execution: StreamingSFTExecution) -> _BackwardEvidence:
    if type(execution) is not StreamingSFTExecution:
        raise AuthenticatedStepError("SFT executor returned non-nominal evidence")
    return _BackwardEvidence(
        execution_digest=execution.digest,
        objective_value_hex=execution.objective_value_hex,
        plan_digest=execution.plan_digest,
        policy_state_digest=execution.provider_policy_state_digest,
        model_provenance_digest=execution.model_provenance_digest,
        parameter_registry_digest=execution.trainable_parameter_registry_digest,
        fp32_manifest_digest=execution.fp32_gradient_buffer_manifest_digest,
        final_manifest_digest=execution.gradient_manifest_digest,
        final_records=cast(tuple[object, ...], execution.gradients),
        forward_graph_count=execution.decision_count,
        backward_call_count=execution.decision_count,
        fp32_cast_count=execution.fp32_contribution_cast_count,
        final_cast_count=execution.final_gradient_cast_count,
    )


def _rl_evidence(execution: StreamingExecutionDiagnostics) -> _BackwardEvidence:
    if type(execution) is not StreamingExecutionDiagnostics:
        raise AuthenticatedStepError("RL executor returned non-nominal evidence")
    if execution.backward_call_count < 1 or execution.finite_gradient_parameter_count < 1:
        raise AuthenticatedStepError("RL plan is not optimizer-step eligible")
    return _BackwardEvidence(
        execution_digest=execution.digest,
        objective_value_hex=execution.replayed_total_loss_hex,
        plan_digest=execution.plan_digest,
        policy_state_digest=execution.policy_state_digest,
        model_provenance_digest=execution.model_provenance_digest,
        parameter_registry_digest=execution.parameter_registry_digest,
        fp32_manifest_digest=execution.fp32_gradient_buffer_manifest_digest,
        final_manifest_digest=execution.final_gradient_manifest_digest,
        final_records=cast(tuple[object, ...], execution.final_gradients),
        forward_graph_count=execution.authenticated_turn_replay_count,
        backward_call_count=execution.backward_call_count,
        fp32_cast_count=execution.fp32_contribution_cast_count,
        final_cast_count=execution.final_gradient_cast_count,
    )


def _sft_source_digest(sources: Sequence[ReferenceTrajectorySFTSource]) -> str:
    return json_digest(
        [
            {
                "episode": source.episode.as_obj(),
                "reference_trajectory": source.reference_trajectory.as_obj(),
            }
            for source in sources
        ],
        domain=_SOURCE_DOMAIN,
    )


def _rl_source_digest(
    records: Sequence[AuthenticatedRolloutRecord],
    episodes: Sequence[HiddenEpisode],
) -> str:
    return json_digest(
        {
            "records": [record.as_obj() for record in records],
            "episodes": [episode.as_obj() for episode in episodes],
        },
        domain=_SOURCE_DOMAIN,
    )


def _cuda_peaks(registry: TrainableParameterRegistry) -> tuple[int, int]:
    devices = {parameter.device for parameter in registry.parameters}
    if len(devices) != 1:
        raise AuthenticatedStepError("atomic registry must occupy one device")
    device = next(iter(devices))
    if device.type != "cuda":
        return 0, 0
    return (
        int(torch.cuda.max_memory_allocated(device)),
        int(torch.cuda.max_memory_reserved(device)),
    )


def _execute_atomic_step(
    runtime: SealedQwenRuntime,
    coordinate: AuthenticatedUpdateCoordinate,
    optimizer_spec: AdamWOptimizerSpec,
    source_digest: str,
    backward: Literal["sft", "rl"],
    verified_plan: _VerifiedStreamingSFTPlan | _VerifiedStreamingObjectivePlan,
    sources: object,
) -> AuthenticatedStepEvidenceBundle:
    if type(coordinate) is not AuthenticatedUpdateCoordinate:
        raise TypeError("coordinate must have exact type AuthenticatedUpdateCoordinate")
    if coordinate.objective_kind != backward:
        raise AuthenticatedStepError("coordinate objective differs from the step entry point")
    if type(optimizer_spec) is not AdamWOptimizerSpec:
        raise TypeError("optimizer_spec must have exact type AdamWOptimizerSpec")
    selected_runtime = require_exact_sealed_qwen_runtime(runtime)
    session: SealedQwenUpdateSession | None = None
    registry: TrainableParameterRegistry | None = None
    snapshots: tuple[torch.Tensor, ...] | None = None
    mutation_possible = False
    try:
        session = selected_runtime._begin_authenticated_update()
        registry = session.trainable_parameter_registry
        provider = session._provider_for_step
        if registry.manifest.digest != provider.trainable_parameter_registry.manifest.digest:
            raise AuthenticatedStepError("session/provider registries diverged")
        if any(parameter.grad is not None for parameter in registry.parameters):
            raise AuthenticatedStepError("atomic step requires absent starting gradients")
        pre_runtime = session.pre_manifest
        pre_tensor_state_manifest = provider.tensor_state_manifest
        pre_tensor_state = pre_tensor_state_manifest.digest
        pre_parameters, pre_parameter_digest = _parameter_records(registry)
        device = registry.parameters[0].device
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        if backward == "sft":
            if type(verified_plan) is not _VerifiedStreamingSFTPlan:
                raise TypeError("verified_plan must be a nominal streaming SFT plan")
            selected_sources = cast(Sequence[ReferenceTrajectorySFTSource], sources)
            sft_execution = execute_streaming_sft_backward(
                verified_plan,
                selected_sources,
                session._tokenizer_for_step,
                session._compiler_for_step,
                provider,
            )
            evidence = _sft_evidence(sft_execution)
            backward_execution: StreamingSFTExecution | StreamingExecutionDiagnostics = (
                sft_execution
            )
        else:
            if type(verified_plan) is not _VerifiedStreamingObjectivePlan:
                raise TypeError("verified_plan must be a nominal streaming RL plan")
            records, episodes = cast(
                tuple[Sequence[AuthenticatedRolloutRecord], Sequence[HiddenEpisode]],
                sources,
            )
            named_parameters = tuple(
                (record.canonical_name, parameter)
                for record, parameter in zip(
                    (
                        record
                        for record in registry.manifest.records
                        if record.requires_grad
                    ),
                    registry.parameters,
                    strict=True,
                )
            )
            rl_execution = execute_verified_streaming_backward(
                verified_plan,
                records,
                episodes,
                provider,
                session._tokenizer_for_step,
                session._compiler_for_step,
                named_parameters,
                controllers=None,
            )
            evidence = _rl_evidence(rl_execution)
            backward_execution = rl_execution

        if (
            evidence.plan_digest != verified_plan.plan.digest
            or evidence.policy_state_digest != pre_runtime.provider_policy_state_digest
            or evidence.model_provenance_digest != pre_runtime.model_provenance_digest
            or evidence.parameter_registry_digest != registry.manifest.digest
        ):
            raise AuthenticatedStepError("backward evidence differs from the held runtime/plan")
        gradient_norm = _validate_live_gradients(registry, evidence.final_records)
        snapshots = _snapshot_parameters(registry)
        optimizer = _construct_optimizer(registry, optimizer_spec)
        pre_optimizer = _optimizer_manifest(optimizer, optimizer_spec, registry)
        if pre_optimizer.update_count != 0 or pre_optimizer.state_entries:
            raise AuthenticatedStepError("fresh AdamW optimizer unexpectedly has state")

        mutation_possible = True
        optimizer_step_call_count = 1
        implementation = _require_trusted_optimizer_implementation()
        _assert_no_optimizer_hooks(optimizer)
        optimizer.step()
        if _require_trusted_optimizer_implementation() != implementation:
            raise AuthenticatedStepError("AdamW implementation changed during optimizer step")
        _assert_no_optimizer_hooks(optimizer)
        optimizer.zero_grad(set_to_none=True)
        if _require_trusted_optimizer_implementation() != implementation:
            raise AuthenticatedStepError("AdamW implementation changed during gradient clearing")
        _assert_no_optimizer_hooks(optimizer)
        if any(parameter.grad is not None for parameter in registry.parameters):
            raise AuthenticatedStepError("optimizer zero_grad did not restore absent gradients")
        post_optimizer = _optimizer_manifest(optimizer, optimizer_spec, registry)
        if post_optimizer.update_count != 1:
            raise AuthenticatedStepError("AdamW did not record exactly one update")
        post_parameters, post_parameter_digest = _parameter_records(registry)
        changed_parameter_count = sum(
            before.sha256 != after.sha256
            for before, after in zip(pre_parameters, post_parameters, strict=True)
        )
        if changed_parameter_count < 1:
            raise AuthenticatedStepError("optimizer step changed no parameter bytes")
        prepared: SealedQwenPreparedUpdate = selected_runtime._prepare_authenticated_update(
            session
        )
        post_runtime = prepared.manifest
        post_tensor_state_manifest = prepared._provider.tensor_state_manifest
        post_tensor_state = post_tensor_state_manifest.digest
        peak_allocated, peak_reserved = _cuda_peaks(registry)
        record = AuthenticatedStepRecord(
            coordinate=coordinate,
            source_digest=source_digest,
            plan_digest=evidence.plan_digest,
            backward_execution_digest=evidence.execution_digest,
            objective_value_hex=evidence.objective_value_hex,
            optimizer_spec_digest=optimizer_spec.digest,
            optimizer_implementation_digest=implementation.digest,
            pre_optimizer_state_digest=pre_optimizer.digest,
            post_optimizer_state_digest=post_optimizer.digest,
            parameter_registry_digest=registry.manifest.digest,
            fp32_gradient_buffer_manifest_digest=evidence.fp32_manifest_digest,
            final_gradient_manifest_digest=evidence.final_manifest_digest,
            pre_runtime_manifest_digest=pre_runtime.digest,
            post_runtime_manifest_digest=post_runtime.digest,
            pre_policy_state_digest=pre_runtime.provider_policy_state_digest,
            post_policy_state_digest=post_runtime.provider_policy_state_digest,
            pre_model_provenance_digest=pre_runtime.model_provenance_digest,
            post_model_provenance_digest=post_runtime.model_provenance_digest,
            pre_tensor_state_manifest_digest=pre_tensor_state,
            post_tensor_state_manifest_digest=post_tensor_state,
            pre_parameter_manifest_digest=pre_parameter_digest,
            post_parameter_manifest_digest=post_parameter_digest,
            changed_parameter_count=changed_parameter_count,
            gradient_global_norm_hex=gradient_norm.hex(),
            clipping_coefficient_hex=1.0.hex(),
            forward_graph_count=evidence.forward_graph_count,
            backward_call_count=evidence.backward_call_count,
            fp32_contribution_cast_count=evidence.fp32_cast_count,
            final_gradient_cast_count=evidence.final_cast_count,
            optimizer_step_call_count=optimizer_step_call_count,
            cuda_peak_allocated_bytes=peak_allocated,
            cuda_peak_reserved_bytes=peak_reserved,
        )
        bundle = AuthenticatedStepEvidenceBundle(
            record=record,
            backward_execution=backward_execution,
            optimizer_spec=optimizer_spec,
            pre_optimizer_state=pre_optimizer,
            post_optimizer_state=post_optimizer,
            pre_runtime_manifest=pre_runtime,
            post_runtime_manifest=post_runtime,
            pre_tensor_state_manifest=pre_tensor_state_manifest,
            post_tensor_state_manifest=post_tensor_state_manifest,
            pre_parameters=pre_parameters,
            post_parameters=post_parameters,
        )
        # Force every failure-prone canonical validation while the update lease
        # and recovery snapshots are still held.
        _ = record.to_json()
        _ = bundle.to_json()
        selected_runtime._publish_authenticated_update(session, prepared)
        return bundle
    except BaseException as exc:
        rollback_error: BaseException | None = None
        if registry is not None:
            _clear_gradients(registry)
        if session is not None and session._active:
            try:
                if mutation_possible:
                    if registry is None or snapshots is None:
                        selected_runtime._mark_authenticated_update_corrupt(session)
                    else:
                        _restore_parameters(registry, snapshots)
                        selected_runtime._rollback_authenticated_update(session)
                else:
                    selected_runtime._rollback_authenticated_update(session)
            except BaseException as recovery_exc:
                rollback_error = recovery_exc
                if session._active:
                    with suppress(BaseException):
                        selected_runtime._mark_authenticated_update_corrupt(session)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        if rollback_error is not None:
            raise AuthenticatedStepError(
                "atomic step failed and recovery could not restore the runtime"
            ) from rollback_error
        if isinstance(exc, AuthenticatedStepError):
            raise
        if isinstance(exc, (SealedQwenRuntimeError, RuntimeError, TypeError, ValueError)):
            raise AuthenticatedStepError("atomic step failed closed without publication") from exc
        raise


def execute_authenticated_sft_step(
    runtime: SealedQwenRuntime,
    verified_plan: _VerifiedStreamingSFTPlan,
    sources: Sequence[ReferenceTrajectorySFTSource],
    optimizer_spec: AdamWOptimizerSpec,
    coordinate: AuthenticatedUpdateCoordinate,
) -> AuthenticatedStepEvidenceBundle:
    """Execute one nonauthorizing SFT smoke step as a sealed transaction."""

    if isinstance(sources, (str, bytes)) or not isinstance(sources, Sequence) or not sources:
        raise AuthenticatedStepError("SFT step requires typed nonempty sources")
    if any(type(source) is not ReferenceTrajectorySFTSource for source in sources):
        raise TypeError("SFT sources must be ReferenceTrajectorySFTSource values")
    source_digest = _sft_source_digest(sources)
    return _execute_atomic_step(
        runtime,
        coordinate,
        optimizer_spec,
        source_digest,
        "sft",
        verified_plan,
        sources,
    )


def execute_authenticated_rl_step(
    runtime: SealedQwenRuntime,
    verified_plan: _VerifiedStreamingObjectivePlan,
    records: Sequence[AuthenticatedRolloutRecord],
    episodes: Sequence[HiddenEpisode],
    optimizer_spec: AdamWOptimizerSpec,
    coordinate: AuthenticatedUpdateCoordinate,
) -> AuthenticatedStepEvidenceBundle:
    """Execute one nonauthorizing eight-rollout RL smoke step as a sealed transaction."""

    if (
        isinstance(records, (str, bytes))
        or not isinstance(records, Sequence)
        or not records
        or any(type(record) is not AuthenticatedRolloutRecord for record in records)
    ):
        raise TypeError("RL records must be typed and nonempty")
    if (
        isinstance(episodes, (str, bytes))
        or not isinstance(episodes, Sequence)
        or not episodes
        or any(type(episode) is not HiddenEpisode for episode in episodes)
    ):
        raise TypeError("RL episodes must be typed and nonempty")
    source_digest = _rl_source_digest(records, episodes)
    return _execute_atomic_step(
        runtime,
        coordinate,
        optimizer_spec,
        source_digest,
        "rl",
        verified_plan,
        (records, episodes),
    )


def authenticated_step_manifest() -> dict[str, object]:
    """Return the fixed nonauthorizing v1 transaction declaration."""

    return {
        "schema_version": AUTHENTICATED_STEP_SCHEMA_VERSION,
        "contract_id": AUTHENTICATED_STEP_CONTRACT_ID,
        "authorizes_execution": AUTHENTICATED_STEP_AUTHORIZES_EXECUTION,
        "optimizer": "internally-owned exact torch.optim.AdamW",
        "optimizer_step_count": 1,
        "scheduler": None,
        "scaler": None,
        "gradient_clipping": None,
        "failure_semantics": "byte-exact rollback or permanently corrupt sealed runtime",
        "publication": "record validated before prepared provider is atomically published",
    }
