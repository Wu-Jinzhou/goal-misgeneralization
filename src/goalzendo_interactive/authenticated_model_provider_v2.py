"""Byte-authenticated, single-device causal-LM provider for G03.

The provider in this module is deliberately an authentication boundary, not
an authorization mechanism.  It derives every digest exposed to the rollout
collector from a strict model-artifact manifest, the observed runtime, the
model configuration, and an independent byte manifest of the in-memory state.
No caller-supplied policy or provenance digest is accepted.

Fast identity/version/metadata guards run around every forward call.  The
more expensive :meth:`AuthenticatedCausalLMProvider.reauthenticate_policy_state`
method rereads every artifact and tensor byte and is intended for rollout-
group boundaries.
"""

from __future__ import annotations

import hashlib
import hmac
import importlib.metadata
import inspect
import math
import os
import platform
import stat
import sys
import textwrap
import threading
import types
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias, TypeGuard, cast

import torch

from ._json import CanonicalJSONError, dump_json, json_digest
from .action_tokenization_v2 import TokenizerBindingManifest
from .authenticated_rollouts_v2 import ModelPolicyProvenance

AUTHENTICATED_MODEL_PROVIDER_SCHEMA_VERSION = 4
AUTHENTICATED_MODEL_PROVIDER_CONTRACT_ID = "goalzendo-authenticated-model-provider-v4"

# This bridge supplies evidence only.  A separate preregistered gate must
# authorize any model execution or weight update.
AUTHENTICATED_MODEL_PROVIDER_AUTHORIZES_EXECUTION = False
AUTHENTICATED_INCREMENTAL_CACHE_SCHEMA_VERSION = 1
AUTHENTICATED_INCREMENTAL_CACHE_CONTRACT_ID = (
    "goalzendo-authenticated-incremental-kv-cache-v1"
)

_ARTIFACT_MANIFEST_DOMAIN = "goalzendo-interactive-model-artifact-manifest-v4"
_RUNTIME_MANIFEST_DOMAIN = "goalzendo-interactive-model-runtime-manifest-v4"
_TENSOR_STATE_MANIFEST_DOMAIN = "goalzendo-interactive-model-tensor-state-manifest-v4"
_SEMANTIC_ATTRIBUTE_MANIFEST_DOMAIN = (
    "goalzendo-interactive-model-semantic-attribute-manifest-v1"
)
_SEMANTIC_ATTRIBUTE_GUARD_DOMAIN = (
    "goalzendo-interactive-model-semantic-attribute-fast-guard-v1"
)
_EXECUTABLE_MANIFEST_DOMAIN = "goalzendo-interactive-model-executable-manifest-v2"
_EXECUTABLE_CODE_DOMAIN = "goalzendo-interactive-python-callable-code-v2"
_TRAINABLE_PARAMETER_REGISTRY_DOMAIN = (
    "goalzendo-interactive-trainable-parameter-registry-v1"
)
_MODEL_CONFIG_DOMAIN = "goalzendo-interactive-model-config-v4"
_POLICY_STATE_DOMAIN = "goalzendo-interactive-model-policy-state-v4"
_CACHE_COMPARISON_DOMAIN = "goalzendo-interactive-incremental-cache-comparison-v1"
_CACHE_TRACE_DOMAIN = "goalzendo-interactive-incremental-cache-trace-v1"
_CACHE_STRUCTURE_DOMAIN = "goalzendo-interactive-derived-cache-structure-v1"

JSONScalar: TypeAlias = bool | int | float | str | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


class AuthenticatedModelProviderError(ValueError):
    """Raised when a model or its authentication evidence fails closed."""


def _is_immutable_revision(value: object) -> bool:
    return (
        type(value) is str
        and len(value) in {40, 64}
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_model_identity(model_identifier: object, revision: object) -> tuple[str, str]:
    if type(model_identifier) is not str or not model_identifier:
        raise AuthenticatedModelProviderError("model_identifier must be nonempty text")
    try:
        model_identifier.encode("ascii")
    except UnicodeEncodeError as exc:
        raise AuthenticatedModelProviderError("model_identifier must be ASCII") from exc
    if any(character.isspace() for character in model_identifier):
        raise AuthenticatedModelProviderError("model_identifier may not contain whitespace")
    if not _is_immutable_revision(revision):
        raise AuthenticatedModelProviderError(
            "revision must be a lowercase 40- or 64-hex immutable commit"
        )
    return model_identifier, cast(str, revision)


def _hash_regular_file(path: Path) -> tuple[int, str]:
    """Hash one file while rejecting symlinks and observable replacement races."""

    try:
        before_path = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise AuthenticatedModelProviderError(f"could not stat model artifact {path.name!r}") from exc
    if not stat.S_ISREG(before_path.st_mode):
        raise AuthenticatedModelProviderError("model artifact tree may contain only regular files")

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            before_fd = os.fstat(handle.fileno())
            if not stat.S_ISREG(before_fd.st_mode):
                raise AuthenticatedModelProviderError(
                    "model artifact changed type while its bytes were read"
                )
            while block := handle.read(1024 * 1024):
                digest.update(block)
            after_fd = os.fstat(handle.fileno())
        after_path = path.stat(follow_symlinks=False)
    except AuthenticatedModelProviderError:
        raise
    except OSError as exc:
        raise AuthenticatedModelProviderError(
            f"could not read model artifact {path.name!r}"
        ) from exc

    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(
        getattr(before_path, field) != getattr(before_fd, field)
        or getattr(before_fd, field) != getattr(after_fd, field)
        or getattr(after_fd, field) != getattr(after_path, field)
        for field in stable_fields
    ):
        raise AuthenticatedModelProviderError("model artifact changed while its bytes were read")
    return before_fd.st_size, digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ModelArtifactFile:
    """One path-independent regular-file attestation."""

    relative_path: str
    size_bytes: int
    sha256: str

    def as_obj(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class ModelArtifactManifest:
    """Strict manifest of all regular files below a model-artifact root."""

    model_identifier: str
    revision: str
    files: tuple[ModelArtifactFile, ...]
    schema_version: int = AUTHENTICATED_MODEL_PROVIDER_SCHEMA_VERSION
    contract_id: str = AUTHENTICATED_MODEL_PROVIDER_CONTRACT_ID

    def as_obj(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "model_identifier": self.model_identifier,
            "revision": self.revision,
            "files": [record.as_obj() for record in self.files],
        }

    @property
    def digest(self) -> str:
        return json_digest(self.as_obj(), domain=_ARTIFACT_MANIFEST_DOMAIN)

    def to_json(self) -> str:
        return dump_json({**self.as_obj(), "digest": self.digest})


def build_model_artifact_manifest(
    artifact_root: str | os.PathLike[str],
    *,
    model_identifier: str,
    revision: str,
) -> ModelArtifactManifest:
    """Read and hash an entire immutable model-artifact tree.

    Absolute roots never enter the manifest.  Symlinks, special files, empty
    trees, noncanonical relative paths, and observable read races are errors.
    """

    identifier, immutable_revision = _require_model_identity(model_identifier, revision)
    root = Path(artifact_root)
    try:
        root_lstat = root.lstat()
    except OSError as exc:
        raise AuthenticatedModelProviderError("model artifact root is not readable") from exc
    if stat.S_ISLNK(root_lstat.st_mode) or not stat.S_ISDIR(root_lstat.st_mode):
        raise AuthenticatedModelProviderError("model artifact root must be a real directory")

    try:
        candidates = sorted(root.rglob("*"), key=lambda path: path.relative_to(root).as_posix())
    except OSError as exc:
        raise AuthenticatedModelProviderError("model artifact tree could not be enumerated") from exc

    records: list[ModelArtifactFile] = []
    for candidate in candidates:
        try:
            candidate_lstat = candidate.lstat()
        except OSError as exc:
            raise AuthenticatedModelProviderError("model artifact entry could not be inspected") from exc
        if stat.S_ISLNK(candidate_lstat.st_mode):
            raise AuthenticatedModelProviderError("model artifact tree may not contain symlinks")
        if stat.S_ISDIR(candidate_lstat.st_mode):
            continue
        if not stat.S_ISREG(candidate_lstat.st_mode):
            raise AuthenticatedModelProviderError(
                "model artifact tree may contain only directories and regular files"
            )
        relative_path = candidate.relative_to(root).as_posix()
        if (
            not relative_path
            or relative_path.startswith("/")
            or "\\" in relative_path
            or any(part in {"", ".", ".."} for part in relative_path.split("/"))
        ):
            raise AuthenticatedModelProviderError("model artifact has a noncanonical relative path")
        size_bytes, sha256 = _hash_regular_file(candidate)
        records.append(ModelArtifactFile(relative_path, size_bytes, sha256))

    if not records:
        raise AuthenticatedModelProviderError("model artifact tree must contain at least one file")
    manifest = ModelArtifactManifest(identifier, immutable_revision, tuple(records))
    # Force canonical serialization at the trust boundary.
    try:
        _ = manifest.to_json()
    except CanonicalJSONError as exc:
        raise AuthenticatedModelProviderError("model artifact manifest is not canonical JSON") from exc
    return manifest


def _callable_owner(selected_class: type[object], attribute: str) -> tuple[type[object], object]:
    for owner in selected_class.__mro__:
        if attribute in owner.__dict__:
            return owner, inspect.getattr_static(selected_class, attribute)
    raise AuthenticatedModelProviderError(
        f"module class does not resolve a {attribute!r} implementation"
    )


def _canonical_code_constant(value: object) -> JSONValue:
    """Return a path-independent representation of a Python code constant."""

    if value is None or type(value) in {bool, int, str}:
        return cast(JSONScalar, value)
    if type(value) is float:
        return {"kind": "float", "hex": value.hex()}
    if type(value) is complex:
        return {
            "kind": "complex",
            "real_hex": value.real.hex(),
            "imag_hex": value.imag.hex(),
        }
    if type(value) is bytes:
        return {"kind": "bytes", "hex": value.hex()}
    if value is Ellipsis:
        return {"kind": "ellipsis"}
    if type(value) is tuple:
        return {
            "kind": "tuple",
            "items": [_canonical_code_constant(item) for item in cast(tuple[object, ...], value)],
        }
    if type(value) is frozenset:
        items = [_canonical_code_constant(item) for item in cast(frozenset[object], value)]
        return {
            "kind": "frozenset",
            "items": sorted(items, key=dump_json),
        }
    if type(value) is slice:
        selected = value
        return {
            "kind": "slice",
            "start": _canonical_code_constant(selected.start),
            "stop": _canonical_code_constant(selected.stop),
            "step": _canonical_code_constant(selected.step),
        }
    if isinstance(value, types.CodeType):
        return {"kind": "code", "manifest": _code_object(value)}
    raise AuthenticatedModelProviderError(
        "resolved executable contains a Python code constant that cannot be canonicalized: "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _code_object(code: types.CodeType) -> dict[str, JSONValue]:
    result: dict[str, JSONValue] = {
        "argcount": code.co_argcount,
        "posonlyargcount": code.co_posonlyargcount,
        "kwonlyargcount": code.co_kwonlyargcount,
        "nlocals": code.co_nlocals,
        "stacksize": code.co_stacksize,
        "flags": code.co_flags,
        "bytecode_sha256": hashlib.sha256(code.co_code).hexdigest(),
        "constants": [_canonical_code_constant(item) for item in code.co_consts],
        "names": list(code.co_names),
        "varnames": list(code.co_varnames),
        "freevars": list(code.co_freevars),
        "cellvars": list(code.co_cellvars),
    }
    exception_table = getattr(code, "co_exceptiontable", b"")
    if type(exception_table) is not bytes:
        raise AuthenticatedModelProviderError("Python code exception table is not bytes")
    result["exception_table_sha256"] = hashlib.sha256(exception_table).hexdigest()
    return result


def _canonical_callable_default(value: object) -> JSONValue:
    """Canonicalize the conservative default-value subset used by model methods."""

    if value is None or type(value) in {bool, int, str}:
        return cast(JSONScalar, value)
    if type(value) is float:
        return {"kind": "float", "hex": value.hex()}
    if type(value) is tuple:
        return {
            "kind": "tuple",
            "items": [
                _canonical_callable_default(item) for item in cast(tuple[object, ...], value)
            ],
        }
    raise AuthenticatedModelProviderError(
        "resolved executable has an unsupported mutable or process-specific default value: "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _resolved_python_callable(
    selected_class: type[object],
    attribute: str,
) -> tuple[type[object], Callable[..., object]]:
    owner, raw = _callable_owner(selected_class, attribute)
    if isinstance(raw, (classmethod, staticmethod)):
        raw = raw.__func__
    if not inspect.isfunction(raw) or not isinstance(raw.__code__, types.CodeType):
        raise AuthenticatedModelProviderError(
            f"resolved {attribute!r} implementation must be a Python function with inspectable code"
        )
    expected_qualname = f"{owner.__qualname__}.{attribute}"
    special_module_alias = (
        owner is torch.nn.Module
        and (
            (attribute == "forward" and raw.__qualname__ == "_forward_unimplemented")
            or (attribute == "__call__" and raw.__qualname__ == "Module._wrapped_call_impl")
        )
    )
    if raw.__module__ != owner.__module__ or (
        raw.__qualname__ != expected_qualname and not special_module_alias
    ):
        raise AuthenticatedModelProviderError(
            f"class {attribute} monkeypatch or noncanonical alias is forbidden"
        )
    return owner, cast(Callable[..., object], raw)


@dataclass(frozen=True, slots=True)
class ExecutableCallableRecord:
    """Canonical source/code evidence for one resolved module-class callable."""

    module_class: str
    role: str
    owner_class: str
    callable_module: str
    callable_qualname: str
    source_sha256: str
    code_sha256: str

    def as_obj(self) -> dict[str, object]:
        return {
            "module_class": self.module_class,
            "role": self.role,
            "owner_class": self.owner_class,
            "callable_module": self.callable_module,
            "callable_qualname": self.callable_qualname,
            "source_sha256": self.source_sha256,
            "code_sha256": self.code_sha256,
        }


@dataclass(frozen=True, slots=True)
class ExecutableBehaviorManifest:
    """Resolved ``forward``/``__call__`` source and code for every module class."""

    callables: tuple[ExecutableCallableRecord, ...]
    schema_version: int = AUTHENTICATED_MODEL_PROVIDER_SCHEMA_VERSION
    contract_id: str = AUTHENTICATED_MODEL_PROVIDER_CONTRACT_ID

    def as_obj(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "callables": [record.as_obj() for record in self.callables],
        }

    @property
    def digest(self) -> str:
        return json_digest(self.as_obj(), domain=_EXECUTABLE_MANIFEST_DOMAIN)

    def to_json(self) -> str:
        return dump_json({**self.as_obj(), "digest": self.digest})


@dataclass(frozen=True, slots=True)
class _ExecutableFastGuardRecord:
    module_class: str
    class_identity: int
    forward_identity: int
    forward_code_identity: int
    forward_defaults_sha256: str
    call_identity: int
    call_code_identity: int
    call_defaults_sha256: str


def _module_class_records(model: torch.nn.Module) -> tuple[tuple[str, type[object]], ...]:
    by_name: dict[str, type[object]] = {}
    for module_record in _direct_named_modules(model):
        module = module_record.module
        selected_class = type(module)
        class_name = f"{selected_class.__module__}.{selected_class.__qualname__}"
        existing = by_name.get(class_name)
        if existing is not None and existing is not selected_class:
            raise AuthenticatedModelProviderError(
                "two distinct module classes have the same canonical module/qualname"
            )
        by_name[class_name] = selected_class
    return tuple(sorted(by_name.items()))


def _executable_fast_guard(model: torch.nn.Module) -> tuple[_ExecutableFastGuardRecord, ...]:
    result: list[_ExecutableFastGuardRecord] = []
    for class_name, selected_class in _module_class_records(model):
        _, forward = _resolved_python_callable(selected_class, "forward")
        _, call = _resolved_python_callable(selected_class, "__call__")
        result.append(
            _ExecutableFastGuardRecord(
                module_class=class_name,
                class_identity=id(selected_class),
                forward_identity=id(forward),
                forward_code_identity=id(cast(Any, forward).__code__),
                forward_defaults_sha256=_callable_defaults_digest(forward),
                call_identity=id(call),
                call_code_identity=id(cast(Any, call).__code__),
                call_defaults_sha256=_callable_defaults_digest(call),
            )
        )
    return tuple(result)


def _callable_defaults_object(selected_callable: Callable[..., object]) -> dict[str, object]:
    function = cast(Any, selected_callable)
    defaults = tuple(function.__defaults__ or ())
    kwdefaults = function.__kwdefaults__ or {}
    if type(kwdefaults) is not dict or any(type(key) is not str for key in kwdefaults):
        raise AuthenticatedModelProviderError("callable keyword defaults are not canonical")
    return {
        "defaults": [_canonical_callable_default(item) for item in defaults],
        "kwdefaults": {
            key: _canonical_callable_default(kwdefaults[key]) for key in sorted(kwdefaults)
        },
    }


def _callable_defaults_digest(selected_callable: Callable[..., object]) -> str:
    return json_digest(
        _callable_defaults_object(selected_callable),
        domain="goalzendo-interactive-python-callable-defaults-v1",
    )


def _callable_manifest_record(
    *,
    class_name: str,
    selected_class: type[object],
    role: str,
) -> ExecutableCallableRecord:
    owner, selected_callable = _resolved_python_callable(selected_class, role)
    function = cast(Any, selected_callable)
    try:
        source = textwrap.dedent(inspect.getsource(selected_callable)).replace("\r\n", "\n")
    except (OSError, TypeError) as exc:
        raise AuthenticatedModelProviderError(
            f"resolved {role!r} implementation source is not inspectable"
        ) from exc
    code_obj = cast(types.CodeType, function.__code__)
    code_manifest: dict[str, object] = {
        "code": _code_object(code_obj),
        **_callable_defaults_object(selected_callable),
    }
    return ExecutableCallableRecord(
        module_class=class_name,
        role=role,
        owner_class=f"{owner.__module__}.{owner.__qualname__}",
        callable_module=cast(str, function.__module__),
        callable_qualname=cast(str, function.__qualname__),
        source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        code_sha256=json_digest(code_manifest, domain=_EXECUTABLE_CODE_DOMAIN),
    )


def _assert_no_execution_monkeypatches_or_hooks(model: torch.nn.Module) -> None:
    module_hook_names = (
        "_forward_pre_hooks",
        "_forward_hooks",
        "_backward_pre_hooks",
        "_backward_hooks",
        "_state_dict_pre_hooks",
        "_state_dict_hooks",
        "_load_state_dict_pre_hooks",
        "_load_state_dict_post_hooks",
    )
    for module_record in _direct_named_modules(model):
        module = module_record.module
        if "forward" in vars(module) or "__call__" in vars(module):
            raise AuthenticatedModelProviderError(
                "instance forward/__call__ monkeypatches are forbidden"
            )
        namespace = vars(module)
        for hook_name in module_hook_names:
            hooks = namespace.get(hook_name)
            if hooks:
                raise AuthenticatedModelProviderError(
                    "execution/state-dictionary module hooks are forbidden"
                )
    for registration in _direct_tensor_registrations(model):
        tensor = registration.tensor
        if getattr(tensor, "_backward_hooks", None):
            raise AuthenticatedModelProviderError("parameter/buffer backward hooks are forbidden")
        if getattr(tensor, "_post_accumulate_grad_hooks", None):
            raise AuthenticatedModelProviderError(
                "parameter/buffer post-accumulate-grad hooks are forbidden"
            )

    module_namespace = torch.nn.modules.module
    global_hook_names = (
        "_global_forward_pre_hooks",
        "_global_forward_hooks",
        "_global_backward_pre_hooks",
        "_global_backward_hooks",
        "_global_module_registration_hooks",
        "_global_parameter_registration_hooks",
        "_global_buffer_registration_hooks",
    )
    if any(getattr(module_namespace, name, None) for name in global_hook_names):
        raise AuthenticatedModelProviderError("global forward/pre/backward module hooks are forbidden")


def build_executable_behavior_manifest(model: torch.nn.Module) -> ExecutableBehaviorManifest:
    """Authenticate resolved Python behavior for every unique module class."""

    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    _assert_no_execution_monkeypatches_or_hooks(model)
    records = tuple(
        _callable_manifest_record(
            class_name=class_name,
            selected_class=selected_class,
            role=role,
        )
        for class_name, selected_class in _module_class_records(model)
        for role in ("__call__", "forward")
    )
    manifest = ExecutableBehaviorManifest(records)
    try:
        _ = manifest.to_json()
    except CanonicalJSONError as exc:
        raise AuthenticatedModelProviderError(
            "model executable manifest is not canonical JSON"
        ) from exc
    return manifest


@dataclass(frozen=True, slots=True)
class RuntimeStackManifest:
    """Observed software, accelerator, determinism, device, and dtype facts."""

    python_version: str
    python_implementation: str
    platform_system: str
    platform_machine: str
    byte_order: str
    torch_version: str
    transformers_version: str | None
    cuda_available: bool
    torch_cuda_version: str | None
    cudnn_version: int | None
    deterministic_algorithms: bool
    deterministic_warn_only: bool
    cudnn_deterministic: bool
    cudnn_benchmark: bool
    cuda_matmul_allow_tf32: bool
    cudnn_allow_tf32: bool
    cuda_matmul_allow_fp16_reduced_precision_reduction: bool
    cuda_matmul_allow_bf16_reduced_precision_reduction: bool
    cuda_matmul_allow_fp16_accumulation: bool
    cublas_workspace_config: str | None
    deterministic_fill_uninitialized_memory: bool
    default_dtype: str
    float32_matmul_precision: str
    grad_enabled: bool
    inference_mode_enabled: bool
    autocast_cache_enabled: bool
    autocast_device_type: str
    autocast_enabled: bool
    autocast_dtype: str
    device: str
    device_name: str | None
    device_capability: tuple[int, int] | None
    model_class: str
    model_dtypes: tuple[str, ...]
    module_modes: tuple[tuple[str, str, bool], ...]
    tensor_requires_grad: tuple[tuple[str, str, bool], ...]
    executable_manifest_sha256: str
    semantic_attribute_manifest_sha256: str
    trainable_parameter_registry_sha256: str
    schema_version: int = AUTHENTICATED_MODEL_PROVIDER_SCHEMA_VERSION
    contract_id: str = AUTHENTICATED_MODEL_PROVIDER_CONTRACT_ID

    def as_obj(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "python_version": self.python_version,
            "python_implementation": self.python_implementation,
            "platform_system": self.platform_system,
            "platform_machine": self.platform_machine,
            "byte_order": self.byte_order,
            "torch_version": self.torch_version,
            "transformers_version": self.transformers_version,
            "cuda_available": self.cuda_available,
            "torch_cuda_version": self.torch_cuda_version,
            "cudnn_version": self.cudnn_version,
            "deterministic_algorithms": self.deterministic_algorithms,
            "deterministic_warn_only": self.deterministic_warn_only,
            "cudnn_deterministic": self.cudnn_deterministic,
            "cudnn_benchmark": self.cudnn_benchmark,
            "cuda_matmul_allow_tf32": self.cuda_matmul_allow_tf32,
            "cudnn_allow_tf32": self.cudnn_allow_tf32,
            "cuda_matmul_allow_fp16_reduced_precision_reduction": (
                self.cuda_matmul_allow_fp16_reduced_precision_reduction
            ),
            "cuda_matmul_allow_bf16_reduced_precision_reduction": (
                self.cuda_matmul_allow_bf16_reduced_precision_reduction
            ),
            "cuda_matmul_allow_fp16_accumulation": (
                self.cuda_matmul_allow_fp16_accumulation
            ),
            "cublas_workspace_config": self.cublas_workspace_config,
            "deterministic_fill_uninitialized_memory": (
                self.deterministic_fill_uninitialized_memory
            ),
            "default_dtype": self.default_dtype,
            "float32_matmul_precision": self.float32_matmul_precision,
            "grad_enabled": self.grad_enabled,
            "inference_mode_enabled": self.inference_mode_enabled,
            "autocast_cache_enabled": self.autocast_cache_enabled,
            "autocast_device_type": self.autocast_device_type,
            "autocast_enabled": self.autocast_enabled,
            "autocast_dtype": self.autocast_dtype,
            "device": self.device,
            "device_name": self.device_name,
            "device_capability": (
                list(self.device_capability) if self.device_capability is not None else None
            ),
            "model_class": self.model_class,
            "model_dtypes": list(self.model_dtypes),
            "module_modes": [
                {"name": name, "class": class_name, "training": training}
                for name, class_name, training in self.module_modes
            ],
            "tensor_requires_grad": [
                {"kind": kind, "name": name, "requires_grad": requires_grad}
                for kind, name, requires_grad in self.tensor_requires_grad
            ],
            "executable_manifest_sha256": self.executable_manifest_sha256,
            "semantic_attribute_manifest_sha256": (
                self.semantic_attribute_manifest_sha256
            ),
            "trainable_parameter_registry_sha256": (
                self.trainable_parameter_registry_sha256
            ),
        }

    @property
    def digest(self) -> str:
        return json_digest(self.as_obj(), domain=_RUNTIME_MANIFEST_DOMAIN)

    def to_json(self) -> str:
        return dump_json({**self.as_obj(), "digest": self.digest})


@dataclass(frozen=True, slots=True)
class TensorStateRecord:
    """Exact direct-registration evidence for one parameter or buffer alias."""

    kind: str
    name: str
    persistent: bool | None
    dtype: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int
    storage_byte_length: int
    tensor_alias_group: int
    storage_alias_group: int
    byte_length: int
    sha256: str
    storage_sha256: str

    def as_obj(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "name": self.name,
            "persistent": self.persistent,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "stride": list(self.stride),
            "storage_offset": self.storage_offset,
            "storage_byte_length": self.storage_byte_length,
            "tensor_alias_group": self.tensor_alias_group,
            "storage_alias_group": self.storage_alias_group,
            "byte_length": self.byte_length,
            "sha256": self.sha256,
            "storage_sha256": self.storage_sha256,
        }


@dataclass(frozen=True, slots=True)
class TensorStateManifest:
    """An independently computed, name-sorted manifest of model-state bytes."""

    tensors: tuple[TensorStateRecord, ...]
    schema_version: int = AUTHENTICATED_MODEL_PROVIDER_SCHEMA_VERSION
    contract_id: str = AUTHENTICATED_MODEL_PROVIDER_CONTRACT_ID

    def as_obj(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "tensors": [record.as_obj() for record in self.tensors],
        }

    @property
    def digest(self) -> str:
        return json_digest(self.as_obj(), domain=_TENSOR_STATE_MANIFEST_DOMAIN)

    def to_json(self) -> str:
        return dump_json({**self.as_obj(), "digest": self.digest})


@dataclass(frozen=True, slots=True)
class TrainableParameterRecord:
    """Stable metadata for one identity-deduplicated model parameter."""

    canonical_name: str
    aliases: tuple[str, ...]
    dtype: str
    device: str
    layout: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int
    numel: int
    requires_grad: bool

    def as_obj(self) -> dict[str, object]:
        return {
            "canonical_name": self.canonical_name,
            "aliases": list(self.aliases),
            "dtype": self.dtype,
            "device": self.device,
            "layout": self.layout,
            "shape": list(self.shape),
            "stride": list(self.stride),
            "storage_offset": self.storage_offset,
            "numel": self.numel,
            "requires_grad": self.requires_grad,
        }


@dataclass(frozen=True, slots=True)
class TrainableParameterRegistryManifest:
    """Complete identity-deduplicated parameter inventory and trainable subset."""

    records: tuple[TrainableParameterRecord, ...]
    named_parameter_alias_count: int
    total_unique_parameter_count: int
    trainable_parameter_count: int
    total_parameter_numel: int
    trainable_parameter_numel: int
    all_parameters_trainable: bool
    schema_version: int = AUTHENTICATED_MODEL_PROVIDER_SCHEMA_VERSION
    contract_id: str = AUTHENTICATED_MODEL_PROVIDER_CONTRACT_ID

    def as_obj(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "records": [record.as_obj() for record in self.records],
            "named_parameter_alias_count": self.named_parameter_alias_count,
            "total_unique_parameter_count": self.total_unique_parameter_count,
            "trainable_parameter_count": self.trainable_parameter_count,
            "total_parameter_numel": self.total_parameter_numel,
            "trainable_parameter_numel": self.trainable_parameter_numel,
            "all_parameters_trainable": self.all_parameters_trainable,
        }

    @property
    def full_model_completeness_digest(self) -> str:
        return json_digest(self.as_obj(), domain=_TRAINABLE_PARAMETER_REGISTRY_DOMAIN)

    @property
    def digest(self) -> str:
        return self.full_model_completeness_digest

    def to_json(self) -> str:
        return dump_json({**self.as_obj(), "digest": self.digest})


_PARAMETER_REGISTRY_SEAL = object()


class TrainableParameterRegistry:
    """Provider-issued actual parameters paired with a complete stable manifest."""

    __slots__ = ("_manifest", "_parameters")

    def __init__(
        self,
        manifest: TrainableParameterRegistryManifest,
        parameters: tuple[torch.nn.Parameter, ...],
        *,
        _seal: object,
    ) -> None:
        if _seal is not _PARAMETER_REGISTRY_SEAL:
            raise TypeError("TrainableParameterRegistry is issued only by an authenticated provider")
        self._manifest = manifest
        self._parameters = parameters

    @property
    def manifest(self) -> TrainableParameterRegistryManifest:
        return self._manifest

    @property
    def parameters(self) -> tuple[torch.nn.Parameter, ...]:
        return self._parameters


def is_exact_trainable_parameter_registry(
    value: object,
) -> TypeGuard[TrainableParameterRegistry]:
    return type(value) is TrainableParameterRegistry


def require_exact_trainable_parameter_registry(
    value: object,
) -> TrainableParameterRegistry:
    if not is_exact_trainable_parameter_registry(value):
        raise TypeError("registry must have exact type TrainableParameterRegistry")
    return value


def _build_trainable_parameter_registry(model: torch.nn.Module) -> TrainableParameterRegistry:
    named = tuple(
        (registration.name, cast(torch.nn.Parameter, registration.tensor))
        for registration in _direct_tensor_registrations(model)
        if registration.kind == "parameter"
    )
    if not named:
        raise AuthenticatedModelProviderError("authenticated model must expose parameters")
    grouped: dict[int, tuple[torch.nn.Parameter, list[str]]] = {}
    for name, parameter in named:
        if type(name) is not str or not name:
            raise AuthenticatedModelProviderError("model parameter names must be nonempty text")
        identity = id(parameter)
        existing = grouped.get(identity)
        if existing is None:
            grouped[identity] = (parameter, [name])
        else:
            if existing[0] is not parameter:
                raise AuthenticatedModelProviderError("parameter identity grouping is inconsistent")
            existing[1].append(name)

    rows: list[tuple[TrainableParameterRecord, torch.nn.Parameter]] = []
    for parameter, aliases_list in grouped.values():
        aliases = tuple(sorted(aliases_list))
        record = TrainableParameterRecord(
            canonical_name=aliases[0],
            aliases=aliases,
            dtype=str(parameter.dtype),
            device=str(parameter.device),
            layout=str(parameter.layout),
            shape=tuple(parameter.shape),
            stride=tuple(parameter.stride()),
            storage_offset=int(parameter.storage_offset()),
            numel=int(parameter.numel()),
            requires_grad=bool(parameter.requires_grad),
        )
        rows.append((record, parameter))
    rows.sort(key=lambda item: item[0].canonical_name)
    records = tuple(record for record, _ in rows)
    trainable_parameters = tuple(
        parameter for record, parameter in rows if record.requires_grad
    )
    manifest = TrainableParameterRegistryManifest(
        records=records,
        named_parameter_alias_count=len(named),
        total_unique_parameter_count=len(records),
        trainable_parameter_count=len(trainable_parameters),
        total_parameter_numel=sum(record.numel for record in records),
        trainable_parameter_numel=sum(
            record.numel for record in records if record.requires_grad
        ),
        all_parameters_trainable=all(record.requires_grad for record in records),
    )
    try:
        _ = manifest.to_json()
    except CanonicalJSONError as exc:
        raise AuthenticatedModelProviderError(
            "trainable-parameter registry is not canonical JSON"
        ) from exc
    return TrainableParameterRegistry(
        manifest,
        trainable_parameters,
        _seal=_PARAMETER_REGISTRY_SEAL,
    )


def _tensor_raw_bytes(tensor: torch.Tensor) -> bytes:
    if tensor.layout is not torch.strided or tensor.is_quantized or tensor.device.type == "meta":
        raise AuthenticatedModelProviderError(
            "authenticated tensor state must use materialized, nonquantized strided tensors"
        )
    try:
        flat = tensor.detach().to(device="cpu").contiguous().reshape(-1)
        return flat.view(torch.uint8).numpy().tobytes(order="C")
    except (RuntimeError, TypeError, ValueError) as exc:
        raise AuthenticatedModelProviderError("could not materialize exact tensor-state bytes") from exc


def _storage_raw_bytes(tensor: torch.Tensor) -> bytes:
    """Materialize every byte reachable through a tensor's untyped storage."""

    try:
        storage = tensor.untyped_storage()
        byte_length = int(storage.nbytes())
        view = torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(
            storage,
            0,
            (byte_length,),
            (1,),
        )
        return view.detach().to(device="cpu").contiguous().numpy().tobytes(order="C")
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise AuthenticatedModelProviderError(
            "could not materialize exact tensor-storage bytes"
        ) from exc


@dataclass(frozen=True, slots=True)
class _DirectNamedModule:
    name: str
    module: torch.nn.Module


@dataclass(frozen=True, slots=True)
class _DirectTensorRegistration:
    kind: str
    name: str
    tensor: torch.Tensor
    persistent: bool | None


def _direct_named_modules(model: torch.nn.Module) -> tuple[_DirectNamedModule, ...]:
    """Traverse Module's concrete registries without overridable enumeration APIs."""

    result: list[_DirectNamedModule] = []

    def visit(module: torch.nn.Module, prefix: str, ancestors: frozenset[int]) -> None:
        identity = id(module)
        if identity in ancestors:
            raise AuthenticatedModelProviderError("model module registry contains a cycle")
        result.append(_DirectNamedModule(prefix, module))
        namespace = vars(module)
        children = namespace.get("_modules")
        if type(children) is not dict:
            raise AuthenticatedModelProviderError("module _modules registry must be an exact dict")
        next_ancestors = ancestors | {identity}
        for local_name in sorted(children):
            if type(local_name) is not str or not local_name or "." in local_name:
                raise AuthenticatedModelProviderError(
                    "registered module names must be nonempty dot-free text"
                )
            child = children[local_name]
            if child is None:
                continue
            if not isinstance(child, torch.nn.Module):
                raise AuthenticatedModelProviderError(
                    "module _modules registry may contain only modules or None"
                )
            child_name = f"{prefix}.{local_name}" if prefix else local_name
            visit(child, child_name, next_ancestors)

    visit(model, "", frozenset())
    return tuple(result)


def _direct_tensor_registrations(
    model: torch.nn.Module,
) -> tuple[_DirectTensorRegistration, ...]:
    result: list[_DirectTensorRegistration] = []
    for module_record in _direct_named_modules(model):
        namespace = vars(module_record.module)
        parameters = namespace.get("_parameters")
        buffers = namespace.get("_buffers")
        nonpersistent = namespace.get("_non_persistent_buffers_set")
        if type(parameters) is not dict or type(buffers) is not dict:
            raise AuthenticatedModelProviderError(
                "module parameter/buffer registries must be exact dicts"
            )
        if type(nonpersistent) is not set or any(
            type(name) is not str for name in nonpersistent
        ):
            raise AuthenticatedModelProviderError(
                "module nonpersistent-buffer registry must be an exact string set"
            )
        if not nonpersistent.issubset(buffers):
            raise AuthenticatedModelProviderError(
                "nonpersistent-buffer registry names an absent buffer"
            )
        for local_name in sorted(parameters):
            if type(local_name) is not str or not local_name or "." in local_name:
                raise AuthenticatedModelProviderError(
                    "registered parameter names must be nonempty dot-free text"
                )
            value = parameters[local_name]
            if value is None:
                continue
            if not isinstance(value, torch.nn.Parameter):
                raise AuthenticatedModelProviderError(
                    "module _parameters registry may contain only Parameters or None"
                )
            name = (
                f"{module_record.name}.{local_name}"
                if module_record.name
                else local_name
            )
            result.append(_DirectTensorRegistration("parameter", name, value, None))
        for local_name in sorted(buffers):
            if type(local_name) is not str or not local_name or "." in local_name:
                raise AuthenticatedModelProviderError(
                    "registered buffer names must be nonempty dot-free text"
                )
            value = buffers[local_name]
            if value is None:
                continue
            if not isinstance(value, torch.Tensor):
                raise AuthenticatedModelProviderError(
                    "module _buffers registry may contain only Tensors or None"
                )
            name = (
                f"{module_record.name}.{local_name}"
                if module_record.name
                else local_name
            )
            result.append(
                _DirectTensorRegistration(
                    "buffer",
                    name,
                    value,
                    local_name not in nonpersistent,
                )
            )
    result.sort(key=lambda record: (record.kind, record.name))
    names = [(record.kind, record.name) for record in result]
    if len(names) != len(set(names)):
        raise AuthenticatedModelProviderError(
            "direct parameter/buffer traversal produced duplicate names"
        )
    return tuple(result)


def build_tensor_state_manifest(model: torch.nn.Module) -> TensorStateManifest:
    """Hash direct parameters and every buffer, including nonpersistent buffers.

    This path deliberately never calls ``state_dict`` and therefore cannot be
    filtered or forged by state-dictionary hooks.
    """

    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    state = _direct_tensor_registrations(model)
    if not state:
        raise AuthenticatedModelProviderError("authenticated model must have nonempty tensor state")

    records: list[TensorStateRecord] = []
    storage_alias_groups: dict[tuple[str, int], int] = {}
    tensor_alias_groups: dict[int, int] = {}
    storage_hashes: dict[tuple[str, int], str] = {}
    parameter_identities_by_storage: dict[tuple[str, int], set[int]] = {}
    for registration in state:
        value = registration.tensor
        raw = _tensor_raw_bytes(value)
        try:
            storage = value.untyped_storage()
            storage_identity = int(cast(Any, storage)._cdata)
            storage_byte_length = int(storage.nbytes())
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            raise AuthenticatedModelProviderError(
                "tensor storage alias metadata could not be authenticated"
            ) from exc
        storage_key = (str(value.device), storage_identity)
        if storage_key not in storage_alias_groups:
            storage_alias_groups[storage_key] = len(storage_alias_groups)
            storage_hashes[storage_key] = hashlib.sha256(
                _storage_raw_bytes(value)
            ).hexdigest()
        tensor_identity = id(value)
        if tensor_identity not in tensor_alias_groups:
            tensor_alias_groups[tensor_identity] = len(tensor_alias_groups)
        if registration.kind == "parameter":
            identities = parameter_identities_by_storage.setdefault(storage_key, set())
            identities.add(tensor_identity)
            if len(identities) != 1:
                raise AuthenticatedModelProviderError(
                    "distinct Parameter objects may not share authenticated storage"
                )
        records.append(
            TensorStateRecord(
                kind=registration.kind,
                name=registration.name,
                persistent=registration.persistent,
                dtype=str(value.dtype),
                shape=tuple(value.shape),
                stride=tuple(value.stride()),
                storage_offset=int(value.storage_offset()),
                storage_byte_length=storage_byte_length,
                tensor_alias_group=tensor_alias_groups[tensor_identity],
                storage_alias_group=storage_alias_groups[storage_key],
                byte_length=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
                storage_sha256=storage_hashes[storage_key],
            )
        )
    manifest = TensorStateManifest(tuple(records))
    try:
        _ = manifest.to_json()
    except CanonicalJSONError as exc:
        raise AuthenticatedModelProviderError("tensor-state manifest is not canonical JSON") from exc
    return manifest


@dataclass(frozen=True, slots=True)
class SemanticModuleRecord:
    """Canonical registration topology and ordinary state for one module path."""

    name: str
    module_class: str
    training: bool
    parameter_registrations: tuple[tuple[str, bool], ...]
    buffer_registrations: tuple[tuple[str, bool, bool], ...]
    module_registrations: tuple[tuple[str, str | None], ...]
    attributes: tuple[tuple[str, JSONValue], ...]

    def as_obj(self) -> dict[str, object]:
        return {
            "name": self.name,
            "module_class": self.module_class,
            "training": self.training,
            "parameter_registrations": [
                {"name": name, "present": present}
                for name, present in self.parameter_registrations
            ],
            "buffer_registrations": [
                {"name": name, "present": present, "persistent": persistent}
                for name, present, persistent in self.buffer_registrations
            ],
            "module_registrations": [
                {"name": name, "class": class_name}
                for name, class_name in self.module_registrations
            ],
            "attributes": {name: value for name, value in self.attributes},
        }


@dataclass(frozen=True, slots=True)
class SemanticModuleAttributeManifest:
    """Fail-closed ordinary module state outside registered parameters/buffers."""

    modules: tuple[SemanticModuleRecord, ...]
    schema_version: int = AUTHENTICATED_MODEL_PROVIDER_SCHEMA_VERSION
    contract_id: str = AUTHENTICATED_MODEL_PROVIDER_CONTRACT_ID

    def as_obj(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "modules": [record.as_obj() for record in self.modules],
        }

    @property
    def digest(self) -> str:
        return json_digest(self.as_obj(), domain=_SEMANTIC_ATTRIBUTE_MANIFEST_DOMAIN)

    def to_json(self) -> str:
        return dump_json({**self.as_obj(), "digest": self.digest})


_FRAMEWORK_MODULE_ATTRIBUTES = frozenset(
    {
        "training",
        "_parameters",
        "_buffers",
        "_modules",
        "_non_persistent_buffers_set",
        "_backward_pre_hooks",
        "_backward_hooks",
        "_is_full_backward_hook",
        "_forward_hooks",
        "_forward_hooks_with_kwargs",
        "_forward_hooks_always_called",
        "_forward_pre_hooks",
        "_forward_pre_hooks_with_kwargs",
        "_state_dict_hooks",
        "_state_dict_pre_hooks",
        "_load_state_dict_pre_hooks",
        "_load_state_dict_post_hooks",
    }
)


@dataclass(slots=True)
class _SemanticValueContext:
    byte_authentication: bool
    root_config: object
    registered_tensor_aliases: dict[int, tuple[str, ...]]
    registered_module_aliases: dict[int, tuple[str, ...]]
    tensor_alias_groups: dict[int, int]
    storage_alias_groups: dict[tuple[str, int], int]
    storage_hashes: dict[tuple[str, int], str]
    active_containers: set[int]


def _semantic_mapping_key(value: object, *, path: str) -> str:
    if type(value) is str and value:
        return value
    if type(value) is int:
        return str(value)
    raise AuthenticatedModelProviderError(
        f"{path} mapping keys must be nonempty strings or exact integers"
    )


def _semantic_tensor_value(
    tensor: torch.Tensor,
    *,
    context: _SemanticValueContext,
) -> dict[str, JSONValue]:
    registered_aliases = context.registered_tensor_aliases.get(id(tensor))
    if registered_aliases is not None:
        return {
            "kind": "registered_tensor_reference",
            "aliases": list(registered_aliases),
        }
    if tensor.layout is not torch.strided or tensor.is_quantized or tensor.device.type == "meta":
        raise AuthenticatedModelProviderError(
            "unregistered semantic tensors must be materialized, nonquantized, and strided"
        )
    try:
        storage = tensor.untyped_storage()
        storage_identity = int(cast(Any, storage)._cdata)
        storage_byte_length = int(storage.nbytes())
        storage_data_pointer = int(storage.data_ptr())
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise AuthenticatedModelProviderError(
            "unregistered semantic tensor storage could not be inspected"
        ) from exc
    storage_key = (str(tensor.device), storage_identity)
    tensor_identity = id(tensor)
    if tensor_identity not in context.tensor_alias_groups:
        context.tensor_alias_groups[tensor_identity] = len(context.tensor_alias_groups)
    if storage_key not in context.storage_alias_groups:
        context.storage_alias_groups[storage_key] = len(context.storage_alias_groups)
    result: dict[str, JSONValue] = {
        "kind": "unregistered_tensor",
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "layout": str(tensor.layout),
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "storage_offset": int(tensor.storage_offset()),
        "storage_byte_length": storage_byte_length,
        "tensor_alias_group": context.tensor_alias_groups[tensor_identity],
        "storage_alias_group": context.storage_alias_groups[storage_key],
        "requires_grad": bool(tensor.requires_grad),
    }
    if context.byte_authentication:
        raw = _tensor_raw_bytes(tensor)
        if storage_key not in context.storage_hashes:
            context.storage_hashes[storage_key] = hashlib.sha256(
                _storage_raw_bytes(tensor)
            ).hexdigest()
        result.update(
            {
                "byte_length": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "storage_sha256": context.storage_hashes[storage_key],
            }
        )
    else:
        try:
            result.update(
                {
                    "identity": tensor_identity,
                    "version": int(tensor._version),
                    "data_pointer": int(tensor.data_ptr()),
                    "storage_data_pointer": storage_data_pointer,
                }
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            raise AuthenticatedModelProviderError(
                "unregistered semantic tensor mutation metadata is unavailable"
            ) from exc
    return result


def _canonicalize_semantic_value(
    value: object,
    *,
    path: str,
    context: _SemanticValueContext,
) -> JSONValue:
    if value is None or type(value) in {bool, int, str}:
        return cast(JSONScalar, value)
    if type(value) is float:
        return {"kind": "float", "hex": value.hex()}
    if isinstance(value, torch.Tensor):
        return _semantic_tensor_value(value, context=context)
    if value is context.root_config:
        return {"kind": "root_model_config_reference"}
    registered_modules = context.registered_module_aliases.get(id(value))
    if registered_modules is not None:
        return {
            "kind": "registered_module_reference",
            "aliases": list(registered_modules),
        }
    if isinstance(value, torch.dtype):
        return {"kind": "torch_dtype", "value": str(value)}
    if type(value) is torch.device:
        return {"kind": "torch_device", "value": str(value)}
    if type(value) is slice:
        selected_slice = value
        return {
            "kind": "slice",
            "start": _canonicalize_semantic_value(
                selected_slice.start,
                path=f"{path}.start",
                context=context,
            ),
            "stop": _canonicalize_semantic_value(
                selected_slice.stop,
                path=f"{path}.stop",
                context=context,
            ),
            "step": _canonicalize_semantic_value(
                selected_slice.step,
                path=f"{path}.step",
                context=context,
            ),
        }
    if inspect.isfunction(value):
        function = cast(Callable[..., object], value)
        dynamic = cast(Any, function)
        if type(dynamic.__module__) is not str or type(dynamic.__qualname__) is not str:
            raise AuthenticatedModelProviderError(
                f"{path} Python callable has noncanonical identity labels"
            )
        try:
            source = textwrap.dedent(inspect.getsource(function)).replace("\r\n", "\n")
        except (OSError, TypeError) as exc:
            raise AuthenticatedModelProviderError(
                f"{path} Python callable source is not inspectable"
            ) from exc
        closure_values: list[JSONValue] = []
        closure = dynamic.__closure__ or ()
        for index, cell in enumerate(closure):
            try:
                cell_value = cell.cell_contents
            except ValueError as exc:
                raise AuthenticatedModelProviderError(
                    f"{path} Python callable has an empty closure cell"
                ) from exc
            closure_values.append(
                _canonicalize_semantic_value(
                    cell_value,
                    path=f"{path}.closure[{index}]",
                    context=context,
                )
            )
        return {
            "kind": "python_function",
            "module": dynamic.__module__,
            "qualname": dynamic.__qualname__,
            "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "code": _code_object(cast(types.CodeType, dynamic.__code__)),
            "defaults": cast(
                JSONValue,
                _callable_defaults_object(function),
            ),
            "closure": closure_values,
        }

    identity = id(value)
    if identity in context.active_containers:
        raise AuthenticatedModelProviderError(f"{path} contains a reference cycle")
    context.active_containers.add(identity)
    try:
        if type(value) is tuple:
            return {
                "kind": "tuple",
                "items": [
                    _canonicalize_semantic_value(
                        item,
                        path=f"{path}[{index}]",
                        context=context,
                    )
                    for index, item in enumerate(cast(tuple[object, ...], value))
                ],
            }
        if type(value) is list:
            return {
                "kind": "list",
                "items": [
                    _canonicalize_semantic_value(
                        item,
                        path=f"{path}[{index}]",
                        context=context,
                    )
                    for index, item in enumerate(cast(list[object], value))
                ],
            }
        if isinstance(value, Mapping):
            selected_mapping = cast(Mapping[object, object], value)
            normalized: dict[str, object] = {}
            for key, item in selected_mapping.items():
                normalized_key = _semantic_mapping_key(key, path=path)
                if normalized_key in normalized:
                    raise AuthenticatedModelProviderError(
                        f"{path} has a key collision after integer canonicalization"
                    )
                normalized[normalized_key] = item
            return {
                "kind": "mapping",
                "items": {
                    key: _canonicalize_semantic_value(
                        normalized[key],
                        path=f"{path}.{key}",
                        context=context,
                    )
                    for key in sorted(normalized)
                },
            }
        if type(value) in {set, frozenset}:
            selected_set = cast(set[object] | frozenset[object], value)
            items = [
                _canonicalize_semantic_value(
                    item,
                    path=f"{path}[]",
                    context=context,
                )
                for item in selected_set
            ]
            return {
                "kind": "set" if type(value) is set else "frozenset",
                "items": sorted(items, key=dump_json),
            }
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict) and "to_dict" not in vars(value):
            try:
                materialized = to_dict()
            except Exception as exc:
                raise AuthenticatedModelProviderError(
                    f"{path} semantic mapping object could not be materialized"
                ) from exc
            canonical = _canonicalize_config_value(materialized, path=path)
            if not isinstance(canonical, dict):
                raise AuthenticatedModelProviderError(
                    f"{path} semantic mapping object must produce a mapping"
                )
            return {
                "kind": "semantic_mapping_object",
                "class": f"{type(value).__module__}.{type(value).__qualname__}",
                "value": canonical,
            }
    finally:
        context.active_containers.remove(identity)
    raise AuthenticatedModelProviderError(
        f"unsupported behavior-relevant module state at {path}: "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _semantic_alias_inventories(
    model: torch.nn.Module,
) -> tuple[dict[int, tuple[str, ...]], dict[int, tuple[str, ...]]]:
    tensor_names: dict[int, list[str]] = {}
    for registration in _direct_tensor_registrations(model):
        tensor_names.setdefault(id(registration.tensor), []).append(
            f"{registration.kind}:{registration.name}"
        )
    module_names: dict[int, list[str]] = {}
    for record in _direct_named_modules(model):
        module_names.setdefault(id(record.module), []).append(record.name)
    return (
        {identity: tuple(sorted(names)) for identity, names in tensor_names.items()},
        {identity: tuple(sorted(names)) for identity, names in module_names.items()},
    )


def _build_semantic_module_attribute_manifest(
    model: torch.nn.Module,
    *,
    byte_authentication: bool,
) -> SemanticModuleAttributeManifest:
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    try:
        root_config = cast(Any, model).config
    except AttributeError as exc:
        raise AuthenticatedModelProviderError("causal LM must expose model.config") from exc
    tensor_aliases, module_aliases = _semantic_alias_inventories(model)
    context = _SemanticValueContext(
        byte_authentication=byte_authentication,
        root_config=root_config,
        registered_tensor_aliases=tensor_aliases,
        registered_module_aliases=module_aliases,
        tensor_alias_groups={},
        storage_alias_groups={},
        storage_hashes={},
        active_containers=set(),
    )
    records: list[SemanticModuleRecord] = []
    for module_record in _direct_named_modules(model):
        module = module_record.module
        namespace = vars(module)
        parameters = cast(dict[str, object], namespace["_parameters"])
        buffers = cast(dict[str, object], namespace["_buffers"])
        children = cast(dict[str, object], namespace["_modules"])
        nonpersistent = cast(set[str], namespace["_non_persistent_buffers_set"])
        attributes = tuple(
            (
                name,
                _canonicalize_semantic_value(
                    namespace[name],
                    path=f"module[{module_record.name!r}].{name}",
                    context=context,
                ),
            )
            for name in sorted(namespace)
            if name not in _FRAMEWORK_MODULE_ATTRIBUTES
        )
        records.append(
            SemanticModuleRecord(
                name=module_record.name,
                module_class=f"{type(module).__module__}.{type(module).__qualname__}",
                training=bool(module.training),
                parameter_registrations=tuple(
                    (name, parameters[name] is not None) for name in sorted(parameters)
                ),
                buffer_registrations=tuple(
                    (
                        name,
                        buffers[name] is not None,
                        name not in nonpersistent,
                    )
                    for name in sorted(buffers)
                ),
                module_registrations=tuple(
                    (
                        name,
                        (
                            None
                            if children[name] is None
                            else (
                                f"{type(children[name]).__module__}."
                                f"{type(children[name]).__qualname__}"
                            )
                        ),
                    )
                    for name in sorted(children)
                ),
                attributes=attributes,
            )
        )
    manifest = SemanticModuleAttributeManifest(tuple(records))
    try:
        _ = manifest.to_json()
    except CanonicalJSONError as exc:
        raise AuthenticatedModelProviderError(
            "semantic module-attribute manifest is not canonical JSON"
        ) from exc
    return manifest


def build_semantic_module_attribute_manifest(
    model: torch.nn.Module,
) -> SemanticModuleAttributeManifest:
    """Authenticate ordinary behavior-relevant attributes and tensor bytes."""

    return _build_semantic_module_attribute_manifest(model, byte_authentication=True)


def _semantic_attribute_fast_guard(model: torch.nn.Module) -> str:
    manifest = _build_semantic_module_attribute_manifest(
        model,
        byte_authentication=False,
    )
    return json_digest(manifest.as_obj(), domain=_SEMANTIC_ATTRIBUTE_GUARD_DOMAIN)


def _canonicalize_config_value(value: object, *, path: str = "config") -> JSONValue:
    if value is None or type(value) in {bool, int, str}:
        return cast(JSONScalar, value)
    if type(value) is float:
        # dump_json is the final non-finite check and unique spelling authority.
        return value
    if isinstance(value, Mapping):
        result: dict[str, JSONValue] = {}
        for key, item in value.items():
            if type(key) is str and key:
                canonical_key = key
            elif type(key) is int:
                canonical_key = str(key)
            else:
                raise AuthenticatedModelProviderError(
                    f"{path} keys must be nonempty strings or exact integers"
                )
            if canonical_key in result:
                raise AuthenticatedModelProviderError(
                    f"duplicate {path} key after integer canonicalization"
                )
            result[canonical_key] = _canonicalize_config_value(
                item,
                path=f"{path}.{canonical_key}",
            )
        return result
    if isinstance(value, (list, tuple)):
        return [
            _canonicalize_config_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise AuthenticatedModelProviderError(
        f"{path} contains unsupported non-JSON value type {type(value).__name__}"
    )


def _model_config_object(model: torch.nn.Module) -> dict[str, JSONValue]:
    try:
        config = cast(Any, model).config
    except AttributeError as exc:
        raise AuthenticatedModelProviderError("causal LM must expose model.config") from exc
    try:
        to_dict = getattr(config, "to_dict", None)
        raw_config = to_dict() if callable(to_dict) else config
    except Exception as exc:
        raise AuthenticatedModelProviderError("model.config could not be materialized") from exc
    canonical = _canonicalize_config_value(raw_config)
    if not isinstance(canonical, dict):
        raise AuthenticatedModelProviderError("model.config must materialize to a JSON object")
    try:
        _ = dump_json(canonical)
    except CanonicalJSONError as exc:
        raise AuthenticatedModelProviderError("model.config is not strict canonical JSON") from exc
    return canonical


def _model_config_digest(model: torch.nn.Module) -> tuple[dict[str, JSONValue], str]:
    config = _model_config_object(model)
    return config, json_digest(config, domain=_MODEL_CONFIG_DOMAIN)


@dataclass(frozen=True, slots=True)
class _TensorGuardRecord:
    kind: str
    name: str
    identity: int
    version: int
    dtype: str
    device: str
    layout: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int
    data_pointer: int
    storage_data_pointer: int
    storage_byte_length: int
    requires_grad: bool


@dataclass(frozen=True, slots=True)
class _ModuleGuardRecord:
    name: str
    identity: int
    class_name: str
    training: bool


@dataclass(frozen=True, slots=True)
class _ExecutionGuardRecord:
    """Mutable ambient runtime settings that can change model arithmetic."""

    deterministic_algorithms: bool
    deterministic_warn_only: bool
    deterministic_fill_uninitialized_memory: bool
    cudnn_deterministic: bool
    cudnn_benchmark: bool
    cuda_matmul_allow_tf32: bool
    cudnn_allow_tf32: bool
    cuda_matmul_allow_fp16_reduced_precision_reduction: bool
    cuda_matmul_allow_bf16_reduced_precision_reduction: bool
    cuda_matmul_allow_fp16_accumulation: bool
    cublas_workspace_config: str | None
    default_dtype: str
    float32_matmul_precision: str
    grad_enabled: bool
    inference_mode_enabled: bool
    autocast_cache_enabled: bool
    autocast_device_type: str
    autocast_enabled: bool
    autocast_dtype: str


def _tensor_guard_records(model: torch.nn.Module) -> tuple[_TensorGuardRecord, ...]:
    named = [
        (registration.kind, registration.name, registration.tensor)
        for registration in _direct_tensor_registrations(model)
    ]
    if not named:
        raise AuthenticatedModelProviderError("authenticated model must expose parameters or buffers")
    result: list[_TensorGuardRecord] = []
    for kind, name, tensor in sorted(named, key=lambda item: (item[0], item[1])):
        if tensor.layout is not torch.strided:
            raise AuthenticatedModelProviderError(
                "authenticated parameters and buffers must be strided tensors"
            )
        try:
            # PyTorch exposes this counter specifically for in-place mutation tracking.
            version = int(tensor._version)
            data_pointer = int(tensor.data_ptr())
            storage = tensor.untyped_storage()
            storage_data_pointer = int(storage.data_ptr())
            storage_byte_length = int(storage.nbytes())
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            raise AuthenticatedModelProviderError("tensor mutation metadata is unavailable") from exc
        result.append(
            _TensorGuardRecord(
                kind=kind,
                name=name,
                identity=id(tensor),
                version=version,
                dtype=str(tensor.dtype),
                device=str(tensor.device),
                layout=str(tensor.layout),
                shape=tuple(tensor.shape),
                stride=tuple(tensor.stride()),
                storage_offset=int(tensor.storage_offset()),
                data_pointer=data_pointer,
                storage_data_pointer=storage_data_pointer,
                storage_byte_length=storage_byte_length,
                requires_grad=bool(tensor.requires_grad),
            )
        )
    return tuple(result)


def _module_guard_records(model: torch.nn.Module) -> tuple[_ModuleGuardRecord, ...]:
    return tuple(
        _ModuleGuardRecord(
            name=record.name,
            identity=id(record.module),
            class_name=(
                f"{type(record.module).__module__}.{type(record.module).__qualname__}"
            ),
            training=bool(record.module.training),
        )
        for record in _direct_named_modules(model)
    )


def _single_model_device(records: tuple[_TensorGuardRecord, ...]) -> torch.device:
    devices = {record.device for record in records}
    if len(devices) != 1:
        raise AuthenticatedModelProviderError("authenticated model must occupy exactly one device")
    device = torch.device(next(iter(devices)))
    if device.type == "meta":
        raise AuthenticatedModelProviderError("authenticated model may not use the meta device")
    return device


def _execution_guard(device: torch.device) -> _ExecutionGuardRecord:
    try:
        autocast_enabled = bool(torch.is_autocast_enabled(device.type))
        autocast_dtype = str(torch.get_autocast_dtype(device.type))
    except (RuntimeError, TypeError, ValueError) as exc:
        raise AuthenticatedModelProviderError(
            "autocast facts are unavailable for the authenticated device"
        ) from exc
    return _ExecutionGuardRecord(
        deterministic_algorithms=bool(torch.are_deterministic_algorithms_enabled()),
        deterministic_warn_only=bool(torch.is_deterministic_algorithms_warn_only_enabled()),
        deterministic_fill_uninitialized_memory=bool(
            cast(Any, torch.utils.deterministic).fill_uninitialized_memory
        ),
        cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
        cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
        cuda_matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
        cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),
        cuda_matmul_allow_fp16_reduced_precision_reduction=bool(
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        ),
        cuda_matmul_allow_bf16_reduced_precision_reduction=bool(
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        ),
        cuda_matmul_allow_fp16_accumulation=bool(
            torch.backends.cuda.matmul.allow_fp16_accumulation
        ),
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        default_dtype=str(torch.get_default_dtype()),
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        grad_enabled=bool(torch.is_grad_enabled()),
        inference_mode_enabled=bool(torch.is_inference_mode_enabled()),
        autocast_cache_enabled=bool(torch.is_autocast_cache_enabled()),
        autocast_device_type=device.type,
        autocast_enabled=autocast_enabled,
        autocast_dtype=autocast_dtype,
    )


def _installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _runtime_manifest(
    model: torch.nn.Module,
    device: torch.device,
    tensor_records: tuple[_TensorGuardRecord, ...],
    *,
    executable_manifest_sha256: str,
    semantic_attribute_manifest_sha256: str,
    trainable_parameter_registry_sha256: str,
) -> RuntimeStackManifest:
    cuda_device = device.type == "cuda"
    device_name: str | None = None
    device_capability: tuple[int, int] | None = None
    if cuda_device:
        try:
            device_name = torch.cuda.get_device_name(device)
            capability = torch.cuda.get_device_capability(device)
            device_capability = (int(capability[0]), int(capability[1]))
        except (AssertionError, RuntimeError, ValueError) as exc:
            raise AuthenticatedModelProviderError("CUDA device facts could not be authenticated") from exc
    cudnn_version_function = cast(
        Callable[[], int | None],
        torch.backends.cudnn.version,
    )
    cudnn_version_raw = cudnn_version_function()
    cudnn_version = int(cudnn_version_raw) if cudnn_version_raw is not None else None
    torch_cuda_version_raw = torch.version.cuda
    execution = _execution_guard(device)
    module_records = _module_guard_records(model)
    return RuntimeStackManifest(
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        platform_system=platform.system(),
        platform_machine=platform.machine(),
        byte_order=sys.byteorder,
        torch_version=str(torch.__version__),
        transformers_version=_installed_version("transformers"),
        cuda_available=bool(torch.cuda.is_available()),
        torch_cuda_version=(
            str(torch_cuda_version_raw) if torch_cuda_version_raw is not None else None
        ),
        cudnn_version=cudnn_version,
        deterministic_algorithms=execution.deterministic_algorithms,
        deterministic_warn_only=execution.deterministic_warn_only,
        cudnn_deterministic=execution.cudnn_deterministic,
        cudnn_benchmark=execution.cudnn_benchmark,
        cuda_matmul_allow_tf32=execution.cuda_matmul_allow_tf32,
        cudnn_allow_tf32=execution.cudnn_allow_tf32,
        cuda_matmul_allow_fp16_reduced_precision_reduction=(
            execution.cuda_matmul_allow_fp16_reduced_precision_reduction
        ),
        cuda_matmul_allow_bf16_reduced_precision_reduction=(
            execution.cuda_matmul_allow_bf16_reduced_precision_reduction
        ),
        cuda_matmul_allow_fp16_accumulation=(
            execution.cuda_matmul_allow_fp16_accumulation
        ),
        cublas_workspace_config=execution.cublas_workspace_config,
        deterministic_fill_uninitialized_memory=(
            execution.deterministic_fill_uninitialized_memory
        ),
        default_dtype=execution.default_dtype,
        float32_matmul_precision=execution.float32_matmul_precision,
        grad_enabled=execution.grad_enabled,
        inference_mode_enabled=execution.inference_mode_enabled,
        autocast_cache_enabled=execution.autocast_cache_enabled,
        autocast_device_type=execution.autocast_device_type,
        autocast_enabled=execution.autocast_enabled,
        autocast_dtype=execution.autocast_dtype,
        device=str(device),
        device_name=device_name,
        device_capability=device_capability,
        model_class=f"{type(model).__module__}.{type(model).__qualname__}",
        model_dtypes=tuple(sorted({record.dtype for record in tensor_records})),
        module_modes=tuple(
            (record.name, record.class_name, record.training) for record in module_records
        ),
        tensor_requires_grad=tuple(
            (record.kind, record.name, record.requires_grad) for record in tensor_records
        ),
        executable_manifest_sha256=executable_manifest_sha256,
        semantic_attribute_manifest_sha256=semantic_attribute_manifest_sha256,
        trainable_parameter_registry_sha256=trainable_parameter_registry_sha256,
    )


def _policy_state_digest(
    artifact_manifest: ModelArtifactManifest,
    runtime_manifest: RuntimeStackManifest,
    *,
    model_config_sha256: str,
    tensor_state_manifest: TensorStateManifest,
    executable_manifest_sha256: str,
    semantic_attribute_manifest_sha256: str,
    trainable_parameter_registry_sha256: str,
    vocabulary_size: int,
) -> str:
    return json_digest(
        {
            "schema_version": AUTHENTICATED_MODEL_PROVIDER_SCHEMA_VERSION,
            "contract_id": AUTHENTICATED_MODEL_PROVIDER_CONTRACT_ID,
            "model_identifier": artifact_manifest.model_identifier,
            "revision": artifact_manifest.revision,
            "artifact_manifest_sha256": artifact_manifest.digest,
            "runtime_stack_sha256": runtime_manifest.digest,
            "model_config_sha256": model_config_sha256,
            "tensor_state_manifest_sha256": tensor_state_manifest.digest,
            "executable_manifest_sha256": executable_manifest_sha256,
            "semantic_attribute_manifest_sha256": (
                semantic_attribute_manifest_sha256
            ),
            "trainable_parameter_registry_sha256": (
                trainable_parameter_registry_sha256
            ),
            "vocabulary_size": vocabulary_size,
        },
        domain=_POLICY_STATE_DOMAIN,
    )


def _config_vocabulary_size(config: Mapping[str, JSONValue]) -> int:
    value = config.get("vocab_size")
    if isinstance(value, bool) or not isinstance(value, int) or value < 2:
        raise AuthenticatedModelProviderError("model.config.vocab_size must be an integer at least two")
    return value


class AuthenticatedCausalLMProvider:
    """A provenance-complete provider for one eval-mode causal LM.

    The constructor intentionally has no digest or provenance parameters.  It
    authenticates the supplied model and artifact root itself.
    """

    __slots__ = (
        "_artifact_manifest",
        "_artifact_root",
        "_config_sha256",
        "_device",
        "_executable_fast_guard",
        "_executable_manifest",
        "_execution_guard",
        "_lock",
        "_model",
        "_module_guard",
        "_provenance",
        "_runtime_manifest",
        "_semantic_attribute_fast_guard",
        "_semantic_attribute_manifest",
        "_tensor_guard",
        "_tensor_state_manifest",
        "_trainable_parameter_registry",
        "_vocabulary_size",
    )

    def __init__(
        self,
        model: torch.nn.Module,
        artifact_root: str | os.PathLike[str],
        *,
        model_identifier: str,
        revision: str,
    ) -> None:
        if not isinstance(model, torch.nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        self._model = model
        self._artifact_root = Path(artifact_root)
        self._lock = threading.RLock()

        tensor_guard = _tensor_guard_records(model)
        module_guard = _module_guard_records(model)
        _assert_no_execution_monkeypatches_or_hooks(model)
        if any(record.training for record in module_guard):
            raise AuthenticatedModelProviderError(
                "authenticated model and every submodule must already be in eval mode"
            )
        device = _single_model_device(tensor_guard)
        artifact_manifest = build_model_artifact_manifest(
            self._artifact_root,
            model_identifier=model_identifier,
            revision=revision,
        )
        config, config_sha256 = _model_config_digest(model)
        vocabulary_size = _config_vocabulary_size(config)
        tensor_state_manifest = build_tensor_state_manifest(model)
        semantic_attribute_manifest = build_semantic_module_attribute_manifest(model)
        semantic_attribute_fast_guard = _semantic_attribute_fast_guard(model)
        trainable_parameter_registry = _build_trainable_parameter_registry(model)
        executable_manifest = build_executable_behavior_manifest(model)
        executable_fast_guard = _executable_fast_guard(model)
        runtime_manifest = _runtime_manifest(
            model,
            device,
            tensor_guard,
            executable_manifest_sha256=executable_manifest.digest,
            semantic_attribute_manifest_sha256=semantic_attribute_manifest.digest,
            trainable_parameter_registry_sha256=(
                trainable_parameter_registry.manifest.digest
            ),
        )
        policy_state_sha256 = _policy_state_digest(
            artifact_manifest,
            runtime_manifest,
            model_config_sha256=config_sha256,
            tensor_state_manifest=tensor_state_manifest,
            executable_manifest_sha256=executable_manifest.digest,
            semantic_attribute_manifest_sha256=semantic_attribute_manifest.digest,
            trainable_parameter_registry_sha256=(
                trainable_parameter_registry.manifest.digest
            ),
            vocabulary_size=vocabulary_size,
        )
        provenance = ModelPolicyProvenance(
            model_identifier=artifact_manifest.model_identifier,
            revision=artifact_manifest.revision,
            artifact_manifest_sha256=artifact_manifest.digest,
            runtime_stack_sha256=runtime_manifest.digest,
            policy_state_digest=policy_state_sha256,
        )

        self._tensor_guard = tensor_guard
        self._module_guard = module_guard
        self._device = device
        self._execution_guard = _execution_guard(device)
        self._executable_fast_guard = executable_fast_guard
        self._executable_manifest = executable_manifest
        self._artifact_manifest = artifact_manifest
        self._config_sha256 = config_sha256
        self._tensor_state_manifest = tensor_state_manifest
        self._semantic_attribute_manifest = semantic_attribute_manifest
        self._semantic_attribute_fast_guard = semantic_attribute_fast_guard
        self._trainable_parameter_registry = trainable_parameter_registry
        self._runtime_manifest = runtime_manifest
        self._vocabulary_size = vocabulary_size
        self._provenance = provenance

    @property
    def policy_state_digest(self) -> str:
        with self._lock:
            self._assert_lightweight_guard()
            return self._provenance.policy_state_digest

    @property
    def model_provenance_digest(self) -> str:
        with self._lock:
            self._assert_lightweight_guard()
            return self._provenance.digest

    @property
    def model_provenance(self) -> ModelPolicyProvenance:
        with self._lock:
            self._assert_lightweight_guard()
            return self._provenance

    @property
    def artifact_manifest(self) -> ModelArtifactManifest:
        return self._artifact_manifest

    @property
    def runtime_manifest(self) -> RuntimeStackManifest:
        return self._runtime_manifest

    @property
    def tensor_state_manifest(self) -> TensorStateManifest:
        return self._tensor_state_manifest

    @property
    def executable_manifest(self) -> ExecutableBehaviorManifest:
        return self._executable_manifest

    @property
    def semantic_attribute_manifest(self) -> SemanticModuleAttributeManifest:
        return self._semantic_attribute_manifest

    @property
    def trainable_parameter_registry(self) -> TrainableParameterRegistry:
        with self._lock:
            self._assert_lightweight_guard()
            return self._trainable_parameter_registry

    @property
    def model_config_sha256(self) -> str:
        return self._config_sha256

    @property
    def vocabulary_size(self) -> int:
        return self._vocabulary_size

    def _assert_lightweight_guard(self) -> None:
        _assert_no_execution_monkeypatches_or_hooks(self._model)
        try:
            current_tensors = _tensor_guard_records(self._model)
            current_modules = _module_guard_records(self._model)
        except AuthenticatedModelProviderError:
            raise
        except Exception as exc:
            raise AuthenticatedModelProviderError("model mutation guard could not be evaluated") from exc
        if current_modules != self._module_guard:
            raise AuthenticatedModelProviderError(
                "model module identity, class, or eval-mode state changed after authentication"
            )
        if current_tensors != self._tensor_guard:
            raise AuthenticatedModelProviderError(
                "model parameter/buffer identity, version, or metadata changed after authentication"
            )
        if _executable_fast_guard(self._model) != self._executable_fast_guard:
            raise AuthenticatedModelProviderError(
                "model class/callable identity or code object changed after authentication"
            )
        if not hmac.compare_digest(
            _semantic_attribute_fast_guard(self._model),
            self._semantic_attribute_fast_guard,
        ):
            raise AuthenticatedModelProviderError(
                "model behavior-relevant module attributes changed after authentication"
            )
        if _execution_guard(self._device) != self._execution_guard:
            raise AuthenticatedModelProviderError(
                "model arithmetic runtime or autocast context changed after authentication"
            )

    def reauthenticate_policy_state(self) -> ModelPolicyProvenance:
        """Reread all artifact/config/tensor/runtime evidence and require identity.

        This is intentionally expensive and should be invoked immediately
        before and after each unchanged-policy eight-rollout group.
        """

        with self._lock:
            self._assert_lightweight_guard()
            artifact = build_model_artifact_manifest(
                self._artifact_root,
                model_identifier=self._artifact_manifest.model_identifier,
                revision=self._artifact_manifest.revision,
            )
            if not hmac.compare_digest(artifact.digest, self._artifact_manifest.digest):
                raise AuthenticatedModelProviderError(
                    "model artifact bytes or relative file manifest changed"
                )
            _, config_sha256 = _model_config_digest(self._model)
            if not hmac.compare_digest(config_sha256, self._config_sha256):
                raise AuthenticatedModelProviderError("model configuration changed")
            tensor_state = build_tensor_state_manifest(self._model)
            if not hmac.compare_digest(tensor_state.digest, self._tensor_state_manifest.digest):
                raise AuthenticatedModelProviderError("model tensor-state bytes changed")
            semantic_attributes = build_semantic_module_attribute_manifest(self._model)
            if not hmac.compare_digest(
                semantic_attributes.digest,
                self._semantic_attribute_manifest.digest,
            ):
                raise AuthenticatedModelProviderError(
                    "model semantic module-attribute state changed"
                )
            executable = build_executable_behavior_manifest(self._model)
            if not hmac.compare_digest(executable.digest, self._executable_manifest.digest):
                raise AuthenticatedModelProviderError("model executable behavior changed")
            trainable_registry = _build_trainable_parameter_registry(self._model)
            if not hmac.compare_digest(
                trainable_registry.manifest.digest,
                self._trainable_parameter_registry.manifest.digest,
            ):
                raise AuthenticatedModelProviderError(
                    "model trainable-parameter registry changed"
                )
            runtime = _runtime_manifest(
                self._model,
                self._device,
                self._tensor_guard,
                executable_manifest_sha256=executable.digest,
                semantic_attribute_manifest_sha256=semantic_attributes.digest,
                trainable_parameter_registry_sha256=trainable_registry.manifest.digest,
            )
            if not hmac.compare_digest(runtime.digest, self._runtime_manifest.digest):
                raise AuthenticatedModelProviderError("model runtime stack changed")
            policy_state = _policy_state_digest(
                artifact,
                runtime,
                model_config_sha256=config_sha256,
                tensor_state_manifest=tensor_state,
                executable_manifest_sha256=executable.digest,
                semantic_attribute_manifest_sha256=semantic_attributes.digest,
                trainable_parameter_registry_sha256=trainable_registry.manifest.digest,
                vocabulary_size=self._vocabulary_size,
            )
            if not hmac.compare_digest(policy_state, self._provenance.policy_state_digest):
                raise AuthenticatedModelProviderError("computed policy-state digest changed")
            regenerated = ModelPolicyProvenance(
                model_identifier=artifact.model_identifier,
                revision=artifact.revision,
                artifact_manifest_sha256=artifact.digest,
                runtime_stack_sha256=runtime.digest,
                policy_state_digest=policy_state,
            )
            if not hmac.compare_digest(regenerated.digest, self._provenance.digest):
                raise AuthenticatedModelProviderError("computed model-provenance digest changed")
            return self._provenance

    def _validated_input(self, input_ids: tuple[int, ...]) -> torch.Tensor:
        if type(input_ids) is not tuple or not input_ids:
            raise AuthenticatedModelProviderError("input_ids must be a nonempty tuple")
        if any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or not 0 <= token_id < self._vocabulary_size
            for token_id in input_ids
        ):
            raise AuthenticatedModelProviderError("input_ids contain an invalid vocabulary index")
        try:
            return torch.tensor((input_ids,), dtype=torch.long, device=self._device)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise AuthenticatedModelProviderError("input_ids could not be placed on the model device") from exc

    def _invoke_model(self, model_input: torch.Tensor) -> torch.Tensor:
        output = self._model(input_ids=model_input)
        candidate: object
        if isinstance(output, torch.Tensor):
            candidate = output
        elif isinstance(output, Mapping):
            candidate = output.get("logits")
        else:
            candidate = getattr(output, "logits", None)
        if not isinstance(candidate, torch.Tensor):
            raise AuthenticatedModelProviderError("causal LM output must expose Tensor logits")
        return candidate

    def _validate_logits(
        self,
        logits: torch.Tensor,
        *,
        sequence_length: int,
        require_differentiable: bool,
    ) -> torch.Tensor:
        if logits.ndim != 3 or tuple(logits.shape) != (
            1,
            sequence_length,
            self._vocabulary_size,
        ):
            raise AuthenticatedModelProviderError(
                "causal LM logits must have exact [1, sequence, vocabulary] shape"
            )
        if not logits.is_floating_point():
            raise AuthenticatedModelProviderError("causal LM logits must have a floating dtype")
        if logits.device != self._device:
            raise AuthenticatedModelProviderError("causal LM logits left the authenticated model device")
        try:
            finite = bool(torch.isfinite(logits.detach()).all().item())
        except (RuntimeError, TypeError, ValueError) as exc:
            raise AuthenticatedModelProviderError("causal LM logits could not be checked") from exc
        if not finite:
            raise AuthenticatedModelProviderError("causal LM logits contain a non-finite value")
        if require_differentiable and not logits.requires_grad:
            raise AuthenticatedModelProviderError(
                "full-forward causal LM logits must retain a differentiable graph"
            )
        return logits

    def next_token_logits(self, input_ids: tuple[int, ...]) -> torch.Tensor:
        """Return detached final-position logits under an eval/no-grad forward."""

        with self._lock:
            self._assert_lightweight_guard()
            model_input = self._validated_input(input_ids)
            try:
                with torch.no_grad():
                    logits = self._invoke_model(model_input)
            finally:
                self._assert_lightweight_guard()
            validated = self._validate_logits(
                logits,
                sequence_length=len(input_ids),
                require_differentiable=False,
            )
            result = validated[0, -1, :].detach()
            if result.ndim != 1 or result.shape[0] != self._vocabulary_size:
                raise AuthenticatedModelProviderError("final-position logits have an invalid shape")
            return result

    def full_forward_logits(self, input_ids: tuple[int, ...]) -> torch.Tensor:
        """Return differentiable ``[sequence, vocabulary]`` logits."""

        with self._lock:
            self._assert_lightweight_guard()
            model_input = self._validated_input(input_ids)
            try:
                logits = self._invoke_model(model_input)
            finally:
                self._assert_lightweight_guard()
            validated = self._validate_logits(
                logits,
                sequence_length=len(input_ids),
                require_differentiable=True,
            )
            result = validated[0]
            if result.ndim != 2 or tuple(result.shape) != (
                len(input_ids),
                self._vocabulary_size,
            ):
                raise AuthenticatedModelProviderError("full-forward logits have an invalid shape")
            return result


def is_exact_authenticated_causal_lm_provider(
    value: object,
) -> TypeGuard[AuthenticatedCausalLMProvider]:
    """Return true only for the nominal provider class, never a Protocol liar/subclass."""

    return type(value) is AuthenticatedCausalLMProvider


def require_exact_authenticated_causal_lm_provider(
    value: object,
) -> AuthenticatedCausalLMProvider:
    """Narrow a future production boundary to the exact authenticated provider type."""

    if not is_exact_authenticated_causal_lm_provider(value):
        raise TypeError("provider must have exact type AuthenticatedCausalLMProvider")
    return value


@dataclass(frozen=True, slots=True)
class _CacheTensorGuard:
    path: str
    identity: int
    version: int
    dtype: str
    device: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int
    data_pointer: int
    storage_data_pointer: int
    storage_byte_length: int


@dataclass(frozen=True, slots=True)
class _DerivedCacheGuard:
    root_identity: int
    root_class: str
    structure_digest: str
    tensors: tuple[_CacheTensorGuard, ...]


def _derived_cache_guard(cache: object, *, device: torch.device) -> _DerivedCacheGuard:
    """Describe derived cache identity/structure without making it policy state."""

    if cache is None:
        raise AuthenticatedModelProviderError("cached model output omitted past_key_values")
    tensors: list[_CacheTensorGuard] = []
    active_containers: set[int] = set()

    def visit(value: object, path: str, *, permit_hf_conversion: bool) -> JSONValue:
        if isinstance(value, torch.Tensor):
            if value.layout is not torch.strided or value.device.type == "meta":
                raise AuthenticatedModelProviderError(
                    "derived KV-cache tensors must be materialized and strided"
                )
            if value.device != device:
                raise AuthenticatedModelProviderError(
                    "derived KV-cache tensor left the authenticated model device"
                )
            if not value.is_floating_point():
                raise AuthenticatedModelProviderError(
                    "derived KV-cache tensors must have floating dtype"
                )
            if value.requires_grad:
                raise AuthenticatedModelProviderError(
                    "derived KV-cache tensors must be detached from autograd"
                )
            if not bool(torch.isfinite(value.detach()).all().item()):
                raise AuthenticatedModelProviderError(
                    "derived KV-cache tensor contains a non-finite value"
                )
            try:
                storage = value.untyped_storage()
                tensors.append(
                    _CacheTensorGuard(
                        path=path,
                        identity=id(value),
                        version=int(value._version),
                        dtype=str(value.dtype),
                        device=str(value.device),
                        shape=tuple(value.shape),
                        stride=tuple(value.stride()),
                        storage_offset=int(value.storage_offset()),
                        data_pointer=int(value.data_ptr()),
                        storage_data_pointer=int(storage.data_ptr()),
                        storage_byte_length=int(storage.nbytes()),
                    )
                )
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                raise AuthenticatedModelProviderError(
                    "derived KV-cache mutation metadata is unavailable"
                ) from exc
            return {"kind": "tensor", "path": path}
        if value is None:
            return {"kind": "none"}

        identity = id(value)
        if identity in active_containers:
            raise AuthenticatedModelProviderError("derived KV-cache contains a reference cycle")
        active_containers.add(identity)
        try:
            if type(value) is tuple:
                selected_tuple = cast(tuple[object, ...], value)
                return {
                    "kind": "tuple",
                    "items": [
                        visit(item, f"{path}[{index}]", permit_hf_conversion=False)
                        for index, item in enumerate(selected_tuple)
                    ],
                }
            if type(value) is list:
                selected_list = cast(list[object], value)
                return {
                    "kind": "list",
                    "items": [
                        visit(item, f"{path}[{index}]", permit_hf_conversion=False)
                        for index, item in enumerate(selected_list)
                    ],
                }
            if isinstance(value, Mapping):
                if any(type(key) is not str or not key for key in value):
                    raise AuthenticatedModelProviderError(
                        "derived KV-cache mappings require nonempty string keys"
                    )
                selected_mapping = cast(Mapping[str, object], value)
                return {
                    "kind": "mapping",
                    "items": {
                        key: visit(
                            selected_mapping[key],
                            f"{path}.{key}",
                            permit_hf_conversion=False,
                        )
                        for key in sorted(selected_mapping)
                    },
                }
            to_legacy_cache = getattr(value, "to_legacy_cache", None)
            if permit_hf_conversion and callable(to_legacy_cache):
                try:
                    legacy_cache = to_legacy_cache()
                except Exception as exc:
                    raise AuthenticatedModelProviderError(
                        "HuggingFace cache could not be converted to its legacy tensor view"
                    ) from exc
                return {
                    "kind": "huggingface_cache",
                    "class": f"{type(value).__module__}.{type(value).__qualname__}",
                    "legacy": visit(
                        legacy_cache,
                        f"{path}.legacy",
                        permit_hf_conversion=False,
                    ),
                }
        finally:
            active_containers.remove(identity)
        raise AuthenticatedModelProviderError(
            f"unsupported derived KV-cache node type {type(value).__name__}"
        )

    structure = visit(cache, "cache", permit_hf_conversion=True)
    if not tensors:
        raise AuthenticatedModelProviderError("derived KV-cache contains no tensors")
    return _DerivedCacheGuard(
        root_identity=id(cache),
        root_class=f"{type(cache).__module__}.{type(cache).__qualname__}",
        structure_digest=json_digest(structure, domain=_CACHE_STRUCTURE_DOMAIN),
        tensors=tuple(tensors),
    )


def _output_member(output: object, name: str) -> object:
    if isinstance(output, Mapping):
        return output.get(name)
    return getattr(output, name, None)


class AuthenticatedIncrementalCacheProvider:
    """Derived KV-cache acceleration over an authenticated stateless provider.

    Cache contents never enter the policy digest.  The exact model and runtime
    binding is delegated to ``AuthenticatedCausalLMProvider`` and checked
    around every call.  Only strict prompt extensions reuse cache state; all
    other prompts deterministically discard it and run from a cold prefix.
    """

    __slots__ = (
        "_base",
        "_cache_guard",
        "_cache_reuse_count",
        "_cached_prefix",
        "_expected_model_provenance",
        "_expected_policy_state",
        "_last_call_reused_cache",
        "_lock",
        "_past_key_values",
        "_reset_count",
    )

    def __init__(self, provider: AuthenticatedCausalLMProvider) -> None:
        if type(provider) is not AuthenticatedCausalLMProvider:
            raise TypeError("provider must be an AuthenticatedCausalLMProvider")
        self._base = provider
        self._lock = threading.RLock()
        self._expected_policy_state = provider.policy_state_digest
        self._expected_model_provenance = provider.model_provenance_digest
        self._past_key_values: object | None = None
        self._cache_guard: _DerivedCacheGuard | None = None
        self._cached_prefix: tuple[int, ...] = ()
        self._reset_count = 0
        self._cache_reuse_count = 0
        self._last_call_reused_cache = False

    def _assert_provider_binding(self) -> None:
        if not hmac.compare_digest(
            self._base.policy_state_digest,
            self._expected_policy_state,
        ):
            raise AuthenticatedModelProviderError(
                "incremental cache provider policy binding changed"
            )
        if not hmac.compare_digest(
            self._base.model_provenance_digest,
            self._expected_model_provenance,
        ):
            raise AuthenticatedModelProviderError(
                "incremental cache provider model binding changed"
            )

    @property
    def policy_state_digest(self) -> str:
        with self._lock:
            self._assert_provider_binding()
            return self._expected_policy_state

    @property
    def model_provenance_digest(self) -> str:
        with self._lock:
            self._assert_provider_binding()
            return self._expected_model_provenance

    @property
    def stateless_provider(self) -> AuthenticatedCausalLMProvider:
        return self._base

    @property
    def reset_count(self) -> int:
        with self._lock:
            self._assert_provider_binding()
            return self._reset_count

    @property
    def cache_reuse_count(self) -> int:
        with self._lock:
            self._assert_provider_binding()
            return self._cache_reuse_count

    @property
    def cached_prefix_length(self) -> int:
        with self._lock:
            self._assert_provider_binding()
            return len(self._cached_prefix)

    @property
    def last_call_reused_cache(self) -> bool:
        with self._lock:
            self._assert_provider_binding()
            return self._last_call_reused_cache

    def _discard_cache(self, *, count_reset: bool) -> None:
        if count_reset and self._past_key_values is not None:
            self._reset_count += 1
        self._past_key_values = None
        self._cache_guard = None
        self._cached_prefix = ()
        self._last_call_reused_cache = False

    def reset_cache(self) -> None:
        """Explicitly discard derived state without changing policy identity."""

        with self._lock:
            self._assert_provider_binding()
            self._discard_cache(count_reset=True)
            self._assert_provider_binding()

    def reauthenticate_policy_state(self) -> ModelPolicyProvenance:
        """Run full byte reauthentication and begin the next group cold."""

        with self._lock:
            self._assert_provider_binding()
            provenance = self._base.reauthenticate_policy_state()
            self._discard_cache(count_reset=True)
            self._assert_provider_binding()
            return provenance

    def _cache_is_reusable_for(self, input_ids: tuple[int, ...]) -> bool:
        return (
            self._past_key_values is not None
            and len(input_ids) > len(self._cached_prefix)
            and input_ids[: len(self._cached_prefix)] == self._cached_prefix
        )

    def next_token_logits(self, input_ids: tuple[int, ...]) -> torch.Tensor:
        """Return final logits while reusing cache only for a strict extension."""

        with self._lock, self._base._lock:
            self._assert_provider_binding()
            # Validate the full public prefix even when only its suffix is evaluated.
            _ = self._base._validated_input(input_ids)
            reuse = self._cache_is_reusable_for(input_ids)
            if reuse:
                if self._cache_guard is None or self._past_key_values is None:
                    raise AuthenticatedModelProviderError(
                        "derived KV-cache bookkeeping is internally inconsistent"
                    )
                current_guard = _derived_cache_guard(
                    self._past_key_values,
                    device=self._base._device,
                )
                if current_guard != self._cache_guard:
                    raise AuthenticatedModelProviderError(
                        "derived KV-cache identity, structure, or tensor metadata changed"
                    )
                suffix = input_ids[len(self._cached_prefix) :]
                past_key_values = self._past_key_values
            else:
                self._discard_cache(count_reset=True)
                suffix = input_ids
                past_key_values = None

            model_input = self._base._validated_input(suffix)
            attention_mask = torch.ones(
                (1, len(input_ids)),
                dtype=torch.long,
                device=self._base._device,
            )
            output: object
            try:
                with torch.no_grad():
                    output = self._base._model(
                        input_ids=model_input,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True,
                    )
                logits_candidate = _output_member(output, "logits")
                if not isinstance(logits_candidate, torch.Tensor):
                    raise AuthenticatedModelProviderError(
                        "cached causal LM output must expose Tensor logits"
                    )
                validated = self._base._validate_logits(
                    logits_candidate,
                    sequence_length=len(suffix),
                    require_differentiable=False,
                )
                next_cache = _output_member(output, "past_key_values")
                next_guard = _derived_cache_guard(
                    next_cache,
                    device=self._base._device,
                )
            except AuthenticatedModelProviderError:
                raise
            except Exception as exc:
                raise AuthenticatedModelProviderError(
                    "cached HuggingFace causal-LM forward failed"
                ) from exc
            finally:
                self._base._assert_lightweight_guard()
                self._assert_provider_binding()

            result = validated[0, -1, :].detach()
            if result.ndim != 1 or result.shape[0] != self._base.vocabulary_size:
                raise AuthenticatedModelProviderError(
                    "cached final-position logits have an invalid shape"
                )
            self._past_key_values = next_cache
            self._cache_guard = next_guard
            self._cached_prefix = input_ids
            self._last_call_reused_cache = reuse
            if reuse:
                self._cache_reuse_count += 1
            return result

    def full_forward_logits(self, input_ids: tuple[int, ...]) -> torch.Tensor:
        """Delegate the unchanged differentiable stateless provider path."""

        with self._lock:
            self._assert_provider_binding()
            result = self._base.full_forward_logits(input_ids)
            self._assert_provider_binding()
            return result


@dataclass(frozen=True, slots=True)
class IncrementalCacheComparison:
    """Prospective cached-versus-stateless evidence for one exact prefix trace."""

    policy_state_digest: str
    model_provenance_digest: str
    tokenizer_binding_digest: str
    tokenizer_repository_id: str
    tokenizer_revision: str
    trace_digest: str
    prefix_lengths: tuple[int, ...]
    absolute_tolerance_hex: str
    maximum_absolute_error_hex: str
    reset_count: int
    reuse_count: int
    within_tolerance: bool
    schema_version: int = AUTHENTICATED_INCREMENTAL_CACHE_SCHEMA_VERSION
    contract_id: str = AUTHENTICATED_INCREMENTAL_CACHE_CONTRACT_ID

    def as_obj(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "authorizes_execution": False,
            "policy_state_digest": self.policy_state_digest,
            "model_provenance_digest": self.model_provenance_digest,
            "tokenizer_binding_digest": self.tokenizer_binding_digest,
            "tokenizer_repository_id": self.tokenizer_repository_id,
            "tokenizer_revision": self.tokenizer_revision,
            "trace_digest": self.trace_digest,
            "prefix_lengths": list(self.prefix_lengths),
            "absolute_tolerance_hex": self.absolute_tolerance_hex,
            "maximum_absolute_error_hex": self.maximum_absolute_error_hex,
            "reset_count": self.reset_count,
            "reuse_count": self.reuse_count,
            "within_tolerance": self.within_tolerance,
        }

    @property
    def digest(self) -> str:
        return json_digest(self.as_obj(), domain=_CACHE_COMPARISON_DOMAIN)

    def to_json(self) -> str:
        return dump_json({**self.as_obj(), "digest": self.digest})


class IncrementalCacheMismatchError(AuthenticatedModelProviderError):
    """Raised with the complete prospective report when cache logits diverge."""

    def __init__(self, report: IncrementalCacheComparison) -> None:
        self.report = report
        super().__init__(
            "cached logits differ from stateless full-prefix logits beyond absolute tolerance"
        )


def compare_incremental_cache_trace(
    provider: AuthenticatedIncrementalCacheProvider,
    tokenizer_manifest: TokenizerBindingManifest,
    exact_trace: tuple[tuple[int, ...], ...],
    *,
    absolute_tolerance: float,
) -> IncrementalCacheComparison:
    """Prospectively require cache equivalence over exact public prefixes."""

    if type(provider) is not AuthenticatedIncrementalCacheProvider:
        raise TypeError("provider must be an AuthenticatedIncrementalCacheProvider")
    if type(tokenizer_manifest) is not TokenizerBindingManifest:
        raise TypeError("tokenizer_manifest must be a TokenizerBindingManifest")
    if type(exact_trace) is not tuple or not exact_trace:
        raise AuthenticatedModelProviderError("exact_trace must be a nonempty tuple of prefixes")
    if (
        isinstance(absolute_tolerance, bool)
        or not isinstance(absolute_tolerance, (int, float))
        or not math.isfinite(float(absolute_tolerance))
        or float(absolute_tolerance) < 0
    ):
        raise AuthenticatedModelProviderError(
            "absolute_tolerance must be explicitly finite and non-negative"
        )
    if tokenizer_manifest.vocabulary_size > provider.stateless_provider.vocabulary_size:
        raise AuthenticatedModelProviderError(
            "tokenizer vocabulary exceeds the authenticated model output vocabulary"
        )

    provider.reset_cache()
    resets_before = provider.reset_count
    reuses_before = provider.cache_reuse_count
    maximum_error = 0.0
    normalized_trace: list[list[int]] = []
    for prefix in exact_trace:
        if type(prefix) is not tuple:
            raise AuthenticatedModelProviderError("every exact_trace prefix must be a tuple")
        cached = provider.next_token_logits(prefix)
        stateless = provider.stateless_provider.next_token_logits(prefix)
        error = torch.max(
            torch.abs(
                cached.detach().to(device="cpu", dtype=torch.float64)
                - stateless.detach().to(device="cpu", dtype=torch.float64)
            )
        )
        error_value = float(error.item())
        if not torch.isfinite(error).item():
            raise AuthenticatedModelProviderError("cache-comparison error is non-finite")
        maximum_error = max(maximum_error, error_value)
        normalized_trace.append(list(prefix))

    tolerance = float(absolute_tolerance)
    report = IncrementalCacheComparison(
        policy_state_digest=provider.policy_state_digest,
        model_provenance_digest=provider.model_provenance_digest,
        tokenizer_binding_digest=tokenizer_manifest.digest,
        tokenizer_repository_id=tokenizer_manifest.repository_id,
        tokenizer_revision=tokenizer_manifest.revision,
        trace_digest=json_digest(normalized_trace, domain=_CACHE_TRACE_DOMAIN),
        prefix_lengths=tuple(len(prefix) for prefix in exact_trace),
        absolute_tolerance_hex=tolerance.hex(),
        maximum_absolute_error_hex=maximum_error.hex(),
        reset_count=provider.reset_count - resets_before,
        reuse_count=provider.cache_reuse_count - reuses_before,
        within_tolerance=maximum_error <= tolerance,
    )
    if not report.within_tolerance:
        raise IncrementalCacheMismatchError(report)
    return report


def provider_manifest(provider: AuthenticatedCausalLMProvider) -> dict[str, object]:
    """Return canonical, nonauthorizing evidence for repository reports."""

    if type(provider) is not AuthenticatedCausalLMProvider:
        raise TypeError("provider must be an AuthenticatedCausalLMProvider")
    provenance = provider.reauthenticate_policy_state()
    return {
        "schema_version": AUTHENTICATED_MODEL_PROVIDER_SCHEMA_VERSION,
        "contract_id": AUTHENTICATED_MODEL_PROVIDER_CONTRACT_ID,
        "authorizes_execution": AUTHENTICATED_MODEL_PROVIDER_AUTHORIZES_EXECUTION,
        "model_provenance": {**provenance.as_obj(), "digest": provenance.digest},
        "artifact_manifest": {
            **provider.artifact_manifest.as_obj(),
            "digest": provider.artifact_manifest.digest,
        },
        "runtime_manifest": {
            **provider.runtime_manifest.as_obj(),
            "digest": provider.runtime_manifest.digest,
        },
        "model_config_sha256": provider.model_config_sha256,
        "tensor_state_manifest": {
            **provider.tensor_state_manifest.as_obj(),
            "digest": provider.tensor_state_manifest.digest,
        },
        "semantic_attribute_manifest": {
            **provider.semantic_attribute_manifest.as_obj(),
            "digest": provider.semantic_attribute_manifest.digest,
        },
        "executable_manifest": {
            **provider.executable_manifest.as_obj(),
            "digest": provider.executable_manifest.digest,
        },
        "trainable_parameter_registry": {
            **provider.trainable_parameter_registry.manifest.as_obj(),
            "digest": provider.trainable_parameter_registry.manifest.digest,
        },
    }
