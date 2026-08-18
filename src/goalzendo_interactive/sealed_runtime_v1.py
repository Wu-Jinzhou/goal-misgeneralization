"""Sealed, local-only construction boundary for the pinned G03 Qwen runtime.

This module owns model/tokenizer loading.  It accepts no injected loader,
model, tokenizer, provider, or compiler and never permits a network fallback.
The returned nominal handle remains nonauthorizing until a separate launch
gate accepts its evidence.
"""

from __future__ import annotations

import hashlib
import hmac
import importlib
import json
import os
import stat
import threading
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, TypeGuard, cast

import torch

from ._json import dump_json, json_digest
from .action_tokenization_v2 import (
    ExactDecodeTokenizerProtocol,
    FragmentActionTokenCompiler,
    FragmentCompilerManifest,
    TokenizerBindingManifest,
)
from .authenticated_model_provider_v2 import (
    AuthenticatedCausalLMProvider,
    AuthenticatedModelProviderError,
    ModelArtifactManifest,
    TrainableParameterRegistry,
    _direct_named_modules,
    _direct_tensor_registrations,
    build_model_artifact_manifest,
    is_exact_authenticated_causal_lm_provider,
    is_exact_trainable_parameter_registry,
)
from .authenticated_rollouts_v2 import ModelPolicyProvenance

SEALED_QWEN_RUNTIME_SCHEMA_VERSION = 2
SEALED_QWEN_RUNTIME_CONTRACT_ID = "goalzendo-sealed-qwen-runtime-v2"
SEALED_QWEN_RUNTIME_AUTHORIZES_EXECUTION = False

G03_QWEN_MODEL_IDENTIFIER = "Qwen/Qwen2.5-1.5B-Instruct"
G03_QWEN_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
G03_QWEN_MODEL_TYPE = "qwen2"
G03_QWEN_ARCHITECTURE = "Qwen2ForCausalLM"
G03_QWEN_MODEL_CLASS = (
    "transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM"
)
G03_QWEN_TOKENIZER_CLASS = (
    "transformers.models.qwen2.tokenization_qwen2_fast.Qwen2TokenizerFast"
)
G03_QWEN_BACKEND_CLASS = "tokenizers.Tokenizer"
G03_QWEN_TOKENIZER_VOCABULARY_SIZE = 151_665
G03_QWEN_MODEL_VOCABULARY_SIZE = 151_936

_LOAD_CONFIG_DOMAIN = "goalzendo-interactive-sealed-qwen-load-config-v2"
_RUNTIME_MANIFEST_DOMAIN = "goalzendo-interactive-sealed-qwen-runtime-manifest-v2"
_TOKENIZER_RUNTIME_GUARD_DOMAIN = (
    "goalzendo-interactive-sealed-tokenizer-runtime-guard-v1"
)
_HANDLE_SEAL = object()
_UPDATE_SESSION_SEAL = object()
_PREPARED_UPDATE_SEAL = object()


class SealedQwenRuntimeError(ValueError):
    """Raised when exact local construction or subsequent reauthentication fails."""


def _resolve_materialized_root(value: str | os.PathLike[str]) -> Path:
    """Resolve once while rejecting every symlink in the supplied path chain."""

    try:
        absolute = Path(os.path.abspath(os.fspath(value)))
    except (TypeError, ValueError, OSError) as exc:
        raise SealedQwenRuntimeError("artifact root path is invalid") from exc
    current = Path(absolute.anchor)
    try:
        for part in absolute.parts[1:]:
            current = current / part
            metadata = current.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise SealedQwenRuntimeError(
                    "artifact root and every ancestor must be symlink-free"
                )
        resolved = absolute.resolve(strict=True)
    except SealedQwenRuntimeError:
        raise
    except OSError as exc:
        raise SealedQwenRuntimeError("artifact root path is not materialized") from exc
    if resolved != absolute:
        raise SealedQwenRuntimeError("artifact root path did not resolve identically")
    return resolved


def _is_sha256(value: object) -> TypeGuard[str]:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: object, *, name: str) -> str:
    if not _is_sha256(value):
        raise SealedQwenRuntimeError(f"{name} must be a lowercase SHA-256")
    return value


def _require_plain_version(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or not value
        or not value.isascii()
        or any(character.isspace() for character in value)
    ):
        raise SealedQwenRuntimeError(f"{name} must be nonempty whitespace-free ASCII")
    return value


@dataclass(frozen=True, slots=True)
class SealedQwenLoadConfig:
    """Exact, canonical Hugging Face load choices accepted by the boundary."""

    transformers_version: str
    tokenizers_version: str
    attention_implementation: str
    low_cpu_mem_usage: bool
    use_safetensors: bool
    maximum_action_tokens: int

    def __post_init__(self) -> None:
        _require_plain_version(self.transformers_version, name="transformers_version")
        _require_plain_version(self.tokenizers_version, name="tokenizers_version")
        if self.attention_implementation != "eager":
            raise SealedQwenRuntimeError(
                "the sealed runtime requires the deterministic eager attention implementation"
            )
        if type(self.low_cpu_mem_usage) is not bool:
            raise SealedQwenRuntimeError("low_cpu_mem_usage must be Boolean")
        if self.use_safetensors is not True:
            raise SealedQwenRuntimeError("the sealed runtime requires safetensors weights")
        if (
            isinstance(self.maximum_action_tokens, bool)
            or not isinstance(self.maximum_action_tokens, int)
            or self.maximum_action_tokens < 1
        ):
            raise SealedQwenRuntimeError("maximum_action_tokens must be positive")

    def as_obj(self) -> dict[str, object]:
        return {
            "schema_version": SEALED_QWEN_RUNTIME_SCHEMA_VERSION,
            "contract_id": SEALED_QWEN_RUNTIME_CONTRACT_ID,
            "transformers_version": self.transformers_version,
            "tokenizers_version": self.tokenizers_version,
            "attention_implementation": self.attention_implementation,
            "low_cpu_mem_usage": self.low_cpu_mem_usage,
            "use_safetensors": self.use_safetensors,
            "maximum_action_tokens": self.maximum_action_tokens,
            "local_files_only": True,
            "trust_remote_code": False,
            "use_fast_tokenizer": True,
        }

    @property
    def digest(self) -> str:
        return json_digest(self.as_obj(), domain=_LOAD_CONFIG_DOMAIN)


@dataclass(frozen=True, slots=True)
class SealedQwenRuntimeManifest:
    model_identifier: str
    revision: str
    artifact_manifest_digest: str
    tokenizer_binding_digest: str
    compiler_manifest_digest: str
    provider_policy_state_digest: str
    model_provenance_digest: str
    executable_manifest_digest: str
    trainable_parameter_registry_digest: str
    tokenizer_runtime_state_digest: str
    tokenizer_vocabulary_size: int
    model_vocabulary_size: int
    dtype: str
    device: str
    load_config_digest: str
    schema_version: int = SEALED_QWEN_RUNTIME_SCHEMA_VERSION
    contract_id: str = SEALED_QWEN_RUNTIME_CONTRACT_ID

    def as_obj(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "authorizes_execution": SEALED_QWEN_RUNTIME_AUTHORIZES_EXECUTION,
            "model_identifier": self.model_identifier,
            "revision": self.revision,
            "artifact_manifest_digest": self.artifact_manifest_digest,
            "tokenizer_binding_digest": self.tokenizer_binding_digest,
            "compiler_manifest_digest": self.compiler_manifest_digest,
            "provider_policy_state_digest": self.provider_policy_state_digest,
            "model_provenance_digest": self.model_provenance_digest,
            "executable_manifest_digest": self.executable_manifest_digest,
            "trainable_parameter_registry_digest": (
                self.trainable_parameter_registry_digest
            ),
            "tokenizer_runtime_state_digest": self.tokenizer_runtime_state_digest,
            "tokenizer_vocabulary_size": self.tokenizer_vocabulary_size,
            "model_vocabulary_size": self.model_vocabulary_size,
            "dtype": self.dtype,
            "device": self.device,
            "load_config_digest": self.load_config_digest,
        }

    @property
    def digest(self) -> str:
        return json_digest(self.as_obj(), domain=_RUNTIME_MANIFEST_DOMAIN)

    def to_json(self) -> str:
        return dump_json({**self.as_obj(), "digest": self.digest})


def _read_stable_regular_file(path: Path) -> bytes:
    try:
        before_path = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(before_path.st_mode):
            raise SealedQwenRuntimeError(f"required artifact {path.name!r} is not a regular file")
        with path.open("rb") as handle:
            before_fd = os.fstat(handle.fileno())
            data = handle.read()
            after_fd = os.fstat(handle.fileno())
        after_path = path.stat(follow_symlinks=False)
    except SealedQwenRuntimeError:
        raise
    except OSError as exc:
        raise SealedQwenRuntimeError(f"required artifact {path.name!r} could not be read") from exc
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(
        getattr(before_path, field) != getattr(before_fd, field)
        or getattr(before_fd, field) != getattr(after_fd, field)
        or getattr(after_fd, field) != getattr(after_path, field)
        for field in fields
    ):
        raise SealedQwenRuntimeError(f"required artifact {path.name!r} changed while read")
    return data


def _strict_json_object(data: bytes, *, name: str) -> dict[str, object]:
    def pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise SealedQwenRuntimeError(f"{name} contains a duplicate JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=pairs_hook)
    except SealedQwenRuntimeError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SealedQwenRuntimeError(f"{name} is not valid UTF-8 JSON") from exc
    if type(value) is not dict:
        raise SealedQwenRuntimeError(f"{name} must contain a JSON object")
    return cast(dict[str, object], value)


def _validate_qwen_artifact_layout(
    root: Path,
    artifact_manifest: ModelArtifactManifest,
) -> tuple[bytes, bytes, str]:
    paths = {record.relative_path for record in artifact_manifest.files}
    for relative_path in paths:
        try:
            metadata = (root / relative_path).stat(follow_symlinks=False)
        except OSError as exc:
            raise SealedQwenRuntimeError("artifact leaf could not be inspected") from exc
        if metadata.st_nlink != 1:
            raise SealedQwenRuntimeError("artifact leaves may not be hard-linked")
    required = {"config.json", "tokenizer.json", "tokenizer_config.json"}
    missing = sorted(required.difference(paths))
    if missing:
        raise SealedQwenRuntimeError(
            "local artifact root is incomplete; missing " + ", ".join(missing)
        )
    safetensors = sorted(path for path in paths if path.endswith(".safetensors"))
    if not safetensors:
        raise SealedQwenRuntimeError("local artifact root contains no safetensors weights")

    config_bytes = _read_stable_regular_file(root / "config.json")
    config = _strict_json_object(config_bytes, name="config.json")
    if config.get("model_type") != G03_QWEN_MODEL_TYPE:
        raise SealedQwenRuntimeError("config.json is not the exact qwen2 model family")
    architectures = config.get("architectures")
    if type(architectures) is not list or architectures != [G03_QWEN_ARCHITECTURE]:
        raise SealedQwenRuntimeError("config.json does not declare exact Qwen2ForCausalLM")

    index_path = root / "model.safetensors.index.json"
    if "model.safetensors.index.json" in paths:
        if "model.safetensors" in paths:
            raise SealedQwenRuntimeError(
                "artifact tree may not mix monolithic and indexed safetensors layouts"
            )
        index = _strict_json_object(
            _read_stable_regular_file(index_path),
            name="model.safetensors.index.json",
        )
        weight_map = index.get("weight_map")
        if type(weight_map) is not dict or not weight_map:
            raise SealedQwenRuntimeError("safetensors index has no nonempty weight_map")
        referenced = set()
        for shard in weight_map.values():
            if type(shard) is not str or not shard.endswith(".safetensors"):
                raise SealedQwenRuntimeError("safetensors index contains an invalid shard name")
            shard_path = Path(shard)
            if shard_path.is_absolute() or ".." in shard_path.parts or "\\" in shard:
                raise SealedQwenRuntimeError("safetensors index contains a nonlocal shard path")
            referenced.add(shard_path.as_posix())
        if not referenced.issubset(paths):
            raise SealedQwenRuntimeError("safetensors index references an absent local shard")
        if referenced != set(safetensors):
            raise SealedQwenRuntimeError(
                "safetensors index shards must exactly match the accepted weight files"
            )
    elif safetensors != ["model.safetensors"]:
        raise SealedQwenRuntimeError(
            "unindexed artifacts require exactly one model.safetensors file"
        )

    tokenizer_bytes = _read_stable_regular_file(root / "tokenizer.json")
    tokenizer_config_bytes = _read_stable_regular_file(root / "tokenizer_config.json")
    tokenizer_config = _strict_json_object(
        tokenizer_config_bytes,
        name="tokenizer_config.json",
    )
    chat_template = tokenizer_config.get("chat_template")
    if type(chat_template) is not str or not chat_template:
        raise SealedQwenRuntimeError("tokenizer_config.json has no nonempty chat_template")
    return tokenizer_bytes, tokenizer_config_bytes, chat_template


def _module_version(module: ModuleType, *, name: str) -> str:
    value = getattr(module, "__version__", None)
    return _require_plain_version(value, name=f"{name}.__version__")


def _tokenizer_special_ids(tokenizer: object, *, vocabulary_size: int) -> tuple[tuple[str, int | None], ...]:
    tokenizer_any = cast(Any, tokenizer)
    entries: list[tuple[str, int | None]] = []
    for role in ("bos", "eos", "pad", "unk"):
        value = getattr(tokenizer_any, f"{role}_token_id", None)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value < vocabulary_size
        ):
            raise SealedQwenRuntimeError(f"live tokenizer {role}_token_id is invalid")
        entries.append((role, cast(int | None, value)))
    additional = getattr(tokenizer_any, "additional_special_tokens", None)
    if type(additional) is not list or any(type(token) is not str or not token for token in additional):
        raise SealedQwenRuntimeError("live tokenizer additional_special_tokens is not exact text")
    if len(set(additional)) != len(additional):
        raise SealedQwenRuntimeError("live tokenizer additional special tokens contain duplicates")
    converter = getattr(tokenizer_any, "convert_tokens_to_ids", None)
    if not callable(converter):
        raise SealedQwenRuntimeError("live tokenizer cannot resolve special-token IDs")
    for token in additional:
        token_id = converter(token)
        if (
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or not 0 <= token_id < vocabulary_size
        ):
            raise SealedQwenRuntimeError("live tokenizer additional special-token ID is invalid")
        entries.append((f"additional:{token}", token_id))
    return tuple(sorted(entries))


def _derive_tokenizer_manifest(
    *,
    tokenizer: object,
    tokenizer_module: ModuleType,
    tokenizer_bytes: bytes,
    tokenizer_config_bytes: bytes,
    artifact_chat_template: str,
    model_identifier: str,
    revision: str,
) -> TokenizerBindingManifest:
    tokenizer_type = type(tokenizer)
    tokenizer_class = f"{tokenizer_type.__module__}.{tokenizer_type.__qualname__}"
    if tokenizer_class != G03_QWEN_TOKENIZER_CLASS:
        raise SealedQwenRuntimeError("AutoTokenizer did not return exact Qwen2TokenizerFast")
    tokenizer_any = cast(Any, tokenizer)
    if getattr(tokenizer_any, "is_fast", None) is not True:
        raise SealedQwenRuntimeError("sealed Qwen tokenizer must use the fast backend")
    live_template = getattr(tokenizer_any, "chat_template", None)
    if type(live_template) is not str or not hmac.compare_digest(
        live_template.encode("utf-8"), artifact_chat_template.encode("utf-8")
    ):
        raise SealedQwenRuntimeError("live chat template differs from tokenizer_config.json")
    backend = getattr(tokenizer_any, "backend_tokenizer", None)
    backend_name = f"{type(backend).__module__}.{type(backend).__qualname__}"
    if backend_name != G03_QWEN_BACKEND_CLASS:
        raise SealedQwenRuntimeError("live tokenizer does not use exact tokenizers.Tokenizer")
    expected_backend_class = getattr(tokenizer_module, "Tokenizer", None)
    if not isinstance(expected_backend_class, type) or type(backend) is not expected_backend_class:
        raise SealedQwenRuntimeError(
            "live tokenizer backend is not the exact imported tokenizers.Tokenizer class"
        )
    try:
        vocabulary_size = len(tokenizer_any)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SealedQwenRuntimeError("live tokenizer vocabulary size is unavailable") from exc
    if (
        isinstance(vocabulary_size, bool)
        or not isinstance(vocabulary_size, int)
        or vocabulary_size < 2
    ):
        raise SealedQwenRuntimeError("live tokenizer vocabulary size is invalid")
    if vocabulary_size != G03_QWEN_TOKENIZER_VOCABULARY_SIZE:
        raise SealedQwenRuntimeError("live tokenizer vocabulary is not the pinned Qwen size")
    return TokenizerBindingManifest(
        repository_id=model_identifier,
        revision=revision,
        tokenizer_json_sha256=hashlib.sha256(tokenizer_bytes).hexdigest(),
        tokenizer_config_sha256=hashlib.sha256(tokenizer_config_bytes).hexdigest(),
        chat_template_sha256=hashlib.sha256(live_template.encode("utf-8")).hexdigest(),
        backend_name=backend_name,
        backend_version=_module_version(tokenizer_module, name="tokenizers"),
        vocabulary_size=vocabulary_size,
        special_token_ids=_tokenizer_special_ids(
            tokenizer,
            vocabulary_size=vocabulary_size,
        ),
    )


def _require_deterministic_runtime(device: torch.device) -> None:
    if not torch.are_deterministic_algorithms_enabled():
        raise SealedQwenRuntimeError("deterministic PyTorch algorithms must be enabled")
    if torch.is_deterministic_algorithms_warn_only_enabled():
        raise SealedQwenRuntimeError("deterministic algorithms may not use warn-only mode")
    if torch.backends.cudnn.benchmark:
        raise SealedQwenRuntimeError("cuDNN benchmarking must be disabled")
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise SealedQwenRuntimeError("TF32 must be disabled")
    if (
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        or torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        or torch.backends.cuda.matmul.allow_fp16_accumulation
    ):
        raise SealedQwenRuntimeError(
            "reduced-precision reductions and FP16 accumulation must be disabled"
        )
    if not torch.is_grad_enabled() or torch.is_inference_mode_enabled():
        raise SealedQwenRuntimeError("sealed trainable construction requires ordinary grad mode")
    if torch.is_autocast_enabled(device.type):
        raise SealedQwenRuntimeError("sealed construction may not run inside autocast")
    if device.type == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {
        ":16:8",
        ":4096:8",
    }:
        raise SealedQwenRuntimeError(
            "CUDA construction requires a deterministic CUBLAS_WORKSPACE_CONFIG"
        )


def _require_dtype(value: object) -> torch.dtype:
    if value not in {torch.float32, torch.float16, torch.bfloat16}:
        raise SealedQwenRuntimeError("dtype must be float32, float16, or bfloat16")
    assert isinstance(value, torch.dtype)
    return value


def _require_device(value: object) -> torch.device:
    if type(value) not in {str, torch.device}:
        raise SealedQwenRuntimeError("device must be exact str or torch.device")
    try:
        device = torch.device(cast(Any, value))
    except (RuntimeError, TypeError, ValueError) as exc:
        raise SealedQwenRuntimeError("device is invalid") from exc
    if device.type not in {"cpu", "cuda"}:
        raise SealedQwenRuntimeError("sealed Qwen device must be CPU or CUDA")
    if device.type == "cuda" and device.index is None:
        raise SealedQwenRuntimeError("CUDA device must have an explicit index")
    return device


def _model_facts(
    model: torch.nn.Module,
    *,
    expected_dtype: torch.dtype,
    expected_device: torch.device,
) -> None:
    model_class = f"{type(model).__module__}.{type(model).__qualname__}"
    if model_class != G03_QWEN_MODEL_CLASS:
        raise SealedQwenRuntimeError("AutoModel did not return exact Qwen2ForCausalLM")
    config = getattr(cast(Any, model), "config", None)
    if getattr(config, "model_type", None) != G03_QWEN_MODEL_TYPE:
        raise SealedQwenRuntimeError("live model config is not qwen2")
    architectures = getattr(config, "architectures", None)
    if architectures != [G03_QWEN_ARCHITECTURE]:
        raise SealedQwenRuntimeError("live model config architecture is not exact Qwen2ForCausalLM")
    if getattr(config, "vocab_size", None) != G03_QWEN_MODEL_VOCABULARY_SIZE:
        raise SealedQwenRuntimeError("live model vocabulary is not the pinned Qwen width")
    try:
        module_records = _direct_named_modules(model)
        tensor_registrations = _direct_tensor_registrations(model)
    except AuthenticatedModelProviderError as exc:
        raise SealedQwenRuntimeError("model direct registries are invalid") from exc
    modules = tuple(record.module for record in module_records)
    if not modules or any(module.training for module in modules):
        raise SealedQwenRuntimeError("model and every submodule must be in eval mode")
    parameters = tuple(
        cast(torch.nn.Parameter, registration.tensor)
        for registration in tensor_registrations
        if registration.kind == "parameter"
    )
    if not parameters or any(not parameter.requires_grad for parameter in parameters):
        raise SealedQwenRuntimeError("every model parameter must remain trainable")
    tensors = tuple(registration.tensor for registration in tensor_registrations)
    if any(tensor.device != expected_device for tensor in tensors):
        raise SealedQwenRuntimeError("model tensors do not occupy the exact requested device")
    floating = tuple(tensor for tensor in tensors if tensor.is_floating_point())
    if not floating or any(tensor.dtype != expected_dtype for tensor in floating):
        raise SealedQwenRuntimeError("floating model state does not use the exact requested dtype")


def _tokenizer_default_value(value: object) -> object:
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        return {"kind": "float", "hex": value.hex()}
    if type(value) is tuple:
        return {
            "kind": "tuple",
            "items": [_tokenizer_default_value(item) for item in cast(tuple[object, ...], value)],
        }
    raise SealedQwenRuntimeError(
        "tokenizer method has an unsupported mutable/process-specific default"
    )


@dataclass(frozen=True, slots=True)
class _TokenizerRuntimeGuard:
    tokenizer_identity: int
    tokenizer_class_identity: int
    methods: tuple[tuple[str, int, int, str, str, str, str], ...]
    backend_identity: int
    backend_class_identity: int
    backend_class: str
    backend_serialization_sha256: str

    @property
    def semantic_digest(self) -> str:
        return json_digest(
            {
                "methods": [
                    {
                        "name": name,
                        "module": module,
                        "qualname": qualname,
                        "bytecode_sha256": bytecode_sha256,
                        "defaults_sha256": defaults_sha256,
                    }
                    for (
                        name,
                        _identity,
                        _code_identity,
                        module,
                        qualname,
                        bytecode_sha256,
                        defaults_sha256,
                    ) in self.methods
                ],
                "backend_class": self.backend_class,
                "backend_serialization_sha256": self.backend_serialization_sha256,
            },
            domain=_TOKENIZER_RUNTIME_GUARD_DOMAIN,
        )


def _method_identity_guard(tokenizer: object) -> _TokenizerRuntimeGuard:
    selected_names = (
        "__len__",
        "encode",
        "decode",
        "apply_chat_template",
        "convert_tokens_to_ids",
    )
    if any(name in vars(tokenizer) for name in selected_names):
        raise SealedQwenRuntimeError("tokenizer instance method monkeypatches are forbidden")
    selected_class = type(tokenizer)
    methods: list[tuple[str, int, int, str, str, str, str]] = []
    for name in selected_names:
        method = getattr(selected_class, name, None)
        if not callable(method):
            raise SealedQwenRuntimeError(f"tokenizer class has no callable {name}")
        dynamic = cast(Any, method)
        code = getattr(dynamic, "__code__", None)
        if code is None or not isinstance(code.co_code, bytes):
            raise SealedQwenRuntimeError(
                f"tokenizer class method {name} has no inspectable Python code"
            )
        module = getattr(dynamic, "__module__", None)
        qualname = getattr(dynamic, "__qualname__", None)
        if type(module) is not str or type(qualname) is not str:
            raise SealedQwenRuntimeError("tokenizer method identity labels are invalid")
        defaults = tuple(dynamic.__defaults__ or ())
        kwdefaults = dynamic.__kwdefaults__ or {}
        if type(kwdefaults) is not dict or any(type(key) is not str for key in kwdefaults):
            raise SealedQwenRuntimeError("tokenizer keyword defaults are not canonical")
        defaults_digest = json_digest(
            {
                "defaults": [_tokenizer_default_value(value) for value in defaults],
                "kwdefaults": {
                    key: _tokenizer_default_value(kwdefaults[key])
                    for key in sorted(kwdefaults)
                },
            },
            domain="goalzendo-interactive-tokenizer-method-defaults-v1",
        )
        methods.append(
            (
                name,
                id(method),
                id(code),
                module,
                qualname,
                hashlib.sha256(code.co_code).hexdigest(),
                defaults_digest,
            )
        )
    backend = getattr(cast(Any, tokenizer), "backend_tokenizer", None)
    serializer = getattr(backend, "to_str", None)
    if not callable(serializer):
        raise SealedQwenRuntimeError("tokenizer backend has no exact serialized-state surface")
    try:
        serialized = serializer()
    except Exception as exc:
        raise SealedQwenRuntimeError("tokenizer backend state could not be serialized") from exc
    if type(serialized) is not str:
        raise SealedQwenRuntimeError("tokenizer backend serialization is not text")
    backend_class = f"{type(backend).__module__}.{type(backend).__qualname__}"
    return _TokenizerRuntimeGuard(
        tokenizer_identity=id(tokenizer),
        tokenizer_class_identity=id(selected_class),
        methods=tuple(methods),
        backend_identity=id(backend),
        backend_class_identity=id(type(backend)),
        backend_class=backend_class,
        backend_serialization_sha256=hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
    )


class SealedQwenUpdateSession:
    """Private-issuance lease for one future authenticated atomic update.

    The class does not authorize a weight update.  Its underscore-prefixed
    runtime entry points exist only so the separately gated atomic-step
    primitive can hold the runtime lock while using the provider-issued exact
    parameter registry and issuing a fresh post-update provider.
    """

    __slots__ = (
        "_active",
        "_owner_thread",
        "_pre_manifest",
        "_registry",
        "_runtime",
    )

    def __init__(
        self,
        runtime: SealedQwenRuntime,
        *,
        registry: TrainableParameterRegistry,
        pre_manifest: SealedQwenRuntimeManifest,
        _seal: object,
    ) -> None:
        if _seal is not _UPDATE_SESSION_SEAL:
            raise TypeError("SealedQwenUpdateSession is issued only by its sealed runtime")
        self._runtime = runtime
        self._registry = registry
        self._pre_manifest = pre_manifest
        self._owner_thread = threading.get_ident()
        self._active = True

    @property
    def pre_manifest(self) -> SealedQwenRuntimeManifest:
        return self._pre_manifest

    @property
    def trainable_parameter_registry(self) -> TrainableParameterRegistry:
        self._runtime._require_update_session(self)
        return self._registry

    @property
    def _provider_for_step(self) -> AuthenticatedCausalLMProvider:
        self._runtime._require_update_session(self)
        return self._runtime._provider

    @property
    def _tokenizer_for_step(self) -> ExactDecodeTokenizerProtocol:
        self._runtime._require_update_session(self)
        return cast(ExactDecodeTokenizerProtocol, self._runtime._tokenizer)

    @property
    def _compiler_for_step(self) -> FragmentActionTokenCompiler:
        self._runtime._require_update_session(self)
        return self._runtime._compiler


class SealedQwenPreparedUpdate:
    """Unpublished post-weight provider candidate held under an update lease."""

    __slots__ = ("_manifest", "_provider", "_runtime", "_session")

    def __init__(
        self,
        runtime: SealedQwenRuntime,
        session: SealedQwenUpdateSession,
        provider: AuthenticatedCausalLMProvider,
        manifest: SealedQwenRuntimeManifest,
        *,
        _seal: object,
    ) -> None:
        if _seal is not _PREPARED_UPDATE_SEAL:
            raise TypeError("SealedQwenPreparedUpdate is issued only by its sealed runtime")
        self._runtime = runtime
        self._session = session
        self._provider = provider
        self._manifest = manifest

    @property
    def manifest(self) -> SealedQwenRuntimeManifest:
        self._runtime._require_update_session(self._session)
        return self._manifest


class SealedQwenRuntime:
    """Nominal exact-type handle issued only after all local bindings agree."""

    __slots__ = (
        "_artifact_root",
        "_compiler",
        "_dtype",
        "_load_config",
        "_manifest",
        "_model",
        "_provider",
        "_tokenizer",
        "_tokenizer_method_guard",
        "_tokenizer_module",
        "_transformers_module",
        "_update_active",
        "_update_corrupt",
        "_update_lock",
    )

    def __init__(
        self,
        *,
        artifact_root: Path,
        model: torch.nn.Module,
        tokenizer: object,
        tokenizer_module: ModuleType,
        transformers_module: ModuleType,
        compiler: FragmentActionTokenCompiler,
        provider: AuthenticatedCausalLMProvider,
        dtype: torch.dtype,
        load_config: SealedQwenLoadConfig,
        manifest: SealedQwenRuntimeManifest,
        tokenizer_runtime_guard: _TokenizerRuntimeGuard,
        _seal: object,
    ) -> None:
        if _seal is not _HANDLE_SEAL:
            raise TypeError("SealedQwenRuntime is issued only by load_sealed_qwen_runtime")
        self._artifact_root = artifact_root
        self._compiler = compiler
        self._dtype = dtype
        self._load_config = load_config
        self._manifest = manifest
        self._model = model
        self._provider = provider
        self._tokenizer = tokenizer
        if _method_identity_guard(tokenizer) != tokenizer_runtime_guard:
            raise SealedQwenRuntimeError("tokenizer runtime changed while handle was issued")
        self._tokenizer_method_guard = tokenizer_runtime_guard
        self._tokenizer_module = tokenizer_module
        self._transformers_module = transformers_module
        self._update_lock = threading.RLock()
        self._update_active = False
        self._update_corrupt = False

    @property
    def manifest(self) -> SealedQwenRuntimeManifest:
        with self._update_lock:
            self._require_public_ready()
            return self._manifest

    @property
    def provider(self) -> AuthenticatedCausalLMProvider:
        with self._update_lock:
            self._require_public_ready()
            return self._provider

    @property
    def tokenizer(self) -> ExactDecodeTokenizerProtocol:
        with self._update_lock:
            self._require_public_ready()
            return cast(ExactDecodeTokenizerProtocol, self._tokenizer)

    @property
    def compiler(self) -> FragmentActionTokenCompiler:
        with self._update_lock:
            self._require_public_ready()
            return self._compiler

    @property
    def trainable_parameter_registry(self) -> TrainableParameterRegistry:
        with self._update_lock:
            self._require_public_ready()
            return self._provider.trainable_parameter_registry

    def reauthenticate(self) -> SealedQwenRuntimeManifest:
        with self._update_lock:
            self._require_public_ready()
            return self._reauthenticate_unlocked()

    def _require_public_ready(self) -> None:
        if self._update_corrupt:
            raise SealedQwenRuntimeError("sealed runtime is marked corrupt")
        if self._update_active:
            raise SealedQwenRuntimeError("sealed runtime has an active authenticated update")

    def _reauthenticate_unlocked(self) -> SealedQwenRuntimeManifest:
        if not is_exact_authenticated_causal_lm_provider(self._provider):
            raise SealedQwenRuntimeError("sealed provider lost its exact nominal type")
        if type(self._compiler) is not FragmentActionTokenCompiler:
            raise SealedQwenRuntimeError("sealed compiler lost its exact nominal type")
        if _method_identity_guard(self._tokenizer) != self._tokenizer_method_guard:
            raise SealedQwenRuntimeError("tokenizer class/method identity changed")
        if (
            _module_version(self._transformers_module, name="transformers")
            != self._load_config.transformers_version
            or _module_version(self._tokenizer_module, name="tokenizers")
            != self._load_config.tokenizers_version
        ):
            raise SealedQwenRuntimeError("sealed transformers/tokenizers stack version changed")
        _require_deterministic_runtime(torch.device(self._manifest.device))
        _model_facts(
            self._model,
            expected_dtype=self._dtype,
            expected_device=torch.device(self._manifest.device),
        )
        artifact = build_model_artifact_manifest(
            self._artifact_root,
            model_identifier=self._manifest.model_identifier,
            revision=self._manifest.revision,
        )
        tokenizer_bytes, tokenizer_config_bytes, chat_template = _validate_qwen_artifact_layout(
            self._artifact_root,
            artifact,
        )
        tokenizer_manifest = _derive_tokenizer_manifest(
            tokenizer=self._tokenizer,
            tokenizer_module=self._tokenizer_module,
            tokenizer_bytes=tokenizer_bytes,
            tokenizer_config_bytes=tokenizer_config_bytes,
            artifact_chat_template=chat_template,
            model_identifier=self._manifest.model_identifier,
            revision=self._manifest.revision,
        )
        if not hmac.compare_digest(artifact.digest, self._manifest.artifact_manifest_digest):
            raise SealedQwenRuntimeError("sealed artifact manifest changed")
        if not hmac.compare_digest(
            tokenizer_manifest.digest,
            self._manifest.tokenizer_binding_digest,
        ):
            raise SealedQwenRuntimeError("sealed tokenizer binding changed")
        compiler_manifest = self._compiler.manifest
        if not hmac.compare_digest(
            compiler_manifest.digest,
            self._manifest.compiler_manifest_digest,
        ):
            raise SealedQwenRuntimeError("sealed compiler manifest changed")
        if compiler_manifest.tokenizer_manifest != tokenizer_manifest:
            raise SealedQwenRuntimeError("compiler/tokenizer live bindings diverged")
        try:
            provenance = self._provider.reauthenticate_policy_state()
        except AuthenticatedModelProviderError as exc:
            raise SealedQwenRuntimeError("sealed provider reauthentication failed") from exc
        registry = self._provider.trainable_parameter_registry
        if not is_exact_trainable_parameter_registry(registry):
            raise SealedQwenRuntimeError("provider registry lost its exact nominal type")
        regenerated = _runtime_manifest(
            artifact=artifact,
            tokenizer_manifest=tokenizer_manifest,
            compiler_manifest=compiler_manifest,
            provenance=provenance,
            provider=self._provider,
            registry=registry,
            dtype=self._dtype,
            device=torch.device(self._manifest.device),
            load_config=self._load_config,
            tokenizer_runtime_guard=self._tokenizer_method_guard,
        )
        if regenerated != self._manifest or not hmac.compare_digest(
            regenerated.digest,
            self._manifest.digest,
        ):
            raise SealedQwenRuntimeError("sealed runtime manifest changed")
        return self._manifest

    def _require_update_session(self, session: object) -> SealedQwenUpdateSession:
        if type(session) is not SealedQwenUpdateSession:
            raise TypeError("session must have exact type SealedQwenUpdateSession")
        selected = session
        if (
            selected._runtime is not self
            or not selected._active
            or not self._update_active
            or self._update_corrupt
            or selected._owner_thread != threading.get_ident()
        ):
            raise SealedQwenRuntimeError("authenticated update session is not active here")
        return selected

    def _begin_authenticated_update(self) -> SealedQwenUpdateSession:
        """Acquire the internal lease used only by the gated atomic-step primitive."""

        self._update_lock.acquire()
        try:
            self._require_public_ready()
            pre_manifest = self._reauthenticate_unlocked()
            registry = self._provider.trainable_parameter_registry
            if not is_exact_trainable_parameter_registry(registry):
                raise SealedQwenRuntimeError("provider registry lost its exact nominal type")
            self._update_active = True
            return SealedQwenUpdateSession(
                self,
                registry=registry,
                pre_manifest=pre_manifest,
                _seal=_UPDATE_SESSION_SEAL,
            )
        except BaseException:
            self._update_lock.release()
            raise

    def _finish_update_session(self, session: SealedQwenUpdateSession) -> None:
        session._active = False
        self._update_active = False
        self._update_lock.release()

    def _prepare_authenticated_update(
        self,
        session: SealedQwenUpdateSession,
    ) -> SealedQwenPreparedUpdate:
        """Build but do not publish fresh provider evidence under the lease."""

        selected = self._require_update_session(session)
        if _method_identity_guard(self._tokenizer) != self._tokenizer_method_guard:
            raise SealedQwenRuntimeError("tokenizer runtime changed during authenticated update")
        _require_deterministic_runtime(torch.device(self._manifest.device))
        _model_facts(
            self._model,
            expected_dtype=self._dtype,
            expected_device=torch.device(self._manifest.device),
        )
        artifact = build_model_artifact_manifest(
            self._artifact_root,
            model_identifier=self._manifest.model_identifier,
            revision=self._manifest.revision,
        )
        tokenizer_bytes, tokenizer_config_bytes, chat_template = (
            _validate_qwen_artifact_layout(self._artifact_root, artifact)
        )
        tokenizer_manifest = _derive_tokenizer_manifest(
            tokenizer=self._tokenizer,
            tokenizer_module=self._tokenizer_module,
            tokenizer_bytes=tokenizer_bytes,
            tokenizer_config_bytes=tokenizer_config_bytes,
            artifact_chat_template=chat_template,
            model_identifier=self._manifest.model_identifier,
            revision=self._manifest.revision,
        )
        compiler_manifest = self._compiler.manifest
        if (
            not hmac.compare_digest(
                artifact.digest,
                selected.pre_manifest.artifact_manifest_digest,
            )
            or not hmac.compare_digest(
                tokenizer_manifest.digest,
                selected.pre_manifest.tokenizer_binding_digest,
            )
            or not hmac.compare_digest(
                compiler_manifest.digest,
                selected.pre_manifest.compiler_manifest_digest,
            )
            or compiler_manifest.tokenizer_manifest != tokenizer_manifest
        ):
            raise SealedQwenRuntimeError(
                "immutable artifact/tokenizer/compiler evidence changed during update"
            )
        provider = AuthenticatedCausalLMProvider(
            self._model,
            self._artifact_root,
            model_identifier=self._manifest.model_identifier,
            revision=self._manifest.revision,
        )
        registry = provider.trainable_parameter_registry
        if (
            not registry.manifest.all_parameters_trainable
            or tokenizer_manifest.vocabulary_size
            != G03_QWEN_TOKENIZER_VOCABULARY_SIZE
            or provider.vocabulary_size != G03_QWEN_MODEL_VOCABULARY_SIZE
            or tokenizer_manifest.vocabulary_size > provider.vocabulary_size
        ):
            raise SealedQwenRuntimeError("post-update provider lost pinned full-model facts")
        provenance = provider.reauthenticate_policy_state()
        manifest = _runtime_manifest(
            artifact=artifact,
            tokenizer_manifest=tokenizer_manifest,
            compiler_manifest=compiler_manifest,
            provenance=provenance,
            provider=provider,
            registry=registry,
            dtype=self._dtype,
            device=torch.device(self._manifest.device),
            load_config=self._load_config,
            tokenizer_runtime_guard=self._tokenizer_method_guard,
        )
        static_fields = (
            "artifact_manifest_digest",
            "tokenizer_binding_digest",
            "compiler_manifest_digest",
            "executable_manifest_digest",
            "trainable_parameter_registry_digest",
            "tokenizer_runtime_state_digest",
            "tokenizer_vocabulary_size",
            "model_vocabulary_size",
            "dtype",
            "device",
            "load_config_digest",
        )
        if any(
            getattr(manifest, field) != getattr(selected.pre_manifest, field)
            for field in static_fields
        ):
            raise SealedQwenRuntimeError("post-update static runtime bindings changed")
        return SealedQwenPreparedUpdate(
            self,
            selected,
            provider,
            manifest,
            _seal=_PREPARED_UPDATE_SEAL,
        )

    def _publish_authenticated_update(
        self,
        session: SealedQwenUpdateSession,
        prepared: SealedQwenPreparedUpdate,
    ) -> SealedQwenRuntimeManifest:
        """Atomically publish a validated candidate and retire the held lease."""

        selected = self._require_update_session(session)
        if (
            type(prepared) is not SealedQwenPreparedUpdate
            or prepared._runtime is not self
            or prepared._session is not selected
        ):
            raise SealedQwenRuntimeError("prepared update was not issued for this lease")
        # Candidate policy bytes may not change between preparation and the
        # final publication point.
        if prepared._provider.reauthenticate_policy_state().policy_state_digest != (
            prepared._manifest.provider_policy_state_digest
        ):
            raise SealedQwenRuntimeError("prepared provider changed before publication")
        self._provider = prepared._provider
        self._manifest = prepared._manifest
        self._finish_update_session(selected)
        return self._manifest

    def _commit_authenticated_update(
        self,
        session: SealedQwenUpdateSession,
    ) -> SealedQwenRuntimeManifest:
        """Convenience prepare/publish path for update-session unit QA."""

        prepared = self._prepare_authenticated_update(session)
        return self._publish_authenticated_update(session, prepared)

    def _rollback_authenticated_update(
        self,
        session: SealedQwenUpdateSession,
    ) -> SealedQwenRuntimeManifest:
        """Verify caller-restored weights and retire the internal update lease."""

        selected = self._require_update_session(session)
        try:
            # Even byte-exact restoration increments PyTorch mutation-version
            # counters, so the old provider must be retired.  The same fresh
            # issuance path is used, then its complete manifest must equal the
            # pre-update manifest byte-for-byte.
            prepared = self._prepare_authenticated_update(selected)
        except BaseException as exc:
            if selected._active:
                self._update_corrupt = True
                self._finish_update_session(selected)
            raise SealedQwenRuntimeError(
                "authenticated update rollback could not restore the sealed runtime"
            ) from exc
        restored = prepared.manifest
        if restored != selected.pre_manifest or not hmac.compare_digest(
            restored.digest,
            selected.pre_manifest.digest,
        ):
            self._update_corrupt = True
            self._finish_update_session(selected)
            raise SealedQwenRuntimeError(
                "authenticated update rollback produced different policy evidence"
            )
        return self._publish_authenticated_update(selected, prepared)

    def _mark_authenticated_update_corrupt(
        self,
        session: SealedQwenUpdateSession,
    ) -> None:
        """Permanently fail closed when the atomic primitive cannot roll back."""

        selected = self._require_update_session(session)
        self._update_corrupt = True
        self._finish_update_session(selected)


def is_exact_sealed_qwen_runtime(value: object) -> TypeGuard[SealedQwenRuntime]:
    """Reject structural Protocol liars and every subclass at production boundaries."""

    return type(value) is SealedQwenRuntime


def require_exact_sealed_qwen_runtime(value: object) -> SealedQwenRuntime:
    if not is_exact_sealed_qwen_runtime(value):
        raise TypeError("runtime must have exact type SealedQwenRuntime")
    value.reauthenticate()
    return value


def _runtime_manifest(
    *,
    artifact: ModelArtifactManifest,
    tokenizer_manifest: TokenizerBindingManifest,
    compiler_manifest: FragmentCompilerManifest,
    provenance: ModelPolicyProvenance,
    provider: AuthenticatedCausalLMProvider,
    registry: TrainableParameterRegistry,
    dtype: torch.dtype,
    device: torch.device,
    load_config: SealedQwenLoadConfig,
    tokenizer_runtime_guard: _TokenizerRuntimeGuard,
) -> SealedQwenRuntimeManifest:
    return SealedQwenRuntimeManifest(
        model_identifier=artifact.model_identifier,
        revision=artifact.revision,
        artifact_manifest_digest=artifact.digest,
        tokenizer_binding_digest=tokenizer_manifest.digest,
        compiler_manifest_digest=compiler_manifest.digest,
        provider_policy_state_digest=provenance.policy_state_digest,
        model_provenance_digest=provenance.digest,
        executable_manifest_digest=provider.executable_manifest.digest,
        trainable_parameter_registry_digest=registry.manifest.digest,
        tokenizer_runtime_state_digest=tokenizer_runtime_guard.semantic_digest,
        tokenizer_vocabulary_size=tokenizer_manifest.vocabulary_size,
        model_vocabulary_size=provider.vocabulary_size,
        dtype=str(dtype),
        device=str(device),
        load_config_digest=load_config.digest,
    )


def load_sealed_qwen_runtime(
    artifact_root: str | os.PathLike[str],
    *,
    expected_model_identifier: str,
    expected_revision: str,
    expected_tokenizer_binding_digest: str,
    expected_compiler_manifest_digest: str,
    expected_artifact_manifest_digest: str,
    dtype: torch.dtype,
    device: str | torch.device,
    load_config: SealedQwenLoadConfig,
) -> SealedQwenRuntime:
    """Load the exact pinned Qwen artifacts locally and return a sealed handle."""

    if expected_model_identifier != G03_QWEN_MODEL_IDENTIFIER:
        raise SealedQwenRuntimeError("unexpected G03 Qwen model identifier")
    if expected_revision != G03_QWEN_REVISION:
        raise SealedQwenRuntimeError("unexpected G03 immutable Qwen revision")
    if type(load_config) is not SealedQwenLoadConfig:
        raise TypeError("load_config must have exact type SealedQwenLoadConfig")
    expected_tokenizer = _require_sha256(
        expected_tokenizer_binding_digest,
        name="expected_tokenizer_binding_digest",
    )
    expected_compiler = _require_sha256(
        expected_compiler_manifest_digest,
        name="expected_compiler_manifest_digest",
    )
    expected_artifact = _require_sha256(
        expected_artifact_manifest_digest,
        name="expected_artifact_manifest_digest",
    )
    selected_dtype = _require_dtype(dtype)
    selected_device = _require_device(device)
    _require_deterministic_runtime(selected_device)

    root = _resolve_materialized_root(artifact_root)
    artifact = build_model_artifact_manifest(
        root,
        model_identifier=expected_model_identifier,
        revision=expected_revision,
    )
    if not hmac.compare_digest(artifact.digest, expected_artifact):
        raise SealedQwenRuntimeError("local artifact manifest differs from the external digest")
    tokenizer_bytes, tokenizer_config_bytes, chat_template = _validate_qwen_artifact_layout(
        root,
        artifact,
    )

    try:
        transformers_module = importlib.import_module("transformers")
        tokenizers_module = importlib.import_module("tokenizers")
    except ImportError as exc:
        raise SealedQwenRuntimeError("pinned transformers/tokenizers stack is unavailable") from exc
    if not isinstance(transformers_module, ModuleType) or not isinstance(
        tokenizers_module,
        ModuleType,
    ):
        raise SealedQwenRuntimeError("transformers/tokenizers imports are not modules")
    if _module_version(transformers_module, name="transformers") != load_config.transformers_version:
        raise SealedQwenRuntimeError("transformers version differs from the exact load config")
    if _module_version(tokenizers_module, name="tokenizers") != load_config.tokenizers_version:
        raise SealedQwenRuntimeError("tokenizers version differs from the exact load config")
    auto_tokenizer = getattr(transformers_module, "AutoTokenizer", None)
    auto_model = getattr(transformers_module, "AutoModelForCausalLM", None)
    if auto_tokenizer is None or auto_model is None:
        raise SealedQwenRuntimeError("transformers Auto classes are unavailable")
    resolved_root = str(root.resolve(strict=True))
    try:
        tokenizer = auto_tokenizer.from_pretrained(
            resolved_root,
            revision=expected_revision,
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )
        model_result = auto_model.from_pretrained(
            resolved_root,
            revision=expected_revision,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=selected_dtype,
            low_cpu_mem_usage=load_config.low_cpu_mem_usage,
            use_safetensors=load_config.use_safetensors,
            attn_implementation=load_config.attention_implementation,
            output_loading_info=True,
        )
    except Exception as exc:
        raise SealedQwenRuntimeError("local-only pinned Qwen loading failed") from exc
    if type(model_result) is not tuple or len(model_result) != 2:
        raise SealedQwenRuntimeError("AutoModel did not return exact loading information")
    model_candidate, loading_info = model_result
    expected_loading_keys = {
        "missing_keys",
        "unexpected_keys",
        "mismatched_keys",
        "error_msgs",
    }
    if type(loading_info) is not dict or set(loading_info) != expected_loading_keys:
        raise SealedQwenRuntimeError("AutoModel loading information has an unexpected schema")
    if any(type(loading_info[key]) is not list for key in expected_loading_keys):
        raise SealedQwenRuntimeError("AutoModel loading information fields must be exact lists")
    if any(loading_info[key] for key in expected_loading_keys):
        raise SealedQwenRuntimeError(
            "AutoModel reported missing, unexpected, mismatched, or failed weights"
        )
    if not isinstance(model_candidate, torch.nn.Module):
        raise SealedQwenRuntimeError("AutoModel did not return a torch.nn.Module")
    model = model_candidate
    try:
        model.to(device=selected_device, dtype=selected_dtype)
        model.eval()
    except (RuntimeError, TypeError, ValueError) as exc:
        raise SealedQwenRuntimeError("loaded model could not enter exact dtype/device/eval state") from exc
    _model_facts(
        model,
        expected_dtype=selected_dtype,
        expected_device=selected_device,
    )
    tokenizer_manifest = _derive_tokenizer_manifest(
        tokenizer=tokenizer,
        tokenizer_module=tokenizers_module,
        tokenizer_bytes=tokenizer_bytes,
        tokenizer_config_bytes=tokenizer_config_bytes,
        artifact_chat_template=chat_template,
        model_identifier=expected_model_identifier,
        revision=expected_revision,
    )
    if not hmac.compare_digest(tokenizer_manifest.digest, expected_tokenizer):
        raise SealedQwenRuntimeError("derived tokenizer binding differs from the external digest")
    compiler = FragmentActionTokenCompiler(
        cast(Any, tokenizer),
        tokenizer_manifest=tokenizer_manifest,
        maximum_action_tokens=load_config.maximum_action_tokens,
    )
    compiler_manifest = compiler.freeze_registered_language()
    if not hmac.compare_digest(compiler_manifest.digest, expected_compiler):
        raise SealedQwenRuntimeError("frozen compiler binding differs from the external digest")
    provider = AuthenticatedCausalLMProvider(
        model,
        root,
        model_identifier=expected_model_identifier,
        revision=expected_revision,
    )
    if not hmac.compare_digest(provider.artifact_manifest.digest, expected_artifact):
        raise SealedQwenRuntimeError("provider artifact binding differs after model loading")
    if (
        tokenizer_manifest.vocabulary_size != G03_QWEN_TOKENIZER_VOCABULARY_SIZE
        or provider.vocabulary_size != G03_QWEN_MODEL_VOCABULARY_SIZE
        or tokenizer_manifest.vocabulary_size > provider.vocabulary_size
    ):
        raise SealedQwenRuntimeError(
            "model/tokenizer vocabularies are not the exact pinned padded pair"
        )
    registry = provider.trainable_parameter_registry
    if not registry.manifest.all_parameters_trainable:
        raise SealedQwenRuntimeError("provider registry is not a complete full-model update")
    provenance = provider.reauthenticate_policy_state()
    tokenizer_runtime_guard = _method_identity_guard(tokenizer)
    manifest = _runtime_manifest(
        artifact=artifact,
        tokenizer_manifest=tokenizer_manifest,
        compiler_manifest=compiler_manifest,
        provenance=provenance,
        provider=provider,
        registry=registry,
        dtype=selected_dtype,
        device=selected_device,
        load_config=load_config,
        tokenizer_runtime_guard=tokenizer_runtime_guard,
    )
    handle = SealedQwenRuntime(
        artifact_root=root,
        model=model,
        tokenizer=tokenizer,
        tokenizer_module=tokenizers_module,
        transformers_module=transformers_module,
        compiler=compiler,
        provider=provider,
        dtype=selected_dtype,
        load_config=load_config,
        manifest=manifest,
        tokenizer_runtime_guard=tokenizer_runtime_guard,
        _seal=_HANDLE_SEAL,
    )
    handle.reauthenticate()
    return handle
