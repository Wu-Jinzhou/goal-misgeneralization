from __future__ import annotations

import hashlib
import inspect
import json
import sys
import types
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest
import torch

import goalzendo_interactive.sealed_runtime_v1 as sealed_runtime_module
from goalzendo_interactive.action_tokenization_v2 import (
    FragmentActionTokenCompiler,
    TokenizerBindingManifest,
)
from goalzendo_interactive.authenticated_model_provider_v2 import (
    build_model_artifact_manifest,
)
from goalzendo_interactive.sealed_runtime_v1 import (
    G03_QWEN_MODEL_IDENTIFIER,
    G03_QWEN_REVISION,
    SEALED_QWEN_RUNTIME_AUTHORIZES_EXECUTION,
    SealedQwenLoadConfig,
    SealedQwenRuntime,
    SealedQwenRuntimeError,
    is_exact_sealed_qwen_runtime,
    load_sealed_qwen_runtime,
    require_exact_sealed_qwen_runtime,
)

_TRANSFORMERS_VERSION = "4.57.6-test"
_TOKENIZERS_VERSION = "0.23.1-test"
_CHAT_TEMPLATE = "{% for message in messages %}{{ message['role'] }}:{{ message['content'] }}\n{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}"
_FAKE_TOKENIZER_VOCABULARY_SIZE = 256
_FAKE_MODEL_VOCABULARY_SIZE = 260


class Tokenizer:
    def __init__(self, *, marker: str = "exact-backend") -> None:
        self.marker = marker

    def to_str(self) -> str:
        return json.dumps({"marker": self.marker}, sort_keys=True)


Tokenizer.__module__ = "tokenizers"


class Qwen2TokenizerFast:
    is_fast = True
    bos_token_id = None
    eos_token_id = 201
    pad_token_id = 0
    unk_token_id = 1
    def __init__(self, *, changed_eos: bool = False) -> None:
        self.backend_tokenizer = Tokenizer()
        self.chat_template = _CHAT_TEMPLATE
        self.additional_special_tokens = ["<|im_start|>", "<|im_end|>"]
        if changed_eos:
            self.eos_token_id = 202

    def __len__(self) -> int:
        return _FAKE_TOKENIZER_VOCABULARY_SIZE

    def apply_chat_template(
        self,
        conversation: Sequence[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        rendered = "".join(
            f"{message['role']}:{message['content']}\n" for message in conversation
        )
        return rendered + ("assistant:" if add_generation_prompt else "")

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return list(text.encode("utf-8"))

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return bytes(token_ids).decode("utf-8")

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"<|im_start|>": 200, "<|im_end|>": 201}[token]


Qwen2TokenizerFast.__module__ = "transformers.models.qwen2.tokenization_qwen2_fast"


class QwenConfig:
    model_type = "qwen2"
    vocab_size = _FAKE_MODEL_VOCABULARY_SIZE

    def __init__(self) -> None:
        self.architectures = ["Qwen2ForCausalLM"]

    def to_dict(self) -> dict[str, object]:
        return {
            "architectures": list(self.architectures),
            "hidden_size": 5,
            "model_type": self.model_type,
            "vocab_size": self.vocab_size,
        }


@dataclass
class FakeOutput:
    logits: torch.Tensor


class Qwen2ForCausalLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = QwenConfig()
        self.embedding = torch.nn.Embedding(_FAKE_MODEL_VOCABULARY_SIZE, 5)
        self.projection = torch.nn.Linear(
            5,
            _FAKE_MODEL_VOCABULARY_SIZE,
            bias=False,
        )
        # This deliberately unused trainable parameter must remain registered.
        self.unused = torch.nn.Parameter(torch.tensor([1.0, 2.0]))

    def forward(self, *, input_ids: torch.Tensor) -> FakeOutput:
        reached_unused = self.unused.sum() * 0
        return FakeOutput(self.projection(self.embedding(input_ids)) + reached_unused)


Qwen2ForCausalLM.__module__ = "transformers.models.qwen2.modeling_qwen2"
Qwen2ForCausalLM.forward.__module__ = "transformers.models.qwen2.modeling_qwen2"


@dataclass
class _FakeStack:
    transformers_module: types.ModuleType
    tokenizers_module: types.ModuleType
    auto_tokenizer: type[object]
    auto_model: type[object]


def _fake_stack(
    *,
    changed_eos: bool = False,
    missing_model_keys: bool = False,
) -> _FakeStack:
    class AutoTokenizer:
        calls: ClassVar[list[tuple[str, dict[str, object]]]] = []

        @classmethod
        def from_pretrained(cls, path: str, **kwargs: object) -> Qwen2TokenizerFast:
            cls.calls.append((path, kwargs))
            return Qwen2TokenizerFast(changed_eos=changed_eos)

    class AutoModelForCausalLM:
        calls: ClassVar[list[tuple[str, dict[str, object]]]] = []

        @classmethod
        def from_pretrained(
            cls,
            path: str,
            **kwargs: object,
        ) -> tuple[Qwen2ForCausalLM, dict[str, list[object]]]:
            cls.calls.append((path, kwargs))
            return (
                Qwen2ForCausalLM(),
                {
                    "missing_keys": ["lm_head.weight"] if missing_model_keys else [],
                    "unexpected_keys": [],
                    "mismatched_keys": [],
                    "error_msgs": [],
                },
            )

    transformers_module = types.ModuleType("transformers")
    transformers_any = cast(Any, transformers_module)
    transformers_any.__version__ = _TRANSFORMERS_VERSION
    transformers_any.AutoTokenizer = AutoTokenizer
    transformers_any.AutoModelForCausalLM = AutoModelForCausalLM
    tokenizers_module = types.ModuleType("tokenizers")
    tokenizers_any = cast(Any, tokenizers_module)
    tokenizers_any.__version__ = _TOKENIZERS_VERSION
    tokenizers_any.Tokenizer = Tokenizer
    return _FakeStack(
        transformers_module=transformers_module,
        tokenizers_module=tokenizers_module,
        auto_tokenizer=AutoTokenizer,
        auto_model=AutoModelForCausalLM,
    )


def _write_artifacts(tmp_path: Path) -> Path:
    root = tmp_path / "qwen-artifacts"
    root.mkdir(parents=True)
    (root / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen2ForCausalLM"],
                "model_type": "qwen2",
                "vocab_size": _FAKE_MODEL_VOCABULARY_SIZE,
            }
        ),
        encoding="utf-8",
    )
    (root / "tokenizer.json").write_bytes(b'{"fake":"exact-tokenizer-bytes"}')
    (root / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": _CHAT_TEMPLATE}),
        encoding="utf-8",
    )
    (root / "model.safetensors").write_bytes(b"fully-materialized-fake-safetensors")
    return root


def _tokenizer_manifest(root: Path) -> TokenizerBindingManifest:
    return TokenizerBindingManifest(
        repository_id=G03_QWEN_MODEL_IDENTIFIER,
        revision=G03_QWEN_REVISION,
        tokenizer_json_sha256=hashlib.sha256((root / "tokenizer.json").read_bytes()).hexdigest(),
        tokenizer_config_sha256=hashlib.sha256(
            (root / "tokenizer_config.json").read_bytes()
        ).hexdigest(),
        chat_template_sha256=hashlib.sha256(_CHAT_TEMPLATE.encode("utf-8")).hexdigest(),
        backend_name="tokenizers.Tokenizer",
        backend_version=_TOKENIZERS_VERSION,
        vocabulary_size=_FAKE_TOKENIZER_VOCABULARY_SIZE,
        special_token_ids=(
            ("additional:<|im_end|>", 201),
            ("additional:<|im_start|>", 200),
            ("bos", None),
            ("eos", 201),
            ("pad", 0),
            ("unk", 1),
        ),
    )


def _load_config() -> SealedQwenLoadConfig:
    return SealedQwenLoadConfig(
        transformers_version=_TRANSFORMERS_VERSION,
        tokenizers_version=_TOKENIZERS_VERSION,
        attention_implementation="eager",
        low_cpu_mem_usage=True,
        use_safetensors=True,
        maximum_action_tokens=2_048,
    )


def _compiler_digest(manifest: TokenizerBindingManifest) -> str:
    compiler = FragmentActionTokenCompiler(
        Qwen2TokenizerFast(),
        tokenizer_manifest=manifest,
        maximum_action_tokens=2_048,
    )
    return compiler.freeze_registered_language().digest


@contextmanager
def _deterministic_runtime() -> Iterator[None]:
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    benchmark = torch.backends.cudnn.benchmark
    cuda_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    fp16_reduced = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    bf16_reduced = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    fp16_accumulation = torch.backends.cuda.matmul.allow_fp16_accumulation
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_fp16_accumulation = False
        yield
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cuda.matmul.allow_tf32 = cuda_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = fp16_reduced
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = bf16_reduced
        torch.backends.cuda.matmul.allow_fp16_accumulation = fp16_accumulation


def _install_fake_stack(monkeypatch: pytest.MonkeyPatch, stack: _FakeStack) -> None:
    monkeypatch.setattr(
        sealed_runtime_module,
        "G03_QWEN_TOKENIZER_VOCABULARY_SIZE",
        _FAKE_TOKENIZER_VOCABULARY_SIZE,
    )
    monkeypatch.setattr(
        sealed_runtime_module,
        "G03_QWEN_MODEL_VOCABULARY_SIZE",
        _FAKE_MODEL_VOCABULARY_SIZE,
    )
    monkeypatch.setitem(sys.modules, "transformers", stack.transformers_module)
    monkeypatch.setitem(sys.modules, "tokenizers", stack.tokenizers_module)


def _load(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    stack: _FakeStack | None = None,
) -> SealedQwenRuntime:
    selected_stack = _fake_stack() if stack is None else stack
    _install_fake_stack(monkeypatch, selected_stack)
    tokenizer_manifest = _tokenizer_manifest(root)
    artifact = build_model_artifact_manifest(
        root,
        model_identifier=G03_QWEN_MODEL_IDENTIFIER,
        revision=G03_QWEN_REVISION,
    )
    with _deterministic_runtime():
        return load_sealed_qwen_runtime(
            root,
            expected_model_identifier=G03_QWEN_MODEL_IDENTIFIER,
            expected_revision=G03_QWEN_REVISION,
            expected_tokenizer_binding_digest=tokenizer_manifest.digest,
            expected_compiler_manifest_digest=_compiler_digest(tokenizer_manifest),
            expected_artifact_manifest_digest=artifact.digest,
            dtype=torch.float32,
            device="cpu",
            load_config=_load_config(),
        )


def test_noninjectable_local_only_loader_seals_exact_runtime_and_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _write_artifacts(tmp_path)
    stack = _fake_stack()
    handle = _load(root, monkeypatch, stack=stack)
    assert is_exact_sealed_qwen_runtime(handle)
    assert SEALED_QWEN_RUNTIME_AUTHORIZES_EXECUTION is False
    assert handle.manifest.model_identifier == G03_QWEN_MODEL_IDENTIFIER
    assert handle.manifest.revision == G03_QWEN_REVISION
    assert (
        handle.manifest.tokenizer_vocabulary_size
        == _FAKE_TOKENIZER_VOCABULARY_SIZE
        < handle.manifest.model_vocabulary_size
        == _FAKE_MODEL_VOCABULARY_SIZE
    )
    assert json.loads(handle.manifest.to_json())["digest"] == handle.manifest.digest
    with _deterministic_runtime():
        assert handle.reauthenticate() == handle.manifest
        assert require_exact_sealed_qwen_runtime(handle) is handle
        registry = handle.trainable_parameter_registry
        assert registry.manifest.all_parameters_trainable is True
        assert registry.manifest.total_unique_parameter_count == 3
        assert any(record.canonical_name == "unused" for record in registry.manifest.records)
        assert len(registry.parameters) == 3

    tokenizer_calls = cast(Any, stack.auto_tokenizer).calls
    model_calls = cast(Any, stack.auto_model).calls
    assert len(tokenizer_calls) == len(model_calls) == 1
    expected_root = str(root.resolve())
    assert tokenizer_calls[0] == (
        expected_root,
        {
            "revision": G03_QWEN_REVISION,
            "local_files_only": True,
            "trust_remote_code": False,
            "use_fast": True,
        },
    )
    assert model_calls[0] == (
        expected_root,
        {
            "revision": G03_QWEN_REVISION,
            "local_files_only": True,
            "trust_remote_code": False,
            "torch_dtype": torch.float32,
            "low_cpu_mem_usage": True,
            "use_safetensors": True,
            "attn_implementation": "eager",
            "output_loading_info": True,
        },
    )
    signature = inspect.signature(load_sealed_qwen_runtime)
    assert {
        "model",
        "tokenizer",
        "provider",
        "compiler",
        "loader",
        "model_loader",
        "tokenizer_loader",
    }.isdisjoint(signature.parameters)


def test_artifact_digest_and_tokenizer_live_label_mismatches_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _write_artifacts(tmp_path)
    stack = _fake_stack()
    _install_fake_stack(monkeypatch, stack)
    manifest = _tokenizer_manifest(root)
    with (
        _deterministic_runtime(),
        pytest.raises(SealedQwenRuntimeError, match="artifact manifest differs"),
    ):
        load_sealed_qwen_runtime(
            root,
            expected_model_identifier=G03_QWEN_MODEL_IDENTIFIER,
            expected_revision=G03_QWEN_REVISION,
            expected_tokenizer_binding_digest=manifest.digest,
            expected_compiler_manifest_digest=_compiler_digest(manifest),
            expected_artifact_manifest_digest="0" * 64,
            dtype=torch.float32,
            device="cpu",
            load_config=_load_config(),
        )
    assert not cast(Any, stack.auto_tokenizer).calls
    assert not cast(Any, stack.auto_model).calls

    changed_stack = _fake_stack(changed_eos=True)
    _install_fake_stack(monkeypatch, changed_stack)
    artifact = build_model_artifact_manifest(
        root,
        model_identifier=G03_QWEN_MODEL_IDENTIFIER,
        revision=G03_QWEN_REVISION,
    )
    with (
        _deterministic_runtime(),
        pytest.raises(SealedQwenRuntimeError, match="tokenizer binding differs"),
    ):
        load_sealed_qwen_runtime(
            root,
            expected_model_identifier=G03_QWEN_MODEL_IDENTIFIER,
            expected_revision=G03_QWEN_REVISION,
            expected_tokenizer_binding_digest=manifest.digest,
            expected_compiler_manifest_digest=_compiler_digest(manifest),
            expected_artifact_manifest_digest=artifact.digest,
            dtype=torch.float32,
            device="cpu",
            load_config=_load_config(),
        )


def test_loader_rejects_incomplete_weight_loading_information(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _write_artifacts(tmp_path)
    stack = _fake_stack(missing_model_keys=True)
    _install_fake_stack(monkeypatch, stack)
    tokenizer_manifest = _tokenizer_manifest(root)
    artifact = build_model_artifact_manifest(
        root,
        model_identifier=G03_QWEN_MODEL_IDENTIFIER,
        revision=G03_QWEN_REVISION,
    )
    with (
        _deterministic_runtime(),
        pytest.raises(SealedQwenRuntimeError, match="reported missing"),
    ):
        load_sealed_qwen_runtime(
            root,
            expected_model_identifier=G03_QWEN_MODEL_IDENTIFIER,
            expected_revision=G03_QWEN_REVISION,
            expected_tokenizer_binding_digest=tokenizer_manifest.digest,
            expected_compiler_manifest_digest=_compiler_digest(tokenizer_manifest),
            expected_artifact_manifest_digest=artifact.digest,
            dtype=torch.float32,
            device="cpu",
            load_config=_load_config(),
        )


def test_exact_type_predicate_rejects_protocol_liar_and_subclass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handle = _load(_write_artifacts(tmp_path), monkeypatch)

    class Liar:
        manifest = handle.manifest
        provider = handle.provider
        tokenizer = handle.tokenizer
        compiler = handle.compiler

        def reauthenticate(self) -> object:
            return self.manifest

    class RuntimeSubclass(SealedQwenRuntime):
        pass

    subclass = cast(Any, object.__new__(RuntimeSubclass))
    assert not is_exact_sealed_qwen_runtime(Liar())
    assert not is_exact_sealed_qwen_runtime(subclass)
    with pytest.raises(TypeError, match="exact type"):
        require_exact_sealed_qwen_runtime(Liar())
    with pytest.raises(TypeError, match="exact type"):
        require_exact_sealed_qwen_runtime(subclass)


def test_handle_reauthentication_rejects_live_tokenizer_label_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handle = _load(_write_artifacts(tmp_path), monkeypatch)
    tokenizer = cast(Any, handle.tokenizer)
    tokenizer.eos_token_id = 202
    with (
        _deterministic_runtime(),
        pytest.raises(SealedQwenRuntimeError, match="tokenizer binding changed"),
    ):
        handle.reauthenticate()


def test_handle_reauthentication_rejects_tokenizer_override_and_backend_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    override_handle = _load(_write_artifacts(tmp_path), monkeypatch)
    override_tokenizer = cast(Any, override_handle.tokenizer)

    def changed_converter(_self: object, _token: str) -> int:
        return 7

    override_tokenizer.convert_tokens_to_ids = types.MethodType(
        changed_converter,
        override_tokenizer,
    )
    with (
        _deterministic_runtime(),
        pytest.raises(SealedQwenRuntimeError, match="instance method monkeypatches"),
    ):
        override_handle.reauthenticate()
    del override_tokenizer.convert_tokens_to_ids

    original_backend = override_tokenizer.backend_tokenizer
    override_tokenizer.backend_tokenizer = Tokenizer()
    with (
        _deterministic_runtime(),
        pytest.raises(SealedQwenRuntimeError, match="tokenizer class/method identity changed"),
    ):
        override_handle.reauthenticate()
    override_tokenizer.backend_tokenizer = original_backend

    original_backend.marker = "changed-backend-state"
    with (
        _deterministic_runtime(),
        pytest.raises(SealedQwenRuntimeError, match="tokenizer class/method identity changed"),
    ):
        override_handle.reauthenticate()


def test_model_facts_ignore_deceptive_module_parameter_and_buffer_enumerators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_stack(monkeypatch, _fake_stack())
    model = Qwen2ForCausalLM().eval()

    def root_only_modules(self: Qwen2ForCausalLM) -> Iterator[torch.nn.Module]:
        yield self

    def one_visible_parameter(
        self: Qwen2ForCausalLM,
    ) -> Iterator[torch.nn.Parameter]:
        yield self.unused

    def no_visible_buffers(self: Qwen2ForCausalLM) -> Iterator[torch.Tensor]:
        _ = self
        yield from ()

    monkeypatch.setattr(Qwen2ForCausalLM, "modules", root_only_modules)
    monkeypatch.setattr(Qwen2ForCausalLM, "parameters", one_visible_parameter)
    monkeypatch.setattr(Qwen2ForCausalLM, "buffers", no_visible_buffers)
    assert tuple(model.modules()) == (model,)
    assert tuple(model.parameters()) == (model.unused,)
    assert tuple(model.buffers()) == ()

    model.projection.training = True
    with pytest.raises(SealedQwenRuntimeError, match="every submodule must be in eval"):
        sealed_runtime_module._model_facts(
            model,
            expected_dtype=torch.float32,
            expected_device=torch.device("cpu"),
        )

    model.projection.training = False
    model.projection.weight.requires_grad_(False)
    with pytest.raises(SealedQwenRuntimeError, match="every model parameter"):
        sealed_runtime_module._model_facts(
            model,
            expected_dtype=torch.float32,
            expected_device=torch.device("cpu"),
        )

    model.projection.weight.requires_grad_(True)
    model.projection.register_buffer(
        "hidden_precision_buffer",
        torch.ones((1,), dtype=torch.float64),
    )
    with pytest.raises(SealedQwenRuntimeError, match="exact requested dtype"):
        sealed_runtime_module._model_facts(
            model,
            expected_dtype=torch.float32,
            expected_device=torch.device("cpu"),
        )


def test_internal_update_session_refreshes_rolls_back_and_marks_corrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handle = _load(_write_artifacts(tmp_path), monkeypatch)
    with _deterministic_runtime():
        session = handle._begin_authenticated_update()
        pre = session.pre_manifest
        parameter = session.trainable_parameter_registry.parameters[0]
        with torch.no_grad():
            parameter.add_(0.125)
        with pytest.raises(SealedQwenRuntimeError, match="active authenticated update"):
            _ = handle.provider
        post = handle._commit_authenticated_update(session)
        assert post.provider_policy_state_digest != pre.provider_policy_state_digest
        assert post.model_provenance_digest != pre.model_provenance_digest
        assert handle.reauthenticate() == post

        rollback_session = handle._begin_authenticated_update()
        rollback_parameter = rollback_session.trainable_parameter_registry.parameters[0]
        snapshot = rollback_parameter.detach().clone()
        with torch.no_grad():
            rollback_parameter.mul_(0.5)
            rollback_parameter.copy_(snapshot)
        assert handle._rollback_authenticated_update(rollback_session) == post

        corrupt_session = handle._begin_authenticated_update()
        handle._mark_authenticated_update_corrupt(corrupt_session)
        with pytest.raises(SealedQwenRuntimeError, match="marked corrupt"):
            handle.reauthenticate()


def test_prepared_update_stays_unpublished_and_bad_rollback_corrupts_under_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handle = _load(_write_artifacts(tmp_path), monkeypatch)
    with _deterministic_runtime():
        session = handle._begin_authenticated_update()
        pre = session.pre_manifest
        parameter = session.trainable_parameter_registry.parameters[0]
        with torch.no_grad():
            parameter.add_(0.25)
        prepared = handle._prepare_authenticated_update(session)
        assert prepared.manifest.provider_policy_state_digest != pre.provider_policy_state_digest
        assert handle._manifest == pre
        assert handle._provider is not prepared._provider
        with pytest.raises(
            SealedQwenRuntimeError,
            match="rollback produced different policy evidence",
        ):
            handle._rollback_authenticated_update(session)
        assert handle._update_corrupt is True
        assert handle._update_active is False
        with pytest.raises(SealedQwenRuntimeError, match="marked corrupt"):
            handle.reauthenticate()


def test_compiler_digest_and_load_config_subclass_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _write_artifacts(tmp_path)
    stack = _fake_stack()
    _install_fake_stack(monkeypatch, stack)
    tokenizer_manifest = _tokenizer_manifest(root)
    artifact = build_model_artifact_manifest(
        root,
        model_identifier=G03_QWEN_MODEL_IDENTIFIER,
        revision=G03_QWEN_REVISION,
    )

    class LoadConfigSubclass(SealedQwenLoadConfig):
        pass

    subclass = LoadConfigSubclass(
        transformers_version=_TRANSFORMERS_VERSION,
        tokenizers_version=_TOKENIZERS_VERSION,
        attention_implementation="eager",
        low_cpu_mem_usage=True,
        use_safetensors=True,
        maximum_action_tokens=2_048,
    )
    with _deterministic_runtime(), pytest.raises(TypeError, match="exact type"):
        load_sealed_qwen_runtime(
            root,
            expected_model_identifier=G03_QWEN_MODEL_IDENTIFIER,
            expected_revision=G03_QWEN_REVISION,
            expected_tokenizer_binding_digest=tokenizer_manifest.digest,
            expected_compiler_manifest_digest="0" * 64,
            expected_artifact_manifest_digest=artifact.digest,
            dtype=torch.float32,
            device="cpu",
            load_config=subclass,
        )
    with (
        _deterministic_runtime(),
        pytest.raises(SealedQwenRuntimeError, match="compiler binding differs"),
    ):
        load_sealed_qwen_runtime(
            root,
            expected_model_identifier=G03_QWEN_MODEL_IDENTIFIER,
            expected_revision=G03_QWEN_REVISION,
            expected_tokenizer_binding_digest=tokenizer_manifest.digest,
            expected_compiler_manifest_digest="0" * 64,
            expected_artifact_manifest_digest=artifact.digest,
            dtype=torch.float32,
            device="cpu",
            load_config=_load_config(),
        )


def test_incomplete_artifacts_and_nondeterministic_runtime_fail_before_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _write_artifacts(tmp_path)
    (root / "model.safetensors").unlink()
    stack = _fake_stack()
    _install_fake_stack(monkeypatch, stack)
    artifact = build_model_artifact_manifest(
        root,
        model_identifier=G03_QWEN_MODEL_IDENTIFIER,
        revision=G03_QWEN_REVISION,
    )
    manifest = _tokenizer_manifest(root)
    with (
        _deterministic_runtime(),
        pytest.raises(SealedQwenRuntimeError, match="no safetensors"),
    ):
        load_sealed_qwen_runtime(
            root,
            expected_model_identifier=G03_QWEN_MODEL_IDENTIFIER,
            expected_revision=G03_QWEN_REVISION,
            expected_tokenizer_binding_digest=manifest.digest,
            expected_compiler_manifest_digest=_compiler_digest(manifest),
            expected_artifact_manifest_digest=artifact.digest,
            dtype=torch.float32,
            device="cpu",
            load_config=_load_config(),
        )
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(False)
        with pytest.raises(SealedQwenRuntimeError, match=r"deterministic.*enabled"):
            load_sealed_qwen_runtime(
                root,
                expected_model_identifier=G03_QWEN_MODEL_IDENTIFIER,
                expected_revision=G03_QWEN_REVISION,
                expected_tokenizer_binding_digest=manifest.digest,
                expected_compiler_manifest_digest=_compiler_digest(manifest),
                expected_artifact_manifest_digest=artifact.digest,
                dtype=torch.float32,
                device="cpu",
                load_config=_load_config(),
            )
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
