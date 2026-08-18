from __future__ import annotations

import hashlib
import inspect
import json
import os
import types
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from goalzendo_interactive.action_tokenization_v2 import (
    FragmentActionTokenCompiler,
    TokenizerBindingManifest,
)
from goalzendo_interactive.authenticated_model_provider_v2 import (
    AUTHENTICATED_INCREMENTAL_CACHE_CONTRACT_ID,
    AUTHENTICATED_MODEL_PROVIDER_AUTHORIZES_EXECUTION,
    AUTHENTICATED_MODEL_PROVIDER_CONTRACT_ID,
    AUTHENTICATED_MODEL_PROVIDER_SCHEMA_VERSION,
    AuthenticatedCausalLMProvider,
    AuthenticatedIncrementalCacheProvider,
    AuthenticatedModelProviderError,
    IncrementalCacheMismatchError,
    TrainableParameterRegistry,
    build_executable_behavior_manifest,
    build_model_artifact_manifest,
    build_semantic_module_attribute_manifest,
    build_tensor_state_manifest,
    compare_incremental_cache_trace,
    is_exact_authenticated_causal_lm_provider,
    is_exact_trainable_parameter_registry,
    provider_manifest,
    require_exact_authenticated_causal_lm_provider,
    require_exact_trainable_parameter_registry,
)
from goalzendo_interactive.authenticated_rollouts_v2 import (
    ModelPolicyProvenance,
    ProvenancedFullForwardLogitsProvider,
)
from goalzendo_interactive.authenticated_sampler_v2 import (
    replay_authenticated_sample,
    sample_authenticated_action,
)
from goalzendo_interactive.dialogue import DialogueMessage
from goalzendo_interactive.policy_randomness_v2 import PolicyTurnSeed

_REVISION = "a" * 40
_MODEL_IDENTIFIER = "test/tiny-causal-lm"


class TinyConfig:
    def __init__(self, *, vocab_size: int = 17, hidden_size: int = 7, marker: str = "v1") -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.marker = marker

    def to_dict(self) -> dict[str, object]:
        return {
            "architectures": ["TinyCausalLM"],
            "hidden_size": self.hidden_size,
            "id2label": {0: "LABEL_0", 1: "LABEL_1"},
            "marker": self.marker,
            "nested": {"activation": "linear", "scale": 1.0},
            "vocab_size": self.vocab_size,
        }


@dataclass
class TinyOutput:
    logits: torch.Tensor


class TinyCausalLM(torch.nn.Module):
    def __init__(self, *, seed: int = 123) -> None:
        super().__init__()
        self.config = TinyConfig()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.embedding = torch.nn.Embedding(self.config.vocab_size, self.config.hidden_size)
            self.projection = torch.nn.Linear(
                self.config.hidden_size,
                self.config.vocab_size,
                bias=False,
            )
        self.logit_scale: torch.Tensor
        self.register_buffer("logit_scale", torch.tensor(0.75, dtype=torch.float32))

    def forward(self, *, input_ids: torch.Tensor) -> TinyOutput:
        # Cumulative context makes the final row depend on every prompt token.
        hidden = self.embedding(input_ids).cumsum(dim=1) * self.logit_scale
        return TinyOutput(self.projection(hidden))


class WrongRankLM(TinyCausalLM):
    def forward(self, *, input_ids: torch.Tensor) -> TinyOutput:
        return TinyOutput(super().forward(input_ids=input_ids).logits[0])


class WrongVocabularyLM(TinyCausalLM):
    def forward(self, *, input_ids: torch.Tensor) -> TinyOutput:
        return TinyOutput(super().forward(input_ids=input_ids).logits[..., :-1])


class IntegerLogitsLM(TinyCausalLM):
    def forward(self, *, input_ids: torch.Tensor) -> TinyOutput:
        logits = super().forward(input_ids=input_ids).logits
        return TinyOutput(logits.to(dtype=torch.int64))


class NonfiniteLogitsLM(TinyCausalLM):
    def forward(self, *, input_ids: torch.Tensor) -> TinyOutput:
        logits = super().forward(input_ids=input_ids).logits.clone()
        logits[..., 0] = torch.inf
        return TinyOutput(logits)


class DetachedLogitsLM(TinyCausalLM):
    def forward(self, *, input_ids: torch.Tensor) -> TinyOutput:
        return TinyOutput(super().forward(input_ids=input_ids).logits.detach())


class MissingLogitsLM(TinyCausalLM):
    def forward(self, *, input_ids: torch.Tensor) -> TinyOutput:
        _ = super().forward(input_ids=input_ids)
        return cast(TinyOutput, object())


class AliasedStateLM(TinyCausalLM):
    def __init__(self, *, seed: int = 123) -> None:
        super().__init__(seed=seed)
        # Duplicate Parameter registration and a noncontiguous buffer view
        # exercise stable alias-group construction without hashing pointers.
        self.tied_embedding = self.embedding.weight
        alias_base = torch.arange(8, dtype=torch.float32)
        self.register_buffer("alias_base", alias_base)
        self.register_buffer("alias_view", alias_base[1::2])


class ForwardMutatingLM(TinyCausalLM):
    def forward(self, *, input_ids: torch.Tensor) -> TinyOutput:
        with torch.no_grad():
            self.logit_scale.add_(0.01)
        return super().forward(input_ids=input_ids)


class UnusedTrainableLM(TinyCausalLM):
    def __init__(self) -> None:
        super().__init__()
        self.unused = torch.nn.Parameter(torch.tensor([1.0, 2.0]))


class NonPersistentBufferLM(TinyCausalLM):
    def __init__(self) -> None:
        super().__init__()
        self._non_persistent_buffers_set.add("logit_scale")


class UnregisteredTensorStateLM(TinyCausalLM):
    def __init__(self) -> None:
        super().__init__()
        self.ordinary_scale = torch.tensor(1.25)

    def forward(self, *, input_ids: torch.Tensor) -> TinyOutput:
        output = super().forward(input_ids=input_ids)
        return TinyOutput(output.logits * self.ordinary_scale)


def _changed_forward(self: TinyCausalLM, *, input_ids: torch.Tensor) -> TinyOutput:
    original = self.embedding(input_ids).cumsum(dim=1) * self.logit_scale
    return TinyOutput(self.projection(original) + 17.0)


@dataclass
class TinyCachedOutput:
    logits: torch.Tensor
    past_key_values: object | None


class TinyCachedCausalLM(torch.nn.Module):
    """Small exact cumulative model with the HuggingFace cache call surface."""

    def __init__(self, *, seed: int = 456, vocab_size: int = 17) -> None:
        super().__init__()
        self.config = TinyConfig(vocab_size=vocab_size)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.embedding = torch.nn.Embedding(self.config.vocab_size, self.config.hidden_size)
            self.projection = torch.nn.Linear(
                self.config.hidden_size,
                self.config.vocab_size,
                bias=False,
            )
        # Binary fractions make cumulative full-prefix and one-token cached
        # arithmetic bit-identical rather than merely numerically close.
        with torch.no_grad():
            embedding_values = torch.arange(self.embedding.weight.numel()).reshape_as(
                self.embedding.weight
            )
            projection_values = torch.arange(self.projection.weight.numel()).reshape_as(
                self.projection.weight
            )
            self.embedding.weight.copy_(
                (torch.remainder(embedding_values + seed, 17) - 8).to(torch.float32) / 16
            )
            self.projection.weight.copy_(
                (torch.remainder(projection_values + seed, 13) - 6).to(torch.float32) / 16
            )
        self.logit_scale: torch.Tensor
        self.register_buffer("logit_scale", torch.tensor(0.5, dtype=torch.float32))

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: object | None = None,
        use_cache: bool = False,
        return_dict: bool = True,
    ) -> TinyCachedOutput:
        assert return_dict
        if attention_mask is not None:
            assert attention_mask.ndim == 2 and attention_mask.shape[0] == 1
        embedded = self.embedding(input_ids)
        prior = torch.zeros(
            (input_ids.shape[0], self.config.hidden_size),
            dtype=embedded.dtype,
            device=embedded.device,
        )
        if past_key_values is not None:
            legacy = cast(tuple[tuple[torch.Tensor, torch.Tensor], ...], past_key_values)
            prior = legacy[0][0]
        cumulative = embedded.cumsum(dim=1) + prior[:, None, :]
        logits = self.projection(cumulative * self.logit_scale)
        next_cache: object | None = None
        if use_cache:
            final_state = cumulative[:, -1, :].detach().clone()
            next_cache = ((final_state, final_state.clone()),)
        return TinyCachedOutput(logits=logits, past_key_values=next_cache)


class CachedBiasCausalLM(TinyCachedCausalLM):
    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: object | None = None,
        use_cache: bool = False,
        return_dict: bool = True,
    ) -> TinyCachedOutput:
        result = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=return_dict,
        )
        if past_key_values is not None:
            result.logits = result.logits + 0.125
        return result


class NonfiniteCacheCausalLM(TinyCachedCausalLM):
    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: object | None = None,
        use_cache: bool = False,
        return_dict: bool = True,
    ) -> TinyCachedOutput:
        result = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=return_dict,
        )
        if use_cache:
            result.past_key_values = ((torch.full((1, 1), torch.inf), torch.ones((1, 1))),)
        return result


class CacheForwardMutatingLM(TinyCachedCausalLM):
    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: object | None = None,
        use_cache: bool = False,
        return_dict: bool = True,
    ) -> TinyCachedOutput:
        with torch.no_grad():
            self.logit_scale.add_(0.01)
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=return_dict,
        )


class ExactCharacterChatTokenizer:
    def apply_chat_template(
        self,
        conversation: Sequence[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        text = "".join(
            f"<|{message['role']}|>\n{message['content']}<|end|>\n" for message in conversation
        )
        return text + ("<|assistant|>\n" if add_generation_prompt else "")

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [ord(character) for character in text]

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return "".join(chr(token_id) for token_id in token_ids)


def _write_artifact_tree(parent: Path, name: str, *, vocab_size: int = 17) -> Path:
    root = parent / name
    (root / "weights").mkdir(parents=True)
    (root / "config.json").write_text(
        json.dumps(
            {"architectures": ["TinyCausalLM"], "vocab_size": vocab_size},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (root / "weights" / "model.bin").write_bytes(bytes(range(64)))
    (root / "tokenizer.model").write_bytes(b"tiny-tokenizer\x00v1")
    return root


def _provider(
    tmp_path: Path,
    *,
    name: str,
    model: TinyCausalLM | None = None,
) -> tuple[AuthenticatedCausalLMProvider, TinyCausalLM, Path]:
    selected_model = TinyCausalLM() if model is None else model
    selected_model.eval()
    root = _write_artifact_tree(tmp_path, name)
    provider = AuthenticatedCausalLMProvider(
        selected_model,
        root,
        model_identifier=_MODEL_IDENTIFIER,
        revision=_REVISION,
    )
    return provider, selected_model, root


def _cached_provider(
    tmp_path: Path,
    *,
    name: str,
    model: TinyCachedCausalLM | None = None,
    vocab_size: int = 17,
) -> tuple[AuthenticatedIncrementalCacheProvider, AuthenticatedCausalLMProvider, TinyCachedCausalLM]:
    selected_model = TinyCachedCausalLM(vocab_size=vocab_size) if model is None else model
    selected_model.eval()
    root = _write_artifact_tree(tmp_path, name, vocab_size=selected_model.config.vocab_size)
    base = AuthenticatedCausalLMProvider(
        selected_model,
        root,
        model_identifier=_MODEL_IDENTIFIER,
        revision=_REVISION,
    )
    return AuthenticatedIncrementalCacheProvider(base), base, selected_model


def _tokenizer_manifest(*, vocabulary_size: int) -> TokenizerBindingManifest:
    return TokenizerBindingManifest(
        repository_id="test/exact-character-tokenizer",
        revision="b" * 40,
        tokenizer_json_sha256="c" * 64,
        tokenizer_config_sha256="d" * 64,
        chat_template_sha256="e" * 64,
        backend_name="test-character-tokenizer",
        backend_version="1.0.0",
        vocabulary_size=vocabulary_size,
        special_token_ids=(
            ("bos", None),
            ("eos", None),
            ("pad", None),
            ("unk", None),
        ),
    )


def test_artifact_and_policy_digests_are_deterministic_and_path_independent(
    tmp_path: Path,
) -> None:
    first_root = _write_artifact_tree(tmp_path, "root-a")
    second_root = _write_artifact_tree(tmp_path, "unrelated/root-b")
    first_artifacts = build_model_artifact_manifest(
        first_root,
        model_identifier=_MODEL_IDENTIFIER,
        revision=_REVISION,
    )
    second_artifacts = build_model_artifact_manifest(
        second_root,
        model_identifier=_MODEL_IDENTIFIER,
        revision=_REVISION,
    )
    assert first_artifacts == second_artifacts
    assert first_artifacts.digest == second_artifacts.digest
    assert [record.relative_path for record in first_artifacts.files] == [
        "config.json",
        "tokenizer.model",
        "weights/model.bin",
    ]
    assert json.loads(first_artifacts.to_json())["digest"] == first_artifacts.digest

    first_model = TinyCausalLM(seed=991).eval()
    second_model = TinyCausalLM(seed=991).eval()
    first = AuthenticatedCausalLMProvider(
        first_model,
        first_root,
        model_identifier=_MODEL_IDENTIFIER,
        revision=_REVISION,
    )
    second = AuthenticatedCausalLMProvider(
        second_model,
        second_root,
        model_identifier=_MODEL_IDENTIFIER,
        revision=_REVISION,
    )
    assert first.policy_state_digest == second.policy_state_digest
    assert first.model_provenance == second.model_provenance
    assert first.model_provenance_digest == second.model_provenance_digest
    assert first.tensor_state_manifest == second.tensor_state_manifest
    assert first.reauthenticate_policy_state() == first.model_provenance


def test_computed_provenance_exactly_integrates_rollout_fields_and_is_nonauthorizing(
    tmp_path: Path,
) -> None:
    provider, _, _ = _provider(tmp_path, name="provenance")
    assert isinstance(provider, ProvenancedFullForwardLogitsProvider)
    provenance = provider.model_provenance
    assert type(provenance) is ModelPolicyProvenance
    assert provenance.model_identifier == _MODEL_IDENTIFIER
    assert provenance.revision == _REVISION
    assert provenance.artifact_manifest_sha256 == provider.artifact_manifest.digest
    assert provenance.runtime_stack_sha256 == provider.runtime_manifest.digest
    assert provenance.policy_state_digest == provider.policy_state_digest
    assert provider.model_provenance_digest == provenance.digest

    manifest = provider_manifest(provider)
    assert manifest["schema_version"] == AUTHENTICATED_MODEL_PROVIDER_SCHEMA_VERSION == 4
    assert manifest["contract_id"] == AUTHENTICATED_MODEL_PROVIDER_CONTRACT_ID
    assert manifest["authorizes_execution"] is False
    assert AUTHENTICATED_MODEL_PROVIDER_AUTHORIZES_EXECUTION is False
    runtime = provider.runtime_manifest
    assert runtime.module_modes
    assert all(not training for _, _, training in runtime.module_modes)
    assert any(requires_grad for _, _, requires_grad in runtime.tensor_requires_grad)
    assert runtime.autocast_device_type == "cpu"
    assert runtime.deterministic_algorithms == torch.are_deterministic_algorithms_enabled()
    assert runtime.executable_manifest_sha256 == provider.executable_manifest.digest
    assert (
        runtime.trainable_parameter_registry_sha256
        == provider.trainable_parameter_registry.manifest.digest
    )
    assert manifest["executable_manifest"] == {
        **provider.executable_manifest.as_obj(),
        "digest": provider.executable_manifest.digest,
    }


def test_alias_storage_and_noncontiguous_views_have_deterministic_manifests(
    tmp_path: Path,
) -> None:
    first_root = _write_artifact_tree(tmp_path, "aliases-a")
    second_root = _write_artifact_tree(tmp_path, "aliases-b")
    first = AuthenticatedCausalLMProvider(
        AliasedStateLM(seed=777).eval(),
        first_root,
        model_identifier=_MODEL_IDENTIFIER,
        revision=_REVISION,
    )
    second = AuthenticatedCausalLMProvider(
        AliasedStateLM(seed=777).eval(),
        second_root,
        model_identifier=_MODEL_IDENTIFIER,
        revision=_REVISION,
    )
    assert first.tensor_state_manifest == second.tensor_state_manifest
    assert first.policy_state_digest == second.policy_state_digest

    records = {record.name: record for record in first.tensor_state_manifest.tensors}
    assert records["embedding.weight"].storage_alias_group == records[
        "tied_embedding"
    ].storage_alias_group
    assert records["alias_base"].storage_alias_group == records[
        "alias_view"
    ].storage_alias_group
    assert records["alias_view"].stride == (2,)
    assert records["alias_view"].storage_offset == 1
    assert records["alias_view"].storage_byte_length == records[
        "alias_base"
    ].storage_byte_length


def test_constructor_cannot_accept_arbitrary_digest_labels(tmp_path: Path) -> None:
    parameters = inspect.signature(AuthenticatedCausalLMProvider).parameters
    forbidden = {
        "artifact_manifest_sha256",
        "runtime_stack_sha256",
        "policy_state_digest",
        "model_provenance_digest",
        "model_provenance",
    }
    assert forbidden.isdisjoint(parameters)

    root = _write_artifact_tree(tmp_path, "no-caller-digest")
    model = TinyCausalLM().eval()
    untyped_constructor: Any = AuthenticatedCausalLMProvider
    with pytest.raises(TypeError):
        untyped_constructor(
            model,
            root,
            model_identifier=_MODEL_IDENTIFIER,
            revision=_REVISION,
            policy_state_digest="f" * 64,
        )


def test_full_reauthentication_rejects_artifact_config_tensor_and_runtime_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_provider, _, artifact_root = _provider(tmp_path, name="artifact-tamper")
    (artifact_root / "config.json").write_bytes(b"changed-artifact-bytes")
    with pytest.raises(AuthenticatedModelProviderError, match="artifact bytes"):
        artifact_provider.reauthenticate_policy_state()

    config_provider, config_model, _ = _provider(tmp_path, name="config-tamper")
    config_model.config.marker = "changed"
    with pytest.raises(AuthenticatedModelProviderError, match="configuration changed"):
        config_provider.reauthenticate_policy_state()

    tensor_provider, tensor_model, _ = _provider(tmp_path, name="tensor-tamper")
    assert tensor_provider.reauthenticate_policy_state() == tensor_provider.model_provenance
    # .data bypasses Tensor._version, so only the explicit byte reauthentication catches it.
    tensor_model.embedding.weight.data.reshape(-1)[0].add_(1.0)
    with pytest.raises(AuthenticatedModelProviderError, match="tensor-state bytes changed"):
        tensor_provider.reauthenticate_policy_state()

    runtime_provider, _, _ = _provider(tmp_path, name="runtime-tamper")
    original_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    changed_cublas = ":16:8" if original_cublas != ":16:8" else ":4096:8"
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", changed_cublas)
    with pytest.raises(AuthenticatedModelProviderError, match="arithmetic runtime"):
        runtime_provider.reauthenticate_policy_state()


def test_direct_manifest_catches_nonpersistent_and_state_dict_hook_hidden_bytes(
    tmp_path: Path,
) -> None:
    nonpersistent = NonPersistentBufferLM().eval()
    provider, model, _ = _provider(
        tmp_path,
        name="nonpersistent-buffer",
        model=nonpersistent,
    )
    records = {
        (record.kind, record.name): record
        for record in provider.tensor_state_manifest.tensors
    }
    assert records[("buffer", "logit_scale")].persistent is False
    before_logits = model(input_ids=torch.tensor([[1, 2]])).logits.detach().clone()
    model.logit_scale.data.mul_(1.25)
    after_logits = model(input_ids=torch.tensor([[1, 2]])).logits.detach().clone()
    assert not torch.equal(before_logits, after_logits)
    with pytest.raises(AuthenticatedModelProviderError, match="tensor-state bytes changed"):
        provider.reauthenticate_policy_state()

    hooked = TinyCausalLM().eval()
    frozen = {name: value.detach().clone() for name, value in hooked.state_dict().items()}

    def hide_live_bytes(
        _module: torch.nn.Module,
        state: dict[str, torch.Tensor],
        _prefix: str,
        _metadata: dict[str, object],
    ) -> None:
        state.clear()
        state.update({name: value.clone() for name, value in frozen.items()})

    cast(Any, hooked).register_state_dict_post_hook(hide_live_bytes)
    direct_before = build_tensor_state_manifest(hooked)
    hooked.embedding.weight.data.reshape(-1)[0].add_(7.0)
    assert all(torch.equal(value, frozen[name]) for name, value in hooked.state_dict().items())
    direct_after = build_tensor_state_manifest(hooked)
    assert direct_after.digest != direct_before.digest
    with pytest.raises(AuthenticatedModelProviderError, match="hooks are forbidden"):
        AuthenticatedCausalLMProvider(
            hooked,
            _write_artifact_tree(tmp_path, "state-dict-hook"),
            model_identifier=_MODEL_IDENTIFIER,
            revision=_REVISION,
        )


def test_semantic_manifest_guards_scalars_and_unregistered_tensor_bytes(tmp_path: Path) -> None:
    scalar_provider, scalar_model, _ = _provider(tmp_path, name="semantic-scalar")
    manifest = build_semantic_module_attribute_manifest(scalar_model)
    embedding = next(record for record in manifest.modules if record.name == "embedding")
    attributes = dict(embedding.attributes)
    assert attributes["scale_grad_by_freq"] is False
    scalar_model.embedding.scale_grad_by_freq = True
    with pytest.raises(AuthenticatedModelProviderError, match="module attributes changed"):
        scalar_provider.full_forward_logits((1, 2))

    tensor_provider, tensor_model, _ = _provider(
        tmp_path,
        name="semantic-unregistered-tensor",
        model=UnregisteredTensorStateLM(),
    )
    selected_tensor_model = cast(UnregisteredTensorStateLM, tensor_model)
    before = selected_tensor_model(input_ids=torch.tensor([[1, 2]])).logits.detach().clone()
    selected_tensor_model.ordinary_scale.data.mul_(1.5)
    after = selected_tensor_model(input_ids=torch.tensor([[1, 2]])).logits.detach().clone()
    assert not torch.equal(before, after)
    with pytest.raises(AuthenticatedModelProviderError, match="semantic module-attribute"):
        tensor_provider.reauthenticate_policy_state()


def test_fast_guard_rejects_optimizer_step(tmp_path: Path) -> None:
    provider, model, _ = _provider(tmp_path, name="optimizer")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss = provider.full_forward_logits((1, 2, 3)).square().sum()
    torch.autograd.backward(loss)
    optimizer.step()
    with pytest.raises(AuthenticatedModelProviderError, match="version, or metadata changed"):
        provider.next_token_logits((1, 2, 3))


def test_fast_guard_rejects_in_place_parameter_mutation(tmp_path: Path) -> None:
    provider, model, _ = _provider(tmp_path, name="in-place")
    with torch.no_grad():
        model.embedding.weight.add_(0.25)
    with pytest.raises(AuthenticatedModelProviderError, match="version, or metadata changed"):
        _ = provider.policy_state_digest


def test_fast_guard_rejects_state_replacement(tmp_path: Path) -> None:
    provider, model, _ = _provider(tmp_path, name="replacement")
    model.projection.weight = torch.nn.Parameter(model.projection.weight.detach().clone())
    with pytest.raises(AuthenticatedModelProviderError, match="identity, version"):
        provider.full_forward_logits((1, 2))


def test_fast_guard_rejects_training_mode_mutation(tmp_path: Path) -> None:
    provider, model, _ = _provider(tmp_path, name="training-mode")
    model.train()
    with pytest.raises(AuthenticatedModelProviderError, match="eval-mode state changed"):
        provider.next_token_logits((1, 2))


def test_fast_guard_rejects_requires_grad_and_ambient_arithmetic_mutation(
    tmp_path: Path,
) -> None:
    grad_provider, grad_model, _ = _provider(tmp_path, name="requires-grad")
    grad_model.embedding.weight.requires_grad_(False)
    with pytest.raises(AuthenticatedModelProviderError, match="version, or metadata changed"):
        _ = grad_provider.policy_state_digest

    autocast_provider, _, _ = _provider(tmp_path, name="autocast")
    with (
        torch.autocast(device_type="cpu", dtype=torch.bfloat16),
        pytest.raises(AuthenticatedModelProviderError, match="autocast context changed"),
    ):
        autocast_provider.next_token_logits((1, 2))

    deterministic_provider, _, _ = _provider(tmp_path, name="determinism")
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    warn_only_before = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(
            not deterministic_before,
            warn_only=warn_only_before,
        )
        with pytest.raises(AuthenticatedModelProviderError, match="arithmetic runtime"):
            _ = deterministic_provider.model_provenance_digest
    finally:
        torch.use_deterministic_algorithms(
            deterministic_before,
            warn_only=warn_only_before,
        )


def test_state_mutation_during_forward_is_rejected_by_post_call_guard(tmp_path: Path) -> None:
    provider, _, _ = _provider(tmp_path, name="forward-mutation", model=ForwardMutatingLM())
    with pytest.raises(AuthenticatedModelProviderError, match="version, or metadata changed"):
        provider.next_token_logits((1, 2, 3))


def test_executable_manifest_binds_inherited_call_and_each_unique_forward(tmp_path: Path) -> None:
    provider, model, _ = _provider(tmp_path, name="executable-manifest")
    manifest = build_executable_behavior_manifest(model)
    assert manifest == provider.executable_manifest
    rows = {(record.module_class, record.role): record for record in manifest.callables}
    model_class = f"{TinyCausalLM.__module__}.{TinyCausalLM.__qualname__}"
    embedding_class = f"{torch.nn.Embedding.__module__}.{torch.nn.Embedding.__qualname__}"
    assert rows[(model_class, "forward")].owner_class == model_class
    assert rows[(model_class, "__call__")].owner_class.endswith(".Module")
    assert rows[(embedding_class, "forward")].owner_class == embedding_class
    assert rows[(model_class, "forward")].source_sha256 != "0" * 64
    assert rows[(model_class, "forward")].code_sha256 != "0" * 64
    assert json.loads(manifest.to_json())["digest"] == manifest.digest


def test_fast_guard_rejects_instance_and_class_forward_monkeypatches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance_provider, instance_model, _ = _provider(tmp_path, name="instance-forward-patch")
    instance_model.forward = types.MethodType(_changed_forward, instance_model)  # type: ignore[method-assign]
    with pytest.raises(AuthenticatedModelProviderError, match="monkeypatches are forbidden"):
        instance_provider.next_token_logits((1, 2, 3))

    class_provider, _, _ = _provider(tmp_path, name="class-forward-patch")
    monkeypatch.setattr(TinyCausalLM, "forward", _changed_forward)
    with pytest.raises(AuthenticatedModelProviderError, match="monkeypatch"):
        class_provider.next_token_logits((1, 2, 3))
    with pytest.raises(AuthenticatedModelProviderError, match="monkeypatch"):
        class_provider.reauthenticate_policy_state()
    with pytest.raises(AuthenticatedModelProviderError, match="class forward monkeypatch"):
        _provider(tmp_path, name="preexisting-class-forward-patch")


def test_constructor_and_fast_guard_reject_all_execution_hook_surfaces(tmp_path: Path) -> None:
    constructor_root = _write_artifact_tree(tmp_path, "constructor-hook")
    constructor_model = TinyCausalLM().eval()
    constructor_model.register_forward_pre_hook(lambda _module, _inputs: None)
    with pytest.raises(AuthenticatedModelProviderError, match="hooks are forbidden"):
        AuthenticatedCausalLMProvider(
            constructor_model,
            constructor_root,
            model_identifier=_MODEL_IDENTIFIER,
            revision=_REVISION,
        )

    forward_provider, forward_model, _ = _provider(tmp_path, name="forward-hook")
    forward_model.register_forward_hook(lambda _module, _inputs, output: output)
    with pytest.raises(AuthenticatedModelProviderError, match="hooks are forbidden"):
        forward_provider.next_token_logits((1, 2))

    backward_provider, backward_model, _ = _provider(tmp_path, name="backward-hook")
    backward_model.register_full_backward_pre_hook(lambda _module, grad_output: grad_output)
    with pytest.raises(AuthenticatedModelProviderError, match="hooks are forbidden"):
        backward_provider.full_forward_logits((1, 2))

    tensor_provider, tensor_model, _ = _provider(tmp_path, name="tensor-hook")
    cast(Any, tensor_model.embedding.weight).register_hook(lambda gradient: gradient)
    with pytest.raises(AuthenticatedModelProviderError, match="backward hooks are forbidden"):
        tensor_provider.full_forward_logits((1, 2))

    post_provider, post_model, _ = _provider(tmp_path, name="post-accumulate-hook")

    def erase_accumulated_gradient(parameter: torch.Tensor) -> None:
        assert parameter.grad is not None
        parameter.grad.zero_()

    cast(Any, post_model.embedding.weight).register_post_accumulate_grad_hook(
        erase_accumulated_gradient
    )
    with pytest.raises(AuthenticatedModelProviderError, match="post-accumulate-grad hooks"):
        post_provider.full_forward_logits((1, 2))

    constructor_post = TinyCausalLM().eval()
    cast(Any, constructor_post.embedding.weight).register_post_accumulate_grad_hook(
        erase_accumulated_gradient
    )
    with pytest.raises(AuthenticatedModelProviderError, match="post-accumulate-grad hooks"):
        AuthenticatedCausalLMProvider(
            constructor_post,
            _write_artifact_tree(tmp_path, "constructor-post-accumulate-hook"),
            model_identifier=_MODEL_IDENTIFIER,
            revision=_REVISION,
        )


def test_direct_hook_scan_rejects_deceptive_modules_enumeration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, model, _ = _provider(tmp_path, name="deceptive-modules-hook")
    inputs = torch.tensor([[1, 2, 3]])
    before = model(input_ids=inputs).logits.detach().clone()

    def root_only_modules(self: TinyCausalLM) -> Iterator[torch.nn.Module]:
        yield self

    monkeypatch.setattr(TinyCausalLM, "modules", root_only_modules)
    assert tuple(model.modules()) == (model,)
    hook = model.projection.register_forward_hook(
        lambda _module, _inputs, output: output + 10.0
    )
    after = model(input_ids=inputs).logits.detach().clone()
    assert torch.equal(after, before + 10.0)

    with pytest.raises(AuthenticatedModelProviderError, match="hooks are forbidden"):
        provider.next_token_logits((1, 2, 3))
    with pytest.raises(AuthenticatedModelProviderError, match="hooks are forbidden"):
        provider.reauthenticate_policy_state()
    hook.remove()


def test_exact_nominal_provider_and_complete_parameter_registry(tmp_path: Path) -> None:
    model = UnusedTrainableLM().eval()
    provider, selected_model, _ = _provider(tmp_path, name="registry", model=model)
    assert selected_model is model
    assert is_exact_authenticated_causal_lm_provider(provider)
    assert require_exact_authenticated_causal_lm_provider(provider) is provider

    class ProviderSubclass(AuthenticatedCausalLMProvider):
        pass

    liar = cast(Any, object.__new__(ProviderSubclass))
    assert not is_exact_authenticated_causal_lm_provider(liar)
    with pytest.raises(TypeError, match="exact type"):
        require_exact_authenticated_causal_lm_provider(liar)

    registry = provider.trainable_parameter_registry
    assert type(registry) is TrainableParameterRegistry
    assert is_exact_trainable_parameter_registry(registry)
    assert require_exact_trainable_parameter_registry(registry) is registry
    assert registry.manifest.all_parameters_trainable is True
    assert registry.manifest.total_unique_parameter_count == 3
    assert registry.manifest.trainable_parameter_count == 3
    assert len(registry.parameters) == 3
    assert any(parameter is model.unused for parameter in registry.parameters)
    assert any(record.canonical_name == "unused" for record in registry.manifest.records)
    assert provider.reauthenticate_policy_state() == provider.model_provenance


def test_full_reauthentication_recomputes_executable_source_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, _, _ = _provider(tmp_path, name="executable-source-reauth")
    original_getsource = inspect.getsource

    def changed_getsource(value: Any) -> str:
        source = original_getsource(value)
        if value is TinyCausalLM.forward:
            return source + "\n# observed source changed after construction\n"
        return source

    monkeypatch.setattr(inspect, "getsource", changed_getsource)
    with pytest.raises(AuthenticatedModelProviderError, match="executable behavior changed"):
        provider.reauthenticate_policy_state()


def test_constructor_requires_eval_mode_single_device_and_strict_artifacts(tmp_path: Path) -> None:
    root = _write_artifact_tree(tmp_path, "requirements")
    with pytest.raises(AuthenticatedModelProviderError, match="eval mode"):
        AuthenticatedCausalLMProvider(
            TinyCausalLM(),
            root,
            model_identifier=_MODEL_IDENTIFIER,
            revision=_REVISION,
        )

    symlink_root = _write_artifact_tree(tmp_path, "symlink-tree")
    (symlink_root / "linked.bin").symlink_to(symlink_root / "weights" / "model.bin")
    with pytest.raises(AuthenticatedModelProviderError, match="symlinks"):
        build_model_artifact_manifest(
            symlink_root,
            model_identifier=_MODEL_IDENTIFIER,
            revision=_REVISION,
        )

    mixed = TinyCausalLM().eval()
    mixed.logit_scale = mixed.logit_scale.to(device="meta")
    with pytest.raises(AuthenticatedModelProviderError, match="exactly one device"):
        AuthenticatedCausalLMProvider(
            mixed,
            root,
            model_identifier=_MODEL_IDENTIFIER,
            revision=_REVISION,
        )


def test_prompt_dependence_no_grad_next_token_and_differentiable_full_forward(
    tmp_path: Path,
) -> None:
    provider, model, _ = _provider(tmp_path, name="forwards")
    first = provider.next_token_logits((1, 2, 3))
    second = provider.next_token_logits((4, 2, 3))
    assert first.shape == (model.config.vocab_size,)
    assert first.is_floating_point()
    assert not first.requires_grad
    assert not torch.equal(first, second)

    model.zero_grad(set_to_none=True)
    full = provider.full_forward_logits((1, 2, 3, 4))
    assert full.shape == (4, model.config.vocab_size)
    assert full.requires_grad
    objective = full[-1].square().sum()
    torch.autograd.backward(objective)
    assert model.embedding.weight.grad is not None
    assert model.projection.weight.grad is not None
    assert torch.count_nonzero(model.embedding.weight.grad).item() > 0
    assert torch.count_nonzero(model.projection.weight.grad).item() > 0
    # Gradient accumulation is not a policy-state mutation.
    assert provider.reauthenticate_policy_state() == provider.model_provenance


@pytest.mark.parametrize(
    "input_ids",
    [
        (),
        cast(tuple[int, ...], [1, 2]),
        (True,),
        (-1,),
        (17,),
        (1.5,),
    ],
)
def test_invalid_input_ids_fail_closed(tmp_path: Path, input_ids: tuple[int, ...]) -> None:
    provider, _, _ = _provider(tmp_path, name=f"bad-input-{input_ids!r}")
    with pytest.raises(AuthenticatedModelProviderError, match="input_ids"):
        provider.next_token_logits(input_ids)
    with pytest.raises(AuthenticatedModelProviderError, match="input_ids"):
        provider.full_forward_logits(input_ids)


@pytest.mark.parametrize(
    ("model", "message", "full_only"),
    [
        (WrongRankLM(), "exact.*shape", False),
        (WrongVocabularyLM(), "exact.*shape", False),
        (IntegerLogitsLM(), "floating dtype", False),
        (NonfiniteLogitsLM(), "non-finite", False),
        (DetachedLogitsLM(), "differentiable graph", True),
        (MissingLogitsLM(), "expose Tensor logits", False),
    ],
)
def test_bad_model_outputs_fail_closed(
    tmp_path: Path,
    model: TinyCausalLM,
    message: str,
    full_only: bool,
) -> None:
    model.eval()
    provider, _, _ = _provider(tmp_path, name=type(model).__name__, model=model)
    if not full_only:
        with pytest.raises(AuthenticatedModelProviderError, match=message):
            provider.next_token_logits((1, 2, 3))
    with pytest.raises(AuthenticatedModelProviderError, match=message):
        provider.full_forward_logits((1, 2, 3))


def test_incremental_cache_matches_stateless_reuses_extensions_and_resets(
    tmp_path: Path,
) -> None:
    cached, stateless, model = _cached_provider(tmp_path, name="incremental-basic")
    assert isinstance(cached, ProvenancedFullForwardLogitsProvider)
    assert cached.policy_state_digest == stateless.policy_state_digest
    assert cached.model_provenance_digest == stateless.model_provenance_digest

    first_prefix = (1, 2)
    first = cached.next_token_logits(first_prefix)
    assert torch.equal(first, stateless.next_token_logits(first_prefix))
    assert cached.cached_prefix_length == 2
    assert cached.reset_count == 0
    assert cached.cache_reuse_count == 0
    assert cached.last_call_reused_cache is False

    extension = (1, 2, 3)
    extended = cached.next_token_logits(extension)
    assert torch.equal(extended, stateless.next_token_logits(extension))
    assert cached.cache_reuse_count == 1
    assert cached.last_call_reused_cache is True

    # Equal, shorter, or divergent prompts are all non-extensions and reset.
    repeated = cached.next_token_logits(extension)
    assert torch.equal(repeated, stateless.next_token_logits(extension))
    assert cached.reset_count == 1
    assert cached.last_call_reused_cache is False
    divergent = (9, 2, 3, 4)
    assert torch.equal(
        cached.next_token_logits(divergent),
        stateless.next_token_logits(divergent),
    )
    assert cached.reset_count == 2

    model.zero_grad(set_to_none=True)
    full = cached.full_forward_logits((1, 2, 3))
    assert full.requires_grad
    torch.autograd.backward(full[-1].square().sum())
    assert model.embedding.weight.grad is not None
    assert torch.count_nonzero(model.embedding.weight.grad).item() > 0


def test_incremental_cache_rejects_cache_output_and_hidden_policy_mutation(
    tmp_path: Path,
) -> None:
    cached, _, _ = _cached_provider(tmp_path, name="cache-tamper")
    _ = cached.next_token_logits((1, 2))
    derived = cast(Any, cached)._past_key_values
    with torch.no_grad():
        derived[0][0].add_(0.5)
    with pytest.raises(AuthenticatedModelProviderError, match="KV-cache identity"):
        cached.next_token_logits((1, 2, 3))

    nonfinite, _, _ = _cached_provider(
        tmp_path,
        name="cache-nonfinite",
        model=NonfiniteCacheCausalLM(),
    )
    with pytest.raises(AuthenticatedModelProviderError, match="non-finite"):
        nonfinite.next_token_logits((1, 2))

    mutating, _, _ = _cached_provider(
        tmp_path,
        name="cache-model-mutation",
        model=CacheForwardMutatingLM(),
    )
    with pytest.raises(AuthenticatedModelProviderError, match="version, or metadata changed"):
        mutating.next_token_logits((1, 2))

    mode_changed, _, mode_model = _cached_provider(tmp_path, name="cache-mode-mutation")
    mode_model.train()
    with pytest.raises(AuthenticatedModelProviderError, match="eval-mode state changed"):
        mode_changed.next_token_logits((1, 2))


def test_prospective_cache_comparator_records_exact_bindings_and_rejects_bias(
    tmp_path: Path,
) -> None:
    cached, stateless, _ = _cached_provider(tmp_path, name="cache-comparator")
    tokenizer_manifest = _tokenizer_manifest(vocabulary_size=17)
    trace = ((1,), (1, 2), (1, 2, 3), (9,), (9, 4))
    report = compare_incremental_cache_trace(
        cached,
        tokenizer_manifest,
        trace,
        absolute_tolerance=0.0,
    )
    assert report.contract_id == AUTHENTICATED_INCREMENTAL_CACHE_CONTRACT_ID
    assert report.policy_state_digest == stateless.policy_state_digest
    assert report.model_provenance_digest == stateless.model_provenance_digest
    assert report.tokenizer_binding_digest == tokenizer_manifest.digest
    assert report.prefix_lengths == (1, 2, 3, 1, 2)
    assert report.reset_count == 1
    assert report.reuse_count == 3
    assert report.maximum_absolute_error_hex == (0.0).hex()
    assert report.within_tolerance is True
    assert json.loads(report.to_json())["digest"] == report.digest
    assert json.loads(report.to_json())["authorizes_execution"] is False

    biased, _, _ = _cached_provider(
        tmp_path,
        name="cache-biased",
        model=CachedBiasCausalLM(),
    )
    with pytest.raises(IncrementalCacheMismatchError) as caught:
        compare_incremental_cache_trace(
            biased,
            tokenizer_manifest,
            ((1,), (1, 2)),
            absolute_tolerance=0.0,
        )
    assert caught.value.report.within_tolerance is False
    assert float.fromhex(caught.value.report.maximum_absolute_error_hex) > 0

    with pytest.raises(AuthenticatedModelProviderError, match="explicitly finite"):
        compare_incremental_cache_trace(
            cached,
            tokenizer_manifest,
            trace,
            absolute_tolerance=float("nan"),
        )


def test_incremental_cache_resamples_identically_through_authenticated_sampler(
    tmp_path: Path,
) -> None:
    tokenizer = ExactCharacterChatTokenizer()
    tokenizer_manifest = _tokenizer_manifest(vocabulary_size=256)
    compiler = FragmentActionTokenCompiler(
        tokenizer,
        tokenizer_manifest=tokenizer_manifest,
        maximum_action_tokens=2_048,
    )
    compiler.freeze_registered_language()
    cached, _, _ = _cached_provider(
        tmp_path,
        name="cache-sampler",
        model=TinyCachedCausalLM(vocab_size=256),
    )
    dialogue = (
        DialogueMessage("system", "Play hidden-law Zendo.", "contract"),
        DialogueMessage("user", "Choose one canonical inquiry action.", "opening"),
    )
    turn_seed = PolicyTurnSeed(
        run_seed=20260811,
        episode_digest=hashlib.sha256(b"cache-sampler-episode").hexdigest(),
        rollout_index=0,
        turn_index=0,
        policy_state_digest=cached.policy_state_digest,
    )
    first = sample_authenticated_action(
        cached,
        tokenizer,
        compiler,
        dialogue,
        turn_seed,
        mode="inquiry",
        terminal_count=None,
        temperature=0.73,
        maximum_sequence_tokens=4_096,
    )
    verified = replay_authenticated_sample(
        first,
        cached,
        tokenizer,
        compiler,
        dialogue,
        absolute_tolerance=0.0,
    )
    assert verified.sample == first
    resets_before = cached.reset_count
    reuses_before = cached.cache_reuse_count
    second = sample_authenticated_action(
        cached,
        tokenizer,
        compiler,
        dialogue,
        turn_seed,
        mode="inquiry",
        terminal_count=None,
        temperature=0.73,
        maximum_sequence_tokens=4_096,
    )
    assert second == first
    assert cached.reset_count == resets_before + 1
    assert cached.cache_reuse_count > reuses_before
