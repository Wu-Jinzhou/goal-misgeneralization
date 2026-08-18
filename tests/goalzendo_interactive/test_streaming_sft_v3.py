from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch.nn import functional as F

import goalzendo_interactive.streaming_sft_v3 as streaming_sft
from goalzendo_interactive._json import json_digest
from goalzendo_interactive.action_tokenization_v2 import (
    FragmentActionTokenCompiler,
    TokenizerBindingManifest,
)
from goalzendo_interactive.actions import ReadyAction, parse_action, serialize_action
from goalzendo_interactive.authenticated_model_provider_v2 import (
    AuthenticatedCausalLMProvider,
    AuthenticatedModelProviderError,
)
from goalzendo_interactive.decision_encoding_v2 import encode_decision_example, verify_decision_example
from goalzendo_interactive.dialogue import Dialogue, DialogueMessage, render_dialogue
from goalzendo_interactive.generation import generate_episode_bank, small_fixture_bank_spec
from goalzendo_interactive.streaming_sft_v3 import (
    STREAMING_SFT_AUTHORIZES_EXECUTION,
    STREAMING_SFT_CONTRACT_ID,
    ReferenceTrajectorySFTSource,
    StreamingSFTError,
    StreamingSFTPlan,
    _action_digest,
    _VerifiedStreamingSFTPlan,
    build_streaming_sft_plan,
    execute_streaming_sft_backward,
)
from goalzendo_interactive.trajectory_banks import (
    ReferenceTrajectoryRecord,
    generate_reference_trajectory_bank,
)
from goalzendo_interactive.trajectory_encoding import IGNORE_INDEX, encode_sft_dialogue

_MODEL_ID = "test/streaming-sft-tiny"
_REVISION = "a" * 40
_MAXIMUM_SEQUENCE_TOKENS = 65_536


class ExactCharacterChatTokenizer:
    def apply_chat_template(
        self,
        conversation: Sequence[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        text = "".join(f"<|{message['role']}|>\n{message['content']}<|end|>\n" for message in conversation)
        return text + ("<|assistant|>\n" if add_generation_prompt else "")

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


class TinyConfig:
    def __init__(self) -> None:
        self.vocab_size = 256
        self.hidden_size = 6

    def to_dict(self) -> dict[str, object]:
        return {
            "architectures": ["StreamingTinyCausalLM"],
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
        }


@dataclass
class TinyOutput:
    logits: torch.Tensor


class StreamingTinyCausalLM(torch.nn.Module):
    def __init__(self, *, seed: int = 2026) -> None:
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

    def forward(self, *, input_ids: torch.Tensor) -> TinyOutput:
        hidden = self.embedding(input_ids).cumsum(dim=1)
        return TinyOutput(self.projection(hidden))


class NonfiniteStreamingLM(StreamingTinyCausalLM):
    def forward(self, *, input_ids: torch.Tensor) -> TinyOutput:
        logits = super().forward(input_ids=input_ids).logits.clone()
        logits[..., 0] = torch.inf
        return TinyOutput(logits)


class MutatingStreamingLM(StreamingTinyCausalLM):
    def __init__(self) -> None:
        super().__init__()
        self.mutation_counter: torch.Tensor
        self.register_buffer("mutation_counter", torch.tensor(0.0, dtype=torch.float64))

    def forward(self, *, input_ids: torch.Tensor) -> TinyOutput:
        with torch.no_grad():
            self.mutation_counter.add_(1.0)
        return super().forward(input_ids=input_ids)


class UnusedTrainableStreamingLM(StreamingTinyCausalLM):
    def __init__(self) -> None:
        super().__init__()
        self.unused = torch.nn.Parameter(torch.ones(3))


def _tokenizer_manifest() -> TokenizerBindingManifest:
    return TokenizerBindingManifest(
        repository_id="test/exact-character-tokenizer",
        revision="b" * 40,
        tokenizer_json_sha256="c" * 64,
        tokenizer_config_sha256="d" * 64,
        chat_template_sha256="e" * 64,
        backend_name="test-character-tokenizer",
        backend_version="1.0.0",
        vocabulary_size=256,
        special_token_ids=(
            ("bos", None),
            ("eos", None),
            ("pad", None),
            ("unk", None),
        ),
    )


def _compiler(tokenizer: ExactCharacterChatTokenizer) -> FragmentActionTokenCompiler:
    compiler = FragmentActionTokenCompiler(
        tokenizer,
        tokenizer_manifest=_tokenizer_manifest(),
        maximum_action_tokens=2_048,
    )
    compiler.freeze_registered_language()
    return compiler


def _write_artifacts(parent: Path, name: str) -> Path:
    root = parent / name
    (root / "weights").mkdir(parents=True)
    (root / "config.json").write_text(
        json.dumps({"architectures": ["StreamingTinyCausalLM"], "vocab_size": 256}),
        encoding="utf-8",
    )
    (root / "weights" / "model.bin").write_bytes(bytes(range(96)))
    return root


def _provider(
    tmp_path: Path,
    name: str,
    *,
    model: StreamingTinyCausalLM | None = None,
) -> tuple[AuthenticatedCausalLMProvider, StreamingTinyCausalLM]:
    selected = StreamingTinyCausalLM().to(dtype=torch.float64) if model is None else model
    selected.eval()
    root = _write_artifacts(tmp_path, name)
    provider = AuthenticatedCausalLMProvider(
        selected,
        root,
        model_identifier=_MODEL_ID,
        revision=_REVISION,
    )
    return provider, selected


def _reference_sources(count: int = 1) -> tuple[ReferenceTrajectorySFTSource, ...]:
    episode_bank = generate_episode_bank(small_fixture_bank_spec())
    trajectory_bank = generate_reference_trajectory_bank(episode_bank)
    return tuple(
        ReferenceTrajectorySFTSource(episode, record)
        for episode, record in zip(
            episode_bank.episodes[:count],
            trajectory_bank.records[:count],
            strict=True,
        )
    )


def _build_plan(
    sources: Sequence[ReferenceTrajectorySFTSource],
    tokenizer: ExactCharacterChatTokenizer,
    compiler: FragmentActionTokenCompiler,
    provider: AuthenticatedCausalLMProvider,
) -> _VerifiedStreamingSFTPlan:
    return build_streaming_sft_plan(
        sources,
        tokenizer,
        compiler,
        provider,
        maximum_sequence_tokens=_MAXIMUM_SEQUENCE_TOKENS,
    )


def _contains_tensor(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if is_dataclass(value) and not isinstance(value, type):
        return any(_contains_tensor(getattr(value, field.name)) for field in fields(value))
    if isinstance(value, dict):
        return any(_contains_tensor(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_tensor(item) for item in value)
    return False


def _dense_reference_backward(
    plan: StreamingSFTPlan,
    tokenizer: ExactCharacterChatTokenizer,
    compiler: FragmentActionTokenCompiler,
    provider: AuthenticatedCausalLMProvider,
) -> float:
    losses: list[torch.Tensor] = []
    for item in plan.items:
        action = parse_action(item.raw_action)
        example = encode_decision_example(
            tokenizer,
            compiler,
            item.dialogue,
            action,
            tokenizer_binding_digest=item.tokenizer_binding_digest,
            maximum_sequence_tokens=item.maximum_sequence_tokens,
        )
        verified = verify_decision_example(example, tokenizer, compiler, item.dialogue)
        logits = provider.full_forward_logits(verified.example.input_ids)
        labels = torch.tensor(example.labels, dtype=torch.long, device=logits.device)
        targets = labels[1:]
        mask = targets.ne(IGNORE_INDEX)
        losses.append(F.cross_entropy(logits[:-1][mask], targets[mask], reduction="sum"))
    dense = torch.stack(losses).sum() / plan.total_supervised_tokens
    objective = float(dense.detach().to(device="cpu", dtype=torch.float64).item())
    torch.autograd.backward(dense)
    return objective


def test_plan_is_canonical_graph_free_bound_and_input_order_independent(tmp_path: Path) -> None:
    tokenizer = ExactCharacterChatTokenizer()
    compiler = _compiler(tokenizer)
    provider, _ = _provider(tmp_path, "canonical")
    sources = _reference_sources(count=2)
    forward_verified = _build_plan(sources, tokenizer, compiler, provider)
    reverse_verified = _build_plan(tuple(reversed(sources)), tokenizer, compiler, provider)
    forward = forward_verified.plan
    reverse = reverse_verified.plan

    assert forward == reverse
    assert forward_verified.verification_digest == reverse_verified.verification_digest
    assert forward.digest == reverse.digest
    assert forward.to_json() == reverse.to_json()
    assert StreamingSFTPlan.from_json(forward.to_json()) == forward
    assert forward.contract_id == STREAMING_SFT_CONTRACT_ID
    assert forward.model_provenance_digest == provider.model_provenance_digest
    assert forward.provider_policy_state_digest == provider.policy_state_digest
    assert len(forward.items) == sum(source.reference_trajectory.assistant_action_count for source in sources)
    assert forward.total_supervised_tokens == sum(item.supervised_token_count for item in forward.items)
    assert not _contains_tensor(forward)
    assert json.loads(forward.to_json())["authorizes_execution"] is False
    assert STREAMING_SFT_AUTHORIZES_EXECUTION is False

    with pytest.raises(StreamingSFTError, match="canonical order"):
        replace(forward, items=tuple(reversed(forward.items)))


def test_sequential_backward_matches_dense_objective_and_every_parameter_gradient(
    tmp_path: Path,
) -> None:
    tokenizer = ExactCharacterChatTokenizer()
    compiler = _compiler(tokenizer)
    streaming_provider, streaming_model = _provider(tmp_path, "streaming")
    dense_provider, dense_model = _provider(tmp_path, "dense")
    sources = _reference_sources()
    verified_plan = _build_plan(sources, tokenizer, compiler, streaming_provider)
    plan = verified_plan.plan
    assert dense_provider.policy_state_digest == plan.provider_policy_state_digest

    weights_before = {
        name: parameter.detach().clone() for name, parameter in streaming_model.named_parameters()
    }
    live_parameters = tuple(streaming_model.parameters())
    released: list[tuple[str, bool]] = []

    def release_hook(item_digest: str, reference: Any) -> None:
        assert all(parameter.grad is None for parameter in live_parameters)
        released.append((item_digest, reference() is None))

    execution = execute_streaming_sft_backward(
        verified_plan,
        sources,
        tokenizer,
        compiler,
        streaming_provider,
        graph_release_hook=release_hook,
    )
    dense_objective = _dense_reference_backward(plan, tokenizer, compiler, dense_provider)

    assert math_isclose(float.fromhex(execution.objective_value_hex), dense_objective)
    streaming_parameters = dict(streaming_model.named_parameters(remove_duplicate=False))
    dense_parameters = dict(dense_model.named_parameters(remove_duplicate=False))
    assert set(streaming_parameters) == set(dense_parameters)
    for name in sorted(streaming_parameters):
        streaming_gradient = streaming_parameters[name].grad
        dense_gradient = dense_parameters[name].grad
        assert streaming_gradient is not None, name
        assert dense_gradient is not None, name
        assert torch.allclose(streaming_gradient, dense_gradient, atol=2e-6, rtol=2e-5), name
    assert [digest for digest, _ in released] == [item.digest for item in plan.items]
    assert all(was_released for _, was_released in released)
    assert execution.released_graph_count == len(plan.items)
    assert execution.parameter_count == len(tuple(streaming_model.parameters()))
    assert execution.registered_parameter_name_count == len(streaming_parameters)
    assert (
        execution.trainable_parameter_registry_digest
        == streaming_provider.trainable_parameter_registry.manifest.digest
    )
    assert execution.gradient_accumulation_dtype == "torch.float32"
    assert execution.gradient_accumulation_order == "canonical_plan_item_then_parameter_name"
    assert execution.final_gradient_cast_count == execution.parameter_count
    assert execution.fp32_contribution_cast_count >= execution.parameter_count
    assert len(execution.fp32_gradient_buffers) == len(execution.gradients) == execution.parameter_count
    assert all(record.gradient_dtype == "torch.float32" for record in execution.fp32_gradient_buffers)
    assert execution.fp32_gradient_buffer_manifest_digest == json_digest(
        [record.as_obj() for record in execution.fp32_gradient_buffers],
        domain=streaming_sft._FP32_BUFFER_MANIFEST_DOMAIN,
    )
    assert execution.gradient_manifest_digest == json_digest(
        [record.as_obj() for record in execution.gradients],
        domain=streaming_sft._GRADIENT_MANIFEST_DOMAIN,
    )
    assert execution.optimizer_step_performed is False
    assert execution.authorizes_execution is False
    assert "optimizer" not in inspect.signature(execute_streaming_sft_backward).parameters
    for name, parameter in streaming_model.named_parameters():
        assert torch.equal(parameter.detach(), weights_before[name])
    assert streaming_provider.reauthenticate_policy_state().digest == plan.model_provenance_digest


def math_isclose(first: float, second: float) -> bool:
    return abs(first - second) <= 1e-12 * max(1.0, abs(first), abs(second))


def test_plan_and_source_tampering_fail_closed(tmp_path: Path) -> None:
    tokenizer = ExactCharacterChatTokenizer()
    compiler = _compiler(tokenizer)
    provider, _ = _provider(tmp_path, "tamper")
    sources = _reference_sources()
    verified_plan = _build_plan(sources, tokenizer, compiler, provider)
    plan = verified_plan.plan

    changed_dialogue: Dialogue = (
        DialogueMessage("system", "Play hidden-law Zendo.", "contract"),
        DialogueMessage("user", "A forged decision prompt.", "opening"),
    )
    changed_item = replace(plan.items[0], dialogue=changed_dialogue)
    changed_plan = replace(plan, items=(changed_item, *plan.items[1:]))
    changed_verified = _VerifiedStreamingSFTPlan._from_derived(changed_plan)
    with pytest.raises(StreamingSFTError, match="fresh authenticated reference sources"):
        execute_streaming_sft_backward(
            changed_verified,
            sources,
            tokenizer,
            compiler,
            provider,
        )

    with pytest.raises(StreamingSFTError, match="freshly nominal verified plan"):
        execute_streaming_sft_backward(
            cast(_VerifiedStreamingSFTPlan, plan),
            sources,
            tokenizer,
            compiler,
            provider,
        )

    first = plan.items[0]
    forged_action = ReadyAction()
    forged_example = encode_decision_example(
        tokenizer,
        compiler,
        first.dialogue,
        forged_action,
        tokenizer_binding_digest=first.tokenizer_binding_digest,
        maximum_sequence_tokens=first.maximum_sequence_tokens,
    )
    forged_verified_decision = verify_decision_example(
        forged_example,
        tokenizer,
        compiler,
        first.dialogue,
    )
    forged_raw_action = serialize_action(forged_action)
    forged_item = replace(
        first,
        raw_action=forged_raw_action,
        action_digest=_action_digest(forged_raw_action),
        action_trace_digest=forged_example.action_trace.digest,
        decision_example_digest=forged_example.digest,
        decision_verification_digest=forged_verified_decision.verification_digest,
        decision_example_json=forged_example.to_json(),
        supervised_token_count=forged_example.action_token_count,
    )
    forged_plan = replace(
        plan,
        items=(forged_item, *plan.items[1:]),
        total_supervised_tokens=(
            plan.total_supervised_tokens - first.supervised_token_count + forged_item.supervised_token_count
        ),
    )
    assert StreamingSFTPlan.from_json(forged_plan.to_json()) == forged_plan
    rehashed_wrong_teacher = _VerifiedStreamingSFTPlan._from_derived(forged_plan)
    with pytest.raises(StreamingSFTError, match="fresh authenticated reference sources"):
        execute_streaming_sft_backward(
            rehashed_wrong_teacher,
            sources,
            tokenizer,
            compiler,
            provider,
        )

    other_source = _reference_sources(count=2)[1]
    with pytest.raises(StreamingSFTError, match="fresh authenticated reference sources"):
        execute_streaming_sft_backward(
            verified_plan,
            (other_source,),
            tokenizer,
            compiler,
            provider,
        )

    raw = json.loads(plan.to_json())
    raw["provider_policy_state_digest"] = "0" * 64
    tampered_json = json.dumps(raw, ensure_ascii=True, separators=(",", ":"))
    with pytest.raises(StreamingSFTError, match="digest check failed"):
        StreamingSFTPlan.from_json(tampered_json)

    wrong_record = replace(
        sources[0].reference_trajectory,
        dialogue_manifest_digest="0" * 64,
    )
    wrong_source = ReferenceTrajectorySFTSource(sources[0].episode, wrong_record)
    with pytest.raises(StreamingSFTError, match="regenerated reference policy record"):
        _build_plan((wrong_source,), tokenizer, compiler, provider)


def test_legacy_whole_dialogue_encoding_cannot_enter_plan(tmp_path: Path) -> None:
    tokenizer = ExactCharacterChatTokenizer()
    compiler = _compiler(tokenizer)
    provider, _ = _provider(tmp_path, "legacy")
    source = _reference_sources()[0]
    full_dialogue = render_dialogue(source.episode, source.reference_trajectory.transcript)
    legacy = encode_sft_dialogue(
        tokenizer,
        full_dialogue,
        maximum_tokens=_MAXIMUM_SEQUENCE_TOKENS,
    )
    with pytest.raises(TypeError, match="ReferenceTrajectoryRecord"):
        ReferenceTrajectorySFTSource(
            source.episode,
            cast(ReferenceTrajectoryRecord, legacy),
        )
    with pytest.raises(TypeError, match="ReferenceTrajectorySFTSource"):
        _build_plan(
            cast(Sequence[ReferenceTrajectorySFTSource], (legacy,)),
            tokenizer,
            compiler,
            provider,
        )
    assert provider.policy_state_digest


def test_model_mutation_before_or_during_execution_fails_without_gradients(tmp_path: Path) -> None:
    tokenizer = ExactCharacterChatTokenizer()
    compiler = _compiler(tokenizer)

    provider, model = _provider(tmp_path, "pre-mutation")
    sources = _reference_sources()
    plan = _build_plan(sources, tokenizer, compiler, provider)
    with torch.no_grad():
        model.embedding.weight.add_(0.01)
    with pytest.raises((StreamingSFTError, AuthenticatedModelProviderError)):
        execute_streaming_sft_backward(plan, sources, tokenizer, compiler, provider)
    assert all(parameter.grad is None for parameter in model.parameters())

    mutating_input = MutatingStreamingLM().to(dtype=torch.float64)
    mutating_provider, returned_mutating_model = _provider(
        tmp_path,
        "during-mutation",
        model=mutating_input,
    )
    mutating_model = cast(MutatingStreamingLM, returned_mutating_model)
    mutating_plan = _build_plan(sources, tokenizer, compiler, mutating_provider)
    with pytest.raises(StreamingSFTError, match="failed closed"):
        execute_streaming_sft_backward(
            mutating_plan,
            sources,
            tokenizer,
            compiler,
            mutating_provider,
        )
    assert all(parameter.grad is None for parameter in mutating_model.parameters())


def test_empty_nonfinite_and_any_existing_gradient_inputs_fail_closed(tmp_path: Path) -> None:
    tokenizer = ExactCharacterChatTokenizer()
    compiler = _compiler(tokenizer)
    provider, model = _provider(tmp_path, "invalid-inputs")
    with pytest.raises(StreamingSFTError, match="at least one"):
        _build_plan((), tokenizer, compiler, provider)

    sources = _reference_sources()
    plan = _build_plan(sources, tokenizer, compiler, provider)
    starting_gradient = torch.zeros_like(model.embedding.weight)
    model.embedding.weight.grad = starting_gradient
    with pytest.raises(StreamingSFTError, match="strictly absent"):
        execute_streaming_sft_backward(plan, sources, tokenizer, compiler, provider)
    assert model.embedding.weight.grad is starting_gradient
    assert torch.count_nonzero(starting_gradient).item() == 0
    model.zero_grad(set_to_none=True)

    nonfinite_input = NonfiniteStreamingLM().to(dtype=torch.float64)
    nonfinite_provider, returned_nonfinite_model = _provider(
        tmp_path,
        "nonfinite-logits",
        model=nonfinite_input,
    )
    nonfinite_model = cast(NonfiniteStreamingLM, returned_nonfinite_model)
    nonfinite_plan = _build_plan(sources, tokenizer, compiler, nonfinite_provider)
    with pytest.raises(StreamingSFTError, match="failed closed"):
        execute_streaming_sft_backward(
            nonfinite_plan,
            sources,
            tokenizer,
            compiler,
            nonfinite_provider,
        )
    assert all(parameter.grad is None for parameter in nonfinite_model.parameters())


def test_nonfinite_accumulated_gradient_is_rejected_and_cleared(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = ExactCharacterChatTokenizer()
    compiler = _compiler(tokenizer)
    provider, model = _provider(tmp_path, "nonfinite-gradient")
    sources = _reference_sources()
    plan = _build_plan(sources, tokenizer, compiler, provider)
    original_accumulate = streaming_sft._FP32GradientAccumulator.accumulate

    def inject_nonfinite_buffer(
        accumulator: streaming_sft._FP32GradientAccumulator,
        loss: torch.Tensor,
    ) -> tuple[frozenset[int], int]:
        result = original_accumulate(accumulator, loss)
        next(iter(accumulator.buffers.values())).fill_(torch.inf)
        return result

    monkeypatch.setattr(
        streaming_sft._FP32GradientAccumulator,
        "accumulate",
        inject_nonfinite_buffer,
    )
    with pytest.raises(StreamingSFTError, match="non-finite"):
        execute_streaming_sft_backward(plan, sources, tokenizer, compiler, provider)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_unused_trainable_parameter_is_rejected_without_partial_gradients(
    tmp_path: Path,
) -> None:
    tokenizer = ExactCharacterChatTokenizer()
    compiler = _compiler(tokenizer)
    model_input = UnusedTrainableStreamingLM().to(dtype=torch.float64)
    provider, returned_model = _provider(
        tmp_path,
        "unused-trainable",
        model=model_input,
    )
    model = cast(UnusedTrainableStreamingLM, returned_model)
    sources = _reference_sources()
    plan = _build_plan(sources, tokenizer, compiler, provider)
    with pytest.raises(StreamingSFTError, match="unused"):
        execute_streaming_sft_backward(plan, sources, tokenizer, compiler, provider)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_bf16_contributions_accumulate_in_fp32_once_with_canonical_tie_dedup() -> None:
    module = torch.nn.Module()
    parameter = torch.nn.Parameter(torch.zeros((), dtype=torch.bfloat16))
    module.register_parameter("z_alias", parameter)
    module.register_parameter("a_alias", parameter)
    parameters = streaming_sft._named_parameters(module)
    accumulator = streaming_sft._FP32GradientAccumulator(parameters)
    small = torch.tensor(0.001, dtype=torch.bfloat16)
    old_sequential_bf16 = torch.zeros((), dtype=torch.bfloat16)
    for _ in range(128):
        reached, cast_count = accumulator.accumulate(parameter * small)
        assert reached == {id(parameter)}
        assert cast_count == 1
        assert parameter.grad is None
        old_sequential_bf16.add_(small)

    buffers, finals = accumulator.finalize()
    expected_fp32 = small.to(dtype=torch.float32) * 128
    expected_final = expected_fp32.to(dtype=torch.bfloat16)
    assert len(parameters) == len(buffers) == len(finals) == 1
    assert parameters[0].aliases == ("a_alias", "z_alias")
    assert accumulator.contribution_cast_count == 128
    assert accumulator.final_cast_count == 1
    assert parameter.grad is not None
    assert torch.equal(parameter.grad, expected_final)
    assert not torch.equal(old_sequential_bf16, expected_final)
    assert buffers[0].gradient_dtype == "torch.float32"
    assert finals[0].gradient_dtype == "torch.bfloat16"
    assert (
        buffers[0].sha256
        == hashlib.sha256(
            expected_fp32.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        ).hexdigest()
    )
    assert (
        finals[0].sha256
        == hashlib.sha256(
            expected_final.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        ).hexdigest()
    )
    parameter.grad = None
    accumulator.clear_buffers()


class _FatalSFTGradientManifestSignal(BaseException):
    pass


def test_base_exception_after_sft_final_commit_is_preserved_and_cleans_gradients(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tokenizer = ExactCharacterChatTokenizer()
    compiler = _compiler(tokenizer)
    provider, model = _provider(tmp_path, "base-exception-cleanup")
    sources = _reference_sources()
    plan = _build_plan(sources, tokenizer, compiler, provider)
    original = streaming_sft._gradient_records
    signal = _FatalSFTGradientManifestSignal()
    calls = 0

    def interrupt_second_manifest(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise signal
        return original(*args, **kwargs)

    monkeypatch.setattr(streaming_sft, "_gradient_records", interrupt_second_manifest)
    with pytest.raises(_FatalSFTGradientManifestSignal) as caught:
        execute_streaming_sft_backward(plan, sources, tokenizer, compiler, provider)
    assert caught.value is signal
    assert calls == 2
    assert all(parameter.grad is None for parameter in model.parameters())
