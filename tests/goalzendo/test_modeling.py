from __future__ import annotations

import json
import math
import os
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from goalzendo.modeling import (
    PINNED_QWEN35_MODELS,
    PINNED_QWEN_MODELS,
    ActionScores,
    TwoActionScorer,
    apply_update_method,
    encode_action_continuations,
    encode_action_labels,
    format_chat_prompt,
    model_provenance,
    resolve_torch_dtype,
    run_model_integration_check,
    score_action_sequences,
)


class CharacterTokenizer:
    pad_token_id = 0
    eos_token_id = 9

    def __init__(self) -> None:
        self.tokens = {character: index + 2 for index, character in enumerate("PQALONG")}

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        encoded = [self.tokens[character] for character in text]
        return [1, *encoded] if add_special_tokens else encoded


class TransitionLM(nn.Module):
    """A causal LM whose next-token logits depend only on the current token."""

    def __init__(self, vocabulary_size: int = 12) -> None:
        super().__init__()
        torch.manual_seed(7)
        self.transitions = nn.Parameter(torch.randn(vocabulary_size, vocabulary_size))
        self.calls = 0
        self.last_batch_shape: tuple[int, ...] | None = None

    def forward(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        del attention_mask
        self.calls += 1
        self.last_batch_shape = tuple(input_ids.shape)
        return SimpleNamespace(logits=self.transitions[input_ids])


def test_sequence_scoring_matches_manual_one_and_multi_token_probabilities() -> None:
    tokenizer = CharacterTokenizer()
    model = TransitionLM()
    result = score_action_sequences(model, tokenizer, ["P", "Q"], ("A", "LONG"))
    assert result.log_scores.shape == (2, 2)
    assert result.token_lengths == (1, 4)

    prompt_last = tokenizer.tokens["P"]
    action_a = tokenizer.encode("A", add_special_tokens=False)
    action_long = tokenizer.encode("LONG", add_special_tokens=False)
    transition_log_probs = model.transitions.log_softmax(dim=-1)
    expected_a = transition_log_probs[prompt_last, action_a[0]]
    preceding = [prompt_last, *action_long[:-1]]
    expected_long = sum(
        transition_log_probs[previous, target]
        for previous, target in zip(preceding, action_long, strict=True)
    )
    assert torch.allclose(result.log_scores[0], torch.stack((expected_a, expected_long)))


def test_sequence_scores_retain_gradient_and_module_wrapper() -> None:
    tokenizer = CharacterTokenizer()
    model = TransitionLM()
    scorer = TwoActionScorer(model, tokenizer, ("A", "LONG"))
    scores = scorer(["P", "Q"])
    scores.sum().backward()
    assert model.transitions.grad is not None
    assert torch.count_nonzero(model.transitions.grad) > 0


def test_single_token_actions_share_one_prompt_forward_with_right_padding() -> None:
    tokenizer = CharacterTokenizer()
    model = TransitionLM()
    result = score_action_sequences(model, tokenizer, ["P", "PQ"], ("A", "L"))
    expected = torch.stack(
        (
            model.transitions[tokenizer.tokens["P"], [tokenizer.tokens["A"], tokenizer.tokens["L"]]],
            model.transitions[tokenizer.tokens["Q"], [tokenizer.tokens["A"], tokenizer.tokens["L"]]],
        )
    )
    assert torch.allclose(result.log_scores, expected)
    assert model.calls == 1
    assert model.last_batch_shape == (2, 3)


class LastLogitTransitionLM(TransitionLM):
    def __init__(self) -> None:
        super().__init__()
        self.received_logits_to_keep: int | None = None
        self.received_position_ids: torch.Tensor | None = None

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        logits_to_keep: int = 0,
    ) -> SimpleNamespace:
        del attention_mask
        self.calls += 1
        self.last_batch_shape = tuple(input_ids.shape)
        self.received_logits_to_keep = logits_to_keep
        self.received_position_ids = position_ids.detach().clone()
        logits = self.transitions[input_ids]
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:, :]
        return SimpleNamespace(logits=logits)


def test_single_token_fast_path_uses_left_padding_and_last_logit_kwarg() -> None:
    tokenizer = CharacterTokenizer()
    model = LastLogitTransitionLM()
    result = score_action_sequences(model, tokenizer, ["P", "PQ"], ("A", "L"))
    expected = torch.stack(
        (
            model.transitions[tokenizer.tokens["P"], [tokenizer.tokens["A"], tokenizer.tokens["L"]]],
            model.transitions[tokenizer.tokens["Q"], [tokenizer.tokens["A"], tokenizer.tokens["L"]]],
        )
    )
    assert torch.allclose(result.log_scores, expected)
    assert model.received_logits_to_keep == 1
    assert model.received_position_ids is not None
    assert torch.equal(model.received_position_ids, torch.tensor([[0, 0, 1], [0, 1, 2]]))


def test_action_labels_must_be_distinct_and_nonempty() -> None:
    tokenizer = CharacterTokenizer()
    with pytest.raises(ValueError, match="distinct"):
        encode_action_labels(tokenizer, ("A", "A"))
    with pytest.raises(ValueError, match="at least one token"):
        encode_action_labels(tokenizer, ("A", ""))
    with pytest.raises(ValueError, match="prefix-free"):
        encode_action_labels(tokenizer, ("A", "AL"))


class BoundaryMergingTokenizer(CharacterTokenizer):
    """Simulate a BPE merge that consumes the final prompt token."""

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        if text == "P":
            return [1, 2] if add_special_tokens else [2]
        if text == "PA":
            return [1, 10] if add_special_tokens else [10]
        if text == "PB":
            return [1, 11] if add_special_tokens else [11]
        return super().encode(text, add_special_tokens=add_special_tokens)


def test_contextual_encoding_fails_closed_on_boundary_merge() -> None:
    with pytest.raises(ValueError, match="not prefix-stable"):
        encode_action_continuations(BoundaryMergingTokenizer(), ["P"], ("A", "B"))


class ContextDependentTokenizer:
    pad_token_id = 0
    eos_token_id = 9

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        del add_special_tokens
        encodings = {
            "P": [1, 2],
            "PA": [1, 2, 4],
            "PB": [1, 2, 5],
            "Q": [1, 3],
            "QA": [1, 3, 6],
            "QB": [1, 3, 7],
            "A": [4],
            "B": [5],
        }
        return list(encodings[text])


def test_single_token_fast_path_uses_contextual_ids_for_each_prompt() -> None:
    tokenizer = ContextDependentTokenizer()
    model = TransitionLM()
    result = score_action_sequences(model, tokenizer, ["P", "Q"], ("A", "B"))
    expected = torch.stack(
        (
            model.transitions[2, [4, 5]],
            model.transitions[3, [6, 7]],
        )
    )
    assert torch.allclose(result.log_scores, expected)
    assert result.continuation_token_ids == (((4,), (5,)), ((6,), (7,)))


class NonFiniteLM(TransitionLM):
    def forward(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        result = super().forward(input_ids=input_ids, attention_mask=attention_mask)
        logits = result.logits.clone()
        logits[0, -1, 0] = torch.nan
        return SimpleNamespace(logits=logits)


def test_action_scoring_rejects_non_finite_model_outputs() -> None:
    with pytest.raises(FloatingPointError, match="model logits"):
        score_action_sequences(NonFiniteLM(), CharacterTokenizer(), ["P"], ("A", "L"))
    with pytest.raises(FloatingPointError, match="action scores"):
        ActionScores(torch.tensor([[0.0, torch.inf]]), ("A", "B"), (1, 1))


def test_dtype_resolution_and_non_lora_update_methods() -> None:
    assert resolve_torch_dtype("bf16") is torch.bfloat16
    assert resolve_torch_dtype("torch.float32") is torch.float32
    with pytest.raises(ValueError, match="unsupported"):
        resolve_torch_dtype("float8")

    full_model = nn.Linear(2, 2)
    full_model.requires_grad_(False)
    assert apply_update_method(full_model, {"method": "full"}) is full_model
    assert all(parameter.requires_grad for parameter in full_model.parameters())

    frozen_model = nn.Linear(2, 2)
    assert apply_update_method(frozen_model, {"method": "frozen"}) is frozen_model
    assert not any(parameter.requires_grad for parameter in frozen_model.parameters())


class ChatTokenizer(CharacterTokenizer):
    name_or_path = "local/test-tokenizer"
    vocab_size = 12
    chat_template = "test-template-v1"

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        assert not tokenize and add_generation_prompt
        mode = "think" if enable_thinking else "direct"
        return "|".join(f"{item['role']}:{item['content']}" for item in messages) + f"|{mode}:"


def test_chat_formatting_and_provenance_record_revision_and_action_tokens() -> None:
    tokenizer = ChatTokenizer()
    prompt = format_chat_prompt(
        tokenizer,
        "choose a koan",
        system_prompt="follow the Law",
        enable_thinking=False,
    )
    assert prompt == "system:follow the Law|user:choose a koan|direct:"

    model = TransitionLM()
    model.config = SimpleNamespace(_commit_hash="resolved-commit")
    provenance = model_provenance(
        model,
        tokenizer,
        {"name": "local/model", "revision": "requested-revision", "dtype": "float32"},
        ("A", "LONG"),
    )
    assert provenance["requested_revision"] == "requested-revision"
    assert provenance["resolved_revision"] == "resolved-commit"
    assert provenance["action_token_ids"] == [
        tokenizer.encode("A", add_special_tokens=False),
        tokenizer.encode("LONG", add_special_tokens=False),
    ]
    assert provenance["trainable_parameter_count"] == provenance["parameter_count"]


class AuditTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    name_or_path = "offline/audit-tokenizer"
    vocab_size = 258
    chat_template = "offline-byte-template"

    def __init__(self, revision: str) -> None:
        self.init_kwargs = {"_commit_hash": revision}

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        values = [byte + 2 for byte in text.encode("utf-8")]
        return [1, *values] if add_special_tokens else values

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        assert not tokenize and add_generation_prompt and not enable_thinking
        return "|".join(f"{item['role']}:{item['content']}" for item in messages) + "|assistant:"


class AuditLM(nn.Module):
    def __init__(self, revision: str) -> None:
        super().__init__()
        torch.manual_seed(3)
        self.transitions = nn.Parameter(torch.randn(258, 258), requires_grad=False)
        self.config = SimpleNamespace(_commit_hash=revision)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        logits_to_keep: int = 0,
    ) -> SimpleNamespace:
        del attention_mask, position_ids
        logits = self.transitions[input_ids]
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:, :]
        return SimpleNamespace(logits=logits)


def test_dependency_injected_chat_model_integration_audit() -> None:
    revision = "offline-revision"

    def loader(
        model_config: object,
        update_config: object,
    ) -> tuple[nn.Module, AuditTokenizer]:
        del model_config
        assert update_config == {"method": "frozen"}
        return AuditLM(revision), AuditTokenizer(revision)

    report = run_model_integration_check(
        {
            "name": "offline/audit-model",
            "revision": revision,
            "dtype": "float32",
        },
        prompts=("P", "Q"),
        device="cpu",
        max_prompt_tokens=256,
        model_loader=loader,
    )
    assert report["passed"]
    assert report["action_boundary"]["prefix_stable"]
    assert report["scores"]["finite"]
    assert report["scores"]["swap_invariant"]
    assert all(
        math.isfinite(value)
        for row in report["scores"]["normalized_log_scores"]
        for value in row
    )
    assert report["scores"]["maximum_normalization_error"] < 1e-6
    assert report["runtime"]["parameter_devices"] == ["cpu"]
    assert report["runtime"]["parameter_dtypes"] == ["torch.float32"]
    assert report["prompt_tokens"]["maximum"] <= 256
    assert report["model"]["resolved_revision"] == revision
    assert json.loads(json.dumps(report))["passed"] is True


@pytest.mark.parametrize("spec", PINNED_QWEN_MODELS, ids=lambda spec: spec.name.rsplit("/", 1)[-1])
@pytest.mark.skipif(
    os.environ.get("GOALZENDO_RUN_QWEN_INTEGRATION") != "1",
    reason="set GOALZENDO_RUN_QWEN_INTEGRATION=1 for the optional network/GPU audit",
)
def test_pinned_qwen_chat_boundaries_live(spec: object) -> None:
    config = spec.as_config()  # type: ignore[attr-defined]
    report = run_model_integration_check(
        config,
        device=os.environ.get("GOALZENDO_QWEN_DEVICE", "cuda"),
        max_prompt_tokens=512,
    )
    assert report["passed"]


def test_modern_qwen_models_are_exactly_pinned() -> None:
    assert [spec.as_config() for spec in PINNED_QWEN35_MODELS] == [
        {
            "name": "Qwen/Qwen3.5-0.8B",
            "revision": "2fc06364715b967f1860aea9cf38778875588b17",
            "dtype": "bfloat16",
            "trust_remote_code": False,
        },
        {
            "name": "Qwen/Qwen3.5-2B",
            "revision": "15852e8c16360a2fea060d615a32b45270f8a8fc",
            "dtype": "bfloat16",
            "trust_remote_code": False,
        },
    ]
