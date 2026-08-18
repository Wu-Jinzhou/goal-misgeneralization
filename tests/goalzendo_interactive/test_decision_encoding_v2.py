from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from goalzendo_interactive._json import json_digest, load_json
from goalzendo_interactive.action_tokenization_v2 import (
    FragmentActionTokenCompiler,
    TokenizerBindingManifest,
)
from goalzendo_interactive.actions import ReadyAction, TestAction
from goalzendo_interactive.decision_encoding_v2 import (
    DECISION_ENCODING_CONTRACT_ID,
    DECISION_ENCODING_SCHEMA_VERSION,
    DecisionEncodingError,
    DecisionOverlengthError,
    DetachedMaskedActionStatistics,
    VerifiedDecisionTokenExample,
    compare_detached_masked_statistics,
    decision_encoding_manifest,
    encode_decision_example,
    masked_action_statistics,
    verified_masked_action_statistics,
    verify_decision_example,
)
from goalzendo_interactive.dialogue import Dialogue, DialogueMessage
from goalzendo_interactive.objectives import trajectory_sft_objective
from goalzendo_interactive.schema import scene_at
from goalzendo_interactive.trajectory_encoding import IGNORE_INDEX


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


def _dialogue(label: str = "opening") -> Dialogue:
    return (
        DialogueMessage("system", "Play hidden-law Zendo.", "contract"),
        DialogueMessage("user", f"Choose a canonical action for {label}.", "opening"),
    )


def _tokenizer_manifest() -> TokenizerBindingManifest:
    return TokenizerBindingManifest(
        repository_id="test/exact-character-tokenizer",
        revision="1" * 40,
        tokenizer_json_sha256="2" * 64,
        tokenizer_config_sha256="3" * 64,
        chat_template_sha256="4" * 64,
        backend_name="test-character-tokenizer",
        backend_version="1.0.0",
        vocabulary_size=256,
        special_token_ids=(("bos", None), ("eos", None), ("pad", None), ("unk", None)),
    )


def _compiler(tokenizer: ExactCharacterChatTokenizer) -> FragmentActionTokenCompiler:
    compiler = FragmentActionTokenCompiler(
        tokenizer,
        tokenizer_manifest=_tokenizer_manifest(),
        maximum_action_tokens=2_048,
    )
    compiler.freeze_registered_language()
    return compiler


def test_one_decision_example_uses_fragment_targets_and_masks_only_prompt() -> None:
    tokenizer = ExactCharacterChatTokenizer()
    compiler = _compiler(tokenizer)
    example = encode_decision_example(
        tokenizer,
        compiler,
        _dialogue(),
        TestAction(scene_at(31)),
        tokenizer_binding_digest=_tokenizer_manifest().digest,
        maximum_sequence_tokens=4_096,
    )

    assert example.schema_version == DECISION_ENCODING_SCHEMA_VERSION == 1
    assert example.contract_id == DECISION_ENCODING_CONTRACT_ID
    assert example.input_ids == example.prompt_token_ids + example.action_trace.action_token_ids
    assert example.labels[: example.action_start] == (IGNORE_INDEX,) * example.action_start
    assert example.labels[example.action_start :] == example.action_trace.sft_labels
    assert example.action_token_count == len(example.action_trace.steps)
    assert len(example.digest) == 64
    verified = verify_decision_example(example, tokenizer, compiler, _dialogue())
    assert type(verified) is VerifiedDecisionTokenExample
    assert verified.example is example
    assert len(verified.verification_digest) == 64


def test_decision_sft_is_differentiable_at_exact_action_positions() -> None:
    tokenizer = ExactCharacterChatTokenizer()
    compiler = _compiler(tokenizer)
    example = encode_decision_example(
        tokenizer,
        compiler,
        _dialogue(),
        ReadyAction(),
        tokenizer_binding_digest=_tokenizer_manifest().digest,
        maximum_sequence_tokens=4_096,
    )
    vocabulary = 256
    logits = torch.zeros(
        (1, len(example.input_ids), vocabulary),
        dtype=torch.float64,
        requires_grad=True,
    )
    labels = torch.tensor((example.labels,), dtype=torch.long)
    result = trajectory_sft_objective(logits, labels)
    assert result.supervised_token_count == example.action_token_count
    result.loss.backward()
    assert logits.grad is not None
    first_prediction = example.action_start - 1
    assert torch.count_nonzero(logits.grad[0, first_prediction]).item() > 0
    assert torch.count_nonzero(logits.grad[0, 0]).item() == 0


def test_sparse_masked_replay_is_differentiable_and_detached_values_verify() -> None:
    tokenizer = ExactCharacterChatTokenizer()
    compiler = _compiler(tokenizer)
    dialogue = _dialogue()
    example = encode_decision_example(
        tokenizer,
        compiler,
        dialogue,
        TestAction(scene_at(91)),
        tokenizer_binding_digest=_tokenizer_manifest().digest,
        maximum_sequence_tokens=4_096,
    )
    logits = torch.linspace(
        -2.0,
        2.0,
        len(example.input_ids) * 256,
        dtype=torch.float64,
    ).reshape(len(example.input_ids), 256)
    logits.requires_grad_(True)

    replayed = verified_masked_action_statistics(
        logits,
        example,
        tokenizer,
        compiler,
        dialogue,
        temperature=0.7,
    )
    assert replayed.action_token_count == example.action_token_count
    assert replayed.sequence_log_probability.item() <= 1e-7
    assert replayed.mean_token_entropy.item() >= 0
    replayed.sequence_log_probability.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad).item() > 0

    verified = verify_decision_example(example, tokenizer, compiler, dialogue)

    detached = DetachedMaskedActionStatistics.from_replay(
        replayed,
        verified,
        temperature=0.7,
    )
    compare_detached_masked_statistics(
        detached,
        replayed,
        verified,
        temperature=0.7,
        absolute_tolerance=0,
    )
    assert len(detached.digest) == 64

    changed_logits = logits.detach().clone()
    first = next(step for step in example.action_trace.steps if len(step.allowed_token_ids) > 1)
    with torch.no_grad():
        prediction_index = example.action_start + first.action_token_index - 1
        changed_logits[prediction_index, first.selected_token_id] += 0.125
    changed = masked_action_statistics(changed_logits, verified, temperature=0.7)
    with pytest.raises(DecisionEncodingError, match="differ from differentiable replay"):
        compare_detached_masked_statistics(
            detached,
            changed,
            verified,
            temperature=0.7,
            absolute_tolerance=1e-12,
        )


def test_exact_regeneration_rejects_prompt_and_legal_mask_tampering() -> None:
    tokenizer = ExactCharacterChatTokenizer()
    compiler = _compiler(tokenizer)
    dialogue = _dialogue()
    example = encode_decision_example(
        tokenizer,
        compiler,
        dialogue,
        ReadyAction(),
        tokenizer_binding_digest=_tokenizer_manifest().digest,
        maximum_sequence_tokens=4_096,
    )

    with pytest.raises(DecisionEncodingError, match="exact tokenizer/compiler regeneration"):
        verify_decision_example(example, tokenizer, compiler, _dialogue("other opening"))

    first = example.action_trace.steps[0]
    extra = max(first.allowed_token_ids) + 1
    altered_first = replace(
        first,
        allowed_token_ids=tuple(sorted((*first.allowed_token_ids, extra))),
    )
    altered_trace = replace(
        example.action_trace,
        steps=(altered_first, *example.action_trace.steps[1:]),
    )
    altered_example = replace(example, action_trace=altered_trace)
    with pytest.raises(DecisionEncodingError, match="exact tokenizer/compiler regeneration"):
        verify_decision_example(altered_example, tokenizer, compiler, dialogue)


def test_decision_example_strict_json_round_trip_and_verified_boundary() -> None:
    tokenizer = ExactCharacterChatTokenizer()
    compiler = _compiler(tokenizer)
    dialogue = _dialogue()
    example = encode_decision_example(
        tokenizer,
        compiler,
        dialogue,
        ReadyAction(),
        tokenizer_binding_digest=_tokenizer_manifest().digest,
        maximum_sequence_tokens=4_096,
    )

    loaded = type(example).from_json(example.to_json())
    assert loaded == example
    assert loaded.digest == example.digest
    assert verify_decision_example(loaded, tokenizer, compiler, dialogue).example == example

    reordered_obj = dict(sorted(json.loads(example.to_json()).items()))
    assert type(example).from_obj(reordered_obj) == example
    reordered = json.dumps(
        reordered_obj,
        sort_keys=True,
        separators=(",", ":"),
    )
    assert reordered != example.to_json()
    with pytest.raises(DecisionEncodingError, match="canonical"):
        type(example).from_json(reordered)

    logits = torch.zeros((len(example.input_ids), 256))
    with pytest.raises(TypeError, match="VerifiedDecisionTokenExample"):
        masked_action_statistics(logits, example, temperature=1.0)  # type: ignore[arg-type]


def test_overlength_nonfinite_vocab_and_manifest_fail_closed() -> None:
    tokenizer = ExactCharacterChatTokenizer()
    compiler = _compiler(tokenizer)
    with pytest.raises(DecisionOverlengthError, match="truncation is forbidden"):
        encode_decision_example(
            tokenizer,
            compiler,
            _dialogue(),
            ReadyAction(),
            tokenizer_binding_digest=_tokenizer_manifest().digest,
            maximum_sequence_tokens=2,
        )

    example = encode_decision_example(
        tokenizer,
        compiler,
        _dialogue(),
        ReadyAction(),
        tokenizer_binding_digest=_tokenizer_manifest().digest,
        maximum_sequence_tokens=4_096,
    )
    logits = torch.zeros((len(example.input_ids), 256))
    logits[0, 0] = torch.nan
    verified = verify_decision_example(example, tokenizer, compiler, _dialogue())
    with pytest.raises(DecisionEncodingError, match="non-finite"):
        masked_action_statistics(logits, verified, temperature=1.0)
    with pytest.raises(DecisionEncodingError, match="out-of-vocabulary"):
        masked_action_statistics(
            torch.zeros((len(example.input_ids), 10)),
            verified,
            temperature=1.0,
        )

    manifest = decision_encoding_manifest()
    assert manifest["live_model_authorization"] is False
    assert manifest["weight_update_authorization"] is False


def test_pinned_qwen_decision_probe_fixture_is_self_authenticating() -> None:
    fixtures = (
        (
            "g03-qwen-decision-probe-v1.json",
            "8931b2eaff29260d4a930f621597bc9206785a4687ffc5d87808bfe8997a259d",
            "goalzendo-interactive-pinned-qwen-decision-probe-v1",
        ),
        (
            "g03-qwen-decision-sampler-probe-v2.json",
            "aedc42a7cbb1b29c5502730a09c722b23f79167758427e7f5fff09c61cb317c6",
            "goalzendo-interactive-pinned-qwen-decision-sampler-probe-v2",
        ),
    )
    for filename, expected_digest, domain in fixtures:
        path = Path(__file__).parent / "fixtures" / filename
        value = load_json(path.read_text(encoding="utf-8"))
        assert type(value) is dict
        supplied = value.pop("report_digest")
        assert supplied == expected_digest
        assert supplied == json_digest(value, domain=domain)
        assert value["authorization"] == {
            "live_model_loaded": False,
            "weight_update_authorized": False,
        }
