from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace

import pytest

from goalzendo_interactive.action_language import GrammarOption, GrammarSegment
from goalzendo_interactive.action_tokenization_v2 import (
    FRAGMENT_ACTION_TOKENIZATION_CONTRACT_ID,
    FRAGMENT_ACTION_TOKENIZATION_SCHEMA_VERSION,
    REGISTERED_FRAGMENT_SPEC_COUNT,
    ActionTokenTrace,
    FragmentActionOverlengthError,
    FragmentActionTokenCompiler,
    FragmentActionTokenizationError,
    FragmentCompilerNotFrozenError,
    FragmentDecodeMismatchError,
    InvalidFragmentTokenError,
    OptionTokenCollisionError,
    TokenizerBindingManifest,
    UnregisteredFragmentSpecError,
    fragment_token_contract_manifest,
)
from goalzendo_interactive.actions import (
    AnswerAction,
    Classification,
    ReadyAction,
    TestAction,
)
from goalzendo_interactive.rules import SYNTACTIC_RULE_COUNT, iter_syntactic_rules
from goalzendo_interactive.schema import scene_at


class BoundaryMergingExactTokenizer:
    """A tiny BPE-like tokenizer whose whole-text encoding merges ``\"`` + ``,``."""

    _QUOTE_COMMA = 1_000

    def __init__(self) -> None:
        self.encode_calls = 0
        self.decode_calls = 0

    def apply_chat_template(
        self,
        conversation: Sequence[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        text = "".join(
            f"<turn:{message['role']}>\n{message['content']}<end>\n"
            for message in conversation
        )
        return text + ("<turn:assistant>\n" if add_generation_prompt else "")

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        self.encode_calls += 1
        result: list[int] = []
        index = 0
        while index < len(text):
            if text.startswith('\",', index):
                result.append(self._QUOTE_COMMA)
                index += 2
            else:
                result.append(ord(text[index]))
                index += 1
        return result

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        self.decode_calls += 1
        return "".join(
            '\",' if token_id == self._QUOTE_COMMA else chr(token_id)
            for token_id in token_ids
        )


class WrongDecodeTokenizer(BoundaryMergingExactTokenizer):
    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        return super().decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        ) + "!"


class InvalidIdTokenizer(BoundaryMergingExactTokenizer):
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert text
        assert add_special_tokens is False
        return [True]


class EmptyEncodingTokenizer(BoundaryMergingExactTokenizer):
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert text
        assert add_special_tokens is False
        return []


class OutOfVocabularyTokenizer(BoundaryMergingExactTokenizer):
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert text
        assert add_special_tokens is False
        return [2_048]


class ConcatenationMismatchTokenizer(BoundaryMergingExactTokenizer):
    """Fragments decode exactly, but the ready-fragment/completion pair does not."""

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        decoded = super().decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )
        if decoded == '{"move":"ready"}':
            return decoded[:-1]
        return decoded


def _compiler(
    tokenizer: BoundaryMergingExactTokenizer | None = None,
    *,
    maximum_action_tokens: int = 2_048,
    freeze: bool = True,
) -> FragmentActionTokenCompiler:
    compiler = FragmentActionTokenCompiler(
        BoundaryMergingExactTokenizer() if tokenizer is None else tokenizer,
        tokenizer_manifest=_test_tokenizer_manifest(),
        maximum_action_tokens=maximum_action_tokens,
    )
    if freeze:
        compiler.freeze_registered_language()
    return compiler


def _test_tokenizer_manifest() -> TokenizerBindingManifest:
    return TokenizerBindingManifest(
        repository_id="tests/adversarial-boundary-merging-exact",
        revision="1" * 40,
        tokenizer_json_sha256="2" * 64,
        tokenizer_config_sha256="3" * 64,
        chat_template_sha256="4" * 64,
        backend_name="tests.BoundaryMergingExactTokenizer",
        backend_version="1.0.0-test",
        vocabulary_size=2_048,
        special_token_ids=(
            ("bos", None),
            ("eos", 3),
            ("pad", 0),
            ("unk", 1),
        ),
    )


def _pinned_qwen_tokenizer_manifest() -> TokenizerBindingManifest:
    additional = (
        ("additional:<|im_start|>", 151_644),
        ("additional:<|im_end|>", 151_645),
        ("additional:<|object_ref_start|>", 151_646),
        ("additional:<|object_ref_end|>", 151_647),
        ("additional:<|box_start|>", 151_648),
        ("additional:<|box_end|>", 151_649),
        ("additional:<|quad_start|>", 151_650),
        ("additional:<|quad_end|>", 151_651),
        ("additional:<|vision_start|>", 151_652),
        ("additional:<|vision_end|>", 151_653),
        ("additional:<|vision_pad|>", 151_654),
        ("additional:<|image_pad|>", 151_655),
        ("additional:<|video_pad|>", 151_656),
    )
    return TokenizerBindingManifest(
        repository_id="Qwen/Qwen2.5-1.5B-Instruct",
        revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        tokenizer_json_sha256=(
            "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
        ),
        tokenizer_config_sha256=(
            "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583"
        ),
        chat_template_sha256=(
            "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f"
        ),
        backend_name="tokenizers.Tokenizer",
        backend_version="0.22.2",
        vocabulary_size=151_665,
        special_token_ids=tuple(
            sorted(
                (
                    *additional,
                    ("bos", None),
                    ("eos", 151_645),
                    ("pad", 151_643),
                    ("unk", None),
                )
            )
        ),
    )


def test_fragment_contract_allows_cumulative_bpe_boundary_merges() -> None:
    tokenizer = BoundaryMergingExactTokenizer()
    compiler = _compiler(tokenizer)
    trace = compiler.trace_action(TestAction(scene_at(0)))

    first, second = trace.fragments[:2]
    first_ids = tuple(tokenizer.encode(first.text, add_special_tokens=False))
    cumulative_ids = tuple(tokenizer.encode(first.text + second.text, add_special_tokens=False))
    assert cumulative_ids[: len(first_ids)] != first_ids

    assert tokenizer.decode(
        trace.action_token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ) == trace.raw_action
    assert "".join(fragment.text for fragment in trace.fragments) == trace.raw_action
    assert tuple(
        token_id for fragment in trace.fragments for token_id in fragment.token_ids
    ) == trace.action_token_ids
    for fragment in trace.fragments:
        assert tokenizer.decode(
            fragment.token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ) == fragment.text


def test_teacher_forced_ready_and_one_thousand_test_actions_have_exact_masks() -> None:
    compiler = _compiler()
    ready = compiler.trace_action(ReadyAction())
    assert ready.raw_action == '{"move":"ready"}'
    assert ready.sft_labels == tuple(step.selected_token_id for step in ready.steps)
    assert ready.allowed_token_ids == tuple(step.allowed_token_ids for step in ready.steps)

    seen: set[str] = set()
    for index in range(1_000):
        trace = compiler.trace_action(TestAction(scene_at(index)))
        seen.add(trace.raw_action)
        assert trace.sft_labels == tuple(step.selected_token_id for step in trace.steps)
        assert all(step.selected_token_id in step.allowed_token_ids for step in trace.steps)
        assert all(
            step.allowed_token_ids == tuple(sorted(set(step.allowed_token_ids)))
            for step in trace.steps
        )
        assert any(len(step.allowed_token_ids) > 1 for step in trace.steps)
    assert len(seen) == 1_000


def test_twelve_label_answer_reuses_one_rule_trie_without_cartesian_expansion() -> None:
    tokenizer = BoundaryMergingExactTokenizer()
    compiler = _compiler(tokenizer)
    rules = iter_syntactic_rules()
    first_rule = next(rules)
    second_rule = next(rules)
    labels: tuple[Classification, ...] = tuple(
        "fits" if index % 2 == 0 else "does_not_fit" for index in range(12)
    )

    first = compiler.trace_action(AnswerAction(first_rule, labels))
    second = compiler.trace_action(AnswerAction(second_rule, tuple(reversed(labels))))
    assert sum(fragment.field_id.startswith("classification.") for fragment in first.fragments) == 12
    assert sum(fragment.field_id.startswith("classification.") for fragment in second.fragments) == 12
    assert all(step.selected_token_id in step.allowed_token_ids for step in first.steps)

    runtime = compiler.runtime_counters
    counts = runtime.as_obj()["compile_counts"]
    assert isinstance(counts, dict)
    assert SYNTACTIC_RULE_COUNT == 18_760
    assert runtime.rule_trie_compile_count == 1
    assert runtime.rule_option_sequence_count == SYNTACTIC_RULE_COUNT
    assert runtime.answer_rule_label_cartesian_product_count == 0
    assert runtime.option_sequence_compile_count < SYNTACTIC_RULE_COUNT + 40
    assert tokenizer.encode_calls == runtime.option_sequence_compile_count
    assert counts["answer_rule_label_cartesian_products"] == 0
    assert runtime.cache_hit_count >= 14


def test_trace_and_compiler_manifests_are_canonical_and_digest_bound() -> None:
    compiler = _compiler()
    trace = compiler.trace_action(TestAction(scene_at(42)))
    serialized = trace.to_json()
    payload = json.loads(serialized)
    assert payload["digest"] == trace.digest
    assert len(trace.digest) == 64
    assert trace.digest == compiler.trace_action(TestAction(scene_at(42))).digest
    assert ActionTokenTrace.from_obj(payload) == trace
    assert ActionTokenTrace.from_json(serialized) == trace
    assert compiler.verified_trace_from_json(serialized) == trace

    with pytest.raises(FragmentActionTokenizationError, match="canonical byte"):
        ActionTokenTrace.from_json(serialized + "\n")
    reordered_payload = {"digest": payload["digest"]}
    reordered_payload.update({key: value for key, value in payload.items() if key != "digest"})
    reordered = json.dumps(reordered_payload, separators=(",", ":"))
    with pytest.raises(FragmentActionTokenizationError, match="canonical byte"):
        ActionTokenTrace.from_json(reordered)
    bad_digest = dict(payload)
    bad_digest["digest"] = "0" * 64
    with pytest.raises(FragmentActionTokenizationError, match="digest check failed"):
        ActionTokenTrace.from_obj(bad_digest)
    extra_field = dict(payload)
    extra_field["unexpected"] = None
    with pytest.raises(FragmentActionTokenizationError, match="noncanonical keys"):
        ActionTokenTrace.from_obj(extra_field)

    manifest = compiler.manifest
    assert manifest.digest == compiler.manifest.digest
    assert len(manifest.digest) == 64
    assert manifest.tokenizer_identifier == _test_tokenizer_manifest().digest
    assert len(manifest.registered_entries) == REGISTERED_FRAGMENT_SPEC_COUNT == 18
    assert all(len(output_digest) == 64 for _, output_digest, _ in manifest.registered_entries)
    assert trace.compiler_manifest_digest == manifest.digest
    assert all(
        fragment.tokenized_output_digest
        in {output_digest for _, output_digest, _ in manifest.registered_entries}
        for fragment in trace.fragments
    )
    contract = fragment_token_contract_manifest()
    assert contract["schema_version"] == FRAGMENT_ACTION_TOKENIZATION_SCHEMA_VERSION == 2
    assert contract["contract_id"] == FRAGMENT_ACTION_TOKENIZATION_CONTRACT_ID
    assert contract["cumulative_prefix_stability_required"] is False
    assert contract["answer_rule_label_cartesian_product_count"] == 0
    assert contract["registered_fragment_spec_count"] == 18
    assert contract["runtime_counters_in_semantic_digest"] is False
    assert contract["live_model_authorization"] is False


def test_cache_returns_same_compiled_trie_and_records_one_compile() -> None:
    compiler = _compiler(freeze=False)
    segment = GrammarSegment(
        "toy.first",
        "prefix:",
        (GrammarOption("a", "alpha"), GrammarOption("b", "beta")),
    )
    first = compiler.compile_segment(segment)
    second = compiler.compile_segment(
        GrammarSegment("toy.other-field", "prefix:", segment.options)
    )
    assert first is second
    assert compiler.runtime_counters.trie_compile_count == 1
    assert compiler.runtime_counters.cache_hit_count == 1
    with pytest.raises(FragmentCompilerNotFrozenError, match="before full registered freeze"):
        _ = compiler.manifest
    with pytest.raises(UnregisteredFragmentSpecError, match="unregistered specs"):
        compiler.freeze_registered_language()


def test_prefix_collision_decode_mismatch_invalid_ids_empty_and_overlength_fail_closed() -> None:
    compiler = _compiler(freeze=False)
    prefix_collision = GrammarSegment(
        "collision",
        "",
        (GrammarOption("short", "x"), GrammarOption("long", "xy")),
    )
    with pytest.raises(OptionTokenCollisionError, match="prefixes another"):
        compiler.compile_segment(prefix_collision)

    with pytest.raises(FragmentCompilerNotFrozenError, match="before tracing"):
        compiler.trace_action(ReadyAction())
    with pytest.raises(FragmentDecodeMismatchError, match="standalone fragment"):
        _compiler(WrongDecodeTokenizer()).trace_action(ReadyAction())
    with pytest.raises(InvalidFragmentTokenError, match="non-negative integers"):
        _compiler(InvalidIdTokenizer()).trace_action(ReadyAction())
    with pytest.raises(InvalidFragmentTokenError, match="no token ids"):
        _compiler(EmptyEncodingTokenizer()).trace_action(ReadyAction())
    with pytest.raises(InvalidFragmentTokenError, match="outside the registered"):
        _compiler(OutOfVocabularyTokenizer()).trace_action(ReadyAction())
    with pytest.raises(FragmentActionOverlengthError, match="truncation is forbidden"):
        _compiler(maximum_action_tokens=1).trace_action(ReadyAction())
    with pytest.raises(FragmentDecodeMismatchError, match="concatenated fragment"):
        _compiler(ConcatenationMismatchTokenizer()).trace_action(ReadyAction())

    frozen = _compiler()
    with pytest.raises(UnregisteredFragmentSpecError, match="unregistered fragment"):
        frozen.compile_segment(prefix_collision)


def test_semantic_manifest_is_history_independent_but_runtime_counters_change() -> None:
    compiler = _compiler()
    manifest_before = compiler.manifest
    runtime_before = compiler.runtime_counters
    for index in range(25):
        compiler.trace_action(TestAction(scene_at(index)))
    manifest_after = compiler.manifest
    runtime_after = compiler.runtime_counters

    assert manifest_after is manifest_before
    assert manifest_after.as_obj() == manifest_before.as_obj()
    assert manifest_after.digest == manifest_before.digest
    assert runtime_after.digest != runtime_before.digest
    assert runtime_after.successful_trace_count == runtime_before.successful_trace_count + 25
    assert runtime_after.cache_hit_count > runtime_before.cache_hit_count


def test_compiler_regeneration_rejects_mask_output_and_vocabulary_tampering() -> None:
    compiler = _compiler()
    trace = compiler.trace_action(TestAction(scene_at(17)))
    multiway_index = next(
        index for index, step in enumerate(trace.steps) if len(step.allowed_token_ids) > 1
    )
    step = trace.steps[multiway_index]
    extra_legal_id = max(step.allowed_token_ids) + 1
    changed_step = replace(
        step,
        allowed_token_ids=tuple(sorted((*step.allowed_token_ids, extra_legal_id))),
    )
    changed_mask = replace(
        trace,
        steps=(*trace.steps[:multiway_index], changed_step, *trace.steps[multiway_index + 1 :]),
    )
    with pytest.raises(FragmentActionTokenizationError, match="exact tokenizer/compiler"):
        compiler.verify_trace(changed_mask)

    changed_fragment = replace(trace.fragments[0], tokenized_output_digest="f" * 64)
    changed_output = replace(
        trace,
        fragments=(changed_fragment, *trace.fragments[1:]),
    )
    with pytest.raises(FragmentActionTokenizationError, match="exact tokenizer/compiler"):
        compiler.verify_trace(changed_output)

    out_of_vocab_step = replace(
        step,
        allowed_token_ids=tuple(
            sorted((*step.allowed_token_ids, _test_tokenizer_manifest().vocabulary_size))
        ),
    )
    out_of_vocab_trace = replace(
        trace,
        steps=(
            *trace.steps[:multiway_index],
            out_of_vocab_step,
            *trace.steps[multiway_index + 1 :],
        ),
    )
    with pytest.raises(InvalidFragmentTokenError, match="outside the registered"):
        compiler.verify_trace(out_of_vocab_trace)


def test_structured_pinned_qwen_manifest_is_canonical_digest_bound_and_bounded() -> None:
    manifest = _pinned_qwen_tokenizer_manifest()
    serialized = manifest.to_json()
    parsed = TokenizerBindingManifest.from_json(serialized)
    assert parsed == manifest
    assert TokenizerBindingManifest.from_obj(json.loads(serialized)) == manifest
    assert manifest.vocabulary_size == 151_665
    assert manifest.digest == "5acec12f5fb95d87c73d445f38aedc56e7b4991c78f51f243ba674206f6d0453"
    assert dict(manifest.special_token_ids)["eos"] == 151_645
    assert dict(manifest.special_token_ids)["additional:<|video_pad|>"] == 151_656

    payload = json.loads(serialized)
    reordered_payload = {"digest": payload["digest"]}
    reordered_payload.update({key: value for key, value in payload.items() if key != "digest"})
    reordered = json.dumps(reordered_payload, separators=(",", ":"))
    with pytest.raises(FragmentActionTokenizationError, match="canonical byte"):
        TokenizerBindingManifest.from_json(reordered)
    payload["digest"] = "0" * 64
    with pytest.raises(FragmentActionTokenizationError, match="digest check failed"):
        TokenizerBindingManifest.from_obj(payload)
    with pytest.raises(FragmentActionTokenizationError, match="within the registered"):
        replace(
            manifest,
            special_token_ids=tuple(
                sorted(
                    (
                        (key, manifest.vocabulary_size if key == "eos" else token_id)
                        for key, token_id in manifest.special_token_ids
                    )
                )
            ),
        )
    with pytest.raises(FragmentActionTokenizationError, match="integer id"):
        replace(
            _test_tokenizer_manifest(),
            special_token_ids=tuple(
                sorted((*_test_tokenizer_manifest().special_token_ids, ("additional:<x>", None)))
            ),
        )
