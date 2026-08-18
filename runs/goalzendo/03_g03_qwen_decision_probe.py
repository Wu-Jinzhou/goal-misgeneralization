#!/usr/bin/env python3
"""Reproduce the local pinned-Qwen G03 decision-tokenization probe.

This script loads tokenizer files already present on disk.  It never downloads
a model, samples a rollout, or authorizes a weight update.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import torch
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer  # type: ignore[import-not-found]
from tokenizers import __version__ as tokenizers_version

from goalzendo_interactive._json import dump_json, json_digest
from goalzendo_interactive.action_tokenization_v2 import (
    FragmentActionTokenCompiler,
    TokenizerBindingManifest,
)
from goalzendo_interactive.actions import (
    AnswerAction,
    ReadyAction,
    TestAction,
    parse_action,
)
from goalzendo_interactive.authenticated_sampler_v2 import (
    SamplingMode,
    replay_authenticated_sample,
    sample_authenticated_action,
)
from goalzendo_interactive.decision_encoding_v2 import (
    encode_decision_example,
    masked_action_statistics,
    verify_decision_example,
)
from goalzendo_interactive.dialogue import DialogueMessage
from goalzendo_interactive.policy_randomness_v2 import PolicyTurnSeed
from goalzendo_interactive.rules import iter_syntactic_rules
from goalzendo_interactive.schema import scene_at

REPOSITORY_ID = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
TOKENIZER_JSON_SHA256 = (
    "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
)
TOKENIZER_CONFIG_SHA256 = (
    "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583"
)
CHAT_TEMPLATE_SHA256 = (
    "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f"
)
TOKENIZER_BINDING_DIGEST = (
    "5acec12f5fb95d87c73d445f38aedc56e7b4991c78f51f243ba674206f6d0453"
)
COMPILER_MANIFEST_DIGEST = (
    "d13f5c9325c031c8f4e965ee2ca6bb51542020edd81094ee4d82bb92595eb00c"
)
VOCABULARY_SIZE = 151_665
MAXIMUM_ACTION_TOKENS = 2_048


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class _PinnedQwenTokenizer:
    def __init__(
        self,
        tokenizer_path: Path,
        chat_template: str,
    ) -> None:
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        environment = ImmutableSandboxedEnvironment(
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self._chat_template = environment.from_string(chat_template)

    def apply_chat_template(
        self,
        conversation: Sequence[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        if tokenize:
            raise ValueError("this exact-text adapter never tokenizes the template itself")
        rendered = self._chat_template.render(
            messages=list(conversation),
            tools=None,
            add_generation_prompt=add_generation_prompt,
        )
        if type(rendered) is not str or not rendered:
            raise ValueError("chat template did not render nonempty text")
        return rendered

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return cast(
            list[int],
            self._tokenizer.encode(
                text,
                add_special_tokens=add_special_tokens,
            ).ids,
        )

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        if clean_up_tokenization_spaces:
            raise ValueError("cleanup is forbidden by the exact-decode contract")
        return cast(
            str,
            self._tokenizer.decode(
                list(token_ids),
                skip_special_tokens=skip_special_tokens,
            ),
        )

    def token_to_id(self, text: str) -> int | None:
        return cast(int | None, self._tokenizer.token_to_id(text))

    @property
    def vocabulary_size(self) -> int:
        return cast(int, self._tokenizer.get_vocab_size(with_added_tokens=True))


class _DeterministicProbeProvider:
    """Prefix-pure logits used only to exercise the sampler without a model."""

    def __init__(self) -> None:
        self._policy_state_digest = hashlib.sha256(
            b"g03-pinned-qwen-tokenizer-only-probe-policy-v1"
        ).hexdigest()
        self._policy_offset = int(self._policy_state_digest[:8], 16)

    @property
    def policy_state_digest(self) -> str:
        return self._policy_state_digest

    def _row(self, input_ids: tuple[int, ...]) -> torch.Tensor:
        prefix_fingerprint = sum(
            (index + 1) * token_id
            for index, token_id in enumerate(input_ids[-32:])
        )
        offset = (
            prefix_fingerprint + 19 * len(input_ids) + self._policy_offset
        ) % 251
        token_ids = torch.arange(VOCABULARY_SIZE, dtype=torch.int64)
        integer_logits = torch.remainder(token_ids * 67 + offset * 29, 251) - 125
        return integer_logits.to(dtype=torch.float64) / 31.0

    def next_token_logits(self, input_ids: tuple[int, ...]) -> torch.Tensor:
        return self._row(input_ids)

    def full_forward_logits(self, input_ids: tuple[int, ...]) -> torch.Tensor:
        result = torch.stack(
            [
                self._row(input_ids[: position + 1])
                for position in range(len(input_ids))
            ]
        )
        result.requires_grad_(True)
        return result


def _special_token_ids() -> tuple[tuple[str, int | None], ...]:
    additional = tuple(
        (f"additional:{text}", token_id)
        for text, token_id in (
            ("<|im_start|>", 151_644),
            ("<|im_end|>", 151_645),
            ("<|object_ref_start|>", 151_646),
            ("<|object_ref_end|>", 151_647),
            ("<|box_start|>", 151_648),
            ("<|box_end|>", 151_649),
            ("<|quad_start|>", 151_650),
            ("<|quad_end|>", 151_651),
            ("<|vision_start|>", 151_652),
            ("<|vision_end|>", 151_653),
            ("<|vision_pad|>", 151_654),
            ("<|image_pad|>", 151_655),
            ("<|video_pad|>", 151_656),
        )
    )
    return tuple(
        sorted(
            (
                *additional,
                ("bos", None),
                ("eos", 151_645),
                ("pad", 151_643),
                ("unk", None),
            )
        )
    )


def _load_and_verify_files(
    tokenizer_path: Path,
    config_path: Path,
) -> tuple[dict[str, Any], str]:
    tokenizer_bytes = tokenizer_path.read_bytes()
    config_bytes = config_path.read_bytes()
    if _sha256_bytes(tokenizer_bytes) != TOKENIZER_JSON_SHA256:
        raise RuntimeError("tokenizer.json SHA-256 differs from the pinned revision")
    if _sha256_bytes(config_bytes) != TOKENIZER_CONFIG_SHA256:
        raise RuntimeError("tokenizer_config.json SHA-256 differs from the pinned revision")
    value = json.loads(config_bytes)
    if type(value) is not dict or type(value.get("chat_template")) is not str:
        raise RuntimeError("tokenizer config does not contain a chat template")
    template = value["chat_template"]
    if _sha256_bytes(template.encode("utf-8")) != CHAT_TEMPLATE_SHA256:
        raise RuntimeError("chat-template SHA-256 differs from the pinned revision")
    return value, template


def run_probe(tokenizer_path: Path, config_path: Path) -> dict[str, object]:
    _, chat_template = _load_and_verify_files(tokenizer_path, config_path)
    if tokenizers_version != "0.22.2":
        raise RuntimeError(
            f"tokenizers backend must be 0.22.2, received {tokenizers_version}"
        )
    tokenizer = _PinnedQwenTokenizer(tokenizer_path, chat_template)
    if tokenizer.vocabulary_size != VOCABULARY_SIZE:
        raise RuntimeError("live tokenizer vocabulary size differs from the binding")
    for key, token_id in _special_token_ids():
        if key.startswith("additional:") and tokenizer.token_to_id(key.removeprefix("additional:")) != token_id:
            raise RuntimeError(f"special token mapping differs for {key}")
    manifest = TokenizerBindingManifest(
        repository_id=REPOSITORY_ID,
        revision=REVISION,
        tokenizer_json_sha256=TOKENIZER_JSON_SHA256,
        tokenizer_config_sha256=TOKENIZER_CONFIG_SHA256,
        chat_template_sha256=CHAT_TEMPLATE_SHA256,
        backend_name="tokenizers.Tokenizer",
        backend_version=tokenizers_version,
        vocabulary_size=VOCABULARY_SIZE,
        special_token_ids=_special_token_ids(),
    )
    if manifest.digest != TOKENIZER_BINDING_DIGEST:
        raise RuntimeError("structured tokenizer-binding digest differs")
    compiler = FragmentActionTokenCompiler(
        tokenizer,
        tokenizer_manifest=manifest,
        maximum_action_tokens=MAXIMUM_ACTION_TOKENS,
    )
    compiler.freeze_registered_language()
    if compiler.manifest.digest != COMPILER_MANIFEST_DIGEST:
        raise RuntimeError("frozen compiler-manifest digest differs")

    dialogue = (
        DialogueMessage(
            "system",
            "Play Hidden-Law Zendo. Return canonical JSON only.",
            "contract",
        ),
        DialogueMessage("user", "Choose the next legal action.", "opening"),
    )
    actions = (
        ReadyAction(),
        TestAction(scene_at(13_715)),
        AnswerAction(
            next(iter_syntactic_rules()),
            tuple(
                "fits" if index % 2 == 0 else "does_not_fit"
                for index in range(12)
            ),
        ),
    )
    decisions: list[dict[str, object]] = []
    ready_example = None
    ready_verified = None
    for action in actions:
        example = encode_decision_example(
            tokenizer,
            compiler,
            dialogue,
            action,
            tokenizer_binding_digest=manifest.digest,
            maximum_sequence_tokens=2_048,
        )
        verified = verify_decision_example(example, tokenizer, compiler, dialogue)
        loaded = type(example).from_json(example.to_json())
        reverified = verify_decision_example(loaded, tokenizer, compiler, dialogue)
        if reverified.verification_digest != verified.verification_digest:
            raise RuntimeError("strict reload changed decision verification")
        decisions.append(
            {
                "action_type": type(action).__name__,
                "prompt_token_count": len(example.prompt_token_ids),
                "action_token_count": example.action_token_count,
                "example_digest": example.digest,
                "verification_digest": verified.verification_digest,
            }
        )
        if type(action) is ReadyAction:
            ready_example = example
            ready_verified = verified
    if ready_example is None or ready_verified is None:
        raise AssertionError("probe omitted the ready decision")

    logits = torch.zeros(
        (len(ready_example.input_ids), VOCABULARY_SIZE),
        dtype=torch.float32,
        requires_grad=True,
    )
    statistics = masked_action_statistics(
        logits,
        ready_verified,
        temperature=0.7,
    )
    objective = -statistics.sequence_log_probability - 0.01 * statistics.mean_token_entropy
    objective.backward()  # type: ignore[no-untyped-call]
    if logits.grad is None:
        raise RuntimeError("masked decision replay produced no gradient")
    nonzero_gradient_count = int(torch.count_nonzero(logits.grad).item())
    if nonzero_gradient_count < 1:
        raise RuntimeError("masked decision replay gradient is identically zero")

    provider = _DeterministicProbeProvider()
    terminal_dialogue = (
        dialogue[0],
        DialogueMessage(
            "user",
            "Inquiry is over. Return one canonical answer with 12 labels.",
            "terminal",
        ),
    )
    authenticated_records: list[dict[str, object]] = []
    verified_samples = []
    sampling_cases: tuple[
        tuple[SamplingMode, tuple[DialogueMessage, ...], int | None], ...
    ] = (
        ("inquiry", dialogue, None),
        ("answer", terminal_dialogue, 12),
    )
    for turn_index, (mode, selected_dialogue, terminal_count) in enumerate(
        sampling_cases
    ):
        turn_seed = PolicyTurnSeed(
            run_seed=20260811,
            episode_digest=hashlib.sha256(
                b"g03-pinned-qwen-tokenizer-only-probe-episode-v1"
            ).hexdigest(),
            rollout_index=0,
            turn_index=turn_index,
            policy_state_digest=provider.policy_state_digest,
        )
        sample = sample_authenticated_action(
            provider,
            tokenizer,
            compiler,
            selected_dialogue,
            turn_seed,
            mode=mode,
            terminal_count=terminal_count,
            temperature=0.7,
            maximum_sequence_tokens=2_048,
        )
        verified_sample = replay_authenticated_sample(
            sample,
            provider,
            tokenizer,
            compiler,
            selected_dialogue,
            absolute_tolerance=0.0,
        )
        verified_samples.append(verified_sample)
        action = parse_action(sample.decision_example.action_trace.raw_action)
        authenticated_records.append(
            {
                "mode": mode,
                "sampled_action_type": type(action).__name__,
                "action_token_count": len(sample.selected_token_ids),
                "sample_digest": sample.digest,
                "verification_digest": verified_sample.verification_digest,
                "absolute_tolerance_hex": verified_sample.absolute_tolerance_hex,
            }
        )
    sampler_objective = -sum(
        (
            item.replayed_statistics.sequence_log_probability
            for item in verified_samples
        ),
        start=torch.zeros((), dtype=torch.float64),
    )
    sampler_objective.backward()  # type: ignore[no-untyped-call]
    if not all(
        item.replayed_statistics.sequence_log_probability.requires_grad
        for item in verified_samples
    ):
        raise RuntimeError("authenticated sampler replay lost its computation graph")

    report: dict[str, object] = {
        "schema_version": 2,
        "report_kind": "g03_pinned_qwen_decision_sampler_probe_v2",
        "tokenizer_binding_digest": manifest.digest,
        "compiler_manifest_digest": compiler.manifest.digest,
        "tokenizers_version": tokenizers_version,
        "vocabulary_size": VOCABULARY_SIZE,
        "decisions": decisions,
        "ready_masked_replay": {
            "temperature": 0.7,
            "action_token_count": statistics.action_token_count,
            "sequence_log_probability": float(
                statistics.sequence_log_probability.detach()
            ).hex(),
            "mean_token_entropy": float(
                statistics.mean_token_entropy.detach()
            ).hex(),
            "nonzero_gradient_count": nonzero_gradient_count,
        },
        "authenticated_sampler": {
            "policy_state_digest": provider.policy_state_digest,
            "temperature": 0.7,
            "records": authenticated_records,
            "differentiable_replay": True,
        },
        "authorization": {
            "live_model_loaded": False,
            "weight_update_authorized": False,
        },
    }
    report["report_digest"] = json_digest(
        report,
        domain="goalzendo-interactive-pinned-qwen-decision-sampler-probe-v2",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--tokenizer-config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_probe(args.tokenizer_json, args.tokenizer_config)
    rendered = dump_json(report) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
