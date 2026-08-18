from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import replace

import pytest
import torch

from goalzendo_interactive.action_tokenization_v2 import (
    FragmentActionTokenCompiler,
    TokenizerBindingManifest,
)
from goalzendo_interactive.actions import AnswerAction, parse_action
from goalzendo_interactive.authenticated_sampler_v2 import (
    AUTHENTICATED_SAMPLER_CONTRACT_ID,
    AUTHENTICATED_SAMPLER_SCHEMA_VERSION,
    AuthenticatedActionSample,
    AuthenticatedSamplingError,
    AuthenticatedSamplingOverlengthError,
    GraphFreeVerifiedAuthenticatedActionSample,
    VerifiedAuthenticatedActionSample,
    authenticated_sampler_manifest,
    replay_authenticated_sample,
    sample_authenticated_action,
    verify_eight_rollout_policy_group,
)
from goalzendo_interactive.dialogue import Dialogue, DialogueMessage, DialoguePhase
from goalzendo_interactive.policy_randomness_v2 import (
    PolicyTokenDraw,
    PolicyTurnSeed,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


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


class DeterministicLogitsProvider:
    def __init__(self, policy_label: str) -> None:
        self._policy_state_digest = _digest(policy_label)
        self._policy_offset = int(self._policy_state_digest[:8], 16)
        self.last_full_forward: torch.Tensor | None = None

    @property
    def policy_state_digest(self) -> str:
        return self._policy_state_digest

    def _row(self, input_ids: tuple[int, ...]) -> torch.Tensor:
        prefix_fingerprint = sum((index + 1) * token_id for index, token_id in enumerate(input_ids[-32:]))
        offset = (prefix_fingerprint + 19 * len(input_ids) + self._policy_offset) % 251
        token_ids = torch.arange(256, dtype=torch.int64)
        integers = torch.remainder(token_ids * 67 + offset * 29, 251) - 125
        return integers.to(dtype=torch.float64) / 31.0

    def next_token_logits(self, input_ids: tuple[int, ...]) -> torch.Tensor:
        return self._row(input_ids)

    def full_forward_logits(self, input_ids: tuple[int, ...]) -> torch.Tensor:
        result = torch.stack([self._row(input_ids[: position + 1]) for position in range(len(input_ids))])
        result.requires_grad_(True)
        self.last_full_forward = result
        return result


class MutatingLogitsProvider(DeterministicLogitsProvider):
    def next_token_logits(self, input_ids: tuple[int, ...]) -> torch.Tensor:
        result = super().next_token_logits(input_ids)
        self._policy_state_digest = _digest("mutated-policy")
        return result


class PerturbedSameDigestProvider(DeterministicLogitsProvider):
    def _row(self, input_ids: tuple[int, ...]) -> torch.Tensor:
        base = super()._row(input_ids)
        return base + torch.arange(256, dtype=torch.float64) / 10_000.0


class FullForwardTamperProvider(DeterministicLogitsProvider):
    def __init__(self, policy_label: str, token_id: int) -> None:
        super().__init__(policy_label)
        self._token_id = token_id

    def full_forward_logits(self, input_ids: tuple[int, ...]) -> torch.Tensor:
        exact = super().full_forward_logits(input_ids)
        delta = torch.zeros_like(exact)
        delta[:, self._token_id] = 0.125
        result = exact + delta
        self.last_full_forward = result
        return result


class NondifferentiableFullForwardProvider(DeterministicLogitsProvider):
    def full_forward_logits(self, input_ids: tuple[int, ...]) -> torch.Tensor:
        return super().full_forward_logits(input_ids).detach()


class PaddedVocabularyLogitsProvider(DeterministicLogitsProvider):
    """Expose model padding rows beyond the tokenizer's legal vocabulary."""

    def _row(self, input_ids: tuple[int, ...]) -> torch.Tensor:
        base = super()._row(input_ids)
        padding = torch.linspace(-500.0, 500.0, 8, dtype=base.dtype)
        return torch.cat((base, padding))


class UndersizedVocabularyLogitsProvider(DeterministicLogitsProvider):
    def _row(self, input_ids: tuple[int, ...]) -> torch.Tensor:
        return super()._row(input_ids)[:-1]


@pytest.fixture(scope="module")
def tokenizer() -> ExactCharacterChatTokenizer:
    return ExactCharacterChatTokenizer()


@pytest.fixture(scope="module")
def tokenizer_manifest() -> TokenizerBindingManifest:
    return TokenizerBindingManifest(
        repository_id="test/exact-character-tokenizer",
        revision="1" * 40,
        tokenizer_json_sha256="2" * 64,
        tokenizer_config_sha256="3" * 64,
        chat_template_sha256="4" * 64,
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


@pytest.fixture(scope="module")
def compiler(
    tokenizer: ExactCharacterChatTokenizer,
    tokenizer_manifest: TokenizerBindingManifest,
) -> FragmentActionTokenCompiler:
    result = FragmentActionTokenCompiler(
        tokenizer,
        tokenizer_manifest=tokenizer_manifest,
        maximum_action_tokens=2_048,
    )
    result.freeze_registered_language()
    return result


def _dialogue(label: str = "opening") -> Dialogue:
    phase: DialoguePhase = "terminal" if label == "terminal" else "opening"
    return (
        DialogueMessage("system", "Play hidden-law Zendo.", "contract"),
        DialogueMessage(
            "user",
            f"Choose one canonical action for {label}.",
            phase,
        ),
    )


def _turn_seed(
    provider: DeterministicLogitsProvider,
    *,
    rollout_index: int = 0,
    turn_index: int = 0,
) -> PolicyTurnSeed:
    return PolicyTurnSeed(
        run_seed=20260811,
        episode_digest=_digest("episode-17"),
        rollout_index=rollout_index,
        turn_index=turn_index,
        policy_state_digest=provider.policy_state_digest,
    )


def _sample(
    tokenizer: ExactCharacterChatTokenizer,
    compiler: FragmentActionTokenCompiler,
    provider: DeterministicLogitsProvider,
    *,
    rollout_index: int = 0,
    temperature: float = 0.73,
) -> AuthenticatedActionSample:
    return sample_authenticated_action(
        provider,
        tokenizer,
        compiler,
        _dialogue(),
        _turn_seed(provider, rollout_index=rollout_index),
        mode="inquiry",
        terminal_count=None,
        temperature=temperature,
        maximum_sequence_tokens=4_096,
    )


def _graph_free_evidence(
    sample: AuthenticatedActionSample,
    provider: DeterministicLogitsProvider,
    tokenizer: ExactCharacterChatTokenizer,
    compiler: FragmentActionTokenCompiler,
    *,
    absolute_tolerance: float = 0,
) -> GraphFreeVerifiedAuthenticatedActionSample:
    verified = replay_authenticated_sample(
        sample,
        provider,
        tokenizer,
        compiler,
        _dialogue(),
        absolute_tolerance=absolute_tolerance,
    )
    return GraphFreeVerifiedAuthenticatedActionSample._from_verified(verified)


def test_sample_round_trip_and_differentiable_same_policy_replay(
    tokenizer: ExactCharacterChatTokenizer,
    compiler: FragmentActionTokenCompiler,
) -> None:
    provider = DeterministicLogitsProvider("policy-a")
    sample = _sample(tokenizer, compiler, provider)

    assert sample.schema_version == AUTHENTICATED_SAMPLER_SCHEMA_VERSION == 1
    assert sample.contract_id == AUTHENTICATED_SAMPLER_CONTRACT_ID
    assert sample.selected_token_ids == sample.decision_example.action_trace.action_token_ids
    assert tuple(draw.action_token_index for draw in sample.draws) == tuple(
        range(len(sample.selected_token_ids))
    )
    assert AuthenticatedActionSample.from_json(sample.to_json()) == sample

    for step, log_probability, entropy in zip(
        sample.decision_example.action_trace.steps,
        sample.detached_statistics.token_log_probabilities,
        sample.detached_statistics.token_entropies,
        strict=True,
    ):
        if len(step.allowed_token_ids) == 1:
            assert log_probability == 0.0
            assert entropy == 0.0

    verified = replay_authenticated_sample(
        sample,
        provider,
        tokenizer,
        compiler,
        _dialogue(),
        absolute_tolerance=0,
    )
    assert type(verified) is VerifiedAuthenticatedActionSample
    assert verified.sample == sample
    assert verified.absolute_tolerance_hex == (0.0).hex()
    assert len(verified.verification_digest) == 64
    torch.autograd.backward(verified.replayed_statistics.sequence_log_probability)
    assert provider.last_full_forward is not None
    assert provider.last_full_forward.grad is not None
    assert torch.count_nonzero(provider.last_full_forward.grad).item() > 0


def test_terminal_answer_samples_rule_and_classifications_without_cartesian_product(
    tokenizer: ExactCharacterChatTokenizer,
    compiler: FragmentActionTokenCompiler,
) -> None:
    provider = DeterministicLogitsProvider("policy-terminal")
    seed = _turn_seed(provider)
    sample = sample_authenticated_action(
        provider,
        tokenizer,
        compiler,
        _dialogue("terminal"),
        seed,
        mode="answer",
        terminal_count=2,
        temperature=1.0,
        maximum_sequence_tokens=4_096,
    )
    action = parse_action(sample.decision_example.action_trace.raw_action)
    assert type(action) is AnswerAction
    assert len(action.classifications) == 2
    assert sample.terminal_count == 2
    assert compiler.runtime_counters.answer_rule_label_cartesian_product_count == 0
    replay_authenticated_sample(
        sample,
        provider,
        tokenizer,
        compiler,
        _dialogue("terminal"),
        absolute_tolerance=0,
    )


def test_model_vocabulary_may_pad_beyond_tokenizer_but_not_undershoot(
    tokenizer: ExactCharacterChatTokenizer,
    compiler: FragmentActionTokenCompiler,
) -> None:
    padded = PaddedVocabularyLogitsProvider("policy-padded-vocabulary")
    sample = _sample(tokenizer, compiler, padded)
    assert max(sample.selected_token_ids) < 256
    replay_authenticated_sample(
        sample,
        padded,
        tokenizer,
        compiler,
        _dialogue(),
        absolute_tolerance=0,
    )

    undersized = UndersizedVocabularyLogitsProvider("policy-undersized-vocabulary")
    with pytest.raises(AuthenticatedSamplingError, match="do not cover"):
        _sample(tokenizer, compiler, undersized)


def test_draw_trace_provider_and_canonical_json_tampering_fail_closed(
    tokenizer: ExactCharacterChatTokenizer,
    compiler: FragmentActionTokenCompiler,
) -> None:
    provider = DeterministicLogitsProvider("policy-tamper")
    sample = _sample(tokenizer, compiler, provider)

    first_draw = sample.draws[0]
    changed_draw = PolicyTokenDraw(
        turn_seed_digest=first_draw.turn_seed_digest,
        action_token_index=first_draw.action_token_index,
        word_u53=first_draw.word_u53 ^ 1,
    )
    with pytest.raises(AuthenticatedSamplingError, match="does not rederive"):
        replace(sample, draws=(changed_draw, *sample.draws[1:]))

    first_step = sample.decision_example.action_trace.steps[0]
    extra_token = next(token_id for token_id in range(256) if token_id not in first_step.allowed_token_ids)
    altered_step = replace(
        first_step,
        allowed_token_ids=tuple(sorted((*first_step.allowed_token_ids, extra_token))),
    )
    altered_trace = replace(
        sample.decision_example.action_trace,
        steps=(altered_step, *sample.decision_example.action_trace.steps[1:]),
    )
    altered_example = replace(sample.decision_example, action_trace=altered_trace)
    altered_detached = replace(
        sample.detached_statistics,
        example_digest=altered_example.digest,
    )
    forged = replace(
        sample,
        decision_example=altered_example,
        detached_statistics=altered_detached,
    )
    with pytest.raises(AuthenticatedSamplingError, match="source regeneration"):
        replay_authenticated_sample(
            forged,
            provider,
            tokenizer,
            compiler,
            _dialogue(),
            absolute_tolerance=0,
        )

    changed_provider = DeterministicLogitsProvider("another-policy")
    with pytest.raises(AuthenticatedSamplingError, match="policy state differs"):
        replay_authenticated_sample(
            sample,
            changed_provider,
            tokenizer,
            compiler,
            _dialogue(),
            absolute_tolerance=0,
        )

    perturbed_provider = PerturbedSameDigestProvider("policy-tamper")
    with pytest.raises(AuthenticatedSamplingError, match="stateless same-policy"):
        replay_authenticated_sample(
            sample,
            perturbed_provider,
            tokenizer,
            compiler,
            _dialogue(),
            absolute_tolerance=0,
        )

    first_branch = next(
        step for step in sample.decision_example.action_trace.steps if len(step.allowed_token_ids) > 1
    )
    full_tamper_provider = FullForwardTamperProvider("policy-tamper", first_branch.selected_token_id)
    with pytest.raises(AuthenticatedSamplingError, match="full-forward replay"):
        replay_authenticated_sample(
            sample,
            full_tamper_provider,
            tokenizer,
            compiler,
            _dialogue(),
            absolute_tolerance=0,
        )

    nondifferentiable = NondifferentiableFullForwardProvider("policy-tamper")
    with pytest.raises(AuthenticatedSamplingError, match="differentiable"):
        replay_authenticated_sample(
            sample,
            nondifferentiable,
            tokenizer,
            compiler,
            _dialogue(),
            absolute_tolerance=0,
        )

    payload = json.loads(sample.to_json())
    payload["draws"][0]["word_u53"] ^= 1
    tampered_json = json.dumps(payload, separators=(",", ":"))
    with pytest.raises(AuthenticatedSamplingError):
        AuthenticatedActionSample.from_json(tampered_json)

    reordered = json.dumps(
        json.loads(sample.to_json()),
        sort_keys=True,
        separators=(",", ":"),
    )
    assert reordered != sample.to_json()
    with pytest.raises(AuthenticatedSamplingError):
        AuthenticatedActionSample.from_json(reordered)


def test_policy_mutation_during_sampling_is_fatal(
    tokenizer: ExactCharacterChatTokenizer,
    compiler: FragmentActionTokenCompiler,
) -> None:
    provider = MutatingLogitsProvider("initial-policy")
    seed = _turn_seed(provider)
    with pytest.raises(AuthenticatedSamplingError, match="changed during token sampling"):
        sample_authenticated_action(
            provider,
            tokenizer,
            compiler,
            _dialogue(),
            seed,
            mode="inquiry",
            terminal_count=None,
            temperature=1.0,
            maximum_sequence_tokens=4_096,
        )

    stable_provider = DeterministicLogitsProvider("phase-policy")
    with pytest.raises(AuthenticatedSamplingError, match="dialogue phase"):
        sample_authenticated_action(
            stable_provider,
            tokenizer,
            compiler,
            _dialogue("terminal"),
            _turn_seed(stable_provider),
            mode="inquiry",
            terminal_count=None,
            temperature=1.0,
            maximum_sequence_tokens=4_096,
        )

    with pytest.raises(
        AuthenticatedSamplingOverlengthError,
        match="maximum_sequence_tokens",
    ):
        sample_authenticated_action(
            stable_provider,
            tokenizer,
            compiler,
            _dialogue(),
            _turn_seed(stable_provider),
            mode="inquiry",
            terminal_count=None,
            temperature=1.0,
            maximum_sequence_tokens=2,
        )


def test_stateless_schedule_order_and_eight_rollout_policy_gate(
    tokenizer: ExactCharacterChatTokenizer,
    compiler: FragmentActionTokenCompiler,
) -> None:
    provider = DeterministicLogitsProvider("policy-schedule")
    forward_order = list(range(8))
    shuffled_order = [5, 1, 7, 0, 3, 6, 2, 4]
    forward = {
        rollout_index: _sample(
            tokenizer,
            compiler,
            provider,
            rollout_index=rollout_index,
        )
        for rollout_index in forward_order
    }
    shuffled = {
        rollout_index: _sample(
            tokenizer,
            compiler,
            provider,
            rollout_index=rollout_index,
        )
        for rollout_index in shuffled_order
    }
    assert {rollout_index: sample.to_json() for rollout_index, sample in forward.items()} == {
        rollout_index: sample.to_json() for rollout_index, sample in shuffled.items()
    }
    verified = {
        rollout_index: _graph_free_evidence(
            sample,
            provider,
            tokenizer,
            compiler,
        )
        for rollout_index, sample in forward.items()
    }
    digest_forward = verify_eight_rollout_policy_group(tuple(verified.values()))
    digest_shuffled = verify_eight_rollout_policy_group(tuple(verified[index] for index in shuffled_order))
    assert digest_forward == digest_shuffled

    with pytest.raises(AuthenticatedSamplingError, match="replay-verified"):
        verify_eight_rollout_policy_group(tuple(forward.values()))  # type: ignore[arg-type]
    graph_bearing = replay_authenticated_sample(
        forward[0],
        provider,
        tokenizer,
        compiler,
        _dialogue(),
        absolute_tolerance=0,
    )
    with pytest.raises(AuthenticatedSamplingError, match="graph-free"):
        verify_eight_rollout_policy_group(
            (graph_bearing, *(verified[index] for index in range(1, 8)))  # type: ignore[arg-type]
        )
    del graph_bearing
    with pytest.raises(AuthenticatedSamplingError, match="all eight"):
        verify_eight_rollout_policy_group(tuple(verified.values())[:-1])

    changed_provider = DeterministicLogitsProvider("changed-schedule-policy")
    changed = _sample(tokenizer, compiler, changed_provider, rollout_index=7)
    changed_verified = _graph_free_evidence(
        changed,
        changed_provider,
        tokenizer,
        compiler,
    )
    mixed = (*tuple(verified[index] for index in range(7)), changed_verified)
    with pytest.raises(AuthenticatedSamplingError, match="policy_state_digest"):
        verify_eight_rollout_policy_group(mixed)

    changed_temperature = _sample(
        tokenizer,
        compiler,
        provider,
        rollout_index=7,
        temperature=0.91,
    )
    changed_temperature_verified = _graph_free_evidence(
        changed_temperature,
        provider,
        tokenizer,
        compiler,
    )
    mixed_temperature = (
        *tuple(verified[index] for index in range(7)),
        changed_temperature_verified,
    )
    with pytest.raises(AuthenticatedSamplingError, match="temperature"):
        verify_eight_rollout_policy_group(mixed_temperature)

    changed_tolerance_verified = _graph_free_evidence(
        forward[7],
        provider,
        tokenizer,
        compiler,
        absolute_tolerance=1e-9,
    )
    mixed_tolerance = (
        *tuple(verified[index] for index in range(7)),
        changed_tolerance_verified,
    )
    with pytest.raises(AuthenticatedSamplingError, match="replay tolerance"):
        verify_eight_rollout_policy_group(mixed_tolerance)


def test_manifest_is_explicitly_nonauthorizing() -> None:
    manifest = authenticated_sampler_manifest()
    assert manifest["live_replay_evidence"] == "ephemeral VerifiedAuthenticatedActionSample"
    assert manifest["retained_action_evidence"] == "GraphFreeVerifiedAuthenticatedActionSample"
    assert manifest["retained_differentiable_graphs"] is False
    assert manifest["stateful_rng_present"] is False
    assert manifest["live_model_authorization"] is False
    assert manifest["weight_update_authorization"] is False
    assert manifest["rollout_group_size"] == 8
