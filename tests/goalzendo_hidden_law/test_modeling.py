from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from goalzendo_hidden_law.modeling import (
    FiniteActionScorer,
    format_chat_prompts,
    score_finite_action_sequences,
    validate_action_labels,
)


class ByteTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self) -> None:
        self.chat_calls: list[tuple[list[dict[str, str]], bool, bool, bool]] = []

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        tokens = [ord(character) + 2 for character in text]
        return [1, *tokens] if add_special_tokens else tokens

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        self.chat_calls.append((messages, tokenize, add_generation_prompt, enable_thinking))
        body = "|".join(f"{message['role']}={message['content']}" for message in messages)
        return f"<{body}|assistant="


class TransitionLM(nn.Module):
    """A deterministic causal LM whose next logits depend on the current token."""

    def __init__(self, vocabulary_size: int = 260) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(17)
        self.transitions = nn.Parameter(torch.randn(vocabulary_size, vocabulary_size, generator=generator))
        self.calls = 0

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> SimpleNamespace:
        del attention_mask
        self.calls += 1
        return SimpleNamespace(logits=self.transitions[input_ids])


class LastLogitTransitionLM(TransitionLM):
    def __init__(self, vocabulary_size: int = 260) -> None:
        super().__init__(vocabulary_size)
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
        self.received_logits_to_keep = logits_to_keep
        self.received_position_ids = position_ids.detach().clone()
        logits = self.transitions[input_ids]
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:, :]
        return SimpleNamespace(logits=logits)


@pytest.mark.parametrize("action_count", [2, 4, 9])
def test_finite_action_scorer_matches_direct_next_token_logits_and_gradients(
    action_count: int,
) -> None:
    tokenizer = ByteTokenizer()
    model = TransitionLM()
    labels = tuple(chr(ord("A") + index) for index in range(action_count))
    prompts = ("P", "PQ")

    result = score_finite_action_sequences(model, tokenizer, prompts, labels)
    action_ids = torch.tensor([tokenizer.encode(label, add_special_tokens=False)[0] for label in labels])
    expected = torch.stack(
        [model.transitions[tokenizer.encode(prompt)[-1], action_ids] for prompt in prompts]
    )

    assert result.log_scores.shape == (2, action_count)
    assert result.action_labels == labels
    assert torch.allclose(result.log_scores, expected)
    assert model.calls == 1
    assert torch.allclose(result.probabilities.sum(dim=-1), torch.ones(2))

    result.log_scores.sum().backward()
    assert model.transitions.grad is not None
    assert torch.count_nonzero(model.transitions.grad) == 2 * action_count


def test_single_token_fast_path_is_exact_for_heterogeneous_prompts_and_nine_actions() -> None:
    tokenizer = ByteTokenizer()
    slow = TransitionLM()
    fast = LastLogitTransitionLM()
    fast.load_state_dict(slow.state_dict())
    prompts = ("P", "PQR")
    labels = tuple("ABCDEFGHI")

    slow_scores = score_finite_action_sequences(slow, tokenizer, prompts, labels).log_scores
    fast_scores = score_finite_action_sequences(fast, tokenizer, prompts, labels).log_scores
    assert torch.allclose(fast_scores, slow_scores)
    slow_scores.sum().backward()
    fast_scores.sum().backward()
    assert torch.equal(fast.transitions.grad, slow.transitions.grad)
    assert fast.received_logits_to_keep == 1
    assert fast.received_position_ids is not None
    assert torch.equal(
        fast.received_position_ids,
        torch.tensor([[0, 0, 0, 1], [0, 1, 2, 3]]),
    )


def test_multi_token_actions_remain_exact_sequence_log_probabilities() -> None:
    tokenizer = ByteTokenizer()
    model = TransitionLM()
    labels = ("A", "BC", "DEF", "G")
    result = score_finite_action_sequences(model, tokenizer, ("P",), labels)
    assert model.calls == 1

    transition_log_probabilities = model.transitions.log_softmax(dim=-1)
    prompt_last = tokenizer.encode("P")[-1]
    expected: list[torch.Tensor] = []
    for label in labels:
        action_ids = tokenizer.encode(label, add_special_tokens=False)
        preceding = [prompt_last, *action_ids[:-1]]
        expected.append(
            sum(
                transition_log_probabilities[previous, target]
                for previous, target in zip(preceding, action_ids, strict=True)
            )
        )
    assert torch.allclose(result.log_scores[0], torch.stack(expected))


def test_shared_chat_formatter_and_module_wrapper_leave_answer_slot_open() -> None:
    tokenizer = ByteTokenizer()
    model = TransitionLM()
    rendered = format_chat_prompts(
        tokenizer,
        ("question one", "question two"),
        system_prompt="system",
        enable_thinking=False,
    )
    assert rendered == (
        "<system=system|user=question one|assistant=",
        "<system=system|user=question two|assistant=",
    )
    assert all(call[0][0] == {"role": "system", "content": "system"} for call in tokenizer.chat_calls)

    scorer = FiniteActionScorer(
        model,
        tokenizer,
        ("A", "B", "C", "D"),
        format_as_chat=True,
        system_prompt="system",
        add_prompt_special_tokens=False,
    )
    scores = scorer(("question one", "question two"))
    expected = model.transitions[
        tokenizer.encode(rendered[0], add_special_tokens=False)[-1],
        torch.tensor([ord(label) + 2 for label in "ABCD"]),
    ]
    assert torch.allclose(scores[0], expected)
    scores.sum().backward()
    assert model.transitions.grad is not None


@pytest.mark.parametrize(
    ("labels", "exception", "match"),
    [
        (("A",), ValueError, "at least two"),
        (("A", "A"), ValueError, "unique"),
        (("A", ""), ValueError, "non-empty"),
        (("A", "   "), ValueError, "non-empty"),
        (("A", "AB"), ValueError, "prefix-free"),
        (("A", 3), TypeError, "string"),
        ("AB", TypeError, "sequence"),
    ],
)
def test_action_labels_fail_closed(
    labels: object,
    exception: type[Exception],
    match: str,
) -> None:
    with pytest.raises(exception, match=match):
        validate_action_labels(labels)  # type: ignore[arg-type]


class TokenPrefixTokenizer(ByteTokenizer):
    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        del add_special_tokens
        return {"A": [4], "B": [4, 5]}[text]


def test_token_level_prefix_safety_uses_the_proven_binary_validator() -> None:
    with pytest.raises(ValueError, match="prefix-free"):
        validate_action_labels(("A", "B"), tokenizer=TokenPrefixTokenizer())


def test_chat_mode_rejects_double_special_token_insertion() -> None:
    with pytest.raises(ValueError, match="special tokens"):
        FiniteActionScorer(
            TransitionLM(),
            ByteTokenizer(),
            ("A", "B"),
            format_as_chat=True,
            add_prompt_special_tokens=True,
        )
