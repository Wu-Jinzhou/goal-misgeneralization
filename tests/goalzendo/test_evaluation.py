from __future__ import annotations

import itertools

import pytest
import torch
from torch import nn

from goalzendo.evaluation import evaluate_batches


class PromptScorer(nn.Module):
    def forward(self, prompts: list[str]) -> torch.Tensor:
        return torch.tensor(
            [[-2.0, 2.0] if "choose-B" in prompt else [2.0, -2.0] for prompt in prompts]
        )


def test_full_factorial_and_prompt_view_summaries() -> None:
    examples = []
    for index, (choice_y, choice_p, choice_q) in enumerate(
        itertools.product((0, 1), repeat=3)
    ):
        answer = "choose-B" if choice_y else "choose-A"
        examples.append(
            {
                "sample_id": f"cell-{index}",
                "candidate_choices": {"y": choice_y, "p": choice_p, "q": choice_q},
                "prompt_views": {
                    "natural": f"natural {answer}",
                    "nonce": f"nonce {answer}",
                },
            }
        )

    scorer = PromptScorer()
    scorer.train()
    result = evaluate_batches((examples[:3], examples[3:]), scorer)
    assert scorer.training  # Evaluation restores the caller's mode.
    assert len(result.records) == 16
    assert len(result.factorial_cells) == 16
    assert {cell["prompt_view"] for cell in result.factorial_cells} == {"natural", "nonce"}
    assert all(cell["agreement_y"] == 1.0 for cell in result.factorial_cells)
    conflict_rows = [
        row for row in result.behavioral_agreement if row["panel"] == "conflict"
    ]
    assert len(conflict_rows) == 2
    assert all(row["agreement_y"] == 1.0 for row in conflict_rows)


def test_matched_intervention_reports_margin_and_action_flip() -> None:
    base = {
        "sample_id": "base",
        "choices": {"law": "A", "herald": "A", "sage": "A"},
        "prompt_views": {"natural": "choose-A", "nonce": "nonce choose-A"},
        "intervention_pair_id": "pair-1",
        "intervention_role": "base",
        "intervention_target": "herald",
    }
    intervention = {
        "sample_id": "flip",
        "choices": {"law": "A", "herald": "B", "sage": "A"},
        "prompt_views": {"natural": "choose-B", "nonce": "nonce choose-B"},
        "intervention_pair_id": "pair-1",
        "intervention_role": "intervention",
        "intervention_target": "herald",
    }
    result = evaluate_batches(([base, intervention],), PromptScorer())
    assert len(result.intervention_effects) == 2
    for effect in result.intervention_effects:
        assert effect.delta_margin_b_minus_a == 8.0
        assert effect.action_flipped
        assert effect.target_choice_changed
        assert effect.target_aligned_delta_margin == 8.0
    assert all(row["action_flip_rate"] == 1.0 for row in result.intervention_summary)


def test_incomplete_intervention_metadata_and_pairs_fail_loudly() -> None:
    incomplete_metadata = {
        "sample_id": "bad",
        "choices": {"y": 0, "p": 0, "q": 0},
        "prompt": "choose-A",
        "intervention_pair_id": "pair",
    }
    with pytest.raises(ValueError, match="supplied together"):
        evaluate_batches(([incomplete_metadata],), PromptScorer())

    unpaired = {
        "sample_id": "base",
        "choices": {"y": 0, "p": 0, "q": 0},
        "prompt": "choose-A",
        "intervention_pair_id": "pair",
        "intervention_role": "base",
        "intervention_target": "herald",
    }
    with pytest.raises(ValueError, match="exactly one base"):
        evaluate_batches(([unpaired],), PromptScorer())


class NonFiniteScorer(nn.Module):
    def forward(self, prompts: list[str]) -> torch.Tensor:
        return torch.tensor([[0.0, torch.nan] for _prompt in prompts])


def test_evaluation_rejects_non_finite_scores_before_recording() -> None:
    example = {
        "sample_id": "bad-score",
        "choices": {"y": 0, "p": 0, "q": 0},
        "prompt": "choose-A",
    }
    with pytest.raises(FloatingPointError, match="NaN or infinity"):
        evaluate_batches(([example],), NonFiniteScorer())
