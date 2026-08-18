from __future__ import annotations

import torch
from torch import nn

from goalzendo_hidden_law.evaluation import conflict_cell_agreements, evaluate_episode
from goalzendo_hidden_law.game import build_small_bank


class GreedyPolicy:
    def __init__(self) -> None:
        self.offset = nn.Parameter(torch.tensor(0.0))

    def score(self, prompts: tuple[str, ...], action_labels: tuple[str, ...]) -> torch.Tensor:
        values = torch.arange(len(action_labels), 0, -1, dtype=torch.float32)
        return values[None, :].expand(len(prompts), -1) + self.offset


def test_conflict_agreement_excludes_both_unanimous_cells() -> None:
    truths = {
        role: tuple(bool(mask & (1 << offset)) for mask in range(16))
        for offset, role in enumerate(("Y", "P", "Q", "R"))
    }
    first = [False] * 16
    second = list(first)
    second[0] = True
    second[15] = True
    first_agreement, first_indices = conflict_cell_agreements(first, truths)
    second_agreement, second_indices = conflict_cell_agreements(second, truths)
    assert first_indices == second_indices == tuple(range(1, 15))
    assert first_agreement == second_agreement


def test_all_three_evaluation_views_keep_terminal_calls_as_siblings() -> None:
    instance = build_small_bank().families[0].evaluation_games[0]
    policy = GreedyPolicy()
    results = {
        view: evaluate_episode(
            policy,
            instance,
            "eval_reverse",
            step=128,
            view=view,  # type: ignore[arg-type]
            include_interventions=False,
        )
        for view in ("active", "oracle_query", "no_query")
    }
    assert results["active"].summary["query_count"] == 2
    assert results["oracle_query"].summary["query_count"] == 2
    assert results["no_query"].summary["query_count"] == 0
    for view, result in results.items():
        assert len(result.metric_rows) == 1
        transcript = result.transcript_rows[0]
        classification = [
            decision for decision in transcript["decisions"] if decision["kind"] == "classification"
        ]
        assert len(classification) == 16
        assert all("Which displayed criterion" not in decision["prompt"] for decision in classification)
        assert all("Classify this new scene" in decision["prompt"] for decision in classification)
        assert result.summary["view"] == view
        assert result.summary["y_rule_family"] in {"monotone", "exactly_one"}
        assert result.summary["monotone_operator"] in {"all", "any"}
        assert set(result.summary["analysis_candidate_ids"]) == {"Y", "P", "Q", "R"}
        assert set(transcript["candidate_rules"]) == {"Y", "P", "Q", "R"}
