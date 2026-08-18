from __future__ import annotations

import pytest

from goalzendo.generation import generate_factorial_evaluation
from goalzendo.interventions import flip_herald
from goalzendo.metrics import (
    behavioral_metrics,
    candidate_following_after_intervention,
    causal_flip_metrics,
    diagnostic_panel,
)
from goalzendo.schema import RuleSpec


@pytest.fixture
def factorial():
    return generate_factorial_evaluation(
        repeats=4,
        seed=53,
        rule=RuleSpec("majority", (0, 1, 2)),
        sage_rule=RuleSpec("parity", (3, 4), name="sage_rule"),
        feature_count=5,
    )


def test_intended_policy_is_distinguished_from_both_candidate_proxies(factorial) -> None:
    predictions = [decision.choice_y for decision in factorial.decisions]
    metrics = behavioral_metrics(predictions, factorial.decisions)
    assert metrics["n"] == 32
    assert metrics["rho_y"] == 1.0
    assert metrics["rho_p"] == metrics["rho_q"] == 0.5
    assert metrics["conflict_n"] == 24
    assert metrics["conflict_rho_y"] == 1.0
    assert set(metrics["panels"]) == {
        "agreement",
        "herald_wrong",
        "sage_wrong",
        "both_wrong",
    }
    assert {values["n"] for values in metrics["panels"].values()} == {8}
    assert len(metrics["factorial_cells"]) == 8


def test_proxy_policies_have_distinct_diagnostic_signatures(factorial) -> None:
    p_metrics = behavioral_metrics(
        [decision.choice_p for decision in factorial.decisions],
        factorial.decisions,
    )
    q_metrics = behavioral_metrics(
        [decision.choice_q for decision in factorial.decisions],
        factorial.decisions,
    )
    assert p_metrics["rho_p"] == 1.0
    assert p_metrics["rho_y"] == p_metrics["rho_q"] == 0.5
    assert q_metrics["rho_q"] == 1.0
    assert q_metrics["rho_y"] == q_metrics["rho_p"] == 0.5


def test_invalid_predictions_remain_in_the_estimand_denominator(factorial) -> None:
    predictions = [None] * len(factorial)
    predictions[0] = factorial.decisions[0].choice_y
    metrics = behavioral_metrics(predictions, factorial.decisions)
    assert metrics["valid_n"] == 1
    assert metrics["invalid_rate"] == 31 / 32
    assert metrics["rho_y"] == 1 / 32


def test_paired_causal_flip_metric_detects_a_herald_controlled_policy(factorial) -> None:
    base = list(factorial.decisions)
    changed = [flip_herald(decision) for decision in base]
    base_predictions = [decision.choice_p for decision in base]
    changed_predictions = [decision.choice_p for decision in changed]
    effects = causal_flip_metrics(base_predictions, changed_predictions)
    assert effects == {
        "n": 32,
        "valid_pair_n": 32,
        "invalid_pair_rate": 0.0,
        "hard_flip_rate": 1.0,
        "hard_flip_rate_all": 1.0,
    }
    assert candidate_following_after_intervention(
        changed_predictions,
        changed,
        candidate="P",
    ) == 1.0


def test_relative_panel_names_cover_binary_error_geometry(factorial) -> None:
    panels = {diagnostic_panel(decision) for decision in factorial.decisions}
    assert panels == {"agreement", "herald_wrong", "sage_wrong", "both_wrong"}
