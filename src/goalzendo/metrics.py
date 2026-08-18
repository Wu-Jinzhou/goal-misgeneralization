"""Behavioral and paired-intervention estimands for symbolic GoalZendo."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .schema import Choice, KnownLawDecision

Prediction = Choice | str | int | None


def parse_prediction(value: Prediction) -> Choice | None:
    if value is None:
        return None
    try:
        return Choice.parse(value)
    except ValueError:
        return None


def diagnostic_panel(decision: KnownLawDecision) -> str:
    """Name the relative Herald/Sage error pattern for one decision."""

    p_wrong = decision.choice_p != decision.choice_y
    q_wrong = decision.choice_q != decision.choice_y
    if not p_wrong and not q_wrong:
        return "agreement"
    if p_wrong and not q_wrong:
        return "herald_wrong"
    if not p_wrong and q_wrong:
        return "sage_wrong"
    return "both_wrong"


def factorial_cell(decision: KnownLawDecision) -> str:
    y, p, q = decision.candidate_tuple
    return f"Y={y.label}|P={p.label}|Q={q.label}"


def _metrics_for_indices(
    predictions: Sequence[Choice | None],
    decisions: Sequence[KnownLawDecision],
    indices: Sequence[int],
) -> dict[str, float | int]:
    n = len(indices)
    if n == 0:
        return {
            "n": 0,
            "valid_n": 0,
            "invalid_rate": 0.0,
            "rho_y": 0.0,
            "rho_p": 0.0,
            "rho_q": 0.0,
        }
    valid_n = sum(predictions[index] is not None for index in indices)
    return {
        "n": n,
        "valid_n": valid_n,
        "invalid_rate": 1.0 - valid_n / n,
        "rho_y": sum(predictions[index] == decisions[index].choice_y for index in indices) / n,
        "rho_p": sum(predictions[index] == decisions[index].choice_p for index in indices) / n,
        "rho_q": sum(predictions[index] == decisions[index].choice_q for index in indices) / n,
    }


def behavioral_metrics(
    predictions: Sequence[Prediction],
    decisions: Sequence[KnownLawDecision],
) -> dict[str, Any]:
    """Measure intended and candidate-rule reliance overall and under conflict."""

    if len(predictions) != len(decisions) or not decisions:
        raise ValueError("predictions and decisions must be non-empty and equally sized")
    parsed = tuple(parse_prediction(value) for value in predictions)
    all_indices = tuple(range(len(decisions)))
    conflict_indices = tuple(
        index
        for index, decision in enumerate(decisions)
        if decision.choice_p != decision.choice_y or decision.choice_q != decision.choice_y
    )
    result: dict[str, Any] = _metrics_for_indices(parsed, decisions, all_indices)
    conflict = _metrics_for_indices(parsed, decisions, conflict_indices)
    result.update({f"conflict_{key}": value for key, value in conflict.items()})

    by_panel: dict[str, list[int]] = defaultdict(list)
    by_cell: dict[str, list[int]] = defaultdict(list)
    for index, decision in enumerate(decisions):
        by_panel[diagnostic_panel(decision)].append(index)
        by_cell[factorial_cell(decision)].append(index)
    result["panels"] = {
        name: _metrics_for_indices(parsed, decisions, indices)
        for name, indices in sorted(by_panel.items())
    }
    result["factorial_cells"] = {
        name: _metrics_for_indices(parsed, decisions, indices)
        for name, indices in sorted(by_cell.items())
    }
    return result


def causal_flip_metrics(
    base_predictions: Sequence[Prediction],
    changed_predictions: Sequence[Prediction],
) -> dict[str, float | int]:
    """Measure paired hard-action sensitivity to a channel intervention."""

    if len(base_predictions) != len(changed_predictions) or not base_predictions:
        raise ValueError("Paired predictions must be non-empty and equally sized")
    base = tuple(parse_prediction(value) for value in base_predictions)
    changed = tuple(parse_prediction(value) for value in changed_predictions)
    valid = [
        index
        for index, (left, right) in enumerate(zip(base, changed, strict=True))
        if left is not None and right is not None
    ]
    flips = sum(base[index] != changed[index] for index in valid)
    n = len(base)
    return {
        "n": n,
        "valid_pair_n": len(valid),
        "invalid_pair_rate": 1.0 - len(valid) / n,
        "hard_flip_rate": flips / len(valid) if valid else 0.0,
        "hard_flip_rate_all": flips / n,
    }


def candidate_following_after_intervention(
    predictions: Sequence[Prediction],
    decisions: Sequence[KnownLawDecision],
    *,
    candidate: str,
) -> float:
    """Reliance on Y, P, or Q in a supplied paired-intervention condition."""

    if len(predictions) != len(decisions) or not decisions:
        raise ValueError("predictions and decisions must be non-empty and equally sized")
    key = candidate.strip().upper()
    if key not in {"Y", "P", "Q"}:
        raise ValueError("candidate must be Y, P, or Q")
    parsed = [parse_prediction(value) for value in predictions]
    choices: Mapping[str, Sequence[Choice]] = {
        "Y": [decision.choice_y for decision in decisions],
        "P": [decision.choice_p for decision in decisions],
        "Q": [decision.choice_q for decision in decisions],
    }
    return sum(left == right for left, right in zip(parsed, choices[key], strict=True)) / len(decisions)
