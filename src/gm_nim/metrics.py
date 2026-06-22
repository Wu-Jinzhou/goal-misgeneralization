from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

from .games import action_to_residue

MOVE_RE = re.compile(r"take\s+(-?\d+)\s+coins?", re.IGNORECASE)
INTEGER_RE = re.compile(r"-?\d+")
PILE_MOVE_RE = re.compile(r"take\s+(-?\d+)\s+from\s+pile\s+(-?\d+)", re.IGNORECASE)
PAIR_RE = re.compile(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")


def parse_bounded_move(text: str) -> int | None:
    match = MOVE_RE.search(text)
    if match:
        return int(match.group(1))
    match = INTEGER_RE.search(text)
    if match:
        return int(match.group(0))
    return None


def parse_modular_answer(text: str) -> int | None:
    match = INTEGER_RE.search(text)
    return int(match.group(0)) if match else None


def parse_multipile_move(text: str) -> tuple[int, int] | None:
    match = PILE_MOVE_RE.search(text)
    if not match:
        return None
    return int(match.group(2)), int(match.group(1))


def parse_pair(text: str) -> tuple[int, int] | None:
    match = PAIR_RE.search(text)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


@dataclass(frozen=True)
class AccuracySummary:
    exact: float
    invalid_rate: float
    n: int
    coarsened: dict[int, float]
    prediction_distribution: dict[int, float]
    confusion: dict[tuple[int, int], int]


def bounded_accuracy(
    predictions: Iterable[int | None],
    labels: Iterable[int],
    mr: int,
    factors: Iterable[int] = (),
) -> AccuracySummary:
    labels_list = list(labels)
    preds_list = list(predictions)
    if len(labels_list) != len(preds_list):
        raise ValueError("predictions and labels must have the same length")
    modulus = mr + 1
    valid_actions = {-1, *range(1, mr + 1)}

    exact_count = 0
    invalid_count = 0
    coarsened_counts = {factor: 0 for factor in factors}
    pred_counter: Counter[int] = Counter()
    confusion: Counter[tuple[int, int]] = Counter()

    for pred, label in zip(preds_list, labels_list):
        if pred not in valid_actions:
            invalid_count += 1
            continue
        pred = int(pred)
        pred_counter[pred] += 1
        exact_count += int(pred == label)
        pred_residue = action_to_residue(pred, modulus)
        label_residue = action_to_residue(label, modulus)
        confusion[(label_residue, pred_residue)] += 1
        for factor in coarsened_counts:
            coarsened_counts[factor] += int(pred_residue % factor == label_residue % factor)

    n = len(labels_list)
    distribution = {action: pred_counter[action] / n for action in sorted(valid_actions)}
    return AccuracySummary(
        exact=exact_count / n if n else 0.0,
        invalid_rate=invalid_count / n if n else 0.0,
        n=n,
        coarsened={factor: count / n for factor, count in coarsened_counts.items()},
        prediction_distribution=distribution,
        confusion=dict(confusion),
    )


def plateau_duration(
    steps: Iterable[int],
    accuracies: Iterable[float],
    target_level: float,
    tolerance: float = 0.03,
) -> int:
    """Approximate duration spent within a target accuracy band."""
    step_list = list(steps)
    acc_list = list(accuracies)
    if len(step_list) != len(acc_list):
        raise ValueError("steps and accuracies must have the same length")
    if len(step_list) < 2:
        return 0
    total = 0
    for left, right, acc in zip(step_list[:-1], step_list[1:], acc_list[:-1]):
        if abs(acc - target_level) <= tolerance:
            total += right - left
    return total


def aggregate_metric_rows(rows: Iterable[dict]) -> dict[str, float]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)):
                buckets[key].append(float(value))
    return {
        f"{key}_mean": sum(values) / len(values)
        for key, values in buckets.items()
        if values
    }

