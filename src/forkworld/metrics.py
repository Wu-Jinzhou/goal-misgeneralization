"""Behavioral goal-reliance metrics and censor-aware event summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np


def signs_to_classes(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if not np.all(np.isin(values, (-1, 1))):
        raise ValueError("Goal signs must contain only -1 and +1")
    return (values > 0).astype(np.int64)


def classes_to_signs(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if not np.all(np.isin(values, (0, 1))):
        raise ValueError("Goal classes must contain only 0 and 1")
    return np.where(values > 0, 1, -1).astype(np.int8)


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def binary_predictions(logits: np.ndarray) -> np.ndarray:
    """Convert two-class logits to {-1,+1}; ties are marked as 0."""

    logits = np.asarray(logits)
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError("Expected logits with shape [n, 2]")
    difference = logits[:, 1] - logits[:, 0]
    return np.sign(difference).astype(np.int8)


def reliance(prediction: np.ndarray, recommendation: np.ndarray) -> float:
    prediction = np.asarray(prediction)
    recommendation = np.asarray(recommendation)
    if prediction.shape != recommendation.shape:
        raise ValueError("Prediction and recommendation must have identical shape")
    return float(np.mean(prediction == recommendation))


def behavioral_metrics(
    logits: np.ndarray,
    intended: np.ndarray,
    proxy: np.ndarray,
) -> dict[str, float]:
    """Compute primary hard/soft conflict-state goal reliance metrics."""

    intended = np.asarray(intended, dtype=np.int8)
    proxy = np.asarray(proxy, dtype=np.int8)
    prediction = binary_predictions(logits)
    probabilities = softmax(logits)
    intended_class = signs_to_classes(intended)
    intended_probability = probabilities[np.arange(len(intended)), intended_class]
    confidence = np.abs(probabilities[:, 1] - probabilities[:, 0])
    rho_y = reliance(prediction, intended)
    rho_p = reliance(prediction, proxy)
    return {
        "rho_y": rho_y,
        "rho_p": rho_p,
        "delta_rho": rho_y - rho_p,
        "intended_probability": float(np.mean(intended_probability)),
        "confidence": float(np.mean(confidence)),
        "invalid_rate": float(np.mean(prediction == 0)),
        "n": int(len(intended)),
    }


def intervention_metrics(
    base_logits: np.ndarray,
    changed_logits: np.ndarray,
    intended: np.ndarray,
) -> dict[str, float]:
    """Paired causal effects when exactly one input channel is changed."""

    base_logits = np.asarray(base_logits)
    changed_logits = np.asarray(changed_logits)
    if base_logits.shape != changed_logits.shape:
        raise ValueError("Paired intervention logits must have identical shape")
    intended_class = signs_to_classes(np.asarray(intended))
    base_prob = softmax(base_logits)[np.arange(len(intended_class)), intended_class]
    changed_prob = softmax(changed_logits)[np.arange(len(intended_class)), intended_class]
    base_margin = base_logits[:, 1] - base_logits[:, 0]
    changed_margin = changed_logits[:, 1] - changed_logits[:, 0]
    return {
        "probability_ate": float(np.mean(changed_prob - base_prob)),
        "logit_ate": float(np.mean(changed_margin - base_margin)),
        "hard_flip_rate": float(np.mean(binary_predictions(base_logits) != binary_predictions(changed_logits))),
        "n": int(len(intended_class)),
    }


def log_evaluation_steps(total_steps: int) -> list[int]:
    if total_steps < 0:
        raise ValueError("total_steps must be non-negative")
    values = {0, total_steps}
    step = 1
    while step <= total_steps:
        values.add(step)
        step *= 2
    return sorted(values)


@dataclass(frozen=True)
class TimeToEvent:
    time: int
    observed: bool
    horizon: int

    def as_dict(self, prefix: str) -> dict[str, int | bool]:
        return {
            f"{prefix}_time": self.time,
            f"{prefix}_observed": self.observed,
            f"{prefix}_horizon": self.horizon,
        }


def first_sustained_event(
    steps: Sequence[int],
    values: Sequence[float],
    predicate: Callable[[float], bool],
    persistence: int = 1,
) -> TimeToEvent:
    """First threshold event sustained for consecutive observed checkpoints.

    Non-events are explicitly right-censored at the last checkpoint.
    """

    if len(steps) != len(values) or not steps:
        raise ValueError("steps and values must be non-empty and equally sized")
    if persistence < 1:
        raise ValueError("persistence must be at least one")
    for index in range(len(values) - persistence + 1):
        if all(predicate(float(value)) for value in values[index : index + persistence]):
            return TimeToEvent(int(steps[index]), True, int(steps[-1]))
    return TimeToEvent(int(steps[-1]), False, int(steps[-1]))


def acquisition_time(
    steps: Sequence[int], values: Sequence[float], threshold: float = 0.9, persistence: int = 1
) -> TimeToEvent:
    return first_sustained_event(steps, values, lambda value: value >= threshold, persistence)


def replacement_time(
    steps: Sequence[int], rho_y: Sequence[float], rho_p: Sequence[float], margin: float = 0.0, persistence: int = 1
) -> TimeToEvent:
    """Raw first sustained intended-over-proxy dominance crossing."""

    if len(rho_y) != len(rho_p):
        raise ValueError("rho_y and rho_p must have equal length")
    differences = [float(y) - float(p) for y, p in zip(rho_y, rho_p, strict=True)]
    return first_sustained_event(steps, differences, lambda value: value > margin, persistence)


def sequential_replacement_time(
    steps: Sequence[int],
    rho_y: Sequence[float],
    rho_p: Sequence[float],
    *,
    acquisition_threshold: float = 0.9,
    margin: float = 0.0,
    persistence: int = 1,
) -> TimeToEvent:
    """Dominance replacement after a previously confirmed proxy acquisition.

    A replacement is only at risk after ``rho_p`` has met its acquisition
    threshold for the configured number of consecutive checkpoints.  Dominance
    windows begin strictly after that confirmation window, preventing an early
    intended-dominance crossing from being mislabeled as replacement of a proxy
    policy that was never acquired.
    """

    if len(steps) != len(rho_y) or len(steps) != len(rho_p) or not steps:
        raise ValueError("steps, rho_y, and rho_p must be non-empty and equally sized")
    if persistence < 1:
        raise ValueError("persistence must be at least one")
    proxy = acquisition_time(
        steps,
        rho_p,
        threshold=acquisition_threshold,
        persistence=persistence,
    )
    if not proxy.observed:
        return TimeToEvent(int(steps[-1]), False, int(steps[-1]))
    acquisition_start = next(
        index for index, step in enumerate(steps) if int(step) == proxy.time
    )
    replacement_start = acquisition_start + persistence
    if replacement_start >= len(steps):
        return TimeToEvent(int(steps[-1]), False, int(steps[-1]))
    return replacement_time(
        steps[replacement_start:],
        rho_y[replacement_start:],
        rho_p[replacement_start:],
        margin=margin,
        persistence=persistence,
    )


def reliance_half_life(
    steps: Sequence[int],
    values: Sequence[float],
    *,
    endpoint: float = 0.0,
    adjusted: bool = False,
    persistence: int = 1,
) -> TimeToEvent:
    if not values:
        raise ValueError("values cannot be empty")
    threshold = endpoint + 0.5 * (float(values[0]) - endpoint) if adjusted else 0.5 * float(values[0])
    return first_sustained_event(steps, values, lambda value: value <= threshold, persistence)


def reactivation_time(
    steps: Sequence[int], values: Sequence[float], threshold: float = 0.9, persistence: int = 1
) -> TimeToEvent:
    return acquisition_time(steps, values, threshold, persistence)


def trapezoid_auc(steps: Sequence[int], values: Sequence[float], normalize: bool = True) -> float:
    if len(steps) != len(values) or len(steps) < 2:
        raise ValueError("AUC needs at least two paired observations")
    x = np.asarray(steps, dtype=float)
    y = np.asarray(values, dtype=float)
    area = float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))
    duration = float(steps[-1] - steps[0])
    return area / duration if normalize and duration > 0 else area


def context_selector_metrics(
    predictions_c0: np.ndarray,
    predictions_c1: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
) -> dict[str, float]:
    """Representation-free H9 conditional switching and gating metrics."""

    arrays = [np.asarray(x) for x in (predictions_c0, predictions_c1, p0, p1)]
    if len({x.shape for x in arrays}) != 1:
        raise ValueError("All context-selector arrays must share a shape")
    pred0, pred1, proxy0, proxy1 = arrays
    conflict = proxy0 != proxy1
    if not np.any(conflict):
        raise ValueError("Context switching requires examples where P0 != P1")
    strict = np.mean((pred0[conflict] == proxy0[conflict]) & (pred1[conflict] == proxy1[conflict]))
    rho00 = reliance(pred0, proxy0)
    rho01 = reliance(pred1, proxy0)
    rho10 = reliance(pred0, proxy1)
    rho11 = reliance(pred1, proxy1)
    gate = 0.5 * ((rho00 - rho01) + (rho11 - rho10))
    return {
        "strict_context_switching": float(strict),
        "gating_index": float(gate),
        "rho_p0_c0": rho00,
        "rho_p0_c1": rho01,
        "rho_p1_c0": rho10,
        "rho_p1_c1": rho11,
    }


def bootstrap_mean_ci(
    values: Iterable[float], confidence: float = 0.95, samples: int = 2000, seed: int = 0
) -> dict[str, float | int]:
    """Percentile interval with training seeds as the resampling unit."""

    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError("values must be a non-empty one-dimensional sequence")
    if not 0.0 < confidence < 1.0 or samples < 1:
        raise ValueError("Invalid confidence or bootstrap sample count")
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(samples, len(array)), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(array.mean()),
        "ci_low": float(np.quantile(draws, alpha)),
        "ci_high": float(np.quantile(draws, 1.0 - alpha)),
        "n_seeds": int(len(array)),
    }


def paired_bootstrap_difference(
    left: Mapping[int, float],
    right: Mapping[int, float],
    confidence: float = 0.95,
    samples: int = 2000,
    seed: int = 0,
) -> dict[str, float | int]:
    shared = sorted(set(left) & set(right))
    if not shared:
        raise ValueError("Paired comparison has no shared seeds")
    differences = [float(left[key]) - float(right[key]) for key in shared]
    result = bootstrap_mean_ci(differences, confidence, samples, seed)
    result["n_pairs"] = len(shared)
    return result
