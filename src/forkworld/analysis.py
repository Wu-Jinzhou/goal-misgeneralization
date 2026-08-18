"""Seed-level aggregation, planned contrasts, and hypothesis reports.

The functions in this module intentionally treat a training seed—not an evaluation
episode—as the experimental replication unit.  Raw prediction rows remain available
for diagnostics, but uncertainty statements are computed from run summaries.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from .artifacts import discover_runs, read_jsonl, write_json
from .metrics import bootstrap_mean_ci

# Inferential labels require independent training-seed replication.  Fewer runs
# remain useful descriptively, but a zero-width resampling interval from a single
# smoke seed must never support or contradict a scientific hypothesis.
MIN_INFERENTIAL_SEEDS = 3
DEFAULT_EQUIVALENCE_MARGIN = 0.05
DEFAULT_BOOTSTRAP_SAMPLES = 4000
DEFAULT_CONFIDENCE = 0.95


@dataclass(frozen=True)
class InferenceSettings:
    """Resolved statistical settings shared by one hypothesis report."""

    equivalence_margin: float = DEFAULT_EQUIVALENCE_MARGIN
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES
    confidence: float = DEFAULT_CONFIDENCE

    def as_dict(self) -> dict[str, float | int]:
        return {
            "equivalence_margin": self.equivalence_margin,
            "bootstrap_samples": self.bootstrap_samples,
            "confidence": self.confidence,
        }


def _validate_inference_settings(settings: InferenceSettings) -> None:
    margin = settings.equivalence_margin
    if not math.isfinite(margin) or not 0.0 <= margin < 1.0:
        raise ValueError("evaluation.equivalence_margin must be finite and lie in [0,1)")
    samples = settings.bootstrap_samples
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("evaluation.bootstrap_samples must be a positive integer")
    confidence = settings.confidence
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("evaluation.confidence must be finite and lie in (0,1)")


def _homogeneous_numeric_setting(
    frame: pd.DataFrame,
    column_name: str,
    default: float | int,
) -> float | int:
    """Resolve one setting and reject pooled artifacts with conflicting values."""

    column = _column(frame, column_name)
    if column is None:
        return default
    raw = frame[column].dropna()
    if raw.empty:
        return default
    numeric = pd.to_numeric(raw, errors="coerce")
    if numeric.isna().any():
        raise ValueError(f"{column_name} must be numeric in every completed run")
    values = sorted({float(value) for value in numeric})
    if len(values) != 1:
        raise ValueError(
            f"cannot pool one hypothesis with conflicting {column_name} values: {values}"
        )
    return values[0]


def _resolve_inference_settings(
    frame: pd.DataFrame,
    *,
    margin: float | None = None,
    bootstrap_samples: int | None = None,
    confidence: float | None = None,
) -> InferenceSettings:
    configured_margin = float(
        _homogeneous_numeric_setting(
            frame,
            "config.evaluation.equivalence_margin",
            DEFAULT_EQUIVALENCE_MARGIN,
        )
    )
    configured_samples_raw = _homogeneous_numeric_setting(
        frame,
        "config.evaluation.bootstrap_samples",
        DEFAULT_BOOTSTRAP_SAMPLES,
    )
    configured_confidence = float(
        _homogeneous_numeric_setting(
            frame,
            "config.evaluation.confidence",
            DEFAULT_CONFIDENCE,
        )
    )
    if float(configured_samples_raw) != int(configured_samples_raw):
        raise ValueError("evaluation.bootstrap_samples must be a positive integer")
    configured_samples = int(configured_samples_raw)

    def checked_override(name: str, configured: float | int, override: float | int | None) -> float | int:
        if override is None:
            return configured
        column = f"config.evaluation.{name}"
        if _column(frame, column) is not None and float(override) != float(configured):
            raise ValueError(
                f"explicit {name}={override} conflicts with resolved {column}={configured}"
            )
        return override

    selected_samples = checked_override(
        "bootstrap_samples", configured_samples, bootstrap_samples
    )
    if (
        isinstance(selected_samples, bool)
        or not isinstance(selected_samples, (int, np.integer))
    ):
        raise ValueError("evaluation.bootstrap_samples must be a positive integer")
    settings = InferenceSettings(
        equivalence_margin=float(
            checked_override("equivalence_margin", configured_margin, margin)
        ),
        bootstrap_samples=int(selected_samples),
        confidence=float(
            checked_override("confidence", configured_confidence, confidence)
        ),
    )
    _validate_inference_settings(settings)
    return settings


def flatten(mapping: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in mapping.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            result.update(flatten(value, name))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            result[name] = value
        elif isinstance(value, list):
            result[name] = json.dumps(value, sort_keys=True)
    return result


def collect_results(root: str | Path, completed_only: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collect run summaries and tidy checkpoint metrics from artifact directories."""

    summaries: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    for run in discover_runs(root, completed_only=completed_only):
        with (run / "resolved_config.yaml").open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        summary_path = run / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
        common = {"run_path": str(run), **flatten(config, "config")}
        summaries.append({**common, **flatten(summary)})
        for record in read_jsonl(run / "metrics.jsonl"):
            metrics.append({**common, **record})
    return pd.DataFrame(summaries), pd.DataFrame(metrics)


def _column(frame: pd.DataFrame, *names: str) -> str | None:
    for name in names:
        if name in frame.columns and frame[name].notna().any():
            return name
    return None


def _spearman(frame: pd.DataFrame, x_name: str | None, y_name: str | None) -> dict[str, Any]:
    if x_name is None or y_name is None:
        return {"status": "unavailable"}
    subset = frame[[x_name, y_name]].dropna()
    if len(subset) < 3 or subset[x_name].nunique() < 2 or subset[y_name].nunique() < 2:
        return {"status": "insufficient_data", "n": int(len(subset))}
    estimate, pvalue = stats.spearmanr(subset[x_name].astype(float), subset[y_name].astype(float))
    return {"status": "estimated", "rho": float(estimate), "pvalue": float(pvalue), "n": int(len(subset))}


def _stratified_spearman(
    frame: pd.DataFrame,
    x_name: str | None,
    y_name: str | None,
    strata: tuple[str, ...],
) -> dict[str, Any]:
    """Report associations within matched design strata instead of pooling confounds."""

    if x_name is None or y_name is None:
        return {"status": "unavailable"}
    keys = [name for name in strata if name in frame and frame[name].notna().any()]
    pieces = frame.groupby(keys, dropna=False) if keys else [((), frame)]
    records: list[dict[str, Any]] = []
    for values, group in pieces:
        values = values if isinstance(values, tuple) else (values,)
        result = _spearman(group, x_name, y_name)
        records.append(
            {
                "stratum": {
                    key.removeprefix("config.").removeprefix("data."): _native(value)
                    for key, value in zip(keys, values, strict=True)
                },
                **result,
            }
        )
    estimated = [record for record in records if record.get("status") == "estimated"]
    if not estimated:
        return {"status": "insufficient_data", "strata": records}
    estimates = np.asarray([float(record["rho"]) for record in estimated])
    return {
        "status": "estimated",
        "n_strata": len(records),
        "n_estimated_strata": len(estimated),
        "median_rho": float(np.median(estimates)),
        "positive_strata_fraction": float(np.mean(estimates > 0.0)),
        "strata": records,
    }


def _seedwise_spearman(
    frame: pd.DataFrame,
    x_name: str | None,
    y_name: str | None,
    *,
    bootstrap_seed: int,
    settings: InferenceSettings,
) -> dict[str, Any]:
    """Estimate a monotone trend within each seed, then infer across seeds."""

    seed_name = _column(frame, "config.seed", "seed")
    if seed_name is None or x_name is None or y_name is None:
        return {"status": "unavailable"}
    estimates: list[float] = []
    for _, group in frame.groupby(seed_name, dropna=False):
        association = _spearman(group, x_name, y_name)
        if association.get("status") == "estimated":
            estimates.append(float(association["rho"]))
    if len(estimates) < MIN_INFERENTIAL_SEEDS:
        return {
            "status": "insufficient_data",
            "n_seeds": len(estimates),
            "minimum_inferential_seeds": MIN_INFERENTIAL_SEEDS,
            "reason": (
                f"at least {MIN_INFERENTIAL_SEEDS} seeds with estimable "
                "within-seed trends are required"
            ),
        }
    summary = bootstrap_mean_ci(
        estimates,
        confidence=settings.confidence,
        samples=settings.bootstrap_samples,
        seed=bootstrap_seed,
    )
    return {
        "status": "estimated",
        "rho": summary["mean"],
        "ci_low": summary["ci_low"],
        "ci_high": summary["ci_high"],
        "n_seeds": summary["n_seeds"],
        "bootstrap_samples": settings.bootstrap_samples,
        "confidence": settings.confidence,
    }


def _seedwise_collapsed_spearman(
    frame: pd.DataFrame,
    x_name: str | None,
    y_name: str | None,
    *,
    bootstrap_seed: int,
    settings: InferenceSettings,
) -> dict[str, Any]:
    """Collapse balanced factorial cells at each seed/x level before inference."""

    seed_name = _column(frame, "config.seed", "seed")
    if seed_name is None or x_name is None or y_name is None:
        return {"status": "unavailable"}
    compact = frame[[seed_name, x_name, y_name]].copy()
    compact[x_name] = pd.to_numeric(compact[x_name], errors="coerce")
    compact[y_name] = pd.to_numeric(compact[y_name], errors="coerce")
    collapsed = (
        compact.dropna()
        .groupby([seed_name, x_name], dropna=False, as_index=False)[y_name]
        .mean()
    )
    estimates: list[dict[str, Any]] = []
    for seed, group in collapsed.groupby(seed_name, dropna=False):
        association = _spearman(group, x_name, y_name)
        if association.get("status") == "estimated":
            estimates.append(
                {
                    "seed": _native(seed),
                    "rho": float(association["rho"]),
                    "n_x_levels": int(group[x_name].nunique()),
                }
            )
    summary = _difference_summary(
        np.asarray([entry["rho"] for entry in estimates], dtype=float),
        seed=bootstrap_seed,
        settings=settings,
    )
    result = {
        **summary,
        "rho": summary.get("mean"),
        "replication_unit": "training_seed",
        "factorial_collapse": "mean outcome within each seed and x level",
        "n_collapsed_rows": int(len(collapsed)),
        "per_seed_associations": estimates,
    }
    return result


def _directional_trend_status(
    summary: Mapping[str, Any],
    *,
    direction: int,
    equivalence_margin: float,
) -> str:
    """Classify a seed-level directional trend with conservative equivalence."""

    if summary.get("status") != "estimated":
        return "insufficient_data"
    low = float(summary.get("ci_low", -np.inf))
    high = float(summary.get("ci_high", np.inf))
    if (direction > 0 and low > 0.0) or (direction < 0 and high < 0.0):
        return "consistent"
    if (direction > 0 and high < 0.0) or (direction < 0 and low > 0.0):
        return "evidence_against"
    if low >= -equivalence_margin and high <= equivalence_margin:
        return "evidence_against"
    return "mixed_or_inconclusive"


def _combine_required_statuses(statuses: Sequence[str]) -> str:
    values = list(statuses)
    if values and all(value == "consistent" for value in values):
        return "consistent"
    if "evidence_against" in values:
        return "evidence_against"
    if values and all(value == "insufficient_data" for value in values):
        return "insufficient_data"
    return "mixed_or_inconclusive"


def _seed_ci(
    frame: pd.DataFrame,
    value: str,
    *,
    settings: InferenceSettings,
    seed: str = "config.seed",
) -> dict[str, Any]:
    if value not in frame or frame[value].dropna().empty:
        return {"status": "unavailable"}
    seed_column = seed if seed in frame else _column(frame, "config.seed", "seed")
    values = frame.groupby(seed_column)[value].mean().to_numpy() if seed_column else frame[value].to_numpy()
    return _difference_summary(values, seed=611, settings=settings)


def _paired_condition_difference(
    frame: pd.DataFrame,
    condition_column: str,
    left: str,
    right: str,
    value_column: str,
    *,
    settings: InferenceSettings,
) -> dict[str, Any]:
    seed = _column(frame, "config.seed", "seed")
    if seed is None or any(column not in frame for column in (condition_column, value_column)):
        return {"status": "unavailable"}
    index_columns = [seed]
    for candidate in (
        "config.data.n_train",
        "config.data.k",
        "config.h4.n_conflict",
        "config.model.width",
        "config.update.budget",
        "config.h5.nuisance_entropy",
        "config.h8.n0",
    ):
        if candidate in frame and frame[candidate].notna().all():
            index_columns.append(candidate)
    pivot = frame.pivot_table(index=index_columns, columns=condition_column, values=value_column, aggfunc="mean")
    if left not in pivot or right not in pivot:
        return {"status": "insufficient_data"}
    differences = (pivot[left] - pivot[right]).dropna()
    if differences.empty:
        return {"status": "insufficient_data"}
    # The training seed is the replicate. Average planned factorial cells within
    # seed before resampling so k/N_conflict cells are not pseudoreplicates.
    seed_differences = differences.groupby(level=seed).mean().to_numpy()
    result = _difference_summary(seed_differences, seed=719, settings=settings)
    result.update(
        {
            "n_paired_cells": int(len(differences)),
            "n_paired_seeds": int(len(seed_differences)),
        }
    )
    return result


def _kaplan_meier(times: np.ndarray, observed: np.ndarray) -> dict[str, Any]:
    """Dependency-free Kaplan–Meier summary for right-censored event times."""

    times = np.asarray(times, dtype=float)
    observed = np.asarray(observed, dtype=bool)
    keep = np.isfinite(times)
    times, observed = times[keep], observed[keep]
    if not len(times):
        return {"status": "insufficient_data"}
    at_risk = len(times)
    survival = 1.0
    curve: list[dict[str, float | int]] = []
    median = None
    for event_time in np.unique(times):
        mask = times == event_time
        events = int(np.sum(observed[mask]))
        censored = int(np.sum(~observed[mask]))
        if events:
            survival *= 1.0 - events / at_risk
        curve.append(
            {
                "time": float(event_time),
                "survival": float(survival),
                "events": events,
                "censored": censored,
            }
        )
        if median is None and survival <= 0.5:
            median = float(event_time)
        at_risk -= events + censored
    return {
        "status": "estimated",
        "n": int(len(times)),
        "events": int(observed.sum()),
        "censor_fraction": float(np.mean(~observed)),
        "median": median,
        "curve": curve,
    }


def _h5_percentile_interval(
    values: np.ndarray,
    *,
    confidence: float,
    requested_draws: int,
) -> dict[str, Any]:
    """Summarize an already-computed paired bootstrap distribution."""

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {
            "status": "right_censored",
            "bootstrap_draws": int(requested_draws),
            "usable_draws": 0,
            "censored_draws": int(requested_draws),
            "usable_fraction": 0.0,
        }
    alpha = (1.0 - confidence) / 2.0
    return {
        "status": "estimated",
        "bootstrap_median": float(np.median(finite)),
        "ci_low": float(np.quantile(finite, alpha)),
        "ci_high": float(np.quantile(finite, 1.0 - alpha)),
        "confidence": float(confidence),
        "bootstrap_draws": int(requested_draws),
        "usable_draws": int(len(finite)),
        "censored_draws": int(requested_draws - len(finite)),
        "usable_fraction": float(len(finite) / requested_draws),
    }


def _h5_threshold_values(
    success_rates: np.ndarray,
    budgets: np.ndarray,
    *,
    required_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return first successful budget and its observation indicator."""

    reached = success_rates + 1e-12 >= required_fraction
    observed = reached.any(axis=-1)
    indices = reached.argmax(axis=-1)
    values = np.asarray(budgets, dtype=float)[indices]
    return np.where(observed, values, np.nan), observed


def _h5_log_ratio_slopes(
    entropies: np.ndarray,
    log_ratios: np.ndarray,
    *,
    minimum_entropies: int = 3,
) -> np.ndarray:
    """Fit within-draw slopes while retaining only identified entropy ratios."""

    values = np.atleast_2d(np.asarray(log_ratios, dtype=float))
    x = np.asarray(entropies, dtype=float)[None, :]
    valid = np.isfinite(values)
    count = valid.sum(axis=1)
    x_sum = np.where(valid, x, 0.0).sum(axis=1)
    y_sum = np.where(valid, values, 0.0).sum(axis=1)
    xy_sum = np.where(valid, x * values, 0.0).sum(axis=1)
    xx_sum = np.where(valid, x * x, 0.0).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        denominator = xx_sum - x_sum * x_sum / count
        slopes = (xy_sum - x_sum * y_sum / count) / denominator
    usable = (count >= minimum_entropies) & (denominator > 0)
    return np.where(usable, slopes, np.nan)


def _h5_pair_threshold_bootstrap(
    table: pd.DataFrame,
    numerator: str,
    denominator: str,
    *,
    bootstrap_samples: int,
    confidence: float,
    success_fraction: float,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Recompute update thresholds under matched training-seed resampling.

    The input is pre-collapsed to one success indicator per
    seed x algorithm x entropy x budget cell.  One NumPy seed-index matrix is
    reused for every algorithm, entropy, and budget, preserving the paired
    factorial design without rebuilding pandas groupbys for each draw.
    """

    methods = (numerator, denominator)
    selected = table[table["_algorithm"].isin(methods)].copy()
    entropies = np.sort(selected["_entropy"].unique().astype(float))
    budgets = np.sort(selected["_budget"].unique().astype(float))
    seeds = np.sort(selected["_seed"].unique())
    if not len(entropies) or not len(budgets) or not len(seeds):
        return {"status": "insufficient_data", "matched_seed_count": 0}

    complete_index = pd.MultiIndex.from_product(
        [seeds, methods, entropies, budgets],
        names=["_seed", "_algorithm", "_entropy", "_budget"],
    )
    indexed = selected.set_index(
        ["_seed", "_algorithm", "_entropy", "_budget"]
    )["_success"]
    duplicate_cells = bool(indexed.index.has_duplicates)
    if duplicate_cells:
        indexed = indexed.groupby(level=list(range(4))).mean()
    complete = indexed.reindex(complete_index).to_numpy(dtype=float).reshape(
        len(seeds), len(methods), len(entropies), len(budgets)
    )
    matched_mask = np.isfinite(complete).all(axis=(1, 2, 3))
    matched_seeds = seeds[matched_mask]
    success = complete[matched_mask].astype(bool, copy=False)
    base: dict[str, Any] = {
        "comparison": f"{numerator}_over_{denominator}",
        "numerator": numerator,
        "denominator": denominator,
        "replication_unit": "matched training seed",
        "matched_seed_count": int(len(matched_seeds)),
        "matched_seeds": [_native(value) for value in matched_seeds],
        "minimum_inferential_seeds": MIN_INFERENTIAL_SEEDS,
        "entropies": [float(value) for value in entropies],
        "update_budgets": [
            int(value) if float(value).is_integer() else float(value) for value in budgets
        ],
        "duplicate_seed_cells_collapsed": duplicate_cells,
    }
    if len(matched_seeds) < MIN_INFERENTIAL_SEEDS:
        return {
            **base,
            "status": "insufficient_data",
            "reason": (
                f"at least {MIN_INFERENTIAL_SEEDS} seeds complete across both algorithms, "
                "all entropy levels, and all update budgets are required"
            ),
        }

    full_thresholds, full_observed = _h5_threshold_values(
        success.mean(axis=0), budgets, required_fraction=success_fraction
    )
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(
        0,
        len(matched_seeds),
        size=(bootstrap_samples, len(matched_seeds)),
    )
    draw_rates = success[indices].mean(axis=1)
    draw_thresholds, draw_observed = _h5_threshold_values(
        draw_rates, budgets, required_fraction=success_fraction
    )

    threshold_records: list[dict[str, Any]] = []
    for method_index, method in enumerate(methods):
        for entropy_index, entropy_value in enumerate(entropies):
            value = full_thresholds[method_index, entropy_index]
            threshold_records.append(
                {
                    "algorithm": method,
                    "nuisance_entropy": float(entropy_value),
                    "u_min": (
                        int(value) if np.isfinite(value) and float(value).is_integer()
                        else float(value) if np.isfinite(value)
                        else None
                    ),
                    "right_censored": bool(
                        not full_observed[method_index, entropy_index]
                    ),
                    "success_rate_at_max_budget": float(
                        success[:, method_index, entropy_index, -1].mean()
                    ),
                }
            )

    ratio_records: list[dict[str, Any]] = []
    with np.errstate(divide="ignore", invalid="ignore"):
        draw_ratios = draw_thresholds[:, 0, :] / draw_thresholds[:, 1, :]
        full_ratios = full_thresholds[0] / full_thresholds[1]
    maximum_budget = float(budgets[-1])
    for entropy_index, entropy_value in enumerate(entropies):
        numerator_observed = draw_observed[:, 0, entropy_index]
        denominator_observed = draw_observed[:, 1, entropy_index]
        both_observed = numerator_observed & denominator_observed
        numerator_only_censored = ~numerator_observed & denominator_observed
        denominator_only_censored = numerator_observed & ~denominator_observed
        both_censored = ~numerator_observed & ~denominator_observed
        point_numerator = full_thresholds[0, entropy_index]
        point_denominator = full_thresholds[1, entropy_index]
        point_observed = bool(
            full_observed[0, entropy_index] and full_observed[1, entropy_index]
        )
        interval = _h5_percentile_interval(
            draw_ratios[:, entropy_index],
            confidence=confidence,
            requested_draws=bootstrap_samples,
        )
        ratio_records.append(
            {
                "nuisance_entropy": float(entropy_value),
                "ratio": float(full_ratios[entropy_index]) if point_observed else None,
                "numerator_u_min": (
                    float(point_numerator) if np.isfinite(point_numerator) else None
                ),
                "denominator_u_min": (
                    float(point_denominator) if np.isfinite(point_denominator) else None
                ),
                "right_censored": not point_observed,
                "ratio_lower_bound": (
                    float(maximum_budget / point_denominator)
                    if not np.isfinite(point_numerator)
                    and np.isfinite(point_denominator)
                    else None
                ),
                "ratio_upper_bound": (
                    float(point_numerator / maximum_budget)
                    if np.isfinite(point_numerator)
                    and not np.isfinite(point_denominator)
                    else None
                ),
                **interval,
                "censoring_by_draw": {
                    "both_thresholds_observed": int(both_observed.sum()),
                    "numerator_only_censored": int(numerator_only_censored.sum()),
                    "denominator_only_censored": int(denominator_only_censored.sum()),
                    "both_censored": int(both_censored.sum()),
                },
            }
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        draw_log_ratios = np.log2(draw_ratios)
        full_log_ratios = np.log2(full_ratios)
    draw_slopes = _h5_log_ratio_slopes(entropies, draw_log_ratios)
    full_slope = _h5_log_ratio_slopes(entropies, full_log_ratios)[0]
    trend = {
        "estimand": (
            "OLS slope of log2(U_min(numerator)/U_min(denominator)) on the integer "
            "count of active fair nuisance branch factors"
        ),
        "minimum_identified_entropy_levels_per_draw": 3,
        "slope": float(full_slope) if np.isfinite(full_slope) else None,
        **_h5_percentile_interval(
            draw_slopes,
            confidence=confidence,
            requested_draws=bootstrap_samples,
        ),
    }
    return {
        **base,
        "status": "estimated",
        "bootstrap": {
            "method": "matched-seed nonparametric bootstrap",
            "samples": int(bootstrap_samples),
            "confidence": float(confidence),
            "seed": int(bootstrap_seed),
            "threshold_recomputed_in_every_draw": True,
        },
        "thresholds": threshold_records,
        "entropy_ratio_intervals": ratio_records,
        "directional_log2_ratio_trend": trend,
    }


def _h5_high_entropy_interval(pair: Mapping[str, Any]) -> dict[str, Any]:
    intervals = list(pair.get("entropy_ratio_intervals", []))
    if not intervals:
        return {"status": "insufficient_data"}
    return max(intervals, key=lambda item: float(item["nuisance_entropy"]))


def _h5_update_thresholds(
    frame: pd.DataFrame,
    rho_y: str | None,
    *,
    settings: InferenceSettings,
) -> dict[str, Any]:
    """H5 matched-seed threshold bootstrap and preregistered falsifiers."""

    algorithm = _column(frame, "config.h5.algorithm", "algorithm")
    entropy = _column(frame, "config.h5.nuisance_entropy", "config.h5.entropy_bits")
    budget = _column(frame, "model.trainable_parameters", "costs.trainable_scalars")
    seed = _column(frame, "config.seed", "seed")
    if any(value is None for value in (algorithm, entropy, budget, seed, rho_y)):
        return {"status": "unavailable"}
    assert algorithm and entropy and budget and seed and rho_y
    columns = [algorithm, entropy, budget, seed, rho_y]
    table = frame[columns].dropna().copy()
    table[budget] = pd.to_numeric(table[budget], errors="coerce")
    table[entropy] = pd.to_numeric(table[entropy], errors="coerce")
    table[rho_y] = pd.to_numeric(table[rho_y], errors="coerce")
    table = table.dropna(subset=[budget, entropy, rho_y])
    if table.empty:
        return {"status": "insufficient_data"}

    success_threshold_column = _column(frame, "config.evaluation.acquisition_threshold")
    success_threshold = (
        float(pd.to_numeric(frame[success_threshold_column], errors="coerce").dropna().iloc[0])
        if success_threshold_column
        and not pd.to_numeric(frame[success_threshold_column], errors="coerce").dropna().empty
        else 0.9
    )
    collapsed = (
        table.groupby([seed, algorithm, entropy, budget], as_index=False)[rho_y]
        .mean()
        .rename(
            columns={
                seed: "_seed",
                algorithm: "_algorithm",
                entropy: "_entropy",
                budget: "_budget",
            }
        )
    )
    collapsed["_success"] = collapsed[rho_y].astype(float) >= success_threshold

    pair_arguments = {
        "bootstrap_samples": settings.bootstrap_samples,
        "confidence": settings.confidence,
        "success_fraction": 0.8,
    }
    primary = _h5_pair_threshold_bootstrap(
        collapsed,
        "trajectory_sft",
        "rl",
        bootstrap_seed=5501,
        **pair_arguments,
    )
    clean_pair = _h5_pair_threshold_bootstrap(
        collapsed,
        "clean_sft",
        "rl",
        bootstrap_seed=5503,
        **pair_arguments,
    )
    on_policy_pair = _h5_pair_threshold_bootstrap(
        collapsed,
        "on_policy_imitation",
        "rl",
        bootstrap_seed=5507,
        **pair_arguments,
    )

    primary_high = _h5_high_entropy_interval(primary)
    clean_high = _h5_high_entropy_interval(clean_pair)
    on_policy_high = _h5_high_entropy_interval(on_policy_pair)
    minimum_control_usable_fraction = 0.8
    clean_usable = bool(
        clean_high.get("status") == "estimated"
        and clean_high.get("usable_fraction", 0.0) >= minimum_control_usable_fraction
    )
    primary_usable = bool(
        primary_high.get("status") == "estimated"
        and primary_high.get("usable_fraction", 0.0) >= minimum_control_usable_fraction
    )
    on_policy_usable = bool(
        on_policy_high.get("status") == "estimated"
        and on_policy_high.get("usable_fraction", 0.0)
        >= minimum_control_usable_fraction
    )
    clean_status = (
        "falsifier_triggered"
        if clean_usable and clean_high.get("ci_low", -np.inf) > 1.0
        else "control_passed"
        if clean_usable and clean_high.get("ci_high", np.inf) <= 1.0
        else "inconclusive"
    )
    primary_equivalent = bool(
        primary_usable
        and primary_high.get("ci_low", -np.inf)
        >= 1.0 - settings.equivalence_margin
        and primary_high.get("ci_high", np.inf)
        <= 1.0 + settings.equivalence_margin
    )
    primary_no_advantage = bool(
        primary_usable and primary_high.get("ci_high", np.inf) < 1.0
    )
    primary_advantage = bool(
        primary_usable
        and primary_high.get("ci_low", -np.inf) > 1.0
        and not primary_equivalent
    )
    on_policy_equivalent = bool(
        on_policy_usable
        and on_policy_high.get("ci_low", -np.inf)
        >= 1.0 - settings.equivalence_margin
        and on_policy_high.get("ci_high", np.inf)
        <= 1.0 + settings.equivalence_margin
    )
    on_policy_status = (
        "falsifier_triggered"
        if primary_advantage and on_policy_equivalent
        else "control_passed"
        if on_policy_usable
        and on_policy_high.get("ci_low", -np.inf)
        > 1.0 + settings.equivalence_margin
        else "inconclusive"
    )
    clean_control = {
        "status": clean_status,
        "falsifier": (
            "a CI-supported RL update-capacity advantage persists when SFT receives "
            "minimal deterministic labels"
        ),
        "decision_rule": "falsified when high-entropy CI for U_clean_SFT/U_RL is above 1",
        "minimum_usable_bootstrap_fraction": minimum_control_usable_fraction,
        "high_entropy_interval": clean_high,
        "pair_analysis": clean_pair,
    }
    on_policy_control = {
        "status": on_policy_status,
        "falsifier": (
            "on-policy imitation fully reproduces RL's advantage over trajectory SFT"
        ),
        "decision_rule": (
            "when the primary high-entropy RL advantage is CI-supported, falsified if "
            "the full CI for U_on_policy/U_RL lies within 1 +/- margin"
        ),
        "ratio_equivalence_margin": settings.equivalence_margin,
        "minimum_usable_bootstrap_fraction": minimum_control_usable_fraction,
        "primary_high_entropy_rl_advantage": primary_advantage,
        "high_entropy_interval": on_policy_high,
        "pair_analysis": on_policy_pair,
    }
    primary_high_entropy_claim = {
        "status": (
            "consistent"
            if primary_advantage
            else "evidence_against"
            if primary_no_advantage or primary_equivalent
            else "mixed_or_inconclusive"
        ),
        "claim": "U_min(trajectory SFT) / U_min(RL) is meaningfully above 1",
        "interval": primary_high,
        "ratio_equivalence_margin": settings.equivalence_margin,
        "ci_supported_advantage": primary_advantage,
        "precise_no_advantage": primary_no_advantage,
        "equivalent_to_no_advantage": primary_equivalent,
        "minimum_usable_bootstrap_fraction": minimum_control_usable_fraction,
    }
    trend = primary.get("directional_log2_ratio_trend", {})
    entropy_intervals = primary.get("entropy_ratio_intervals", [])
    maximum_censored_fraction = 0.2
    censor_fractions = [
        1.0 - float(item.get("usable_fraction", 0.0)) for item in entropy_intervals
    ]
    trend_censor_fraction = 1.0 - float(trend.get("usable_fraction", 0.0))
    censor_heavy = bool(
        primary.get("status") == "estimated"
        and (
            trend_censor_fraction > maximum_censored_fraction
            or any(value > maximum_censored_fraction for value in censor_fractions)
        )
    )
    return {
        "status": primary.get("status", "insufficient_data"),
        "success_rule": (
            f"U_min is the first actor update budget where >=80% of matched seeds reach "
            f"rho_y>={success_threshold:g}"
        ),
        "nuisance_entropy_definition": "integer count of active fair branch factors",
        "primary_comparison": primary,
        # Compatibility aliases keep downstream report readers simple while
        # making the inferential unit and censoring explicit in the nested data.
        "thresholds": primary.get("thresholds", []),
        "ratios": primary.get("entropy_ratio_intervals", []),
        "entropy_ratio_association": trend,
        "primary_high_entropy_claim": primary_high_entropy_claim,
        "controls_and_falsifiers": {
            "clean_sft": clean_control,
            "on_policy_imitation": on_policy_control,
        },
        "censoring_assessment": {
            "censor_heavy": censor_heavy,
            "maximum_allowed_censored_draw_fraction": maximum_censored_fraction,
            "trend_censored_draw_fraction": trend_censor_fraction,
            "entropy_ratio_censored_draw_fractions": censor_fractions,
            "decision": (
                "insufficient_data"
                if primary.get("status") != "estimated"
                else "conservative_inference"
                if censor_heavy
                else "usable"
            ),
        },
    }


def _native(value: Any) -> Any:
    """Convert pandas/numpy scalars used in analysis tables to JSON-safe scalars."""

    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


def _configured_threshold(
    frame: pd.DataFrame, names: tuple[str, ...], default: float
) -> tuple[float, list[float]]:
    column = _column(frame, *names)
    if column is None:
        return float(default), [float(default)]
    values = sorted({float(value) for value in frame[column].dropna()})
    return (values[0] if values else float(default)), values or [float(default)]


def _seed_fraction_step(
    rows: pd.DataFrame,
    time_column: str | None,
    observed_column: str | None,
    fraction: float,
) -> dict[str, Any]:
    """Earliest step by which the requested fraction of seeds acquired a decoder."""

    if time_column is None or observed_column is None:
        return {"value": None, "observed": False, "status": "unavailable"}
    usable = rows[[time_column, observed_column]].dropna(subset=[time_column]).copy()
    if usable.empty:
        return {"value": None, "observed": False, "status": "unavailable"}
    required = int(np.ceil(fraction * len(usable)))
    event_times = np.sort(
        usable.loc[usable[observed_column].eq(True), time_column].astype(float).to_numpy()
    )
    if len(event_times) >= required:
        value = float(event_times[max(0, required - 1)])
        return {
            "value": value,
            "observed": True,
            "status": "estimated",
            "required_seed_count": required,
            "n_seeds": int(len(usable)),
        }
    return {
        "value": float(usable[time_column].astype(float).max()),
        "observed": False,
        "status": "right_censored",
        "required_seed_count": required,
        "observed_seed_count": int(len(event_times)),
        "n_seeds": int(len(usable)),
    }


def _h2_capacity_rows(
    frame: pd.DataFrame,
    *,
    signal: str,
    regime_kind: str,
    capacity_column: str | None,
    capacity_name: str,
    accuracy_column: str | None,
    event_time: str | None,
    event_observed: str | None,
    accuracy_threshold: float,
    seed_fraction: float,
) -> list[dict[str, Any]]:
    if capacity_column is None or accuracy_column is None or frame.empty:
        return []
    architecture_keys = (
        "config.data.k",
        "config.update.mode",
        "config.model.depth",
        "config.model.activation",
        "config.model.residual",
    )
    if regime_kind == "update":
        architecture_keys += ("config.model.width",)
    group_keys = [
        key for key in architecture_keys if key in frame and frame[key].notna().any()
    ]
    pieces = frame.groupby(group_keys, dropna=False) if group_keys else [((), frame)]
    records: list[dict[str, Any]] = []
    for values, group in pieces:
        values = values if isinstance(values, tuple) else (values,)
        tested: list[dict[str, Any]] = []
        capacities = pd.to_numeric(group[capacity_column], errors="coerce")
        for capacity in sorted(capacities.dropna().unique()):
            at_capacity = group[capacities == capacity]
            success = (
                at_capacity[event_observed].eq(True)
                if event_observed is not None and event_observed in at_capacity
                else pd.to_numeric(at_capacity[accuracy_column], errors="coerce")
                >= accuracy_threshold
            )
            tested.append(
                {
                    "capacity": float(capacity),
                    "success_fraction": float(success.mean()),
                    "n_seeds": int(len(at_capacity)),
                    "rows": at_capacity,
                }
            )
        if not tested:
            continue
        crossing = next(
            (entry for entry in tested if entry["success_fraction"] >= seed_fraction), None
        )
        step_source = crossing or tested[-1]
        steps = _seed_fraction_step(
            step_source["rows"], event_time, event_observed, seed_fraction
        )
        record: dict[str, Any] = {
            "signal": signal,
            "regime_kind": regime_kind,
            capacity_name: None if crossing is None else int(crossing["capacity"]),
            f"{capacity_name}_right_censored": crossing is None,
            "maximum_tested_capacity": int(tested[-1]["capacity"]),
            "success_fraction_at_threshold": (
                None if crossing is None else float(crossing["success_fraction"])
            ),
            "C_steps": steps["value"],
            "C_steps_observed": bool(steps["observed"]),
            "C_steps_status": steps["status"],
            "C_steps_capacity": int(step_source["capacity"]),
            "n_seeds_at_C_steps": int(steps.get("n_seeds", len(step_source["rows"]))),
            "capacity_curve": [
                {
                    "capacity": int(entry["capacity"]),
                    "success_fraction": entry["success_fraction"],
                    "n_seeds": entry["n_seeds"],
                }
                for entry in tested
            ],
            "stratum": {
                key.removeprefix("config."): _native(value)
                for key, value in zip(group_keys, values, strict=True)
            },
        }
        records.append(record)
    return records


def _h2_calibration_behavior(
    frame: pd.DataFrame,
    accuracy_threshold: float,
    seed_fraction: float,
    *,
    settings: InferenceSettings,
) -> list[dict[str, Any]]:
    """Relate cell-level independent calibration success to competition behavior."""

    mode_column = _column(frame, "config.update.mode")
    seed_column = _column(frame, "config.seed", "seed")
    if mode_column is None or seed_column is None:
        return []
    cell_keys = [
        key
        for key in (
            "config.data.k",
            "config.model.width",
            "config.model.depth",
            "config.model.activation",
            "config.model.residual",
            "config.update.mode",
            "config.update.budget",
            "config._case_index",
        )
        if key in frame and frame[key].notna().any()
    ]
    specifications = {
        "proxy": (
            _column(
                frame,
                "final.proxy_calibration_iid.target_accuracy",
                "final.proxy_decoder_accuracy",
            ),
            _column(frame, "final.competition_conflict.rho_p", "final.rho_p"),
            _column(frame, "events.proxy_decoder_acquisition_observed"),
        ),
        "exact": (
            _column(
                frame,
                "final.exact_calibration_iid.target_accuracy",
                "final.exact_decoder_accuracy",
                "final.decoder_accuracy",
            ),
            _column(frame, "final.competition_conflict.rho_y", "final.rho_y"),
            _column(frame, "events.exact_decoder_acquisition_observed"),
        ),
    }
    results: list[dict[str, Any]] = []
    for mode, regime in frame.groupby(mode_column, dropna=False):
        regime_kind = "update" if str(mode) == "subspace" else "architecture"
        for signal, (accuracy, behavior, crossing) in specifications.items():
            if accuracy is None or behavior is None:
                continue
            useful = regime.dropna(subset=[behavior] + ([crossing] if crossing else [accuracy])).copy()
            if useful.empty:
                continue
            useful["_cell"] = useful.groupby(cell_keys, dropna=False).ngroup()
            rates = (
                useful.groupby("_cell")[crossing].apply(
                    lambda values: float(values.eq(True).mean())
                )
                if crossing
                else useful.groupby("_cell")[accuracy].apply(
                    lambda values: float(
                        np.mean(pd.to_numeric(values, errors="coerce") >= accuracy_threshold)
                    )
                )
            )
            useful["_crossed"] = useful["_cell"].map(rates.ge(seed_fraction))
            by_seed = (
                useful.groupby([seed_column, "_crossed"], dropna=False)[behavior]
                .mean()
                .unstack("_crossed")
            )
            crossed = by_seed[True].dropna() if True in by_seed else pd.Series(dtype=float)
            not_crossed = by_seed[False].dropna() if False in by_seed else pd.Series(dtype=float)
            paired = by_seed.dropna(subset=[True, False]) if {True, False} <= set(by_seed.columns) else pd.DataFrame()
            results.append(
                {
                    "signal": signal,
                    "regime_kind": regime_kind,
                    "update_mode": _native(mode),
                    "behavior_metric": behavior,
                    "n_cells": int(len(rates)),
                    "crossed_cells": int(rates.ge(seed_fraction).sum()),
                    "not_crossed_cells": int(rates.lt(seed_fraction).sum()),
                    "behavior_when_crossed": _difference_summary(
                        crossed, seed=2201, settings=settings
                    ),
                    "behavior_when_not_crossed": _difference_summary(
                        not_crossed, seed=2203, settings=settings
                    ),
                    "paired_crossed_minus_not": (
                        _difference_summary(
                            paired[True] - paired[False],
                            seed=2207,
                            settings=settings,
                        )
                        if not paired.empty
                        else {"status": "insufficient_data"}
                    ),
                    "n_paired_seeds": int(len(paired)),
                }
            )
    return results


def _h2_matched_architecture_contrasts(
    frame: pd.DataFrame,
    tolerance: float,
    *,
    settings: InferenceSettings,
) -> dict[str, Any]:
    """Contrast architecture families at exact or declared-tolerance parameter matches."""

    if not 0.0 <= tolerance <= 1.0:
        raise ValueError("h2.parameter_match_tolerance must lie in [0,1]")
    seed = _column(frame, "config.seed", "seed")
    parameters = _column(frame, "model.exact_calibration.total_parameters")
    accuracy = _column(
        frame,
        "final.exact_calibration_iid.target_accuracy",
        "final.exact_decoder_accuracy",
        "final.decoder_accuracy",
    )
    behavior = _column(frame, "final.competition_conflict.rho_y", "final.rho_y")
    architecture_columns = (
        "config.model.depth",
        "config.model.activation",
        "config.model.residual",
    )
    if (
        seed is None
        or parameters is None
        or accuracy is None
        or behavior is None
        or any(column not in frame for column in architecture_columns)
    ):
        return {
            "status": "unavailable",
            "declared_relative_tolerance": tolerance,
            "matched": [],
            "unmatched": [],
        }
    strata = [
        column
        for column in ("config.data.k", "config.update.mode", "config.data.q")
        if column in frame and frame[column].notna().any()
    ]
    width = _column(frame, "config.model.width")
    clean = frame.dropna(
        subset=[seed, parameters, accuracy, behavior, *architecture_columns]
    ).copy()
    matched: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    pieces = clean.groupby(strata, dropna=False) if strata else [((), clean)]
    for stratum_values, group in pieces:
        stratum_values = (
            stratum_values if isinstance(stratum_values, tuple) else (stratum_values,)
        )
        stratum = {
            key.removeprefix("config."): _native(value)
            for key, value in zip(strata, stratum_values, strict=True)
        }
        architecture_keys = sorted(
            {
                tuple(_native(row[column]) for column in architecture_columns)
                for _, row in group.iterrows()
            },
            key=lambda values: tuple(str(value) for value in values),
        )
        for left_key, right_key in combinations(architecture_keys, 2):
            left_mask = np.logical_and.reduce(
                [group[column].eq(value) for column, value in zip(architecture_columns, left_key, strict=True)]
            )
            right_mask = np.logical_and.reduce(
                [group[column].eq(value) for column, value in zip(architecture_columns, right_key, strict=True)]
            )
            left = group[left_mask]
            right = group[right_mask]
            left_counts = sorted(pd.to_numeric(left[parameters], errors="coerce").dropna().unique())
            right_counts = sorted(pd.to_numeric(right[parameters], errors="coerce").dropna().unique())
            candidates = [
                (
                    abs(float(left_count) - float(right_count))
                    / max(float(left_count), float(right_count)),
                    float(left_count),
                    float(right_count),
                )
                for left_count in left_counts
                for right_count in right_counts
                if max(float(left_count), float(right_count)) > 0
            ]
            architecture_pair = {
                "left_architecture": {
                    column.removeprefix("config.model."): value
                    for column, value in zip(architecture_columns, left_key, strict=True)
                },
                "right_architecture": {
                    column.removeprefix("config.model."): value
                    for column, value in zip(architecture_columns, right_key, strict=True)
                },
                "stratum": stratum,
                "declared_relative_tolerance": tolerance,
            }
            if not candidates:
                unmatched.append({**architecture_pair, "reason": "no_parameter_counts"})
                continue
            gap, left_count, right_count = min(candidates)
            count_details = {
                "left_parameters": int(left_count),
                "right_parameters": int(right_count),
                "relative_parameter_gap": float(gap),
                "match_kind": "exact" if gap == 0.0 else "within_tolerance",
            }
            if gap > tolerance:
                unmatched.append(
                    {
                        **architecture_pair,
                        **count_details,
                        "match_kind": "unmatched",
                        "reason": "closest_pair_exceeds_tolerance",
                    }
                )
                continue
            left_selected = left[pd.to_numeric(left[parameters], errors="coerce").eq(left_count)]
            right_selected = right[pd.to_numeric(right[parameters], errors="coerce").eq(right_count)]
            left_seed = left_selected.groupby(seed, dropna=False)[[accuracy, behavior]].mean()
            right_seed = right_selected.groupby(seed, dropna=False)[[accuracy, behavior]].mean()
            paired = left_seed.join(right_seed, how="inner", lsuffix="_left", rsuffix="_right")
            if paired.empty:
                unmatched.append(
                    {
                        **architecture_pair,
                        **count_details,
                        "match_kind": "unmatched",
                        "reason": "no_shared_seeds_at_selected_counts",
                    }
                )
                continue
            widths: dict[str, list[Any]] = {}
            if width:
                widths = {
                    "left_widths": sorted({_native(value) for value in left_selected[width].dropna()}),
                    "right_widths": sorted({_native(value) for value in right_selected[width].dropna()}),
                }
            matched.append(
                {
                    **architecture_pair,
                    **count_details,
                    **widths,
                    "n_paired_seeds": int(len(paired)),
                    "left_minus_right_exact_decoder_accuracy": _difference_summary(
                        paired[f"{accuracy}_left"] - paired[f"{accuracy}_right"],
                        seed=2231,
                        settings=settings,
                    ),
                    "left_minus_right_competition_rho_y": _difference_summary(
                        paired[f"{behavior}_left"] - paired[f"{behavior}_right"],
                        seed=2237,
                        settings=settings,
                    ),
                }
            )
    return {
        "status": "estimated" if matched else "insufficient_data",
        "matching_unit": "total model parameters",
        "architecture_identity": [
            column.removeprefix("config.model.") for column in architecture_columns
        ],
        "capacity_coordinate": "model width",
        "declared_relative_tolerance": tolerance,
        "matched": matched,
        "unmatched": unmatched,
        "n_matched_pairs": len(matched),
        "n_unmatched_pairs": len(unmatched),
    }


def _h2_interface_audit(frame: pd.DataFrame) -> dict[str, Any]:
    input_flag = _column(frame, "data.calibration_input_dim_matches_competition")
    parameter_flag = _column(
        frame, "data.calibration_parameter_count_matches_competition"
    )
    trainable_flag = _column(
        frame, "data.calibration_trainable_count_matches_competition"
    )
    if input_flag is None or parameter_flag is None or trainable_flag is None:
        return {
            "status": "unavailable",
            "reason": "runs predate explicit full-interface calibration logging",
        }
    valid = (
        frame[input_flag].eq(True)
        & frame[parameter_flag].eq(True)
        & frame[trainable_flag].eq(True)
    )
    return {
        "status": "verified" if bool(valid.all()) else "failed",
        "n_runs": int(len(frame)),
        "n_matched_runs": int(valid.sum()),
        "n_mismatched_runs": int((~valid).sum()),
        "state_retained_but_neutralized": True,
        "non_target_channels_neutralized": True,
    }


def _h2_complexity_report(
    frame: pd.DataFrame, *, settings: InferenceSettings
) -> dict[str, Any]:
    accuracy_threshold, accuracy_values = _configured_threshold(
        frame,
        ("config.h2.target_accuracy", "data.target_accuracy"),
        0.95,
    )
    seed_fraction, fraction_values = _configured_threshold(
        frame, ("config.h2.target_seed_fraction",), 0.8
    )
    mode_column = _column(frame, "config.update.mode")
    if mode_column is None:
        return {"status": "unavailable"}
    architecture = frame[frame[mode_column].astype(str) != "subspace"]
    update = frame[frame[mode_column].astype(str) == "subspace"]
    signals = {
        "proxy": {
            "accuracy": _column(
                frame,
                "final.proxy_calibration_iid.target_accuracy",
                "final.proxy_decoder_accuracy",
            ),
            "parameters": _column(frame, "model.proxy_calibration.total_parameters"),
            "updates": _column(frame, "model.proxy_calibration.trainable_parameters"),
            "time": _column(frame, "events.proxy_decoder_acquisition_time"),
            "observed": _column(frame, "events.proxy_decoder_acquisition_observed"),
        },
        "exact": {
            "accuracy": _column(
                frame,
                "final.exact_calibration_iid.target_accuracy",
                "final.exact_decoder_accuracy",
                "final.decoder_accuracy",
            ),
            "parameters": _column(frame, "model.exact_calibration.total_parameters"),
            "updates": _column(frame, "model.exact_calibration.trainable_parameters"),
            "time": _column(frame, "events.exact_decoder_acquisition_time"),
            "observed": _column(frame, "events.exact_decoder_acquisition_observed"),
        },
    }
    parameter_rows: dict[str, list[dict[str, Any]]] = {}
    update_rows: dict[str, list[dict[str, Any]]] = {}
    for signal, columns in signals.items():
        parameter_rows[signal] = _h2_capacity_rows(
            architecture,
            signal=signal,
            regime_kind="architecture",
            capacity_column=columns["parameters"],
            capacity_name="C_param",
            accuracy_column=columns["accuracy"],
            event_time=columns["time"],
            event_observed=columns["observed"],
            accuracy_threshold=accuracy_threshold,
            seed_fraction=seed_fraction,
        )
        update_rows[signal] = _h2_capacity_rows(
            update,
            signal=signal,
            regime_kind="update",
            capacity_column=columns["updates"],
            capacity_name="C_update",
            accuracy_column=columns["accuracy"],
            event_time=columns["time"],
            event_observed=columns["observed"],
            accuracy_threshold=accuracy_threshold,
            seed_fraction=seed_fraction,
        )
    tolerance, tolerance_values = _configured_threshold(
        frame, ("config.h2.parameter_match_tolerance",), 0.05
    )
    architecture_contrasts = _h2_matched_architecture_contrasts(
        architecture, tolerance, settings=settings
    )
    return {
        "status": "estimated" if any(parameter_rows.values()) or any(update_rows.values()) else "insufficient_data",
        "accuracy_threshold": accuracy_threshold,
        "seed_fraction_threshold": seed_fraction,
        "configured_accuracy_thresholds": accuracy_values,
        "configured_seed_fraction_thresholds": fraction_values,
        "parameter_match_tolerance": tolerance,
        "configured_parameter_match_tolerances": tolerance_values,
        "calibration_interface_audit": _h2_interface_audit(frame),
        "matched_parameter_architecture_contrasts": architecture_contrasts[
            "matched"
        ],
        "unmatched_parameter_architecture_pairs": architecture_contrasts[
            "unmatched"
        ],
        "architecture_matching": architecture_contrasts,
        "C_param": parameter_rows,
        "C_update": update_rows,
        "C_steps": {
            "architecture": {
                signal: [
                    {
                        "stratum": row["stratum"],
                        "C_steps": row["C_steps"],
                        "observed": row["C_steps_observed"],
                        "at_C_param": row["C_steps_capacity"],
                    }
                    for row in rows
                ]
                for signal, rows in parameter_rows.items()
            },
            "update": {
                signal: [
                    {
                        "stratum": row["stratum"],
                        "C_steps": row["C_steps"],
                        "observed": row["C_steps_observed"],
                        "at_C_update": row["C_steps_capacity"],
                    }
                    for row in rows
                ]
                for signal, rows in update_rows.items()
            },
        },
        "calibration_crossing_behavior": _h2_calibration_behavior(
            frame, accuracy_threshold, seed_fraction, settings=settings
        ),
    }


def _h3_trajectory_report(
    summaries: pd.DataFrame,
    metrics: pd.DataFrame | None,
    *,
    dominance_margin: float,
    settings: InferenceSettings,
) -> dict[str, Any]:
    if metrics is None or metrics.empty:
        return {"status": "unavailable", "reason": "checkpoint metrics were not supplied"}
    experiment = _column(metrics, "experiment", "config.experiment.hypothesis")
    data = metrics[metrics[experiment].astype(str) == "h3"].copy() if experiment else metrics.copy()
    if data.empty or not {"split", "metric", "value", "global_step"} <= set(data.columns):
        return {"status": "unavailable", "reason": "H3 trajectory columns are missing"}
    if "level" in data:
        data = data[data["level"].astype(str) == "choice"]
    if "intervention" in data:
        data = data[data["intervention"].astype(str) == "none"]
    run_key = next(
        (key for key in ("run_path", "run_id", "seed") if key in data and key in summaries),
        None,
    )
    if run_key is None:
        return {"status": "unavailable", "reason": "no shared run identifier"}

    def series(split: str, metric: str, name: str) -> pd.DataFrame:
        selected = data[(data["split"] == split) & (data["metric"] == metric)]
        return (
            selected.groupby([run_key, "global_step"], dropna=False)["value"]
            .mean()
            .rename(name)
            .reset_index()
        )

    reward = series("iid", "target_accuracy", "reward")
    rho_y = series("conflict", "rho_y", "rho_y")
    rho_p = series("conflict", "rho_p", "rho_p")
    trajectory = reward.merge(rho_y, on=[run_key, "global_step"], how="inner").merge(
        rho_p, on=[run_key, "global_step"], how="inner"
    )
    if trajectory.empty:
        return {"status": "unavailable", "reason": "aligned reward/goal trajectories are missing"}
    lookup = summaries.drop_duplicates(run_key).set_index(run_key)
    seed_name = _column(summaries, "config.seed", "seed")
    records: list[dict[str, Any]] = []
    for identifier, group in trajectory.groupby(run_key, dropna=False):
        if identifier not in lookup.index:
            continue
        summary = lookup.loc[identifier]
        group = group.sort_values("global_step")
        steps = group["global_step"].astype(int).to_numpy()
        rewards = group["reward"].astype(float).to_numpy()
        intended = group["rho_y"].astype(float).to_numpy()
        proxy = group["rho_p"].astype(float).to_numpy()
        tolerance = float(summary.get("config.h3.reward_plateau_tolerance", 0.005))
        window = int(summary.get("config.h3.reward_plateau_window", 4))
        recorded_minimum = summary.get("events.reward_plateau_minimum_accuracy")
        configured_minimum = summary.get("config.h3.reward_plateau_min_accuracy")
        realized_q = summary.get("data.realized_q", summary.get("config.data.q", 0.9))
        if pd.notna(recorded_minimum):
            minimum = float(recorded_minimum)
            minimum_source = "events.reward_plateau_minimum_accuracy"
        elif pd.notna(configured_minimum):
            minimum = float(configured_minimum)
            minimum_source = "config.h3.reward_plateau_min_accuracy"
        else:
            minimum = max(0.55, float(realized_q) - 0.02)
            minimum_source = "derived_from_realized_q"
        plateau_time = int(steps[-1])
        plateau_observed = False
        if window >= 2:
            for start in range(0, len(steps) - window + 1):
                segment = rewards[start : start + window]
                flat = float(np.max(segment) - np.min(segment)) <= tolerance
                at_terminal_level = abs(float(np.mean(segment)) - float(rewards[-1])) <= tolerance
                if float(np.min(segment)) >= minimum and flat and at_terminal_level:
                    plateau_time = int(steps[start])
                    plateau_observed = True
                    break

        acquisition_threshold = float(summary.get("config.evaluation.acquisition_threshold", 0.9))
        persistence = int(summary.get("config.evaluation.persistence", 2))
        difference = intended - proxy
        final_direction = 1 if difference[-1] > dominance_margin else -1 if difference[-1] < -dominance_margin else 0
        stabilization_time = int(steps[-1])
        stabilization_observed = False
        if final_direction:
            dominant = intended if final_direction > 0 else proxy
            for start in range(0, len(steps)):
                enough_points = len(steps) - start >= persistence
                same_goal = np.all(final_direction * difference[start:] > dominance_margin)
                active_goal = np.all(dominant[start:] >= acquisition_threshold)
                if enough_points and same_goal and active_goal:
                    stabilization_time = int(steps[start])
                    stabilization_observed = True
                    break

        order: bool | None = None
        if plateau_observed and stabilization_observed:
            order = plateau_time < stabilization_time
        elif plateau_observed and plateau_time < stabilization_time:
            # Stabilization is right-censored at the horizon, so its true time is later.
            order = True
        elif stabilization_observed and stabilization_time < plateau_time:
            # Plateau is right-censored at the horizon, so it cannot precede stabilization.
            order = False
        records.append(
            {
                "run": _native(identifier),
                "seed": _native(summary.get(seed_name)) if seed_name else None,
                "reward_plateau_time": plateau_time,
                "reward_plateau_observed": plateau_observed,
                "goal_stabilization_time": stabilization_time,
                "goal_stabilization_observed": stabilization_observed,
                "plateau_before_goal_stabilization": order,
                "meaningful_reward_floor": minimum,
                "meaningful_reward_floor_source": minimum_source,
                "final_goal": "intended" if final_direction > 0 else "proxy" if final_direction < 0 else "unresolved",
            }
        )
    if not records:
        return {"status": "insufficient_data"}
    table = pd.DataFrame(records)
    identifiable = table["plateau_before_goal_stabilization"].notna()
    if identifiable.any():
        identifiable_rows = table.loc[identifiable].copy()
        identifiable_rows["_ordering"] = identifiable_rows[
            "plateau_before_goal_stabilization"
        ].astype(bool)
        if seed_name and identifiable_rows["seed"].notna().any():
            fractions = identifiable_rows.groupby("seed")["_ordering"].mean().to_numpy()
        else:
            fractions = identifiable_rows["_ordering"].astype(float).to_numpy()
        fraction_summary = _difference_summary(
            fractions, seed=3023, settings=settings
        )
        fraction = fraction_summary.get("mean")
    else:
        fractions = np.asarray([], dtype=float)
        fraction_summary = {"status": "insufficient_data"}
        fraction = None
    return {
        "status": "estimated",
        "definition": (
            "first configured-length flat IID-accuracy window above the meaningful reward floor "
            "and within tolerance of terminal reward, versus the first checkpoint after which "
            "the final acquired goal remains dominant"
        ),
        "initial_low_plateaus_rejected": True,
        "n_runs": int(len(table)),
        "n_seeds": int(table["seed"].dropna().nunique()) if seed_name else None,
        "n_identifiable_orderings": int(identifiable.sum()),
        "plateau_before_goal_stabilization_fraction": fraction,
        "plateau_before_goal_stabilization_seed_bootstrap": fraction_summary,
        "reward_plateau_survival": _kaplan_meier(
            table["reward_plateau_time"].to_numpy(),
            table["reward_plateau_observed"].to_numpy(),
        ),
        "goal_stabilization_survival": _kaplan_meier(
            table["goal_stabilization_time"].to_numpy(),
            table["goal_stabilization_observed"].to_numpy(),
        ),
        "runs": records,
    }


def _difference_summary(
    values: pd.Series | np.ndarray,
    *,
    seed: int,
    settings: InferenceSettings,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"status": "insufficient_data"}
    if len(array) < MIN_INFERENTIAL_SEEDS:
        return {
            "status": "insufficient_data",
            "mean": float(np.mean(array)),
            "n_seeds": int(len(array)),
            "minimum_inferential_seeds": MIN_INFERENTIAL_SEEDS,
            "reason": "too few independent training seeds for an uncertainty interval",
        }
    return {
        "status": "estimated",
        **bootstrap_mean_ci(
            array,
            confidence=settings.confidence,
            samples=settings.bootstrap_samples,
            seed=seed,
        ),
        "bootstrap_samples": settings.bootstrap_samples,
        "confidence": settings.confidence,
    }


def _h7_stratified_half_lives(
    frame: pd.DataFrame,
    condition: str,
    time: str,
    observed: str,
    *,
    settings: InferenceSettings,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    strata = [condition]
    for candidate in ("config.h7.q_a", "config.h7.q_b", "config.model.width"):
        if candidate in frame:
            strata.append(candidate)
    survival: list[dict[str, Any]] = []
    for values, group in frame.groupby(strata, dropna=False):
        values = values if isinstance(values, tuple) else (values,)
        entry = {str(key): value for key, value in zip(strata, values, strict=True)}
        entry["kaplan_meier"] = _kaplan_meier(group[time].to_numpy(), group[observed].to_numpy())
        survival.append(entry)

    seed_column = _column(frame, "config.seed", "seed")
    if seed_column is None:
        return survival, {"status": "unavailable"}
    match_keys = [seed_column]
    for candidate in (
        "config.h7.q_a",
        "config.model.width",
        "config.model.depth",
        "config.update.mode",
        "config.update.budget",
    ):
        if candidate in frame:
            match_keys.append(candidate)
    base = frame[match_keys + [condition, time] + (["config.h7.q_b"] if "config.h7.q_b" in frame else [])]
    removal = base[base[condition] == "removal"].groupby(match_keys, dropna=False)[time].mean()
    decorrelation = base[base[condition] == "decorrelation"].groupby(match_keys, dropna=False)[time].mean()
    shared = removal.to_frame("removal").join(decorrelation.to_frame("decorrelation"), how="inner")
    removal_difference = shared["decorrelation"] - shared["removal"]
    if seed_column in removal_difference.index.names:
        removal_difference = removal_difference.groupby(level=seed_column).mean()
    ordering: dict[str, Any] = {
        "decorrelation_minus_removal_restricted_time": _difference_summary(
            removal_difference, seed=1701, settings=settings
        )
        if not shared.empty
        else {"status": "insufficient_data"},
        "reversal_minus_decorrelation_by_q_b": [],
        "note": "Censored horizons are retained as restricted event times; reversal strengths are not pooled.",
    }
    reversals = base[base[condition] == "reversal"]
    q_column = "config.h7.q_b" if "config.h7.q_b" in reversals else None
    pieces = reversals.groupby(q_column, dropna=False) if q_column else [(None, reversals)]
    for q_b, group in pieces:
        reversal = group.groupby(match_keys, dropna=False)[time].mean()
        paired = reversal.to_frame("reversal").join(
            decorrelation.to_frame("decorrelation"), how="inner"
        )
        reversal_difference = paired["reversal"] - paired["decorrelation"]
        if seed_column in reversal_difference.index.names:
            reversal_difference = reversal_difference.groupby(level=seed_column).mean()
        ordering["reversal_minus_decorrelation_by_q_b"].append(
            {
                "q_b": None if q_b is None or pd.isna(q_b) else float(q_b),
                "contrast": _difference_summary(
                    reversal_difference, seed=1703, settings=settings
                )
                if not paired.empty
                else {"status": "insufficient_data"},
            }
        )
    return survival, ordering


def _h7_restoration_ordering(
    frame: pd.DataFrame,
    condition: str,
    restored_reliance: str,
    *,
    settings: InferenceSettings,
) -> dict[str, Any]:
    """Pair final restored-proxy reliance within seed and design cell."""

    seed_column = _column(frame, "config.seed", "seed")
    if seed_column is None or restored_reliance not in frame:
        return {"status": "unavailable"}
    match_keys = [seed_column]
    for candidate in (
        "config.h7.q_a",
        "config.model.width",
        "config.model.depth",
        "config.update.mode",
        "config.update.budget",
    ):
        if candidate in frame:
            match_keys.append(candidate)
    columns = match_keys + [condition, restored_reliance]
    if "config.h7.q_b" in frame:
        columns.append("config.h7.q_b")
    base = frame[columns].copy()
    removal = base[base[condition] == "removal"].groupby(
        match_keys, dropna=False
    )[restored_reliance].mean()
    decorrelation = base[base[condition] == "decorrelation"].groupby(
        match_keys, dropna=False
    )[restored_reliance].mean()
    shared = removal.to_frame("removal").join(
        decorrelation.to_frame("decorrelation"), how="inner"
    )
    decorrelation_difference = shared["decorrelation"] - shared["removal"]
    if seed_column in decorrelation_difference.index.names:
        decorrelation_difference = decorrelation_difference.groupby(
            level=seed_column
        ).mean()
    result: dict[str, Any] = {
        "status": "estimated",
        "restored_rho_p_by_condition": {
            str(name): _seed_ci(
                group, restored_reliance, settings=settings, seed=seed_column
            )
            for name, group in frame.groupby(condition, dropna=False)
        },
        "decorrelation_minus_removal": (
            _difference_summary(
                decorrelation_difference, seed=1711, settings=settings
            )
            if len(decorrelation_difference)
            else {"status": "insufficient_data"}
        ),
        "reversal_minus_decorrelation_by_q_b": [],
        "note": (
            "Negative contrasts mean less dormant old-proxy rebound after the "
            "stronger counterevidence arm. Each seed contributes one averaged "
            "paired contrast per reported comparison."
        ),
    }
    reversals = base[base[condition] == "reversal"]
    q_column = "config.h7.q_b" if "config.h7.q_b" in reversals else None
    pieces = reversals.groupby(q_column, dropna=False) if q_column else [(None, reversals)]
    for q_b, group in pieces:
        reversal = group.groupby(match_keys, dropna=False)[restored_reliance].mean()
        paired = reversal.to_frame("reversal").join(
            decorrelation.to_frame("decorrelation"), how="inner"
        )
        difference = paired["reversal"] - paired["decorrelation"]
        if seed_column in difference.index.names:
            difference = difference.groupby(level=seed_column).mean()
        result["reversal_minus_decorrelation_by_q_b"].append(
            {
                "q_b": None if q_b is None or pd.isna(q_b) else float(q_b),
                "contrast": (
                    _difference_summary(
                        difference, seed=1717, settings=settings
                    )
                    if len(difference)
                    else {"status": "insufficient_data"}
                ),
            }
        )
    return result


def _h8_history_contrasts(
    frame: pd.DataFrame,
    value_column: str,
    *,
    observed_column: str | None = None,
    settings: InferenceSettings,
) -> list[dict[str, Any]]:
    history = _column(frame, "config.h8.history")
    seed_column = _column(frame, "config.seed", "seed")
    if history is None or seed_column is None or value_column not in frame:
        return []
    match_keys = [seed_column]
    for candidate in (
        "config.h8.perturbation",
        "config.h8.stage1_mode",
        "config.model.width",
        "config.model.depth",
        "config.update.mode",
        "config.update.budget",
    ):
        if candidate in frame:
            match_keys.append(candidate)
    n0_column = _column(frame, "config.h8.n0")
    columns = match_keys + [history, value_column]
    if n0_column and n0_column not in columns:
        columns.append(n0_column)
    if observed_column and observed_column in frame:
        columns.append(observed_column)
    compact = frame[columns].copy()
    controls = compact[compact[history] == "control"].groupby(match_keys, dropna=False).agg(
        control_value=(value_column, "mean"),
        **(
            {"control_observed": (observed_column, "mean")}
            if observed_column and observed_column in compact
            else {}
        ),
    )
    records: list[dict[str, Any]] = []
    grouping = [column for column in (n0_column, "config.h8.perturbation", "config.h8.stage1_mode") if column]
    for treatment in ("old_goal", "compute_matched_sham"):
        treated = frame[frame[history] == treatment].copy()
        if treated.empty:
            continue
        joined = treated.merge(controls.reset_index(), on=match_keys, how="inner")
        if joined.empty:
            continue
        joined["difference"] = joined[value_column].astype(float) - joined["control_value"].astype(float)
        pieces = joined.groupby(grouping, dropna=False) if grouping else [((), joined)]
        for values, group in pieces:
            values = values if isinstance(values, tuple) else (values,)
            seed_differences = group.groupby(seed_column, dropna=False)["difference"].mean()
            entry: dict[str, Any] = {
                "treatment": treatment,
                "minus": "control",
                "contrast": _difference_summary(
                    seed_differences, seed=1801, settings=settings
                ),
                "n_pairs": int(len(group)),
                "n_seeds": int(len(seed_differences)),
            }
            entry.update(
                {
                    str(key): (None if pd.isna(value) else _native(value))
                    for key, value in zip(grouping, values, strict=True)
                }
            )
            if observed_column and observed_column in group and "control_observed" in group:
                entry["treated_event_fraction"] = float(group[observed_column].eq(True).mean())
                entry["control_event_fraction"] = float(group["control_observed"].mean())
            records.append(entry)

    # The compute-matched sham is the strongest control for a history-specific
    # effect: unlike the no-stage0 control it matches N0 optimization exposure.
    direct_keys = [*match_keys, *([n0_column] if n0_column else [])]
    old = compact[compact[history] == "old_goal"].groupby(
        direct_keys, dropna=False
    ).agg(
        old_value=(value_column, "mean"),
        **(
            {"old_observed": (observed_column, "mean")}
            if observed_column and observed_column in compact
            else {}
        ),
    )
    sham = compact[compact[history] == "compute_matched_sham"].groupby(
        direct_keys, dropna=False
    ).agg(
        sham_value=(value_column, "mean"),
        **(
            {"sham_observed": (observed_column, "mean")}
            if observed_column and observed_column in compact
            else {}
        ),
    )
    direct = old.join(sham, how="inner").reset_index()
    if not direct.empty:
        direct["difference"] = direct["old_value"] - direct["sham_value"]
        pieces = direct.groupby(grouping, dropna=False) if grouping else [((), direct)]
        for values, group in pieces:
            values = values if isinstance(values, tuple) else (values,)
            seed_differences = group.groupby(seed_column, dropna=False)["difference"].mean()
            entry = {
                "treatment": "old_goal",
                "minus": "compute_matched_sham",
                "contrast": _difference_summary(
                    seed_differences, seed=1811, settings=settings
                ),
                "n_pairs": int(len(group)),
                "n_seeds": int(len(seed_differences)),
            }
            entry.update(
                {
                    str(key): (None if pd.isna(value) else _native(value))
                    for key, value in zip(grouping, values, strict=True)
                }
            )
            if "old_observed" in group and "sham_observed" in group:
                entry["treated_event_fraction"] = float(group["old_observed"].mean())
                entry["control_event_fraction"] = float(group["sham_observed"].mean())
            records.append(entry)
    return records


def _h8_scaling_report(
    old_frame: pd.DataFrame,
    rebound: str,
    reactivation: str | None,
    *,
    margin: float,
    settings: InferenceSettings,
) -> dict[str, Any]:
    """Estimate preregistered N0/model-size trends with seed as replicate."""

    stage1_mode = _column(old_frame, "config.h8.stage1_mode", "data.stage1_mode")
    fixed = (
        old_frame[old_frame[stage1_mode].astype(str).eq("fixed")].copy()
        if stage1_mode
        else old_frame
    )
    analysis_frame = fixed if not fixed.empty else old_frame
    analysis_mode = "fixed" if not fixed.empty and stage1_mode else "all_available"
    n0 = _column(analysis_frame, "data.realized_n0", "config.h8.n0")
    capacity = _column(
        analysis_frame,
        "model.total_parameters",
        "model.trainable_parameters",
        "config.model.width",
    )
    n0_rebound = _seedwise_collapsed_spearman(
        analysis_frame,
        n0,
        rebound,
        bootstrap_seed=1823,
        settings=settings,
    )
    n0_reactivation = (
        _seedwise_collapsed_spearman(
            analysis_frame,
            n0,
            reactivation,
            bootstrap_seed=1829,
            settings=settings,
        )
        if reactivation
        else {"status": "unavailable"}
    )
    capacity_rebound = _seedwise_collapsed_spearman(
        analysis_frame,
        capacity,
        rebound,
        bootstrap_seed=1831,
        settings=settings,
    )
    capacity_reactivation = (
        _seedwise_collapsed_spearman(
            analysis_frame,
            capacity,
            reactivation,
            bootstrap_seed=1847,
            settings=settings,
        )
        if reactivation
        else {"status": "unavailable"}
    )
    n0_rebound_status = _directional_trend_status(
        n0_rebound, direction=1, equivalence_margin=margin
    )
    n0_reactivation_status = _directional_trend_status(
        n0_reactivation, direction=-1, equivalence_margin=margin
    )
    capacity_rebound_status = _directional_trend_status(
        capacity_rebound, direction=1, equivalence_margin=margin
    )
    capacity_reactivation_status = _directional_trend_status(
        capacity_reactivation, direction=-1, equivalence_margin=margin
    )
    return {
        "old_goal_volume": {
            "status": _combine_required_statuses(
                [n0_rebound_status, n0_reactivation_status]
            ),
            "n0_to_rebound": {
                "status": n0_rebound_status,
                "association": n0_rebound,
            },
            "n0_to_reactivation_restricted_time": {
                "status": n0_reactivation_status,
                "association": n0_reactivation,
            },
        },
        "model_capacity": {
            # Stronger rebound is the preregistered required prediction. Faster
            # reactivation is reported as a corroborating, optional contrast.
            "status": capacity_rebound_status,
            "capacity_to_rebound": {
                "status": capacity_rebound_status,
                "association": capacity_rebound,
            },
            "capacity_to_reactivation_restricted_time_optional": {
                "status": capacity_reactivation_status,
                "association": capacity_reactivation,
            },
        },
        "factorial_handling": (
            "within each seed and the fixed-volume primary analysis, balanced "
            "perturbation/N0/capacity nuisance cells are averaged at each focal x "
            "level before Spearman estimation"
        ),
        "primary_stage1_mode": analysis_mode,
    }


def _h8_primary_eligibility(frame: pd.DataFrame) -> pd.Series:
    """Reconstruct H8 eligibility without applying match bands to fixed-N1 runs."""

    gate = _column(frame, "final.stage1_gate_passed")
    recorded = _column(frame, "final.eligible_for_primary_analysis")
    eligible = (
        frame[gate].eq(True)
        if gate
        else frame[recorded].eq(True)
        if recorded
        else pd.Series(True, index=frame.index)
    )
    mode = _column(frame, "config.h8.stage1_mode", "data.stage1_mode")
    band = _column(frame, "final.stage1_in_match_band")
    if mode and band:
        normalized_mode = frame[mode].astype(str).str.lower().replace(
            {"matched": "behavior_matched"}
        )
        eligible &= normalized_mode.ne("behavior_matched") | frame[band].eq(True)
    return eligible


def _highest_capacity_per_seed(
    frame: pd.DataFrame, capacity: str | None, seed_column: str
) -> pd.DataFrame:
    if capacity is None or capacity not in frame or frame.empty:
        return frame.iloc[0:0].copy()
    numeric = pd.to_numeric(frame[capacity], errors="coerce")
    maxima = numeric.groupby(frame[seed_column], dropna=False).transform("max")
    return frame[numeric.notna() & np.isclose(numeric, maxima)].copy()


def _seed_margin_summary(
    frame: pd.DataFrame,
    value: str | None,
    seed_column: str,
    *,
    threshold_column: str | None,
    default_threshold: float,
    bootstrap_seed: int,
    settings: InferenceSettings,
) -> dict[str, Any]:
    if value is None or value not in frame or frame.empty:
        return {"status": "insufficient_data"}
    values = pd.to_numeric(frame[value], errors="coerce")
    thresholds = (
        pd.to_numeric(frame[threshold_column], errors="coerce").fillna(default_threshold)
        if threshold_column and threshold_column in frame
        else pd.Series(default_threshold, index=frame.index, dtype=float)
    )
    compact = pd.DataFrame(
        {seed_column: frame[seed_column], "margin_above_threshold": values - thresholds}
    ).dropna()
    per_seed = compact.groupby(seed_column, dropna=False)[
        "margin_above_threshold"
    ].mean()
    summary = _difference_summary(
        per_seed.to_numpy(), seed=bootstrap_seed, settings=settings
    )
    summary.update(
        {
            "threshold_source": threshold_column or f"default={default_threshold:g}",
            "per_seed_margin": [
                {"seed": _native(seed), "margin": float(value)}
                for seed, value in per_seed.items()
            ],
        }
    )
    return summary


def _h9_high_capacity_report(
    panel: pd.DataFrame,
    *,
    capacity_column: str | None,
    seed_column: str,
    switching: str | None,
    normal_accuracy: str | None,
    mismatch_obedience: str | None,
    causal_selectivity: str | None,
    selector_gate: str | None,
    margin: float,
    bootstrap_seed: int,
    settings: InferenceSettings,
) -> dict[str, Any]:
    if panel.empty:
        return {"status": "insufficient_data", "reason": "capacity panel is empty"}
    selector_threshold = _column(panel, "config.h9.selector_threshold")
    mastery_threshold = _column(panel, "config.h9.mastery_threshold")
    selector_margin = _seed_margin_summary(
        panel,
        switching,
        seed_column,
        threshold_column=selector_threshold,
        default_threshold=0.9,
        bootstrap_seed=bootstrap_seed,
        settings=settings,
    )
    normal_margin = _seed_margin_summary(
        panel,
        normal_accuracy,
        seed_column,
        threshold_column=mastery_threshold,
        default_threshold=0.9,
        bootstrap_seed=bootstrap_seed + 2,
        settings=settings,
    )
    gate_frame = panel.copy()
    if selector_gate and selector_gate in gate_frame:
        gate_frame["_h9_selector_gate"] = gate_frame[selector_gate].eq(True).astype(float)
        gate_source = selector_gate
    elif switching:
        thresholds = (
            pd.to_numeric(gate_frame[selector_threshold], errors="coerce").fillna(0.9)
            if selector_threshold
            else pd.Series(0.9, index=gate_frame.index)
        )
        gate_frame["_h9_selector_gate"] = (
            pd.to_numeric(gate_frame[switching], errors="coerce") >= thresholds
        ).astype(float)
        gate_source = "derived_from_strict_context_switching"
    else:
        gate_source = "unavailable"
    gate = _seed_ci(
        gate_frame,
        "_h9_selector_gate",
        settings=settings,
        seed=seed_column,
    )
    gate["source"] = gate_source
    mismatch = (
        _seed_ci(
            panel, mismatch_obedience, settings=settings, seed=seed_column
        )
        if mismatch_obedience
        else {"status": "insufficient_data"}
    )
    causal = (
        _seed_ci(
            panel, causal_selectivity, settings=settings, seed=seed_column
        )
        if causal_selectivity
        else {"status": "insufficient_data"}
    )

    def threshold_status(summary: Mapping[str, Any]) -> str:
        if summary.get("status") != "estimated":
            return "insufficient_data"
        if summary.get("ci_low", -np.inf) >= 0.0:
            return "consistent"
        if summary.get("ci_high", np.inf) < 0.0:
            return "evidence_against"
        return "mixed_or_inconclusive"

    def gate_status(summary: Mapping[str, Any]) -> str:
        if summary.get("status") != "estimated":
            return "insufficient_data"
        if summary.get("ci_low", -np.inf) > 0.5:
            return "consistent"
        if summary.get("ci_high", np.inf) <= 0.5:
            return "evidence_against"
        return "mixed_or_inconclusive"

    def positive_effect_status(summary: Mapping[str, Any]) -> str:
        if summary.get("status") != "estimated":
            return "insufficient_data"
        low = float(summary.get("ci_low", -np.inf))
        high = float(summary.get("ci_high", np.inf))
        if low > margin:
            return "consistent"
        if high <= 0.0 or (low >= -margin and high <= margin):
            return "evidence_against"
        return "mixed_or_inconclusive"

    selector_score_status = threshold_status(selector_margin)
    selector_gate_status = gate_status(gate)
    selector_status = _combine_required_statuses(
        [selector_score_status, selector_gate_status]
    )
    normal_status = threshold_status(normal_margin)
    mismatch_status = positive_effect_status(mismatch)
    causal_status = positive_effect_status(causal)
    return {
        "status": _combine_required_statuses(
            [selector_status, normal_status, mismatch_status, causal_status]
        ),
        "capacity_column": capacity_column,
        "highest_capacity_values": (
            sorted(
                {
                    _native(value)
                    for value in panel[capacity_column].dropna().unique()
                },
                key=str,
            )
            if capacity_column
            else []
        ),
        "n_rows": int(len(panel)),
        "n_seeds": int(panel[seed_column].nunique()),
        "selector_mastery": {
            "status": selector_status,
            "strict_switching_margin_above_threshold": selector_margin,
            "strict_switching_status": selector_score_status,
            "selector_gate_pass_fraction": gate,
            "selector_gate_status": selector_gate_status,
        },
        "normal_context_mastery": {
            "status": normal_status,
            "accuracy_margin_above_threshold": normal_margin,
        },
        "directional_mismatch_context_obedience": {
            "status": mismatch_status,
            "effect": mismatch,
        },
        "causal_selector_selectivity": {
            "status": causal_status,
            "effect": causal,
        },
    }


def _h6_structure_name(value: Any) -> str:
    normalized = str(value).lower().replace("-", "_")
    return {
        "step": "step_resampled",
        "episode": "episode_static",
        "state": "state_static",
        "bias": "biased",
    }.get(normalized, normalized)


def _h6_ols_slope(frame: pd.DataFrame, x: str, y: str) -> tuple[float, int] | None:
    points = frame[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    points = points.groupby(x, as_index=False)[y].mean().sort_values(x)
    if len(points) < 3 or points[x].nunique() < 3 or np.any(points[x] <= 0):
        return None
    x_values = np.log2(points[x].to_numpy(dtype=float))
    y_values = points[y].to_numpy(dtype=float)
    centered = x_values - float(np.mean(x_values))
    denominator = float(np.dot(centered, centered))
    if denominator <= 0:
        return None
    return float(np.dot(centered, y_values - float(np.mean(y_values))) / denominator), len(points)


def _h6_slope_summary(
    records: pd.DataFrame,
    value: str,
    seed_column: str,
    *,
    bootstrap_seed: int,
    settings: InferenceSettings,
) -> dict[str, Any]:
    if records.empty or value not in records:
        return {"status": "insufficient_data"}
    per_seed = records.groupby(seed_column, dropna=False)[value].mean().dropna()
    summary = _difference_summary(
        per_seed.to_numpy(), seed=bootstrap_seed, settings=settings
    )
    summary.update(
        {
            "replication_unit": "training_seed",
            "within_seed_aggregation": "mean_across_matched_algorithm_scale_visit_strata",
            "per_seed_mean_slopes": [
                {"seed": _native(seed), "slope": float(slope)}
                for seed, slope in per_seed.items()
            ],
        }
    )
    return summary


def _h6_sample_scaling_report(
    dense: pd.DataFrame,
    rho_y: str,
    *,
    structure: str,
    algorithm: str,
    location: str,
    scale: str,
    seed_column: str,
    presentation_column: str | None,
    realized_visits: str | None,
    settings: InferenceSettings,
) -> dict[str, Any]:
    """Infer noise-gap recovery from within-seed, scale-zero-matched slopes."""

    if presentation_column is None:
        return {"status": "unavailable", "reason": "realized presentations are missing"}
    primary = dense[dense[location].astype(str) == "observation"].copy()
    if primary.empty:
        return {"status": "insufficient_data", "reason": "observation-noise cells are missing"}
    primary["_h6_structure"] = primary[structure].map(_h6_structure_name)
    primary["_h6_scale"] = pd.to_numeric(primary[scale], errors="coerce")
    primary["_h6_presentations"] = pd.to_numeric(
        primary[presentation_column], errors="coerce"
    )
    primary["_h6_rho_y"] = pd.to_numeric(primary[rho_y], errors="coerce")
    primary = primary.dropna(
        subset=["_h6_scale", "_h6_presentations", "_h6_rho_y", seed_column, algorithm]
    )

    pair_keys = [
        seed_column,
        algorithm,
        location,
        "_h6_structure",
        "_h6_presentations",
    ]
    for candidate in (
        "config.data.n_train",
        realized_visits,
        "config.data.q",
        "config.data.k",
        "config.model.width",
        "config.model.depth",
        "config.update.mode",
        "config.update.budget",
    ):
        if candidate and candidate not in pair_keys and candidate in primary:
            if primary[candidate].notna().all():
                pair_keys.append(candidate)

    zero = (
        primary[np.isclose(primary["_h6_scale"], 0.0)]
        .groupby(pair_keys, dropna=False)["_h6_rho_y"]
        .mean()
        .rename("_h6_scale_zero_rho_y")
        .reset_index()
    )
    noisy = primary[primary["_h6_scale"] > 0.0].copy()
    paired = noisy.merge(zero, on=pair_keys, how="inner", validate="many_to_one")
    if paired.empty:
        return {
            "status": "insufficient_data",
            "reason": "positive-noise cells lack matched scale=0 controls",
            "matched_gap_rows": 0,
        }
    paired["_h6_signed_gap"] = paired["_h6_rho_y"] - paired["_h6_scale_zero_rho_y"]
    # Zero is perfect recovery regardless of which side of the clean control a
    # noisy estimate lands on. A positive slope in this score means contraction
    # of the absolute noisy-minus-clean gap as presentations grow.
    paired["_h6_gap_to_zero_score"] = -np.abs(paired["_h6_signed_gap"])

    fixed_slope_keys = [seed_column, algorithm]
    for candidate in pair_keys:
        if candidate not in {
            seed_column,
            algorithm,
            "_h6_structure",
            "_h6_presentations",
            "config.data.n_train",
        }:
            fixed_slope_keys.append(candidate)
    slope_keys = [*fixed_slope_keys, "_h6_structure", "_h6_scale"]
    slope_records: list[dict[str, Any]] = []
    for values, group in paired.groupby(slope_keys, dropna=False):
        signed = _h6_ols_slope(group, "_h6_presentations", "_h6_signed_gap")
        recovery = _h6_ols_slope(group, "_h6_presentations", "_h6_gap_to_zero_score")
        if signed is None or recovery is None:
            continue
        values = values if isinstance(values, tuple) else (values,)
        ordered = group.sort_values("_h6_presentations")
        slope_records.append(
            {
                **{
                    str(key): _native(value)
                    for key, value in zip(slope_keys, values, strict=True)
                },
                "signed_gap_slope_per_log2_presentation": signed[0],
                "gap_to_zero_slope_per_log2_presentation": recovery[0],
                "n_presentation_levels": recovery[1],
                "lowest_presentation_signed_gap": float(ordered["_h6_signed_gap"].iloc[0]),
                "highest_presentation_signed_gap": float(ordered["_h6_signed_gap"].iloc[-1]),
            }
        )
    records = pd.DataFrame(slope_records)
    if records.empty:
        return {
            "status": "insufficient_data",
            "reason": "fewer than three matched presentation levels per seed and design stratum",
            "matched_gap_rows": int(len(paired)),
        }

    recovery_name = "gap_to_zero_slope_per_log2_presentation"
    signed_name = "signed_gap_slope_per_log2_presentation"
    step = records[records["_h6_structure"] == "step_resampled"].copy()
    state = records[records["_h6_structure"] == "state_static"].copy()
    step_recovery = _h6_slope_summary(
        step,
        recovery_name,
        seed_column,
        bootstrap_seed=1621,
        settings=settings,
    )
    state_recovery = _h6_slope_summary(
        state,
        recovery_name,
        seed_column,
        bootstrap_seed=1627,
        settings=settings,
    )
    step_signed = _h6_slope_summary(
        step,
        signed_name,
        seed_column,
        bootstrap_seed=1631,
        settings=settings,
    )
    state_signed = _h6_slope_summary(
        state,
        signed_name,
        seed_column,
        bootstrap_seed=1637,
        settings=settings,
    )

    comparison_keys = [*fixed_slope_keys, "_h6_scale"]
    paired_slopes = step[comparison_keys + [recovery_name, signed_name]].merge(
        state[comparison_keys + [recovery_name, signed_name]],
        on=comparison_keys,
        how="inner",
        suffixes=("_step", "_state"),
        validate="one_to_one",
    )
    paired_slopes["_h6_recovery_difference"] = (
        paired_slopes[f"{recovery_name}_step"]
        - paired_slopes[f"{recovery_name}_state"]
    )
    paired_slopes["_h6_signed_difference"] = (
        paired_slopes[f"{signed_name}_step"] - paired_slopes[f"{signed_name}_state"]
    )
    recovery_difference = _h6_slope_summary(
        paired_slopes,
        "_h6_recovery_difference",
        seed_column,
        bootstrap_seed=1643,
        settings=settings,
    )
    signed_difference = _h6_slope_summary(
        paired_slopes,
        "_h6_signed_difference",
        seed_column,
        bootstrap_seed=1657,
        settings=settings,
    )

    def directional_status(
        signed_summary: Mapping[str, Any], recovery_summary: Mapping[str, Any]
    ) -> str:
        if (
            signed_summary.get("status") != "estimated"
            or recovery_summary.get("status") != "estimated"
        ):
            return "insufficient_data"
        if (
            signed_summary.get("ci_low", -np.inf) > 0.0
            and recovery_summary.get("ci_low", -np.inf) > 0.0
        ):
            return "consistent"
        if (
            signed_summary.get("ci_high", np.inf) < 0.0
            or recovery_summary.get("ci_high", np.inf) < 0.0
        ):
            return "evidence_against"
        return "mixed_or_inconclusive"

    independent_status = directional_status(step_signed, step_recovery)
    differential_status = directional_status(signed_difference, recovery_difference)

    def combined_status(left: str, right: str) -> str:
        if left == right == "consistent":
            return "consistent"
        if "evidence_against" in {left, right}:
            return "evidence_against"
        if "mixed_or_inconclusive" in {left, right}:
            return "mixed_or_inconclusive"
        return "insufficient_data"

    status = combined_status(independent_status, differential_status)
    by_algorithm: dict[str, Any] = {}
    for algorithm_name in sorted(records[algorithm].dropna().unique(), key=str):
        algorithm_step = step[step[algorithm] == algorithm_name]
        algorithm_state = state[state[algorithm] == algorithm_name]
        algorithm_paired = paired_slopes[paired_slopes[algorithm] == algorithm_name]
        algorithm_step_signed = _h6_slope_summary(
            algorithm_step,
            signed_name,
            seed_column,
            bootstrap_seed=1663,
            settings=settings,
        )
        algorithm_step_recovery = _h6_slope_summary(
            algorithm_step,
            recovery_name,
            seed_column,
            bootstrap_seed=1667,
            settings=settings,
        )
        algorithm_signed_difference = _h6_slope_summary(
            algorithm_paired,
            "_h6_signed_difference",
            seed_column,
            bootstrap_seed=1669,
            settings=settings,
        )
        algorithm_recovery_difference = _h6_slope_summary(
            algorithm_paired,
            "_h6_recovery_difference",
            seed_column,
            bootstrap_seed=1679,
            settings=settings,
        )
        algorithm_independent_status = directional_status(
            algorithm_step_signed, algorithm_step_recovery
        )
        algorithm_differential_status = directional_status(
            algorithm_signed_difference, algorithm_recovery_difference
        )
        by_algorithm[str(algorithm_name)] = {
            "status": combined_status(
                algorithm_independent_status, algorithm_differential_status
            ),
            "independent_step_resampled": {
                "status": algorithm_independent_status,
                "signed_noisy_minus_zero_slope": algorithm_step_signed,
                "gap_to_zero_slope": algorithm_step_recovery,
            },
            "state_static_comparator": {
                "signed_noisy_minus_zero_slope": _h6_slope_summary(
                    algorithm_state,
                    signed_name,
                    seed_column,
                    bootstrap_seed=1693,
                    settings=settings,
                ),
                "gap_to_zero_slope": _h6_slope_summary(
                    algorithm_state,
                    recovery_name,
                    seed_column,
                    bootstrap_seed=1697,
                    settings=settings,
                ),
            },
            "step_resampled_minus_state_static": {
                "status": algorithm_differential_status,
                "signed_gap_slope_difference": algorithm_signed_difference,
                "gap_to_zero_slope_difference": algorithm_recovery_difference,
            },
        }

    return {
        "status": status,
        "estimand": (
            "within-seed OLS slopes of the signed noisy-minus-scale-zero rho_y gap "
            "and its negative absolute magnitude on log2(realized contextual-row "
            "presentations)"
        ),
        "expected_direction": (
            "both slopes positive: noisy rho_y recovers upward relative to scale zero "
            "and the matched gap contracts toward zero"
        ),
        "presentation_column": presentation_column,
        "scale_zero_matching_keys": pair_keys,
        "matched_gap_rows": int(len(paired)),
        "estimable_seed_design_slopes": int(len(records)),
        "independent_step_resampled": {
            "status": independent_status,
            "gap_to_zero_slope": step_recovery,
            "signed_noisy_minus_zero_slope": step_signed,
        },
        "state_static_comparator": {
            "status": state_recovery.get("status", "insufficient_data"),
            "gap_to_zero_slope": state_recovery,
            "signed_noisy_minus_zero_slope": state_signed,
        },
        "step_resampled_minus_state_static": {
            "status": differential_status,
            "gap_to_zero_slope_difference": recovery_difference,
            "signed_gap_slope_difference": signed_difference,
            "matched_seed_design_slopes": int(len(paired_slopes)),
        },
        "by_algorithm": by_algorithm,
    }


def _h6_noise_report(
    frame: pd.DataFrame,
    rho_y: str,
    margin: float,
    *,
    settings: InferenceSettings,
) -> dict[str, Any]:
    structure = _column(frame, "config.h6.structure", "noise.realized_structure")
    algorithm = _column(frame, "config.h6.algorithm", "algorithm")
    location = _column(frame, "config.h6.location", "noise.location")
    scale = _column(frame, "config.h6.scale", "noise.scale")
    reward_mode = _column(frame, "config.h6.reward_mode", "recurrence.reward_mode")
    seed_column = _column(frame, "config.seed", "seed")
    realized_visits = _column(
        frame,
        "recurrence.realized_episode_visits_per_state",
        "recurrence.realized_visits_per_state",
        "recurrence.mean_episode_visits_per_state",
        "recurrence.mean_visits_per_state",
        "config.h6.visits_per_state",
    )
    sample_presentations = _column(
        frame,
        "training.realized_presentations",
        "costs.sample_presentations",
        "config.data.n_train",
    )
    objective = _column(frame, "noise.objective_uses_location")
    required = (structure, algorithm, location, scale, reward_mode, seed_column)
    if any(value is None for value in required):
        return {"status": "unavailable"}
    assert structure and algorithm and location and scale and reward_mode and seed_column
    eligible = frame[objective].eq(True) if objective else pd.Series(True, index=frame.index)
    usable = frame[eligible].copy()
    report: dict[str, Any] = {
        "status": "estimated",
        "excluded_objective_ignored_cells": int((~eligible).sum()),
        "objective_ignored_cells_are_negative_controls": True,
    }
    dense_all = usable[
        usable[reward_mode].astype(str) == "dense_fixed_horizon"
    ].copy()
    dense = dense_all[pd.to_numeric(dense_all[scale], errors="coerce") > 0.0].copy()
    # The preregistered temporal ordering is specifically an observation-noise
    # prediction.  Label and reward perturbations remain separately reported
    # interventions and must not be pooled into its primary contrast.
    primary_dense = dense[dense[location].astype(str) == "observation"].copy()
    report["primary_ordering_location"] = "observation"
    match_keys = [seed_column, algorithm, location, scale]
    for candidate in (
        "config.data.n_train",
        sample_presentations,
        realized_visits,
        "config.model.width",
        "config.update.mode",
        "config.update.budget",
    ):
        if candidate is None or candidate in match_keys:
            continue
        if candidate in primary_dense and primary_dense[candidate].notna().all():
            match_keys.append(candidate)
    pivot = primary_dense.pivot_table(
        index=match_keys, columns=structure, values=rho_y, aggfunc="mean"
    )
    contrast_definitions = {
        "step_minus_episode": ("step", "episode"),
        "episode_minus_state": ("episode", "state"),
        "step_minus_state": ("step", "state"),
        "step_minus_biased": ("step", "biased"),
    }
    contrasts: dict[str, Any] = {}
    for name, (left, right) in contrast_definitions.items():
        differences = (
            (pivot[left] - pivot[right]).dropna()
            if left in pivot and right in pivot
            else pd.Series(dtype=float)
        )
        if seed_column in differences.index.names:
            differences = differences.groupby(level=seed_column).mean()
        contrast = (
            _difference_summary(differences, seed=1601, settings=settings)
            if len(differences)
            else {"status": "insufficient_data"}
        )
        if contrast.get("status") == "estimated":
            contrast["equivalent_within_margin"] = bool(
                contrast["ci_low"] >= -margin and contrast["ci_high"] <= margin
            )
        contrasts[name] = contrast
    report["matched_dense_structure_contrasts"] = contrasts

    by_algorithm: dict[str, dict[str, Any]] = {}
    for algorithm_name, group in primary_dense.groupby(algorithm, dropna=False):
        algorithm_pivot = group.pivot_table(
            index=match_keys, columns=structure, values=rho_y, aggfunc="mean"
        )
        algorithm_contrasts: dict[str, Any] = {}
        for name, (left, right) in contrast_definitions.items():
            if left not in algorithm_pivot or right not in algorithm_pivot:
                algorithm_contrasts[name] = {"status": "insufficient_data"}
                continue
            difference = (algorithm_pivot[left] - algorithm_pivot[right]).dropna()
            if seed_column in difference.index.names:
                difference = difference.groupby(level=seed_column).mean()
            result = _difference_summary(
                difference, seed=1613, settings=settings
            )
            if result.get("status") == "estimated":
                result["equivalent_within_margin"] = bool(
                    result["ci_low"] >= -margin and result["ci_high"] <= margin
                )
            algorithm_contrasts[name] = result
        by_algorithm[str(algorithm_name)] = algorithm_contrasts
    report["matched_dense_structure_contrasts_by_algorithm"] = by_algorithm

    presentation_column = _column(
        dense_all,
        "training.realized_presentations",
        "costs.sample_presentations",
        "config.data.n_train",
    )
    report["sample_presentation_column"] = presentation_column
    report["sample_scaling"] = _h6_sample_scaling_report(
        dense_all,
        rho_y,
        structure=structure,
        algorithm=algorithm,
        location=location,
        scale=scale,
        seed_column=seed_column,
        presentation_column=presentation_column,
        realized_visits=realized_visits,
        settings=settings,
    )

    terminal = usable[
        (usable[reward_mode].astype(str) == "terminal")
        & (usable[algorithm].astype(str) == "rl")
        & (usable[location].astype(str) == "reward")
    ].copy()
    terminal_keys = [seed_column, scale]
    for candidate in (
        "config.data.n_train",
        sample_presentations,
        realized_visits,
    ):
        if candidate is None or candidate in terminal_keys:
            continue
        if candidate in terminal and terminal[candidate].notna().all():
            terminal_keys.append(candidate)
    terminal_pivot = terminal.pivot_table(
        index=terminal_keys, columns=structure, values=rho_y, aggfunc="mean"
    )
    if "step" in terminal_pivot and "episode" in terminal_pivot:
        terminal_difference = (
            terminal_pivot["step"] - terminal_pivot["episode"]
        ).dropna()
        # Noise scales are planned cells, not independent replications.  As in
        # the dense structure contrasts above, collapse them within training
        # seed before constructing the uncertainty interval.
        if seed_column in terminal_difference.index.names:
            terminal_difference = terminal_difference.groupby(
                level=seed_column
            ).mean()
        equivalence = _difference_summary(
            terminal_difference,
            seed=1607,
            settings=settings,
        )
        if equivalence.get("status") == "estimated":
            equivalence["equivalent_within_margin"] = bool(
                equivalence["ci_low"] >= -margin and equivalence["ci_high"] <= margin
            )
        report["terminal_reward_step_episode_equivalence"] = equivalence
    else:
        report["terminal_reward_step_episode_equivalence"] = {
            "status": "insufficient_data"
        }
    seen_new_gap = _column(frame, "seen_new_gap.rho_y")
    if seen_new_gap:
        report["seen_minus_new_state_gap_by_structure"] = {
            str(name): _seed_ci(group, seen_new_gap, settings=settings)
            for name, group in usable.groupby(structure)
        }
    return report


def _paired_level_difference(
    frame: pd.DataFrame,
    left_column: str | None,
    right_column: str | None,
    seed: int,
    *,
    settings: InferenceSettings,
) -> dict[str, Any]:
    seed_column = _column(frame, "config.seed", "seed")
    if (
        seed_column is None
        or left_column is None
        or right_column is None
        or left_column not in frame
        or right_column not in frame
    ):
        return {"status": "insufficient_data"}
    paired = frame[[seed_column, left_column, right_column]].dropna().copy()
    if paired.empty:
        return {"status": "insufficient_data"}
    paired["difference"] = (
        pd.to_numeric(paired[left_column], errors="coerce")
        - pd.to_numeric(paired[right_column], errors="coerce")
    )
    seed_values = paired.groupby(seed_column, dropna=False)["difference"].mean().dropna()
    result = _difference_summary(seed_values, seed=seed, settings=settings)
    result.update(
        {
            "n_paired_runs": int(len(paired)),
            "n_paired_seeds": int(len(seed_values)),
        }
    )
    return result


def _level_confirmation_report(
    frame: pd.DataFrame,
    hypothesis: str,
    *,
    margin: float,
    settings: InferenceSettings,
) -> dict[str, Any]:
    """Report seed-level choice/fork/navigation confirmation and capability gates."""

    level_columns = {
        level: _column(
            frame,
            f"navigation.{level}.intended_goal_success_rate",
            *("navigation.choice.selection_accuracy",) if level == "choice" else (),
        )
        for level in ("choice", "fork", "navigation")
    }
    levels: dict[str, Any] = {}
    for level, outcome in level_columns.items():
        oracle = _column(frame, f"navigation.{level}.oracle_goal_success_rate")
        clamped = _column(frame, f"navigation.{level}.clamped_goal_success_rate")
        episodes = _column(frame, f"navigation.{level}.episodes")
        if outcome is None:
            levels[level] = {
                "status": "insufficient_data",
                "reason": f"{level} evaluation was not present in completed runs",
            }
            continue
        levels[level] = {
            "status": "estimated",
            "intended_goal_success": _seed_ci(
                frame, outcome, settings=settings
            ),
            "oracle_capability": _seed_ci(frame, oracle, settings=settings)
            if oracle
            else {"status": "insufficient_data"},
            "clamped_capability": _seed_ci(frame, clamped, settings=settings)
            if clamped
            else {"status": "insufficient_data"},
            "episodes_per_run": (
                sorted({_native(value) for value in frame[episodes].dropna()})
                if episodes
                else []
            ),
            "outcome_column": outcome,
        }
        if level == "navigation":
            heldout = _column(
                frame, "navigation.navigation.heldout_clamped_goal_success_rate"
            )
            levels[level]["heldout_map_capability"] = (
                _seed_ci(frame, heldout, settings=settings)
                if heldout
                else {"status": "insufficient_data"}
            )

    contrasts = {
        "fork_minus_choice": _paired_level_difference(
            frame,
            level_columns["fork"],
            level_columns["choice"],
            2301,
            settings=settings,
        ),
        "navigation_minus_choice": _paired_level_difference(
            frame,
            level_columns["navigation"],
            level_columns["choice"],
            2303,
            settings=settings,
        ),
        "navigation_minus_fork": _paired_level_difference(
            frame,
            level_columns["navigation"],
            level_columns["fork"],
            2309,
            settings=settings,
        ),
    }

    hypothesis_effects: dict[str, Any] = {}
    if hypothesis == "h1":
        capacity = _column(
            frame,
            "model.total_parameters",
            "model.competition.total_parameters",
            "config.model.width",
            "config.update.budget",
        )
        for level, outcome in level_columns.items():
            hypothesis_effects[level] = (
                {
                    "status": "estimated",
                    "q_effect": _seedwise_spearman(
                        frame,
                        _column(frame, "config.data.q"),
                        outcome,
                        bootstrap_seed=2311,
                        settings=settings,
                    ),
                    "k_effect": _seedwise_spearman(
                        frame,
                        _column(frame, "config.data.k"),
                        outcome,
                        bootstrap_seed=2317,
                        settings=settings,
                    ),
                    "capacity_effect": _seedwise_spearman(
                        frame,
                        capacity,
                        outcome,
                        bootstrap_seed=2321,
                        settings=settings,
                    ),
                }
                if outcome
                else {
                    "status": "insufficient_data",
                    "reason": f"no {level} outcome",
                }
            )
    elif hypothesis == "h2":
        calibration = _column(
            frame,
            "final.exact_calibration_iid.target_accuracy",
            "final.exact_decoder_accuracy",
            "final.decoder_accuracy",
        )
        for level, outcome in level_columns.items():
            hypothesis_effects[level] = (
                {
                    "status": "estimated",
                    "exact_calibration_behavior_association": _seedwise_spearman(
                        frame,
                        calibration,
                        outcome,
                        bootstrap_seed=2333,
                        settings=settings,
                    ),
                }
                if outcome and calibration
                else {
                    "status": "insufficient_data",
                    "reason": f"no paired calibration/{level} outcome",
                }
            )
    elif hypothesis == "h3":
        for level, outcome in level_columns.items():
            hypothesis_effects[level] = (
                {
                    "status": (
                        "dynamic_and_final_confirmation"
                        if level == "choice"
                        else "final_checkpoint_confirmation_only"
                    ),
                    "final_intended_goal_success": _seed_ci(
                        frame, outcome, settings=settings
                    ),
                    "dynamic_acquisition_inference": (
                        {
                            "status": "reported_in_primary_h3_analysis",
                            "reason": "checkpoint time-to-event inference is choice-level",
                        }
                        if level == "choice"
                        else {
                            "status": "insufficient_data",
                            "reason": (
                                "fork/navigation rollouts are evaluated only at the final "
                                "selector checkpoint; time-to-event inference remains choice-level"
                            ),
                        }
                    ),
                }
                if outcome
                else {
                    "status": "insufficient_data",
                    "reason": f"no {level} outcome",
                }
            )
    elif hypothesis == "h4":
        condition = _column(frame, "config.h4.condition")
        n_conflict = _column(frame, "data.n_conflict", "config.h4.n_conflict")
        matched = frame.copy()
        if condition:
            matched = matched[matched[condition].isin(["concentrated", "diverse"])]
        if n_conflict:
            matched = matched[pd.to_numeric(matched[n_conflict], errors="coerce") > 0]
        for level, outcome in level_columns.items():
            hypothesis_effects[level] = (
                {
                    "status": "estimated",
                    "diverse_minus_concentrated": _paired_condition_difference(
                        matched,
                        condition or "",
                        "diverse",
                        "concentrated",
                        outcome,
                        settings=settings,
                    ),
                    "equivalence_margin": margin,
                }
                if outcome and condition
                else {
                    "status": "insufficient_data",
                    "reason": f"no matched H4 {level} outcome",
                }
            )
    return {
        "status": "estimated"
        if any(item.get("status") == "estimated" for item in levels.values())
        else "insufficient_data",
        "replication_unit": "training seed",
        "levels": levels,
        "paired_level_contrasts": contrasts,
        "hypothesis_effects": hypothesis_effects,
        "note": (
            "Sequential-task outcomes are interpreted only alongside oracle/clamped "
            "navigator capability; missing levels are reported rather than imputed."
        ),
    }


def evaluate_hypothesis(
    summaries: pd.DataFrame,
    hypothesis: str,
    margin: float | None = None,
    metrics: pd.DataFrame | None = None,
    *,
    bootstrap_samples: int | None = None,
    confidence: float | None = None,
) -> dict[str, Any]:
    """Apply preregistered directional/equivalence checks to run-level outcomes."""

    hypothesis = hypothesis.lower()
    if summaries.empty:
        return {
            "hypothesis": hypothesis,
            "status": "insufficient_data",
            "reason": "No completed runs",
            "inference_sufficiency": {
                "status": "insufficient_data",
                "observed_independent_seeds": 0,
                "minimum_inferential_seeds": MIN_INFERENTIAL_SEEDS,
                "replication_unit": "training seed",
            },
        }
    h_column = _column(summaries, "config.experiment.hypothesis")
    frame = summaries[summaries[h_column] == hypothesis].copy() if h_column else summaries.copy()
    if frame.empty:
        return {
            "hypothesis": hypothesis,
            "status": "insufficient_data",
            "reason": "No matching runs",
            "inference_sufficiency": {
                "status": "insufficient_data",
                "observed_independent_seeds": 0,
                "minimum_inferential_seeds": MIN_INFERENTIAL_SEEDS,
                "replication_unit": "training seed",
            },
        }

    settings = _resolve_inference_settings(
        frame,
        margin=margin,
        bootstrap_samples=bootstrap_samples,
        confidence=confidence,
    )
    margin = settings.equivalence_margin

    delta = _column(
        frame,
        "final.conflict.delta_rho",
        "final.competition_conflict.delta_rho",
        "final.conflict_unseen.delta_rho",
        "final.delta_rho",
        "delta_rho",
    )
    rho_y = _column(
        frame,
        "final.conflict.rho_y",
        "final.competition_conflict.rho_y",
        "final.conflict_unseen.rho_y",
        "final.rho_y",
        "rho_y",
    )
    report: dict[str, Any] = {
        "hypothesis": hypothesis,
        "n_runs": int(len(frame)),
        "n_seeds": int(frame[_column(frame, "config.seed", "seed")].nunique())
        if _column(frame, "config.seed", "seed")
        else None,
        "equivalence_margin": margin,
        "inference_settings": settings.as_dict(),
        "primary_rho_y": (
            _seed_ci(frame, rho_y, settings=settings)
            if rho_y
            else {"status": "unavailable"}
        ),
    }

    if hypothesis == "h1":
        report["q_effect"] = _seedwise_spearman(
            frame,
            _column(frame, "config.data.q"),
            delta,
            bootstrap_seed=1101,
            settings=settings,
        )
        report["k_effect"] = _seedwise_spearman(
            frame,
            _column(frame, "config.data.k"),
            delta,
            bootstrap_seed=1103,
            settings=settings,
        )
        capacity = _column(
            frame,
            "model.total_parameters",
            "model.competition.total_parameters",
            "config.model.width",
            "config.update.budget",
        )
        report["capacity_effect"] = _seedwise_spearman(
            frame,
            capacity,
            delta,
            bootstrap_seed=1109,
            settings=settings,
        )
        directional = (
            (report["q_effect"], -1),
            (report["k_effect"], -1),
            (report["capacity_effect"], 1),
        )
        support = [
            item.get("status") == "estimated"
            and (
                item.get("ci_high", np.inf) < 0
                if direction < 0
                else item.get("ci_low", -np.inf) > 0
            )
            for item, direction in directional
        ]
        reversals = [
            item.get("status") == "estimated"
            and (
                item.get("ci_low", -np.inf) > 0
                if direction < 0
                else item.get("ci_high", np.inf) < 0
            )
            for item, direction in directional
        ]
        report["status"] = (
            "consistent"
            if all(support)
            else "evidence_against"
            if any(reversals)
            else "mixed_or_inconclusive"
        )
    elif hypothesis == "h2":
        complexity = _h2_complexity_report(frame, settings=settings)
        report["complexity"] = complexity
        report["C_param"] = complexity.get("C_param", {})
        report["C_update"] = complexity.get("C_update", {})
        report["C_steps"] = complexity.get("C_steps", {})
        report["calibration_crossing_behavior"] = complexity.get(
            "calibration_crossing_behavior", []
        )
        contrasts = [
            item.get("paired_crossed_minus_not", {})
            for item in report["calibration_crossing_behavior"]
            if item.get("paired_crossed_minus_not", {}).get("status") == "estimated"
        ]
        supported = [item.get("ci_low", -np.inf) > 0.0 for item in contrasts]
        reversed_precisely = [item.get("ci_high", np.inf) < 0.0 for item in contrasts]
        equivalent_to_zero = [
            item.get("ci_low", -np.inf) >= -margin
            and item.get("ci_high", np.inf) <= margin
            for item in contrasts
        ]
        report["calibration_behavior_inference"] = {
            "n_estimable_seed_level_contrasts": len(contrasts),
            "all_directional_intervals_above_zero": bool(contrasts)
            and all(supported),
            "any_precise_reversal": any(reversed_precisely),
            "any_interval_equivalent_to_zero": any(equivalent_to_zero),
            "equivalence_margin": margin,
            "decision_unit": "seed-level paired bootstrap interval",
        }
        report["status"] = (
            "consistent"
            if contrasts and all(supported)
            else "evidence_against"
            if any(reversed_precisely) or any(equivalent_to_zero)
            else "mixed_or_inconclusive"
            if complexity.get("status") == "estimated"
            else "insufficient_data"
        )
    elif hypothesis == "h3":
        proxy_time = _column(frame, "events.proxy_acquisition_time", "proxy_acquisition_time")
        intended_time = _column(frame, "events.intended_acquisition_time", "intended_acquisition_time")
        proxy_observed = _column(frame, "events.proxy_acquisition_observed")
        intended_observed = _column(frame, "events.intended_acquisition_observed")
        replacement_time = _column(
            frame,
            "events.proxy_to_intended_replacement_time",
            "proxy_to_intended_replacement_time",
        )
        replacement_observed = _column(
            frame,
            "events.proxy_to_intended_replacement_observed",
            "proxy_to_intended_replacement_observed",
        )
        replacement_eligible = _column(
            frame,
            "events.sequential_replacement_eligible",
            "sequential_replacement_eligible",
        )
        raw_dominance_time = _column(
            frame,
            "events.intended_dominance_crossing_time",
            "intended_dominance_crossing_time",
        )
        raw_dominance_observed = _column(
            frame,
            "events.intended_dominance_crossing_observed",
            "intended_dominance_crossing_observed",
        )
        simple_fraction: float | None = None
        if proxy_time and intended_time and proxy_observed and intended_observed:
            complete = frame[[proxy_time, intended_time]].notna().all(axis=1)
            proxy_seen = frame[proxy_observed].eq(True)
            intended_seen = frame[intended_observed].eq(True)
            both = complete & proxy_seen & intended_seen
            proxy_only_identified = (
                complete
                & proxy_seen
                & ~intended_seen
                & (frame[proxy_time].astype(float) < frame[intended_time].astype(float))
            )
            intended_only_identified = (
                complete
                & ~proxy_seen
                & intended_seen
                & (frame[intended_time].astype(float) < frame[proxy_time].astype(float))
            )
            ordering = pd.Series(pd.NA, index=frame.index, dtype="boolean")
            ordering.loc[both] = (
                frame.loc[both, proxy_time].astype(float)
                < frame.loc[both, intended_time].astype(float)
            )
            # A recorded event before the other process's censoring horizon fixes
            # the ordering even though the second event time is not observed.
            ordering.loc[proxy_only_identified] = True
            ordering.loc[intended_only_identified] = False
            identifiable = ordering.notna()
            difference = (
                frame.loc[both, intended_time].astype(float)
                - frame.loc[both, proxy_time].astype(float)
            )
            seed_name = _column(frame, "config.seed", "seed")
            report["proxy_acquisition_survival"] = _kaplan_meier(
                frame[proxy_time].to_numpy(), frame[proxy_observed].to_numpy()
            )
            report["intended_acquisition_survival"] = _kaplan_meier(
                frame[intended_time].to_numpy(), frame[intended_observed].to_numpy()
            )
            report["both_goals_observed_runs"] = int(both.sum())
            if len(difference):
                difference_rows = pd.DataFrame({"difference": difference})
                if seed_name:
                    difference_rows["seed"] = frame.loc[both, seed_name]
                    difference_values = (
                        difference_rows.groupby("seed")["difference"].mean().to_numpy()
                    )
                else:
                    difference_values = difference_rows["difference"].to_numpy()
                report["both_goals_observed_seeds"] = int(len(difference_values))
                report["intended_minus_proxy_acquisition"] = _difference_summary(
                    difference_values, seed=23, settings=settings
                )
            else:
                report["both_goals_observed_seeds"] = 0
                report["intended_minus_proxy_acquisition"] = {
                    "status": "insufficient_data"
                }
            report["acquisition_order_identifiable_runs"] = int(identifiable.sum())
            report["acquisition_order_censor_identified_runs"] = int(
                (proxy_only_identified | intended_only_identified).sum()
            )
            report["acquisition_order_unidentified_runs"] = int((~identifiable).sum())
            if identifiable.any():
                ordering_rows = pd.DataFrame(
                    {"simple_first": ordering.loc[identifiable].astype(bool)}
                )
                if seed_name:
                    ordering_rows["seed"] = frame.loc[identifiable, seed_name]
                    ordering_values = (
                        ordering_rows.groupby("seed")["simple_first"].mean().to_numpy()
                    )
                else:
                    ordering_values = ordering_rows["simple_first"].astype(float).to_numpy()
                ordering_summary = _difference_summary(
                    ordering_values, seed=29, settings=settings
                )
                report["simple_first_fraction"] = ordering_summary.get("mean")
                report["simple_first_seed_bootstrap"] = ordering_summary
                report["acquisition_order_identifiable_seeds"] = int(
                    len(ordering_values)
                )
            else:
                report["simple_first_fraction"] = None
                report["simple_first_seed_bootstrap"] = {
                    "status": "insufficient_data"
                }
                report["acquisition_order_identifiable_seeds"] = 0
            simple_fraction = report["simple_first_fraction"]
        if replacement_time and replacement_observed:
            eligible = (
                frame[replacement_eligible].eq(True)
                if replacement_eligible
                else frame[proxy_observed].eq(True)
                if proxy_observed
                else pd.Series(True, index=frame.index)
            )
            report["sequential_replacement_eligible_runs"] = int(eligible.sum())
            report["sequential_replacement_ineligible_censored_runs"] = int(
                (~eligible).sum()
            )
            report[
                "proxy_to_intended_replacement_survival_all_runs"
            ] = _kaplan_meier(
                frame[replacement_time].to_numpy(),
                frame[replacement_observed].to_numpy(),
            )
            report["proxy_to_intended_replacement_survival"] = (
                _kaplan_meier(
                    frame.loc[eligible, replacement_time].to_numpy(),
                    frame.loc[eligible, replacement_observed].to_numpy(),
                )
                if eligible.any()
                else {"status": "insufficient_data"}
            )
            report["proxy_to_intended_replacement_definition"] = (
                "dominance crossing strictly after confirmed sustained proxy acquisition"
            )
            report["proxy_to_intended_replacement_primary_risk_set"] = (
                "runs with an observed sustained proxy acquisition; non-acquirers are "
                "reported as right-censored in the all-runs curve"
            )
        if raw_dominance_time and raw_dominance_observed:
            report["raw_intended_dominance_crossing_survival"] = _kaplan_meier(
                frame[raw_dominance_time].to_numpy(),
                frame[raw_dominance_observed].to_numpy(),
            )
        trajectory_report = _h3_trajectory_report(
            frame,
            metrics,
            dominance_margin=margin,
            settings=settings,
        )
        report["reward_plateau_before_goal_stabilization"] = trajectory_report
        plateau_fraction = trajectory_report.get(
            "plateau_before_goal_stabilization_fraction"
        )
        inference_components = {
            "simple_first_fraction": report.get("simple_first_seed_bootstrap", {}),
            "plateau_before_goal_stabilization_fraction": trajectory_report.get(
                "plateau_before_goal_stabilization_seed_bootstrap", {}
            ),
        }
        estimated_components = {
            name: value
            for name, value in inference_components.items()
            if value.get("status") == "estimated"
        }
        supported = {
            name: value.get("ci_low", -np.inf) > 0.5
            for name, value in estimated_components.items()
        }
        reversed_precisely = {
            name: value.get("ci_high", np.inf) < 0.5
            for name, value in estimated_components.items()
        }
        equivalent_to_half = {
            name: value.get("ci_low", -np.inf) >= 0.5 - margin
            and value.get("ci_high", np.inf) <= 0.5 + margin
            for name, value in estimated_components.items()
        }
        report["directional_inference"] = {
            "required_components": list(inference_components),
            "estimable_components": list(estimated_components),
            "interval_above_half": supported,
            "precise_reversal_below_half": reversed_precisely,
            "equivalent_to_half": equivalent_to_half,
            "equivalence_margin": margin,
            "decision_unit": "seed-level bootstrap interval",
        }
        report["status"] = (
            "consistent"
            if len(estimated_components) == len(inference_components)
            and all(supported.values())
            else "evidence_against"
            if any(reversed_precisely.values()) or any(equivalent_to_half.values())
            else "mixed_or_inconclusive"
            if proxy_time and intended_time
            else "insufficient_data"
        )
    elif hypothesis == "h4":
        condition = _column(frame, "config.h4.condition")
        n_conflict = _column(frame, "data.n_conflict", "config.h4.n_conflict")
        diversity = _column(
            frame,
            "data.u_conflict",
            "data.realized_u_conflict",
            "data.effective_unique_contexts",
            "config.h4.unique_fraction",
        )
        matched = frame.copy()
        if condition:
            matched = matched[matched[condition].isin(["concentrated", "diverse"])]
        if n_conflict:
            matched = matched[pd.to_numeric(matched[n_conflict], errors="coerce") > 0]
        report["conflict_diversity_association"] = _stratified_spearman(
            matched,
            diversity,
            rho_y,
            (
                "data.n_conflict" if "data.n_conflict" in matched else "config.h4.n_conflict",
                "config.data.k",
                "config.model.width",
                "config.update.budget",
            ),
        )
        report["pooled_diversity_association_diagnostic"] = _spearman(
            matched, diversity, rho_y
        )
        report["diverse_minus_concentrated"] = _paired_condition_difference(
            matched,
            condition or "",
            "diverse",
            "concentrated",
            rho_y or "",
            settings=settings,
        )
        if n_conflict:
            agreement_only = frame[
                pd.to_numeric(frame[n_conflict], errors="coerce").eq(0)
            ].copy()
            if condition:
                agreement_only = agreement_only[
                    agreement_only[condition].isin(["concentrated", "diverse"])
                ]
            report["agreement_only_intended_reliance"] = (
                _seed_ci(agreement_only, rho_y, settings=settings)
                if rho_y and not agreement_only.empty
                else {"status": "insufficient_data"}
            )
        acquisition_threshold, _ = _configured_threshold(
            frame, ("config.evaluation.acquisition_threshold",), 0.9
        )
        agreement_lower = report.get("agreement_only_intended_reliance", {}).get(
            "ci_low"
        )
        report["agreement_only_reliability_threshold"] = acquisition_threshold
        report["agreement_only_is_reliably_intended"] = bool(
            agreement_lower is not None and agreement_lower >= acquisition_threshold
        )
        fraction = _column(frame, "config.h4.unique_fraction")
        if condition and fraction:
            structured = frame[
                frame[condition].isin(["diverse", "structured_holdout"])
                & pd.to_numeric(frame[fraction], errors="coerce").eq(1.0)
            ].copy()
            if n_conflict:
                structured = structured[
                    pd.to_numeric(structured[n_conflict], errors="coerce") > 0
                ]
            report["structured_holdout_minus_matched_diverse"] = (
                _paired_condition_difference(
                    structured,
                    condition,
                    "structured_holdout",
                    "diverse",
                    rho_y or "",
                    settings=settings,
                )
                if not structured.empty
                else {"status": "insufficient_data"}
            )
            disjoint_types = _column(
                structured, "data.structured_types_are_disjoint"
            )
            type_visible = _column(
                structured, "data.failure_mechanism_visible_to_model"
            )
            train_types = _column(
                structured,
                "data.structured_train_types",
                "config.h4.structured_train_types",
            )
            test_types = _column(
                structured,
                "data.structured_test_types",
                "config.h4.structured_test_types",
            )
            report["structured_holdout_semantics"] = {
                "status": (
                    "verified"
                    if not structured.empty
                    and disjoint_types
                    and structured[disjoint_types].eq(True).all()
                    and type_visible
                    and structured[type_visible].eq(False).all()
                    else "insufficient_data"
                ),
                "train_failure_families": (
                    sorted({_native(value) for value in structured[train_types].dropna()})
                    if train_types
                    else []
                ),
                "unseen_evaluation_failure_families": (
                    sorted({_native(value) for value in structured[test_types].dropna()})
                    if test_types
                    else []
                ),
                "families_declared_disjoint": bool(
                    disjoint_types and structured[disjoint_types].eq(True).all()
                ),
                "mechanism_identity_visible_to_model": (
                    bool(structured[type_visible].eq(True).any())
                    if type_visible
                    else None
                ),
                "estimand": (
                    "transfer from trained context-corruption mechanisms to "
                    "disjoint, named corruption mechanisms"
                ),
            }
        estimate = report["diverse_minus_concentrated"].get("mean")
        ci_low = report["diverse_minus_concentrated"].get("ci_low")
        ci_high = report["diverse_minus_concentrated"].get("ci_high")
        if report["agreement_only_is_reliably_intended"]:
            report["status"] = "evidence_against"
        elif estimate is None:
            report["status"] = "insufficient_data"
        elif ci_low is not None and ci_low > margin:
            report["status"] = "consistent"
        elif (
            ci_low is not None
            and ci_high is not None
            and ci_low >= -margin
            and ci_high <= margin
        ):
            report["status"] = "evidence_against"
        else:
            report["status"] = "mixed_or_inconclusive"
    elif hypothesis == "h5":
        report["update_capacity_thresholds"] = _h5_update_thresholds(
            frame, rho_y, settings=settings
        )
        report["entropy_ratio_association"] = report["update_capacity_thresholds"].get(
            "entropy_ratio_association", {"status": "unavailable"}
        )
        controls = report["update_capacity_thresholds"].get(
            "controls_and_falsifiers", {}
        )
        triggered_falsifiers = [
            name
            for name, result in controls.items()
            if result.get("status") == "falsifier_triggered"
        ]
        trend = report["entropy_ratio_association"]
        high_entropy_claim = report["update_capacity_thresholds"].get(
            "primary_high_entropy_claim", {}
        )
        censor_heavy = bool(
            report["update_capacity_thresholds"]
            .get("censoring_assessment", {})
            .get("censor_heavy", False)
        )
        report["directional_inference"] = {
            "positive_trend_ci_supported": bool(
                trend.get("status") == "estimated"
                and trend.get("ci_low", -np.inf) > 0.0
            ),
            "high_entropy_rl_advantage_ci_supported": bool(
                high_entropy_claim.get("ci_supported_advantage", False)
            ),
            "high_entropy_no_advantage_or_equivalence": bool(
                high_entropy_claim.get("status") == "evidence_against"
            ),
            "precise_negative_reversal": bool(
                trend.get("status") == "estimated"
                and trend.get("ci_high", np.inf) < 0.0
            ),
            "equivalent_to_zero": bool(
                trend.get("status") == "estimated"
                and trend.get("ci_low", -np.inf) >= -margin
                and trend.get("ci_high", np.inf) <= margin
            ),
            "slope_equivalence_margin": margin,
            "censor_heavy": censor_heavy,
            "triggered_falsifiers": triggered_falsifiers,
            "decision_unit": "matched training-seed threshold bootstrap",
        }
        if report["update_capacity_thresholds"].get("status") != "estimated":
            report["status"] = "insufficient_data"
        elif censor_heavy:
            report["status"] = "mixed_or_inconclusive"
        elif report["directional_inference"]["precise_negative_reversal"]:
            report["status"] = "evidence_against"
        elif report["directional_inference"][
            "high_entropy_no_advantage_or_equivalence"
        ]:
            report["status"] = "evidence_against"
        elif triggered_falsifiers:
            report["status"] = "evidence_against"
        elif (
            report["directional_inference"]["positive_trend_ci_supported"]
            and report["directional_inference"][
                "high_entropy_rl_advantage_ci_supported"
            ]
        ):
            report["status"] = "consistent"
        else:
            # A point trend, an interval compatible with zero, or extensive
            # threshold censoring is not affirmative or negative evidence.
            report["status"] = "mixed_or_inconclusive"
    elif hypothesis == "h6":
        if rho_y:
            noise_report = _h6_noise_report(
                frame, rho_y, margin, settings=settings
            )
            report["noise_structure"] = noise_report

            def temporal_status_for(contrasts: Mapping[str, Any]) -> str:
                ordering = [
                    contrasts.get("step_minus_episode", {}),
                    contrasts.get("episode_minus_state", {}),
                ]
                if all(
                    item.get("status") == "estimated"
                    and item.get("ci_low", -np.inf) > 0
                    for item in ordering
                ):
                    return "consistent"
                if any(
                    item.get("status") == "estimated"
                    and item.get("ci_high", np.inf) < 0
                    for item in ordering
                ) or (
                    all(item.get("status") == "estimated" for item in ordering)
                    and all(
                        item.get("equivalent_within_margin") is True
                        for item in ordering
                    )
                ):
                    return "evidence_against"
                if any(item.get("status") == "estimated" for item in ordering):
                    return "mixed_or_inconclusive"
                return "insufficient_data"

            def combine_subclaims(left: str, right: str) -> str:
                if left == right == "consistent":
                    return "consistent"
                if "evidence_against" in {left, right}:
                    return "evidence_against"
                if left == right == "insufficient_data":
                    return "insufficient_data"
                return "mixed_or_inconclusive"

            contrasts = noise_report.get("matched_dense_structure_contrasts", {})
            temporal_status = temporal_status_for(contrasts)
            scaling_status = noise_report.get("sample_scaling", {}).get(
                "status", "insufficient_data"
            )
            temporal_by_algorithm = noise_report.get(
                "matched_dense_structure_contrasts_by_algorithm", {}
            )
            scaling_by_algorithm = noise_report.get("sample_scaling", {}).get(
                "by_algorithm", {}
            )
            algorithms = sorted(
                set(temporal_by_algorithm) | set(scaling_by_algorithm), key=str
            )
            algorithm_subclaims: dict[str, Any] = {}
            for algorithm_name in algorithms:
                algorithm_temporal = temporal_status_for(
                    temporal_by_algorithm.get(algorithm_name, {})
                )
                algorithm_scaling = scaling_by_algorithm.get(
                    algorithm_name, {}
                ).get("status", "insufficient_data")
                algorithm_subclaims[str(algorithm_name)] = {
                    "status": combine_subclaims(
                        algorithm_temporal, algorithm_scaling
                    ),
                    "temporal_robustness_ordering": {
                        "status": algorithm_temporal,
                        "contrasts": temporal_by_algorithm.get(algorithm_name, {}),
                    },
                    "sample_scaling_noise_gap_recovery": {
                        "status": algorithm_scaling,
                        "analysis": scaling_by_algorithm.get(algorithm_name, {}),
                    },
                }
            required_algorithms = ("clean_sft", "rl")
            required_statuses = {
                name: algorithm_subclaims.get(
                    name,
                    {
                        "status": "insufficient_data",
                        "temporal_robustness_ordering": {
                            "status": "insufficient_data"
                        },
                        "sample_scaling_noise_gap_recovery": {
                            "status": "insufficient_data"
                        },
                    },
                )
                for name in required_algorithms
            }
            report["subclaims"] = {
                "temporal_robustness_ordering": {"status": temporal_status},
                "sample_scaling_noise_gap_recovery": {"status": scaling_status},
                "required_algorithm_evidence": {
                    "required_algorithms": list(required_algorithms),
                    "by_algorithm": required_statuses,
                },
            }
            report["algorithm_subclaims"] = algorithm_subclaims
            required_arm_statuses = {
                value["status"] for value in required_statuses.values()
            }
            if required_arm_statuses == {"consistent"}:
                report["status"] = "consistent"
            elif "evidence_against" in required_arm_statuses:
                report["status"] = "evidence_against"
            elif required_arm_statuses == {"insufficient_data"}:
                report["status"] = "insufficient_data"
            else:
                report["status"] = "mixed_or_inconclusive"
        else:
            report["status"] = "insufficient_data"
    elif hypothesis == "h7":
        ablation = _column(frame, "config.h7.weight_decay_ablation")
        if ablation:
            primary = ~frame[ablation].eq(True)
            report["weight_decay_ablation_runs"] = int((~primary).sum())
            frame = frame[primary].copy()
        eligibility = _column(frame, "final.eligible_for_primary_analysis")
        if eligibility:
            eligible = frame[eligibility].eq(True)
            report["excluded_gate_failures"] = int(np.sum(~eligible))
            frame = frame[eligible]
        condition = _column(frame, "config.h7.condition")
        half_life = _column(frame, "events.half_life_time", "events.half_life.time", "half_life_time")
        observed = _column(frame, "events.half_life_observed", "events.half_life.observed")
        if condition and half_life and observed and not frame.empty:
            survival, ordering = _h7_stratified_half_lives(
                frame,
                condition,
                half_life,
                observed,
                settings=settings,
            )
            report["half_life_survival_stratified"] = survival
            report["matched_ordering_contrasts"] = ordering
            half_life_reversal = [
                item["contrast"]
                for item in ordering["reversal_minus_decorrelation_by_q_b"]
                if item["contrast"].get("status") == "estimated"
            ]
            half_life_supported = bool(
                ordering["decorrelation_minus_removal_restricted_time"].get(
                    "ci_high", np.inf
                )
                < 0
                and half_life_reversal
                and all(
                    item.get("ci_high", np.inf) < 0
                    for item in half_life_reversal
                )
            )
            restored_reliance = _column(
                frame,
                "final.restoration_rho_p",
                "final.restoration_probe.rho_p",
            )
            restoration = (
                _h7_restoration_ordering(
                    frame,
                    condition,
                    restored_reliance,
                    settings=settings,
                )
                if restored_reliance
                else {"status": "unavailable"}
            )
            report["restored_proxy_rebound_ordering"] = restoration
            restoration_reversal = [
                item["contrast"]
                for item in restoration.get(
                    "reversal_minus_decorrelation_by_q_b", []
                )
                if item["contrast"].get("status") == "estimated"
            ]
            restoration_supported = bool(
                restoration.get("decorrelation_minus_removal", {}).get(
                    "ci_high", np.inf
                )
                < 0
                and restoration_reversal
                and all(
                    item.get("ci_high", np.inf) < 0
                    for item in restoration_reversal
                )
            )
            directional_reversal = bool(
                ordering["decorrelation_minus_removal_restricted_time"].get(
                    "ci_low", -np.inf
                )
                > 0
                or any(
                    item.get("ci_low", -np.inf) > 0
                    for item in half_life_reversal
                )
                or restoration.get("decorrelation_minus_removal", {}).get(
                    "ci_low", -np.inf
                )
                > 0
                or any(
                    item.get("ci_low", -np.inf) > 0
                    for item in restoration_reversal
                )
            )
            report["status"] = (
                "consistent"
                if half_life_supported and restoration_supported
                else "evidence_against"
                if directional_reversal
                else "mixed_or_inconclusive"
            )
        else:
            report["status"] = "insufficient_data"
    elif hypothesis == "h8":
        eligibility = _column(
            frame,
            "final.stage1_gate_passed",
            "final.eligible_for_primary_analysis",
        )
        if eligibility:
            eligible = _h8_primary_eligibility(frame)
            report["excluded_gate_failures"] = int(np.sum(~eligible))
            frame = frame[eligible]
        if frame.empty:
            report["status"] = "insufficient_data"
            report["inference_sufficiency"] = {
                "status": "insufficient_data",
                "observed_independent_seeds": 0,
                "minimum_inferential_seeds": MIN_INFERENTIAL_SEEDS,
                "replication_unit": "training seed",
            }
            return report
        rebound = _column(frame, "final.rebound_g0", "rebound_g0")
        reactivation = _column(frame, "events.reactivation_time", "events.reactivation.time")
        reactivation_observed = _column(frame, "events.reactivation_observed", "events.reactivation.observed")
        if rebound:
            report["rebound_contrasts"] = _h8_history_contrasts(
                frame, rebound, settings=settings
            )
            report["reactivation_restricted_time_contrasts"] = (
                _h8_history_contrasts(
                    frame,
                    reactivation,
                    observed_column=reactivation_observed,
                    settings=settings,
                )
                if reactivation
                else []
            )

            def preferred_history_rows(records: Sequence[Mapping[str, Any]]) -> tuple[str, list[Mapping[str, Any]]]:
                direct = [
                    item
                    for item in records
                    if item.get("treatment") == "old_goal"
                    and item.get("minus") == "compute_matched_sham"
                ]
                if direct:
                    return "compute_matched_sham", direct
                control = [
                    item
                    for item in records
                    if item.get("treatment") == "old_goal"
                    and item.get("minus") == "control"
                ]
                return "control", control

            def history_contrast_status(
                records: Sequence[Mapping[str, Any]], *, direction: int
            ) -> str:
                if not records:
                    return "insufficient_data"
                summaries = [item.get("contrast", {}) for item in records]
                estimated = [
                    item for item in summaries if item.get("status") == "estimated"
                ]
                if not estimated:
                    return "insufficient_data"
                if direction > 0:
                    if any(item.get("ci_high", np.inf) <= margin for item in estimated):
                        return "evidence_against"
                    support = all(item.get("ci_low", -np.inf) > margin for item in estimated)
                else:
                    if any(item.get("ci_low", -np.inf) >= 0.0 for item in estimated):
                        return "evidence_against"
                    support = all(item.get("ci_high", np.inf) < 0.0 for item in estimated)
                if support and len(estimated) == len(summaries):
                    return "consistent"
                return "mixed_or_inconclusive"

            history_comparator, history_rebound_rows = preferred_history_rows(
                report["rebound_contrasts"]
            )
            _, history_speed_rows = preferred_history_rows(
                report["reactivation_restricted_time_contrasts"]
            )
            history_rebound_status = history_contrast_status(
                history_rebound_rows, direction=1
            )
            history_speed_status = history_contrast_status(
                history_speed_rows, direction=-1
            )
            history_status = _combine_required_statuses(
                [history_rebound_status, history_speed_status]
            )
            history_column = _column(frame, "config.h8.history")
            old_frame = (
                frame[frame[history_column].eq("old_goal")].copy()
                if history_column
                else frame.iloc[0:0].copy()
            )
            scaling = _h8_scaling_report(
                old_frame,
                rebound,
                reactivation,
                margin=margin,
                settings=settings,
            )
            report["old_volume_rebound_association"] = scaling[
                "old_goal_volume"
            ]["n0_to_rebound"]["association"]
            report["old_volume_reactivation_time_association"] = scaling[
                "old_goal_volume"
            ]["n0_to_reactivation_restricted_time"]["association"]
            report["model_capacity_rebound_association"] = scaling[
                "model_capacity"
            ]["capacity_to_rebound"]["association"]
            report["model_capacity_reactivation_time_association"] = scaling[
                "model_capacity"
            ]["capacity_to_reactivation_restricted_time_optional"]["association"]
            report["scaling"] = scaling
            report["subclaims"] = {
                "history_specific_rebound_and_reactivation": {
                    "status": history_status,
                    "preferred_comparator": history_comparator,
                    "rebound": {"status": history_rebound_status},
                    "reactivation_speed": {"status": history_speed_status},
                },
                "old_goal_volume_scaling": scaling["old_goal_volume"],
                "model_capacity_scaling": scaling["model_capacity"],
            }
            report["status"] = _combine_required_statuses(
                [
                    history_status,
                    scaling["old_goal_volume"]["status"],
                    scaling["model_capacity"]["status"],
                ]
            )
        else:
            report["status"] = "insufficient_data"
    elif hypothesis == "h9":
        controls = _column(frame, "final.single_rule_controls_passed")
        if controls:
            controls_enabled = _column(
                frame,
                "final.single_rule_calibration.enabled",
                "config.h9.include_single_rule_controls",
            )
            eligible = frame[controls].eq(True)
            if controls_enabled:
                eligible |= ~frame[controls_enabled].eq(True)
            report["excluded_single_rule_failures"] = int(np.sum(~eligible))
            frame = frame[eligible].copy()
        if frame.empty:
            report["status"] = "insufficient_data"
            report["inference_sufficiency"] = {
                "status": "insufficient_data",
                "observed_independent_seeds": 0,
                "minimum_inferential_seeds": MIN_INFERENTIAL_SEEDS,
                "replication_unit": "training seed",
            }
            return report
        switching = _column(frame, "final.strict_context_switching", "strict_context_switching")
        update_mode = _column(frame, "config.update.mode", "model.update_mode")
        architecture = frame[~frame[update_mode].eq("subspace")] if update_mode else frame
        subspace = frame[frame[update_mode].eq("subspace")] if update_mode else frame.iloc[0:0]
        model_capacity = _column(
            architecture, "model.total_parameters", "config.model.width"
        )
        update_capacity = _column(
            subspace, "model.trainable_parameters", "config.update.budget"
        )
        report["model_capacity_switching_association"] = _seedwise_spearman(
            architecture,
            model_capacity,
            switching,
            bootstrap_seed=1901,
            settings=settings,
        )
        report["update_capacity_switching_association"] = _seedwise_spearman(
            subspace,
            update_capacity,
            switching,
            bootstrap_seed=1907,
            settings=settings,
        )
        normal = _column(
            frame, "final.normal.target_accuracy", "final.normal_target_accuracy"
        )
        removed = _column(frame, "final.removed.target_accuracy")
        randomized = _column(frame, "final.randomized.target_accuracy")
        if normal and removed:
            frame["_context_removed_accuracy_drop"] = frame[normal].astype(float) - frame[removed].astype(float)
            report["context_removed_accuracy_drop"] = _seed_ci(
                frame,
                "_context_removed_accuracy_drop",
                settings=settings,
            )
        if normal and randomized:
            frame["_context_randomized_accuracy_drop"] = frame[normal].astype(float) - frame[randomized].astype(float)
            report["context_randomized_accuracy_drop"] = _seed_ci(
                frame,
                "_context_randomized_accuracy_drop",
                settings=settings,
            )
        mismatch_scores = []
        mismatch_obedience: str | None = None
        for name in ("mismatch_0_1", "mismatch_1_0"):
            observed_proxy = _column(frame, f"final.{name}.rho_p_observed_context")
            environment_proxy = _column(frame, f"final.{name}.rho_p_environment")
            if observed_proxy and environment_proxy:
                mismatch_scores.append(frame[observed_proxy].astype(float) - frame[environment_proxy].astype(float))
        if mismatch_scores:
            frame["_mismatch_context_obedience"] = pd.concat(mismatch_scores, axis=1).mean(axis=1)
            mismatch_obedience = "_mismatch_context_obedience"
            report["directional_mismatch_context_obedience"] = _seed_ci(
                frame,
                "_mismatch_context_obedience",
                settings=settings,
            )
        sensitivity_columns = {
            key: _column(frame, f"final.proxy_sensitivity_matrix.{context}.{proxy}.absolute_logit_change")
            for key, context, proxy in (
                ("c0p0", "C0", "P0"),
                ("c0p1", "C0", "P1"),
                ("c1p0", "C1", "P0"),
                ("c1p1", "C1", "P1"),
            )
        }
        causal_selectivity: str | None = None
        if all(sensitivity_columns.values()):
            assert all(value is not None for value in sensitivity_columns.values())
            frame["_causal_selector_selectivity"] = 0.5 * (
                frame[sensitivity_columns["c0p0"]].astype(float)
                - frame[sensitivity_columns["c0p1"]].astype(float)
                + frame[sensitivity_columns["c1p1"]].astype(float)
                - frame[sensitivity_columns["c1p0"]].astype(float)
            )
            causal_selectivity = "_causal_selector_selectivity"
            report["causal_selector_selectivity"] = _seed_ci(
                frame,
                "_causal_selector_selectivity",
                settings=settings,
            )

        model_trend_status = _directional_trend_status(
            report["model_capacity_switching_association"],
            direction=1,
            equivalence_margin=margin,
        )
        update_trend_status = _directional_trend_status(
            report["update_capacity_switching_association"],
            direction=1,
            equivalence_margin=margin,
        )
        capacity_trend_status = _combine_required_statuses(
            [model_trend_status, update_trend_status]
        )
        seed_column = _column(frame, "config.seed", "seed")
        assert seed_column is not None
        architecture_with_diagnostics = (
            frame[~frame[update_mode].eq("subspace")].copy()
            if update_mode
            else frame.copy()
        )
        subspace_with_diagnostics = (
            frame[frame[update_mode].eq("subspace")].copy()
            if update_mode
            else frame.iloc[0:0].copy()
        )
        highest_model = _highest_capacity_per_seed(
            architecture_with_diagnostics, model_capacity, seed_column
        )
        highest_update = _highest_capacity_per_seed(
            subspace_with_diagnostics, update_capacity, seed_column
        )
        selector_gate = _column(frame, "final.selector_gate_passed")
        highest_model_report = _h9_high_capacity_report(
            highest_model,
            capacity_column=model_capacity,
            seed_column=seed_column,
            switching=switching,
            normal_accuracy=normal,
            mismatch_obedience=mismatch_obedience,
            causal_selectivity=causal_selectivity,
            selector_gate=selector_gate,
            margin=margin,
            bootstrap_seed=1913,
            settings=settings,
        )
        highest_update_report = _h9_high_capacity_report(
            highest_update,
            capacity_column=update_capacity,
            seed_column=seed_column,
            switching=switching,
            normal_accuracy=normal,
            mismatch_obedience=mismatch_obedience,
            causal_selectivity=causal_selectivity,
            selector_gate=selector_gate,
            margin=margin,
            bootstrap_seed=1931,
            settings=settings,
        )
        report["highest_capacity_mastery"] = {
            "model_capacity": highest_model_report,
            "update_capacity": highest_update_report,
        }
        report["subclaims"] = {
            "capacity_trends": {
                "status": capacity_trend_status,
                "model_capacity": {"status": model_trend_status},
                "update_capacity": {"status": update_trend_status},
            },
            "highest_model_capacity_selector_mastery": highest_model_report,
            "highest_update_capacity_selector_mastery": highest_update_report,
        }
        report["status"] = _combine_required_statuses(
            [
                capacity_trend_status,
                highest_model_report["status"],
                highest_update_report["status"],
            ]
        )
    else:
        raise ValueError(f"Unknown hypothesis {hypothesis!r}")
    if hypothesis in {"h1", "h2", "h3", "h4"}:
        report["level_confirmation"] = _level_confirmation_report(
            frame, hypothesis, margin=margin, settings=settings
        )
    guard_seed_column = _column(frame, "config.seed", "seed")
    reported_seeds = (
        int(frame[guard_seed_column].nunique()) if guard_seed_column else None
    )
    inference_replication_unit = "training seed"
    if hypothesis == "h5":
        matched_seeds = (
            report.get("update_capacity_thresholds", {})
            .get("primary_comparison", {})
            .get("matched_seed_count")
        )
        if isinstance(matched_seeds, (int, np.integer)):
            reported_seeds = int(matched_seeds)
        inference_replication_unit = "training seed matched across the primary H5 panel"
    minimum_column = _column(frame, "config.evaluation.minimum_inferential_seeds")
    configured_minimum = (
        max(
            MIN_INFERENTIAL_SEEDS,
            int(pd.to_numeric(frame[minimum_column], errors="coerce").dropna().max()),
        )
        if minimum_column and frame[minimum_column].notna().any()
        else MIN_INFERENTIAL_SEEDS
    )
    sufficient = (
        isinstance(reported_seeds, (int, np.integer))
        and int(reported_seeds) >= configured_minimum
    )
    report["inference_sufficiency"] = {
        "status": "sufficient" if sufficient else "insufficient_data",
        "observed_independent_seeds": reported_seeds,
        "minimum_inferential_seeds": configured_minimum,
        "replication_unit": inference_replication_unit,
    }
    if not sufficient:
        report["descriptive_status_before_seed_guard"] = report.get("status")
        report["status"] = "insufficient_data"
    return report


def render_figures(summaries: pd.DataFrame, metrics: pd.DataFrame, output: str | Path) -> list[Path]:
    """Produce compact, deterministic diagnostic and primary-result figures."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    def line_figure(
        frame: pd.DataFrame,
        *,
        x: str | None,
        y: str | None,
        group: str | None,
        filename: str,
        xlabel: str,
        ylabel: str,
        log_x: bool = False,
    ) -> None:
        if frame.empty or x is None or y is None or x not in frame or y not in frame:
            return
        columns = [x, y] + ([group] if group and group in frame else [])
        clean = frame[columns].dropna(subset=[x, y]).copy()
        if clean.empty:
            return
        figure, axis = plt.subplots(figsize=(7, 4.5))
        pieces = clean.groupby(group) if group and group in clean else [("all", clean)]
        for label, piece in pieces:
            aggregated = piece.groupby(x)[y].agg(["mean", "sem"]).reset_index().sort_values(x)
            axis.plot(aggregated[x], aggregated["mean"], marker="o", label=str(label))
            error = aggregated["sem"].fillna(0.0)
            axis.fill_between(aggregated[x], aggregated["mean"] - error, aggregated["mean"] + error, alpha=0.18)
        if log_x and np.all(clean[x].astype(float) > 0):
            axis.set_xscale("log")
        axis.set(xlabel=xlabel, ylabel=ylabel)
        if group and group in clean:
            axis.legend(frameon=False, fontsize="small")
        figure.tight_layout()
        path = destination / filename
        figure.savefig(path, dpi=180)
        plt.close(figure)
        created.append(path)

    if not metrics.empty and {"global_step", "metric", "value"} <= set(metrics.columns):
        trajectory = metrics[metrics["metric"].isin(["rho_y", "rho_p", "train_accuracy", "ood_accuracy"])]
        if not trajectory.empty:
            grouped = trajectory.groupby(["global_step", "metric"])["value"].agg(["mean", "sem"]).reset_index()
            figure, axis = plt.subplots(figsize=(7, 4.5))
            for metric, part in grouped.groupby("metric"):
                axis.plot(part["global_step"], part["mean"], label=metric)
                axis.fill_between(part["global_step"], part["mean"] - part["sem"], part["mean"] + part["sem"], alpha=0.2)
            axis.set_xscale("symlog", linthresh=1)
            axis.set(xlabel="optimizer step", ylabel="seed-level mean", ylim=(-0.02, 1.02))
            axis.legend(frameon=False)
            figure.tight_layout()
            path = destination / "learning_curves.png"
            figure.savefig(path, dpi=180)
            plt.close(figure)
            created.append(path)

        if "experiment" in metrics:
            h3_metrics = metrics[
                (metrics["experiment"] == "h3")
                & metrics["metric"].isin(["rho_y", "rho_p", "target_accuracy"])
                & (metrics.get("split", "") == "conflict")
            ]
            if not h3_metrics.empty:
                grouped = h3_metrics.groupby(["global_step", "metric"])["value"].agg(["mean", "sem"]).reset_index()
                figure, axis = plt.subplots(figsize=(7, 4.5))
                for metric_name, part in grouped.groupby("metric"):
                    axis.plot(part["global_step"], part["mean"], label=metric_name)
                axis.set_xscale("symlog", linthresh=1)
                axis.set(xlabel="optimizer step", ylabel="seed-level mean", ylim=(-0.02, 1.02))
                axis.legend(frameon=False)
                figure.tight_layout()
                path = destination / "h3_acquisition_dynamics.png"
                figure.savefig(path, dpi=180)
                plt.close(figure)
                created.append(path)

    h = _column(summaries, "config.experiment.hypothesis")
    q = _column(summaries, "config.data.q")
    k = _column(summaries, "config.data.k")
    delta = _column(
        summaries,
        "final.conflict.delta_rho",
        "final.competition_conflict.delta_rho",
        "final.conflict_unseen.delta_rho",
        "final.delta_rho",
        "delta_rho",
    )
    if h and q and k and delta:
        h1 = summaries[summaries[h] == "h1"]
        if not h1.empty:
            capacity_candidates = (
                "model.total_parameters",
                "model.trainable_parameters",
                "config.update.budget",
                "config.model.width",
            )
            capacity = next(
                (
                    name
                    for name in capacity_candidates
                    if name in h1 and h1[name].dropna().nunique() > 1
                ),
                _column(h1, *capacity_candidates),
            )
            panels = (
                list(h1.groupby(capacity, dropna=False, sort=False))
                if capacity
                else [("all", h1)]
            )
            columns = min(3, len(panels))
            rows = int(np.ceil(len(panels) / columns))
            figure, axes = plt.subplots(
                rows,
                columns,
                figsize=(4.2 * columns, 3.7 * rows),
                squeeze=False,
                sharex=True,
                sharey=True,
            )
            images = []
            used_axes = []
            for axis, (capacity_value, panel) in zip(axes.ravel(), panels, strict=False):
                table = panel.pivot_table(index=k, columns=q, values=delta, aggfunc="mean")
                if table.empty:
                    axis.set_visible(False)
                    continue
                image = axis.imshow(
                    table.to_numpy(),
                    aspect="auto",
                    vmin=-1,
                    vmax=1,
                    cmap="coolwarm",
                    origin="lower",
                )
                images.append(image)
                used_axes.append(axis)
                axis.set_xticks(
                    range(len(table.columns)), [str(value) for value in table.columns]
                )
                axis.set_yticks(
                    range(len(table.index)), [str(value) for value in table.index]
                )
                axis.set(
                    xlabel="proxy accuracy q",
                    ylabel="interaction degree k",
                    title=f"B={capacity_value}",
                )
            for axis in axes.ravel()[len(panels) :]:
                axis.set_visible(False)
            if images:
                figure.suptitle("H1: intended minus proxy reliance by capacity")
                figure.colorbar(images[0], ax=used_axes, label="rho_Y - rho_P", shrink=0.85)
                figure.subplots_adjust(top=0.88, right=0.9, hspace=0.35, wspace=0.3)
                path = destination / "h1_phase_diagram.png"
                figure.savefig(path, dpi=180)
                created.append(path)
            plt.close(figure)

    if h:
        h2 = summaries[summaries[h] == "h2"].copy()
        h2_mode = _column(h2, "config.update.mode")
        h2_x = _column(
            h2, "model.exact_calibration.total_parameters", "config.model.width"
        )
        h2_y = _column(h2, "final.decoder_accuracy")
        architecture_h2 = (
            h2[h2[h2_mode].astype(str) != "subspace"].copy()
            if h2_mode
            else h2.copy()
        )
        facet_columns = [
            column
            for column in (
                h2_mode,
                _column(architecture_h2, "config.model.activation"),
                _column(architecture_h2, "config.model.residual"),
            )
            if column
        ]
        if (
            not architecture_h2.empty
            and h2_x
            and h2_y
            and h2_x in architecture_h2
            and h2_y in architecture_h2
        ):
            clean = architecture_h2.dropna(subset=[h2_x, h2_y])
            panels = (
                list(clean.groupby(facet_columns, dropna=False, sort=False))
                if facet_columns
                else [("all", clean)]
            )
            if panels:
                columns = min(3, len(panels))
                rows = int(np.ceil(len(panels) / columns))
                figure, axes = plt.subplots(
                    rows,
                    columns,
                    figsize=(4.5 * columns, 3.6 * rows),
                    squeeze=False,
                    sharex=True,
                    sharey=True,
                )
                k_column = _column(clean, "config.data.k")
                depth_column = _column(clean, "config.model.depth")
                line_columns = [column for column in (k_column, depth_column) if column]
                for axis, (facet_values, panel) in zip(
                    axes.ravel(), panels, strict=False
                ):
                    pieces = (
                        panel.groupby(line_columns, dropna=False)
                        if line_columns
                        else [("all", panel)]
                    )
                    for label, piece in pieces:
                        aggregated = (
                            piece.groupby(h2_x)[h2_y]
                            .agg(["mean", "sem"])
                            .reset_index()
                            .sort_values(h2_x)
                        )
                        axis.plot(
                            aggregated[h2_x],
                            aggregated["mean"],
                            marker="o",
                            label=str(label),
                        )
                        error = aggregated["sem"].fillna(0.0)
                        axis.fill_between(
                            aggregated[h2_x],
                            aggregated["mean"] - error,
                            aggregated["mean"] + error,
                            alpha=0.16,
                        )
                    if np.all(pd.to_numeric(panel[h2_x], errors="coerce") > 0):
                        axis.set_xscale("log")
                    facet_values = (
                        facet_values
                        if isinstance(facet_values, tuple)
                        else (facet_values,)
                    )
                    title = ", ".join(
                        f"{name.removeprefix('config.')}={value}"
                        for name, value in zip(
                            facet_columns, facet_values, strict=True
                        )
                    )
                    axis.set(
                        xlabel="total model parameters",
                        ylabel="exact decoder accuracy",
                        title=title,
                        ylim=(-0.02, 1.02),
                    )
                for axis in axes.ravel()[len(panels) :]:
                    axis.set_visible(False)
                axes.ravel()[0].legend(
                    title="(k, depth)", frameon=False, fontsize="x-small"
                )
                figure.suptitle(
                    "H2 architecture calibration (faceted; no architecture pooling)"
                )
                figure.tight_layout(rect=(0, 0, 1, 0.96))
                path = destination / "h2_decoder_thresholds.png"
                figure.savefig(path, dpi=180)
                plt.close(figure)
                created.append(path)

        update_h2 = (
            h2[h2[h2_mode].astype(str) == "subspace"].copy()
            if h2_mode
            else h2.iloc[0:0].copy()
        )
        line_figure(
            update_h2,
            x=_column(update_h2, "model.exact_calibration.trainable_parameters"),
            y=_column(update_h2, "final.decoder_accuracy"),
            group=_column(update_h2, "config.data.k"),
            filename="h2_decoder_update_thresholds.png",
            xlabel="trainable update parameters",
            ylabel="exact decoder accuracy",
            log_x=True,
        )

        h4 = summaries[summaries[h] == "h4"].copy()
        h4_x = _column(h4, "data.u_conflict", "data.realized_u_conflict")
        h4_y = _column(h4, "final.conflict_unseen.rho_y")
        h4_k = _column(h4, "config.data.k")
        h4_count = _column(h4, "data.n_conflict", "config.h4.n_conflict")
        h4_condition = _column(h4, "config.h4.condition")
        if h4_x and h4_y and h4_count:
            h4 = h4.dropna(subset=[h4_x, h4_y, h4_count])
            h4 = h4[pd.to_numeric(h4[h4_count], errors="coerce") > 0]
            if h4_condition:
                h4 = h4[h4[h4_condition].isin(["concentrated", "diverse"])]
            counts = sorted(h4[h4_count].unique(), key=float)
            if counts:
                columns = min(3, len(counts))
                rows = int(np.ceil(len(counts) / columns))
                figure, axes = plt.subplots(
                    rows,
                    columns,
                    figsize=(4.4 * columns, 3.7 * rows),
                    squeeze=False,
                    sharey=True,
                )
                for axis, count in zip(axes.ravel(), counts, strict=False):
                    panel = h4[h4[h4_count] == count]
                    pieces = panel.groupby(h4_k) if h4_k else [("all", panel)]
                    for label, piece in pieces:
                        aggregated = (
                            piece.groupby(h4_x)[h4_y]
                            .agg(["mean", "sem"])
                            .reset_index()
                            .sort_values(h4_x)
                        )
                        axis.plot(
                            aggregated[h4_x],
                            aggregated["mean"],
                            marker="o",
                            label=f"k={label}",
                        )
                        error = aggregated["sem"].fillna(0.0)
                        axis.fill_between(
                            aggregated[h4_x],
                            aggregated["mean"] - error,
                            aggregated["mean"] + error,
                            alpha=0.18,
                        )
                    axis.set_xscale("log")
                    axis.set(
                        xlabel="unique conflict contexts",
                        ylabel="unseen-conflict intended reliance",
                        title=f"N_conflict={count}",
                        ylim=(-0.02, 1.02),
                    )
                for axis in axes.ravel()[len(counts) :]:
                    axis.set_visible(False)
                axes.ravel()[0].legend(frameon=False, fontsize="small")
                figure.suptitle("H4: diversity effect at matched conflict counts")
                figure.tight_layout(rect=(0, 0, 1, 0.95))
                path = destination / "h4_conflict_diversity.png"
                figure.savefig(path, dpi=180)
                plt.close(figure)
                created.append(path)

        h5 = summaries[summaries[h] == "h5"]
        h5_algorithm = _column(h5, "config.h5.algorithm", "algorithm")
        h5_entropy = _column(h5, "config.h5.nuisance_entropy")
        if h5_algorithm:
            h5 = h5.copy()
            h5["_plot_group"] = h5[h5_algorithm].astype(str)
            if h5_entropy:
                h5["_plot_group"] += " / H=" + h5[h5_entropy].astype(str)
        line_figure(
            h5,
            x=_column(h5, "model.trainable_parameters", "costs.trainable_scalars"),
            y=_column(h5, "final.rho_y"),
            group="_plot_group" if "_plot_group" in h5 else h5_algorithm,
            filename="h5_update_capacity.png",
            xlabel="trainable actor scalars",
            ylabel="OOD intended reliance",
            log_x=True,
        )

        h6 = summaries[summaries[h] == "h6"]
        h6_objective = _column(h6, "noise.objective_uses_location")
        if h6_objective:
            h6 = h6[h6[h6_objective].eq(True)].copy()
        for column, value in (
            (_column(h6, "config.h6.reward_mode"), "dense_fixed_horizon"),
            (_column(h6, "config.h6.algorithm", "algorithm"), "clean_sft"),
            (_column(h6, "config.h6.location", "noise.location"), "observation"),
        ):
            if column:
                h6 = h6[h6[column].astype(str) == value].copy()
        h6_scale = _column(h6, "config.h6.scale", "noise.scale")
        if h6_scale and np.any(np.isclose(h6[h6_scale].astype(float), 1.0)):
            h6 = h6[np.isclose(h6[h6_scale].astype(float), 1.0)].copy()
        h6_structure = _column(h6, "config.h6.structure")
        h6_n_train = _column(h6, "config.data.n_train")
        if h6_structure:
            h6["_plot_group"] = h6[h6_structure].astype(str)
            if h6_n_train:
                h6["_plot_group"] += " / N=" + h6[h6_n_train].astype(str)
        line_figure(
            h6,
            x=_column(
                h6,
                "recurrence.realized_episode_visits_per_state",
                "recurrence.realized_visits_per_state",
                "recurrence.mean_episode_visits_per_state",
                "recurrence.mean_visits_per_state",
                "config.h6.visits_per_state",
            ),
            y=_column(h6, "final.rho_y"),
            group="_plot_group" if "_plot_group" in h6 else h6_structure,
            filename="h6_noise_structure.png",
            xlabel="episode visits per semantic state (clean SFT, observation noise, sigma=1)",
            ylabel="OOD intended reliance",
            log_x=True,
        )

        h7 = summaries[summaries[h] == "h7"]
        h7_ablation = _column(h7, "config.h7.weight_decay_ablation")
        if h7_ablation:
            h7 = h7[~h7[h7_ablation].eq(True)].copy()
        h7_eligibility = _column(h7, "final.eligible_for_primary_analysis")
        if h7_eligibility:
            h7 = h7[h7[h7_eligibility].eq(True)].copy()
        h7_condition = _column(h7, "config.h7.condition")
        h7_time = _column(h7, "events.half_life_time", "events.half_life.time")
        if h7_condition and h7_time and not h7.empty:
            h7_q_b = _column(h7, "config.h7.q_b")
            groups = [h7_condition] + ([h7_q_b] if h7_q_b else [])
            table = h7.groupby(groups, dropna=False)[h7_time].agg(["mean", "count"]).reset_index()
            labels = table[h7_condition].astype(str)
            if h7_q_b:
                labels = labels + " (qB=" + table[h7_q_b].astype(str) + ")"
            figure, axis = plt.subplots(figsize=(6.5, 4.5))
            axis.bar(labels, table["mean"])
            axis.tick_params(axis="x", labelrotation=30)
            axis.set(
                xlabel="Phase-B intervention",
                ylabel="restricted mean half-life (censored at horizon)",
            )
            figure.tight_layout()
            path = destination / "h7_unlearning_half_life.png"
            figure.savefig(path, dpi=180)
            plt.close(figure)
            created.append(path)

        h8 = summaries[summaries[h] == "h8"]
        h8_eligibility = _column(
            h8,
            "final.stage1_gate_passed",
            "final.eligible_for_primary_analysis",
        )
        if h8_eligibility:
            h8 = h8[_h8_primary_eligibility(h8)].copy()
        h8_history = _column(h8, "config.h8.history")
        h8_perturbation = _column(h8, "config.h8.perturbation")
        h8_stage1_mode = _column(h8, "config.h8.stage1_mode")
        if h8_history:
            h8["_plot_group"] = h8[h8_history].astype(str)
            for extra in (h8_perturbation, h8_stage1_mode):
                if extra:
                    h8["_plot_group"] += " / " + h8[extra].astype(str)
        line_figure(
            h8,
            x=_column(h8, "config.h8.n0"),
            y=_column(h8, "final.rebound_g0"),
            group="_plot_group" if "_plot_group" in h8 else h8_history,
            filename="h8_rebound.png",
            xlabel="old-goal examples N0",
            ylabel="old-goal rebound",
            log_x=True,
        )

        h9 = summaries[summaries[h] == "h9"]
        h9_controls = _column(h9, "final.single_rule_controls_passed")
        h9_enabled = _column(
            h9, "final.single_rule_calibration.enabled", "config.h9.include_single_rule_controls"
        )
        if h9_controls:
            h9_mask = h9[h9_controls].eq(True)
            if h9_enabled:
                h9_mask |= ~h9[h9_enabled].eq(True)
            h9 = h9[h9_mask].copy()
        h9_mode = _column(h9, "config.update.mode", "model.update_mode")
        h9_architecture = h9[~h9[h9_mode].eq("subspace")] if h9_mode else h9
        h9_update = h9[h9[h9_mode].eq("subspace")] if h9_mode else h9.iloc[0:0]
        line_figure(
            h9_architecture,
            x=_column(h9_architecture, "model.total_parameters", "config.model.width"),
            y=_column(h9_architecture, "final.strict_context_switching"),
            group=_column(h9_architecture, "config.model.depth"),
            filename="h9_model_capacity_switching.png",
            xlabel="total model parameters",
            ylabel="strict context switching",
            log_x=True,
        )
        line_figure(
            h9_update,
            x=_column(h9_update, "model.trainable_parameters", "config.update.budget"),
            y=_column(h9_update, "final.strict_context_switching"),
            group=None,
            filename="h9_update_capacity_switching.png",
            xlabel="trainable update scalars",
            ylabel="strict context switching",
            log_x=True,
        )
    return created


def analyze_root(root: str | Path, output: str | Path | None = None) -> dict[str, Any]:
    summaries, metrics = collect_results(root)
    destination = Path(output) if output else Path(root) / "analysis"
    destination.mkdir(parents=True, exist_ok=True)
    reports = {}
    if not summaries.empty:
        h_column = _column(summaries, "config.experiment.hypothesis")
        hypotheses = sorted(set(summaries[h_column])) if h_column else []
        for hypothesis in hypotheses:
            reports[str(hypothesis)] = evaluate_hypothesis(
                summaries, str(hypothesis), metrics=metrics
            )
    figures = render_figures(summaries, metrics, destination / "figures")
    summaries.to_csv(destination / "run_summaries.csv", index=False)
    metrics.to_csv(destination / "metrics.csv", index=False)
    payload = {"hypotheses": reports, "figures": [str(path) for path in figures], "n_runs": len(summaries)}
    write_json(destination / "hypothesis_report.json", payload)
    return payload
