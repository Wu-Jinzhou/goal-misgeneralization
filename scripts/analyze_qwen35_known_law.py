#!/usr/bin/env python3
"""Analyze the frozen Qwen3.5 known-Law panel at the six-seed block level."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import product
from pathlib import Path
from typing import Any

from goalzendo.analysis import AnalysisPanel, load_analysis_panel
from goalzendo.artifacts import discover_runs
from goalzendo.config import get_path, load_config
from goalzendo.runner import RunSpec, build_plan

ROOT = Path(__file__).resolve().parents[1]
MAIN_CONFIG = ROOT / "configs" / "goalzendo" / "qwen35_known_law_main.yaml"

SEEDS = (21011, 21013, 21017, 21019, 21023, 21031)
MODELS = ("Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B")
ALGORITHMS = ("sft", "outcome_rl")
LAWS = ("parity", "majority")
Q_P_VALUES = (0.95, 1.0)
FINAL_STEP = 1000
BOOTSTRAP_RESAMPLES = len(SEEDS) ** len(SEEDS)
SIGN_FLIP_ASSIGNMENTS = 2 ** len(SEEDS)


class Qwen35AnalysisError(ValueError):
    """Raised when the frozen panel or an estimand is not exactly identified."""


def _condition(spec_or_config: RunSpec | Mapping[str, Any]) -> tuple[int, str, str, str, float]:
    if isinstance(spec_or_config, RunSpec):
        seed = spec_or_config.seed
        config = spec_or_config.config
    else:
        config = spec_or_config
        seed = int(get_path(config, "seed"))
    return (
        int(seed),
        str(get_path(config, "model.name")),
        str(get_path(config, "train.algorithm")),
        str(get_path(config, "data.rule_family")),
        float(get_path(config, "data.q_p")),
    )


def _expected_plan(config: Mapping[str, Any]) -> dict[str, RunSpec]:
    plan = build_plan(config)
    expected_conditions = set(product(SEEDS, MODELS, ALGORITHMS, LAWS, Q_P_VALUES))
    observed_conditions = {_condition(spec) for spec in plan}
    errors: list[str] = []
    if str(get_path(config, "experiment.id")) != "qwen35_known_law_main":
        errors.append("experiment.id is not qwen35_known_law_main")
    if len(plan) != 96:
        errors.append(f"plan has {len(plan)} rows rather than 96")
    if len({spec.plan_key for spec in plan}) != len(plan):
        errors.append("plan keys are not unique")
    if observed_conditions != expected_conditions:
        errors.append("model/algorithm/Law/q_P/seed cells differ from the frozen 96-row design")
    if Counter(spec.seed for spec in plan) != Counter({seed: 16 for seed in SEEDS}):
        errors.append("the six paired seed blocks are incomplete")
    for spec in plan:
        if get_path(spec.config, "run.launch_guard") is not None:
            errors.append("the main design unexpectedly has a launch guard")
            break
        if int(get_path(spec.config, "train.steps")) != FINAL_STEP:
            errors.append("the main design does not end at step 1000")
            break
        if tuple(get_path(spec.config, "evaluation.prompt_views")) != (
            "full",
            "audit_law_matched",
        ):
            errors.append("the registered prompt views changed")
            break
        if tuple(get_path(spec.config, "evaluation.causal_prompt_views")) != ("full",):
            errors.append("the registered causal prompt view changed")
            break
    if errors:
        raise Qwen35AnalysisError("invalid expected main config: " + "; ".join(errors))
    return {spec.plan_key: spec for spec in plan}


def _load_exact_panel(artifacts: Path, config: Mapping[str, Any]) -> AnalysisPanel:
    all_runs = discover_runs(artifacts, completed_only=False)
    completed_runs = discover_runs(artifacts, completed_only=True)
    incomplete = sorted(set(all_runs) - set(completed_runs))
    if incomplete:
        raise Qwen35AnalysisError(
            f"artifact root contains {len(incomplete)} incomplete run director"
            f"{'y' if len(incomplete) == 1 else 'ies'}"
        )
    return load_analysis_panel(
        artifacts,
        expected_config=config,
        require_complete_metrics=True,
        required_prompt_views=("full", "audit_law_matched"),
        allow_incomplete_runs=False,
    )


def _validate_run_rows(panel: AnalysisPanel, expected: Mapping[str, RunSpec]) -> None:
    rows = panel.runs.to_dict(orient="records")
    run_ids = [str(row["run_id"]) for row in rows]
    plan_keys = [str(row["plan_key"]) for row in rows]
    if len(rows) != 96:
        raise Qwen35AnalysisError(f"completed panel has {len(rows)} rows rather than 96")
    if len(set(run_ids)) != len(run_ids):
        raise Qwen35AnalysisError("completed panel contains duplicate run IDs")
    if len(set(plan_keys)) != len(plan_keys):
        raise Qwen35AnalysisError("completed panel contains duplicate plan rows")
    missing = sorted(set(expected) - set(plan_keys))
    unexpected = sorted(set(plan_keys) - set(expected))
    if missing or unexpected:
        raise Qwen35AnalysisError(
            f"completed plan rows differ from registration: {len(missing)} missing, {len(unexpected)} extra"
        )
    for row in rows:
        plan_key = str(row["plan_key"])
        config = row.get("config")
        if not isinstance(config, Mapping):
            raise Qwen35AnalysisError(f"{plan_key} lacks a resolved config")
        if _condition(config) != _condition(expected[plan_key]):
            raise Qwen35AnalysisError(f"{plan_key} has a condition inconsistent with its plan row")


def _finite_unit_interval(value: Any, label: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise Qwen35AnalysisError(f"{label} must be finite and lie in [0, 1]")
    return numeric


def _extract_endpoints(panel: AnalysisPanel, expected: Mapping[str, RunSpec]) -> list[dict[str, Any]]:
    _validate_run_rows(panel, expected)
    required_behavior = {
        "run_id",
        "step",
        "prompt_view",
        "split",
        "panel",
        "rho_y",
        "rho_p",
        "rho_q",
    }
    required_causal = {
        "run_id",
        "step",
        "prompt_view",
        "split",
        "target",
        "action_flip_rate",
    }
    if not required_behavior.issubset(panel.trajectory.columns):
        missing = sorted(required_behavior - set(panel.trajectory.columns))
        raise Qwen35AnalysisError(f"trajectory table lacks columns: {missing}")
    if not required_causal.issubset(panel.causal_effects.columns):
        missing = sorted(required_causal - set(panel.causal_effects.columns))
        raise Qwen35AnalysisError(f"causal table lacks columns: {missing}")

    behavior = panel.trajectory[
        (panel.trajectory["step"] == FINAL_STEP)
        & (panel.trajectory["prompt_view"] == "full")
        & (panel.trajectory["split"] == "final_factorial")
        & (panel.trajectory["panel"] == "conflict")
    ]
    causal = panel.causal_effects[
        (panel.causal_effects["step"] == FINAL_STEP)
        & (panel.causal_effects["prompt_view"] == "full")
        & (panel.causal_effects["split"] == "final_factorial")
        & panel.causal_effects["target"].isin(("y", "p", "q", "d"))
    ]
    behavior_counts = Counter(str(value) for value in behavior["run_id"])
    causal_counts = Counter(str(value) for value in causal["run_id"])

    endpoints: list[dict[str, Any]] = []
    for run in panel.runs.to_dict(orient="records"):
        run_id = str(run["run_id"])
        plan_key = str(run["plan_key"])
        if behavior_counts[run_id] != 1:
            raise Qwen35AnalysisError(
                f"{run_id} has {behavior_counts[run_id]} final full-view conflict rows; expected one"
            )
        if causal_counts[run_id] != 4:
            raise Qwen35AnalysisError(
                f"{run_id} has {causal_counts[run_id]} final full-view causal rows; expected four"
            )
        behavior_row = behavior[behavior["run_id"] == run_id].iloc[0]
        causal_rows = causal[causal["run_id"] == run_id]
        target_counts = Counter(str(value) for value in causal_rows["target"])
        if target_counts != Counter({"y": 1, "p": 1, "q": 1, "d": 1}):
            raise Qwen35AnalysisError(f"{run_id} does not have one causal row for each target")
        flips = {
            str(row["target"]): _finite_unit_interval(
                row["action_flip_rate"], f"{run_id} flip_{row['target']}"
            )
            for row in causal_rows.to_dict(orient="records")
        }
        rho_y = _finite_unit_interval(behavior_row["rho_y"], f"{run_id} rho_y")
        rho_p = _finite_unit_interval(behavior_row["rho_p"], f"{run_id} rho_p")
        rho_q = _finite_unit_interval(behavior_row["rho_q"], f"{run_id} rho_q")
        seed, model, algorithm, law, q_p = _condition(expected[plan_key])
        endpoints.append(
            {
                "seed": seed,
                "model": model,
                "algorithm": algorithm,
                "law": law,
                "q_p": q_p,
                "run_id": run_id,
                "plan_key": plan_key,
                "conflict_agreement": {
                    "rho_y": rho_y,
                    "rho_p": rho_p,
                    "rho_q": rho_q,
                },
                "causal_action_flip": {
                    "flip_y": flips["y"],
                    "flip_p": flips["p"],
                    "flip_q": flips["q"],
                    "flip_d": flips["d"],
                },
                "primary_d_rho_p_minus_rho_y": rho_p - rho_y,
                "causal_companion_flip_p_minus_flip_y": flips["p"] - flips["y"],
            }
        )
    return sorted(
        endpoints,
        key=lambda row: (
            int(row["seed"]),
            str(row["model"]),
            str(row["algorithm"]),
            str(row["law"]),
            float(row["q_p"]),
        ),
    )


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def _linear_percentile(sorted_values: Sequence[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _bootstrap_interval(values: Sequence[float]) -> tuple[float, float]:
    if len(values) != len(SEEDS):
        raise Qwen35AnalysisError("bootstrap requires exactly six seed-block values")
    estimates = sorted(
        _mean(tuple(values[index] for index in indices))
        for indices in product(range(len(SEEDS)), repeat=len(SEEDS))
    )
    if len(estimates) != BOOTSTRAP_RESAMPLES:
        raise AssertionError("exhaustive bootstrap enumeration is incomplete")
    return _linear_percentile(estimates, 0.025), _linear_percentile(estimates, 0.975)


def _sign_flip_p_value(values: Sequence[float]) -> float:
    if len(values) != len(SEEDS):
        raise Qwen35AnalysisError("sign-flip test requires exactly six seed-block values")
    observed = abs(_mean(values))
    exceedances = sum(
        abs(_mean(tuple(sign * value for sign, value in zip(signs, values, strict=True)))) >= observed
        for signs in product((-1.0, 1.0), repeat=len(SEEDS))
    )
    return exceedances / SIGN_FLIP_ASSIGNMENTS


def _summarize_seed_blocks(values_by_seed: Mapping[int, float]) -> dict[str, Any]:
    if set(values_by_seed) != set(SEEDS):
        missing = sorted(set(SEEDS) - set(values_by_seed))
        extra = sorted(set(values_by_seed) - set(SEEDS))
        raise Qwen35AnalysisError(
            f"estimand does not have the six registered seed blocks: {len(missing)} missing, "
            f"{len(extra)} extra"
        )
    values = tuple(float(values_by_seed[seed]) for seed in SEEDS)
    if not all(math.isfinite(value) for value in values):
        raise Qwen35AnalysisError("estimand contains a non-finite seed-block value")
    lower, upper = _bootstrap_interval(values)
    return {
        "per_seed": [{"seed": seed, "value": value} for seed, value in zip(SEEDS, values, strict=True)],
        "mean": _mean(values),
        "bootstrap_95_percentile_interval": {"lower": lower, "upper": upper},
        "descriptive_unadjusted_two_sided_sign_flip_p_value": _sign_flip_p_value(values),
    }


def _endpoint_index(
    endpoints: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, str, str, str, float], Mapping[str, Any]]:
    indexed: dict[tuple[int, str, str, str, float], Mapping[str, Any]] = {}
    for row in endpoints:
        key = (
            int(row["seed"]),
            str(row["model"]),
            str(row["algorithm"]),
            str(row["law"]),
            float(row["q_p"]),
        )
        if key in indexed:
            raise Qwen35AnalysisError(f"duplicate endpoint condition: {key}")
        indexed[key] = row
    expected = set(product(SEEDS, MODELS, ALGORITHMS, LAWS, Q_P_VALUES))
    if set(indexed) != expected:
        raise Qwen35AnalysisError("endpoint conditions do not form the exact 96-row factorial")
    return indexed


def _rho_y(row: Mapping[str, Any]) -> float:
    agreement = row["conflict_agreement"]
    if not isinstance(agreement, Mapping):
        raise Qwen35AnalysisError("endpoint lacks conflict agreement")
    return float(agreement["rho_y"])


def _pool(strata_values: Sequence[Mapping[int, float]]) -> dict[int, float]:
    return {seed: _mean(tuple(values[seed] for values in strata_values)) for seed in SEEDS}


def _primary(indexed: Mapping[tuple[int, str, str, str, float], Mapping[str, Any]]) -> dict[str, Any]:
    strata: list[dict[str, Any]] = []
    d_values: list[dict[int, float]] = []
    causal_values: list[dict[int, float]] = []
    for model, algorithm, law in product(MODELS, ALGORITHMS, LAWS):
        d_by_seed = {
            seed: float(indexed[(seed, model, algorithm, law, 1.0)]["primary_d_rho_p_minus_rho_y"])
            for seed in SEEDS
        }
        causal_by_seed = {
            seed: float(indexed[(seed, model, algorithm, law, 1.0)]["causal_companion_flip_p_minus_flip_y"])
            for seed in SEEDS
        }
        d_values.append(d_by_seed)
        causal_values.append(causal_by_seed)
        strata.append(
            {
                "model": model,
                "algorithm": algorithm,
                "law": law,
                "d_rho_p_minus_rho_y": _summarize_seed_blocks(d_by_seed),
                "causal_companion_flip_p_minus_flip_y": _summarize_seed_blocks(causal_by_seed),
            }
        )
    return {
        "q_p": 1.0,
        "definition": "D = final full-view conflict rho_P - rho_Y",
        "causal_companion_definition": ("final full-view Herald action-flip rate - Law action-flip rate"),
        "stratified_by_model_algorithm_law": strata,
        "pooled_equal_weight_within_seed": {
            "d_rho_p_minus_rho_y": _summarize_seed_blocks(_pool(d_values)),
            "causal_companion_flip_p_minus_flip_y": _summarize_seed_blocks(_pool(causal_values)),
        },
    }


def _contrast_report(
    *,
    indexed: Mapping[tuple[int, str, str, str, float], Mapping[str, Any]],
    name: str,
) -> dict[str, Any]:
    strata: list[dict[str, Any]] = []
    all_values: list[dict[int, float]] = []
    if name == "evidence":
        definition = "final full-view conflict rho_Y(q_P=.95) - rho_Y(q_P=1)"
        for model, algorithm, law in product(MODELS, ALGORITHMS, LAWS):
            values = {
                seed: _rho_y(indexed[(seed, model, algorithm, law, 0.95)])
                - _rho_y(indexed[(seed, model, algorithm, law, 1.0)])
                for seed in SEEDS
            }
            factors = {"model": model, "algorithm": algorithm, "law": law}
            all_values.append(values)
            strata.append({**factors, "effect": _summarize_seed_blocks(values)})
        stratification = "model_algorithm_law"
    elif name == "algorithm":
        definition = "final full-view conflict rho_Y(outcome RL) - rho_Y(SFT)"
        for model, law, q_p in product(MODELS, LAWS, Q_P_VALUES):
            values = {
                seed: _rho_y(indexed[(seed, model, "outcome_rl", law, q_p)])
                - _rho_y(indexed[(seed, model, "sft", law, q_p)])
                for seed in SEEDS
            }
            factors = {"model": model, "law": law, "q_p": q_p}
            all_values.append(values)
            strata.append({**factors, "effect": _summarize_seed_blocks(values)})
        stratification = "model_law_q_p"
    elif name == "scale":
        definition = "final full-view conflict rho_Y(2B) - rho_Y(.8B)"
        for algorithm, law, q_p in product(ALGORITHMS, LAWS, Q_P_VALUES):
            values = {
                seed: _rho_y(indexed[(seed, MODELS[1], algorithm, law, q_p)])
                - _rho_y(indexed[(seed, MODELS[0], algorithm, law, q_p)])
                for seed in SEEDS
            }
            factors = {"algorithm": algorithm, "law": law, "q_p": q_p}
            all_values.append(values)
            strata.append({**factors, "effect": _summarize_seed_blocks(values)})
        stratification = "algorithm_law_q_p"
    else:  # pragma: no cover - private caller supplies a literal
        raise Qwen35AnalysisError(f"unknown contrast: {name}")
    return {
        "definition": definition,
        f"stratified_by_{stratification}": strata,
        "pooled_equal_weight_within_seed": _summarize_seed_blocks(_pool(all_values)),
    }


def analyze_panel(
    panel: AnalysisPanel,
    config: Mapping[str, Any],
    *,
    config_sha256: str | None = None,
) -> dict[str, Any]:
    """Compute the registered endpoints from a fully loaded main panel."""

    expected = _expected_plan(config)
    endpoints = _extract_endpoints(panel, expected)
    indexed = _endpoint_index(endpoints)
    audit = dict(panel.audit)
    return {
        "schema": "goalzendo.qwen35_known_law_analysis",
        "schema_version": 1,
        "panel": {
            "completed_only": True,
            "expected_run_count": 96,
            "observed_run_count": len(endpoints),
            "seed_blocks": list(SEEDS),
            "expected_config_sha256": config_sha256,
            "implementation_fingerprint": audit.get("implementation_fingerprint"),
        },
        "inference": {
            "unit": "paired training-seed block",
            "seed_block_count": len(SEEDS),
            "bootstrap": {
                "method": "deterministic exhaustive seed-block bootstrap with replacement",
                "resamples": BOOTSTRAP_RESAMPLES,
                "interval": "two-sided 95% linear percentile",
            },
            "p_values": {
                "method": "exact two-sided paired sign-flip over six seed blocks",
                "assignments": SIGN_FLIP_ASSIGNMENTS,
                "status": "descriptive_unadjusted",
                "binary_claims": False,
            },
            "pooling": "equal-weight stratum mean computed within each seed before inference",
        },
        "seed_endpoints": endpoints,
        "primary": _primary(indexed),
        "secondary_contrasts": {
            "evidence": _contrast_report(indexed=indexed, name="evidence"),
            "algorithm": _contrast_report(indexed=indexed, name="algorithm"),
            "scale": _contrast_report(indexed=indexed, name="scale"),
        },
    }


def analyze_main(artifacts: Path) -> dict[str, Any]:
    config = load_config(MAIN_CONFIG)
    _expected_plan(config)
    panel = _load_exact_panel(artifacts, config)
    config_sha256 = hashlib.sha256(MAIN_CONFIG.read_bytes()).hexdigest()
    return analyze_panel(panel, config, config_sha256=config_sha256)


def canonical_json(value: Mapping[str, Any]) -> str:
    """Return one deterministic JSON document with no non-JSON float spellings."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze the exact completed 96-run Qwen3.5 known-Law main panel"
    )
    parser.add_argument("artifacts", type=Path, help="root containing the main run artifacts")
    args = parser.parse_args(argv)
    print(canonical_json(analyze_main(args.artifacts)), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
