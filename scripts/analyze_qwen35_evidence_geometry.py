#!/usr/bin/env python3
"""Analyze the frozen 36-run Qwen3.5 evidence-geometry panel."""

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

import pandas as pd

from goalzendo.analysis import AnalysisPanel, load_analysis_panel
from goalzendo.artifacts import discover_runs
from goalzendo.config import canonical_config, get_path, load_config
from goalzendo.runner import RunSpec, build_plan

ROOT = Path(__file__).resolve().parents[1]
MAIN_CONFIG = ROOT / "configs" / "goalzendo" / "qwen35_evidence_geometry.yaml"

SEEDS = (22011, 22013, 22017, 22019, 22023, 22031)
MODEL_REVISIONS = {
    "Qwen/Qwen3.5-0.8B": "2fc06364715b967f1860aea9cf38778875588b17",
    "Qwen/Qwen3.5-2B": "15852e8c16360a2fea060d615a32b45270f8a8fc",
}
MODELS = tuple(MODEL_REVISIONS)
GEOMETRIES = (
    (0.005, "diverse"),
    (0.040, "diverse"),
    (0.040, "concentrated"),
)
EVAL_STEPS = (0, 1, 4, 16, 64, 256, 512, 1000)
PROMPT_VIEWS = ("full", "audit_law_matched")
CAUSAL_TARGETS = ("y", "p", "q", "d")
FINAL_STEP = 1000
DIAGNOSTIC_FACTORIAL_PER_CELL = 16
FINAL_FACTORIAL_PER_CELL = 64
DIAGNOSTIC_CAUSAL_PER_CELL = 8
FINAL_CAUSAL_PER_CELL = 16
FACTORIAL_CELL_COUNT = 8
CONFLICT_CELL_COUNT = 6
IID_VALIDATION_COUNT = 1000
BOOTSTRAP_RESAMPLES = len(SEEDS) ** len(SEEDS)
SIGN_FLIP_ASSIGNMENTS = 2 ** len(SEEDS)
REGISTERED_CONFIG_IDENTITY_SHA256 = "5650ac4efb96557671ac87d69a1c93750c70c8ed0460f55dc205ad6ae909a456"


class Qwen35EvidenceGeometryAnalysisError(ValueError):
    """Raised when the frozen panel or a registered estimand is not identified."""


def _condition(
    spec_or_config: RunSpec | Mapping[str, Any],
) -> tuple[int, str, str, float, str]:
    if isinstance(spec_or_config, RunSpec):
        seed = spec_or_config.seed
        config = spec_or_config.config
    else:
        config = spec_or_config
        seed = int(get_path(config, "seed"))
    return (
        int(seed),
        str(get_path(config, "model.name")),
        str(get_path(config, "model.revision")),
        float(get_path(config, "data.joint_error_rate")),
        str(get_path(config, "data.conflict_diversity")),
    )


def _validate_resolved_settings(spec: RunSpec) -> list[str]:
    config = spec.config
    errors: list[str] = []

    def require(path: str, expected: Any) -> None:
        observed = get_path(config, path)
        if observed != expected:
            errors.append(f"{path}={observed!r}, expected {expected!r}")

    require("run.launch_guard", None)
    require("data.n_train", 10000)
    require("data.n_validation", 1000)
    require("data.rule_family", "parity")
    require("data.sage_rule_family", "parity")
    require("data.q_p", 0.95)
    require("data.q_q", 0.90)
    require("data.error_geometry", "specified")
    require("data.training_view", "full")
    require("data.counterbalance", True)
    require("update.method", "full")
    require("train.algorithm", "outcome_rl")
    require("train.steps", FINAL_STEP)
    require("train.learning_rate", 0.000003)
    require("train.entropy_coefficient", 0.01)
    require("train.eval_steps", list(EVAL_STEPS))
    require("evaluation.prompt_views", list(PROMPT_VIEWS))
    require("evaluation.causal_prompt_views", ["full"])
    require("evaluation.mirror_pairs", True)
    require("evaluation.save_predictions", True)
    model = str(get_path(config, "model.name"))
    if model not in MODEL_REVISIONS:
        errors.append(f"model.name={model!r} is not registered")
    elif str(get_path(config, "model.revision")) != MODEL_REVISIONS[model]:
        errors.append(f"model.revision for {model} differs from the pinned revision")
    diversity = str(get_path(config, "data.conflict_diversity"))
    concentration = get_path(config, "data.concentrated_unique_conflicts_per_tuple", None)
    if diversity == "concentrated" and concentration != 16:
        errors.append("the concentrated cell must retain 16 unique conflicts per tuple")
    if diversity == "diverse" and concentration not in (None, 16):
        errors.append("the inactive concentration setting differs from its registered value")
    return errors


def _expected_plan(config: Mapping[str, Any]) -> dict[str, RunSpec]:
    config_identity = hashlib.sha256(
        json.dumps(
            canonical_config(config),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if config_identity != REGISTERED_CONFIG_IDENTITY_SHA256:
        raise Qwen35EvidenceGeometryAnalysisError(
            "invalid expected evidence-geometry config: canonical identity differs from registration"
        )
    try:
        plan = build_plan(config)
    except (TypeError, ValueError) as exc:
        raise Qwen35EvidenceGeometryAnalysisError(
            f"invalid expected evidence-geometry config: {exc}"
        ) from exc
    expected_conditions = {
        (seed, model, revision, joint_error_rate, diversity)
        for seed, model, (joint_error_rate, diversity) in product(SEEDS, MODELS, GEOMETRIES)
        for revision in (MODEL_REVISIONS[model],)
    }
    observed_conditions = {_condition(spec) for spec in plan}
    errors: list[str] = []
    if str(get_path(config, "experiment.id")) != "qwen35_evidence_geometry":
        errors.append("experiment.id is not qwen35_evidence_geometry")
    if str(get_path(config, "experiment.status")) != "prospective":
        errors.append("experiment.status is not prospective")
    if tuple(int(seed) for seed in get_path(config, "run.seeds")) != SEEDS:
        errors.append("run.seeds differ from the six registered seed blocks")
    if len(plan) != 36:
        errors.append(f"plan has {len(plan)} rows rather than 36")
    if len({spec.plan_key for spec in plan}) != len(plan):
        errors.append("plan keys are not unique")
    if len({spec.cell_id for spec in plan}) != 6:
        errors.append("plan does not contain exactly six model-by-geometry cells")
    if observed_conditions != expected_conditions:
        errors.append("model/revision/geometry/seed cells differ from the frozen 36-row design")
    if Counter(spec.seed for spec in plan) != Counter({seed: 6 for seed in SEEDS}):
        errors.append("the six paired seed blocks are incomplete")
    for spec in plan:
        errors.extend(f"{spec.plan_key}: {message}" for message in _validate_resolved_settings(spec))
    if errors:
        raise Qwen35EvidenceGeometryAnalysisError(
            "invalid expected evidence-geometry config: " + "; ".join(errors)
        )
    return {spec.plan_key: spec for spec in plan}


def _load_exact_panel(artifacts: Path, config: Mapping[str, Any]) -> AnalysisPanel:
    all_runs = discover_runs(artifacts, completed_only=False)
    completed_runs = discover_runs(artifacts, completed_only=True)
    incomplete = sorted(set(all_runs) - set(completed_runs))
    if incomplete:
        raise Qwen35EvidenceGeometryAnalysisError(
            f"artifact root contains {len(incomplete)} incomplete run director"
            f"{'y' if len(incomplete) == 1 else 'ies'}"
        )
    return load_analysis_panel(
        artifacts,
        expected_config=config,
        require_complete_metrics=True,
        required_prompt_views=PROMPT_VIEWS,
        allow_incomplete_runs=False,
    )


def _validate_run_rows(panel: AnalysisPanel, expected: Mapping[str, RunSpec]) -> None:
    rows = panel.runs.to_dict(orient="records")
    run_ids = [str(row["run_id"]) for row in rows]
    plan_keys = [str(row["plan_key"]) for row in rows]
    if len(rows) != 36:
        raise Qwen35EvidenceGeometryAnalysisError(f"completed panel has {len(rows)} rows rather than 36")
    if len(set(run_ids)) != len(run_ids):
        raise Qwen35EvidenceGeometryAnalysisError("completed panel contains duplicate run IDs")
    if len(set(plan_keys)) != len(plan_keys):
        raise Qwen35EvidenceGeometryAnalysisError("completed panel contains duplicate plan rows")
    missing = sorted(set(expected) - set(plan_keys))
    unexpected = sorted(set(plan_keys) - set(expected))
    if missing or unexpected:
        raise Qwen35EvidenceGeometryAnalysisError(
            f"completed plan rows differ from registration: {len(missing)} missing, {len(unexpected)} extra"
        )
    for row in rows:
        plan_key = str(row["plan_key"])
        spec = expected[plan_key]
        config = row.get("config")
        if not isinstance(config, Mapping):
            raise Qwen35EvidenceGeometryAnalysisError(f"{plan_key} lacks a resolved config")
        if _condition(config) != _condition(spec):
            raise Qwen35EvidenceGeometryAnalysisError(
                f"{plan_key} has a condition inconsistent with its registered plan row"
            )
        if int(row["seed"]) != spec.seed or int(get_path(config, "seed")) != spec.seed:
            raise Qwen35EvidenceGeometryAnalysisError(
                f"{plan_key} has a seed inconsistent with its registered plan row"
            )
        if str(row["cell_id"]) != spec.cell_id:
            raise Qwen35EvidenceGeometryAnalysisError(
                f"{plan_key} has a cell identity inconsistent with its registered plan row"
            )


def _finite_unit_interval(value: Any, label: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise Qwen35EvidenceGeometryAnalysisError(f"{label} must be finite and lie in [0, 1]")
    return numeric


def _exact_positive_count(value: Any, expected: int, label: str) -> int:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 1 or not numeric.is_integer() or int(numeric) != expected:
        raise Qwen35EvidenceGeometryAnalysisError(
            f"{label} must equal the registered count {expected}; observed {value!r}"
        )
    return int(numeric)


def _preferred_split(step: int) -> str:
    return "final_factorial" if step == FINAL_STEP else "diagnostic_factorial"


def _single_behavior_row(
    panel: AnalysisPanel,
    *,
    run_id: str,
    step: int,
    prompt_view: str,
) -> pd.Series[Any]:
    selected = panel.trajectory[
        (panel.trajectory["run_id"] == run_id)
        & (panel.trajectory["step"] == step)
        & (panel.trajectory["prompt_view"] == prompt_view)
        & (panel.trajectory["split"] == _preferred_split(step))
        & (panel.trajectory["panel"] == "conflict")
    ]
    if len(selected) != 1:
        raise Qwen35EvidenceGeometryAnalysisError(
            f"{run_id} has {len(selected)} step-{step} {prompt_view} conflict rows; expected one"
        )
    return selected.iloc[0]


def _causal_flips(panel: AnalysisPanel, *, run_id: str, step: int) -> tuple[dict[str, float], int]:
    selected = panel.causal_effects[
        (panel.causal_effects["run_id"] == run_id)
        & (panel.causal_effects["step"] == step)
        & (panel.causal_effects["prompt_view"] == "full")
        & (panel.causal_effects["split"] == _preferred_split(step))
        & panel.causal_effects["target"].isin(CAUSAL_TARGETS)
    ]
    target_counts = Counter(str(value) for value in selected["target"])
    if target_counts != Counter({target: 1 for target in CAUSAL_TARGETS}):
        raise Qwen35EvidenceGeometryAnalysisError(
            f"{run_id} step {step} does not have one full-view causal row for every target"
        )
    expected_pairs = FACTORIAL_CELL_COUNT * (
        FINAL_CAUSAL_PER_CELL if step == FINAL_STEP else DIAGNOSTIC_CAUSAL_PER_CELL
    )
    pair_counts = {
        _exact_positive_count(
            row["n_pairs"],
            expected_pairs,
            f"{run_id} step {step} causal {row['target']} n_pairs",
        )
        for row in selected.to_dict(orient="records")
    }
    if pair_counts != {expected_pairs}:
        raise Qwen35EvidenceGeometryAnalysisError(f"{run_id} step {step} has inconsistent causal pair counts")
    flips = {
        f"flip_{row['target']}": _finite_unit_interval(
            row["action_flip_rate"], f"{run_id} step {step} flip_{row['target']}"
        )
        for row in selected.to_dict(orient="records")
    }
    return flips, expected_pairs


def _conflict_sample_count(
    panel: AnalysisPanel,
    *,
    run_id: str,
    step: int,
    prompt_view: str,
) -> int:
    selected = panel.raw_metrics[
        (panel.raw_metrics["run_id"] == run_id)
        & (panel.raw_metrics["step"] == step)
        & (panel.raw_metrics["split"] == _preferred_split(step))
        & (panel.raw_metrics["prompt_view"] == prompt_view)
        & (panel.raw_metrics["kind"] == "behavioral_agreement")
        & (panel.raw_metrics["panel"] == "conflict")
    ]
    if len(selected) != 1:
        raise Qwen35EvidenceGeometryAnalysisError(
            f"{run_id} step {step} has {len(selected)} {prompt_view} conflict summaries; expected one"
        )
    per_cell = FINAL_FACTORIAL_PER_CELL if step == FINAL_STEP else DIAGNOSTIC_FACTORIAL_PER_CELL
    return _exact_positive_count(
        selected.iloc[0]["n"],
        CONFLICT_CELL_COUNT * per_cell,
        f"{run_id} step {step} {prompt_view} conflict n",
    )


def _iid_rho_y(panel: AnalysisPanel, *, run_id: str, step: int) -> float:
    selected = panel.raw_metrics[
        (panel.raw_metrics["run_id"] == run_id)
        & (panel.raw_metrics["step"] == step)
        & (panel.raw_metrics["split"] == "iid_validation")
        & (panel.raw_metrics["prompt_view"] == "full")
        & (panel.raw_metrics["kind"] == "behavioral_agreement")
        & (panel.raw_metrics["panel"] == "all")
    ]
    if len(selected) != 1:
        raise Qwen35EvidenceGeometryAnalysisError(
            f"{run_id} step {step} has {len(selected)} full-view IID rows; expected one"
        )
    _exact_positive_count(
        selected.iloc[0]["n"],
        IID_VALIDATION_COUNT,
        f"{run_id} step {step} full-view IID n",
    )
    values: list[float] = []
    for row in selected.to_dict(orient="records"):
        value = row.get("rho_y", row.get("agreement_y", row.get("accuracy")))
        if value is not None and not pd.isna(value):
            values.append(_finite_unit_interval(value, f"{run_id} step {step} IID rho_y"))
    unique = sorted(set(values))
    if len(unique) != 1:
        raise Qwen35EvidenceGeometryAnalysisError(
            f"{run_id} step {step} needs one unambiguous full-view IID rho_y; observed {unique}"
        )
    return unique[0]


def _final_truth_table(panel: AnalysisPanel, *, run_id: str) -> list[dict[str, Any]]:
    selected = panel.factorial[
        (panel.factorial["run_id"] == run_id)
        & (panel.factorial["step"] == FINAL_STEP)
        & (panel.factorial["prompt_view"] == "full")
        & (panel.factorial["split"] == "final_factorial")
    ]
    expected_choices = set(product((0, 1), repeat=3))
    counts: Counter[tuple[int, int, int]] = Counter()
    rows: list[dict[str, Any]] = []
    for record in selected.to_dict(orient="records"):
        choices = tuple(int(record[key]) for key in ("choice_y", "choice_p", "choice_q"))
        if choices not in expected_choices:
            raise Qwen35EvidenceGeometryAnalysisError(
                f"{run_id} final full-view truth table contains a non-binary candidate tuple"
            )
        counts[choices] += 1
        raw_n = record["n"]
        exact_n = _exact_positive_count(
            raw_n,
            FINAL_FACTORIAL_PER_CELL,
            f"{run_id} final full-view truth table {choices} n",
        )
        rows.append(
            {
                "choice_y": choices[0],
                "choice_p": choices[1],
                "choice_q": choices[2],
                "action_b_rate": _finite_unit_interval(
                    record["action_b_rate"],
                    f"{run_id} final truth table {choices} action_b_rate",
                ),
                "n": exact_n,
            }
        )
    if counts != Counter({choices: 1 for choices in expected_choices}):
        raise Qwen35EvidenceGeometryAnalysisError(
            f"{run_id} needs exactly one final full-view row for each of the eight Y/P/Q tuples"
        )
    return sorted(rows, key=lambda row: (row["choice_y"], row["choice_p"], row["choice_q"]))


def _validate_metric_axes(panel: AnalysisPanel) -> None:
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
        "n_pairs",
    }
    required_iid = {"run_id", "step", "prompt_view", "split", "kind", "panel", "n"}
    required_factorial = {
        "run_id",
        "step",
        "prompt_view",
        "split",
        "choice_y",
        "choice_p",
        "choice_q",
        "action_b_rate",
        "n",
    }
    for label, frame, required in (
        ("trajectory", panel.trajectory, required_behavior),
        ("causal", panel.causal_effects, required_causal),
        ("raw metric", panel.raw_metrics, required_iid),
        ("factorial", panel.factorial, required_factorial),
    ):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise Qwen35EvidenceGeometryAnalysisError(f"{label} table lacks columns: {missing}")

    for run_id in panel.runs["run_id"].astype(str):
        behavior = panel.trajectory[
            (panel.trajectory["run_id"] == run_id)
            & (panel.trajectory["panel"] == "conflict")
            & panel.trajectory["split"].isin(("diagnostic_factorial", "final_factorial"))
        ]
        observed_behavior_axes = {
            (int(row["step"]), str(row["prompt_view"]), str(row["split"]))
            for row in behavior.to_dict(orient="records")
        }
        expected_behavior_axes = {
            (step, view, _preferred_split(step)) for step, view in product(EVAL_STEPS, PROMPT_VIEWS)
        }
        if observed_behavior_axes != expected_behavior_axes or len(behavior) != len(expected_behavior_axes):
            raise Qwen35EvidenceGeometryAnalysisError(
                f"{run_id} does not have exactly the registered conflict-view trajectory"
            )

        causal = panel.causal_effects[panel.causal_effects["run_id"] == run_id]
        observed_causal_axes = Counter(
            (
                int(row["step"]),
                str(row["prompt_view"]),
                str(row["split"]),
                str(row["target"]),
            )
            for row in causal.to_dict(orient="records")
        )
        expected_causal_axes = Counter(
            (step, "full", _preferred_split(step), target)
            for step, target in product(EVAL_STEPS, CAUSAL_TARGETS)
        )
        if observed_causal_axes != expected_causal_axes:
            raise Qwen35EvidenceGeometryAnalysisError(
                f"{run_id} does not have exactly the registered full-view causal trajectory"
            )

        iid_full = panel.raw_metrics[
            (panel.raw_metrics["run_id"] == run_id)
            & (panel.raw_metrics["prompt_view"] == "full")
            & (panel.raw_metrics["split"] == "iid_validation")
            & (panel.raw_metrics["kind"] == "behavioral_agreement")
            & (panel.raw_metrics["panel"] == "all")
        ]
        observed_iid_steps = Counter(int(value) for value in iid_full["step"])
        if observed_iid_steps != Counter({step: 1 for step in EVAL_STEPS}):
            raise Qwen35EvidenceGeometryAnalysisError(
                f"{run_id} does not have exactly the registered full-view IID trajectory"
            )


def _extract_trajectories(panel: AnalysisPanel, expected: Mapping[str, RunSpec]) -> list[dict[str, Any]]:
    _validate_run_rows(panel, expected)
    _validate_metric_axes(panel)
    trajectories: list[dict[str, Any]] = []
    for run in panel.runs.to_dict(orient="records"):
        run_id = str(run["run_id"])
        plan_key = str(run["plan_key"])
        seed, model, revision, joint_error_rate, diversity = _condition(expected[plan_key])
        checkpoints: list[dict[str, Any]] = []
        for step in EVAL_STEPS:
            full = _single_behavior_row(panel, run_id=run_id, step=step, prompt_view="full")
            matched = _single_behavior_row(
                panel,
                run_id=run_id,
                step=step,
                prompt_view="audit_law_matched",
            )
            conflict = {
                "rho_y": _finite_unit_interval(full["rho_y"], f"{run_id} step {step} rho_y"),
                "rho_p": _finite_unit_interval(full["rho_p"], f"{run_id} step {step} rho_p"),
                "rho_q": _finite_unit_interval(full["rho_q"], f"{run_id} step {step} rho_q"),
            }
            flips, causal_n_pairs = _causal_flips(panel, run_id=run_id, step=step)
            full_conflict_n = _conflict_sample_count(
                panel,
                run_id=run_id,
                step=step,
                prompt_view="full",
            )
            matched_conflict_n = _conflict_sample_count(
                panel,
                run_id=run_id,
                step=step,
                prompt_view="audit_law_matched",
            )
            checkpoints.append(
                {
                    "step": step,
                    "full_conflict_agreement": conflict,
                    "full_conflict_n": full_conflict_n,
                    "full_causal_action_flip": flips,
                    "full_causal_n_pairs_per_target": causal_n_pairs,
                    "iid_full_rho_y": _iid_rho_y(panel, run_id=run_id, step=step),
                    "iid_full_n": IID_VALIDATION_COUNT,
                    "audit_law_matched_conflict_rho_y": _finite_unit_interval(
                        matched["rho_y"],
                        f"{run_id} step {step} matched-audit rho_y",
                    ),
                    "audit_law_matched_conflict_n": matched_conflict_n,
                    "causal_companion_c": flips["flip_y"] - 0.5 * (flips["flip_p"] + flips["flip_q"]),
                }
            )
        trajectories.append(
            {
                "seed": seed,
                "model": model,
                "model_revision": revision,
                "joint_error_rate": joint_error_rate,
                "conflict_diversity": diversity,
                "run_id": run_id,
                "plan_key": plan_key,
                "checkpoints": checkpoints,
                "final_full_truth_table": _final_truth_table(panel, run_id=run_id),
            }
        )
    return sorted(
        trajectories,
        key=lambda row: (
            int(row["seed"]),
            str(row["model"]),
            float(row["joint_error_rate"]),
            str(row["conflict_diversity"]),
        ),
    )


def _extract_endpoints(trajectories: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    endpoints: list[dict[str, Any]] = []
    for trajectory in trajectories:
        checkpoints = trajectory["checkpoints"]
        if not isinstance(checkpoints, Sequence) or len(checkpoints) != len(EVAL_STEPS):
            raise Qwen35EvidenceGeometryAnalysisError("trajectory is not complete")
        final = checkpoints[-1]
        if not isinstance(final, Mapping) or int(final["step"]) != FINAL_STEP:
            raise Qwen35EvidenceGeometryAnalysisError("trajectory lacks its final registered step")
        endpoints.append(
            {
                key: trajectory[key]
                for key in (
                    "seed",
                    "model",
                    "model_revision",
                    "joint_error_rate",
                    "conflict_diversity",
                    "run_id",
                    "plan_key",
                )
            }
            | {
                "full_conflict_agreement": dict(final["full_conflict_agreement"]),
                "full_conflict_n": int(final["full_conflict_n"]),
                "full_causal_action_flip": dict(final["full_causal_action_flip"]),
                "full_causal_n_pairs_per_target": int(final["full_causal_n_pairs_per_target"]),
                "iid_full_rho_y": float(final["iid_full_rho_y"]),
                "iid_full_n": int(final["iid_full_n"]),
                "audit_law_matched_conflict_rho_y": float(final["audit_law_matched_conflict_rho_y"]),
                "audit_law_matched_conflict_n": int(final["audit_law_matched_conflict_n"]),
                "causal_companion_c": float(final["causal_companion_c"]),
                "final_full_truth_table": [dict(row) for row in trajectory["final_full_truth_table"]],
            }
        )
    return endpoints


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
        raise Qwen35EvidenceGeometryAnalysisError("bootstrap requires exactly six seed-block values")
    estimates = sorted(
        _mean(tuple(values[index] for index in indices))
        for indices in product(range(len(SEEDS)), repeat=len(SEEDS))
    )
    if len(estimates) != BOOTSTRAP_RESAMPLES:
        raise AssertionError("exhaustive bootstrap enumeration is incomplete")
    return _linear_percentile(estimates, 0.025), _linear_percentile(estimates, 0.975)


def _sign_flip_p_value(values: Sequence[float]) -> float:
    if len(values) != len(SEEDS):
        raise Qwen35EvidenceGeometryAnalysisError("sign-flip test requires exactly six seed-block values")
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
        raise Qwen35EvidenceGeometryAnalysisError(
            "estimand does not have the six registered seed blocks: "
            f"{len(missing)} missing, {len(extra)} extra"
        )
    values = tuple(float(values_by_seed[seed]) for seed in SEEDS)
    if not all(math.isfinite(value) for value in values):
        raise Qwen35EvidenceGeometryAnalysisError("estimand contains a non-finite seed-block value")
    lower, upper = _bootstrap_interval(values)
    return {
        "per_seed": [{"seed": seed, "value": value} for seed, value in zip(SEEDS, values, strict=True)],
        "mean": _mean(values),
        "bootstrap_95_percentile_interval": {"lower": lower, "upper": upper},
        "descriptive_unadjusted_two_sided_sign_flip_p_value": _sign_flip_p_value(values),
    }


def _endpoint_index(
    endpoints: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, str, float, str], Mapping[str, Any]]:
    indexed: dict[tuple[int, str, float, str], Mapping[str, Any]] = {}
    for row in endpoints:
        key = (
            int(row["seed"]),
            str(row["model"]),
            float(row["joint_error_rate"]),
            str(row["conflict_diversity"]),
        )
        if key in indexed:
            raise Qwen35EvidenceGeometryAnalysisError(f"duplicate endpoint condition: {key}")
        indexed[key] = row
    expected = {
        (seed, model, joint_error_rate, diversity)
        for seed, model, (joint_error_rate, diversity) in product(SEEDS, MODELS, GEOMETRIES)
    }
    if set(indexed) != expected:
        raise Qwen35EvidenceGeometryAnalysisError(
            "endpoint conditions do not form the exact 36-row factorial"
        )
    return indexed


def _rho_y(row: Mapping[str, Any]) -> float:
    agreement = row["full_conflict_agreement"]
    if not isinstance(agreement, Mapping):
        raise Qwen35EvidenceGeometryAnalysisError("endpoint lacks full conflict agreement")
    return float(agreement["rho_y"])


def _causal_c(row: Mapping[str, Any]) -> float:
    return float(row["causal_companion_c"])


def _raw_endpoint(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "run_id": str(row["run_id"]),
        "rho_y": _rho_y(row),
        "rho_p": float(row["full_conflict_agreement"]["rho_p"]),
        "rho_q": float(row["full_conflict_agreement"]["rho_q"]),
        **{key: float(value) for key, value in row["full_causal_action_flip"].items()},
        "causal_companion_c": _causal_c(row),
        "iid_full_rho_y": float(row["iid_full_rho_y"]),
        "iid_full_n": int(row["iid_full_n"]),
        "full_conflict_n": int(row["full_conflict_n"]),
        "full_causal_n_pairs_per_target": int(row["full_causal_n_pairs_per_target"]),
        "audit_law_matched_conflict_rho_y": float(row["audit_law_matched_conflict_rho_y"]),
        "audit_law_matched_conflict_n": int(row["audit_law_matched_conflict_n"]),
        "full_truth_table": [dict(value) for value in row["final_full_truth_table"]],
    }


def _contrast_report(
    *,
    indexed: Mapping[tuple[int, str, float, str], Mapping[str, Any]],
    name: str,
) -> dict[str, Any]:
    if name == "primary":
        comparison = (0.040, "diverse")
        reference = (0.005, "diverse")
        label = "joint-rich diverse minus product-overlap diverse"
    elif name == "secondary":
        comparison = (0.040, "diverse")
        reference = (0.040, "concentrated")
        label = "joint-rich diverse minus joint-rich concentrated"
    else:  # pragma: no cover - private caller supplies a literal
        raise Qwen35EvidenceGeometryAnalysisError(f"unknown contrast: {name}")

    rho_by_model: list[dict[int, float]] = []
    causal_by_model: list[dict[int, float]] = []
    model_reports: list[dict[str, Any]] = []
    for model in MODELS:
        rho_values: dict[int, float] = {}
        causal_values: dict[int, float] = {}
        pairs: list[dict[str, Any]] = []
        for seed in SEEDS:
            comparison_row = indexed[(seed, model, *comparison)]
            reference_row = indexed[(seed, model, *reference)]
            rho_effect = _rho_y(comparison_row) - _rho_y(reference_row)
            causal_effect = _causal_c(comparison_row) - _causal_c(reference_row)
            rho_values[seed] = rho_effect
            causal_values[seed] = causal_effect
            pairs.append(
                {
                    "seed": seed,
                    "comparison": _raw_endpoint(comparison_row),
                    "reference": _raw_endpoint(reference_row),
                    "rho_y_effect": rho_effect,
                    "causal_companion_c_effect": causal_effect,
                }
            )
        rho_by_model.append(rho_values)
        causal_by_model.append(causal_values)
        model_reports.append(
            {
                "model": model,
                "raw_paired_endpoints": pairs,
                "rho_y_effect": _summarize_seed_blocks(rho_values),
                "causal_companion_c_effect": _summarize_seed_blocks(causal_values),
            }
        )

    pooled_rho = {seed: _mean(tuple(values[seed] for values in rho_by_model)) for seed in SEEDS}
    pooled_causal = {seed: _mean(tuple(values[seed] for values in causal_by_model)) for seed in SEEDS}
    return {
        "label": label,
        "comparison_condition": {
            "joint_error_rate": comparison[0],
            "conflict_diversity": comparison[1],
        },
        "reference_condition": {
            "joint_error_rate": reference[0],
            "conflict_diversity": reference[1],
        },
        "rho_y_definition": "comparison minus reference final full-view conflict rho_Y",
        "causal_companion_definition": (
            "C = flip_Y - 0.5 * (flip_P + flip_Q); comparison minus reference final C"
        ),
        "per_model": model_reports,
        "pooled_equal_weight_across_models_within_seed": {
            "rho_y_effect": _summarize_seed_blocks(pooled_rho),
            "causal_companion_c_effect": _summarize_seed_blocks(pooled_causal),
        },
    }


def analyze_panel(
    panel: AnalysisPanel,
    config: Mapping[str, Any],
    *,
    config_sha256: str | None = None,
) -> dict[str, Any]:
    """Compute the registered evidence-geometry endpoints from a complete panel."""

    expected = _expected_plan(config)
    trajectories = _extract_trajectories(panel, expected)
    endpoints = _extract_endpoints(trajectories)
    indexed = _endpoint_index(endpoints)
    audit = dict(panel.audit)
    return {
        "schema": "goalzendo.qwen35_evidence_geometry_analysis",
        "schema_version": 1,
        "panel": {
            "completed_only": True,
            "expected_run_count": 36,
            "observed_run_count": len(endpoints),
            "seed_blocks": list(SEEDS),
            "models": list(MODELS),
            "registered_eval_steps": list(EVAL_STEPS),
            "expected_config_sha256": config_sha256,
            "registered_config_identity_sha256": REGISTERED_CONFIG_IDENTITY_SHA256,
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
            "pooling": "equal-weight average across the two models within each seed before inference",
        },
        "interpretation_guard": {
            "claim_scope": "behavioral control, not internal objectives or representations",
            "raw_endpoint_policy": (
                "retain rho_Y, rho_P, rho_Q, flip_Y, flip_P, flip_Q, and flip_D so a change "
                "between proxy controllers is not mislabeled as Law acquisition; retain the "
                "eight-cell Y/P/Q truth table so conditional policies are not hidden by marginals"
            ),
        },
        "seed_endpoints": endpoints,
        "seed_trajectories": trajectories,
        "primary_contrast": _contrast_report(indexed=indexed, name="primary"),
        "registered_secondary_contrast": _contrast_report(indexed=indexed, name="secondary"),
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
        description="Analyze the exact completed 36-run Qwen3.5 evidence-geometry panel"
    )
    parser.add_argument("artifacts", type=Path, help="root containing the evidence-geometry run artifacts")
    args = parser.parse_args(argv)
    print(canonical_json(analyze_main(args.artifacts)), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
