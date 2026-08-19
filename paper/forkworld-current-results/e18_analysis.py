#!/usr/bin/env python3
"""Strict audit, analysis, and visualization for E18 evidence ordering.

E18 is adaptive and post-hoc.  The independent unit is a training seed; schedule
arms, checkpoints, probes, and factorial rows are paired repeated measurements.
``pilot`` accepts exactly the separate three-seed engineering grid and emits no
confirmatory estimate.  ``full`` accepts exactly the frozen 20 x 3 grid and is
the only mode that emits scientific contrasts or a figure.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias, cast

os.environ.setdefault("MPLCONFIGDIR", "/tmp/forkworld-e18-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/forkworld-e18-xdg")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch
import yaml  # type: ignore[import-untyped]

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forkworld.artifacts import implementation_provenance  # noqa: E402
from forkworld.competing import make_competing_factorial_dataset  # noqa: E402
from forkworld.config import expand_sweep, load_config  # noqa: E402
from forkworld.evidence_order import (  # noqa: E402
    audit_atomic_evidence_plan,
    audit_evidence_strata,
    make_atomic_evidence_plan,
    make_identical_evidence_dataset,
)
from forkworld.handoff import (  # noqa: E402
    pure_control,
    semantic_batch_digest,
    stable_state_digest,
)
from forkworld.protocols import build_model  # noqa: E402

CONFIG_PATH = REPO / "configs" / "e18_identical_evidence_order.yaml"
DEFAULT_ARTIFACTS = REPO / "artifacts-e18"
DEFAULT_PILOT_ARTIFACTS = REPO / "artifacts-e18-pilot"
DERIVED = HERE / "derived"
FIGURES = HERE / "figures"

CONFIRMATORY_SEEDS = (
    269,
    271,
    277,
    281,
    283,
    293,
    307,
    311,
    313,
    317,
    331,
    337,
    347,
    349,
    353,
    359,
    367,
    373,
    379,
    383,
)
PILOT_SEEDS = (389, 397, 401)
SCHEDULES = ("b_then_d", "d_then_b", "interleave")
GOALS = ("P", "Q", "Y")
LAYERS = ("raw", "first_hidden", "final_hidden")
SECOND_BLOCK_STEPS = (
    0,
    1,
    2,
    3,
    4,
    5,
    7,
    9,
    13,
    17,
    24,
    33,
    45,
    62,
    85,
    117,
    128,
    161,
    200,
)
WASHOUT_STEPS = (
    0,
    1,
    2,
    3,
    4,
    5,
    7,
    9,
    13,
    17,
    24,
    33,
    45,
    62,
    85,
    117,
    128,
    161,
    222,
    304,
    360,
)
LATE_STEPS = (222, 304, 360)
PREFIX_STEPS = 3_240
BLOCK_STEPS = 200
DIAGNOSTICS_END = 3_640
TOTAL_STEPS = 4_000
BATCH_SIZE = 250
PROBE_TRAIN_N = 2_048
PROBE_EVAL_N = 4_096
AUC_HORIZON = 128
PURE_THRESHOLD = 0.90
PURE_MARGIN = 0.10
PRIMARY_THRESHOLD = 0.10
EQUIVALENCE_MARGIN = 0.05
TERMINAL_THRESHOLD = 0.10
MINIMUM_SIGN_COUNT = 15
LATE_TOLERANCE = 0.02
BOOTSTRAP_DRAWS = 4_000
ARTIFACT_SCHEMA_VERSION = 1
SOURCE_FINGERPRINT_SCHEMA_VERSION = 1
NON_SCIENTIFIC_RUN_FIELDS = frozenset({"output_root", "resume", "seeds"})

INK = "#17212B"
GRAY = "#667085"
LIGHT_GRAY = "#E6E9EE"
BLUE = "#2673B8"
ORANGE = "#D55E00"
TEAL = "#009E73"

MetricKey: TypeAlias = tuple[str, str, str, str]


def get(mapping: Mapping[str, Any], path: str, default: Any = None) -> Any:
    value: Any = mapping
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def load_json(path: Path) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read JSON object {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise RuntimeError(f"expected JSON object: {path}")
    return cast(Mapping[str, Any], value)


def load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as error:
        raise RuntimeError(f"cannot read YAML mapping {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise RuntimeError(f"expected YAML mapping: {path}")
    return cast(Mapping[str, Any], value)


def expect_equal(
    errors: list[str], run_id: str, actual: Any, expected: Any, label: str
) -> None:
    if actual != expected:
        errors.append(f"{run_id}: {label}={actual!r}, expected {expected!r}")


def expect_close(
    errors: list[str],
    run_id: str,
    actual: Any,
    expected: float,
    label: str,
    *,
    atol: float = 1e-12,
) -> None:
    try:
        numeric = float(actual)
    except (TypeError, ValueError):
        errors.append(f"{run_id}: {label}={actual!r} is not numeric")
        return
    if not math.isfinite(numeric) or not math.isclose(
        numeric, expected, rel_tol=0.0, abs_tol=atol
    ):
        errors.append(f"{run_id}: {label}={numeric!r}, expected {expected!r}")


def _raise_audit(name: str, errors: Sequence[str]) -> None:
    preview = "\n".join(f"  - {error}" for error in errors[:100])
    suffix = "" if len(errors) <= 100 else f"\n  ... and {len(errors) - 100} more"
    raise RuntimeError(f"{name} failed with {len(errors)} issue(s):\n{preview}{suffix}")


def _hash_is_valid(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _scientific_config(config: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        str(key): copy.deepcopy(value)
        for key, value in config.items()
        if key != "seed" and not str(key).startswith("_")
    }
    run = dict(cast(Mapping[str, Any], result.get("run", {})))
    for field in NON_SCIENTIFIC_RUN_FIELDS:
        run.pop(field, None)
    result["run"] = run
    return result


def expected_artifact_run_id(
    config: Mapping[str, Any], metadata: Mapping[str, Any]
) -> str:
    canonical = json.dumps(
        _scientific_config(config), sort_keys=True, separators=(",", ":"), default=str
    )
    implementation = get(metadata, "implementation", {})
    identity = {
        "config": canonical,
        "seed": int(get(config, "seed")),
        "artifact_schema_version": get(
            implementation, "artifact_schema_version"
        ),
        "source_fingerprint_schema_version": get(
            implementation, "source_fingerprint_schema_version"
        ),
        "implementation_fingerprint": get(
            implementation, "implementation_fingerprint"
        ),
    }
    raw = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _expected_configs() -> dict[str, Mapping[str, Any]]:
    base = load_config(CONFIG_PATH)
    cells = expand_sweep(base)
    result: dict[str, Mapping[str, Any]] = {}
    for cell in cells:
        schedule = str(get(cell, "h15.schedule"))
        expected = copy.deepcopy(dict(cell))
        run = dict(cast(Mapping[str, Any], expected["run"]))
        run["device"] = "cpu"
        expected["run"] = run
        result[schedule] = _scientific_config(expected)
    if set(result) != set(SCHEDULES):
        raise RuntimeError(f"current E18 config schedules drifted: {sorted(result)}")
    return result


@dataclass(frozen=True)
class MetricPoint:
    value: float
    n: int
    global_step: int
    stage_step: int
    examples_seen: int


@dataclass(frozen=True)
class Run:
    path: Path
    config: Mapping[str, Any]
    summary: Mapping[str, Any]
    metadata: Mapping[str, Any]
    metrics: Mapping[MetricKey, Mapping[int, MetricPoint]]
    metric_line_count: int

    @property
    def seed(self) -> int:
        return int(self.summary["seed"])

    @property
    def schedule(self) -> str:
        return str(self.summary["schedule"])

    def snapshot(self, phase: str, step: int) -> Mapping[str, Any]:
        if phase == "prefix":
            result = self.summary.get("prefix_snapshot")
        else:
            result = get(self.summary, f"{phase}_snapshots.{step}")
        if not isinstance(result, Mapping):
            raise RuntimeError(f"{self.path.name}: missing {phase} snapshot {step}")
        return cast(Mapping[str, Any], result)


def _metric_is_needed(stage: str, split: str, intervention: str, metric: str) -> bool:
    if stage in {"prefix_behavior", "diagnostic_behavior", "washout_behavior"}:
        return (
            split == "factorial_eval"
            and intervention == "none"
            and metric
            in {"rho_p", "rho_q", "rho_y_code", "target_accuracy", "m_y"}
        )
    if stage in {"prefix_causal", "diagnostic_causal", "washout_causal"}:
        return (
            split == "factorial_eval"
            and intervention in GOALS
            and metric in {"causal_score", "causal_prob_score"}
        )
    if stage in {"prefix_probe", "diagnostic_probe", "washout_probe"}:
        return (
            split == "factorial_probe"
            and intervention == "none"
            and (
                metric
                in {
                    "raw_codeword_count",
                    "truth_table_control_positive_fraction",
                    "n_train",
                    "n_heldout",
                    "n_labels",
                }
                or any(
                    metric == f"representations__{layer}__heldout_accuracy__{label}"
                    for layer in LAYERS
                    for label in (*GOALS, "truth_table_control")
                )
            )
        )
    if stage in {
        "prefix_truth_table",
        "diagnostic_truth_table",
        "washout_truth_table",
    }:
        return (
            split == "factorial_eval"
            and intervention == "none"
            and metric
            in {
                "boolean_signature_int",
                "tuple_consistency",
                "codeword_consistency",
                "nuisance_consistency",
                "sign_inversion_symmetry",
            }
        )
    if stage in {
        "prefix_optimization",
        "diagnostic_optimization",
        "washout_optimization",
    }:
        return (
            split == "train_minibatch"
            and intervention == "none"
            and metric
            in {"loss", "primary_loss", "train_batch_accuracy", "optimizer_steps"}
        )
    return False


def _load_metric_index(
    path: Path,
    *,
    run_id: str,
    seed: int,
    schedule: str,
    errors: list[str],
    blind: bool = False,
) -> tuple[dict[MetricKey, dict[int, MetricPoint]], int]:
    result: dict[MetricKey, dict[int, MetricPoint]] = defaultdict(dict)
    line_count = 0
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as error:
        raise RuntimeError(f"cannot read metrics {path}: {error}") from error
    with handle:
        for line_number, line in enumerate(handle, start=1):
            line_count += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                errors.append(f"{run_id}: malformed metric line {line_number}: {error}")
                continue
            if not isinstance(record, Mapping):
                errors.append(f"{run_id}: metric line {line_number} is not an object")
                continue
            if record.get("run_id") != run_id:
                errors.append(f"{run_id}: metric line {line_number} run-id mismatch")
            if record.get("seed") != seed:
                errors.append(f"{run_id}: metric line {line_number} seed mismatch")
            if record.get("experiment") != "h15":
                errors.append(f"{run_id}: metric line {line_number} experiment mismatch")
            stage = str(record.get("stage"))
            expected_condition = (
                "adaptive_identical_evidence_order"
                if stage == "final"
                else f"identical_evidence_order:{schedule}"
            )
            if record.get("condition") != expected_condition:
                errors.append(f"{run_id}: metric line {line_number} condition mismatch")
            if record.get("level") != "choice":
                errors.append(f"{run_id}: metric line {line_number} level mismatch")
            split = str(record.get("split"))
            intervention = str(record.get("intervention", "none"))
            metric = str(record.get("metric"))
            if not _metric_is_needed(stage, split, intervention, metric):
                continue
            try:
                local = int(record["stage_step"])
                point = MetricPoint(
                    # Engineering-pilot analysis is intentionally outcome-blind.
                    # Parsing JSON necessarily materializes each object, but the
                    # scientific value field is never dereferenced in that mode.
                    value=0.0 if blind else float(record["value"]),
                    n=int(record["n"]),
                    global_step=int(record["global_step"]),
                    stage_step=local,
                    examples_seen=int(record["examples_seen"]),
                )
            except (KeyError, TypeError, ValueError) as error:
                errors.append(
                    f"{run_id}: malformed selected metric line {line_number}: {error}"
                )
                continue
            if not math.isfinite(point.value):
                errors.append(f"{run_id}: non-finite metric line {line_number}")
            key = (stage, split, intervention, metric)
            if local in result[key]:
                errors.append(f"{run_id}: duplicate selected metric {key} at {local}")
            result[key][local] = point
    return dict(result), line_count


def _require_metric_steps(
    run: Run, errors: list[str], key: MetricKey, expected: Sequence[int]
) -> None:
    observed = set(run.metrics.get(key, {}))
    wanted = set(expected)
    if observed != wanted:
        errors.append(
            f"{run.path.name}: {key} has missing={sorted(wanted - observed)} "
            f"unexpected={sorted(observed - wanted)}"
        )


def intended_margin(snapshot: Mapping[str, Any]) -> float:
    behavior = cast(Mapping[str, Any], snapshot["behavior"])
    causal = cast(Mapping[str, Any], snapshot["causal"])
    return 0.5 * (
        float(behavior["Y"]) - max(float(behavior["P"]), float(behavior["Q"]))
        + float(causal["Y"]) - max(float(causal["P"]), float(causal["Q"]))
    )


def normalized_auc(
    snapshots: Mapping[int, Mapping[str, Any]], *, horizon: int = AUC_HORIZON
) -> float:
    steps = sorted(step for step in snapshots if 0 <= step <= horizon)
    if not steps or steps[0] != 0 or steps[-1] != horizon:
        raise RuntimeError("washout AUC lacks directly observed endpoints")
    area = 0.0
    for index in range(len(steps) - 1):
        left, right = steps[index], steps[index + 1]
        area += (right - left) * (
            intended_margin(snapshots[left]) + intended_margin(snapshots[right])
        ) / 2.0
    return area / horizon


def _pure_goal(snapshot: Mapping[str, Any]) -> str | None:
    behavior = cast(Mapping[str, float], snapshot["behavior"])
    causal = cast(Mapping[str, float], snapshot["causal"])
    winners = [
        goal
        for goal in GOALS
        if pure_control(
            behavior,
            causal,
            goal,
            threshold=PURE_THRESHOLD,
            margin=PURE_MARGIN,
        )
    ]
    return winners[0] if len(winners) == 1 else None


def _snapshot_metric_value(snapshot: Mapping[str, Any], key: MetricKey) -> float:
    stage, _split, intervention, metric = key
    if stage.endswith("_behavior"):
        field = {"rho_p": "P", "rho_q": "Q", "rho_y_code": "Y"}.get(metric)
        if field is not None:
            return float(get(snapshot, f"behavior.{field}"))
        if metric == "m_y":
            return float(snapshot["m_y"])
        return float(snapshot["target_accuracy"])
    if stage.endswith("_causal"):
        family = "causal_probability" if metric == "causal_prob_score" else "causal"
        return float(get(snapshot, f"{family}.{intervention}"))
    if stage.endswith("_probe"):
        if metric.startswith("representations__"):
            _, layer, _, label = metric.split("__", 3)
            return float(get(snapshot, f"probe_heldout_accuracy.{layer}.{label}"))
        lookup = {
            "raw_codeword_count": 64,
            "truth_table_control_positive_fraction": 0.5,
            "n_train": PROBE_TRAIN_N,
            "n_heldout": PROBE_EVAL_N,
            "n_labels": 8,
        }
        return float(lookup[metric])
    if stage.endswith("_truth_table"):
        return float(get(snapshot, f"truth_table.{metric}"))
    raise KeyError(key)


def _validate_snapshot(
    run: Run,
    snapshot: Mapping[str, Any],
    errors: list[str],
    *,
    label: str,
    local_step: int,
    global_step: int,
) -> None:
    run_id = run.path.name
    expect_equal(errors, run_id, snapshot.get("local_step"), local_step, f"{label}.local_step")
    expect_equal(errors, run_id, snapshot.get("global_step"), global_step, f"{label}.global_step")
    expect_equal(
        errors,
        run_id,
        snapshot.get("examples_seen"),
        global_step * BATCH_SIZE,
        f"{label}.examples_seen",
    )
    for family in ("behavior", "causal", "causal_probability"):
        values = snapshot.get(family)
        if not isinstance(values, Mapping) or set(values) != set(GOALS):
            errors.append(f"{run_id}: {label}.{family} is not an exact P/Q/Y mapping")
            continue
        for goal, value in values.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                errors.append(f"{run_id}: {label}.{family}.{goal} is not numeric")
                continue
            if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
                errors.append(f"{run_id}: {label}.{family}.{goal}={numeric!r} outside [0,1]")
    if isinstance(snapshot.get("behavior"), Mapping) and isinstance(
        snapshot.get("causal"), Mapping
    ):
        expect_close(errors, run_id, snapshot.get("m_y"), intended_margin(snapshot), f"{label}.m_y")
        expected_goal = _pure_goal(snapshot)
        expect_equal(errors, run_id, snapshot.get("pure_goal"), expected_goal, f"{label}.pure_goal")
        pure = snapshot.get("pure_control")
        expected_pure = {goal: goal == expected_goal for goal in GOALS}
        expect_equal(errors, run_id, pure, expected_pure, f"{label}.pure_control")
    expect_close(
        errors,
        run_id,
        snapshot.get("target_accuracy"),
        float(get(snapshot, "behavior.Y", float("nan"))),
        f"{label}.target_accuracy/Y agreement",
    )
    truth = snapshot.get("truth_table")
    if not isinstance(truth, Mapping):
        errors.append(f"{run_id}: {label}.truth_table missing")
    else:
        signature = truth.get("boolean_signature")
        if not isinstance(signature, str) or len(signature) != 8 or set(signature) - {"0", "1"}:
            errors.append(f"{run_id}: {label} malformed Boolean signature {signature!r}")
        else:
            expect_equal(
                errors,
                run_id,
                truth.get("boolean_signature_int"),
                int(signature, 2),
                f"{label}.boolean_signature_int",
            )
            expect_equal(
                errors,
                run_id,
                snapshot.get("boolean_signature"),
                signature,
                f"{label}.boolean_signature alias",
            )
        for field, goal in (("rho_p", "P"), ("rho_q", "Q"), ("rho_y", "Y")):
            expect_close(
                errors,
                run_id,
                truth.get(field),
                float(get(snapshot, f"behavior.{goal}", float("nan"))),
                f"{label}.truth_table.{field}",
            )
    probes = snapshot.get("probe_heldout_accuracy")
    if not isinstance(probes, Mapping) or set(probes) != set(LAYERS):
        errors.append(f"{run_id}: {label}.probe_heldout_accuracy layer set is malformed")
    else:
        for layer in LAYERS:
            values = probes.get(layer)
            required = {*GOALS, "truth_table_control"}
            if not isinstance(values, Mapping) or not required <= set(values):
                errors.append(f"{run_id}: {label} probe labels missing at {layer}")
                continue
            for probe_label in required:
                value = float(values[probe_label])
                if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    errors.append(
                        f"{run_id}: {label} probe {layer}/{probe_label} outside [0,1]"
                    )
    selective = snapshot.get("selective_final_hidden_probe")
    if isinstance(probes, Mapping) and isinstance(selective, Mapping):
        for goal in GOALS:
            expected = float(get(probes, f"final_hidden.{goal}")) - float(
                get(probes, "final_hidden.truth_table_control")
            )
            expect_close(
                errors,
                run_id,
                selective.get(goal),
                expected,
                f"{label}.selective_probe.{goal}",
            )


def _config_check(run: Run, errors: list[str], expected_configs: Mapping[str, Mapping[str, Any]]) -> None:
    run_id = run.path.name
    if run.schedule not in expected_configs:
        errors.append(f"{run_id}: unknown schedule {run.schedule!r}")
        return
    actual = _scientific_config(run.config)
    expected = expected_configs[run.schedule]
    if actual != expected:
        actual_text = json.dumps(actual, sort_keys=True, separators=(",", ":"), default=str)
        expected_text = json.dumps(expected, sort_keys=True, separators=(",", ":"), default=str)
        if actual_text != expected_text:
            errors.append(f"{run_id}: resolved scientific config differs from frozen current cell")


def validate_run(
    run: Run, errors: list[str], expected_configs: Mapping[str, Mapping[str, Any]]
) -> None:
    run_id = run.path.name
    _config_check(run, errors, expected_configs)
    expect_equal(errors, run_id, int(get(run.config, "seed", -1)), run.seed, "seed")
    expect_equal(errors, run_id, get(run.config, "h15.schedule"), run.schedule, "schedule")
    summary_fixed: Mapping[str, Any] = {
        "hypothesis": "h15",
        "condition": f"identical_evidence_order:{run.schedule}",
        "design_status": "adaptive_posthoc_identical_evidence_order_frozen_pre_outcome",
        "data.n_train": 10_000,
        "data.q_p": 0.90,
        "data.q_q": 0.95,
        "data.k_q": 2,
        "data.k_y": 3,
        "data.max_k_q": 3,
        "data.max_k_y": 5,
        "data.state_dim": 8,
        "data.probe_splits_disjoint": True,
        "plan.schedule": run.schedule,
        "plan.prefix_steps": PREFIX_STEPS,
        "plan.block_steps": BLOCK_STEPS,
        "plan.diagnostics_end": DIAGNOSTICS_END,
        "plan.washout_steps": 360,
        "plan.total_steps": TOTAL_STEPS,
        "plan.batch_size": BATCH_SIZE,
        "plan_audit.n_rows": 10_000,
        "plan_audit.total_steps": TOTAL_STEPS,
        "plan_audit.batch_size": BATCH_SIZE,
        "plan_audit.prefix_steps": PREFIX_STEPS,
        "plan_audit.block_steps": BLOCK_STEPS,
        "plan_audit.diagnostics_end": DIAGNOSTICS_END,
        "plan_audit.washout_steps": 360,
        "plan_audit.total_presentations": 1_000_000,
        "plan_audit.per_row_presentation_min": 100,
        "plan_audit.per_row_presentation_max": 100,
        "plan_audit.per_batch_raw_count_min": 15,
        "plan_audit.per_batch_raw_count_max": 16,
        "plan_audit.all_batches_label_balanced": True,
        "plan_audit.all_batches_raw_balanced": True,
        "plan_audit.all_batches_full": True,
        "plan_audit.all_rows_unique_within_batch": True,
        "plan_audit.every_row_once_per_registered_repetition": True,
        "plan_audit.phase_categories_verified": True,
        "measurement.probe_train_n": PROBE_TRAIN_N,
        "measurement.probe_eval_n": PROBE_EVAL_N,
        "measurement.probe_ridge": 0.001,
        "measurement.truth_table_control_seed": 1_500_450_271,
        "measurement.second_block_checkpoints": list(SECOND_BLOCK_STEPS[1:]),
        "measurement.washout_checkpoints": list(WASHOUT_STEPS),
        "measurement.direct_auc_endpoints": [0, AUC_HORIZON],
        "measurement.pure_threshold": PURE_THRESHOLD,
        "measurement.pure_margin": PURE_MARGIN,
        "measurement.late_stability_steps": list(LATE_STEPS),
        "measurement.late_stability_tolerance": LATE_TOLERANCE,
        "measurement.primary_effect_threshold": PRIMARY_THRESHOLD,
        "measurement.equivalence_margin": EQUIVALENCE_MARGIN,
        "measurement.terminal_effect_threshold": TERMINAL_THRESHOLD,
        "measurement.minimum_sign_count": MINIMUM_SIGN_COUNT,
        "training.batch_size": BATCH_SIZE,
        "training.all_minibatches_full": True,
        "training.prefix_steps": PREFIX_STEPS,
        "training.diagnostic_steps": 2 * BLOCK_STEPS,
        "training.washout_steps": 360,
        "training.total_steps": TOTAL_STEPS,
        "training.phase_examples_seen": {
            "prefix": 810_000,
            "diagnostic": 100_000,
            "washout": 90_000,
        },
        "training.total_examples_seen": 1_000_000,
        "training.replay_examples_seen_excluded_from_treatment_total": 810_000,
    }
    for path, expected in summary_fixed.items():
        expect_equal(errors, run_id, get(run.summary, path), expected, f"summary.{path}")
    wall = get(run.summary, "training.wall_seconds")
    if not isinstance(wall, (int, float)) or not math.isfinite(float(wall)) or wall <= 0:
        errors.append(f"{run_id}: invalid wall time {wall!r}")

    diagnostic = get(run.summary, "diagnostic_snapshots", {})
    washout = get(run.summary, "washout_snapshots", {})
    if not isinstance(diagnostic, Mapping) or set(diagnostic) != {
        str(step) for step in SECOND_BLOCK_STEPS
    }:
        errors.append(f"{run_id}: diagnostic snapshot lattice differs from frozen offsets")
    if not isinstance(washout, Mapping) or set(washout) != {
        str(step) for step in WASHOUT_STEPS
    }:
        errors.append(f"{run_id}: washout snapshot lattice differs from frozen offsets")
    prefix = run.summary.get("prefix_snapshot")
    if isinstance(prefix, Mapping):
        _validate_snapshot(
            run,
            cast(Mapping[str, Any], prefix),
            errors,
            label="prefix",
            local_step=PREFIX_STEPS,
            global_step=PREFIX_STEPS,
        )
    else:
        errors.append(f"{run_id}: missing prefix snapshot")
    if isinstance(diagnostic, Mapping) and set(diagnostic) == {
        str(step) for step in SECOND_BLOCK_STEPS
    }:
        for step in SECOND_BLOCK_STEPS:
            snapshot = cast(Mapping[str, Any], diagnostic[str(step)])
            _validate_snapshot(
                run,
                snapshot,
                errors,
                label=f"diagnostic.{step}",
                local_step=step,
                global_step=PREFIX_STEPS + BLOCK_STEPS + step,
            )
        expect_equal(
            errors,
            run_id,
            run.summary.get("first_half_snapshot"),
            diagnostic["0"],
            "first_half_snapshot alias",
        )
        expect_equal(
            errors,
            run_id,
            run.summary.get("post_diagnostic_snapshot"),
            diagnostic[str(BLOCK_STEPS)],
            "post_diagnostic_snapshot alias",
        )
    if isinstance(washout, Mapping) and set(washout) == {
        str(step) for step in WASHOUT_STEPS
    }:
        for step in WASHOUT_STEPS:
            snapshot = cast(Mapping[str, Any], washout[str(step)])
            _validate_snapshot(
                run,
                snapshot,
                errors,
                label=f"washout.{step}",
                local_step=step,
                global_step=DIAGNOSTICS_END + step,
            )
        expect_equal(
            errors,
            run_id,
            run.summary.get("final"),
            washout["360"],
            "final snapshot alias",
        )
        reconstructed = {
            step: cast(Mapping[str, Any], washout[str(step)]) for step in WASHOUT_STEPS
        }
        auc = normalized_auc(reconstructed)
        expect_close(
            errors,
            run_id,
            get(run.summary, "outcomes.primary_auc_m_y"),
            auc,
            "reconstructed washout AUC",
        )
        final = reconstructed[360]
        expect_close(
            errors,
            run_id,
            get(run.summary, "outcomes.terminal_m_y"),
            intended_margin(final),
            "reconstructed terminal m_y",
        )
        expect_equal(
            errors,
            run_id,
            get(run.summary, "outcomes.terminal_pure_goal"),
            _pure_goal(final),
            "terminal pure goal",
        )
    expect_equal(
        errors,
        run_id,
        get(run.summary, "outcomes.auc_horizon"),
        AUC_HORIZON,
        "AUC horizon",
    )
    if isinstance(prefix, Mapping):
        expect_equal(
            errors,
            run_id,
            get(run.summary, "outcomes.prefix_pure_goal"),
            _pure_goal(cast(Mapping[str, Any], prefix)),
            "prefix pure goal",
        )
    if isinstance(diagnostic, Mapping) and "0" in diagnostic and "200" in diagnostic:
        expect_equal(
            errors,
            run_id,
            get(run.summary, "outcomes.first_half_pure_goal"),
            _pure_goal(cast(Mapping[str, Any], diagnostic["0"])),
            "first-half pure goal",
        )
        expect_equal(
            errors,
            run_id,
            get(run.summary, "outcomes.post_diagnostic_pure_goal"),
            _pure_goal(cast(Mapping[str, Any], diagnostic["200"])),
            "post-diagnostic pure goal",
        )
        if isinstance(washout, Mapping) and "0" in washout:
            left = dict(cast(Mapping[str, Any], diagnostic["200"]))
            right = dict(cast(Mapping[str, Any], washout["0"]))
            left.pop("local_step", None)
            right.pop("local_step", None)
            expect_equal(
                errors,
                run_id,
                right,
                left,
                "post-diagnostic/washout-zero measurement equality",
            )

    replay = get(run.summary, "replay", {})
    expected_replay_checks = {
        "model_reports_equal",
        "initial_models_equal",
        "final_models_equal",
        "final_optimizers_equal",
        "samples_seen_equal",
        "optimizer_steps_equal",
        "training_batches_equal",
        "ordered_streams_equal",
    }
    checks = get(replay, "checks", {}) if isinstance(replay, Mapping) else {}
    if not isinstance(checks, Mapping) or set(checks) != expected_replay_checks or not all(
        value is True for value in checks.values()
    ):
        errors.append(f"{run_id}: deterministic observer-free replay checks failed")
    expect_equal(
        errors,
        run_id,
        get(replay, "observed_samples_seen", None) if isinstance(replay, Mapping) else None,
        810_000,
        "replay observed samples",
    )
    expect_equal(
        errors,
        run_id,
        get(replay, "unobserved_samples_seen", None) if isinstance(replay, Mapping) else None,
        810_000,
        "replay unobserved samples",
    )
    replay_hashes = get(replay, "hashes", {}) if isinstance(replay, Mapping) else {}
    required_replay_hashes = {
        "observed_initial_model",
        "unobserved_initial_model",
        "observed_final_model",
        "unobserved_final_model",
        "observed_final_optimizer",
        "unobserved_final_optimizer",
        "observed_training_batch",
        "unobserved_training_batch",
        "observed_prefix_atomic_stream",
        "unobserved_prefix_atomic_stream",
    }
    if not isinstance(replay_hashes, Mapping) or set(replay_hashes) != required_replay_hashes:
        errors.append(f"{run_id}: replay hash schema is malformed")
    elif not all(_hash_is_valid(value) for value in replay_hashes.values()):
        errors.append(f"{run_id}: replay contains malformed hashes")
    else:
        for left_key, right_key in (
            ("observed_initial_model", "unobserved_initial_model"),
            ("observed_final_model", "unobserved_final_model"),
            ("observed_final_optimizer", "unobserved_final_optimizer"),
            ("observed_training_batch", "unobserved_training_batch"),
            ("observed_prefix_atomic_stream", "unobserved_prefix_atomic_stream"),
        ):
            expect_equal(
                errors,
                run_id,
                replay_hashes[left_key],
                replay_hashes[right_key],
                f"replay {left_key}",
            )

    reset_expected = {
        "prefix_to_diagnostics": PREFIX_STEPS,
        "diagnostics_to_washout": DIAGNOSTICS_END,
    }
    for name, boundary in reset_expected.items():
        reset = get(run.summary, f"resets.{name}", {})
        for field, expected in {
            "boundary_global_step": boundary,
            "semantic_reset_implemented_by_fresh_object": True,
            "low_level_reset_optimizer_flag": False,
            "state_entry_count": 0,
            "adam_step_min": None,
            "adam_step_max": None,
            "adam_step_entry_count": 0,
            "param_group_count": 1,
            "optimized_parameter_count": 6,
        }.items():
            expect_equal(errors, run_id, get(reset, field), expected, f"reset {name}.{field}")
        if not _hash_is_valid(get(reset, "actual_state_digest")):
            errors.append(f"{run_id}: reset {name} has malformed optimizer digest")
    for path in (
        "data.training_batch_digest",
        "data.probe_train_digest",
        "data.probe_eval_digest",
        "plan.ordered_digest",
        "plan.row_exposure_digest",
        "plan.atomic_batch_multiset_digest",
        "plan_audit.ordered_index_digest",
        "plan_audit.ordered_sample_id_digest",
        "plan_audit.row_exposure_digest",
        "plan_audit.atomic_batch_multiset_digest",
        "hashes.initial_model",
        "hashes.prefix_final_model",
        "hashes.prefix_final_optimizer",
        "hashes.first_reset_optimizer",
        "hashes.post_diagnostic_model",
        "hashes.post_diagnostic_optimizer",
        "hashes.second_reset_optimizer",
        "hashes.final_model",
        "hashes.final_optimizer",
    ):
        if not _hash_is_valid(get(run.summary, path)):
            errors.append(f"{run_id}: malformed summary.{path}")
    expect_equal(
        errors,
        run_id,
        get(run.summary, "hashes.first_reset_optimizer"),
        get(run.summary, "resets.prefix_to_diagnostics.actual_state_digest"),
        "first reset hash alias",
    )
    expect_equal(
        errors,
        run_id,
        get(run.summary, "hashes.second_reset_optimizer"),
        get(run.summary, "resets.diagnostics_to_washout.actual_state_digest"),
        "second reset hash alias",
    )
    expect_equal(
        errors,
        run_id,
        get(run.summary, "data.training_batch_digest"),
        get(run.summary, "data.strata_audit.batch_digest"),
        "training/strata batch digest",
    )
    expect_equal(
        errors,
        run_id,
        get(run.summary, "plan.ordered_digest"),
        get(run.summary, "plan_audit.ordered_index_digest"),
        "ordered plan/audit digest",
    )
    for field in ("row_exposure_digest", "atomic_batch_multiset_digest"):
        expect_equal(
            errors,
            run_id,
            get(run.summary, f"plan.{field}"),
            get(run.summary, f"plan_audit.{field}"),
            f"plan/audit {field}",
        )

    phases: tuple[tuple[str, Sequence[int]], ...] = (
        ("prefix", (PREFIX_STEPS,)),
        ("diagnostic", SECOND_BLOCK_STEPS),
        ("washout", WASHOUT_STEPS),
    )
    for phase, steps in phases:
        for suffix, split, interventions, metrics in (
            (
                "behavior",
                "factorial_eval",
                ("none",),
                ("rho_p", "rho_q", "rho_y_code", "target_accuracy", "m_y"),
            ),
            (
                "causal",
                "factorial_eval",
                GOALS,
                ("causal_score", "causal_prob_score"),
            ),
            (
                "probe",
                "factorial_probe",
                ("none",),
                (
                    *(
                        f"representations__{layer}__heldout_accuracy__{label}"
                        for layer in LAYERS
                        for label in (*GOALS, "truth_table_control")
                    ),
                    "raw_codeword_count",
                    "truth_table_control_positive_fraction",
                    "n_train",
                    "n_heldout",
                    "n_labels",
                ),
            ),
            (
                "truth_table",
                "factorial_eval",
                ("none",),
                (
                    "boolean_signature_int",
                    "tuple_consistency",
                    "codeword_consistency",
                    "nuisance_consistency",
                    "sign_inversion_symmetry",
                ),
            ),
        ):
            stage = f"{phase}_{suffix}"
            for intervention in interventions:
                for metric in metrics:
                    key = (stage, split, intervention, metric)
                    _require_metric_steps(run, errors, key, steps)
                    for step in steps:
                        point = run.metrics.get(key, {}).get(step)
                        if point is None:
                            continue
                        if phase == "prefix":
                            global_step = PREFIX_STEPS
                        elif phase == "diagnostic":
                            global_step = PREFIX_STEPS + BLOCK_STEPS + step
                        else:
                            global_step = DIAGNOSTICS_END + step
                        if point.global_step != global_step or point.stage_step != step:
                            errors.append(f"{run_id}: {key} counters wrong at {step}")
                        if point.examples_seen != global_step * BATCH_SIZE:
                            errors.append(f"{run_id}: {key} examples_seen wrong at {step}")
                        if point.n != PROBE_EVAL_N:
                            errors.append(f"{run_id}: {key} n={point.n}, expected {PROBE_EVAL_N}")
                        snapshot = run.snapshot(phase, step)
                        expected_value = _snapshot_metric_value(snapshot, key)
                        if not math.isclose(
                            point.value, expected_value, rel_tol=0.0, abs_tol=1e-12
                        ):
                            errors.append(f"{run_id}: {key} metric/summary mismatch at {step}")

    optimization_steps: Mapping[str, Sequence[int]] = {
        "prefix": (PREFIX_STEPS,),
        "diagnostic": tuple(
            sorted({BLOCK_STEPS, *(BLOCK_STEPS + step for step in SECOND_BLOCK_STEPS[1:])})
        ),
        "washout": WASHOUT_STEPS[1:],
    }
    for phase, steps in optimization_steps.items():
        for metric in ("loss", "primary_loss", "train_batch_accuracy", "optimizer_steps"):
            key = (f"{phase}_optimization", "train_minibatch", "none", metric)
            _require_metric_steps(run, errors, key, steps)
            for step, point in run.metrics.get(key, {}).items():
                offset = 0 if phase == "prefix" else PREFIX_STEPS if phase == "diagnostic" else DIAGNOSTICS_END
                global_step = offset + step
                if (
                    point.global_step != global_step
                    or point.stage_step != step
                    or point.examples_seen != global_step * BATCH_SIZE
                    or point.n != 10_000
                ):
                    errors.append(f"{run_id}: malformed {phase} optimizer counters for {key} at {step}")
                if metric == "optimizer_steps" and point.value != step:
                    errors.append(f"{run_id}: wrong {phase} optimizer step value at {step}")


def _validate_metric_structure_blind(run: Run, errors: list[str]) -> None:
    """Audit pilot checkpoint coordinates without dereferencing outcomes."""

    run_id = run.path.name
    phase_steps: Mapping[str, Sequence[int]] = {
        "prefix": (PREFIX_STEPS,),
        "diagnostic": SECOND_BLOCK_STEPS,
        "washout": WASHOUT_STEPS,
    }
    for phase, steps in phase_steps.items():
        for suffix, split, interventions, metrics in (
            (
                "behavior",
                "factorial_eval",
                ("none",),
                ("rho_p", "rho_q", "rho_y_code", "target_accuracy", "m_y"),
            ),
            (
                "causal",
                "factorial_eval",
                GOALS,
                ("causal_score", "causal_prob_score"),
            ),
            (
                "probe",
                "factorial_probe",
                ("none",),
                (
                    *(
                        f"representations__{layer}__heldout_accuracy__{label}"
                        for layer in LAYERS
                        for label in (*GOALS, "truth_table_control")
                    ),
                    "raw_codeword_count",
                    "truth_table_control_positive_fraction",
                    "n_train",
                    "n_heldout",
                    "n_labels",
                ),
            ),
            (
                "truth_table",
                "factorial_eval",
                ("none",),
                (
                    "boolean_signature_int",
                    "tuple_consistency",
                    "codeword_consistency",
                    "nuisance_consistency",
                    "sign_inversion_symmetry",
                ),
            ),
        ):
            stage = f"{phase}_{suffix}"
            for intervention in interventions:
                for metric in metrics:
                    key = (stage, split, intervention, metric)
                    _require_metric_steps(run, errors, key, steps)
                    for step, point in run.metrics.get(key, {}).items():
                        global_step = (
                            PREFIX_STEPS
                            if phase == "prefix"
                            else PREFIX_STEPS + BLOCK_STEPS + step
                            if phase == "diagnostic"
                            else DIAGNOSTICS_END + step
                        )
                        if (
                            point.stage_step != step
                            or point.global_step != global_step
                            or point.examples_seen != global_step * BATCH_SIZE
                            or point.n != PROBE_EVAL_N
                        ):
                            errors.append(
                                f"{run_id}: blinded checkpoint counters malformed for {key} at {step}"
                            )
    optimization_steps: Mapping[str, Sequence[int]] = {
        "prefix": (PREFIX_STEPS,),
        "diagnostic": tuple(
            sorted({BLOCK_STEPS, *(BLOCK_STEPS + step for step in SECOND_BLOCK_STEPS[1:])})
        ),
        "washout": WASHOUT_STEPS[1:],
    }
    for phase, steps in optimization_steps.items():
        for metric in ("loss", "primary_loss", "train_batch_accuracy", "optimizer_steps"):
            key = (f"{phase}_optimization", "train_minibatch", "none", metric)
            _require_metric_steps(run, errors, key, steps)
            for step, point in run.metrics.get(key, {}).items():
                offset = (
                    0
                    if phase == "prefix"
                    else PREFIX_STEPS
                    if phase == "diagnostic"
                    else DIAGNOSTICS_END
                )
                global_step = offset + step
                if (
                    point.stage_step != step
                    or point.global_step != global_step
                    or point.examples_seen != global_step * BATCH_SIZE
                    or point.n != 10_000
                ):
                    errors.append(
                        f"{run_id}: blinded optimizer counters malformed for {key} at {step}"
                    )


def validate_run_pilot(
    run: Run, errors: list[str], expected_configs: Mapping[str, Mapping[str, Any]]
) -> None:
    """Validate integrity and the registered manipulation fields only.

    This function intentionally never reads ``outcomes``, scientific snapshot
    values after the first half, or any metric ``value``.  Keeping it separate
    from ``validate_run`` makes accidental pilot outcome analysis difficult.
    """

    run_id = run.path.name
    _config_check(run, errors, expected_configs)
    expect_equal(errors, run_id, int(get(run.config, "seed", -1)), run.seed, "seed")
    expect_equal(errors, run_id, get(run.config, "h15.schedule"), run.schedule, "schedule")
    integrity_fixed: Mapping[str, Any] = {
        "hypothesis": "h15",
        "condition": f"identical_evidence_order:{run.schedule}",
        "design_status": "adaptive_posthoc_identical_evidence_order_frozen_pre_outcome",
        "data.n_train": 10_000,
        "data.q_p": 0.90,
        "data.q_q": 0.95,
        "data.k_q": 2,
        "data.k_y": 3,
        "data.max_k_q": 3,
        "data.max_k_y": 5,
        "data.state_dim": 8,
        "data.probe_splits_disjoint": True,
        "plan.schedule": run.schedule,
        "plan.prefix_steps": PREFIX_STEPS,
        "plan.block_steps": BLOCK_STEPS,
        "plan.diagnostics_end": DIAGNOSTICS_END,
        "plan.washout_steps": 360,
        "plan.total_steps": TOTAL_STEPS,
        "plan.batch_size": BATCH_SIZE,
        "plan_audit.n_rows": 10_000,
        "plan_audit.total_steps": TOTAL_STEPS,
        "plan_audit.batch_size": BATCH_SIZE,
        "plan_audit.prefix_steps": PREFIX_STEPS,
        "plan_audit.block_steps": BLOCK_STEPS,
        "plan_audit.diagnostics_end": DIAGNOSTICS_END,
        "plan_audit.washout_steps": 360,
        "plan_audit.total_presentations": 1_000_000,
        "plan_audit.per_row_presentation_min": 100,
        "plan_audit.per_row_presentation_max": 100,
        "plan_audit.per_batch_raw_count_min": 15,
        "plan_audit.per_batch_raw_count_max": 16,
        "plan_audit.all_batches_label_balanced": True,
        "plan_audit.all_batches_raw_balanced": True,
        "plan_audit.all_batches_full": True,
        "plan_audit.all_rows_unique_within_batch": True,
        "plan_audit.every_row_once_per_registered_repetition": True,
        "plan_audit.phase_categories_verified": True,
        "measurement.second_block_checkpoints": list(SECOND_BLOCK_STEPS[1:]),
        "measurement.washout_checkpoints": list(WASHOUT_STEPS),
        "measurement.direct_auc_endpoints": [0, AUC_HORIZON],
        "measurement.late_stability_steps": list(LATE_STEPS),
        "measurement.late_stability_tolerance": LATE_TOLERANCE,
        "measurement.primary_effect_threshold": PRIMARY_THRESHOLD,
        "measurement.equivalence_margin": EQUIVALENCE_MARGIN,
        "measurement.terminal_effect_threshold": TERMINAL_THRESHOLD,
        "measurement.minimum_sign_count": MINIMUM_SIGN_COUNT,
        "training.batch_size": BATCH_SIZE,
        "training.all_minibatches_full": True,
        "training.prefix_steps": PREFIX_STEPS,
        "training.diagnostic_steps": 2 * BLOCK_STEPS,
        "training.washout_steps": 360,
        "training.total_steps": TOTAL_STEPS,
        "training.phase_examples_seen": {
            "prefix": 810_000,
            "diagnostic": 100_000,
            "washout": 90_000,
        },
        "training.total_examples_seen": 1_000_000,
        "training.replay_examples_seen_excluded_from_treatment_total": 810_000,
    }
    for path, expected in integrity_fixed.items():
        expect_equal(errors, run_id, get(run.summary, path), expected, f"summary.{path}")

    prefix = run.summary.get("prefix_snapshot")
    first_half = run.summary.get("first_half_snapshot")
    if not isinstance(prefix, Mapping) or not isinstance(first_half, Mapping):
        errors.append(f"{run_id}: pilot gate snapshots are missing")
    else:
        expect_equal(
            errors,
            run_id,
            prefix.get("local_step"),
            PREFIX_STEPS,
            "prefix checkpoint local step",
        )
        expect_equal(
            errors,
            run_id,
            prefix.get("global_step"),
            PREFIX_STEPS,
            "prefix checkpoint global step",
        )
        expect_equal(
            errors,
            run_id,
            prefix.get("examples_seen"),
            810_000,
            "prefix checkpoint examples",
        )
        if prefix.get("pure_goal") not in {None, *GOALS}:
            errors.append(f"{run_id}: prefix pure_goal is malformed")
        expect_equal(
            errors,
            run_id,
            first_half.get("local_step"),
            0,
            "first-half local step",
        )
        expect_equal(
            errors,
            run_id,
            first_half.get("global_step"),
            PREFIX_STEPS + BLOCK_STEPS,
            "first-half global step",
        )
        expect_equal(
            errors,
            run_id,
            first_half.get("examples_seen"),
            (PREFIX_STEPS + BLOCK_STEPS) * BATCH_SIZE,
            "first-half examples",
        )
        if first_half.get("pure_goal") not in {None, *GOALS}:
            errors.append(f"{run_id}: first-half pure_goal is malformed")

    diagnostic = run.summary.get("diagnostic_snapshots")
    washout = run.summary.get("washout_snapshots")
    if not isinstance(diagnostic, Mapping) or set(diagnostic) != {
        str(step) for step in SECOND_BLOCK_STEPS
    }:
        errors.append(f"{run_id}: pilot diagnostic checkpoint lattice is malformed")
    elif isinstance(first_half, Mapping):
        # Only coordinate fields and the permitted gate field are compared.
        first_record = cast(Mapping[str, Any], diagnostic["0"])
        for field in ("local_step", "global_step", "examples_seen", "pure_goal"):
            expect_equal(
                errors,
                run_id,
                first_record.get(field),
                first_half.get(field),
                f"first-half alias {field}",
            )
        for step in SECOND_BLOCK_STEPS:
            record = cast(Mapping[str, Any], diagnostic[str(step)])
            expect_equal(errors, run_id, record.get("local_step"), step, f"diagnostic {step} local")
            expect_equal(
                errors,
                run_id,
                record.get("global_step"),
                PREFIX_STEPS + BLOCK_STEPS + step,
                f"diagnostic {step} global",
            )
            expect_equal(
                errors,
                run_id,
                record.get("examples_seen"),
                (PREFIX_STEPS + BLOCK_STEPS + step) * BATCH_SIZE,
                f"diagnostic {step} examples",
            )
    if not isinstance(washout, Mapping) or set(washout) != {
        str(step) for step in WASHOUT_STEPS
    }:
        errors.append(f"{run_id}: pilot washout checkpoint lattice is malformed")
    else:
        for step in WASHOUT_STEPS:
            record = cast(Mapping[str, Any], washout[str(step)])
            # Scientific values are intentionally not read here.
            expect_equal(errors, run_id, record.get("local_step"), step, f"washout {step} local")
            expect_equal(
                errors,
                run_id,
                record.get("global_step"),
                DIAGNOSTICS_END + step,
                f"washout {step} global",
            )
            expect_equal(
                errors,
                run_id,
                record.get("examples_seen"),
                (DIAGNOSTICS_END + step) * BATCH_SIZE,
                f"washout {step} examples",
            )

    replay = run.summary.get("replay")
    checks = get(cast(Mapping[str, Any], replay), "checks", {}) if isinstance(replay, Mapping) else {}
    expected_checks = {
        "model_reports_equal",
        "initial_models_equal",
        "final_models_equal",
        "final_optimizers_equal",
        "samples_seen_equal",
        "optimizer_steps_equal",
        "training_batches_equal",
        "ordered_streams_equal",
    }
    if not isinstance(checks, Mapping) or set(checks) != expected_checks or not all(
        value is True for value in checks.values()
    ):
        errors.append(f"{run_id}: pilot prefix replay checks failed")
    expect_equal(errors, run_id, get(run.summary, "replay.observed_samples_seen"), 810_000, "replay observed samples")
    expect_equal(errors, run_id, get(run.summary, "replay.unobserved_samples_seen"), 810_000, "replay unobserved samples")
    replay_hashes = get(run.summary, "replay.hashes", {})
    if not isinstance(replay_hashes, Mapping) or not all(
        _hash_is_valid(value) for value in replay_hashes.values()
    ):
        errors.append(f"{run_id}: pilot replay hashes malformed")
    elif set(replay_hashes) != {
        "observed_initial_model",
        "unobserved_initial_model",
        "observed_final_model",
        "unobserved_final_model",
        "observed_final_optimizer",
        "unobserved_final_optimizer",
        "observed_training_batch",
        "unobserved_training_batch",
        "observed_prefix_atomic_stream",
        "unobserved_prefix_atomic_stream",
    }:
        errors.append(f"{run_id}: pilot replay hash key set malformed")
    else:
        for left, right in (
            ("observed_initial_model", "unobserved_initial_model"),
            ("observed_final_model", "unobserved_final_model"),
            ("observed_final_optimizer", "unobserved_final_optimizer"),
            ("observed_training_batch", "unobserved_training_batch"),
            ("observed_prefix_atomic_stream", "unobserved_prefix_atomic_stream"),
        ):
            expect_equal(errors, run_id, replay_hashes[left], replay_hashes[right], f"pilot replay {left}")

    for name, boundary in (
        ("prefix_to_diagnostics", PREFIX_STEPS),
        ("diagnostics_to_washout", DIAGNOSTICS_END),
    ):
        reset = get(run.summary, f"resets.{name}", {})
        for field, expected in {
            "boundary_global_step": boundary,
            "semantic_reset_implemented_by_fresh_object": True,
            "low_level_reset_optimizer_flag": False,
            "state_entry_count": 0,
            "adam_step_min": None,
            "adam_step_max": None,
            "adam_step_entry_count": 0,
            "param_group_count": 1,
            "optimized_parameter_count": 6,
        }.items():
            expect_equal(errors, run_id, get(reset, field), expected, f"pilot reset {name}.{field}")
        if not _hash_is_valid(get(reset, "actual_state_digest")):
            errors.append(f"{run_id}: pilot reset {name} digest malformed")

    for path in (
        "data.training_batch_digest",
        "data.probe_train_digest",
        "data.probe_eval_digest",
        "plan.ordered_digest",
        "plan.row_exposure_digest",
        "plan.atomic_batch_multiset_digest",
        "plan_audit.ordered_index_digest",
        "plan_audit.ordered_sample_id_digest",
        "plan_audit.row_exposure_digest",
        "plan_audit.atomic_batch_multiset_digest",
        "hashes.initial_model",
        "hashes.prefix_final_model",
        "hashes.prefix_final_optimizer",
        "hashes.first_reset_optimizer",
        "hashes.post_diagnostic_model",
        "hashes.post_diagnostic_optimizer",
        "hashes.second_reset_optimizer",
        "hashes.final_model",
        "hashes.final_optimizer",
    ):
        if not _hash_is_valid(get(run.summary, path)):
            errors.append(f"{run_id}: pilot integrity hash malformed at {path}")
    for field in ("row_exposure_digest", "atomic_batch_multiset_digest"):
        expect_equal(
            errors,
            run_id,
            get(run.summary, f"plan.{field}"),
            get(run.summary, f"plan_audit.{field}"),
            f"pilot plan/audit {field}",
        )
    expect_equal(
        errors,
        run_id,
        get(run.summary, "plan.ordered_digest"),
        get(run.summary, "plan_audit.ordered_index_digest"),
        "pilot ordered digest alias",
    )
    expect_equal(
        errors,
        run_id,
        get(run.summary, "hashes.first_reset_optimizer"),
        get(run.summary, "resets.prefix_to_diagnostics.actual_state_digest"),
        "pilot first reset alias",
    )
    expect_equal(
        errors,
        run_id,
        get(run.summary, "hashes.second_reset_optimizer"),
        get(run.summary, "resets.diagnostics_to_washout.actual_state_digest"),
        "pilot second reset alias",
    )
    _validate_metric_structure_blind(run, errors)


def _reconstruct_seed_integrity(
    seed: int, runs: Mapping[str, Run]
) -> dict[str, Any]:
    reference = runs[SCHEDULES[0]]
    train_batch = make_identical_evidence_dataset(
        10_000,
        q_p=0.90,
        q_q=0.95,
        k_q=2,
        k_y=3,
        seed=seed,
        max_k_q=3,
        max_k_y=5,
        state_dim=8,
    )
    strata = audit_evidence_strata(
        train_batch, expected_counts={"A": 9_000, "B": 500, "D": 500}
    )
    probe_train = make_competing_factorial_dataset(
        PROBE_TRAIN_N,
        k_q=2,
        k_y=3,
        seed=50_000_000 + seed,
        control_seed=1_500_450_271,
        max_k_q=3,
        max_k_y=5,
        state_dim=8,
        split="factorial_probe_train",
        id_offset=2_000_000_000,
    )
    probe_eval = make_competing_factorial_dataset(
        PROBE_EVAL_N,
        k_q=2,
        k_y=3,
        seed=60_000_000 + seed,
        control_seed=1_500_450_271,
        max_k_q=3,
        max_k_y=5,
        state_dim=8,
        split="factorial_probe_eval",
        id_offset=2_100_000_000,
    )
    model, model_report = build_model(train_batch, reference.config, seed)
    initial_model_hash = stable_state_digest(model.state_dict())
    empty_optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.003,
        weight_decay=0.0,
    )
    empty_optimizer_hash = stable_state_digest(empty_optimizer.state_dict())
    plans: dict[str, dict[str, Any]] = {}
    for schedule in SCHEDULES:
        plan = make_atomic_evidence_plan(
            train_batch,
            schedule=cast(Any, schedule),
            prefix_a_repetitions=90,
            washout_a_repetitions=10,
            diagnostic_repetitions=100,
            batch_size=BATCH_SIZE,
            seed=91_000_000 + seed,
        )
        audit = audit_atomic_evidence_plan(
            train_batch,
            plan,
            prefix_a_repetitions=90,
            washout_a_repetitions=10,
            diagnostic_repetitions=100,
        )
        plans[schedule] = {
            "summary": {
                "schedule": schedule,
                "prefix_steps": plan.prefix_steps,
                "block_steps": plan.block_steps,
                "diagnostics_end": plan.diagnostics_end,
                "washout_steps": plan.washout_steps,
                "total_steps": plan.total_steps,
                "batch_size": plan.batch_size,
                "component_digests": dict(plan.component_digests),
                "ordered_digest": plan.ordered_digest,
                "row_exposure_digest": plan.row_exposure_digest,
                "atomic_batch_multiset_digest": plan.atomic_batch_multiset_digest,
            },
            "audit": audit,
        }
    return {
        "training_batch_digest": semantic_batch_digest(train_batch),
        "strata_audit": strata,
        "probe_train_digest": semantic_batch_digest(probe_train),
        "probe_eval_digest": semantic_batch_digest(probe_eval),
        "initial_model_hash": initial_model_hash,
        "empty_optimizer_hash": empty_optimizer_hash,
        "model_report": model_report,
        "plans": plans,
    }


def _cross_run_audit(runs: Sequence[Run], errors: list[str]) -> dict[str, Any]:
    grouped: dict[int, dict[str, Run]] = defaultdict(dict)
    for run in runs:
        grouped[run.seed][run.schedule] = run
    reconstruction_rows: list[dict[str, Any]] = []
    for seed, arms in sorted(grouped.items()):
        if set(arms) != set(SCHEDULES):
            errors.append(f"seed {seed}: schedule block is incomplete")
            continue
        rebuilt = _reconstruct_seed_integrity(seed, arms)
        for schedule, run in arms.items():
            run_id = run.path.name
            expect_equal(
                errors,
                run_id,
                get(run.summary, "data.training_batch_digest"),
                rebuilt["training_batch_digest"],
                "reconstructed training batch",
            )
            expect_equal(
                errors,
                run_id,
                get(run.summary, "data.strata_audit"),
                rebuilt["strata_audit"],
                "reconstructed strata audit",
            )
            expect_equal(
                errors,
                run_id,
                get(run.summary, "data.probe_train_digest"),
                rebuilt["probe_train_digest"],
                "reconstructed probe train",
            )
            expect_equal(
                errors,
                run_id,
                get(run.summary, "data.probe_eval_digest"),
                rebuilt["probe_eval_digest"],
                "reconstructed probe eval",
            )
            expect_equal(
                errors,
                run_id,
                get(run.summary, "model"),
                rebuilt["model_report"],
                "reconstructed model report",
            )
            expect_equal(
                errors,
                run_id,
                get(run.summary, "hashes.initial_model"),
                rebuilt["initial_model_hash"],
                "reconstructed initial model",
            )
            expect_equal(
                errors,
                run_id,
                get(run.summary, "plan"),
                get(cast(Mapping[str, Any], rebuilt["plans"]), f"{schedule}.summary"),
                "reconstructed atomic plan",
            )
            expect_equal(
                errors,
                run_id,
                get(run.summary, "plan_audit"),
                get(cast(Mapping[str, Any], rebuilt["plans"]), f"{schedule}.audit"),
                "reconstructed atomic plan audit",
            )
            for reset_name, hash_name in (
                ("prefix_to_diagnostics", "first_reset_optimizer"),
                ("diagnostics_to_washout", "second_reset_optimizer"),
            ):
                expect_equal(
                    errors,
                    run_id,
                    get(run.summary, f"resets.{reset_name}.actual_state_digest"),
                    rebuilt["empty_optimizer_hash"],
                    f"reconstructed empty optimizer at {reset_name}",
                )
                expect_equal(
                    errors,
                    run_id,
                    get(run.summary, f"hashes.{hash_name}"),
                    rebuilt["empty_optimizer_hash"],
                    f"reconstructed reset hash {hash_name}",
                )
            expect_equal(
                errors,
                run_id,
                get(run.summary, "replay.hashes.observed_training_batch"),
                rebuilt["training_batch_digest"],
                "replay training batch",
            )
            expect_equal(
                errors,
                run_id,
                get(run.summary, "replay.hashes.observed_prefix_atomic_stream"),
                get(run.summary, "plan.component_digests.A_prefix"),
                "replay prefix stream",
            )
            reconstruction_rows.append(
                {
                    "seed": seed,
                    "schedule": schedule,
                    "training_batch_digest": rebuilt["training_batch_digest"],
                    "ordered_digest": get(run.summary, "plan.ordered_digest"),
                    "row_exposure_digest": get(run.summary, "plan.row_exposure_digest"),
                    "atomic_batch_multiset_digest": get(
                        run.summary, "plan.atomic_batch_multiset_digest"
                    ),
                    "initial_model_hash": rebuilt["initial_model_hash"],
                    "empty_optimizer_hash": rebuilt["empty_optimizer_hash"],
                }
            )

        common_paths = (
            "data.training_batch_digest",
            "data.probe_train_digest",
            "data.probe_eval_digest",
            "plan.row_exposure_digest",
            "plan.atomic_batch_multiset_digest",
            "hashes.initial_model",
            "hashes.prefix_final_model",
            "hashes.prefix_final_optimizer",
            "hashes.first_reset_optimizer",
            "hashes.second_reset_optimizer",
            "replay.hashes.observed_final_model",
            "replay.hashes.observed_final_optimizer",
        )
        for path in common_paths:
            values = {get(run.summary, path) for run in arms.values()}
            if len(values) != 1:
                errors.append(f"seed {seed}: cross-schedule {path} differs")
        components = {
            json.dumps(get(run.summary, "plan.component_digests"), sort_keys=True)
            for run in arms.values()
        }
        if len(components) != 1:
            errors.append(f"seed {seed}: within-category component streams differ")
        ordered = {get(run.summary, "plan.ordered_digest") for run in arms.values()}
        if len(ordered) != 3:
            errors.append(f"seed {seed}: the three ordered streams are not distinct")

    return {
        "seed_blocks": len(grouped),
        "reconstruction_rows": reconstruction_rows,
        "data_reconstructed": True,
        "atomic_plans_reconstructed": True,
        "initial_models_reconstructed": True,
        "empty_optimizers_reconstructed": True,
        "cross_schedule_prefix_hashes_equal": True,
        "cross_schedule_atomic_multisets_equal": True,
        "ordered_streams_distinct": True,
    }


def load_runs(root: Path, *, mode: str) -> tuple[list[Run], dict[str, Any]]:
    directory = root / "h15" / "identical_evidence_order"
    if not directory.is_dir():
        raise RuntimeError(f"E18 {mode} audit: missing artifact directory {directory}")
    config_paths = sorted(directory.glob("*/resolved_config.yaml"))
    if not config_paths:
        raise RuntimeError(f"E18 {mode} audit: no materialized runs in {directory}")
    expected_seeds = PILOT_SEEDS if mode == "pilot" else CONFIRMATORY_SEEDS
    expected_count = len(expected_seeds) * len(SCHEDULES)
    if len(config_paths) != expected_count:
        raise RuntimeError(
            f"E18 {mode} audit: found {len(config_paths)} artifacts, expected {expected_count}"
        )
    errors: list[str] = []
    expected_configs = _expected_configs()
    runs: list[Run] = []
    for config_path in config_paths:
        path = config_path.parent
        run_id = path.name
        for required in (
            "COMPLETE",
            "status.json",
            "metadata.json",
            "summary.json",
            "metrics.jsonl",
        ):
            if not (path / required).is_file():
                errors.append(f"{run_id}: missing {required}")
        if errors and not (path / "summary.json").is_file():
            continue
        config = load_yaml(config_path)
        summary = load_json(path / "summary.json")
        metadata = load_json(path / "metadata.json")
        status = load_json(path / "status.json")
        seed = int(summary.get("seed", -1))
        schedule = str(summary.get("schedule", ""))
        expect_equal(errors, run_id, status.get("state"), "complete", "status state")
        expect_equal(errors, run_id, status.get("run_id"), run_id, "status run-id")
        expect_equal(errors, run_id, metadata.get("run_id"), run_id, "metadata run-id")
        expect_equal(errors, run_id, metadata.get("seed"), seed, "metadata seed")
        expect_equal(
            errors,
            run_id,
            expected_artifact_run_id(config, metadata),
            run_id,
            "artifact run-id reconstruction",
        )
        try:
            complete_text = (path / "COMPLETE").read_text(encoding="utf-8")
        except OSError as error:
            errors.append(f"{run_id}: cannot read COMPLETE: {error}")
        else:
            expect_equal(errors, run_id, complete_text, "complete\n", "COMPLETE contents")
        if (path / "predictions.jsonl").is_file() and (path / "predictions.jsonl").stat().st_size:
            errors.append(f"{run_id}: predictions were saved despite frozen suppression")
        checkpoint_dir = path / "checkpoints"
        if checkpoint_dir.is_dir() and any(checkpoint_dir.iterdir()):
            errors.append(f"{run_id}: checkpoints were saved despite frozen suppression")
        metric_index, line_count = _load_metric_index(
            path / "metrics.jsonl",
            run_id=run_id,
            seed=seed,
            schedule=schedule,
            errors=errors,
            blind=mode == "pilot",
        )
        run = Run(
            path=path,
            config=config,
            summary=summary,
            metadata=metadata,
            metrics=metric_index,
            metric_line_count=line_count,
        )
        if mode == "pilot":
            validate_run_pilot(run, errors, expected_configs)
        else:
            validate_run(run, errors, expected_configs)
        runs.append(run)

    keys = [(run.seed, run.schedule) for run in runs]
    expected_grid = {(seed, schedule) for seed in expected_seeds for schedule in SCHEDULES}
    if len(keys) != len(set(keys)):
        errors.append("duplicate seed/schedule artifacts")
    if set(keys) != expected_grid:
        errors.append(
            f"E18 {mode} grid missing={sorted(expected_grid - set(keys))} "
            f"unexpected={sorted(set(keys) - expected_grid)}"
        )
    current = implementation_provenance(REPO)
    current_fingerprint = current["implementation_fingerprint"]
    fingerprints = {
        get(run.metadata, "implementation.implementation_fingerprint") for run in runs
    }
    if fingerprints != {current_fingerprint}:
        errors.append(
            "artifact implementation fingerprint does not match current ForkWorld source; "
            f"artifact={sorted(str(value) for value in fingerprints)}, current={current_fingerprint}"
        )
    expect_fingerprint = "905d7ec6a7a2003de15ebe7ad0b0bdd0028573e71fcef52b579bc22932a24720"
    if mode == "pilot" and current_fingerprint != expect_fingerprint:
        errors.append(
            f"pilot runner fingerprint drifted from frozen pre-pilot value {expect_fingerprint}"
        )
    for run in runs:
        expect_equal(
            errors,
            run.path.name,
            get(run.metadata, "implementation.artifact_schema_version"),
            ARTIFACT_SCHEMA_VERSION,
            "artifact schema version",
        )
        expect_equal(
            errors,
            run.path.name,
            get(run.metadata, "implementation.source_fingerprint_schema_version"),
            SOURCE_FINGERPRINT_SCHEMA_VERSION,
            "source fingerprint schema version",
        )
        expect_equal(
            errors,
            run.path.name,
            get(run.metadata, "implementation.source_file_count"),
            current["source_file_count"],
            "source file count",
        )
    if errors:
        _raise_audit(f"E18 {mode} artifact audit", errors)
    cross_errors: list[str] = []
    cross = _cross_run_audit(runs, cross_errors)
    if cross_errors:
        _raise_audit(f"E18 {mode} reconstruction/cross-run audit", cross_errors)
    audit = {
        "mode": mode,
        "artifact_root": str(root.resolve()),
        "expected_runs": expected_count,
        "completed_runs": len(runs),
        "seeds": list(expected_seeds),
        "schedules": list(SCHEDULES),
        "metric_line_count": sum(run.metric_line_count for run in runs),
        "implementation_fingerprint": current_fingerprint,
        "source_file_count": current["source_file_count"],
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "source_fingerprint_schema_version": SOURCE_FINGERPRINT_SCHEMA_VERSION,
        "all_complete": True,
        "exact_grid": True,
        "outcome_blinded": mode == "pilot",
        **{key: value for key, value in cross.items() if key != "reconstruction_rows"},
        "reconstruction_rows": cross["reconstruction_rows"],
    }
    return sorted(runs, key=lambda run: (run.seed, run.schedule)), audit


def pilot_gate_rows(runs: Sequence[Run]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return only the frozen prefix/first-half manipulation gate."""

    by_seed = {(run.seed, run.schedule): run for run in runs}
    rows: list[dict[str, Any]] = []
    passing_seeds: list[int] = []
    for seed in PILOT_SEEDS:
        prefix_goals = {
            schedule: get(by_seed[(seed, schedule)].summary, "prefix_snapshot.pure_goal")
            for schedule in SCHEDULES
        }
        prefix_pure_p = all(goal == "P" for goal in prefix_goals.values())
        b_first_goal = get(
            by_seed[(seed, "b_then_d")].summary, "first_half_snapshot.pure_goal"
        )
        d_first_goal = get(
            by_seed[(seed, "d_then_b")].summary, "first_half_snapshot.pure_goal"
        )
        passed = prefix_pure_p and b_first_goal == "Q" and d_first_goal == "Y"
        if passed:
            passing_seeds.append(seed)
        rows.append(
            {
                "seed": seed,
                "prefix_b_then_d_pure_goal": prefix_goals["b_then_d"],
                "prefix_d_then_b_pure_goal": prefix_goals["d_then_b"],
                "prefix_interleave_pure_goal": prefix_goals["interleave"],
                "prefix_all_pure_P": prefix_pure_p,
                "b_then_d_first_half_pure_goal": b_first_goal,
                "d_then_b_first_half_pure_goal": d_first_goal,
                "registered_manipulation_pass": passed,
            }
        )
    decision = {
        "integrity_gate_passed": True,
        "manipulation_passing_seed_count": len(passing_seeds),
        "manipulation_passing_seeds": passing_seeds,
        "minimum_required": 2,
        "pilot_gate_passed": len(passing_seeds) >= 2,
        "confirmatory_panel_authorized_by_frozen_gate": len(passing_seeds) >= 2,
        "outcome_blinded": True,
        "fields_inspected": ["prefix_snapshot.pure_goal", "first_half_snapshot.pure_goal"],
        "fields_not_inspected": [
            "outcomes.primary_auc_m_y",
            "outcomes.terminal_m_y",
            "post_diagnostic_snapshot scientific values",
            "washout snapshot scientific values",
            "metric values",
        ],
    }
    return rows, decision


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({str(key) for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_safe(row.get(key)) for key in fieldnames})


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _pilot_outputs(audit: Mapping[str, Any], runs: Sequence[Run]) -> dict[str, Any]:
    rows, decision = pilot_gate_rows(runs)
    audit_path = DERIVED / "e18_pilot_audit.json"
    gate_path = DERIVED / "e18_pilot_gate.csv"
    decision_path = DERIVED / "e18_pilot_gate.json"
    write_json(audit_path, audit)
    write_csv(gate_path, rows)
    write_json(decision_path, decision)
    return {
        "mode": "pilot",
        "audit": str(audit_path),
        "gate_csv": str(gate_path),
        "gate_json": str(decision_path),
        "decision": decision,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "full"), default="full")
    parser.add_argument("--artifacts", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = args.artifacts or (
        DEFAULT_PILOT_ARTIFACTS if args.mode == "pilot" else DEFAULT_ARTIFACTS
    )
    runs, audit = load_runs(artifacts, mode=args.mode)
    if args.mode == "pilot":
        result = _pilot_outputs(audit, runs)
        print(json.dumps(_json_safe(result), indent=2, sort_keys=True))
        return
    raise RuntimeError("E18 full analysis outputs are unavailable until the blinded pilot gate closes")


if __name__ == "__main__":
    main()
