#!/usr/bin/env python3
"""Strict audit and analysis for E16 conditional-support completion.

E16 is an explicitly adaptive, post-hoc intervention on the recurrent E15
signature-113 endpoint.  This analyzer is intentionally fail-closed.  It accepts
exactly the frozen seven support counts by twenty training seeds, verifies the
artifact identity and measurement lattice, reconstructs every dataset with the
current source, requires exact agreement with the archived design and exposure
metrics, proves the within-seed pairing invariants, and requires exact equality
between the ``m=0`` trajectories and matching archived E15 runs.

The training seed is the independent unit.  Examples, repeated presentations,
factorial rows, and checkpoints are measurements within a seed, never
independent replicates.  All intervals therefore resample the twenty paired
training seeds as clusters.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import warnings
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, TypeAlias, cast

os.environ.setdefault("MPLCONFIGDIR", "/tmp/forkworld-e16-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/forkworld-e16-xdg")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import yaml  # type: ignore[import-untyped]
from matplotlib.axes import Axes
from numpy.typing import NDArray

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forkworld.artifacts import implementation_provenance  # noqa: E402
from forkworld.competing import (  # noqa: E402
    CompetingGoalBundle,
    make_competing_bundle,
    make_competing_factorial_dataset,
)
from forkworld.data import SemanticBatch  # noqa: E402
from forkworld.protocols_dynamics import (  # noqa: E402
    q_only_codeword_partition,
    static_subset_exposure,
)

DEFAULT_ARTIFACTS = REPO / "artifacts-e16"
DEFAULT_REFINEMENT_ARTIFACTS = REPO / "artifacts-e16-refinement"
DEFAULT_E15_ARTIFACTS = REPO / "artifacts-e15"
DERIVED = HERE / "derived"
FIGURES = HERE / "figures"

ORIGINAL_SEEDS = (11, 23, 37, 41, 53, 67, 71, 83, 97, 101)
FRESH_SEEDS = (103, 107, 109, 113, 127, 131, 137, 139, 149, 151)
ALL_SEEDS = ORIGINAL_SEEDS + FRESH_SEEDS
SUPPORT_COUNTS = (0, 2, 10, 50, 100, 250, 450)
REFINEMENT_COUNTS = (4, 6, 8)
COMBINED_COUNTS = tuple(sorted((*SUPPORT_COUNTS, *REFINEMENT_COUNTS)))
GOALS = ("P", "Q", "Y")
LAYERS = ("raw", "first_hidden", "final_hidden")
TUPLE_KEYS = ("---", "--+", "-+-", "-++", "+--", "+-+", "++-", "+++")
Q_ONLY_TUPLES = ("-+-", "+-+")

COMPETITION_STEPS = (
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
    161,
    222,
    304,
    418,
    575,
    790,
    1_085,
    1_491,
    2_048,
    2_435,
    2_896,
    3_444,
    4_096,
    4_871,
    5_793,
    6_890,
    8_192,
)
CALIBRATION_STEPS = tuple(step for step in COMPETITION_STEPS if step <= 2_048)
EXPECTED_E15_BRIDGES = len(ALL_SEEDS)
BOOTSTRAP_DRAWS = 4_000
ARTIFACT_SCHEMA_VERSION = 1
SOURCE_FINGERPRINT_SCHEMA_VERSION = 1
NON_SCIENTIFIC_RUN_FIELDS = frozenset({"output_root", "resume", "seeds"})
INFERENCE_STATUS = "adaptive_exploratory_posthoc_paired_support_intervention"

STABLE_GATE_SIGNATURE = 113
TUPLE_CONSISTENCY_THRESHOLD = 0.90
PURE_P_SIGNATURE = 15
PURE_Q_SIGNATURE = 51
PURE_Y_SIGNATURE = 85
REFINEMENT_TOTAL_DROP = 0.30
REFINEMENT_ADJACENT_DROP = 0.20

INK = "#17212B"
GRAY = "#667085"
LIGHT_GRAY = "#E6E9EE"
BLUE = "#2673B8"
ORANGE = "#D55E00"
TEAL = "#009E73"
PURPLE = "#7A5195"
GOLD = "#C99700"
ROSE = "#CC79A7"

MetricKey: TypeAlias = tuple[str, str, str, str]
FullMetricKey: TypeAlias = tuple[str, str, str, str, int]


def get(mapping: Mapping[str, Any], path: str, default: Any = None) -> Any:
    """Read a dot-separated path from a nested mapping."""

    value: Any = mapping
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value


def rounded(value: Any) -> float:
    return round(float(value), 8)


def stable_seed(*parts: Any) -> int:
    digest = hashlib.blake2b(digest_size=8, person=b"forke16")
    for part in parts:
        encoded = str(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
    return int.from_bytes(digest.digest(), "little")


def expected_artifact_run_id(
    config: Mapping[str, Any], metadata: Mapping[str, Any]
) -> str:
    """Reconstruct the immutable RunStore identity."""

    scientific = {
        str(key): value
        for key, value in config.items()
        if key != "seed" and not str(key).startswith("_")
    }
    run = dict(cast(Mapping[str, Any], scientific.get("run", {})))
    for field in NON_SCIENTIFIC_RUN_FIELDS:
        run.pop(field, None)
    scientific["run"] = run
    canonical = json.dumps(scientific, sort_keys=True, separators=(",", ":"), default=str)
    implementation = get(metadata, "implementation", {})
    identity = {
        "config": canonical,
        "seed": int(get(config, "seed")),
        "artifact_schema_version": get(implementation, "artifact_schema_version"),
        "source_fingerprint_schema_version": get(
            implementation, "source_fingerprint_schema_version"
        ),
        "implementation_fingerprint": get(
            implementation, "implementation_fingerprint"
        ),
    }
    raw = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


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


def expected_grid(counts: Sequence[int]) -> set[tuple[int, int]]:
    return {(count, seed) for count in counts for seed in ALL_SEEDS}


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


def _raise_audit(name: str, errors: Sequence[str]) -> None:
    preview = "\n".join(f"  - {error}" for error in errors[:100])
    suffix = "" if len(errors) <= 100 else f"\n  ... and {len(errors) - 100} more"
    raise RuntimeError(f"{name} failed with {len(errors)} issue(s):\n{preview}{suffix}")


def _source_fingerprint_status(
    artifact_fingerprints: Iterable[str], current_fingerprint: str
) -> dict[str, Any]:
    """Classify current-worktree drift without weakening artifact uniformity.

    Artifact fingerprints are immutable provenance for the completed runs and
    must still be uniform.  The current package fingerprint covers every
    ``src/forkworld`` Python file, however, so adding a later experiment can
    change it without changing any implementation used to regenerate E16.
    Exact regeneration, metric, and archived-bridge audits decide whether such
    drift is relevant; merely observing a different whole-tree hash does not.
    """

    fingerprints = sorted(set(artifact_fingerprints))
    if len(fingerprints) != 1:
        raise ValueError(
            f"expected one implementation fingerprint, observed {fingerprints}"
        )
    artifact_fingerprint = fingerprints[0]
    matches = artifact_fingerprint == current_fingerprint
    return {
        "artifact_implementation_fingerprint": artifact_fingerprint,
        "current_source_fingerprint": current_fingerprint,
        "current_source_matches_artifact": matches,
        "current_worktree_drift_detected": not matches,
        "current_worktree_drift_policy": (
            "report_nonfatal_only_after_exact_current_source_regeneration_metric_"
            "and_archived_bridge_audits_pass"
        ),
    }


def _metric_is_needed(stage: str, split: str, metric: str) -> bool:
    if stage in {"P_calibration", "Q_calibration", "Y_calibration"}:
        return split == f"{stage}_iid" and metric == "target_accuracy"
    if stage == "competition_candidates":
        return split in {"competition_iid", "competition_q_wrong"} and metric in {
            "rho_p",
            "rho_q",
            "rho_y_code",
            "target_accuracy",
        }
    if stage == "competition_behavior":
        return split == "factorial_eval" and metric in {
            "rho_p",
            "rho_q",
            "rho_y_code",
            "target_accuracy",
        }
    if stage == "competition_truth_table":
        structural = {
            "tuple_consistency",
            "codeword_consistency",
            "codeword_consistency_row_weighted",
            "nuisance_consistency",
            "n_codeword_groups",
            "boolean_signature_int",
            "sign_inversion_symmetry",
        }
        return split == "factorial_eval" and (
            metric in structural
            or (
                metric.startswith("tuples__")
                and metric.rsplit("__", 1)[-1]
                in {"positive_rate", "mean_positive_probability", "modal_action"}
            )
        )
    if stage == "competition_probe":
        return split == "factorial_probe" and (
            metric
            in {
                "raw_codeword_count",
                "truth_table_control_positive_fraction",
                "n_train",
                "n_heldout",
                "n_labels",
            }
            or "__heldout_accuracy__" in metric
            or metric.endswith("__dimension")
            or metric.endswith("__alpha")
        )
    if stage == "competition_causal":
        return split == "factorial_eval" and metric in {
            "c_hard",
            "c_prob",
            "causal_score",
            "causal_prob_score",
            "hard_abs_change",
            "prob_abs_change",
            "hard_abs_share",
        }
    if stage == "support_completion_design":
        return split == "competition_train"
    if stage == "support_completion_exposure":
        return split == "competition_train" and metric in {
            "cumulative_q_only_presentations",
            "unique_q_only_rows_seen",
            "q_only_training_rows",
            "q_only_presentation_fraction",
        }
    if stage == "support_completion_generalization":
        return split in {
            "factorial_q_only_all",
            "factorial_q_only_seen_codewords",
            "factorial_q_only_unseen_codewords",
        } and metric in {
            "rho_p",
            "rho_q",
            "rho_y_code",
            "target_accuracy",
            "panel_rows",
            "mean_target_probability",
            "mean_target_logit_margin",
        }
    return False


@dataclass(frozen=True)
class MetricPoint:
    value: float
    n: int
    examples_seen: int
    stage_step: int


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
    def m(self) -> int:
        return int(get(self.config, "h13.q_only_error_count"))

    @property
    def cohort(self) -> str:
        return "original" if self.seed in ORIGINAL_SEEDS else "fresh"

    def series(
        self, stage: str, split: str, metric: str, intervention: str = "none"
    ) -> Mapping[int, MetricPoint]:
        key = (stage, split, intervention, metric)
        if key not in self.metrics:
            raise RuntimeError(f"{self.path.name}: missing metric series {key}")
        return self.metrics[key]

    def point(
        self,
        stage: str,
        split: str,
        metric: str,
        step: int,
        intervention: str = "none",
    ) -> MetricPoint:
        series = self.series(stage, split, metric, intervention)
        if step not in series:
            raise RuntimeError(
                f"{self.path.name}: missing {stage}/{split}/{intervention}/{metric} at {step}"
            )
        return series[step]

    def value(
        self,
        stage: str,
        split: str,
        metric: str,
        step: int,
        intervention: str = "none",
    ) -> float:
        return float(self.point(stage, split, metric, step, intervention).value)


def _load_metric_index(
    path: Path, run_id: str, seed: int, errors: list[str]
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
                errors.append(f"{run_id}: metric line {line_number} run_id mismatch")
            try:
                record_seed = int(record.get("seed", -1))
            except (TypeError, ValueError):
                record_seed = -1
            if record_seed != seed:
                errors.append(f"{run_id}: metric line {line_number} seed mismatch")
            if record.get("experiment") != "h13":
                errors.append(f"{run_id}: metric line {line_number} experiment mismatch")
            if record.get("condition") != "adaptive_support_completion":
                errors.append(f"{run_id}: metric line {line_number} condition mismatch")
            stage = str(record.get("stage"))
            split = str(record.get("split"))
            intervention = str(record.get("intervention", "none"))
            metric = str(record.get("metric"))
            if not _metric_is_needed(stage, split, metric):
                continue
            try:
                step = int(record["global_step"])
                stage_step = int(record["stage_step"])
                value = float(record["value"])
                n = int(record["n"])
                examples_seen = int(record["examples_seen"])
            except (KeyError, TypeError, ValueError) as error:
                errors.append(
                    f"{run_id}: malformed selected metric line {line_number}: {error}"
                )
                continue
            if step != stage_step:
                errors.append(
                    f"{run_id}: selected metric line {line_number} has stage/global mismatch"
                )
            if not math.isfinite(value):
                errors.append(f"{run_id}: non-finite selected metric at line {line_number}")
            if stage.startswith("competition_") or stage.startswith("support_completion_"):
                if step not in COMPETITION_STEPS:
                    errors.append(f"{run_id}: unexpected competition checkpoint {step}")
            elif stage.endswith("_calibration") and step not in CALIBRATION_STEPS:
                errors.append(f"{run_id}: unexpected calibration checkpoint {step}")
            key = (stage, split, intervention, metric)
            if step in result[key]:
                errors.append(f"{run_id}: duplicate selected metric {key} at step {step}")
            result[key][step] = MetricPoint(value, n, examples_seen, stage_step)
    return dict(result), line_count


def _require_steps(
    run: Run,
    errors: list[str],
    key: MetricKey,
    expected: Sequence[int],
) -> None:
    actual = set(run.metrics.get(key, {}))
    wanted = set(expected)
    if actual != wanted:
        errors.append(
            f"{run.path.name}: {key} has {len(wanted - actual)} missing and "
            f"{len(actual - wanted)} unexpected checkpoints"
        )


def _check_series_metadata(
    run: Run,
    errors: list[str],
    key: MetricKey,
    *,
    expected_n: int | None,
    calibration: bool = False,
) -> None:
    for step, point in run.metrics.get(key, {}).items():
        wanted_examples = step * 250
        if point.examples_seen != wanted_examples:
            errors.append(
                f"{run.path.name}: {key} at {step} examples_seen={point.examples_seen}, "
                f"expected {wanted_examples}"
            )
        if expected_n is not None and point.n != expected_n:
            errors.append(
                f"{run.path.name}: {key} at {step} n={point.n}, expected {expected_n}"
            )
        if calibration and step not in CALIBRATION_STEPS:
            errors.append(f"{run.path.name}: {key} has non-calibration step {step}")


def validate_run(run: Run, errors: list[str]) -> None:
    run_id = run.path.name
    fixed = {
        "schema_version": 1,
        "experiment.hypothesis": "h13",
        "experiment.name": "conditional_support_completion",
        "experiment.mode": "adaptive_support_completion",
        "run.device": "cpu",
        "run.task_levels": ["choice"],
        "run.save_checkpoints": False,
        "evaluation.save_predictions": False,
        "evaluation.bootstrap_samples": BOOTSTRAP_DRAWS,
        "evaluation.persistence": 2,
        "data.n_train": 10_000,
        "data.n_validation": 4_000,
        "data.n_eval": 10_000,
        "data.max_k": 5,
        "data.state_dim": 8,
        "model.width": 64,
        "model.depth": 2,
        "model.activation": "relu",
        "model.residual": False,
        "update.mode": "full",
        "update.budget": "full",
        "train.algorithm": "clean_sft",
        "train.steps": 8_192,
        "train.batch_size": 250,
        "train.learning_rate": 0.003,
        "train.weight_decay": 0.0,
        "h13.q_p": 0.90,
        "h13.q_q": 0.95,
        "h13.k_q": 2,
        "h13.k_y": 5,
        "h13.max_k_q": 3,
        "h13.max_k_y": 5,
        "h13.error_structure": "nested",
        "h13.calibration_steps": 2_048,
        "h13.competition_steps": 8_192,
        "h13.probe_train_n": 2_048,
        "h13.probe_eval_n": 4_096,
        "h13.probe_ridge": 0.001,
        "h13.truth_table_control_seed": 1_500_450_271,
        "h13.bridge_step": 2_048,
    }
    for path, expected in fixed.items():
        expect_equal(errors, run_id, get(run.config, path), expected, path)
    expect_equal(
        errors,
        run_id,
        get(run.config, "train.eval_steps"),
        list(COMPETITION_STEPS[1:]),
        "train.eval_steps",
    )
    expect_equal(errors, run_id, int(get(run.config, "seed")), run.seed, "config seed")
    summary_fixed = {
        "hypothesis": "h13",
        "condition": "adaptive_support_completion",
        "design_status": "adaptive_posthoc_support_completion",
        "data.n_train": 10_000,
        "data.n_validation": 4_000,
        "data.n_eval_per_legacy_panel": 10_000,
        "data.probe_train_n": 2_048,
        "data.probe_eval_n": 4_096,
        "data.q_p": 0.90,
        "data.q_q": 0.95,
        "data.k_q": 2,
        "data.k_y": 5,
        "data.max_k_q": 3,
        "data.max_k_y": 5,
        "data.state_dim": 8,
        "data.error_structure": "nested",
        "data.support_intervention_scope": "training_only",
        "data.calibration_interface_matches": True,
        "data.factorial_interface_matches": True,
        "data.probe_splits_disjoint": True,
        "data.truth_table_control_seed": 1_500_450_271,
        "measurement.ridge_alpha": 0.001,
        "measurement.probe_standardization": "fit_split_only",
        "measurement.probe_layers": ["raw", "first_hidden", "final_hidden"],
        "measurement.probe_controls": ["permuted_labels", "orthogonal_truth_table"],
        "measurement.checkpoint_count_including_initialization": 33,
        "training.calibration_steps_each": 2_048,
        "training.competition_steps": 8_192,
        "training.examples_seen": 3_584_000,
        "bridge.step": 2_048,
        "bridge.available": True,
        "bridge.validation_rule": "descriptive temporal landmark; no archived h13 state",
    }
    for path, expected in summary_fixed.items():
        expect_equal(errors, run_id, get(run.summary, path), expected, f"summary.{path}")
    expect_equal(errors, run_id, int(get(run.summary, "data.q_only_error_count")), run.m, "summary m")
    expect_equal(
        errors,
        run_id,
        run.summary.get("seed_cohort"),
        "e15_bridge_seed" if run.seed in ORIGINAL_SEEDS else "e15_fresh_seed",
        "summary seed cohort",
    )
    expect_equal(
        errors,
        run_id,
        get(run.summary, "data.e15_reference.nested_exact_data_replay_candidate"),
        run.m == 0,
        "m0 E15 replay flag",
    )
    expect_equal(
        errors,
        run_id,
        get(run.summary, "data.e15_reference.independent_cell_count_match"),
        run.m == 450,
        "independence count flag",
    )
    expect_equal(
        errors,
        run_id,
        get(run.summary, "data.e15_reference.independent_row_allocation_match_claimed"),
        False,
        "independent row allocation claim",
    )
    expect_equal(
        errors,
        run_id,
        get(run.summary, "data.factorial_candidate_counts_eval"),
        {str(index): 512 for index in range(8)},
        "factorial eval counts",
    )
    expect_equal(
        errors,
        run_id,
        get(run.summary, "data.factorial_candidate_counts_train"),
        {str(index): 256 for index in range(8)},
        "factorial train counts",
    )
    if get(run.summary, "initial") is None or get(run.summary, "final.dynamics") is None:
        errors.append(f"{run_id}: missing initial or final dynamics snapshot")
    if get(run.summary, "bridge.dynamics") is None:
        errors.append(f"{run_id}: missing bridge dynamics snapshot")

    for section in ("calibration", "competition"):
        reports = get(run.summary, f"model.{section}")
        if section == "calibration":
            if not isinstance(reports, Mapping) or set(reports) != set(GOALS):
                errors.append(f"{run_id}: malformed calibration model reports")
                continue
            reports_iter: Iterable[Any] = reports.values()
        else:
            reports_iter = (reports,)
        for report in reports_iter:
            if not isinstance(report, Mapping):
                errors.append(f"{run_id}: malformed {section} model report")
                continue
            for name, expected in {
                "input_dim": 19,
                "total_parameters": 5_505,
                "trainable_parameters": 5_505,
                "update_mode": "full",
                "requested_budget": "full",
            }.items():
                if report.get(name) != expected:
                    errors.append(f"{run_id}: {section} model {name}={report.get(name)!r}")

    for goal in GOALS:
        key = (f"{goal}_calibration", f"{goal}_calibration_iid", "none", "target_accuracy")
        _require_steps(run, errors, key, CALIBRATION_STEPS)
        _check_series_metadata(run, errors, key, expected_n=4_000, calibration=True)
    for split, expected_n in (("competition_iid", 4_000), ("competition_q_wrong", 10_000)):
        for metric in ("rho_p", "rho_q", "rho_y_code", "target_accuracy"):
            key = ("competition_candidates", split, "none", metric)
            _require_steps(run, errors, key, COMPETITION_STEPS)
            _check_series_metadata(run, errors, key, expected_n=expected_n)
    for metric in ("rho_p", "rho_q", "rho_y_code", "target_accuracy"):
        key = ("competition_behavior", "factorial_eval", "none", metric)
        _require_steps(run, errors, key, COMPETITION_STEPS)
        _check_series_metadata(run, errors, key, expected_n=4_096)
    for metric in (
        "tuple_consistency",
        "codeword_consistency",
        "codeword_consistency_row_weighted",
        "nuisance_consistency",
        "n_codeword_groups",
        "boolean_signature_int",
        "sign_inversion_symmetry",
    ):
        key = ("competition_truth_table", "factorial_eval", "none", metric)
        _require_steps(run, errors, key, COMPETITION_STEPS)
        _check_series_metadata(run, errors, key, expected_n=4_096)
    for tuple_key in TUPLE_KEYS:
        for suffix in ("positive_rate", "mean_positive_probability", "modal_action"):
            key = (
                "competition_truth_table",
                "factorial_eval",
                "none",
                f"tuples__{tuple_key}__{suffix}",
            )
            _require_steps(run, errors, key, COMPETITION_STEPS)
            _check_series_metadata(run, errors, key, expected_n=4_096)

    probe_labels = (
        "P",
        "Q",
        "Y",
        "truth_table_control",
        "P_permuted",
        "Q_permuted",
        "Y_permuted",
        "truth_table_control_permuted",
    )
    for layer in LAYERS:
        for label in probe_labels:
            key = (
                "competition_probe",
                "factorial_probe",
                "none",
                f"representations__{layer}__heldout_accuracy__{label}",
            )
            _require_steps(run, errors, key, COMPETITION_STEPS)
            _check_series_metadata(run, errors, key, expected_n=4_096)
    for metric, expected in {
        "raw_codeword_count": 256,
        "truth_table_control_positive_fraction": 0.5,
        "n_train": 2_048,
        "n_heldout": 4_096,
        "n_labels": 8,
    }.items():
        key = ("competition_probe", "factorial_probe", "none", metric)
        _require_steps(run, errors, key, COMPETITION_STEPS)
        _check_series_metadata(run, errors, key, expected_n=4_096)
        if any(point.value != expected for point in run.metrics.get(key, {}).values()):
            errors.append(f"{run_id}: probe invariant {metric} differs from {expected}")

    for goal in GOALS:
        for metric in (
            "c_hard",
            "c_prob",
            "causal_score",
            "causal_prob_score",
            "hard_abs_change",
            "prob_abs_change",
            "hard_abs_share",
        ):
            key = ("competition_causal", "factorial_eval", goal, metric)
            _require_steps(run, errors, key, COMPETITION_STEPS)
            _check_series_metadata(run, errors, key, expected_n=4_096)

    design_metrics = (
        "p_error_count",
        "q_error_count",
        "both_error_count",
        "p_only_error_count",
        "q_only_error_count",
        "neither_error_count",
        "both_error_rate",
        "independence_expected_count",
        "overlap_excess_count",
        "error_phi",
        "requested_q_only_error_count",
        "q_only_seen_active_codewords",
        "q_only_possible_active_codewords",
        "q_only_active_codeword_coverage",
    )
    for metric in design_metrics:
        key = ("support_completion_design", "competition_train", "none", metric)
        _require_steps(run, errors, key, (0,))
        _check_series_metadata(run, errors, key, expected_n=10_000)
    for metric in (
        "cumulative_q_only_presentations",
        "unique_q_only_rows_seen",
        "q_only_training_rows",
        "q_only_presentation_fraction",
    ):
        key = ("support_completion_exposure", "competition_train", "none", metric)
        _require_steps(run, errors, key, COMPETITION_STEPS)
        _check_series_metadata(run, errors, key, expected_n=10_000)

    general_metrics = (
        "rho_p",
        "rho_q",
        "rho_y_code",
        "target_accuracy",
        "panel_rows",
        "mean_target_probability",
        "mean_target_logit_margin",
    )
    for panel in ("all", "seen_codewords", "unseen_codewords"):
        split = f"factorial_q_only_{panel}"
        panel_rows_key = (
            "support_completion_generalization",
            split,
            "none",
            "panel_rows",
        )
        _require_steps(run, errors, panel_rows_key, COMPETITION_STEPS)
        rows_series = run.metrics.get(panel_rows_key, {})
        if not rows_series:
            continue
        panel_n = round(next(iter(rows_series.values())).value)
        if any(round(point.value) != panel_n or point.n != panel_n for point in rows_series.values()):
            errors.append(f"{run_id}: {split} panel size is not constant or does not match n")
        for metric in general_metrics:
            key = ("support_completion_generalization", split, "none", metric)
            expected_steps: Sequence[int] = COMPETITION_STEPS if panel_n > 0 or metric == "panel_rows" else ()
            _require_steps(run, errors, key, expected_steps)
            _check_series_metadata(run, errors, key, expected_n=panel_n)


def load_runs(
    root: Path,
    *,
    counts: Sequence[int] = SUPPORT_COUNTS,
    panel: str = "primary",
) -> tuple[list[Run], dict[str, Any]]:
    directory = root / "h13" / "conditional_support_completion"
    expected_runs = len(counts) * len(ALL_SEEDS)
    if not directory.is_dir():
        raise RuntimeError(
            f"E16 {panel} hard audit failed: 0/{expected_runs} materialized runs; "
            f"missing artifact directory {directory}"
        )
    config_paths = sorted(directory.glob("*/resolved_config.yaml"))
    materialized = {path.parent for path in config_paths}
    if len(materialized) != expected_runs:
        raise RuntimeError(
            f"E16 {panel} hard audit failed: {len(materialized)}/{expected_runs} materialized runs"
        )

    errors: list[str] = []
    runs: list[Run] = []
    states: Counter[str] = Counter()
    for config_path in config_paths:
        run_dir = config_path.parent
        required = ("COMPLETE", "summary.json", "metadata.json", "status.json", "metrics.jsonl")
        missing = [name for name in required if not (run_dir / name).is_file()]
        if missing:
            errors.append(f"{run_dir.name}: missing {', '.join(missing)}")
            continue
        marker = (run_dir / "COMPLETE").read_text(encoding="utf-8")
        if marker != "complete\n":
            errors.append(f"{run_dir.name}: invalid COMPLETE marker {marker!r}")
        status = load_json(run_dir / "status.json")
        states[str(status.get("state"))] += 1
        if status.get("state") != "complete" or status.get("run_id") != run_dir.name:
            errors.append(f"{run_dir.name}: invalid completion status")
        config = load_yaml(config_path)
        summary = load_json(run_dir / "summary.json")
        metadata = load_json(run_dir / "metadata.json")
        try:
            seed = int(summary.get("seed", -1))
        except (TypeError, ValueError):
            seed = -1
        if int(get(config, "seed", -2)) != seed:
            errors.append(f"{run_dir.name}: config/summary seed mismatch")
        if metadata.get("run_id") != run_dir.name or int(metadata.get("seed", -2)) != seed:
            errors.append(f"{run_dir.name}: metadata identity mismatch")
        if expected_artifact_run_id(config, metadata) != run_dir.name:
            errors.append(f"{run_dir.name}: artifact identity does not reconstruct")
        metric_index, line_count = _load_metric_index(
            run_dir / "metrics.jsonl", run_dir.name, seed, errors
        )
        run = Run(run_dir, config, summary, metadata, metric_index, line_count)
        validate_run(run, errors)
        coverage_count = int(
            get(summary, "data.q_only_codeword_coverage.seen_active_codewords", -1)
        )
        expected_lines = 13_077 if coverage_count in {0, 64} else 13_275
        if line_count != expected_lines:
            errors.append(
                f"{run_dir.name}: metric line count={line_count}, expected {expected_lines} "
                f"for seen-codeword count {coverage_count}"
            )
        runs.append(run)

    orphan_artifacts = {
        path.parent
        for pattern in ("*/COMPLETE", "*/summary.json", "*/status.json", "*/metrics.jsonl")
        for path in directory.glob(pattern)
        if path.parent not in materialized
    }
    if orphan_artifacts:
        errors.append(f"{len(orphan_artifacts)} artifact directories lack resolved configurations")
    keys = [(run.m, run.seed) for run in runs]
    duplicates = [key for key, count in Counter(keys).items() if count > 1]
    expected_keys = expected_grid(counts)
    unexpected = set(keys) - expected_keys
    missing_keys = expected_keys - set(keys)
    if duplicates or unexpected or missing_keys:
        errors.append(
            f"grid has {len(duplicates)} duplicates, {len(unexpected)} unexpected, "
            f"and {len(missing_keys)} missing keys"
        )
    cell_counts = Counter(run.m for run in runs)
    if set(cell_counts) != set(counts) or any(
        cell_counts[count] != len(ALL_SEEDS) for count in counts
    ):
        errors.append(f"invalid seven-cell replication counts: {dict(cell_counts)}")

    fingerprints_raw = [
        get(run.metadata, "implementation.implementation_fingerprint") for run in runs
    ]
    fingerprints = {value for value in fingerprints_raw if isinstance(value, str)}
    current = implementation_provenance(REPO)
    current_fingerprint = str(current["implementation_fingerprint"])
    try:
        fingerprint_status = _source_fingerprint_status(
            fingerprints, current_fingerprint
        )
    except ValueError as error:
        errors.append(str(error))
        fingerprint_status = {}
    for run, fingerprint in zip(runs, fingerprints_raw, strict=True):
        if (
            not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in fingerprint)
        ):
            errors.append(f"{run.path.name}: malformed implementation fingerprint")
        if get(run.metadata, "implementation.artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
            errors.append(f"{run.path.name}: wrong artifact schema version")
        if (
            get(run.metadata, "implementation.source_fingerprint_schema_version")
            != SOURCE_FINGERPRINT_SCHEMA_VERSION
        ):
            errors.append(f"{run.path.name}: wrong source-fingerprint schema version")
    if errors:
        _raise_audit(f"E16 {panel} hard audit", errors)
    total_metric_records = sum(run.metric_line_count for run in runs)
    expected_metric_records = sum(
        13_077
        if int(get(run.summary, "data.q_only_codeword_coverage.seen_active_codewords"))
        in {0, 64}
        else 13_275
        for run in runs
    )
    if total_metric_records != expected_metric_records:
        raise RuntimeError(
            f"E16 {panel} hard audit failed: metric record total={total_metric_records:,}, "
            f"expected {expected_metric_records:,}"
        )
    audit = {
        "artifacts_root": str(root.resolve()),
        "materialized_runs": len(materialized),
        "complete_runs": states.get("complete", 0),
        "expected_runs": expected_runs,
        "cells": len(cell_counts),
        "panel": panel,
        "support_counts": list(counts),
        "seeds": list(ALL_SEEDS),
        "implementation_fingerprints": sorted(str(value) for value in fingerprints),
        **fingerprint_status,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "source_fingerprint_schema_version": SOURCE_FINGERPRINT_SCHEMA_VERSION,
        "metric_records": total_metric_records,
        "strict_grid_and_measurement_audit_passed": True,
    }
    return sorted(runs, key=lambda run: (run.seed, run.m)), audit


@dataclass(frozen=True)
class CanonicalMetric:
    value: float
    n: int
    examples_seen: int
    stage_step: int
    level: str


def _load_canonical_metrics(
    path: Path,
    *,
    expected_experiment: str,
    expected_run_id: str,
    expected_seed: int,
) -> dict[FullMetricKey, CanonicalMetric]:
    result: dict[FullMetricKey, CanonicalMetric] = {}
    errors: list[str] = []
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as error:
        raise RuntimeError(f"cannot read bridge metrics {path}: {error}") from error
    with handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                errors.append(f"line {line_number}: malformed JSON: {error}")
                continue
            if not isinstance(record, Mapping):
                errors.append(f"line {line_number}: record is not an object")
                continue
            if record.get("experiment") != expected_experiment:
                errors.append(f"line {line_number}: experiment mismatch")
            if record.get("run_id") != expected_run_id:
                errors.append(f"line {line_number}: run_id mismatch")
            try:
                seed = int(record.get("seed", -1))
                step = int(record["global_step"])
                point = CanonicalMetric(
                    value=float(record["value"]),
                    n=int(record["n"]),
                    examples_seen=int(record["examples_seen"]),
                    stage_step=int(record["stage_step"]),
                    level=str(record.get("level", "choice")),
                )
            except (KeyError, TypeError, ValueError) as error:
                errors.append(f"line {line_number}: malformed record: {error}")
                continue
            if seed != expected_seed:
                errors.append(f"line {line_number}: seed mismatch")
            if not math.isfinite(point.value):
                errors.append(f"line {line_number}: non-finite value")
            key = (
                str(record.get("stage")),
                str(record.get("split")),
                str(record.get("intervention", "none")),
                str(record.get("metric")),
                step,
            )
            if key in result:
                errors.append(f"line {line_number}: duplicate canonical metric {key}")
            result[key] = point
    if errors:
        _raise_audit(f"bridge metric audit for {path.parent.name}", errors)
    return result


def _canonical_value(
    metrics: Mapping[FullMetricKey, CanonicalMetric],
    stage: str,
    split: str,
    metric: str,
    step: int,
    intervention: str = "none",
) -> float:
    key = (stage, split, intervention, metric, step)
    if key not in metrics:
        raise RuntimeError(f"missing canonical bridge metric {key}")
    return float(metrics[key].value)


E15_CAUSAL_RENAMES = {
    "c_hard_l1_share": "c_hard_signed_l1_allocation",
    "c_prob_l1_share": "c_prob_signed_l1_allocation",
}
H13_ONLY_CAUSAL_METRICS = frozenset({"c_hard_abs_l1_share", "c_prob_abs_l1_share"})


def _canonicalize_bridge_metric_map(
    metrics: Mapping[FullMetricKey, CanonicalMetric],
    *,
    reference: bool,
) -> dict[FullMetricKey, CanonicalMetric]:
    result: dict[FullMetricKey, CanonicalMetric] = {}
    for key, point in metrics.items():
        stage, split, intervention, metric, step = key
        if stage.startswith("support_completion_"):
            continue
        if metric in H13_ONLY_CAUSAL_METRICS:
            continue
        canonical_metric = E15_CAUSAL_RENAMES.get(metric, metric) if reference else metric
        canonical_key = (stage, split, intervention, canonical_metric, step)
        if canonical_key in result:
            raise RuntimeError(f"bridge canonicalization produced duplicate {canonical_key}")
        result[canonical_key] = point
    return result


def _canonicalize_bridge_summary(value: Any) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            if key in H13_ONLY_CAUSAL_METRICS:
                continue
            result[E15_CAUSAL_RENAMES.get(key, key)] = _canonicalize_bridge_summary(item)
        return result
    if isinstance(value, list):
        return [_canonicalize_bridge_summary(item) for item in value]
    return value


def _e15_reference_key(config: Mapping[str, Any]) -> tuple[str, int] | None:
    if rounded(get(config, "h12.q_p")) != 0.90:
        return None
    if rounded(get(config, "h12.q_q")) != 0.95:
        return None
    if int(get(config, "h12.k_q")) != 2 or int(get(config, "h12.k_y")) != 5:
        return None
    if int(get(config, "model.width")) != 64 or int(get(config, "model.depth")) != 2:
        return None
    overlap = str(get(config, "h12.error_structure"))
    seed = int(get(config, "seed"))
    if overlap not in {"nested", "independent"} or seed not in ALL_SEEDS:
        return None
    return overlap, seed


def e15_bridge_audit(
    runs: Sequence[Run], e15_root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Require exact m=0 replay and audit the count-matched m=450 endpoint."""

    directory = e15_root / "h12" / "multigoal_temporal_bridge"
    if not directory.is_dir():
        raise RuntimeError(f"missing archived E15 artifact directory: {directory}")
    expected = {(overlap, seed) for overlap in ("nested", "independent") for seed in ALL_SEEDS}
    references: dict[
        tuple[str, int],
        tuple[Path, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]],
    ] = {}
    fingerprints: set[str] = set()
    for config_path in sorted(directory.glob("*/resolved_config.yaml")):
        config = load_yaml(config_path)
        key = _e15_reference_key(config)
        if key is None:
            continue
        run_dir = config_path.parent
        for name in ("COMPLETE", "summary.json", "metadata.json", "status.json", "metrics.jsonl"):
            if not (run_dir / name).is_file():
                raise RuntimeError(f"E15 bridge reference {run_dir.name} lacks {name}")
        if (run_dir / "COMPLETE").read_text(encoding="utf-8") != "complete\n":
            raise RuntimeError(f"E15 bridge reference {run_dir.name} has invalid marker")
        summary = load_json(run_dir / "summary.json")
        metadata = load_json(run_dir / "metadata.json")
        status = load_json(run_dir / "status.json")
        if status.get("state") != "complete" or status.get("run_id") != run_dir.name:
            raise RuntimeError(f"E15 bridge reference {run_dir.name} has invalid status")
        if metadata.get("run_id") != run_dir.name or int(metadata.get("seed", -1)) != key[1]:
            raise RuntimeError(f"E15 bridge reference {run_dir.name} has invalid metadata")
        if expected_artifact_run_id(config, metadata) != run_dir.name:
            raise RuntimeError(f"E15 bridge reference {run_dir.name} identity does not reconstruct")
        if summary.get("hypothesis") != "h12" or int(summary.get("seed", -1)) != key[1]:
            raise RuntimeError(f"E15 bridge reference {run_dir.name} summary identity mismatch")
        fingerprint = get(metadata, "implementation.implementation_fingerprint")
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise RuntimeError(f"E15 bridge reference {run_dir.name} malformed fingerprint")
        fingerprints.add(fingerprint)
        if key in references:
            raise RuntimeError(f"duplicate E15 bridge reference for {key}")
        references[key] = (run_dir, config, summary, metadata)
    if set(references) != expected:
        raise RuntimeError(
            f"E15 bridge reference grid mismatch: {len(references)}/{len(expected)}"
        )

    run_index = {(run.m, run.seed): run for run in runs}
    exact_metric_checks = 0
    exact_summary_checks: Counter[str] = Counter()
    reference_metric_records = 0
    independent_rows: list[dict[str, Any]] = []
    for seed in ALL_SEEDS:
        run = run_index[(0, seed)]
        reference_dir, _, reference_summary, _ = references[("nested", seed)]
        actual_metrics = _load_canonical_metrics(
            run.path / "metrics.jsonl",
            expected_experiment="h13",
            expected_run_id=run.path.name,
            expected_seed=seed,
        )
        reference_metrics = _load_canonical_metrics(
            reference_dir / "metrics.jsonl",
            expected_experiment="h12",
            expected_run_id=reference_dir.name,
            expected_seed=seed,
        )
        shared_actual = _canonicalize_bridge_metric_map(actual_metrics, reference=False)
        canonical_reference = _canonicalize_bridge_metric_map(
            reference_metrics, reference=True
        )
        if shared_actual != canonical_reference:
            missing = set(canonical_reference) - set(shared_actual)
            extra = set(shared_actual) - set(canonical_reference)
            differing = {
                key
                for key in set(shared_actual) & set(canonical_reference)
                if shared_actual[key] != canonical_reference[key]
            }
            raise RuntimeError(
                f"E16/E15 exact metric bridge failed for seed {seed}: "
                f"{len(missing)} missing, {len(extra)} extra, {len(differing)} unequal"
            )
        exact_metric_checks += 1
        reference_metric_records += len(canonical_reference)
        comparisons = {
            "initial_dynamics": (get(run.summary, "initial"), get(reference_summary, "initial")),
            "step_2048_dynamics": (
                get(run.summary, "bridge.dynamics"),
                get(reference_summary, "bridge.dynamics"),
            ),
            "final_dynamics": (
                get(run.summary, "final.dynamics"),
                get(reference_summary, "final.dynamics"),
            ),
            "standalone_calibrations": (
                get(run.summary, "final.calibration"),
                get(reference_summary, "final.calibration"),
            ),
            "model_interface_reports": (get(run.summary, "model"), get(reference_summary, "model")),
            "training_error_overlap": (
                get(run.summary, "data.training_overlap"),
                get(reference_summary, "data.training_overlap"),
            ),
        }
        for name, (actual, wanted) in comparisons.items():
            if _canonicalize_bridge_summary(actual) != _canonicalize_bridge_summary(wanted):
                raise RuntimeError(f"E16/E15 exact summary bridge failed for seed {seed}: {name}")
            exact_summary_checks[name] += 1

        independent_dir, _, _, _ = references[("independent", seed)]
        independent_metrics = _load_canonical_metrics(
            independent_dir / "metrics.jsonl",
            expected_experiment="h12",
            expected_run_id=independent_dir.name,
            expected_seed=seed,
        )
        e16 = run_index[(450, seed)]
        e15_signature = round(
            _canonical_value(
                independent_metrics,
                "competition_truth_table",
                "factorial_eval",
                "boolean_signature_int",
                8_192,
            )
        )
        e15_consistency = _canonical_value(
            independent_metrics,
            "competition_truth_table",
            "factorial_eval",
            "tuple_consistency",
            8_192,
        )
        e15_q_probability = 0.5 * (
            1.0
            - _canonical_value(
                independent_metrics,
                "competition_truth_table",
                "factorial_eval",
                "tuples__-+-__mean_positive_probability",
                8_192,
            )
            + _canonical_value(
                independent_metrics,
                "competition_truth_table",
                "factorial_eval",
                "tuples__+-+__mean_positive_probability",
                8_192,
            )
        )
        independent_rows.append(
            {
                "seed": seed,
                "cohort": "original" if seed in ORIGINAL_SEEDS else "fresh",
                "e16_m450_stable_gate": int(
                    round(
                        e16.value(
                            "competition_truth_table",
                            "factorial_eval",
                            "boolean_signature_int",
                            8_192,
                        )
                    )
                    == STABLE_GATE_SIGNATURE
                    and e16.value(
                        "competition_truth_table",
                        "factorial_eval",
                        "tuple_consistency",
                        8_192,
                    )
                    >= TUPLE_CONSISTENCY_THRESHOLD
                ),
                "e15_independent_stable_gate": int(
                    e15_signature == STABLE_GATE_SIGNATURE
                    and e15_consistency >= TUPLE_CONSISTENCY_THRESHOLD
                ),
                "e16_m450_q_only_diagnostic_accuracy": e16.value(
                    "competition_candidates",
                    "competition_q_wrong",
                    "target_accuracy",
                    8_192,
                ),
                "e15_independent_q_only_diagnostic_accuracy": _canonical_value(
                    independent_metrics,
                    "competition_candidates",
                    "competition_q_wrong",
                    "target_accuracy",
                    8_192,
                ),
                "e16_m450_q_only_target_probability": e16.value(
                    "support_completion_generalization",
                    "factorial_q_only_all",
                    "mean_target_probability",
                    8_192,
                ),
                "e15_independent_q_only_target_probability": e15_q_probability,
            }
        )

    if exact_metric_checks != EXPECTED_E15_BRIDGES or any(
        count != EXPECTED_E15_BRIDGES for count in exact_summary_checks.values()
    ):
        raise RuntimeError(
            f"E16/E15 bridge count mismatch: metrics={exact_metric_checks}, summaries={exact_summary_checks}"
        )
    for row in independent_rows:
        row["stable_gate_difference"] = (
            int(row["e16_m450_stable_gate"]) - int(row["e15_independent_stable_gate"])
        )
        row["q_only_diagnostic_accuracy_difference"] = float(
            row["e16_m450_q_only_diagnostic_accuracy"]
        ) - float(row["e15_independent_q_only_diagnostic_accuracy"])
        row["q_only_target_probability_difference"] = float(
            row["e16_m450_q_only_target_probability"]
        ) - float(row["e15_independent_q_only_target_probability"])
    bridge = {
        "e15_artifacts_root": str(e15_root.resolve()),
        "reference_runs": len(references),
        "m0_exact_metric_bridges": exact_metric_checks,
        "m0_exact_shared_metric_records": reference_metric_records,
        "m0_summary_sections_compared_exactly": dict(exact_summary_checks),
        "e15_reference_implementation_fingerprints": sorted(fingerprints),
        "m0_matches_archived_e15_nested_exactly": True,
        "m450_independent_comparison_runs": len(independent_rows),
        "m450_matches_independent_cell_counts_only": True,
        "m450_row_allocation_equality_claimed": False,
        "state_hash_claimed": False,
    }
    return bridge, independent_rows


def _batch_arrays(batch: SemanticBatch) -> dict[str, NDArray[Any]]:
    arrays: dict[str, NDArray[Any]] = {
        "core.y": np.asarray(batch.y),
        "core.target": np.asarray(batch.target),
        "core.reward": np.asarray(batch.reward),
    }
    for name in ("state", "sample_id", "state_id", "episode_id", "step_id"):
        value = getattr(batch, name)
        if value is not None:
            arrays[f"core.{name}"] = np.asarray(value)
    arrays.update(
        {f"channel.{name}": np.asarray(value) for name, value in sorted(batch.channels.items())}
    )
    arrays.update(
        {f"latent.{name}": np.asarray(value) for name, value in sorted(batch.latents.items())}
    )
    return arrays


def _batch_digest(batch: SemanticBatch) -> str:
    digest = hashlib.sha256()
    for name, array in sorted(_batch_arrays(batch).items()):
        contiguous = np.ascontiguousarray(array)
        encoded_name = name.encode("utf-8")
        encoded_dtype = str(contiguous.dtype).encode("ascii")
        digest.update(len(encoded_name).to_bytes(4, "big"))
        digest.update(encoded_name)
        digest.update(len(encoded_dtype).to_bytes(4, "big"))
        digest.update(encoded_dtype)
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _compare_batch_arrays(
    errors: list[str],
    label: str,
    actual: SemanticBatch,
    expected: SemanticBatch,
    *,
    allowed_differences: frozenset[str] = frozenset(),
) -> None:
    actual_arrays = _batch_arrays(actual)
    expected_arrays = _batch_arrays(expected)
    if set(actual_arrays) != set(expected_arrays):
        errors.append(
            f"{label}: array keys differ: "
            f"{sorted(set(actual_arrays) ^ set(expected_arrays))}"
        )
        return
    for name in sorted(actual_arrays):
        equal = np.array_equal(actual_arrays[name], expected_arrays[name])
        if name in allowed_differences:
            continue
        if not equal:
            errors.append(f"{label}: unexpected array difference in {name}")


def _metric_matches(
    errors: list[str],
    run: Run,
    stage: str,
    split: str,
    metric: str,
    step: int,
    expected: float,
    *,
    atol: float = 0.0,
) -> None:
    actual = run.value(stage, split, metric, step)
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=atol):
        errors.append(
            f"{run.path.name}: {stage}/{split}/{metric} at {step}={actual}, expected {expected}"
        )


def _make_bundle(seed: int, m: int | None) -> CompetingGoalBundle:
    return make_competing_bundle(
        n_train=10_000,
        n_iid=4_000,
        n_diagnostic=10_000,
        q_p=0.90,
        q_q=0.95,
        k_q=2,
        k_y=5,
        seed=seed,
        max_k_q=3,
        max_k_y=5,
        overlap="nested",
        q_only_error_count=m,
        state_dim=8,
    )


def dataset_pairing_audit(
    runs: Sequence[Run],
    *,
    counts: Sequence[int] = SUPPORT_COUNTS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Regenerate every arm and prove the registered row-level intervention."""

    run_index = {(run.seed, run.m): run for run in runs}
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    exact_array_checks = 0
    exact_exposure_checks = 0
    exact_metric_checks = 0
    for seed in ALL_SEEDS:
        if 0 not in counts:
            raise ValueError("dataset pairing audit requires the m=0 baseline")
        bundles = {m: _make_bundle(seed, m) for m in counts}
        baseline = bundles[0]
        legacy = _make_bundle(seed, None)
        _compare_batch_arrays(errors, f"seed {seed} m0 legacy train", baseline.train, legacy.train)
        _compare_batch_arrays(errors, f"seed {seed} m0 legacy iid", baseline.iid, legacy.iid)
        exact_array_checks += 2

        baseline_train = _batch_arrays(baseline.train)
        baseline_iid_digest = _batch_digest(baseline.iid)
        baseline_diagnostic_digests = {
            name: _batch_digest(batch) for name, batch in baseline.diagnostics.items()
        }
        baseline_shared = np.asarray(baseline.train.latents["both_wrong"], dtype=bool)
        previous_shared = baseline_shared
        previous_q_only = np.asarray(baseline.train.latents["q_wrong"], dtype=bool)
        previous_m = 0
        for m in counts:
            run = run_index[(seed, m)]
            bundle = bundles[m]
            train = bundle.train
            expected_counts = {
                "p_error_count": 1_000,
                "q_error_count": 500,
                "both_error_count": 500 - m,
                "p_only_error_count": 500 + m,
                "q_only_error_count": m,
                "neither_error_count": 9_000 - m,
            }
            for name, expected in expected_counts.items():
                actual = int(train.metadata[name])
                if actual != expected:
                    errors.append(f"seed {seed} m={m}: metadata {name}={actual}, expected {expected}")
                _metric_matches(
                    errors,
                    run,
                    "support_completion_design",
                    "competition_train",
                    name,
                    0,
                    float(expected),
                )
                exact_metric_checks += 1
            if float(train.metadata["realized_q_p"]) != 0.90:
                errors.append(f"seed {seed} m={m}: P marginal changed")
            if float(train.metadata["realized_q_q"]) != 0.95:
                errors.append(f"seed {seed} m={m}: Q marginal changed")

            _compare_batch_arrays(
                errors,
                f"seed {seed} m={m} paired train",
                train,
                baseline.train,
                allowed_differences=frozenset(
                    {
                        "channel.Q_2",
                        "latent.Q_goal",
                        "latent.Q_error",
                        "latent.both_wrong",
                        "latent.p_wrong",
                        "latent.q_wrong",
                    }
                ),
            )
            exact_array_checks += len(baseline_train) - 6
            q_goal_change = np.asarray(train.latents["Q_goal"]) != np.asarray(
                baseline.train.latents["Q_goal"]
            )
            q_error_change = np.asarray(train.latents["Q_error"]) != np.asarray(
                baseline.train.latents["Q_error"]
            )
            q_bit_change = np.asarray(train.channels["Q_2"]) != np.asarray(
                baseline.train.channels["Q_2"]
            )
            if not np.array_equal(q_goal_change, q_error_change) or not np.array_equal(
                q_goal_change, q_bit_change
            ):
                errors.append(f"seed {seed} m={m}: Q goal/error/parity change masks differ")
            hamming = int(np.sum(q_goal_change))
            if hamming != 2 * m:
                errors.append(f"seed {seed} m={m}: Q Hamming={hamming}, expected {2 * m}")
            q_only = np.asarray(train.latents["q_wrong"], dtype=bool)
            shared = np.asarray(train.latents["both_wrong"], dtype=bool)
            if int(np.sum(q_goal_change & baseline_shared)) != m:
                errors.append(f"seed {seed} m={m}: shared-to-P-only change count is wrong")
            if int(np.sum(q_goal_change & q_only)) != m:
                errors.append(f"seed {seed} m={m}: neither-to-Q-only change count is wrong")
            if np.any(shared & ~previous_shared):
                errors.append(f"seed {seed} m={m}: retained shared errors are not nested")
            if np.any(previous_q_only & ~q_only):
                errors.append(f"seed {seed} m={m}: Q-only errors are not a growing prefix")
            adjacent_hamming = int(
                np.sum(
                    np.asarray(train.latents["Q_goal"])
                    != np.asarray(bundles[previous_m].train.latents["Q_goal"])
                )
            )
            if adjacent_hamming != 2 * (m - previous_m):
                errors.append(
                    f"seed {seed} {previous_m}->{m}: adjacent Hamming={adjacent_hamming}, "
                    f"expected {2 * (m - previous_m)}"
                )
            previous_shared = shared
            previous_q_only = q_only
            previous_m = m

            if _batch_digest(bundle.iid) != baseline_iid_digest:
                errors.append(f"seed {seed} m={m}: common IID array changed")
            for name, batch in bundle.diagnostics.items():
                if _batch_digest(batch) != baseline_diagnostic_digests[name]:
                    errors.append(f"seed {seed} m={m}: diagnostic {name} array changed")
            exact_array_checks += 1 + len(bundle.diagnostics)

            factorial = make_competing_factorial_dataset(
                n=4_096,
                k_q=2,
                k_y=5,
                seed=60_000_000 + seed,
                control_seed=1_500_450_271,
                max_k_q=3,
                max_k_y=5,
                state_dim=8,
                split="factorial_probe_eval",
                id_offset=2_100_000_000,
            )
            partition = q_only_codeword_partition(train, factorial)
            seen_codewords = int(partition["seen_codeword_count"])
            possible_codewords = int(partition["possible_codeword_count"])
            seen_rows = int(np.sum(np.asarray(partition["seen"], dtype=bool)))
            unseen_rows = int(np.sum(np.asarray(partition["unseen"], dtype=bool)))
            if possible_codewords != 64 or seen_rows != 16 * seen_codewords:
                errors.append(f"seed {seed} m={m}: malformed raw-codeword partition")
            coverage = seen_codewords / possible_codewords
            coverage_summary = get(run.summary, "data.q_only_codeword_coverage", {})
            for path, expected_count in {
                "training_rows": m,
                "seen_active_codewords": seen_codewords,
                "possible_active_codewords": possible_codewords,
                "factorial_seen_rows": seen_rows,
                "factorial_unseen_rows": unseen_rows,
            }.items():
                expect_equal(
                    errors,
                    run.path.name,
                    get(coverage_summary, path),
                    expected_count,
                    f"coverage.{path}",
                )
            for metric, expected_metric in {
                "q_only_seen_active_codewords": float(seen_codewords),
                "q_only_possible_active_codewords": float(possible_codewords),
                "q_only_active_codeword_coverage": coverage,
            }.items():
                _metric_matches(
                    errors,
                    run,
                    "support_completion_design",
                    "competition_train",
                    metric,
                    0,
                    expected_metric,
                    atol=1e-15,
                )
                exact_metric_checks += 1

            exposure = static_subset_exposure(
                q_only,
                batch_size=250,
                steps=8_192,
                seed=seed,
                shuffle=True,
            )
            cumulative = cast(Mapping[int, int], exposure["cumulative_by_step"])
            unique = cast(Mapping[int, int], exposure["unique_by_step"])
            summary_cumulative = get(run.summary, "training.q_only_presentations_by_checkpoint")
            summary_unique = get(run.summary, "training.unique_q_only_rows_seen_by_checkpoint")
            expected_summary_cumulative = {str(step): cumulative[step] for step in COMPETITION_STEPS}
            expected_summary_unique = {str(step): unique[step] for step in COMPETITION_STEPS}
            expect_equal(
                errors,
                run.path.name,
                summary_cumulative,
                expected_summary_cumulative,
                "summary cumulative exposure map",
            )
            expect_equal(
                errors,
                run.path.name,
                summary_unique,
                expected_summary_unique,
                "summary unique exposure map",
            )
            expect_equal(
                errors,
                run.path.name,
                get(run.summary, "training.first_q_only_presentation_step"),
                exposure["first_presentation_step"],
                "first Q-only presentation step",
            )
            expect_equal(
                errors,
                run.path.name,
                get(run.summary, "training.all_unique_q_only_seen_step"),
                exposure["all_unique_seen_step"],
                "all Q-only rows seen step",
            )
            for step in COMPETITION_STEPS:
                expected_fraction = cumulative[step] / (step * 250) if step else 0.0
                for metric, expected_exposure in {
                    "cumulative_q_only_presentations": float(cumulative[step]),
                    "unique_q_only_rows_seen": float(unique[step]),
                    "q_only_training_rows": float(m),
                    "q_only_presentation_fraction": expected_fraction,
                }.items():
                    _metric_matches(
                        errors,
                        run,
                        "support_completion_exposure",
                        "competition_train",
                        metric,
                        step,
                        expected_exposure,
                        atol=1e-15,
                    )
                    exact_exposure_checks += 1
            for panel, panel_rows in (
                ("all", 1_024),
                ("seen_codewords", seen_rows),
                ("unseen_codewords", unseen_rows),
            ):
                for step in COMPETITION_STEPS:
                    _metric_matches(
                        errors,
                        run,
                        "support_completion_generalization",
                        f"factorial_q_only_{panel}",
                        "panel_rows",
                        step,
                        float(panel_rows),
                    )
                    exact_metric_checks += 1

            training_overlap = get(run.summary, "data.training_overlap", {})
            iid_overlap = get(run.summary, "data.iid_overlap", {})
            for name in (
                "p_error_count",
                "q_error_count",
                "both_error_count",
                "p_only_error_count",
                "q_only_error_count",
                "error_phi",
            ):
                expect_equal(
                    errors,
                    run.path.name,
                    get(training_overlap, name),
                    train.metadata[name],
                    f"training overlap {name}",
                )
                expect_equal(
                    errors,
                    run.path.name,
                    get(iid_overlap, name),
                    bundle.iid.metadata[name],
                    f"IID overlap {name}",
                )

            rows.append(
                {
                    "inference_status": INFERENCE_STATUS,
                    "run_id": run.path.name,
                    "seed": seed,
                    "cohort": run.cohort,
                    "q_only_error_count": m,
                    **expected_counts,
                    "q_goal_hamming_from_m0": hamming,
                    "q_goal_hamming_from_previous_count": adjacent_hamming,
                    "seen_active_codewords": seen_codewords,
                    "possible_active_codewords": possible_codewords,
                    "active_codeword_coverage": coverage,
                    "factorial_seen_rows": seen_rows,
                    "factorial_unseen_rows": unseen_rows,
                    "final_q_only_presentations": cumulative[8_192],
                    "first_q_only_presentation_step": exposure["first_presentation_step"],
                    "all_unique_q_only_seen_step": exposure["all_unique_seen_step"],
                    "train_array_digest": _batch_digest(train),
                    "common_iid_array_digest": _batch_digest(bundle.iid),
                    "diagnostic_array_digest": hashlib.sha256(
                        "".join(
                            _batch_digest(bundle.diagnostics[name])
                            for name in sorted(bundle.diagnostics)
                        ).encode("ascii")
                    ).hexdigest(),
                }
            )

    if errors:
        _raise_audit("E16 regenerated dataset/pairing audit", errors)
    audit = {
        "runs_regenerated": len(rows),
        "seeds_audited": len(ALL_SEEDS),
        "support_counts": list(counts),
        "m0_legacy_array_replays": 2 * len(ALL_SEEDS),
        "unchanged_array_and_panel_checks": exact_array_checks,
        "exact_exposure_metric_checks": exact_exposure_checks,
        "exact_design_and_panel_metric_checks": exact_metric_checks,
        "marginals_fixed": True,
        "q_goal_and_final_parity_hamming_equals_2m": True,
        "shared_errors_nested_downward": True,
        "q_only_errors_nested_upward": True,
        "iid_and_diagnostics_fixed_across_m": True,
        "possible_q_only_active_codewords": 64,
        "current_source_regeneration_audit_passed": True,
    }
    return rows, audit


def _optional_value(
    run: Run,
    stage: str,
    split: str,
    metric: str,
    step: int,
    intervention: str = "none",
) -> float | None:
    series = run.metrics.get((stage, split, intervention, metric), {})
    point = series.get(step)
    return None if point is None else float(point.value)


def endpoint_taxonomy(signature: int, tuple_consistency: float, nuisance_consistency: float) -> str:
    if signature == STABLE_GATE_SIGNATURE and tuple_consistency >= TUPLE_CONSISTENCY_THRESHOLD:
        return "stable gate 113"
    if signature == STABLE_GATE_SIGNATURE:
        return "modal 113, unstable"
    if tuple_consistency >= TUPLE_CONSISTENCY_THRESHOLD:
        if signature == PURE_Y_SIGNATURE:
            return "pure Y"
        if signature == PURE_Q_SIGNATURE:
            return "pure Q"
        if signature == PURE_P_SIGNATURE:
            return "pure P"
        return "other semantic rule"
    if nuisance_consistency >= TUPLE_CONSISTENCY_THRESHOLD:
        return "raw-codeword patch"
    return "nuisance/state-sensitive"


def checkpoint_rows(runs: Sequence[Run]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for run in runs:
        coverage = get(run.summary, "data.q_only_codeword_coverage", {})
        seen_codewords = int(get(coverage, "seen_active_codewords"))
        for step in COMPETITION_STEPS:
            signature = round(
                run.value(
                    "competition_truth_table",
                    "factorial_eval",
                    "boolean_signature_int",
                    step,
                )
            )
            tuple_consistency = run.value(
                "competition_truth_table",
                "factorial_eval",
                "tuple_consistency",
                step,
            )
            nuisance_consistency = run.value(
                "competition_truth_table",
                "factorial_eval",
                "nuisance_consistency",
                step,
            )
            all_accuracy = run.value(
                "support_completion_generalization",
                "factorial_q_only_all",
                "target_accuracy",
                step,
            )
            all_probability = run.value(
                "support_completion_generalization",
                "factorial_q_only_all",
                "mean_target_probability",
                step,
            )
            seen_accuracy = _optional_value(
                run,
                "support_completion_generalization",
                "factorial_q_only_seen_codewords",
                "target_accuracy",
                step,
            )
            seen_probability = _optional_value(
                run,
                "support_completion_generalization",
                "factorial_q_only_seen_codewords",
                "mean_target_probability",
                step,
            )
            unseen_accuracy = _optional_value(
                run,
                "support_completion_generalization",
                "factorial_q_only_unseen_codewords",
                "target_accuracy",
                step,
            )
            unseen_probability = _optional_value(
                run,
                "support_completion_generalization",
                "factorial_q_only_unseen_codewords",
                "mean_target_probability",
                step,
            )
            for panel in ("all", "seen_codewords", "unseen_codewords"):
                split = f"factorial_q_only_{panel}"
                target = _optional_value(
                    run, "support_completion_generalization", split, "target_accuracy", step
                )
                if target is None:
                    continue
                rho_p = run.value(
                    "support_completion_generalization", split, "rho_p", step
                )
                rho_q = run.value(
                    "support_completion_generalization", split, "rho_q", step
                )
                rho_y = run.value(
                    "support_completion_generalization", split, "rho_y_code", step
                )
                if not (
                    math.isclose(target, rho_p, abs_tol=1e-12)
                    and math.isclose(target, rho_y, abs_tol=1e-12)
                    and math.isclose(rho_q, 1.0 - target, abs_tol=1e-12)
                ):
                    errors.append(
                        f"{run.path.name}: Q-only semantic identity failed for {panel} at {step}"
                    )
            behavior = {
                goal: run.value(
                    "competition_behavior",
                    "factorial_eval",
                    {"P": "rho_p", "Q": "rho_q", "Y": "rho_y_code"}[goal],
                    step,
                )
                for goal in GOALS
            }
            causal = {
                goal: run.value(
                    "competition_causal",
                    "factorial_eval",
                    "causal_score",
                    step,
                    goal,
                )
                for goal in GOALS
            }
            probes = {
                goal: run.value(
                    "competition_probe",
                    "factorial_probe",
                    f"representations__final_hidden__heldout_accuracy__{goal}",
                    step,
                )
                for goal in GOALS
            }
            row: dict[str, Any] = {
                "inference_status": INFERENCE_STATUS,
                "run_id": run.path.name,
                "seed": run.seed,
                "cohort": run.cohort,
                "q_only_error_count": run.m,
                "step": step,
                "examples_seen": step * 250,
                "cumulative_q_only_presentations": round(
                    run.value(
                        "support_completion_exposure",
                        "competition_train",
                        "cumulative_q_only_presentations",
                        step,
                    )
                ),
                "unique_q_only_rows_seen": round(
                    run.value(
                        "support_completion_exposure",
                        "competition_train",
                        "unique_q_only_rows_seen",
                        step,
                    )
                ),
                "seen_active_codewords": seen_codewords,
                "active_codeword_coverage": seen_codewords / 64.0,
                "signature": signature,
                "tuple_consistency": tuple_consistency,
                "codeword_consistency": run.value(
                    "competition_truth_table",
                    "factorial_eval",
                    "codeword_consistency",
                    step,
                ),
                "nuisance_consistency": nuisance_consistency,
                "stable_gate": int(
                    signature == STABLE_GATE_SIGNATURE
                    and tuple_consistency >= TUPLE_CONSISTENCY_THRESHOLD
                ),
                "taxonomy": endpoint_taxonomy(
                    signature, tuple_consistency, nuisance_consistency
                ),
                "q_only_diagnostic_accuracy": run.value(
                    "competition_candidates",
                    "competition_q_wrong",
                    "target_accuracy",
                    step,
                ),
                "common_iid_target_accuracy": run.value(
                    "competition_candidates",
                    "competition_iid",
                    "target_accuracy",
                    step,
                ),
                "q_only_factorial_accuracy": all_accuracy,
                "q_only_target_probability": all_probability,
                "q_only_seen_accuracy": seen_accuracy,
                "q_only_seen_target_probability": seen_probability,
                "q_only_unseen_accuracy": unseen_accuracy,
                "q_only_unseen_target_probability": unseen_probability,
                "seen_minus_unseen_probability": (
                    None
                    if seen_probability is None or unseen_probability is None
                    else seen_probability - unseen_probability
                ),
            }
            row.update({f"behavior_{goal}": behavior[goal] for goal in GOALS})
            row.update({f"causal_{goal}": causal[goal] for goal in GOALS})
            row.update({f"probe_{goal}": probes[goal] for goal in GOALS})
            for key in Q_ONLY_TUPLES:
                row[f"tuple_{key}_positive_rate"] = run.value(
                    "competition_truth_table",
                    "factorial_eval",
                    f"tuples__{key}__positive_rate",
                    step,
                )
                row[f"tuple_{key}_target_probability"] = (
                    1.0
                    - run.value(
                        "competition_truth_table",
                        "factorial_eval",
                        f"tuples__{key}__mean_positive_probability",
                        step,
                    )
                    if key == "-+-"
                    else run.value(
                        "competition_truth_table",
                        "factorial_eval",
                        f"tuples__{key}__mean_positive_probability",
                        step,
                    )
                )
            rows.append(row)
    if errors:
        _raise_audit("E16 checkpoint semantic audit", errors)
    return rows


def _mean_interval(values: Sequence[float], *, label: str) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{label}: bootstrap values must be finite and non-empty")
    rng = np.random.default_rng(stable_seed("bootstrap", label))
    indices = rng.integers(0, len(array), size=(BOOTSTRAP_DRAWS, len(array)))
    draws = np.mean(array[indices], axis=1)
    low, high = np.quantile(draws, (0.025, 0.975))
    return {
        "estimate": float(np.mean(array)),
        "ci_low": float(low),
        "ci_high": float(high),
        "seed_clusters": len(array),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
    }


def _finite_seed_values(rows: Sequence[Mapping[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            values.append(float(value))
    return values


ENDPOINT_OUTCOMES = (
    "stable_gate",
    "q_only_diagnostic_accuracy",
    "q_only_factorial_accuracy",
    "q_only_target_probability",
    "q_only_seen_accuracy",
    "q_only_seen_target_probability",
    "q_only_unseen_accuracy",
    "q_only_unseen_target_probability",
    "seen_minus_unseen_probability",
    "common_iid_target_accuracy",
    "active_codeword_coverage",
    "behavior_P",
    "behavior_Q",
    "behavior_Y",
    "causal_P",
    "causal_Q",
    "causal_Y",
)


def run_endpoint_rows(
    checkpoints: Sequence[Mapping[str, Any]], runs: Sequence[Run]
) -> list[dict[str, Any]]:
    final = [dict(row) for row in checkpoints if int(row["step"]) == 8_192]
    run_index = {(run.seed, run.m): run for run in runs}
    for row in final:
        run = run_index[(int(row["seed"]), int(row["q_only_error_count"]))]
        row["first_q_only_presentation_step"] = get(
            run.summary, "training.first_q_only_presentation_step"
        )
        row["all_unique_q_only_seen_step"] = get(
            run.summary, "training.all_unique_q_only_seen_step"
        )
        row["design_panel"] = (
            "triggered_refinement"
            if int(row["q_only_error_count"]) in REFINEMENT_COUNTS
            else "primary_registered"
        )
        row["pure_Y"] = int(
            int(row["signature"]) == PURE_Y_SIGNATURE
            and float(row["tuple_consistency"]) >= TUPLE_CONSISTENCY_THRESHOLD
        )
        row["pure_Q"] = int(
            int(row["signature"]) == PURE_Q_SIGNATURE
            and float(row["tuple_consistency"]) >= TUPLE_CONSISTENCY_THRESHOLD
        )
        row["pure_P"] = int(
            int(row["signature"]) == PURE_P_SIGNATURE
            and float(row["tuple_consistency"]) >= TUPLE_CONSISTENCY_THRESHOLD
        )
    return sorted(final, key=lambda row: (int(row["seed"]), int(row["q_only_error_count"])))


def dose_summaries(
    endpoints: Sequence[Mapping[str, Any]], *, counts: Sequence[int] = SUPPORT_COUNTS
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for m in counts:
        subset = [row for row in endpoints if int(row["q_only_error_count"]) == m]
        if len(subset) != len(ALL_SEEDS):
            raise RuntimeError(f"dose summary m={m} has {len(subset)} rows")
        for outcome in (*ENDPOINT_OUTCOMES, "pure_Y", "pure_Q", "pure_P"):
            values = _finite_seed_values(subset, outcome)
            if not values:
                continue
            result.append(
                {
                    "inference_status": INFERENCE_STATUS,
                    "q_only_error_count": m,
                    "design_panel": (
                        "triggered_refinement" if m in REFINEMENT_COUNTS else "primary_registered"
                    ),
                    "outcome": outcome,
                    "identified_seeds": len(values),
                    **_mean_interval(values, label=f"dose-{m}-{outcome}"),
                }
            )
        taxonomy_counts = Counter(str(row["taxonomy"]) for row in subset)
        for taxonomy, count in sorted(taxonomy_counts.items()):
            values = [float(str(row["taxonomy"]) == taxonomy) for row in subset]
            result.append(
                {
                    "inference_status": INFERENCE_STATUS,
                    "q_only_error_count": m,
                    "design_panel": (
                        "triggered_refinement" if m in REFINEMENT_COUNTS else "primary_registered"
                    ),
                    "outcome": f"taxonomy:{taxonomy}",
                    "identified_seeds": len(values),
                    "count": count,
                    **_mean_interval(values, label=f"dose-{m}-taxonomy-{taxonomy}"),
                }
            )
    return result


def gate_transition_rows(
    endpoints: Sequence[Mapping[str, Any]], *, counts: Sequence[int] = SUPPORT_COUNTS
) -> list[dict[str, Any]]:
    indexed = {
        (int(row["seed"]), int(row["q_only_error_count"])): row for row in endpoints
    }
    result: list[dict[str, Any]] = []
    for seed in ALL_SEEDS:
        gates = [int(indexed[(seed, m)]["stable_gate"]) for m in counts]
        probabilities = [
            float(indexed[(seed, m)]["q_only_target_probability"]) for m in counts
        ]
        gate_reversals = sum(
            gates[index] == 0 and gates[index + 1] == 1
            for index in range(len(gates) - 1)
        )
        probability_reversals = sum(
            probabilities[index + 1] < probabilities[index] - 1e-12
            for index in range(len(probabilities) - 1)
        )
        material_probability_reversals = sum(
            probabilities[index + 1] < probabilities[index] - 0.05
            for index in range(len(probabilities) - 1)
        )
        first_loss = next((m for m, gate in zip(counts, gates, strict=True) if gate == 0), None)
        for left, right in pairwise(counts):
            left_row = indexed[(seed, left)]
            right_row = indexed[(seed, right)]
            result.append(
                {
                    "inference_status": INFERENCE_STATUS,
                    "seed": seed,
                    "cohort": "original" if seed in ORIGINAL_SEEDS else "fresh",
                    "left_count": left,
                    "right_count": right,
                    "gate_transition": f"{int(left_row['stable_gate'])}->{int(right_row['stable_gate'])}",
                    "stable_gate_difference": int(right_row["stable_gate"])
                    - int(left_row["stable_gate"]),
                    "q_only_target_probability_difference": float(
                        right_row["q_only_target_probability"]
                    )
                    - float(left_row["q_only_target_probability"]),
                    "q_only_diagnostic_accuracy_difference": float(
                        right_row["q_only_diagnostic_accuracy"]
                    )
                    - float(left_row["q_only_diagnostic_accuracy"]),
                    "gate_reversals_across_all_counts": gate_reversals,
                    "probability_reversals_across_all_counts": probability_reversals,
                    "material_probability_reversals_across_all_counts": material_probability_reversals,
                    "first_stable_gate_loss_count": first_loss,
                }
            )
    return result


def paired_contrasts(endpoints: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (int(row["seed"]), int(row["q_only_error_count"])): row for row in endpoints
    }
    result: list[dict[str, Any]] = []
    outcomes = (
        "stable_gate",
        "q_only_diagnostic_accuracy",
        "q_only_factorial_accuracy",
        "q_only_target_probability",
        "q_only_unseen_target_probability",
        "common_iid_target_accuracy",
        "pure_Y",
    )
    comparisons = (
        (0, 2, "presence jump: 0 to 2"),
        *(
            (left, right, f"adjacent: {left} to {right}")
            for left, right in pairwise(SUPPORT_COUNTS)
            if (left, right) != (0, 2)
        ),
        *(
            (left, right, f"triggered refinement: {left} to {right}")
            for left, right in pairwise((2, *REFINEMENT_COUNTS, 10))
        ),
    )
    for left, right, comparison in comparisons:
        for outcome in outcomes:
            differences: list[float] = []
            for seed in ALL_SEEDS:
                left_value = indexed[(seed, left)].get(outcome)
                right_value = indexed[(seed, right)].get(outcome)
                if not isinstance(left_value, (int, float)) or not isinstance(
                    right_value, (int, float)
                ):
                    continue
                differences.append(float(right_value) - float(left_value))
            if differences:
                result.append(
                    {
                        "inference_status": INFERENCE_STATUS,
                        "contrast_kind": "paired_difference",
                        "comparison": comparison,
                        "left_count": left,
                        "right_count": right,
                        "outcome": outcome,
                        **_mean_interval(
                            differences,
                            label=f"contrast-{left}-{right}-{outcome}",
                        ),
                    }
                )
    x = np.log2(np.asarray(SUPPORT_COUNTS[1:], dtype=np.float64) / 2.0)
    for outcome in outcomes:
        slopes: list[float] = []
        for seed in ALL_SEEDS:
            y_values: list[float] = []
            x_values: list[float] = []
            for x_value, m in zip(x, SUPPORT_COUNTS[1:], strict=True):
                value = indexed[(seed, m)].get(outcome)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    x_values.append(float(x_value))
                    y_values.append(float(value))
            if len(y_values) >= 2:
                slopes.append(float(np.polyfit(x_values, y_values, 1)[0]))
        if slopes:
            result.append(
                {
                    "inference_status": INFERENCE_STATUS,
                    "contrast_kind": "positive_dose_seed_slope",
                    "comparison": "m>0 slope per doubling from log2(m/2)",
                    "left_count": None,
                    "right_count": None,
                    "outcome": outcome,
                    **_mean_interval(slopes, label=f"positive-dose-slope-{outcome}"),
                }
            )
    return result


def _half_crossing(counts: Sequence[int], fitted: Sequence[float]) -> float | None:
    for index, value in enumerate(fitted):
        if value > 0.5:
            continue
        if index == 0:
            return float(counts[0])
        left_m = float(counts[index - 1])
        right_m = float(counts[index])
        left_y = float(fitted[index - 1])
        right_y = float(value)
        if math.isclose(left_y, right_y, abs_tol=1e-15):
            return right_m
        fraction = (left_y - 0.5) / (left_y - right_y)
        return left_m + fraction * (right_m - left_m)
    return None


def refinement_results(endpoints: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = (2, 4, 6, 8, 10)
    indexed = {
        (int(row["seed"]), int(row["q_only_error_count"])): row for row in endpoints
    }
    prevalence = [
        float(np.mean([float(indexed[(seed, m)]["stable_gate"]) for seed in ALL_SEEDS]))
        for m in counts
    ]
    fitted = _decreasing_isotonic(prevalence)
    crossing = _half_crossing(counts, fitted)
    rng = np.random.default_rng(stable_seed("refinement-m50"))
    bootstrap_crossings: list[float] = []
    for _ in range(BOOTSTRAP_DRAWS):
        sampled = rng.choice(np.asarray(ALL_SEEDS), size=len(ALL_SEEDS), replace=True)
        sampled_prevalence = [
            float(np.mean([float(indexed[(int(seed), m)]["stable_gate"]) for seed in sampled]))
            for m in counts
        ]
        sampled_crossing = _half_crossing(counts, _decreasing_isotonic(sampled_prevalence))
        if sampled_crossing is not None:
            bootstrap_crossings.append(sampled_crossing)
    reversal_seeds = 0
    probability_reversal_seeds = 0
    for seed in ALL_SEEDS:
        gates = [int(indexed[(seed, m)]["stable_gate"]) for m in COMBINED_COUNTS]
        probabilities = [
            float(indexed[(seed, m)]["q_only_target_probability"]) for m in COMBINED_COUNTS
        ]
        reversal_seeds += int(
            any(left == 0 and right == 1 for left, right in pairwise(gates))
        )
        probability_reversal_seeds += int(
            any(
                right < left - 0.05
                for left, right in pairwise(probabilities)
            )
        )
    interval: list[float | None]
    if bootstrap_crossings:
        low, high = np.quantile(np.asarray(bootstrap_crossings), (0.025, 0.975))
        interval = [float(low), float(high)]
    else:
        interval = [None, None]
    return {
        "inference_status": INFERENCE_STATUS,
        "conditional_panel_counts": list(REFINEMENT_COUNTS),
        "bracket_counts_including_registered_endpoints": list(counts),
        "stable_gate_counts": {
            str(m): round(value * len(ALL_SEEDS))
            for m, value in zip(counts, prevalence, strict=True)
        },
        "stable_gate_prevalence": {
            str(m): value for m, value in zip(counts, prevalence, strict=True)
        },
        "isotonic_prevalence": {
            str(m): value for m, value in zip(counts, fitted, strict=True)
        },
        "linearly_interpolated_isotonic_m50": crossing,
        "m50_seed_cluster_bootstrap_interval": interval,
        "m50_identified_bootstrap_draws": len(bootstrap_crossings),
        "seed_clusters": len(ALL_SEEDS),
        "stable_gate_reappearance_seeds_across_combined_counts": reversal_seeds,
        "material_q_only_probability_reversal_seeds_across_combined_counts": probability_reversal_seeds,
    }


def independent_reference_summary(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for outcome in (
        "stable_gate",
        "q_only_diagnostic_accuracy",
        "q_only_target_probability",
    ):
        values = [float(row[f"{outcome}_difference"]) for row in rows]
        result.append(
            {
                "inference_status": INFERENCE_STATUS,
                "comparison": "E16 m=450 minus archived E15 independent; count-matched, row allocation differs",
                "outcome": outcome,
                **_mean_interval(values, label=f"independent-reference-{outcome}"),
            }
        )
    return result


def _decreasing_isotonic(values: Sequence[float]) -> list[float]:
    blocks: list[tuple[float, int]] = []
    for value in values:
        blocks.append((float(value), 1))
        while len(blocks) >= 2 and blocks[-2][0] < blocks[-1][0]:
            right_value, right_weight = blocks.pop()
            left_value, left_weight = blocks.pop()
            weight = left_weight + right_weight
            blocks.append(
                (
                    (left_value * left_weight + right_value * right_weight) / weight,
                    weight,
                )
            )
    result: list[float] = []
    for value, weight in blocks:
        result.extend([value] * weight)
    return result


def refinement_decision(endpoints: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    prevalence = {
        m: float(
            np.mean(
                [
                    float(row["stable_gate"])
                    for row in endpoints
                    if int(row["q_only_error_count"]) == m
                ]
            )
        )
        for m in SUPPORT_COUNTS
    }
    drops = [
        prevalence[left] - prevalence[right]
        for left, right in pairwise(SUPPORT_COUNTS)
    ]
    maximum_drop = max(drops)
    selected_index = next(
        index for index, value in enumerate(drops) if math.isclose(value, maximum_drop, abs_tol=1e-12)
    )
    left = SUPPORT_COUNTS[selected_index]
    right = SUPPORT_COUNTS[selected_index + 1]
    candidates: list[int] = []
    for fraction in (0.25, 0.50, 0.75):
        raw = left + fraction * (right - left)
        value = math.floor(raw + 0.5)
        if value not in SUPPORT_COUNTS and value not in candidates and left < value < right:
            candidates.append(value)
    total_drop = prevalence[0] - prevalence[450]
    triggered = (
        total_drop >= REFINEMENT_TOTAL_DROP
        and maximum_drop >= REFINEMENT_ADJACENT_DROP
        and bool(candidates)
    )
    isotonic = _decreasing_isotonic([prevalence[m] for m in SUPPORT_COUNTS])
    first_half = next(
        (m for m, fitted in zip(SUPPORT_COUNTS, isotonic, strict=True) if fitted <= 0.5),
        None,
    )
    return {
        "inference_status": INFERENCE_STATUS,
        "rule_frozen_before_outcome_inspection": True,
        "stable_gate_definition": (
            "modal Boolean signature 113 and factorial tuple consistency >= 0.90 at step 8192"
        ),
        "prevalence_by_count": {str(m): prevalence[m] for m in SUPPORT_COUNTS},
        "prevalence_counts": {
            str(m): round(prevalence[m] * len(ALL_SEEDS)) for m in SUPPORT_COUNTS
        },
        "adjacent_drops": [
            {"left": left_m, "right": right_m, "drop": drop}
            for left_m, right_m, drop in zip(
                SUPPORT_COUNTS[:-1], SUPPORT_COUNTS[1:], drops, strict=True
            )
        ],
        "total_drop_0_to_450": total_drop,
        "required_total_drop": REFINEMENT_TOTAL_DROP,
        "largest_adjacent_drop": maximum_drop,
        "required_adjacent_drop": REFINEMENT_ADJACENT_DROP,
        "earliest_largest_drop_interval": [left, right],
        "rounded_quarter_point_candidates": candidates,
        "triggered": triggered,
        "action": (
            f"run counts {candidates} with the same 20 seeds"
            if triggered
            else "stop the E16 refinement line"
        ),
        "isotonic_prevalence": {
            str(m): value for m, value in zip(SUPPORT_COUNTS, isotonic, strict=True)
        },
        "isotonic_first_count_at_or_below_half": first_half,
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(_json_safe(row))


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def plot_font_family() -> str:
    """Register Myriad Pro explicitly when Matplotlib has not discovered it."""

    directories: list[Path] = []
    configured = os.environ.get("FORKWORLD_FONT_DIR")
    if configured:
        directories.append(Path(configured).expanduser())
    directories.extend((Path.home() / "Library" / "Fonts", Path("/Library/Fonts")))
    font_paths: list[Path] = []
    for directory in directories:
        if directory.is_dir():
            font_paths.extend(
                path
                for path in directory.iterdir()
                if path.is_file()
                and "myriad" in path.name.lower()
                and path.suffix.lower() in {".otf", ".ttf", ".ttc"}
            )
    names: list[str] = []
    for path in sorted(set(font_paths)):
        try:
            fm.fontManager.addfont(str(path))
            names.append(fm.FontProperties(fname=str(path)).get_name())
        except RuntimeError:
            continue
    if names:
        regular = [name for name in names if name.lower() == "myriad pro"]
        return regular[0] if regular else names[0]
    discovered = sorted(
        {font.name for font in fm.fontManager.ttflist if "myriad" in font.name.lower()}
    )
    if discovered:
        return discovered[0]
    warnings.warn(
        "Myriad Pro was not found; falling back to DejaVu Sans.",
        RuntimeWarning,
        stacklevel=2,
    )
    return "DejaVu Sans"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": plot_font_family(),
            "font.size": 9.2,
            "axes.titlesize": 10.4,
            "axes.titleweight": 600,
            "axes.labelsize": 9.2,
            "axes.labelcolor": INK,
            "axes.edgecolor": "#AAB2BD",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": "#4B5563",
            "ytick.color": "#4B5563",
            "legend.frameon": False,
            "legend.fontsize": 8.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(axis: Axes, label: str) -> None:
    axis.annotate(
        label,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(-12, 8),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=11.0,
        fontweight=600,
        color=INK,
        annotation_clip=False,
    )


def _dose_summary_index(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, str], Mapping[str, Any]]:
    return {
        (int(row["q_only_error_count"]), str(row["outcome"])): row for row in rows
    }


def _mean_checkpoint_curve(
    checkpoints: Sequence[Mapping[str, Any]], count: int
) -> tuple[np.ndarray, np.ndarray]:
    subset = [row for row in checkpoints if int(row["q_only_error_count"]) == count]
    x_values: list[float] = []
    y_values: list[float] = []
    for step in COMPETITION_STEPS:
        at_step = [row for row in subset if int(row["step"]) == step]
        if len(at_step) != len(ALL_SEEDS):
            raise RuntimeError(f"figure curve m={count}, step={step} has {len(at_step)} seeds")
        x_values.append(
            float(np.mean([float(row["cumulative_q_only_presentations"]) for row in at_step]))
        )
        y_values.append(
            float(np.mean([float(row["q_only_target_probability"]) for row in at_step]))
        )
    return np.asarray(x_values), np.asarray(y_values)


def make_figure(
    checkpoints: Sequence[Mapping[str, Any]],
    endpoints: Sequence[Mapping[str, Any]],
    dose_summary: Sequence[Mapping[str, Any]],
    refinement: Mapping[str, Any],
) -> None:
    """Draw the E16 summary with fixed margins and panel-local legends."""

    configure_style()
    figure = plt.figure(figsize=(12.8, 8.25))
    grid = figure.add_gridspec(
        2,
        2,
        left=0.075,
        right=0.965,
        bottom=0.105,
        top=0.855,
        hspace=0.60,
        wspace=0.28,
    )
    axis_a = figure.add_subplot(grid[0, 0])
    axis_b = figure.add_subplot(grid[0, 1])
    axis_c = figure.add_subplot(grid[1, 0])
    axis_d = figure.add_subplot(grid[1, 1])
    axes = (axis_a, axis_b, axis_c, axis_d)
    summary_index = _dose_summary_index(dose_summary)
    x = np.arange(len(COMBINED_COUNTS), dtype=np.float64)

    for axis in (axis_a, axis_b):
        axis.axvspan(1.5, 4.5, color="#F2EFF7", alpha=0.95, zorder=0)
    axis_a.text(
        3.0,
        0.975,
        "triggered refinement",
        ha="center",
        va="top",
        color=PURPLE,
        fontsize=7.6,
        transform=axis_a.get_xaxis_transform(),
        bbox={"facecolor": "#F2EFF7", "edgecolor": "none", "pad": 1.4},
    )

    for outcome, color, marker, label in (
        ("stable_gate", PURPLE, "o", "stable signature-113 gate"),
        ("q_only_target_probability", TEAL, "s", "Q-only target probability"),
    ):
        means = np.asarray(
            [float(summary_index[(count, outcome)]["estimate"]) for count in COMBINED_COUNTS]
        )
        lows = np.asarray(
            [float(summary_index[(count, outcome)]["ci_low"]) for count in COMBINED_COUNTS]
        )
        highs = np.asarray(
            [float(summary_index[(count, outcome)]["ci_high"]) for count in COMBINED_COUNTS]
        )
        axis_a.errorbar(
            x,
            means,
            yerr=np.vstack((means - lows, highs - means)),
            color=color,
            marker=marker,
            ms=4.8,
            lw=1.7,
            elinewidth=1.0,
            capsize=2.0,
            label=label,
            zorder=3,
        )
    axis_a.axhline(0.5, color="#AAB2BD", lw=0.8, ls=(0, (3, 3)), zorder=1)
    axis_a.annotate(
        f"interpolated 50% crossing: {float(refinement['linearly_interpolated_isotonic_m50']):.1f} rows",
        xy=(2.75, 0.50),
        xytext=(4.8, 0.63),
        arrowprops={"arrowstyle": "-", "color": GRAY, "lw": 0.8},
        color=GRAY,
        fontsize=7.8,
        ha="left",
    )
    axis_a.set_ylim(-0.04, 1.04)
    axis_a.set_xticks(x, [str(count) for count in COMBINED_COUNTS])
    axis_a.set_ylabel("Fraction or mean probability")
    axis_a.set_xlabel("Q-only counterexamples in 10,000 training rows")
    axis_a.grid(axis="y", color=LIGHT_GRAY, lw=0.65)
    axis_a.set_title("A handful of counterexamples breaks the support-perfect gate", loc="left", pad=9)
    axis_a.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=2,
        columnspacing=1.5,
        handlelength=2.0,
    )

    taxonomy_groups = (
        ("stable gate 113", PURPLE, "stable gate"),
        ("pure Y", TEAL, "intended rule (pure Y)"),
        ("raw-codeword patch", ORANGE, "raw-codeword patch"),
        ("other", "#B7BEC8", "other / unstable"),
    )
    bottoms = np.zeros(len(COMBINED_COUNTS), dtype=np.float64)
    for taxonomy, color, label in taxonomy_groups:
        values: list[float] = []
        for count in COMBINED_COUNTS:
            subset = [
                row for row in endpoints if int(row["q_only_error_count"]) == count
            ]
            if taxonomy == "other":
                number = sum(
                    str(row["taxonomy"])
                    not in {"stable gate 113", "pure Y", "raw-codeword patch"}
                    for row in subset
                )
            else:
                number = sum(str(row["taxonomy"]) == taxonomy for row in subset)
            values.append(number / len(ALL_SEEDS))
        heights = np.asarray(values)
        axis_b.bar(
            x,
            heights,
            bottom=bottoms,
            width=0.72,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            label=label,
            zorder=2,
        )
        bottoms += heights
    axis_b.set_ylim(0, 1.0)
    axis_b.set_xticks(x, [str(count) for count in COMBINED_COUNTS])
    axis_b.set_ylabel("Fraction of 20 training seeds")
    axis_b.set_xlabel("Q-only counterexamples in 10,000 training rows")
    axis_b.grid(axis="y", color=LIGHT_GRAY, lw=0.65, zorder=0)
    axis_b.set_title("The endpoint shifts from a gate to the intended rule", loc="left", pad=9)
    axis_b.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=2,
        columnspacing=1.35,
        handlelength=1.5,
    )

    heat_steps = (0, 3, 9, 33, 117, 418, 1_491, 3_444, 8_192)
    heat = np.empty((len(COMBINED_COUNTS), len(heat_steps)), dtype=np.float64)
    for row_index, count in enumerate(COMBINED_COUNTS):
        for column_index, step in enumerate(heat_steps):
            subset = [
                row
                for row in checkpoints
                if int(row["q_only_error_count"]) == count and int(row["step"]) == step
            ]
            if len(subset) != len(ALL_SEEDS):
                raise RuntimeError(
                    f"figure heatmap m={count}, step={step} has {len(subset)} seeds"
                )
            heat[row_index, column_index] = float(
                np.mean([float(row["q_only_target_probability"]) for row in subset])
            )
    image = axis_c.imshow(
        heat,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        vmin=0,
        vmax=1,
        cmap="viridis",
    )
    axis_c.set_yticks(np.arange(len(COMBINED_COUNTS)), [str(count) for count in COMBINED_COUNTS])
    axis_c.set_xticks(
        np.arange(len(heat_steps)),
        ["0", "3", "9", "33", "117", "418", "1.5k", "3.4k", "8.2k"],
        rotation=0,
    )
    axis_c.set_ylabel("Q-only counterexamples")
    axis_c.set_xlabel("Training step")
    axis_c.set_title("Counterevidence changes the learning trajectory early", loc="left", pad=9)
    colorbar = figure.colorbar(image, ax=axis_c, fraction=0.046, pad=0.025)
    colorbar.set_label("Mean Q-only target probability", rotation=270, labelpad=14)

    curve_colors = {
        2: "#4C78A8",
        4: "#7A5195",
        6: "#D55E00",
        8: "#E6A700",
        10: "#009E73",
    }
    for count, color in curve_colors.items():
        exposure, probability = _mean_checkpoint_curve(checkpoints, count)
        axis_d.plot(
            exposure,
            probability,
            color=color,
            lw=1.7,
            marker="o",
            markevery=(0, 5),
            ms=2.8,
            label=f"m={count}",
        )
    axis_d.set_xscale("symlog", linthresh=1.0)
    axis_d.set_xlim(-0.15, 2_300)
    axis_d.set_ylim(-0.04, 1.04)
    axis_d.set_xticks((0, 1, 3, 10, 30, 100, 300, 1_000, 2_000))
    axis_d.set_xticklabels(("0", "1", "3", "10", "30", "100", "300", "1k", "2k"))
    axis_d.set_xlabel("Cumulative presentations of Q-only rows")
    axis_d.set_ylabel("Mean Q-only target probability")
    axis_d.grid(color=LIGHT_GRAY, lw=0.65)
    axis_d.set_title("The response depends on more than cumulative exposure", loc="left", pad=9)
    axis_d.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=5,
        columnspacing=1.0,
        handlelength=1.7,
    )

    for label, axis in zip("abcd", axes, strict=True):
        panel_label(axis, label)
    figure.suptitle(
        "A few disambiguating examples can redirect a learned conditional goal",
        x=0.075,
        y=0.975,
        ha="left",
        fontsize=13.0,
        fontweight=600,
        color=INK,
    )
    figure.text(
        0.075,
        0.925,
        "Paired support completion · 20 fixed training seeds per dose · intervals resample seeds; stacks show observed fractions",
        ha="left",
        va="center",
        fontsize=8.4,
        color=GRAY,
    )
    figure.text(
        0.965,
        0.025,
        "Adaptive exploratory follow-up; refinement doses 4, 6, and 8 were selected by a frozen trigger",
        ha="right",
        va="bottom",
        fontsize=7.6,
        color=GRAY,
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / "fig16_support_completion.pdf", bbox_inches="tight")
    figure.savefig(
        FIGURES / "fig16_support_completion.png",
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(figure)


def selected_findings(
    endpoints: Sequence[Mapping[str, Any]],
    dose_summary: Sequence[Mapping[str, Any]],
    contrasts: Sequence[Mapping[str, Any]],
    transitions: Sequence[Mapping[str, Any]],
    independent_summary: Sequence[Mapping[str, Any]],
    refinement: Mapping[str, Any],
) -> dict[str, Any]:
    summary_index = _dose_summary_index(dose_summary)
    taxonomy_by_count: dict[str, dict[str, int]] = {}
    for count in COMBINED_COUNTS:
        taxonomy_by_count[str(count)] = dict(
            Counter(
                str(row["taxonomy"])
                for row in endpoints
                if int(row["q_only_error_count"]) == count
            )
        )
    key_outcomes = (
        "stable_gate",
        "q_only_diagnostic_accuracy",
        "q_only_factorial_accuracy",
        "q_only_target_probability",
        "q_only_seen_target_probability",
        "q_only_unseen_target_probability",
        "seen_minus_unseen_probability",
        "common_iid_target_accuracy",
        "active_codeword_coverage",
    )
    dose_response: dict[str, dict[str, Any]] = {}
    for count in COMBINED_COUNTS:
        cell: dict[str, Any] = {}
        for outcome in key_outcomes:
            row = summary_index.get((count, outcome))
            if row is not None:
                cell[outcome] = {
                    "estimate": row["estimate"],
                    "seed_cluster_interval": [row["ci_low"], row["ci_high"]],
                    "seeds": row["seed_clusters"],
                }
        dose_response[str(count)] = cell
    key_comparisons = [
        dict(row)
        for row in contrasts
        if str(row["contrast_kind"]) == "paired_difference"
        and (
            (int(row["left_count"]), int(row["right_count"]))
            in {(0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 50)}
        )
        and str(row["outcome"])
        in {"stable_gate", "q_only_target_probability", "common_iid_target_accuracy"}
    ]
    per_seed_transition_summary = {
        "stable_gate_reappearance_seeds": len(
            {
                int(row["seed"])
                for row in transitions
                if int(row["gate_reversals_across_all_counts"]) > 0
            }
        ),
        "any_probability_reversal_seeds": len(
            {
                int(row["seed"])
                for row in transitions
                if int(row["probability_reversals_across_all_counts"]) > 0
            }
        ),
        "material_probability_reversal_seeds": len(
            {
                int(row["seed"])
                for row in transitions
                if int(row["material_probability_reversals_across_all_counts"]) > 0
            }
        ),
        "first_stable_gate_loss_count": dict(
            Counter(
                str(row["first_stable_gate_loss_count"])
                for row in transitions
                if int(row["left_count"]) == COMBINED_COUNTS[0]
            )
        ),
    }
    return {
        "stable_gate_definition": (
            "Boolean signature 113 and factorial tuple consistency >= 0.90 at step 8192"
        ),
        "dose_response": dose_response,
        "endpoint_taxonomy_counts": taxonomy_by_count,
        "frozen_refinement_result": dict(refinement),
        "selected_paired_seed_contrasts": key_comparisons,
        "per_seed_monotonicity_diagnostics": per_seed_transition_summary,
        "count_matched_independent_reference": list(independent_summary),
    }


def _final_source_fingerprint_audit(
    base_audit: Mapping[str, Any],
    refinement_audit: Mapping[str, Any],
    pairing_audit: Mapping[str, Any],
    bridge_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Finalize drift disposition only after every relevant exact audit passes."""

    panel_fingerprints = {
        str(value)
        for audit in (base_audit, refinement_audit)
        for value in cast(Sequence[Any], audit["implementation_fingerprints"])
    }
    if len(panel_fingerprints) != 1:
        raise RuntimeError(
            "E16 primary/refinement artifact fingerprints are not uniform: "
            f"{sorted(panel_fingerprints)}"
        )
    current_fingerprints = {
        str(audit["current_source_fingerprint"])
        for audit in (base_audit, refinement_audit)
    }
    current_fingerprints.add(
        str(implementation_provenance(REPO)["implementation_fingerprint"])
    )
    if len(current_fingerprints) != 1:
        raise RuntimeError(
            "current source fingerprint changed during E16 analysis: "
            f"{sorted(current_fingerprints)}"
        )
    required_exact_audits = {
        "primary_grid_and_measurements": bool(
            base_audit["strict_grid_and_measurement_audit_passed"]
        ),
        "refinement_grid_and_measurements": bool(
            refinement_audit["strict_grid_and_measurement_audit_passed"]
        ),
        "current_source_dataset_regeneration_and_metric_replay": bool(
            pairing_audit["current_source_regeneration_audit_passed"]
        )
        and int(pairing_audit["unchanged_array_and_panel_checks"]) > 0
        and int(pairing_audit["exact_design_and_panel_metric_checks"]) > 0
        and int(pairing_audit["exact_exposure_metric_checks"]) > 0,
        "archived_e15_exact_metric_and_summary_bridge": bool(
            bridge_audit["m0_matches_archived_e15_nested_exactly"]
        )
        and int(bridge_audit["m0_exact_metric_bridges"]) == EXPECTED_E15_BRIDGES,
    }
    if not all(required_exact_audits.values()):
        raise RuntimeError(
            "current-worktree drift cannot be dispositioned because a required exact "
            f"audit did not pass: {required_exact_audits}"
        )
    artifact_fingerprint = next(iter(panel_fingerprints))
    current_fingerprint = next(iter(current_fingerprints))
    drift_detected = artifact_fingerprint != current_fingerprint
    return {
        "artifact_fingerprint_uniformity_within_and_across_e16_panels_passed": True,
        "artifact_implementation_fingerprint": artifact_fingerprint,
        "current_source_fingerprint": current_fingerprint,
        "current_source_matches_artifact": not drift_detected,
        "current_worktree_drift_detected": drift_detected,
        "current_worktree_drift_fatal": False,
        "current_worktree_drift_disposition": (
            "reported_nonfatal_after_exact_relevant_regeneration_metric_and_bridge_"
            "audits_passed"
            if drift_detected
            else "current_source_matches_artifacts"
        ),
        "required_exact_audits": required_exact_audits,
    }


def write_outputs(
    *,
    checkpoints: Sequence[Mapping[str, Any]],
    endpoints: Sequence[Mapping[str, Any]],
    dose_summary: Sequence[Mapping[str, Any]],
    contrasts: Sequence[Mapping[str, Any]],
    transitions: Sequence[Mapping[str, Any]],
    pairing_rows: Sequence[Mapping[str, Any]],
    independent_rows: Sequence[Mapping[str, Any]],
    independent_summary: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
    refinement: Mapping[str, Any],
    base_audit: Mapping[str, Any],
    refinement_audit: Mapping[str, Any],
    pairing_audit: Mapping[str, Any],
    bridge_audit: Mapping[str, Any],
    source_fingerprint_audit: Mapping[str, Any],
    artifacts: Path,
    refinement_artifacts: Path,
    e15_artifacts: Path,
) -> Mapping[str, Any]:
    paths = {
        "checkpoint_csv": DERIVED / "e16_checkpoint_dynamics.csv",
        "endpoint_csv": DERIVED / "e16_run_endpoints.csv",
        "dose_summary_csv": DERIVED / "e16_dose_summary.csv",
        "paired_contrast_csv": DERIVED / "e16_paired_contrasts.csv",
        "gate_transition_csv": DERIVED / "e16_gate_transitions.csv",
        "pairing_audit_csv": DERIVED / "e16_pairing_audit.csv",
        "independent_reference_csv": DERIVED / "e16_independent_reference.csv",
        "independent_reference_summary_csv": DERIVED
        / "e16_independent_reference_summary.csv",
    }
    for name, path in paths.items():
        rows: Sequence[Mapping[str, Any]]
        if name == "checkpoint_csv":
            rows = checkpoints
        elif name == "endpoint_csv":
            rows = endpoints
        elif name == "dose_summary_csv":
            rows = dose_summary
        elif name == "paired_contrast_csv":
            rows = contrasts
        elif name == "gate_transition_csv":
            rows = transitions
        elif name == "pairing_audit_csv":
            rows = pairing_rows
        elif name == "independent_reference_csv":
            rows = independent_rows
        else:
            rows = independent_summary
        write_csv(path, rows)
    findings = selected_findings(
        endpoints,
        dose_summary,
        contrasts,
        transitions,
        independent_summary,
        refinement,
    )
    grid_audit = {
        "base": dict(base_audit),
        "triggered_refinement": dict(refinement_audit),
        "regenerated_dataset_and_pairing": dict(pairing_audit),
        "source_fingerprint": dict(source_fingerprint_audit),
    }
    report = {
        "experiment": "E16_conditional_support_completion",
        "inference_status": INFERENCE_STATUS,
        "artifacts_root": str(artifacts.resolve()),
        "refinement_artifacts_root": str(refinement_artifacts.resolve()),
        "e15_artifacts_root": str(e15_artifacts.resolve()),
        "design": {
            "registered_primary_counts": list(SUPPORT_COUNTS),
            "training_seeds": list(ALL_SEEDS),
            "frozen_refinement_decision": dict(decision),
            "conditionally_run_refinement_counts": list(REFINEMENT_COUNTS),
            "combined_counts": list(COMBINED_COUNTS),
        },
        "audit": grid_audit,
        "e15_bridge_audit": dict(bridge_audit),
        "findings": findings,
        "outputs": {
            **{name: str(path.resolve()) for name, path in paths.items()},
            "grid_audit_json": str((DERIVED / "e16_grid_audit.json").resolve()),
            "bridge_audit_json": str((DERIVED / "e16_e15_bridge_audit.json").resolve()),
            "refinement_decision_json": str(
                (DERIVED / "e16_refinement_decision.json").resolve()
            ),
            "refinement_results_json": str(
                (DERIVED / "e16_refinement_results.json").resolve()
            ),
            "figure_pdf": str((FIGURES / "fig16_support_completion.pdf").resolve()),
            "figure_png": str((FIGURES / "fig16_support_completion.png").resolve()),
        },
    }
    write_json(DERIVED / "e16_grid_audit.json", grid_audit)
    write_json(DERIVED / "e16_e15_bridge_audit.json", bridge_audit)
    write_json(DERIVED / "e16_refinement_decision.json", decision)
    write_json(DERIVED / "e16_refinement_results.json", refinement)
    write_json(DERIVED / "e16_analysis.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=DEFAULT_ARTIFACTS,
        help=f"E16 registered artifact root (default: {DEFAULT_ARTIFACTS})",
    )
    parser.add_argument(
        "--refinement-artifacts",
        type=Path,
        default=DEFAULT_REFINEMENT_ARTIFACTS,
        help=f"E16 triggered-refinement artifact root (default: {DEFAULT_REFINEMENT_ARTIFACTS})",
    )
    parser.add_argument(
        "--e15-artifacts",
        type=Path,
        default=DEFAULT_E15_ARTIFACTS,
        help=f"archived E15 artifact root (default: {DEFAULT_E15_ARTIFACTS})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # The primary grid and exact E15 bridge are audited before the adaptive rule
    # is evaluated. Refinement artifacts cannot influence the trigger.
    base_runs, base_audit = load_runs(args.artifacts, counts=SUPPORT_COUNTS, panel="primary")
    bridge_audit, independent_rows = e15_bridge_audit(base_runs, args.e15_artifacts)
    base_checkpoints = checkpoint_rows(base_runs)
    base_endpoints = run_endpoint_rows(base_checkpoints, base_runs)
    decision = refinement_decision(base_endpoints)
    if not bool(decision["triggered"]):
        raise RuntimeError(
            "the frozen E16 refinement rule did not trigger; refusing to inspect refinement artifacts"
        )
    if tuple(decision["rounded_quarter_point_candidates"]) != REFINEMENT_COUNTS:
        raise RuntimeError(
            "frozen refinement counts disagree with the implemented conditional panel: "
            f"{decision['rounded_quarter_point_candidates']} != {list(REFINEMENT_COUNTS)}"
        )

    refinement_runs, refinement_audit = load_runs(
        args.refinement_artifacts,
        counts=REFINEMENT_COUNTS,
        panel="triggered_refinement",
    )
    runs = sorted((*base_runs, *refinement_runs), key=lambda run: (run.seed, run.m))
    pairing_rows, pairing_audit = dataset_pairing_audit(runs, counts=COMBINED_COUNTS)
    checkpoints = checkpoint_rows(runs)
    endpoints = run_endpoint_rows(checkpoints, runs)
    dose_summary = dose_summaries(endpoints, counts=COMBINED_COUNTS)
    contrasts = paired_contrasts(endpoints)
    transitions = gate_transition_rows(endpoints, counts=COMBINED_COUNTS)
    refinement = refinement_results(endpoints)
    independent_summary = independent_reference_summary(independent_rows)
    source_fingerprint_audit = _final_source_fingerprint_audit(
        base_audit,
        refinement_audit,
        pairing_audit,
        bridge_audit,
    )

    report = write_outputs(
        checkpoints=checkpoints,
        endpoints=endpoints,
        dose_summary=dose_summary,
        contrasts=contrasts,
        transitions=transitions,
        pairing_rows=pairing_rows,
        independent_rows=independent_rows,
        independent_summary=independent_summary,
        decision=decision,
        refinement=refinement,
        base_audit=base_audit,
        refinement_audit=refinement_audit,
        pairing_audit=pairing_audit,
        bridge_audit=bridge_audit,
        source_fingerprint_audit=source_fingerprint_audit,
        artifacts=args.artifacts,
        refinement_artifacts=args.refinement_artifacts,
        e15_artifacts=args.e15_artifacts,
    )
    make_figure(checkpoints, endpoints, dose_summary, refinement)
    print(
        "E16 strict analysis passed: "
        f"{base_audit['complete_runs']} primary + "
        f"{refinement_audit['complete_runs']} triggered-refinement runs"
    )
    print(
        "  metric records audited: "
        f"{int(base_audit['metric_records']) + int(refinement_audit['metric_records']):,}"
    )
    print(
        "  stable-gate counts at m=2,4,6,8,10: "
        f"{refinement['stable_gate_counts']}"
    )
    print(f"  interpolated m50: {refinement['linearly_interpolated_isotonic_m50']}")
    if source_fingerprint_audit["current_worktree_drift_detected"]:
        print(
            "  current-worktree drift: detected and reported nonfatal after exact "
            "regeneration, metric, and archived-bridge audits"
        )
    print(f"  report: {DERIVED / 'e16_analysis.json'}")
    print(f"  figure: {FIGURES / 'fig16_support_completion.pdf'}")
    if not report:
        raise RuntimeError("internal error: empty E16 report")


if __name__ == "__main__":
    main()
