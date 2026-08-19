#!/usr/bin/env python3
"""Strict audit and analysis for the adaptive E15 multi-goal dynamics panel.

E15 is an explicitly exploratory bridge from E12.  This script is deliberately
all-or-nothing: it accepts exactly the frozen 12-cell by 20-seed grid, checks
artifact identities and the complete checkpoint measurement lattice, and then
requires exact equality at optimizer step 2,048 with the matching archived E12
runs for the original ten seeds.

The training seed is the independent unit.  Factor contrasts are paired within
seed and nuisance cell before deterministic 4,000-draw seed bootstraps.  The ten
archived bridge seeds and ten fresh seeds are always shown separately before a
pooled summary is reported.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import warnings
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, TypeAlias

os.environ.setdefault("MPLCONFIGDIR", "/tmp/forkworld-e15-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/forkworld-e15-xdg")
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
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_ARTIFACTS = REPO / "artifacts-e15"
DEFAULT_E12_ARTIFACTS = REPO / "artifacts-followups"
DERIVED = HERE / "derived"
FIGURES = HERE / "figures"

ORIGINAL_SEEDS = (11, 23, 37, 41, 53, 67, 71, 83, 97, 101)
FRESH_SEEDS = (103, 107, 109, 113, 127, 131, 137, 139, 149, 151)
ALL_SEEDS = ORIGINAL_SEEDS + FRESH_SEEDS
Q_Q_LEVELS = (0.90, 0.95, 0.99)
K_Y_LEVELS = (3, 5)
OVERLAPS = ("independent", "nested")
GOALS = ("P", "Q", "Y")
LAYERS = ("raw", "first_hidden", "final_hidden")
TUPLE_KEYS = ("---", "--+", "-+-", "-++", "+--", "+-+", "++-", "+++")

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
EXPECTED_RUNS = 240
EXPECTED_BRIDGES = 120
BOOTSTRAP_DRAWS = 4_000
ARTIFACT_SCHEMA_VERSION = 1
SOURCE_FINGERPRINT_SCHEMA_VERSION = 1
NON_SCIENTIFIC_RUN_FIELDS = frozenset({"output_root", "resume", "seeds"})
INFERENCE_STATUS = "adaptive_exploratory_posthoc_with_fresh_seed_replication"

BEHAVIOR_THRESHOLD = 0.90
CAUSAL_THRESHOLD = 0.90
SOLO_THRESHOLD = 0.95
GENERIC_PROBE_THRESHOLD = 0.95
SELECTIVE_PROBE_ACCURACY = 0.90
SELECTIVE_PROBE_MARGIN = 0.20
PURE_MARGIN = 0.10
PERSISTENCE = 2
TAXONOMY_CONSISTENCY = 0.90

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
    digest = hashlib.blake2b(digest_size=8, person=b"forke15")
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
    run = dict(scientific.get("run", {}))
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
    return value


def load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as error:
        raise RuntimeError(f"cannot read YAML mapping {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise RuntimeError(f"expected YAML mapping: {path}")
    return value


def expect_equal(
    errors: list[str], run_id: str, actual: Any, expected: Any, label: str
) -> None:
    if actual != expected:
        errors.append(f"{run_id}: {label}={actual!r}, expected {expected!r}")


def expected_grid() -> set[tuple[float, int, str, int]]:
    return {
        (rounded(q_q), k_y, overlap, seed)
        for q_q in Q_Q_LEVELS
        for k_y in K_Y_LEVELS
        for overlap in OVERLAPS
        for seed in ALL_SEEDS
    }


def _metric_is_needed(stage: str, split: str, intervention: str, metric: str) -> bool:
    if stage in {"P_calibration", "Q_calibration", "Y_calibration"}:
        return split == f"{stage}_iid" and metric == "target_accuracy"
    if stage == "competition_behavior":
        return split == "factorial_eval" and metric in {"rho_p", "rho_q", "rho_y_code"}
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
    return False


@dataclass(frozen=True)
class Run:
    path: Path
    config: Mapping[str, Any]
    summary: Mapping[str, Any]
    metadata: Mapping[str, Any]
    metrics: Mapping[MetricKey, Mapping[int, float]]
    metric_line_count: int

    @property
    def seed(self) -> int:
        return int(self.summary["seed"])

    @property
    def q_q(self) -> float:
        return rounded(get(self.config, "h12.q_q"))

    @property
    def k_y(self) -> int:
        return int(get(self.config, "h12.k_y"))

    @property
    def overlap(self) -> str:
        return str(get(self.config, "h12.error_structure"))

    @property
    def cohort(self) -> str:
        return "original" if self.seed in ORIGINAL_SEEDS else "fresh"

    def series(
        self, stage: str, split: str, metric: str, intervention: str = "none"
    ) -> Mapping[int, float]:
        key = (stage, split, intervention, metric)
        if key not in self.metrics:
            raise RuntimeError(f"{self.path.name}: missing metric series {key}")
        return self.metrics[key]

    def value(
        self,
        stage: str,
        split: str,
        metric: str,
        step: int,
        intervention: str = "none",
    ) -> float:
        series = self.series(stage, split, metric, intervention)
        if step not in series:
            raise RuntimeError(
                f"{self.path.name}: missing {stage}/{split}/{intervention}/{metric} at {step}"
            )
        return float(series[step])


def _load_metric_index(
    path: Path, run_id: str, seed: int, errors: list[str]
) -> tuple[dict[MetricKey, dict[int, float]], int]:
    result: dict[MetricKey, dict[int, float]] = defaultdict(dict)
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
            if int(record.get("seed", -1)) != seed:
                errors.append(f"{run_id}: metric line {line_number} seed mismatch")
            if record.get("experiment") != "h12":
                errors.append(f"{run_id}: metric line {line_number} experiment mismatch")
            stage = str(record.get("stage"))
            split = str(record.get("split"))
            intervention = str(record.get("intervention", "none"))
            metric = str(record.get("metric"))
            if not _metric_is_needed(stage, split, intervention, metric):
                continue
            try:
                step = int(record["global_step"])
                stage_step = int(record["stage_step"])
                value = float(record["value"])
                n = int(record["n"])
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
            expected_n = 4_000 if stage.endswith("_calibration") else 4_096
            if n != expected_n:
                errors.append(
                    f"{run_id}: selected metric line {line_number} n={n}, expected {expected_n}"
                )
            if stage.startswith("competition_") and step not in COMPETITION_STEPS:
                errors.append(f"{run_id}: unexpected competition checkpoint {step}")
            if stage.endswith("_calibration") and step not in CALIBRATION_STEPS:
                errors.append(f"{run_id}: unexpected calibration checkpoint {step}")
            key = (stage, split, intervention, metric)
            if step in result[key]:
                errors.append(f"{run_id}: duplicate selected metric {key} at step {step}")
            result[key][step] = value
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


def validate_run(run: Run, errors: list[str]) -> None:
    run_id = run.path.name
    fixed = {
        "schema_version": 1,
        "experiment.hypothesis": "h12",
        "experiment.name": "multigoal_temporal_bridge",
        "experiment.mode": "adaptive_multigoal_dynamics",
        "run.device": "cpu",
        "run.task_levels": ["choice"],
        "run.save_checkpoints": False,
        "evaluation.save_predictions": False,
        "evaluation.bootstrap_samples": BOOTSTRAP_DRAWS,
        "evaluation.persistence": PERSISTENCE,
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
        "h12.q_p": 0.90,
        "h12.k_q": 2,
        "h12.max_k_q": 3,
        "h12.max_k_y": 5,
        "h12.calibration_steps": 2_048,
        "h12.competition_steps": 8_192,
        "h12.probe_train_n": 2_048,
        "h12.probe_eval_n": 4_096,
        "h12.probe_ridge": 0.001,
        "h12.truth_table_control_seed": 1_500_450_271,
        "h12.bridge_step": 2_048,
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
    expect_equal(errors, run_id, run.summary.get("hypothesis"), "h12", "summary hypothesis")
    expect_equal(
        errors,
        run_id,
        run.summary.get("condition"),
        "adaptive_multigoal_dynamics",
        "summary condition",
    )
    expect_equal(
        errors,
        run_id,
        run.summary.get("design_status"),
        "adaptive_posthoc_with_fresh_seed_replication",
        "design status",
    )
    expect_equal(
        errors,
        run_id,
        run.summary.get("seed_cohort"),
        "bridge" if run.cohort == "original" else "fresh",
        "summary seed cohort",
    )

    summary_fixed = {
        "data.n_train": 10_000,
        "data.n_validation": 4_000,
        "data.n_eval_per_legacy_panel": 10_000,
        "data.probe_train_n": 2_048,
        "data.probe_eval_n": 4_096,
        "data.q_p": 0.90,
        "data.k_q": 2,
        "data.max_k_q": 3,
        "data.max_k_y": 5,
        "data.state_dim": 8,
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
        "bridge.validation_rule": "exact archived discrete diagnostics and intervention summaries",
    }
    for path, expected in summary_fixed.items():
        expect_equal(errors, run_id, get(run.summary, path), expected, f"summary.{path}")
    expect_equal(errors, run_id, rounded(get(run.summary, "data.q_q")), run.q_q, "summary q_q")
    expect_equal(errors, run_id, int(get(run.summary, "data.k_y")), run.k_y, "summary k_y")
    expect_equal(
        errors, run_id, str(get(run.summary, "data.error_structure")), run.overlap, "summary overlap"
    )
    candidate_counts_eval = get(run.summary, "data.factorial_candidate_counts_eval")
    candidate_counts_train = get(run.summary, "data.factorial_candidate_counts_train")
    expect_equal(
        errors,
        run_id,
        candidate_counts_eval,
        {str(index): 512 for index in range(8)},
        "factorial eval counts",
    )
    expect_equal(
        errors,
        run_id,
        candidate_counts_train,
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
            reports_iter = list(reports.values())
        else:
            reports_iter = [reports]
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
        _require_steps(
            run,
            errors,
            (f"{goal}_calibration", f"{goal}_calibration_iid", "none", "target_accuracy"),
            CALIBRATION_STEPS,
        )
    for metric in ("rho_p", "rho_q", "rho_y_code"):
        _require_steps(
            run,
            errors,
            ("competition_behavior", "factorial_eval", "none", metric),
            COMPETITION_STEPS,
        )
    for metric in (
        "tuple_consistency",
        "codeword_consistency",
        "codeword_consistency_row_weighted",
        "nuisance_consistency",
        "n_codeword_groups",
        "boolean_signature_int",
        "sign_inversion_symmetry",
    ):
        _require_steps(
            run,
            errors,
            ("competition_truth_table", "factorial_eval", "none", metric),
            COMPETITION_STEPS,
        )
    for tuple_key in TUPLE_KEYS:
        for suffix in ("positive_rate", "mean_positive_probability", "modal_action"):
            _require_steps(
                run,
                errors,
                (
                    "competition_truth_table",
                    "factorial_eval",
                    "none",
                    f"tuples__{tuple_key}__{suffix}",
                ),
                COMPETITION_STEPS,
            )
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
            _require_steps(
                run,
                errors,
                (
                    "competition_probe",
                    "factorial_probe",
                    "none",
                    f"representations__{layer}__heldout_accuracy__{label}",
                ),
                COMPETITION_STEPS,
            )
    for metric, expected in {
        "raw_codeword_count": 2 ** (1 + 2 + run.k_y),
        "truth_table_control_positive_fraction": 0.5,
        "n_train": 2_048,
        "n_heldout": 4_096,
        "n_labels": 8,
    }.items():
        _require_steps(
            run,
            errors,
            ("competition_probe", "factorial_probe", "none", metric),
            COMPETITION_STEPS,
        )
        values = run.metrics.get(("competition_probe", "factorial_probe", "none", metric), {})
        if any(value != expected for value in values.values()):
            errors.append(f"{run_id}: probe invariant {metric} differs from {expected}")
    expected_components = (
        "flip_P",
        "flip_Q_1",
        "flip_Q_2",
        *(f"flip_Y_{index}" for index in range(1, run.k_y + 1)),
    )
    for intervention in (*GOALS, *expected_components):
        for metric in ("c_hard", "c_prob", "causal_score", "hard_abs_change"):
            _require_steps(
                run,
                errors,
                ("competition_causal", "factorial_eval", intervention, metric),
                COMPETITION_STEPS,
            )
    for goal in GOALS:
        _require_steps(
            run,
            errors,
            ("competition_causal", "factorial_eval", goal, "hard_abs_share"),
            COMPETITION_STEPS,
        )


def load_runs(root: Path) -> tuple[list[Run], dict[str, Any]]:
    directory = root / "h12" / "multigoal_temporal_bridge"
    if not directory.is_dir():
        raise RuntimeError(f"missing E15 artifact directory: {directory}")
    config_paths = sorted(directory.glob("*/resolved_config.yaml"))
    materialized = {path.parent for path in config_paths}
    if len(materialized) != EXPECTED_RUNS:
        raise RuntimeError(
            f"E15 hard audit failed: {len(materialized)}/{EXPECTED_RUNS} materialized runs"
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
        seed = int(summary.get("seed", -1))
        if int(get(config, "seed", -2)) != seed:
            errors.append(f"{run_dir.name}: config/summary seed mismatch")
        if metadata.get("run_id") != run_dir.name or int(metadata.get("seed", -2)) != seed:
            errors.append(f"{run_dir.name}: metadata identity mismatch")
        if expected_artifact_run_id(config, metadata) != run_dir.name:
            errors.append(f"{run_dir.name}: artifact identity does not reconstruct")
        metric_index, metric_line_count = _load_metric_index(
            run_dir / "metrics.jsonl", run_dir.name, seed, errors
        )
        run = Run(
            path=run_dir,
            config=config,
            summary=summary,
            metadata=metadata,
            metrics=metric_index,
            metric_line_count=metric_line_count,
        )
        validate_run(run, errors)
        runs.append(run)

    orphan_artifacts = [
        path.parent.name
        for pattern in ("*/COMPLETE", "*/summary.json", "*/status.json")
        for path in directory.glob(pattern)
        if path.parent not in materialized
    ]
    if orphan_artifacts:
        errors.append(f"{len(orphan_artifacts)} artifacts lack resolved configurations")
    keys = [(run.q_q, run.k_y, run.overlap, run.seed) for run in runs]
    duplicates = [key for key, count in Counter(keys).items() if count > 1]
    unexpected = set(keys) - expected_grid()
    missing_keys = expected_grid() - set(keys)
    if duplicates or unexpected or missing_keys:
        errors.append(
            f"grid has {len(duplicates)} duplicates, {len(unexpected)} unexpected, "
            f"and {len(missing_keys)} missing keys"
        )
    cell_counts = Counter((run.q_q, run.k_y, run.overlap) for run in runs)
    if len(cell_counts) != 12 or any(count != 20 for count in cell_counts.values()):
        errors.append(f"invalid 12-cell replication counts: {dict(cell_counts)}")

    fingerprints_raw = [
        get(run.metadata, "implementation.implementation_fingerprint") for run in runs
    ]
    fingerprints = {value for value in fingerprints_raw if isinstance(value, str)}
    if len(fingerprints) != 1:
        errors.append(f"expected one implementation fingerprint, observed {sorted(fingerprints)}")
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
        preview = "\n".join(f"  - {error}" for error in errors[:80])
        suffix = "" if len(errors) <= 80 else f"\n  ... and {len(errors) - 80} more"
        raise RuntimeError(f"E15 hard audit failed with {len(errors)} issue(s):\n{preview}{suffix}")
    audit = {
        "artifacts_root": str(root.resolve()),
        "materialized_runs": len(materialized),
        "complete_runs": states.get("complete", 0),
        "expected_runs": EXPECTED_RUNS,
        "cells": len(cell_counts),
        "seeds": list(ALL_SEEDS),
        "original_seeds": list(ORIGINAL_SEEDS),
        "fresh_seeds": list(FRESH_SEEDS),
        "implementation_fingerprints": sorted(str(value) for value in fingerprints),
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "source_fingerprint_schema_version": SOURCE_FINGERPRINT_SCHEMA_VERSION,
        "metric_records": sum(run.metric_line_count for run in runs),
        "strict_grid_and_measurement_audit_passed": True,
    }
    return sorted(runs, key=lambda run: (run.seed, run.q_q, run.k_y, run.overlap)), audit


def _e12_key(config: Mapping[str, Any]) -> tuple[float, int, str, int]:
    return (
        rounded(get(config, "h10.q_q")),
        int(get(config, "h10.k_y")),
        str(get(config, "h10.error_structure")),
        int(get(config, "seed")),
    )


def bridge_audit(runs: Sequence[Run], e12_root: Path) -> dict[str, Any]:
    directory = e12_root / "h10" / "competing_goal_frontier"
    if not directory.is_dir():
        raise RuntimeError(f"missing archived E12 artifact directory: {directory}")
    expected_reference_keys = {
        (rounded(q_q), k_y, overlap, seed)
        for q_q in Q_Q_LEVELS
        for k_y in K_Y_LEVELS
        for overlap in OVERLAPS
        for seed in ORIGINAL_SEEDS
    }
    references: dict[tuple[float, int, str, int], tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
    reference_fingerprints: set[str] = set()
    for config_path in sorted(directory.glob("*/resolved_config.yaml")):
        config = load_yaml(config_path)
        if rounded(get(config, "h10.q_p")) != 0.90:
            continue
        if int(get(config, "h10.k_q")) != 2 or int(get(config, "model.width")) != 64:
            continue
        key = _e12_key(config)
        if key not in expected_reference_keys:
            continue
        run_dir = config_path.parent
        for name in ("COMPLETE", "summary.json", "metadata.json", "status.json"):
            if not (run_dir / name).is_file():
                raise RuntimeError(f"E12 bridge reference {run_dir.name} lacks {name}")
        if (run_dir / "COMPLETE").read_text(encoding="utf-8") != "complete\n":
            raise RuntimeError(f"E12 bridge reference {run_dir.name} has invalid marker")
        summary = load_json(run_dir / "summary.json")
        metadata = load_json(run_dir / "metadata.json")
        status = load_json(run_dir / "status.json")
        if status.get("state") != "complete" or status.get("run_id") != run_dir.name:
            raise RuntimeError(f"E12 bridge reference {run_dir.name} has invalid status")
        if metadata.get("run_id") != run_dir.name or metadata.get("seed") != key[-1]:
            raise RuntimeError(f"E12 bridge reference {run_dir.name} has invalid metadata")
        if expected_artifact_run_id(config, metadata) != run_dir.name:
            raise RuntimeError(f"E12 bridge reference {run_dir.name} identity does not reconstruct")
        if summary.get("hypothesis") != "h10" or int(summary.get("seed", -1)) != key[-1]:
            raise RuntimeError(f"E12 bridge reference {run_dir.name} summary identity mismatch")
        fingerprint = get(metadata, "implementation.implementation_fingerprint")
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise RuntimeError(f"E12 bridge reference {run_dir.name} malformed fingerprint")
        reference_fingerprints.add(fingerprint)
        if key in references:
            raise RuntimeError(f"duplicate E12 bridge reference for {key}")
        references[key] = (summary, metadata)
    if set(references) != expected_reference_keys:
        raise RuntimeError(
            f"E12 bridge reference grid mismatch: {len(references)}/{EXPECTED_BRIDGES}"
        )

    checks = 0
    section_checks: Counter[str] = Counter()
    for run in runs:
        if run.seed not in ORIGINAL_SEEDS:
            continue
        key = (run.q_q, run.k_y, run.overlap, run.seed)
        reference, _ = references[key]
        comparisons = {
            "competition_diagnostics": (
                get(run.summary, "bridge.dynamics.diagnostics"),
                get(reference, "final.competition"),
            ),
            "diagnostic_interventions": (
                get(run.summary, "bridge.dynamics.diagnostic_interventions"),
                get(reference, "final.interventions"),
            ),
            "standalone_calibrations": (
                get(run.summary, "final.calibration"),
                get(reference, "final.calibration"),
            ),
            "model_interface_reports": (get(run.summary, "model"), get(reference, "model")),
            "training_error_overlap": (
                get(run.summary, "data.training_overlap"),
                get(reference, "data.training_overlap"),
            ),
        }
        for name, (actual, expected) in comparisons.items():
            if actual != expected:
                raise RuntimeError(f"E15/E12 exact bridge failed for {key}, section {name}")
            section_checks[name] += 1
        checks += 1
    if checks != EXPECTED_BRIDGES or any(
        count != EXPECTED_BRIDGES for count in section_checks.values()
    ):
        raise RuntimeError(f"E15 bridge audit count mismatch: {checks}, {section_checks}")
    return {
        "e12_artifacts_root": str(e12_root.resolve()),
        "reference_runs": len(references),
        "original_seed_bridge_checks": checks,
        "sections_compared_exactly": dict(section_checks),
        "e12_reference_implementation_fingerprints": sorted(reference_fingerprints),
        "e15_step_2048_matches_archived_e12_exactly": True,
        "state_hash_claimed": False,
        "note": "Archived E12 contains no model-state hash; equality covers every shared discrete diagnostic, causal summary, calibration, interface report, and error-overlap field.",
    }


def _signature_bits(signature: int) -> tuple[int, ...]:
    if not 0 <= signature <= 255:
        raise ValueError(f"Boolean signature must be in [0,255], got {signature}")
    return tuple(1 if character == "1" else -1 for character in f"{signature:08b}")


def _candidate_signature(index: int) -> int:
    bits = []
    for key in TUPLE_KEYS:
        bits.append("1" if key[index] == "+" else "0")
    return int("".join(bits), 2)


NAMED_SIGNATURES = {
    0: "constant_negative",
    255: "constant_positive",
    _candidate_signature(0): "pure_P",
    255 - _candidate_signature(0): "inverse_P",
    _candidate_signature(1): "pure_Q",
    255 - _candidate_signature(1): "inverse_Q",
    _candidate_signature(2): "pure_Y",
    255 - _candidate_signature(2): "inverse_Y",
    23: "majority_PQY",
    232: "minority_PQY",
    105: "parity_PQY",
    150: "inverse_parity_PQY",
    # This decision list is perfect on the nested-error training support:
    # Q is always correct when P and Q disagree, while Y resolves their joint
    # errors.  It fails specifically on the Q-only-error tuples absent there.
    113: "nested_support_gate_Y_if_P_equals_Q_else_Q",
}


def _essential_variables(bits: Sequence[int]) -> tuple[str, ...]:
    essential: list[str] = []
    tuples = [tuple(1 if character == "+" else -1 for character in key) for key in TUPLE_KEYS]
    lookup = {values: bits[index] for index, values in enumerate(tuples)}
    for index, goal in enumerate(GOALS):
        changed = False
        for values in tuples:
            if values[index] != -1:
                continue
            flipped = list(values)
            flipped[index] = 1
            if lookup[values] != lookup[tuple(flipped)]:
                changed = True
                break
        if changed:
            essential.append(goal)
    return tuple(essential)


def _is_monotone(bits: Sequence[int]) -> bool:
    tuples = [tuple(1 if character == "+" else -1 for character in key) for key in TUPLE_KEYS]
    lookup = {values: bits[index] for index, values in enumerate(tuples)}
    for values in tuples:
        for index in range(3):
            if values[index] != -1:
                continue
            flipped = list(values)
            flipped[index] = 1
            if lookup[values] > lookup[tuple(flipped)]:
                return False
    return True


def signature_structure(signature: int) -> dict[str, Any]:
    bits = _signature_bits(signature)
    essential = _essential_variables(bits)
    named = NAMED_SIGNATURES.get(signature)
    if named is not None:
        family = named
    elif len(essential) == 1:
        family = "one_candidate_nonstandard"
    elif len(essential) == 2:
        family = "two_candidate_monotone_composite" if _is_monotone(bits) else "two_candidate_interaction"
    elif len(essential) == 3:
        family = "three_candidate_monotone_composite" if _is_monotone(bits) else "three_candidate_interaction"
    else:
        family = "constant"
    named_distances = {
        name: sum(left != right for left, right in zip(bits, _signature_bits(code), strict=True))
        for code, name in NAMED_SIGNATURES.items()
    }
    nearest_distance = min(named_distances.values())
    nearest = sorted(name for name, distance in named_distances.items() if distance == nearest_distance)
    return {
        "signature": signature,
        "signature_bits": "".join("1" if value > 0 else "0" for value in bits),
        "named_rule": named,
        "function_family": family,
        "essential_variables": "+".join(essential) if essential else "none",
        "essential_variable_count": len(essential),
        "monotone_in_candidates": int(_is_monotone(bits)),
        "nearest_named_rules": "+".join(nearest),
        "nearest_named_hamming": nearest_distance,
    }


def _classify_phase(behavior: Mapping[str, float], causal: Mapping[str, float]) -> str:
    candidates: list[str] = []
    for goal in GOALS:
        other = [candidate for candidate in GOALS if candidate != goal]
        if (
            behavior[goal] >= BEHAVIOR_THRESHOLD
            and causal[goal] >= CAUSAL_THRESHOLD
            and behavior[goal] - max(behavior[candidate] for candidate in other) >= PURE_MARGIN
            and causal[goal] - max(causal[candidate] for candidate in other) >= PURE_MARGIN
        ):
            candidates.append(goal)
    return candidates[0] if len(candidates) == 1 else "D"


def _classify_taxonomy(row: Mapping[str, Any]) -> str:
    if float(row["tuple_consistency"]) >= TAXONOMY_CONSISTENCY:
        return f"tuple:{row['function_family']}"
    if float(row["nuisance_consistency"]) >= TAXONOMY_CONSISTENCY:
        return "raw_codeword_patch"
    return "nuisance_or_state_sensitive"


def checkpoint_rows(runs: Sequence[Run]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    behavior_metrics = {"P": "rho_p", "Q": "rho_q", "Y": "rho_y_code"}
    for run in runs:
        for step in COMPETITION_STEPS:
            row: dict[str, Any] = {
                "inference_status": INFERENCE_STATUS,
                "run_id": run.path.name,
                "seed": run.seed,
                "cohort": run.cohort,
                "q_q": run.q_q,
                "k_y": run.k_y,
                "overlap": run.overlap,
                "step": step,
            }
            behavior: dict[str, float] = {}
            causal: dict[str, float] = {}
            for goal in GOALS:
                behavior[goal] = run.value(
                    "competition_behavior",
                    "factorial_eval",
                    behavior_metrics[goal],
                    step,
                )
                causal[goal] = run.value(
                    "competition_causal",
                    "factorial_eval",
                    "causal_score",
                    step,
                    goal,
                )
                row[f"behavior_{goal}"] = behavior[goal]
                row[f"causal_{goal}"] = causal[goal]
                row[f"causal_c_hard_{goal}"] = run.value(
                    "competition_causal", "factorial_eval", "c_hard", step, goal
                )
                row[f"causal_abs_{goal}"] = run.value(
                    "competition_causal", "factorial_eval", "hard_abs_change", step, goal
                )
                row[f"causal_share_{goal}"] = run.value(
                    "competition_causal", "factorial_eval", "hard_abs_share", step, goal
                )
            control_by_layer: dict[str, float] = {}
            for layer in LAYERS:
                control = run.value(
                    "competition_probe",
                    "factorial_probe",
                    f"representations__{layer}__heldout_accuracy__truth_table_control",
                    step,
                )
                control_by_layer[layer] = control
                row[f"probe_{layer}_control"] = control
                row[f"probe_{layer}_control_permuted"] = run.value(
                    "competition_probe",
                    "factorial_probe",
                    f"representations__{layer}__heldout_accuracy__truth_table_control_permuted",
                    step,
                )
                for goal in GOALS:
                    accuracy = run.value(
                        "competition_probe",
                        "factorial_probe",
                        f"representations__{layer}__heldout_accuracy__{goal}",
                        step,
                    )
                    permuted = run.value(
                        "competition_probe",
                        "factorial_probe",
                        f"representations__{layer}__heldout_accuracy__{goal}_permuted",
                        step,
                    )
                    row[f"probe_{layer}_{goal}"] = accuracy
                    row[f"probe_{layer}_{goal}_permuted"] = permuted
                    row[f"probe_{layer}_{goal}_selectivity"] = accuracy - control
            for metric in (
                "tuple_consistency",
                "codeword_consistency",
                "codeword_consistency_row_weighted",
                "nuisance_consistency",
                "sign_inversion_symmetry",
            ):
                row[metric] = run.value(
                    "competition_truth_table", "factorial_eval", metric, step
                )
            signature = round(
                run.value(
                    "competition_truth_table",
                    "factorial_eval",
                    "boolean_signature_int",
                    step,
                )
            )
            row.update(signature_structure(signature))
            for tuple_key in TUPLE_KEYS:
                row[f"tuple_{tuple_key}_positive_rate"] = run.value(
                    "competition_truth_table",
                    "factorial_eval",
                    f"tuples__{tuple_key}__positive_rate",
                    step,
                )
                row[f"tuple_{tuple_key}_positive_probability"] = run.value(
                    "competition_truth_table",
                    "factorial_eval",
                    f"tuples__{tuple_key}__mean_positive_probability",
                    step,
                )
            row["phase"] = _classify_phase(behavior, causal)
            row["taxonomy"] = _classify_taxonomy(row)
            rows.append(row)
    return rows


@dataclass(frozen=True)
class Event:
    observed: bool
    event_step: int | None
    confirmation_step: int | None
    lower_bound: int
    upper_bound: int | None
    censoring: str
    restricted_step: int

    def fields(self, prefix: str) -> dict[str, Any]:
        return {
            f"{prefix}_observed": int(self.observed),
            f"{prefix}_event_step": self.event_step,
            f"{prefix}_confirmation_step": self.confirmation_step,
            f"{prefix}_lower_bound": self.lower_bound,
            f"{prefix}_upper_bound": self.upper_bound,
            f"{prefix}_censoring": self.censoring,
            f"{prefix}_restricted_step": self.restricted_step,
        }


def persistent_event(
    steps: Sequence[int], flags: Sequence[bool], *, horizon: int
) -> Event:
    if len(steps) != len(flags) or len(steps) < PERSISTENCE:
        raise ValueError("persistent event requires aligned steps and at least two observations")
    if tuple(sorted(steps)) != tuple(steps):
        raise ValueError("event steps must be sorted")
    for index in range(len(steps) - 1):
        if flags[index] and flags[index + 1]:
            step = int(steps[index])
            if index == 0:
                return Event(True, step, int(steps[index + 1]), 0, step, "left", step)
            return Event(
                True,
                step,
                int(steps[index + 1]),
                int(steps[index - 1]),
                step,
                "interval",
                step,
            )
    return Event(False, None, None, horizon, None, "right", horizon)


def _compress(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if not result or result[-1] != value:
            result.append(value)
    return result


def _persistent_goal_episodes(phases: Sequence[str]) -> tuple[list[str], list[str]]:
    compressed_all = _compress(phases)
    episodes: list[str] = []
    start = 0
    while start < len(phases):
        end = start + 1
        while end < len(phases) and phases[end] == phases[start]:
            end += 1
        if phases[start] in GOALS and end - start >= PERSISTENCE:
            episodes.append(phases[start])
        start = end
    goal_sequence = _compress(episodes)
    return compressed_all, goal_sequence


def _phase_time_shares(phases: Sequence[str]) -> dict[str, float]:
    x = np.asarray(COMPETITION_STEPS, dtype=np.float64)
    result: dict[str, float] = {}
    for phase in (*GOALS, "D"):
        y = np.asarray([float(value == phase) for value in phases], dtype=np.float64)
        try:
            area = float(np.trapezoid(y, x))
        except AttributeError:  # NumPy 1.26
            area = float(np.trapz(y, x))  # type: ignore[attr-defined]
        result[phase] = area / float(COMPETITION_STEPS[-1])
    return result


def _tuple_target_accuracy(checkpoint: Mapping[str, Any], tuple_key: str) -> float:
    positive_rate = float(checkpoint[f"tuple_{tuple_key}_positive_rate"])
    return positive_rate if tuple_key[-1] == "+" else 1.0 - positive_rate


def run_event_rows(
    runs: Sequence[Run], checkpoints: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    by_run: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in checkpoints:
        by_run[str(row["run_id"])].append(row)
    result: list[dict[str, Any]] = []
    for run in runs:
        rows = sorted(by_run[run.path.name], key=lambda row: int(row["step"]))
        if tuple(int(row["step"]) for row in rows) != COMPETITION_STEPS:
            raise RuntimeError(f"{run.path.name}: checkpoint rows do not match frozen grid")
        output: dict[str, Any] = {
            "inference_status": INFERENCE_STATUS,
            "run_id": run.path.name,
            "seed": run.seed,
            "cohort": run.cohort,
            "q_q": run.q_q,
            "k_y": run.k_y,
            "overlap": run.overlap,
        }
        for goal in GOALS:
            calibration = run.series(
                f"{goal}_calibration", f"{goal}_calibration_iid", "target_accuracy"
            )
            solo_flags = [float(calibration[step]) >= SOLO_THRESHOLD for step in CALIBRATION_STEPS]
            events = {
                "solo": persistent_event(CALIBRATION_STEPS, solo_flags, horizon=2_048),
                "probe_generic": persistent_event(
                    COMPETITION_STEPS,
                    [
                        float(row[f"probe_final_hidden_{goal}"]) >= GENERIC_PROBE_THRESHOLD
                        for row in rows
                    ],
                    horizon=8_192,
                ),
                "probe_selective": persistent_event(
                    COMPETITION_STEPS,
                    [
                        float(row[f"probe_final_hidden_{goal}"]) >= SELECTIVE_PROBE_ACCURACY
                        and float(row[f"probe_final_hidden_{goal}_selectivity"])
                        >= SELECTIVE_PROBE_MARGIN
                        for row in rows
                    ],
                    horizon=8_192,
                ),
                "behavior": persistent_event(
                    COMPETITION_STEPS,
                    [float(row[f"behavior_{goal}"]) >= BEHAVIOR_THRESHOLD for row in rows],
                    horizon=8_192,
                ),
                "causal": persistent_event(
                    COMPETITION_STEPS,
                    [float(row[f"causal_{goal}"]) >= CAUSAL_THRESHOLD for row in rows],
                    horizon=8_192,
                ),
                "pure": persistent_event(
                    COMPETITION_STEPS,
                    [str(row["phase"]) == goal for row in rows],
                    horizon=8_192,
                ),
            }
            for stage, event in events.items():
                output.update(event.fields(f"{goal}_{stage}"))
            for left_name, right_name in (
                ("probe_selective", "causal"),
                ("probe_selective", "behavior"),
                ("causal", "behavior"),
            ):
                left = events[left_name]
                right = events[right_name]
                prefix = f"{goal}_{right_name}_minus_{left_name}"
                if not left.observed or not right.observed:
                    output[f"{prefix}_lag"] = None
                else:
                    assert left.event_step is not None and right.event_step is not None
                    output[f"{prefix}_lag"] = right.event_step - left.event_step
                if not left.observed or not right.observed:
                    relation = "censored"
                elif left.upper_bound is not None and left.upper_bound < right.lower_bound:
                    relation = f"{left_name}_before_{right_name}"
                elif right.upper_bound is not None and right.upper_bound < left.lower_bound:
                    relation = f"{right_name}_before_{left_name}"
                else:
                    relation = "unresolved_on_log_grid"
                output[f"{prefix}_interval_relation"] = relation

        phases = [str(row["phase"]) for row in rows]
        compressed, goal_sequence = _persistent_goal_episodes(phases)
        output["compressed_phase_sequence"] = " -> ".join(compressed)
        output["goal_sequence"] = " -> ".join(goal_sequence) if goal_sequence else "none"
        output["goal_transition_count"] = max(0, len(goal_sequence) - 1)
        output["distinct_pure_goals"] = len(set(goal_sequence))
        output["goal_cycle"] = int(
            any(goal_sequence[index] in goal_sequence[: index - 1] for index in range(2, len(goal_sequence)))
        )
        time_shares = _phase_time_shares(phases)
        for phase, share in time_shares.items():
            output[f"time_share_{phase}"] = share
        final = rows[-1]
        bridge = next(row for row in rows if int(row["step"]) == 2_048)
        output["bridge_phase"] = bridge["phase"]
        output["bridge_taxonomy"] = bridge["taxonomy"]
        output["bridge_signature"] = bridge["signature"]
        output["bridge_to_final_phase"] = f"{bridge['phase']} -> {final['phase']}"
        output["bridge_to_final_taxonomy"] = (
            f"{bridge['taxonomy']} -> {final['taxonomy']}"
        )
        for goal in GOALS:
            output[f"bridge_behavior_{goal}"] = bridge[f"behavior_{goal}"]
            output[f"bridge_causal_{goal}"] = bridge[f"causal_{goal}"]
            output[f"final_behavior_{goal}"] = final[f"behavior_{goal}"]
            output[f"final_causal_{goal}"] = final[f"causal_{goal}"]
            output[f"final_probe_{goal}"] = final[f"probe_final_hidden_{goal}"]
            output[f"final_probe_selectivity_{goal}"] = final[
                f"probe_final_hidden_{goal}_selectivity"
            ]
            output[f"final_pure_{goal}"] = int(final["phase"] == goal)
        for name in (
            "phase",
            "taxonomy",
            "signature",
            "signature_bits",
            "function_family",
            "essential_variables",
            "essential_variable_count",
            "monotone_in_candidates",
            "nearest_named_rules",
            "nearest_named_hamming",
            "tuple_consistency",
            "codeword_consistency",
            "nuisance_consistency",
        ):
            output[f"final_{name}"] = final[name]
        output["final_tuple_composite"] = int(
            str(final["taxonomy"]).startswith("tuple:")
            and str(final["function_family"])
            not in {
                "pure_P",
                "pure_Q",
                "pure_Y",
                "inverse_P",
                "inverse_Q",
                "inverse_Y",
                "constant_negative",
                "constant_positive",
            }
        )
        output["final_codeword_patch"] = int(final["taxonomy"] == "raw_codeword_patch")
        output["final_nuisance_sensitive"] = int(
            final["taxonomy"] == "nuisance_or_state_sensitive"
        )
        nested_support = ("---", "--+", "-++", "+--", "++-", "+++")
        absent_q_only = ("-+-", "+-+")

        output["final_nested_training_support_accuracy"] = float(
            np.mean([_tuple_target_accuracy(final, key) for key in nested_support])
        )
        output["final_absent_q_only_accuracy"] = float(
            np.mean([_tuple_target_accuracy(final, key) for key in absent_q_only])
        )
        result.append(output)
    return result


def bootstrap_interval(values: Sequence[float], *, label: str) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0 or not np.all(np.isfinite(array)):
        return {"estimate": None, "ci_low": None, "ci_high": None, "n_seeds": len(array)}
    estimate = float(np.mean(array))
    rng = np.random.default_rng(stable_seed(label, BOOTSTRAP_DRAWS))
    draws = np.mean(rng.choice(array, size=(BOOTSTRAP_DRAWS, len(array)), replace=True), axis=1)
    low, high = np.quantile(draws, (0.025, 0.975))
    return {
        "estimate": estimate,
        "ci_low": float(low),
        "ci_high": float(high),
        "n_seeds": len(array),
    }


def _seed_means(
    rows: Sequence[Mapping[str, Any]], outcome: str
) -> dict[int, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(outcome)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            grouped[int(row["seed"])].append(float(value))
    return {seed: float(np.mean(values)) for seed, values in grouped.items() if values}


def event_summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    cohorts = {
        "original": set(ORIGINAL_SEEDS),
        "fresh": set(FRESH_SEEDS),
        "pooled": set(ALL_SEEDS),
    }
    for cohort, seeds in cohorts.items():
        subset = [row for row in rows if int(row["seed"]) in seeds]
        for goal in GOALS:
            for stage in ("solo", "probe_generic", "probe_selective", "causal", "behavior", "pure"):
                observed_name = f"{goal}_{stage}_observed"
                restricted_name = f"{goal}_{stage}_restricted_step"
                observed_by_seed = _seed_means(subset, observed_name)
                restricted_by_seed = _seed_means(subset, restricted_name)
                observed_steps = [
                    int(row[f"{goal}_{stage}_event_step"])
                    for row in subset
                    if int(row[observed_name]) == 1
                ]
                result.append(
                    {
                        "inference_status": INFERENCE_STATUS,
                        "cohort": cohort,
                        "goal": goal,
                        "stage": stage,
                        "n_runs": len(subset),
                        "events": int(sum(int(row[observed_name]) for row in subset)),
                        "run_event_fraction": float(
                            np.mean([float(row[observed_name]) for row in subset])
                        ),
                        "median_observed_step": median(observed_steps) if observed_steps else None,
                        **{
                            f"event_fraction_{key}": value
                            for key, value in bootstrap_interval(
                                list(observed_by_seed.values()),
                                label=f"e15-event-{cohort}-{goal}-{stage}",
                            ).items()
                        },
                        **{
                            f"restricted_step_{key}": value
                            for key, value in bootstrap_interval(
                                list(restricted_by_seed.values()),
                                label=f"e15-restricted-{cohort}-{goal}-{stage}",
                            ).items()
                        },
                    }
                )
    return result


def sequence_summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    cohorts = {
        "original": set(ORIGINAL_SEEDS),
        "fresh": set(FRESH_SEEDS),
        "pooled": set(ALL_SEEDS),
    }
    pooled_sequences = Counter(str(row["goal_sequence"]) for row in rows)
    sequences = [name for name, _ in pooled_sequences.most_common()]
    for cohort, seeds in cohorts.items():
        subset = [row for row in rows if int(row["seed"]) in seeds]
        for sequence in sequences:
            per_seed: dict[int, list[float]] = defaultdict(list)
            for row in subset:
                per_seed[int(row["seed"])].append(float(row["goal_sequence"] == sequence))
            seed_values = [float(np.mean(values)) for values in per_seed.values()]
            count = sum(str(row["goal_sequence"]) == sequence for row in subset)
            result.append(
                {
                    "inference_status": INFERENCE_STATUS,
                    "cohort": cohort,
                    "sequence": sequence,
                    "count": count,
                    "n_runs": len(subset),
                    "fraction": count / len(subset),
                    **bootstrap_interval(
                        seed_values, label=f"e15-sequence-{cohort}-{sequence}"
                    ),
                }
            )
    return result


def taxonomy_summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    cohorts = {
        "original": set(ORIGINAL_SEEDS),
        "fresh": set(FRESH_SEEDS),
        "pooled": set(ALL_SEEDS),
    }
    categories = sorted({str(row["final_taxonomy"]) for row in rows})
    for cohort, seeds in cohorts.items():
        subset = [row for row in rows if int(row["seed"]) in seeds]
        for category in categories:
            per_seed: dict[int, list[float]] = defaultdict(list)
            for row in subset:
                per_seed[int(row["seed"])].append(float(row["final_taxonomy"] == category))
            count = sum(str(row["final_taxonomy"]) == category for row in subset)
            result.append(
                {
                    "inference_status": INFERENCE_STATUS,
                    "cohort": cohort,
                    "taxonomy": category,
                    "count": count,
                    "n_runs": len(subset),
                    "fraction": count / len(subset),
                    **bootstrap_interval(
                        [float(np.mean(values)) for values in per_seed.values()],
                        label=f"e15-taxonomy-{cohort}-{category}",
                    ),
                }
            )
    return result


def transition_summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Summarize the continuation from the exact E12 bridge to step 8,192."""

    result: list[dict[str, Any]] = []
    cohorts = {
        "original": set(ORIGINAL_SEEDS),
        "fresh": set(FRESH_SEEDS),
        "pooled": set(ALL_SEEDS),
    }
    transition_names = sorted({str(row["bridge_to_final_phase"]) for row in rows})
    for cohort, seeds in cohorts.items():
        subset = [row for row in rows if int(row["seed"]) in seeds]
        for transition in transition_names:
            grouped: dict[int, list[float]] = defaultdict(list)
            for row in subset:
                grouped[int(row["seed"])].append(
                    float(row["bridge_to_final_phase"] == transition)
                )
            count = sum(str(row["bridge_to_final_phase"]) == transition for row in subset)
            result.append(
                {
                    "inference_status": INFERENCE_STATUS,
                    "cohort": cohort,
                    "transition": transition,
                    "count": count,
                    "n_runs": len(subset),
                    "fraction": count / len(subset),
                    **bootstrap_interval(
                        [float(np.mean(values)) for values in grouped.values()],
                        label=f"e15-transition-{cohort}-{transition}",
                    ),
                }
            )
    return result


def special_composite_summary(
    rows: Sequence[Mapping[str, Any]], checkpoints: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Audit the recurrent support-perfect Boolean signature 113."""

    selected = [row for row in rows if int(row["final_signature"]) == 113]
    by_cohort = Counter(str(row["cohort"]) for row in selected)
    by_factor = Counter(
        (float(row["q_q"]), int(row["k_y"]), str(row["overlap"])) for row in selected
    )
    support_by_seed = _seed_means(selected, "final_nested_training_support_accuracy")
    absent_by_seed = _seed_means(selected, "final_absent_q_only_accuracy")
    onset_counts: Counter[int] = Counter()
    checkpoint_by_run: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for checkpoint in checkpoints:
        checkpoint_by_run[str(checkpoint["run_id"])].append(checkpoint)
    for row in selected:
        series = sorted(checkpoint_by_run[str(row["run_id"])], key=lambda item: int(item["step"]))
        event = persistent_event(
            COMPETITION_STEPS,
            [int(item["signature"]) == 113 for item in series],
            horizon=8_192,
        )
        if event.observed and event.event_step is not None:
            onset_counts[event.event_step] += 1
    return {
        "signature_int": 113,
        "signature_bits": "01110001",
        "tuple_order": list(TUPLE_KEYS),
        "rule": "use Y when P equals Q; otherwise use Q",
        "training_support_interpretation": (
            "For nested errors with q_Q > q_P, Q is always correct when P and Q "
            "disagree, while Y is needed when both proxies are wrong. The rule is "
            "therefore perfect on all observed candidate tuples."
        ),
        "failure_interpretation": (
            "It follows Q, and is wrong, on the two Q-only-error tuples (-+-, +-+) "
            "that nesting removes from training support."
        ),
        "final_count": len(selected),
        "by_cohort": dict(sorted(by_cohort.items())),
        "by_factor_cell": [
            {"q_q": key[0], "k_y": key[1], "overlap": key[2], "count": count}
            for key, count in sorted(by_factor.items())
        ],
        "all_nested": bool(selected) and all(str(row["overlap"]) == "nested" for row in selected),
        "all_q_q_above_q_p": bool(selected)
        and all(float(row["q_q"]) in {0.95, 0.99} for row in selected),
        "support_accuracy": bootstrap_interval(
            list(support_by_seed.values()), label="e15-signature113-support"
        ),
        "absent_q_only_accuracy": bootstrap_interval(
            list(absent_by_seed.values()), label="e15-signature113-absent"
        ),
        "persistent_onset_steps": [
            {"step": step, "count": count} for step, count in sorted(onset_counts.items())
        ],
    }


FACTOR_OUTCOMES = (
    "final_behavior_P",
    "final_behavior_Q",
    "final_behavior_Y",
    "final_causal_P",
    "final_causal_Q",
    "final_causal_Y",
    "final_probe_selectivity_Q",
    "final_probe_selectivity_Y",
    "final_pure_P",
    "final_pure_Q",
    "final_pure_Y",
    "Q_behavior_observed",
    "Y_behavior_observed",
    "Q_causal_observed",
    "Y_causal_observed",
    "Q_behavior_restricted_step",
    "Y_behavior_restricted_step",
    "Q_causal_restricted_step",
    "Y_causal_restricted_step",
    "time_share_D",
    "final_tuple_composite",
    "final_codeword_patch",
    "final_tuple_consistency",
)


def factor_contrasts(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (
            rounded(row["q_q"]),
            int(row["k_y"]),
            str(row["overlap"]),
            int(row["seed"]),
        ): row
        for row in rows
    }
    specifications: list[tuple[str, str, str]] = []
    for high, low in ((0.95, 0.90), (0.99, 0.95), (0.99, 0.90)):
        specifications.append(
            (
                "q_q",
                f"{high:g} - {low:g}",
                f"qQ_{high:g}_minus_{low:g}",
            )
        )
    specifications.extend(
        [
            (
                "k_y",
                "5 - 3",
                "kY_5_minus_3",
            ),
            (
                "overlap",
                "nested - independent",
                "nested_minus_independent",
            ),
        ]
    )

    result: list[dict[str, Any]] = []
    cohorts = {
        "original": set(ORIGINAL_SEEDS),
        "fresh": set(FRESH_SEEDS),
        "pooled": set(ALL_SEEDS),
    }
    for factor, comparison, label in specifications:
        for outcome in FACTOR_OUTCOMES:
            for cohort, seeds in cohorts.items():
                differences: dict[int, list[float]] = defaultdict(list)
                if factor == "q_q":
                    high, low = (float(value) for value in comparison.split(" - "))
                    for seed in seeds:
                        for k_y in K_Y_LEVELS:
                            for overlap in OVERLAPS:
                                upper = indexed[(rounded(high), k_y, overlap, seed)]
                                lower = indexed[(rounded(low), k_y, overlap, seed)]
                                differences[seed].append(float(upper[outcome]) - float(lower[outcome]))
                elif factor == "k_y":
                    for seed in seeds:
                        for q_q in Q_Q_LEVELS:
                            for overlap in OVERLAPS:
                                upper = indexed[(rounded(q_q), 5, overlap, seed)]
                                lower = indexed[(rounded(q_q), 3, overlap, seed)]
                                differences[seed].append(float(upper[outcome]) - float(lower[outcome]))
                else:
                    for seed in seeds:
                        for q_q in Q_Q_LEVELS:
                            for k_y in K_Y_LEVELS:
                                upper = indexed[(rounded(q_q), k_y, "nested", seed)]
                                lower = indexed[(rounded(q_q), k_y, "independent", seed)]
                                differences[seed].append(float(upper[outcome]) - float(lower[outcome]))
                seed_values = [float(np.mean(differences[seed])) for seed in sorted(differences)]
                result.append(
                    {
                        "inference_status": INFERENCE_STATUS,
                        "factor": factor,
                        "comparison": comparison,
                        "cohort": cohort,
                        "outcome": outcome,
                        "matched_pairs": sum(len(values) for values in differences.values()),
                        **bootstrap_interval(
                            seed_values,
                            label=f"e15-factor-{label}-{cohort}-{outcome}",
                        ),
                    }
                )
    return result


COHORT_OUTCOMES = (
    "final_behavior_P",
    "final_behavior_Q",
    "final_behavior_Y",
    "final_causal_P",
    "final_causal_Q",
    "final_causal_Y",
    "final_pure_P",
    "final_pure_Q",
    "final_pure_Y",
    "Q_behavior_observed",
    "Y_behavior_observed",
    "Q_causal_observed",
    "Y_causal_observed",
    "time_share_D",
    "goal_cycle",
    "final_tuple_composite",
    "final_codeword_patch",
)


def cohort_comparisons(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    original = [row for row in rows if int(row["seed"]) in ORIGINAL_SEEDS]
    fresh = [row for row in rows if int(row["seed"]) in FRESH_SEEDS]
    for outcome in COHORT_OUTCOMES:
        original_means = np.asarray(list(_seed_means(original, outcome).values()), dtype=float)
        fresh_means = np.asarray(list(_seed_means(fresh, outcome).values()), dtype=float)
        estimate = float(np.mean(fresh_means) - np.mean(original_means))
        rng = np.random.default_rng(stable_seed("e15-cohort", outcome))
        draws = np.mean(
            rng.choice(fresh_means, size=(BOOTSTRAP_DRAWS, len(fresh_means)), replace=True),
            axis=1,
        ) - np.mean(
            rng.choice(
                original_means, size=(BOOTSTRAP_DRAWS, len(original_means)), replace=True
            ),
            axis=1,
        )
        low, high = np.quantile(draws, (0.025, 0.975))
        result.append(
            {
                "inference_status": INFERENCE_STATUS,
                "comparison": "fresh - original",
                "outcome": outcome,
                "original_mean": float(np.mean(original_means)),
                "fresh_mean": float(np.mean(fresh_means)),
                "estimate": estimate,
                "ci_low": float(low),
                "ci_high": float(high),
                "original_seeds": len(original_means),
                "fresh_seeds": len(fresh_means),
            }
        )
    return result


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


def plot_font_family() -> str:
    """Register Myriad Pro from the user's font directory when available."""

    font_dirs: list[Path] = []
    configured = os.environ.get("FORKWORLD_FONT_DIR")
    if configured:
        font_dirs.append(Path(configured).expanduser())
    home_fonts = Path.home() / "Library" / "Fonts"
    if home_fonts not in font_dirs:
        font_dirs.append(home_fonts)
    filenames = (
        "MYRIADPRO-REGULAR.OTF",
        "MYRIADPRO-SEMIBOLD.OTF",
        "MYRIADPRO-BOLD.OTF",
        "MyriadPro-Light.otf",
    )
    names: list[str] = []
    for directory in font_dirs:
        for filename in filenames:
            path = directory / filename
            if path.is_file():
                fm.fontManager.addfont(str(path))
                names.append(fm.FontProperties(fname=str(path)).get_name())
    if not names:
        warnings.warn(
            "Myriad Pro was not found; falling back to DejaVu Sans.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "DejaVu Sans"
    return names[0]


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": plot_font_family(),
            "font.size": 9.2,
            "axes.titlesize": 10.2,
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
            "legend.fontsize": 8.2,
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
        xytext=(-11, 7),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=11,
        fontweight=600,
        color=INK,
        annotation_clip=False,
    )


def _event_lookup(
    event_rows: Sequence[Mapping[str, Any]], goal: str, stage: str
) -> tuple[float, float, float, float]:
    outcome = f"{goal}_{stage}_restricted_step"
    observed = f"{goal}_{stage}_observed"
    restricted = _seed_means(event_rows, outcome)
    acquired = _seed_means(event_rows, observed)
    interval = bootstrap_interval(
        list(restricted.values()), label=f"e15-figure-event-{goal}-{stage}"
    )
    return (
        float(interval["estimate"]),
        float(interval["ci_low"]),
        float(interval["ci_high"]),
        float(np.mean(list(acquired.values()))),
    )


def make_figure(
    checkpoints: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    sequences: Sequence[Mapping[str, Any]],
) -> None:
    configure_style()
    figure = plt.figure(figsize=(13.2, 5.35))
    grid = figure.add_gridspec(
        1,
        3,
        width_ratios=(1.08, 1.18, 1.06),
        left=0.075,
        right=0.985,
        bottom=0.18,
        top=0.75,
        wspace=0.34,
    )
    axis_a = figure.add_subplot(grid[0, 0])
    axis_b = figure.add_subplot(grid[0, 1])
    axis_c = figure.add_subplot(grid[0, 2])

    stage_style = {
        "probe_selective": (PURPLE, "selective hidden probe"),
        "causal": (ORANGE, "causal control"),
        "behavior": (BLUE, "behavioral agreement"),
    }
    event_positions: list[tuple[str, str]] = [
        (goal, stage) for goal in GOALS for stage in stage_style
    ]
    y_values = np.arange(len(event_positions))[::-1]
    for y, (goal, stage) in zip(y_values, event_positions, strict=True):
        estimate, low, high, incidence = _event_lookup(events, goal, stage)
        color = stage_style[stage][0]
        axis_a.errorbar(
            estimate,
            y,
            xerr=np.asarray([[estimate - low], [high - estimate]]),
            fmt="o",
            ms=4.8,
            color=color,
            ecolor=color,
            elinewidth=1.2,
            capsize=2.2,
            zorder=3,
        )
        axis_a.text(
            1.02,
            y,
            f"{incidence:.0%}",
            transform=axis_a.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=7.5,
            color=GRAY,
            clip_on=False,
        )
    axis_a.set_yticks(
        y_values,
        [f"{goal} · {stage_style[stage][1]}" for goal, stage in event_positions],
    )
    axis_a.set_xscale("symlog", linthresh=1.0)
    axis_a.set_xlim(-0.15, 10_500)
    axis_a.set_xticks((0, 10, 100, 1_000, 8_192), ("0", "10", "100", "1k", "8.2k"))
    axis_a.grid(axis="x", color=LIGHT_GRAY, lw=0.65)
    axis_a.set_xlabel("Restricted acquisition step\n(nonacquirers retained at 8,192)")
    axis_a.set_title("Availability, control, and behavior separate in time", loc="left", pad=8)
    axis_a.text(
        1.02,
        1.035,
        "acquired",
        transform=axis_a.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.3,
        color=GRAY,
    )

    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in checkpoints:
        grouped[int(row["step"])].append(row)
    phase_colors = {"P": ORANGE, "Q": GOLD, "Y": BLUE, "D": GRAY}
    phase_labels = {"P": "pure P", "Q": "pure Q", "Y": "pure Y", "D": "distributed / unresolved"}
    for phase in ("P", "Q", "Y", "D"):
        values = [
            float(np.mean([float(row["phase"] == phase) for row in grouped[step]]))
            for step in COMPETITION_STEPS
        ]
        axis_b.plot(
            COMPETITION_STEPS,
            values,
            color=phase_colors[phase],
            lw=2.0 if phase != "D" else 1.6,
            label=phase_labels[phase],
        )
    axis_b.set_xscale("symlog", linthresh=1.0)
    axis_b.set_xlim(-0.15, 8_700)
    axis_b.set_xticks((0, 10, 100, 1_000, 8_192), ("0", "10", "100", "1k", "8.2k"))
    axis_b.set_ylim(-0.025, 1.025)
    axis_b.set_yticks((0, 0.25, 0.5, 0.75, 1.0))
    axis_b.grid(axis="y", color=LIGHT_GRAY, lw=0.65)
    axis_b.set_xlabel("Competition optimizer step")
    axis_b.set_ylabel("Fraction of models")
    axis_b.set_title("Pure control moves through an ordered cascade", loc="left", pad=8)

    pooled = [row for row in sequences if row["cohort"] == "pooled"]
    top_sequences = [
        str(row["sequence"])
        for row in sorted(pooled, key=lambda row: int(row["count"]), reverse=True)[:6]
    ]
    y = np.arange(len(top_sequences))[::-1]
    width = 0.32
    cohort_style = {"original": (TEAL, -width / 2), "fresh": (PURPLE, width / 2)}
    for cohort, (color, offset) in cohort_style.items():
        lookup = {
            str(row["sequence"]): float(row["fraction"])
            for row in sequences
            if row["cohort"] == cohort
        }
        axis_c.barh(
            y + offset,
            [lookup.get(sequence, 0.0) for sequence in top_sequences],
            height=width * 0.86,
            color=color,
            alpha=0.92,
            label=f"{cohort} seeds",
        )
    axis_c.set_yticks(
        y,
        [sequence.replace(" -> ", r" $\rightarrow$ ") for sequence in top_sequences],
    )
    axis_c.set_xlim(0, max(0.15, axis_c.get_xlim()[1] * 1.08))
    axis_c.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    axis_c.grid(axis="x", color=LIGHT_GRAY, lw=0.65)
    axis_c.set_xlabel("Fraction within seed cohort")
    axis_c.set_title("The principal phase sequences replicate on fresh seeds", loc="left", pad=8)

    panel_label(axis_a, "a")
    panel_label(axis_b, "b")
    panel_label(axis_c, "c")
    legend_handles = [
        Line2D([0], [0], color=color, lw=2, label=label)
        for _, (color, label) in stage_style.items()
    ]
    legend_handles.extend(
        Line2D([0], [0], color=color, lw=2, label=phase_labels[phase])
        for phase, color in phase_colors.items()
    )
    legend_handles.extend(
        Line2D([0], [0], marker="s", ls="", color=color, label=f"{cohort} seeds")
        for cohort, (color, _) in cohort_style.items()
    )
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.885),
        ncol=5,
        columnspacing=1.25,
        handlelength=1.8,
    )
    figure.suptitle(
        "Multi-goal learning separates representation, causal control, and behavior",
        x=0.075,
        y=0.985,
        ha="left",
        fontsize=12.4,
        fontweight=600,
        color=INK,
    )
    figure.text(
        0.985,
        0.035,
        "Exploratory E15 · 12 cells by 20 seeds · intervals resample training seeds",
        ha="right",
        va="bottom",
        fontsize=7.6,
        color=GRAY,
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / "fig15_multigoal_dynamics.pdf", bbox_inches="tight")
    figure.savefig(FIGURES / "fig15_multigoal_dynamics.png", dpi=240, bbox_inches="tight")
    plt.close(figure)


def selected_findings(
    events: Sequence[Mapping[str, Any]],
    event_summary: Sequence[Mapping[str, Any]],
    sequences: Sequence[Mapping[str, Any]],
    taxonomy: Sequence[Mapping[str, Any]],
    transitions: Sequence[Mapping[str, Any]],
    contrasts: Sequence[Mapping[str, Any]],
    cohorts: Sequence[Mapping[str, Any]],
    composite: Mapping[str, Any],
) -> dict[str, Any]:
    pooled_events = {
        (str(row["goal"]), str(row["stage"])): row
        for row in event_summary
        if row["cohort"] == "pooled"
    }
    pooled_sequences = [row for row in sequences if row["cohort"] == "pooled" and int(row["count"]) > 0]
    pooled_taxonomy = [row for row in taxonomy if row["cohort"] == "pooled" and int(row["count"]) > 0]
    lag_summary: dict[str, Any] = {}
    for goal in GOALS:
        for comparison in (
            "causal_minus_probe_selective",
            "behavior_minus_probe_selective",
            "behavior_minus_causal",
        ):
            field = f"{goal}_{comparison}_lag"
            eligible = [row for row in events if isinstance(row.get(field), (int, float))]
            seed_values = _seed_means(eligible, field)
            relation_field = f"{goal}_{comparison}_interval_relation"
            lag_summary[f"{goal}_{comparison}"] = {
                "n_both_observed_runs": len(eligible),
                "interval_relations": dict(
                    Counter(str(row[relation_field]) for row in events)
                ),
                **bootstrap_interval(
                    list(seed_values.values()), label=f"e15-lag-{goal}-{comparison}"
                ),
            }
    return {
        "event_summary": {
            f"{goal}_{stage}": {
                "events": int(row["events"]),
                "n_runs": int(row["n_runs"]),
                "event_fraction": row["event_fraction_estimate"],
                "event_fraction_ci": [row["event_fraction_ci_low"], row["event_fraction_ci_high"]],
                "restricted_mean_step": row["restricted_step_estimate"],
                "restricted_mean_step_ci": [
                    row["restricted_step_ci_low"],
                    row["restricted_step_ci_high"],
                ],
            }
            for (goal, stage), row in pooled_events.items()
        },
        "logged_event_lags": lag_summary,
        "goal_sequences": [
            {
                "sequence": row["sequence"],
                "count": row["count"],
                "fraction": row["fraction"],
                "ci": [row["ci_low"], row["ci_high"]],
            }
            for row in sorted(pooled_sequences, key=lambda row: int(row["count"]), reverse=True)
        ],
        "final_taxonomy": [
            {
                "taxonomy": row["taxonomy"],
                "count": row["count"],
                "fraction": row["fraction"],
                "ci": [row["ci_low"], row["ci_high"]],
            }
            for row in sorted(pooled_taxonomy, key=lambda row: int(row["count"]), reverse=True)
        ],
        "step_2048_to_8192_transitions": [
            {
                "transition": row["transition"],
                "count": row["count"],
                "fraction": row["fraction"],
                "ci": [row["ci_low"], row["ci_high"]],
            }
            for row in sorted(
                (row for row in transitions if row["cohort"] == "pooled"),
                key=lambda row: int(row["count"]),
                reverse=True,
            )
            if int(row["count"]) > 0
        ],
        "signature_113_nested_support_gate": dict(composite),
        "factor_contrasts": list(contrasts),
        "fresh_minus_original": list(cohorts),
        "cycles": int(sum(int(row["goal_cycle"]) for row in events)),
    }


def write_outputs(
    checkpoints: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    event_summary: Sequence[Mapping[str, Any]],
    sequences: Sequence[Mapping[str, Any]],
    taxonomy: Sequence[Mapping[str, Any]],
    transitions: Sequence[Mapping[str, Any]],
    contrasts: Sequence[Mapping[str, Any]],
    cohorts: Sequence[Mapping[str, Any]],
    composite: Mapping[str, Any],
    audit: Mapping[str, Any],
    bridge: Mapping[str, Any],
    artifacts: Path,
    e12_artifacts: Path,
) -> Mapping[str, Any]:
    write_csv(DERIVED / "e15_checkpoint_dynamics.csv", checkpoints)
    write_csv(DERIVED / "e15_run_events.csv", events)
    write_csv(DERIVED / "e15_event_summary.csv", event_summary)
    write_csv(DERIVED / "e15_sequence_counts.csv", sequences)
    write_csv(DERIVED / "e15_truth_table_taxonomy.csv", taxonomy)
    write_csv(DERIVED / "e15_bridge_to_final_transitions.csv", transitions)
    write_csv(DERIVED / "e15_factor_contrasts.csv", contrasts)
    write_csv(DERIVED / "e15_cohort_comparisons.csv", cohorts)
    findings = selected_findings(
        events,
        event_summary,
        sequences,
        taxonomy,
        transitions,
        contrasts,
        cohorts,
        composite,
    )
    report = {
        "experiment": "E15_adaptive_multigoal_temporal_bridge",
        "inference_status": INFERENCE_STATUS,
        "artifacts_root": str(artifacts.resolve()),
        "e12_artifacts_root": str(e12_artifacts.resolve()),
        "thresholds": {
            "solo_accessibility": SOLO_THRESHOLD,
            "generic_probe": GENERIC_PROBE_THRESHOLD,
            "selective_probe_accuracy": SELECTIVE_PROBE_ACCURACY,
            "selective_probe_true_minus_orthogonal_control": SELECTIVE_PROBE_MARGIN,
            "behavior": BEHAVIOR_THRESHOLD,
            "causal": CAUSAL_THRESHOLD,
            "pure_margin": PURE_MARGIN,
            "persistence_checkpoints": PERSISTENCE,
            "taxonomy_consistency": TAXONOMY_CONSISTENCY,
        },
        "audit": dict(audit),
        "bridge_audit": dict(bridge),
        "findings": findings,
        "outputs": {
            "checkpoint_csv": str((DERIVED / "e15_checkpoint_dynamics.csv").resolve()),
            "run_event_csv": str((DERIVED / "e15_run_events.csv").resolve()),
            "event_summary_csv": str((DERIVED / "e15_event_summary.csv").resolve()),
            "sequence_csv": str((DERIVED / "e15_sequence_counts.csv").resolve()),
            "taxonomy_csv": str((DERIVED / "e15_truth_table_taxonomy.csv").resolve()),
            "bridge_to_final_csv": str(
                (DERIVED / "e15_bridge_to_final_transitions.csv").resolve()
            ),
            "factor_contrasts_csv": str((DERIVED / "e15_factor_contrasts.csv").resolve()),
            "cohort_comparisons_csv": str((DERIVED / "e15_cohort_comparisons.csv").resolve()),
            "figure_pdf": str((FIGURES / "fig15_multigoal_dynamics.pdf").resolve()),
            "figure_png": str((FIGURES / "fig15_multigoal_dynamics.png").resolve()),
        },
    }
    DERIVED.mkdir(parents=True, exist_ok=True)
    with (DERIVED / "e15_bridge_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(bridge), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    with (DERIVED / "e15_analysis.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(report), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=DEFAULT_ARTIFACTS,
        help=f"E15 artifact root (default: {DEFAULT_ARTIFACTS})",
    )
    parser.add_argument(
        "--e12-artifacts",
        type=Path,
        default=DEFAULT_E12_ARTIFACTS,
        help=f"archived E12 artifact root (default: {DEFAULT_E12_ARTIFACTS})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs, audit = load_runs(args.artifacts)
    bridge = bridge_audit(runs, args.e12_artifacts)
    checkpoints = checkpoint_rows(runs)
    events = run_event_rows(runs, checkpoints)
    event_summary = event_summaries(events)
    sequences = sequence_summaries(events)
    taxonomy = taxonomy_summaries(events)
    transitions = transition_summaries(events)
    composite = special_composite_summary(events, checkpoints)
    contrasts = factor_contrasts(events)
    cohorts = cohort_comparisons(events)
    make_figure(checkpoints, events, sequences)
    report = write_outputs(
        checkpoints,
        events,
        event_summary,
        sequences,
        taxonomy,
        transitions,
        contrasts,
        cohorts,
        composite,
        audit,
        bridge,
        args.artifacts,
        args.e12_artifacts,
    )
    findings = get(report, "findings", {})
    print(
        f"E15 strict analysis passed: {audit['complete_runs']}/{audit['expected_runs']} runs, "
        f"{bridge['original_seed_bridge_checks']} exact E12 bridges"
    )
    print(f"  metric records audited: {audit['metric_records']:,}")
    print(f"  cycles between distinct pure goals: {findings.get('cycles')}")
    for row in findings.get("goal_sequences", [])[:8]:
        print(f"  sequence {row['sequence']}: {row['count']}/240 ({row['fraction']:.3f})")
    print(f"  report: {DERIVED / 'e15_analysis.json'}")
    print(f"  figure: {FIGURES / 'fig15_multigoal_dynamics.pdf'}")


if __name__ == "__main__":
    main()
