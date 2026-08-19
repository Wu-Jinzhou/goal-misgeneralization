#!/usr/bin/env python3
"""Strict audit, seed-level analysis, and plotting for exploratory E14.

E14 was designed after the completed E10 grid was inspected.  Every output is
therefore labelled exploratory/post-hoc and boundary-specific.  The default
invocation is intentionally all-or-nothing: it refuses to analyse anything
other than the complete registered 2 x 6 x 10 grid and verifies the phase and
reset invariants before computing a statistic.

The training seed is the independent unit.  Intervals are deterministic
4,000-draw percentile bootstraps over the ten seeds, and all schedule contrasts
are paired by seed within cell.
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
from itertools import product
from pathlib import Path
from statistics import mean
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/forkworld-e14-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/forkworld-e14-xdg")
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

from forkworld.metrics import binary_predictions
from forkworld.protocols import predict_logits, run_protocol

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_ARTIFACTS = REPO / "artifacts-e14"
DEFAULT_E10_ARTIFACTS = REPO / "artifacts-followups"
DERIVED = HERE / "derived"
FIGURES = HERE / "figures"

SEEDS = (11, 23, 37, 41, 53, 67, 71, 83, 97, 101)
CELLS = ((0.75, 4, "responsive boundary"), (0.90, 3, "preselected comparison"))
DISPLAY_CELL_LABELS = {
    "responsive boundary": "responsive boundary",
    "preselected comparison": "canonical comparison",
}
SCHEDULES = (
    "zero_zero",
    "high_high",
    "early_only",
    "delayed_carry",
    "delayed_actor_reset",
    "delayed_critic_reset",
)
SCHEDULE_SETTINGS: dict[str, tuple[float, float, bool, bool, bool]] = {
    # phase-A beta, phase-B beta, actor-Adam reset, critic-weight reset,
    # critic-Adam reset
    "zero_zero": (0.0, 0.0, False, False, False),
    "high_high": (0.3, 0.3, False, False, False),
    "early_only": (0.3, 0.0, False, False, False),
    "delayed_carry": (0.0, 0.3, False, False, False),
    "delayed_actor_reset": (0.0, 0.3, True, False, False),
    "delayed_critic_reset": (0.0, 0.3, False, True, True),
}
CONTRASTS = (
    ("early_only", "zero_zero", "early-only - never"),
    ("delayed_carry", "zero_zero", "delayed - never"),
    ("early_only", "delayed_carry", "early-only - delayed"),
    ("high_high", "early_only", "continuous - early-only"),
    ("delayed_actor_reset", "delayed_carry", "actor-Adam reset - delayed"),
    ("delayed_critic_reset", "delayed_carry", "critic reset - delayed"),
)
PHASE_A_STEPS = (1, 2, 4, 8, 16, 32, 64)
PHASE_B_LOCAL_STEPS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1984)
PHASE_B_GLOBAL_STEPS = tuple(64 + step for step in PHASE_B_LOCAL_STEPS)
EXPECTED_RUNS = 120
BOOTSTRAP_DRAWS = 4_000
INFERENCE_STATUS = "exploratory_post_hoc_boundary_specific"
ARTIFACT_SCHEMA_VERSION = 1
SOURCE_FINGERPRINT_SCHEMA_VERSION = 1
NON_SCIENTIFIC_RUN_FIELDS = frozenset({"output_root", "resume", "seeds"})
REPLAY_CASES = (
    (0.75, 4, "delayed_carry", 11),
    (0.75, 4, "delayed_carry", 37),
    (0.75, 4, "high_high", 71),
    (0.90, 3, "delayed_carry", 23),
)

INK = "#17212B"
GRAY = "#667085"
LIGHT_GRAY = "#E6E9EE"
BLUE = "#2673B8"
ORANGE = "#D55E00"
TEAL = "#009E73"
PURPLE = "#7A5195"
GOLD = "#C99700"
SKY = "#56B4E9"
ROSE = "#CC79A7"
SCHEDULE_COLORS = {
    "zero_zero": GRAY,
    "high_high": PURPLE,
    "early_only": BLUE,
    "delayed_carry": ORANGE,
    "delayed_actor_reset": TEAL,
    "delayed_critic_reset": GOLD,
}
SCHEDULE_LABELS = {
    "zero_zero": "never",
    "high_high": "continuous",
    "early_only": "early only",
    "delayed_carry": "delayed",
    "delayed_actor_reset": "delayed + actor-Adam reset",
    "delayed_critic_reset": "delayed + critic reset",
}


def get(mapping: Mapping[str, Any], path: str, default: Any = None) -> Any:
    """Read a dot-delimited path from a nested mapping."""
    value: Any = mapping
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value


def f(value: Any) -> float:
    return round(float(value), 8)


def trapezoidal_integral(y: np.ndarray[Any, Any], x: np.ndarray[Any, Any]) -> float:
    """Integrate with the NumPy 2 API and its numerically identical 1.26 alias."""

    try:
        return float(np.trapezoid(y, x))
    except AttributeError:  # NumPy 1.26
        return float(np.trapz(y, x))  # type: ignore[attr-defined]


def expected_artifact_run_id(
    config: Mapping[str, Any], metadata: Mapping[str, Any]
) -> str:
    """Reconstruct a RunStore identity from an immutable resolved artifact."""

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
        "implementation_fingerprint": get(implementation, "implementation_fingerprint"),
    }
    raw = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def stable_seed(*parts: Any) -> int:
    digest = hashlib.blake2b(digest_size=8, person=b"forke14")
    for part in parts:
        encoded = str(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
    return int.from_bytes(digest.digest(), "little")


@dataclass(frozen=True)
class Run:
    path: Path
    config: Mapping[str, Any]
    summary: Mapping[str, Any]
    metadata: Mapping[str, Any]
    metrics: tuple[Mapping[str, Any], ...]

    @property
    def seed(self) -> int:
        return int(self.summary["seed"])

    @property
    def q(self) -> float:
        return f(get(self.config, "data.q"))

    @property
    def k(self) -> int:
        return int(get(self.config, "data.k"))

    @property
    def schedule(self) -> str:
        return str(get(self.config, "h5.timing_schedule"))


def expected_keys() -> set[tuple[float, int, str, int]]:
    return {(f(q), k, schedule, seed) for (q, k, _), schedule, seed in product(CELLS, SCHEDULES, SEEDS)}


def load_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def load_metrics(path: Path) -> tuple[Mapping[str, Any], ...]:
    records: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise RuntimeError(f"non-object metric at {path}:{line_number}")
            records.append(value)
    return tuple(records)


def expect_equal(errors: list[str], run_id: str, actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        errors.append(f"{run_id}: {label}={actual!r}, expected {expected!r}")


def validate_run(run: Run, errors: list[str]) -> None:
    """Validate fixed settings, exploratory metadata, and phase/reset audit."""
    run_id = run.path.name
    config, summary = run.config, run.summary
    fixed = {
        "schema_version": 1,
        "experiment.hypothesis": "h5",
        "experiment.name": "entropy_timing",
        "experiment.mode": "entropy_timing",
        "run.task_levels": ["choice"],
        "run.save_checkpoints": False,
        "evaluation.save_predictions": False,
        "evaluation.bootstrap_samples": BOOTSTRAP_DRAWS,
        "data.n_train": 10_000,
        "data.n_validation": 4_000,
        "data.n_eval": 10_000,
        "data.max_k": 5,
        "data.state_dim": 8,
        "data.target_rule": "parity",
        "model.width": 64,
        "model.depth": 2,
        "model.activation": "relu",
        "model.residual": False,
        "update.mode": "full",
        "update.budget": "full",
        "train.steps": 2_048,
        "train.batch_size": 250,
        "train.learning_rate": 0.003,
        "train.weight_decay": 0.0,
        "h5.algorithm": "rl",
        "h5.rl_estimator": "actor_critic",
        "h5.critic_width": 128,
        "h5.critic_depth": 3,
        "h5.nuisance_bits": 8,
        "h5.nuisance_entropy": 0,
        "h5.nuisance_weight": 1.0,
        "h5.phase_a_steps": 64,
        "h5.phase_b_steps": 1_984,
    }
    for path, expected in fixed.items():
        expect_equal(errors, run_id, get(config, path), expected, path)
    expect_equal(errors, run_id, int(get(config, "seed")), run.seed, "config seed")
    expect_equal(errors, run_id, summary.get("hypothesis"), "h5", "summary hypothesis")
    expect_equal(errors, run_id, summary.get("condition"), run.schedule, "condition")
    expect_equal(errors, run_id, f(get(summary, "cell.q")), run.q, "cell.q")
    expect_equal(errors, run_id, int(get(summary, "cell.k")), run.k, "cell.k")

    design_expected = {
        "experiment": "E14_entropy_timing",
        "registration_status": "exploratory_post_hoc_after_complete_E10",
        "confirmatory": False,
        "fixed_design_document": "followups.md#e14-exploratory-timing-of-entropy-exposure",
        "independent_replication_unit": "training_seed",
        "reason_for_cell_selection": "largest_E10_beta_0_30_response",
    }
    for name, expected in design_expected.items():
        expect_equal(errors, run_id, get(summary, f"design.{name}"), expected, f"design.{name}")
    expect_equal(
        errors,
        run_id,
        get(summary, "design.responsive_cell"),
        {"q": 0.75, "k": 4},
        "responsive cell metadata",
    )
    expect_equal(
        errors,
        run_id,
        get(summary, "design.negative_control_cell"),
        {"q": 0.9, "k": 3},
        "negative-control metadata",
    )

    phase_a, phase_b, actor_reset, critic_weights_reset, critic_reset = SCHEDULE_SETTINGS[run.schedule]
    schedule_expected = {
        "name": run.schedule,
        "phase_a_entropy_coefficient": phase_a,
        "phase_b_entropy_coefficient": phase_b,
        "phase_a_steps": 64,
        "phase_b_steps": 1_984,
        "phase_boundary_global_step": 64,
    }
    for name, expected in schedule_expected.items():
        expect_equal(errors, run_id, get(summary, f"schedule.{name}"), expected, f"schedule.{name}")
    reset_expected = {
        "actor_weights_reset": False,
        "actor_optimizer_reset": actor_reset,
        "critic_weights_reset": critic_weights_reset,
        "critic_optimizer_reset": critic_reset,
        "actor_optimizer_carried_observed": not actor_reset,
        "critic_optimizer_carried_observed": not critic_reset,
        "critic_object_reused_observed": True,
        "action_generator_state_continued": True,
        "data_generator_state_continued": True,
        "semantic_sampler_reused": True,
    }
    for name, expected in reset_expected.items():
        expect_equal(
            errors,
            run_id,
            get(summary, f"schedule.reset_audit.{name}"),
            expected,
            f"reset_audit.{name}",
        )

    for path, expected in (
        ("boundary.global_step", 64),
        ("boundary.examples_seen", 16_000),
        ("boundary.final.n", 10_000),
        ("boundary.iid.n", 4_000),
        ("final.n", 10_000),
        ("iid.n", 4_000),
        ("costs.optimizer_steps", 2_048),
        ("costs.episode_or_example_presentations", 512_000),
        ("trajectory.persistence_evaluations", 2),
        ("trajectory.persistence_threshold", 0.5),
    ):
        expect_equal(errors, run_id, get(summary, path), expected, path)

    if not isinstance(get(summary, "trajectory.phase_b_rho_y_area"), (int, float)):
        errors.append(f"{run_id}: missing phase-B intended-reliance area")
    for stage_name in ("boundary", "final"):
        intervention_root = "boundary.interventions" if stage_name == "boundary" else "interventions"
        for intervention in ("flip_P", "flip_R_mean"):
            if get(summary, f"{intervention_root}.{intervention}.hard_flip_rate") is None:
                errors.append(f"{run_id}: missing {stage_name} {intervention} hard flip")
            if get(summary, f"{intervention_root}.{intervention}.probability_ate") is None:
                errors.append(f"{run_id}: missing {stage_name} {intervention} probability effect")

    expected_metric_keys = (
        {("phase_a", "conflict_eval", "rho_y", step, step) for step in PHASE_A_STEPS}
        | {
            ("phase_b", "conflict_eval", "rho_y", local, global_step)
            for local, global_step in zip(PHASE_B_LOCAL_STEPS, PHASE_B_GLOBAL_STEPS, strict=True)
        }
        | {
            ("phase_b", "train", "value_loss", local, global_step)
            for local, global_step in zip(PHASE_B_LOCAL_STEPS, PHASE_B_GLOBAL_STEPS, strict=True)
        }
    )
    observed_metric_keys = {
        (
            str(record.get("stage")),
            str(record.get("split")),
            str(record.get("metric")),
            int(record.get("stage_step", -1)),
            int(record.get("global_step", -1)),
        )
        for record in run.metrics
        if (
            record.get("metric") == "rho_y"
            and record.get("split") == "conflict_eval"
            and record.get("stage") in {"phase_a", "phase_b"}
        )
        or (
            record.get("metric") == "value_loss"
            and record.get("split") == "train"
            and record.get("stage") == "phase_b"
        )
    }
    missing_metrics = expected_metric_keys - observed_metric_keys
    extra_metrics = observed_metric_keys - expected_metric_keys
    if missing_metrics or extra_metrics:
        errors.append(
            f"{run_id}: trajectory metric grid has {len(missing_metrics)} missing and "
            f"{len(extra_metrics)} unexpected records"
        )


def load_runs(root: Path) -> tuple[list[Run], dict[str, Any]]:
    directory = root / "h5" / "entropy_timing"
    if not directory.is_dir():
        raise RuntimeError(f"missing E14 artifact directory: {directory}")
    config_paths = sorted(directory.glob("*/resolved_config.yaml"))
    materialized = {path.parent for path in config_paths}
    if len(materialized) != EXPECTED_RUNS:
        raise RuntimeError(f"E14 hard audit failed: {len(materialized)}/{EXPECTED_RUNS} materialized runs")
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
        status = load_json(run_dir / "status.json")
        state = str(status.get("state"))
        states[state] += 1
        if state != "complete":
            errors.append(f"{run_dir.name}: status={state!r}")
        if status.get("run_id") != run_dir.name:
            errors.append(f"{run_dir.name}: status run_id mismatch")
        marker = (run_dir / "COMPLETE").read_text(encoding="utf-8")
        if marker != "complete\n":
            errors.append(f"{run_dir.name}: invalid COMPLETE marker contents {marker!r}")
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        if not isinstance(config, Mapping):
            errors.append(f"{run_dir.name}: resolved configuration is not a mapping")
            continue
        run = Run(
            path=run_dir,
            config=config,
            summary=load_json(run_dir / "summary.json"),
            metadata=load_json(run_dir / "metadata.json"),
            metrics=load_metrics(run_dir / "metrics.jsonl"),
        )
        if int(get(run.metadata, "seed", -1)) != run.seed:
            errors.append(f"{run_dir.name}: metadata seed mismatch")
        if run.metadata.get("run_id") != run_dir.name:
            errors.append(f"{run_dir.name}: metadata run_id mismatch")
        if expected_artifact_run_id(run.config, run.metadata) != run_dir.name:
            errors.append(f"{run_dir.name}: run identity does not reconstruct")
        validate_run(run, errors)
        runs.append(run)

    orphan_artifacts = [
        path.parent.name
        for pattern in ("*/COMPLETE", "*/summary.json", "*/status.json")
        for path in directory.glob(pattern)
        if path.parent not in materialized
    ]
    if orphan_artifacts:
        errors.append(f"{len(orphan_artifacts)} artifacts lack a resolved configuration")
    keys = [(run.q, run.k, run.schedule, run.seed) for run in runs]
    duplicates = [key for key, count in Counter(keys).items() if count > 1]
    unexpected = set(keys) - expected_keys()
    missing_keys = expected_keys() - set(keys)
    if duplicates or unexpected or missing_keys:
        errors.append(
            f"grid: {len(duplicates)} duplicates, {len(unexpected)} unexpected, {len(missing_keys)} missing"
        )
    counts = Counter((run.q, run.k, run.schedule) for run in runs)
    bad_counts = {key: count for key, count in counts.items() if count != len(SEEDS)}
    if len(counts) != 12 or bad_counts:
        errors.append(f"cell/schedule replication counts invalid: {bad_counts}")
    raw_fingerprints = [get(run.metadata, "implementation.implementation_fingerprint") for run in runs]
    fingerprints = {str(value) for value in raw_fingerprints if isinstance(value, str) and value}
    if len(fingerprints) != 1 or any(not isinstance(value, str) or not value for value in raw_fingerprints):
        errors.append(f"expected one nonempty implementation fingerprint, observed {fingerprints}")
    malformed_fingerprints = [
        value
        for value in raw_fingerprints
        if not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ]
    if malformed_fingerprints:
        errors.append(f"malformed implementation fingerprints={malformed_fingerprints!r}")
    schema_versions = {get(run.metadata, "implementation.artifact_schema_version") for run in runs}
    if schema_versions != {ARTIFACT_SCHEMA_VERSION}:
        errors.append(
            f"artifact schema versions={schema_versions!r}, expected "
            f"{{{ARTIFACT_SCHEMA_VERSION}}}"
        )
    source_schema_versions = {
        get(run.metadata, "implementation.source_fingerprint_schema_version") for run in runs
    }
    if source_schema_versions != {SOURCE_FINGERPRINT_SCHEMA_VERSION}:
        errors.append(
            f"source-fingerprint schema versions={source_schema_versions!r}, expected "
            f"{{{SOURCE_FINGERPRINT_SCHEMA_VERSION}}}"
        )
    caution_present = all(
        any("exploratory/post-hoc" in str(item) for item in run.metadata.get("cautions", [])) for run in runs
    )
    if not caution_present:
        errors.append("one or more artifacts lack the exploratory/post-hoc metadata caution")
    if errors:
        preview = "\n  - ".join(errors[:20])
        raise RuntimeError(f"E14 hard audit failed ({len(errors)} errors):\n  - {preview}")
    audit = {
        "expected_runs": EXPECTED_RUNS,
        "completed_runs": len(runs),
        "status_counts": dict(states),
        "unique_grid_keys": len(set(keys)),
        "seeds": list(SEEDS),
        "runs_per_cell_schedule": 10,
        "implementation_fingerprint": next(iter(fingerprints)),
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "source_fingerprint_schema_version": SOURCE_FINGERPRINT_SCHEMA_VERSION,
        "run_identities_validated": len(runs),
        "artifacts_root": str(root.resolve()),
        "exploratory_metadata_caution_present_in_all_runs": caution_present,
        "complete": True,
    }
    return runs, audit


def canonical_subset(value: Any, *, omit: frozenset[str] = frozenset()) -> Any:
    """Recursively remove explicitly new E14-only fields before exact comparison."""
    if isinstance(value, Mapping):
        return {
            str(key): canonical_subset(item, omit=omit) for key, item in value.items() if str(key) not in omit
        }
    if isinstance(value, list):
        return [canonical_subset(item, omit=omit) for item in value]
    return value


def audit_phase_invariance(runs: Sequence[Run], e10_root: Path) -> dict[str, Any]:
    """Check common-prefix determinism and E10 equivalence of continuous schedules."""
    by_key = {(run.q, run.k, run.schedule, run.seed): run for run in runs}
    phase_groups = {
        "beta_zero_prefix": (
            "zero_zero",
            "delayed_carry",
            "delayed_actor_reset",
            "delayed_critic_reset",
        ),
        "beta_point_three_prefix": ("high_high", "early_only"),
    }
    boundary_fields = ("train", "iid", "final", "interventions")
    prefix_checks = 0
    for q, k, _ in CELLS:
        for seed in SEEDS:
            for names in phase_groups.values():
                reference = by_key[(f(q), k, names[0], seed)]
                for name in names[1:]:
                    candidate = by_key[(f(q), k, name, seed)]
                    for field in boundary_fields:
                        if get(reference.summary, f"boundary.{field}") != get(
                            candidate.summary, f"boundary.{field}"
                        ):
                            raise RuntimeError(
                                "phase-A common-prefix invariance failed for "
                                f"q={q}, k={k}, seed={seed}, {names[0]} vs {name}, {field}"
                            )
                    prefix_checks += 1

    e10_directory = e10_root / "h5" / "rl_entropy_shortcut_boundary"
    if not e10_directory.is_dir():
        raise RuntimeError(f"missing E10 artifacts for invariance audit: {e10_directory}")
    e10: dict[tuple[float, int, float, int], tuple[Mapping[str, Any], str]] = {}
    for config_path in sorted(e10_directory.glob("*/resolved_config.yaml")):
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        if not isinstance(config, Mapping):
            continue
        if get(config, "update.mode") != "full":
            continue
        if get(config, "h5.rl_estimator") != "actor_critic":
            continue
        key = (
            f(get(config, "data.q")),
            int(get(config, "data.k")),
            f(get(config, "h5.entropy_coefficient")),
            int(get(config, "seed")),
        )
        if key in e10:
            raise RuntimeError(f"duplicate E10 continuous-reference key: {key}")
        run_dir = config_path.parent
        summary_path = config_path.parent / "summary.json"
        marker_path = run_dir / "COMPLETE"
        if not summary_path.is_file() or not marker_path.is_file():
            continue
        metadata_path = run_dir / "metadata.json"
        status_path = run_dir / "status.json"
        if not metadata_path.is_file() or not status_path.is_file():
            raise RuntimeError(f"E10 reference {run_dir.name} lacks metadata or status")
        if marker_path.read_text(encoding="utf-8") != "complete\n":
            raise RuntimeError(f"E10 reference {run_dir.name} has an invalid COMPLETE marker")
        metadata = load_json(metadata_path)
        status = load_json(status_path)
        seed = int(get(config, "seed"))
        if status.get("state") != "complete" or status.get("run_id") != run_dir.name:
            raise RuntimeError(f"E10 reference {run_dir.name} has invalid completion status")
        if metadata.get("run_id") != run_dir.name or metadata.get("seed") != seed:
            raise RuntimeError(f"E10 reference {run_dir.name} has inconsistent metadata identity")
        if expected_artifact_run_id(config, metadata) != run_dir.name:
            raise RuntimeError(f"E10 reference {run_dir.name} run identity does not reconstruct")
        summary = load_json(summary_path)
        if summary.get("seed") != seed or summary.get("hypothesis") != "h5":
            raise RuntimeError(f"E10 reference {run_dir.name} has inconsistent summary identity")
        artifact_schema = get(metadata, "implementation.artifact_schema_version")
        source_schema = get(metadata, "implementation.source_fingerprint_schema_version")
        fingerprint = get(metadata, "implementation.implementation_fingerprint")
        if artifact_schema != ARTIFACT_SCHEMA_VERSION:
            raise RuntimeError(f"E10 reference {run_dir.name} has artifact schema {artifact_schema!r}")
        if source_schema != SOURCE_FINGERPRINT_SCHEMA_VERSION:
            raise RuntimeError(f"E10 reference {run_dir.name} has source schema {source_schema!r}")
        if (
            not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in fingerprint)
        ):
            raise RuntimeError(f"E10 reference {run_dir.name} has an invalid source fingerprint")
        e10[key] = (summary, fingerprint)

    exact_sections = ("train", "iid", "interventions", "navigation")
    continuous_checks = 0
    selected_reference_fingerprints: set[str] = set()
    for run in runs:
        if run.schedule not in {"zero_zero", "high_high"}:
            continue
        beta = 0.0 if run.schedule == "zero_zero" else 0.3
        reference_key = (run.q, run.k, f(beta), run.seed)
        if reference_key not in e10:
            raise RuntimeError(f"missing E10 continuous reference: {reference_key}")
        e10_summary, e10_fingerprint = e10[reference_key]
        selected_reference_fingerprints.add(e10_fingerprint)
        for section in exact_sections:
            if run.summary.get(section) != e10_summary.get(section):
                raise RuntimeError(f"continuous schedule differs from E10 for {reference_key}, {section}")
        if canonical_subset(run.summary.get("final"), omit=frozenset({"policy_entropy"})) != (
            canonical_subset(e10_summary.get("final"), omit=frozenset({"policy_entropy"}))
        ):
            raise RuntimeError(f"continuous schedule differs from E10 for {reference_key}, final")
        continuous_checks += 1
    if continuous_checks != 40:
        raise RuntimeError(f"expected 40 continuous-reference checks, got {continuous_checks}")
    if len(selected_reference_fingerprints) != 1:
        raise RuntimeError(
            "continuous E10 references mix implementation fingerprints: "
            f"{sorted(selected_reference_fingerprints)}"
        )
    return {
        "e10_reference_artifacts_root": str(e10_root.resolve()),
        "e10_reference_implementation_fingerprints": sorted(selected_reference_fingerprints),
        "e10_primary_artifacts_identity_audited": len(e10),
        "phase_a_common_prefix_exact": True,
        "phase_a_prefix_pair_checks": prefix_checks,
        "continuous_schedules_match_uninterrupted_E10_exactly": True,
        "continuous_schedule_reference_checks": continuous_checks,
        "continuous_sections_compared": [*exact_sections, "final_except_policy_entropy"],
        "generator_and_sampler_continuation_audited_in_all_runs": True,
    }


def metric_series(run: Run, *, stage: str, split: str, metric: str) -> list[tuple[int, int, float]]:
    return sorted(
        (
            int(record["stage_step"]),
            int(record["global_step"]),
            float(record["value"]),
        )
        for record in run.metrics
        if record.get("stage") == stage and record.get("split") == split and record.get("metric") == metric
    )


def log_grid_mean(series: Sequence[tuple[int, int, float]]) -> float:
    """Trapezoidal mean over the logged phase-B update grid."""
    x = np.asarray([item[0] for item in series], dtype=float)
    y = np.asarray([item[2] for item in series], dtype=float)
    if len(x) < 2 or x[-1] == x[0]:
        return float("nan")
    return trapezoidal_integral(y, x) / float(x[-1] - x[0])


def run_row(run: Run) -> dict[str, Any]:
    summary = run.summary
    crossing = get(summary, "trajectory.phase_b_updates_to_persistent_rho_y_0_5")
    observed = crossing is not None
    critic_series = metric_series(run, stage="phase_b", split="train", metric="value_loss")
    row: dict[str, Any] = {
        "inference_status": INFERENCE_STATUS,
        "run_id": run.path.name,
        "q": run.q,
        "k": run.k,
        "cell": "responsive" if (run.q, run.k) == (0.75, 4) else "negative_control",
        "cell_label": next(label for q, k, label in CELLS if (f(q), k) == (run.q, run.k)),
        "schedule": run.schedule,
        "schedule_label": SCHEDULE_LABELS[run.schedule],
        "seed": run.seed,
        "boundary_rho_y": float(get(summary, "boundary.final.rho_y")),
        "boundary_intended_probability": float(get(summary, "boundary.final.intended_probability")),
        "boundary_policy_entropy": float(get(summary, "boundary.final.policy_entropy")),
        "boundary_iid_accuracy": float(get(summary, "boundary.iid.target_accuracy")),
        "final_rho_y": float(get(summary, "final.rho_y")),
        "final_intended_probability": float(get(summary, "final.intended_probability")),
        "final_policy_entropy": float(get(summary, "final.policy_entropy")),
        "final_iid_accuracy": float(get(summary, "iid.target_accuracy")),
        "final_iid_reward": float(get(summary, "iid.target_accuracy")),
        "phase_b_rho_y_area": float(get(summary, "trajectory.phase_b_rho_y_area")),
        "phase_b_mean_rho_y": float(get(summary, "trajectory.phase_b_mean_rho_y")),
        "crossing_observed": int(observed),
        "crossing_global_step": get(summary, "trajectory.persistent_rho_y_0_5_global_step"),
        "crossing_phase_b_updates": crossing,
        "restricted_crossing_phase_b_updates": float(crossing) if observed else 1_984.0,
        "crossing_censor_updates": 1_984,
        "boundary_proxy_hard_flip": float(get(summary, "boundary.interventions.flip_P.hard_flip_rate")),
        "boundary_proxy_probability_ate": float(
            get(summary, "boundary.interventions.flip_P.probability_ate")
        ),
        "boundary_exact_hard_flip": float(get(summary, "boundary.interventions.flip_R_mean.hard_flip_rate")),
        "boundary_exact_probability_ate": float(
            get(summary, "boundary.interventions.flip_R_mean.probability_ate")
        ),
        "final_proxy_hard_flip": float(get(summary, "interventions.flip_P.hard_flip_rate")),
        "final_proxy_probability_ate": float(get(summary, "interventions.flip_P.probability_ate")),
        "final_exact_hard_flip": float(get(summary, "interventions.flip_R_mean.hard_flip_rate")),
        "final_exact_probability_ate": float(get(summary, "interventions.flip_R_mean.probability_ate")),
        "actor_l2_from_boundary": float(get(summary, "parameter_dynamics.actor_l2_from_boundary")),
        "actor_relative_l2_from_boundary": float(
            get(summary, "parameter_dynamics.actor_relative_l2_from_boundary")
        ),
        "critic_l2_from_boundary": float(get(summary, "parameter_dynamics.critic_l2_from_boundary")),
        "critic_relative_l2_from_boundary": float(
            get(summary, "parameter_dynamics.critic_relative_l2_from_boundary")
        ),
        "phase_b_critic_loss_log_grid_mean": log_grid_mean(critic_series),
        "phase_b_critic_loss_final_logged": critic_series[-1][2],
    }
    row["exact_hard_flip_change"] = row["final_exact_hard_flip"] - row["boundary_exact_hard_flip"]
    row["proxy_hard_flip_change"] = row["final_proxy_hard_flip"] - row["boundary_proxy_hard_flip"]
    row["exact_probability_ate_change"] = (
        row["final_exact_probability_ate"] - row["boundary_exact_probability_ate"]
    )
    row["proxy_probability_ate_change"] = (
        row["final_proxy_probability_ate"] - row["boundary_proxy_probability_ate"]
    )
    interaction_cells = 2**run.k
    plateau_index = round(float(row["final_rho_y"]) * interaction_cells)
    quantized_rho_y = plateau_index / interaction_cells
    row["final_quantized_rho_y"] = quantized_rho_y
    row["final_quantization_plateau_index"] = plateau_index
    row["final_quantization_abs_residual"] = abs(float(row["final_rho_y"]) - quantized_rho_y)
    row["final_within_0_01_of_interaction_plateau"] = int(
        float(row["final_quantization_abs_residual"]) <= 0.01
    )
    return row


def bootstrap_interval(values: Sequence[float], *, label: str) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {"estimate": None, "ci_low": None, "ci_high": None, "n_seeds": 0}
    rng = np.random.default_rng(stable_seed("bootstrap", label))
    sampled = rng.choice(array, size=(BOOTSTRAP_DRAWS, len(array)), replace=True).mean(axis=1)
    low, high = np.quantile(sampled, (0.025, 0.975))
    return {
        "estimate": float(array.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "n_seeds": len(array),
    }


def outcome_estimate(rows: Sequence[Mapping[str, Any]], outcome: str, *, label: str) -> dict[str, Any]:
    by_seed: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        raw = row.get(outcome)
        if raw is not None and math.isfinite(float(raw)):
            by_seed[int(row["seed"])].append(float(raw))
    collapsed = [mean(by_seed[seed]) for seed in sorted(by_seed)]
    return {**bootstrap_interval(collapsed, label=label), "n_rows": len(rows)}


OUTCOMES = (
    "final_rho_y",
    "final_intended_probability",
    "phase_b_rho_y_area",
    "phase_b_mean_rho_y",
    "crossing_observed",
    "restricted_crossing_phase_b_updates",
    "final_iid_accuracy",
    "final_iid_reward",
    "final_policy_entropy",
    "boundary_rho_y",
    "boundary_policy_entropy",
    "boundary_proxy_hard_flip",
    "boundary_proxy_probability_ate",
    "boundary_exact_hard_flip",
    "boundary_exact_probability_ate",
    "final_proxy_hard_flip",
    "final_proxy_probability_ate",
    "final_exact_hard_flip",
    "final_exact_probability_ate",
    "proxy_hard_flip_change",
    "exact_hard_flip_change",
    "proxy_probability_ate_change",
    "exact_probability_ate_change",
    "actor_l2_from_boundary",
    "actor_relative_l2_from_boundary",
    "critic_l2_from_boundary",
    "critic_relative_l2_from_boundary",
    "phase_b_critic_loss_log_grid_mean",
    "phase_b_critic_loss_final_logged",
    "final_quantization_abs_residual",
    "final_within_0_01_of_interaction_plateau",
)


def analyse_rows(
    rows: Sequence[Mapping[str, Any]], runs: Sequence[Run]
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    cell_schedule: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    crossing: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = []
    for q, k, cell_label in CELLS:
        for schedule in SCHEDULES:
            part = [
                row
                for row in rows
                if f(row["q"]) == f(q) and int(row["k"]) == k and row["schedule"] == schedule
            ]
            if len(part) != 10:
                raise RuntimeError(f"analysis cell q={q}, k={k}, {schedule} has {len(part)} rows")
            entry: dict[str, Any] = {
                "inference_status": INFERENCE_STATUS,
                "q": q,
                "k": k,
                "cell_label": cell_label,
                "schedule": schedule,
                "schedule_label": SCHEDULE_LABELS[schedule],
            }
            for outcome in OUTCOMES:
                estimate = outcome_estimate(part, outcome, label=f"e14-cell-{q}-{k}-{schedule}-{outcome}")
                entry.update({f"{outcome}_{name}": value for name, value in estimate.items()})
            cell_schedule.append(entry)
            crossing.append(
                {
                    "inference_status": INFERENCE_STATUS,
                    "q": q,
                    "k": k,
                    "cell_label": cell_label,
                    "schedule": schedule,
                    "schedule_label": SCHEDULE_LABELS[schedule],
                    "events": sum(int(row["crossing_observed"]) for row in part),
                    "censored": sum(1 - int(row["crossing_observed"]) for row in part),
                    **outcome_estimate(
                        part,
                        "crossing_observed",
                        label=f"e14-crossing-rate-{q}-{k}-{schedule}",
                    ),
                    **{
                        f"restricted_time_{name}": value
                        for name, value in outcome_estimate(
                            part,
                            "restricted_crossing_phase_b_updates",
                            label=f"e14-crossing-time-{q}-{k}-{schedule}",
                        ).items()
                    },
                }
            )

        for high, low, contrast_label in CONTRASTS:
            for outcome in OUTCOMES:
                high_rows = {
                    int(row["seed"]): row
                    for row in rows
                    if f(row["q"]) == f(q) and int(row["k"]) == k and row["schedule"] == high
                }
                low_rows = {
                    int(row["seed"]): row
                    for row in rows
                    if f(row["q"]) == f(q) and int(row["k"]) == k and row["schedule"] == low
                }
                if set(high_rows) != set(SEEDS) or set(low_rows) != set(SEEDS):
                    raise RuntimeError(f"unpaired contrast {contrast_label} in q={q}, k={k}")
                differences = [
                    float(high_rows[seed][outcome]) - float(low_rows[seed][outcome]) for seed in SEEDS
                ]
                paired.append(
                    {
                        "inference_status": INFERENCE_STATUS,
                        "q": q,
                        "k": k,
                        "cell_label": cell_label,
                        "high_schedule": high,
                        "low_schedule": low,
                        "contrast": contrast_label,
                        "outcome": outcome,
                        "primary_endpoint": int(outcome == "final_rho_y"),
                        **bootstrap_interval(
                            differences,
                            label=f"e14-paired-{q}-{k}-{high}-{low}-{outcome}",
                        ),
                        "n_pairs": len(differences),
                    }
                )

    for run in runs:
        boundary_rho = float(get(run.summary, "boundary.final.rho_y"))
        trajectory.append(
            {
                "inference_status": INFERENCE_STATUS,
                "q": run.q,
                "k": run.k,
                "cell": "responsive" if (run.q, run.k) == (0.75, 4) else "negative_control",
                "schedule": run.schedule,
                "seed": run.seed,
                "phase": "boundary",
                "phase_step": 0,
                "global_step": 64,
                "rho_y": boundary_rho,
            }
        )
        for local, global_step, value in metric_series(
            run, stage="phase_b", split="conflict_eval", metric="rho_y"
        ):
            trajectory.append(
                {
                    "inference_status": INFERENCE_STATUS,
                    "q": run.q,
                    "k": run.k,
                    "cell": "responsive" if (run.q, run.k) == (0.75, 4) else "negative_control",
                    "schedule": run.schedule,
                    "seed": run.seed,
                    "phase": "phase_b",
                    "phase_step": local,
                    "global_step": global_step,
                    "rho_y": value,
                }
            )
    return cell_schedule, paired, crossing, trajectory


def factorial_timing_effects(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Post-hoc 2x2 decomposition of the four carry-state schedules."""
    definitions: tuple[tuple[str, tuple[tuple[str, float], ...]], ...] = (
        (
            "phase_A_effect_when_phase_B_zero",
            (("early_only", 1.0), ("zero_zero", -1.0)),
        ),
        (
            "phase_A_effect_when_phase_B_high",
            (("high_high", 1.0), ("delayed_carry", -1.0)),
        ),
        (
            "phase_B_effect_when_phase_A_zero",
            (("delayed_carry", 1.0), ("zero_zero", -1.0)),
        ),
        (
            "phase_B_effect_when_phase_A_high",
            (("high_high", 1.0), ("early_only", -1.0)),
        ),
        (
            "phase_A_by_phase_B_interaction",
            (
                ("high_high", 1.0),
                ("early_only", -1.0),
                ("delayed_carry", -1.0),
                ("zero_zero", 1.0),
            ),
        ),
    )
    results: list[dict[str, Any]] = []
    for q, k, cell_label in CELLS:
        by_schedule_seed = {
            (str(row["schedule"]), int(row["seed"])): row
            for row in rows
            if f(row["q"]) == f(q) and int(row["k"]) == k
        }
        for effect_name, terms in definitions:
            for outcome in OUTCOMES:
                seed_values = [
                    sum(
                        coefficient * float(by_schedule_seed[(schedule, seed)][outcome])
                        for schedule, coefficient in terms
                    )
                    for seed in SEEDS
                ]
                results.append(
                    {
                        "inference_status": INFERENCE_STATUS,
                        "contrast_status": "unregistered_post_hoc_factorial_decomposition",
                        "q": q,
                        "k": k,
                        "cell_label": cell_label,
                        "effect": effect_name,
                        "formula": " + ".join(
                            f"{coefficient:+g}*{schedule}" for schedule, coefficient in terms
                        ).lstrip("+"),
                        "outcome": outcome,
                        **bootstrap_interval(
                            seed_values,
                            label=f"e14-factorial-{q}-{k}-{effect_name}-{outcome}",
                        ),
                        "n_pairs": len(seed_values),
                    }
                )
    row_index = {(f(row["q"]), int(row["k"]), str(row["schedule"]), int(row["seed"])): row for row in rows}
    cross_cell_effects = {
        "phase_A_effect_when_phase_B_high",
        "phase_B_effect_when_phase_A_zero",
    }
    for effect_name, terms in definitions:
        if effect_name not in cross_cell_effects:
            continue
        for outcome in OUTCOMES:
            seed_values = []
            for seed in SEEDS:
                responsive = sum(
                    coefficient * float(row_index[(0.75, 4, schedule, seed)][outcome])
                    for schedule, coefficient in terms
                )
                comparison = sum(
                    coefficient * float(row_index[(0.90, 3, schedule, seed)][outcome])
                    for schedule, coefficient in terms
                )
                seed_values.append(responsive - comparison)
            results.append(
                {
                    "inference_status": INFERENCE_STATUS,
                    "contrast_status": "unregistered_post_hoc_cross_cell_difference_in_simple_effects",
                    "q": "0.75 minus 0.90",
                    "k": "4 minus 3",
                    "cell_label": "responsive boundary minus preselected comparison",
                    "effect": f"cross_cell_difference_in_{effect_name}",
                    "formula": (
                        "responsive[("
                        + " + ".join(
                            f"{coefficient:+g}*{schedule}" for schedule, coefficient in terms
                        ).lstrip("+")
                        + ")] - comparison[same effect]"
                    ),
                    "outcome": outcome,
                    **bootstrap_interval(
                        seed_values,
                        label=f"e14-factorial-cross-cell-{effect_name}-{outcome}",
                    ),
                    "n_pairs": len(seed_values),
                }
            )
    return results


def plot_font_family() -> str:
    """Register Myriad Pro from a portable location, or use a bundled fallback."""

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
    for font_dir in font_dirs:
        for filename in filenames:
            path = font_dir / filename
            if path.is_file():
                fm.fontManager.addfont(str(path))
                names.append(fm.FontProperties(fname=str(path)).get_name())
    if not names:
        warnings.warn(
            "Myriad Pro was not found in FORKWORLD_FONT_DIR or ~/Library/Fonts; "
            "falling back to DejaVu Sans. Figure geometry may differ slightly.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "DejaVu Sans"
    return names[0]


def configure_style() -> None:
    font_family = plot_font_family()
    mpl.rcParams.update(
        {
            "font.family": font_family,
            "font.size": 9.3,
            "axes.titlesize": 10.3,
            "axes.titleweight": 600,
            "axes.labelsize": 9.3,
            "axes.labelcolor": INK,
            "axes.edgecolor": "#AAB2BD",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": "#4B5563",
            "ytick.color": "#4B5563",
            "legend.frameon": False,
            "legend.fontsize": 8.1,
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
        xytext=(-12, 7),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=11,
        fontweight=600,
        color=INK,
        annotation_clip=False,
    )


def point_ci(values: Sequence[float], label: str) -> tuple[float, float, float]:
    result = bootstrap_interval(values, label=label)
    return float(result["estimate"]), float(result["ci_low"]), float(result["ci_high"])


def figure_e14(
    rows: Sequence[Mapping[str, Any]],
    paired: Sequence[Mapping[str, Any]],
    trajectory: Sequence[Mapping[str, Any]],
) -> None:
    configure_style()
    figure = plt.figure(figsize=(7.35, 8.15))
    grid = figure.add_gridspec(
        3,
        2,
        height_ratios=(1.0, 1.12, 0.40),
        left=0.16,
        right=0.96,
        top=0.885,
        bottom=0.035,
        wspace=0.34,
        hspace=0.62,
    )
    axes = [figure.add_subplot(grid[row, column]) for row in range(2) for column in range(2)]
    legend_axis = figure.add_subplot(grid[2, :])
    legend_axis.set_axis_off()

    for axis_index, (axis, (q, k, cell_title)) in enumerate(zip(axes[:2], CELLS, strict=True)):
        for schedule in SCHEDULES:
            trajectory_part = [
                item
                for item in trajectory
                if f(item["q"]) == f(q) and int(item["k"]) == k and item["schedule"] == schedule
            ]
            steps = (0, *PHASE_B_LOCAL_STEPS)
            trajectory_centers: list[float] = []
            trajectory_lows: list[float] = []
            trajectory_highs: list[float] = []
            for step in steps:
                values = [float(item["rho_y"]) for item in trajectory_part if int(item["phase_step"]) == step]
                center, low, high = point_ci(values, f"fig14-trajectory-{q}-{k}-{schedule}-{step}")
                trajectory_centers.append(center)
                trajectory_lows.append(low)
                trajectory_highs.append(high)
            x = np.asarray(steps, dtype=float)
            axis.plot(x, trajectory_centers, color=SCHEDULE_COLORS[schedule], lw=1.7)
            axis.fill_between(
                x,
                trajectory_lows,
                trajectory_highs,
                color=SCHEDULE_COLORS[schedule],
                alpha=0.09,
                lw=0,
            )
        axis.axhline(0.5, color="#98A2B3", lw=0.8, ls=":")
        axis.set_xscale("symlog", linthresh=1, base=2)
        axis.set_xticks((0, 1, 8, 64, 1984), ("0", "1", "8", "64", "1,984"))
        axis.set_ylim(-0.035, 1.035)
        axis.set_xlabel("Updates after the phase boundary")
        axis.set_ylabel("Conflict-set intended reliance" if axis_index == 0 else "")
        if axis_index == 1:
            axis.tick_params(labelleft=False)
        axis.set_title(
            f"{DISPLAY_CELL_LABELS[cell_title]}  ($q={q:g},\ k={k}$)",
            loc="left",
            pad=7,
        )
        axis.grid(axis="y", color=LIGHT_GRAY, lw=0.65)
        panel_label(axis, chr(ord("a") + axis_index))

    axis_c = axes[2]
    contrast_labels = [label for _, _, label in CONTRASTS]
    contrast_tick_labels = (
        "early-only\n$-$ never",
        "delayed\n$-$ never",
        "early-only\n$-$ delayed",
        "continuous\n$-$ early-only",
        "actor-Adam reset\n$-$ delayed",
        "critic reset\n$-$ delayed",
    )
    y = np.arange(len(contrast_labels), dtype=float)
    for offset, (q, k, cell_label), color, marker in (
        (-0.10, CELLS[0], SKY, "o"),
        (0.10, CELLS[1], ROSE, "s"),
    ):
        paired_lookup = {
            str(item["contrast"]): item
            for item in paired
            if f(item["q"]) == f(q) and int(item["k"]) == k and item["outcome"] == "final_rho_y"
        }
        contrast_centers = np.asarray([float(paired_lookup[label]["estimate"]) for label in contrast_labels])
        contrast_lows = np.asarray([float(paired_lookup[label]["ci_low"]) for label in contrast_labels])
        contrast_highs = np.asarray([float(paired_lookup[label]["ci_high"]) for label in contrast_labels])
        axis_c.errorbar(
            contrast_centers,
            y + offset,
            xerr=np.vstack((contrast_centers - contrast_lows, contrast_highs - contrast_centers)),
            fmt=marker,
            color=color,
            ms=4.2,
            capsize=2,
            lw=1.2,
            label=DISPLAY_CELL_LABELS[cell_label],
        )
    axis_c.axvline(0, color="#98A2B3", lw=0.9)
    axis_c.set_yticks(y, contrast_tick_labels)
    axis_c.invert_yaxis()
    axis_c.set_xlim(-0.72, 0.72)
    axis_c.set_xlabel("Paired difference in final intended reliance")
    axis_c.tick_params(axis="y", labelsize=9.6, pad=5)
    axis_c.tick_params(axis="x", labelsize=9.2)
    axis_c.set_title("Fixed seed-paired contrasts", loc="left", pad=7, fontsize=10.7)
    axis_c.grid(axis="x", color=LIGHT_GRAY, lw=0.65)
    panel_label(axis_c, "c")

    axis_d = axes[3]
    x_schedule = np.arange(len(SCHEDULES), dtype=float)
    for plateau in np.arange(0, 17) / 16:
        axis_d.axhline(
            plateau,
            color="#D9DEE5" if plateau % 4 else "#BAC2CC",
            lw=0.45 if plateau % 4 else 0.65,
            zorder=0,
        )
    for cell_offset, q, k, label, color, marker in (
        (-0.13, 0.75, 4, "responsive boundary", SKY, "o"),
        (0.13, 0.90, 3, "canonical comparison", ROSE, "s"),
    ):
        for schedule_index, schedule in enumerate(SCHEDULES):
            schedule_rows = sorted(
                (
                    row
                    for row in rows
                    if f(row["q"]) == f(q) and int(row["k"]) == k and row["schedule"] == schedule
                ),
                key=lambda row: int(row["seed"]),
            )
            jitter = np.linspace(-0.055, 0.055, len(schedule_rows))
            axis_d.scatter(
                schedule_index + cell_offset + jitter,
                [float(row["final_rho_y"]) for row in schedule_rows],
                color=color,
                marker=marker,
                s=15,
                alpha=0.72,
                linewidths=0,
                zorder=2,
                label=label if schedule_index == 0 else None,
            )
    axis_d.set_xticks(
        x_schedule,
        ("never", "cont.", "early\nonly", "delayed", "actor\nreset", "critic\nreset"),
        rotation=0,
        ha="center",
    )
    axis_d.tick_params(axis="x", labelsize=8.8, pad=5)
    axis_d.set_ylim(-0.035, 1.035)
    axis_d.set_ylabel("Final intended reliance")
    axis_d.set_title(
        "Final reliance clusters near\ndiscrete-code fractions",
        loc="left",
        pad=7,
        linespacing=1.0,
    )
    axis_d.text(
        0.985,
        0.965,
        "grid spacing $=1/2^k$",
        transform=axis_d.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color=GRAY,
    )
    panel_label(axis_d, "d")

    schedule_handles = [
        Line2D(
            [0],
            [0],
            color=SCHEDULE_COLORS[name],
            lw=1.8,
            label=SCHEDULE_LABELS[name],
        )
        for name in SCHEDULES
    ]
    cell_handles = [
        Line2D([0], [0], color=SKY, marker="o", lw=1.4, ms=4, label="responsive boundary"),
        Line2D(
            [0],
            [0],
            color=ROSE,
            marker="s",
            lw=1.4,
            ms=4,
            label="canonical comparison",
        ),
    ]
    legend_axis.legend(
        handles=[*schedule_handles, *cell_handles],
        loc="center",
        ncol=4,
        columnspacing=1.30,
        handlelength=1.9,
        labelspacing=0.85,
        fontsize=8.7,
    )
    figure.suptitle(
        "Entropy timing around policy saturation",
        x=0.16,
        ha="left",
        y=0.975,
        fontsize=12.3,
        fontweight=600,
    )
    figure.text(
        0.16,
        0.935,
        "Exploratory, post-hoc, and specific to the step-64 boundary; bands and intervals are 95% seed bootstraps.",
        ha="left",
        va="center",
        fontsize=9.2,
        color=GRAY,
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / "fig14_entropy_timing_exploratory.pdf")
    figure.savefig(FIGURES / "fig14_entropy_timing_exploratory.png", dpi=220)
    plt.close(figure)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def key_results(
    cell_schedule: Sequence[Mapping[str, Any]], paired: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    primary = [row for row in paired if row["outcome"] == "final_rho_y"]
    auc = [row for row in paired if row["outcome"] == "phase_b_rho_y_area"]
    reset = [
        row
        for row in paired
        if row["high_schedule"] in {"delayed_actor_reset", "delayed_critic_reset"}
        and row["outcome"]
        in {
            "final_rho_y",
            "phase_b_rho_y_area",
            "phase_b_critic_loss_log_grid_mean",
            "actor_relative_l2_from_boundary",
            "critic_relative_l2_from_boundary",
        }
    ]
    final_cells = [
        {
            key: value
            for key, value in row.items()
            if key
            in {
                "q",
                "k",
                "cell_label",
                "schedule",
                "final_rho_y_estimate",
                "final_rho_y_ci_low",
                "final_rho_y_ci_high",
                "phase_b_rho_y_area_estimate",
                "crossing_observed_estimate",
                "boundary_policy_entropy_estimate",
                "final_policy_entropy_estimate",
                "final_exact_hard_flip_estimate",
                "final_proxy_hard_flip_estimate",
                "final_quantization_abs_residual_estimate",
                "final_within_0_01_of_interaction_plateau_estimate",
            }
        }
        for row in cell_schedule
    ]
    return {
        "caveat": (
            "E14 is exploratory/post-hoc after E10 and its estimates are specific to "
            "the chosen step-64 boundary; they require independent replication."
        ),
        "primary_final_rho_y_paired_contrasts": primary,
        "phase_b_area_paired_contrasts": auc,
        "component_reset_diagnostics": reset,
        "cell_schedule_snapshot": final_cells,
    }


def quantization_diagnostic(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Summarise the unregistered discrete-code plateau pattern.

    In a balanced degree-k binary interaction, a deterministic policy that gets
    a subset of interaction codes right has intended reliance in increments of
    1 / 2**k.  Near-grid values are compatible with class-level acquisition,
    but are partly induced by the discrete evaluation support and do not by
    themselves identify a learning mechanism.
    """
    summaries: list[dict[str, Any]] = []
    for q, k, cell_label in CELLS:
        for schedule in (*SCHEDULES, "all_schedules"):
            part = [
                row
                for row in rows
                if f(row["q"]) == f(q)
                and int(row["k"]) == k
                and (schedule == "all_schedules" or row["schedule"] == schedule)
            ]
            residuals = [float(row["final_quantization_abs_residual"]) for row in part]
            histogram = Counter(int(row["final_quantization_plateau_index"]) for row in part)
            summaries.append(
                {
                    "inference_status": INFERENCE_STATUS,
                    "diagnostic_status": "unregistered_exploratory_description",
                    "q": q,
                    "k": k,
                    "cell_label": cell_label,
                    "schedule": schedule,
                    "plateau_spacing": 1 / (2**k),
                    "n_runs": len(part),
                    "n_within_0_01": sum(
                        int(row["final_within_0_01_of_interaction_plateau"]) for row in part
                    ),
                    "fraction_within_0_01": mean(
                        float(row["final_within_0_01_of_interaction_plateau"]) for row in part
                    ),
                    "mean_absolute_residual": mean(residuals),
                    "max_absolute_residual": max(residuals),
                    "plateau_index_counts": json.dumps(dict(sorted(histogram.items()))),
                }
            )
    all_residuals = [float(row["final_quantization_abs_residual"]) for row in rows]
    overall = {
        "diagnostic_status": "unregistered_exploratory_description",
        "n_runs": len(rows),
        "n_within_0_01": sum(int(row["final_within_0_01_of_interaction_plateau"]) for row in rows),
        "fraction_within_0_01": mean(float(row["final_within_0_01_of_interaction_plateau"]) for row in rows),
        "mean_absolute_residual": mean(all_residuals),
        "max_absolute_residual": max(all_residuals),
        "max_abs_exact_flip_minus_intended_reliance": max(
            abs(float(row["final_exact_hard_flip"]) - float(row["final_rho_y"])) for row in rows
        ),
        "max_abs_proxy_flip_minus_complement_reliance": max(
            abs(float(row["final_proxy_hard_flip"]) - (1.0 - float(row["final_rho_y"]))) for row in rows
        ),
        "causal_identity_runs": sum(
            math.isclose(
                float(row["final_exact_hard_flip"]),
                float(row["final_rho_y"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and math.isclose(
                float(row["final_proxy_hard_flip"]),
                1.0 - float(row["final_rho_y"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for row in rows
        ),
        "interpretive_guardrail": (
            "Near-grid values are compatible with acquiring whole interaction-code "
            "classes, but the grid is also implied by deterministic behavior on a "
            "discrete balanced support; this is not independent mechanistic evidence."
        ),
    }
    return summaries, overall


def replay_codeword_policies(
    runs: Sequence[Run],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Retrain four fixed representatives and audit behavior by R codeword.

    Representatives were selected after seeing the endpoint quantization, so
    this replay is a post-hoc mechanistic description rather than another
    independent experiment.  The artifact endpoint is checked exactly before a
    codeword row is admitted.
    """
    lookup = {(run.q, run.k, run.schedule, run.seed): run for run in runs}
    codeword_rows: list[dict[str, Any]] = []
    replay_summaries: list[dict[str, Any]] = []
    for q, k, schedule, seed in REPLAY_CASES:
        key = (f(q), k, schedule, seed)
        if key not in lookup:
            raise RuntimeError(f"missing fixed codeword-replay case: {key}")
        artifact_run = lookup[key]
        replay = run_protocol(artifact_run.config, seed)
        batch = replay.evaluation_batch
        if batch is None:
            raise RuntimeError(f"codeword replay exposed no evaluation batch: {key}")
        logits = predict_logits(replay.model, batch, artifact_run.config)
        predictions = binary_predictions(logits)
        intended = np.asarray(batch.y, dtype=np.int8)
        proxy = np.asarray(batch.channels["P"], dtype=np.int8)
        codes = np.column_stack(
            [np.asarray(batch.channels[f"R_{index}"], dtype=np.int8) for index in range(1, k + 1)]
        )
        replay_rho_y = float(np.mean(predictions == intended))
        replay_rho_p = float(np.mean(predictions == proxy))
        artifact_rho_y = float(get(artifact_run.summary, "final.rho_y"))
        artifact_rho_p = float(get(artifact_run.summary, "final.rho_p"))
        if replay_rho_y != artifact_rho_y or replay_rho_p != artifact_rho_p:
            raise RuntimeError(
                "fixed replay failed exact endpoint equality for "
                f"{key}: artifact ({artifact_rho_y}, {artifact_rho_p}) vs "
                f"replay ({replay_rho_y}, {replay_rho_p})"
            )
        weighted_rho_y = 0.0
        case_rows: list[dict[str, Any]] = []
        for code in product((-1, 1), repeat=k):
            code_array = np.asarray(code, dtype=np.int8)
            mask = np.all(codes == code_array[None, :], axis=1)
            n = int(np.sum(mask))
            if n == 0:
                raise RuntimeError(f"fixed replay {key} omitted codeword {code}")
            intended_values = np.unique(intended[mask])
            proxy_values = np.unique(proxy[mask])
            if len(intended_values) != 1 or len(proxy_values) != 1:
                raise RuntimeError(f"codeword did not uniquely determine Y and P in {key}: {code}")
            y_rate = float(np.mean(predictions[mask] == intended[mask]))
            p_rate = float(np.mean(predictions[mask] == proxy[mask]))
            tie_rate = float(np.mean(predictions[mask] == 0))
            if not math.isclose(y_rate + p_rate + tie_rate, 1.0, abs_tol=1e-12):
                raise RuntimeError(f"codeword replay accounting failed in {key}: {code}")
            if y_rate == 1.0:
                routing = "intended_on_all_rows"
            elif p_rate == 1.0:
                routing = "proxy_on_all_rows"
            elif y_rate > p_rate:
                routing = "mostly_intended"
            elif p_rate > y_rate:
                routing = "mostly_proxy"
            else:
                routing = "tie_or_other"
            row = {
                "inference_status": INFERENCE_STATUS,
                "diagnostic_status": "fixed_post_hoc_retraining_replay",
                "q": q,
                "k": k,
                "schedule": schedule,
                "seed": seed,
                "codeword": " ".join(f"{value:+d}" for value in code),
                "codeword_index": sum((value > 0) << index for index, value in enumerate(code)),
                "rows": n,
                "intended_sign": int(intended_values[0]),
                "proxy_sign": int(proxy_values[0]),
                "intended_follow_rate": y_rate,
                "proxy_follow_rate": p_rate,
                "tie_rate": tie_rate,
                "routing_class": routing,
            }
            case_rows.append(row)
            codeword_rows.append(row)
            weighted_rho_y += n * y_rate / len(predictions)
        if not math.isclose(weighted_rho_y, replay_rho_y, rel_tol=0.0, abs_tol=1e-15):
            raise RuntimeError(
                f"codeword grouping failed exact rho_y reconstruction for {key}: "
                f"{weighted_rho_y} vs {replay_rho_y}"
            )
        replay_summaries.append(
            {
                "inference_status": INFERENCE_STATUS,
                "diagnostic_status": "fixed_post_hoc_retraining_replay",
                "q": q,
                "k": k,
                "schedule": schedule,
                "seed": seed,
                "artifact_rho_y": artifact_rho_y,
                "replayed_rho_y": replay_rho_y,
                "artifact_rho_p": artifact_rho_p,
                "replayed_rho_p": replay_rho_p,
                "codeword_weighted_rho_y": weighted_rho_y,
                "strict_intended_codewords": sum(
                    row["routing_class"] == "intended_on_all_rows" for row in case_rows
                ),
                "strict_proxy_codewords": sum(
                    row["routing_class"] == "proxy_on_all_rows" for row in case_rows
                ),
                "mixed_codewords": sum(
                    row["routing_class"] not in {"intended_on_all_rows", "proxy_on_all_rows"}
                    for row in case_rows
                ),
                "predominantly_intended_codewords": sum(
                    float(row["intended_follow_rate"]) > 0.5 for row in case_rows
                ),
                "total_codewords": 2**k,
                "endpoint_equality_exact": True,
            }
        )
    if len(codeword_rows) != sum(2**k for _, k, _, _ in REPLAY_CASES):
        raise RuntimeError("fixed codeword replay emitted an unexpected row count")
    return codeword_rows, replay_summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--e10-artifacts", type=Path, default=DEFAULT_E10_ARTIFACTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts_root = args.artifacts.resolve()
    e10_artifacts_root = args.e10_artifacts.resolve()
    runs, audit = load_runs(artifacts_root)
    invariance = audit_phase_invariance(runs, e10_artifacts_root)
    audit.update(
        {
            "e10_reference_artifacts_root": str(e10_artifacts_root),
            "e10_reference_implementation_fingerprints": invariance[
                "e10_reference_implementation_fingerprints"
            ],
        }
    )
    rows = [run_row(run) for run in runs]
    cell_schedule, paired, crossing, trajectory = analyse_rows(rows, runs)
    factorial = factorial_timing_effects(rows)
    quantization_rows, quantization_overall = quantization_diagnostic(rows)
    codeword_rows, codeword_summaries = replay_codeword_policies(runs)

    write_csv(DERIVED / "e14_runs.csv", rows)
    write_csv(DERIVED / "e14_cell_schedule_estimates.csv", cell_schedule)
    write_csv(DERIVED / "e14_fixed_paired_contrasts.csv", paired)
    write_csv(DERIVED / "e14_crossing_summary.csv", crossing)
    write_csv(DERIVED / "e14_trajectory.csv", trajectory)
    write_csv(DERIVED / "e14_factorial_timing_effects.csv", factorial)
    write_csv(DERIVED / "e14_quantization_diagnostic.csv", quantization_rows)
    write_csv(DERIVED / "e14_codeword_replay.csv", codeword_rows)
    write_csv(DERIVED / "e14_codeword_replay_summary.csv", codeword_summaries)
    analysis = {
        "inference_status": INFERENCE_STATUS,
        "audit": audit,
        "provenance": {
            "e14_artifacts_root": str(artifacts_root),
            "e14_implementation_fingerprint": audit["implementation_fingerprint"],
            "e10_reference_artifacts_root": str(e10_artifacts_root),
            "e10_reference_implementation_fingerprints": invariance[
                "e10_reference_implementation_fingerprints"
            ],
        },
        "phase_and_continuous_schedule_invariance": invariance,
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "interval": "two-sided percentile 95%",
            "independent_unit": "training_seed",
            "paired_contrasts": True,
        },
        "crossing": {
            "definition": "first of two consecutive evaluations with rho_y >= 0.50",
            "censoring": "not observed by phase-B update 1984",
            "restricted_time_for_censored_runs": 1_984,
            "summary": crossing,
        },
        "critic_loss": {
            "summary": (
                "trapezoidal time-weighted mean over the fixed logged phase-B update "
                "grid (local updates 1 through 1984)"
            ),
            "confirmatory_status": "secondary exploratory diagnostic",
        },
        "quantization_diagnostic": quantization_overall,
        "post_hoc_factorial_timing_effects": factorial,
        "fixed_post_hoc_codeword_replays": {
            "selection": [list(case) for case in REPLAY_CASES],
            "selection_status": "representatives_fixed_after_endpoint_quantization_was_observed",
            "artifact_endpoint_equality_required": True,
            "summaries": codeword_summaries,
            "interpretation": (
                "The representatives test whether aggregate intended reliance is a "
                "within-row stochastic mixture or routing by discrete R codeword. "
                "They are descriptive replays, not independent replications."
            ),
        },
        "results": key_results(cell_schedule, paired),
    }
    json_dump(DERIVED / "e14_analysis.json", analysis)
    figure_e14(rows, paired, trajectory)
    print(
        "E14 strict audit passed: 120/120 COMPLETE, one implementation fingerprint, "
        "phase/reset/continuous invariants verified."
    )
    print(f"Wrote E14 analysis to {DERIVED}")
    print(f"Wrote Figure 14 to {FIGURES}")


if __name__ == "__main__":
    main()
