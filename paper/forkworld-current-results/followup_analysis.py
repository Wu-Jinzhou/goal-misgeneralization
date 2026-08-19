#!/usr/bin/env python3
"""Audit, analyse, and plot the prospectively specified Forkworld follow-ups.

The default invocation is deliberately strict::

    .venv/bin/python paper/forkworld-current-results/followup_analysis.py

It refuses to analyse a partial or malformed grid.  During an active run,
``--allow-incomplete`` produces explicitly provisional tables and figures from
the completed subset while retaining all structural checks that can already be
made.  Training seed is the inferential unit throughout: planned nuisance cells
are collapsed within seed before the 4,000-draw percentile bootstrap.

Inputs are the immutable run summaries, their sibling ``resolved_config.yaml``
files, and (for the RL trajectory outcomes only) ``metrics.jsonl``.  Outputs are
machine-readable CSV/JSON files in ``derived/`` and four publication figures in
``figures/``.  No source code or paper narrative is read or modified.
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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from statistics import mean
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/forkworld-followup-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/forkworld-followup-xdg")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import yaml  # type: ignore[import-untyped]
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_ARTIFACTS = REPO / "artifacts-followups"
FIGURES = HERE / "figures"
DERIVED = HERE / "derived"

SEEDS = (11, 23, 37, 41, 53, 67, 71, 83, 97, 101)
BOOTSTRAP_DRAWS = 4_000
ARTIFACT_SCHEMA_VERSION = 1
SOURCE_FINGERPRINT_SCHEMA_VERSION = 1
NON_SCIENTIFIC_RUN_FIELDS = frozenset({"output_root", "resume", "seeds"})
REQUIRED_RUN_FILES = frozenset(
    {"COMPLETE", "metadata.json", "metrics.jsonl", "status.json", "summary.json"}
)

EXPERIMENT_MODES = {
    "e10": "exploration_entropy",
    "e11": "calibrated_rule_competition",
    "e12": "calibrated_three_goal_competition",
    "e13": "repeated_reward_relevant_forks",
}

BLUE = "#2673B8"
ORANGE = "#D55E00"
TEAL = "#009E73"
PURPLE = "#7A5195"
GOLD = "#C99700"
SKY = "#56B4E9"
ROSE = "#CC79A7"
GRAY = "#667085"
LIGHT_GRAY = "#E6E9EE"
INK = "#17212B"


def get(mapping: Mapping[str, Any], path: str, default: Any = None) -> Any:
    value: Any = mapping
    for name in path.split("."):
        if not isinstance(value, Mapping) or name not in value:
            return default
        value = value[name]
    return value


def f(value: Any) -> float:
    return round(float(value), 8)


def trapezoidal_integral(y: np.ndarray[Any, Any], x: np.ndarray[Any, Any]) -> float:
    """Integrate with the NumPy 2 API and its numerically identical 1.26 alias."""

    try:
        return float(np.trapezoid(y, x))
    except AttributeError:  # NumPy 1.26
        return float(np.trapz(y, x))  # type: ignore[attr-defined]


def stable_seed(*parts: Any) -> int:
    digest = hashlib.blake2b(digest_size=8, person=b"fwfollow")
    for part in parts:
        encoded = str(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
    return int.from_bytes(digest.digest(), "little")


@dataclass(frozen=True)
class Run:
    family: str
    path: Path
    summary: Mapping[str, Any]
    config: Mapping[str, Any]

    @property
    def seed(self) -> int:
        return int(self.summary["seed"])


@dataclass(frozen=True)
class ExperimentSpec:
    family: str
    hypothesis: str
    name: str
    expected_count: int
    key: Callable[[Mapping[str, Any], int], tuple[Any, ...]]
    expected_keys: Callable[[], set[tuple[Any, ...]]]


def e10_arm(config: Mapping[str, Any]) -> str:
    mode = str(get(config, "update.mode", "full"))
    estimator = str(get(config, "h5.rl_estimator", "actor_critic"))
    if mode == "subspace":
        return "actor_critic_subspace1024"
    if estimator == "reinforce":
        return "reinforce_full"
    return "actor_critic_full"


def e10_key(config: Mapping[str, Any], seed: int) -> tuple[Any, ...]:
    return (
        e10_arm(config),
        f(get(config, "data.q")),
        int(get(config, "data.k")),
        f(get(config, "h5.entropy_coefficient")),
        seed,
    )


def expected_e10() -> set[tuple[Any, ...]]:
    result = {
        ("actor_critic_full", f(q), k, f(beta), seed)
        for q, k, beta, seed in product(
            (0.50, 0.75, 0.90, 0.99),
            (2, 3, 4),
            (0.0, 0.003, 0.01, 0.03, 0.10, 0.30),
            SEEDS,
        )
    }
    for arm in ("reinforce_full", "actor_critic_subspace1024"):
        result.update(
            (arm, f(0.9), 3, f(beta), seed)
            for beta, seed in product((0.0, 0.003, 0.01, 0.03, 0.10, 0.30), SEEDS)
        )
    return result


def e11_key(config: Mapping[str, Any], seed: int) -> tuple[Any, ...]:
    return (
        str(get(config, "data.target_rule")),
        f(get(config, "data.q")),
        int(get(config, "model.width")),
        int(get(config, "model.depth")),
        seed,
    )


def expected_e11() -> set[tuple[Any, ...]]:
    return {
        (rule, f(q), width, depth, seed)
        for rule, q, width, depth, seed in product(
            ("parity", "majority", "conjunction", "multiplexer"),
            (0.75, 0.90, 0.99),
            (8, 16, 32, 64, 128),
            (1, 2),
            SEEDS,
        )
    }


def e12_key(config: Mapping[str, Any], seed: int) -> tuple[Any, ...]:
    return (
        f(get(config, "h10.q_p")),
        f(get(config, "h10.q_q")),
        int(get(config, "h10.k_q")),
        int(get(config, "h10.k_y")),
        int(get(config, "model.width")),
        str(get(config, "h10.error_structure")),
        seed,
    )


def expected_e12() -> set[tuple[Any, ...]]:
    return {
        (f(qp), f(qq), kq, ky, width, overlap, seed)
        for qp, qq, kq, ky, width, overlap, seed in product(
            (0.90, 0.99),
            (0.90, 0.95, 0.99),
            (2, 3),
            (3, 5),
            (16, 64),
            ("independent", "nested"),
            SEEDS,
        )
    }


def e13_key(config: Mapping[str, Any], seed: int) -> tuple[Any, ...]:
    return (
        int(get(config, "h11.route_depth")),
        f(get(config, "data.q")),
        int(get(config, "data.k")),
        int(get(config, "model.width")),
        str(get(config, "h11.evidence_regime")),
        seed,
    )


def expected_e13() -> set[tuple[Any, ...]]:
    return {
        (depth, f(q), k, width, regime, seed)
        for depth, q, k, width, regime, seed in product(
            (1, 2, 4),
            (0.75, 0.95, 0.99),
            (2, 4),
            (16, 64),
            ("fixed_total", "per_fork_matched"),
            SEEDS,
        )
    }


SPECS = (
    ExperimentSpec("e10", "h5", "rl_entropy_shortcut_boundary", 840, e10_key, expected_e10),
    ExperimentSpec("e11", "h2", "exact_rule_families", 1_200, e11_key, expected_e11),
    ExperimentSpec("e12", "h10", "competing_goal_frontier", 960, e12_key, expected_e12),
    ExperimentSpec("e13", "h11", "routeworld_complexity", 720, e13_key, expected_e13),
)


def expected_artifact_run_id(
    config: Mapping[str, Any], metadata: Mapping[str, Any]
) -> str:
    """Reconstruct the RunStore identity from an immutable resolved artifact.

    ``resolved_config.yaml`` adds the seed after the run identity is computed,
    so it is removed here along with the explicitly non-scientific run fields.
    Top-level expansion bookkeeping (the underscore-prefixed keys) is excluded
    by the same canonicalisation rule used by :mod:`forkworld.config`.
    """

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


def read_json_object(path: Path) -> Mapping[str, Any]:
    """Read one strict JSON object, adding the artifact path to failures."""

    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid JSON artifact {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise RuntimeError(f"JSON artifact is not an object: {path}")
    return value


def validate_metrics_file(
    path: Path, *, run_id: str, seed: int, hypothesis: str
) -> int:
    """Validate JSONL syntax, record schema, and per-record run identity."""

    required = {
        "condition",
        "examples_seen",
        "experiment",
        "global_step",
        "intervention",
        "level",
        "metric",
        "n",
        "run_id",
        "seed",
        "split",
        "stage",
        "stage_step",
        "value",
    }
    records = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise RuntimeError(f"blank metrics record at {path}:{line_number}")
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RuntimeError(
                        f"invalid metrics JSON at {path}:{line_number}: {error}"
                    ) from error
                if not isinstance(record, Mapping):
                    raise RuntimeError(f"metrics record is not an object at {path}:{line_number}")
                missing = required - set(record)
                if missing:
                    raise RuntimeError(
                        f"metrics record at {path}:{line_number} lacks {sorted(missing)}"
                    )
                record_seed = record["seed"]
                if (
                    record["run_id"] != run_id
                    or not isinstance(record_seed, int)
                    or isinstance(record_seed, bool)
                    or record_seed != seed
                ):
                    raise RuntimeError(
                        f"metrics identity mismatch at {path}:{line_number}: "
                        f"run_id={record['run_id']!r}, seed={record['seed']!r}"
                    )
                if record["experiment"] != hypothesis:
                    raise RuntimeError(
                        f"metrics experiment mismatch at {path}:{line_number}: "
                        f"{record['experiment']!r}"
                    )
                for field in ("examples_seen", "global_step", "n", "stage_step"):
                    value = record[field]
                    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                        raise RuntimeError(
                            f"invalid {field} at {path}:{line_number}: {value!r}"
                        )
                for field in (
                    "condition",
                    "experiment",
                    "intervention",
                    "level",
                    "metric",
                    "split",
                    "stage",
                ):
                    if not isinstance(record[field], str) or not record[field]:
                        raise RuntimeError(
                            f"invalid {field} at {path}:{line_number}: {record[field]!r}"
                        )
                try:
                    metric_value = float(record["value"])
                except (TypeError, ValueError) as error:
                    raise RuntimeError(
                        f"non-numeric metric value at {path}:{line_number}: {record['value']!r}"
                    ) from error
                if not math.isfinite(metric_value):
                    raise RuntimeError(
                        f"non-finite metric value at {path}:{line_number}: {record['value']!r}"
                    )
                records += 1
    except OSError as error:
        raise RuntimeError(f"cannot read metrics artifact {path}: {error}") from error
    if records == 0:
        raise RuntimeError(f"empty metrics artifact: {path}")
    return records


def validate_fixed_design(spec: ExperimentSpec, config: Mapping[str, Any]) -> list[str]:
    """Return deviations from preregistered non-swept settings."""

    errors: list[str] = []

    def expect(path: str, expected: Any) -> None:
        actual = get(config, path)
        if actual != expected:
            errors.append(f"{path}={actual!r}, expected {expected!r}")

    for path, expected in (
        ("schema_version", 1),
        ("cases", []),
        ("sweep", {}),
        ("experiment.hypothesis", spec.hypothesis),
        ("experiment.mode", EXPERIMENT_MODES[spec.family]),
        ("run.device", "cpu"),
        ("run.task_levels", ["choice"]),
        ("run.navigation_episodes", 128),
        ("run.save_checkpoints", False),
        ("evaluation.acquisition_threshold", 0.9),
        ("evaluation.persistence", 2),
        ("evaluation.equivalence_margin", 0.05),
        ("evaluation.bootstrap_samples", BOOTSTRAP_DRAWS),
        ("evaluation.confidence", 0.95),
        ("evaluation.minimum_inferential_seeds", 3),
        ("evaluation.save_predictions", False),
        ("model.activation", "relu"),
        ("model.residual", False),
        ("model.bias", True),
        ("model.nuisance_bits", 0),
        ("train.algorithm", "clean_sft"),
        ("train.learning_rate", 0.003),
        ("train.weight_decay", 0.0),
        ("train.eval_steps", "log"),
        ("train.grad_clip", 1.0),
        ("update.subspace_seed", 1_729),
    ):
        expect(path, expected)
    if spec.family == "e10":
        for path, expected in (
            ("data.n_train", 10_000),
            ("data.n_validation", 4_000),
            ("data.n_eval", 10_000),
            ("data.max_k", 5),
            ("data.state_dim", 8),
            ("model.width", 64),
            ("model.depth", 2),
            ("train.steps", 2_048),
            ("train.batch_size", 250),
            ("h5.algorithm", "rl"),
            ("h5.nuisance_bits", 8),
            ("h5.nuisance_entropy", 0),
            ("h5.nuisance_weight", 1.0),
            ("h5.critic_width", 128),
            ("h5.critic_depth", 3),
        ):
            expect(path, expected)
        arm = e10_arm(config)
        if arm == "actor_critic_subspace1024":
            expect("update.budget", 1_024)
            expect("h5.rl_estimator", "actor_critic")
        elif arm == "reinforce_full":
            expect("update.mode", "full")
            expect("update.budget", "full")
        else:
            expect("update.mode", "full")
            expect("update.budget", "full")
            expect("h5.rl_estimator", "actor_critic")
    elif spec.family == "e11":
        for path, expected in (
            ("data.n_train", 10_000),
            ("data.n_validation", 4_000),
            ("data.n_eval", 10_000),
            ("data.max_k", 6),
            ("data.state_dim", 8),
            ("train.steps", 4_096),
            ("train.batch_size", 250),
            ("h2.target_accuracy", 0.95),
            ("h2.target_seed_fraction", 0.8),
            ("h2.persistence", 2),
            ("h2.parameter_match_tolerance", 0.05),
            ("update.mode", "full"),
            ("update.budget", "full"),
        ):
            expect(path, expected)
        expected_k = 6 if get(config, "data.target_rule") == "multiplexer" else 5
        expect("data.k", expected_k)
    elif spec.family == "e12":
        for path, expected in (
            ("data.n_train", 10_000),
            ("data.n_validation", 4_000),
            ("data.n_eval", 10_000),
            ("data.q", 0.9),
            ("data.k", 3),
            ("data.max_k", 5),
            ("data.state_dim", 8),
            ("model.depth", 2),
            ("train.steps", 2_048),
            ("train.batch_size", 250),
            ("h10.max_k_q", 3),
            ("h10.max_k_y", 5),
            ("h10.calibration_steps", 2_048),
            ("h10.mastery_threshold", 0.9),
            ("update.mode", "full"),
            ("update.budget", "full"),
        ):
            expect(path, expected)
    elif spec.family == "e13":
        for path, expected in (
            ("data.n_train", 4_000),
            ("data.n_validation", 2_000),
            ("data.n_eval", 2_000),
            ("data.max_k", 4),
            ("data.state_dim", 0),
            ("model.depth", 2),
            ("train.steps", 1_024),
            ("train.batch_size", 256),
            ("h11.max_depth", 4),
            ("h11.base_steps", 1_024),
            ("h11.physical_rollouts", 128),
            ("update.mode", "full"),
            ("update.budget", "full"),
        ):
            expect(path, expected)
    return errors


def load_experiment(
    root: Path, spec: ExperimentSpec, *, allow_incomplete: bool
) -> tuple[list[Run], dict[str, Any]]:
    directory = root / spec.hypothesis / spec.name
    if not directory.is_dir() and not allow_incomplete:
        raise RuntimeError(f"missing experiment directory: {directory}")

    if len(spec.expected_keys()) != spec.expected_count:
        raise RuntimeError(f"internal expected-grid count mismatch for {spec.family}")
    runs: list[Run] = []
    incomplete_dirs: list[str] = []
    orphan_complete: list[str] = []
    unexpected_name: list[str] = []
    fixed_design_errors: list[str] = []
    artifact_errors: list[str] = []
    fingerprints: set[str] = set()
    schema_versions: set[tuple[int, int]] = set()
    source_file_counts: set[int] = set()
    metric_records_validated = 0
    all_run_dirs = (
        {path for path in directory.iterdir() if path.is_dir()} if directory.is_dir() else set()
    )
    configured_dirs = {path.parent for path in directory.glob("*/resolved_config.yaml")}
    unconfigured_dirs = sorted(all_run_dirs - configured_dirs)
    unconfigured_artifacts = [
        run_dir.name
        for run_dir in unconfigured_dirs
        if any((run_dir / name).exists() for name in REQUIRED_RUN_FILES)
    ]
    if unconfigured_artifacts:
        raise RuntimeError(
            f"{spec.family} has artifacts without resolved configurations: "
            f"{len(unconfigured_artifacts)} run directories"
        )
    incomplete_dirs.extend(run_dir.name for run_dir in unconfigured_dirs)
    for config_path in sorted(directory.glob("*/resolved_config.yaml")):
        run_dir = config_path.parent
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        if not isinstance(config, Mapping):
            artifact_errors.append(f"{run_dir.name}: resolved configuration is not a mapping")
            continue
        if str(get(config, "experiment.name")) != spec.name:
            unexpected_name.append(run_dir.name)
            continue
        deviations = validate_fixed_design(spec, config)
        if deviations:
            fixed_design_errors.extend(f"{run_dir.name}: {message}" for message in deviations)

        seed_value = get(config, "seed")
        if not isinstance(seed_value, int) or isinstance(seed_value, bool):
            artifact_errors.append(f"{run_dir.name}: invalid config seed {seed_value!r}")
            continue
        seed = seed_value

        missing_files = sorted(name for name in REQUIRED_RUN_FILES if not (run_dir / name).is_file())
        marker_exists = (run_dir / "COMPLETE").is_file()
        if missing_files:
            incomplete_dirs.append(run_dir.name)
            if marker_exists:
                orphan_complete.append(run_dir.name)
                artifact_errors.append(
                    f"{run_dir.name}: COMPLETE exists but required files are missing: {missing_files}"
                )
            continue

        marker = (run_dir / "COMPLETE").read_text(encoding="utf-8")
        if marker != "complete\n":
            artifact_errors.append(f"{run_dir.name}: invalid COMPLETE marker contents {marker!r}")
        summary = read_json_object(run_dir / "summary.json")
        metadata = read_json_object(run_dir / "metadata.json")
        status = read_json_object(run_dir / "status.json")
        summary_seed = summary.get("seed")
        if (
            not isinstance(summary_seed, int)
            or isinstance(summary_seed, bool)
            or seed != summary_seed
        ):
            artifact_errors.append(f"{run_dir.name}: config/summary seed mismatch")
        if str(summary.get("hypothesis")) != spec.hypothesis:
            artifact_errors.append(f"{run_dir.name}: summary hypothesis mismatch")

        run_id = run_dir.name
        if metadata.get("run_id") != run_id:
            artifact_errors.append(f"{run_id}: metadata run_id mismatch")
        metadata_seed = metadata.get("seed")
        if (
            not isinstance(metadata_seed, int)
            or isinstance(metadata_seed, bool)
            or metadata_seed != seed
        ):
            artifact_errors.append(f"{run_id}: config/metadata seed mismatch")
        if status.get("run_id") != run_id:
            artifact_errors.append(f"{run_id}: status run_id mismatch")
        if status.get("state") != "complete":
            artifact_errors.append(f"{run_id}: status state is {status.get('state')!r}, not 'complete'")

        implementation = metadata.get("implementation")
        if not isinstance(implementation, Mapping):
            artifact_errors.append(f"{run_id}: missing implementation metadata")
        else:
            artifact_schema = implementation.get("artifact_schema_version")
            source_schema = implementation.get("source_fingerprint_schema_version")
            fingerprint = implementation.get("implementation_fingerprint")
            if (
                not isinstance(artifact_schema, int)
                or isinstance(artifact_schema, bool)
                or artifact_schema != ARTIFACT_SCHEMA_VERSION
            ):
                artifact_errors.append(
                    f"{run_id}: artifact schema {artifact_schema!r}, "
                    f"expected {ARTIFACT_SCHEMA_VERSION}"
                )
            if (
                not isinstance(source_schema, int)
                or isinstance(source_schema, bool)
                or source_schema != SOURCE_FINGERPRINT_SCHEMA_VERSION
            ):
                artifact_errors.append(
                    f"{run_id}: source-fingerprint schema {source_schema!r}, "
                    f"expected {SOURCE_FINGERPRINT_SCHEMA_VERSION}"
                )
            if (
                not isinstance(fingerprint, str)
                or len(fingerprint) != 64
                or any(character not in "0123456789abcdef" for character in fingerprint)
            ):
                artifact_errors.append(f"{run_id}: invalid implementation fingerprint")
            else:
                fingerprints.add(fingerprint)
            if (
                isinstance(artifact_schema, int)
                and not isinstance(artifact_schema, bool)
                and isinstance(source_schema, int)
                and not isinstance(source_schema, bool)
            ):
                schema_versions.add((artifact_schema, source_schema))
            source_file_count = implementation.get("source_file_count")
            if (
                not isinstance(source_file_count, int)
                or isinstance(source_file_count, bool)
                or source_file_count < 1
            ):
                artifact_errors.append(
                    f"{run_id}: invalid source_file_count={source_file_count!r}"
                )
            else:
                source_file_counts.add(source_file_count)

        try:
            reconstructed = expected_artifact_run_id(config, metadata)
        except (TypeError, ValueError) as error:
            artifact_errors.append(f"{run_id}: cannot reconstruct run identity: {error}")
        else:
            if reconstructed != run_id:
                artifact_errors.append(
                    f"{run_id}: reconstructed run_id is {reconstructed}, not directory name"
                )
        try:
            metric_records_validated += validate_metrics_file(
                run_dir / "metrics.jsonl",
                run_id=run_id,
                seed=seed,
                hypothesis=spec.hypothesis,
            )
        except RuntimeError as error:
            artifact_errors.append(str(error))
        runs.append(Run(spec.family, run_dir, summary, config))

    expected = spec.expected_keys()
    observed_list = [spec.key(run.config, run.seed) for run in runs]
    duplicates = [key for key, count in Counter(observed_list).items() if count > 1]
    observed = set(observed_list)
    unexpected = sorted(observed - expected, key=str)
    missing = expected - observed
    if len(fingerprints) > 1:
        artifact_errors.append(
            f"mixed implementation fingerprints: {sorted(fingerprints)}"
        )
    if len(schema_versions) > 1:
        artifact_errors.append(f"mixed artifact schema pairs: {sorted(schema_versions)}")
    if len(source_file_counts) > 1:
        artifact_errors.append(f"mixed source-file counts: {sorted(source_file_counts)}")
    if duplicates or unexpected or unexpected_name or fixed_design_errors or artifact_errors:
        if artifact_errors:
            detail = artifact_errors[0]
        elif fixed_design_errors:
            detail = fixed_design_errors[0]
        elif unexpected_name:
            detail = f"wrong experiment name in {unexpected_name[0]}"
        elif unexpected:
            detail = f"unexpected registered key {unexpected[0]!r}"
        else:
            detail = f"duplicate registered key {duplicates[0]!r}"
        raise RuntimeError(
            f"{spec.family} grid corruption: {len(duplicates)} duplicate keys, "
            f"{len(unexpected)} unexpected keys, {len(unexpected_name)} wrong names, "
            f"{len(fixed_design_errors)} fixed-setting deviations, "
            f"{len(artifact_errors)} artifact-integrity errors; first: {detail}"
        )
    complete = (
        len(observed) == spec.expected_count
        and not missing
        and not incomplete_dirs
        and not orphan_complete
    )
    audit = {
        "family": spec.family,
        "hypothesis": spec.hypothesis,
        "experiment_name": spec.name,
        "expected_runs": spec.expected_count,
        "completed_runs": len(runs),
        "unique_completed_keys": len(observed),
        "missing_runs": len(missing),
        "incomplete_run_directories": len(incomplete_dirs),
        "materialized_run_directories": len(all_run_dirs),
        "configured_run_directories": len(configured_dirs),
        "unconfigured_run_directories": len(unconfigured_dirs),
        "complete_markers_without_summary": len(orphan_complete),
        "fixed_setting_deviations": len(fixed_design_errors),
        "artifact_integrity_errors": len(artifact_errors),
        "metric_records_validated": metric_records_validated,
        "implementation_fingerprints": sorted(fingerprints),
        "artifact_schema_versions": sorted({pair[0] for pair in schema_versions}),
        "source_fingerprint_schema_versions": sorted({pair[1] for pair in schema_versions}),
        "source_file_counts": sorted(source_file_counts),
        "run_identities_validated": len(runs),
        "complete_status_records": len(runs),
        "seeds_observed": sorted({run.seed for run in runs}),
        "seeds_expected": list(SEEDS),
        "completed_by_seed": {str(seed): sum(run.seed == seed for run in runs) for seed in SEEDS},
        "expected_per_seed": spec.expected_count // len(SEEDS),
        "complete": complete,
        "provisional": not complete,
    }
    if not complete and not allow_incomplete:
        raise RuntimeError(
            f"{spec.family} is incomplete: {len(runs)}/{spec.expected_count} completed "
            f"({len(missing)} registered cells missing)"
        )
    return runs, audit


def bootstrap_interval(
    values: Sequence[float], *, label: str, draws: int = BOOTSTRAP_DRAWS
) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {"estimate": None, "ci_low": None, "ci_high": None, "n_seeds": 0}
    rng = np.random.default_rng(stable_seed("bootstrap", label))
    sampled = rng.choice(array, size=(draws, len(array)), replace=True).mean(axis=1)
    low, high = np.quantile(sampled, (0.025, 0.975))
    return {
        "estimate": float(array.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "n_seeds": len(array),
    }


def estimate_by_seed(
    rows: Sequence[Mapping[str, Any]],
    value: str | Callable[[Mapping[str, Any]], float | None],
    *,
    label: str,
) -> dict[str, Any]:
    by_seed: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        raw = value(row) if callable(value) else row.get(value)
        if raw is None or not math.isfinite(float(raw)):
            continue
        by_seed[int(row["seed"])].append(float(raw))
    collapsed = [mean(by_seed[seed]) for seed in sorted(by_seed) if by_seed[seed]]
    return {**bootstrap_interval(collapsed, label=label), "n_cells": len(rows)}


def seed_bootstrap_ratio(
    rows: Sequence[Mapping[str, Any]],
    *,
    numerator: Callable[[Mapping[str, Any]], bool],
    denominator: Callable[[Mapping[str, Any]], bool],
    label: str,
) -> dict[str, Any]:
    """Estimate a pooled ratio while resampling independent training seeds."""

    by_seed: dict[int, tuple[int, int]] = {}
    for seed in SEEDS:
        part = [row for row in rows if int(row["seed"]) == seed]
        by_seed[seed] = (
            sum(int(numerator(row)) for row in part),
            sum(int(denominator(row)) for row in part),
        )
    counts = [by_seed[seed] for seed in SEEDS]
    numerator_total = sum(item[0] for item in counts)
    denominator_total = sum(item[1] for item in counts)
    if denominator_total == 0:
        return {
            "estimate": None,
            "ci_low": None,
            "ci_high": None,
            "n_seeds": len(SEEDS),
            "numerator_count": numerator_total,
            "denominator_count": denominator_total,
        }
    rng = np.random.default_rng(stable_seed("ratio-bootstrap", label))
    draws: list[float] = []
    for _ in range(BOOTSTRAP_DRAWS):
        sampled = rng.integers(0, len(counts), size=len(counts))
        sampled_num = sum(counts[int(index)][0] for index in sampled)
        sampled_den = sum(counts[int(index)][1] for index in sampled)
        if sampled_den > 0:
            draws.append(sampled_num / sampled_den)
    low, high = np.quantile(np.asarray(draws), (0.025, 0.975))
    return {
        "estimate": numerator_total / denominator_total,
        "ci_low": float(low),
        "ci_high": float(high),
        "n_seeds": len(SEEDS),
        "numerator_count": numerator_total,
        "denominator_count": denominator_total,
    }


def paired_contrast(
    rows: Sequence[Mapping[str, Any]],
    *,
    value: str,
    factor: str,
    high: Any,
    low: Any,
    match: Sequence[str],
    label: str,
) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], dict[Any, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        raw = row.get(value)
        if raw is None or not math.isfinite(float(raw)):
            continue
        key = tuple(row[name] for name in match)
        groups[key][row[factor]].append(float(raw))
    by_seed: dict[int, list[float]] = defaultdict(list)
    matched = 0
    for key, levels in groups.items():
        if high not in levels or low not in levels:
            continue
        delta = mean(levels[high]) - mean(levels[low])
        seed = int(key[match.index("seed")])
        by_seed[seed].append(delta)
        matched += 1
    collapsed = [mean(by_seed[seed]) for seed in sorted(by_seed) if by_seed[seed]]
    return {
        "factor": factor,
        "high": high,
        "low": low,
        "outcome": value,
        **bootstrap_interval(collapsed, label=label),
        "n_matched_cells": matched,
    }


def ranks(values: Sequence[float]) -> np.ndarray:
    raw = np.asarray(values, dtype=float)
    order = np.argsort(raw, kind="mergesort")
    result: np.ndarray = np.empty(len(raw), dtype=float)
    start = 0
    while start < len(raw):
        end = start + 1
        while end < len(raw) and raw[order[end]] == raw[order[start]]:
            end += 1
        result[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return result


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 2:
        return float("nan")
    rx, ry = ranks(x), ranks(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for name in row:
            if name not in fields:
                fields.append(name)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fields:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def last_metric(run: Run, metric: str, *, split: str = "train") -> float | None:
    path = run.path / "metrics.jsonl"
    if not path.is_file():
        return None
    best_step = -1
    selected: float | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("metric") != metric or record.get("split") != split:
                continue
            step = int(record.get("global_step", -1))
            if step >= best_step:
                selected = float(record["value"])
                best_step = step
    return selected


def trajectory_metrics(run: Run) -> dict[str, float | int | None]:
    """Return planned acquisition/AUC summaries from H5 log-step evaluations."""

    path = run.path / "metrics.jsonl"
    points: dict[int, dict[str, float]] = defaultdict(dict)
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if (
                    record.get("split") == "conflict_eval"
                    and record.get("stage") == "train"
                    and record.get("metric") in {"rho_y", "intended_probability"}
                ):
                    points[int(record["global_step"])][str(record["metric"])] = float(record["value"])
    usable = sorted(
        (step, values)
        for step, values in points.items()
        if "rho_y" in values and "intended_probability" in values
    )
    acquired = _persistent_first_step(
        [(step, values["rho_y"]) for step, values in usable], 0.9
    )
    auc: float | None = None
    if len(usable) >= 2:
        xs = np.asarray([item[0] for item in usable], dtype=float)
        ys = np.asarray([item[1]["intended_probability"] for item in usable], dtype=float)
        auc = trapezoidal_integral(ys, xs) / float(xs[-1] - xs[0])
    return {
        "acquisition_step": acquired,
        "acquisition_observed": int(acquired is not None),
        "intended_probability_auc": auc,
        "trajectory_points": len(usable),
    }


def _persistent_first_step(points: Sequence[tuple[int, float]], threshold: float) -> int | None:
    """Return the first checkpoint in the first qualifying adjacent pair."""

    previous: tuple[int, float] | None = None
    for point in sorted(points):
        if previous is not None and previous[1] >= threshold and point[1] >= threshold:
            return previous[0]
        previous = point
    return None


def e11_acquisition_metrics(run: Run) -> dict[str, int | None]:
    """Read the two persistent acquisition events used in the E11 narrative."""

    exact: list[tuple[int, float]] = []
    competition: list[tuple[int, float]] = []
    path = run.path / "metrics.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if (
                record.get("stage") == "exact_calibration"
                and record.get("split") == "exact_calibration_iid"
                and record.get("metric") == "target_accuracy"
            ):
                exact.append((int(record["global_step"]), float(record["value"])))
            elif (
                record.get("stage") == "competition"
                and record.get("split") == "competition_conflict"
                and record.get("metric") == "rho_y"
            ):
                competition.append((int(record["global_step"]), float(record["value"])))
    exact_step = _persistent_first_step(exact, 0.95)
    competition_step = _persistent_first_step(competition, 0.90)
    return {
        "exact_acquisition_step": exact_step,
        "exact_acquisition_observed": int(exact_step is not None),
        "exact_restricted_acquisition_step": 4_096 if exact_step is None else exact_step,
        "competition_acquisition_step": competition_step,
        "competition_acquisition_observed": int(competition_step is not None),
        "competition_restricted_acquisition_step": (4_096 if competition_step is None else competition_step),
    }


def rows_e10(runs: Sequence[Run]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        trajectory = trajectory_metrics(run)
        rows.append(
            {
                "run_id": run.path.name,
                "seed": run.seed,
                "arm": e10_arm(run.config),
                "q": f(get(run.config, "data.q")),
                "k": int(get(run.config, "data.k")),
                "beta": f(get(run.config, "h5.entropy_coefficient")),
                "rho_y": float(get(run.summary, "final.rho_y")),
                "intended_probability": float(get(run.summary, "final.intended_probability")),
                "iid_accuracy": float(get(run.summary, "iid.target_accuracy")),
                "iid_intended_probability": float(get(run.summary, "iid.intended_probability")),
                "final_policy_entropy": last_metric(run, "policy_entropy"),
                "proxy_flip_rate": float(get(run.summary, "interventions.flip_P.hard_flip_rate")),
                "exact_flip_rate": float(get(run.summary, "interventions.flip_R_mean.hard_flip_rate")),
                "proxy_probability_ate": float(get(run.summary, "interventions.flip_P.probability_ate")),
                "exact_probability_ate": float(get(run.summary, "interventions.flip_R_mean.probability_ate")),
                "reliably_intended_seed": int(float(get(run.summary, "final.rho_y")) > 0.9),
                **trajectory,
            }
        )
    return rows


def analyse_e10(rows: Sequence[Mapping[str, Any]], *, complete: bool) -> dict[str, Any]:
    cell_rows: list[dict[str, Any]] = []
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["arm"], row["q"], row["k"], row["beta"])].append(row)
    for key, part in sorted(groups.items(), key=lambda item: str(item[0])):
        arm, q, k, beta = key
        entry: dict[str, Any] = {"arm": arm, "q": q, "k": k, "beta": beta}
        for outcome in (
            "rho_y",
            "intended_probability",
            "iid_accuracy",
            "final_policy_entropy",
            "intended_probability_auc",
            "acquisition_observed",
            "proxy_flip_rate",
            "exact_flip_rate",
        ):
            estimate = estimate_by_seed(part, outcome, label=f"e10-cell-{key}-{outcome}")
            entry.update({f"{outcome}_{name}": value for name, value in estimate.items()})
        intended = sum(float(row["rho_y"]) > 0.9 for row in part)
        entry["reliably_intended_seeds"] = intended
        entry["reliably_intended_cell"] = int(len(part) == 10 and intended >= 8)
        cell_rows.append(entry)

    primary = [row for row in rows if row["arm"] == "actor_critic_full"]
    contrast_rows: list[dict[str, Any]] = []
    for q, k, beta, outcome in product(
        (0.50, 0.75, 0.90, 0.99),
        (2, 3, 4),
        (0.10, 0.30),
        ("rho_y", "intended_probability", "iid_accuracy", "intended_probability_auc"),
    ):
        selected = [row for row in primary if row["q"] == q and row["k"] == k]
        estimate = paired_contrast(
            selected,
            value=outcome,
            factor="beta",
            high=f(beta),
            low=f(0.0),
            match=("q", "k", "seed"),
            label=f"e10-{q}-{k}-{beta}-{outcome}",
        )
        contrast_rows.append({"scope": "within_q_k", "q": q, "k": k, **estimate})

    for beta, outcome in product(
        (0.10, 0.30),
        ("rho_y", "intended_probability", "iid_accuracy", "intended_probability_auc"),
    ):
        contrast_rows.append(
            {
                "scope": "collapsed_primary_grid",
                "q": "all",
                "k": "all",
                **paired_contrast(
                    primary,
                    value=outcome,
                    factor="beta",
                    high=f(beta),
                    low=f(0.0),
                    match=("q", "k", "seed"),
                    label=f"e10-collapsed-{beta}-{outcome}",
                ),
            }
        )

    # Fixed estimator/capacity comparisons at the canonical cell.  First
    # compare arms at each entropy level, then compare their entropy response
    # against the full actor--critic response (a paired difference-in-differences).
    canonical = [row for row in rows if row["q"] == 0.9 and row["k"] == 3]
    for arm, beta, outcome in product(
        ("reinforce_full", "actor_critic_subspace1024"),
        (0.0, 0.003, 0.01, 0.03, 0.10, 0.30),
        ("rho_y", "iid_accuracy"),
    ):
        contrast_rows.append(
            {
                "scope": "canonical_arm_minus_actor_critic",
                "q": 0.9,
                "k": 3,
                **paired_contrast(
                    [row for row in canonical if row["beta"] == f(beta)],
                    value=outcome,
                    factor="arm",
                    high=arm,
                    low="actor_critic_full",
                    match=("beta", "seed"),
                    label=f"e10-arm-{arm}-{beta}-{outcome}",
                ),
            }
        )

    did_rows: list[dict[str, Any]] = []
    index = {(str(row["arm"]), f(row["beta"]), int(row["seed"])): row for row in canonical}
    for arm, beta, outcome in product(
        ("reinforce_full", "actor_critic_subspace1024"),
        (0.003, 0.01, 0.03, 0.10, 0.30),
        ("rho_y", "iid_accuracy"),
    ):
        values: list[float] = []
        for seed in SEEDS:
            keys = [
                (arm, f(beta), seed),
                (arm, f(0), seed),
                ("actor_critic_full", f(beta), seed),
                ("actor_critic_full", f(0), seed),
            ]
            if not all(key in index for key in keys):
                continue
            values.append(
                (float(index[keys[0]][outcome]) - float(index[keys[1]][outcome]))
                - (float(index[keys[2]][outcome]) - float(index[keys[3]][outcome]))
            )
        did_rows.append(
            {
                "arm": arm,
                "beta": beta,
                "outcome": outcome,
                **bootstrap_interval(values, label=f"e10-did-{arm}-{beta}-{outcome}"),
            }
        )

    # The registered mechanistic follow-up trigger is evaluated only at the
    # canonical full actor--critic cell.  Incomplete data can display the
    # provisional estimates but can never fire the trigger.
    trigger_candidates: list[dict[str, Any]] = []
    trigger_rows = [row for row in primary if row["q"] == 0.9 and row["k"] == 3]
    for beta in (0.003, 0.01, 0.03, 0.10, 0.30):
        result = paired_contrast(
            trigger_rows,
            value="rho_y",
            factor="beta",
            high=f(beta),
            low=f(0),
            match=("q", "k", "seed"),
            label=f"e10-trigger-{beta}",
        )
        candidate_complete = result["n_seeds"] == len(SEEDS)
        passes = bool(
            candidate_complete
            and result["estimate"] is not None
            and float(result["estimate"]) >= 0.20
            and float(result["ci_low"]) > 0.0
        )
        trigger_candidates.append(
            {"beta": beta, **result, "complete_pairs": candidate_complete, "passes": passes}
        )
    fired = bool(complete and any(item["passes"] for item in trigger_candidates))
    trigger = {
        "registered_scope": "q=.90, k=3, actor_critic_full",
        "criterion": "paired mean rho_y improvement >= .20 and 95% seed-bootstrap CI low > 0",
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "grid_complete": complete,
        "status": "fired" if fired else ("not_fired" if complete else "pending"),
        "fired": fired,
        "candidates": trigger_candidates,
    }
    falsifier_rows = [row for row in primary if row["q"] == 0.5]
    falsifier = {
        "scope": "q=.50 actor_critic_full",
        "interpretation": (
            "parity is not identifiable through a useful proxy; low intended reliance here "
            "is a general RL decoder failure rather than evidence of proxy lock-in"
        ),
        "rho_y": estimate_by_seed(falsifier_rows, "rho_y", label="e10-q50-rhoy"),
        "reliably_intended_cells": sum(
            int(row["reliably_intended_cell"])
            for row in cell_rows
            if row["arm"] == "actor_critic_full" and row["q"] == 0.5
        ),
        "registered_cells": 18,
    }
    return {
        "cell_estimates": cell_rows,
        "paired_contrasts": contrast_rows,
        "arm_interactions": did_rows,
        "trigger": trigger,
        "q50_falsifier": falsifier,
    }


def rows_e11(runs: Sequence[Run]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        acquisition = e11_acquisition_metrics(run)
        rows.append(
            {
                "run_id": run.path.name,
                "seed": run.seed,
                "rule": str(get(run.config, "data.target_rule")),
                "q": f(get(run.config, "data.q")),
                "width": int(get(run.config, "model.width")),
                "depth": int(get(run.config, "model.depth")),
                "parameters": int(get(run.summary, "model.competition.total_parameters")),
                "proxy_decoder_accuracy": float(get(run.summary, "final.proxy_decoder_accuracy")),
                "exact_decoder_accuracy": float(get(run.summary, "final.exact_decoder_accuracy")),
                "decoder_advantage": float(get(run.summary, "final.exact_decoder_accuracy"))
                - float(get(run.summary, "final.proxy_decoder_accuracy")),
                "rho_y": float(get(run.summary, "final.competition_conflict.rho_y")),
                "rho_p": float(get(run.summary, "final.competition_conflict.rho_p")),
                "iid_accuracy": float(get(run.summary, "final.competition_iid.target_accuracy")),
                "single_channel_bayes": float(
                    get(run.summary, "data.best_single_rule_channel_bayes_accuracy")
                ),
                "proxy_flip_rate": float(
                    get(run.summary, "final.competition_interventions.flip_P.hard_flip_rate")
                ),
                "rule_flip_rate": float(
                    get(
                        run.summary,
                        "final.competition_interventions.flip_exact_rule_output.hard_flip_rate",
                    )
                ),
                "proxy_probability_ate": float(
                    get(run.summary, "final.competition_interventions.flip_P.probability_ate")
                ),
                "rule_probability_ate": float(
                    get(
                        run.summary,
                        "final.competition_interventions.flip_exact_rule_output.probability_ate",
                    )
                ),
                **acquisition,
            }
        )
    return rows


def analyse_e11(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    cell_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["rule"], row["q"], row["width"], row["depth"])].append(row)
    for key, part in sorted(grouped.items(), key=lambda item: str(item[0])):
        entry: dict[str, Any] = dict(zip(("rule", "q", "width", "depth"), key, strict=True))
        for outcome in (
            "exact_decoder_accuracy",
            "proxy_decoder_accuracy",
            "decoder_advantage",
            "rho_y",
            "iid_accuracy",
            "single_channel_bayes",
            "proxy_flip_rate",
            "rule_flip_rate",
        ):
            estimate = estimate_by_seed(part, outcome, label=f"e11-cell-{key}-{outcome}")
            entry.update({f"{outcome}_{name}": value for name, value in estimate.items()})
        cell_rows.append(entry)

    contrasts: list[dict[str, Any]] = []
    for rule, outcome in product(
        ("majority", "conjunction", "multiplexer"),
        ("exact_decoder_accuracy", "rho_y", "single_channel_bayes", "rule_flip_rate"),
    ):
        selected = [row for row in rows if row["rule"] in {"parity", rule}]
        contrasts.append(
            {
                "comparison": f"{rule}-parity",
                **paired_contrast(
                    selected,
                    value=outcome,
                    factor="rule",
                    high=rule,
                    low="parity",
                    match=("q", "width", "depth", "seed"),
                    label=f"e11-{rule}-parity-{outcome}",
                ),
            }
        )

    # Correlations are calculated separately inside every training seed, then
    # the ten correlations are bootstrapped.  This avoids treating 120 design
    # cells per seed as independent replications.
    association_rows: list[dict[str, Any]] = []
    for x_name in ("exact_decoder_accuracy", "decoder_advantage", "single_channel_bayes"):
        values: list[float] = []
        for seed in SEEDS:
            part = [row for row in rows if int(row["seed"]) == seed]
            correlation = spearman(
                [float(row[x_name]) for row in part],
                [float(row["rho_y"]) for row in part],
            )
            if math.isfinite(correlation):
                values.append(correlation)
        association_rows.append(
            {
                "x": x_name,
                "y": "rho_y",
                "method": "within-seed Spearman; bootstrap across seeds",
                **bootstrap_interval(values, label=f"e11-association-{x_name}"),
            }
        )

    q99_values: list[float] = []
    for seed in SEEDS:
        part = [row for row in rows if int(row["seed"]) == seed and float(row["q"]) == 0.99]
        correlation = spearman(
            [float(row["single_channel_bayes"]) for row in part],
            [float(row["rho_y"]) for row in part],
        )
        if math.isfinite(correlation):
            q99_values.append(correlation)
    association_rows.append(
        {
            "x": "single_channel_bayes",
            "y": "rho_y",
            "scope": "q=.99 competition grid",
            "method": "within-seed Spearman; bootstrap across seeds",
            **bootstrap_interval(q99_values, label="e11-association-q99-single-channel"),
        }
    )

    # Exact-only calibration is repeated unchanged at the three q levels.
    # Collapse those deterministic duplicates before reporting the 100 unique
    # rule/architecture/seed controls for each rule.
    exact_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        exact_groups[(row["rule"], row["width"], row["depth"], row["seed"])].append(row)
    unique_exact: list[Mapping[str, Any]] = []
    for key, part in exact_groups.items():
        if len(part) != 3:
            raise RuntimeError(f"E11 exact-control duplicate group {key} has {len(part)} rows")
        signatures = {
            (
                row["exact_acquisition_step"],
                row["exact_acquisition_observed"],
                row["exact_restricted_acquisition_step"],
                row["exact_decoder_accuracy"],
            )
            for row in part
        }
        if len(signatures) != 1:
            raise RuntimeError(f"E11 exact-control q duplicates diverge for {key}")
        unique_exact.append(part[0])
    exact_acquisition: list[dict[str, Any]] = []
    for rule in ("majority", "conjunction", "multiplexer", "parity"):
        part = [row for row in unique_exact if row["rule"] == rule]
        if len(part) != 100:
            raise RuntimeError(f"E11 {rule} has {len(part)} unique exact controls")
        exact_acquisition.append(
            {
                "rule": rule,
                "threshold": 0.95,
                "persistence_evaluations": 2,
                "event_time": "first checkpoint in qualifying adjacent pair",
                "censor_horizon": 4_096,
                "unique_controls": len(part),
                "events": sum(int(row["exact_acquisition_observed"]) for row in part),
                "censored": sum(1 - int(row["exact_acquisition_observed"]) for row in part),
                **estimate_by_seed(
                    part,
                    "exact_restricted_acquisition_step",
                    label=f"e11-exact-rmst-{rule}",
                ),
            }
        )

    q99_competition: list[dict[str, Any]] = []
    for rule in ("majority", "conjunction", "multiplexer", "parity"):
        part = [row for row in rows if row["rule"] == rule and float(row["q"]) == 0.99]
        if len(part) != 100:
            raise RuntimeError(f"E11 q=.99 {rule} has {len(part)} competition runs")
        q99_competition.append(
            {
                "rule": rule,
                "q": 0.99,
                "threshold": 0.90,
                "persistence_evaluations": 2,
                "event_time": "first checkpoint in qualifying adjacent pair",
                "censor_horizon": 4_096,
                "runs": len(part),
                "events": sum(int(row["competition_acquisition_observed"]) for row in part),
                "censored": sum(1 - int(row["competition_acquisition_observed"]) for row in part),
                **{
                    f"event_fraction_{name}": value
                    for name, value in estimate_by_seed(
                        part,
                        "competition_acquisition_observed",
                        label=f"e11-q99-competition-events-{rule}",
                    ).items()
                },
                **{
                    f"restricted_time_{name}": value
                    for name, value in estimate_by_seed(
                        part,
                        "competition_restricted_acquisition_step",
                        label=f"e11-q99-competition-rmst-{rule}",
                    ).items()
                },
            }
        )

    accessible = [row for row in rows if float(row["exact_decoder_accuracy"]) >= 0.95]
    decoder_control_gap = [row for row in accessible if float(row["rho_y"]) < 0.5]
    threshold = {
        "decoder_accessible_runs": len(accessible),
        "accessible_but_proxy_controlled_runs": len(decoder_control_gap),
        "fraction_accessible_but_proxy_controlled": (
            len(decoder_control_gap) / len(accessible) if accessible else None
        ),
    }
    return {
        "cell_estimates": cell_rows,
        "matched_rule_contrasts": contrasts,
        "associations": association_rows,
        "exact_only_acquisition": exact_acquisition,
        "q99_competition_acquisition": q99_competition,
        "decoder_control_threshold": threshold,
    }


def _diagnostic_winner(values: Mapping[str, float]) -> tuple[str, float, float]:
    observed = np.asarray([values["both_wrong"], values["p_wrong"], values["q_wrong"]])
    signatures = {
        "P": np.asarray([0.0, 0.0, 1.0]),
        "Q": np.asarray([0.0, 1.0, 0.0]),
        "Y": np.asarray([1.0, 1.0, 1.0]),
    }
    distances = sorted(
        (
            (float(np.sqrt(np.mean((observed - signature) ** 2))), name)
            for name, signature in signatures.items()
        )
    )
    return distances[0][1], distances[0][0], distances[1][0] - distances[0][0]


def rows_e12(runs: Sequence[Run]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        panels = ("both_wrong", "p_wrong", "q_wrong")
        target = {
            panel: float(get(run.summary, f"final.competition.{panel}.target_accuracy")) for panel in panels
        }
        # Under a context-independent stochastic mixture of the three pure
        # rules, the three diagnostic accuracies identify the mixture weights
        # exactly.  The inversion can yield negative weights or fail to close
        # to one when control is input-dependent or otherwise non-mixture-like.
        mixture_y = target["both_wrong"]
        mixture_q = target["p_wrong"] - target["both_wrong"]
        mixture_p = target["q_wrong"] - target["both_wrong"]
        mixture_sum = mixture_p + mixture_q + mixture_y
        closure_error = mixture_sum - 1.0
        mixture_tolerance = float(get(run.config, "evaluation.equivalence_margin", 0.05))
        mixture_compatible = bool(
            abs(closure_error) <= mixture_tolerance
            and all(
                -mixture_tolerance <= weight <= 1.0 + mixture_tolerance
                for weight in (mixture_p, mixture_q, mixture_y)
            )
        )
        winner, distance, margin = _diagnostic_winner(target)
        influences: dict[str, float] = {}
        for candidate, name in (("P", "flip_P"), ("Q", "Q_mean"), ("Y", "Y_mean")):
            influences[candidate] = mean(
                float(get(run.summary, f"final.interventions.{panel}.{name}.hard_flip_rate"))
                for panel in panels
            )
        total = sum(influences.values())
        shares = {name: (value / total if total > 0 else 1.0 / 3.0) for name, value in influences.items()}
        causal_winner = max(influences, key=lambda name: influences[name])
        max_signature_error = {
            "P": max(abs(target["both_wrong"]), abs(target["p_wrong"]), abs(target["q_wrong"] - 1)),
            "Q": max(abs(target["both_wrong"]), abs(target["p_wrong"] - 1), abs(target["q_wrong"])),
            "Y": max(abs(target["both_wrong"] - 1), abs(target["p_wrong"] - 1), abs(target["q_wrong"] - 1)),
        }[winner]
        pure = bool(causal_winner == winner and max_signature_error <= 0.10 and influences[winner] >= 0.50)
        rows.append(
            {
                "run_id": run.path.name,
                "seed": run.seed,
                "q_p": f(get(run.config, "h10.q_p")),
                "q_q": f(get(run.config, "h10.q_q")),
                "k_q": int(get(run.config, "h10.k_q")),
                "k_y": int(get(run.config, "h10.k_y")),
                "width": int(get(run.config, "model.width")),
                "overlap": str(get(run.config, "h10.error_structure")),
                "error_phi": get(run.summary, "data.training_overlap.error_phi"),
                "calibration_p": float(get(run.summary, "final.calibration.P.iid.decoder_accuracy")),
                "calibration_q": float(get(run.summary, "final.calibration.Q.iid.decoder_accuracy")),
                "calibration_y": float(get(run.summary, "final.calibration.Y.iid.decoder_accuracy")),
                "iid_accuracy": float(get(run.summary, "final.competition.iid.target_accuracy")),
                "both_wrong_accuracy": target["both_wrong"],
                "p_wrong_accuracy": target["p_wrong"],
                "q_wrong_accuracy": target["q_wrong"],
                "mixture_weight_p": mixture_p,
                "mixture_weight_q": mixture_q,
                "mixture_weight_y": mixture_y,
                "mixture_weight_sum": mixture_sum,
                "mixture_closure_error": closure_error,
                "mixture_closure_abs": abs(closure_error),
                "mixture_tolerance": mixture_tolerance,
                "static_mixture_compatible": int(mixture_compatible),
                "influence_p": influences["P"],
                "influence_q": influences["Q"],
                "influence_y": influences["Y"],
                "share_p": shares["P"],
                "share_q": shares["Q"],
                "share_y": shares["Y"],
                "diagnostic_winner": winner,
                "diagnostic_distance": distance,
                "diagnostic_margin": margin,
                "causal_winner": causal_winner,
                "diagnostic_causal_agree": int(winner == causal_winner),
                "pure_goal_selection": int(pure),
                "selected_goal": winner if pure else "mixture_or_unresolved",
            }
        )
    return rows


def analyse_e12(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    overall: list[dict[str, Any]] = []
    for outcome in (
        "calibration_p",
        "calibration_q",
        "calibration_y",
        "iid_accuracy",
        "both_wrong_accuracy",
        "p_wrong_accuracy",
        "q_wrong_accuracy",
        "share_p",
        "share_q",
        "share_y",
        "diagnostic_causal_agree",
        "pure_goal_selection",
        "mixture_weight_p",
        "mixture_weight_q",
        "mixture_weight_y",
        "mixture_closure_error",
        "mixture_closure_abs",
        "static_mixture_compatible",
    ):
        overall.append(
            {
                "outcome": outcome,
                **estimate_by_seed(rows, outcome, label=f"e12-overall-{outcome}"),
            }
        )

    contrasts: list[dict[str, Any]] = []
    definitions = (
        ("overlap", "nested", "independent", ("q_p", "q_q", "k_q", "k_y", "width", "seed")),
        ("q_p", 0.99, 0.90, ("q_q", "k_q", "k_y", "width", "overlap", "seed")),
        ("q_q", 0.99, 0.90, ("q_p", "k_q", "k_y", "width", "overlap", "seed")),
        ("k_q", 3, 2, ("q_p", "q_q", "k_y", "width", "overlap", "seed")),
        ("k_y", 5, 3, ("q_p", "q_q", "k_q", "width", "overlap", "seed")),
        ("width", 64, 16, ("q_p", "q_q", "k_q", "k_y", "overlap", "seed")),
    )
    for factor, high, low, match in definitions:
        for outcome in (
            "both_wrong_accuracy",
            "p_wrong_accuracy",
            "q_wrong_accuracy",
            "share_p",
            "share_q",
            "share_y",
            "iid_accuracy",
        ):
            contrasts.append(
                paired_contrast(
                    rows,
                    value=outcome,
                    factor=factor,
                    high=high,
                    low=low,
                    match=match,
                    label=f"e12-{factor}-{high}-{low}-{outcome}",
                )
            )

    classifications = Counter(str(row["selected_goal"]) for row in rows)
    nearest = Counter(str(row["diagnostic_winner"]) for row in rows)
    causal = Counter(str(row["causal_winner"]) for row in rows)
    classification = {
        "registered_runs": len(rows),
        "strict_selected_goal_counts": dict(sorted(classifications.items())),
        "nearest_diagnostic_counts": dict(sorted(nearest.items())),
        "largest_causal_influence_counts": dict(sorted(causal.items())),
        "strict_rule_definition": (
            "diagnostic signature max error <= .10, matching largest causal influence, "
            "and winner mean hard-flip rate >= .50"
        ),
    }
    mixture_groups: list[dict[str, Any]] = []
    for overlap, k_y in product(("independent", "nested"), (3, 5)):
        part = [row for row in rows if row["overlap"] == overlap and int(row["k_y"]) == k_y]
        entry: dict[str, Any] = {"overlap": overlap, "k_y": k_y}
        for outcome in (
            "mixture_weight_p",
            "mixture_weight_q",
            "mixture_weight_y",
            "mixture_closure_error",
            "mixture_closure_abs",
            "static_mixture_compatible",
        ):
            estimate = estimate_by_seed(part, outcome, label=f"e12-mixture-{overlap}-{k_y}-{outcome}")
            entry.update({f"{outcome}_{name}": value for name, value in estimate.items()})
        mixture_groups.append(entry)

    mixture_contrasts: list[dict[str, Any]] = []
    for k_y, outcome in product(
        (3, 5),
        ("mixture_closure_error", "mixture_closure_abs", "static_mixture_compatible"),
    ):
        part = [row for row in rows if int(row["k_y"]) == k_y]
        mixture_contrasts.append(
            {
                "comparison": "nested-independent",
                "k_y": k_y,
                **paired_contrast(
                    part,
                    value=outcome,
                    factor="overlap",
                    high="nested",
                    low="independent",
                    match=("q_p", "q_q", "k_q", "k_y", "width", "seed"),
                    label=f"e12-mixture-overlap-{k_y}-{outcome}",
                ),
            }
        )
    for overlap, outcome in product(
        ("independent", "nested"),
        ("mixture_closure_error", "mixture_closure_abs", "static_mixture_compatible"),
    ):
        part = [row for row in rows if row["overlap"] == overlap]
        mixture_contrasts.append(
            {
                "comparison": "kY5-kY3",
                "k_y": "5-3",
                "overlap": overlap,
                **paired_contrast(
                    part,
                    value=outcome,
                    factor="k_y",
                    high=5,
                    low=3,
                    match=("q_p", "q_q", "k_q", "width", "overlap", "seed"),
                    label=f"e12-mixture-ky-{overlap}-{outcome}",
                ),
            }
        )

    purity_strata: list[dict[str, Any]] = []
    for pure, compatible in product((0, 1), (0, 1)):

        def stratum_indicator(
            row: Mapping[str, Any],
            pure_level: int = pure,
            compatible_level: int = compatible,
        ) -> float:
            return float(
                int(row["pure_goal_selection"]) == pure_level
                and int(row["static_mixture_compatible"]) == compatible_level
            )

        estimate = estimate_by_seed(
            rows,
            stratum_indicator,
            label=f"e12-mixture-purity-{pure}-compatibility-{compatible}",
        )
        purity_strata.append(
            {
                "summary_type": "full_grid_crosstab",
                "pure_goal_selection": pure,
                "static_mixture_compatible": compatible,
                "count": sum(
                    int(row["pure_goal_selection"]) == pure
                    and int(row["static_mixture_compatible"]) == compatible
                    for row in rows
                ),
                **estimate,
            }
        )
    conditional_nonpure = seed_bootstrap_ratio(
        rows,
        numerator=lambda row: (
            int(row["pure_goal_selection"]) == 0 and int(row["static_mixture_compatible"]) == 1
        ),
        denominator=lambda row: int(row["pure_goal_selection"]) == 0,
        label="e12-mixture-compatible-given-nonpure",
    )
    conditional_pure = seed_bootstrap_ratio(
        rows,
        numerator=lambda row: (
            int(row["pure_goal_selection"]) == 1 and int(row["static_mixture_compatible"]) == 1
        ),
        denominator=lambda row: int(row["pure_goal_selection"]) == 1,
        label="e12-mixture-compatible-given-pure",
    )
    for stratum, result in (
        ("non_pure", conditional_nonpure),
        ("pure", conditional_pure),
    ):
        purity_strata.append(
            {
                "summary_type": "conditional_compatibility",
                "stratum": stratum,
                **result,
            }
        )
    nonpure_compatible_all = next(
        row
        for row in purity_strata
        if row.get("summary_type") == "full_grid_crosstab"
        and row.get("pure_goal_selection") == 0
        and row.get("static_mixture_compatible") == 1
    )
    mixture = {
        "inversion": {
            "w_y": "A_both_wrong",
            "w_q": "A_p_wrong - A_both_wrong",
            "w_p": "A_q_wrong - A_both_wrong",
            "closure_error": "w_p + w_q + w_y - 1",
            "compatibility": (
                "absolute closure error <= evaluation.equivalence_margin and every "
                "weight lies in [-margin, 1+margin]"
            ),
            "equivalence_margin": 0.05,
        },
        "by_overlap_and_k_y": mixture_groups,
        "paired_contrasts": mixture_contrasts,
        "purity_stratification": {
            "table": purity_strata,
            "nonpure_and_compatible_fraction_of_all": {
                name: nonpure_compatible_all[name]
                for name in ("estimate", "ci_low", "ci_high", "n_seeds", "count")
            },
            "compatibility_conditional_on_nonpure": conditional_nonpure,
            "compatibility_conditional_on_pure": conditional_pure,
            "interpretation": (
                "Aggregate static-mixture compatibility is stratified by the existing "
                "strict pure-goal classification; compatibility among non-pure policies "
                "is the relevant diagnostic for a genuinely mixed behavioral account."
            ),
        },
    }
    return {
        "overall": overall,
        "factorial_contrasts": contrasts,
        "classification": classification,
        "diagnostic_mixture": mixture,
    }


def rows_e13(runs: Sequence[Run]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        depth = int(get(run.config, "h11.route_depth"))
        singles = [get(run.summary, f"final.single_conflict_{stage}") for stage in range(depth)]
        all_conflict = get(run.summary, "final.all_conflict")
        conflicted_stage_accuracy = mean(
            float(singles[stage]["stage_branch_accuracy"][stage]) for stage in range(depth)
        )
        nonconflicted_values = [
            float(singles[conflict_stage]["stage_branch_accuracy"][stage])
            for conflict_stage in range(depth)
            for stage in range(depth)
            if stage != conflict_stage
        ]
        reversed_accuracy = float(get(run.summary, "final.all_conflict_address_reversed.branch_accuracy"))
        removed_accuracy = float(get(run.summary, "final.all_conflict_address_removed.branch_accuracy"))
        base_accuracy = float(all_conflict["branch_accuracy"])
        rows.append(
            {
                "run_id": run.path.name,
                "seed": run.seed,
                "route_depth": depth,
                "q": f(get(run.config, "data.q")),
                "k": int(get(run.config, "data.k")),
                "width": int(get(run.config, "model.width")),
                "regime": str(get(run.config, "h11.evidence_regime")),
                "steps": int(get(run.summary, "training.steps")),
                "iid_branch_accuracy": float(get(run.summary, "final.iid.branch_accuracy")),
                "iid_full_route_success": float(get(run.summary, "final.iid.full_route_success")),
                "branch_accuracy": float(all_conflict["branch_accuracy"]),
                "full_route_success": float(all_conflict["full_route_success"]),
                "effective_per_fork_success": float(all_conflict["effective_per_fork_success"]),
                "independent_prediction": float(all_conflict["independent_compounding_prediction"]),
                "pooled_independent_prediction": float(all_conflict["pooled_independent_prediction"]),
                "compounding_gap": float(all_conflict["compounding_gap"]),
                "pooled_compounding_gap": float(all_conflict["full_route_success"])
                - float(all_conflict["pooled_independent_prediction"]),
                "proxy_branch_agreement": float(all_conflict["proxy_branch_agreement"]),
                "first_divergence_mean": float(all_conflict["first_divergence_mean"]),
                # The registered single-conflict estimand is the response at
                # the intervened stage.  Overall panel branch accuracy is kept
                # separately because it is diluted by D-1 easy stages.
                "single_conflict_branch_accuracy": conflicted_stage_accuracy,
                "single_conflict_conflicted_stage_accuracy": conflicted_stage_accuracy,
                "single_conflict_overall_branch_accuracy": mean(
                    float(item["branch_accuracy"]) for item in singles
                ),
                "single_conflict_nonconflicted_stage_accuracy": (
                    mean(nonconflicted_values) if nonconflicted_values else None
                ),
                "single_conflict_full_route_success": mean(
                    float(item["full_route_success"]) for item in singles
                ),
                "address_reversed_branch_accuracy": reversed_accuracy,
                "address_removed_branch_accuracy": removed_accuracy,
                "address_reversed_delta": reversed_accuracy - base_accuracy,
                "address_removed_delta": removed_accuracy - base_accuracy,
                "address_reversed_abs_delta": abs(reversed_accuracy - base_accuracy),
                "address_removed_abs_delta": abs(removed_accuracy - base_accuracy),
                "proxy_flip_rate": float(
                    get(run.summary, "final.interventions.flip_P_stage_local_mean.hard_flip_rate")
                ),
                "rule_flip_rate": float(
                    get(run.summary, "final.interventions.flip_R1_stage_local_mean.hard_flip_rate")
                ),
                "proxy_spillover": float(
                    get(
                        run.summary,
                        "final.interventions.flip_P_other_stage_spillover_mean.hard_flip_rate",
                        0.0,
                    )
                ),
                "rule_spillover": float(
                    get(
                        run.summary,
                        "final.interventions.flip_R1_other_stage_spillover_mean.hard_flip_rate",
                        0.0,
                    )
                ),
                "physical_success": float(get(run.summary, "final.physical_rollouts.full_route_success")),
                "physical_branch_accuracy": float(
                    get(run.summary, "final.physical_rollouts.branch_accuracy")
                ),
                "physical_collision_rate": float(get(run.summary, "final.physical_rollouts.collision_rate")),
                "physical_matches_vectorized": int(
                    bool(get(run.summary, "final.physical_rollouts.matches_vectorized_success"))
                ),
                **{
                    f"first_divergence_rate_{stage}": (
                        float(all_conflict["first_divergence_rate"][stage])
                        if stage < len(all_conflict["first_divergence_rate"])
                        else None
                    )
                    for stage in range(5)
                },
            }
        )
    return rows


def analyse_e13(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    cell_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["route_depth"], row["q"], row["k"], row["width"], row["regime"])].append(row)
    for key, part in sorted(grouped.items(), key=lambda item: str(item[0])):
        entry: dict[str, Any] = dict(zip(("route_depth", "q", "k", "width", "regime"), key, strict=True))
        for outcome in (
            "branch_accuracy",
            "full_route_success",
            "effective_per_fork_success",
            "independent_prediction",
            "compounding_gap",
            "pooled_compounding_gap",
            "first_divergence_mean",
            "single_conflict_branch_accuracy",
            "single_conflict_conflicted_stage_accuracy",
            "single_conflict_overall_branch_accuracy",
            "single_conflict_nonconflicted_stage_accuracy",
            "proxy_flip_rate",
            "rule_flip_rate",
            "address_reversed_branch_accuracy",
            "address_removed_branch_accuracy",
            "address_reversed_delta",
            "address_removed_delta",
            "address_reversed_abs_delta",
            "address_removed_abs_delta",
            "proxy_spillover",
            "rule_spillover",
        ):
            estimate = estimate_by_seed(part, outcome, label=f"e13-cell-{key}-{outcome}")
            entry.update({f"{outcome}_{name}": value for name, value in estimate.items()})
        cell_rows.append(entry)

    contrasts: list[dict[str, Any]] = []
    for regime, outcome in product(
        ("fixed_total", "per_fork_matched"),
        ("branch_accuracy", "full_route_success", "effective_per_fork_success", "rule_flip_rate"),
    ):
        selected = [row for row in rows if row["regime"] == regime]
        contrasts.append(
            {
                "comparison": "depth4-depth1",
                "route_depth": "4-1",
                "regime": regime,
                **paired_contrast(
                    selected,
                    value=outcome,
                    factor="route_depth",
                    high=4,
                    low=1,
                    match=("q", "k", "width", "regime", "seed"),
                    label=f"e13-depth-{regime}-{outcome}",
                ),
            }
        )
    # At D=1 the two evidence regimes both use exactly 1,024 updates and are
    # deterministic duplicates for a matched seed/configuration.  A paired
    # "regime effect" there is therefore structural zero, not an estimate of a
    # substantive intervention, and is intentionally omitted.
    for depth, outcome in product(
        (2, 4),
        ("branch_accuracy", "full_route_success", "effective_per_fork_success", "rule_flip_rate"),
    ):
        selected = [row for row in rows if int(row["route_depth"]) == depth]
        contrasts.append(
            {
                "comparison": "per_fork_matched-fixed_total",
                "route_depth": depth,
                "regime": "paired",
                **paired_contrast(
                    selected,
                    value=outcome,
                    factor="regime",
                    high="per_fork_matched",
                    low="fixed_total",
                    match=("route_depth", "q", "k", "width", "seed"),
                    label=f"e13-regime-{depth}-{outcome}",
                ),
            }
        )

    # Difference-in-differences: does matching evidence per fork remove the
    # depth-4 versus depth-1 shift in local goal reliance?
    index = {
        (
            str(row["regime"]),
            int(row["route_depth"]),
            float(row["q"]),
            int(row["k"]),
            int(row["width"]),
            int(row["seed"]),
        ): row
        for row in rows
    }
    did_rows: list[dict[str, Any]] = []
    for outcome in ("branch_accuracy", "effective_per_fork_success", "rule_flip_rate"):
        by_seed: dict[int, list[float]] = defaultdict(list)
        for q, k, width, seed in product((0.75, 0.95, 0.99), (2, 4), (16, 64), SEEDS):
            keys = [
                ("per_fork_matched", 4, q, k, width, seed),
                ("per_fork_matched", 1, q, k, width, seed),
                ("fixed_total", 4, q, k, width, seed),
                ("fixed_total", 1, q, k, width, seed),
            ]
            if not all(key in index for key in keys):
                continue
            by_seed[seed].append(
                (float(index[keys[0]][outcome]) - float(index[keys[1]][outcome]))
                - (float(index[keys[2]][outcome]) - float(index[keys[3]][outcome]))
            )
        collapsed = [mean(by_seed[seed]) for seed in sorted(by_seed) if by_seed[seed]]
        did_rows.append(
            {
                "outcome": outcome,
                "estimand": "(D4-D1)_per_fork_matched - (D4-D1)_fixed_total",
                **bootstrap_interval(collapsed, label=f"e13-did-{outcome}"),
            }
        )

    compounding = {
        "stage_specific_prediction_gap": estimate_by_seed(
            rows, "compounding_gap", label="e13-overall-compounding-gap"
        ),
        "pooled_branch_accuracy_power_gap": estimate_by_seed(
            rows, "pooled_compounding_gap", label="e13-overall-pooled-compounding-gap"
        ),
    }
    depth_regime_summary: list[dict[str, Any]] = []
    depth_regime_cells = (
        (1, "identical_regimes_collapsed"),
        (2, "fixed_total"),
        (2, "per_fork_matched"),
        (4, "fixed_total"),
        (4, "per_fork_matched"),
    )
    for depth, regime in depth_regime_cells:
        # Keep one of the bit-identical D=1 arms so run count cannot be mistaken
        # for twice as much independent evidence.
        actual_regime = "fixed_total" if depth == 1 else regime
        part = [row for row in rows if int(row["route_depth"]) == depth and row["regime"] == actual_regime]
        summary_entry: dict[str, Any] = {
            "route_depth": depth,
            "regime": regime,
            "d1_duplicate_regimes_collapsed": int(depth == 1),
        }
        for outcome in (
            "branch_accuracy",
            "full_route_success",
            "effective_per_fork_success",
            "pooled_independent_prediction",
            "independent_prediction",
            "pooled_compounding_gap",
            "proxy_flip_rate",
            "rule_flip_rate",
        ):
            estimate = estimate_by_seed(part, outcome, label=f"e13-depth-regime-{depth}-{regime}-{outcome}")
            summary_entry.update({f"{outcome}_{name}": value for name, value in estimate.items()})
        depth_regime_summary.append(summary_entry)
    physical = {
        "all_rollouts_match_vectorized": all(bool(row["physical_matches_vectorized"]) for row in rows),
        "max_collision_rate": max((float(row["physical_collision_rate"]) for row in rows), default=None),
        "physical_minus_vectorized_branch_max_abs": max(
            (abs(float(row["physical_branch_accuracy"]) - float(row["branch_accuracy"])) for row in rows),
            default=None,
        ),
    }
    depth_specific_controls: list[dict[str, Any]] = []
    for depth, regime in depth_regime_cells:
        actual_regime = "fixed_total" if depth == 1 else regime
        part = [row for row in rows if int(row["route_depth"]) == depth and row["regime"] == actual_regime]
        control_entry: dict[str, Any] = {
            "route_depth": depth,
            "regime": regime,
            "d1_duplicate_regimes_collapsed": int(depth == 1),
        }
        for outcome in (
            "address_reversed_branch_accuracy",
            "address_removed_branch_accuracy",
            "address_reversed_delta",
            "address_removed_delta",
            "address_reversed_abs_delta",
            "address_removed_abs_delta",
        ):
            estimate = estimate_by_seed(part, outcome, label=f"e13-control-{depth}-{regime}-{outcome}")
            control_entry.update({f"{outcome}_{name}": value for name, value in estimate.items()})
        if depth == 1:
            control_entry.update(
                {
                    "spillover_applicable": 0,
                    "proxy_spillover_estimate": None,
                    "rule_spillover_estimate": None,
                }
            )
        else:
            control_entry["spillover_applicable"] = 1
            for outcome in ("proxy_spillover", "rule_spillover"):
                estimate = estimate_by_seed(part, outcome, label=f"e13-control-{depth}-{regime}-{outcome}")
                control_entry.update({f"{outcome}_{name}": value for name, value in estimate.items()})
        depth_specific_controls.append(control_entry)

    controls = {
        "address_reversed_delta": estimate_by_seed(
            rows, "address_reversed_delta", label="e13-address-reversed"
        ),
        "address_removed_delta": estimate_by_seed(rows, "address_removed_delta", label="e13-address-removed"),
        "address_reversed_abs_delta": estimate_by_seed(
            rows, "address_reversed_abs_delta", label="e13-address-reversed-absolute"
        ),
        "address_removed_abs_delta": estimate_by_seed(
            rows, "address_removed_abs_delta", label="e13-address-removed-absolute"
        ),
        "proxy_other_stage_spillover": estimate_by_seed(
            [row for row in rows if int(row["route_depth"]) > 1],
            "proxy_spillover",
            label="e13-proxy-spillover",
        ),
        "rule_other_stage_spillover": estimate_by_seed(
            [row for row in rows if int(row["route_depth"]) > 1],
            "rule_spillover",
            label="e13-rule-spillover",
        ),
        "by_depth_and_regime": depth_specific_controls,
        "d1_note": (
            "fixed_total and per_fork_matched are deterministic duplicate arms at D=1; "
            "only fixed_total is retained in depth-specific summaries"
        ),
    }
    return {
        "cell_estimates": cell_rows,
        "paired_contrasts": contrasts,
        "depth_regime_interactions": did_rows,
        "depth_regime_summary": depth_regime_summary,
        "overall_compounding_gap": compounding,
        "physical_validation": physical,
        "evaluation_controls": controls,
    }


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
    registered: list[str] = []
    for font_dir in font_dirs:
        for filename in filenames:
            path = font_dir / filename
            if path.is_file():
                fm.fontManager.addfont(str(path))
                registered.append(fm.FontProperties(fname=str(path)).get_name())
    if not registered:
        warnings.warn(
            "Myriad Pro was not found in FORKWORLD_FONT_DIR or ~/Library/Fonts; "
            "falling back to DejaVu Sans. Figure geometry may differ slightly.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "DejaVu Sans"
    return registered[0]


def configure_style() -> None:
    font_family = plot_font_family()
    mpl.rcParams.update(
        {
            "font.family": font_family,
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.titleweight": 600,
            "axes.labelsize": 9.5,
            "axes.labelcolor": INK,
            "axes.edgecolor": "#AAB2BD",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": "#4B5563",
            "ytick.color": "#4B5563",
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "legend.frameon": False,
            "legend.fontsize": 8.4,
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
        xytext=(-13, 8),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=11,
        fontweight=600,
        color=INK,
        annotation_clip=False,
    )


def provisional_note(figure: Figure, provisional: bool) -> None:
    if provisional:
        figure.text(
            0.99,
            0.008,
            "Provisional: completed subset only",
            ha="right",
            va="bottom",
            fontsize=7.2,
            color=GRAY,
        )


def empty_axis(axis: Axes, title: str) -> None:
    axis.set_axis_off()
    axis.text(0.5, 0.55, title, ha="center", va="center", weight=600, color=INK)
    axis.text(
        0.5,
        0.43,
        "No completed runs yet",
        ha="center",
        va="center",
        color=GRAY,
        fontsize=8.5,
    )


def save_figure(figure: Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    figure.savefig(FIGURES / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def point_interval(
    rows: Sequence[Mapping[str, Any]], outcome: str, label: str
) -> tuple[float, float, float] | None:
    estimate = estimate_by_seed(rows, outcome, label=label)
    if estimate["estimate"] is None:
        return None
    return (
        float(estimate["estimate"]),
        float(estimate["ci_low"]),
        float(estimate["ci_high"]),
    )


def draw_interval_line(
    axis: Axes,
    xs: Sequence[float],
    estimates: Sequence[tuple[float, float, float] | None],
    *,
    color: str,
    marker: str = "o",
    linestyle: str = "-",
    label: str | None = None,
    alpha: float = 1.0,
) -> None:
    present = [(x, value) for x, value in zip(xs, estimates, strict=True) if value is not None]
    if not present:
        return
    px = np.asarray([item[0] for item in present])
    center = np.asarray([item[1][0] for item in present])
    low = np.asarray([item[1][1] for item in present])
    high = np.asarray([item[1][2] for item in present])
    axis.plot(
        px,
        center,
        color=color,
        marker=marker,
        ms=4.2,
        lw=1.8,
        linestyle=linestyle,
        label=label,
        alpha=alpha,
    )
    axis.fill_between(px, low, high, color=color, alpha=0.10 * alpha, linewidth=0)


def figure_e10(rows: Sequence[Mapping[str, Any]], *, provisional: bool) -> None:
    figure = plt.figure(figsize=(7.35, 3.80))
    grid = figure.add_gridspec(
        2,
        3,
        height_ratios=(1.0, 0.19),
        left=0.085,
        right=0.985,
        top=0.83,
        bottom=0.045,
        wspace=0.28,
        hspace=0.60,
    )
    axes = [figure.add_subplot(grid[0, index]) for index in range(3)]
    legend_axis = figure.add_subplot(grid[1, :])
    legend_axis.set_axis_off()
    betas = (0.0, 0.003, 0.01, 0.03, 0.10, 0.30)
    q_colors = {0.50: GRAY, 0.75: SKY, 0.90: TEAL, 0.99: ORANGE}
    selected = [row for row in rows if row["arm"] == "actor_critic_full"]
    for index, (axis, k) in enumerate(zip(axes, (2, 3, 4), strict=True)):
        if not selected:
            empty_axis(axis, f"Parity degree {k}")
            continue
        for q in (0.50, 0.75, 0.90, 0.99):
            estimates = [
                point_interval(
                    [row for row in selected if row["k"] == k and row["q"] == q and row["beta"] == f(beta)],
                    "rho_y",
                    f"fig-e10-{k}-{q}-{beta}",
                )
                for beta in betas
            ]
            draw_interval_line(axis, range(len(betas)), estimates, color=q_colors[q])
        axis.axhline(0.9, color=GRAY, ls=":", lw=0.9)
        axis.set_xticks(range(len(betas)), ("0", ".003", ".01", ".03", ".10", ".30"), rotation=35)
        axis.set_ylim(-0.03, 1.03)
        axis.set_xlabel("Entropy coefficient $\\beta$")
        if index == 0:
            axis.set_ylabel("Conflict-set intended reliance")
        else:
            axis.tick_params(labelleft=False)
        axis.set_title(f"Parity degree $k={k}$", loc="left", pad=7)
        axis.grid(axis="y", color=LIGHT_GRAY, lw=0.65)
        panel_label(axis, chr(ord("a") + index))
    handles = [
        Line2D([0], [0], color=q_colors[q], marker="o", ms=4, lw=1.8, label=f"$q={q:g}$")
        for q in (0.50, 0.75, 0.90, 0.99)
    ]
    legend_axis.legend(handles=handles, loc="center", ncol=4)
    figure.suptitle(
        "Entropy changes the RL shortcut boundary", x=0.085, ha="left", y=0.975, fontsize=12, fontweight=600
    )
    provisional_note(figure, provisional)
    save_figure(figure, "fig10_rl_entropy_followup")


def figure_e11(rows: Sequence[Mapping[str, Any]], *, provisional: bool) -> None:
    figure = plt.figure(figsize=(7.35, 6.55))
    grid = figure.add_gridspec(
        3,
        2,
        height_ratios=(1.0, 1.0, 0.18),
        left=0.10,
        right=0.985,
        top=0.91,
        bottom=0.055,
        wspace=0.34,
        hspace=0.68,
    )
    axis_a = figure.add_subplot(grid[0, 0])
    axis_b = figure.add_subplot(grid[0, 1])
    axis_c = figure.add_subplot(grid[1, :])
    legend_axis = figure.add_subplot(grid[2, :])
    legend_axis.set_axis_off()
    rules = ("parity", "majority", "conjunction", "multiplexer")
    colors = {"parity": PURPLE, "majority": BLUE, "conjunction": ORANGE, "multiplexer": TEAL}
    widths = (8, 16, 32, 64, 128)
    hard = [row for row in rows if row["q"] == 0.99 and row["depth"] == 2]
    for axis, outcome, title, ylabel in (
        (axis_a, "exact_decoder_accuracy", "Standalone exact-rule calibration", "Decoder accuracy"),
        (axis_b, "rho_y", "Control in competition with a 99% proxy", "Conflict-set intended reliance"),
    ):
        if not rows:
            empty_axis(axis, title)
            continue
        for rule in rules:
            estimates = [
                point_interval(
                    [row for row in hard if row["rule"] == rule and row["width"] == width],
                    outcome,
                    f"fig-e11-{outcome}-{rule}-{width}",
                )
                for width in widths
            ]
            draw_interval_line(axis, range(len(widths)), estimates, color=colors[rule])
        axis.set_xticks(range(len(widths)), [str(width) for width in widths])
        axis.set_xlabel("Hidden width")
        axis.set_ylabel(ylabel)
        axis.set_ylim(-0.03, 1.03)
        axis.grid(axis="y", color=LIGHT_GRAY, lw=0.65)
        axis.set_title(title, loc="left", pad=7)
    panel_label(axis_a, "a")
    panel_label(axis_b, "b")

    if rows:
        grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[(row["rule"], row["q"], row["width"], row["depth"])].append(row)
        for rule in rules:
            means = [
                (
                    mean(float(item["exact_decoder_accuracy"]) for item in part),
                    mean(float(item["rho_y"]) for item in part),
                )
                for key, part in grouped.items()
                if key[0] == rule
            ]
            axis_c.scatter(
                [item[0] for item in means],
                [item[1] for item in means],
                s=21,
                color=colors[rule],
                alpha=0.58,
                linewidths=0,
                label=rule,
            )
        axis_c.plot([0, 1], [0, 1], color=GRAY, lw=1, ls="--")
        axis_c.axvspan(0.95, 1.0, color=LIGHT_GRAY, alpha=0.32, linewidth=0)
        axis_c.set_xlim(0.45, 1.015)
        axis_c.set_ylim(-0.03, 1.03)
        axis_c.set_xlabel("Standalone exact-rule decoder accuracy (cell mean)")
        axis_c.set_ylabel("Competition intended reliance (cell mean)")
        axis_c.grid(color=LIGHT_GRAY, lw=0.55)
        axis_c.set_title("Accessibility and behavioral control across all matched cells", loc="left", pad=7)
    else:
        empty_axis(axis_c, "Accessibility and behavioral control")
    panel_label(axis_c, "c")
    handles = [
        Line2D([0], [0], marker="o", color=colors[rule], lw=1.8, label=rule.capitalize()) for rule in rules
    ]
    legend_axis.legend(handles=handles, loc="center", ncol=4, title="Exact encoding")
    figure.suptitle("Exact encodings beyond parity", x=0.10, ha="left", y=0.985, fontsize=12, fontweight=600)
    provisional_note(figure, provisional)
    save_figure(figure, "fig11_rule_families_followup")


def figure_e12(rows: Sequence[Mapping[str, Any]], *, provisional: bool) -> None:
    figure = plt.figure(figsize=(7.35, 6.55))
    grid = figure.add_gridspec(
        3,
        2,
        height_ratios=(1.0, 1.0, 0.20),
        left=0.10,
        right=0.985,
        top=0.91,
        bottom=0.05,
        wspace=0.34,
        hspace=0.68,
    )
    axis_a = figure.add_subplot(grid[0, 0])
    axis_b = figure.add_subplot(grid[0, 1])
    axis_c = figure.add_subplot(grid[1, :])
    legend_axis = figure.add_subplot(grid[2, :])
    legend_axis.set_axis_off()
    candidate_colors = {"P": ORANGE, "Q": TEAL, "Y": BLUE}
    if not rows:
        for axis, title in (
            (axis_a, "Candidate calibration"),
            (axis_b, "Diagnostic behavior"),
            (axis_c, "Causal control shares"),
        ):
            empty_axis(axis, title)
    else:
        candidates = (("P", "calibration_p"), ("Q", "calibration_q"), ("Y", "calibration_y"))
        for index, (candidate, outcome) in enumerate(candidates):
            point = point_interval(rows, outcome, f"fig-e12-calibration-{candidate}")
            if point is None:
                continue
            center, low, high = point
            axis_a.errorbar(
                index,
                center,
                yerr=[[center - low], [high - center]],
                fmt="o",
                ms=6,
                capsize=3,
                color=candidate_colors[candidate],
                lw=1.4,
            )
        axis_a.set_xticks(range(3), ("$P$ (direct)", "$Q$ (encoded)", "$Y$ (exact)"))
        axis_a.set_ylim(-0.03, 1.03)
        axis_a.set_ylabel("Standalone decoder accuracy")
        axis_a.grid(axis="y", color=LIGHT_GRAY, lw=0.65)
        axis_a.set_title("Accessibility of each candidate", loc="left", pad=7)

        panels = (
            ("both wrong", "both_wrong_accuracy"),
            ("$P$ wrong", "p_wrong_accuracy"),
            ("$Q$ wrong", "q_wrong_accuracy"),
        )
        overlap_styles = {"independent": (PURPLE, "o"), "nested": (GOLD, "s")}
        for overlap, (color, marker) in overlap_styles.items():
            estimates = [
                point_interval(
                    [row for row in rows if row["overlap"] == overlap],
                    outcome,
                    f"fig-e12-panel-{overlap}-{outcome}",
                )
                for _, outcome in panels
            ]
            draw_interval_line(
                axis_b,
                range(3),
                estimates,
                color=color,
                marker=marker,
                linestyle="-",
            )
        axis_b.set_xticks(range(3), [name for name, _ in panels])
        axis_b.set_ylim(-0.03, 1.03)
        axis_b.set_ylabel("Accuracy on intended goal")
        axis_b.grid(axis="y", color=LIGHT_GRAY, lw=0.65)
        axis_b.set_title("Three panels identify rule control", loc="left", pad=7)

        relation_order = ("$q_P<q_Q$", "$q_P=q_Q$", "$q_P>q_Q$")
        relation_rows: dict[str, list[Mapping[str, Any]]] = {
            relation_order[0]: [row for row in rows if float(row["q_p"]) < float(row["q_q"])],
            relation_order[1]: [row for row in rows if float(row["q_p"]) == float(row["q_q"])],
            relation_order[2]: [row for row in rows if float(row["q_p"]) > float(row["q_q"])],
        }
        bottoms = np.zeros(3)
        for candidate, outcome in (("P", "share_p"), ("Q", "share_q"), ("Y", "share_y")):
            values = []
            for relation in relation_order:
                point = point_interval(
                    relation_rows[relation], outcome, f"fig-e12-share-{relation}-{candidate}"
                )
                values.append(0.0 if point is None else point[0])
            axis_c.bar(
                range(3),
                values,
                bottom=bottoms,
                color=candidate_colors[candidate],
                width=0.58,
                label=candidate,
            )
            bottoms += np.asarray(values)
        axis_c.set_xticks(range(3), relation_order)
        axis_c.set_ylim(0, 1.02)
        axis_c.set_ylabel("Normalized mean causal hard-flip influence")
        axis_c.grid(axis="y", color=LIGHT_GRAY, lw=0.65, zorder=0)
        axis_c.set_axisbelow(True)
        axis_c.set_title("Control shifts with the relative fidelity of the two proxies", loc="left", pad=7)
    panel_label(axis_a, "a")
    panel_label(axis_b, "b")
    panel_label(axis_c, "c")
    handles: list[Any] = [
        Patch(facecolor=candidate_colors[name], label=f"{name} channel") for name in ("P", "Q", "Y")
    ]
    handles.extend(
        [
            Line2D([0], [0], color=PURPLE, marker="o", lw=1.8, label="Independent proxy errors"),
            Line2D([0], [0], color=GOLD, marker="s", lw=1.8, label="Nested proxy errors"),
        ]
    )
    legend_axis.legend(handles=handles, loc="center", ncol=5)
    figure.suptitle(
        "Two proxies compete with an exact goal", x=0.10, ha="left", y=0.985, fontsize=12, fontweight=600
    )
    provisional_note(figure, provisional)
    save_figure(figure, "fig12_competing_goals_followup")


def figure_e13(rows: Sequence[Mapping[str, Any]], *, provisional: bool) -> None:
    figure = plt.figure(figsize=(7.35, 6.55))
    grid = figure.add_gridspec(
        3,
        2,
        height_ratios=(1.0, 1.0, 0.20),
        left=0.10,
        right=0.985,
        top=0.91,
        bottom=0.045,
        wspace=0.34,
        hspace=0.68,
    )
    axis_a = figure.add_subplot(grid[0, 0])
    axis_b = figure.add_subplot(grid[0, 1])
    axis_c = figure.add_subplot(grid[1, :])
    legend_axis = figure.add_subplot(grid[2, :])
    legend_axis.set_axis_off()
    depths = (1, 2, 4)
    regime_style = {
        "fixed_total": (ORANGE, "--", "Fixed total updates"),
        "per_fork_matched": (BLUE, "-", "Updates matched per fork"),
    }
    if not rows:
        for axis, title in (
            (axis_a, "Local branch behavior"),
            (axis_b, "Whole-route compounding"),
            (axis_c, "Causal rule control"),
        ):
            empty_axis(axis, title)
    else:
        for regime, (color, linestyle, _) in regime_style.items():
            estimates = [
                point_interval(
                    [row for row in rows if row["regime"] == regime and row["route_depth"] == depth],
                    "branch_accuracy",
                    f"fig-e13-branch-{regime}-{depth}",
                )
                for depth in depths
            ]
            draw_interval_line(axis_a, range(3), estimates, color=color, linestyle=linestyle)
        axis_a.set_xticks(range(3), [str(depth) for depth in depths])
        axis_a.set_xlabel("Reward-relevant forks")
        axis_a.set_ylabel("All-conflict branch accuracy")
        axis_a.set_ylim(-0.03, 1.03)
        axis_a.grid(axis="y", color=LIGHT_GRAY, lw=0.65)
        axis_a.set_title("Local behavior as routes deepen", loc="left", pad=7)

        # Actual whole-route success and an exploratory product of the
        # stage-specific branch accuracies are shown together; color denotes
        # the estimand and line style denotes evidence regime.  The registered
        # pooled branch_accuracy**D approximation remains available in derived data.
        for regime, (_, linestyle, _) in regime_style.items():
            for outcome, color, marker in (
                ("full_route_success", TEAL, "o"),
                ("independent_prediction", PURPLE, "s"),
            ):
                estimates = [
                    point_interval(
                        [row for row in rows if row["regime"] == regime and row["route_depth"] == depth],
                        outcome,
                        f"fig-e13-route-{regime}-{depth}-{outcome}",
                    )
                    for depth in depths
                ]
                draw_interval_line(
                    axis_b,
                    range(3),
                    estimates,
                    color=color,
                    marker=marker,
                    linestyle=linestyle,
                )
        axis_b.set_xticks(range(3), [str(depth) for depth in depths])
        axis_b.set_xlabel("Reward-relevant forks")
        axis_b.set_ylabel("Whole-route success")
        axis_b.set_ylim(-0.03, 1.03)
        axis_b.grid(axis="y", color=LIGHT_GRAY, lw=0.65)
        axis_b.set_title("Actual success versus compounding prediction", loc="left", pad=7)

        x = np.arange(3)
        offsets = {"fixed_total": -0.04, "per_fork_matched": 0.04}
        for regime, (_, linestyle, _) in regime_style.items():
            for outcome, color, marker in (
                ("proxy_flip_rate", ORANGE, "o"),
                ("rule_flip_rate", BLUE, "s"),
            ):
                estimates = [
                    point_interval(
                        [row for row in rows if row["regime"] == regime and row["route_depth"] == depth],
                        outcome,
                        f"fig-e13-causal-{regime}-{depth}-{outcome}",
                    )
                    for depth in depths
                ]
                causal_x = [
                    float(index) if depth == 1 else float(index) + offsets[regime]
                    for index, depth in enumerate(depths)
                ]
                draw_interval_line(
                    axis_c,
                    causal_x,
                    estimates,
                    color=color,
                    marker=marker,
                    linestyle=linestyle,
                )
        axis_c.set_xticks(x, [str(depth) for depth in depths])
        axis_c.set_xlabel("Reward-relevant forks")
        axis_c.set_ylabel("Stage-local causal hard-flip rate")
        axis_c.set_ylim(-0.03, 1.03)
        axis_c.grid(axis="y", color=LIGHT_GRAY, lw=0.65)
        axis_c.set_title("Stage-local interventions separate proxy and exact-rule control", loc="left", pad=7)
    panel_label(axis_a, "a")
    panel_label(axis_b, "b")
    panel_label(axis_c, "c")
    handles = [
        Line2D([0], [0], color=color, ls=linestyle, lw=1.8, label=label)
        for color, linestyle, label in regime_style.values()
    ]
    handles.extend(
        [
            Line2D([0], [0], color=TEAL, marker="o", lw=1.8, label="Observed route success"),
            Line2D([0], [0], color=PURPLE, marker="s", lw=1.8, label="Stage-product prediction"),
            Line2D([0], [0], color=ORANGE, marker="o", lw=1.8, label="Proxy flip"),
            Line2D([0], [0], color=BLUE, marker="s", lw=1.8, label="Exact-rule flip"),
        ]
    )
    legend_axis.legend(handles=handles, loc="center", ncol=3, columnspacing=1.6)
    figure.suptitle(
        "RouteWorld repeats the reward-relevant decision",
        x=0.10,
        ha="left",
        y=0.985,
        fontsize=12,
        fontweight=600,
    )
    provisional_note(figure, provisional)
    save_figure(figure, "fig13_routeworld_followup")


def write_outputs(
    rows: Mapping[str, Sequence[Mapping[str, Any]]],
    analyses: Mapping[str, Mapping[str, Any]],
    audits: Sequence[Mapping[str, Any]],
    *,
    allow_incomplete: bool,
    artifacts: Path,
) -> None:
    for family in ("e10", "e11", "e12", "e13"):
        write_csv(DERIVED / f"followup_{family}_runs.csv", rows[family])
    write_csv(DERIVED / "followup_e10_cell_estimates.csv", analyses["e10"]["cell_estimates"])
    write_csv(DERIVED / "followup_e10_paired_contrasts.csv", analyses["e10"]["paired_contrasts"])
    write_csv(DERIVED / "followup_e10_arm_interactions.csv", analyses["e10"]["arm_interactions"])
    write_csv(DERIVED / "followup_e11_cell_estimates.csv", analyses["e11"]["cell_estimates"])
    write_csv(DERIVED / "followup_e11_rule_contrasts.csv", analyses["e11"]["matched_rule_contrasts"])
    write_csv(DERIVED / "followup_e11_associations.csv", analyses["e11"]["associations"])
    write_csv(
        DERIVED / "followup_e11_exact_only_acquisition.csv",
        analyses["e11"]["exact_only_acquisition"],
    )
    write_csv(
        DERIVED / "followup_e11_q99_competition_acquisition.csv",
        analyses["e11"]["q99_competition_acquisition"],
    )
    write_csv(DERIVED / "followup_e12_overall.csv", analyses["e12"]["overall"])
    write_csv(DERIVED / "followup_e12_factorial_contrasts.csv", analyses["e12"]["factorial_contrasts"])
    write_csv(
        DERIVED / "followup_e12_mixture_by_overlap_ky.csv",
        analyses["e12"]["diagnostic_mixture"]["by_overlap_and_k_y"],
    )
    write_csv(
        DERIVED / "followup_e12_mixture_contrasts.csv",
        analyses["e12"]["diagnostic_mixture"]["paired_contrasts"],
    )
    write_csv(
        DERIVED / "followup_e12_mixture_purity_strata.csv",
        analyses["e12"]["diagnostic_mixture"]["purity_stratification"]["table"],
    )
    write_csv(DERIVED / "followup_e13_cell_estimates.csv", analyses["e13"]["cell_estimates"])
    write_csv(DERIVED / "followup_e13_paired_contrasts.csv", analyses["e13"]["paired_contrasts"])
    write_csv(
        DERIVED / "followup_e13_depth_regime_interactions.csv",
        analyses["e13"]["depth_regime_interactions"],
    )
    write_csv(
        DERIVED / "followup_e13_depth_regime_summary.csv",
        analyses["e13"]["depth_regime_summary"],
    )
    write_csv(
        DERIVED / "followup_e13_depth_specific_controls.csv",
        analyses["e13"]["evaluation_controls"]["by_depth_and_regime"],
    )
    machine = {
        "analysis_version": 2,
        "artifacts_root": str(artifacts.resolve()),
        "allow_incomplete": allow_incomplete,
        "provisional": any(not bool(audit["complete"]) for audit in audits),
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "interval": "percentile 95%",
            "inference_unit": "training seed",
            "collapse_rule": "average registered nuisance cells within seed before resampling",
            "randomness": "deterministic label-derived BLAKE2 seeds",
        },
        "audits": list(audits),
        "e10": analyses["e10"],
        "e11": analyses["e11"],
        "e12": analyses["e12"],
        "e13": analyses["e13"],
        "figures": [
            "fig10_rl_entropy_followup",
            "fig11_rule_families_followup",
            "fig12_competing_goals_followup",
            "fig13_routeworld_followup",
        ],
    }
    json_dump(DERIVED / "followup_analysis.json", machine)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=DEFAULT_ARTIFACTS,
        help=f"run artifact root (default: {DEFAULT_ARTIFACTS})",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="analyse the completed subset and mark every output provisional",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_runs: dict[str, list[Run]] = {}
    audits: list[dict[str, Any]] = []
    for spec in SPECS:
        runs, audit = load_experiment(args.artifacts, spec, allow_incomplete=bool(args.allow_incomplete))
        all_runs[spec.family] = runs
        audits.append(audit)

    fingerprints = {
        str(fingerprint)
        for audit in audits
        for fingerprint in audit["implementation_fingerprints"]
    }
    if len(fingerprints) > 1:
        raise RuntimeError(
            "E10--E13 mix implementation fingerprints across experiment families: "
            f"{sorted(fingerprints)}"
        )
    if not args.allow_incomplete and len(fingerprints) != 1:
        raise RuntimeError(
            "complete E10--E13 analysis requires exactly one implementation fingerprint"
        )

    rows: dict[str, list[dict[str, Any]]] = {
        "e10": rows_e10(all_runs["e10"]),
        "e11": rows_e11(all_runs["e11"]),
        "e12": rows_e12(all_runs["e12"]),
        "e13": rows_e13(all_runs["e13"]),
    }
    complete = {str(audit["family"]): bool(audit["complete"]) for audit in audits}
    analyses: dict[str, dict[str, Any]] = {
        "e10": analyse_e10(rows["e10"], complete=complete["e10"]),
        "e11": analyse_e11(rows["e11"]),
        "e12": analyse_e12(rows["e12"]),
        "e13": analyse_e13(rows["e13"]),
    }
    write_outputs(
        rows,
        analyses,
        audits,
        allow_incomplete=bool(args.allow_incomplete),
        artifacts=args.artifacts,
    )
    configure_style()
    figure_e10(rows["e10"], provisional=not complete["e10"])
    figure_e11(rows["e11"], provisional=not complete["e11"])
    figure_e12(rows["e12"], provisional=not complete["e12"])
    figure_e13(rows["e13"], provisional=not complete["e13"])

    print("Forkworld follow-up analysis")
    for audit in audits:
        status = "complete" if audit["complete"] else "provisional"
        print(f"  {audit['family']}: {audit['completed_runs']}/{audit['expected_runs']} completed ({status})")
    trigger = analyses["e10"]["trigger"]
    print(f"  E10 staged-follow-up trigger: {trigger['status']}")
    print(f"  derived data: {DERIVED / 'followup_analysis.json'}")
    print(f"  figures: {FIGURES / 'fig10_rl_entropy_followup.pdf'} through fig13")


if __name__ == "__main__":
    main()
