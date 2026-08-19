#!/usr/bin/env python3
"""Strict audit, analysis, and visualization for E17 winner knockout.

E17 is adaptive and post-hoc.  The independent unit is a training seed; the six
trajectories, checkpoints, probes, and factorial rows are repeated measurements.
This file deliberately has two modes.  ``pilot`` audits complete six-arm blocks
for a nonempty subset of the registered seeds and emits only integrity outputs.
``full`` accepts exactly the frozen 20 x 6 grid and is the only mode that emits
inferential contrasts or a scientific figure.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias, cast

os.environ.setdefault("MPLCONFIGDIR", "/tmp/forkworld-e17-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/forkworld-e17-xdg")
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
    make_competing_bundle,
    make_competing_factorial_dataset,
)
from forkworld.handoff import (  # noqa: E402
    audit_compute_sham,
    audit_handoff_phase_b,
    make_compute_sham,
    make_handoff_phase_b,
    normalized_control_auc,
    persistent_handoff,
    pure_control,
    semantic_batch_digest,
    stable_state_digest,
    static_sampler_digest,
)
from forkworld.protocols import build_model  # noqa: E402

DEFAULT_ARTIFACTS = REPO / "artifacts-e17"
DEFAULT_PILOT_ARTIFACTS = REPO / "artifacts-e17-pilot"
DERIVED = HERE / "derived"
FIGURES = HERE / "figures"

SEEDS = (
    157,
    163,
    167,
    173,
    179,
    181,
    191,
    193,
    197,
    199,
    211,
    223,
    227,
    229,
    233,
    239,
    241,
    251,
    257,
    263,
)
TRAJECTORIES = (
    "independent_carry",
    "independent_reset",
    "nested_carry",
    "nested_reset",
    "scratch",
    "sham",
)
HISTORICAL = frozenset(TRAJECTORIES[:4])
PHASE_A_STEPS = (33, 45)
PHASE_B_STEPS = (
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
    418,
    575,
    790,
    1024,
)
GOALS = ("P", "Q", "Y")
LAYERS = ("first_hidden", "final_hidden")
BOOTSTRAP_DRAWS = 4_000
THRESHOLD = 0.90
MARGIN = 0.10
AUC_HORIZON = 128
CENSOR_HORIZON = 1024
ARTIFACT_SCHEMA_VERSION = 1
SOURCE_FINGERPRINT_SCHEMA_VERSION = 1
NON_SCIENTIFIC_RUN_FIELDS = frozenset({"output_root", "resume", "seeds"})

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


def expect_equal(errors: list[str], run_id: str, actual: Any, expected: Any, label: str) -> None:
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
    if not math.isfinite(numeric) or not math.isclose(numeric, expected, rel_tol=0.0, abs_tol=atol):
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


def expected_artifact_run_id(config: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    scientific = {
        str(key): value for key, value in config.items() if key != "seed" and not str(key).startswith("_")
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
        "source_fingerprint_schema_version": get(implementation, "source_fingerprint_schema_version"),
        "implementation_fingerprint": get(implementation, "implementation_fingerprint"),
    }
    raw = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


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
    def trajectory(self) -> str:
        return str(self.summary["trajectory"])

    def snapshot(self, phase: str, step: int) -> Mapping[str, Any]:
        result = get(self.summary, f"{phase}_snapshots.{step}")
        if not isinstance(result, Mapping):
            raise RuntimeError(f"{self.path.name}: missing {phase} snapshot {step}")
        return cast(Mapping[str, Any], result)

    def metric(self, stage: str, split: str, intervention: str, metric: str, step: int) -> MetricPoint:
        key = (stage, split, intervention, metric)
        if key not in self.metrics or step not in self.metrics[key]:
            raise RuntimeError(f"{self.path.name}: missing metric {key} at local step {step}")
        return self.metrics[key][step]


def _metric_is_needed(stage: str, split: str, intervention: str, metric: str) -> bool:
    if stage in {"phase_a_behavior", "phase_b_behavior"}:
        return (
            split == "factorial_eval"
            and intervention == "none"
            and metric
            in {
                "rho_p",
                "rho_q",
                "rho_y_code",
                "target_accuracy",
            }
        )
    if stage in {"phase_a_causal", "phase_b_causal"}:
        return (
            split == "factorial_eval"
            and intervention in GOALS
            and metric in {"causal_score", "causal_prob_score"}
        )
    if stage in {"phase_a_probe", "phase_b_probe"}:
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
    if stage in {"phase_a_truth_table", "phase_b_truth_table"}:
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
        "phase_a_optimization",
        "phase_a_sham_optimization",
        "phase_b_optimization",
    }:
        return (
            split == "train_minibatch"
            and intervention == "none"
            and metric
            in {
                "loss",
                "primary_loss",
                "train_batch_accuracy",
                "optimizer_steps",
            }
        )
    return False


def _load_metric_index(
    path: Path,
    *,
    run_id: str,
    seed: int,
    trajectory: str,
    errors: list[str],
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
            if record.get("experiment") != "h14":
                errors.append(f"{run_id}: metric line {line_number} experiment mismatch")
            stage = str(record.get("stage"))
            expected_condition = (
                "adaptive_winner_knockout" if stage == "final" else f"winner_knockout:{trajectory}"
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
                    value=float(record["value"]),
                    n=int(record["n"]),
                    global_step=int(record["global_step"]),
                    stage_step=local,
                    examples_seen=int(record["examples_seen"]),
                )
            except (KeyError, TypeError, ValueError) as error:
                errors.append(f"{run_id}: malformed selected metric line {line_number}: {error}")
                continue
            if not math.isfinite(point.value):
                errors.append(f"{run_id}: non-finite metric line {line_number}")
            key = (stage, split, intervention, metric)
            if local in result[key]:
                errors.append(f"{run_id}: duplicate selected metric {key} at {local}")
            result[key][local] = point
    return dict(result), line_count


def _require_metric_steps(run: Run, errors: list[str], key: MetricKey, expected: Sequence[int]) -> None:
    observed = set(run.metrics.get(key, {}))
    wanted = set(expected)
    if observed != wanted:
        errors.append(
            f"{run.path.name}: {key} has missing={sorted(wanted - observed)} "
            f"unexpected={sorted(observed - wanted)}"
        )


def _snapshot_metric_value(snapshot: Mapping[str, Any], key: MetricKey) -> float:
    stage, _split, intervention, metric = key
    if stage.endswith("_behavior"):
        field = {"rho_p": "P", "rho_q": "Q", "rho_y_code": "Y"}.get(metric)
        if field is not None:
            return float(get(snapshot, f"behavior.{field}"))
        return float(snapshot["target_accuracy"])
    if stage.endswith("_causal"):
        field = "causal_probability" if metric == "causal_prob_score" else "causal"
        return float(get(snapshot, f"{field}.{intervention}"))
    if stage.endswith("_probe"):
        if metric.startswith("representations__"):
            _, layer, _, label = metric.split("__", 3)
            return float(get(snapshot, f"probe_heldout_accuracy.{layer}.{label}"))
        lookup = {
            "raw_codeword_count": 64,
            "truth_table_control_positive_fraction": 0.5,
            "n_train": 2048,
            "n_heldout": 4096,
            "n_labels": 8,
        }
        return float(lookup[metric])
    if stage.endswith("_truth_table"):
        return float(get(snapshot, f"truth_table.{metric}"))
    raise KeyError(key)


def validate_run(run: Run, errors: list[str]) -> None:
    run_id = run.path.name
    trajectory = run.trajectory
    fixed_config: Mapping[str, Any] = {
        "schema_version": 1,
        "experiment.hypothesis": "h14",
        "experiment.name": "winner_knockout_handoff",
        "experiment.mode": "adaptive_winner_knockout",
        "run.device": "cpu",
        "run.task_levels": ["choice"],
        "run.save_checkpoints": False,
        "evaluation.save_predictions": False,
        "evaluation.bootstrap_samples": 4_000,
        "evaluation.persistence": 2,
        "data.n_train": 10_000,
        "data.n_validation": 4_000,
        "data.n_eval": 10_000,
        "data.q": 0.90,
        "data.k": 3,
        "data.max_k": 5,
        "data.state_dim": 8,
        "model.width": 64,
        "model.depth": 2,
        "model.activation": "relu",
        "model.residual": False,
        "model.bias": True,
        "model.nuisance_bits": 0,
        "update.mode": "full",
        "update.budget": "full",
        "train.algorithm": "clean_sft",
        "train.steps": 1_024,
        "train.batch_size": 250,
        "train.learning_rate": 0.003,
        "train.weight_decay": 0.0,
        "train.grad_clip": 1.0,
        "h14.q_p": 0.90,
        "h14.q_q": 0.90,
        "h14.k_q": 2,
        "h14.k_y": 3,
        "h14.max_k_q": 3,
        "h14.max_k_y": 5,
        "h14.phase_a_steps": 45,
        "h14.phase_b_steps": 1_024,
        "h14.eligibility_steps": list(PHASE_A_STEPS),
        "h14.phase_b_checkpoints": list(PHASE_B_STEPS),
        "h14.probe_train_n": 2_048,
        "h14.probe_eval_n": 4_096,
        "h14.probe_ridge": 0.001,
        "h14.truth_table_control_seed": 1_500_450_271,
        "h14.sham_source_repeats": 79,
        "h14.auc_horizon": AUC_HORIZON,
        "h14.pure_threshold": THRESHOLD,
        "h14.pure_margin": MARGIN,
    }
    for path, expected in fixed_config.items():
        expect_equal(errors, run_id, get(run.config, path), expected, path)
    expect_equal(
        errors,
        run_id,
        get(run.config, "train.eval_steps"),
        list(PHASE_B_STEPS[1:]),
        "train.eval_steps",
    )
    expect_equal(errors, run_id, get(run.config, "h14.trajectory"), trajectory, "trajectory")
    expect_equal(errors, run_id, int(get(run.config, "seed", -1)), run.seed, "seed")

    phase_kind = {
        "independent_carry": "independent_history",
        "independent_reset": "independent_history",
        "nested_carry": "nested_history",
        "nested_reset": "nested_history",
        "scratch": "scratch",
        "sham": "generic_compute_sham",
    }[trajectory]
    summary_fixed: Mapping[str, Any] = {
        "hypothesis": "h14",
        "condition": f"winner_knockout:{trajectory}",
        "trajectory": trajectory,
        "design_status": "adaptive_posthoc_winner_knockout",
        "phase_a_kind": phase_kind,
        "optimizer_transition": ("carry" if trajectory.endswith("_carry") else "reset_or_fresh"),
        "eligibility.historical_arm": trajectory in HISTORICAL,
        "eligibility.threshold": THRESHOLD,
        "eligibility.margin": MARGIN,
        "eligibility.primary_analysis_rule": "intersection eligible across both histories",
        "eligibility.minimum_intersection_size": 15,
        "data.n_train": 10_000,
        "data.n_validation": 4_000,
        "data.n_eval": 10_000,
        "data.q_p_phase_a": 0.90,
        "data.q_q": 0.90,
        "data.k_q": 2,
        "data.k_y": 3,
        "data.max_k_q": 3,
        "data.max_k_y": 5,
        "data.state_dim": 8,
        "data.phase_b_pairing_verified": True,
        "data.phase_b_all_qr_codewords_represented": True,
        "data.probe_splits_disjoint": True,
        "measurement.probe_train_n": 2_048,
        "measurement.probe_eval_n": 4_096,
        "measurement.probe_ridge": 0.001,
        "measurement.truth_table_control_seed": 1_500_450_271,
        "measurement.phase_b_checkpoints": list(PHASE_B_STEPS),
        "measurement.direct_checkpoint_128": True,
        "training.batch_size": 250,
        "training.all_minibatches_full": True,
        "training.phase_b_steps": 1_024,
        "training.phase_b_examples_seen": 256_000,
    }
    for path, expected in summary_fixed.items():
        expect_equal(errors, run_id, get(run.summary, path), expected, f"summary.{path}")
    phase_a_steps = 0 if trajectory == "scratch" else 45
    phase_a_examples = 0 if trajectory == "scratch" else 11_250
    expect_equal(
        errors,
        run_id,
        get(run.summary, "training.phase_a_steps"),
        phase_a_steps,
        "summary.training.phase_a_steps",
    )
    expect_equal(
        errors,
        run_id,
        get(run.summary, "training.phase_a_examples_seen"),
        phase_a_examples,
        "summary.training.phase_a_examples_seen",
    )
    expect_equal(
        errors,
        run_id,
        get(run.summary, "training.total_examples_seen"),
        phase_a_examples + 256_000,
        "summary.training.total_examples_seen",
    )
    wall = get(run.summary, "training.wall_seconds")
    if not isinstance(wall, (int, float)) or not math.isfinite(float(wall)) or wall <= 0:
        errors.append(f"{run_id}: invalid wall time {wall!r}")

    phase_a = get(run.summary, "phase_a_snapshots", {})
    phase_b = get(run.summary, "phase_b_snapshots", {})
    wanted_a = {str(step) for step in PHASE_A_STEPS} if trajectory in HISTORICAL else set()
    if not isinstance(phase_a, Mapping) or set(phase_a) != wanted_a:
        errors.append(f"{run_id}: phase-A snapshot lattice differs from {sorted(wanted_a)}")
    if not isinstance(phase_b, Mapping) or set(phase_b) != {str(step) for step in PHASE_B_STEPS}:
        errors.append(f"{run_id}: phase-B snapshot lattice is not the frozen 24 checkpoints")
    if isinstance(phase_b, Mapping):
        if get(run.summary, "phase_b_local_zero") != phase_b.get("0"):
            errors.append(f"{run_id}: phase_b_local_zero is not the direct snapshot")
        if get(run.summary, "final") != phase_b.get("1024"):
            errors.append(f"{run_id}: final is not checkpoint 1024")

    historical = trajectory in HISTORICAL
    eligibility = get(run.summary, "eligibility.per_checkpoint", {})
    if historical:
        if not isinstance(eligibility, Mapping) or set(eligibility) != {"33", "45"}:
            errors.append(f"{run_id}: malformed historical eligibility checkpoints")
        expected_eligible = bool(
            isinstance(eligibility, Mapping) and all(bool(value) for value in eligibility.values())
        )
        expect_equal(
            errors,
            run_id,
            get(run.summary, "eligibility.eligible_both_registered_checkpoints"),
            expected_eligible,
            "summary eligibility endpoint",
        )
    else:
        expect_equal(errors, run_id, eligibility, {}, "nonhistorical eligibility")
        expect_equal(
            errors,
            run_id,
            get(run.summary, "eligibility.eligible_both_registered_checkpoints"),
            None,
            "nonhistorical eligibility endpoint",
        )

    model = get(run.summary, "model", {})
    if not isinstance(model, Mapping):
        errors.append(f"{run_id}: missing model report")
    else:
        for field, expected in {
            "update_mode": "full",
            "requested_budget": "full",
        }.items():
            expect_equal(errors, run_id, model.get(field), expected, f"model.{field}")
        for field in ("input_dim", "total_parameters", "trainable_parameters"):
            value = model.get(field)
            if not isinstance(value, int) or value <= 0:
                errors.append(f"{run_id}: model.{field} is invalid: {value!r}")

    for hash_path in (
        "hashes.phase_b_initial_model",
        "hashes.phase_b_initial_optimizer",
        "hashes.phase_b_final_model",
        "hashes.phase_b_final_optimizer",
    ):
        if not _hash_is_valid(get(run.summary, hash_path)):
            errors.append(f"{run_id}: malformed summary.{hash_path}")

    transition = get(run.summary, "optimizer_transition_audit", {})
    carry = trajectory.endswith("_carry")
    transition_expected = {
        "source": (
            "phase_a_carry" if carry else "fresh_scratch" if trajectory == "scratch" else "fresh_reset"
        ),
        "provided_optimizer": True,
        "low_level_reset_optimizer_flag": False,
        "semantic_reset_implemented_by_fresh_object": not carry,
        "optimized_parameter_count": 6,
        "param_group_count": 1,
        "state_entry_count": 6 if carry else 0,
        "adam_step_entry_count": 6 if carry else 0,
        "adam_step_min": 45 if carry else None,
        "adam_step_max": 45 if carry else None,
    }
    if not isinstance(transition, Mapping):
        errors.append(f"{run_id}: missing optimizer transition audit")
    else:
        for field, expected in transition_expected.items():
            expect_equal(
                errors,
                run_id,
                transition.get(field),
                expected,
                f"optimizer transition {field}",
            )
        expect_equal(
            errors,
            run_id,
            transition.get("actual_state_digest"),
            get(run.summary, "hashes.phase_b_initial_optimizer"),
            "optimizer transition digest",
        )
    for digest_path in (
        "data.phase_b_batch_digest",
        "data.phase_b_sampler_digest",
        "data.probe_train_digest",
        "data.probe_eval_digest",
    ):
        if not _hash_is_valid(get(run.summary, digest_path)):
            errors.append(f"{run_id}: malformed summary.{digest_path}")

    replay = get(run.summary, "replay", {})
    if not isinstance(replay, Mapping):
        errors.append(f"{run_id}: missing replay section")
    elif historical:
        overlap = trajectory.split("_", 1)[0]
        expect_equal(errors, run_id, replay.get("overlap"), overlap, "replay.overlap")
        expect_equal(errors, run_id, replay.get("samples_seen"), 11_250, "replay samples")
        expect_equal(errors, run_id, replay.get("optimizer_steps"), 45, "replay steps")
        checks = replay.get("checks")
        if (
            not isinstance(checks, Mapping)
            or set(checks)
            != {
                "initial_models_equal",
                "final_models_equal",
                "final_optimizers_equal",
                "samples_seen_equal",
                "optimizer_steps_equal",
                "registered_snapshot_steps_equal",
            }
            or not all(value is True for value in checks.values())
        ):
            errors.append(f"{run_id}: deterministic replay checks did not all pass")
        hashes = replay.get("hashes")
        required_hashes = {
            "initial_model",
            "replay_initial_model",
            "observed_final_model",
            "replay_final_model",
            "observed_final_optimizer",
            "replay_final_optimizer",
            "phase_a_batch",
            "phase_a_sampler",
            "phase_a_snapshot_metrics",
        }
        if not isinstance(hashes, Mapping) or set(hashes) != required_hashes:
            errors.append(f"{run_id}: historical replay hash set is malformed")
        elif not all(_hash_is_valid(value) for value in hashes.values()):
            errors.append(f"{run_id}: historical replay contains malformed hashes")
        else:
            for left, right in (
                ("initial_model", "replay_initial_model"),
                ("observed_final_model", "replay_final_model"),
                ("observed_final_optimizer", "replay_final_optimizer"),
            ):
                expect_equal(errors, run_id, hashes[left], hashes[right], f"replay {left}/{right}")
            expect_equal(
                errors,
                run_id,
                get(run.summary, "hashes.phase_b_initial_model"),
                hashes["observed_final_model"],
                "branch model hash",
            )
            expect_equal(
                errors,
                run_id,
                hashes["phase_a_snapshot_metrics"],
                stable_state_digest(
                    {
                        int(step): value
                        for step, value in cast(
                            Mapping[str, Any], get(run.summary, "phase_a_snapshots")
                        ).items()
                    }
                ),
                "phase-A snapshot digest",
            )
    elif trajectory == "scratch":
        expect_equal(errors, run_id, replay.get("overlap"), None, "scratch overlap")
        expect_equal(errors, run_id, replay.get("samples_seen"), 0, "scratch samples")
        expect_equal(errors, run_id, replay.get("optimizer_steps"), 0, "scratch steps")
        expect_equal(errors, run_id, replay.get("checks"), {"scratch_initialization": True}, "scratch check")
    else:
        expect_equal(errors, run_id, replay.get("overlap"), None, "sham overlap")
        expect_equal(errors, run_id, replay.get("samples_seen"), 11_250, "sham samples")
        expect_equal(errors, run_id, replay.get("optimizer_steps"), 45, "sham steps")
        expect_equal(errors, run_id, replay.get("checks"), {"sham_data_null": True}, "sham check")

    if trajectory == "sham":
        if not isinstance(get(run.summary, "data.sham"), Mapping):
            errors.append(f"{run_id}: sham arm lacks its data audit")
    elif get(run.summary, "data.sham") is not None:
        errors.append(f"{run_id}: non-sham arm contains a sham data audit")

    # Reconstruct every endpoint from snapshots; do not trust summary outcomes.
    if isinstance(phase_b, Mapping) and set(phase_b) == {str(step) for step in PHASE_B_STEPS}:
        event_input = {
            step: {
                "behavior": cast(Mapping[str, Any], phase_b[str(step)])["behavior"],
                "causal": cast(Mapping[str, Any], phase_b[str(step)])["causal"],
            }
            for step in PHASE_B_STEPS
        }
        for goal in GOALS:
            auc = normalized_control_auc(event_input, goal, horizon=AUC_HORIZON)
            event = persistent_handoff(event_input, goal, threshold=THRESHOLD, margin=MARGIN)
            expect_close(
                errors,
                run_id,
                get(run.summary, f"outcomes.normalized_control_auc_through_direct_checkpoint.{goal}"),
                auc,
                f"reconstructed {goal} AUC",
            )
            expect_equal(
                errors,
                run_id,
                get(run.summary, f"outcomes.persistent_control_events.{goal}"),
                event,
                f"reconstructed {goal} event",
            )
        expect_equal(
            errors,
            run_id,
            get(run.summary, "outcomes.auc_horizon"),
            AUC_HORIZON,
            "outcome AUC horizon",
        )

    # Long-form records must exactly reproduce selected compact snapshots.
    for phase, steps in (("phase_a", PHASE_A_STEPS if historical else ()), ("phase_b", PHASE_B_STEPS)):
        for suffix, split, interventions, metrics in (
            ("behavior", "factorial_eval", ("none",), ("rho_p", "rho_q", "rho_y_code", "target_accuracy")),
            ("causal", "factorial_eval", GOALS, ("causal_score", "causal_prob_score")),
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
                        wanted_global = step if phase == "phase_a" else 45 + step
                        wanted_examples = step * 250 if phase == "phase_a" else phase_a_examples + step * 250
                        if point.global_step != wanted_global or point.stage_step != step:
                            errors.append(f"{run_id}: {key} step/global counters are wrong at {step}")
                        if point.examples_seen != wanted_examples:
                            errors.append(f"{run_id}: {key} examples_seen is wrong at {step}")
                        if point.n != 4_096:
                            errors.append(f"{run_id}: {key} n={point.n}, expected 4096")
                        snapshot = run.snapshot(phase, step)
                        expected_value = _snapshot_metric_value(snapshot, key)
                        if not math.isclose(point.value, expected_value, rel_tol=0.0, abs_tol=1e-12):
                            errors.append(f"{run_id}: {key} metric/summary mismatch at {step}")

    optimization_stage = (
        None
        if trajectory == "scratch"
        else "phase_a_sham_optimization"
        if trajectory == "sham"
        else "phase_a_optimization"
    )
    if optimization_stage is not None:
        for metric in ("loss", "primary_loss", "train_batch_accuracy", "optimizer_steps"):
            key = (optimization_stage, "train_minibatch", "none", metric)
            _require_metric_steps(run, errors, key, PHASE_A_STEPS)
            for step, point in run.metrics.get(key, {}).items():
                if (
                    point.global_step != step
                    or point.stage_step != step
                    or point.examples_seen != step * 250
                    or point.n != 10_000
                ):
                    errors.append(f"{run_id}: malformed phase-A optimizer counters for {key} at {step}")
                if metric == "optimizer_steps" and point.value != step:
                    errors.append(f"{run_id}: wrong phase-A optimizer step value at {step}")
    for metric in ("loss", "primary_loss", "train_batch_accuracy", "optimizer_steps"):
        key = ("phase_b_optimization", "train_minibatch", "none", metric)
        _require_metric_steps(run, errors, key, PHASE_B_STEPS[1:])
        for step, point in run.metrics.get(key, {}).items():
            if (
                point.global_step != 45 + step
                or point.stage_step != step
                or point.examples_seen != phase_a_examples + step * 250
                or point.n != 10_000
            ):
                errors.append(f"{run_id}: malformed phase-B optimizer counters for {key} at {step}")
            if metric == "optimizer_steps" and point.value != step:
                errors.append(f"{run_id}: wrong phase-B optimizer step value at {step}")


def _reconstruct_seed_data(seed: int, config: Mapping[str, Any]) -> dict[str, Any]:
    phase_b = make_handoff_phase_b(
        10_000,
        q_q=0.90,
        k_q=2,
        k_y=3,
        seed=80_000_000 + seed,
        max_k_q=3,
        max_k_y=5,
        state_dim=8,
    )
    phase_b_audit = audit_handoff_phase_b(phase_b, expected_q_q=0.90)
    probe_train = make_competing_factorial_dataset(
        2_048,
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
        4_096,
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
    historical: dict[str, Any] = {}
    for overlap in ("independent", "nested"):
        bundle = make_competing_bundle(
            10_000,
            4_000,
            10_000,
            0.90,
            0.90,
            2,
            3,
            seed,
            max_k_q=3,
            max_k_y=5,
            overlap=cast(Any, overlap),
            state_dim=8,
        )
        historical[overlap] = {
            "batch_digest": semantic_batch_digest(bundle.train),
            "sampler_digest": static_sampler_digest(
                10_000, batch_size=250, steps=45, seed=seed, shuffle=True
            ),
            "training_overlap": {
                key: bundle.train.metadata[key]
                for key in (
                    "p_error_count",
                    "q_error_count",
                    "both_error_count",
                    "p_only_error_count",
                    "q_only_error_count",
                    "error_phi",
                )
            },
        }
    sham = make_compute_sham(
        n=10_000,
        repeats=79,
        k_q=2,
        k_y=3,
        seed=81_000_000 + seed,
        control_seed=1_500_450_271,
        max_k_q=3,
        max_k_y=5,
        state_dim=8,
    )
    sham_audit = audit_compute_sham(sham)
    initial_model, model_report = build_model(phase_b, config, seed)
    return {
        "phase_b": phase_b_audit,
        "phase_b_sampler": static_sampler_digest(
            10_000,
            batch_size=250,
            steps=1_024,
            seed=90_000_000 + seed,
            shuffle=True,
        ),
        "probe_train_digest": semantic_batch_digest(probe_train),
        "probe_eval_digest": semantic_batch_digest(probe_eval),
        "historical": historical,
        "sham": sham_audit,
        "sham_sampler": static_sampler_digest(
            10_000,
            batch_size=250,
            steps=45,
            seed=81_500_000 + seed,
            shuffle=True,
        ),
        "initial_model_hash": stable_state_digest(initial_model.state_dict()),
        "model_report": model_report,
    }


def _cross_run_audit(runs: Sequence[Run], errors: list[str]) -> dict[str, Any]:
    by_seed: defaultdict[int, dict[str, Run]] = defaultdict(dict)
    reconstruction_rows: list[dict[str, Any]] = []
    for run in runs:
        by_seed[run.seed][run.trajectory] = run
    for seed, arms in sorted(by_seed.items()):
        exemplar = arms[TRAJECTORIES[0]]
        rebuilt = _reconstruct_seed_data(seed, exemplar.config)
        common_fields = {
            "data.phase_b": rebuilt["phase_b"],
            "data.phase_b_batch_digest": rebuilt["phase_b"]["batch_digest"],
            "data.phase_b_sampler_digest": rebuilt["phase_b_sampler"],
            "data.probe_train_digest": rebuilt["probe_train_digest"],
            "data.probe_eval_digest": rebuilt["probe_eval_digest"],
            "model": rebuilt["model_report"],
        }
        for trajectory, run in arms.items():
            for path, expected in common_fields.items():
                expect_equal(
                    errors,
                    run.path.name,
                    get(run.summary, path),
                    _json_safe(expected),
                    f"reconstructed {path}",
                )
            expect_equal(
                errors,
                run.path.name,
                get(run.summary, "replay.hashes.initial_model"),
                rebuilt["initial_model_hash"],
                "reconstructed initial model",
            )
            if trajectory == "scratch":
                expect_equal(
                    errors,
                    run.path.name,
                    get(run.summary, "hashes.phase_b_initial_model"),
                    rebuilt["initial_model_hash"],
                    "scratch initialization hash",
                )
            if trajectory == "sham":
                expect_equal(
                    errors,
                    run.path.name,
                    get(run.summary, "data.sham"),
                    _json_safe(rebuilt["sham"]),
                    "reconstructed sham audit",
                )
                expect_equal(
                    errors,
                    run.path.name,
                    get(run.summary, "replay.hashes.phase_a_batch"),
                    rebuilt["sham"]["batch_digest"],
                    "reconstructed sham batch digest",
                )
                expect_equal(
                    errors,
                    run.path.name,
                    get(run.summary, "replay.hashes.phase_a_sampler"),
                    rebuilt["sham_sampler"],
                    "reconstructed sham sampler digest",
                )
            if trajectory in HISTORICAL:
                overlap = trajectory.split("_", 1)[0]
                expected = rebuilt["historical"][overlap]
                expect_equal(
                    errors,
                    run.path.name,
                    get(run.summary, "replay.hashes.phase_a_batch"),
                    expected["batch_digest"],
                    "reconstructed historical batch digest",
                )
                expect_equal(
                    errors,
                    run.path.name,
                    get(run.summary, "replay.hashes.phase_a_sampler"),
                    expected["sampler_digest"],
                    "reconstructed historical sampler digest",
                )
                expect_equal(
                    errors,
                    run.path.name,
                    get(run.summary, "replay.training_overlap"),
                    _json_safe(expected["training_overlap"]),
                    "reconstructed historical overlap counts",
                )
            reconstruction_rows.append(
                {
                    "seed": seed,
                    "trajectory": trajectory,
                    "phase_b_batch_digest": get(run.summary, "data.phase_b_batch_digest"),
                    "phase_b_sampler_digest": get(run.summary, "data.phase_b_sampler_digest"),
                    "probe_train_digest": get(run.summary, "data.probe_train_digest"),
                    "probe_eval_digest": get(run.summary, "data.probe_eval_digest"),
                    "reconstruction_passed": True,
                }
            )

        common_paths = (
            "data.phase_b_batch_digest",
            "data.phase_b_sampler_digest",
            "data.probe_train_digest",
            "data.probe_eval_digest",
        )
        for path in common_paths:
            values = {get(run.summary, path) for run in arms.values()}
            if len(values) != 1:
                errors.append(f"seed {seed}: {path} is not common across six arms")
        initial_hashes = {get(run.summary, "replay.hashes.initial_model") for run in arms.values()}
        if len(initial_hashes) != 1:
            errors.append(f"seed {seed}: initial model differs across six arms")

        for overlap in ("independent", "nested"):
            carry = arms[f"{overlap}_carry"]
            reset = arms[f"{overlap}_reset"]
            for path in (
                "phase_a_snapshots",
                "eligibility",
                "replay",
                "hashes.phase_b_initial_model",
                "phase_b_local_zero",
            ):
                if get(carry.summary, path) != get(reset.summary, path):
                    errors.append(f"seed {seed}: {overlap} carry/reset differ at {path}")
            carry_optimizer = get(carry.summary, "hashes.phase_b_initial_optimizer")
            reset_optimizer = get(reset.summary, "hashes.phase_b_initial_optimizer")
            if carry_optimizer == reset_optimizer:
                errors.append(f"seed {seed}: {overlap} carry/reset optimizer hashes coincide")

        # All model initializations are the same before their phase-A histories.
        scratch_initial = get(arms["scratch"].summary, "hashes.phase_b_initial_model")
        sham_initial = get(arms["sham"].summary, "replay.hashes.initial_model")
        if scratch_initial != sham_initial:
            errors.append(f"seed {seed}: scratch and sham raw initializations differ")

    return {
        "seed_blocks": len(by_seed),
        "trajectories_per_seed": len(TRAJECTORIES),
        "reconstructed_artifacts": len(reconstruction_rows),
        "reconstruction_rows": reconstruction_rows,
        "phase_b_common_within_seed": True,
        "prefix_carry_reset_equality": True,
        "current_source_regeneration_passed": True,
    }


def load_runs(root: Path, *, mode: str) -> tuple[list[Run], dict[str, Any]]:
    directory = root / "h14" / "winner_knockout_handoff"
    if not directory.is_dir():
        raise RuntimeError(f"E17 {mode} audit: missing artifact directory {directory}")
    config_paths = sorted(directory.glob("*/resolved_config.yaml"))
    if not config_paths:
        raise RuntimeError(f"E17 {mode} audit: no materialized runs in {directory}")
    if mode == "full" and len(config_paths) != len(SEEDS) * len(TRAJECTORIES):
        raise RuntimeError(
            f"E17 full hard audit: {len(config_paths)}/120 materialized runs; full interpretation is disabled"
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
        if (run_dir / "COMPLETE").read_text(encoding="utf-8") != "complete\n":
            errors.append(f"{run_dir.name}: invalid COMPLETE marker")
        config = load_yaml(config_path)
        summary = load_json(run_dir / "summary.json")
        metadata = load_json(run_dir / "metadata.json")
        status = load_json(run_dir / "status.json")
        states[str(status.get("state"))] += 1
        seed = int(summary.get("seed", -1))
        trajectory = str(summary.get("trajectory", ""))
        if status != {"state": "complete", "run_id": run_dir.name}:
            errors.append(f"{run_dir.name}: invalid completion status")
        if metadata.get("run_id") != run_dir.name or metadata.get("seed") != seed:
            errors.append(f"{run_dir.name}: metadata identity mismatch")
        if expected_artifact_run_id(config, metadata) != run_dir.name:
            errors.append(f"{run_dir.name}: artifact identity does not reconstruct")
        metrics, line_count = _load_metric_index(
            run_dir / "metrics.jsonl",
            run_id=run_dir.name,
            seed=seed,
            trajectory=trajectory,
            errors=errors,
        )
        run = Run(run_dir, config, summary, metadata, metrics, line_count)
        if seed not in SEEDS:
            errors.append(f"{run_dir.name}: unregistered seed {seed}")
        if trajectory not in TRAJECTORIES:
            errors.append(f"{run_dir.name}: unregistered trajectory {trajectory}")
        else:
            validate_run(run, errors)
        runs.append(run)

    keys = [(run.seed, run.trajectory) for run in runs]
    duplicates = [key for key, count in Counter(keys).items() if count > 1]
    if duplicates:
        errors.append(f"duplicate seed/trajectory keys: {duplicates}")
    realized_seeds = tuple(sorted({run.seed for run in runs}))
    expected_keys = {(seed, trajectory) for seed in realized_seeds for trajectory in TRAJECTORIES}
    missing_keys = expected_keys - set(keys)
    unexpected_keys = set(keys) - {(seed, trajectory) for seed in SEEDS for trajectory in TRAJECTORIES}
    if missing_keys or unexpected_keys:
        errors.append(
            f"grid has {len(missing_keys)} missing within-seed arms and "
            f"{len(unexpected_keys)} unexpected keys"
        )
    if mode == "full" and set(keys) != {(seed, trajectory) for seed in SEEDS for trajectory in TRAJECTORIES}:
        errors.append("full grid is not exactly the frozen 20 x 6 design")

    fingerprints = {get(run.metadata, "implementation.implementation_fingerprint") for run in runs}
    if len(fingerprints) != 1 or not all(_hash_is_valid(value) for value in fingerprints):
        errors.append(f"artifact implementation fingerprints are invalid: {fingerprints}")
    current = implementation_provenance(REPO)
    current_fingerprint = current["implementation_fingerprint"]
    if fingerprints != {current_fingerprint}:
        errors.append(
            "artifact fingerprint does not match current ForkWorld source; "
            "refusing current-source reconstruction"
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
            "fingerprint schema version",
        )

    # Require exact record counts within trajectory classes.  This is stronger
    # than merely finding the selected metrics and catches truncated JSONL tails.
    record_counts: dict[str, set[int]] = defaultdict(set)
    for run in runs:
        record_counts[run.trajectory].add(run.metric_line_count)
    for trajectory, counts in record_counts.items():
        expected_count = {
            "independent_carry": 6_397,
            "independent_reset": 6_397,
            "nested_carry": 6_397,
            "nested_reset": 6_397,
            "scratch": 5_905,
            "sham": 5_913,
        }[trajectory]
        if counts != {expected_count}:
            errors.append(f"{trajectory}: metric line counts {sorted(counts)}, expected {expected_count}")

    if errors:
        _raise_audit(f"E17 {mode} artifact audit", errors)
    cross_errors: list[str] = []
    cross = _cross_run_audit(runs, cross_errors)
    if cross_errors:
        _raise_audit(f"E17 {mode} data/hash cross-audit", cross_errors)
    audit = {
        "mode": mode,
        "artifacts_root": str(root.resolve()),
        "materialized_runs": len(config_paths),
        "complete_runs": states.get("complete", 0),
        "expected_runs": 120 if mode == "full" else len(realized_seeds) * 6,
        "seeds": list(realized_seeds),
        "trajectories": list(TRAJECTORIES),
        "metric_records": sum(run.metric_line_count for run in runs),
        "metric_line_counts_by_trajectory": {key: sorted(value) for key, value in record_counts.items()},
        "implementation_fingerprint": current_fingerprint,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "source_fingerprint_schema_version": SOURCE_FINGERPRINT_SCHEMA_VERSION,
        "strict_grid_config_metric_digest_and_reconstruction_audit_passed": True,
        **{key: value for key, value in cross.items() if key != "reconstruction_rows"},
    }
    return sorted(runs, key=lambda run: (run.seed, TRAJECTORIES.index(run.trajectory))), {
        **audit,
        "reconstruction_rows": cross["reconstruction_rows"],
    }


def _snapshot_values(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "target_accuracy": float(snapshot["target_accuracy"]),
        "boolean_signature": str(snapshot["boolean_signature"]),
        "boolean_signature_int": int(get(snapshot, "truth_table.boolean_signature_int")),
        "tuple_consistency": float(get(snapshot, "truth_table.tuple_consistency")),
    }
    for goal in GOALS:
        result[f"behavior_{goal}"] = float(get(snapshot, f"behavior.{goal}"))
        result[f"causal_{goal}"] = float(get(snapshot, f"causal.{goal}"))
        result[f"causal_probability_{goal}"] = float(get(snapshot, f"causal_probability.{goal}"))
        result[f"probe_final_hidden_{goal}"] = float(
            get(snapshot, f"probe_heldout_accuracy.final_hidden.{goal}")
        )
        result[f"probe_first_hidden_{goal}"] = float(
            get(snapshot, f"probe_heldout_accuracy.first_hidden.{goal}")
        )
        result[f"probe_selective_final_hidden_{goal}"] = float(
            get(snapshot, f"selective_final_hidden_probe.{goal}")
        )
        result[f"control_{goal}"] = 0.5 * (result[f"behavior_{goal}"] + result[f"causal_{goal}"])
        result[f"pure_{goal}"] = pure_control(
            cast(Mapping[str, float], snapshot["behavior"]),
            cast(Mapping[str, float], snapshot["causal"]),
            goal,
            threshold=THRESHOLD,
            margin=MARGIN,
        )
    pure = [goal for goal in GOALS if result[f"pure_{goal}"]]
    result["control_taxonomy"] = pure[0] if len(pure) == 1 else "mixed_or_none"
    return result


def eligibility_rows(runs: Sequence[Run]) -> tuple[list[dict[str, Any]], tuple[int, ...]]:
    indexed = {(run.seed, run.trajectory): run for run in runs}
    rows: list[dict[str, Any]] = []
    intersection: list[int] = []
    for seed in sorted({run.seed for run in runs}):
        independent = indexed[(seed, "independent_reset")]
        nested = indexed[(seed, "nested_reset")]
        independent_eligible = bool(
            get(independent.summary, "eligibility.eligible_both_registered_checkpoints")
        )
        nested_eligible = bool(get(nested.summary, "eligibility.eligible_both_registered_checkpoints"))
        paired = independent_eligible and nested_eligible
        if paired:
            intersection.append(seed)
        for step in PHASE_A_STEPS:
            i_snapshot = independent.snapshot("phase_a", step)
            n_snapshot = nested.snapshot("phase_a", step)
            i_values = _snapshot_values(i_snapshot)
            n_values = _snapshot_values(n_snapshot)
            rows.append(
                {
                    "seed": seed,
                    "step": step,
                    "independent_eligible": independent_eligible,
                    "nested_eligible": nested_eligible,
                    "paired_intersection_eligible": paired,
                    "independent_P_behavior": i_values["behavior_P"],
                    "nested_P_behavior": n_values["behavior_P"],
                    "independent_P_causal": i_values["causal_P"],
                    "nested_P_causal": n_values["causal_P"],
                    "independent_Q_selective_probe": i_values["probe_selective_final_hidden_Q"],
                    "nested_Q_selective_probe": n_values["probe_selective_final_hidden_Q"],
                    "independent_Y_probe": i_values["probe_final_hidden_Y"],
                    "nested_Y_probe": n_values["probe_final_hidden_Y"],
                }
            )
    return rows, tuple(intersection)


def checkpoint_rows(runs: Sequence[Run], eligible_seeds: Sequence[int]) -> list[dict[str, Any]]:
    eligible = set(eligible_seeds)
    rows: list[dict[str, Any]] = []
    for run in runs:
        for step in PHASE_B_STEPS:
            row = {
                "seed": run.seed,
                "trajectory": run.trajectory,
                "local_step": step,
                "global_step": 45 + step,
                "eligible_intersection": run.seed in eligible,
            }
            row.update(_snapshot_values(run.snapshot("phase_b", step)))
            rows.append(row)
    return rows


def _restricted_time(event: Mapping[str, Any]) -> float:
    if bool(event["observed"]):
        return float(event["confirmation_step"])
    return float(CENSOR_HORIZON)


def run_endpoint_rows(runs: Sequence[Run], eligible_seeds: Sequence[int]) -> list[dict[str, Any]]:
    eligible = set(eligible_seeds)
    rows: list[dict[str, Any]] = []
    for run in runs:
        zero = _snapshot_values(run.snapshot("phase_b", 0))
        final = _snapshot_values(run.snapshot("phase_b", CENSOR_HORIZON))
        row: dict[str, Any] = {
            "seed": run.seed,
            "trajectory": run.trajectory,
            "eligible_intersection": run.seed in eligible,
            "first_non_p_goal": get(run.summary, "outcomes.first_non_p_controlled_goal.goal"),
            "first_non_p_confirmation_step": get(
                run.summary, "outcomes.first_non_p_controlled_goal.confirmation_step"
            ),
            "final_control_taxonomy": final["control_taxonomy"],
            "final_boolean_signature": final["boolean_signature"],
            "final_boolean_signature_int": final["boolean_signature_int"],
            "final_target_accuracy": final["target_accuracy"],
        }
        for goal in GOALS:
            row[f"auc128_{goal}"] = float(
                get(
                    run.summary,
                    f"outcomes.normalized_control_auc_through_direct_checkpoint.{goal}",
                )
            )
            event = cast(
                Mapping[str, Any],
                get(run.summary, f"outcomes.persistent_control_events.{goal}"),
            )
            row[f"handoff_{goal}_observed"] = bool(event["observed"])
            row[f"handoff_{goal}_first_qualifying"] = event["first_qualifying_step"]
            row[f"handoff_{goal}_confirmation"] = event["confirmation_step"]
            row[f"restricted_time_{goal}"] = _restricted_time(event)
            for prefix, values in (("zero", zero), ("final", final)):
                for family in (
                    "behavior",
                    "causal",
                    "probe_final_hidden",
                    "probe_selective_final_hidden",
                    "control",
                ):
                    row[f"{prefix}_{family}_{goal}"] = values[f"{family}_{goal}"]
        rows.append(row)
    return rows


def _bootstrap_paired(differences: NDArray[np.float64], *, key: str) -> dict[str, float | int]:
    if differences.ndim != 1 or len(differences) < 1 or not np.all(np.isfinite(differences)):
        raise ValueError(f"invalid paired differences for {key}")
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8, person=b"forke17")
    rng = np.random.default_rng(int.from_bytes(digest.digest(), "little"))
    indices = rng.integers(0, len(differences), size=(BOOTSTRAP_DRAWS, len(differences)))
    draws = np.mean(differences[indices], axis=1)
    low, high = np.quantile(draws, (0.025, 0.975))
    return {
        "estimate": float(np.mean(differences)),
        "ci_low": float(low),
        "ci_high": float(high),
        "n_seeds": len(differences),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
    }


def _paired_values(
    endpoints: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    left: str,
    right: str,
    field: str,
) -> NDArray[np.float64]:
    index = {(int(row["seed"]), str(row["trajectory"])): row for row in endpoints}
    return np.asarray(
        [float(index[(seed, left)][field]) - float(index[(seed, right)][field]) for seed in seeds],
        dtype=np.float64,
    )


CONTRASTS = (
    ("independent_reset_minus_nested_reset", "independent_reset", "nested_reset"),
    ("independent_reset_minus_scratch", "independent_reset", "scratch"),
    ("independent_reset_minus_sham", "independent_reset", "sham"),
    ("independent_carry_minus_independent_reset", "independent_carry", "independent_reset"),
    ("nested_carry_minus_nested_reset", "nested_carry", "nested_reset"),
)


def paired_contrast_rows(
    endpoints: Sequence[Mapping[str, Any]], eligible_seeds: Sequence[int]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, left, right in CONTRASTS:
        for outcome, field in (
            ("Q_control_auc_0_128", "auc128_Q"),
            ("Q_restricted_handoff_time_0_1024", "restricted_time_Q"),
            ("Y_restricted_handoff_time_0_1024", "restricted_time_Y"),
            ("final_target_accuracy", "final_target_accuracy"),
        ):
            result = _bootstrap_paired(
                _paired_values(endpoints, eligible_seeds, left, right, field),
                key=f"{name}:{outcome}",
            )
            rows.append(
                {
                    "contrast": name,
                    "left": left,
                    "right": right,
                    "outcome": outcome,
                    **result,
                }
            )
    return rows


def manipulation_check_rows(
    runs: Sequence[Run], seeds: Sequence[int]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    index = {(run.seed, run.trajectory): run for run in runs}
    measures = (
        ("Q_selective_final_hidden_probe", "probe_selective_final_hidden_Q"),
        ("Q_raw_final_hidden_probe", "probe_final_hidden_Q"),
        ("P_behavior", "behavior_P"),
        ("P_causal", "causal_P"),
        ("Y_final_hidden_probe", "probe_final_hidden_Y"),
        ("Y_selective_final_hidden_probe", "probe_selective_final_hidden_Y"),
    )
    rows: list[dict[str, Any]] = []
    for label, field in measures:
        differences = []
        for seed in seeds:
            independent = _snapshot_values(index[(seed, "independent_reset")].snapshot("phase_a", 45))
            nested = _snapshot_values(index[(seed, "nested_reset")].snapshot("phase_a", 45))
            differences.append(float(independent[field]) - float(nested[field]))
        result = _bootstrap_paired(np.asarray(differences, dtype=np.float64), key=f"manipulation:{label}")
        rows.append({"measure": label, "contrast": "independent_minus_nested", **result})
    by_measure = {str(row["measure"]): row for row in rows}
    q = by_measure["Q_selective_final_hidden_probe"]
    q_pass = float(q["estimate"]) >= 0.10 and float(q["ci_low"]) > 0.0
    specificity_measures = ("P_behavior", "P_causal", "Y_final_hidden_probe")
    specificity = all(
        float(by_measure[name]["ci_low"]) >= -0.05 and float(by_measure[name]["ci_high"]) <= 0.05
        for name in specificity_measures
    )
    decision = {
        "q_accessibility_manipulation_passed": q_pass,
        "q_specificity_intervals_inside_plus_minus_0_05": specificity,
        "q_specific_readiness_language_allowed": q_pass and specificity,
        "manipulation_threshold": 0.10,
        "specificity_equivalence_margin": 0.05,
        "uses_all_registered_seeds_without_eligibility_selection": True,
    }
    return rows, decision


def starting_state_rows(endpoints: Sequence[Mapping[str, Any]], seeds: Sequence[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    comparisons = CONTRASTS[:3]
    fields = tuple(
        (f"{family}_{goal}", f"zero_{family}_{goal}")
        for goal in GOALS
        for family in ("behavior", "causal", "probe_final_hidden", "probe_selective_final_hidden")
    )
    for contrast, left, right in comparisons:
        for measure, field in fields:
            result = _bootstrap_paired(
                _paired_values(endpoints, seeds, left, right, field),
                key=f"starting:{contrast}:{measure}",
            )
            rows.append(
                {
                    "contrast": contrast,
                    "left": left,
                    "right": right,
                    "measure": measure,
                    **result,
                }
            )
    return rows


def taxonomy_rows(
    endpoints: Sequence[Mapping[str, Any]], eligible_seeds: Sequence[int]
) -> list[dict[str, Any]]:
    eligible = set(eligible_seeds)
    counts: Counter[tuple[str, str, str, Any]] = Counter()
    for row in endpoints:
        if int(row["seed"]) not in eligible:
            continue
        counts[
            (
                str(row["trajectory"]),
                str(row["final_control_taxonomy"]),
                str(row["final_boolean_signature"]),
                row["first_non_p_goal"],
            )
        ] += 1
    return [
        {
            "trajectory": trajectory,
            "final_control_taxonomy": taxonomy,
            "final_boolean_signature": signature,
            "first_non_p_goal": first_goal,
            "count": count,
            "n_eligible_seeds": len(eligible_seeds),
            "fraction": count / len(eligible_seeds),
        }
        for (trajectory, taxonomy, signature, first_goal), count in sorted(
            counts.items(), key=lambda item: (TRAJECTORIES.index(item[0][0]), item[0][1:])
        )
    ]


def interpretation_decisions(
    contrasts: Sequence[Mapping[str, Any]], manipulation: Mapping[str, Any]
) -> dict[str, Any]:
    index = {(str(row["contrast"]), str(row["outcome"])): row for row in contrasts}
    primary_names = tuple(name for name, _, _ in CONTRASTS[:3])
    readiness_details: dict[str, Any] = {}
    for name in primary_names:
        auc = index[(name, "Q_control_auc_0_128")]
        time = index[(name, "Q_restricted_handoff_time_0_1024")]
        readiness_details[name] = {
            "auc_ci_above_zero": float(auc["ci_low"]) > 0.0,
            "restricted_time_direction_matches": float(time["estimate"]) < 0.0,
            "auc_practically_equivalent": (float(auc["ci_low"]) >= -0.05 and float(auc["ci_high"]) <= 0.05),
            "restricted_time_practically_equivalent": (
                float(time["ci_low"]) >= -16.0 and float(time["ci_high"]) <= 16.0
            ),
        }
    history_readiness = all(
        value["auc_ci_above_zero"] and value["restricted_time_direction_matches"]
        for value in readiness_details.values()
    )
    optimizer: dict[str, Any] = {}
    for name in tuple(item[0] for item in CONTRASTS[3:]):
        auc = index[(name, "Q_control_auc_0_128")]
        estimate = float(auc["estimate"])
        excludes_zero = float(auc["ci_low"]) > 0.0 or float(auc["ci_high"]) < 0.0
        optimizer[name] = {
            "material": abs(estimate) >= 0.05 and excludes_zero,
            "estimate": estimate,
            "ci_low": float(auc["ci_low"]),
            "ci_high": float(auc["ci_high"]),
        }
    stop_reason: str | None = None
    if all(
        value["auc_practically_equivalent"] and value["restricted_time_practically_equivalent"]
        for value in readiness_details.values()
    ):
        stop_reason = "reset-independent is practically equivalent to all three controls"
    elif history_readiness and not any(value["material"] for value in optimizer.values()):
        stop_reason = "readiness is carried by weights/representations, not optimizer moments"
    return {
        "history_specific_readiness_criterion_met": history_readiness,
        "q_specific_readiness_criterion_met": history_readiness
        and bool(manipulation["q_specific_readiness_language_allowed"]),
        "primary_comparisons": readiness_details,
        "optimizer_history": optimizer,
        "frozen_line_stops": stop_reason is not None,
        "stop_reason": stop_reason,
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def plot_font_family() -> str:
    directories = [Path.home() / "Library" / "Fonts", Path("/Library/Fonts")]
    configured = os.environ.get("FORKWORLD_FONT_DIR")
    if configured:
        directories.insert(0, Path(configured).expanduser())
    names: list[str] = []
    for directory in directories:
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if (
                path.is_file()
                and "myriad" in path.name.lower()
                and path.suffix.lower() in {".otf", ".ttf", ".ttc"}
            ):
                try:
                    fm.fontManager.addfont(str(path))
                    names.append(fm.FontProperties(fname=str(path)).get_name())
                except RuntimeError:
                    pass
    if names:
        exact = [name for name in names if name.lower() == "myriad pro"]
        return exact[0] if exact else names[0]
    discovered = sorted({font.name for font in fm.fontManager.ttflist if "myriad" in font.name.lower()})
    return discovered[0] if discovered else "DejaVu Sans"


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
            "legend.fontsize": 8.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _panel_label(axis: Axes, label: str) -> None:
    axis.text(
        -0.105,
        1.055,
        label,
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=11.0,
        fontweight=600,
        color=INK,
        clip_on=False,
    )


def _mean_interval(values: NDArray[np.float64]) -> tuple[float, float, float]:
    mean = float(np.mean(values))
    if len(values) == 1:
        return mean, mean, mean
    rng = np.random.default_rng(1_701_701 + values.shape[-1])
    indices = rng.integers(0, len(values), size=(2_000, len(values)))
    draws = np.mean(values[indices], axis=1)
    low, high = np.quantile(draws, (0.025, 0.975))
    return mean, float(low), float(high)


def make_figure(
    checkpoints: Sequence[Mapping[str, Any]],
    endpoints: Sequence[Mapping[str, Any]],
    contrasts: Sequence[Mapping[str, Any]],
    manipulation_rows: Sequence[Mapping[str, Any]],
    eligible_seeds: Sequence[int],
) -> None:
    configure_style()
    eligible = set(eligible_seeds)
    figure, axes_grid = plt.subplots(2, 2, figsize=(11.4, 8.1))
    axes = list(axes_grid.ravel())
    figure.subplots_adjust(left=0.105, right=0.97, bottom=0.10, top=0.79, wspace=0.38, hspace=0.48)

    colors = {
        "independent_carry": BLUE,
        "independent_reset": BLUE,
        "nested_carry": ORANGE,
        "nested_reset": ORANGE,
        "scratch": GRAY,
        "sham": PURPLE,
    }
    linestyles = {
        "independent_carry": "--",
        "independent_reset": "-",
        "nested_carry": "--",
        "nested_reset": "-",
        "scratch": ":",
        "sham": "-.",
    }
    labels = {
        "independent_carry": "Independent history · carry AdamW",
        "independent_reset": "Independent history · reset AdamW",
        "nested_carry": "Nested history · carry AdamW",
        "nested_reset": "Nested history · reset AdamW",
        "scratch": "Scratch",
        "sham": "Compute sham",
    }

    axis_a = axes[0]
    for trajectory in TRAJECTORIES:
        means: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        for step in PHASE_B_STEPS:
            values = np.asarray(
                [
                    float(row["control_Q"])
                    for row in checkpoints
                    if row["trajectory"] == trajectory
                    and int(row["local_step"]) == step
                    and int(row["seed"]) in eligible
                ],
                dtype=np.float64,
            )
            mean, low, high = _mean_interval(values)
            means.append(mean)
            lows.append(low)
            highs.append(high)
        axis_a.plot(
            PHASE_B_STEPS,
            means,
            color=colors[trajectory],
            ls=linestyles[trajectory],
            lw=1.75,
            label=labels[trajectory],
        )
        axis_a.fill_between(PHASE_B_STEPS, lows, highs, color=colors[trajectory], alpha=0.075, lw=0)
    axis_a.axhline(0.9, color="#AAB2BD", lw=0.8, ls=(0, (2, 3)))
    axis_a.set_xscale("symlog", linthresh=1.0)
    axis_a.set_xticks((0, 1, 3, 9, 33, 128, 1024))
    axis_a.set_xticklabels(("0", "1", "3", "9", "33", "128", "1,024"))
    axis_a.set_ylim(0.43, 1.02)
    axis_a.set_xlabel("Phase-B optimizer update")
    axis_a.set_ylabel("Mean Q behavior and causal control")
    axis_a.grid(color=LIGHT_GRAY, lw=0.65)
    axis_a.set_title("Q takes control transiently before yielding to the exact goal", loc="left", pad=8)

    axis_b = axes[1]
    forest_names = [name for name, _, _ in CONTRASTS]
    contrast_index = {(str(row["contrast"]), str(row["outcome"])): row for row in contrasts}
    display = (
        "Indep. reset - nested reset",
        "Indep. reset - scratch",
        "Indep. reset - sham",
        "Indep. carry - reset",
        "Nested carry - reset",
    )
    y = np.arange(len(forest_names))[::-1]
    for position, name in zip(y, forest_names, strict=True):
        row = contrast_index[(name, "Q_control_auc_0_128")]
        estimate = float(row["estimate"])
        axis_b.errorbar(
            estimate,
            position,
            xerr=np.asarray([[estimate - float(row["ci_low"])], [float(row["ci_high"]) - estimate]]),
            fmt="o",
            color=BLUE if position >= 2 else GOLD,
            ecolor=BLUE if position >= 2 else GOLD,
            capsize=2.5,
            ms=4.8,
            lw=1.3,
        )
    axis_b.axvspan(-0.05, 0.05, color=LIGHT_GRAY, alpha=0.75, zorder=-2)
    axis_b.axvline(0, color="#98A2B3", lw=0.8)
    axis_b.set_yticks(y, display)
    axis_b.set_xlabel("Paired difference in Q-control AUC, updates 0-128")
    axis_b.grid(axis="x", color=LIGHT_GRAY, lw=0.65)
    axis_b.set_title("Frozen seed-paired contrasts", loc="left", pad=8)

    axis_c = axes[2]
    endpoint_index = {(int(row["seed"]), str(row["trajectory"])): row for row in endpoints}
    for trajectory in TRAJECTORIES:
        survival = []
        for step in PHASE_B_STEPS:
            still_waiting = []
            for seed in eligible_seeds:
                row = endpoint_index[(seed, trajectory)]
                confirmation = row["handoff_Q_confirmation"]
                still_waiting.append(confirmation is None or int(confirmation) > step)
            survival.append(float(np.mean(still_waiting)))
        axis_c.step(
            PHASE_B_STEPS,
            survival,
            where="post",
            color=colors[trajectory],
            ls=linestyles[trajectory],
            lw=1.75,
        )
    axis_c.set_xscale("symlog", linthresh=1.0)
    axis_c.set_xticks((0, 1, 3, 9, 33, 128, 1024))
    axis_c.set_xticklabels(("0", "1", "3", "9", "33", "128", "1,024"))
    axis_c.set_ylim(-0.03, 1.03)
    axis_c.set_xlabel("Phase-B optimizer update")
    axis_c.set_ylabel("Fraction without confirmed Q handoff")
    axis_c.grid(color=LIGHT_GRAY, lw=0.65)
    axis_c.set_title("Persistent handoff timing and right-censoring", loc="left", pad=8)

    axis_d = axes[3]
    manipulation_index = {str(row["measure"]): row for row in manipulation_rows}
    manipulation_names = (
        "Q_selective_final_hidden_probe",
        "P_behavior",
        "P_causal",
        "Y_final_hidden_probe",
    )
    manipulation_display = (
        "Selective Q probe",
        "P behavior",
        "P causal control",
        "Y probe",
    )
    y_d = np.arange(len(manipulation_names))[::-1]
    axis_d.axvspan(-0.05, 0.05, color=LIGHT_GRAY, alpha=0.75, zorder=-2)
    axis_d.axvline(0, color="#98A2B3", lw=0.8)
    axis_d.axvline(0.10, color=BLUE, lw=0.8, ls=(0, (2, 3)))
    for position, name in zip(y_d, manipulation_names, strict=True):
        row = manipulation_index[name]
        estimate = float(row["estimate"])
        axis_d.errorbar(
            estimate,
            position,
            xerr=np.asarray([[estimate - float(row["ci_low"])], [float(row["ci_high"]) - estimate]]),
            fmt="o",
            color=TEAL if name.startswith("Q_") else INK,
            ecolor=TEAL if name.startswith("Q_") else INK,
            capsize=2.5,
            ms=4.8,
            lw=1.3,
        )
    axis_d.set_yticks(y_d, manipulation_display)
    axis_d.set_xlabel("Independent - nested at phase-A update 45")
    axis_d.grid(axis="x", color=LIGHT_GRAY, lw=0.65)
    axis_d.set_title("History manipulation and specificity checks", loc="left", pad=8)

    for label, axis in zip("abcd", axes, strict=True):
        _panel_label(axis, label)
    handles = [
        mpl.lines.Line2D([], [], color=colors[name], ls=linestyles[name], lw=1.8, label=labels[name])
        for name in TRAJECTORIES
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.887),
        ncol=3,
        columnspacing=1.7,
        handlelength=2.6,
    )
    figure.suptitle(
        "Representation history predicts handoff after winner knockout",
        x=0.105,
        y=0.977,
        ha="left",
        fontsize=13.2,
        fontweight=600,
        color=INK,
    )
    figure.text(
        0.105,
        0.939,
        f"Winner-knockout handoff · {len(eligible_seeds)} paired eligible seeds · shaded bands and intervals resample seeds",
        ha="left",
        va="center",
        fontsize=8.5,
        color=GRAY,
    )
    figure.text(
        0.97,
        0.027,
        "Adaptive exploratory follow-up; handoff requires two consecutive behavioral-and-causal checkpoints",
        ha="right",
        fontsize=7.6,
        color=GRAY,
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / "fig17_winner_knockout.pdf", bbox_inches="tight")
    figure.savefig(FIGURES / "fig17_winner_knockout.png", dpi=240, bbox_inches="tight")
    plt.close(figure)


def _pilot_outputs(audit: Mapping[str, Any], runs: Sequence[Run]) -> None:
    DERIVED.mkdir(parents=True, exist_ok=True)
    reconstruction = cast(Sequence[Mapping[str, Any]], audit["reconstruction_rows"])
    clean_audit = {key: value for key, value in audit.items() if key != "reconstruction_rows"}
    write_json(DERIVED / "e17_pilot_audit.json", clean_audit)
    write_csv(DERIVED / "e17_pilot_reconstruction.csv", reconstruction)
    # Sanity summaries are explicitly labelled pilot and never used by full mode.
    eligibility, intersection = eligibility_rows(runs)
    endpoints = run_endpoint_rows(runs, intersection)
    write_csv(DERIVED / "e17_pilot_eligibility.csv", eligibility)
    write_csv(DERIVED / "e17_pilot_endpoints.csv", endpoints)


def _full_outputs(
    *,
    audit: Mapping[str, Any],
    eligibility: Sequence[Mapping[str, Any]],
    checkpoints: Sequence[Mapping[str, Any]],
    endpoints: Sequence[Mapping[str, Any]],
    contrasts: Sequence[Mapping[str, Any]],
    manipulation_rows: Sequence[Mapping[str, Any]],
    manipulation: Mapping[str, Any],
    starting: Sequence[Mapping[str, Any]],
    taxonomies: Sequence[Mapping[str, Any]],
    decisions: Mapping[str, Any],
    eligible_seeds: Sequence[int],
) -> Mapping[str, Any]:
    DERIVED.mkdir(parents=True, exist_ok=True)
    reconstruction = cast(Sequence[Mapping[str, Any]], audit["reconstruction_rows"])
    clean_audit = {key: value for key, value in audit.items() if key != "reconstruction_rows"}
    files = {
        "checkpoint_dynamics": "e17_checkpoint_dynamics.csv",
        "run_endpoints": "e17_run_endpoints.csv",
        "paired_contrasts": "e17_paired_contrasts.csv",
        "eligibility": "e17_eligibility.csv",
        "manipulation_checks": "e17_manipulation_checks.csv",
        "starting_state_balance": "e17_starting_state_balance.csv",
        "endpoint_taxonomy": "e17_endpoint_taxonomy.csv",
        "reconstruction": "e17_reconstruction_audit.csv",
    }
    for name, rows in (
        (files["checkpoint_dynamics"], checkpoints),
        (files["run_endpoints"], endpoints),
        (files["paired_contrasts"], contrasts),
        (files["eligibility"], eligibility),
        (files["manipulation_checks"], manipulation_rows),
        (files["starting_state_balance"], starting),
        (files["endpoint_taxonomy"], taxonomies),
        (files["reconstruction"], reconstruction),
    ):
        write_csv(DERIVED / name, rows)
    write_json(DERIVED / "e17_audit.json", clean_audit)
    report: Mapping[str, Any] = {
        "experiment": "E17 adaptive winner-knockout handoff",
        "inference_status": "adaptive_exploratory_posthoc_seed_paired",
        "independent_unit": "training seed",
        "registered_runs": 120,
        "eligible_intersection_size": len(eligible_seeds),
        "eligible_intersection_seeds": list(eligible_seeds),
        "eligibility_design_minimum": 15,
        "hard_audit": clean_audit,
        "manipulation_check": dict(manipulation),
        "interpretation_decisions": dict(decisions),
        "paired_contrasts": list(contrasts),
        "endpoint_taxonomy": list(taxonomies),
        "outputs": files,
        "figure": "figures/fig17_winner_knockout.pdf",
    }
    write_json(DERIVED / "e17_analysis.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "full"), default="full")
    parser.add_argument("--artifacts", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = args.artifacts or (DEFAULT_PILOT_ARTIFACTS if args.mode == "pilot" else DEFAULT_ARTIFACTS)
    runs, audit = load_runs(artifacts, mode=args.mode)
    if args.mode == "pilot":
        _pilot_outputs(audit, runs)
        _, intersection = eligibility_rows(runs)
        print(
            f"E17 pilot strict audit passed: {len(runs)} runs, "
            f"{len(intersection)}/{len(runs) // 6} paired-intersection eligible seeds"
        )
        print(f"  metric records audited: {audit['metric_records']:,}")
        print(f"  audit: {DERIVED / 'e17_pilot_audit.json'}")
        return

    eligibility, intersection = eligibility_rows(runs)
    if len(intersection) < 15:
        clean_audit = {key: value for key, value in audit.items() if key != "reconstruction_rows"}
        failure = {
            **clean_audit,
            "eligible_intersection_size": len(intersection),
            "eligible_intersection_seeds": list(intersection),
            "eligibility_design_minimum": 15,
            "design_failure": True,
            "full_interpretation_emitted": False,
        }
        write_json(DERIVED / "e17_design_failure_audit.json", failure)
        raise RuntimeError(
            f"E17 eligibility design failure: {len(intersection)}/20 < 15; "
            "inferential outputs were not emitted"
        )
    checkpoints = checkpoint_rows(runs, intersection)
    endpoints = run_endpoint_rows(runs, intersection)
    contrasts = paired_contrast_rows(endpoints, intersection)
    manipulation_rows, manipulation = manipulation_check_rows(runs, SEEDS)
    starting = starting_state_rows(endpoints, intersection)
    taxonomies = taxonomy_rows(endpoints, intersection)
    decisions = interpretation_decisions(contrasts, manipulation)
    report = _full_outputs(
        audit=audit,
        eligibility=eligibility,
        checkpoints=checkpoints,
        endpoints=endpoints,
        contrasts=contrasts,
        manipulation_rows=manipulation_rows,
        manipulation=manipulation,
        starting=starting,
        taxonomies=taxonomies,
        decisions=decisions,
        eligible_seeds=intersection,
    )
    make_figure(checkpoints, endpoints, contrasts, manipulation_rows, intersection)
    print(f"E17 strict analysis passed: {len(runs)}/120 runs")
    print(f"  paired-intersection eligible seeds: {len(intersection)}/20")
    print(
        "  Q-specific readiness criterion: "
        f"{report['interpretation_decisions']['q_specific_readiness_criterion_met']}"
    )
    print(f"  report: {DERIVED / 'e17_analysis.json'}")
    print(f"  figure: {FIGURES / 'fig17_winner_knockout.pdf'}")


if __name__ == "__main__":
    main()
