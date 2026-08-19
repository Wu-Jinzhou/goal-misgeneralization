#!/usr/bin/env python3
"""Frozen full-panel analysis for E19's active Q-input intervention.

This program is deliberately separate from :mod:`e19_pilot_gate`.  It accepts
only the frozen 20-seed by six-arm adaptive/post-hoc panel, audits the complete
artifact and pairing contract, independently reconstructs the registered
control AUC, and then evaluates the frozen all-seed estimands.  Eligibility is
reported only as a sensitivity analysis; it never selects the primary sample.

The resulting language is intentionally narrow.  A positive result identifies
a contribution of the edited first-layer Q input columns to later Q control.
It does not establish mediation by Q representations or by a unique circuit.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np
import torch
import yaml  # type: ignore[import-untyped]
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
from forkworld.config import expand_sweep, load_config  # noqa: E402
from forkworld.handoff import (  # noqa: E402
    audit_handoff_phase_b,
    make_handoff_phase_b,
    persistent_handoff,
    pure_control,
    semantic_batch_digest,
    stable_state_digest,
    static_sampler_digest,
)
from forkworld.protocols import build_model  # noqa: E402
from forkworld.q_pathway import (  # noqa: E402
    EXPECTED_FEATURE_NAMES,
    EXPECTED_STATE_SHAPES,
    PADDING_SHAM_COLUMN_INDICES,
    Q_COLUMN_INDICES,
    Q_PATHWAY_BRANCHES,
)

EXPERIMENT = "active_q_input_pathway_intervention"
CONFIG_PATH = REPO / "configs" / "e19_q_pathway_mediation.yaml"
DEFAULT_ARTIFACTS = REPO / "artifacts-e19"
DEFAULT_OUTPUT = HERE / "derived"

SEEDS = (
    409,
    419,
    421,
    431,
    433,
    439,
    443,
    449,
    457,
    461,
    463,
    467,
    479,
    487,
    491,
    499,
    503,
    509,
    521,
    523,
)
BRANCHES = tuple(Q_PATHWAY_BRANCHES)
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
LAYERS = ("raw", "first_hidden", "final_hidden")
PROBE_LABELS = (
    "P",
    "Q",
    "Y",
    "truth_table_control",
    "P_permuted",
    "Q_permuted",
    "Y_permuted",
    "truth_table_control_permuted",
)
FROZEN_SOURCE_FINGERPRINT = "ab91250cbb8379c1e6afd0f960abe0754b08a745bd2d9fd8922be0f65a1ba7a9"
FROZEN_SOURCE_FILE_COUNT = 31
FROZEN_CONFIG_SHA256 = "79e24a5e68b95e76e3bffd61d62abfdd86f6f6c439a45c76c376a2f548b74345"
ARTIFACT_SCHEMA_VERSION = 1
SOURCE_FINGERPRINT_SCHEMA_VERSION = 1
BOOTSTRAP_DRAWS = 4_000
AUC_HORIZON = 128
THRESHOLD = 0.90
MARGIN = 0.10
PRIMARY_EFFECT = 0.02
SHAM_AUC_MARGIN = 0.01
PROBE_SHIFT = 0.05
PRESERVATION_MARGIN = 0.05
SHAM_DELTA_NORM_TOLERANCE = 1e-6
MINIMUM_SIGN_COUNT = 15
MINIMUM_ELIGIBLE = 15
NON_SCIENTIFIC_RUN_FIELDS = frozenset({"output_root", "resume", "seeds"})

NOOP_FOR = {
    "independent_q_restore": "independent_noop",
    "independent_padding_sham": "independent_noop",
    "nested_q_transplant": "nested_noop",
    "nested_padding_sham": "nested_noop",
}
EDIT_BRANCHES = tuple(NOOP_FOR)
PRESERVATION_FIELDS = {
    "P_behavior": "postedit_behavior_P",
    "P_causal": "postedit_causal_P",
    "selective_P": "postedit_selective_P",
    "selective_Y": "postedit_selective_Y",
}

MetricKey: TypeAlias = tuple[str, str, str, str]


def get(mapping: Mapping[str, Any], path: str, default: Any = None) -> Any:
    value: Any = mapping
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
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


def _expected_run_id(config: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    implementation = get(metadata, "implementation", {})
    identity = {
        "config": _canonical(_scientific_config(config)),
        "seed": int(config["seed"]),
        "artifact_schema_version": get(implementation, "artifact_schema_version"),
        "source_fingerprint_schema_version": get(implementation, "source_fingerprint_schema_version"),
        "implementation_fingerprint": get(implementation, "implementation_fingerprint"),
    }
    raw = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def _hash_is_valid(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite(value: Any) -> bool:
    return bool(
        isinstance(value, (int, float, np.number))
        and not isinstance(value, (bool, np.bool_))
        and math.isfinite(float(value))
    )


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read JSON object {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise RuntimeError(f"expected JSON object: {path}")
    return cast(Mapping[str, Any], value)


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise RuntimeError(f"cannot read YAML mapping {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise RuntimeError(f"expected YAML mapping: {path}")
    return cast(Mapping[str, Any], value)


def _expect(errors: list[str], run_id: str, actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        errors.append(f"{run_id}: {label}={actual!r}, expected {expected!r}")


def _expect_close(
    errors: list[str],
    run_id: str,
    actual: Any,
    expected: float,
    label: str,
    *,
    atol: float = 1e-12,
) -> None:
    if not _finite(actual) or not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=atol):
        errors.append(f"{run_id}: {label}={actual!r}, expected {expected!r}")


def _fail(name: str, errors: Sequence[str]) -> None:
    preview = "\n".join(f"  - {error}" for error in errors[:100])
    suffix = "" if len(errors) <= 100 else f"\n  ... and {len(errors) - 100} more"
    raise RuntimeError(f"{name} failed with {len(errors)} issue(s):\n{preview}{suffix}")


def direct_control_auc(
    snapshots: Mapping[int, Mapping[str, Any]],
    goal: str,
    *,
    horizon: int = AUC_HORIZON,
) -> float:
    """Independently calculate the registered linearly interpolated AUC."""

    if goal not in GOALS or horizon < 1:
        raise ValueError("goal must be P/Q/Y and horizon must be positive")
    steps = sorted(int(step) for step in snapshots)
    if not steps or steps[0] != 0 or steps[-1] < horizon or horizon not in steps:
        raise ValueError("snapshots must directly span step zero through the horizon")
    x_source = np.asarray(steps, dtype=np.float64)
    y_source = np.asarray(
        [
            0.5
            * (
                float(get(snapshots[step], f"behavior.{goal}"))
                + float(get(snapshots[step], f"causal.{goal}"))
            )
            for step in steps
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(y_source)):
        raise ValueError("AUC snapshots contain non-finite values")
    interior = x_source[x_source < horizon]
    x = np.concatenate((interior, np.asarray([float(horizon)])))
    y = np.interp(x, x_source, y_source)
    return float(np.trapezoid(y, x) / float(horizon))


def _bootstrap_paired(differences: NDArray[np.float64], *, key: str) -> dict[str, float | int]:
    """Frozen deterministic seed bootstrap over paired differences."""

    if differences.ndim != 1 or len(differences) < 1 or not np.all(np.isfinite(differences)):
        raise ValueError(f"invalid paired differences for {key}")
    digest = hashlib.blake2b(key.encode(), digest_size=8, person=b"forke19")
    rng = np.random.default_rng(int.from_bytes(digest.digest(), "little"))
    indices = rng.integers(0, len(differences), size=(BOOTSTRAP_DRAWS, len(differences)))
    draws = np.mean(differences[indices], axis=1)
    low, high = np.quantile(draws, (0.025, 0.975))
    return {
        "estimate": float(np.mean(differences)),
        "ci_low": float(low),
        "ci_high": float(high),
        "positive_count": int(np.sum(differences > 0.0)),
        "negative_count": int(np.sum(differences < 0.0)),
        "zero_count": int(np.sum(differences == 0.0)),
        "n_seeds": len(differences),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
    }


def _expected_configs() -> dict[str, Mapping[str, Any]]:
    if _sha256(CONFIG_PATH) != FROZEN_CONFIG_SHA256:
        raise RuntimeError("the E19 source configuration no longer has its frozen SHA-256")
    loaded = load_config(CONFIG_PATH)
    cells = expand_sweep(loaded)
    result = {str(get(cell, "h16.branch")): _scientific_config(cell) for cell in cells}
    if tuple(result) != BRANCHES:
        raise RuntimeError(f"the expanded E19 branch order changed: {tuple(result)!r}")
    return result


@dataclass(frozen=True)
class MetricPoint:
    value: float
    n: int
    global_step: int
    stage_step: int
    examples_seen: int


@dataclass(frozen=True)
class FullRun:
    path: Path
    config: Mapping[str, Any]
    summary: Mapping[str, Any]
    metadata: Mapping[str, Any]
    metrics: Mapping[MetricKey, Mapping[int, MetricPoint]]
    metric_lines: int

    @property
    def seed(self) -> int:
        return int(self.summary["seed"])

    @property
    def branch(self) -> str:
        return str(self.summary["branch"])

    def snapshot(self, step: int) -> Mapping[str, Any]:
        value = get(self.summary, f"phase_b_snapshots.{step}")
        if not isinstance(value, Mapping):
            raise RuntimeError(f"{self.path.name}: missing phase-B snapshot {step}")
        return cast(Mapping[str, Any], value)


def _metric_is_selected(stage: str, split: str, intervention: str, metric: str) -> bool:
    if stage.endswith("_behavior"):
        return (
            split == "factorial_eval"
            and intervention == "none"
            and metric in {"rho_p", "rho_q", "rho_y_code", "target_accuracy"}
        )
    if stage.endswith("_causal"):
        return (
            split == "factorial_eval"
            and intervention in GOALS
            and metric in {"causal_score", "causal_prob_score"}
        )
    if stage.endswith("_probe"):
        return (
            split == "factorial_probe"
            and intervention == "none"
            and any(
                metric == f"representations__{layer}__heldout_accuracy__{label}"
                for layer in LAYERS
                for label in PROBE_LABELS
            )
        )
    if stage.endswith("_truth_table"):
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
    if stage.endswith("_optimization"):
        return (
            split == "train_minibatch"
            and intervention == "none"
            and metric in {"loss", "primary_loss", "train_batch_accuracy", "optimizer_steps"}
        )
    return False


def _expected_condition(branch: str, stage: str) -> str:
    base = f"active_q_input_pathway:{branch}"
    if stage.startswith("independent_"):
        return f"{base}:independent_prefix"
    if stage.startswith("nested_"):
        return f"{base}:nested_prefix"
    if stage == "final":
        return "adaptive_q_pathway_intervention"
    return base


def _load_metric_index(
    path: Path,
    *,
    run_id: str,
    seed: int,
    branch: str,
    errors: list[str],
) -> tuple[dict[MetricKey, dict[int, MetricPoint]], int]:
    selected: dict[MetricKey, dict[int, MetricPoint]] = defaultdict(dict)
    stages: set[str] = set()
    line_count = 0
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as error:
        errors.append(f"{run_id}: cannot read metrics.jsonl: {error}")
        return {}, 0
    with handle:
        for line_number, line in enumerate(handle, 1):
            line_count += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                errors.append(f"{run_id}: malformed metric line {line_number}: {error}")
                continue
            if not isinstance(row, Mapping):
                errors.append(f"{run_id}: metric line {line_number} is not an object")
                continue
            stage = str(row.get("stage", ""))
            stages.add(stage)
            if row.get("run_id") != run_id or row.get("seed") != seed:
                errors.append(f"{run_id}: metric identity mismatch at line {line_number}")
            if row.get("experiment") != "h16" or row.get("level") != "choice":
                errors.append(f"{run_id}: metric design mismatch at line {line_number}")
            if row.get("condition") != _expected_condition(branch, stage):
                errors.append(f"{run_id}: metric condition mismatch at line {line_number}")
            if not _finite(row.get("value")):
                errors.append(f"{run_id}: non-finite metric value at line {line_number}")
            if not isinstance(row.get("n"), int) or isinstance(row.get("n"), bool):
                errors.append(f"{run_id}: malformed metric n at line {line_number}")
            split = str(row.get("split", ""))
            intervention = str(row.get("intervention", "none"))
            metric = str(row.get("metric", ""))
            if not _metric_is_selected(stage, split, intervention, metric):
                continue
            try:
                local = int(row["stage_step"])
                point = MetricPoint(
                    value=float(row["value"]),
                    n=int(row["n"]),
                    global_step=int(row["global_step"]),
                    stage_step=local,
                    examples_seen=int(row["examples_seen"]),
                )
            except (KeyError, TypeError, ValueError) as error:
                errors.append(f"{run_id}: malformed selected metric line {line_number}: {error}")
                continue
            key = (stage, split, intervention, metric)
            if local in selected[key]:
                errors.append(f"{run_id}: duplicate selected metric {key} at {local}")
            selected[key][local] = point
    required_stages = {
        *(
            f"{overlap}_phase_a_{kind}"
            for overlap in ("independent", "nested")
            for kind in ("behavior", "causal", "probe", "strength", "truth_table", "optimization")
        ),
        *(f"postedit_{kind}" for kind in ("behavior", "causal", "probe", "strength", "truth_table")),
        *(
            f"phase_b_{kind}"
            for kind in ("behavior", "causal", "probe", "strength", "truth_table", "optimization")
        ),
        "final",
    }
    if stages != required_stages:
        errors.append(
            f"{run_id}: metric stages missing={sorted(required_stages - stages)}, "
            f"unexpected={sorted(stages - required_stages)}"
        )
    return dict(selected), line_count


def _metric_steps(
    run: FullRun,
    errors: list[str],
    key: MetricKey,
    expected: Sequence[int],
) -> None:
    actual = set(run.metrics.get(key, {}))
    wanted = set(expected)
    if actual != wanted:
        errors.append(
            f"{run.path.name}: {key} has missing={sorted(wanted - actual)}, "
            f"unexpected={sorted(actual - wanted)}"
        )


def _snapshot_metric_value(snapshot: Mapping[str, Any], key: MetricKey) -> float:
    stage, _split, intervention, metric = key
    if stage.endswith("_behavior"):
        field = {"rho_p": "P", "rho_q": "Q", "rho_y_code": "Y"}.get(metric)
        return (
            float(get(snapshot, f"behavior.{field}"))
            if field is not None
            else float(snapshot["target_accuracy"])
        )
    if stage.endswith("_causal"):
        section = "causal_probability" if metric == "causal_prob_score" else "causal"
        return float(get(snapshot, f"{section}.{intervention}"))
    if stage.endswith("_probe"):
        _, layer, _, label = metric.split("__", 3)
        return float(get(snapshot, f"probe_heldout_accuracy.{layer}.{label}"))
    if stage.endswith("_truth_table"):
        return float(get(snapshot, f"truth_table.{metric}"))
    raise KeyError(key)


def _audit_snapshot(
    snapshot: Any,
    run_id: str,
    label: str,
    errors: list[str],
    *,
    local_step: int,
    global_step: int,
) -> None:
    if not isinstance(snapshot, Mapping):
        errors.append(f"{run_id}: missing {label} snapshot")
        return
    for field, expected in {
        "local_step": local_step,
        "global_step": global_step,
        "examples_seen": global_step * 250,
    }.items():
        _expect(errors, run_id, snapshot.get(field), expected, f"{label}.{field}")
    for section in ("behavior", "causal", "causal_probability", "selective_final_hidden_probe"):
        values = snapshot.get(section)
        if not isinstance(values, Mapping) or set(values) != set(GOALS):
            errors.append(f"{run_id}: malformed {label}.{section}")
            continue
        for goal, value in values.items():
            if not _finite(value):
                errors.append(f"{run_id}: non-finite {label}.{section}.{goal}")
    target = snapshot.get("target_accuracy")
    if not _finite(target):
        errors.append(f"{run_id}: non-finite {label}.target_accuracy")
    probes = snapshot.get("probe_heldout_accuracy")
    labels = set(PROBE_LABELS)
    if not isinstance(probes, Mapping) or set(probes) != set(LAYERS):
        errors.append(f"{run_id}: malformed {label}.probe_heldout_accuracy")
    else:
        for layer in LAYERS:
            values = probes.get(layer)
            if not isinstance(values, Mapping) or set(values) != labels:
                errors.append(f"{run_id}: malformed {label} probe layer {layer}")
            elif not all(_finite(value) for value in values.values()):
                errors.append(f"{run_id}: non-finite {label} probe layer {layer}")
    truth = snapshot.get("truth_table")
    if not isinstance(truth, Mapping):
        errors.append(f"{run_id}: missing {label}.truth_table")
    else:
        for field in (
            "boolean_signature_int",
            "tuple_consistency",
            "codeword_consistency",
            "nuisance_consistency",
            "sign_inversion_symmetry",
        ):
            if not _finite(truth.get(field)):
                errors.append(f"{run_id}: non-finite {label}.truth_table.{field}")
        _expect(
            errors,
            run_id,
            snapshot.get("boolean_signature"),
            truth.get("boolean_signature"),
            f"{label} boolean signature",
        )


def _audit_snapshot_metrics(
    run: FullRun,
    snapshot: Mapping[str, Any],
    errors: list[str],
    *,
    stage_root: str,
    step: int,
    global_step: int,
) -> None:
    stage_kinds = {
        "behavior": ("factorial_eval", "none", ("rho_p", "rho_q", "rho_y_code", "target_accuracy")),
        "causal": ("factorial_eval", None, ("causal_score", "causal_prob_score")),
        "probe": (
            "factorial_probe",
            "none",
            tuple(
                f"representations__{layer}__heldout_accuracy__{label}"
                for layer in LAYERS
                for label in PROBE_LABELS
            ),
        ),
        "truth_table": (
            "factorial_eval",
            "none",
            (
                "boolean_signature_int",
                "tuple_consistency",
                "codeword_consistency",
                "nuisance_consistency",
                "sign_inversion_symmetry",
            ),
        ),
    }
    for kind, (split, fixed_intervention, metrics) in stage_kinds.items():
        stage = f"{stage_root}_{kind}"
        if stage_root in {"independent_phase_a", "nested_phase_a"}:
            expected_steps = PHASE_A_STEPS
        elif stage_root == "postedit":
            expected_steps = (0,)
        elif stage_root == "phase_b":
            expected_steps = PHASE_B_STEPS[1:]
        else:  # pragma: no cover - internal caller contract
            raise ValueError(f"unknown snapshot stage root {stage_root}")
        interventions = GOALS if kind == "causal" else (cast(str, fixed_intervention),)
        for intervention in interventions:
            for metric in metrics:
                key = (stage, split, intervention, metric)
                _metric_steps(run, errors, key, expected_steps)
                point = run.metrics.get(key, {}).get(step)
                if point is None:
                    continue
                if (
                    point.n != 4_096
                    or point.global_step != global_step
                    or point.examples_seen != global_step * 250
                ):
                    errors.append(f"{run.path.name}: malformed coordinates for {key} at {step}")
                expected = _snapshot_metric_value(snapshot, key)
                if not math.isclose(point.value, expected, rel_tol=0.0, abs_tol=1e-12):
                    errors.append(f"{run.path.name}: metric/snapshot mismatch for {key} at {step}")


def _audit_replay(run: FullRun, errors: list[str]) -> None:
    for overlap in ("independent", "nested"):
        replay = get(run.summary, f"replay.{overlap}")
        if not isinstance(replay, Mapping):
            errors.append(f"{run.path.name}: missing {overlap} replay")
            continue
        for field, expected in {
            "overlap": overlap,
            "samples_seen": 11_250,
            "optimizer_steps": 45,
            "batch_size": 250,
            "all_minibatches_full": True,
        }.items():
            _expect(
                errors,
                run.path.name,
                replay.get(field),
                expected,
                f"replay.{overlap}.{field}",
            )
        checks = replay.get("checks")
        expected_checks = {
            "initial_models_equal",
            "final_models_equal",
            "final_optimizers_equal",
            "samples_seen_equal",
            "optimizer_steps_equal",
        }
        if (
            not isinstance(checks, Mapping)
            or set(checks) != expected_checks
            or not all(value is True for value in checks.values())
        ):
            errors.append(f"{run.path.name}: {overlap} replay checks failed")
        hashes = replay.get("hashes")
        expected_hashes = {
            "initial_model",
            "replay_initial_model",
            "observed_final_model",
            "replay_final_model",
            "observed_final_optimizer",
            "replay_final_optimizer",
            "phase_a_batch",
            "phase_a_sampler",
        }
        if not isinstance(hashes, Mapping) or set(hashes) != expected_hashes:
            errors.append(f"{run.path.name}: malformed {overlap} replay hashes")
            continue
        if not all(_hash_is_valid(value) for value in hashes.values()):
            errors.append(f"{run.path.name}: invalid {overlap} replay digest")
        for left, right in (
            ("initial_model", "replay_initial_model"),
            ("observed_final_model", "replay_final_model"),
            ("observed_final_optimizer", "replay_final_optimizer"),
        ):
            _expect(
                errors,
                run.path.name,
                hashes.get(left),
                hashes.get(right),
                f"{overlap} replay {left}",
            )
        _expect(
            errors,
            run.path.name,
            hashes.get("initial_model"),
            get(run.summary, "hashes.initial_model"),
            f"{overlap} initial hash",
        )
        _expect(
            errors,
            run.path.name,
            hashes.get("observed_final_model"),
            get(run.summary, f"hashes.{overlap}_prefix_model"),
            f"{overlap} final hash",
        )
        training_overlap = replay.get("training_overlap")
        expected_overlap_fields = {
            "p_error_count",
            "q_error_count",
            "both_error_count",
            "p_only_error_count",
            "q_only_error_count",
            "error_phi",
        }
        if (
            not isinstance(training_overlap, Mapping)
            or set(training_overlap) != expected_overlap_fields
            or not all(_finite(value) for value in training_overlap.values())
        ):
            errors.append(f"{run.path.name}: malformed {overlap} overlap audit")


def _audit_delta_summary(
    value: Any,
    run_id: str,
    label: str,
    errors: list[str],
    *,
    edited: bool,
) -> None:
    if not isinstance(value, Mapping):
        errors.append(f"{run_id}: missing {label} delta summary")
        return
    required = {
        "shape",
        "target_scalar_count",
        "nonzero_scalar_count",
        "digest",
        "l1_norm",
        "l2_norm",
        "linf_norm",
        "finite",
    }
    if set(value) != required:
        errors.append(f"{run_id}: malformed {label} delta fields")
    expected_count = 128 if edited else 0
    _expect(
        errors,
        run_id,
        value.get("shape"),
        [64, 2] if edited else [64, 0],
        f"{label}.shape",
    )
    _expect(
        errors,
        run_id,
        value.get("target_scalar_count"),
        expected_count,
        f"{label}.target_scalar_count",
    )
    _expect(
        errors,
        run_id,
        value.get("nonzero_scalar_count"),
        expected_count,
        f"{label}.nonzero_scalar_count",
    )
    _expect(errors, run_id, value.get("finite"), True, f"{label}.finite")
    if not _hash_is_valid(value.get("digest")):
        errors.append(f"{run_id}: malformed {label}.digest")
    for field in ("l1_norm", "l2_norm", "linf_norm"):
        numeric = value.get(field)
        if not _finite(numeric) or (float(numeric) <= 0.0 if edited else float(numeric) != 0.0):
            errors.append(f"{run_id}: malformed {label}.{field}")


def _audit_sham_delta_against_intended(
    sham: Any,
    intended: Any,
    run_id: str,
    label: str,
    errors: list[str],
) -> None:
    """Audit the registered float32-tolerant sham displacement amendment."""

    _audit_delta_summary(sham, run_id, label, errors, edited=True)
    if not isinstance(sham, Mapping) or not isinstance(intended, Mapping):
        return
    for norm in ("l1_norm", "l2_norm", "linf_norm"):
        if not (_finite(sham.get(norm)) and _finite(intended.get(norm))) or not math.isclose(
            float(sham[norm]),
            float(intended[norm]),
            rel_tol=0.0,
            abs_tol=SHAM_DELTA_NORM_TOLERANCE,
        ):
            errors.append(
                f"{run_id}: {label}/{norm} differs from intended by more than {SHAM_DELTA_NORM_TOLERANCE:g}"
            )


def _audit_edit(run: FullRun, errors: list[str]) -> None:
    run_id = run.path.name
    edit = get(run.summary, "edit")
    if not isinstance(edit, Mapping):
        errors.append(f"{run_id}: missing edit evidence")
        return
    audit = edit.get("audit")
    if not isinstance(audit, Mapping):
        errors.append(f"{run_id}: missing edit construction audit")
        return
    for field in (
        "donors_unchanged",
        "all_six_branches_verified",
        "all_unaffected_state_exact",
        "all_edits_target_exactly_128_scalars",
    ):
        _expect(errors, run_id, audit.get(field), True, f"edit.audit.{field}")
    contract = audit.get("contract")
    expected_contract = {
        "feature_names": list(EXPECTED_FEATURE_NAMES),
        "feature_names_digest": stable_state_digest(EXPECTED_FEATURE_NAMES),
        "q_column_indices": list(Q_COLUMN_INDICES),
        "q_column_names": ["Q_1", "Q_2"],
        "padding_sham_column_indices": list(PADDING_SHAM_COLUMN_INDICES),
        "padding_sham_column_names": ["R_4", "R_5"],
        "state_shapes": {key: list(value) for key, value in EXPECTED_STATE_SHAPES.items()},
        "parameter_dtype": "torch.float32",
        "cpu_only": True,
        "contract_verified": True,
    }
    if not isinstance(contract, Mapping):
        errors.append(f"{run_id}: missing edit contract")
    else:
        for field, expected in expected_contract.items():
            _expect(
                errors,
                run_id,
                contract.get(field),
                expected,
                f"edit.contract.{field}",
            )
    donors = audit.get("donors")
    if (
        not isinstance(donors, Mapping)
        or set(donors) != {"initial", "independent", "nested"}
        or not all(_hash_is_valid(value) for value in donors.values())
    ):
        errors.append(f"{run_id}: malformed edit donor hashes")
    _expect(
        errors,
        run_id,
        edit.get("donor_state_digests"),
        donors,
        "duplicated donor hashes",
    )
    if isinstance(donors, Mapping):
        for donor, field in {
            "initial": "initial_model",
            "independent": "independent_prefix_model",
            "nested": "nested_prefix_model",
        }.items():
            _expect(
                errors,
                run_id,
                donors.get(donor),
                get(run.summary, f"hashes.{field}"),
                f"donor {donor}",
            )
    branches = audit.get("branches")
    branch_hashes = edit.get("branch_model_hashes")
    if not isinstance(branches, Mapping) or set(branches) != set(BRANCHES):
        errors.append(f"{run_id}: malformed audited branch set")
    if (
        not isinstance(branch_hashes, Mapping)
        or set(branch_hashes) != set(BRANCHES)
        or not all(_hash_is_valid(value) for value in branch_hashes.values())
    ):
        errors.append(f"{run_id}: malformed branch model hashes")
    edited_columns = {
        "independent_noop": [],
        "independent_q_restore": [8, 9],
        "independent_padding_sham": [5, 6],
        "nested_noop": [],
        "nested_q_transplant": [8, 9],
        "nested_padding_sham": [5, 6],
    }
    if isinstance(branches, Mapping) and isinstance(branch_hashes, Mapping):
        for branch in BRANCHES:
            item = branches.get(branch)
            if not isinstance(item, Mapping):
                errors.append(f"{run_id}: missing branch edit audit {branch}")
                continue
            _expect(
                errors,
                run_id,
                item.get("branch_state_digest"),
                branch_hashes.get(branch),
                f"{branch} state hash",
            )
            if not _hash_is_valid(item.get("unchanged_state_digest")):
                errors.append(f"{run_id}: malformed {branch} unchanged-state hash")
            for field in ("unchanged_state_verified", "designated_columns_verified"):
                _expect(errors, run_id, item.get(field), True, f"{branch}.{field}")
            columns = edited_columns[branch]
            _expect(errors, run_id, item.get("edited_columns"), columns, f"{branch} columns")
            _expect(
                errors,
                run_id,
                item.get("target_scalar_count"),
                128 if columns else 0,
                f"{branch} scalar count",
            )
            _audit_delta_summary(
                item.get("realized_delta"),
                run_id,
                f"{branch}.realized_delta",
                errors,
                edited=bool(columns),
            )
        _expect(
            errors,
            run_id,
            branch_hashes.get("independent_noop"),
            donors.get("independent") if isinstance(donors, Mapping) else None,
            "independent no-op donor identity",
        )
        _expect(
            errors,
            run_id,
            branch_hashes.get("nested_noop"),
            donors.get("nested") if isinstance(donors, Mapping) else None,
            "nested no-op donor identity",
        )
        _expect(
            errors,
            run_id,
            get(run.summary, "hashes.selected_postedit_model"),
            branch_hashes.get(run.branch),
            "selected branch hash",
        )
    edits = audit.get("edits")
    if not isinstance(edits, Mapping) or set(edits) != {"restore", "transplant"}:
        errors.append(f"{run_id}: malformed edit displacement set")
    else:
        edit_branch_names = {
            "restore": (
                "independent_q_restore",
                "independent_padding_sham",
            ),
            "transplant": (
                "nested_q_transplant",
                "nested_padding_sham",
            ),
        }
        for name in ("restore", "transplant"):
            item = edits.get(name)
            if not isinstance(item, Mapping):
                errors.append(f"{run_id}: missing {name} displacement")
                continue
            intended = item.get("intended_delta")
            _audit_delta_summary(intended, run_id, f"{name}.intended_delta", errors, edited=True)
            _expect(
                errors,
                run_id,
                item.get("active_realized_delta"),
                intended,
                f"{name} active realized delta",
            )
            active_branch, sham_branch = edit_branch_names[name]
            _expect(
                errors,
                run_id,
                get(branches, f"{active_branch}.realized_delta") if isinstance(branches, Mapping) else None,
                item.get("active_realized_delta"),
                f"{name} active branch/edit delta identity",
            )
            sham = item.get("sham_realized_delta")
            _audit_sham_delta_against_intended(
                sham,
                intended,
                run_id,
                f"{name}.sham_realized_delta",
                errors,
            )
            _expect(
                errors,
                run_id,
                get(branches, f"{sham_branch}.realized_delta") if isinstance(branches, Mapping) else None,
                sham,
                f"{name} sham branch/edit delta identity",
            )
            _expect(errors, run_id, item.get("same_intended_delta_applied"), True, f"{name} paired delta")
            _expect(errors, run_id, item.get("active_target"), "Q_1,Q_2", f"{name} active target")
            _expect(errors, run_id, item.get("sham_target"), "R_4,R_5", f"{name} sham target")
    audit_payload = {
        "contract": audit.get("contract"),
        "donors": audit.get("donors"),
        "branches": audit.get("branches"),
        "edits": audit.get("edits"),
    }
    _expect(
        errors,
        run_id,
        audit.get("audit_digest"),
        stable_state_digest(audit_payload),
        "edit audit digest",
    )
    if isinstance(branch_hashes, Mapping) and isinstance(edits, Mapping):
        construction_payload = {
            "donors": donors,
            "branches": dict(branch_hashes),
            "restore_delta": get(edits, "restore.intended_delta"),
            "transplant_delta": get(edits, "transplant.intended_delta"),
            "features": EXPECTED_FEATURE_NAMES,
        }
        _expect(
            errors,
            run_id,
            audit.get("construction_digest"),
            stable_state_digest(construction_payload),
            "edit construction digest",
        )
        _expect(
            errors,
            run_id,
            edit.get("branch_set_digest"),
            stable_state_digest(dict(branch_hashes)),
            "branch-set digest",
        )
    preactivation = edit.get("preactivation_effects")
    if not isinstance(preactivation, Mapping):
        errors.append(f"{run_id}: missing preactivation audit")
    else:
        for field, expected in {
            "panel_n": 4_096,
            "panel_digest": get(run.summary, "data.probe_eval_digest"),
            "feature_names_digest": stable_state_digest(EXPECTED_FEATURE_NAMES),
            "max_k": 5,
            "include_state": True,
            "all_values_finite": True,
        }.items():
            _expect(errors, run_id, preactivation.get(field), expected, f"preactivation.{field}")
        comparisons = preactivation.get("comparisons")
        if not isinstance(comparisons, Mapping) or set(comparisons) != set(EDIT_BRANCHES):
            errors.append(f"{run_id}: malformed preactivation comparison set")
        else:
            digest_fields = {
                "baseline_preactivation_digest",
                "edited_preactivation_digest",
                "preactivation_delta_digest",
            }
            numeric_fields = {
                "baseline_preactivation_rms",
                "edited_preactivation_rms",
                "preactivation_delta_rms",
                "relative_delta_rms",
                "preactivation_delta_mean_absolute",
                "preactivation_delta_max_absolute",
                "relu_state_flip_rate",
            }
            for branch, comparison in comparisons.items():
                if not isinstance(comparison, Mapping):
                    errors.append(f"{run_id}: malformed preactivation {branch}")
                    continue
                _expect(errors, run_id, comparison.get("n_rows"), 4_096, f"{branch} rows")
                _expect(errors, run_id, comparison.get("hidden_units"), 64, f"{branch} units")
                _expect(errors, run_id, comparison.get("activation_count"), 262_144, f"{branch} activations")
                if not all(_hash_is_valid(comparison.get(field)) for field in digest_fields):
                    errors.append(f"{run_id}: malformed preactivation digest for {branch}")
                if not all(_finite(comparison.get(field)) for field in numeric_fields):
                    errors.append(f"{run_id}: non-finite preactivation value for {branch}")
                count = comparison.get("relu_state_flip_count")
                rate = comparison.get("relu_state_flip_rate")
                if not isinstance(count, int) or isinstance(count, bool) or not 0 <= count <= 262_144:
                    errors.append(f"{run_id}: malformed ReLU flip count for {branch}")
                elif _finite(rate) and not math.isclose(
                    float(rate), count / 262_144, rel_tol=0.0, abs_tol=1e-12
                ):
                    errors.append(f"{run_id}: ReLU flip rate/count mismatch for {branch}")


def _audit_optimizer_metrics(run: FullRun, errors: list[str]) -> None:
    specifications = (
        ("independent_phase_a_optimization", PHASE_A_STEPS, 0),
        ("nested_phase_a_optimization", PHASE_A_STEPS, 0),
        ("phase_b_optimization", PHASE_B_STEPS[1:], 45),
    )
    metrics = ("loss", "primary_loss", "train_batch_accuracy", "optimizer_steps")
    for stage, steps, offset in specifications:
        for metric in metrics:
            key = (stage, "train_minibatch", "none", metric)
            _metric_steps(run, errors, key, steps)
            for step, point in run.metrics.get(key, {}).items():
                expected_global = offset + step
                if (
                    point.n != 10_000
                    or point.global_step != expected_global
                    or point.examples_seen != expected_global * 250
                ):
                    errors.append(f"{run.path.name}: malformed optimization coordinates for {key} at {step}")
                if metric == "optimizer_steps" and point.value != float(step):
                    errors.append(f"{run.path.name}: optimizer step metric differs at {stage}:{step}")


def _validate_run(run: FullRun, expected_config: Mapping[str, Any], errors: list[str]) -> None:
    run_id = run.path.name
    if _scientific_config(run.config) != expected_config:
        errors.append(f"{run_id}: resolved scientific config differs from frozen full config")
    fixed = {
        "hypothesis": "h16",
        "seed": run.seed,
        "branch": run.branch,
        "condition": f"active_q_input_pathway:{run.branch}",
        "pilot_only": False,
        "design_status": "adaptive_posthoc_active_q_input_pathway_intervention_frozen_pre_outcome",
        "data.phase_b_constructed": True,
        "data.probe_splits_disjoint": True,
        "data.phase_b_pairing_verified": True,
        "measurement.probe_train_n": 2_048,
        "measurement.probe_eval_n": 4_096,
        "measurement.probe_ridge": 0.001,
        "measurement.truth_table_control_seed": 1_500_450_271,
        "measurement.phase_b_checkpoints": list(PHASE_B_STEPS),
        "measurement.direct_auc_horizon": True,
        "measurement.directional_causal_normalization": "(1 + E[g*(a-a_flip)/2]) / 2",
        "training.batch_size": 250,
        "training.phase_a_steps_per_history": 45,
        "training.phase_a_examples_seen_per_history": 11_250,
        "training.phase_b_steps": 1_024,
        "training.phase_b_examples_seen": 256_000,
        "training.total_examples_seen_selected_trajectory": 267_250,
        "training.all_minibatches_full": True,
        "model.input_dim": 19,
        "model.total_parameters": 5_505,
        "model.trainable_parameters": 5_505,
        "model.update_mode": "full",
        "model.requested_budget": "full",
        "outcomes.auc_horizon": AUC_HORIZON,
    }
    for path, expected in fixed.items():
        _expect(errors, run_id, get(run.summary, path), expected, f"summary.{path}")
    wall = get(run.summary, "training.wall_seconds")
    if not _finite(wall) or float(wall) <= 0.0:
        errors.append(f"{run_id}: invalid wall time")

    prefix = get(run.summary, "prefix_snapshots")
    if not isinstance(prefix, Mapping) or set(prefix) != {"independent", "nested"}:
        errors.append(f"{run_id}: malformed prefix snapshots")
    else:
        for overlap in ("independent", "nested"):
            snapshots = prefix.get(overlap)
            if not isinstance(snapshots, Mapping) or set(snapshots) != {"33", "45"}:
                errors.append(f"{run_id}: malformed {overlap} prefix lattice")
                continue
            for step in PHASE_A_STEPS:
                snapshot = cast(Mapping[str, Any], snapshots[str(step)])
                _audit_snapshot(
                    snapshot,
                    run_id,
                    f"prefix.{overlap}.{step}",
                    errors,
                    local_step=step,
                    global_step=step,
                )
                _audit_snapshot_metrics(
                    run,
                    snapshot,
                    errors,
                    stage_root=f"{overlap}_phase_a",
                    step=step,
                    global_step=step,
                )
    postedit = get(run.summary, "postedit_snapshot")
    _audit_snapshot(
        postedit,
        run_id,
        "postedit",
        errors,
        local_step=0,
        global_step=45,
    )
    if isinstance(postedit, Mapping):
        _audit_snapshot_metrics(
            run,
            cast(Mapping[str, Any], postedit),
            errors,
            stage_root="postedit",
            step=0,
            global_step=45,
        )
        postedit_pure = pure_control(
            cast(Mapping[str, float], postedit["behavior"]),
            cast(Mapping[str, float], postedit["causal"]),
            "P",
            threshold=THRESHOLD,
            margin=MARGIN,
        )
        _expect(
            errors,
            run_id,
            get(run.summary, "postedit_pure_p"),
            postedit_pure,
            "postedit pure-P",
        )

    eligibility = get(run.summary, "eligibility")
    if not isinstance(eligibility, Mapping):
        errors.append(f"{run_id}: missing eligibility audit")
    else:
        _expect(errors, run_id, eligibility.get("threshold"), THRESHOLD, "eligibility threshold")
        _expect(errors, run_id, eligibility.get("margin"), MARGIN, "eligibility margin")
        history_results: list[bool] = []
        for overlap in ("independent", "nested"):
            item = eligibility.get(overlap)
            if not isinstance(item, Mapping):
                errors.append(f"{run_id}: malformed {overlap} eligibility")
                continue
            _expect(errors, run_id, item.get("threshold"), THRESHOLD, f"{overlap} threshold")
            _expect(errors, run_id, item.get("margin"), MARGIN, f"{overlap} margin")
            checkpoints = item.get("per_checkpoint")
            if not isinstance(checkpoints, Mapping) or set(checkpoints) != {"33", "45"}:
                errors.append(f"{run_id}: malformed {overlap} eligibility checkpoints")
                continue
            recomputed = []
            for step in PHASE_A_STEPS:
                snapshot = get(run.summary, f"prefix_snapshots.{overlap}.{step}")
                if not isinstance(snapshot, Mapping):
                    continue
                qualifies = pure_control(
                    cast(Mapping[str, float], snapshot["behavior"]),
                    cast(Mapping[str, float], snapshot["causal"]),
                    "P",
                    threshold=THRESHOLD,
                    margin=MARGIN,
                )
                recomputed.append(qualifies)
                _expect(
                    errors,
                    run_id,
                    checkpoints.get(str(step)),
                    qualifies,
                    f"{overlap} eligibility at {step}",
                )
            endpoint = len(recomputed) == 2 and all(recomputed)
            history_results.append(endpoint)
            _expect(
                errors,
                run_id,
                item.get("eligible_both_registered_checkpoints"),
                endpoint,
                f"{overlap} eligibility endpoint",
            )
        if len(history_results) == 2:
            _expect(
                errors,
                run_id,
                eligibility.get("paired_intersection_eligible"),
                all(history_results),
                "paired eligibility endpoint",
            )

    phase_b = get(run.summary, "phase_b_snapshots")
    if not isinstance(phase_b, Mapping) or set(phase_b) != {str(step) for step in PHASE_B_STEPS}:
        errors.append(f"{run_id}: phase-B snapshot lattice differs from frozen direct checkpoints")
    else:
        _expect(errors, run_id, get(run.summary, "phase_b_local_zero"), phase_b["0"], "phase-B local zero")
        _expect(errors, run_id, phase_b["0"], postedit, "postedit/direct-zero identity")
        _expect(errors, run_id, get(run.summary, "final"), phase_b["1024"], "phase-B final")
        for step in PHASE_B_STEPS:
            snapshot = cast(Mapping[str, Any], phase_b[str(step)])
            _audit_snapshot(
                snapshot,
                run_id,
                f"phase_b.{step}",
                errors,
                local_step=step,
                global_step=45 + step,
            )
            # Local zero is the already-audited post-edit record, not a duplicate
            # phase-B metric record.
            if step > 0:
                _audit_snapshot_metrics(
                    run,
                    snapshot,
                    errors,
                    stage_root="phase_b",
                    step=step,
                    global_step=45 + step,
                )
        direct = {int(step): cast(Mapping[str, Any], value) for step, value in phase_b.items()}
        event_input = {
            step: {"behavior": snap["behavior"], "causal": snap["causal"]} for step, snap in direct.items()
        }
        for goal in GOALS:
            stored_auc = get(
                run.summary,
                f"outcomes.normalized_control_auc_through_direct_checkpoint.{goal}",
            )
            calculated = direct_control_auc(direct, goal, horizon=AUC_HORIZON)
            _expect_close(
                errors,
                run_id,
                stored_auc,
                calculated,
                f"independent AUC({goal})",
            )
            event = persistent_handoff(
                event_input,
                goal,
                threshold=THRESHOLD,
                margin=MARGIN,
            )
            _expect(
                errors,
                run_id,
                get(run.summary, f"outcomes.persistent_control_events.{goal}"),
                event,
                f"persistent {goal} event",
            )

    hashes = get(run.summary, "hashes")
    expected_hashes = {
        "initial_model",
        "independent_prefix_model",
        "nested_prefix_model",
        "selected_postedit_model",
        "phase_b_initial_model",
        "phase_b_initial_optimizer",
        "phase_b_final_model",
        "phase_b_final_optimizer",
    }
    if (
        not isinstance(hashes, Mapping)
        or set(hashes) != expected_hashes
        or not all(_hash_is_valid(value) for value in hashes.values())
    ):
        errors.append(f"{run_id}: malformed full hash set")
    elif hashes["phase_b_initial_model"] != hashes["selected_postedit_model"]:
        errors.append(f"{run_id}: phase-B model did not start at selected edit")

    optimizer = get(run.summary, "optimizer_transition_audit")
    optimizer_fixed = {
        "source": "fresh_empty_after_edit",
        "semantic_reset_implemented_by_fresh_object": True,
        "provided_optimizer": True,
        "low_level_reset_optimizer_flag": False,
        "state_entry_count": 0,
        "adam_step_min": None,
        "adam_step_max": None,
        "adam_step_entry_count": 0,
        "param_group_count": 1,
        "optimized_parameter_count": 6,
    }
    if not isinstance(optimizer, Mapping):
        errors.append(f"{run_id}: missing optimizer transition audit")
    else:
        for field, expected in optimizer_fixed.items():
            _expect(errors, run_id, optimizer.get(field), expected, f"optimizer.{field}")
        if not _hash_is_valid(optimizer.get("actual_state_digest")):
            errors.append(f"{run_id}: malformed empty optimizer digest")
        _expect(
            errors,
            run_id,
            optimizer.get("actual_state_digest"),
            get(run.summary, "hashes.phase_b_initial_optimizer"),
            "empty optimizer hash",
        )

    data = get(run.summary, "data")
    expected_data_fields = {
        "phase_b_constructed",
        "probe_train_digest",
        "probe_eval_digest",
        "probe_splits_disjoint",
        "phase_b",
        "phase_b_batch_digest",
        "phase_b_sampler_digest",
        "phase_b_pairing_verified",
    }
    if not isinstance(data, Mapping) or set(data) != expected_data_fields:
        errors.append(f"{run_id}: malformed full data evidence")
    else:
        for field in (
            "probe_train_digest",
            "probe_eval_digest",
            "phase_b_batch_digest",
            "phase_b_sampler_digest",
        ):
            if not _hash_is_valid(data.get(field)):
                errors.append(f"{run_id}: malformed data digest {field}")
        _expect(
            errors,
            run_id,
            get(data, "phase_b.batch_digest"),
            data.get("phase_b_batch_digest"),
            "duplicated phase-B batch digest",
        )

    _audit_replay(run, errors)
    _audit_edit(run, errors)
    _audit_optimizer_metrics(run, errors)


def _cross_run_audit(runs: Mapping[tuple[int, str], FullRun], errors: list[str]) -> dict[str, Any]:
    reconstruction_rows: list[dict[str, Any]] = []
    common_fields = (
        "prefix_snapshots",
        "eligibility",
        "edit",
        "replay",
        "data",
        "model",
        "measurement",
    )
    for seed in SEEDS:
        arms = {branch: runs[(seed, branch)] for branch in BRANCHES}
        anchor = arms[BRANCHES[0]]
        for run in arms.values():
            for field in common_fields:
                if run.summary.get(field) != anchor.summary.get(field):
                    errors.append(f"seed {seed}: cross-arm pairing differs at {field}")
            for field in (
                "initial_model",
                "independent_prefix_model",
                "nested_prefix_model",
                "phase_b_initial_optimizer",
            ):
                if get(run.summary, f"hashes.{field}") != get(anchor.summary, f"hashes.{field}"):
                    errors.append(f"seed {seed}: cross-arm pairing differs at hashes.{field}")

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
        expected_data = {
            "probe_train_digest": semantic_batch_digest(probe_train),
            "probe_eval_digest": semantic_batch_digest(probe_eval),
            "phase_b_batch_digest": semantic_batch_digest(phase_b),
            "phase_b_sampler_digest": static_sampler_digest(
                10_000,
                batch_size=250,
                steps=1_024,
                seed=90_000_000 + seed,
                shuffle=True,
            ),
        }
        for field, expected in expected_data.items():
            _expect(
                errors,
                anchor.path.name,
                get(anchor.summary, f"data.{field}"),
                expected,
                f"reconstructed {field}",
            )
        _expect(
            errors,
            anchor.path.name,
            get(anchor.summary, "data.phase_b"),
            phase_b_audit,
            "reconstructed phase-B audit",
        )
        feature_names = phase_b.feature_names(max_k=5, include_state=True)
        _expect(
            errors,
            anchor.path.name,
            feature_names,
            EXPECTED_FEATURE_NAMES,
            "reconstructed phase-B feature contract",
        )
        if (
            probe_train.feature_names(max_k=5, include_state=True) != feature_names
            or probe_eval.feature_names(max_k=5, include_state=True) != feature_names
        ):
            errors.append(f"seed {seed}: reconstructed probe interfaces differ")
        if set(np.asarray(probe_train.sample_id).tolist()) & set(np.asarray(probe_eval.sample_id).tolist()):
            errors.append(f"seed {seed}: reconstructed probe panels overlap")

        model, model_report = build_model(probe_eval, anchor.config, seed)
        initial_hash = stable_state_digest(model.state_dict())
        _expect(
            errors,
            anchor.path.name,
            anchor.summary.get("model"),
            model_report,
            "reconstructed model report",
        )
        _expect(
            errors,
            anchor.path.name,
            get(anchor.summary, "hashes.initial_model"),
            initial_hash,
            "reconstructed initial model",
        )
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=0.003,
            weight_decay=0.0,
        )
        empty_optimizer_digest = stable_state_digest(optimizer.state_dict())
        _expect(
            errors,
            anchor.path.name,
            get(anchor.summary, "hashes.phase_b_initial_optimizer"),
            empty_optimizer_digest,
            "reconstructed empty optimizer",
        )

        overlap_rows: dict[str, Any] = {}
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
                overlap=overlap,  # type: ignore[arg-type]
                state_dim=8,
            )
            expected_batch = semantic_batch_digest(bundle.train)
            expected_sampler = static_sampler_digest(
                10_000,
                batch_size=250,
                steps=45,
                seed=seed,
                shuffle=True,
            )
            _expect(
                errors,
                anchor.path.name,
                get(anchor.summary, f"replay.{overlap}.hashes.phase_a_batch"),
                expected_batch,
                f"reconstructed {overlap} phase-A batch",
            )
            _expect(
                errors,
                anchor.path.name,
                get(anchor.summary, f"replay.{overlap}.hashes.phase_a_sampler"),
                expected_sampler,
                f"reconstructed {overlap} phase-A sampler",
            )
            training_overlap = {
                key: bundle.train.metadata[key]
                for key in (
                    "p_error_count",
                    "q_error_count",
                    "both_error_count",
                    "p_only_error_count",
                    "q_only_error_count",
                    "error_phi",
                )
            }
            _expect(
                errors,
                anchor.path.name,
                get(anchor.summary, f"replay.{overlap}.training_overlap"),
                training_overlap,
                f"reconstructed {overlap} overlap audit",
            )
            overlap_rows[overlap] = {
                "phase_a_batch_digest": expected_batch,
                "phase_a_sampler_digest": expected_sampler,
                "training_overlap": training_overlap,
            }
        reconstruction_rows.append(
            {
                "seed": seed,
                **expected_data,
                "initial_model_digest": initial_hash,
                "empty_optimizer_digest": empty_optimizer_digest,
                "histories": overlap_rows,
            }
        )
    return {
        "seed_blocks": len(SEEDS),
        "arms_per_seed": len(BRANCHES),
        "phase_b_probe_sampler_and_optimizer_digests_common_within_seed": True,
        "current_source_data_model_and_optimizer_reconstruction_passed": True,
        "sham_displacement_norm_tolerance": SHAM_DELTA_NORM_TOLERANCE,
        "reconstruction_rows": reconstruction_rows,
    }


def _artifact_directory(root: Path) -> Path:
    nested = root / "h16" / EXPERIMENT
    if nested.is_dir():
        return nested
    if root.name == EXPERIMENT and root.parent.name == "h16" and root.is_dir():
        return root
    raise RuntimeError(f"missing E19 full artifact directory {nested}")


def load_and_audit(
    root: Path,
) -> tuple[dict[tuple[int, str], FullRun], dict[str, Any]]:
    """Load only an exact frozen full panel and perform all integrity audits."""

    expected_configs = _expected_configs()
    frozen_implementation = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "source_fingerprint_schema_version": SOURCE_FINGERPRINT_SCHEMA_VERSION,
        "implementation_fingerprint": FROZEN_SOURCE_FINGERPRINT,
        "source_file_count": FROZEN_SOURCE_FILE_COUNT,
    }
    current = implementation_provenance(REPO)
    if current != frozen_implementation:
        raise RuntimeError(f"current ForkWorld source differs from the E19 freeze: {current}")

    directory = _artifact_directory(root)
    children = sorted(path for path in directory.iterdir() if path.is_dir())
    errors: list[str] = []
    if len(children) != len(SEEDS) * len(BRANCHES):
        errors.append(f"materialized run directories={len(children)}, expected exactly 120")
    runs: dict[tuple[int, str], FullRun] = {}
    required = (
        "COMPLETE",
        "resolved_config.yaml",
        "summary.json",
        "metadata.json",
        "status.json",
        "metrics.jsonl",
    )
    for run_dir in children:
        missing = [name for name in required if not (run_dir / name).is_file()]
        if missing:
            errors.append(f"{run_dir.name}: missing {', '.join(missing)}")
            continue
        if (run_dir / "COMPLETE").read_text(encoding="utf-8") != "complete\n":
            errors.append(f"{run_dir.name}: invalid COMPLETE marker")
        config = _load_yaml(run_dir / "resolved_config.yaml")
        summary = _load_json(run_dir / "summary.json")
        metadata = _load_json(run_dir / "metadata.json")
        status = _load_json(run_dir / "status.json")
        seed = int(summary.get("seed", -1))
        branch = str(summary.get("branch", ""))
        if status != {"state": "complete", "run_id": run_dir.name}:
            errors.append(f"{run_dir.name}: invalid completion status")
        if (
            metadata.get("run_id") != run_dir.name
            or metadata.get("seed") != seed
            or metadata.get("cautions") != []
        ):
            errors.append(f"{run_dir.name}: metadata identity/cautions mismatch")
        if metadata.get("implementation") != frozen_implementation:
            errors.append(f"{run_dir.name}: artifact implementation differs from freeze")
        if _expected_run_id(config, metadata) != run_dir.name:
            errors.append(f"{run_dir.name}: artifact identity does not reconstruct")
        if seed not in SEEDS or branch not in BRANCHES:
            errors.append(f"{run_dir.name}: unregistered full key {(seed, branch)!r}")
        if int(config.get("seed", -1)) != seed or get(config, "h16.branch") != branch:
            errors.append(f"{run_dir.name}: config/summary identity mismatch")
        prediction_path = run_dir / "predictions.jsonl"
        if prediction_path.exists() and prediction_path.stat().st_size:
            errors.append(f"{run_dir.name}: saved predictions are forbidden")
        checkpoint_dir = run_dir / "checkpoints"
        if checkpoint_dir.is_dir() and any(checkpoint_dir.iterdir()):
            errors.append(f"{run_dir.name}: saved checkpoints are forbidden")
        metrics, metric_lines = _load_metric_index(
            run_dir / "metrics.jsonl",
            run_id=run_dir.name,
            seed=seed,
            branch=branch,
            errors=errors,
        )
        run = FullRun(run_dir, config, summary, metadata, metrics, metric_lines)
        if branch in expected_configs:
            _validate_run(run, expected_configs[branch], errors)
        key = (seed, branch)
        if key in runs:
            errors.append(f"duplicate full-panel key {key!r}")
        runs[key] = run

    expected_keys = {(seed, branch) for seed in SEEDS for branch in BRANCHES}
    if set(runs) != expected_keys:
        errors.append(
            f"full grid missing={sorted(expected_keys - set(runs))}, "
            f"unexpected={sorted(set(runs) - expected_keys)}"
        )
    metric_counts = {run.metric_lines for run in runs.values()}
    if len(metric_counts) != 1 or metric_counts == {0}:
        errors.append(f"full runs have inconsistent metric line counts: {sorted(metric_counts)}")
    if errors:
        _fail("E19 full artifact audit", errors)
    cross_errors: list[str] = []
    cross = _cross_run_audit(runs, cross_errors)
    if cross_errors:
        _fail("E19 full pairing/reconstruction audit", cross_errors)
    audit = {
        "artifacts_root": str(root.resolve()),
        "complete_artifacts": len(runs),
        "registered_seeds": list(SEEDS),
        "registered_branches": list(BRANCHES),
        "metric_records_audited": sum(run.metric_lines for run in runs.values()),
        "metric_line_count_per_run": next(iter(metric_counts)),
        "source_fingerprint": FROZEN_SOURCE_FINGERPRINT,
        "source_file_count": FROZEN_SOURCE_FILE_COUNT,
        "config_source_sha256": FROZEN_CONFIG_SHA256,
        "scientific_config_fingerprints_by_branch": {
            branch: hashlib.sha256(_canonical(config).encode()).hexdigest()
            for branch, config in expected_configs.items()
        },
        "exact_complete_config_metadata_metric_replay_edit_hash_data_checkpoint_optimizer_and_pairing_audit_passed": True,
        **{key: value for key, value in cross.items() if key != "reconstruction_rows"},
    }
    return runs, {**audit, "reconstruction_rows": cross["reconstruction_rows"]}


def _snapshot_values(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {
        "target_accuracy": float(snapshot["target_accuracy"]),
        "boolean_signature": str(snapshot["boolean_signature"]),
    }
    for goal in GOALS:
        values[f"behavior_{goal}"] = float(get(snapshot, f"behavior.{goal}"))
        values[f"causal_{goal}"] = float(get(snapshot, f"causal.{goal}"))
        values[f"selective_{goal}"] = float(get(snapshot, f"selective_final_hidden_probe.{goal}"))
        values[f"control_{goal}"] = 0.5 * (values[f"behavior_{goal}"] + values[f"causal_{goal}"])
    return values


def eligibility_rows(runs: Mapping[tuple[int, str], FullRun]) -> tuple[list[dict[str, Any]], tuple[int, ...]]:
    rows: list[dict[str, Any]] = []
    eligible: list[int] = []
    for seed in SEEDS:
        anchor = runs[(seed, BRANCHES[0])]
        paired = bool(get(anchor.summary, "eligibility.paired_intersection_eligible"))
        if paired:
            eligible.append(seed)
        row: dict[str, Any] = {
            "seed": seed,
            "paired_intersection_eligible": paired,
            "independent_eligible": bool(
                get(
                    anchor.summary,
                    "eligibility.independent.eligible_both_registered_checkpoints",
                )
            ),
            "nested_eligible": bool(
                get(
                    anchor.summary,
                    "eligibility.nested.eligible_both_registered_checkpoints",
                )
            ),
        }
        for overlap in ("independent", "nested"):
            for step in PHASE_A_STEPS:
                snapshot = cast(
                    Mapping[str, Any],
                    get(anchor.summary, f"prefix_snapshots.{overlap}.{step}"),
                )
                values = _snapshot_values(snapshot)
                row[f"{overlap}_{step}_P_behavior"] = values["behavior_P"]
                row[f"{overlap}_{step}_P_causal"] = values["causal_P"]
                row[f"{overlap}_{step}_Q_selective"] = values["selective_Q"]
        rows.append(row)
    return rows, tuple(eligible)


def run_endpoint_rows(
    runs: Mapping[tuple[int, str], FullRun], eligible_seeds: Sequence[int]
) -> list[dict[str, Any]]:
    eligible = set(eligible_seeds)
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for branch in BRANCHES:
            run = runs[(seed, branch)]
            snapshots = {step: run.snapshot(step) for step in PHASE_B_STEPS}
            postedit = _snapshot_values(run.snapshot(0))
            final = _snapshot_values(run.snapshot(1_024))
            stored_q = float(
                get(
                    run.summary,
                    "outcomes.normalized_control_auc_through_direct_checkpoint.Q",
                )
            )
            recomputed_q = direct_control_auc(snapshots, "Q", horizon=AUC_HORIZON)
            row: dict[str, Any] = {
                "seed": seed,
                "branch": branch,
                "eligible_intersection": seed in eligible,
                "auc128_Q_stored": stored_q,
                "auc128_Q_recomputed": recomputed_q,
                "auc128_Q_recomputation_error": recomputed_q - stored_q,
                "postedit_pure_P": bool(get(run.summary, "postedit_pure_p")),
                "final_target_accuracy": final["target_accuracy"],
                "final_boolean_signature": final["boolean_signature"],
            }
            for goal in GOALS:
                row[f"auc128_{goal}"] = float(
                    get(
                        run.summary,
                        f"outcomes.normalized_control_auc_through_direct_checkpoint.{goal}",
                    )
                )
                for prefix, values in (("postedit", postedit), ("final", final)):
                    for family in ("behavior", "causal", "selective", "control"):
                        row[f"{prefix}_{family}_{goal}"] = values[f"{family}_{goal}"]
            rows.append(row)
    return rows


def checkpoint_rows(
    runs: Mapping[tuple[int, str], FullRun], eligible_seeds: Sequence[int]
) -> list[dict[str, Any]]:
    eligible = set(eligible_seeds)
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for branch in BRANCHES:
            run = runs[(seed, branch)]
            for step in PHASE_B_STEPS:
                row: dict[str, Any] = {
                    "seed": seed,
                    "branch": branch,
                    "local_step": step,
                    "global_step": 45 + step,
                    "eligible_intersection": seed in eligible,
                }
                row.update(_snapshot_values(run.snapshot(step)))
                rows.append(row)
    return rows


def _paired_values(
    endpoints: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    left: str,
    right: str,
    field: str,
) -> NDArray[np.float64]:
    index = {(int(row["seed"]), str(row["branch"])): row for row in endpoints}
    return np.asarray(
        [float(index[(seed, left)][field]) - float(index[(seed, right)][field]) for seed in seeds],
        dtype=np.float64,
    )


REGISTERED_CONTRASTS = (
    (
        "Delta_N",
        "independent_padding_sham",
        "independent_q_restore",
        "co_primary_active_necessity",
    ),
    (
        "Delta_S",
        "nested_q_transplant",
        "nested_padding_sham",
        "co_primary_active_sufficiency",
    ),
    (
        "independent_sham_minus_noop",
        "independent_padding_sham",
        "independent_noop",
        "registered_sham_equivalence",
    ),
    (
        "nested_sham_minus_noop",
        "nested_padding_sham",
        "nested_noop",
        "registered_sham_equivalence",
    ),
)


def paired_contrast_rows(
    endpoints: Sequence[Mapping[str, Any]],
    primary_seeds: Sequence[int],
    eligible_seeds: Sequence[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    populations = [("fixed_all_20", tuple(primary_seeds), True)]
    if eligible_seeds:
        populations.append(("eligibility_intersection_sensitivity", tuple(eligible_seeds), False))
    for population, seeds, primary in populations:
        for name, left, right, role in REGISTERED_CONTRASTS:
            differences = _paired_values(endpoints, seeds, left, right, "auc128_Q")
            result = _bootstrap_paired(
                differences,
                key=f"{population}:{name}:Q_control_auc_0_128",
            )
            rows.append(
                {
                    "population": population,
                    "is_primary_population": primary,
                    "contrast": name,
                    "role": role,
                    "left": left,
                    "right": right,
                    "outcome": "Q_control_auc_0_128",
                    **result,
                }
            )
    return rows


def manipulation_check_rows(
    endpoints: Sequence[Mapping[str, Any]], seeds: Sequence[int]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fields = {"selective_Q": "postedit_selective_Q", **PRESERVATION_FIELDS}
    for branch in EDIT_BRANCHES:
        noop = NOOP_FOR[branch]
        for measure, field in fields.items():
            result = _bootstrap_paired(
                _paired_values(endpoints, seeds, branch, noop, field),
                key=f"manipulation:{branch}:{measure}",
            )
            rows.append(
                {
                    "branch": branch,
                    "noop": noop,
                    "measure": measure,
                    "population": "fixed_all_20",
                    **result,
                }
            )
    index = {(str(row["branch"]), str(row["measure"])): row for row in rows}
    restore = index[("independent_q_restore", "selective_Q")]
    transplant = index[("nested_q_transplant", "selective_Q")]
    restore_pass = float(restore["estimate"]) <= -PROBE_SHIFT and float(restore["ci_high"]) < 0.0
    transplant_pass = float(transplant["estimate"]) >= PROBE_SHIFT and float(transplant["ci_low"]) > 0.0
    sham_q: dict[str, bool] = {}
    for branch in ("independent_padding_sham", "nested_padding_sham"):
        row = index[(branch, "selective_Q")]
        sham_q[branch] = (
            float(row["ci_low"]) >= -PRESERVATION_MARGIN and float(row["ci_high"]) <= PRESERVATION_MARGIN
        )
    preservation: dict[str, dict[str, bool]] = {}
    for branch in EDIT_BRANCHES:
        preservation[branch] = {}
        for measure in PRESERVATION_FIELDS:
            row = index[(branch, measure)]
            preservation[branch][measure] = (
                float(row["ci_low"]) >= -PRESERVATION_MARGIN and float(row["ci_high"]) <= PRESERVATION_MARGIN
            )
    decision = {
        "active_restore_selective_Q_manipulation_valid": restore_pass,
        "active_transplant_selective_Q_manipulation_valid": transplant_pass,
        "sham_selective_Q_equivalence": sham_q,
        "all_sham_selective_Q_intervals_inside_plus_minus_0_05": all(sham_q.values()),
        "preservation_intervals_inside_plus_minus_0_05": preservation,
        "all_P_behavior_P_causal_selective_P_and_selective_Y_preserved": all(
            passed for branch_values in preservation.values() for passed in branch_values.values()
        ),
        "full_manipulation_contract_valid": (
            restore_pass
            and transplant_pass
            and all(sham_q.values())
            and all(passed for branch_values in preservation.values() for passed in branch_values.values())
        ),
        "active_probe_shift_threshold": PROBE_SHIFT,
        "equivalence_and_preservation_margin": PRESERVATION_MARGIN,
        "uses_all_20_registered_seeds": len(seeds) == 20,
    }
    return rows, decision


def interpretation_decisions(
    contrasts: Sequence[Mapping[str, Any]],
    manipulation: Mapping[str, Any],
    *,
    eligible_count: int,
) -> dict[str, Any]:
    primary = {str(row["contrast"]): row for row in contrasts if row["population"] == "fixed_all_20"}
    active: dict[str, bool] = {}
    for name in ("Delta_N", "Delta_S"):
        row = primary[name]
        active[name] = (
            float(row["estimate"]) >= PRIMARY_EFFECT
            and float(row["ci_low"]) > 0.0
            and int(row["positive_count"]) >= MINIMUM_SIGN_COUNT
            and int(row["n_seeds"]) == 20
        )
    sham: dict[str, bool] = {}
    for name in ("independent_sham_minus_noop", "nested_sham_minus_noop"):
        row = primary[name]
        sham[name] = (
            float(row["ci_low"]) >= -SHAM_AUC_MARGIN
            and float(row["ci_high"]) <= SHAM_AUC_MARGIN
            and int(row["n_seeds"]) == 20
        )
    bidirectional = (
        all(active.values()) and all(sham.values()) and bool(manipulation["full_manipulation_contract_valid"])
    )
    narrow_claim = (
        "The registered paired intervention supports a bidirectional contribution "
        "of the active first-layer Q input columns to later Q control."
        if bidirectional
        else "The registered criteria do not support the bidirectional active-column claim."
    )
    return {
        "co_primary_active_criteria": active,
        "both_co_primary_active_criteria_met": all(active.values()),
        "registered_sham_auc_equivalence": sham,
        "both_registered_sham_auc_equivalence_criteria_met": all(sham.values()),
        "manipulation_contract_valid": bool(manipulation["full_manipulation_contract_valid"]),
        "bidirectional_registered_criteria_met": bidirectional,
        "eligible_intersection_count": eligible_count,
        "minimum_eligible_intersection": MINIMUM_ELIGIBLE,
        "eligible_intersection_at_least_15": eligible_count >= MINIMUM_ELIGIBLE,
        "bidirectional_handoff_sensitivity_interpretation_allowed": (
            bidirectional and eligible_count >= MINIMUM_ELIGIBLE
        ),
        "primary_population": "all 20 registered seeds, fixed before outcomes",
        "eligibility_never_filters_primary": True,
        "allowed_interpretation": narrow_claim,
        "scope_guard": (
            "This intervention identifies the active first-layer Q input-pathway "
            "contribution only; it is not a mediation analysis and does not identify "
            "every representation or circuit that can encode Q."
        ),
        "primary_effect_threshold": PRIMARY_EFFECT,
        "minimum_positive_seed_count": MINIMUM_SIGN_COUNT,
        "sham_auc_equivalence_margin": SHAM_AUC_MARGIN,
    }


def analyze_runs(
    runs: Mapping[tuple[int, str], FullRun],
) -> tuple[dict[str, Any], dict[str, Sequence[Mapping[str, Any]]]]:
    eligibility, eligible_seeds = eligibility_rows(runs)
    endpoints = run_endpoint_rows(runs, eligible_seeds)
    checkpoints = checkpoint_rows(runs, eligible_seeds)
    contrasts = paired_contrast_rows(endpoints, SEEDS, eligible_seeds)
    manipulation_rows, manipulation = manipulation_check_rows(endpoints, SEEDS)
    decisions = interpretation_decisions(contrasts, manipulation, eligible_count=len(eligible_seeds))
    analysis = {
        "experiment": "E19 frozen active first-layer Q-input-pathway intervention",
        "inference_status": "frozen_adaptive_posthoc_full_panel",
        "registered_seed_count": len(SEEDS),
        "registered_arm_count": len(BRANCHES),
        "eligibility": {
            "intersection_seeds": list(eligible_seeds),
            "intersection_count": len(eligible_seeds),
            "minimum_for_handoff_sensitivity": MINIMUM_ELIGIBLE,
            "meets_minimum": len(eligible_seeds) >= MINIMUM_ELIGIBLE,
            "primary_analysis_filtered": False,
        },
        "manipulation": manipulation,
        "decisions": decisions,
        "auc_recomputation": {
            "horizon": AUC_HORIZON,
            "stored_and_independent_direct_snapshot_calculations_match": all(
                math.isclose(
                    float(row["auc128_Q_recomputation_error"]),
                    0.0,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                for row in endpoints
            ),
            "maximum_absolute_Q_error": max(
                abs(float(row["auc128_Q_recomputation_error"])) for row in endpoints
            ),
        },
    }
    tables: dict[str, Sequence[Mapping[str, Any]]] = {
        "eligibility": eligibility,
        "endpoints": endpoints,
        "checkpoints": checkpoints,
        "contrasts": contrasts,
        "manipulation": manipulation_rows,
    }
    return analysis, tables


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


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _text_report(
    audit: Mapping[str, Any],
    analysis: Mapping[str, Any],
    contrasts: Sequence[Mapping[str, Any]],
) -> str:
    decision = cast(Mapping[str, Any], analysis["decisions"])
    manipulation = cast(Mapping[str, Any], analysis["manipulation"])
    primary = [row for row in contrasts if row["population"] == "fixed_all_20"]
    lines = [
        "E19 frozen full-panel analysis",
        "======================================",
        "",
        "Integrity",
        f"- Complete artifacts audited: {audit['complete_artifacts']}/120",
        f"- Frozen source fingerprint: {audit['source_fingerprint']}",
        f"- Frozen config SHA-256: {audit['config_source_sha256']}",
        f"- Sham displacement norm tolerance: {audit['sham_displacement_norm_tolerance']}",
        "- Exact config, metadata, replay, edit, hash, data, checkpoint, optimizer, and pairing audits passed.",
        "",
        "Registered all-seed Q-control AUC contrasts (0--128 updates)",
    ]
    for row in primary:
        lines.append(
            f"- {row['contrast']}: mean={float(row['estimate']):.6f}, "
            f"95% bootstrap CI [{float(row['ci_low']):.6f}, "
            f"{float(row['ci_high']):.6f}], positive seeds="
            f"{row['positive_count']}/20"
        )
    lines.extend(
        [
            "",
            "Manipulation and decision",
            f"- Full post-edit manipulation contract valid: {manipulation['full_manipulation_contract_valid']}",
            f"- Eligibility intersection: {decision['eligible_intersection_count']}/20 "
            f"(minimum {decision['minimum_eligible_intersection']})",
            f"- Bidirectional registered criteria met: {decision['bidirectional_registered_criteria_met']}",
            f"- {decision['allowed_interpretation']}",
            f"- Scope: {decision['scope_guard']}",
            "",
            "The fixed primary population is all 20 registered seeds. Eligibility is shown only as a sensitivity analysis.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(
    output_dir: Path,
    audit: Mapping[str, Any],
    analysis: Mapping[str, Any],
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "e19_full_audit.json", audit)
    analysis_payload = {
        **dict(analysis),
        "registered_contrasts": list(tables["contrasts"]),
        "manipulation_checks": list(tables["manipulation"]),
    }
    _write_json(output_dir / "e19_full_analysis.json", analysis_payload)
    _write_csv(output_dir / "e19_full_run_endpoints.csv", tables["endpoints"])
    _write_csv(output_dir / "e19_full_paired_contrasts.csv", tables["contrasts"])
    _write_csv(output_dir / "e19_full_manipulation_checks.csv", tables["manipulation"])
    _write_csv(output_dir / "e19_full_eligibility.csv", tables["eligibility"])
    _write_csv(output_dir / "e19_full_checkpoint_dynamics.csv", tables["checkpoints"])
    (output_dir / "e19_full_report.txt").write_text(
        _text_report(audit, analysis, tables["contrasts"]), encoding="utf-8"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    runs, audit = load_and_audit(args.artifacts)
    analysis, tables = analyze_runs(runs)
    write_outputs(args.output_dir, audit, analysis, tables)
    decision = cast(Mapping[str, Any], analysis["decisions"])
    print(
        "E19 FULL: 120/120 exact audits passed; "
        f"bidirectional criteria={decision['bidirectional_registered_criteria_met']}; "
        f"eligible sensitivity n={decision['eligible_intersection_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
