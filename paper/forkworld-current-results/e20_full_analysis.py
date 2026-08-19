#!/usr/bin/env python3
"""Frozen adaptive/post-hoc full-panel analyzer for Forkworld E20.

The registered independent unit is a seed.  All schedule, checkpoint, raw-row,
and intervention measurements are paired repeated observations.  This module
accepts only the exact frozen 20-seed by six-schedule grid and fails closed on
identity, data, reset, replay, pairing, checkpoint, and metric drift before it
constructs any scientific estimate.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np
import yaml  # type: ignore[import-untyped]
from numpy.typing import NDArray

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forkworld.artifacts import implementation_provenance  # noqa: E402
from forkworld.config import expand_sweep, load_config  # noqa: E402
from forkworld.counterbalanced_order import (  # noqa: E402
    audit_counterbalanced_atomic_plan,
    audit_counterbalanced_bundle,
    audit_counterbalanced_factorial_panel,
    audit_schedule_multiset_equality,
    make_all_counterbalanced_atomic_plans,
    make_counterbalanced_bundle,
    make_counterbalanced_factorial_panel,
)
from forkworld.handoff import semantic_batch_digest  # noqa: E402
from forkworld.protocols_counterbalanced import replay_h17_observer_free  # noqa: E402

DESIGN_MEMO = HERE / "e20_repaired_order_design.md"
DESIGN_MEMO_SHA256 = "1386cb401aef8ed187389dc84aa210b09b564436bc16c9246bbd22fce257529a"
GATE_ANALYZER_PATH = HERE / "e20_pilot_gate.py"
CONFIG_PATH = REPO / "configs" / "e20_counterbalanced_order.yaml"
LAUNCHER_PATH = REPO / "runs" / "21_counterbalanced_order.sh"
DEFAULT_ARTIFACTS = REPO / "artifacts-e20"
DEFAULT_OUTPUT = HERE / "derived"
DEFAULT_PILOT_GATE_RECORD = DEFAULT_OUTPUT / "e20_pilot_gate.json"
EXPERIMENT = "counterbalanced_identical_evidence_order"
HYPOTHESIS = "h17"

SEEDS = (
    577,
    587,
    593,
    599,
    601,
    607,
    613,
    617,
    619,
    631,
    641,
    643,
    647,
    653,
    659,
    661,
    673,
    677,
    683,
    691,
)
PILOT_SEEDS = (563, 569, 571)
SCHEDULE_GOALS: Mapping[str, tuple[str, str, str]] = {
    "p_q_y": ("P", "Q", "Y"),
    "p_y_q": ("P", "Y", "Q"),
    "q_p_y": ("Q", "P", "Y"),
    "q_y_p": ("Q", "Y", "P"),
    "y_p_q": ("Y", "P", "Q"),
    "y_q_p": ("Y", "Q", "P"),
}
SCHEDULES = tuple(SCHEDULE_GOALS)
GOALS = ("P", "Q", "Y")
NAMED_RULES = (*GOALS, "M")
FEATURE_NAMES = (
    "P",
    "P_present",
    "R_1",
    "R_2",
    "R_3",
    "Q_present",
    "Q_1",
    "Q_2",
)
LOCAL_CHECKPOINTS = (
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
    256,
)
PRIMARY_STEPS = (33, 45, 62, 85, 117, 128)
EARLY_STEPS = (0, 1, 2, 3, 4, 5, 7, 9, 13, 17, 24, 33)
LATE_STABILITY_STEPS = (161, 222, 256)
GATE_OFFSETS = (161, 222, 256)
GATE_LEVEL = 0.90
GATE_MARGIN = 0.10
MINIMUM_JOINT_SEEDS = 2
BLOCK_STEPS = 256
TOTAL_STEPS = 1_024
BATCH_SIZE = 288
COMPONENT_ROWS = 9_216
COMPONENT_SAMPLES = 73_728
TOTAL_SAMPLES = 294_912
BOOTSTRAP_DRAWS = 4_000
PRIMARY_MATERIAL_MEAN = 0.10
PRACTICAL_BOUNDARY = 0.05
PREVALENCE_COUNT = 15
MATERIAL_DIRECTION = 0.10
ACQUISITION_LEVEL = 0.90
ACQUISITION_MARGIN = 0.10
PROBE_ADVANTAGE = 0.20
PROBE_LEVEL = 0.90
DATA_SEED = 171_000_001
STREAM_SEED = 171_000_002
PHASE_SEEDS = {
    "component_P": 171_000_101,
    "component_Q": 171_000_102,
    "component_Y": 171_000_103,
    "washout": 171_000_104,
}
ARTIFACT_SCHEMA_VERSION = 1
SOURCE_FINGERPRINT_SCHEMA_VERSION = 1
FOLD_DIGEST = "25d5e9584a2ea74d42d48a673645a3b34c1b76e75e8230bf5722985e66968ec7"
CONTROL_DIGEST = "c7a186308102e807e594fd76b3fc5e7ef1678314a95de067b6fd347b390e53d0"
CONTROL_BITS = "0110001001011011111001011000100101011110100110000010011010110101"
NON_SCIENTIFIC_RUN_FIELDS = frozenset({"output_root", "resume", "seeds"})

# These strings are part of the prospective analysis freeze.  Changing a key
# changes the deterministic resampling stream and therefore requires a new
# externally recorded analyzer hash before any scientific artifact exists.
BOOTSTRAP_KEYS: Mapping[str, str] = {
    "primary_dispersion": "e20:v1:bootstrap:primary:hamming-dispersion-auc33-128",
    "signed_recency": "e20:v1:bootstrap:hierarchical:signed-recency-auc33-128",
    "goal_P_recency": "e20:v1:bootstrap:hierarchical:P-position-auc33-128",
    "goal_Q_recency": "e20:v1:bootstrap:hierarchical:Q-position-auc33-128",
    "goal_Y_recency": "e20:v1:bootstrap:hierarchical:Y-position-auc33-128",
    "early_dispersion": "e20:v1:bootstrap:secondary:dispersion-auc0-33",
    "terminal_dispersion": "e20:v1:bootstrap:secondary:dispersion-at256",
    "curvature_P": "e20:v1:bootstrap:secondary:P-middle-curvature-auc33-128",
    "curvature_Q": "e20:v1:bootstrap:secondary:Q-middle-curvature-auc33-128",
    "curvature_Y": "e20:v1:bootstrap:secondary:Y-middle-curvature-auc33-128",
    "middle_first_P": "e20:v1:bootstrap:secondary:P-middle-minus-first-auc33-128",
    "middle_first_Q": "e20:v1:bootstrap:secondary:Q-middle-minus-first-auc33-128",
    "middle_first_Y": "e20:v1:bootstrap:secondary:Y-middle-minus-first-auc33-128",
    "last_middle_P": "e20:v1:bootstrap:secondary:P-last-minus-middle-auc33-128",
    "last_middle_Q": "e20:v1:bootstrap:secondary:Q-last-minus-middle-auc33-128",
    "last_middle_Y": "e20:v1:bootstrap:secondary:Y-last-minus-middle-auc33-128",
    "behavior_position_P": "e20:v1:bootstrap:secondary:P-behavior-last-minus-first-auc33-128",
    "behavior_position_Q": "e20:v1:bootstrap:secondary:Q-behavior-last-minus-first-auc33-128",
    "behavior_position_Y": "e20:v1:bootstrap:secondary:Y-behavior-last-minus-first-auc33-128",
    "causal_position_P": "e20:v1:bootstrap:secondary:P-causal-last-minus-first-auc33-128",
    "causal_position_Q": "e20:v1:bootstrap:secondary:Q-causal-last-minus-first-auc33-128",
    "causal_position_Y": "e20:v1:bootstrap:secondary:Y-causal-last-minus-first-auc33-128",
}

# Patched only after implementation source/config/launcher freeze, before pilot.
FROZEN_SOURCE_FINGERPRINT = "4ca022c5b2d2d9a75c8d443cdc0e422d178989bae825c2c2d37d41fc7a14e539"
FROZEN_SOURCE_FILE_COUNT = 34
FROZEN_CONFIG_SHA256 = "3f680421f4f43b71e139feec254f38c58eac842b62aa610f91d208d6adfa0de7"
FROZEN_LAUNCHER_SHA256 = "33d98c4115960a3260c6329af88b5d33aea3b64153cbc747f2ab3ace8875999a"
FROZEN_PILOT_CONFIG_SHA256 = "868981c1f250c161ef22ff44ad5f3dfa33cd564a3f1169e194e4f20cffc2aefc"
FROZEN_GATE_ANALYZER_SHA256 = "faaafa98f08eed830cfed4ec1c126d8e4f9c3079f91f8ab0cd27a99eaded3feb"

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


def _numeric_values_finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_numeric_values_finite(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return all(_numeric_values_finite(item) for item in value)
    if isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)):
        return math.isfinite(float(value))
    return True


def _fail(name: str, errors: Sequence[str]) -> None:
    preview = "\n".join(f"  - {error}" for error in errors[:100])
    suffix = "" if len(errors) <= 100 else f"\n  ... and {len(errors) - 100} more"
    raise RuntimeError(f"{name} failed with {len(errors)} issue(s):\n{preview}{suffix}")


def _expect(errors: list[str], run_id: str, actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        errors.append(f"{run_id}: {label}={actual!r}, expected {expected!r}")


def _load_json(path: Path) -> Mapping[str, Any]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicates)
    except (OSError, json.JSONDecodeError, ValueError) as error:
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


def reconstruct_fold_and_control() -> dict[str, Any]:
    """Independently reconstruct the memo-frozen folds and control labels."""

    records: list[dict[str, int]] = []
    tuple_members: dict[int, list[int]] = defaultdict(list)
    rules: dict[int, dict[str, int]] = {}
    for raw_id in range(64):
        signs = [1 if raw_id & (1 << shift) else -1 for shift in (5, 4, 3, 2, 1, 0)]
        p, r1, r2, r3, q1, q2 = signs
        q = q1 * q2
        y = r1 * r2 * r3
        majority = 1 if p + q + y > 0 else -1
        tuple_id = 4 * int(p > 0) + 2 * int(q > 0) + int(y > 0)
        tuple_members[tuple_id].append(raw_id)
        rules[raw_id] = {"P": p, "Q": q, "Y": y, "M": majority, "tuple_id": tuple_id}
    folds = [0] * 64
    for tuple_id, raw_ids in tuple_members.items():
        if len(raw_ids) != 8:
            raise RuntimeError(f"tuple {tuple_id} has {len(raw_ids)} rather than eight codewords")
        for fold, raw_id in enumerate(sorted(raw_ids)):
            folds[raw_id] = fold
            records.append({"fold": fold, "raw_id": raw_id, "tuple_id": tuple_id})
    records.sort(key=lambda row: row["raw_id"])
    control = [
        1 if (folds[raw_id] - rules[raw_id]["tuple_id"]) % 8 in {0, 1, 3, 4} else -1 for raw_id in range(64)
    ]
    fold_digest = hashlib.sha256(_canonical(records).encode()).hexdigest()
    control_digest = hashlib.sha256(_canonical(control).encode()).hexdigest()
    control_bits = "".join("1" if value > 0 else "0" for value in control)
    errors: list[str] = []
    if fold_digest != FOLD_DIGEST:
        errors.append(f"fold digest {fold_digest} differs from freeze")
    if control_digest != CONTROL_DIGEST:
        errors.append(f"control digest {control_digest} differs from freeze")
    if control_bits != CONTROL_BITS:
        errors.append("control bit string differs from freeze")
    if Counter(folds) != Counter({fold: 8 for fold in range(8)}):
        errors.append("folds do not contain eight raw codewords each")
    if Counter(control) != Counter({-1: 32, 1: 32}):
        errors.append("control is not sign balanced")
    for tuple_id, raw_ids in tuple_members.items():
        if Counter(folds[raw_id] for raw_id in raw_ids) != Counter(range(8)):
            errors.append(f"tuple {tuple_id} does not span all folds")
        if Counter(control[raw_id] for raw_id in raw_ids) != Counter({-1: 4, 1: 4}):
            errors.append(f"control is not balanced within tuple {tuple_id}")
    for fold in range(8):
        selected = [raw_id for raw_id in range(64) if folds[raw_id] == fold]
        if Counter(control[raw_id] for raw_id in selected) != Counter({-1: 4, 1: 4}):
            errors.append(f"control is not balanced within fold {fold}")
    for goal in NAMED_RULES:
        agreement = sum(control[raw_id] == rules[raw_id][goal] for raw_id in range(64))
        if agreement != 32:
            errors.append(f"control agrees with {goal} on {agreement}/64 rows")
    if errors:
        _fail("E20 fold/control reconstruction", errors)
    return {
        "records": records,
        "folds": folds,
        "control": control,
        "control_bits": control_bits,
        "fold_digest": fold_digest,
        "control_digest": control_digest,
        "rules": rules,
        "all_balance_constraints_verified": True,
    }


FROZEN_PANEL = reconstruct_fold_and_control()
EXPECTED_SIGNATURES: Mapping[str, str] = {
    goal: "".join(
        "1" if cast(Mapping[str, int], FROZEN_PANEL["rules"])[raw_id][goal] > 0 else "0"
        for raw_id in range(64)
    )
    for goal in NAMED_RULES
}


def _normalize_signature(value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64 or set(value) - {"0", "1"}:
        raise ValueError("hard signature must be an ordered 64-character 0/1 string")
    return value


def classify_truth_table(signature: str) -> str:
    """Apply the registered hierarchical truth-table taxonomy."""

    value = _normalize_signature(signature)
    for goal in NAMED_RULES:
        if value == EXPECTED_SIGNATURES[goal]:
            return goal
    actions = np.asarray([1 if bit == "1" else -1 for bit in value], dtype=np.int8)
    rules = cast(Mapping[int, Mapping[str, int]], FROZEN_PANEL["rules"])
    for tuple_id in range(8):
        indices = [raw_id for raw_id in range(64) if rules[raw_id]["tuple_id"] == tuple_id]
        if len(set(actions[indices].tolist())) != 1:
            return "raw_codeword_specific_composite"
    return "other_candidate_tuple_consistent"


def hamming_distance(left: str, right: str) -> float:
    a = _normalize_signature(left)
    b = _normalize_signature(right)
    return sum(x != y for x, y in zip(a, b, strict=True)) / 64.0


def schedule_dispersion(signatures: Mapping[str, str]) -> float:
    if set(signatures) != set(SCHEDULES):
        raise ValueError("dispersion requires the exact six schedules")
    distances = [
        hamming_distance(signatures[left], signatures[right])
        for index, left in enumerate(SCHEDULES)
        for right in SCHEDULES[index + 1 :]
    ]
    if len(distances) != 15:
        raise RuntimeError("six schedules did not yield fifteen pairs")
    result = float(np.mean(distances))
    if not 0.0 <= result <= 0.60 + 1e-12:
        raise RuntimeError(f"binary six-schedule dispersion outside [0,.60]: {result}")
    return result


def direct_auc(
    values: Mapping[int, float],
    *,
    steps: Sequence[int],
    start: int,
    stop: int,
) -> float:
    """Trapezoidal AUC from an exact, directly observed checkpoint lattice."""

    wanted = tuple(int(step) for step in steps)
    if tuple(sorted(set(wanted))) != wanted or wanted[0] != start or wanted[-1] != stop:
        raise ValueError("AUC steps must be unique, increasing, and include both boundaries")
    if set(values) != set(wanted):
        raise ValueError(f"AUC requires exactly direct checkpoints {wanted}; observed {sorted(values)}")
    y = np.asarray([float(values[step]) for step in wanted], dtype=np.float64)
    if not np.all(np.isfinite(y)):
        raise ValueError("AUC values must be finite")
    return float(np.trapezoid(y, np.asarray(wanted, dtype=np.float64)) / (stop - start))


def _bootstrap_draws(values: NDArray[np.float64], *, key: str) -> NDArray[np.float64]:
    if values.ndim != 1 or len(values) != 20 or not np.all(np.isfinite(values)):
        raise ValueError(f"{key} requires exactly 20 finite seed values")
    digest = hashlib.blake2b(key.encode(), digest_size=8, person=b"forke20")
    rng = np.random.default_rng(int.from_bytes(digest.digest(), "little"))
    indices = rng.integers(0, 20, size=(BOOTSTRAP_DRAWS, 20))
    return np.mean(values[indices], axis=1)


def bootstrap_summary(
    values: Sequence[float],
    *,
    key: str,
    confidence: float = 0.95,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    draws = _bootstrap_draws(array, key=key)
    alpha = 1.0 - confidence
    low, high = np.quantile(draws, (alpha / 2.0, 1.0 - alpha / 2.0))
    return {
        "estimate": float(np.mean(array)),
        "ci_low": float(low),
        "ci_high": float(high),
        "confidence": confidence,
        "n_seeds": 20,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_key": key,
        "positive_count": int(np.sum(array > 0.0)),
        "negative_count": int(np.sum(array < 0.0)),
        "zero_count": int(np.sum(array == 0.0)),
        "at_least_0_05_count": int(np.sum(array >= PRACTICAL_BOUNDARY)),
        "values": array.tolist(),
    }


def primary_decision(summary: Mapping[str, Any]) -> str:
    material = (
        float(summary["estimate"]) >= PRIMARY_MATERIAL_MEAN
        and float(summary["ci_low"]) > PRACTICAL_BOUNDARY
        and int(summary["at_least_0_05_count"]) >= PREVALENCE_COUNT
    )
    equivalent = float(summary["ci_high"]) <= PRACTICAL_BOUNDARY
    if material:
        return "material_persistent_order_dependence"
    if equivalent:
        return "practical_behavioral_equivalence"
    return "inconclusive_on_registered_persistence_scale"


def direction_decision(summary: Mapping[str, Any], *, formal_confidence: float = 0.95) -> str:
    if not math.isclose(float(summary["confidence"]), formal_confidence, abs_tol=1e-12):
        raise ValueError("direction decision received the wrong formal confidence level")
    estimate = float(summary["estimate"])
    low = float(summary["ci_low"])
    high = float(summary["ci_high"])
    if estimate >= MATERIAL_DIRECTION and low > 0.0 and int(summary["positive_count"]) >= 15:
        return "material_recency"
    if estimate <= -MATERIAL_DIRECTION and high < 0.0 and int(summary["negative_count"]) >= 15:
        return "material_primacy"
    if low >= -PRACTICAL_BOUNDARY and high <= PRACTICAL_BOUNDARY:
        return "practical_directional_equivalence"
    return "mixed_or_inconclusive_direction"


def _candidate_margin(snapshot: Mapping[str, Any], goal: str) -> float:
    others = tuple(candidate for candidate in GOALS if candidate != goal)
    behavior = cast(Mapping[str, Any], snapshot["behavior"])
    causal = cast(Mapping[str, Any], snapshot["causal"])
    return 0.5 * (
        float(behavior[goal])
        - max(float(behavior[other]) for other in others)
        + float(causal[goal])
        - max(float(causal[other]) for other in others)
    )


def _mean_pairwise_absolute(vectors: Mapping[str, Sequence[float]]) -> float:
    if set(vectors) != set(SCHEDULES):
        raise ValueError("pairwise dispersion requires all schedules")
    arrays = {name: np.asarray(value, dtype=np.float64) for name, value in vectors.items()}
    shapes = {array.shape for array in arrays.values()}
    if len(shapes) != 1 or not all(np.all(np.isfinite(array)) for array in arrays.values()):
        raise ValueError("pairwise vectors must have one common finite shape")
    return float(
        np.mean(
            [
                np.mean(np.abs(arrays[left] - arrays[right]))
                for index, left in enumerate(SCHEDULES)
                for right in SCHEDULES[index + 1 :]
            ]
        )
    )


def seed_washout_outcomes(snapshots: Mapping[str, Mapping[int, Mapping[str, Any]]]) -> dict[str, Any]:
    """Construct every registered seed-level washout estimand from snapshots."""

    if set(snapshots) != set(SCHEDULES):
        raise ValueError("seed outcome requires all six schedules")
    for schedule in SCHEDULES:
        if set(snapshots[schedule]) != set(LOCAL_CHECKPOINTS):
            raise ValueError(f"{schedule} washout checkpoint lattice differs from freeze")
    time_rows: list[dict[str, Any]] = []
    dispersion: dict[int, float] = {}
    recency: dict[int, float] = {}
    per_goal_recency: dict[str, dict[int, float]] = {goal: {} for goal in GOALS}
    curvature: dict[str, dict[int, float]] = {goal: {} for goal in GOALS}
    middle_minus_first: dict[str, dict[int, float]] = {goal: {} for goal in GOALS}
    last_minus_middle: dict[str, dict[int, float]] = {goal: {} for goal in GOALS}
    behavior_position: dict[str, dict[int, float]] = {goal: {} for goal in GOALS}
    causal_position: dict[str, dict[int, float]] = {goal: {} for goal in GOALS}
    probability_dispersion: dict[int, float] = {}
    control_vector_dispersion: dict[int, float] = {}
    margin_trajectories: dict[str, dict[str, dict[int, float]]] = {
        schedule: {goal: {} for goal in GOALS} for schedule in SCHEDULES
    }
    for step in LOCAL_CHECKPOINTS:
        at_step = {schedule: snapshots[schedule][step] for schedule in SCHEDULES}
        signatures = {
            schedule: _normalize_signature(str(snapshot["raw_hard_signature"]))
            for schedule, snapshot in at_step.items()
        }
        dispersion[step] = schedule_dispersion(signatures)
        margins = {
            schedule: {goal: _candidate_margin(snapshot, goal) for goal in GOALS}
            for schedule, snapshot in at_step.items()
        }
        for schedule in SCHEDULES:
            for goal in GOALS:
                margin_trajectories[schedule][goal][step] = margins[schedule][goal]
        recency[step] = float(
            np.mean(
                [
                    margins[schedule][SCHEDULE_GOALS[schedule][-1]]
                    - margins[schedule][SCHEDULE_GOALS[schedule][0]]
                    for schedule in SCHEDULES
                ]
            )
        )
        for goal in GOALS:
            by_position = {
                position: [
                    margins[schedule][goal]
                    for schedule in SCHEDULES
                    if SCHEDULE_GOALS[schedule][position - 1] == goal
                ]
                for position in (1, 2, 3)
            }
            if any(len(values) != 2 for values in by_position.values()):
                raise RuntimeError(f"goal {goal} is not twice represented at each position")
            position_mean = {position: float(np.mean(values)) for position, values in by_position.items()}
            per_goal_recency[goal][step] = position_mean[3] - position_mean[1]
            curvature[goal][step] = position_mean[2] - 0.5 * (position_mean[1] + position_mean[3])
            middle_minus_first[goal][step] = position_mean[2] - position_mean[1]
            last_minus_middle[goal][step] = position_mean[3] - position_mean[2]
            for family, target in (
                ("behavior", behavior_position),
                ("causal", causal_position),
            ):
                family_position = {
                    position: float(
                        np.mean(
                            [
                                float(get(at_step[schedule], f"{family}.{goal}"))
                                for schedule in SCHEDULES
                                if SCHEDULE_GOALS[schedule][position - 1] == goal
                            ]
                        )
                    )
                    for position in (1, 3)
                }
                target[goal][step] = family_position[3] - family_position[1]
        probability_dispersion[step] = _mean_pairwise_absolute(
            {
                schedule: cast(Sequence[float], snapshot["raw_probability_table"])
                for schedule, snapshot in at_step.items()
            }
        )
        control_vector_dispersion[step] = _mean_pairwise_absolute(
            {
                schedule: [
                    *[float(get(snapshot, f"behavior.{goal}")) for goal in GOALS],
                    *[float(get(snapshot, f"causal.{goal}")) for goal in GOALS],
                ]
                for schedule, snapshot in at_step.items()
            }
        )
        time_rows.append(
            {
                "washout_offset": step,
                "hamming_dispersion": dispersion[step],
                "signed_recency_margin": recency[step],
                "probability_dispersion": probability_dispersion[step],
                "candidate_control_vector_dispersion": control_vector_dispersion[step],
                **{f"{goal}_last_minus_first_margin": per_goal_recency[goal][step] for goal in GOALS},
            }
        )
    primary = direct_auc(
        {step: dispersion[step] for step in PRIMARY_STEPS},
        steps=PRIMARY_STEPS,
        start=33,
        stop=128,
    )
    signed = direct_auc(
        {step: recency[step] for step in PRIMARY_STEPS},
        steps=PRIMARY_STEPS,
        start=33,
        stop=128,
    )
    goal_auc = {
        goal: direct_auc(
            {step: per_goal_recency[goal][step] for step in PRIMARY_STEPS},
            steps=PRIMARY_STEPS,
            start=33,
            stop=128,
        )
        for goal in GOALS
    }
    if not math.isclose(signed, sum(goal_auc.values()) / 3.0, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("signed recency AUC violates its per-goal algebraic identity")
    schedule_arm_rows: list[dict[str, Any]] = []
    for schedule in SCHEDULES:
        order = SCHEDULE_GOALS[schedule]
        goal_aucs: dict[str, float] = {}
        for goal in GOALS:
            margin_auc = direct_auc(
                {step: margin_trajectories[schedule][goal][step] for step in PRIMARY_STEPS},
                steps=PRIMARY_STEPS,
                start=33,
                stop=128,
            )
            goal_aucs[goal] = margin_auc
            schedule_arm_rows.append(
                {
                    "schedule": schedule,
                    "goal": goal,
                    "position": order.index(goal) + 1,
                    "candidate_control_margin_auc_33_128": margin_auc,
                    "behavior_auc_33_128": direct_auc(
                        {
                            step: float(get(snapshots[schedule][step], f"behavior.{goal}"))
                            for step in PRIMARY_STEPS
                        },
                        steps=PRIMARY_STEPS,
                        start=33,
                        stop=128,
                    ),
                    "causal_auc_33_128": direct_auc(
                        {
                            step: float(get(snapshots[schedule][step], f"causal.{goal}"))
                            for step in PRIMARY_STEPS
                        },
                        steps=PRIMARY_STEPS,
                        start=33,
                        stop=128,
                    ),
                }
            )
        first, last = order[0], order[-1]
        schedule_contrast = goal_aucs[last] - goal_aucs[first]
        for row in schedule_arm_rows[-3:]:
            row["schedule_last_minus_first_margin_auc_33_128"] = schedule_contrast
    return {
        "primary_hamming_dispersion_auc_33_128": primary,
        "signed_recency_auc_33_128": signed,
        "per_goal_recency_auc_33_128": goal_auc,
        "early_hamming_dispersion_auc_0_33": direct_auc(
            {step: dispersion[step] for step in EARLY_STEPS},
            steps=EARLY_STEPS,
            start=0,
            stop=33,
        ),
        "terminal_hamming_dispersion_256": dispersion[256],
        "curvature_auc_33_128": {
            goal: direct_auc(
                {step: curvature[goal][step] for step in PRIMARY_STEPS},
                steps=PRIMARY_STEPS,
                start=33,
                stop=128,
            )
            for goal in GOALS
        },
        "middle_minus_first_auc_33_128": {
            goal: direct_auc(
                {step: middle_minus_first[goal][step] for step in PRIMARY_STEPS},
                steps=PRIMARY_STEPS,
                start=33,
                stop=128,
            )
            for goal in GOALS
        },
        "last_minus_middle_auc_33_128": {
            goal: direct_auc(
                {step: last_minus_middle[goal][step] for step in PRIMARY_STEPS},
                steps=PRIMARY_STEPS,
                start=33,
                stop=128,
            )
            for goal in GOALS
        },
        "behavior_last_minus_first_auc_33_128": {
            goal: direct_auc(
                {step: behavior_position[goal][step] for step in PRIMARY_STEPS},
                steps=PRIMARY_STEPS,
                start=33,
                stop=128,
            )
            for goal in GOALS
        },
        "causal_last_minus_first_auc_33_128": {
            goal: direct_auc(
                {step: causal_position[goal][step] for step in PRIMARY_STEPS},
                steps=PRIMARY_STEPS,
                start=33,
                stop=128,
            )
            for goal in GOALS
        },
        "schedule_arm_rows": schedule_arm_rows,
        "time_rows": time_rows,
    }


def _persistent_event(
    snapshots: Mapping[int, Mapping[str, Any]],
    predicate: Callable[[Mapping[str, Any]], bool],
) -> dict[str, Any]:
    steps = tuple(sorted(snapshots))
    if steps != LOCAL_CHECKPOINTS:
        raise ValueError("event construction requires the exact direct checkpoint lattice")
    qualified = [bool(predicate(snapshots[step])) for step in steps]
    for index in range(1, len(steps)):
        if qualified[index - 1] and qualified[index]:
            return {
                "observed": True,
                "first_qualifying_offset": steps[index - 1],
                "confirmation_offset": steps[index],
                "censor_offset": 256,
            }
    return {
        "observed": False,
        "first_qualifying_offset": None,
        "confirmation_offset": None,
        "censor_offset": 256,
    }


def _goal_dominates(values: Mapping[str, Any], goal: str) -> bool:
    return bool(
        float(values[goal]) >= ACQUISITION_LEVEL
        and all(
            float(values[goal]) - float(values[other]) >= ACQUISITION_MARGIN
            for other in GOALS
            if other != goal
        )
    )


def _pure_goal(snapshot: Mapping[str, Any]) -> str | None:
    pure = [
        goal
        for goal in GOALS
        if _goal_dominates(cast(Mapping[str, Any], snapshot["behavior"]), goal)
        and _goal_dominates(cast(Mapping[str, Any], snapshot["causal"]), goal)
    ]
    return pure[0] if len(pure) == 1 else None


def _event_lag(later: Mapping[str, Any], earlier: Mapping[str, Any]) -> int | None:
    if not bool(later["observed"]) or not bool(earlier["observed"]):
        return None
    return int(later["first_qualifying_offset"]) - int(earlier["first_qualifying_offset"])


def _compressed_control_sequence(
    snapshots: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for step in LOCAL_CHECKPOINTS:
        snapshot = snapshots[step]
        pure = _pure_goal(snapshot)
        truth_class = classify_truth_table(str(snapshot["raw_hard_signature"]))
        state = f"pure_{pure}" if pure is not None else truth_class
        if result and result[-1]["state"] == state:
            result[-1]["end_offset"] = step
        else:
            result.append({"state": state, "start_offset": step, "end_offset": step})
    return result


def block_phase_transition_rows(
    *,
    seed: int,
    schedule: str,
    block_position: int,
    requested_goal: str,
    snapshots: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Frozen secondary acquisition/event summaries for one component block."""

    if schedule not in SCHEDULES or requested_goal not in GOALS:
        raise ValueError("unknown E20 schedule or requested goal")
    if block_position not in (1, 2, 3):
        raise ValueError("block position must be 1, 2, or 3")
    if set(snapshots) != set(LOCAL_CHECKPOINTS):
        raise ValueError("block snapshots differ from the frozen lattice")
    rows: list[dict[str, Any]] = []
    for goal in GOALS:
        behavior = _persistent_event(
            snapshots,
            lambda snapshot, g=goal: _goal_dominates(cast(Mapping[str, Any], snapshot["behavior"]), g),
        )
        causal = _persistent_event(
            snapshots,
            lambda snapshot, g=goal: _goal_dominates(cast(Mapping[str, Any], snapshot["causal"]), g),
        )
        pure = _persistent_event(snapshots, lambda snapshot, g=goal: _pure_goal(snapshot) == g)
        for layer in ("first_hidden", "final_hidden"):
            probe = _persistent_event(
                snapshots,
                lambda snapshot, g=goal, layer_name=layer: (
                    float(
                        get(
                            snapshot,
                            f"probe_cross_validated_accuracy.{layer_name}.{g}",
                        )
                    )
                    >= PROBE_LEVEL
                    and float(
                        get(
                            snapshot,
                            f"probe_cross_validated_accuracy.{layer_name}.{g}",
                        )
                    )
                    - float(
                        get(
                            snapshot,
                            f"probe_cross_validated_accuracy.{layer_name}.truth_table_control",
                        )
                    )
                    >= PROBE_ADVANTAGE
                ),
            )
            rows.append(
                {
                    "seed": seed,
                    "schedule": schedule,
                    "block_position": block_position,
                    "requested_goal": requested_goal,
                    "goal": goal,
                    "probe_layer": layer,
                    "probe_observed": probe["observed"],
                    "probe_first_offset": probe["first_qualifying_offset"],
                    "probe_confirmation_offset": probe["confirmation_offset"],
                    "behavior_observed": behavior["observed"],
                    "behavior_first_offset": behavior["first_qualifying_offset"],
                    "behavior_confirmation_offset": behavior["confirmation_offset"],
                    "causal_observed": causal["observed"],
                    "causal_first_offset": causal["first_qualifying_offset"],
                    "causal_confirmation_offset": causal["confirmation_offset"],
                    "pure_observed": pure["observed"],
                    "pure_first_offset": pure["first_qualifying_offset"],
                    "pure_confirmation_offset": pure["confirmation_offset"],
                    "probe_to_behavior_lag": _event_lag(behavior, probe),
                    "probe_to_causal_lag": _event_lag(causal, probe),
                    "censor_offset": 256,
                    "first_controlled_goal": _pure_goal(snapshots[0]),
                    "terminal_controlled_goal": _pure_goal(snapshots[256]),
                    "block_end_signature": _normalize_signature(str(snapshots[256]["raw_hard_signature"])),
                    "block_end_truth_table_class": classify_truth_table(
                        str(snapshots[256]["raw_hard_signature"])
                    ),
                    "compressed_control_sequence": json.dumps(
                        _compressed_control_sequence(snapshots),
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                }
            )
    return rows


def late_stability(snapshot_map: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    if not all(step in snapshot_map for step in LATE_STABILITY_STEPS):
        raise ValueError("late-stability checkpoints are not all direct observations")
    signatures = [
        _normalize_signature(str(snapshot_map[step]["raw_hard_signature"])) for step in LATE_STABILITY_STEPS
    ]
    changes = {
        f"{family}_{goal}": abs(
            float(get(snapshot_map[256], f"{family}.{goal}"))
            - float(get(snapshot_map[222], f"{family}.{goal}"))
        )
        for family in ("behavior", "causal")
        for goal in GOALS
    }
    return {
        "signature_unchanged_161_222_256": len(set(signatures)) == 1,
        "absolute_changes_222_256": changes,
        "all_six_changes_at_most_0_02": all(value <= 0.02 for value in changes.values()),
        "late_stable": len(set(signatures)) == 1 and all(value <= 0.02 for value in changes.values()),
    }


def aggregate_seed_outcomes(
    outcomes: Mapping[int, Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Bootstrap the fixed 20 seed outcomes and apply the registered hierarchy."""

    if set(outcomes) != set(SEEDS):
        raise ValueError("aggregate analysis requires exactly all 20 frozen full seeds")

    def values(path: str) -> list[float]:
        result = [float(get(outcomes[seed], path)) for seed in SEEDS]
        if not all(math.isfinite(value) for value in result):
            raise ValueError(f"non-finite seed outcome at {path}")
        return result

    primary = bootstrap_summary(
        values("primary_hamming_dispersion_auc_33_128"),
        key=BOOTSTRAP_KEYS["primary_dispersion"],
    )
    primary["decision"] = primary_decision(primary)
    signed = bootstrap_summary(
        values("signed_recency_auc_33_128"),
        key=BOOTSTRAP_KEYS["signed_recency"],
    )
    signed["decision"] = direction_decision(signed)
    bonferroni_confidence = 1.0 - 0.05 / 3.0
    per_goal: dict[str, Any] = {}
    inferential_rows: list[dict[str, Any]] = [
        {"endpoint": "primary_hamming_dispersion_auc_33_128", **primary},
        {"endpoint": "signed_recency_auc_33_128", **signed},
    ]
    for goal in GOALS:
        seed_values = values(f"per_goal_recency_auc_33_128.{goal}")
        ordinary = bootstrap_summary(
            seed_values,
            key=BOOTSTRAP_KEYS[f"goal_{goal}_recency"],
        )
        bonferroni = bootstrap_summary(
            seed_values,
            key=BOOTSTRAP_KEYS[f"goal_{goal}_recency"],
            confidence=bonferroni_confidence,
        )
        bonferroni["decision"] = direction_decision(bonferroni, formal_confidence=bonferroni_confidence)
        per_goal[goal] = {
            "ordinary_95_percent": ordinary,
            "formal_bonferroni_98_33_percent": bonferroni,
        }
        inferential_rows.extend(
            [
                {
                    "endpoint": f"{goal}_last_minus_first_margin_auc_33_128",
                    "interval_role": "ordinary_95_percent",
                    **ordinary,
                },
                {
                    "endpoint": f"{goal}_last_minus_first_margin_auc_33_128",
                    "interval_role": "formal_bonferroni_98_33_percent",
                    **bonferroni,
                },
            ]
        )
    secondary_specs = (
        (
            "early_hamming_dispersion_auc_0_33",
            "early_hamming_dispersion_auc_0_33",
            "early_dispersion",
        ),
        (
            "terminal_hamming_dispersion_256",
            "terminal_hamming_dispersion_256",
            "terminal_dispersion",
        ),
        *(
            (
                f"{goal}_middle_curvature_auc_33_128",
                f"curvature_auc_33_128.{goal}",
                f"curvature_{goal}",
            )
            for goal in GOALS
        ),
        *(
            (
                f"{goal}_middle_minus_first_auc_33_128",
                f"middle_minus_first_auc_33_128.{goal}",
                f"middle_first_{goal}",
            )
            for goal in GOALS
        ),
        *(
            (
                f"{goal}_last_minus_middle_auc_33_128",
                f"last_minus_middle_auc_33_128.{goal}",
                f"last_middle_{goal}",
            )
            for goal in GOALS
        ),
        *(
            (
                f"{goal}_behavior_last_minus_first_auc_33_128",
                f"behavior_last_minus_first_auc_33_128.{goal}",
                f"behavior_position_{goal}",
            )
            for goal in GOALS
        ),
        *(
            (
                f"{goal}_causal_last_minus_first_auc_33_128",
                f"causal_last_minus_first_auc_33_128.{goal}",
                f"causal_position_{goal}",
            )
            for goal in GOALS
        ),
    )
    secondary_rows = [
        {
            "endpoint": endpoint,
            **bootstrap_summary(values(path), key=BOOTSTRAP_KEYS[key]),
        }
        for endpoint, path, key in secondary_specs
    ]
    primary_material = primary["decision"] == "material_persistent_order_dependence"
    direction = str(signed["decision"])
    if primary_material and direction == "material_recency":
        hierarchy = "persistent_order_dependence_with_recency"
    elif primary_material and direction == "material_primacy":
        hierarchy = "persistent_order_dependence_with_primacy"
    elif primary_material and direction == "practical_directional_equivalence":
        hierarchy = "nonpositional_or_schedule_specific_persistent_path_dependence"
    elif primary_material:
        hierarchy = "persistent_order_dependence_with_mixed_or_inconclusive_direction"
    elif direction in {"material_recency", "material_primacy"}:
        hierarchy = "candidate_control_shift_without_registered_behavioral_path_dependence"
    else:
        hierarchy = str(primary["decision"])
    seed_rows = [
        {
            "seed": seed,
            **{
                key: value
                for key, value in outcomes[seed].items()
                if key not in {"time_rows", "schedule_arm_rows"}
            },
        }
        for seed in SEEDS
    ]
    analysis = {
        "primary": primary,
        "hierarchical_signed_direction": signed,
        "per_goal_position_effects": per_goal,
        "hierarchical_interpretation": hierarchy,
        "primary_population": "all 20 frozen seeds without response-based filtering",
        "independent_unit": "seed",
        "within_seed_pairing_preserved": True,
        "bonferroni_confidence": bonferroni_confidence,
        "adaptive_posthoc_scope": (
            "path dependence under the exact E20 component-order construction; "
            "not a universal phase transition and not label-order isolation"
        ),
    }
    return analysis, seed_rows, [*inferential_rows, *secondary_rows]


@dataclass(frozen=True)
class FullRun:
    path: Path
    config: Mapping[str, Any]
    summary: Mapping[str, Any]
    metadata: Mapping[str, Any]
    metric_lines: int

    @property
    def seed(self) -> int:
        return int(self.summary["seed"])

    @property
    def schedule(self) -> str:
        return str(self.summary["schedule"])


def _expected_configs() -> dict[str, Mapping[str, Any]]:
    if _sha256(DESIGN_MEMO) != DESIGN_MEMO_SHA256:
        raise RuntimeError("E20 design memo differs from its authoritative SHA-256")
    if FROZEN_CONFIG_SHA256 == "PENDING" or _sha256(CONFIG_PATH) != FROZEN_CONFIG_SHA256:
        raise RuntimeError("E20 full config is not at its frozen SHA-256")
    if FROZEN_LAUNCHER_SHA256 == "PENDING" or _sha256(LAUNCHER_PATH) != FROZEN_LAUNCHER_SHA256:
        raise RuntimeError("E20 launcher is not at its frozen SHA-256")
    cells = expand_sweep(load_config(CONFIG_PATH))
    result = {str(get(cell, "h17.schedule")): _scientific_config(cell) for cell in cells}
    if tuple(result) != SCHEDULES:
        raise RuntimeError(f"E20 full expansion differs from the six frozen schedules: {tuple(result)}")
    return result


def _frozen_implementation() -> dict[str, Any]:
    if FROZEN_SOURCE_FINGERPRINT == "PENDING" or FROZEN_SOURCE_FILE_COUNT < 1:
        raise RuntimeError("E20 source freeze constants are still pending")
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "source_fingerprint_schema_version": SOURCE_FINGERPRINT_SCHEMA_VERSION,
        "implementation_fingerprint": FROZEN_SOURCE_FINGERPRINT,
        "source_file_count": FROZEN_SOURCE_FILE_COUNT,
    }


def _expected_protocol_truth_class(signature: str, tuple_signature: str) -> dict[str, Any]:
    classification = classify_truth_table(signature)
    if classification in NAMED_RULES:
        return {
            "kind": "exact_named_rule",
            "name": classification,
            "label": f"exact_{classification}",
            "tuple_signature": tuple_signature,
        }
    if classification == "other_candidate_tuple_consistent":
        return {
            "kind": classification,
            "name": None,
            "label": "other_tuple_rule",
            "tuple_signature": tuple_signature,
        }
    return {
        "kind": "raw_codeword_specific_composite",
        "name": None,
        "label": "raw_composite",
        "tuple_signature": tuple_signature,
    }


def audit_snapshot(
    snapshot: Any,
    *,
    local_step: int,
    global_step: int,
    run_id: str,
    errors: list[str],
) -> None:
    """Independently reconstruct the exhaustive raw trajectory measurement."""

    if not isinstance(snapshot, Mapping):
        errors.append(f"{run_id}: missing snapshot at global step {global_step}")
        return
    for field, expected in {
        "local_step": local_step,
        "global_step": global_step,
        "examples_seen": global_step * BATCH_SIZE,
    }.items():
        _expect(errors, run_id, snapshot.get(field), expected, f"snapshot {global_step}.{field}")
    required = {
        "local_step",
        "global_step",
        "examples_seen",
        "behavior",
        "causal",
        "causal_probability",
        "control_margin",
        "pure_control",
        "pure_goal",
        "zero_logit_count",
        "raw_hard_signature",
        "raw_probability_table",
        "raw_codeword_table",
        "truth_table",
        "tuple_signature",
        "truth_table_class",
        "causal_details",
        "probes",
        "probe_cross_validated_accuracy",
        "selective_probe_advantage",
    }
    if set(snapshot) != required:
        errors.append(
            f"{run_id}: snapshot {global_step} fields missing={sorted(required - set(snapshot))}, "
            f"unexpected={sorted(set(snapshot) - required)}"
        )
    table = snapshot.get("raw_codeword_table")
    probabilities = snapshot.get("raw_probability_table")
    if (
        not isinstance(table, Sequence)
        or isinstance(table, (str, bytes))
        or len(table) != 64
        or not isinstance(probabilities, Sequence)
        or isinstance(probabilities, (str, bytes))
        or len(probabilities) != 64
    ):
        errors.append(f"{run_id}: snapshot {global_step} lacks the exact 64-row panel")
        return
    actions: list[int] = []
    direct_probabilities: list[float] = []
    rules = cast(Mapping[int, Mapping[str, int]], FROZEN_PANEL["rules"])
    folds = cast(Sequence[int], FROZEN_PANEL["folds"])
    control = cast(Sequence[int], FROZEN_PANEL["control"])
    for raw_id, raw in enumerate(table):
        if not isinstance(raw, Mapping):
            errors.append(f"{run_id}: snapshot {global_step} raw row {raw_id} is malformed")
            continue
        signs = [1 if raw_id & (1 << shift) else -1 for shift in (5, 4, 3, 2, 1, 0)]
        expected_features = [
            float(signs[0]),
            1.0,
            float(signs[1]),
            float(signs[2]),
            float(signs[3]),
            1.0,
            float(signs[4]),
            float(signs[5]),
        ]
        expected = {
            "raw_id": raw_id,
            "tuple_id": rules[raw_id]["tuple_id"],
            "fold": folds[raw_id],
            "features": expected_features,
            "P": rules[raw_id]["P"],
            "Q": rules[raw_id]["Q"],
            "Y": rules[raw_id]["Y"],
            "M": rules[raw_id]["M"],
            "truth_table_control": control[raw_id],
        }
        for field, value in expected.items():
            if raw.get(field) != value:
                errors.append(f"{run_id}: raw row {raw_id}.{field} differs from reconstruction")
        logit = raw.get("logit")
        probability = raw.get("probability_positive")
        action = raw.get("hard_action")
        if not _finite(logit) or not _finite(probability) or action not in (-1, 1):
            errors.append(f"{run_id}: raw row {raw_id} has malformed model outputs")
            continue
        expected_action = 1 if float(logit) >= 0.0 else -1
        expected_probability = (
            1.0 / (1.0 + math.exp(-float(logit)))
            if float(logit) >= 0.0
            else math.exp(float(logit)) / (1.0 + math.exp(float(logit)))
        )
        if int(action) != expected_action:
            errors.append(f"{run_id}: raw row {raw_id} violates the hard-action convention")
        if abs(float(probability) - expected_probability) > 1e-12:
            errors.append(f"{run_id}: raw row {raw_id} probability differs from its logit")
        if (
            not _finite(probabilities[raw_id])
            or abs(float(probabilities[raw_id]) - float(probability)) > 1e-15
        ):
            errors.append(f"{run_id}: raw row {raw_id} differs from probability table")
        actions.append(int(action))
        direct_probabilities.append(float(probability))
    if len(actions) != 64:
        return
    signature = "".join("1" if action > 0 else "0" for action in actions)
    _expect(errors, run_id, snapshot.get("raw_hard_signature"), signature, "direct raw signature")
    behavior = {
        goal: sum(actions[raw_id] == rules[raw_id][goal] for raw_id in range(64)) / 64.0
        for goal in NAMED_RULES
    }
    _expect(errors, run_id, snapshot.get("behavior"), behavior, "direct behavior reconstruction")
    _expect(
        errors,
        run_id,
        snapshot.get("zero_logit_count"),
        sum(cast(Mapping[str, Any], table[index])["logit"] == 0.0 for index in range(64)),
        "exact-zero logit count",
    )
    tuple_actions: list[int] = []
    tuple_mode_matches = 0
    for tuple_id in range(8):
        selected = [actions[index] for index in range(64) if rules[index]["tuple_id"] == tuple_id]
        positive = sum(value > 0 for value in selected)
        mode = 1 if positive >= 4 else -1
        tuple_actions.append(mode)
        tuple_mode_matches += sum(value == mode for value in selected)
    tuple_signature = "".join("1" if value > 0 else "0" for value in tuple_actions)
    _expect(errors, run_id, snapshot.get("tuple_signature"), tuple_signature, "tuple signature")
    truth = snapshot.get("truth_table")
    if not isinstance(truth, Mapping):
        errors.append(f"{run_id}: missing truth-table diagnostics")
    else:
        _expect(errors, run_id, truth.get("boolean_signature"), tuple_signature, "truth boolean signature")
        _expect(
            errors, run_id, truth.get("boolean_signature_int"), int(tuple_signature, 2), "truth signature int"
        )
        _expect(
            errors, run_id, truth.get("tuple_consistency"), tuple_mode_matches / 64.0, "tuple consistency"
        )
        if (
            _finite(truth.get("mean_positive_probability"))
            and abs(float(truth["mean_positive_probability"]) - float(np.mean(direct_probabilities))) > 1e-12
        ):
            errors.append(f"{run_id}: mean positive probability differs from raw rows")
    _expect(
        errors,
        run_id,
        snapshot.get("truth_table_class"),
        _expected_protocol_truth_class(signature, tuple_signature),
        "hierarchical truth-table class",
    )
    causal_details = snapshot.get("causal_details")
    if not isinstance(causal_details, Mapping) or not isinstance(causal_details.get("family_means"), Mapping):
        errors.append(f"{run_id}: missing causal family details")
    else:
        if (
            set(causal_details) != {"per_intervention", "family_means", "normalization", "n"}
            or causal_details.get("n") != 64
            or set(cast(Mapping[str, Any], causal_details.get("per_intervention", {})))
            != {"flip_P", "flip_Q_1", "flip_Q_2", "flip_Y_1", "flip_Y_2", "flip_Y_3"}
            or set(cast(Mapping[str, Any], causal_details["family_means"])) != set(GOALS)
            or not _numeric_values_finite(causal_details)
        ):
            errors.append(f"{run_id}: causal intervention schema is incomplete or non-finite")
        family = cast(Mapping[str, Mapping[str, Any]], causal_details["family_means"])
        expected_causal = {goal: float(family[goal]["causal_score"]) for goal in GOALS}
        expected_probability_causal = {goal: float(family[goal]["causal_prob_score"]) for goal in GOALS}
        _expect(errors, run_id, snapshot.get("causal"), expected_causal, "causal family means")
        _expect(
            errors,
            run_id,
            snapshot.get("causal_probability"),
            expected_probability_causal,
            "probability causal family means",
        )
    causal = snapshot.get("causal")
    if isinstance(causal, Mapping) and set(causal) == set(GOALS):
        margins = {
            goal: 0.5
            * (
                behavior[goal]
                - max(behavior[other] for other in GOALS if other != goal)
                + float(causal[goal])
                - max(float(causal[other]) for other in GOALS if other != goal)
            )
            for goal in GOALS
        }
        _expect(errors, run_id, snapshot.get("control_margin"), margins, "control margins")
        pure = {
            goal: behavior[goal] >= ACQUISITION_LEVEL
            and float(causal[goal]) >= ACQUISITION_LEVEL
            and all(
                behavior[goal] - behavior[other] >= ACQUISITION_MARGIN for other in GOALS if other != goal
            )
            and all(
                float(causal[goal]) - float(causal[other]) >= ACQUISITION_MARGIN
                for other in GOALS
                if other != goal
            )
            for goal in GOALS
        }
        _expect(errors, run_id, snapshot.get("pure_control"), pure, "pure-control labels")
        pure_goals = [goal for goal, value in pure.items() if value]
        _expect(
            errors,
            run_id,
            snapshot.get("pure_goal"),
            pure_goals[0] if len(pure_goals) == 1 else None,
            "unique pure goal",
        )
    probes = snapshot.get("probes")
    accuracies = snapshot.get("probe_cross_validated_accuracy")
    if not isinstance(probes, Mapping) or not isinstance(accuracies, Mapping):
        errors.append(f"{run_id}: missing prospective probe diagnostics")
    else:
        for field, expected in {
            "alpha": 0.001,
            "fold_digest": FOLD_DIGEST,
            "truth_table_control_digest": CONTROL_DIGEST,
            "label_names": ["P", "Q", "Y", "truth_table_control"],
            "probe_kind": "deterministic_affine_ridge_eight_fold",
            "standardization": "seven_training_folds_only",
        }.items():
            _expect(errors, run_id, probes.get(field), expected, f"probe.{field}")
        representations = probes.get("representations")
        if not isinstance(representations, Mapping) or set(representations) != {
            "first_hidden",
            "final_hidden",
        }:
            errors.append(f"{run_id}: prospective probe layer schema differs from freeze")
            representations = {}
        for layer in ("first_hidden", "final_hidden"):
            layer_values = representations.get(layer) if isinstance(representations, Mapping) else None
            if not isinstance(layer_values, Mapping):
                errors.append(f"{run_id}: missing {layer} prospective probe")
                continue
            folds_by_id = layer_values.get("fold_accuracy")
            if (
                layer_values.get("n") != 64
                or layer_values.get("dimension") != 64
                or not isinstance(folds_by_id, Mapping)
                or set(folds_by_id) != {str(index) for index in range(8)}
                or any(
                    not isinstance(values, Mapping)
                    or set(values) != {*GOALS, "truth_table_control"}
                    or not all(_finite(value) and 0.0 <= float(value) <= 1.0 for value in values.values())
                    for values in cast(Mapping[str, Any], folds_by_id or {}).values()
                )
            ):
                errors.append(f"{run_id}: {layer} fold-probe schema differs from freeze")
            expected_accuracy = get(probes, f"representations.{layer}.cross_validated_accuracy")
            if (
                not isinstance(expected_accuracy, Mapping)
                or set(expected_accuracy) != {*GOALS, "truth_table_control"}
                or not all(
                    _finite(value) and 0.0 <= float(value) <= 1.0
                    for value in cast(Mapping[str, Any], expected_accuracy or {}).values()
                )
            ):
                errors.append(f"{run_id}: {layer} cross-validated probe schema differs")
            _expect(errors, run_id, accuracies.get(layer), expected_accuracy, f"{layer} probe accuracy")
            if isinstance(expected_accuracy, Mapping):
                expected_advantage = {
                    goal: float(expected_accuracy[goal]) - float(expected_accuracy["truth_table_control"])
                    for goal in GOALS
                }
                _expect(
                    errors,
                    run_id,
                    get(snapshot, f"selective_probe_advantage.{layer}"),
                    expected_advantage,
                    f"{layer} selective probe advantage",
                )


def _metric_expected_value(
    snapshot: Mapping[str, Any], stage_suffix: str, intervention: str, metric: str
) -> float | int | None:
    if stage_suffix == "behavior":
        mapping = {
            "rho_p": get(snapshot, "behavior.P"),
            "rho_q": get(snapshot, "behavior.Q"),
            "rho_y": get(snapshot, "behavior.Y"),
            "rho_m": get(snapshot, "behavior.M"),
            "m_p": get(snapshot, "control_margin.P"),
            "m_q": get(snapshot, "control_margin.Q"),
            "m_y": get(snapshot, "control_margin.Y"),
            "zero_logit_count": get(snapshot, "zero_logit_count"),
        }
        return cast(float | int | None, mapping.get(metric))
    if stage_suffix == "causal" and intervention in GOALS:
        mapping = {
            "causal_score": get(snapshot, f"causal.{intervention}"),
            "causal_prob_score": get(snapshot, f"causal_probability.{intervention}"),
        }
        return cast(float | int | None, mapping.get(metric))
    if stage_suffix == "probe":
        for layer in ("first_hidden", "final_hidden"):
            for label in (*GOALS, "truth_table_control"):
                name = f"representations__{layer}__cross_validated_accuracy__{label}"
                if metric == name:
                    return cast(
                        float | int | None, get(snapshot, f"probe_cross_validated_accuracy.{layer}.{label}")
                    )
    if stage_suffix == "truth_table":
        mapping = {
            "boolean_signature_int": get(snapshot, "truth_table.boolean_signature_int"),
            "tuple_consistency": get(snapshot, "truth_table.tuple_consistency"),
            "mean_positive_probability": get(snapshot, "truth_table.mean_positive_probability"),
        }
        return cast(float | int | None, mapping.get(metric))
    return None


METRIC_STAGE_SUFFIXES = ("optimization", "truth_table", "behavior", "causal", "probe")


def _parse_metric_stage(stage: str) -> tuple[str, str, int]:
    """Parse one exact non-final metric stage and its frozen global-step origin."""

    matches = [suffix for suffix in METRIC_STAGE_SUFFIXES if stage.endswith(f"_{suffix}")]
    if not matches:
        raise ValueError(f"metric stage has no frozen suffix: {stage!r}")
    suffix = max(matches, key=len)
    base = stage[: -(len(suffix) + 1)]
    if base == "washout":
        return base, suffix, 3 * BLOCK_STEPS
    parts = base.split("_")
    if (
        len(parts) != 3
        or parts[0] != "component"
        or parts[1] not in {"1", "2", "3"}
        or parts[2] not in {goal.lower() for goal in GOALS}
    ):
        raise ValueError(f"metric stage has malformed frozen base: {stage!r}")
    return base, suffix, (int(parts[1]) - 1) * BLOCK_STEPS


def _metric_audit(
    path: Path,
    *,
    run_id: str,
    seed: int,
    schedule: str,
    summary: Mapping[str, Any],
    errors: list[str],
) -> int:
    """Require the exact registered metric grid and cross-check stored snapshots."""

    snapshots_by_stage: dict[str, Mapping[int, Mapping[str, Any]]] = {}
    histories_by_stage: dict[str, Mapping[int, Mapping[str, Any]]] = {}
    groups = summary.get("component_snapshots")
    if isinstance(groups, Sequence) and not isinstance(groups, (str, bytes)):
        for group in groups:
            if not isinstance(group, Mapping):
                continue
            position = int(group.get("position", -1))
            goal = str(group.get("goal", "")).lower()
            stored = group.get("snapshots")
            if isinstance(stored, Mapping):
                snapshots_by_stage[f"component_{position}_{goal}"] = {
                    int(step): cast(Mapping[str, Any], value)
                    for step, value in stored.items()
                    if isinstance(value, Mapping)
                }
    washout = summary.get("washout_snapshots")
    if isinstance(washout, Mapping):
        snapshots_by_stage["washout"] = {
            int(step): cast(Mapping[str, Any], value)
            for step, value in washout.items()
            if isinstance(value, Mapping)
        }
    blocks = get(summary, "training.blocks")
    if isinstance(blocks, Sequence) and not isinstance(blocks, (str, bytes)):
        for block in blocks:
            if not isinstance(block, Mapping) or not isinstance(block.get("history"), Sequence):
                continue
            boundary = str(block.get("boundary_key", ""))
            base = boundary.lower() if boundary != "washout" else "washout"
            histories_by_stage[base] = {
                int(record["step"]): cast(Mapping[str, Any], record)
                for record in cast(Sequence[Any], block["history"])
                if isinstance(record, Mapping) and "step" in record
            }
    expected_bases = {
        f"component_{position}_{goal.lower()}"
        for position, goal in enumerate(SCHEDULE_GOALS.get(schedule, ()), start=1)
    } | {"washout"}
    expected_stages = {
        f"{base}_{suffix}"
        for base in expected_bases
        for suffix in ("behavior", "truth_table", "probe", "causal", "optimization")
    } | {"final"}
    stage_coordinates: dict[str, set[tuple[int, int, int]]] = defaultdict(set)
    stage_counts: Counter[str] = Counter()
    point_counts: Counter[tuple[str, int]] = Counter()
    identities: set[tuple[str, int, str, str, str]] = set()
    line_count = 0
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as error:
        errors.append(f"{run_id}: cannot read metrics: {error}")
        return 0
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
            if stage not in expected_stages:
                errors.append(f"{run_id}: unexpected metric stage {stage!r}")
                continue
            if row.get("run_id") != run_id or row.get("seed") != seed:
                errors.append(f"{run_id}: metric identity mismatch at line {line_number}")
            if row.get("experiment") != HYPOTHESIS or row.get("level") != "choice":
                errors.append(f"{run_id}: metric design mismatch at line {line_number}")
            expected_condition = (
                "frozen_adaptive_posthoc_full_panel"
                if stage == "final"
                else f"counterbalanced_order:{schedule}"
            )
            if row.get("condition") != expected_condition:
                errors.append(f"{run_id}: metric condition mismatch at line {line_number}")
            if not _finite(row.get("value")) or not isinstance(row.get("n"), int):
                errors.append(f"{run_id}: malformed metric scalar at line {line_number}")
            try:
                local = int(row["stage_step"])
                global_step = int(row["global_step"])
                examples = int(row["examples_seen"])
            except (KeyError, TypeError, ValueError):
                errors.append(f"{run_id}: malformed metric coordinates at line {line_number}")
                continue
            stage_coordinates[stage].add((local, global_step, examples))
            stage_counts[stage] += 1
            point_counts[(stage, local)] += 1
            identity = (
                stage,
                local,
                str(row.get("split")),
                str(row.get("intervention")),
                str(row.get("metric")),
            )
            if identity in identities:
                errors.append(f"{run_id}: duplicate metric identity at line {line_number}: {identity}")
            identities.add(identity)
            if stage == "final":
                if (local, global_step, examples) != (TOTAL_STEPS, TOTAL_STEPS, TOTAL_SAMPLES):
                    errors.append(f"{run_id}: final metric coordinate differs from freeze")
                expected_final = get(summary, f"navigation.choice.{row.get('metric')}")
                if expected_final is None or not math.isclose(
                    float(row["value"]), float(expected_final), rel_tol=0.0, abs_tol=1e-12
                ):
                    errors.append(f"{run_id}: final navigation metric differs from summary")
                continue
            try:
                base, suffix, base_global = _parse_metric_stage(stage)
            except ValueError as error:
                errors.append(f"{run_id}: malformed metric stage at line {line_number}: {error}")
                continue
            if global_step != base_global + local or examples != global_step * BATCH_SIZE:
                errors.append(f"{run_id}: metric exposure mismatch at line {line_number}")
            snapshot = snapshots_by_stage.get(base, {}).get(local)
            expected_value = (
                None
                if snapshot is None
                else _metric_expected_value(
                    snapshot,
                    suffix,
                    str(row.get("intervention")),
                    str(row.get("metric")),
                )
            )
            if suffix == "optimization":
                history = histories_by_stage.get(base, {}).get(local)
                expected_value = (
                    None
                    if history is None
                    else {
                        "loss": history.get("loss"),
                        "primary_loss": history.get("primary_loss"),
                        "train_batch_accuracy": history.get("train_accuracy"),
                        "optimizer_steps": history.get("optimizer_steps"),
                    }.get(str(row.get("metric")))
                )
            if expected_value is not None and not math.isclose(
                float(row["value"]), float(expected_value), rel_tol=0.0, abs_tol=1e-12
            ):
                errors.append(f"{run_id}: metric/snapshot mismatch at line {line_number}")
    if line_count != 20_149:
        errors.append(f"{run_id}: metric record count={line_count}, expected 20149")
    if set(stage_counts) != expected_stages:
        errors.append(f"{run_id}: metric stage set differs from freeze")
    family_count = {"behavior": 8, "truth_table": 70, "probe": 77, "causal": 93}
    for base in expected_bases:
        for suffix, count in family_count.items():
            stage = f"{base}_{suffix}"
            expected_coords = {
                (
                    local,
                    (768 if base == "washout" else (int(base.split("_")[1]) - 1) * 256) + local,
                    ((768 if base == "washout" else (int(base.split("_")[1]) - 1) * 256) + local) * 288,
                )
                for local in LOCAL_CHECKPOINTS
            }
            if stage_coordinates.get(stage) != expected_coords:
                errors.append(f"{run_id}: {stage} coordinate lattice differs from freeze")
            if any(point_counts[(stage, local)] != count for local in LOCAL_CHECKPOINTS):
                errors.append(f"{run_id}: {stage} metric count per checkpoint differs from {count}")
        optimization = f"{base}_optimization"
        if any(point_counts[(optimization, local)] != 4 for local in LOCAL_CHECKPOINTS[1:]):
            errors.append(f"{run_id}: {optimization} count differs from four")
        if any(point_counts[(optimization, local)] for local in (0,)):
            errors.append(f"{run_id}: {optimization} incorrectly records offset zero")
    if stage_counts.get("final") != 5:
        errors.append(f"{run_id}: final metric count differs from five")
    return line_count


def _audit_empty_optimizer(
    reset: Any,
    *,
    run_id: str,
    boundary_global_step: int,
    model_hash: Any,
    errors: list[str],
) -> None:
    if not isinstance(reset, Mapping):
        errors.append(f"{run_id}: missing fresh optimizer audit at {boundary_global_step}")
        return
    fixed = {
        "optimizer_class": "AdamW",
        "state_entry_count": 0,
        "adam_step_min": None,
        "adam_step_max": None,
        "adam_step_entry_count": 0,
        "param_group_count": 1,
        "optimized_parameter_count": 6,
        "learning_rate": 0.003,
        "weight_decay": 0.0,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "boundary_global_step": boundary_global_step,
        "semantic_reset_implemented_by_fresh_object": True,
        "model_unchanged_by_optimizer_reset": True,
        "model_hash_before_optimizer_construction": model_hash,
        "model_hash_after_optimizer_construction": model_hash,
    }
    for field, expected in fixed.items():
        _expect(errors, run_id, reset.get(field), expected, f"optimizer reset {boundary_global_step}.{field}")
    if not _hash_is_valid(reset.get("actual_state_digest")):
        errors.append(f"{run_id}: malformed empty optimizer digest")


def _audit_training_blocks(
    summary: Mapping[str, Any],
    *,
    run_id: str,
    order: Sequence[str],
    errors: list[str],
) -> None:
    blocks = get(summary, "training.blocks")
    if not isinstance(blocks, Sequence) or isinstance(blocks, (str, bytes)) or len(blocks) != 4:
        errors.append(f"{run_id}: training history is not exactly four blocks")
        return
    expected = [
        (f"component_{position}_{goal}", "component", position, goal, PHASE_SEEDS[f"component_{goal}"])
        for position, goal in enumerate(order, 1)
    ] + [("washout", "washout", 4, None, PHASE_SEEDS["washout"])]
    for block, (boundary, kind, position, goal, phase_seed) in zip(blocks, expected, strict=True):
        if not isinstance(block, Mapping):
            errors.append(f"{run_id}: malformed training block {position}")
            continue
        for field, value in {
            "boundary_key": boundary,
            "kind": kind,
            "position": position,
            "goal": goal,
            "optimizer_steps": 256,
            "samples_seen": 73_728,
            "phase_rng_seed": phase_seed,
        }.items():
            _expect(errors, run_id, block.get(field), value, f"training block {position}.{field}")
        stream_name = "washout" if goal is None else goal
        _expect(
            errors,
            run_id,
            block.get("stream_digest"),
            get(summary, f"observer_on_execution.streams.{stream_name}.ordered_stream_digest"),
            f"training block {position} stream digest",
        )
        _expect(
            errors,
            run_id,
            block.get("row_exposure_digest"),
            get(summary, f"observer_on_execution.streams.{stream_name}.row_exposure_digest"),
            f"training block {position} exposure digest",
        )
        history = block.get("history")
        if not isinstance(history, Sequence) or isinstance(history, (str, bytes)) or len(history) != 19:
            errors.append(f"{run_id}: training block {position} history is not 19 direct checkpoints")
        else:
            for step, record in zip(LOCAL_CHECKPOINTS[1:], history, strict=True):
                if not isinstance(record, Mapping):
                    errors.append(f"{run_id}: malformed training history at block {position}, step {step}")
                    continue
                for field, value in {
                    "step": step,
                    "samples_seen": step * BATCH_SIZE,
                    "optimizer_steps": step,
                    "auxiliary_loss": 0.0,
                }.items():
                    _expect(errors, run_id, record.get(field), value, f"history {position}:{step}.{field}")
                for field in ("loss", "primary_loss", "train_accuracy"):
                    if not _finite(record.get(field)):
                        errors.append(f"{run_id}: non-finite history {position}:{step}.{field}")
                if (
                    _finite(record.get("loss"))
                    and _finite(record.get("primary_loss"))
                    and not math.isclose(
                        float(record["loss"]), float(record["primary_loss"]), rel_tol=0.0, abs_tol=1e-12
                    )
                ):
                    errors.append(f"{run_id}: auxiliary-free loss differs from primary loss")
        final = block.get("optimizer_final_audit")
        if not isinstance(final, Mapping):
            errors.append(f"{run_id}: missing final optimizer audit for block {position}")
        else:
            for field, value in {
                "optimizer_class": "AdamW",
                "state_entry_count": 6,
                "adam_step_min": 256.0,
                "adam_step_max": 256.0,
                "adam_step_entry_count": 6,
                "param_group_count": 1,
                "optimized_parameter_count": 6,
                "learning_rate": 0.003,
                "weight_decay": 0.0,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
            }.items():
                _expect(errors, run_id, final.get(field), value, f"final optimizer {position}.{field}")
            if not _hash_is_valid(final.get("actual_state_digest")):
                errors.append(f"{run_id}: malformed final optimizer state digest")


def _protocol_factorial_audit() -> dict[str, Any]:
    panel = make_counterbalanced_factorial_panel()
    public = audit_counterbalanced_factorial_panel(panel)
    rules = cast(Mapping[int, Mapping[str, int]], FROZEN_PANEL["rules"])
    return {
        "n": 64,
        "feature_names": list(FEATURE_NAMES),
        "input_dim": 8,
        "raw_ids_complete": True,
        "candidate_tuple_count": 8,
        "raw_codewords_per_tuple": 8,
        "fold_count": 8,
        "rows_per_fold": 8,
        "fold_digest": FOLD_DIGEST,
        "truth_table_control_digest": CONTROL_DIGEST,
        "truth_table_control_bits": CONTROL_BITS,
        "truth_table_control_positive_count": 32,
        "folds": list(FROZEN_PANEL["folds"]),
        "tuple_ids": [rules[index]["tuple_id"] for index in range(64)],
        "truth_table_control": list(FROZEN_PANEL["control"]),
        "semantic_batch_digest": semantic_batch_digest(panel),
        "data_module_audit": public,
    }


EXPECTED_FACTORIAL_AUDIT = _protocol_factorial_audit()

REPLAY_FIELDS = (
    "mode",
    "condition",
    "seed",
    "schedule",
    "isolated_goal",
    "order",
    "randomness",
    "boundary_order",
    "hashes",
    "resets",
    "streams",
    "plan",
    "plan_audit",
    "training",
    "trajectory_fingerprint",
)


def audit_replay_equivalence(
    observed: Any,
    replay: Any,
    contract: Any,
    *,
    run_id: str,
    errors: list[str],
) -> None:
    if not isinstance(observed, Mapping) or not isinstance(replay, Mapping):
        errors.append(f"{run_id}: missing observer-on or observer-free execution")
        return
    expected_fields = {"observer_enabled", *REPLAY_FIELDS}
    if set(observed) != expected_fields or set(replay) != expected_fields:
        errors.append(f"{run_id}: observer execution field set differs from freeze")
    _expect(errors, run_id, observed.get("observer_enabled"), True, "observer-on flag")
    _expect(errors, run_id, replay.get("observer_enabled"), False, "observer-free flag")
    for field in REPLAY_FIELDS:
        _expect(errors, run_id, observed.get(field), replay.get(field), f"observer-free replay.{field}")
    fixed_contract = {
        "public_helper": "forkworld.protocols_counterbalanced.replay_h17_observer_free",
        "analyzer_reconstructs_independently": True,
        "protocol_double_trained": False,
        "expected_boundary_order": replay.get("boundary_order"),
        "observer_on_trajectory_fingerprint": replay.get("trajectory_fingerprint"),
    }
    _expect(errors, run_id, contract, fixed_contract, "observer-free replay contract")


def _validate_run(
    run: FullRun,
    expected_config: Mapping[str, Any],
    reconstructed: Mapping[str, Any],
    errors: list[str],
) -> None:
    run_id = run.path.name
    schedule = run.schedule
    order = SCHEDULE_GOALS[schedule]
    if _scientific_config(run.config) != expected_config:
        errors.append(f"{run_id}: scientific config differs from the frozen schedule cell")
    fixed = {
        "hypothesis": HYPOTHESIS,
        "seed": run.seed,
        "condition": f"counterbalanced_order:{schedule}",
        "pilot_only": False,
        "isolated_goal": None,
        "schedule": schedule,
        "order": list(order),
        "design_memo_sha256": DESIGN_MEMO_SHA256,
        "design_status": "frozen_adaptive_posthoc_full_panel",
        "model.input_dim": 8,
        "model.total_parameters": 4_801,
        "model.trainable_parameters": 4_801,
        "model.update_mode": "full",
        "model.requested_budget": "full",
        "model.bias": True,
        "model.nuisance_bits": 0,
        "model.auxiliary_head_count": 0,
        "randomness.nominal_run_seed": run.seed,
        "randomness.model_initialization_seed": run.seed,
        "randomness.data_seed": DATA_SEED,
        "randomness.stream_seed": STREAM_SEED,
        "randomness.phase_seeds": PHASE_SEEDS,
        "randomness.run_seed_affects_only_model_initialization": True,
        "randomness.data_stream_and_phase_rng_independent_of_run_seed": True,
        "data.feature_names": list(FEATURE_NAMES),
        "data.expected_input_dim": 8,
        "data.state_dim": 0,
        "data.padding_columns": [],
        "data.presence_constants": {"P_present": 1, "Q_present": 1},
        "data.factorial_panel": EXPECTED_FACTORIAL_AUDIT,
        "measurement.component_checkpoints": list(LOCAL_CHECKPOINTS),
        "measurement.washout_checkpoints": list(LOCAL_CHECKPOINTS),
        "measurement.all_claimed_checkpoints_directly_observed": True,
        "measurement.raw_codeword_count": 64,
        "measurement.hard_action_convention": "+1 iff logit >= 0",
        "measurement.probe_kind": "deterministic_affine_ridge_eight_fold",
        "measurement.probe_folds": 8,
        "measurement.probe_ridge": 0.001,
        "measurement.fold_digest": FOLD_DIGEST,
        "measurement.truth_table_control_digest": CONTROL_DIGEST,
        "measurement.pure_threshold": ACQUISITION_LEVEL,
        "measurement.pure_margin": ACQUISITION_MARGIN,
        "measurement.m_G_definition": "0.5*((rho_G-max_other_rho)+(c_G-max_other_c))",
        "measurement.directional_causal_normalization": "(1 + E[g*(a-a_flip)/2]) / 2",
        "measurement.primary_auc_window": [33, 128],
        "measurement.primary_effect_threshold": PRIMARY_MATERIAL_MEAN,
        "measurement.primary_ci_lower_boundary": PRACTICAL_BOUNDARY,
        "measurement.primary_equivalence_margin": PRACTICAL_BOUNDARY,
        "measurement.minimum_prevalence_count": PREVALENCE_COUNT,
        "training.total_steps": TOTAL_STEPS,
        "training.total_examples_seen": TOTAL_SAMPLES,
        "training.batch_size": BATCH_SIZE,
        "training.all_minibatches_full": True,
        "training.fresh_optimizer_before_every_constructed_block": True,
        "training.model_weights_continued_between_constructed_blocks": True,
        "training.label_smoothing": 0.0,
        "training.gradient_clip_norm": 1.0,
        "outcomes.order_estimand_computed": False,
        "outcomes.cross_schedule_primary_requires_strict_analyzer": True,
    }
    for path, expected in fixed.items():
        _expect(errors, run_id, get(run.summary, path), expected, f"summary.{path}")
    _audit_training_blocks(run.summary, run_id=run_id, order=order, errors=errors)
    _expect(
        errors, run_id, get(run.summary, "data.bundle_audit"), reconstructed["bundle_audit"], "bundle audit"
    )
    components = get(run.summary, "data.components")
    if not isinstance(components, Mapping) or set(components) != set(GOALS):
        errors.append(f"{run_id}: data components are not exactly P/Q/Y")
    else:
        for goal in GOALS:
            expected_component = reconstructed["components"][goal]
            _expect(
                errors, run_id, get(components, f"{goal}.constructed"), True, f"component {goal} constructed"
            )
            _expect(
                errors,
                run_id,
                get(components, f"{goal}.semantic_batch_digest"),
                expected_component["semantic_batch_digest"],
                f"component {goal} semantic digest",
            )
            _expect(
                errors,
                run_id,
                get(components, f"{goal}.audit"),
                expected_component["audit"],
                f"component {goal} audit",
            )
    for path, expected in {
        "data.washout.constructed": True,
        "data.washout.semantic_batch_digest": reconstructed["washout_semantic_digest"],
        "data.washout.audit": reconstructed["washout_audit"],
        "plan": reconstructed["plan_summary"],
        "plan_audit": reconstructed["plan_audit"],
    }.items():
        _expect(errors, run_id, get(run.summary, path), expected, f"reconstructed {path}")
    isolation = {
        "constructed_components": list(GOALS),
        "unconstructed_components": [],
        "other_components_constructed": True,
        "component_bundle_constructed": True,
        "schedule_defined": True,
        "schedule_constructed": True,
        "washout_constructed": True,
        "pooled_data_constructed": True,
        "order_estimand_computed": False,
        "optimizer_count": 4,
        "optimizer_steps": TOTAL_STEPS,
    }
    _expect(errors, run_id, run.summary.get("construction_isolation"), isolation, "full construction proof")

    groups = run.summary.get("component_snapshots")
    if not isinstance(groups, Sequence) or isinstance(groups, (str, bytes)) or len(groups) != 3:
        errors.append(f"{run_id}: component snapshot groups are not exactly three")
    else:
        for position, (goal, group) in enumerate(zip(order, groups, strict=True), start=1):
            if not isinstance(group, Mapping):
                errors.append(f"{run_id}: malformed component group {position}")
                continue
            for field, expected in {
                "position": position,
                "goal": goal,
                "boundary_key": f"component_{position}_{goal}",
            }.items():
                _expect(errors, run_id, group.get(field), expected, f"component group {position}.{field}")
            snapshots = group.get("snapshots")
            if not isinstance(snapshots, Mapping) or set(snapshots) != {
                str(step) for step in LOCAL_CHECKPOINTS
            }:
                errors.append(f"{run_id}: component {position} checkpoint lattice differs from freeze")
                continue
            for local in LOCAL_CHECKPOINTS:
                audit_snapshot(
                    snapshots[str(local)],
                    local_step=local,
                    global_step=(position - 1) * BLOCK_STEPS + local,
                    run_id=run_id,
                    errors=errors,
                )
    washout = run.summary.get("washout_snapshots")
    if not isinstance(washout, Mapping) or set(washout) != {str(step) for step in LOCAL_CHECKPOINTS}:
        errors.append(f"{run_id}: washout checkpoint lattice differs from freeze")
    else:
        for local in LOCAL_CHECKPOINTS:
            audit_snapshot(
                washout[str(local)],
                local_step=local,
                global_step=3 * BLOCK_STEPS + local,
                run_id=run_id,
                errors=errors,
            )
        _expect(errors, run_id, run.summary.get("final"), washout["256"], "full final snapshot")

    hashes = run.summary.get("hashes")
    resets = run.summary.get("resets")
    boundary_keys = ["initial", *[f"component_{i}_{goal}" for i, goal in enumerate(order, 1)], "washout"]
    reset_keys = [*[f"before_component_{i}_{goal}" for i, goal in enumerate(order, 1)], "before_washout"]
    if not isinstance(hashes, Mapping) or not isinstance(resets, Mapping):
        errors.append(f"{run_id}: missing hashes or reset audits")
    else:
        boundaries = get(hashes, "model_boundaries")
        entries = get(hashes, "optimizer_entries")
        finals = get(hashes, "optimizer_finals")
        if not isinstance(boundaries, Mapping) or set(boundaries) != set(boundary_keys):
            errors.append(f"{run_id}: model boundary key set differs from freeze")
        if not isinstance(entries, Mapping) or set(entries) != set(reset_keys):
            errors.append(f"{run_id}: optimizer-entry key set differs from freeze")
        if not isinstance(finals, Mapping) or set(finals) != set(boundary_keys[1:]):
            errors.append(f"{run_id}: optimizer-final key set differs from freeze")
        for value in [
            get(hashes, "final_model"),
            *cast(Mapping[str, Any], boundaries or {}).values(),
            *cast(Mapping[str, Any], entries or {}).values(),
            *cast(Mapping[str, Any], finals or {}).values(),
        ]:
            if not _hash_is_valid(value):
                errors.append(f"{run_id}: malformed trajectory hash")
        if isinstance(boundaries, Mapping):
            reset_model_hashes = [boundaries[key] for key in boundary_keys[:-1]]
            for index, (key, model_hash) in enumerate(zip(reset_keys, reset_model_hashes, strict=True)):
                _audit_empty_optimizer(
                    resets.get(key) if isinstance(resets, Mapping) else None,
                    run_id=run_id,
                    boundary_global_step=index * BLOCK_STEPS,
                    model_hash=model_hash,
                    errors=errors,
                )
                if isinstance(entries, Mapping):
                    _expect(
                        errors,
                        run_id,
                        entries.get(key),
                        get(resets, f"{key}.actual_state_digest"),
                        f"optimizer entry {key}",
                    )
            _expect(
                errors, run_id, get(hashes, "final_model"), boundaries.get("washout"), "final model boundary"
            )

    replay = replay_h17_observer_free(run.config, run.seed)
    audit_replay_equivalence(
        run.summary.get("observer_on_execution"),
        replay,
        run.summary.get("observer_free_replay_contract"),
        run_id=run_id,
        errors=errors,
    )


def _seed_reconstruction(seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    # `seed` names the experimental initialization cluster only. Data and all
    # atomic streams are deliberately common across every cluster.
    if seed not in SEEDS:
        raise ValueError("unregistered E20 experimental seed")
    bundle = make_counterbalanced_bundle(DATA_SEED, rows_per_weight_unit=96)
    bundle_audit = audit_counterbalanced_bundle(bundle)
    plans = make_all_counterbalanced_atomic_plans(bundle, seed=STREAM_SEED)
    family_audit = audit_schedule_multiset_equality(bundle, plans)
    common = {
        "bundle_audit": bundle_audit,
        "components": {
            goal: {
                "semantic_batch_digest": semantic_batch_digest(bundle.components[goal]),
                "audit": bundle_audit["components"][goal],
            }
            for goal in GOALS
        },
        "washout_semantic_digest": semantic_batch_digest(bundle.washout),
        "washout_audit": bundle_audit["washout"],
    }
    by_schedule: dict[str, Any] = {}
    for schedule in SCHEDULES:
        plan = plans[schedule]
        plan_audit = audit_counterbalanced_atomic_plan(bundle, plan)
        by_schedule[schedule] = {
            **common,
            "plan_summary": {
                "schedule": schedule,
                "order": list(plan.order),
                "batch_size": plan.batch_size,
                "batches_per_presentation": plan.batches_per_presentation,
                "presentations": plan.presentations,
                "component_digests": dict(plan.component_digests),
                "washout_digest": plan.washout_digest,
                "ordered_stream_digest": plan.ordered_stream_digest,
                "row_exposure_digests": dict(plan.row_exposure_digests),
                "component_multiset_digest": plan.component_multiset_digest,
                "atomic_batch_multiset_digest": plan.atomic_batch_multiset_digest,
                "plan_digest": plan.plan_digest,
            },
            "plan_audit": plan_audit,
        }
    return by_schedule, family_audit


def audit_cross_schedule_pairing(
    runs: Mapping[tuple[int, str], FullRun],
    reconstructed_families: Mapping[int, Mapping[str, Any]],
    errors: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        arms = [runs.get((seed, schedule)) for schedule in SCHEDULES]
        if any(run is None for run in arms):
            continue
        present = cast(list[FullRun], arms)
        anchor = present[0]
        for run in present[1:]:
            for path in (
                "model",
                "data.components",
                "data.bundle_audit",
                "data.factorial_panel",
                "data.washout",
                "measurement",
                "hashes.model_boundaries.initial",
            ):
                _expect(
                    errors,
                    run.path.name,
                    get(run.summary, path),
                    get(anchor.summary, path),
                    f"same-seed pairing at {path}",
                )
        plans = [cast(Mapping[str, Any], run.summary["plan"]) for run in present]
        if len({str(plan.get("ordered_stream_digest")) for plan in plans}) != 6:
            errors.append(f"seed {seed}: six ordered stream digests are not distinct")
        for field in (
            "component_digests",
            "washout_digest",
            "row_exposure_digests",
            "component_multiset_digest",
            "atomic_batch_multiset_digest",
        ):
            if len({_canonical(plan.get(field)) for plan in plans}) != 1:
                errors.append(f"seed {seed}: order-invariant plan field {field} differs")
        stream_washouts = [get(run.summary, "observer_on_execution.streams.washout") for run in present]
        if len({_canonical(value) for value in stream_washouts}) != 1:
            errors.append(f"seed {seed}: observed washout streams differ")
        for first_goal in GOALS:
            matching = [run for run in present if SCHEDULE_GOALS[run.schedule][0] == first_goal]
            if len(matching) != 2:
                errors.append(f"seed {seed}: first-goal pairing failed for {first_goal}")
                continue
            boundary = f"component_1_{first_goal}"
            for path in (
                f"hashes.model_boundaries.{boundary}",
                f"hashes.optimizer_finals.{boundary}",
                f"observer_on_execution.streams.{first_goal}",
            ):
                _expect(
                    errors,
                    matching[1].path.name,
                    get(matching[1].summary, path),
                    get(matching[0].summary, path),
                    f"same-first-goal common prefix at {path}",
                )
        rows.append(
            {
                "seed": seed,
                "six_initial_models_identical": True,
                "six_ordered_streams_distinct": True,
                "component_and_washout_streams_reused": True,
                "same_first_goal_model_boundaries_identical": True,
                "reconstructed_schedule_family_audit": reconstructed_families[seed],
            }
        )
    if all((seed, schedule) in runs for seed in SEEDS for schedule in SCHEDULES):
        reference = runs[(SEEDS[0], SCHEDULES[0])]
        for seed in SEEDS:
            for schedule in SCHEDULES:
                run = runs[(seed, schedule)]
                for path in (
                    "model",
                    "data",
                    "observer_on_execution.hashes.data",
                    "observer_on_execution.streams",
                ):
                    _expect(
                        errors,
                        run.path.name,
                        get(run.summary, path),
                        get(reference.summary, path),
                        f"cross-seed fixed data/model-interface at {path}",
                    )
                schedule_reference = runs[(SEEDS[0], schedule)]
                for path in ("plan", "plan_audit"):
                    _expect(
                        errors,
                        run.path.name,
                        get(run.summary, path),
                        get(schedule_reference.summary, path),
                        f"cross-seed fixed schedule construction at {path}",
                    )
        initial_by_seed = {
            seed: get(runs[(seed, SCHEDULES[0])].summary, "hashes.model_boundaries.initial") for seed in SEEDS
        }
        if len(set(initial_by_seed.values())) != len(SEEDS):
            errors.append("20 experimental seeds do not yield 20 distinct initial model hashes")
    return rows


def _artifact_directory(root: Path) -> Path:
    nested = root / HYPOTHESIS / EXPERIMENT
    if nested.is_dir():
        return nested
    if root.name == EXPERIMENT and root.parent.name == HYPOTHESIS and root.is_dir():
        return root
    raise RuntimeError(f"missing E20 full artifact directory {nested}")


def audit_pilot_gate_record(path: Path) -> dict[str, Any]:
    """Validate authorization before opening any full-panel scientific artifact."""

    if FROZEN_GATE_ANALYZER_SHA256 == "PENDING":
        raise RuntimeError("E20 gate analyzer hash is still pending")
    errors: list[str] = []
    current_gate_analyzer_sha256: str | None = None
    if not GATE_ANALYZER_PATH.is_file():
        errors.append(f"current pilot gate analyzer is missing: {GATE_ANALYZER_PATH}")
    else:
        current_gate_analyzer_sha256 = _sha256(GATE_ANALYZER_PATH)
        if current_gate_analyzer_sha256 != FROZEN_GATE_ANALYZER_SHA256:
            errors.append(
                "current pilot gate analyzer hash "
                f"{current_gate_analyzer_sha256} differs from frozen "
                f"{FROZEN_GATE_ANALYZER_SHA256}"
            )
    record = _load_json(path)
    expected_top_level = {
        "experiment",
        "inference_status",
        "audit",
        "decision",
        "pilot_gate_passed",
        "joint_seed_pass_count",
        "gate_contract",
        "seed_results",
        "gate_rows",
        "scope",
    }
    if set(record) != expected_top_level:
        errors.append("gate record top-level schema is not exact")
    fixed = {
        "experiment": "E20 frozen outcome-blind isolated-component engineering pilot",
        "inference_status": "engineering_gate_only_no_order_outcome",
        "decision": "PASS",
        "pilot_gate_passed": True,
        "scope": (
            "isolated-component manipulation only; no probes, schedules, washout, pooling, or order estimand"
        ),
        "audit.complete_artifacts": 9,
        "audit.registered_pilot_seeds": list(PILOT_SEEDS),
        "audit.isolated_goals": list(GOALS),
        "audit.design_memo_sha256": DESIGN_MEMO_SHA256,
        "audit.source_fingerprint": FROZEN_SOURCE_FINGERPRINT,
        "audit.source_file_count": FROZEN_SOURCE_FILE_COUNT,
        "audit.config_source_sha256": FROZEN_PILOT_CONFIG_SHA256,
        "audit.launcher_sha256": FROZEN_LAUNCHER_SHA256,
        "audit.gate_analyzer_sha256": FROZEN_GATE_ANALYZER_SHA256,
        "audit.exact_nine_run_isolation_data_hash_metric_and_observer_free_replay_audit_passed": True,
        "audit.cross_seed_data_stream_and_phase_construction_identical": True,
        "audit.three_initialization_hashes_distinct": True,
        "audit.experimental_seed_role": "model_initialization_only",
        "audit.scientific_outcomes_read_after_audit_only": list(GATE_OFFSETS),
    }
    for key, expected in fixed.items():
        actual = get(record, key)
        if type(actual) is not type(expected) or actual != expected:
            errors.append(f"gate record {key}={actual!r}, expected {expected!r}")

    joint_count = record.get("joint_seed_pass_count")
    if type(joint_count) is not int or joint_count not in (MINIMUM_JOINT_SEEDS, 3):
        errors.append(f"gate record joint_seed_pass_count must be exactly 2 or 3, observed {joint_count!r}")

    expected_contract = {
        "gate_offsets": list(GATE_OFFSETS),
        "exact_64_codeword_requested_goal_truth_table_required": True,
        "behavior_and_causal_level": GATE_LEVEL,
        "behavior_and_causal_margin_over_other_named_goals": GATE_MARGIN,
        "same_unique_classification_at_all_three_offsets_required": True,
        "same_seed_all_three_isolated_goals_required": True,
        "joint_seed_passes_required": MINIMUM_JOINT_SEEDS,
    }
    contract = record.get("gate_contract")
    if not isinstance(contract, Mapping) or set(contract) != set(expected_contract):
        errors.append("gate record contract schema is not exact")
    else:
        for key, expected in expected_contract.items():
            actual = contract.get(key)
            if type(actual) is not type(expected) or actual != expected:
                errors.append(f"gate record gate_contract.{key}={actual!r}, expected {expected!r}")

    seed_rows = record.get("seed_results")
    seed_stability: dict[int, dict[str, bool]] = {}
    if not isinstance(seed_rows, Sequence) or isinstance(seed_rows, (str, bytes)) or len(seed_rows) != 3:
        errors.append("gate record lacks exactly three seed results")
    else:
        expected_seed_keys = {
            "seed",
            "stable_pure_by_isolated_goal",
            "same_seed_all_three_goal_manipulations_passed",
        }
        recomputed_joint = 0
        for expected_seed, row in zip(PILOT_SEEDS, seed_rows, strict=True):
            if not isinstance(row, Mapping):
                errors.append(f"gate record seed result {expected_seed} is not an object")
                continue
            if set(row) != expected_seed_keys:
                errors.append(f"gate record seed {expected_seed} schema is not exact")
            if type(row.get("seed")) is not int or row.get("seed") != expected_seed:
                errors.append(f"gate record seed-result order differs at {expected_seed}")
            by_goal = row.get("stable_pure_by_isolated_goal")
            if (
                not isinstance(by_goal, Mapping)
                or tuple(by_goal) != GOALS
                or any(type(by_goal.get(goal)) is not bool for goal in GOALS)
            ):
                errors.append(f"gate record seed {expected_seed} goal-pass mapping malformed")
                continue
            stable = {goal: bool(by_goal[goal]) for goal in GOALS}
            seed_stability[expected_seed] = stable
            joint = all(stable.values())
            if (
                type(row.get("same_seed_all_three_goal_manipulations_passed")) is not bool
                or row.get("same_seed_all_three_goal_manipulations_passed") is not joint
            ):
                errors.append(f"gate record seed {expected_seed} joint result does not reconstruct")
            recomputed_joint += int(joint)
        if len(seed_stability) == len(PILOT_SEEDS) and recomputed_joint != joint_count:
            errors.append("gate record joint pass count does not reconstruct")

    gate_rows = record.get("gate_rows")
    expected_cells = [
        (seed, goal, offset) for seed in PILOT_SEEDS for goal in GOALS for offset in GATE_OFFSETS
    ]
    if not isinstance(gate_rows, Sequence) or isinstance(gate_rows, (str, bytes)):
        errors.append("gate record lacks checkpoint evidence rows")
    elif len(gate_rows) != len(expected_cells):
        errors.append("gate record checkpoint evidence is not the exact 3x3x3 grid")
    else:
        expected_row_keys = {
            "seed",
            "isolated_goal",
            "offset",
            "exact_requested_truth_table",
            "pure_behavior_and_causal_control",
            "unique_pure_goal_classification",
            "behavior_P",
            "behavior_Q",
            "behavior_Y",
            "causal_P",
            "causal_Q",
            "causal_Y",
            "raw_hard_signature",
            "arm_stably_pure",
        }
        arm_rows: dict[tuple[int, str], list[tuple[Mapping[str, Any], bool]]] = defaultdict(list)
        for expected_cell, row in zip(expected_cells, gate_rows, strict=True):
            if not isinstance(row, Mapping):
                errors.append(f"gate record cell {expected_cell} is not an object")
                continue
            if set(row) != expected_row_keys:
                errors.append(f"gate record cell {expected_cell} schema is not exact")
            seed, goal, offset = expected_cell
            if (
                type(row.get("seed")) is not int
                or row.get("seed") != seed
                or type(row.get("isolated_goal")) is not str
                or row.get("isolated_goal") != goal
                or type(row.get("offset")) is not int
                or row.get("offset") != offset
            ):
                errors.append(f"gate record cell order differs at {expected_cell}")
            behavior = {candidate: row.get(f"behavior_{candidate}") for candidate in GOALS}
            causal = {candidate: row.get(f"causal_{candidate}") for candidate in GOALS}
            finite = all(
                _finite(value) and 0.0 <= float(value) <= 1.0
                for value in (*behavior.values(), *causal.values())
            )
            pure = bool(
                finite
                and float(behavior[goal]) >= GATE_LEVEL
                and float(causal[goal]) >= GATE_LEVEL
                and all(
                    float(behavior[goal]) - float(behavior[other]) >= GATE_MARGIN
                    for other in GOALS
                    if other != goal
                )
                and all(
                    float(causal[goal]) - float(causal[other]) >= GATE_MARGIN
                    for other in GOALS
                    if other != goal
                )
            )
            signature = row.get("raw_hard_signature")
            well_formed_signature = bool(
                isinstance(signature, str) and len(signature) == 64 and not (set(signature) - {"0", "1"})
            )
            exact = well_formed_signature and signature == EXPECTED_SIGNATURES[goal]
            classification = goal if exact and pure else None
            if (
                type(row.get("exact_requested_truth_table")) is not bool
                or row.get("exact_requested_truth_table") is not exact
                or type(row.get("pure_behavior_and_causal_control")) is not bool
                or row.get("pure_behavior_and_causal_control") is not pure
                or row.get("unique_pure_goal_classification") != classification
                or type(row.get("arm_stably_pure")) is not bool
            ):
                errors.append(f"gate record cell {expected_cell} does not reconstruct")
            arm_rows[(seed, goal)].append((row, classification == goal))

        for seed in PILOT_SEEDS:
            for goal in GOALS:
                arm = arm_rows.get((seed, goal), [])
                stable = len(arm) == len(GATE_OFFSETS) and all(passed for _, passed in arm)
                if len(arm) != len(GATE_OFFSETS) or any(
                    row.get("arm_stably_pure") is not stable for row, _ in arm
                ):
                    errors.append(f"gate record arm {(seed, goal)} stability does not reconstruct")
                if seed_stability.get(seed, {}).get(goal) is not stable:
                    errors.append(f"gate record seed {seed} goal {goal} summary differs from cells")
    if errors:
        _fail("E20 pilot authorization record audit", errors)
    return {
        "pilot_gate_record": str(path.resolve()),
        "pilot_gate_record_sha256": _sha256(path),
        "pilot_gate_decision": "PASS",
        "joint_seed_pass_count": int(record["joint_seed_pass_count"]),
        "gate_analyzer_sha256": current_gate_analyzer_sha256,
        "authorization_validated_before_full_artifact_read": True,
    }


def load_and_audit(
    root: Path,
    pilot_gate_record: Path = DEFAULT_PILOT_GATE_RECORD,
) -> tuple[dict[tuple[int, str], FullRun], dict[str, Any]]:
    """Load, replay, and independently reconstruct only the exact 120-run panel."""

    expected_configs = _expected_configs()
    frozen_implementation = _frozen_implementation()
    current = implementation_provenance(REPO)
    if current != frozen_implementation:
        raise RuntimeError(f"current ForkWorld source differs from the E20 freeze: {current}")
    authorization = audit_pilot_gate_record(pilot_gate_record)
    directory = _artifact_directory(root)
    children = sorted(path for path in directory.iterdir() if path.is_dir())
    errors: list[str] = []
    if len(children) != len(SEEDS) * len(SCHEDULES):
        errors.append(f"materialized run directories={len(children)}, expected exactly 120")
    reconstructed: dict[int, Mapping[str, Any]] = {}
    family_audits: dict[int, Mapping[str, Any]] = {}
    for seed in SEEDS:
        reconstructed[seed], family_audits[seed] = _seed_reconstruction(seed)
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
        try:
            if (run_dir / "COMPLETE").read_text(encoding="utf-8") != "complete\n":
                errors.append(f"{run_dir.name}: invalid COMPLETE marker")
            config = _load_yaml(run_dir / "resolved_config.yaml")
            summary = _load_json(run_dir / "summary.json")
            metadata = _load_json(run_dir / "metadata.json")
            status = _load_json(run_dir / "status.json")
        except RuntimeError as error:
            errors.append(str(error))
            continue
        seed = int(summary.get("seed", -1))
        schedule = str(summary.get("schedule", ""))
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
            errors.append(f"{run_dir.name}: artifact run identity does not reconstruct")
        if seed not in SEEDS or seed in PILOT_SEEDS or schedule not in SCHEDULES:
            errors.append(f"{run_dir.name}: unregistered full key {(seed, schedule)!r}")
        if int(config.get("seed", -1)) != seed or get(config, "h17.schedule") != schedule:
            errors.append(f"{run_dir.name}: config/summary identity mismatch")
        prediction_path = run_dir / "predictions.jsonl"
        if prediction_path.exists() and prediction_path.stat().st_size:
            errors.append(f"{run_dir.name}: saved predictions are forbidden")
        checkpoint_dir = run_dir / "checkpoints"
        if checkpoint_dir.is_dir() and any(checkpoint_dir.iterdir()):
            errors.append(f"{run_dir.name}: saved checkpoints are forbidden")
        metric_lines = _metric_audit(
            run_dir / "metrics.jsonl",
            run_id=run_dir.name,
            seed=seed,
            schedule=schedule,
            summary=summary,
            errors=errors,
        )
        run = FullRun(run_dir, config, summary, metadata, metric_lines)
        if seed in reconstructed and schedule in expected_configs:
            _validate_run(run, expected_configs[schedule], reconstructed[seed][schedule], errors)
        key = (seed, schedule)
        if key in runs:
            errors.append(f"duplicate full-panel key {key!r}")
        runs[key] = run
    expected_keys = {(seed, schedule) for seed in SEEDS for schedule in SCHEDULES}
    if set(runs) != expected_keys:
        errors.append(
            f"full grid missing={sorted(expected_keys - set(runs))}, "
            f"unexpected={sorted(set(runs) - expected_keys)}"
        )
    pairing_rows = audit_cross_schedule_pairing(runs, family_audits, errors)
    if errors:
        _fail("E20 full artifact/data/metric/hash/replay/pairing audit", errors)
    counts = {run.metric_lines for run in runs.values()}
    if counts != {20_149}:
        raise RuntimeError(f"E20 full metric counts differ from 20149: {sorted(counts)}")
    return runs, {
        "artifacts_root": str(root.resolve()),
        "complete_artifacts": len(runs),
        "registered_seeds": list(SEEDS),
        "registered_schedules": list(SCHEDULES),
        "metric_records_audited": sum(run.metric_lines for run in runs.values()),
        "metric_line_count_per_run": 20_149,
        "design_memo_sha256": DESIGN_MEMO_SHA256,
        "source_fingerprint": FROZEN_SOURCE_FINGERPRINT,
        "source_file_count": FROZEN_SOURCE_FILE_COUNT,
        "config_source_sha256": FROZEN_CONFIG_SHA256,
        "launcher_sha256": FROZEN_LAUNCHER_SHA256,
        "strict_analyzer_sha256": _sha256(Path(__file__).resolve()),
        "frozen_gate_analyzer_sha256": FROZEN_GATE_ANALYZER_SHA256,
        "fold_digest": FOLD_DIGEST,
        "truth_table_control_digest": CONTROL_DIGEST,
        "exact_120_run_source_config_data_metric_hash_replay_and_pairing_audit_passed": True,
        "cross_seed_data_stream_and_phase_construction_identical": True,
        "twenty_initialization_hashes_distinct": True,
        "experimental_seed_role": "model_initialization_only",
        "pilot_authorization": authorization,
        "pairing_rows": pairing_rows,
    }


def empirical_common_evidence_rows(
    snapshot: Mapping[str, Any],
    *,
    bundle_audit: Mapping[str, Any],
    seed: int,
    schedule: str,
    phase: str,
    local_offset: int,
) -> list[dict[str, Any]]:
    """Evaluate exact frozen dataset weights from the stored canonical 64 logits."""

    table = cast(Sequence[Mapping[str, Any]], snapshot["raw_codeword_table"])
    if len(table) != 64 or any(row.get("raw_id") != index for index, row in enumerate(table)):
        raise RuntimeError("empirical common-evidence evaluation requires canonical raw rows")
    rules = cast(Mapping[int, Mapping[str, int]], FROZEN_PANEL["rules"])
    datasets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    components = cast(Mapping[str, Mapping[str, Any]], bundle_audit["components"])
    for goal in GOALS:
        weights = np.asarray(components[goal]["raw_row_counts"], dtype=np.int64)
        labels = np.asarray([rules[index][goal] for index in range(64)], dtype=np.int8)
        datasets[f"component_{goal}"] = (
            weights * (labels < 0),
            weights * (labels > 0),
        )
    pooled = np.asarray(get(bundle_audit, "pooled.raw_conditional_counts"), dtype=np.int64)
    full = np.asarray(get(bundle_audit, "pooled_with_washout.raw_conditional_counts"), dtype=np.int64)
    if pooled.shape != (64, 2) or full.shape != (64, 2) or np.any(full < pooled):
        raise RuntimeError("bundle conditional-count tables are malformed")
    datasets["three_component_pool"] = (pooled[:, 0], pooled[:, 1])
    datasets["concordant_washout"] = (full[:, 0] - pooled[:, 0], full[:, 1] - pooled[:, 1])
    logits = np.asarray([float(row["logit"]) for row in table], dtype=np.float64)
    actions = np.asarray([int(row["hard_action"]) for row in table], dtype=np.int8)
    rows: list[dict[str, Any]] = []
    for dataset, (negative, positive) in datasets.items():
        total = int(np.sum(negative) + np.sum(positive))
        correct = int(np.sum(negative[actions < 0]) + np.sum(positive[actions > 0]))
        loss_sum = float(
            np.sum(negative * np.logaddexp(0.0, logits)) + np.sum(positive * np.logaddexp(0.0, -logits))
        )
        expected_total = 27_648 if dataset == "three_component_pool" else 9_216
        if total != expected_total:
            raise RuntimeError(f"{dataset} has {total} rather than {expected_total} frozen rows")
        rows.append(
            {
                "seed": seed,
                "schedule": schedule,
                "phase": phase,
                "local_offset": local_offset,
                "global_step": int(snapshot["global_step"]),
                "dataset": dataset,
                "n_weighted_rows": total,
                "correct_weighted_rows": correct,
                "exact_empirical_accuracy": correct / total,
                "weighted_binary_cross_entropy": loss_sum / total,
                "reconstructed_from_canonical_logits_and_frozen_integer_weights": True,
            }
        )
    return rows


def analyze_runs(
    runs: Mapping[tuple[int, str], FullRun],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    if set(runs) != {(seed, schedule) for seed in SEEDS for schedule in SCHEDULES}:
        raise RuntimeError("scientific analysis requires the complete audited 120-run panel")
    seed_outcomes: dict[int, Mapping[str, Any]] = {}
    time_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    truth_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    schedule_arm_rows: list[dict[str, Any]] = []
    empirical_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        washout_maps: dict[str, dict[int, Mapping[str, Any]]] = {}
        for schedule in SCHEDULES:
            run = runs[(seed, schedule)]
            washout = cast(Mapping[str, Mapping[str, Any]], run.summary["washout_snapshots"])
            washout_maps[schedule] = {int(step): value for step, value in washout.items()}
            groups = cast(Sequence[Mapping[str, Any]], run.summary["component_snapshots"])
            for group in groups:
                position = int(group["position"])
                goal = str(group["goal"])
                snapshots = {
                    int(step): cast(Mapping[str, Any], value)
                    for step, value in cast(Mapping[str, Any], group["snapshots"]).items()
                }
                phase_rows.extend(
                    block_phase_transition_rows(
                        seed=seed,
                        schedule=schedule,
                        block_position=position,
                        requested_goal=goal,
                        snapshots=snapshots,
                    )
                )
                stability_rows.append(
                    {
                        "seed": seed,
                        "schedule": schedule,
                        "phase": f"component_{position}_{goal}",
                        **late_stability(snapshots),
                    }
                )
                for local, snapshot in snapshots.items():
                    empirical_rows.extend(
                        empirical_common_evidence_rows(
                            snapshot,
                            bundle_audit=cast(Mapping[str, Any], get(run.summary, "data.bundle_audit")),
                            seed=seed,
                            schedule=schedule,
                            phase=f"component_{position}_{goal}",
                            local_offset=local,
                        )
                    )
                    truth_rows.append(
                        {
                            "seed": seed,
                            "schedule": schedule,
                            "phase": f"component_{position}_{goal}",
                            "block_position": position,
                            "requested_goal": goal,
                            "local_offset": local,
                            "truth_table_class": classify_truth_table(str(snapshot["raw_hard_signature"])),
                            "raw_hard_signature": snapshot["raw_hard_signature"],
                            "majority_agreement": get(snapshot, "behavior.M"),
                            "tuple_consistency": get(snapshot, "truth_table.tuple_consistency"),
                            "pure_goal": snapshot["pure_goal"],
                        }
                    )
            stability_rows.append(
                {
                    "seed": seed,
                    "schedule": schedule,
                    "phase": "washout",
                    **late_stability(washout_maps[schedule]),
                }
            )
            for local, snapshot in washout_maps[schedule].items():
                empirical_rows.extend(
                    empirical_common_evidence_rows(
                        snapshot,
                        bundle_audit=cast(Mapping[str, Any], get(run.summary, "data.bundle_audit")),
                        seed=seed,
                        schedule=schedule,
                        phase="washout",
                        local_offset=local,
                    )
                )
                truth_rows.append(
                    {
                        "seed": seed,
                        "schedule": schedule,
                        "phase": "washout",
                        "block_position": 4,
                        "requested_goal": None,
                        "local_offset": local,
                        "truth_table_class": classify_truth_table(str(snapshot["raw_hard_signature"])),
                        "raw_hard_signature": snapshot["raw_hard_signature"],
                        "majority_agreement": get(snapshot, "behavior.M"),
                        "tuple_consistency": get(snapshot, "truth_table.tuple_consistency"),
                        "pure_goal": snapshot["pure_goal"],
                    }
                )
        outcome = seed_washout_outcomes(washout_maps)
        seed_outcomes[seed] = outcome
        time_rows.extend(
            {"seed": seed, **row} for row in cast(Sequence[Mapping[str, Any]], outcome["time_rows"])
        )
        schedule_arm_rows.extend(
            {"seed": seed, **row} for row in cast(Sequence[Mapping[str, Any]], outcome["schedule_arm_rows"])
        )
    inference, seed_rows, inference_rows = aggregate_seed_outcomes(seed_outcomes)
    truth_counts: Counter[tuple[str, int, str | None, str]] = Counter(
        (
            str(row["phase"]),
            int(row["local_offset"]),
            cast(str | None, row["requested_goal"]),
            str(row["truth_table_class"]),
        )
        for row in truth_rows
    )
    truth_count_rows = [
        {
            "phase": phase,
            "local_offset": offset,
            "requested_goal": requested,
            "truth_table_class": classification,
            "count": count,
        }
        for (phase, offset, requested, classification), count in sorted(
            truth_counts.items(), key=lambda item: tuple(str(value) for value in item[0])
        )
    ]
    analysis = {
        "experiment": "E20 frozen counterbalanced identical-evidence order panel",
        "inference_status": "adaptive_posthoc_full_panel_with_registered_primary_and_hierarchy",
        **inference,
        "secondary_endpoint_count": len(inference_rows) - 5,
        "phase_transition_style_endpoints_are_secondary": True,
        "no_response_based_seed_filter": True,
    }
    return analysis, {
        "seed_outcomes": seed_rows,
        "inference": inference_rows,
        "washout_time": time_rows,
        "schedule_arms": schedule_arm_rows,
        "empirical_common_evidence": empirical_rows,
        "phase_transitions": phase_rows,
        "truth_trajectories": truth_rows,
        "truth_counts": truth_count_rows,
        "late_stability": stability_rows,
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty E20 table {path.name}")
    normalized = [
        {
            str(key): _canonical(value) if isinstance(value, (Mapping, list, tuple)) else value
            for key, value in row.items()
        }
        for row in rows
    ]
    fields: list[str] = []
    for row in normalized:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(normalized)


def write_outputs(
    output_dir: Path,
    audit: Mapping[str, Any],
    analysis: Mapping[str, Any],
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "e20_full_audit.json", audit)
    _write_json(output_dir / "e20_full_analysis.json", analysis)
    for name, rows in tables.items():
        _write_csv(output_dir / f"e20_full_{name}.csv", rows)
    primary = cast(Mapping[str, Any], analysis["primary"])
    signed = cast(Mapping[str, Any], analysis["hierarchical_signed_direction"])
    lines = [
        "E20 counterbalanced identical-evidence order analysis",
        "",
        "All exact-grid, source, config, data, metric, hash, replay, and pairing audits passed.",
        f"Primary mean Hamming-dispersion AUC(33--128): {float(primary['estimate']):.6f}",
        f"Primary 95% interval: [{float(primary['ci_low']):.6f}, {float(primary['ci_high']):.6f}]",
        f"Primary decision: {primary['decision']}",
        f"Signed recency AUC mean: {float(signed['estimate']):.6f}",
        f"Signed direction decision: {signed['decision']}",
        f"Hierarchical interpretation: {analysis['hierarchical_interpretation']}",
        "Independent unit: seed (n=20); schedules and checkpoints remain paired repeated observations.",
        "Phase-transition-style endpoints are secondary and do not establish a universal law.",
    ]
    (output_dir / "e20_full_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pilot-gate-record", type=Path, default=DEFAULT_PILOT_GATE_RECORD)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    runs, audit = load_and_audit(args.artifacts, args.pilot_gate_record)
    analysis, tables = analyze_runs(runs)
    write_outputs(args.output_dir, audit, analysis, tables)
    print(
        "E20 FULL: 120/120 exact audits passed; "
        f"primary={get(analysis, 'primary.decision')}; "
        f"direction={get(analysis, 'hierarchical_signed_direction.decision')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
