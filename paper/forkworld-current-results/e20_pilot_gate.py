#!/usr/bin/env python3
"""Outcome-blind isolated-component engineering gate for Forkworld E20.

This program can accept only the frozen nine isolated pilot artifacts.  Its
module contains no schedule, washout, pooled-order, or order-estimand
construction.  Structural and observer-free replay audits complete before the
gate reads the three registered late isolated-component checkpoints.
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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml  # type: ignore[import-untyped]

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forkworld.artifacts import implementation_provenance  # noqa: E402
from forkworld.config import expand_sweep, load_config  # noqa: E402
from forkworld.counterbalanced_order import (  # noqa: E402
    audit_counterbalanced_component,
    audit_counterbalanced_factorial_panel,
    audit_counterbalanced_fold_control,
    make_atomic_component_stream,
    make_counterbalanced_component,
    make_counterbalanced_factorial_panel,
)
from forkworld.handoff import semantic_batch_digest, stable_state_digest  # noqa: E402
from forkworld.protocols_counterbalanced import replay_h17_observer_free  # noqa: E402

DESIGN_MEMO = HERE / "e20_repaired_order_design.md"
DESIGN_MEMO_SHA256 = "1386cb401aef8ed187389dc84aa210b09b564436bc16c9246bbd22fce257529a"
CONFIG_PATH = REPO / "configs" / "e20_counterbalanced_order_pilot.yaml"
LAUNCHER_PATH = REPO / "runs" / "21_counterbalanced_order.sh"
DEFAULT_ARTIFACTS = REPO / "artifacts-e20-pilot"
DEFAULT_OUTPUT = HERE / "derived"
EXPERIMENT = "counterbalanced_identical_evidence_order_pilot"
HYPOTHESIS = "h17"
PILOT_SEEDS = (563, 569, 571)
GOALS = ("P", "Q", "Y")
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
GATE_OFFSETS = (161, 222, 256)
LEVEL = 0.90
MARGIN = 0.10
MINIMUM_JOINT_SEEDS = 2
DATA_SEED = 171_000_001
STREAM_SEED = 171_000_002
PHASE_SEEDS = {
    "component_P": 171_000_101,
    "component_Q": 171_000_102,
    "component_Y": 171_000_103,
    "washout": 171_000_104,
}
FOLD_DIGEST = "25d5e9584a2ea74d42d48a673645a3b34c1b76e75e8230bf5722985e66968ec7"
CONTROL_DIGEST = "c7a186308102e807e594fd76b3fc5e7ef1678314a95de067b6fd347b390e53d0"
NON_SCIENTIFIC_RUN_FIELDS = frozenset({"output_root", "resume", "seeds"})
ARTIFACT_SCHEMA_VERSION = 1
SOURCE_FINGERPRINT_SCHEMA_VERSION = 1

# Patched after source/config/launcher freeze and before the first pilot model.
FROZEN_SOURCE_FINGERPRINT = "4ca022c5b2d2d9a75c8d443cdc0e422d178989bae825c2c2d37d41fc7a14e539"
FROZEN_SOURCE_FILE_COUNT = 34
FROZEN_CONFIG_SHA256 = "868981c1f250c161ef22ff44ad5f3dfa33cd564a3f1169e194e4f20cffc2aefc"


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


def reconstruct_frozen_panel() -> dict[str, Any]:
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
        for fold, raw_id in enumerate(sorted(raw_ids)):
            folds[raw_id] = fold
            records.append({"fold": fold, "raw_id": raw_id, "tuple_id": tuple_id})
    records.sort(key=lambda row: row["raw_id"])
    control = [
        1 if (folds[raw_id] - rules[raw_id]["tuple_id"]) % 8 in {0, 1, 3, 4} else -1 for raw_id in range(64)
    ]
    fold_digest = hashlib.sha256(_canonical(records).encode()).hexdigest()
    control_digest = hashlib.sha256(_canonical(control).encode()).hexdigest()
    errors: list[str] = []
    if fold_digest != FOLD_DIGEST or control_digest != CONTROL_DIGEST:
        errors.append("fold/control digest differs from the design memo")
    if Counter(folds) != Counter({fold: 8 for fold in range(8)}):
        errors.append("folds are not balanced")
    if Counter(control) != Counter({-1: 32, 1: 32}):
        errors.append("control is not sign balanced")
    for tuple_id, raw_ids in tuple_members.items():
        if Counter(folds[raw_id] for raw_id in raw_ids) != Counter(range(8)):
            errors.append(f"tuple {tuple_id} does not span all folds")
        if Counter(control[raw_id] for raw_id in raw_ids) != Counter({-1: 4, 1: 4}):
            errors.append(f"control is not balanced in tuple {tuple_id}")
    for goal in GOALS:
        if sum(control[raw_id] == rules[raw_id][goal] for raw_id in range(64)) != 32:
            errors.append(f"control is not chance-aligned with {goal}")
    if errors:
        _fail("E20 pilot fold/control reconstruction", errors)
    return {
        "records": records,
        "folds": folds,
        "control": control,
        "fold_digest": fold_digest,
        "control_digest": control_digest,
        "rules": rules,
    }


FROZEN_PANEL = reconstruct_frozen_panel()
EXPECTED_SIGNATURES = {
    goal: "".join(
        "1" if cast(Mapping[int, Mapping[str, int]], FROZEN_PANEL["rules"])[raw_id][goal] > 0 else "0"
        for raw_id in range(64)
    )
    for goal in GOALS
}


@dataclass(frozen=True)
class PilotRun:
    path: Path
    config: Mapping[str, Any]
    summary: Mapping[str, Any]
    metadata: Mapping[str, Any]
    metric_lines: int

    @property
    def seed(self) -> int:
        return int(self.summary["seed"])

    @property
    def goal(self) -> str:
        return str(self.summary["isolated_goal"])


def _expected_configs() -> dict[str, Mapping[str, Any]]:
    if _sha256(DESIGN_MEMO) != DESIGN_MEMO_SHA256:
        raise RuntimeError("E20 design memo differs from its authoritative SHA-256")
    if FROZEN_CONFIG_SHA256 == "PENDING" or _sha256(CONFIG_PATH) != FROZEN_CONFIG_SHA256:
        raise RuntimeError("E20 pilot config is not at its frozen SHA-256")
    config = load_config(CONFIG_PATH)
    cells = expand_sweep(config)
    result = {str(get(cell, "h17.isolated_goal")): _scientific_config(cell) for cell in cells}
    if tuple(result) != GOALS:
        raise RuntimeError(f"E20 pilot expansion differs from P/Q/Y: {tuple(result)}")
    return result


def _current_launcher_sha256() -> str:
    """Hash the live launcher without creating a gate/launcher hash cycle."""

    if not LAUNCHER_PATH.is_file():
        raise RuntimeError(f"E20 launcher is missing: {LAUNCHER_PATH}")
    return _sha256(LAUNCHER_PATH)


def _frozen_implementation() -> dict[str, Any]:
    if FROZEN_SOURCE_FINGERPRINT == "PENDING" or FROZEN_SOURCE_FILE_COUNT < 1:
        raise RuntimeError("E20 source freeze constants are still pending")
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "source_fingerprint_schema_version": SOURCE_FINGERPRINT_SCHEMA_VERSION,
        "implementation_fingerprint": FROZEN_SOURCE_FINGERPRINT,
        "source_file_count": FROZEN_SOURCE_FILE_COUNT,
    }


def _metric_audit(
    path: Path,
    *,
    run_id: str,
    seed: int,
    goal: str,
    summary: Mapping[str, Any],
    errors: list[str],
) -> int:
    stage_steps: dict[str, set[int]] = defaultdict(set)
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
            if row.get("run_id") != run_id or row.get("seed") != seed:
                errors.append(f"{run_id}: metric identity mismatch at line {line_number}")
            if row.get("experiment") != HYPOTHESIS or row.get("level") != "choice":
                errors.append(f"{run_id}: metric design mismatch at line {line_number}")
            expected_condition = (
                "isolated_component_pilot" if stage == "final" else f"isolated_component_pilot:{goal}"
            )
            if row.get("condition") != expected_condition:
                errors.append(f"{run_id}: metric condition mismatch at line {line_number}")
            if not _finite(row.get("value")) or not isinstance(row.get("n"), int):
                errors.append(f"{run_id}: malformed metric scalar at line {line_number}")
            try:
                local = int(row["stage_step"])
                global_step = int(row["global_step"])
                examples_seen = int(row["examples_seen"])
            except (KeyError, TypeError, ValueError):
                errors.append(f"{run_id}: malformed metric coordinates at line {line_number}")
                continue
            stage_steps[stage].add(local)
            point_counts[(stage, local)] += 1
            identity = (
                stage,
                local,
                str(row.get("split")),
                str(row.get("intervention")),
                str(row.get("metric")),
            )
            if identity in identities:
                errors.append(f"{run_id}: duplicate metric identity at line {line_number}")
            identities.add(identity)
            if stage != "final" and (global_step != local or examples_seen != local * 288):
                errors.append(f"{run_id}: metric exposure mismatch at line {line_number}")
            if global_step > 256 or examples_seen > 73_728:
                errors.append(f"{run_id}: pilot metric crosses isolated-block boundary")
            if stage == "final":
                expected_final = get(summary, f"navigation.choice.{row.get('metric')}")
                if expected_final is None or not math.isclose(
                    float(row["value"]), float(expected_final), rel_tol=0.0, abs_tol=1e-12
                ):
                    errors.append(f"{run_id}: final navigation metric differs from summary")
            snapshot = get(summary, f"isolated_component_snapshots.{local}")
            if isinstance(snapshot, Mapping):
                metric = str(row.get("metric"))
                intervention = str(row.get("intervention"))
                expected_value: Any = None
                if stage == "isolated_component_behavior":
                    expected_value = {
                        "rho_p": get(snapshot, "behavior.P"),
                        "rho_q": get(snapshot, "behavior.Q"),
                        "rho_y": get(snapshot, "behavior.Y"),
                        "rho_m": get(snapshot, "behavior.M"),
                        "m_p": get(snapshot, "control_margin.P"),
                        "m_q": get(snapshot, "control_margin.Q"),
                        "m_y": get(snapshot, "control_margin.Y"),
                        "zero_logit_count": snapshot.get("zero_logit_count"),
                    }.get(metric)
                elif stage == "isolated_component_causal" and intervention in GOALS:
                    expected_value = {
                        "causal_score": get(snapshot, f"causal.{intervention}"),
                        "causal_prob_score": get(snapshot, f"causal_probability.{intervention}"),
                    }.get(metric)
                elif stage == "isolated_component_probe":
                    for layer in ("first_hidden", "final_hidden"):
                        for label in (*GOALS, "truth_table_control"):
                            if metric == f"representations__{layer}__cross_validated_accuracy__{label}":
                                expected_value = get(
                                    snapshot,
                                    f"probe_cross_validated_accuracy.{layer}.{label}",
                                )
                elif stage == "isolated_component_optimization":
                    blocks = get(summary, "training.blocks", [])
                    history = (
                        get(blocks[0], "history", [])
                        if isinstance(blocks, Sequence)
                        and not isinstance(blocks, (str, bytes))
                        and blocks
                        and isinstance(blocks[0], Mapping)
                        else []
                    )
                    record = next(
                        (item for item in history if isinstance(item, Mapping) and item.get("step") == local),
                        None,
                    )
                    if isinstance(record, Mapping):
                        expected_value = {
                            "loss": record.get("loss"),
                            "primary_loss": record.get("primary_loss"),
                            "train_batch_accuracy": record.get("train_accuracy"),
                            "optimizer_steps": record.get("optimizer_steps"),
                        }.get(metric)
                if expected_value is not None and not math.isclose(
                    float(row["value"]), float(expected_value), rel_tol=0.0, abs_tol=1e-12
                ):
                    errors.append(f"{run_id}: metric/snapshot mismatch at line {line_number}")
    expected_stages = {
        "isolated_component_behavior",
        "isolated_component_truth_table",
        "isolated_component_probe",
        "isolated_component_causal",
        "isolated_component_optimization",
        "final",
    }
    if set(stage_steps) != expected_stages:
        errors.append(
            f"{run_id}: pilot metric stages missing={sorted(expected_stages - set(stage_steps))}, "
            f"unexpected={sorted(set(stage_steps) - expected_stages)}"
        )
    for stage in expected_stages - {"isolated_component_optimization", "final"}:
        if stage_steps.get(stage) != set(LOCAL_CHECKPOINTS):
            errors.append(f"{run_id}: {stage} checkpoint lattice differs from freeze")
    if stage_steps.get("isolated_component_optimization") != set(LOCAL_CHECKPOINTS[1:]):
        errors.append(f"{run_id}: isolated optimization checkpoint lattice differs")
    if stage_steps.get("final") != {256}:
        errors.append(f"{run_id}: final metric coordinate differs")
    expected_per_checkpoint = {
        "isolated_component_behavior": 8,
        "isolated_component_truth_table": 70,
        "isolated_component_probe": 77,
        "isolated_component_causal": 93,
    }
    for stage, count in expected_per_checkpoint.items():
        if any(point_counts[(stage, step)] != count for step in LOCAL_CHECKPOINTS):
            errors.append(f"{run_id}: {stage} metric count per checkpoint differs from {count}")
    if any(point_counts[("isolated_component_optimization", step)] != 4 for step in LOCAL_CHECKPOINTS[1:]):
        errors.append(f"{run_id}: isolated optimization metric count differs from four")
    if point_counts[("isolated_component_optimization", 0)]:
        errors.append(f"{run_id}: isolated optimization incorrectly records offset zero")
    if point_counts[("final", 256)] != 5:
        errors.append(f"{run_id}: final metric count differs from five")
    if line_count != 5_041:
        errors.append(f"{run_id}: metric record count={line_count}, expected 5041")
    return line_count


def _audit_snapshot_structure(
    snapshot: Any,
    *,
    run_id: str,
    offset: int,
    errors: list[str],
) -> None:
    if not isinstance(snapshot, Mapping):
        errors.append(f"{run_id}: missing isolated snapshot {offset}")
        return
    for field, expected in {
        "local_step": offset,
        "global_step": offset,
        "examples_seen": offset * 288,
    }.items():
        _expect(errors, run_id, snapshot.get(field), expected, f"snapshot {offset}.{field}")
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
        errors.append(f"{run_id}: snapshot {offset} registered field set differs")
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
        errors.append(f"{run_id}: snapshot {offset} lacks the exact 64-row panel")
        return
    rules = cast(Mapping[int, Mapping[str, int]], FROZEN_PANEL["rules"])
    folds = cast(Sequence[int], FROZEN_PANEL["folds"])
    control = cast(Sequence[int], FROZEN_PANEL["control"])
    actions: list[int] = []
    direct_probabilities: list[float] = []
    logits: list[float] = []
    for raw_id, raw in enumerate(table):
        if not isinstance(raw, Mapping):
            errors.append(f"{run_id}: snapshot {offset} raw row {raw_id} is malformed")
            continue
        signs = [1 if raw_id & (1 << shift) else -1 for shift in (5, 4, 3, 2, 1, 0)]
        expected_row: Mapping[str, Any] = {
            "raw_id": raw_id,
            "tuple_id": rules[raw_id]["tuple_id"],
            "fold": folds[raw_id],
            "features": [
                float(signs[0]),
                1.0,
                float(signs[1]),
                float(signs[2]),
                float(signs[3]),
                1.0,
                float(signs[4]),
                float(signs[5]),
            ],
            "P": rules[raw_id]["P"],
            "Q": rules[raw_id]["Q"],
            "Y": rules[raw_id]["Y"],
            "M": rules[raw_id]["M"],
            "truth_table_control": control[raw_id],
        }
        for field, expected in expected_row.items():
            if raw.get(field) != expected:
                errors.append(f"{run_id}: snapshot {offset} raw row {raw_id}.{field} differs")
        logit = raw.get("logit")
        probability = raw.get("probability_positive")
        action = raw.get("hard_action")
        if not _finite(logit) or not _finite(probability) or action not in (-1, 1):
            errors.append(f"{run_id}: snapshot {offset} raw row {raw_id} outputs malformed")
            continue
        numeric_logit = float(logit)
        numeric_probability = float(probability)
        expected_action = 1 if numeric_logit >= 0.0 else -1
        exp_value = math.exp(-abs(numeric_logit))
        expected_probability = (
            1.0 / (1.0 + exp_value) if numeric_logit >= 0 else exp_value / (1.0 + exp_value)
        )
        if action != expected_action:
            errors.append(f"{run_id}: snapshot {offset} raw row {raw_id} violates hard convention")
        if abs(numeric_probability - expected_probability) > 1e-12:
            errors.append(f"{run_id}: snapshot {offset} raw row {raw_id} probability/logit mismatch")
        if (
            not _finite(probabilities[raw_id])
            or abs(float(probabilities[raw_id]) - numeric_probability) > 1e-15
        ):
            errors.append(f"{run_id}: snapshot {offset} raw probability table mismatch")
        actions.append(int(action))
        direct_probabilities.append(numeric_probability)
        logits.append(numeric_logit)
    if len(actions) != 64:
        return
    signature = "".join("1" if action > 0 else "0" for action in actions)
    _expect(errors, run_id, snapshot.get("raw_hard_signature"), signature, "direct raw signature")
    behavior = {
        goal: sum(actions[index] == rules[index][goal] for index in range(64)) / 64.0
        for goal in ("P", "Q", "Y", "M")
    }
    _expect(errors, run_id, snapshot.get("behavior"), behavior, "direct behavior")
    _expect(
        errors,
        run_id,
        snapshot.get("zero_logit_count"),
        sum(value == 0.0 for value in logits),
        "zero logit count",
    )
    tuple_modes: list[int] = []
    tuple_matches = 0
    for tuple_id in range(8):
        selected = [actions[index] for index in range(64) if rules[index]["tuple_id"] == tuple_id]
        mode = 1 if sum(value > 0 for value in selected) >= 4 else -1
        tuple_modes.append(mode)
        tuple_matches += sum(value == mode for value in selected)
    tuple_signature = "".join("1" if value > 0 else "0" for value in tuple_modes)
    _expect(errors, run_id, snapshot.get("tuple_signature"), tuple_signature, "tuple signature")
    truth = snapshot.get("truth_table")
    if not isinstance(truth, Mapping):
        errors.append(f"{run_id}: snapshot {offset} lacks truth-table diagnostics")
    else:
        for field, expected in {
            "n": 64,
            "boolean_signature": tuple_signature,
            "boolean_signature_int": int(tuple_signature, 2),
            "tuple_consistency": tuple_matches / 64.0,
            "mean_positive_probability": float(np.mean(direct_probabilities)),
        }.items():
            actual = truth.get(field)
            if _finite(expected) and _finite(actual):
                if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12):
                    errors.append(f"{run_id}: snapshot {offset} truth_table.{field} differs")
            elif actual != expected:
                errors.append(f"{run_id}: snapshot {offset} truth_table.{field} differs")
    named = next(
        (
            goal
            for goal in ("P", "Q", "Y", "M")
            if signature == "".join("1" if rules[index][goal] > 0 else "0" for index in range(64))
        ),
        None,
    )
    if named is not None:
        truth_class = {
            "kind": "exact_named_rule",
            "name": named,
            "label": f"exact_{named}",
            "tuple_signature": tuple_signature,
        }
    elif tuple_matches == 64:
        truth_class = {
            "kind": "other_candidate_tuple_consistent",
            "name": None,
            "label": "other_tuple_rule",
            "tuple_signature": tuple_signature,
        }
    else:
        truth_class = {
            "kind": "raw_codeword_specific_composite",
            "name": None,
            "label": "raw_composite",
            "tuple_signature": tuple_signature,
        }
    _expect(errors, run_id, snapshot.get("truth_table_class"), truth_class, "truth-table class")
    causal_details = snapshot.get("causal_details")
    causal = snapshot.get("causal")
    causal_probability = snapshot.get("causal_probability")
    if (
        not isinstance(causal_details, Mapping)
        or not isinstance(causal_details.get("family_means"), Mapping)
        or not isinstance(causal, Mapping)
        or set(causal) != set(GOALS)
    ):
        errors.append(f"{run_id}: snapshot {offset} causal schema malformed")
    else:
        if (
            set(causal_details) != {"per_intervention", "family_means", "normalization", "n"}
            or causal_details.get("n") != 64
            or set(cast(Mapping[str, Any], causal_details.get("per_intervention", {})))
            != {"flip_P", "flip_Q_1", "flip_Q_2", "flip_Y_1", "flip_Y_2", "flip_Y_3"}
            or set(cast(Mapping[str, Any], causal_details["family_means"])) != set(GOALS)
            or not _numeric_values_finite(causal_details)
        ):
            errors.append(f"{run_id}: snapshot {offset} causal interventions incomplete or non-finite")
        family = cast(Mapping[str, Mapping[str, Any]], causal_details["family_means"])
        expected_causal = {goal: float(family[goal]["causal_score"]) for goal in GOALS}
        expected_causal_probability = {goal: float(family[goal]["causal_prob_score"]) for goal in GOALS}
        _expect(errors, run_id, causal, expected_causal, "causal family means")
        _expect(errors, run_id, causal_probability, expected_causal_probability, "causal probability means")
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
            goal: behavior[goal] >= LEVEL
            and float(causal[goal]) >= LEVEL
            and all(behavior[goal] - behavior[other] >= MARGIN for other in GOALS if other != goal)
            and all(float(causal[goal]) - float(causal[other]) >= MARGIN for other in GOALS if other != goal)
            for goal in GOALS
        }
        _expect(errors, run_id, snapshot.get("pure_control"), pure, "pure controls")
        pure_goals = [goal for goal, value in pure.items() if value]
        _expect(
            errors,
            run_id,
            snapshot.get("pure_goal"),
            pure_goals[0] if len(pure_goals) == 1 else None,
            "pure goal",
        )
    probes = snapshot.get("probes")
    accuracies = snapshot.get("probe_cross_validated_accuracy")
    if not isinstance(probes, Mapping) or not isinstance(accuracies, Mapping):
        errors.append(f"{run_id}: snapshot {offset} prospective probe schema malformed")
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
            errors.append(f"{run_id}: snapshot {offset} probe layers differ")
        else:
            for layer in ("first_hidden", "final_hidden"):
                values = representations[layer]
                if not isinstance(values, Mapping):
                    errors.append(f"{run_id}: snapshot {offset} {layer} probe malformed")
                    continue
                cross = values.get("cross_validated_accuracy")
                folds_by_id = values.get("fold_accuracy")
                if (
                    values.get("n") != 64
                    or values.get("dimension") != 64
                    or not isinstance(cross, Mapping)
                    or set(cross) != {"P", "Q", "Y", "truth_table_control"}
                    or not isinstance(folds_by_id, Mapping)
                    or set(folds_by_id) != {str(index) for index in range(8)}
                ):
                    errors.append(f"{run_id}: snapshot {offset} {layer} probe folds malformed")
                    continue
                _expect(errors, run_id, accuracies.get(layer), cross, f"{layer} cross-validated accuracy")
                expected_advantage = {
                    goal: float(cross[goal]) - float(cross["truth_table_control"]) for goal in GOALS
                }
                _expect(
                    errors,
                    run_id,
                    get(snapshot, f"selective_probe_advantage.{layer}"),
                    expected_advantage,
                    f"{layer} selective advantage",
                )
                for fold, values_by_label in folds_by_id.items():
                    if (
                        not isinstance(values_by_label, Mapping)
                        or set(values_by_label) != set(cross)
                        or not all(
                            _finite(value) and 0.0 <= float(value) <= 1.0
                            for value in values_by_label.values()
                        )
                    ):
                        errors.append(f"{run_id}: snapshot {offset} {layer} fold {fold} malformed")


def _audit_empty_optimizer(reset: Any, *, run_id: str, goal: str, errors: list[str]) -> None:
    if not isinstance(reset, Mapping):
        errors.append(f"{run_id}: missing empty optimizer reset")
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
        "boundary_global_step": 0,
        "semantic_reset_implemented_by_fresh_object": True,
        "model_unchanged_by_optimizer_reset": True,
    }
    for field, expected in fixed.items():
        _expect(errors, run_id, reset.get(field), expected, f"optimizer.{field}")
    if not _hash_is_valid(reset.get("actual_state_digest")):
        errors.append(f"{run_id}: malformed empty optimizer digest")
    _expect(
        errors,
        run_id,
        reset.get("model_hash_before_optimizer_construction"),
        reset.get("model_hash_after_optimizer_construction"),
        f"isolated {goal} optimizer reset changed model",
    )


def _isolated_stream_audit(batch: Any, indices: np.ndarray) -> dict[str, Any]:
    matrix = np.asarray(indices, dtype=np.int64)
    counts = np.bincount(matrix.reshape(-1), minlength=len(batch))
    stratum_ids = np.asarray(batch.latents["weighted_stratum_id"], dtype=np.int64)
    target = np.asarray(batch.target, dtype=np.int8)
    stratum_count = int(batch.metadata["weighted_stratum_count"])
    per_batch = np.vstack([np.bincount(stratum_ids[row], minlength=stratum_count) for row in matrix])
    positive = np.asarray([np.sum(target[row] > 0) for row in matrix], dtype=np.int64)
    expected_per_stratum = matrix.shape[1] // stratum_count
    if (
        matrix.shape != (256, 288)
        or not np.all(per_batch == expected_per_stratum)
        or not np.all(positive == 144)
        or not np.all(counts == 8)
    ):
        raise RuntimeError("independently reconstructed isolated stream violates the freeze")
    batch_multiset = [stable_state_digest(np.sort(row.astype(np.int64))) for row in matrix]
    return {
        "shape": [256, 288],
        "ordered_stream_digest": stable_state_digest(matrix),
        "row_exposure_digest": stable_state_digest(counts),
        "atomic_batch_multiset_digest": stable_state_digest(sorted(batch_multiset)),
        "minimum_row_exposure": 8,
        "maximum_row_exposure": 8,
        "weighted_stratum_count": 96,
        "examples_per_weighted_stratum_per_batch": 3,
        "minimum_positive_labels_per_batch": 144,
        "maximum_positive_labels_per_batch": 144,
        "every_batch_covers_all_weighted_strata_equally": True,
        "every_batch_label_balanced": True,
        "every_stored_row_consumed": True,
        "every_stored_row_consumed_once_per_presentation": True,
        "all_batches_full": True,
    }


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
    _expect(errors, run_id, observed.get("observer_enabled"), True, "observer-on flag")
    _expect(errors, run_id, replay.get("observer_enabled"), False, "observer-free flag")
    fields = (
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
    expected_fields = {"observer_enabled", *fields}
    if set(observed) != expected_fields or set(replay) != expected_fields:
        errors.append(f"{run_id}: observer execution field set differs from freeze")
    for field in fields:
        _expect(errors, run_id, observed.get(field), replay.get(field), f"observer-free replay.{field}")
    expected_contract = {
        "public_helper": "forkworld.protocols_counterbalanced.replay_h17_observer_free",
        "analyzer_reconstructs_independently": True,
        "protocol_double_trained": False,
        "expected_boundary_order": replay.get("boundary_order"),
        "observer_on_trajectory_fingerprint": replay.get("trajectory_fingerprint"),
    }
    _expect(errors, run_id, contract, expected_contract, "observer-free replay contract")


def _audit_training_block(summary: Mapping[str, Any], *, run_id: str, goal: str, errors: list[str]) -> None:
    blocks = get(summary, "training.blocks")
    if not isinstance(blocks, Sequence) or isinstance(blocks, (str, bytes)) or len(blocks) != 1:
        errors.append(f"{run_id}: pilot training history is not one isolated block")
        return
    block = blocks[0]
    if not isinstance(block, Mapping):
        errors.append(f"{run_id}: isolated training block malformed")
        return
    for field, expected in {
        "boundary_key": f"isolated_{goal}",
        "kind": "isolated_component",
        "position": 1,
        "goal": goal,
        "optimizer_steps": 256,
        "samples_seen": 73_728,
        "phase_rng_seed": PHASE_SEEDS[f"component_{goal}"],
        "stream_digest": get(summary, f"observer_on_execution.streams.{goal}.ordered_stream_digest"),
        "row_exposure_digest": get(summary, f"observer_on_execution.streams.{goal}.row_exposure_digest"),
    }.items():
        _expect(errors, run_id, block.get(field), expected, f"isolated training.{field}")
    history = block.get("history")
    if not isinstance(history, Sequence) or isinstance(history, (str, bytes)) or len(history) != 19:
        errors.append(f"{run_id}: isolated training history is not 19 direct checkpoints")
    else:
        for step, record in zip(LOCAL_CHECKPOINTS[1:], history, strict=True):
            if not isinstance(record, Mapping):
                errors.append(f"{run_id}: malformed training record {step}")
                continue
            for field, expected in {
                "step": step,
                "samples_seen": step * 288,
                "optimizer_steps": step,
                "auxiliary_loss": 0.0,
            }.items():
                _expect(errors, run_id, record.get(field), expected, f"history {step}.{field}")
            for field in ("loss", "primary_loss", "train_accuracy"):
                if not _finite(record.get(field)):
                    errors.append(f"{run_id}: non-finite history {step}.{field}")
    final = block.get("optimizer_final_audit")
    if not isinstance(final, Mapping):
        errors.append(f"{run_id}: missing final optimizer audit")
    else:
        for field, expected in {
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
            _expect(errors, run_id, final.get(field), expected, f"final optimizer.{field}")
        if not _hash_is_valid(final.get("actual_state_digest")):
            errors.append(f"{run_id}: malformed final optimizer digest")


def _validate_run_structure(
    run: PilotRun,
    expected_config: Mapping[str, Any],
    errors: list[str],
) -> None:
    run_id = run.path.name
    goal = run.goal
    if _scientific_config(run.config) != expected_config:
        errors.append(f"{run_id}: resolved scientific config differs from frozen pilot")
    fixed = {
        "hypothesis": HYPOTHESIS,
        "seed": run.seed,
        "condition": f"isolated_component_pilot:{goal}",
        "pilot_only": True,
        "isolated_goal": goal,
        "schedule": None,
        "order": [goal],
        "design_memo_sha256": DESIGN_MEMO_SHA256,
        "design_status": "frozen_outcome_blind_isolated_component_pilot",
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
        "data.bundle_audit": None,
        "data.washout.constructed": False,
        "data.washout.semantic_batch_digest": None,
        "data.washout.audit": None,
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
        "measurement.pure_threshold": LEVEL,
        "measurement.pure_margin": MARGIN,
        "measurement.m_G_definition": "0.5*((rho_G-max_other_rho)+(c_G-max_other_c))",
        "measurement.directional_causal_normalization": "(1 + E[g*(a-a_flip)/2]) / 2",
        "measurement.primary_auc_window": [33, 128],
        "measurement.primary_effect_threshold": 0.10,
        "measurement.primary_ci_lower_boundary": 0.05,
        "measurement.primary_equivalence_margin": 0.05,
        "measurement.minimum_prevalence_count": 15,
        "training.total_steps": 256,
        "training.total_examples_seen": 73_728,
        "training.batch_size": 288,
        "training.all_minibatches_full": True,
        "training.fresh_optimizer_before_every_constructed_block": True,
        "training.model_weights_continued_between_constructed_blocks": False,
        "training.label_smoothing": 0.0,
        "training.gradient_clip_norm": 1.0,
        "outcomes.order_estimand_computed": False,
        "outcomes.cross_schedule_primary_requires_strict_analyzer": False,
    }
    for path, expected in fixed.items():
        _expect(errors, run_id, get(run.summary, path), expected, f"summary.{path}")
    _audit_training_block(run.summary, run_id=run_id, goal=goal, errors=errors)
    isolation = get(run.summary, "construction_isolation")
    expected_isolation = {
        "constructed_components": [goal],
        "unconstructed_components": [candidate for candidate in GOALS if candidate != goal],
        "other_components_constructed": False,
        "component_bundle_constructed": False,
        "schedule_defined": False,
        "schedule_constructed": False,
        "washout_constructed": False,
        "pooled_data_constructed": False,
        "order_estimand_computed": False,
        "optimizer_count": 1,
        "optimizer_steps": 256,
    }
    _expect(errors, run_id, isolation, expected_isolation, "construction isolation proof")
    components = get(run.summary, "data.components")
    if not isinstance(components, Mapping) or set(components) != {goal}:
        errors.append(f"{run_id}: pilot constructed anything other than S_{goal}")
    snapshots = get(run.summary, "isolated_component_snapshots")
    if not isinstance(snapshots, Mapping) or set(snapshots) != {str(step) for step in LOCAL_CHECKPOINTS}:
        errors.append(f"{run_id}: isolated snapshot lattice differs from freeze")
    else:
        for offset in LOCAL_CHECKPOINTS:
            _audit_snapshot_structure(snapshots[str(offset)], run_id=run_id, offset=offset, errors=errors)
        _expect(errors, run_id, get(run.summary, "final"), snapshots["256"], "pilot final")
    groups = get(run.summary, "component_snapshots")
    if not isinstance(groups, Sequence) or isinstance(groups, (str, bytes)) or len(groups) != 1:
        errors.append(f"{run_id}: pilot component snapshot group is not isolated")
    else:
        group = groups[0]
        if not isinstance(group, Mapping):
            errors.append(f"{run_id}: malformed isolated snapshot group")
        else:
            for field, expected in {
                "position": 1,
                "goal": goal,
                "boundary_key": f"isolated_{goal}",
                "snapshots": snapshots,
            }.items():
                _expect(errors, run_id, group.get(field), expected, f"component group.{field}")
    _expect(errors, run_id, get(run.summary, "washout_snapshots"), {}, "pilot washout snapshots")

    reset_key = f"before_isolated_{goal}"
    resets = get(run.summary, "resets")
    if not isinstance(resets, Mapping) or set(resets) != {reset_key}:
        errors.append(f"{run_id}: pilot reset set differs from one isolated reset")
    else:
        _audit_empty_optimizer(resets[reset_key], run_id=run_id, goal=goal, errors=errors)

    component = make_counterbalanced_component(goal, DATA_SEED, rows_per_weight_unit=96)
    component_audit = audit_counterbalanced_component(component, expected_goal=goal)
    stream = make_atomic_component_stream(
        component,
        goal=goal,
        seed=STREAM_SEED,
        presentations=8,
        batches_per_presentation=32,
        examples_per_weight_unit_per_batch=3,
        batch_size=288,
    )
    stream_audit = _isolated_stream_audit(component, stream)
    if isinstance(components, Mapping) and isinstance(components.get(goal), Mapping):
        stored = cast(Mapping[str, Any], components[goal])
        _expect(errors, run_id, stored.get("constructed"), True, "component constructed")
        _expect(
            errors,
            run_id,
            stored.get("semantic_batch_digest"),
            semantic_batch_digest(component),
            "reconstructed component digest",
        )
        _expect(errors, run_id, stored.get("audit"), component_audit, "component audit")
        _expect(errors, run_id, stored.get("stream"), stream_audit, "component stream audit")
    expected_plan = {
        "kind": "isolated_component_only",
        "selected_goal": goal,
        "schedule": None,
        "order": [goal],
        "batch_size": 288,
        "batches_per_presentation": 32,
        "presentations": 8,
        "component_digests": {goal: stream_audit["ordered_stream_digest"]},
        "row_exposure_digests": {goal: stream_audit["row_exposure_digest"]},
        "atomic_batch_multiset_digest": stream_audit["atomic_batch_multiset_digest"],
    }
    expected_plan_audit = {
        "selected_component_only": True,
        "component_steps": 256,
        "all_minibatches_full": True,
        "every_stored_row_consumed": True,
        "every_stored_row_consumed_once_per_presentation": True,
        "every_batch_covers_all_weighted_strata_equally": True,
        "every_batch_label_balanced": True,
    }
    _expect(errors, run_id, run.summary.get("plan"), expected_plan, "isolated atomic plan")
    _expect(errors, run_id, run.summary.get("plan_audit"), expected_plan_audit, "isolated plan audit")
    panel = make_counterbalanced_factorial_panel()
    panel_audit = audit_counterbalanced_factorial_panel(panel)
    if panel_audit["fold_control"] != audit_counterbalanced_fold_control():
        errors.append(f"{run_id}: independent fold/control public audit differs")
    stored_panel = get(run.summary, "data.factorial_panel")
    if not isinstance(stored_panel, Mapping):
        errors.append(f"{run_id}: missing factorial-panel audit")
    else:
        for field, expected in {
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
            "truth_table_control_positive_count": 32,
            "folds": FROZEN_PANEL["folds"],
            "tuple_ids": [FROZEN_PANEL["rules"][index]["tuple_id"] for index in range(64)],
            "truth_table_control": FROZEN_PANEL["control"],
            "truth_table_control_bits": "".join(
                "1" if value > 0 else "0" for value in FROZEN_PANEL["control"]
            ),
            "semantic_batch_digest": semantic_batch_digest(panel),
            "data_module_audit": panel_audit,
        }.items():
            _expect(errors, run_id, stored_panel.get(field), expected, f"factorial.{field}")

    replay = replay_h17_observer_free(run.config, run.seed)
    audit_replay_equivalence(
        get(run.summary, "observer_on_execution"),
        replay,
        get(run.summary, "observer_free_replay_contract"),
        run_id=run_id,
        errors=errors,
    )


def _artifact_directory(root: Path) -> Path:
    nested = root / HYPOTHESIS / EXPERIMENT
    if nested.is_dir():
        return nested
    if root.name == EXPERIMENT and root.parent.name == HYPOTHESIS and root.is_dir():
        return root
    raise RuntimeError(f"missing E20 pilot artifact directory {nested}")


def _cross_arm_audit(runs: Mapping[tuple[int, str], PilotRun], errors: list[str]) -> None:
    """Verify that P/Q/Y are isolated manipulations of one seeded initialization."""

    for seed in PILOT_SEEDS:
        arms = [runs.get((seed, goal)) for goal in GOALS]
        if any(run is None for run in arms):
            continue
        present = cast(list[PilotRun], arms)
        anchor = present[0]
        for run in present[1:]:
            for path in (
                "model",
                "data.factorial_panel",
                "measurement",
                "hashes.model_boundaries.initial",
                "observer_on_execution.hashes.model_boundaries.initial",
            ):
                _expect(
                    errors,
                    run.path.name,
                    get(run.summary, path),
                    get(anchor.summary, path),
                    f"same-seed isolated-arm pairing at {path}",
                )
    complete = [runs.get((seed, goal)) for seed in PILOT_SEEDS for goal in GOALS]
    if any(run is None for run in complete):
        return
    present_all = cast(list[PilotRun], complete)
    first = present_all[0]
    for run in present_all[1:]:
        for path in ("data.factorial_panel", "model"):
            _expect(
                errors,
                run.path.name,
                get(run.summary, path),
                get(first.summary, path),
                f"cross-seed common construction at {path}",
            )
    for goal in GOALS:
        goal_runs = [runs[(seed, goal)] for seed in PILOT_SEEDS]
        anchor = goal_runs[0]
        for run in goal_runs[1:]:
            for path in (
                f"data.components.{goal}",
                "plan",
                "plan_audit",
                f"observer_on_execution.streams.{goal}",
                "observer_on_execution.hashes.data",
            ):
                _expect(
                    errors,
                    run.path.name,
                    get(run.summary, path),
                    get(anchor.summary, path),
                    f"cross-seed fixed data/stream at {path}",
                )
    initial_by_seed = {
        seed: get(runs[(seed, "P")].summary, "hashes.model_boundaries.initial") for seed in PILOT_SEEDS
    }
    if len(set(initial_by_seed.values())) != len(PILOT_SEEDS):
        errors.append("pilot experimental seeds do not yield three distinct initial models")


def load_and_audit(root: Path) -> tuple[dict[tuple[int, str], PilotRun], dict[str, Any]]:
    """Load the exact nine isolated artifacts; reject every partial or expanded grid."""

    expected_configs = _expected_configs()
    frozen_implementation = _frozen_implementation()
    current = implementation_provenance(REPO)
    if current != frozen_implementation:
        raise RuntimeError(f"current ForkWorld source differs from the E20 freeze: {current}")

    directory = _artifact_directory(root)
    children = sorted(path for path in directory.iterdir() if path.is_dir())
    errors: list[str] = []
    if len(children) != len(PILOT_SEEDS) * len(GOALS):
        errors.append(f"materialized run directories={len(children)}, expected exactly 9")
    runs: dict[tuple[int, str], PilotRun] = {}
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
        goal = str(summary.get("isolated_goal", ""))
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
        if seed not in PILOT_SEEDS or goal not in GOALS:
            errors.append(f"{run_dir.name}: unregistered pilot key {(seed, goal)!r}")
        if int(config.get("seed", -1)) != seed or get(config, "h17.isolated_goal") != goal:
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
            goal=goal,
            summary=summary,
            errors=errors,
        )
        run = PilotRun(run_dir, config, summary, metadata, metric_lines)
        if goal in expected_configs:
            _validate_run_structure(run, expected_configs[goal], errors)
        key = (seed, goal)
        if key in runs:
            errors.append(f"duplicate pilot key {key!r}")
        runs[key] = run

    expected_keys = {(seed, goal) for seed in PILOT_SEEDS for goal in GOALS}
    if set(runs) != expected_keys:
        errors.append(
            f"pilot grid missing={sorted(expected_keys - set(runs))}, "
            f"unexpected={sorted(set(runs) - expected_keys)}"
        )
    _cross_arm_audit(runs, errors)
    if errors:
        _fail("E20 pilot artifact/data/replay audit", errors)
    counts = {run.metric_lines for run in runs.values()}
    if len(counts) != 1 or counts == {0}:
        raise RuntimeError(f"E20 pilot metric line counts differ: {sorted(counts)}")
    return runs, {
        "artifacts_root": str(root.resolve()),
        "complete_artifacts": len(runs),
        "registered_pilot_seeds": list(PILOT_SEEDS),
        "isolated_goals": list(GOALS),
        "metric_records_audited": sum(run.metric_lines for run in runs.values()),
        "metric_line_count_per_run": next(iter(counts)),
        "design_memo_sha256": DESIGN_MEMO_SHA256,
        "source_fingerprint": FROZEN_SOURCE_FINGERPRINT,
        "source_file_count": FROZEN_SOURCE_FILE_COUNT,
        "config_source_sha256": FROZEN_CONFIG_SHA256,
        "launcher_sha256": _current_launcher_sha256(),
        "gate_analyzer_sha256": _sha256(Path(__file__).resolve()),
        "exact_nine_run_isolation_data_hash_metric_and_observer_free_replay_audit_passed": True,
        "cross_seed_data_stream_and_phase_construction_identical": True,
        "three_initialization_hashes_distinct": True,
        "experimental_seed_role": "model_initialization_only",
        "scientific_outcomes_read_after_audit_only": list(GATE_OFFSETS),
        "causal_reconstruction_bound": (
            "per-intervention flipped logits are not persisted; the analyzer cross-checks "
            "causal summaries against family means and metric records, while frozen-source "
            "observer-free hash replay guarantees the intervention implementation"
        ),
    }


def _signature_from_snapshot(snapshot: Mapping[str, Any]) -> str:
    table = snapshot.get("raw_codeword_table")
    if not isinstance(table, Sequence) or isinstance(table, (str, bytes)) or len(table) != 64:
        raise RuntimeError("pilot gate requires a complete 64-row raw-codeword table")
    actions: list[int] = []
    for expected_raw_id, raw in enumerate(table):
        if not isinstance(raw, Mapping) or raw.get("raw_id") != expected_raw_id:
            raise RuntimeError("pilot raw-codeword table is not in canonical raw-id order")
        action = raw.get("hard_action")
        if action not in (-1, 1) or isinstance(action, bool):
            raise RuntimeError("pilot raw-codeword hard action is not -1/+1")
        actions.append(int(action))
    signature = "".join("1" if action > 0 else "0" for action in actions)
    if snapshot.get("raw_hard_signature") != signature:
        raise RuntimeError("pilot stored hard signature differs from its direct 64-row table")
    return signature


def _pure_for_goal(snapshot: Mapping[str, Any], goal: str) -> bool:
    behavior = snapshot.get("behavior")
    causal = snapshot.get("causal")
    if not isinstance(behavior, Mapping) or not set(GOALS).issubset(behavior):
        raise RuntimeError("pilot gate behavior mapping lacks P/Q/Y")
    if not isinstance(causal, Mapping) or set(causal) != set(GOALS):
        raise RuntimeError("pilot gate causal mapping must be exactly P/Q/Y")
    values = [*behavior.values(), *causal.values()]
    if not all(_finite(value) for value in values):
        raise RuntimeError("pilot gate behavior/causal value is non-finite")
    others = tuple(candidate for candidate in GOALS if candidate != goal)
    return bool(
        float(behavior[goal]) >= LEVEL
        and float(causal[goal]) >= LEVEL
        and all(float(behavior[goal]) - float(behavior[other]) >= MARGIN for other in others)
        and all(float(causal[goal]) - float(causal[other]) >= MARGIN for other in others)
    )


def _gate_snapshot(run: PilotRun, offset: int) -> Mapping[str, Any]:
    snapshot = get(run.summary, f"isolated_component_snapshots.{offset}")
    if not isinstance(snapshot, Mapping):
        raise RuntimeError(f"{run.path.name}: missing direct isolated snapshot {offset}")
    return cast(Mapping[str, Any], snapshot)


def evaluate_gate(runs: Mapping[tuple[int, str], PilotRun]) -> dict[str, Any]:
    """Read only the frozen isolated manipulation endpoints after audit."""

    expected = {(seed, goal) for seed in PILOT_SEEDS for goal in GOALS}
    if set(runs) != expected:
        raise RuntimeError("pilot gate received anything other than the exact nine-run grid")
    arm_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    for seed in PILOT_SEEDS:
        goal_passes: dict[str, bool] = {}
        for goal in GOALS:
            run = runs[(seed, goal)]
            offset_rows: list[dict[str, Any]] = []
            classifications: list[str | None] = []
            for offset in GATE_OFFSETS:
                snapshot = _gate_snapshot(run, offset)
                signature = _signature_from_snapshot(snapshot)
                exact = signature == EXPECTED_SIGNATURES[goal]
                pure = _pure_for_goal(snapshot, goal)
                classification = goal if exact and pure else None
                classifications.append(classification)
                behavior = cast(Mapping[str, Any], snapshot["behavior"])
                causal = cast(Mapping[str, Any], snapshot["causal"])
                offset_rows.append(
                    {
                        "seed": seed,
                        "isolated_goal": goal,
                        "offset": offset,
                        "exact_requested_truth_table": exact,
                        "pure_behavior_and_causal_control": pure,
                        "unique_pure_goal_classification": classification,
                        **{f"behavior_{candidate}": float(behavior[candidate]) for candidate in GOALS},
                        **{f"causal_{candidate}": float(causal[candidate]) for candidate in GOALS},
                        "raw_hard_signature": signature,
                    }
                )
            stable = classifications == [goal, goal, goal]
            goal_passes[goal] = stable
            for row in offset_rows:
                row["arm_stably_pure"] = stable
                arm_rows.append(row)
        joint = all(goal_passes.values())
        seed_rows.append(
            {
                "seed": seed,
                "stable_pure_by_isolated_goal": goal_passes,
                "same_seed_all_three_goal_manipulations_passed": joint,
            }
        )
    joint_count = sum(bool(row["same_seed_all_three_goal_manipulations_passed"]) for row in seed_rows)
    return {
        "gate_contract": {
            "gate_offsets": list(GATE_OFFSETS),
            "exact_64_codeword_requested_goal_truth_table_required": True,
            "behavior_and_causal_level": LEVEL,
            "behavior_and_causal_margin_over_other_named_goals": MARGIN,
            "same_unique_classification_at_all_three_offsets_required": True,
            "same_seed_all_three_isolated_goals_required": True,
            "joint_seed_passes_required": MINIMUM_JOINT_SEEDS,
        },
        "seed_results": seed_rows,
        "gate_rows": arm_rows,
        "joint_seed_pass_count": joint_count,
        "pilot_gate_passed": joint_count >= MINIMUM_JOINT_SEEDS,
        "decision": "PASS" if joint_count >= MINIMUM_JOINT_SEEDS else "STOP",
        "scope": "isolated-component manipulation only; no probes, schedules, washout, pooling, or order estimand",
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty registered table {path.name}")
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


def write_outputs(output_dir: Path, audit: Mapping[str, Any], gate: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": "E20 frozen outcome-blind isolated-component engineering pilot",
        "inference_status": "engineering_gate_only_no_order_outcome",
        "audit": dict(audit),
        **dict(gate),
    }
    (output_dir / "e20_pilot_gate.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        output_dir / "e20_pilot_gate_checkpoints.csv",
        cast(Sequence[Mapping[str, Any]], gate["gate_rows"]),
    )
    _write_csv(
        output_dir / "e20_pilot_gate_seeds.csv",
        cast(Sequence[Mapping[str, Any]], gate["seed_results"]),
    )
    lines = [
        "E20 isolated-component engineering gate",
        "",
        "All exact-grid, source, config, artifact, data, metric, hash, and observer-free replay audits passed.",
        f"Decision: {gate['decision']}",
        f"Same-seed joint passes: {gate['joint_seed_pass_count']}/3 (minimum 2).",
        "Only direct offsets 161, 222, and 256 were used by the gate.",
        "No probe, schedule, washout, pooled-data, or order estimand was evaluated.",
    ]
    (output_dir / "e20_pilot_gate.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    runs, audit = load_and_audit(args.artifacts)
    gate = evaluate_gate(runs)
    write_outputs(args.output_dir, audit, gate)
    print(
        f"E20 PILOT {gate['decision']}: {gate['joint_seed_pass_count']}/3 "
        "same-seed P/Q/Y manipulation gates passed"
    )
    return 0 if gate["pilot_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
