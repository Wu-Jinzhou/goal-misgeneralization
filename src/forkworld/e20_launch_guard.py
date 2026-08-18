"""Fail-closed authorization for the E20 full-panel launcher.

The full panel may start only after the canonical pilot gate has emitted PASS
under the exact frozen memo, implementation, pilot config, and launcher. This
module is part of the implementation fingerprint, while the shell launcher is
separately hashed by the gate, so neither side of the preflight can drift.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from .artifacts import implementation_provenance

DESIGN_MEMO_SHA256 = "1386cb401aef8ed187389dc84aa210b09b564436bc16c9246bbd22fce257529a"
PILOT_SEEDS = (563, 569, 571)
GOALS = ("P", "Q", "Y")
GATE_OFFSETS = (161, 222, 256)
MINIMUM_JOINT_SEEDS = 2
LEVEL = 0.90
MARGIN = 0.10


def _goal_signatures() -> dict[str, str]:
    signatures: dict[str, list[str]] = {goal: [] for goal in GOALS}
    for raw_id in range(64):
        p, r1, r2, r3, q1, q2 = (
            1 if raw_id & (1 << shift) else -1 for shift in (5, 4, 3, 2, 1, 0)
        )
        values = {"P": p, "Q": q1 * q2, "Y": r1 * r2 * r3}
        for goal in GOALS:
            signatures[goal].append("1" if values[goal] > 0 else "0")
    return {goal: "".join(bits) for goal, bits in signatures.items()}


EXPECTED_SIGNATURES = _goal_signatures()

GATE_RELATIVE_PATH = Path("paper/forkworld-current-results/derived/e20_pilot_gate.json")
MEMO_RELATIVE_PATH = Path("paper/forkworld-current-results/e20_repaired_order_design.md")
PILOT_CONFIG_RELATIVE_PATH = Path("configs/e20_counterbalanced_order_pilot.yaml")
LAUNCHER_RELATIVE_PATH = Path("runs/21_counterbalanced_order.sh")
GATE_ANALYZER_RELATIVE_PATH = Path("paper/forkworld-current-results/e20_pilot_gate.py")


class LaunchAuthorizationError(RuntimeError):
    """The canonical E20 pilot evidence does not authorize full runs."""


def _require_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise LaunchAuthorizationError(f"{name} must be an explicit lowercase SHA-256 digest")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LaunchAuthorizationError(f"pilot gate JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _load_gate(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise LaunchAuthorizationError(f"canonical pilot gate is missing: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise LaunchAuthorizationError(f"cannot parse canonical pilot gate: {error}") from error
    if not isinstance(value, Mapping):
        raise LaunchAuthorizationError("canonical pilot gate must be a JSON object")
    return value


def _exact(mapping: Mapping[str, Any], key: str, expected: Any, scope: str) -> None:
    value = mapping.get(key)
    if type(value) is not type(expected) or value != expected:
        raise LaunchAuthorizationError(
            f"{scope}.{key} must equal {expected!r}; observed {value!r}"
        )


def _validate_seed_results(
    record: Mapping[str, Any], joint_count: int
) -> dict[int, dict[str, bool]]:
    rows = record.get("seed_results")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise LaunchAuthorizationError("pilot gate seed_results must be a sequence")
    if len(rows) != len(PILOT_SEEDS):
        raise LaunchAuthorizationError("pilot gate must contain exactly three seed results")
    realized_joint = 0
    stable_by_seed: dict[int, dict[str, bool]] = {}
    for expected_seed, row in zip(PILOT_SEEDS, rows, strict=True):
        if not isinstance(row, Mapping):
            raise LaunchAuthorizationError("pilot gate seed result must be an object")
        expected_keys = {
            "seed",
            "stable_pure_by_isolated_goal",
            "same_seed_all_three_goal_manipulations_passed",
        }
        if set(row) != expected_keys:
            raise LaunchAuthorizationError("pilot gate seed result schema is not exact")
        _exact(row, "seed", expected_seed, "seed_results")
        stable = row.get("stable_pure_by_isolated_goal")
        if not isinstance(stable, Mapping) or tuple(stable) != GOALS:
            raise LaunchAuthorizationError(
                "pilot gate stable-goal mapping must be the exact ordered P/Q/Y panel"
            )
        if any(type(stable[goal]) is not bool for goal in GOALS):
            raise LaunchAuthorizationError("pilot gate stable-goal values must be booleans")
        joint = row.get("same_seed_all_three_goal_manipulations_passed")
        expected_joint = all(bool(stable[goal]) for goal in GOALS)
        if type(joint) is not bool or joint is not expected_joint:
            raise LaunchAuthorizationError("pilot gate joint seed result is internally inconsistent")
        realized_joint += int(joint)
        stable_by_seed[expected_seed] = {goal: bool(stable[goal]) for goal in GOALS}
    if realized_joint != joint_count:
        raise LaunchAuthorizationError(
            "pilot gate joint_seed_pass_count differs from its exact seed results"
        )
    return stable_by_seed


def _finite_unit_interval(value: Any) -> bool:
    return bool(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )


def _validate_gate_rows(
    record: Mapping[str, Any], stable_by_seed: Mapping[int, Mapping[str, bool]]
) -> None:
    rows = record.get("gate_rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise LaunchAuthorizationError("pilot gate lacks checkpoint evidence gate_rows")
    expected_coordinates = [
        (seed, goal, offset)
        for seed in PILOT_SEEDS
        for goal in GOALS
        for offset in GATE_OFFSETS
    ]
    if len(rows) != len(expected_coordinates):
        raise LaunchAuthorizationError("pilot gate must contain exactly 27 checkpoint rows")
    expected_keys = {
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
    by_arm: dict[tuple[int, str], list[tuple[Mapping[str, Any], bool]]] = {}
    for expected_coordinate, row in zip(expected_coordinates, rows, strict=True):
        if not isinstance(row, Mapping):
            raise LaunchAuthorizationError("pilot gate checkpoint row must be an object")
        if set(row) != expected_keys:
            raise LaunchAuthorizationError("pilot gate checkpoint row schema is not exact")
        seed, goal, offset = expected_coordinate
        if (
            type(row.get("seed")) is not int
            or row.get("seed") != seed
            or type(row.get("isolated_goal")) is not str
            or row.get("isolated_goal") != goal
            or type(row.get("offset")) is not int
            or row.get("offset") != offset
        ):
            raise LaunchAuthorizationError(
                f"pilot gate checkpoint rows are missing, duplicated, or out of order at {expected_coordinate}"
            )

        behavior = {candidate: row.get(f"behavior_{candidate}") for candidate in GOALS}
        causal = {candidate: row.get(f"causal_{candidate}") for candidate in GOALS}
        if not all(
            _finite_unit_interval(value) for value in (*behavior.values(), *causal.values())
        ):
            raise LaunchAuthorizationError(
                f"pilot gate checkpoint row {expected_coordinate} has malformed behavior/causal values"
            )
        behavior_float = {
            candidate: float(cast(int | float, behavior[candidate])) for candidate in GOALS
        }
        causal_float = {
            candidate: float(cast(int | float, causal[candidate])) for candidate in GOALS
        }
        others = tuple(candidate for candidate in GOALS if candidate != goal)
        pure = bool(
            behavior_float[goal] >= LEVEL
            and causal_float[goal] >= LEVEL
            and all(
                behavior_float[goal] - behavior_float[other] >= MARGIN
                for other in others
            )
            and all(
                causal_float[goal] - causal_float[other] >= MARGIN
                for other in others
            )
        )
        signature = row.get("raw_hard_signature")
        if (
            not isinstance(signature, str)
            or len(signature) != 64
            or set(signature) - {"0", "1"}
        ):
            raise LaunchAuthorizationError(
                f"pilot gate checkpoint row {expected_coordinate} has malformed signature"
            )
        exact = signature == EXPECTED_SIGNATURES[goal]
        classification = goal if exact and pure else None
        if (
            type(row.get("exact_requested_truth_table")) is not bool
            or row.get("exact_requested_truth_table") is not exact
            or type(row.get("pure_behavior_and_causal_control")) is not bool
            or row.get("pure_behavior_and_causal_control") is not pure
            or row.get("unique_pure_goal_classification") != classification
            or type(row.get("arm_stably_pure")) is not bool
        ):
            raise LaunchAuthorizationError(
                f"pilot gate checkpoint row {expected_coordinate} does not reconstruct"
            )
        by_arm.setdefault((seed, goal), []).append((row, classification == goal))

    for seed in PILOT_SEEDS:
        for goal in GOALS:
            arm = by_arm[(seed, goal)]
            stable = len(arm) == len(GATE_OFFSETS) and all(passed for _, passed in arm)
            if any(row["arm_stably_pure"] is not stable for row, _ in arm):
                raise LaunchAuthorizationError(
                    f"pilot gate arm {(seed, goal)} stability does not reconstruct"
                )
            if stable_by_seed[seed][goal] is not stable:
                raise LaunchAuthorizationError(
                    f"pilot gate seed summary for {(seed, goal)} differs from checkpoint rows"
                )


def validate_gate_record(
    record: Mapping[str, Any],
    *,
    design_memo_sha256: str,
    source_fingerprint: str,
    source_file_count: int,
    pilot_config_sha256: str,
    launcher_sha256: str,
    expected_gate_analyzer_sha256: str,
) -> None:
    """Validate a gate record against current frozen inputs, raising on drift."""

    _exact(
        record,
        "experiment",
        "E20 frozen outcome-blind isolated-component engineering pilot",
        "gate",
    )
    _exact(record, "inference_status", "engineering_gate_only_no_order_outcome", "gate")
    _exact(record, "decision", "PASS", "gate")
    _exact(record, "pilot_gate_passed", True, "gate")
    _exact(
        record,
        "scope",
        "isolated-component manipulation only; no probes, schedules, washout, pooling, or order estimand",
        "gate",
    )
    joint_count = record.get("joint_seed_pass_count")
    if type(joint_count) is not int or joint_count < MINIMUM_JOINT_SEEDS:
        raise LaunchAuthorizationError(
            f"gate.joint_seed_pass_count must be an integer >= {MINIMUM_JOINT_SEEDS}"
        )

    contract = record.get("gate_contract")
    if not isinstance(contract, Mapping):
        raise LaunchAuthorizationError("pilot gate lacks gate_contract")
    expected_contract = {
        "gate_offsets": list(GATE_OFFSETS),
        "exact_64_codeword_requested_goal_truth_table_required": True,
        "behavior_and_causal_level": LEVEL,
        "behavior_and_causal_margin_over_other_named_goals": MARGIN,
        "same_unique_classification_at_all_three_offsets_required": True,
        "same_seed_all_three_isolated_goals_required": True,
        "joint_seed_passes_required": MINIMUM_JOINT_SEEDS,
    }
    if set(contract) != set(expected_contract):
        raise LaunchAuthorizationError("pilot gate gate_contract schema is not exact")
    for key, expected in expected_contract.items():
        _exact(contract, key, expected, "gate_contract")

    audit = record.get("audit")
    if not isinstance(audit, Mapping):
        raise LaunchAuthorizationError("pilot gate lacks audit")
    gate_analyzer_sha256 = _require_sha256(
        expected_gate_analyzer_sha256, "expected gate analyzer SHA-256"
    )
    expected_audit = {
        "complete_artifacts": 9,
        "registered_pilot_seeds": list(PILOT_SEEDS),
        "isolated_goals": list(GOALS),
        "design_memo_sha256": design_memo_sha256,
        "source_fingerprint": source_fingerprint,
        "source_file_count": source_file_count,
        "config_source_sha256": pilot_config_sha256,
        "launcher_sha256": launcher_sha256,
        "gate_analyzer_sha256": gate_analyzer_sha256,
        "exact_nine_run_isolation_data_hash_metric_and_observer_free_replay_audit_passed": True,
        "cross_seed_data_stream_and_phase_construction_identical": True,
        "three_initialization_hashes_distinct": True,
        "experimental_seed_role": "model_initialization_only",
        "scientific_outcomes_read_after_audit_only": list(GATE_OFFSETS),
    }
    for key, expected in expected_audit.items():
        _exact(audit, key, expected, "audit")
    stable_by_seed = _validate_seed_results(record, joint_count)
    _validate_gate_rows(record, stable_by_seed)


def authorize_full_launch(
    repo_root: Path, expected_gate_analyzer_sha256: str
) -> dict[str, Any]:
    """Authorize the current repository against its canonical E20 pilot gate."""

    root = repo_root.resolve()
    expected_gate_sha256 = _require_sha256(
        expected_gate_analyzer_sha256, "expected gate analyzer SHA-256"
    )
    gate_path = root / GATE_RELATIVE_PATH
    memo_path = root / MEMO_RELATIVE_PATH
    pilot_config_path = root / PILOT_CONFIG_RELATIVE_PATH
    launcher_path = root / LAUNCHER_RELATIVE_PATH
    gate_analyzer_path = root / GATE_ANALYZER_RELATIVE_PATH
    for name, path in (
        ("design memo", memo_path),
        ("pilot config", pilot_config_path),
        ("launcher", launcher_path),
        ("pilot gate analyzer", gate_analyzer_path),
    ):
        if not path.is_file():
            raise LaunchAuthorizationError(f"current {name} is missing: {path}")

    memo_sha256 = _sha256(memo_path)
    if memo_sha256 != DESIGN_MEMO_SHA256:
        raise LaunchAuthorizationError(
            "current E20 design memo differs from the prospectively frozen SHA-256"
        )
    provenance = implementation_provenance(root)
    pilot_config_sha256 = _sha256(pilot_config_path)
    launcher_sha256 = _sha256(launcher_path)
    gate_analyzer_sha256 = _sha256(gate_analyzer_path)
    if gate_analyzer_sha256 != expected_gate_sha256:
        raise LaunchAuthorizationError(
            "current canonical pilot gate analyzer differs from the caller-supplied frozen SHA-256"
        )
    record = _load_gate(gate_path)
    validate_gate_record(
        record,
        design_memo_sha256=memo_sha256,
        source_fingerprint=str(provenance["implementation_fingerprint"]),
        source_file_count=int(provenance["source_file_count"]),
        pilot_config_sha256=pilot_config_sha256,
        launcher_sha256=launcher_sha256,
        expected_gate_analyzer_sha256=expected_gate_sha256,
    )
    return {
        "authorized": True,
        "canonical_gate": str(gate_path),
        "canonical_gate_sha256": _sha256(gate_path),
        "joint_seed_pass_count": int(record["joint_seed_pass_count"]),
        "design_memo_sha256": memo_sha256,
        "source_fingerprint": provenance["implementation_fingerprint"],
        "source_file_count": provenance["source_file_count"],
        "pilot_config_sha256": pilot_config_sha256,
        "launcher_sha256": launcher_sha256,
        "gate_analyzer_sha256": gate_analyzer_sha256,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--expected-gate-analyzer-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        proof = authorize_full_launch(
            args.repo_root, args.expected_gate_analyzer_sha256
        )
    except (LaunchAuthorizationError, KeyError, TypeError, ValueError) as error:
        print(f"E20 full launch refused: {error}", file=sys.stderr)
        return 2
    print(
        "E20 full launch authorized by canonical pilot gate "
        f"({proof['joint_seed_pass_count']}/3 joint seeds; {proof['canonical_gate_sha256']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DESIGN_MEMO_SHA256",
    "EXPECTED_SIGNATURES",
    "GATE_ANALYZER_RELATIVE_PATH",
    "GATE_RELATIVE_PATH",
    "LaunchAuthorizationError",
    "authorize_full_launch",
    "validate_gate_record",
]
