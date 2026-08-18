"""Fail-closed full-launch authorization checks for E20/H17."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import forkworld.e20_launch_guard as guard

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gate_record(
    *,
    design: str = "a" * 64,
    source: str = "b" * 64,
    source_count: int = 34,
    pilot_config: str = "c" * 64,
    launcher: str = "d" * 64,
    gate_analyzer: str = "e" * 64,
) -> dict[str, Any]:
    seed_results = []
    gate_rows = []
    for seed, joint in zip(guard.PILOT_SEEDS, (True, True, False), strict=True):
        stable = {"P": True, "Q": True, "Y": joint}
        seed_results.append(
            {
                "seed": seed,
                "stable_pure_by_isolated_goal": stable,
                "same_seed_all_three_goal_manipulations_passed": joint,
            }
        )
        for goal in guard.GOALS:
            arm_stable = stable[goal]
            for offset in guard.GATE_OFFSETS:
                behavior = {
                    candidate: (
                        0.95 if candidate == goal and arm_stable else 0.70 if candidate == goal else 0.40
                    )
                    for candidate in guard.GOALS
                }
                gate_rows.append(
                    {
                        "seed": seed,
                        "isolated_goal": goal,
                        "offset": offset,
                        "exact_requested_truth_table": True,
                        "pure_behavior_and_causal_control": arm_stable,
                        "unique_pure_goal_classification": goal if arm_stable else None,
                        **{
                            f"behavior_{candidate}": value
                            for candidate, value in behavior.items()
                        },
                        **{
                            f"causal_{candidate}": value
                            for candidate, value in behavior.items()
                        },
                        "raw_hard_signature": guard.EXPECTED_SIGNATURES[goal],
                        "arm_stably_pure": arm_stable,
                    }
                )
    return {
        "experiment": "E20 frozen outcome-blind isolated-component engineering pilot",
        "inference_status": "engineering_gate_only_no_order_outcome",
        "decision": "PASS",
        "pilot_gate_passed": True,
        "joint_seed_pass_count": 2,
        "scope": (
            "isolated-component manipulation only; no probes, schedules, washout, "
            "pooling, or order estimand"
        ),
        "gate_contract": {
            "gate_offsets": [161, 222, 256],
            "exact_64_codeword_requested_goal_truth_table_required": True,
            "behavior_and_causal_level": 0.9,
            "behavior_and_causal_margin_over_other_named_goals": 0.1,
            "same_unique_classification_at_all_three_offsets_required": True,
            "same_seed_all_three_isolated_goals_required": True,
            "joint_seed_passes_required": 2,
        },
        "seed_results": seed_results,
        "gate_rows": gate_rows,
        "audit": {
            "complete_artifacts": 9,
            "registered_pilot_seeds": [563, 569, 571],
            "isolated_goals": ["P", "Q", "Y"],
            "design_memo_sha256": design,
            "source_fingerprint": source,
            "source_file_count": source_count,
            "config_source_sha256": pilot_config,
            "launcher_sha256": launcher,
            "gate_analyzer_sha256": gate_analyzer,
            "exact_nine_run_isolation_data_hash_metric_and_observer_free_replay_audit_passed": True,
            "cross_seed_data_stream_and_phase_construction_identical": True,
            "three_initialization_hashes_distinct": True,
            "experimental_seed_role": "model_initialization_only",
            "scientific_outcomes_read_after_audit_only": [161, 222, 256],
        },
    }


def _validate(record: dict[str, Any]) -> None:
    guard.validate_gate_record(
        record,
        design_memo_sha256="a" * 64,
        source_fingerprint="b" * 64,
        source_file_count=34,
        pilot_config_sha256="c" * 64,
        launcher_sha256="d" * 64,
        expected_gate_analyzer_sha256="e" * 64,
    )


def test_exact_canonical_pass_record_authorizes() -> None:
    _validate(_gate_record())


def test_authorization_requires_explicit_frozen_gate_analyzer_digest(tmp_path: Path) -> None:
    with pytest.raises(guard.LaunchAuthorizationError, match="explicit lowercase SHA-256"):
        guard.authorize_full_launch(tmp_path, "PENDING")


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("decision",), "STOP", "decision"),
        (("pilot_gate_passed",), False, "pilot_gate_passed"),
        (("joint_seed_pass_count",), 1, "joint_seed_pass_count"),
        (("audit", "design_memo_sha256"), "0" * 64, "design_memo_sha256"),
        (("audit", "source_fingerprint"), "0" * 64, "source_fingerprint"),
        (("audit", "source_file_count"), 35, "source_file_count"),
        (("audit", "config_source_sha256"), "0" * 64, "config_source_sha256"),
        (("audit", "launcher_sha256"), "0" * 64, "launcher_sha256"),
        (("audit", "gate_analyzer_sha256"), "0" * 64, "gate_analyzer_sha256"),
    ],
)
def test_gate_validation_fails_closed_on_decision_and_freeze_drift(
    path: tuple[str, ...], value: Any, match: str
) -> None:
    record = _gate_record()
    target: dict[str, Any] = record
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(guard.LaunchAuthorizationError, match=match):
        _validate(record)


def test_gate_validation_recomputes_joint_count_from_exact_seed_rows() -> None:
    record = _gate_record()
    record["seed_results"][2]["same_seed_all_three_goal_manipulations_passed"] = True

    with pytest.raises(guard.LaunchAuthorizationError, match="internally inconsistent"):
        _validate(record)


def test_gate_validation_requires_complete_exact_contract() -> None:
    record = _gate_record()
    del record["gate_contract"]["behavior_and_causal_level"]

    with pytest.raises(guard.LaunchAuthorizationError, match="contract schema"):
        _validate(record)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "malformed"])
def test_gate_validation_requires_exact_reconstructable_checkpoint_rows(
    mutation: str,
) -> None:
    record = _gate_record()
    if mutation == "missing":
        record["gate_rows"].pop()
    elif mutation == "duplicate":
        record["gate_rows"][-1] = copy.deepcopy(record["gate_rows"][0])
    else:
        record["gate_rows"][0]["raw_hard_signature"] = "malformed"

    with pytest.raises(guard.LaunchAuthorizationError, match=r"checkpoint|signature"):
        _validate(record)


def test_gate_validation_reconstructs_row_booleans_and_seed_summaries() -> None:
    record = _gate_record()
    record["gate_rows"][0]["pure_behavior_and_causal_control"] = False
    with pytest.raises(guard.LaunchAuthorizationError, match="does not reconstruct"):
        _validate(record)

    record = _gate_record()
    record["seed_results"][0]["stable_pure_by_isolated_goal"]["P"] = False
    record["seed_results"][0]["same_seed_all_three_goal_manipulations_passed"] = False
    record["joint_seed_pass_count"] = 1
    with pytest.raises(guard.LaunchAuthorizationError, match="joint_seed_pass_count"):
        _validate(record)


def test_authorization_reads_only_the_canonical_gate_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memo = tmp_path / guard.MEMO_RELATIVE_PATH
    pilot_config = tmp_path / guard.PILOT_CONFIG_RELATIVE_PATH
    launcher = tmp_path / guard.LAUNCHER_RELATIVE_PATH
    gate_analyzer = tmp_path / guard.GATE_ANALYZER_RELATIVE_PATH
    gate_path = tmp_path / guard.GATE_RELATIVE_PATH
    for path in (memo, pilot_config, launcher, gate_analyzer, gate_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    memo.write_bytes((ROOT / guard.MEMO_RELATIVE_PATH).read_bytes())
    pilot_config.write_text("frozen pilot\n", encoding="utf-8")
    launcher.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    gate_analyzer.write_text("# frozen gate analyzer\n", encoding="utf-8")
    provenance = {
        "implementation_fingerprint": "e" * 64,
        "source_file_count": 34,
    }
    monkeypatch.setattr(guard, "implementation_provenance", lambda _root: provenance)
    record = _gate_record(
        design=guard.DESIGN_MEMO_SHA256,
        source="e" * 64,
        pilot_config=_sha256(pilot_config),
        launcher=_sha256(launcher),
        gate_analyzer=_sha256(gate_analyzer),
    )
    gate_path.write_text(json.dumps(record), encoding="utf-8")

    frozen_gate_analyzer_sha256 = _sha256(gate_analyzer)
    proof = guard.authorize_full_launch(tmp_path, frozen_gate_analyzer_sha256)

    assert proof["authorized"] is True
    assert proof["canonical_gate"] == str(gate_path)
    assert proof["joint_seed_pass_count"] == 2

    gate_analyzer.write_text("# altered gate analyzer\n", encoding="utf-8")
    with pytest.raises(guard.LaunchAuthorizationError, match="caller-supplied frozen"):
        guard.authorize_full_launch(tmp_path, frozen_gate_analyzer_sha256)

    record["audit"]["gate_analyzer_sha256"] = _sha256(gate_analyzer)
    gate_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(guard.LaunchAuthorizationError, match="caller-supplied frozen"):
        guard.authorize_full_launch(tmp_path, frozen_gate_analyzer_sha256)


def test_authorization_rejects_missing_or_duplicate_key_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memo = tmp_path / guard.MEMO_RELATIVE_PATH
    pilot_config = tmp_path / guard.PILOT_CONFIG_RELATIVE_PATH
    launcher = tmp_path / guard.LAUNCHER_RELATIVE_PATH
    gate_analyzer = tmp_path / guard.GATE_ANALYZER_RELATIVE_PATH
    for path in (memo, pilot_config, launcher, gate_analyzer):
        path.parent.mkdir(parents=True, exist_ok=True)
    memo.write_bytes((ROOT / guard.MEMO_RELATIVE_PATH).read_bytes())
    pilot_config.write_text("frozen pilot\n", encoding="utf-8")
    launcher.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    gate_analyzer.write_text("# frozen gate analyzer\n", encoding="utf-8")
    monkeypatch.setattr(
        guard,
        "implementation_provenance",
        lambda _root: {"implementation_fingerprint": "e" * 64, "source_file_count": 34},
    )

    with pytest.raises(guard.LaunchAuthorizationError, match="missing"):
        guard.authorize_full_launch(tmp_path, _sha256(gate_analyzer))

    gate_path = tmp_path / guard.GATE_RELATIVE_PATH
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_text('{"decision":"PASS","decision":"STOP"}', encoding="utf-8")
    with pytest.raises(guard.LaunchAuthorizationError, match="duplicate key"):
        guard.authorize_full_launch(tmp_path, _sha256(gate_analyzer))
