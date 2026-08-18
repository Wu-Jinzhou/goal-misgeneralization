from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "paper" / "forkworld-current-results" / "e20_full_analysis.py"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("e20_full_analysis_for_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


E20 = _module()

POST_FREEZE_ANALYSIS_SHA256 = (
    "810ab7b75b9d805191c6b425ff4759b2691c109c542fcb0370996b2a872da83a"
)
FROZEN_ANALYSIS_SHA256 = (
    "41580d85e019d08a514a35c5d6415924d5f0ddb6e96e747c40073668f213b234"
)


def _recovered_frozen_implementation() -> dict[str, Any]:
    """Reconstruct source-v1 from the exact, non-scientific post-freeze edit."""

    source_root = ROOT / "src" / "forkworld"
    files = sorted(
        (
            path
            for path in source_root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and (path.suffix == ".py" or path.name == "py.typed")
        ),
        key=lambda path: path.relative_to(source_root).as_posix(),
    )
    digest = hashlib.sha256()
    digest.update(b"forkworld-source-v1\0")
    for path in files:
        relative = path.relative_to(source_root).as_posix()
        payload = path.read_bytes()
        if relative == "analysis.py":
            assert hashlib.sha256(payload).hexdigest() == POST_FREEZE_ANALYSIS_SHA256
            replacements = (
                (b"from collections.abc import Mapping, Sequence\n", b""),
                (b"from typing import Any\n", b"from typing import Any, Mapping\n"),
                (
                    b"from .metrics import bootstrap_mean_ci\n\n# Inferential labels",
                    b"from .metrics import bootstrap_mean_ci\n\n\n# Inferential labels",
                ),
            )
            for current, frozen in replacements:
                assert payload.count(current) == 1
                payload = payload.replace(current, frozen)
            assert hashlib.sha256(payload).hexdigest() == FROZEN_ANALYSIS_SHA256
        encoded_path = relative.encode()
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return {
        "artifact_schema_version": E20.ARTIFACT_SCHEMA_VERSION,
        "source_fingerprint_schema_version": E20.SOURCE_FINGERPRINT_SCHEMA_VERSION,
        "implementation_fingerprint": digest.hexdigest(),
        "source_file_count": len(files),
    }


def test_frozen_trust_anchors_match_recorded_evidence_gate_and_launcher() -> None:
    pilot_gate = json.loads(
        (SCRIPT.parent / "derived" / "e20_pilot_gate.json").read_text(encoding="utf-8")
    )
    full_audit = json.loads(
        (SCRIPT.parent / "derived" / "e20_full_audit.json").read_text(encoding="utf-8")
    )
    frozen = E20._frozen_implementation()

    assert _recovered_frozen_implementation() == frozen
    assert frozen == {
        "artifact_schema_version": E20.ARTIFACT_SCHEMA_VERSION,
        "source_fingerprint_schema_version": E20.SOURCE_FINGERPRINT_SCHEMA_VERSION,
        "implementation_fingerprint": E20.FROZEN_SOURCE_FINGERPRINT,
        "source_file_count": E20.FROZEN_SOURCE_FILE_COUNT,
    }
    for evidence in (pilot_gate["audit"], full_audit):
        assert evidence["source_fingerprint"] == E20.FROZEN_SOURCE_FINGERPRINT
        assert evidence["source_file_count"] == E20.FROZEN_SOURCE_FILE_COUNT

    assert full_audit["design_memo_sha256"] == E20.DESIGN_MEMO_SHA256
    assert full_audit["config_source_sha256"] == E20.FROZEN_CONFIG_SHA256
    assert pilot_gate["audit"]["config_source_sha256"] == E20.FROZEN_PILOT_CONFIG_SHA256
    assert full_audit["launcher_sha256"] == E20.FROZEN_LAUNCHER_SHA256
    assert (
        full_audit["frozen_gate_analyzer_sha256"]
        == E20.FROZEN_GATE_ANALYZER_SHA256
    )
    assert full_audit["strict_analyzer_sha256"] == E20._sha256(SCRIPT)
    assert E20._sha256(E20.GATE_ANALYZER_PATH) == E20.FROZEN_GATE_ANALYZER_SHA256
    assert E20._sha256(E20.LAUNCHER_PATH) == E20.FROZEN_LAUNCHER_SHA256


def test_full_loader_rejects_source_drift_before_opening_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen = E20._frozen_implementation()
    drifted = {**frozen, "implementation_fingerprint": "0" * 64}
    monkeypatch.setattr(E20, "_expected_configs", lambda: {})
    monkeypatch.setattr(E20, "implementation_provenance", lambda _repo: drifted)

    with pytest.raises(RuntimeError, match="current ForkWorld source differs"):
        E20.load_and_audit(tmp_path)


@pytest.mark.parametrize(
    ("stage", "expected"),
    [
        ("component_1_p_truth_table", ("component_1_p", "truth_table", 0)),
        ("component_2_q_behavior", ("component_2_q", "behavior", 256)),
        ("component_3_y_optimization", ("component_3_y", "optimization", 512)),
        ("washout_probe", ("washout", "probe", 768)),
        ("washout_causal", ("washout", "causal", 768)),
    ],
)
def test_metric_stage_parser_uses_exact_longest_suffix_and_frozen_origin(
    stage: str, expected: tuple[str, str, int]
) -> None:
    assert E20._parse_metric_stage(stage) == expected


@pytest.mark.parametrize(
    "stage",
    [
        "component_1_p_truth",
        "component_1_p_table",
        "component_1_p_truth_table_extra",
        "component_4_p_behavior",
        "component_1_m_behavior",
        "component_x_p_probe",
        "final",
    ],
)
def test_metric_stage_parser_rejects_non_frozen_stage_mutations(stage: str) -> None:
    with pytest.raises(ValueError, match="metric stage"):
        E20._parse_metric_stage(stage)


def _auditable_snapshot(goal: str, *, local: int = 0, global_step: int = 0) -> dict[str, Any]:
    rules = E20.FROZEN_PANEL["rules"]
    signature = E20.EXPECTED_SIGNATURES[goal]
    rows = []
    probabilities = []
    for raw_id, bit in enumerate(signature):
        signs = [1 if raw_id & (1 << shift) else -1 for shift in (5, 4, 3, 2, 1, 0)]
        action = 1 if bit == "1" else -1
        logit = 2.0 * action
        probability = 1.0 / (1.0 + math.exp(-logit))
        probabilities.append(probability)
        rows.append(
            {
                "raw_id": raw_id,
                "tuple_id": rules[raw_id]["tuple_id"],
                "fold": E20.FROZEN_PANEL["folds"][raw_id],
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
                **{candidate: rules[raw_id][candidate] for candidate in E20.NAMED_RULES},
                "truth_table_control": E20.FROZEN_PANEL["control"][raw_id],
                "logit": logit,
                "probability_positive": probability,
                "hard_action": action,
            }
        )
    behavior = {
        candidate: sum(
            (1 if bit == "1" else -1) == rules[raw_id][candidate] for raw_id, bit in enumerate(signature)
        )
        / 64.0
        for candidate in E20.NAMED_RULES
    }
    causal = {candidate: 1.0 if candidate == goal else 0.5 for candidate in E20.GOALS}
    margins = {
        candidate: 0.5
        * (
            behavior[candidate]
            - max(behavior[other] for other in E20.GOALS if other != candidate)
            + causal[candidate]
            - max(causal[other] for other in E20.GOALS if other != candidate)
        )
        for candidate in E20.GOALS
    }
    pure = {candidate: candidate == goal for candidate in E20.GOALS}
    cross = {
        candidate: 0.95 if candidate == goal else 0.5 for candidate in (*E20.GOALS, "truth_table_control")
    }
    fold_accuracy = {str(index): dict(cross) for index in range(8)}
    representations = {
        layer: {
            "n": 64,
            "dimension": 64,
            "cross_validated_accuracy": dict(cross),
            "fold_accuracy": copy.deepcopy(fold_accuracy),
        }
        for layer in ("first_hidden", "final_hidden")
    }
    tuple_signature = {
        "P": "00001111",
        "Q": "00110011",
        "Y": "01010101",
    }[goal]
    family_means = {
        candidate: {"causal_score": causal[candidate], "causal_prob_score": causal[candidate]}
        for candidate in E20.GOALS
    }
    return {
        "local_step": local,
        "global_step": global_step,
        "examples_seen": global_step * E20.BATCH_SIZE,
        "behavior": behavior,
        "causal": causal,
        "causal_probability": dict(causal),
        "control_margin": margins,
        "pure_control": pure,
        "pure_goal": goal,
        "zero_logit_count": 0,
        "raw_hard_signature": signature,
        "raw_probability_table": probabilities,
        "raw_codeword_table": rows,
        "truth_table": {
            "n": 64,
            "boolean_signature": tuple_signature,
            "boolean_signature_int": int(tuple_signature, 2),
            "tuple_consistency": 1.0,
            "mean_positive_probability": float(np.mean(probabilities)),
        },
        "tuple_signature": tuple_signature,
        "truth_table_class": {
            "kind": "exact_named_rule",
            "name": goal,
            "label": f"exact_{goal}",
            "tuple_signature": tuple_signature,
        },
        "causal_details": {
            "per_intervention": {
                name: {"n": 64, "causal_score": 0.5}
                for name in ("flip_P", "flip_Q_1", "flip_Q_2", "flip_Y_1", "flip_Y_2", "flip_Y_3")
            },
            "family_means": family_means,
            "normalization": {"hard_abs_total": 1.0},
            "n": 64,
        },
        "probes": {
            "alpha": 0.001,
            "fold_digest": E20.FOLD_DIGEST,
            "truth_table_control_digest": E20.CONTROL_DIGEST,
            "label_names": ["P", "Q", "Y", "truth_table_control"],
            "probe_kind": "deterministic_affine_ridge_eight_fold",
            "standardization": "seven_training_folds_only",
            "representations": representations,
        },
        "probe_cross_validated_accuracy": {layer: dict(cross) for layer in representations},
        "selective_probe_advantage": {
            layer: {candidate: cross[candidate] - cross["truth_table_control"] for candidate in E20.GOALS}
            for layer in representations
        },
    }


def test_fold_control_reconstruction_matches_every_frozen_constraint() -> None:
    panel = E20.reconstruct_fold_and_control()

    assert panel["fold_digest"] == E20.FOLD_DIGEST
    assert panel["control_digest"] == E20.CONTROL_DIGEST
    assert panel["control_bits"] == E20.CONTROL_BITS
    assert len(panel["records"]) == 64
    assert panel["all_balance_constraints_verified"] is True


def test_truth_table_taxonomy_is_hierarchical_and_fail_closed() -> None:
    for goal in E20.NAMED_RULES:
        assert E20.classify_truth_table(E20.EXPECTED_SIGNATURES[goal]) == goal

    # One constant action per candidate tuple, but not one of P/Q/Y/M.
    rules = E20.FROZEN_PANEL["rules"]
    tuple_signature = "".join(
        "1" if rules[raw_id]["tuple_id"] in {0, 1, 2, 4} else "0" for raw_id in range(64)
    )
    assert E20.classify_truth_table(tuple_signature) == "other_candidate_tuple_consistent"

    composite = list(E20.EXPECTED_SIGNATURES["P"])
    composite[0] = "1" if composite[0] == "0" else "0"
    assert E20.classify_truth_table("".join(composite)) == "raw_codeword_specific_composite"
    with pytest.raises(ValueError, match="64-character"):
        E20.classify_truth_table("01")


def test_hamming_dispersion_scale_and_direct_auc_lattice() -> None:
    signatures = {
        schedule: E20.EXPECTED_SIGNATURES[goals[-1]] for schedule, goals in E20.SCHEDULE_GOALS.items()
    }
    assert E20.schedule_dispersion(signatures) == pytest.approx(0.40)
    values = {step: 0.40 for step in E20.PRIMARY_STEPS}
    assert E20.direct_auc(values, steps=E20.PRIMARY_STEPS, start=33, stop=128) == pytest.approx(0.40)
    del values[62]
    with pytest.raises(ValueError, match="exactly direct checkpoints"):
        E20.direct_auc(values, steps=E20.PRIMARY_STEPS, start=33, stop=128)


def test_bootstrap_is_deterministic_and_resamples_exactly_20_seeds() -> None:
    values = np.linspace(0.0, 0.20, 20)
    first = E20.bootstrap_summary(values, key=E20.BOOTSTRAP_KEYS["primary_dispersion"])
    second = E20.bootstrap_summary(values, key=E20.BOOTSTRAP_KEYS["primary_dispersion"])

    assert first == second
    assert first["bootstrap_draws"] == 4_000
    assert first["n_seeds"] == 20
    with pytest.raises(ValueError, match="exactly 20"):
        E20.bootstrap_summary(values[:19], key="bad")


def _decision_summary(
    *,
    estimate: float,
    low: float,
    high: float,
    above: int = 15,
    positive: int = 15,
    negative: int = 0,
    confidence: float = 0.95,
) -> dict[str, Any]:
    return {
        "estimate": estimate,
        "ci_low": low,
        "ci_high": high,
        "at_least_0_05_count": above,
        "positive_count": positive,
        "negative_count": negative,
        "confidence": confidence,
    }


@pytest.mark.parametrize(
    ("summary", "expected"),
    [
        (
            _decision_summary(estimate=0.10, low=0.050001, high=0.15),
            "material_persistent_order_dependence",
        ),
        (
            _decision_summary(estimate=0.10, low=0.05, high=0.15),
            "inconclusive_on_registered_persistence_scale",
        ),
        (
            _decision_summary(estimate=0.10, low=0.06, high=0.15, above=14),
            "inconclusive_on_registered_persistence_scale",
        ),
        (
            _decision_summary(estimate=0.03, low=0.01, high=0.05),
            "practical_behavioral_equivalence",
        ),
        (
            _decision_summary(estimate=0.03, low=0.01, high=0.050001),
            "inconclusive_on_registered_persistence_scale",
        ),
    ],
)
def test_primary_decision_boundaries(summary: dict[str, Any], expected: str) -> None:
    assert E20.primary_decision(summary) == expected


@pytest.mark.parametrize(
    ("summary", "expected"),
    [
        (
            _decision_summary(estimate=0.10, low=1e-9, high=0.20, positive=15),
            "material_recency",
        ),
        (
            _decision_summary(estimate=-0.10, low=-0.20, high=-1e-9, positive=0, negative=15),
            "material_primacy",
        ),
        (
            _decision_summary(estimate=0.0, low=-0.05, high=0.05),
            "practical_directional_equivalence",
        ),
        (
            _decision_summary(estimate=0.10, low=0.0, high=0.20, positive=15),
            "mixed_or_inconclusive_direction",
        ),
    ],
)
def test_direction_decision_boundaries(summary: dict[str, Any], expected: str) -> None:
    assert E20.direction_decision(summary) == expected


def _snapshot(goal: str) -> dict[str, Any]:
    signature = E20.EXPECTED_SIGNATURES[goal]
    probabilities = [0.9 if bit == "1" else 0.1 for bit in signature]
    return {
        "raw_hard_signature": signature,
        "raw_probability_table": probabilities,
        "behavior": {candidate: 1.0 if candidate == goal else 0.5 for candidate in E20.GOALS},
        "causal": {candidate: 1.0 if candidate == goal else 0.5 for candidate in E20.GOALS},
        "probe_cross_validated_accuracy": {
            layer: {
                **{candidate: 0.95 if candidate == goal else 0.5 for candidate in E20.GOALS},
                "truth_table_control": 0.5,
            }
            for layer in ("first_hidden", "final_hidden")
        },
    }


def test_seed_outcomes_reproduce_last_goal_scale_and_algebra() -> None:
    snapshots = {
        schedule: {step: _snapshot(goals[-1]) for step in E20.LOCAL_CHECKPOINTS}
        for schedule, goals in E20.SCHEDULE_GOALS.items()
    }

    result = E20.seed_washout_outcomes(snapshots)

    assert result["primary_hamming_dispersion_auc_33_128"] == pytest.approx(0.40)
    assert result["early_hamming_dispersion_auc_0_33"] == pytest.approx(0.40)
    assert result["terminal_hamming_dispersion_256"] == pytest.approx(0.40)
    assert result["signed_recency_auc_33_128"] == pytest.approx(
        sum(result["per_goal_recency_auc_33_128"].values()) / 3.0
    )
    assert len(result["schedule_arm_rows"]) == 18
    assert {(row["schedule"], row["goal"], row["position"]) for row in result["schedule_arm_rows"]} == {
        (schedule, goal, goals.index(goal) + 1)
        for schedule, goals in E20.SCHEDULE_GOALS.items()
        for goal in E20.GOALS
    }


def test_phase_transition_events_use_two_consecutive_direct_checkpoints() -> None:
    snapshots = {step: _snapshot("Q") for step in E20.LOCAL_CHECKPOINTS}
    for step in (0,):
        snapshots[step] = _snapshot("P")

    rows = E20.block_phase_transition_rows(
        seed=577,
        schedule="p_q_y",
        block_position=2,
        requested_goal="Q",
        snapshots=snapshots,
    )
    q_final = next(row for row in rows if row["goal"] == "Q" and row["probe_layer"] == "final_hidden")
    assert q_final["probe_first_offset"] == 1
    assert q_final["probe_confirmation_offset"] == 2
    assert q_final["behavior_first_offset"] == 1
    assert q_final["pure_first_offset"] == 1

    missing = copy.deepcopy(snapshots)
    del missing[128]
    with pytest.raises(ValueError, match="frozen lattice"):
        E20.block_phase_transition_rows(
            seed=577,
            schedule="p_q_y",
            block_position=2,
            requested_goal="Q",
            snapshots=missing,
        )


@pytest.mark.parametrize(
    "mutation",
    ("hard_convention", "probability", "behavior", "causal", "probe_fold"),
)
def test_snapshot_audit_fails_closed_on_scientific_mutations(mutation: str) -> None:
    snapshot = _auditable_snapshot("Q")
    clean_errors: list[str] = []
    E20.audit_snapshot(snapshot, local_step=0, global_step=0, run_id="synthetic", errors=clean_errors)
    assert clean_errors == []

    mutated = copy.deepcopy(snapshot)
    if mutation == "hard_convention":
        mutated["raw_codeword_table"][0]["hard_action"] *= -1
    elif mutation == "probability":
        mutated["raw_probability_table"][0] += 0.01
    elif mutation == "behavior":
        mutated["behavior"]["Q"] -= 0.1
    elif mutation == "causal":
        mutated["causal_details"]["family_means"]["Q"]["causal_score"] -= 0.1
    else:
        del mutated["probes"]["representations"]["first_hidden"]["fold_accuracy"]["7"]
    errors: list[str] = []
    E20.audit_snapshot(mutated, local_step=0, global_step=0, run_id="synthetic", errors=errors)
    assert errors, mutation


def _gate_record(analyzer_sha: str) -> dict[str, Any]:
    seed_results = []
    gate_rows = []
    for seed in E20.PILOT_SEEDS:
        seed_results.append(
            {
                "seed": seed,
                "stable_pure_by_isolated_goal": {goal: True for goal in E20.GOALS},
                "same_seed_all_three_goal_manipulations_passed": True,
            }
        )
        for goal in E20.GOALS:
            for offset in (161, 222, 256):
                behavior = {candidate: 0.95 if candidate == goal else 0.40 for candidate in E20.GOALS}
                gate_rows.append(
                    {
                        "seed": seed,
                        "isolated_goal": goal,
                        "offset": offset,
                        "exact_requested_truth_table": True,
                        "pure_behavior_and_causal_control": True,
                        "unique_pure_goal_classification": goal,
                        **{f"behavior_{candidate}": value for candidate, value in behavior.items()},
                        **{f"causal_{candidate}": value for candidate, value in behavior.items()},
                        "raw_hard_signature": E20.EXPECTED_SIGNATURES[goal],
                        "arm_stably_pure": True,
                    }
                )
    return {
        "experiment": "E20 frozen outcome-blind isolated-component engineering pilot",
        "inference_status": "engineering_gate_only_no_order_outcome",
        "decision": "PASS",
        "pilot_gate_passed": True,
        "joint_seed_pass_count": 3,
        "scope": (
            "isolated-component manipulation only; no probes, schedules, washout, pooling, or order estimand"
        ),
        "audit": {
            "complete_artifacts": 9,
            "registered_pilot_seeds": list(E20.PILOT_SEEDS),
            "isolated_goals": list(E20.GOALS),
            "design_memo_sha256": E20.DESIGN_MEMO_SHA256,
            "source_fingerprint": E20.FROZEN_SOURCE_FINGERPRINT,
            "source_file_count": E20.FROZEN_SOURCE_FILE_COUNT,
            "config_source_sha256": E20.FROZEN_PILOT_CONFIG_SHA256,
            "launcher_sha256": E20.FROZEN_LAUNCHER_SHA256,
            "gate_analyzer_sha256": analyzer_sha,
            "exact_nine_run_isolation_data_hash_metric_and_observer_free_replay_audit_passed": True,
            "cross_seed_data_stream_and_phase_construction_identical": True,
            "three_initialization_hashes_distinct": True,
            "experimental_seed_role": "model_initialization_only",
            "scientific_outcomes_read_after_audit_only": [161, 222, 256],
        },
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
    }


def test_full_analyzer_requires_an_exact_pass_authorization_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    analyzer = tmp_path / "e20_pilot_gate.py"
    analyzer.write_text("# synthetic frozen gate analyzer\n", encoding="utf-8")
    analyzer_sha = hashlib.sha256(analyzer.read_bytes()).hexdigest()
    monkeypatch.setattr(E20, "GATE_ANALYZER_PATH", analyzer)
    monkeypatch.setattr(E20, "FROZEN_GATE_ANALYZER_SHA256", analyzer_sha)
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(_gate_record(analyzer_sha)), encoding="utf-8")
    result = E20.audit_pilot_gate_record(path)
    assert result["authorization_validated_before_full_artifact_read"] is True
    assert result["gate_analyzer_sha256"] == analyzer_sha

    mutated = _gate_record(analyzer_sha)
    mutated["gate_rows"][0]["behavior_P"] = 0.4
    path.write_text(json.dumps(mutated), encoding="utf-8")
    with pytest.raises(RuntimeError, match="authorization record audit"):
        E20.audit_pilot_gate_record(path)

    path.write_text(json.dumps(_gate_record(analyzer_sha)), encoding="utf-8")
    analyzer.write_text("# drifted gate analyzer\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="current pilot gate analyzer hash"):
        E20.audit_pilot_gate_record(path)


@pytest.mark.parametrize(
    "mutation",
    [
        "contract",
        "missing_row",
        "reordered_rows",
        "extra_row_field",
        "malformed_signature",
        "derived_flag",
        "seed_order",
        "gate_record_hash",
        "launcher_record_hash",
    ],
)
def test_full_gate_pre_read_fails_closed_on_contract_and_evidence_mutations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    analyzer = tmp_path / "e20_pilot_gate.py"
    analyzer.write_text("# synthetic frozen gate analyzer\n", encoding="utf-8")
    analyzer_sha = hashlib.sha256(analyzer.read_bytes()).hexdigest()
    monkeypatch.setattr(E20, "GATE_ANALYZER_PATH", analyzer)
    monkeypatch.setattr(E20, "FROZEN_GATE_ANALYZER_SHA256", analyzer_sha)
    record = _gate_record(analyzer_sha)
    if mutation == "contract":
        del record["gate_contract"]["behavior_and_causal_level"]
    elif mutation == "missing_row":
        record["gate_rows"].pop()
    elif mutation == "reordered_rows":
        record["gate_rows"][0], record["gate_rows"][1] = (
            record["gate_rows"][1],
            record["gate_rows"][0],
        )
    elif mutation == "extra_row_field":
        record["gate_rows"][0]["unregistered"] = True
    elif mutation == "malformed_signature":
        record["gate_rows"][0]["raw_hard_signature"] = "malformed"
    elif mutation == "derived_flag":
        record["gate_rows"][0]["pure_behavior_and_causal_control"] = False
    elif mutation == "seed_order":
        record["seed_results"][0], record["seed_results"][1] = (
            record["seed_results"][1],
            record["seed_results"][0],
        )
    elif mutation == "gate_record_hash":
        record["audit"]["gate_analyzer_sha256"] = "0" * 64
    else:
        record["audit"]["launcher_sha256"] = "0" * 64
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(RuntimeError, match="authorization record audit"):
        E20.audit_pilot_gate_record(path)


def test_replay_equivalence_rejects_one_hash_mutation() -> None:
    observed: dict[str, Any] = {"observer_enabled": True}
    for field in E20.REPLAY_FIELDS:
        observed[field] = {"digest": "a" * 64} if field == "hashes" else field
    replay = copy.deepcopy(observed)
    replay["observer_enabled"] = False
    contract = {
        "public_helper": "forkworld.protocols_counterbalanced.replay_h17_observer_free",
        "analyzer_reconstructs_independently": True,
        "protocol_double_trained": False,
        "expected_boundary_order": replay["boundary_order"],
        "observer_on_trajectory_fingerprint": replay["trajectory_fingerprint"],
    }
    errors: list[str] = []
    E20.audit_replay_equivalence(observed, replay, contract, run_id="synthetic", errors=errors)
    assert errors == []
    replay["hashes"]["digest"] = "b" * 64
    E20.audit_replay_equivalence(observed, replay, contract, run_id="synthetic", errors=errors)
    assert errors


def test_empirical_metrics_reconstruct_integer_weighted_datasets() -> None:
    bundle = E20.make_counterbalanced_bundle(123, rows_per_weight_unit=96)
    audit = E20.audit_counterbalanced_bundle(bundle)
    rows = E20.empirical_common_evidence_rows(
        _auditable_snapshot("P"),
        bundle_audit=audit,
        seed=577,
        schedule="p_q_y",
        phase="component_1_P",
        local_offset=0,
    )
    assert {row["dataset"] for row in rows} == {
        "component_P",
        "component_Q",
        "component_Y",
        "three_component_pool",
        "concordant_washout",
    }
    by_name = {row["dataset"]: row for row in rows}
    assert by_name["component_P"]["exact_empirical_accuracy"] == 1.0
    assert by_name["component_P"]["n_weighted_rows"] == 9_216
    assert by_name["three_component_pool"]["n_weighted_rows"] == 27_648


def test_scientific_analysis_rejects_any_partial_full_grid() -> None:
    with pytest.raises(RuntimeError, match="complete audited 120-run panel"):
        E20.analyze_runs({})
