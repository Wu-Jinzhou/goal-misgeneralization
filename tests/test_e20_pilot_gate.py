from __future__ import annotations

import copy
import hashlib
import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "paper" / "forkworld-current-results" / "e20_pilot_gate.py"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("e20_pilot_gate_for_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


E20 = _module()


def test_pilot_records_live_launcher_hash_without_embedded_launcher_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = tmp_path / "launcher.sh"
    launcher.write_text("#!/bin/sh\n# first\n", encoding="utf-8")
    monkeypatch.setattr(E20, "LAUNCHER_PATH", launcher)

    first = E20._current_launcher_sha256()
    assert first == hashlib.sha256(launcher.read_bytes()).hexdigest()
    launcher.write_text("#!/bin/sh\n# second\n", encoding="utf-8")
    assert E20._current_launcher_sha256() != first
    assert not hasattr(E20, "FROZEN_LAUNCHER_SHA256")


def _auditable_snapshot(goal: str) -> dict[str, Any]:
    rules = E20.FROZEN_PANEL["rules"]
    signature = E20.EXPECTED_SIGNATURES[goal]
    rows: list[dict[str, Any]] = []
    probabilities: list[float] = []
    actions = [1 if bit == "1" else -1 for bit in signature]
    for raw_id, action in enumerate(actions):
        signs = [1 if raw_id & (1 << shift) else -1 for shift in (5, 4, 3, 2, 1, 0)]
        logit = float(2 * action)
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
                **{candidate: rules[raw_id][candidate] for candidate in ("P", "Q", "Y", "M")},
                "truth_table_control": E20.FROZEN_PANEL["control"][raw_id],
                "logit": logit,
                "probability_positive": probability,
                "hard_action": action,
            }
        )
    behavior = {
        candidate: sum(actions[index] == rules[index][candidate] for index in range(64)) / 64.0
        for candidate in ("P", "Q", "Y", "M")
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
    cross = {
        candidate: 0.95 if candidate == goal else 0.5 for candidate in (*E20.GOALS, "truth_table_control")
    }
    representations = {
        layer: {
            "n": 64,
            "dimension": 64,
            "cross_validated_accuracy": dict(cross),
            "fold_accuracy": {str(index): dict(cross) for index in range(8)},
        }
        for layer in ("first_hidden", "final_hidden")
    }
    tuple_signature = {"P": "00001111", "Q": "00110011", "Y": "01010101"}[goal]
    family = {
        candidate: {"causal_score": causal[candidate], "causal_prob_score": causal[candidate]}
        for candidate in E20.GOALS
    }
    return {
        "local_step": 0,
        "global_step": 0,
        "examples_seen": 0,
        "behavior": behavior,
        "causal": causal,
        "causal_probability": dict(causal),
        "control_margin": margins,
        "pure_control": {candidate: candidate == goal for candidate in E20.GOALS},
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
            "mean_positive_probability": sum(probabilities) / 64.0,
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
            "family_means": family,
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


def _snapshot(goal: str, *, pure: bool = True) -> dict[str, Any]:
    signature = E20.EXPECTED_SIGNATURES[goal]
    behavior = {
        candidate: (0.95 if candidate == goal and pure else 0.70 if candidate == goal else 0.40)
        for candidate in E20.GOALS
    }
    causal = dict(behavior)
    return {
        "raw_codeword_table": [
            {
                "raw_id": raw_id,
                "hard_action": 1 if bit == "1" else -1,
            }
            for raw_id, bit in enumerate(signature)
        ],
        "raw_hard_signature": signature,
        "behavior": behavior,
        "causal": causal,
        # The gate must neither read nor emit these secondary probe values.
        "probe_cross_validated_accuracy": {"forbidden_to_gate": 0.99},
    }


def _runs(failing_seeds: set[int] | None = None) -> dict[Any, Any]:
    failing_seeds = failing_seeds or set()
    runs: dict[Any, Any] = {}
    for seed in E20.PILOT_SEEDS:
        for goal in E20.GOALS:
            snapshots = {
                str(offset): _snapshot(
                    goal,
                    pure=not (seed in failing_seeds and goal == "Y" and offset == 222),
                )
                for offset in E20.LOCAL_CHECKPOINTS
            }
            summary = {
                "seed": seed,
                "isolated_goal": goal,
                "isolated_component_snapshots": snapshots,
            }
            runs[(seed, goal)] = E20.PilotRun(Path(f"{seed}-{goal}"), {}, summary, {}, 0)
    return runs


def test_gate_requires_same_two_seeds_to_pass_all_three_isolated_goals() -> None:
    passed = E20.evaluate_gate(_runs())
    assert passed["decision"] == "PASS"
    assert passed["joint_seed_pass_count"] == 3

    two = E20.evaluate_gate(_runs({563}))
    assert two["decision"] == "PASS"
    assert two["joint_seed_pass_count"] == 2

    stopped = E20.evaluate_gate(_runs({563, 569}))
    assert stopped["decision"] == "STOP"
    assert stopped["joint_seed_pass_count"] == 1


def test_gate_requires_all_three_registered_offsets_and_exact_truth_table() -> None:
    runs = _runs()
    run = runs[(563, "P")]
    mutated = copy.deepcopy(run.summary)
    table = mutated["isolated_component_snapshots"]["222"]["raw_codeword_table"]
    table[0]["hard_action"] *= -1
    mutated["isolated_component_snapshots"]["222"]["raw_hard_signature"] = "".join(
        "1" if row["hard_action"] > 0 else "0" for row in table
    )
    runs[(563, "P")] = E20.PilotRun(run.path, {}, mutated, {}, 0)

    result = E20.evaluate_gate(runs)

    seed = next(row for row in result["seed_results"] if row["seed"] == 563)
    assert seed["stable_pure_by_isolated_goal"]["P"] is False


def test_gate_fails_closed_on_malformed_or_missing_raw_rows() -> None:
    runs = _runs()
    run = runs[(563, "Q")]
    mutated = copy.deepcopy(run.summary)
    mutated["isolated_component_snapshots"]["161"]["raw_codeword_table"].pop()
    runs[(563, "Q")] = E20.PilotRun(run.path, {}, mutated, {}, 0)

    with pytest.raises(RuntimeError, match="complete 64-row"):
        E20.evaluate_gate(runs)


def test_gate_output_contains_no_probe_or_order_outcome() -> None:
    result = E20.evaluate_gate(_runs())
    encoded = str(result).lower()

    assert "probe_cross_validated_accuracy" not in encoded
    assert "dispersion" not in encoded
    assert "auc" not in encoded
    assert "schedule" in result["scope"]  # only the explicit exclusion statement


def test_frozen_panel_reconstructs_without_full_analyzer_import() -> None:
    panel = E20.reconstruct_frozen_panel()
    assert panel["fold_digest"] == E20.FOLD_DIGEST
    assert panel["control_digest"] == E20.CONTROL_DIGEST


@pytest.mark.parametrize(
    "mutation",
    ("hard_convention", "probability", "behavior", "causal", "probe_fold"),
)
def test_pilot_structural_audit_fails_closed_before_gate(mutation: str) -> None:
    snapshot = _auditable_snapshot("Y")
    clean_errors: list[str] = []
    E20._audit_snapshot_structure(snapshot, run_id="synthetic", offset=0, errors=clean_errors)
    assert clean_errors == []

    mutated = copy.deepcopy(snapshot)
    if mutation == "hard_convention":
        mutated["raw_codeword_table"][0]["hard_action"] *= -1
    elif mutation == "probability":
        mutated["raw_probability_table"][0] += 0.01
    elif mutation == "behavior":
        mutated["behavior"]["Y"] -= 0.1
    elif mutation == "causal":
        mutated["causal_details"]["family_means"]["Y"]["causal_score"] -= 0.1
    else:
        del mutated["probes"]["representations"]["final_hidden"]["fold_accuracy"]["7"]
    errors: list[str] = []
    E20._audit_snapshot_structure(mutated, run_id="synthetic", offset=0, errors=errors)
    assert errors, mutation


def test_optimizer_reset_audit_rejects_nonempty_state() -> None:
    digest = "a" * 64
    reset = {
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
        "actual_state_digest": digest,
        "model_hash_before_optimizer_construction": digest,
        "model_hash_after_optimizer_construction": digest,
    }
    errors: list[str] = []
    E20._audit_empty_optimizer(reset, run_id="synthetic", goal="P", errors=errors)
    assert errors == []
    reset["state_entry_count"] = 1
    E20._audit_empty_optimizer(reset, run_id="synthetic", goal="P", errors=errors)
    assert errors


def test_pilot_replay_equivalence_rejects_trajectory_mutation() -> None:
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
    observed: dict[str, Any] = {"observer_enabled": True, **{field: field for field in fields}}
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
    replay["trajectory_fingerprint"] = "mutated"
    E20.audit_replay_equivalence(observed, replay, contract, run_id="synthetic", errors=errors)
    assert errors
