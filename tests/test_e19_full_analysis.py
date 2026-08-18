from __future__ import annotations

import copy
import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "paper" / "forkworld-current-results" / "e19_full_analysis.py"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("e19_full_analysis_for_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


E19 = _module()


def test_direct_auc_is_recomputed_from_snapshots_without_protocol_helper() -> None:
    snapshots = {
        0: {"behavior": {"Q": 0.0}, "causal": {"Q": 0.0}},
        64: {"behavior": {"Q": 0.5}, "causal": {"Q": 0.5}},
        128: {"behavior": {"Q": 1.0}, "causal": {"Q": 1.0}},
        256: {"behavior": {"Q": 0.0}, "causal": {"Q": 0.0}},
    }

    assert E19.direct_control_auc(snapshots, "Q", horizon=128) == pytest.approx(0.5)
    del snapshots[128]
    with pytest.raises(ValueError, match="directly span"):
        E19.direct_control_auc(snapshots, "Q", horizon=128)


def _probe_schema_snapshot() -> dict[str, Any]:
    probe_layer = {label: 0.5 for label in E19.PROBE_LABELS}
    return {
        "local_step": 0,
        "global_step": 45,
        "examples_seen": 11_250,
        "behavior": {goal: 0.5 for goal in E19.GOALS},
        "causal": {goal: 0.5 for goal in E19.GOALS},
        "causal_probability": {goal: 0.5 for goal in E19.GOALS},
        "selective_final_hidden_probe": {goal: 0.0 for goal in E19.GOALS},
        "target_accuracy": 0.5,
        "probe_heldout_accuracy": {layer: dict(probe_layer) for layer in E19.LAYERS},
        "truth_table": {
            "boolean_signature": "--------",
            "boolean_signature_int": 0,
            "tuple_consistency": 1.0,
            "codeword_consistency": 1.0,
            "nuisance_consistency": 1.0,
            "sign_inversion_symmetry": 1.0,
        },
        "boolean_signature": "--------",
    }


def test_snapshot_probe_schema_requires_all_genuine_and_permuted_controls() -> None:
    snapshot = _probe_schema_snapshot()
    errors: list[str] = []

    E19._audit_snapshot(
        snapshot,
        "synthetic",
        "postedit",
        errors,
        local_step=0,
        global_step=45,
    )
    assert errors == []
    assert E19._metric_is_selected(
        "phase_b_probe",
        "factorial_probe",
        "none",
        "representations__raw__heldout_accuracy__P_permuted",
    )

    missing = copy.deepcopy(snapshot)
    del missing["probe_heldout_accuracy"]["raw"]["P_permuted"]
    errors = []
    E19._audit_snapshot(
        missing,
        "synthetic",
        "postedit",
        errors,
        local_step=0,
        global_step=45,
    )
    assert errors == ["synthetic: malformed postedit probe layer raw"]

    extra = copy.deepcopy(snapshot)
    extra["probe_heldout_accuracy"]["raw"]["unexpected"] = 0.5
    errors = []
    E19._audit_snapshot(
        extra,
        "synthetic",
        "postedit",
        errors,
        local_step=0,
        global_step=45,
    )
    assert errors == ["synthetic: malformed postedit probe layer raw"]


def test_paired_bootstrap_is_deterministic_and_has_frozen_draw_count() -> None:
    values = np.linspace(-0.02, 0.08, 20, dtype=np.float64)

    first = E19._bootstrap_paired(values, key="registered:test")
    second = E19._bootstrap_paired(values, key="registered:test")

    assert first == second
    assert first["bootstrap_draws"] == 4_000
    assert first["n_seeds"] == 20
    assert first["positive_count"] == int(np.sum(values > 0.0))


def _synthetic_endpoints() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    auc = {
        "independent_noop": 0.50,
        "independent_q_restore": 0.43,
        "independent_padding_sham": 0.50,
        "nested_noop": 0.40,
        "nested_q_transplant": 0.47,
        "nested_padding_sham": 0.40,
    }
    q_probe = {
        "independent_noop": 0.50,
        "independent_q_restore": 0.42,
        "independent_padding_sham": 0.50,
        "nested_noop": 0.40,
        "nested_q_transplant": 0.47,
        "nested_padding_sham": 0.40,
    }
    for seed in E19.SEEDS:
        seed_offset = (seed % 7) * 1e-5
        for branch in E19.BRANCHES:
            rows.append(
                {
                    "seed": seed,
                    "branch": branch,
                    "auc128_Q": auc[branch] + seed_offset,
                    "postedit_selective_Q": q_probe[branch] + seed_offset,
                    "postedit_behavior_P": 0.95 + seed_offset,
                    "postedit_causal_P": 0.94 + seed_offset,
                    "postedit_selective_P": 0.20 + seed_offset,
                    "postedit_selective_Y": 0.10 + seed_offset,
                }
            )
    return rows


def test_registered_passing_panel_requires_both_active_shams_and_manipulation() -> None:
    endpoints = _synthetic_endpoints()
    contrasts = E19.paired_contrast_rows(endpoints, E19.SEEDS, E19.SEEDS[:15])
    manipulation_rows, manipulation = E19.manipulation_check_rows(endpoints, E19.SEEDS)
    decision = E19.interpretation_decisions(contrasts, manipulation, eligible_count=15)

    assert len(manipulation_rows) == 20
    assert manipulation["full_manipulation_contract_valid"] is True
    assert decision["co_primary_active_criteria"] == {
        "Delta_N": True,
        "Delta_S": True,
    }
    assert all(decision["registered_sham_auc_equivalence"].values())
    assert decision["bidirectional_registered_criteria_met"] is True
    assert decision["bidirectional_handoff_sensitivity_interpretation_allowed"] is True
    assert "mediation analysis" in decision["scope_guard"]


def test_eligibility_is_sensitivity_only_and_never_filters_primary() -> None:
    endpoints = _synthetic_endpoints()
    # Make the five ineligible seeds adverse. The registered estimate must still
    # include them, while the separately labelled sensitivity estimate does not.
    ineligible = set(E19.SEEDS[15:])
    for row in endpoints:
        if row["seed"] in ineligible and row["branch"] == "independent_q_restore":
            row["auc128_Q"] = 0.60
    contrasts = E19.paired_contrast_rows(endpoints, E19.SEEDS, E19.SEEDS[:15])
    primary = next(
        row for row in contrasts if row["population"] == "fixed_all_20" and row["contrast"] == "Delta_N"
    )
    sensitivity = next(
        row
        for row in contrasts
        if row["population"] == "eligibility_intersection_sensitivity" and row["contrast"] == "Delta_N"
    )

    assert primary["n_seeds"] == 20
    assert sensitivity["n_seeds"] == 15
    assert primary["estimate"] < sensitivity["estimate"]
    assert primary["is_primary_population"] is True
    assert sensitivity["is_primary_population"] is False


def _delta_summary(*, digest: str = "a" * 64, norm_offset: float = 0.0) -> dict[str, Any]:
    return {
        "shape": [64, 2],
        "target_scalar_count": 128,
        "nonzero_scalar_count": 128,
        "digest": digest,
        "l1_norm": 2.0 + norm_offset,
        "l2_norm": 1.0 + norm_offset,
        "linf_norm": 0.2 + norm_offset,
        "finite": True,
    }


def test_sham_float32_amendment_allows_only_registered_norm_tolerance() -> None:
    intended = _delta_summary()
    sham = _delta_summary(digest="b" * 64, norm_offset=5e-7)
    errors: list[str] = []

    E19._audit_sham_delta_against_intended(sham, intended, "synthetic", "restore.sham", errors)
    assert errors == []

    sham["l2_norm"] = intended["l2_norm"] + 2e-6
    E19._audit_sham_delta_against_intended(sham, intended, "synthetic", "restore.sham", errors)
    assert any("l2_norm" in error for error in errors)


def _replay_summary() -> dict[str, Any]:
    digest = "c" * 64
    replay: dict[str, Any] = {}
    for overlap in ("independent", "nested"):
        replay[overlap] = {
            "overlap": overlap,
            "samples_seen": 11_250,
            "optimizer_steps": 45,
            "batch_size": 250,
            "all_minibatches_full": True,
            "checks": {
                "initial_models_equal": True,
                "final_models_equal": True,
                "final_optimizers_equal": True,
                "samples_seen_equal": True,
                "optimizer_steps_equal": True,
            },
            "hashes": {
                "initial_model": digest,
                "replay_initial_model": digest,
                "observed_final_model": digest,
                "replay_final_model": digest,
                "observed_final_optimizer": digest,
                "replay_final_optimizer": digest,
                "phase_a_batch": digest,
                "phase_a_sampler": digest,
            },
            "training_overlap": {
                "p_error_count": 1_000,
                "q_error_count": 1_000,
                "both_error_count": 100,
                "p_only_error_count": 900,
                "q_only_error_count": 900,
                "error_phi": 0.0,
            },
        }
    return {
        "seed": E19.SEEDS[0],
        "branch": E19.BRANCHES[0],
        "hashes": {
            "initial_model": digest,
            "independent_prefix_model": digest,
            "nested_prefix_model": digest,
        },
        "replay": replay,
    }


def test_replay_audit_fails_closed_on_mutated_check() -> None:
    summary = _replay_summary()
    run = E19.FullRun(Path("synthetic"), {}, summary, {}, {}, 0)
    errors: list[str] = []
    E19._audit_replay(run, errors)
    assert errors == []

    mutated = copy.deepcopy(summary)
    mutated["replay"]["nested"]["checks"]["final_models_equal"] = False
    errors = []
    E19._audit_replay(E19.FullRun(Path("synthetic"), {}, mutated, {}, {}, 0), errors)
    assert any("replay checks failed" in error for error in errors)


def test_partial_synthetic_artifact_root_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "h16" / E19.EXPERIMENT).mkdir(parents=True)

    # A later experiment may legitimately extend ``src/forkworld``.  The
    # frozen E19 analyzer must then fail on provenance before it reaches the
    # intentionally partial artifact grid; either fail-closed boundary is
    # correct for this synthetic rejection test.
    with pytest.raises(
        RuntimeError,
        match=r"source differs from the E19 freeze|expected exactly 120",
    ):
        E19.load_and_audit(tmp_path)


def test_registered_constants_remain_frozen() -> None:
    assert len(E19.SEEDS) == 20
    assert len(E19.BRANCHES) == 6
    assert E19.BOOTSTRAP_DRAWS == 4_000
    assert E19.FROZEN_SOURCE_FINGERPRINT == (
        "ab91250cbb8379c1e6afd0f960abe0754b08a745bd2d9fd8922be0f65a1ba7a9"
    )
    assert math.isclose(E19.SHAM_DELTA_NORM_TOLERANCE, 1e-6)
