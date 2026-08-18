"""Protocol separation and pairing checks for the active-Q-pathway intervention."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from forkworld.handoff import static_sampler_digest
from forkworld.protocols_mediation import run_h16


def _tiny_config(*, pilot_only: bool, branch: str) -> dict[str, Any]:
    return {
        "experiment": {
            "hypothesis": "h16",
            "name": "q_pathway_mediation",
        },
        "run": {"device": "cpu", "save_checkpoints": False},
        "evaluation": {"save_predictions": False},
        "data": {
            "n_train": 256,
            "n_validation": 128,
            "n_eval": 256,
            "max_k": 5,
            "state_dim": 8,
        },
        # q_pathway deliberately fails closed on the exact E17 architecture.
        "model": {
            "width": 64,
            "depth": 2,
            "activation": "relu",
            "residual": False,
            "bias": True,
        },
        "update": {"mode": "full", "budget": "full"},
        "train": {
            "steps": 16,
            "batch_size": 64,
            "learning_rate": 0.003,
            "weight_decay": 0.0,
            "optimizer": "adamw",
            "shuffle": True,
            "deterministic": True,
        },
        "h16": {
            "branch": branch,
            "pilot_only": pilot_only,
            "q_p": 0.75,
            "q_q": 0.75,
            "k_q": 2,
            # The smoke-sized source uses degree two so every raw codeword is
            # represented at n=256; max_k_y=5 preserves the 19-column contract.
            "k_y": 2,
            "max_k_q": 3,
            "max_k_y": 5,
            "phase_a_steps": 4,
            "phase_b_steps": 16,
            "eligibility_steps": [3, 4],
            "phase_b_checkpoints": [0, 1, 2, 3, 4, 5, 7, 8, 9, 13, 16],
            "probe_train_n": 128,
            "probe_eval_n": 256,
            "probe_ridge": 0.001,
            "truth_table_control_seed": 1_500_450_271,
            "auc_horizon": 8,
            "pure_threshold": 0.90,
            "pure_margin": 0.10,
        },
    }


def _fast_snapshot(
    _model: Any,
    *,
    stage: str,
    local_step: int,
    global_step: int,
    examples_seen: int,
    **_kwargs: Any,
) -> dict[str, Any]:
    if stage == "phase_b":
        q_control = min(1.0, 0.5 + local_step / 8.0)
        p_control = max(0.5, 1.0 - local_step / 8.0)
    else:
        p_control, q_control = 1.0, 0.5
    behavior = {"P": p_control, "Q": q_control, "Y": 0.5}
    causal = dict(behavior)
    return {
        "local_step": local_step,
        "global_step": global_step,
        "examples_seen": examples_seen,
        "behavior": behavior,
        "causal": causal,
        "target_accuracy": 0.5,
        "selective_final_hidden_probe": {"P": 0.4, "Q": 0.2, "Y": 0.0},
        "boolean_signature": "00001111",
    }


def test_pilot_returns_before_phase_b_is_read_or_constructed(monkeypatch: pytest.MonkeyPatch) -> None:
    import forkworld.protocols_mediation as protocol

    config = _tiny_config(
        pilot_only=True, branch="independent_q_restore"
    )
    # Removing the full-run settings proves the pilot does not even read them.
    for key in ("phase_b_steps", "phase_b_checkpoints", "auc_horizon"):
        del config["h16"][key]

    def forbidden_phase_b(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("a manipulation-only pilot constructed phase B")

    monkeypatch.setattr(protocol, "make_handoff_phase_b", forbidden_phase_b)
    monkeypatch.setattr(protocol, "_compact_snapshot", _fast_snapshot)
    result = run_h16(config, seed=541)
    summary = result.summary

    assert summary["pilot_only"] is True
    assert summary["data"]["phase_b_constructed"] is False
    assert "phase_b_snapshots" not in summary
    assert "outcomes" not in summary
    assert summary["training"]["phase_b_steps"] == 0
    assert summary["postedit_pure_p"] is True
    assert set(summary["edit"]["branch_model_hashes"]) == {
        "independent_noop",
        "independent_q_restore",
        "independent_padding_sham",
        "nested_noop",
        "nested_q_transplant",
        "nested_padding_sham",
    }
    assert summary["edit"]["audit"]["all_six_branches_verified"] is True
    assert summary["edit"]["preactivation_effects"]["all_values_finite"] is True
    assert summary["eligibility"]["paired_intersection_eligible"] is True
    assert all(
        all(report["checks"].values()) for report in summary["replay"].values()
    )
    assert result.evaluation_batch is not None


def test_full_run_uses_fresh_optimizer_and_exact_common_knockout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import forkworld.protocols_mediation as protocol

    config = _tiny_config(
        pilot_only=False, branch="nested_q_transplant"
    )
    monkeypatch.setattr(protocol, "_compact_snapshot", _fast_snapshot)
    result = run_h16(copy.deepcopy(config), seed=541)
    summary = result.summary

    assert summary["pilot_only"] is False
    assert summary["data"]["phase_b_constructed"] is True
    assert summary["data"]["phase_b_pairing_verified"] is True
    assert summary["data"]["phase_b_batch_digest"]
    assert summary["data"]["phase_b_sampler_digest"] == static_sampler_digest(
        256,
        batch_size=64,
        steps=16,
        seed=90_000_541,
        shuffle=True,
    )
    optimizer = summary["optimizer_transition_audit"]
    assert optimizer["source"] == "fresh_empty_after_edit"
    assert optimizer["state_entry_count"] == 0
    assert optimizer["adam_step_entry_count"] == 0
    assert summary["hashes"]["phase_b_initial_model"] == summary["hashes"][
        "selected_postedit_model"
    ]
    assert list(summary["phase_b_snapshots"]) == [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "7",
        "8",
        "9",
        "13",
        "16",
    ]
    assert summary["outcomes"]["auc_horizon"] == 8
    assert summary["outcomes"][
        "normalized_control_auc_through_direct_checkpoint"
    ]["Q"] > 0.5
    assert summary["training"]["phase_b_steps"] == 16
    assert summary["training"]["phase_b_examples_seen"] == 16 * 64
    assert {record["experiment"] for record in result.metrics} == {"h16"}
    assert "phase_b_optimization" in {
        record["stage"] for record in result.metrics
    }
