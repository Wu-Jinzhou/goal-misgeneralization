"""Protocol integration checks for E20/H17 counterbalanced order."""

from __future__ import annotations

import copy
from collections.abc import Mapping

import numpy as np
import pytest

from forkworld.config import load_config
from forkworld.protocols import run_protocol
from forkworld.protocols_counterbalanced import (
    DATA_SEED,
    PHASE_SEEDS,
    STREAM_SEED,
    replay_h17_observer_free,
)
from forkworld.runner import smoke_config

SCHEDULES = ("p_q_y", "p_y_q", "q_p_y", "q_y_p", "y_p_q", "y_q_p")


@pytest.fixture(scope="module")
def pilot_smoke_summaries() -> dict[str, Mapping[str, object]]:
    base = smoke_config(load_config("configs/e20_counterbalanced_order_pilot.yaml"))
    summaries: dict[str, Mapping[str, object]] = {}
    for goal in ("P", "Q", "Y"):
        config = copy.deepcopy(base)
        config["h17"]["isolated_goal"] = goal
        summaries[goal] = run_protocol(config, seed=0).summary
    return summaries


@pytest.fixture(scope="module")
def full_smoke_summaries() -> dict[str, Mapping[str, object]]:
    base = smoke_config(load_config("configs/e20_counterbalanced_order.yaml"))
    summaries: dict[str, Mapping[str, object]] = {}
    for schedule in SCHEDULES:
        config = copy.deepcopy(base)
        config["h17"]["schedule"] = schedule
        summaries[schedule] = run_protocol(config, seed=0).summary
    return summaries


def test_isolated_pilot_constructs_only_selected_component(
    pilot_smoke_summaries: Mapping[str, Mapping[str, object]],
) -> None:
    initial_hashes = set()
    for goal, raw_summary in pilot_smoke_summaries.items():
        summary = dict(raw_summary)
        isolation = summary["construction_isolation"]
        assert isinstance(isolation, Mapping)
        assert isolation["constructed_components"] == [goal]
        assert isolation["unconstructed_components"] == [
            candidate for candidate in ("P", "Q", "Y") if candidate != goal
        ]
        for field in (
            "other_components_constructed",
            "component_bundle_constructed",
            "schedule_defined",
            "schedule_constructed",
            "washout_constructed",
            "pooled_data_constructed",
            "order_estimand_computed",
        ):
            assert isolation[field] is False
        assert summary["schedule"] is None
        assert summary["washout_snapshots"] == {}
        assert list(summary["isolated_component_snapshots"]) == ["0", "1", "2"]
        assert summary["observer_on_execution"]["boundary_order"] == [
            "initial",
            f"isolated_{goal}",
        ]
        assert set(summary["data"]["components"]) == {goal}
        assert summary["data"]["washout"] == {
            "constructed": False,
            "semantic_batch_digest": None,
            "audit": None,
        }
        assert summary["pilot_gate"]["joint_seed_gate_computed"] is False
        assert summary["pilot_gate"]["full_panel_authorized_here"] is False
        assert summary["training"]["total_steps"] == 2
        initial_hashes.add(summary["hashes"]["model_boundaries"]["initial"])
    assert len(initial_hashes) == 1


def test_full_schedules_reuse_component_and_washout_streams_but_reorder_them(
    full_smoke_summaries: Mapping[str, Mapping[str, object]],
) -> None:
    summaries = list(full_smoke_summaries.values())
    assert len({summary["hashes"]["model_boundaries"]["initial"] for summary in summaries}) == 1
    for goal in ("P", "Q", "Y"):
        assert len(
            {
                summary["observer_on_execution"]["streams"][goal][
                    "ordered_stream_digest"
                ]
                for summary in summaries
            }
        ) == 1
        assert len(
            {
                summary["data"]["components"][goal]["semantic_batch_digest"]
                for summary in summaries
            }
        ) == 1
    assert len(
        {
            summary["observer_on_execution"]["streams"]["washout"][
                "ordered_stream_digest"
            ]
            for summary in summaries
        }
    ) == 1
    assert len({summary["plan"]["ordered_stream_digest"] for summary in summaries}) == 6
    assert len(
        {summary["plan"]["atomic_batch_multiset_digest"] for summary in summaries}
    ) == 1

    for first_goal in ("P", "Q", "Y"):
        pair = [
            summary
            for summary in summaries
            if summary["order"][0] == first_goal
        ]
        assert len(pair) == 2
        key = f"component_1_{first_goal}"
        assert pair[0]["hashes"]["model_boundaries"][key] == pair[1]["hashes"][
            "model_boundaries"
        ][key]


def test_full_smoke_reset_checkpoint_and_snapshot_schema(
    full_smoke_summaries: Mapping[str, Mapping[str, object]],
) -> None:
    summary = full_smoke_summaries["p_q_y"]
    assert summary["observer_on_execution"]["boundary_order"] == [
        "initial",
        "component_1_P",
        "component_2_Q",
        "component_3_Y",
        "washout",
    ]
    assert [block["goal"] for block in summary["component_snapshots"]] == [
        "P",
        "Q",
        "Y",
    ]
    for block in summary["component_snapshots"]:
        assert list(block["snapshots"]) == ["0", "1", "2"]
    assert list(summary["washout_snapshots"]) == ["0", "1", "2"]
    assert summary["training"]["total_steps"] == 8
    assert summary["training"]["total_examples_seen"] == 8 * 288
    assert summary["training"]["all_minibatches_full"] is True

    assert len(summary["resets"]) == 4
    for reset in summary["resets"].values():
        assert reset["optimizer_class"] == "AdamW"
        assert reset["state_entry_count"] == 0
        assert reset["adam_step_entry_count"] == 0
        assert reset["betas"] == [0.9, 0.999]
        assert reset["eps"] == 1e-8
        assert reset["model_unchanged_by_optimizer_reset"] is True

    snapshot = summary["washout_snapshots"]["2"]
    assert set(snapshot["behavior"]) == {"P", "Q", "Y", "M"}
    assert set(snapshot["causal"]) == {"P", "Q", "Y"}
    assert set(snapshot["control_margin"]) == {"P", "Q", "Y"}
    assert len(snapshot["raw_hard_signature"]) == 64
    assert len(snapshot["raw_probability_table"]) == 64
    assert [row["raw_id"] for row in snapshot["raw_codeword_table"]] == list(range(64))
    assert all(row["features"][1] == row["features"][5] == 1.0 for row in snapshot["raw_codeword_table"])
    assert set(snapshot["probe_cross_validated_accuracy"]) == {
        "first_hidden",
        "final_hidden",
    }
    for layer in ("first_hidden", "final_hidden"):
        assert set(snapshot["probe_cross_validated_accuracy"][layer]) == {
            "P",
            "Q",
            "Y",
            "truth_table_control",
        }
        assert set(snapshot["selective_probe_advantage"][layer]) == {"P", "Q", "Y"}
    assert snapshot["zero_logit_count"] >= 0
    assert snapshot["truth_table_class"]["kind"] in {
        "exact_named_rule",
        "other_candidate_tuple_consistent",
        "raw_codeword_specific_composite",
    }


def test_observer_free_replay_matches_stored_smoke_fingerprint() -> None:
    config = smoke_config(load_config("configs/e20_counterbalanced_order.yaml"))
    observed = run_protocol(config, seed=17)
    replay = replay_h17_observer_free(config, seed=17)
    stored = observed.summary["observer_on_execution"]
    assert replay["observer_enabled"] is False
    assert replay["trajectory_fingerprint"] == stored["trajectory_fingerprint"]
    assert replay["boundary_order"] == stored["boundary_order"]
    assert replay["hashes"] == stored["hashes"]
    assert replay["resets"] == stored["resets"]
    assert replay["streams"] == stored["streams"]
    assert replay["training"] == stored["training"]

    second = run_protocol(config, seed=17)
    assert second.summary["observer_on_execution"]["trajectory_fingerprint"] == stored[
        "trajectory_fingerprint"
    ]
    assert second.checkpoints == observed.checkpoints == {}
    assert second.predictions == observed.predictions == []
    assert np.array_equal(
        np.asarray(second.summary["final"]["raw_probability_table"]),
        np.asarray(observed.summary["final"]["raw_probability_table"]),
    )


def test_registered_run_seeds_share_exact_data_stream_and_phase_rng() -> None:
    config = smoke_config(load_config("configs/e20_counterbalanced_order.yaml"))
    first = run_protocol(config, seed=17).summary
    second = run_protocol(config, seed=19).summary

    for summary, run_seed in ((first, 17), (second, 19)):
        assert summary["randomness"] == {
            "nominal_run_seed": run_seed,
            "model_initialization_seed": run_seed,
            "data_seed": DATA_SEED,
            "stream_seed": STREAM_SEED,
            "phase_seeds": PHASE_SEEDS,
            "run_seed_affects_only_model_initialization": True,
            "data_stream_and_phase_rng_independent_of_run_seed": True,
        }
        assert [
            block["phase_rng_seed"] for block in summary["training"]["blocks"]
        ] == [
            PHASE_SEEDS["component_P"],
            PHASE_SEEDS["component_Q"],
            PHASE_SEEDS["component_Y"],
            PHASE_SEEDS["washout"],
        ]
        assert summary["model"]["bias"] is True
        assert summary["model"]["nuisance_bits"] == 0
        assert summary["model"]["auxiliary_head_count"] == 0
        assert summary["training"]["label_smoothing"] == 0.0
        assert summary["training"]["gradient_clip_norm"] == 1.0

    assert first["hashes"]["data"] == second["hashes"]["data"]
    assert first["plan"] == second["plan"]
    assert first["observer_on_execution"]["streams"] == second[
        "observer_on_execution"
    ]["streams"]
    assert first["hashes"]["model_boundaries"]["initial"] != second["hashes"][
        "model_boundaries"
    ]["initial"]


def test_held_initialization_removes_all_nominal_run_seed_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import forkworld.protocols_counterbalanced as protocol

    config = smoke_config(load_config("configs/e20_counterbalanced_order.yaml"))
    original_build_model = protocol.build_model

    def fixed_build_model(
        batch: object,
        raw_config: Mapping[str, object],
        _nominal_seed: int,
        **kwargs: object,
    ) -> object:
        return original_build_model(batch, raw_config, 424_242, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(protocol, "build_model", fixed_build_model)
    first = protocol.replay_h17_observer_free(config, seed=17)
    second = protocol.replay_h17_observer_free(config, seed=19)

    assert first["hashes"] == second["hashes"]
    assert first["streams"] == second["streams"]
    assert first["plan"] == second["plan"]
    assert first["training"] == second["training"]
    # The artifact trajectory identity deliberately retains nominal/model seed
    # audit metadata even when a synthetic test holds initialization fixed.
    assert first["trajectory_fingerprint"] != second["trajectory_fingerprint"]


def test_full_smoke_metric_stage_schema() -> None:
    config = smoke_config(load_config("configs/e20_counterbalanced_order.yaml"))
    result = run_protocol(config, seed=0)
    stages = {record["stage"] for record in result.metrics}
    for position, goal in enumerate(("p", "q", "y"), start=1):
        prefix = f"component_{position}_{goal}"
        assert {
            f"{prefix}_behavior",
            f"{prefix}_truth_table",
            f"{prefix}_probe",
            f"{prefix}_causal",
            f"{prefix}_optimization",
        } <= stages
    assert {
        "washout_behavior",
        "washout_truth_table",
        "washout_probe",
        "washout_causal",
        "washout_optimization",
    } <= stages
