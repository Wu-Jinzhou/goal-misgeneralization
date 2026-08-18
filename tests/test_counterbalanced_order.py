"""Exact construction and fail-closed stream audits for E20."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from forkworld.counterbalanced_order import (
    CONTROL_BIT_STRING,
    CONTROL_DIGEST_SHA256,
    FEATURE_NAMES,
    FOLD_DIGEST_SHA256,
    GOALS,
    ORDER_SCHEDULES,
    audit_concordant_washout,
    audit_counterbalanced_atomic_plan,
    audit_counterbalanced_bundle,
    audit_counterbalanced_component,
    audit_counterbalanced_factorial_panel,
    audit_counterbalanced_fold_control,
    audit_schedule_multiset_equality,
    counterbalanced_fold_ids,
    counterbalanced_truth_table_control,
    make_all_counterbalanced_atomic_plans,
    make_atomic_component_stream,
    make_counterbalanced_atomic_plan,
    make_counterbalanced_bundle,
    make_counterbalanced_component,
    make_counterbalanced_factorial_panel,
)


@pytest.fixture(scope="module")
def full_bundle():
    return make_counterbalanced_bundle(563)


@pytest.fixture(scope="module")
def smoke_bundle():
    return make_counterbalanced_bundle(569, rows_per_weight_unit=3)


def test_frozen_factorial_panel_reconstructs_fold_and_control_digests() -> None:
    panel = make_counterbalanced_factorial_panel()
    audit = audit_counterbalanced_factorial_panel(panel)
    fold_control = audit_counterbalanced_fold_control()
    raw_id = np.asarray(panel.latents["raw_codeword_id"], dtype=np.int64)
    tuple_id = np.asarray(panel.latents["candidate_tuple_id"], dtype=np.int64)
    folds = np.asarray(panel.latents["fold_id"], dtype=np.int64)
    control = np.asarray(panel.latents["truth_table_control"], dtype=np.int8)

    assert panel.feature_names(max_k=3, include_state=False) == FEATURE_NAMES
    assert panel.features(max_k=3, include_state=False).shape == (64, 8)
    assert np.array_equal(raw_id, np.arange(64))
    assert np.array_equal(folds, counterbalanced_fold_ids())
    assert np.array_equal(control, counterbalanced_truth_table_control())
    assert "".join("1" if value > 0 else "0" for value in control) == CONTROL_BIT_STRING
    assert fold_control["fold_digest_sha256"] == FOLD_DIGEST_SHA256
    assert fold_control["control_digest_sha256"] == CONTROL_DIGEST_SHA256
    assert audit["fold_control"] == fold_control
    assert np.array_equal(np.bincount(tuple_id, minlength=8), np.full(8, 8))
    assert np.array_equal(np.bincount(folds, minlength=8), np.full(8, 8))
    for fold in range(8):
        assert set(tuple_id[folds == fold]) == set(range(8))
        assert np.sum(control[folds == fold] > 0) == 4


def test_full_components_realize_exact_weights_targets_and_pooled_proof(full_bundle) -> None:
    audits = audit_counterbalanced_bundle(full_bundle)
    all_ids: list[np.ndarray] = []

    for goal in GOALS:
        batch = full_bundle.component(goal)
        component = audit_counterbalanced_component(batch, expected_goal=goal)
        raw_id = np.asarray(batch.latents["raw_codeword_id"], dtype=np.int64)
        raw_counts = np.bincount(raw_id, minlength=64)
        stratum = np.asarray(batch.latents["weighted_stratum_id"], dtype=np.int64)

        assert len(batch) == 9_216
        assert component["weighted_stratum_count"] == 96
        assert component["label_counts"] == {"-1": 4_608, "+1": 4_608}
        assert np.array_equal(np.bincount(stratum, minlength=96), np.full(96, 96))
        assert set(raw_counts) == {96, 192}
        assert np.sum(raw_counts == 96) == np.sum(raw_counts == 192) == 32
        assert component["accuracy"][goal]["value"] == 1.0
        assert component["accuracy"]["majority"]["numerator"] == 2
        assert component["accuracy"]["majority"]["denominator"] == 3
        for competitor in GOALS:
            if competitor != goal:
                assert component["accuracy"][competitor]["value"] == 0.5
        assert np.array_equal(batch.y, batch.latents["Y_goal"])
        assert np.array_equal(batch.target, batch.latents[f"{goal}_goal"])
        all_ids.append(np.asarray(batch.sample_id, dtype=np.int64))

    washout = full_bundle.washout
    washout_audit = audit_concordant_washout(washout)
    washout_raw = np.asarray(washout.latents["raw_codeword_id"], dtype=np.int64)
    assert len(washout) == 9_216
    assert washout_audit["raw_codeword_count"] == 16
    assert set(np.unique(washout_raw, return_counts=True)[1]) == {576}
    assert all(washout_audit["accuracy"][name]["value"] == 1.0 for name in (*GOALS, "majority"))
    all_ids.append(np.asarray(washout.sample_id, dtype=np.int64))
    assert len(np.unique(np.concatenate(all_ids))) == 4 * 9_216

    assert audits["feature_interface_equal"] is True
    assert audits["component_visible_support_equal"] is True
    assert audits["semantic_ids_pairwise_disjoint"] is True
    assert set(audits["component_semantic_digests"]) == set(GOALS)
    for name in (*GOALS, "majority", "bayes_raw_input"):
        pooled = audits["pooled"]["accuracy"][name]
        full = audits["pooled_with_washout"]["accuracy"][name]
        assert (pooled["numerator"], pooled["denominator"]) == (2, 3)
        assert (full["numerator"], full["denominator"]) == (3, 4)

    conditional = np.asarray(audits["pooled"]["raw_conditional_counts"], dtype=np.int64)
    sorted_counts = np.sort(conditional, axis=1)
    assert np.sum(np.all(sorted_counts == (0, 576), axis=1)) == 16
    assert np.sum(np.all(sorted_counts == (192, 192), axis=1)) == 48


def test_full_atomic_plan_has_exact_quotas_exposures_and_replay(full_bundle) -> None:
    plan = make_counterbalanced_atomic_plan(full_bundle, "p_q_y", seed=171_000_563)
    audit = audit_counterbalanced_atomic_plan(full_bundle, plan)

    assert plan.component_steps == 256
    assert plan.washout_steps == 256
    assert plan.total_steps == 1_024
    assert audit["total_sample_presentations"] == 294_912
    for goal in GOALS:
        stream = plan.component_indices[goal]
        target = np.asarray(full_bundle.component(goal).target, dtype=np.int8)
        raw_id = np.asarray(
            full_bundle.component(goal).latents["raw_codeword_id"], dtype=np.int64
        )
        stratum = np.asarray(
            full_bundle.component(goal).latents["weighted_stratum_id"], dtype=np.int64
        )
        assert stream.shape == (256, 288)
        for presentation in range(8):
            assert np.array_equal(stream[presentation * 32 : (presentation + 1) * 32], stream[:32])
        assert np.array_equal(
            np.bincount(stream.reshape(-1), minlength=9_216), np.full(9_216, 8)
        )
        assert np.all(np.sum(target[stream] > 0, axis=1) == 144)
        assert set(np.unique(raw_id[stream[0]], return_counts=True)[1]) == {3, 6}
        assert np.array_equal(
            np.bincount(stratum[stream[0]], minlength=96), np.full(96, 3)
        )
        stream_audit = audit["components"][goal]
        assert stream_audit["row_exposure_count_values"] == [8]
        assert stream_audit["raw_quota_values_per_batch"] == [3, 6]
        assert len(stream_audit["ordered_atomic_batch_digests"]) == 256

    washout_stream = plan.washout_indices
    washout_raw = np.asarray(full_bundle.washout.latents["raw_codeword_id"], dtype=np.int64)
    assert washout_stream.shape == (256, 288)
    assert set(np.unique(washout_raw[washout_stream[0]], return_counts=True)[1]) == {18}
    assert audit["washout"]["raw_quota_values_per_batch"] == [18]


def test_six_smoke_orders_reuse_streams_but_have_distinct_order_digests(smoke_bundle) -> None:
    plans = make_all_counterbalanced_atomic_plans(
        smoke_bundle,
        seed=171_000_569,
        presentations=2,
        batches_per_presentation=1,
    )
    audit = audit_schedule_multiset_equality(smoke_bundle, plans)

    assert tuple(plans) == ORDER_SCHEDULES
    assert audit["schedule_count"] == 6
    assert audit["component_streams_bit_identical"] is True
    assert audit["washout_stream_bit_identical"] is True
    assert audit["order_invariant_multiset_equal"] is True
    assert len(set(audit["ordered_stream_digests"].values())) == 6
    reference = plans[ORDER_SCHEDULES[0]]
    for plan in plans.values():
        assert plan.total_steps == 8
        for goal in GOALS:
            assert np.array_equal(reference.component_indices[goal], plan.component_indices[goal])
        assert np.array_equal(reference.washout_indices, plan.washout_indices)


def test_streams_are_deterministic_and_seed_sensitive(smoke_bundle) -> None:
    first = make_counterbalanced_atomic_plan(
        smoke_bundle, "q_y_p", 17, presentations=2, batches_per_presentation=1
    )
    repeated = make_counterbalanced_atomic_plan(
        smoke_bundle, "q_y_p", 17, presentations=2, batches_per_presentation=1
    )
    changed = make_counterbalanced_atomic_plan(
        smoke_bundle, "q_y_p", 19, presentations=2, batches_per_presentation=1
    )

    assert first.plan_digest == repeated.plan_digest
    assert first.ordered_stream_digest == repeated.ordered_stream_digest
    assert all(
        np.array_equal(first.component_indices[goal], repeated.component_indices[goal])
        for goal in GOALS
    )
    assert first.ordered_stream_digest != changed.ordered_stream_digest


def test_component_audits_fail_closed_on_target_and_visible_cue_corruption(smoke_bundle) -> None:
    component = smoke_bundle.component("P")
    target = np.array(component.target, copy=True)
    target[0] *= -1
    corrupted_target = component.with_updates(target=target, reward=target.astype(np.float32))
    with pytest.raises(RuntimeError, match=r"must train on SemanticBatch.target"):
        audit_counterbalanced_component(corrupted_target, expected_goal="P")

    channels = dict(component.channels)
    channels["component_id"] = np.zeros(len(component), dtype=np.int8)
    visible_cue = component.with_updates(channels=channels)
    with pytest.raises(RuntimeError, match="feature names/order changed"):
        audit_counterbalanced_component(visible_cue, expected_goal="P")


def test_fold_and_bundle_audits_fail_closed_on_metadata_corruption(smoke_bundle) -> None:
    panel = make_counterbalanced_factorial_panel()
    latents = dict(panel.latents)
    fold = np.array(latents["fold_id"], copy=True)
    fold[0] = 7
    latents["fold_id"] = fold
    corrupted_panel = panel.with_updates(latents=latents)
    with pytest.raises(RuntimeError, match="fold IDs differ"):
        audit_counterbalanced_factorial_panel(corrupted_panel)

    p_ids = np.asarray(smoke_bundle.component("P").sample_id, dtype=np.int64)
    overlapping_q = make_counterbalanced_component(
        "Q", 569, rows_per_weight_unit=3, id_offset=int(p_ids.min())
    )
    components = dict(smoke_bundle.components)
    components["Q"] = overlapping_q
    overlapping = replace(smoke_bundle, components=components, audits={})
    with pytest.raises(RuntimeError, match="values overlap"):
        audit_counterbalanced_bundle(overlapping)


def test_atomic_audits_fail_closed_on_index_and_schedule_corruption(smoke_bundle) -> None:
    plan = make_counterbalanced_atomic_plan(
        smoke_bundle, "p_y_q", 23, presentations=2, batches_per_presentation=1
    )
    component_indices = dict(plan.component_indices)
    corrupted_p = np.array(component_indices["P"], copy=True)
    corrupted_p[0, 0] = corrupted_p[0, 1]
    component_indices["P"] = corrupted_p
    corrupted = replace(plan, component_indices=component_indices)
    with pytest.raises(RuntimeError, match="repeats a stored row"):
        audit_counterbalanced_atomic_plan(smoke_bundle, corrupted)

    with pytest.raises(RuntimeError, match="each of the six permutations"):
        audit_schedule_multiset_equality(smoke_bundle, [plan])


def test_invalid_goals_dimensions_schedules_and_id_ranges_fail_early(smoke_bundle) -> None:
    with pytest.raises(ValueError, match="goal must be one of"):
        make_counterbalanced_component("Z", 1)
    with pytest.raises(ValueError, match="rows_per_weight_unit must be a positive integer"):
        make_counterbalanced_component("P", 1, rows_per_weight_unit=0)
    with pytest.raises(ValueError, match="int64 ID range"):
        make_counterbalanced_component("P", 1, id_offset=np.iinfo(np.int64).max)
    with pytest.raises(ValueError, match="schedule must be one of"):
        make_counterbalanced_atomic_plan(
            smoke_bundle, "p_p_y", 1, presentations=2, batches_per_presentation=1
        )
    with pytest.raises(ValueError, match="rows_per_weight_unit must equal"):
        make_atomic_component_stream(
            smoke_bundle.component("P"),
            "P",
            1,
            presentations=2,
            batches_per_presentation=2,
        )
