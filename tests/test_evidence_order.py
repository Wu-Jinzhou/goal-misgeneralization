"""Exact-data and atomic-stream invariants for E18/H15."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from forkworld.evidence_order import (
    ORDER_SCHEDULES,
    audit_atomic_evidence_plan,
    audit_evidence_strata,
    evidence_strata,
    make_atomic_evidence_plan,
    make_identical_evidence_dataset,
)


def _full_batch():
    return make_identical_evidence_dataset(
        10_000,
        q_p=0.90,
        q_q=0.95,
        k_q=2,
        k_y=3,
        seed=269,
        max_k_q=3,
        max_k_y=5,
        state_dim=8,
    )


def _full_plan(batch, schedule: str = "b_then_d", seed: int = 91_000_269):
    return make_atomic_evidence_plan(
        batch,
        schedule=schedule,
        prefix_a_repetitions=90,
        washout_a_repetitions=10,
        diagnostic_repetitions=100,
        batch_size=250,
        seed=seed,
    )


def test_factorial_selection_has_exact_registered_strata_and_raw_counts() -> None:
    batch = _full_batch()
    audit = audit_evidence_strata(
        batch, expected_counts={"A": 9_000, "B": 500, "D": 500}
    )

    assert audit["q_only_count"] == 0
    assert {name: item["total"] for name, item in audit["strata"].items()} == {
        "A": 9_000,
        "B": 500,
        "D": 500,
    }
    assert all(item["raw_codeword_count"] == 16 for item in audit["strata"].values())
    assert set(audit["strata"]["A"]["raw_codeword_counts"].values()) == {562, 563}
    assert set(audit["strata"]["B"]["raw_codeword_counts"].values()) == {31, 32}
    assert set(audit["strata"]["D"]["raw_codeword_counts"].values()) == {31, 32}
    assert len(np.unique(batch.sample_id)) == len(batch)
    assert len(np.unique(batch.state_id)) == len(batch)


def test_all_schedules_share_atomic_multiset_and_exact_raw_balancing() -> None:
    batch = _full_batch()
    plans = [_full_plan(batch, schedule) for schedule in ORDER_SCHEDULES]

    assert {(plan.prefix_steps, plan.block_steps, plan.washout_steps) for plan in plans} == {
        (3_240, 200, 360)
    }
    assert {plan.total_steps for plan in plans} == {4_000}
    assert len({plan.row_exposure_digest for plan in plans}) == 1
    assert len({plan.atomic_batch_multiset_digest for plan in plans}) == 1
    assert len({tuple(sorted(plan.component_digests.items())) for plan in plans}) == 1
    assert len({plan.ordered_digest for plan in plans}) == 3
    assert all(np.array_equal(plans[0].phase("prefix"), plan.phase("prefix")) for plan in plans)
    assert all(np.array_equal(plans[0].phase("washout"), plan.phase("washout")) for plan in plans)

    raw_ids = np.asarray(batch.latents["raw_codeword_id"], dtype=np.int64)
    targets = np.asarray(batch.target, dtype=np.int8)
    masks = evidence_strata(batch)

    def assert_once_per_repetition(
        stream: np.ndarray, mask: np.ndarray, repetitions: int
    ) -> None:
        batches_per_repetition = int(np.sum(mask)) // 250
        assert len(stream) == repetitions * batches_per_repetition
        for repetition in range(repetitions):
            start = repetition * batches_per_repetition
            chunk = stream[start : start + batches_per_repetition]
            counts = np.bincount(chunk.reshape(-1), minlength=len(batch))
            assert np.array_equal(counts, mask.astype(np.int64))

    for plan in plans:
        audit = audit_atomic_evidence_plan(
            batch,
            plan,
            prefix_a_repetitions=90,
            washout_a_repetitions=10,
            diagnostic_repetitions=100,
        )
        assert audit["per_row_presentation_min"] == 100
        assert audit["per_row_presentation_max"] == 100
        assert audit["per_batch_raw_count_min"] == 15
        assert audit["per_batch_raw_count_max"] == 16
        assert audit["every_row_once_per_registered_repetition"] is True
        assert audit["batches_per_component_repetition"] == {
            "A_prefix": 36,
            "B": 2,
            "D": 2,
            "A_washout": 36,
        }
        assert np.all(np.sum(targets[plan.indices] > 0, axis=1) == 125)
        for row in plan.indices:
            assert set(np.unique(raw_ids[row], return_counts=True)[1]) <= {15, 16}
        if plan.schedule == "b_then_d":
            b_stream, d_stream = plan.phase("block_one"), plan.phase("block_two")
        elif plan.schedule == "d_then_b":
            d_stream, b_stream = plan.phase("block_one"), plan.phase("block_two")
        else:
            b_stream = plan.phase("diagnostics")[0::2]
            d_stream = plan.phase("diagnostics")[1::2]
        assert_once_per_repetition(plan.phase("prefix"), masks["A"], 90)
        assert_once_per_repetition(b_stream, masks["B"], 100)
        assert_once_per_repetition(d_stream, masks["D"], 100)
        assert_once_per_repetition(plan.phase("washout"), masks["A"], 10)


def test_component_streams_are_bit_identical_but_schedule_order_differs() -> None:
    batch = _full_batch()
    b_then_d = _full_plan(batch, "b_then_d")
    d_then_b = _full_plan(batch, "d_then_b")
    interleave = _full_plan(batch, "interleave")

    assert np.array_equal(
        b_then_d.phase("block_one"), d_then_b.phase("block_two")
    )
    assert np.array_equal(
        b_then_d.phase("block_two"), d_then_b.phase("block_one")
    )
    assert np.array_equal(interleave.phase("diagnostics")[0::2], b_then_d.phase("block_one"))
    assert np.array_equal(interleave.phase("diagnostics")[1::2], b_then_d.phase("block_two"))


def test_smoke_scale_preserves_all_atomic_invariants() -> None:
    batch = make_identical_evidence_dataset(
        400,
        q_p=0.75,
        q_q=0.875,
        k_q=2,
        k_y=2,
        seed=389,
        max_k_q=3,
        max_k_y=5,
        state_dim=2,
    )
    audit = audit_evidence_strata(
        batch, expected_counts={"A": 300, "B": 50, "D": 50}
    )
    assert all(item["raw_codeword_count"] == 8 for item in audit["strata"].values())

    plans = [
        make_atomic_evidence_plan(
            batch,
            schedule=schedule,
            prefix_a_repetitions=3,
            washout_a_repetitions=1,
            diagnostic_repetitions=4,
            batch_size=50,
            seed=17_000_389,
        )
        for schedule in ORDER_SCHEDULES
    ]
    assert {(plan.prefix_steps, plan.block_steps, plan.washout_steps) for plan in plans} == {
        (18, 4, 6)
    }
    assert {plan.total_steps for plan in plans} == {32}
    assert len({plan.atomic_batch_multiset_digest for plan in plans}) == 1
    for plan in plans:
        plan_audit = audit_atomic_evidence_plan(
            batch,
            plan,
            prefix_a_repetitions=3,
            washout_a_repetitions=1,
            diagnostic_repetitions=4,
        )
        assert (plan_audit["per_batch_raw_count_min"], plan_audit["per_batch_raw_count_max"]) == (
            6,
            7,
        )
        assert (plan_audit["per_row_presentation_min"], plan_audit["per_row_presentation_max"]) == (
            4,
            4,
        )


def test_audit_fails_closed_on_phase_mutation() -> None:
    batch = _full_batch()
    plan = _full_plan(batch)
    masks = evidence_strata(batch)
    corrupted_indices = np.array(plan.indices, copy=True)
    original_target = batch.target[corrupted_indices[0, 0]]
    replacement = np.flatnonzero(masks["B"] & (batch.target == original_target))[0]
    corrupted_indices[0, 0] = replacement
    corrupted = replace(plan, indices=corrupted_indices)

    with pytest.raises(RuntimeError, match="prefix contains non-A"):
        audit_atomic_evidence_plan(
            batch,
            corrupted,
            prefix_a_repetitions=90,
            washout_a_repetitions=10,
            diagnostic_repetitions=100,
        )


def test_atomic_plan_is_deterministic_and_seed_sensitive() -> None:
    batch = _full_batch()
    first = _full_plan(batch, seed=91_000_269)
    repeat = _full_plan(batch, seed=91_000_269)
    changed = _full_plan(batch, seed=91_000_271)

    assert first.ordered_digest == repeat.ordered_digest
    assert first.atomic_batch_multiset_digest == repeat.atomic_batch_multiset_digest
    assert first.ordered_digest != changed.ordered_digest
    assert first.atomic_batch_multiset_digest != changed.atomic_batch_multiset_digest
