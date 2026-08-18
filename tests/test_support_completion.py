"""Focused invariants for the adaptive support-completion intervention."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from forkworld.competing import (
    make_competing_bundle,
    make_competing_dataset,
    make_competing_factorial_dataset,
)
from forkworld.protocols_dynamics import (
    cumulative_subset_presentations,
    q_only_codeword_partition,
    static_subset_exposure,
)


def _batch(m: int):
    return make_competing_dataset(
        10_000,
        q_p=0.90,
        q_q=0.95,
        k_q=2,
        k_y=5,
        seed=41,
        max_k_q=3,
        max_k_y=5,
        overlap="nested",
        q_only_error_count=m,
        state_dim=3,
        split="train",
    )


@pytest.mark.parametrize("m", [0, 2, 10, 50, 100, 250, 450])
def test_support_completion_preserves_marginals_and_realizes_registered_cells(m: int) -> None:
    batch = _batch(m)
    metadata = batch.metadata

    assert metadata["realized_q_p"] == 0.90
    assert metadata["realized_q_q"] == 0.95
    assert metadata["p_error_count"] == 1_000
    assert metadata["q_error_count"] == 500
    assert metadata["both_error_count"] == 500 - m
    assert metadata["p_only_error_count"] == 500 + m
    assert metadata["q_only_error_count"] == m
    assert metadata["neither_error_count"] == 9_000 - m
    assert metadata["overlap_mode"] == "support_completion"
    assert metadata["reference_overlap_mode"] == "nested"
    assert metadata["requested_q_only_error_count"] == m


def test_zero_support_completion_is_array_identical_to_nested_reference() -> None:
    reference = make_competing_dataset(
        10_000,
        q_p=0.90,
        q_q=0.95,
        k_q=2,
        k_y=5,
        seed=41,
        max_k_q=3,
        max_k_y=5,
        overlap="nested",
        state_dim=3,
        split="train",
    )
    completed = _batch(0)

    assert np.array_equal(completed.y, reference.y)
    assert completed.sample_id is not None and reference.sample_id is not None
    assert completed.state is not None and reference.state is not None
    assert np.array_equal(completed.sample_id, reference.sample_id)
    assert np.array_equal(completed.state, reference.state)
    assert set(completed.channels) == set(reference.channels)
    assert set(completed.latents) == set(reference.latents)
    for name in completed.channels:
        assert np.array_equal(completed.channels[name], reference.channels[name])
    for name in completed.latents:
        assert np.array_equal(completed.latents[name], reference.latents[name])


def test_support_completion_arms_are_row_paired_and_nested() -> None:
    counts = (0, 2, 10, 50, 100, 250, 450)
    batches = [_batch(m) for m in counts]
    baseline = batches[0]

    previous_shared = np.asarray(baseline.latents["P_error"]) & np.asarray(
        baseline.latents["Q_error"]
    )
    previous_q_only = ~np.asarray(baseline.latents["P_error"]) & np.asarray(
        baseline.latents["Q_error"]
    )
    for batch in batches:
        assert np.array_equal(batch.y, baseline.y)
        assert np.array_equal(batch.sample_id, baseline.sample_id)
        assert np.array_equal(batch.state, baseline.state)
        assert np.array_equal(batch.channels["P"], baseline.channels["P"])
        assert np.array_equal(batch.latents["P_error"], baseline.latents["P_error"])
        for name in ("P_present", "Q_present", "Q_1", "Q_3", "R_1", "R_2", "R_3", "R_4", "R_5"):
            assert np.array_equal(batch.channels[name], baseline.channels[name])

        shared = np.asarray(batch.latents["P_error"]) & np.asarray(batch.latents["Q_error"])
        q_only = ~np.asarray(batch.latents["P_error"]) & np.asarray(batch.latents["Q_error"])
        assert not np.any(shared & ~previous_shared)
        assert not np.any(previous_q_only & ~q_only)
        previous_shared = shared
        previous_q_only = q_only

        q_changed = np.asarray(batch.latents["Q_goal"]) != np.asarray(
            baseline.latents["Q_goal"]
        )
        assert np.array_equal(
            batch.channels["Q_2"] != baseline.channels["Q_2"], q_changed
        )


@pytest.mark.parametrize("m", [-1, 501, True])
def test_support_completion_rejects_infeasible_counts(m: object) -> None:
    with pytest.raises(ValueError, match="q_only_error_count"):
        make_competing_dataset(
            10_000,
            q_p=0.90,
            q_q=0.95,
            k_q=2,
            k_y=5,
            overlap="nested",
            q_only_error_count=m,  # type: ignore[arg-type]
        )


def test_support_completion_requires_nested_reference_ranking() -> None:
    with pytest.raises(ValueError, match="nested reference"):
        make_competing_dataset(
            10_000,
            q_p=0.90,
            q_q=0.95,
            k_q=2,
            k_y=5,
            overlap="independent",
            q_only_error_count=2,
        )


def test_bundle_intervenes_on_training_only_and_keeps_common_iid() -> None:
    common: dict[str, Any] = dict(
        n_train=10_000,
        n_iid=4_000,
        n_diagnostic=256,
        q_p=0.90,
        q_q=0.95,
        k_q=2,
        k_y=5,
        seed=23,
        max_k_q=3,
        max_k_y=5,
        overlap="nested",
        state_dim=2,
    )
    low = make_competing_bundle(**common, q_only_error_count=0)
    high = make_competing_bundle(**common, q_only_error_count=450)

    assert low.train.metadata["q_only_error_count"] == 0
    assert high.train.metadata["q_only_error_count"] == 450
    assert low.iid.metadata["q_only_error_count"] == 0
    assert high.iid.metadata["q_only_error_count"] == 0
    assert np.array_equal(low.iid.y, high.iid.y)
    assert low.iid.state is not None and high.iid.state is not None
    assert np.array_equal(low.iid.state, high.iid.state)
    for name in low.iid.channels:
        assert np.array_equal(low.iid.channels[name], high.iid.channels[name])
    for name in low.iid.latents:
        assert np.array_equal(low.iid.latents[name], high.iid.latents[name])


def test_m450_matches_independent_cell_counts_but_not_row_allocation() -> None:
    completed = _batch(450)
    independent = make_competing_dataset(
        10_000,
        q_p=0.90,
        q_q=0.95,
        k_q=2,
        k_y=5,
        seed=41,
        max_k_q=3,
        max_k_y=5,
        overlap="independent",
        state_dim=3,
        split="train",
    )
    fields = (
        "p_error_count",
        "q_error_count",
        "both_error_count",
        "p_only_error_count",
        "q_only_error_count",
        "neither_error_count",
    )
    assert {name: completed.metadata[name] for name in fields} == {
        name: independent.metadata[name] for name in fields
    }
    assert not np.array_equal(
        completed.latents["Q_error"], independent.latents["Q_error"]
    )


def test_static_subset_exposure_matches_direct_shuffled_batching() -> None:
    mask = np.asarray([False, True, False, True, False, False, True, False, False, True])
    result = static_subset_exposure(mask, batch_size=4, steps=8, seed=79)

    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(79)
    order = torch.empty(0, dtype=torch.long)
    cursor = 0
    cumulative = 0
    seen: set[int] = set()
    expected_cumulative = {0: 0}
    expected_unique = {0: 0}
    first: int | None = None
    all_seen: int | None = None
    for step in range(1, 9):
        if cursor >= len(order):
            order = torch.randperm(len(mask), generator=generator)
            cursor = 0
        stop = min(cursor + 4, len(order))
        selected = order[cursor:stop].tolist()
        selected_subset = [index for index in selected if mask[index]]
        cumulative += len(selected_subset)
        seen.update(selected_subset)
        if selected_subset and first is None:
            first = step
        if len(seen) == int(np.sum(mask)) and all_seen is None:
            all_seen = step
        cursor = stop
        expected_cumulative[step] = cumulative
        expected_unique[step] = len(seen)

    assert result["cumulative_by_step"] == expected_cumulative
    assert result["unique_by_step"] == expected_unique
    assert result["first_presentation_step"] == first
    assert result["all_unique_seen_step"] == all_seen
    assert cumulative_subset_presentations(
        mask, batch_size=4, steps=8, seed=79
    ) == expected_cumulative


def test_q_only_factorial_partition_identifies_seen_and_unseen_active_codes() -> None:
    training = _batch(2)
    factorial = make_competing_factorial_dataset(
        n=4_096,
        k_q=2,
        k_y=5,
        seed=97,
        max_k_q=3,
        max_k_y=5,
        state_dim=3,
    )
    partition = q_only_codeword_partition(training, factorial)

    assert partition["training_q_only_rows"] == 2
    assert partition["seen_codeword_count"] == 2
    assert partition["possible_codeword_count"] == 64
    assert int(np.sum(partition["q_only"])) == 1_024
    assert int(np.sum(partition["seen"])) == 32
    assert int(np.sum(partition["unseen"])) == 992
    assert not np.any(np.asarray(partition["seen"]) & np.asarray(partition["unseen"]))
    assert np.array_equal(
        np.asarray(partition["seen"]) | np.asarray(partition["unseen"]),
        np.asarray(partition["q_only"]),
    )
