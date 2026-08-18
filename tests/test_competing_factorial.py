from __future__ import annotations

import numpy as np
import pytest

from forkworld.competing import (
    competing_causal_flip_batches,
    decode_competing_rules,
    make_competing_factorial_dataset,
)


def _active_raw_rows(batch: object) -> np.ndarray:
    channel_order = batch.metadata["raw_channel_order"]  # type: ignore[attr-defined]
    return np.column_stack(
        [np.asarray(batch.channels[name], dtype=np.int8) for name in channel_order]  # type: ignore[attr-defined]
    )


def test_factorial_dataset_exhausts_raw_codes_and_balances_candidates() -> None:
    batch = make_competing_factorial_dataset(
        repeats=3,
        k_q=2,
        k_y=3,
        seed=17,
        control_seed=29,
        max_k_q=3,
        max_k_y=4,
        state_dim=2,
    )
    rules = decode_competing_rules(batch)
    raw_ids = np.asarray(batch.latents["raw_codeword_id"], dtype=np.int64)
    candidate_ids = np.asarray(batch.latents["candidate_tuple_id"], dtype=np.int64)

    # There are 2^(1 + k_q + k_y) active observation codewords, each repeated
    # exactly three times with unambiguous within-codeword replicate IDs.
    assert len(batch) == 64 * 3
    assert np.array_equal(np.bincount(raw_ids, minlength=64), np.full(64, 3))
    for raw_id in range(64):
        replicates = np.asarray(batch.latents["replicate_id"])[raw_ids == raw_id]
        assert np.array_equal(np.sort(replicates), np.arange(3))

    active_rows = _active_raw_rows(batch)
    assert len(np.unique(active_rows, axis=0)) == 64
    expected_rows = 2 * (
        (raw_ids[:, None].astype(np.uint64) >> np.arange(6, dtype=np.uint64)) & 1
    ).astype(np.int8) - 1
    assert np.array_equal(active_rows, expected_rows)

    # P, product(Q_i), and product(R_i)=Y realize all eight candidate tuples
    # equally often.  This is stronger than pairwise independence.
    expected_candidate_ids = (
        (rules["P"] > 0).astype(np.int64)
        | ((rules["Q"] > 0).astype(np.int64) << 1)
        | ((rules["Y_code"] > 0).astype(np.int64) << 2)
    )
    assert np.array_equal(candidate_ids, expected_candidate_ids)
    assert np.array_equal(np.bincount(candidate_ids, minlength=8), np.full(8, len(batch) // 8))
    assert np.array_equal(rules["Y_code"], batch.y)
    for left, right in (("P", "Q"), ("P", "Y_code"), ("Q", "Y_code")):
        assert np.mean(rules[left] * rules[right]) == 0.0

    assert batch.metadata["raw_codeword_count"] == 64
    assert batch.metadata["repeats_per_codeword"] == 3
    assert batch.metadata["q_active_mask"] == (True, True, False)
    assert batch.metadata["r_active_mask"] == (True, True, True, False)
    assert batch.state is not None and batch.state.shape == (192, 2)
    assert batch.features(max_k=4).shape == (192, 2 + 4 + 1 + 3 + 2)


def test_control_truth_table_is_exactly_orthogonal_and_shared_across_splits() -> None:
    common = dict(k_q=2, k_y=3, control_seed=101, max_k_q=3, max_k_y=4)
    train = make_competing_factorial_dataset(
        n=128,
        seed=3,
        split="probe_train",
        id_offset=10_000,
        **common,
    )
    evaluation = make_competing_factorial_dataset(
        n=256,
        seed=97,
        split="probe_eval",
        id_offset=20_000,
        **common,
    )

    assert set(np.asarray(train.sample_id)).isdisjoint(set(np.asarray(evaluation.sample_id)))
    assert train.metadata["control_construction"] == "balanced_within_candidate_tuple"
    for batch in (train, evaluation):
        rules = decode_competing_rules(batch)
        control = np.asarray(batch.latents["truth_table_control"], dtype=np.int8)
        candidates = np.asarray(batch.latents["candidate_tuple_id"], dtype=np.int64)
        assert np.mean(control) == 0.0
        assert np.mean(control * rules["P"]) == 0.0
        assert np.mean(control * rules["Q"]) == 0.0
        assert np.mean(control * rules["Y_code"]) == 0.0
        # The ordinary construction is balanced conditional on the *entire*
        # candidate tuple, not only orthogonal to each coordinate.
        for candidate_id in range(8):
            assert np.sum(control[candidates == candidate_id]) == 0

    def truth_table(batch: object) -> dict[int, int]:
        raw_ids = np.asarray(batch.latents["raw_codeword_id"], dtype=np.int64)  # type: ignore[attr-defined]
        control = np.asarray(batch.latents["truth_table_control"], dtype=np.int8)  # type: ignore[attr-defined]
        return {int(raw_id): int(np.unique(control[raw_ids == raw_id]).item()) for raw_id in range(64)}

    # Split, observation seed, repeat count, and row order cannot change a
    # codeword's seeded control label.
    assert truth_table(train) == truth_table(evaluation)


def test_every_active_single_channel_flip_stays_on_factorial_support() -> None:
    batch = make_competing_factorial_dataset(
        repeats=1,
        k_q=2,
        k_y=2,
        seed=41,
        max_k_q=3,
        max_k_y=3,
    )
    raw_ids = np.asarray(batch.latents["raw_codeword_id"], dtype=np.int64)
    rows = _active_raw_rows(batch)
    row_by_id = {int(raw_id): rows[index] for index, raw_id in enumerate(raw_ids)}
    channel_order = tuple(batch.metadata["raw_channel_order"])

    intervention_by_channel = {
        "P": "flip_P",
        "Q_1": "flip_Q_1",
        "Q_2": "flip_Q_2",
        "R_1": "flip_Y_1",
        "R_2": "flip_Y_2",
    }
    interventions = competing_causal_flip_batches(batch)
    for bit, channel in enumerate(channel_order):
        changed = interventions[intervention_by_channel[channel]]
        changed_rows = _active_raw_rows(changed)
        for index, raw_id in enumerate(raw_ids):
            # Canonical IDs make the on-support counterpart explicit: flipping
            # active bit j is exactly XOR with 2^j.
            expected = row_by_id[int(raw_id) ^ (1 << bit)]
            assert np.array_equal(changed_rows[index], expected)


def test_minimal_three_bit_control_retains_exact_orthogonality() -> None:
    batch = make_competing_factorial_dataset(
        repeats=5,
        k_q=1,
        k_y=1,
        control_seed=7,
    )
    rules = decode_competing_rules(batch)
    control = np.asarray(batch.latents["truth_table_control"], dtype=np.int8)
    assert batch.metadata["control_construction"] == "minimal_walsh_interaction"
    assert np.mean(control) == 0.0
    assert np.mean(control * rules["P"]) == 0.0
    assert np.mean(control * rules["Q"]) == 0.0
    assert np.mean(control * rules["Y_code"]) == 0.0


def test_factorial_size_validation_accepts_total_or_repeats() -> None:
    by_total = make_competing_factorial_dataset(n=64, k_q=1, k_y=2)
    by_repeats = make_competing_factorial_dataset(repeats=4, k_q=1, k_y=2)
    consistent = make_competing_factorial_dataset(n=64, repeats=4, k_q=1, k_y=2)
    assert len(by_total) == len(by_repeats) == len(consistent) == 64

    with pytest.raises(ValueError, match="provide n, repeats"):
        make_competing_factorial_dataset(k_q=1, k_y=2)
    with pytest.raises(ValueError, match="multiple of the 16"):
        make_competing_factorial_dataset(n=48 + 1, k_q=1, k_y=2)
    with pytest.raises(ValueError, match="inconsistent"):
        make_competing_factorial_dataset(n=64, repeats=3, k_q=1, k_y=2)
    with pytest.raises(ValueError, match="repeats must be a positive integer"):
        make_competing_factorial_dataset(repeats=0, k_q=1, k_y=2)

