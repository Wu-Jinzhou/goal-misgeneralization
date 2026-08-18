"""Causal and replay invariants for the adaptive winner-knockout study."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from forkworld.handoff import (
    audit_compute_sham,
    audit_handoff_phase_b,
    make_compute_sham,
    make_handoff_phase_b,
    normalized_control_auc,
    persistent_handoff,
    pure_control,
    stable_state_digest,
    static_sampler_digest,
)


def test_phase_b_is_paired_and_p_balanced_in_every_qr_codeword() -> None:
    batch = make_handoff_phase_b(
        10_000,
        q_q=0.90,
        k_q=2,
        k_y=3,
        seed=157,
        max_k_q=3,
        max_k_y=5,
        state_dim=8,
    )
    audit = audit_handoff_phase_b(batch, expected_q_q=0.90)

    assert len(batch) == 10_000
    assert audit["agreements"] == {"p_y": 0.5, "p_q": 0.5, "q_y": 0.9}
    assert audit["raw_codeword_count"] == 32
    assert audit["paired_clone_count"] == 5_000
    assert all(
        counts["negative"] == counts["positive"] > 0
        for counts in audit["raw_codeword_p_counts"].values()
    )
    assert all(count > 0 for count in audit["candidate_tuple_counts"].values())

    pairs = np.asarray(batch.latents["handoff_pair_id"])
    p_column = batch.feature_names(max_k=5).index("P")
    features = batch.features(max_k=5)
    for pair_id in (0, 1, 123, 4_999):
        positions = np.flatnonzero(pairs == pair_id)
        assert len(positions) == 2
        assert np.array_equal(
            np.delete(features[positions[0]], p_column),
            np.delete(features[positions[1]], p_column),
        )
        assert sorted(features[positions, p_column].tolist()) == [-1.0, 1.0]


def test_compute_sham_has_exactly_ten_thousand_null_rows() -> None:
    batch = make_compute_sham(
        n=10_000,
        repeats=79,
        k_q=2,
        k_y=3,
        seed=157,
        control_seed=101,
        max_k_q=3,
        max_k_y=5,
        state_dim=8,
    )
    audit = audit_compute_sham(batch)

    assert len(batch) == 10_000
    assert audit["paired_clone_count"] == 5_000
    assert audit["raw_codeword_count"] == 64
    assert min(audit["raw_codeword_counts"].values()) == 156
    assert max(audit["raw_codeword_counts"].values()) == 158
    assert audit["candidate_agreements"] == {"P": 0.5, "Q": 0.5, "Y": 0.5}
    assert "repeats_per_codeword" not in batch.metadata

    pairs = np.asarray(batch.latents["sham_pair_id"])
    features = batch.features(max_k=5)
    labels = np.asarray(batch.target)
    for pair_id in (0, 1, 123, 4_999):
        positions = np.flatnonzero(pairs == pair_id)
        assert np.array_equal(features[positions[0]], features[positions[1]])
        assert sorted(labels[positions].tolist()) == [-1, 1]


def test_smoke_scale_sham_preserves_all_raw_codewords() -> None:
    batch = make_compute_sham(
        n=256,
        repeats=4,
        k_q=2,
        k_y=2,
        seed=3,
        control_seed=11,
        max_k_q=3,
        max_k_y=5,
        state_dim=2,
    )
    audit = audit_compute_sham(batch)
    assert audit["paired_clone_count"] == 128
    assert audit["raw_codeword_count"] == 32
    assert set(audit["raw_codeword_counts"].values()) == {8}


def test_state_and_sampler_hashes_are_deterministic_and_sensitive() -> None:
    model = torch.nn.Linear(3, 1)
    first = stable_state_digest(model.state_dict())
    assert first == stable_state_digest(model.state_dict())
    with torch.no_grad():
        model.weight[0, 0] += 1
    assert first != stable_state_digest(model.state_dict())

    sampler = static_sampler_digest(10_000, batch_size=250, steps=45, seed=157)
    assert sampler == static_sampler_digest(
        10_000, batch_size=250, steps=45, seed=157
    )
    assert sampler != static_sampler_digest(
        10_000, batch_size=250, steps=45, seed=163
    )


def _snapshot(p: float, q: float, y: float) -> dict[str, dict[str, float]]:
    return {
        "behavior": {"P": p, "Q": q, "Y": y},
        "causal": {"P": p, "Q": q, "Y": y},
    }


def test_frozen_eligibility_event_and_direct_auc_definitions() -> None:
    assert pure_control(
        _snapshot(0.95, 0.80, 0.50)["behavior"],
        _snapshot(0.95, 0.80, 0.50)["causal"],
        "P",
    )
    assert not pure_control(
        _snapshot(0.95, 0.86, 0.50)["behavior"],
        _snapshot(0.95, 0.86, 0.50)["causal"],
        "P",
    )

    snapshots = {
        0: _snapshot(1.0, 0.5, 0.5),
        64: _snapshot(0.5, 0.95, 0.5),
        128: _snapshot(0.5, 1.0, 0.5),
        161: _snapshot(0.5, 1.0, 0.5),
    }
    event = persistent_handoff(snapshots, "Q")
    assert event["observed"] is True
    assert event["first_qualifying_step"] == 64
    assert event["confirmation_step"] == 128
    assert normalized_control_auc(snapshots, "Q", horizon=128) == pytest.approx(
        0.85
    )
    with pytest.raises(ValueError, match="span"):
        normalized_control_auc({0: snapshots[0], 117: snapshots[64]}, "Q", horizon=128)
