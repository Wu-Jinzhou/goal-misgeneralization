"""Protocol-level checks for identical-evidence order training."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from forkworld.config import load_config
from forkworld.evidence_order import make_identical_evidence_dataset
from forkworld.handoff import stable_state_digest
from forkworld.protocols import run_protocol
from forkworld.protocols_order import AtomicBatchSource, _normalized_auc
from forkworld.runner import smoke_config


def test_normalized_auc_uses_direct_piecewise_linear_observations() -> None:
    snapshots = {
        0: {"m_y": -0.5},
        1: {"m_y": 0.5},
        3: {"m_y": 1.0},
        5: {"m_y": -1.0},
    }
    # Through horizon 3: [0,1] contributes zero and [1,3] contributes 1.5.
    assert _normalized_auc(snapshots, horizon=3) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="endpoints"):
        _normalized_auc({1: snapshots[1], 3: snapshots[3]}, horizon=3)


def test_atomic_source_maps_phase_local_steps_exactly() -> None:
    config = smoke_config(
        load_config("configs/e18_identical_evidence_order.yaml")
    )
    batch = make_identical_evidence_dataset(
        256,
        q_p=0.75,
        q_q=0.875,
        k_q=2,
        k_y=2,
        seed=389,
        max_k_q=3,
        max_k_y=5,
        state_dim=2,
    )
    indices = np.asarray([[7, 3, 5, 1], [2, 4, 6, 8]], dtype=np.int64)
    source = AtomicBatchSource(batch, indices, config)

    x_first, y_first = source.sample_batch(4, 1)
    x_second, y_second = source.sample_batch(4, 2)
    expected_x, expected_y = source.x, source.y
    assert np.array_equal(x_first.numpy(), expected_x[indices[0]].numpy())
    assert np.array_equal(y_first.numpy(), expected_y[indices[0]].numpy())
    assert np.array_equal(x_second.numpy(), expected_x[indices[1]].numpy())
    assert np.array_equal(y_second.numpy(), expected_y[indices[1]].numpy())
    with pytest.raises(IndexError, match="step"):
        source.sample_batch(4, 0)
    with pytest.raises(ValueError, match="batch_size"):
        source.sample_batch(8, 1)


def test_smoke_schedules_share_prefix_multiset_and_real_empty_resets() -> None:
    base = smoke_config(load_config("configs/e18_identical_evidence_order.yaml"))
    base["run"]["device"] = "cpu"
    summaries = []
    stages = set()
    for schedule in ("b_then_d", "d_then_b", "interleave"):
        config = copy.deepcopy(base)
        config["h15"]["schedule"] = schedule
        result = run_protocol(config, seed=389)
        summaries.append(result.summary)
        stages.update(record["stage"] for record in result.metrics)

    assert len({item["hashes"]["initial_model"] for item in summaries}) == 1
    assert len({item["hashes"]["prefix_final_model"] for item in summaries}) == 1
    assert len(
        {stable_state_digest(item["prefix_snapshot"]) for item in summaries}
    ) == 1
    assert len({item["data"]["training_batch_digest"] for item in summaries}) == 1
    assert len({item["plan"]["row_exposure_digest"] for item in summaries}) == 1
    assert len(
        {item["plan"]["atomic_batch_multiset_digest"] for item in summaries}
    ) == 1
    assert len(
        {tuple(sorted(item["plan"]["component_digests"].items())) for item in summaries}
    ) == 1
    assert len({item["plan"]["ordered_digest"] for item in summaries}) == 3

    for summary in summaries:
        assert all(summary["replay"]["checks"].values())
        assert summary["resets"]["prefix_to_diagnostics"]["state_entry_count"] == 0
        assert summary["resets"]["diagnostics_to_washout"]["state_entry_count"] == 0
        assert summary["plan_audit"]["every_row_once_per_registered_repetition"]
        assert list(summary["diagnostic_snapshots"]) == ["0", "1", "2", "3"]
        assert list(summary["washout_snapshots"]) == [
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
        ]
        assert summary["training"]["total_steps"] == 24
        assert summary["training"]["all_minibatches_full"] is True

    assert {
        "prefix_behavior",
        "prefix_causal",
        "prefix_probe",
        "diagnostic_behavior",
        "diagnostic_causal",
        "diagnostic_probe",
        "washout_behavior",
        "washout_causal",
        "washout_probe",
    } <= stages
