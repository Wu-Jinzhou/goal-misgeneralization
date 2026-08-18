from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

from forkworld.analysis import _h8_primary_eligibility
from forkworld.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    SOURCE_FINGERPRINT_SCHEMA_VERSION,
    RunStore,
    discover_runs,
    implementation_provenance,
    write_json,
)
from forkworld.config import ConfigError, DEFAULT_CONFIG, deep_merge, expand_sweep, validate_config
from forkworld.metrics import (
    acquisition_time,
    behavioral_metrics,
    context_selector_metrics,
    intervention_metrics,
    reliance_half_life,
    replacement_time,
    sequential_replacement_time,
)


def test_sweep_expansion_is_deterministic() -> None:
    config = deep_merge(
        DEFAULT_CONFIG,
        {"sweep": {"data.q": [0.8, 0.9], "model.width": [8, 16]}},
    )
    cells = expand_sweep(config)
    assert len(cells) == 4
    assert [cell["_sweep_values"] for cell in cells] == [
        {"data.q": 0.8, "model.width": 8},
        {"data.q": 0.8, "model.width": 16},
        {"data.q": 0.9, "model.width": 8},
        {"data.q": 0.9, "model.width": 16},
    ]


def test_h4_rejects_inconsistent_q_count_triplet() -> None:
    config = deep_merge(
        DEFAULT_CONFIG,
        {
            "experiment": {"hypothesis": "h4"},
            "data": {"n_train": 1000},
            "h4": {"n_conflict": 100, "q": 0.8},
        },
    )
    with pytest.raises(ConfigError, match="inconsistent"):
        validate_config(config)


def test_h2_parameter_match_tolerance_is_declared_and_validated() -> None:
    config = deep_merge(
        DEFAULT_CONFIG,
        {
            "experiment": {"hypothesis": "h2"},
            "h2": {"parameter_match_tolerance": 1.1},
        },
    )
    with pytest.raises(ConfigError, match="parameter_match_tolerance"):
        validate_config(config)


def test_h4_failure_mechanism_families_must_be_known_and_disjoint() -> None:
    base = deep_merge(
        DEFAULT_CONFIG,
        {
            "experiment": {"hypothesis": "h4"},
            "h4": {
                "structured_train_types": ["location_reflection"],
                "structured_test_types": ["location_reflection"],
            },
        },
    )
    with pytest.raises(ConfigError, match="must be disjoint"):
        validate_config(base)
    base["h4"]["structured_test_types"] = ["unknown_corruption"]
    with pytest.raises(ConfigError, match="unknown failure mechanism"):
        validate_config(base)


def test_h5_on_policy_rollout_depth_is_bounded() -> None:
    config = deep_merge(
        DEFAULT_CONFIG,
        {
            "experiment": {"hypothesis": "h5"},
            "h5": {
                "nuisance_bits": 2,
                "nuisance_entropy": 1,
                "on_policy_rollout_depth": 17,
            },
        },
    )
    with pytest.raises(ConfigError, match="on_policy_rollout_depth"):
        validate_config(config)


def test_minimum_inferential_seed_floor_cannot_be_weakened() -> None:
    config = deep_merge(
        DEFAULT_CONFIG,
        {"evaluation": {"minimum_inferential_seeds": 2}},
    )
    with pytest.raises(ConfigError, match="minimum_inferential_seeds"):
        validate_config(config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("equivalence_margin", -0.01),
        ("equivalence_margin", float("inf")),
        ("bootstrap_samples", 0),
        ("bootstrap_samples", 2.5),
        ("confidence", 0.0),
        ("confidence", 1.0),
        ("confidence", float("nan")),
    ],
)
def test_inference_settings_are_validated(field: str, value: object) -> None:
    config = deep_merge(DEFAULT_CONFIG, {"evaluation": {field: value}})

    with pytest.raises(ConfigError, match=field):
        validate_config(config)


def test_h6_terminal_reward_equivalence_is_recorded() -> None:
    config = deep_merge(
        DEFAULT_CONFIG,
        {
            "experiment": {"hypothesis": "h6"},
            "data": {"n_train": 10_240},
            "h6": {
                "locations": ["reward"],
                "structures": ["step", "episode"],
                "reward_mode": "terminal",
            },
        },
    )
    cautions = validate_config(config)
    assert any("equivalence control" in caution for caution in cautions)


def test_h6_rejects_inexact_recurrence_and_on_policy_exposure_aliases() -> None:
    inexact = deep_merge(
        DEFAULT_CONFIG,
        {
            "experiment": {"hypothesis": "h6"},
            "data": {"n_train": 2_560},
            "h6": {"visits_per_state": 3},
        },
    )
    with pytest.raises(ConfigError, match="divisible by visits_per_state"):
        validate_config(inexact)

    mismatched_collection = deep_merge(
        DEFAULT_CONFIG,
        {
            "experiment": {"hypothesis": "h6"},
            "data": {"n_train": 2_560},
            "h6": {
                "algorithm": "on_policy_imitation",
                "collection_batch_size": 64,
                "visits_per_state": 8,
            },
        },
    )
    with pytest.raises(ConfigError, match="collection_batch_size"):
        validate_config(mismatched_collection)


def test_h8_analysis_keeps_fixed_volume_mastery_that_overshoots_match_band() -> None:
    frame = pd.DataFrame(
        [
            {
                "config.h8.stage1_mode": "fixed",
                "final.stage1_gate_passed": True,
                "final.stage1_in_match_band": False,
                # Simulate an artifact produced before eligibility was repaired.
                "final.eligible_for_primary_analysis": False,
            },
            {
                "config.h8.stage1_mode": "behavior_matched",
                "final.stage1_gate_passed": True,
                "final.stage1_in_match_band": False,
                "final.eligible_for_primary_analysis": False,
            },
        ]
    )
    assert _h8_primary_eligibility(frame).tolist() == [True, False]


def test_h9_proxy_degrees_must_be_positive_integers() -> None:
    config = deep_merge(
        DEFAULT_CONFIG,
        {
            "experiment": {"hypothesis": "h9"},
            "h9": {"proxy0_degree": 0, "proxy1_degree": 2},
        },
    )
    with pytest.raises(ConfigError, match="proxy0_degree"):
        validate_config(config)


def test_conflict_reliance_and_intervention_metrics() -> None:
    y = np.array([-1, 1, -1, 1], dtype=np.int8)
    p = -y
    logits = np.array([[4.0, -2.0], [-2.0, 4.0], [3.0, 0.0], [0.0, 3.0]])
    result = behavioral_metrics(logits, y, p)
    assert result["rho_y"] == 1.0
    assert result["rho_p"] == 0.0
    assert result["delta_rho"] == 1.0
    changed = logits[:, ::-1]
    effect = intervention_metrics(logits, changed, y)
    assert effect["hard_flip_rate"] == 1.0
    assert effect["probability_ate"] < 0


def test_censor_aware_events() -> None:
    steps = [0, 1, 2, 4, 8]
    proxy = [0.5, 0.91, 0.93, 0.7, 0.4]
    intended = [0.5, 0.1, 0.2, 0.7, 0.95]
    assert acquisition_time(steps, proxy, persistence=2).time == 1
    assert acquisition_time(steps, intended, persistence=2).observed is False
    assert replacement_time(steps, intended, proxy, persistence=1).time == 8
    half = reliance_half_life(steps, [0.9, 0.8, 0.7, 0.6, 0.55], endpoint=0.5, adjusted=True)
    assert half.observed
    assert half.time == 2


def test_sequential_replacement_requires_prior_sustained_proxy_acquisition() -> None:
    steps = [0, 1, 2, 3, 4, 5, 6]
    proxy = [0.1, 0.2, 0.91, 0.93, 0.4, 0.2, 0.1]
    intended = [0.8, 0.7, 0.1, 0.1, 0.6, 0.8, 0.9]

    assert replacement_time(steps, intended, proxy, persistence=2).time == 0
    sequential = sequential_replacement_time(
        steps,
        intended,
        proxy,
        acquisition_threshold=0.9,
        persistence=2,
    )
    assert sequential.observed is True
    assert sequential.time == 4

    never_acquired = sequential_replacement_time(
        steps,
        intended,
        [0.1] * len(steps),
        acquisition_threshold=0.9,
        persistence=2,
    )
    assert never_acquired.observed is False
    assert never_acquired.time == steps[-1]


def test_context_selector_metrics() -> None:
    p0 = np.array([-1, 1, -1, 1])
    p1 = -p0
    result = context_selector_metrics(p0, p1, p0, p1)
    assert result["strict_context_switching"] == 1.0
    assert result["gating_index"] == 1.0


def test_run_store_is_stable_and_complete(tmp_path: Path) -> None:
    config = deep_merge(DEFAULT_CONFIG, {"experiment": {"name": "test"}})
    first = RunStore(tmp_path, config, 3, tmp_path)
    second = RunStore(tmp_path, config, 3, tmp_path)
    assert first.run_id == second.run_id
    first.initialize()
    first.append_metrics({"metric": "rho_y", "value": 0.75})
    first.finalize({"final": {"rho_y": 0.75}})
    assert first.complete
    assert discover_runs(tmp_path) == [first.path]
    payload = json.loads((first.path / "summary.json").read_text())
    resolved = yaml.safe_load((first.path / "resolved_config.yaml").read_text())
    assert payload["final"]["rho_y"] == 0.75
    assert resolved["seed"] == 3


def test_run_store_identity_ignores_seed_scheduling_and_storage_policy(tmp_path: Path) -> None:
    base = deep_merge(DEFAULT_CONFIG, {"experiment": {"name": "identity"}})
    changed = deep_merge(
        base,
        {
            "run": {
                "output_root": "/a/different/artifact/root",
                "resume": False,
                "seeds": [3, 5, 8, 13],
            }
        },
    )
    assert RunStore(tmp_path, base, 3, tmp_path).run_id == RunStore(
        tmp_path, changed, 3, tmp_path
    ).run_id

    changed_device = deep_merge(base, {"run": {"device": "cuda:0"}})
    assert RunStore(tmp_path, base, 3, tmp_path).run_id != RunStore(
        tmp_path, changed_device, 3, tmp_path
    ).run_id


def test_run_store_identity_and_metadata_cover_source_provenance(tmp_path: Path) -> None:
    source = tmp_path / "src" / "forkworld"
    source.mkdir(parents=True)
    implementation = source / "protocol.py"
    implementation.write_text("RESULT = 1\n", encoding="utf-8")
    config = deep_merge(DEFAULT_CONFIG, {"experiment": {"name": "provenance"}})

    first = RunStore(tmp_path / "artifacts", config, 3, tmp_path)
    first.initialize()
    metadata = json.loads((first.path / "metadata.json").read_text(encoding="utf-8"))
    recorded = metadata["implementation"]
    assert recorded["artifact_schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert (
        recorded["source_fingerprint_schema_version"]
        == SOURCE_FINGERPRINT_SCHEMA_VERSION
    )
    assert recorded["implementation_fingerprint"] == first.implementation[
        "implementation_fingerprint"
    ]

    # File mtimes and artifact locations are not scientific identity inputs.
    stat = implementation.stat()
    os.utime(
        implementation,
        ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000),
    )
    relocated = RunStore(tmp_path / "elsewhere", config, 3, tmp_path)
    assert relocated.run_id == first.run_id

    # A byte-level implementation change gets a new run directory, so the old
    # COMPLETE marker cannot silently suppress execution of the new code.
    implementation.write_text("RESULT = 2\n", encoding="utf-8")
    changed = RunStore(tmp_path / "artifacts", config, 3, tmp_path)
    assert changed.run_id != first.run_id
    assert not changed.complete


def test_source_fingerprint_cache_uses_stat_signature(tmp_path: Path) -> None:
    source = tmp_path / "src" / "forkworld"
    source.mkdir(parents=True)
    implementation = source / "model.py"
    implementation.write_text("VALUE = 1\n", encoding="utf-8")
    original_read_bytes = Path.read_bytes
    reads = 0

    def counted_read_bytes(path: Path) -> bytes:
        nonlocal reads
        reads += 1
        return original_read_bytes(path)

    with patch.object(Path, "read_bytes", counted_read_bytes):
        first = implementation_provenance(tmp_path)
        repeated = implementation_provenance(tmp_path)
        assert repeated == first
        assert reads == 1

        # A changed mtime invalidates the cheap cache signature but not the
        # content-derived identity. A byte change invalidates both.
        stat = implementation.stat()
        os.utime(
            implementation,
            ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000),
        )
        touched = implementation_provenance(tmp_path)
        assert touched["implementation_fingerprint"] == first["implementation_fingerprint"]
        assert reads == 2
        implementation.write_text("VALUE = 2\n", encoding="utf-8")
        changed = implementation_provenance(tmp_path)
        assert changed["implementation_fingerprint"] != first["implementation_fingerprint"]
        assert reads == 3


def test_json_artifacts_are_strict_and_replace_nonfinite_values(tmp_path: Path) -> None:
    target = tmp_path / "strict.json"
    write_json(target, {"nan": float("nan"), "positive_infinity": float("inf")})
    raw = target.read_text(encoding="utf-8")
    assert "NaN" not in raw and "Infinity" not in raw
    assert json.loads(raw) == {"nan": None, "positive_infinity": None}


def test_restart_clears_partial_jsonl_artifacts(tmp_path: Path) -> None:
    config = deep_merge(DEFAULT_CONFIG, {"experiment": {"name": "interrupted"}})
    store = RunStore(tmp_path, config, 5, tmp_path)
    store.initialize()
    store.append_metrics({"metric": "stale", "value": 1.0})
    store.append_predictions([{"sample_id": 1, "prediction": -1}])

    store.initialize(reset=True)
    assert not store.metrics_path.exists()
    assert not store.predictions_path.exists()
