from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

from goalzendo.artifacts import (
    SOURCE_FINGERPRINT_SCHEMA_VERSION,
    RunStore,
    _identity_config,
    stable_hash,
)
from goalzendo.config import ConfigError, deep_merge, expand_sweep, get_path, load_config
from goalzendo.runner import build_plan

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "goalzendo" / "g00a4_reward_gradient_shape.yaml"
FROZEN_PLAN = ROOT / "docs" / "goalzendo" / "plans" / "g00a4-reward-gradient-shape.jsonl"
SMOKE_CONFIG = ROOT / "configs" / "goalzendo" / "g00a4_full_model_smoke.yaml"
FROZEN_SOURCE_FINGERPRINT = (
    "1dacbdff006e1d30e31bc263781531627f62d60b5623f12967f1a13a9ef97801"
)
FROZEN_ARTIFACT_SCHEMA_VERSION = 1
SEED_BLOCK_CONFIGS = {
    seed: ROOT
    / "configs"
    / "goalzendo"
    / f"g00a4_reward_gradient_shape_seed{seed}.yaml"
    for seed in (9401, 9427, 9461)
}


def test_g00a4_gradient_shape_panel_is_frozen_and_complete() -> None:
    config = load_config(CONFIG)
    cells = expand_sweep(config)
    plan = build_plan(config)
    assert config["experiment"] == {
        "id": "g00a4",
        "name": "reward_gradient_shape",
        "status": "adaptive",
    }
    assert config["run"]["seeds"] == [9401, 9427, 9461]
    assert config["run"]["protocol_unlocked"] is True
    assert config["run"]["launch_guard"] is None
    assert config["run"]["checkpoint_steps"] == [256, 512]
    assert config["run"]["snapshot_steps"] == []
    assert len(cells) == 4
    assert len(plan) == 12
    assert len({item.plan_key for item in plan}) == 12
    assert {item.seed for item in plan} == {9401, 9427, 9461}

    settings = {
        get_path(cell, "train.algorithm"): (
            get_path(cell, "train.learning_rate"),
            get_path(cell, "train.entropy_coefficient"),
            get_path(cell, "train.reward_gradient_exponent"),
        )
        for cell in cells
    }
    assert settings == {
        "sft": (1e-5, 0.0, None),
        "expected_outcome_rl": (3e-6, 0.01, None),
        "tempered_outcome_control": (3e-6, 0.01, 0.5),
        "logprob_outcome_control": (3e-6, 0.01, 0.0),
    }
    for cell in cells:
        assert get_path(cell, "data.q_p") == 0.5
        assert get_path(cell, "data.q_q") == 0.9
        assert get_path(cell, "data.rule_family") == "parity"
        assert get_path(cell, "data.training_view") == "full"
        assert get_path(cell, "model.name") == "Qwen/Qwen2.5-0.5B-Instruct"
        assert get_path(cell, "model.revision") == (
            "7ae557604adf67be50417f59c2c2f167def9a775"
        )
        assert get_path(cell, "update.method") == "full"
        assert get_path(cell, "train.steps") == 512
        assert get_path(cell, "train.batch_size") == 10
        assert get_path(cell, "train.gradient_accumulation_steps") == 5
        assert round(get_path(cell, "train.steps") * get_path(cell, "train.warmup_ratio")) == 6
        assert get_path(cell, "train.deterministic_algorithms") is True
        assert get_path(cell, "train.allow_tf32") is False
        assert get_path(cell, "train.cublas_workspace_config") == ":4096:8"
        assert get_path(cell, "evaluation.prompt_views") == [
            "full",
            "audit_law_matched",
            "law_only",
        ]


def test_g00a4_is_paired_on_every_derived_randomness_lane() -> None:
    plan = build_plan(load_config(CONFIG))
    by_seed: dict[int, list[object]] = defaultdict(list)
    for item in plan:
        by_seed[item.seed].append(item)
    assert set(by_seed) == {9401, 9427, 9461}
    for items in by_seed.values():
        assert len(items) == 4
        assert len({tuple(sorted(item.seeds.items())) for item in items}) == 1  # type: ignore[attr-defined]
        assert {
            get_path(item.config, "train.algorithm")  # type: ignore[attr-defined]
            for item in items
        } == {
            "sft",
            "expected_outcome_rl",
            "tempered_outcome_control",
            "logprob_outcome_control",
        }


def test_g00a4_operational_seed_blocks_partition_the_frozen_plan() -> None:
    full_plan = build_plan(load_config(CONFIG))
    full_by_key = {item.plan_key: item for item in full_plan}
    observed_keys: set[str] = set()

    for seed, path in SEED_BLOCK_CONFIGS.items():
        config = load_config(path)
        block = build_plan(config)
        assert config["run"]["seeds"] == [seed]
        assert len(block) == 4
        assert {item.seed for item in block} == {seed}
        assert {
            get_path(item.config, "train.algorithm") for item in block
        } == {
            "sft",
            "expected_outcome_rl",
            "tempered_outcome_control",
            "logprob_outcome_control",
        }
        for item in block:
            assert item.plan_key in full_by_key
            expected = full_by_key[item.plan_key]
            assert item.cell_id == expected.cell_id
            assert item.seed == expected.seed
            assert item.seeds == expected.seeds
            block_store = RunStore(
                Path(get_path(item.config, "run.output_root")),
                item.config,
                item.seed,
                ROOT,
            )
            expected_store = RunStore(
                Path(get_path(expected.config, "run.output_root")),
                expected.config,
                expected.seed,
                ROOT,
            )
            assert block_store.run_id == expected_store.run_id
            observed_keys.add(item.plan_key)

    assert observed_keys == set(full_by_key)


def test_g00a4_frozen_plan_matches_all_logical_and_artifact_identities() -> None:
    config = load_config(CONFIG)
    plan = build_plan(config)
    frozen = [
        json.loads(line)
        for line in FROZEN_PLAN.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(frozen) == len(plan) == 12
    for row, spec in zip(frozen, plan, strict=True):
        frozen_identity = {
            "config": _identity_config(spec.config),
            "seed": spec.seed,
            "artifact_schema_version": FROZEN_ARTIFACT_SCHEMA_VERSION,
            "source_fingerprint_schema_version": SOURCE_FINGERPRINT_SCHEMA_VERSION,
            "implementation_fingerprint": FROZEN_SOURCE_FINGERPRINT,
        }
        assert row["global_index"] == spec.global_index
        assert row["cell_index"] == spec.cell_index
        assert row["cell_id"] == spec.cell_id
        assert row["plan_key"] == spec.plan_key
        assert row["seed"] == spec.seed
        assert row["derived_seeds"] == dict(spec.seeds)
        assert row["sweep_values"] == spec.config["_sweep_values"]
        expected_run_id = stable_hash(frozen_identity, 20)
        assert row["run_id"] == expected_run_id
        assert Path(row["path"]).name == expected_run_id


def test_g00a4_full_model_smoke_covers_both_new_objectives() -> None:
    config = load_config(SMOKE_CONFIG)
    plan = build_plan(config)
    assert len(plan) == 2
    assert config["run"]["seeds"] == [9499]
    assert config["run"]["checkpoint_steps"] == [1, 2]
    assert config["run"]["snapshot_steps"] == [2]
    assert config["model"]["name"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert config["model"]["revision"] == (
        "7ae557604adf67be50417f59c2c2f167def9a775"
    )
    assert config["update"]["method"] == "full"
    assert config["train"]["steps"] == 2
    assert config["train"]["batch_size"] == 10
    assert config["train"]["gradient_accumulation_steps"] == 2
    assert config["train"]["deterministic_algorithms"] is True
    assert config["train"]["allow_tf32"] is False
    assert {
        (
            get_path(item.config, "train.algorithm"),
            get_path(item.config, "train.reward_gradient_exponent"),
        )
        for item in plan
    } == {
        ("tempered_outcome_control", 0.5),
        ("logprob_outcome_control", 0.0),
    }
    assert len({tuple(sorted(item.seeds.items())) for item in plan}) == 1


@pytest.mark.parametrize(
    ("algorithm", "exponent", "message"),
    [
        ("tempered_outcome_control", None, "strictly in"),
        ("tempered_outcome_control", 0.0, "strictly in"),
        ("tempered_outcome_control", 1.0, "strictly in"),
        ("logprob_outcome_control", None, "requires"),
        ("logprob_outcome_control", 0.5, "requires"),
        ("expected_outcome_rl", 0.5, "only valid"),
        ("sft", 0.0, "only valid"),
    ],
)
def test_power_gradient_algorithm_configuration_fails_closed(
    algorithm: str,
    exponent: float | None,
    message: str,
) -> None:
    base = load_config(ROOT / "configs" / "goalzendo" / "base.yaml")
    invalid = deep_merge(
        base,
        {
            "train": {
                "algorithm": algorithm,
                "reward_gradient_exponent": exponent,
            }
        },
    )
    from goalzendo.config import validate_config

    with pytest.raises(ConfigError, match=message):
        validate_config(invalid)
