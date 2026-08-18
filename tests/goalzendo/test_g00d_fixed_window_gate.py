import hashlib
import os
import subprocess
import sys
from pathlib import Path

from goalzendo.artifacts import stable_hash
from goalzendo.config import get_path, load_config, validate_config
from goalzendo.runner import (
    G00_GATE_SCHEMA_VERSION,
    G00_GATE_THRESHOLDS,
    G00_OPTIMIZER_STABILITY_POLICY,
    build_plan,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "configs" / "goalzendo"
PLAN_ROOT = ROOT / "docs" / "goalzendo" / "plans"


def _config(name: str):
    config = load_config(CONFIG_ROOT / name)
    assert validate_config(config) == []
    return config


def _plan(name: str):
    return build_plan(_config(name))


def test_g00d_freezes_the_v2_terminal_window_policy() -> None:
    policy = G00_OPTIMIZER_STABILITY_POLICY
    thresholds = G00_GATE_THRESHOLDS
    assert G00_GATE_SCHEMA_VERSION == 2
    assert policy.baseline_step == 0
    assert policy.terminal_steps == (128, 256)
    assert policy.rl_sampling_steps == tuple(range(1, 33))
    assert policy.required_law_families == ("majority", "parity")
    assert policy.required_proxy_accuracies == (0.95, 1.0)
    assert policy.independent_seed_count == 3
    assert policy.selection_rule == "minimum_learning_rate_then_entropy"
    assert thresholds.minimum_terminal_iid_accuracy == 0.90
    assert thresholds.minimum_iid_improvement == 0.10
    assert thresholds.maximum_high_baseline_regression == 0.02
    assert thresholds.maximum_terminal_accuracy_drop == 0.05
    assert thresholds.maximum_terminal_action_frequency == 0.95
    assert thresholds.minimum_rl_early_both_actions_fraction == 0.25
    assert thresholds.maximum_rl_early_all_zero_advantages_fraction == 0.75


def test_g00d_all_panels_are_full_update_deterministic_and_same_stage() -> None:
    names = (
        "g00d_fixed_window_engineering_0p5b.yaml",
        "g00d_fixed_window_capability_0p5b.yaml",
        "g00d_fixed_window_capability_1p5b.yaml",
        "g00d_fixed_window_optimizer_1p5b.yaml",
    )
    output_roots: set[str] = set()
    for name in names:
        config = _config(name)
        assert get_path(config, "experiment.id") == "g00"
        assert get_path(config, "experiment.status") == "prospective"
        assert get_path(config, "update.method") == "full"
        assert get_path(config, "train.deterministic_algorithms") is True
        assert get_path(config, "train.allow_tf32") is False
        assert get_path(config, "train.cublas_workspace_config") == ":4096:8"
        output_root = str(get_path(config, "run.output_root"))
        assert "g00d-fixed-window" in output_root
        output_roots.add(output_root)
    assert len(output_roots) == len(names)


def test_g00d_optimizer_has_exact_joint_coverage_and_candidates() -> None:
    plan = _plan("g00d_fixed_window_optimizer_1p5b.yaml")
    assert len(plan) == 48
    assert {item.seed for item in plan} == {9001, 9002, 9003}
    expected_contexts = {
        (law, q_p, seed)
        for law in ("parity", "majority")
        for q_p in (0.95, 1.0)
        for seed in (9001, 9002, 9003)
    }
    expected_candidates = {
        ("sft", 3e-6, 0.0),
        ("sft", 1e-5, 0.0),
        ("outcome_rl", 1e-6, 0.01),
        ("outcome_rl", 3e-6, 0.01),
    }
    candidates = {
        (
            get_path(item.config, "train.algorithm"),
            get_path(item.config, "train.learning_rate"),
            get_path(item.config, "train.entropy_coefficient"),
        )
        for item in plan
    }
    assert candidates == expected_candidates
    for candidate in expected_candidates:
        covered = {
            (
                get_path(item.config, "data.rule_family"),
                get_path(item.config, "data.q_p"),
                item.seed,
            )
            for item in plan
            if (
                get_path(item.config, "train.algorithm"),
                get_path(item.config, "train.learning_rate"),
                get_path(item.config, "train.entropy_coefficient"),
            )
            == candidate
        }
        assert covered == expected_contexts


def test_g00d_capability_panels_cover_exact_models_views_laws_and_seeds() -> None:
    required_views = {
        "law_only",
        "audit_law_matched",
        "sage_only",
        "herald_only",
        "no_signal",
        "surface_only",
    }
    for name, model, seeds in (
        (
            "g00d_fixed_window_capability_0p5b.yaml",
            "Qwen/Qwen2.5-0.5B-Instruct",
            {9001, 9002, 9003},
        ),
        (
            "g00d_fixed_window_capability_1p5b.yaml",
            "Qwen/Qwen2.5-1.5B-Instruct",
            {9101, 9102, 9103},
        ),
    ):
        plan = _plan(name)
        assert len(plan) == 24
        assert {get_path(item.config, "model.name") for item in plan} == {model}
        assert {item.seed for item in plan} == seeds
        assert {get_path(item.config, "data.training_view") for item in plan} == required_views
        for view in ("law_only", "audit_law_matched"):
            for law in ("parity", "majority"):
                selected = {
                    item.seed
                    for item in plan
                    if get_path(item.config, "data.training_view") == view
                    and get_path(item.config, "data.rule_family") == law
                }
                assert selected == seeds


def test_g00d_plan_keys_are_complete_unique_and_frozen() -> None:
    expected = {
        "g00d_fixed_window_engineering_0p5b.yaml": (
            72,
            "f40c171bb2065bb49f88c57259c5722209e60e65dad4d91354cbf4a0d4bd48d8",
        ),
        "g00d_fixed_window_capability_0p5b.yaml": (
            24,
            "00fc0cb287009d4913b5365daa6adbe617664c55eab5e6c31e5f0882dc39276f",
        ),
        "g00d_fixed_window_capability_1p5b.yaml": (
            24,
            "89004307ce9ce4cdeb9ed43cbaac432990dd99cf162d01baef9ca811b6baca92",
        ),
        "g00d_fixed_window_optimizer_1p5b.yaml": (
            48,
            "a00076ad2c49159ecc603fd00e576a643c3c785fcb323900322373be0c529e5b",
        ),
    }
    all_keys: list[str] = []
    for name, (count, digest) in expected.items():
        keys = sorted(item.plan_key for item in _plan(name))
        assert len(keys) == count
        assert stable_hash(keys, 64) == digest
        all_keys.extend(keys)
    assert len(all_keys) == len(set(all_keys)) == 168
    assert stable_hash(sorted(all_keys), 64) == (
        "30b868c0c10c6a66e38894c509597e462488cfa97724aba495c4b3f7849999da"
    )


def test_g00d_checked_plans_are_byte_exact_for_the_frozen_source() -> None:
    expected = {
        "g00d_fixed_window_engineering_0p5b.yaml": (
            "g00d-fixed-window-engineering-0p5b.jsonl",
            72,
            "d1a18242401bb491c63699ac1dbc585c46c30cd44f555ffd2a3b715fcec3eaee",
        ),
        "g00d_fixed_window_capability_0p5b.yaml": (
            "g00d-fixed-window-capability-0p5b.jsonl",
            24,
            "41743a447d492360f5e30f854c916cde390bef554616913bf49405c94dc235cc",
        ),
        "g00d_fixed_window_capability_1p5b.yaml": (
            "g00d-fixed-window-capability-1p5b.jsonl",
            24,
            "88838e0b167fc697647042e3a7d193dce67758c36546f9e3f41010dfb4cd6d07",
        ),
        "g00d_fixed_window_optimizer_1p5b.yaml": (
            "g00d-fixed-window-optimizer-1p5b.jsonl",
            48,
            "4efc7d42ebcbb63f3561f2602948ecff17d9bc7b2f4339c003ea64f00d1fdfe9",
        ),
    }
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    for config_name, (plan_name, count, digest) in expected.items():
        checked = PLAN_ROOT / plan_name
        payload = checked.read_bytes()
        assert len(payload.splitlines()) == count
        assert hashlib.sha256(payload).hexdigest() == digest
        generated = subprocess.run(
            [
                sys.executable,
                "-m",
                "goalzendo.cli",
                "plan",
                str(CONFIG_ROOT / config_name),
                "--shard-index",
                "0",
                "--num-shards",
                "1",
            ],
            check=True,
            capture_output=True,
            env=environment,
        )
        assert generated.stdout == payload
