from __future__ import annotations

from pathlib import Path

from goalzendo.config import get_path, load_config, validate_config
from goalzendo.runner import build_plan

REPO = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO / "configs" / "goalzendo"
SEEDS = {9001, 9002, 9003}


def _load(name: str) -> dict[str, object]:
    config = load_config(CONFIG_DIR / name)
    assert validate_config(config) == []
    return config


def test_g00a_wave_a_is_the_frozen_six_run_cross_view_design() -> None:
    config = _load("g00a_matched_accessibility_wave_a.yaml")
    g01 = _load("g01_known_law.yaml")
    plan = build_plan(config)

    assert len(plan) == 6
    assert {spec.seed for spec in plan} == SEEDS
    assert get_path(config, "experiment.status") == "adaptive"
    assert get_path(config, "run.launch_guard") is None
    assert get_path(config, "update.method") == "full"
    assert get_path(config, "model.name") == get_path(g01, "model.name")
    assert get_path(config, "model.revision") == get_path(g01, "model.revision")
    assert get_path(config, "data.rule_family") == "parity"
    assert get_path(config, "train.algorithm") == "sft"
    assert get_path(config, "train.learning_rate") == 1e-5
    assert get_path(config, "train.batch_size") == 10
    assert get_path(config, "train.gradient_accumulation_steps") == 4
    assert get_path(config, "run.snapshot_steps") == []

    by_view = {
        str(get_path(spec.config, "data.training_view")): spec
        for spec in plan
        if spec.seed == 9001
    }
    assert set(by_view) == {"audit_law_matched", "law_only"}

    matched = by_view["audit_law_matched"].config
    assert get_path(matched, "train.steps") == 512
    assert get_path(matched, "train.warmup_ratio") == 6 / 512
    assert get_path(matched, "train.eval_steps") == [
        0,
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
    ]
    assert get_path(matched, "run.checkpoint_steps") == [512]
    assert get_path(matched, "evaluation.prompt_views") == [
        "audit_law_matched",
        "law_only",
    ]

    law_only = by_view["law_only"].config
    assert get_path(law_only, "train.steps") == 128
    assert get_path(law_only, "train.warmup_ratio") == 0.05
    assert get_path(law_only, "run.checkpoint_steps") == [128]
    assert get_path(law_only, "evaluation.prompt_views") == [
        "law_only",
        "audit_law_matched",
    ]


def test_g00a_wave_b_is_an_executable_but_fail_closed_three_run_contingency() -> None:
    config = _load("g00a_matched_accessibility_wave_b.yaml")
    plan = build_plan(config)

    assert len(plan) == 3
    assert {spec.seed for spec in plan} == SEEDS
    assert get_path(config, "experiment.status") == "adaptive"
    assert get_path(config, "run.launch_guard") == (
        "G00A_WAVE_B_REQUIRES_WAVE_A_MATCHED_NONPASS"
    )
    assert get_path(config, "run.protocol_unlocked") is False
    assert get_path(config, "update.method") == "full"
    assert get_path(config, "data.rule_family") == "parity"
    assert get_path(config, "data.training_view") == "audit_law_matched"
    assert get_path(config, "train.algorithm") == "sft"
    assert get_path(config, "train.learning_rate") == 3e-5
    assert get_path(config, "train.steps") == 512
    assert get_path(config, "train.warmup_ratio") == 6 / 512
    assert get_path(config, "run.checkpoint_steps") == [512]
    assert get_path(config, "run.snapshot_steps") == []
    assert get_path(config, "evaluation.prompt_views") == [
        "audit_law_matched",
        "law_only",
    ]
    assert all(
        get_path(spec.config, "run.launch_guard")
        == "G00A_WAVE_B_REQUIRES_WAVE_A_MATCHED_NONPASS"
        for spec in plan
    )
