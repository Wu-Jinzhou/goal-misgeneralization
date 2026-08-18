from __future__ import annotations

from pathlib import Path

from goalzendo.config import expand_sweep, load_config
from goalzendo.runner import build_plan

ROOT = Path(__file__).resolve().parents[2]


def test_g00a2_preflight_is_two_exact_deterministic_replicas() -> None:
    config = load_config(
        ROOT / "configs" / "goalzendo" / "g00a2_deterministic_reproduction_preflight.yaml"
    )
    cells = expand_sweep(config)
    plan = build_plan(config)
    assert len(cells) == 2
    assert len(plan) == 2
    assert {cell["experiment"]["reproducibility_replica"] for cell in cells} == {"a", "b"}
    for cell in cells:
        assert cell["model"]["name"] == "Qwen/Qwen2.5-1.5B-Instruct"
        assert cell["update"]["method"] == "full"
        assert cell["train"]["deterministic_algorithms"] is True
        assert cell["train"]["allow_tf32"] is False
        assert cell["train"]["cublas_workspace_config"] == ":4096:8"
        assert cell["train"]["steps"] == 128
        assert cell["run"]["snapshot_steps"] == [128]
        assert cell["evaluation"]["prompt_views"] == ["audit_law_matched"]

    first, second = plan
    assert first.seeds == second.seeds
    assert first.plan_key != second.plan_key


def test_g00a2_horizon_is_unlocked_after_exact_preflight_and_paired_at_update_128() -> None:
    config = load_config(
        ROOT / "configs" / "goalzendo" / "g00a2_deterministic_matched_horizon.yaml"
    )
    cells = expand_sweep(config)
    assert len(cells) == 2
    assert len(build_plan(config)) == 6
    assert config["run"]["launch_guard"] is None
    assert config["run"]["protocol_unlocked"] is True
    by_arm = {cell["experiment"]["horizon_arm"]: cell for cell in cells}
    baseline = by_arm["baseline_128"]
    extended = by_arm["extended_512"]
    for cell in cells:
        assert cell["train"]["deterministic_algorithms"] is True
        assert cell["train"]["learning_rate"] == 0.00001
        assert cell["train"]["batch_size"] == 10
        assert cell["train"]["gradient_accumulation_steps"] == 4
    assert baseline["train"]["steps"] == 128
    assert baseline["run"]["snapshot_steps"] == [128]
    assert baseline["evaluation"]["prompt_views"] == ["audit_law_matched"]
    assert extended["train"]["steps"] == 512
    assert extended["run"]["snapshot_steps"] == [128, 256, 512]
    assert extended["evaluation"]["prompt_views"] == ["audit_law_matched", "law_only"]
    assert round(baseline["train"]["steps"] * baseline["train"]["warmup_ratio"]) == 6
    assert round(extended["train"]["steps"] * extended["train"]["warmup_ratio"]) == 6
