from __future__ import annotations

from pathlib import Path

from goalzendo.config import expand_sweep, load_config
from goalzendo.runner import build_plan

ROOT = Path(__file__).resolve().parents[2]


def test_g00a3_enumerated_gradient_panel_is_frozen_and_complete() -> None:
    config = load_config(
        ROOT / "configs" / "goalzendo" / "g00a3_enumerated_outcome_gradient.yaml"
    )
    cells = expand_sweep(config)
    plan = build_plan(config)
    assert len(cells) == 12
    assert len(plan) == 36
    assert config["experiment"] == {
        "id": "g00a3",
        "name": "enumerated_outcome_gradient",
        "status": "adaptive",
    }
    assert config["run"]["seeds"] == [9301, 9323, 9341]
    assert config["run"]["protocol_unlocked"] is True
    assert config["run"]["launch_guard"] is None
    assert config["run"]["snapshot_steps"] == [512]
    assert {cell["data"]["q_p"] for cell in cells} == {0.5, 0.8, 0.95, 1.0}
    assert {cell["data"]["rule_family"] for cell in cells} == {"parity"}
    assert {cell["train"]["algorithm"] for cell in cells} == {
        "sft",
        "outcome_rl",
        "expected_outcome_rl",
    }
    settings = {
        cell["train"]["algorithm"]: (
            cell["train"]["learning_rate"],
            cell["train"]["entropy_coefficient"],
        )
        for cell in cells
    }
    assert settings == {
        "sft": (1e-5, 0.0),
        "outcome_rl": (3e-6, 0.01),
        "expected_outcome_rl": (3e-6, 0.01),
    }
    for cell in cells:
        assert cell["model"]["name"] == "Qwen/Qwen2.5-0.5B-Instruct"
        assert cell["update"]["method"] == "full"
        assert cell["train"]["steps"] == 512
        assert cell["train"]["batch_size"] == 10
        assert cell["train"]["gradient_accumulation_steps"] == 5
        assert round(cell["train"]["steps"] * cell["train"]["warmup_ratio"]) == 6
        assert cell["train"]["deterministic_algorithms"] is True
        assert cell["train"]["allow_tf32"] is False
        assert cell["evaluation"]["prompt_views"] == [
            "full",
            "audit_law_matched",
            "law_only",
        ]


def test_g00a3_expected_outcome_smoke_is_one_full_update_run() -> None:
    config = load_config(
        ROOT / "configs" / "goalzendo" / "g00a3_expected_outcome_smoke.yaml"
    )
    plan = build_plan(config)
    assert len(plan) == 1
    assert config["train"]["algorithm"] == "expected_outcome_rl"
    assert config["train"]["steps"] == 2
    assert config["train"]["deterministic_algorithms"] is True
    assert config["update"]["method"] == "full"
    assert config["run"]["snapshot_steps"] == [2]
