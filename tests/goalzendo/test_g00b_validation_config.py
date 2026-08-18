from pathlib import Path

from goalzendo.config import get_path, load_config, validate_config
from goalzendo.runner import build_plan

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "goalzendo" / "g00b_deterministic_capability_validation.yaml"


def test_g00b_fresh_validation_is_frozen() -> None:
    config = load_config(CONFIG)
    assert validate_config(config) == []
    assert get_path(config, "experiment.status") == "prospective"
    assert get_path(config, "run.seeds") == [9101, 9102, 9103]
    assert get_path(config, "model.name") == "Qwen/Qwen2.5-1.5B-Instruct"
    assert get_path(config, "model.revision") == (
        "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    )
    assert get_path(config, "update.method") == "full"
    assert get_path(config, "train.algorithm") == "sft"
    assert get_path(config, "train.steps") == 512
    assert get_path(config, "train.batch_size") == 10
    assert get_path(config, "train.gradient_accumulation_steps") == 5
    assert get_path(config, "train.learning_rate") == 1e-5
    assert get_path(config, "train.warmup_ratio") == 6 / 512
    assert get_path(config, "train.deterministic_algorithms") is True
    assert get_path(config, "train.allow_tf32") is False
    assert get_path(config, "train.cublas_workspace_config") == ":4096:8"
    assert get_path(config, "run.checkpoint_steps") == [512]
    assert get_path(config, "run.snapshot_steps") == []


def test_g00b_repeats_the_complete_original_case_panel() -> None:
    config = load_config(CONFIG)
    plan = build_plan(config)
    assert len(plan) == 24
    assert {item.seed for item in plan} == {9101, 9102, 9103}
    cases = {
        (
            get_path(item.config, "data.rule_family"),
            get_path(item.config, "data.training_view"),
            tuple(get_path(item.config, "evaluation.prompt_views")),
        )
        for item in plan
    }
    assert cases == {
        ("parity", "law_only", ("law_only",)),
        ("majority", "law_only", ("law_only",)),
        ("parity", "audit_law_matched", ("audit_law_matched",)),
        ("majority", "audit_law_matched", ("audit_law_matched",)),
        ("parity", "sage_only", ("sage_only",)),
        ("parity", "herald_only", ("herald_only",)),
        ("parity", "no_signal", ("no_signal",)),
        ("parity", "surface_only", ("surface_only",)),
    }
