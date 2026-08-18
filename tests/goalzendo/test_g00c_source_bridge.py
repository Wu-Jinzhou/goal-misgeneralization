import hashlib
import json
from pathlib import Path

from goalzendo.artifacts import stable_hash
from goalzendo.config import get_path, load_config, validate_config
from goalzendo.runner import build_plan

ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "configs" / "goalzendo"


def _config(name: str):
    config = load_config(CONFIG_ROOT / name)
    assert validate_config(config) == []
    return config


def _assert_strict_full_update(config) -> None:
    assert get_path(config, "update.method") == "full"
    assert get_path(config, "train.deterministic_algorithms") is True
    assert get_path(config, "train.allow_tf32") is False
    assert get_path(config, "train.cublas_workspace_config") == ":4096:8"


def test_g00c_engineering_repeats_the_complete_original_grid() -> None:
    config = _config("g00c_strict_engineering_0p5b.yaml")
    _assert_strict_full_update(config)
    plan = build_plan(config)
    assert len(plan) == 72
    assert {item.seed for item in plan} == {9001, 9002, 9003}
    assert {get_path(item.config, "data.rule_family") for item in plan} == {
        "parity",
        "majority",
    }
    assert {get_path(item.config, "data.q_p") for item in plan} == {0.95, 1.0}
    assert {
        (
            get_path(item.config, "train.algorithm"),
            get_path(item.config, "train.learning_rate"),
        )
        for item in plan
    } == {
        ("sft", 3e-6),
        ("sft", 1e-5),
        ("sft", 3e-5),
        ("outcome_rl", 1e-6),
        ("outcome_rl", 3e-6),
        ("outcome_rl", 1e-5),
    }


def test_g00c_0p5b_capability_covers_every_required_view() -> None:
    config = _config("g00c_strict_capability_0p5b.yaml")
    _assert_strict_full_update(config)
    plan = build_plan(config)
    assert len(plan) == 24
    assert get_path(config, "train.steps") == 512
    assert get_path(config, "train.batch_size") == 10
    assert get_path(config, "train.gradient_accumulation_steps") == 5
    assert get_path(config, "train.warmup_ratio") == 6 / 512
    cases = {
        (
            get_path(item.config, "data.rule_family"),
            get_path(item.config, "data.training_view"),
        )
        for item in plan
    }
    assert cases == {
        ("parity", "law_only"),
        ("majority", "law_only"),
        ("parity", "audit_law_matched"),
        ("majority", "audit_law_matched"),
        ("parity", "sage_only"),
        ("parity", "herald_only"),
        ("parity", "no_signal"),
        ("parity", "surface_only"),
    }


def test_g00c_target_optimizer_panel_freezes_candidates_and_coverage() -> None:
    config = _config("g00c_strict_optimizer_1p5b.yaml")
    _assert_strict_full_update(config)
    plan = build_plan(config)
    assert len(plan) == 48
    assert get_path(config, "experiment.name") == "deployed_model_full_context_pilot"
    assert get_path(config, "model.name") == "Qwen/Qwen2.5-1.5B-Instruct"
    assert get_path(config, "model.revision") == (
        "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    )
    assert get_path(config, "train.batch_size") == 10
    assert get_path(config, "train.gradient_accumulation_steps") == 5
    assert {get_path(item.config, "data.rule_family") for item in plan} == {
        "parity",
        "majority",
    }
    assert {get_path(item.config, "data.q_p") for item in plan} == {0.95, 1.0}
    assert {
        (
            get_path(item.config, "train.algorithm"),
            get_path(item.config, "train.learning_rate"),
            get_path(item.config, "train.entropy_coefficient"),
        )
        for item in plan
    } == {
        ("sft", 3e-6, 0.0),
        ("sft", 1e-5, 0.0),
        ("outcome_rl", 1e-6, 0.01),
        ("outcome_rl", 3e-6, 0.01),
    }


def test_g00c_complete_gate_plan_is_unique_and_digest_frozen() -> None:
    names_and_digests = {
        "g00c_strict_engineering_0p5b.yaml": (
            72,
            "956c233351897c1e93dfc8f9730c28060fcd16908997a8f9c1e34d545aa2bee8",
        ),
        "g00c_strict_capability_0p5b.yaml": (
            24,
            "75aa6d82a09140d32e48634523952dc7c77ccfb51fa3b85c8660304147d9f738",
        ),
        "g00c_strict_optimizer_1p5b.yaml": (
            48,
            "a00076ad2c49159ecc603fd00e576a643c3c785fcb323900322373be0c529e5b",
        ),
        "g00b_deterministic_capability_validation.yaml": (
            24,
            "9defaab71d1e61dec02113ad5ed089eb4e8e769a307d1710ca558c1ad1048f4f",
        ),
    }
    all_keys: list[str] = []
    for name, (expected_count, expected_digest) in names_and_digests.items():
        keys = sorted(item.plan_key for item in build_plan(_config(name)))
        assert len(keys) == expected_count
        assert stable_hash(keys, 64) == expected_digest
        all_keys.extend(keys)
    assert len(all_keys) == len(set(all_keys)) == 168
    assert stable_hash(sorted(all_keys), 64) == (
        "0a9f0f861603059e19e4800163e5218cc94b409a304ab48a14eeac18ad2c3005"
    )


def test_g00c_checked_in_plans_match_the_configured_work_exactly() -> None:
    expected = {
        "g00c_strict_engineering_0p5b.yaml": (
            "g00c-strict-engineering-0p5b.jsonl",
            "6a8c011f5a1a87ffdbba0bb3b45d0383c49cc0b219dad937ea0e0c92979ba8bb",
        ),
        "g00c_strict_capability_0p5b.yaml": (
            "g00c-strict-capability-0p5b.jsonl",
            "f02916fe4e4a3b79f8c539ff1dbcfd23a7f97fa80c7cee9cf119f099321282ef",
        ),
        "g00c_strict_optimizer_1p5b.yaml": (
            "g00c-strict-optimizer-1p5b.jsonl",
            "8b02a8a382fa87675b106c44fb338119d0e1a0dd18e88e618fd52b554dd46dac",
        ),
    }
    plan_root = ROOT / "docs" / "goalzendo" / "plans"
    for config_name, (plan_name, expected_sha) in expected.items():
        plan_path = plan_root / plan_name
        records = [json.loads(line) for line in plan_path.read_text().splitlines()]
        configured = build_plan(_config(config_name))
        assert [record["plan_key"] for record in records] == [
            item.plan_key for item in configured
        ]
        assert hashlib.sha256(plan_path.read_bytes()).hexdigest() == expected_sha


def test_g00c_capability_panels_have_static_gate_coverage() -> None:
    required_views = {
        "law_only",
        "audit_law_matched",
        "sage_only",
        "herald_only",
        "no_signal",
        "surface_only",
    }
    for name, expected_model in (
        ("g00c_strict_capability_0p5b.yaml", "Qwen/Qwen2.5-0.5B-Instruct"),
        ("g00b_deterministic_capability_validation.yaml", "Qwen/Qwen2.5-1.5B-Instruct"),
    ):
        plan = build_plan(_config(name))
        assert {get_path(item.config, "model.name") for item in plan} == {expected_model}
        assert {get_path(item.config, "data.training_view") for item in plan} == required_views
        for view in ("law_only", "audit_law_matched"):
            selected = [
                item for item in plan if get_path(item.config, "data.training_view") == view
            ]
            assert {get_path(item.config, "data.rule_family") for item in selected} == {
                "parity",
                "majority",
            }
            assert {
                item.seed
                for item in selected
                if get_path(item.config, "data.rule_family") == "parity"
            } == set(get_path(_config(name), "run.seeds"))
            assert {
                item.seed
                for item in selected
                if get_path(item.config, "data.rule_family") == "majority"
            } == set(get_path(_config(name), "run.seeds"))
