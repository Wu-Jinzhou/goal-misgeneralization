import copy
import hashlib
import inspect
import json
import math
from pathlib import Path

import pytest

from goalzendo import analysis
from goalzendo.artifacts import stable_hash
from goalzendo.config import get_path, load_config, validate_config
from goalzendo.runner import (
    LaunchGuardError,
    assert_launch_unlocked,
    build_plan,
    implementation_provenance,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "configs" / "goalzendo"
PROTOCOL = ROOT / "docs" / "goalzendo" / "protocols" / "g00f-capability-repair.md"
ASSESSMENT = (
    ROOT / "reproducibility" / "goalzendo" / "g00d-gate-20260811" / "g00-gate-assessment-derived-pre-fix.json"
)
GUARD = "G00F_DESIGN_ONLY__FRESH_GATE_AND_EXECUTION_MANIFEST_REQUIRED"

CONFIGS = {
    "g00f_capability_repair_0p5b.yaml": {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
        "seeds": {10007, 10009, 10037, 10039, 10061, 10067, 10069, 10079, 10091, 10093},
        "yaml_sha256": "3c1ef3113be76373892b256b60ec5540419bd932d813a3db230d9a3f913cea30",
        "plan_digest": "1b7ffcd2b588ca69103be025e4e28a63f8877bbe04a53154937348be8407c473",
    },
    "g00f_capability_repair_1p5b.yaml": {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        "seeds": {10103, 10111, 10133, 10139, 10141, 10151, 10159, 10163, 10169, 10177},
        "yaml_sha256": "d8fe04d5f6ea5041481f13e1c0fd93498520f40c7aa793c3bfe35d2e81bd2dc0",
        "plan_digest": "7f9619c3903cce5f94511bbaf4c4de684fc78969a6408d967b36f1edbbe96e6a",
    },
}

EXPECTED_CASES = {
    ("parity", "law_only", ("law_only",)),
    ("majority", "law_only", ("law_only",)),
    ("parity", "audit_law_matched", ("audit_law_matched",)),
    ("majority", "audit_law_matched", ("audit_law_matched",)),
    ("parity", "sage_only", ("sage_only",)),
    ("parity", "herald_only", ("herald_only",)),
    ("parity", "no_signal", ("no_signal",)),
    ("parity", "surface_only", ("surface_only",)),
}


def _config(name: str):
    config = load_config(CONFIG_ROOT / name)
    assert validate_config(config) == []
    return config


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _central_binomial_interval(n: int) -> tuple[int, int]:
    """Exact equal-tail interval for p=.5 and alpha=.05 using integers."""

    denominator = 1 << n
    cumulative = 0
    lower: int | None = None
    for successes in range(n + 1):
        cumulative += math.comb(n, successes)
        if lower is None and 40 * cumulative >= denominator:
            lower = successes
        if 40 * cumulative >= 39 * denominator:
            assert lower is not None
            return lower, successes
    raise AssertionError("unreachable binomial interval")


def test_g00f_binds_the_immutable_failed_evidence_and_legacy_source() -> None:
    assert _sha256(ASSESSMENT) == ("c1d60b12dad3ab695f5aa5ce8a909a0c3b00d3affad6b8bc6df0ca66ad209488")
    payload = json.loads(ASSESSMENT.read_text())
    assert payload["assessment_digest"] == (
        "9ff51f5c54d10a9b98a611363060cc738e9baf066ed7dc5f1f0d859932364a59"
    )
    assert payload["evidence_source_fingerprint"] == (
        "1a8146377b9a9620690025671614edb2dd20d214f4e195da3cf3528809b2c694"
    )
    assert implementation_provenance(ROOT)["implementation_fingerprint"] == (
        "1a8146377b9a9620690025671614edb2dd20d214f4e195da3cf3528809b2c694"
    )

    measurements = payload["measurements"]
    assert measurements["rule_adapter_position_agreement"]["audit_law_matched"] == {
        "A": 0.9462890625,
        "B": 0.9469401041666666,
    }
    assert measurements["constrained_scorer"] == {
        "finite_probability_fraction": 1.0,
        "maximum_absolute_position_bias": 0.5,
        "maximum_probability_sum_error": 2.220446049250313e-16,
    }

    by_model = {
        value["model_identity"]["requested_model"]: value
        for value in measurements["capability_by_model"].values()
    }
    assert by_model["Qwen/Qwen2.5-0.5B-Instruct"]["rule_adapter_position_agreement"]["audit_law_matched"] == {
        "A": 0.8977864583333334,
        "B": 0.89453125,
    }
    assert by_model["Qwen/Qwen2.5-1.5B-Instruct"]["rule_adapter_position_agreement"]["audit_law_matched"] == {
        "A": 0.9947916666666666,
        "B": 0.9993489583333334,
    }
    assert {value["constrained_scorer"]["maximum_absolute_position_bias"] for value in by_model.values()} == {
        0.5
    }


def test_g00d_scorer_statistic_is_an_unpaired_maximum_over_all_runs() -> None:
    source = inspect.getsource(analysis._capability_measurement_panel)
    assert "scorer_frames = [predictions[run.run_id] for run in runs]" in source
    assert '(frame["predicted_action"] == 1).mean()' in source
    assert "for frame in scorer_frames" in source


@pytest.mark.parametrize("name", tuple(CONFIGS))
def test_g00f_configs_are_fixed_full_update_singleton_repairs(name: str) -> None:
    expected = CONFIGS[name]
    config = _config(name)

    assert get_path(config, "experiment.id") == "g00f"
    assert get_path(config, "experiment.status") == "prospective"
    assert get_path(config, "run.launch_guard") == GUARD
    assert get_path(config, "run.protocol_unlocked") is False
    assert get_path(config, "model.name") == expected["model"]
    assert get_path(config, "model.revision") == expected["revision"]
    assert get_path(config, "update.method") == "full"
    assert get_path(config, "data.n_train") == 10_000
    assert get_path(config, "data.n_validation") == 1_000
    assert get_path(config, "data.n_eval_per_cell") == 64
    assert get_path(config, "train.algorithm") == "sft"
    assert get_path(config, "train.steps") == 1_000
    assert get_path(config, "train.learning_rate") == 1e-5
    assert get_path(config, "train.batch_size") == 10
    assert get_path(config, "train.gradient_accumulation_steps") == 5
    assert get_path(config, "train.deterministic_algorithms") is True
    assert get_path(config, "train.allow_tf32") is False
    assert get_path(config, "train.cublas_workspace_config") == ":4096:8"
    assert get_path(config, "evaluation.mirror_pairs") is True
    assert get_path(config, "evaluation.final_eval_per_cell") == 64
    assert get_path(config, "run.checkpoint_steps") == [1_000]
    assert get_path(config, "run.snapshot_steps") == []
    assert get_path(config, "sweep") == {}
    assert set(get_path(config, "run.seeds")) == expected["seeds"]
    assert _sha256(CONFIG_ROOT / name) == expected["yaml_sha256"]


def test_g00f_has_exact_complete_fresh_panels_and_plan_identities() -> None:
    all_keys: set[str] = set()
    all_seeds: set[int] = set()
    for name, expected in CONFIGS.items():
        plan = build_plan(_config(name))
        assert len(plan) == 80
        assert {item.seed for item in plan} == expected["seeds"]
        cases = {
            (
                get_path(item.config, "data.rule_family"),
                get_path(item.config, "data.training_view"),
                tuple(get_path(item.config, "evaluation.prompt_views")),
            )
            for item in plan
        }
        assert cases == EXPECTED_CASES
        keys = {item.plan_key for item in plan}
        assert len(keys) == 80
        assert stable_hash(sorted(keys), 64) == expected["plan_digest"]
        assert all_keys.isdisjoint(keys)
        assert all_seeds.isdisjoint(expected["seeds"])
        all_keys.update(keys)
        all_seeds.update(expected["seeds"])

    old_seeds: set[int] = set()
    for old_name in (
        "g00d_fixed_window_capability_0p5b.yaml",
        "g00d_fixed_window_capability_1p5b.yaml",
    ):
        old_seeds.update(get_path(_config(old_name), "run.seeds"))
    assert all_seeds.isdisjoint(old_seeds)
    assert len(all_keys) == 160


@pytest.mark.parametrize("name", tuple(CONFIGS))
def test_g00f_design_guard_fails_closed_even_with_protocol_override(name: str) -> None:
    config = _config(name)
    with pytest.raises(LaunchGuardError, match="experiment launch is locked"):
        assert_launch_unlocked(config, repo=ROOT)

    attempted_override = copy.deepcopy(config)
    attempted_override["run"]["protocol_unlocked"] = True
    with pytest.raises(LaunchGuardError, match="experiment launch is locked"):
        assert_launch_unlocked(attempted_override, repo=ROOT)


def test_g00f_exact_counts_power_and_runtime_budget_are_frozen() -> None:
    assert _central_binomial_interval(5_120) == (2_490, 2_630)
    assert _central_binomial_interval(10_240) == (5_021, 5_219)
    assert math.floor(0.02 * 256) == 5
    assert pytest.approx(0.9436864852905273) == 1 - (1 - 0.25) ** 10
    assert pytest.approx(0.9717524751) == 1 - (1 - 0.30) ** 10
    assert pytest.approx(0.9826584700841674) == 1 - (1 - 1 / 3) ** 10
    assert pytest.approx(0.2588655508930523) == 1 - 0.05 ** (1 / 10)

    scale = (80 * 1_000) / (24 * 512)
    hours_0p5b = 2 * (4_837 / 3_600) * scale
    hours_1p5b = 2 * (6_525 / 3_600) * scale
    assert scale == pytest.approx(6.5104166667)
    assert hours_0p5b == pytest.approx(17.49493634)
    assert hours_1p5b == pytest.approx(23.60026042)
    assert hours_0p5b + hours_1p5b < 56


def test_g01_settings_and_guard_remain_locked() -> None:
    config = _config("g01_known_law.yaml")
    assert get_path(config, "run.launch_guard") == ("G00_NOT_PASSED__LEARNING_RATES_NOT_FROZEN")
    plan = build_plan(config)
    selected = {
        (
            get_path(item.config, "train.algorithm"),
            get_path(item.config, "train.learning_rate"),
            get_path(item.config, "train.entropy_coefficient"),
        )
        for item in plan
    }
    assert selected == {
        ("sft", 3e-6, 0.0),
        ("outcome_rl", 3e-6, 0.01),
    }


def test_protocol_names_the_new_estimand_without_rewriting_the_old_failure() -> None:
    text = PROTOCOL.read_text()
    assert "`g00f_paired_candidate_order_symmetry_v1`" in text
    assert "keeps the frozen G00-D scorer result false" in text
    assert "There is one candidate" in text
    assert "exactly 160 independent trained-model runs" in text
    assert "No Runpod job was started" in text
    assert "G01 remains locked" in text
    assert GUARD in text
