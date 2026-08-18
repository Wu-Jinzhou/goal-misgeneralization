from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from goalzendo.config import (
    DEFAULT_CONFIG,
    PROTECTED_GUARD_SIGNATURES,
    ConfigError,
    canonical_config,
    deep_merge,
    expand_sweep,
    load_config,
    protected_guard_signature,
    smoke_config,
    validate_config,
)
from goalzendo.schema import default_feature_names


def test_default_and_smoke_configs_validate() -> None:
    assert validate_config(DEFAULT_CONFIG) == []
    smoke = smoke_config(DEFAULT_CONFIG)
    assert smoke["train"]["steps"] == 2
    assert smoke["data"]["n_train"] == 40
    assert smoke["run"]["snapshot_steps"] == [2]


@pytest.mark.parametrize(
    "override",
    ("run.launch_guard=null", "run.launch_guard=changed", "experiment.id=g00"),
)
def test_registered_confirmatory_guard_cannot_be_overridden(override: str) -> None:
    path = Path(__file__).resolve().parents[2] / "configs" / "goalzendo" / "g01_known_law.yaml"
    with pytest.raises(ConfigError, match=r"guarded experiment|launch_guard"):
        load_config(path, [override])


def test_registered_confirmatory_study_cannot_be_copied_and_relabelled(
    tmp_path: Path,
) -> None:
    path = Path(__file__).resolve().parents[2] / "configs" / "goalzendo" / "g01_known_law.yaml"
    copied = canonical_config(load_config(path))
    copied["experiment"] = {
        "id": "g00",
        "name": "unregistered_copy",
        "status": "exploratory",
    }
    copied["run"]["output_root"] = str(tmp_path / "unregistered")
    copied["run"]["launch_guard"] = None
    copied["run"]["resume"] = False
    copied["evaluation"]["save_activations"] = True
    for case in copied["cases"]:
        case["train.learning_rate"] = 0.123
        case["train.entropy_coefficient"] = 0.456
    relabelled = tmp_path / "unregistered.yaml"
    relabelled.write_text(yaml.safe_dump(copied, sort_keys=True), encoding="utf-8")

    with pytest.raises(ConfigError, match="guarded experiment cannot be copied or relabeled"):
        load_config(relabelled)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda config: config["data"].update({"guard_bypass_note": "ignored"}), "unknown data"),
        (
            lambda config: config.update(
                {"cases": [*config["cases"], {"guard_bypass_note": "ignored"}]}
            ),
            "unknown case configuration path",
        ),
    ),
)
def test_unknown_keys_cannot_evade_registered_study_guard(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    path = Path(__file__).resolve().parents[2] / "configs" / "goalzendo" / "g01_known_law.yaml"
    copied = canonical_config(load_config(path))
    copied["experiment"]["id"] = "g00"
    copied["run"]["launch_guard"] = None
    mutation(copied)
    relabelled = tmp_path / "unknown-key.yaml"
    relabelled.write_text(yaml.safe_dump(copied, sort_keys=True), encoding="utf-8")

    with pytest.raises(ConfigError, match=message):
        load_config(relabelled)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda config: config["update"].update({"rank": 999}),
        lambda config: config["update"].update({"dropout": 0.321}),
        lambda config: config["update"].update({"target_modules": ["ignored_proj"]}),
        lambda config: config["data"].update({"joint_error_rate": 0.0}),
        lambda config: config["data"].update({"renderer": "nonce"}),
        lambda config: config["data"].update(
            {"distractor_features": list(reversed(config["data"]["distractor_features"]))}
        ),
        lambda config: config["data"].update(
            {"distractor_features": [*config["data"]["distractor_features"], 5]}
        ),
        lambda config: config["data"].update(
            {"feature_names": list(default_feature_names(config["data"]["feature_count"]))}
        ),
        lambda config: config["evaluation"].update({"acquisition_threshold": 0.123}),
        lambda config: config["train"].update({"weight_decay": 0}),
        lambda config: config["train"].update({"weight_decay": -0.0}),
        lambda config: config["train"].update({"grad_clip": 1}),
        lambda config: config["model"].update({"dtype": "BF16"}),
        lambda config: config["run"].update(
            {"seeds": list(reversed(config["run"]["seeds"]))}
        ),
        lambda config: config["cases"][0].update({"train.samples_per_prompt": 8}),
        lambda config: config["cases"].append(
            {"evaluation.acquisition_threshold": 0.123}
        ),
    ),
)
def test_training_equivalent_edits_cannot_evade_registered_study_guard(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], None],
) -> None:
    path = Path(__file__).resolve().parents[2] / "configs" / "goalzendo" / "g01_known_law.yaml"
    copied = canonical_config(load_config(path))
    copied["experiment"]["id"] = "g00"
    copied["run"]["launch_guard"] = None
    mutation(copied)
    relabelled = tmp_path / "training-equivalent.yaml"
    relabelled.write_text(yaml.safe_dump(copied, sort_keys=True), encoding="utf-8")

    with pytest.raises(ConfigError, match="guarded experiment cannot be copied or relabeled"):
        load_config(relabelled)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda config: config["update"].update(
            {"target_modules": list(reversed(config["update"]["target_modules"]))}
        ),
        lambda config: config["update"].update(
            {"target_modules": [*config["update"]["target_modules"], "q_proj"]}
        ),
        lambda config: config["update"].update(
            {"target_modules": [*config["update"]["target_modules"], "unused_proj"]}
        ),
        lambda config: config["update"].pop("rank"),
        lambda config: config["update"].pop("alpha"),
        lambda config: config["update"].pop("dropout"),
        lambda config: config["update"].pop("bias"),
    ),
)
def test_training_equivalent_lora_edits_remain_guarded(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], Any],
) -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "goalzendo"
        / "g01_known_law_lora_secondary.yaml"
    )
    copied = canonical_config(load_config(path))
    copied["experiment"]["id"] = "g00"
    copied["run"]["launch_guard"] = None
    mutation(copied)
    relabelled = tmp_path / "training-equivalent-lora.yaml"
    relabelled.write_text(yaml.safe_dump(copied, sort_keys=True), encoding="utf-8")

    with pytest.raises(ConfigError, match="guarded experiment cannot be copied or relabeled"):
        load_config(relabelled)


def test_schema_version_rejects_numeric_lookalikes() -> None:
    config = deep_merge(DEFAULT_CONFIG, {"schema_version": 1.0})
    with pytest.raises(ConfigError, match="integer 1"):
        validate_config(config)


@pytest.mark.parametrize(
    "name",
    (
        "g01_known_law.yaml",
        "g02_evidence_geometry.yaml",
        "g01_known_law_lora_secondary.yaml",
        "g01_longitudinal_full_ft.yaml",
    ),
)
def test_guarded_studies_use_the_deterministic_cuda_contract(name: str) -> None:
    path = Path(__file__).resolve().parents[2] / "configs" / "goalzendo" / name
    config = load_config(path)
    assert config["train"]["deterministic_algorithms"] is True
    assert config["train"]["allow_tf32"] is False
    assert config["train"]["cublas_workspace_config"] == ":4096:8"
    signature = protected_guard_signature(config)
    assert PROTECTED_GUARD_SIGNATURES[signature] == (
        config["experiment"]["id"],
        config["run"]["launch_guard"],
    )


@pytest.mark.parametrize(
    "override",
    (
        "cases=[{experiment.id: g00, run.launch_guard: null, _declared_experiment_id: g00}]",
        "sweep={_declared_experiment_id: [g00]}",
        "sweep={run.launch_guard: [null]}",
    ),
)
def test_sweep_or_case_cannot_relabel_a_guarded_experiment(override: str) -> None:
    path = Path(__file__).resolve().parents[2] / "configs" / "goalzendo" / "g01_known_law.yaml"
    with pytest.raises(ConfigError, match=r"unknown (case|sweep)|may not alter experiment identity"):
        config = load_config(path, [override])
        expand_sweep(config)


@pytest.mark.parametrize(
    ("run", "message"),
    [
        ({"snapshot_steps": [2, 1]}, "sorted and unique"),
        ({"snapshot_steps": [1, 1]}, "sorted and unique"),
        ({"snapshot_steps": [3]}, "scheduled evaluation"),
        ({"snapshot_steps": [True]}, "contain integers"),
        ({"save_checkpoints": False, "snapshot_steps": [2]}, "requires"),
    ],
)
def test_snapshot_retention_config_fails_closed(
    run: dict[str, object],
    message: str,
) -> None:
    config = deep_merge(
        DEFAULT_CONFIG,
        {
            "run": run,
            "train": {"steps": 2, "eval_steps": [0, 1, 2]},
        },
    )
    with pytest.raises(ConfigError, match=message):
        validate_config(config)


def test_snapshot_retention_accepts_final_step_even_if_not_explicitly_listed() -> None:
    config = deep_merge(
        DEFAULT_CONFIG,
        {
            "run": {
                "save_checkpoints": True,
                "checkpoint_steps": [2],
                "snapshot_steps": [2],
            },
            "train": {"steps": 2, "eval_steps": [0, 1]},
        },
    )
    assert validate_config(config) == []


@pytest.mark.parametrize(
    ("checkpoint_steps", "message"),
    [
        ([0, 2], "step zero"),
        ([2, 1], "sorted and unique"),
        ([1, 1, 2], "sorted and unique"),
        ([3], "scheduled evaluation"),
        ([True, 2], "contain integers"),
        ([1], "final training step"),
    ],
)
def test_resumable_checkpoint_schedule_fails_closed(
    checkpoint_steps: list[object],
    message: str,
) -> None:
    config = deep_merge(
        DEFAULT_CONFIG,
        {
            "run": {"save_checkpoints": True, "checkpoint_steps": checkpoint_steps},
            "train": {"steps": 2, "eval_steps": [0, 1, 2]},
        },
    )
    with pytest.raises(ConfigError, match=message):
        validate_config(config)


def test_resumable_schedule_is_sparse_and_independent_of_snapshot_schedule() -> None:
    config = deep_merge(
        DEFAULT_CONFIG,
        {
            "run": {
                "save_checkpoints": True,
                "checkpoint_steps": [2, 4],
                "snapshot_steps": [1, 4],
            },
            "train": {"steps": 4, "eval_steps": [0, 1, 2, 3, 4]},
        },
    )
    assert validate_config(config) == []


@pytest.mark.parametrize("value", [-1, 0.5, True])
def test_parameter_finite_check_interval_must_be_nonnegative_integer(value: object) -> None:
    config = deep_merge(DEFAULT_CONFIG, {"train": {"parameter_finite_check_interval": value}})
    with pytest.raises(ConfigError, match="parameter_finite_check_interval"):
        validate_config(config)


@pytest.mark.parametrize(
    ("train", "message"),
    [
        ({"deterministic_algorithms": "yes"}, "deterministic_algorithms"),
        ({"allow_tf32": 1}, "allow_tf32"),
        (
            {"deterministic_algorithms": True, "allow_tf32": True},
            "require train.allow_tf32=false",
        ),
        ({"cublas_workspace_config": ":bad"}, "cublas_workspace_config"),
    ],
)
def test_deterministic_execution_config_fails_closed(
    train: dict[str, object],
    message: str,
) -> None:
    config = deep_merge(DEFAULT_CONFIG, {"train": train})
    with pytest.raises(ConfigError, match=message):
        validate_config(config)


def test_deterministic_execution_config_accepts_cuda_contract() -> None:
    config = deep_merge(
        DEFAULT_CONFIG,
        {
            "train": {
                "deterministic_algorithms": True,
                "allow_tf32": False,
                "cublas_workspace_config": ":4096:8",
            }
        },
    )
    assert validate_config(config) == []


def test_expected_outcome_rl_is_a_valid_training_algorithm() -> None:
    config = deep_merge(
        DEFAULT_CONFIG,
        {"train": {"algorithm": "expected_outcome_rl"}},
    )
    assert validate_config(config) == []


def test_law_and_sage_features_must_be_disjoint() -> None:
    config = deep_merge(DEFAULT_CONFIG, {"data": {"sage_features": [2, 4]}})
    with pytest.raises(ConfigError, match="disjoint"):
        validate_config(config)


def test_proxy_rate_must_be_exactly_representable() -> None:
    config = deep_merge(DEFAULT_CONFIG, {"data": {"n_train": 12, "q_p": 0.95}})
    with pytest.raises(ConfigError, match="represented exactly"):
        validate_config(config)


def test_per_tuple_conflict_concentration_accepts_a_positive_integer() -> None:
    assert "concentrated_unique_conflicts_per_tuple" not in DEFAULT_CONFIG["data"]
    config = deep_merge(
        DEFAULT_CONFIG,
        {"data": {"concentrated_unique_conflicts_per_tuple": 16}},
    )
    assert validate_config(config) == []


@pytest.mark.parametrize("value", [0, -1])
def test_per_tuple_conflict_concentration_must_be_positive(value: int) -> None:
    config = deep_merge(
        DEFAULT_CONFIG,
        {"data": {"concentrated_unique_conflicts_per_tuple": value}},
    )
    with pytest.raises(ConfigError, match="concentrated_unique_conflicts_per_tuple must be positive"):
        validate_config(config)


@pytest.mark.parametrize("value", [True, 1.5])
def test_per_tuple_conflict_concentration_must_be_an_integer(value: object) -> None:
    config = deep_merge(
        DEFAULT_CONFIG,
        {"data": {"concentrated_unique_conflicts_per_tuple": value}},
    )
    with pytest.raises(ConfigError, match="concentrated_unique_conflicts_per_tuple must be an integer"):
        validate_config(config)


def test_expand_sweep_combines_axes_and_coupled_cases() -> None:
    config = deep_merge(
        DEFAULT_CONFIG,
        {
            "sweep": {"data.q_p": [0.9, 0.99]},
            "cases": [
                {"data.rule_family": "parity"},
                {"data.rule_family": "majority"},
            ],
        },
    )
    cells = expand_sweep(config)
    assert len(cells) == 4
    assert {(cell["data"]["q_p"], cell["data"]["rule_family"]) for cell in cells} == {
        (0.9, "parity"),
        (0.9, "majority"),
        (0.99, "parity"),
        (0.99, "majority"),
    }


def test_load_config_supports_extends_and_overrides(tmp_path: Path) -> None:
    parent = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    parent.write_text("data:\n  n_train: 100\n  q_p: 0.9\n  q_q: 0.9\n", encoding="utf-8")
    child.write_text("extends: base.yaml\ntrain:\n  steps: 8\n  eval_steps: [0, 1, 2, 4, 8]\n", encoding="utf-8")
    config = load_config(child, ["model.name=local/tiny"])
    assert config["data"]["n_train"] == 100
    assert config["model"]["name"] == "local/tiny"


def test_g00_maps_are_disjoint_and_1p5b_controls_match_g01_model() -> None:
    repo = Path(__file__).resolve().parents[2]
    g00 = load_config(repo / "configs/goalzendo/g00_capability_controls.yaml")
    g00_1p5 = load_config(repo / "configs/goalzendo/g00_capability_controls_1p5b.yaml")
    pilot_1p5 = load_config(repo / "configs/goalzendo/g00_pilot_1p5b.yaml")
    smoke = load_config(repo / "configs/goalzendo/smoke.yaml")
    g01 = load_config(repo / "configs/goalzendo/g01_known_law.yaml")

    g00_active = set(g00["data"]["law_features"]) | set(g00["data"]["sage_features"])
    g01_active = set(g01["data"]["law_features"]) | set(g01["data"]["sage_features"])
    assert g00_active.isdisjoint(g01_active)
    assert smoke["data"]["law_features"] == g00["data"]["law_features"]
    assert smoke["data"]["sage_features"] == g00["data"]["sage_features"]
    assert smoke["data"]["feature_names"] == g00["data"]["feature_names"]
    assert set(g00["data"]["feature_names"]).isdisjoint(
        default_feature_names(g01["data"]["feature_count"])
    )

    assert g00_1p5["model"]["name"] == g01["model"]["name"]
    assert g00_1p5["model"]["revision"] == g01["model"]["revision"]
    cells = expand_sweep(g00_1p5)
    assert len(cells) == 8
    assert {cell["data"]["training_view"] for cell in cells} == {
        "law_only",
        "audit_law_matched",
        "sage_only",
        "herald_only",
        "no_signal",
        "surface_only",
    }
    assert {
        cell["data"]["rule_family"]
        for cell in cells
        if cell["data"]["training_view"] in {"law_only", "audit_law_matched"}
    } == {"parity", "majority"}
    assert all(cell["data"]["law_features"] == [5, 6, 7] for cell in cells)

    assert pilot_1p5["model"]["name"] == g01["model"]["name"]
    assert pilot_1p5["model"]["revision"] == g01["model"]["revision"]
    pilot_cells = expand_sweep(pilot_1p5)
    assert len(pilot_cells) == 16
    assert {
        (
            cell["data"]["rule_family"],
            cell["data"]["q_p"],
            cell["train"]["algorithm"],
        )
        for cell in pilot_cells
    } == {
        (family, q_p, algorithm)
        for family in ("parity", "majority")
        for q_p in (0.95, 1.0)
        for algorithm in ("sft", "outcome_rl")
    }
    assert all(cell["data"]["training_view"] == "full" for cell in pilot_cells)
    assert {cell["train"]["learning_rate"] for cell in pilot_cells} == {
        0.000001,
        0.000003,
        0.00001,
    }
    assert all(cell["update"]["method"] == "full" for cell in pilot_cells)
    assert all(
        cell["train"]["batch_size"] * cell["train"]["gradient_accumulation_steps"] == 50
        for cell in pilot_cells
    )


def test_primary_full_ft_and_secondary_retention_families_are_separate() -> None:
    repo = Path(__file__).resolve().parents[2]
    primary = load_config(repo / "configs/goalzendo/g01_known_law.yaml")
    lora = load_config(repo / "configs/goalzendo/g01_known_law_lora_secondary.yaml")
    longitudinal = load_config(repo / "configs/goalzendo/g01_longitudinal_full_ft.yaml")

    assert primary["update"] == {"method": "full"}
    assert primary["train"]["batch_size"] == 10
    assert primary["train"]["gradient_accumulation_steps"] == 5
    assert primary["train"]["max_sequence_length"] == 640
    assert primary["run"]["checkpoint_steps"] == [256, 512, 768, 1000]
    assert primary["run"]["snapshot_steps"] == [1000]
    assert primary["experiment"]["status"] == "prospective"
    assert primary["run"]["launch_guard"] == "G00_NOT_PASSED__LEARNING_RATES_NOT_FROZEN"
    assert {
        (
            cell["train"]["algorithm"],
            cell["train"]["learning_rate"],
            cell["train"]["entropy_coefficient"],
        )
        for cell in expand_sweep(primary)
    } == {
        ("sft", 0.000003, 0.0),
        ("outcome_rl", 0.000003, 0.01),
    }
    assert str(primary["run"]["output_root"]).startswith("/workspace/")

    assert lora["update"]["method"] == "lora"
    assert lora["experiment"]["id"] != primary["experiment"]["id"]
    assert lora["run"]["output_root"] != primary["run"]["output_root"]
    assert longitudinal["update"]["method"] == "full"
    assert longitudinal["experiment"]["status"] == "adaptive"
    assert longitudinal["run"]["snapshot_steps"] == [64, 256, 1000]
