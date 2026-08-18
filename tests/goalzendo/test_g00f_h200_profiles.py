from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml  # type: ignore[import-untyped]

from goalzendo.config import (
    canonical_config,
    get_path,
    load_config,
    set_path,
    validate_config,
)
from goalzendo.runner import RunSpec, build_plan
from goalzendo.training import deterministic_batch_indices

ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "configs" / "goalzendo"
GUARD = "G00F_H200_FROZEN_SELECTOR_AND_EXECUTION_MANIFEST_REQUIRED"

MODELS = {
    "0p5b": "g00f_capability_repair_0p5b.yaml",
    "1p5b": "g00f_capability_repair_1p5b.yaml",
}
PROFILES = {
    "baseline": {
        "batch_size": 10,
        "gradient_accumulation_steps": 5,
        "gradient_checkpointing": True,
        "evaluation_batch_size": 16,
    },
    "tuned": {
        "batch_size": 50,
        "gradient_accumulation_steps": 1,
        "gradient_checkpointing": False,
        "evaluation_batch_size": 128,
    },
}

AUTHORIZED_CHILD_PATHS = (
    "experiment.name",
    "run.output_root",
    "run.launch_guard",
    "run.protocol_unlocked",
    "train.batch_size",
    "train.gradient_accumulation_steps",
    "train.gradient_checkpointing",
    "evaluation.batch_size",
)


def _profile_path(profile: str, model: str) -> Path:
    return CONFIG_ROOT / f"g00f_h200_{profile}_{model}.yaml"


def _load(name: str | Path) -> dict[str, Any]:
    path = CONFIG_ROOT / name if isinstance(name, str) else name
    config = load_config(path)
    assert validate_config(config) == []
    return config


def _normalized_child(config: dict[str, Any], parent: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(canonical_config(config))
    for path in AUTHORIZED_CHILD_PATHS:
        set_path(normalized, path, get_path(parent, path))
    return normalized


def _schedule_signature(plan: tuple[RunSpec, ...]) -> list[tuple[Any, ...]]:
    return [
        (
            spec.global_index,
            spec.cell_index,
            spec.seed,
            tuple(sorted(spec.seeds.items())),
            get_path(spec.config, "data.rule_family"),
            get_path(spec.config, "data.training_view"),
            tuple(get_path(spec.config, "evaluation.prompt_views")),
        )
        for spec in plan
    ]


def _optimizer_step_examples(config: dict[str, Any], *, seed: int, step: int) -> torch.Tensor:
    batch_size = int(get_path(config, "train.batch_size"))
    accumulation = int(get_path(config, "train.gradient_accumulation_steps"))
    return torch.cat(
        [
            deterministic_batch_indices(
                int(get_path(config, "data.n_train")),
                batch_size,
                step * accumulation + offset,
                seed=seed,
            )
            for offset in range(accumulation)
        ]
    )


@pytest.mark.parametrize("model", tuple(MODELS))
@pytest.mark.parametrize("profile", tuple(PROFILES))
def test_h200_profile_is_an_explicit_single_profile_child(model: str, profile: str) -> None:
    path = _profile_path(profile, model)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = PROFILES[profile]

    assert set(raw) == {"extends", "experiment", "run", "train", "evaluation"}
    assert raw["extends"] == MODELS[model]
    assert raw["experiment"] == {
        "id": "g00f",
        "name": f"capability_repair_h200_{profile}_{model}",
    }
    assert raw["run"] == {
        "output_root": f"/workspace/artifacts-goalzendo/g00f-h200-{profile}-{model}",
        "launch_guard": GUARD,
        "protocol_unlocked": False,
    }
    assert raw["train"] == {
        "batch_size": expected["batch_size"],
        "gradient_accumulation_steps": expected["gradient_accumulation_steps"],
        "gradient_checkpointing": expected["gradient_checkpointing"],
    }
    assert raw["evaluation"] == {"batch_size": expected["evaluation_batch_size"]}
    assert "cases" not in raw and "sweep" not in raw


@pytest.mark.parametrize("model", tuple(MODELS))
@pytest.mark.parametrize("profile", tuple(PROFILES))
def test_h200_resolved_config_changes_only_authorized_profile_fields(
    model: str,
    profile: str,
) -> None:
    parent = _load(MODELS[model])
    config = _load(_profile_path(profile, model))
    expected = PROFILES[profile]

    assert get_path(config, "experiment.id") == "g00f"
    assert get_path(config, "experiment.status") == "prospective"
    assert get_path(config, "run.launch_guard") == GUARD
    assert get_path(config, "run.protocol_unlocked") is False
    assert get_path(config, "train.batch_size") == expected["batch_size"]
    assert get_path(config, "train.gradient_accumulation_steps") == expected["gradient_accumulation_steps"]
    assert get_path(config, "train.gradient_checkpointing") is expected["gradient_checkpointing"]
    assert get_path(config, "evaluation.batch_size") == expected["evaluation_batch_size"]
    assert get_path(config, "train.batch_size") * get_path(config, "train.gradient_accumulation_steps") == 50
    assert _normalized_child(config, parent) == canonical_config(parent)


@pytest.mark.parametrize("model", tuple(MODELS))
def test_h200_profiles_preserve_registered_schedule_but_have_fresh_plan_identities(
    model: str,
) -> None:
    parent_plan = build_plan(_load(MODELS[model]))
    baseline_plan = build_plan(_load(_profile_path("baseline", model)))
    tuned_plan = build_plan(_load(_profile_path("tuned", model)))

    assert len(parent_plan) == len(baseline_plan) == len(tuned_plan) == 80
    assert _schedule_signature(baseline_plan) == _schedule_signature(parent_plan)
    assert _schedule_signature(tuned_plan) == _schedule_signature(parent_plan)
    key_sets = [{spec.plan_key for spec in plan} for plan in (parent_plan, baseline_plan, tuned_plan)]
    assert all(len(keys) == 80 for keys in key_sets)
    assert key_sets[0].isdisjoint(key_sets[1])
    assert key_sets[0].isdisjoint(key_sets[2])
    assert key_sets[1].isdisjoint(key_sets[2])


@pytest.mark.parametrize("model", tuple(MODELS))
def test_tuned_profile_preserves_exact_example_order_at_every_optimizer_step(
    model: str,
) -> None:
    baseline = _load(_profile_path("baseline", model))
    tuned = _load(_profile_path("tuned", model))

    assert get_path(baseline, "data.n_train") == get_path(tuned, "data.n_train") == 10_000
    assert get_path(baseline, "train.steps") == get_path(tuned, "train.steps") == 1_000
    assert get_path(baseline, "run.seeds") == get_path(tuned, "run.seeds")
    for seed in get_path(baseline, "run.seeds"):
        for step in range(get_path(baseline, "train.steps")):
            baseline_examples = _optimizer_step_examples(baseline, seed=seed, step=step)
            tuned_examples = _optimizer_step_examples(tuned, seed=seed, step=step)
            assert len(baseline_examples) == len(tuned_examples) == 50
            assert torch.equal(baseline_examples, tuned_examples)
