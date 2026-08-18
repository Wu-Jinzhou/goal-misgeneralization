from __future__ import annotations

import copy
from pathlib import Path

import pytest

from goalzendo_hidden_law.config import (
    HiddenLawConfigError,
    build_hidden_law_plan,
    canonical_digest,
    load_hidden_law_config,
    scientific_config,
    validate_hidden_law_config,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/goalzendo/qwen35_hidden_law_finite_choice.yaml"


def test_registered_hidden_law_plan_is_exact_and_paired() -> None:
    config = load_hidden_law_config(CONFIG)
    plan = build_hidden_law_plan(config)

    assert len(plan) == 24
    assert len({condition.plan_key for condition in plan}) == 24
    assert len({condition.run_id for condition in plan}) == 24
    assert {condition.algorithm for condition in plan} == {"process_sft", "outcome_rl"}
    assert {condition.seed for condition in plan} == {23011, 23013, 23017, 23019, 23023, 23031}
    for seed in {condition.seed for condition in plan}:
        seed_rows = [condition for condition in plan if condition.seed == seed]
        assert len(seed_rows) == 4
        assert {(row.model_name, row.algorithm) for row in seed_rows} == {
            ("Qwen/Qwen3.5-0.8B", "process_sft"),
            ("Qwen/Qwen3.5-0.8B", "outcome_rl"),
            ("Qwen/Qwen3.5-2B", "process_sft"),
            ("Qwen/Qwen3.5-2B", "outcome_rl"),
        }


def test_only_operational_fields_are_excluded_from_identity() -> None:
    config = load_hidden_law_config(CONFIG)
    original = canonical_digest(scientific_config(config))

    moved = copy.deepcopy(config)
    moved["run"]["output_root"] = "/another/location"
    moved["run"]["smoke_seed"] = 23999
    moved["experiment"]["status"] = "prospective_frozen"
    validate_hidden_law_config(moved)
    assert canonical_digest(scientific_config(moved)) == original

    changed = copy.deepcopy(config)
    changed["training"]["blocks"] = 127
    with pytest.raises(HiddenLawConfigError, match="128 role blocks"):
        validate_hidden_law_config(changed)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("seeds",), [23011], "six registered"),
        (("game", "candidate_count"), 8, "four candidates"),
        (("game", "max_queries"), 3, "menu/query/terminal"),
        (("algorithms", 1, "trajectories_per_official"), 8, "four trajectories"),
        (("algorithms", 1, "classification_reward_weight"), 0.5, "exactly .25/.75"),
        (("training", "warmup_steps"), 6, "training runtime"),
        (("training", "scheduler"), "cosine", "training runtime"),
        (("evaluation", "final_quartets"), 8, "quartet counts"),
    ],
)
def test_registered_design_mutations_fail_closed(
    path: tuple[object, ...], value: object, message: str
) -> None:
    config = load_hidden_law_config(CONFIG)
    target: object = config
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]
    with pytest.raises(HiddenLawConfigError, match=message):
        validate_hidden_law_config(config)
