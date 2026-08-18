from __future__ import annotations

from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any

from goalzendo.config import get_path, load_config
from goalzendo.runner import RunSpec, build_plan

ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "configs" / "goalzendo"

MODEL_REVISIONS = {
    "Qwen/Qwen3.5-0.8B": "2fc06364715b967f1860aea9cf38778875588b17",
    "Qwen/Qwen3.5-2B": "15852e8c16360a2fea060d615a32b45270f8a8fc",
}
ALGORITHM_SETTINGS = {
    "sft": (0.00001, 0.0),
    "outcome_rl": (0.000003, 0.01),
}
PILOT_SEEDS = [20903]
MAIN_SEEDS = [21011, 21013, 21017, 21019, 21023, 21031]


def _load(name: str) -> dict[str, Any]:
    return load_config(CONFIG_ROOT / name)


def _cell_identity(spec: RunSpec) -> tuple[str, str, str, float, float]:
    config = spec.config
    return (
        str(get_path(config, "model.name")),
        str(get_path(config, "model.revision")),
        str(get_path(config, "train.algorithm")),
        float(get_path(config, "train.learning_rate")),
        float(get_path(config, "train.entropy_coefficient")),
    )


def _expected_model_algorithm_cells() -> set[tuple[str, str, str, float, float]]:
    return {
        (model, revision, algorithm, learning_rate, entropy)
        for model, revision in MODEL_REVISIONS.items()
        for algorithm, (learning_rate, entropy) in ALGORITHM_SETTINGS.items()
    }


def _assert_common_resolved_settings(spec: RunSpec) -> None:
    config = spec.config
    assert get_path(config, "run.launch_guard") is None
    assert get_path(config, "run.protocol_unlocked") is None
    assert get_path(config, "run.device") == "cuda"
    assert get_path(config, "run.resume") is True
    assert get_path(config, "run.save_checkpoints") is True
    assert get_path(config, "data.n_train") == 10000
    assert get_path(config, "data.n_validation") == 1000
    assert get_path(config, "data.sage_rule_family") == "parity"
    assert get_path(config, "data.q_q") == 0.90
    assert get_path(config, "data.error_geometry") == "independent"
    assert get_path(config, "data.conflict_diversity") == "diverse"
    assert get_path(config, "data.training_view") == "full"
    assert get_path(config, "data.renderer") == "natural"
    assert get_path(config, "data.counterbalance") is True
    assert get_path(config, "model.dtype") == "bfloat16"
    assert get_path(config, "model.chat_template") is True
    assert get_path(config, "model.trust_remote_code") is False
    assert get_path(config, "model.action_labels") == ["A", "B"]
    assert get_path(config, "update.method") == "full"
    assert get_path(config, "train.batch_size") == 10
    assert get_path(config, "train.gradient_accumulation_steps") == 5
    assert get_path(config, "train.max_sequence_length") == 768
    assert get_path(config, "train.gradient_checkpointing") is True
    assert get_path(config, "train.deterministic_algorithms") is False
    assert get_path(config, "train.allow_tf32") is False
    assert get_path(config, "train.kl_coefficient") == 0.0
    assert get_path(config, "train.samples_per_prompt") == 4
    assert get_path(config, "evaluation.prompt_views") == [
        "full",
        "audit_law_matched",
    ]
    assert get_path(config, "evaluation.causal_prompt_views") == ["full"]
    assert get_path(config, "evaluation.mirror_pairs") is True
    assert get_path(config, "evaluation.save_predictions") is True
    model = str(get_path(config, "model.name"))
    assert get_path(config, "model.revision") == MODEL_REVISIONS[model]
    algorithm = str(get_path(config, "train.algorithm"))
    assert (
        get_path(config, "train.learning_rate"),
        get_path(config, "train.entropy_coefficient"),
    ) == ALGORITHM_SETTINGS[algorithm]


def test_qwen35_capability_pilot_is_exactly_four_unguarded_runs() -> None:
    config = _load("qwen35_known_law_pilot.yaml")
    plan = build_plan(config)

    assert get_path(config, "experiment.id") == "qwen35_known_law_pilot"
    assert get_path(config, "experiment.status") == "prospective"
    assert get_path(config, "run.seeds") == PILOT_SEEDS
    assert get_path(config, "run.launch_guard") is None
    assert get_path(config, "sweep") == {}
    assert len(config["cases"]) == 4
    assert len(plan) == 4
    assert len({spec.plan_key for spec in plan}) == 4
    assert {spec.seed for spec in plan} == set(PILOT_SEEDS)
    assert {_cell_identity(spec) for spec in plan} == _expected_model_algorithm_cells()

    for spec in plan:
        _assert_common_resolved_settings(spec)
        resolved = spec.config
        assert get_path(resolved, "data.rule_family") == "parity"
        assert get_path(resolved, "data.q_p") == 0.95
        assert get_path(resolved, "data.n_eval_per_cell") == 8
        assert get_path(resolved, "train.steps") == 512
        assert get_path(resolved, "train.eval_steps") == [0, 16, 64, 256, 512]
        assert get_path(resolved, "run.checkpoint_steps") == [256, 512]
        assert get_path(resolved, "run.snapshot_steps") == []
        assert get_path(resolved, "evaluation.causal_per_cell") == 4
        assert get_path(resolved, "evaluation.final_causal_per_cell") == 16
        assert get_path(resolved, "evaluation.final_eval_per_cell") == 64


def test_qwen35_main_panel_is_exactly_96_paired_unguarded_runs() -> None:
    config = _load("qwen35_known_law_main.yaml")
    plan = build_plan(config)

    assert get_path(config, "experiment.id") == "qwen35_known_law_main"
    assert get_path(config, "experiment.status") == "prospective"
    assert get_path(config, "run.seeds") == MAIN_SEEDS
    assert get_path(config, "run.launch_guard") is None
    assert get_path(config, "sweep") == {
        "data.rule_family": ["parity", "majority"],
        "data.q_p": [0.95, 1.00],
    }
    assert len(config["cases"]) == 4
    assert len(plan) == 96
    assert len({spec.plan_key for spec in plan}) == 96
    assert Counter(spec.seed for spec in plan) == Counter({seed: 16 for seed in MAIN_SEEDS})

    expected_cells = {
        (*model_algorithm, law, q_p)
        for model_algorithm, law, q_p in product(
            _expected_model_algorithm_cells(),
            ("parity", "majority"),
            (0.95, 1.00),
        )
    }
    observed = Counter(
        (
            *_cell_identity(spec),
            str(get_path(spec.config, "data.rule_family")),
            float(get_path(spec.config, "data.q_p")),
        )
        for spec in plan
    )
    assert set(observed) == expected_cells
    assert set(observed.values()) == {6}

    for spec in plan:
        _assert_common_resolved_settings(spec)
        resolved = spec.config
        assert get_path(resolved, "data.n_eval_per_cell") == 16
        assert get_path(resolved, "train.steps") == 1000
        assert get_path(resolved, "train.eval_steps") == [0, 1, 4, 16, 64, 256, 512, 1000]
        assert get_path(resolved, "run.checkpoint_steps") == [512, 1000]
        assert get_path(resolved, "run.snapshot_steps") == []
        assert get_path(resolved, "evaluation.causal_per_cell") == 8
        assert get_path(resolved, "evaluation.final_causal_per_cell") == 16
        assert get_path(resolved, "evaluation.final_eval_per_cell") == 64

    pilot_plan = build_plan(_load("qwen35_known_law_pilot.yaml"))
    assert {spec.seed for spec in pilot_plan}.isdisjoint({spec.seed for spec in plan})
