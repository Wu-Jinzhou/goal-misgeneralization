from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from goalzendo.config import get_path, load_config
from goalzendo.experiment import ExperimentBanks, materialize_banks
from goalzendo.runner import RunSpec, build_plan
from goalzendo.schema import stable_digest

ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "configs" / "goalzendo"

SEEDS = [22011, 22013, 22017, 22019, 22023, 22031]
MODEL_REVISIONS = {
    "Qwen/Qwen3.5-0.8B": "2fc06364715b967f1860aea9cf38778875588b17",
    "Qwen/Qwen3.5-2B": "15852e8c16360a2fea060d615a32b45270f8a8fc",
}
EVIDENCE_CELLS = {
    (0.005, "diverse", None),
    (0.040, "diverse", None),
    (0.040, "concentrated", 16),
}
CONFLICT_TUPLES = {"AAB", "ABA", "ABB", "BAA", "BAB", "BBA"}
EXPECTED_TRAIN_COUNTS = {
    0.005: {
        "AAA": 4275,
        "AAB": 475,
        "ABA": 225,
        "ABB": 25,
        "BAA": 25,
        "BAB": 225,
        "BBA": 475,
        "BBB": 4275,
    },
    0.040: {
        "AAA": 4450,
        "AAB": 300,
        "ABA": 50,
        "ABB": 200,
        "BAA": 200,
        "BAB": 50,
        "BBA": 300,
        "BBB": 4450,
    },
}

# Filled from the prospective symbolic construction before any model launch.
# Each digest covers metadata plus every ordered effective-training, factorial,
# causal-base, and causal-intervention decision for seed 22011.
EXPECTED_BANK_DIGESTS = {
    (0.005, "diverse"): ("0a199e68bf9ef6a515463ec6901c8d6cd19e39ed36b838a41617a3494e6119ad"),
    (0.040, "diverse"): ("3e96a646aa6b5608b5824ed9c068220e49d8bbbd4bc9918c021f2f938452cce9"),
    (0.040, "concentrated"): ("f47d2cf5339beb8dc0cba8f719447547a0316e5b1aa1aa704e85ba44d2cd9064"),
}

MATCHED_MAIN_PATHS = (
    "run.resume",
    "run.save_checkpoints",
    "run.checkpoint_steps",
    "run.snapshot_steps",
    "data.n_train",
    "data.n_validation",
    "data.n_eval_per_cell",
    "data.sage_rule_family",
    "data.q_q",
    "data.training_view",
    "data.renderer",
    "data.counterbalance",
    "model.dtype",
    "model.chat_template",
    "model.trust_remote_code",
    "model.action_labels",
    "update.method",
    "train.algorithm",
    "train.steps",
    "train.batch_size",
    "train.gradient_accumulation_steps",
    "train.learning_rate",
    "train.weight_decay",
    "train.warmup_ratio",
    "train.grad_clip",
    "train.max_sequence_length",
    "train.gradient_checkpointing",
    "train.parameter_finite_check_interval",
    "train.deterministic_algorithms",
    "train.allow_tf32",
    "train.eval_steps",
    "train.kl_coefficient",
    "train.entropy_coefficient",
    "train.samples_per_prompt",
    "evaluation.prompt_views",
    "evaluation.causal_prompt_views",
    "evaluation.causal_per_cell",
    "evaluation.final_causal_per_cell",
    "evaluation.final_eval_per_cell",
    "evaluation.mirror_pairs",
    "evaluation.save_predictions",
)


def _load(name: str) -> dict[str, Any]:
    return load_config(CONFIG_ROOT / name)


def _cell(spec: RunSpec) -> tuple[float, str, int | None]:
    return (
        float(get_path(spec.config, "data.joint_error_rate")),
        str(get_path(spec.config, "data.conflict_diversity")),
        get_path(spec.config, "data.concentrated_unique_conflicts_per_tuple"),
    )


def _tuple_key(decision: Any) -> str:
    return "".join(choice.label for choice in decision.candidate_tuple)


def _ordered_decision_digests(values: Any) -> list[str]:
    return [decision.digest for decision in values]


def _bank_contract_digest(banks: ExperimentBanks) -> str:
    return stable_digest(
        {
            "metadata": banks.metadata,
            "effective_train": _ordered_decision_digests(banks.effective_train_decisions),
            "diagnostic_causal_base": _ordered_decision_digests(banks.diagnostic_causal_base),
            "final_causal_base": _ordered_decision_digests(banks.final_causal_base),
            "diagnostic_interventions": {
                target: _ordered_decision_digests(values)
                for target, values in sorted(banks.diagnostic_interventions.items())
            },
            "final_interventions": {
                target: _ordered_decision_digests(values)
                for target, values in sorted(banks.final_interventions.items())
            },
        }
    )


def test_evidence_geometry_plan_is_exactly_36_fresh_paired_runs() -> None:
    config = _load("qwen35_evidence_geometry.yaml")
    plan = build_plan(config)

    assert get_path(config, "experiment.id") == "qwen35_evidence_geometry"
    assert get_path(config, "experiment.status") == "prospective"
    assert get_path(config, "run.seeds") == SEEDS
    assert get_path(config, "run.launch_guard") is None
    assert get_path(config, "run.protocol_unlocked") is None
    assert get_path(config, "sweep") == {}
    assert len(config["cases"]) == 6
    assert len(plan) == 36
    assert len({spec.plan_key for spec in plan}) == 36
    assert Counter(spec.seed for spec in plan) == Counter({seed: 6 for seed in SEEDS})

    observed = Counter(
        (
            str(get_path(spec.config, "model.name")),
            str(get_path(spec.config, "model.revision")),
            *_cell(spec),
        )
        for spec in plan
    )
    expected = {
        (model, revision, *cell) for model, revision in MODEL_REVISIONS.items() for cell in EVIDENCE_CELLS
    }
    assert set(observed) == expected
    assert set(observed.values()) == {6}

    old_seeds = {
        spec.seed
        for name in ("qwen35_known_law_pilot.yaml", "qwen35_known_law_main.yaml")
        for spec in build_plan(_load(name))
    }
    assert set(SEEDS).isdisjoint(old_seeds)

    main_plan = build_plan(_load("qwen35_known_law_main.yaml"))
    for spec in plan:
        assert get_path(spec.config, "run.device") == "cuda"
        assert get_path(spec.config, "data.rule_family") == "parity"
        assert get_path(spec.config, "data.q_p") == 0.95
        assert get_path(spec.config, "data.error_geometry") == "specified"
        assert get_path(spec.config, "train.algorithm") == "outcome_rl"
        assert get_path(spec.config, "update.method") == "full"
        model = str(get_path(spec.config, "model.name"))
        assert get_path(spec.config, "model.revision") == MODEL_REVISIONS[model]

        matching_main = next(
            candidate
            for candidate in main_plan
            if candidate.seed == 21011
            and get_path(candidate.config, "model.name") == model
            and get_path(candidate.config, "train.algorithm") == "outcome_rl"
            and get_path(candidate.config, "data.rule_family") == "parity"
            and get_path(candidate.config, "data.q_p") == 0.95
        )
        for path in MATCHED_MAIN_PATHS:
            assert get_path(spec.config, path) == get_path(matching_main.config, path), path

    # Model size is paired over an identical, outcome-blind data construction.
    for seed in SEEDS:
        by_cell: dict[tuple[float, str, int | None], list[RunSpec]] = {}
        for spec in plan:
            if spec.seed == seed:
                by_cell.setdefault(_cell(spec), []).append(spec)
        assert set(by_cell) == EVIDENCE_CELLS
        for paired_specs in by_cell.values():
            assert len(paired_specs) == 2
            left, right = paired_specs
            assert left.seeds == right.seeds
            assert get_path(left.config, "data") == get_path(right.config, "data")


def test_registered_seed_22011_banks_have_exact_outcome_blind_identity() -> None:
    plan = build_plan(_load("qwen35_evidence_geometry.yaml"))
    representatives = {
        _cell(spec): spec
        for spec in plan
        if spec.seed == 22011 and get_path(spec.config, "model.name") == "Qwen/Qwen3.5-0.8B"
    }
    assert set(representatives) == EVIDENCE_CELLS

    for (joint_rate, diversity, unique_per_tuple), spec in representatives.items():
        banks = materialize_banks(spec.config, spec.seeds)
        expected_counts = EXPECTED_TRAIN_COUNTS[joint_rate]
        assert banks.metadata["banks"]["train"]["candidate_counts"] == expected_counts
        assert banks.metadata["banks"]["train"]["realized_q_p"] == 0.95
        assert banks.metadata["banks"]["train"]["realized_q_q"] == 0.90
        assert banks.metadata["banks"]["train"]["realized_joint_error_rate"] == joint_rate
        assert Counter(_tuple_key(item) for item in banks.effective_train_decisions) == Counter(
            expected_counts
        )

        unique_by_tuple = {
            key: len(
                {
                    decision.digest
                    for decision in banks.effective_train_decisions
                    if _tuple_key(decision) == key
                }
            )
            for key in expected_counts
        }
        effective = banks.metadata["effective_training"]
        if diversity == "diverse":
            assert unique_per_tuple is None
            assert unique_by_tuple == expected_counts
            assert effective.get("requested_unique_conflicts_per_tuple") is None
        else:
            assert unique_per_tuple == 16
            assert {key: unique_by_tuple[key] for key in CONFLICT_TUPLES} == {
                key: 16 for key in CONFLICT_TUPLES
            }
            assert unique_by_tuple["AAA"] == expected_counts["AAA"]
            assert unique_by_tuple["BBB"] == expected_counts["BBB"]
            assert effective["requested_unique_conflicts_per_tuple"] == 16
            assert effective["unique_conflict_by_candidate_tuple"] == {
                key: 16 for key in sorted(CONFLICT_TUPLES)
            }

        for name in ("diagnostic_factorial", "final_factorial"):
            per_cell = 16 if name == "diagnostic_factorial" else 64
            assert banks.metadata["banks"][name]["candidate_counts"] == {
                key: per_cell for key in sorted(EXPECTED_TRAIN_COUNTS[0.005])
            }
        assert all(
            overlap == 0 for overlap in banks.metadata["semantic_scene_audit"]["cross_bank_overlap"].values()
        )
        assert banks.metadata["semantic_scene_audit"]["intervention_cross_overlap"] == 0
        assert banks.metadata["semantic_scene_audit"]["intervention_base_overlap"] == 0
        assert _bank_contract_digest(banks) == EXPECTED_BANK_DIGESTS[(joint_rate, diversity)]
