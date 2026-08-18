"""Strict configuration and stable plan identities for the finite hidden-law study."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml  # type: ignore[import-untyped]


class HiddenLawConfigError(ValueError):
    """Raised when a hidden-law configuration is incomplete or inconsistent."""


EXPECTED_SEEDS = (23011, 23013, 23017, 23019, 23023, 23031)
EXPECTED_MODEL_REVISIONS = {
    "Qwen/Qwen3.5-0.8B": "2fc06364715b967f1860aea9cf38778875588b17",
    "Qwen/Qwen3.5-2B": "15852e8c16360a2fea060d615a32b45270f8a8fc",
}
EXPECTED_ALGORITHMS = {
    "process_sft": (1e-5, 0.0),
    "outcome_rl": (3e-6, 0.01),
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _mapping(value: object, name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise HiddenLawConfigError(f"{name} must be a mapping")
    return cast(dict[str, Any], value)


def _exact_keys(value: dict[str, Any], keys: set[str], name: str) -> None:
    if set(value) != keys:
        missing = sorted(keys - set(value))
        extra = sorted(set(value) - keys)
        raise HiddenLawConfigError(f"{name} keys differ; missing={missing}, extra={extra}")


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HiddenLawConfigError(f"{name} must be a positive integer")
    return value


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HiddenLawConfigError(f"{name} must be numeric")
    result = float(value)
    if not 0.0 <= result < float("inf"):
        raise HiddenLawConfigError(f"{name} must be finite and non-negative")
    return result


@dataclass(frozen=True, slots=True)
class HiddenLawCondition:
    model_name: str
    model_revision: str
    model_dtype: str
    algorithm: str
    learning_rate: float
    entropy_coefficient: float
    seed: int
    config_digest: str

    @property
    def plan_key(self) -> str:
        return canonical_digest(self.as_obj())[:20]

    @property
    def run_id(self) -> str:
        return canonical_digest({"kind": "qwen35-hidden-law-run-v1", **self.as_obj()})[:20]

    def as_obj(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "model_dtype": self.model_dtype,
            "algorithm": self.algorithm,
            "learning_rate": self.learning_rate,
            "entropy_coefficient": self.entropy_coefficient,
            "seed": self.seed,
            "config_digest": self.config_digest,
        }


def load_hidden_law_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        value = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise HiddenLawConfigError(f"cannot load hidden-law config: {source}") from exc
    config = _mapping(value, "config")
    validate_hidden_law_config(config)
    return copy.deepcopy(config)


def scientific_config(config: dict[str, Any]) -> dict[str, Any]:
    """Remove only operational and prelaunch-smoke fields from run identity."""

    result = copy.deepcopy(config)
    result["experiment"].pop("status", None)
    result["run"].pop("output_root", None)
    result["run"].pop("smoke_seed", None)
    return result


def validate_hidden_law_config(config: dict[str, Any]) -> None:
    _exact_keys(
        config,
        {
            "schema",
            "schema_version",
            "experiment",
            "models",
            "algorithms",
            "seeds",
            "game",
            "training",
            "evaluation",
            "analysis",
            "run",
        },
        "config",
    )
    if config["schema"] != "goalzendo.hidden_law_finite_choice" or config["schema_version"] != 1:
        raise HiddenLawConfigError("unsupported hidden-law config schema")

    experiment = _mapping(config["experiment"], "experiment")
    _exact_keys(experiment, {"id", "status", "protocol"}, "experiment")
    if experiment["id"] != "qwen35_hidden_law_finite_choice":
        raise HiddenLawConfigError("wrong experiment id")
    if experiment["protocol"] != "docs/goalzendo/protocols/qwen35-hidden-law-finite-choice.md":
        raise HiddenLawConfigError("wrong protocol path")
    if experiment["status"] not in {"prospective_draft", "prospective_frozen"}:
        raise HiddenLawConfigError("experiment status must be prospective_draft or prospective_frozen")

    models = config["models"]
    if not isinstance(models, list) or len(models) != 2:
        raise HiddenLawConfigError("models must contain exactly two entries")
    observed_models: dict[str, str] = {}
    model_order: list[str] = []
    for index, raw in enumerate(models):
        model = _mapping(raw, f"models[{index}]")
        _exact_keys(model, {"name", "revision", "dtype", "trust_remote_code"}, f"models[{index}]")
        if model["dtype"] != "bfloat16" or model["trust_remote_code"] is not False:
            raise HiddenLawConfigError("models must use bfloat16 with trust_remote_code=false")
        model_order.append(str(model["name"]))
        observed_models[str(model["name"])] = str(model["revision"])
    if observed_models != EXPECTED_MODEL_REVISIONS:
        raise HiddenLawConfigError("model identities differ from the registered pair")
    if model_order != list(EXPECTED_MODEL_REVISIONS):
        raise HiddenLawConfigError("model order differs from the registered pair")

    algorithms = config["algorithms"]
    if not isinstance(algorithms, list) or len(algorithms) != 2:
        raise HiddenLawConfigError("algorithms must contain exactly two entries")
    observed_algorithms: dict[str, tuple[float, float]] = {}
    algorithm_order: list[str] = []
    for index, raw in enumerate(algorithms):
        algorithm = _mapping(raw, f"algorithms[{index}]")
        name = str(algorithm.get("name", ""))
        algorithm_order.append(name)
        common = {"name", "learning_rate", "entropy_coefficient"}
        expected = (
            common | {"phase_weights"}
            if name == "process_sft"
            else common
            | {
                "trajectories_per_official",
                "exact_rule_reward_weight",
                "classification_reward_weight",
            }
        )
        _exact_keys(algorithm, expected, f"algorithms[{index}]")
        observed_algorithms[name] = (
            _finite_nonnegative(algorithm["learning_rate"], "learning rate"),
            _finite_nonnegative(algorithm["entropy_coefficient"], "entropy coefficient"),
        )
        if name == "process_sft":
            phase_weights = _mapping(algorithm["phase_weights"], "process SFT phase weights")
            expected_phase_weights = {
                "inquiry": 1.0 / 3.0,
                "rule": 1.0 / 3.0,
                "classification": 1.0 / 3.0,
            }
            if phase_weights != expected_phase_weights or sum(phase_weights.values()) != 1.0:
                raise HiddenLawConfigError("process SFT phase weights must be equal thirds")
        if name == "outcome_rl":
            if algorithm["trajectories_per_official"] != 4:
                raise HiddenLawConfigError("outcome RL requires four trajectories per Official")
            weights = (
                float(algorithm["exact_rule_reward_weight"]),
                float(algorithm["classification_reward_weight"]),
            )
            if weights != (0.25, 0.75) or sum(weights) != 1.0:
                raise HiddenLawConfigError("outcome reward weights must be exactly .25/.75")
    if observed_algorithms != EXPECTED_ALGORITHMS:
        raise HiddenLawConfigError("algorithm identities differ from the registered pair")
    if algorithm_order != list(EXPECTED_ALGORITHMS):
        raise HiddenLawConfigError("algorithm order differs from the registered pair")

    if tuple(config["seeds"]) != EXPECTED_SEEDS:
        raise HiddenLawConfigError("seeds differ from the six registered paired seeds")

    game = _mapping(config["game"], "game")
    _exact_keys(
        game,
        {
            "candidate_count",
            "candidate_families",
            "opening_examples",
            "opening_positive",
            "query_menu_size",
            "max_queries",
            "query_partition_profile",
            "terminal_truth_cells",
            "train_renderers",
            "eval_renderers",
        },
        "game",
    )
    if game["candidate_count"] != 4 or game["opening_examples"] != 10 or game["opening_positive"] != 5:
        raise HiddenLawConfigError("game must use four candidates and a balanced ten-example opening")
    if game["query_menu_size"] != 8 or game["max_queries"] != 2 or game["terminal_truth_cells"] != 16:
        raise HiddenLawConfigError("game menu/query/terminal dimensions differ from the design")
    if game["candidate_families"] != [
        "placard_literal",
        "piece_literal",
        "monotone_composed",
        "exactly_one_composed",
    ]:
        raise HiddenLawConfigError("candidate-family order differs from the design")
    profile = _mapping(game["query_partition_profile"], "query_partition_profile")
    if profile != {"balanced_2v2": 4, "imbalanced_1v3": 4, "uninformative": 0}:
        raise HiddenLawConfigError("query partition profile differs from the design")
    if game["train_renderers"] != [
        "train_compact",
        "train_positional",
        "train_tabletop",
        "train_inventory",
    ] or game["eval_renderers"] != ["eval_reverse", "eval_ledger"]:
        raise HiddenLawConfigError("renderer families differ from the design")

    training = _mapping(config["training"], "training")
    if training.get("blocks") != 128 or training.get("steps") != 128:
        raise HiddenLawConfigError("training requires 128 role blocks and optimizer steps")
    if training.get("official_rotations_per_block") != 4:
        raise HiddenLawConfigError("every role block must contain four Official rotations")
    if training.get("evaluation_steps") != [0, 8, 32, 128]:
        raise HiddenLawConfigError("evaluation steps differ from the design")
    if training.get("checkpoint_steps") != [32, 128]:
        raise HiddenLawConfigError("checkpoint steps differ from the design")
    if training.get("max_prompt_tokens") != 1536:
        raise HiddenLawConfigError("max prompt length differs from the design")
    required_training = {
        "blocks",
        "official_rotations_per_block",
        "steps",
        "full_model_update",
        "optimizer",
        "weight_decay",
        "scheduler",
        "warmup_steps",
        "gradient_clip_norm",
        "gradient_checkpointing",
        "use_cache",
        "scoring_microbatch",
        "max_prompt_tokens",
        "deterministic_algorithms",
        "allow_tf32",
        "checkpoint_steps",
        "evaluation_steps",
        "resume",
    }
    _exact_keys(training, required_training, "training")
    expected_training_scalars = {
        "full_model_update": True,
        "optimizer": "adamw",
        "weight_decay": 0.0,
        "scheduler": "constant_after_linear_warmup",
        "warmup_steps": 7,
        "gradient_clip_norm": 1.0,
        "gradient_checkpointing": True,
        "use_cache": False,
        "scoring_microbatch": 4,
        "deterministic_algorithms": False,
        "allow_tf32": False,
        "resume": True,
    }
    if any(training[key] != value for key, value in expected_training_scalars.items()):
        raise HiddenLawConfigError("training runtime differs from the registered design")

    evaluation = _mapping(config["evaluation"], "evaluation")
    _exact_keys(
        evaluation,
        {
            "interim_quartets",
            "final_quartets",
            "evidence_cells",
            "final_views",
            "terminal_one_per_four_rule_truth_cell",
            "matched_interventions",
            "intervention_pairs_per_target_per_quartet",
            "intervention_atomic_edit",
            "isolated_terminal_calls",
            "save_transcripts",
            "save_predictions",
        },
        "evaluation",
    )
    if evaluation.get("interim_quartets") != 4 or evaluation.get("final_quartets") != 16:
        raise HiddenLawConfigError("evaluation quartet counts differ from the design")
    if evaluation.get("evidence_cells") != ["both_perfect", "p_noisy", "q_noisy", "both_noisy"]:
        raise HiddenLawConfigError("evidence cells differ from the design")
    if evaluation.get("final_views") != ["active", "oracle_query", "no_query"]:
        raise HiddenLawConfigError("final views differ from the design")
    if evaluation.get("matched_interventions") != [
        "official",
        "placard",
        "semantic",
        "other_composed",
        "distractor",
    ]:
        raise HiddenLawConfigError("matched interventions differ from the design")
    if (
        evaluation.get("intervention_pairs_per_target_per_quartet") != 2
        or evaluation.get("intervention_atomic_edit") != "one_primitive_field"
    ):
        raise HiddenLawConfigError("intervention construction differs from the design")
    if any(
        evaluation.get(key) is not True
        for key in (
            "terminal_one_per_four_rule_truth_cell",
            "isolated_terminal_calls",
            "save_transcripts",
            "save_predictions",
        )
    ):
        raise HiddenLawConfigError("evaluation recording and isolation must remain enabled")

    analysis = _mapping(config["analysis"], "analysis")
    _exact_keys(
        analysis,
        {
            "seed_is_inferential_unit",
            "primary_outcomes",
            "causal_companion",
            "exhaustive_seed_bootstrap",
            "exact_sign_flip",
            "binary_claims",
        },
        "analysis",
    )
    if analysis.get("primary_outcomes") != [
        "information_fraction",
        "exact_rule_recovery",
        "law_control_margin",
    ]:
        raise HiddenLawConfigError("primary outcomes differ from the design")
    if analysis.get("causal_companion") != "causal_law_control_margin":
        raise HiddenLawConfigError("wrong causal companion")
    if analysis.get("binary_claims") is not False or any(
        analysis.get(key) is not True
        for key in (
            "seed_is_inferential_unit",
            "exhaustive_seed_bootstrap",
            "exact_sign_flip",
        )
    ):
        raise HiddenLawConfigError("analysis status differs from the registered design")

    run = _mapping(config["run"], "run")
    _exact_keys(
        run,
        {"output_root", "smoke_seed", "scientific_pilot", "outcome_gate"},
        "run",
    )
    if type(run.get("output_root")) is not str or not run["output_root"]:
        raise HiddenLawConfigError("run output_root must be nonempty text")
    if (
        run.get("smoke_seed") != 23999
        or run.get("scientific_pilot") is not False
        or run.get("outcome_gate") is not False
    ):
        raise HiddenLawConfigError("run section must retain a non-scientific smoke and no outcome gate")


def build_hidden_law_plan(config: dict[str, Any]) -> tuple[HiddenLawCondition, ...]:
    validate_hidden_law_config(config)
    digest = canonical_digest(scientific_config(config))
    conditions: list[HiddenLawCondition] = []
    for raw_model in config["models"]:
        model = cast(dict[str, Any], raw_model)
        for raw_algorithm in config["algorithms"]:
            algorithm = cast(dict[str, Any], raw_algorithm)
            for seed in config["seeds"]:
                conditions.append(
                    HiddenLawCondition(
                        model_name=str(model["name"]),
                        model_revision=str(model["revision"]),
                        model_dtype=str(model["dtype"]),
                        algorithm=str(algorithm["name"]),
                        learning_rate=float(algorithm["learning_rate"]),
                        entropy_coefficient=float(algorithm["entropy_coefficient"]),
                        seed=int(seed),
                        config_digest=digest,
                    )
                )
    if len(conditions) != 24 or len({condition.plan_key for condition in conditions}) != 24:
        raise HiddenLawConfigError("registered plan must contain exactly 24 unique conditions")
    return tuple(conditions)
