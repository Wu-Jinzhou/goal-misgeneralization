from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter
from dataclasses import replace
from itertools import product
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest

from goalzendo_hidden_law.artifacts import HiddenLawRunStore
from goalzendo_hidden_law.config import build_hidden_law_plan, load_hidden_law_config
from goalzendo_hidden_law.experiment import QueryObservation, reference_inquiry
from goalzendo_hidden_law.game import GameInstance, ProductionBank, build_production_bank, query_information
from goalzendo_interactive.rules import BinaryRule
from goalzendo_interactive.schema import scene_at

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "analyze_qwen35_hidden_law.py"
CONFIG = ROOT / "configs" / "goalzendo" / "qwen35_hidden_law_finite_choice.yaml"
FINGERPRINT = "f" * 64


def _load_script() -> ModuleType:
    specification = importlib.util.spec_from_file_location("analyze_qwen35_hidden_law_for_test", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


analyzer = _load_script()


def _model_manifest(condition: Any) -> dict[str, Any]:
    return {
        "requested_model": condition.model_name,
        "requested_revision": condition.model_revision,
        "resolved_revision": None,
        "tokenizer_resolved_revision": None,
        "requested_dtype": condition.model_dtype,
        "model_class": "transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5ForCausalLM",
        "tokenizer_class": "transformers.tokenization_utils_tokenizers.TokenizersBackend",
        "tokenizer_name_or_path": condition.model_name,
        "action_labels": ["A", "B"],
        "action_token_ids": [[101], [102]],
        "parameter_count": analyzer.MODEL_PARAMETER_COUNTS[condition.model_name],
        "trainable_parameter_count": analyzer.MODEL_PARAMETER_COUNTS[condition.model_name],
        "trainable_parameter_dtype_counts": {
            "torch.bfloat16": analyzer.MODEL_PARAMETER_COUNTS[condition.model_name]
        },
        "dependency_versions": {
            "torch": "2.8.0+cu128",
            "torch_cuda": "12.8",
            **analyzer.PINNED_DEPENDENCIES,
        },
        "finite_action_alphabets": {
            "binary": ["A", "B"],
            "candidate": ["A", "B", "C", "D"],
            "query": ["A", "B", "C", "D", "E", "F", "G", "H", "I"],
        },
        "gradient_checkpointing": True,
        "use_cache": False,
        "full_model_update": True,
    }


def _query_records(
    instance: GameInstance,
    observations: tuple[QueryObservation, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    live = instance.initial_live_ids
    observation_rows: list[dict[str, Any]] = []
    turn_rows: list[dict[str, Any]] = []
    for turn, observation in enumerate(observations, start=1):
        information = query_information(instance, observation.option_id, live)
        after = information.accepted_ids if observation.accepted else information.rejected_ids
        observation_rows.append({"option_id": observation.option_id, "accepted": observation.accepted})
        turn_rows.append(
            {
                "turn": turn,
                "option_id": observation.option_id,
                "accepted": observation.accepted,
                "before_count": len(live),
                "after_count": len(after),
                "expected_information_bits": information.expected_information_bits,
                "best_expected_information_bits": information.best_expected_information_bits,
                "regret_bits": information.regret_bits,
                "realized_information_bits": math.log2(len(live)) - math.log2(len(after)),
            }
        )
        live = after
    return observation_rows, turn_rows, len(live)


def _episode(
    *,
    seed: int,
    algorithm: str,
    step: int,
    view: str,
    family_index: int,
    p_evidence: str,
    q_evidence: str,
    bank: ProductionBank,
) -> tuple[dict[str, Any], dict[str, Any]]:
    assert bank.seed == seed
    family = bank.evaluation_families[family_index]
    y_role = family.evaluation_y_role
    r_role = family.evaluation_r_role
    candidates = {
        "Y": family.candidate_by_role[y_role],
        "P": family.candidate_by_role["P"],
        "Q": family.candidate_by_role["Q"],
        "R": family.candidate_by_role[r_role],
    }
    ids = {role: candidate.candidate_id for role, candidate in candidates.items()}
    rules = {role: candidate.rule.as_obj() for role, candidate in candidates.items()}
    y_family = "monotone" if y_role == "M" else "exactly_one"
    monotone_operator = cast(BinaryRule, family.candidate_by_role["M"].rule).op
    family_id = family.family_id
    condition_id = f"p-{p_evidence}_q-{q_evidence}"
    instance = next(
        game for game in family.evaluation_games if game.instance_id == f"{family_id}-eval-{condition_id}"
    )
    instance_id = instance.instance_id
    if view == "oracle_query" or (view == "active" and algorithm == "process_sft"):
        controller = "Y"
        requested_observations = reference_inquiry(instance)
    elif view == "active":
        controller = "P"
        requested_observations = ()
    else:
        controller = "Q"
        requested_observations = ()

    truths = {
        role: [candidate.rule.evaluate(scene_at(index)) for index in instance.material.terminal]
        for role, candidate in candidates.items()
    }
    predictions = list(truths[controller])
    conflict = [
        index for index in range(16) if len({truths[role][index] for role in analyzer.ANALYSIS_ROLES}) > 1
    ]
    agreement = {
        role: sum(predictions[index] is truths[role][index] for index in conflict) / 14
        for role in analyzer.ANALYSIS_ROLES
    }
    law_margin = agreement["Y"] - max(agreement[role] for role in ("P", "Q", "R"))
    start = len(instance.initial_live_ids)
    observations, query_turns, end = _query_records(instance, requested_observations)
    information = math.log2(start / end) / math.log2(start)

    final = step == analyzer.FINAL_STEP
    interventions: list[dict[str, Any]] = []
    flip_rates: dict[str, float] = {}
    if final:
        for record in family.matched_interventions:
            target_flips = record.target == controller and record.target != "distractor"
            interventions.append(
                {
                    **record.as_obj(),
                    "before_prediction": False,
                    "after_prediction": target_flips,
                    "hard_flip": target_flips,
                    "before_fit_probability": 0.25,
                    "after_fit_probability": 0.75 if target_flips else 0.25,
                }
            )
        flip_rates = {
            target: float(target == controller and target != "distractor")
            for target in analyzer.INTERVENTION_TARGETS
        }
        causal = flip_rates["Y"] - max(flip_rates[role] for role in ("P", "Q", "R"))
    else:
        causal = None

    renderer = "eval_reverse" if family_index % 2 == 0 else "eval_ledger"
    prefix = f"eval:{step:04d}:{view}:{instance_id}"
    common = {
        "step": step,
        "view": view,
        "family_id": family_id,
        "instance_id": instance_id,
        "renderer": renderer,
        "condition_id": condition_id,
        "p_evidence": p_evidence,
        "q_evidence": q_evidence,
        "y_rule_family": y_family,
        "monotone_operator": monotone_operator,
        "analysis_candidate_ids": ids,
    }
    metric = {
        "record_id": f"{prefix}:metrics",
        "kind": "evaluation_episode",
        **common,
        "initial_live_count": start,
        "final_live_count": end,
        "query_count": len(observations),
        "declared_ready": False,
        "information_fraction": information,
        "query_turns": query_turns,
        "selected_candidate_id": ids[controller],
        "official_candidate_id": ids["Y"],
        "exact_rule_recovery": controller == "Y",
        "terminal_classification_accuracy_all_16": sum(
            left is right for left, right in zip(predictions, truths["Y"], strict=True)
        )
        / 16,
        "agreement": agreement,
        **{f"rho_{role}": agreement[role] for role in analyzer.ANALYSIS_ROLES},
        "law_control_margin": law_margin,
        "intervention_flip_rates": flip_rates,
        **{f"flip_{target}": flip_rates.get(target) for target in analyzer.INTERVENTION_TARGETS},
        "causal_law_control_margin": causal,
    }
    transcript = {
        "record_id": f"{prefix}:transcript",
        "kind": "evaluation_transcript",
        **common,
        "opening": [item.as_obj() for item in instance.material.opening],
        "query_observations": observations,
        "decisions": (
            [{"kind": "query", "selected_label": "A"} for _ in observations] if view == "active" else []
        ),
        "terminal_scene_indices": list(instance.material.terminal),
        "terminal_predictions": predictions,
        "terminal_truths": truths,
        "candidate_rules": rules,
        "interventions": interventions,
    }
    return metric, transcript


def _aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(key: str) -> float:
        return sum(float(row[key]) for row in rows) / len(rows)

    return {
        "episode_count": len(rows),
        "information_fraction": mean("information_fraction"),
        "exact_rule_recovery": sum(bool(row["exact_rule_recovery"]) for row in rows) / len(rows),
        "terminal_classification_accuracy_all_16": mean("terminal_classification_accuracy_all_16"),
        "law_control_margin": mean("law_control_margin"),
        "causal_law_control_margin": (
            mean("causal_law_control_margin")
            if all(row["causal_law_control_margin"] is not None for row in rows)
            else None
        ),
    }


def _synthetic_runs() -> tuple[list[Any], dict[str, Any]]:
    config = load_hidden_law_config(CONFIG)
    implementation = {"implementation_fingerprint": FINGERPRINT, "source_files": []}
    runs: list[Any] = []
    banks = {seed: build_production_bank(seed) for seed in analyzer.SEEDS}
    for condition in build_hidden_law_plan(config):
        production_bank = banks[condition.seed]
        metrics: list[dict[str, Any]] = []
        transcripts: list[dict[str, Any]] = []
        for step in range(1, analyzer.FINAL_STEP + 1):
            metrics.append(
                {
                    "record_id": f"train:{step:04d}:block",
                    "kind": "optimization_block",
                    "step": step,
                    "algorithm": condition.algorithm,
                    "loss": 1.0,
                    "learning_rate": condition.learning_rate,
                    "gradient_norm": 1.0,
                    "rotation_metrics": [{"loss": 1.0}] * 4,
                }
            )
            count = 4 if condition.algorithm == "process_sft" else 16
            for index in range(count):
                transcripts.append(
                    {
                        "record_id": f"train:{step:04d}:trajectory:{index:03d}",
                        "kind": "training_trajectory",
                        "step": step,
                        "algorithm": condition.algorithm,
                        "trajectory": {"instance_id": f"train-{step}-{index}"},
                    }
                )

        for step in analyzer.EVAL_STEPS:
            family_indices = range(16) if step == analyzer.FINAL_STEP else (0, 5, 10, 15)
            views = analyzer.FINAL_VIEWS if step == analyzer.FINAL_STEP else ("active",)
            checkpoint_rows: list[dict[str, Any]] = []
            for family_index, (p_evidence, q_evidence), view in product(
                family_indices, analyzer.EVIDENCE_CELLS, views
            ):
                metric, transcript = _episode(
                    seed=condition.seed,
                    algorithm=condition.algorithm,
                    step=step,
                    view=view,
                    family_index=family_index,
                    p_evidence=p_evidence,
                    q_evidence=q_evidence,
                    bank=production_bank,
                )
                metrics.append(metric)
                transcripts.append(transcript)
                checkpoint_rows.append(metric)
            by_view = {
                view: _aggregate_metrics([row for row in checkpoint_rows if row["view"] == view])
                for view in views
            }
            checkpoint_summary = {
                "record_id": f"eval:{step:04d}:checkpoint-summary",
                "kind": "evaluation_checkpoint_summary",
                "step": step,
                "final": step == analyzer.FINAL_STEP,
                "quartet_count": len(tuple(family_indices)),
                "episode_count": len(checkpoint_rows),
                "views": by_view,
            }
            metrics.append(checkpoint_summary)
            if step == analyzer.FINAL_STEP:
                latest_evaluation = {
                    key: value
                    for key, value in checkpoint_summary.items()
                    if key not in {"record_id", "kind"}
                }

        bank = {
            "schema": "goalzendo.hidden_law_bank_manifest",
            "schema_version": 1,
            "bank_digest": production_bank.digest,
            "pairing_key": production_bank.pairing_key,
            "manifest": production_bank.manifest,
        }
        summary = {
            "schema": "goalzendo.hidden_law_run_summary",
            "schema_version": 1,
            "run_id": condition.run_id,
            "plan_key": condition.plan_key,
            "smoke": False,
            "last_step": analyzer.FINAL_STEP,
            "algorithm": condition.algorithm,
            "model_name": condition.model_name,
            "seed": condition.seed,
            "bank_digest": production_bank.digest,
            "evaluation": latest_evaluation,
            "operational": {
                "device": "synthetic",
                "forward_calls": 1,
                "scored_prompt_count": 1,
                "scored_prompt_tokens_unpadded": 1,
                "token_count_definition": analyzer.TOKEN_COUNT_DEFINITION,
                "maximum_prompt_tokens": 100,
            },
        }
        runs.append(
            analyzer.LoadedRun(
                condition=condition,
                path=Path("/synthetic") / condition.run_id,
                identity={},
                implementation=implementation,
                bank_manifest=bank,
                model_manifest=_model_manifest(condition),
                summary=summary,
                metrics=tuple(metrics),
                transcripts=tuple(transcripts),
            )
        )
    return runs, config


def test_exact_six_block_inference_and_canonical_json_are_deterministic() -> None:
    values = {seed: 0.25 for seed in analyzer.SEEDS}
    first = analyzer._summarize_seed_blocks(values)
    second = analyzer._summarize_seed_blocks(values)
    assert first == second
    assert first["bootstrap_95_percentile_interval"] == {"lower": 0.25, "upper": 0.25}
    assert first["descriptive_unadjusted_two_sided_sign_flip_p_value"] == 2 / 64
    assert analyzer._sign_flip_p_value([0.0] * 6) == 1.0
    assert analyzer.BOOTSTRAP_RESAMPLES == 6**6
    assert analyzer.SIGN_FLIP_ASSIGNMENTS == 2**6


def test_complete_synthetic_panel_reports_registered_contrasts_and_raw_records() -> None:
    runs, config = _synthetic_runs()
    report = analyzer.analyze_runs(
        runs,
        config,
        config_file_sha256="synthetic-config",
        implementation_fingerprint=FINGERPRINT,
    )
    assert report["panel"]["observed_run_count"] == 24
    assert len(report["seed_endpoints"]) == 24
    assert len(report["seed_trajectories"]) == 24
    assert len(report["candidate_formula_registry"]) == 24
    assert Counter(row["seed"] for row in report["seed_endpoints"]) == Counter(
        {seed: 4 for seed in analyzer.SEEDS}
    )
    for endpoint in report["seed_endpoints"]:
        assert len(endpoint["final_active_truth_table"]) == 16
        assert {row["n"] for row in endpoint["final_active_truth_table"]} == {64}
        assert set(endpoint["condition_cells"]) == {
            "both_perfect",
            "p_noisy",
            "q_noisy",
            "both_noisy",
        }
        assert set(endpoint["operational"]) == {
            "device",
            "forward_calls",
            "scored_prompt_count",
            "scored_prompt_tokens_unpadded",
            "maximum_prompt_tokens",
            "token_count_definition",
        }
        assert "named_proxy_only_margin" in endpoint["active"]
        assert "stated_rule_behavioral_agreement" in endpoint["active"]
        assert len(endpoint["active"]["query_turns"]) == 2
    for trajectory in report["seed_trajectories"]:
        assert [row["step"] for row in trajectory["checkpoints"]] == list(analyzer.EVAL_STEPS)

    primary = report["primary_sft_minus_outcome_rl"]
    for model in primary["per_model"]:
        effects = model["sft_minus_outcome_rl"]
        assert effects["information_fraction"]["mean"] == pytest.approx(1.0)
        assert effects["exact_rule_recovery"]["mean"] == pytest.approx(1.0)
        assert effects["law_control_margin"]["mean"] == pytest.approx(8 / 7)
        assert effects["causal_law_control_margin"]["mean"] == pytest.approx(2.0)
    pooled = primary["pooled_equal_weight_across_models_within_seed"]
    assert pooled["law_control_margin"]["mean"] == pytest.approx(8 / 7)

    secondary = report["registered_secondary"]
    interactions = secondary["p_q_evidence_factorial_on_law_control_margin"][
        "sft_minus_outcome_rl_interactions"
    ]["pooled_equal_weight_across_models_within_seed"]
    assert all(interactions[name]["mean"] == pytest.approx(0.0) for name in interactions)
    assert len(secondary["final_view_gaps"]) == 4
    assert len(secondary["formula_family_descriptions"]) == 4
    assert len(secondary["model_size_descriptions"]) == 2

    descriptives = report["registered_descriptives"]
    assert descriptives["status"] == "fixed_descriptive_only_no_additional_inference"
    operational = descriptives["operational_counts"]
    assert len(operational["per_run"]) == 24
    assert len(operational["model_by_algorithm_descriptions"]) == 4
    inquiry = descriptives["active_inquiry_by_checkpoint"]
    assert len(inquiry["per_run"]) == 24
    assert all(len(row["checkpoints"]) == 4 for row in inquiry["per_run"])
    sft_final = next(
        row["checkpoints"][-1] for row in inquiry["per_run"] if row["algorithm"] == "process_sft"
    )
    rl_final = next(row["checkpoints"][-1] for row in inquiry["per_run"] if row["algorithm"] == "outcome_rl")
    assert sft_final["query_turns"][0]["turn_conditioned_denominator"] == 64
    assert rl_final["query_turns"][0]["turn_conditioned_denominator"] == 0
    assert len(descriptives["stated_rule_and_behavioral_agreement"]["per_run_final_active"]) == 24

    serialized = analyzer.canonical_json(report)
    assert serialized.endswith("\n") and serialized.count("\n") == 1
    assert json.loads(serialized) == report
    assert "significant" not in serialized and "reject_null" not in serialized
    assert report["inference"]["p_values"]["binary_claims"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_run",
        "bank_pair",
        "source",
        "model_manifest",
        "episode",
        "truth_table",
        "evidence_cell",
        "condition_game",
        "query_replay",
        "oracle_path",
    ],
)
def test_incomplete_or_inconsistent_panels_fail_closed(mutation: str) -> None:
    runs, config = _synthetic_runs()
    if mutation == "missing_run":
        runs.pop()
    elif mutation == "bank_pair":
        bank = dict(runs[0].bank_manifest)
        bank["bank_digest"] = "0" * 64
        runs[0] = replace(runs[0], bank_manifest=bank)
    elif mutation == "source":
        runs[0] = replace(
            runs[0],
            implementation={"implementation_fingerprint": "0" * 64, "source_files": []},
        )
    elif mutation == "model_manifest":
        runs[0] = replace(runs[0], model_manifest={})
    elif mutation == "episode":
        metrics = list(runs[0].metrics)
        metrics.pop(next(index for index, row in enumerate(metrics) if row["kind"] == "evaluation_episode"))
        runs[0] = replace(runs[0], metrics=tuple(metrics))
    elif mutation == "truth_table":
        transcripts = [copy.deepcopy(dict(row)) for row in runs[0].transcripts]
        row = next(item for item in transcripts if item["kind"] == "evaluation_transcript")
        row["terminal_truths"]["Y"] = list(row["terminal_truths"]["P"])
        runs[0] = replace(runs[0], transcripts=tuple(transcripts))
    elif mutation == "evidence_cell":
        metrics = [copy.deepcopy(dict(row)) for row in runs[0].metrics]
        transcripts = [copy.deepcopy(dict(row)) for row in runs[0].transcripts]
        row = next(
            item
            for item in metrics
            if item.get("kind") == "evaluation_episode"
            and item["step"] == 0
            and item["p_evidence"] == "perfect"
            and item["q_evidence"] == "noisy"
        )
        for item in (
            row,
            next(item for item in transcripts if item["record_id"][:-11] == row["record_id"][:-8]),
        ):
            item["p_evidence"] = "noisy"
            item["q_evidence"] = "perfect"
            item["condition_id"] = "p-noisy_q-perfect"
        runs[0] = replace(runs[0], metrics=tuple(metrics), transcripts=tuple(transcripts))
    elif mutation == "condition_game":
        metrics = [copy.deepcopy(dict(row)) for row in runs[0].metrics]
        transcripts = [copy.deepcopy(dict(row)) for row in runs[0].transcripts]
        episode_rows = [
            item
            for item in metrics
            if item.get("kind") == "evaluation_episode"
            and item["step"] == 0
            and item["family_id"]
            == next(
                candidate["family_id"]
                for candidate in metrics
                if candidate.get("kind") == "evaluation_episode" and candidate["step"] == 0
            )
        ]
        first_id, second_id = episode_rows[0]["instance_id"], episode_rows[1]["instance_id"]
        episode_rows[0]["instance_id"], episode_rows[1]["instance_id"] = second_id, first_id
        for item in transcripts:
            if item.get("kind") != "evaluation_transcript" or item.get("step") != 0:
                continue
            if item["instance_id"] == first_id:
                item["instance_id"] = second_id
            elif item["instance_id"] == second_id:
                item["instance_id"] = first_id
        runs[0] = replace(runs[0], metrics=tuple(metrics), transcripts=tuple(transcripts))
    elif mutation == "query_replay":
        metrics = [copy.deepcopy(dict(row)) for row in runs[0].metrics]
        row = next(
            item
            for item in metrics
            if item.get("kind") == "evaluation_episode"
            and item["view"] == "active"
            and item["query_count"] > 0
        )
        row["query_turns"][0]["expected_information_bits"] -= 0.125
        row["query_turns"][0]["best_expected_information_bits"] -= 0.125
        runs[0] = replace(runs[0], metrics=tuple(metrics))
    else:
        metrics = [copy.deepcopy(dict(row)) for row in runs[0].metrics]
        transcripts = [copy.deepcopy(dict(row)) for row in runs[0].transcripts]
        row = next(
            item
            for item in metrics
            if item.get("kind") == "evaluation_episode" and item["view"] == "oracle_query"
        )
        row["final_live_count"] = row["initial_live_count"]
        row["query_count"] = 0
        row["information_fraction"] = 0.0
        row["query_turns"] = []
        transcript = next(
            item
            for item in transcripts
            if item.get("kind") == "evaluation_transcript"
            and item["step"] == row["step"]
            and item["view"] == row["view"]
            and item["instance_id"] == row["instance_id"]
        )
        transcript["query_observations"] = []
        runs[0] = replace(runs[0], metrics=tuple(metrics), transcripts=tuple(transcripts))
    with pytest.raises(analyzer.Qwen35HiddenLawAnalysisError):
        analyzer.analyze_runs(
            runs,
            config,
            config_file_sha256="synthetic-config",
            implementation_fingerprint=FINGERPRINT,
        )


def _minimal_sealed_panel(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    config = load_hidden_law_config(CONFIG)
    output = tmp_path / "sealed"
    implementation = {"implementation_fingerprint": FINGERPRINT}
    for condition in build_hidden_law_plan(config):
        store = HiddenLawRunStore(output, condition, config, implementation)
        assert store.initialize() == "new"
        digest = hashlib.sha256(f"bank-{condition.seed}".encode()).hexdigest()
        store.write_bank_manifest(
            {
                "schema": "goalzendo.hidden_law_bank_manifest",
                "schema_version": 1,
                "bank_digest": digest,
                "pairing_key": hashlib.sha256(f"pair-{condition.seed}".encode()).hexdigest(),
                "manifest": {"seed": condition.seed},
            }
        )
        store.write_model_manifest(_model_manifest(condition))
        store.finish(
            {
                "run_id": condition.run_id,
                "last_step": analyzer.FINAL_STEP,
                "bank_digest": digest,
            }
        )
    return output, config


def test_artifact_loader_requires_intact_complete_seals(tmp_path: Path) -> None:
    output, config = _minimal_sealed_panel(tmp_path)
    loaded = analyzer.load_exact_runs(
        output,
        config,
        registered_implementation_fingerprint=FINGERPRINT,
    )
    assert len(loaded) == 24
    target = loaded[0].path / "metrics.jsonl"
    target.write_text('{"record_id":"changed"}\n', encoding="utf-8")
    with pytest.raises(analyzer.Qwen35HiddenLawAnalysisError, match="sealed"):
        analyzer.load_exact_runs(
            output,
            config,
            registered_implementation_fingerprint=FINGERPRINT,
        )


def test_model_manifest_accepts_null_resolutions_but_rejects_wrong_non_null_revision() -> None:
    condition = build_hidden_law_plan(load_hidden_law_config(CONFIG))[0]
    manifest = _model_manifest(condition)
    analyzer._validate_model_manifest(condition, manifest)
    manifest["resolved_revision"] = "wrong-revision"
    with pytest.raises(analyzer.Qwen35HiddenLawAnalysisError, match="resolved_revision"):
        analyzer._validate_model_manifest(condition, manifest)
    manifest = _model_manifest(condition)
    manifest["trainable_parameter_dtype_counts"] = {"torch.float32": manifest["parameter_count"]}
    with pytest.raises(analyzer.Qwen35HiddenLawAnalysisError, match="trainable_parameter_dtype_counts"):
        analyzer._validate_model_manifest(condition, manifest)


def test_cli_registration_bindings_match_frozen_sources() -> None:
    assert analyzer._sha256(analyzer.MAIN_CONFIG) == analyzer.REGISTERED_CONFIG_FILE_SHA256
    assert analyzer._sha256(analyzer.PROTOCOL) == analyzer.REGISTERED_PROTOCOL_SHA256
    config = load_hidden_law_config(analyzer.MAIN_CONFIG)
    assert (
        analyzer.canonical_digest(analyzer.scientific_config(config))
        == analyzer.REGISTERED_SCIENTIFIC_CONFIG_DIGEST
    )
    assert (
        analyzer.implementation_provenance(ROOT)["implementation_fingerprint"]
        == analyzer.REGISTERED_IMPLEMENTATION_FINGERPRINT
    )


def test_cli_rejects_a_current_source_fingerprint_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        analyzer,
        "implementation_provenance",
        lambda _root: {"implementation_fingerprint": "0" * 64},
    )
    with pytest.raises(analyzer.Qwen35HiddenLawAnalysisError, match="current implementation source"):
        analyzer.analyze_main(tmp_path)
