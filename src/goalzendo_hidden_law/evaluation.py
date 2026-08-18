"""Held-out active-inquiry and behavioral-control evaluation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import torch

from goalzendo_interactive.rendering import RendererName
from goalzendo_interactive.rules import BinaryRule
from goalzendo_interactive.schema import scene_at

from .experiment import (
    BINARY_ACTION_LABELS,
    CANDIDATE_ACTION_LABELS,
    MAX_QUERY_TURNS,
    ActionPolicy,
    Decision,
    QueryObservation,
    _probabilities,
    candidate_prompt,
    classification_prompt,
    query_prompt,
    reference_inquiry,
)
from .game import GameFamily, GameInstance, ProductionBank, query_information

EvaluationView = Literal["active", "oracle_query", "no_query"]


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    metric_rows: tuple[dict[str, Any], ...]
    transcript_rows: tuple[dict[str, Any], ...]
    prediction_rows: tuple[dict[str, Any], ...]
    summary: Mapping[str, Any]


def conflict_cell_agreements(
    predictions: Sequence[bool],
    truths: Mapping[str, Sequence[bool]],
) -> tuple[dict[str, float], tuple[int, ...]]:
    """Agreement on the 14 non-unanimous four-candidate truth cells only."""

    if len(predictions) != 16 or set(truths) != {"Y", "P", "Q", "R"}:
        raise ValueError("conflict agreement requires predictions and four candidate truth vectors")
    if any(len(values) != 16 for values in truths.values()):
        raise ValueError("each candidate truth vector must contain sixteen cells")
    conflict_indices = tuple(
        index for index in range(16) if len({values[index] for values in truths.values()}) > 1
    )
    if len(conflict_indices) != 14:
        raise ValueError("terminal census must contain exactly fourteen non-unanimous cells")
    agreements = {
        role: sum(predictions[index] is values[index] for index in conflict_indices) / len(conflict_indices)
        for role, values in truths.items()
    }
    return agreements, conflict_indices


def _decision(
    *,
    kind: str,
    prompt: str,
    labels: tuple[str, ...],
    scores: torch.Tensor,
    row: int = 0,
) -> Decision:
    probabilities = _probabilities(scores)[row]
    selected = max(range(len(probabilities)), key=probabilities.__getitem__)
    return Decision(kind, prompt, labels, selected, probabilities)


@torch.no_grad()
def _active_inquiry(
    policy: ActionPolicy,
    instance: GameInstance,
    renderer: RendererName,
) -> tuple[tuple[QueryObservation, ...], tuple[Decision, ...]]:
    observations: list[QueryObservation] = []
    decisions: list[Decision] = []
    for _turn in range(MAX_QUERY_TURNS):
        prompt, labels, meanings = query_prompt(instance, renderer, observations)
        scores = policy.score((prompt,), labels)
        decision = _decision(kind="query", prompt=prompt, labels=labels, scores=scores)
        decisions.append(decision)
        selected = meanings[decision.selected_index]
        if selected == "READY":
            break
        observations.append(QueryObservation(selected, instance.oracle_label(selected)))
    return tuple(observations), tuple(decisions)


def _condition_metadata(instance: GameInstance) -> dict[str, Any]:
    opening = tuple((item.scene_index, item.accepted) for item in instance.material.opening)
    matches = [
        condition
        for condition in instance.family.evaluation_conditions
        if tuple((item.scene_index, item.accepted) for item in condition.opening) == opening
    ]
    if len(matches) != 1:
        raise ValueError("evaluation instance does not match exactly one evidence condition")
    condition = matches[0]
    family = instance.family
    by_role = family.candidate_by_role
    y_role = family.evaluation_y_role
    r_role = family.evaluation_r_role
    return {
        "condition_id": condition.condition_id,
        "p_evidence": condition.p_evidence,
        "q_evidence": condition.q_evidence,
        "y_rule_family": "monotone" if y_role == "M" else "exactly_one",
        "monotone_operator": cast(BinaryRule, by_role["M"].rule).op,
        "analysis_candidate_ids": {
            "Y": by_role[y_role].candidate_id,
            "P": by_role["P"].candidate_id,
            "Q": by_role["Q"].candidate_id,
            "R": by_role[r_role].candidate_id,
        },
    }


def _query_metrics(
    instance: GameInstance,
    observations: Sequence[QueryObservation],
) -> tuple[list[dict[str, Any]], int, float]:
    live = instance.initial_live_ids
    rows: list[dict[str, Any]] = []
    for turn, observation in enumerate(observations, start=1):
        information = query_information(instance, observation.option_id, live)
        after = information.accepted_ids if observation.accepted else information.rejected_ids
        rows.append(
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
    start = len(instance.initial_live_ids)
    information_fraction = 0.0 if start <= 1 else math.log2(start / len(live)) / math.log2(start)
    return rows, len(live), information_fraction


def _candidate_truths(family: GameFamily, scene_indices: Sequence[int]) -> dict[str, tuple[bool, ...]]:
    return {
        role: tuple(candidate.rule.evaluate(scene_at(index)) for index in scene_indices)
        for role, candidate in {
            "Y": family.candidate_by_role[family.evaluation_y_role],
            "P": family.candidate_by_role["P"],
            "Q": family.candidate_by_role["Q"],
            "R": family.candidate_by_role[family.evaluation_r_role],
        }.items()
    }


def _prediction_rows(
    *,
    prefix: str,
    decisions: Sequence[Decision],
) -> list[dict[str, Any]]:
    return [
        {
            "record_id": f"{prefix}:decision:{index:03d}",
            "kind": "evaluation_prediction",
            "decision_index": index,
            "decision_kind": decision.kind,
            "action_labels": list(decision.action_labels),
            "selected_index": decision.selected_index,
            "selected_label": decision.selected_label,
            "probabilities": list(decision.probabilities),
        }
        for index, decision in enumerate(decisions)
    ]


@torch.no_grad()
def evaluate_episode(
    policy: ActionPolicy,
    instance: GameInstance,
    renderer: RendererName,
    *,
    step: int,
    view: EvaluationView,
    include_interventions: bool,
) -> EvaluationResult:
    if view == "active":
        observations, query_decisions = _active_inquiry(policy, instance, renderer)
    elif view == "oracle_query":
        observations, query_decisions = reference_inquiry(instance), ()
    elif view == "no_query":
        observations, query_decisions = (), ()
    else:
        raise ValueError(f"unknown evaluation view: {view!r}")

    candidate_user_prompt = candidate_prompt(instance, renderer, observations)
    candidate_scores = policy.score((candidate_user_prompt,), CANDIDATE_ACTION_LABELS)
    candidate_decision = _decision(
        kind="candidate",
        prompt=candidate_user_prompt,
        labels=CANDIDATE_ACTION_LABELS,
        scores=candidate_scores,
    )

    terminal_prompts = tuple(
        classification_prompt(instance, renderer, observations, scene_at(index))
        for index in instance.material.terminal
    )
    terminal_scores = policy.score(terminal_prompts, BINARY_ACTION_LABELS)
    terminal_decisions = tuple(
        _decision(
            kind="classification",
            prompt=prompt,
            labels=BINARY_ACTION_LABELS,
            scores=terminal_scores,
            row=index,
        )
        for index, prompt in enumerate(terminal_prompts)
    )
    terminal_predictions = tuple(decision.selected_index == 0 for decision in terminal_decisions)
    truths = _candidate_truths(instance.family, instance.material.terminal)
    agreement, _conflict_indices = conflict_cell_agreements(terminal_predictions, truths)
    all_cell_accuracy = (
        sum(left is right for left, right in zip(terminal_predictions, truths["Y"], strict=True)) / 16
    )

    intervention_rows: list[dict[str, Any]] = []
    intervention_decisions: list[Decision] = []
    flip_rates: dict[str, float] = {}
    if include_interventions:
        for target in ("Y", "P", "Q", "R", "distractor"):
            records = [record for record in instance.family.matched_interventions if record.target == target]
            if len(records) != 2:
                raise ValueError(f"evaluation family lacks two {target} interventions")
            prompts = tuple(
                classification_prompt(instance, renderer, observations, scene_at(index))
                for record in records
                for index in (record.before_scene_index, record.after_scene_index)
            )
            scores = policy.score(prompts, BINARY_ACTION_LABELS)
            decisions = tuple(
                _decision(
                    kind=f"intervention_{target}",
                    prompt=prompt,
                    labels=BINARY_ACTION_LABELS,
                    scores=scores,
                    row=index,
                )
                for index, prompt in enumerate(prompts)
            )
            intervention_decisions.extend(decisions)
            flips: list[bool] = []
            for pair_offset, record in enumerate(records):
                before, after = decisions[2 * pair_offset : 2 * pair_offset + 2]
                flipped = before.selected_index != after.selected_index
                flips.append(flipped)
                intervention_rows.append(
                    {
                        "target": target,
                        "pair_index": record.pair_index,
                        "before_scene_index": record.before_scene_index,
                        "after_scene_index": record.after_scene_index,
                        "changed_field": record.changed_field,
                        "before_prediction": before.selected_index == 0,
                        "after_prediction": after.selected_index == 0,
                        "hard_flip": flipped,
                        "before_fit_probability": before.probabilities[0],
                        "after_fit_probability": after.probabilities[0],
                    }
                )
            flip_rates[target] = sum(flips) / len(flips)

    query_rows, final_live_count, information_fraction = _query_metrics(instance, observations)
    prefix = f"eval:{step:04d}:{view}:{instance.instance_id}"
    condition = _condition_metadata(instance)
    selected_candidate = candidate_decision.selected_label
    y_id = instance.family.candidate_by_role[instance.family.evaluation_y_role].candidate_id
    metric = {
        "record_id": f"{prefix}:metrics",
        "kind": "evaluation_episode",
        "step": step,
        "view": view,
        "family_id": instance.family.family_id,
        "instance_id": instance.instance_id,
        "renderer": renderer,
        **condition,
        "initial_live_count": len(instance.initial_live_ids),
        "final_live_count": final_live_count,
        "query_count": len(observations),
        "declared_ready": any(decision.selected_label == "I" for decision in query_decisions),
        "information_fraction": information_fraction,
        "query_turns": query_rows,
        "selected_candidate_id": selected_candidate,
        "official_candidate_id": y_id,
        "exact_rule_recovery": selected_candidate == y_id,
        "terminal_classification_accuracy_all_16": all_cell_accuracy,
        "agreement": agreement,
        "rho_Y": agreement["Y"],
        "rho_P": agreement["P"],
        "rho_Q": agreement["Q"],
        "rho_R": agreement["R"],
        "law_control_margin": agreement["Y"] - max(agreement[role] for role in ("P", "Q", "R")),
        "intervention_flip_rates": flip_rates,
        "flip_Y": flip_rates.get("Y"),
        "flip_P": flip_rates.get("P"),
        "flip_Q": flip_rates.get("Q"),
        "flip_R": flip_rates.get("R"),
        "flip_distractor": flip_rates.get("distractor"),
        "causal_law_control_margin": (
            None if not flip_rates else flip_rates["Y"] - max(flip_rates[role] for role in ("P", "Q", "R"))
        ),
    }
    all_decisions = (
        *query_decisions,
        candidate_decision,
        *terminal_decisions,
        *intervention_decisions,
    )
    predictions = _prediction_rows(prefix=prefix, decisions=all_decisions)
    for row in predictions:
        row.update({"step": step, "view": view, "instance_id": instance.instance_id})
    transcript = {
        "record_id": f"{prefix}:transcript",
        "kind": "evaluation_transcript",
        "step": step,
        "view": view,
        "family_id": instance.family.family_id,
        "instance_id": instance.instance_id,
        "renderer": renderer,
        **condition,
        "opening": [
            {
                "scene_index": item.scene_index,
                "accepted": item.accepted,
            }
            for item in instance.material.opening
        ],
        "query_observations": [
            {"option_id": item.option_id, "accepted": item.accepted} for item in observations
        ],
        "decisions": [decision.as_obj() for decision in all_decisions],
        "terminal_scene_indices": list(instance.material.terminal),
        "terminal_predictions": list(terminal_predictions),
        "terminal_truths": {key: list(value) for key, value in truths.items()},
        "candidate_rules": {
            role: candidate.rule.as_obj()
            for role, candidate in {
                "Y": instance.family.candidate_by_role[instance.family.evaluation_y_role],
                "P": instance.family.candidate_by_role["P"],
                "Q": instance.family.candidate_by_role["Q"],
                "R": instance.family.candidate_by_role[instance.family.evaluation_r_role],
            }.items()
        },
        "interventions": intervention_rows,
    }
    return EvaluationResult((metric,), (transcript,), tuple(predictions), metric)


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def evaluate_checkpoint(
    policy: ActionPolicy,
    bank: ProductionBank,
    *,
    step: int,
    final: bool,
    eval_renderers: Sequence[RendererName],
) -> EvaluationResult:
    if len(eval_renderers) != 2:
        raise ValueError("evaluation requires the two registered renderers")
    families = bank.evaluation_families if final else bank.interim_evaluation_families
    views: tuple[EvaluationView, ...] = ("active", "oracle_query", "no_query") if final else ("active",)
    metrics: list[dict[str, Any]] = []
    transcripts: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for family_index, family in enumerate(families):
        renderer = eval_renderers[family_index % len(eval_renderers)]
        for instance in family.evaluation_games:
            for view in views:
                result = evaluate_episode(
                    policy,
                    instance,
                    renderer,
                    step=step,
                    view=view,
                    include_interventions=final,
                )
                metrics.extend(result.metric_rows)
                transcripts.extend(result.transcript_rows)
                predictions.extend(result.prediction_rows)
    expected = len(families) * 4 * len(views)
    if len(metrics) != expected:
        raise RuntimeError("evaluation did not emit exactly one metric row per episode")
    by_view = {view: [row for row in metrics if row["view"] == view] for view in views}
    summary = {
        "step": step,
        "final": final,
        "quartet_count": len(families),
        "episode_count": len(metrics),
        "views": {
            view: {
                "episode_count": len(rows),
                "information_fraction": _mean(rows, "information_fraction"),
                "exact_rule_recovery": sum(bool(row["exact_rule_recovery"]) for row in rows) / len(rows),
                "terminal_classification_accuracy_all_16": _mean(
                    rows, "terminal_classification_accuracy_all_16"
                ),
                "law_control_margin": _mean(rows, "law_control_margin"),
                "causal_law_control_margin": (_mean(rows, "causal_law_control_margin") if final else None),
            }
            for view, rows in by_view.items()
        },
    }
    metrics.append(
        {
            "record_id": f"eval:{step:04d}:checkpoint-summary",
            "kind": "evaluation_checkpoint_summary",
            **summary,
        }
    )
    return EvaluationResult(tuple(metrics), tuple(transcripts), tuple(predictions), summary)
