"""Factorial and causal-intervention evaluation for GoalZendo policies."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn

ScoreFunction = Callable[[Sequence[str]], Tensor | Any]


def _choice_index(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("Boolean values are not valid GoalZendo choices")
    if hasattr(value, "value") and isinstance(value.value, int):
        value = value.value
    if isinstance(value, int) and value in (0, 1):
        return value
    normalized = str(value).strip().upper()
    if normalized in {"A", "0"}:
        return 0
    if normalized in {"B", "1"}:
        return 1
    raise ValueError(f"invalid GoalZendo choice: {value!r}")


def _candidate_choices(example: Mapping[str, Any]) -> tuple[int, int, int]:
    nested = example.get("candidate_choices", example.get("choices", {}))
    if nested is None:
        nested = {}
    if not isinstance(nested, Mapping):
        raise ValueError("candidate_choices must be a mapping")

    def find(short: str, long: str) -> int:
        for mapping, keys in (
            (nested, (short, long, f"choice_{short}")),
            (example, (f"choice_{short}", short, long)),
        ):
            for key in keys:
                if key in mapping:
                    return _choice_index(mapping[key])
        raise ValueError(f"example is missing the {long} candidate choice")

    return find("y", "law"), find("p", "herald"), find("q", "sage")


def _prompt_views(example: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    raw_views = example.get("prompt_views")
    views: dict[str, str] = {}
    if raw_views is not None:
        if not isinstance(raw_views, Mapping):
            raise ValueError("prompt_views must be a mapping from view name to text")
        for name, prompt in raw_views.items():
            if not isinstance(prompt, str) or not prompt:
                raise ValueError("every prompt view must be non-empty text")
            views[str(name)] = prompt
    if "prompt" in example:
        prompt = example["prompt"]
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt must be non-empty text")
        default_name = str(example.get("prompt_view", "default"))
        if default_name in views and views[default_name] != prompt:
            raise ValueError(f"duplicate, conflicting prompt view {default_name!r}")
        views[default_name] = prompt
    if not views:
        raise ValueError("each evaluation example needs prompt or prompt_views")
    return tuple(sorted(views.items()))


@dataclass(frozen=True)
class PredictionRecord:
    """One constrained prediction for one rendered view of one symbolic item."""

    sample_id: str
    prompt_view: str
    choice_y: int
    choice_p: int
    choice_q: int
    predicted_action: int
    score_a: float
    score_b: float
    probability_b: float
    margin_b_minus_a: float
    intervention_pair_id: str | None = None
    intervention_role: str | None = None
    intervention_target: str | None = None

    @property
    def factorial_cell(self) -> tuple[int, int, int]:
        return self.choice_y, self.choice_p, self.choice_q

    @property
    def is_conflict(self) -> bool:
        return len(set(self.factorial_cell)) > 1

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["factorial_cell"] = list(self.factorial_cell)
        result["is_conflict"] = self.is_conflict
        return result


@dataclass(frozen=True)
class InterventionEffect:
    """A within-pair causal input intervention effect."""

    pair_id: str
    prompt_view: str
    target: str
    base_sample_id: str
    intervention_sample_id: str
    delta_margin_b_minus_a: float
    delta_probability_b: float
    action_flipped: bool
    target_choice_changed: bool
    target_aligned_delta_margin: float | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationResult:
    """All row-level evidence and compact summaries from an evaluation pass."""

    records: tuple[PredictionRecord, ...]
    factorial_cells: tuple[dict[str, Any], ...]
    behavioral_agreement: tuple[dict[str, Any], ...]
    intervention_effects: tuple[InterventionEffect, ...]
    intervention_summary: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "records": [record.as_dict() for record in self.records],
            "factorial_cells": list(self.factorial_cells),
            "behavioral_agreement": list(self.behavioral_agreement),
            "intervention_effects": [effect.as_dict() for effect in self.intervention_effects],
            "intervention_summary": list(self.intervention_summary),
        }


def _score_tensor(value: Tensor | Any) -> Tensor:
    scores = value.log_scores if hasattr(value, "log_scores") else value
    if not isinstance(scores, Tensor):
        raise TypeError("score function must return a Tensor or an object with .log_scores")
    if scores.ndim != 2 or scores.shape[1] != 2:
        raise ValueError("score function must return shape [batch, 2]")
    if not scores.is_floating_point():
        raise TypeError("score function must return floating-point scores")
    if not bool(torch.isfinite(scores).all().detach().cpu()):
        raise FloatingPointError("score function returned NaN or infinity")
    return scores


def _records_for_batch(
    examples: Sequence[Mapping[str, Any]],
    score_fn: ScoreFunction,
) -> list[PredictionRecord]:
    flattened: list[tuple[Mapping[str, Any], str, str]] = []
    for example in examples:
        for view_name, prompt in _prompt_views(example):
            flattened.append((example, view_name, prompt))
    if not flattened:
        return []
    prompts = [item[2] for item in flattened]
    scores = _score_tensor(score_fn(prompts))
    if scores.shape[0] != len(flattened):
        raise ValueError("score function returned the wrong batch length")
    probabilities = scores.softmax(dim=-1)
    if not bool(torch.isfinite(probabilities).all().detach().cpu()):
        raise FloatingPointError("evaluation probabilities contain NaN or infinity")
    row_sums = probabilities.sum(dim=-1)
    if not bool(
        torch.isclose(
            row_sums,
            torch.ones_like(row_sums),
            rtol=5e-3,
            atol=5e-3,
        ).all()
        .detach()
        .cpu()
    ):
        raise FloatingPointError("evaluation probabilities are not normalized")
    probabilities_b = probabilities[:, 1]
    predictions = scores.argmax(dim=-1)
    records: list[PredictionRecord] = []
    for row, (example, view_name, _prompt) in enumerate(flattened):
        choice_y, choice_p, choice_q = _candidate_choices(example)
        sample_id = str(example.get("sample_id", ""))
        if not sample_id:
            raise ValueError("each evaluation example requires sample_id")
        role = example.get("intervention_role")
        target = example.get("intervention_target")
        pair_id = example.get("intervention_pair_id")
        if any(value is not None for value in (role, target, pair_id)) and not all(
            value is not None for value in (role, target, pair_id)
        ):
            raise ValueError(
                "intervention_pair_id, intervention_role, and intervention_target "
                "must be supplied together"
            )
        if role is not None and str(role).lower() not in {"base", "intervention"}:
            raise ValueError("intervention_role must be 'base' or 'intervention'")
        score_a = float(scores[row, 0].detach().cpu())
        score_b = float(scores[row, 1].detach().cpu())
        margin = score_b - score_a
        if not all(math.isfinite(value) for value in (score_a, score_b, margin)):
            raise FloatingPointError("evaluation record contains a non-finite score or margin")
        records.append(
            PredictionRecord(
                sample_id=sample_id,
                prompt_view=view_name,
                choice_y=choice_y,
                choice_p=choice_p,
                choice_q=choice_q,
                predicted_action=int(predictions[row].detach().cpu()),
                score_a=score_a,
                score_b=score_b,
                probability_b=float(probabilities_b[row].detach().cpu()),
                margin_b_minus_a=margin,
                intervention_pair_id=None if pair_id is None else str(pair_id),
                intervention_role=None if role is None else str(role).lower(),
                intervention_target=None if target is None else str(target).lower(),
            )
        )
    return records


def summarize_factorial(records: Sequence[PredictionRecord]) -> tuple[dict[str, Any], ...]:
    """Summarize behavior in every observed ``(Y, P, Q)`` factorial cell."""

    grouped: dict[tuple[str, int, int, int], list[PredictionRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.prompt_view, *record.factorial_cell)].append(record)
    summaries: list[dict[str, Any]] = []
    for (view, choice_y, choice_p, choice_q), group in sorted(grouped.items()):
        count = len(group)
        summaries.append(
            {
                "prompt_view": view,
                "choice_y": choice_y,
                "choice_p": choice_p,
                "choice_q": choice_q,
                "cell": f"Y={'AB'[choice_y]}|P={'AB'[choice_p]}|Q={'AB'[choice_q]}",
                "n": count,
                "is_conflict": len({choice_y, choice_p, choice_q}) > 1,
                "action_b_rate": sum(item.predicted_action == 1 for item in group) / count,
                "agreement_y": sum(item.predicted_action == choice_y for item in group) / count,
                "agreement_p": sum(item.predicted_action == choice_p for item in group) / count,
                "agreement_q": sum(item.predicted_action == choice_q for item in group) / count,
                "mean_margin_b_minus_a": sum(item.margin_b_minus_a for item in group) / count,
            }
        )
    return tuple(summaries)


def summarize_behavioral_agreement(
    records: Sequence[PredictionRecord],
) -> tuple[dict[str, Any], ...]:
    """Report rule agreement by rendering and by agreement/conflict status."""

    grouped: dict[tuple[str, str], list[PredictionRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.prompt_view, "all")].append(record)
        grouped[(record.prompt_view, "conflict" if record.is_conflict else "agreement")].append(record)
    summaries: list[dict[str, Any]] = []
    for (view, panel), group in sorted(grouped.items()):
        count = len(group)
        summaries.append(
            {
                "prompt_view": view,
                "panel": panel,
                "n": count,
                "agreement_y": sum(item.predicted_action == item.choice_y for item in group) / count,
                "agreement_p": sum(item.predicted_action == item.choice_p for item in group) / count,
                "agreement_q": sum(item.predicted_action == item.choice_q for item in group) / count,
                "action_b_rate": sum(item.predicted_action == 1 for item in group) / count,
            }
        )
    return tuple(summaries)


def _target_choice(record: PredictionRecord, target: str) -> int | None:
    normalized = target.lower()
    if normalized in {"y", "law"}:
        return record.choice_y
    if normalized in {"p", "herald"}:
        return record.choice_p
    if normalized in {"q", "sage"}:
        return record.choice_q
    return None


def pair_interventions(records: Sequence[PredictionRecord]) -> tuple[InterventionEffect, ...]:
    """Pair base/intervened prompts and compute causal changes in action scores."""

    grouped: dict[tuple[str, str, str], list[PredictionRecord]] = defaultdict(list)
    for record in records:
        if record.intervention_pair_id is not None:
            assert record.intervention_target is not None
            grouped[
                (record.prompt_view, record.intervention_target, record.intervention_pair_id)
            ].append(record)
    effects: list[InterventionEffect] = []
    for (view, target, pair_id), group in sorted(grouped.items()):
        bases = [item for item in group if item.intervention_role == "base"]
        interventions = [item for item in group if item.intervention_role == "intervention"]
        if len(bases) != 1 or len(interventions) != 1:
            raise ValueError(
                f"intervention pair {pair_id!r}/{view!r}/{target!r} needs exactly "
                "one base and one intervention record"
            )
        base, intervention = bases[0], interventions[0]
        delta_margin = intervention.margin_b_minus_a - base.margin_b_minus_a
        delta_probability = intervention.probability_b - base.probability_b
        if not math.isfinite(delta_margin) or not math.isfinite(delta_probability):
            raise FloatingPointError("intervention effect contains a non-finite difference")
        base_target = _target_choice(base, target)
        intervention_target = _target_choice(intervention, target)
        changed = (
            base_target is not None
            and intervention_target is not None
            and base_target != intervention_target
        )
        aligned = None
        if changed:
            aligned = delta_margin if intervention_target == 1 else -delta_margin
        effects.append(
            InterventionEffect(
                pair_id=pair_id,
                prompt_view=view,
                target=target,
                base_sample_id=base.sample_id,
                intervention_sample_id=intervention.sample_id,
                delta_margin_b_minus_a=delta_margin,
                delta_probability_b=delta_probability,
                action_flipped=base.predicted_action != intervention.predicted_action,
                target_choice_changed=changed,
                target_aligned_delta_margin=aligned,
            )
        )
    return tuple(effects)


def summarize_interventions(
    effects: Sequence[InterventionEffect],
) -> tuple[dict[str, Any], ...]:
    grouped: dict[tuple[str, str], list[InterventionEffect]] = defaultdict(list)
    for effect in effects:
        grouped[(effect.prompt_view, effect.target)].append(effect)
    summaries: list[dict[str, Any]] = []
    for (view, target), group in sorted(grouped.items()):
        aligned = [
            effect.target_aligned_delta_margin
            for effect in group
            if effect.target_aligned_delta_margin is not None
        ]
        summaries.append(
            {
                "prompt_view": view,
                "target": target,
                "n_pairs": len(group),
                "mean_delta_margin_b_minus_a": sum(
                    effect.delta_margin_b_minus_a for effect in group
                )
                / len(group),
                "mean_absolute_delta_margin": sum(
                    abs(effect.delta_margin_b_minus_a) for effect in group
                )
                / len(group),
                "action_flip_rate": sum(effect.action_flipped for effect in group) / len(group),
                "target_choice_change_rate": sum(
                    effect.target_choice_changed for effect in group
                )
                / len(group),
                "mean_target_aligned_delta_margin": (
                    None if not aligned else sum(aligned) / len(aligned)
                ),
            }
        )
    return tuple(summaries)


def evaluate_batches(
    batches: Iterable[Sequence[Mapping[str, Any]]],
    score_fn: ScoreFunction,
) -> EvaluationResult:
    """Evaluate mapping-based batches without coupling to a data-loader class.

    Each mapping supplies ``sample_id``, candidate choices under ``Y/P/Q``, and
    either ``prompt`` or a ``prompt_views`` mapping.  Matched causal examples
    additionally supply ``intervention_pair_id``, ``intervention_role`` and
    ``intervention_target``.  This deliberately small contract lets symbolic
    generators and future Hugging Face datasets share the same evaluator.
    """

    module = score_fn if isinstance(score_fn, nn.Module) else None
    was_training = module.training if module is not None else None
    if module is not None:
        module.eval()
    records: list[PredictionRecord] = []
    try:
        with torch.no_grad():
            for batch in batches:
                records.extend(_records_for_batch(batch, score_fn))
    finally:
        if module is not None and was_training is not None:
            module.train(was_training)

    factorial = summarize_factorial(records)
    agreement = summarize_behavioral_agreement(records)
    effects = pair_interventions(records)
    return EvaluationResult(
        records=tuple(records),
        factorial_cells=factorial,
        behavioral_agreement=agreement,
        intervention_effects=effects,
        intervention_summary=summarize_interventions(effects),
    )
