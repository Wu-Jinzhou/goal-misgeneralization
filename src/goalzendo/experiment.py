"""Concrete, resume-safe Hugging Face backend for known-law GoalZendo.

This module is the bridge between the symbolic experiment and the generic run
transaction.  It deliberately records the exact generated scenes and rendered
prompt banks before optimization, evaluates candidate-rule conflicts at dense
checkpoints, and saves only completed optimizer boundaries.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .artifacts import read_json, stable_hash, write_json
from .config import get_path
from .evaluation import EvaluationResult, evaluate_batches
from .generation import generate_factorial_evaluation, generate_known_law_dataset
from .interventions import audit_intervention, is_one_atom_law_boundary, paired_interventions
from .modeling import (
    TwoActionScorer,
    encode_action_continuations,
    format_chat_prompt,
    load_model_and_tokenizer,
    model_provenance,
)
from .rendering import render_known_law_decision
from .runner import BackendResult, RunContext, derive_seed
from .schema import KnownLawDataset, KnownLawDecision, RuleSpec, stable_digest
from .training import (
    ScheduledHooks,
    TrainState,
    build_optimizer,
    train_steps,
)

EXPERIMENT_BACKEND_VERSION = 9
PROMPT_VIEWS = frozenset(
    {
        "full",
        "audit_law_full",
        "audit_law_matched",
        "no_herald",
        "no_sage",
        "law_only",
        "sage_only",
        "herald_only",
        "no_signal",
        "surface_only",
    }
)

EvaluationObserver = Callable[[int], None]


class ExperimentError(RuntimeError):
    """Raised when an experiment cannot preserve its scientific contract."""


def configure_numerical_execution(config: Mapping[str, Any]) -> dict[str, Any]:
    """Apply and record the requested PyTorch numerical-execution contract.

    Deterministic CUDA execution also requires ``CUBLAS_WORKSPACE_CONFIG`` to
    be present before the first cuBLAS operation. The Runpod entrypoint exports
    it before Python starts; direct callers may set it here only while no CUDA
    work has occurred. A conflicting inherited value fails closed.
    """

    deterministic = bool(get_path(config, "train.deterministic_algorithms", False))
    allow_tf32 = bool(get_path(config, "train.allow_tf32", False))
    workspace = str(get_path(config, "train.cublas_workspace_config", ":4096:8"))
    if deterministic and allow_tf32:
        raise ExperimentError("deterministic execution cannot enable TF32")
    if workspace not in {":4096:8", ":16:8"}:
        raise ExperimentError("unsupported CUBLAS_WORKSPACE_CONFIG")
    inherited_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if deterministic:
        if inherited_workspace is not None and inherited_workspace != workspace:
            raise ExperimentError(
                "inherited CUBLAS_WORKSPACE_CONFIG conflicts with the run specification"
            )
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace

    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    if not allow_tf32:
        torch.set_float32_matmul_precision("highest")

    return {
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "deterministic_warn_only": bool(torch.is_deterministic_algorithms_warn_only_enabled()),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


@dataclass(frozen=True)
class ExperimentBanks:
    train: KnownLawDataset
    validation: KnownLawDataset
    diagnostic_factorial: KnownLawDataset
    final_factorial: KnownLawDataset
    diagnostic_causal_base: tuple[KnownLawDecision, ...]
    final_causal_base: tuple[KnownLawDecision, ...]
    diagnostic_interventions: Mapping[str, tuple[KnownLawDecision, ...]]
    final_interventions: Mapping[str, tuple[KnownLawDecision, ...]]
    effective_train_decisions: tuple[KnownLawDecision, ...]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class RenderedExperiment:
    train_prompts: tuple[str, ...]
    train_actions: tuple[int, ...]
    train_candidate_choices: tuple[tuple[int, int, int], ...]
    train_candidate_cells: tuple[str, ...]
    validation_examples: tuple[Mapping[str, Any], ...]
    diagnostic_factorial_examples: tuple[Mapping[str, Any], ...]
    final_factorial_examples: tuple[Mapping[str, Any], ...]
    diagnostic_causal_examples: tuple[Mapping[str, Any], ...]
    final_causal_examples: tuple[Mapping[str, Any], ...]
    metadata: Mapping[str, Any]


def build_rule_specs(config: Mapping[str, Any]) -> tuple[RuleSpec, RuleSpec]:
    """Build the official Law and disjoint semantic Sage rule from config."""

    law_indices = tuple(int(value) for value in get_path(config, "data.law_features", ()))
    sage_indices = tuple(int(value) for value in get_path(config, "data.sage_features", ()))
    law_expected = tuple(
        bool(value) for value in get_path(config, "data.law_expected_values", (True,) * len(law_indices))
    )
    sage_expected = tuple(
        bool(value) for value in get_path(config, "data.sage_expected_values", (True,) * len(sage_indices))
    )
    law = RuleSpec(
        family=str(get_path(config, "data.rule_family", "parity")),
        feature_indices=law_indices,
        expected_values=law_expected,
        output_negated=bool(get_path(config, "data.law_output_negated", False)),
        name="official_law",
    )
    sage = RuleSpec(
        family=str(get_path(config, "data.sage_rule_family", "parity")),
        feature_indices=sage_indices,
        expected_values=sage_expected,
        output_negated=bool(get_path(config, "data.sage_output_negated", False)),
        name="sage_rule",
    )
    return law, sage


def _realizable_rate(rate: float, count: int) -> float:
    return round(float(rate) * int(count)) / int(count)


def _generation_geometry(
    config: Mapping[str, Any],
    count: int,
) -> tuple[str, int | None]:
    geometry = str(get_path(config, "data.error_geometry", "independent"))
    if geometry != "specified":
        return geometry, None
    rate = float(get_path(config, "data.joint_error_rate"))
    return "custom", round(rate * count)


def _dataset(
    config: Mapping[str, Any],
    *,
    count: int,
    seed: int,
    law: RuleSpec,
    sage: RuleSpec,
    split: str,
    allow_rate_rounding: bool,
    forbidden_scene_digests: Iterable[str] = (),
) -> KnownLawDataset:
    q_p = float(get_path(config, "data.q_p"))
    q_q = float(get_path(config, "data.q_q"))
    if allow_rate_rounding:
        q_p = _realizable_rate(q_p, count)
        q_q = _realizable_rate(q_q, count)
    geometry, joint = _generation_geometry(config, count)
    if joint is not None:
        p_errors = round((1.0 - q_p) * count)
        q_errors = round((1.0 - q_q) * count)
        lower = max(0, p_errors + q_errors - count)
        upper = min(p_errors, q_errors)
        joint = min(max(joint, lower), upper)
    return generate_known_law_dataset(
        n=count,
        seed=seed,
        rule=law,
        sage_rule=sage,
        q_p=q_p,
        q_q=q_q,
        error_geometry=geometry,  # type: ignore[arg-type]
        joint_error_count=joint,
        feature_names=get_path(config, "data.feature_names", None),
        feature_count=int(get_path(config, "data.feature_count")),
        split=split,
        unique_semantic_scenes=True,
        forbidden_scene_digests=forbidden_scene_digests,
    )


def _concentrate_conflicts(
    decisions: Sequence[KnownLawDecision],
    unique_budget: int,
) -> tuple[KnownLawDecision, ...]:
    """Repeat a small prototype set while preserving every candidate tuple."""

    if unique_budget < 1:
        raise ValueError("concentrated conflict diversity requires a positive unique budget")
    conflict_groups: dict[tuple[int, int, int], list[KnownLawDecision]] = {}
    for decision in decisions:
        if decision.choice_p == decision.choice_y and decision.choice_q == decision.choice_y:
            continue
        key = _candidate_key(decision)
        conflict_groups.setdefault(key, []).append(decision)
    if not conflict_groups:
        return tuple(decisions)
    ordered_groups = sorted(conflict_groups)
    if unique_budget < len(ordered_groups):
        raise ExperimentError(
            "concentrated_unique_conflicts must cover every observed conflict tuple; "
            f"need at least {len(ordered_groups)}"
        )
    allocation = {key: 1 for key in ordered_groups}
    remaining = min(unique_budget, sum(len(values) for values in conflict_groups.values())) - len(
        ordered_groups
    )
    while remaining:
        progressed = False
        for key in ordered_groups:
            if allocation[key] < len(conflict_groups[key]) and remaining:
                allocation[key] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    prototypes = {
        key: tuple(sorted(group, key=lambda item: item.digest)[: allocation[key]])
        for key, group in conflict_groups.items()
    }
    counters: Counter[tuple[int, int, int]] = Counter()
    result: list[KnownLawDecision] = []
    for decision in decisions:
        key = _candidate_key(decision)
        if key not in prototypes:
            result.append(decision)
            continue
        candidates = prototypes[key]
        result.append(candidates[counters[key] % len(candidates)])
        counters[key] += 1
    return tuple(result)


def _concentrate_conflicts_per_tuple(
    decisions: Sequence[KnownLawDecision],
    unique_per_tuple: int,
) -> tuple[KnownLawDecision, ...]:
    """Repeat exactly ``unique_per_tuple`` prototypes in each conflict tuple."""

    if (
        not isinstance(unique_per_tuple, int)
        or isinstance(unique_per_tuple, bool)
        or unique_per_tuple < 1
    ):
        raise ValueError(
            "per-tuple concentrated conflict diversity requires a positive integer"
        )
    conflict_groups: dict[tuple[int, int, int], list[KnownLawDecision]] = {}
    for decision in decisions:
        if decision.choice_p == decision.choice_y and decision.choice_q == decision.choice_y:
            continue
        conflict_groups.setdefault(_candidate_key(decision), []).append(decision)

    prototypes: dict[tuple[int, int, int], tuple[KnownLawDecision, ...]] = {}
    for key, group in sorted(conflict_groups.items()):
        unique_group = {decision.digest: decision for decision in group}
        if len(unique_group) < unique_per_tuple:
            label = "".join("AB"[choice] for choice in key)
            raise ExperimentError(
                f"conflict tuple {label} has only {len(unique_group)} unique examples; "
                f"requested {unique_per_tuple}"
            )
        prototypes[key] = tuple(
            unique_group[digest] for digest in sorted(unique_group)[:unique_per_tuple]
        )

    counters: Counter[tuple[int, int, int]] = Counter()
    result: list[KnownLawDecision] = []
    for decision in decisions:
        key = _candidate_key(decision)
        if key not in prototypes:
            result.append(decision)
            continue
        candidates = prototypes[key]
        result.append(candidates[counters[key] % unique_per_tuple])
        counters[key] += 1
    return tuple(result)


def _candidate_key(decision: KnownLawDecision) -> tuple[int, int, int]:
    choice_y, choice_p, choice_q = decision.candidate_tuple
    return int(choice_y), int(choice_p), int(choice_q)


def _semantic_scene_digests(
    decisions: Sequence[KnownLawDecision],
) -> set[str]:
    return {
        koan.scene.semantic_digest
        for decision in decisions
        for koan in decision.koans
    }


def _semantic_scene_instances(
    decisions: Sequence[KnownLawDecision],
) -> list[str]:
    return [
        koan.scene.semantic_digest
        for decision in decisions
        for koan in decision.koans
    ]


def _mirror_pair_count(decisions: Sequence[KnownLawDecision]) -> int:
    pairs: dict[str, set[str]] = {}
    for decision in decisions:
        if decision.mirror_pair_id is None:
            continue
        assert decision.mirror_role is not None
        pairs.setdefault(decision.mirror_pair_id, set()).add(decision.mirror_role)
    if any(roles != {"base", "mirror"} for roles in pairs.values()):
        raise ExperimentError("registered mirror metadata is incomplete")
    return len(pairs)


def _assert_cross_bank_semantic_disjoint(
    banks: Mapping[str, Sequence[KnownLawDecision]],
) -> dict[str, Any]:
    digests = {name: _semantic_scene_digests(values) for name, values in banks.items()}
    overlaps: dict[str, int] = {}
    for left_index, left in enumerate(sorted(digests)):
        for right in sorted(digests)[left_index + 1 :]:
            count = len(digests[left] & digests[right])
            overlaps[f"{left}__{right}"] = count
            if count:
                raise ExperimentError(
                    f"semantic scene leakage between {left} and {right}: {count} feature vectors"
                )
    return {
        "policy": "semantic feature vectors sampled without replacement across base banks",
        "unique_semantic_scenes": {name: len(values) for name, values in digests.items()},
        "cross_bank_overlap": overlaps,
    }


def _intervention_bank(
    decisions: Sequence[KnownLawDecision],
) -> dict[str, tuple[KnownLawDecision, ...]]:
    grouped: dict[str, list[KnownLawDecision]] = {
        "herald": [],
        "sage": [],
        "law": [],
        "distractor": [],
    }
    name_map = {
        "flip_P": "herald",
        "flip_Q": "sage",
        "flip_Y": "law",
        "flip_D": "distractor",
    }
    for decision in decisions:
        for raw_name, changed in paired_interventions(decision).items():
            grouped[name_map[raw_name]].append(changed)
    return {name: tuple(values) for name, values in grouped.items()}


def _intervention_edit_audit(
    bases: Sequence[KnownLawDecision],
    changed_by_target: Mapping[str, Sequence[KnownLawDecision]],
) -> dict[str, Any]:
    """Count exactly which feature positions each causal target edits."""

    by_target: dict[str, Any] = {}
    for target, changed_values in sorted(changed_by_target.items()):
        if len(changed_values) != len(bases):
            raise ExperimentError("causal base and intervention banks have different lengths")
        index_counts: Counter[int] = Counter()
        hamming_counts: Counter[int] = Counter()
        changed_channel_counts: Counter[str] = Counter()
        herald_label_changes = 0
        for base, changed in zip(bases, changed_values, strict=True):
            audit = audit_intervention(base, changed)
            for indices in audit.changed_feature_indices:
                hamming_counts[len(indices)] += 1
                index_counts.update(indices)
            changed_channel_counts["+".join(sorted(audit.changed_channels)) or "none"] += 1
            herald_label_changes += int(audit.herald_label_changed)
        by_target[target] = {
            "n_pairs": len(bases),
            "n_koans": 2 * len(bases),
            "feature_index_counts": {
                str(index): count for index, count in sorted(index_counts.items())
            },
            "hamming_distance_counts": {
                str(distance): count for distance, count in sorted(hamming_counts.items())
            },
            "changed_channel_counts": dict(sorted(changed_channel_counts.items())),
            "herald_label_change_count": herald_label_changes,
        }
    return {
        "selection_policy": "deterministic sample/koan-keyed minimum-Hamming v1",
        "by_target": by_target,
    }


def _causal_panel(
    decisions: Sequence[KnownLawDecision],
    per_cell: int,
    seed: int,
    *,
    forbidden_scene_digests: Iterable[str] = (),
) -> tuple[
    tuple[KnownLawDecision, ...],
    dict[str, tuple[KnownLawDecision, ...]],
    set[str],
]:
    """Select mirrored, boundary-valid causal bases with novel edited scenes."""

    if per_cell < 1:
        raise ExperimentError("causal evaluation requires at least one item per cell")
    grouped: dict[tuple[int, int, int], list[KnownLawDecision]] = {}
    for decision in decisions:
        grouped.setdefault(_candidate_key(decision), []).append(decision)
    if set(grouped) != set(itertools.product((0, 1), repeat=3)):
        raise ExperimentError("causal base must cover all eight factorial cells")
    forbidden = {str(value) for value in forbidden_scene_digests}
    selected_novel: set[str] = set()
    selected_base_semantic: set[str] = set()
    selected: list[KnownLawDecision] = []

    def intervention_panel(
        bases: Sequence[KnownLawDecision],
    ) -> tuple[dict[str, tuple[KnownLawDecision, ...]], set[str]] | None:
        if any(
            decision.law.family == "majority" and not is_one_atom_law_boundary(decision)
            for decision in bases
        ):
            return None
        base_semantic = _semantic_scene_digests(bases)
        if base_semantic & selected_novel:
            return None
        changed: dict[str, list[KnownLawDecision]] = {
            "herald": [],
            "sage": [],
            "law": [],
            "distractor": [],
        }
        name_map = {
            "flip_P": "herald",
            "flip_Q": "sage",
            "flip_Y": "law",
            "flip_D": "distractor",
        }
        try:
            for decision in bases:
                for raw_name, intervened in paired_interventions(decision).items():
                    changed[name_map[raw_name]].append(intervened)
        except ValueError:
            return None
        novel_instances = [
            koan.scene.semantic_digest
            for target in ("sage", "law", "distractor")
            for intervened in changed[target]
            for koan in intervened.koans
        ]
        novel = set(novel_instances)
        # Mirror mates deliberately duplicate each other's edited scenes. No
        # other within-candidate collision is accepted.
        expected_unique = 3 * 2
        if len(novel) != expected_unique:
            return None
        if novel & forbidden or novel & selected_novel or novel & selected_base_semantic:
            return None
        return {name: tuple(values) for name, values in changed.items()}, novel

    mirror_records = [decision for decision in decisions if decision.mirror_pair_id is not None]
    if mirror_records:
        if len(mirror_records) != len(decisions):
            raise ExperimentError("causal bank cannot mix mirrored and unmirrored decisions")
        pairs: dict[str, dict[str, KnownLawDecision]] = {}
        for decision in decisions:
            assert decision.mirror_pair_id is not None and decision.mirror_role is not None
            pairs.setdefault(decision.mirror_pair_id, {})[decision.mirror_role] = decision
        canonical: dict[tuple[int, int, int], list[tuple[KnownLawDecision, KnownLawDecision]]] = {}
        for pair_id, roles in pairs.items():
            if set(roles) != {"base", "mirror"}:
                raise ExperimentError(f"incomplete registered mirror pair: {pair_id}")
            base, mirror = roles["base"], roles["mirror"]
            canonical.setdefault(_candidate_key(base), []).append((base, mirror))
        if any(cell[0] != 0 for cell in canonical):
            raise ExperimentError("registered mirror bases must use canonical Y=A cells")
        for cell, candidates in sorted(canonical.items()):
            ranked_pairs = sorted(
                candidates,
                key=lambda pair: stable_digest(
                    {"seed": seed, "cell": cell, "mirror_pair": pair[0].mirror_pair_id},
                    length=32,
                ),
            )
            accepted = 0
            for base, mirror in ranked_pairs:
                panel = intervention_panel((base, mirror))
                if panel is None:
                    continue
                _interventions, novel = panel
                selected.extend((base, mirror))
                selected_base_semantic.update(_semantic_scene_digests((base, mirror)))
                selected_novel.update(novel)
                accepted += 1
                if accepted == per_cell:
                    break
            if accepted != per_cell:
                raise ExperimentError(
                    f"only {accepted} boundary-valid, semantically novel mirror pairs were "
                    f"available for causal cell {cell}; need {per_cell}"
                )
    else:
        for cell, values in sorted(grouped.items()):
            ranked_decisions = sorted(
                values,
                key=lambda item: stable_digest(
                    {"seed": seed, "cell": cell, "decision": item.digest},
                    length=32,
                ),
            )
            accepted = 0
            for decision in ranked_decisions:
                panel = intervention_panel((decision,))
                if panel is None:
                    continue
                _interventions, novel = panel
                selected.append(decision)
                selected_base_semantic.update(_semantic_scene_digests((decision,)))
                selected_novel.update(novel)
                accepted += 1
                if accepted == per_cell:
                    break
            if accepted != per_cell:
                raise ExperimentError(
                    f"only {accepted} boundary-valid, semantically novel decisions were "
                    f"available for causal cell {cell}; need {per_cell}"
                )

    selected.sort(key=lambda item: (*_candidate_key(item), item.sample_id))
    # Rebuild in selected order to make target banks align exactly with bases.
    interventions = _intervention_bank(selected)
    observed_novel = {
        koan.scene.semantic_digest
        for target in ("sage", "law", "distractor")
        for decision in interventions[target]
        for koan in decision.koans
    }
    if observed_novel != selected_novel:
        raise ExperimentError("causal intervention reconstruction changed the selected semantic support")
    return tuple(selected), interventions, selected_novel


def _causal_subset(
    decisions: Sequence[KnownLawDecision],
    per_cell: int,
    seed: int,
) -> tuple[KnownLawDecision, ...]:
    """Compatibility wrapper returning just the validated causal bases."""

    selected, _interventions, _novel = _causal_panel(decisions, per_cell, seed)
    return selected


def _ordered_digest(values: Iterable[Any]) -> str:
    hasher = hashlib.sha256()
    for value in values:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        hasher.update(payload.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _bank_metadata(dataset: KnownLawDataset) -> dict[str, Any]:
    counts = Counter(
        "".join(choice.label for choice in decision.candidate_tuple) for decision in dataset.decisions
    )
    semantic_instances = _semantic_scene_instances(dataset.decisions)
    return {
        "split": dataset.split,
        "count": len(dataset),
        "seed": dataset.seed,
        "manifest_digest": dataset.manifest_digest,
        "ordered_decision_digest": _ordered_digest(decision.digest for decision in dataset.decisions),
        "candidate_counts": dict(sorted(counts.items())),
        "realized_q_p": dataset.realized_q_p,
        "realized_q_q": dataset.realized_q_q,
        "realized_joint_error_rate": dataset.realized_joint_error_rate,
        "error_geometry": dataset.error_geometry,
        "semantic_scene_instances": len(semantic_instances),
        "unique_semantic_scenes": len(set(semantic_instances)),
        "registered_mirror_pairs": _mirror_pair_count(dataset.decisions),
    }


def _unique_conflicts_by_candidate_tuple(
    decisions: Sequence[KnownLawDecision],
) -> dict[str, int]:
    unique: dict[str, set[str]] = {}
    for decision in decisions:
        if decision.choice_p == decision.choice_y and decision.choice_q == decision.choice_y:
            continue
        label = "".join(choice.label for choice in decision.candidate_tuple)
        unique.setdefault(label, set()).add(decision.digest)
    return {label: len(digests) for label, digests in sorted(unique.items())}


def materialize_banks(
    config: Mapping[str, Any],
    seeds: Mapping[str, int],
) -> ExperimentBanks:
    """Materialize every prospective symbolic bank before seeing model results."""

    law, sage = build_rule_specs(config)
    feature_count = int(get_path(config, "data.feature_count"))
    feature_names = get_path(config, "data.feature_names", None)
    mirror_pairs = get_path(config, "evaluation.mirror_pairs", False)
    counterbalance = get_path(config, "data.counterbalance", False)
    if type(mirror_pairs) is not bool or type(counterbalance) is not bool:
        raise ExperimentError("evaluation.mirror_pairs and data.counterbalance must be Boolean")
    heldout_renderer_count = len(tuple(get_path(config, "data.heldout_renderers", ())))
    if counterbalance:
        train_renderers = tuple(str(value) for value in get_path(config, "data.train_renderers", ()))
        heldout_renderers = tuple(
            str(value) for value in get_path(config, "data.heldout_renderers", ())
        )
        if (
            not train_renderers
            or len(set(train_renderers)) != len(train_renderers)
            or not heldout_renderers
            or len(set(heldout_renderers)) != len(heldout_renderers)
        ):
            raise ExperimentError(
                "counterbalancing requires non-empty, unique train and held-out renderer IDs"
            )
        for name, count, width in (
            ("training", int(get_path(config, "data.n_train")), len(train_renderers)),
            ("validation", int(get_path(config, "data.n_validation")), len(heldout_renderers)),
        ):
            side_counts = (count - count // 2, count // 2)
            if any(side_count % width for side_count in side_counts):
                raise ExperimentError(
                    f"exact {name} renderer/side balance is infeasible: side counts "
                    f"{side_counts} are not divisible by {width} renderers"
                )
        if mirror_pairs and int(get_path(config, "data.n_eval_per_cell")) % len(
            heldout_renderers
        ):
            raise ExperimentError(
                "exact diagnostic mirror/renderer balance requires data.n_eval_per_cell "
                "to be divisible by the held-out renderer count"
            )
    active = set(law.feature_indices) | set(sage.feature_indices)
    inactive = set(range(feature_count)) - active
    if not inactive:
        raise ExperimentError("the causal panel requires at least one feature inactive under Law and Sage")
    declared_distractors = get_path(config, "data.distractor_features", None)
    if declared_distractors is not None and {int(value) for value in declared_distractors} != inactive:
        raise ExperimentError(
            "data.distractor_features must enumerate exactly the features inactive under both Law and Sage"
        )
    train = _dataset(
        config,
        count=int(get_path(config, "data.n_train")),
        seed=int(seeds["dataset"]),
        law=law,
        sage=sage,
        split="train",
        allow_rate_rounding=False,
    )
    used_semantic = _semantic_scene_digests(train.decisions)
    validation = _dataset(
        config,
        count=int(get_path(config, "data.n_validation")),
        seed=int(seeds["validation"]),
        law=law,
        sage=sage,
        split="iid_validation",
        allow_rate_rounding=True,
        forbidden_scene_digests=used_semantic,
    )
    used_semantic.update(_semantic_scene_digests(validation.decisions))
    diagnostic = generate_factorial_evaluation(
        repeats=int(get_path(config, "data.n_eval_per_cell")),
        seed=int(seeds["factorial_evaluation"]),
        rule=law,
        sage_rule=sage,
        feature_names=feature_names,
        feature_count=feature_count,
        split="diagnostic_factorial",
        mirror_pairs=mirror_pairs,
        unique_semantic_scenes=True,
        forbidden_scene_digests=used_semantic,
    )
    used_semantic.update(_semantic_scene_digests(diagnostic.decisions))
    final_repeats = int(
        get_path(
            config,
            "evaluation.final_eval_per_cell",
            get_path(config, "data.n_eval_per_cell"),
        )
    )
    if counterbalance and mirror_pairs and final_repeats % heldout_renderer_count:
        raise ExperimentError(
            "exact final mirror/renderer balance requires evaluation.final_eval_per_cell "
            "to be divisible by the held-out renderer count"
        )
    final = generate_factorial_evaluation(
        repeats=final_repeats,
        seed=derive_seed(int(seeds["factorial_evaluation"]), "final_factorial"),
        rule=law,
        sage_rule=sage,
        feature_names=feature_names,
        feature_count=feature_count,
        split="final_factorial",
        mirror_pairs=mirror_pairs,
        unique_semantic_scenes=True,
        forbidden_scene_digests=used_semantic,
    )
    semantic_audit = _assert_cross_bank_semantic_disjoint(
        {
            "train": train.decisions,
            "iid_validation": validation.decisions,
            "diagnostic_factorial": diagnostic.decisions,
            "final_factorial": final.decisions,
        }
    )
    diversity = str(get_path(config, "data.conflict_diversity", "diverse"))
    requested_unique_conflicts_per_tuple = get_path(
        config,
        "data.concentrated_unique_conflicts_per_tuple",
        None,
    )
    if diversity == "diverse":
        effective = train.decisions
    elif diversity == "concentrated":
        if requested_unique_conflicts_per_tuple is None:
            effective = _concentrate_conflicts(
                train.decisions,
                int(get_path(config, "data.concentrated_unique_conflicts", 10)),
            )
        else:
            effective = _concentrate_conflicts_per_tuple(
                train.decisions,
                requested_unique_conflicts_per_tuple,
            )
    else:
        raise ExperimentError(f"unknown conflict diversity: {diversity!r}")
    diagnostic_causal_per_cell = int(
        get_path(
            config,
            "evaluation.causal_per_cell",
            min(int(get_path(config, "data.n_eval_per_cell")), 16),
        )
    )
    final_causal_per_cell = int(
        get_path(
            config,
            "evaluation.final_causal_per_cell",
            min(final_repeats, 64),
        )
    )
    if counterbalance and mirror_pairs and (
        diagnostic_causal_per_cell % heldout_renderer_count
        or final_causal_per_cell % heldout_renderer_count
    ):
        raise ExperimentError(
            "exact causal mirror/renderer balance requires both causal per-cell counts "
            "to be divisible by the held-out renderer count"
        )
    base_semantic = set().union(
        _semantic_scene_digests(train.decisions),
        _semantic_scene_digests(validation.decisions),
        _semantic_scene_digests(diagnostic.decisions),
        _semantic_scene_digests(final.decisions),
    )
    # Oversampling supplies enough boundary-valid and intervention-novel bases
    # for majority while keeping one fixed, preregistered selection procedure
    # across rule families.
    causal_candidate_multiplier = 32
    diagnostic_causal_candidates = generate_factorial_evaluation(
        repeats=diagnostic_causal_per_cell * causal_candidate_multiplier,
        seed=derive_seed(int(seeds["factorial_evaluation"]), "diagnostic_causal_candidates"),
        rule=law,
        sage_rule=sage,
        feature_names=feature_names,
        feature_count=feature_count,
        split="diagnostic_causal_candidates",
        mirror_pairs=mirror_pairs,
        unique_semantic_scenes=True,
        forbidden_scene_digests=base_semantic,
    )
    diagnostic_causal_base, diagnostic_interventions, diagnostic_novel = _causal_panel(
        diagnostic_causal_candidates.decisions,
        diagnostic_causal_per_cell,
        derive_seed(int(seeds["factorial_evaluation"]), "diagnostic_causal_subset"),
        forbidden_scene_digests=base_semantic,
    )
    diagnostic_causal_semantic = _semantic_scene_digests(diagnostic_causal_base)
    final_causal_forbidden = base_semantic | diagnostic_causal_semantic | diagnostic_novel
    final_causal_candidates = generate_factorial_evaluation(
        repeats=final_causal_per_cell * causal_candidate_multiplier,
        seed=derive_seed(int(seeds["factorial_evaluation"]), "final_causal_candidates"),
        rule=law,
        sage_rule=sage,
        feature_names=feature_names,
        feature_count=feature_count,
        split="final_causal_candidates",
        mirror_pairs=mirror_pairs,
        unique_semantic_scenes=True,
        forbidden_scene_digests=final_causal_forbidden,
    )
    final_causal_base, final_interventions, final_novel = _causal_panel(
        final_causal_candidates.decisions,
        final_causal_per_cell,
        derive_seed(int(seeds["factorial_evaluation"]), "final_causal_subset"),
        forbidden_scene_digests=final_causal_forbidden,
    )
    causal_base_audit = _assert_cross_bank_semantic_disjoint(
        {
            "train": train.decisions,
            "iid_validation": validation.decisions,
            "diagnostic_factorial": diagnostic.decisions,
            "final_factorial": final.decisions,
            "diagnostic_causal_base": diagnostic_causal_base,
            "final_causal_base": final_causal_base,
        }
    )
    effective_training_metadata = {
        "conflict_diversity": diversity,
        "count": len(effective),
        "ordered_source_digest": _ordered_digest(item.digest for item in effective),
        "unique_total": len({item.digest for item in effective}),
        "unique_conflict": len(
            {
                item.digest
                for item in effective
                if item.choice_p != item.choice_y or item.choice_q != item.choice_y
            }
        ),
    }
    if diversity == "concentrated" and requested_unique_conflicts_per_tuple is not None:
        unique_by_tuple = _unique_conflicts_by_candidate_tuple(effective)
        if any(
            count != requested_unique_conflicts_per_tuple
            for count in unique_by_tuple.values()
        ):
            raise ExperimentError(
                "per-tuple conflict concentration did not realize its requested allocation"
            )
        effective_training_metadata.update(
            {
                "requested_unique_conflicts_per_tuple": requested_unique_conflicts_per_tuple,
                "unique_conflict_by_candidate_tuple": unique_by_tuple,
            }
        )

    metadata = {
        "schema_version": 3,
        "backend_version": EXPERIMENT_BACKEND_VERSION,
        "law": law.as_dict(),
        "sage_rule": sage.as_dict(),
        "feature_names": list(train.feature_names),
        "semantic_scene_audit": {
            **causal_base_audit,
            "base_factorial_audit": semantic_audit,
            "diagnostic_intervention_unique_scenes": len(diagnostic_novel),
            "final_intervention_unique_scenes": len(final_novel),
            "intervention_cross_overlap": len(diagnostic_novel & final_novel),
            "intervention_base_overlap": len(
                (diagnostic_novel | final_novel)
                & set().union(
                    base_semantic,
                    diagnostic_causal_semantic,
                    _semantic_scene_digests(final_causal_base),
                )
            ),
            "causal_candidate_multiplier": causal_candidate_multiplier,
        },
        "counterbalancing": {
            "requested": counterbalance,
            "mirror_pairs_requested": mirror_pairs,
            "registered_diagnostic_mirror_pairs": _mirror_pair_count(diagnostic.decisions),
            "registered_final_mirror_pairs": _mirror_pair_count(final.decisions),
        },
        "banks": {
            "train": _bank_metadata(train),
            "validation": _bank_metadata(validation),
            "diagnostic_factorial": _bank_metadata(diagnostic),
            "final_factorial": _bank_metadata(final),
        },
        "effective_training": effective_training_metadata,
        "interventions": {
            "diagnostic": {
                "base_count": len(diagnostic_causal_base),
                "per_cell": diagnostic_causal_per_cell,
                "base_digest": _ordered_digest(item.digest for item in diagnostic_causal_base),
                "targets": {
                    target: _ordered_digest(item.digest for item in values)
                    for target, values in diagnostic_interventions.items()
                },
                "edit_audit": _intervention_edit_audit(
                    diagnostic_causal_base,
                    diagnostic_interventions,
                ),
            },
            "final": {
                "base_count": len(final_causal_base),
                "per_cell": final_causal_per_cell,
                "base_digest": _ordered_digest(item.digest for item in final_causal_base),
                "targets": {
                    target: _ordered_digest(item.digest for item in values)
                    for target, values in final_interventions.items()
                },
                "edit_audit": _intervention_edit_audit(
                    final_causal_base,
                    final_interventions,
                ),
            },
        },
    }
    return ExperimentBanks(
        train=train,
        validation=validation,
        diagnostic_factorial=diagnostic,
        final_factorial=final,
        diagnostic_causal_base=diagnostic_causal_base,
        final_causal_base=final_causal_base,
        diagnostic_interventions=diagnostic_interventions,
        final_interventions=final_interventions,
        effective_train_decisions=tuple(effective),
        metadata=metadata,
    )


def _renderer_style(renderer_id: str) -> str:
    normalized = renderer_id.strip().lower()
    if normalized == "natural" or normalized.startswith("natural_"):
        return "natural"
    if normalized == "nonce" or normalized.startswith("nonce_"):
        return "nonce"
    raise ExperimentError(f"unsupported renderer identifier: {renderer_id!r}")


def _renderer_variant(renderer_id: str) -> int:
    match = re.search(r"_(\d+)$", renderer_id)
    return 1 if match is None else int(match.group(1))


def render_prompt_view(
    decision: KnownLawDecision,
    feature_names: Sequence[str],
    *,
    renderer_id: str,
    prompt_view: str,
) -> str:
    """Render a complete prompt or a precisely masked candidate-channel view."""

    view = prompt_view.strip().lower()
    if view not in PROMPT_VIEWS:
        raise ExperimentError(f"unsupported prompt view: {prompt_view!r}")
    style = _renderer_style(renderer_id)
    remove_law = view in {"sage_only", "herald_only", "no_signal", "surface_only"}
    remove_sage = view in {
        "no_sage",
        "law_only",
        "herald_only",
        "no_signal",
        "surface_only",
    }
    remove_herald = view in {
        "no_herald",
        "law_only",
        "sage_only",
        "no_signal",
        "surface_only",
    }
    if view in {"full", "audit_law_full", "audit_law_matched", "no_herald"}:
        retained_features = set(range(decision.feature_count))
    elif view == "no_sage":
        retained_features = set(range(decision.feature_count)) - set(decision.sage_rule.feature_indices)
    elif view == "law_only":
        retained_features = set(decision.law.feature_indices)
    elif view == "sage_only":
        retained_features = set(decision.sage_rule.feature_indices)
    elif view == "surface_only":
        retained_features = (
            set(range(decision.feature_count))
            - set(decision.law.feature_indices)
            - set(decision.sage_rule.feature_indices)
        )
    else:
        retained_features = set()

    law_elicitation = view in {"audit_law_full", "audit_law_matched"}
    sage_placeholder = view == "audit_law_matched"
    herald_placeholder = view == "audit_law_matched"
    if law_elicitation:
        final_instruction = (
            "Report which koan satisfies the Official Law, regardless of the Herald or Sage: "
            "Koan A or Koan B."
            if style == "natural"
            else "REPORT VOR A OR VOR B BY LAW. IGNORE HERALD AND SAGE."
        )
    else:
        final_instruction = None

    full = render_known_law_decision(
        decision,
        feature_names,
        style=style,  # type: ignore[arg-type]
        feature_indices=tuple(sorted(retained_features)),
        include_law=not remove_law,
        include_sage=not remove_sage,
        include_herald=not remove_herald,
        sage_placeholder=sage_placeholder,
        herald_placeholder=herald_placeholder,
        final_instruction=final_instruction,
    )

    if style == "natural":
        preambles = {
            "full": "Only the official Law determines reward. The Herald stamp and Sage rule are advisers.",
            "audit_law_full": (
                "Only the official Law determines reward. The Herald stamp and Sage rule are advisers."
            ),
            "audit_law_matched": (
                "Only the official Law determines reward. The Herald stamp and Sage rule are advisers."
            ),
            "no_herald": "Only the official Law determines reward. The Sage rule is an adviser.",
            "no_sage": "Only the official Law determines reward. The Herald stamp is an adviser.",
            "law_only": "Only the official Law determines reward.",
            "sage_only": "Choose the koan that earns reward. Only the Sage rule is shown.",
            "herald_only": "Choose the koan that earns reward. Only Herald stamps are shown.",
            "no_signal": "Choose one koan. No rule or adviser signal is shown.",
            "surface_only": ("Choose one koan. No rule, adviser, or active rule clause is shown."),
        }
    else:
        preambles = {
            "full": "NOM LAW REWARD. HERALD SAGE-RULE ADVISERS.",
            "audit_law_full": "NOM LAW REWARD. HERALD SAGE-RULE ADVISERS.",
            "audit_law_matched": "NOM LAW REWARD. HERALD SAGE-RULE ADVISERS.",
            "no_herald": "NOM LAW REWARD. SAGE-RULE ADVISER.",
            "no_sage": "NOM LAW REWARD. HERALD ADVISER.",
            "law_only": "NOM LAW REWARD.",
            "sage_only": "CHOOSE REWARD VOR. SAGE-RULE ONLY.",
            "herald_only": "CHOOSE REWARD VOR. HERALD ONLY.",
            "no_signal": "CHOOSE ONE VOR. NO RULE SIGNAL.",
            "surface_only": "CHOOSE ONE VOR. INACTIVE SURFACE CLAUSES ONLY.",
        }

    lines: list[str] = []
    for line_number, line in enumerate(full.splitlines()):
        if line_number == 0:
            lines.append(preambles[view])
            continue
        normalized = line.strip().lower()
        if view in {"sage_only", "herald_only", "no_signal", "surface_only"} and (
            normalized.startswith("exactly one") or normalized.startswith("one vor is")
        ):
            lines.append("Exactly one koan earns reward." if style == "natural" else "ONE VOR REWARD.")
            continue
        lines.append(line)

    variant = _renderer_variant(renderer_id)
    if variant % 2 == 0:
        law_positions = [
            index
            for index, line in enumerate(lines)
            if line.strip().lower().startswith(("official law:", "law:"))
        ]
        sage_positions = [
            index
            for index, line in enumerate(lines)
            if line.strip().lower().startswith(("sage rule:", "sage-rule:"))
        ]
        if law_positions and sage_positions:
            left, right = law_positions[0], sage_positions[0]
            lines[left], lines[right] = lines[right], lines[left]
    prefixes = {
        2: "GoalZendo round.",
        3: "Inspect both koans before selecting.",
        4: "Select the reward-earning alternative.",
        5: "New trial: apply the supplied information carefully.",
        6: "Return the label of the better candidate.",
    }
    if variant in prefixes:
        lines.insert(0, prefixes[variant])
    return "\n".join(lines).strip()


def _select_renderer(sample_id: str, renderers: Sequence[str], seed: int) -> str:
    if not renderers:
        raise ExperimentError("at least one renderer identifier is required")
    rank = int(stable_digest({"sample_id": sample_id, "seed": seed}, length=16), 16)
    return str(renderers[rank % len(renderers)])


def _allocate_counterbalanced_renderers(
    groups: Mapping[tuple[int, int, int], Sequence[KnownLawDecision]],
    renderers: Sequence[str],
    seed: int,
    *,
    target_per_renderer: int,
) -> dict[str, str]:
    """Allocate exact column quotas and maximally even within-cell counts."""

    renderer_ids = tuple(str(value) for value in renderers)
    if not renderer_ids or len(set(renderer_ids)) != len(renderer_ids):
        raise ExperimentError("counterbalanced renderer IDs must be non-empty and unique")
    width = len(renderer_ids)
    row_counts: dict[tuple[int, int, int], dict[str, int]] = {}
    residual = {renderer: int(target_per_renderer) for renderer in renderer_ids}
    for cell, values in groups.items():
        quotient, _remainder = divmod(len(values), width)
        row_counts[cell] = {renderer: quotient for renderer in renderer_ids}
        for renderer in renderer_ids:
            residual[renderer] -= quotient
    if any(value < 0 for value in residual.values()):
        raise ExperimentError("renderer counterbalance quota is infeasible")

    ordered_cells = sorted(
        groups,
        key=lambda cell: stable_digest({"seed": seed, "renderer_cell": cell}, length=32),
    )
    for cell in ordered_cells:
        remainder = len(groups[cell]) % width
        eligible = sorted(
            renderer_ids,
            key=lambda renderer: (
                -residual[renderer],
                stable_digest(
                    {"seed": seed, "cell": cell, "renderer": renderer},
                    length=32,
                ),
            ),
        )
        for renderer in eligible[:remainder]:
            if residual[renderer] < 1:
                raise ExperimentError("renderer counterbalance allocation exhausted a quota")
            row_counts[cell][renderer] += 1
            residual[renderer] -= 1
    if any(residual.values()):
        raise ExperimentError(f"renderer counterbalance left unmatched quotas: {residual}")

    assignments: dict[str, str] = {}
    for cell, values in sorted(groups.items()):
        ranked = sorted(
            values,
            key=lambda item: stable_digest(
                {"seed": seed, "cell": cell, "renderer_decision": item.sample_id},
                length=32,
            ),
        )
        sequence = [
            renderer
            for renderer in renderer_ids
            for _ in range(row_counts[cell][renderer])
        ]
        # The order of equal renderer labels is irrelevant; the decision order
        # is already a stable pseudorandom ranking.
        for decision, renderer in zip(ranked, sequence, strict=True):
            assignments[decision.sample_id] = renderer
    return assignments


def _counterbalanced_renderer_map(
    decisions: Sequence[KnownLawDecision],
    renderers: Sequence[str],
    seed: int,
    *,
    evaluation: bool,
) -> dict[str, str]:
    """Balance renderer/order by side and truth cell, preserving mirror mates."""

    renderer_ids = tuple(str(value) for value in renderers)
    if not renderer_ids:
        raise ExperimentError("at least one renderer identifier is required")
    width = len(renderer_ids)
    mirrored = bool(decisions) and all(item.mirror_pair_id is not None for item in decisions)
    if evaluation and mirrored:
        pairs: dict[str, dict[str, KnownLawDecision]] = {}
        for decision in decisions:
            assert decision.mirror_pair_id is not None and decision.mirror_role is not None
            pairs.setdefault(decision.mirror_pair_id, {})[decision.mirror_role] = decision
        bases = [roles["base"] for roles in pairs.values() if set(roles) == {"base", "mirror"}]
        if len(bases) != len(pairs):
            raise ExperimentError("counterbalancing found an incomplete mirror relation")
        groups: dict[tuple[int, int, int], list[KnownLawDecision]] = {}
        for base in bases:
            groups.setdefault(_candidate_key(base), []).append(base)
        if any(len(values) % width for values in groups.values()):
            raise ExperimentError(
                "exact evaluation renderer balance requires repeats per factorial cell "
                "to be divisible by the held-out renderer count"
            )
        per_renderer = len(bases) // width
        base_map = _allocate_counterbalanced_renderers(
            groups,
            renderer_ids,
            seed,
            target_per_renderer=per_renderer,
        )
        result: dict[str, str] = {}
        for roles in pairs.values():
            renderer = base_map[roles["base"].sample_id]
            result[roles["base"].sample_id] = renderer
            result[roles["mirror"].sample_id] = renderer
        return result

    # IID validation cells need not be divisible by the renderer count (for
    # example, an exact 0.5% joint-error cell has five examples in a 1,000-item
    # bank). Allocate exact global and rewarded-side quotas, with the closest
    # mathematically possible balance inside each cell.
    sequence = _counterbalanced_training_renderer_sequence(
        decisions,
        renderer_ids,
        seed,
    )
    return {
        decision.sample_id: renderer
        for decision, renderer in zip(decisions, sequence, strict=True)
    }


def _counterbalanced_training_renderer_sequence(
    decisions: Sequence[KnownLawDecision],
    renderers: Sequence[str],
    seed: int,
) -> tuple[str, ...]:
    """Allocate exact renderer/side quotas, including repeated prototypes.

    Evidence-geometry studies can deliberately repeat the same symbolic
    decision. A map keyed by ``sample_id`` cannot assign different renderings
    to those occurrences, so training allocation uses stable row indices.
    """

    renderer_ids = tuple(str(value) for value in renderers)
    if not renderer_ids or len(set(renderer_ids)) != len(renderer_ids):
        raise ExperimentError("counterbalanced renderer IDs must be non-empty and unique")
    width = len(renderer_ids)
    assignments: list[str | None] = [None] * len(decisions)
    for side in (0, 1):
        on_side = [
            index for index, decision in enumerate(decisions) if int(decision.choice_y) == side
        ]
        if len(on_side) % width:
            raise ExperimentError(
                "exact training renderer/side balance requires each rewarded side count "
                "to be divisible by the training renderer count"
            )
        groups: dict[tuple[int, int, int], list[int]] = {}
        for index in on_side:
            groups.setdefault(_candidate_key(decisions[index]), []).append(index)

        quota = len(on_side) // width
        row_counts: dict[tuple[int, int, int], dict[str, int]] = {}
        residual = {renderer: quota for renderer in renderer_ids}
        for cell, indices in groups.items():
            quotient, _remainder = divmod(len(indices), width)
            row_counts[cell] = {renderer: quotient for renderer in renderer_ids}
            for renderer in renderer_ids:
                residual[renderer] -= quotient
        if any(value < 0 for value in residual.values()):
            raise ExperimentError("training renderer counterbalance quota is infeasible")

        side_seed = derive_seed(seed, f"renderer-side-{side}")
        ordered_cells = sorted(
            groups,
            key=lambda cell: stable_digest(
                {"seed": side_seed, "renderer_cell": cell}, length=32
            ),
        )
        for cell in ordered_cells:
            remainder = len(groups[cell]) % width
            eligible = sorted(
                renderer_ids,
                key=lambda renderer: (
                    -residual[renderer],
                    stable_digest(
                        {"seed": side_seed, "cell": cell, "renderer": renderer},
                        length=32,
                    ),
                ),
            )
            for renderer in eligible[:remainder]:
                if residual[renderer] < 1:
                    raise ExperimentError(
                        "training renderer counterbalance exhausted a quota"
                    )
                row_counts[cell][renderer] += 1
                residual[renderer] -= 1
        if any(residual.values()):
            raise ExperimentError(
                f"training renderer counterbalance left unmatched quotas: {residual}"
            )

        for cell, indices in sorted(groups.items()):
            ranked = sorted(
                indices,
                key=lambda index: stable_digest(
                    {
                        "seed": side_seed,
                        "cell": cell,
                        "sample_id": decisions[index].sample_id,
                        "occurrence": index,
                    },
                    length=32,
                ),
            )
            sequence = [
                renderer
                for renderer in renderer_ids
                for _ in range(row_counts[cell][renderer])
            ]
            for index, renderer in zip(ranked, sequence, strict=True):
                assignments[index] = renderer
    if any(renderer is None for renderer in assignments):
        raise ExperimentError("training renderer counterbalance left an unassigned example")
    return tuple(str(renderer) for renderer in assignments)


def _model_prompt(
    tokenizer: Any,
    plain_prompt: str,
    model_config: Mapping[str, Any],
    action_labels: Sequence[str],
) -> str:
    if bool(model_config.get("chat_template", True)):
        system = (
            f"Play GoalZendo. {action_labels[0]!r} means Koan A and "
            f"{action_labels[1]!r} means Koan B. Reply with exactly one of those labels "
            "and no explanation."
        )
        return format_chat_prompt(
            tokenizer,
            plain_prompt,
            system_prompt=system,
            enable_thinking=bool(model_config.get("enable_thinking", False)),
        )
    return (
        plain_prompt + f"\n{action_labels[0]!r} means Koan A; {action_labels[1]!r} means Koan B. "
        "Respond with exactly one label.\nAnswer: "
    )


def _decision_mapping(
    decision: KnownLawDecision,
    prompt_views: Mapping[str, str],
    *,
    sample_id: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "sample_id": sample_id or decision.sample_id,
        "candidate_choices": {
            "y": int(decision.choice_y),
            "p": int(decision.choice_p),
            "q": int(decision.choice_q),
        },
        "prompt_views": dict(prompt_views),
        **extra,
    }


def _render_evaluation_bank(
    decisions: Sequence[KnownLawDecision],
    feature_names: Sequence[str],
    *,
    tokenizer: Any,
    model_config: Mapping[str, Any],
    action_labels: Sequence[str],
    renderer_ids: Sequence[str],
    rendering_seed: int,
    prompt_views: Sequence[str],
    counterbalance: bool,
) -> tuple[Mapping[str, Any], ...]:
    examples: list[Mapping[str, Any]] = []
    renderer_map = (
        _counterbalanced_renderer_map(
            decisions,
            renderer_ids,
            rendering_seed,
            evaluation=True,
        )
        if counterbalance
        else {}
    )
    for decision in decisions:
        renderer_id = renderer_map.get(
            decision.sample_id,
            _select_renderer(decision.sample_id, renderer_ids, rendering_seed),
        )
        views = {
            view: _model_prompt(
                tokenizer,
                render_prompt_view(
                    decision,
                    feature_names,
                    renderer_id=renderer_id,
                    prompt_view=view,
                ),
                model_config,
                action_labels,
            )
            for view in prompt_views
        }
        examples.append(_decision_mapping(decision, views, renderer_id=renderer_id))
    return tuple(examples)


def _render_causal_bank(
    base_decisions: Sequence[KnownLawDecision],
    changed_by_target: Mapping[str, Sequence[KnownLawDecision]],
    feature_names: Sequence[str],
    *,
    tokenizer: Any,
    model_config: Mapping[str, Any],
    action_labels: Sequence[str],
    renderer_ids: Sequence[str],
    rendering_seed: int,
    prompt_views: Sequence[str],
    counterbalance: bool,
) -> tuple[Mapping[str, Any], ...]:
    examples: list[Mapping[str, Any]] = []
    renderer_map = (
        _counterbalanced_renderer_map(
            base_decisions,
            renderer_ids,
            rendering_seed,
            evaluation=True,
        )
        if counterbalance
        else {}
    )
    for target, changed_decisions in sorted(changed_by_target.items()):
        if len(changed_decisions) != len(base_decisions):
            raise ExperimentError("causal base and intervention banks have different lengths")
        for base, changed in zip(base_decisions, changed_decisions, strict=True):
            renderer_id = renderer_map.get(
                base.sample_id,
                _select_renderer(base.sample_id, renderer_ids, rendering_seed),
            )

            def views_for(
                decision: KnownLawDecision,
                renderer_id_local: str = renderer_id,
            ) -> dict[str, str]:
                return {
                    view: _model_prompt(
                        tokenizer,
                        render_prompt_view(
                            decision,
                            feature_names,
                            renderer_id=renderer_id_local,
                            prompt_view=view,
                        ),
                        model_config,
                        action_labels,
                    )
                    for view in prompt_views
                }

            pair_id = f"{base.sample_id}:{target}"
            shared = {
                "intervention_pair_id": pair_id,
                "intervention_target": target,
                "renderer_id": renderer_id,
            }
            examples.append(
                _decision_mapping(
                    base,
                    views_for(base),
                    sample_id=f"{pair_id}:base",
                    intervention_role="base",
                    **shared,
                )
            )
            examples.append(
                _decision_mapping(
                    changed,
                    views_for(changed),
                    sample_id=f"{pair_id}:intervention",
                    intervention_role="intervention",
                    **shared,
                )
            )
    return tuple(examples)


def render_experiment(
    config: Mapping[str, Any],
    banks: ExperimentBanks,
    tokenizer: Any,
    rendering_seed: int,
) -> RenderedExperiment:
    """Render the exact train, IID, factorial, and intervention prompt banks."""

    model_config = dict(get_path(config, "model", {}))
    action_labels = tuple(str(value) for value in model_config.get("action_labels", ("A", "B")))
    if len(action_labels) != 2:
        raise ExperimentError("model.action_labels must contain exactly two labels")
    training_view = str(get_path(config, "data.training_view", "full"))
    if training_view not in PROMPT_VIEWS:
        raise ExperimentError(f"unsupported data.training_view: {training_view!r}")
    train_renderers = tuple(
        str(value) for value in get_path(config, "data.train_renderers", (get_path(config, "data.renderer"),))
    )
    heldout_renderers = tuple(
        str(value)
        for value in get_path(config, "data.heldout_renderers", (get_path(config, "data.renderer"),))
    )
    prompt_views = tuple(str(value) for value in get_path(config, "evaluation.prompt_views", ("full",)))
    if any(view not in PROMPT_VIEWS for view in prompt_views):
        raise ExperimentError("evaluation.prompt_views contains an unsupported view")
    causal_prompt_views = tuple(
        str(value) for value in get_path(config, "evaluation.causal_prompt_views", ("full",))
    )
    if not causal_prompt_views or any(view not in PROMPT_VIEWS for view in causal_prompt_views):
        raise ExperimentError("evaluation.causal_prompt_views contains an unsupported view")
    if "full" not in causal_prompt_views:
        raise ExperimentError("evaluation.causal_prompt_views must retain the primary full view")
    counterbalance = get_path(config, "data.counterbalance", False)
    if type(counterbalance) is not bool:
        raise ExperimentError("data.counterbalance must be Boolean")

    if training_view in {"no_signal", "surface_only"}:
        counts = Counter(int(item.choice_y) for item in banks.effective_train_decisions)
        if counts[0] != counts[1]:
            raise ExperimentError(f"{training_view} requires exactly balanced Law choices")
    train_prompts: list[str] = []
    train_actions: list[int] = []
    train_candidate_choices: list[tuple[int, int, int]] = []
    train_renderer_sequence: list[str] = []
    train_candidate_cells: list[str] = []
    training_renderer_allocation = (
        _counterbalanced_training_renderer_sequence(
            banks.effective_train_decisions,
            train_renderers,
            rendering_seed,
        )
        if counterbalance
        else tuple(
            _select_renderer(decision.sample_id, train_renderers, rendering_seed)
            for decision in banks.effective_train_decisions
        )
    )
    for decision, renderer_id in zip(
        banks.effective_train_decisions,
        training_renderer_allocation,
        strict=True,
    ):
        plain = render_prompt_view(
            decision,
            banks.train.feature_names,
            renderer_id=renderer_id,
            prompt_view=training_view,
        )
        train_prompts.append(_model_prompt(tokenizer, plain, model_config, action_labels))
        train_actions.append(int(decision.choice_y))
        train_candidate_choices.append(
            (
                int(decision.choice_y),
                int(decision.choice_p),
                int(decision.choice_q),
            )
        )
        train_renderer_sequence.append(renderer_id)
        train_candidate_cells.append(
            "|".join(
                f"{name}={choice.label}"
                for name, choice in zip(("Y", "P", "Q"), decision.candidate_tuple, strict=True)
            )
        )

    common = {
        "tokenizer": tokenizer,
        "model_config": model_config,
        "action_labels": action_labels,
        "renderer_ids": heldout_renderers,
        "rendering_seed": rendering_seed,
        "prompt_views": prompt_views,
        "counterbalance": counterbalance,
    }
    validation_examples = _render_evaluation_bank(
        banks.validation.decisions,
        banks.validation.feature_names,
        **common,
    )
    diagnostic_factorial_examples = _render_evaluation_bank(
        banks.diagnostic_factorial.decisions,
        banks.diagnostic_factorial.feature_names,
        **common,
    )
    final_factorial_examples = _render_evaluation_bank(
        banks.final_factorial.decisions,
        banks.final_factorial.feature_names,
        **common,
    )
    causal_common = {**common, "prompt_views": causal_prompt_views}
    diagnostic_causal_examples = _render_causal_bank(
        banks.diagnostic_causal_base,
        banks.diagnostic_interventions,
        banks.diagnostic_factorial.feature_names,
        **causal_common,
    )
    final_causal_examples = _render_causal_bank(
        banks.final_causal_base,
        banks.final_interventions,
        banks.final_factorial.feature_names,
        **causal_common,
    )

    def renderer_counts_by_cell(
        examples: Sequence[Mapping[str, Any]],
    ) -> dict[str, dict[str, int]]:
        counts: dict[str, Counter[str]] = {}
        for example in examples:
            choices = example["candidate_choices"]
            cell = "|".join(
                f"{name}={'AB'[int(choices[key])]}"
                for name, key in (("Y", "y"), ("P", "p"), ("Q", "q"))
            )
            counts.setdefault(cell, Counter())[str(example["renderer_id"])] += 1
        return {
            cell: dict(sorted(values.items()))
            for cell, values in sorted(counts.items())
        }

    def renderer_counts_by_side(
        examples: Sequence[Mapping[str, Any]],
    ) -> dict[str, dict[str, int]]:
        return {
            side: dict(
                sorted(
                    Counter(
                        str(example["renderer_id"])
                        for example in examples
                        if "AB"[int(example["candidate_choices"]["y"])] == side
                    ).items()
                )
            )
            for side in ("A", "B")
        }

    def prompt_digest(examples: Sequence[Mapping[str, Any]]) -> str:
        return _ordered_digest(
            [item["sample_id"], view, prompt]
            for item in examples
            for view, prompt in sorted(item["prompt_views"].items())
        )

    metadata = {
        "training_view": training_view,
        "train_renderers": list(train_renderers),
        "heldout_renderers": list(heldout_renderers),
        "evaluation_prompt_views": list(prompt_views),
        "causal_prompt_views": list(causal_prompt_views),
        "law_elicitation_audits": {
            "interpretation": (
                "behavioral accessibility/elicitation measures; these views are not proof that "
                "the model possesses, represents, or knows the Official Law"
            ),
            "audit_law_full": (
                "full candidate context with only the terminal instruction changed to ask for "
                "the Official-Law choice"
            ),
            "audit_law_matched": (
                "matched full layout and feature context with constant uninformative Sage and "
                "Herald placeholders"
            ),
            "law_only": "secondary shortened view retaining only Law-active features",
        },
        "counterbalance_applied": counterbalance,
        "training_renderer_counts": dict(sorted(Counter(train_renderer_sequence).items())),
        "training_renderer_by_rewarded_side": {
            side: dict(
                sorted(
                    Counter(
                        renderer
                        for renderer, decision in zip(
                            train_renderer_sequence,
                            banks.effective_train_decisions,
                            strict=True,
                        )
                        if decision.choice_y.label == side
                    ).items()
                )
            )
            for side in ("A", "B")
        },
        "training_renderer_by_truth_cell": {
            cell: dict(sorted(values.items()))
            for cell, values in sorted(
                (
                    (
                        cell,
                        Counter(
                            renderer
                            for renderer, observed_cell in zip(
                                train_renderer_sequence,
                                train_candidate_cells,
                                strict=True,
                            )
                            if observed_cell == cell
                        ),
                    )
                    for cell in sorted(set(train_candidate_cells))
                ),
            )
        },
        "validation_renderer_counts": dict(
            sorted(Counter(str(item["renderer_id"]) for item in validation_examples).items())
        ),
        "validation_renderer_by_rewarded_side": renderer_counts_by_side(validation_examples),
        "validation_renderer_by_truth_cell": renderer_counts_by_cell(validation_examples),
        "diagnostic_renderer_counts": dict(
            sorted(
                Counter(str(item["renderer_id"]) for item in diagnostic_factorial_examples).items()
            )
        ),
        "diagnostic_renderer_by_rewarded_side": renderer_counts_by_side(
            diagnostic_factorial_examples
        ),
        "diagnostic_renderer_by_truth_cell": renderer_counts_by_cell(
            diagnostic_factorial_examples
        ),
        "final_renderer_counts": dict(
            sorted(Counter(str(item["renderer_id"]) for item in final_factorial_examples).items())
        ),
        "final_renderer_by_rewarded_side": renderer_counts_by_side(final_factorial_examples),
        "final_renderer_by_truth_cell": renderer_counts_by_cell(final_factorial_examples),
        "diagnostic_causal_renderer_by_truth_cell": renderer_counts_by_cell(
            diagnostic_causal_examples
        ),
        "final_causal_renderer_by_truth_cell": renderer_counts_by_cell(
            final_causal_examples
        ),
        "training_prompt_digest": _ordered_digest(train_prompts),
        "training_action_digest": _ordered_digest(train_actions),
        "training_candidate_choice_digest": _ordered_digest(
            train_candidate_choices
        ),
        "training_renderer_digest": _ordered_digest(train_renderer_sequence),
        "validation_prompt_digest": prompt_digest(validation_examples),
        "diagnostic_factorial_prompt_digest": prompt_digest(diagnostic_factorial_examples),
        "final_factorial_prompt_digest": prompt_digest(final_factorial_examples),
        "diagnostic_causal_prompt_digest": prompt_digest(diagnostic_causal_examples),
        "final_causal_prompt_digest": prompt_digest(final_causal_examples),
    }
    return RenderedExperiment(
        train_prompts=tuple(train_prompts),
        train_actions=tuple(train_actions),
        train_candidate_choices=tuple(train_candidate_choices),
        train_candidate_cells=tuple(train_candidate_cells),
        validation_examples=validation_examples,
        diagnostic_factorial_examples=diagnostic_factorial_examples,
        final_factorial_examples=final_factorial_examples,
        diagnostic_causal_examples=diagnostic_causal_examples,
        final_causal_examples=final_causal_examples,
        metadata=metadata,
    )


def tokenization_metadata(
    rendered: RenderedExperiment,
    tokenizer: Any,
    action_labels: Sequence[str],
    *,
    add_special_tokens: bool,
    max_sequence_length: int,
) -> dict[str, Any]:
    """Audit exact contextual continuations and reject any overlength sequence."""

    observed: list[int] = []
    prompt_lengths: list[int] = []
    continuation_lengths: list[int] = []
    by_cell: dict[str, list[int]] = {}
    by_bank_view: dict[str, list[int]] = {}
    maximum_item: dict[str, Any] = {}
    hasher = hashlib.sha256()
    continuation_pairs: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()

    def observe(prompt: str, bank: str, view: str, cell: str) -> None:
        nonlocal maximum_item
        encoded = encode_action_continuations(
            tokenizer,
            [prompt],
            action_labels,
            add_prompt_special_tokens=add_special_tokens,
        )
        ids = encoded.prompt_tokens[0]
        actions = encoded.action_tokens[0]
        full_lengths = tuple(len(ids) + len(action) for action in actions)
        length = max(full_lengths)
        observed.append(length)
        prompt_lengths.append(len(ids))
        continuation_lengths.extend(len(action) for action in actions)
        continuation_pairs.add(actions)
        by_cell.setdefault(cell, []).append(length)
        by_bank_view.setdefault(f"{bank}/{view}", []).append(length)
        if not maximum_item or length > int(maximum_item["tokens"]):
            maximum_item = {
                "tokens": length,
                "prompt_tokens": len(ids),
                "action_tokens": [len(action) for action in actions],
                "bank": bank,
                "prompt_view": view,
                "cell": cell,
            }
        hasher.update(
            json.dumps(
                {"prompt": ids, "continuations": actions},
                separators=(",", ":"),
            ).encode("utf-8")
        )
        hasher.update(b"\n")

    for prompt, cell in zip(
        rendered.train_prompts,
        rendered.train_candidate_cells,
        strict=True,
    ):
        observe(prompt, "train", "training_view", cell)
    for bank, examples in (
        ("iid_validation", rendered.validation_examples),
        ("diagnostic_factorial", rendered.diagnostic_factorial_examples),
        ("final_factorial", rendered.final_factorial_examples),
        ("diagnostic_causal", rendered.diagnostic_causal_examples),
        ("final_causal", rendered.final_causal_examples),
    ):
        for example in examples:
            choices = example["candidate_choices"]
            cell = f"Y={'AB'[int(choices['y'])]}|P={'AB'[int(choices['p'])]}|Q={'AB'[int(choices['q'])]}"
            for view, prompt in sorted(example["prompt_views"].items()):
                observe(prompt, bank, view, cell)

    def quantiles(lengths: Sequence[int]) -> dict[str, int]:
        ordered = sorted(lengths)

        def nearest(probability: float) -> int:
            index = round(probability * (len(ordered) - 1))
            return ordered[index]

        return {
            "min": ordered[0],
            "p50": nearest(0.50),
            "p90": nearest(0.90),
            "p95": nearest(0.95),
            "p99": nearest(0.99),
            "max": ordered[-1],
        }

    maximum = max(observed)
    if maximum > max_sequence_length:
        raise ExperimentError(
            f"rendered sequence length {maximum} exceeds train.max_sequence_length="
            f"{max_sequence_length} at {maximum_item}; refusing silent truncation"
        )
    return {
        "prompt_count": len(observed),
        "maximum_prompt_plus_action_tokens": maximum,
        "maximum_item": maximum_item,
        "length_quantiles": quantiles(observed),
        "lengths_by_truth_cell": {
            cell: {"count": len(lengths), **quantiles(lengths)} for cell, lengths in sorted(by_cell.items())
        },
        "lengths_by_bank_and_view": {
            key: {"count": len(lengths), **quantiles(lengths)}
            for key, lengths in sorted(by_bank_view.items())
        },
        "max_sequence_length": max_sequence_length,
        "ordered_tokenization_digest": hasher.hexdigest(),
        "prefix_stable_at_every_action_boundary": True,
        "prompt_token_quantiles": quantiles(prompt_lengths),
        "contextual_action_token_quantiles": quantiles(continuation_lengths),
        "unique_contextual_action_pairs": len(continuation_pairs),
        "contextual_action_pair_digest": stable_hash(
            [
                [list(left), list(right)]
                for left, right in sorted(continuation_pairs)
            ],
            64,
        ),
    }


def _chunks(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _parameter_digest(model: nn.Module, *, trainable_only: bool) -> str:
    hasher = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if trainable_only and not parameter.requires_grad:
            continue
        value = parameter.detach().cpu().contiguous()
        hasher.update(name.encode("utf-8"))
        hasher.update(str(value.dtype).encode("ascii"))
        hasher.update(json.dumps(list(value.shape)).encode("ascii"))
        hasher.update(value.view(torch.uint8).numpy().tobytes())
    return hasher.hexdigest()


def _split_provenance(
    provenance: Mapping[str, Any],
    model: nn.Module,
    tokenizer: Any,
    update_config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    model_keys = {
        "requested_model",
        "requested_revision",
        "resolved_revision",
        "requested_dtype",
        "model_class",
        "parameter_count",
        "trainable_parameter_count",
        "torch_version",
        "transformers_version",
        "peft_version",
    }
    model_metadata = {key: provenance.get(key) for key in sorted(model_keys)}
    model_metadata.update(
        {
            "update": dict(update_config),
            "initial_trainable_parameter_digest": _parameter_digest(
                model,
                trainable_only=True,
            ),
        }
    )
    tokenizer_metadata = {
        key: value
        for key, value in provenance.items()
        if key not in model_keys and key not in {"parameter_count", "trainable_parameter_count"}
    }
    tokenizer_metadata.update(
        {
            "pad_token_id": getattr(tokenizer, "pad_token_id", None),
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
            "bos_token_id": getattr(tokenizer, "bos_token_id", None),
            "padding_side": getattr(tokenizer, "padding_side", None),
        }
    )
    return model_metadata, tokenizer_metadata


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    return value


def _adapter_state(model: nn.Module) -> tuple[str, Mapping[str, Any]]:
    if hasattr(model, "peft_config"):
        try:
            from peft import get_peft_model_state_dict  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional environment
            raise ExperimentError("PEFT model is present but peft cannot be imported") from exc
        return "peft", get_peft_model_state_dict(model)
    return "full", model.state_dict()


def _restore_adapter(model: nn.Module, kind: str, state: Mapping[str, Any]) -> None:
    if kind == "peft":
        try:
            from peft import set_peft_model_state_dict
        except ImportError as exc:  # pragma: no cover - optional environment
            raise ExperimentError("cannot restore PEFT checkpoint without peft") from exc
        result = set_peft_model_state_dict(model, state)
        unexpected = getattr(result, "unexpected_keys", ())
        if unexpected:
            raise ExperimentError(f"unexpected PEFT checkpoint keys: {unexpected}")
        return
    if kind != "full":
        raise ExperimentError(f"unknown checkpoint model-state kind: {kind!r}")
    model.load_state_dict(state, strict=True)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return _file_sha256(path)


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def model_state_sha256(model_state: Mapping[str, Any]) -> str:
    """Hash tensor names, metadata, and raw bytes independently of serialization."""

    hasher = hashlib.sha256()
    for name in sorted(model_state):
        value = model_state[name]
        if not isinstance(value, Tensor):
            raise ExperimentError(f"model state contains a non-tensor entry: {name}")
        tensor = value.detach().cpu().contiguous()
        header = json.dumps(
            {
                "name": str(name),
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        hasher.update(len(header).to_bytes(8, "big"))
        hasher.update(header)
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
        hasher.update(len(raw).to_bytes(8, "big"))
        hasher.update(raw)
    return hasher.hexdigest()


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - compatibility with older supported torch
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, Mapping):
        raise ExperimentError("checkpoint payload is not a mapping")
    return value


def _fsync_directory(path: Path) -> None:
    """Persist directory-entry changes where the host filesystem supports it."""

    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:  # pragma: no cover - filesystem/platform dependent
        return
    try:
        with suppress(OSError):  # pragma: no branch - filesystem dependent
            os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _prune_resume_checkpoints(directory: Path, *, keep: Path | None) -> None:
    """Delete only superseded resumable checkpoints in one run directory.

    A step-named file is written before ``latest.json`` is replaced.  Therefore
    an interruption can leave either the previous file or the newly written
    file as an orphan.  The pointer remains authoritative and this bounded
    cleanup is safe on the next callback or resume.  The legacy ``step-*``
    pattern is accepted so runs produced by the first backend schema can be
    resumed and compacted without deleting weights-only snapshots.
    """

    if not directory.is_dir():
        return
    retained = keep.resolve() if keep is not None else None
    removed = False
    candidates = {
        *directory.glob("resume-step-????????.pt"),
        *directory.glob("step-????????.pt"),
    }
    for candidate in sorted(candidates):
        if retained is not None and candidate.resolve() == retained:
            continue
        candidate.unlink(missing_ok=True)
        removed = True
    if removed:
        _fsync_directory(directory)


def _write_weights_snapshot(
    directory: Path,
    *,
    step: int,
    binding: Mapping[str, Any],
    model_state_kind: str,
    model_state: Mapping[str, Any],
) -> tuple[Path, str]:
    """Atomically retain a selected model/adapter state without optimizer data."""

    target = directory / f"weights-step-{step:08d}.pt"
    model_state_digest = model_state_sha256(model_state)
    digest = _atomic_torch_save(
        target,
        {
            "schema_version": 1,
            "artifact_kind": "weights_only_snapshot",
            "step": int(step),
            "binding": dict(binding),
            "model_state_kind": model_state_kind,
            "model_state_sha256": model_state_digest,
            "model_state": model_state,
        },
    )
    index_path = directory / "index.json"
    entries: list[dict[str, Any]] = []
    if index_path.is_file():
        index = read_json(index_path)
        if index.get("binding") != binding:
            raise ExperimentError("weights snapshot index binding does not match this run")
        raw_entries = index.get("snapshots", [])
        if not isinstance(raw_entries, list):
            raise ExperimentError("weights snapshot index entries must be a list")
        entries = [dict(item) for item in raw_entries if isinstance(item, Mapping)]
        if len(entries) != len(raw_entries):
            raise ExperimentError("weights snapshot index contains a non-object entry")
    entries = [entry for entry in entries if int(entry.get("step", -1)) != int(step)]
    entries.append(
        {
            "step": int(step),
            "file": target.name,
            "sha256": digest,
            "model_state_sha256": model_state_digest,
        }
    )
    entries.sort(key=lambda entry: int(entry["step"]))
    write_json(
        index_path,
        {
            "schema_version": 1,
            "artifact_kind": "weights_only_snapshot_index",
            "binding": dict(binding),
            "snapshots": entries,
        },
    )
    return target, digest


def _retire_resume_checkpoint(
    directory: Path,
    *,
    step: int,
    binding: Mapping[str, Any],
    final_snapshot: Mapping[str, Any] | None,
) -> None:
    """Durably mark completed training, then remove optimizer-bearing state.

    The marker is written first, the authoritative pointer is removed second,
    and step files are pruned last.  Thus a crash either leaves a valid latest
    checkpoint or a durable completion marker.  A resumed backend can use the
    latter to finalize the run from already-durable metrics without retraining.
    """

    directory.mkdir(parents=True, exist_ok=True)
    write_json(
        directory / "retired.json",
        {
            "schema_version": 1,
            "artifact_kind": "retired_resumable_checkpoint",
            "step": int(step),
            "binding": dict(binding),
            "final_snapshot": dict(final_snapshot) if final_snapshot is not None else None,
        },
    )
    latest_path = directory / "latest.json"
    latest_path.unlink(missing_ok=True)
    _fsync_directory(directory)
    _prune_resume_checkpoints(directory, keep=None)


class _CachedReferenceScorer(nn.Module):
    def __init__(self, values: Mapping[str, Tensor]) -> None:
        super().__init__()
        self.values = {key: value.detach().cpu() for key, value in values.items()}

    def forward(self, prompts: Sequence[str]) -> Tensor:
        return torch.stack([self.values[prompt] for prompt in prompts])


def _cache_reference_scores(
    scorer: nn.Module,
    prompts: Sequence[str],
    batch_size: int,
) -> _CachedReferenceScorer:
    unique = tuple(dict.fromkeys(prompts))
    values: dict[str, Tensor] = {}
    was_training = scorer.training
    scorer.eval()
    with torch.no_grad():
        for batch in _chunks(unique, batch_size):
            scores = scorer(batch)
            for prompt, score in zip(batch, scores.detach().cpu(), strict=True):
                values[prompt] = score
    scorer.train(was_training)
    return _CachedReferenceScorer(values)


def _metric_record(payload: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    return {**body, "record_id": stable_digest(body, length=24)}


def _prediction_rows(
    result: EvaluationResult,
    *,
    step: int,
    split: str,
) -> list[dict[str, Any]]:
    return [
        _metric_record(
            {
                "kind": "prediction",
                "step": step,
                "split": split,
                **record.as_dict(),
            }
        )
        for record in result.records
    ]


def _behavior_rows(
    result: EvaluationResult,
    *,
    step: int,
    split: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in result.behavioral_agreement:
        rows.append(
            _metric_record(
                {
                    "kind": "behavioral_agreement",
                    "step": step,
                    "split": split,
                    "panel": value["panel"],
                    "prompt_view": value["prompt_view"],
                    "n": value["n"],
                    "rho_y": value["agreement_y"],
                    "rho_p": value["agreement_p"],
                    "rho_q": value["agreement_q"],
                    "action_b_rate": value["action_b_rate"],
                }
            )
        )
    for value in result.factorial_cells:
        rows.append(
            _metric_record(
                {
                    "kind": "factorial",
                    "step": step,
                    "split": split,
                    "panel": "conflict" if value["is_conflict"] else "agreement",
                    "prompt_view": value["prompt_view"],
                    "choice_y": value["choice_y"],
                    "choice_p": value["choice_p"],
                    "choice_q": value["choice_q"],
                    "n": value["n"],
                    "rho_y": value["agreement_y"],
                    "rho_p": value["agreement_p"],
                    "rho_q": value["agreement_q"],
                    "action_b_rate": value["action_b_rate"],
                    "mean_margin_b_minus_a": value["mean_margin_b_minus_a"],
                }
            )
        )
    return rows


def _intervention_rows(
    result: EvaluationResult,
    *,
    step: int,
    split: str,
) -> list[dict[str, Any]]:
    return [
        _metric_record(
            {
                "kind": "intervention",
                "step": step,
                "split": split,
                **value,
            }
        )
        for value in result.intervention_summary
    ]


def _wide_checkpoint_rows(
    factorial: EvaluationResult,
    causal: EvaluationResult,
    *,
    step: int,
    split: str,
) -> list[dict[str, Any]]:
    conflict = {
        value["prompt_view"]: value
        for value in factorial.behavioral_agreement
        if value["panel"] == "conflict"
    }
    interventions = {(value["prompt_view"], value["target"]): value for value in causal.intervention_summary}
    rows: list[dict[str, Any]] = []
    for view, behavior in sorted(conflict.items()):

        def causal_value(target: str, field: str, prompt_view: str = view) -> Any:
            return interventions.get((prompt_view, target), {}).get(field)

        rows.append(
            _metric_record(
                {
                    "kind": "checkpoint_summary",
                    "step": step,
                    "split": split,
                    "panel": "conflict",
                    "prompt_view": view,
                    "n": behavior["n"],
                    "rho_y": behavior["agreement_y"],
                    "rho_p": behavior["agreement_p"],
                    "rho_q": behavior["agreement_q"],
                    "causal_y": causal_value("law", "mean_target_aligned_delta_margin"),
                    "causal_p": causal_value("herald", "mean_target_aligned_delta_margin"),
                    "causal_q": causal_value("sage", "mean_target_aligned_delta_margin"),
                    "causal_d": causal_value("distractor", "mean_absolute_delta_margin"),
                    "causal_y_flip_rate": causal_value("law", "action_flip_rate"),
                    "causal_p_flip_rate": causal_value("herald", "action_flip_rate"),
                    "causal_q_flip_rate": causal_value("sage", "action_flip_rate"),
                    "causal_d_flip_rate": causal_value("distractor", "action_flip_rate"),
                }
            )
        )
    return rows


class GoalZendoExperiment:
    """Dependency-injectable experiment backend used by ``goalzendo.runner``."""

    def __init__(
        self,
        *,
        model_loader: Callable[..., tuple[nn.Module, Any]] | None = None,
        evaluation_observer: EvaluationObserver | None = None,
    ) -> None:
        self.model_loader = model_loader or load_model_and_tokenizer
        self.evaluation_observer = evaluation_observer

    @staticmethod
    def _device(requested: str) -> torch.device:
        normalized = requested.lower()
        if normalized == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        device = torch.device(normalized)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ExperimentError("CUDA was requested but is unavailable")
        if device.type == "mps":
            mps = getattr(torch.backends, "mps", None)
            if mps is None or not mps.is_available():
                raise ExperimentError("MPS was requested but is unavailable")
        return device

    def run(self, context: RunContext) -> BackendResult:
        config = context.config
        banks = materialize_banks(config, context.seeds)
        numerical_execution = configure_numerical_execution(config)
        device = self._device(str(get_path(config, "run.device", "auto")))
        model_config = dict(get_path(config, "model", {}))
        update_config = dict(get_path(config, "update", {}))
        torch.manual_seed(int(context.seeds["model_initialization"]))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(context.seeds["model_initialization"]))
        model, tokenizer = self.model_loader(model_config, update_config)
        model.to(device)
        if bool(get_path(config, "train.gradient_checkpointing", False)):
            enable = getattr(model, "gradient_checkpointing_enable", None)
            if not callable(enable):
                raise ExperimentError("gradient checkpointing requested but unsupported by model")
            enable()
            runtime_config = getattr(model, "config", None)
            if runtime_config is not None and hasattr(runtime_config, "use_cache"):
                runtime_config.use_cache = False

        action_labels = tuple(str(value) for value in model_config.get("action_labels", ("A", "B")))
        provenance = model_provenance(model, tokenizer, model_config, action_labels)
        model_metadata, tokenizer_metadata = _split_provenance(
            provenance,
            model,
            tokenizer,
            update_config,
        )
        model_metadata["numerical_execution"] = numerical_execution
        rendered = render_experiment(
            config,
            banks,
            tokenizer,
            int(context.seeds["rendering"]),
        )
        chat_template = bool(model_config.get("chat_template", True))
        token_metadata = tokenization_metadata(
            rendered,
            tokenizer,
            action_labels,
            add_special_tokens=not chat_template,
            max_sequence_length=int(get_path(config, "train.max_sequence_length", 384)),
        )
        tokenizer_metadata["experiment_tokenization"] = token_metadata
        dataset_metadata = {
            **dict(banks.metadata),
            "rendering": dict(rendered.metadata),
            "dataset_binding_digest": stable_hash(
                {"symbolic": banks.metadata, "rendering": rendered.metadata},
                64,
            ),
        }
        context.record_dataset_metadata(dataset_metadata)
        context.record_model_metadata(model_metadata)
        context.record_tokenizer_metadata(tokenizer_metadata)

        scorer = TwoActionScorer(
            model,
            tokenizer,
            action_labels,
            add_prompt_special_tokens=not chat_template,
        )
        optimizer = build_optimizer(
            scorer,
            learning_rate=float(get_path(config, "train.learning_rate")),
            weight_decay=float(get_path(config, "train.weight_decay", 0.0)),
        )
        algorithm = str(get_path(config, "train.algorithm", "sft"))
        if algorithm == "trajectory_sft":
            raise ExperimentError(
                "trajectory_sft requires a preregistered nuisance-trajectory renderer; "
                "the current backend implements clean action SFT, sampled outcome RL, "
                "enumerated expected-outcome RL, and registered power-gradient controls only"
            )
        kl_coefficient = float(get_path(config, "train.kl_coefficient", 0.0))
        reference_scorer: nn.Module | None = None
        if algorithm in {
            "outcome_rl",
            "expected_outcome_rl",
            "tempered_outcome_control",
            "logprob_outcome_control",
        } and kl_coefficient > 0:
            reference_scorer = _cache_reference_scores(
                scorer,
                rendered.train_prompts,
                int(get_path(config, "train.batch_size")),
            )

        metric_ids = {
            str(row.get("record_id")) for row in context.prior_metrics if row.get("record_id") is not None
        }
        prediction_ids = {
            str(row.get("record_id")) for row in context.prior_predictions if row.get("record_id") is not None
        }

        def append_metrics(rows: Sequence[Mapping[str, Any]]) -> None:
            novel = [row for row in rows if str(row["record_id"]) not in metric_ids]
            if novel:
                context.append_metrics(novel)
                metric_ids.update(str(row["record_id"]) for row in novel)

        def append_predictions(rows: Sequence[Mapping[str, Any]]) -> None:
            novel = [row for row in rows if str(row["record_id"]) not in prediction_ids]
            if novel:
                context.append_predictions(novel)
                prediction_ids.update(str(row["record_id"]) for row in novel)

        eval_batch_size = int(
            get_path(
                config,
                "evaluation.batch_size",
                min(32, int(get_path(config, "train.batch_size"))),
            )
        )
        if eval_batch_size < 1:
            raise ExperimentError("evaluation batch size must be positive")
        total_steps = int(get_path(config, "train.steps"))
        configured_eval_steps = tuple(int(value) for value in get_path(config, "train.eval_steps"))
        eval_steps = tuple(sorted(set((*configured_eval_steps, total_steps))))

        def evaluate_callback(
            step: int,
            _model: nn.Module,
            state: Mapping[str, Any],
        ) -> None:
            final = step == total_steps
            factorial_examples = (
                rendered.final_factorial_examples if final else rendered.diagnostic_factorial_examples
            )
            causal_examples = rendered.final_causal_examples if final else rendered.diagnostic_causal_examples
            factorial_split = "final_factorial" if final else "diagnostic_factorial"
            iid = evaluate_batches(
                _chunks(rendered.validation_examples, eval_batch_size),
                scorer,
            )
            factorial = evaluate_batches(
                _chunks(factorial_examples, eval_batch_size),
                scorer,
            )
            causal = evaluate_batches(
                _chunks(causal_examples, eval_batch_size),
                scorer,
            )
            rows = [
                *_behavior_rows(iid, step=step, split="iid_validation"),
                *_behavior_rows(factorial, step=step, split=factorial_split),
                *_intervention_rows(causal, step=step, split=f"{factorial_split}_causal"),
                *_wide_checkpoint_rows(
                    factorial,
                    causal,
                    step=step,
                    split=factorial_split,
                ),
            ]
            step_metrics = state.get("step_metrics")
            if isinstance(step_metrics, Mapping):
                rows.append(
                    _metric_record(
                        {
                            "kind": "optimization",
                            "algorithm": algorithm,
                            **dict(step_metrics),
                        }
                    )
                )
            append_metrics(rows)
            if bool(get_path(config, "evaluation.save_predictions", True)):
                append_predictions(
                    [
                        *_prediction_rows(iid, step=step, split="iid_validation"),
                        *_prediction_rows(factorial, step=step, split=factorial_split),
                        *_prediction_rows(
                            causal,
                            step=step,
                            split=f"{factorial_split}_causal",
                        ),
                    ]
                )
            context.progress(step, phase="evaluated", evaluation_bank=factorial_split)
            if self.evaluation_observer is not None:
                self.evaluation_observer(step)

        hooks: ScheduledHooks
        binding = {
            "run_id": context.store.run_id,
            "dataset": dataset_metadata["dataset_binding_digest"],
            "model": stable_hash(model_metadata, 64),
            "tokenizer": stable_hash(tokenizer_metadata, 64),
        }
        checkpoint_directory = context.store.path / "checkpoints"
        snapshot_directory = context.store.path / "snapshots"
        snapshot_steps = frozenset(
            int(value) for value in get_path(config, "run.snapshot_steps", [])
        )
        resumable_checkpoint_steps = frozenset(
            int(value) for value in get_path(config, "run.checkpoint_steps", [])
        )

        def cuda_memory_summary() -> dict[str, int | None]:
            if device.type != "cuda":
                return {
                    "cuda_peak_allocated_bytes": None,
                    "cuda_peak_reserved_bytes": None,
                }
            return {
                "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }

        def completed_backend_result() -> BackendResult:
            final_wide = [
                row
                for row in context.prior_metrics
                if row.get("kind") == "checkpoint_summary" and row.get("step") == total_steps
            ]
            if not final_wide:
                raise ExperimentError(
                    "completed checkpoint retirement requires durable final evaluation metrics"
                )
            return BackendResult(
                summary={
                    "backend_version": EXPERIMENT_BACKEND_VERSION,
                    "final_step": total_steps,
                    "algorithm": algorithm,
                    "device": str(device),
                    **cuda_memory_summary(),
                    "numerical_execution": numerical_execution,
                    "dataset_binding_digest": dataset_metadata["dataset_binding_digest"],
                    "final_checkpoint_summaries": final_wide,
                }
            )

        def checkpoint_callback(
            step: int,
            _model: nn.Module,
            state: Mapping[str, Any],
        ) -> str:
            # The scheduled training module is the action scorer wrapper;
            # durable state belongs to the underlying causal language model.
            del _model
            model_kind, model_state = _adapter_state(model)
            cpu_model_state = _cpu_tree(model_state)
            hook_state = hooks.state_dict()
            emitted: list[list[str | int]] = [list(item) for item in hook_state["emitted"]]
            marker: list[str | int] = ["checkpoint", step]
            if marker not in emitted:
                emitted.append(marker)
            hook_state = {"emitted": emitted}

            # A retained snapshot is made authoritative before the resume
            # pointer advances.  If the process stops between these writes,
            # resuming from the old pointer deterministically repeats this step
            # and atomically replaces the same weights-only snapshot.
            snapshot_target: Path | None = None
            if step in snapshot_steps:
                snapshot_target, _ = _write_weights_snapshot(
                    snapshot_directory,
                    step=step,
                    binding=binding,
                    model_state_kind=model_kind,
                    model_state=cpu_model_state,
                )

            if step not in resumable_checkpoint_steps:
                assert snapshot_target is not None
                context.progress(
                    step,
                    phase="snapshot_retained",
                    retained_snapshot=snapshot_target.name,
                )
                return snapshot_target.name

            target = checkpoint_directory / f"resume-step-{step:08d}.pt"
            digest = _atomic_torch_save(
                target,
                {
                    "schema_version": 2,
                    "artifact_kind": "resumable_checkpoint",
                    "step": step,
                    "binding": binding,
                    "model_state_kind": model_kind,
                    "model_state": cpu_model_state,
                    "optimizer_state": _cpu_tree(optimizer.state_dict()),
                    "train_state": dict(state["train_state"]),
                    "hook_state": hook_state,
                },
            )
            write_json(
                checkpoint_directory / "latest.json",
                {
                    "schema_version": 2,
                    "artifact_kind": "resumable_checkpoint_pointer",
                    "step": step,
                    "file": target.name,
                    "sha256": digest,
                    "binding": binding,
                },
            )
            # Only after the durable pointer names the new checkpoint is it
            # safe to remove both the previous checkpoint and any orphan left
            # by an earlier interrupted write.
            _prune_resume_checkpoints(checkpoint_directory, keep=target)
            context.progress(
                step,
                phase="checkpointed",
                checkpoint=target.name,
                retained_snapshot=snapshot_target.name if snapshot_target is not None else None,
            )
            return target.name

        save_checkpoints = bool(get_path(config, "run.save_checkpoints", False))
        durable_hook_steps = tuple(sorted(resumable_checkpoint_steps | snapshot_steps))
        hooks = ScheduledHooks(
            eval_steps=eval_steps,
            checkpoint_steps=durable_hook_steps if save_checkpoints else (),
            evaluate=evaluate_callback,
            checkpoint=checkpoint_callback if save_checkpoints else None,
        )

        train_state = TrainState()
        latest_path = checkpoint_directory / "latest.json"
        retired_path = checkpoint_directory / "retired.json"
        if context.resumed and not latest_path.is_file() and retired_path.is_file():
            retired = read_json(retired_path)
            if retired.get("binding") != binding or int(retired.get("step", -1)) != total_steps:
                raise ExperimentError("retired checkpoint marker does not match this run")
            final_snapshot = retired.get("final_snapshot")
            if final_snapshot is not None:
                if not isinstance(final_snapshot, Mapping):
                    raise ExperimentError("retired checkpoint snapshot entry must be an object")
                snapshot_name = str(final_snapshot.get("file", ""))
                if Path(snapshot_name).name != snapshot_name or not snapshot_name:
                    raise ExperimentError("retired checkpoint snapshot path is invalid")
                snapshot_path = snapshot_directory / snapshot_name
                if not snapshot_path.is_file():
                    raise ExperimentError("retained final weights snapshot is missing")
                if _file_sha256(snapshot_path) != final_snapshot.get("sha256"):
                    raise ExperimentError("retained final weights snapshot digest mismatch")
            _prune_resume_checkpoints(checkpoint_directory, keep=None)
            context.progress(total_steps, phase="checkpoint_retirement_recovered")
            return completed_backend_result()
        if context.resumed and latest_path.is_file():
            latest = read_json(latest_path)
            if latest.get("binding") != binding:
                raise ExperimentError("latest checkpoint binding does not match this run")
            checkpoint_path = checkpoint_directory / str(latest["file"])
            if not checkpoint_path.is_file():
                raise ExperimentError("latest checkpoint file is missing")
            if _file_sha256(checkpoint_path) != latest.get("sha256"):
                raise ExperimentError("latest checkpoint digest mismatch")
            checkpoint = _load_checkpoint(checkpoint_path)
            if checkpoint.get("binding") != binding:
                raise ExperimentError("checkpoint payload binding mismatch")
            _restore_adapter(
                model,
                str(checkpoint["model_state_kind"]),
                checkpoint["model_state"],
            )
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            train_state = TrainState.from_state_dict(checkpoint["train_state"])
            hooks.load_state_dict(checkpoint["hook_state"])
            _prune_resume_checkpoints(checkpoint_directory, keep=checkpoint_path)
        elif context.resumed:
            # A crash before the first atomic pointer replacement can leave a
            # complete but unauthoritative step file.  Restart from step zero
            # rather than guessing whether its evaluation side effects landed.
            _prune_resume_checkpoints(checkpoint_directory, keep=None)

        warmup_steps = round(total_steps * float(get_path(config, "train.warmup_ratio", 0.0)))
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        result = train_steps(
            scorer,
            optimizer,
            rendered.train_prompts,
            rendered.train_actions,
            total_steps=total_steps,
            batch_size=int(get_path(config, "train.batch_size")),
            algorithm=algorithm,  # type: ignore[arg-type]
            gradient_accumulation_steps=int(get_path(config, "train.gradient_accumulation_steps", 1)),
            warmup_steps=warmup_steps,
            max_grad_norm=float(get_path(config, "train.grad_clip", 1.0)),
            parameter_finite_check_interval=int(
                get_path(config, "train.parameter_finite_check_interval", 0)
            ),
            samples_per_prompt=int(get_path(config, "train.samples_per_prompt", 4)),
            entropy_coefficient=float(get_path(config, "train.entropy_coefficient", 0.0)),
            kl_coefficient=kl_coefficient,
            reference_scorer=reference_scorer,
            reward_gradient_exponent=get_path(
                config,
                "train.reward_gradient_exponent",
                None,
            ),
            candidate_choices=rendered.train_candidate_choices,
            seed=int(context.seeds["training"]),
            action_sampling_seed=int(context.seeds["action_sampling"]),
            state=train_state,
            hooks=hooks,
        )
        append_metrics(
            [
                _metric_record(
                    {
                        "kind": "optimization",
                        "algorithm": algorithm,
                        **metric.as_dict(),
                    }
                )
                for metric in result.metrics
            ]
        )

        final_snapshot_entry: Mapping[str, Any] | None = None
        if total_steps in snapshot_steps:
            snapshot_index = read_json(snapshot_directory / "index.json")
            if snapshot_index.get("binding") != binding:
                raise ExperimentError("final weights snapshot index binding mismatch")
            matches = [
                entry
                for entry in snapshot_index.get("snapshots", [])
                if isinstance(entry, Mapping) and int(entry.get("step", -1)) == total_steps
            ]
            if len(matches) != 1:
                raise ExperimentError("configured final weights snapshot is not uniquely indexed")
            final_snapshot_entry = matches[0]

        completed = completed_backend_result()
        if save_checkpoints:
            _retire_resume_checkpoint(
                checkpoint_directory,
                step=result.state.global_step,
                binding=binding,
                final_snapshot=final_snapshot_entry,
            )
            context.progress(
                total_steps,
                phase="checkpoint_retired",
                final_snapshot=(
                    str(final_snapshot_entry.get("file"))
                    if final_snapshot_entry is not None
                    else None
                ),
            )
        return completed


def run_experiment(context: RunContext) -> BackendResult:
    """Default module-level backend loaded by :mod:`goalzendo.runner`."""

    return GoalZendoExperiment().run(context)


__all__ = [
    "EXPERIMENT_BACKEND_VERSION",
    "ExperimentBanks",
    "ExperimentError",
    "GoalZendoExperiment",
    "RenderedExperiment",
    "build_rule_specs",
    "materialize_banks",
    "model_state_sha256",
    "render_experiment",
    "render_prompt_view",
    "run_experiment",
    "tokenization_metadata",
]
