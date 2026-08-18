"""Fail-closed QA for the prospectively frozen schema-v5 candidate."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, cast

from ._json import json_digest
from .capability_qa_v3 import (
    CAPABILITY_V3_MASK_POLICY,
    CAPABILITY_V3_SURFACE_TARGETS,
    SemanticGateResultV3,
    _fold_digest,
    _semantic_vector,
)
from .capability_v5 import (
    OBSERVATION_TARGETS_V5,
    EpisodeBankV5,
    MaterializedCandidateV5,
    ProspectiveDesignV5,
    build_prospective_design_v5,
    materialize_prospective_design_v5,
    registered_outcome_folds_v5,
)
from .leakage import (
    SurfaceLeakageAuditReport,
    SurfaceLeakageConfig,
    SurfaceLeakageSample,
    SurfaceLeakageTarget,
    audit_surface_targets,
)
from .rendering import EVAL_RENDERERS, TRAIN_RENDERERS
from .schema import NONEMPTY_ARRANGEMENT_COUNT

CAPABILITY_QA_SCHEMA_VERSION_V5 = 5


def _bool_label(value: bool) -> str:
    return "true" if value else "false"


def hidden_episode_bank_v5_surface_targets(
    bank: EpisodeBankV5,
) -> tuple[SurfaceLeakageTarget, ...]:
    target_label: list[SurfaceLeakageSample] = []
    formula: list[SurfaceLeakageSample] = []
    proxy: list[SurfaceLeakageSample] = []
    placard_error: list[SurfaceLeakageSample] = []
    shadow_error: list[SurfaceLeakageSample] = []
    for episode_index, episode in enumerate(bank.episodes):
        group_id = f"episode-{episode_index}"
        episode_features = (
            f"renderer={episode.request.renderer}",
            f"opening_count={len(episode.opening)}",
            f"terminal_count={len(episode.terminal)}",
        )
        if episode.request.target_family == "binary_piece":
            formula.append(
                SurfaceLeakageSample(
                    f"{group_id}-formula",
                    group_id,
                    episode_features,
                    cast(str, episode.request.formula_label),
                )
            )
        proxy.append(
            SurfaceLeakageSample(
                f"{group_id}-proxy-orientation",
                group_id,
                episode_features,
                episode.request.proxy_orientation,
            )
        )
        for phase_name, observations in (
            ("opening", episode.opening),
            ("terminal", episode.terminal),
        ):
            for position, observation in enumerate(observations):
                sample_id = f"{group_id}-{phase_name}-{position}"
                features = (
                    *episode_features,
                    f"phase={phase_name}",
                    f"phase_count={len(observations)}",
                    f"position={position}",
                )
                p = observation.scene_index < NONEMPTY_ARRANGEMENT_COUNT
                y = observation.accepted
                q = episode.shadow.truth[observation.scene_index]
                target_label.append(
                    SurfaceLeakageSample(
                        sample_id, group_id, features, _bool_label(y)
                    )
                )
                placard_error.append(
                    SurfaceLeakageSample(
                        sample_id, group_id, features, _bool_label(y is not p)
                    )
                )
                shadow_error.append(
                    SurfaceLeakageSample(
                        sample_id, group_id, features, _bool_label(y is not q)
                    )
                )
    binary = ("false", "true")
    targets = (
        SurfaceLeakageTarget("target_label", binary, tuple(target_label)),
        SurfaceLeakageTarget(
            "target_formula_stratum",
            ("all_or_any", "exactly_one"),
            tuple(formula),
        ),
        SurfaceLeakageTarget(
            "proxy_orientation",
            ("proxy_low", "proxy_high"),
            tuple(proxy),
        ),
        SurfaceLeakageTarget("placard_error_status", binary, tuple(placard_error)),
        SurfaceLeakageTarget("shadow_error_status", binary, tuple(shadow_error)),
    )
    if tuple(target.name for target in targets) != CAPABILITY_V3_SURFACE_TARGETS:
        raise RuntimeError("v5 surface target registry changed")
    return targets


def audit_hidden_episode_bank_v5_surface_leakage(
    bank: EpisodeBankV5,
    *,
    config: SurfaceLeakageConfig,
) -> SurfaceLeakageAuditReport:
    return audit_surface_targets(
        bank.spec.bank_id,
        hidden_episode_bank_v5_surface_targets(bank),
        config=config,
    )


def _registered_fold_digests_v5(bank: EpisodeBankV5) -> dict[str, str]:
    formula = {
        f"episode-{index}": cast(int, episode.request.registered_formula_fold)
        for index, episode in enumerate(bank.episodes)
        if episode.request.target_family == "binary_piece"
    }
    proxy = {
        f"episode-{index}": episode.request.registered_proxy_fold
        for index, episode in enumerate(bank.episodes)
    }
    result = {
        "target_formula_stratum": _fold_digest(formula),
        "proxy_orientation": _fold_digest(proxy),
    }
    for target_name in OBSERVATION_TARGETS_V5:
        folds = registered_outcome_folds_v5(bank.spec.requests, target_name)
        result[target_name] = _fold_digest(
            {f"episode-{index}": fold for index, fold in enumerate(folds)}
        )
    return result


def _actual_balance_v5(bank: EpisodeBankV5) -> tuple[str, str, bool]:
    renderers = (
        TRAIN_RENDERERS
        if bank.episodes[0].request.stage == "format_warm_start"
        else EVAL_RENDERERS
    )
    renderer_rows: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []
    passed = True
    for target_index, target_name in enumerate(OBSERVATION_TARGETS_V5):
        folds = registered_outcome_folds_v5(bank.spec.requests, target_name)
        for fold in range(5):
            indices = tuple(
                index for index, assigned in enumerate(folds) if assigned == fold
            )
            for renderer in renderers:
                false_count = 0
                true_count = 0
                for index in indices:
                    episode = bank.episodes[index]
                    if episode.request.renderer != renderer:
                        continue
                    for phase in (episode.opening, episode.terminal):
                        for observation in phase:
                            y = observation.accepted
                            p = observation.scene_index < NONEMPTY_ARRANGEMENT_COUNT
                            q = episode.shadow.truth[observation.scene_index]
                            outcome = (y, y is not p, y is not q)[target_index]
                            true_count += int(outcome)
                            false_count += int(not outcome)
                passed = passed and false_count == true_count
                renderer_rows.append(
                    {
                        "target": target_name,
                        "fold": fold,
                        "renderer": renderer,
                        "false": false_count,
                        "true": true_count,
                    }
                )
            for phase_name in ("opening", "terminal"):
                for position in range(10):
                    true_count = 0
                    for index in indices:
                        episode = bank.episodes[index]
                        observation = getattr(episode, phase_name)[position]
                        y = observation.accepted
                        p = observation.scene_index < NONEMPTY_ARRANGEMENT_COUNT
                        q = episode.shadow.truth[observation.scene_index]
                        true_count += int((y, y is not p, y is not q)[target_index])
                    false_count = len(indices) - true_count
                    passed = passed and false_count == true_count
                    position_rows.append(
                        {
                            "target": target_name,
                            "fold": fold,
                            "phase": phase_name,
                            "position": position,
                            "false": false_count,
                            "true": true_count,
                        }
                    )
    return (
        json_digest(renderer_rows, domain="goalzendo-interactive-actual-renderer-balance-v5"),
        json_digest(position_rows, domain="goalzendo-interactive-actual-position-balance-v5"),
        passed,
    )


@dataclass(frozen=True, slots=True)
class BankMetricsV5:
    bank_id: str
    spec_digest: str
    bank_digest: str
    episode_count: int
    target_family_counts: tuple[tuple[str, int], ...]
    target_operator_counts: tuple[tuple[str, int], ...]
    renderer_counts: tuple[tuple[str, int], ...]
    proxy_orientation_counts: tuple[tuple[str, int], ...]
    target_identity_count: int
    pair_rank_counts: tuple[tuple[int, int], ...]
    registered_fold_digests: tuple[tuple[str, str], ...]
    actual_renderer_balance_digest: str
    actual_position_balance_digest: str
    registered_folds_match_audit: bool
    exact_actual_balance: bool
    exact_phase_profiles: bool

    @property
    def passed(self) -> bool:
        return (
            self.target_identity_count == self.episode_count
            and self.registered_folds_match_audit
            and self.exact_actual_balance
            and self.exact_phase_profiles
        )

    def as_obj(self) -> dict[str, Any]:
        return {
            "bank_id": self.bank_id,
            "spec_digest": self.spec_digest,
            "bank_digest": self.bank_digest,
            "episode_count": self.episode_count,
            "target_family_counts": dict(self.target_family_counts),
            "target_operator_counts": dict(self.target_operator_counts),
            "renderer_counts": dict(self.renderer_counts),
            "proxy_orientation_counts": dict(self.proxy_orientation_counts),
            "target_identity_count": self.target_identity_count,
            "pair_rank_counts": dict(self.pair_rank_counts),
            "registered_fold_digests": dict(self.registered_fold_digests),
            "actual_renderer_balance_digest": self.actual_renderer_balance_digest,
            "actual_position_balance_digest": self.actual_position_balance_digest,
            "invariants": {
                "registered_folds_match_audit": self.registered_folds_match_audit,
                "exact_actual_balance": self.exact_actual_balance,
                "exact_phase_profiles": self.exact_phase_profiles,
                "passed": self.passed,
            },
        }

    @property
    def digest(self) -> str:
        return json_digest(self.as_obj(), domain="goalzendo-interactive-bank-metrics-v5")


def bank_metrics_v5(
    bank: EpisodeBankV5,
    leakage: SurfaceLeakageAuditReport,
) -> BankMetricsV5:
    family_counts = Counter(episode.request.target_family for episode in bank.episodes)
    op_counts = Counter(episode.request.target_op or "control" for episode in bank.episodes)
    renderer_counts = Counter(episode.request.renderer for episode in bank.episodes)
    orientation_counts = Counter(
        episode.request.proxy_orientation for episode in bank.episodes
    )
    pair_ranks = Counter(record.pair_rank for record in bank.generation_records)
    registered = _registered_fold_digests_v5(bank)
    reported = {result.target_name: result.fold_digest for result in leakage.results}
    folds_match = all(reported[name] == digest for name, digest in registered.items())
    renderer_digest, position_digest, balance_passed = _actual_balance_v5(bank)
    phase_profiles = all(
        len(phase) == 10 and sum(item.accepted for item in phase) == 5
        for episode in bank.episodes
        for phase in (episode.opening, episode.terminal)
    )
    return BankMetricsV5(
        bank.spec.bank_id,
        bank.spec.digest,
        bank.digest,
        len(bank.episodes),
        tuple(sorted(family_counts.items())),
        tuple(sorted(op_counts.items())),
        tuple(sorted(renderer_counts.items())),
        tuple(sorted(orientation_counts.items())),
        len({episode.target.truth_digest for episode in bank.episodes}),
        tuple(sorted(pair_ranks.items())),
        tuple(sorted(registered.items())),
        renderer_digest,
        position_digest,
        folds_match,
        balance_passed,
        phase_profiles,
    )


def _semantic_gate(candidate: MaterializedCandidateV5) -> SemanticGateResultV3:
    warm = _semantic_vector(cast(Any, candidate.warm))
    capability = _semantic_vector(cast(Any, candidate.capability))
    differences = tuple(abs(left - right) for left, right in zip(warm, capability, strict=True))
    thresholds = candidate.design.semantic_gate.upper_thresholds
    return SemanticGateResultV3(
        candidate.design.semantic_gate.digest,
        warm,
        capability,
        differences,
        thresholds,
        tuple(
            value <= threshold
            for value, threshold in zip(differences, thresholds, strict=True)
        ),
    )


@dataclass(frozen=True, slots=True)
class CandidateAuditV5:
    design_digest: str
    pair_population_audit_digest: str
    outcome_fold_population_digest: str
    warm: BankMetricsV5
    capability: BankMetricsV5
    warm_surface_leakage: SurfaceLeakageAuditReport
    capability_surface_leakage: SurfaceLeakageAuditReport
    semantic_gate: SemanticGateResultV3
    stage_target_identities_disjoint: bool
    blocker_codes: tuple[str, ...]

    @property
    def structural_invariants_passed(self) -> bool:
        return self.warm.passed and self.capability.passed and self.stage_target_identities_disjoint

    @property
    def surface_leakage_passed(self) -> bool:
        return self.warm_surface_leakage.passed and self.capability_surface_leakage.passed

    @property
    def production_bank_generation_authorized(self) -> bool:
        return False

    @property
    def weight_updates_authorized(self) -> bool:
        return False

    def as_obj(self) -> dict[str, Any]:
        return {
            "schema_version": CAPABILITY_QA_SCHEMA_VERSION_V5,
            "scope": "single-prospective-local-materialization-non-authorizing",
            "design_digest": self.design_digest,
            "pair_population_audit_digest": self.pair_population_audit_digest,
            "outcome_fold_population_digest": self.outcome_fold_population_digest,
            "surface_mask_policy": CAPABILITY_V3_MASK_POLICY,
            "warm": self.warm.as_obj(),
            "capability": self.capability.as_obj(),
            "warm_surface_leakage": self.warm_surface_leakage.as_obj(),
            "capability_surface_leakage": self.capability_surface_leakage.as_obj(),
            "semantic_gate": self.semantic_gate.as_obj(),
            "stage_target_identities_disjoint": self.stage_target_identities_disjoint,
            "structural_invariants_passed": self.structural_invariants_passed,
            "surface_leakage_passed": self.surface_leakage_passed,
            "blocker_codes": list(self.blocker_codes),
            "production_bank_generation_authorized": False,
            "weight_updates_authorized": False,
        }

    @property
    def digest(self) -> str:
        return json_digest(self.as_obj(), domain="goalzendo-interactive-candidate-audit-v5")


@lru_cache(maxsize=2)
def build_candidate_audit_v5(
    config: SurfaceLeakageConfig | None = None,
    design: ProspectiveDesignV5 | None = None,
) -> CandidateAuditV5:
    if config is None:
        config = SurfaceLeakageConfig()
    if design is None:
        design = build_prospective_design_v5()
    candidate = materialize_prospective_design_v5(design)
    warm_leakage = audit_hidden_episode_bank_v5_surface_leakage(
        candidate.warm, config=config
    )
    capability_leakage = audit_hidden_episode_bank_v5_surface_leakage(
        candidate.capability, config=config
    )
    warm_metrics = bank_metrics_v5(candidate.warm, warm_leakage)
    capability_metrics = bank_metrics_v5(candidate.capability, capability_leakage)
    semantic = _semantic_gate(candidate)
    warm_targets = {episode.target.truth_digest for episode in candidate.warm.episodes}
    capability_targets = {
        episode.target.truth_digest for episode in candidate.capability.episodes
    }
    disjoint = not bool(warm_targets & capability_targets)
    blockers: list[str] = []
    if not warm_leakage.passed or not capability_leakage.passed:
        blockers.append("surface_leakage_gate_failed")
    if not semantic.passed:
        blockers.append("prospective_semantic_gate_failed")
    if not warm_metrics.passed or not capability_metrics.passed or not disjoint:
        blockers.append("structural_or_counterbalance_invariant_failed")
    blockers.extend(
        (
            "model_pipeline_integration_not_audited",
            "production_bank_generation_not_authorized",
            "weight_updates_not_authorized",
        )
    )
    return CandidateAuditV5(
        design.digest,
        design.pair_population_audit.digest,
        design.outcome_fold_population.digest,
        warm_metrics,
        capability_metrics,
        warm_leakage,
        capability_leakage,
        semantic,
        disjoint,
        tuple(blockers),
    )
