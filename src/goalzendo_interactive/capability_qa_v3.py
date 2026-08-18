"""Fail-closed QA for the prospectively frozen schema-v3 capability design."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, cast

from ._json import json_digest
from .capability_v3 import (
    SEMANTIC_METRICS_V3,
    V3_BINARY_EPISODES_PER_STAGE,
    V3_FOLD_COUNT,
    EpisodeBankV3,
    HiddenEpisodeV3,
    MaterializedCandidateV3,
    ProspectiveDesignV3,
    build_prospective_design_v3,
    materialize_prospective_design_v3,
    registered_observation_folds_v3,
)
from .leakage import (
    EPISODE_SURFACE_FEATURE_KEYS,
    SurfaceLeakageAuditReport,
    SurfaceLeakageConfig,
    SurfaceLeakageSample,
    SurfaceLeakageTarget,
    audit_surface_targets,
)
from .rendering import EVAL_RENDERERS, TRAIN_RENDERERS
from .rules import ATOM_OPS, BinaryRule
from .schema import NONEMPTY_ARRANGEMENT_COUNT
from .stage_partitions_v2 import target_shadow_cells_v2

CAPABILITY_QA_SCHEMA_VERSION_V3 = 3
CAPABILITY_V3_SURFACE_TARGETS = (
    "target_label",
    "target_formula_stratum",
    "proxy_orientation",
    "placard_error_status",
    "shadow_error_status",
)
CAPABILITY_V3_SURFACE_FEATURE_KEYS = EPISODE_SURFACE_FEATURE_KEYS
CAPABILITY_V3_MASK_POLICY = (
    "allowlist-only: renderer and structural count/phase/position tokens; "
    "scene, target-label, Official-Law, placard, shadow, proxy, fold, identifier, "
    "digest, request-stratum, and generation-provenance content are omitted"
)


def _bool_label(value: bool) -> str:
    return "true" if value else "false"


def _episode_features(episode: HiddenEpisodeV3) -> tuple[str, ...]:
    return (
        f"renderer={episode.request.renderer}",
        f"opening_count={len(episode.opening)}",
        f"terminal_count={len(episode.terminal)}",
    )


def _observation_features(
    episode_features: tuple[str, ...],
    phase: str,
    phase_count: int,
    position: int,
) -> tuple[str, ...]:
    return (
        *episode_features,
        f"phase={phase}",
        f"phase_count={phase_count}",
        f"position={position}",
    )


def hidden_episode_bank_v3_surface_targets(
    bank: EpisodeBankV3,
) -> tuple[SurfaceLeakageTarget, ...]:
    """Build every leakage target from the frozen surface allowlist."""

    if type(bank) is not EpisodeBankV3:
        raise TypeError("hidden_episode_bank_v3_surface_targets requires EpisodeBankV3")
    target_label: list[SurfaceLeakageSample] = []
    formula: list[SurfaceLeakageSample] = []
    proxy: list[SurfaceLeakageSample] = []
    placard_error: list[SurfaceLeakageSample] = []
    shadow_error: list[SurfaceLeakageSample] = []
    for episode_index, episode in enumerate(bank.episodes):
        group_id = f"episode-{episode_index}"
        episode_features = _episode_features(episode)
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
                features = _observation_features(
                    episode_features,
                    phase_name,
                    len(observations),
                    position,
                )
                is_sun = observation.scene_index < NONEMPTY_ARRANGEMENT_COUNT
                target_label.append(
                    SurfaceLeakageSample(
                        sample_id,
                        group_id,
                        features,
                        _bool_label(observation.accepted),
                    )
                )
                placard_error.append(
                    SurfaceLeakageSample(
                        sample_id,
                        group_id,
                        features,
                        _bool_label(is_sun is not observation.accepted),
                    )
                )
                shadow_error.append(
                    SurfaceLeakageSample(
                        sample_id,
                        group_id,
                        features,
                        _bool_label(
                            episode.shadow.truth[observation.scene_index]
                            is not observation.accepted
                        ),
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
        raise RuntimeError("v3 surface target registry changed")
    return targets


def audit_hidden_episode_bank_v3_surface_leakage(
    bank: EpisodeBankV3,
    *,
    config: SurfaceLeakageConfig | None = None,
) -> SurfaceLeakageAuditReport:
    return audit_surface_targets(
        bank.spec.bank_id,
        hidden_episode_bank_v3_surface_targets(bank),
        config=config,
    )


def _fold_digest(assignments: dict[str, int]) -> str:
    digest = hashlib.sha256(b"goalzendo-surface-leakage-folds-v1\0")
    for group_id in sorted(assignments):
        digest.update(f"{group_id}:{assignments[group_id]}".encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _request_fold_digests(bank: EpisodeBankV3) -> tuple[str, str]:
    formula = {
        f"episode-{index}": cast(int, episode.request.registered_formula_fold)
        for index, episode in enumerate(bank.episodes)
        if episode.request.target_family == "binary_piece"
    }
    proxy = {
        f"episode-{index}": episode.request.registered_proxy_fold
        for index, episode in enumerate(bank.episodes)
    }
    return _fold_digest(formula), _fold_digest(proxy)


def _position_balance_digest(bank: EpisodeBankV3) -> tuple[str, bool]:
    rows: list[dict[str, Any]] = []
    target_names = (
        "target_label",
        "placard_error_status",
        "shadow_error_status",
    )
    folds_by_target = {
        name: registered_observation_folds_v3(len(bank.episodes), name)
        for name in target_names
    }
    passed = True
    for target_index, target_name in enumerate(target_names):
        folds = folds_by_target[target_name]
        for fold in range(V3_FOLD_COUNT):
            episode_indices = tuple(
                index for index, assigned in enumerate(folds) if assigned == fold
            )
            for phase_name in ("opening", "terminal"):
                for position in range(10):
                    true_count = 0
                    for episode_index in episode_indices:
                        episode = bank.episodes[episode_index]
                        observation = getattr(episode, phase_name)[position]
                        y = observation.accepted
                        p = observation.scene_index < NONEMPTY_ARRANGEMENT_COUNT
                        q = episode.shadow.truth[observation.scene_index]
                        outcomes = (y, y is not p, y is not q)
                        true_count += int(outcomes[target_index])
                    false_count = len(episode_indices) - true_count
                    passed = passed and true_count == false_count
                    rows.append(
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
        json_digest(rows, domain="goalzendo-interactive-position-balance-v3"),
        passed,
    )


def _renderer_fold_balance_digest(bank: EpisodeBankV3) -> tuple[str, bool]:
    renderers = (
        TRAIN_RENDERERS
        if bank.episodes[0].request.stage == "format_warm_start"
        else EVAL_RENDERERS
    )
    rows: list[dict[str, Any]] = []
    passed = True
    for fold in range(V3_FOLD_COUNT):
        for target, values in (
            ("target_formula_stratum", ("all_or_any", "exactly_one")),
            ("proxy_orientation", ("proxy_low", "proxy_high")),
        ):
            for value in values:
                counts: Counter[str] = Counter()
                for episode in bank.episodes:
                    request = episode.request
                    included = (
                        request.registered_formula_fold == fold
                        and request.formula_label == value
                        if target == "target_formula_stratum"
                        else request.registered_proxy_fold == fold
                        and request.proxy_orientation == value
                    )
                    if included:
                        counts[request.renderer] += 1
                vector = tuple(counts[renderer] for renderer in renderers)
                passed = passed and bool(vector) and len(set(vector)) == 1 and vector[0] > 0
                rows.append(
                    {
                        "target": target,
                        "fold": fold,
                        "class": value,
                        "renderer_counts": {
                            renderer: counts[renderer] for renderer in renderers
                        },
                    }
                )
    return (
        json_digest(rows, domain="goalzendo-interactive-fold-renderer-balance-v3"),
        passed,
    )


@dataclass(frozen=True, slots=True)
class BankMetricsV3:
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
    formula_fold_digest: str
    proxy_fold_digest: str
    renderer_fold_balance_digest: str
    position_balance_digest: str
    registered_folds_match_audit: bool
    exact_renderer_fold_balance: bool
    exact_position_balance: bool
    exact_phase_profiles: bool

    @property
    def passed(self) -> bool:
        return (
            self.target_identity_count == self.episode_count
            and self.registered_folds_match_audit
            and self.exact_renderer_fold_balance
            and self.exact_position_balance
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
            "formula_fold_digest": self.formula_fold_digest,
            "proxy_fold_digest": self.proxy_fold_digest,
            "renderer_fold_balance_digest": self.renderer_fold_balance_digest,
            "position_balance_digest": self.position_balance_digest,
            "invariants": {
                "registered_folds_match_audit": self.registered_folds_match_audit,
                "exact_renderer_fold_balance": self.exact_renderer_fold_balance,
                "exact_position_balance": self.exact_position_balance,
                "exact_phase_profiles": self.exact_phase_profiles,
                "passed": self.passed,
            },
        }

    @property
    def digest(self) -> str:
        return json_digest(self.as_obj(), domain="goalzendo-interactive-bank-metrics-v3")


def bank_metrics_v3(
    bank: EpisodeBankV3,
    leakage: SurfaceLeakageAuditReport,
) -> BankMetricsV3:
    family_counts = Counter(episode.request.target_family for episode in bank.episodes)
    op_counts = Counter(episode.request.target_op or "control" for episode in bank.episodes)
    renderer_counts = Counter(episode.request.renderer for episode in bank.episodes)
    orientation_counts = Counter(
        episode.request.proxy_orientation for episode in bank.episodes
    )
    pair_ranks = Counter(record.pair_rank for record in bank.generation_records)
    formula_fold, proxy_fold = _request_fold_digests(bank)
    result_by_name = {result.target_name: result for result in leakage.results}
    folds_match = (
        result_by_name["target_formula_stratum"].fold_digest == formula_fold
        and result_by_name["proxy_orientation"].fold_digest == proxy_fold
    )
    renderer_digest, renderer_passed = _renderer_fold_balance_digest(bank)
    position_digest, position_passed = _position_balance_digest(bank)
    phase_profiles = all(
        len(phase) == 10 and sum(item.accepted for item in phase) == 5
        for episode in bank.episodes
        for phase in (episode.opening, episode.terminal)
    )
    return BankMetricsV3(
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
        formula_fold,
        proxy_fold,
        renderer_digest,
        position_digest,
        folds_match,
        renderer_passed,
        position_passed,
        phase_profiles,
    )


def _semantic_vector(bank: EpisodeBankV3) -> tuple[int, ...]:
    episodes = tuple(
        episode
        for episode in bank.episodes
        if episode.request.target_family == "binary_piece"
    )
    if len(episodes) != V3_BINARY_EPISODES_PER_STAGE:
        raise ValueError("v3 semantic estimand requires exactly 400 binary-piece episodes")
    totals = [0] * len(SEMANTIC_METRICS_V3)
    for episode in episodes:
        rule = cast(BinaryRule, episode.target.rule)
        totals[0] += episode.target.truth.true_count
        totals[1] += len(rule.canonical_json)
        present = {literal.atom.op for literal in rule.args}
        for index, atom_op in enumerate(ATOM_OPS, start=2):
            totals[index] += int(atom_op in present)
        offset = 2 + len(ATOM_OPS)
        for index, value in enumerate(target_shadow_cells_v2(episode.target, episode.shadow)):
            totals[offset + index] += value
    return tuple(totals)


@dataclass(frozen=True, slots=True)
class SemanticGateResultV3:
    derivation_digest: str
    warm_totals: tuple[int, ...]
    capability_totals: tuple[int, ...]
    absolute_differences: tuple[int, ...]
    upper_thresholds: tuple[int, ...]
    component_passes: tuple[bool, ...]

    @property
    def passed(self) -> bool:
        return all(self.component_passes)

    def as_obj(self) -> dict[str, Any]:
        return {
            "derivation_digest": self.derivation_digest,
            "metric_names": list(SEMANTIC_METRICS_V3),
            "warm_totals": dict(zip(SEMANTIC_METRICS_V3, self.warm_totals, strict=True)),
            "capability_totals": dict(
                zip(SEMANTIC_METRICS_V3, self.capability_totals, strict=True)
            ),
            "absolute_differences": dict(
                zip(SEMANTIC_METRICS_V3, self.absolute_differences, strict=True)
            ),
            "upper_thresholds": dict(
                zip(SEMANTIC_METRICS_V3, self.upper_thresholds, strict=True)
            ),
            "component_passes": dict(
                zip(SEMANTIC_METRICS_V3, self.component_passes, strict=True)
            ),
            "passed": self.passed,
        }

    @property
    def digest(self) -> str:
        return json_digest(self.as_obj(), domain="goalzendo-interactive-semantic-gate-result-v3")


def semantic_gate_result_v3(
    candidate: MaterializedCandidateV3,
) -> SemanticGateResultV3:
    warm = _semantic_vector(candidate.warm)
    capability = _semantic_vector(candidate.capability)
    differences = tuple(abs(left - right) for left, right in zip(warm, capability, strict=True))
    thresholds = candidate.design.semantic_gate.upper_thresholds
    return SemanticGateResultV3(
        candidate.design.semantic_gate.digest,
        warm,
        capability,
        differences,
        thresholds,
        tuple(value <= threshold for value, threshold in zip(differences, thresholds, strict=True)),
    )


@dataclass(frozen=True, slots=True)
class CandidateAuditV3:
    design_digest: str
    warm: BankMetricsV3
    capability: BankMetricsV3
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
            "schema_version": CAPABILITY_QA_SCHEMA_VERSION_V3,
            "scope": "single-prospective-local-materialization-non-authorizing",
            "design_digest": self.design_digest,
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
        return json_digest(self.as_obj(), domain="goalzendo-interactive-candidate-audit-v3")


@lru_cache(maxsize=2)
def build_candidate_audit_v3(
    config: SurfaceLeakageConfig | None = None,
    design: ProspectiveDesignV3 | None = None,
) -> CandidateAuditV3:
    """Materialize and audit the frozen design; authorization remains false."""

    if config is None:
        config = SurfaceLeakageConfig()
    if design is None:
        design = build_prospective_design_v3()
    candidate = materialize_prospective_design_v3(design)
    warm_leakage = audit_hidden_episode_bank_v3_surface_leakage(
        candidate.warm, config=config
    )
    capability_leakage = audit_hidden_episode_bank_v3_surface_leakage(
        candidate.capability, config=config
    )
    warm_metrics = bank_metrics_v3(candidate.warm, warm_leakage)
    capability_metrics = bank_metrics_v3(candidate.capability, capability_leakage)
    semantic = semantic_gate_result_v3(candidate)
    warm_targets = {episode.target.truth_digest for episode in candidate.warm.episodes}
    cap_targets = {
        episode.target.truth_digest for episode in candidate.capability.episodes
    }
    disjoint = not bool(warm_targets & cap_targets)
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
    return CandidateAuditV3(
        design.digest,
        warm_metrics,
        capability_metrics,
        warm_leakage,
        capability_leakage,
        semantic,
        disjoint,
        tuple(blockers),
    )
