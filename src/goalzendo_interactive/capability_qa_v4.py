"""Fail-closed QA for the prospectively frozen schema-v4 repair."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, cast

from ._json import json_digest
from .capability_qa_v3 import (
    BankMetricsV3,
    SemanticGateResultV3,
    _semantic_vector,
    bank_metrics_v3,
    hidden_episode_bank_v3_surface_targets,
)
from .capability_v4 import (
    MaterializedCandidateV4,
    ProspectiveDesignV4,
    build_prospective_design_v4,
    materialize_prospective_design_v4,
)
from .leakage import (
    SurfaceLeakageAuditReport,
    SurfaceLeakageConfig,
    audit_surface_targets,
)

CAPABILITY_QA_SCHEMA_VERSION_V4 = 4


def _audit_bank(
    candidate_bank: Any,
    *,
    config: SurfaceLeakageConfig,
) -> SurfaceLeakageAuditReport:
    view = candidate_bank.v3_view()
    return audit_surface_targets(
        candidate_bank.spec.bank_id,
        hidden_episode_bank_v3_surface_targets(view),
        config=config,
    )


def _semantic_gate(candidate: MaterializedCandidateV4) -> SemanticGateResultV3:
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
class CandidateAuditV4:
    design_digest: str
    pair_population_audit_digest: str
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
            "schema_version": CAPABILITY_QA_SCHEMA_VERSION_V4,
            "scope": "single-prospective-local-materialization-non-authorizing",
            "design_digest": self.design_digest,
            "pair_population_audit_digest": self.pair_population_audit_digest,
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
        return json_digest(self.as_obj(), domain="goalzendo-interactive-candidate-audit-v4")


@lru_cache(maxsize=2)
def build_candidate_audit_v4(
    config: SurfaceLeakageConfig | None = None,
    design: ProspectiveDesignV4 | None = None,
) -> CandidateAuditV4:
    """Materialize the frozen v4 design once and preserve its gate outcomes."""

    if config is None:
        config = SurfaceLeakageConfig()
    if design is None:
        design = build_prospective_design_v4()
    candidate = materialize_prospective_design_v4(design)
    warm_leakage = _audit_bank(candidate.warm, config=config)
    capability_leakage = _audit_bank(candidate.capability, config=config)
    warm_metrics = bank_metrics_v3(cast(Any, candidate.warm), warm_leakage)
    capability_metrics = bank_metrics_v3(
        cast(Any, candidate.capability), capability_leakage
    )
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
    return CandidateAuditV4(
        design.digest,
        design.pair_population_audit.digest,
        warm_metrics,
        capability_metrics,
        warm_leakage,
        capability_leakage,
        semantic,
        disjoint,
        tuple(blockers),
    )
