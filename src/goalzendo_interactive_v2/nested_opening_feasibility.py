"""Source-only nested-opening construction feasibility for prospective G03-v2.

This module consumes an exact :class:`EvaluationCensusPlanV1` only as an
identity schedule.  It plans and, in a separate observed report, attempts one
shared thirteen-scene union for each planned composed-rule identity.  Four
ten-scene openings are canonical prefixes of that union.  The construction is
outcome blind: it never selects an ``m/q`` cell, matches difficulty, renders a
prompt, materializes a protocol quartet, or authorizes execution.

The production plan API accepts only the frozen 144-attempt parent budget.
Reduced artifacts have a distinct test-only kind/status and can be built or
parsed only through functions whose names end in ``_for_testing_v1``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, cast

from goalzendo_interactive import catalog as catalog_module
from goalzendo_interactive import query as query_module
from goalzendo_interactive import stage_partitions_v2 as stage_partitions_module
from goalzendo_interactive.catalog import CatalogEntry, build_rule_catalog
from goalzendo_interactive.provenance import interactive_source_provenance
from goalzendo_interactive.schema import SCENE_COUNT

from . import challenge_query as challenge_query_module
from . import evaluation_census as evaluation_census_module
from . import population_audit as population_audit_module
from . import role_schema as role_schema_module
from .evaluation_census import (
    PRODUCTION_MIRROR_ATTEMPTS_PER_STRATUM,
    SUPPORTED_RANKING_ROW_COUNT,
    CensusAttemptPlanV1,
    CensusConstructedOpeningV1,
    CensusMemberPlanV1,
    CensusRankingTableV1,
    EvaluationCensusPlanV1,
    ScheduleVariant,
    assess_evaluation_opening_v1,
    parse_evaluation_census_plan_v1,
)
from .population_audit import COMPOSED_STRATA, build_supported_catalog_contract_v2
from .role_schema import EVIDENCE_GEOMETRIES, OfficialTargetSideV2, build_evidence_schedule_v2

NESTED_OPENING_FEASIBILITY_SCHEMA_VERSION = 1
PRODUCTION_MIRROR_ATTEMPT_COUNT = 144
MEMBERS_PER_ATTEMPT = 2
PRODUCTION_MEMBER_CONSTRUCTION_COUNT = 288
DERIVED_OPENINGS_PER_MEMBER = 4
PRODUCTION_OPENING_ASSESSMENT_COUNT = 1_152
COMMON_UNION_SLOT_COUNT = 13
CANDIDATE_WINDOW_K = 32
SUPPORTED_VERSION_SPACE_FLOOR = 8

_PLAN_KIND = "g03-v2-nested-opening-construction-feasibility-plan-v1"
_TEST_PLAN_KIND = "g03-v2-nested-opening-construction-feasibility-plan-for-testing-v1"
_REPORT_KIND = "g03-v2-nested-opening-construction-feasibility-report-v1"
_TEST_REPORT_KIND = "g03-v2-nested-opening-construction-feasibility-report-for-testing-v1"
_PLAN_STATUS = "prospective_outcome_free_nonauthorizing"
_TEST_PLAN_STATUS = "prospective_engineering_fixture_outcome_free_nonauthorizing"
_REPORT_STATUS = "observed_complete_nonauthorizing"
_TEST_REPORT_STATUS = "observed_complete_engineering_fixture_nonauthorizing"

_PLAN_DOMAIN = "goalzendo-interactive-v2-nested-opening-construction-feasibility-plan-v1"
_REPORT_DOMAIN = "goalzendo-interactive-v2-nested-opening-construction-feasibility-report-v1"
_SOURCE_MANIFEST_DOMAIN = "goalzendo-interactive-v2-nested-opening-feasibility-source-manifest-v1"
_UPSTREAM_IDENTITY_DOMAIN = (
    "goalzendo-interactive-v2-nested-opening-feasibility-upstream-identity-schedule-v1"
)
_GENERATOR_CONTRACT_DOMAIN = "goalzendo-interactive-v2-nested-opening-feasibility-generator-contract-v1"
_ATTEMPT_PLAN_DOMAIN = "goalzendo-interactive-v2-nested-opening-feasibility-attempt-plan-v1"
_MEMBER_PLAN_DOMAIN = "goalzendo-interactive-v2-nested-opening-feasibility-member-plan-v1"
_SLOT_SET_DOMAIN = "goalzendo-interactive-v2-nested-opening-feasibility-slot-set-v1"
_CANDIDATE_WINDOW_DOMAIN = "goalzendo-interactive-v2-nested-opening-feasibility-candidate-window-v1"
_SELECTION_STEP_DOMAIN = "goalzendo-interactive-v2-nested-opening-feasibility-selection-step-v1"
_SELECTION_TRACE_DOMAIN = "goalzendo-interactive-v2-nested-opening-feasibility-selection-trace-v1"
_COMMON_UNION_DOMAIN = "goalzendo-interactive-v2-nested-opening-feasibility-common-union-v1"
_DERIVED_OPENING_DOMAIN = "goalzendo-interactive-v2-nested-opening-feasibility-derived-opening-v1"
_EXACT_MATCH_DOMAIN = "goalzendo-interactive-v2-nested-opening-feasibility-exact-match-v1"
_MEMBER_OBSERVATION_DOMAIN = "goalzendo-interactive-v2-nested-opening-feasibility-member-observation-v1"
_ATTEMPT_OBSERVATION_DOMAIN = "goalzendo-interactive-v2-nested-opening-feasibility-attempt-observation-v1"
_MIRROR_OVERLAP_DOMAIN = "goalzendo-interactive-v2-nested-opening-feasibility-mirror-overlap-v1"
_MQ_SUMMARY_DOMAIN = "goalzendo-interactive-v2-nested-opening-feasibility-m-q-summary-v1"
_GEOMETRY_DISPLAY_BINDING_DOMAIN = (
    "goalzendo-interactive-v2-nested-opening-feasibility-geometry-display-binding-v1"
)
_CENSUS_ATTEMPT_DOMAIN = "goalzendo-interactive-v2-evaluation-census-attempt-v1"
_SLOT_ORDER_DOMAIN = b"goalzendo-interactive-v2-nested-opening-feasibility-slot-order-v1\0"
_SCENE_ORDER_DOMAIN = b"goalzendo-interactive-v2-nested-opening-feasibility-scene-order-v1\0"
_GEOMETRY_DISPLAY_ORDER_DOMAIN = (
    b"goalzendo-interactive-v2-nested-opening-feasibility-geometry-display-order-v1\0"
)

_AUTHORIZATION: dict[str, str | bool] = {
    "scope": "nested_opening_construction_feasibility_only",
    "g01_authorized": False,
    "g03_capability_launch_authorized": False,
    "g03_scientific_launch_authorized": False,
    "production_bank_authorized": False,
    "model_execution_authorized": False,
    "weight_updates_authorized": False,
    "launch_authorized": False,
}

_SELECTION_BOUNDARY: dict[str, None] = {
    "external_registration_reference": None,
    "registered_power_artifact_digest": None,
    "positive_m_q_quota": None,
    "selected_m_q_cells": None,
    "matched_bank_size": None,
    "matcher_specification": None,
    "canonical_integer_matching_cost": None,
    "oversupply_selection_rule": None,
    "aggregate_balance_target": None,
    "renderer_assignment": None,
    "token_length_bins": None,
    "challenge_panel_generator": None,
    "intervention_generator": None,
    "production_manifest_digest": None,
    "pre_evaluation_checkpoint_digest": None,
    "runtime_receipt_digest": None,
}

_TRUE_CONSTRUCTION_CLAIMS = (
    "engineering_nested_opening_construction_feasibility_described",
    "exact_identity_schedule_replayed",
    "fixed_budget_ledger_complete",
    "common_union_prefix_contract_rederived",
    "completed_opening_assessments_rederived",
)
_FALSE_CONSTRUCTION_CLAIMS = (
    "protocol_quartet_candidate",
    "production_nested_opening_pool_frozen",
    "positive_m_q_quota_selected",
    "matched_bank_size_selected",
    "difficulty_matcher_run",
    "oversupply_selected",
    "aggregate_balance_audited",
    "planned_display_order_balance_verified",
    "renderer_assignment_bound",
    "renderer_balance_audited",
    "model_visible_demonstration_order_bound",
    "rendered_prompt_bytes_bound",
    "rendered_token_length_bound",
    "tokenizer_artifact_bound",
    "challenge_reservoir_materialized",
    "eleven_panels_materialized",
    "panel_version_space_coverage_verified",
    "minimum_challenge_cores_materialized",
    "primary_panel_salt_registered",
    "primary_panel_resolved",
    "common_untouched_panel_runtime_selected",
    "intervention_bank_materialized",
    "selected_bank_cross_triple_scene_disjointness_verified",
    "cross_mirror_unit_scene_disjointness_verified",
    "cross_stage_scene_disjointness_verified",
    "evaluation_checkpoint_bound",
    "runtime_episode_order_bound",
    "context_reset_observed",
    "weight_update_absence_observed",
    "private_ast_branch_observed",
    "ast_blind_terminal_replay_observed",
    "model_calls_observed",
    "model_outcomes_present",
    "runtime_measurements_present",
    "external_registration_verified",
    "registered_power_verified",
    "production_manifest_bound",
    "production_bank_authorized",
    "g01_authorized",
    "g03_capability_launch_authorized",
    "g03_scientific_launch_authorized",
    "launch_authorized",
)

_GEOMETRY_SLUGS = tuple(geometry.slug for geometry in EVIDENCE_GEOMETRIES)
_UNION_BY_SIDE: dict[int, tuple[int, ...]] = {
    0: (5, 1, 1, 0, 0, 0, 1, 5),
    1: (5, 1, 0, 0, 0, 1, 1, 5),
}


class NestedOpeningFeasibilityV2Error(ValueError):
    """Raised when a feasibility artifact fails exact replay."""


def _dump_json(value: Any, *, sort_keys: bool = False) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=sort_keys,
        )
    except (TypeError, ValueError) as exc:
        raise NestedOpeningFeasibilityV2Error(f"value is not canonical JSON: {exc}") from exc


def _load_json(text: str) -> Any:
    if type(text) is not str or not text:
        raise NestedOpeningFeasibilityV2Error("JSON input must be a nonempty string")

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise NestedOpeningFeasibilityV2Error(f"duplicate JSON object key: {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise NestedOpeningFeasibilityV2Error(f"non-finite JSON constant is forbidden: {value}")

    try:
        return json.loads(text, object_pairs_hook=no_duplicates, parse_constant=reject_constant)
    except NestedOpeningFeasibilityV2Error:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise NestedOpeningFeasibilityV2Error(f"invalid JSON: {exc}") from exc


def _digest(value: Any, *, domain: str) -> str:
    payload = domain.encode("ascii") + b"\0" + _dump_json(value, sort_keys=True).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _upstream_census_digest(value: Any, *, domain: str) -> str:
    """Replay the upstream census' insertion-ordered digest convention."""

    payload = domain.encode("ascii") + b"\0" + _dump_json(value).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _hash_parts(domain: bytes, *parts: str | int) -> bytes:
    digest = hashlib.sha256()
    digest.update(domain)
    for part in parts:
        digest.update(str(part).encode("ascii"))
        digest.update(b"\0")
    return digest.digest()


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: object, *, name: str) -> str:
    if not _is_sha256(value):
        raise NestedOpeningFeasibilityV2Error(f"{name} must be a lowercase SHA-256")
    return cast(str, value)


def _require_integer(
    value: object,
    *,
    name: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise NestedOpeningFeasibilityV2Error(f"{name} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise NestedOpeningFeasibilityV2Error(f"{name} must be an integer <= {maximum}")
    return value


def _require_mapping(value: object, fields: tuple[str, ...], *, name: str) -> Mapping[str, Any]:
    if type(value) is not dict or tuple(value) != fields:
        raise NestedOpeningFeasibilityV2Error(f"{name} has noncanonical or reordered fields")
    return cast(Mapping[str, Any], value)


def _require_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise NestedOpeningFeasibilityV2Error(f"{name} must be a boolean")
    return value


def _source_sha(module_file: str | None, *, name: str) -> str:
    if module_file is None:
        raise NestedOpeningFeasibilityV2Error(f"{name} source has no file")
    path = Path(module_file)
    if path.is_symlink() or not path.is_file():
        raise NestedOpeningFeasibilityV2Error(f"{name} source must be one ordinary file")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _generator_contract_obj() -> dict[str, Any]:
    return {
        "artifact_scope": "nested-opening construction feasibility only",
        "parent_usage": "canonical EvaluationCensusPlanV1 identity schedule only",
        "production_identity_budget": "9 formula strata x 16 attempts x 2 mirror members",
        "union_vectors_by_noisy_p_target_side": {
            "0": list(_UNION_BY_SIDE[0]),
            "1": list(_UNION_BY_SIDE[1]),
        },
        "common_union_slot_count": COMMON_UNION_SLOT_COUNT,
        "candidate_window_k": CANDIDATE_WINDOW_K,
        "supported_version_space_floor": SUPPORTED_VERSION_SPACE_FLOOR,
        "slot_order": "ascending prefix rank, slot-order hash, then joint-cell index",
        "scene_order_inputs": [
            "generator_seed",
            "parent_attempt_digest",
            "member_position",
            "triple_digest",
            "noisy_p_target_side",
            "joint_cell_index",
            "scene_index",
        ],
        "simultaneous_filtering": "all affected geometry spaces from one pre-step snapshot; commit all or none",
        "selection_uses_m_q_salience_or_assessor_outcomes": False,
        "display_order_is_private_precommitted_metadata_only": True,
        "mirror_overlap_is_diagnostic_not_acceptance_gate": True,
        "no_sliding_window": True,
        "no_resampling": True,
        "no_backtracking": True,
        "no_retry": True,
        "no_early_stop": True,
        "no_identity_replacement": True,
    }


@dataclass(frozen=True, slots=True)
class NestedSourceBindingV1:
    frozen_v1_source_fingerprint: str
    catalog_source_sha256: str
    query_source_sha256: str
    stage_partitions_source_sha256: str
    role_schema_source_sha256: str
    population_audit_source_sha256: str
    challenge_query_source_sha256: str
    evaluation_census_source_sha256: str
    nested_opening_feasibility_source_sha256: str
    source_manifest_digest: str

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            _require_sha256(getattr(self, field_name), name=field_name)
        if self.source_manifest_digest != _digest(self._manifest_obj(), domain=_SOURCE_MANIFEST_DOMAIN):
            raise NestedOpeningFeasibilityV2Error("source-manifest digest is inconsistent")

    def _manifest_obj(self) -> dict[str, str]:
        return {
            "frozen_v1_source_fingerprint": self.frozen_v1_source_fingerprint,
            "catalog_source_sha256": self.catalog_source_sha256,
            "query_source_sha256": self.query_source_sha256,
            "stage_partitions_source_sha256": self.stage_partitions_source_sha256,
            "role_schema_source_sha256": self.role_schema_source_sha256,
            "population_audit_source_sha256": self.population_audit_source_sha256,
            "challenge_query_source_sha256": self.challenge_query_source_sha256,
            "evaluation_census_source_sha256": self.evaluation_census_source_sha256,
            "nested_opening_feasibility_source_sha256": (self.nested_opening_feasibility_source_sha256),
        }

    def as_obj(self) -> dict[str, Any]:
        return {**self._manifest_obj(), "source_manifest_digest": self.source_manifest_digest}


def _current_source_binding() -> NestedSourceBindingV1:
    manifest = {
        "frozen_v1_source_fingerprint": interactive_source_provenance().fingerprint,
        "catalog_source_sha256": _source_sha(catalog_module.__file__, name="catalog"),
        "query_source_sha256": _source_sha(query_module.__file__, name="query"),
        "stage_partitions_source_sha256": _source_sha(
            stage_partitions_module.__file__, name="stage partitions"
        ),
        "role_schema_source_sha256": _source_sha(role_schema_module.__file__, name="role schema"),
        "population_audit_source_sha256": _source_sha(
            population_audit_module.__file__, name="population audit"
        ),
        "challenge_query_source_sha256": _source_sha(challenge_query_module.__file__, name="challenge query"),
        "evaluation_census_source_sha256": _source_sha(
            evaluation_census_module.__file__, name="evaluation census"
        ),
        "nested_opening_feasibility_source_sha256": _source_sha(__file__, name="nested-opening feasibility"),
    }
    return NestedSourceBindingV1(
        **manifest,
        source_manifest_digest=_digest(manifest, domain=_SOURCE_MANIFEST_DOMAIN),
    )


@dataclass(frozen=True, slots=True)
class NestedUpstreamIdentityBindingV1:
    canonical_census_plan_bytes_sha256: str
    canonical_census_plan_byte_count: int
    evaluation_census_plan_digest: str
    evaluation_census_source_manifest_digest: str
    evaluation_census_generator_contract_digest: str
    source_catalog_digest: str
    supported_catalog_digest: str
    supported_identity_count: int
    stage_partition_digest: str
    identity_schedule_digest: str
    mirror_attempt_count: int
    member_count: int

    def __post_init__(self) -> None:
        for name in (
            "canonical_census_plan_bytes_sha256",
            "evaluation_census_plan_digest",
            "evaluation_census_source_manifest_digest",
            "evaluation_census_generator_contract_digest",
            "source_catalog_digest",
            "supported_catalog_digest",
            "stage_partition_digest",
            "identity_schedule_digest",
        ):
            _require_sha256(getattr(self, name), name=name)
        _require_integer(
            self.canonical_census_plan_byte_count,
            name="canonical census plan byte count",
            minimum=1,
        )
        _require_integer(
            self.supported_identity_count,
            name="supported identity count",
            minimum=SUPPORTED_RANKING_ROW_COUNT,
            maximum=SUPPORTED_RANKING_ROW_COUNT,
        )
        _require_integer(self.mirror_attempt_count, name="mirror attempt count", minimum=1)
        _require_integer(self.member_count, name="member count", minimum=2)
        if self.member_count != self.mirror_attempt_count * MEMBERS_PER_ATTEMPT:
            raise NestedOpeningFeasibilityV2Error("upstream member count is inconsistent")

    def as_obj(self) -> dict[str, Any]:
        return {
            "canonical_census_plan_bytes_sha256": self.canonical_census_plan_bytes_sha256,
            "canonical_census_plan_byte_count": self.canonical_census_plan_byte_count,
            "evaluation_census_plan_digest": self.evaluation_census_plan_digest,
            "evaluation_census_source_manifest_digest": (self.evaluation_census_source_manifest_digest),
            "evaluation_census_generator_contract_digest": (self.evaluation_census_generator_contract_digest),
            "source_catalog_digest": self.source_catalog_digest,
            "supported_catalog_digest": self.supported_catalog_digest,
            "supported_identity_count": self.supported_identity_count,
            "stage_partition_digest": self.stage_partition_digest,
            "identity_schedule_digest": self.identity_schedule_digest,
            "mirror_attempt_count": self.mirror_attempt_count,
            "member_count": self.member_count,
        }


@dataclass(frozen=True, slots=True)
class NestedGeneratorBindingV1:
    seed: str
    contract_digest: str
    source_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.seed, name="generator seed")
        _require_sha256(self.contract_digest, name="generator contract digest")
        _require_sha256(self.source_sha256, name="generator source sha256")
        if self.contract_digest != _digest(_generator_contract_obj(), domain=_GENERATOR_CONTRACT_DOMAIN):
            raise NestedOpeningFeasibilityV2Error("generator-contract digest is inconsistent")

    def as_obj(self) -> dict[str, Any]:
        return {
            "generator_schema_version": NESTED_OPENING_FEASIBILITY_SCHEMA_VERSION,
            "generator_seed": self.seed,
            "generator_contract": _generator_contract_obj(),
            "generator_contract_digest": self.contract_digest,
            "generator_source_sha256": self.source_sha256,
        }


@dataclass(frozen=True, slots=True)
class NestedOpeningSlotPlanV1:
    slot_position: int
    prefix_rank: int
    joint_cell_index: int
    joint_cell_slug: str
    affected_geometry_slugs: tuple[str, ...]
    slot_order_hash: str

    def __post_init__(self) -> None:
        _require_integer(self.slot_position, name="slot position", maximum=12)
        _require_integer(self.prefix_rank, name="prefix rank", maximum=4)
        _require_integer(self.joint_cell_index, name="joint cell index", maximum=7)
        if self.joint_cell_slug != f"{self.joint_cell_index:03b}":
            raise NestedOpeningFeasibilityV2Error("slot joint-cell slug is inconsistent")
        if (
            type(self.affected_geometry_slugs) is not tuple
            or not self.affected_geometry_slugs
            or any(slug not in _GEOMETRY_SLUGS for slug in self.affected_geometry_slugs)
            or self.affected_geometry_slugs
            != tuple(slug for slug in _GEOMETRY_SLUGS if slug in self.affected_geometry_slugs)
        ):
            raise NestedOpeningFeasibilityV2Error(
                "slot affected geometries must be a nonempty canonical subsequence"
            )
        _require_sha256(self.slot_order_hash, name="slot order hash")

    def as_obj(self) -> dict[str, Any]:
        return {
            "slot_position": self.slot_position,
            "cell_prefix_rank": self.prefix_rank,
            "joint_cell_index": self.joint_cell_index,
            "joint_cell_slug": self.joint_cell_slug,
            "affected_geometry_slugs": list(self.affected_geometry_slugs),
            "slot_order_hash": self.slot_order_hash,
        }


@dataclass(frozen=True, slots=True)
class NestedOpeningMemberPlanV1:
    member_position: int
    composed_rule_id: str
    composed_truth_digest: str
    triple_digest: str
    noisy_p_target_side: int
    schedule_variants: tuple[ScheduleVariant, ...]
    union_joint_truth_cell_counts: tuple[int, ...]
    planned_common_union_slot_count: int
    planned_slots: tuple[NestedOpeningSlotPlanV1, ...]
    slot_set_digest: str
    geometry_display_order: tuple[str, ...]
    geometry_display_binding_digest: str
    member_plan_digest: str

    def __post_init__(self) -> None:
        if type(self.planned_slots) is not tuple or any(
            type(slot) is not NestedOpeningSlotPlanV1 for slot in self.planned_slots
        ):
            raise NestedOpeningFeasibilityV2Error("planned slots require exact slot-plan objects")
        _require_integer(self.member_position, name="member position", maximum=1)
        _require_sha256(self.composed_truth_digest, name="composed truth digest")
        _require_sha256(self.triple_digest, name="triple digest")
        _require_integer(self.noisy_p_target_side, name="noisy P target side", maximum=1)
        expected_variants = cast(
            tuple[ScheduleVariant, ...],
            (
                "a0_b0",
                f"a1_b0_y{self.noisy_p_target_side}",
                "a0_b2",
                f"a1_b2_y{self.noisy_p_target_side}",
            ),
        )
        if self.schedule_variants != expected_variants:
            raise NestedOpeningFeasibilityV2Error("member schedule variants are inconsistent")
        if self.union_joint_truth_cell_counts != _UNION_BY_SIDE[self.noisy_p_target_side]:
            raise NestedOpeningFeasibilityV2Error("member union vector is not frozen Y0 or Y1")
        if sum(self.union_joint_truth_cell_counts) != COMMON_UNION_SLOT_COUNT:
            raise NestedOpeningFeasibilityV2Error("member union must contain thirteen scenes")
        _require_integer(
            self.planned_common_union_slot_count,
            name="planned common-union slot count",
            minimum=COMMON_UNION_SLOT_COUNT,
            maximum=COMMON_UNION_SLOT_COUNT,
        )
        if len(self.planned_slots) != COMMON_UNION_SLOT_COUNT:
            raise NestedOpeningFeasibilityV2Error("member must preserve all thirteen slot records")
        if tuple(slot.slot_position for slot in self.planned_slots) != tuple(range(COMMON_UNION_SLOT_COUNT)):
            raise NestedOpeningFeasibilityV2Error("member slots are not in canonical position order")
        if len({(slot.joint_cell_index, slot.prefix_rank) for slot in self.planned_slots}) != 13:
            raise NestedOpeningFeasibilityV2Error("member slots are not distinct")
        observed_counts = tuple(
            sum(slot.joint_cell_index == cell for slot in self.planned_slots) for cell in range(8)
        )
        if observed_counts != self.union_joint_truth_cell_counts:
            raise NestedOpeningFeasibilityV2Error("member slots do not realize the union vector")
        if self.slot_set_digest != _digest(
            [slot.as_obj() for slot in self.planned_slots], domain=_SLOT_SET_DOMAIN
        ):
            raise NestedOpeningFeasibilityV2Error("slot-set digest is inconsistent")
        if tuple(sorted(self.geometry_display_order)) != tuple(sorted(_GEOMETRY_SLUGS)):
            raise NestedOpeningFeasibilityV2Error("geometry display order is not a permutation")
        _require_sha256(self.geometry_display_binding_digest, name="geometry display binding digest")
        _require_sha256(self.member_plan_digest, name="member plan digest")
        if self.member_plan_digest != _digest(self._unsigned_obj(), domain=_MEMBER_PLAN_DOMAIN):
            raise NestedOpeningFeasibilityV2Error("member-plan digest is inconsistent")

    def _unsigned_obj(self) -> dict[str, Any]:
        return {
            "member_position": self.member_position,
            "composed_rule_id": self.composed_rule_id,
            "composed_truth_digest": self.composed_truth_digest,
            "triple_digest": self.triple_digest,
            "noisy_p_target_side": self.noisy_p_target_side,
            "schedule_variants": list(self.schedule_variants),
            "union_cell_counts": list(self.union_joint_truth_cell_counts),
            "planned_union_slots": [slot.as_obj() for slot in self.planned_slots],
            "planned_union_slots_digest": self.slot_set_digest,
            "planned_geometry_display_order": list(self.geometry_display_order),
            "planned_geometry_display_order_digest": self.geometry_display_binding_digest,
        }

    def as_obj(self) -> dict[str, Any]:
        return {**self._unsigned_obj(), "member_plan_digest": self.member_plan_digest}


@dataclass(frozen=True, slots=True)
class NestedOpeningAttemptPlanV1:
    formula_stratum: str
    canonical_position: int
    parent_attempt_digest: str
    placard_rule_id: str
    placard_truth_digest: str
    literal_rule_id: str
    literal_truth_digest: str
    members: tuple[NestedOpeningMemberPlanV1, NestedOpeningMemberPlanV1]
    attempt_plan_digest: str

    def __post_init__(self) -> None:
        if (
            type(self.members) is not tuple
            or len(self.members) != MEMBERS_PER_ATTEMPT
            or any(type(member) is not NestedOpeningMemberPlanV1 for member in self.members)
        ):
            raise NestedOpeningFeasibilityV2Error("attempt members require two exact member-plan objects")
        _require_integer(self.canonical_position, name="canonical position", maximum=15)
        for value, name in (
            (self.parent_attempt_digest, "parent attempt digest"),
            (self.placard_truth_digest, "placard truth digest"),
            (self.literal_truth_digest, "literal truth digest"),
            (self.attempt_plan_digest, "attempt plan digest"),
        ):
            _require_sha256(value, name=name)
        if tuple(member.member_position for member in self.members) != (0, 1):
            raise NestedOpeningFeasibilityV2Error("attempt members are reordered")
        if self.members[0].noisy_p_target_side + self.members[1].noisy_p_target_side != 1:
            raise NestedOpeningFeasibilityV2Error("attempt members are not opposite-side mirrors")
        if self.attempt_plan_digest != _digest(self._unsigned_obj(), domain=_ATTEMPT_PLAN_DOMAIN):
            raise NestedOpeningFeasibilityV2Error("attempt-plan digest is inconsistent")

    def _unsigned_obj(self) -> dict[str, Any]:
        return {
            "formula_stratum": self.formula_stratum,
            "canonical_position": self.canonical_position,
            "upstream_attempt_digest": self.parent_attempt_digest,
            "placard_rule_id": self.placard_rule_id,
            "placard_truth_digest": self.placard_truth_digest,
            "literal_rule_id": self.literal_rule_id,
            "literal_truth_digest": self.literal_truth_digest,
            "members": [member.as_obj() for member in self.members],
        }

    def as_obj(self) -> dict[str, Any]:
        return {**self._unsigned_obj(), "attempt_plan_digest": self.attempt_plan_digest}


@dataclass(frozen=True, slots=True)
class NestedOpeningFeasibilityPlanV1:
    source_binding: NestedSourceBindingV1
    upstream_identity_plan_binding: NestedUpstreamIdentityBindingV1
    generator_binding: NestedGeneratorBindingV1
    attempts_per_formula_stratum: int
    attempts: tuple[NestedOpeningAttemptPlanV1, ...]
    engineering_budget_override: bool

    def __post_init__(self) -> None:
        if type(self.source_binding) is not NestedSourceBindingV1:
            raise NestedOpeningFeasibilityV2Error("plan requires an exact source binding")
        if type(self.upstream_identity_plan_binding) is not NestedUpstreamIdentityBindingV1:
            raise NestedOpeningFeasibilityV2Error("plan requires an exact upstream binding")
        if type(self.generator_binding) is not NestedGeneratorBindingV1:
            raise NestedOpeningFeasibilityV2Error("plan requires an exact generator binding")
        if type(self.attempts) is not tuple or any(
            type(attempt) is not NestedOpeningAttemptPlanV1 for attempt in self.attempts
        ):
            raise NestedOpeningFeasibilityV2Error("plan requires exact attempt-plan objects")
        _require_integer(
            self.attempts_per_formula_stratum,
            name="attempts per formula stratum",
            minimum=1,
            maximum=PRODUCTION_MIRROR_ATTEMPTS_PER_STRATUM,
        )
        _require_bool(self.engineering_budget_override, name="engineering budget override")
        expected_count = len(COMPOSED_STRATA) * self.attempts_per_formula_stratum
        if len(self.attempts) != expected_count:
            raise NestedOpeningFeasibilityV2Error("plan does not preserve the fixed parent budget")
        if self.upstream_identity_plan_binding.mirror_attempt_count != len(self.attempts):
            raise NestedOpeningFeasibilityV2Error("plan and upstream attempt counts differ")
        production = self.attempts_per_formula_stratum == PRODUCTION_MIRROR_ATTEMPTS_PER_STRATUM
        if self.engineering_budget_override == production:
            raise NestedOpeningFeasibilityV2Error("production and test-only budget markers are inconsistent")
        expected_keys = tuple(
            (f"{op}__neg{negated}", position)
            for op, negated in COMPOSED_STRATA
            for position in range(self.attempts_per_formula_stratum)
        )
        if tuple((row.formula_stratum, row.canonical_position) for row in self.attempts) != expected_keys:
            raise NestedOpeningFeasibilityV2Error("attempts are not in canonical parent order")
        composed_ids = tuple(
            member.composed_rule_id for attempt in self.attempts for member in attempt.members
        )
        if len(composed_ids) != len(set(composed_ids)):
            raise NestedOpeningFeasibilityV2Error("composed identities are reused")
        if self.generator_binding.source_sha256 != (
            self.source_binding.nested_opening_feasibility_source_sha256
        ):
            raise NestedOpeningFeasibilityV2Error("generator and source bindings disagree")
        parent_rows: list[dict[str, Any]] = []
        for attempt in self.attempts:
            parent_members = tuple(
                CensusMemberPlanV1(
                    member.member_position,
                    member.composed_rule_id,
                    member.composed_truth_digest,
                    member.triple_digest,
                    member.noisy_p_target_side,
                )
                for member in attempt.members
            )
            parent_unsigned = {
                "formula_stratum": attempt.formula_stratum,
                "canonical_position": attempt.canonical_position,
                "placard_rule_id": attempt.placard_rule_id,
                "placard_truth_digest": attempt.placard_truth_digest,
                "literal_rule_id": attempt.literal_rule_id,
                "literal_truth_digest": attempt.literal_truth_digest,
                "members": [member.as_obj() for member in parent_members],
            }
            if attempt.parent_attempt_digest != _upstream_census_digest(
                parent_unsigned, domain=_CENSUS_ATTEMPT_DOMAIN
            ):
                raise NestedOpeningFeasibilityV2Error("nested attempt differs from its parent attempt digest")
            parent_attempt = CensusAttemptPlanV1(
                attempt.formula_stratum,
                attempt.canonical_position,
                attempt.placard_rule_id,
                attempt.placard_truth_digest,
                attempt.literal_rule_id,
                attempt.literal_truth_digest,
                cast(
                    tuple[CensusMemberPlanV1, CensusMemberPlanV1],
                    parent_members,
                ),
                attempt.parent_attempt_digest,
            )
            expected_members = tuple(
                _build_member_plan(self.generator_binding.seed, parent_attempt, parent_member)
                for parent_member in parent_attempt.members
            )
            if attempt.members != expected_members:
                raise NestedOpeningFeasibilityV2Error(
                    "nested member plan differs from slot/display rederivation"
                )
            parent_rows.append(parent_attempt.as_obj())
        if self.upstream_identity_plan_binding.identity_schedule_digest != _digest(
            parent_rows, domain=_UPSTREAM_IDENTITY_DOMAIN
        ):
            raise NestedOpeningFeasibilityV2Error(
                "upstream identity-schedule digest differs from nested attempts"
            )

    @property
    def exact_144x2_budget_complete(self) -> bool:
        return not self.engineering_budget_override

    @property
    def plan_kind(self) -> str:
        return _TEST_PLAN_KIND if self.engineering_budget_override else _PLAN_KIND

    @property
    def status(self) -> str:
        return _TEST_PLAN_STATUS if self.engineering_budget_override else _PLAN_STATUS

    def _fixed_budget_obj(self) -> dict[str, Any]:
        member_count = len(self.attempts) * MEMBERS_PER_ATTEMPT
        return {
            "formula_stratum_count": len(COMPOSED_STRATA),
            "attempts_per_formula_stratum": self.attempts_per_formula_stratum,
            "mirror_attempt_count": len(self.attempts),
            "members_per_attempt": MEMBERS_PER_ATTEMPT,
            "planned_member_construction_count": member_count,
            "derived_openings_per_member": DERIVED_OPENINGS_PER_MEMBER,
            "planned_opening_assessment_count": member_count * DERIVED_OPENINGS_PER_MEMBER,
            "common_union_slots_per_member": COMMON_UNION_SLOT_COUNT,
            "candidate_window_k": CANDIDATE_WINDOW_K,
            "no_early_stop": True,
            "no_identity_replacement": True,
            "uses_exact_144x2_budget": self.exact_144x2_budget_complete,
            "engineering_budget_override": self.engineering_budget_override,
        }

    def _unsigned_obj_unverified(self) -> dict[str, Any]:
        return {
            "schema_version": NESTED_OPENING_FEASIBILITY_SCHEMA_VERSION,
            "plan_kind": self.plan_kind,
            "status": self.status,
            "protocol_quartet_candidate": False,
            "authorization": dict(_AUTHORIZATION),
            "source_binding": self.source_binding.as_obj(),
            "upstream_identity_plan_binding": self.upstream_identity_plan_binding.as_obj(),
            "generator_binding": self.generator_binding.as_obj(),
            "fixed_budget": self._fixed_budget_obj(),
            "selection_boundary": dict(_SELECTION_BOUNDARY),
            "attempts": [attempt.as_obj() for attempt in self.attempts],
        }

    def _artifact_obj_unverified(self) -> dict[str, Any]:
        """Private raw projection; authoritative public paths replay first."""

        return {
            **self._unsigned_obj_unverified(),
            "prospective_plan_digest": _plan_digest_unverified(self),
        }


def _plan_digest_unverified(plan: NestedOpeningFeasibilityPlanV1) -> str:
    return _digest(plan._unsigned_obj_unverified(), domain=_PLAN_DOMAIN)


def _sha256_text_bytes(text: str, *, name: str) -> tuple[str, int]:
    if type(text) is not str or not text:
        raise NestedOpeningFeasibilityV2Error(f"{name} must be nonempty text")
    try:
        payload = text.encode("ascii")
    except UnicodeEncodeError as exc:
        raise NestedOpeningFeasibilityV2Error(f"{name} must be ASCII") from exc
    return hashlib.sha256(payload).hexdigest(), len(payload)


def _parse_exact_upstream(
    census_plan_text: str,
    *,
    expected_census_plan_digest: str,
    expected_census_plan_bytes_sha256: str,
) -> EvaluationCensusPlanV1:
    _require_sha256(expected_census_plan_bytes_sha256, name="expected census plan bytes sha256")
    observed_sha, _count = _sha256_text_bytes(census_plan_text, name="census plan")
    if observed_sha != expected_census_plan_bytes_sha256:
        raise NestedOpeningFeasibilityV2Error("census plan bytes differ from external expectation")
    try:
        return parse_evaluation_census_plan_v1(
            census_plan_text,
            expected_digest=expected_census_plan_digest,
        )
    except (TypeError, ValueError) as exc:
        raise NestedOpeningFeasibilityV2Error(f"upstream census plan is invalid: {exc}") from exc


def _identity_schedule_obj(parent: EvaluationCensusPlanV1) -> list[dict[str, Any]]:
    return [attempt.as_obj() for attempt in parent.attempts]


def _upstream_binding(
    parent: EvaluationCensusPlanV1,
    census_plan_text: str,
) -> NestedUpstreamIdentityBindingV1:
    byte_sha, byte_count = _sha256_text_bytes(census_plan_text, name="census plan")
    return NestedUpstreamIdentityBindingV1(
        byte_sha,
        byte_count,
        parent.digest,
        parent.source_binding.source_manifest_digest,
        parent.generator_binding.generator_contract_digest,
        parent.catalog_binding.source_catalog_digest,
        parent.catalog_binding.supported_catalog_digest,
        parent.catalog_binding.supported_identity_count,
        parent.catalog_binding.stage_partition_digest,
        _digest(_identity_schedule_obj(parent), domain=_UPSTREAM_IDENTITY_DOMAIN),
        len(parent.attempts),
        len(parent.attempts) * MEMBERS_PER_ATTEMPT,
    )


def _slot_hash(
    seed: str,
    attempt: CensusAttemptPlanV1,
    member: CensusMemberPlanV1,
    prefix_rank: int,
    cell: int,
) -> str:
    return _hash_parts(
        _SLOT_ORDER_DOMAIN,
        seed,
        attempt.attempt_digest,
        member.member_position,
        member.triple_digest,
        member.noisy_p_target_side,
        prefix_rank,
        cell,
    ).hex()


def _member_variants(member: CensusMemberPlanV1) -> tuple[ScheduleVariant, ...]:
    side = member.noisy_p_target_side
    return cast(
        tuple[ScheduleVariant, ...],
        ("a0_b0", f"a1_b0_y{side}", "a0_b2", f"a1_b2_y{side}"),
    )


def _variant_schedule(variant: ScheduleVariant) -> tuple[int, ...]:
    if variant == "a0_b0":
        slug, side = "a0_b0", None
    elif variant == "a0_b2":
        slug, side = "a0_b2", None
    elif variant in ("a1_b0_y0", "a1_b0_y1"):
        slug, side = "a1_b0", OfficialTargetSideV2(int(variant[-1]))
    elif variant in ("a1_b2_y0", "a1_b2_y1"):
        slug, side = "a1_b2", OfficialTargetSideV2(int(variant[-1]))
    else:
        raise NestedOpeningFeasibilityV2Error(f"unknown schedule variant: {variant!r}")
    geometry = next(item for item in EVIDENCE_GEOMETRIES if item.slug == slug)
    return build_evidence_schedule_v2(
        geometry,
        a_error_target_side=side,
    ).joint_truth_cell_counts


def _display_order(
    seed: str,
    attempt: CensusAttemptPlanV1,
    member: CensusMemberPlanV1,
) -> tuple[tuple[str, ...], str]:
    indexed = tuple(enumerate(_GEOMETRY_SLUGS))
    ordered = tuple(
        slug
        for _index, slug in sorted(
            indexed,
            key=lambda item: (
                _hash_parts(
                    _GEOMETRY_DISPLAY_ORDER_DOMAIN,
                    seed,
                    attempt.attempt_digest,
                    member.member_position,
                    member.triple_digest,
                    item[1],
                ),
                item[0],
            ),
        )
    )
    binding = {
        "generator_seed": seed,
        "upstream_attempt_digest": attempt.attempt_digest,
        "member_position": member.member_position,
        "triple_digest": member.triple_digest,
        "planned_geometry_display_order": list(ordered),
    }
    return ordered, _digest(binding, domain=_GEOMETRY_DISPLAY_BINDING_DOMAIN)


def _build_member_plan(
    seed: str,
    attempt: CensusAttemptPlanV1,
    member: CensusMemberPlanV1,
) -> NestedOpeningMemberPlanV1:
    variants = _member_variants(member)
    schedules = {
        slug: _variant_schedule(variant) for slug, variant in zip(_GEOMETRY_SLUGS, variants, strict=True)
    }
    union = _UNION_BY_SIDE[member.noisy_p_target_side]
    raw_slots = [
        (
            prefix_rank,
            _slot_hash(seed, attempt, member, prefix_rank, cell),
            cell,
            tuple(slug for slug in _GEOMETRY_SLUGS if schedules[slug][cell] > prefix_rank),
        )
        for cell, count in enumerate(union)
        for prefix_rank in range(count)
    ]
    raw_slots.sort(key=lambda row: (row[0], row[1], row[2]))
    slots = tuple(
        NestedOpeningSlotPlanV1(position, rank, cell, f"{cell:03b}", affected, slot_hash)
        for position, (rank, slot_hash, cell, affected) in enumerate(raw_slots)
    )
    display_order, display_digest = _display_order(seed, attempt, member)
    unsigned: dict[str, Any] = {
        "member_position": member.member_position,
        "composed_rule_id": member.composed_rule_id,
        "composed_truth_digest": member.composed_truth_digest,
        "triple_digest": member.triple_digest,
        "noisy_p_target_side": member.noisy_p_target_side,
        "schedule_variants": list(variants),
        "union_cell_counts": list(union),
        "planned_union_slots": [slot.as_obj() for slot in slots],
        "planned_union_slots_digest": _digest([slot.as_obj() for slot in slots], domain=_SLOT_SET_DOMAIN),
        "planned_geometry_display_order": list(display_order),
        "planned_geometry_display_order_digest": display_digest,
    }
    return NestedOpeningMemberPlanV1(
        member.member_position,
        member.composed_rule_id,
        member.composed_truth_digest,
        member.triple_digest,
        member.noisy_p_target_side,
        variants,
        union,
        COMMON_UNION_SLOT_COUNT,
        slots,
        cast(str, unsigned["planned_union_slots_digest"]),
        display_order,
        display_digest,
        _digest(unsigned, domain=_MEMBER_PLAN_DOMAIN),
    )


def _build_attempt_plan(
    seed: str,
    attempt: CensusAttemptPlanV1,
) -> NestedOpeningAttemptPlanV1:
    members = cast(
        tuple[NestedOpeningMemberPlanV1, NestedOpeningMemberPlanV1],
        tuple(_build_member_plan(seed, attempt, member) for member in attempt.members),
    )
    unsigned = {
        "formula_stratum": attempt.formula_stratum,
        "canonical_position": attempt.canonical_position,
        "upstream_attempt_digest": attempt.attempt_digest,
        "placard_rule_id": attempt.placard_rule_id,
        "placard_truth_digest": attempt.placard_truth_digest,
        "literal_rule_id": attempt.literal_rule_id,
        "literal_truth_digest": attempt.literal_truth_digest,
        "members": [member.as_obj() for member in members],
    }
    return NestedOpeningAttemptPlanV1(
        attempt.formula_stratum,
        attempt.canonical_position,
        attempt.attempt_digest,
        attempt.placard_rule_id,
        attempt.placard_truth_digest,
        attempt.literal_rule_id,
        attempt.literal_truth_digest,
        members,
        _digest(unsigned, domain=_ATTEMPT_PLAN_DOMAIN),
    )


def _build_plan(
    census_plan_text: str,
    *,
    expected_census_plan_digest: str,
    expected_census_plan_bytes_sha256: str,
    generator_seed: str,
    for_testing: bool,
) -> NestedOpeningFeasibilityPlanV1:
    _require_sha256(generator_seed, name="generator seed")
    parent = _parse_exact_upstream(
        census_plan_text,
        expected_census_plan_digest=expected_census_plan_digest,
        expected_census_plan_bytes_sha256=expected_census_plan_bytes_sha256,
    )
    production_parent = (
        parent.attempts_per_formula_stratum == PRODUCTION_MIRROR_ATTEMPTS_PER_STRATUM
        and len(parent.attempts) == PRODUCTION_MIRROR_ATTEMPT_COUNT
        and parent.uses_production_attempt_budget
    )
    if for_testing:
        if production_parent:
            raise NestedOpeningFeasibilityV2Error("test-only plan helper refuses a production 144x2 parent")
    elif not production_parent:
        raise NestedOpeningFeasibilityV2Error(
            "production feasibility plan requires the exact 144x2 upstream budget"
        )
    source = _current_source_binding()
    generator = NestedGeneratorBindingV1(
        generator_seed,
        _digest(_generator_contract_obj(), domain=_GENERATOR_CONTRACT_DOMAIN),
        source.nested_opening_feasibility_source_sha256,
    )
    upstream = _upstream_binding(parent, census_plan_text)
    attempts = tuple(_build_attempt_plan(generator_seed, attempt) for attempt in parent.attempts)
    return NestedOpeningFeasibilityPlanV1(
        source,
        upstream,
        generator,
        parent.attempts_per_formula_stratum,
        attempts,
        for_testing,
    )


def build_nested_opening_construction_feasibility_plan_v1(
    census_plan_text: str,
    *,
    expected_census_plan_digest: str,
    expected_census_plan_bytes_sha256: str,
    generator_seed: str,
) -> NestedOpeningFeasibilityPlanV1:
    """Build a source-only plan from the exact 144-attempt census identity schedule."""

    return _build_plan(
        census_plan_text,
        expected_census_plan_digest=expected_census_plan_digest,
        expected_census_plan_bytes_sha256=expected_census_plan_bytes_sha256,
        generator_seed=generator_seed,
        for_testing=False,
    )


def build_nested_opening_construction_feasibility_plan_for_testing_v1(
    census_plan_text: str,
    *,
    expected_census_plan_digest: str,
    expected_census_plan_bytes_sha256: str,
    generator_seed: str,
) -> NestedOpeningFeasibilityPlanV1:
    """Build a nominally test-only reduced plan that cannot be promoted."""

    return _build_plan(
        census_plan_text,
        expected_census_plan_digest=expected_census_plan_digest,
        expected_census_plan_bytes_sha256=expected_census_plan_bytes_sha256,
        generator_seed=generator_seed,
        for_testing=True,
    )


def _derive_plan_digest(
    plan: NestedOpeningFeasibilityPlanV1,
    census_plan_text: str,
    *,
    expected_census_plan_digest: str,
    expected_census_plan_bytes_sha256: str,
    for_testing: bool,
) -> str:
    if type(plan) is not NestedOpeningFeasibilityPlanV1:
        raise TypeError("plan must be a NestedOpeningFeasibilityPlanV1")
    expected = _build_plan(
        census_plan_text,
        expected_census_plan_digest=expected_census_plan_digest,
        expected_census_plan_bytes_sha256=expected_census_plan_bytes_sha256,
        generator_seed=plan.generator_binding.seed,
        for_testing=for_testing,
    )
    if plan != expected:
        raise NestedOpeningFeasibilityV2Error("plan differs from exact parent-bound replay")
    return _plan_digest_unverified(expected)


def derive_nested_opening_construction_feasibility_plan_digest_v1(
    plan: NestedOpeningFeasibilityPlanV1,
    census_plan_text: str,
    *,
    expected_census_plan_digest: str,
    expected_census_plan_bytes_sha256: str,
) -> str:
    """Return a production plan digest only after exact parent replay."""

    if plan.engineering_budget_override:
        raise NestedOpeningFeasibilityV2Error("production digest derivation refuses a test-only reduced plan")
    return _derive_plan_digest(
        plan,
        census_plan_text,
        expected_census_plan_digest=expected_census_plan_digest,
        expected_census_plan_bytes_sha256=expected_census_plan_bytes_sha256,
        for_testing=False,
    )


def derive_nested_opening_construction_feasibility_plan_digest_for_testing_v1(
    plan: NestedOpeningFeasibilityPlanV1,
    census_plan_text: str,
    *,
    expected_census_plan_digest: str,
    expected_census_plan_bytes_sha256: str,
) -> str:
    """Return a reduced-plan digest only after exact test-parent replay."""

    if not plan.engineering_budget_override:
        raise NestedOpeningFeasibilityV2Error("test-only digest derivation refuses a production plan")
    return _derive_plan_digest(
        plan,
        census_plan_text,
        expected_census_plan_digest=expected_census_plan_digest,
        expected_census_plan_bytes_sha256=expected_census_plan_bytes_sha256,
        for_testing=True,
    )


def serialize_nested_opening_construction_feasibility_plan_v1(
    plan: NestedOpeningFeasibilityPlanV1,
    census_plan_text: str,
    *,
    expected_census_plan_digest: str,
    expected_census_plan_bytes_sha256: str,
    expected_plan_digest: str,
) -> str:
    """Serialize a production plan only after exact parent replay."""

    if type(plan) is not NestedOpeningFeasibilityPlanV1:
        raise TypeError("plan must be a NestedOpeningFeasibilityPlanV1")
    if plan.engineering_budget_override:
        raise NestedOpeningFeasibilityV2Error("production serializer refuses a test-only reduced plan")
    expected = _build_plan(
        census_plan_text,
        expected_census_plan_digest=expected_census_plan_digest,
        expected_census_plan_bytes_sha256=expected_census_plan_bytes_sha256,
        generator_seed=plan.generator_binding.seed,
        for_testing=False,
    )
    _require_sha256(expected_plan_digest, name="expected nested plan digest")
    if _plan_digest_unverified(expected) != expected_plan_digest or plan != expected:
        raise NestedOpeningFeasibilityV2Error("plan differs from exact parent-bound replay")
    return _serialize_plan_after_replay(expected, for_testing=False)


def serialize_nested_opening_construction_feasibility_plan_for_testing_v1(
    plan: NestedOpeningFeasibilityPlanV1,
    census_plan_text: str,
    *,
    expected_census_plan_digest: str,
    expected_census_plan_bytes_sha256: str,
    expected_plan_digest: str,
) -> str:
    """Serialize a reduced plan only after exact engineering-parent replay."""

    if type(plan) is not NestedOpeningFeasibilityPlanV1:
        raise TypeError("plan must be a NestedOpeningFeasibilityPlanV1")
    if not plan.engineering_budget_override:
        raise NestedOpeningFeasibilityV2Error("test-only serializer refuses a production plan")
    expected = _build_plan(
        census_plan_text,
        expected_census_plan_digest=expected_census_plan_digest,
        expected_census_plan_bytes_sha256=expected_census_plan_bytes_sha256,
        generator_seed=plan.generator_binding.seed,
        for_testing=True,
    )
    _require_sha256(expected_plan_digest, name="expected nested plan digest")
    if _plan_digest_unverified(expected) != expected_plan_digest or plan != expected:
        raise NestedOpeningFeasibilityV2Error("test-only plan differs from exact parent-bound replay")
    return _serialize_plan_after_replay(expected, for_testing=True)


def _serialize_plan_after_replay(
    plan: NestedOpeningFeasibilityPlanV1,
    *,
    for_testing: bool,
) -> str:
    """Private projection used only after exact parent replay."""

    if plan.engineering_budget_override != for_testing:
        raise NestedOpeningFeasibilityV2Error("plan kind differs from serializer boundary")
    return _dump_json(plan._artifact_obj_unverified()) + "\n"


def _parse_plan(
    text: str,
    *,
    census_plan_text: str,
    expected_census_plan_digest: str,
    expected_census_plan_bytes_sha256: str,
    expected_plan_digest: str,
    for_testing: bool,
) -> NestedOpeningFeasibilityPlanV1:
    _require_sha256(expected_plan_digest, name="expected nested plan digest")
    obj = _require_mapping(
        _load_json(text),
        (
            "schema_version",
            "plan_kind",
            "status",
            "protocol_quartet_candidate",
            "authorization",
            "source_binding",
            "upstream_identity_plan_binding",
            "generator_binding",
            "fixed_budget",
            "selection_boundary",
            "attempts",
            "prospective_plan_digest",
        ),
        name="nested-opening feasibility plan",
    )
    expected_kind = _TEST_PLAN_KIND if for_testing else _PLAN_KIND
    expected_status = _TEST_PLAN_STATUS if for_testing else _PLAN_STATUS
    if (
        obj["schema_version"] != NESTED_OPENING_FEASIBILITY_SCHEMA_VERSION
        or obj["plan_kind"] != expected_kind
        or obj["status"] != expected_status
        or obj["protocol_quartet_candidate"] is not False
        or obj["authorization"] != _AUTHORIZATION
        or obj["selection_boundary"] != _SELECTION_BOUNDARY
    ):
        raise NestedOpeningFeasibilityV2Error("plan kind, status, or false/null boundary changed")
    generator = _require_mapping(
        obj["generator_binding"],
        (
            "generator_schema_version",
            "generator_seed",
            "generator_contract",
            "generator_contract_digest",
            "generator_source_sha256",
        ),
        name="generator binding",
    )
    seed = _require_sha256(generator["generator_seed"], name="generator seed")
    rebuilt = _build_plan(
        census_plan_text,
        expected_census_plan_digest=expected_census_plan_digest,
        expected_census_plan_bytes_sha256=expected_census_plan_bytes_sha256,
        generator_seed=seed,
        for_testing=for_testing,
    )
    rebuilt_digest = _plan_digest_unverified(rebuilt)
    if rebuilt_digest != expected_plan_digest or obj["prospective_plan_digest"] != rebuilt_digest:
        raise NestedOpeningFeasibilityV2Error("nested plan differs from externally expected digest")
    if _dump_json(obj) != _dump_json(rebuilt._artifact_obj_unverified()):
        raise NestedOpeningFeasibilityV2Error("nested plan differs from exact upstream replay")
    canonical = _serialize_plan_after_replay(rebuilt, for_testing=for_testing)
    if canonical != text:
        raise NestedOpeningFeasibilityV2Error("nested plan is not canonical newline-terminated JSON")
    return rebuilt


def parse_nested_opening_construction_feasibility_plan_v1(
    text: str,
    *,
    census_plan_text: str,
    expected_census_plan_digest: str,
    expected_census_plan_bytes_sha256: str,
    expected_plan_digest: str,
) -> NestedOpeningFeasibilityPlanV1:
    """Parse only an exact production-kind feasibility plan."""

    return _parse_plan(
        text,
        census_plan_text=census_plan_text,
        expected_census_plan_digest=expected_census_plan_digest,
        expected_census_plan_bytes_sha256=expected_census_plan_bytes_sha256,
        expected_plan_digest=expected_plan_digest,
        for_testing=False,
    )


def parse_nested_opening_construction_feasibility_plan_for_testing_v1(
    text: str,
    *,
    census_plan_text: str,
    expected_census_plan_digest: str,
    expected_census_plan_bytes_sha256: str,
    expected_plan_digest: str,
) -> NestedOpeningFeasibilityPlanV1:
    """Parse only the distinct reduced engineering-fixture plan kind."""

    return _parse_plan(
        text,
        census_plan_text=census_plan_text,
        expected_census_plan_digest=expected_census_plan_digest,
        expected_census_plan_bytes_sha256=expected_census_plan_bytes_sha256,
        expected_plan_digest=expected_plan_digest,
        for_testing=True,
    )


SelectionOutcome = Literal["selected", "bounded_window_exhausted"]
MemberConstructionStatus = Literal["completed", "bounded_window_failure"]


@dataclass(frozen=True, slots=True)
class NestedOpeningSelectionStepV1:
    slot_position: int
    prefix_rank: int
    joint_cell_index: int
    joint_cell_slug: str
    affected_geometry_slugs: tuple[str, ...]
    survivor_counts_before: tuple[int, int, int, int]
    available_candidate_count: int
    candidate_window_count: int
    candidate_window_digest: str
    examined_candidate_count: int
    outcome: SelectionOutcome
    selected_candidate_rank: int | None
    selected_scene_index: int | None
    survivor_counts_after: tuple[int, int, int, int] | None
    selection_step_digest: str

    def __post_init__(self) -> None:
        _require_integer(self.slot_position, name="step slot position", maximum=12)
        _require_integer(self.prefix_rank, name="step prefix rank", maximum=4)
        _require_integer(self.joint_cell_index, name="step cell index", maximum=7)
        if self.joint_cell_slug != f"{self.joint_cell_index:03b}":
            raise NestedOpeningFeasibilityV2Error("step cell slug is inconsistent")
        if (
            type(self.affected_geometry_slugs) is not tuple
            or self.affected_geometry_slugs
            != tuple(slug for slug in _GEOMETRY_SLUGS if slug in self.affected_geometry_slugs)
            or not self.affected_geometry_slugs
        ):
            raise NestedOpeningFeasibilityV2Error("step affected geometries are noncanonical")
        if type(self.survivor_counts_before) is not tuple or len(self.survivor_counts_before) != 4:
            raise NestedOpeningFeasibilityV2Error("step needs four pre-step survivor counts")
        for count in self.survivor_counts_before:
            _require_integer(count, name="pre-step survivor count", minimum=8)
        _require_integer(self.available_candidate_count, name="available candidate count")
        _require_integer(
            self.candidate_window_count,
            name="candidate window count",
            maximum=CANDIDATE_WINDOW_K,
        )
        if self.candidate_window_count != min(CANDIDATE_WINDOW_K, self.available_candidate_count):
            raise NestedOpeningFeasibilityV2Error("candidate window is not the fixed K prefix")
        _require_sha256(self.candidate_window_digest, name="candidate window digest")
        _require_integer(
            self.examined_candidate_count,
            name="examined candidate count",
            maximum=self.candidate_window_count,
        )
        if self.outcome not in ("selected", "bounded_window_exhausted"):
            raise NestedOpeningFeasibilityV2Error("unknown selection-step outcome")
        if self.outcome == "selected":
            rank = _require_integer(
                self.selected_candidate_rank,
                name="selected candidate rank",
                minimum=1,
                maximum=self.candidate_window_count,
            )
            _require_integer(
                self.selected_scene_index,
                name="selected scene index",
                maximum=SCENE_COUNT - 1,
            )
            if rank != self.examined_candidate_count:
                raise NestedOpeningFeasibilityV2Error(
                    "selected rank must equal the first passing examined candidate"
                )
            if type(self.survivor_counts_after) is not tuple or len(self.survivor_counts_after) != 4:
                raise NestedOpeningFeasibilityV2Error("selected step needs four post-step survivor counts")
            for count in self.survivor_counts_after:
                _require_integer(count, name="post-step survivor count", minimum=8)
        elif (
            self.selected_candidate_rank is not None
            or self.selected_scene_index is not None
            or self.survivor_counts_after is not None
            or self.examined_candidate_count != self.candidate_window_count
        ):
            raise NestedOpeningFeasibilityV2Error(
                "bounded-window failure must preserve exact null terminal fields"
            )
        _require_sha256(self.selection_step_digest, name="selection-step digest")
        if self.selection_step_digest != _digest(self._unsigned_obj(), domain=_SELECTION_STEP_DOMAIN):
            raise NestedOpeningFeasibilityV2Error("selection-step digest is inconsistent")

    def _unsigned_obj(self) -> dict[str, Any]:
        return {
            "slot_position": self.slot_position,
            "cell_prefix_rank": self.prefix_rank,
            "joint_cell_index": self.joint_cell_index,
            "joint_cell_slug": self.joint_cell_slug,
            "affected_geometry_slugs": list(self.affected_geometry_slugs),
            "survivor_counts_before": list(self.survivor_counts_before),
            "available_remaining_scene_count": self.available_candidate_count,
            "candidate_window_count": self.candidate_window_count,
            "candidate_window_digest": self.candidate_window_digest,
            "candidate_count_examined": self.examined_candidate_count,
            "outcome": self.outcome,
            "selected_candidate_window_rank": self.selected_candidate_rank,
            "selected_scene_index": self.selected_scene_index,
            "survivor_counts_after": (
                None if self.survivor_counts_after is None else list(self.survivor_counts_after)
            ),
        }

    def as_obj(self) -> dict[str, Any]:
        return {**self._unsigned_obj(), "selection_step_digest": self.selection_step_digest}


@dataclass(frozen=True, slots=True)
class NestedDerivedOpeningV1:
    geometry_slug: str
    schedule_variant: ScheduleVariant
    noisy_p_target_side: int | None
    opening_scene_indices_by_joint_cell: tuple[tuple[int, ...], ...]
    construction_scene_indices_cell_major: tuple[int, ...]
    assessment: CensusConstructedOpeningV1
    ranking_table_digest: str
    derived_opening_digest: str

    def __post_init__(self) -> None:
        if self.geometry_slug not in _GEOMETRY_SLUGS:
            raise NestedOpeningFeasibilityV2Error("derived opening has unknown geometry")
        if (
            type(self.opening_scene_indices_by_joint_cell) is not tuple
            or len(self.opening_scene_indices_by_joint_cell) != 8
        ):
            raise NestedOpeningFeasibilityV2Error("derived opening needs eight cell prefixes")
        if any(type(cell) is not tuple for cell in self.opening_scene_indices_by_joint_cell):
            raise NestedOpeningFeasibilityV2Error("derived opening cell prefixes must be tuples")
        flat = tuple(scene for cell in self.opening_scene_indices_by_joint_cell for scene in cell)
        if flat != self.construction_scene_indices_cell_major or len(flat) != 10:
            raise NestedOpeningFeasibilityV2Error(
                "derived opening is not its exact ten-scene cell-major flattening"
            )
        if len(set(flat)) != 10:
            raise NestedOpeningFeasibilityV2Error("derived opening repeats a scene")
        expected_counts = _variant_schedule(self.schedule_variant)
        if tuple(map(len, self.opening_scene_indices_by_joint_cell)) != expected_counts:
            raise NestedOpeningFeasibilityV2Error(
                "derived opening does not use the geometry's canonical cell prefixes"
            )
        expected_side = int(self.schedule_variant[-1]) if "_y" in self.schedule_variant else None
        if self.noisy_p_target_side != expected_side:
            raise NestedOpeningFeasibilityV2Error("derived opening side is inconsistent")
        if type(self.assessment) is not CensusConstructedOpeningV1:
            raise NestedOpeningFeasibilityV2Error(
                "derived opening requires the exact public census assessment type"
            )
        if (
            self.assessment.schedule_variant != self.schedule_variant
            or self.assessment.construction_scene_indices != flat
        ):
            raise NestedOpeningFeasibilityV2Error("derived opening and public assessment disagree")
        _require_sha256(self.ranking_table_digest, name="ranking-table digest")
        if self.assessment.ranking_table_digest != self.ranking_table_digest:
            raise NestedOpeningFeasibilityV2Error("derived opening ranking reference is inconsistent")
        _require_sha256(self.derived_opening_digest, name="derived-opening digest")
        if self.derived_opening_digest != _digest(self._unsigned_obj(), domain=_DERIVED_OPENING_DOMAIN):
            raise NestedOpeningFeasibilityV2Error("derived-opening digest is inconsistent")

    def _unsigned_obj(self) -> dict[str, Any]:
        return {
            "geometry_slug": self.geometry_slug,
            "schedule_variant": self.schedule_variant,
            "a_error_target_side": self.noisy_p_target_side,
            "opening_scene_ids_by_joint_cell": [
                list(cell) for cell in self.opening_scene_indices_by_joint_cell
            ],
            "opening_scene_indices_cell_major": list(self.construction_scene_indices_cell_major),
            "assessment": self.assessment.as_obj(),
        }

    def as_obj(self) -> dict[str, Any]:
        return {**self._unsigned_obj(), "derived_opening_digest": self.derived_opening_digest}


@dataclass(frozen=True, slots=True)
class NestedExactMatchRowV1:
    geometry_slug: str
    observed_m: int
    exact_minimax_depth: int | None
    target_formula_stratum: str
    greedy_reference_query_count: int | None
    best_first_query_branch_sizes: tuple[int, int] | None

    def __post_init__(self) -> None:
        if self.geometry_slug not in _GEOMETRY_SLUGS:
            raise NestedOpeningFeasibilityV2Error("exact-match row has unknown geometry")
        _require_integer(self.observed_m, name="observed m", minimum=1)
        if self.exact_minimax_depth is not None:
            _require_integer(self.exact_minimax_depth, name="exact minimax depth", minimum=1)
        if self.greedy_reference_query_count is not None:
            _require_integer(
                self.greedy_reference_query_count,
                name="greedy reference query count",
                minimum=1,
            )
        if self.best_first_query_branch_sizes is not None:
            if (
                type(self.best_first_query_branch_sizes) is not tuple
                or len(self.best_first_query_branch_sizes) != 2
            ):
                raise NestedOpeningFeasibilityV2Error("best first-query split must contain two branches")
            for size in self.best_first_query_branch_sizes:
                _require_integer(size, name="first-query branch size", minimum=1)

    @property
    def matching_key(self) -> tuple[Any, ...]:
        return (
            self.observed_m,
            self.exact_minimax_depth,
            self.target_formula_stratum,
            self.greedy_reference_query_count,
            self.best_first_query_branch_sizes,
        )

    def as_obj(self) -> dict[str, Any]:
        return {
            "geometry_slug": self.geometry_slug,
            "version_space_size": self.observed_m,
            "minimax_depth": self.exact_minimax_depth,
            "target_formula_stratum": self.target_formula_stratum,
            "greedy_reference_query_count": self.greedy_reference_query_count,
            "best_first_query_branch_size_pair": (
                None
                if self.best_first_query_branch_sizes is None
                else list(self.best_first_query_branch_sizes)
            ),
        }


_EXACT_MATCH_FIELDS = (
    "version_space_size",
    "minimax_depth",
    "target_formula_stratum",
    "greedy_reference_query_count",
    "best_first_query_branch_size_pair",
)


@dataclass(frozen=True, slots=True)
class NestedExactMatchEvidenceV1:
    rows: tuple[
        NestedExactMatchRowV1,
        NestedExactMatchRowV1,
        NestedExactMatchRowV1,
        NestedExactMatchRowV1,
    ]
    common_key: dict[str, Any] | None
    passed: bool
    exact_match_digest: str

    def __post_init__(self) -> None:
        if (
            type(self.rows) is not tuple
            or len(self.rows) != 4
            or any(type(row) is not NestedExactMatchRowV1 for row in self.rows)
            or tuple(row.geometry_slug for row in self.rows) != _GEOMETRY_SLUGS
        ):
            raise NestedOpeningFeasibilityV2Error("exact-match rows must cover four canonical geometries")
        _require_bool(self.passed, name="exact-match passed")
        observed_pass = len({row.matching_key for row in self.rows}) == 1
        if self.passed != observed_pass:
            raise NestedOpeningFeasibilityV2Error("exact-match decision is inconsistent")
        expected_common = _matching_key_obj(self.rows[0]) if observed_pass else None
        if self.common_key != expected_common:
            raise NestedOpeningFeasibilityV2Error("exact-match common key is inconsistent")
        _require_sha256(self.exact_match_digest, name="exact-match digest")
        if self.exact_match_digest != _digest(self._unsigned_obj(), domain=_EXACT_MATCH_DOMAIN):
            raise NestedOpeningFeasibilityV2Error("exact-match digest is inconsistent")

    def _unsigned_obj(self) -> dict[str, Any]:
        return {
            "required_fields": list(_EXACT_MATCH_FIELDS),
            "per_geometry_rows": [row.as_obj() for row in self.rows],
            "common_key": self.common_key,
            "passed": self.passed,
        }

    def as_obj(self) -> dict[str, Any]:
        return {**self._unsigned_obj(), "exact_match_digest": self.exact_match_digest}


def _matching_key_obj(row: NestedExactMatchRowV1) -> dict[str, Any]:
    return {
        "version_space_size": row.observed_m,
        "minimax_depth": row.exact_minimax_depth,
        "target_formula_stratum": row.target_formula_stratum,
        "greedy_reference_query_count": row.greedy_reference_query_count,
        "best_first_query_branch_size_pair": (
            None if row.best_first_query_branch_sizes is None else list(row.best_first_query_branch_sizes)
        ),
    }


@dataclass(frozen=True, slots=True)
class NestedOpeningMemberObservationV1:
    member_plan: NestedOpeningMemberPlanV1
    exact_member_plan_binding: dict[str, str]
    status: MemberConstructionStatus
    selection_steps: tuple[NestedOpeningSelectionStepV1, ...]
    selection_trace_digest: str
    common_union_scene_indices_by_joint_cell: tuple[tuple[int, ...], ...] | None
    common_union_digest: str | None
    remaining_scene_count_by_joint_cell: tuple[int, ...] | None
    openings: (
        tuple[
            NestedDerivedOpeningV1,
            NestedDerivedOpeningV1,
            NestedDerivedOpeningV1,
            NestedDerivedOpeningV1,
        ]
        | None
    )
    exact_match: NestedExactMatchEvidenceV1 | None
    reason_codes: tuple[str, ...]
    member_feasibility_witness: bool
    protocol_quartet_candidate: bool
    member_observation_digest: str

    def __post_init__(self) -> None:
        if type(self.member_plan) is not NestedOpeningMemberPlanV1:
            raise NestedOpeningFeasibilityV2Error("member observation needs an exact member plan")
        if self.exact_member_plan_binding != {"member_plan_digest": self.member_plan.member_plan_digest}:
            raise NestedOpeningFeasibilityV2Error("member-plan binding is inconsistent")
        if type(self.selection_steps) is not tuple or any(
            type(step) is not NestedOpeningSelectionStepV1 for step in self.selection_steps
        ):
            raise NestedOpeningFeasibilityV2Error("member trace has nonexact step objects")
        if tuple(step.slot_position for step in self.selection_steps) != tuple(
            range(len(self.selection_steps))
        ):
            raise NestedOpeningFeasibilityV2Error("member trace steps are reordered")
        _require_sha256(self.selection_trace_digest, name="selection-trace digest")
        if self.selection_trace_digest != _digest(
            [step.as_obj() for step in self.selection_steps], domain=_SELECTION_TRACE_DOMAIN
        ):
            raise NestedOpeningFeasibilityV2Error("selection-trace digest is inconsistent")
        _require_bool(self.member_feasibility_witness, name="member feasibility witness")
        if self.protocol_quartet_candidate is not False:
            raise NestedOpeningFeasibilityV2Error(
                "member observation can never be a protocol quartet candidate"
            )
        if self.reason_codes != tuple(sorted(set(self.reason_codes))):
            raise NestedOpeningFeasibilityV2Error("member reason codes are noncanonical")
        if self.status == "bounded_window_failure":
            if (
                not self.selection_steps
                or self.selection_steps[-1].outcome != "bounded_window_exhausted"
                or self.common_union_scene_indices_by_joint_cell is not None
                or self.common_union_digest is not None
                or self.remaining_scene_count_by_joint_cell is not None
                or self.openings is not None
                or self.exact_match is not None
                or self.member_feasibility_witness
            ):
                raise NestedOpeningFeasibilityV2Error(
                    "bounded failure must preserve partial trace and exact null outputs"
                )
        elif self.status == "completed":
            if len(self.selection_steps) != COMMON_UNION_SLOT_COUNT or any(
                step.outcome != "selected" for step in self.selection_steps
            ):
                raise NestedOpeningFeasibilityV2Error(
                    "completed member must preserve thirteen selected steps"
                )
            if (
                type(self.common_union_scene_indices_by_joint_cell) is not tuple
                or len(self.common_union_scene_indices_by_joint_cell) != 8
                or any(type(cell) is not tuple for cell in self.common_union_scene_indices_by_joint_cell)
                or tuple(map(len, self.common_union_scene_indices_by_joint_cell))
                != self.member_plan.union_joint_truth_cell_counts
            ):
                raise NestedOpeningFeasibilityV2Error("completed common union is noncanonical")
            _require_sha256(self.common_union_digest, name="common-union digest")
            if self.common_union_digest != _digest(
                [list(cell) for cell in self.common_union_scene_indices_by_joint_cell],
                domain=_COMMON_UNION_DOMAIN,
            ):
                raise NestedOpeningFeasibilityV2Error("common-union digest is inconsistent")
            if (
                type(self.remaining_scene_count_by_joint_cell) is not tuple
                or len(self.remaining_scene_count_by_joint_cell) != 8
            ):
                raise NestedOpeningFeasibilityV2Error("completed member needs eight remaining-cell counts")
            for count in self.remaining_scene_count_by_joint_cell:
                _require_integer(count, name="remaining scene count")
            if (
                type(self.openings) is not tuple
                or len(self.openings) != 4
                or any(type(opening) is not NestedDerivedOpeningV1 for opening in self.openings)
                or tuple(opening.geometry_slug for opening in self.openings) != _GEOMETRY_SLUGS
                or type(self.exact_match) is not NestedExactMatchEvidenceV1
            ):
                raise NestedOpeningFeasibilityV2Error(
                    "completed member needs four assessed openings and exact-match evidence"
                )
            expected_witness = self.exact_match.passed and all(
                opening.assessment.disposition == "cell_witness" for opening in self.openings
            )
            if self.member_feasibility_witness != expected_witness:
                raise NestedOpeningFeasibilityV2Error("member feasibility witness is inconsistent")
        else:
            raise NestedOpeningFeasibilityV2Error("unknown member construction status")
        _require_sha256(self.member_observation_digest, name="member-observation digest")
        if self.member_observation_digest != _digest(self._unsigned_obj(), domain=_MEMBER_OBSERVATION_DOMAIN):
            raise NestedOpeningFeasibilityV2Error("member-observation digest is inconsistent")

    def _unsigned_obj(self) -> dict[str, Any]:
        return {
            "member_plan": self.member_plan.as_obj(),
            "binding": self.exact_member_plan_binding,
            "construction_status": self.status,
            "selection_steps": [step.as_obj() for step in self.selection_steps],
            "selection_trace_digest": self.selection_trace_digest,
            "opening_union_scene_ids_by_joint_cell": (
                None
                if self.common_union_scene_indices_by_joint_cell is None
                else [list(cell) for cell in self.common_union_scene_indices_by_joint_cell]
            ),
            "opening_union_digest": self.common_union_digest,
            "remaining_scene_count_by_joint_cell": (
                None
                if self.remaining_scene_count_by_joint_cell is None
                else list(self.remaining_scene_count_by_joint_cell)
            ),
            "openings": (None if self.openings is None else [opening.as_obj() for opening in self.openings]),
            "exact_match_evidence": None if self.exact_match is None else self.exact_match.as_obj(),
            "reason_codes": list(self.reason_codes),
            "nested_opening_construction_feasibility_witness": self.member_feasibility_witness,
            "protocol_quartet_candidate": self.protocol_quartet_candidate,
        }

    def as_obj(self) -> dict[str, Any]:
        return {
            **self._unsigned_obj(),
            "member_observation_digest": self.member_observation_digest,
        }


@dataclass(frozen=True, slots=True)
class NestedOpeningAttemptObservationV1:
    attempt_plan: NestedOpeningAttemptPlanV1
    members: tuple[
        NestedOpeningMemberObservationV1,
        NestedOpeningMemberObservationV1,
    ]
    mirror_overlap_scene_indices: tuple[int, ...]
    mirror_overlap_digest: str
    mirror_pair_disjoint: bool
    mirror_pair_feasibility_witness: bool
    reason_codes: tuple[str, ...]
    protocol_quartet_candidate: bool
    attempt_observation_digest: str

    def __post_init__(self) -> None:
        if type(self.attempt_plan) is not NestedOpeningAttemptPlanV1:
            raise NestedOpeningFeasibilityV2Error("attempt observation needs an exact plan")
        if (
            type(self.members) is not tuple
            or len(self.members) != 2
            or any(type(member) is not NestedOpeningMemberObservationV1 for member in self.members)
            or tuple(member.member_plan for member in self.members) != self.attempt_plan.members
        ):
            raise NestedOpeningFeasibilityV2Error("attempt observations are reordered")
        if self.mirror_overlap_scene_indices != tuple(sorted(set(self.mirror_overlap_scene_indices))):
            raise NestedOpeningFeasibilityV2Error("mirror overlap must be sorted and unique")
        both_complete = all(member.status == "completed" for member in self.members)
        expected_disjoint = both_complete and not self.mirror_overlap_scene_indices
        _require_bool(self.mirror_pair_disjoint, name="mirror pair disjoint")
        if self.mirror_pair_disjoint != expected_disjoint:
            raise NestedOpeningFeasibilityV2Error("mirror-pair disjoint flag is inconsistent")
        _require_sha256(self.mirror_overlap_digest, name="mirror-overlap digest")
        overlap_obj = {
            "both_common_unions_complete": both_complete,
            "member_union_overlap_scene_indices": list(self.mirror_overlap_scene_indices),
        }
        if self.mirror_overlap_digest != _digest(overlap_obj, domain=_MIRROR_OVERLAP_DOMAIN):
            raise NestedOpeningFeasibilityV2Error("mirror-overlap digest is inconsistent")
        expected_pair = _attempt_pair_witness(self.members, self.mirror_pair_disjoint)
        _require_bool(
            self.mirror_pair_feasibility_witness,
            name="mirror pair feasibility witness",
        )
        if self.mirror_pair_feasibility_witness != expected_pair:
            raise NestedOpeningFeasibilityV2Error("mirror-pair witness is inconsistent")
        if self.reason_codes != tuple(sorted(set(self.reason_codes))):
            raise NestedOpeningFeasibilityV2Error("attempt reason codes are noncanonical")
        if self.protocol_quartet_candidate is not False:
            raise NestedOpeningFeasibilityV2Error(
                "attempt observation can never be a protocol quartet candidate"
            )
        _require_sha256(self.attempt_observation_digest, name="attempt-observation digest")
        if self.attempt_observation_digest != _digest(
            self._unsigned_obj(), domain=_ATTEMPT_OBSERVATION_DOMAIN
        ):
            raise NestedOpeningFeasibilityV2Error("attempt-observation digest is inconsistent")

    def _unsigned_obj(self) -> dict[str, Any]:
        return {
            "attempt_plan": self.attempt_plan.as_obj(),
            "members": [member.as_obj() for member in self.members],
            "member_union_overlap_scene_indices": list(self.mirror_overlap_scene_indices),
            "member_union_overlap_digest": self.mirror_overlap_digest,
            "mirror_pair_disjoint": self.mirror_pair_disjoint,
            "mirror_pair_nested_opening_feasibility_witness": self.mirror_pair_feasibility_witness,
            "reason_codes": list(self.reason_codes),
            "protocol_quartet_candidate": self.protocol_quartet_candidate,
        }

    def as_obj(self) -> dict[str, Any]:
        return {
            **self._unsigned_obj(),
            "attempt_observation_digest": self.attempt_observation_digest,
        }


def _attempt_pair_witness(
    members: tuple[NestedOpeningMemberObservationV1, NestedOpeningMemberObservationV1],
    mirror_pair_disjoint: bool,
) -> bool:
    if not mirror_pair_disjoint or not all(member.member_feasibility_witness for member in members):
        return False
    exact = cast(
        tuple[NestedExactMatchEvidenceV1, NestedExactMatchEvidenceV1],
        tuple(member.exact_match for member in members),
    )
    return exact[0].common_key == exact[1].common_key


@dataclass(frozen=True, slots=True)
class NestedMQSummaryRowV1:
    m: int
    q: int
    positive_quota: None
    completed_opening_count: int
    salient_opening_count: int
    individual_feasibility_witness_count: int
    member_feasibility_witness_count: int
    mirror_pair_feasibility_witness_count: int

    def __post_init__(self) -> None:
        _require_integer(self.m, name="summary m", minimum=8, maximum=16)
        _require_integer(self.q, name="summary q", minimum=1, maximum=4)
        if self.positive_quota is not None:
            raise NestedOpeningFeasibilityV2Error("positive m/q quota must remain null")
        for name in (
            "completed_opening_count",
            "salient_opening_count",
            "individual_feasibility_witness_count",
            "member_feasibility_witness_count",
            "mirror_pair_feasibility_witness_count",
        ):
            _require_integer(getattr(self, name), name=name)

    def as_obj(self) -> dict[str, Any]:
        return {
            "m": self.m,
            "q": self.q,
            "positive_quota": self.positive_quota,
            "completed_opening_count": self.completed_opening_count,
            "salient_opening_count": self.salient_opening_count,
            "individual_cell_witness_count": self.individual_feasibility_witness_count,
            "nested_opening_feasibility_witness_count": self.member_feasibility_witness_count,
            "mirror_pair_feasibility_witness_count": self.mirror_pair_feasibility_witness_count,
        }


@dataclass(frozen=True)
class NestedOpeningFeasibilityReportV1:
    source_binding: NestedSourceBindingV1
    prospective_plan_digest: str
    canonical_plan_bytes_sha256: str
    canonical_plan_byte_count: int
    upstream_census_plan_digest: str
    upstream_census_plan_bytes_sha256: str
    generator_contract_digest: str
    identity_schedule_digest: str
    attempts: tuple[NestedOpeningAttemptObservationV1, ...]
    ranking_tables: tuple[CensusRankingTableV1, ...]
    observed_m_q_summary: tuple[NestedMQSummaryRowV1, ...]
    exact_144x2_budget_complete: bool
    engineering_budget_override: bool

    def __post_init__(self) -> None:
        if type(self.source_binding) is not NestedSourceBindingV1:
            raise NestedOpeningFeasibilityV2Error("report requires an exact source binding")
        for value, name in (
            (self.prospective_plan_digest, "prospective plan digest"),
            (self.canonical_plan_bytes_sha256, "canonical plan bytes sha256"),
            (self.upstream_census_plan_digest, "upstream census plan digest"),
            (self.upstream_census_plan_bytes_sha256, "upstream census plan bytes sha256"),
            (self.generator_contract_digest, "generator contract digest"),
            (self.identity_schedule_digest, "identity schedule digest"),
        ):
            _require_sha256(value, name=name)
        _require_integer(self.canonical_plan_byte_count, name="canonical plan byte count", minimum=1)
        if type(self.attempts) is not tuple or any(
            type(attempt) is not NestedOpeningAttemptObservationV1 for attempt in self.attempts
        ):
            raise NestedOpeningFeasibilityV2Error("report attempts require exact observations")
        if type(self.ranking_tables) is not tuple or any(
            type(table) is not CensusRankingTableV1 for table in self.ranking_tables
        ):
            raise NestedOpeningFeasibilityV2Error("report ranking tables have wrong types")
        ranking_digests = tuple(table.digest for table in self.ranking_tables)
        if ranking_digests != tuple(sorted(set(ranking_digests))):
            raise NestedOpeningFeasibilityV2Error("ranking tables must be unique and digest sorted")
        references = {
            opening.ranking_table_digest
            for attempt in self.attempts
            for member in attempt.members
            for opening in (member.openings or ())
        }
        if references != set(ranking_digests):
            raise NestedOpeningFeasibilityV2Error(
                "ranking table ledger does not exactly cover opening references"
            )
        expected_cells = tuple((m, q) for m in range(8, 17) for q in range(1, 5))
        if (
            type(self.observed_m_q_summary) is not tuple
            or any(type(row) is not NestedMQSummaryRowV1 for row in self.observed_m_q_summary)
            or tuple((row.m, row.q) for row in self.observed_m_q_summary) != expected_cells
        ):
            raise NestedOpeningFeasibilityV2Error("report must preserve all 36 m/q rows")
        if self.observed_m_q_summary != _build_mq_summary(self.attempts):
            raise NestedOpeningFeasibilityV2Error("m/q summary differs from exact ledger")
        _require_bool(self.exact_144x2_budget_complete, name="exact 144x2 budget complete")
        _require_bool(self.engineering_budget_override, name="engineering budget override")
        if self.exact_144x2_budget_complete == self.engineering_budget_override:
            raise NestedOpeningFeasibilityV2Error("report budget markers are inconsistent")
        expected_attempt_count = (
            PRODUCTION_MIRROR_ATTEMPT_COUNT if self.exact_144x2_budget_complete else len(COMPOSED_STRATA)
        )
        if self.exact_144x2_budget_complete and len(self.attempts) != expected_attempt_count:
            raise NestedOpeningFeasibilityV2Error("production report lacks 144 attempts")
        if not self.attempts:
            raise NestedOpeningFeasibilityV2Error("report cannot be empty")

    @property
    def report_kind(self) -> str:
        return _REPORT_KIND if self.exact_144x2_budget_complete else _TEST_REPORT_KIND

    @property
    def status(self) -> str:
        return _REPORT_STATUS if self.exact_144x2_budget_complete else _TEST_REPORT_STATUS

    def _fixed_budget_accounting_obj(self) -> dict[str, Any]:
        members = tuple(member for attempt in self.attempts for member in attempt.members)
        completed = tuple(member for member in members if member.status == "completed")
        return {
            "mirror_attempt_count": len(self.attempts),
            "member_record_count": len(members),
            "planned_opening_assessment_count": len(members) * DERIVED_OPENINGS_PER_MEMBER,
            "completed_opening_assessment_count": sum(len(member.openings or ()) for member in members),
            "common_union_completed_count": len(completed),
            "bounded_slot_failure_count": len(members) - len(completed),
            "nested_opening_feasibility_witness_count": sum(
                member.member_feasibility_witness for member in members
            ),
            "mirror_pair_feasibility_witness_count": sum(
                attempt.mirror_pair_feasibility_witness for attempt in self.attempts
            ),
            "all_planned_attempts_preserved": True,
            "early_stop_used": False,
            "identity_replacement_used": False,
            "exact_144x2_budget_complete": self.exact_144x2_budget_complete,
        }

    def _construction_claim_boundary_obj(self) -> dict[str, bool]:
        return {
            **{name: True for name in _TRUE_CONSTRUCTION_CLAIMS},
            **{name: False for name in _FALSE_CONSTRUCTION_CLAIMS},
        }

    def _unsigned_obj_unverified(self) -> dict[str, Any]:
        rows_obj = [row.as_obj() for row in self.observed_m_q_summary]
        return {
            "schema_version": NESTED_OPENING_FEASIBILITY_SCHEMA_VERSION,
            "report_kind": self.report_kind,
            "status": self.status,
            "protocol_quartet_candidate": False,
            "authorization": dict(_AUTHORIZATION),
            "source_binding": self.source_binding.as_obj(),
            "exact_plan_binding": {
                "prospective_plan_digest": self.prospective_plan_digest,
                "canonical_plan_bytes_sha256": self.canonical_plan_bytes_sha256,
                "canonical_plan_byte_count": self.canonical_plan_byte_count,
                "upstream_census_plan_digest": self.upstream_census_plan_digest,
                "upstream_census_plan_bytes_sha256": self.upstream_census_plan_bytes_sha256,
                "generator_contract_digest": self.generator_contract_digest,
                "identity_schedule_digest": self.identity_schedule_digest,
            },
            "fixed_budget_accounting": self._fixed_budget_accounting_obj(),
            "attempts": [attempt.as_obj() for attempt in self.attempts],
            "ranking_tables": [table.as_obj() for table in self.ranking_tables],
            "observed_m_q_summary": {
                "rows": rows_obj,
                "m_q_summary_digest": _digest(rows_obj, domain=_MQ_SUMMARY_DOMAIN),
            },
            "construction_claim_boundary": self._construction_claim_boundary_obj(),
            "selection_boundary": dict(_SELECTION_BOUNDARY),
        }

    def _artifact_obj_unverified(self) -> dict[str, Any]:
        """Private raw projection; authoritative public paths replay first."""

        return {
            **self._unsigned_obj_unverified(),
            "observed_report_digest": _report_digest_unverified(self),
        }


def _report_digest_unverified(report: NestedOpeningFeasibilityReportV1) -> str:
    return _digest(report._unsigned_obj_unverified(), domain=_REPORT_DOMAIN)


@lru_cache(maxsize=1)
def _catalog_entries() -> tuple[tuple[CatalogEntry, ...], dict[str, CatalogEntry]]:
    catalog = tuple(build_rule_catalog())
    return catalog, {entry.rule_id: entry for entry in catalog}


def _joint_cell_index_for_entries(
    placard: CatalogEntry,
    literal: CatalogEntry,
    composed: CatalogEntry,
    scene_index: int,
) -> int:
    return (
        (int(composed.truth[scene_index]) << 2)
        | (int(placard.truth[scene_index]) << 1)
        | int(literal.truth[scene_index])
    )


def _scene_pools(
    placard: CatalogEntry,
    literal: CatalogEntry,
    composed: CatalogEntry,
) -> tuple[tuple[int, ...], ...]:
    pools: list[list[int]] = [[] for _ in range(8)]
    for scene_index in range(SCENE_COUNT):
        pools[_joint_cell_index_for_entries(placard, literal, composed, scene_index)].append(scene_index)
    return tuple(tuple(pool) for pool in pools)


def _fixed_scene_orders(
    seed: str,
    attempt: NestedOpeningAttemptPlanV1,
    member: NestedOpeningMemberPlanV1,
    pools: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(
            sorted(
                pool,
                key=lambda scene_index: (
                    _hash_parts(
                        _SCENE_ORDER_DOMAIN,
                        seed,
                        attempt.parent_attempt_digest,
                        member.member_position,
                        member.triple_digest,
                        member.noisy_p_target_side,
                        cell,
                        scene_index,
                    ),
                    scene_index,
                ),
            )
        )
        for cell, pool in enumerate(pools)
    )


def _make_step(
    slot: NestedOpeningSlotPlanV1,
    before: tuple[int, int, int, int],
    available_count: int,
    window: tuple[int, ...],
    examined: int,
    outcome: SelectionOutcome,
    selected_rank: int | None,
    selected_scene: int | None,
    after: tuple[int, int, int, int] | None,
) -> NestedOpeningSelectionStepV1:
    window_obj = {
        "slot_order_hash": slot.slot_order_hash,
        "candidate_scene_indices": list(window),
    }
    unsigned = {
        "slot_position": slot.slot_position,
        "cell_prefix_rank": slot.prefix_rank,
        "joint_cell_index": slot.joint_cell_index,
        "joint_cell_slug": slot.joint_cell_slug,
        "affected_geometry_slugs": list(slot.affected_geometry_slugs),
        "survivor_counts_before": list(before),
        "available_remaining_scene_count": available_count,
        "candidate_window_count": len(window),
        "candidate_window_digest": _digest(window_obj, domain=_CANDIDATE_WINDOW_DOMAIN),
        "candidate_count_examined": examined,
        "outcome": outcome,
        "selected_candidate_window_rank": selected_rank,
        "selected_scene_index": selected_scene,
        "survivor_counts_after": None if after is None else list(after),
    }
    return NestedOpeningSelectionStepV1(
        slot.slot_position,
        slot.prefix_rank,
        slot.joint_cell_index,
        slot.joint_cell_slug,
        slot.affected_geometry_slugs,
        before,
        available_count,
        len(window),
        cast(str, unsigned["candidate_window_digest"]),
        examined,
        outcome,
        selected_rank,
        selected_scene,
        after,
        _digest(unsigned, domain=_SELECTION_STEP_DOMAIN),
    )


def _failed_member_observation(
    member: NestedOpeningMemberPlanV1,
    steps: tuple[NestedOpeningSelectionStepV1, ...],
) -> NestedOpeningMemberObservationV1:
    trace_digest = _digest([step.as_obj() for step in steps], domain=_SELECTION_TRACE_DOMAIN)
    unsigned = {
        "member_plan": member.as_obj(),
        "binding": {"member_plan_digest": member.member_plan_digest},
        "construction_status": "bounded_window_failure",
        "selection_steps": [step.as_obj() for step in steps],
        "selection_trace_digest": trace_digest,
        "opening_union_scene_ids_by_joint_cell": None,
        "opening_union_digest": None,
        "remaining_scene_count_by_joint_cell": None,
        "openings": None,
        "exact_match_evidence": None,
        "reason_codes": ["bounded_candidate_window_cannot_preserve_four_version_space_floors"],
        "nested_opening_construction_feasibility_witness": False,
        "protocol_quartet_candidate": False,
    }
    return NestedOpeningMemberObservationV1(
        member,
        {"member_plan_digest": member.member_plan_digest},
        "bounded_window_failure",
        steps,
        trace_digest,
        None,
        None,
        None,
        None,
        None,
        ("bounded_candidate_window_cannot_preserve_four_version_space_floors",),
        False,
        False,
        _digest(unsigned, domain=_MEMBER_OBSERVATION_DOMAIN),
    )


def _exact_match_evidence(
    openings: tuple[
        NestedDerivedOpeningV1,
        NestedDerivedOpeningV1,
        NestedDerivedOpeningV1,
        NestedDerivedOpeningV1,
    ],
    formula_stratum: str,
) -> NestedExactMatchEvidenceV1:
    rows = cast(
        tuple[
            NestedExactMatchRowV1,
            NestedExactMatchRowV1,
            NestedExactMatchRowV1,
            NestedExactMatchRowV1,
        ],
        tuple(
            NestedExactMatchRowV1(
                opening.geometry_slug,
                opening.assessment.observed_m,
                opening.assessment.difficulty.minimax_depth,
                formula_stratum,
                opening.assessment.difficulty.greedy_official_recovery_query_count,
                opening.assessment.difficulty.best_first_query_branch_sizes,
            )
            for opening in openings
        ),
    )
    passed = len({row.matching_key for row in rows}) == 1
    common = _matching_key_obj(rows[0]) if passed else None
    unsigned = {
        "required_fields": list(_EXACT_MATCH_FIELDS),
        "per_geometry_rows": [row.as_obj() for row in rows],
        "common_key": common,
        "passed": passed,
    }
    return NestedExactMatchEvidenceV1(
        rows,
        common,
        passed,
        _digest(unsigned, domain=_EXACT_MATCH_DOMAIN),
    )


def _construct_member(
    plan: NestedOpeningFeasibilityPlanV1,
    attempt: NestedOpeningAttemptPlanV1,
    member: NestedOpeningMemberPlanV1,
    ranking_tables: dict[str, CensusRankingTableV1],
) -> NestedOpeningMemberObservationV1:
    catalog, entries = _catalog_entries()
    placard = entries[attempt.placard_rule_id]
    literal = entries[attempt.literal_rule_id]
    composed = entries[member.composed_rule_id]
    pools = _scene_pools(placard, literal, composed)
    scene_orders = _fixed_scene_orders(
        plan.generator_binding.seed,
        attempt,
        member,
        pools,
    )
    supported = build_supported_catalog_contract_v2().supported_indices
    if len(supported) != SUPPORTED_RANKING_ROW_COUNT:
        raise NestedOpeningFeasibilityV2Error("supported V0 does not contain exactly 6970 rules")
    survivors: dict[str, tuple[int, ...]] = {slug: supported for slug in _GEOMETRY_SLUGS}
    selected_by_cell: list[list[int]] = [[] for _ in range(8)]
    steps: list[NestedOpeningSelectionStepV1] = []
    for slot in member.planned_slots:
        before = cast(
            tuple[int, int, int, int],
            tuple(len(survivors[slug]) for slug in _GEOMETRY_SLUGS),
        )
        already = set(selected_by_cell[slot.joint_cell_index])
        available = tuple(scene for scene in scene_orders[slot.joint_cell_index] if scene not in already)
        window = available[:CANDIDATE_WINDOW_K]
        selected_scene: int | None = None
        selected_rank: int | None = None
        selected_updates: dict[str, tuple[int, ...]] | None = None
        for rank, scene_index in enumerate(window, start=1):
            actual_cell = _joint_cell_index_for_entries(placard, literal, composed, scene_index)
            declared_label = bool(slot.joint_cell_index & 0b100)
            if actual_cell != slot.joint_cell_index or composed.truth[scene_index] is not declared_label:
                raise NestedOpeningFeasibilityV2Error(
                    "candidate differs from its declared joint cell or C label"
                )
            updates = {
                slug: tuple(
                    rule_index
                    for rule_index in survivors[slug]
                    if catalog[rule_index].truth[scene_index] is declared_label
                )
                for slug in slot.affected_geometry_slugs
            }
            if all(len(after) >= SUPPORTED_VERSION_SPACE_FLOOR for after in updates.values()):
                selected_scene = scene_index
                selected_rank = rank
                selected_updates = updates
                break
        if selected_scene is None or selected_rank is None or selected_updates is None:
            steps.append(
                _make_step(
                    slot,
                    before,
                    len(available),
                    window,
                    len(window),
                    "bounded_window_exhausted",
                    None,
                    None,
                    None,
                )
            )
            return _failed_member_observation(member, tuple(steps))
        # All affected filters were computed from the same pre-step snapshot.
        # Mutation happens only after every affected geometry passed the floor.
        for slug, after_space in selected_updates.items():
            survivors[slug] = after_space
        selected_by_cell[slot.joint_cell_index].append(selected_scene)
        after_counts = cast(
            tuple[int, int, int, int],
            tuple(len(survivors[slug]) for slug in _GEOMETRY_SLUGS),
        )
        steps.append(
            _make_step(
                slot,
                before,
                len(available),
                window,
                selected_rank,
                "selected",
                selected_rank,
                selected_scene,
                after_counts,
            )
        )

    union = tuple(tuple(cell) for cell in selected_by_cell)
    trace = tuple(steps)
    trace_digest = _digest([step.as_obj() for step in trace], domain=_SELECTION_TRACE_DOMAIN)
    union_obj = [list(cell) for cell in union]
    union_digest = _digest(union_obj, domain=_COMMON_UNION_DOMAIN)
    remaining = tuple(len(pool) - len(selected) for pool, selected in zip(pools, union, strict=True))
    opening_rows: list[NestedDerivedOpeningV1] = []
    for geometry_slug, variant in zip(_GEOMETRY_SLUGS, member.schedule_variants, strict=True):
        counts = _variant_schedule(variant)
        matrix = tuple(tuple(union[cell][:count]) for cell, count in enumerate(counts))
        flat = tuple(scene for cell in matrix for scene in cell)
        recomputed = tuple(
            rule_index
            for rule_index in supported
            if all(catalog[rule_index].truth[scene] is composed.truth[scene] for scene in flat)
        )
        if recomputed != survivors[geometry_slug]:
            raise NestedOpeningFeasibilityV2Error(
                "incremental geometry survivor space differs from full recomputation"
            )
        # This public assessment occurs only after all thirteen slots have been
        # selected and therefore cannot influence construction.
        assessed = assess_evaluation_opening_v1(
            attempt.placard_rule_id,
            attempt.literal_rule_id,
            member.composed_rule_id,
            variant,
            flat,
        )
        existing = ranking_tables.setdefault(assessed.ranking_table.digest, assessed.ranking_table)
        if existing != assessed.ranking_table:
            raise NestedOpeningFeasibilityV2Error("ranking-table digest collision")
        side = int(variant[-1]) if "_y" in variant else None
        unsigned = {
            "geometry_slug": geometry_slug,
            "schedule_variant": variant,
            "a_error_target_side": side,
            "opening_scene_ids_by_joint_cell": [list(cell) for cell in matrix],
            "opening_scene_indices_cell_major": list(flat),
            "assessment": assessed.record.as_obj(),
        }
        opening_rows.append(
            NestedDerivedOpeningV1(
                geometry_slug,
                variant,
                side,
                matrix,
                flat,
                assessed.record,
                assessed.ranking_table.digest,
                _digest(unsigned, domain=_DERIVED_OPENING_DOMAIN),
            )
        )
    openings = cast(
        tuple[
            NestedDerivedOpeningV1,
            NestedDerivedOpeningV1,
            NestedDerivedOpeningV1,
            NestedDerivedOpeningV1,
        ],
        tuple(opening_rows),
    )
    exact_match = _exact_match_evidence(openings, attempt.formula_stratum)
    reasons: list[str] = []
    if not exact_match.passed:
        reasons.append("cross_geometry_exact_match_failed")
    if any(opening.assessment.disposition != "cell_witness" for opening in openings):
        reasons.append("one_or_more_openings_fail_registered_salience_or_difficulty")
    witness = not reasons
    member_unsigned: dict[str, Any] = {
        "member_plan": member.as_obj(),
        "binding": {"member_plan_digest": member.member_plan_digest},
        "construction_status": "completed",
        "selection_steps": [step.as_obj() for step in trace],
        "selection_trace_digest": trace_digest,
        "opening_union_scene_ids_by_joint_cell": union_obj,
        "opening_union_digest": union_digest,
        "remaining_scene_count_by_joint_cell": list(remaining),
        "openings": [opening.as_obj() for opening in openings],
        "exact_match_evidence": exact_match.as_obj(),
        "reason_codes": sorted(reasons),
        "nested_opening_construction_feasibility_witness": witness,
        "protocol_quartet_candidate": False,
    }
    return NestedOpeningMemberObservationV1(
        member,
        {"member_plan_digest": member.member_plan_digest},
        "completed",
        trace,
        trace_digest,
        union,
        union_digest,
        remaining,
        openings,
        exact_match,
        tuple(sorted(reasons)),
        witness,
        False,
        _digest(member_unsigned, domain=_MEMBER_OBSERVATION_DOMAIN),
    )


def _construct_attempt(
    plan: NestedOpeningFeasibilityPlanV1,
    attempt: NestedOpeningAttemptPlanV1,
    ranking_tables: dict[str, CensusRankingTableV1],
) -> NestedOpeningAttemptObservationV1:
    members = cast(
        tuple[
            NestedOpeningMemberObservationV1,
            NestedOpeningMemberObservationV1,
        ],
        tuple(_construct_member(plan, attempt, member, ranking_tables) for member in attempt.members),
    )
    both_complete = all(member.status == "completed" for member in members)
    if both_complete:
        first = {
            scene
            for cell in cast(
                tuple[tuple[int, ...], ...],
                members[0].common_union_scene_indices_by_joint_cell,
            )
            for scene in cell
        }
        second = {
            scene
            for cell in cast(
                tuple[tuple[int, ...], ...],
                members[1].common_union_scene_indices_by_joint_cell,
            )
            for scene in cell
        }
        overlap = tuple(sorted(first & second))
    else:
        overlap = ()
    overlap_obj = {
        "both_common_unions_complete": both_complete,
        "member_union_overlap_scene_indices": list(overlap),
    }
    overlap_digest = _digest(overlap_obj, domain=_MIRROR_OVERLAP_DOMAIN)
    disjoint = both_complete and not overlap
    pair_witness = _attempt_pair_witness(members, disjoint)
    reasons = {reason for member in members for reason in member.reason_codes}
    if not both_complete:
        reasons.add("one_or_more_bounded_member_failures")
    elif overlap:
        reasons.add("mirror_common_union_overlap_present")
    if both_complete and not pair_witness:
        reasons.add("mirror_pair_feasibility_witness_absent")
    unsigned = {
        "attempt_plan": attempt.as_obj(),
        "members": [member.as_obj() for member in members],
        "member_union_overlap_scene_indices": list(overlap),
        "member_union_overlap_digest": overlap_digest,
        "mirror_pair_disjoint": disjoint,
        "mirror_pair_nested_opening_feasibility_witness": pair_witness,
        "reason_codes": sorted(reasons),
        "protocol_quartet_candidate": False,
    }
    return NestedOpeningAttemptObservationV1(
        attempt,
        members,
        overlap,
        overlap_digest,
        disjoint,
        pair_witness,
        tuple(sorted(reasons)),
        False,
        _digest(unsigned, domain=_ATTEMPT_OBSERVATION_DOMAIN),
    )


def _build_mq_summary(
    attempts: tuple[NestedOpeningAttemptObservationV1, ...],
) -> tuple[NestedMQSummaryRowV1, ...]:
    openings = tuple(
        opening for attempt in attempts for member in attempt.members for opening in (member.openings or ())
    )
    members = tuple(member for attempt in attempts for member in attempt.members)
    rows: list[NestedMQSummaryRowV1] = []
    for m in range(8, 17):
        for q in range(1, 5):
            in_cell = tuple(
                opening
                for opening in openings
                if opening.assessment.observed_m == m and opening.assessment.observed_q == q
            )
            member_witnesses = tuple(
                member
                for member in members
                if member.member_feasibility_witness
                and member.exact_match is not None
                and member.exact_match.common_key is not None
                and member.exact_match.common_key["version_space_size"] == m
                and member.exact_match.common_key["greedy_reference_query_count"] == q
            )
            pair_witnesses = tuple(
                attempt
                for attempt in attempts
                if attempt.mirror_pair_feasibility_witness
                and all(
                    member.exact_match is not None
                    and member.exact_match.common_key is not None
                    and member.exact_match.common_key["version_space_size"] == m
                    and member.exact_match.common_key["greedy_reference_query_count"] == q
                    for member in attempt.members
                )
            )
            rows.append(
                NestedMQSummaryRowV1(
                    m,
                    q,
                    None,
                    len(in_cell),
                    sum(opening.assessment.salience.status == "eligible" for opening in in_cell),
                    sum(opening.assessment.disposition == "cell_witness" for opening in in_cell),
                    len(member_witnesses),
                    len(pair_witnesses),
                )
            )
    return tuple(rows)


def _execute_report(
    plan_text: str,
    *,
    census_plan_text: str,
    expected_census_plan_digest: str,
    expected_census_plan_bytes_sha256: str,
    expected_plan_digest: str,
    for_testing: bool,
) -> NestedOpeningFeasibilityReportV1:
    # Parse the upstream bytes independently of the nested-plan parser.  The
    # report never trusts an identity row merely because the nested plan stored it.
    parent = _parse_exact_upstream(
        census_plan_text,
        expected_census_plan_digest=expected_census_plan_digest,
        expected_census_plan_bytes_sha256=expected_census_plan_bytes_sha256,
    )
    plan = _parse_plan(
        plan_text,
        census_plan_text=census_plan_text,
        expected_census_plan_digest=expected_census_plan_digest,
        expected_census_plan_bytes_sha256=expected_census_plan_bytes_sha256,
        expected_plan_digest=expected_plan_digest,
        for_testing=for_testing,
    )
    if tuple(attempt.parent_attempt_digest for attempt in plan.attempts) != tuple(
        attempt.attempt_digest for attempt in parent.attempts
    ):
        raise NestedOpeningFeasibilityV2Error("report parent replay changed attempt identities")
    ranking_tables: dict[str, CensusRankingTableV1] = {}
    attempts = tuple(_construct_attempt(plan, attempt, ranking_tables) for attempt in plan.attempts)
    plan_sha, plan_count = _sha256_text_bytes(plan_text, name="nested plan")
    tables = tuple(sorted(ranking_tables.values(), key=lambda table: table.digest))
    summary = _build_mq_summary(attempts)
    report = NestedOpeningFeasibilityReportV1(
        plan.source_binding,
        _plan_digest_unverified(plan),
        plan_sha,
        plan_count,
        parent.digest,
        plan.upstream_identity_plan_binding.canonical_census_plan_bytes_sha256,
        plan.generator_binding.contract_digest,
        plan.upstream_identity_plan_binding.identity_schedule_digest,
        attempts,
        tables,
        summary,
        plan.exact_144x2_budget_complete,
        plan.engineering_budget_override,
    )
    if len(report.attempts) != len(plan.attempts):
        raise NestedOpeningFeasibilityV2Error("report did not preserve every planned attempt")
    return report


def build_nested_opening_construction_feasibility_report_v1(
    plan_text: str,
    *,
    census_plan_text: str,
    expected_census_plan_digest: str,
    expected_census_plan_bytes_sha256: str,
    expected_plan_digest: str,
) -> NestedOpeningFeasibilityReportV1:
    """Execute the exact production plan without selecting or authorizing a bank."""

    return _execute_report(
        plan_text,
        census_plan_text=census_plan_text,
        expected_census_plan_digest=expected_census_plan_digest,
        expected_census_plan_bytes_sha256=expected_census_plan_bytes_sha256,
        expected_plan_digest=expected_plan_digest,
        for_testing=False,
    )


def build_nested_opening_construction_feasibility_report_for_testing_v1(
    plan_text: str,
    *,
    census_plan_text: str,
    expected_census_plan_digest: str,
    expected_census_plan_bytes_sha256: str,
    expected_plan_digest: str,
) -> NestedOpeningFeasibilityReportV1:
    """Execute only a nominally distinct reduced engineering plan."""

    return _execute_report(
        plan_text,
        census_plan_text=census_plan_text,
        expected_census_plan_digest=expected_census_plan_digest,
        expected_census_plan_bytes_sha256=expected_census_plan_bytes_sha256,
        expected_plan_digest=expected_plan_digest,
        for_testing=True,
    )


def _derive_report_digest(
    report: NestedOpeningFeasibilityReportV1,
    plan_text: str,
    *,
    census_plan_text: str,
    expected_census_plan_digest: str,
    expected_census_plan_bytes_sha256: str,
    expected_plan_digest: str,
    for_testing: bool,
) -> str:
    if type(report) is not NestedOpeningFeasibilityReportV1:
        raise TypeError("report must be a NestedOpeningFeasibilityReportV1")
    expected = _execute_report(
        plan_text,
        census_plan_text=census_plan_text,
        expected_census_plan_digest=expected_census_plan_digest,
        expected_census_plan_bytes_sha256=expected_census_plan_bytes_sha256,
        expected_plan_digest=expected_plan_digest,
        for_testing=for_testing,
    )
    if report != expected:
        raise NestedOpeningFeasibilityV2Error("report differs from full construction replay")
    return _report_digest_unverified(expected)


def derive_nested_opening_construction_feasibility_report_digest_v1(
    report: NestedOpeningFeasibilityReportV1,
    plan_text: str,
    *,
    census_plan_text: str,
    expected_census_plan_digest: str,
    expected_census_plan_bytes_sha256: str,
    expected_plan_digest: str,
) -> str:
    """Return a production report digest only after complete fresh replay."""

    if report.engineering_budget_override:
        raise NestedOpeningFeasibilityV2Error(
            "production digest derivation refuses a test-only reduced report"
        )
    return _derive_report_digest(
        report,
        plan_text,
        census_plan_text=census_plan_text,
        expected_census_plan_digest=expected_census_plan_digest,
        expected_census_plan_bytes_sha256=expected_census_plan_bytes_sha256,
        expected_plan_digest=expected_plan_digest,
        for_testing=False,
    )


def derive_nested_opening_construction_feasibility_report_digest_for_testing_v1(
    report: NestedOpeningFeasibilityReportV1,
    plan_text: str,
    *,
    census_plan_text: str,
    expected_census_plan_digest: str,
    expected_census_plan_bytes_sha256: str,
    expected_plan_digest: str,
) -> str:
    """Return a reduced report digest only after complete fresh replay."""

    if not report.engineering_budget_override:
        raise NestedOpeningFeasibilityV2Error("test-only digest derivation refuses a production report")
    return _derive_report_digest(
        report,
        plan_text,
        census_plan_text=census_plan_text,
        expected_census_plan_digest=expected_census_plan_digest,
        expected_census_plan_bytes_sha256=expected_census_plan_bytes_sha256,
        expected_plan_digest=expected_plan_digest,
        for_testing=True,
    )


def serialize_nested_opening_construction_feasibility_report_v1(
    report: NestedOpeningFeasibilityReportV1,
    plan_text: str,
    *,
    census_plan_text: str,
    expected_census_plan_digest: str,
    expected_census_plan_bytes_sha256: str,
    expected_plan_digest: str,
    expected_report_digest: str,
) -> str:
    """Serialize only after full replay against both externally bound plans."""

    if type(report) is not NestedOpeningFeasibilityReportV1:
        raise TypeError("report must be a NestedOpeningFeasibilityReportV1")
    if report.engineering_budget_override:
        raise NestedOpeningFeasibilityV2Error("production serializer refuses a test-only reduced report")
    verified = verify_nested_opening_construction_feasibility_report_v1(
        report,
        plan_text,
        census_plan_text=census_plan_text,
        expected_census_plan_digest=expected_census_plan_digest,
        expected_census_plan_bytes_sha256=expected_census_plan_bytes_sha256,
        expected_plan_digest=expected_plan_digest,
        expected_report_digest=expected_report_digest,
    )
    return _serialize_report_after_replay(verified, for_testing=False)


def serialize_nested_opening_construction_feasibility_report_for_testing_v1(
    report: NestedOpeningFeasibilityReportV1,
    plan_text: str,
    *,
    census_plan_text: str,
    expected_census_plan_digest: str,
    expected_census_plan_bytes_sha256: str,
    expected_plan_digest: str,
    expected_report_digest: str,
) -> str:
    """Serialize a reduced report only after full engineering-plan replay."""

    if type(report) is not NestedOpeningFeasibilityReportV1:
        raise TypeError("report must be a NestedOpeningFeasibilityReportV1")
    if not report.engineering_budget_override:
        raise NestedOpeningFeasibilityV2Error("test-only serializer refuses a production report")
    verified = verify_nested_opening_construction_feasibility_report_for_testing_v1(
        report,
        plan_text,
        census_plan_text=census_plan_text,
        expected_census_plan_digest=expected_census_plan_digest,
        expected_census_plan_bytes_sha256=expected_census_plan_bytes_sha256,
        expected_plan_digest=expected_plan_digest,
        expected_report_digest=expected_report_digest,
    )
    return _serialize_report_after_replay(verified, for_testing=True)


def _serialize_report_after_replay(
    report: NestedOpeningFeasibilityReportV1,
    *,
    for_testing: bool,
) -> str:
    """Private projection used only after an exact replay in the caller."""

    if report.engineering_budget_override != for_testing:
        raise NestedOpeningFeasibilityV2Error("report kind differs from serializer boundary")
    return _dump_json(report._artifact_obj_unverified()) + "\n"


def verify_nested_opening_construction_feasibility_report_v1(
    report: NestedOpeningFeasibilityReportV1,
    plan_text: str,
    *,
    census_plan_text: str,
    expected_census_plan_digest: str,
    expected_census_plan_bytes_sha256: str,
    expected_plan_digest: str,
    expected_report_digest: str,
) -> NestedOpeningFeasibilityReportV1:
    """Independently replay only a production-kind report."""

    if report.engineering_budget_override:
        raise NestedOpeningFeasibilityV2Error("production verifier refuses a test-only reduced report")
    return _verify_report(
        report,
        plan_text,
        census_plan_text=census_plan_text,
        expected_census_plan_digest=expected_census_plan_digest,
        expected_census_plan_bytes_sha256=expected_census_plan_bytes_sha256,
        expected_plan_digest=expected_plan_digest,
        expected_report_digest=expected_report_digest,
        for_testing=False,
    )


def verify_nested_opening_construction_feasibility_report_for_testing_v1(
    report: NestedOpeningFeasibilityReportV1,
    plan_text: str,
    *,
    census_plan_text: str,
    expected_census_plan_digest: str,
    expected_census_plan_bytes_sha256: str,
    expected_plan_digest: str,
    expected_report_digest: str,
) -> NestedOpeningFeasibilityReportV1:
    """Independently replay only a reduced engineering report."""

    if not report.engineering_budget_override:
        raise NestedOpeningFeasibilityV2Error("test-only verifier refuses a production report")
    return _verify_report(
        report,
        plan_text,
        census_plan_text=census_plan_text,
        expected_census_plan_digest=expected_census_plan_digest,
        expected_census_plan_bytes_sha256=expected_census_plan_bytes_sha256,
        expected_plan_digest=expected_plan_digest,
        expected_report_digest=expected_report_digest,
        for_testing=True,
    )


def _verify_report(
    report: NestedOpeningFeasibilityReportV1,
    plan_text: str,
    *,
    census_plan_text: str,
    expected_census_plan_digest: str,
    expected_census_plan_bytes_sha256: str,
    expected_plan_digest: str,
    expected_report_digest: str,
    for_testing: bool,
) -> NestedOpeningFeasibilityReportV1:
    """Private shared verifier with an explicit nominal-kind selector."""

    if type(report) is not NestedOpeningFeasibilityReportV1:
        raise TypeError("report must be a NestedOpeningFeasibilityReportV1")
    _require_sha256(expected_report_digest, name="expected report digest")
    # Reject false exact-replay bindings before the expensive construction
    # replay.  This is only a preflight: a matching report is still rebuilt in
    # full below, including every construction step and public assessment.
    parent = _parse_exact_upstream(
        census_plan_text,
        expected_census_plan_digest=expected_census_plan_digest,
        expected_census_plan_bytes_sha256=expected_census_plan_bytes_sha256,
    )
    plan = _parse_plan(
        plan_text,
        census_plan_text=census_plan_text,
        expected_census_plan_digest=expected_census_plan_digest,
        expected_census_plan_bytes_sha256=expected_census_plan_bytes_sha256,
        expected_plan_digest=expected_plan_digest,
        for_testing=for_testing,
    )
    plan_sha, plan_count = _sha256_text_bytes(plan_text, name="nested plan")
    exact_binding = (
        report.source_binding == plan.source_binding
        and report.prospective_plan_digest == expected_plan_digest
        and report.canonical_plan_bytes_sha256 == plan_sha
        and report.canonical_plan_byte_count == plan_count
        and report.upstream_census_plan_digest == parent.digest
        and report.upstream_census_plan_bytes_sha256 == expected_census_plan_bytes_sha256
        and report.generator_contract_digest == plan.generator_binding.contract_digest
        and report.identity_schedule_digest == plan.upstream_identity_plan_binding.identity_schedule_digest
        and report.exact_144x2_budget_complete == plan.exact_144x2_budget_complete
        and report.engineering_budget_override == plan.engineering_budget_override
    )
    if not exact_binding:
        raise NestedOpeningFeasibilityV2Error("report exact-plan binding differs from external replay")
    expected = _execute_report(
        plan_text,
        census_plan_text=census_plan_text,
        expected_census_plan_digest=expected_census_plan_digest,
        expected_census_plan_bytes_sha256=expected_census_plan_bytes_sha256,
        expected_plan_digest=expected_plan_digest,
        for_testing=for_testing,
    )
    if _report_digest_unverified(expected) != expected_report_digest:
        raise NestedOpeningFeasibilityV2Error("replayed report differs from externally expected digest")
    if report != expected:
        raise NestedOpeningFeasibilityV2Error("report differs from full construction replay")
    return expected


def _parse_report(
    text: str,
    *,
    plan_text: str,
    census_plan_text: str,
    expected_census_plan_digest: str,
    expected_census_plan_bytes_sha256: str,
    expected_plan_digest: str,
    expected_report_digest: str,
    for_testing: bool,
) -> NestedOpeningFeasibilityReportV1:
    _require_sha256(expected_report_digest, name="expected report digest")
    obj = _require_mapping(
        _load_json(text),
        (
            "schema_version",
            "report_kind",
            "status",
            "protocol_quartet_candidate",
            "authorization",
            "source_binding",
            "exact_plan_binding",
            "fixed_budget_accounting",
            "attempts",
            "ranking_tables",
            "observed_m_q_summary",
            "construction_claim_boundary",
            "selection_boundary",
            "observed_report_digest",
        ),
        name="nested-opening feasibility report",
    )
    expected_kind = _TEST_REPORT_KIND if for_testing else _REPORT_KIND
    expected_status = _TEST_REPORT_STATUS if for_testing else _REPORT_STATUS
    if (
        obj["schema_version"] != NESTED_OPENING_FEASIBILITY_SCHEMA_VERSION
        or obj["report_kind"] != expected_kind
        or obj["status"] != expected_status
        or obj["protocol_quartet_candidate"] is not False
        or obj["authorization"] != _AUTHORIZATION
        or obj["selection_boundary"] != _SELECTION_BOUNDARY
    ):
        raise NestedOpeningFeasibilityV2Error(
            "report kind, status, authorization, or false/null boundary changed"
        )
    expected = _execute_report(
        plan_text,
        census_plan_text=census_plan_text,
        expected_census_plan_digest=expected_census_plan_digest,
        expected_census_plan_bytes_sha256=expected_census_plan_bytes_sha256,
        expected_plan_digest=expected_plan_digest,
        for_testing=for_testing,
    )
    expected_digest = _report_digest_unverified(expected)
    if expected_digest != expected_report_digest or obj["observed_report_digest"] != expected_digest:
        raise NestedOpeningFeasibilityV2Error("report differs from externally expected digest")
    if _dump_json(obj) != _dump_json(expected._artifact_obj_unverified()):
        raise NestedOpeningFeasibilityV2Error("report differs from exact independent replay")
    canonical = _serialize_report_after_replay(expected, for_testing=for_testing)
    if canonical != text:
        raise NestedOpeningFeasibilityV2Error("report is not canonical newline-terminated JSON")
    return expected


def parse_nested_opening_construction_feasibility_report_v1(
    text: str,
    *,
    plan_text: str,
    census_plan_text: str,
    expected_census_plan_digest: str,
    expected_census_plan_bytes_sha256: str,
    expected_plan_digest: str,
    expected_report_digest: str,
) -> NestedOpeningFeasibilityReportV1:
    """Parse and replay only a production-kind report."""

    return _parse_report(
        text,
        plan_text=plan_text,
        census_plan_text=census_plan_text,
        expected_census_plan_digest=expected_census_plan_digest,
        expected_census_plan_bytes_sha256=expected_census_plan_bytes_sha256,
        expected_plan_digest=expected_plan_digest,
        expected_report_digest=expected_report_digest,
        for_testing=False,
    )


def parse_nested_opening_construction_feasibility_report_for_testing_v1(
    text: str,
    *,
    plan_text: str,
    census_plan_text: str,
    expected_census_plan_digest: str,
    expected_census_plan_bytes_sha256: str,
    expected_plan_digest: str,
    expected_report_digest: str,
) -> NestedOpeningFeasibilityReportV1:
    """Parse and replay only the distinct reduced engineering report kind."""

    return _parse_report(
        text,
        plan_text=plan_text,
        census_plan_text=census_plan_text,
        expected_census_plan_digest=expected_census_plan_digest,
        expected_census_plan_bytes_sha256=expected_census_plan_bytes_sha256,
        expected_plan_digest=expected_plan_digest,
        expected_report_digest=expected_report_digest,
        for_testing=True,
    )
