"""Bounded, CPU-only evaluation-opening feasibility census for G03-v2.

The prospective plan in this module contains only source/catalog/generator
identities and deterministic catalog draw specifications.  A separate observed
report executes every planned mirror-pair attempt, preserves every constructed
opening or typed pre-opening failure, and summarizes feasibility over the full
``m=8..16`` by ``q=1..4`` grid.  It never chooses a positive cell quota or a
matched-bank size and cannot authorize G01 or a G03 launch.

One census mirror attempt binds one ``P/Q`` pair and two distinct composed
rules from the same formula stratum.  Each composed rule receives all four
evidence geometries; its two noisy-P geometries share one Official target
side, and the other composed rule uses the opposite side.  These eight
opening draws are evidence about construction feasibility only.  They are not
materialized evaluation quartets or a selected mirror-balanced bank.
"""

from __future__ import annotations

import hashlib
import heapq
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

from goalzendo_interactive import catalog as catalog_module
from goalzendo_interactive import query as query_module
from goalzendo_interactive import stage_partitions_v2 as stage_partitions_module
from goalzendo_interactive.catalog import CatalogEntry, VersionSpace, build_rule_catalog
from goalzendo_interactive.provenance import interactive_source_provenance
from goalzendo_interactive.rules import BinaryOp, BinaryRule
from goalzendo_interactive.rules import Literal as RuleLiteral
from goalzendo_interactive.schema import SCENE_COUNT
from goalzendo_interactive.stage_partitions_v2 import (
    RULE_PARTITION_SCHEMA_VERSION_V2,
    build_rule_identity_partitions_v2,
)

from . import bank_audits as bank_audits_module
from . import challenge_query as challenge_query_module
from . import population_audit as population_audit_module
from . import role_schema as role_schema_module
from .bank_audits import build_rule_triple_bindings_batch_for_audit_v2
from .challenge_query import build_query_policy_ceiling_report_v2
from .population_audit import (
    COMPOSED_STRATA,
    build_supported_catalog_contract_v2,
    classify_catalog_identity_v2,
    composed_stratum_v2,
    joint_truth_cell_counts_v2,
)
from .role_schema import (
    EVIDENCE_GEOMETRIES,
    EvidenceGeometryV2,
    OfficialTargetSideV2,
    build_evidence_schedule_v2,
)

EVALUATION_CENSUS_SCHEMA_VERSION = 1
PRODUCTION_MIRROR_ATTEMPTS_PER_STRATUM = 16
MAXIMUM_MIRROR_ATTEMPTS_PER_STRATUM = 16
DEFAULT_CANDIDATE_POOL_SIZE = 32
MAXIMUM_CANDIDATE_POOL_SIZE = 64
MINIMUM_SUPPORTED_VERSION_SPACE_SIZE = 8
MAXIMUM_SUPPORTED_VERSION_SPACE_SIZE = 16
MAXIMUM_GREEDY_RECOVERY_QUERIES = 4
SUPPORTED_RANKING_ROW_COUNT = 6_970
REGISTERED_POPULATION_RESERVE_PER_JOINT_CELL = 27

_PLAN_KIND = "g03-v2-evaluation-feasibility-census-prospective-plan"
_REPORT_KIND = "g03-v2-evaluation-feasibility-census-observed-report"
_PLAN_DOMAIN = "goalzendo-interactive-v2-evaluation-census-plan-v1"
_REPORT_DOMAIN = "goalzendo-interactive-v2-evaluation-census-report-v1"
_ATTEMPT_DOMAIN = "goalzendo-interactive-v2-evaluation-census-attempt-v1"
_GENERATOR_CONTRACT_DOMAIN = "goalzendo-interactive-v2-evaluation-census-generator-v1"
_SOURCE_MANIFEST_DOMAIN = "goalzendo-interactive-v2-evaluation-census-sources-v1"
_CATALOG_INPUT_DOMAIN = "goalzendo-interactive-v2-evaluation-census-catalog-input-v1"
_RANKING_TABLE_DOMAIN = "goalzendo-interactive-v2-evaluation-census-ranking-table-v1"
_COMPLETE_RANKING_ROWS_DOMAIN = b"goalzendo-interactive-v2-evaluation-census-complete-ranking-rows-v1\0"
_FIRST_SPLIT_DOMAIN = "goalzendo-interactive-v2-evaluation-census-first-splits-v1"
_SCENE_ORDER_DOMAIN = b"goalzendo-interactive-v2-evaluation-census-scene-order-v1\0"
_OCCURRENCE_ORDER_DOMAIN = b"goalzendo-interactive-v2-evaluation-census-occurrence-order-v1\0"
_IDENTITY_ORDER_DOMAIN = b"goalzendo-interactive-v2-evaluation-census-identity-order-v1\0"

_AUTHORIZATION: dict[str, str | bool] = {
    "scope": "cpu_only_evaluation_opening_feasibility_census",
    "g01_authorized": False,
    "g03_capability_launch_authorized": False,
    "g03_scientific_launch_authorized": False,
}

_CLAIM_BOUNDARY: dict[str, bool] = {
    "positive_m_q_quota_selected": False,
    "matched_bank_size_selected": False,
    "evaluation_quartets_materialized": False,
    "nested_opening_unions_materialized": False,
    "difficulty_matcher_run": False,
    "oversupply_selected": False,
    "challenge_reservoirs_materialized": False,
    "challenge_panels_materialized": False,
    "interventions_materialized": False,
    "render_balance_audited": False,
    "cross_stage_disjointness_audited": False,
    "leakage_audited": False,
    "power_analysis_completed": False,
    "model_outcomes_present": False,
    "runtime_measurements_present": False,
    "optimizer_state_present": False,
    "checkpoint_state_present": False,
    "reward_data_present": False,
    "g01_authorized": False,
    "launch_authorized": False,
}

_PLAN_FORBIDDEN_KEYS = frozenset(
    {
        "opening",
        "openings",
        "result",
        "results",
        "failure",
        "failures",
        "observed",
        "scene_indices",
        "ranking_rows",
        "disposition",
        "reason_codes",
    }
)

ScheduleVariant = Literal["a0_b0", "a0_b2", "a1_b0_y0", "a1_b0_y1", "a1_b2_y0", "a1_b2_y1"]
ConstructionStage = Literal["joint_cell_supply", "minimum_space_guard"]
OpeningDisposition = Literal["cell_witness", "constructed_rejection"]
GroupDisposition = Literal["candidate_group", "preserved_rejection"]
AttemptDisposition = Literal["complete_census_pair", "preserved_rejection"]


class EvaluationCensusV2Error(ValueError):
    """Raised when a census plan or observed report is noncanonical."""


def _dump_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise EvaluationCensusV2Error(f"value is not canonical JSON: {exc}") from exc


def _load_json(text: str) -> Any:
    if type(text) is not str or not text:
        raise EvaluationCensusV2Error("JSON input must be a nonempty string")

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise EvaluationCensusV2Error(f"duplicate JSON object key: {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise EvaluationCensusV2Error(f"non-finite JSON constant is forbidden: {value}")

    try:
        return json.loads(text, object_pairs_hook=no_duplicates, parse_constant=reject_constant)
    except EvaluationCensusV2Error:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise EvaluationCensusV2Error(f"invalid JSON: {exc}") from exc


def _digest(value: Any, *, domain: str) -> str:
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
        raise EvaluationCensusV2Error(f"{name} must be a lowercase SHA-256")
    return cast(str, value)


def _require_integer(
    value: object,
    *,
    name: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise EvaluationCensusV2Error(f"{name} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise EvaluationCensusV2Error(f"{name} must be an integer <= {maximum}")
    return value


def _require_mapping(value: object, fields: tuple[str, ...], *, name: str) -> Mapping[str, Any]:
    if type(value) is not dict or tuple(value) != fields:
        raise EvaluationCensusV2Error(f"{name} has noncanonical or reordered fields")
    return cast(Mapping[str, Any], value)


def _source_file_sha256(module_file: str | None, *, name: str) -> str:
    if module_file is None:
        raise EvaluationCensusV2Error(f"{name} source has no file")
    path = Path(module_file)
    if path.is_symlink() or not path.is_file():
        raise EvaluationCensusV2Error(f"{name} source must be one ordinary file")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stratum_slug(op: BinaryOp, negated_literal_count: int) -> str:
    return f"{op}__neg{negated_literal_count}"


def _parse_stratum_slug(slug: object) -> tuple[BinaryOp, int]:
    if type(slug) is not str:
        raise EvaluationCensusV2Error("formula stratum slug must be a string")
    for op, negated in COMPOSED_STRATA:
        if slug == _stratum_slug(op, negated):
            return op, negated
    raise EvaluationCensusV2Error(f"unknown formula stratum: {slug!r}")


def _schedule_variants() -> tuple[ScheduleVariant, ...]:
    return (
        "a0_b0",
        "a0_b2",
        "a1_b0_y0",
        "a1_b0_y1",
        "a1_b2_y0",
        "a1_b2_y1",
    )


def _generator_contract_obj() -> dict[str, Any]:
    return {
        "schema_version": EVALUATION_CENSUS_SCHEMA_VERSION,
        "execution_class": "bounded_cpu_only",
        "formula_strata": [
            {
                "operator": op,
                "negated_literal_count": negated,
                "slug": _stratum_slug(op, negated),
            }
            for op, negated in COMPOSED_STRATA
        ],
        "schedule_variants": list(_schedule_variants()),
        "mirror_attempt_semantics": (
            "one P/Q identity pair, two distinct same-stratum C identities, four geometries "
            "per C, one shared noisy-P target side within C, opposite side across C identities"
        ),
        "catalog_draw_order": (
            "hash-ranked P and Q per stratum/position; hash-ranked C identities per stratum; "
            "canonical positions consume consecutive nonreused C pairs"
        ),
        "demonstration_count": 10,
        "supported_version_space_floor_during_construction": MINIMUM_SUPPORTED_VERSION_SPACE_SIZE,
        "candidate_selection": (
            "hash-rank the bounded per-cell scene pool; take the first ranked candidate that "
            "preserves the fixed protocol floor of eight supported rules; never optimize or target "
            "the final m or q cell"
        ),
        "candidate_pool_size_default": DEFAULT_CANDIDATE_POOL_SIZE,
        "candidate_pool_size_maximum": MAXIMUM_CANDIDATE_POOL_SIZE,
        "population_reserve_per_joint_cell": REGISTERED_POPULATION_RESERVE_PER_JOINT_CELL,
        "fixed_budget_no_early_stop": True,
        "ranking_order": (
            "descending correct-demonstration count, ascending literal count, ascending explicit "
            "negation count, ascending canonical rule id"
        ),
        "ranking_row_fields": [
            "total_rank",
            "source_catalog_index",
            "rule_id",
            "truth_digest",
            "supported_family",
            "correct_demonstrations",
            "error_demonstrations",
            "literal_count",
            "explicit_negation_count",
        ],
        "ranking_scope": "all 6970 v2-supported canonical identities for every constructed candidate",
        "m_definition": "exact supported opening version-space size",
        "q_definition": "entropy-greedy Official recovery query count",
        "reported_m_values": list(range(8, 17)),
        "reported_q_values": list(range(1, 5)),
        "query_exclusions": "the ten demonstration scenes",
        "preconstruction_failure_stages": ["joint_cell_supply", "minimum_space_guard"],
        "no_positive_m_q_quota": True,
        "no_selected_bank_size": True,
    }


@dataclass(frozen=True, slots=True)
class CensusSourceBindingV1:
    frozen_v1_source_fingerprint: str
    catalog_source_sha256: str
    query_source_sha256: str
    stage_partitions_source_sha256: str
    role_schema_source_sha256: str
    population_audit_source_sha256: str
    challenge_query_source_sha256: str
    bank_audits_source_sha256: str
    census_source_sha256: str
    source_manifest_digest: str

    def __post_init__(self) -> None:
        for name in (
            "frozen_v1_source_fingerprint",
            "catalog_source_sha256",
            "query_source_sha256",
            "stage_partitions_source_sha256",
            "role_schema_source_sha256",
            "population_audit_source_sha256",
            "challenge_query_source_sha256",
            "bank_audits_source_sha256",
            "census_source_sha256",
            "source_manifest_digest",
        ):
            _require_sha256(getattr(self, name), name=name)
        if self.source_manifest_digest != _digest(self._manifest_obj(), domain=_SOURCE_MANIFEST_DOMAIN):
            raise EvaluationCensusV2Error("source-manifest digest is inconsistent")

    def _manifest_obj(self) -> dict[str, str]:
        return {
            "frozen_v1_source_fingerprint": self.frozen_v1_source_fingerprint,
            "catalog_source_sha256": self.catalog_source_sha256,
            "query_source_sha256": self.query_source_sha256,
            "stage_partitions_source_sha256": self.stage_partitions_source_sha256,
            "role_schema_source_sha256": self.role_schema_source_sha256,
            "population_audit_source_sha256": self.population_audit_source_sha256,
            "challenge_query_source_sha256": self.challenge_query_source_sha256,
            "bank_audits_source_sha256": self.bank_audits_source_sha256,
            "census_source_sha256": self.census_source_sha256,
        }

    def as_obj(self) -> dict[str, Any]:
        return {**self._manifest_obj(), "source_manifest_digest": self.source_manifest_digest}


@lru_cache(maxsize=1)
def _current_source_binding() -> CensusSourceBindingV1:
    manifest = {
        "frozen_v1_source_fingerprint": interactive_source_provenance().fingerprint,
        "catalog_source_sha256": _source_file_sha256(catalog_module.__file__, name="catalog"),
        "query_source_sha256": _source_file_sha256(query_module.__file__, name="query"),
        "stage_partitions_source_sha256": _source_file_sha256(
            stage_partitions_module.__file__, name="stage partitions"
        ),
        "role_schema_source_sha256": _source_file_sha256(role_schema_module.__file__, name="role schema"),
        "population_audit_source_sha256": _source_file_sha256(
            population_audit_module.__file__, name="population audit"
        ),
        "challenge_query_source_sha256": _source_file_sha256(
            challenge_query_module.__file__, name="challenge query"
        ),
        "bank_audits_source_sha256": _source_file_sha256(bank_audits_module.__file__, name="bank audits"),
        "census_source_sha256": _source_file_sha256(__file__, name="evaluation census"),
    }
    return CensusSourceBindingV1(
        **manifest,
        source_manifest_digest=_digest(manifest, domain=_SOURCE_MANIFEST_DOMAIN),
    )


@dataclass(frozen=True, slots=True)
class CensusCatalogBindingV1:
    source_catalog_digest: str
    supported_catalog_digest: str
    supported_identity_count: int
    stage_partition_schema_version: int
    stage_partition_digest: str
    catalog_input_digest: str

    def __post_init__(self) -> None:
        _require_sha256(self.source_catalog_digest, name="source catalog digest")
        _require_sha256(self.supported_catalog_digest, name="supported catalog digest")
        _require_integer(
            self.supported_identity_count,
            name="supported identity count",
            minimum=SUPPORTED_RANKING_ROW_COUNT,
            maximum=SUPPORTED_RANKING_ROW_COUNT,
        )
        if self.stage_partition_schema_version != RULE_PARTITION_SCHEMA_VERSION_V2:
            raise EvaluationCensusV2Error("stage-partition schema version is inconsistent")
        _require_sha256(self.stage_partition_digest, name="stage partition digest")
        _require_sha256(self.catalog_input_digest, name="catalog input digest")
        if self.catalog_input_digest != _digest(self._input_obj(), domain=_CATALOG_INPUT_DOMAIN):
            raise EvaluationCensusV2Error("catalog-input digest is inconsistent")

    def _input_obj(self) -> dict[str, Any]:
        return {
            "source_catalog_digest": self.source_catalog_digest,
            "supported_catalog_digest": self.supported_catalog_digest,
            "supported_identity_count": self.supported_identity_count,
            "stage_partition_schema_version": self.stage_partition_schema_version,
            "stage_partition_digest": self.stage_partition_digest,
            "candidate_families": [
                "placard_literal",
                "one_literal_piece",
                "composed_two_literal_piece",
            ],
            "composed_formula_strata": [[op, negated] for op, negated in COMPOSED_STRATA],
        }

    def as_obj(self) -> dict[str, Any]:
        return {**self._input_obj(), "catalog_input_digest": self.catalog_input_digest}


@lru_cache(maxsize=1)
def _current_catalog_binding() -> CensusCatalogBindingV1:
    catalog = build_rule_catalog()
    contract = build_supported_catalog_contract_v2()
    if contract.source_catalog_digest != catalog.digest:
        raise EvaluationCensusV2Error("supported contract and source catalog disagree")
    if len(contract.supported_indices) != SUPPORTED_RANKING_ROW_COUNT:
        raise EvaluationCensusV2Error("supported catalog no longer contains exactly 6970 identities")
    partitions = build_rule_identity_partitions_v2()
    payload = {
        "source_catalog_digest": catalog.digest,
        "supported_catalog_digest": contract.supported_catalog_digest,
        "supported_identity_count": len(contract.supported_indices),
        "stage_partition_schema_version": RULE_PARTITION_SCHEMA_VERSION_V2,
        "stage_partition_digest": partitions.digest,
        "candidate_families": [
            "placard_literal",
            "one_literal_piece",
            "composed_two_literal_piece",
        ],
        "composed_formula_strata": [[op, negated] for op, negated in COMPOSED_STRATA],
    }
    return CensusCatalogBindingV1(
        source_catalog_digest=catalog.digest,
        supported_catalog_digest=contract.supported_catalog_digest,
        supported_identity_count=len(contract.supported_indices),
        stage_partition_schema_version=RULE_PARTITION_SCHEMA_VERSION_V2,
        stage_partition_digest=partitions.digest,
        catalog_input_digest=_digest(payload, domain=_CATALOG_INPUT_DOMAIN),
    )


@dataclass(frozen=True, slots=True)
class CensusGeneratorBindingV1:
    generator_schema_version: int
    generator_contract_digest: str
    generator_source_sha256: str

    def __post_init__(self) -> None:
        if self.generator_schema_version != EVALUATION_CENSUS_SCHEMA_VERSION:
            raise EvaluationCensusV2Error("generator schema version is inconsistent")
        _require_sha256(self.generator_contract_digest, name="generator contract digest")
        _require_sha256(self.generator_source_sha256, name="generator source sha256")
        if self.generator_contract_digest != _digest(
            _generator_contract_obj(), domain=_GENERATOR_CONTRACT_DOMAIN
        ):
            raise EvaluationCensusV2Error("generator-contract digest is inconsistent")

    def as_obj(self) -> dict[str, Any]:
        return {
            "generator_schema_version": self.generator_schema_version,
            "generator_contract": _generator_contract_obj(),
            "generator_contract_digest": self.generator_contract_digest,
            "generator_source_sha256": self.generator_source_sha256,
        }


def _current_generator_binding() -> CensusGeneratorBindingV1:
    source = _current_source_binding()
    return CensusGeneratorBindingV1(
        EVALUATION_CENSUS_SCHEMA_VERSION,
        _digest(_generator_contract_obj(), domain=_GENERATOR_CONTRACT_DOMAIN),
        source.census_source_sha256,
    )


@dataclass(frozen=True, slots=True)
class CensusMemberPlanV1:
    member_position: int
    composed_rule_id: str
    composed_truth_digest: str
    triple_digest: str
    noisy_p_target_side: int

    def __post_init__(self) -> None:
        _require_integer(self.member_position, name="member position", maximum=1)
        _require_sha256(self.composed_truth_digest, name="composed truth digest")
        _require_sha256(self.triple_digest, name="triple digest")
        _require_integer(self.noisy_p_target_side, name="noisy P target side", maximum=1)

    def as_obj(self) -> dict[str, Any]:
        return {
            "member_position": self.member_position,
            "composed_rule_id": self.composed_rule_id,
            "composed_truth_digest": self.composed_truth_digest,
            "triple_digest": self.triple_digest,
            "noisy_p_target_side": self.noisy_p_target_side,
        }


@dataclass(frozen=True, slots=True)
class CensusAttemptPlanV1:
    formula_stratum: str
    canonical_position: int
    placard_rule_id: str
    placard_truth_digest: str
    literal_rule_id: str
    literal_truth_digest: str
    members: tuple[CensusMemberPlanV1, CensusMemberPlanV1]
    attempt_digest: str

    def __post_init__(self) -> None:
        _parse_stratum_slug(self.formula_stratum)
        _require_integer(
            self.canonical_position,
            name="canonical position",
            maximum=MAXIMUM_MIRROR_ATTEMPTS_PER_STRATUM - 1,
        )
        _require_sha256(self.placard_truth_digest, name="placard truth digest")
        _require_sha256(self.literal_truth_digest, name="literal truth digest")
        if tuple(member.member_position for member in self.members) != (0, 1):
            raise EvaluationCensusV2Error("attempt members must occupy canonical positions zero and one")
        if self.members[0].composed_rule_id == self.members[1].composed_rule_id:
            raise EvaluationCensusV2Error("a mirror attempt requires two distinct C identities")
        if {member.noisy_p_target_side for member in self.members} != {0, 1}:
            raise EvaluationCensusV2Error("mirror members must use opposite noisy-P target sides")
        _require_sha256(self.attempt_digest, name="attempt digest")
        if self.attempt_digest != _digest(self._unsigned_obj(), domain=_ATTEMPT_DOMAIN):
            raise EvaluationCensusV2Error("attempt digest is inconsistent")

    def _unsigned_obj(self) -> dict[str, Any]:
        return {
            "formula_stratum": self.formula_stratum,
            "canonical_position": self.canonical_position,
            "placard_rule_id": self.placard_rule_id,
            "placard_truth_digest": self.placard_truth_digest,
            "literal_rule_id": self.literal_rule_id,
            "literal_truth_digest": self.literal_truth_digest,
            "members": [member.as_obj() for member in self.members],
        }

    def as_obj(self) -> dict[str, Any]:
        return {**self._unsigned_obj(), "attempt_digest": self.attempt_digest}


def _identity_accounting(attempts: Iterable[CensusAttemptPlanV1]) -> dict[str, int | bool]:
    rows = tuple(attempts)
    placards = tuple(row.placard_rule_id for row in rows)
    literals = tuple(row.literal_rule_id for row in rows)
    pairs = tuple((row.placard_rule_id, row.literal_rule_id) for row in rows)
    composed = tuple(member.composed_rule_id for row in rows for member in row.members)
    return {
        "placard_identity_occurrence_count": len(placards),
        "distinct_placard_identity_count": len(set(placards)),
        "placard_reuse_occurrence_count": len(placards) - len(set(placards)),
        "literal_identity_occurrence_count": len(literals),
        "distinct_literal_identity_count": len(set(literals)),
        "literal_reuse_occurrence_count": len(literals) - len(set(literals)),
        "placard_literal_pair_occurrence_count": len(pairs),
        "distinct_placard_literal_pair_count": len(set(pairs)),
        "placard_literal_pair_reuse_occurrence_count": len(pairs) - len(set(pairs)),
        "composed_identity_occurrence_count": len(composed),
        "distinct_composed_identity_count": len(set(composed)),
        "composed_reuse_occurrence_count": len(composed) - len(set(composed)),
        "one_shared_placard_literal_pair_per_mirror_attempt": True,
    }


@dataclass(frozen=True, slots=True)
class EvaluationCensusPlanV1:
    source_binding: CensusSourceBindingV1
    catalog_binding: CensusCatalogBindingV1
    generator_binding: CensusGeneratorBindingV1
    generator_seed: str
    attempts_per_formula_stratum: int
    candidate_pool_size: int
    attempts: tuple[CensusAttemptPlanV1, ...]

    def __post_init__(self) -> None:
        _require_sha256(self.generator_seed, name="generator seed")
        _require_integer(
            self.attempts_per_formula_stratum,
            name="attempts per formula stratum",
            minimum=1,
            maximum=MAXIMUM_MIRROR_ATTEMPTS_PER_STRATUM,
        )
        _require_integer(
            self.candidate_pool_size,
            name="candidate pool size",
            minimum=1,
            maximum=MAXIMUM_CANDIDATE_POOL_SIZE,
        )
        expected_count = len(COMPOSED_STRATA) * self.attempts_per_formula_stratum
        if len(self.attempts) != expected_count:
            raise EvaluationCensusV2Error("plan does not contain the complete fixed attempt budget")
        expected_keys = tuple(
            (_stratum_slug(op, negated), position)
            for op, negated in COMPOSED_STRATA
            for position in range(self.attempts_per_formula_stratum)
        )
        observed_keys = tuple(
            (attempt.formula_stratum, attempt.canonical_position) for attempt in self.attempts
        )
        if observed_keys != expected_keys:
            raise EvaluationCensusV2Error(
                "plan attempts are not in complete canonical stratum/position order"
            )
        if len({attempt.attempt_digest for attempt in self.attempts}) != len(self.attempts):
            raise EvaluationCensusV2Error("plan contains duplicate attempt identities")
        c_ids = [member.composed_rule_id for attempt in self.attempts for member in attempt.members]
        if len(c_ids) != len(set(c_ids)):
            raise EvaluationCensusV2Error("planned evaluation C identities must not be reused")
        catalog = build_rule_catalog()
        entries = {entry.rule_id: entry for entry in catalog}
        partitions = build_rule_identity_partitions_v2()
        triples = tuple(
            (
                attempt.placard_rule_id,
                attempt.literal_rule_id,
                member.composed_rule_id,
            )
            for attempt in self.attempts
            for member in attempt.members
        )
        bindings = build_rule_triple_bindings_batch_for_audit_v2(triples)
        for attempt, first_binding, second_binding in zip(
            self.attempts,
            bindings[::2],
            bindings[1::2],
            strict=True,
        ):
            placard = entries[attempt.placard_rule_id]
            literal = entries[attempt.literal_rule_id]
            for member, binding in zip(
                attempt.members,
                (first_binding, second_binding),
                strict=True,
            ):
                composed = entries[member.composed_rule_id]
                if binding.digest != member.triple_digest:
                    raise EvaluationCensusV2Error("planned triple digest differs from exact binding")
                if (
                    placard.truth_digest != attempt.placard_truth_digest
                    or literal.truth_digest != attempt.literal_truth_digest
                    or composed.truth_digest != member.composed_truth_digest
                ):
                    raise EvaluationCensusV2Error("planned truth identity differs from the catalog")
                if partitions.for_entry(composed) != "evaluation":
                    raise EvaluationCensusV2Error("planned C identity is outside the evaluation partition")
                if _stratum_slug(*composed_stratum_v2(composed)) != attempt.formula_stratum:
                    raise EvaluationCensusV2Error("planned C identity is in the wrong formula stratum")
                if min(joint_truth_cell_counts_v2(placard, literal, composed)) < (
                    REGISTERED_POPULATION_RESERVE_PER_JOINT_CELL
                ):
                    raise EvaluationCensusV2Error("planned triple lacks the registered 27-scene reserve")

    @property
    def uses_production_attempt_budget(self) -> bool:
        return (
            self.attempts_per_formula_stratum == PRODUCTION_MIRROR_ATTEMPTS_PER_STRATUM
            and self.candidate_pool_size == DEFAULT_CANDIDATE_POOL_SIZE
        )

    def _unsigned_obj(self) -> dict[str, Any]:
        return {
            "schema_version": EVALUATION_CENSUS_SCHEMA_VERSION,
            "plan_kind": _PLAN_KIND,
            "status": "prospective_outcome_free_nonauthorizing",
            "authorization": dict(_AUTHORIZATION),
            "source_binding": self.source_binding.as_obj(),
            "catalog_binding": self.catalog_binding.as_obj(),
            "generator_binding": self.generator_binding.as_obj(),
            "generator_seed": self.generator_seed,
            "fixed_budget": {
                "formula_stratum_count": len(COMPOSED_STRATA),
                "attempts_per_formula_stratum": self.attempts_per_formula_stratum,
                "mirror_attempt_count": len(self.attempts),
                "draws_per_mirror_attempt": 8,
                "planned_draw_count": len(self.attempts) * 8,
                "production_attempts_per_formula_stratum": PRODUCTION_MIRROR_ATTEMPTS_PER_STRATUM,
                "uses_production_attempt_budget": self.uses_production_attempt_budget,
                "candidate_pool_size": self.candidate_pool_size,
                "no_early_stop": True,
            },
            "identity_accounting": _identity_accounting(self.attempts),
            "selection_boundary": {
                "positive_m_q_quota": None,
                "matched_bank_size": None,
                "matcher_specification": None,
            },
            "attempts": [attempt.as_obj() for attempt in self.attempts],
        }

    @property
    def digest(self) -> str:
        return _digest(self._unsigned_obj(), domain=_PLAN_DOMAIN)

    def as_obj(self) -> dict[str, Any]:
        value = {**self._unsigned_obj(), "prospective_plan_digest": self.digest}
        _assert_plan_has_no_observed_payload(value)
        return value


def _assert_plan_has_no_observed_payload(value: object) -> None:
    if type(value) is dict:
        for key, item in cast(dict[str, Any], value).items():
            if key in _PLAN_FORBIDDEN_KEYS:
                raise EvaluationCensusV2Error(f"prospective plan contains forbidden observed key: {key}")
            _assert_plan_has_no_observed_payload(item)
    elif type(value) is list:
        for item in value:
            _assert_plan_has_no_observed_payload(item)


def _family_pools() -> tuple[
    tuple[CatalogEntry, ...],
    tuple[CatalogEntry, ...],
    dict[tuple[BinaryOp, int], tuple[CatalogEntry, ...]],
]:
    placards: list[CatalogEntry] = []
    literals: list[CatalogEntry] = []
    composed: dict[tuple[BinaryOp, int], list[CatalogEntry]] = {stratum: [] for stratum in COMPOSED_STRATA}
    for entry in build_rule_catalog():
        family = classify_catalog_identity_v2(entry)
        if family == "placard_literal":
            placards.append(entry)
        elif family == "one_literal_piece":
            literals.append(entry)
        elif family == "composed_two_literal_piece":
            composed[composed_stratum_v2(entry)].append(entry)
    return (
        tuple(placards),
        tuple(literals),
        {key: tuple(value) for key, value in composed.items()},
    )


def _build_attempt_specs(
    generator_seed: str,
    attempts_per_formula_stratum: int,
) -> tuple[CensusAttemptPlanV1, ...]:
    placards, literals, composed_by_stratum = _family_pools()
    raw: list[
        tuple[
            str,
            int,
            CatalogEntry,
            CatalogEntry,
            CatalogEntry,
            CatalogEntry,
            int,
        ]
    ] = []
    triples: list[tuple[str, str, str]] = []
    partitions = build_rule_identity_partitions_v2()
    for op, negated in COMPOSED_STRATA:
        slug = _stratum_slug(op, negated)
        ranked_composed = tuple(
            entry
            for entry in sorted(
                composed_by_stratum[(op, negated)],
                key=lambda entry: (
                    _hash_parts(_IDENTITY_ORDER_DOMAIN, generator_seed, slug, "C", entry.rule_id),
                    entry.index,
                ),
            )
            if partitions.for_entry(entry) == "evaluation"
        )
        needed = 2 * attempts_per_formula_stratum
        if len(ranked_composed) < needed:
            raise EvaluationCensusV2Error(f"formula stratum {slug} lacks {needed} C identities")
        used_composed: set[str] = set()
        for position in range(attempts_per_formula_stratum):
            placard = min(
                placards,
                key=lambda entry: (
                    _hash_parts(
                        _IDENTITY_ORDER_DOMAIN,
                        generator_seed,
                        slug,
                        position,
                        "P",
                        entry.rule_id,
                    ),
                    entry.index,
                ),
            )
            literal = min(
                literals,
                key=lambda entry: (
                    _hash_parts(
                        _IDENTITY_ORDER_DOMAIN,
                        generator_seed,
                        slug,
                        position,
                        "Q",
                        entry.rule_id,
                    ),
                    entry.index,
                ),
            )
            eligible = tuple(
                entry
                for entry in ranked_composed
                if entry.rule_id not in used_composed
                and min(joint_truth_cell_counts_v2(placard, literal, entry))
                >= REGISTERED_POPULATION_RESERVE_PER_JOINT_CELL
            )
            if len(eligible) < 2:
                raise EvaluationCensusV2Error(
                    f"formula stratum {slug} position {position} lacks two unused "
                    "evaluation C identities at the registered reserve"
                )
            first, second = eligible[:2]
            used_composed.update((first.rule_id, second.rule_id))
            side = (
                int.from_bytes(
                    _hash_parts(_IDENTITY_ORDER_DOMAIN, generator_seed, slug, position, "side")[:2],
                    "big",
                )
                % 2
            )
            raw.append((slug, position, placard, literal, first, second, side))
            triples.extend(
                (
                    (placard.rule_id, literal.rule_id, first.rule_id),
                    (placard.rule_id, literal.rule_id, second.rule_id),
                )
            )
    bindings = build_rule_triple_bindings_batch_for_audit_v2(triples)
    if len(bindings) != 2 * len(raw):
        raise AssertionError("batch triple binding count changed")
    attempts: list[CensusAttemptPlanV1] = []
    for offset, (slug, position, placard, literal, first, second, side) in enumerate(raw):
        first_binding = bindings[2 * offset]
        second_binding = bindings[2 * offset + 1]
        members = (
            CensusMemberPlanV1(
                0,
                first.rule_id,
                first.truth_digest,
                first_binding.digest,
                side,
            ),
            CensusMemberPlanV1(
                1,
                second.rule_id,
                second.truth_digest,
                second_binding.digest,
                1 - side,
            ),
        )
        unsigned = {
            "formula_stratum": slug,
            "canonical_position": position,
            "placard_rule_id": placard.rule_id,
            "placard_truth_digest": placard.truth_digest,
            "literal_rule_id": literal.rule_id,
            "literal_truth_digest": literal.truth_digest,
            "members": [member.as_obj() for member in members],
        }
        attempts.append(
            CensusAttemptPlanV1(
                slug,
                position,
                placard.rule_id,
                placard.truth_digest,
                literal.rule_id,
                literal.truth_digest,
                members,
                _digest(unsigned, domain=_ATTEMPT_DOMAIN),
            )
        )
    return tuple(attempts)


def build_evaluation_census_plan_v1(
    generator_seed: str,
    *,
    attempts_per_formula_stratum: int = PRODUCTION_MIRROR_ATTEMPTS_PER_STRATUM,
    candidate_pool_size: int = DEFAULT_CANDIDATE_POOL_SIZE,
) -> EvaluationCensusPlanV1:
    """Build the outcome-free, exact-budget prospective census plan."""

    _require_sha256(generator_seed, name="generator seed")
    _require_integer(
        attempts_per_formula_stratum,
        name="attempts per formula stratum",
        minimum=1,
        maximum=MAXIMUM_MIRROR_ATTEMPTS_PER_STRATUM,
    )
    _require_integer(
        candidate_pool_size,
        name="candidate pool size",
        minimum=1,
        maximum=MAXIMUM_CANDIDATE_POOL_SIZE,
    )
    return EvaluationCensusPlanV1(
        _current_source_binding(),
        _current_catalog_binding(),
        _current_generator_binding(),
        generator_seed,
        attempts_per_formula_stratum,
        candidate_pool_size,
        _build_attempt_specs(generator_seed, attempts_per_formula_stratum),
    )


def serialize_evaluation_census_plan_v1(plan: EvaluationCensusPlanV1) -> str:
    if type(plan) is not EvaluationCensusPlanV1:
        raise TypeError("plan must be an EvaluationCensusPlanV1")
    return _dump_json(plan.as_obj()) + "\n"


def evaluation_census_plan_v1_from_obj(
    value: object,
    *,
    expected_digest: str,
) -> EvaluationCensusPlanV1:
    _require_sha256(expected_digest, name="expected plan digest")
    obj = _require_mapping(
        value,
        (
            "schema_version",
            "plan_kind",
            "status",
            "authorization",
            "source_binding",
            "catalog_binding",
            "generator_binding",
            "generator_seed",
            "fixed_budget",
            "identity_accounting",
            "selection_boundary",
            "attempts",
            "prospective_plan_digest",
        ),
        name="prospective census plan",
    )
    if obj["schema_version"] != EVALUATION_CENSUS_SCHEMA_VERSION or obj["plan_kind"] != _PLAN_KIND:
        raise EvaluationCensusV2Error("unknown census-plan schema or kind")
    if obj["status"] != "prospective_outcome_free_nonauthorizing":
        raise EvaluationCensusV2Error("prospective census status changed")
    if obj["authorization"] != _AUTHORIZATION:
        raise EvaluationCensusV2Error("prospective census authorization must remain false")
    budget = _require_mapping(
        obj["fixed_budget"],
        (
            "formula_stratum_count",
            "attempts_per_formula_stratum",
            "mirror_attempt_count",
            "draws_per_mirror_attempt",
            "planned_draw_count",
            "production_attempts_per_formula_stratum",
            "uses_production_attempt_budget",
            "candidate_pool_size",
            "no_early_stop",
        ),
        name="fixed budget",
    )
    attempts_per = _require_integer(
        budget["attempts_per_formula_stratum"],
        name="attempts per formula stratum",
        minimum=1,
        maximum=MAXIMUM_MIRROR_ATTEMPTS_PER_STRATUM,
    )
    pool_size = _require_integer(
        budget["candidate_pool_size"],
        name="candidate pool size",
        minimum=1,
        maximum=MAXIMUM_CANDIDATE_POOL_SIZE,
    )
    rebuilt = build_evaluation_census_plan_v1(
        _require_sha256(obj["generator_seed"], name="generator seed"),
        attempts_per_formula_stratum=attempts_per,
        candidate_pool_size=pool_size,
    )
    if _dump_json(value) != _dump_json(rebuilt.as_obj()):
        raise EvaluationCensusV2Error("prospective census plan differs from exact rederivation")
    if obj["prospective_plan_digest"] != rebuilt.digest or rebuilt.digest != expected_digest:
        raise EvaluationCensusV2Error("prospective census plan digest differs from expected")
    return rebuilt


def parse_evaluation_census_plan_v1(
    text: str,
    *,
    expected_digest: str,
    require_canonical: bool = True,
) -> EvaluationCensusPlanV1:
    plan = evaluation_census_plan_v1_from_obj(_load_json(text), expected_digest=expected_digest)
    if require_canonical and serialize_evaluation_census_plan_v1(plan) != text:
        raise EvaluationCensusV2Error("prospective census plan is not canonical newline-terminated JSON")
    return plan


@dataclass(frozen=True, slots=True)
class CensusRankingRowV1:
    total_rank: int
    source_catalog_index: int
    rule_id: str
    truth_digest: str
    supported_family: str
    correct_demonstrations: int
    error_demonstrations: int
    literal_count: int
    explicit_negation_count: int

    def __post_init__(self) -> None:
        _require_integer(
            self.total_rank,
            name="total rank",
            minimum=1,
            maximum=SUPPORTED_RANKING_ROW_COUNT,
        )
        _require_integer(self.source_catalog_index, name="source catalog index")
        _require_sha256(self.truth_digest, name="ranking truth digest")
        if self.supported_family not in {
            "placard_literal",
            "one_literal_piece",
            "composed_two_literal_piece",
        }:
            raise EvaluationCensusV2Error("ranking row has an unsupported family")
        _require_integer(
            self.correct_demonstrations,
            name="correct demonstrations",
            maximum=10,
        )
        _require_integer(
            self.error_demonstrations,
            name="error demonstrations",
            maximum=10,
        )
        if self.correct_demonstrations + self.error_demonstrations != 10:
            raise EvaluationCensusV2Error("ranking accuracy counts do not sum to ten")
        _require_integer(self.literal_count, name="literal count", minimum=1, maximum=2)
        _require_integer(
            self.explicit_negation_count,
            name="explicit negation count",
            maximum=self.literal_count,
        )

    def as_obj(self) -> dict[str, Any]:
        return {
            "total_rank": self.total_rank,
            "source_catalog_index": self.source_catalog_index,
            "rule_id": self.rule_id,
            "truth_digest": self.truth_digest,
            "supported_family": self.supported_family,
            "correct_demonstrations": self.correct_demonstrations,
            "error_demonstrations": self.error_demonstrations,
            "literal_count": self.literal_count,
            "explicit_negation_count": self.explicit_negation_count,
        }


def _complete_ranking_rows_digest(rows: Iterable[CensusRankingRowV1]) -> str:
    digest = hashlib.sha256()
    digest.update(_COMPLETE_RANKING_ROWS_DOMAIN)
    row_count = 0
    for row in rows:
        row_count += 1
        digest.update(_dump_json(row.as_obj()).encode("ascii"))
        digest.update(b"\n")
    digest.update(row_count.to_bytes(8, "big"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class CensusRankingTableV1:
    catalog_digest: str
    supported_catalog_digest: str
    correct_counts_by_supported_index: tuple[int, ...]
    total_rank_source_catalog_indices: tuple[int, ...]
    complete_ranking_rows_digest: str

    def __post_init__(self) -> None:
        _require_sha256(self.catalog_digest, name="ranking catalog digest")
        _require_sha256(self.supported_catalog_digest, name="ranking supported-catalog digest")
        _require_sha256(self.complete_ranking_rows_digest, name="complete ranking rows digest")
        contract = build_supported_catalog_contract_v2()
        catalog = build_rule_catalog()
        if self.catalog_digest != catalog.digest or (
            self.supported_catalog_digest != contract.supported_catalog_digest
        ):
            raise EvaluationCensusV2Error("ranking table catalog binding differs from current source")
        if len(self.correct_counts_by_supported_index) != SUPPORTED_RANKING_ROW_COUNT:
            raise EvaluationCensusV2Error("ranking table must retain all 6970 accuracy counts")
        for count in self.correct_counts_by_supported_index:
            _require_integer(count, name="ranking correct count", maximum=10)
        for index in self.total_rank_source_catalog_indices:
            _require_integer(
                index,
                name="total-rank source catalog index",
                maximum=len(catalog) - 1,
            )
        if len(self.total_rank_source_catalog_indices) != SUPPORTED_RANKING_ROW_COUNT or set(
            self.total_rank_source_catalog_indices
        ) != set(contract.supported_indices):
            raise EvaluationCensusV2Error("ranking table must retain one total rank for every identity")
        correct_by_index = dict(
            zip(contract.supported_indices, self.correct_counts_by_supported_index, strict=True)
        )
        expected_order = tuple(
            sorted(
                contract.supported_indices,
                key=lambda index: (
                    -correct_by_index[index],
                    *_rule_complexity(catalog[index]),
                    catalog[index].rule_id,
                ),
            )
        )
        if self.total_rank_source_catalog_indices != expected_order:
            raise EvaluationCensusV2Error("ranking table total-rank permutation is inconsistent")
        if self.complete_ranking_rows_digest != _complete_ranking_rows_digest(self._iter_materialized_rows()):
            raise EvaluationCensusV2Error("complete ranking-row digest is inconsistent")

    @property
    def row_count(self) -> int:
        return len(self.correct_counts_by_supported_index)

    def _iter_materialized_rows(self) -> Iterable[CensusRankingRowV1]:
        catalog = build_rule_catalog()
        contract = build_supported_catalog_contract_v2()
        correct_by_index = dict(
            zip(contract.supported_indices, self.correct_counts_by_supported_index, strict=True)
        )
        for rank, index in enumerate(self.total_rank_source_catalog_indices, start=1):
            entry = catalog[index]
            correct = correct_by_index[index]
            literal_count, negated_count = _rule_complexity(entry)
            yield CensusRankingRowV1(
                total_rank=rank,
                source_catalog_index=index,
                rule_id=entry.rule_id,
                truth_digest=entry.truth_digest,
                supported_family=classify_catalog_identity_v2(entry),
                correct_demonstrations=correct,
                error_demonstrations=10 - correct,
                literal_count=literal_count,
                explicit_negation_count=negated_count,
            )

    def materialize_rows(self) -> tuple[CensusRankingRowV1, ...]:
        """Materialize all committed rows for inspection without storing duplicates."""

        return tuple(self._iter_materialized_rows())

    def correct_count(self, source_catalog_index: int) -> int:
        contract = build_supported_catalog_contract_v2()
        try:
            position = contract.supported_indices.index(source_catalog_index)
        except ValueError as exc:
            raise EvaluationCensusV2Error("requested rank identity is outside the supported catalog") from exc
        return self.correct_counts_by_supported_index[position]

    def _unsigned_obj(self) -> dict[str, Any]:
        return {
            "schema_version": EVALUATION_CENSUS_SCHEMA_VERSION,
            "catalog_digest": self.catalog_digest,
            "supported_catalog_digest": self.supported_catalog_digest,
            "row_count": self.row_count,
            "ranking_order": [
                "correct_demonstrations_descending",
                "literal_count_ascending",
                "explicit_negation_count_ascending",
                "rule_id_ascending",
            ],
            "compact_materialization": {
                "correct_count_order": "ascending v2-supported source catalog indices",
                "correct_counts_by_supported_index": list(self.correct_counts_by_supported_index),
                "total_rank_source_catalog_indices": list(self.total_rank_source_catalog_indices),
                "full_rows_rederived_from_bound_catalog": True,
                "complete_ranking_rows_stream_encoding": (
                    "domain, canonical row JSON plus newline for ranks 1..6970, final u64 row count"
                ),
                "complete_ranking_rows_digest": self.complete_ranking_rows_digest,
            },
        }

    @property
    def digest(self) -> str:
        return _digest(self._unsigned_obj(), domain=_RANKING_TABLE_DOMAIN)

    def as_obj(self) -> dict[str, Any]:
        return {**self._unsigned_obj(), "ranking_table_digest": self.digest}


@dataclass(frozen=True, slots=True)
class FirstQuerySplitFrequencyV1:
    smaller_branch_size: int
    larger_branch_size: int
    legal_scene_count: int

    def as_obj(self) -> dict[str, int]:
        return {
            "smaller_branch_size": self.smaller_branch_size,
            "larger_branch_size": self.larger_branch_size,
            "legal_scene_count": self.legal_scene_count,
        }


@dataclass(frozen=True, slots=True)
class CensusDifficultyEvidenceV1:
    status: str
    minimax_depth: int | None
    greedy_official_recovery_query_count: int | None
    best_first_query_scene_index: int | None
    best_first_query_branch_sizes: tuple[int, int] | None
    first_query_split_multiset: tuple[FirstQuerySplitFrequencyV1, ...]
    root_informative_scene_count: int | None
    first_query_split_multiset_digest: str | None
    greedy_branch_signature: tuple[tuple[int, int, int, int, int], ...]
    query_report_digest: str | None

    def as_obj(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "minimax_depth": self.minimax_depth,
            "greedy_official_recovery_query_count": self.greedy_official_recovery_query_count,
            "best_first_query_scene_index": self.best_first_query_scene_index,
            "best_first_query_branch_sizes": (
                None
                if self.best_first_query_branch_sizes is None
                else list(self.best_first_query_branch_sizes)
            ),
            "first_query_split_multiset": [row.as_obj() for row in self.first_query_split_multiset],
            "root_informative_scene_count": self.root_informative_scene_count,
            "first_query_split_multiset_digest": self.first_query_split_multiset_digest,
            "greedy_branch_signature_encoding": [
                "scene_index",
                "before_count",
                "rejected_count",
                "accepted_count",
                "after_count",
            ],
            "greedy_branch_signature": [list(row) for row in self.greedy_branch_signature],
            "query_report_digest": self.query_report_digest,
        }


@dataclass(frozen=True, slots=True)
class CensusSalienceEvidenceV1:
    status: str
    placard_rule_accuracy: int
    placard_rule_live: bool
    literal_rule_accuracy: int
    literal_rule_live: bool
    live_one_literal_rule_ids: tuple[str, ...]
    maximum_other_one_literal_accuracy: int
    criterion: str
    reason_codes: tuple[str, ...]

    def as_obj(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "placard_rule_accuracy": self.placard_rule_accuracy,
            "placard_rule_live": self.placard_rule_live,
            "literal_rule_accuracy": self.literal_rule_accuracy,
            "literal_rule_live": self.literal_rule_live,
            "live_one_literal_rule_ids": list(self.live_one_literal_rule_ids),
            "maximum_other_one_literal_accuracy": self.maximum_other_one_literal_accuracy,
            "criterion": self.criterion,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True, slots=True)
class CensusConstructedOpeningV1:
    record_kind: str
    schedule_variant: ScheduleVariant
    schedule: dict[str, Any]
    construction_scene_indices: tuple[int, ...]
    construction_joint_cells: tuple[str, ...]
    official_labels: tuple[bool, ...]
    version_space_rule_ids: tuple[str, ...]
    version_space_truth_digests: tuple[str, ...]
    observed_m: int
    observed_q: int | None
    ranking_table_digest: str
    salience: CensusSalienceEvidenceV1
    difficulty: CensusDifficultyEvidenceV1
    disposition: OpeningDisposition
    reason_codes: tuple[str, ...]

    def as_obj(self) -> dict[str, Any]:
        return {
            "record_kind": self.record_kind,
            "schedule_variant": self.schedule_variant,
            "schedule": self.schedule,
            "construction_scene_indices": list(self.construction_scene_indices),
            "construction_joint_cells": list(self.construction_joint_cells),
            "official_labels": list(self.official_labels),
            "version_space_rule_ids": list(self.version_space_rule_ids),
            "version_space_truth_digests": list(self.version_space_truth_digests),
            "observed_m": self.observed_m,
            "observed_q": self.observed_q,
            "ranking_table_digest": self.ranking_table_digest,
            "salience": self.salience.as_obj(),
            "difficulty": self.difficulty.as_obj(),
            "disposition": self.disposition,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True, slots=True)
class CensusPreOpeningFailureV1:
    record_kind: str
    schedule_variant: ScheduleVariant
    stage: ConstructionStage
    reason_code: str
    selected_scene_count: int
    required_demonstration_count: int = 10
    ranking_table_materialized: bool = False

    def as_obj(self) -> dict[str, Any]:
        return {
            "record_kind": self.record_kind,
            "schedule_variant": self.schedule_variant,
            "stage": self.stage,
            "reason_code": self.reason_code,
            "selected_scene_count": self.selected_scene_count,
            "required_demonstration_count": self.required_demonstration_count,
            "ranking_table_materialized": self.ranking_table_materialized,
        }


CensusOpeningRecordV1: TypeAlias = CensusConstructedOpeningV1 | CensusPreOpeningFailureV1


@dataclass(frozen=True, slots=True)
class CensusMemberObservationV1:
    member_plan: CensusMemberPlanV1
    openings: tuple[CensusOpeningRecordV1, ...]
    disposition: GroupDisposition
    observed_m: int | None
    observed_q: int | None
    exact_minimax_depth: int | None
    best_first_query_branch_sizes: tuple[int, int] | None
    reason_codes: tuple[str, ...]

    def as_obj(self) -> dict[str, Any]:
        return {
            "member_plan": self.member_plan.as_obj(),
            "openings": [opening.as_obj() for opening in self.openings],
            "disposition": self.disposition,
            "observed_m": self.observed_m,
            "observed_q": self.observed_q,
            "exact_minimax_depth": self.exact_minimax_depth,
            "best_first_query_branch_sizes": (
                None
                if self.best_first_query_branch_sizes is None
                else list(self.best_first_query_branch_sizes)
            ),
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True, slots=True)
class CensusAttemptObservationV1:
    attempt_plan: CensusAttemptPlanV1
    members: tuple[CensusMemberObservationV1, CensusMemberObservationV1]
    disposition: AttemptDisposition
    reason_codes: tuple[str, ...]

    def as_obj(self) -> dict[str, Any]:
        return {
            "attempt_plan": self.attempt_plan.as_obj(),
            "members": [member.as_obj() for member in self.members],
            "disposition": self.disposition,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True, slots=True)
class CensusCellSummaryV1:
    m: int
    q: int
    constructed_opening_count: int
    salient_opening_count: int
    cell_witness_opening_count: int
    candidate_group_count: int
    complete_census_pair_count: int

    def as_obj(self) -> dict[str, int | None]:
        return {
            "m": self.m,
            "q": self.q,
            "positive_quota": None,
            "constructed_opening_count": self.constructed_opening_count,
            "salient_opening_count": self.salient_opening_count,
            "cell_witness_opening_count": self.cell_witness_opening_count,
            "candidate_group_count": self.candidate_group_count,
            "complete_census_pair_count": self.complete_census_pair_count,
        }


@dataclass(frozen=True)
class EvaluationCensusReportV1:
    source_binding: CensusSourceBindingV1
    catalog_binding: CensusCatalogBindingV1
    generator_binding: CensusGeneratorBindingV1
    prospective_plan_digest: str
    prospective_plan_bytes_sha256: str
    prospective_plan_byte_count: int
    prospective_candidate_pool_size: int
    attempts: tuple[CensusAttemptObservationV1, ...]
    ranking_tables: tuple[CensusRankingTableV1, ...]
    cell_summary: tuple[CensusCellSummaryV1, ...]
    constructed_opening_count: int
    preopening_failure_count: int
    candidate_group_count: int
    complete_census_pair_count: int
    production_attempt_budget_complete: bool

    def __post_init__(self) -> None:
        _require_sha256(self.prospective_plan_digest, name="prospective plan digest")
        _require_sha256(self.prospective_plan_bytes_sha256, name="prospective plan bytes sha256")
        _require_integer(self.prospective_plan_byte_count, name="prospective plan byte count", minimum=1)
        _require_integer(
            self.prospective_candidate_pool_size,
            name="prospective candidate pool size",
            minimum=1,
            maximum=MAXIMUM_CANDIDATE_POOL_SIZE,
        )
        expected_cells = tuple((m, q) for m in range(8, 17) for q in range(1, 5))
        if tuple((row.m, row.q) for row in self.cell_summary) != expected_cells:
            raise EvaluationCensusV2Error("observed summary must enumerate all 36 m/q cells")
        if any(len(member.openings) != 4 for attempt in self.attempts for member in attempt.members):
            raise EvaluationCensusV2Error("each observed member must preserve four geometry records")
        openings = tuple(
            opening for attempt in self.attempts for member in attempt.members for opening in member.openings
        )
        if len(openings) != len(self.attempts) * 8:
            raise EvaluationCensusV2Error("observed report must preserve eight records per attempt")
        expected_counts = (
            sum(type(opening) is CensusConstructedOpeningV1 for opening in openings),
            sum(type(opening) is CensusPreOpeningFailureV1 for opening in openings),
            sum(
                member.disposition == "candidate_group"
                for attempt in self.attempts
                for member in attempt.members
            ),
            sum(attempt.disposition == "complete_census_pair" for attempt in self.attempts),
        )
        if expected_counts != (
            self.constructed_opening_count,
            self.preopening_failure_count,
            self.candidate_group_count,
            self.complete_census_pair_count,
        ):
            raise EvaluationCensusV2Error("observed report accounting differs from its full ledger")
        if self.cell_summary != _cell_summaries(self.attempts):
            raise EvaluationCensusV2Error("observed 36-cell summary differs from exact recomputation")
        digests = tuple(table.digest for table in self.ranking_tables)
        if digests != tuple(sorted(set(digests))):
            raise EvaluationCensusV2Error("ranking tables must be unique and digest-sorted")
        referenced_digests = {
            opening.ranking_table_digest
            for attempt in self.attempts
            for member in attempt.members
            for opening in member.openings
            if type(opening) is CensusConstructedOpeningV1
        }
        if referenced_digests != set(digests):
            raise EvaluationCensusV2Error(
                "ranking tables must exactly cover every constructed opening reference"
            )
        expected_production = (
            self.prospective_candidate_pool_size == DEFAULT_CANDIDATE_POOL_SIZE
            and len(self.attempts) == len(COMPOSED_STRATA) * PRODUCTION_MIRROR_ATTEMPTS_PER_STRATUM
            and Counter(attempt.attempt_plan.formula_stratum for attempt in self.attempts)
            == {
                _stratum_slug(*stratum): PRODUCTION_MIRROR_ATTEMPTS_PER_STRATUM for stratum in COMPOSED_STRATA
            }
        )
        if type(self.production_attempt_budget_complete) is not bool or (
            self.production_attempt_budget_complete != expected_production
        ):
            raise EvaluationCensusV2Error("production-budget completion status is inconsistent")

    @property
    def status(self) -> str:
        if self.production_attempt_budget_complete:
            return "observed_complete_production_census_ledger_nonauthorizing"
        return "observed_complete_engineering_fixture_ledger_nonauthorizing"

    def _claim_boundary_obj(self) -> dict[str, bool]:
        return {
            "engineering_opening_evidence_described": True,
            "full_production_opening_feasibility_census_described": (self.production_attempt_budget_complete),
            **_CLAIM_BOUNDARY,
        }

    def _unsigned_obj(self) -> dict[str, Any]:
        return {
            "schema_version": EVALUATION_CENSUS_SCHEMA_VERSION,
            "report_kind": _REPORT_KIND,
            "status": self.status,
            "authorization": dict(_AUTHORIZATION),
            "source_binding": self.source_binding.as_obj(),
            "catalog_binding": self.catalog_binding.as_obj(),
            "generator_binding": self.generator_binding.as_obj(),
            "prospective_plan_binding": {
                "prospective_plan_digest": self.prospective_plan_digest,
                "exact_plan_bytes_sha256": self.prospective_plan_bytes_sha256,
                "exact_plan_byte_count": self.prospective_plan_byte_count,
                "candidate_pool_size": self.prospective_candidate_pool_size,
            },
            "attempt_accounting": {
                "mirror_attempt_count": len(self.attempts),
                "opening_record_count": sum(
                    len(member.openings) for attempt in self.attempts for member in attempt.members
                ),
                "constructed_opening_count": self.constructed_opening_count,
                "preopening_failure_count": self.preopening_failure_count,
                "candidate_group_count": self.candidate_group_count,
                "complete_census_pair_count": self.complete_census_pair_count,
                "production_attempt_budget_complete": self.production_attempt_budget_complete,
                "identity_accounting": _identity_accounting(
                    attempt.attempt_plan for attempt in self.attempts
                ),
                "all_planned_attempts_preserved": True,
                "early_stop_used": False,
            },
            "attempts": [attempt.as_obj() for attempt in self.attempts],
            "ranking_tables": [table.as_obj() for table in self.ranking_tables],
            "observed_m_q_summary": [row.as_obj() for row in self.cell_summary],
            "selection_boundary": {
                "positive_m_q_quota": None,
                "matched_bank_size": None,
                "matcher_specification": None,
            },
            "claim_boundary": self._claim_boundary_obj(),
        }

    @property
    def digest(self) -> str:
        return _digest(self._unsigned_obj(), domain=_REPORT_DOMAIN)

    def as_obj(self) -> dict[str, Any]:
        return {**self._unsigned_obj(), "observed_report_digest": self.digest}


@dataclass(frozen=True, slots=True)
class AssessedEvaluationOpeningV1:
    """One independently rederived completed opening plus its full rank table."""

    record: CensusConstructedOpeningV1
    ranking_table: CensusRankingTableV1

    def __post_init__(self) -> None:
        if self.record.ranking_table_digest != self.ranking_table.digest:
            raise EvaluationCensusV2Error("opening record and ranking table are not bound")

    def as_obj(self) -> dict[str, Any]:
        return {
            "opening_record": self.record.as_obj(),
            "ranking_table": self.ranking_table.as_obj(),
        }


@lru_cache(maxsize=1)
def _catalog_by_rule_id() -> dict[str, CatalogEntry]:
    return {entry.rule_id: entry for entry in build_rule_catalog()}


def _resolve_entry(rule_id: str, *, family: str) -> CatalogEntry:
    try:
        entry = _catalog_by_rule_id()[rule_id]
    except KeyError as exc:
        raise EvaluationCensusV2Error(f"unknown catalog rule id: {rule_id!r}") from exc
    if classify_catalog_identity_v2(entry) != family:
        raise EvaluationCensusV2Error(f"{rule_id} is not in required family {family}")
    return entry


def _rule_complexity(entry: CatalogEntry) -> tuple[int, int]:
    rule = entry.rule
    if type(rule) is RuleLiteral:
        return 1, int(rule.negated)
    if type(rule) is not BinaryRule:  # pragma: no cover - public grammar is closed
        raise EvaluationCensusV2Error("unknown rule type reached syntactic ranking")
    return 2, sum(int(literal.negated) for literal in rule.args)


def _variant_parts(variant: ScheduleVariant) -> tuple[EvidenceGeometryV2, OfficialTargetSideV2 | None]:
    if variant == "a0_b0":
        slug, side = "a0_b0", None
    elif variant == "a0_b2":
        slug, side = "a0_b2", None
    elif variant == "a1_b0_y0":
        slug, side = "a1_b0", OfficialTargetSideV2.NONFITTING
    elif variant == "a1_b0_y1":
        slug, side = "a1_b0", OfficialTargetSideV2.FITTING
    elif variant == "a1_b2_y0":
        slug, side = "a1_b2", OfficialTargetSideV2.NONFITTING
    elif variant == "a1_b2_y1":
        slug, side = "a1_b2", OfficialTargetSideV2.FITTING
    else:  # pragma: no cover - Literal plus runtime parser guard
        raise EvaluationCensusV2Error(f"unknown schedule variant: {variant!r}")
    geometry = next(item for item in EVIDENCE_GEOMETRIES if item.slug == slug)
    return geometry, side


def _member_schedule_variants(side: int) -> tuple[ScheduleVariant, ...]:
    _require_integer(side, name="noisy P target side", maximum=1)
    return cast(
        tuple[ScheduleVariant, ...],
        (
            "a0_b0",
            f"a1_b0_y{side}",
            "a0_b2",
            f"a1_b2_y{side}",
        ),
    )


def _joint_cell_index(
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


def _ranking_table(
    composed: CatalogEntry,
    construction_scene_indices: tuple[int, ...],
) -> CensusRankingTableV1:
    if len(construction_scene_indices) != 10 or len(set(construction_scene_indices)) != 10:
        raise EvaluationCensusV2Error("a ranking table requires ten distinct construction scenes")
    selected_mask = sum(1 << scene_index for scene_index in construction_scene_indices)
    contract = build_supported_catalog_contract_v2()
    catalog = build_rule_catalog()
    correct_counts = tuple(
        10 - ((catalog[index].truth.bits ^ composed.truth.bits) & selected_mask).bit_count()
        for index in contract.supported_indices
    )
    correct_by_index = dict(zip(contract.supported_indices, correct_counts, strict=True))
    ranked_indices = tuple(
        sorted(
            contract.supported_indices,
            key=lambda index: (
                -correct_by_index[index],
                *_rule_complexity(catalog[index]),
                catalog[index].rule_id,
            ),
        )
    )

    def rows() -> Iterable[CensusRankingRowV1]:
        for rank, index in enumerate(ranked_indices, start=1):
            entry = catalog[index]
            correct = correct_by_index[index]
            literal_count, negated_count = _rule_complexity(entry)
            yield CensusRankingRowV1(
                total_rank=rank,
                source_catalog_index=index,
                rule_id=entry.rule_id,
                truth_digest=entry.truth_digest,
                supported_family=classify_catalog_identity_v2(entry),
                correct_demonstrations=correct,
                error_demonstrations=10 - correct,
                literal_count=literal_count,
                explicit_negation_count=negated_count,
            )

    return CensusRankingTableV1(
        catalog.digest,
        contract.supported_catalog_digest,
        correct_counts,
        ranked_indices,
        _complete_ranking_rows_digest(rows()),
    )


def _salience_evidence(
    ranking: CensusRankingTableV1,
    *,
    placard_rule_id: str,
    literal_rule_id: str,
    schedule_variant: ScheduleVariant,
) -> CensusSalienceEvidenceV1:
    geometry, _side = _variant_parts(schedule_variant)
    catalog = build_rule_catalog()
    contract = build_supported_catalog_contract_v2()
    placard = _resolve_entry(placard_rule_id, family="placard_literal")
    literal = _resolve_entry(literal_rule_id, family="one_literal_piece")
    placard_accuracy = ranking.correct_count(placard.index)
    literal_accuracy = ranking.correct_count(literal.index)
    one_literal_accuracies = tuple(
        (catalog[index], correct)
        for index, correct in zip(
            contract.supported_indices,
            ranking.correct_counts_by_supported_index,
            strict=True,
        )
        if classify_catalog_identity_v2(catalog[index]) == "one_literal_piece"
    )
    live_one_literal_ids = tuple(
        sorted(entry.rule_id for entry, correct in one_literal_accuracies if correct == 10)
    )
    maximum_other_accuracy = max(
        correct for entry, correct in one_literal_accuracies if entry.rule_id != literal_rule_id
    )
    reasons: list[str] = []
    a_noisy = geometry.alternative_a_errors.value == 1
    b_noisy = geometry.alternative_b_errors.value == 2
    if placard_accuracy != (9 if a_noisy else 10):
        reasons.append("placard_accuracy_differs_from_schedule")
    if (placard_accuracy == 10) is a_noisy:
        reasons.append("placard_live_membership_differs_from_schedule")
    if literal_accuracy != (8 if b_noisy else 10):
        reasons.append("designated_literal_accuracy_differs_from_schedule")
    if b_noisy:
        criterion = "every other one-literal piece rule has accuracy strictly below Q"
        if maximum_other_accuracy >= literal_accuracy:
            reasons.append("noisy_q_not_uniquely_salient")
    else:
        criterion = "Q is the only live one-literal piece rule"
        if live_one_literal_ids != (literal_rule_id,):
            reasons.append("perfect_q_not_unique_live_one_literal")
    return CensusSalienceEvidenceV1(
        status="eligible" if not reasons else "rejected",
        placard_rule_accuracy=placard_accuracy,
        placard_rule_live=placard_accuracy == 10,
        literal_rule_accuracy=literal_accuracy,
        literal_rule_live=literal_accuracy == 10,
        live_one_literal_rule_ids=live_one_literal_ids,
        maximum_other_one_literal_accuracy=maximum_other_accuracy,
        criterion=criterion,
        reason_codes=tuple(reasons),
    )


def _first_query_split_multiset(
    space: VersionSpace,
    exclusions: tuple[int, ...],
) -> tuple[FirstQuerySplitFrequencyV1, ...]:
    excluded = set(exclusions)
    frequencies: Counter[tuple[int, int]] = Counter()
    for scene_index in range(SCENE_COUNT):
        if scene_index in excluded:
            continue
        rejected, accepted = space.label_counts(scene_index)
        if rejected == 0 or accepted == 0:
            continue
        branch_sizes = cast(tuple[int, int], tuple(sorted((rejected, accepted))))
        frequencies[branch_sizes] += 1
    return tuple(
        FirstQuerySplitFrequencyV1(smaller, larger, count)
        for (smaller, larger), count in sorted(frequencies.items())
    )


def _difficulty_evidence(
    space: VersionSpace,
    composed: CatalogEntry,
    exclusions: tuple[int, ...],
) -> CensusDifficultyEvidenceV1:
    if not MINIMUM_SUPPORTED_VERSION_SPACE_SIZE <= len(space) <= MAXIMUM_SUPPORTED_VERSION_SPACE_SIZE:
        return CensusDifficultyEvidenceV1(
            "outside_registered_m_range",
            None,
            None,
            None,
            None,
            (),
            None,
            None,
            (),
            None,
        )
    report = build_query_policy_ceiling_report_v2(
        space,
        composed,
        excluded_scene_indices=exclusions,
    )
    splits = _first_query_split_multiset(space, exclusions)
    if sum(row.legal_scene_count for row in splits) != report.root_informative_scene_count:
        raise EvaluationCensusV2Error("legal first-query split multiset disagrees with query-policy evidence")
    split_obj = [row.as_obj() for row in splits]
    split_digest = _digest(split_obj, domain=_FIRST_SPLIT_DOMAIN)
    greedy = report.budgets[-1].greedy
    first = greedy.first_query
    if first is None:
        raise EvaluationCensusV2Error("multi-rule evaluation opening has no greedy first query")
    first_sizes = tuple(sorted((len(first.rejected_rule_ids), len(first.accepted_rule_ids))))
    signature = tuple(
        (
            step.query.scene_index,
            len(step.before_rule_ids),
            len(step.query.rejected_rule_ids),
            len(step.query.accepted_rule_ids),
            len(step.after_rule_ids),
        )
        for step in greedy.official_path
    )
    return CensusDifficultyEvidenceV1(
        "registered_difficulty_rederived",
        report.excluded_exact_minimax_depth,
        report.greedy_official_recovery_query_count,
        first.scene_index,
        cast(tuple[int, int], first_sizes),
        splits,
        report.root_informative_scene_count,
        split_digest,
        signature,
        report.digest,
    )


def _assess_completed_opening(
    placard: CatalogEntry,
    literal: CatalogEntry,
    composed: CatalogEntry,
    schedule_variant: ScheduleVariant,
    construction_scene_indices: tuple[int, ...],
) -> AssessedEvaluationOpeningV1:
    if build_rule_identity_partitions_v2().for_entry(composed) != "evaluation":
        raise EvaluationCensusV2Error(
            "evaluation-opening assessment requires an evaluation-partition C identity"
        )
    geometry, side = _variant_parts(schedule_variant)
    schedule = build_evidence_schedule_v2(geometry, a_error_target_side=side)
    if len(construction_scene_indices) != 10 or len(set(construction_scene_indices)) != 10:
        raise EvaluationCensusV2Error("completed candidate must contain ten distinct scenes")
    if any(
        isinstance(scene_index, bool)
        or not isinstance(scene_index, int)
        or not 0 <= scene_index < SCENE_COUNT
        for scene_index in construction_scene_indices
    ):
        raise EvaluationCensusV2Error("construction scene lies outside the finite universe")
    cells = tuple(
        _joint_cell_index(placard, literal, composed, scene_index)
        for scene_index in construction_scene_indices
    )
    observed_counts = tuple(cells.count(cell) for cell in range(8))
    if observed_counts != schedule.joint_truth_cell_counts:
        raise EvaluationCensusV2Error("completed candidate does not rederive its exact schedule")
    labels = tuple(composed.truth[scene_index] for scene_index in construction_scene_indices)
    ranking = _ranking_table(composed, construction_scene_indices)
    contract = build_supported_catalog_contract_v2()
    v0_indices = tuple(
        index
        for index, correct in zip(
            contract.supported_indices,
            ranking.correct_counts_by_supported_index,
            strict=True,
        )
        if correct == 10
    )
    catalog = build_rule_catalog()
    space = VersionSpace(catalog, v0_indices)
    if composed.index not in space.indices:
        raise EvaluationCensusV2Error("the Official C identity is absent from its own opening")
    salience = _salience_evidence(
        ranking,
        placard_rule_id=placard.rule_id,
        literal_rule_id=literal.rule_id,
        schedule_variant=schedule_variant,
    )
    exclusions = tuple(sorted(construction_scene_indices))
    difficulty = _difficulty_evidence(space, composed, exclusions)
    reasons: list[str] = list(salience.reason_codes)
    if not MINIMUM_SUPPORTED_VERSION_SPACE_SIZE <= len(space) <= MAXIMUM_SUPPORTED_VERSION_SPACE_SIZE:
        reasons.append("observed_m_outside_8_16")
    q = difficulty.greedy_official_recovery_query_count
    if q is None or not 1 <= q <= MAXIMUM_GREEDY_RECOVERY_QUERIES:
        reasons.append("observed_q_outside_1_4")
    depth = difficulty.minimax_depth
    if depth is None or depth > 4:
        reasons.append("minimax_depth_above_4_or_unavailable")
    disposition: OpeningDisposition = "cell_witness" if not reasons else "constructed_rejection"
    record = CensusConstructedOpeningV1(
        record_kind="constructed_ten_scene_candidate",
        schedule_variant=schedule_variant,
        schedule=schedule.as_obj(),
        construction_scene_indices=construction_scene_indices,
        construction_joint_cells=tuple(f"{cell:03b}" for cell in cells),
        official_labels=labels,
        version_space_rule_ids=tuple(catalog[index].rule_id for index in v0_indices),
        version_space_truth_digests=tuple(catalog[index].truth_digest for index in v0_indices),
        observed_m=len(space),
        observed_q=q,
        ranking_table_digest=ranking.digest,
        salience=salience,
        difficulty=difficulty,
        disposition=disposition,
        reason_codes=tuple(sorted(set(reasons))),
    )
    return AssessedEvaluationOpeningV1(record, ranking)


def assess_evaluation_opening_v1(
    placard_rule_id: str,
    literal_rule_id: str,
    composed_rule_id: str,
    schedule_variant: ScheduleVariant,
    construction_scene_indices: Iterable[int],
) -> AssessedEvaluationOpeningV1:
    """Rederive schedule, all 6,970 ranks, salience, and exact difficulty."""

    placard = _resolve_entry(placard_rule_id, family="placard_literal")
    literal = _resolve_entry(literal_rule_id, family="one_literal_piece")
    composed = _resolve_entry(composed_rule_id, family="composed_two_literal_piece")
    return _assess_completed_opening(
        placard,
        literal,
        composed,
        schedule_variant,
        tuple(construction_scene_indices),
    )


@lru_cache(maxsize=512)
def _joint_cell_pools(
    placard_rule_id: str,
    literal_rule_id: str,
    composed_rule_id: str,
) -> tuple[tuple[int, ...], ...]:
    placard = _resolve_entry(placard_rule_id, family="placard_literal")
    literal = _resolve_entry(literal_rule_id, family="one_literal_piece")
    composed = _resolve_entry(composed_rule_id, family="composed_two_literal_piece")
    pools: list[list[int]] = [[] for _ in range(8)]
    for scene_index in range(SCENE_COUNT):
        pools[_joint_cell_index(placard, literal, composed, scene_index)].append(scene_index)
    return tuple(tuple(pool) for pool in pools)


def _construct_one(
    plan: EvaluationCensusPlanV1,
    attempt: CensusAttemptPlanV1,
    member: CensusMemberPlanV1,
    schedule_variant: ScheduleVariant,
) -> tuple[CensusOpeningRecordV1, CensusRankingTableV1 | None]:
    placard = _resolve_entry(attempt.placard_rule_id, family="placard_literal")
    literal = _resolve_entry(attempt.literal_rule_id, family="one_literal_piece")
    composed = _resolve_entry(member.composed_rule_id, family="composed_two_literal_piece")
    geometry, side = _variant_parts(schedule_variant)
    expected_side = OfficialTargetSideV2(member.noisy_p_target_side)
    if side is not None and side is not expected_side:
        raise EvaluationCensusV2Error("member schedule uses the wrong mirror side")
    schedule = build_evidence_schedule_v2(geometry, a_error_target_side=side)
    pools = _joint_cell_pools(placard.rule_id, literal.rule_id, composed.rule_id)
    for cell, count in enumerate(schedule.joint_truth_cell_counts):
        if len(pools[cell]) < count:
            return (
                CensusPreOpeningFailureV1(
                    "preopening_failure",
                    schedule_variant,
                    "joint_cell_supply",
                    "registered_schedule_cell_has_insufficient_scenes",
                    0,
                ),
                None,
            )
    occurrences = [
        (cell, occurrence)
        for cell, count in enumerate(schedule.joint_truth_cell_counts)
        for occurrence in range(count)
    ]
    occurrences.sort(
        key=lambda item: (
            _hash_parts(
                _OCCURRENCE_ORDER_DOMAIN,
                plan.generator_seed,
                attempt.attempt_digest,
                member.member_position,
                schedule_variant,
                item[0],
                item[1],
            ),
            item,
        )
    )
    ranked_pools = {
        cell: tuple(
            heapq.nsmallest(
                plan.candidate_pool_size + 10,
                pools[cell],
                key=lambda scene_index: (
                    _hash_parts(
                        _SCENE_ORDER_DOMAIN,
                        plan.generator_seed,
                        attempt.attempt_digest,
                        member.member_position,
                        schedule_variant,
                        cell,
                        scene_index,
                    ),
                    scene_index,
                ),
            )
        )
        for cell in {cell for cell, _occurrence in occurrences}
    }
    contract = build_supported_catalog_contract_v2()
    catalog = build_rule_catalog()
    survivors = contract.supported_indices
    selected: list[int] = []
    for cell, _occurrence in occurrences:
        candidates = tuple(scene_index for scene_index in ranked_pools[cell] if scene_index not in selected)[
            : plan.candidate_pool_size
        ]
        target_label = composed.truth[candidates[0]] if candidates else bool(cell & 0b100)
        chosen_scene: int | None = None
        chosen_survivors: tuple[int, ...] | None = None
        for scene_index in candidates:
            after = tuple(
                rule_index
                for rule_index in survivors
                if catalog[rule_index].truth[scene_index] is target_label
            )
            if len(after) >= MINIMUM_SUPPORTED_VERSION_SPACE_SIZE:
                chosen_scene = scene_index
                chosen_survivors = after
                break
        if chosen_scene is None or chosen_survivors is None:
            return (
                CensusPreOpeningFailureV1(
                    "preopening_failure",
                    schedule_variant,
                    "minimum_space_guard",
                    "bounded_candidates_cannot_preserve_eight_supported_rules",
                    len(selected),
                ),
                None,
            )
        survivors = chosen_survivors
        selected.append(chosen_scene)
    assessed = _assess_completed_opening(
        placard,
        literal,
        composed,
        schedule_variant,
        tuple(selected),
    )
    return assessed.record, assessed.ranking_table


def _member_observation(
    member_plan: CensusMemberPlanV1,
    openings: tuple[CensusOpeningRecordV1, ...],
) -> CensusMemberObservationV1:
    reasons: list[str] = []
    if len(openings) != 4:
        raise EvaluationCensusV2Error("a census member must preserve four geometry records")
    if any(type(item) is CensusPreOpeningFailureV1 for item in openings):
        reasons.append("one_or_more_preopening_failures")
    completed = tuple(item for item in openings if type(item) is CensusConstructedOpeningV1)
    if any(item.disposition != "cell_witness" for item in completed):
        reasons.append("one_or_more_constructed_rejections")
    fields: tuple[tuple[int | None, int | None, int | None, tuple[int, int] | None], ...] = tuple(
        (
            item.observed_m,
            item.observed_q,
            item.difficulty.minimax_depth,
            item.difficulty.best_first_query_branch_sizes,
        )
        for item in completed
    )
    if len(completed) != 4:
        reasons.append("four_completed_geometries_absent")
    elif len(set(fields)) != 1:
        reasons.append("cross_geometry_exact_difficulty_mismatch")
    disposition: GroupDisposition = "candidate_group" if not reasons else "preserved_rejection"
    shared = fields[0] if disposition == "candidate_group" else (None, None, None, None)
    return CensusMemberObservationV1(
        member_plan,
        openings,
        disposition,
        shared[0],
        shared[1],
        shared[2],
        shared[3],
        tuple(sorted(set(reasons))),
    )


def _attempt_observation(
    plan: EvaluationCensusPlanV1,
    attempt: CensusAttemptPlanV1,
    ranking_tables: dict[str, CensusRankingTableV1],
) -> CensusAttemptObservationV1:
    members: list[CensusMemberObservationV1] = []
    for member in attempt.members:
        opening_records: list[CensusOpeningRecordV1] = []
        for variant in _member_schedule_variants(member.noisy_p_target_side):
            record, table = _construct_one(plan, attempt, member, variant)
            opening_records.append(record)
            if table is not None:
                existing = ranking_tables.setdefault(table.digest, table)
                if existing != table:
                    raise EvaluationCensusV2Error("ranking digest collision")
        members.append(_member_observation(member, tuple(opening_records)))
    typed_members = cast(tuple[CensusMemberObservationV1, CensusMemberObservationV1], tuple(members))
    reasons = tuple(sorted({reason for member in typed_members for reason in member.reason_codes}))
    disposition: AttemptDisposition = (
        "complete_census_pair"
        if all(member.disposition == "candidate_group" for member in typed_members)
        else "preserved_rejection"
    )
    return CensusAttemptObservationV1(attempt, typed_members, disposition, reasons)


def _cell_summaries(
    attempts: tuple[CensusAttemptObservationV1, ...],
) -> tuple[CensusCellSummaryV1, ...]:
    completed = tuple(
        opening
        for attempt in attempts
        for member in attempt.members
        for opening in member.openings
        if type(opening) is CensusConstructedOpeningV1
    )
    rows: list[CensusCellSummaryV1] = []
    for m in range(8, 17):
        for q in range(1, 5):
            in_cell = tuple(
                opening for opening in completed if opening.observed_m == m and opening.observed_q == q
            )
            groups = tuple(
                member
                for attempt in attempts
                for member in attempt.members
                if member.disposition == "candidate_group"
                and member.observed_m == m
                and member.observed_q == q
            )
            pairs = tuple(
                attempt
                for attempt in attempts
                if attempt.disposition == "complete_census_pair"
                and all(member.observed_m == m and member.observed_q == q for member in attempt.members)
            )
            rows.append(
                CensusCellSummaryV1(
                    m,
                    q,
                    len(in_cell),
                    sum(opening.salience.status == "eligible" for opening in in_cell),
                    sum(opening.disposition == "cell_witness" for opening in in_cell),
                    len(groups),
                    len(pairs),
                )
            )
    return tuple(rows)


@lru_cache(maxsize=8)
def _execute_census_cached(
    plan_text: str,
    expected_plan_digest: str,
) -> EvaluationCensusReportV1:
    plan = parse_evaluation_census_plan_v1(
        plan_text,
        expected_digest=expected_plan_digest,
    )
    ranking_tables: dict[str, CensusRankingTableV1] = {}
    attempts = tuple(_attempt_observation(plan, attempt, ranking_tables) for attempt in plan.attempts)
    constructed_count = sum(
        type(opening) is CensusConstructedOpeningV1
        for attempt in attempts
        for member in attempt.members
        for opening in member.openings
    )
    failure_count = sum(
        type(opening) is CensusPreOpeningFailureV1
        for attempt in attempts
        for member in attempt.members
        for opening in member.openings
    )
    if constructed_count + failure_count != len(plan.attempts) * 8:
        raise EvaluationCensusV2Error("observed ledger does not preserve all planned opening draws")
    tables = tuple(sorted(ranking_tables.values(), key=lambda table: table.digest))
    return EvaluationCensusReportV1(
        plan.source_binding,
        plan.catalog_binding,
        plan.generator_binding,
        plan.digest,
        hashlib.sha256(plan_text.encode("ascii")).hexdigest(),
        len(plan_text.encode("ascii")),
        plan.candidate_pool_size,
        attempts,
        tables,
        _cell_summaries(attempts),
        constructed_count,
        failure_count,
        sum(member.disposition == "candidate_group" for attempt in attempts for member in attempt.members),
        sum(attempt.disposition == "complete_census_pair" for attempt in attempts),
        plan.uses_production_attempt_budget,
    )


def build_evaluation_census_report_v1(
    plan_text: str,
    *,
    expected_plan_digest: str,
) -> EvaluationCensusReportV1:
    """Execute every exact planned attempt and preserve the full observed ledger."""

    return _execute_census_cached(plan_text, expected_plan_digest)


def verify_evaluation_census_report_v1(
    report: EvaluationCensusReportV1,
    plan_text: str,
    *,
    expected_plan_digest: str,
    expected_report_digest: str,
) -> EvaluationCensusReportV1:
    if type(report) is not EvaluationCensusReportV1:
        raise TypeError("report must be an EvaluationCensusReportV1")
    _require_sha256(expected_report_digest, name="expected report digest")
    expected = _execute_census_cached(plan_text, expected_plan_digest)
    if expected.digest != expected_report_digest:
        raise EvaluationCensusV2Error("rederived report differs from externally expected digest")
    if report != expected:
        raise EvaluationCensusV2Error("report differs from exact plan-bound re-execution")
    return expected


def serialize_evaluation_census_report_v1(report: EvaluationCensusReportV1) -> str:
    if type(report) is not EvaluationCensusReportV1:
        raise TypeError("report must be an EvaluationCensusReportV1")
    return _dump_json(report.as_obj()) + "\n"


def evaluation_census_report_v1_from_obj(
    value: object,
    *,
    plan_text: str,
    expected_plan_digest: str,
    expected_report_digest: str,
) -> EvaluationCensusReportV1:
    _require_sha256(expected_report_digest, name="expected report digest")
    obj = _require_mapping(
        value,
        (
            "schema_version",
            "report_kind",
            "status",
            "authorization",
            "source_binding",
            "catalog_binding",
            "generator_binding",
            "prospective_plan_binding",
            "attempt_accounting",
            "attempts",
            "ranking_tables",
            "observed_m_q_summary",
            "selection_boundary",
            "claim_boundary",
            "observed_report_digest",
        ),
        name="observed census report",
    )
    if obj["schema_version"] != EVALUATION_CENSUS_SCHEMA_VERSION or obj["report_kind"] != _REPORT_KIND:
        raise EvaluationCensusV2Error("unknown observed census schema or kind")
    expected = _execute_census_cached(plan_text, expected_plan_digest)
    if obj["status"] != expected.status:
        raise EvaluationCensusV2Error("observed census status changed")
    if obj["authorization"] != _AUTHORIZATION:
        raise EvaluationCensusV2Error("observed census authorization must remain false")
    if expected.digest != expected_report_digest:
        raise EvaluationCensusV2Error("rederived census report differs from expected digest")
    if obj["observed_report_digest"] != expected.digest:
        raise EvaluationCensusV2Error("stored observed-report digest differs")
    if _dump_json(value) != _dump_json(expected.as_obj()):
        raise EvaluationCensusV2Error(
            "observed census report differs from schedule, ranking, salience, or difficulty rederivation"
        )
    return expected


def parse_evaluation_census_report_v1(
    text: str,
    *,
    plan_text: str,
    expected_plan_digest: str,
    expected_report_digest: str,
    require_canonical: bool = True,
) -> EvaluationCensusReportV1:
    report = evaluation_census_report_v1_from_obj(
        _load_json(text),
        plan_text=plan_text,
        expected_plan_digest=expected_plan_digest,
        expected_report_digest=expected_report_digest,
    )
    if require_canonical and serialize_evaluation_census_report_v1(report) != text:
        raise EvaluationCensusV2Error("observed census report is not canonical newline-terminated JSON")
    return report
