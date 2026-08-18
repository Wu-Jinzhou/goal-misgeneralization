"""Additive prospective v4 repair after the preserved v3 construction failure.

V4 reuses the already-frozen v3 requests, folds, ordinal schedules, and semantic
gate.  Its sole construction change is a generic, prospectively audited pair
predicate: every requested Y/P/Q cell must contain enough distinct scenes for
both phases plus one unused reserve scene.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from typing import Any

from ._json import json_digest
from .capability_v3 import (
    EpisodeBankV3,
    EpisodeRequestV3,
    GenerationRecordV3,
    HiddenEpisodeV3,
    ProspectiveDesignV3,
    SemanticGateDerivationV3,
    StageSpecV3,
    _PairCandidateV3,
    _phase_from_schedule,
    _request_pairs_v3,
    build_prospective_design_v3,
    expected_cell_counts_v3,
)
from .catalog import CatalogEntry, build_rule_catalog
from .schema import NONEMPTY_ARRANGEMENT_COUNT, SCENE_COUNT
from .stage_partitions_v2 import (
    build_eligible_target_shadow_table_v2,
    build_rule_identity_partitions_v2,
)

CAPABILITY_SCHEMA_VERSION_V4 = 4
PAIR_SCENE_RESERVE_V4 = 1
V3_FAILED_DESIGN_DIGEST = (
    "51beb1a5cdfb21947a3ce09ae4c251c91034279c77ba425060112c9ba9dc1e39"
)
V3_FROZEN_SOURCE_SHA256 = (
    "09efbd744e3b81aace7677a3bf9079b4a52204525337008ac5fc80900bc08d42"
)


class CapabilityV4Error(ValueError):
    """Raised when a prospective v4 invariant is violated."""


def _ypq_scene_counts(target: CatalogEntry, shadow: CatalogEntry) -> tuple[int, ...]:
    universe = (1 << SCENE_COUNT) - 1
    placard = (1 << NONEMPTY_ARRANGEMENT_COUNT) - 1
    target_bits = target.truth.bits
    shadow_bits = shadow.truth.bits
    counts: list[int] = []
    for cell in range(8):
        y = bool(cell & 0b100)
        p = bool(cell & 0b010)
        q = bool(cell & 0b001)
        mask = target_bits if y else universe ^ target_bits
        mask &= placard if p else universe ^ placard
        mask &= shadow_bits if q else universe ^ shadow_bits
        counts.append(mask.bit_count())
    return tuple(counts)


def pair_feasible_for_request_v4(
    request: EpisodeRequestV3,
    pair: _PairCandidateV3,
    *,
    reserve_per_requested_cell: int = PAIR_SCENE_RESERVE_V4,
) -> bool:
    """Apply the generic two-phase scene-capacity predicate."""

    if (
        isinstance(reserve_per_requested_cell, bool)
        or not isinstance(reserve_per_requested_cell, int)
        or reserve_per_requested_cell < 0
    ):
        raise CapabilityV4Error("v4 scene reserve must be a non-negative integer")
    catalog = build_rule_catalog()
    available = _ypq_scene_counts(
        catalog[pair.target_index], catalog[pair.shadow_index]
    )
    required_per_phase = expected_cell_counts_v3(request)
    return all(
        count == 0
        or available[cell] >= 2 * count + reserve_per_requested_cell
        for cell, count in enumerate(required_per_phase)
    )


def feasible_request_pairs_v4(
    request: EpisodeRequestV3,
    *,
    reserve_per_requested_cell: int = PAIR_SCENE_RESERVE_V4,
) -> tuple[_PairCandidateV3, ...]:
    return tuple(
        pair
        for pair in _request_pairs_v3(request)
        if pair_feasible_for_request_v4(
            request,
            pair,
            reserve_per_requested_cell=reserve_per_requested_cell,
        )
    )


def _stratum_key(request: EpisodeRequestV3) -> str:
    return ":".join(
        (
            request.partition,
            request.target_family,
            request.target_op or "none",
            request.proxy_orientation,
            request.placard_polarity or "none",
        )
    )


def _hall_group_key(request: EpisodeRequestV3) -> str:
    return ":".join(
        (
            request.partition,
            request.target_family,
            request.target_op or "none",
        )
    )


@dataclass(frozen=True, slots=True)
class PairPopulationRowV4:
    stratum: str
    requested: int
    eligible_target_identities: int
    eligible_pairs: int

    @property
    def marginal_capacity_sufficient(self) -> bool:
        return self.eligible_target_identities >= self.requested

    def as_obj(self) -> dict[str, Any]:
        return {
            "stratum": self.stratum,
            "requested": self.requested,
            "eligible_target_identities": self.eligible_target_identities,
            "eligible_pairs": self.eligible_pairs,
            "marginal_capacity_sufficient": self.marginal_capacity_sufficient,
        }


@dataclass(frozen=True, slots=True)
class HallCapacityRowV4:
    group: str
    stratum_subset: tuple[str, ...]
    requested: int
    union_eligible_target_identities: int

    @property
    def sufficient(self) -> bool:
        return self.union_eligible_target_identities >= self.requested

    def as_obj(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "stratum_subset": list(self.stratum_subset),
            "requested": self.requested,
            "union_eligible_target_identities": self.union_eligible_target_identities,
            "sufficient": self.sufficient,
        }


@dataclass(frozen=True, slots=True)
class PairPopulationAuditV4:
    reserve_per_requested_cell: int
    rows: tuple[PairPopulationRowV4, ...]
    hall_rows: tuple[HallCapacityRowV4, ...]

    @property
    def passed(self) -> bool:
        return all(row.marginal_capacity_sufficient for row in self.rows) and all(
            row.sufficient for row in self.hall_rows
        )

    def as_obj(self) -> dict[str, Any]:
        return {
            "predicate": (
                "for every requested Y/P/Q cell c: exact_available(c) >= "
                "2 * per_phase_required(c) + reserve_per_requested_cell"
            ),
            "reserve_per_requested_cell": self.reserve_per_requested_cell,
            "full_eligible_population_audited": True,
            "rows": [row.as_obj() for row in self.rows],
            "hall_rows": [row.as_obj() for row in self.hall_rows],
            "passed": self.passed,
        }

    @property
    def digest(self) -> str:
        return json_digest(
            self.as_obj(), domain="goalzendo-interactive-pair-population-audit-v4"
        )


@lru_cache(maxsize=1)
def build_pair_population_audit_v4() -> PairPopulationAuditV4:
    """Audit all strata and Hall subsets before any candidate selection."""

    base = build_prospective_design_v3()
    requests = (*base.warm.requests, *base.capability.requests)
    representatives: dict[str, EpisodeRequestV3] = {}
    requested = Counter[str]()
    for request in requests:
        key = _stratum_key(request)
        requested[key] += 1
        representatives.setdefault(key, request)
    identity_sets: dict[str, frozenset[int]] = {}
    rows: list[PairPopulationRowV4] = []
    for key in sorted(representatives):
        pairs = feasible_request_pairs_v4(representatives[key])
        identities = frozenset(pair.target_index for pair in pairs)
        identity_sets[key] = identities
        rows.append(PairPopulationRowV4(key, requested[key], len(identities), len(pairs)))
    group_strata: dict[str, list[str]] = defaultdict(list)
    for key, request in representatives.items():
        group_strata[_hall_group_key(request)].append(key)
    hall_rows: list[HallCapacityRowV4] = []
    for group in sorted(group_strata):
        strata = tuple(sorted(group_strata[group]))
        for size in range(1, len(strata) + 1):
            for subset in combinations(strata, size):
                union = frozenset().union(*(identity_sets[key] for key in subset))
                hall_rows.append(
                    HallCapacityRowV4(
                        group,
                        subset,
                        sum(requested[key] for key in subset),
                        len(union),
                    )
                )
    result = PairPopulationAuditV4(
        PAIR_SCENE_RESERVE_V4,
        tuple(rows),
        tuple(hall_rows),
    )
    if not result.passed:
        raise CapabilityV4Error("prospective v4 pair-population capacity fails")
    return result


@dataclass(frozen=True, slots=True)
class StageSpecV4:
    base: StageSpecV3
    pair_population_audit_digest: str
    reserve_per_requested_cell: int = PAIR_SCENE_RESERVE_V4

    @property
    def bank_id(self) -> str:
        return self.base.bank_id.replace("v3", "v4", 1)

    @property
    def requests(self) -> tuple[EpisodeRequestV3, ...]:
        return self.base.requests

    @property
    def cell_schedule(self) -> tuple[tuple[int, ...], ...]:
        return self.base.cell_schedule

    @property
    def max_pair_attempts(self) -> int:
        return self.base.max_pair_attempts

    def as_obj(self) -> dict[str, Any]:
        return {
            "schema_version": CAPABILITY_SCHEMA_VERSION_V4,
            "bank_id": self.bank_id,
            "frozen_v3_stage_spec_digest": self.base.digest,
            "requests": [request.as_obj() for request in self.requests],
            "cell_schedule": [list(row) for row in self.cell_schedule],
            "pair_population_audit_digest": self.pair_population_audit_digest,
            "reserve_per_requested_cell": self.reserve_per_requested_cell,
            "max_pair_attempts": self.max_pair_attempts,
        }

    @property
    def digest(self) -> str:
        return json_digest(self.as_obj(), domain="goalzendo-interactive-stage-spec-v4")


@dataclass(frozen=True, slots=True)
class ProspectiveDesignV4:
    failed_v3: ProspectiveDesignV3
    warm: StageSpecV4
    capability: StageSpecV4
    semantic_gate: SemanticGateDerivationV3
    pair_population_audit: PairPopulationAuditV4

    def __post_init__(self) -> None:
        if self.failed_v3.digest != V3_FAILED_DESIGN_DIGEST:
            raise CapabilityV4Error("preserved v3 freeze digest changed")
        if self.semantic_gate.digest != self.failed_v3.semantic_gate.digest:
            raise CapabilityV4Error("v4 must retain the exact prospective semantic gate")
        if not self.pair_population_audit.passed:
            raise CapabilityV4Error("v4 pair population audit must pass before freezing")

    def as_obj(self) -> dict[str, Any]:
        return {
            "schema_version": CAPABILITY_SCHEMA_VERSION_V4,
            "design_id": "g03-v4-prospective-pair-feasible-counterbalanced-powered",
            "status": "frozen_before_materialization_non_authorizing",
            "preserved_failed_v3": {
                "design_digest": V3_FAILED_DESIGN_DIGEST,
                "source_sha256": V3_FROZEN_SOURCE_SHA256,
                "failure": "selected_pair_missing_required_ypq_cells",
            },
            "unchanged_from_v3": {
                "requests": True,
                "registered_folds": True,
                "renderer_assignment": True,
                "ordinal_cell_schedules": True,
                "semantic_metric_order_thresholds_and_acceptance": True,
                "salts": True,
            },
            "pair_feasibility_change": self.pair_population_audit.as_obj(),
            "warm": self.warm.as_obj(),
            "capability": self.capability.as_obj(),
            "semantic_gate": self.semantic_gate.as_obj(),
            "catalog_digest": build_rule_catalog().digest,
            "partitions_digest": build_rule_identity_partitions_v2().digest,
            "eligible_pairs_v2_digest": build_eligible_target_shadow_table_v2().digest,
            "production_bank_generation_authorized": False,
            "weight_updates_authorized": False,
        }

    @property
    def digest(self) -> str:
        return json_digest(
            self.as_obj(), domain="goalzendo-interactive-prospective-design-v4"
        )


@lru_cache(maxsize=1)
def build_prospective_design_v4() -> ProspectiveDesignV4:
    failed_v3 = build_prospective_design_v3()
    population = build_pair_population_audit_v4()
    warm = StageSpecV4(failed_v3.warm, population.digest)
    capability = StageSpecV4(failed_v3.capability, population.digest)
    return ProspectiveDesignV4(
        failed_v3,
        warm,
        capability,
        failed_v3.semantic_gate,
        population,
    )


@dataclass(frozen=True, slots=True)
class EpisodeBankV4:
    spec: StageSpecV4
    episodes: tuple[HiddenEpisodeV3, ...]
    generation_records: tuple[GenerationRecordV3, ...]

    def __post_init__(self) -> None:
        episodes = tuple(self.episodes)
        records = tuple(self.generation_records)
        object.__setattr__(self, "episodes", episodes)
        object.__setattr__(self, "generation_records", records)
        if len(episodes) != len(self.spec.requests) or len(records) != len(episodes):
            raise CapabilityV4Error("v4 bank rows do not align")
        targets = [episode.target.truth_digest for episode in episodes]
        if len(set(targets)) != len(targets):
            raise CapabilityV4Error("v4 Official-Law identities must be unique")
        for index, (request, episode, record) in enumerate(
            zip(self.spec.requests, episodes, records, strict=True)
        ):
            if episode.request != request or record.request_id != request.request_id:
                raise CapabilityV4Error("v4 request binding mismatch")
            schedule = self.spec.cell_schedule[index]
            for phase in (episode.opening, episode.terminal):
                observed: list[int] = []
                for item in phase:
                    y = int(episode.target.truth[item.scene_index])
                    p = int(item.scene_index < NONEMPTY_ARRANGEMENT_COUNT)
                    q = int(episode.shadow.truth[item.scene_index])
                    observed.append(y * 4 + p * 2 + q)
                if tuple(observed) != schedule:
                    raise CapabilityV4Error("v4 changed the frozen ordinal schedule")

    def as_obj(self) -> dict[str, Any]:
        return {
            "schema_version": CAPABILITY_SCHEMA_VERSION_V4,
            "spec": self.spec.as_obj(),
            "episodes": [episode.as_obj() for episode in self.episodes],
            "generation_records": [record.as_obj() for record in self.generation_records],
            "production_bank_generation_authorized": False,
            "weight_updates_authorized": False,
        }

    @property
    def digest(self) -> str:
        return json_digest(self.as_obj(), domain="goalzendo-interactive-episode-bank-v4")

    def v3_view(self) -> EpisodeBankV3:
        """Return a validation-compatible view for unchanged v3 QA primitives."""

        return EpisodeBankV3(self.spec.base, self.episodes, self.generation_records)


def _select_feasible_pair_v4(
    request: EpisodeRequestV3,
    *,
    reserved_targets: set[str],
    spec: StageSpecV4,
) -> tuple[int, _PairCandidateV3]:
    catalog = build_rule_catalog()
    pairs = feasible_request_pairs_v4(
        request,
        reserve_per_requested_cell=spec.reserve_per_requested_cell,
    )
    for rank, pair in enumerate(pairs[: spec.max_pair_attempts]):
        if catalog[pair.target_index].truth_digest not in reserved_targets:
            return rank, pair
    raise CapabilityV4Error(
        f"bounded v4 feasible-pair selection failed for {request.request_id!r}"
    )


@lru_cache(maxsize=4)
def generate_episode_bank_v4(spec: StageSpecV4) -> EpisodeBankV4:
    if type(spec) is not StageSpecV4:
        raise TypeError("generate_episode_bank_v4 requires StageSpecV4")
    catalog = build_rule_catalog()
    reserved: set[str] = set()
    episodes: list[HiddenEpisodeV3] = []
    records: list[GenerationRecordV3] = []
    for index, request in enumerate(spec.requests):
        pair_rank, pair = _select_feasible_pair_v4(
            request,
            reserved_targets=reserved,
            spec=spec,
        )
        target = catalog[pair.target_index]
        shadow = catalog[pair.shadow_index]
        excluded: set[int] = set()
        schedule = spec.cell_schedule[index]
        opening = _phase_from_schedule(
            request, pair, target, shadow, schedule, "opening", excluded
        )
        terminal = _phase_from_schedule(
            request, pair, target, shadow, schedule, "terminal", excluded
        )
        episode = HiddenEpisodeV3(
            (
                f"{spec.bank_id}-{request.request_id}-"
                f"{target.truth_digest[:10]}-{shadow.truth_digest[:10]}"
            ),
            request,
            target,
            shadow,
            opening,
            terminal,
        )
        episodes.append(episode)
        records.append(
            GenerationRecordV3(
                request.request_id,
                pair_rank,
                pair.order_digest,
                json_digest(
                    [item.scene_index for item in opening],
                    domain="goalzendo-interactive-opening-scenes-v4",
                ),
                json_digest(
                    [item.scene_index for item in terminal],
                    domain="goalzendo-interactive-terminal-scenes-v4",
                ),
            )
        )
        reserved.add(target.truth_digest)
    return EpisodeBankV4(spec, tuple(episodes), tuple(records))


@dataclass(frozen=True, slots=True)
class MaterializedCandidateV4:
    design: ProspectiveDesignV4
    warm: EpisodeBankV4
    capability: EpisodeBankV4

    def __post_init__(self) -> None:
        warm_targets = {episode.target.truth_digest for episode in self.warm.episodes}
        capability_targets = {
            episode.target.truth_digest for episode in self.capability.episodes
        }
        if warm_targets & capability_targets:
            raise CapabilityV4Error("v4 target identities overlap across stages")

    @property
    def production_bank_generation_authorized(self) -> bool:
        return False

    @property
    def weight_updates_authorized(self) -> bool:
        return False


@lru_cache(maxsize=1)
def materialize_prospective_design_v4(
    design: ProspectiveDesignV4,
) -> MaterializedCandidateV4:
    if type(design) is not ProspectiveDesignV4:
        raise TypeError("materialize_prospective_design_v4 requires ProspectiveDesignV4")
    return MaterializedCandidateV4(
        design,
        generate_episode_bank_v4(design.warm),
        generate_episode_bank_v4(design.capability),
    )
