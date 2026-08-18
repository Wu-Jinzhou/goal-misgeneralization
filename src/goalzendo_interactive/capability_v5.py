"""Additive prospective v5 repair with exact request-vector fold modeling.

V5 preserves the v4 quotas, pair predicate, reserve, salts, and semantic gate.
It generalizes the pre-outcome fold model from an assumed balanced episode to
the exact target-independent outcome vector implied by every request profile.
Renderer assignment and ordinal schedules are then solved against those exact
registered folds before any Official-Law or shadow identity is selected.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp  # type: ignore[import-untyped]
from scipy.sparse import lil_matrix  # type: ignore[import-untyped]

from ._json import json_digest
from .capability_v3 import (
    EpisodeRequestV3,
    GenerationRecordV3,
    HiddenEpisodeV3,
    SemanticGateDerivationV3,
    _outcome_bits,
    _PairCandidateV3,
    _phase_from_schedule,
    _registered_fold_assignments,
    expected_cell_counts_v3,
)
from .capability_v4 import (
    PAIR_SCENE_RESERVE_V4,
    PairPopulationAuditV4,
    ProspectiveDesignV4,
    _select_feasible_pair_v4,
    build_prospective_design_v4,
)
from .catalog import build_rule_catalog
from .rendering import EVAL_RENDERERS, TRAIN_RENDERERS, RendererName
from .schema import NONEMPTY_ARRANGEMENT_COUNT
from .stage_partitions_v2 import (
    build_eligible_target_shadow_table_v2,
    build_rule_identity_partitions_v2,
)

CAPABILITY_SCHEMA_VERSION_V5 = 5
V4_FROZEN_DESIGN_DIGEST = (
    "12ed009bdcad9b5b1b4f5e07e8f5fe6e86049e020ae7c5e1d4287310a56a1740"
)
V4_FROZEN_AUDIT_DIGEST = (
    "05924b050646795b00885bb73a5a60d0ffed9768e1416bae6c98e02d57289b20"
)
OBSERVATION_TARGETS_V5 = (
    "target_label",
    "placard_error_status",
    "shadow_error_status",
)


class CapabilityV5Error(ValueError):
    """Raised when a prospective v5 exact-fold invariant is violated."""


def request_outcome_vector_v5(
    request: EpisodeRequestV3,
    target_name: str,
) -> tuple[int, int]:
    """Return exact two-phase false/true counts without selecting a target."""

    try:
        target_index = OBSERVATION_TARGETS_V5.index(target_name)
    except ValueError as exc:
        raise CapabilityV5Error("unknown v5 observation target") from exc
    true_count = 2 * sum(
        count * _outcome_bits(cell)[target_index]
        for cell, count in enumerate(expected_cell_counts_v3(request))
    )
    return 20 - true_count, true_count


def registered_outcome_folds_v5(
    requests: tuple[EpisodeRequestV3, ...],
    target_name: str,
) -> tuple[int, ...]:
    vectors = tuple(request_outcome_vector_v5(request, target_name) for request in requests)
    return _registered_fold_assignments(
        target_name,
        ("false", "true"),
        vectors,
    )


def _solve_renderers_v5(
    requests: tuple[EpisodeRequestV3, ...],
) -> tuple[EpisodeRequestV3, ...]:
    renderers = (
        TRAIN_RENDERERS
        if requests[0].stage == "format_warm_start"
        else EVAL_RENDERERS
    )
    renderer_count = len(renderers)
    variable_count = len(requests) * renderer_count

    def variable(episode: int, renderer: int) -> int:
        return episode * renderer_count + renderer

    rows: list[list[tuple[int, int]]] = []
    lower: list[int] = []
    upper: list[int] = []
    for episode in range(len(requests)):
        rows.append(
            [(variable(episode, renderer), 1) for renderer in range(renderer_count)]
        )
        lower.append(1)
        upper.append(1)
    for fold in range(5):
        for target, values in (
            ("formula", ("all_or_any", "exactly_one")),
            ("proxy", ("proxy_low", "proxy_high")),
        ):
            for value in values:
                episodes = tuple(
                    index
                    for index, request in enumerate(requests)
                    if (
                        request.registered_formula_fold == fold
                        and request.formula_label == value
                        if target == "formula"
                        else request.registered_proxy_fold == fold
                        and request.proxy_orientation == value
                    )
                )
                if len(episodes) % renderer_count:
                    raise CapabilityV5Error(
                        "formula/proxy fold cannot be divided across renderers"
                    )
                for renderer in range(renderer_count):
                    rows.append(
                        [(variable(episode, renderer), 1) for episode in episodes]
                    )
                    lower.append(len(episodes) // renderer_count)
                    upper.append(len(episodes) // renderer_count)
    for target_name in OBSERVATION_TARGETS_V5:
        vectors = tuple(
            request_outcome_vector_v5(request, target_name) for request in requests
        )
        folds = registered_outcome_folds_v5(requests, target_name)
        for fold in range(5):
            episodes = tuple(
                index for index, assigned in enumerate(folds) if assigned == fold
            )
            for renderer in range(renderer_count):
                rows.append(
                    [
                        (
                            variable(episode, renderer),
                            vectors[episode][0] - vectors[episode][1],
                        )
                        for episode in episodes
                        if vectors[episode][0] != vectors[episode][1]
                    ]
                )
                lower.append(0)
                upper.append(0)
    matrix = lil_matrix((len(rows), variable_count), dtype=np.float64)
    for row_index, row in enumerate(rows):
        for column, coefficient in row:
            matrix[row_index, column] = coefficient
    result = milp(
        np.zeros(variable_count, dtype=np.float64),
        integrality=np.ones(variable_count, dtype=np.int8),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(matrix.tocsr(), lower, upper),
        options={"mip_rel_gap": 0.0, "presolve": True},
    )
    if not result.success or result.x is None:
        raise CapabilityV5Error(
            f"exact v5 renderer balance is infeasible: {result.message}"
        )
    assigned: list[EpisodeRequestV3] = []
    for episode, request in enumerate(requests):
        selected = tuple(
            renderer
            for renderer in range(renderer_count)
            if result.x[variable(episode, renderer)] > 0.5
        )
        if len(selected) != 1:
            raise CapabilityV5Error("v5 renderer solver did not choose exactly one renderer")
        assigned.append(replace(request, renderer=renderers[selected[0]]))
    frozen = tuple(assigned)
    _validate_renderer_balance_v5(frozen)
    return frozen


def _validate_renderer_balance_v5(requests: tuple[EpisodeRequestV3, ...]) -> None:
    renderers = (
        TRAIN_RENDERERS
        if requests[0].stage == "format_warm_start"
        else EVAL_RENDERERS
    )
    for fold in range(5):
        for target, values in (
            ("formula", ("all_or_any", "exactly_one")),
            ("proxy", ("proxy_low", "proxy_high")),
        ):
            for value in values:
                counts = Counter(
                    request.renderer
                    for request in requests
                    if (
                        request.registered_formula_fold == fold
                        and request.formula_label == value
                        if target == "formula"
                        else request.registered_proxy_fold == fold
                        and request.proxy_orientation == value
                    )
                )
                if len({counts[renderer] for renderer in renderers}) != 1:
                    raise CapabilityV5Error("v5 formula/proxy renderer balance failed")
    for target_name in OBSERVATION_TARGETS_V5:
        vectors = tuple(
            request_outcome_vector_v5(request, target_name) for request in requests
        )
        folds = registered_outcome_folds_v5(requests, target_name)
        for fold in range(5):
            for renderer in renderers:
                totals = tuple(
                    sum(
                        vectors[index][class_index]
                        for index, request in enumerate(requests)
                        if folds[index] == fold and request.renderer == renderer
                    )
                    for class_index in range(2)
                )
                if totals[0] != totals[1]:
                    raise CapabilityV5Error(
                        "v5 observation outcome is renderer-imbalanced inside a fold"
                    )


def _solve_schedule_v5(
    requests: tuple[EpisodeRequestV3, ...],
) -> tuple[tuple[int, ...], ...]:
    profiles = tuple(expected_cell_counts_v3(request) for request in requests)
    target_folds = tuple(
        registered_outcome_folds_v5(requests, target_name)
        for target_name in OBSERVATION_TARGETS_V5
    )
    variables = tuple(
        (episode, cell, position)
        for episode, profile in enumerate(profiles)
        for cell, count in enumerate(profile)
        if count
        for position in range(10)
    )
    variable_index = {value: index for index, value in enumerate(variables)}
    rows: list[list[tuple[int, int]]] = []
    lower: list[int] = []
    upper: list[int] = []
    for episode, profile in enumerate(profiles):
        for cell, count in enumerate(profile):
            if count:
                rows.append(
                    [
                        (variable_index[(episode, cell, position)], 1)
                        for position in range(10)
                    ]
                )
                lower.append(count)
                upper.append(count)
    for episode, profile in enumerate(profiles):
        for position in range(10):
            rows.append(
                [
                    (variable_index[(episode, cell, position)], 1)
                    for cell, count in enumerate(profile)
                    if count
                ]
            )
            lower.append(1)
            upper.append(1)
    for target_index, folds in enumerate(target_folds):
        for fold in range(5):
            episodes = tuple(
                index for index, assigned in enumerate(folds) if assigned == fold
            )
            for position in range(10):
                rows.append(
                    [
                        (variable_index[(episode, cell, position)], 1)
                        for episode in episodes
                        for cell, count in enumerate(profiles[episode])
                        if count and _outcome_bits(cell)[target_index]
                    ]
                )
                required_true = sum(
                    request_outcome_vector_v5(requests[episode], OBSERVATION_TARGETS_V5[target_index])[1]
                    for episode in episodes
                )
                if required_true % 20:
                    raise CapabilityV5Error("v5 fold cannot balance every ordinal position")
                per_position = required_true // 20
                lower.append(per_position)
                upper.append(per_position)
    matrix = lil_matrix((len(rows), len(variables)), dtype=np.float64)
    for row_index, row in enumerate(rows):
        for column, value in row:
            matrix[row_index, column] = value
    result = milp(
        np.zeros(len(variables), dtype=np.float64),
        integrality=np.ones(len(variables), dtype=np.int8),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(matrix.tocsr(), lower, upper),
        options={"mip_rel_gap": 0.0, "presolve": True},
    )
    if not result.success or result.x is None:
        raise CapabilityV5Error(
            f"exact v5 ordinal balance is infeasible: {result.message}"
        )
    schedule: list[list[int]] = [[-1] * 10 for _ in requests]
    for (episode, cell, position), value in zip(variables, result.x, strict=True):
        if value > 0.5:
            if schedule[episode][position] != -1:
                raise CapabilityV5Error("v5 schedule assigned two cells to one position")
            schedule[episode][position] = cell
    frozen = tuple(tuple(row) for row in schedule)
    _validate_schedule_v5(requests, frozen)
    return frozen


def _validate_schedule_v5(
    requests: tuple[EpisodeRequestV3, ...],
    schedule: tuple[tuple[int, ...], ...],
) -> None:
    if len(schedule) != len(requests) or any(len(row) != 10 for row in schedule):
        raise CapabilityV5Error("v5 schedule shape mismatch")
    for request, row in zip(requests, schedule, strict=True):
        if Counter(row) != Counter(
            {
                cell: count
                for cell, count in enumerate(expected_cell_counts_v3(request))
                if count
            }
        ):
            raise CapabilityV5Error("v5 schedule row changes its proxy profile")
    for target_index, target_name in enumerate(OBSERVATION_TARGETS_V5):
        folds = registered_outcome_folds_v5(requests, target_name)
        for fold in range(5):
            episodes = tuple(
                index for index, assigned in enumerate(folds) if assigned == fold
            )
            totals = tuple(
                sum(
                    request_outcome_vector_v5(requests[episode], target_name)[class_index]
                    for episode in episodes
                )
                for class_index in range(2)
            )
            if totals[0] != totals[1]:
                raise CapabilityV5Error("v5 registered fold is not class balanced")
            for position in range(10):
                true_count = sum(
                    _outcome_bits(schedule[episode][position])[target_index]
                    for episode in episodes
                )
                if true_count * 2 != len(episodes):
                    raise CapabilityV5Error("v5 position is imbalanced inside a fold")


@dataclass(frozen=True, slots=True)
class OutcomeFoldRowV5:
    stage: str
    target_name: str
    fold: int
    group_count: int
    false_sample_count: int
    true_sample_count: int
    renderer_false_true_counts: tuple[tuple[str, int, int], ...]

    @property
    def passed(self) -> bool:
        return self.false_sample_count == self.true_sample_count and all(
            false_count == true_count
            for _, false_count, true_count in self.renderer_false_true_counts
        )

    def as_obj(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "target_name": self.target_name,
            "fold": self.fold,
            "group_count": self.group_count,
            "false_sample_count": self.false_sample_count,
            "true_sample_count": self.true_sample_count,
            "renderer_false_true_counts": {
                renderer: {"false": false_count, "true": true_count}
                for renderer, false_count, true_count in self.renderer_false_true_counts
            },
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class OutcomeFoldPopulationAuditV5:
    rows: tuple[OutcomeFoldRowV5, ...]

    @property
    def passed(self) -> bool:
        return bool(self.rows) and all(row.passed for row in self.rows)

    def as_obj(self) -> dict[str, Any]:
        return {
            "input_model": (
                "exact target-independent two-phase request outcome vectors derived "
                "from frozen Y/P/Q profiles, including placard 20/0 and 0/20"
            ),
            "rows": [row.as_obj() for row in self.rows],
            "passed": self.passed,
        }

    @property
    def digest(self) -> str:
        return json_digest(
            self.as_obj(), domain="goalzendo-interactive-outcome-fold-population-v5"
        )


def _fold_population_audit_v5(
    stages: tuple[tuple[str, tuple[EpisodeRequestV3, ...]], ...],
) -> OutcomeFoldPopulationAuditV5:
    rows: list[OutcomeFoldRowV5] = []
    for stage, requests in stages:
        renderers: tuple[RendererName, ...] = (
            TRAIN_RENDERERS if stage == "format_warm_start" else EVAL_RENDERERS
        )
        for target_name in OBSERVATION_TARGETS_V5:
            vectors = tuple(
                request_outcome_vector_v5(request, target_name) for request in requests
            )
            folds = registered_outcome_folds_v5(requests, target_name)
            for fold in range(5):
                indices = tuple(
                    index for index, assigned in enumerate(folds) if assigned == fold
                )
                false_count = sum(vectors[index][0] for index in indices)
                true_count = sum(vectors[index][1] for index in indices)
                renderer_counts = tuple(
                    (
                        renderer,
                        sum(
                            vectors[index][0]
                            for index in indices
                            if requests[index].renderer == renderer
                        ),
                        sum(
                            vectors[index][1]
                            for index in indices
                            if requests[index].renderer == renderer
                        ),
                    )
                    for renderer in renderers
                )
                rows.append(
                    OutcomeFoldRowV5(
                        stage,
                        target_name,
                        fold,
                        len(indices),
                        false_count,
                        true_count,
                        renderer_counts,
                    )
                )
    result = OutcomeFoldPopulationAuditV5(tuple(rows))
    if not result.passed:
        raise CapabilityV5Error("v5 exact fold-population balance is infeasible")
    return result


@dataclass(frozen=True, slots=True)
class StageSpecV5:
    bank_id: str
    requests: tuple[EpisodeRequestV3, ...]
    cell_schedule: tuple[tuple[int, ...], ...]
    pair_population_audit_digest: str
    outcome_fold_population_digest: str
    reserve_per_requested_cell: int = PAIR_SCENE_RESERVE_V4
    max_pair_attempts: int = 4_096

    def __post_init__(self) -> None:
        requests = tuple(self.requests)
        schedule = tuple(tuple(row) for row in self.cell_schedule)
        object.__setattr__(self, "requests", requests)
        object.__setattr__(self, "cell_schedule", schedule)
        _validate_renderer_balance_v5(requests)
        _validate_schedule_v5(requests, schedule)

    def as_obj(self) -> dict[str, Any]:
        return {
            "schema_version": CAPABILITY_SCHEMA_VERSION_V5,
            "bank_id": self.bank_id,
            "requests": [request.as_obj() for request in self.requests],
            "cell_schedule": [list(row) for row in self.cell_schedule],
            "pair_population_audit_digest": self.pair_population_audit_digest,
            "outcome_fold_population_digest": self.outcome_fold_population_digest,
            "reserve_per_requested_cell": self.reserve_per_requested_cell,
            "max_pair_attempts": self.max_pair_attempts,
        }

    @property
    def digest(self) -> str:
        return json_digest(self.as_obj(), domain="goalzendo-interactive-stage-spec-v5")


@dataclass(frozen=True, slots=True)
class ProspectiveDesignV5:
    failed_v4: ProspectiveDesignV4
    warm: StageSpecV5
    capability: StageSpecV5
    semantic_gate: SemanticGateDerivationV3
    pair_population_audit: PairPopulationAuditV4
    outcome_fold_population: OutcomeFoldPopulationAuditV5

    def __post_init__(self) -> None:
        if self.failed_v4.digest != V4_FROZEN_DESIGN_DIGEST:
            raise CapabilityV5Error("preserved v4 design digest changed")
        if self.semantic_gate.digest != self.failed_v4.semantic_gate.digest:
            raise CapabilityV5Error("v5 changed the frozen semantic gate")
        if not self.pair_population_audit.passed or not self.outcome_fold_population.passed:
            raise CapabilityV5Error("v5 prospective population audit failed")

    def as_obj(self) -> dict[str, Any]:
        return {
            "schema_version": CAPABILITY_SCHEMA_VERSION_V5,
            "design_id": "g03-v5-exact-outcome-fold-counterbalanced-powered",
            "status": "frozen_before_materialization_non_authorizing",
            "preserved_failed_v4": {
                "design_digest": V4_FROZEN_DESIGN_DIGEST,
                "audit_digest": V4_FROZEN_AUDIT_DIGEST,
                "failure": "capability_placard_error_inverse_leakage",
            },
            "unchanged_from_v4": {
                "request_ids_families_operators_orientations_and_folds": True,
                "binary_and_control_quotas": True,
                "pair_feasibility_predicate_and_reserve": True,
                "pair_and_scene_selection_salts": True,
                "semantic_metric_order_thresholds_and_acceptance": True,
            },
            "prospective_change": (
                "derive exact per-request outcome vectors, recompute registered "
                "observation folds, then solve renderer and ordinal balance"
            ),
            "pair_population_audit": self.pair_population_audit.as_obj(),
            "outcome_fold_population": self.outcome_fold_population.as_obj(),
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
            self.as_obj(), domain="goalzendo-interactive-prospective-design-v5"
        )


@lru_cache(maxsize=1)
def build_prospective_design_v5() -> ProspectiveDesignV5:
    failed_v4 = build_prospective_design_v4()
    warm_requests = _solve_renderers_v5(failed_v4.warm.requests)
    capability_requests = _solve_renderers_v5(failed_v4.capability.requests)
    warm_schedule = _solve_schedule_v5(warm_requests)
    capability_schedule = _solve_schedule_v5(capability_requests)
    fold_population = _fold_population_audit_v5(
        (
            ("format_warm_start", warm_requests),
            ("capability", capability_requests),
        )
    )
    warm = StageSpecV5(
        "g03-v5-format-warm-start-400",
        warm_requests,
        warm_schedule,
        failed_v4.pair_population_audit.digest,
        fold_population.digest,
    )
    capability = StageSpecV5(
        "g03-v5-capability-420",
        capability_requests,
        capability_schedule,
        failed_v4.pair_population_audit.digest,
        fold_population.digest,
    )
    return ProspectiveDesignV5(
        failed_v4,
        warm,
        capability,
        failed_v4.semantic_gate,
        failed_v4.pair_population_audit,
        fold_population,
    )


@dataclass(frozen=True, slots=True)
class EpisodeBankV5:
    spec: StageSpecV5
    episodes: tuple[HiddenEpisodeV3, ...]
    generation_records: tuple[GenerationRecordV3, ...]

    def __post_init__(self) -> None:
        episodes = tuple(self.episodes)
        records = tuple(self.generation_records)
        object.__setattr__(self, "episodes", episodes)
        object.__setattr__(self, "generation_records", records)
        if len(episodes) != len(self.spec.requests) or len(records) != len(episodes):
            raise CapabilityV5Error("v5 bank rows do not align")
        targets = [episode.target.truth_digest for episode in episodes]
        if len(set(targets)) != len(targets):
            raise CapabilityV5Error("v5 Official-Law identities must be unique")
        for index, (request, episode, record) in enumerate(
            zip(self.spec.requests, episodes, records, strict=True)
        ):
            if episode.request != request or record.request_id != request.request_id:
                raise CapabilityV5Error("v5 request binding mismatch")
            schedule = self.spec.cell_schedule[index]
            for phase in (episode.opening, episode.terminal):
                observed: list[int] = []
                for item in phase:
                    y = int(episode.target.truth[item.scene_index])
                    p = int(item.scene_index < NONEMPTY_ARRANGEMENT_COUNT)
                    q = int(episode.shadow.truth[item.scene_index])
                    observed.append(y * 4 + p * 2 + q)
                if tuple(observed) != schedule:
                    raise CapabilityV5Error("v5 changed the frozen ordinal schedule")

    def as_obj(self) -> dict[str, Any]:
        return {
            "schema_version": CAPABILITY_SCHEMA_VERSION_V5,
            "spec": self.spec.as_obj(),
            "episodes": [episode.as_obj() for episode in self.episodes],
            "generation_records": [record.as_obj() for record in self.generation_records],
            "production_bank_generation_authorized": False,
            "weight_updates_authorized": False,
        }

    @property
    def digest(self) -> str:
        return json_digest(self.as_obj(), domain="goalzendo-interactive-episode-bank-v5")


def _select_pair_v5(
    request: EpisodeRequestV3,
    *,
    reserved_targets: set[str],
    spec: StageSpecV5,
) -> tuple[int, _PairCandidateV3]:
    return _select_feasible_pair_v4(
        request,
        reserved_targets=reserved_targets,
        spec=spec,  # type: ignore[arg-type]
    )


@lru_cache(maxsize=4)
def generate_episode_bank_v5(spec: StageSpecV5) -> EpisodeBankV5:
    if type(spec) is not StageSpecV5:
        raise TypeError("generate_episode_bank_v5 requires StageSpecV5")
    catalog = build_rule_catalog()
    reserved: set[str] = set()
    episodes: list[HiddenEpisodeV3] = []
    records: list[GenerationRecordV3] = []
    for index, request in enumerate(spec.requests):
        pair_rank, pair = _select_pair_v5(
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
                    domain="goalzendo-interactive-opening-scenes-v5",
                ),
                json_digest(
                    [item.scene_index for item in terminal],
                    domain="goalzendo-interactive-terminal-scenes-v5",
                ),
            )
        )
        reserved.add(target.truth_digest)
    return EpisodeBankV5(spec, tuple(episodes), tuple(records))


@dataclass(frozen=True, slots=True)
class MaterializedCandidateV5:
    design: ProspectiveDesignV5
    warm: EpisodeBankV5
    capability: EpisodeBankV5

    def __post_init__(self) -> None:
        warm_targets = {episode.target.truth_digest for episode in self.warm.episodes}
        capability_targets = {
            episode.target.truth_digest for episode in self.capability.episodes
        }
        if warm_targets & capability_targets:
            raise CapabilityV5Error("v5 target identities overlap across stages")

    @property
    def production_bank_generation_authorized(self) -> bool:
        return False

    @property
    def weight_updates_authorized(self) -> bool:
        return False


@lru_cache(maxsize=1)
def materialize_prospective_design_v5(
    design: ProspectiveDesignV5,
) -> MaterializedCandidateV5:
    if type(design) is not ProspectiveDesignV5:
        raise TypeError("materialize_prospective_design_v5 requires ProspectiveDesignV5")
    return MaterializedCandidateV5(
        design,
        generate_episode_bank_v5(design.warm),
        generate_episode_bank_v5(design.capability),
    )
