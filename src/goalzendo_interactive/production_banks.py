"""Fail-closed production-bank planning for the prospective G03 program.

This module constructs request-only :class:`EpisodeBankSpec` objects.  It does
not generate an episode, write a fixture, or authorize a weight update.  The
current v1 engine cannot express two required stages, so the canonical plan is
deliberately non-authorizing and records the smallest required v2 extensions.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal, cast

from ._json import dump_json, json_digest
from .catalog import build_rule_catalog
from .episodes import OpeningRegime, TerminalKind
from .generation import EpisodeBankSpec, EpisodeRequest
from .leakage import minimum_resolution_groups
from .partitions import (
    RULE_PARTITIONS,
    RulePartition,
    build_eligible_pair_table,
    build_rule_identity_partitions,
)
from .rendering import EVAL_RENDERERS, TRAIN_RENDERERS, RendererName
from .rules import BinaryOp, BinaryRule

PRODUCTION_BANK_PLAN_SCHEMA_VERSION = 1
PRODUCTION_BANK_QA_SCHEMA_VERSION = 1
PRODUCTION_REQUEST_DOMAIN = "g03-production-request-v1"

ProductionStage = Literal[
    "format_warm_start",
    "engine_leakage",
    "capability",
    "pilot",
    "confirmatory_train",
    "validation",
    "evaluation_active",
    "evaluation_oracle_replay",
    "evaluation_no_query",
]

PRODUCTION_STAGES: tuple[ProductionStage, ...] = (
    "format_warm_start",
    "engine_leakage",
    "capability",
    "pilot",
    "confirmatory_train",
    "validation",
    "evaluation_active",
    "evaluation_oracle_replay",
    "evaluation_no_query",
)

ENGINE_LEAKAGE_EPISODES = 768
PLANNING_PILOT_EPISODES = 256
PLANNING_CONFIRMATORY_TRAIN_EPISODES = 768
PLANNING_VALIDATION_EPISODES = 256
EVALUATION_EPISODES_PER_VIEW = 256
PLANNING_FORMAT_WARM_START_EPISODES = 256
PLANNING_CAPABILITY_EPISODES = 256

_OPERATOR_SLOTS: tuple[BinaryOp, ...] = (
    "all",
    "any",
    "exactly_one",
    "exactly_one",
)
_EVIDENCE_SLOTS: tuple[tuple[OpeningRegime, bool | None], ...] = (
    ("perfect_ambiguity", None),
    ("perfect_ambiguity", None),
    ("noisy_shortcuts", False),
    ("noisy_shortcuts", True),
)


@dataclass(frozen=True, slots=True)
class ProductionStageSlice:
    """A contiguous, independently balanced slice of one generation unit."""

    stage: ProductionStage
    unit_id: str
    start: int
    stop: int
    partition: RulePartition
    renderers: tuple[RendererName, ...]
    terminal_kinds: tuple[TerminalKind, ...]
    count_basis: str

    def __post_init__(self) -> None:
        if self.stage not in PRODUCTION_STAGES:
            raise ValueError(f"unknown production stage: {self.stage!r}")
        if type(self.unit_id) is not str or not self.unit_id or not self.unit_id.isascii():
            raise ValueError("production unit id must be nonempty ASCII")
        if (
            isinstance(self.start, bool)
            or isinstance(self.stop, bool)
            or not isinstance(self.start, int)
            or not isinstance(self.stop, int)
            or self.start < 0
            or self.stop <= self.start
        ):
            raise ValueError("production slice bounds must be nonempty non-negative integers")
        if self.partition not in RULE_PARTITIONS:
            raise ValueError(f"unknown rule partition: {self.partition!r}")
        if not self.renderers or len(set(self.renderers)) != len(self.renderers):
            raise ValueError("production slice renderers must be nonempty and unique")
        if not self.terminal_kinds or len(set(self.terminal_kinds)) != len(
            self.terminal_kinds
        ):
            raise ValueError("production slice terminal kinds must be nonempty and unique")
        if type(self.count_basis) is not str or not self.count_basis or not self.count_basis.isascii():
            raise ValueError("count basis must be nonempty ASCII")

    @property
    def episode_count(self) -> int:
        return self.stop - self.start

    def as_obj(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "unit_id": self.unit_id,
            "start": self.start,
            "stop": self.stop,
            "episode_count": self.episode_count,
            "partition": self.partition,
            "renderers": list(self.renderers),
            "terminal_kinds": list(self.terminal_kinds),
            "count_basis": self.count_basis,
        }


@dataclass(frozen=True, slots=True)
class ProductionGenerationUnit:
    """One spec whose constructor enforces target uniqueness across its slices."""

    unit_id: str
    spec: EpisodeBankSpec

    def __post_init__(self) -> None:
        if type(self.unit_id) is not str or not self.unit_id or not self.unit_id.isascii():
            raise ValueError("production unit id must be nonempty ASCII")
        if type(self.spec) is not EpisodeBankSpec:
            raise TypeError("production generation unit requires an EpisodeBankSpec")

    def as_obj(self) -> dict[str, Any]:
        return {"unit_id": self.unit_id, "spec": self.spec.as_obj()}


@dataclass(frozen=True, slots=True)
class BlockedProductionStage:
    """A required stage that cannot be represented by the v1 engine API."""

    stage: ProductionStage
    planned_episode_count: int
    required_partition: str
    required_renderers: tuple[RendererName, ...]
    blocker_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.stage not in PRODUCTION_STAGES:
            raise ValueError(f"unknown production stage: {self.stage!r}")
        if (
            isinstance(self.planned_episode_count, bool)
            or not isinstance(self.planned_episode_count, int)
            or self.planned_episode_count <= 0
        ):
            raise ValueError("blocked stage count must be a positive integer")
        if (
            type(self.required_partition) is not str
            or not self.required_partition
            or not self.required_partition.isascii()
        ):
            raise ValueError("required partition must be nonempty ASCII")
        if not self.required_renderers or len(set(self.required_renderers)) != len(
            self.required_renderers
        ):
            raise ValueError("blocked stage renderers must be nonempty and unique")
        if not self.blocker_codes or any(
            type(code) is not str or not code or not code.isascii()
            for code in self.blocker_codes
        ):
            raise ValueError("blocked stage must name ASCII blocker codes")

    def as_obj(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "planned_episode_count": self.planned_episode_count,
            "required_partition": self.required_partition,
            "required_renderers": list(self.required_renderers),
            "blocker_codes": list(self.blocker_codes),
        }


@dataclass(frozen=True, slots=True)
class RequiredGeneratorExtension:
    code: str
    smallest_change: str

    def __post_init__(self) -> None:
        for name, value in (("code", self.code), ("smallest_change", self.smallest_change)):
            if type(value) is not str or not value or not value.isascii():
                raise ValueError(f"{name} must be nonempty ASCII")

    def as_obj(self) -> dict[str, str]:
        return {"code": self.code, "smallest_change": self.smallest_change}


@dataclass(frozen=True, slots=True)
class ProductionBankPlan:
    """Canonical request plan; never evidence that episodes were generated."""

    catalog_digest: str
    partitions_digest: str
    eligible_pairs_digest: str
    leakage_minimum_independent_groups: int
    units: tuple[ProductionGenerationUnit, ...]
    slices: tuple[ProductionStageSlice, ...]
    blocked_stages: tuple[BlockedProductionStage, ...]
    required_extensions: tuple[RequiredGeneratorExtension, ...]

    @property
    def planned_request_count(self) -> int:
        return sum(len(unit.spec.requests) for unit in self.units)

    @property
    def generated_episode_count(self) -> int:
        return 0

    @property
    def materialization_authorized(self) -> bool:
        return False

    def as_obj(self) -> dict[str, Any]:
        return {
            "schema_version": PRODUCTION_BANK_PLAN_SCHEMA_VERSION,
            "scope": "request-only-planning-no-episodes-generated",
            "catalog_digest": self.catalog_digest,
            "partitions_digest": self.partitions_digest,
            "eligible_pairs_digest": self.eligible_pairs_digest,
            "leakage_minimum_independent_groups": self.leakage_minimum_independent_groups,
            "planned_request_count": self.planned_request_count,
            "generated_episode_count": self.generated_episode_count,
            "materialization_authorized": self.materialization_authorized,
            "units": [unit.as_obj() for unit in self.units],
            "slices": [item.as_obj() for item in self.slices],
            "blocked_stages": [item.as_obj() for item in self.blocked_stages],
            "required_extensions": [item.as_obj() for item in self.required_extensions],
        }

    @property
    def digest(self) -> str:
        return json_digest(self.as_obj(), domain="goalzendo-production-bank-plan-v1")


@dataclass(frozen=True, slots=True)
class IdentityCapacity:
    partition: RulePartition
    target_op: BinaryOp
    requested: int
    available: int

    @property
    def sufficient(self) -> bool:
        return self.requested <= self.available

    def as_obj(self) -> dict[str, int | str | bool]:
        return {
            "partition": self.partition,
            "target_op": self.target_op,
            "requested": self.requested,
            "available": self.available,
            "sufficient": self.sufficient,
        }


@dataclass(frozen=True, slots=True)
class ProductionBankQAReport:
    plan_digest: str
    request_count: int
    request_ids_unique: bool
    one_generation_unit_per_partition: bool
    slices_exactly_balanced: bool
    leakage_group_resolution_satisfied: bool
    identity_capacity: tuple[IdentityCapacity, ...]
    blocked_stage_codes: tuple[str, ...]

    @property
    def structural_checks_passed(self) -> bool:
        return (
            self.request_ids_unique
            and self.one_generation_unit_per_partition
            and self.slices_exactly_balanced
            and self.leakage_group_resolution_satisfied
            and all(item.sufficient for item in self.identity_capacity)
        )

    @property
    def materialization_authorized(self) -> bool:
        return self.structural_checks_passed and not self.blocked_stage_codes

    def as_obj(self) -> dict[str, Any]:
        return {
            "schema_version": PRODUCTION_BANK_QA_SCHEMA_VERSION,
            "plan_digest": self.plan_digest,
            "request_count": self.request_count,
            "request_ids_unique": self.request_ids_unique,
            "one_generation_unit_per_partition": self.one_generation_unit_per_partition,
            "slices_exactly_balanced": self.slices_exactly_balanced,
            "leakage_group_resolution_satisfied": self.leakage_group_resolution_satisfied,
            "identity_capacity": [item.as_obj() for item in self.identity_capacity],
            "structural_checks_passed": self.structural_checks_passed,
            "blocked_stage_codes": list(self.blocked_stage_codes),
            "materialization_authorized": self.materialization_authorized,
            "boundary": (
                "This report validates a request plan only. It cannot authorize generation, "
                "training, or a G03 weight update."
            ),
        }

    @property
    def digest(self) -> str:
        return json_digest(self.as_obj(), domain="goalzendo-production-bank-qa-v1")


def _request_id(
    stage: ProductionStage,
    *,
    repeat: int,
    renderer_index: int,
    operator_slot: int,
    evidence_slot: int,
    terminal_index: int,
) -> str:
    return (
        f"{PRODUCTION_REQUEST_DOMAIN}-{stage}-k{repeat:02d}-r{renderer_index}-"
        f"o{operator_slot}-e{evidence_slot}-t{terminal_index}"
    )


def _balanced_requests(
    *,
    stage: ProductionStage,
    partition: RulePartition,
    renderers: tuple[RendererName, ...],
    terminal_kinds: tuple[TerminalKind, ...],
    episode_count: int,
) -> tuple[EpisodeRequest, ...]:
    block_size = (
        len(_OPERATOR_SLOTS)
        * len(_EVIDENCE_SLOTS)
        * len(renderers)
        * len(terminal_kinds)
    )
    if episode_count <= 0 or episode_count % block_size:
        raise ValueError(
            f"{stage} count {episode_count} must be a positive multiple of {block_size}"
        )
    requests: list[EpisodeRequest] = []
    for repeat in range(episode_count // block_size):
        for renderer_index, renderer in enumerate(renderers):
            for operator_slot, target_op in enumerate(_OPERATOR_SLOTS):
                for evidence_slot, (regime, error_target) in enumerate(_EVIDENCE_SLOTS):
                    for terminal_index, terminal_kind in enumerate(terminal_kinds):
                        requests.append(
                            EpisodeRequest(
                                request_id=_request_id(
                                    stage,
                                    repeat=repeat,
                                    renderer_index=renderer_index,
                                    operator_slot=operator_slot,
                                    evidence_slot=evidence_slot,
                                    terminal_index=terminal_index,
                                ),
                                partition=partition,
                                target_op=target_op,
                                regime=regime,
                                terminal_kind=terminal_kind,
                                renderer=renderer,
                                noisy_placard_error_target=error_target,
                            )
                        )
    return tuple(requests)


def _spec(bank_id: str, requests: tuple[EpisodeRequest, ...]) -> EpisodeBankSpec:
    return EpisodeBankSpec(
        bank_id=bank_id,
        requests=requests,
        candidate_pool_size=128,
        max_pair_attempts=4096,
        opening_attempts_per_pair=8,
        minimum_version_size=8,
        maximum_version_size=64,
        maximum_minimax_depth=4,
        maximum_reference_queries=6,
    )


def _slice(
    stage: ProductionStage,
    unit_id: str,
    start: int,
    requests: tuple[EpisodeRequest, ...],
    *,
    partition: RulePartition,
    renderers: tuple[RendererName, ...],
    terminal_kinds: tuple[TerminalKind, ...],
    count_basis: str,
) -> ProductionStageSlice:
    return ProductionStageSlice(
        stage=stage,
        unit_id=unit_id,
        start=start,
        stop=start + len(requests),
        partition=partition,
        renderers=renderers,
        terminal_kinds=terminal_kinds,
        count_basis=count_basis,
    )


@lru_cache(maxsize=1)
def build_production_bank_plan() -> ProductionBankPlan:
    """Build the deterministic v1 request plan without generating episodes."""

    units: list[ProductionGenerationUnit] = []
    slices: list[ProductionStageSlice] = []

    ordinary: tuple[
        tuple[
            ProductionStage,
            RulePartition,
            int,
            tuple[RendererName, ...],
            tuple[TerminalKind, ...],
            str,
        ],
        ...,
    ] = (
        (
            "engine_leakage",
            "engineering",
            ENGINE_LEAKAGE_EPISODES,
            TRAIN_RENDERERS,
            ("train_like", "factorial"),
            "derived-minimum-for-381-noisy-independent-groups",
        ),
        (
            "pilot",
            "pilot",
            PLANNING_PILOT_EPISODES,
            TRAIN_RENDERERS,
            ("train_like",),
            "planning-placeholder-unfrozen-by-g03-p",
        ),
        (
            "confirmatory_train",
            "confirmatory_train",
            PLANNING_CONFIRMATORY_TRAIN_EPISODES,
            TRAIN_RENDERERS,
            ("train_like",),
            "planning-placeholder-unfrozen-by-g03-p",
        ),
        (
            "validation",
            "validation",
            PLANNING_VALIDATION_EPISODES,
            TRAIN_RENDERERS,
            ("train_like",),
            "planning-placeholder-unfrozen-by-g03-p",
        ),
    )
    for stage, partition, count, renderers, terminal_kinds, count_basis in ordinary:
        unit_id = f"g03-production-{stage}-v1"
        requests = _balanced_requests(
            stage=stage,
            partition=partition,
            renderers=renderers,
            terminal_kinds=terminal_kinds,
            episode_count=count,
        )
        units.append(ProductionGenerationUnit(unit_id, _spec(unit_id, requests)))
        slices.append(
            _slice(
                stage,
                unit_id,
                0,
                requests,
                partition=partition,
                renderers=renderers,
                terminal_kinds=terminal_kinds,
                count_basis=count_basis,
            )
        )

    evaluation_unit_id = "g03-production-evaluation-suite-v1"
    evaluation_requests: list[EpisodeRequest] = []
    evaluation_rows: tuple[tuple[ProductionStage, TerminalKind], ...] = (
        ("evaluation_active", "factorial"),
        ("evaluation_oracle_replay", "factorial"),
        ("evaluation_no_query", "train_like"),
    )
    for stage, terminal_kind in evaluation_rows:
        requests = _balanced_requests(
            stage=stage,
            partition="evaluation",
            renderers=EVAL_RENDERERS,
            terminal_kinds=(terminal_kind,),
            episode_count=EVALUATION_EPISODES_PER_VIEW,
        )
        start = len(evaluation_requests)
        evaluation_requests.extend(requests)
        slices.append(
            _slice(
                stage,
                evaluation_unit_id,
                start,
                requests,
                partition="evaluation",
                renderers=EVAL_RENDERERS,
                terminal_kinds=(terminal_kind,),
                count_basis="protocol-fixed-256-episodes-per-evaluation-view",
            )
        )
    units.append(
        ProductionGenerationUnit(
            evaluation_unit_id,
            _spec(evaluation_unit_id, tuple(evaluation_requests)),
        )
    )

    blocked = (
        BlockedProductionStage(
            stage="format_warm_start",
            planned_episode_count=PLANNING_FORMAT_WARM_START_EPISODES,
            required_partition="warm_start",
            required_renderers=TRAIN_RENDERERS,
            blocker_codes=("unsupported_chance_balanced_proxy_profile",),
        ),
        BlockedProductionStage(
            stage="capability",
            planned_episode_count=PLANNING_CAPABILITY_EPISODES,
            required_partition="capability",
            required_renderers=EVAL_RENDERERS,
            blocker_codes=(
                "missing_capability_rule_partition",
                "unsupported_capability_official_law_families",
            ),
        ),
    )
    extensions = (
        RequiredGeneratorExtension(
            "unsupported_chance_balanced_proxy_profile",
            "Version EpisodeRequest and HiddenEpisode with a chance_balanced proxy profile "
            "whose opening and terminal constructors balance P and Q at chance.",
        ),
        RequiredGeneratorExtension(
            "missing_capability_rule_partition",
            "Add a capability identity to RulePartition v2 before re-hashing the complete "
            "truth-identity catalog; never alias capability to engineering.",
        ),
        RequiredGeneratorExtension(
            "unsupported_capability_official_law_families",
            "Version EpisodeRequest, pair selection, and HiddenEpisode validation with explicit "
            "binary_piece, literal_piece, and placard_literal target families.",
        ),
    )
    catalog = build_rule_catalog()
    partitions = build_rule_identity_partitions()
    pairs = build_eligible_pair_table()
    plan = ProductionBankPlan(
        catalog_digest=catalog.digest,
        partitions_digest=partitions.digest,
        eligible_pairs_digest=pairs.digest,
        leakage_minimum_independent_groups=minimum_resolution_groups(2),
        units=tuple(units),
        slices=tuple(slices),
        blocked_stages=blocked,
        required_extensions=extensions,
    )
    build_production_bank_qa_report(plan)
    return plan


def _unit_by_id(plan: ProductionBankPlan) -> dict[str, ProductionGenerationUnit]:
    units = {unit.unit_id: unit for unit in plan.units}
    if len(units) != len(plan.units):
        raise ValueError("production generation unit ids must be globally unique")
    return units


def requests_for_stage(
    plan: ProductionBankPlan,
    stage: ProductionStage,
) -> tuple[EpisodeRequest, ...]:
    """Return a representable stage slice; blocked stages fail explicitly."""

    if type(plan) is not ProductionBankPlan:
        raise TypeError("requests_for_stage requires a ProductionBankPlan")
    if stage not in PRODUCTION_STAGES:
        raise ValueError(f"unknown production stage: {stage!r}")
    blocked = {item.stage: item for item in plan.blocked_stages}
    if stage in blocked:
        codes = ",".join(blocked[stage].blocker_codes)
        raise ValueError(f"production stage {stage!r} is blocked: {codes}")
    matches = [item for item in plan.slices if item.stage == stage]
    if len(matches) != 1:
        raise ValueError(f"production stage {stage!r} must have exactly one slice")
    item = matches[0]
    unit = _unit_by_id(plan)[item.unit_id]
    if item.stop > len(unit.spec.requests):
        raise ValueError("production slice lies outside its generation unit")
    return unit.spec.requests[item.start : item.stop]


def _validate_slice_balance(
    item: ProductionStageSlice,
    requests: tuple[EpisodeRequest, ...],
) -> None:
    if len(requests) != item.episode_count:
        raise ValueError(f"{item.stage} slice count mismatch")
    if any(request.partition != item.partition for request in requests):
        raise ValueError(f"{item.stage} contains the wrong rule partition")
    if Counter(request.target_op for request in requests) != {
        "all": item.episode_count // 4,
        "any": item.episode_count // 4,
        "exactly_one": item.episode_count // 2,
    }:
        raise ValueError(f"{item.stage} operator/formula balance is not exact")
    if Counter(request.regime for request in requests) != {
        "perfect_ambiguity": item.episode_count // 2,
        "noisy_shortcuts": item.episode_count // 2,
    }:
        raise ValueError(f"{item.stage} opening-regime balance is not exact")
    if Counter(request.renderer for request in requests) != {
        renderer: item.episode_count // len(item.renderers) for renderer in item.renderers
    }:
        raise ValueError(f"{item.stage} renderer balance is not exact")
    if Counter(request.terminal_kind for request in requests) != {
        kind: item.episode_count // len(item.terminal_kinds) for kind in item.terminal_kinds
    }:
        raise ValueError(f"{item.stage} terminal-kind balance is not exact")
    noisy = [request for request in requests if request.regime == "noisy_shortcuts"]
    if Counter(request.noisy_placard_error_target for request in noisy) != {
        False: len(noisy) // 2,
        True: len(noisy) // 2,
    }:
        raise ValueError(f"{item.stage} noisy error-target balance is not exact")
    if any(
        request.noisy_placard_error_target is not None
        for request in requests
        if request.regime == "perfect_ambiguity"
    ):
        raise ValueError(f"{item.stage} perfect episodes cannot have an error target")

    block_size = (
        len(_OPERATOR_SLOTS)
        * len(_EVIDENCE_SLOTS)
        * len(item.renderers)
        * len(item.terminal_kinds)
    )
    repeats = item.episode_count // block_size
    expected: Counter[tuple[BinaryOp, RendererName, TerminalKind, OpeningRegime, bool | None]] = (
        Counter()
    )
    for target_op in _OPERATOR_SLOTS:
        for renderer in item.renderers:
            for terminal_kind in item.terminal_kinds:
                for regime, error_target in _EVIDENCE_SLOTS:
                    expected[(target_op, renderer, terminal_kind, regime, error_target)] += repeats
    observed = Counter(
        (
            request.target_op,
            request.renderer,
            request.terminal_kind,
            request.regime,
            request.noisy_placard_error_target,
        )
        for request in requests
    )
    if observed != expected:
        raise ValueError(f"{item.stage} joint balance is not the registered full factorial")


def _identity_capacities(plan: ProductionBankPlan) -> tuple[IdentityCapacity, ...]:
    catalog = build_rule_catalog()
    table = build_eligible_pair_table()
    available: dict[tuple[RulePartition, BinaryOp], set[str]] = defaultdict(set)
    for pair in table.pairs:
        target = catalog[pair.target_index]
        if type(target.rule) is not BinaryRule:  # pragma: no cover - pair-table invariant
            raise ValueError("eligible-pair target must be binary")
        target_op = target.rule.op
        available[(pair.partition, target_op)].add(target.truth_digest)
    requested = Counter(
        (request.partition, request.target_op)
        for unit in plan.units
        for request in unit.spec.requests
    )
    return tuple(
        IdentityCapacity(
            partition=partition,
            target_op=target_op,
            requested=requested[(partition, target_op)],
            available=len(available[(partition, target_op)]),
        )
        for partition in RULE_PARTITIONS
        for target_op in cast(tuple[BinaryOp, ...], ("all", "any", "exactly_one"))
        if requested[(partition, target_op)]
    )


def build_production_bank_qa_report(plan: ProductionBankPlan) -> ProductionBankQAReport:
    """Validate every planning invariant and return a non-authorizing QA report."""

    if type(plan) is not ProductionBankPlan:
        raise TypeError("build_production_bank_qa_report requires a ProductionBankPlan")
    if plan.catalog_digest != build_rule_catalog().digest:
        raise ValueError("production plan catalog digest is stale")
    if plan.partitions_digest != build_rule_identity_partitions().digest:
        raise ValueError("production plan partition digest is stale")
    if plan.eligible_pairs_digest != build_eligible_pair_table().digest:
        raise ValueError("production plan eligible-pair digest is stale")
    if plan.leakage_minimum_independent_groups != minimum_resolution_groups(2):
        raise ValueError("production plan leakage-resolution minimum is stale")

    units = _unit_by_id(plan)
    stage_names = tuple(item.stage for item in plan.slices) + tuple(
        item.stage for item in plan.blocked_stages
    )
    if len(stage_names) != len(PRODUCTION_STAGES) or set(stage_names) != set(
        PRODUCTION_STAGES
    ):
        raise ValueError("production stages must occur exactly once")
    expected_slices: dict[
        ProductionStage,
        tuple[
            RulePartition,
            int,
            tuple[RendererName, ...],
            tuple[TerminalKind, ...],
        ],
    ] = {
        "engine_leakage": (
            "engineering",
            ENGINE_LEAKAGE_EPISODES,
            TRAIN_RENDERERS,
            ("train_like", "factorial"),
        ),
        "pilot": (
            "pilot",
            PLANNING_PILOT_EPISODES,
            TRAIN_RENDERERS,
            ("train_like",),
        ),
        "confirmatory_train": (
            "confirmatory_train",
            PLANNING_CONFIRMATORY_TRAIN_EPISODES,
            TRAIN_RENDERERS,
            ("train_like",),
        ),
        "validation": (
            "validation",
            PLANNING_VALIDATION_EPISODES,
            TRAIN_RENDERERS,
            ("train_like",),
        ),
        "evaluation_active": (
            "evaluation",
            EVALUATION_EPISODES_PER_VIEW,
            EVAL_RENDERERS,
            ("factorial",),
        ),
        "evaluation_oracle_replay": (
            "evaluation",
            EVALUATION_EPISODES_PER_VIEW,
            EVAL_RENDERERS,
            ("factorial",),
        ),
        "evaluation_no_query": (
            "evaluation",
            EVALUATION_EPISODES_PER_VIEW,
            EVAL_RENDERERS,
            ("train_like",),
        ),
    }
    for item in plan.slices:
        if item.unit_id not in units:
            raise ValueError("production slice names an unknown generation unit")
        expected_shape = expected_slices.get(item.stage)
        if expected_shape is None:
            raise ValueError(f"{item.stage} cannot be a representable v1 production slice")
        if (
            item.partition,
            item.episode_count,
            item.renderers,
            item.terminal_kinds,
        ) != expected_shape:
            raise ValueError(f"{item.stage} differs from its registered planning shape")
        requests = requests_for_stage(plan, item.stage)
        _validate_slice_balance(item, requests)
        expected_requests = _balanced_requests(
            stage=item.stage,
            partition=item.partition,
            renderers=item.renderers,
            terminal_kinds=item.terminal_kinds,
            episode_count=item.episode_count,
        )
        if requests != expected_requests:
            raise ValueError(
                f"{item.stage} differs from the deterministic request construction"
            )

    slices_by_unit: dict[str, list[ProductionStageSlice]] = defaultdict(list)
    for item in plan.slices:
        slices_by_unit[item.unit_id].append(item)
    for unit in plan.units:
        cursor = 0
        for item in sorted(slices_by_unit[unit.unit_id], key=lambda value: value.start):
            if item.start != cursor:
                raise ValueError("production slices must cover each unit contiguously once")
            cursor = item.stop
        if cursor != len(unit.spec.requests):
            raise ValueError("production slices must cover each unit contiguously once")
        if unit.spec != _spec(unit.unit_id, unit.spec.requests):
            raise ValueError("production generation-unit search settings are not canonical")

    request_ids = [
        request.request_id for unit in plan.units for request in unit.spec.requests
    ]
    request_ids_unique = len(request_ids) == len(set(request_ids))
    if not request_ids_unique:
        raise ValueError("production request ids must be globally unique")

    partition_units: dict[RulePartition, set[str]] = defaultdict(set)
    for unit in plan.units:
        for request in unit.spec.requests:
            partition_units[request.partition].add(unit.unit_id)
    one_unit_per_partition = all(len(unit_ids) == 1 for unit_ids in partition_units.values())
    if not one_unit_per_partition:
        raise ValueError(
            "each represented partition must use one combined generation unit so target "
            "identity reservation cannot reset between banks"
        )

    capacities = _identity_capacities(plan)
    if not all(item.sufficient for item in capacities):
        raise ValueError("production request demand exceeds eligible unique target identities")

    engine = requests_for_stage(plan, "engine_leakage")
    noisy_engine_groups = sum(
        request.regime == "noisy_shortcuts" for request in engine
    )
    noisy_error_groups = Counter(
        request.noisy_placard_error_target
        for request in engine
        if request.regime == "noisy_shortcuts"
    )
    leakage_satisfied = (
        len(engine) >= plan.leakage_minimum_independent_groups
        and noisy_engine_groups >= plan.leakage_minimum_independent_groups
        and noisy_error_groups[False]
        >= (plan.leakage_minimum_independent_groups + 1) // 2
        and noisy_error_groups[True]
        >= (plan.leakage_minimum_independent_groups + 1) // 2
    )
    if not leakage_satisfied:
        raise ValueError("engine/leakage slice is underpowered for a registered target")

    blocker_codes = tuple(
        code for stage in plan.blocked_stages for code in stage.blocker_codes
    )
    blocked_shapes = {
        item.stage: (
            item.planned_episode_count,
            item.required_partition,
            item.required_renderers,
            item.blocker_codes,
        )
        for item in plan.blocked_stages
    }
    if blocked_shapes != {
        "format_warm_start": (
            PLANNING_FORMAT_WARM_START_EPISODES,
            "warm_start",
            TRAIN_RENDERERS,
            ("unsupported_chance_balanced_proxy_profile",),
        ),
        "capability": (
            PLANNING_CAPABILITY_EPISODES,
            "capability",
            EVAL_RENDERERS,
            (
                "missing_capability_rule_partition",
                "unsupported_capability_official_law_families",
            ),
        ),
    }:
        raise ValueError("blocked production-stage shapes are not canonical")
    extension_codes = tuple(item.code for item in plan.required_extensions)
    if blocker_codes != extension_codes:
        raise ValueError("blocked-stage codes and required extensions must match canonically")

    return ProductionBankQAReport(
        plan_digest=plan.digest,
        request_count=len(request_ids),
        request_ids_unique=request_ids_unique,
        one_generation_unit_per_partition=one_unit_per_partition,
        slices_exactly_balanced=True,
        leakage_group_resolution_satisfied=leakage_satisfied,
        identity_capacity=capacities,
        blocked_stage_codes=blocker_codes,
    )


def serialize_production_bank_plan(plan: ProductionBankPlan) -> str:
    if type(plan) is not ProductionBankPlan:
        raise TypeError("serialize_production_bank_plan requires a ProductionBankPlan")
    build_production_bank_qa_report(plan)
    return dump_json(plan.as_obj())
