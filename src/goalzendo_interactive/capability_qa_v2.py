"""Fail-closed materialization and QA for schema-v2 capability candidates.

This module is deliberately additive.  It does not change the released v2
partition, episode, or 256-request plan.  It materializes that plan as
non-authorizing evidence, preserves a symmetric 384-stage candidate that is
still underpowered for the binary-rule formula estimand, and defines a 384
warm-start / 402 capability candidate that reaches the registered floor.

Surface-classifier inputs are rebuilt from an allowlist.  Canonical scenes,
labels, rules, proxy orientations and profiles, identifiers, digests, and
generation provenance never enter a classifier feature vector.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal, cast

from ._json import json_digest
from .capability import (
    EpisodeBankSpecV2,
    EpisodeBankV2,
    EpisodeRequestV2,
    HiddenEpisodeV2,
    ProxyOrientationV2,
    build_stage_request_plan_v2,
    generate_episode_bank_v2,
    verify_episode_bank_v2,
)
from .leakage import (
    EPISODE_SURFACE_FEATURE_KEYS,
    SurfaceLeakageAuditReport,
    SurfaceLeakageConfig,
    SurfaceLeakageSample,
    SurfaceLeakageTarget,
    audit_surface_targets,
    minimum_resolution_groups,
)
from .rendering import EVAL_RENDERERS, TRAIN_RENDERERS, RendererName
from .rules import ATOM_OPS, BinaryOp, BinaryRule
from .rules import Literal as RuleLiteral
from .schema import NONEMPTY_ARRANGEMENT_COUNT, SCENE_COUNT
from .stage_partitions_v2 import (
    TARGET_FAMILIES_V2,
    TargetFamilyV2,
    build_eligible_target_shadow_table_v2,
    target_shadow_cells_v2,
)

CAPABILITY_QA_SCHEMA_VERSION_V2 = 2
POWERED_WARM_EPISODES_V2 = 384
POWERED_CAPABILITY_EPISODES_V2 = 402
REGISTERED_BINARY_GROUP_FLOOR = 381

CAPABILITY_V2_SURFACE_TARGETS = (
    "target_label",
    "target_formula_stratum",
    "proxy_orientation",
    "placard_error_status",
    "shadow_error_status",
)
CAPABILITY_V2_SURFACE_FEATURE_KEYS = EPISODE_SURFACE_FEATURE_KEYS
CAPABILITY_V2_MASK_POLICY = (
    "allowlist-only: renderer and structural count/phase/position tokens; "
    "scene, target-label, Official-Law, placard, shadow, proxy, identifier, "
    "digest, request-stratum, and generation-provenance content are omitted"
)

CandidateNameV2 = Literal["registered_256", "symmetric_384", "powered_402"]


def _bool_label(value: bool) -> str:
    return "true" if value else "false"


def _episode_surface_features(episode: HiddenEpisodeV2) -> tuple[str, ...]:
    return (
        f"renderer={episode.request.renderer}",
        f"opening_count={len(episode.opening)}",
        f"terminal_count={len(episode.terminal)}",
    )


def _observation_surface_features(
    episode_features: tuple[str, ...],
    *,
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


def hidden_episode_bank_v2_surface_targets(
    bank: EpisodeBankV2,
) -> tuple[SurfaceLeakageTarget, ...]:
    """Adapt the registered grouped audit without passing sensitive content.

    Proxy orientation is an episode-level *outcome* in this v2 registry.  It is
    not a feature.  The other four outcomes retain their v1 definitions; a
    one-literal Official Law is not assigned to a binary formula stratum.  The
    formula target therefore contains binary-piece Official Laws only.
    """

    if type(bank) is not EpisodeBankV2:
        raise TypeError(
            "hidden_episode_bank_v2_surface_targets requires an EpisodeBankV2"
        )

    target_label: list[SurfaceLeakageSample] = []
    formula_stratum: list[SurfaceLeakageSample] = []
    proxy_orientation: list[SurfaceLeakageSample] = []
    placard_error_status: list[SurfaceLeakageSample] = []
    shadow_error_status: list[SurfaceLeakageSample] = []

    for episode_index, episode in enumerate(bank.episodes):
        group_id = f"episode-{episode_index}"
        episode_features = _episode_surface_features(episode)
        if episode.request.target_family == "binary_piece":
            formula_stratum.append(
                SurfaceLeakageSample(
                    sample_id=f"{group_id}-formula",
                    group_id=group_id,
                    features=episode_features,
                    label=(
                        "exactly_one"
                        if episode.request.target_op == "exactly_one"
                        else "all_or_any"
                    ),
                )
            )
        proxy_orientation.append(
            SurfaceLeakageSample(
                sample_id=f"{group_id}-proxy-orientation",
                group_id=group_id,
                features=episode_features,
                label=episode.request.proxy_orientation,
            )
        )

        phases = (("opening", episode.opening), ("terminal", episode.terminal))
        for phase, observations in phases:
            for position, observation in enumerate(observations):
                sample_id = f"{group_id}-{phase}-{position}"
                features = _observation_surface_features(
                    episode_features,
                    phase=phase,
                    phase_count=len(observations),
                    position=position,
                )
                is_sun = observation.scene_index < NONEMPTY_ARRANGEMENT_COUNT
                target_label.append(
                    SurfaceLeakageSample(
                        sample_id=sample_id,
                        group_id=group_id,
                        features=features,
                        label=_bool_label(observation.accepted),
                    )
                )
                placard_error_status.append(
                    SurfaceLeakageSample(
                        sample_id=sample_id,
                        group_id=group_id,
                        features=features,
                        label=_bool_label(is_sun is not observation.accepted),
                    )
                )
                shadow_error_status.append(
                    SurfaceLeakageSample(
                        sample_id=sample_id,
                        group_id=group_id,
                        features=features,
                        label=_bool_label(
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
            tuple(formula_stratum),
        ),
        SurfaceLeakageTarget(
            "proxy_orientation",
            ("proxy_low", "proxy_high"),
            tuple(proxy_orientation),
        ),
        SurfaceLeakageTarget(
            "placard_error_status", binary, tuple(placard_error_status)
        ),
        SurfaceLeakageTarget(
            "shadow_error_status", binary, tuple(shadow_error_status)
        ),
    )
    if tuple(target.name for target in targets) != CAPABILITY_V2_SURFACE_TARGETS:
        raise RuntimeError("schema-v2 surface-target registry is inconsistent")
    return targets


def audit_hidden_episode_bank_v2_surface_leakage(
    bank: EpisodeBankV2,
    *,
    config: SurfaceLeakageConfig | None = None,
) -> SurfaceLeakageAuditReport:
    """Run every registered v2 surface target with episode grouping."""

    if type(bank) is not EpisodeBankV2:
        raise TypeError(
            "audit_hidden_episode_bank_v2_surface_leakage requires an EpisodeBankV2"
        )
    return audit_surface_targets(
        bank.spec.bank_id,
        hidden_episode_bank_v2_surface_targets(bank),
        config=config,
    )


@dataclass(frozen=True, slots=True)
class IntegerHistogramV2:
    """An exact integer distribution with no floating-point summaries."""

    counts: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        counts = tuple(self.counts)
        object.__setattr__(self, "counts", counts)
        if not counts or tuple(sorted(counts)) != counts:
            raise ValueError("integer histogram must be nonempty and sorted")
        if len({value for value, _ in counts}) != len(counts):
            raise ValueError("integer histogram values must be unique")
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            for value, count in counts
        ):
            raise ValueError("integer histogram values/counts must be valid integers")

    @classmethod
    def from_values(cls, values: tuple[int, ...]) -> IntegerHistogramV2:
        if not values:
            raise ValueError("cannot construct an empty integer histogram")
        counter = Counter(values)
        return cls(tuple(sorted(counter.items())))

    @property
    def sample_count(self) -> int:
        return sum(count for _, count in self.counts)

    @property
    def minimum(self) -> int:
        return self.counts[0][0]

    @property
    def maximum(self) -> int:
        return self.counts[-1][0]

    @property
    def total(self) -> int:
        return sum(value * count for value, count in self.counts)

    def as_obj(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "total": self.total,
            "counts": [
                {"value": value, "count": count} for value, count in self.counts
            ],
        }


def _ordered_counts(counter: Counter[Any], keys: tuple[str, ...]) -> dict[str, int]:
    return {key: counter[key] for key in keys}


def _joint_counts(bank: EpisodeBankV2) -> dict[str, int]:
    counts = Counter(
        f"{episode.request.renderer}:{episode.request.proxy_orientation}"
        for episode in bank.episodes
    )
    renderers = (
        TRAIN_RENDERERS
        if bank.episodes[0].request.stage == "format_warm_start"
        else EVAL_RENDERERS
    )
    return {
        f"{renderer}:{orientation}": counts[f"{renderer}:{orientation}"]
        for renderer in renderers
        for orientation in ("proxy_low", "proxy_high")
    }


def _ypq_profile(episode: HiddenEpisodeV2, phase: str) -> tuple[int, ...]:
    observations = episode.opening if phase == "opening" else episode.terminal
    counts = [0] * 8
    for observation in observations:
        y = int(observation.accepted)
        p = int(observation.scene_index < NONEMPTY_ARRANGEMENT_COUNT)
        q = int(episode.shadow.truth[observation.scene_index])
        counts[y * 4 + p * 2 + q] += 1
    return tuple(counts)


@dataclass(frozen=True, slots=True)
class BankDistributionMetricsV2:
    """Exact structural and semantic descriptors for one materialized bank."""

    bank_id: str
    spec_digest: str
    bank_digest: str
    episode_count: int
    target_family_counts: dict[str, int]
    target_operator_counts: dict[str, int]
    renderer_counts: dict[str, int]
    proxy_orientation_counts: dict[str, int]
    renderer_orientation_counts: dict[str, int]
    target_true_count: IntegerHistogramV2
    canonical_rule_json_length: IntegerHistogramV2
    atom_operator_presence: dict[str, int]
    target_shadow_cells: tuple[IntegerHistogramV2, ...]
    opening_ypq_profiles: dict[str, int]
    terminal_ypq_profiles: dict[str, int]
    pair_rank: IntegerHistogramV2
    exact_request_binding: bool
    unique_official_laws: bool
    exact_phase_sizes_and_labels: bool
    exact_renderer_orientation_balance: bool

    @property
    def structural_invariants_passed(self) -> bool:
        return (
            self.exact_request_binding
            and self.unique_official_laws
            and self.exact_phase_sizes_and_labels
            and self.exact_renderer_orientation_balance
        )

    def as_obj(self) -> dict[str, Any]:
        return {
            "bank_id": self.bank_id,
            "spec_digest": self.spec_digest,
            "bank_digest": self.bank_digest,
            "episode_count": self.episode_count,
            "target_family_counts": self.target_family_counts,
            "target_operator_counts": self.target_operator_counts,
            "renderer_counts": self.renderer_counts,
            "proxy_orientation_counts": self.proxy_orientation_counts,
            "renderer_orientation_counts": self.renderer_orientation_counts,
            "target_true_count_numerator_over_scene_count": {
                "denominator": SCENE_COUNT,
                "histogram": self.target_true_count.as_obj(),
            },
            "canonical_rule_json_length": self.canonical_rule_json_length.as_obj(),
            "atom_operator_presence": self.atom_operator_presence,
            "target_shadow_cells_y0q0_y0q1_y1q0_y1q1": [
                item.as_obj() for item in self.target_shadow_cells
            ],
            "opening_ypq_profiles": self.opening_ypq_profiles,
            "terminal_ypq_profiles": self.terminal_ypq_profiles,
            "pair_rank": self.pair_rank.as_obj(),
            "structural_invariants": {
                "exact_request_binding": self.exact_request_binding,
                "unique_official_laws": self.unique_official_laws,
                "exact_phase_sizes_and_labels": self.exact_phase_sizes_and_labels,
                "exact_renderer_orientation_balance": (
                    self.exact_renderer_orientation_balance
                ),
                "passed": self.structural_invariants_passed,
            },
        }

    @property
    def digest(self) -> str:
        return json_digest(
            self.as_obj(), domain="goalzendo-interactive-capability-distribution-v2"
        )


def bank_distribution_metrics_v2(bank: EpisodeBankV2) -> BankDistributionMetricsV2:
    """Compute exact distributions and validate non-statistical invariants."""

    if type(bank) is not EpisodeBankV2:
        raise TypeError("bank_distribution_metrics_v2 requires an EpisodeBankV2")
    verify_episode_bank_v2(bank)
    episodes = bank.episodes
    family_counts = Counter(episode.request.target_family for episode in episodes)
    operator_counts = Counter(
        episode.request.target_op or "one_literal" for episode in episodes
    )
    renderer_counts = Counter(episode.request.renderer for episode in episodes)
    orientation_counts = Counter(
        episode.request.proxy_orientation for episode in episodes
    )

    atom_presence: Counter[str] = Counter()
    cells: list[list[int]] = [[], [], [], []]
    opening_profiles: Counter[str] = Counter()
    terminal_profiles: Counter[str] = Counter()
    for episode in episodes:
        rule = episode.target.rule
        literals = (rule,) if type(rule) is RuleLiteral else cast(BinaryRule, rule).args
        for op in {literal.atom.op for literal in literals}:
            atom_presence[op] += 1
        for index, value in enumerate(target_shadow_cells_v2(episode.target, episode.shadow)):
            cells[index].append(value)
        opening_profiles[",".join(map(str, _ypq_profile(episode, "opening")))] += 1
        terminal_profiles[",".join(map(str, _ypq_profile(episode, "terminal")))] += 1

    stage = episodes[0].request.stage
    renderers = TRAIN_RENDERERS if stage == "format_warm_start" else EVAL_RENDERERS
    joint_counts = _joint_counts(bank)
    exact_joint = (
        len(set(renderer_counts[renderer] for renderer in renderers)) == 1
        and orientation_counts["proxy_low"] == orientation_counts["proxy_high"]
        and max(joint_counts.values()) - min(joint_counts.values()) <= 1
    )
    exact_phase = all(
        len(phase) == 10 and sum(item.accepted for item in phase) == 5
        for episode in episodes
        for phase in (episode.opening, episode.terminal)
    )
    exact_binding = all(
        request == episode.request == bank.spec.requests[index]
        for index, (request, episode) in enumerate(
            zip(bank.spec.requests, episodes, strict=True)
        )
    )
    return BankDistributionMetricsV2(
        bank_id=bank.spec.bank_id,
        spec_digest=bank.spec.digest,
        bank_digest=bank.digest,
        episode_count=len(episodes),
        target_family_counts=_ordered_counts(family_counts, TARGET_FAMILIES_V2),
        target_operator_counts=_ordered_counts(
            operator_counts, ("all", "any", "exactly_one", "one_literal")
        ),
        renderer_counts=_ordered_counts(renderer_counts, cast(tuple[str, ...], renderers)),
        proxy_orientation_counts=_ordered_counts(
            orientation_counts, ("proxy_low", "proxy_high")
        ),
        renderer_orientation_counts=joint_counts,
        target_true_count=IntegerHistogramV2.from_values(
            tuple(episode.target.truth.true_count for episode in episodes)
        ),
        canonical_rule_json_length=IntegerHistogramV2.from_values(
            tuple(len(episode.target.rule.canonical_json) for episode in episodes)
        ),
        atom_operator_presence=_ordered_counts(atom_presence, cast(tuple[str, ...], ATOM_OPS)),
        target_shadow_cells=tuple(
            IntegerHistogramV2.from_values(tuple(values)) for values in cells
        ),
        opening_ypq_profiles=dict(sorted(opening_profiles.items())),
        terminal_ypq_profiles=dict(sorted(terminal_profiles.items())),
        pair_rank=IntegerHistogramV2.from_values(
            tuple(record.pair_rank for record in bank.generation_records)
        ),
        exact_request_binding=exact_binding,
        unique_official_laws=(
            len({episode.target.truth_digest for episode in episodes}) == len(episodes)
        ),
        exact_phase_sizes_and_labels=exact_phase,
        exact_renderer_orientation_balance=exact_joint,
    )


@dataclass(frozen=True, slots=True)
class SemanticComparisonV2:
    """Descriptive cross-stage metrics; no post-hoc acceptance threshold."""

    warm_distribution_digest: str
    capability_distribution_digest: str
    warm_binary_episode_count: int
    capability_binary_episode_count: int
    target_true_count_total: tuple[int, int]
    canonical_rule_json_length_total: tuple[int, int]
    atom_operator_presence: tuple[dict[str, int], dict[str, int]]
    target_shadow_cell_totals: tuple[tuple[int, ...], tuple[int, ...]]

    @property
    def acceptance_rule_preregistered(self) -> bool:
        return False

    @property
    def passed(self) -> bool:
        return False

    def as_obj(self) -> dict[str, Any]:
        return {
            "warm_distribution_digest": self.warm_distribution_digest,
            "capability_distribution_digest": self.capability_distribution_digest,
            "binary_episode_counts": {
                "warm_start": self.warm_binary_episode_count,
                "capability": self.capability_binary_episode_count,
            },
            "target_true_count_total": {
                "warm_start": self.target_true_count_total[0],
                "capability": self.target_true_count_total[1],
                "denominator_per_episode": SCENE_COUNT,
            },
            "canonical_rule_json_length_total": {
                "warm_start": self.canonical_rule_json_length_total[0],
                "capability": self.canonical_rule_json_length_total[1],
            },
            "atom_operator_presence": {
                "warm_start": self.atom_operator_presence[0],
                "capability": self.atom_operator_presence[1],
            },
            "target_shadow_cell_totals": {
                "warm_start": list(self.target_shadow_cell_totals[0]),
                "capability": list(self.target_shadow_cell_totals[1]),
            },
            "acceptance_rule_preregistered": self.acceptance_rule_preregistered,
            "passed": self.passed,
            "reason": (
                "descriptive semantic distributions are pinned, but no prospective "
                "cross-stage equivalence bound was registered before materialization"
            ),
        }

    @property
    def digest(self) -> str:
        return json_digest(
            self.as_obj(), domain="goalzendo-interactive-semantic-comparison-v2"
        )


def _binary_metric_values(
    bank: EpisodeBankV2,
) -> tuple[tuple[int, ...], tuple[int, ...], Counter[str], tuple[int, ...]]:
    episodes = tuple(
        episode for episode in bank.episodes if episode.request.target_family == "binary_piece"
    )
    presence: Counter[str] = Counter()
    cell_totals = [0, 0, 0, 0]
    for episode in episodes:
        for op in {literal.atom.op for literal in cast(BinaryRule, episode.target.rule).args}:
            presence[op] += 1
        for index, value in enumerate(target_shadow_cells_v2(episode.target, episode.shadow)):
            cell_totals[index] += value
    return (
        tuple(episode.target.truth.true_count for episode in episodes),
        tuple(len(episode.target.rule.canonical_json) for episode in episodes),
        presence,
        tuple(cell_totals),
    )


def semantic_comparison_v2(
    warm: EpisodeBankV2,
    capability: EpisodeBankV2,
    warm_metrics: BankDistributionMetricsV2,
    capability_metrics: BankDistributionMetricsV2,
) -> SemanticComparisonV2:
    warm_true, warm_length, warm_atoms, warm_cells = _binary_metric_values(warm)
    cap_true, cap_length, cap_atoms, cap_cells = _binary_metric_values(capability)
    return SemanticComparisonV2(
        warm_distribution_digest=warm_metrics.digest,
        capability_distribution_digest=capability_metrics.digest,
        warm_binary_episode_count=len(warm_true),
        capability_binary_episode_count=len(cap_true),
        target_true_count_total=(sum(warm_true), sum(cap_true)),
        canonical_rule_json_length_total=(sum(warm_length), sum(cap_length)),
        atom_operator_presence=(
            _ordered_counts(warm_atoms, cast(tuple[str, ...], ATOM_OPS)),
            _ordered_counts(cap_atoms, cast(tuple[str, ...], ATOM_OPS)),
        ),
        target_shadow_cell_totals=(warm_cells, cap_cells),
    )


def _powered_warm_requests_v2() -> tuple[EpisodeRequestV2, ...]:
    requests: list[EpisodeRequestV2] = []
    operator_slots: tuple[BinaryOp, ...] = (
        "all",
        "any",
        "exactly_one",
        "exactly_one",
    )
    index = 0
    for _repeat in range(12):
        for renderer in TRAIN_RENDERERS:
            for orientation in ("proxy_low", "proxy_high"):
                for op in operator_slots:
                    requests.append(
                        EpisodeRequestV2(
                            request_id=f"g03-v2-powered-warm-{index:04d}",
                            stage="format_warm_start",
                            partition="warm_start",
                            target_family="binary_piece",
                            target_op=op,
                            proxy_profile="chance_balanced",
                            proxy_orientation=cast(ProxyOrientationV2, orientation),
                            renderer=renderer,
                        )
                    )
                    index += 1
    return tuple(requests)


def _symmetric_capability_requests_v2() -> tuple[EpisodeRequestV2, ...]:
    # The initially requested symmetric design uses 122 laws under each binary
    # operator plus 18 controls.  Its formula estimand has only 366 independent
    # binary-rule groups and is therefore preserved as rejected evidence.
    joint: tuple[
        tuple[ProxyOrientationV2, RendererName, tuple[int, int, int, int, int]], ...
    ] = (
        ("proxy_low", "eval_reverse", (31, 30, 30, 4, 1)),
        ("proxy_high", "eval_ledger", (30, 31, 31, 4, 0)),
        ("proxy_low", "eval_ledger", (30, 31, 31, 4, 0)),
        ("proxy_high", "eval_reverse", (31, 30, 30, 4, 1)),
    )
    requests: list[EpisodeRequestV2] = []
    index = 0
    for orientation, renderer, quotas in joint:
        slots: tuple[tuple[str, BinaryOp | None, int], ...] = (
            ("binary_piece", "all", quotas[0]),
            ("binary_piece", "any", quotas[1]),
            ("binary_piece", "exactly_one", quotas[2]),
            ("literal_piece", None, quotas[3]),
            ("placard_literal", None, quotas[4]),
        )
        for family, op, count in slots:
            for _ in range(count):
                requests.append(
                    EpisodeRequestV2(
                        request_id=f"g03-v2-symmetric-capability-{index:04d}",
                        stage="capability",
                        partition="capability",
                        target_family=cast(Any, family),
                        target_op=op,
                        proxy_profile="oracle_diagnostic",
                        proxy_orientation=orientation,
                        renderer=renderer,
                    )
                )
                index += 1
    return tuple(requests)


@lru_cache(maxsize=1)
def build_powered_warm_spec_v2() -> EpisodeBankSpecV2:
    """Return the shared powered warm-start spec used by both later candidates."""

    return EpisodeBankSpecV2(
        "g03-v2-powered-format-warm-start-384",
        _powered_warm_requests_v2(),
    )


@lru_cache(maxsize=1)
def build_symmetric_stage_specs_v2() -> tuple[EpisodeBankSpecV2, EpisodeBankSpecV2]:
    """Return the symmetric 384-stage candidate retained as rejected evidence."""

    warm = build_powered_warm_spec_v2()
    capability = EpisodeBankSpecV2(
        "g03-v2-symmetric-capability-384",
        _symmetric_capability_requests_v2(),
    )
    return warm, capability


def _powered_capability_requests_v2() -> tuple[EpisodeRequestV2, ...]:
    # The 18 controls remain outside the binary formula estimand.  Each joint
    # cell contains 24 all, 24 any, 48 exactly_one, and four literal controls.
    # One placard control is added to two opposite renderer/orientation
    # marginals, yielding exact 201/201 renderer and orientation balance.
    joint: tuple[
        tuple[ProxyOrientationV2, RendererName, tuple[int, int, int, int, int]], ...
    ] = (
        ("proxy_low", "eval_reverse", (24, 24, 48, 4, 1)),
        ("proxy_high", "eval_ledger", (24, 24, 48, 4, 1)),
        ("proxy_low", "eval_ledger", (24, 24, 48, 4, 0)),
        ("proxy_high", "eval_reverse", (24, 24, 48, 4, 0)),
    )
    requests: list[EpisodeRequestV2] = []
    index = 0
    for orientation, renderer, quotas in joint:
        slots: tuple[tuple[str, BinaryOp | None, int], ...] = (
            ("binary_piece", "all", quotas[0]),
            ("binary_piece", "any", quotas[1]),
            ("binary_piece", "exactly_one", quotas[2]),
            ("literal_piece", None, quotas[3]),
            ("placard_literal", None, quotas[4]),
        )
        for family, op, count in slots:
            for _ in range(count):
                requests.append(
                    EpisodeRequestV2(
                        request_id=f"g03-v2-powered-capability-{index:04d}",
                        stage="capability",
                        partition="capability",
                        target_family=cast(Any, family),
                        target_op=op,
                        proxy_profile="oracle_diagnostic",
                        proxy_orientation=orientation,
                        renderer=renderer,
                    )
                )
                index += 1
    return tuple(requests)


@lru_cache(maxsize=1)
def build_powered_stage_specs_v2() -> tuple[EpisodeBankSpecV2, EpisodeBankSpecV2]:
    """Return the non-authorizing 384-warm / 402-capability candidate."""

    warm = build_powered_warm_spec_v2()
    capability = EpisodeBankSpecV2(
        "g03-v2-powered-capability-402",
        _powered_capability_requests_v2(),
    )
    return warm, capability


@dataclass(frozen=True, slots=True)
class CandidateCapacityRowV2:
    partition: Literal["warm_start", "capability"]
    target_family: TargetFamilyV2
    target_op: BinaryOp | None
    requested: int
    available: int

    @property
    def sufficient(self) -> bool:
        return self.requested <= self.available

    def as_obj(self) -> dict[str, Any]:
        return {
            "partition": self.partition,
            "target_family": self.target_family,
            "target_op": self.target_op,
            "requested": self.requested,
            "available": self.available,
            "sufficient": self.sufficient,
        }


def candidate_identity_capacity_v2(
    warm: EpisodeBankSpecV2,
    capability: EpisodeBankSpecV2,
) -> tuple[CandidateCapacityRowV2, ...]:
    """Audit unique eligible Official-Law identities for both candidate units."""

    table = build_eligible_target_shadow_table_v2()
    requested: Counter[tuple[str, str, str | None]] = Counter(
        (
            str(request.partition),
            str(request.target_family),
            None if request.target_op is None else str(request.target_op),
        )
        for spec in (warm, capability)
        for request in spec.requests
    )
    rows: list[CandidateCapacityRowV2] = []
    partitions: tuple[Literal["warm_start", "capability"], ...] = (
        "warm_start",
        "capability",
    )
    operators: tuple[BinaryOp, ...] = ("all", "any", "exactly_one")
    for partition in partitions:
        for op in operators:
            count = requested[(partition, "binary_piece", op)]
            if count:
                rows.append(
                    CandidateCapacityRowV2(
                        partition,
                        "binary_piece",
                        op,
                        count,
                        table.binary_operator_target_identity_counts[
                            f"{partition}:{op}"
                        ],
                    )
                )
        for family in ("literal_piece", "placard_literal"):
            count = requested[(partition, family, None)]
            if count:
                rows.append(
                    CandidateCapacityRowV2(
                        partition,
                        cast(TargetFamilyV2, family),
                        None,
                        count,
                        table.target_identity_counts[f"{partition}:{family}"],
                    )
                )
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class CandidateAuditV2:
    candidate: CandidateNameV2
    warm: BankDistributionMetricsV2
    capability: BankDistributionMetricsV2
    warm_surface_leakage: SurfaceLeakageAuditReport
    capability_surface_leakage: SurfaceLeakageAuditReport
    semantic_comparison: SemanticComparisonV2
    identity_capacity: tuple[CandidateCapacityRowV2, ...]
    blocker_codes: tuple[str, ...]

    @property
    def qa_episode_count(self) -> int:
        return self.warm.episode_count + self.capability.episode_count

    @property
    def structural_distributions_passed(self) -> bool:
        return (
            self.warm.structural_invariants_passed
            and self.capability.structural_invariants_passed
        )

    @property
    def identity_capacity_passed(self) -> bool:
        return bool(self.identity_capacity) and all(
            row.sufficient for row in self.identity_capacity
        )

    @property
    def leakage_powered(self) -> bool:
        return all(
            result.group_count >= result.minimum_groups_required
            and result.decision != "insufficient_data"
            for report in (self.warm_surface_leakage, self.capability_surface_leakage)
            for result in report.results
        )

    @property
    def surface_leakage_passed(self) -> bool:
        return (
            self.warm_surface_leakage.passed
            and self.capability_surface_leakage.passed
        )

    @property
    def production_bank_generation_authorized(self) -> bool:
        return False

    @property
    def weight_updates_authorized(self) -> bool:
        return False

    def as_obj(self) -> dict[str, Any]:
        return {
            "schema_version": CAPABILITY_QA_SCHEMA_VERSION_V2,
            "candidate": self.candidate,
            "scope": "locally-materialized-qa-candidate-non-authorizing",
            "stage_episode_counts": {
                "format_warm_start": self.warm.episode_count,
                "capability": self.capability.episode_count,
            },
            "qa_episode_count": self.qa_episode_count,
            "registered_binary_group_floor": REGISTERED_BINARY_GROUP_FLOOR,
            "surface_target_registry": list(CAPABILITY_V2_SURFACE_TARGETS),
            "surface_mask_policy": CAPABILITY_V2_MASK_POLICY,
            "warm": self.warm.as_obj(),
            "capability": self.capability.as_obj(),
            "warm_surface_leakage": self.warm_surface_leakage.as_obj(),
            "capability_surface_leakage": self.capability_surface_leakage.as_obj(),
            "semantic_comparison": self.semantic_comparison.as_obj(),
            "identity_capacity": [row.as_obj() for row in self.identity_capacity],
            "identity_capacity_passed": self.identity_capacity_passed,
            "structural_distributions_passed": self.structural_distributions_passed,
            "leakage_powered": self.leakage_powered,
            "surface_leakage_passed": self.surface_leakage_passed,
            "blocker_codes": list(self.blocker_codes),
            "production_bank_generation_authorized": (
                self.production_bank_generation_authorized
            ),
            "weight_updates_authorized": self.weight_updates_authorized,
        }

    @property
    def digest(self) -> str:
        return json_digest(
            self.as_obj(), domain="goalzendo-interactive-capability-candidate-audit-v2"
        )


def _candidate_specs_v2(
    candidate: CandidateNameV2,
) -> tuple[EpisodeBankSpecV2, EpisodeBankSpecV2]:
    if candidate == "registered_256":
        plan = build_stage_request_plan_v2()
        return plan.warm_start, plan.capability
    if candidate == "symmetric_384":
        return build_symmetric_stage_specs_v2()
    if candidate == "powered_402":
        return build_powered_stage_specs_v2()
    raise ValueError(f"unknown schema-v2 QA candidate: {candidate!r}")


@lru_cache(maxsize=8)
def build_candidate_audit_v2(
    candidate: CandidateNameV2,
    config: SurfaceLeakageConfig | None = None,
) -> CandidateAuditV2:
    """Materialize and audit one candidate; authorization is always false."""

    if config is None:
        config = SurfaceLeakageConfig()
    if type(config) is not SurfaceLeakageConfig:
        raise TypeError("config must be a SurfaceLeakageConfig")
    warm_spec, capability_spec = _candidate_specs_v2(candidate)
    capacity = candidate_identity_capacity_v2(warm_spec, capability_spec)
    warm_bank = generate_episode_bank_v2(warm_spec)
    capability_bank = generate_episode_bank_v2(capability_spec)
    warm_metrics = bank_distribution_metrics_v2(warm_bank)
    capability_metrics = bank_distribution_metrics_v2(capability_bank)
    warm_leakage = audit_hidden_episode_bank_v2_surface_leakage(
        warm_bank, config=config
    )
    capability_leakage = audit_hidden_episode_bank_v2_surface_leakage(
        capability_bank, config=config
    )
    semantic = semantic_comparison_v2(
        warm_bank,
        capability_bank,
        warm_metrics,
        capability_metrics,
    )
    blockers: list[str] = []
    if not all(
        result.group_count >= result.minimum_groups_required
        and result.decision != "insufficient_data"
        for report in (warm_leakage, capability_leakage)
        for result in report.results
    ):
        minimum_observed = min(
            result.group_count
            for report in (warm_leakage, capability_leakage)
            for result in report.results
        )
        blockers.append(
            f"surface_leakage_underpowered:{minimum_observed}"
            f"<{minimum_resolution_groups(2)}"
        )
    elif not warm_leakage.passed or not capability_leakage.passed:
        blockers.append("surface_leakage_gate_failed")
    if not (
        warm_metrics.structural_invariants_passed
        and capability_metrics.structural_invariants_passed
    ):
        blockers.append("structural_distribution_invariant_failed")
    if not all(row.sufficient for row in capacity):
        blockers.append("insufficient_unique_official_law_capacity")
    blockers.extend(
        (
            "semantic_distribution_acceptance_rule_not_preregistered",
            "model_pipeline_integration_not_audited",
            "no_weight_update_authorization",
        )
    )
    return CandidateAuditV2(
        candidate=candidate,
        warm=warm_metrics,
        capability=capability_metrics,
        warm_surface_leakage=warm_leakage,
        capability_surface_leakage=capability_leakage,
        semantic_comparison=semantic,
        identity_capacity=capacity,
        blocker_codes=tuple(blockers),
    )
