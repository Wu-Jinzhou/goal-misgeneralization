"""Prospective schema-v3 construction for the G03 capability slice.

This module is strictly additive.  It leaves the schema-v2 request and episode
bytes untouched, and it keeps every production and weight-update authorization
false.  The v3 construction fixes its request folds, renderer assignment,
ordinal cell schedule, and semantic acceptance rule before selecting a single
Official-Law identity.
"""

from __future__ import annotations

import hashlib
import heapq
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal, cast

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp  # type: ignore[import-untyped]
from scipy.sparse import lil_matrix  # type: ignore[import-untyped]

from ._json import json_digest
from .capability import ProxyOrientationV2, ProxyProfileV2, StageV2
from .catalog import CatalogEntry, build_rule_catalog
from .episodes import (
    MAX_TARGET_SHADOW_DISAGREEMENT,
    MIN_TARGET_SHADOW_CELL_COUNT,
    MIN_TARGET_SHADOW_DISAGREEMENT,
    Observation,
)
from .rendering import EVAL_RENDERERS, TRAIN_RENDERERS, RendererName
from .rules import ATOM_OPS, BinaryOp, BinaryRule
from .rules import Literal as RuleLiteral
from .schema import NONEMPTY_ARRANGEMENT_COUNT, SCENE_COUNT
from .stage_partitions_v2 import (
    EligibleTargetShadowPairV2,
    RulePartitionV2,
    build_eligible_target_shadow_table_v2,
    build_rule_identity_partitions_v2,
    catalog_rule_family_v2,
    target_shadow_cells_v2,
)

CAPABILITY_SCHEMA_VERSION_V3 = 3
V3_FOLD_COUNT = 5
V3_WARM_EPISODES = 400
V3_CAPABILITY_EPISODES = 420
V3_BINARY_EPISODES_PER_STAGE = 400
V3_SEMANTIC_RESAMPLES = 10_000
V3_SEMANTIC_FAMILYWISE_ALPHA_NUMERATOR = 5
V3_SEMANTIC_FAMILYWISE_ALPHA_DENOMINATOR = 100

TargetFamilyV3 = Literal[
    "binary_piece",
    "literal_piece",
    "placard_literal",
    "binary_mixed_control",
]
PlacardPolarityV3 = Literal["agrees_with_sun", "disagrees_with_sun"]

TARGET_FAMILIES_V3: tuple[TargetFamilyV3, ...] = (
    "binary_piece",
    "literal_piece",
    "placard_literal",
    "binary_mixed_control",
)
SEMANTIC_METRICS_V3: tuple[str, ...] = (
    "target_true_count_total",
    "canonical_rule_json_length_total",
    *(f"atom_presence_total:{op}" for op in ATOM_OPS),
    "target_shadow_cells_total:y0q0",
    "target_shadow_cells_total:y0q1",
    "target_shadow_cells_total:y1q0",
    "target_shadow_cells_total:y1q1",
)

_REQUEST_PAIR_ORDER_DOMAIN_V3 = b"goalzendo-interactive-request-pair-order-v3\0"
_PAIR_IDENTITY_DOMAIN_V3 = b"goalzendo-interactive-eligible-pair-v3\0"
_SCENE_SELECTION_DOMAIN_V3 = b"goalzendo-interactive-scene-selection-v3\0"
_FOLD_ORDER_DOMAIN = b"goalzendo-surface-leakage-group-order-v1\0"
_SEMANTIC_SEED_DOMAIN_V3 = b"goalzendo-interactive-semantic-resampling-v3\0"


class CapabilityV3Error(ValueError):
    """Raised when a prospective v3 construction invariant is violated."""


def _hash_parts(domain: bytes, *parts: str) -> bytes:
    digest = hashlib.sha256(domain)
    for part in parts:
        digest.update(part.encode("ascii"))
        digest.update(b"\0")
    return digest.digest()


def _stable_group_key(target_name: str, group_id: str) -> bytes:
    """Match the registered leakage fold-order hash byte-for-byte."""

    return hashlib.sha256(
        _FOLD_ORDER_DOMAIN
        + target_name.encode("ascii")
        + b"\0"
        + group_id.encode("ascii")
    ).digest()


def _is_sun(scene_index: int) -> bool:
    return bool(scene_index < NONEMPTY_ARRANGEMENT_COUNT)


def _cell_number(target: CatalogEntry, shadow: CatalogEntry, scene_index: int) -> int:
    return (
        int(target.truth[scene_index]) * 4
        + int(_is_sun(scene_index)) * 2
        + int(shadow.truth[scene_index])
    )


def _outcome_bits(cell: int) -> tuple[int, int, int]:
    y = (cell >> 2) & 1
    p = (cell >> 1) & 1
    q = cell & 1
    return y, y ^ p, y ^ q


def _placard_agrees(target: CatalogEntry) -> bool:
    if catalog_rule_family_v2(target) != "placard_literal":
        raise CapabilityV3Error("placard polarity requires a placard literal")
    return not cast(RuleLiteral, target.rule).negated


def _family_v3(entry: CatalogEntry) -> TargetFamilyV3:
    family = catalog_rule_family_v2(entry)
    if family == "binary_mixed":
        return "binary_mixed_control"
    return cast(TargetFamilyV3, family)


@dataclass(frozen=True, slots=True)
class EpisodeRequestV3:
    """One prospectively folded schema-v3 request."""

    request_id: str
    stage: StageV2
    partition: RulePartitionV2
    target_family: TargetFamilyV3
    target_op: BinaryOp | None
    proxy_profile: ProxyProfileV2
    proxy_orientation: ProxyOrientationV2
    renderer: RendererName
    placard_polarity: PlacardPolarityV3 | None
    registered_formula_fold: int | None
    registered_proxy_fold: int

    def __post_init__(self) -> None:
        if type(self.request_id) is not str or not self.request_id or not self.request_id.isascii():
            raise CapabilityV3Error("v3 request_id must be nonempty ASCII")
        if self.target_family not in TARGET_FAMILIES_V3:
            raise CapabilityV3Error("unknown v3 target family")
        if self.target_family == "binary_piece":
            if self.target_op not in ("all", "any", "exactly_one"):
                raise CapabilityV3Error("binary-piece v3 requests require target_op")
            if self.registered_formula_fold is None:
                raise CapabilityV3Error("binary-piece v3 requests require a formula fold")
        elif self.target_op is not None or self.registered_formula_fold is not None:
            raise CapabilityV3Error("control requests are outside the formula estimand")
        if self.target_family == "placard_literal":
            if self.placard_polarity not in (
                "agrees_with_sun",
                "disagrees_with_sun",
            ):
                raise CapabilityV3Error("placard controls require a frozen polarity")
        elif self.placard_polarity is not None:
            raise CapabilityV3Error("placard polarity is exclusive to placard controls")
        for value in (self.registered_proxy_fold,):
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < V3_FOLD_COUNT:
                raise CapabilityV3Error("registered proxy fold lies outside the v3 registry")
        if self.registered_formula_fold is not None and not (
            0 <= self.registered_formula_fold < V3_FOLD_COUNT
        ):
            raise CapabilityV3Error("registered formula fold lies outside the v3 registry")
        if self.stage == "format_warm_start":
            if (
                self.partition != "warm_start"
                or self.target_family != "binary_piece"
                or self.proxy_profile != "chance_balanced"
                or self.renderer not in TRAIN_RENDERERS
            ):
                raise CapabilityV3Error("invalid v3 warm-start request")
        elif self.stage == "capability":
            if (
                self.partition != "capability"
                or self.proxy_profile != "oracle_diagnostic"
                or self.renderer not in EVAL_RENDERERS
            ):
                raise CapabilityV3Error("invalid v3 capability request")
        else:
            raise CapabilityV3Error("unknown v3 stage")

    @property
    def formula_label(self) -> str | None:
        if self.target_family != "binary_piece":
            return None
        return "exactly_one" if self.target_op == "exactly_one" else "all_or_any"

    def as_obj(self) -> dict[str, Any]:
        return {
            "schema_version": CAPABILITY_SCHEMA_VERSION_V3,
            "request_id": self.request_id,
            "stage": self.stage,
            "partition": self.partition,
            "target_family": self.target_family,
            "target_op": self.target_op,
            "proxy_profile": self.proxy_profile,
            "proxy_orientation": self.proxy_orientation,
            "renderer": self.renderer,
            "placard_polarity": self.placard_polarity,
            "registered_formula_fold": self.registered_formula_fold,
            "registered_proxy_fold": self.registered_proxy_fold,
        }

    @property
    def digest(self) -> str:
        return json_digest(self.as_obj(), domain="goalzendo-interactive-episode-request-v3")


def expected_cell_counts_v3(request: EpisodeRequestV3) -> tuple[int, ...]:
    """Return the prospective ten-scene ``Y,P,Q`` profile for a request."""

    high = request.proxy_orientation == "proxy_high"
    if request.target_family != "placard_literal":
        counts = [1] * 8
        for cell in ((0b011, 0b111) if high else (0b000, 0b100)):
            counts[cell] += 1
        return tuple(counts)
    low_q0, low_q1 = ((2, 3) if high else (3, 2))
    counts = [0] * 8
    if request.placard_polarity == "agrees_with_sun":
        counts[0b000], counts[0b001] = low_q0, low_q1
        counts[0b110], counts[0b111] = low_q0, low_q1
    else:
        counts[0b010], counts[0b011] = low_q0, low_q1
        counts[0b100], counts[0b101] = low_q0, low_q1
    return tuple(counts)


def _registered_fold_assignments(
    target_name: str,
    classes: tuple[str, ...],
    group_vectors: tuple[tuple[int, ...], ...],
) -> tuple[int, ...]:
    """Reproduce the frozen grouped-fold allocator on prospective group vectors."""

    if any(len(vector) != len(classes) or sum(vector) <= 0 for vector in group_vectors):
        raise CapabilityV3Error("invalid prospective fold vector")
    buckets: dict[int, list[int]] = defaultdict(list)
    for index, vector in enumerate(group_vectors):
        primary = max(range(len(classes)), key=lambda item: (vector[item], -item))
        buckets[primary].append(index)
    fold_class_counts = [[0] * len(classes) for _ in range(V3_FOLD_COUNT)]
    fold_sample_counts = [0] * V3_FOLD_COUNT
    fold_group_counts = [0] * V3_FOLD_COUNT
    assignments = [-1] * len(group_vectors)
    for primary in range(len(classes)):
        ordered = sorted(
            buckets[primary],
            key=lambda index: (
                -sum(group_vectors[index]),
                tuple(-count for count in group_vectors[index]),
                _stable_group_key(target_name, f"episode-{index}"),
            ),
        )
        for index in ordered:
            vector = group_vectors[index]
            fold = min(
                range(V3_FOLD_COUNT),
                key=lambda candidate: (
                    fold_class_counts[candidate][primary],
                    fold_sample_counts[candidate],
                    fold_group_counts[candidate],
                    candidate,
                ),
            )
            assignments[index] = fold
            fold_group_counts[fold] += 1
            fold_sample_counts[fold] += sum(vector)
            for class_index, count in enumerate(vector):
                fold_class_counts[fold][class_index] += count
    if any(value < 0 for value in assignments):
        raise CapabilityV3Error("prospective fold assignment dropped a group")
    return tuple(assignments)


def registered_observation_folds_v3(episode_count: int, target_name: str) -> tuple[int, ...]:
    if target_name not in {
        "target_label",
        "placard_error_status",
        "shadow_error_status",
    }:
        raise CapabilityV3Error("unknown v3 observation target")
    return _registered_fold_assignments(
        target_name,
        ("false", "true"),
        tuple((10, 10) for _ in range(episode_count)),
    )


@dataclass(frozen=True, slots=True)
class _RequestSlotV3:
    target_family: TargetFamilyV3
    target_op: BinaryOp | None
    orientation: ProxyOrientationV2
    placard_polarity: PlacardPolarityV3 | None = None

    @property
    def formula_label(self) -> str | None:
        if self.target_family != "binary_piece":
            return None
        return "exactly_one" if self.target_op == "exactly_one" else "all_or_any"


def _euler_halves(
    edge_ids: tuple[int, ...],
    endpoints: tuple[tuple[str, str], ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    adjacency: dict[str, list[int]] = defaultdict(list)
    for edge in edge_ids:
        left, right = endpoints[edge]
        adjacency[left].append(edge)
        adjacency[right].append(edge)
    if any(len(items) % 2 for items in adjacency.values()):
        raise CapabilityV3Error("renderer-balance graph has an odd-degree cell")
    for items in adjacency.values():
        items.sort(reverse=True)
    unused = set(edge_ids)
    halves: tuple[list[int], list[int]] = ([], [])
    while unused:
        first_edge = min(unused)
        start = endpoints[first_edge][0]
        node_stack = [start]
        edge_stack: list[int] = []
        circuit: list[int] = []
        while node_stack:
            node = node_stack[-1]
            while adjacency[node] and adjacency[node][-1] not in unused:
                adjacency[node].pop()
            if adjacency[node]:
                edge = adjacency[node].pop()
                if edge not in unused:
                    continue
                unused.remove(edge)
                left, right = endpoints[edge]
                node_stack.append(right if node == left else left)
                edge_stack.append(edge)
            else:
                node_stack.pop()
                if edge_stack:
                    circuit.append(edge_stack.pop())
        circuit.reverse()
        if len(circuit) % 2:
            raise CapabilityV3Error("bipartite renderer circuit must have even length")
        for position, edge in enumerate(circuit):
            halves[position % 2].append(edge)
    return tuple(sorted(halves[0])), tuple(sorted(halves[1]))


def _balanced_renderer_colors(
    slots: tuple[_RequestSlotV3, ...],
    formula_folds: tuple[int | None, ...],
    proxy_folds: tuple[int, ...],
    color_count: int,
) -> tuple[int, ...]:
    endpoints = tuple(
        (
            (
                f"formula:{formula_folds[index]}:{slot.formula_label}"
                if slot.formula_label is not None
                else "formula:controls"
            ),
            f"proxy:{proxy_folds[index]}:{slot.orientation}",
        )
        for index, slot in enumerate(slots)
    )
    partitions: tuple[tuple[int, ...], ...] = (tuple(range(len(slots))),)
    while len(partitions) < color_count:
        partitions = tuple(
            half
            for group in partitions
            for half in _euler_halves(group, endpoints)
        )
    if len(partitions) != color_count:
        raise CapabilityV3Error("renderer count must be a power of two")
    colors = [-1] * len(slots)
    for color, group in enumerate(partitions):
        for index in group:
            colors[index] = color
    if any(value < 0 for value in colors):
        raise CapabilityV3Error("renderer assignment dropped a request")
    return tuple(colors)


def _slot_folds(
    slots: tuple[_RequestSlotV3, ...],
) -> tuple[tuple[int | None, ...], tuple[int, ...]]:
    binary_indices = tuple(
        index for index, slot in enumerate(slots) if slot.formula_label is not None
    )
    formula_labels = tuple(cast(str, slots[index].formula_label) for index in binary_indices)
    formula_local = _registered_fold_assignments(
        "target_formula_stratum",
        ("all_or_any", "exactly_one"),
        tuple(
            (1, 0) if label == "all_or_any" else (0, 1)
            for label in formula_labels
        ),
    )
    # The leakage adapter uses global episode indices as group ids.  Re-run the
    # allocator with those exact ids rather than assuming the binary rows are a
    # contiguous local table.
    buckets: dict[str, list[int]] = defaultdict(list)
    for index in binary_indices:
        buckets[cast(str, slots[index].formula_label)].append(index)
    class_counts = [[0, 0] for _ in range(V3_FOLD_COUNT)]
    sample_counts = [0] * V3_FOLD_COUNT
    group_counts = [0] * V3_FOLD_COUNT
    global_formula: list[int | None] = [None] * len(slots)
    for class_index, label in enumerate(("all_or_any", "exactly_one")):
        ordered = sorted(
            buckets[label],
            key=lambda index: _stable_group_key(
                "target_formula_stratum", f"episode-{index}"
            ),
        )
        for index in ordered:
            fold = min(
                range(V3_FOLD_COUNT),
                key=lambda candidate: (
                    class_counts[candidate][class_index],
                    sample_counts[candidate],
                    group_counts[candidate],
                    candidate,
                ),
            )
            global_formula[index] = fold
            class_counts[fold][class_index] += 1
            sample_counts[fold] += 1
            group_counts[fold] += 1
    del formula_local
    proxy = _registered_fold_assignments(
        "proxy_orientation",
        ("proxy_low", "proxy_high"),
        tuple(
            (1, 0) if slot.orientation == "proxy_low" else (0, 1)
            for slot in slots
        ),
    )
    return tuple(global_formula), proxy


def _build_requests(
    stage: StageV2,
    slots: tuple[_RequestSlotV3, ...],
) -> tuple[EpisodeRequestV3, ...]:
    formula_folds, proxy_folds = _slot_folds(slots)
    renderers = TRAIN_RENDERERS if stage == "format_warm_start" else EVAL_RENDERERS
    colors = _balanced_renderer_colors(
        slots,
        formula_folds,
        proxy_folds,
        len(renderers),
    )
    prefix = "warm" if stage == "format_warm_start" else "capability"
    partition: RulePartitionV2 = "warm_start" if stage == "format_warm_start" else "capability"
    profile: ProxyProfileV2 = "chance_balanced" if stage == "format_warm_start" else "oracle_diagnostic"
    requests = tuple(
        EpisodeRequestV3(
            request_id=f"g03-v3-{prefix}-{index:04d}",
            stage=stage,
            partition=partition,
            target_family=slot.target_family,
            target_op=slot.target_op,
            proxy_profile=profile,
            proxy_orientation=slot.orientation,
            renderer=renderers[colors[index]],
            placard_polarity=slot.placard_polarity,
            registered_formula_fold=formula_folds[index],
            registered_proxy_fold=proxy_folds[index],
        )
        for index, slot in enumerate(slots)
    )
    _validate_fold_renderer_balance(requests)
    return requests


def _validate_fold_renderer_balance(requests: tuple[EpisodeRequestV3, ...]) -> None:
    renderers = TRAIN_RENDERERS if requests[0].stage == "format_warm_start" else EVAL_RENDERERS
    for fold in range(V3_FOLD_COUNT):
        for label in ("all_or_any", "exactly_one"):
            counts = Counter(
                request.renderer
                for request in requests
                if request.registered_formula_fold == fold and request.formula_label == label
            )
            values = tuple(counts[renderer] for renderer in renderers)
            if not values or len(set(values)) != 1 or values[0] == 0:
                raise CapabilityV3Error("renderer x formula balance failed inside a fold")
        for orientation in ("proxy_low", "proxy_high"):
            counts = Counter(
                request.renderer
                for request in requests
                if request.registered_proxy_fold == fold
                and request.proxy_orientation == orientation
            )
            values = tuple(counts[renderer] for renderer in renderers)
            if not values or len(set(values)) != 1 or values[0] == 0:
                raise CapabilityV3Error("renderer x proxy balance failed inside a fold")


def _warm_slots() -> tuple[_RequestSlotV3, ...]:
    operators = cast(
        tuple[BinaryOp, ...],
        ("all",) * 100 + ("any",) * 100 + ("exactly_one",) * 200,
    )
    return tuple(
        _RequestSlotV3(
            "binary_piece",
            op,
            "proxy_low" if index < 200 else "proxy_high",
        )
        for index, op in enumerate(operators)
    )


def _capability_slots() -> tuple[_RequestSlotV3, ...]:
    base: list[tuple[TargetFamilyV3, BinaryOp | None, PlacardPolarityV3 | None]] = []
    base.extend(("binary_piece", "all", None) for _ in range(100))
    base.extend(("binary_piece", "any", None) for _ in range(100))
    base.extend(("binary_piece", "exactly_one", None) for _ in range(200))
    base.extend(("literal_piece", None, None) for _ in range(16))
    base.extend(("binary_mixed_control", None, None) for _ in range(2))
    base.extend(
        (
            ("placard_literal", None, "agrees_with_sun"),
            ("placard_literal", None, "disagrees_with_sun"),
        )
    )
    nonplacard_low_remaining = 208
    slots: list[_RequestSlotV3] = []
    for family, op, polarity in base:
        if family == "placard_literal":
            orientation: ProxyOrientationV2 = "proxy_low"
        elif nonplacard_low_remaining:
            orientation = "proxy_low"
            nonplacard_low_remaining -= 1
        else:
            orientation = "proxy_high"
        slots.append(_RequestSlotV3(family, op, orientation, polarity))
    if nonplacard_low_remaining:
        raise CapabilityV3Error("capability orientation quota was not exhausted")
    return tuple(slots)


def _counterbalanced_cell_schedule(
    requests: tuple[EpisodeRequestV3, ...],
) -> tuple[tuple[int, ...], ...]:
    """Solve the frozen zero-objective exact feasibility system once."""

    profiles = tuple(expected_cell_counts_v3(request) for request in requests)
    target_names = (
        "target_label",
        "placard_error_status",
        "shadow_error_status",
    )
    observation_folds = tuple(
        registered_observation_folds_v3(len(requests), target_name)
        for target_name in target_names
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
                    [(variable_index[(episode, cell, position)], 1) for position in range(10)]
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
    for target_index, folds in enumerate(observation_folds):
        for fold in range(V3_FOLD_COUNT):
            episodes = tuple(index for index, value in enumerate(folds) if value == fold)
            if len(episodes) % 2:
                raise CapabilityV3Error("observation fold has odd episode count")
            for position in range(10):
                rows.append(
                    [
                        (variable_index[(episode, cell, position)], 1)
                        for episode in episodes
                        for cell, count in enumerate(profiles[episode])
                        if count and _outcome_bits(cell)[target_index]
                    ]
                )
                lower.append(len(episodes) // 2)
                upper.append(len(episodes) // 2)
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
        raise CapabilityV3Error(
            "exact phase/ordinal counterbalance is mathematically infeasible: "
            f"{result.message}"
        )
    schedule: list[list[int]] = [[-1] * 10 for _ in requests]
    for (episode, cell, position), value in zip(variables, result.x, strict=True):
        if value > 0.5:
            if schedule[episode][position] != -1:
                raise CapabilityV3Error("counterbalance solver assigned two cells to a slot")
            schedule[episode][position] = cell
    frozen = tuple(tuple(row) for row in schedule)
    _validate_counterbalance_schedule(requests, frozen)
    return frozen


def _validate_counterbalance_schedule(
    requests: tuple[EpisodeRequestV3, ...],
    schedule: tuple[tuple[int, ...], ...],
) -> None:
    if len(schedule) != len(requests) or any(len(row) != 10 for row in schedule):
        raise CapabilityV3Error("counterbalance schedule shape mismatch")
    for request, row in zip(requests, schedule, strict=True):
        if Counter(row) != Counter(
            {
                cell: count
                for cell, count in enumerate(expected_cell_counts_v3(request))
                if count
            }
        ):
            raise CapabilityV3Error("counterbalance row does not match its exact Y,P,Q profile")
    target_names = (
        "target_label",
        "placard_error_status",
        "shadow_error_status",
    )
    for target_index, target_name in enumerate(target_names):
        folds = registered_observation_folds_v3(len(requests), target_name)
        for fold in range(V3_FOLD_COUNT):
            episodes = tuple(index for index, value in enumerate(folds) if value == fold)
            for position in range(10):
                true_count = sum(
                    _outcome_bits(schedule[episode][position])[target_index]
                    for episode in episodes
                )
                if true_count * 2 != len(episodes):
                    raise CapabilityV3Error(
                        "phase/ordinal outcome balance failed inside a registered fold"
                    )


@dataclass(frozen=True, slots=True)
class SemanticGateDerivationV3:
    metric_names: tuple[str, ...]
    resample_count: int
    familywise_alpha_numerator: int
    familywise_alpha_denominator: int
    per_metric_exceedance_budget: int
    upper_thresholds: tuple[int, ...]
    population_target_counts: tuple[tuple[str, int], ...]
    population_pair_counts: tuple[tuple[str, int], ...]
    seed_digest: str

    def as_obj(self) -> dict[str, Any]:
        return {
            "rule": (
                "accept iff every absolute warm-minus-capability binary-piece aggregate "
                "is at most its exhaustive-population resampling threshold"
            ),
            "binary_quota_per_stage": {
                "all": 100,
                "any": 100,
                "exactly_one": 200,
            },
            "controls_excluded_without_relabeling": True,
            "metric_names": list(self.metric_names),
            "resample_count": self.resample_count,
            "familywise_alpha": {
                "numerator": self.familywise_alpha_numerator,
                "denominator": self.familywise_alpha_denominator,
            },
            "bonferroni_per_metric_exceedance_budget": self.per_metric_exceedance_budget,
            "upper_thresholds": dict(zip(self.metric_names, self.upper_thresholds, strict=True)),
            "population_target_counts": dict(self.population_target_counts),
            "population_pair_counts": dict(self.population_pair_counts),
            "resampling_unit": (
                "unique target identity without replacement within partition/operator; "
                "one eligible shadow sampled uniformly for each target"
            ),
            "seed_digest": self.seed_digest,
            "derived_before_candidate_materialization": True,
        }

    @property
    def digest(self) -> str:
        return json_digest(
            self.as_obj(),
            domain="goalzendo-interactive-semantic-gate-derivation-v3",
        )


def _binary_population(
    partition: Literal["warm_start", "capability"],
    op: BinaryOp,
) -> tuple[np.ndarray[Any, np.dtype[np.int64]], tuple[np.ndarray[Any, np.dtype[np.int64]], ...]]:
    catalog = build_rule_catalog()
    pairs = build_eligible_target_shadow_table_v2().candidates(partition, "binary_piece")
    grouped: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)
    for pair in pairs:
        target = catalog[pair.target_index]
        if cast(BinaryRule, target.rule).op == op:
            grouped[pair.target_index].append(pair.cells)
    target_indices = tuple(sorted(grouped))
    base_rows: list[list[int]] = []
    pair_rows: list[np.ndarray[Any, np.dtype[np.int64]]] = []
    for target_index in target_indices:
        target = catalog[target_index]
        rule = cast(BinaryRule, target.rule)
        present = {literal.atom.op for literal in rule.args}
        base_rows.append(
            [
                target.truth.true_count,
                len(rule.canonical_json),
                *(int(atom_op in present) for atom_op in ATOM_OPS),
            ]
        )
        pair_rows.append(np.asarray(sorted(grouped[target_index]), dtype=np.int64))
    return np.asarray(base_rows, dtype=np.int64), tuple(pair_rows)


@lru_cache(maxsize=1)
def build_semantic_gate_derivation_v3() -> SemanticGateDerivationV3:
    """Derive the semantic envelope without observing a v3 candidate bank."""

    quotas: tuple[tuple[BinaryOp, int], ...] = (
        ("all", 100),
        ("any", 100),
        ("exactly_one", 200),
    )
    populations: dict[
        tuple[str, BinaryOp],
        tuple[np.ndarray[Any, np.dtype[np.int64]], tuple[np.ndarray[Any, np.dtype[np.int64]], ...]],
    ] = {}
    target_counts: list[tuple[str, int]] = []
    pair_counts: list[tuple[str, int]] = []
    for partition in ("warm_start", "capability"):
        for op, quota in quotas:
            population = _binary_population(cast(Any, partition), op)
            if len(population[0]) < quota:
                raise CapabilityV3Error("semantic resampling quota exceeds eligible population")
            populations[(partition, op)] = population
            key = f"{partition}:{op}"
            target_counts.append((key, len(population[0])))
            pair_counts.append((key, sum(len(rows) for rows in population[1])))
    seed_digest = hashlib.sha256(
        _SEMANTIC_SEED_DOMAIN_V3
        + build_rule_catalog().digest.encode("ascii")
        + b"\0"
        + build_eligible_target_shadow_table_v2().digest.encode("ascii")
    ).hexdigest()
    generator = np.random.Generator(
        np.random.PCG64(int.from_bytes(bytes.fromhex(seed_digest)[:16], "big"))
    )
    differences = np.zeros(
        (V3_SEMANTIC_RESAMPLES, len(SEMANTIC_METRICS_V3)), dtype=np.int64
    )
    for replicate in range(V3_SEMANTIC_RESAMPLES):
        totals: list[np.ndarray[Any, np.dtype[np.int64]]] = []
        for partition in ("warm_start", "capability"):
            stage_total = np.zeros(len(SEMANTIC_METRICS_V3), dtype=np.int64)
            for op, quota in quotas:
                base, pair_cells = populations[(partition, op)]
                selected = np.asarray(
                    generator.choice(len(base), size=quota, replace=False),
                    dtype=np.int64,
                )
                stage_total[: 2 + len(ATOM_OPS)] += base[selected].sum(axis=0)
                pair_draws = np.floor(
                    generator.random(quota)
                    * np.asarray([len(pair_cells[index]) for index in selected])
                ).astype(np.int64)
                stage_total[2 + len(ATOM_OPS) :] += np.asarray(
                    [
                        pair_cells[target_index][pair_index]
                        for target_index, pair_index in zip(
                            selected, pair_draws, strict=True
                        )
                    ],
                    dtype=np.int64,
                ).sum(axis=0)
            totals.append(stage_total)
        differences[replicate] = np.abs(totals[0] - totals[1])
    exceedance_budget = (
        V3_SEMANTIC_RESAMPLES
        * V3_SEMANTIC_FAMILYWISE_ALPHA_NUMERATOR
        // V3_SEMANTIC_FAMILYWISE_ALPHA_DENOMINATOR
        // len(SEMANTIC_METRICS_V3)
    )
    threshold_index = V3_SEMANTIC_RESAMPLES - exceedance_budget - 1
    thresholds = tuple(
        int(value)
        for value in np.sort(differences, axis=0)[threshold_index].tolist()
    )
    return SemanticGateDerivationV3(
        SEMANTIC_METRICS_V3,
        V3_SEMANTIC_RESAMPLES,
        V3_SEMANTIC_FAMILYWISE_ALPHA_NUMERATOR,
        V3_SEMANTIC_FAMILYWISE_ALPHA_DENOMINATOR,
        exceedance_budget,
        thresholds,
        tuple(target_counts),
        tuple(pair_counts),
        seed_digest,
    )


@dataclass(frozen=True, slots=True)
class StageSpecV3:
    bank_id: str
    requests: tuple[EpisodeRequestV3, ...]
    cell_schedule: tuple[tuple[int, ...], ...]
    max_pair_attempts: int = 4_096

    def __post_init__(self) -> None:
        requests = tuple(self.requests)
        schedule = tuple(tuple(row) for row in self.cell_schedule)
        object.__setattr__(self, "requests", requests)
        object.__setattr__(self, "cell_schedule", schedule)
        if not requests or len({request.request_id for request in requests}) != len(requests):
            raise CapabilityV3Error("v3 stage requires unique requests")
        if len({request.stage for request in requests}) != 1:
            raise CapabilityV3Error("v3 stage spec cannot mix stages")
        expected = V3_WARM_EPISODES if requests[0].stage == "format_warm_start" else V3_CAPABILITY_EPISODES
        if len(requests) != expected:
            raise CapabilityV3Error("v3 stage has the wrong powered episode count")
        _validate_fold_renderer_balance(requests)
        _validate_counterbalance_schedule(requests, schedule)
        if not 1 <= self.max_pair_attempts <= 100_000:
            raise CapabilityV3Error("v3 max_pair_attempts lies outside its bound")

    def as_obj(self) -> dict[str, Any]:
        return {
            "schema_version": CAPABILITY_SCHEMA_VERSION_V3,
            "bank_id": self.bank_id,
            "requests": [request.as_obj() for request in self.requests],
            "cell_schedule": [list(row) for row in self.cell_schedule],
            "max_pair_attempts": self.max_pair_attempts,
        }

    @property
    def digest(self) -> str:
        return json_digest(self.as_obj(), domain="goalzendo-interactive-stage-spec-v3")


@dataclass(frozen=True, slots=True)
class ProspectiveDesignV3:
    warm: StageSpecV3
    capability: StageSpecV3
    semantic_gate: SemanticGateDerivationV3

    def as_obj(self) -> dict[str, Any]:
        return {
            "schema_version": CAPABILITY_SCHEMA_VERSION_V3,
            "design_id": "g03-v3-prospective-counterbalanced-powered",
            "status": "frozen_before_materialization_non_authorizing",
            "registered_fold_count": V3_FOLD_COUNT,
            "formula_estimand": "binary_piece:all_or_any_vs_exactly_one",
            "formula_controls_relabelled": False,
            "request_assignment_rule": (
                "frozen v1 grouped folds followed by deterministic bipartite Euler "
                "renderer balancing inside every formula and proxy fold cell"
            ),
            "scene_counterbalance_rule": (
                "one frozen zero-objective integer-feasibility schedule; exact target, "
                "placard-error, and shadow-error balance at every ordinal position "
                "inside each target's registered fold; reused for both phases"
            ),
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
        return json_digest(self.as_obj(), domain="goalzendo-interactive-prospective-design-v3")


@lru_cache(maxsize=1)
def build_prospective_design_v3() -> ProspectiveDesignV3:
    """Freeze every request, fold, position, and gate before materialization."""

    warm_requests = _build_requests("format_warm_start", _warm_slots())
    capability_requests = _build_requests("capability", _capability_slots())
    warm = StageSpecV3(
        "g03-v3-format-warm-start-400",
        warm_requests,
        _counterbalanced_cell_schedule(warm_requests),
    )
    capability = StageSpecV3(
        "g03-v3-capability-420",
        capability_requests,
        _counterbalanced_cell_schedule(capability_requests),
    )
    return ProspectiveDesignV3(warm, capability, build_semantic_gate_derivation_v3())


@dataclass(frozen=True, slots=True)
class _PairCandidateV3:
    target_index: int
    shadow_index: int
    cells: tuple[int, int, int, int]
    order_digest: str


def _wrap_v2_pair(pair: EligibleTargetShadowPairV2) -> _PairCandidateV3:
    return _PairCandidateV3(
        pair.target_index,
        pair.shadow_index,
        pair.cells,
        hashlib.sha256(
            _PAIR_IDENTITY_DOMAIN_V3 + pair.order_digest.encode("ascii")
        ).hexdigest(),
    )


@lru_cache(maxsize=2)
def _mixed_pairs_v3(partition: RulePartitionV2) -> tuple[_PairCandidateV3, ...]:
    catalog = build_rule_catalog()
    partitions = build_rule_identity_partitions_v2()
    targets = tuple(
        entry
        for entry in catalog
        if partitions.for_entry(entry) == partition
        and catalog_rule_family_v2(entry) == "binary_mixed"
    )
    shadows = tuple(
        entry
        for entry in catalog
        if partitions.for_entry(entry) == partition
        and catalog_rule_family_v2(entry) == "literal_piece"
    )
    minimum = math.ceil(MIN_TARGET_SHADOW_DISAGREEMENT * SCENE_COUNT)
    maximum = math.floor(MAX_TARGET_SHADOW_DISAGREEMENT * SCENE_COUNT)
    pairs: list[_PairCandidateV3] = []
    for target in targets:
        for shadow in shadows:
            cells = target_shadow_cells_v2(target, shadow)
            disagreement = cells[1] + cells[2]
            if min(cells) < MIN_TARGET_SHADOW_CELL_COUNT or not minimum <= disagreement <= maximum:
                continue
            order_digest = hashlib.sha256(
                _PAIR_IDENTITY_DOMAIN_V3
                + target.truth_digest.encode("ascii")
                + b"\0"
                + shadow.truth_digest.encode("ascii")
            ).hexdigest()
            pairs.append(
                _PairCandidateV3(target.index, shadow.index, cells, order_digest)
            )
    return tuple(sorted(pairs, key=lambda pair: pair.order_digest))


def _request_pairs_v3(request: EpisodeRequestV3) -> tuple[_PairCandidateV3, ...]:
    catalog = build_rule_catalog()
    if request.target_family == "binary_mixed_control":
        candidates = _mixed_pairs_v3(request.partition)
    else:
        candidates = tuple(
            _wrap_v2_pair(pair)
            for pair in build_eligible_target_shadow_table_v2().candidates(
                request.partition,
                cast(Any, request.target_family),
            )
        )
    filtered = tuple(
        pair
        for pair in candidates
        if (
            request.target_op is None
            or cast(BinaryRule, catalog[pair.target_index].rule).op == request.target_op
        )
        and (
            request.placard_polarity is None
            or _placard_agrees(catalog[pair.target_index])
            is (request.placard_polarity == "agrees_with_sun")
        )
    )
    return tuple(
        sorted(
            filtered,
            key=lambda pair: _hash_parts(
                _REQUEST_PAIR_ORDER_DOMAIN_V3,
                request.request_id,
                pair.order_digest,
            )
            + pair.order_digest.encode("ascii"),
        )
    )


@dataclass(frozen=True, slots=True)
class HiddenEpisodeV3:
    episode_id: str
    request: EpisodeRequestV3
    target: CatalogEntry
    shadow: CatalogEntry
    opening: tuple[Observation, ...]
    terminal: tuple[Observation, ...]

    def __post_init__(self) -> None:
        catalog = build_rule_catalog()
        partitions = build_rule_identity_partitions_v2()
        if not 0 <= self.target.index < len(catalog) or catalog[self.target.index] != self.target:
            raise CapabilityV3Error("v3 target is not canonical")
        if not 0 <= self.shadow.index < len(catalog) or catalog[self.shadow.index] != self.shadow:
            raise CapabilityV3Error("v3 shadow is not canonical")
        if partitions.for_entry(self.target) != self.request.partition:
            raise CapabilityV3Error("v3 target is outside its request partition")
        if partitions.for_entry(self.shadow) != self.request.partition:
            raise CapabilityV3Error("v3 shadow is outside its request partition")
        if _family_v3(self.target) != self.request.target_family:
            raise CapabilityV3Error("v3 target family mismatch")
        if catalog_rule_family_v2(self.shadow) != "literal_piece":
            raise CapabilityV3Error("v3 shadow must remain a one-literal piece law")
        if self.request.target_op is not None and (
            cast(BinaryRule, self.target.rule).op != self.request.target_op
        ):
            raise CapabilityV3Error("v3 target operator mismatch")
        if self.request.placard_polarity is not None and (
            _placard_agrees(self.target)
            is not (self.request.placard_polarity == "agrees_with_sun")
        ):
            raise CapabilityV3Error("v3 placard polarity mismatch")
        cells = target_shadow_cells_v2(self.target, self.shadow)
        disagreement = cells[1] + cells[2]
        if min(cells) < MIN_TARGET_SHADOW_CELL_COUNT or not (
            math.ceil(MIN_TARGET_SHADOW_DISAGREEMENT * SCENE_COUNT)
            <= disagreement
            <= math.floor(MAX_TARGET_SHADOW_DISAGREEMENT * SCENE_COUNT)
        ):
            raise CapabilityV3Error("v3 target-shadow pair violates the frozen bounds")
        expected = expected_cell_counts_v3(self.request)
        opening = tuple(self.opening)
        terminal = tuple(self.terminal)
        object.__setattr__(self, "opening", opening)
        object.__setattr__(self, "terminal", terminal)
        if len(opening) != 10 or len(terminal) != 10:
            raise CapabilityV3Error("v3 phases must each contain ten observations")
        opening_indices = tuple(item.scene_index for item in opening)
        terminal_indices = tuple(item.scene_index for item in terminal)
        if (
            len(set(opening_indices)) != 10
            or len(set(terminal_indices)) != 10
            or set(opening_indices) & set(terminal_indices)
        ):
            raise CapabilityV3Error("v3 phase scenes must be unique and disjoint")
        for phase in (opening, terminal):
            if any(item.accepted is not self.target.truth[item.scene_index] for item in phase):
                raise CapabilityV3Error("v3 labels must be exact Official-Law labels")
            observed = Counter(
                _cell_number(self.target, self.shadow, item.scene_index) for item in phase
            )
            if observed != Counter(
                {cell: count for cell, count in enumerate(expected) if count}
            ):
                raise CapabilityV3Error("v3 phase misses its exact proxy profile")

    def as_obj(self) -> dict[str, Any]:
        return {
            "schema_version": CAPABILITY_SCHEMA_VERSION_V3,
            "episode_id": self.episode_id,
            "request": self.request.as_obj(),
            "target_rule_id": self.target.rule_id,
            "target_truth_digest": self.target.truth_digest,
            "shadow_rule_id": self.shadow.rule_id,
            "shadow_truth_digest": self.shadow.truth_digest,
            "opening": [item.as_obj() for item in self.opening],
            "terminal": [item.as_obj() for item in self.terminal],
        }

    @property
    def digest(self) -> str:
        return json_digest(self.as_obj(), domain="goalzendo-interactive-hidden-episode-v3")


@dataclass(frozen=True, slots=True)
class GenerationRecordV3:
    request_id: str
    pair_rank: int
    pair_order_digest: str
    opening_scene_digest: str
    terminal_scene_digest: str

    def as_obj(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "pair_rank": self.pair_rank,
            "pair_order_digest": self.pair_order_digest,
            "opening_scene_digest": self.opening_scene_digest,
            "terminal_scene_digest": self.terminal_scene_digest,
        }


@dataclass(frozen=True, slots=True)
class EpisodeBankV3:
    spec: StageSpecV3
    episodes: tuple[HiddenEpisodeV3, ...]
    generation_records: tuple[GenerationRecordV3, ...]

    def __post_init__(self) -> None:
        episodes = tuple(self.episodes)
        records = tuple(self.generation_records)
        object.__setattr__(self, "episodes", episodes)
        object.__setattr__(self, "generation_records", records)
        if len(episodes) != len(self.spec.requests) or len(records) != len(episodes):
            raise CapabilityV3Error("v3 bank rows do not align")
        targets = [episode.target.truth_digest for episode in episodes]
        if len(set(targets)) != len(targets):
            raise CapabilityV3Error("v3 Official-Law identities must be unique")
        for index, (request, episode, record) in enumerate(
            zip(self.spec.requests, episodes, records, strict=True)
        ):
            if episode.request != request or record.request_id != request.request_id:
                raise CapabilityV3Error("v3 request binding mismatch")
            schedule = self.spec.cell_schedule[index]
            for phase in (episode.opening, episode.terminal):
                observed = tuple(
                    _cell_number(episode.target, episode.shadow, item.scene_index)
                    for item in phase
                )
                if observed != schedule:
                    raise CapabilityV3Error("v3 materialization changed the frozen cell schedule")

    def as_obj(self) -> dict[str, Any]:
        return {
            "schema_version": CAPABILITY_SCHEMA_VERSION_V3,
            "spec": self.spec.as_obj(),
            "catalog_digest": build_rule_catalog().digest,
            "partitions_digest": build_rule_identity_partitions_v2().digest,
            "eligible_pairs_v2_digest": build_eligible_target_shadow_table_v2().digest,
            "episodes": [episode.as_obj() for episode in self.episodes],
            "generation_records": [record.as_obj() for record in self.generation_records],
            "production_bank_generation_authorized": False,
            "weight_updates_authorized": False,
        }

    @property
    def digest(self) -> str:
        return json_digest(self.as_obj(), domain="goalzendo-interactive-episode-bank-v3")


def _select_pair(
    request: EpisodeRequestV3,
    *,
    reserved_targets: set[str],
    max_attempts: int,
) -> tuple[int, _PairCandidateV3]:
    catalog = build_rule_catalog()
    for rank, pair in enumerate(_request_pairs_v3(request)[:max_attempts]):
        if catalog[pair.target_index].truth_digest not in reserved_targets:
            return rank, pair
    raise CapabilityV3Error(
        f"bounded v3 pair selection failed for {request.request_id!r}"
    )


def _phase_from_schedule(
    request: EpisodeRequestV3,
    pair: _PairCandidateV3,
    target: CatalogEntry,
    shadow: CatalogEntry,
    schedule: tuple[int, ...],
    phase: Literal["opening", "terminal"],
    excluded: set[int],
) -> tuple[Observation, ...]:
    pools: list[list[int]] = [[] for _ in range(8)]
    for scene_index in range(SCENE_COUNT):
        pools[_cell_number(target, shadow, scene_index)].append(scene_index)
    selected: dict[int, list[int]] = {}
    for cell, count in Counter(schedule).items():
        candidates = heapq.nsmallest(
            count,
            (scene for scene in pools[cell] if scene not in excluded),
            key=lambda scene: _hash_parts(
                _SCENE_SELECTION_DOMAIN_V3,
                request.request_id,
                pair.order_digest,
                phase,
                str(cell),
                str(scene),
            ),
        )
        if len(candidates) != count:
            raise CapabilityV3Error("eligible v3 pair lacks enough disjoint scenes")
        candidates.sort(
            key=lambda scene: _hash_parts(
                _SCENE_SELECTION_DOMAIN_V3,
                request.request_id,
                pair.order_digest,
                phase,
                "within-cell-order",
                str(cell),
                str(scene),
            )
        )
        selected[cell] = candidates
        excluded.update(candidates)
    offsets = Counter[int]()
    observations: list[Observation] = []
    for cell in schedule:
        scene = selected[cell][offsets[cell]]
        offsets[cell] += 1
        observations.append(Observation(scene, target.truth[scene]))
    return tuple(observations)


@lru_cache(maxsize=4)
def generate_episode_bank_v3(spec: StageSpecV3) -> EpisodeBankV3:
    """Materialize one already-frozen v3 stage without changing its schedule."""

    if type(spec) is not StageSpecV3:
        raise TypeError("generate_episode_bank_v3 requires StageSpecV3")
    catalog = build_rule_catalog()
    reserved: set[str] = set()
    episodes: list[HiddenEpisodeV3] = []
    records: list[GenerationRecordV3] = []
    for index, request in enumerate(spec.requests):
        pair_rank, pair = _select_pair(
            request,
            reserved_targets=reserved,
            max_attempts=spec.max_pair_attempts,
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
                    domain="goalzendo-interactive-opening-scenes-v3",
                ),
                json_digest(
                    [item.scene_index for item in terminal],
                    domain="goalzendo-interactive-terminal-scenes-v3",
                ),
            )
        )
        reserved.add(target.truth_digest)
    return EpisodeBankV3(spec, tuple(episodes), tuple(records))


@dataclass(frozen=True, slots=True)
class MaterializedCandidateV3:
    design: ProspectiveDesignV3
    warm: EpisodeBankV3
    capability: EpisodeBankV3

    def __post_init__(self) -> None:
        if self.warm.spec != self.design.warm or self.capability.spec != self.design.capability:
            raise CapabilityV3Error("v3 banks are not bound to the frozen design")
        warm_targets = {episode.target.truth_digest for episode in self.warm.episodes}
        cap_targets = {episode.target.truth_digest for episode in self.capability.episodes}
        if warm_targets & cap_targets:
            raise CapabilityV3Error("v3 stage target identities overlap")

    @property
    def production_bank_generation_authorized(self) -> bool:
        return False

    @property
    def weight_updates_authorized(self) -> bool:
        return False


@lru_cache(maxsize=1)
def materialize_prospective_design_v3(
    design: ProspectiveDesignV3,
) -> MaterializedCandidateV3:
    if type(design) is not ProspectiveDesignV3:
        raise TypeError("materialize_prospective_design_v3 requires ProspectiveDesignV3")
    return MaterializedCandidateV3(
        design,
        generate_episode_bank_v3(design.warm),
        generate_episode_bank_v3(design.capability),
    )
