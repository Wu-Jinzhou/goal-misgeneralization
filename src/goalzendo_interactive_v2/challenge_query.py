"""Exact whole-version-space challenges and query ceilings for G03-v2.

The routines in this module are prospective evaluation tools.  They neither
generate an episode bank nor authorize a model update.  They consume the
public immutable catalog and version-space APIs from the frozen v1 engine,
but live in a separate package so the pending G03-G v1 smoke remains bound to
its exact source bytes.

Two finite optimizations are implemented:

* an exact minimum-cardinality static set of scenes that separates a supplied
  Official Law from every other rule in a live version space; and
* exact uniform-prior, budgeted expected-identification and minimax query
  policies, compared with recursive greedy information gain.

Both searches collapse scenes that have the same relevant rule partition,
use domain-separated deterministic hash tie-breaks, and emit graph-free
canonical reports that strict parsers reconstruct from the public catalog.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

from goalzendo_interactive import (
    SCENE_COUNT,
    CatalogEntry,
    RuleCatalog,
    Scene,
    VersionSpace,
    build_rule_catalog,
    exact_minimax_identification_depth,
    scene_index,
)
from goalzendo_interactive_v2.population_audit import (
    PopulationAuditV2Error,
    SupportedCatalogContractV2,
    build_supported_catalog_contract_v2,
)

CHALLENGE_SET_SCHEMA_VERSION = 2
QUERY_POLICY_CEILING_SCHEMA_VERSION = 2
MAX_EXACT_VERSION_SPACE_SIZE = 16
MAX_QUERY_BUDGET = 6
CHALLENGE_RESERVOIR_PANEL_COUNT = 11
CHALLENGE_RESERVOIR_PANEL_SIZE = 16
GREEDY_REFERENCE_RECOVERY_BUDGET_CEILING = 4

_CHALLENGE_KIND = "g03-v2-minimum-static-version-space-challenge"
_QUERY_KIND = "g03-v2-exact-budgeted-query-policy-ceiling"
_CHALLENGE_REPORT_DOMAIN = "goalzendo-interactive-v2-challenge-report-v2"
_QUERY_REPORT_DOMAIN = "goalzendo-interactive-v2-query-ceiling-report-v2"
_EXCLUSIONS_DOMAIN = "goalzendo-interactive-v2-excluded-scenes-v1"
_CANDIDATE_POOL_DOMAIN = "goalzendo-interactive-v2-challenge-candidate-pool-v1"
_CHALLENGE_TIE_DOMAIN = b"goalzendo-interactive-v2-challenge-scene-tie-v1\0"
_QUERY_TIE_DOMAIN = b"goalzendo-interactive-v2-query-scene-tie-v1\0"
_CHALLENGE_LAYER_DOMAIN = "goalzendo-interactive-v2-challenge-dp-layer-v1"
_CHALLENGE_CANDIDATES_DOMAIN = "goalzendo-interactive-v2-challenge-candidates-v1"
_ROOT_PARTITIONS_DOMAIN = "goalzendo-interactive-v2-root-query-partitions-v1"
_POLICY_SUMMARY_DOMAIN = "goalzendo-interactive-v2-policy-summary-v1"

_AUTHORIZATION = {
    "scope": "prospective-engineering-evaluation-only",
    "production_bank_materialized": False,
    "weight_updates_authorized": False,
}

PolicyKind = Literal["expected_identification", "greedy_information", "minimax"]
_POLICY_KINDS: tuple[PolicyKind, ...] = (
    "expected_identification",
    "greedy_information",
    "minimax",
)


class ChallengeQueryV2Error(ValueError):
    """Raised when an exact v2 report cannot be constructed or verified."""


def _dump_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ChallengeQueryV2Error(f"value is not canonical JSON: {exc}") from exc


def _load_json(text: str) -> Any:
    if type(text) is not str or not text:
        raise ChallengeQueryV2Error("JSON input must be a nonempty string")

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ChallengeQueryV2Error(f"duplicate JSON object key: {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ChallengeQueryV2Error(f"non-finite JSON constant is forbidden: {value}")

    try:
        return json.loads(
            text,
            object_pairs_hook=no_duplicates,
            parse_constant=reject_constant,
        )
    except ChallengeQueryV2Error:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ChallengeQueryV2Error(f"invalid JSON: {exc}") from exc


def _json_digest(value: Any, *, domain: str) -> str:
    digest = hashlib.sha256()
    digest.update(domain.encode("ascii"))
    digest.update(b"\0")
    digest.update(_dump_json(value).encode("ascii"))
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_exact_keys(value: object, expected: set[str], *, name: str) -> Mapping[str, Any]:
    if type(value) is not dict or set(value) != expected or len(value) != len(expected):
        raise ChallengeQueryV2Error(f"{name} has noncanonical fields")
    return cast(Mapping[str, Any], value)


def _bounded_integer(
    value: object,
    *,
    name: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ChallengeQueryV2Error(f"{name} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise ChallengeQueryV2Error(f"{name} must be an integer <= {maximum}")
    return value


def _normalize_exclusions(values: Iterable[int | Scene]) -> tuple[int, ...]:
    normalized: list[int] = []
    seen: set[int] = set()
    for value in values:
        index = scene_index(value) if type(value) is Scene else value
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < SCENE_COUNT:
            raise ChallengeQueryV2Error(f"excluded scene index must lie in [0, {SCENE_COUNT})")
        if index in seen:
            raise ChallengeQueryV2Error("excluded scene indices must not contain duplicates")
        seen.add(index)
        normalized.append(index)
    return tuple(sorted(normalized))


def select_common_untouched_panel_v2(
    panels: Iterable[Iterable[int | Scene]],
    *,
    active_query_indices: Iterable[int | Scene] = (),
    greedy_query_indices: Iterable[int | Scene] = (),
) -> int:
    """Return the first panel untouched by both registered query paths.

    The reservoir contract has eleven pairwise-disjoint 16-scene panels.  An
    active policy can touch at most six panels and the registered greedy
    oracle at most four more, so their union must leave at least one panel.
    Input order is the already-canonical panel rank; the smallest surviving
    zero-based rank is selected.
    """

    normalized = tuple(_normalize_exclusions(panel) for panel in panels)
    if len(normalized) != CHALLENGE_RESERVOIR_PANEL_COUNT:
        raise ChallengeQueryV2Error(
            f"challenge reservoir must contain exactly {CHALLENGE_RESERVOIR_PANEL_COUNT} panels"
        )
    if any(len(panel) != CHALLENGE_RESERVOIR_PANEL_SIZE for panel in normalized):
        raise ChallengeQueryV2Error(
            f"every challenge reservoir panel must contain {CHALLENGE_RESERVOIR_PANEL_SIZE} scenes"
        )
    flattened = tuple(scene for panel in normalized for scene in panel)
    if len(set(flattened)) != len(flattened):
        raise ChallengeQueryV2Error("challenge reservoir panels must be pairwise scene-disjoint")

    def query_actions(
        values: Iterable[int | Scene],
        *,
        name: str,
        maximum: int,
    ) -> set[int]:
        raw: list[int] = []
        for value in values:
            index = scene_index(value) if type(value) is Scene else value
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < SCENE_COUNT:
                raise ChallengeQueryV2Error(f"{name} scene index must lie in [0, {SCENE_COUNT})")
            raw.append(index)
        if len(raw) > maximum:
            raise ChallengeQueryV2Error(f"{name} set exceeds its {maximum}-query budget")
        return set(raw)

    active = query_actions(
        active_query_indices,
        name="active query",
        maximum=MAX_QUERY_BUDGET,
    )
    greedy = query_actions(
        greedy_query_indices,
        name="greedy query",
        maximum=GREEDY_REFERENCE_RECOVERY_BUDGET_CEILING,
    )
    touched = active | greedy
    for rank, panel in enumerate(normalized):
        if touched.isdisjoint(panel):
            return rank
    raise ChallengeQueryV2Error("no common untouched challenge panel remains")


def _exclusion_digest(exclusions: tuple[int, ...]) -> str:
    return _json_digest(
        {"scene_count": SCENE_COUNT, "excluded_scene_indices": list(exclusions)},
        domain=_EXCLUSIONS_DOMAIN,
    )


def _candidate_pool_digest(candidates: tuple[int, ...] | None) -> str:
    return _json_digest(
        {
            "scene_count": SCENE_COUNT,
            "candidate_scene_indices": None if candidates is None else list(candidates),
        },
        domain=_CANDIDATE_POOL_DOMAIN,
    )


def _supported_contract(catalog: RuleCatalog) -> SupportedCatalogContractV2:
    try:
        contract = build_supported_catalog_contract_v2()
    except PopulationAuditV2Error as exc:  # pragma: no cover - fail-closed dependency boundary
        raise ChallengeQueryV2Error(f"cannot derive the v2 supported catalog: {exc}") from exc
    if contract.source_catalog_digest != catalog.digest:
        raise ChallengeQueryV2Error("version-space catalog differs from the v2 allowlist namespace")
    return contract


def _validate_context(
    space: VersionSpace,
    official: CatalogEntry,
) -> tuple[RuleCatalog, tuple[int, ...], int]:
    if type(space) is not VersionSpace:
        raise TypeError("space must be a VersionSpace")
    if type(official) is not CatalogEntry:
        raise TypeError("official must be a CatalogEntry")
    if not space.indices:
        raise ChallengeQueryV2Error("version space cannot be empty")
    if len(space) > MAX_EXACT_VERSION_SPACE_SIZE:
        raise ChallengeQueryV2Error(
            "exact v2 searches require at most "
            f"{MAX_EXACT_VERSION_SPACE_SIZE} live rules; received {len(space)}"
        )
    catalog = space.catalog
    if not 0 <= official.index < len(catalog) or catalog[official.index] != official:
        raise ChallengeQueryV2Error("Official Law is not bound to the version-space catalog")
    if official.index not in space.indices:
        raise ChallengeQueryV2Error("Official Law is absent from the supplied version space")
    contract = _supported_contract(catalog)
    supported = set(contract.supported_indices)
    unsupported = tuple(index for index in space.indices if index not in supported)
    if unsupported:
        rule_ids = _entry_rule_ids(catalog, unsupported)
        raise ChallengeQueryV2Error(
            f"version space contains rules outside the v2 supported allowlist: {rule_ids!r}"
        )
    if official.index not in supported:
        raise ChallengeQueryV2Error("Official Law is outside the v2 supported allowlist")
    official_offset = space.indices.index(official.index)
    return catalog, space.indices, official_offset


def _scene_tie_digest(domain: bytes, context_digest: str, scene: int) -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(bytes.fromhex(context_digest))
    digest.update(scene.to_bytes(8, "big"))
    return digest.hexdigest()


def _entry_rule_ids(catalog: RuleCatalog, indices: Iterable[int]) -> tuple[str, ...]:
    return tuple(catalog[index].rule_id for index in indices)


def _catalog_entry_from_identity(
    catalog: RuleCatalog,
    rule_id: object,
    truth_digest: object,
    *,
    name: str,
) -> CatalogEntry:
    if (
        type(rule_id) is not str
        or len(rule_id) != 9
        or not rule_id.startswith("g03r")
        or not rule_id[4:].isdigit()
    ):
        raise ChallengeQueryV2Error(f"invalid {name} rule id")
    index = int(rule_id[4:])
    if not 0 <= index < len(catalog):
        raise ChallengeQueryV2Error(f"{name} rule id lies outside the catalog")
    result = catalog[index]
    if result.rule_id != rule_id or result.truth_digest != truth_digest:
        raise ChallengeQueryV2Error(f"{name} rule identity or truth digest mismatch")
    return result


def _space_from_rule_ids(catalog: RuleCatalog, values: object) -> VersionSpace:
    if type(values) is not list or not values:
        raise ChallengeQueryV2Error("version_space_rule_ids must be a nonempty array")
    indices: list[int] = []
    for value in values:
        if (
            type(value) is not str
            or len(value) != 9
            or not value.startswith("g03r")
            or not value[4:].isdigit()
        ):
            raise ChallengeQueryV2Error("version-space rule id is invalid")
        index = int(value[4:])
        if not 0 <= index < len(catalog) or catalog[index].rule_id != value:
            raise ChallengeQueryV2Error("version-space rule id lies outside the catalog")
        indices.append(index)
    try:
        return VersionSpace(catalog, tuple(indices))
    except ValueError as exc:
        raise ChallengeQueryV2Error(str(exc)) from exc


@dataclass(frozen=True, slots=True)
class ChallengeCandidateClassV2:
    """All legal scenes with one Official-versus-alternatives coverage mask."""

    coverage_mask: int
    covered_rule_ids: tuple[str, ...]
    representative_scene_index: int
    equivalent_scene_count: int
    tie_break_digest: str

    def __post_init__(self) -> None:
        _bounded_integer(self.coverage_mask, name="coverage_mask", minimum=1)
        if (
            not self.covered_rule_ids
            or len(set(self.covered_rule_ids)) != len(self.covered_rule_ids)
            or any(type(rule_id) is not str for rule_id in self.covered_rule_ids)
        ):
            raise ChallengeQueryV2Error("covered_rule_ids must be nonempty and unique")
        _bounded_integer(
            self.representative_scene_index,
            name="representative_scene_index",
            maximum=SCENE_COUNT - 1,
        )
        _bounded_integer(self.equivalent_scene_count, name="equivalent_scene_count", minimum=1)
        if not _is_sha256(self.tie_break_digest):
            raise ChallengeQueryV2Error("candidate tie_break_digest must be a SHA-256")

    def as_obj(self) -> dict[str, Any]:
        return {
            "coverage_mask": self.coverage_mask,
            "covered_rule_ids": list(self.covered_rule_ids),
            "representative_scene_index": self.representative_scene_index,
            "equivalent_scene_count": self.equivalent_scene_count,
            "tie_break_digest": self.tie_break_digest,
        }


@dataclass(frozen=True, slots=True)
class ChallengeDPLayerV2:
    """One exhaustive union-DP layer used as a minimality certificate."""

    cardinality: int
    reachable_coverage_count: int
    reachable_coverages_digest: str
    full_coverage_reachable: bool

    def __post_init__(self) -> None:
        _bounded_integer(self.cardinality, name="cardinality")
        _bounded_integer(
            self.reachable_coverage_count,
            name="reachable_coverage_count",
            minimum=1,
        )
        if not _is_sha256(self.reachable_coverages_digest):
            raise ChallengeQueryV2Error("DP layer digest must be a SHA-256")
        if type(self.full_coverage_reachable) is not bool:
            raise ChallengeQueryV2Error("full_coverage_reachable must be Boolean")

    def as_obj(self) -> dict[str, Any]:
        return {
            "cardinality": self.cardinality,
            "reachable_coverage_count": self.reachable_coverage_count,
            "reachable_coverages_digest": self.reachable_coverages_digest,
            "full_coverage_reachable": self.full_coverage_reachable,
        }


@dataclass(frozen=True, slots=True)
class MinimumChallengeSetV2:
    """Canonical exact minimum challenge-set report."""

    catalog: RuleCatalog
    space: VersionSpace
    official: CatalogEntry
    excluded_scene_indices: tuple[int, ...]
    candidate_scene_indices: tuple[int, ...] | None
    alternative_rule_ids: tuple[str, ...]
    candidate_classes: tuple[ChallengeCandidateClassV2, ...]
    nonseparating_legal_scene_count: int
    dp_layers: tuple[ChallengeDPLayerV2, ...]
    selected_scene_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        catalog, _, _ = _validate_context(self.space, self.official)
        if self.catalog is not catalog:
            raise ChallengeQueryV2Error("report catalog is not the version-space catalog")
        if self.excluded_scene_indices != tuple(sorted(set(self.excluded_scene_indices))):
            raise ChallengeQueryV2Error("report exclusions must be sorted and unique")
        if self.candidate_scene_indices is not None and (
            not self.candidate_scene_indices
            or self.candidate_scene_indices != tuple(sorted(set(self.candidate_scene_indices)))
        ):
            raise ChallengeQueryV2Error(
                "candidate_scene_indices must be null or a nonempty sorted unique tuple"
            )
        if self.alternative_rule_ids != tuple(
            entry.rule_id for entry in self.space if entry.index != self.official.index
        ):
            raise ChallengeQueryV2Error("alternative rule identities are inconsistent")
        masks = tuple(candidate.coverage_mask for candidate in self.candidate_classes)
        if self.alternative_rule_ids and not masks:
            raise ChallengeQueryV2Error("nontrivial challenge report has no candidate masks")
        if masks != tuple(sorted(set(masks))):
            raise ChallengeQueryV2Error("candidate masks must be nonempty, sorted, and unique")
        _bounded_integer(
            self.nonseparating_legal_scene_count,
            name="nonseparating_legal_scene_count",
        )
        if not self.dp_layers or tuple(layer.cardinality for layer in self.dp_layers) != tuple(
            range(len(self.dp_layers))
        ):
            raise ChallengeQueryV2Error("DP layers must start at zero and be contiguous")
        if any(layer.full_coverage_reachable for layer in self.dp_layers[:-1]):
            raise ChallengeQueryV2Error("a pre-optimum DP layer reaches full coverage")
        if not self.dp_layers[-1].full_coverage_reachable:
            raise ChallengeQueryV2Error("final DP layer must reach full coverage")
        if len(self.selected_scene_indices) != self.optimum_cardinality:
            raise ChallengeQueryV2Error("selected set size differs from optimum cardinality")
        if len(set(self.selected_scene_indices)) != len(self.selected_scene_indices):
            raise ChallengeQueryV2Error("selected challenge scenes must be unique")
        if set(self.selected_scene_indices) & set(self.excluded_scene_indices):
            raise ChallengeQueryV2Error("selected challenge scenes overlap exclusions")
        if self.candidate_scene_indices is not None and not set(self.selected_scene_indices).issubset(
            self.candidate_scene_indices
        ):
            raise ChallengeQueryV2Error("selected challenge scene lies outside its candidate panel")

    @property
    def optimum_cardinality(self) -> int:
        return self.dp_layers[-1].cardinality

    @property
    def full_coverage_mask(self) -> int:
        return (1 << len(self.alternative_rule_ids)) - 1

    @property
    def candidate_classes_digest(self) -> str:
        return _json_digest(
            [candidate.as_obj() for candidate in self.candidate_classes],
            domain=_CHALLENGE_CANDIDATES_DOMAIN,
        )

    @property
    def legal_scene_count(self) -> int:
        if self.candidate_scene_indices is None:
            return int(SCENE_COUNT) - len(self.excluded_scene_indices)
        return len(set(self.candidate_scene_indices) - set(self.excluded_scene_indices))

    def _selected_objects(self) -> list[dict[str, Any]]:
        candidates = {candidate.representative_scene_index: candidate for candidate in self.candidate_classes}
        result: list[dict[str, Any]] = []
        for display_order, scene in enumerate(self.selected_scene_indices):
            candidate = candidates[scene]
            result.append(
                {
                    "display_order": display_order,
                    "scene_index": scene,
                    "official_accepted": self.official.truth[scene],
                    "coverage_mask": candidate.coverage_mask,
                    "covered_rule_ids": list(candidate.covered_rule_ids),
                    "tie_break_digest": candidate.tie_break_digest,
                }
            )
        return result

    def as_obj(self) -> dict[str, Any]:
        supported_contract = _supported_contract(self.catalog)
        return {
            "schema_version": CHALLENGE_SET_SCHEMA_VERSION,
            "report_kind": _CHALLENGE_KIND,
            "authorization": dict(_AUTHORIZATION),
            "catalog_digest": self.catalog.digest,
            "supported_catalog_digest": supported_contract.supported_catalog_digest,
            "version_space_rule_ids": list(_entry_rule_ids(self.catalog, self.space.indices)),
            "official_rule_id": self.official.rule_id,
            "official_truth_digest": self.official.truth_digest,
            "excluded_scene_indices": list(self.excluded_scene_indices),
            "exclusions_digest": _exclusion_digest(self.excluded_scene_indices),
            "candidate_scene_indices": (
                None if self.candidate_scene_indices is None else list(self.candidate_scene_indices)
            ),
            "candidate_pool_digest": _candidate_pool_digest(self.candidate_scene_indices),
            "search_universe_kind": (
                "complete-scene-universe" if self.candidate_scene_indices is None else "supplied-static-panel"
            ),
            "alternative_rule_ids": list(self.alternative_rule_ids),
            "full_coverage_mask": self.full_coverage_mask,
            "legal_scene_count": self.legal_scene_count,
            "separating_scene_count": sum(
                candidate.equivalent_scene_count for candidate in self.candidate_classes
            ),
            "nonseparating_legal_scene_count": self.nonseparating_legal_scene_count,
            "candidate_class_count": len(self.candidate_classes),
            "candidate_classes_digest": self.candidate_classes_digest,
            "candidate_classes": [candidate.as_obj() for candidate in self.candidate_classes],
            "dp_optimality_certificate": {
                "algorithm": "exhaustive-union-dp-over-alternative-rule-bitmasks",
                "optimum_cardinality": self.optimum_cardinality,
                "layers": [layer.as_obj() for layer in self.dp_layers],
            },
            "selected_scenes": self._selected_objects(),
        }

    @property
    def digest(self) -> str:
        return _json_digest(self.as_obj(), domain=_CHALLENGE_REPORT_DOMAIN)


def _challenge_candidates(
    space: VersionSpace,
    official: CatalogEntry,
    exclusions: tuple[int, ...],
    candidate_scene_indices: tuple[int, ...] | None,
) -> tuple[tuple[ChallengeCandidateClassV2, ...], int]:
    catalog = space.catalog
    alternatives = tuple(index for index in space.indices if index != official.index)
    if not alternatives:
        legal_count = (
            SCENE_COUNT - len(exclusions)
            if candidate_scene_indices is None
            else len(set(candidate_scene_indices) - set(exclusions))
        )
        return (), legal_count
    excluded = set(exclusions)
    context: dict[str, Any] = {
        "catalog_digest": catalog.digest,
        "supported_catalog_digest": _supported_contract(catalog).supported_catalog_digest,
        "version_space_rule_ids": list(_entry_rule_ids(catalog, space.indices)),
        "official_rule_id": official.rule_id,
        "exclusions_digest": _exclusion_digest(exclusions),
    }
    # Preserve the original complete-universe tie contract while binding a
    # restricted panel explicitly whenever one is supplied.
    if candidate_scene_indices is not None:
        context["candidate_pool_digest"] = _candidate_pool_digest(candidate_scene_indices)
    context_digest = _json_digest(
        context,
        domain="goalzendo-interactive-v2-challenge-context-v2",
    )
    # mask -> [count, representative scene, representative tie digest]
    classes: dict[int, list[Any]] = {}
    nonseparating = 0
    official_bits = official.truth.bits
    alternative_bits = tuple(catalog[index].truth.bits for index in alternatives)
    search_scenes: Iterable[int] = (
        range(SCENE_COUNT) if candidate_scene_indices is None else candidate_scene_indices
    )
    for scene in search_scenes:
        if scene in excluded:
            continue
        official_label = (official_bits >> scene) & 1
        mask = 0
        for offset, bits in enumerate(alternative_bits):
            if ((bits >> scene) & 1) != official_label:
                mask |= 1 << offset
        if mask == 0:
            nonseparating += 1
            continue
        tie = _scene_tie_digest(_CHALLENGE_TIE_DOMAIN, context_digest, scene)
        current = classes.get(mask)
        if current is None:
            classes[mask] = [1, scene, tie]
        else:
            current[0] += 1
            if (tie, scene) < (current[2], current[1]):
                current[1] = scene
                current[2] = tie
    candidates = tuple(
        ChallengeCandidateClassV2(
            coverage_mask=mask,
            covered_rule_ids=tuple(
                catalog[index].rule_id for offset, index in enumerate(alternatives) if mask & (1 << offset)
            ),
            representative_scene_index=cast(int, values[1]),
            equivalent_scene_count=cast(int, values[0]),
            tie_break_digest=cast(str, values[2]),
        )
        for mask, values in sorted(classes.items())
    )
    return candidates, nonseparating


def _reachable_layer_digest(cardinality: int, states: set[int]) -> str:
    return _json_digest(
        {"cardinality": cardinality, "reachable_coverages": sorted(states)},
        domain=_CHALLENGE_LAYER_DOMAIN,
    )


def _challenge_dp(
    candidates: tuple[ChallengeCandidateClassV2, ...],
    full_mask: int,
) -> tuple[tuple[ChallengeDPLayerV2, ...], tuple[int, ...]]:
    if full_mask == 0:
        layer = ChallengeDPLayerV2(0, 1, _reachable_layer_digest(0, {0}), True)
        return (layer,), ()
    if not candidates or not any(candidate.coverage_mask & full_mask for candidate in candidates):
        raise ChallengeQueryV2Error("legal scenes cannot separate the Official Law")

    ordered_indices = tuple(
        sorted(
            range(len(candidates)),
            key=lambda index: (
                candidates[index].tie_break_digest,
                candidates[index].representative_scene_index,
                candidates[index].coverage_mask,
            ),
        )
    )
    rank = {candidate_index: order for order, candidate_index in enumerate(ordered_indices)}
    reachable = {0}
    # One canonical exact-cardinality path per reachable union.  Reusing a
    # candidate is harmless for nonoptimal states; a first-hit full cover can
    # never contain such a redundant reuse.
    paths: dict[int, tuple[int, ...]] = {0: ()}
    layers: list[ChallengeDPLayerV2] = [
        ChallengeDPLayerV2(0, 1, _reachable_layer_digest(0, reachable), False)
    ]
    for cardinality in range(1, len(candidates) + 1):
        next_reachable: set[int] = set()
        next_paths: dict[int, tuple[int, ...]] = {}
        for state in reachable:
            path = paths[state]
            for candidate_index, candidate in enumerate(candidates):
                combined = state | candidate.coverage_mask
                next_reachable.add(combined)
                candidate_path = tuple(sorted((*path, candidate_index), key=lambda index: rank[index]))
                previous = next_paths.get(combined)
                if previous is None or tuple(rank[index] for index in candidate_path) < tuple(
                    rank[index] for index in previous
                ):
                    next_paths[combined] = candidate_path
        full_reached = full_mask in next_reachable
        layers.append(
            ChallengeDPLayerV2(
                cardinality,
                len(next_reachable),
                _reachable_layer_digest(cardinality, next_reachable),
                full_reached,
            )
        )
        if full_reached:
            selected = tuple(candidates[index].representative_scene_index for index in next_paths[full_mask])
            return tuple(layers), selected
        reachable = next_reachable
        paths = next_paths
    raise ChallengeQueryV2Error("exhaustive hitting-set DP found no full cover")


def build_minimum_challenge_set_v2(
    space: VersionSpace,
    official: CatalogEntry,
    *,
    excluded_scene_indices: Iterable[int | Scene] = (),
    candidate_scene_indices: Iterable[int | Scene] | None = None,
) -> MinimumChallengeSetV2:
    """Construct an exact minimum static separator for the Official Law.

    The returned scenes exclude every supplied index, cover every alternative
    rule, and are minimum cardinality over the supplied candidate panel or,
    when no panel is supplied, all remaining scenes in the 13,716-scene
    universe. Hashes choose the representative for each equal-coverage class
    and the unique displayed optimum among cardinality ties.
    """

    catalog, _, _ = _validate_context(space, official)
    exclusions = _normalize_exclusions(excluded_scene_indices)
    candidates = None if candidate_scene_indices is None else _normalize_exclusions(candidate_scene_indices)
    if candidates == ():
        raise ChallengeQueryV2Error("candidate_scene_indices cannot be empty")
    alternatives = tuple(entry.rule_id for entry in space if entry.index != official.index)
    if not alternatives:
        trivial_layers = (ChallengeDPLayerV2(0, 1, _reachable_layer_digest(0, {0}), True),)
        return MinimumChallengeSetV2(
            catalog,
            space,
            official,
            exclusions,
            candidates,
            alternatives,
            (),
            (SCENE_COUNT - len(exclusions) if candidates is None else len(set(candidates) - set(exclusions))),
            trivial_layers,
            (),
        )
    candidate_classes, nonseparating = _challenge_candidates(
        space,
        official,
        exclusions,
        candidates,
    )
    full_mask = (1 << len(alternatives)) - 1
    if not candidate_classes or not any(
        candidate.coverage_mask == full_mask for candidate in candidate_classes
    ):
        union = 0
        for candidate in candidate_classes:
            union |= candidate.coverage_mask
        if union != full_mask:
            missing = [rule_id for offset, rule_id in enumerate(alternatives) if not union & (1 << offset)]
            raise ChallengeQueryV2Error(
                f"exclusions remove every separator for alternative rules: {missing!r}"
            )
    layers, selected = _challenge_dp(candidate_classes, full_mask)
    return MinimumChallengeSetV2(
        catalog,
        space,
        official,
        exclusions,
        candidates,
        alternatives,
        candidate_classes,
        nonseparating,
        layers,
        selected,
    )


def verify_minimum_challenge_set_v2(report: MinimumChallengeSetV2) -> MinimumChallengeSetV2:
    """Recompute every candidate, DP layer, tie-break, and selected scene."""

    if type(report) is not MinimumChallengeSetV2:
        raise TypeError("report must be a MinimumChallengeSetV2")
    expected = build_minimum_challenge_set_v2(
        report.space,
        report.official,
        excluded_scene_indices=report.excluded_scene_indices,
        candidate_scene_indices=report.candidate_scene_indices,
    )
    if report.as_obj() != expected.as_obj():
        raise ChallengeQueryV2Error("minimum challenge-set report failed exact replay")
    return report


def serialize_minimum_challenge_set_v2(report: MinimumChallengeSetV2) -> str:
    if type(report) is not MinimumChallengeSetV2:
        raise TypeError("report must be a MinimumChallengeSetV2")
    return _dump_json(report.as_obj())


def minimum_challenge_set_v2_from_obj(
    value: object,
    *,
    catalog: RuleCatalog | None = None,
) -> MinimumChallengeSetV2:
    expected_keys = {
        "schema_version",
        "report_kind",
        "authorization",
        "catalog_digest",
        "supported_catalog_digest",
        "version_space_rule_ids",
        "official_rule_id",
        "official_truth_digest",
        "excluded_scene_indices",
        "exclusions_digest",
        "candidate_scene_indices",
        "candidate_pool_digest",
        "search_universe_kind",
        "alternative_rule_ids",
        "full_coverage_mask",
        "legal_scene_count",
        "separating_scene_count",
        "nonseparating_legal_scene_count",
        "candidate_class_count",
        "candidate_classes_digest",
        "candidate_classes",
        "dp_optimality_certificate",
        "selected_scenes",
    }
    obj = _require_exact_keys(value, expected_keys, name="minimum challenge-set report")
    if (
        type(obj["schema_version"]) is not int
        or obj["schema_version"] != CHALLENGE_SET_SCHEMA_VERSION
        or obj["report_kind"] != _CHALLENGE_KIND
    ):
        raise ChallengeQueryV2Error("unsupported minimum challenge-set report")
    selected_catalog = build_rule_catalog() if catalog is None else catalog
    if obj["catalog_digest"] != selected_catalog.digest:
        raise ChallengeQueryV2Error("challenge report catalog digest mismatch")
    if obj["supported_catalog_digest"] != _supported_contract(selected_catalog).supported_catalog_digest:
        raise ChallengeQueryV2Error("challenge report supported-catalog digest mismatch")
    space = _space_from_rule_ids(selected_catalog, obj["version_space_rule_ids"])
    official = _catalog_entry_from_identity(
        selected_catalog,
        obj["official_rule_id"],
        obj["official_truth_digest"],
        name="Official Law",
    )
    raw_exclusions = obj["excluded_scene_indices"]
    if type(raw_exclusions) is not list:
        raise ChallengeQueryV2Error("excluded_scene_indices must be an array")
    exclusions = _normalize_exclusions(cast(list[int], raw_exclusions))
    if list(exclusions) != raw_exclusions:
        raise ChallengeQueryV2Error("excluded_scene_indices are not canonical")
    raw_candidates = obj["candidate_scene_indices"]
    if raw_candidates is None:
        candidates = None
    else:
        if type(raw_candidates) is not list:
            raise ChallengeQueryV2Error("candidate_scene_indices must be null or an array")
        candidates = _normalize_exclusions(cast(list[int], raw_candidates))
        if list(candidates) != raw_candidates or not candidates:
            raise ChallengeQueryV2Error("candidate_scene_indices are not canonical")
    expected = build_minimum_challenge_set_v2(
        space,
        official,
        excluded_scene_indices=exclusions,
        candidate_scene_indices=candidates,
    )
    if _dump_json(expected.as_obj()) != _dump_json(value):
        raise ChallengeQueryV2Error("minimum challenge-set report is inconsistent or tampered")
    return expected


def parse_minimum_challenge_set_v2(
    text: str,
    *,
    catalog: RuleCatalog | None = None,
    require_canonical: bool = True,
) -> MinimumChallengeSetV2:
    value = _load_json(text)
    result = minimum_challenge_set_v2_from_obj(value, catalog=catalog)
    if require_canonical and serialize_minimum_challenge_set_v2(result) != text:
        raise ChallengeQueryV2Error("challenge report is valid but not canonical JSON")
    return result


@dataclass(frozen=True, slots=True)
class QueryChoiceV2:
    scene_index: int
    tie_break_digest: str
    equivalent_scene_count: int
    rejected_rule_ids: tuple[str, ...]
    accepted_rule_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _bounded_integer(self.scene_index, name="query scene_index", maximum=SCENE_COUNT - 1)
        if not _is_sha256(self.tie_break_digest):
            raise ChallengeQueryV2Error("query tie_break_digest must be a SHA-256")
        _bounded_integer(self.equivalent_scene_count, name="equivalent_scene_count", minimum=1)
        if not self.rejected_rule_ids or not self.accepted_rule_ids:
            raise ChallengeQueryV2Error("query choice must have two nonempty children")
        if set(self.rejected_rule_ids) & set(self.accepted_rule_ids):
            raise ChallengeQueryV2Error("query children must be disjoint")

    def as_obj(self) -> dict[str, Any]:
        return {
            "scene_index": self.scene_index,
            "tie_break_digest": self.tie_break_digest,
            "equivalent_scene_count": self.equivalent_scene_count,
            "rejected_rule_ids": list(self.rejected_rule_ids),
            "accepted_rule_ids": list(self.accepted_rule_ids),
            "rejected_count": len(self.rejected_rule_ids),
            "accepted_count": len(self.accepted_rule_ids),
        }


@dataclass(frozen=True, slots=True)
class PolicyPathStepV2:
    turn: int
    before_rule_ids: tuple[str, ...]
    query: QueryChoiceV2
    official_accepted: bool
    after_rule_ids: tuple[str, ...]

    def as_obj(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "before_rule_ids": list(self.before_rule_ids),
            "query": self.query.as_obj(),
            "official_accepted": self.official_accepted,
            "after_rule_ids": list(self.after_rule_ids),
        }


@dataclass(frozen=True, slots=True)
class PolicySummaryV2:
    policy_kind: PolicyKind
    budget: int
    prior_rule_count: int
    identified_rule_count: int
    total_queries_over_uniform_rules: int
    worst_case_remaining_rule_count: int
    terminal_state_size_mass: tuple[tuple[int, int], ...]
    first_query: QueryChoiceV2 | None
    official_path: tuple[PolicyPathStepV2, ...]
    official_terminal_rule_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.policy_kind not in _POLICY_KINDS:
            raise ChallengeQueryV2Error("unknown policy kind")
        _bounded_integer(self.budget, name="budget", maximum=MAX_QUERY_BUDGET)
        _bounded_integer(self.prior_rule_count, name="prior_rule_count", minimum=1)
        _bounded_integer(
            self.identified_rule_count,
            name="identified_rule_count",
            maximum=self.prior_rule_count,
        )
        _bounded_integer(
            self.total_queries_over_uniform_rules,
            name="total_queries_over_uniform_rules",
        )
        _bounded_integer(
            self.worst_case_remaining_rule_count,
            name="worst_case_remaining_rule_count",
            minimum=1,
            maximum=self.prior_rule_count,
        )
        if not self.terminal_state_size_mass:
            raise ChallengeQueryV2Error("terminal_state_size_mass cannot be empty")
        if tuple(sorted(self.terminal_state_size_mass)) != self.terminal_state_size_mass:
            raise ChallengeQueryV2Error("terminal_state_size_mass must be sorted")
        if sum(mass for _, mass in self.terminal_state_size_mass) != self.prior_rule_count:
            raise ChallengeQueryV2Error("terminal-state prior mass must sum to prior_rule_count")
        if dict(self.terminal_state_size_mass).get(1, 0) != self.identified_rule_count:
            raise ChallengeQueryV2Error("identified count differs from singleton prior mass")
        if len(self.official_path) > self.budget:
            raise ChallengeQueryV2Error("Official path exceeds the query budget")
        if not self.official_terminal_rule_ids:
            raise ChallengeQueryV2Error("Official path must end in a nonempty version space")

    def _body_obj(self) -> dict[str, Any]:
        return {
            "policy_kind": self.policy_kind,
            "budget": self.budget,
            "uniform_prior_denominator": self.prior_rule_count,
            "identified_rule_count": self.identified_rule_count,
            "total_queries_over_uniform_rules": self.total_queries_over_uniform_rules,
            "worst_case_remaining_rule_count": self.worst_case_remaining_rule_count,
            "terminal_state_size_mass": [
                {"remaining_rule_count": size, "prior_rule_count": mass}
                for size, mass in self.terminal_state_size_mass
            ],
            "first_query": None if self.first_query is None else self.first_query.as_obj(),
            "official_path": [step.as_obj() for step in self.official_path],
            "official_terminal_rule_ids": list(self.official_terminal_rule_ids),
        }

    def as_obj(self) -> dict[str, Any]:
        body = self._body_obj()
        return {**body, "policy_digest": _json_digest(body, domain=_POLICY_SUMMARY_DOMAIN)}


@dataclass(frozen=True, slots=True)
class PolicyBudgetComparisonV2:
    budget: int
    optimal: PolicySummaryV2
    greedy: PolicySummaryV2
    minimax: PolicySummaryV2

    def __post_init__(self) -> None:
        if {self.optimal.budget, self.greedy.budget, self.minimax.budget} != {self.budget}:
            raise ChallengeQueryV2Error("policy summaries do not share their budget")
        if (
            self.optimal.policy_kind != "expected_identification"
            or self.greedy.policy_kind != "greedy_information"
            or self.minimax.policy_kind != "minimax"
        ):
            raise ChallengeQueryV2Error("budget comparison has mislabelled policies")

    def as_obj(self) -> dict[str, Any]:
        denominator = self.optimal.prior_rule_count
        return {
            "budget": self.budget,
            "optimal_expected_identification": self.optimal.as_obj(),
            "greedy_information": self.greedy.as_obj(),
            "exact_minimax": self.minimax.as_obj(),
            "greedy_identification_regret_numerator": (
                self.optimal.identified_rule_count - self.greedy.identified_rule_count
            ),
            "minimax_identification_regret_numerator": (
                self.optimal.identified_rule_count - self.minimax.identified_rule_count
            ),
            "regret_denominator": denominator,
        }


@dataclass(frozen=True, slots=True)
class QueryPolicyCeilingReportV2:
    catalog: RuleCatalog
    space: VersionSpace
    official: CatalogEntry
    excluded_scene_indices: tuple[int, ...]
    legal_scene_count: int
    root_label_pattern_count: int
    root_informative_scene_count: int
    root_partition_class_count: int
    root_partition_classes_digest: str
    budgets: tuple[PolicyBudgetComparisonV2, ...]
    unrestricted_v1_minimax_depth: int | None

    def __post_init__(self) -> None:
        catalog, _, _ = _validate_context(self.space, self.official)
        if self.catalog is not catalog:
            raise ChallengeQueryV2Error("report catalog is not the version-space catalog")
        if self.excluded_scene_indices != tuple(sorted(set(self.excluded_scene_indices))):
            raise ChallengeQueryV2Error("query report exclusions must be sorted and unique")
        if self.legal_scene_count != SCENE_COUNT - len(self.excluded_scene_indices):
            raise ChallengeQueryV2Error("legal_scene_count is inconsistent")
        for name in ("root_label_pattern_count",):
            _bounded_integer(getattr(self, name), name=name, minimum=1)
        for name in ("root_informative_scene_count", "root_partition_class_count"):
            minimum = 0 if len(self.space) == 1 else 1
            _bounded_integer(getattr(self, name), name=name, minimum=minimum)
        if not _is_sha256(self.root_partition_classes_digest):
            raise ChallengeQueryV2Error("root partition digest must be a SHA-256")
        if tuple(comparison.budget for comparison in self.budgets) != tuple(range(MAX_QUERY_BUDGET + 1)):
            raise ChallengeQueryV2Error("query report must contain budgets zero through six")
        if self.unrestricted_v1_minimax_depth is not None:
            _bounded_integer(
                self.unrestricted_v1_minimax_depth,
                name="unrestricted_v1_minimax_depth",
                maximum=MAX_QUERY_BUDGET,
            )

    @property
    def excluded_exact_minimax_depth(self) -> int | None:
        return next(
            (
                comparison.budget
                for comparison in self.budgets
                if comparison.minimax.worst_case_remaining_rule_count == 1
            ),
            None,
        )

    @property
    def exact_expected_full_identification_budget(self) -> int | None:
        return next(
            (
                comparison.budget
                for comparison in self.budgets
                if comparison.optimal.identified_rule_count == len(self.space)
            ),
            None,
        )

    @property
    def greedy_full_identification_budget(self) -> int | None:
        return next(
            (
                comparison.budget
                for comparison in self.budgets
                if comparison.greedy.identified_rule_count == len(self.space)
            ),
            None,
        )

    @property
    def greedy_official_recovery_query_count(self) -> int | None:
        return next(
            (
                len(comparison.greedy.official_path)
                for comparison in self.budgets
                if comparison.greedy.official_terminal_rule_ids == (self.official.rule_id,)
            ),
            None,
        )

    @property
    def greedy_reference_recovery_within_ceiling(self) -> bool:
        query_count = self.greedy_official_recovery_query_count
        return query_count is not None and query_count <= GREEDY_REFERENCE_RECOVERY_BUDGET_CEILING

    def as_obj(self) -> dict[str, Any]:
        excluded_depth = self.excluded_exact_minimax_depth
        unrestricted = self.unrestricted_v1_minimax_depth
        supported_contract = _supported_contract(self.catalog)
        if not self.excluded_scene_indices:
            minimax_consistent = excluded_depth == unrestricted
        elif unrestricted is None:
            minimax_consistent = excluded_depth is None
        elif excluded_depth is None:
            minimax_consistent = True
        else:
            minimax_consistent = unrestricted <= excluded_depth
        return {
            "schema_version": QUERY_POLICY_CEILING_SCHEMA_VERSION,
            "report_kind": _QUERY_KIND,
            "authorization": dict(_AUTHORIZATION),
            "catalog_digest": self.catalog.digest,
            "supported_catalog_digest": supported_contract.supported_catalog_digest,
            "version_space_rule_ids": list(_entry_rule_ids(self.catalog, self.space.indices)),
            "official_rule_id": self.official.rule_id,
            "official_truth_digest": self.official.truth_digest,
            "excluded_scene_indices": list(self.excluded_scene_indices),
            "exclusions_digest": _exclusion_digest(self.excluded_scene_indices),
            "uniform_prior_rule_count": len(self.space),
            "legal_scene_count": self.legal_scene_count,
            "root_label_pattern_count": self.root_label_pattern_count,
            "root_informative_scene_count": self.root_informative_scene_count,
            "root_partition_class_count": self.root_partition_class_count,
            "root_partition_classes_digest": self.root_partition_classes_digest,
            "partition_collapse": (
                "unordered accepted/rejected live-rule subsets; one hash-selected "
                "representative scene per class"
            ),
            "budgets": [comparison.as_obj() for comparison in self.budgets],
            "exact_expected_full_identification_budget": (self.exact_expected_full_identification_budget),
            "excluded_scene_exact_minimax_depth": excluded_depth,
            "unrestricted_v1_exact_minimax_depth": unrestricted,
            "minimax_cross_check": {
                "comparable": not self.excluded_scene_indices,
                "consistent": minimax_consistent,
            },
            "greedy_full_identification_budget": self.greedy_full_identification_budget,
            "greedy_official_recovery_query_count": self.greedy_official_recovery_query_count,
            "greedy_reference_recovery_gate": {
                "budget_ceiling": GREEDY_REFERENCE_RECOVERY_BUDGET_CEILING,
                "criterion": ("Official-Law path ends in the singleton Official rule within the ceiling"),
                "official_rule_id": self.official.rule_id,
                "official_query_count": self.greedy_official_recovery_query_count,
                "passed": self.greedy_reference_recovery_within_ceiling,
            },
        }

    @property
    def digest(self) -> str:
        return _json_digest(self.as_obj(), domain=_QUERY_REPORT_DOMAIN)


@dataclass(frozen=True, slots=True)
class _Pattern:
    label_mask: int
    scene_count: int
    representative_scene_index: int
    tie_break_digest: str


@dataclass(frozen=True, slots=True)
class _QueryClass:
    partition_mask: int
    accepted_mask: int
    rejected_mask: int
    scene_count: int
    representative_scene_index: int
    tie_break_digest: str


@dataclass(frozen=True, slots=True)
class _PolicyNode:
    identified: int
    total_queries: int
    worst_remaining: int
    terminal_mass: tuple[tuple[int, int], ...]
    choice: _QueryClass | None


class _QuerySolver:
    def __init__(
        self,
        space: VersionSpace,
        official: CatalogEntry,
        exclusions: tuple[int, ...],
    ) -> None:
        self.space = space
        self.catalog = space.catalog
        self.official = official
        self.exclusions = exclusions
        self.indices = space.indices
        self.full_state = (1 << len(space)) - 1
        self.official_bit = 1 << self.indices.index(official.index)
        context_digest = _json_digest(
            {
                "catalog_digest": self.catalog.digest,
                "supported_catalog_digest": _supported_contract(self.catalog).supported_catalog_digest,
                "version_space_rule_ids": list(_entry_rule_ids(self.catalog, self.indices)),
                "exclusions_digest": _exclusion_digest(exclusions),
            },
            domain="goalzendo-interactive-v2-query-context-v2",
        )
        excluded = set(exclusions)
        truth_bits = tuple(self.catalog[index].truth.bits for index in self.indices)
        grouped: dict[int, list[Any]] = {}
        for scene in range(SCENE_COUNT):
            if scene in excluded:
                continue
            labels = 0
            for offset, bits in enumerate(truth_bits):
                if (bits >> scene) & 1:
                    labels |= 1 << offset
            tie = _scene_tie_digest(_QUERY_TIE_DOMAIN, context_digest, scene)
            current = grouped.get(labels)
            if current is None:
                grouped[labels] = [1, scene, tie]
            else:
                current[0] += 1
                if (tie, scene) < (current[2], current[1]):
                    current[1] = scene
                    current[2] = tie
        if not grouped:
            raise ChallengeQueryV2Error("query policy has no legal scenes")
        self.patterns = tuple(
            _Pattern(mask, values[0], values[1], values[2]) for mask, values in sorted(grouped.items())
        )
        self._classes_cache: dict[int, tuple[_QueryClass, ...]] = {}
        self._optimal_cache: dict[tuple[int, int], _PolicyNode] = {}
        self._greedy_cache: dict[tuple[int, int], _PolicyNode] = {}
        self._minimax_cache: dict[tuple[int, int], _PolicyNode] = {}

    def classes(self, state: int) -> tuple[_QueryClass, ...]:
        if state <= 0 or state & ~self.full_state:
            raise ChallengeQueryV2Error("query solver received an invalid state")
        cached = self._classes_cache.get(state)
        if cached is not None:
            return cached
        grouped: dict[int, list[Any]] = {}
        for pattern in self.patterns:
            accepted = pattern.label_mask & state
            rejected = state ^ accepted
            if not accepted or not rejected:
                continue
            key = min(accepted, rejected)
            current = grouped.get(key)
            if current is None:
                grouped[key] = [
                    pattern.scene_count,
                    pattern.representative_scene_index,
                    pattern.tie_break_digest,
                    accepted,
                    rejected,
                ]
            else:
                current[0] += pattern.scene_count
                if (pattern.tie_break_digest, pattern.representative_scene_index) < (
                    current[2],
                    current[1],
                ):
                    current[1] = pattern.representative_scene_index
                    current[2] = pattern.tie_break_digest
                    current[3] = accepted
                    current[4] = rejected
        result = tuple(
            _QueryClass(
                partition_mask=partition,
                accepted_mask=values[3],
                rejected_mask=values[4],
                scene_count=values[0],
                representative_scene_index=values[1],
                tie_break_digest=values[2],
            )
            for partition, values in sorted(grouped.items())
        )
        self._classes_cache[state] = result
        return result

    @staticmethod
    def _base(state: int, budget: int) -> _PolicyNode | None:
        size = state.bit_count()
        if size == 1:
            return _PolicyNode(1, 0, 1, ((1, 1),), None)
        if budget == 0:
            return _PolicyNode(0, 0, size, ((size, size),), None)
        return None

    @staticmethod
    def _combine(state: int, choice: _QueryClass, left: _PolicyNode, right: _PolicyNode) -> _PolicyNode:
        mass = Counter(dict(left.terminal_mass))
        mass.update(dict(right.terminal_mass))
        return _PolicyNode(
            identified=left.identified + right.identified,
            total_queries=state.bit_count() + left.total_queries + right.total_queries,
            worst_remaining=max(left.worst_remaining, right.worst_remaining),
            terminal_mass=tuple(sorted(mass.items())),
            choice=choice,
        )

    def optimal(self, state: int, budget: int) -> _PolicyNode:
        key_cache = (state, budget)
        cached = self._optimal_cache.get(key_cache)
        if cached is not None:
            return cached
        base = self._base(state, budget)
        if base is not None:
            self._optimal_cache[key_cache] = base
            return base
        choices = self.classes(state)
        if not choices:
            raise ChallengeQueryV2Error("distinct live rules have no legal separating query")
        best: _PolicyNode | None = None
        best_key: tuple[int, str, int, int] | None = None
        for choice in choices:
            node = self._combine(
                state,
                choice,
                self.optimal(choice.rejected_mask, budget - 1),
                self.optimal(choice.accepted_mask, budget - 1),
            )
            key = (
                -node.identified,
                choice.tie_break_digest,
                choice.representative_scene_index,
                choice.partition_mask,
            )
            if best_key is None or key < best_key:
                best = node
                best_key = key
        if best is None:
            raise AssertionError("nonempty choice set produced no optimal policy")
        self._optimal_cache[key_cache] = best
        return best

    def greedy(self, state: int, budget: int) -> _PolicyNode:
        key_cache = (state, budget)
        cached = self._greedy_cache.get(key_cache)
        if cached is not None:
            return cached
        base = self._base(state, budget)
        if base is not None:
            self._greedy_cache[key_cache] = base
            return base
        choices = self.classes(state)
        if not choices:
            raise ChallengeQueryV2Error("distinct live rules have no legal separating query")
        choice = min(
            choices,
            key=lambda item: (
                -min(item.accepted_mask.bit_count(), item.rejected_mask.bit_count()),
                item.tie_break_digest,
                item.representative_scene_index,
                item.partition_mask,
            ),
        )
        result = self._combine(
            state,
            choice,
            self.greedy(choice.rejected_mask, budget - 1),
            self.greedy(choice.accepted_mask, budget - 1),
        )
        self._greedy_cache[key_cache] = result
        return result

    def minimax(self, state: int, budget: int) -> _PolicyNode:
        key_cache = (state, budget)
        cached = self._minimax_cache.get(key_cache)
        if cached is not None:
            return cached
        base = self._base(state, budget)
        if base is not None:
            self._minimax_cache[key_cache] = base
            return base
        choices = self.classes(state)
        if not choices:
            raise ChallengeQueryV2Error("distinct live rules have no legal separating query")
        best: _PolicyNode | None = None
        best_key: tuple[int, int, str, int, int] | None = None
        for choice in choices:
            node = self._combine(
                state,
                choice,
                self.minimax(choice.rejected_mask, budget - 1),
                self.minimax(choice.accepted_mask, budget - 1),
            )
            key = (
                node.worst_remaining,
                -node.identified,
                choice.tie_break_digest,
                choice.representative_scene_index,
                choice.partition_mask,
            )
            if best_key is None or key < best_key:
                best = node
                best_key = key
        if best is None:
            raise AssertionError("nonempty choice set produced no minimax policy")
        self._minimax_cache[key_cache] = best
        return best

    def _choice_obj(self, state: int, choice: _QueryClass) -> QueryChoiceV2:
        return QueryChoiceV2(
            scene_index=choice.representative_scene_index,
            tie_break_digest=choice.tie_break_digest,
            equivalent_scene_count=choice.scene_count,
            rejected_rule_ids=_entry_rule_ids(
                self.catalog,
                (index for offset, index in enumerate(self.indices) if choice.rejected_mask & (1 << offset)),
            ),
            accepted_rule_ids=_entry_rule_ids(
                self.catalog,
                (index for offset, index in enumerate(self.indices) if choice.accepted_mask & (1 << offset)),
            ),
        )

    def summary(self, kind: PolicyKind, budget: int) -> PolicySummaryV2:
        solvers = {
            "expected_identification": self.optimal,
            "greedy_information": self.greedy,
            "minimax": self.minimax,
        }
        solve = solvers[kind]
        root = solve(self.full_state, budget)
        first = None if root.choice is None else self._choice_obj(self.full_state, root.choice)
        state = self.full_state
        remaining_budget = budget
        path: list[PolicyPathStepV2] = []
        while state.bit_count() > 1 and remaining_budget > 0:
            node = solve(state, remaining_budget)
            choice = node.choice
            if choice is None:
                break
            accepted = bool(choice.accepted_mask & self.official_bit)
            after = choice.accepted_mask if accepted else choice.rejected_mask
            path.append(
                PolicyPathStepV2(
                    turn=len(path),
                    before_rule_ids=_entry_rule_ids(
                        self.catalog,
                        (index for offset, index in enumerate(self.indices) if state & (1 << offset)),
                    ),
                    query=self._choice_obj(state, choice),
                    official_accepted=accepted,
                    after_rule_ids=_entry_rule_ids(
                        self.catalog,
                        (index for offset, index in enumerate(self.indices) if after & (1 << offset)),
                    ),
                )
            )
            state = after
            remaining_budget -= 1
        terminal_ids = _entry_rule_ids(
            self.catalog,
            (index for offset, index in enumerate(self.indices) if state & (1 << offset)),
        )
        return PolicySummaryV2(
            policy_kind=kind,
            budget=budget,
            prior_rule_count=len(self.space),
            identified_rule_count=root.identified,
            total_queries_over_uniform_rules=root.total_queries,
            worst_case_remaining_rule_count=root.worst_remaining,
            terminal_state_size_mass=root.terminal_mass,
            first_query=first,
            official_path=tuple(path),
            official_terminal_rule_ids=terminal_ids,
        )

    def root_partition_evidence(self) -> tuple[int, int, str]:
        classes = self.classes(self.full_state)
        evidence = [
            {
                "partition_mask": item.partition_mask,
                "accepted_mask": item.accepted_mask,
                "rejected_mask": item.rejected_mask,
                "scene_count": item.scene_count,
                "representative_scene_index": item.representative_scene_index,
                "tie_break_digest": item.tie_break_digest,
            }
            for item in classes
        ]
        return (
            sum(item.scene_count for item in classes),
            len(classes),
            _json_digest(evidence, domain=_ROOT_PARTITIONS_DOMAIN),
        )


def build_query_policy_ceiling_report_v2(
    space: VersionSpace,
    official: CatalogEntry,
    *,
    excluded_scene_indices: Iterable[int | Scene] = (),
) -> QueryPolicyCeilingReportV2:
    """Compute exact uniform-prior policy ceilings for budgets zero through six."""

    catalog, _, _ = _validate_context(space, official)
    exclusions = _normalize_exclusions(excluded_scene_indices)
    solver = _QuerySolver(space, official, exclusions)
    informative_count, class_count, classes_digest = solver.root_partition_evidence()
    comparisons = tuple(
        PolicyBudgetComparisonV2(
            budget,
            solver.summary("expected_identification", budget),
            solver.summary("greedy_information", budget),
            solver.summary("minimax", budget),
        )
        for budget in range(MAX_QUERY_BUDGET + 1)
    )
    unrestricted_depth = exact_minimax_identification_depth(
        space,
        max_depth=MAX_QUERY_BUDGET,
    )
    report = QueryPolicyCeilingReportV2(
        catalog=catalog,
        space=space,
        official=official,
        excluded_scene_indices=exclusions,
        legal_scene_count=SCENE_COUNT - len(exclusions),
        root_label_pattern_count=len(solver.patterns),
        root_informative_scene_count=informative_count,
        root_partition_class_count=class_count,
        root_partition_classes_digest=classes_digest,
        budgets=comparisons,
        unrestricted_v1_minimax_depth=unrestricted_depth,
    )
    if not report.as_obj()["minimax_cross_check"]["consistent"]:
        raise ChallengeQueryV2Error("exact minimax implementations disagree")
    if report.exact_expected_full_identification_budget != report.excluded_exact_minimax_depth:
        raise ChallengeQueryV2Error(
            "expected-identification and minimax policies disagree on full-identification depth"
        )
    return report


def verify_query_policy_ceiling_report_v2(
    report: QueryPolicyCeilingReportV2,
) -> QueryPolicyCeilingReportV2:
    """Recompute all partitions, policies, paths, baselines, and tie-breaks."""

    if type(report) is not QueryPolicyCeilingReportV2:
        raise TypeError("report must be a QueryPolicyCeilingReportV2")
    expected = build_query_policy_ceiling_report_v2(
        report.space,
        report.official,
        excluded_scene_indices=report.excluded_scene_indices,
    )
    if expected.as_obj() != report.as_obj():
        raise ChallengeQueryV2Error("query-policy ceiling report failed exact replay")
    return report


def serialize_query_policy_ceiling_report_v2(report: QueryPolicyCeilingReportV2) -> str:
    if type(report) is not QueryPolicyCeilingReportV2:
        raise TypeError("report must be a QueryPolicyCeilingReportV2")
    return _dump_json(report.as_obj())


def query_policy_ceiling_report_v2_from_obj(
    value: object,
    *,
    catalog: RuleCatalog | None = None,
) -> QueryPolicyCeilingReportV2:
    expected_keys = {
        "schema_version",
        "report_kind",
        "authorization",
        "catalog_digest",
        "supported_catalog_digest",
        "version_space_rule_ids",
        "official_rule_id",
        "official_truth_digest",
        "excluded_scene_indices",
        "exclusions_digest",
        "uniform_prior_rule_count",
        "legal_scene_count",
        "root_label_pattern_count",
        "root_informative_scene_count",
        "root_partition_class_count",
        "root_partition_classes_digest",
        "partition_collapse",
        "budgets",
        "exact_expected_full_identification_budget",
        "excluded_scene_exact_minimax_depth",
        "unrestricted_v1_exact_minimax_depth",
        "minimax_cross_check",
        "greedy_full_identification_budget",
        "greedy_official_recovery_query_count",
        "greedy_reference_recovery_gate",
    }
    obj = _require_exact_keys(value, expected_keys, name="query-policy ceiling report")
    if (
        type(obj["schema_version"]) is not int
        or obj["schema_version"] != QUERY_POLICY_CEILING_SCHEMA_VERSION
        or obj["report_kind"] != _QUERY_KIND
    ):
        raise ChallengeQueryV2Error("unsupported query-policy ceiling report")
    selected_catalog = build_rule_catalog() if catalog is None else catalog
    if obj["catalog_digest"] != selected_catalog.digest:
        raise ChallengeQueryV2Error("query report catalog digest mismatch")
    if obj["supported_catalog_digest"] != _supported_contract(selected_catalog).supported_catalog_digest:
        raise ChallengeQueryV2Error("query report supported-catalog digest mismatch")
    space = _space_from_rule_ids(selected_catalog, obj["version_space_rule_ids"])
    official = _catalog_entry_from_identity(
        selected_catalog,
        obj["official_rule_id"],
        obj["official_truth_digest"],
        name="Official Law",
    )
    raw_exclusions = obj["excluded_scene_indices"]
    if type(raw_exclusions) is not list:
        raise ChallengeQueryV2Error("excluded_scene_indices must be an array")
    exclusions = _normalize_exclusions(cast(list[int], raw_exclusions))
    if list(exclusions) != raw_exclusions:
        raise ChallengeQueryV2Error("excluded_scene_indices are not canonical")
    expected = build_query_policy_ceiling_report_v2(
        space,
        official,
        excluded_scene_indices=exclusions,
    )
    if _dump_json(expected.as_obj()) != _dump_json(value):
        raise ChallengeQueryV2Error("query-policy ceiling report is inconsistent or tampered")
    return expected


def parse_query_policy_ceiling_report_v2(
    text: str,
    *,
    catalog: RuleCatalog | None = None,
    require_canonical: bool = True,
) -> QueryPolicyCeilingReportV2:
    value = _load_json(text)
    result = query_policy_ceiling_report_v2_from_obj(value, catalog=catalog)
    if require_canonical and serialize_query_policy_ceiling_report_v2(result) != text:
        raise ChallengeQueryV2Error("query-policy report is valid but not canonical JSON")
    return result
