"""Exact conditional and descriptive statistical leakage audits for G03-v2.

The primary meta-role population is *hypothesis complete*: for one fixed live
version space, every supported candidate is Official exactly once.  This is a
necessary correction to the historical three-role P/Q/C rotation.  With
``8 <= |V0| <= 16`` and one fitting placard rule, the historical rotation lets
the required ``placard_inclusion`` feature identify the Official with accuracy
``1/3`` although the registered full-space chance is at most ``1/8``.

Exact within-opening conditional balance is authoritative.  The deterministic
five-fold target-rate classifier and no-refit cluster bootstrap are descriptive
only: they do not have unconditional coverage authorization.  This module is
deliberately nonauthorizing, does not replace the exact public terminal-law
audit, and cannot authorize a bank, model execution, or a weight update.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from typing import Any, Literal, cast

import numpy as np

from goalzendo_interactive.catalog import VersionSpace, build_rule_catalog
from goalzendo_interactive.rendering import RENDERERS, TRAIN_RENDERERS, RendererName, render_scene
from goalzendo_interactive.rules import BinaryRule
from goalzendo_interactive.rules import Literal as RuleLiteral
from goalzendo_interactive.schema import POSITIONS, SCENE_COUNT, Scene, scene_at

from .population_audit import (
    RuleTripleBindingV2,
    build_supported_catalog_contract_v2,
)

STATISTICAL_LEAKAGE_SCHEMA_VERSION = 2
STATISTICAL_LEAKAGE_FOLDS = 5
STATISTICAL_LEAKAGE_BOOTSTRAP_REPLICATES = 10_000
STATISTICAL_LEAKAGE_MINIMUM_GROUPS = 384
STATISTICAL_LEAKAGE_CONFIDENCE_LEVEL = 0.95
STATISTICAL_LEAKAGE_EXCESS_MARGIN = 0.05
MIN_LIVE_RULES = 8
MAX_LIVE_RULES = 16
REGISTERED_HYPOTHESIS_COMPLETE_SIZES = (8, 12, 16)

AuditKindV2 = Literal["meta_role", "terminal_one_item"]
DecisionV2 = Literal["pass", "leakage", "insufficient_data"]

_REPORT_KIND = "g03-v2-exact-conditional-and-grouped-statistical-leakage-audit"
_REPORT_DOMAIN = "goalzendo-interactive-v2-statistical-leakage-report-v2"
_DATASET_DOMAIN = "goalzendo-interactive-v2-statistical-leakage-dataset-v2"
_BLOCK_OPENING_DOMAIN = "goalzendo-interactive-v2-hypothesis-complete-block-opening-v2"
_FREQUENCY_DOMAIN = "goalzendo-interactive-v2-catalog-frequency-table-v2"
_CONSTRUCTION_CLUSTER_DOMAIN = "goalzendo-interactive-v2-construction-cluster-v2"
_CONDITIONAL_CELL_DOMAIN = "goalzendo-interactive-v2-exact-conditional-cells-v2"
_PROMPT_SURFACE_DOMAIN = "goalzendo-interactive-v2-rendered-static-prompt-v2"
_FOLD_DOMAIN = b"goalzendo-interactive-v2-statistical-leakage-fold-v2\0"
_FOLD_REPORT_DOMAIN = "goalzendo-interactive-v2-statistical-leakage-fold-report-v2"
_PREDICTION_DOMAIN = "goalzendo-interactive-v2-statistical-leakage-predictions-v2"
_PREDICTION_PATTERN_DOMAIN = (
    "goalzendo-interactive-v2-statistical-leakage-prediction-pattern-v2"
)
_BOOTSTRAP_SEED_DOMAIN = b"goalzendo-interactive-v2-statistical-leakage-bootstrap-v2\0"

_AUTHORIZATION = {
    "scope": "prospective_exact_conditional_and_descriptive_statistical_engineering_only",
    "rotation_surfaces_rederived_from_bank_manifest": False,
    "construction_lineage_rederived_from_bank_manifest": False,
    "oof_cluster_bootstrap_unconditional_coverage_authorized": False,
    "held_out_rule_conditional_selection_audit_substituted": False,
    "analytic_public_generator_law_audit_substituted": False,
    "production_bank_authorized": False,
    "model_execution_authorized": False,
    "weight_updates_authorized": False,
}

_METHOD_CONTRACT = {
    "classifier": (
        "additive categorical out-of-fold target-rate scorer; each channel uses an "
        "exact integer numerator centered at the training-fold class prevalence, "
        "then exact rational accumulation; every view also contributes one "
        "canonical full-joint feature-cell token as a descriptive OOF stress; the "
        "separate exact within-opening conditional-balance gate is authoritative"
    ),
    "top_one_ties": "fractional credit uniformly over every exact maximum-score candidate",
    "folds": (
        "five deterministic folds grouped by rederived construction clusters; openings "
        "sharing any exact nine-of-ten semantic projection remain in one cluster"
    ),
    "bootstrap": (
        "deterministic construction-cluster percentile bootstrap over fixed OOF predictions; "
        "empirical order-statistic 95-percent descriptive interval"
    ),
    "full_v0_chance": "episode mean of 1/|V0|",
    "designated_p_q_c_chance": "1/3; historical diagnostic only and never a launch gate",
    "one_v_rest_macro_ba_chance": "1/2",
    "descriptive_equivalence_rule": (
        "full-V0 excess interval contains zero and upper excess is strictly below .05; "
        "macro-BA interval contains .5 and upper endpoint is strictly below .55"
    ),
    "minimum_power": "at least 384 content-rederived construction clusters",
    "interval_scope": (
        "descriptive conditional-on-fitted-OOF-predictions interval only; the current "
        "no-refit bootstrap has no unconditional coverage authorization"
    ),
    "exact_conditional_gate": (
        "mandatory exact Official-independence within opening x exact candidate x complete "
        "model-visible surface; n0*positive_count must equal row_count in every cell"
    ),
    "registered_hypothesis_complete_sizes": "fixed n0 must be one of 8, 12, or 16",
    "terminal_scope": (
        "Official-oblivious shared draws audit hypothesis-complete training stress only; "
        "it is not the held-out rule-conditional evaluation selection audit"
    ),
}

_META_VIEW_CONTRACT: tuple[tuple[str, bool], ...] = (
    ("catalog_structure", True),
    ("renderer", True),
    ("token_length_bin", True),
    ("global_training_frequency", True),
    ("full_model_visible", True),
    ("schedule_position", False),
    ("request_position", False),
    ("bank_prefix", False),
    ("full_executor_diagnostic", False),
)

_TERMINAL_VIEW_CONTRACT: tuple[tuple[str, bool], ...] = (
    ("catalog_structure", True),
    ("renderer", True),
    ("scene_attributes", True),
    ("candidate_truth", True),
    ("opening", True),
    ("global_training_frequency", True),
    ("selection_rank", False),
    ("full_model_visible", True),
)

_FEATURE_CONTRACT_ITEMS: tuple[tuple[str, str | tuple[str, ...]], ...] = (
    (
        "catalog_candidate_channels",
        (
            "literal_count",
            "operator",
            "placard_inclusion",
            "atom_family",
            "prevalence_bin",
        ),
    ),
    (
        "meta_surface_channels",
        (
            "renderer",
            "token_length_bin",
            "global_training_frequency",
        ),
    ),
    (
        "executor_only_diagnostic_channels",
        (
            "schedule_position",
            "request_position",
            "bank_prefix",
        ),
    ),
    (
        "terminal_model_visible_channels",
        (
            "catalog_candidate_channels",
            "global_training_frequency",
            "visible_opening_scene_attributes_and_labels",
            "renderer",
            "one_unlabeled_terminal_scene_attributes",
            "candidate_truth_on_that_scene",
            "rendered_length_bins",
        ),
    ),
    ("terminal_evaluator_only_diagnostic", ("selection_rank",)),
    (
        "forbidden_channels",
        (
            "model_output",
            "hidden_official_role",
            "factorial_cell",
            "request_id",
            "episode_id",
            "rule_id",
            "truth_digest",
            "binding_digest",
            "panel_id",
            "other_terminal_scenes",
            "other_terminal_answers",
        ),
    ),
    (
        "derivation",
        (
            "candidate channels are recomputed from canonical catalog ASTs, exact truth vectors, "
            "and block-inclusion frequencies recomputed from the complete audited population; "
            "hypothesis completion makes block inclusion equal Official exposure, and caller rule "
            "labels are never features"
        ),
    ),
    (
        "surface_input_boundary",
        (
            "rotation_surfaces are structural audit inputs and must be independently rederived and "
            "exact-compared with the future bank manifest before launch"
        ),
    ),
    (
        "exact_conditional_balance",
        (
            "auditor-only cells bind exact opening content, exact candidate identity, and every "
            "model-visible static surface; terminal cells additionally bind the isolated scene "
            "and rank; these identities are never classifier inputs"
        ),
    ),
)


def _feature_contract_obj() -> dict[str, Any]:
    """Return fresh JSON containers from the immutable canonical contract."""

    return {key: list(value) if isinstance(value, tuple) else value for key, value in _FEATURE_CONTRACT_ITEMS}


class StatisticalLeakageV2Error(ValueError):
    """Raised when statistical-leakage evidence is invalid or noncanonical."""


def _dump_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise StatisticalLeakageV2Error(f"value is not canonical JSON: {exc}") from exc


def _load_json(text: str) -> Any:
    if type(text) is not str or not text:
        raise StatisticalLeakageV2Error("JSON input must be a nonempty string")

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise StatisticalLeakageV2Error(f"duplicate JSON object key: {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise StatisticalLeakageV2Error(f"non-finite JSON constant is forbidden: {value}")

    try:
        return json.loads(text, object_pairs_hook=no_duplicates, parse_constant=reject_constant)
    except StatisticalLeakageV2Error:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise StatisticalLeakageV2Error(f"invalid JSON: {exc}") from exc


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


def _require_sha256(value: object, *, name: str) -> str:
    if not _is_sha256(value):
        raise StatisticalLeakageV2Error(f"{name} must be a lowercase SHA-256")
    return cast(str, value)


def _require_integer(
    value: object,
    *,
    name: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise StatisticalLeakageV2Error(f"{name} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise StatisticalLeakageV2Error(f"{name} must be an integer <= {maximum}")
    return value


def _require_signed_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise StatisticalLeakageV2Error(f"{name} must be an integer")
    return value


def _require_float(value: object, *, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or type(value) not in (int, float):
        raise StatisticalLeakageV2Error(f"{name} must be numeric")
    result = float(cast(float | int, value))
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise StatisticalLeakageV2Error(f"{name} must lie in [{minimum}, {maximum}]")
    return result


def _require_mapping(value: object, fields: tuple[str, ...], *, name: str) -> Mapping[str, Any]:
    if type(value) is not dict or tuple(value) != fields:
        raise StatisticalLeakageV2Error(f"{name} has noncanonical or reordered fields")
    return cast(Mapping[str, Any], value)


def _require_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise StatisticalLeakageV2Error(f"{name} must be Boolean")
    return value


def _scene_index(value: object, *, name: str) -> int:
    return _require_integer(value, name=name, maximum=SCENE_COUNT - 1)


def _rule_index(rule_id: str) -> int:
    if len(rule_id) != 9 or not rule_id.startswith("g03r") or not rule_id[4:].isdigit():
        raise StatisticalLeakageV2Error("bound rule id is malformed")
    index = int(rule_id[4:])
    catalog = build_rule_catalog()
    if not 0 <= index < len(catalog) or catalog[index].rule_id != rule_id:
        raise StatisticalLeakageV2Error("bound rule id is outside the canonical catalog")
    return index


def historical_three_role_full_v0_excess_lower_bound_v2(live_rule_count: int) -> Fraction:
    """Return the placard-oracle excess forced by the historical P/Q/C block.

    The old block makes its unique fitting placard candidate Official once in
    three episodes.  A placard-family oracle therefore scores ``1/3`` against
    the registered full-space chance ``1/|V0|``.  This function is an explicit
    infeasibility witness, not a replacement baseline.
    """

    _require_integer(
        live_rule_count,
        name="live_rule_count",
        minimum=MIN_LIVE_RULES,
        maximum=MAX_LIVE_RULES,
    )
    return Fraction(1, 3) - Fraction(1, live_rule_count)


@dataclass(frozen=True, slots=True)
class StatisticalLeakageConfigV2:
    """Frozen numerical settings; only test-time bootstrap replication varies."""

    fold_count: int = STATISTICAL_LEAKAGE_FOLDS
    bootstrap_replicates: int = STATISTICAL_LEAKAGE_BOOTSTRAP_REPLICATES

    def __post_init__(self) -> None:
        if isinstance(self.fold_count, bool) or self.fold_count != STATISTICAL_LEAKAGE_FOLDS:
            raise StatisticalLeakageV2Error("the preregistered audit requires exactly five folds")
        if isinstance(self.bootstrap_replicates, bool) or not isinstance(self.bootstrap_replicates, int):
            raise StatisticalLeakageV2Error("bootstrap_replicates must be an integer")
        if self.bootstrap_replicates < 1_000:
            raise StatisticalLeakageV2Error("bootstrap_replicates must be at least 1000")

    def as_obj(self) -> dict[str, int]:
        return {
            "fold_count": self.fold_count,
            "bootstrap_replicates": self.bootstrap_replicates,
            "minimum_construction_clusters": STATISTICAL_LEAKAGE_MINIMUM_GROUPS,
        }


DEFAULT_STATISTICAL_LEAKAGE_CONFIG_V2 = StatisticalLeakageConfigV2()


@dataclass(frozen=True, slots=True)
class CatalogFrequencyTableV2:
    """Training-manifest occurrence counts aligned to the exact v2 allowlist."""

    counts: tuple[int, ...]

    def __post_init__(self) -> None:
        contract = build_supported_catalog_contract_v2()
        if type(self.counts) is not tuple or len(self.counts) != len(contract.supported_indices):
            raise StatisticalLeakageV2Error(
                "catalog frequency counts must align exactly to the supported-index allowlist"
            )
        for count in self.counts:
            _require_integer(count, name="catalog frequency")

    @property
    def catalog_digest(self) -> str:
        return build_supported_catalog_contract_v2().source_catalog_digest

    @property
    def supported_catalog_digest(self) -> str:
        return build_supported_catalog_contract_v2().supported_catalog_digest

    @property
    def digest(self) -> str:
        return _json_digest(
            {
                "catalog_digest": self.catalog_digest,
                "supported_catalog_digest": self.supported_catalog_digest,
                "counts": list(self.counts),
            },
            domain=_FREQUENCY_DOMAIN,
        )

    def lookup(self, catalog_index: int) -> int:
        contract = build_supported_catalog_contract_v2()
        try:
            position = _supported_position_map()[catalog_index]
        except KeyError as exc:
            raise StatisticalLeakageV2Error("frequency lookup requested an unsupported rule") from exc
        if contract.supported_indices[position] != catalog_index:  # pragma: no cover - invariant
            raise RuntimeError("supported-index position cache is inconsistent")
        return self.counts[position]


def build_catalog_frequency_table_v2(
    counts: Mapping[int, int] | Sequence[int],
) -> CatalogFrequencyTableV2:
    """Build a frequency table without accepting caller rule labels as features."""

    contract = build_supported_catalog_contract_v2()
    if isinstance(counts, Mapping):
        if any(isinstance(key, bool) or not isinstance(key, int) for key in counts):
            raise StatisticalLeakageV2Error("frequency-table keys must be catalog indices")
        if set(counts) != set(contract.supported_indices) or len(counts) != len(contract.supported_indices):
            raise StatisticalLeakageV2Error("frequency mapping must cover the exact v2 allowlist")
        aligned = tuple(counts[index] for index in contract.supported_indices)
    else:
        aligned = tuple(counts)
    return CatalogFrequencyTableV2(aligned)


_SUPPORTED_POSITION_CACHE: dict[int, int] | None = None


def _supported_position_map() -> dict[int, int]:
    global _SUPPORTED_POSITION_CACHE
    if _SUPPORTED_POSITION_CACHE is None:
        _SUPPORTED_POSITION_CACHE = {
            index: position
            for position, index in enumerate(build_supported_catalog_contract_v2().supported_indices)
        }
    return _SUPPORTED_POSITION_CACHE


@dataclass(frozen=True, slots=True)
class MetaSurfaceFieldsV2:
    """One rotation's visible surface plus executor-only scheduling diagnostics.

    ``renderer`` and ``token_length_bin`` describe model-visible static input.
    The positions and bank prefix are executor metadata: they are audited for
    bounded canonical structure but are not represented as model-visible
    classifier channels.  A future manifest verifier must rederive every field.
    """

    renderer: RendererName
    schedule_position: int
    request_position: int
    token_length_bin: int
    bank_prefix: int

    def __post_init__(self) -> None:
        if type(self.renderer) is not str or self.renderer not in RENDERERS:
            raise StatisticalLeakageV2Error("surface renderer is not a registered renderer")
        _require_integer(self.schedule_position, name="schedule_position")
        _require_integer(self.request_position, name="request_position")
        _require_integer(self.token_length_bin, name="token_length_bin")
        _require_integer(self.bank_prefix, name="bank_prefix")

    def as_obj(self) -> dict[str, Any]:
        return {
            "renderer": self.renderer,
            "schedule_position": self.schedule_position,
            "request_position": self.request_position,
            "token_length_bin": self.token_length_bin,
            "bank_prefix": self.bank_prefix,
        }

    def model_visible_obj(self) -> dict[str, Any]:
        return {
            "renderer": self.renderer,
            "token_length_bin": self.token_length_bin,
        }

    def executor_only_obj(self) -> dict[str, int]:
        return {
            "schedule_position": self.schedule_position,
            "request_position": self.request_position,
            "bank_prefix": self.bank_prefix,
        }


def _validate_space(space: VersionSpace, binding: RuleTripleBindingV2) -> tuple[int, ...]:
    if type(space) is not VersionSpace:
        raise TypeError("version_space must be a VersionSpace")
    if type(binding) is not RuleTripleBindingV2:
        raise TypeError("binding must be a RuleTripleBindingV2")
    contract = build_supported_catalog_contract_v2()
    if space.catalog is not build_rule_catalog():
        raise StatisticalLeakageV2Error("version space uses the wrong catalog")
    if binding.catalog_digest != contract.source_catalog_digest:
        raise StatisticalLeakageV2Error("bound triple uses the wrong catalog")
    if binding.supported_catalog_digest != contract.supported_catalog_digest:
        raise StatisticalLeakageV2Error("bound triple uses the wrong supported allowlist")
    if not MIN_LIVE_RULES <= len(space) <= MAX_LIVE_RULES:
        raise StatisticalLeakageV2Error(
            f"hypothesis-complete audits require {MIN_LIVE_RULES} <= |V0| <= {MAX_LIVE_RULES}"
        )
    supported = set(contract.supported_indices)
    if any(index not in supported for index in space.indices):
        raise StatisticalLeakageV2Error("version space contains an unsupported v2 rule")
    designated = (
        _rule_index(binding.placard_rule_id),
        _rule_index(binding.literal_rule_id),
        _rule_index(binding.composed_rule_id),
    )
    if not set(designated).issubset(space.indices):
        raise StatisticalLeakageV2Error("bound P/Q/C candidates must all be live in V0")
    return designated


@dataclass(frozen=True, slots=True)
class HypothesisCompleteMetaBlockV2:
    """One fixed V0 with every candidate Official exactly once.

    ``rotation_surfaces[i]`` belongs to the episode whose Official is
    ``version_space.indices[i]``.  Canonical index alignment removes the need
    for caller-supplied target labels.
    """

    binding: RuleTripleBindingV2
    version_space: VersionSpace
    opening: tuple[VisibleOpeningObservationV2, ...]
    rotation_surfaces: tuple[MetaSurfaceFieldsV2, ...]

    def __post_init__(self) -> None:
        _validate_space(self.version_space, self.binding)
        if type(self.opening) is not tuple or len(self.opening) != 10:
            raise StatisticalLeakageV2Error(
                "a hypothesis-complete block requires exactly ten visible opening observations"
            )
        if any(type(item) is not VisibleOpeningObservationV2 for item in self.opening):
            raise StatisticalLeakageV2Error("opening contains a foreign observation")
        opening_scenes = tuple(item.scene_index for item in self.opening)
        if len(opening_scenes) != len(set(opening_scenes)):
            raise StatisticalLeakageV2Error("opening scenes must be unique")
        if sum(item.accepted for item in self.opening) != 5:
            raise StatisticalLeakageV2Error(
                "the ten-row hypothesis-complete opening must contain exactly 5 accepted and 5 rejected"
            )
        unfiltered = self.version_space.catalog.version_space(
            (item.scene_index, item.accepted) for item in self.opening
        )
        supported = set(build_supported_catalog_contract_v2().supported_indices)
        exact_supported = tuple(index for index in unfiltered.indices if index in supported)
        if self.version_space.indices != exact_supported:
            raise StatisticalLeakageV2Error(
                "supplied V0 omits or adds candidates relative to exact opening recomputation"
            )
        if type(self.rotation_surfaces) is not tuple or len(self.rotation_surfaces) != len(
            self.version_space
        ):
            raise StatisticalLeakageV2Error(
                "a hypothesis-complete block requires one surface row per live Official rule"
            )
        if any(type(surface) is not MetaSurfaceFieldsV2 for surface in self.rotation_surfaces):
            raise StatisticalLeakageV2Error("rotation surfaces contain a foreign value")
        size = len(self.version_space)
        if tuple(sorted(surface.schedule_position for surface in self.rotation_surfaces)) != tuple(
            range(size)
        ):
            raise StatisticalLeakageV2Error(
                "schedule positions must be the exact bounded local permutation 0..|V0|-1"
            )
        if tuple(sorted(surface.request_position for surface in self.rotation_surfaces)) != tuple(
            range(size)
        ):
            raise StatisticalLeakageV2Error(
                "request positions must be the exact bounded local permutation 0..|V0|-1"
            )
        visible_surfaces = {
            _dump_json(surface.model_visible_obj()) for surface in self.rotation_surfaces
        }
        if len(visible_surfaces) != 1:
            raise StatisticalLeakageV2Error(
                "every Official rotation must have an identical model-visible static surface"
            )
        if len({surface.bank_prefix for surface in self.rotation_surfaces}) != 1:
            raise StatisticalLeakageV2Error(
                "bank prefix must be constant inside one atomic hypothesis-complete block"
            )
        renderer = self.rotation_surfaces[0].renderer
        if renderer not in TRAIN_RENDERERS:
            raise StatisticalLeakageV2Error(
                "hypothesis-complete training blocks require a registered training renderer"
            )
        expected_length_bin = derive_rendered_static_prompt_length_bin_v2(self.opening, renderer)
        if self.rotation_surfaces[0].token_length_bin != expected_length_bin:
            raise StatisticalLeakageV2Error(
                "token_length_bin differs from exact rendered-static-prompt rederivation"
            )

    @property
    def designated_indices(self) -> tuple[int, int, int]:
        return cast(tuple[int, int, int], _validate_space(self.version_space, self.binding))

    @property
    def block_opening_digest(self) -> str:
        """Semantic grouping identity, independent of the designated P/Q/C binding."""

        return _json_digest(
            {
                "catalog_digest": build_supported_catalog_contract_v2().source_catalog_digest,
                "supported_catalog_digest": (build_supported_catalog_contract_v2().supported_catalog_digest),
                "live_rule_bindings": [
                    {
                        "rule_id": self.version_space.catalog[index].rule_id,
                        "truth_digest": self.version_space.catalog[index].truth_digest,
                    }
                    for index in self.version_space.indices
                ],
                "canonical_semantic_opening": [
                    item.as_obj() for item in sorted(self.opening, key=lambda value: value.scene_index)
                ],
            },
            domain=_BLOCK_OPENING_DOMAIN,
        )

    @property
    def model_visible_prompt_digest(self) -> str:
        surface = self.rotation_surfaces[0]
        return rendered_static_prompt_digest_v2(self.opening, surface.renderer)

    def as_binding_obj(self) -> dict[str, Any]:
        return {
            "block_opening_digest": self.block_opening_digest,
            "model_visible_prompt_digest": self.model_visible_prompt_digest,
            "bound_triple_digest": self.binding.digest,
            "live_rule_bindings": [
                {
                    "rule_id": self.version_space.catalog[index].rule_id,
                    "truth_digest": self.version_space.catalog[index].truth_digest,
                }
                for index in self.version_space.indices
            ],
            "ordered_model_visible_opening": [item.as_obj() for item in self.opening],
            "rotation_surfaces": [surface.as_obj() for surface in self.rotation_surfaces],
        }


@dataclass(frozen=True, slots=True)
class VisibleOpeningObservationV2:
    """One model-visible opening scene and its Official feedback bit."""

    scene_index: int
    accepted: bool

    def __post_init__(self) -> None:
        _scene_index(self.scene_index, name="opening scene index")
        if type(self.accepted) is not bool:
            raise StatisticalLeakageV2Error("opening accepted bit must be Boolean")

    def as_obj(self) -> dict[str, Any]:
        return {"scene_index": self.scene_index, "accepted": self.accepted}


def _rendered_static_prompt_obj_v2(
    opening: tuple[VisibleOpeningObservationV2, ...],
    renderer: RendererName,
) -> dict[str, Any]:
    if type(opening) is not tuple or any(
        type(item) is not VisibleOpeningObservationV2 for item in opening
    ):
        raise StatisticalLeakageV2Error("rendered prompt requires an exact opening tuple")
    if type(renderer) is not str or renderer not in RENDERERS:
        raise StatisticalLeakageV2Error("rendered prompt uses an unregistered renderer")
    return {
        "renderer": renderer,
        "ordered_opening": [
            {
                "scene_index": item.scene_index,
                "accepted": item.accepted,
                "rendered_scene": render_scene(scene_at(item.scene_index), renderer),
            }
            for item in opening
        ],
    }


@cache
def rendered_static_prompt_digest_v2(
    opening: tuple[VisibleOpeningObservationV2, ...],
    renderer: RendererName,
) -> str:
    """Bind the exact rendered, ordered static opening visible to the model."""

    return _json_digest(
        _rendered_static_prompt_obj_v2(opening, renderer),
        domain=_PROMPT_SURFACE_DOMAIN,
    )


@cache
def derive_rendered_static_prompt_length_bin_v2(
    opening: tuple[VisibleOpeningObservationV2, ...],
    renderer: RendererName,
) -> int:
    """Rederive the registered canonical rendered-character length bin.

    This is deliberately named and derived here rather than accepting a
    tokenizer-dependent caller assertion.  A production tokenizer-token bin,
    if desired, requires its own pinned tokenizer contract and manifest audit.
    """

    return len(_dump_json(_rendered_static_prompt_obj_v2(opening, renderer))) // 16


@dataclass(frozen=True, slots=True)
class TerminalPanelDrawsV2:
    """One panel's ranked draws, aligned to canonical Official rotations."""

    panel_index: int
    scene_indices_by_official: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        _require_integer(self.panel_index, name="panel_index")
        if type(self.scene_indices_by_official) is not tuple or not self.scene_indices_by_official:
            raise StatisticalLeakageV2Error("terminal panel requires Official-aligned draws")
        item_counts = {len(items) for items in self.scene_indices_by_official}
        if len(item_counts) != 1 or next(iter(item_counts)) < 1:
            raise StatisticalLeakageV2Error(
                "every Official rotation must contribute the same positive number of panel draws"
            )
        for items in self.scene_indices_by_official:
            if type(items) is not tuple:
                raise StatisticalLeakageV2Error("terminal draws must be tuples")
            for scene_index in items:
                _scene_index(scene_index, name="terminal scene index")

    @property
    def official_oblivious_shared(self) -> bool:
        return len(set(self.scene_indices_by_official)) == 1

    @property
    def item_count(self) -> int:
        return len(self.scene_indices_by_official[0])

    def as_obj(self) -> dict[str, Any]:
        return {
            "panel_index": self.panel_index,
            "scene_indices_by_official": [list(items) for items in self.scene_indices_by_official],
        }


@dataclass(frozen=True, slots=True)
class HypothesisCompleteTerminalBlockV2:
    """A hypothesis-complete block with isolated one-item terminal draws."""

    meta_block: HypothesisCompleteMetaBlockV2
    panels: tuple[TerminalPanelDrawsV2, ...]

    def __post_init__(self) -> None:
        if type(self.meta_block) is not HypothesisCompleteMetaBlockV2:
            raise StatisticalLeakageV2Error("terminal block requires a hypothesis-complete meta block")
        if type(self.panels) is not tuple or not self.panels:
            raise StatisticalLeakageV2Error("terminal block requires at least one panel")
        if any(type(panel) is not TerminalPanelDrawsV2 for panel in self.panels):
            raise StatisticalLeakageV2Error("terminal block contains a foreign panel")
        if tuple(panel.panel_index for panel in self.panels) != tuple(
            sorted({panel.panel_index for panel in self.panels})
        ):
            raise StatisticalLeakageV2Error("terminal panels must be sorted and uniquely indexed")
        expected_rotations = len(self.meta_block.version_space)
        if any(len(panel.scene_indices_by_official) != expected_rotations for panel in self.panels):
            raise StatisticalLeakageV2Error(
                "each panel must align draws to every hypothesis-complete Official rotation"
            )

    @property
    def official_oblivious_shared(self) -> bool:
        return all(panel.official_oblivious_shared for panel in self.panels)

    def as_binding_obj(self) -> dict[str, Any]:
        return {
            "meta_block": self.meta_block.as_binding_obj(),
            "panels": [panel.as_obj() for panel in self.panels],
        }


@dataclass(frozen=True, slots=True)
class _Episode:
    group_id: str
    opening_digest: str
    official_index: int
    candidate_indices: tuple[int, ...]
    designated_indices: tuple[int, int, int]
    surface: MetaSurfaceFieldsV2
    opening: tuple[VisibleOpeningObservationV2, ...] = ()
    terminal_panel_index: int | None = None
    terminal_scene_index: int | None = None
    selection_rank: int | None = None


@cache
def _catalog_core_tokens_for_index(candidate_index: int) -> tuple[str, ...]:
    entry = build_rule_catalog()[candidate_index]
    rule = entry.rule
    literals: tuple[RuleLiteral, ...]
    if type(rule) is RuleLiteral:
        literals = (rule,)
        operator = "literal"
    elif type(rule) is BinaryRule:
        literals = rule.args
        operator = rule.op
    else:  # pragma: no cover - grammar exhaustiveness
        raise RuntimeError("unknown canonical rule AST")
    atom_family = "+".join(sorted(literal.atom.op for literal in literals))
    placard = any(literal.atom.op == "placard_is" for literal in literals)
    prevalence_bin = min(19, (entry.truth.true_count * 20) // SCENE_COUNT)
    return (
        f"literal_count={len(literals)}",
        f"operator={operator}",
        f"placard_inclusion={'true' if placard else 'false'}",
        f"atom_family={atom_family}",
        f"prevalence_bin={prevalence_bin}",
    )


def _scene_tokens(scene: Scene, *, prefix: str) -> tuple[str, ...]:
    fields: list[str] = [
        f"{prefix}.placard={scene.placard}",
        f"{prefix}.occupied_count={scene.occupied_count}",
    ]
    for position in POSITIONS:
        piece = scene.piece_at(position)
        if piece is None:
            fields.append(f"{prefix}.{position}.occupancy=empty")
        else:
            fields.extend(
                (
                    f"{prefix}.{position}.occupancy=occupied",
                    f"{prefix}.{position}.color={piece.color}",
                    f"{prefix}.{position}.shape={piece.shape}",
                    f"{prefix}.{position}.size={piece.size}",
                )
            )
    return tuple(fields)


@cache
def _scene_tokens_at(scene_index: int, prefix: str) -> tuple[str, ...]:
    return _scene_tokens(scene_at(scene_index), prefix=prefix)


@cache
def _rendered_scene_length_bin(scene_index: int, renderer: RendererName) -> int:
    return len(render_scene(scene_at(scene_index), renderer)) // 16


def _interacted(core: tuple[str, ...], extras: tuple[str, ...]) -> tuple[str, ...]:
    joint = "full_joint_cell_sha256=" + hashlib.sha256(
        _dump_json([*core, *extras]).encode("ascii")
    ).hexdigest()
    return (
        *core,
        *extras,
        *(f"interaction[{left}|{right}]" for left in core for right in extras),
        joint,
    )


@cache
def _candidate_extras_tokens_cached(
    candidate_index: int,
    extras: tuple[str, ...],
) -> tuple[str, ...]:
    """Reuse one exact feature vector across hypothesis-complete rotations."""

    return _interacted(_catalog_core_tokens_for_index(candidate_index), extras)


def _meta_feature_tokens(
    episode: _Episode,
    candidate_index: int,
    view_name: str,
    frequencies: CatalogFrequencyTableV2,
) -> tuple[str, ...]:
    surface = episode.surface
    values: dict[str, tuple[str, ...]] = {
        "catalog_structure": (),
        "renderer": (f"renderer={surface.renderer}",),
        "schedule_position": (f"schedule_position={surface.schedule_position}",),
        "request_position": (f"request_position={surface.request_position}",),
        "token_length_bin": (f"token_length_bin={surface.token_length_bin}",),
        "bank_prefix": (f"bank_prefix={surface.bank_prefix}",),
        "global_training_frequency": (f"global_training_frequency={frequencies.lookup(candidate_index)}",),
    }
    if view_name == "full_model_visible":
        extras = (*values["renderer"], *values["token_length_bin"], *values["global_training_frequency"])
    elif view_name == "full_executor_diagnostic":
        extras = tuple(
            token
            for name in (
                "schedule_position",
                "request_position",
                "bank_prefix",
            )
            for token in values[name]
        )
    else:
        try:
            extras = values[view_name]
        except KeyError as exc:  # pragma: no cover - internal contract
            raise RuntimeError(f"unknown meta feature view: {view_name}") from exc
    return _candidate_extras_tokens_cached(candidate_index, extras)


@cache
def _opening_tokens_cached(
    opening: tuple[VisibleOpeningObservationV2, ...],
    renderer: RendererName,
) -> tuple[str, ...]:
    result: list[str] = [f"opening.count={len(opening)}"]
    for position, observation in enumerate(opening):
        prefix = f"opening[{position}]"
        result.append(f"{prefix}.accepted={'true' if observation.accepted else 'false'}")
        result.extend(_scene_tokens_at(observation.scene_index, prefix))
        result.append(
            f"{prefix}.rendered_length_bin="
            f"{_rendered_scene_length_bin(observation.scene_index, renderer)}"
        )
    return tuple(result)


@cache
def _terminal_feature_tokens_cached(
    opening_observations: tuple[VisibleOpeningObservationV2, ...],
    renderer_name: RendererName,
    terminal_scene_index: int,
    selection_rank_value: int,
    candidate_index: int,
    view_name: str,
    frequency: int,
) -> tuple[str, ...]:
    catalog = build_rule_catalog()
    entry = catalog[candidate_index]
    renderer = (f"renderer={renderer_name}",)
    scene_features = _scene_tokens_at(terminal_scene_index, "terminal")
    scene_features = (
        *scene_features,
        f"terminal.rendered_length_bin="
        f"{_rendered_scene_length_bin(terminal_scene_index, renderer_name)}",
    )
    truth = (f"candidate_truth={'true' if entry.truth[terminal_scene_index] else 'false'}",)
    opening = _opening_tokens_cached(opening_observations, renderer_name)
    frequency_token = (f"global_training_frequency={frequency}",)
    selection_rank = (f"selection_rank={selection_rank_value}",)
    values: dict[str, tuple[str, ...]] = {
        "catalog_structure": (),
        "renderer": renderer,
        "scene_attributes": scene_features,
        "candidate_truth": truth,
        "opening": opening,
        "global_training_frequency": frequency_token,
        "selection_rank": selection_rank,
    }
    if view_name == "full_model_visible":
        extras = (*renderer, *scene_features, *truth, *opening, *frequency_token)
    else:
        try:
            extras = values[view_name]
        except KeyError as exc:  # pragma: no cover - internal contract
            raise RuntimeError(f"unknown terminal feature view: {view_name}") from exc
    return _candidate_extras_tokens_cached(candidate_index, extras)


def _terminal_feature_tokens(
    episode: _Episode,
    candidate_index: int,
    view_name: str,
    frequencies: CatalogFrequencyTableV2,
) -> tuple[str, ...]:
    if episode.terminal_scene_index is None or episode.selection_rank is None:
        raise RuntimeError("terminal feature request lacks one isolated scene")
    return _terminal_feature_tokens_cached(
        episode.opening,
        episode.surface.renderer,
        episode.terminal_scene_index,
        episode.selection_rank,
        candidate_index,
        view_name,
        frequencies.lookup(candidate_index),
    )


FeatureBuilder = Any
PreparedFeatureRows = tuple[tuple[_Episode, dict[int, tuple[str, ...]]], ...]


@dataclass(frozen=True, slots=True)
class ConstructionClusterV2:
    """A content-rederived conservative cluster of closely related openings."""

    construction_cluster_digest: str
    member_opening_digests: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_sha256(self.construction_cluster_digest, name="construction cluster digest")
        if type(self.member_opening_digests) is not tuple or not self.member_opening_digests:
            raise StatisticalLeakageV2Error("construction cluster must contain opening digests")
        if any(not _is_sha256(value) for value in self.member_opening_digests):
            raise StatisticalLeakageV2Error("construction cluster contains a malformed opening digest")
        if self.member_opening_digests != tuple(sorted(set(self.member_opening_digests))):
            raise StatisticalLeakageV2Error(
                "construction-cluster opening digests must be sorted and unique"
            )
        expected = _json_digest(
            {"member_opening_digests": list(self.member_opening_digests)},
            domain=_CONSTRUCTION_CLUSTER_DOMAIN,
        )
        if self.construction_cluster_digest != expected:
            raise StatisticalLeakageV2Error("construction cluster digest is inconsistent")

    def as_obj(self) -> dict[str, Any]:
        return {
            "construction_cluster_digest": self.construction_cluster_digest,
            "member_opening_digests": list(self.member_opening_digests),
        }


def _derive_construction_clusters(
    blocks: tuple[HypothesisCompleteMetaBlockV2, ...],
) -> tuple[dict[str, str], tuple[ConstructionClusterV2, ...]]:
    """Cluster openings sharing an exact nine-of-ten semantic projection.

    This is a conservative, content-derived lower bound on construction
    dependence.  It prevents one base opening plus 384 redundant-row variants
    from masquerading as 384 independent units.  It is not a substitute for
    future frozen-generator lineage rederivation, which remains unauthorized.
    """

    count = len(blocks)
    parent = list(range(count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    projection_owner: dict[tuple[tuple[int, bool], ...], int] = {}
    for block_position, block in enumerate(blocks):
        rows = tuple(sorted((item.scene_index, item.accepted) for item in block.opening))
        for omitted in range(len(rows)):
            projection = (*rows[:omitted], *rows[omitted + 1 :])
            owner = projection_owner.setdefault(projection, block_position)
            union(block_position, owner)

    members: dict[int, list[str]] = {}
    for position, block in enumerate(blocks):
        members.setdefault(find(position), []).append(block.block_opening_digest)

    clusters: list[ConstructionClusterV2] = []
    opening_to_cluster: dict[str, str] = {}
    for values in members.values():
        member_digests = tuple(sorted(values))
        digest = _json_digest(
            {"member_opening_digests": list(member_digests)},
            domain=_CONSTRUCTION_CLUSTER_DOMAIN,
        )
        cluster = ConstructionClusterV2(digest, member_digests)
        clusters.append(cluster)
        for opening_digest in member_digests:
            opening_to_cluster[opening_digest] = digest
    return opening_to_cluster, tuple(
        sorted(clusters, key=lambda value: value.construction_cluster_digest)
    )


def _stable_fold_assignments(group_ids: tuple[str, ...], fold_count: int) -> dict[str, int]:
    ordered = sorted(
        group_ids,
        key=lambda group_id: hashlib.sha256(_FOLD_DOMAIN + bytes.fromhex(group_id)).digest(),
    )
    return {group_id: position % fold_count for position, group_id in enumerate(ordered)}


def _fold_digest(assignments: Mapping[str, int]) -> str:
    return _json_digest(
        [
            {"construction_cluster_digest": group_id, "fold": assignments[group_id]}
            for group_id in sorted(assignments)
        ],
        domain=_FOLD_REPORT_DOMAIN,
    )


@dataclass(frozen=True, slots=True)
class PredictionPatternV2:
    """One pooled, exact per-candidate prediction pattern.

    Pooling keeps the full 384 x 16 x 11 audit representation practical while
    retaining every candidate score and prediction bit.  Episode evidence
    below names the pattern it uses and records its hidden Official target.
    """

    candidate_indices: tuple[int, ...]
    exact_scores: tuple[tuple[int, int], ...]
    predicted_official: tuple[bool, ...]
    winner_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.candidate_indices) is not tuple or not self.candidate_indices:
            raise StatisticalLeakageV2Error("prediction pattern requires candidate indices")
        if self.candidate_indices != tuple(sorted(set(self.candidate_indices))):
            raise StatisticalLeakageV2Error(
                "prediction-pattern candidate indices must be sorted and unique"
            )
        supported = set(build_supported_catalog_contract_v2().supported_indices)
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index not in supported
            for index in self.candidate_indices
        ):
            raise StatisticalLeakageV2Error(
                "prediction pattern contains an unsupported candidate index"
            )
        if type(self.exact_scores) is not tuple or len(self.exact_scores) != len(
            self.candidate_indices
        ):
            raise StatisticalLeakageV2Error(
                "prediction-pattern exact scores do not align to candidates"
            )
        fractions: list[Fraction] = []
        for score in self.exact_scores:
            if type(score) is not tuple or len(score) != 2:
                raise StatisticalLeakageV2Error("an exact prediction score is malformed")
            numerator = _require_signed_integer(score[0], name="score numerator")
            denominator = _require_integer(score[1], name="score denominator", minimum=1)
            exact = Fraction(numerator, denominator)
            if (exact.numerator, exact.denominator) != score:
                raise StatisticalLeakageV2Error(
                    "exact prediction scores must use reduced positive-denominator form"
                )
            fractions.append(exact)
        if type(self.predicted_official) is not tuple or len(self.predicted_official) != len(
            self.candidate_indices
        ):
            raise StatisticalLeakageV2Error(
                "prediction bits do not align to pattern candidates"
            )
        if any(type(value) is not bool for value in self.predicted_official):
            raise StatisticalLeakageV2Error("prediction bits must be Boolean")
        expected_predictions = tuple(score > 0 for score in fractions)
        if self.predicted_official != expected_predictions:
            raise StatisticalLeakageV2Error(
                "per-candidate prediction bits differ from the exact zero threshold"
            )
        maximum = max(fractions)
        expected_winners = tuple(
            index
            for index, score in zip(self.candidate_indices, fractions, strict=True)
            if score == maximum
        )
        if self.winner_indices != expected_winners:
            raise StatisticalLeakageV2Error(
                "winner indices differ from the complete exact maximum-score tie"
            )

    @property
    def digest(self) -> str:
        return _json_digest(self.as_obj(), domain=_PREDICTION_PATTERN_DOMAIN)

    def as_obj(self) -> dict[str, Any]:
        return {
            "candidate_indices": list(self.candidate_indices),
            "exact_scores": [list(score) for score in self.exact_scores],
            "predicted_official": list(self.predicted_official),
            "winner_indices": list(self.winner_indices),
        }


@dataclass(frozen=True, slots=True)
class PredictionEpisodeEvidenceV2:
    """One target-bearing episode row referencing an exact score pattern."""

    construction_cluster_digest: str
    block_opening_digest: str
    fold: int
    official_index: int
    terminal_panel_index: int | None
    terminal_scene_index: int | None
    selection_rank: int | None
    pattern_index: int
    credit: tuple[int, int]

    def __post_init__(self) -> None:
        _require_sha256(self.construction_cluster_digest, name="evidence construction cluster")
        _require_sha256(self.block_opening_digest, name="evidence block opening")
        _require_integer(self.fold, name="evidence fold", maximum=STATISTICAL_LEAKAGE_FOLDS - 1)
        supported = set(build_supported_catalog_contract_v2().supported_indices)
        if (
            isinstance(self.official_index, bool)
            or not isinstance(self.official_index, int)
            or self.official_index not in supported
        ):
            raise StatisticalLeakageV2Error("prediction evidence has an unsupported Official")
        terminal_fields = (
            self.terminal_panel_index,
            self.terminal_scene_index,
            self.selection_rank,
        )
        if any(value is None for value in terminal_fields) and not all(
            value is None for value in terminal_fields
        ):
            raise StatisticalLeakageV2Error(
                "terminal prediction identity fields must be all present or all absent"
            )
        if self.terminal_panel_index is not None:
            _require_integer(self.terminal_panel_index, name="evidence terminal panel")
            _scene_index(self.terminal_scene_index, name="evidence terminal scene")
            _require_integer(self.selection_rank, name="evidence selection rank")
        _require_integer(self.pattern_index, name="prediction pattern index")
        if type(self.credit) is not tuple or len(self.credit) != 2:
            raise StatisticalLeakageV2Error("prediction credit is malformed")
        numerator = _require_integer(self.credit[0], name="credit numerator")
        denominator = _require_integer(self.credit[1], name="credit denominator", minimum=1)
        exact = Fraction(numerator, denominator)
        if (exact.numerator, exact.denominator) != self.credit or not Fraction() <= exact <= 1:
            raise StatisticalLeakageV2Error(
                "prediction credit must be a reduced rational in [0,1]"
            )

    @property
    def canonical_key(self) -> tuple[Any, ...]:
        return (
            self.block_opening_digest,
            -1 if self.terminal_panel_index is None else self.terminal_panel_index,
            self.official_index,
            -1 if self.selection_rank is None else self.selection_rank,
            -1 if self.terminal_scene_index is None else self.terminal_scene_index,
            self.construction_cluster_digest,
        )

    def as_obj(self) -> dict[str, Any]:
        return {
            "construction_cluster_digest": self.construction_cluster_digest,
            "block_opening_digest": self.block_opening_digest,
            "fold": self.fold,
            "official_index": self.official_index,
            "terminal_panel_index": self.terminal_panel_index,
            "terminal_scene_index": self.terminal_scene_index,
            "selection_rank": self.selection_rank,
            "pattern_index": self.pattern_index,
            "credit": list(self.credit),
        }


@dataclass(frozen=True, slots=True)
class PredictionEvidenceV2:
    """Canonical exact prediction evidence, with score patterns pooled losslessly."""

    target_name: str
    status: Literal["estimated", "not_estimable"]
    nominal_observation_count: int
    nominal_candidate_row_count: int
    patterns: tuple[PredictionPatternV2, ...]
    episodes: tuple[PredictionEpisodeEvidenceV2, ...]

    def __post_init__(self) -> None:
        if self.target_name not in {"full_v0_official", "designated_p_q_c_official"}:
            raise StatisticalLeakageV2Error("prediction evidence has an unknown target")
        if self.status not in {"estimated", "not_estimable"}:
            raise StatisticalLeakageV2Error("prediction evidence has an unknown status")
        _require_integer(
            self.nominal_observation_count,
            name="prediction nominal observation count",
            minimum=1,
        )
        _require_integer(
            self.nominal_candidate_row_count,
            name="prediction nominal candidate-row count",
            minimum=1,
        )
        if type(self.patterns) is not tuple or any(
            type(pattern) is not PredictionPatternV2 for pattern in self.patterns
        ):
            raise StatisticalLeakageV2Error("prediction patterns must be an immutable exact tuple")
        if type(self.episodes) is not tuple or any(
            type(episode) is not PredictionEpisodeEvidenceV2 for episode in self.episodes
        ):
            raise StatisticalLeakageV2Error("prediction episodes must be an immutable exact tuple")
        if self.status == "not_estimable":
            if self.patterns or self.episodes:
                raise StatisticalLeakageV2Error(
                    "not-estimable prediction evidence cannot contain fabricated scores"
                )
            return
        if not self.patterns or not self.episodes:
            raise StatisticalLeakageV2Error("estimated prediction evidence cannot be empty")
        pattern_digests = tuple(pattern.digest for pattern in self.patterns)
        if pattern_digests != tuple(sorted(set(pattern_digests))):
            raise StatisticalLeakageV2Error(
                "prediction patterns must be digest-sorted and unique"
            )
        if self.episodes != tuple(sorted(self.episodes, key=lambda row: row.canonical_key)):
            raise StatisticalLeakageV2Error("prediction episodes must be canonically sorted")
        if len({row.canonical_key for row in self.episodes}) != len(self.episodes):
            raise StatisticalLeakageV2Error("prediction episode identities must be unique")
        if len(self.episodes) != self.nominal_observation_count:
            raise StatisticalLeakageV2Error("prediction evidence observation count is inconsistent")
        candidate_rows = 0
        for row in self.episodes:
            if row.pattern_index >= len(self.patterns):
                raise StatisticalLeakageV2Error("prediction episode references an absent pattern")
            pattern = self.patterns[row.pattern_index]
            candidate_rows += len(pattern.candidate_indices)
            if row.official_index not in pattern.candidate_indices:
                raise StatisticalLeakageV2Error(
                    "prediction evidence Official is absent from its target candidate set"
                )
            expected_credit = (
                Fraction(1, len(pattern.winner_indices))
                if row.official_index in pattern.winner_indices
                else Fraction()
            )
            if row.credit != (expected_credit.numerator, expected_credit.denominator):
                raise StatisticalLeakageV2Error(
                    "prediction credit differs from the exact winner/Official relation"
                )
            expected_count = 3 if self.target_name == "designated_p_q_c_official" else None
            if expected_count is not None and len(pattern.candidate_indices) != expected_count:
                raise StatisticalLeakageV2Error(
                    "designated P/Q/C prediction evidence must contain exactly three candidates"
                )
        if candidate_rows != self.nominal_candidate_row_count:
            raise StatisticalLeakageV2Error("prediction evidence candidate-row count is inconsistent")

    @property
    def digest(self) -> str:
        return _json_digest(self._unsigned_obj(), domain=_PREDICTION_DOMAIN)

    def _unsigned_obj(self) -> dict[str, Any]:
        return {
            "target_name": self.target_name,
            "status": self.status,
            "nominal_observation_count": self.nominal_observation_count,
            "nominal_candidate_row_count": self.nominal_candidate_row_count,
            "patterns": [pattern.as_obj() for pattern in self.patterns],
            "episodes": [episode.as_obj() for episode in self.episodes],
        }

    def as_obj(self) -> dict[str, Any]:
        return {**self._unsigned_obj(), "prediction_evidence_digest": self.digest}

    def top_one_observations(self) -> tuple[tuple[Fraction, Fraction, str], ...]:
        if self.status != "estimated":
            raise StatisticalLeakageV2Error("not-estimable evidence has no top-one observations")
        return tuple(
            (
                Fraction(*row.credit),
                Fraction(1, len(self.patterns[row.pattern_index].candidate_indices)),
                row.construction_cluster_digest,
            )
            for row in self.episodes
        )

    def balanced_accuracy_cells(self) -> tuple[tuple[int, int, int, int, str], ...]:
        if self.status != "estimated":
            raise StatisticalLeakageV2Error("not-estimable evidence has no BA cells")
        result: list[tuple[int, int, int, int, str]] = []
        for row in self.episodes:
            pattern = self.patterns[row.pattern_index]
            tp = fn = tn = fp = 0
            for candidate_index, predicted in zip(
                pattern.candidate_indices,
                pattern.predicted_official,
                strict=True,
            ):
                positive = candidate_index == row.official_index
                tp += int(positive and predicted)
                fn += int(positive and not predicted)
                fp += int(not positive and predicted)
                tn += int(not positive and not predicted)
            result.append((tp, fn, tn, fp, row.construction_cluster_digest))
        return tuple(result)


def _prepare_feature_rows(
    episodes: tuple[_Episode, ...],
    feature_builder: FeatureBuilder,
) -> PreparedFeatureRows:
    """Encode every episode/candidate feature vector once for one view."""

    return tuple(
        (
            episode,
            {
                candidate_index: tuple(feature_builder(episode, candidate_index))
                for candidate_index in episode.candidate_indices
            },
        )
        for episode in episodes
    )


def _score_rows(
    prepared_feature_rows: PreparedFeatureRows,
    *,
    target_name: str,
    assignments: Mapping[str, int],
    candidate_selector: Any,
) -> tuple[
    tuple[tuple[Fraction, Fraction, str], ...],
    tuple[tuple[int, int, int, int, str], ...],
    PredictionEvidenceV2,
]:
    """Return top-one observations, BA cells, and a deterministic digest."""

    fold_count = STATISTICAL_LEAKAGE_FOLDS
    total_counts: Counter[str] = Counter()
    positive_counts: Counter[str] = Counter()
    fold_totals = [Counter[str]() for _ in range(fold_count)]
    fold_positives = [Counter[str]() for _ in range(fold_count)]
    total_rows = 0
    total_positives = 0
    fold_rows = [0] * fold_count
    fold_positive_rows = [0] * fold_count

    prepared_rows: list[tuple[_Episode, tuple[int, ...], dict[int, tuple[str, ...]]]] = []
    for episode, feature_rows in prepared_feature_rows:
        candidates = tuple(sorted(candidate_selector(episode)))
        if episode.official_index not in candidates:
            continue
        prepared_rows.append((episode, candidates, feature_rows))
        fold = assignments[episode.group_id]
        for candidate_index in candidates:
            features = feature_rows[candidate_index]
            if len(features) != len(set(features)):
                raise RuntimeError("statistical feature builder emitted a duplicate token")
            positive = candidate_index == episode.official_index
            total_counts.update(features)
            fold_totals[fold].update(features)
            total_rows += 1
            fold_rows[fold] += 1
            if positive:
                positive_counts.update(features)
                fold_positives[fold].update(features)
                total_positives += 1
                fold_positive_rows[fold] += 1

    if not prepared_rows:
        raise StatisticalLeakageV2Error("statistical target contains no eligible episodes")

    top_one: list[tuple[Fraction, Fraction, str]] = []
    ba_cells: list[tuple[int, int, int, int, str]] = []
    raw_evidence: list[tuple[_Episode, PredictionPatternV2, Fraction]] = []
    for episode, candidates, feature_rows in prepared_rows:
        fold = assignments[episode.group_id]
        training_rows = total_rows - fold_rows[fold]
        training_positives = total_positives - fold_positive_rows[fold]
        if training_rows <= 0 or not 0 < training_positives < training_rows:
            raise StatisticalLeakageV2Error("a grouped fold lacks both one-v-rest classes")
        if training_rows % training_positives:
            raise StatisticalLeakageV2Error(
                "hypothesis-complete training folds must have reciprocal-integer prevalence"
            )
        prevalence_denominator = training_rows // training_positives
        scores: dict[int, Fraction] = {}
        row_predictions: dict[int, bool] = {}
        for candidate_index in candidates:
            score = Fraction()
            for feature in feature_rows[candidate_index]:
                count = total_counts[feature] - fold_totals[fold][feature]
                positive_count = positive_counts[feature] - fold_positives[fold][feature]
                if count:
                    centered_numerator = prevalence_denominator * positive_count - count
                    if centered_numerator:
                        score += Fraction(centered_numerator, count + 1)
            scores[candidate_index] = score
            row_predictions[candidate_index] = score > 0
        maximum = max(scores.values())
        winners = tuple(index for index in candidates if scores[index] == maximum)
        credit = Fraction(1, len(winners)) if episode.official_index in winners else Fraction()
        chance = Fraction(1, len(candidates))
        top_one.append((credit, chance, episode.group_id))
        tp = fn = tn = fp = 0
        for candidate_index in candidates:
            positive = candidate_index == episode.official_index
            predicted = row_predictions[candidate_index]
            tp += int(positive and predicted)
            fn += int(positive and not predicted)
            fp += int(not positive and predicted)
            tn += int(not positive and not predicted)
        ba_cells.append((tp, fn, tn, fp, episode.group_id))
        raw_evidence.append(
            (
                episode,
                PredictionPatternV2(
                    candidates,
                    tuple(
                        (scores[index].numerator, scores[index].denominator)
                        for index in candidates
                    ),
                    tuple(row_predictions[index] for index in candidates),
                    winners,
                ),
                credit,
            )
        )
    patterns = tuple(
        sorted(
            set(pattern for _, pattern, _ in raw_evidence),
            key=lambda pattern: pattern.digest,
        )
    )
    pattern_positions = {pattern: position for position, pattern in enumerate(patterns)}
    evidence_rows = tuple(
        sorted(
            (
                PredictionEpisodeEvidenceV2(
                    episode.group_id,
                    episode.opening_digest,
                    assignments[episode.group_id],
                    episode.official_index,
                    episode.terminal_panel_index,
                    episode.terminal_scene_index,
                    episode.selection_rank,
                    pattern_positions[pattern],
                    (credit.numerator, credit.denominator),
                )
                for episode, pattern, credit in raw_evidence
            ),
            key=lambda row: row.canonical_key,
        )
    )
    evidence = PredictionEvidenceV2(
        target_name,
        "estimated",
        len(evidence_rows),
        sum(len(patterns[row.pattern_index].candidate_indices) for row in evidence_rows),
        patterns,
        evidence_rows,
    )
    return (
        tuple(top_one),
        tuple(ba_cells),
        evidence,
    )


def _bootstrap_seed(dataset_digest: str, view_name: str, target_name: str) -> int:
    digest = hashlib.sha256(
        _BOOTSTRAP_SEED_DOMAIN
        + bytes.fromhex(dataset_digest)
        + b"\0"
        + view_name.encode("ascii")
        + b"\0"
        + target_name.encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _order_stat_interval(values: np.ndarray) -> tuple[float, float]:
    ordered = np.sort(values)
    count = len(ordered)
    lower_index = math.floor((count - 1) * 0.025)
    upper_index = math.ceil((count - 1) * 0.975)
    return float(ordered[lower_index]), float(ordered[upper_index])


def _bootstrap_top_one(
    observations: tuple[tuple[Fraction, Fraction, str], ...],
    *,
    group_ids: tuple[str, ...],
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    group_position = {group_id: position for position, group_id in enumerate(group_ids)}
    credit = np.zeros(len(group_ids), dtype=np.float64)
    chance = np.zeros(len(group_ids), dtype=np.float64)
    counts = np.zeros(len(group_ids), dtype=np.float64)
    for observed_credit, observed_chance, group_id in observations:
        position = group_position[group_id]
        credit[position] += float(observed_credit)
        chance[position] += float(observed_chance)
        counts[position] += 1.0
    rng = np.random.default_rng(seed)
    results = np.empty(replicates, dtype=np.float64)
    chunk_size = 256
    for start in range(0, replicates, chunk_size):
        stop = min(replicates, start + chunk_size)
        sampled = rng.integers(0, len(group_ids), size=(stop - start, len(group_ids)))
        denominator = counts[sampled].sum(axis=1)
        results[start:stop] = (credit[sampled].sum(axis=1) - chance[sampled].sum(axis=1)) / denominator
    return _order_stat_interval(results)


def _bootstrap_ba(
    cells: tuple[tuple[int, int, int, int, str], ...],
    *,
    group_ids: tuple[str, ...],
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    group_position = {group_id: position for position, group_id in enumerate(group_ids)}
    values = np.zeros((len(group_ids), 4), dtype=np.float64)
    for tp, fn, tn, fp, group_id in cells:
        values[group_position[group_id]] += (tp, fn, tn, fp)
    rng = np.random.default_rng(seed)
    results = np.empty(replicates, dtype=np.float64)
    chunk_size = 256
    for start in range(0, replicates, chunk_size):
        stop = min(replicates, start + chunk_size)
        sampled = rng.integers(0, len(group_ids), size=(stop - start, len(group_ids)))
        sums = values[sampled].sum(axis=1)
        positive_denominator = sums[:, 0] + sums[:, 1]
        negative_denominator = sums[:, 2] + sums[:, 3]
        results[start:stop] = 0.5 * (sums[:, 0] / positive_denominator + sums[:, 2] / negative_denominator)
    return _order_stat_interval(results)


@dataclass(frozen=True, slots=True)
class ExactConditionalBalanceResultV2:
    """Exact Official-independence over auditor-only visible-input cells."""

    candidate_row_count: int
    conditional_cell_count: int
    violating_cell_count: int
    maximum_absolute_excess_numerator: int
    maximum_absolute_excess_denominator: int
    conditional_cell_digest: str

    def __post_init__(self) -> None:
        _require_integer(self.candidate_row_count, name="conditional candidate_row_count", minimum=1)
        _require_integer(self.conditional_cell_count, name="conditional_cell_count", minimum=1)
        _require_integer(self.violating_cell_count, name="violating_cell_count")
        if self.violating_cell_count > self.conditional_cell_count:
            raise StatisticalLeakageV2Error("too many violating conditional cells")
        _require_integer(
            self.maximum_absolute_excess_numerator,
            name="maximum_absolute_excess_numerator",
        )
        _require_integer(
            self.maximum_absolute_excess_denominator,
            name="maximum_absolute_excess_denominator",
            minimum=1,
        )
        excess = Fraction(
            self.maximum_absolute_excess_numerator,
            self.maximum_absolute_excess_denominator,
        )
        if not Fraction() <= excess <= Fraction(1):
            raise StatisticalLeakageV2Error("conditional-cell excess is outside [0,1]")
        if (self.violating_cell_count == 0) is not (excess == 0):
            raise StatisticalLeakageV2Error("conditional-cell violations and maximum excess disagree")
        _require_sha256(self.conditional_cell_digest, name="conditional cell digest")

    @property
    def passed(self) -> bool:
        return self.violating_cell_count == 0

    def as_obj(self) -> dict[str, Any]:
        return {
            "candidate_row_count": self.candidate_row_count,
            "conditional_cell_count": self.conditional_cell_count,
            "violating_cell_count": self.violating_cell_count,
            "maximum_absolute_excess": [
                self.maximum_absolute_excess_numerator,
                self.maximum_absolute_excess_denominator,
            ],
            "conditional_cell_digest": self.conditional_cell_digest,
            "passed": self.passed,
        }


def _exact_conditional_balance(
    audit_kind: AuditKindV2,
    episodes: tuple[_Episode, ...],
    *,
    fixed_size: int,
) -> ExactConditionalBalanceResultV2:
    catalog = build_rule_catalog()
    cells: dict[str, list[int]] = {}
    cell_objects: dict[str, dict[str, Any]] = {}
    for episode in episodes:
        prompt_digest = rendered_static_prompt_digest_v2(
            episode.opening,
            episode.surface.renderer,
        )
        for candidate_index in episode.candidate_indices:
            entry = catalog[candidate_index]
            terminal: dict[str, int] | None
            if audit_kind == "terminal_one_item":
                if episode.terminal_scene_index is None or episode.selection_rank is None:
                    raise RuntimeError("terminal conditional cell lacks its isolated item")
                terminal = {
                    "scene_index": episode.terminal_scene_index,
                    "selection_rank": episode.selection_rank,
                }
            else:
                terminal = None
            cell_obj = {
                "block_opening_digest": episode.opening_digest,
                "candidate_rule_id": entry.rule_id,
                "candidate_truth_digest": entry.truth_digest,
                "model_visible_prompt_digest": prompt_digest,
                "model_visible_surface": episode.surface.model_visible_obj(),
                "terminal": terminal,
            }
            key = _dump_json(cell_obj)
            counts = cells.setdefault(key, [0, 0])
            counts[0] += 1
            counts[1] += int(candidate_index == episode.official_index)
            cell_objects[key] = cell_obj

    rows: list[dict[str, Any]] = []
    violating = 0
    maximum = Fraction()
    for key in sorted(cells):
        total, positives = cells[key]
        difference = abs(fixed_size * positives - total)
        excess = Fraction(difference, fixed_size * total)
        violating += int(difference != 0)
        maximum = max(maximum, excess)
        rows.append(
            {
                "cell": cell_objects[key],
                "row_count": total,
                "positive_count": positives,
                "expected_positive_rate": [1, fixed_size],
                "exact_balance_passed": difference == 0,
            }
        )
    return ExactConditionalBalanceResultV2(
        sum(total for total, _ in cells.values()),
        len(rows),
        violating,
        maximum.numerator,
        maximum.denominator,
        _json_digest(rows, domain=_CONDITIONAL_CELL_DOMAIN),
    )


@dataclass(frozen=True, slots=True)
class TopOneLeakageResultV2:
    target_name: str
    primary_descriptive_target: bool
    observation_count: int
    point_accuracy: float
    chance_mean: float
    point_excess: float
    excess_interval_lower: float
    excess_interval_upper: float
    interval_contains_zero: bool
    upper_excess_below_margin: bool
    powered_adequate: bool
    decision: DecisionV2
    prediction_evidence_digest: str

    def __post_init__(self) -> None:
        if self.target_name not in {"full_v0_official", "designated_p_q_c_official"}:
            raise StatisticalLeakageV2Error("unknown top-one target")
        if type(self.primary_descriptive_target) is not bool:
            raise StatisticalLeakageV2Error("primary_descriptive_target must be Boolean")
        _require_integer(self.observation_count, name="top-one observation_count", minimum=1)
        for name, minimum, maximum in (
            ("point_accuracy", 0.0, 1.0),
            ("chance_mean", 0.0, 1.0),
            ("point_excess", -1.0, 1.0),
            ("excess_interval_lower", -1.0, 1.0),
            ("excess_interval_upper", -1.0, 1.0),
        ):
            _require_float(getattr(self, name), name=name, minimum=minimum, maximum=maximum)
        if self.excess_interval_lower > self.excess_interval_upper:
            raise StatisticalLeakageV2Error("top-one interval endpoints are reversed")
        if not math.isclose(
            self.point_excess,
            self.point_accuracy - self.chance_mean,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise StatisticalLeakageV2Error("top-one point excess is inconsistent")
        expected_contains = self.excess_interval_lower <= 0 <= self.excess_interval_upper
        expected_upper = self.excess_interval_upper < STATISTICAL_LEAKAGE_EXCESS_MARGIN
        if self.interval_contains_zero is not expected_contains:
            raise StatisticalLeakageV2Error("top-one zero-containment flag is inconsistent")
        if self.upper_excess_below_margin is not expected_upper:
            raise StatisticalLeakageV2Error("top-one upper-margin flag is inconsistent")
        if type(self.powered_adequate) is not bool:
            raise StatisticalLeakageV2Error("powered_adequate must be Boolean")
        expected_decision: DecisionV2
        if not self.powered_adequate:
            expected_decision = "insufficient_data"
        elif expected_contains and expected_upper:
            expected_decision = "pass"
        else:
            expected_decision = "leakage"
        if self.decision != expected_decision:
            raise StatisticalLeakageV2Error("top-one decision is inconsistent")
        _require_sha256(
            self.prediction_evidence_digest,
            name="top-one prediction evidence digest",
        )

    @property
    def passed(self) -> bool:
        return self.decision == "pass"

    def as_obj(self) -> dict[str, Any]:
        return {
            "target_name": self.target_name,
            "primary_descriptive_target": self.primary_descriptive_target,
            "observation_count": self.observation_count,
            "point_accuracy": self.point_accuracy,
            "chance_mean": self.chance_mean,
            "point_excess": self.point_excess,
            "excess_interval_lower": self.excess_interval_lower,
            "excess_interval_upper": self.excess_interval_upper,
            "interval_contains_zero": self.interval_contains_zero,
            "upper_excess_below_margin": self.upper_excess_below_margin,
            "powered_adequate": self.powered_adequate,
            "decision": self.decision,
            "passed": self.passed,
            "prediction_evidence_digest": self.prediction_evidence_digest,
        }


@dataclass(frozen=True, slots=True)
class BalancedAccuracyLeakageResultV2:
    observation_count: int
    point_balanced_accuracy: float
    interval_lower: float
    interval_upper: float
    interval_contains_half: bool
    upper_below_point_five_five: bool
    powered_adequate: bool
    decision: DecisionV2
    prediction_evidence_digest: str

    def __post_init__(self) -> None:
        _require_integer(self.observation_count, name="BA observation_count", minimum=1)
        for name in ("point_balanced_accuracy", "interval_lower", "interval_upper"):
            _require_float(getattr(self, name), name=name, minimum=0.0, maximum=1.0)
        if self.interval_lower > self.interval_upper:
            raise StatisticalLeakageV2Error("BA interval endpoints are reversed")
        expected_contains = self.interval_lower <= 0.5 <= self.interval_upper
        expected_upper = self.interval_upper < 0.55
        if self.interval_contains_half is not expected_contains:
            raise StatisticalLeakageV2Error("BA half-containment flag is inconsistent")
        if self.upper_below_point_five_five is not expected_upper:
            raise StatisticalLeakageV2Error("BA upper-margin flag is inconsistent")
        if type(self.powered_adequate) is not bool:
            raise StatisticalLeakageV2Error("powered_adequate must be Boolean")
        expected_decision: DecisionV2
        if not self.powered_adequate:
            expected_decision = "insufficient_data"
        elif expected_contains and expected_upper:
            expected_decision = "pass"
        else:
            expected_decision = "leakage"
        if self.decision != expected_decision:
            raise StatisticalLeakageV2Error("BA decision is inconsistent")
        _require_sha256(
            self.prediction_evidence_digest,
            name="BA prediction evidence digest",
        )

    @property
    def passed(self) -> bool:
        return self.decision == "pass"

    def as_obj(self) -> dict[str, Any]:
        return {
            "target_name": "one_v_rest_macro_balanced_accuracy",
            "primary_descriptive_target": True,
            "observation_count": self.observation_count,
            "point_balanced_accuracy": self.point_balanced_accuracy,
            "interval_lower": self.interval_lower,
            "interval_upper": self.interval_upper,
            "interval_contains_half": self.interval_contains_half,
            "upper_below_point_five_five": self.upper_below_point_five_five,
            "powered_adequate": self.powered_adequate,
            "decision": self.decision,
            "passed": self.passed,
            "prediction_evidence_digest": self.prediction_evidence_digest,
        }


@dataclass(frozen=True, slots=True)
class FeatureViewLeakageResultV2:
    view_name: str
    model_visible: bool
    full_v0_prediction_evidence: PredictionEvidenceV2
    designated_p_q_c_prediction_evidence: PredictionEvidenceV2
    full_v0_top_one: TopOneLeakageResultV2
    designated_p_q_c_top_one: TopOneLeakageResultV2
    macro_balanced_accuracy: BalancedAccuracyLeakageResultV2

    def __post_init__(self) -> None:
        if type(self.view_name) is not str or not self.view_name:
            raise StatisticalLeakageV2Error("feature-view name must be a nonempty string")
        if type(self.model_visible) is not bool:
            raise StatisticalLeakageV2Error("feature-view visibility must be Boolean")
        if type(self.full_v0_prediction_evidence) is not PredictionEvidenceV2:
            raise StatisticalLeakageV2Error("feature view lacks full-V0 prediction evidence")
        if type(self.designated_p_q_c_prediction_evidence) is not PredictionEvidenceV2:
            raise StatisticalLeakageV2Error("feature view lacks P/Q/C prediction evidence")
        if self.full_v0_prediction_evidence.target_name != "full_v0_official":
            raise StatisticalLeakageV2Error("full-V0 prediction evidence has the wrong target")
        if (
            self.designated_p_q_c_prediction_evidence.target_name
            != "designated_p_q_c_official"
        ):
            raise StatisticalLeakageV2Error("P/Q/C prediction evidence has the wrong target")
        if type(self.full_v0_top_one) is not TopOneLeakageResultV2:
            raise StatisticalLeakageV2Error("feature view lacks its exact full-V0 result")
        if type(self.designated_p_q_c_top_one) is not TopOneLeakageResultV2:
            raise StatisticalLeakageV2Error("feature view lacks its exact P/Q/C diagnostic")
        if type(self.macro_balanced_accuracy) is not BalancedAccuracyLeakageResultV2:
            raise StatisticalLeakageV2Error("feature view lacks its exact balanced-accuracy result")
        if (
            self.full_v0_top_one.target_name != "full_v0_official"
            or not self.full_v0_top_one.primary_descriptive_target
        ):
            raise StatisticalLeakageV2Error("full-V0 result must be the primary descriptive target")
        if (
            self.designated_p_q_c_top_one.target_name != "designated_p_q_c_official"
            or self.designated_p_q_c_top_one.primary_descriptive_target
        ):
            raise StatisticalLeakageV2Error("P/Q/C result must remain a secondary diagnostic")
        if (
            self.full_v0_top_one.prediction_evidence_digest
            != self.full_v0_prediction_evidence.digest
        ):
            raise StatisticalLeakageV2Error(
                "full-V0 result does not bind its canonical prediction evidence"
            )
        if (
            self.macro_balanced_accuracy.prediction_evidence_digest
            != self.full_v0_prediction_evidence.digest
        ):
            raise StatisticalLeakageV2Error(
                "balanced-accuracy result does not bind full-V0 prediction evidence"
            )
        if (
            self.designated_p_q_c_top_one.prediction_evidence_digest
            != self.designated_p_q_c_prediction_evidence.digest
        ):
            raise StatisticalLeakageV2Error(
                "P/Q/C result does not bind its canonical prediction evidence"
            )
        if (
            self.full_v0_top_one.observation_count
            != self.full_v0_prediction_evidence.nominal_observation_count
            or self.macro_balanced_accuracy.observation_count
            != self.full_v0_prediction_evidence.nominal_candidate_row_count
            or self.designated_p_q_c_top_one.observation_count
            != self.designated_p_q_c_prediction_evidence.nominal_observation_count
        ):
            raise StatisticalLeakageV2Error(
                "prediction evidence and reported observation counts disagree"
            )
        statuses = {
            self.full_v0_prediction_evidence.status,
            self.designated_p_q_c_prediction_evidence.status,
        }
        if len(statuses) != 1:
            raise StatisticalLeakageV2Error("feature-view evidence statuses disagree")
        if self.full_v0_prediction_evidence.status == "estimated":
            full_observations = self.full_v0_prediction_evidence.top_one_observations()
            expected_accuracy = float(
                sum((row[0] for row in full_observations), Fraction())
                / len(full_observations)
            )
            expected_chance = float(
                sum((row[1] for row in full_observations), Fraction())
                / len(full_observations)
            )
            designated_observations = (
                self.designated_p_q_c_prediction_evidence.top_one_observations()
            )
            designated_accuracy = float(
                sum((row[0] for row in designated_observations), Fraction())
                / len(designated_observations)
            )
            designated_chance = float(
                sum((row[1] for row in designated_observations), Fraction())
                / len(designated_observations)
            )
            full_cells = self.full_v0_prediction_evidence.balanced_accuracy_cells()
            tp = sum(row[0] for row in full_cells)
            fn = sum(row[1] for row in full_cells)
            tn = sum(row[2] for row in full_cells)
            fp = sum(row[3] for row in full_cells)
            expected_ba = 0.5 * (tp / (tp + fn) + tn / (tn + fp))
            comparisons = (
                (self.full_v0_top_one.point_accuracy, expected_accuracy, "full-V0 accuracy"),
                (self.full_v0_top_one.chance_mean, expected_chance, "full-V0 chance"),
                (
                    self.designated_p_q_c_top_one.point_accuracy,
                    designated_accuracy,
                    "P/Q/C accuracy",
                ),
                (
                    self.designated_p_q_c_top_one.chance_mean,
                    designated_chance,
                    "P/Q/C chance",
                ),
                (
                    self.macro_balanced_accuracy.point_balanced_accuracy,
                    expected_ba,
                    "balanced accuracy",
                ),
            )
            if any(
                not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12)
                for observed, expected, _ in comparisons
            ):
                failed = next(
                    name
                    for observed, expected, name in comparisons
                    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12)
                )
                raise StatisticalLeakageV2Error(
                    f"{failed} is not recomputed by the canonical prediction evidence"
                )
        elif any(
            result.decision != "insufficient_data"
            for result in (
                self.full_v0_top_one,
                self.designated_p_q_c_top_one,
                self.macro_balanced_accuracy,
            )
        ):
            raise StatisticalLeakageV2Error(
                "not-estimable evidence cannot support a statistical decision"
            )

    @property
    def descriptive_gate_passed(self) -> bool:
        return self.full_v0_top_one.passed and self.macro_balanced_accuracy.passed

    def as_obj(self) -> dict[str, Any]:
        return {
            "view_name": self.view_name,
            "model_visible": self.model_visible,
            "descriptive_gate_passed": self.descriptive_gate_passed,
            "full_v0_prediction_evidence": self.full_v0_prediction_evidence.as_obj(),
            "designated_p_q_c_prediction_evidence": (
                self.designated_p_q_c_prediction_evidence.as_obj()
            ),
            "full_v0_top_one": self.full_v0_top_one.as_obj(),
            "designated_p_q_c_top_one": self.designated_p_q_c_top_one.as_obj(),
            "macro_balanced_accuracy": self.macro_balanced_accuracy.as_obj(),
        }


@dataclass(frozen=True, slots=True)
class GroupFoldAssignmentV2:
    construction_cluster_digest: str
    fold: int

    def __post_init__(self) -> None:
        _require_sha256(self.construction_cluster_digest, name="construction cluster digest")
        _require_integer(self.fold, name="fold", maximum=STATISTICAL_LEAKAGE_FOLDS - 1)

    def as_obj(self) -> dict[str, Any]:
        return {"construction_cluster_digest": self.construction_cluster_digest, "fold": self.fold}


@dataclass(frozen=True, slots=True)
class StatisticalLeakageAuditReportV2:
    audit_kind: AuditKindV2
    catalog_digest: str
    supported_catalog_digest: str
    frequency_table_digest: str
    config: StatisticalLeakageConfigV2
    dataset_digest: str
    fixed_version_space_size: int
    distinct_block_opening_group_count: int
    construction_cluster_count: int
    distinct_designated_binding_count: int
    terminal_panel_count: int
    episode_sample_count: int
    candidate_row_count: int
    hypothesis_complete_blocks: bool
    official_oblivious_shared_terminal_draws: bool | None
    construction_clusters: tuple[ConstructionClusterV2, ...]
    exact_conditional_balance: ExactConditionalBalanceResultV2
    fold_assignments: tuple[GroupFoldAssignmentV2, ...]
    fold_digest: str
    views: tuple[FeatureViewLeakageResultV2, ...]

    def __post_init__(self) -> None:
        if self.audit_kind not in {"meta_role", "terminal_one_item"}:
            raise StatisticalLeakageV2Error("unknown statistical audit kind")
        for name in (
            "catalog_digest",
            "supported_catalog_digest",
            "frequency_table_digest",
            "dataset_digest",
            "fold_digest",
        ):
            _require_sha256(getattr(self, name), name=name)
        catalog_contract = build_supported_catalog_contract_v2()
        if self.catalog_digest != catalog_contract.source_catalog_digest:
            raise StatisticalLeakageV2Error("report uses the wrong canonical source catalog")
        if self.supported_catalog_digest != catalog_contract.supported_catalog_digest:
            raise StatisticalLeakageV2Error("report uses the wrong supported-catalog allowlist")
        if type(self.config) is not StatisticalLeakageConfigV2:
            raise StatisticalLeakageV2Error("report config has the wrong type")
        _require_integer(
            self.fixed_version_space_size,
            name="fixed_version_space_size",
            minimum=MIN_LIVE_RULES,
            maximum=MAX_LIVE_RULES,
        )
        if self.fixed_version_space_size not in REGISTERED_HYPOTHESIS_COMPLETE_SIZES:
            raise StatisticalLeakageV2Error(
                "the powered training stress report requires fixed n0 in {8, 12, 16}"
            )
        _require_integer(
            self.distinct_block_opening_group_count,
            name="distinct_block_opening_group_count",
            minimum=1,
        )
        _require_integer(
            self.construction_cluster_count,
            name="construction_cluster_count",
            minimum=1,
        )
        if self.construction_cluster_count > self.distinct_block_opening_group_count:
            raise StatisticalLeakageV2Error(
                "construction clusters cannot outnumber distinct openings"
            )
        _require_integer(
            self.distinct_designated_binding_count,
            name="distinct_designated_binding_count",
            minimum=1,
        )
        if self.distinct_designated_binding_count > self.distinct_block_opening_group_count:
            raise StatisticalLeakageV2Error(
                "distinct designated bindings cannot outnumber independent openings"
            )
        _require_integer(self.terminal_panel_count, name="terminal_panel_count")
        _require_integer(self.episode_sample_count, name="episode_sample_count", minimum=1)
        _require_integer(self.candidate_row_count, name="candidate_row_count", minimum=1)
        if self.candidate_row_count != self.episode_sample_count * self.fixed_version_space_size:
            raise StatisticalLeakageV2Error("candidate-row count is inconsistent with fixed |V0|")
        if type(self.hypothesis_complete_blocks) is not bool or not self.hypothesis_complete_blocks:
            raise StatisticalLeakageV2Error("only hypothesis-complete blocks are supported")
        if self.audit_kind == "meta_role":
            if self.terminal_panel_count != 0 or self.official_oblivious_shared_terminal_draws is not None:
                raise StatisticalLeakageV2Error("meta-role report contains terminal-only fields")
        elif type(self.official_oblivious_shared_terminal_draws) is not bool:
            raise StatisticalLeakageV2Error("terminal report lacks its exact shared-draw check")
        if type(self.construction_clusters) is not tuple or len(self.construction_clusters) != (
            self.construction_cluster_count
        ):
            raise StatisticalLeakageV2Error("construction clusters do not cover the report")
        if any(type(item) is not ConstructionClusterV2 for item in self.construction_clusters):
            raise StatisticalLeakageV2Error("construction clusters contain a foreign value")
        if self.construction_clusters != tuple(
            sorted(
                self.construction_clusters,
                key=lambda item: item.construction_cluster_digest,
            )
        ):
            raise StatisticalLeakageV2Error("construction clusters must be canonically sorted")
        member_openings = tuple(
            opening
            for cluster in self.construction_clusters
            for opening in cluster.member_opening_digests
        )
        if len(member_openings) != self.distinct_block_opening_group_count or len(
            set(member_openings)
        ) != len(member_openings):
            raise StatisticalLeakageV2Error(
                "construction clusters do not partition the distinct openings"
            )
        if type(self.exact_conditional_balance) is not ExactConditionalBalanceResultV2:
            raise StatisticalLeakageV2Error("report lacks exact conditional-balance evidence")
        if self.exact_conditional_balance.candidate_row_count != self.candidate_row_count:
            raise StatisticalLeakageV2Error("conditional-balance row count is inconsistent")
        if type(self.fold_assignments) is not tuple or len(self.fold_assignments) != (
            self.construction_cluster_count
        ):
            raise StatisticalLeakageV2Error("fold assignments do not cover every construction cluster")
        if any(type(item) is not GroupFoldAssignmentV2 for item in self.fold_assignments):
            raise StatisticalLeakageV2Error("fold assignments contain a foreign value")
        if self.fold_assignments != tuple(
            sorted(self.fold_assignments, key=lambda item: item.construction_cluster_digest)
        ):
            raise StatisticalLeakageV2Error("fold assignments must be canonically sorted")
        if len({item.construction_cluster_digest for item in self.fold_assignments}) != len(
            self.fold_assignments
        ):
            raise StatisticalLeakageV2Error("a construction cluster appears in multiple folds")
        cluster_digests = {
            cluster.construction_cluster_digest for cluster in self.construction_clusters
        }
        if {item.construction_cluster_digest for item in self.fold_assignments} != cluster_digests:
            raise StatisticalLeakageV2Error("fold assignments differ from construction clusters")
        expected_fold_digest = _fold_digest(
            {item.construction_cluster_digest: item.fold for item in self.fold_assignments}
        )
        if self.fold_digest != expected_fold_digest:
            raise StatisticalLeakageV2Error("fold digest is inconsistent")
        group_ids = tuple(item.construction_cluster_digest for item in self.fold_assignments)
        if {item.construction_cluster_digest: item.fold for item in self.fold_assignments} != (
            _stable_fold_assignments(group_ids, self.config.fold_count)
        ):
            raise StatisticalLeakageV2Error("fold assignments differ from the deterministic mapping")
        if type(self.views) is not tuple or any(
            type(view) is not FeatureViewLeakageResultV2 for view in self.views
        ):
            raise StatisticalLeakageV2Error("report feature views must be an immutable exact tuple")
        expected_views = _META_VIEW_CONTRACT if self.audit_kind == "meta_role" else _TERMINAL_VIEW_CONTRACT
        if tuple((view.view_name, view.model_visible) for view in self.views) != expected_views:
            raise StatisticalLeakageV2Error("feature views differ from the frozen contract")
        powered = self.powered_group_requirement_met
        opening_to_cluster = {
            opening_digest: cluster.construction_cluster_digest
            for cluster in self.construction_clusters
            for opening_digest in cluster.member_opening_digests
        }
        fold_by_cluster = {
            item.construction_cluster_digest: item.fold for item in self.fold_assignments
        }
        designated_observations, remainder = divmod(
            3 * self.episode_sample_count, self.fixed_version_space_size
        )
        if remainder:
            raise StatisticalLeakageV2Error("episode count is incompatible with hypothesis completion")
        reference_full_episode_keys: tuple[tuple[Any, ...], ...] | None = None
        reference_designated_episode_keys: tuple[tuple[Any, ...], ...] | None = None
        for view in self.views:
            if view.full_v0_top_one.observation_count != self.episode_sample_count:
                raise StatisticalLeakageV2Error("full-V0 observation count is inconsistent")
            if view.designated_p_q_c_top_one.observation_count != designated_observations:
                raise StatisticalLeakageV2Error("P/Q/C diagnostic observation count is inconsistent")
            if view.macro_balanced_accuracy.observation_count != self.candidate_row_count:
                raise StatisticalLeakageV2Error("balanced-accuracy row count is inconsistent")
            if (
                view.full_v0_top_one.prediction_evidence_digest
                != view.macro_balanced_accuracy.prediction_evidence_digest
            ):
                raise StatisticalLeakageV2Error("full-V0 and balanced-accuracy predictions diverge")
            expected_status = "estimated" if self.construction_cluster_count >= 2 else "not_estimable"
            if {
                view.full_v0_prediction_evidence.status,
                view.designated_p_q_c_prediction_evidence.status,
            } != {expected_status}:
                raise StatisticalLeakageV2Error(
                    "prediction-evidence status disagrees with construction-cluster count"
                )
            for evidence in (
                view.full_v0_prediction_evidence,
                view.designated_p_q_c_prediction_evidence,
            ):
                if evidence.status != "estimated":
                    continue
                for row in evidence.episodes:
                    expected_cluster = opening_to_cluster.get(row.block_opening_digest)
                    if row.construction_cluster_digest != expected_cluster:
                        raise StatisticalLeakageV2Error(
                            "prediction evidence uses a forged opening-to-cluster assignment"
                        )
                    if row.fold != fold_by_cluster[row.construction_cluster_digest]:
                        raise StatisticalLeakageV2Error(
                            "prediction evidence uses a forged cluster-to-fold assignment"
                        )
                    terminal_present = row.terminal_scene_index is not None
                    if terminal_present is not (self.audit_kind == "terminal_one_item"):
                        raise StatisticalLeakageV2Error(
                            "prediction evidence has the wrong audit-kind episode identity"
                        )
            if view.full_v0_prediction_evidence.status == "estimated":
                full_sets: dict[str, tuple[int, ...]] = {}
                completion_cells: dict[tuple[str, int, int], list[int]] = {}
                for row in view.full_v0_prediction_evidence.episodes:
                    candidate_set = view.full_v0_prediction_evidence.patterns[
                        row.pattern_index
                    ].candidate_indices
                    if len(candidate_set) != self.fixed_version_space_size:
                        raise StatisticalLeakageV2Error(
                            "full-V0 prediction evidence has the wrong candidate count"
                        )
                    prior = full_sets.setdefault(row.block_opening_digest, candidate_set)
                    if prior != candidate_set:
                        raise StatisticalLeakageV2Error(
                            "full-V0 candidate target changes within an opening block"
                        )
                    completion_key = (
                        row.block_opening_digest,
                        -1 if row.terminal_panel_index is None else row.terminal_panel_index,
                        -1 if row.selection_rank is None else row.selection_rank,
                    )
                    completion_cells.setdefault(completion_key, []).append(row.official_index)
                for (opening_digest, _, _), officials in completion_cells.items():
                    if tuple(sorted(officials)) != full_sets[opening_digest]:
                        raise StatisticalLeakageV2Error(
                            "prediction evidence is not hypothesis-complete within an episode cell"
                        )
                designated_sets: dict[str, tuple[int, ...]] = {}
                for row in view.designated_p_q_c_prediction_evidence.episodes:
                    candidate_set = view.designated_p_q_c_prediction_evidence.patterns[
                        row.pattern_index
                    ].candidate_indices
                    prior = designated_sets.setdefault(row.block_opening_digest, candidate_set)
                    if prior != candidate_set:
                        raise StatisticalLeakageV2Error(
                            "designated P/Q/C target changes within an opening block"
                        )
                    if not set(candidate_set).issubset(full_sets[row.block_opening_digest]):
                        raise StatisticalLeakageV2Error(
                            "designated P/Q/C target is not a subset of the exact full V0"
                        )
                expected_designated_keys = {
                    row.canonical_key
                    for row in view.full_v0_prediction_evidence.episodes
                    if row.official_index in designated_sets.get(row.block_opening_digest, ())
                }
                actual_designated_keys = {
                    row.canonical_key
                    for row in view.designated_p_q_c_prediction_evidence.episodes
                }
                if actual_designated_keys != expected_designated_keys:
                    raise StatisticalLeakageV2Error(
                        "designated evidence is not the exact Official-filtered full episode set"
                    )
                full_keys = tuple(
                    row.canonical_key for row in view.full_v0_prediction_evidence.episodes
                )
                designated_keys = tuple(
                    row.canonical_key
                    for row in view.designated_p_q_c_prediction_evidence.episodes
                )
                if reference_full_episode_keys is None:
                    reference_full_episode_keys = full_keys
                    reference_designated_episode_keys = designated_keys
                elif (
                    full_keys != reference_full_episode_keys
                    or designated_keys != reference_designated_episode_keys
                ):
                    raise StatisticalLeakageV2Error(
                        "feature views do not share identical Official-bearing episode evidence"
                    )
            if any(
                result.powered_adequate is not powered
                for result in (
                    view.full_v0_top_one,
                    view.designated_p_q_c_top_one,
                    view.macro_balanced_accuracy,
                )
            ):
                raise StatisticalLeakageV2Error("feature result power flags are inconsistent")

    @property
    def powered_group_requirement_met(self) -> bool:
        return self.construction_cluster_count >= STATISTICAL_LEAKAGE_MINIMUM_GROUPS

    @property
    def exact_structural_gate_passed(self) -> bool:
        shared = (
            True if self.audit_kind == "meta_role" else self.official_oblivious_shared_terminal_draws is True
        )
        return shared and self.exact_conditional_balance.passed

    @property
    def descriptive_oof_gate_passed(self) -> bool:
        return self.powered_group_requirement_met and all(
            view.descriptive_gate_passed for view in self.views
        )

    @property
    def statistical_interval_authorized(self) -> bool:
        return False

    @property
    def primary_statistical_gate_passed(self) -> bool:
        return (
            self.exact_structural_gate_passed
            and self.descriptive_oof_gate_passed
            and self.statistical_interval_authorized
        )

    def _unsigned_obj(self) -> dict[str, Any]:
        return {
            "schema_version": STATISTICAL_LEAKAGE_SCHEMA_VERSION,
            "report_kind": _REPORT_KIND,
            "audit_kind": self.audit_kind,
            "authorization": dict(_AUTHORIZATION),
            "catalog_digest": self.catalog_digest,
            "supported_catalog_digest": self.supported_catalog_digest,
            "frequency_table_digest": self.frequency_table_digest,
            "method_contract": dict(_METHOD_CONTRACT),
            "feature_contract": _feature_contract_obj(),
            "config": self.config.as_obj(),
            "dataset_digest": self.dataset_digest,
            "fixed_version_space_size": self.fixed_version_space_size,
            "distinct_block_opening_group_count": self.distinct_block_opening_group_count,
            "construction_cluster_count": self.construction_cluster_count,
            "distinct_designated_binding_count": self.distinct_designated_binding_count,
            "terminal_panel_count": self.terminal_panel_count,
            "episode_sample_count": self.episode_sample_count,
            "candidate_row_count": self.candidate_row_count,
            "hypothesis_complete_blocks": self.hypothesis_complete_blocks,
            "official_oblivious_shared_terminal_draws": self.official_oblivious_shared_terminal_draws,
            "construction_clusters": [item.as_obj() for item in self.construction_clusters],
            "exact_conditional_balance": self.exact_conditional_balance.as_obj(),
            "exact_structural_gate_passed": self.exact_structural_gate_passed,
            "powered_group_requirement_met": self.powered_group_requirement_met,
            "fold_assignments": [item.as_obj() for item in self.fold_assignments],
            "fold_digest": self.fold_digest,
            "views": [view.as_obj() for view in self.views],
            "descriptive_oof_gate_passed": self.descriptive_oof_gate_passed,
            "statistical_interval_authorized": self.statistical_interval_authorized,
            "statistical_interval_authorization_reason": (
                "the current cluster bootstrap resamples fixed OOF predictions without refitting; "
                "it has no unconditional coverage authorization"
            ),
            "primary_statistical_gate_passed": self.primary_statistical_gate_passed,
            "historical_three_role_design_is_launch_eligible": False,
            "historical_three_role_designated_target_is_diagnostic_only": True,
            "analytic_public_generator_law_audit_required": True,
            "analytic_public_generator_law_audit_passed": False,
            "production_bank_authorized": False,
        }

    @property
    def digest(self) -> str:
        return _json_digest(self._unsigned_obj(), domain=_REPORT_DOMAIN)

    def as_obj(self) -> dict[str, Any]:
        return {**self._unsigned_obj(), "statistical_leakage_audit_digest": self.digest}


def _top_one_result(
    target_name: str,
    primary_descriptive_target: bool,
    observations: tuple[tuple[Fraction, Fraction, str], ...],
    prediction_evidence_digest: str,
    *,
    group_ids: tuple[str, ...],
    powered: bool,
    config: StatisticalLeakageConfigV2,
    dataset_digest: str,
    view_name: str,
) -> TopOneLeakageResultV2:
    accuracy = float(sum((item[0] for item in observations), Fraction()) / len(observations))
    chance = float(sum((item[1] for item in observations), Fraction()) / len(observations))
    lower, upper = _bootstrap_top_one(
        observations,
        group_ids=group_ids,
        replicates=config.bootstrap_replicates,
        seed=_bootstrap_seed(dataset_digest, view_name, target_name),
    )
    contains = lower <= 0 <= upper
    below = upper < STATISTICAL_LEAKAGE_EXCESS_MARGIN
    decision: DecisionV2 = "insufficient_data" if not powered else "pass" if contains and below else "leakage"
    return TopOneLeakageResultV2(
        target_name,
        primary_descriptive_target,
        len(observations),
        accuracy,
        chance,
        accuracy - chance,
        lower,
        upper,
        contains,
        below,
        powered,
        decision,
        prediction_evidence_digest,
    )


def _ba_result(
    cells: tuple[tuple[int, int, int, int, str], ...],
    prediction_evidence_digest: str,
    *,
    group_ids: tuple[str, ...],
    powered: bool,
    config: StatisticalLeakageConfigV2,
    dataset_digest: str,
    view_name: str,
) -> BalancedAccuracyLeakageResultV2:
    tp = sum(item[0] for item in cells)
    fn = sum(item[1] for item in cells)
    tn = sum(item[2] for item in cells)
    fp = sum(item[3] for item in cells)
    point = 0.5 * (tp / (tp + fn) + tn / (tn + fp))
    lower, upper = _bootstrap_ba(
        cells,
        group_ids=group_ids,
        replicates=config.bootstrap_replicates,
        seed=_bootstrap_seed(
            dataset_digest,
            view_name,
            "one_v_rest_macro_balanced_accuracy",
        ),
    )
    contains = lower <= 0.5 <= upper
    below = upper < 0.55
    decision: DecisionV2 = "insufficient_data" if not powered else "pass" if contains and below else "leakage"
    return BalancedAccuracyLeakageResultV2(
        sum(tp_fn_tn_fp for tp_fn_tn_fp in (tp, fn, tn, fp)),
        point,
        lower,
        upper,
        contains,
        below,
        powered,
        decision,
        prediction_evidence_digest,
    )


def _build_views(
    audit_kind: AuditKindV2,
    episodes: tuple[_Episode, ...],
    *,
    frequencies: CatalogFrequencyTableV2,
    assignments: Mapping[str, int],
    group_ids: tuple[str, ...],
    powered: bool,
    config: StatisticalLeakageConfigV2,
    dataset_digest: str,
) -> tuple[FeatureViewLeakageResultV2, ...]:
    contract = _META_VIEW_CONTRACT if audit_kind == "meta_role" else _TERMINAL_VIEW_CONTRACT
    if len(group_ids) < 2:
        full_observations = len(episodes)
        designated_observations = sum(
            episode.official_index in episode.designated_indices for episode in episodes
        )
        candidate_rows = sum(len(episode.candidate_indices) for episode in episodes)
        chance = float(
            sum((Fraction(1, len(episode.candidate_indices)) for episode in episodes), Fraction())
            / full_observations
        )
        insufficient_results: list[FeatureViewLeakageResultV2] = []
        for view_name, model_visible in contract:
            full_evidence = PredictionEvidenceV2(
                "full_v0_official",
                "not_estimable",
                full_observations,
                candidate_rows,
                (),
                (),
            )
            designated_evidence = PredictionEvidenceV2(
                "designated_p_q_c_official",
                "not_estimable",
                designated_observations,
                designated_observations * 3,
                (),
                (),
            )
            insufficient_results.append(
                FeatureViewLeakageResultV2(
                    view_name,
                    model_visible,
                    full_evidence,
                    designated_evidence,
                    TopOneLeakageResultV2(
                        "full_v0_official",
                        True,
                        full_observations,
                        chance,
                        chance,
                        0.0,
                        -1.0,
                        1.0,
                        True,
                        False,
                        False,
                        "insufficient_data",
                        full_evidence.digest,
                    ),
                    TopOneLeakageResultV2(
                        "designated_p_q_c_official",
                        False,
                        designated_observations,
                        1 / 3,
                        1 / 3,
                        0.0,
                        -1.0,
                        1.0,
                        True,
                        False,
                        False,
                        "insufficient_data",
                        designated_evidence.digest,
                    ),
                    BalancedAccuracyLeakageResultV2(
                        candidate_rows,
                        0.5,
                        0.0,
                        1.0,
                        True,
                        False,
                        False,
                        "insufficient_data",
                        full_evidence.digest,
                    ),
                )
            )
        return tuple(insufficient_results)
    results: list[FeatureViewLeakageResultV2] = []
    for view_name, model_visible in contract:
        if audit_kind == "meta_role":
            feature_builder = lambda episode, candidate, name=view_name: _meta_feature_tokens(  # noqa: E731
                episode, candidate, name, frequencies
            )
        else:
            feature_builder = lambda episode, candidate, name=view_name: _terminal_feature_tokens(  # noqa: E731
                episode, candidate, name, frequencies
            )
        prepared_feature_rows = _prepare_feature_rows(episodes, feature_builder)
        full_top, full_ba, full_evidence = _score_rows(
            prepared_feature_rows,
            target_name="full_v0_official",
            assignments=assignments,
            candidate_selector=lambda episode: episode.candidate_indices,
        )
        designated_top, _, designated_evidence = _score_rows(
            prepared_feature_rows,
            target_name="designated_p_q_c_official",
            assignments=assignments,
            candidate_selector=lambda episode: episode.designated_indices,
        )
        results.append(
            FeatureViewLeakageResultV2(
                view_name,
                model_visible,
                full_evidence,
                designated_evidence,
                _top_one_result(
                    "full_v0_official",
                    True,
                    full_top,
                    full_evidence.digest,
                    group_ids=group_ids,
                    powered=powered,
                    config=config,
                    dataset_digest=dataset_digest,
                    view_name=view_name,
                ),
                _top_one_result(
                    "designated_p_q_c_official",
                    False,
                    designated_top,
                    designated_evidence.digest,
                    group_ids=group_ids,
                    powered=powered,
                    config=config,
                    dataset_digest=dataset_digest,
                    view_name=view_name,
                ),
                _ba_result(
                    full_ba,
                    full_evidence.digest,
                    group_ids=group_ids,
                    powered=powered,
                    config=config,
                    dataset_digest=dataset_digest,
                    view_name=view_name,
                ),
            )
        )
    return tuple(results)


def _validate_blocks(
    blocks: tuple[HypothesisCompleteMetaBlockV2, ...],
) -> tuple[
    int,
    tuple[str, ...],
    dict[str, str],
    tuple[ConstructionClusterV2, ...],
]:
    if not blocks:
        raise StatisticalLeakageV2Error("statistical audit requires at least one block")
    if any(type(block) is not HypothesisCompleteMetaBlockV2 for block in blocks):
        raise TypeError("meta audit accepts only HypothesisCompleteMetaBlockV2 values")
    group_ids = tuple(block.block_opening_digest for block in blocks)
    if len(group_ids) != len(set(group_ids)):
        raise StatisticalLeakageV2Error("exact block/opening semantic digests must be distinct and unique")
    sizes = {len(block.version_space) for block in blocks}
    if len(sizes) != 1:
        raise StatisticalLeakageV2Error("the powered stress population requires fixed |V0|")
    fixed_size = next(iter(sizes))
    if fixed_size not in REGISTERED_HYPOTHESIS_COMPLETE_SIZES:
        raise StatisticalLeakageV2Error("the powered training stress gate requires fixed n0 in {8, 12, 16}")
    opening_to_cluster, clusters = _derive_construction_clusters(blocks)
    return fixed_size, tuple(sorted(group_ids)), opening_to_cluster, clusters


def derive_catalog_frequency_table_v2(
    blocks: Iterable[HypothesisCompleteMetaBlockV2],
) -> CatalogFrequencyTableV2:
    """Count exact block inclusion, equal to Official exposure under completion."""

    materialized = tuple(blocks)
    if not materialized:
        raise StatisticalLeakageV2Error("frequency derivation requires at least one block")
    if any(type(block) is not HypothesisCompleteMetaBlockV2 for block in materialized):
        raise TypeError("frequency derivation accepts only HypothesisCompleteMetaBlockV2 values")
    contract = build_supported_catalog_contract_v2()
    counts = [0] * len(contract.supported_indices)
    positions = _supported_position_map()
    for block in materialized:
        for candidate_index in block.version_space.indices:
            counts[positions[candidate_index]] += 1
    return CatalogFrequencyTableV2(tuple(counts))


def _require_recomputed_frequencies(
    blocks: tuple[HypothesisCompleteMetaBlockV2, ...],
    supplied: CatalogFrequencyTableV2 | None,
) -> CatalogFrequencyTableV2:
    derived = derive_catalog_frequency_table_v2(blocks)
    if supplied is not None:
        if type(supplied) is not CatalogFrequencyTableV2:
            raise TypeError("frequencies must be a CatalogFrequencyTableV2 or None")
        if supplied != derived:
            raise StatisticalLeakageV2Error(
                "supplied catalog frequencies differ from exact block-population recomputation"
            )
    return derived


def build_meta_role_statistical_leakage_audit_v2(
    blocks: Iterable[HypothesisCompleteMetaBlockV2],
    frequencies: CatalogFrequencyTableV2 | None = None,
    *,
    config: StatisticalLeakageConfigV2 = DEFAULT_STATISTICAL_LEAKAGE_CONFIG_V2,
) -> StatisticalLeakageAuditReportV2:
    """Audit hypothesis-complete meta-role blocks with exact grouped OOF folds."""

    materialized = tuple(blocks)
    if type(config) is not StatisticalLeakageConfigV2:
        raise TypeError("config must be a StatisticalLeakageConfigV2")
    fixed_size, opening_ids, opening_to_cluster, clusters = _validate_blocks(materialized)
    canonical = tuple(sorted(materialized, key=lambda block: block.block_opening_digest))
    frequencies = _require_recomputed_frequencies(canonical, frequencies)
    dataset_digest = _json_digest(
        {
            "audit_kind": "meta_role",
            "frequency_table_digest": frequencies.digest,
            "blocks": [block.as_binding_obj() for block in canonical],
        },
        domain=_DATASET_DOMAIN,
    )
    cluster_ids = tuple(cluster.construction_cluster_digest for cluster in clusters)
    assignments = _stable_fold_assignments(cluster_ids, config.fold_count)
    episodes = tuple(
        _Episode(
            group_id=opening_to_cluster[block.block_opening_digest],
            opening_digest=block.block_opening_digest,
            official_index=official_index,
            candidate_indices=block.version_space.indices,
            designated_indices=block.designated_indices,
            surface=block.rotation_surfaces[position],
            opening=block.opening,
        )
        for block in canonical
        for position, official_index in enumerate(block.version_space.indices)
    )
    powered = len(cluster_ids) >= STATISTICAL_LEAKAGE_MINIMUM_GROUPS
    exact_balance = _exact_conditional_balance("meta_role", episodes, fixed_size=fixed_size)
    views = _build_views(
        "meta_role",
        episodes,
        frequencies=frequencies,
        assignments=assignments,
        group_ids=cluster_ids,
        powered=powered,
        config=config,
        dataset_digest=dataset_digest,
    )
    fold_rows = tuple(
        GroupFoldAssignmentV2(group_id, assignments[group_id]) for group_id in sorted(cluster_ids)
    )
    return StatisticalLeakageAuditReportV2(
        "meta_role",
        frequencies.catalog_digest,
        frequencies.supported_catalog_digest,
        frequencies.digest,
        config,
        dataset_digest,
        fixed_size,
        len(opening_ids),
        len(cluster_ids),
        len({block.binding.digest for block in canonical}),
        0,
        len(episodes),
        len(episodes) * fixed_size,
        True,
        None,
        clusters,
        exact_balance,
        fold_rows,
        _fold_digest(assignments),
        views,
    )


def build_terminal_statistical_leakage_audit_v2(
    blocks: Iterable[HypothesisCompleteTerminalBlockV2],
    frequencies: CatalogFrequencyTableV2 | None = None,
    *,
    config: StatisticalLeakageConfigV2 = DEFAULT_STATISTICAL_LEAKAGE_CONFIG_V2,
) -> StatisticalLeakageAuditReportV2:
    """Audit one-item terminals while keeping every opening and its panels grouped."""

    materialized = tuple(blocks)
    if not materialized:
        raise StatisticalLeakageV2Error("terminal statistical audit requires at least one block")
    if any(type(block) is not HypothesisCompleteTerminalBlockV2 for block in materialized):
        raise TypeError("terminal audit accepts only HypothesisCompleteTerminalBlockV2 values")
    meta_blocks = tuple(block.meta_block for block in materialized)
    fixed_size, opening_ids, opening_to_cluster, clusters = _validate_blocks(meta_blocks)
    canonical = tuple(sorted(materialized, key=lambda block: block.meta_block.block_opening_digest))
    frequencies = _require_recomputed_frequencies(tuple(block.meta_block for block in canonical), frequencies)
    panel_shapes = {tuple(panel.item_count for panel in block.panels) for block in canonical}
    if len(panel_shapes) != 1:
        raise StatisticalLeakageV2Error(
            "every block/opening group must contribute the same terminal panel/item shape"
        )
    panel_indices = {tuple(panel.panel_index for panel in block.panels) for block in canonical}
    if len(panel_indices) != 1:
        raise StatisticalLeakageV2Error("terminal panel indices must be identical across triples")
    dataset_digest = _json_digest(
        {
            "audit_kind": "terminal_one_item",
            "frequency_table_digest": frequencies.digest,
            "blocks": [block.as_binding_obj() for block in canonical],
        },
        domain=_DATASET_DOMAIN,
    )
    cluster_ids = tuple(cluster.construction_cluster_digest for cluster in clusters)
    assignments = _stable_fold_assignments(cluster_ids, config.fold_count)
    episodes: list[_Episode] = []
    for block in canonical:
        meta = block.meta_block
        for panel in block.panels:
            for official_position, official_index in enumerate(meta.version_space.indices):
                surface = meta.rotation_surfaces[official_position]
                for rank, scene_index in enumerate(panel.scene_indices_by_official[official_position]):
                    episodes.append(
                        _Episode(
                            group_id=opening_to_cluster[meta.block_opening_digest],
                            opening_digest=meta.block_opening_digest,
                            official_index=official_index,
                            candidate_indices=meta.version_space.indices,
                            designated_indices=meta.designated_indices,
                            surface=surface,
                            opening=meta.opening,
                            terminal_panel_index=panel.panel_index,
                            terminal_scene_index=scene_index,
                            selection_rank=rank,
                        )
                    )
    episode_tuple = tuple(episodes)
    powered = len(cluster_ids) >= STATISTICAL_LEAKAGE_MINIMUM_GROUPS
    exact_balance = _exact_conditional_balance(
        "terminal_one_item",
        episode_tuple,
        fixed_size=fixed_size,
    )
    views = _build_views(
        "terminal_one_item",
        episode_tuple,
        frequencies=frequencies,
        assignments=assignments,
        group_ids=cluster_ids,
        powered=powered,
        config=config,
        dataset_digest=dataset_digest,
    )
    fold_rows = tuple(
        GroupFoldAssignmentV2(group_id, assignments[group_id]) for group_id in sorted(cluster_ids)
    )
    panel_count = sum(len(block.panels) for block in canonical)
    return StatisticalLeakageAuditReportV2(
        "terminal_one_item",
        frequencies.catalog_digest,
        frequencies.supported_catalog_digest,
        frequencies.digest,
        config,
        dataset_digest,
        fixed_size,
        len(opening_ids),
        len(cluster_ids),
        len({block.meta_block.binding.digest for block in canonical}),
        panel_count,
        len(episode_tuple),
        len(episode_tuple) * fixed_size,
        True,
        all(block.official_oblivious_shared for block in canonical),
        clusters,
        exact_balance,
        fold_rows,
        _fold_digest(assignments),
        views,
    )


def _top_one_from_obj(value: object) -> TopOneLeakageResultV2:
    fields = (
        "target_name",
        "primary_descriptive_target",
        "observation_count",
        "point_accuracy",
        "chance_mean",
        "point_excess",
        "excess_interval_lower",
        "excess_interval_upper",
        "interval_contains_zero",
        "upper_excess_below_margin",
        "powered_adequate",
        "decision",
        "passed",
        "prediction_evidence_digest",
    )
    obj = _require_mapping(value, fields, name="top-one leakage result")
    target_name = obj["target_name"]
    decision = obj["decision"]
    if type(target_name) is not str or type(decision) is not str:
        raise StatisticalLeakageV2Error("top-one target or decision is not a string")
    result = TopOneLeakageResultV2(
        target_name,
        _require_bool(
            obj["primary_descriptive_target"],
            name="primary_descriptive_target",
        ),
        _require_integer(obj["observation_count"], name="observation_count", minimum=1),
        _require_float(obj["point_accuracy"], name="point_accuracy", minimum=0, maximum=1),
        _require_float(obj["chance_mean"], name="chance_mean", minimum=0, maximum=1),
        _require_float(obj["point_excess"], name="point_excess", minimum=-1, maximum=1),
        _require_float(obj["excess_interval_lower"], name="excess_interval_lower", minimum=-1, maximum=1),
        _require_float(obj["excess_interval_upper"], name="excess_interval_upper", minimum=-1, maximum=1),
        _require_bool(obj["interval_contains_zero"], name="interval_contains_zero"),
        _require_bool(obj["upper_excess_below_margin"], name="upper_excess_below_margin"),
        _require_bool(obj["powered_adequate"], name="powered_adequate"),
        cast(DecisionV2, decision),
        _require_sha256(
            obj["prediction_evidence_digest"],
            name="prediction_evidence_digest",
        ),
    )
    if obj["passed"] is not result.passed:
        raise StatisticalLeakageV2Error("top-one serialized pass flag is inconsistent")
    return result


def _ba_from_obj(value: object) -> BalancedAccuracyLeakageResultV2:
    fields = (
        "target_name",
        "primary_descriptive_target",
        "observation_count",
        "point_balanced_accuracy",
        "interval_lower",
        "interval_upper",
        "interval_contains_half",
        "upper_below_point_five_five",
        "powered_adequate",
        "decision",
        "passed",
        "prediction_evidence_digest",
    )
    obj = _require_mapping(value, fields, name="balanced-accuracy leakage result")
    if (
        obj["target_name"] != "one_v_rest_macro_balanced_accuracy"
        or obj["primary_descriptive_target"] is not True
    ):
        raise StatisticalLeakageV2Error("balanced-accuracy target identity is inconsistent")
    decision = obj["decision"]
    if type(decision) is not str:
        raise StatisticalLeakageV2Error("balanced-accuracy decision must be a string")
    result = BalancedAccuracyLeakageResultV2(
        _require_integer(obj["observation_count"], name="observation_count", minimum=1),
        _require_float(
            obj["point_balanced_accuracy"],
            name="point_balanced_accuracy",
            minimum=0,
            maximum=1,
        ),
        _require_float(obj["interval_lower"], name="interval_lower", minimum=0, maximum=1),
        _require_float(obj["interval_upper"], name="interval_upper", minimum=0, maximum=1),
        _require_bool(obj["interval_contains_half"], name="interval_contains_half"),
        _require_bool(obj["upper_below_point_five_five"], name="upper_below_point_five_five"),
        _require_bool(obj["powered_adequate"], name="powered_adequate"),
        cast(DecisionV2, decision),
        _require_sha256(
            obj["prediction_evidence_digest"],
            name="prediction_evidence_digest",
        ),
    )
    if obj["passed"] is not result.passed:
        raise StatisticalLeakageV2Error("balanced-accuracy serialized pass flag is inconsistent")
    return result


def _prediction_pattern_from_obj(value: object) -> PredictionPatternV2:
    obj = _require_mapping(
        value,
        (
            "candidate_indices",
            "exact_scores",
            "predicted_official",
            "winner_indices",
        ),
        name="prediction pattern",
    )
    raw_candidates = obj["candidate_indices"]
    raw_scores = obj["exact_scores"]
    raw_predictions = obj["predicted_official"]
    raw_winners = obj["winner_indices"]
    if any(type(value) is not list for value in (raw_candidates, raw_scores, raw_predictions, raw_winners)):
        raise StatisticalLeakageV2Error("prediction-pattern vectors must be arrays")
    exact_scores: list[tuple[int, int]] = []
    for raw_score in raw_scores:
        if type(raw_score) is not list or len(raw_score) != 2:
            raise StatisticalLeakageV2Error("serialized exact prediction score is malformed")
        exact_scores.append(
            (
                _require_signed_integer(raw_score[0], name="score numerator"),
                _require_integer(raw_score[1], name="score denominator", minimum=1),
            )
        )
    return PredictionPatternV2(
        tuple(
            _require_integer(index, name="candidate index") for index in raw_candidates
        ),
        tuple(exact_scores),
        tuple(_require_bool(bit, name="predicted_official") for bit in raw_predictions),
        tuple(_require_integer(index, name="winner index") for index in raw_winners),
    )


def _prediction_episode_from_obj(value: object) -> PredictionEpisodeEvidenceV2:
    obj = _require_mapping(
        value,
        (
            "construction_cluster_digest",
            "block_opening_digest",
            "fold",
            "official_index",
            "terminal_panel_index",
            "terminal_scene_index",
            "selection_rank",
            "pattern_index",
            "credit",
        ),
        name="prediction episode evidence",
    )
    raw_credit = obj["credit"]
    if type(raw_credit) is not list or len(raw_credit) != 2:
        raise StatisticalLeakageV2Error("serialized prediction credit is malformed")

    def optional_index(raw: object, *, name: str, scene: bool = False) -> int | None:
        if raw is None:
            return None
        if scene:
            return _scene_index(raw, name=name)
        return _require_integer(raw, name=name)

    return PredictionEpisodeEvidenceV2(
        _require_sha256(
            obj["construction_cluster_digest"],
            name="evidence construction cluster",
        ),
        _require_sha256(obj["block_opening_digest"], name="evidence block opening"),
        _require_integer(
            obj["fold"],
            name="evidence fold",
            maximum=STATISTICAL_LEAKAGE_FOLDS - 1,
        ),
        _require_integer(obj["official_index"], name="evidence Official"),
        optional_index(obj["terminal_panel_index"], name="evidence terminal panel"),
        optional_index(
            obj["terminal_scene_index"],
            name="evidence terminal scene",
            scene=True,
        ),
        optional_index(obj["selection_rank"], name="evidence selection rank"),
        _require_integer(obj["pattern_index"], name="prediction pattern index"),
        (
            _require_integer(raw_credit[0], name="credit numerator"),
            _require_integer(raw_credit[1], name="credit denominator", minimum=1),
        ),
    )


def _prediction_evidence_from_obj(value: object) -> PredictionEvidenceV2:
    obj = _require_mapping(
        value,
        (
            "target_name",
            "status",
            "nominal_observation_count",
            "nominal_candidate_row_count",
            "patterns",
            "episodes",
            "prediction_evidence_digest",
        ),
        name="prediction evidence",
    )
    target_name = obj["target_name"]
    status = obj["status"]
    if type(target_name) is not str or type(status) is not str:
        raise StatisticalLeakageV2Error("prediction target and status must be strings")
    raw_patterns = obj["patterns"]
    raw_episodes = obj["episodes"]
    if type(raw_patterns) is not list or type(raw_episodes) is not list:
        raise StatisticalLeakageV2Error("prediction patterns and episodes must be arrays")
    result = PredictionEvidenceV2(
        target_name,
        cast(Literal["estimated", "not_estimable"], status),
        _require_integer(
            obj["nominal_observation_count"],
            name="prediction nominal observation count",
            minimum=1,
        ),
        _require_integer(
            obj["nominal_candidate_row_count"],
            name="prediction nominal candidate-row count",
            minimum=1,
        ),
        tuple(_prediction_pattern_from_obj(raw) for raw in raw_patterns),
        tuple(_prediction_episode_from_obj(raw) for raw in raw_episodes),
    )
    if _require_sha256(
        obj["prediction_evidence_digest"],
        name="prediction_evidence_digest",
    ) != result.digest:
        raise StatisticalLeakageV2Error("prediction evidence digest is inconsistent")
    return result


def _view_from_obj(value: object) -> FeatureViewLeakageResultV2:
    obj = _require_mapping(
        value,
        (
            "view_name",
            "model_visible",
            "descriptive_gate_passed",
            "full_v0_prediction_evidence",
            "designated_p_q_c_prediction_evidence",
            "full_v0_top_one",
            "designated_p_q_c_top_one",
            "macro_balanced_accuracy",
        ),
        name="feature-view leakage result",
    )
    if type(obj["view_name"]) is not str:
        raise StatisticalLeakageV2Error("view name must be a string")
    result = FeatureViewLeakageResultV2(
        obj["view_name"],
        _require_bool(obj["model_visible"], name="model_visible"),
        _prediction_evidence_from_obj(obj["full_v0_prediction_evidence"]),
        _prediction_evidence_from_obj(obj["designated_p_q_c_prediction_evidence"]),
        _top_one_from_obj(obj["full_v0_top_one"]),
        _top_one_from_obj(obj["designated_p_q_c_top_one"]),
        _ba_from_obj(obj["macro_balanced_accuracy"]),
    )
    if obj["descriptive_gate_passed"] is not result.descriptive_gate_passed:
        raise StatisticalLeakageV2Error("feature-view serialized descriptive flag is inconsistent")
    return result


def _construction_cluster_from_obj(value: object) -> ConstructionClusterV2:
    obj = _require_mapping(
        value,
        ("construction_cluster_digest", "member_opening_digests"),
        name="construction cluster",
    )
    raw_members = obj["member_opening_digests"]
    if type(raw_members) is not list:
        raise StatisticalLeakageV2Error("construction-cluster members must be an array")
    return ConstructionClusterV2(
        _require_sha256(
            obj["construction_cluster_digest"],
            name="construction_cluster_digest",
        ),
        tuple(_require_sha256(item, name="member opening digest") for item in raw_members),
    )


def _exact_conditional_balance_from_obj(value: object) -> ExactConditionalBalanceResultV2:
    obj = _require_mapping(
        value,
        (
            "candidate_row_count",
            "conditional_cell_count",
            "violating_cell_count",
            "maximum_absolute_excess",
            "conditional_cell_digest",
            "passed",
        ),
        name="exact conditional balance",
    )
    raw_excess = obj["maximum_absolute_excess"]
    if type(raw_excess) is not list or len(raw_excess) != 2:
        raise StatisticalLeakageV2Error("maximum conditional excess must be a rational pair")
    result = ExactConditionalBalanceResultV2(
        _require_integer(obj["candidate_row_count"], name="candidate_row_count", minimum=1),
        _require_integer(obj["conditional_cell_count"], name="conditional_cell_count", minimum=1),
        _require_integer(obj["violating_cell_count"], name="violating_cell_count"),
        _require_integer(raw_excess[0], name="maximum excess numerator"),
        _require_integer(raw_excess[1], name="maximum excess denominator", minimum=1),
        _require_sha256(obj["conditional_cell_digest"], name="conditional_cell_digest"),
    )
    if obj["passed"] is not result.passed:
        raise StatisticalLeakageV2Error("conditional-balance pass flag is inconsistent")
    return result


def statistical_leakage_audit_v2_from_obj(
    value: object,
    *,
    expected_digest: str,
) -> StatisticalLeakageAuditReportV2:
    """Reconstruct only evidence whose digest was pinned outside the report."""

    fields = (
        "schema_version",
        "report_kind",
        "audit_kind",
        "authorization",
        "catalog_digest",
        "supported_catalog_digest",
        "frequency_table_digest",
        "method_contract",
        "feature_contract",
        "config",
        "dataset_digest",
        "fixed_version_space_size",
        "distinct_block_opening_group_count",
        "construction_cluster_count",
        "distinct_designated_binding_count",
        "terminal_panel_count",
        "episode_sample_count",
        "candidate_row_count",
        "hypothesis_complete_blocks",
        "official_oblivious_shared_terminal_draws",
        "construction_clusters",
        "exact_conditional_balance",
        "exact_structural_gate_passed",
        "powered_group_requirement_met",
        "fold_assignments",
        "fold_digest",
        "views",
        "descriptive_oof_gate_passed",
        "statistical_interval_authorized",
        "statistical_interval_authorization_reason",
        "primary_statistical_gate_passed",
        "historical_three_role_design_is_launch_eligible",
        "historical_three_role_designated_target_is_diagnostic_only",
        "analytic_public_generator_law_audit_required",
        "analytic_public_generator_law_audit_passed",
        "production_bank_authorized",
        "statistical_leakage_audit_digest",
    )
    obj = _require_mapping(value, fields, name="statistical leakage audit")
    if (
        type(obj["schema_version"]) is not int
        or obj["schema_version"] != STATISTICAL_LEAKAGE_SCHEMA_VERSION
        or obj["report_kind"] != _REPORT_KIND
    ):
        raise StatisticalLeakageV2Error("statistical audit schema identity mismatch")
    authorization = _require_mapping(obj["authorization"], tuple(_AUTHORIZATION), name="authorization")
    if dict(authorization) != _AUTHORIZATION:
        raise StatisticalLeakageV2Error("statistical audit is not explicitly nonauthorizing")
    if obj["method_contract"] != _METHOD_CONTRACT or obj["feature_contract"] != _feature_contract_obj():
        raise StatisticalLeakageV2Error("statistical method or feature contract was changed")
    config_obj = _require_mapping(
        obj["config"],
        (
            "fold_count",
            "bootstrap_replicates",
            "minimum_construction_clusters",
        ),
        name="statistical config",
    )
    if config_obj["minimum_construction_clusters"] != STATISTICAL_LEAKAGE_MINIMUM_GROUPS:
        raise StatisticalLeakageV2Error("minimum powered group count was changed")
    config = StatisticalLeakageConfigV2(
        _require_integer(config_obj["fold_count"], name="fold_count"),
        _require_integer(config_obj["bootstrap_replicates"], name="bootstrap_replicates", minimum=1_000),
    )
    raw_folds = obj["fold_assignments"]
    if type(raw_folds) is not list:
        raise StatisticalLeakageV2Error("fold assignments must be an array")
    fold_assignments: list[GroupFoldAssignmentV2] = []
    for raw in raw_folds:
        fold_obj = _require_mapping(
            raw,
            ("construction_cluster_digest", "fold"),
            name="fold assignment",
        )
        fold_assignments.append(
            GroupFoldAssignmentV2(
                _require_sha256(
                    fold_obj["construction_cluster_digest"],
                    name="construction_cluster_digest",
                ),
                _require_integer(fold_obj["fold"], name="fold", maximum=STATISTICAL_LEAKAGE_FOLDS - 1),
            )
        )
    raw_clusters = obj["construction_clusters"]
    if type(raw_clusters) is not list:
        raise StatisticalLeakageV2Error("construction clusters must be an array")
    construction_clusters = tuple(
        _construction_cluster_from_obj(raw) for raw in raw_clusters
    )
    raw_views = obj["views"]
    if type(raw_views) is not list:
        raise StatisticalLeakageV2Error("feature views must be an array")
    audit_kind = obj["audit_kind"]
    if audit_kind not in {"meta_role", "terminal_one_item"}:
        raise StatisticalLeakageV2Error("unknown audit kind")
    official_oblivious = obj["official_oblivious_shared_terminal_draws"]
    if official_oblivious is not None:
        official_oblivious = _require_bool(
            official_oblivious, name="official_oblivious_shared_terminal_draws"
        )
    report = StatisticalLeakageAuditReportV2(
        cast(AuditKindV2, audit_kind),
        _require_sha256(obj["catalog_digest"], name="catalog_digest"),
        _require_sha256(obj["supported_catalog_digest"], name="supported_catalog_digest"),
        _require_sha256(obj["frequency_table_digest"], name="frequency_table_digest"),
        config,
        _require_sha256(obj["dataset_digest"], name="dataset_digest"),
        _require_integer(
            obj["fixed_version_space_size"],
            name="fixed_version_space_size",
            minimum=MIN_LIVE_RULES,
            maximum=MAX_LIVE_RULES,
        ),
        _require_integer(
            obj["distinct_block_opening_group_count"],
            name="distinct_block_opening_group_count",
            minimum=1,
        ),
        _require_integer(
            obj["construction_cluster_count"],
            name="construction_cluster_count",
            minimum=1,
        ),
        _require_integer(
            obj["distinct_designated_binding_count"],
            name="distinct_designated_binding_count",
            minimum=1,
        ),
        _require_integer(obj["terminal_panel_count"], name="terminal_panel_count"),
        _require_integer(obj["episode_sample_count"], name="episode_sample_count", minimum=1),
        _require_integer(obj["candidate_row_count"], name="candidate_row_count", minimum=1),
        _require_bool(obj["hypothesis_complete_blocks"], name="hypothesis_complete_blocks"),
        cast(bool | None, official_oblivious),
        construction_clusters,
        _exact_conditional_balance_from_obj(obj["exact_conditional_balance"]),
        tuple(fold_assignments),
        _require_sha256(obj["fold_digest"], name="fold_digest"),
        tuple(_view_from_obj(raw) for raw in raw_views),
    )
    for field, expected in (
        ("exact_structural_gate_passed", report.exact_structural_gate_passed),
        ("powered_group_requirement_met", report.powered_group_requirement_met),
        ("descriptive_oof_gate_passed", report.descriptive_oof_gate_passed),
        ("statistical_interval_authorized", report.statistical_interval_authorized),
        (
            "statistical_interval_authorization_reason",
            "the current cluster bootstrap resamples fixed OOF predictions without refitting; "
            "it has no unconditional coverage authorization",
        ),
        ("primary_statistical_gate_passed", report.primary_statistical_gate_passed),
        ("historical_three_role_design_is_launch_eligible", False),
        ("historical_three_role_designated_target_is_diagnostic_only", True),
        ("analytic_public_generator_law_audit_required", True),
        ("analytic_public_generator_law_audit_passed", False),
        ("production_bank_authorized", False),
    ):
        if (type(expected) is bool and obj[field] is not expected) or (
            type(expected) is not bool and obj[field] != expected
        ):
            raise StatisticalLeakageV2Error(f"derived report field is inconsistent: {field}")
    observed_digest = _require_sha256(
        obj["statistical_leakage_audit_digest"], name="statistical_leakage_audit_digest"
    )
    if observed_digest != report.digest:
        raise StatisticalLeakageV2Error("statistical leakage report digest mismatch")
    if report.digest != _require_sha256(expected_digest, name="expected_digest"):
        raise StatisticalLeakageV2Error("report differs from the externally expected digest")
    if _dump_json(obj) != _dump_json(report.as_obj()):
        raise StatisticalLeakageV2Error("statistical leakage report has inconsistent derived data")
    return report


def verify_statistical_leakage_audit_v2(
    report: StatisticalLeakageAuditReportV2,
) -> StatisticalLeakageAuditReportV2:
    if type(report) is not StatisticalLeakageAuditReportV2:
        raise TypeError("verify requires a StatisticalLeakageAuditReportV2")
    return statistical_leakage_audit_v2_from_obj(report.as_obj(), expected_digest=report.digest)


def serialize_statistical_leakage_audit_v2(report: StatisticalLeakageAuditReportV2) -> str:
    return _dump_json(verify_statistical_leakage_audit_v2(report).as_obj())


def parse_statistical_leakage_audit_v2(
    text: str,
    *,
    expected_digest: str,
) -> StatisticalLeakageAuditReportV2:
    report = statistical_leakage_audit_v2_from_obj(_load_json(text), expected_digest=expected_digest)
    if serialize_statistical_leakage_audit_v2(report) != text:
        raise StatisticalLeakageV2Error("statistical leakage JSON is not canonical compact serialization")
    return report
