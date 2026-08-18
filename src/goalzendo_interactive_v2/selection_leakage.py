"""Exact selection-channel leakage audit for prospective G03-v2 terminals.

The public terminal generator may depend on the hidden Official Law.  Even
when terminal items are presented one at a time, that dependence can reveal
the Law through the selected scene itself.  This module computes the exact
Bayes top-one identification accuracy induced by the public rational law
``P(scene | rule)`` under the registered uniform live-rule prior.

This analytic audit is deliberately distinct from the powered grouped
surface-leakage audit over a materialized bank.  Both are required before a
scientific launch; neither report authorizes model loading or a weight update.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, cast

from goalzendo_interactive.catalog import RuleCatalog, VersionSpace
from goalzendo_interactive.schema import SCENE_COUNT

from .population_audit import build_supported_catalog_contract_v2
from .reward_query import TerminalItemGeneratorLawV2

SELECTION_LEAKAGE_SCHEMA_VERSION = 1
SELECTION_LEAKAGE_CHANCE_MARGIN = Fraction(1, 20)

_REPORT_KIND = "g03-v2-exact-terminal-selection-channel-leakage"
_REPORT_DOMAIN = "goalzendo-interactive-v2-selection-leakage-v1"
_TIE_CONTRACT = (
    "sum, for every observable cell, the largest exact joint mass under the "
    "uniform live-rule prior; posterior ties therefore receive their exact "
    "Bayes-optimal top-one mass without an identity-dependent tie rule"
)
_AUTHORIZATION = {
    "scope": "exact_public_generator_selection_channel_audit_only",
    "materialized_bank_surface_audit_passed": False,
    "production_bank_authorized": False,
    "model_execution_authorized": False,
    "weight_updates_authorized": False,
}


class SelectionLeakageV2Error(ValueError):
    """Raised when exact G03-v2 selection-leakage evidence is invalid."""


def _dump_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise SelectionLeakageV2Error(f"value is not canonical JSON: {exc}") from exc


def _load_json(text: str) -> Any:
    if type(text) is not str or not text:
        raise SelectionLeakageV2Error("JSON input must be a nonempty string")

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SelectionLeakageV2Error(f"duplicate JSON object key: {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise SelectionLeakageV2Error(f"non-finite JSON constant is forbidden: {value}")

    try:
        return json.loads(
            text,
            object_pairs_hook=no_duplicates,
            parse_constant=reject_constant,
        )
    except SelectionLeakageV2Error:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SelectionLeakageV2Error(f"invalid JSON: {exc}") from exc


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


def _fraction_obj(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _fraction_from_obj(value: object, *, name: str) -> Fraction:
    if not isinstance(value, Mapping) or set(value) != {"numerator", "denominator"}:
        raise SelectionLeakageV2Error(f"{name} must be an exact fraction object")
    numerator = value["numerator"]
    denominator = value["denominator"]
    if type(numerator) is not int or type(denominator) is not int or denominator <= 0:
        raise SelectionLeakageV2Error(f"{name} contains invalid exact integers")
    result = Fraction(numerator, denominator)
    if result.numerator != numerator or result.denominator != denominator:
        raise SelectionLeakageV2Error(f"{name} is not in canonical lowest terms")
    return result


def _require_exact_keys(value: object, expected: set[str], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise SelectionLeakageV2Error(f"{name} must be an object with string keys")
    keys = set(value)
    if keys != expected:
        raise SelectionLeakageV2Error(
            f"{name} keys differ: missing={sorted(expected - keys)}, extra={sorted(keys - expected)}"
        )
    return cast(Mapping[str, Any], value)


def _rule_bindings(catalog: RuleCatalog, indices: tuple[int, ...]) -> list[dict[str, str]]:
    return [
        {
            "rule_id": catalog[index].rule_id,
            "truth_digest": catalog[index].truth_digest,
        }
        for index in indices
    ]


def _validate_inputs(
    space: VersionSpace,
    law: TerminalItemGeneratorLawV2,
) -> tuple[RuleCatalog, tuple[int, ...], dict[int, dict[int, Fraction]]]:
    if type(space) is not VersionSpace or not space.indices:
        raise SelectionLeakageV2Error("selection audit requires a nonempty VersionSpace")
    if type(law) is not TerminalItemGeneratorLawV2:
        raise TypeError("law must be a TerminalItemGeneratorLawV2")
    catalog = space.catalog
    contract = build_supported_catalog_contract_v2()
    if catalog.digest != contract.source_catalog_digest:
        raise SelectionLeakageV2Error("version space uses the wrong source catalog")
    supported = set(contract.supported_indices)
    if any(index not in supported for index in space.indices):
        raise SelectionLeakageV2Error("version space contains an unsupported v2 rule")
    if law.catalog_digest != catalog.digest:
        raise SelectionLeakageV2Error("generator law uses the wrong source catalog")
    if law.supported_catalog_digest != contract.supported_catalog_digest:
        raise SelectionLeakageV2Error("generator law uses the wrong supported catalog")
    expected_rule_ids = tuple(catalog[index].rule_id for index in space.indices)
    if tuple(item.rule_id for item in law.conditional_rules) != expected_rule_ids:
        raise SelectionLeakageV2Error("generator law does not bind exactly the live rules")
    expected_truth_digests = tuple(catalog[index].truth_digest for index in space.indices)
    if tuple(item.truth_digest for item in law.conditional_rules) != expected_truth_digests:
        raise SelectionLeakageV2Error("generator law conditional truth digests differ from the live catalog")
    maps = law.probability_maps()
    if set(maps) != set(space.indices):
        raise SelectionLeakageV2Error("generator law probability maps differ from the live rules")
    return catalog, space.indices, maps


def _bayes_top_one_accuracy(
    probability_maps: Mapping[int, Mapping[int, Fraction]],
    observations: Mapping[int, int],
) -> Fraction:
    """Return exact Bayes top-one accuracy for an observation partition.

    ``observations`` maps every scene in the union support to an observable
    cell.  The full-scene oracle uses the scene index itself; the semantic
    oracle uses the complete live-rule truth pattern.
    """

    indices = tuple(sorted(probability_maps))
    if not indices:
        raise SelectionLeakageV2Error("Bayes audit requires at least one live rule")
    prior = Fraction(1, len(indices))
    masses: dict[int, dict[int, Fraction]] = defaultdict(lambda: defaultdict(Fraction))
    for rule_index in indices:
        for scene_index, probability in probability_maps[rule_index].items():
            if scene_index not in observations:
                raise SelectionLeakageV2Error("observation map omits a generator-law scene")
            masses[observations[scene_index]][rule_index] += prior * probability
    return sum(
        (max(rule_masses.values()) for rule_masses in masses.values()),
        Fraction(),
    )


def _truth_pattern_observations(
    catalog: RuleCatalog,
    indices: tuple[int, ...],
    scenes: tuple[int, ...],
) -> dict[int, int]:
    result: dict[int, int] = {}
    for scene_index in scenes:
        if not 0 <= scene_index < SCENE_COUNT:
            raise SelectionLeakageV2Error("generator law contains an invalid scene index")
        mask = 0
        for position, rule_index in enumerate(indices):
            if catalog[rule_index].truth[scene_index]:
                mask |= 1 << position
        result[scene_index] = mask
    return result


def _maximum_pairwise_total_variation(
    probability_maps: Mapping[int, Mapping[int, Fraction]],
) -> Fraction:
    indices = tuple(sorted(probability_maps))
    maximum = Fraction()
    for left_position, left in enumerate(indices):
        for right in indices[left_position + 1 :]:
            scenes = set(probability_maps[left]) | set(probability_maps[right])
            distance = (
                sum(
                    (
                        abs(
                            probability_maps[left].get(scene, Fraction())
                            - probability_maps[right].get(scene, Fraction())
                        )
                        for scene in scenes
                    ),
                    Fraction(),
                )
                / 2
            )
            maximum = max(maximum, distance)
    return maximum


@dataclass(frozen=True, slots=True)
class ExactTerminalSelectionLeakageReportV2:
    """Exact inference channel induced by one public terminal-item draw."""

    catalog_digest: str
    supported_catalog_digest: str
    live_rule_bindings: tuple[tuple[str, str], ...]
    generator_law_digest: str
    public_derivation_attestation_digest: str
    chance_accuracy: Fraction
    full_scene_bayes_accuracy: Fraction
    truth_pattern_bayes_accuracy: Fraction
    maximum_pairwise_total_variation: Fraction
    official_independent: bool

    def __post_init__(self) -> None:
        for name in (
            "catalog_digest",
            "supported_catalog_digest",
            "generator_law_digest",
            "public_derivation_attestation_digest",
        ):
            if not _is_sha256(getattr(self, name)):
                raise SelectionLeakageV2Error(f"{name} must be a SHA-256")
        if len(self.live_rule_bindings) < 2:
            raise SelectionLeakageV2Error("selection audit requires at least two live rules")
        if tuple(rule_id for rule_id, _ in self.live_rule_bindings) != tuple(
            sorted({rule_id for rule_id, _ in self.live_rule_bindings})
        ):
            raise SelectionLeakageV2Error("live-rule bindings must be sorted and unique")
        if any(not _is_sha256(truth_digest) for _, truth_digest in self.live_rule_bindings):
            raise SelectionLeakageV2Error("live-rule truth digest is malformed")
        for name in (
            "chance_accuracy",
            "full_scene_bayes_accuracy",
            "truth_pattern_bayes_accuracy",
            "maximum_pairwise_total_variation",
        ):
            value = getattr(self, name)
            if type(value) is not Fraction or not 0 <= value <= 1:
                raise SelectionLeakageV2Error(f"{name} must be an exact probability")
        if self.full_scene_bayes_accuracy < self.truth_pattern_bayes_accuracy:
            raise SelectionLeakageV2Error("full-scene Bayes accuracy cannot trail its coarsening")
        if self.truth_pattern_bayes_accuracy < self.chance_accuracy:
            raise SelectionLeakageV2Error("Bayes accuracy cannot trail uniform chance")
        if self.official_independent != (self.maximum_pairwise_total_variation == 0):
            raise SelectionLeakageV2Error("Official-independence flag disagrees with total variation")

    @property
    def live_rule_count(self) -> int:
        return len(self.live_rule_bindings)

    @property
    def full_scene_excess(self) -> Fraction:
        return self.full_scene_bayes_accuracy - self.chance_accuracy

    @property
    def truth_pattern_excess(self) -> Fraction:
        return self.truth_pattern_bayes_accuracy - self.chance_accuracy

    @property
    def passed(self) -> bool:
        return (
            self.full_scene_excess < SELECTION_LEAKAGE_CHANCE_MARGIN
            and self.truth_pattern_excess < SELECTION_LEAKAGE_CHANCE_MARGIN
        )

    @property
    def digest(self) -> str:
        return _json_digest(self.as_obj(), domain=_REPORT_DOMAIN)

    def as_obj(self) -> dict[str, Any]:
        return {
            "schema_version": SELECTION_LEAKAGE_SCHEMA_VERSION,
            "report_kind": _REPORT_KIND,
            "authorization": dict(_AUTHORIZATION),
            "catalog_digest": self.catalog_digest,
            "supported_catalog_digest": self.supported_catalog_digest,
            "live_rule_count": self.live_rule_count,
            "live_rule_bindings": [
                {"rule_id": rule_id, "truth_digest": truth_digest}
                for rule_id, truth_digest in self.live_rule_bindings
            ],
            "generator_law_digest": self.generator_law_digest,
            "public_derivation_attestation_digest": self.public_derivation_attestation_digest,
            "public_derivation_attestation_externally_verified": False,
            "uniform_live_rule_prior": True,
            "secret_realized_panel_support_used": False,
            "observable_channels": {
                "full_scene_identity": "complete isolated rendered scene semantics",
                "truth_pattern": "complete live-rule truth pattern of the isolated scene",
            },
            "tie_contract": _TIE_CONTRACT,
            "chance_margin": _fraction_obj(SELECTION_LEAKAGE_CHANCE_MARGIN),
            "chance_accuracy": _fraction_obj(self.chance_accuracy),
            "full_scene_bayes_accuracy": _fraction_obj(self.full_scene_bayes_accuracy),
            "full_scene_excess": _fraction_obj(self.full_scene_excess),
            "truth_pattern_bayes_accuracy": _fraction_obj(self.truth_pattern_bayes_accuracy),
            "truth_pattern_excess": _fraction_obj(self.truth_pattern_excess),
            "maximum_pairwise_total_variation": _fraction_obj(self.maximum_pairwise_total_variation),
            "official_independent": self.official_independent,
            "passed": self.passed,
            "powered_materialized_bank_surface_audit_still_required": True,
        }


def build_exact_terminal_selection_leakage_report_v2(
    space: VersionSpace,
    law: TerminalItemGeneratorLawV2,
) -> ExactTerminalSelectionLeakageReportV2:
    """Compute the exact one-draw Official-Law identification channel."""

    catalog, indices, probability_maps = _validate_inputs(space, law)
    if len(indices) < 2:
        raise SelectionLeakageV2Error("selection leakage is undefined for a singleton version space")
    scenes = tuple(sorted({scene for values in probability_maps.values() for scene in values}))
    full_observations = {scene: scene for scene in scenes}
    truth_observations = _truth_pattern_observations(catalog, indices, scenes)
    full_accuracy = _bayes_top_one_accuracy(probability_maps, full_observations)
    truth_accuracy = _bayes_top_one_accuracy(probability_maps, truth_observations)
    maximum_tv = _maximum_pairwise_total_variation(probability_maps)
    contract = build_supported_catalog_contract_v2()
    return ExactTerminalSelectionLeakageReportV2(
        catalog_digest=catalog.digest,
        supported_catalog_digest=contract.supported_catalog_digest,
        live_rule_bindings=tuple((catalog[index].rule_id, catalog[index].truth_digest) for index in indices),
        generator_law_digest=law.digest,
        public_derivation_attestation_digest=law.public_derivation_attestation_digest,
        chance_accuracy=Fraction(1, len(indices)),
        full_scene_bayes_accuracy=full_accuracy,
        truth_pattern_bayes_accuracy=truth_accuracy,
        maximum_pairwise_total_variation=maximum_tv,
        official_independent=maximum_tv == 0,
    )


def verify_exact_terminal_selection_leakage_report_v2(
    report: ExactTerminalSelectionLeakageReportV2,
    space: VersionSpace,
    law: TerminalItemGeneratorLawV2,
) -> None:
    """Regenerate and byte-compare an exact selection-leakage report."""

    if type(report) is not ExactTerminalSelectionLeakageReportV2:
        raise TypeError("report must be an ExactTerminalSelectionLeakageReportV2")
    expected = build_exact_terminal_selection_leakage_report_v2(space, law)
    if report.as_obj() != expected.as_obj():
        raise SelectionLeakageV2Error("selection-leakage report differs from exact regeneration")


def serialize_exact_terminal_selection_leakage_report_v2(
    report: ExactTerminalSelectionLeakageReportV2,
) -> str:
    if type(report) is not ExactTerminalSelectionLeakageReportV2:
        raise TypeError("report must be an ExactTerminalSelectionLeakageReportV2")
    return _dump_json(report.as_obj())


def _report_from_obj(
    value: object,
) -> ExactTerminalSelectionLeakageReportV2:
    obj = _require_exact_keys(
        value,
        {
            "schema_version",
            "report_kind",
            "authorization",
            "catalog_digest",
            "supported_catalog_digest",
            "live_rule_count",
            "live_rule_bindings",
            "generator_law_digest",
            "public_derivation_attestation_digest",
            "public_derivation_attestation_externally_verified",
            "uniform_live_rule_prior",
            "secret_realized_panel_support_used",
            "observable_channels",
            "tie_contract",
            "chance_margin",
            "chance_accuracy",
            "full_scene_bayes_accuracy",
            "full_scene_excess",
            "truth_pattern_bayes_accuracy",
            "truth_pattern_excess",
            "maximum_pairwise_total_variation",
            "official_independent",
            "passed",
            "powered_materialized_bank_surface_audit_still_required",
        },
        name="selection-leakage report",
    )
    if type(obj["schema_version"]) is not int or obj["schema_version"] != SELECTION_LEAKAGE_SCHEMA_VERSION:
        raise SelectionLeakageV2Error("selection-leakage schema version is invalid")
    if obj["report_kind"] != _REPORT_KIND:
        raise SelectionLeakageV2Error("selection-leakage report kind is invalid")
    if obj["authorization"] != _AUTHORIZATION:
        raise SelectionLeakageV2Error("selection-leakage authorization boundary differs")
    bindings_obj = obj["live_rule_bindings"]
    if type(bindings_obj) is not list:
        raise SelectionLeakageV2Error("live-rule bindings must be a list")
    bindings: list[tuple[str, str]] = []
    for item in bindings_obj:
        binding = _require_exact_keys(item, {"rule_id", "truth_digest"}, name="rule binding")
        if type(binding["rule_id"]) is not str or type(binding["truth_digest"]) is not str:
            raise SelectionLeakageV2Error("rule binding fields must be strings")
        bindings.append((binding["rule_id"], binding["truth_digest"]))
    if type(obj["official_independent"]) is not bool:
        raise SelectionLeakageV2Error("Official-independence flag must be Boolean")
    result = ExactTerminalSelectionLeakageReportV2(
        catalog_digest=cast(str, obj["catalog_digest"]),
        supported_catalog_digest=cast(str, obj["supported_catalog_digest"]),
        live_rule_bindings=tuple(bindings),
        generator_law_digest=cast(str, obj["generator_law_digest"]),
        public_derivation_attestation_digest=cast(str, obj["public_derivation_attestation_digest"]),
        chance_accuracy=_fraction_from_obj(obj["chance_accuracy"], name="chance_accuracy"),
        full_scene_bayes_accuracy=_fraction_from_obj(
            obj["full_scene_bayes_accuracy"], name="full_scene_bayes_accuracy"
        ),
        truth_pattern_bayes_accuracy=_fraction_from_obj(
            obj["truth_pattern_bayes_accuracy"], name="truth_pattern_bayes_accuracy"
        ),
        maximum_pairwise_total_variation=_fraction_from_obj(
            obj["maximum_pairwise_total_variation"],
            name="maximum_pairwise_total_variation",
        ),
        official_independent=obj["official_independent"],
    )
    if type(obj["live_rule_count"]) is not int or obj["live_rule_count"] != result.live_rule_count:
        raise SelectionLeakageV2Error("live-rule count differs from bindings")
    if obj["uniform_live_rule_prior"] is not True:
        raise SelectionLeakageV2Error("uniform prior attestation is invalid")
    if obj["public_derivation_attestation_externally_verified"] is not False:
        raise SelectionLeakageV2Error(
            "unverified public-derivation evidence cannot be promoted by this report"
        )
    if obj["secret_realized_panel_support_used"] is not False:
        raise SelectionLeakageV2Error("secret realized support is forbidden")
    if obj["observable_channels"] != result.as_obj()["observable_channels"]:
        raise SelectionLeakageV2Error("observable-channel contract differs")
    if obj["tie_contract"] != _TIE_CONTRACT:
        raise SelectionLeakageV2Error("tie contract differs")
    if _fraction_from_obj(obj["chance_margin"], name="chance_margin") != SELECTION_LEAKAGE_CHANCE_MARGIN:
        raise SelectionLeakageV2Error("chance margin differs")
    for name, expected in (
        ("full_scene_excess", result.full_scene_excess),
        ("truth_pattern_excess", result.truth_pattern_excess),
    ):
        if _fraction_from_obj(obj[name], name=name) != expected:
            raise SelectionLeakageV2Error(f"{name} differs from exact fields")
    if type(obj["passed"]) is not bool or obj["passed"] != result.passed:
        raise SelectionLeakageV2Error("selection-leakage pass flag differs")
    if obj["powered_materialized_bank_surface_audit_still_required"] is not True:
        raise SelectionLeakageV2Error("powered bank audit requirement was removed")
    if _dump_json(result.as_obj()) != _dump_json(value):
        raise SelectionLeakageV2Error("selection-leakage object is not canonical")
    return result


def parse_exact_terminal_selection_leakage_report_v2(
    text: str,
    space: VersionSpace,
    law: TerminalItemGeneratorLawV2,
) -> ExactTerminalSelectionLeakageReportV2:
    """Parse canonically, then regenerate from the exact public law."""

    value = _load_json(text)
    if _dump_json(value) != text:
        raise SelectionLeakageV2Error("selection-leakage JSON is not canonical")
    result = _report_from_obj(value)
    verify_exact_terminal_selection_leakage_report_v2(result, space, law)
    return result
