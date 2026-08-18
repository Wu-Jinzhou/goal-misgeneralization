"""Exact composite-reward query-policy ceiling for prospective G03-v2.

This module is additive and nonauthorizing.  It solves the finite-horizon
Bayes decision problem induced by the registered G03 reward, a uniform prior
over an exact live version space, and a public exact per-item generator law
``P(scene | rule)``.  Unlike the identification-only ceiling, every state
compares an immediate stop action with every legal informative query.

All arithmetic that can affect an action is exact :class:`fractions.Fraction`
arithmetic.  Canonical reports bind the catalog, live rules, generator law,
query exclusions, reward, budget, information set, and deterministic tie
rules.  Neither the Official Law nor secret realized reservoir state is
available to the query policy.  The Official is used only to replay the
already-computed policy's target-conditional path.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Literal, cast

from goalzendo_interactive import (
    SCENE_COUNT,
    CatalogEntry,
    RuleCatalog,
    Scene,
    VersionSpace,
    build_rule_catalog,
    scene_index,
)
from goalzendo_interactive_v2.population_audit import (
    PopulationAuditV2Error,
    build_supported_catalog_contract_v2,
    require_supported_catalog_identity_v2,
)

REWARD_QUERY_POLICY_SCHEMA_VERSION = 2
MAX_REWARD_QUERY_BUDGET = 6
MAX_REWARD_VERSION_SPACE_SIZE = 16
TERMINAL_ITEM_COUNT = 16

_REPORT_KIND = "g03-v2-exact-composite-reward-query-policy-ceiling-v2"
_REPORT_DOMAIN = "goalzendo-interactive-v2-reward-query-report-v2"
_GENERATOR_LAW_DOMAIN = "goalzendo-interactive-v2-terminal-item-generator-law-v2"
_EXCLUSIONS_DOMAIN = "goalzendo-interactive-v2-reward-query-exclusions-v1"
_REWARD_DOMAIN = "goalzendo-interactive-v2-reward-specification-v1"
_TIE_RULES_DOMAIN = "goalzendo-interactive-v2-reward-tie-rules-v1"
_INFORMATION_SET_DOMAIN = "goalzendo-interactive-v2-reward-information-set-v1"
_QUERY_TIE_CONTEXT_DOMAIN = "goalzendo-interactive-v2-reward-query-tie-context-v2"
_POLICY_CONTEXT_DOMAIN = "goalzendo-interactive-v2-reward-policy-context-v2"
_REPORT_BINDING_DOMAIN = "goalzendo-interactive-v2-reward-report-binding-v2"
_ROOT_PARTITIONS_DOMAIN = "goalzendo-interactive-v2-reward-root-partitions-v2"
_ITEM_POLICY_DOMAIN = "goalzendo-interactive-v2-reward-terminal-item-policy-v2"
_QUERY_TIE_DOMAIN = b"goalzendo-interactive-v2-reward-query-scene-tie-v2\0"
_MAP_TIE_DOMAIN = b"goalzendo-interactive-v2-reward-map-rule-tie-v2\0"

_AUTHORIZATION = {
    "scope": "prospective-engineering-evaluation-only",
    "production_bank_materialized": False,
    "weight_updates_authorized": False,
}

_REWARD_SPECIFICATION = {
    "formula": "(7/10)*terminal_accuracy+(1/4)*exact_rule+(1/20)*(1-queries/6)",
    "terminal_accuracy_weight": {"numerator": 7, "denominator": 10},
    "exact_rule_weight": {"numerator": 1, "denominator": 4},
    "query_efficiency_weight": {"numerator": 1, "denominator": 20},
    "query_efficiency_denominator": 6,
}

_TIE_RULES = {
    "posterior": "uniform-over-surviving-rules",
    "majority_label_tie": "predict-rejected-false",
    "map_rule_tie": "minimum-domain-separated-sha256-then-rule-id",
    "query_partition_equivalence": "unordered-accepted-rejected-posterior-split",
    "query_representative_tie": "minimum-domain-separated-sha256-then-scene-index",
    "equal_query_value_tie": ("minimum-representative-tie-digest-then-scene-index-then-partition-mask"),
    "equal_stop_query_value_tie": "prefer-stop",
    "identification_leaf_tie": "same-query-tie-rule",
}

_INFORMATION_SET = {
    "query_policy_observes": [
        "public-catalog-and-live-version-space",
        "query-feedback-history-through-current-posterior",
        "remaining-query-budget-and-query-count",
        "public-exact-terminal-item-generator-law-P(scene|rule)",
        "public-model-visible-legal-query-exclusion-set",
    ],
    "query_policy_does_not_observe": [
        "Official-Law-identity",
        "secret-reservoir-panels-salt-and-realized-item-set",
        "secret-query-exclusions-derived-from-reservoir-or-Official",
        "future-query-feedback",
    ],
    "terminal_classifier_observes": [
        "locked-inquiry-transcript-and-current-posterior",
        "one-terminal-scene-when-presented",
    ],
    "terminal_classifier_does_not_observe": [
        "panel-membership-or-support-rank",
        "item-order",
        "other-terminal-items",
        "prior-terminal-answer-prefix",
        "submitted-AST-or-rule-rationale",
    ],
    "submitted_rule_observes": ["locked-inquiry-transcript-and-current-posterior"],
    "submitted_rule_does_not_observe": ["any-terminal-scene-or-secret-reservoir-state"],
    "event_order": [
        "inquiry",
        "ready",
        "fork-private-sibling-submit-and-irrevocably-lock-AST",
        "draw-backend-items-under-public-generator-law",
        "fork-independent-classification-siblings-from-sealed-pre-AST-transcript",
    ],
}

ActionKind = Literal["stop", "query"]


class RewardQueryV2Error(ValueError):
    """Raised when a reward-query report cannot be built or replayed exactly."""


def _dump_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise RewardQueryV2Error(f"value is not canonical JSON: {exc}") from exc


def _load_json(text: str) -> Any:
    if type(text) is not str or not text:
        raise RewardQueryV2Error("JSON input must be a nonempty string")

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RewardQueryV2Error(f"duplicate JSON object key: {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise RewardQueryV2Error(f"non-finite JSON constant is forbidden: {value}")

    try:
        return json.loads(
            text,
            object_pairs_hook=no_duplicates,
            parse_constant=reject_constant,
        )
    except RewardQueryV2Error:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RewardQueryV2Error(f"invalid JSON: {exc}") from exc


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
        raise RewardQueryV2Error(f"{name} has noncanonical fields")
    return cast(Mapping[str, Any], value)


def _bounded_integer(
    value: object,
    *,
    name: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RewardQueryV2Error(f"{name} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise RewardQueryV2Error(f"{name} must be an integer <= {maximum}")
    return value


def _fraction_obj(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _normalize_scene_index(value: int | Scene, *, name: str) -> int:
    index = scene_index(value) if type(value) is Scene else value
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < SCENE_COUNT:
        raise RewardQueryV2Error(f"{name} must lie in [0, {SCENE_COUNT})")
    return index


@dataclass(frozen=True, slots=True)
class TerminalSceneProbabilityV2:
    """One positive exact probability in a rule-conditional item law."""

    scene_index: int
    probability: Fraction

    def __post_init__(self) -> None:
        _normalize_scene_index(self.scene_index, name="generator-law scene index")
        if type(self.probability) is not Fraction or not 0 < self.probability <= 1:
            raise RewardQueryV2Error("generator-law probability must be an exact fraction in (0, 1]")

    def as_obj(self) -> dict[str, Any]:
        return {
            "scene_index": self.scene_index,
            "probability": _fraction_obj(self.probability),
        }


@dataclass(frozen=True, slots=True)
class RuleConditionalItemLawV2:
    """The public exact one-item marginal P(scene | candidate rule)."""

    rule_id: str
    truth_digest: str
    scene_probabilities: tuple[TerminalSceneProbabilityV2, ...]

    def __post_init__(self) -> None:
        if len(self.rule_id) != 9 or not self.rule_id.startswith("g03r") or not self.rule_id[4:].isdigit():
            raise RewardQueryV2Error("conditional generator law has an invalid rule id")
        if not _is_sha256(self.truth_digest):
            raise RewardQueryV2Error("conditional generator law truth digest is invalid")
        if not self.scene_probabilities or tuple(
            probability.scene_index for probability in self.scene_probabilities
        ) != tuple(sorted({probability.scene_index for probability in self.scene_probabilities})):
            raise RewardQueryV2Error("generator-law scene support must be nonempty, sorted, and unique")
        if (
            sum(
                (probability.probability for probability in self.scene_probabilities),
                Fraction(),
            )
            != 1
        ):
            raise RewardQueryV2Error("each rule-conditional generator law must sum exactly to one")

    def as_obj(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "truth_digest": self.truth_digest,
            "scene_support_count": len(self.scene_probabilities),
            "scene_probabilities": [value.as_obj() for value in self.scene_probabilities],
        }


@dataclass(frozen=True, slots=True)
class TerminalItemGeneratorLawV2:
    """Public rule-conditional one-item law used by the implementable policy."""

    catalog_digest: str
    supported_catalog_digest: str
    public_derivation_attestation_digest: str
    conditional_rules: tuple[RuleConditionalItemLawV2, ...]

    def __post_init__(self) -> None:
        if not _is_sha256(self.catalog_digest):
            raise RewardQueryV2Error("generator-law catalog digest must be a SHA-256")
        if not _is_sha256(self.supported_catalog_digest):
            raise RewardQueryV2Error("generator-law supported-catalog digest must be a SHA-256")
        if not _is_sha256(self.public_derivation_attestation_digest):
            raise RewardQueryV2Error("generator-law public derivation digest must be a SHA-256")
        if not self.conditional_rules or tuple(item.rule_id for item in self.conditional_rules) != tuple(
            sorted({item.rule_id for item in self.conditional_rules})
        ):
            raise RewardQueryV2Error("generator-law rules must be nonempty, sorted, and unique")

    @property
    def digest(self) -> str:
        return _json_digest(self.as_obj(), domain=_GENERATOR_LAW_DOMAIN)

    @property
    def union_scene_support(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                {
                    value.scene_index
                    for conditional in self.conditional_rules
                    for value in conditional.scene_probabilities
                }
            )
        )

    def as_obj(self) -> dict[str, Any]:
        return {
            "law_kind": "public-exact-rule-conditional-one-item-marginal",
            "catalog_digest": self.catalog_digest,
            "supported_catalog_digest": self.supported_catalog_digest,
            "public_derivation_attestation_digest": self.public_derivation_attestation_digest,
            "derivation_contract": (
                "model-visible-opening-plus-public-generator-with-evaluator-private-latents-marginalized"
            ),
            "external_generator_verification_required": True,
            "backend_item_count": TERMINAL_ITEM_COUNT,
            "secret_realized_support_included": False,
            "conditional_rules": [item.as_obj() for item in self.conditional_rules],
            "union_scene_support_count": len(self.union_scene_support),
        }

    def probability_maps(self) -> dict[int, dict[int, Fraction]]:
        return {
            int(conditional.rule_id[4:]): {
                value.scene_index: value.probability for value in conditional.scene_probabilities
            }
            for conditional in self.conditional_rules
        }


TerminalItemGeneratorInput = Mapping[int, Mapping[int, Fraction]]


def build_terminal_item_generator_law_v2(
    space: VersionSpace,
    probabilities: TerminalItemGeneratorInput,
    *,
    public_derivation_attestation_digest: str,
) -> TerminalItemGeneratorLawV2:
    """Canonicalize and bind a public exact P(scene | rule) law."""

    if type(space) is not VersionSpace or not space.indices:
        raise RewardQueryV2Error("generator law requires a nonempty VersionSpace")
    _validate_context(space, space.catalog[space.indices[0]])
    if not _is_sha256(public_derivation_attestation_digest):
        raise RewardQueryV2Error("public derivation attestation digest must be a SHA-256")
    if not isinstance(probabilities, Mapping):
        raise TypeError("probabilities must be a mapping")
    raw_keys = tuple(probabilities)
    if any(isinstance(index, bool) or not isinstance(index, int) for index in raw_keys):
        raise RewardQueryV2Error("generator-law rule keys must be catalog indices")
    if set(raw_keys) != set(space.indices) or len(raw_keys) != len(space.indices):
        raise RewardQueryV2Error("generator law must contain exactly every live version-space rule")
    conditionals: list[RuleConditionalItemLawV2] = []
    for index in space.indices:
        raw_scene_map = probabilities[index]
        if not isinstance(raw_scene_map, Mapping) or not raw_scene_map:
            raise RewardQueryV2Error("each generator-law rule requires a nonempty scene map")
        scene_values: list[TerminalSceneProbabilityV2] = []
        for raw_scene, probability in raw_scene_map.items():
            scene = _normalize_scene_index(raw_scene, name="generator-law scene index")
            scene_values.append(TerminalSceneProbabilityV2(scene, probability))
        scene_values.sort(key=lambda item: item.scene_index)
        entry = space.catalog[index]
        conditionals.append(
            RuleConditionalItemLawV2(
                rule_id=entry.rule_id,
                truth_digest=entry.truth_digest,
                scene_probabilities=tuple(scene_values),
            )
        )
    contract = build_supported_catalog_contract_v2()
    return TerminalItemGeneratorLawV2(
        space.catalog.digest,
        contract.supported_catalog_digest,
        public_derivation_attestation_digest,
        tuple(conditionals),
    )


def _normalize_exclusions(values: Iterable[int | Scene]) -> tuple[int, ...]:
    raw = tuple(_normalize_scene_index(value, name="excluded query scene index") for value in values)
    if len(set(raw)) != len(raw):
        raise RewardQueryV2Error("excluded query scene indices must be unique")
    return tuple(sorted(raw))


def _rule_bindings(catalog: RuleCatalog, indices: Iterable[int]) -> list[dict[str, str]]:
    return [
        {
            "rule_id": catalog[index].rule_id,
            "truth_digest": catalog[index].truth_digest,
        }
        for index in indices
    ]


def _validate_context(
    space: VersionSpace,
    official: CatalogEntry,
) -> tuple[RuleCatalog, tuple[int, ...], int]:
    if type(space) is not VersionSpace:
        raise TypeError("space must be a VersionSpace")
    if type(official) is not CatalogEntry:
        raise TypeError("official must be a CatalogEntry")
    if not space.indices:
        raise RewardQueryV2Error("version space cannot be empty")
    if len(space) > MAX_REWARD_VERSION_SPACE_SIZE:
        raise RewardQueryV2Error(
            "exact reward search requires at most "
            f"{MAX_REWARD_VERSION_SPACE_SIZE} live rules; received {len(space)}"
        )
    catalog = space.catalog
    contract = build_supported_catalog_contract_v2()
    if contract.source_catalog_digest != catalog.digest:
        raise RewardQueryV2Error("version-space catalog differs from the v2 supported-catalog source")
    supported = set(contract.supported_indices)
    for index in space.indices:
        if index not in supported:
            entry = catalog[index]
            try:
                require_supported_catalog_identity_v2(entry)
            except PopulationAuditV2Error as exc:
                raise RewardQueryV2Error(
                    f"unsupported v2 version-space identity {entry.rule_id}: {exc}"
                ) from exc
            raise RewardQueryV2Error(
                f"version-space identity {entry.rule_id} is absent from the supported allowlist"
            )
    if not 0 <= official.index < len(catalog) or catalog[official.index] != official:
        raise RewardQueryV2Error("Official Law is not bound to the version-space catalog")
    if official.index not in space.indices:
        raise RewardQueryV2Error("Official Law is absent from the supplied version space")
    return catalog, space.indices, space.indices.index(official.index)


def _exclusions_digest(exclusions: tuple[int, ...]) -> str:
    return _json_digest(
        {"scene_count": SCENE_COUNT, "excluded_query_scene_indices": list(exclusions)},
        domain=_EXCLUSIONS_DOMAIN,
    )


def _reward_digest() -> str:
    return _json_digest(_REWARD_SPECIFICATION, domain=_REWARD_DOMAIN)


def _tie_rules_digest() -> str:
    return _json_digest(_TIE_RULES, domain=_TIE_RULES_DOMAIN)


def _information_set_digest() -> str:
    return _json_digest(_INFORMATION_SET, domain=_INFORMATION_SET_DOMAIN)


def _hash_tie(domain: bytes, context_digest: str, payload: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(bytes.fromhex(context_digest))
    digest.update(payload)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class RewardQueryChoiceV2:
    """One deterministic representative of an informative partition class."""

    scene_index: int
    tie_break_digest: str
    equivalent_scene_count: int
    partition_mask: int
    rejected_rule_ids: tuple[str, ...]
    accepted_rule_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _bounded_integer(self.scene_index, name="query scene index", maximum=SCENE_COUNT - 1)
        if not _is_sha256(self.tie_break_digest):
            raise RewardQueryV2Error("query tie_break_digest must be a SHA-256")
        _bounded_integer(self.equivalent_scene_count, name="equivalent_scene_count", minimum=1)
        _bounded_integer(self.partition_mask, name="partition_mask", minimum=1)
        if not self.rejected_rule_ids or not self.accepted_rule_ids:
            raise RewardQueryV2Error("query choice must have two nonempty children")
        if set(self.rejected_rule_ids) & set(self.accepted_rule_ids):
            raise RewardQueryV2Error("query child rule identities must be disjoint")

    def as_obj(self) -> dict[str, Any]:
        return {
            "scene_index": self.scene_index,
            "tie_break_digest": self.tie_break_digest,
            "equivalent_scene_count": self.equivalent_scene_count,
            "partition_mask": self.partition_mask,
            "rejected_rule_ids": list(self.rejected_rule_ids),
            "accepted_rule_ids": list(self.accepted_rule_ids),
            "rejected_count": len(self.rejected_rule_ids),
            "accepted_count": len(self.accepted_rule_ids),
        }


@dataclass(frozen=True, slots=True)
class TerminalItemDecisionV2:
    """One isolated Bayes action after updating on public selection likelihood."""

    scene_index: int
    rejected_likelihood_sum: Fraction
    accepted_likelihood_sum: Fraction
    marginal_scene_probability: Fraction
    predicted_accepted: bool
    official_accepted: bool
    official_scene_probability: Fraction

    def __post_init__(self) -> None:
        _bounded_integer(self.scene_index, name="terminal item scene index", maximum=SCENE_COUNT - 1)
        if (
            type(self.rejected_likelihood_sum) is not Fraction
            or type(self.accepted_likelihood_sum) is not Fraction
            or self.rejected_likelihood_sum < 0
            or self.accepted_likelihood_sum < 0
            or self.rejected_likelihood_sum + self.accepted_likelihood_sum <= 0
        ):
            raise RewardQueryV2Error("terminal item likelihood sums are invalid")
        for name in ("marginal_scene_probability", "official_scene_probability"):
            value = getattr(self, name)
            if type(value) is not Fraction or not 0 <= value <= 1:
                raise RewardQueryV2Error(f"{name} must be an exact fraction in [0, 1]")
        if type(self.predicted_accepted) is not bool or type(self.official_accepted) is not bool:
            raise RewardQueryV2Error("terminal item labels must be Boolean")

    def as_obj(self) -> dict[str, Any]:
        return {
            "scene_index": self.scene_index,
            "rejected_likelihood_sum": _fraction_obj(self.rejected_likelihood_sum),
            "accepted_likelihood_sum": _fraction_obj(self.accepted_likelihood_sum),
            "marginal_scene_probability": _fraction_obj(self.marginal_scene_probability),
            "predicted_accepted": self.predicted_accepted,
            "official_accepted": self.official_accepted,
            "official_scene_probability": _fraction_obj(self.official_scene_probability),
        }


@dataclass(frozen=True, slots=True)
class RewardStopDecisionV2:
    """Bayes-optimal isolated classifications and pre-reveal locked MAP AST."""

    query_count: int
    posterior_rule_ids: tuple[str, ...]
    submitted_rule_id: str
    submitted_rule_truth_digest: str
    submitted_rule_ast: Mapping[str, Any]
    submitted_rule_tie_break_digest: str
    terminal_item_policy: tuple[TerminalItemDecisionV2, ...]
    terminal_item_policy_digest: str
    expected_terminal_accuracy: Fraction
    expected_exact_rule: Fraction
    query_efficiency: Fraction
    expected_reward: Fraction
    official_expected_terminal_accuracy: Fraction
    official_exact_rule: bool
    official_expected_reward: Fraction

    def __post_init__(self) -> None:
        _bounded_integer(self.query_count, name="query_count", maximum=MAX_REWARD_QUERY_BUDGET)
        if not self.posterior_rule_ids:
            raise RewardQueryV2Error("stop posterior cannot be empty")
        if self.submitted_rule_id not in self.posterior_rule_ids:
            raise RewardQueryV2Error("submitted MAP rule is outside the posterior")
        if not _is_sha256(self.submitted_rule_truth_digest):
            raise RewardQueryV2Error("submitted rule truth digest must be a SHA-256")
        if not _is_sha256(self.submitted_rule_tie_break_digest):
            raise RewardQueryV2Error("submitted rule tie digest must be a SHA-256")
        if not self.terminal_item_policy or tuple(
            decision.scene_index for decision in self.terminal_item_policy
        ) != tuple(sorted({decision.scene_index for decision in self.terminal_item_policy})):
            raise RewardQueryV2Error("terminal item policy must have unique sorted scene indices")
        if not _is_sha256(self.terminal_item_policy_digest):
            raise RewardQueryV2Error("terminal item policy digest must be a SHA-256")
        if type(self.official_exact_rule) is not bool:
            raise RewardQueryV2Error("official_exact_rule must be Boolean")
        for name in (
            "expected_terminal_accuracy",
            "expected_exact_rule",
            "query_efficiency",
            "expected_reward",
            "official_expected_terminal_accuracy",
            "official_expected_reward",
        ):
            value = getattr(self, name)
            if type(value) is not Fraction or not 0 <= value <= 1:
                raise RewardQueryV2Error(f"{name} must be an exact fraction in [0, 1]")

    def as_obj(self) -> dict[str, Any]:
        return {
            "query_count": self.query_count,
            "posterior_rule_ids": list(self.posterior_rule_ids),
            "posterior_rule_count": len(self.posterior_rule_ids),
            "submitted_rule_id": self.submitted_rule_id,
            "submitted_rule_truth_digest": self.submitted_rule_truth_digest,
            "submitted_rule_ast": dict(self.submitted_rule_ast),
            "submitted_rule_tie_break_digest": self.submitted_rule_tie_break_digest,
            "terminal_item_information_unit": ("sealed-pre-AST-transcript-plus-one-scene-only"),
            "terminal_item_policy": [decision.as_obj() for decision in self.terminal_item_policy],
            "terminal_item_policy_digest": self.terminal_item_policy_digest,
            "expected_terminal_accuracy": _fraction_obj(self.expected_terminal_accuracy),
            "expected_exact_rule": _fraction_obj(self.expected_exact_rule),
            "query_efficiency": _fraction_obj(self.query_efficiency),
            "expected_reward": _fraction_obj(self.expected_reward),
            "official_expected_terminal_accuracy": _fraction_obj(self.official_expected_terminal_accuracy),
            "official_exact_rule": self.official_exact_rule,
            "official_expected_reward": _fraction_obj(self.official_expected_reward),
        }


@dataclass(frozen=True, slots=True)
class RewardPolicyPathStepV2:
    """One target-Official transition under the composite-reward policy."""

    turn: int
    remaining_budget_before: int
    before_rule_ids: tuple[str, ...]
    immediate_stop_expected_reward: Fraction
    selected_query_expected_reward: Fraction
    optimal_expected_reward: Fraction
    query: RewardQueryChoiceV2
    official_accepted: bool
    after_rule_ids: tuple[str, ...]

    def as_obj(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "remaining_budget_before": self.remaining_budget_before,
            "before_rule_ids": list(self.before_rule_ids),
            "immediate_stop_expected_reward": _fraction_obj(self.immediate_stop_expected_reward),
            "selected_query_expected_reward": _fraction_obj(self.selected_query_expected_reward),
            "optimal_expected_reward": _fraction_obj(self.optimal_expected_reward),
            "query": self.query.as_obj(),
            "official_accepted": self.official_accepted,
            "after_rule_ids": list(self.after_rule_ids),
        }


@dataclass(frozen=True, slots=True)
class IdentificationPathStepV2:
    """One target-Official transition under identification-only optimization."""

    turn: int
    remaining_budget_before: int
    before_rule_ids: tuple[str, ...]
    query: RewardQueryChoiceV2
    official_accepted: bool
    after_rule_ids: tuple[str, ...]

    def as_obj(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "remaining_budget_before": self.remaining_budget_before,
            "before_rule_ids": list(self.before_rule_ids),
            "query": self.query.as_obj(),
            "official_accepted": self.official_accepted,
            "after_rule_ids": list(self.after_rule_ids),
        }


@dataclass(frozen=True, slots=True)
class IdentificationOnlySummaryV2:
    """Exact finite-budget singleton-recovery ceiling, without task reward."""

    maximum_query_budget: int
    exact_identification_probability: Fraction
    first_query: RewardQueryChoiceV2 | None
    official_path: tuple[IdentificationPathStepV2, ...]
    official_terminal_rule_ids: tuple[str, ...]

    def as_obj(self) -> dict[str, Any]:
        return {
            "objective": "maximize-exact-identification-probability-only",
            "maximum_query_budget": self.maximum_query_budget,
            "exact_identification_probability": _fraction_obj(self.exact_identification_probability),
            "first_action_kind": "stop" if self.first_query is None else "query",
            "first_query": None if self.first_query is None else self.first_query.as_obj(),
            "official_path": [step.as_obj() for step in self.official_path],
            "official_query_count": len(self.official_path),
            "official_terminal_rule_ids": list(self.official_terminal_rule_ids),
            "official_identified": self.official_terminal_rule_ids == (self.official_terminal_rule_ids[0],)
            if self.official_terminal_rule_ids
            else False,
        }


@dataclass(frozen=True, slots=True)
class CompositeRewardSummaryV2:
    """Exact task-reward ceiling plus the target-Official replay path."""

    maximum_query_budget: int
    optimal_expected_reward: Fraction
    expected_terminal_accuracy: Fraction
    expected_exact_rule: Fraction
    expected_query_efficiency: Fraction
    expected_query_count: Fraction
    immediate_stop_expected_reward: Fraction
    root_action_kind: ActionKind
    first_query: RewardQueryChoiceV2 | None
    official_path: tuple[RewardPolicyPathStepV2, ...]
    official_terminal_decision: RewardStopDecisionV2

    def __post_init__(self) -> None:
        if self.root_action_kind not in ("stop", "query"):
            raise RewardQueryV2Error("unknown composite root action")
        if (self.root_action_kind == "stop") != (self.first_query is None):
            raise RewardQueryV2Error("root action and first query are inconsistent")
        if self.optimal_expected_reward < self.immediate_stop_expected_reward:
            raise RewardQueryV2Error("optimal reward cannot be below immediate-stop reward")
        for name in (
            "expected_terminal_accuracy",
            "expected_exact_rule",
            "expected_query_efficiency",
        ):
            value = getattr(self, name)
            if type(value) is not Fraction or not 0 <= value <= 1:
                raise RewardQueryV2Error(f"{name} must be an exact fraction in [0, 1]")
        if (
            type(self.expected_query_count) is not Fraction
            or not 0 <= self.expected_query_count <= self.maximum_query_budget
        ):
            raise RewardQueryV2Error("expected_query_count is outside the policy horizon")
        recomposed = (
            Fraction(7, 10) * self.expected_terminal_accuracy
            + Fraction(1, 4) * self.expected_exact_rule
            + Fraction(1, 20) * self.expected_query_efficiency
        )
        if recomposed != self.optimal_expected_reward:
            raise RewardQueryV2Error("composite reward components do not recompose exactly")
        if self.expected_query_efficiency != 1 - self.expected_query_count / 6:
            raise RewardQueryV2Error("query efficiency and expected query count disagree")

    def as_obj(self) -> dict[str, Any]:
        return {
            "objective": "maximize-registered-composite-reward-with-explicit-stop",
            "maximum_query_budget": self.maximum_query_budget,
            "optimal_expected_reward": _fraction_obj(self.optimal_expected_reward),
            "expected_reward_components": {
                "terminal_accuracy": _fraction_obj(self.expected_terminal_accuracy),
                "exact_rule": _fraction_obj(self.expected_exact_rule),
                "query_efficiency": _fraction_obj(self.expected_query_efficiency),
                "query_count": _fraction_obj(self.expected_query_count),
            },
            "immediate_stop_expected_reward": _fraction_obj(self.immediate_stop_expected_reward),
            "query_value_over_stop": _fraction_obj(
                self.optimal_expected_reward - self.immediate_stop_expected_reward
            ),
            "root_action_kind": self.root_action_kind,
            "first_query": None if self.first_query is None else self.first_query.as_obj(),
            "official_path": [step.as_obj() for step in self.official_path],
            "official_query_count": len(self.official_path),
            "official_terminal_decision": self.official_terminal_decision.as_obj(),
        }


@dataclass(frozen=True, slots=True)
class RewardQueryPolicyCeilingReportV2:
    """Canonical exact composite-reward query-policy report."""

    catalog: RuleCatalog
    space: VersionSpace
    official: CatalogEntry
    supported_catalog_digest: str
    terminal_item_generator_law: TerminalItemGeneratorLawV2
    excluded_query_scene_indices: tuple[int, ...]
    maximum_query_budget: int
    query_tie_context_digest: str
    policy_context_digest: str
    report_binding_digest: str
    root_label_pattern_count: int
    root_informative_scene_count: int
    root_partition_class_count: int
    root_partition_classes_digest: str
    identification_only: IdentificationOnlySummaryV2
    composite_reward: CompositeRewardSummaryV2

    def __post_init__(self) -> None:
        catalog, _, _ = _validate_context(self.space, self.official)
        if self.catalog is not catalog:
            raise RewardQueryV2Error("report catalog is not the version-space catalog")
        contract = build_supported_catalog_contract_v2()
        if self.supported_catalog_digest != contract.supported_catalog_digest:
            raise RewardQueryV2Error("report has the wrong supported-catalog digest")
        if self.terminal_item_generator_law.catalog_digest != catalog.digest:
            raise RewardQueryV2Error("report generator law has the wrong catalog")
        if self.terminal_item_generator_law.supported_catalog_digest != self.supported_catalog_digest:
            raise RewardQueryV2Error("report and generator-law allowlist digests disagree")
        if tuple(item.rule_id for item in self.terminal_item_generator_law.conditional_rules) != tuple(
            catalog[index].rule_id for index in self.space.indices
        ):
            raise RewardQueryV2Error("report generator law has the wrong live rules")
        if self.excluded_query_scene_indices != tuple(sorted(set(self.excluded_query_scene_indices))):
            raise RewardQueryV2Error("report query exclusions must be sorted and unique")
        _bounded_integer(
            self.maximum_query_budget,
            name="maximum_query_budget",
            maximum=MAX_REWARD_QUERY_BUDGET,
        )
        if not _is_sha256(self.query_tie_context_digest):
            raise RewardQueryV2Error("query tie context digest must be a SHA-256")
        if not _is_sha256(self.policy_context_digest):
            raise RewardQueryV2Error("policy context digest must be a SHA-256")
        if not _is_sha256(self.report_binding_digest):
            raise RewardQueryV2Error("report binding digest must be a SHA-256")
        for name in ("root_label_pattern_count",):
            _bounded_integer(getattr(self, name), name=name, minimum=1)
        for name in ("root_informative_scene_count", "root_partition_class_count"):
            _bounded_integer(getattr(self, name), name=name)

    @property
    def digest(self) -> str:
        return _json_digest(self.as_obj(), domain=_REPORT_DOMAIN)

    def as_obj(self) -> dict[str, Any]:
        same_first_query: bool | None
        if self.identification_only.first_query is None or self.composite_reward.first_query is None:
            same_first_query = (
                self.identification_only.first_query is None and self.composite_reward.first_query is None
            )
        else:
            same_first_query = (
                self.identification_only.first_query.scene_index
                == self.composite_reward.first_query.scene_index
            )
        return {
            "schema_version": REWARD_QUERY_POLICY_SCHEMA_VERSION,
            "report_kind": _REPORT_KIND,
            "authorization": dict(_AUTHORIZATION),
            "catalog_digest": self.catalog.digest,
            "supported_catalog_digest": self.supported_catalog_digest,
            "version_space_rules": _rule_bindings(self.catalog, self.space.indices),
            "uniform_prior_rule_count": len(self.space),
            "official_rule_id": self.official.rule_id,
            "official_truth_digest": self.official.truth_digest,
            "terminal_item_generator_law": self.terminal_item_generator_law.as_obj(),
            "terminal_item_generator_law_digest": self.terminal_item_generator_law.digest,
            "terminal_item_generator_law_public_before_inquiry": True,
            "public_generator_derivation_verification_required": True,
            "secret_realized_panel_support_included": False,
            "one_item_selection_leakage_gate_required": True,
            "excluded_query_scene_indices": list(self.excluded_query_scene_indices),
            "query_exclusions_digest": _exclusions_digest(self.excluded_query_scene_indices),
            "query_exclusions_public_derivation_externally_verified": False,
            "primary_protocol_empty_query_exclusions": not self.excluded_query_scene_indices,
            "legal_query_scene_count": SCENE_COUNT - len(self.excluded_query_scene_indices),
            "maximum_query_budget": self.maximum_query_budget,
            "reward_specification": copy.deepcopy(_REWARD_SPECIFICATION),
            "reward_specification_digest": _reward_digest(),
            "tie_rules": copy.deepcopy(_TIE_RULES),
            "tie_rules_digest": _tie_rules_digest(),
            "information_set": copy.deepcopy(_INFORMATION_SET),
            "information_set_digest": _information_set_digest(),
            "official_excluded_from_policy_optimization": True,
            "secret_reservoir_state_excluded_from_query_policy": True,
            "all_terminal_scenes_excluded_from_AST_choice": True,
            "AST_private_sibling_locked_before_terminal_draw": True,
            "AST_excluded_from_classification_replays": True,
            "terminal_items_use_independent_replay": True,
            "query_tie_context_digest": self.query_tie_context_digest,
            "policy_context_digest": self.policy_context_digest,
            "report_binding_digest": self.report_binding_digest,
            "root_query_partition_evidence": {
                "label_pattern_count": self.root_label_pattern_count,
                "informative_scene_count": self.root_informative_scene_count,
                "partition_class_count": self.root_partition_class_count,
                "partition_classes_digest": self.root_partition_classes_digest,
            },
            "objective_distinction": {
                "identification_only_uses_reward": False,
                "composite_integrates_public_P_scene_given_rule": True,
                "terminal_classification_uses_scene_selection_likelihood": True,
                "composite_conditions_queries_on_secret_reservoir_state": False,
                "composite_has_explicit_stop_at_every_state": True,
                "same_first_query": same_first_query,
            },
            "identification_only_ceiling": self.identification_only.as_obj(),
            "composite_reward_ceiling": self.composite_reward.as_obj(),
        }


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
class _CompositeNode:
    value: Fraction
    expected_terminal_accuracy: Fraction
    expected_exact_rule: Fraction
    expected_query_efficiency: Fraction
    expected_query_count: Fraction
    stop: RewardStopDecisionV2
    choice: _QueryClass | None
    selected_query_value: Fraction | None


@dataclass(frozen=True, slots=True)
class _IdentificationNode:
    identified_leaf_count: int
    choice: _QueryClass | None


class _RewardQuerySolver:
    def __init__(
        self,
        space: VersionSpace,
        official: CatalogEntry,
        terminal_item_generator_law: TerminalItemGeneratorLawV2,
        exclusions: tuple[int, ...],
        maximum_budget: int,
    ) -> None:
        self.space = space
        self.catalog = space.catalog
        self.official = official
        self.terminal_item_generator_law = terminal_item_generator_law
        self.generator_probabilities = terminal_item_generator_law.probability_maps()
        self.exclusions = exclusions
        self.maximum_budget = maximum_budget
        self.indices = space.indices
        self.full_state = (1 << len(space)) - 1
        self.official_bit = 1 << self.indices.index(official.index)
        self.query_tie_context_obj = {
            "catalog_digest": self.catalog.digest,
            "supported_catalog_digest": build_supported_catalog_contract_v2().supported_catalog_digest,
            "version_space_rules": _rule_bindings(self.catalog, self.indices),
            "excluded_query_scene_indices": list(exclusions),
            "query_exclusions_digest": _exclusions_digest(exclusions),
            "tie_rules_digest": _tie_rules_digest(),
        }
        self.query_tie_context_digest = _json_digest(
            self.query_tie_context_obj,
            domain=_QUERY_TIE_CONTEXT_DOMAIN,
        )
        self.policy_context_obj = {
            **self.query_tie_context_obj,
            "query_tie_context_digest": self.query_tie_context_digest,
            "terminal_item_generator_law_digest": terminal_item_generator_law.digest,
            "maximum_query_budget": maximum_budget,
            "reward_specification_digest": _reward_digest(),
            "information_set_digest": _information_set_digest(),
        }
        self.policy_context_digest = _json_digest(
            self.policy_context_obj,
            domain=_POLICY_CONTEXT_DOMAIN,
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
            tie = _hash_tie(
                _QUERY_TIE_DOMAIN,
                self.query_tie_context_digest,
                scene.to_bytes(8, "big"),
            )
            current = grouped.get(labels)
            if current is None:
                grouped[labels] = [1, scene, tie]
            else:
                current[0] += 1
                if (tie, scene) < (current[2], current[1]):
                    current[1] = scene
                    current[2] = tie
        self.patterns = tuple(
            _Pattern(mask, values[0], values[1], values[2]) for mask, values in sorted(grouped.items())
        )
        self._classes_cache: dict[int, tuple[_QueryClass, ...]] = {}
        self._stop_cache: dict[tuple[int, int], RewardStopDecisionV2] = {}
        self._composite_cache: dict[tuple[int, int, int], _CompositeNode] = {}
        self._identification_cache: dict[tuple[int, int], _IdentificationNode] = {}

    def _state_indices(self, state: int) -> tuple[int, ...]:
        if state <= 0 or state & ~self.full_state:
            raise RewardQueryV2Error("reward query solver received an invalid state")
        return tuple(index for offset, index in enumerate(self.indices) if state & (1 << offset))

    def _rule_ids(self, state: int) -> tuple[str, ...]:
        return tuple(self.catalog[index].rule_id for index in self._state_indices(state))

    def classes(self, state: int) -> tuple[_QueryClass, ...]:
        cached = self._classes_cache.get(state)
        if cached is not None:
            return cached
        grouped: dict[int, list[Any]] = {}
        for pattern in self.patterns:
            accepted = pattern.label_mask & state
            rejected = state ^ accepted
            if not accepted or not rejected:
                continue
            partition = min(accepted, rejected)
            current = grouped.get(partition)
            if current is None:
                grouped[partition] = [
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

    def _query_choice(self, state: int, choice: _QueryClass) -> RewardQueryChoiceV2:
        return RewardQueryChoiceV2(
            scene_index=choice.representative_scene_index,
            tie_break_digest=choice.tie_break_digest,
            equivalent_scene_count=choice.scene_count,
            partition_mask=choice.partition_mask,
            rejected_rule_ids=self._rule_ids(choice.rejected_mask),
            accepted_rule_ids=self._rule_ids(choice.accepted_mask),
        )

    def stop(self, state: int, query_count: int) -> RewardStopDecisionV2:
        key = (state, query_count)
        cached = self._stop_cache.get(key)
        if cached is not None:
            return cached
        indices = self._state_indices(state)
        size = len(indices)
        ranked = sorted(
            indices,
            key=lambda index: (
                _hash_tie(
                    _MAP_TIE_DOMAIN,
                    self.query_tie_context_digest,
                    (self.catalog[index].rule_id + "\0" + self.catalog[index].truth_digest).encode("ascii"),
                ),
                self.catalog[index].rule_id,
            ),
        )
        # The AST is selected and locked in a private sibling before any
        # terminal scene is drawn.  Its tie context contains neither the
        # generator law nor Official.
        submitted = self.catalog[ranked[0]]
        submitted_tie = _hash_tie(
            _MAP_TIE_DOMAIN,
            self.query_tie_context_digest,
            (submitted.rule_id + "\0" + submitted.truth_digest).encode("ascii"),
        )
        unique_terminal_scenes = tuple(
            sorted({scene for index in indices for scene in self.generator_probabilities[index]})
        )
        item_policy: list[TerminalItemDecisionV2] = []
        expected_accuracy = Fraction()
        official_expected_accuracy = Fraction()
        official_probabilities = self.generator_probabilities[self.official.index]
        for scene in unique_terminal_scenes:
            rejected_likelihood = sum(
                (
                    self.generator_probabilities[index].get(scene, Fraction())
                    for index in indices
                    if not self.catalog[index].truth[scene]
                ),
                Fraction(),
            )
            accepted_likelihood = sum(
                (
                    self.generator_probabilities[index].get(scene, Fraction())
                    for index in indices
                    if self.catalog[index].truth[scene]
                ),
                Fraction(),
            )
            predicted = accepted_likelihood > rejected_likelihood
            marginal = (accepted_likelihood + rejected_likelihood) / size
            official_probability = official_probabilities.get(scene, Fraction())
            decision = TerminalItemDecisionV2(
                scene_index=scene,
                rejected_likelihood_sum=rejected_likelihood,
                accepted_likelihood_sum=accepted_likelihood,
                marginal_scene_probability=marginal,
                predicted_accepted=predicted,
                official_accepted=self.official.truth[scene],
                official_scene_probability=official_probability,
            )
            item_policy.append(decision)
            expected_accuracy += max(rejected_likelihood, accepted_likelihood) / size
            if predicted is self.official.truth[scene]:
                official_expected_accuracy += official_probability
        item_policy_digest = _json_digest(
            [
                {
                    "scene_index": decision.scene_index,
                    "rejected_likelihood_sum": _fraction_obj(decision.rejected_likelihood_sum),
                    "accepted_likelihood_sum": _fraction_obj(decision.accepted_likelihood_sum),
                    "marginal_scene_probability": _fraction_obj(decision.marginal_scene_probability),
                    "predicted_accepted": decision.predicted_accepted,
                }
                for decision in item_policy
            ],
            domain=_ITEM_POLICY_DOMAIN,
        )
        expected_exact = Fraction(1, size)
        efficiency = Fraction(MAX_REWARD_QUERY_BUDGET - query_count, MAX_REWARD_QUERY_BUDGET)
        expected_reward = (
            Fraction(7, 10) * expected_accuracy
            + Fraction(1, 4) * expected_exact
            + Fraction(1, 20) * efficiency
        )
        official_exact = submitted.index == self.official.index
        official_reward = (
            Fraction(7, 10) * official_expected_accuracy
            + Fraction(1, 4) * int(official_exact)
            + Fraction(1, 20) * efficiency
        )
        result = RewardStopDecisionV2(
            query_count=query_count,
            posterior_rule_ids=tuple(self.catalog[index].rule_id for index in indices),
            submitted_rule_id=submitted.rule_id,
            submitted_rule_truth_digest=submitted.truth_digest,
            submitted_rule_ast=submitted.rule.as_obj(),
            submitted_rule_tie_break_digest=submitted_tie,
            terminal_item_policy=tuple(item_policy),
            terminal_item_policy_digest=item_policy_digest,
            expected_terminal_accuracy=expected_accuracy,
            expected_exact_rule=expected_exact,
            query_efficiency=efficiency,
            expected_reward=expected_reward,
            official_expected_terminal_accuracy=official_expected_accuracy,
            official_exact_rule=official_exact,
            official_expected_reward=official_reward,
        )
        self._stop_cache[key] = result
        return result

    def composite(self, state: int, remaining_budget: int, query_count: int) -> _CompositeNode:
        key_cache = (state, remaining_budget, query_count)
        cached = self._composite_cache.get(key_cache)
        if cached is not None:
            return cached
        stop = self.stop(state, query_count)
        # Equal stop/query values prefer stopping.  Starting from the stop
        # node and replacing it only on strict improvement implements that
        # registered rule without floating-point comparisons.
        best = _CompositeNode(
            stop.expected_reward,
            stop.expected_terminal_accuracy,
            stop.expected_exact_rule,
            stop.query_efficiency,
            Fraction(query_count),
            stop,
            None,
            None,
        )
        if remaining_budget > 0:
            size = state.bit_count()
            best_query: _QueryClass | None = None
            best_query_value: Fraction | None = None
            best_query_components: tuple[Fraction, Fraction, Fraction, Fraction] | None = None
            best_query_key: tuple[str, int, int] | None = None
            for choice in self.classes(state):
                rejected = self.composite(
                    choice.rejected_mask,
                    remaining_budget - 1,
                    query_count + 1,
                )
                accepted = self.composite(
                    choice.accepted_mask,
                    remaining_budget - 1,
                    query_count + 1,
                )
                query_value = (
                    Fraction(choice.rejected_mask.bit_count(), size) * rejected.value
                    + Fraction(choice.accepted_mask.bit_count(), size) * accepted.value
                )
                rejected_weight = Fraction(choice.rejected_mask.bit_count(), size)
                accepted_weight = Fraction(choice.accepted_mask.bit_count(), size)
                query_components = tuple(
                    rejected_weight * rejected_value + accepted_weight * accepted_value
                    for rejected_value, accepted_value in zip(
                        (
                            rejected.expected_terminal_accuracy,
                            rejected.expected_exact_rule,
                            rejected.expected_query_efficiency,
                            rejected.expected_query_count,
                        ),
                        (
                            accepted.expected_terminal_accuracy,
                            accepted.expected_exact_rule,
                            accepted.expected_query_efficiency,
                            accepted.expected_query_count,
                        ),
                        strict=True,
                    )
                )
                query_key = (
                    choice.tie_break_digest,
                    choice.representative_scene_index,
                    choice.partition_mask,
                )
                if (
                    best_query_value is None
                    or query_value > best_query_value
                    or (
                        query_value == best_query_value
                        and query_key < cast(tuple[str, int, int], best_query_key)
                    )
                ):
                    best_query = choice
                    best_query_value = query_value
                    best_query_components = cast(
                        tuple[Fraction, Fraction, Fraction, Fraction],
                        query_components,
                    )
                    best_query_key = query_key
            if (
                best_query is not None
                and best_query_value is not None
                and best_query_components is not None
                and best_query_value > stop.expected_reward
            ):
                best = _CompositeNode(
                    best_query_value,
                    *best_query_components,
                    stop,
                    best_query,
                    best_query_value,
                )
        self._composite_cache[key_cache] = best
        return best

    def identification(self, state: int, remaining_budget: int) -> _IdentificationNode:
        key_cache = (state, remaining_budget)
        cached = self._identification_cache.get(key_cache)
        if cached is not None:
            return cached
        if state.bit_count() == 1:
            result = _IdentificationNode(1, None)
        elif remaining_budget == 0:
            result = _IdentificationNode(0, None)
        else:
            best_count = 0
            best_choice: _QueryClass | None = None
            best_key: tuple[str, int, int] | None = None
            for choice in self.classes(state):
                count = (
                    self.identification(choice.rejected_mask, remaining_budget - 1).identified_leaf_count
                    + self.identification(choice.accepted_mask, remaining_budget - 1).identified_leaf_count
                )
                choice_key = (
                    choice.tie_break_digest,
                    choice.representative_scene_index,
                    choice.partition_mask,
                )
                if (
                    best_choice is None
                    or count > best_count
                    or (count == best_count and choice_key < cast(tuple[str, int, int], best_key))
                ):
                    best_count = count
                    best_choice = choice
                    best_key = choice_key
            result = _IdentificationNode(best_count, best_choice)
        self._identification_cache[key_cache] = result
        return result

    def identification_summary(self) -> IdentificationOnlySummaryV2:
        root = self.identification(self.full_state, self.maximum_budget)
        first = None if root.choice is None else self._query_choice(self.full_state, root.choice)
        state = self.full_state
        remaining = self.maximum_budget
        path: list[IdentificationPathStepV2] = []
        while state.bit_count() > 1 and remaining > 0:
            node = self.identification(state, remaining)
            choice = node.choice
            if choice is None:
                break
            accepted = bool(choice.accepted_mask & self.official_bit)
            after = choice.accepted_mask if accepted else choice.rejected_mask
            path.append(
                IdentificationPathStepV2(
                    turn=len(path),
                    remaining_budget_before=remaining,
                    before_rule_ids=self._rule_ids(state),
                    query=self._query_choice(state, choice),
                    official_accepted=accepted,
                    after_rule_ids=self._rule_ids(after),
                )
            )
            state = after
            remaining -= 1
        return IdentificationOnlySummaryV2(
            maximum_query_budget=self.maximum_budget,
            exact_identification_probability=Fraction(
                root.identified_leaf_count,
                len(self.space),
            ),
            first_query=first,
            official_path=tuple(path),
            official_terminal_rule_ids=self._rule_ids(state),
        )

    def composite_summary(self) -> CompositeRewardSummaryV2:
        root = self.composite(self.full_state, self.maximum_budget, 0)
        first = None if root.choice is None else self._query_choice(self.full_state, root.choice)
        state = self.full_state
        remaining = self.maximum_budget
        query_count = 0
        path: list[RewardPolicyPathStepV2] = []
        while remaining > 0:
            node = self.composite(state, remaining, query_count)
            choice = node.choice
            if choice is None or node.selected_query_value is None:
                break
            accepted = bool(choice.accepted_mask & self.official_bit)
            after = choice.accepted_mask if accepted else choice.rejected_mask
            path.append(
                RewardPolicyPathStepV2(
                    turn=query_count,
                    remaining_budget_before=remaining,
                    before_rule_ids=self._rule_ids(state),
                    immediate_stop_expected_reward=node.stop.expected_reward,
                    selected_query_expected_reward=node.selected_query_value,
                    optimal_expected_reward=node.value,
                    query=self._query_choice(state, choice),
                    official_accepted=accepted,
                    after_rule_ids=self._rule_ids(after),
                )
            )
            state = after
            remaining -= 1
            query_count += 1
        terminal = self.stop(state, query_count)
        return CompositeRewardSummaryV2(
            maximum_query_budget=self.maximum_budget,
            optimal_expected_reward=root.value,
            expected_terminal_accuracy=root.expected_terminal_accuracy,
            expected_exact_rule=root.expected_exact_rule,
            expected_query_efficiency=root.expected_query_efficiency,
            expected_query_count=root.expected_query_count,
            immediate_stop_expected_reward=root.stop.expected_reward,
            root_action_kind="stop" if root.choice is None else "query",
            first_query=first,
            official_path=tuple(path),
            official_terminal_decision=terminal,
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


def build_reward_query_policy_ceiling_report_v2(
    space: VersionSpace,
    official: CatalogEntry,
    *,
    terminal_item_generator_law: TerminalItemGeneratorLawV2,
    excluded_query_scene_indices: Iterable[int | Scene] = (),
    maximum_query_budget: int = MAX_REWARD_QUERY_BUDGET,
) -> RewardQueryPolicyCeilingReportV2:
    """Solve the implementable policy under a public exact one-item law."""

    catalog, indices, _ = _validate_context(space, official)
    if type(terminal_item_generator_law) is not TerminalItemGeneratorLawV2:
        raise TypeError("terminal_item_generator_law must be a TerminalItemGeneratorLawV2")
    expected_law = build_terminal_item_generator_law_v2(
        space,
        terminal_item_generator_law.probability_maps(),
        public_derivation_attestation_digest=(
            terminal_item_generator_law.public_derivation_attestation_digest
        ),
    )
    if expected_law != terminal_item_generator_law:
        raise RewardQueryV2Error("terminal item generator law is not bound to the version space")
    exclusions = _normalize_exclusions(excluded_query_scene_indices)
    budget = _bounded_integer(
        maximum_query_budget,
        name="maximum_query_budget",
        maximum=MAX_REWARD_QUERY_BUDGET,
    )
    solver = _RewardQuerySolver(
        space,
        official,
        terminal_item_generator_law,
        exclusions,
        budget,
    )
    informative_count, class_count, classes_digest = solver.root_partition_evidence()
    report_binding = _json_digest(
        {
            **solver.policy_context_obj,
            "policy_context_digest": solver.policy_context_digest,
            "official_rule_id": official.rule_id,
            "official_truth_digest": official.truth_digest,
        },
        domain=_REPORT_BINDING_DOMAIN,
    )
    return RewardQueryPolicyCeilingReportV2(
        catalog=catalog,
        space=VersionSpace(catalog, indices),
        official=official,
        supported_catalog_digest=build_supported_catalog_contract_v2().supported_catalog_digest,
        terminal_item_generator_law=terminal_item_generator_law,
        excluded_query_scene_indices=exclusions,
        maximum_query_budget=budget,
        query_tie_context_digest=solver.query_tie_context_digest,
        policy_context_digest=solver.policy_context_digest,
        report_binding_digest=report_binding,
        root_label_pattern_count=len(solver.patterns),
        root_informative_scene_count=informative_count,
        root_partition_class_count=class_count,
        root_partition_classes_digest=classes_digest,
        identification_only=solver.identification_summary(),
        composite_reward=solver.composite_summary(),
    )


def verify_reward_query_policy_ceiling_report_v2(
    report: RewardQueryPolicyCeilingReportV2,
) -> RewardQueryPolicyCeilingReportV2:
    """Recompute every policy value, action, tie, path, decision, and digest."""

    if type(report) is not RewardQueryPolicyCeilingReportV2:
        raise TypeError("report must be a RewardQueryPolicyCeilingReportV2")
    expected = build_reward_query_policy_ceiling_report_v2(
        report.space,
        report.official,
        terminal_item_generator_law=report.terminal_item_generator_law,
        excluded_query_scene_indices=report.excluded_query_scene_indices,
        maximum_query_budget=report.maximum_query_budget,
    )
    if _dump_json(expected.as_obj()) != _dump_json(report.as_obj()):
        raise RewardQueryV2Error("reward-query report failed exact replay")
    return report


def serialize_reward_query_policy_ceiling_report_v2(
    report: RewardQueryPolicyCeilingReportV2,
) -> str:
    if type(report) is not RewardQueryPolicyCeilingReportV2:
        raise TypeError("report must be a RewardQueryPolicyCeilingReportV2")
    return _dump_json(report.as_obj())


def _space_from_bindings(catalog: RuleCatalog, value: object) -> VersionSpace:
    if type(value) is not list or not value:
        raise RewardQueryV2Error("version_space_rules must be a nonempty array")
    indices: list[int] = []
    for offset, raw in enumerate(value):
        obj = _require_exact_keys(
            raw,
            {"rule_id", "truth_digest"},
            name=f"version_space_rules[{offset}]",
        )
        rule_id = obj["rule_id"]
        truth_digest = obj["truth_digest"]
        if (
            type(rule_id) is not str
            or len(rule_id) != 9
            or not rule_id.startswith("g03r")
            or not rule_id[4:].isdigit()
        ):
            raise RewardQueryV2Error("version-space rule id is invalid")
        index = int(rule_id[4:])
        if not 0 <= index < len(catalog):
            raise RewardQueryV2Error("version-space rule id lies outside the catalog")
        entry = catalog[index]
        if entry.rule_id != rule_id or entry.truth_digest != truth_digest:
            raise RewardQueryV2Error("version-space rule identity or truth digest mismatch")
        indices.append(index)
    try:
        return VersionSpace(catalog, tuple(indices))
    except ValueError as exc:
        raise RewardQueryV2Error(str(exc)) from exc


def _official_from_identity(
    catalog: RuleCatalog,
    rule_id: object,
    truth_digest: object,
) -> CatalogEntry:
    if (
        type(rule_id) is not str
        or len(rule_id) != 9
        or not rule_id.startswith("g03r")
        or not rule_id[4:].isdigit()
    ):
        raise RewardQueryV2Error("Official rule id is invalid")
    index = int(rule_id[4:])
    if not 0 <= index < len(catalog):
        raise RewardQueryV2Error("Official rule id lies outside the catalog")
    result = catalog[index]
    if result.rule_id != rule_id or result.truth_digest != truth_digest:
        raise RewardQueryV2Error("Official rule identity or truth digest mismatch")
    return result


def _exact_fraction_from_obj(value: object, *, name: str) -> Fraction:
    obj = _require_exact_keys(value, {"numerator", "denominator"}, name=name)
    numerator = _bounded_integer(obj["numerator"], name=f"{name} numerator", minimum=1)
    denominator = _bounded_integer(obj["denominator"], name=f"{name} denominator", minimum=1)
    result = Fraction(numerator, denominator)
    if _fraction_obj(result) != obj:
        raise RewardQueryV2Error(f"{name} is not a reduced positive fraction")
    return result


def _generator_law_from_obj(
    space: VersionSpace,
    value: object,
) -> TerminalItemGeneratorLawV2:
    obj = _require_exact_keys(
        value,
        {
            "law_kind",
            "catalog_digest",
            "supported_catalog_digest",
            "public_derivation_attestation_digest",
            "derivation_contract",
            "external_generator_verification_required",
            "backend_item_count",
            "secret_realized_support_included",
            "conditional_rules",
            "union_scene_support_count",
        },
        name="terminal item generator law",
    )
    if not _is_sha256(obj["public_derivation_attestation_digest"]):
        raise RewardQueryV2Error("generator-law public derivation digest is malformed")
    if type(obj["backend_item_count"]) is not int:
        raise RewardQueryV2Error("generator-law backend item count must be an exact integer")
    raw_rules = obj["conditional_rules"]
    if type(raw_rules) is not list:
        raise RewardQueryV2Error("generator-law conditional_rules must be an array")
    probabilities: dict[int, dict[int, Fraction]] = {}
    for offset, raw_rule in enumerate(raw_rules):
        rule_obj = _require_exact_keys(
            raw_rule,
            {
                "rule_id",
                "truth_digest",
                "scene_support_count",
                "scene_probabilities",
            },
            name=f"generator-law conditional_rules[{offset}]",
        )
        rule_id = rule_obj["rule_id"]
        if (
            type(rule_id) is not str
            or len(rule_id) != 9
            or not rule_id.startswith("g03r")
            or not rule_id[4:].isdigit()
        ):
            raise RewardQueryV2Error("generator-law rule id is invalid")
        index = int(rule_id[4:])
        raw_probabilities = rule_obj["scene_probabilities"]
        if type(raw_probabilities) is not list:
            raise RewardQueryV2Error("generator-law scene_probabilities must be an array")
        scene_map: dict[int, Fraction] = {}
        for scene_offset, raw_probability in enumerate(raw_probabilities):
            probability_obj = _require_exact_keys(
                raw_probability,
                {"scene_index", "probability"},
                name=(f"generator-law conditional_rules[{offset}].scene_probabilities[{scene_offset}]"),
            )
            scene = _normalize_scene_index(
                cast(int, probability_obj["scene_index"]),
                name="generator-law scene index",
            )
            if scene in scene_map:
                raise RewardQueryV2Error("generator-law scene support contains duplicates")
            scene_map[scene] = _exact_fraction_from_obj(
                probability_obj["probability"],
                name="generator-law scene probability",
            )
        if index in probabilities:
            raise RewardQueryV2Error("generator-law rule support contains duplicates")
        probabilities[index] = scene_map
    expected = build_terminal_item_generator_law_v2(
        space,
        probabilities,
        public_derivation_attestation_digest=cast(
            str,
            obj["public_derivation_attestation_digest"],
        ),
    )
    if _dump_json(expected.as_obj()) != _dump_json(value):
        raise RewardQueryV2Error("terminal item generator law is inconsistent or tampered")
    return expected


def reward_query_policy_ceiling_report_v2_from_obj(
    value: object,
    *,
    catalog: RuleCatalog | None = None,
) -> RewardQueryPolicyCeilingReportV2:
    expected_keys = {
        "schema_version",
        "report_kind",
        "authorization",
        "catalog_digest",
        "supported_catalog_digest",
        "version_space_rules",
        "uniform_prior_rule_count",
        "official_rule_id",
        "official_truth_digest",
        "terminal_item_generator_law",
        "terminal_item_generator_law_digest",
        "terminal_item_generator_law_public_before_inquiry",
        "public_generator_derivation_verification_required",
        "secret_realized_panel_support_included",
        "one_item_selection_leakage_gate_required",
        "excluded_query_scene_indices",
        "query_exclusions_digest",
        "query_exclusions_public_derivation_externally_verified",
        "primary_protocol_empty_query_exclusions",
        "legal_query_scene_count",
        "maximum_query_budget",
        "reward_specification",
        "reward_specification_digest",
        "tie_rules",
        "tie_rules_digest",
        "information_set",
        "information_set_digest",
        "official_excluded_from_policy_optimization",
        "secret_reservoir_state_excluded_from_query_policy",
        "all_terminal_scenes_excluded_from_AST_choice",
        "AST_private_sibling_locked_before_terminal_draw",
        "AST_excluded_from_classification_replays",
        "terminal_items_use_independent_replay",
        "query_tie_context_digest",
        "policy_context_digest",
        "report_binding_digest",
        "root_query_partition_evidence",
        "objective_distinction",
        "identification_only_ceiling",
        "composite_reward_ceiling",
    }
    obj = _require_exact_keys(value, expected_keys, name="reward-query report")
    if (
        type(obj["schema_version"]) is not int
        or obj["schema_version"] != REWARD_QUERY_POLICY_SCHEMA_VERSION
        or obj["report_kind"] != _REPORT_KIND
    ):
        raise RewardQueryV2Error("unsupported reward-query report")
    selected_catalog = build_rule_catalog() if catalog is None else catalog
    if type(selected_catalog) is not RuleCatalog:
        raise TypeError("catalog must be a RuleCatalog")
    if obj["catalog_digest"] != selected_catalog.digest:
        raise RewardQueryV2Error("reward-query report catalog digest mismatch")
    contract = build_supported_catalog_contract_v2()
    if obj["supported_catalog_digest"] != contract.supported_catalog_digest:
        raise RewardQueryV2Error("reward-query report supported-catalog digest mismatch")
    space = _space_from_bindings(selected_catalog, obj["version_space_rules"])
    official = _official_from_identity(
        selected_catalog,
        obj["official_rule_id"],
        obj["official_truth_digest"],
    )
    generator_law = _generator_law_from_obj(space, obj["terminal_item_generator_law"])
    raw_exclusions = obj["excluded_query_scene_indices"]
    if type(raw_exclusions) is not list:
        raise RewardQueryV2Error("query exclusions must be an array")
    exclusions = _normalize_exclusions(cast(list[int], raw_exclusions))
    if list(exclusions) != raw_exclusions:
        raise RewardQueryV2Error("query exclusions are not canonical")
    budget = _bounded_integer(
        obj["maximum_query_budget"],
        name="maximum_query_budget",
        maximum=MAX_REWARD_QUERY_BUDGET,
    )
    expected = build_reward_query_policy_ceiling_report_v2(
        space,
        official,
        terminal_item_generator_law=generator_law,
        excluded_query_scene_indices=exclusions,
        maximum_query_budget=budget,
    )
    if _dump_json(expected.as_obj()) != _dump_json(value):
        raise RewardQueryV2Error("reward-query report is inconsistent or tampered")
    return expected


def parse_reward_query_policy_ceiling_report_v2(
    text: str,
    *,
    catalog: RuleCatalog | None = None,
    require_canonical: bool = True,
) -> RewardQueryPolicyCeilingReportV2:
    value = _load_json(text)
    result = reward_query_policy_ceiling_report_v2_from_obj(value, catalog=catalog)
    if require_canonical and serialize_reward_query_policy_ceiling_report_v2(result) != text:
        raise RewardQueryV2Error("reward-query report is valid but not canonical JSON")
    return result
