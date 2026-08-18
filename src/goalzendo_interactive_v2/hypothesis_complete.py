"""Exact, nonauthorizing hypothesis-complete training contracts for G03-v2.

The old G03-v2 three-role cover rotated only a designated ``P/Q/C`` triple.
That construction is not a valid training-prior audit: a syntactic feature
could identify one of three candidates even when the opening left many more
hypotheses live.  This module instead binds one exact supported opening and
rotates *every* member of its supported version space through the Official
role once, at one atomic optimizer boundary.

The schemas here are deliberately structural.  They replay the public catalog,
the v2 supported-index allowlist, the opening, exact equal weights, and the
optimizer chain.  They do not materialize a production bank or authorize a
weight update.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, cast

from goalzendo_interactive.catalog import CatalogEntry, VersionSpace, build_rule_catalog
from goalzendo_interactive.rendering import (
    TRAIN_RENDERERS,
    RendererName,
    render_scene,
    renderer_digest,
)
from goalzendo_interactive.rules import BinaryRule
from goalzendo_interactive.rules import Literal as RuleLiteral
from goalzendo_interactive.schema import SCENE_COUNT, scene_at

from .population_audit import (
    CANDIDATE_RULE_FAMILIES,
    build_supported_catalog_contract_v2,
    classify_catalog_identity_v2,
)

HYPOTHESIS_COMPLETE_TRAINING_SCHEMA_VERSION = 2
HYPOTHESIS_COMPLETE_MANIFEST_SCHEMA_VERSION = 2
UNCONDITIONAL_TERMINAL_LAW_SCHEMA_VERSION = 2
MATERIALIZED_TRAINING_PANEL_SCHEMA_VERSION = 1
ALLOWED_HYPOTHESIS_COUNTS: tuple[int, ...] = (8, 12, 16)
OPENING_DEMONSTRATION_COUNT = 10
TRAIN_TERMINAL_ITEM_COUNT = 16
SCIENTIFIC_TRAINING_EPISODE_BUDGET = 384
POWERED_STRESS_MINIMUM_INDEPENDENT_BLOCKS = 384

_BLOCK_KIND = "g03-v2-hypothesis-complete-atomic-training-block"
_MANIFEST_KIND = "g03-v2-hypothesis-complete-training-manifest-audit"
_TERMINAL_LAW_KIND = "public-exact-unconditional-one-item-marginal"

_OPENING_DOMAIN = "goalzendo-interactive-v2-hypothesis-complete-opening-v2"
_OPENING_CONTENT_DOMAIN = "goalzendo-interactive-v2-hypothesis-complete-opening-content-v2"
_OPENING_SCENE_SET_DOMAIN = "goalzendo-interactive-v2-hypothesis-complete-opening-scenes-v1"
_TERMINAL_LAW_DOMAIN = "goalzendo-interactive-v2-unconditional-train-terminal-law-v2"
_PANEL_DOMAIN = "goalzendo-interactive-v2-materialized-training-panel-v1"
_HIDDEN_BINDING_DOMAIN = "goalzendo-interactive-v2-hidden-order-display-binding-v2"
_HIDDEN_RANK_DOMAIN = "goalzendo-interactive-v2-hidden-order-display-rank-v2"
_STATIC_INPUT_DOMAIN = "goalzendo-interactive-v2-role-neutral-static-model-input-v2"
_ROTATION_DOMAIN = "goalzendo-interactive-v2-hypothesis-complete-rotation-v2"
_COMMIT_DOMAIN = "goalzendo-interactive-v2-hypothesis-complete-atomic-commit-v2"
_BLOCK_DOMAIN = "goalzendo-interactive-v2-hypothesis-complete-training-block-v2"
_MANIFEST_DOMAIN = "goalzendo-interactive-v2-hypothesis-complete-training-manifest-v2"

_AUTHORIZATION = {
    "scope": "prospective-structural-training-schema-only",
    "production_bank_materialized": False,
    "capability_run_authorized": False,
    "weight_updates_authorized": False,
}

_LEGACY_THREE_ROLE_KINDS = frozenset(
    {
        "g03-v2-role-counterbalanced-evidence-block",
        "g03-v2-catalog-bound-perfect-training-role-audit",
        "g03-v2-metadata-only-role-balance-audit",
    }
)
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
_RULE_ID_PATTERN = re.compile(r"g03r[0-9]{5}")
_FEATURE_ORDER = (
    "family",
    "literal_count",
    "operator",
    "placard_inclusion",
    "atom_family",
    "prevalence_bin",
)
_MODEL_VISIBLE_FORBIDDEN_KEYS = frozenset(
    {
        "p",
        "q",
        "a",
        "b",
        "cover",
        "stage",
        "candidate_role",
        "official",
        "official_rule_id",
        "official_truth_digest",
        "rotation_position",
        "block_id",
        "opening_id",
    }
)


class HypothesisCompleteV2Error(ValueError):
    """Raised when a hypothesis-complete contract fails exact replay."""


def _dump_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise HypothesisCompleteV2Error(f"value is not canonical JSON: {exc}") from exc


def _load_json(text: str) -> Any:
    if type(text) is not str or not text:
        raise HypothesisCompleteV2Error("JSON input must be a nonempty string")

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise HypothesisCompleteV2Error(f"duplicate JSON object key: {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise HypothesisCompleteV2Error(f"non-finite JSON constant is forbidden: {value}")

    try:
        return json.loads(text, object_pairs_hook=no_duplicates, parse_constant=reject_constant)
    except HypothesisCompleteV2Error:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HypothesisCompleteV2Error(f"invalid JSON: {exc}") from exc


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
        raise HypothesisCompleteV2Error(f"{name} must be a lowercase SHA-256")
    return cast(str, value)


def _require_identifier(value: object, *, name: str) -> str:
    if type(value) is not str or _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise HypothesisCompleteV2Error(f"{name} is not a canonical identifier")
    return value


def _require_integer(
    value: object,
    *,
    name: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise HypothesisCompleteV2Error(f"{name} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise HypothesisCompleteV2Error(f"{name} must be an integer <= {maximum}")
    return value


def _require_boolean(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise HypothesisCompleteV2Error(f"{name} must be a Boolean")
    return value


def _require_mapping(
    value: object,
    fields: tuple[str, ...],
    *,
    name: str,
) -> Mapping[str, Any]:
    if type(value) is not dict or tuple(value) != fields:
        raise HypothesisCompleteV2Error(f"{name} has noncanonical or reordered fields")
    return cast(Mapping[str, Any], value)


def _require_authorization(value: object) -> None:
    obj = _require_mapping(value, tuple(_AUTHORIZATION), name="authorization")
    if dict(obj) != _AUTHORIZATION:
        raise HypothesisCompleteV2Error("authorization must remain false and nonauthorizing")


def _reject_legacy_three_role_substitute(value: object) -> None:
    if type(value) is not dict:
        return
    report_kind = value.get("report_kind")
    if report_kind in _LEGACY_THREE_ROLE_KINDS:
        raise HypothesisCompleteV2Error(
            "legacy three-role schemas are nonauthorizing and cannot substitute "
            "for a hypothesis-complete block or manifest"
        )


def _catalog_entry(rule_id: object) -> CatalogEntry:
    if type(rule_id) is not str or _RULE_ID_PATTERN.fullmatch(rule_id) is None:
        raise HypothesisCompleteV2Error(f"malformed public rule id: {rule_id!r}")
    index = int(rule_id[4:])
    catalog = build_rule_catalog()
    if not 0 <= index < len(catalog) or catalog[index].rule_id != rule_id:
        raise HypothesisCompleteV2Error(f"unknown public rule id: {rule_id!r}")
    return catalog[index]


@dataclass(frozen=True, slots=True)
class OpeningObservationV2:
    """One canonical, evaluator-stored opening observation."""

    scene_index: int
    accepted: bool

    def __post_init__(self) -> None:
        _require_integer(
            self.scene_index,
            name="opening scene index",
            maximum=SCENE_COUNT - 1,
        )
        _require_boolean(self.accepted, name="opening accepted label")

    def as_obj(self) -> dict[str, Any]:
        return {"scene_index": self.scene_index, "accepted": self.accepted}


def _supported_version_space(
    observations: tuple[OpeningObservationV2, ...],
) -> VersionSpace:
    catalog = build_rule_catalog()
    contract = build_supported_catalog_contract_v2()
    if contract.source_catalog_digest != catalog.digest:
        raise HypothesisCompleteV2Error("supported allowlist uses a different full-catalog namespace")
    return VersionSpace(catalog, contract.supported_indices).observe_many(
        (observation.scene_index, observation.accepted) for observation in observations
    )


@dataclass(frozen=True, slots=True)
class CanonicalSupportedOpeningV2:
    """One ten-example opening plus its recomputed exact supported ``V0``."""

    opening_id: str
    catalog_digest: str
    supported_catalog_digest: str
    observations: tuple[OpeningObservationV2, ...]
    n0: int
    version_space_rule_ids: tuple[str, ...]
    version_space_truth_digests: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_identifier(self.opening_id, name="opening_id")
        if type(self.observations) is not tuple or len(self.observations) != OPENING_DEMONSTRATION_COUNT:
            raise HypothesisCompleteV2Error(
                f"opening must contain exactly {OPENING_DEMONSTRATION_COUNT} observations"
            )
        if any(type(item) is not OpeningObservationV2 for item in self.observations):
            raise HypothesisCompleteV2Error("opening contains a foreign observation type")
        scene_indices = tuple(item.scene_index for item in self.observations)
        if scene_indices != tuple(sorted(set(scene_indices))):
            raise HypothesisCompleteV2Error("opening scenes must be sorted, unique, and canonical")
        if sum(item.accepted for item in self.observations) != OPENING_DEMONSTRATION_COUNT // 2:
            raise HypothesisCompleteV2Error("opening labels must be exactly balanced five/five")

        catalog = build_rule_catalog()
        contract = build_supported_catalog_contract_v2()
        if self.catalog_digest != catalog.digest:
            raise HypothesisCompleteV2Error("opening uses the wrong full-catalog namespace")
        if self.supported_catalog_digest != contract.supported_catalog_digest:
            raise HypothesisCompleteV2Error("opening uses the wrong supported-catalog allowlist")
        if self.n0 not in ALLOWED_HYPOTHESIS_COUNTS:
            raise HypothesisCompleteV2Error(f"opening n0 must be one of {ALLOWED_HYPOTHESIS_COUNTS}")

        exact_space = _supported_version_space(self.observations)
        exact_entries = tuple(exact_space)
        exact_ids = tuple(entry.rule_id for entry in exact_entries)
        exact_truth = tuple(entry.truth_digest for entry in exact_entries)
        if len(exact_entries) != self.n0:
            raise HypothesisCompleteV2Error(
                f"opening recomputes to {len(exact_entries)} supported rules, not n0={self.n0}"
            )
        if self.version_space_rule_ids != exact_ids:
            raise HypothesisCompleteV2Error("opening V0 rule identities differ from exact recomputation")
        if self.version_space_truth_digests != exact_truth:
            raise HypothesisCompleteV2Error("opening V0 truth identities differ from exact recomputation")
        if any(entry.index not in set(contract.supported_indices) for entry in exact_entries):
            raise HypothesisCompleteV2Error("an unsupported full-catalog identity entered V0")

    def _content_obj(self) -> dict[str, Any]:
        return {
            "catalog_digest": self.catalog_digest,
            "supported_catalog_digest": self.supported_catalog_digest,
            "observations": [item.as_obj() for item in self.observations],
            "n0": self.n0,
            "version_space_rules": [
                {"rule_id": rule_id, "truth_digest": truth_digest}
                for rule_id, truth_digest in zip(
                    self.version_space_rule_ids,
                    self.version_space_truth_digests,
                    strict=True,
                )
            ],
        }

    @property
    def content_digest(self) -> str:
        """Semantic opening identity, deliberately excluding the caller label."""

        return _json_digest(self._content_obj(), domain=_OPENING_CONTENT_DOMAIN)

    @property
    def scene_set_digest(self) -> str:
        """Identity of the unordered scene set, independent of labels and IDs."""

        return _json_digest(
            {
                "scene_count": SCENE_COUNT,
                "scene_indices": [item.scene_index for item in self.observations],
            },
            domain=_OPENING_SCENE_SET_DOMAIN,
        )

    def _unsigned_obj(self) -> dict[str, Any]:
        return {
            "opening_id": self.opening_id,
            **self._content_obj(),
            "opening_scene_set_digest": self.scene_set_digest,
            "opening_content_digest": self.content_digest,
        }

    @property
    def digest(self) -> str:
        return _json_digest(self._unsigned_obj(), domain=_OPENING_DOMAIN)

    def as_obj(self) -> dict[str, Any]:
        return {**self._unsigned_obj(), "opening_digest": self.digest}


def build_canonical_supported_opening_v2(
    opening_id: str,
    observations: Iterable[OpeningObservationV2 | tuple[int, bool]],
) -> CanonicalSupportedOpeningV2:
    """Canonicalize an opening and accept it only when ``|V0|`` is registered."""

    materialized: list[OpeningObservationV2] = []
    for value in observations:
        if type(value) is OpeningObservationV2:
            materialized.append(value)
        elif type(value) is tuple and len(value) == 2:
            materialized.append(OpeningObservationV2(value[0], value[1]))
        else:
            raise TypeError("observations must be OpeningObservationV2 values or (scene, label) tuples")
    canonical = tuple(sorted(materialized, key=lambda item: item.scene_index))
    space = _supported_version_space(canonical)
    entries = tuple(space)
    catalog = build_rule_catalog()
    contract = build_supported_catalog_contract_v2()
    return CanonicalSupportedOpeningV2(
        opening_id=opening_id,
        catalog_digest=catalog.digest,
        supported_catalog_digest=contract.supported_catalog_digest,
        observations=canonical,
        n0=len(entries),
        version_space_rule_ids=tuple(entry.rule_id for entry in entries),
        version_space_truth_digests=tuple(entry.truth_digest for entry in entries),
    )


def canonical_supported_opening_v2_from_obj(value: object) -> CanonicalSupportedOpeningV2:
    obj = _require_mapping(
        value,
        (
            "opening_id",
            "catalog_digest",
            "supported_catalog_digest",
            "observations",
            "n0",
            "version_space_rules",
            "opening_scene_set_digest",
            "opening_content_digest",
            "opening_digest",
        ),
        name="canonical supported opening",
    )
    raw_observations = obj["observations"]
    if type(raw_observations) is not list:
        raise HypothesisCompleteV2Error("opening observations must be an array")
    observations: list[OpeningObservationV2] = []
    for raw in raw_observations:
        item = _require_mapping(raw, ("scene_index", "accepted"), name="opening observation")
        observations.append(
            OpeningObservationV2(
                _require_integer(
                    item["scene_index"],
                    name="opening scene index",
                    maximum=SCENE_COUNT - 1,
                ),
                _require_boolean(item["accepted"], name="opening accepted label"),
            )
        )
    raw_rules = obj["version_space_rules"]
    if type(raw_rules) is not list:
        raise HypothesisCompleteV2Error("opening version-space rules must be an array")
    rule_ids: list[str] = []
    truth_digests: list[str] = []
    for raw in raw_rules:
        item = _require_mapping(raw, ("rule_id", "truth_digest"), name="opening V0 identity")
        if type(item["rule_id"]) is not str:
            raise HypothesisCompleteV2Error("opening V0 rule ID must be a string")
        rule_ids.append(item["rule_id"])
        truth_digests.append(_require_sha256(item["truth_digest"], name="opening V0 truth digest"))
    opening = CanonicalSupportedOpeningV2(
        opening_id=_require_identifier(obj["opening_id"], name="opening_id"),
        catalog_digest=_require_sha256(obj["catalog_digest"], name="catalog digest"),
        supported_catalog_digest=_require_sha256(
            obj["supported_catalog_digest"], name="supported-catalog digest"
        ),
        observations=tuple(observations),
        n0=_require_integer(obj["n0"], name="n0", minimum=1),
        version_space_rule_ids=tuple(rule_ids),
        version_space_truth_digests=tuple(truth_digests),
    )
    if obj["opening_scene_set_digest"] != opening.scene_set_digest:
        raise HypothesisCompleteV2Error("canonical opening scene-set digest is inconsistent")
    if obj["opening_content_digest"] != opening.content_digest:
        raise HypothesisCompleteV2Error("canonical opening content digest is inconsistent")
    if obj["opening_digest"] != opening.digest or _dump_json(obj) != _dump_json(opening.as_obj()):
        raise HypothesisCompleteV2Error("canonical opening digest or derived metadata is inconsistent")
    return opening


@dataclass(frozen=True, slots=True)
class UnconditionalSceneProbabilityV2:
    """One positive exact mass in a rule-oblivious terminal scene law."""

    scene_index: int
    probability: Fraction

    def __post_init__(self) -> None:
        _require_integer(
            self.scene_index,
            name="terminal-law scene index",
            maximum=SCENE_COUNT - 1,
        )
        if type(self.probability) is not Fraction or not 0 < self.probability <= 1:
            raise HypothesisCompleteV2Error("terminal-law probability must be an exact fraction in (0,1]")

    def as_obj(self) -> dict[str, Any]:
        return {
            "scene_index": self.scene_index,
            "probability": {
                "numerator": self.probability.numerator,
                "denominator": self.probability.denominator,
            },
        }


@dataclass(frozen=True, slots=True)
class UnconditionalTerminalSceneLawV2:
    """A single public exact ``P(scene)`` shared by every block rotation."""

    catalog_digest: str
    supported_catalog_digest: str
    public_derivation_attestation_digest: str
    scene_probabilities: tuple[UnconditionalSceneProbabilityV2, ...]

    def __post_init__(self) -> None:
        catalog = build_rule_catalog()
        contract = build_supported_catalog_contract_v2()
        if self.catalog_digest != catalog.digest:
            raise HypothesisCompleteV2Error("terminal law uses the wrong full-catalog namespace")
        if self.supported_catalog_digest != contract.supported_catalog_digest:
            raise HypothesisCompleteV2Error("terminal law uses the wrong supported-catalog allowlist")
        _require_sha256(
            self.public_derivation_attestation_digest,
            name="public derivation attestation digest",
        )
        if (
            type(self.scene_probabilities) is not tuple
            or len(self.scene_probabilities) < TRAIN_TERMINAL_ITEM_COUNT
        ):
            raise HypothesisCompleteV2Error(
                f"unconditional terminal law requires support of at least {TRAIN_TERMINAL_ITEM_COUNT} scenes"
            )
        if any(type(item) is not UnconditionalSceneProbabilityV2 for item in self.scene_probabilities):
            raise HypothesisCompleteV2Error("terminal law contains a foreign probability type")
        indices = tuple(item.scene_index for item in self.scene_probabilities)
        if indices != tuple(sorted(set(indices))):
            raise HypothesisCompleteV2Error("terminal-law support must be sorted, unique, and canonical")
        if sum((item.probability for item in self.scene_probabilities), Fraction()) != 1:
            raise HypothesisCompleteV2Error("unconditional terminal-law mass must sum exactly to one")

    def _unsigned_obj(self) -> dict[str, Any]:
        return {
            "schema_version": UNCONDITIONAL_TERMINAL_LAW_SCHEMA_VERSION,
            "law_kind": _TERMINAL_LAW_KIND,
            "catalog_digest": self.catalog_digest,
            "supported_catalog_digest": self.supported_catalog_digest,
            "public_derivation_attestation_digest": self.public_derivation_attestation_digest,
            "public_derivation_attestation_externally_verified": False,
            "external_generator_verification_required": True,
            "terminal_item_count": TRAIN_TERMINAL_ITEM_COUNT,
            "scene_support_count": len(self.scene_probabilities),
            "scene_probabilities": [item.as_obj() for item in self.scene_probabilities],
        }

    @property
    def digest(self) -> str:
        return _json_digest(self._unsigned_obj(), domain=_TERMINAL_LAW_DOMAIN)

    def as_obj(self) -> dict[str, Any]:
        return {**self._unsigned_obj(), "unconditional_scene_law_digest": self.digest}


def build_unconditional_terminal_scene_law_v2(
    scene_probabilities: Iterable[tuple[int, Fraction]],
    *,
    public_derivation_attestation_digest: str,
) -> UnconditionalTerminalSceneLawV2:
    """Build one exact, Official-independent terminal marginal ``P(scene)``."""

    probabilities = tuple(
        sorted(
            (
                UnconditionalSceneProbabilityV2(scene_index, probability)
                for scene_index, probability in scene_probabilities
            ),
            key=lambda item: item.scene_index,
        )
    )
    catalog = build_rule_catalog()
    contract = build_supported_catalog_contract_v2()
    return UnconditionalTerminalSceneLawV2(
        catalog_digest=catalog.digest,
        supported_catalog_digest=contract.supported_catalog_digest,
        public_derivation_attestation_digest=public_derivation_attestation_digest,
        scene_probabilities=probabilities,
    )


def unconditional_terminal_scene_law_v2_from_obj(
    value: object,
) -> UnconditionalTerminalSceneLawV2:
    obj = _require_mapping(
        value,
        (
            "schema_version",
            "law_kind",
            "catalog_digest",
            "supported_catalog_digest",
            "public_derivation_attestation_digest",
            "public_derivation_attestation_externally_verified",
            "external_generator_verification_required",
            "terminal_item_count",
            "scene_support_count",
            "scene_probabilities",
            "unconditional_scene_law_digest",
        ),
        name="unconditional terminal scene law",
    )
    if (
        type(obj["schema_version"]) is not int
        or obj["schema_version"] != UNCONDITIONAL_TERMINAL_LAW_SCHEMA_VERSION
        or obj["law_kind"] != _TERMINAL_LAW_KIND
    ):
        raise HypothesisCompleteV2Error("unconditional terminal-law schema identity mismatch")
    if _require_boolean(
        obj["public_derivation_attestation_externally_verified"],
        name="public derivation external verification",
    ):
        raise HypothesisCompleteV2Error("public derivation attestation must remain externally unverified")
    if not _require_boolean(
        obj["external_generator_verification_required"],
        name="external generator verification requirement",
    ):
        raise HypothesisCompleteV2Error("external generator verification must remain required")
    if _require_integer(obj["terminal_item_count"], name="terminal item count") != TRAIN_TERMINAL_ITEM_COUNT:
        raise HypothesisCompleteV2Error("terminal item count differs from the registered train terminal")
    raw_probabilities = obj["scene_probabilities"]
    if type(raw_probabilities) is not list:
        raise HypothesisCompleteV2Error("terminal scene probabilities must be an array")
    probabilities: list[UnconditionalSceneProbabilityV2] = []
    for raw in raw_probabilities:
        item = _require_mapping(raw, ("scene_index", "probability"), name="terminal scene mass")
        fraction = _require_mapping(
            item["probability"],
            ("numerator", "denominator"),
            name="terminal scene exact probability",
        )
        numerator = _require_integer(fraction["numerator"], name="probability numerator", minimum=1)
        denominator = _require_integer(fraction["denominator"], name="probability denominator", minimum=1)
        probabilities.append(
            UnconditionalSceneProbabilityV2(
                _require_integer(
                    item["scene_index"],
                    name="terminal scene index",
                    maximum=SCENE_COUNT - 1,
                ),
                Fraction(numerator, denominator),
            )
        )
    law = UnconditionalTerminalSceneLawV2(
        catalog_digest=_require_sha256(obj["catalog_digest"], name="catalog digest"),
        supported_catalog_digest=_require_sha256(
            obj["supported_catalog_digest"], name="supported-catalog digest"
        ),
        public_derivation_attestation_digest=_require_sha256(
            obj["public_derivation_attestation_digest"],
            name="public derivation attestation digest",
        ),
        scene_probabilities=tuple(probabilities),
    )
    if _require_integer(obj["scene_support_count"], name="scene support count", minimum=1) != len(
        law.scene_probabilities
    ):
        raise HypothesisCompleteV2Error("terminal-law support count is inconsistent")
    if obj["unconditional_scene_law_digest"] != law.digest or _dump_json(obj) != _dump_json(law.as_obj()):
        raise HypothesisCompleteV2Error("unconditional terminal-law digest or metadata is inconsistent")
    return law


def _fraction_obj(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _truth_pattern(entries: tuple[CatalogEntry, ...], scene_index: int) -> int:
    return sum(1 << position for position, entry in enumerate(entries) if entry.truth[scene_index])


@dataclass(frozen=True, slots=True)
class MaterializedTrainingPanelV2:
    """Evaluator-private 16-scene terminal panel shared by all rotations."""

    catalog_digest: str
    supported_catalog_digest: str
    opening_content_digest: str
    version_space_rule_ids: tuple[str, ...]
    version_space_truth_digests: tuple[str, ...]
    scene_indices: tuple[int, ...]
    external_generator_receipt_digest: str

    def __post_init__(self) -> None:
        catalog = build_rule_catalog()
        contract = build_supported_catalog_contract_v2()
        if self.catalog_digest != catalog.digest:
            raise HypothesisCompleteV2Error("materialized panel uses the wrong catalog namespace")
        if self.supported_catalog_digest != contract.supported_catalog_digest:
            raise HypothesisCompleteV2Error("materialized panel uses the wrong supported allowlist")
        _require_sha256(self.opening_content_digest, name="panel opening-content digest")
        _require_sha256(
            self.external_generator_receipt_digest,
            name="external panel-generator receipt digest",
        )
        if (
            type(self.version_space_rule_ids) is not tuple
            or len(self.version_space_rule_ids) not in ALLOWED_HYPOTHESIS_COUNTS
            or self.version_space_rule_ids != tuple(sorted(set(self.version_space_rule_ids)))
        ):
            raise HypothesisCompleteV2Error("panel V0 identities must be canonical with allowed n0")
        if type(self.version_space_truth_digests) is not tuple or len(
            self.version_space_truth_digests
        ) != len(self.version_space_rule_ids):
            raise HypothesisCompleteV2Error("panel V0 truth identities do not align")
        entries = tuple(_catalog_entry(rule_id) for rule_id in self.version_space_rule_ids)
        supported = set(contract.supported_indices)
        for entry, truth_digest in zip(
            entries,
            self.version_space_truth_digests,
            strict=True,
        ):
            if entry.index not in supported or entry.truth_digest != truth_digest:
                raise HypothesisCompleteV2Error("panel contains an unsupported or false V0 identity")
        if (
            type(self.scene_indices) is not tuple
            or len(self.scene_indices) != TRAIN_TERMINAL_ITEM_COUNT
            or self.scene_indices != tuple(sorted(set(self.scene_indices)))
        ):
            raise HypothesisCompleteV2Error(
                f"materialized panel must contain {TRAIN_TERMINAL_ITEM_COUNT} sorted distinct scenes"
            )
        for scene_index in self.scene_indices:
            _require_integer(
                scene_index,
                name="materialized panel scene index",
                maximum=SCENE_COUNT - 1,
            )
            if _truth_pattern(entries, scene_index).bit_count() != len(entries) // 2:
                raise HypothesisCompleteV2Error("every materialized panel scene must be half/half across V0")
        full_pattern_mask = (1 << len(entries)) - 1
        pattern_counts = Counter(_truth_pattern(entries, scene_index) for scene_index in self.scene_indices)
        if any(
            pattern_counts[pattern] != pattern_counts[full_pattern_mask ^ pattern]
            for pattern in pattern_counts
        ):
            raise HypothesisCompleteV2Error(
                "materialized panel must be sampled in complementary truth-pattern pairs"
            )
        for entry in entries:
            accepted = sum(entry.truth[scene_index] for scene_index in self.scene_indices)
            if accepted != TRAIN_TERMINAL_ITEM_COUNT // 2:
                raise HypothesisCompleteV2Error(
                    "every V0 rule must label the materialized panel exactly eight/eight"
                )

    def _entries(self) -> tuple[CatalogEntry, ...]:
        return tuple(_catalog_entry(rule_id) for rule_id in self.version_space_rule_ids)

    def _truth_pattern_rows(self) -> list[dict[str, Any]]:
        entries = self._entries()
        return [
            {
                "scene_index": scene_index,
                "truth_pattern_mask": _truth_pattern(entries, scene_index),
                "accepted_candidate_count": len(entries) // 2,
                "rejected_candidate_count": len(entries) // 2,
            }
            for scene_index in self.scene_indices
        ]

    def _rule_balance_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "rule_id": entry.rule_id,
                "truth_digest": entry.truth_digest,
                "accepted_count": sum(entry.truth[index] for index in self.scene_indices),
                "rejected_count": sum(not entry.truth[index] for index in self.scene_indices),
            }
            for entry in self._entries()
        ]

    def _unsigned_obj(self) -> dict[str, Any]:
        return {
            "schema_version": MATERIALIZED_TRAINING_PANEL_SCHEMA_VERSION,
            "panel_kind": "evaluator-private-shared-sixteen-scene-training-panel",
            "catalog_digest": self.catalog_digest,
            "supported_catalog_digest": self.supported_catalog_digest,
            "opening_content_digest": self.opening_content_digest,
            "version_space_rules": [
                {"rule_id": rule_id, "truth_digest": truth_digest}
                for rule_id, truth_digest in zip(
                    self.version_space_rule_ids,
                    self.version_space_truth_digests,
                    strict=True,
                )
            ],
            "scene_indices": list(self.scene_indices),
            "truth_pattern_rows": self._truth_pattern_rows(),
            "rule_balance_rows": self._rule_balance_rows(),
            "external_generator_receipt_digest": self.external_generator_receipt_digest,
            "external_generator_receipt_verified": False,
            "external_generator_verification_required": True,
            "panel_binding_visible_to_model": False,
            "pairwise_rule_separation_required_here": False,
            "evaluation_challenge_bank_owns_pairwise_separation": True,
        }

    @property
    def digest(self) -> str:
        return _json_digest(self._unsigned_obj(), domain=_PANEL_DOMAIN)

    def as_obj(self) -> dict[str, Any]:
        return {**self._unsigned_obj(), "materialized_training_panel_digest": self.digest}


def build_materialized_training_panel_v2(
    opening: CanonicalSupportedOpeningV2,
    scene_indices: Iterable[int],
    *,
    external_generator_receipt_digest: str,
) -> MaterializedTrainingPanelV2:
    """Bind and exactly replay one hidden shared training panel."""

    if type(opening) is not CanonicalSupportedOpeningV2:
        raise TypeError("panel builder requires a CanonicalSupportedOpeningV2")
    return MaterializedTrainingPanelV2(
        catalog_digest=opening.catalog_digest,
        supported_catalog_digest=opening.supported_catalog_digest,
        opening_content_digest=opening.content_digest,
        version_space_rule_ids=opening.version_space_rule_ids,
        version_space_truth_digests=opening.version_space_truth_digests,
        scene_indices=tuple(sorted(scene_indices)),
        external_generator_receipt_digest=external_generator_receipt_digest,
    )


def _materialized_training_panel_from_obj(
    value: object,
) -> MaterializedTrainingPanelV2:
    obj = _require_mapping(
        value,
        (
            "schema_version",
            "panel_kind",
            "catalog_digest",
            "supported_catalog_digest",
            "opening_content_digest",
            "version_space_rules",
            "scene_indices",
            "truth_pattern_rows",
            "rule_balance_rows",
            "external_generator_receipt_digest",
            "external_generator_receipt_verified",
            "external_generator_verification_required",
            "panel_binding_visible_to_model",
            "pairwise_rule_separation_required_here",
            "evaluation_challenge_bank_owns_pairwise_separation",
            "materialized_training_panel_digest",
        ),
        name="materialized training panel",
    )
    if (
        type(obj["schema_version"]) is not int
        or obj["schema_version"] != MATERIALIZED_TRAINING_PANEL_SCHEMA_VERSION
        or obj["panel_kind"] != "evaluator-private-shared-sixteen-scene-training-panel"
    ):
        raise HypothesisCompleteV2Error("materialized training-panel schema identity mismatch")
    for field, required in (
        ("external_generator_receipt_verified", False),
        ("external_generator_verification_required", True),
        ("panel_binding_visible_to_model", False),
        ("pairwise_rule_separation_required_here", False),
        ("evaluation_challenge_bank_owns_pairwise_separation", True),
    ):
        if _require_boolean(obj[field], name=field) is not required:
            raise HypothesisCompleteV2Error(f"materialized training-panel {field} is invalid")
    raw_rules = obj["version_space_rules"]
    raw_scenes = obj["scene_indices"]
    if type(raw_rules) is not list or type(raw_scenes) is not list:
        raise HypothesisCompleteV2Error("panel V0 and scenes must be arrays")
    rule_ids: list[str] = []
    truth_digests: list[str] = []
    for raw in raw_rules:
        row = _require_mapping(raw, ("rule_id", "truth_digest"), name="panel V0 identity")
        if type(row["rule_id"]) is not str:
            raise HypothesisCompleteV2Error("panel V0 rule ID must be a string")
        rule_ids.append(row["rule_id"])
        truth_digests.append(_require_sha256(row["truth_digest"], name="panel truth digest"))
    panel = MaterializedTrainingPanelV2(
        catalog_digest=_require_sha256(obj["catalog_digest"], name="catalog digest"),
        supported_catalog_digest=_require_sha256(
            obj["supported_catalog_digest"], name="supported-catalog digest"
        ),
        opening_content_digest=_require_sha256(obj["opening_content_digest"], name="opening-content digest"),
        version_space_rule_ids=tuple(rule_ids),
        version_space_truth_digests=tuple(truth_digests),
        scene_indices=tuple(
            _require_integer(item, name="panel scene index", maximum=SCENE_COUNT - 1) for item in raw_scenes
        ),
        external_generator_receipt_digest=_require_sha256(
            obj["external_generator_receipt_digest"],
            name="external panel-generator receipt digest",
        ),
    )
    if obj["materialized_training_panel_digest"] != panel.digest or _dump_json(obj) != _dump_json(
        panel.as_obj()
    ):
        raise HypothesisCompleteV2Error("materialized training panel is tampered or inconsistent")
    return panel


def _exact_terminal_balance_audit(
    opening: CanonicalSupportedOpeningV2,
    law: UnconditionalTerminalSceneLawV2,
    panel: MaterializedTrainingPanelV2,
) -> dict[str, Any]:
    entries = tuple(_catalog_entry(rule_id) for rule_id in opening.version_space_rule_ids)
    n0 = len(entries)
    probability_by_scene = {item.scene_index: item.probability for item in law.scene_probabilities}
    law_rows: list[dict[str, Any]] = []
    law_baseline = Fraction()
    for scene_index in sorted(probability_by_scene):
        pattern = _truth_pattern(entries, scene_index)
        accepted_count = pattern.bit_count()
        scene_bayes = Fraction(max(accepted_count, n0 - accepted_count), n0)
        probability = probability_by_scene[scene_index]
        law_baseline += probability * scene_bayes
        law_rows.append(
            {
                "scene_index": scene_index,
                "probability": _fraction_obj(probability),
                "truth_pattern_mask": pattern,
                "accepted_candidate_count": accepted_count,
                "rejected_candidate_count": n0 - accepted_count,
                "no_query_bayes_accuracy": _fraction_obj(scene_bayes),
            }
        )

    rule_mass_rows = []
    for entry in entries:
        accepted_mass = sum(
            (
                probability
                for scene_index, probability in probability_by_scene.items()
                if entry.truth[scene_index]
            ),
            Fraction(),
        )
        rule_mass_rows.append(
            {
                "rule_id": entry.rule_id,
                "truth_digest": entry.truth_digest,
                "accepted_mass": _fraction_obj(accepted_mass),
                "rejected_mass": _fraction_obj(1 - accepted_mass),
            }
        )

    separation_rows: list[dict[str, Any]] = []
    support = tuple(sorted(probability_by_scene))
    for left_position, left in enumerate(entries):
        for right in entries[left_position + 1 :]:
            witness = next(
                (
                    scene_index
                    for scene_index in support
                    if left.truth[scene_index] is not right.truth[scene_index]
                ),
                None,
            )
            separation_rows.append(
                {
                    "left_rule_id": left.rule_id,
                    "right_rule_id": right.rule_id,
                    "witness_scene_index": witness,
                    "separated": witness is not None,
                }
            )

    full_pattern_mask = (1 << n0) - 1
    pattern_masses: dict[int, Fraction] = {}
    for scene_index, probability in probability_by_scene.items():
        pattern = _truth_pattern(entries, scene_index)
        pattern_masses[pattern] = pattern_masses.get(pattern, Fraction()) + probability
    complementary_pattern_mass_rows = [
        {
            "truth_pattern_mask": pattern,
            "complement_truth_pattern_mask": full_pattern_mask ^ pattern,
            "pattern_mass": _fraction_obj(pattern_masses[pattern]),
            "complement_pattern_mass": _fraction_obj(
                pattern_masses.get(full_pattern_mask ^ pattern, Fraction())
            ),
            "exactly_matched": (
                pattern_masses[pattern] == pattern_masses.get(full_pattern_mask ^ pattern, Fraction())
            ),
        }
        for pattern in sorted(pattern_masses)
    ]

    panel_baseline = sum(
        (
            Fraction(
                max(
                    _truth_pattern(entries, scene_index).bit_count(),
                    n0 - _truth_pattern(entries, scene_index).bit_count(),
                ),
                n0 * TRAIN_TERMINAL_ITEM_COUNT,
            )
            for scene_index in panel.scene_indices
        ),
        Fraction(),
    )
    return {
        "prior": {
            "kind": "uniform-over-exact-supported-V0",
            "per_rule_probability": _fraction_obj(Fraction(1, n0)),
        },
        "required_no_query_bayes_accuracy": _fraction_obj(Fraction(1, 2)),
        "law_no_query_bayes_accuracy": _fraction_obj(law_baseline),
        "materialized_panel_no_query_bayes_accuracy": _fraction_obj(panel_baseline),
        "law_support_rows": law_rows,
        "exact_rule_acceptance_mass_rows": rule_mass_rows,
        "law_pairwise_separation_rows": separation_rows,
        "law_complementary_pattern_mass_rows": complementary_pattern_mass_rows,
        "law_support_every_scene_half_half": all(
            row["accepted_candidate_count"] == n0 // 2 for row in law_rows
        ),
        "law_every_rule_exact_half_mass": all(
            row["accepted_mass"] == _fraction_obj(Fraction(1, 2)) for row in rule_mass_rows
        ),
        "law_complementary_pattern_pair_sampling_exact": all(
            row["exactly_matched"] is True for row in complementary_pattern_mass_rows
        ),
        "law_support_pairwise_separation_observed_descriptively": all(
            row["separated"] is True for row in separation_rows
        ),
        "law_support_pairwise_separation_required_here": False,
        "evaluation_challenge_and_query_banks_own_pairwise_separation": True,
        "law_and_panel_no_query_baselines_exact_half": (
            law_baseline == Fraction(1, 2) and panel_baseline == Fraction(1, 2)
        ),
    }


@dataclass(frozen=True, slots=True)
class HiddenOrderDisplayBindingV2:
    """Evaluator-private precommitment for rotation and opening-display order."""

    opening_digest: str
    official_rotation_order_rule_ids: tuple[str, ...]
    opening_display_scene_indices: tuple[int, ...]
    independence_precommitment_digest: str

    def __post_init__(self) -> None:
        _require_sha256(self.opening_digest, name="hidden binding opening digest")
        _require_sha256(
            self.independence_precommitment_digest,
            name="hidden order/display independence precommitment digest",
        )
        if type(self.official_rotation_order_rule_ids) is not tuple or not (
            self.official_rotation_order_rule_ids
        ):
            raise HypothesisCompleteV2Error("hidden Official order must be a nonempty tuple")
        if len(set(self.official_rotation_order_rule_ids)) != len(self.official_rotation_order_rule_ids):
            raise HypothesisCompleteV2Error("hidden Official order must not repeat a rule")
        for rule_id in self.official_rotation_order_rule_ids:
            _catalog_entry(rule_id)
        if type(self.opening_display_scene_indices) is not tuple:
            raise HypothesisCompleteV2Error("hidden opening display order must be a tuple")
        if len(set(self.opening_display_scene_indices)) != len(self.opening_display_scene_indices):
            raise HypothesisCompleteV2Error("hidden opening display order must not repeat a scene")
        for scene_index in self.opening_display_scene_indices:
            _require_integer(
                scene_index,
                name="hidden display scene index",
                maximum=SCENE_COUNT - 1,
            )

    def _unsigned_obj(self) -> dict[str, Any]:
        return {
            "binding_kind": "evaluator-private-independent-order-display-precommitment",
            "opening_digest": self.opening_digest,
            "official_rotation_order_rule_ids": list(self.official_rotation_order_rule_ids),
            "opening_display_scene_indices": list(self.opening_display_scene_indices),
            "independence_precommitment_digest": self.independence_precommitment_digest,
            "binding_metadata_visible_to_model": False,
            "independence_externally_verified": False,
        }

    @property
    def digest(self) -> str:
        return _json_digest(self._unsigned_obj(), domain=_HIDDEN_BINDING_DOMAIN)

    def as_obj(self) -> dict[str, Any]:
        return {**self._unsigned_obj(), "hidden_order_display_binding_digest": self.digest}


def _canonical_private_orders(
    opening: CanonicalSupportedOpeningV2,
    independence_precommitment_digest: str,
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    _require_sha256(
        independence_precommitment_digest,
        name="hidden order/display independence precommitment digest",
    )

    def rank(kind: str, value: str | int) -> str:
        return _json_digest(
            {
                "opening_content_digest": opening.content_digest,
                "independence_precommitment_digest": independence_precommitment_digest,
                "sequence_kind": kind,
                "value": value,
            },
            domain=_HIDDEN_RANK_DOMAIN,
        )

    rule_order = tuple(
        sorted(
            opening.version_space_rule_ids,
            key=lambda rule_id: (rank("official-rotation", rule_id), rule_id),
        )
    )
    scene_order = tuple(
        sorted(
            (item.scene_index for item in opening.observations),
            key=lambda scene_index: (rank("opening-display", scene_index), scene_index),
        )
    )
    return rule_order, scene_order


def build_hidden_order_display_binding_v2(
    opening: CanonicalSupportedOpeningV2,
    *,
    independence_precommitment_digest: str,
    official_rotation_order_rule_ids: Iterable[str] | None = None,
    opening_display_scene_indices: Iterable[int] | None = None,
) -> HiddenOrderDisplayBindingV2:
    """Bind private orders; the caller's precommitment remains externally audited."""

    if type(opening) is not CanonicalSupportedOpeningV2:
        raise TypeError("hidden binding requires a CanonicalSupportedOpeningV2")
    canonical_rule_order, canonical_scene_order = _canonical_private_orders(
        opening,
        independence_precommitment_digest,
    )
    rule_order = (
        canonical_rule_order
        if official_rotation_order_rule_ids is None
        else tuple(official_rotation_order_rule_ids)
    )
    scene_order = (
        canonical_scene_order
        if opening_display_scene_indices is None
        else tuple(opening_display_scene_indices)
    )
    if rule_order != canonical_rule_order or scene_order != canonical_scene_order:
        raise HypothesisCompleteV2Error(
            "hidden orders must equal the canonical hash order derived from the precommitment"
        )
    binding = HiddenOrderDisplayBindingV2(
        opening_digest=opening.digest,
        official_rotation_order_rule_ids=rule_order,
        opening_display_scene_indices=scene_order,
        independence_precommitment_digest=independence_precommitment_digest,
    )
    if (
        set(binding.official_rotation_order_rule_ids) != set(opening.version_space_rule_ids)
        or len(binding.official_rotation_order_rule_ids) != opening.n0
    ):
        raise HypothesisCompleteV2Error("hidden Official order must be exactly one permutation of V0")
    if set(binding.opening_display_scene_indices) != {
        item.scene_index for item in opening.observations
    } or len(binding.opening_display_scene_indices) != len(opening.observations):
        raise HypothesisCompleteV2Error(
            "hidden display order must be exactly one permutation of the canonical opening"
        )
    return binding


def _hidden_binding_from_obj(value: object) -> HiddenOrderDisplayBindingV2:
    obj = _require_mapping(
        value,
        (
            "binding_kind",
            "opening_digest",
            "official_rotation_order_rule_ids",
            "opening_display_scene_indices",
            "independence_precommitment_digest",
            "binding_metadata_visible_to_model",
            "independence_externally_verified",
            "hidden_order_display_binding_digest",
        ),
        name="hidden order/display binding",
    )
    if obj["binding_kind"] != "evaluator-private-independent-order-display-precommitment":
        raise HypothesisCompleteV2Error("hidden order/display binding kind is wrong")
    if _require_boolean(obj["binding_metadata_visible_to_model"], name="hidden-binding model visibility"):
        raise HypothesisCompleteV2Error("hidden order/display metadata must never be model-visible")
    if _require_boolean(obj["independence_externally_verified"], name="hidden-binding external verification"):
        raise HypothesisCompleteV2Error("hidden-order independence remains externally unverified")
    raw_rules = obj["official_rotation_order_rule_ids"]
    raw_scenes = obj["opening_display_scene_indices"]
    if type(raw_rules) is not list or type(raw_scenes) is not list:
        raise HypothesisCompleteV2Error("hidden order/display sequences must be arrays")
    if any(type(rule_id) is not str for rule_id in raw_rules):
        raise HypothesisCompleteV2Error("hidden Official order contains a non-string rule ID")
    binding = HiddenOrderDisplayBindingV2(
        opening_digest=_require_sha256(obj["opening_digest"], name="opening digest"),
        official_rotation_order_rule_ids=tuple(raw_rules),
        opening_display_scene_indices=tuple(
            _require_integer(item, name="display scene index", maximum=SCENE_COUNT - 1) for item in raw_scenes
        ),
        independence_precommitment_digest=_require_sha256(
            obj["independence_precommitment_digest"],
            name="independence precommitment digest",
        ),
    )
    if obj["hidden_order_display_binding_digest"] != binding.digest or _dump_json(obj) != _dump_json(
        binding.as_obj()
    ):
        raise HypothesisCompleteV2Error("hidden order/display binding digest is inconsistent")
    return binding


@dataclass(frozen=True, slots=True)
class HypothesisRotationExecutionV2:
    """One Official identity's objective, collected before the atomic commit."""

    rotation_position: int
    official_rule_id: str
    official_truth_digest: str
    exact_weight_numerator: int
    exact_weight_denominator: int
    static_model_input_digest: str
    pre_update_checkpoint_digest: str
    update_batch_id: str
    optimizer_step_before: int
    optimizer_step_after_objective_collection: int
    context_reset_before_episode: bool = True
    cache_reset_before_episode: bool = True
    objective_collected_before_block_commit: bool = True
    parameter_update_committed_before_block_commit: bool = False

    def __post_init__(self) -> None:
        _require_integer(self.rotation_position, name="rotation position")
        entry = _catalog_entry(self.official_rule_id)
        if self.official_truth_digest != entry.truth_digest:
            raise HypothesisCompleteV2Error("rotation Official truth digest differs from the catalog")
        if isinstance(self.exact_weight_numerator, bool) or self.exact_weight_numerator != 1:
            raise HypothesisCompleteV2Error("every hypothesis rotation must have exact numerator one")
        if self.exact_weight_denominator not in ALLOWED_HYPOTHESIS_COUNTS:
            raise HypothesisCompleteV2Error("rotation weight denominator is not an allowed n0")
        _require_sha256(self.static_model_input_digest, name="static model-input digest")
        _require_sha256(self.pre_update_checkpoint_digest, name="pre-update checkpoint digest")
        _require_identifier(self.update_batch_id, name="update_batch_id")
        _require_integer(self.optimizer_step_before, name="optimizer step before")
        if (
            isinstance(self.optimizer_step_after_objective_collection, bool)
            or self.optimizer_step_after_objective_collection != self.optimizer_step_before
        ):
            raise HypothesisCompleteV2Error("an optimizer update occurred before the block commit")
        for name in (
            "context_reset_before_episode",
            "cache_reset_before_episode",
            "objective_collected_before_block_commit",
            "parameter_update_committed_before_block_commit",
        ):
            _require_boolean(getattr(self, name), name=name)
        if not self.context_reset_before_episode or not self.cache_reset_before_episode:
            raise HypothesisCompleteV2Error("context and cache must reset before every rotation")
        if not self.objective_collected_before_block_commit:
            raise HypothesisCompleteV2Error("every rotation objective must be collected before commit")
        if self.parameter_update_committed_before_block_commit:
            raise HypothesisCompleteV2Error("no parameter update may occur during or between rotations")

    def _unsigned_obj(self) -> dict[str, Any]:
        return {
            "rotation_position": self.rotation_position,
            "official_rule_id": self.official_rule_id,
            "official_truth_digest": self.official_truth_digest,
            "exact_objective_weight": {
                "numerator": self.exact_weight_numerator,
                "denominator": self.exact_weight_denominator,
            },
            "static_model_input_digest": self.static_model_input_digest,
            "pre_update_checkpoint_digest": self.pre_update_checkpoint_digest,
            "update_batch_id": self.update_batch_id,
            "optimizer_step_before": self.optimizer_step_before,
            "optimizer_step_after_objective_collection": (self.optimizer_step_after_objective_collection),
            "context_reset_before_episode": self.context_reset_before_episode,
            "cache_reset_before_episode": self.cache_reset_before_episode,
            "objective_collected_before_block_commit": self.objective_collected_before_block_commit,
            "parameter_update_committed_before_block_commit": (
                self.parameter_update_committed_before_block_commit
            ),
        }

    @property
    def digest(self) -> str:
        return _json_digest(self._unsigned_obj(), domain=_ROTATION_DOMAIN)

    def as_obj(self) -> dict[str, Any]:
        return {**self._unsigned_obj(), "rotation_execution_digest": self.digest}


def _rotation_from_obj(value: object) -> HypothesisRotationExecutionV2:
    obj = _require_mapping(
        value,
        (
            "rotation_position",
            "official_rule_id",
            "official_truth_digest",
            "exact_objective_weight",
            "static_model_input_digest",
            "pre_update_checkpoint_digest",
            "update_batch_id",
            "optimizer_step_before",
            "optimizer_step_after_objective_collection",
            "context_reset_before_episode",
            "cache_reset_before_episode",
            "objective_collected_before_block_commit",
            "parameter_update_committed_before_block_commit",
            "rotation_execution_digest",
        ),
        name="hypothesis rotation execution",
    )
    weight = _require_mapping(
        obj["exact_objective_weight"],
        ("numerator", "denominator"),
        name="exact objective weight",
    )
    rotation = HypothesisRotationExecutionV2(
        rotation_position=_require_integer(obj["rotation_position"], name="rotation position"),
        official_rule_id=cast(str, obj["official_rule_id"]),
        official_truth_digest=_require_sha256(obj["official_truth_digest"], name="Official truth digest"),
        exact_weight_numerator=_require_integer(weight["numerator"], name="weight numerator", minimum=1),
        exact_weight_denominator=_require_integer(
            weight["denominator"], name="weight denominator", minimum=1
        ),
        static_model_input_digest=_require_sha256(
            obj["static_model_input_digest"], name="static model-input digest"
        ),
        pre_update_checkpoint_digest=_require_sha256(
            obj["pre_update_checkpoint_digest"], name="pre-update checkpoint digest"
        ),
        update_batch_id=_require_identifier(obj["update_batch_id"], name="update_batch_id"),
        optimizer_step_before=_require_integer(obj["optimizer_step_before"], name="optimizer step before"),
        optimizer_step_after_objective_collection=_require_integer(
            obj["optimizer_step_after_objective_collection"],
            name="optimizer step after objective collection",
        ),
        context_reset_before_episode=_require_boolean(
            obj["context_reset_before_episode"], name="context reset"
        ),
        cache_reset_before_episode=_require_boolean(obj["cache_reset_before_episode"], name="cache reset"),
        objective_collected_before_block_commit=_require_boolean(
            obj["objective_collected_before_block_commit"], name="objective-before-commit"
        ),
        parameter_update_committed_before_block_commit=_require_boolean(
            obj["parameter_update_committed_before_block_commit"], name="update-before-commit"
        ),
    )
    if obj["rotation_execution_digest"] != rotation.digest or _dump_json(obj) != _dump_json(
        rotation.as_obj()
    ):
        raise HypothesisCompleteV2Error("rotation execution digest or metadata is inconsistent")
    return rotation


@dataclass(frozen=True, slots=True)
class AtomicBlockCommitV2:
    """The single optimizer transition after all ``n0`` objectives exist."""

    update_batch_id: str
    pre_update_checkpoint_digest: str
    post_update_checkpoint_digest: str
    optimizer_step_before: int
    optimizer_step_after: int
    collected_objective_count: int
    commit_disposition: str
    parameter_updates_before_commit: int = 0
    atomic_commit_count: int = 1
    all_objectives_collected_before_commit: bool = True

    def __post_init__(self) -> None:
        _require_identifier(self.update_batch_id, name="update_batch_id")
        _require_sha256(self.pre_update_checkpoint_digest, name="pre-update checkpoint digest")
        _require_sha256(self.post_update_checkpoint_digest, name="post-update checkpoint digest")
        if type(self.commit_disposition) is not str or self.commit_disposition not in {
            "committed_state_changed",
            "zero_gradient_no_state_change",
        }:
            raise HypothesisCompleteV2Error("atomic commit has an unknown disposition")
        state_changed = self.post_update_checkpoint_digest != self.pre_update_checkpoint_digest
        if state_changed is not (self.commit_disposition == "committed_state_changed"):
            raise HypothesisCompleteV2Error(
                "atomic commit disposition disagrees with the checkpoint transition"
            )
        _require_integer(self.optimizer_step_before, name="optimizer step before")
        if isinstance(self.optimizer_step_after, bool) or self.optimizer_step_after != (
            self.optimizer_step_before + 1
        ):
            raise HypothesisCompleteV2Error("block must commit exactly one optimizer step")
        if self.collected_objective_count not in ALLOWED_HYPOTHESIS_COUNTS:
            raise HypothesisCompleteV2Error("commit objective count must equal an allowed n0")
        if (
            isinstance(self.parameter_updates_before_commit, bool)
            or self.parameter_updates_before_commit != 0
        ):
            raise HypothesisCompleteV2Error("no parameter update may occur before the atomic commit")
        if isinstance(self.atomic_commit_count, bool) or self.atomic_commit_count != 1:
            raise HypothesisCompleteV2Error("a complete block must contain exactly one commit")
        if type(self.all_objectives_collected_before_commit) is not bool or not (
            self.all_objectives_collected_before_commit
        ):
            raise HypothesisCompleteV2Error("all objectives must exist before the one atomic commit")

    def _unsigned_obj(self) -> dict[str, Any]:
        return {
            "update_batch_id": self.update_batch_id,
            "pre_update_checkpoint_digest": self.pre_update_checkpoint_digest,
            "post_update_checkpoint_digest": self.post_update_checkpoint_digest,
            "optimizer_step_before": self.optimizer_step_before,
            "optimizer_step_after": self.optimizer_step_after,
            "collected_objective_count": self.collected_objective_count,
            "commit_disposition": self.commit_disposition,
            "parameter_updates_before_commit": self.parameter_updates_before_commit,
            "atomic_commit_count": self.atomic_commit_count,
            "all_objectives_collected_before_commit": self.all_objectives_collected_before_commit,
        }

    @property
    def digest(self) -> str:
        return _json_digest(self._unsigned_obj(), domain=_COMMIT_DOMAIN)

    def as_obj(self) -> dict[str, Any]:
        return {**self._unsigned_obj(), "atomic_commit_digest": self.digest}


def _commit_from_obj(value: object) -> AtomicBlockCommitV2:
    obj = _require_mapping(
        value,
        (
            "update_batch_id",
            "pre_update_checkpoint_digest",
            "post_update_checkpoint_digest",
            "optimizer_step_before",
            "optimizer_step_after",
            "collected_objective_count",
            "commit_disposition",
            "parameter_updates_before_commit",
            "atomic_commit_count",
            "all_objectives_collected_before_commit",
            "atomic_commit_digest",
        ),
        name="atomic block commit",
    )
    commit = AtomicBlockCommitV2(
        update_batch_id=_require_identifier(obj["update_batch_id"], name="update_batch_id"),
        pre_update_checkpoint_digest=_require_sha256(
            obj["pre_update_checkpoint_digest"], name="pre-update checkpoint digest"
        ),
        post_update_checkpoint_digest=_require_sha256(
            obj["post_update_checkpoint_digest"], name="post-update checkpoint digest"
        ),
        optimizer_step_before=_require_integer(obj["optimizer_step_before"], name="optimizer step before"),
        optimizer_step_after=_require_integer(
            obj["optimizer_step_after"], name="optimizer step after", minimum=1
        ),
        collected_objective_count=_require_integer(
            obj["collected_objective_count"], name="collected objective count", minimum=1
        ),
        commit_disposition=cast(str, obj["commit_disposition"]),
        parameter_updates_before_commit=_require_integer(
            obj["parameter_updates_before_commit"], name="updates before commit"
        ),
        atomic_commit_count=_require_integer(
            obj["atomic_commit_count"], name="atomic commit count", minimum=1
        ),
        all_objectives_collected_before_commit=_require_boolean(
            obj["all_objectives_collected_before_commit"], name="all objectives before commit"
        ),
    )
    if obj["atomic_commit_digest"] != commit.digest or _dump_json(obj) != _dump_json(commit.as_obj()):
        raise HypothesisCompleteV2Error("atomic commit digest or metadata is inconsistent")
    return commit


def _prevalence_bin(entry: CatalogEntry) -> str:
    scaled = entry.truth.true_count * 8
    if scaled < SCENE_COUNT * 3:
        return "p25_to_lt_p37_5"
    if scaled < SCENE_COUNT * 4:
        return "p37_5_to_lt_p50"
    if scaled < SCENE_COUNT * 5:
        return "p50_to_lt_p62_5"
    return "p62_5_to_p75"


def _candidate_features(entry: CatalogEntry) -> dict[str, str]:
    family = classify_catalog_identity_v2(entry)
    if family not in CANDIDATE_RULE_FAMILIES:
        raise HypothesisCompleteV2Error("unsupported rule reached a hypothesis feature histogram")
    rule = entry.rule
    atom_family: str
    if type(rule) is RuleLiteral:
        literal_count = "1"
        operator = "literal"
        atom_family = rule.atom.op
    elif type(rule) is BinaryRule:
        literal_count = "2"
        operator = rule.op
        atom_family = "+".join(sorted(item.atom.op for item in rule.args))
    else:  # pragma: no cover - closed public grammar
        raise HypothesisCompleteV2Error("unknown public AST reached the feature audit")
    return {
        "family": family,
        "literal_count": literal_count,
        "operator": operator,
        "placard_inclusion": "yes" if family == "placard_literal" else "no",
        "atom_family": atom_family,
        "prevalence_bin": _prevalence_bin(entry),
    }


def _feature_histograms(openings: Iterable[CanonicalSupportedOpeningV2]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str]] = Counter()
    for opening in openings:
        for rule_id in opening.version_space_rule_ids:
            features = _candidate_features(_catalog_entry(rule_id))
            for feature in _FEATURE_ORDER:
                counts[(feature, features[feature])] += 1
    return [
        {
            "feature": feature,
            "value": value,
            "candidate_count": counts[(feature, value)],
            "official_count": counts[(feature, value)],
        }
        for feature in _FEATURE_ORDER
        for value in sorted(value for observed_feature, value in counts if observed_feature == feature)
    ]


def _assert_role_neutral_static_input(value: object) -> None:
    if type(value) is dict:
        for key, child in value.items():
            normalized = key.lower().replace("-", "_")
            if (
                normalized in _MODEL_VISIBLE_FORBIDDEN_KEYS
                or "candidate_role" in normalized
                or normalized.startswith("cover_")
                or normalized.startswith("stage_")
                or normalized.startswith("official_")
            ):
                raise HypothesisCompleteV2Error(
                    f"model-visible static data contains forbidden role marker: {key!r}"
                )
            _assert_role_neutral_static_input(child)
    elif type(value) is list:
        for child in value:
            _assert_role_neutral_static_input(child)


def _assert_exact_model_visible_static_schema(value: object) -> None:
    """Fail closed on additions to the only model-visible static payload."""

    obj = _require_mapping(
        value,
        (
            "opening_examples",
            "renderer_contract",
            "public_train_terminal_scene_law",
        ),
        name="model-visible static data",
    )
    examples = obj["opening_examples"]
    if type(examples) is not list or len(examples) != OPENING_DEMONSTRATION_COUNT:
        raise HypothesisCompleteV2Error(
            "model-visible opening examples must be the exact ten-item public payload"
        )
    for raw in examples:
        example = _require_mapping(
            raw,
            ("accepted", "scene_text"),
            name="model-visible opening example",
        )
        _require_boolean(example["accepted"], name="model-visible accepted label")
        if type(example["scene_text"]) is not str or not example["scene_text"]:
            raise HypothesisCompleteV2Error("model-visible scene text must be nonempty text")
    renderer = _require_mapping(
        obj["renderer_contract"],
        ("renderer_name", "renderer_registry_digest"),
        name="model-visible renderer contract",
    )
    if type(renderer["renderer_name"]) is not str or renderer["renderer_name"] not in TRAIN_RENDERERS:
        raise HypothesisCompleteV2Error("model-visible renderer is not registered for training")
    if (
        _require_sha256(
            renderer["renderer_registry_digest"],
            name="model-visible renderer-registry digest",
        )
        != renderer_digest()
    ):
        raise HypothesisCompleteV2Error("model-visible renderer-registry digest is wrong")
    unconditional_terminal_scene_law_v2_from_obj(obj["public_train_terminal_scene_law"])
    _assert_role_neutral_static_input(obj)


def _private_bindings_absent_from_static(
    block: HypothesisCompleteTrainingBlockV2,
) -> bool:
    """Check private identifiers and receipts, in addition to the exact key whitelist."""

    def string_leaves(value: object) -> set[str]:
        if type(value) is dict:
            return set().union(*(string_leaves(child) for child in cast(Mapping[str, Any], value).values()))
        if type(value) is list:
            return set().union(*(string_leaves(child) for child in value))
        return {value} if type(value) is str else set()

    visible_strings = string_leaves(block.static_model_input_obj())
    sensitive_values = {
        block.block_id,
        block.opening.opening_id,
        block.opening.digest,
        block.opening.content_digest,
        block.opening.scene_set_digest,
        *block.opening.version_space_rule_ids,
        *block.opening.version_space_truth_digests,
        block.materialized_training_panel.digest,
        block.materialized_training_panel.external_generator_receipt_digest,
        block.hidden_order_display_binding.digest,
        block.hidden_order_display_binding.independence_precommitment_digest,
        block.atomic_commit.update_batch_id,
        block.atomic_commit.pre_update_checkpoint_digest,
        block.atomic_commit.post_update_checkpoint_digest,
        *(rotation.digest for rotation in block.rotations),
    }
    return sensitive_values.isdisjoint(visible_strings)


@dataclass(frozen=True, slots=True)
class HypothesisCompleteTrainingBlockV2:
    """One exact ``V0`` permutation collected from a single checkpoint."""

    bank_position: int
    block_id: str
    opening: CanonicalSupportedOpeningV2
    terminal_law: UnconditionalTerminalSceneLawV2
    materialized_training_panel: MaterializedTrainingPanelV2
    renderer_name: str
    renderer_registry_digest: str
    hidden_order_display_binding: HiddenOrderDisplayBindingV2
    rotations: tuple[HypothesisRotationExecutionV2, ...]
    atomic_commit: AtomicBlockCommitV2

    def __post_init__(self) -> None:
        _require_integer(self.bank_position, name="bank position")
        _require_identifier(self.block_id, name="block_id")
        if type(self.opening) is not CanonicalSupportedOpeningV2:
            raise HypothesisCompleteV2Error("training block requires a canonical supported opening")
        if type(self.terminal_law) is not UnconditionalTerminalSceneLawV2:
            raise HypothesisCompleteV2Error("training block requires one unconditional terminal law")
        if type(self.materialized_training_panel) is not MaterializedTrainingPanelV2:
            raise HypothesisCompleteV2Error("training block requires one materialized training panel")
        if type(self.renderer_name) is not str or self.renderer_name not in TRAIN_RENDERERS:
            raise HypothesisCompleteV2Error("training block renderer must be a registered train renderer")
        if self.renderer_registry_digest != renderer_digest():
            raise HypothesisCompleteV2Error("renderer registry digest differs from the public renderer")
        if type(self.hidden_order_display_binding) is not HiddenOrderDisplayBindingV2:
            raise HypothesisCompleteV2Error("training block requires a private order/display binding")
        hidden = self.hidden_order_display_binding
        if hidden.opening_digest != self.opening.digest:
            raise HypothesisCompleteV2Error("hidden order/display binding refers to another opening")
        expected_rule_order, expected_scene_order = _canonical_private_orders(
            self.opening,
            hidden.independence_precommitment_digest,
        )
        if (
            hidden.official_rotation_order_rule_ids != expected_rule_order
            or hidden.opening_display_scene_indices != expected_scene_order
        ):
            raise HypothesisCompleteV2Error(
                "hidden order/display binding differs from its canonical precommitment order"
            )
        if (
            set(hidden.official_rotation_order_rule_ids) != set(self.opening.version_space_rule_ids)
            or len(hidden.official_rotation_order_rule_ids) != self.opening.n0
        ):
            raise HypothesisCompleteV2Error("hidden Official order is not an exact V0 permutation")
        opening_scenes = {item.scene_index for item in self.opening.observations}
        if set(hidden.opening_display_scene_indices) != opening_scenes or len(
            hidden.opening_display_scene_indices
        ) != len(opening_scenes):
            raise HypothesisCompleteV2Error("hidden display order is not an exact opening permutation")
        if self.terminal_law.catalog_digest != self.opening.catalog_digest or (
            self.terminal_law.supported_catalog_digest != self.opening.supported_catalog_digest
        ):
            raise HypothesisCompleteV2Error("opening and terminal law use different catalog contracts")
        panel = self.materialized_training_panel
        if (
            panel.catalog_digest != self.opening.catalog_digest
            or panel.supported_catalog_digest != self.opening.supported_catalog_digest
            or panel.opening_content_digest != self.opening.content_digest
            or panel.version_space_rule_ids != self.opening.version_space_rule_ids
            or panel.version_space_truth_digests != self.opening.version_space_truth_digests
        ):
            raise HypothesisCompleteV2Error("materialized panel differs from its exact opening V0")
        law_support = {item.scene_index for item in self.terminal_law.scene_probabilities}
        if opening_scenes & law_support or opening_scenes & set(panel.scene_indices):
            raise HypothesisCompleteV2Error(
                "opening, public law support, and materialized training panel must be disjoint"
            )
        if not set(panel.scene_indices) <= law_support:
            raise HypothesisCompleteV2Error(
                "every materialized training-panel scene must lie in the public law support"
            )
        terminal_audit = _exact_terminal_balance_audit(
            self.opening,
            self.terminal_law,
            panel,
        )
        for field in (
            "law_support_every_scene_half_half",
            "law_every_rule_exact_half_mass",
            "law_complementary_pattern_pair_sampling_exact",
            "law_and_panel_no_query_baselines_exact_half",
        ):
            if terminal_audit[field] is not True:
                raise HypothesisCompleteV2Error(f"terminal law/panel exact gate failed: {field}")
        if type(self.atomic_commit) is not AtomicBlockCommitV2:
            raise HypothesisCompleteV2Error("training block requires one atomic commit")

        if type(self.rotations) is not tuple or len(self.rotations) != self.opening.n0:
            raise HypothesisCompleteV2Error("block rotations must be complete: exactly one per V0 identity")
        if any(type(rotation) is not HypothesisRotationExecutionV2 for rotation in self.rotations):
            raise HypothesisCompleteV2Error("training block contains a foreign rotation type")
        if tuple(rotation.rotation_position for rotation in self.rotations) != tuple(range(self.opening.n0)):
            raise HypothesisCompleteV2Error("rotation positions must be complete and canonical")
        if tuple(rotation.official_rule_id for rotation in self.rotations) != (
            hidden.official_rotation_order_rule_ids
        ):
            raise HypothesisCompleteV2Error("rotation Official order differs from the hidden precommitment")
        if {rotation.official_rule_id for rotation in self.rotations} != set(
            self.opening.version_space_rule_ids
        ):
            raise HypothesisCompleteV2Error("each V0 identity must be Official exactly once")
        truth_by_id = dict(
            zip(
                self.opening.version_space_rule_ids,
                self.opening.version_space_truth_digests,
                strict=True,
            )
        )
        static_digest = self.static_model_input_digest
        for rotation in self.rotations:
            if rotation.official_truth_digest != truth_by_id[rotation.official_rule_id]:
                raise HypothesisCompleteV2Error("rotation truth identity differs from opening V0")
            if (rotation.exact_weight_numerator, rotation.exact_weight_denominator) != (
                1,
                self.opening.n0,
            ):
                raise HypothesisCompleteV2Error("rotation objectives must have exact weight 1/n0")
            if rotation.static_model_input_digest != static_digest:
                raise HypothesisCompleteV2Error("static model-visible input differs across rotations")
            if (
                rotation.pre_update_checkpoint_digest != self.atomic_commit.pre_update_checkpoint_digest
                or rotation.update_batch_id != self.atomic_commit.update_batch_id
                or rotation.optimizer_step_before != self.atomic_commit.optimizer_step_before
                or rotation.optimizer_step_after_objective_collection
                != self.atomic_commit.optimizer_step_before
            ):
                raise HypothesisCompleteV2Error(
                    "rotations do not share the atomic commit's checkpoint, batch, and optimizer step"
                )
        if self.atomic_commit.collected_objective_count != self.opening.n0:
            raise HypothesisCompleteV2Error("atomic commit did not collect every V0 objective")
        _assert_exact_model_visible_static_schema(self.static_model_input_obj())
        if not _private_bindings_absent_from_static(self):
            raise HypothesisCompleteV2Error(
                "evaluator-private bindings entered the model-visible static payload"
            )

    def static_model_input_obj(self) -> dict[str, Any]:
        observations = {item.scene_index: item for item in self.opening.observations}
        renderer = cast(RendererName, self.renderer_name)
        return {
            "opening_examples": [
                {
                    "accepted": observations[scene_index].accepted,
                    "scene_text": render_scene(scene_at(scene_index), renderer),
                }
                for scene_index in self.hidden_order_display_binding.opening_display_scene_indices
            ],
            "renderer_contract": {
                "renderer_name": self.renderer_name,
                "renderer_registry_digest": self.renderer_registry_digest,
            },
            "public_train_terminal_scene_law": self.terminal_law.as_obj(),
        }

    @property
    def static_model_input_digest(self) -> str:
        value = self.static_model_input_obj()
        _assert_exact_model_visible_static_schema(value)
        return _json_digest(value, domain=_STATIC_INPUT_DOMAIN)

    def _candidate_official_counts(self) -> list[dict[str, Any]]:
        counts = Counter(rotation.official_rule_id for rotation in self.rotations)
        return [
            {"rule_id": rule_id, "official_count": counts[rule_id]}
            for rule_id in self.opening.version_space_rule_ids
        ]

    def _checks(self) -> dict[str, bool]:
        rotations = self.rotations
        terminal_audit = _exact_terminal_balance_audit(
            self.opening,
            self.terminal_law,
            self.materialized_training_panel,
        )
        return {
            "full_catalog_and_supported_allowlist_bound": True,
            "opening_v0_exactly_recomputed": len(self.opening.version_space_rule_ids) == self.opening.n0,
            "every_v0_identity_official_exactly_once": all(
                row["official_count"] == 1 for row in self._candidate_official_counts()
            ),
            "same_role_neutral_static_model_input": len(
                {rotation.static_model_input_digest for rotation in rotations}
            )
            == 1,
            "single_unconditional_rule_oblivious_terminal_law": True,
            "public_law_support_at_least_sixteen": len(self.terminal_law.scene_probabilities)
            >= TRAIN_TERMINAL_ITEM_COUNT,
            "public_law_every_support_scene_half_half": cast(
                bool,
                terminal_audit["law_support_every_scene_half_half"],
            ),
            "public_law_every_rule_exact_half_mass": cast(
                bool,
                terminal_audit["law_every_rule_exact_half_mass"],
            ),
            "public_law_complementary_pattern_pair_sampling_exact": cast(
                bool,
                terminal_audit["law_complementary_pattern_pair_sampling_exact"],
            ),
            "pairwise_identification_delegated_to_challenge_and_query_banks": (
                terminal_audit["law_support_pairwise_separation_required_here"] is False
                and terminal_audit["evaluation_challenge_and_query_banks_own_pairwise_separation"] is True
            ),
            "materialized_panel_shared_balanced_and_private": True,
            "law_and_panel_no_query_bayes_baselines_exact_half": cast(
                bool,
                terminal_audit["law_and_panel_no_query_baselines_exact_half"],
            ),
            "equal_exact_one_over_n0_weights": all(
                (rotation.exact_weight_numerator, rotation.exact_weight_denominator) == (1, self.opening.n0)
                for rotation in rotations
            ),
            "same_pre_update_checkpoint_optimizer_step_and_batch": len(
                {
                    (
                        rotation.pre_update_checkpoint_digest,
                        rotation.optimizer_step_before,
                        rotation.update_batch_id,
                    )
                    for rotation in rotations
                }
            )
            == 1,
            "all_contexts_and_caches_reset": all(
                rotation.context_reset_before_episode and rotation.cache_reset_before_episode
                for rotation in rotations
            ),
            "no_update_and_all_objectives_before_commit": all(
                not rotation.parameter_update_committed_before_block_commit
                and rotation.objective_collected_before_block_commit
                and rotation.optimizer_step_after_objective_collection == rotation.optimizer_step_before
                for rotation in rotations
            ),
            "exactly_one_optimizer_step_after_all_rotations": (
                self.atomic_commit.atomic_commit_count == 1
                and self.atomic_commit.optimizer_step_after == self.atomic_commit.optimizer_step_before + 1
                and self.atomic_commit.collected_objective_count == self.opening.n0
            ),
            "exact_model_visible_static_whitelist_enforced": True,
            "all_private_bindings_absent_from_model_input": _private_bindings_absent_from_static(self),
        }

    @property
    def structural_audit_passed(self) -> bool:
        return all(self._checks().values())

    def _unsigned_obj(self) -> dict[str, Any]:
        return {
            "schema_version": HYPOTHESIS_COMPLETE_TRAINING_SCHEMA_VERSION,
            "report_kind": _BLOCK_KIND,
            "authorization": dict(_AUTHORIZATION),
            "catalog_digest": self.opening.catalog_digest,
            "supported_catalog_digest": self.opening.supported_catalog_digest,
            "bank_position": self.bank_position,
            "block_id": self.block_id,
            "n0": self.opening.n0,
            "opening": self.opening.as_obj(),
            "unconditional_terminal_scene_law": self.terminal_law.as_obj(),
            "materialized_training_panel": self.materialized_training_panel.as_obj(),
            "exact_terminal_balance_audit": _exact_terminal_balance_audit(
                self.opening,
                self.terminal_law,
                self.materialized_training_panel,
            ),
            "renderer_binding": {
                "renderer_name": self.renderer_name,
                "renderer_registry_digest": self.renderer_registry_digest,
            },
            "hidden_order_display_binding": self.hidden_order_display_binding.as_obj(),
            "model_visible_static_data": self.static_model_input_obj(),
            "model_visible_static_data_digest": self.static_model_input_digest,
            "rotations": [rotation.as_obj() for rotation in self.rotations],
            "atomic_commit": self.atomic_commit.as_obj(),
            "candidate_official_counts": self._candidate_official_counts(),
            "descriptive_feature_histograms": _feature_histograms((self.opening,)),
            "checks": [{"name": name, "passed": passed} for name, passed in self._checks().items()],
            "structural_audit_passed": self.structural_audit_passed,
            "substitution_contract": {
                "legacy_three_role_block_may_substitute": False,
                "legacy_three_role_training_audit_may_substitute": False,
                "evaluation_quartet_audit_may_substitute": False,
                "powered_stress_audit_may_substitute": False,
            },
        }

    @property
    def digest(self) -> str:
        return _json_digest(self._unsigned_obj(), domain=_BLOCK_DOMAIN)

    def as_obj(self) -> dict[str, Any]:
        return {**self._unsigned_obj(), "hypothesis_complete_block_digest": self.digest}


def build_hypothesis_complete_training_block_v2(
    opening: CanonicalSupportedOpeningV2,
    terminal_law: UnconditionalTerminalSceneLawV2,
    materialized_training_panel: MaterializedTrainingPanelV2,
    hidden_order_display_binding: HiddenOrderDisplayBindingV2,
    *,
    bank_position: int,
    block_id: str,
    renderer_name: str,
    pre_update_checkpoint_digest: str,
    post_update_checkpoint_digest: str,
    update_batch_id: str,
    optimizer_step_before: int,
) -> HypothesisCompleteTrainingBlockV2:
    """Build all ``n0`` objectives and their single planned atomic commit."""

    if type(opening) is not CanonicalSupportedOpeningV2:
        raise TypeError("block builder requires a CanonicalSupportedOpeningV2")
    if type(terminal_law) is not UnconditionalTerminalSceneLawV2:
        raise TypeError("block builder requires an UnconditionalTerminalSceneLawV2")
    if type(materialized_training_panel) is not MaterializedTrainingPanelV2:
        raise TypeError("block builder requires a MaterializedTrainingPanelV2")
    if type(hidden_order_display_binding) is not HiddenOrderDisplayBindingV2:
        raise TypeError("block builder requires a HiddenOrderDisplayBindingV2")
    renderer_registry = renderer_digest()
    observations = {item.scene_index: item for item in opening.observations}
    renderer = cast(RendererName, renderer_name)
    static_input = {
        "opening_examples": [
            {
                "accepted": observations[scene_index].accepted,
                "scene_text": render_scene(scene_at(scene_index), renderer),
            }
            for scene_index in hidden_order_display_binding.opening_display_scene_indices
        ],
        "renderer_contract": {
            "renderer_name": renderer_name,
            "renderer_registry_digest": renderer_registry,
        },
        "public_train_terminal_scene_law": terminal_law.as_obj(),
    }
    _assert_exact_model_visible_static_schema(static_input)
    static_digest = _json_digest(static_input, domain=_STATIC_INPUT_DOMAIN)
    rotations = tuple(
        HypothesisRotationExecutionV2(
            rotation_position=position,
            official_rule_id=rule_id,
            official_truth_digest=_catalog_entry(rule_id).truth_digest,
            exact_weight_numerator=1,
            exact_weight_denominator=opening.n0,
            static_model_input_digest=static_digest,
            pre_update_checkpoint_digest=pre_update_checkpoint_digest,
            update_batch_id=update_batch_id,
            optimizer_step_before=optimizer_step_before,
            optimizer_step_after_objective_collection=optimizer_step_before,
        )
        for position, rule_id in enumerate(hidden_order_display_binding.official_rotation_order_rule_ids)
    )
    commit = AtomicBlockCommitV2(
        update_batch_id=update_batch_id,
        pre_update_checkpoint_digest=pre_update_checkpoint_digest,
        post_update_checkpoint_digest=post_update_checkpoint_digest,
        optimizer_step_before=optimizer_step_before,
        optimizer_step_after=optimizer_step_before + 1,
        collected_objective_count=opening.n0,
        commit_disposition=(
            "committed_state_changed"
            if post_update_checkpoint_digest != pre_update_checkpoint_digest
            else "zero_gradient_no_state_change"
        ),
    )
    return HypothesisCompleteTrainingBlockV2(
        bank_position=bank_position,
        block_id=block_id,
        opening=opening,
        terminal_law=terminal_law,
        materialized_training_panel=materialized_training_panel,
        renderer_name=renderer_name,
        renderer_registry_digest=renderer_registry,
        hidden_order_display_binding=hidden_order_display_binding,
        rotations=rotations,
        atomic_commit=commit,
    )


def hypothesis_complete_training_block_v2_from_obj(
    value: object,
    *,
    expected_digest: str | None = None,
) -> HypothesisCompleteTrainingBlockV2:
    _reject_legacy_three_role_substitute(value)
    obj = _require_mapping(
        value,
        (
            "schema_version",
            "report_kind",
            "authorization",
            "catalog_digest",
            "supported_catalog_digest",
            "bank_position",
            "block_id",
            "n0",
            "opening",
            "unconditional_terminal_scene_law",
            "materialized_training_panel",
            "exact_terminal_balance_audit",
            "renderer_binding",
            "hidden_order_display_binding",
            "model_visible_static_data",
            "model_visible_static_data_digest",
            "rotations",
            "atomic_commit",
            "candidate_official_counts",
            "descriptive_feature_histograms",
            "checks",
            "structural_audit_passed",
            "substitution_contract",
            "hypothesis_complete_block_digest",
        ),
        name="hypothesis-complete training block",
    )
    if (
        type(obj["schema_version"]) is not int
        or obj["schema_version"] != HYPOTHESIS_COMPLETE_TRAINING_SCHEMA_VERSION
        or obj["report_kind"] != _BLOCK_KIND
    ):
        raise HypothesisCompleteV2Error("hypothesis-complete block schema identity mismatch")
    _require_authorization(obj["authorization"])
    renderer_binding = _require_mapping(
        obj["renderer_binding"],
        ("renderer_name", "renderer_registry_digest"),
        name="renderer binding",
    )
    raw_rotations = obj["rotations"]
    if type(raw_rotations) is not list:
        raise HypothesisCompleteV2Error("hypothesis rotations must be an array")
    block = HypothesisCompleteTrainingBlockV2(
        bank_position=_require_integer(obj["bank_position"], name="bank position"),
        block_id=_require_identifier(obj["block_id"], name="block_id"),
        opening=canonical_supported_opening_v2_from_obj(obj["opening"]),
        terminal_law=unconditional_terminal_scene_law_v2_from_obj(obj["unconditional_terminal_scene_law"]),
        materialized_training_panel=_materialized_training_panel_from_obj(obj["materialized_training_panel"]),
        renderer_name=cast(str, renderer_binding["renderer_name"]),
        renderer_registry_digest=_require_sha256(
            renderer_binding["renderer_registry_digest"], name="renderer registry digest"
        ),
        hidden_order_display_binding=_hidden_binding_from_obj(obj["hidden_order_display_binding"]),
        rotations=tuple(_rotation_from_obj(item) for item in raw_rotations),
        atomic_commit=_commit_from_obj(obj["atomic_commit"]),
    )
    if obj["catalog_digest"] != block.opening.catalog_digest or (
        obj["supported_catalog_digest"] != block.opening.supported_catalog_digest
    ):
        raise HypothesisCompleteV2Error("block-level catalog bindings are inconsistent")
    if _require_integer(obj["n0"], name="n0", minimum=1) != block.opening.n0:
        raise HypothesisCompleteV2Error("block-level n0 is inconsistent")
    _assert_exact_model_visible_static_schema(obj["model_visible_static_data"])
    if obj["hypothesis_complete_block_digest"] != block.digest:
        raise HypothesisCompleteV2Error("hypothesis-complete block digest mismatch")
    if expected_digest is not None and block.digest != _require_sha256(
        expected_digest, name="expected block digest"
    ):
        raise HypothesisCompleteV2Error("block differs from the externally expected digest")
    if _dump_json(obj) != _dump_json(block.as_obj()):
        raise HypothesisCompleteV2Error("hypothesis-complete block contains tampered derived metadata")
    return block


def verify_hypothesis_complete_training_block_v2(
    block: HypothesisCompleteTrainingBlockV2,
) -> HypothesisCompleteTrainingBlockV2:
    if type(block) is not HypothesisCompleteTrainingBlockV2:
        raise TypeError("verify requires a HypothesisCompleteTrainingBlockV2")
    return hypothesis_complete_training_block_v2_from_obj(block.as_obj(), expected_digest=block.digest)


def serialize_hypothesis_complete_training_block_v2(
    block: HypothesisCompleteTrainingBlockV2,
) -> str:
    return _dump_json(verify_hypothesis_complete_training_block_v2(block).as_obj())


def parse_hypothesis_complete_training_block_v2(
    text: str,
    *,
    expected_digest: str | None = None,
) -> HypothesisCompleteTrainingBlockV2:
    block = hypothesis_complete_training_block_v2_from_obj(_load_json(text), expected_digest=expected_digest)
    if serialize_hypothesis_complete_training_block_v2(block) != text:
        raise HypothesisCompleteV2Error("block JSON is valid but not canonical compact JSON")
    return block


@dataclass(frozen=True, slots=True)
class HypothesisCompleteTrainingManifestV2:
    """Exact cross-block audit for one planned G03-v2 training run."""

    blocks: tuple[HypothesisCompleteTrainingBlockV2, ...]
    registered_episode_budget: int = SCIENTIFIC_TRAINING_EPISODE_BUDGET
    engineering_budget_override: bool = False

    def __post_init__(self) -> None:
        if type(self.blocks) is not tuple or not self.blocks:
            raise HypothesisCompleteV2Error("training manifest requires a nonempty tuple of blocks")
        if any(type(block) is not HypothesisCompleteTrainingBlockV2 for block in self.blocks):
            raise HypothesisCompleteV2Error("training manifest contains a foreign block type")
        _require_integer(self.registered_episode_budget, name="registered episode budget", minimum=1)
        _require_boolean(self.engineering_budget_override, name="engineering budget override")
        override_required = self.registered_episode_budget != SCIENTIFIC_TRAINING_EPISODE_BUDGET
        if self.engineering_budget_override is not override_required:
            raise HypothesisCompleteV2Error(
                "a non-384 engineering budget requires an explicit override, and 384 forbids one"
            )
        n0s = {block.opening.n0 for block in self.blocks}
        if len(n0s) != 1:
            raise HypothesisCompleteV2Error("one manifest must freeze exactly one n0")
        if self.episode_count != self.registered_episode_budget:
            raise HypothesisCompleteV2Error(
                "complete-block episode count differs from the registered episode budget"
            )
        if tuple(block.bank_position for block in self.blocks) != tuple(range(len(self.blocks))):
            raise HypothesisCompleteV2Error("blocks must occupy every bank position in order")
        for field, values in (
            ("block IDs", [block.block_id for block in self.blocks]),
            ("opening IDs", [block.opening.opening_id for block in self.blocks]),
            (
                "opening contents",
                [block.opening.content_digest for block in self.blocks],
            ),
            ("update-batch IDs", [block.atomic_commit.update_batch_id for block in self.blocks]),
        ):
            if len(values) != len(set(values)):
                raise HypothesisCompleteV2Error(f"manifest {field} must be unique")
        for left, right in zip(self.blocks, self.blocks[1:], strict=False):
            if left.atomic_commit.optimizer_step_after != right.atomic_commit.optimizer_step_before:
                raise HypothesisCompleteV2Error("manifest optimizer-step chain is discontinuous")
            if (
                left.atomic_commit.post_update_checkpoint_digest
                != right.atomic_commit.pre_update_checkpoint_digest
            ):
                raise HypothesisCompleteV2Error("manifest checkpoint chain is discontinuous")
        catalog_digests = {block.opening.catalog_digest for block in self.blocks}
        supported_digests = {block.opening.supported_catalog_digest for block in self.blocks}
        if len(catalog_digests) != 1 or len(supported_digests) != 1:
            raise HypothesisCompleteV2Error("all manifest blocks must share both catalog contracts")
        if not all(block.structural_audit_passed for block in self.blocks):
            raise HypothesisCompleteV2Error("a structurally incomplete block entered the manifest")

    @property
    def n0(self) -> int:
        return self.blocks[0].opening.n0

    @property
    def episode_count(self) -> int:
        return len(self.blocks) * self.n0

    @property
    def catalog_digest(self) -> str:
        return self.blocks[0].opening.catalog_digest

    @property
    def supported_catalog_digest(self) -> str:
        return self.blocks[0].opening.supported_catalog_digest

    def _checks(self) -> dict[str, bool]:
        return {
            "one_fixed_n0": len({block.opening.n0 for block in self.blocks}) == 1,
            "registered_episode_budget_exact": self.episode_count == self.registered_episode_budget,
            "complete_hypothesis_rotation_in_every_block": all(
                len(block.rotations) == self.n0
                and {rotation.official_rule_id for rotation in block.rotations}
                == set(block.opening.version_space_rule_ids)
                for block in self.blocks
            ),
            "unique_block_opening_content_and_update_batch_identities": all(
                len(values) == len(set(values))
                for values in (
                    [block.block_id for block in self.blocks],
                    [block.opening.opening_id for block in self.blocks],
                    [block.opening.content_digest for block in self.blocks],
                    [block.atomic_commit.update_batch_id for block in self.blocks],
                )
            ),
            "optimizer_and_checkpoint_chain_exact": all(
                left.atomic_commit.optimizer_step_after == right.atomic_commit.optimizer_step_before
                and left.atomic_commit.post_update_checkpoint_digest
                == right.atomic_commit.pre_update_checkpoint_digest
                for left, right in zip(self.blocks, self.blocks[1:], strict=False)
            ),
            "all_blocks_structurally_pass": all(block.structural_audit_passed for block in self.blocks),
            "feature_histograms_descriptive_only": True,
            "no_powered_384_group_statistics_claimed": True,
            "legacy_three_role_schemas_rejected": True,
        }

    @property
    def hypothesis_completion_structure_passed(self) -> bool:
        return all(self._checks().values())

    def _opening_scene_set_summary(self) -> dict[str, Any]:
        counts = Counter(block.opening.scene_set_digest for block in self.blocks)
        return {
            "scene_set_rows": [
                {"opening_scene_set_digest": digest, "block_count": counts[digest]}
                for digest in sorted(counts)
            ],
            "unique_opening_scene_set_count": len(counts),
            "structural_manifest_requires_unique_opening_scene_sets": False,
            "powered_audit_must_group_by_opening_scene_set_digest": True,
            "powered_audit_must_reject_scene_set_reuse_as_independent_support": True,
        }

    def _unsigned_obj(self) -> dict[str, Any]:
        feature_histograms = _feature_histograms(block.opening for block in self.blocks)
        return {
            "schema_version": HYPOTHESIS_COMPLETE_MANIFEST_SCHEMA_VERSION,
            "report_kind": _MANIFEST_KIND,
            "authorization": dict(_AUTHORIZATION),
            "catalog_digest": self.catalog_digest,
            "supported_catalog_digest": self.supported_catalog_digest,
            "fixed_n0": self.n0,
            "registered_episode_budget": self.registered_episode_budget,
            "engineering_budget_override": self.engineering_budget_override,
            "block_count": len(self.blocks),
            "episode_count": self.episode_count,
            "blocks": [block.as_obj() for block in self.blocks],
            "descriptive_feature_histograms": feature_histograms,
            "opening_scene_set_summary": self._opening_scene_set_summary(),
            "inference_scope": {
                "distinct_opening_block_count": len(self.blocks),
                "episode_row_count": self.episode_count,
                "powered_stress_minimum_independent_blocks": (POWERED_STRESS_MINIMUM_INDEPENDENT_BLOCKS),
                "structural_manifest_establishes_sampling_independence": False,
                "feature_histograms_descriptive_only": True,
                "powered_384_group_statistics_claimed": False,
            },
            "verification_boundaries": {
                "hypothesis_completion_structure_verified": (self.hypothesis_completion_structure_passed),
                "training_bank_balance_verified": False,
                "training_bank_balance_verification_required": True,
                "powered_stress_support_containment_verified": False,
                "powered_stress_support_containment_verification_required": True,
                "sampling_independence_verified": False,
                "sampling_independence_verification_required": True,
                "runtime_execution_verified": False,
                "runtime_execution_verification_required": True,
                "launch_gate_passed": False,
                "launch_gate_verification_required": True,
            },
            "checks": [{"name": name, "passed": passed} for name, passed in self._checks().items()],
            "hypothesis_completion_structure_passed": (self.hypothesis_completion_structure_passed),
            "substitution_contract": {
                "legacy_three_role_block_may_substitute": False,
                "legacy_three_role_training_audit_may_substitute": False,
                "evaluation_quartet_audit_may_substitute": False,
                "powered_stress_audit_may_substitute": False,
            },
        }

    @property
    def digest(self) -> str:
        return _json_digest(self._unsigned_obj(), domain=_MANIFEST_DOMAIN)

    def as_obj(self) -> dict[str, Any]:
        return {**self._unsigned_obj(), "hypothesis_complete_manifest_digest": self.digest}


def build_hypothesis_complete_training_manifest_v2(
    blocks: Iterable[HypothesisCompleteTrainingBlockV2],
    *,
    registered_episode_budget: int = SCIENTIFIC_TRAINING_EPISODE_BUDGET,
    engineering_budget_override: bool = False,
) -> HypothesisCompleteTrainingManifestV2:
    """Build a scientific 384-episode manifest or an explicitly marked test override."""

    return HypothesisCompleteTrainingManifestV2(
        blocks=tuple(blocks),
        registered_episode_budget=registered_episode_budget,
        engineering_budget_override=engineering_budget_override,
    )


def hypothesis_complete_training_manifest_v2_from_obj(
    value: object,
    *,
    expected_digest: str | None = None,
) -> HypothesisCompleteTrainingManifestV2:
    _reject_legacy_three_role_substitute(value)
    obj = _require_mapping(
        value,
        (
            "schema_version",
            "report_kind",
            "authorization",
            "catalog_digest",
            "supported_catalog_digest",
            "fixed_n0",
            "registered_episode_budget",
            "engineering_budget_override",
            "block_count",
            "episode_count",
            "blocks",
            "descriptive_feature_histograms",
            "opening_scene_set_summary",
            "inference_scope",
            "verification_boundaries",
            "checks",
            "hypothesis_completion_structure_passed",
            "substitution_contract",
            "hypothesis_complete_manifest_digest",
        ),
        name="hypothesis-complete training manifest",
    )
    if (
        type(obj["schema_version"]) is not int
        or obj["schema_version"] != HYPOTHESIS_COMPLETE_MANIFEST_SCHEMA_VERSION
        or obj["report_kind"] != _MANIFEST_KIND
    ):
        raise HypothesisCompleteV2Error("hypothesis-complete manifest schema identity mismatch")
    _require_authorization(obj["authorization"])
    raw_blocks = obj["blocks"]
    if type(raw_blocks) is not list:
        raise HypothesisCompleteV2Error("manifest blocks must be an array")
    manifest = HypothesisCompleteTrainingManifestV2(
        blocks=tuple(hypothesis_complete_training_block_v2_from_obj(item) for item in raw_blocks),
        registered_episode_budget=_require_integer(
            obj["registered_episode_budget"], name="registered episode budget", minimum=1
        ),
        engineering_budget_override=_require_boolean(
            obj["engineering_budget_override"], name="engineering budget override"
        ),
    )
    if obj["catalog_digest"] != manifest.catalog_digest or (
        obj["supported_catalog_digest"] != manifest.supported_catalog_digest
    ):
        raise HypothesisCompleteV2Error("manifest catalog bindings are inconsistent")
    if _require_integer(obj["fixed_n0"], name="fixed n0", minimum=1) != manifest.n0:
        raise HypothesisCompleteV2Error("manifest fixed n0 is inconsistent")
    if _require_integer(obj["block_count"], name="block count", minimum=1) != len(manifest.blocks):
        raise HypothesisCompleteV2Error("manifest block count is inconsistent")
    if _require_integer(obj["episode_count"], name="episode count", minimum=1) != manifest.episode_count:
        raise HypothesisCompleteV2Error("manifest episode count is inconsistent")
    if obj["hypothesis_complete_manifest_digest"] != manifest.digest:
        raise HypothesisCompleteV2Error("hypothesis-complete manifest digest mismatch")
    if expected_digest is not None and manifest.digest != _require_sha256(
        expected_digest, name="expected manifest digest"
    ):
        raise HypothesisCompleteV2Error("manifest differs from the externally expected digest")
    if _dump_json(obj) != _dump_json(manifest.as_obj()):
        raise HypothesisCompleteV2Error("hypothesis-complete manifest contains tampered metadata")
    return manifest


def verify_hypothesis_complete_training_manifest_v2(
    manifest: HypothesisCompleteTrainingManifestV2,
) -> HypothesisCompleteTrainingManifestV2:
    if type(manifest) is not HypothesisCompleteTrainingManifestV2:
        raise TypeError("verify requires a HypothesisCompleteTrainingManifestV2")
    return hypothesis_complete_training_manifest_v2_from_obj(
        manifest.as_obj(), expected_digest=manifest.digest
    )


def serialize_hypothesis_complete_training_manifest_v2(
    manifest: HypothesisCompleteTrainingManifestV2,
) -> str:
    return _dump_json(verify_hypothesis_complete_training_manifest_v2(manifest).as_obj())


def parse_hypothesis_complete_training_manifest_v2(
    text: str,
    *,
    expected_digest: str | None = None,
) -> HypothesisCompleteTrainingManifestV2:
    manifest = hypothesis_complete_training_manifest_v2_from_obj(
        _load_json(text), expected_digest=expected_digest
    )
    if serialize_hypothesis_complete_training_manifest_v2(manifest) != text:
        raise HypothesisCompleteV2Error("manifest JSON is valid but not canonical compact JSON")
    return manifest
