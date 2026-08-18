"""Independent, nonauthorizing frozen-manifest rederivation for G03-v2.

The producer schemas in :mod:`goalzendo_interactive_v2.hypothesis_complete`
and :mod:`goalzendo_interactive_v2.statistical_leakage` deliberately leave a
future bank-manifest bridge unresolved.  This module is that additive bridge.
It consumes the producer's canonical manifest bytes plus a small canonical
meta-surface artifact, then rederives their live semantics from the public
catalog.  It never calls either producer parser and never authorizes a bank,
model execution, or a weight update.

All expected hashes are external inputs.  A caller who computes and supplies
them in the same unreviewed operation has established integrity, not
provenance or preregistration.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

import goalzendo_interactive.catalog as catalog_module
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

from .population_audit import build_supported_catalog_contract_v2

FROZEN_MANIFEST_VERIFIER_SCHEMA_VERSION = 1
FROZEN_META_SURFACE_SCHEMA_VERSION = 1

_BLOCK_KIND = "g03-v2-hypothesis-complete-atomic-training-block"
_MANIFEST_KIND = "g03-v2-hypothesis-complete-training-manifest-audit"
_META_KIND = "g03-v2-frozen-hypothesis-complete-meta-surfaces"
_REPORT_KIND = "g03-v2-frozen-manifest-independent-rederivation"
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
_BLOCK_OPENING_DOMAIN = "goalzendo-interactive-v2-hypothesis-complete-block-opening-v2"
_PROMPT_SURFACE_DOMAIN = "goalzendo-interactive-v2-rendered-static-prompt-v2"
_CONSTRUCTION_CLUSTER_DOMAIN = "goalzendo-interactive-v2-construction-cluster-v2"
_META_BLOCK_DOMAIN = "goalzendo-interactive-v2-frozen-meta-surface-block-v1"
_REPORT_DOMAIN = "goalzendo-interactive-v2-frozen-manifest-rederivation-v1"

_ALLOWED_N0 = (8, 12, 16)
_OPENING_COUNT = 10
_TERMINAL_COUNT = 16
_SCIENTIFIC_EPISODE_BUDGET = 384
_POWERED_CLUSTER_MINIMUM = 384

_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
_RULE_ID_PATTERN = re.compile(r"g03r[0-9]{5}")

_PRODUCER_AUTHORIZATION = {
    "scope": "prospective-structural-training-schema-only",
    "production_bank_materialized": False,
    "capability_run_authorized": False,
    "weight_updates_authorized": False,
}
_REPORT_AUTHORIZATION = {
    "scope": "prospective-frozen-manifest-rederivation-only",
    "training_bank_authorized": False,
    "evaluation_bank_authorized": False,
    "model_execution_authorized": False,
    "weight_updates_authorized": False,
    "launch_authorized": False,
}
_META_AUTHORIZATION = {
    "scope": "planned-meta-surface-companion-only",
    "training_bank_authorized": False,
    "model_execution_authorized": False,
    "weight_updates_authorized": False,
}


class FrozenManifestV2Error(ValueError):
    """Raised when frozen bytes fail an independent exact rederivation."""


def _dump_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise FrozenManifestV2Error(f"value is not canonical JSON: {exc}") from exc


def _json_digest(value: Any, *, domain: str) -> str:
    digest = hashlib.sha256()
    digest.update(domain.encode("ascii"))
    digest.update(b"\0")
    digest.update(_dump_json(value).encode("ascii"))
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: object, *, name: str) -> str:
    if not _is_sha256(value):
        raise FrozenManifestV2Error(f"{name} must be a lowercase SHA-256")
    return cast(str, value)


def _require_integer(
    value: object,
    *,
    name: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise FrozenManifestV2Error(f"{name} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise FrozenManifestV2Error(f"{name} must be an integer <= {maximum}")
    return value


def _require_boolean(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise FrozenManifestV2Error(f"{name} must be a Boolean")
    return value


def _require_identifier(value: object, *, name: str) -> str:
    if type(value) is not str or _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise FrozenManifestV2Error(f"{name} is not a canonical identifier")
    return value


def _require_mapping(
    value: object,
    fields: tuple[str, ...],
    *,
    name: str,
) -> Mapping[str, Any]:
    if type(value) is not dict or tuple(value) != fields:
        raise FrozenManifestV2Error(f"{name} has noncanonical or reordered fields")
    return cast(Mapping[str, Any], value)


def _load_canonical_bytes(value: bytes, *, name: str) -> Mapping[str, Any]:
    if type(value) is not bytes or not value:
        raise FrozenManifestV2Error(f"{name} must be nonempty exact bytes")
    try:
        text = value.decode("ascii")
    except UnicodeDecodeError as exc:
        raise FrozenManifestV2Error(f"{name} must be canonical ASCII JSON") from exc

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, child in pairs:
            if key in result:
                raise FrozenManifestV2Error(f"duplicate JSON object key: {key!r}")
            result[key] = child
        return result

    def reject_constant(constant: str) -> None:
        raise FrozenManifestV2Error(f"non-finite JSON constant is forbidden: {constant}")

    try:
        parsed = json.loads(text, object_pairs_hook=no_duplicates, parse_constant=reject_constant)
    except FrozenManifestV2Error:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise FrozenManifestV2Error(f"invalid {name}: {exc}") from exc
    if type(parsed) is not dict:
        raise FrozenManifestV2Error(f"{name} root must be an object")
    if _dump_json(parsed) != text:
        raise FrozenManifestV2Error(f"{name} is not canonical compact JSON bytes")
    return cast(Mapping[str, Any], parsed)


def _module_sha256(path: Path, *, name: str) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError as exc:
        raise FrozenManifestV2Error(f"cannot read {name} source bytes: {exc}") from exc


def _producer_source_digests() -> tuple[str, str, str]:
    package_directory = Path(__file__).resolve().parent
    catalog_path_value = catalog_module.__file__
    if catalog_path_value is None:
        raise FrozenManifestV2Error("public catalog module has no source path")
    return (
        _module_sha256(package_directory / "hypothesis_complete.py", name="hypothesis-complete"),
        _module_sha256(package_directory / "statistical_leakage.py", name="statistical-leakage"),
        _module_sha256(Path(catalog_path_value).resolve(), name="public-catalog"),
    )


@dataclass(frozen=True, slots=True)
class FrozenManifestExpectedBindingsV2:
    """Externally frozen hashes required before semantic verification."""

    manifest_bytes_sha256: str
    meta_surface_bytes_sha256: str
    hypothesis_complete_source_sha256: str
    statistical_leakage_source_sha256: str
    catalog_source_sha256: str
    catalog_digest: str
    supported_catalog_digest: str
    renderer_registry_digest: str

    def __post_init__(self) -> None:
        for name in (
            "manifest_bytes_sha256",
            "meta_surface_bytes_sha256",
            "hypothesis_complete_source_sha256",
            "statistical_leakage_source_sha256",
            "catalog_source_sha256",
            "catalog_digest",
            "supported_catalog_digest",
            "renderer_registry_digest",
        ):
            _require_sha256(getattr(self, name), name=name)

    def as_obj(self) -> dict[str, str]:
        return {
            "manifest_bytes_sha256": self.manifest_bytes_sha256,
            "meta_surface_bytes_sha256": self.meta_surface_bytes_sha256,
            "hypothesis_complete_source_sha256": self.hypothesis_complete_source_sha256,
            "statistical_leakage_source_sha256": self.statistical_leakage_source_sha256,
            "catalog_source_sha256": self.catalog_source_sha256,
            "catalog_digest": self.catalog_digest,
            "supported_catalog_digest": self.supported_catalog_digest,
            "renderer_registry_digest": self.renderer_registry_digest,
        }


def capture_frozen_manifest_expected_bindings_v2(
    manifest_bytes: bytes,
    meta_surface_bytes: bytes,
) -> FrozenManifestExpectedBindingsV2:
    """Capture current hashes for engineering workflows.

    The returned object must be persisted and reviewed outside the later
    verification operation to serve as a genuine frozen expectation.
    """

    if type(manifest_bytes) is not bytes or type(meta_surface_bytes) is not bytes:
        raise TypeError("frozen artifacts must be exact bytes")
    hypothesis_source, statistical_source, catalog_source = _producer_source_digests()
    catalog = build_rule_catalog()
    contract = build_supported_catalog_contract_v2()
    return FrozenManifestExpectedBindingsV2(
        manifest_bytes_sha256=_sha256_bytes(manifest_bytes),
        meta_surface_bytes_sha256=_sha256_bytes(meta_surface_bytes),
        hypothesis_complete_source_sha256=hypothesis_source,
        statistical_leakage_source_sha256=statistical_source,
        catalog_source_sha256=catalog_source,
        catalog_digest=catalog.digest,
        supported_catalog_digest=contract.supported_catalog_digest,
        renderer_registry_digest=renderer_digest(),
    )


@dataclass(frozen=True, slots=True)
class _Opening:
    opening_id: str
    catalog_digest: str
    supported_catalog_digest: str
    observations: tuple[tuple[int, bool], ...]
    entries: tuple[CatalogEntry, ...]
    scene_set_digest: str
    content_digest: str
    digest: str

    @property
    def n0(self) -> int:
        return len(self.entries)


@dataclass(frozen=True, slots=True)
class _TerminalLaw:
    probabilities: tuple[tuple[int, Fraction], ...]
    digest: str
    raw_obj: Mapping[str, Any]
    attestation_digest: str


@dataclass(frozen=True, slots=True)
class _Panel:
    scene_indices: tuple[int, ...]
    digest: str
    receipt_digest: str


@dataclass(frozen=True, slots=True)
class _Block:
    bank_position: int
    block_id: str
    opening: _Opening
    terminal_law: _TerminalLaw
    panel: _Panel
    renderer_name: RendererName
    hidden_display_order: tuple[int, ...]
    hidden_official_order: tuple[str, ...]
    rotations: tuple[Mapping[str, Any], ...]
    static_obj: Mapping[str, Any]
    static_digest: str
    block_digest: str
    pre_checkpoint_digest: str
    post_checkpoint_digest: str
    update_batch_id: str
    optimizer_step_before: int
    optimizer_step_after: int


@dataclass(frozen=True, slots=True)
class _Manifest:
    raw_obj: Mapping[str, Any]
    blocks: tuple[_Block, ...]
    digest: str
    fixed_n0: int
    registered_episode_budget: int
    episode_count: int


def _catalog_entry(rule_id: object) -> CatalogEntry:
    if type(rule_id) is not str or _RULE_ID_PATTERN.fullmatch(rule_id) is None:
        raise FrozenManifestV2Error(f"malformed public rule id: {rule_id!r}")
    index = int(rule_id[4:])
    catalog = build_rule_catalog()
    if not 0 <= index < len(catalog) or catalog[index].rule_id != rule_id:
        raise FrozenManifestV2Error(f"unknown public rule id: {rule_id!r}")
    return catalog[index]


def _fraction_obj(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _parse_fraction(value: object, *, name: str, positive: bool = False) -> Fraction:
    obj = _require_mapping(value, ("numerator", "denominator"), name=name)
    numerator = _require_integer(
        obj["numerator"],
        name=f"{name} numerator",
        minimum=1 if positive else 0,
    )
    denominator = _require_integer(obj["denominator"], name=f"{name} denominator", minimum=1)
    result = Fraction(numerator, denominator)
    if obj != _fraction_obj(result):
        raise FrozenManifestV2Error(f"{name} must be reduced canonical fraction fields")
    return result


def _parse_opening(value: object) -> _Opening:
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
    opening_id = _require_identifier(obj["opening_id"], name="opening_id")
    catalog = build_rule_catalog()
    contract = build_supported_catalog_contract_v2()
    if obj["catalog_digest"] != catalog.digest:
        raise FrozenManifestV2Error("opening catalog digest differs from the live public catalog")
    if obj["supported_catalog_digest"] != contract.supported_catalog_digest:
        raise FrozenManifestV2Error("opening supported digest differs from the live allowlist")
    raw_observations = obj["observations"]
    if type(raw_observations) is not list or len(raw_observations) != _OPENING_COUNT:
        raise FrozenManifestV2Error("opening must contain exactly ten observations")
    observations: list[tuple[int, bool]] = []
    for raw in raw_observations:
        row = _require_mapping(raw, ("scene_index", "accepted"), name="opening observation")
        observations.append(
            (
                _require_integer(
                    row["scene_index"],
                    name="opening scene index",
                    maximum=SCENE_COUNT - 1,
                ),
                _require_boolean(row["accepted"], name="opening accepted label"),
            )
        )
    materialized = tuple(observations)
    if tuple(index for index, _ in materialized) != tuple(sorted({index for index, _ in materialized})):
        raise FrozenManifestV2Error("opening observations must be sorted and scene-unique")
    if sum(accepted for _, accepted in materialized) != _OPENING_COUNT // 2:
        raise FrozenManifestV2Error("opening labels must be exactly five/five")

    exact_space = VersionSpace(catalog, contract.supported_indices).observe_many(materialized)
    entries = tuple(exact_space)
    n0 = _require_integer(obj["n0"], name="n0", minimum=1)
    if n0 not in _ALLOWED_N0 or len(entries) != n0:
        raise FrozenManifestV2Error("opening does not rederive an allowed exact supported V0")
    raw_rules = obj["version_space_rules"]
    if type(raw_rules) is not list or len(raw_rules) != n0:
        raise FrozenManifestV2Error("opening V0 rows do not match n0")
    expected_rules = [{"rule_id": entry.rule_id, "truth_digest": entry.truth_digest} for entry in entries]
    if raw_rules != expected_rules:
        raise FrozenManifestV2Error("opening V0 identities differ from live catalog recomputation")

    content_obj = {
        "catalog_digest": catalog.digest,
        "supported_catalog_digest": contract.supported_catalog_digest,
        "observations": [
            {"scene_index": scene_index, "accepted": accepted} for scene_index, accepted in materialized
        ],
        "n0": n0,
        "version_space_rules": expected_rules,
    }
    scene_set_digest = _json_digest(
        {
            "scene_count": SCENE_COUNT,
            "scene_indices": [scene_index for scene_index, _ in materialized],
        },
        domain=_OPENING_SCENE_SET_DOMAIN,
    )
    content_digest = _json_digest(content_obj, domain=_OPENING_CONTENT_DOMAIN)
    unsigned = {
        "opening_id": opening_id,
        **content_obj,
        "opening_scene_set_digest": scene_set_digest,
        "opening_content_digest": content_digest,
    }
    digest = _json_digest(unsigned, domain=_OPENING_DOMAIN)
    if obj["opening_scene_set_digest"] != scene_set_digest:
        raise FrozenManifestV2Error("opening scene-set digest differs from live rederivation")
    if obj["opening_content_digest"] != content_digest:
        raise FrozenManifestV2Error("opening content digest differs from live rederivation")
    if obj["opening_digest"] != digest:
        raise FrozenManifestV2Error("opening digest differs from live rederivation")
    return _Opening(
        opening_id,
        catalog.digest,
        contract.supported_catalog_digest,
        materialized,
        entries,
        scene_set_digest,
        content_digest,
        digest,
    )


def _parse_terminal_law(value: object) -> _TerminalLaw:
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
    catalog = build_rule_catalog()
    contract = build_supported_catalog_contract_v2()
    if (
        type(obj["schema_version"]) is not int
        or obj["schema_version"] != 2
        or obj["law_kind"] != _TERMINAL_LAW_KIND
    ):
        raise FrozenManifestV2Error("terminal-law schema identity mismatch")
    if obj["catalog_digest"] != catalog.digest or (
        obj["supported_catalog_digest"] != contract.supported_catalog_digest
    ):
        raise FrozenManifestV2Error("terminal law differs from the live catalog contracts")
    attestation = _require_sha256(
        obj["public_derivation_attestation_digest"],
        name="public derivation attestation digest",
    )
    if _require_boolean(
        obj["public_derivation_attestation_externally_verified"],
        name="public derivation external verification",
    ):
        raise FrozenManifestV2Error("public derivation must remain externally unverified")
    if not _require_boolean(
        obj["external_generator_verification_required"],
        name="terminal-law generator verification requirement",
    ):
        raise FrozenManifestV2Error("terminal-law external generator verification must be required")
    if _require_integer(obj["terminal_item_count"], name="terminal item count") != _TERMINAL_COUNT:
        raise FrozenManifestV2Error("terminal item count is not the registered sixteen")
    raw_probabilities = obj["scene_probabilities"]
    if type(raw_probabilities) is not list or len(raw_probabilities) < _TERMINAL_COUNT:
        raise FrozenManifestV2Error("terminal law must support at least sixteen scenes")
    probabilities: list[tuple[int, Fraction]] = []
    for raw in raw_probabilities:
        row = _require_mapping(raw, ("scene_index", "probability"), name="terminal scene mass")
        probabilities.append(
            (
                _require_integer(
                    row["scene_index"],
                    name="terminal scene index",
                    maximum=SCENE_COUNT - 1,
                ),
                _parse_fraction(row["probability"], name="terminal probability", positive=True),
            )
        )
    materialized = tuple(probabilities)
    if tuple(index for index, _ in materialized) != tuple(sorted({index for index, _ in materialized})):
        raise FrozenManifestV2Error("terminal-law support must be sorted and unique")
    if sum((probability for _, probability in materialized), Fraction()) != 1:
        raise FrozenManifestV2Error("terminal-law exact mass does not sum to one")
    if _require_integer(obj["scene_support_count"], name="scene support count", minimum=1) != len(
        materialized
    ):
        raise FrozenManifestV2Error("terminal-law support count is inconsistent")
    unsigned = {key: obj[key] for key in tuple(obj)[:-1]}
    digest = _json_digest(unsigned, domain=_TERMINAL_LAW_DOMAIN)
    if obj["unconditional_scene_law_digest"] != digest:
        raise FrozenManifestV2Error("terminal-law digest differs from canonical bytes")
    return _TerminalLaw(materialized, digest, obj, attestation)


def _truth_pattern(entries: tuple[CatalogEntry, ...], scene_index: int) -> int:
    return sum(1 << position for position, entry in enumerate(entries) if entry.truth[scene_index])


def _parse_panel(value: object, opening: _Opening) -> _Panel:
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
        or obj["schema_version"] != 1
        or obj["panel_kind"] != "evaluator-private-shared-sixteen-scene-training-panel"
    ):
        raise FrozenManifestV2Error("training-panel schema identity mismatch")
    if (
        obj["catalog_digest"] != opening.catalog_digest
        or obj["supported_catalog_digest"] != opening.supported_catalog_digest
        or obj["opening_content_digest"] != opening.content_digest
    ):
        raise FrozenManifestV2Error("panel catalog/opening identity differs from its block")
    expected_rules = [
        {"rule_id": entry.rule_id, "truth_digest": entry.truth_digest} for entry in opening.entries
    ]
    if obj["version_space_rules"] != expected_rules:
        raise FrozenManifestV2Error("panel V0 identities differ from the opening V0")
    raw_scenes = obj["scene_indices"]
    if type(raw_scenes) is not list or len(raw_scenes) != _TERMINAL_COUNT:
        raise FrozenManifestV2Error("training panel must contain exactly sixteen scenes")
    scenes = tuple(
        _require_integer(item, name="panel scene index", maximum=SCENE_COUNT - 1) for item in raw_scenes
    )
    if scenes != tuple(sorted(set(scenes))):
        raise FrozenManifestV2Error("training-panel scenes must be sorted and unique")
    patterns = tuple(_truth_pattern(opening.entries, scene_index) for scene_index in scenes)
    if any(pattern.bit_count() != opening.n0 // 2 for pattern in patterns):
        raise FrozenManifestV2Error("every training-panel scene must be half/half across V0")
    counts = Counter(patterns)
    full_mask = (1 << opening.n0) - 1
    if any(counts[pattern] != counts[full_mask ^ pattern] for pattern in counts):
        raise FrozenManifestV2Error("training-panel truth patterns lack exact complements")
    truth_rows = [
        {
            "scene_index": scene_index,
            "truth_pattern_mask": pattern,
            "accepted_candidate_count": opening.n0 // 2,
            "rejected_candidate_count": opening.n0 // 2,
        }
        for scene_index, pattern in zip(scenes, patterns, strict=True)
    ]
    rule_rows = [
        {
            "rule_id": entry.rule_id,
            "truth_digest": entry.truth_digest,
            "accepted_count": sum(entry.truth[index] for index in scenes),
            "rejected_count": sum(not entry.truth[index] for index in scenes),
        }
        for entry in opening.entries
    ]
    if any(row["accepted_count"] != _TERMINAL_COUNT // 2 for row in rule_rows):
        raise FrozenManifestV2Error("each V0 rule must label the panel exactly eight/eight")
    if obj["truth_pattern_rows"] != truth_rows or obj["rule_balance_rows"] != rule_rows:
        raise FrozenManifestV2Error("training-panel derived rows differ from live truth vectors")
    receipt = _require_sha256(
        obj["external_generator_receipt_digest"],
        name="external panel-generator receipt digest",
    )
    expected_booleans = (
        ("external_generator_receipt_verified", False),
        ("external_generator_verification_required", True),
        ("panel_binding_visible_to_model", False),
        ("pairwise_rule_separation_required_here", False),
        ("evaluation_challenge_bank_owns_pairwise_separation", True),
    )
    for field, expected in expected_booleans:
        if _require_boolean(obj[field], name=field) is not expected:
            raise FrozenManifestV2Error(f"training-panel {field} has an unauthorized value")
    unsigned = {key: obj[key] for key in tuple(obj)[:-1]}
    digest = _json_digest(unsigned, domain=_PANEL_DOMAIN)
    if obj["materialized_training_panel_digest"] != digest:
        raise FrozenManifestV2Error("training-panel digest differs from canonical bytes")
    return _Panel(scenes, digest, receipt)


def _terminal_balance_audit(
    opening: _Opening,
    law: _TerminalLaw,
    panel: _Panel,
) -> dict[str, Any]:
    probability_by_scene = dict(law.probabilities)
    law_rows: list[dict[str, Any]] = []
    law_baseline = Fraction()
    for scene_index in sorted(probability_by_scene):
        pattern = _truth_pattern(opening.entries, scene_index)
        accepted_count = pattern.bit_count()
        scene_bayes = Fraction(max(accepted_count, opening.n0 - accepted_count), opening.n0)
        probability = probability_by_scene[scene_index]
        law_baseline += probability * scene_bayes
        law_rows.append(
            {
                "scene_index": scene_index,
                "probability": _fraction_obj(probability),
                "truth_pattern_mask": pattern,
                "accepted_candidate_count": accepted_count,
                "rejected_candidate_count": opening.n0 - accepted_count,
                "no_query_bayes_accuracy": _fraction_obj(scene_bayes),
            }
        )
    rule_rows: list[dict[str, Any]] = []
    for entry in opening.entries:
        accepted_mass = sum(
            (probability for scene_index, probability in law.probabilities if entry.truth[scene_index]),
            Fraction(),
        )
        rule_rows.append(
            {
                "rule_id": entry.rule_id,
                "truth_digest": entry.truth_digest,
                "accepted_mass": _fraction_obj(accepted_mass),
                "rejected_mass": _fraction_obj(1 - accepted_mass),
            }
        )
    support = tuple(scene_index for scene_index, _ in law.probabilities)
    separation_rows: list[dict[str, Any]] = []
    for left_position, left in enumerate(opening.entries):
        for right in opening.entries[left_position + 1 :]:
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
    pattern_masses: dict[int, Fraction] = {}
    for scene_index, probability in law.probabilities:
        pattern = _truth_pattern(opening.entries, scene_index)
        pattern_masses[pattern] = pattern_masses.get(pattern, Fraction()) + probability
    full_mask = (1 << opening.n0) - 1
    complement_rows = [
        {
            "truth_pattern_mask": pattern,
            "complement_truth_pattern_mask": full_mask ^ pattern,
            "pattern_mass": _fraction_obj(pattern_masses[pattern]),
            "complement_pattern_mass": _fraction_obj(pattern_masses.get(full_mask ^ pattern, Fraction())),
            "exactly_matched": pattern_masses[pattern] == pattern_masses.get(full_mask ^ pattern, Fraction()),
        }
        for pattern in sorted(pattern_masses)
    ]
    panel_baseline = sum(
        (
            Fraction(
                max(
                    _truth_pattern(opening.entries, scene_index).bit_count(),
                    opening.n0 - _truth_pattern(opening.entries, scene_index).bit_count(),
                ),
                opening.n0 * _TERMINAL_COUNT,
            )
            for scene_index in panel.scene_indices
        ),
        Fraction(),
    )
    support_half = all(row["accepted_candidate_count"] == opening.n0 // 2 for row in law_rows)
    rules_half = all(row["accepted_mass"] == _fraction_obj(Fraction(1, 2)) for row in rule_rows)
    complements = all(row["exactly_matched"] is True for row in complement_rows)
    return {
        "prior": {
            "kind": "uniform-over-exact-supported-V0",
            "per_rule_probability": _fraction_obj(Fraction(1, opening.n0)),
        },
        "required_no_query_bayes_accuracy": _fraction_obj(Fraction(1, 2)),
        "law_no_query_bayes_accuracy": _fraction_obj(law_baseline),
        "materialized_panel_no_query_bayes_accuracy": _fraction_obj(panel_baseline),
        "law_support_rows": law_rows,
        "exact_rule_acceptance_mass_rows": rule_rows,
        "law_pairwise_separation_rows": separation_rows,
        "law_complementary_pattern_mass_rows": complement_rows,
        "law_support_every_scene_half_half": support_half,
        "law_every_rule_exact_half_mass": rules_half,
        "law_complementary_pattern_pair_sampling_exact": complements,
        "law_support_pairwise_separation_observed_descriptively": all(
            row["separated"] is True for row in separation_rows
        ),
        "law_support_pairwise_separation_required_here": False,
        "evaluation_challenge_and_query_banks_own_pairwise_separation": True,
        "law_and_panel_no_query_baselines_exact_half": (
            law_baseline == Fraction(1, 2) and panel_baseline == Fraction(1, 2)
        ),
    }


def _candidate_features(entry: CatalogEntry) -> dict[str, str]:
    rule = entry.rule
    operator: str
    atom_family: str
    if type(rule) is RuleLiteral:
        family = "placard_literal" if rule.atom.op == "placard_is" else "one_literal_piece"
        literal_count = "1"
        operator = "literal"
        atom_family = rule.atom.op
    elif type(rule) is BinaryRule:
        if any(item.atom.op == "placard_is" for item in rule.args):
            raise FrozenManifestV2Error("unsupported placard composition entered V0")
        family = "composed_two_literal_piece"
        literal_count = "2"
        operator = rule.op
        atom_family = "+".join(sorted(item.atom.op for item in rule.args))
    else:  # pragma: no cover - closed public grammar
        raise FrozenManifestV2Error("unknown public rule AST entered V0")
    scaled = entry.truth.true_count * 8
    if scaled < SCENE_COUNT * 3:
        prevalence = "p25_to_lt_p37_5"
    elif scaled < SCENE_COUNT * 4:
        prevalence = "p37_5_to_lt_p50"
    elif scaled < SCENE_COUNT * 5:
        prevalence = "p50_to_lt_p62_5"
    else:
        prevalence = "p62_5_to_p75"
    return {
        "family": family,
        "literal_count": literal_count,
        "operator": operator,
        "placard_inclusion": "yes" if family == "placard_literal" else "no",
        "atom_family": atom_family,
        "prevalence_bin": prevalence,
    }


def _feature_histograms(openings: Iterable[_Opening]) -> list[dict[str, Any]]:
    order = (
        "family",
        "literal_count",
        "operator",
        "placard_inclusion",
        "atom_family",
        "prevalence_bin",
    )
    counts: Counter[tuple[str, str]] = Counter()
    for opening in openings:
        for entry in opening.entries:
            features = _candidate_features(entry)
            for feature in order:
                counts[(feature, features[feature])] += 1
    return [
        {
            "feature": feature,
            "value": value,
            "candidate_count": counts[(feature, value)],
            "official_count": counts[(feature, value)],
        }
        for feature in order
        for value in sorted(value for observed, value in counts if observed == feature)
    ]


def _parse_hidden_binding(
    value: object,
    opening: _Opening,
) -> tuple[tuple[str, ...], tuple[int, ...], str]:
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
        raise FrozenManifestV2Error("hidden order/display binding kind is wrong")
    if obj["opening_digest"] != opening.digest:
        raise FrozenManifestV2Error("hidden binding refers to another opening")
    if _require_boolean(obj["binding_metadata_visible_to_model"], name="hidden metadata visibility"):
        raise FrozenManifestV2Error("hidden metadata must not be model-visible")
    if _require_boolean(obj["independence_externally_verified"], name="hidden independence status"):
        raise FrozenManifestV2Error("hidden-order independence must remain externally unverified")
    raw_rules = obj["official_rotation_order_rule_ids"]
    raw_scenes = obj["opening_display_scene_indices"]
    if type(raw_rules) is not list or any(type(item) is not str for item in raw_rules):
        raise FrozenManifestV2Error("hidden Official order must be a string array")
    if type(raw_scenes) is not list:
        raise FrozenManifestV2Error("hidden opening display order must be an integer array")
    rules = tuple(cast(list[str], raw_rules))
    scenes = tuple(
        _require_integer(item, name="hidden display scene", maximum=SCENE_COUNT - 1) for item in raw_scenes
    )
    if len(rules) != opening.n0 or set(rules) != {entry.rule_id for entry in opening.entries}:
        raise FrozenManifestV2Error("hidden Official order is not one exact V0 permutation")
    opening_scene_set = {scene_index for scene_index, _ in opening.observations}
    if len(scenes) != _OPENING_COUNT or set(scenes) != opening_scene_set:
        raise FrozenManifestV2Error("hidden display order is not one exact opening permutation")
    precommitment = _require_sha256(
        obj["independence_precommitment_digest"],
        name="hidden independence precommitment",
    )

    def rank(kind: str, item: str | int) -> str:
        return _json_digest(
            {
                "opening_content_digest": opening.content_digest,
                "independence_precommitment_digest": precommitment,
                "sequence_kind": kind,
                "value": item,
            },
            domain=_HIDDEN_RANK_DOMAIN,
        )

    expected_rules = tuple(
        sorted(
            (entry.rule_id for entry in opening.entries),
            key=lambda rule_id: (rank("official-rotation", rule_id), rule_id),
        )
    )
    expected_scenes = tuple(
        sorted(
            opening_scene_set,
            key=lambda scene_index: (rank("opening-display", scene_index), scene_index),
        )
    )
    if rules != expected_rules or scenes != expected_scenes:
        raise FrozenManifestV2Error("hidden orders differ from live canonical hash ranking")
    unsigned = {key: obj[key] for key in tuple(obj)[:-1]}
    digest = _json_digest(unsigned, domain=_HIDDEN_BINDING_DOMAIN)
    if obj["hidden_order_display_binding_digest"] != digest:
        raise FrozenManifestV2Error("hidden binding digest differs from canonical bytes")
    return rules, scenes, digest


def _parse_commit(value: object, *, n0: int) -> tuple[str, str, str, int, int, str]:
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
        name="atomic commit",
    )
    batch = _require_identifier(obj["update_batch_id"], name="update_batch_id")
    pre = _require_sha256(obj["pre_update_checkpoint_digest"], name="pre-update checkpoint")
    post = _require_sha256(obj["post_update_checkpoint_digest"], name="post-update checkpoint")
    before = _require_integer(obj["optimizer_step_before"], name="optimizer step before")
    after = _require_integer(obj["optimizer_step_after"], name="optimizer step after", minimum=1)
    if after != before + 1:
        raise FrozenManifestV2Error("atomic commit must advance exactly one optimizer step")
    if _require_integer(obj["collected_objective_count"], name="objective count", minimum=1) != n0:
        raise FrozenManifestV2Error("atomic commit did not collect all V0 objectives")
    disposition = obj["commit_disposition"]
    expected_disposition = "committed_state_changed" if pre != post else "zero_gradient_no_state_change"
    if disposition != expected_disposition:
        raise FrozenManifestV2Error("atomic commit disposition disagrees with state transition")
    if _require_integer(obj["parameter_updates_before_commit"], name="early updates") != 0:
        raise FrozenManifestV2Error("an update occurred before atomic commit")
    if _require_integer(obj["atomic_commit_count"], name="atomic commit count", minimum=1) != 1:
        raise FrozenManifestV2Error("block must contain exactly one atomic commit")
    if not _require_boolean(
        obj["all_objectives_collected_before_commit"], name="all objectives before commit"
    ):
        raise FrozenManifestV2Error("not all objectives were collected before commit")
    unsigned = {key: obj[key] for key in tuple(obj)[:-1]}
    digest = _json_digest(unsigned, domain=_COMMIT_DOMAIN)
    if obj["atomic_commit_digest"] != digest:
        raise FrozenManifestV2Error("atomic commit digest differs from canonical bytes")
    return batch, pre, post, before, after, digest


def _parse_rotation(
    value: object,
    *,
    position: int,
    official_rule_id: str,
    n0: int,
    static_digest: str,
    batch: str,
    pre: str,
    step_before: int,
) -> Mapping[str, Any]:
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
        name="rotation execution",
    )
    if _require_integer(obj["rotation_position"], name="rotation position") != position:
        raise FrozenManifestV2Error("rotation rows are reordered or incomplete")
    if obj["official_rule_id"] != official_rule_id:
        raise FrozenManifestV2Error("rotation Official order differs from hidden binding")
    entry = _catalog_entry(official_rule_id)
    if obj["official_truth_digest"] != entry.truth_digest:
        raise FrozenManifestV2Error("rotation Official truth digest differs from catalog")
    weight = _require_mapping(
        obj["exact_objective_weight"],
        ("numerator", "denominator"),
        name="exact objective weight",
    )
    if (
        _require_integer(weight["numerator"], name="weight numerator", minimum=1) != 1
        or _require_integer(weight["denominator"], name="weight denominator", minimum=1) != n0
    ):
        raise FrozenManifestV2Error("rotation weight must be exactly 1/n0")
    if obj["static_model_input_digest"] != static_digest:
        raise FrozenManifestV2Error("rotation static prompt differs within block")
    if obj["pre_update_checkpoint_digest"] != pre or obj["update_batch_id"] != batch:
        raise FrozenManifestV2Error("rotation checkpoint or batch differs within atomic block")
    if (
        _require_integer(obj["optimizer_step_before"], name="rotation optimizer step") != step_before
        or _require_integer(
            obj["optimizer_step_after_objective_collection"],
            name="post-collection optimizer step",
        )
        != step_before
    ):
        raise FrozenManifestV2Error("optimizer changed during objective collection")
    expected_flags = (
        ("context_reset_before_episode", True),
        ("cache_reset_before_episode", True),
        ("objective_collected_before_block_commit", True),
        ("parameter_update_committed_before_block_commit", False),
    )
    for field, expected in expected_flags:
        if _require_boolean(obj[field], name=field) is not expected:
            raise FrozenManifestV2Error(f"rotation flag {field} violates atomic execution")
    unsigned = {key: obj[key] for key in tuple(obj)[:-1]}
    digest = _json_digest(unsigned, domain=_ROTATION_DOMAIN)
    if obj["rotation_execution_digest"] != digest:
        raise FrozenManifestV2Error("rotation digest differs from canonical bytes")
    return obj


def _string_leaves(value: object) -> set[str]:
    if type(value) is dict:
        return set().union(*(_string_leaves(child) for child in cast(dict[str, Any], value).values()))
    if type(value) is list:
        return set().union(*(_string_leaves(child) for child in value))
    return {value} if type(value) is str else set()


def _parse_block(value: object, *, expected_position: int) -> _Block:
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
        name="hypothesis-complete block",
    )
    if (
        type(obj["schema_version"]) is not int
        or obj["schema_version"] != 2
        or obj["report_kind"] != _BLOCK_KIND
    ):
        raise FrozenManifestV2Error("hypothesis-complete block schema identity mismatch")
    authorization = _require_mapping(
        obj["authorization"], tuple(_PRODUCER_AUTHORIZATION), name="block authorization"
    )
    if dict(authorization) != _PRODUCER_AUTHORIZATION:
        raise FrozenManifestV2Error("block authorization must remain false")
    position = _require_integer(obj["bank_position"], name="bank position")
    if position != expected_position:
        raise FrozenManifestV2Error("block order or bank position is noncanonical")
    block_id = _require_identifier(obj["block_id"], name="block_id")
    opening = _parse_opening(obj["opening"])
    if obj["catalog_digest"] != opening.catalog_digest or (
        obj["supported_catalog_digest"] != opening.supported_catalog_digest
    ):
        raise FrozenManifestV2Error("block catalog bindings differ from opening")
    if _require_integer(obj["n0"], name="block n0", minimum=1) != opening.n0:
        raise FrozenManifestV2Error("block n0 differs from exact opening V0")
    law = _parse_terminal_law(obj["unconditional_terminal_scene_law"])
    panel = _parse_panel(obj["materialized_training_panel"], opening)
    opening_scenes = {scene_index for scene_index, _ in opening.observations}
    law_scenes = {scene_index for scene_index, _ in law.probabilities}
    panel_scenes = set(panel.scene_indices)
    if opening_scenes & law_scenes or opening_scenes & panel_scenes:
        raise FrozenManifestV2Error("opening and terminal stages reuse a scene")
    if not panel_scenes <= law_scenes:
        raise FrozenManifestV2Error("materialized panel is not contained in public law support")
    terminal_audit = _terminal_balance_audit(opening, law, panel)
    if obj["exact_terminal_balance_audit"] != terminal_audit:
        raise FrozenManifestV2Error("terminal audit differs from live truth-vector rederivation")
    required_terminal_checks = (
        "law_support_every_scene_half_half",
        "law_every_rule_exact_half_mass",
        "law_complementary_pattern_pair_sampling_exact",
        "law_and_panel_no_query_baselines_exact_half",
    )
    if any(terminal_audit[name] is not True for name in required_terminal_checks):
        raise FrozenManifestV2Error("terminal law or panel fails its exact no-query-neutral gate")

    renderer_binding = _require_mapping(
        obj["renderer_binding"],
        ("renderer_name", "renderer_registry_digest"),
        name="renderer binding",
    )
    renderer_value = renderer_binding["renderer_name"]
    if type(renderer_value) is not str or renderer_value not in TRAIN_RENDERERS:
        raise FrozenManifestV2Error("block renderer is not a registered training renderer")
    renderer = renderer_value
    if renderer_binding["renderer_registry_digest"] != renderer_digest():
        raise FrozenManifestV2Error("block renderer registry digest differs from live registry")
    official_order, display_order, hidden_digest = _parse_hidden_binding(
        obj["hidden_order_display_binding"], opening
    )
    labels = dict(opening.observations)
    static_obj = {
        "opening_examples": [
            {
                "accepted": labels[scene_index],
                "scene_text": render_scene(scene_at(scene_index), renderer),
            }
            for scene_index in display_order
        ],
        "renderer_contract": {
            "renderer_name": renderer,
            "renderer_registry_digest": renderer_digest(),
        },
        "public_train_terminal_scene_law": law.raw_obj,
    }
    if obj["model_visible_static_data"] != static_obj:
        raise FrozenManifestV2Error("model-visible static prompt differs from live rendering")
    static_digest = _json_digest(static_obj, domain=_STATIC_INPUT_DOMAIN)
    if obj["model_visible_static_data_digest"] != static_digest:
        raise FrozenManifestV2Error("model-visible static digest differs from live rederivation")
    raw_rotations = obj["rotations"]
    if type(raw_rotations) is not list or len(raw_rotations) != opening.n0:
        raise FrozenManifestV2Error("block does not contain exactly one rotation per V0 rule")
    batch, pre, post, step_before, step_after, _commit_digest = _parse_commit(
        obj["atomic_commit"], n0=opening.n0
    )
    rotations = tuple(
        _parse_rotation(
            raw,
            position=rotation_position,
            official_rule_id=official_order[rotation_position],
            n0=opening.n0,
            static_digest=static_digest,
            batch=batch,
            pre=pre,
            step_before=step_before,
        )
        for rotation_position, raw in enumerate(raw_rotations)
    )
    expected_counts = [{"rule_id": entry.rule_id, "official_count": 1} for entry in opening.entries]
    if obj["candidate_official_counts"] != expected_counts:
        raise FrozenManifestV2Error("candidate Official counts differ from complete rotation")
    if obj["descriptive_feature_histograms"] != _feature_histograms((opening,)):
        raise FrozenManifestV2Error("block feature histograms differ from live catalog ASTs")
    sensitive = {
        block_id,
        opening.opening_id,
        opening.digest,
        opening.content_digest,
        opening.scene_set_digest,
        *(entry.rule_id for entry in opening.entries),
        *(entry.truth_digest for entry in opening.entries),
        panel.digest,
        panel.receipt_digest,
        hidden_digest,
        batch,
        pre,
        post,
        *(cast(str, row["rotation_execution_digest"]) for row in rotations),
    }
    private_absent = sensitive.isdisjoint(_string_leaves(static_obj))
    checks = {
        "full_catalog_and_supported_allowlist_bound": True,
        "opening_v0_exactly_recomputed": True,
        "every_v0_identity_official_exactly_once": True,
        "same_role_neutral_static_model_input": True,
        "single_unconditional_rule_oblivious_terminal_law": True,
        "public_law_support_at_least_sixteen": len(law.probabilities) >= _TERMINAL_COUNT,
        "public_law_every_support_scene_half_half": cast(
            bool, terminal_audit["law_support_every_scene_half_half"]
        ),
        "public_law_every_rule_exact_half_mass": cast(bool, terminal_audit["law_every_rule_exact_half_mass"]),
        "public_law_complementary_pattern_pair_sampling_exact": cast(
            bool, terminal_audit["law_complementary_pattern_pair_sampling_exact"]
        ),
        "pairwise_identification_delegated_to_challenge_and_query_banks": True,
        "materialized_panel_shared_balanced_and_private": True,
        "law_and_panel_no_query_bayes_baselines_exact_half": cast(
            bool, terminal_audit["law_and_panel_no_query_baselines_exact_half"]
        ),
        "equal_exact_one_over_n0_weights": True,
        "same_pre_update_checkpoint_optimizer_step_and_batch": True,
        "all_contexts_and_caches_reset": True,
        "no_update_and_all_objectives_before_commit": True,
        "exactly_one_optimizer_step_after_all_rotations": True,
        "exact_model_visible_static_whitelist_enforced": True,
        "all_private_bindings_absent_from_model_input": private_absent,
    }
    expected_checks = [{"name": name, "passed": passed} for name, passed in checks.items()]
    if obj["checks"] != expected_checks or any(not passed for passed in checks.values()):
        raise FrozenManifestV2Error("block check rows differ from independent rederivation")
    if _require_boolean(obj["structural_audit_passed"], name="block structural pass") is not True:
        raise FrozenManifestV2Error("block structural pass cannot be trusted and rederived false")
    if obj["substitution_contract"] != {
        "legacy_three_role_block_may_substitute": False,
        "legacy_three_role_training_audit_may_substitute": False,
        "evaluation_quartet_audit_may_substitute": False,
        "powered_stress_audit_may_substitute": False,
    }:
        raise FrozenManifestV2Error("block substitution contract is not fail-closed")
    unsigned = {key: obj[key] for key in tuple(obj)[:-1]}
    block_digest = _json_digest(unsigned, domain=_BLOCK_DOMAIN)
    if obj["hypothesis_complete_block_digest"] != block_digest:
        raise FrozenManifestV2Error("block digest differs from canonical bytes")
    return _Block(
        position,
        block_id,
        opening,
        law,
        panel,
        renderer,
        display_order,
        official_order,
        rotations,
        cast(Mapping[str, Any], static_obj),
        static_digest,
        block_digest,
        pre,
        post,
        batch,
        step_before,
        step_after,
    )


def _parse_manifest(value: Mapping[str, Any]) -> _Manifest:
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
        name="hypothesis-complete manifest",
    )
    if (
        type(obj["schema_version"]) is not int
        or obj["schema_version"] != 2
        or obj["report_kind"] != _MANIFEST_KIND
    ):
        raise FrozenManifestV2Error("hypothesis-complete manifest schema identity mismatch")
    authorization = _require_mapping(
        obj["authorization"], tuple(_PRODUCER_AUTHORIZATION), name="manifest authorization"
    )
    if dict(authorization) != _PRODUCER_AUTHORIZATION:
        raise FrozenManifestV2Error("manifest authorization must remain false")
    raw_blocks = obj["blocks"]
    if type(raw_blocks) is not list or not raw_blocks:
        raise FrozenManifestV2Error("manifest must contain a nonempty block array")
    blocks = tuple(_parse_block(raw, expected_position=position) for position, raw in enumerate(raw_blocks))
    n0_values = {block.opening.n0 for block in blocks}
    if len(n0_values) != 1:
        raise FrozenManifestV2Error("manifest must freeze exactly one n0")
    n0 = next(iter(n0_values))
    if _require_integer(obj["fixed_n0"], name="fixed n0", minimum=1) != n0:
        raise FrozenManifestV2Error("manifest fixed n0 differs from live openings")
    block_count = _require_integer(obj["block_count"], name="block count", minimum=1)
    if block_count != len(blocks):
        raise FrozenManifestV2Error("manifest block count is inconsistent")
    episode_count = len(blocks) * n0
    if _require_integer(obj["episode_count"], name="episode count", minimum=1) != episode_count:
        raise FrozenManifestV2Error("manifest episode count is inconsistent")
    budget = _require_integer(obj["registered_episode_budget"], name="episode budget", minimum=1)
    if budget != episode_count:
        raise FrozenManifestV2Error("manifest episode budget differs from complete rotations")
    override = _require_boolean(obj["engineering_budget_override"], name="engineering override")
    if override is not (budget != _SCIENTIFIC_EPISODE_BUDGET):
        raise FrozenManifestV2Error("engineering budget override is inconsistent")
    catalog = build_rule_catalog()
    contract = build_supported_catalog_contract_v2()
    if obj["catalog_digest"] != catalog.digest or (
        obj["supported_catalog_digest"] != contract.supported_catalog_digest
    ):
        raise FrozenManifestV2Error("manifest catalog contracts differ from live derivation")
    identity_groups = (
        ("block IDs", tuple(block.block_id for block in blocks)),
        ("opening IDs", tuple(block.opening.opening_id for block in blocks)),
        ("opening contents", tuple(block.opening.content_digest for block in blocks)),
        ("update batches", tuple(block.update_batch_id for block in blocks)),
    )
    for name, values in identity_groups:
        if len(values) != len(set(values)):
            raise FrozenManifestV2Error(f"manifest {name} are not unique")
    for left, right in pairwise(blocks):
        if left.optimizer_step_after != right.optimizer_step_before:
            raise FrozenManifestV2Error("manifest optimizer-step chain is discontinuous")
        if left.post_checkpoint_digest != right.pre_checkpoint_digest:
            raise FrozenManifestV2Error("manifest checkpoint chain is discontinuous")
    if obj["descriptive_feature_histograms"] != _feature_histograms(block.opening for block in blocks):
        raise FrozenManifestV2Error("manifest feature histograms differ from live catalog ASTs")
    scene_counts = Counter(block.opening.scene_set_digest for block in blocks)
    scene_summary = {
        "scene_set_rows": [
            {"opening_scene_set_digest": digest, "block_count": scene_counts[digest]}
            for digest in sorted(scene_counts)
        ],
        "unique_opening_scene_set_count": len(scene_counts),
        "structural_manifest_requires_unique_opening_scene_sets": False,
        "powered_audit_must_group_by_opening_scene_set_digest": True,
        "powered_audit_must_reject_scene_set_reuse_as_independent_support": True,
    }
    if obj["opening_scene_set_summary"] != scene_summary:
        raise FrozenManifestV2Error("opening scene-set summary differs from live content")
    inference_scope = {
        "distinct_opening_block_count": len(blocks),
        "episode_row_count": episode_count,
        "powered_stress_minimum_independent_blocks": _POWERED_CLUSTER_MINIMUM,
        "structural_manifest_establishes_sampling_independence": False,
        "feature_histograms_descriptive_only": True,
        "powered_384_group_statistics_claimed": False,
    }
    if obj["inference_scope"] != inference_scope:
        raise FrozenManifestV2Error("manifest inference scope is not the registered nonclaim")
    boundaries = {
        "hypothesis_completion_structure_verified": True,
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
    }
    if obj["verification_boundaries"] != boundaries:
        raise FrozenManifestV2Error("manifest verification boundaries are not fail-closed")
    checks = {
        "one_fixed_n0": True,
        "registered_episode_budget_exact": True,
        "complete_hypothesis_rotation_in_every_block": True,
        "unique_block_opening_content_and_update_batch_identities": True,
        "optimizer_and_checkpoint_chain_exact": True,
        "all_blocks_structurally_pass": True,
        "feature_histograms_descriptive_only": True,
        "no_powered_384_group_statistics_claimed": True,
        "legacy_three_role_schemas_rejected": True,
    }
    if obj["checks"] != [{"name": name, "passed": passed} for name, passed in checks.items()]:
        raise FrozenManifestV2Error("manifest checks differ from live rederivation")
    if (
        _require_boolean(
            obj["hypothesis_completion_structure_passed"],
            name="manifest structural pass",
        )
        is not True
    ):
        raise FrozenManifestV2Error("manifest structural pass rederived false")
    if obj["substitution_contract"] != {
        "legacy_three_role_block_may_substitute": False,
        "legacy_three_role_training_audit_may_substitute": False,
        "evaluation_quartet_audit_may_substitute": False,
        "powered_stress_audit_may_substitute": False,
    }:
        raise FrozenManifestV2Error("manifest substitution contract is not fail-closed")
    unsigned = {key: obj[key] for key in tuple(obj)[:-1]}
    digest = _json_digest(unsigned, domain=_MANIFEST_DOMAIN)
    if obj["hypothesis_complete_manifest_digest"] != digest:
        raise FrozenManifestV2Error("manifest digest differs from canonical bytes")
    return _Manifest(obj, blocks, digest, n0, budget, episode_count)


def _construction_opening_digest(block: _Block) -> str:
    return _json_digest(
        {
            "catalog_digest": block.opening.catalog_digest,
            "supported_catalog_digest": block.opening.supported_catalog_digest,
            "live_rule_bindings": [
                {"rule_id": entry.rule_id, "truth_digest": entry.truth_digest}
                for entry in block.opening.entries
            ],
            "canonical_semantic_opening": [
                {"scene_index": scene_index, "accepted": accepted}
                for scene_index, accepted in sorted(block.opening.observations)
            ],
        },
        domain=_BLOCK_OPENING_DOMAIN,
    )


def _prompt_surface(block: _Block) -> tuple[str, int]:
    labels = dict(block.opening.observations)
    prompt = {
        "renderer": block.renderer_name,
        "ordered_opening": [
            {
                "scene_index": scene_index,
                "accepted": labels[scene_index],
                "rendered_scene": render_scene(scene_at(scene_index), block.renderer_name),
            }
            for scene_index in block.hidden_display_order
        ],
    }
    encoded = _dump_json(prompt)
    return _json_digest(prompt, domain=_PROMPT_SURFACE_DOMAIN), len(encoded) // 16


def _construction_clusters(
    blocks: tuple[_Block, ...],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    parent = list(range(len(blocks)))

    def find(position: int) -> int:
        while parent[position] != position:
            parent[position] = parent[parent[position]]
            position = parent[position]
        return position

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    projection_owner: dict[tuple[tuple[int, bool], ...], int] = {}
    for block_position, block in enumerate(blocks):
        rows = tuple(sorted(block.opening.observations))
        for omitted in range(len(rows)):
            projection = (*rows[:omitted], *rows[omitted + 1 :])
            owner = projection_owner.setdefault(projection, block_position)
            union(block_position, owner)
    members: dict[int, list[str]] = {}
    for position, block in enumerate(blocks):
        members.setdefault(find(position), []).append(_construction_opening_digest(block))
    result: list[tuple[str, tuple[str, ...]]] = []
    for values in members.values():
        member_digests = tuple(sorted(values))
        cluster_digest = _json_digest(
            {"member_opening_digests": list(member_digests)},
            domain=_CONSTRUCTION_CLUSTER_DOMAIN,
        )
        result.append((cluster_digest, member_digests))
    return tuple(sorted(result))


def _meta_block_obj(
    block: _Block,
    *,
    schedule_positions: Sequence[int],
    request_positions: Sequence[int],
    bank_prefix: int,
) -> dict[str, Any]:
    if len(schedule_positions) != block.opening.n0 or len(request_positions) != block.opening.n0:
        raise FrozenManifestV2Error("meta position sequences must align to every rotation")
    for name, values in (
        ("schedule", schedule_positions),
        ("request", request_positions),
    ):
        checked = tuple(
            _require_integer(item, name=f"{name} position", maximum=block.opening.n0 - 1) for item in values
        )
        if tuple(sorted(checked)) != tuple(range(block.opening.n0)):
            raise FrozenManifestV2Error(f"{name} positions must be one bounded local permutation")
    prompt_digest, length_bin = _prompt_surface(block)
    rotations = [
        {
            "official_identity": {
                "rule_id": rotation["official_rule_id"],
                "truth_digest": rotation["official_truth_digest"],
            },
            "model_visible_surface": {
                "renderer": block.renderer_name,
                "rendered_static_prompt_digest": prompt_digest,
                "rendered_static_prompt_length_bin": length_bin,
                "full_static_model_input_digest": block.static_digest,
            },
            "executor_only_surface": {
                "schedule_position": schedule_positions[position],
                "request_position": request_positions[position],
                "bank_prefix": bank_prefix,
            },
        }
        for position, rotation in enumerate(block.rotations)
    ]
    unsigned = {
        "bank_position": block.bank_position,
        "hypothesis_complete_block_digest": block.block_digest,
        "opening_content_digest": block.opening.content_digest,
        "construction_opening_digest": _construction_opening_digest(block),
        "rotations": rotations,
    }
    return {**unsigned, "meta_surface_block_digest": _json_digest(unsigned, domain=_META_BLOCK_DOMAIN)}


def serialize_frozen_meta_surfaces_v2(
    manifest_bytes: bytes,
    *,
    schedule_positions_by_block: Sequence[Sequence[int]] | None = None,
    request_positions_by_block: Sequence[Sequence[int]] | None = None,
) -> bytes:
    """Materialize the canonical planned meta-surface companion artifact.

    Position values remain executor-only.  This helper establishes a planned
    artifact, not evidence that a runtime respected it.
    """

    manifest_obj = _load_canonical_bytes(manifest_bytes, name="manifest")
    manifest = _parse_manifest(manifest_obj)
    defaults = tuple(tuple(range(block.opening.n0)) for block in manifest.blocks)
    schedule_rows = (
        defaults
        if schedule_positions_by_block is None
        else tuple(tuple(values) for values in schedule_positions_by_block)
    )
    request_rows = (
        defaults
        if request_positions_by_block is None
        else tuple(tuple(values) for values in request_positions_by_block)
    )
    if len(schedule_rows) != len(manifest.blocks) or len(request_rows) != len(manifest.blocks):
        raise FrozenManifestV2Error("meta position block count differs from manifest")
    prefix = 0
    blocks: list[dict[str, Any]] = []
    for block, schedule, request in zip(manifest.blocks, schedule_rows, request_rows, strict=True):
        blocks.append(
            _meta_block_obj(
                block,
                schedule_positions=schedule,
                request_positions=request,
                bank_prefix=prefix,
            )
        )
        prefix += block.opening.n0
    value = {
        "schema_version": FROZEN_META_SURFACE_SCHEMA_VERSION,
        "report_kind": _META_KIND,
        "authorization": dict(_META_AUTHORIZATION),
        "source_manifest_bytes_sha256": _sha256_bytes(manifest_bytes),
        "source_manifest_digest": manifest.digest,
        "catalog_digest": build_rule_catalog().digest,
        "supported_catalog_digest": build_supported_catalog_contract_v2().supported_catalog_digest,
        "blocks": blocks,
    }
    return _dump_json(value).encode("ascii")


def _verify_meta_surfaces(value: Mapping[str, Any], manifest: _Manifest, manifest_bytes: bytes) -> None:
    obj = _require_mapping(
        value,
        (
            "schema_version",
            "report_kind",
            "authorization",
            "source_manifest_bytes_sha256",
            "source_manifest_digest",
            "catalog_digest",
            "supported_catalog_digest",
            "blocks",
        ),
        name="frozen meta-surface artifact",
    )
    if (
        type(obj["schema_version"]) is not int
        or obj["schema_version"] != FROZEN_META_SURFACE_SCHEMA_VERSION
        or obj["report_kind"] != _META_KIND
    ):
        raise FrozenManifestV2Error("meta-surface schema identity mismatch")
    authorization = _require_mapping(
        obj["authorization"], tuple(_META_AUTHORIZATION), name="meta-surface authorization"
    )
    if dict(authorization) != _META_AUTHORIZATION:
        raise FrozenManifestV2Error("meta-surface authorization must remain false")
    if obj["source_manifest_bytes_sha256"] != _sha256_bytes(manifest_bytes) or (
        obj["source_manifest_digest"] != manifest.digest
    ):
        raise FrozenManifestV2Error("meta surfaces refer to different manifest bytes")
    if obj["catalog_digest"] != build_rule_catalog().digest or (
        obj["supported_catalog_digest"] != build_supported_catalog_contract_v2().supported_catalog_digest
    ):
        raise FrozenManifestV2Error("meta surfaces differ from live catalog contracts")
    raw_blocks = obj["blocks"]
    if type(raw_blocks) is not list or len(raw_blocks) != len(manifest.blocks):
        raise FrozenManifestV2Error("meta block count differs from manifest")
    prefix = 0
    for block, raw in zip(manifest.blocks, raw_blocks, strict=True):
        row = _require_mapping(
            raw,
            (
                "bank_position",
                "hypothesis_complete_block_digest",
                "opening_content_digest",
                "construction_opening_digest",
                "rotations",
                "meta_surface_block_digest",
            ),
            name="meta-surface block",
        )
        if _require_integer(row["bank_position"], name="meta bank position") != block.bank_position:
            raise FrozenManifestV2Error("meta blocks are reordered")
        if row["hypothesis_complete_block_digest"] != block.block_digest or (
            row["opening_content_digest"] != block.opening.content_digest
        ):
            raise FrozenManifestV2Error("meta block identity differs from semantic manifest block")
        if row["construction_opening_digest"] != _construction_opening_digest(block):
            raise FrozenManifestV2Error("meta construction identity uses caller labels or false content")
        raw_rotations = row["rotations"]
        if type(raw_rotations) is not list or len(raw_rotations) != block.opening.n0:
            raise FrozenManifestV2Error("meta rotations do not cover the complete V0")
        prompt_digest, length_bin = _prompt_surface(block)
        expected_visible = {
            "renderer": block.renderer_name,
            "rendered_static_prompt_digest": prompt_digest,
            "rendered_static_prompt_length_bin": length_bin,
            "full_static_model_input_digest": block.static_digest,
        }
        schedules: list[int] = []
        requests: list[int] = []
        for rotation, manifest_rotation in zip(raw_rotations, block.rotations, strict=True):
            surface = _require_mapping(
                rotation,
                ("official_identity", "model_visible_surface", "executor_only_surface"),
                name="meta rotation surface",
            )
            identity = _require_mapping(
                surface["official_identity"],
                ("rule_id", "truth_digest"),
                name="meta Official identity",
            )
            if identity != {
                "rule_id": manifest_rotation["official_rule_id"],
                "truth_digest": manifest_rotation["official_truth_digest"],
            }:
                raise FrozenManifestV2Error("meta Official identities are relabeled or reordered")
            visible = _require_mapping(
                surface["model_visible_surface"],
                (
                    "renderer",
                    "rendered_static_prompt_digest",
                    "rendered_static_prompt_length_bin",
                    "full_static_model_input_digest",
                ),
                name="model-visible meta surface",
            )
            if dict(visible) != expected_visible:
                raise FrozenManifestV2Error(
                    "renderer/static prompt differs across Official rotations or from live rendering"
                )
            executor = _require_mapping(
                surface["executor_only_surface"],
                ("schedule_position", "request_position", "bank_prefix"),
                name="executor-only meta surface",
            )
            schedules.append(
                _require_integer(
                    executor["schedule_position"],
                    name="schedule position",
                    maximum=block.opening.n0 - 1,
                )
            )
            requests.append(
                _require_integer(
                    executor["request_position"],
                    name="request position",
                    maximum=block.opening.n0 - 1,
                )
            )
            if _require_integer(executor["bank_prefix"], name="bank prefix") != prefix:
                raise FrozenManifestV2Error("executor bank prefix differs from manifest episode prefix")
        bounded = tuple(range(block.opening.n0))
        if tuple(sorted(schedules)) != bounded or tuple(sorted(requests)) != bounded:
            raise FrozenManifestV2Error("executor positions are not exact bounded local permutations")
        unsigned = {key: row[key] for key in tuple(row)[:-1]}
        if row["meta_surface_block_digest"] != _json_digest(unsigned, domain=_META_BLOCK_DOMAIN):
            raise FrozenManifestV2Error("meta-surface block digest differs from canonical bytes")
        prefix += block.opening.n0


@dataclass(frozen=True, slots=True)
class FrozenManifestRederivationV2:
    """Narrow evidence returned only after live semantic rederivation."""

    expected_bindings: FrozenManifestExpectedBindingsV2
    manifest_digest: str
    fixed_n0: int
    block_count: int
    episode_count: int
    construction_clusters: tuple[tuple[str, tuple[str, ...]], ...]
    block_stage_rows: tuple[dict[str, Any], ...]

    def __post_init__(self) -> None:
        if type(self.expected_bindings) is not FrozenManifestExpectedBindingsV2:
            raise FrozenManifestV2Error("rederivation requires exact expected bindings")
        _require_sha256(self.manifest_digest, name="manifest digest")
        if self.fixed_n0 not in _ALLOWED_N0:
            raise FrozenManifestV2Error("report fixed n0 is not registered")
        _require_integer(self.block_count, name="report block count", minimum=1)
        _require_integer(self.episode_count, name="report episode count", minimum=1)
        if self.episode_count != self.block_count * self.fixed_n0:
            raise FrozenManifestV2Error("report episode count differs from block completion")

    @property
    def digest(self) -> str:
        return _json_digest(self._unsigned_obj(), domain=_REPORT_DOMAIN)

    def _unsigned_obj(self) -> dict[str, Any]:
        return {
            "schema_version": FROZEN_MANIFEST_VERIFIER_SCHEMA_VERSION,
            "report_kind": _REPORT_KIND,
            "authorization": dict(_REPORT_AUTHORIZATION),
            "expected_and_live_bindings": self.expected_bindings.as_obj(),
            "manifest_digest": self.manifest_digest,
            "fixed_n0": self.fixed_n0,
            "block_count": self.block_count,
            "episode_count": self.episode_count,
            "construction_identity": {
                "identity_basis": (
                    "exact catalog-bound semantic opening plus transitive shared-nine-of-ten projections"
                ),
                "caller_block_and_opening_labels_used": False,
                "cluster_count": len(self.construction_clusters),
                "clusters": [
                    {
                        "construction_cluster_digest": digest,
                        "member_construction_opening_digests": list(members),
                    }
                    for digest, members in self.construction_clusters
                ],
            },
            "block_stage_identity_and_disjointness": list(self.block_stage_rows),
            "rederived_guarantees": {
                "canonical_manifest_bytes_bound": True,
                "canonical_meta_surface_bytes_bound": True,
                "producer_and_catalog_source_bytes_bound": True,
                "live_catalog_and_supported_allowlist_bound": True,
                "supported_v0_and_opening_content_rederived": True,
                "every_official_rotation_rederived": True,
                "identical_renderer_and_static_prompt_within_block": True,
                "executor_positions_bounded_and_absent_from_model_visible_surface": True,
                "exact_equal_objective_weights": True,
                "one_pre_post_atomic_update_disposition_rederived": True,
                "public_terminal_law_and_private_panel_identities_rederived": True,
                "available_cross_stage_scene_disjointness_rederived": True,
                "construction_clusters_content_rederived": True,
            },
            "unprovable_without_future_evidence": {
                "frozen_bank_generator_lineage_verified": False,
                "terminal_law_generator_attestation_verified": False,
                "panel_generator_receipt_verified": False,
                "precommitment_seed_independence_verified": False,
                "runtime_request_schedule_matched_planned_meta": False,
                "runtime_context_and_cache_resets_observed": False,
                "checkpoint_bytes_match_claimed_digests": False,
                "optimizer_atomicity_observed_at_runtime": False,
                "cross_block_scene_independence_verified": False,
                "query_challenge_and_evaluation_bank_identities_bound": False,
                "train_query_challenge_evaluation_scene_disjointness_verified": False,
                "training_bank_balance_verified": False,
                "powered_stress_support_containment_verified": False,
                "sampling_independence_verified": False,
            },
            "rederivation_passed": True,
        }

    def as_obj(self) -> dict[str, Any]:
        return {**self._unsigned_obj(), "frozen_manifest_rederivation_digest": self.digest}


def verify_frozen_hypothesis_complete_manifest_v2(
    manifest_bytes: bytes,
    meta_surface_bytes: bytes,
    *,
    expected_bindings: FrozenManifestExpectedBindingsV2,
) -> FrozenManifestRederivationV2:
    """Verify frozen inputs against external hashes and live semantic replay."""

    if type(expected_bindings) is not FrozenManifestExpectedBindingsV2:
        raise TypeError("expected_bindings must be FrozenManifestExpectedBindingsV2")
    if type(manifest_bytes) is not bytes or type(meta_surface_bytes) is not bytes:
        raise TypeError("manifest and meta-surface inputs must be exact bytes")
    if _sha256_bytes(manifest_bytes) != expected_bindings.manifest_bytes_sha256:
        raise FrozenManifestV2Error("manifest bytes differ from the externally frozen SHA-256")
    if _sha256_bytes(meta_surface_bytes) != expected_bindings.meta_surface_bytes_sha256:
        raise FrozenManifestV2Error("meta-surface bytes differ from the externally frozen SHA-256")
    hypothesis_source, statistical_source, catalog_source = _producer_source_digests()
    if hypothesis_source != expected_bindings.hypothesis_complete_source_sha256:
        raise FrozenManifestV2Error("hypothesis-complete producer source differs from frozen bytes")
    if statistical_source != expected_bindings.statistical_leakage_source_sha256:
        raise FrozenManifestV2Error("statistical-leakage producer source differs from frozen bytes")
    if catalog_source != expected_bindings.catalog_source_sha256:
        raise FrozenManifestV2Error("public catalog source differs from frozen bytes")
    catalog = build_rule_catalog()
    contract = build_supported_catalog_contract_v2()
    if catalog.digest != expected_bindings.catalog_digest:
        raise FrozenManifestV2Error("live catalog digest differs from frozen expectation")
    if contract.source_catalog_digest != catalog.digest:
        raise FrozenManifestV2Error("supported allowlist uses a different source catalog")
    if contract.supported_catalog_digest != expected_bindings.supported_catalog_digest:
        raise FrozenManifestV2Error("live supported allowlist differs from frozen expectation")
    if renderer_digest() != expected_bindings.renderer_registry_digest:
        raise FrozenManifestV2Error("live renderer registry differs from frozen expectation")

    manifest_obj = _load_canonical_bytes(manifest_bytes, name="manifest")
    manifest = _parse_manifest(manifest_obj)
    meta_obj = _load_canonical_bytes(meta_surface_bytes, name="meta-surface artifact")
    _verify_meta_surfaces(meta_obj, manifest, manifest_bytes)
    clusters = _construction_clusters(manifest.blocks)
    stage_rows = tuple(
        {
            "bank_position": block.bank_position,
            "construction_opening_digest": _construction_opening_digest(block),
            "opening_content_digest": block.opening.content_digest,
            "opening_scene_set_digest": block.opening.scene_set_digest,
            "version_space_rule_ids": [entry.rule_id for entry in block.opening.entries],
            "unconditional_scene_law_digest": block.terminal_law.digest,
            "materialized_training_panel_digest": block.panel.digest,
            "opening_and_public_law_support_disjoint": True,
            "opening_and_materialized_panel_disjoint": True,
            "materialized_panel_subset_of_public_law_support": True,
            "cross_block_scene_disjointness_claimed": False,
        }
        for block in manifest.blocks
    )
    return FrozenManifestRederivationV2(
        expected_bindings,
        manifest.digest,
        manifest.fixed_n0,
        len(manifest.blocks),
        manifest.episode_count,
        clusters,
        stage_rows,
    )


def serialize_frozen_manifest_rederivation_v2(
    report: FrozenManifestRederivationV2,
) -> bytes:
    """Serialize a freshly rederived report; no parse-to-authorization API exists."""

    if type(report) is not FrozenManifestRederivationV2:
        raise TypeError("report must be FrozenManifestRederivationV2")
    return _dump_json(report.as_obj()).encode("ascii")
