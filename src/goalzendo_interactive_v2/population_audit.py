"""Exact, nonauthorizing rule-triple population audit for prospective G03-v2.

The audit enumerates every extensionally distinct ``(P, Q, C)`` triple from
the public G03 catalog, where ``P`` is a placard literal, ``Q`` is a
one-literal piece rule, and ``C`` is a two-literal piece composition.  It
computes all eight ``(C, P, Q)`` truth-cell counts over the complete scene
universe and streams those rows into content digests rather than retaining a
million-row table in memory.

This is a population/capacity audit only.  In particular, a sufficient joint
truth-cell reserve does not prove that matched openings, challenge panels, or
an episode bank can be constructed.  Nothing in this module authorizes a
model weight update.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal, cast

from goalzendo_interactive.catalog import (
    RULE_CATALOG_SCHEMA_VERSION,
    TRUTH_VECTOR_SCHEMA_VERSION,
    CatalogEntry,
    build_rule_catalog,
)
from goalzendo_interactive.provenance import (
    INTERACTIVE_SOURCE_PROVENANCE_SCHEMA_VERSION,
    interactive_source_provenance,
)
from goalzendo_interactive.rules import (
    ATOM_OPS,
    BINARY_OPS,
    RULE_SCHEMA_VERSION,
    BinaryOp,
    BinaryRule,
)
from goalzendo_interactive.rules import Literal as RuleLiteral
from goalzendo_interactive.schema import SCENE_COUNT, SCENE_SCHEMA_VERSION
from goalzendo_interactive.stage_partitions_v2 import (
    RULE_PARTITION_SCHEMA_VERSION_V2,
    RULE_PARTITIONS_V2,
    RulePartitionV2,
    build_rule_identity_partitions_v2,
)

POPULATION_AUDIT_SCHEMA_VERSION = 1
RESERVE_THRESHOLDS: tuple[int, ...] = (22, 27, 32)
ENGINEERING_LEAKAGE_REQUIRED_TARGETS = 384

CandidateRuleFamilyV2 = Literal[
    "placard_literal",
    "one_literal_piece",
    "composed_two_literal_piece",
]
CatalogAuditFamilyV2 = Literal[
    "placard_literal",
    "one_literal_piece",
    "composed_two_literal_piece",
    "excluded_placard_composition",
]

CANDIDATE_RULE_FAMILIES: tuple[CandidateRuleFamilyV2, ...] = (
    "placard_literal",
    "one_literal_piece",
    "composed_two_literal_piece",
)
CATALOG_AUDIT_FAMILIES: tuple[CatalogAuditFamilyV2, ...] = (
    *CANDIDATE_RULE_FAMILIES,
    "excluded_placard_composition",
)
COMPOSED_STRATA: tuple[tuple[BinaryOp, int], ...] = tuple(
    (op, negated_literal_count) for op in BINARY_OPS for negated_literal_count in range(3)
)
JOINT_CELL_ORDER: tuple[str, ...] = tuple(
    f"{composed}{placard}{literal}" for composed in (0, 1) for placard in (0, 1) for literal in (0, 1)
)

_UNIVERSE_MASK = (1 << SCENE_COUNT) - 1
_REPORT_KIND = "g03-v2-exhaustive-rule-triple-population-audit"
_REPORT_DOMAIN = "goalzendo-interactive-v2-population-audit-v1"
_DEPENDENCY_DOMAIN = "goalzendo-interactive-v2-population-dependencies-v1"
_FAMILY_IDENTITY_DOMAIN = b"goalzendo-interactive-v2-family-identities-v1\0"
_SUPPORTED_CATALOG_DOMAIN = b"goalzendo-interactive-v2-supported-catalog-v1\0"
_EXCLUDED_CATALOG_DOMAIN = b"goalzendo-interactive-v2-excluded-catalog-v1\0"
_TRIPLE_IDENTITY_DOMAIN = b"goalzendo-interactive-v2-triple-identities-v1\0"
_TRIPLE_ROW_DOMAIN = b"goalzendo-interactive-v2-triple-cell-row-v1\0"
_TRIPLE_TABLE_DOMAIN = b"goalzendo-interactive-v2-triple-cell-table-v1\0"
_TRIPLE_BINDING_DOMAIN = "goalzendo-interactive-v2-rule-triple-binding-v2"
_CELL_VECTOR_HISTOGRAM_DOMAIN = "goalzendo-interactive-v2-cell-vector-histogram-v1"
_MINIMUM_HISTOGRAM_DOMAIN = "goalzendo-interactive-v2-minimum-cell-histogram-v1"
_ELIGIBLE_TABLE_DOMAIN_PREFIX = b"goalzendo-interactive-v2-eligible-triples-v1\0"
_PARTNER_TABLE_DOMAIN_PREFIX = b"goalzendo-interactive-v2-partner-counts-v1\0"
_ELIGIBLE_TARGET_DOMAIN_PREFIX = b"goalzendo-interactive-v2-eligible-targets-v1\0"

_FAMILY_CODES: dict[CatalogAuditFamilyV2, bytes] = {
    "placard_literal": b"P",
    "one_literal_piece": b"Q",
    "composed_two_literal_piece": b"C",
    "excluded_placard_composition": b"X",
}

_THRESHOLD_RATIONALES = {
    22: "eleven 16-scene panels with two scenes from each joint truth cell",
    27: "the 22-panel reserve plus the worst registered five-scene opening-cell demand",
    32: "a conservative engineering sensitivity reserve beyond panels and opening",
}

_AUTHORIZATION = {
    "scope": "prospective_population_capacity_audit_only",
    "production_bank_materialized": False,
    "matched_opening_feasibility_proven": False,
    "challenge_panel_feasibility_proven": False,
    "bank_feasibility_proven": False,
    "weight_updates_authorized": False,
}


class PopulationAuditV2Error(ValueError):
    """Raised when a G03-v2 population report cannot be verified exactly."""


def _dump_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise PopulationAuditV2Error(f"value is not canonical JSON: {exc}") from exc


def _load_json(text: str) -> Any:
    if type(text) is not str or not text:
        raise PopulationAuditV2Error("JSON input must be a nonempty string")

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PopulationAuditV2Error(f"duplicate JSON object key: {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise PopulationAuditV2Error(f"non-finite JSON constant is forbidden: {value}")

    try:
        return json.loads(
            text,
            object_pairs_hook=no_duplicates,
            parse_constant=reject_constant,
        )
    except PopulationAuditV2Error:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise PopulationAuditV2Error(f"invalid JSON: {exc}") from exc


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


def _new_hasher(domain: bytes) -> Any:
    digest = hashlib.sha256()
    digest.update(domain)
    return digest


def _identity_bytes(entry: CatalogEntry) -> bytes:
    return entry.index.to_bytes(4, "big") + bytes.fromhex(entry.truth_digest)


def _finalize_counted_digest(digest: Any, count: int) -> str:
    digest.update(count.to_bytes(8, "big"))
    return cast(str, digest.hexdigest())


def classify_catalog_identity_v2(entry: CatalogEntry) -> CatalogAuditFamilyV2:
    """Classify a canonical AST, recognizing mixed placard rules as excluded.

    The classifier is intentionally closed over the released ``Literal`` and
    ``BinaryRule`` AST classes.  A new rule node, atom operation, or binary
    operation raises instead of being silently assigned to a family.
    """

    if type(entry) is not CatalogEntry:
        raise TypeError("classify_catalog_identity_v2 requires a CatalogEntry")
    rule = entry.rule
    if type(rule) is RuleLiteral:
        if rule.atom.op not in ATOM_OPS:
            raise PopulationAuditV2Error("literal uses an unknown atom operation")
        return "placard_literal" if rule.atom.op == "placard_is" else "one_literal_piece"
    if type(rule) is BinaryRule:
        if rule.op not in BINARY_OPS:
            raise PopulationAuditV2Error("composition uses an unknown binary operation")
        if len(rule.args) != 2 or any(type(item) is not RuleLiteral for item in rule.args):
            raise PopulationAuditV2Error("composition is not exactly two canonical literals")
        atom_ops = tuple(item.atom.op for item in rule.args)
        if any(op not in ATOM_OPS for op in atom_ops):
            raise PopulationAuditV2Error("composition uses an unknown atom operation")
        if "placard_is" in atom_ops:
            return "excluded_placard_composition"
        return "composed_two_literal_piece"
    raise PopulationAuditV2Error(f"unknown public rule AST family: {type(rule).__name__}")


def require_supported_catalog_identity_v2(
    entry: CatalogEntry,
) -> CandidateRuleFamilyV2:
    """Return a v2 family or fail if the identity is outside the v2 grammar."""

    family = classify_catalog_identity_v2(entry)
    if family not in CANDIDATE_RULE_FAMILIES:
        raise PopulationAuditV2Error(f"catalog family {family!r} is unsupported in v2 Official roles and V0")
    return family


@dataclass(frozen=True, slots=True)
class SupportedCatalogContractV2:
    """Immutable full-catalog namespace plus the admissible v2 allowlist."""

    source_catalog_digest: str
    supported_indices: tuple[int, ...]
    supported_catalog_digest: str
    excluded_indices: tuple[int, ...]
    excluded_catalog_digest: str

    def __post_init__(self) -> None:
        if not _is_sha256(self.source_catalog_digest):
            raise PopulationAuditV2Error("supported-catalog source digest is malformed")
        if not _is_sha256(self.supported_catalog_digest) or not _is_sha256(self.excluded_catalog_digest):
            raise PopulationAuditV2Error("supported-catalog allowlist digest is malformed")
        if self.supported_indices != tuple(sorted(set(self.supported_indices))):
            raise PopulationAuditV2Error("supported catalog indices must be sorted and unique")
        if self.excluded_indices != tuple(sorted(set(self.excluded_indices))):
            raise PopulationAuditV2Error("excluded catalog indices must be sorted and unique")
        if set(self.supported_indices) & set(self.excluded_indices):
            raise PopulationAuditV2Error("supported and excluded catalog indices overlap")


@lru_cache(maxsize=1)
def build_supported_catalog_contract_v2() -> SupportedCatalogContractV2:
    """Derive the exact supported-index allowlist without filtering IDs."""

    catalog = build_rule_catalog()
    supported_digest = _new_hasher(_SUPPORTED_CATALOG_DOMAIN)
    supported_digest.update(bytes.fromhex(catalog.digest))
    excluded_digest = _new_hasher(_EXCLUDED_CATALOG_DOMAIN)
    excluded_digest.update(bytes.fromhex(catalog.digest))
    supported: list[int] = []
    excluded: list[int] = []
    for entry in catalog:
        family = classify_catalog_identity_v2(entry)
        if family in CANDIDATE_RULE_FAMILIES:
            supported_family = require_supported_catalog_identity_v2(entry)
            supported.append(entry.index)
            supported_digest.update(_identity_bytes(entry))
            supported_digest.update(_FAMILY_CODES[supported_family])
        elif family == "excluded_placard_composition":
            excluded.append(entry.index)
            excluded_digest.update(_identity_bytes(entry))
            excluded_digest.update(_FAMILY_CODES[family])
        else:  # pragma: no cover - exhaustive classifier
            raise PopulationAuditV2Error("an unknown catalog family reached the allowlist")
    if len(supported) + len(excluded) != len(catalog):
        raise PopulationAuditV2Error("supported/excluded catalog allowlist is incomplete")
    return SupportedCatalogContractV2(
        source_catalog_digest=catalog.digest,
        supported_indices=tuple(supported),
        supported_catalog_digest=_finalize_counted_digest(supported_digest, len(supported)),
        excluded_indices=tuple(excluded),
        excluded_catalog_digest=_finalize_counted_digest(excluded_digest, len(excluded)),
    )


def composed_stratum_v2(entry: CatalogEntry) -> tuple[BinaryOp, int]:
    """Return the operator and exact number of negated literals for ``C``."""

    if classify_catalog_identity_v2(entry) != "composed_two_literal_piece":
        raise PopulationAuditV2Error("composed_stratum_v2 requires a composed piece rule")
    rule = cast(BinaryRule, entry.rule)
    negated_literal_count = sum(literal.negated for literal in rule.args)
    if (rule.op, negated_literal_count) not in COMPOSED_STRATA:
        raise PopulationAuditV2Error("composition lies outside a registered stratum")
    return rule.op, negated_literal_count


def _validate_triple_families(
    placard: CatalogEntry,
    literal: CatalogEntry,
    composed: CatalogEntry,
) -> None:
    observed = (
        require_supported_catalog_identity_v2(placard),
        require_supported_catalog_identity_v2(literal),
        require_supported_catalog_identity_v2(composed),
    )
    expected: tuple[CandidateRuleFamilyV2, ...] = CANDIDATE_RULE_FAMILIES
    if observed != expected:
        raise PopulationAuditV2Error(f"triple families must be {expected!r}, observed {observed!r}")
    if len({placard.truth.bits, literal.truth.bits, composed.truth.bits}) != 3:
        raise PopulationAuditV2Error("triple identities must be extensionally distinct")


def joint_truth_cell_counts_v2(
    placard: CatalogEntry,
    literal: CatalogEntry,
    composed: CatalogEntry,
) -> tuple[int, int, int, int, int, int, int, int]:
    """Compute full-universe counts in canonical ``C,P,Q=000..111`` order."""

    _validate_triple_families(placard, literal, composed)
    return _joint_truth_cell_counts_unchecked(placard, literal, composed)


def _joint_truth_cell_counts_unchecked(
    placard: CatalogEntry,
    literal: CatalogEntry,
    composed: CatalogEntry,
) -> tuple[int, int, int, int, int, int, int, int]:
    p_bits = placard.truth.bits
    q_bits = literal.truth.bits
    c_bits = composed.truth.bits
    not_p = _UNIVERSE_MASK ^ p_bits
    not_q = _UNIVERSE_MASK ^ q_bits
    not_c = _UNIVERSE_MASK ^ c_bits
    pq_masks = (
        not_p & not_q,
        not_p & q_bits,
        p_bits & not_q,
        p_bits & q_bits,
    )
    return cast(
        tuple[int, int, int, int, int, int, int, int],
        tuple((c_mask & pq_mask).bit_count() for c_mask in (not_c, c_bits) for pq_mask in pq_masks),
    )


def _triple_cell_row_payload(
    placard: CatalogEntry,
    literal: CatalogEntry,
    composed: CatalogEntry,
    cells: tuple[int, int, int, int, int, int, int, int],
) -> bytes:
    if len(cells) != 8 or any(
        isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= SCENE_COUNT
        for count in cells
    ):
        raise PopulationAuditV2Error("joint truth cells must be eight bounded integers")
    if sum(cells) != SCENE_COUNT:
        raise PopulationAuditV2Error("joint truth cells must partition the complete universe")
    return b"".join(
        (
            _identity_bytes(placard),
            _identity_bytes(literal),
            _identity_bytes(composed),
            *(count.to_bytes(2, "big") for count in cells),
        )
    )


def triple_cell_digest_v2(
    placard: CatalogEntry,
    literal: CatalogEntry,
    composed: CatalogEntry,
) -> str:
    """Return the domain-separated digest of one exact triple/cell row."""

    cells = joint_truth_cell_counts_v2(placard, literal, composed)
    digest = _new_hasher(_TRIPLE_ROW_DOMAIN)
    digest.update(_triple_cell_row_payload(placard, literal, composed, cells))
    return cast(str, digest.hexdigest())


def _resolve_catalog_rule_id(rule_id: str) -> CatalogEntry:
    if (
        type(rule_id) is not str
        or len(rule_id) != 9
        or not rule_id.startswith("g03r")
        or not rule_id[4:].isdigit()
    ):
        raise PopulationAuditV2Error(f"malformed public rule id: {rule_id!r}")
    index = int(rule_id[4:])
    catalog = build_rule_catalog()
    if not 0 <= index < len(catalog) or catalog[index].rule_id != rule_id:
        raise PopulationAuditV2Error(f"unknown public rule id: {rule_id!r}")
    return catalog[index]


def _binding_preimage(
    catalog_digest: str,
    supported_catalog_digest: str,
    placard: CatalogEntry,
    literal: CatalogEntry,
    composed: CatalogEntry,
    cells: tuple[int, int, int, int, int, int, int, int],
    cell_digest: str,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "catalog_digest": catalog_digest,
        "supported_catalog_digest": supported_catalog_digest,
        "candidate_order": ["P", "Q", "C"],
        "candidates": [
            {
                "slot": slot,
                "family": family,
                "rule_id": entry.rule_id,
                "truth_digest": entry.truth_digest,
            }
            for slot, family, entry in (
                ("P", "placard_literal", placard),
                ("Q", "one_literal_piece", literal),
                ("C", "composed_two_literal_piece", composed),
            )
        ],
        "joint_cell_order_c_p_q": list(JOINT_CELL_ORDER),
        "exact_joint_truth_cell_counts": list(cells),
        "exact_joint_cell_digest": cell_digest,
    }


@dataclass(frozen=True, slots=True)
class RuleTripleBindingV2:
    """Catalog-resolved semantic identity for one ordered ``(P,Q,C)`` triple.

    Construction is deliberately expensive enough to be trustworthy: the
    public catalog is resolved by rule id, truth digests and families are
    checked, the complete-universe cell vector is recomputed, and both the
    row and final binding digests are derived again.  The final
    ``triple_digest`` is therefore suitable as a later role block's semantic
    triple identifier; an arbitrary caller-provided SHA-256 is not.
    """

    catalog_digest: str
    supported_catalog_digest: str
    placard_rule_id: str
    placard_truth_digest: str
    literal_rule_id: str
    literal_truth_digest: str
    composed_rule_id: str
    composed_truth_digest: str
    exact_joint_truth_cell_counts: tuple[int, int, int, int, int, int, int, int]
    exact_joint_cell_digest: str
    triple_digest: str

    def __post_init__(self) -> None:
        catalog = build_rule_catalog()
        if self.catalog_digest != catalog.digest:
            raise PopulationAuditV2Error("triple binding has the wrong catalog digest")
        contract = build_supported_catalog_contract_v2()
        if self.supported_catalog_digest != contract.supported_catalog_digest:
            raise PopulationAuditV2Error("triple binding has the wrong supported-catalog digest")
        placard = _resolve_catalog_rule_id(self.placard_rule_id)
        literal = _resolve_catalog_rule_id(self.literal_rule_id)
        composed = _resolve_catalog_rule_id(self.composed_rule_id)
        for name, observed, entry in (
            ("placard", self.placard_truth_digest, placard),
            ("literal", self.literal_truth_digest, literal),
            ("composed", self.composed_truth_digest, composed),
        ):
            if observed != entry.truth_digest:
                raise PopulationAuditV2Error(
                    f"triple binding {name} truth digest differs from the public catalog"
                )
        _validate_triple_families(placard, literal, composed)
        expected_cells = _joint_truth_cell_counts_unchecked(placard, literal, composed)
        if self.exact_joint_truth_cell_counts != expected_cells:
            raise PopulationAuditV2Error("triple binding cells differ from full-universe recomputation")
        expected_cell_digest = triple_cell_digest_v2(placard, literal, composed)
        if self.exact_joint_cell_digest != expected_cell_digest:
            raise PopulationAuditV2Error("triple binding cell digest is inconsistent")
        expected_triple_digest = _json_digest(
            _binding_preimage(
                catalog.digest,
                contract.supported_catalog_digest,
                placard,
                literal,
                composed,
                expected_cells,
                expected_cell_digest,
            ),
            domain=_TRIPLE_BINDING_DOMAIN,
        )
        if self.triple_digest != expected_triple_digest:
            raise PopulationAuditV2Error("derived semantic-triple digest is inconsistent")

    @property
    def digest(self) -> str:
        return self.triple_digest

    def as_obj(self) -> dict[str, Any]:
        placard = _resolve_catalog_rule_id(self.placard_rule_id)
        literal = _resolve_catalog_rule_id(self.literal_rule_id)
        composed = _resolve_catalog_rule_id(self.composed_rule_id)
        value = _binding_preimage(
            self.catalog_digest,
            self.supported_catalog_digest,
            placard,
            literal,
            composed,
            self.exact_joint_truth_cell_counts,
            self.exact_joint_cell_digest,
        )
        value["triple_digest"] = self.triple_digest
        return value


def build_rule_triple_binding_v2(
    placard_rule_id: str,
    literal_rule_id: str,
    composed_rule_id: str,
) -> RuleTripleBindingV2:
    """Resolve three public rule ids and derive their canonical binding."""

    catalog = build_rule_catalog()
    contract = build_supported_catalog_contract_v2()
    placard = _resolve_catalog_rule_id(placard_rule_id)
    literal = _resolve_catalog_rule_id(literal_rule_id)
    composed = _resolve_catalog_rule_id(composed_rule_id)
    _validate_triple_families(placard, literal, composed)
    cells = _joint_truth_cell_counts_unchecked(placard, literal, composed)
    cell_digest = triple_cell_digest_v2(placard, literal, composed)
    triple_digest = _json_digest(
        _binding_preimage(
            catalog.digest,
            contract.supported_catalog_digest,
            placard,
            literal,
            composed,
            cells,
            cell_digest,
        ),
        domain=_TRIPLE_BINDING_DOMAIN,
    )
    return RuleTripleBindingV2(
        catalog.digest,
        contract.supported_catalog_digest,
        placard.rule_id,
        placard.truth_digest,
        literal.rule_id,
        literal.truth_digest,
        composed.rule_id,
        composed.truth_digest,
        cells,
        cell_digest,
        triple_digest,
    )


def verify_rule_triple_binding_v2(binding: RuleTripleBindingV2) -> RuleTripleBindingV2:
    if type(binding) is not RuleTripleBindingV2:
        raise TypeError("verify_rule_triple_binding_v2 requires a RuleTripleBindingV2")
    expected = build_rule_triple_binding_v2(
        binding.placard_rule_id,
        binding.literal_rule_id,
        binding.composed_rule_id,
    )
    if binding != expected:
        raise PopulationAuditV2Error("rule-triple binding differs from exact catalog recomputation")
    return expected


def serialize_rule_triple_binding_v2(binding: RuleTripleBindingV2) -> str:
    if type(binding) is not RuleTripleBindingV2:
        raise TypeError("serialize_rule_triple_binding_v2 requires a RuleTripleBindingV2")
    verify_rule_triple_binding_v2(binding)
    return _dump_json(binding.as_obj())


def _require_ordered_mapping(
    value: object,
    keys: tuple[str, ...],
    *,
    name: str,
) -> Mapping[str, Any]:
    if type(value) is not dict or tuple(value) != keys:
        raise PopulationAuditV2Error(f"{name} has noncanonical or reordered fields")
    return cast(Mapping[str, Any], value)


def rule_triple_binding_v2_from_obj(value: object) -> RuleTripleBindingV2:
    obj = _require_ordered_mapping(
        value,
        (
            "schema_version",
            "catalog_digest",
            "supported_catalog_digest",
            "candidate_order",
            "candidates",
            "joint_cell_order_c_p_q",
            "exact_joint_truth_cell_counts",
            "exact_joint_cell_digest",
            "triple_digest",
        ),
        name="rule-triple binding",
    )
    if type(obj["schema_version"]) is not int or obj["schema_version"] != 2:
        raise PopulationAuditV2Error("unknown rule-triple binding schema version")
    if obj["candidate_order"] != ["P", "Q", "C"]:
        raise PopulationAuditV2Error("rule-triple binding candidate order is not P,Q,C")
    if obj["joint_cell_order_c_p_q"] != list(JOINT_CELL_ORDER):
        raise PopulationAuditV2Error("rule-triple binding joint-cell order differs")
    candidates = obj["candidates"]
    if type(candidates) is not list or len(candidates) != 3:
        raise PopulationAuditV2Error("rule-triple binding requires three candidates")
    parsed_candidates: list[Mapping[str, Any]] = []
    for position, (slot, family) in enumerate(
        (
            ("P", "placard_literal"),
            ("Q", "one_literal_piece"),
            ("C", "composed_two_literal_piece"),
        )
    ):
        candidate = _require_ordered_mapping(
            candidates[position],
            ("slot", "family", "rule_id", "truth_digest"),
            name=f"rule-triple candidate {position}",
        )
        if (candidate["slot"], candidate["family"]) != (slot, family):
            raise PopulationAuditV2Error("rule-triple candidate slot/family is inconsistent")
        if type(candidate["rule_id"]) is not str or not _is_sha256(candidate["truth_digest"]):
            raise PopulationAuditV2Error("rule-triple candidate identity is malformed")
        parsed_candidates.append(candidate)
    raw_cells = obj["exact_joint_truth_cell_counts"]
    if (
        type(raw_cells) is not list
        or len(raw_cells) != 8
        or any(isinstance(item, bool) or not isinstance(item, int) for item in raw_cells)
    ):
        raise PopulationAuditV2Error("rule-triple binding cell vector is malformed")
    if not all(
        _is_sha256(obj[field])
        for field in (
            "catalog_digest",
            "supported_catalog_digest",
            "exact_joint_cell_digest",
            "triple_digest",
        )
    ):
        raise PopulationAuditV2Error("rule-triple binding contains a malformed digest")
    return RuleTripleBindingV2(
        cast(str, obj["catalog_digest"]),
        cast(str, obj["supported_catalog_digest"]),
        cast(str, parsed_candidates[0]["rule_id"]),
        cast(str, parsed_candidates[0]["truth_digest"]),
        cast(str, parsed_candidates[1]["rule_id"]),
        cast(str, parsed_candidates[1]["truth_digest"]),
        cast(str, parsed_candidates[2]["rule_id"]),
        cast(str, parsed_candidates[2]["truth_digest"]),
        cast(tuple[int, int, int, int, int, int, int, int], tuple(raw_cells)),
        cast(str, obj["exact_joint_cell_digest"]),
        cast(str, obj["triple_digest"]),
    )


def parse_rule_triple_binding_v2(
    text: str,
    *,
    require_canonical: bool = True,
    expected_digest: str | None = None,
) -> RuleTripleBindingV2:
    value = _load_json(text)
    binding = rule_triple_binding_v2_from_obj(value)
    canonical_obj_text = _dump_json(binding.as_obj())
    if canonical_obj_text != _dump_json(value):
        raise PopulationAuditV2Error("rule-triple binding has inconsistent derived fields")
    if require_canonical and canonical_obj_text != text:
        raise PopulationAuditV2Error("rule-triple binding JSON is not canonical compact JSON")
    if expected_digest is not None:
        if not _is_sha256(expected_digest):
            raise PopulationAuditV2Error("expected rule-triple digest is malformed")
        if binding.digest != expected_digest:
            raise PopulationAuditV2Error("rule-triple binding digest does not match expected")
    return binding


@dataclass(frozen=True, slots=True)
class FamilyCountV2:
    family: CatalogAuditFamilyV2
    identity_count: int

    def as_obj(self) -> dict[str, Any]:
        return {"family": self.family, "identity_count": self.identity_count}


@dataclass(frozen=True, slots=True)
class PartnerCountFrequencyV2:
    eligible_partner_count: int
    composed_target_count: int

    def as_obj(self) -> dict[str, int]:
        return {
            "eligible_partner_count": self.eligible_partner_count,
            "composed_target_count": self.composed_target_count,
        }


@dataclass(frozen=True, slots=True)
class StageStratumCapacityV2:
    stage_partition: RulePartitionV2
    composed_operator: BinaryOp
    negated_literal_count: int
    composed_target_count: int
    eligible_composed_target_count: int
    eligible_triple_count: int

    def as_obj(self) -> dict[str, Any]:
        return {
            "stage_partition": self.stage_partition,
            "composed_operator": self.composed_operator,
            "negated_literal_count": self.negated_literal_count,
            "composed_target_count": self.composed_target_count,
            "eligible_composed_target_count": self.eligible_composed_target_count,
            "eligible_triple_count": self.eligible_triple_count,
        }


@dataclass(frozen=True, slots=True)
class StageCapacityV2:
    stage_partition: RulePartitionV2
    composed_target_count: int
    eligible_composed_target_count: int
    eligible_triple_count: int

    def as_obj(self) -> dict[str, Any]:
        return {
            "stage_partition": self.stage_partition,
            "composed_target_count": self.composed_target_count,
            "eligible_composed_target_count": self.eligible_composed_target_count,
            "eligible_triple_count": self.eligible_triple_count,
        }


@dataclass(frozen=True, slots=True)
class ReserveCapacityV2:
    reserve_per_joint_cell: int
    rationale: str
    threshold_frozen: bool
    eligible_triple_count: int
    eligible_composed_target_count: int
    eligible_triple_table_digest: str
    eligible_composed_target_digest: str
    composed_partner_count_table_digest: str
    partner_count_histogram: tuple[PartnerCountFrequencyV2, ...]
    by_stage_partition: tuple[StageCapacityV2, ...]
    by_stage_and_composed_stratum: tuple[StageStratumCapacityV2, ...]

    def as_obj(self) -> dict[str, Any]:
        return {
            "reserve_per_joint_cell": self.reserve_per_joint_cell,
            "rationale": self.rationale,
            "threshold_frozen": self.threshold_frozen,
            "eligible_triple_count": self.eligible_triple_count,
            "eligible_composed_target_count": self.eligible_composed_target_count,
            "eligible_triple_table_digest": self.eligible_triple_table_digest,
            "eligible_composed_target_digest": self.eligible_composed_target_digest,
            "composed_partner_count_table_digest": self.composed_partner_count_table_digest,
            "partner_count_histogram": [row.as_obj() for row in self.partner_count_histogram],
            "by_stage_partition": [row.as_obj() for row in self.by_stage_partition],
            "by_stage_and_composed_stratum": [row.as_obj() for row in self.by_stage_and_composed_stratum],
        }


@dataclass(frozen=True, slots=True)
class PopulationAuditV2:
    source_fingerprint: str
    catalog_digest: str
    stage_partition_digest: str
    dependency_manifest_digest: str
    family_counts: tuple[FamilyCountV2, ...]
    family_identity_table_digest: str
    supported_catalog_identity_count: int
    supported_catalog_digest: str
    excluded_catalog_identity_count: int
    excluded_catalog_digest: str
    cartesian_triple_count: int
    extensionally_distinct_triple_count: int
    triple_identity_table_digest: str
    exact_joint_cell_table_digest: str
    unique_joint_cell_vector_count: int
    joint_cell_vector_histogram_digest: str
    distinct_minimum_cell_count_count: int
    minimum_cell_count: int
    maximum_minimum_cell_count: int
    minimum_cell_count_histogram_digest: str
    below_conservative_threshold_histogram: tuple[tuple[int, int], ...]
    reserve_capacities: tuple[ReserveCapacityV2, ...]
    engineering_leakage_observed_targets_at_32: int
    engineering_leakage_capacity_passed_at_32: bool

    def __post_init__(self) -> None:
        for value in (
            self.source_fingerprint,
            self.catalog_digest,
            self.stage_partition_digest,
            self.dependency_manifest_digest,
            self.family_identity_table_digest,
            self.supported_catalog_digest,
            self.excluded_catalog_digest,
            self.triple_identity_table_digest,
            self.exact_joint_cell_table_digest,
            self.joint_cell_vector_histogram_digest,
            self.minimum_cell_count_histogram_digest,
        ):
            if not _is_sha256(value):
                raise PopulationAuditV2Error("population audit contains a malformed digest")
        if tuple(row.family for row in self.family_counts) != CATALOG_AUDIT_FAMILIES:
            raise PopulationAuditV2Error("family-count rows are not canonical")
        if tuple(row.reserve_per_joint_cell for row in self.reserve_capacities) != RESERVE_THRESHOLDS:
            raise PopulationAuditV2Error("reserve-capacity rows are not canonical")
        if any(row.threshold_frozen for row in self.reserve_capacities):
            raise PopulationAuditV2Error("prospective reserve thresholds must remain unfrozen")
        if self.engineering_leakage_capacity_passed_at_32 != (
            self.engineering_leakage_observed_targets_at_32 >= ENGINEERING_LEAKAGE_REQUIRED_TARGETS
        ):
            raise PopulationAuditV2Error("engineering leakage capacity decision is inconsistent")

    @property
    def digest(self) -> str:
        return _json_digest(self.as_obj(), domain=_REPORT_DOMAIN)

    def as_obj(self) -> dict[str, Any]:
        dependency_manifest = {
            "interactive_source_provenance_schema_version": (INTERACTIVE_SOURCE_PROVENANCE_SCHEMA_VERSION),
            "interactive_source_fingerprint": self.source_fingerprint,
            "scene_schema_version": SCENE_SCHEMA_VERSION,
            "scene_count": SCENE_COUNT,
            "rule_schema_version": RULE_SCHEMA_VERSION,
            "truth_vector_schema_version": TRUTH_VECTOR_SCHEMA_VERSION,
            "rule_catalog_schema_version": RULE_CATALOG_SCHEMA_VERSION,
            "catalog_digest": self.catalog_digest,
            "stage_partition_schema_version": RULE_PARTITION_SCHEMA_VERSION_V2,
            "stage_partition_digest": self.stage_partition_digest,
        }
        return {
            "schema_version": POPULATION_AUDIT_SCHEMA_VERSION,
            "report_kind": _REPORT_KIND,
            "status": "prospective_unfrozen_nonauthorizing",
            "dependencies": dependency_manifest,
            "dependency_manifest_digest": self.dependency_manifest_digest,
            "family_classification": {
                "candidate_families": list(CANDIDATE_RULE_FAMILIES),
                "recognized_exclusion": "excluded_placard_composition",
                "unknown_ast_policy": "fail_closed",
                "counts": [row.as_obj() for row in self.family_counts],
                "identity_table_digest": self.family_identity_table_digest,
            },
            "v2_supported_catalog": {
                "source_catalog_digest": self.catalog_digest,
                "official_and_version_space_families": list(CANDIDATE_RULE_FAMILIES),
                "supported_identity_count": self.supported_catalog_identity_count,
                "supported_catalog_digest": self.supported_catalog_digest,
                "excluded_family": "excluded_placard_composition",
                "excluded_identity_count": self.excluded_catalog_identity_count,
                "excluded_catalog_digest": self.excluded_catalog_digest,
                "unsupported_identity_eligible_count": 0,
                "mixed_placard_compositions_allowed_as_official": False,
                "mixed_placard_compositions_allowed_in_v0": False,
                "unknown_or_unsupported_eligibility_policy": "fail_closed",
            },
            "identity_reuse_contract": {
                "placard_identity_reuse": ("permitted_across_scene_disjoint_role_cover_blocks"),
                "one_literal_piece_identity_reuse": ("permitted_across_scene_disjoint_role_cover_blocks"),
                "composed_primary_identity_reuse": "forbidden",
                "composed_primary_stage_overlap": "forbidden_by_total_stage_partition",
                "partner_same_partition_required": False,
                "stage_partition_applies_to": "composed_primary_identity",
            },
            "triple_population": {
                "ordering": (
                    "ascending catalog index: placard outer, one-literal piece middle, composed inner"
                ),
                "joint_cell_order_c_p_q": list(JOINT_CELL_ORDER),
                "row_digest_encoding": (
                    "domain || P(index_u32,digest32) || Q(index_u32,digest32) || "
                    "C(index_u32,digest32) || eight_count_u16_be"
                ),
                "cartesian_triple_count": self.cartesian_triple_count,
                "extensionally_distinct_triple_count": (self.extensionally_distinct_triple_count),
                "triple_identity_table_digest": self.triple_identity_table_digest,
                "exact_joint_cell_table_digest": self.exact_joint_cell_table_digest,
                "unique_joint_cell_vector_count": self.unique_joint_cell_vector_count,
                "joint_cell_vector_histogram_digest": (self.joint_cell_vector_histogram_digest),
                "distinct_minimum_cell_count_count": (self.distinct_minimum_cell_count_count),
                "minimum_cell_count": self.minimum_cell_count,
                "maximum_minimum_cell_count": self.maximum_minimum_cell_count,
                "minimum_cell_count_histogram_digest": (self.minimum_cell_count_histogram_digest),
                "below_conservative_threshold_histogram": [
                    {"minimum_cell_count": minimum, "triple_count": count}
                    for minimum, count in self.below_conservative_threshold_histogram
                ],
            },
            "performance_accounting": {
                "triple_rows_materialized": 0,
                "triple_rows_streamed": self.extensionally_distinct_triple_count,
                "exact_joint_cell_counts_computed": (8 * self.extensionally_distinct_triple_count),
                "composed_partner_counter_slots": (
                    sum(
                        row.identity_count
                        for row in self.family_counts
                        if row.family == "composed_two_literal_piece"
                    )
                    * len(RESERVE_THRESHOLDS)
                ),
                "wall_clock_in_digest": False,
            },
            "reserve_thresholds": [row.as_obj() for row in self.reserve_capacities],
            "engineering_leakage_capacity": {
                "stage_partition": "engineering",
                "reserve_per_joint_cell": 32,
                "required_unique_composed_targets": (ENGINEERING_LEAKAGE_REQUIRED_TARGETS),
                "observed_unique_composed_targets_with_partner": (
                    self.engineering_leakage_observed_targets_at_32
                ),
                "population_capacity_passed": (self.engineering_leakage_capacity_passed_at_32),
                "claim_boundary": (
                    "This is only the rule-population prerequisite for the grouped metadata leakage audit."
                ),
            },
            "claim_boundary": {
                "exhaustive_rule_population_and_cell_counts": True,
                "matched_openings_or_difficulty_balance": False,
                "eleven_disjoint_challenge_panels": False,
                "version_space_coverage": False,
                "episode_bank_feasibility": False,
                "scientific_authorization": False,
            },
            "authorization": dict(_AUTHORIZATION),
        }


def _dependency_manifest(
    *,
    source_fingerprint: str,
    catalog_digest: str,
    stage_partition_digest: str,
) -> dict[str, Any]:
    return {
        "interactive_source_provenance_schema_version": (INTERACTIVE_SOURCE_PROVENANCE_SCHEMA_VERSION),
        "interactive_source_fingerprint": source_fingerprint,
        "scene_schema_version": SCENE_SCHEMA_VERSION,
        "scene_count": SCENE_COUNT,
        "rule_schema_version": RULE_SCHEMA_VERSION,
        "truth_vector_schema_version": TRUTH_VECTOR_SCHEMA_VERSION,
        "rule_catalog_schema_version": RULE_CATALOG_SCHEMA_VERSION,
        "catalog_digest": catalog_digest,
        "stage_partition_schema_version": RULE_PARTITION_SCHEMA_VERSION_V2,
        "stage_partition_digest": stage_partition_digest,
    }


@lru_cache(maxsize=1)
def build_population_audit_v2() -> PopulationAuditV2:
    """Recompute the complete G03-v2 rule-triple population audit."""

    catalog = build_rule_catalog()
    partitions = build_rule_identity_partitions_v2()
    provenance = interactive_source_provenance()

    family_counts_counter: Counter[CatalogAuditFamilyV2] = Counter()
    family_digest = _new_hasher(_FAMILY_IDENTITY_DOMAIN)
    supported_digest = _new_hasher(_SUPPORTED_CATALOG_DOMAIN)
    supported_digest.update(bytes.fromhex(catalog.digest))
    excluded_digest = _new_hasher(_EXCLUDED_CATALOG_DOMAIN)
    excluded_digest.update(bytes.fromhex(catalog.digest))
    pools: dict[CandidateRuleFamilyV2, list[CatalogEntry]] = {
        family: [] for family in CANDIDATE_RULE_FAMILIES
    }
    for entry in catalog:
        family = classify_catalog_identity_v2(entry)
        family_counts_counter[family] += 1
        family_digest.update(_identity_bytes(entry))
        family_digest.update(_FAMILY_CODES[family])
        if family in CANDIDATE_RULE_FAMILIES:
            supported_family = require_supported_catalog_identity_v2(entry)
            pools[supported_family].append(entry)
            supported_digest.update(_identity_bytes(entry))
            supported_digest.update(_FAMILY_CODES[supported_family])
        elif family == "excluded_placard_composition":
            excluded_digest.update(_identity_bytes(entry))
            excluded_digest.update(_FAMILY_CODES[family])
        else:  # pragma: no cover - closed Literal type plus exhaustive classifier
            raise PopulationAuditV2Error("an unknown catalog family reached eligibility")
    family_identity_table_digest = _finalize_counted_digest(family_digest, len(catalog))

    placards = tuple(pools["placard_literal"])
    literals = tuple(pools["one_literal_piece"])
    composed = tuple(pools["composed_two_literal_piece"])
    supported_count = sum(len(pool) for pool in pools.values())
    excluded_count = family_counts_counter["excluded_placard_composition"]
    if supported_count + excluded_count != len(catalog):
        raise PopulationAuditV2Error("v2 supported/excluded catalog accounting is incomplete")
    supported_catalog_digest = _finalize_counted_digest(supported_digest, supported_count)
    excluded_catalog_digest = _finalize_counted_digest(excluded_digest, excluded_count)
    contract = build_supported_catalog_contract_v2()
    if (
        contract.source_catalog_digest != catalog.digest
        or contract.supported_indices
        != tuple(sorted(entry.index for family in CANDIDATE_RULE_FAMILIES for entry in pools[family]))
        or contract.supported_catalog_digest != supported_catalog_digest
        or contract.excluded_catalog_digest != excluded_catalog_digest
    ):
        raise PopulationAuditV2Error("population audit and supported-catalog contract disagree")
    cartesian_count = len(placards) * len(literals) * len(composed)

    composed_metadata: list[tuple[CatalogEntry, RulePartitionV2, BinaryOp, int]] = []
    base_stratum_counts: Counter[tuple[RulePartitionV2, BinaryOp, int]] = Counter()
    for entry in composed:
        stage = partitions.for_entry(entry)
        op, negated_count = composed_stratum_v2(entry)
        composed_metadata.append((entry, stage, op, negated_count))
        base_stratum_counts[(stage, op, negated_count)] += 1

    identity_table_digest = _new_hasher(_TRIPLE_IDENTITY_DOMAIN)
    cell_table_digest = _new_hasher(_TRIPLE_TABLE_DOMAIN)
    eligible_digests = {
        threshold: _new_hasher(_ELIGIBLE_TABLE_DOMAIN_PREFIX + threshold.to_bytes(2, "big") + b"\0")
        for threshold in RESERVE_THRESHOLDS
    }
    partner_counts = {threshold: [0] * len(composed) for threshold in RESERVE_THRESHOLDS}
    eligible_stratum_counts: Counter[tuple[int, RulePartitionV2, BinaryOp, int]] = Counter()
    cell_vector_histogram: Counter[tuple[int, int, int, int, int, int, int, int]] = Counter()
    minimum_histogram: Counter[int] = Counter()
    distinct_count = 0

    for placard in placards:
        for literal in literals:
            for composed_position, (target, stage, op, negated_count) in enumerate(composed_metadata):
                if len({placard.truth.bits, literal.truth.bits, target.truth.bits}) != 3:
                    continue
                cells = _joint_truth_cell_counts_unchecked(placard, literal, target)
                if sum(cells) != SCENE_COUNT:
                    raise PopulationAuditV2Error(
                        "computed joint truth cells do not partition the scene universe"
                    )
                row_payload = _triple_cell_row_payload(placard, literal, target, cells)
                row_digest = _new_hasher(_TRIPLE_ROW_DOMAIN)
                row_digest.update(row_payload)
                row_digest_bytes = row_digest.digest()

                identity_table_digest.update(
                    _identity_bytes(placard) + _identity_bytes(literal) + _identity_bytes(target)
                )
                cell_table_digest.update(row_digest_bytes)
                cell_vector_histogram[cells] += 1
                minimum = min(cells)
                minimum_histogram[minimum] += 1
                distinct_count += 1

                for threshold in RESERVE_THRESHOLDS:
                    if minimum < threshold:
                        continue
                    eligible_digests[threshold].update(row_digest_bytes)
                    partner_counts[threshold][composed_position] += 1
                    eligible_stratum_counts[(threshold, stage, op, negated_count)] += 1

    triple_identity_table_digest = _finalize_counted_digest(identity_table_digest, distinct_count)
    exact_joint_cell_table_digest = _finalize_counted_digest(cell_table_digest, distinct_count)
    vector_histogram_rows = [
        {"cells_c_p_q": list(cells), "triple_count": count}
        for cells, count in sorted(cell_vector_histogram.items())
    ]
    minimum_histogram_rows = [
        {"minimum_cell_count": minimum, "triple_count": count}
        for minimum, count in sorted(minimum_histogram.items())
    ]

    reserve_rows: list[ReserveCapacityV2] = []
    for threshold in RESERVE_THRESHOLDS:
        counts = partner_counts[threshold]
        eligible_count = sum(counts)
        eligible_target_positions = tuple(position for position, count in enumerate(counts) if count > 0)
        target_digest = _new_hasher(_ELIGIBLE_TARGET_DOMAIN_PREFIX + threshold.to_bytes(2, "big") + b"\0")
        partner_digest = _new_hasher(_PARTNER_TABLE_DOMAIN_PREFIX + threshold.to_bytes(2, "big") + b"\0")
        for position, (entry, _stage, _op, _negated_count) in enumerate(composed_metadata):
            partner_digest.update(_identity_bytes(entry))
            partner_digest.update(counts[position].to_bytes(2, "big"))
            if counts[position] > 0:
                target_digest.update(_identity_bytes(entry))
        partner_histogram = Counter(counts)

        by_stage: list[StageCapacityV2] = []
        by_stratum: list[StageStratumCapacityV2] = []
        for stage in RULE_PARTITIONS_V2:
            stage_positions = tuple(
                position
                for position, (_entry, candidate_stage, _op, _negated) in enumerate(composed_metadata)
                if candidate_stage == stage
            )
            by_stage.append(
                StageCapacityV2(
                    stage,
                    len(stage_positions),
                    sum(counts[position] > 0 for position in stage_positions),
                    sum(counts[position] for position in stage_positions),
                )
            )
            for op, negated_count in COMPOSED_STRATA:
                positions = tuple(
                    position
                    for position, (
                        _entry,
                        candidate_stage,
                        candidate_op,
                        candidate_negated,
                    ) in enumerate(composed_metadata)
                    if (
                        candidate_stage,
                        candidate_op,
                        candidate_negated,
                    )
                    == (stage, op, negated_count)
                )
                by_stratum.append(
                    StageStratumCapacityV2(
                        stage,
                        op,
                        negated_count,
                        base_stratum_counts[(stage, op, negated_count)],
                        sum(counts[position] > 0 for position in positions),
                        eligible_stratum_counts[(threshold, stage, op, negated_count)],
                    )
                )

        reserve_rows.append(
            ReserveCapacityV2(
                threshold,
                _THRESHOLD_RATIONALES[threshold],
                False,
                eligible_count,
                len(eligible_target_positions),
                _finalize_counted_digest(eligible_digests[threshold], eligible_count),
                _finalize_counted_digest(target_digest, len(eligible_target_positions)),
                _finalize_counted_digest(partner_digest, len(composed_metadata)),
                tuple(
                    PartnerCountFrequencyV2(partner_count, target_count)
                    for partner_count, target_count in sorted(partner_histogram.items())
                ),
                tuple(by_stage),
                tuple(by_stratum),
            )
        )

    reserve_32 = reserve_rows[RESERVE_THRESHOLDS.index(32)]
    engineering_32 = next(
        row for row in reserve_32.by_stage_partition if row.stage_partition == "engineering"
    )
    family_rows = tuple(
        FamilyCountV2(family, family_counts_counter[family]) for family in CATALOG_AUDIT_FAMILIES
    )
    dependency_manifest = _dependency_manifest(
        source_fingerprint=provenance.fingerprint,
        catalog_digest=catalog.digest,
        stage_partition_digest=partitions.digest,
    )
    below_32 = tuple((minimum, count) for minimum, count in sorted(minimum_histogram.items()) if minimum < 32)
    return PopulationAuditV2(
        provenance.fingerprint,
        catalog.digest,
        partitions.digest,
        _json_digest(dependency_manifest, domain=_DEPENDENCY_DOMAIN),
        family_rows,
        family_identity_table_digest,
        supported_count,
        supported_catalog_digest,
        excluded_count,
        excluded_catalog_digest,
        cartesian_count,
        distinct_count,
        triple_identity_table_digest,
        exact_joint_cell_table_digest,
        len(cell_vector_histogram),
        _json_digest(vector_histogram_rows, domain=_CELL_VECTOR_HISTOGRAM_DOMAIN),
        len(minimum_histogram),
        min(minimum_histogram),
        max(minimum_histogram),
        _json_digest(minimum_histogram_rows, domain=_MINIMUM_HISTOGRAM_DOMAIN),
        below_32,
        tuple(reserve_rows),
        engineering_32.eligible_composed_target_count,
        (engineering_32.eligible_composed_target_count >= ENGINEERING_LEAKAGE_REQUIRED_TARGETS),
    )


def verify_population_audit_v2(report: PopulationAuditV2) -> PopulationAuditV2:
    """Verify every report field by deterministic exhaustive recomputation."""

    if type(report) is not PopulationAuditV2:
        raise TypeError("verify_population_audit_v2 requires a PopulationAuditV2")
    expected = build_population_audit_v2()
    if report != expected:
        raise PopulationAuditV2Error("population audit differs from exhaustive deterministic recomputation")
    return expected


def serialize_population_audit_v2(report: PopulationAuditV2) -> str:
    if type(report) is not PopulationAuditV2:
        raise TypeError("serialize_population_audit_v2 requires a PopulationAuditV2")
    verify_population_audit_v2(report)
    return _dump_json(report.as_obj())


def parse_population_audit_v2(
    text: str,
    *,
    require_canonical: bool = True,
    expected_digest: str | None = None,
) -> PopulationAuditV2:
    """Parse only the one report reproduced by the current exact dependencies."""

    value = _load_json(text)
    expected = build_population_audit_v2()
    expected_text = _dump_json(expected.as_obj())
    if _dump_json(value) != expected_text:
        raise PopulationAuditV2Error("population audit differs from exhaustive deterministic recomputation")
    if require_canonical and expected_text != text:
        raise PopulationAuditV2Error("population audit JSON is not canonical compact JSON")
    if expected_digest is not None:
        if not _is_sha256(expected_digest):
            raise PopulationAuditV2Error("expected report digest is malformed")
        if expected.digest != expected_digest:
            raise PopulationAuditV2Error("population audit digest does not match expected digest")
    return expected
