"""Canonical, non-authorizing G03-E engineering audit reports.

This module audits the small deterministic engine fixture.  It can establish
that the local implementation and fixture artifacts are internally
reproducible; it can never authorize model weight updates or relabel the
engineering fixture as a pilot or confirmatory study bank.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from ._json import CanonicalJSONError, dump_json, json_digest, load_json
from .action_language import (
    action_language_digest,
    action_language_manifest,
    build_answer_action,
    build_inquiry_action,
)
from .actions import AnswerAction, TestAction, parse_action, serialize_action
from .catalog import (
    MAX_TRUE_COUNT,
    MIN_TRUE_COUNT,
    RuleCatalog,
    build_rule_catalog,
    truth_vector,
    verify_truth_vector,
)
from .dialogue import dialogue_as_obj, dialogue_digest, render_dialogue
from .environment import play_reference_episode, replay_transcript
from .episodes import terminal_classifications
from .generation import (
    EpisodeBank,
    generate_episode_bank,
    parse_episode_bank,
    serialize_episode_bank,
    small_fixture_bank_spec,
    verify_episode_bank,
)
from .interventions import (
    InterventionBank,
    generate_intervention_bank,
    parse_intervention_bank,
    serialize_intervention_bank,
    verify_intervention_bank,
)
from .leakage import audit_episode_bank_surface_leakage
from .partitions import (
    build_eligible_pair_table,
    build_rule_identity_partitions,
    parse_eligible_pair_table,
    parse_rule_identity_partitions,
    serialize_eligible_pair_table,
    serialize_rule_identity_partitions,
)
from .provenance import interactive_source_provenance
from .schema import (
    SCENE_COUNT,
    parse_scene,
    scene_at,
    scene_index,
    serialize_scene,
)
from .transcripts import AnswerEvent

EngineSemanticMode = Literal["derived", "exhaustive"]
EngineCheckStatus = Literal["pass", "failed", "insufficient_data", "not_run"]
LeakageStatus = Literal["pass", "failed", "insufficient_data", "missing"]

ENGINE_AUDIT_SCHEMA_VERSION = 1
ENGINE_AUDIT_REPORT_ID = "g03-e-engineering-fixture-audit-v1"
ENGINE_AUDIT_KIND = "non_authorizing_engineering_fixture"
EPISODE_FIXTURE_PATH = "tests/goalzendo_interactive/fixtures/g03-engine-small-fixture-v1.json"
INTERVENTION_FIXTURE_PATH = (
    "tests/goalzendo_interactive/fixtures/g03-engine-small-interventions-v1.json"
)
ENGINE_CHECK_IDS = (
    "scene_universe_roundtrip",
    "canonical_test_action_roundtrip",
    "rule_catalog_semantics",
    "rule_identity_partitions",
    "eligible_target_shadow_pairs",
    "episode_fixture_manifest",
    "intervention_fixture_manifest",
    "reference_expert_trajectories",
    "public_dialogue_action_language",
    "interactive_source_provenance",
    "surface_leakage",
)

_CHECK_EVIDENCE_DOMAIN = "goalzendo-interactive-engine-audit-check-v1"
_CHECK_BUNDLE_DOMAIN = "goalzendo-interactive-engine-audit-check-bundle-v1"
_REPORT_DOMAIN = "goalzendo-interactive-engine-audit-report-v1"


class EngineAuditError(RuntimeError):
    """Raised when an engineering invariant or artifact fails closed."""


class EngineAuditValidationError(ValueError):
    """Raised when a serialized audit report is malformed or tampered."""


class WeightUpdateAuthorizationError(PermissionError):
    """Raised whenever this engineering report is used as an update gate."""


def _valid_digest(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_evidence(value: Any) -> str:
    if type(value) is not dict:
        raise EngineAuditValidationError("check evidence must be a JSON object")
    try:
        text = dump_json(value)
        parsed = load_json(text)
    except CanonicalJSONError as exc:
        raise EngineAuditValidationError(str(exc)) from exc
    if parsed != value:
        raise EngineAuditValidationError("check evidence is not stable canonical JSON")
    return text


@dataclass(frozen=True, slots=True)
class EngineAuditCheck:
    """One immutable claim and the canonical evidence that supports it."""

    check_id: str
    status: EngineCheckStatus
    exhaustive: bool
    item_count: int
    claim: str
    evidence_json: str

    def __post_init__(self) -> None:
        if type(self.check_id) is not str or not self.check_id or not self.check_id.isascii():
            raise EngineAuditValidationError("check id must be nonempty ASCII")
        if self.status not in {"pass", "failed", "insufficient_data", "not_run"}:
            raise EngineAuditValidationError(f"unknown engine-check status: {self.status!r}")
        if type(self.exhaustive) is not bool:
            raise EngineAuditValidationError("check exhaustive flag must be Boolean")
        if (
            isinstance(self.item_count, bool)
            or not isinstance(self.item_count, int)
            or self.item_count < 0
        ):
            raise EngineAuditValidationError("check item count must be a non-negative integer")
        if type(self.claim) is not str or not self.claim or not self.claim.isascii():
            raise EngineAuditValidationError("check claim must be nonempty ASCII")
        if type(self.evidence_json) is not str:
            raise EngineAuditValidationError("check evidence JSON must be a string")
        try:
            evidence = load_json(self.evidence_json)
        except CanonicalJSONError as exc:
            raise EngineAuditValidationError(str(exc)) from exc
        if type(evidence) is not dict or dump_json(evidence) != self.evidence_json:
            raise EngineAuditValidationError("check evidence JSON is not canonical")

    @classmethod
    def create(
        cls,
        check_id: str,
        *,
        status: EngineCheckStatus,
        exhaustive: bool,
        item_count: int,
        claim: str,
        evidence: dict[str, Any],
    ) -> EngineAuditCheck:
        return cls(
            check_id,
            status,
            exhaustive,
            item_count,
            claim,
            _canonical_evidence(evidence),
        )

    @property
    def evidence(self) -> dict[str, Any]:
        return cast(dict[str, Any], load_json(self.evidence_json))

    @property
    def evidence_digest(self) -> str:
        return json_digest(
            {"check_id": self.check_id, "evidence": self.evidence},
            domain=_CHECK_EVIDENCE_DOMAIN,
        )

    def as_obj(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "status": self.status,
            "exhaustive": self.exhaustive,
            "item_count": self.item_count,
            "claim": self.claim,
            "evidence_digest": self.evidence_digest,
            "evidence": self.evidence,
        }


def engine_audit_check_from_obj(value: Any) -> EngineAuditCheck:
    expected = {
        "check_id",
        "status",
        "exhaustive",
        "item_count",
        "claim",
        "evidence_digest",
        "evidence",
    }
    if type(value) is not dict or set(value) != expected or len(value) != len(expected):
        raise EngineAuditValidationError("engine audit check has noncanonical fields")
    result = EngineAuditCheck.create(
        value["check_id"],
        status=cast(EngineCheckStatus, value["status"]),
        exhaustive=value["exhaustive"],
        item_count=value["item_count"],
        claim=value["claim"],
        evidence=value["evidence"],
    )
    if result.evidence_digest != value["evidence_digest"] or result.as_obj() != value:
        raise EngineAuditValidationError("engine audit check evidence digest mismatch")
    return result


def _authorization_reasons(
    semantic_mode: EngineSemanticMode,
    leakage_status: LeakageStatus,
) -> tuple[str, ...]:
    reasons = ["engineering_fixture_non_authorizing"]
    if semantic_mode != "exhaustive":
        reasons.append("catalog_semantics_not_independently_exhausted")
    if leakage_status == "missing":
        reasons.append("surface_leakage_audit_missing")
    elif leakage_status == "insufficient_data":
        reasons.append("surface_leakage_audit_underpowered")
    elif leakage_status == "failed":
        reasons.append("surface_leakage_audit_failed")
    return tuple(reasons)


@dataclass(frozen=True, slots=True)
class EngineAuditReport:
    """A deterministic report whose authorization boundary is always closed."""

    semantic_mode: EngineSemanticMode
    source_fingerprint: str
    checks: tuple[EngineAuditCheck, ...]

    def __post_init__(self) -> None:
        if self.semantic_mode not in {"derived", "exhaustive"}:
            raise EngineAuditValidationError(
                f"unknown engine semantic mode: {self.semantic_mode!r}"
            )
        if not _valid_digest(self.source_fingerprint):
            raise EngineAuditValidationError("source fingerprint must be a SHA-256 digest")
        checks = tuple(self.checks)
        object.__setattr__(self, "checks", checks)
        if any(type(check) is not EngineAuditCheck for check in checks):
            raise EngineAuditValidationError("engine audit report contains a non-check")
        if tuple(check.check_id for check in checks) != ENGINE_CHECK_IDS:
            raise EngineAuditValidationError("engine audit checks are missing or out of order")
        source_check = self.check("interactive_source_provenance")
        if source_check.evidence.get("fingerprint") != self.source_fingerprint:
            raise EngineAuditValidationError("source check and report fingerprint disagree")
        catalog_check = self.check("rule_catalog_semantics")
        expected_exhaustive = self.semantic_mode == "exhaustive"
        if catalog_check.exhaustive is not expected_exhaustive:
            raise EngineAuditValidationError("semantic mode and catalog check disagree")
        leakage_check = self.check("surface_leakage")
        leakage_evidence = leakage_check.evidence
        if set(leakage_evidence) != {"leakage_status", "report_digest", "report"}:
            raise EngineAuditValidationError("surface-leakage evidence has noncanonical fields")
        leakage_report = leakage_evidence["report"]
        if type(leakage_report) is not dict:
            raise EngineAuditValidationError("surface-leakage report evidence must be an object")
        expected_report_digest = json_digest(
            leakage_report,
            domain="goalzendo-interactive-engine-audit-surface-leakage-v1",
        )
        if leakage_evidence["report_digest"] != expected_report_digest:
            raise EngineAuditValidationError("surface-leakage report digest mismatch")
        stated_status = leakage_evidence["leakage_status"]
        expected_status: LeakageStatus = (
            "missing" if stated_status == "missing" and not leakage_report
            else _leakage_status(leakage_report)
        )
        if stated_status != expected_status:
            raise EngineAuditValidationError("surface-leakage status is inconsistent")

    def check(self, check_id: str) -> EngineAuditCheck:
        for check in self.checks:
            if check.check_id == check_id:
                return check
        raise KeyError(check_id)

    @property
    def leakage_status(self) -> LeakageStatus:
        status = self.check("surface_leakage").evidence.get("leakage_status")
        if status not in {"pass", "failed", "insufficient_data", "missing"}:
            raise EngineAuditValidationError("surface-leakage check lacks a valid status")
        return cast(LeakageStatus, status)

    @property
    def core_checks_passed(self) -> bool:
        return all(
            check.status == "pass"
            for check in self.checks
            if check.check_id != "surface_leakage"
        )

    @property
    def all_checks_passed(self) -> bool:
        return all(check.status == "pass" for check in self.checks)

    @property
    def weight_updates_authorized(self) -> bool:
        return False

    @property
    def authorization_reasons(self) -> tuple[str, ...]:
        return _authorization_reasons(self.semantic_mode, self.leakage_status)

    @property
    def check_evidence_bundle_digest(self) -> str:
        return json_digest(
            [check.as_obj() for check in self.checks],
            domain=_CHECK_BUNDLE_DOMAIN,
        )

    def as_obj(self) -> dict[str, Any]:
        return {
            "schema_version": ENGINE_AUDIT_SCHEMA_VERSION,
            "report_id": ENGINE_AUDIT_REPORT_ID,
            "audit_kind": ENGINE_AUDIT_KIND,
            "semantic_mode": self.semantic_mode,
            "source_fingerprint": self.source_fingerprint,
            "checks": [check.as_obj() for check in self.checks],
            "check_evidence_bundle_digest": self.check_evidence_bundle_digest,
            "core_checks_passed": self.core_checks_passed,
            "all_checks_passed": self.all_checks_passed,
            "surface_leakage_status": self.leakage_status,
            "authorization": {
                "scope": "none",
                "weight_updates_authorized": False,
                "reasons": list(self.authorization_reasons),
                "boundary": (
                    "This 12-episode G03-E engineering fixture can never authorize "
                    "model weight updates."
                ),
            },
        }

    @property
    def digest(self) -> str:
        return json_digest(self.as_obj(), domain=_REPORT_DOMAIN)


def engine_audit_report_from_obj(value: Any) -> EngineAuditReport:
    expected = {
        "schema_version",
        "report_id",
        "audit_kind",
        "semantic_mode",
        "source_fingerprint",
        "checks",
        "check_evidence_bundle_digest",
        "core_checks_passed",
        "all_checks_passed",
        "surface_leakage_status",
        "authorization",
    }
    if type(value) is not dict or set(value) != expected or len(value) != len(expected):
        raise EngineAuditValidationError("engine audit report has noncanonical fields")
    if value["schema_version"] != ENGINE_AUDIT_SCHEMA_VERSION:
        raise EngineAuditValidationError("unsupported engine audit schema version")
    if value["report_id"] != ENGINE_AUDIT_REPORT_ID or value["audit_kind"] != ENGINE_AUDIT_KIND:
        raise EngineAuditValidationError("engine audit identity mismatch")
    if type(value["checks"]) is not list:
        raise EngineAuditValidationError("engine audit checks must be an array")
    result = EngineAuditReport(
        cast(EngineSemanticMode, value["semantic_mode"]),
        value["source_fingerprint"],
        tuple(engine_audit_check_from_obj(item) for item in value["checks"]),
    )
    if result.as_obj() != value:
        raise EngineAuditValidationError("engine audit derived fields are inconsistent")
    return result


def serialize_engine_audit_report(report: EngineAuditReport) -> str:
    if type(report) is not EngineAuditReport:
        raise TypeError("serialize_engine_audit_report requires an EngineAuditReport")
    return dump_json(report.as_obj())


def parse_engine_audit_report(
    text: str,
    *,
    require_canonical: bool = True,
) -> EngineAuditReport:
    try:
        value = load_json(text)
    except CanonicalJSONError as exc:
        raise EngineAuditValidationError(str(exc)) from exc
    result = engine_audit_report_from_obj(value)
    if require_canonical and serialize_engine_audit_report(result) != text:
        raise EngineAuditValidationError("engine audit JSON is valid but not canonical")
    return result


def require_weight_update_authorization(report: EngineAuditReport) -> None:
    """Always reject: G03-E engineering evidence is non-authorizing by design."""

    if type(report) is not EngineAuditReport:
        raise TypeError("report must be an EngineAuditReport")
    raise WeightUpdateAuthorizationError(
        "G03-E engineering audit cannot authorize weight updates: "
        + ",".join(report.authorization_reasons)
    )


def _rolling_digest(domain: bytes, values: list[bytes]) -> str:
    digest = hashlib.sha256(domain)
    for value in values:
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return digest.hexdigest()


def _scene_and_action_checks() -> tuple[EngineAuditCheck, EngineAuditCheck]:
    scene_payloads: list[bytes] = []
    action_payloads: list[bytes] = []
    seen_scenes: set[str] = set()
    seen_actions: set[str] = set()
    for index in range(SCENE_COUNT):
        scene = scene_at(index)
        encoded_scene = serialize_scene(scene)
        if scene_index(scene) != index or parse_scene(encoded_scene) != scene:
            raise EngineAuditError(f"scene round trip failed at index {index}")
        seen_scenes.add(encoded_scene)
        scene_payloads.append(encoded_scene.encode("ascii"))

        action = TestAction(scene)
        encoded_action = serialize_action(action)
        parsed = parse_action(encoded_action, expected_move="test")
        state = build_inquiry_action(action)
        if parsed != action or state.action != action or state.text != encoded_action:
            raise EngineAuditError(f"test-action round trip failed at scene index {index}")
        seen_actions.add(encoded_action)
        action_payloads.append(encoded_action.encode("ascii"))
    if len(seen_scenes) != SCENE_COUNT or len(seen_actions) != SCENE_COUNT:
        raise EngineAuditError("scene or test-action universe contains duplicates")
    scene_check = EngineAuditCheck.create(
        "scene_universe_roundtrip",
        status="pass",
        exhaustive=True,
        item_count=SCENE_COUNT,
        claim="Every canonical scene round-trips through index and JSON bijections.",
        evidence={
            "scene_count": SCENE_COUNT,
            "unique_scene_count": len(seen_scenes),
            "ordered_serialization_digest": _rolling_digest(
                b"goalzendo-engine-audit-scenes-v1\0", scene_payloads
            ),
        },
    )
    action_check = EngineAuditCheck.create(
        "canonical_test_action_roundtrip",
        status="pass",
        exhaustive=True,
        item_count=SCENE_COUNT,
        claim=(
            "Every canonical test action round-trips through parser and field-factorized grammar."
        ),
        evidence={
            "test_action_count": SCENE_COUNT,
            "unique_test_action_count": len(seen_actions),
            "ordered_serialization_digest": _rolling_digest(
                b"goalzendo-engine-audit-test-actions-v1\0", action_payloads
            ),
            "action_language_digest": action_language_digest(),
        },
    )
    return scene_check, action_check


def _catalog_check(mode: EngineSemanticMode) -> tuple[RuleCatalog, EngineAuditCheck]:
    if mode not in {"derived", "exhaustive"}:
        raise ValueError(f"unknown semantic mode: {mode!r}")
    catalog = build_rule_catalog()
    bits = {entry.truth.bits for entry in catalog}
    digests = {entry.truth_digest for entry in catalog}
    if len(bits) != len(catalog) or len(digests) != len(catalog):
        raise EngineAuditError("catalog truth-vector identities are not unique")
    if any(
        entry.index != index
        or entry.rule_id != f"g03r{index:05d}"
        or not MIN_TRUE_COUNT <= entry.truth.true_count <= MAX_TRUE_COUNT
        or truth_vector(entry.rule).bits != entry.truth.bits
        for index, entry in enumerate(catalog)
    ):
        raise EngineAuditError("catalog derived consistency check failed")

    independent_count = 0
    if mode == "exhaustive":
        for entry in catalog:
            if not verify_truth_vector(entry.rule, entry.truth):
                raise EngineAuditError(
                    f"independent semantic verification failed for {entry.rule_id}"
                )
            independent_count += 1
    verification_level = (
        "independent_full_universe_evaluation"
        if mode == "exhaustive"
        else "derived_internal_consistency_only"
    )
    return catalog, EngineAuditCheck.create(
        "rule_catalog_semantics",
        status="pass",
        exhaustive=mode == "exhaustive",
        item_count=len(catalog),
        claim=(
            "Catalog identities and prevalence bounds hold; independent semantics are claimed "
            "only in exhaustive mode."
        ),
        evidence={
            "catalog_digest": catalog.digest,
            "catalog_stats": catalog.stats.as_obj(),
            "retained_rule_count": len(catalog),
            "unique_truth_vector_count": len(bits),
            "unique_truth_digest_count": len(digests),
            "verification_level": verification_level,
            "independent_direct_evaluation_count": independent_count,
            "independent_scene_evaluations": independent_count * SCENE_COUNT,
        },
    )


def _partition_and_pair_checks() -> tuple[EngineAuditCheck, EngineAuditCheck]:
    partitions = build_rule_identity_partitions()
    encoded_partitions = serialize_rule_identity_partitions(partitions)
    if parse_rule_identity_partitions(encoded_partitions) is not partitions:
        raise EngineAuditError("partition manifest did not round-trip to its canonical mapping")
    partition_check = EngineAuditCheck.create(
        "rule_identity_partitions",
        status="pass",
        exhaustive=True,
        item_count=len(partitions.assignments),
        claim="Every retained truth-vector identity occurs in exactly one registered partition.",
        evidence={
            "partition_digest": partitions.digest,
            "catalog_digest": partitions.catalog_digest,
            "assignment_count": len(partitions.assignments),
            "counts": partitions.counts,
            "manifest_sha256": hashlib.sha256(encoded_partitions.encode("ascii")).hexdigest(),
        },
    )

    pairs = build_eligible_pair_table()
    encoded_pairs = serialize_eligible_pair_table(pairs)
    if parse_eligible_pair_table(encoded_pairs).digest != pairs.digest:
        raise EngineAuditError("eligible-pair table did not round-trip canonically")
    pair_check = EngineAuditCheck.create(
        "eligible_target_shadow_pairs",
        status="pass",
        exhaustive=True,
        item_count=len(pairs.pairs),
        claim="Every registered eligible target-shadow pair satisfies the exact table constructor.",
        evidence={
            "eligible_pair_table_digest": pairs.digest,
            "catalog_digest": pairs.catalog_digest,
            "partitions_digest": pairs.partitions_digest,
            "pair_count": len(pairs.pairs),
            "counts": pairs.counts,
            "manifest_sha256": hashlib.sha256(encoded_pairs.encode("ascii")).hexdigest(),
        },
    )
    return partition_check, pair_check


def _fixture_text(root: Path, relative_path: str) -> tuple[str, str]:
    path = root / relative_path
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise EngineAuditError(f"cannot read required fixture {relative_path}: {exc}") from exc
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise EngineAuditError(f"fixture {relative_path} must end in exactly one newline")
    try:
        text = payload[:-1].decode("ascii")
    except UnicodeDecodeError as exc:
        raise EngineAuditError(f"fixture {relative_path} is not canonical ASCII JSON") from exc
    return text, hashlib.sha256(payload).hexdigest()


def _episode_fixture_check(root: Path) -> tuple[EpisodeBank, EngineAuditCheck]:
    text, file_sha = _fixture_text(root, EPISODE_FIXTURE_PATH)
    bank = parse_episode_bank(text)
    expected = generate_episode_bank(small_fixture_bank_spec())
    if serialize_episode_bank(bank) != serialize_episode_bank(expected):
        raise EngineAuditError("stored episode fixture differs from deterministic regeneration")
    verify_episode_bank(bank)
    return bank, EngineAuditCheck.create(
        "episode_fixture_manifest",
        status="pass",
        exhaustive=True,
        item_count=len(bank.episodes),
        claim="Every stored engineering episode and generation attestation regenerates exactly.",
        evidence={
            "fixture_path": EPISODE_FIXTURE_PATH,
            "fixture_file_sha256": file_sha,
            "episode_bank_id": bank.spec.bank_id,
            "episode_bank_digest": bank.digest,
            "catalog_digest": bank.catalog_digest,
            "partitions_digest": bank.partitions_digest,
            "eligible_pairs_digest": bank.eligible_pairs_digest,
            "episode_count": len(bank.episodes),
            "formula_counts": bank.formula_counts,
            "partition_counts": bank.partition_counts,
        },
    )


def _intervention_fixture_check(
    root: Path,
    episode_bank: EpisodeBank,
) -> tuple[InterventionBank, EngineAuditCheck]:
    text, file_sha = _fixture_text(root, INTERVENTION_FIXTURE_PATH)
    bank = parse_intervention_bank(text, source_episode_bank=episode_bank)
    expected = generate_intervention_bank(episode_bank)
    if serialize_intervention_bank(bank) != serialize_intervention_bank(expected):
        raise EngineAuditError("stored intervention fixture differs from deterministic regeneration")
    verify_intervention_bank(bank)
    return bank, EngineAuditCheck.create(
        "intervention_fixture_manifest",
        status="pass",
        exhaustive=True,
        item_count=len(bank.records),
        claim="Every stored intervention reconstructs and the complete bank regenerates exactly.",
        evidence={
            "fixture_path": INTERVENTION_FIXTURE_PATH,
            "fixture_file_sha256": file_sha,
            "intervention_bank_id": bank.bank_id,
            "intervention_bank_digest": bank.digest,
            "source_episode_bank_digest": episode_bank.digest,
            "record_count": len(bank.records),
            "selected_scene_count": len(bank.records) * 2,
            "family_counts": bank.family_counts,
            "reserved_scene_count": len(bank.reserved_scene_indices),
        },
    )


def _reference_check(bank: EpisodeBank) -> EngineAuditCheck:
    rows: list[dict[str, Any]] = []
    records = {
        record.request_id: record for record in bank.generation_records
    }
    for request, episode in zip(bank.spec.requests, bank.episodes, strict=True):
        transcript = play_reference_episode(episode)
        replayed = replay_transcript(episode, transcript)
        answer_events = [event for event in transcript.events if type(event) is AnswerEvent]
        if len(answer_events) != 1:
            raise EngineAuditError(f"reference episode {episode.episode_id} lacks one answer")
        answer = answer_events[0]
        classifications = terminal_classifications(episode)
        if (
            transcript.state != "complete"
            or replayed.state != "complete"
            or len(replayed.version_space) != 1
            or replayed.version_space.indices != (episode.target.index,)
            or not answer.score.rule_equivalent
            or answer.score.classification_correct != answer.score.classification_total
            or answer.score.classification_total != len(episode.terminal)
            or answer.action.classifications != classifications
            or truth_vector(answer.action.rule).bits != episode.target.truth.bits
            or answer.score.query_count != records[request.request_id].reference_query_count
        ):
            raise EngineAuditError(
                f"reference episode {episode.episode_id} failed exact recovery or classification"
            )
        answer_state = build_answer_action(
            AnswerAction(episode.target.rule, cast(tuple, classifications))
        )
        if answer_state.action != answer.action:
            raise EngineAuditError("reference answer is outside the canonical action language")
        dialogue = render_dialogue(episode, transcript)
        rows.append(
            {
                "episode_id": episode.episode_id,
                "episode_digest": episode.digest,
                "transcript_digest": transcript.digest,
                "dialogue_digest": json_digest(
                    dialogue_as_obj(dialogue),
                    domain="goalzendo-interactive-reference-dialogue-v1",
                ),
                "query_count": answer.score.query_count,
                "terminal_count": answer.score.classification_total,
                "classification_correct": answer.score.classification_correct,
                "rule_equivalent": answer.score.rule_equivalent,
                "identified_rule_id": episode.target.rule_id,
            }
        )
    return EngineAuditCheck.create(
        "reference_expert_trajectories",
        status="pass",
        exhaustive=True,
        item_count=len(rows),
        claim=(
            "Every fixture reference trajectory identifies the exact rule and classifies every "
            "terminal scene correctly."
        ),
        evidence={
            "episode_bank_digest": bank.digest,
            "episode_count": len(rows),
            "all_rule_equivalent": all(row["rule_equivalent"] for row in rows),
            "classification_correct": sum(row["classification_correct"] for row in rows),
            "classification_total": sum(row["terminal_count"] for row in rows),
            "rows": rows,
        },
    )


def _public_language_check() -> EngineAuditCheck:
    manifest = action_language_manifest()
    if (
        manifest["test_action_count"] != SCENE_COUNT
        or manifest["candidate_koan_list_presented_to_model"] is not False
    ):
        raise EngineAuditError("action-language manifest contradicts the active-game contract")
    return EngineAuditCheck.create(
        "public_dialogue_action_language",
        status="pass",
        exhaustive=True,
        item_count=SCENE_COUNT,
        claim="Public dialogue and action grammars are digest-bound and expose all test koans.",
        evidence={
            "dialogue_digest": dialogue_digest(),
            "action_language_digest": action_language_digest(),
            "action_language_manifest": manifest,
        },
    )


def _source_check(root: Path) -> tuple[str, EngineAuditCheck]:
    provenance = interactive_source_provenance(root / "src" / "goalzendo_interactive")
    provenance_obj = provenance.as_obj()
    provenance_digest = json_digest(
        provenance_obj,
        domain="goalzendo-interactive-engine-audit-source-manifest-v1",
    )
    return provenance.fingerprint, EngineAuditCheck.create(
        "interactive_source_provenance",
        status="pass",
        exhaustive=True,
        item_count=len(provenance.files),
        claim="Every persistent file in the separate interactive package is content-addressed.",
        evidence={
            "fingerprint": provenance.fingerprint,
            "file_count": len(provenance.files),
            "total_bytes": provenance_obj["total_bytes"],
            "provenance_manifest_digest": provenance_digest,
            "files": [record.as_obj() for record in provenance.files],
        },
    )


def _leakage_obj(report: object) -> dict[str, Any]:
    if type(report) is dict:
        value = report
    else:
        as_obj = getattr(report, "as_obj", None)
        if not callable(as_obj):
            raise EngineAuditValidationError(
                "surface leakage report must be a canonical mapping or expose as_obj()"
            )
        value = as_obj()
    if type(value) is not dict:
        raise EngineAuditValidationError("surface leakage report must serialize to an object")
    return cast(dict[str, Any], load_json(_canonical_evidence(value)))


def _leakage_status(report_obj: dict[str, Any] | None) -> LeakageStatus:
    if report_obj is None:
        return "missing"
    results = report_obj.get("results")
    if type(results) is list and results:
        decisions = [result.get("decision") for result in results if type(result) is dict]
        if len(decisions) != len(results):
            return "failed"
        if any(decision == "insufficient_data" for decision in decisions):
            return "insufficient_data"
        return "pass" if report_obj.get("passed") is True and all(
            result.get("passed") is True for result in results
        ) else "failed"
    adequately_powered = report_obj.get("adequately_powered")
    if adequately_powered is False:
        return "insufficient_data"
    if report_obj.get("status") == "missing":
        return "missing"
    return "pass" if adequately_powered is True and report_obj.get("passed") is True else "failed"


def _surface_leakage_check(report_obj: dict[str, Any] | None) -> EngineAuditCheck:
    status = _leakage_status(report_obj)
    check_status: EngineCheckStatus
    if status == "pass":
        check_status = "pass"
    elif status == "insufficient_data":
        check_status = "insufficient_data"
    elif status == "missing":
        check_status = "not_run"
    else:
        check_status = "failed"
    evidence_report: dict[str, Any] = {} if report_obj is None else report_obj
    report_digest = json_digest(
        evidence_report,
        domain="goalzendo-interactive-engine-audit-surface-leakage-v1",
    )
    results = evidence_report.get("results", [])
    return EngineAuditCheck.create(
        "surface_leakage",
        status=check_status,
        exhaustive=bool(results) and status != "missing",
        item_count=len(results) if type(results) is list else 0,
        claim=(
            "Surface leakage must be present, adequately powered, and pass every registered target."
        ),
        evidence={
            "leakage_status": status,
            "report_digest": report_digest,
            "report": evidence_report,
        },
    )


def run_engineering_audit(
    repo_root: str | Path,
    *,
    semantic_mode: EngineSemanticMode = "derived",
    surface_leakage_report: object | None = None,
    run_surface_leakage: bool = True,
) -> EngineAuditReport:
    """Run every registered G03-E check and return a canonical report.

    ``derived`` mode verifies catalog construction and identity invariants but
    records zero independent semantic truth-vector evaluations.  ``exhaustive``
    mode independently evaluates all 7,300 retained rules on all 13,716 scenes
    and is intentionally slow.
    """

    root = Path(repo_root).resolve()
    if not root.is_dir():
        raise EngineAuditError(f"repository root is not a directory: {root}")
    if semantic_mode not in {"derived", "exhaustive"}:
        raise ValueError(f"unknown semantic mode: {semantic_mode!r}")
    scene_check, action_check = _scene_and_action_checks()
    _, catalog_check = _catalog_check(semantic_mode)
    partition_check, pair_check = _partition_and_pair_checks()
    episode_bank, episode_check = _episode_fixture_check(root)
    _, intervention_check = _intervention_fixture_check(root, episode_bank)
    reference_check = _reference_check(episode_bank)
    public_check = _public_language_check()
    source_fingerprint, source_check = _source_check(root)

    if surface_leakage_report is not None:
        leakage_obj = _leakage_obj(surface_leakage_report)
    elif run_surface_leakage:
        leakage_obj = _leakage_obj(audit_episode_bank_surface_leakage(episode_bank))
    else:
        leakage_obj = None
    leakage_check = _surface_leakage_check(leakage_obj)
    return EngineAuditReport(
        semantic_mode,
        source_fingerprint,
        (
            scene_check,
            action_check,
            catalog_check,
            partition_check,
            pair_check,
            episode_check,
            intervention_check,
            reference_check,
            public_check,
            source_check,
            leakage_check,
        ),
    )


def verify_engineering_audit(
    report: EngineAuditReport,
    repo_root: str | Path,
) -> EngineAuditReport:
    """Regenerate the exact report against the current repository and fixtures."""

    if type(report) is not EngineAuditReport:
        raise TypeError("verify_engineering_audit requires an EngineAuditReport")
    leakage_check = report.check("surface_leakage")
    leakage_report = leakage_check.evidence["report"]
    regenerated = run_engineering_audit(
        repo_root,
        semantic_mode=report.semantic_mode,
        surface_leakage_report=(
            None if report.leakage_status == "missing" else leakage_report
        ),
        run_surface_leakage=False,
    )
    if serialize_engine_audit_report(regenerated) != serialize_engine_audit_report(report):
        raise EngineAuditValidationError(
            "engine audit does not regenerate byte-for-byte from current dependencies"
        )
    return report
