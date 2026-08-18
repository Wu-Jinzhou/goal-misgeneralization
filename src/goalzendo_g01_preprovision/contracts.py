"""Pure, nonauthorizing contracts before any G01Q provider transaction.

Constructors only normalize caller-supplied, unauthenticated facts into
deterministic review candidates.  They do not register, persist, inspect,
provision, load a model, or confer authority.  Every operational function is
an unconditional refusal until an independent registrar is designated.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, NoReturn, cast

REFUSAL = "G01Q_REGISTRAR_NOT_DESIGNATED"
STUDY_ID = "g01_compute_qualification"
POLICY_TEMPLATE_SCHEMA = "goalzendo.g01q_preprovision_policy_template"
CAMPAIGN_CANDIDATE_SCHEMA = "goalzendo.g01q_unregistered_campaign_intent_candidate"
SCHEMA_VERSION = 1
CAMPAIGN_STATE = "UNREGISTERED"

CHECKPOINT_A_BRIDGE_SOURCE_DIGEST = "e0f3df053557cdf06c028a08f847db33b55dac3a7388a52e0fae0d94b07d284f"
CHECKPOINT_B_REFUSAL_SOURCE_DIGEST = "77d9ba4928fa29cc42a055201660304f2059d0ca6358371cc14618407bdc714b"
CHECKPOINT_B_CAPSULE_SOURCE_DIGEST = "efb6b0427895f60e16794f4e20b87d620522673edffb94f9ce277089b5581782"
G01Q_SOURCE_DIGEST = "efde59caf3830d021223940b2d6e1a8e9631c90f0f8e51262041008ea177c7b3"
G01Q_TARGET_BINDING_DIGEST = "3315d20a6f9bdae3c5fdaf9567c5bce7b592890d0ebd26e10682002d816bf0c6"
GOALZENDO_SOURCE_FINGERPRINT = "1a8146377b9a9620690025671614edb2dd20d214f4e195da3cf3528809b2c694"
G01Q_PRODUCTION_SHAPE_DIGEST = "4a166cd03ace10d1b2fa59b9be5c776bf3caa948027ce465ce4af3c0271bb566"
G01Q_CANDIDATE_POLICY_DIGEST = "084f5d5b976a17357dff39f767973c18bf41bca945826cda57b9ca10f2de2fff"
G01Q_REVIEW_CONTRACT_DIGEST = "1c26c1e1e4255fe2b0619dc9f880598ad67ba60781fe081bb624939f88613a7a"
DEPENDENCY_CONSTRAINTS_SHA256 = "5c26180b1d9ec43e4249174dae98cfecd0187dc81f36ae964affb60e6f22910b"
MODEL_LEAF_MANIFEST_DIGEST = "ff6162df6d6d022e565f916bf04c8902d82c6a82ee176de77ec59209e78a5bbc"

_DATA_CENTER_ID = re.compile(r"[A-Z0-9]+(?:-[A-Z0-9]+){1,7}")
_NETWORK_VOLUME_ID = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9_-]{0,126}[A-Za-z0-9])?")
_SHA256 = re.compile(r"[0-9a-f]{64}")

_CANDIDATE_PROFILES = (
    ("h200x8", "NVIDIA H200", 8, 79_200),
    ("h100-hbm3x8", "NVIDIA H100 80GB HBM3", 8, 79_200),
    ("h200x4", "NVIDIA H200", 4, 151_200),
    ("h100-hbm3x4", "NVIDIA H100 80GB HBM3", 4, 151_200),
)
_ENGINEERING_SEEDS = tuple(range(8_611_107, 8_611_115))
_ENGINEERING_CELLS = (
    ("parity", 8_000),
    ("parity", 9_500),
    ("parity", 10_000),
    ("majority", 8_000),
    ("majority", 9_500),
    ("majority", 10_000),
)
_SCIENTIFIC_DEADLINE_MENU_SECONDS = (172_800, 259_200, 345_600, 432_000)
_PROVIDER_SETUP_SECONDS = 7_200
_REGISTRAR_HANDOFF_SECONDS = 3_600
_PROVIDER_TERMINAL_MARGIN_SECONDS = 3_600

_MODEL_LEAVES = (
    (".gitattributes", 1_519, "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361"),
    ("LICENSE", 11_343, "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e"),
    ("README.md", 4_917, "2e1bcd8bd964728a820be709fa0f7b9dd54817a94fd2254c535df70c5e67fada"),
    ("config.json", 660, "98d2ff8cc47488d08a2b0b3acf4eb99ef210779b42bd48605f6b8e36acdbf670"),
    (
        "generation_config.json",
        242,
        "e558847a8b4402616f1273797b015104dc266fe4b520056fca88823ba8f8ebe6",
    ),
    ("merges.txt", 1_671_839, "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3"),
    (
        "model.safetensors",
        3_087_467_144,
        "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee",
    ),
    (
        "tokenizer.json",
        7_031_645,
        "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    ),
    (
        "tokenizer_config.json",
        7_305,
        "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
    ),
    ("vocab.json", 2_776_833, "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
)

_FUTURE_COMPUTE_FREEZE_FIELDS = (
    "schema",
    "schema_version",
    "study_id",
    "policy_template_digest",
    "campaign_intent_candidate_binding",
    "campaign_intent_registration_binding",
    "execution_identities",
    "checkpoint_a_token_binding",
    "candidate_tuple",
    "prior_candidate_dispositions_digest",
    "checkpoint_b_capsule_binding",
    "source_runtime_model_bindings",
    "source_runtime_model_bindings_digest",
    "fixed_control_envelope_binding",
    "operator_limits",
    "provider_market_binding",
    "canonical_roots",
    "authorization",
    "freeze_digest",
)

_AUTHORIZATION = MappingProxyType(
    {
        "registrar_designated": False,
        "canonical_campaign_intent_candidate_artifact_created": False,
        "campaign_intent_registered": False,
        "compute_freeze_created": False,
        "compute_freeze_registered": False,
        "checkpoint_b_capsule_built": False,
        "checkpoint_b_capsule_staged": False,
        "runtime_frozen": False,
        "model_snapshot_authenticated": False,
        "provider_access_performed": False,
        "provision_receipt_created": False,
        "qualification_execution_authorized": False,
        "qualification_executed": False,
        "compute_route_qualified": False,
        "itt_created": False,
        "training_started": False,
        "g01_launch_authorized": False,
        "outcomes_seen": False,
    }
)

_FORBIDDEN_EXACT_KEYS = frozenset(
    {
        "route",
        "route_lock",
        "route_selection",
        "checkpoint_a_evidence",
        "eligibility_evidence",
        "detailed_a",
        "detailed_a_evidence",
        "detailed_checkpoint_a_evidence",
        "selected_g00f_profile",
        "token_path",
        "token_inode",
        "token_device",
        "token_route",
    }
)
_FORBIDDEN_OUTCOME_KEY_FRAGMENTS = (
    "outcome",
    "metric",
    "prediction",
    "reward",
    "score",
    "scientific_result",
)


class PreprovisionError(RuntimeError):
    """A pure preprovision contract or refusal invariant failed."""


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if type(key) is not str:
                raise PreprovisionError("canonical JSON object keys must be exact text")
            result[key] = _plain(child)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain(child) for child in value]
    if value is None or type(value) in {str, int, bool}:
        return value
    raise PreprovisionError(f"unsupported canonical JSON type: {type(value).__name__}")


def _freeze(value: Any) -> Any:
    if type(value) is dict:
        return MappingProxyType({key: _freeze(child) for key, child in value.items()})
    if type(value) is list:
        return tuple(_freeze(child) for child in value)
    if value is None or type(value) in {str, int, bool}:
        return value
    raise PreprovisionError(f"cannot freeze unsupported value: {type(value).__name__}")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _plain(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise PreprovisionError("ceil-div denominator must be positive")
    return -(-numerator // denominator)


def _strict_canonical_object(payload: bytes, label: str) -> dict[str, Any]:
    if type(payload) is not bytes:
        raise PreprovisionError(f"{label} must be exact bytes")

    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, child in pairs:
            if key in result:
                raise PreprovisionError(f"{label} contains duplicate JSON key {key!r}")
            result[key] = child
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_float=lambda _value: (_ for _ in ()).throw(
                PreprovisionError(f"{label} contains a floating-point number")
            ),
            parse_constant=lambda constant: (_ for _ in ()).throw(
                PreprovisionError(f"{label} contains non-finite {constant}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise PreprovisionError(f"{label} is not strict UTF-8 JSON") from error
    if type(value) is not dict:
        raise PreprovisionError(f"{label} must contain one exact object")
    if _canonical_json_bytes(value) != payload:
        raise PreprovisionError(f"{label} is not in the exact canonical byte encoding")
    return value


def _exact_object(value: Any, fields: frozenset[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise PreprovisionError(f"{label} exact field set changed")
    return value


def _exact_equal(observed: Any, expected: Any, label: str) -> None:
    if type(observed) is not type(expected):
        raise PreprovisionError(f"{label} JSON type changed")
    if type(expected) is dict:
        if set(observed) != set(expected):
            raise PreprovisionError(f"{label} JSON field set changed")
        for key in expected:
            _exact_equal(observed[key], expected[key], f"{label}.{key}")
    elif type(expected) is list:
        if len(observed) != len(expected):
            raise PreprovisionError(f"{label} JSON list length changed")
        for index, (child, expected_child) in enumerate(zip(observed, expected, strict=True)):
            _exact_equal(child, expected_child, f"{label}[{index}]")
    elif observed != expected:
        raise PreprovisionError(f"{label} JSON value changed")


def _exact_int(value: Any, label: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise PreprovisionError(f"{label} must be an exact integer >= {minimum}")
    return value


def _sha256(value: Any, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise PreprovisionError(f"{label} must be one lowercase SHA-256")
    return value


def _uuid4(value: Any, label: str) -> str:
    if type(value) is not str:
        raise PreprovisionError(f"{label} must be one canonical UUIDv4")
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, ValueError) as error:
        raise PreprovisionError(f"{label} must be one canonical UUIDv4") from error
    if parsed.version != 4 or str(parsed) != value:
        raise PreprovisionError(f"{label} must be one canonical UUIDv4")
    return value


def _data_center_id(value: Any) -> str:
    if type(value) is not str or _DATA_CENTER_ID.fullmatch(value) is None:
        raise PreprovisionError("data_center_id must be one uppercase provider identifier")
    return value


def _network_volume_id(value: Any) -> str:
    if type(value) is not str or _NETWORK_VOLUME_ID.fullmatch(value) is None:
        raise PreprovisionError("network_volume_id must be one exact provider identifier")
    return value


def _scan_forbidden_keys(value: Any, label: str = "candidate") -> None:
    if type(value) is dict:
        for key, child in value.items():
            normalized = key.lower().replace("-", "_")
            negative_boundary = normalized.endswith(("_absent", "_seen")) and child is False
            asserted_absence = normalized.endswith("_absent") and child is True
            if not (negative_boundary or asserted_absence) and (
                normalized in _FORBIDDEN_EXACT_KEYS
                or any(fragment in normalized for fragment in _FORBIDDEN_OUTCOME_KEY_FRAGMENTS)
                or (
                    "token" in normalized
                    and any(term in normalized for term in ("path", "inode", "device", "route"))
                )
                or ("detailed" in normalized and ("checkpoint_a" in normalized or "detailed_a" in normalized))
            ):
                raise PreprovisionError(f"{label} contains forbidden field {key!r}")
            _scan_forbidden_keys(child, f"{label}.{key}")
    elif type(value) is list:
        for index, child in enumerate(value):
            _scan_forbidden_keys(child, f"{label}[{index}]")


def _validate_dispositions(value: Any, priority: int) -> list[dict[str, Any]]:
    if type(value) not in {list, tuple} or len(value) != priority:
        raise PreprovisionError("all earlier profiles require one exact disposition")
    fields = frozenset({"profile_id", "state", "registrar_record_sha256"})
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise PreprovisionError(f"prior_candidate_dispositions[{index}] must be an object")
        disposition = _exact_object(_plain(raw), fields, f"prior_candidate_dispositions[{index}]")
        if disposition["profile_id"] != _CANDIDATE_PROFILES[index][0]:
            raise PreprovisionError("prior candidate disposition order/profile binding changed")
        if type(disposition["state"]) is not str or disposition["state"] not in {
            "skipped_no_stock",
            "consumed_failure",
        }:
            raise PreprovisionError("prior candidate disposition state changed")
        _sha256(
            disposition["registrar_record_sha256"],
            f"prior_candidate_dispositions[{index}].registrar_record_sha256",
        )
        result.append(disposition)
    return result


def build_policy_template() -> Mapping[str, Any]:
    """Return a fresh immutable policy template containing no live facts."""

    body = {
        "schema": POLICY_TEMPLATE_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "milestone": {
            "state": "PROSPECTIVE_SOURCE_ONLY",
            "source_review_required": True,
            "source_review_accepted": False,
            "canonical_artifact_created": False,
            "registrar_object_defined": False,
            "timestamps_recorded": False,
        },
        "accepted_bindings": {
            "g01_target_binding_digest": G01Q_TARGET_BINDING_DIGEST,
            "goalzendo_source_fingerprint": GOALZENDO_SOURCE_FINGERPRINT,
            "checkpoint_a_bridge_source_digest": CHECKPOINT_A_BRIDGE_SOURCE_DIGEST,
            "checkpoint_b_refusal_source_digest": CHECKPOINT_B_REFUSAL_SOURCE_DIGEST,
            "checkpoint_b_source_capsule_source_digest": CHECKPOINT_B_CAPSULE_SOURCE_DIGEST,
            "g01q_source_digest": G01Q_SOURCE_DIGEST,
            "g01q_production_shape_digest": G01Q_PRODUCTION_SHAPE_DIGEST,
            "g01q_candidate_policy_digest": G01Q_CANDIDATE_POLICY_DIGEST,
            "g01q_schedule_w4_digest": ("e610df289b4d391ad8c25d6294cb45bf7ee630631cb95fc8bc9d94ba883c5887"),
            "g01q_schedule_w8_digest": ("50d725e32ae8900af6a74ccb492a3f03dd9e8b64f367873b7238987efbbab426"),
            "g01q_review_contract_digest": G01Q_REVIEW_CONTRACT_DIGEST,
            "dependency_constraints_sha256": DEPENDENCY_CONSTRAINTS_SHA256,
        },
        "execution_identity_policy": {
            "eligibility_execution_uuid_is_checkpoint_a_provenance_only": True,
            "campaign_uuid_must_equal_g01_execution_uuid": True,
            "eligibility_and_g01_execution_uuids_must_differ": True,
            "qualification_uuid_is_subordinate_attempt_nonce": True,
            "qualification_uuid_must_differ_from_both_execution_uuids": True,
            "candidate_id_formula": "g01q-<qualification_uuid>",
            "one_g01_execution_uuid_across_fallback_campaign": True,
            "fresh_qualification_uuid_per_candidate": True,
            "checkpoint_a_route_must_not_select_b_resources": True,
            "token_path_inode_device_route_and_detailed_a_forbidden": True,
        },
        "qualification_policy": {
            "ordered_profiles": [
                {
                    "candidate_profile_id": profile_id,
                    "gpu_id": gpu_id,
                    "gpu_count": gpu_count,
                    "qualification_ceiling_seconds": ceiling,
                }
                for profile_id, gpu_id, gpu_count, ceiling in _CANDIDATE_PROFILES
            ],
            "candidate_policy_digest": G01Q_CANDIDATE_POLICY_DIGEST,
            "cloud_type": "SECURE",
            "network_volume_type": "HIGH_PERFORMANCE",
            "network_volume_mount": "/workspace",
            "engineering_pair_count": 8,
            "engineering_run_count": 16,
            "engineering_seeds": list(_ENGINEERING_SEEDS),
            "engineering_cells": [
                {"rule_family": family, "q_p_basis_points": basis_points}
                for family, basis_points in _ENGINEERING_CELLS
            ],
            "scientific_pair_count": 60,
            "scientific_run_count": 120,
            "maximum_pairs_per_worker_formula": "ceil(60/worker_count)",
            "raw_wall_seconds_formula": "maximum_pairs_per_worker*maximum_pair_wall_seconds",
            "wall_multiplier_numerator": 135,
            "wall_multiplier_denominator": 100,
            "terminal_reserve_seconds": 7_200,
            "qualified_wall_seconds_formula": "ceil_to_hour(1.35*raw_wall_seconds+7200)",
            "deadline_menu_seconds": list(_SCIENTIFIC_DEADLINE_MENU_SECONDS),
            "projected_gpu_seconds_formula": "worker_count*selected_deadline_seconds",
            "projected_compute_cost_formula": (
                "ceil(price_micro_usd_per_gpu_hour*projected_gpu_seconds/3600)"
            ),
            "storage_multiplier_numerator": 5,
            "storage_multiplier_denominator": 4,
            "minimum_free_bytes": 1 << 40,
            "minimum_free_inodes": 1_000_000,
            "minimum_gpu_headroom_bytes": 8 * (1 << 30),
            "minimum_gpu_headroom_fraction_numerator": 1,
            "minimum_gpu_headroom_fraction_denominator": 10,
            "production_shape_digest": G01Q_PRODUCTION_SHAPE_DIGEST,
            "runtime_expectation": {
                "image_reference": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
                "image_reference_authenticated": False,
                "oci_manifest_digest_required_before_future_freeze": True,
                "rootfs_closure_required_before_future_freeze": True,
                "python_implementation": "CPython",
                "python_version": "3.12.3",
                "platform_system": "Linux",
                "platform_machine": "x86_64",
                "isolated_flags": ["-I", "-S"],
                "dependency_constraints_sha256": DEPENDENCY_CONSTRAINTS_SHA256,
                "runtime_receipt_required_before_future_freeze": True,
                "runtime_authenticated": False,
            },
            "model_expectation": {
                "repo_id": "Qwen/Qwen2.5-1.5B-Instruct",
                "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
                "materialization": "fresh_regular_files_no_links_exact_10_leaf_full_repository",
                "leaf_files": [
                    {"path": path, "bytes": byte_count, "sha256": sha256}
                    for path, byte_count, sha256 in _MODEL_LEAVES
                ],
                "leaf_manifest_digest": MODEL_LEAF_MANIFEST_DIGEST,
                "expectation_only": True,
                "offline_loading_required": True,
                "remote_code_disabled": True,
                "snapshot_receipt_required_before_future_freeze": True,
                "expected_observed_leaf_equality_required": True,
                "model_snapshot_authenticated": False,
            },
            "control_envelope_requirements": {
                "metric_prediction_stream_fixed_padded_uncompressed": True,
                "model_weight_serialization_uncompressed": True,
                "control_log_fixed_schema": True,
                "raw_scientific_tree_absent_before_control_handoff": True,
                "six_integer_size_count_fields_required": [
                    "final_bytes_per_run",
                    "maximum_live_bytes_per_run",
                    "final_inodes_per_run",
                    "maximum_live_inodes_per_run",
                    "bytes_written_per_run",
                    "completion_file_count",
                ],
            },
            "caps_are_exact_positive_integers": True,
            "candidate_caps_only_tighten_operator_caps": True,
            "candidate_wall_cap_must_be_one_deadline_menu_value": True,
            "candidate_gpu_cap_must_cover_gpu_count_times_wall_cap": True,
            "candidate_cost_cap_must_cover_deadline_at_price_ceiling": True,
            "candidate_caps_fit_per_attempt_caps": True,
            "per_attempt_caps_fit_provider_lifetime_caps": True,
            "per_attempt_provider_seconds_minimum_formula": (
                "7200+qualification_ceiling_seconds+3600+candidate_maximum_wall_seconds+3600"
            ),
            "per_attempt_provider_gpu_seconds_minimum_formula": (
                "gpu_count*per_attempt_provider_seconds_minimum"
            ),
            "per_attempt_provider_cost_minimum_formula": (
                "ceil(price_ceiling_micro_usd_per_gpu_hour*per_attempt_provider_gpu_seconds_minimum/3600)"
            ),
            "provider_setup_seconds": _PROVIDER_SETUP_SECONDS,
            "registrar_handoff_seconds": _REGISTRAR_HANDOFF_SECONDS,
            "provider_terminal_margin_seconds": _PROVIDER_TERMINAL_MARGIN_SECONDS,
            "provider_lifetime_caps_never_reset": True,
            "failed_or_cleaned_attempts_consume_provider_lifetime_caps": True,
            "observed_provider_gpu_price_must_not_exceed_price_ceiling": True,
            "r1_registrar_debits_all_prior_provider_attempts": True,
            "r1_rejects_unless_remaining_lifetime_caps_cover_current_attempt": True,
            "micro_usd_scope": "GPU_COMPUTE_ONLY",
            "storage_and_egress_excluded": True,
            "excluded_charges_require_external_acceptance_before_r1": True,
            "projection_is_not_total_provider_cost": True,
        },
        "canonical_roots": {
            "frozen_source": "/workspace/inputs-goalzendo/g01-executions/<B_UUID>/frozen-source",
            "preexecution": "/workspace/status-goalzendo/g01-preexecution/<B_UUID>",
            "execution_status": "/workspace/status-goalzendo/g01-executions/<B_UUID>",
            "scientific_artifacts": "/workspace/artifacts-goalzendo/g01-known-law",
            "execution_status_absent_through_qualification": True,
            "scientific_artifacts_absent_through_qualification": True,
            "qualification_root_not_defined_by_this_milestone": True,
        },
        "future_compute_freeze_schema": {
            "exact_fields": list(_FUTURE_COMPUTE_FREEZE_FIELDS),
            "builder_implemented": False,
            "validator_conferring_authority_implemented": False,
            "filename_defined": False,
            "canonical_artifact_created": False,
            "requirements": {
                "candidate_whole_file_sha_and_semantic_digest_from_r1": True,
                "campaign_intent_external_registration_record_sha": True,
                "new_launch_and_qualification_capable_checkpoint_b_capsule_revision": True,
                "current_refusing_34_member_capsule_forbidden": True,
                "capsule_freeze_archive_manifest_stage_receipt_and_tree_digests": True,
                "full_source_closure_and_authenticated_launcher_first_byte": True,
                "oci_manifest_and_rootfs_closure": True,
                "interpreter_stdlib_dependency_native_cuda_driver_nccl_environment_invocation_closure": True,
                "exact_model_snapshot_receipt_and_expected_observed_leaf_equality": True,
                "fixed_control_envelope_manifest_and_six_integer_size_count_fields": True,
                "raw_catalog_price_stock_and_preexisting_dc_hp_volume_evidence": True,
                "pod_id_forbidden_before_r2": True,
                "provision_receipt_forbidden_before_r2": True,
            },
        },
        "registrar_order": {
            "steps": [
                "release_pinned_source",
                "unregistered_candidate",
                "external_registrar_record_r1",
                "capsule_stage_and_read_only_runtime_model_catalog_volume_evidence",
                "compute_freeze",
                "external_registrar_record_r2",
                "provider_create",
                "provision_receipt_and_external_registrar_record_r3",
                "qualification_receipts_report_selection",
                "still_no_g01_launch",
            ],
            "future_registrar_enforces_exact_next_priority": True,
            "future_registrar_enforces_at_most_one_live_attempt": True,
            "candidate_has_no_self_whole_file_sha": True,
            "r1_pins_candidate_whole_file_sha": True,
            "r2_pins_compute_freeze": True,
            "mutual_future_hash_references_forbidden": True,
            "registrar_schema_defined": False,
        },
        "authorization": _plain(_AUTHORIZATION),
    }
    return cast(Mapping[str, Any], _freeze({**body, "template_digest": _digest(body)}))


def validate_policy_template_bytes(payload: bytes) -> Mapping[str, Any]:
    """Replay the sole canonical byte encoding of the policy template."""

    value = _strict_canonical_object(payload, "G01Q preprovision policy template")
    expected = _plain(build_policy_template())
    _exact_equal(value, expected, "G01Q preprovision policy template")
    body = {key: child for key, child in value.items() if key != "template_digest"}
    if value["template_digest"] != _digest(body):
        raise PreprovisionError("policy template semantic digest changed")
    return cast(Mapping[str, Any], _freeze(value))


def build_unregistered_campaign_intent_candidate(
    *,
    eligibility_execution_uuid: str,
    g01_execution_uuid: str,
    campaign_uuid: str,
    qualification_uuid: str,
    checkpoint_a_token_file_sha256: str,
    checkpoint_a_token_digest: str,
    checkpoint_a_bridge_source_digest: str,
    checkpoint_a_token_portable_bytes: int,
    candidate_priority: int,
    data_center_id: str,
    network_volume_id: str,
    prior_candidate_dispositions: Sequence[Mapping[str, Any]],
    prospective_transaction_source_digest: str,
    price_ceiling_micro_usd_per_gpu_hour: int,
    operator_maximum_wall_seconds: int,
    operator_maximum_gpu_seconds: int,
    operator_maximum_cost_micro_usd: int,
    candidate_maximum_wall_seconds: int,
    candidate_maximum_gpu_seconds: int,
    candidate_maximum_cost_micro_usd: int,
    operator_maximum_total_provider_seconds: int,
    operator_maximum_total_provider_gpu_seconds: int,
    operator_maximum_total_provider_cost_micro_usd: int,
    operator_maximum_per_attempt_provider_seconds: int,
    operator_maximum_per_attempt_provider_gpu_seconds: int,
    operator_maximum_per_attempt_provider_cost_micro_usd: int,
) -> Mapping[str, Any]:
    """Build an immutable UNREGISTERED candidate from unverified caller facts."""

    eligibility_uuid = _uuid4(eligibility_execution_uuid, "eligibility_execution_uuid")
    execution_uuid = _uuid4(g01_execution_uuid, "g01_execution_uuid")
    campaign_uuid_value = _uuid4(campaign_uuid, "campaign_uuid")
    qualification_uuid_value = _uuid4(qualification_uuid, "qualification_uuid")
    if execution_uuid != campaign_uuid_value:
        raise PreprovisionError("g01_execution_uuid must equal campaign_uuid")
    if len({eligibility_uuid, execution_uuid, qualification_uuid_value}) != 3:
        raise PreprovisionError("eligibility, G01 execution, and qualification UUIDs must differ")

    priority = _exact_int(candidate_priority, "candidate_priority", minimum=0)
    if priority >= len(_CANDIDATE_PROFILES):
        raise PreprovisionError("candidate_priority is outside the exact profile order")
    profile_id, gpu_id, gpu_count, qualification_ceiling = _CANDIDATE_PROFILES[priority]
    dispositions = _validate_dispositions(prior_candidate_dispositions, priority)

    limits = {
        "price_ceiling_micro_usd_per_gpu_hour": _exact_int(
            price_ceiling_micro_usd_per_gpu_hour,
            "price_ceiling_micro_usd_per_gpu_hour",
        ),
        "operator_maximum_wall_seconds": _exact_int(
            operator_maximum_wall_seconds, "operator_maximum_wall_seconds"
        ),
        "operator_maximum_gpu_seconds": _exact_int(
            operator_maximum_gpu_seconds, "operator_maximum_gpu_seconds"
        ),
        "operator_maximum_cost_micro_usd": _exact_int(
            operator_maximum_cost_micro_usd, "operator_maximum_cost_micro_usd"
        ),
        "candidate_maximum_wall_seconds": _exact_int(
            candidate_maximum_wall_seconds, "candidate_maximum_wall_seconds"
        ),
        "candidate_maximum_gpu_seconds": _exact_int(
            candidate_maximum_gpu_seconds, "candidate_maximum_gpu_seconds"
        ),
        "candidate_maximum_cost_micro_usd": _exact_int(
            candidate_maximum_cost_micro_usd, "candidate_maximum_cost_micro_usd"
        ),
        "operator_maximum_total_provider_seconds": _exact_int(
            operator_maximum_total_provider_seconds,
            "operator_maximum_total_provider_seconds",
        ),
        "operator_maximum_total_provider_gpu_seconds": _exact_int(
            operator_maximum_total_provider_gpu_seconds,
            "operator_maximum_total_provider_gpu_seconds",
        ),
        "operator_maximum_total_provider_cost_micro_usd": _exact_int(
            operator_maximum_total_provider_cost_micro_usd,
            "operator_maximum_total_provider_cost_micro_usd",
        ),
        "operator_maximum_per_attempt_provider_seconds": _exact_int(
            operator_maximum_per_attempt_provider_seconds,
            "operator_maximum_per_attempt_provider_seconds",
        ),
        "operator_maximum_per_attempt_provider_gpu_seconds": _exact_int(
            operator_maximum_per_attempt_provider_gpu_seconds,
            "operator_maximum_per_attempt_provider_gpu_seconds",
        ),
        "operator_maximum_per_attempt_provider_cost_micro_usd": _exact_int(
            operator_maximum_per_attempt_provider_cost_micro_usd,
            "operator_maximum_per_attempt_provider_cost_micro_usd",
        ),
    }
    if limits["candidate_maximum_wall_seconds"] not in _SCIENTIFIC_DEADLINE_MENU_SECONDS:
        raise PreprovisionError("candidate wall cap must be one exact scientific deadline")
    for suffix in ("wall_seconds", "gpu_seconds", "cost_micro_usd"):
        if limits[f"candidate_maximum_{suffix}"] > limits[f"operator_maximum_{suffix}"]:
            raise PreprovisionError("candidate scientific caps must not exceed operator caps")
    for suffix in ("seconds", "gpu_seconds", "cost_micro_usd"):
        if (
            limits[f"operator_maximum_per_attempt_provider_{suffix}"]
            > limits[f"operator_maximum_total_provider_{suffix}"]
        ):
            raise PreprovisionError("per-attempt provider caps must not exceed lifetime caps")
    if (
        limits["candidate_maximum_wall_seconds"] > limits["operator_maximum_per_attempt_provider_seconds"]
        or limits["candidate_maximum_gpu_seconds"]
        > limits["operator_maximum_per_attempt_provider_gpu_seconds"]
        or limits["candidate_maximum_cost_micro_usd"]
        > limits["operator_maximum_per_attempt_provider_cost_micro_usd"]
    ):
        raise PreprovisionError("candidate scientific caps must fit within one provider attempt")
    minimum_scientific_gpu_seconds = gpu_count * limits["candidate_maximum_wall_seconds"]
    if limits["candidate_maximum_gpu_seconds"] < minimum_scientific_gpu_seconds:
        raise PreprovisionError("candidate GPU-second cap cannot cover its wall deadline")
    minimum_scientific_cost = _ceil_div(
        limits["price_ceiling_micro_usd_per_gpu_hour"] * minimum_scientific_gpu_seconds,
        3_600,
    )
    if limits["candidate_maximum_cost_micro_usd"] < minimum_scientific_cost:
        raise PreprovisionError("candidate cost cap cannot cover its deadline at the price ceiling")
    minimum_attempt_seconds = (
        _PROVIDER_SETUP_SECONDS
        + qualification_ceiling
        + _REGISTRAR_HANDOFF_SECONDS
        + limits["candidate_maximum_wall_seconds"]
        + _PROVIDER_TERMINAL_MARGIN_SECONDS
    )
    minimum_attempt_gpu_seconds = gpu_count * minimum_attempt_seconds
    minimum_attempt_cost = _ceil_div(
        limits["price_ceiling_micro_usd_per_gpu_hour"] * minimum_attempt_gpu_seconds,
        3_600,
    )
    if (
        limits["operator_maximum_per_attempt_provider_seconds"] < minimum_attempt_seconds
        or limits["operator_maximum_per_attempt_provider_gpu_seconds"] < minimum_attempt_gpu_seconds
        or limits["operator_maximum_per_attempt_provider_cost_micro_usd"] < minimum_attempt_cost
    ):
        raise PreprovisionError("per-attempt caps cannot cover setup, qualification, handoff, and science")

    policy_digest = cast(str, build_policy_template()["template_digest"])
    body = {
        "schema": CAMPAIGN_CANDIDATE_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "state": CAMPAIGN_STATE,
        "policy_template_digest": policy_digest,
        "execution_identities": {
            "eligibility_execution_uuid": eligibility_uuid,
            "g01_execution_uuid": execution_uuid,
            "campaign_uuid": campaign_uuid_value,
            "qualification_uuid": qualification_uuid_value,
            "candidate_id": f"g01q-{qualification_uuid_value}",
        },
        "checkpoint_a_token": {
            "file_sha256": _sha256(checkpoint_a_token_file_sha256, "checkpoint_a_token_file_sha256"),
            "token_digest": _sha256(checkpoint_a_token_digest, "checkpoint_a_token_digest"),
            "bridge_source_digest": _sha256(
                checkpoint_a_bridge_source_digest, "checkpoint_a_bridge_source_digest"
            ),
            "portability": {
                "representation": "PORTABLE_BYTES",
                "byte_count": _exact_int(
                    checkpoint_a_token_portable_bytes, "checkpoint_a_token_portable_bytes"
                ),
                "path_inode_device_and_route_absent": True,
                "detailed_checkpoint_a_evidence_absent": True,
            },
            "externally_authenticated": False,
        },
        "candidate_tuple": {
            "candidate_priority": priority,
            "candidate_profile_id": profile_id,
            "gpu_id": gpu_id,
            "gpu_count": gpu_count,
            "cloud_type": "SECURE",
            "data_center_id": _data_center_id(data_center_id),
            "network_volume_id": _network_volume_id(network_volume_id),
            "network_volume_type": "HIGH_PERFORMANCE",
            "network_volume_mount": "/workspace",
            "qualification_ceiling_seconds": qualification_ceiling,
            "candidate_policy_digest": G01Q_CANDIDATE_POLICY_DIGEST,
            "prior_candidate_dispositions": dispositions,
        },
        "operator_limits": limits,
        "expected_source_binding": {
            "prospective_transaction_source_digest": _sha256(
                prospective_transaction_source_digest,
                "prospective_transaction_source_digest",
            ),
            "g01_target_binding_digest": G01Q_TARGET_BINDING_DIGEST,
            "goalzendo_source_fingerprint": GOALZENDO_SOURCE_FINGERPRINT,
            "checkpoint_a_bridge_source_digest": CHECKPOINT_A_BRIDGE_SOURCE_DIGEST,
            "checkpoint_b_refusal_source_digest": CHECKPOINT_B_REFUSAL_SOURCE_DIGEST,
            "checkpoint_b_source_capsule_source_digest": CHECKPOINT_B_CAPSULE_SOURCE_DIGEST,
            "g01q_source_digest": G01Q_SOURCE_DIGEST,
            "new_checkpoint_b_capsule_revision_required": True,
            "canonical_capsule_artifacts_present": False,
            "source_binding_authenticated": False,
        },
        "expected_runtime_binding": {
            "image_reference": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
            "image_reference_authenticated": False,
            "oci_manifest_digest_required_before_future_freeze": True,
            "rootfs_closure_required_before_future_freeze": True,
            "python_implementation": "CPython",
            "python_version": "3.12.3",
            "platform_system": "Linux",
            "platform_machine": "x86_64",
            "isolated_flags": ["-I", "-S"],
            "dependency_constraints_sha256": DEPENDENCY_CONSTRAINTS_SHA256,
            "runtime_receipt_required_before_future_freeze": True,
            "runtime_authenticated": False,
        },
        "expected_model_binding": {
            "repo_id": "Qwen/Qwen2.5-1.5B-Instruct",
            "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
            "materialization": "fresh_regular_files_no_links_exact_10_leaf_full_repository",
            "leaf_files": [
                {"path": path, "bytes": byte_count, "sha256": sha256}
                for path, byte_count, sha256 in _MODEL_LEAVES
            ],
            "leaf_manifest_digest": MODEL_LEAF_MANIFEST_DIGEST,
            "expectation_only": True,
            "offline_loading_required": True,
            "remote_code_disabled": True,
            "snapshot_receipt_required_before_future_freeze": True,
            "expected_observed_leaf_equality_required": True,
            "model_snapshot_authenticated": False,
        },
        "future_compute_freeze_requirements": {
            "exact_fields": list(_FUTURE_COMPUTE_FREEZE_FIELDS),
            "r1_candidate_whole_file_sha_required": True,
            "r1_registration_record_sha_required": True,
            "new_checkpoint_b_capsule_revision_required": True,
            "capsule_freeze_archive_manifest_stage_receipt_and_tree_digests_required": True,
            "source_runtime_model_closure_required": True,
            "fixed_control_envelope_and_six_integer_fields_required": True,
            "provider_market_dc_volume_evidence_required": True,
            "pod_id_and_provision_receipt_forbidden_before_r2": True,
            "builder_or_validator_present": False,
            "canonical_artifact_present": False,
        },
        "external_trust": {
            "caller_supplied_fields_label": "caller_supplied_unverified",
            "caller_supplied_uuid_facts_verified": False,
            "caller_supplied_token_facts_verified": False,
            "caller_supplied_provider_tuple_verified": False,
            "caller_supplied_limit_facts_verified": False,
            "caller_supplied_disposition_records_verified": False,
            "caller_supplied_transaction_source_digest_verified": False,
            "storage_and_egress_external_acceptance_present": False,
            "excluded_charges_require_separate_external_acceptance_before_r1": True,
            "registrar_designated": False,
            "external_registration_record_present": False,
        },
        "authorization": _plain(_AUTHORIZATION),
    }
    if body["checkpoint_a_token"]["bridge_source_digest"] != CHECKPOINT_A_BRIDGE_SOURCE_DIGEST:
        raise PreprovisionError("thin token bridge source digest changed from checkpoint A")
    _scan_forbidden_keys(body)
    return cast(Mapping[str, Any], _freeze({**body, "candidate_digest": _digest(body)}))


def validate_unregistered_campaign_intent_candidate_bytes(payload: bytes) -> Mapping[str, Any]:
    """Strictly replay a candidate without authenticating any supplied fact."""

    value = _strict_canonical_object(payload, "G01Q unregistered campaign intent candidate")
    _scan_forbidden_keys(value)
    _exact_object(
        value,
        frozenset(
            {
                "schema",
                "schema_version",
                "study_id",
                "state",
                "policy_template_digest",
                "execution_identities",
                "checkpoint_a_token",
                "candidate_tuple",
                "operator_limits",
                "expected_source_binding",
                "expected_runtime_binding",
                "expected_model_binding",
                "future_compute_freeze_requirements",
                "external_trust",
                "authorization",
                "candidate_digest",
            }
        ),
        "candidate",
    )
    identities = _exact_object(
        value["execution_identities"],
        frozenset(
            {
                "eligibility_execution_uuid",
                "g01_execution_uuid",
                "campaign_uuid",
                "qualification_uuid",
                "candidate_id",
            }
        ),
        "execution_identities",
    )
    token = _exact_object(
        value["checkpoint_a_token"],
        frozenset(
            {
                "file_sha256",
                "token_digest",
                "bridge_source_digest",
                "portability",
                "externally_authenticated",
            }
        ),
        "checkpoint_a_token",
    )
    portability = _exact_object(
        token["portability"],
        frozenset(
            {
                "representation",
                "byte_count",
                "path_inode_device_and_route_absent",
                "detailed_checkpoint_a_evidence_absent",
            }
        ),
        "checkpoint_a_token.portability",
    )
    candidate_tuple = _exact_object(
        value["candidate_tuple"],
        frozenset(
            {
                "candidate_priority",
                "candidate_profile_id",
                "gpu_id",
                "gpu_count",
                "cloud_type",
                "data_center_id",
                "network_volume_id",
                "network_volume_type",
                "network_volume_mount",
                "qualification_ceiling_seconds",
                "candidate_policy_digest",
                "prior_candidate_dispositions",
            }
        ),
        "candidate_tuple",
    )
    limit_fields = frozenset(
        {
            "price_ceiling_micro_usd_per_gpu_hour",
            "operator_maximum_wall_seconds",
            "operator_maximum_gpu_seconds",
            "operator_maximum_cost_micro_usd",
            "candidate_maximum_wall_seconds",
            "candidate_maximum_gpu_seconds",
            "candidate_maximum_cost_micro_usd",
            "operator_maximum_total_provider_seconds",
            "operator_maximum_total_provider_gpu_seconds",
            "operator_maximum_total_provider_cost_micro_usd",
            "operator_maximum_per_attempt_provider_seconds",
            "operator_maximum_per_attempt_provider_gpu_seconds",
            "operator_maximum_per_attempt_provider_cost_micro_usd",
        }
    )
    limits = _exact_object(value["operator_limits"], limit_fields, "operator_limits")
    source = _exact_object(
        value["expected_source_binding"],
        frozenset(
            {
                "prospective_transaction_source_digest",
                "g01_target_binding_digest",
                "goalzendo_source_fingerprint",
                "checkpoint_a_bridge_source_digest",
                "checkpoint_b_refusal_source_digest",
                "checkpoint_b_source_capsule_source_digest",
                "g01q_source_digest",
                "new_checkpoint_b_capsule_revision_required",
                "canonical_capsule_artifacts_present",
                "source_binding_authenticated",
            }
        ),
        "expected_source_binding",
    )
    expected = build_unregistered_campaign_intent_candidate(
        eligibility_execution_uuid=identities["eligibility_execution_uuid"],
        g01_execution_uuid=identities["g01_execution_uuid"],
        campaign_uuid=identities["campaign_uuid"],
        qualification_uuid=identities["qualification_uuid"],
        checkpoint_a_token_file_sha256=token["file_sha256"],
        checkpoint_a_token_digest=token["token_digest"],
        checkpoint_a_bridge_source_digest=token["bridge_source_digest"],
        checkpoint_a_token_portable_bytes=portability["byte_count"],
        candidate_priority=candidate_tuple["candidate_priority"],
        data_center_id=candidate_tuple["data_center_id"],
        network_volume_id=candidate_tuple["network_volume_id"],
        prior_candidate_dispositions=candidate_tuple["prior_candidate_dispositions"],
        prospective_transaction_source_digest=source["prospective_transaction_source_digest"],
        **cast(dict[str, int], limits),
    )
    _exact_equal(value, _plain(expected), "G01Q unregistered campaign intent candidate")
    body = {key: child for key, child in value.items() if key != "candidate_digest"}
    if value["candidate_digest"] != _digest(body):
        raise PreprovisionError("candidate semantic digest changed")
    return cast(Mapping[str, Any], _freeze(value))


def register_campaign(*_args: Any, **_kwargs: Any) -> NoReturn:
    raise PreprovisionError(REFUSAL)


def create_compute_freeze(*_args: Any, **_kwargs: Any) -> NoReturn:
    raise PreprovisionError(REFUSAL)


def build_capsule(*_args: Any, **_kwargs: Any) -> NoReturn:
    raise PreprovisionError(REFUSAL)


def stage_capsule(*_args: Any, **_kwargs: Any) -> NoReturn:
    raise PreprovisionError(REFUSAL)


def materialize_model(*_args: Any, **_kwargs: Any) -> NoReturn:
    raise PreprovisionError(REFUSAL)


def provision(*_args: Any, **_kwargs: Any) -> NoReturn:
    raise PreprovisionError(REFUSAL)


def run_qualification(*_args: Any, **_kwargs: Any) -> NoReturn:
    raise PreprovisionError(REFUSAL)


def execute_preprovision(*_args: Any, **_kwargs: Any) -> NoReturn:
    raise PreprovisionError(REFUSAL)
