#!/usr/bin/env python3
"""Verify the dated, non-authorizing repository release policy.

``quick`` is a read-only integrity and reviewed-debt check. ``full`` adds the
expensive test, static-analysis, paper-rebuild, and isolated-wheel checks. The
policy deliberately does not bind a Git commit or authorize any experiment.
"""

from __future__ import annotations

import argparse
import configparser
import hashlib
import importlib.metadata
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = REPOSITORY_ROOT / "reproducibility/releases/2026-08-11/release-policy.json"
POLICY_CANONICALIZATION = (
    "UTF-8 JSON; keys sorted lexicographically; two-space indentation; "
    "ensure_ascii=false; LF line endings; one terminal LF"
)
RUFF_NORMALIZATION = (
    "ruff-json-v1: repository-relative POSIX path and exactly "
    "{path,row,column,code,message}; records sorted by "
    "(path,row,column,code,message); compact sorted-key UTF-8 JSON plus terminal LF"
)
MYPY_NORMALIZATION = (
    "mypy-json-v1: severity=error records only; repository-relative POSIX path and exactly "
    "{path,row,column,code,message,severity}; records sorted by "
    "(path,row,column,str(code),message,severity); compact sorted-key UTF-8 JSON plus terminal LF"
)
SOURCE_TREE_NORMALIZATION = (
    "source-tree-v1: regular non-symlink files recursively; exclude __pycache__, *.pyc, and .DS_Store; "
    "records exactly {path,sha256} with package-relative POSIX paths sorted by path; "
    "compact sorted-key UTF-8 JSON plus terminal LF"
)
PYPROJECT_SHA256 = "086e55d4e9cebcd2fe278c618055a8cad6e51de70d8c2a00195e8f2c88267dcf"
G00F_SOURCE_MANIFEST_SCHEMA = "goalzendo.g00f_additive_source_manifest"
G00F_H200_SOURCE_MANIFEST_SCHEMA = "goalzendo.g00f_h200_additive_source_manifest"
G00F_SOURCE_MANIFEST_VERSION = 1
COMPONENT_MANIFEST_CONTRACTS = {
    "goalzendo_g00f": {
        "freeze_path": "reproducibility/goalzendo/g00f-execution-freeze-20260811/execution-freeze.json",
        "schema": G00F_SOURCE_MANIFEST_SCHEMA,
    },
    "goalzendo_g00f_h200": {
        "freeze_path": (
            "reproducibility/goalzendo/g00f-h200-execution-freeze-20260811/execution-freeze.json"
        ),
        "schema": G00F_H200_SOURCE_MANIFEST_SCHEMA,
    },
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MANIFEST_LINE_RE = re.compile(r"^([0-9a-f]{64})  ([^\r\n]+)$")
CHECKPOINT_A_BUNDLE_ROOT = "reproducibility/goalzendo/g00f-g01-bridge-prebootstrap-20260812"
CHECKPOINT_A_ARCHIVE_NAME = "g00f-g01-bridge-bundle.tar.gz"
CHECKPOINT_A_MANIFEST_NAME = "g00f-g01-bridge-bundle-manifest.json"
CHECKPOINT_A_FREEZE_NAME = "g00f-g01-bridge-bundle-freeze.json"
CHECKPOINT_A_SOURCE_DATE_EPOCH = 1_786_492_800
CHECKPOINT_A_TRANSPORT_PREBUILD_TEST_AUDIT_SHA256 = (
    "2035a819cf96695dcc478d1f4902e856f2f3f7c233bf528248a86b0b51880b11"
)
CHECKPOINT_A_AUTHORIZATION = {
    "dedicated_global_coordinator_required": True,
    "direct_g01_launch_authorized": False,
    "g00f_outcomes_seen": False,
    "g01_scientifically_eligible": False,
}
CHECKPOINT_A_TRANSACTION = {
    "bundle_targets_must_all_be_absent": True,
    "canonical_program_root": "/workspace/status-goalzendo/g00f-executions",
    "exclusive_regular_writes": True,
    "frozen_source_directory": "frozen-source",
    "partial_failure_is_permanent": True,
    "receipt_written_last_outside_uuid_root": True,
    "selected_route_freeze_target_must_be_absent": True,
}
CHECKPOINT_B_REFUSAL = "B_RUNTIME_OVERLAY_NOT_FROZEN"
CHECKPOINT_B_COORDINATOR_SOURCE_DIGEST = "77d9ba4928fa29cc42a055201660304f2059d0ca6358371cc14618407bdc714b"
CHECKPOINT_B_PACKAGE_SOURCE_TREE_SHA256 = "1be9a56e256bc640d877dc2fc9c1ba4d2d397cef783adec248e037c9faf382b7"
CHECKPOINT_B_ACCEPTED_FILES = {
    "docs/goalzendo/protocols/g01-global-coordinator-checkpoint-b.md": {
        "role": "accepted_source_checkpoint_protocol",
        "sha256": "93acc6262758dd5478e342e2f89fd5cb1ab3b61a03bed34b1a6d5083cacaaeba",
    },
    "runs/goalzendo/run_g01_global_coordinator.py": {
        "role": "accepted_refusing_bootstrap_entrypoint",
        "sha256": "69f5ec520d8c681d5e9cab1dd9687d7fd39c948f08ff397331bd5ce74f824a06",
    },
    "src/goalzendo_g01_coordinator/__init__.py": {
        "role": "accepted_source_only_package_surface",
        "sha256": "85a9454fc21c2dc8c70e6ebf9eb605aa0208e6c13e73547ec0f4abdd08abad35",
    },
    "src/goalzendo_g01_coordinator/coordinator.py": {
        "role": "accepted_refusing_coordinator_source",
        "sha256": "a3390dad47ea1fd2aa7b1ca22e4475b9a510c47fc85f9b296ed925303663b302",
    },
    "tests/goalzendo/test_g01_global_coordinator.py": {
        "role": "accepted_adversarial_source_checkpoint_tests",
        "sha256": "f0ec9d0c4815e6bd69e7dda2a72a8f9ba03bf4405dced5e12f05f99660be207b",
    },
}
CHECKPOINT_B_AUTHORIZATION = {
    "checkpoint_b_complete": False,
    "g01_launch_authorized": False,
    "model_execution_authorized": False,
}
CHECKPOINT_B_OPERATIONAL_STATUS = {
    "authenticated_launcher_created": False,
    "compute_estimate_resolved": False,
    "four_gpu_provision_operationally_accepted": False,
    "measured_pilot_scheduling_resolved": False,
    "outcomes_seen": False,
    "provision_transaction_created": False,
    "prospective_lifecycle_operationally_accepted": False,
    "runtime_overlay_created": False,
    "scientific_runpod_pod_created": False,
    "scientific_runpod_provision_receipt_created": False,
    "source_checkpoint_accepted": True,
    "stager_created": False,
    "supported_entrypoint_refusal_active": True,
    "wall_ceiling_operationally_accepted": False,
}
CHECKPOINT_B_SOURCE_CAPSULE_DIGEST = "efb6b0427895f60e16794f4e20b87d620522673edffb94f9ce277089b5581782"
CHECKPOINT_B_SOURCE_CAPSULE_ACCEPTED_FILES = {
    "docs/goalzendo/protocols/g01-b-source-capsule.md": {
        "role": "accepted_nonauthorizing_source_capsule_protocol",
        "sha256": "cf9d73edc279b6dacd1e64823ffff016a8d409b2cdb9cd434e5e244d17687251",
    },
    "runs/goalzendo/build_g01_b_source_capsule.py": {
        "role": "accepted_canonical-build-held_source_capsule_builder",
        "sha256": "c1255da5ed0bebe5c91bbcc724213e671e9b22b2b79dc52915426e51ea6aa593",
    },
    "runs/goalzendo/g01_b_source_capsule_stage.py": {
        "role": "accepted_nonauthorizing_source_capsule_stager_and_verifier",
        "sha256": "253685cefcf31a31324a8aa14ec854b9fef09babf38d24a5e930c68da80afa55",
    },
    "tests/goalzendo/test_g01_b_source_capsule.py": {
        "role": "accepted_adversarial_source_capsule_tests",
        "sha256": "fe0919f58adc131c774ab5f7387a7ac0d3b2c2a4d83c7dd4503b9ba77fe6abd3",
    },
}
CHECKPOINT_B_SOURCE_CAPSULE_AUTHORIZATION = {
    "canonical_capsule_build_authorized": False,
    "capsule_stage_authorized": False,
    "checkpoint_b_complete": False,
    "g01_launch_authorized": False,
    "model_execution_authorized": False,
}
CHECKPOINT_B_SOURCE_CAPSULE_OPERATIONAL_STATUS = {
    "authenticated_launcher_created": False,
    "canonical_capsule_built": False,
    "capsule_staged": False,
    "fresh_stage_verification_receipt_created": False,
    "g01_artifact_root_created": False,
    "g01_execution_status_root_created": False,
    "outcomes_seen": False,
    "provision_transaction_created": False,
    "runtime_overlay_created": False,
    "source_implementation_accepted": True,
    "source_stage_receipt_created": False,
}
G01Q_REFUSAL = "G01Q_RUNTIME_PROVISION_NOT_FROZEN"
G01Q_QUALIFICATION_SOURCE_DIGEST = "efde59caf3830d021223940b2d6e1a8e9631c90f0f8e51262041008ea177c7b3"
G01Q_PACKAGE_SOURCE_TREE_SHA256 = "68ed88badaf8c95e96b9aa4ff80711c8ed53ad78772534878f865af7201ec315"
G01Q_ACCEPTED_FILES = {
    "docs/goalzendo/protocols/g01-compute-qualification.md": {
        "role": "accepted_nonauthorizing_compute_qualification_protocol",
        "sha256": "e4997c48f0611117c08ca6850d438f5f0fb54bcdc2dba155d23e0b77cbdec355",
    },
    "runs/goalzendo/run_g01_compute_qualification.py": {
        "role": "accepted_refusing_compute_qualification_entrypoint",
        "sha256": "2531edd47bbe5883741e674ccf41b4ddf77e16530f6b67c92a1ed51a90cb7eb4",
    },
    "src/goalzendo_g01_qualification/__init__.py": {
        "role": "accepted_source_only_qualification_package_surface",
        "sha256": "075ea0981ba3fe5da1a7ce95b481f3ef988be8e0be62c9d185b30f0c0d699fab",
    },
    "src/goalzendo_g01_qualification/qualification.py": {
        "role": "accepted_nonauthorizing_compute_qualification_source",
        "sha256": "5edc9e9fb756081040c62da8965f462eb9baa1d26b276becd41872810e646355",
    },
    "tests/goalzendo/test_g01_compute_qualification.py": {
        "role": "accepted_adversarial_compute_qualification_tests",
        "sha256": "a4acde81cf393673daf6ad49b3bfb6708a10b5b1f2b14f19e1a47b905f543c6e",
    },
}
G01Q_AUTHORIZATION = {
    "g01_launch_authorized": False,
    "g01_training_authorized": False,
    "model_execution_authorized": False,
    "qualification_execution_authorized": False,
}
G01Q_OPERATIONAL_STATUS = {
    "compute_route_qualified": False,
    "itt_created": False,
    "outcomes_seen": False,
    "qualification_campaign_created": False,
    "qualification_cleanup_receipt_created": False,
    "qualification_evidence_created": False,
    "qualification_executed": False,
    "qualification_pair_receipts_created": False,
    "qualification_provision_receipt_created": False,
    "qualification_report_created": False,
    "qualification_run_receipts_created": False,
    "qualification_runtime_frozen": False,
    "qualification_selection_created": False,
    "source_checkpoint_accepted": True,
    "supported_entrypoint_refusal_active": True,
    "training_started": False,
}
G01Q_PREPROVISION_REFUSAL = "G01Q_REGISTRAR_NOT_DESIGNATED"
G01Q_PREPROVISION_SOURCE_DIGEST = "76dc4cf6f8273325cb688b77efbb2861a123052bec38d6765174932dfd994b7d"
G01Q_PREPROVISION_PACKAGE_SOURCE_TREE_SHA256 = (
    "2285f1412f6418e2be1af3bf60f12e10cca0d8101e25061698ae5e962b738f76"
)
G01Q_PREPROVISION_ACCEPTED_FILES = {
    "docs/goalzendo/protocols/g01-preprovision.md": {
        "role": "accepted_nonauthorizing_g01q_preprovision_protocol",
        "sha256": "f55f8c5c080e3bff33b4bbaac7afc8b4469b062b03134103382576ea76203263",
    },
    "runs/goalzendo/run_g01_preprovision.py": {
        "role": "accepted_refusing_g01q_preprovision_entrypoint",
        "sha256": "a01bddbe35baea1989dd88ffe14fc32b82b616ee7de7fc7d4d34e7e7c35efa7b",
    },
    "src/goalzendo_g01_preprovision/__init__.py": {
        "role": "accepted_source_only_g01q_preprovision_package_surface",
        "sha256": "cf7c84aee4803446f16220ab69c1251c42c076345d9d9e0aa5d6968bfc305f51",
    },
    "src/goalzendo_g01_preprovision/contracts.py": {
        "role": "accepted_nonauthorizing_g01q_preprovision_contracts",
        "sha256": "de00c80a15e9be8e09adefee702dd691da946bb97a682e534945cec8a19fac95",
    },
    "tests/goalzendo/test_g01_preprovision.py": {
        "role": "accepted_adversarial_g01q_preprovision_tests",
        "sha256": "c6ed55fa12f722616dd333313b85d6cee394845043be8c32bda9ef213dc2bf70",
    },
}
G01Q_PREPROVISION_AUTHORIZATION = {
    "g01_launch_authorized": False,
    "g01_training_authorized": False,
    "model_execution_authorized": False,
    "qualification_execution_authorized": False,
}
G01Q_PREPROVISION_OPERATIONAL_STATUS = {
    "canonical_campaign_intent_candidate_artifact_created": False,
    "campaign_intent_registered": False,
    "checkpoint_b_capsule_built": False,
    "checkpoint_b_capsule_staged": False,
    "compute_freeze_created": False,
    "compute_freeze_registered": False,
    "compute_route_qualified": False,
    "itt_created": False,
    "model_snapshot_authenticated": False,
    "outcomes_seen": False,
    "provider_access_performed": False,
    "provision_receipt_created": False,
    "qualification_executed": False,
    "registrar_designated": False,
    "runtime_frozen": False,
    "source_checkpoint_accepted": True,
    "supported_entrypoint_refusal_active": True,
    "training_started": False,
}


class PolicyError(ValueError):
    """The release policy is malformed or non-canonical."""


class VerificationError(RuntimeError):
    """A release-policy check failed."""


class _CaseSensitiveConfigParser(configparser.ConfigParser):
    def optionxform(self, optionstr: str) -> str:
        return optionstr


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


def _duplicates_rejected(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PolicyError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _canonical_json_bytes(value: Any, *, pretty: bool) -> bytes:
    if pretty:
        text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    else:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return (text + "\n").encode("utf-8")


def _as_object(value: Any, where: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise PolicyError(f"{where} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], where: str, expected: set[str]) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise PolicyError(f"{where} keys differ; missing={missing}, unknown={unknown}")


def _as_string(value: Any, where: str, *, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value):
        raise PolicyError(f"{where} must be a{' nonempty' if nonempty else ''} string")
    return value


def _as_bool(value: Any, where: str) -> bool:
    if type(value) is not bool:
        raise PolicyError(f"{where} must be a boolean")
    return value


def _as_int(value: Any, where: str, *, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        suffix = f" >= {minimum}" if minimum is not None else ""
        raise PolicyError(f"{where} must be an integer{suffix}")
    return value


def _as_list(value: Any, where: str) -> list[Any]:
    if type(value) is not list:
        raise PolicyError(f"{where} must be an array")
    return value


def _validate_sha256(value: Any, where: str) -> str:
    digest = _as_string(value, where)
    if not SHA256_RE.fullmatch(digest):
        raise PolicyError(f"{where} must be a lowercase SHA-256 digest")
    return digest


def _validate_relative_path(value: Any, where: str) -> str:
    raw = _as_string(value, where)
    if "\\" in raw:
        raise PolicyError(f"{where} must use POSIX separators")
    path = PurePosixPath(raw)
    if path.is_absolute() or raw != path.as_posix() or any(part in {"", ".", ".."} for part in path.parts):
        raise PolicyError(f"{where} must be a normalized repository-relative path")
    return raw


def _validate_unique_strings(value: Any, where: str, *, paths: bool = False) -> list[str]:
    items = _as_list(value, where)
    result = [
        _validate_relative_path(item, f"{where}[{index}]") if paths else _as_string(item, f"{where}[{index}]")
        for index, item in enumerate(items)
    ]
    if len(set(result)) != len(result):
        raise PolicyError(f"{where} must not contain duplicates")
    if result != sorted(result):
        raise PolicyError(f"{where} must be sorted")
    return result


def _validate_file_binding(
    value: Any,
    where: str,
    *,
    role: bool,
    extra_sha256_keys: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    obj = _as_object(value, where)
    keys = ({"path", "sha256", "role"} if role else {"path", "sha256"}) | set(extra_sha256_keys)
    _exact_keys(obj, where, keys)
    _validate_relative_path(obj["path"], f"{where}.path")
    _validate_sha256(obj["sha256"], f"{where}.sha256")
    for key in extra_sha256_keys:
        _validate_sha256(obj[key], f"{where}.{key}")
    if role:
        _as_string(obj["role"], f"{where}.role")
    return obj


def _validate_checkpoint_a_bundle_binding(value: Any) -> None:
    where = "bindings.checkpoint_a_canonical_bundle"
    binding = _as_object(value, where)
    _exact_keys(
        binding,
        where,
        {
            "archive",
            "authorization",
            "bridge_source_digest",
            "build_input_transport",
            "build_runtime_digest",
            "builder",
            "freeze",
            "frozen_transaction_digest",
            "manifest",
            "operational_status",
            "stager",
        },
    )
    expected_paths = {
        "archive": f"{CHECKPOINT_A_BUNDLE_ROOT}/{CHECKPOINT_A_ARCHIVE_NAME}",
        "freeze": f"{CHECKPOINT_A_BUNDLE_ROOT}/{CHECKPOINT_A_FREEZE_NAME}",
        "manifest": f"{CHECKPOINT_A_BUNDLE_ROOT}/{CHECKPOINT_A_MANIFEST_NAME}",
        "builder": "runs/goalzendo/build_g00f_g01_bridge_bundle.py",
        "stager": "runs/goalzendo/g00f_g01_bridge_bundle_stage.py",
    }
    for name, expected_path in expected_paths.items():
        row = _as_object(binding[name], f"{where}.{name}")
        keys = (
            {"path", "sha256", f"{name}_digest"}
            if name in {"freeze", "manifest"}
            else {
                "path",
                "sha256",
            }
        )
        _exact_keys(row, f"{where}.{name}", keys)
        if _validate_relative_path(row["path"], f"{where}.{name}.path") != expected_path:
            raise PolicyError(f"{where}.{name}.path must equal {expected_path}")
        _validate_sha256(row["sha256"], f"{where}.{name}.sha256")
        if name in {"freeze", "manifest"}:
            _validate_sha256(row[f"{name}_digest"], f"{where}.{name}.{name}_digest")

    _validate_sha256(binding["bridge_source_digest"], f"{where}.bridge_source_digest")
    _validate_sha256(binding["build_runtime_digest"], f"{where}.build_runtime_digest")
    _validate_sha256(binding["frozen_transaction_digest"], f"{where}.frozen_transaction_digest")

    authorization = _as_object(binding["authorization"], f"{where}.authorization")
    _exact_keys(authorization, f"{where}.authorization", set(CHECKPOINT_A_AUTHORIZATION))
    if authorization != CHECKPOINT_A_AUTHORIZATION:
        raise PolicyError(f"{where}.authorization must remain exactly non-authorizing")

    transport = _as_list(binding["build_input_transport"], f"{where}.build_input_transport")
    transport_rows = []
    for index, row in enumerate(transport):
        raw_row = _as_object(row, f"{where}.build_input_transport[{index}]")
        extra_keys = (
            frozenset({"prebuild_audit_sha256"})
            if raw_row.get("path") == "tests/goalzendo/test_g00f_g01_build_input_transport.py"
            else frozenset()
        )
        transport_rows.append(
            _validate_file_binding(
                row,
                f"{where}.build_input_transport[{index}]",
                role=True,
                extra_sha256_keys=extra_keys,
            )
        )
    expected_transport = {
        "docs/goalzendo/protocols/g00f-g01-build-input-transport.md": ("build_input_transport_protocol"),
        "runs/goalzendo/g00f_g01_build_input_transport.py": ("build_input_transport_controller"),
        "tests/goalzendo/test_g00f_g01_build_input_transport.py": (
            "phase_compatible_post_land_build_input_transport_tests"
        ),
    }
    observed_transport = {row["path"]: row["role"] for row in transport_rows}
    if observed_transport != expected_transport or [row["path"] for row in transport_rows] != sorted(
        expected_transport
    ):
        raise PolicyError(f"{where}.build_input_transport must bind the exact sorted source closure")
    test_row = next(
        row
        for row in transport_rows
        if row["path"] == "tests/goalzendo/test_g00f_g01_build_input_transport.py"
    )
    if test_row["prebuild_audit_sha256"] != CHECKPOINT_A_TRANSPORT_PREBUILD_TEST_AUDIT_SHA256:
        raise PolicyError(f"{where}.build_input_transport historical prebuild audit changed")

    operational = _as_object(binding["operational_status"], f"{where}.operational_status")
    expected_operational = {
        "bundle_staged": False,
        "build_intent_transcript_witness_recorded": True,
        "build_intent_witness_durable_independent_registrar": False,
        "build_intent_witness_reverified_by_release": False,
        "build_result_transcript_witness_recorded": True,
        "build_result_witness_durable_independent_registrar": False,
        "build_result_witness_reverified_by_release": False,
        "canonical_bundle_built": True,
        "checkpoint_b_global_coordinator_created": False,
        "coordinator_token_created": False,
        "direct_g01_launch_authorized": False,
        "disposable_build_pod_created": True,
        "disposable_build_pod_deleted": True,
        "outcomes_seen": False,
        "route_lock_created": False,
        "scientific_eligibility_artifact_created": False,
        "scientific_runpod_pod_created": False,
        "scientific_runpod_provision_receipt_created": False,
        "stage_receipt_created": False,
    }
    _exact_keys(operational, f"{where}.operational_status", set(expected_operational))
    if operational != expected_operational:
        raise PolicyError(f"{where}.operational_status must remain at the post-build pre-stage boundary")


def _validate_checkpoint_b_source_binding(value: Any) -> None:
    where = "bindings.checkpoint_b_source_milestone"
    binding = _as_object(value, where)
    _exact_keys(
        binding,
        where,
        {
            "accepted_files",
            "authorization",
            "coordinator_source_digest",
            "operational_status",
            "package_source_tree_sha256",
            "refusal_code",
        },
    )
    accepted = _as_list(binding["accepted_files"], f"{where}.accepted_files")
    observed: dict[str, dict[str, str]] = {}
    paths: list[str] = []
    for index, value in enumerate(accepted):
        row_where = f"{where}.accepted_files[{index}]"
        row = _validate_file_binding(value, row_where, role=True)
        path = row["path"]
        paths.append(path)
        observed[path] = {"role": row["role"], "sha256": row["sha256"]}
    if paths != sorted(CHECKPOINT_B_ACCEPTED_FILES) or observed != CHECKPOINT_B_ACCEPTED_FILES:
        raise PolicyError(f"{where}.accepted_files must bind the exact accepted five-file closure")
    if (
        _validate_sha256(
            binding["coordinator_source_digest"],
            f"{where}.coordinator_source_digest",
        )
        != CHECKPOINT_B_COORDINATOR_SOURCE_DIGEST
    ):
        raise PolicyError(f"{where}.coordinator_source_digest changed")
    if (
        _validate_sha256(
            binding["package_source_tree_sha256"],
            f"{where}.package_source_tree_sha256",
        )
        != CHECKPOINT_B_PACKAGE_SOURCE_TREE_SHA256
    ):
        raise PolicyError(f"{where}.package_source_tree_sha256 changed")
    if binding["refusal_code"] != CHECKPOINT_B_REFUSAL:
        raise PolicyError(f"{where}.refusal_code must remain the accepted hard refusal")
    authorization = _as_object(binding["authorization"], f"{where}.authorization")
    _exact_keys(authorization, f"{where}.authorization", set(CHECKPOINT_B_AUTHORIZATION))
    for key in CHECKPOINT_B_AUTHORIZATION:
        _as_bool(authorization[key], f"{where}.authorization.{key}")
    if authorization != CHECKPOINT_B_AUTHORIZATION:
        raise PolicyError(f"{where}.authorization must remain exactly non-authorizing")
    operational = _as_object(binding["operational_status"], f"{where}.operational_status")
    _exact_keys(operational, f"{where}.operational_status", set(CHECKPOINT_B_OPERATIONAL_STATUS))
    for key in CHECKPOINT_B_OPERATIONAL_STATUS:
        _as_bool(operational[key], f"{where}.operational_status.{key}")
    if operational != CHECKPOINT_B_OPERATIONAL_STATUS:
        raise PolicyError(f"{where}.operational_status must remain at the source-only refusal boundary")


def _validate_checkpoint_b_source_capsule_binding(value: Any) -> None:
    where = "bindings.checkpoint_b_source_capsule_milestone"
    binding = _as_object(value, where)
    _exact_keys(
        binding,
        where,
        {"accepted_files", "authorization", "operational_status", "source_capsule_digest"},
    )
    accepted = _as_list(binding["accepted_files"], f"{where}.accepted_files")
    observed: dict[str, dict[str, str]] = {}
    paths: list[str] = []
    for index, value in enumerate(accepted):
        row_where = f"{where}.accepted_files[{index}]"
        row = _validate_file_binding(value, row_where, role=True)
        paths.append(row["path"])
        observed[row["path"]] = {"role": row["role"], "sha256": row["sha256"]}
    if (
        paths != sorted(CHECKPOINT_B_SOURCE_CAPSULE_ACCEPTED_FILES)
        or observed != CHECKPOINT_B_SOURCE_CAPSULE_ACCEPTED_FILES
    ):
        raise PolicyError(f"{where}.accepted_files must bind the exact accepted four-file closure")
    if (
        _validate_sha256(binding["source_capsule_digest"], f"{where}.source_capsule_digest")
        != CHECKPOINT_B_SOURCE_CAPSULE_DIGEST
    ):
        raise PolicyError(f"{where}.source_capsule_digest changed")
    authorization = _as_object(binding["authorization"], f"{where}.authorization")
    _exact_keys(
        authorization,
        f"{where}.authorization",
        set(CHECKPOINT_B_SOURCE_CAPSULE_AUTHORIZATION),
    )
    for key in CHECKPOINT_B_SOURCE_CAPSULE_AUTHORIZATION:
        _as_bool(authorization[key], f"{where}.authorization.{key}")
    if authorization != CHECKPOINT_B_SOURCE_CAPSULE_AUTHORIZATION:
        raise PolicyError(f"{where}.authorization must remain exactly non-authorizing")
    operational = _as_object(binding["operational_status"], f"{where}.operational_status")
    _exact_keys(
        operational,
        f"{where}.operational_status",
        set(CHECKPOINT_B_SOURCE_CAPSULE_OPERATIONAL_STATUS),
    )
    for key in CHECKPOINT_B_SOURCE_CAPSULE_OPERATIONAL_STATUS:
        _as_bool(operational[key], f"{where}.operational_status.{key}")
    if operational != CHECKPOINT_B_SOURCE_CAPSULE_OPERATIONAL_STATUS:
        raise PolicyError(f"{where}.operational_status must remain at the unbuilt source boundary")


def _validate_g01q_source_binding(value: Any) -> None:
    where = "bindings.g01q_source_milestone"
    binding = _as_object(value, where)
    _exact_keys(
        binding,
        where,
        {
            "accepted_files",
            "authorization",
            "operational_status",
            "package_source_tree_sha256",
            "qualification_source_digest",
            "refusal_code",
        },
    )
    accepted = _as_list(binding["accepted_files"], f"{where}.accepted_files")
    observed: dict[str, dict[str, str]] = {}
    paths: list[str] = []
    for index, value in enumerate(accepted):
        row_where = f"{where}.accepted_files[{index}]"
        row = _validate_file_binding(value, row_where, role=True)
        path = row["path"]
        paths.append(path)
        observed[path] = {"role": row["role"], "sha256": row["sha256"]}
    if paths != sorted(G01Q_ACCEPTED_FILES) or observed != G01Q_ACCEPTED_FILES:
        raise PolicyError(f"{where}.accepted_files must bind the exact accepted five-file closure")
    if (
        _validate_sha256(
            binding["qualification_source_digest"],
            f"{where}.qualification_source_digest",
        )
        != G01Q_QUALIFICATION_SOURCE_DIGEST
    ):
        raise PolicyError(f"{where}.qualification_source_digest changed")
    if (
        _validate_sha256(
            binding["package_source_tree_sha256"],
            f"{where}.package_source_tree_sha256",
        )
        != G01Q_PACKAGE_SOURCE_TREE_SHA256
    ):
        raise PolicyError(f"{where}.package_source_tree_sha256 changed")
    if binding["refusal_code"] != G01Q_REFUSAL:
        raise PolicyError(f"{where}.refusal_code must remain the accepted hard refusal")
    authorization = _as_object(binding["authorization"], f"{where}.authorization")
    _exact_keys(authorization, f"{where}.authorization", set(G01Q_AUTHORIZATION))
    for key in G01Q_AUTHORIZATION:
        _as_bool(authorization[key], f"{where}.authorization.{key}")
    if authorization != G01Q_AUTHORIZATION:
        raise PolicyError(f"{where}.authorization must remain exactly non-authorizing")
    operational = _as_object(binding["operational_status"], f"{where}.operational_status")
    _exact_keys(operational, f"{where}.operational_status", set(G01Q_OPERATIONAL_STATUS))
    for key in G01Q_OPERATIONAL_STATUS:
        _as_bool(operational[key], f"{where}.operational_status.{key}")
    if operational != G01Q_OPERATIONAL_STATUS:
        raise PolicyError(f"{where}.operational_status must remain at the source-only refusal boundary")


def _validate_g01q_preprovision_source_binding(value: Any) -> None:
    where = "bindings.g01q_preprovision_source_milestone"
    binding = _as_object(value, where)
    _exact_keys(
        binding,
        where,
        {
            "accepted_files",
            "authorization",
            "operational_status",
            "package_source_tree_sha256",
            "preprovision_source_digest",
            "refusal_code",
        },
    )
    accepted = _as_list(binding["accepted_files"], f"{where}.accepted_files")
    observed: dict[str, dict[str, str]] = {}
    paths: list[str] = []
    for index, value in enumerate(accepted):
        row_where = f"{where}.accepted_files[{index}]"
        row = _validate_file_binding(value, row_where, role=True)
        path = row["path"]
        paths.append(path)
        observed[path] = {"role": row["role"], "sha256": row["sha256"]}
    if paths != sorted(G01Q_PREPROVISION_ACCEPTED_FILES) or observed != G01Q_PREPROVISION_ACCEPTED_FILES:
        raise PolicyError(f"{where}.accepted_files must bind the exact accepted five-file closure")
    if (
        _validate_sha256(
            binding["preprovision_source_digest"],
            f"{where}.preprovision_source_digest",
        )
        != G01Q_PREPROVISION_SOURCE_DIGEST
    ):
        raise PolicyError(f"{where}.preprovision_source_digest changed")
    if (
        _validate_sha256(
            binding["package_source_tree_sha256"],
            f"{where}.package_source_tree_sha256",
        )
        != G01Q_PREPROVISION_PACKAGE_SOURCE_TREE_SHA256
    ):
        raise PolicyError(f"{where}.package_source_tree_sha256 changed")
    if binding["refusal_code"] != G01Q_PREPROVISION_REFUSAL:
        raise PolicyError(f"{where}.refusal_code must remain the accepted hard refusal")
    authorization = _as_object(binding["authorization"], f"{where}.authorization")
    _exact_keys(authorization, f"{where}.authorization", set(G01Q_PREPROVISION_AUTHORIZATION))
    for key in G01Q_PREPROVISION_AUTHORIZATION:
        _as_bool(authorization[key], f"{where}.authorization.{key}")
    if authorization != G01Q_PREPROVISION_AUTHORIZATION:
        raise PolicyError(f"{where}.authorization must remain exactly non-authorizing")
    operational = _as_object(binding["operational_status"], f"{where}.operational_status")
    _exact_keys(operational, f"{where}.operational_status", set(G01Q_PREPROVISION_OPERATIONAL_STATUS))
    for key in G01Q_PREPROVISION_OPERATIONAL_STATUS:
        _as_bool(operational[key], f"{where}.operational_status.{key}")
    if operational != G01Q_PREPROVISION_OPERATIONAL_STATUS:
        raise PolicyError(f"{where}.operational_status must remain at the source-only refusal boundary")


def _validate_policy_shape(policy: dict[str, Any]) -> None:
    top_keys = {
        "authority",
        "bindings",
        "canonicalization",
        "full_checks",
        "known_absences",
        "schema_name",
        "schema_version",
        "scope",
        "static_debt_baselines",
        "status_as_of",
    }
    _exact_keys(policy, "policy", top_keys)
    if policy["schema_name"] != "goalzendo.repository_release_policy":
        raise PolicyError("policy.schema_name is unsupported")
    if policy["schema_version"] != 1 or type(policy["schema_version"]) is not int:
        raise PolicyError("policy.schema_version must equal integer 1")
    if policy["canonicalization"] != POLICY_CANONICALIZATION:
        raise PolicyError("policy.canonicalization is unsupported")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", _as_string(policy["status_as_of"], "status_as_of")):
        raise PolicyError("status_as_of must use YYYY-MM-DD")

    authority = _as_object(policy["authority"], "authority")
    _exact_keys(
        authority,
        "authority",
        {"experiment_launch", "final_release_identity", "model_execution", "statement", "study_gate"},
    )
    for key in ("experiment_launch", "final_release_identity", "model_execution", "study_gate"):
        if _as_bool(authority[key], f"authority.{key}"):
            raise PolicyError(f"authority.{key} must remain false")
    _as_string(authority["statement"], "authority.statement")

    scope = _as_object(policy["scope"], "scope")
    _exact_keys(
        scope,
        "scope",
        {
            "binds_final_release_tree",
            "binds_git_commit",
            "forkworld_boundary",
            "release_candidate_only",
            "repository_program",
        },
    )
    if scope["repository_program"] != "GoalZendo with preserved, separate FORKWORLD":
        raise PolicyError("scope.repository_program is unsupported")
    _as_string(scope["forkworld_boundary"], "scope.forkworld_boundary")
    if _as_bool(scope["binds_git_commit"], "scope.binds_git_commit"):
        raise PolicyError("scope.binds_git_commit must remain false")
    if _as_bool(scope["binds_final_release_tree"], "scope.binds_final_release_tree"):
        raise PolicyError("scope.binds_final_release_tree must remain false")
    if not _as_bool(scope["release_candidate_only"], "scope.release_candidate_only"):
        raise PolicyError("scope.release_candidate_only must remain true")

    bindings = _as_object(policy["bindings"], "bindings")
    _exact_keys(
        bindings,
        "bindings",
        {
            "checkpoint_a_canonical_bundle",
            "checkpoint_b_source_capsule_milestone",
            "checkpoint_b_source_milestone",
            "checksum_manifests",
            "compact_inputs",
            "g01q_preprovision_source_milestone",
            "g01q_source_milestone",
            "ledger",
            "pdfs",
            "source_packages",
            "tool_versions",
        },
    )
    _validate_checkpoint_a_bundle_binding(bindings["checkpoint_a_canonical_bundle"])
    _validate_checkpoint_b_source_capsule_binding(bindings["checkpoint_b_source_capsule_milestone"])
    _validate_checkpoint_b_source_binding(bindings["checkpoint_b_source_milestone"])
    _validate_g01q_source_binding(bindings["g01q_source_milestone"])
    _validate_g01q_preprovision_source_binding(bindings["g01q_preprovision_source_milestone"])
    ledger = _as_object(bindings["ledger"], "bindings.ledger")
    _exact_keys(ledger, "bindings.ledger", {"path", "sha256", "verify_repository_evidence"})
    if _validate_relative_path(ledger["path"], "bindings.ledger.path") != (
        "reproducibility/goalzendo/study-status-ledger.json"
    ):
        raise PolicyError("bindings.ledger.path must name the canonical study-status ledger")
    _validate_sha256(ledger["sha256"], "bindings.ledger.sha256")
    if not _as_bool(ledger["verify_repository_evidence"], "bindings.ledger.verify_repository_evidence"):
        raise PolicyError("bindings.ledger.verify_repository_evidence must be true")

    manifests = _as_list(bindings["checksum_manifests"], "bindings.checksum_manifests")
    if not manifests:
        raise PolicyError("bindings.checksum_manifests must not be empty")
    manifest_paths: list[str] = []
    for index, value in enumerate(manifests):
        where = f"bindings.checksum_manifests[{index}]"
        obj = _as_object(value, where)
        _exact_keys(obj, where, {"entry_base", "path", "sha256"})
        manifest_paths.append(_validate_relative_path(obj["path"], f"{where}.path"))
        _validate_sha256(obj["sha256"], f"{where}.sha256")
        if obj["entry_base"] not in {"manifest_parent", "repository_root"}:
            raise PolicyError(f"{where}.entry_base is unsupported")
    if len(set(manifest_paths)) != len(manifest_paths) or manifest_paths != sorted(manifest_paths):
        raise PolicyError("bindings.checksum_manifests paths must be unique and sorted")
    required_manifests = {
        "reproducibility/goalzendo/frozen-sources/SHA256SUMS",
        "reproducibility/goalzendo/g00d-gate-20260811/SHA256SUMS",
        "reproducibility/goalzendo/g00f-execution-freeze-20260811/SHA256SUMS",
        "reproducibility/goalzendo/g03g-smoke-20260811/SHA256SUMS",
    }
    if not required_manifests.issubset(manifest_paths):
        raise PolicyError("bindings.checksum_manifests omits preserved evidence")

    pdfs = _as_list(bindings["pdfs"], "bindings.pdfs")
    if not pdfs:
        raise PolicyError("bindings.pdfs must not be empty")
    pdf_paths = [
        _validate_file_binding(value, f"bindings.pdfs[{index}]", role=True)["path"]
        for index, value in enumerate(pdfs)
    ]
    if len(set(pdf_paths)) != len(pdf_paths) or pdf_paths != sorted(pdf_paths):
        raise PolicyError("bindings.pdfs paths must be unique and sorted")
    required_pdfs = {
        "paper/forkworld-current-results/lesswrong.pdf",
        "paper/forkworld-current-results/main.pdf",
        "paper/goalzendo-current-results/zendo.pdf",
    }
    if not required_pdfs.issubset(pdf_paths):
        raise PolicyError("bindings.pdfs omits a canonical paper")

    compact = _as_object(bindings["compact_inputs"], "bindings.compact_inputs")
    _exact_keys(compact, "bindings.compact_inputs", {"entry_base", "path", "sha256"})
    if _validate_relative_path(compact["path"], "bindings.compact_inputs.path") != (
        "paper/goalzendo-current-results/compact-inputs.sha256"
    ):
        raise PolicyError("bindings.compact_inputs.path must name the canonical compact-input manifest")
    _validate_sha256(compact["sha256"], "bindings.compact_inputs.sha256")
    if compact["entry_base"] != "manifest_parent":
        raise PolicyError("bindings.compact_inputs.entry_base must equal manifest_parent")

    source_packages = _as_object(bindings["source_packages"], "bindings.source_packages")
    _exact_keys(
        source_packages,
        "bindings.source_packages",
        {"component_manifests", "normalization", "packages", "source_only_packages"},
    )
    if source_packages["normalization"] != SOURCE_TREE_NORMALIZATION:
        raise PolicyError("bindings.source_packages.normalization is unsupported")
    package_bindings = _as_list(source_packages["packages"], "bindings.source_packages.packages")
    source_package_names: list[str] = []
    for index, value in enumerate(package_bindings):
        where = f"bindings.source_packages.packages[{index}]"
        obj = _as_object(value, where)
        _exact_keys(obj, where, {"name", "path", "sha256"})
        name = _as_string(obj["name"], f"{where}.name")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise PolicyError(f"{where}.name must be a top-level Python package")
        if _validate_relative_path(obj["path"], f"{where}.path") != f"src/{name}":
            raise PolicyError(f"{where}.path must equal src/{name}")
        _validate_sha256(obj["sha256"], f"{where}.sha256")
        source_package_names.append(name)
    if source_package_names != sorted(set(source_package_names)):
        raise PolicyError("bindings.source_packages.packages names must be unique and sorted")
    source_only_packages = _validate_unique_strings(
        source_packages["source_only_packages"],
        "bindings.source_packages.source_only_packages",
    )
    if not set(source_only_packages).issubset(source_package_names):
        raise PolicyError("source-only packages must name bound source packages")
    expected_source_only_packages = [
        "goalzendo_g00f_g01_bridge",
        "goalzendo_g00f_h200",
        "goalzendo_g01_coordinator",
        "goalzendo_g01_preprovision",
        "goalzendo_g01_qualification",
    ]
    if source_only_packages != expected_source_only_packages:
        raise PolicyError(
            "checkpoint-A bridge, H200 runtime, checkpoint-B coordinator, and G01Q source-only packages "
            "must be the exact source-only packages"
        )

    component_manifests = _as_list(
        source_packages["component_manifests"],
        "bindings.source_packages.component_manifests",
    )
    component_manifest_packages: list[str] = []
    for index, value in enumerate(component_manifests):
        where = f"bindings.source_packages.component_manifests[{index}]"
        obj = _as_object(value, where)
        _exact_keys(
            obj,
            where,
            {
                "freeze_path",
                "freeze_sha256",
                "manifest_digest",
                "package",
                "path",
                "schema",
                "schema_version",
                "sha256",
                "source_digest",
            },
        )
        package = _as_string(obj["package"], f"{where}.package")
        if package not in source_package_names:
            raise PolicyError(f"{where}.package must name a bound source package")
        expected_path = f"src/{package}/source_manifest.json"
        if _validate_relative_path(obj["path"], f"{where}.path") != expected_path:
            raise PolicyError(f"{where}.path must equal {expected_path}")
        contract = COMPONENT_MANIFEST_CONTRACTS.get(package)
        if contract is None:
            raise PolicyError(f"{where}.package has no supported component-manifest contract")
        expected_freeze_path = contract["freeze_path"]
        if _validate_relative_path(obj["freeze_path"], f"{where}.freeze_path") != expected_freeze_path:
            raise PolicyError(f"{where}.freeze_path must equal {expected_freeze_path}")
        _validate_sha256(obj["freeze_sha256"], f"{where}.freeze_sha256")
        _validate_sha256(obj["sha256"], f"{where}.sha256")
        _validate_sha256(obj["source_digest"], f"{where}.source_digest")
        _validate_sha256(obj["manifest_digest"], f"{where}.manifest_digest")
        if obj["schema"] != contract["schema"]:
            raise PolicyError(f"{where}.schema is unsupported")
        if obj["schema_version"] != G00F_SOURCE_MANIFEST_VERSION or type(obj["schema_version"]) is not int:
            raise PolicyError(f"{where}.schema_version is unsupported")
        component_manifest_packages.append(package)
    if component_manifest_packages != sorted(set(component_manifest_packages)):
        raise PolicyError("component-manifest packages must be unique and sorted")
    if set(component_manifest_packages) != set(COMPONENT_MANIFEST_CONTRACTS):
        raise PolicyError("both G00-F runtime packages must have exact component-manifest bindings")

    tools = _as_object(bindings["tool_versions"], "bindings.tool_versions")
    _exact_keys(tools, "bindings.tool_versions", {"external_tools", "python", "python_distributions"})
    python_tool = _as_object(tools["python"], "bindings.tool_versions.python")
    _exact_keys(python_tool, "bindings.tool_versions.python", {"implementation", "version"})
    if python_tool["implementation"] != "CPython":
        raise PolicyError("only CPython release policies are supported")
    if not re.fullmatch(r"\d+\.\d+\.\d+", _as_string(python_tool["version"], "python.version")):
        raise PolicyError("python.version must be a three-component version")
    distributions = _as_list(tools["python_distributions"], "python_distributions")
    distribution_names: list[str] = []
    for index, value in enumerate(distributions):
        where = f"python_distributions[{index}]"
        obj = _as_object(value, where)
        _exact_keys(obj, where, {"name", "version"})
        distribution_names.append(_as_string(obj["name"], f"{where}.name"))
        _as_string(obj["version"], f"{where}.version")
    if len({name.casefold() for name in distribution_names}) != len(
        distribution_names
    ) or distribution_names != sorted(distribution_names, key=str.casefold):
        raise PolicyError("python_distributions names must be case-insensitively unique and sorted")
    required_distributions = {
        "matplotlib",
        "mypy",
        "numpy",
        "pandas",
        "pip",
        "pytest",
        "PyYAML",
        "ruff",
        "scipy",
        "setuptools",
        "torch",
    }
    if not required_distributions.issubset(distribution_names):
        raise PolicyError("python_distributions omits a required release dependency/tool")
    external_tools = _as_list(tools["external_tools"], "external_tools")
    external_names: list[str] = []
    for index, value in enumerate(external_tools):
        where = f"external_tools[{index}]"
        obj = _as_object(value, where)
        _exact_keys(obj, where, {"command", "expected", "name", "version_pattern"})
        external_names.append(_as_string(obj["name"], f"{where}.name"))
        command = [
            _as_string(item, f"{where}.command") for item in _as_list(obj["command"], f"{where}.command")
        ]
        if not command:
            raise PolicyError(f"{where}.command must not be empty")
        pattern = _as_string(obj["version_pattern"], f"{where}.version_pattern")
        try:
            compiled = re.compile(pattern, re.MULTILINE)
        except re.error as exc:
            raise PolicyError(f"{where}.version_pattern is invalid: {exc}") from exc
        if compiled.groups != 1:
            raise PolicyError(f"{where}.version_pattern must contain exactly one capture group")
        _as_string(obj["expected"], f"{where}.expected")
    if external_names != sorted(set(external_names)):
        raise PolicyError("external_tools names must be unique and sorted")
    if not {"latexmk", "make", "pdftex", "shasum"}.issubset(external_names):
        raise PolicyError("external_tools omits a paper/release tool")

    baselines = _as_object(policy["static_debt_baselines"], "static_debt_baselines")
    _exact_keys(baselines, "static_debt_baselines", {"mypy", "ruff"})
    for tool, normalization in (("ruff", RUFF_NORMALIZATION), ("mypy", MYPY_NORMALIZATION)):
        where = f"static_debt_baselines.{tool}"
        obj = _as_object(baselines[tool], where)
        _exact_keys(
            obj,
            where,
            {"command", "diagnostic_count", "expected_exit_code", "normalization", "normalized_sha256"},
        )
        command = [
            _as_string(item, f"{where}.command") for item in _as_list(obj["command"], f"{where}.command")
        ]
        expected_command = (
            ["ruff", "check", ".", "--output-format=json"] if tool == "ruff" else ["mypy", "--output=json"]
        )
        if command != expected_command:
            raise PolicyError(f"{where}.command must equal {expected_command!r}")
        if _as_int(obj["expected_exit_code"], f"{where}.expected_exit_code") != 1:
            raise PolicyError(f"{where}.expected_exit_code must equal 1")
        _as_int(obj["diagnostic_count"], f"{where}.diagnostic_count", minimum=1)
        if obj["normalization"] != normalization:
            raise PolicyError(f"{where}.normalization is unsupported")
        _validate_sha256(obj["normalized_sha256"], f"{where}.normalized_sha256")

    full = _as_object(policy["full_checks"], "full_checks")
    _exact_keys(full, "full_checks", {"goalzendo_static", "paper", "pytest", "wheel"})
    pytest_spec = _as_object(full["pytest"], "full_checks.pytest")
    _exact_keys(pytest_spec, "full_checks.pytest", {"command", "expected_exit_code"})
    if pytest_spec["command"] != ["{python}", "-m", "pytest"]:
        raise PolicyError("full_checks.pytest.command is unsupported")
    if _as_int(pytest_spec["expected_exit_code"], "full_checks.pytest.expected_exit_code") != 0:
        raise PolicyError("full_checks.pytest.expected_exit_code must equal 0")

    scoped = _as_object(full["goalzendo_static"], "full_checks.goalzendo_static")
    _exact_keys(scoped, "full_checks.goalzendo_static", {"mypy_paths", "ruff_paths"})
    ruff_paths = _validate_unique_strings(scoped["ruff_paths"], "goalzendo_static.ruff_paths", paths=True)
    mypy_paths = _validate_unique_strings(scoped["mypy_paths"], "goalzendo_static.mypy_paths", paths=True)
    required_packages = {
        "src/goalzendo",
        "src/goalzendo_g00e",
        "src/goalzendo_g00f",
        "src/goalzendo_g00f_g01_bridge",
        "src/goalzendo_g00f_h200",
        "src/goalzendo_g01_coordinator",
        "src/goalzendo_g01_preprovision",
        "src/goalzendo_g01_qualification",
        "src/goalzendo_hidden_law",
        "src/goalzendo_interactive",
        "src/goalzendo_interactive_v2",
    }
    if not required_packages.issubset(ruff_paths) or not required_packages.issubset(mypy_paths):
        raise PolicyError("GoalZendo static scopes omit a required source package")
    required_ruff_paths = required_packages | {
        "paper/goalzendo-current-results/analysis",
        "runs/goalzendo",
        "scripts/verify_release.py",
        "tests/goalzendo",
        "tests/goalzendo_hidden_law",
        "tests/goalzendo_interactive",
        "tests/goalzendo_interactive_v2",
        "tests/test_release_policy.py",
    }
    if not required_ruff_paths.issubset(ruff_paths):
        raise PolicyError("GoalZendo Ruff scope omits release code, tests, runs, or paper helpers")

    paper = _as_object(full["paper"], "full_checks.paper")
    _exact_keys(paper, "full_checks.paper", {"commands", "outputs", "source_dir"})
    if _validate_relative_path(paper["source_dir"], "full_checks.paper.source_dir") != (
        "paper/goalzendo-current-results"
    ):
        raise PolicyError("full_checks.paper.source_dir must name the GoalZendo manuscript")
    commands = _as_list(paper["commands"], "full_checks.paper.commands")
    if commands != [["make", "clean"], ["make", "compact"]]:
        raise PolicyError("full_checks.paper.commands must be make clean then make compact")
    outputs = _as_list(paper["outputs"], "full_checks.paper.outputs")
    if not outputs:
        raise PolicyError("full_checks.paper.outputs must not be empty")
    output_paths = [
        _validate_file_binding(value, f"full_checks.paper.outputs[{index}]", role=False)["path"]
        for index, value in enumerate(outputs)
    ]
    if output_paths != sorted(set(output_paths)):
        raise PolicyError("full_checks.paper.outputs paths must be unique and sorted")
    if output_paths != ["main.pdf", "zendo.pdf"]:
        raise PolicyError("full_checks.paper.outputs must bind main.pdf and zendo.pdf")

    wheel = _as_object(full["wheel"], "full_checks.wheel")
    _exact_keys(
        wheel,
        "full_checks.wheel",
        {
            "build_backend",
            "console_scripts",
            "dependency_statement",
            "network_allowed",
            "no_deps_install",
            "package_data",
            "packages",
            "project_file",
            "system_site_packages",
        },
    )
    if wheel["build_backend"] != "pip-wheel-no-build-isolation":
        raise PolicyError("full_checks.wheel.build_backend is unsupported")
    if _as_bool(wheel["network_allowed"], "full_checks.wheel.network_allowed"):
        raise PolicyError("full_checks.wheel.network_allowed must remain false")
    if not _as_bool(wheel["no_deps_install"], "full_checks.wheel.no_deps_install"):
        raise PolicyError("full_checks.wheel.no_deps_install must remain true")
    if not _as_bool(wheel["system_site_packages"], "full_checks.wheel.system_site_packages"):
        raise PolicyError("full_checks.wheel.system_site_packages must remain true")
    _as_string(wheel["dependency_statement"], "full_checks.wheel.dependency_statement")
    project_file = _validate_file_binding(
        wheel["project_file"],
        "full_checks.wheel.project_file",
        role=True,
    )
    if (
        project_file["path"] != "pyproject.toml"
        or project_file["sha256"] != PYPROJECT_SHA256
        or project_file["role"] != "exact_release_wheel_project_configuration"
    ):
        raise PolicyError("full_checks.wheel.project_file must bind the exact accepted pyproject.toml")
    packages = _validate_unique_strings(wheel["packages"], "full_checks.wheel.packages")
    for index, package in enumerate(packages):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", package):
            raise PolicyError(f"full_checks.wheel.packages[{index}] must be a top-level Python package")
    expected_wheel_packages = sorted(set(source_package_names) - set(source_only_packages))
    if packages != expected_wheel_packages:
        raise PolicyError("wheel packages must exactly match non-source-only package names")
    for package in required_packages:
        package_name = PurePosixPath(package).name
        if package_name in source_only_packages:
            continue
        if package_name not in packages:
            raise PolicyError(f"full_checks.wheel.packages omits {PurePosixPath(package).name}")
    for package in source_package_names:
        if not package.startswith("goalzendo"):
            continue
        source_path = f"src/{package}"
        if source_path not in ruff_paths or source_path not in mypy_paths:
            raise PolicyError(
                f"declared GoalZendo source package {package!r} must appear in both scoped static path lists"
            )
    package_data = _as_list(wheel["package_data"], "full_checks.wheel.package_data")
    package_data_paths: list[str] = []
    for index, value in enumerate(package_data):
        where = f"full_checks.wheel.package_data[{index}]"
        obj = _validate_file_binding(value, where, role=False)
        relative = obj["path"]
        top_level = PurePosixPath(relative).parts[0]
        if top_level not in packages or len(PurePosixPath(relative).parts) < 2:
            raise PolicyError(f"{where}.path must be package-relative wheel data")
        package_data_paths.append(relative)
    if package_data_paths != sorted(set(package_data_paths)):
        raise PolicyError("full_checks.wheel.package_data paths must be unique and sorted")
    required_package_data = "goalzendo_g00f/source_manifest.json"
    if required_package_data not in package_data_paths:
        raise PolicyError(f"full_checks.wheel.package_data omits {required_package_data}")
    scripts = _as_list(wheel["console_scripts"], "full_checks.wheel.console_scripts")
    script_names: list[str] = []
    for index, value in enumerate(scripts):
        where = f"full_checks.wheel.console_scripts[{index}]"
        obj = _as_object(value, where)
        _exact_keys(obj, where, {"entry_point", "name"})
        script_names.append(_as_string(obj["name"], f"{where}.name"))
        entry_point = _as_string(obj["entry_point"], f"{where}.entry_point")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*", entry_point):
            raise PolicyError(f"{where}.entry_point must be module:function")
        entry_module = entry_point.split(":", 1)[0]
        if entry_module.split(".", 1)[0] not in packages:
            raise PolicyError(f"{where}.entry_point refers to an undeclared wheel package")
    if script_names != sorted(set(script_names)):
        raise PolicyError("full_checks.wheel.console_scripts names must be unique and sorted")

    absences = _as_list(policy["known_absences"], "known_absences")
    absence_ids: list[str] = []
    for index, value in enumerate(absences):
        where = f"known_absences[{index}]"
        obj = _as_object(value, where)
        _exact_keys(
            obj,
            where,
            {"absence_id", "evidence_role", "scope", "sha256", "statement", "status", "study_id"},
        )
        absence_ids.append(_as_string(obj["absence_id"], f"{where}.absence_id"))
        _as_string(obj["scope"], f"{where}.scope")
        if obj["status"] not in {"not_release_bound", "not_restored"}:
            raise PolicyError(f"{where}.status is unsupported")
        for key in ("study_id", "evidence_role"):
            if obj[key] is not None:
                _as_string(obj[key], f"{where}.{key}")
        if obj["sha256"] is not None:
            _validate_sha256(obj["sha256"], f"{where}.sha256")
        _as_string(obj["statement"], f"{where}.statement")
    if absence_ids != sorted(set(absence_ids)):
        raise PolicyError("known_absences identifiers must be unique and sorted")
    if {"g00b-execution-source-archive", "goalzendo-raw-run-archives"} - set(absence_ids):
        raise PolicyError("known_absences must record raw artifacts and the G00-B archive")


def load_policy(path: Path = DEFAULT_POLICY) -> dict[str, Any]:
    """Load a strictly shaped policy and require its exact canonical bytes."""

    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise PolicyError(f"cannot read policy {path}: {exc}") from exc
    try:
        text = raw.decode("utf-8")
        value = json.loads(text, object_pairs_hook=_duplicates_rejected)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PolicyError(f"invalid policy JSON: {exc}") from exc
    policy = _as_object(value, "policy")
    _validate_policy_shape(policy)
    if raw != _canonical_json_bytes(policy, pretty=True):
        raise PolicyError("policy bytes are not in the declared canonical JSON form")
    return policy


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_tree_digest(path: Path) -> tuple[int, str]:
    if not path.is_dir():
        raise VerificationError(f"source package directory is missing: {path}")
    records: list[dict[str, str]] = []
    for candidate in sorted(path.rglob("*")):
        relative = candidate.relative_to(path)
        if "__pycache__" in relative.parts or candidate.name == ".DS_Store" or candidate.suffix == ".pyc":
            continue
        if candidate.is_symlink():
            raise VerificationError(f"source package contains a symlink: {candidate}")
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise VerificationError(f"source package contains a non-regular entry: {candidate}")
        records.append({"path": relative.as_posix(), "sha256": _sha256_file(candidate)})
    if not records:
        raise VerificationError(f"source package has no bound files: {path}")
    records.sort(key=lambda item: item["path"])
    return len(records), _sha256_bytes(_canonical_json_bytes(records, pretty=False))


def _resolve_repository_path(root: Path, relative: str) -> Path:
    _validate_relative_path(relative, "runtime path")
    root = root.resolve()
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise VerificationError(f"path escapes repository: {relative}")
    return candidate


def _require_file_hash(root: Path, binding: Mapping[str, Any]) -> Path:
    relative = str(binding["path"])
    path = _resolve_repository_path(root, relative)
    if not path.is_file():
        raise VerificationError(f"missing file: {relative}")
    actual = _sha256_file(path)
    if actual != binding["sha256"]:
        raise VerificationError(
            f"SHA-256 mismatch for {relative}: expected {binding['sha256']}, got {actual}"
        )
    return path


def _load_json_no_duplicates(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_duplicates_rejected)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, PolicyError) as exc:
        raise VerificationError(f"invalid JSON in {path}: {exc}") from exc


def _verify_ledger(root: Path, spec: Mapping[str, Any]) -> str:
    path = _require_file_hash(root, spec)
    ledger = _load_json_no_duplicates(path)
    if type(ledger) is not dict:
        raise VerificationError("study-status ledger must be an object")
    raw = path.read_bytes()
    expected = (json.dumps(ledger, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if raw != expected:
        raise VerificationError("study-status ledger is not in its declared canonical form")
    checked = 0
    for study in ledger.get("studies", []):
        if type(study) is not dict:
            raise VerificationError("ledger study entry must be an object")
        for evidence in study.get("evidence", []):
            if type(evidence) is not dict:
                raise VerificationError("ledger evidence entry must be an object")
            if evidence.get("availability") != "repository":
                continue
            relative = evidence.get("path")
            digest = evidence.get("sha256")
            if type(relative) is not str or type(digest) is not str or not SHA256_RE.fullmatch(digest):
                raise VerificationError(f"malformed repository evidence in study {study.get('study_id')!r}")
            _require_file_hash(root, {"path": relative, "sha256": digest})
            checked += 1
    if checked == 0:
        raise VerificationError("ledger contains no repository evidence bindings")
    return f"ledger and {checked} repository evidence files match"


def _manifest_entries(path: Path) -> list[tuple[str, str]]:
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise VerificationError(f"cannot read checksum manifest {path}: {exc}") from exc
    if not text.endswith("\n") or "\r" in text:
        raise VerificationError(f"checksum manifest must use LF and one terminal LF: {path}")
    lines = text[:-1].split("\n")
    if not lines:
        raise VerificationError(f"checksum manifest is empty: {path}")
    entries: list[tuple[str, str]] = []
    seen: set[str] = set()
    for number, line in enumerate(lines, start=1):
        match = MANIFEST_LINE_RE.fullmatch(line)
        if match is None:
            raise VerificationError(f"non-canonical checksum line {path}:{number}")
        digest, relative = match.groups()
        _validate_relative_path(relative, f"{path}:{number}")
        if relative in seen:
            raise VerificationError(f"duplicate checksum path {relative!r} in {path}")
        seen.add(relative)
        entries.append((digest, relative))
    return entries


def _verify_checksum_manifest(root: Path, spec: Mapping[str, Any]) -> str:
    manifest = _require_file_hash(root, spec)
    base = root if spec["entry_base"] == "repository_root" else manifest.parent
    entries = _manifest_entries(manifest)
    for expected, relative in entries:
        candidate = (base / relative).resolve()
        allowed = root.resolve() if spec["entry_base"] == "repository_root" else base.resolve()
        if not candidate.is_relative_to(allowed):
            raise VerificationError(f"checksum path escapes its declared base: {relative}")
        if not candidate.is_file():
            raise VerificationError(f"checksum target is missing: {candidate}")
        actual = _sha256_file(candidate)
        if actual != expected:
            raise VerificationError(f"checksum mismatch for {candidate}: expected {expected}, got {actual}")
    return f"{len(entries)} checksum entries match"


def _verify_source_packages(root: Path, spec: Mapping[str, Any]) -> str:
    source_root = _resolve_repository_path(root, "src")
    discovered = {
        candidate.name
        for candidate in source_root.iterdir()
        if candidate.is_dir()
        and not candidate.name.endswith(".egg-info")
        and any(
            path.is_file() and "__pycache__" not in path.relative_to(candidate).parts
            for path in candidate.rglob("*.py")
        )
    }
    expected = {item["name"] for item in spec["packages"]}
    if discovered != expected:
        raise VerificationError(
            f"source package set differs: expected {sorted(expected)}, got {sorted(discovered)}"
        )
    total_files = 0
    for item in spec["packages"]:
        count, actual = _source_tree_digest(_resolve_repository_path(root, item["path"]))
        if actual != item["sha256"]:
            raise VerificationError(
                f"source package hash mismatch for {item['name']}: expected {item['sha256']}, got {actual}"
            )
        total_files += count
    return f"{len(expected)} source packages ({total_files} files) match exact component hashes"


def _semantic_json_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _verify_component_manifests(root: Path, spec: Mapping[str, Any]) -> str:
    checked_files = 0
    for binding in spec["component_manifests"]:
        manifest_path = _require_file_hash(root, binding)
        manifest = _load_json_no_duplicates(manifest_path)
        if type(manifest) is not dict:
            raise VerificationError(f"component manifest must be an object: {manifest_path}")
        _exact_keys(
            manifest,
            str(manifest_path),
            {"manifest_digest", "schema", "schema_version", "source_digest", "source_files"},
        )
        canonical = (
            json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        if manifest_path.read_bytes() != canonical:
            raise VerificationError(f"component manifest is not canonical JSON: {manifest_path}")
        if manifest["schema"] != binding["schema"] or manifest["schema_version"] != binding["schema_version"]:
            raise VerificationError(f"component manifest schema changed: {manifest_path}")

        files = _as_object(manifest["source_files"], f"{manifest_path}.source_files")
        if not files:
            raise VerificationError(f"component manifest has no source files: {manifest_path}")
        package_root = _resolve_repository_path(root, f"src/{binding['package']}")
        declared: set[str] = set()
        for relative, raw_digest in files.items():
            normalized = _validate_relative_path(relative, f"{manifest_path}.source_files")
            if normalized == "source_manifest.json":
                raise VerificationError("a component manifest must not hash itself")
            expected = _validate_sha256(raw_digest, f"{manifest_path}.source_files[{relative!r}]")
            candidate = (package_root / normalized).resolve()
            if not candidate.is_relative_to(package_root.resolve()) or candidate.is_symlink():
                raise VerificationError(f"unsafe component-manifest source path: {relative}")
            if not candidate.is_file() or _sha256_file(candidate) != expected:
                raise VerificationError(f"component-manifest source mismatch: {relative}")
            declared.add(normalized)

        discovered: set[str] = set()
        for candidate in package_root.rglob("*"):
            relative_path = candidate.relative_to(package_root)
            if (
                "__pycache__" in relative_path.parts
                or candidate.name == ".DS_Store"
                or candidate.suffix == ".pyc"
            ):
                continue
            if candidate.is_symlink():
                raise VerificationError(f"source package contains a symlink: {candidate}")
            if candidate.is_dir() or relative_path.as_posix() == "source_manifest.json":
                continue
            if not candidate.is_file():
                raise VerificationError(f"source package contains a non-regular entry: {candidate}")
            discovered.add(relative_path.as_posix())
        if declared != discovered:
            raise VerificationError(
                f"component manifest file set differs for {binding['package']}: "
                f"declared={sorted(declared)}, discovered={sorted(discovered)}"
            )

        source_digest = _semantic_json_digest(files)
        body = {key: value for key, value in manifest.items() if key != "manifest_digest"}
        manifest_digest = _semantic_json_digest(body)
        if manifest["source_digest"] != source_digest or binding["source_digest"] != source_digest:
            raise VerificationError(f"component source digest changed: {manifest_path}")
        if manifest["manifest_digest"] != manifest_digest or binding["manifest_digest"] != manifest_digest:
            raise VerificationError(f"component manifest digest changed: {manifest_path}")

        freeze_path = _require_file_hash(
            root,
            {"path": binding["freeze_path"], "sha256": binding["freeze_sha256"]},
        )
        freeze = _load_json_no_duplicates(freeze_path)
        if type(freeze) is not dict or freeze.get("additive_source") != {
            "manifest_path": binding["path"],
            "manifest_sha256": binding["sha256"],
            "source_digest": binding["source_digest"],
        }:
            raise VerificationError("G00-F freeze/additive-source binding changed")
        checked_files += len(declared)
    count = len(spec["component_manifests"])
    label = "component manifest binds" if count == 1 else "component manifests bind"
    return f"{count} {label} {checked_files} exact source files"


def _verify_checkpoint_a_canonical_bundle(root: Path, spec: Mapping[str, Any]) -> str:
    archive_path = _require_file_hash(root, spec["archive"])
    manifest_path = _require_file_hash(root, spec["manifest"])
    freeze_path = _require_file_hash(root, spec["freeze"])
    _require_file_hash(root, spec["builder"])
    _require_file_hash(root, spec["stager"])
    for row in spec["build_input_transport"]:
        _require_file_hash(root, row)
    transport_by_path = {row["path"]: row for row in spec["build_input_transport"]}
    transport_test = transport_by_path["tests/goalzendo/test_g00f_g01_build_input_transport.py"]
    if transport_test["prebuild_audit_sha256"] != CHECKPOINT_A_TRANSPORT_PREBUILD_TEST_AUDIT_SHA256:
        raise VerificationError("checkpoint-A historical prebuild transport-test audit changed")

    expected_parent = _resolve_repository_path(root, CHECKPOINT_A_BUNDLE_ROOT)
    if any(path.parent != expected_parent for path in (archive_path, manifest_path, freeze_path)):
        raise VerificationError("checkpoint-A bundle files do not share their canonical directory")
    if {path.name for path in expected_parent.iterdir()} != {
        CHECKPOINT_A_ARCHIVE_NAME,
        CHECKPOINT_A_MANIFEST_NAME,
        CHECKPOINT_A_FREEZE_NAME,
    }:
        raise VerificationError("checkpoint-A canonical bundle directory inventory changed")
    for path in (archive_path, manifest_path, freeze_path):
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o644
        ):
            raise VerificationError(f"checkpoint-A artifact is not single-link mode 0644: {path}")

    manifest = _load_json_no_duplicates(manifest_path)
    freeze = _load_json_no_duplicates(freeze_path)
    if type(manifest) is not dict or type(freeze) is not dict:
        raise VerificationError("checkpoint-A manifest and freeze must be JSON objects")
    for path, value in ((manifest_path, manifest), (freeze_path, freeze)):
        expected_bytes = (
            json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        if path.read_bytes() != expected_bytes:
            raise VerificationError(f"checkpoint-A JSON is not canonical: {path}")

    _exact_keys(
        manifest,
        str(manifest_path),
        {
            "authorization",
            "bridge_source",
            "compatible_routes",
            "g01_identity",
            "manifest_digest",
            "members",
            "schema",
            "schema_version",
            "source_date_epoch",
            "study_id",
        },
    )
    _exact_keys(
        freeze,
        str(freeze_path),
        {
            "authorization",
            "bridge_source",
            "build_runtime",
            "bundle",
            "compatible_routes",
            "controller_files",
            "freeze_digest",
            "g01_identity",
            "outcomes_seen",
            "schema",
            "schema_version",
            "source_date_epoch",
            "study_id",
            "transaction",
        },
    )
    manifest_body = {key: value for key, value in manifest.items() if key != "manifest_digest"}
    freeze_body = {key: value for key, value in freeze.items() if key != "freeze_digest"}
    if (
        manifest["schema"] != "goalzendo.g00f_g01_bridge_bundle_payload_manifest"
        or freeze["schema"] != "goalzendo.g00f_g01_bridge_bundle_freeze"
        or manifest["schema_version"] != 1
        or freeze["schema_version"] != 1
        or manifest["study_id"] != "g00f_to_g01_checkpoint_a"
        or freeze["study_id"] != "g00f_to_g01_checkpoint_a"
        or manifest["source_date_epoch"] != CHECKPOINT_A_SOURCE_DATE_EPOCH
        or freeze["source_date_epoch"] != CHECKPOINT_A_SOURCE_DATE_EPOCH
        or freeze["outcomes_seen"] is not False
        or manifest["manifest_digest"] != spec["manifest"]["manifest_digest"]
        or manifest["manifest_digest"] != _semantic_json_digest(manifest_body)
        or freeze["freeze_digest"] != spec["freeze"]["freeze_digest"]
        or freeze["freeze_digest"] != _semantic_json_digest(freeze_body)
        or manifest["authorization"] != CHECKPOINT_A_AUTHORIZATION
        or freeze["authorization"] != CHECKPOINT_A_AUTHORIZATION
        or spec["authorization"] != CHECKPOINT_A_AUTHORIZATION
        or manifest["bridge_source"].get("source_digest") != spec["bridge_source_digest"]
        or freeze["bridge_source"] != manifest["bridge_source"]
        or freeze["compatible_routes"] != manifest["compatible_routes"]
        or freeze["g01_identity"] != manifest["g01_identity"]
    ):
        raise VerificationError("checkpoint-A semantic identity or authorization changed")

    bundle = _as_object(freeze["bundle"], "checkpoint-A freeze.bundle")
    if bundle != {
        "archive_name": CHECKPOINT_A_ARCHIVE_NAME,
        "archive_sha256": spec["archive"]["sha256"],
        "archive_total_member_count": 6,
        "manifest_digest": spec["manifest"]["manifest_digest"],
        "manifest_name": CHECKPOINT_A_MANIFEST_NAME,
        "manifest_sha256": spec["manifest"]["sha256"],
        "payload_member_count": 6,
        "selected_route_freeze_copy_count": 1,
    }:
        raise VerificationError("checkpoint-A freeze bundle binding changed")
    if freeze["controller_files"] != {
        "builder": {"path": spec["builder"]["path"], "sha256": spec["builder"]["sha256"]},
        "stager": {"path": spec["stager"]["path"], "sha256": spec["stager"]["sha256"]},
    }:
        raise VerificationError("checkpoint-A accepted controller binding changed")
    if (
        freeze["transaction"] != CHECKPOINT_A_TRANSACTION
        or _semantic_json_digest(freeze["transaction"]) != spec["frozen_transaction_digest"]
    ):
        raise VerificationError("checkpoint-A frozen transaction changed")

    routes = _as_object(freeze["compatible_routes"], "checkpoint-A freeze.compatible_routes")
    if set(routes) != {"h100", "h200"}:
        raise VerificationError("checkpoint-A compatible route set changed")
    for route, raw_binding in routes.items():
        binding = _as_object(raw_binding, f"checkpoint-A compatible_routes.{route}")
        route_freeze = _as_object(binding["freeze"], f"checkpoint-A {route} freeze binding")
        route_freeze_path = _require_file_hash(
            root,
            {"path": route_freeze["path"], "sha256": route_freeze["file_sha256"]},
        )
        route_freeze_value = _load_json_no_duplicates(route_freeze_path)
        if type(route_freeze_value) is not dict:
            raise VerificationError(f"checkpoint-A {route} freeze is not an object")
        route_freeze_body = {
            key: value for key, value in route_freeze_value.items() if key != "freeze_digest"
        }
        if (
            route_freeze_value.get("freeze_digest") != route_freeze["freeze_digest"]
            or _semantic_json_digest(route_freeze_body) != route_freeze["freeze_digest"]
            or route_freeze_value.get("outcomes_seen") is not False
            or route_freeze_value.get("authorization", {}).get("g01_launch_authorized") is not False
        ):
            raise VerificationError(f"checkpoint-A {route} route-freeze identity changed")
        source = _as_object(binding["source_bundle"], f"checkpoint-A {route} source bundle")
        _require_file_hash(root, {"path": source["archive_path"], "sha256": source["archive_sha256"]})
        source_manifest_path = _require_file_hash(
            root,
            {"path": source["manifest_path"], "sha256": source["manifest_sha256"]},
        )
        source_manifest = _load_json_no_duplicates(source_manifest_path)
        if type(source_manifest) is not dict:
            raise VerificationError(f"checkpoint-A {route} source manifest is not an object")
        source_manifest_body = {
            key: value for key, value in source_manifest.items() if key != "manifest_digest"
        }
        if (
            source_manifest.get("manifest_digest") != source["manifest_digest"]
            or _semantic_json_digest(source_manifest_body) != source["manifest_digest"]
            or type(source_manifest.get("members")) is not list
            or len(source_manifest["members"]) != source["member_count"]
        ):
            raise VerificationError(f"checkpoint-A {route} source-manifest identity changed")
        for controller in _as_object(
            binding["controller_files"],
            f"checkpoint-A {route} controllers",
        ).values():
            _require_file_hash(root, controller)
        launcher = _as_object(binding["launcher"], f"checkpoint-A {route} launcher")
        _require_file_hash(
            root,
            {"path": launcher["path"], "sha256": launcher["file_sha256"]},
        )

    runtime = _as_object(freeze["build_runtime"], "checkpoint-A freeze.build_runtime")
    if (
        _semantic_json_digest(runtime) != spec["build_runtime_digest"]
        or runtime.get("trusted_image") != "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
        or runtime.get("python_executable") != "/workspace/.venvs/goalzendo/bin/python"
        or runtime.get("python_implementation") != "CPython"
        or runtime.get("python_version_info") != [3, 12, 3, "final", 0]
        or runtime.get("python_flags")
        != {
            "ignore_environment": True,
            "isolated": True,
            "no_user_site": True,
            "safe_path": True,
        }
        or runtime.get("platform") != {"byteorder": "little", "machine": "x86_64", "system": "Linux"}
        or runtime.get("zlib_build_version") != "1.3"
        or runtime.get("zlib_runtime_version") != "1.3"
        or type(runtime.get("stdlib_modules")) is not dict
        or set(runtime["stdlib_modules"])
        != {"gzip", "json", "json.decoder", "json.encoder", "json.scanner", "tarfile"}
    ):
        raise VerificationError("checkpoint-A canonical build runtime receipt changed")

    rows = manifest["members"]
    if type(rows) is not list or len(rows) != 6:
        raise VerificationError("checkpoint-A manifest must contain exactly six members")
    expected_rows: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(rows):
        row = _as_object(value, f"checkpoint-A members[{index}]")
        _exact_keys(row, f"checkpoint-A members[{index}]", {"bytes", "mode", "path", "sha256", "type"})
        relative = _validate_relative_path(row["path"], f"checkpoint-A members[{index}].path")
        if (
            relative in expected_rows
            or row["type"] != "file"
            or row["mode"] != 0o644
            or type(row["bytes"]) is not int
            or row["bytes"] < 0
            or not SHA256_RE.fullmatch(str(row["sha256"]))
        ):
            raise VerificationError("checkpoint-A manifest member identity changed")
        expected_rows[relative] = row

    try:
        with tarfile.open(fileobj=io.BytesIO(archive_path.read_bytes()), mode="r:gz") as archive:
            members = archive.getmembers()
            if [member.name for member in members] != list(expected_rows):
                raise VerificationError("checkpoint-A archive member inventory or order changed")
            for member in members:
                row = expected_rows[member.name]
                if (
                    member.type != tarfile.REGTYPE
                    or not member.isfile()
                    or member.mode != 0o644
                    or member.size != row["bytes"]
                    or member.uid != 0
                    or member.gid != 0
                    or member.uname != ""
                    or member.gname != ""
                    or member.mtime != CHECKPOINT_A_SOURCE_DATE_EPOCH
                    or member.linkname != ""
                    or member.pax_headers
                ):
                    raise VerificationError(f"checkpoint-A archive metadata changed: {member.name}")
                handle = archive.extractfile(member)
                if handle is None:
                    raise VerificationError(f"checkpoint-A archive member unreadable: {member.name}")
                payload = handle.read()
                if len(payload) != row["bytes"] or _sha256_bytes(payload) != row["sha256"]:
                    raise VerificationError(f"checkpoint-A archive member bytes changed: {member.name}")
                if _resolve_repository_path(root, member.name).read_bytes() != payload:
                    raise VerificationError(f"checkpoint-A payload differs from source: {member.name}")
    except tarfile.TarError as exc:
        raise VerificationError("checkpoint-A archive cannot be parsed") from exc

    status = spec["operational_status"]
    if (
        status["canonical_bundle_built"] is not True
        or status["build_intent_transcript_witness_recorded"] is not True
        or status["build_result_transcript_witness_recorded"] is not True
        or status["disposable_build_pod_created"] is not True
        or status["disposable_build_pod_deleted"] is not True
        or any(
            status[key]
            for key in (
                "bundle_staged",
                "build_intent_witness_durable_independent_registrar",
                "build_intent_witness_reverified_by_release",
                "build_result_witness_durable_independent_registrar",
                "build_result_witness_reverified_by_release",
                "checkpoint_b_global_coordinator_created",
                "coordinator_token_created",
                "direct_g01_launch_authorized",
                "outcomes_seen",
                "route_lock_created",
                "scientific_eligibility_artifact_created",
                "scientific_runpod_pod_created",
                "scientific_runpod_provision_receipt_created",
                "stage_receipt_created",
            )
        )
    ):
        raise VerificationError("checkpoint-A status is not the exact post-build boundary")

    ledger_path = _resolve_repository_path(
        root,
        "reproducibility/goalzendo/study-status-ledger.json",
    )
    ledger = _load_json_no_duplicates(ledger_path)
    if type(ledger) is not dict or type(ledger.get("studies")) is not list:
        raise VerificationError("checkpoint-A ledger cross-binding cannot be read")
    studies = {
        study.get("study_id"): study
        for study in ledger["studies"]
        if type(study) is dict and study.get("study_id") in {"G00-F", "G01"}
    }
    if set(studies) != {"G00-F", "G01"}:
        raise VerificationError("checkpoint-A ledger cross-binding omits G00-F or G01")
    true_fields = {
        "checkpoint_a_build_intent_transcript_witness_recorded",
        "checkpoint_a_build_result_transcript_witness_recorded",
        "checkpoint_a_canonical_bundle_audit_accepted",
        "checkpoint_a_canonical_bundle_built",
        "checkpoint_a_disposable_build_pod_created",
        "checkpoint_a_disposable_build_pod_deleted",
        "checkpoint_a_source_audit_accepted",
    }
    false_fields = {
        "checkpoint_a_build_intent_witness_durable_independent_registrar",
        "checkpoint_a_build_intent_witness_reverified_by_release",
        "checkpoint_a_build_result_witness_durable_independent_registrar",
        "checkpoint_a_build_result_witness_reverified_by_release",
        "checkpoint_a_bundle_staged",
        "checkpoint_a_coordinator_token_created",
        "checkpoint_a_route_lock_created",
        "checkpoint_a_scientific_eligibility_artifact_created",
        "checkpoint_a_stage_receipt_created",
        "checkpoint_b_global_coordinator_created",
        "direct_g01_launch_authorized",
        "g01_launch_authorized",
    }
    for study_id, study in studies.items():
        identities = study.get("identities")
        if type(identities) is not dict:
            raise VerificationError(f"checkpoint-A ledger identities malformed for {study_id}")
        if any(identities.get(field) is not True for field in true_fields) or any(
            identities.get(field) is not False for field in false_fields
        ):
            raise VerificationError(f"checkpoint-A ledger operational status changed for {study_id}")
        for name, identity in (
            ("archive", "checkpoint_a_canonical_bundle_archive_sha256"),
            ("manifest", "checkpoint_a_canonical_bundle_manifest_sha256"),
            ("freeze", "checkpoint_a_canonical_bundle_freeze_sha256"),
        ):
            if identities.get(identity) != spec[name]["sha256"]:
                raise VerificationError(f"checkpoint-A ledger {name} binding changed for {study_id}")
        semantic_bindings = {
            "checkpoint_a_canonical_bundle_manifest_digest": spec["manifest"]["manifest_digest"],
            "checkpoint_a_canonical_bundle_freeze_digest": spec["freeze"]["freeze_digest"],
            "checkpoint_a_canonical_bundle_build_runtime_digest": spec["build_runtime_digest"],
            "checkpoint_a_canonical_bundle_transaction_digest": spec["frozen_transaction_digest"],
        }
        if any(identities.get(field) != expected for field, expected in semantic_bindings.items()):
            raise VerificationError(f"checkpoint-A ledger semantic bundle binding changed for {study_id}")
        if identities.get("checkpoint_a_bridge_source_digest") != spec["bridge_source_digest"]:
            raise VerificationError(f"checkpoint-A ledger bridge binding changed for {study_id}")
    g01_identities = studies["G01"]["identities"]
    if any(
        g01_identities.get(field) is not False
        for field in (
            "outcomes_seen",
            "scientific_runpod_pod_created",
            "scientific_runpod_provision_receipt_created",
        )
    ):
        raise VerificationError("checkpoint-A ledger G01 scientific runtime status changed")
    g00f_identities = studies["G00-F"]["identities"]
    if (
        g00f_identities.get("checkpoint_a_canonical_bundle_member_count") != len(manifest["members"])
        or g00f_identities.get("checkpoint_a_selected_route_freeze_copy_count") != 1
    ):
        raise VerificationError("checkpoint-A ledger bundle or route-copy count changed")
    transport_ledger_fields = {
        "docs/goalzendo/protocols/g00f-g01-build-input-transport.md": (
            "checkpoint_a_build_input_transport_protocol_sha256"
        ),
        "runs/goalzendo/g00f_g01_build_input_transport.py": (
            "checkpoint_a_build_input_transport_controller_sha256"
        ),
        "tests/goalzendo/test_g00f_g01_build_input_transport.py": (
            "checkpoint_a_build_input_transport_test_sha256"
        ),
    }
    if any(
        g00f_identities.get(identity) != transport_by_path[path]["sha256"]
        for path, identity in transport_ledger_fields.items()
    ) or (
        g00f_identities.get("checkpoint_a_build_input_transport_prebuild_test_audit_sha256")
        != transport_test["prebuild_audit_sha256"]
    ):
        raise VerificationError("checkpoint-A ledger transport source or audit binding changed")
    if any(
        g00f_identities.get(field) is not False
        for field in (
            "outcomes_seen",
            "runpod_pod_created",
            "runpod_provision_receipt_created",
            "h200_outcomes_seen",
            "h200_runpod_pod_created",
            "h200_runpod_provision_receipt_created",
        )
    ):
        raise VerificationError("checkpoint-A ledger scientific runtime status changed")
    return "canonical checkpoint-A 6-file bundle, runtime receipt, and nonauthorization match"


def _verify_checkpoint_b_source_milestone(
    root: Path,
    spec: Mapping[str, Any],
    source_packages: Mapping[str, Any],
) -> str:
    accepted_by_path = {row["path"]: row for row in spec["accepted_files"]}
    for row in spec["accepted_files"]:
        _require_file_hash(root, row)

    source_paths = (
        "src/goalzendo_g01_coordinator/__init__.py",
        "src/goalzendo_g01_coordinator/coordinator.py",
        "runs/goalzendo/run_g01_global_coordinator.py",
    )
    source_files = {path: accepted_by_path[path]["sha256"] for path in source_paths}
    if (
        _semantic_json_digest(source_files) != spec["coordinator_source_digest"]
        or spec["coordinator_source_digest"] != CHECKPOINT_B_COORDINATOR_SOURCE_DIGEST
    ):
        raise VerificationError("checkpoint-B accepted coordinator source digest changed")

    package_binding = next(
        (row for row in source_packages["packages"] if row["name"] == "goalzendo_g01_coordinator"),
        None,
    )
    if (
        package_binding is None
        or package_binding["path"] != "src/goalzendo_g01_coordinator"
        or package_binding["sha256"] != spec["package_source_tree_sha256"]
        or "goalzendo_g01_coordinator" not in source_packages["source_only_packages"]
    ):
        raise VerificationError("checkpoint-B package is not bound as the exact source-only tree")
    package_count, package_digest = _source_tree_digest(
        _resolve_repository_path(root, "src/goalzendo_g01_coordinator")
    )
    if package_count != 2 or package_digest != spec["package_source_tree_sha256"]:
        raise VerificationError("checkpoint-B source-only package inventory or digest changed")

    ledger = _load_json_no_duplicates(
        _resolve_repository_path(root, "reproducibility/goalzendo/study-status-ledger.json")
    )
    if type(ledger) is not dict or type(ledger.get("studies")) is not list:
        raise VerificationError("checkpoint-B ledger cross-binding cannot be read")
    studies = {
        study.get("study_id"): study
        for study in ledger["studies"]
        if type(study) is dict and study.get("study_id") in {"G00-F", "G01"}
    }
    if set(studies) != {"G00-F", "G01"}:
        raise VerificationError("checkpoint-B ledger cross-binding omits G00-F or G01")
    true_fields = {
        "checkpoint_b_source_checkpoint_accepted",
        "checkpoint_b_supported_entrypoint_refusal_active",
    }
    false_fields = {
        "checkpoint_b_authenticated_launcher_created",
        "checkpoint_b_complete",
        "checkpoint_b_compute_estimate_resolved",
        "checkpoint_b_four_gpu_provision_operationally_accepted",
        "checkpoint_b_global_coordinator_created",
        "checkpoint_b_measured_pilot_scheduling_resolved",
        "checkpoint_b_provision_transaction_created",
        "checkpoint_b_prospective_lifecycle_operationally_accepted",
        "checkpoint_b_runtime_overlay_created",
        "checkpoint_b_stager_created",
        "checkpoint_b_wall_ceiling_operationally_accepted",
        "g01_launch_authorized",
    }
    for study_id, study in studies.items():
        identities = study.get("identities")
        if type(identities) is not dict:
            raise VerificationError(f"checkpoint-B ledger identities malformed for {study_id}")
        if any(identities.get(field) is not True for field in true_fields) or any(
            identities.get(field) is not False for field in false_fields
        ):
            raise VerificationError(f"checkpoint-B ledger source/refusal status changed for {study_id}")
        if (
            identities.get("checkpoint_b_coordinator_source_digest") != spec["coordinator_source_digest"]
            or identities.get("checkpoint_b_package_source_tree_sha256") != spec["package_source_tree_sha256"]
        ):
            raise VerificationError(f"checkpoint-B ledger source binding changed for {study_id}")

    g01 = studies["G01"]
    execution = g01.get("execution")
    assessment = g01.get("assessment")
    if (
        type(execution) is not dict
        or execution.get("state") != "not_launched"
        or execution.get("planned_runs") != 120
        or execution.get("completed_runs") != 0
        or type(assessment) is not dict
        or assessment.get("scientific_result_state") != "not_started"
    ):
        raise VerificationError("checkpoint-B ledger G01 execution state changed")
    g01_identities = g01["identities"]
    if any(
        g01_identities.get(field) is not False
        for field in (
            "direct_g01_launch_authorized",
            "outcomes_seen",
            "scientific_runpod_pod_created",
            "scientific_runpod_provision_receipt_created",
        )
    ):
        raise VerificationError("checkpoint-B ledger G01 runtime or authorization changed")

    accepted_identity_fields = {
        "docs/goalzendo/protocols/g01-global-coordinator-checkpoint-b.md": ("checkpoint_b_protocol_sha256"),
        "runs/goalzendo/run_g01_global_coordinator.py": "checkpoint_b_entrypoint_sha256",
        "src/goalzendo_g01_coordinator/__init__.py": "checkpoint_b_init_sha256",
        "src/goalzendo_g01_coordinator/coordinator.py": "checkpoint_b_coordinator_sha256",
        "tests/goalzendo/test_g01_global_coordinator.py": "checkpoint_b_test_sha256",
    }
    if any(
        g01_identities.get(identity) != accepted_by_path[path]["sha256"]
        for path, identity in accepted_identity_fields.items()
    ):
        raise VerificationError("checkpoint-B ledger accepted-file binding changed")

    evidence = g01.get("evidence")
    if type(evidence) is not list:
        raise VerificationError("checkpoint-B ledger G01 evidence is malformed")
    accepted_evidence = [row for row in evidence if type(row) is dict and row.get("path") in accepted_by_path]
    evidence_by_path = {row.get("path"): row for row in accepted_evidence}
    if len(accepted_evidence) != len(accepted_by_path) or set(evidence_by_path) != set(accepted_by_path):
        raise VerificationError("checkpoint-B ledger omits accepted source evidence")
    for path, accepted in accepted_by_path.items():
        row = evidence_by_path[path]
        if (
            row.get("availability") != "repository"
            or row.get("role") != accepted["role"]
            or row.get("sha256") != accepted["sha256"]
        ):
            raise VerificationError(f"checkpoint-B ledger evidence binding changed: {path}")

    if spec["authorization"] != CHECKPOINT_B_AUTHORIZATION:
        raise VerificationError("checkpoint-B release authorization changed")
    if spec["operational_status"] != CHECKPOINT_B_OPERATIONAL_STATUS:
        raise VerificationError("checkpoint-B release operational boundary changed")
    return "accepted checkpoint-B five-file source/refusal milestone matches; runtime transaction absent"


def _verify_checkpoint_b_source_capsule_milestone(
    root: Path,
    spec: Mapping[str, Any],
) -> str:
    accepted_by_path = {row["path"]: row for row in spec["accepted_files"]}
    for row in spec["accepted_files"]:
        _require_file_hash(root, row)
    source_files = {path: accepted_by_path[path]["sha256"] for path in sorted(accepted_by_path)}
    if (
        _semantic_json_digest(source_files) != spec["source_capsule_digest"]
        or spec["source_capsule_digest"] != CHECKPOINT_B_SOURCE_CAPSULE_DIGEST
    ):
        raise VerificationError("checkpoint-B source-capsule accepted source digest changed")

    ledger = _load_json_no_duplicates(
        _resolve_repository_path(root, "reproducibility/goalzendo/study-status-ledger.json")
    )
    if type(ledger) is not dict or type(ledger.get("studies")) is not list:
        raise VerificationError("checkpoint-B source-capsule ledger cross-binding cannot be read")
    matching_studies = [
        study for study in ledger["studies"] if type(study) is dict and study.get("study_id") == "G01"
    ]
    if len(matching_studies) != 1 or type(matching_studies[0].get("identities")) is not dict:
        raise VerificationError("checkpoint-B source-capsule requires exactly one G01 ledger identity")
    identities = matching_studies[0]["identities"]
    capsule_authorization_ledger_fields = {
        "canonical_capsule_build_authorized": "checkpoint_b_source_capsule_build_authorized",
        "capsule_stage_authorized": "checkpoint_b_source_capsule_stage_authorized",
        "checkpoint_b_complete": "checkpoint_b_complete",
        "g01_launch_authorized": "g01_launch_authorized",
        "model_execution_authorized": "model_execution_authorized",
    }
    capsule_operational_ledger_fields = {
        "authenticated_launcher_created": "checkpoint_b_source_capsule_authenticated_launcher_created",
        "canonical_capsule_built": "checkpoint_b_source_capsule_built",
        "capsule_staged": "checkpoint_b_source_capsule_staged",
        "fresh_stage_verification_receipt_created": (
            "checkpoint_b_source_capsule_fresh_verification_receipt_created"
        ),
        "g01_artifact_root_created": "checkpoint_b_source_capsule_g01_artifact_root_created",
        "g01_execution_status_root_created": (
            "checkpoint_b_source_capsule_g01_execution_status_root_created"
        ),
        "outcomes_seen": "outcomes_seen",
        "provision_transaction_created": "checkpoint_b_source_capsule_provision_transaction_created",
        "runtime_overlay_created": "checkpoint_b_source_capsule_runtime_overlay_created",
        "source_implementation_accepted": "checkpoint_b_source_capsule_implementation_accepted",
        "source_stage_receipt_created": "checkpoint_b_source_capsule_stage_receipt_created",
    }
    ledger_authorization = {
        key: identities.get(field) for key, field in capsule_authorization_ledger_fields.items()
    }
    ledger_operational = {
        key: identities.get(field) for key, field in capsule_operational_ledger_fields.items()
    }
    if (
        any(
            type(ledger_authorization[key]) is not bool or ledger_authorization[key] is not expected
            for key, expected in CHECKPOINT_B_SOURCE_CAPSULE_AUTHORIZATION.items()
        )
        or any(
            type(ledger_operational[key]) is not bool or ledger_operational[key] is not expected
            for key, expected in CHECKPOINT_B_SOURCE_CAPSULE_OPERATIONAL_STATUS.items()
        )
        or identities.get("scientific_runpod_pod_created") is not False
        or identities.get("scientific_runpod_provision_receipt_created") is not False
    ):
        raise VerificationError("checkpoint-B source-capsule ledger operational status changed")
    if identities.get("checkpoint_b_source_capsule_digest") != spec["source_capsule_digest"]:
        raise VerificationError("checkpoint-B source-capsule ledger digest changed")
    accepted_identity_fields = {
        "docs/goalzendo/protocols/g01-b-source-capsule.md": "checkpoint_b_source_capsule_protocol_sha256",
        "runs/goalzendo/build_g01_b_source_capsule.py": "checkpoint_b_source_capsule_builder_sha256",
        "runs/goalzendo/g01_b_source_capsule_stage.py": "checkpoint_b_source_capsule_stager_sha256",
        "tests/goalzendo/test_g01_b_source_capsule.py": "checkpoint_b_source_capsule_test_sha256",
    }
    if any(
        identities.get(identity) != accepted_by_path[path]["sha256"]
        for path, identity in accepted_identity_fields.items()
    ):
        raise VerificationError("checkpoint-B source-capsule ledger accepted-file binding changed")
    evidence = matching_studies[0].get("evidence")
    if type(evidence) is not list:
        raise VerificationError("checkpoint-B source-capsule ledger evidence is malformed")
    evidence_by_path = {
        row.get("path"): row for row in evidence if type(row) is dict and row.get("path") in accepted_by_path
    }
    accepted_evidence = [row for row in evidence if type(row) is dict and row.get("path") in accepted_by_path]
    if len(accepted_evidence) != len(accepted_by_path) or set(evidence_by_path) != set(accepted_by_path):
        raise VerificationError("checkpoint-B source-capsule ledger omits accepted source evidence")
    for path, accepted in accepted_by_path.items():
        row = evidence_by_path[path]
        if (
            row.get("availability") != "repository"
            or row.get("role") != accepted["role"]
            or row.get("sha256") != accepted["sha256"]
        ):
            raise VerificationError(f"checkpoint-B source-capsule ledger evidence changed: {path}")
    if spec["authorization"] != CHECKPOINT_B_SOURCE_CAPSULE_AUTHORIZATION:
        raise VerificationError("checkpoint-B source-capsule release authorization changed")
    if spec["operational_status"] != CHECKPOINT_B_SOURCE_CAPSULE_OPERATIONAL_STATUS:
        raise VerificationError("checkpoint-B source-capsule release operational boundary changed")

    canonical_paths = (
        root / "reproducibility/goalzendo/g01-b-source-capsule-20260812",
        Path("/workspace/inputs-goalzendo/g01-executions"),
        Path("/workspace/status-goalzendo/g01-preexecution"),
        Path("/workspace/status-goalzendo/g01-executions"),
        Path("/workspace/artifacts-goalzendo/g01-known-law"),
    )
    present = [str(path) for path in canonical_paths if path.exists() or path.is_symlink()]
    if present:
        raise VerificationError(f"checkpoint-B source-capsule canonical path unexpectedly exists: {present}")
    return "accepted checkpoint-B source-capsule implementation matches; canonical transaction absent"


def _verify_g01q_source_milestone(
    root: Path,
    spec: Mapping[str, Any],
    source_packages: Mapping[str, Any],
) -> str:
    accepted_by_path = {row["path"]: row for row in spec["accepted_files"]}
    for row in spec["accepted_files"]:
        _require_file_hash(root, row)

    source_paths = (
        "src/goalzendo_g01_qualification/__init__.py",
        "src/goalzendo_g01_qualification/qualification.py",
        "runs/goalzendo/run_g01_compute_qualification.py",
    )
    source_files = {path: accepted_by_path[path]["sha256"] for path in source_paths}
    if (
        _semantic_json_digest(source_files) != spec["qualification_source_digest"]
        or spec["qualification_source_digest"] != G01Q_QUALIFICATION_SOURCE_DIGEST
    ):
        raise VerificationError("G01Q accepted qualification source digest changed")

    package_binding = next(
        (row for row in source_packages["packages"] if row["name"] == "goalzendo_g01_qualification"),
        None,
    )
    if (
        package_binding is None
        or package_binding["path"] != "src/goalzendo_g01_qualification"
        or package_binding["sha256"] != spec["package_source_tree_sha256"]
        or "goalzendo_g01_qualification" not in source_packages["source_only_packages"]
    ):
        raise VerificationError("G01Q package is not bound as the exact source-only tree")
    package_count, package_digest = _source_tree_digest(
        _resolve_repository_path(root, "src/goalzendo_g01_qualification")
    )
    if package_count != 2 or package_digest != spec["package_source_tree_sha256"]:
        raise VerificationError("G01Q source-only package inventory or digest changed")

    ledger = _load_json_no_duplicates(
        _resolve_repository_path(root, "reproducibility/goalzendo/study-status-ledger.json")
    )
    if type(ledger) is not dict or type(ledger.get("studies")) is not list:
        raise VerificationError("G01Q ledger cross-binding cannot be read")
    matching_studies = [
        study for study in ledger["studies"] if type(study) is dict and study.get("study_id") == "G01"
    ]
    if len(matching_studies) != 1:
        raise VerificationError("G01Q ledger cross-binding requires exactly one G01 study")
    g01 = matching_studies[0]
    identities = g01.get("identities")
    if type(identities) is not dict:
        raise VerificationError("G01Q ledger identities are malformed")
    ledger_authorization = {
        "g01_launch_authorized": identities.get("g01_launch_authorized"),
        "g01_training_authorized": identities.get("g01_training_authorized"),
        "model_execution_authorized": identities.get("model_execution_authorized"),
        "qualification_execution_authorized": identities.get("qualification_execution_authorized"),
    }
    ledger_operational = {key: identities.get(f"g01q_{key}") for key in G01Q_OPERATIONAL_STATUS}
    if any(
        type(ledger_authorization[key]) is not bool or ledger_authorization[key] is not expected
        for key, expected in G01Q_AUTHORIZATION.items()
    ) or any(
        type(ledger_operational[key]) is not bool or ledger_operational[key] is not expected
        for key, expected in G01Q_OPERATIONAL_STATUS.items()
    ):
        raise VerificationError("G01Q ledger source/refusal status changed")
    if (
        identities.get("g01q_qualification_source_digest") != spec["qualification_source_digest"]
        or identities.get("g01q_package_source_tree_sha256") != spec["package_source_tree_sha256"]
    ):
        raise VerificationError("G01Q ledger source binding changed")

    execution = g01.get("execution")
    assessment = g01.get("assessment")
    if (
        type(execution) is not dict
        or execution.get("state") != "not_launched"
        or type(execution.get("planned_runs")) is not int
        or execution.get("planned_runs") != 120
        or type(execution.get("completed_runs")) is not int
        or execution.get("completed_runs") != 0
        or type(assessment) is not dict
        or assessment.get("scientific_result_state") != "not_started"
    ):
        raise VerificationError("G01Q ledger G01 execution state changed")

    accepted_identity_fields = {
        "docs/goalzendo/protocols/g01-compute-qualification.md": "g01q_protocol_sha256",
        "runs/goalzendo/run_g01_compute_qualification.py": "g01q_entrypoint_sha256",
        "src/goalzendo_g01_qualification/__init__.py": "g01q_init_sha256",
        "src/goalzendo_g01_qualification/qualification.py": "g01q_qualification_sha256",
        "tests/goalzendo/test_g01_compute_qualification.py": "g01q_test_sha256",
    }
    if any(
        identities.get(identity) != accepted_by_path[path]["sha256"]
        for path, identity in accepted_identity_fields.items()
    ):
        raise VerificationError("G01Q ledger accepted-file binding changed")

    evidence = g01.get("evidence")
    if type(evidence) is not list:
        raise VerificationError("G01Q ledger G01 evidence is malformed")
    accepted_evidence = [row for row in evidence if type(row) is dict and row.get("path") in accepted_by_path]
    evidence_by_path = {row.get("path"): row for row in accepted_evidence}
    if len(accepted_evidence) != len(accepted_by_path) or set(evidence_by_path) != set(accepted_by_path):
        raise VerificationError("G01Q ledger omits accepted source evidence")
    for path, accepted in accepted_by_path.items():
        row = evidence_by_path[path]
        if (
            row.get("availability") != "repository"
            or row.get("role") != accepted["role"]
            or row.get("sha256") != accepted["sha256"]
        ):
            raise VerificationError(f"G01Q ledger evidence binding changed: {path}")

    if spec["authorization"] != G01Q_AUTHORIZATION:
        raise VerificationError("G01Q release authorization changed")
    if spec["operational_status"] != G01Q_OPERATIONAL_STATUS:
        raise VerificationError("G01Q release operational boundary changed")

    entrypoint = _resolve_repository_path(root, "runs/goalzendo/run_g01_compute_qualification.py")
    with tempfile.TemporaryDirectory(prefix="goalzendo-g01q-release-") as temporary:
        temporary_root = Path(temporary)
        completed = _run(
            [sys.executable, str(entrypoint), "--release-verifier-must-still-refuse"],
            cwd=temporary_root,
        )
        expected_stderr = f"run_g01_compute_qualification: error: {G01Q_REFUSAL}\n"
        if (
            completed.returncode != 2
            or completed.stdout != ""
            or completed.stderr != expected_stderr
            or any(temporary_root.iterdir())
        ):
            raise VerificationError("G01Q wrapper did not produce the exact side-effect-free refusal")
    return "accepted G01Q five-file source/refusal milestone matches; qualification runtime absent"


def _verify_g01q_preprovision_source_milestone(
    root: Path,
    spec: Mapping[str, Any],
    source_packages: Mapping[str, Any],
) -> str:
    accepted_by_path = {row["path"]: row for row in spec["accepted_files"]}
    for row in spec["accepted_files"]:
        _require_file_hash(root, row)

    source_paths = (
        "src/goalzendo_g01_preprovision/__init__.py",
        "src/goalzendo_g01_preprovision/contracts.py",
        "runs/goalzendo/run_g01_preprovision.py",
    )
    source_files = {path: accepted_by_path[path]["sha256"] for path in source_paths}
    if (
        _semantic_json_digest(source_files) != spec["preprovision_source_digest"]
        or spec["preprovision_source_digest"] != G01Q_PREPROVISION_SOURCE_DIGEST
    ):
        raise VerificationError("G01Q preprovision accepted source digest changed")

    package_name = "goalzendo_g01_preprovision"
    package_binding = next(
        (row for row in source_packages["packages"] if row["name"] == package_name),
        None,
    )
    if (
        package_binding is None
        or package_binding["path"] != f"src/{package_name}"
        or package_binding["sha256"] != spec["package_source_tree_sha256"]
        or package_name not in source_packages["source_only_packages"]
    ):
        raise VerificationError("G01Q preprovision package is not bound as the exact source-only tree")
    package_count, package_digest = _source_tree_digest(_resolve_repository_path(root, f"src/{package_name}"))
    if package_count != 2 or package_digest != spec["package_source_tree_sha256"]:
        raise VerificationError("G01Q preprovision source-only package inventory or digest changed")

    ledger = _load_json_no_duplicates(
        _resolve_repository_path(root, "reproducibility/goalzendo/study-status-ledger.json")
    )
    if type(ledger) is not dict or type(ledger.get("studies")) is not list:
        raise VerificationError("G01Q preprovision ledger cross-binding cannot be read")
    matching = [
        study for study in ledger["studies"] if type(study) is dict and study.get("study_id") == "G01"
    ]
    if len(matching) != 1:
        raise VerificationError("G01Q preprovision ledger requires exactly one G01 study")
    g01 = matching[0]
    identities = g01.get("identities")
    if type(identities) is not dict:
        raise VerificationError("G01Q preprovision ledger identities are malformed")
    ledger_authorization = {
        "g01_launch_authorized": identities.get("g01_launch_authorized"),
        "g01_training_authorized": identities.get("g01_training_authorized"),
        "model_execution_authorized": identities.get("model_execution_authorized"),
        "qualification_execution_authorized": identities.get("qualification_execution_authorized"),
    }
    ledger_operational = {
        key: identities.get(f"g01q_preprovision_{key}") for key in G01Q_PREPROVISION_OPERATIONAL_STATUS
    }
    if any(
        type(ledger_authorization[key]) is not bool or ledger_authorization[key] is not expected
        for key, expected in G01Q_PREPROVISION_AUTHORIZATION.items()
    ) or any(
        type(ledger_operational[key]) is not bool or ledger_operational[key] is not expected
        for key, expected in G01Q_PREPROVISION_OPERATIONAL_STATUS.items()
    ):
        raise VerificationError("G01Q preprovision ledger source/refusal status changed")
    if (
        identities.get("g01q_preprovision_source_digest") != spec["preprovision_source_digest"]
        or identities.get("g01q_preprovision_package_source_tree_sha256")
        != spec["package_source_tree_sha256"]
    ):
        raise VerificationError("G01Q preprovision ledger source binding changed")

    accepted_identity_fields = {
        "docs/goalzendo/protocols/g01-preprovision.md": "g01q_preprovision_protocol_sha256",
        "runs/goalzendo/run_g01_preprovision.py": "g01q_preprovision_entrypoint_sha256",
        "src/goalzendo_g01_preprovision/__init__.py": "g01q_preprovision_init_sha256",
        "src/goalzendo_g01_preprovision/contracts.py": "g01q_preprovision_contracts_sha256",
        "tests/goalzendo/test_g01_preprovision.py": "g01q_preprovision_test_sha256",
    }
    if any(
        identities.get(identity) != accepted_by_path[path]["sha256"]
        for path, identity in accepted_identity_fields.items()
    ):
        raise VerificationError("G01Q preprovision ledger accepted-file binding changed")

    evidence = g01.get("evidence")
    if type(evidence) is not list:
        raise VerificationError("G01Q preprovision ledger evidence is malformed")
    rows = [row for row in evidence if type(row) is dict and row.get("path") in accepted_by_path]
    by_path = {row.get("path"): row for row in rows}
    if len(rows) != len(accepted_by_path) or set(by_path) != set(accepted_by_path):
        raise VerificationError("G01Q preprovision ledger omits or duplicates accepted source evidence")
    for path, accepted in accepted_by_path.items():
        row = by_path[path]
        if (
            row.get("availability") != "repository"
            or row.get("role") != accepted["role"]
            or row.get("sha256") != accepted["sha256"]
        ):
            raise VerificationError(f"G01Q preprovision ledger evidence binding changed: {path}")

    if spec["authorization"] != G01Q_PREPROVISION_AUTHORIZATION:
        raise VerificationError("G01Q preprovision release authorization changed")
    if spec["operational_status"] != G01Q_PREPROVISION_OPERATIONAL_STATUS:
        raise VerificationError("G01Q preprovision release operational boundary changed")

    entrypoint = _resolve_repository_path(root, "runs/goalzendo/run_g01_preprovision.py")
    for arguments in ((), ("--release-verifier-must-still-refuse",), ("--help",), ("--version",)):
        with tempfile.TemporaryDirectory(prefix="goalzendo-g01q-preprovision-release-") as temporary:
            temporary_root = Path(temporary)
            completed = _run(
                [sys.executable, "-I", "-S", str(entrypoint), *arguments],
                cwd=temporary_root,
            )
            expected_stderr = f"run_g01_preprovision: error: {G01Q_PREPROVISION_REFUSAL}\n"
            if (
                completed.returncode != 2
                or completed.stdout != ""
                or completed.stderr != expected_stderr
                or any(temporary_root.iterdir())
            ):
                raise VerificationError(
                    "G01Q preprovision wrapper did not produce the exact isolated side-effect-free refusal"
                )
    return "accepted G01Q preprovision five-file source/refusal milestone matches; registrar absent"


def _release_environment() -> dict[str, str]:
    env = os.environ.copy()
    env.update({"LANG": "C", "LC_ALL": "C", "MYPY_FORCE_COLOR": "0", "NO_COLOR": "1", "PYTHONUTF8": "1"})
    return env


def _run(
    command: Sequence[str], *, cwd: Path, timeout: float | None = None
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(command),
            cwd=cwd,
            env=_release_environment(),
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise VerificationError(f"could not run {list(command)!r}: {exc}") from exc


def _trim_output(completed: subprocess.CompletedProcess[str], *, limit: int = 4000) -> str:
    output = (completed.stdout + completed.stderr).strip()
    if len(output) > limit:
        return "..." + output[-limit:]
    return output


def _verify_tool_versions(spec: Mapping[str, Any], root: Path) -> str:
    expected_python = spec["python"]
    actual_python = ".".join(str(part) for part in sys.version_info[:3])
    actual_implementation = sys.implementation.name
    if actual_implementation != "cpython" or expected_python["implementation"] != "CPython":
        raise VerificationError(f"Python implementation mismatch: {actual_implementation}")
    if actual_python != expected_python["version"]:
        raise VerificationError(
            f"Python version mismatch: expected {expected_python['version']}, got {actual_python}"
        )
    checked = 1
    for item in spec["python_distributions"]:
        try:
            actual = importlib.metadata.version(item["name"])
        except importlib.metadata.PackageNotFoundError as exc:
            raise VerificationError(f"missing Python distribution {item['name']}") from exc
        if actual != item["version"]:
            raise VerificationError(
                f"distribution version mismatch for {item['name']}: expected {item['version']}, got {actual}"
            )
        checked += 1
    for item in spec["external_tools"]:
        completed = _run(item["command"], cwd=root)
        if completed.returncode != 0:
            raise VerificationError(
                f"version command failed for {item['name']} with {completed.returncode}: {_trim_output(completed)}"
            )
        output = completed.stdout + completed.stderr
        match = re.search(item["version_pattern"], output, re.MULTILINE)
        actual_external = match.group(1) if match is not None else None
        if actual_external != item["expected"]:
            raise VerificationError(
                f"external tool mismatch for {item['name']}: expected {item['expected']!r}, "
                f"got {actual_external!r}"
            )
        checked += 1
    return f"{checked} exact runtime/tool versions match"


def normalize_ruff(stdout: str, root: Path) -> tuple[int, str]:
    """Return the exact normalized Ruff finding count and SHA-256."""

    try:
        raw = json.loads(stdout, object_pairs_hook=_duplicates_rejected)
    except (json.JSONDecodeError, PolicyError) as exc:
        raise VerificationError(f"Ruff did not emit valid duplicate-free JSON: {exc}") from exc
    if type(raw) is not list:
        raise VerificationError("Ruff JSON output must be an array")
    records: list[dict[str, Any]] = []
    resolved_root = root.resolve()
    for index, item in enumerate(raw):
        if type(item) is not dict:
            raise VerificationError(f"Ruff diagnostic {index} must be an object")
        try:
            filename = Path(item["filename"]).resolve()
            relative = filename.relative_to(resolved_root).as_posix()
            location = item["location"]
            record = {
                "path": relative,
                "row": int(location["row"]),
                "column": int(location["column"]),
                "code": str(item["code"]),
                "message": str(item["message"]),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise VerificationError(f"malformed Ruff diagnostic {index}: {exc}") from exc
        records.append(record)
    records.sort(key=lambda item: (item["path"], item["row"], item["column"], item["code"], item["message"]))
    return len(records), _sha256_bytes(_canonical_json_bytes(records, pretty=False))


def normalize_mypy(stdout: str, root: Path) -> tuple[int, str]:
    """Return the exact normalized mypy error count and SHA-256."""

    records: list[dict[str, Any]] = []
    resolved_root = root.resolve()
    for number, line in enumerate(stdout.splitlines(), start=1):
        try:
            item = json.loads(line, object_pairs_hook=_duplicates_rejected)
        except (json.JSONDecodeError, PolicyError) as exc:
            raise VerificationError(f"invalid mypy JSON on output line {number}: {exc}") from exc
        if type(item) is not dict:
            raise VerificationError(f"mypy diagnostic {number} must be an object")
        if item.get("severity") != "error":
            continue
        try:
            raw_path = _as_string(item["file"], f"mypy diagnostic {number}.file")
            path = Path(raw_path)
            if path.is_absolute():
                relative = path.resolve().relative_to(resolved_root).as_posix()
            else:
                relative = PurePosixPath(raw_path).as_posix()
                _validate_relative_path(relative, f"mypy diagnostic {number}.file")
            record = {
                "path": relative,
                "row": int(item["line"]),
                "column": int(item["column"]),
                "code": item["code"],
                "message": str(item["message"]),
                "severity": str(item["severity"]),
            }
        except (KeyError, TypeError, ValueError, PolicyError) as exc:
            raise VerificationError(f"malformed mypy diagnostic {number}: {exc}") from exc
        records.append(record)
    records.sort(
        key=lambda item: (
            item["path"],
            item["row"],
            item["column"],
            str(item["code"]),
            item["message"],
            item["severity"],
        )
    )
    return len(records), _sha256_bytes(_canonical_json_bytes(records, pretty=False))


def _verify_static_baseline(tool: str, spec: Mapping[str, Any], root: Path) -> str:
    completed = _run(spec["command"], cwd=root)
    if completed.returncode != spec["expected_exit_code"]:
        raise VerificationError(
            f"global {tool} exit changed: expected {spec['expected_exit_code']}, got {completed.returncode}; "
            f"{_trim_output(completed)}"
        )
    if completed.stderr.strip():
        raise VerificationError(f"global {tool} emitted unexpected stderr: {completed.stderr.strip()}")
    count, digest = (
        normalize_ruff(completed.stdout, root) if tool == "ruff" else normalize_mypy(completed.stdout, root)
    )
    if count != spec["diagnostic_count"] or digest != spec["normalized_sha256"]:
        raise VerificationError(
            f"global {tool} debt changed: expected count/hash "
            f"{spec['diagnostic_count']}/{spec['normalized_sha256']}, got {count}/{digest}"
        )
    return f"reviewed legacy debt matches exactly ({count} diagnostics; expected nonzero exit 1)"


def _verify_known_absences(policy: Mapping[str, Any], root: Path) -> str:
    ledger_path = _resolve_repository_path(root, policy["bindings"]["ledger"]["path"])
    ledger = _load_json_no_duplicates(ledger_path)
    checked = 0
    for absence in policy["known_absences"]:
        if absence["study_id"] is None:
            checked += 1
            continue
        matches: list[dict[str, Any]] = []
        for study in ledger.get("studies", []):
            if study.get("study_id") != absence["study_id"]:
                continue
            matches.extend(
                evidence
                for evidence in study.get("evidence", [])
                if evidence.get("role") == absence["evidence_role"]
            )
        if len(matches) != 1:
            raise VerificationError(
                f"known absence {absence['absence_id']} has {len(matches)} matching ledger evidence records"
            )
        evidence = matches[0]
        if evidence.get("availability") != absence["status"]:
            raise VerificationError(
                f"known absence {absence['absence_id']} status changed from {absence['status']} "
                f"to {evidence.get('availability')}"
            )
        if evidence.get("sha256") != absence["sha256"]:
            raise VerificationError(f"known absence {absence['absence_id']} digest changed")
        checked += 1
    return f"{checked} limitations/absences remain explicitly recorded"


def _capture(name: str, operation: Callable[[], str]) -> CheckResult:
    try:
        return CheckResult(name=name, passed=True, detail=operation())
    except (OSError, PolicyError, VerificationError, subprocess.SubprocessError) as exc:
        return CheckResult(name=name, passed=False, detail=str(exc))


def _verify_bound_file(root: Path, binding: Mapping[str, Any]) -> str:
    _require_file_hash(root, binding)
    return f"{binding['role']} hash matches"


def verify_quick(policy: Mapping[str, Any], root: Path = REPOSITORY_ROOT) -> list[CheckResult]:
    """Run the read-only integrity, toolchain, and reviewed-debt checks."""

    bindings = policy["bindings"]
    results = [
        _capture("ledger-and-repository-evidence", lambda: _verify_ledger(root, bindings["ledger"])),
        _capture(
            "checkpoint-a-canonical-bundle",
            lambda: _verify_checkpoint_a_canonical_bundle(
                root,
                bindings["checkpoint_a_canonical_bundle"],
            ),
        ),
        _capture(
            "checkpoint-b-source-refusal",
            lambda: _verify_checkpoint_b_source_milestone(
                root,
                bindings["checkpoint_b_source_milestone"],
                bindings["source_packages"],
            ),
        ),
        _capture(
            "checkpoint-b-source-capsule",
            lambda: _verify_checkpoint_b_source_capsule_milestone(
                root,
                bindings["checkpoint_b_source_capsule_milestone"],
            ),
        ),
        _capture(
            "g01q-source-refusal",
            lambda: _verify_g01q_source_milestone(
                root,
                bindings["g01q_source_milestone"],
                bindings["source_packages"],
            ),
        ),
        _capture(
            "g01q-preprovision-source-refusal",
            lambda: _verify_g01q_preprovision_source_milestone(
                root,
                bindings["g01q_preprovision_source_milestone"],
                bindings["source_packages"],
            ),
        ),
        _capture(
            "source-package-bindings", lambda: _verify_source_packages(root, bindings["source_packages"])
        ),
        _capture(
            "source-component-manifests",
            lambda: _verify_component_manifests(root, bindings["source_packages"]),
        ),
    ]
    for manifest in bindings["checksum_manifests"]:
        results.append(
            _capture(
                f"checksum-manifest:{manifest['path']}",
                partial(_verify_checksum_manifest, root, manifest),
            )
        )
    compact = bindings["compact_inputs"]
    results.append(_capture("compact-paper-inputs", lambda: _verify_checksum_manifest(root, compact)))
    for pdf in bindings["pdfs"]:
        results.append(
            _capture(
                f"pdf:{pdf['path']}",
                partial(_verify_bound_file, root, pdf),
            )
        )
    project_file = policy["full_checks"]["wheel"]["project_file"]
    results.append(
        _capture(
            "wheel-project-source",
            partial(_verify_bound_file, root, project_file),
        )
    )
    results.extend(
        [
            _capture("tool-versions", lambda: _verify_tool_versions(bindings["tool_versions"], root)),
            _capture(
                "legacy-ruff-baseline",
                lambda: _verify_static_baseline("ruff", policy["static_debt_baselines"]["ruff"], root),
            ),
            _capture(
                "legacy-mypy-baseline",
                lambda: _verify_static_baseline("mypy", policy["static_debt_baselines"]["mypy"], root),
            ),
            _capture("known-absences", lambda: _verify_known_absences(policy, root)),
        ]
    )
    return results


def _verify_scoped_static(policy: Mapping[str, Any], root: Path) -> str:
    spec = policy["full_checks"]["goalzendo_static"]
    commands = [
        ["ruff", "check", *spec["ruff_paths"]],
        # The frozen H200 qualification producer carries a small number of
        # compatibility suppressions around dynamically typed Torch/CUDA
        # calls.  Whether those suppressions are reported as unused varies
        # with the installed Torch stubs; substantive mypy diagnostics still
        # remain fatal.
        ["mypy", "--no-warn-unused-ignores", *spec["mypy_paths"]],
    ]
    for command in commands:
        completed = _run(command, cwd=root)
        if completed.returncode != 0:
            raise VerificationError(
                f"scoped GoalZendo static command failed ({completed.returncode}): {command!r}; "
                f"{_trim_output(completed)}"
            )
    return "scoped GoalZendo Ruff and mypy checks pass with zero substantive diagnostics"


def _verify_paper_rebuild(policy: Mapping[str, Any], root: Path) -> str:
    spec = policy["full_checks"]["paper"]
    source = _resolve_repository_path(root, spec["source_dir"])
    if not source.is_dir():
        raise VerificationError(f"paper source directory is missing: {spec['source_dir']}")
    with tempfile.TemporaryDirectory(prefix="goalzendo-paper-release-") as temporary:
        temporary_root = Path(temporary)
        staged = temporary_root / "paper" / "goalzendo-current-results"
        staged.parent.mkdir(parents=True)
        shutil.copytree(source, staged)
        for output in spec["outputs"]:
            candidate = staged / output["path"]
            if candidate.exists():
                candidate.unlink()
        for command in spec["commands"]:
            completed = _run(command, cwd=staged)
            if completed.returncode != 0:
                raise VerificationError(
                    f"clean paper command failed ({completed.returncode}): {command!r}; {_trim_output(completed)}"
                )
        for output in spec["outputs"]:
            candidate = staged / output["path"]
            if not candidate.is_file():
                raise VerificationError(f"paper rebuild omitted {output['path']}")
            actual = _sha256_file(candidate)
            if actual != output["sha256"]:
                raise VerificationError(
                    f"paper rebuild hash mismatch for {output['path']}: expected {output['sha256']}, got {actual}"
                )
    return f"clean temporary paper rebuild produced {len(spec['outputs'])} exact canonical hashes"


def _copy_wheel_source(root: Path, destination: Path, packages: Sequence[str]) -> None:
    for relative in ("pyproject.toml", "README.md", "LICENSE"):
        source = _resolve_repository_path(root, relative)
        if not source.is_file():
            raise VerificationError(f"wheel source file is missing: {relative}")
        shutil.copy2(source, destination / relative)

    def ignored(_directory: str, names: list[str]) -> set[str]:
        return {
            name
            for name in names
            if name in {"__pycache__", ".DS_Store"} or name.endswith(".egg-info") or name.endswith(".pyc")
        }

    source_root = destination / "src"
    source_root.mkdir()
    for package in packages:
        shutil.copytree(
            _resolve_repository_path(root, f"src/{package}"),
            source_root / package,
            ignore=ignored,
        )


def _wheel_top_level_packages(wheel_path: Path) -> set[str]:
    with zipfile.ZipFile(wheel_path) as archive:
        packages = {
            name.split("/", 1)[0]
            for name in archive.namelist()
            if "/" in name and name.endswith((".py", "py.typed")) and ".dist-info/" not in name
        }
    return packages


def _wheel_typed_packages(wheel_path: Path) -> set[str]:
    with zipfile.ZipFile(wheel_path) as archive:
        return {
            name.split("/", 1)[0]
            for name in archive.namelist()
            if name.endswith("/py.typed") and name.count("/") == 1
        }


def _verify_wheel_package_data(wheel_path: Path, bindings: Sequence[Mapping[str, Any]]) -> None:
    with zipfile.ZipFile(wheel_path) as archive:
        names = set(archive.namelist())
        for binding in bindings:
            relative = str(binding["path"])
            if relative not in names:
                raise VerificationError(f"wheel omits bound package data: {relative}")
            actual = _sha256_bytes(archive.read(relative))
            if actual != binding["sha256"]:
                raise VerificationError(
                    f"wheel package-data hash mismatch for {relative}: "
                    f"expected {binding['sha256']}, got {actual}"
                )


def _wheel_console_scripts(wheel_path: Path) -> dict[str, str]:
    with zipfile.ZipFile(wheel_path) as archive:
        candidates = [name for name in archive.namelist() if name.endswith(".dist-info/entry_points.txt")]
        if len(candidates) != 1:
            raise VerificationError(f"wheel contains {len(candidates)} entry_points.txt files")
        parser = _CaseSensitiveConfigParser(interpolation=None)
        try:
            parser.read_string(archive.read(candidates[0]).decode("utf-8"))
        except (UnicodeDecodeError, configparser.Error) as exc:
            raise VerificationError(f"wheel entry_points.txt is invalid: {exc}") from exc
        script_payloads = sorted(name for name in archive.namelist() if ".data/scripts/" in name)
    if parser.defaults() or parser.sections() != ["console_scripts"]:
        raise VerificationError(
            f"wheel entry-point groups differ: expected exactly ['console_scripts'], got {parser.sections()}"
        )
    if script_payloads:
        raise VerificationError(f"wheel contains unexpected .data/scripts payloads: {script_payloads}")
    return {name: value.strip() for name, value in parser.items("console_scripts")}


def _verify_wheel(policy: Mapping[str, Any], root: Path) -> str:
    spec = policy["full_checks"]["wheel"]
    _require_file_hash(root, spec["project_file"])
    expected_packages = set(spec["packages"])
    expected_scripts = {item["name"]: item["entry_point"] for item in spec["console_scripts"]}
    with tempfile.TemporaryDirectory(prefix="goalzendo-wheel-release-") as temporary:
        temporary_root = Path(temporary)
        source = temporary_root / "source"
        wheelhouse = temporary_root / "wheelhouse"
        outside = temporary_root / "outside-checkout"
        source.mkdir()
        wheelhouse.mkdir()
        outside.mkdir()
        _copy_wheel_source(root, source, sorted(expected_packages))
        build = _run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--disable-pip-version-check",
                "--no-deps",
                "--no-index",
                "--no-build-isolation",
                "--wheel-dir",
                str(wheelhouse),
                str(source),
            ],
            cwd=outside,
        )
        if build.returncode != 0:
            raise VerificationError(f"network-free wheel build failed: {_trim_output(build)}")
        wheels = sorted(wheelhouse.glob("*.whl"))
        if len(wheels) != 1:
            raise VerificationError(f"wheel build produced {len(wheels)} wheels")
        wheel = wheels[0]
        actual_packages = _wheel_top_level_packages(wheel)
        if actual_packages != expected_packages:
            raise VerificationError(
                f"wheel package set differs: expected {sorted(expected_packages)}, got {sorted(actual_packages)}"
            )
        typed_packages = _wheel_typed_packages(wheel)
        if typed_packages != expected_packages:
            raise VerificationError(
                f"wheel py.typed marker set differs: expected {sorted(expected_packages)}, "
                f"got {sorted(typed_packages)}"
            )
        _verify_wheel_package_data(wheel, spec["package_data"])
        actual_scripts = _wheel_console_scripts(wheel)
        if actual_scripts != expected_scripts:
            raise VerificationError(
                f"wheel console scripts differ: expected {expected_scripts}, got {actual_scripts}"
            )
        environment = temporary_root / "venv"
        create = _run([sys.executable, "-m", "venv", "--system-site-packages", str(environment)], cwd=outside)
        if create.returncode != 0:
            raise VerificationError(f"temporary venv creation failed: {_trim_output(create)}")
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        scripts_dir = environment / ("Scripts" if os.name == "nt" else "bin")
        install = _run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-deps",
                "--no-index",
                str(wheel),
            ],
            cwd=outside,
        )
        if install.returncode != 0:
            raise VerificationError(f"network-free wheel install failed: {_trim_output(install)}")
        import_program = (
            "import importlib,json,pathlib;"
            f"names={sorted(expected_packages)!r};"
            "mods=[importlib.import_module(n) for n in names];"
            "print(json.dumps({m.__name__:str(pathlib.Path(m.__file__).resolve()) for m in mods},sort_keys=True))"
        )
        imported = _run([str(python), "-I", "-c", import_program], cwd=outside)
        if imported.returncode != 0:
            raise VerificationError(f"installed-package import check failed: {_trim_output(imported)}")
        try:
            locations = json.loads(imported.stdout)
        except json.JSONDecodeError as exc:
            raise VerificationError(f"import check emitted invalid JSON: {exc}") from exc
        environment_root = environment.resolve()
        for package, raw_location in locations.items():
            if not Path(raw_location).resolve().is_relative_to(environment_root):
                raise VerificationError(f"{package} imported outside temporary venv: {raw_location}")
        for name in sorted(expected_scripts):
            executable = scripts_dir / name
            completed = _run([str(executable), "--help"], cwd=outside)
            if completed.returncode != 0:
                raise VerificationError(f"installed CLI {name} --help failed: {_trim_output(completed)}")
    return (
        f"network-free temp wheel contains/imports {len(expected_packages)} packages and runs "
        f"{len(expected_scripts)} installed CLIs outside the checkout; "
        f"{len(spec['package_data'])} package-data files match"
    )


def _verify_pytest(policy: Mapping[str, Any], root: Path) -> str:
    spec = policy["full_checks"]["pytest"]
    command = [sys.executable if part == "{python}" else part for part in spec["command"]]
    completed = _run(command, cwd=root)
    if completed.returncode != spec["expected_exit_code"]:
        raise VerificationError(
            f"full pytest failed ({completed.returncode}): {_trim_output(completed, limit=12000)}"
        )
    summary = _trim_output(completed, limit=1000)
    return f"full pytest exits 0; tail={summary!r}"


def verify_full(policy: Mapping[str, Any], root: Path = REPOSITORY_ROOT) -> list[CheckResult]:
    """Run quick verification plus all expensive release-candidate checks."""

    results = verify_quick(policy, root)
    results.extend(
        [
            _capture("goalzendo-scoped-static", lambda: _verify_scoped_static(policy, root)),
            _capture("deterministic-goalzendo-paper", lambda: _verify_paper_rebuild(policy, root)),
            _capture("isolated-wheel", lambda: _verify_wheel(policy, root)),
            _capture("full-pytest", lambda: _verify_pytest(policy, root)),
        ]
    )
    return results


def _report(policy_path: Path, mode: str, results: Iterable[CheckResult]) -> dict[str, Any]:
    checks = list(results)
    return {
        "authorizing": False,
        "checks": [asdict(item) for item in checks],
        "mode": mode,
        "passed": all(item.passed for item in checks),
        "policy_path": str(policy_path),
        "policy_sha256": _sha256_file(policy_path),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--report", type=Path, help="optionally write the canonical JSON report to this path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        policy = load_policy(args.policy)
    except PolicyError as exc:
        print(f"release policy error: {exc}", file=sys.stderr)
        return 2
    results = verify_quick(policy) if args.mode == "quick" else verify_full(policy)
    report = _report(args.policy, args.mode, results)
    rendered = _canonical_json_bytes(report, pretty=True).decode("utf-8")
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
