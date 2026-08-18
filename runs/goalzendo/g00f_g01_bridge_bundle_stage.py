#!/usr/bin/env python3
"""Authenticate and exclusively stage the checkpoint-A prebootstrap bundle.

This program is stdlib-only and runs after one immutable G00-F source archive
has been bootstrapped, but before the route lock or ITT ledger exists.  Its
production paths are source-enforced.  It writes the exact six common bundle
members, copies the authenticated selected-route freeze as a seventh
non-bundle prerequisite, and writes its append-only receipt last.
"""

from __future__ import annotations

import argparse
import errno
import gzip
import hashlib
import io
import json
import json.decoder
import json.encoder
import json.scanner
import os
import platform
import stat
import sys
import tarfile
import uuid
import zlib
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

Route = Literal["h100", "h200"]

SOURCE_DATE_EPOCH = 1_786_492_800
ARCHIVE_NAME = "g00f-g01-bridge-bundle.tar.gz"
MANIFEST_NAME = "g00f-g01-bridge-bundle-manifest.json"
FREEZE_NAME = "g00f-g01-bridge-bundle-freeze.json"
MANIFEST_SCHEMA = "goalzendo.g00f_g01_bridge_bundle_payload_manifest"
FREEZE_SCHEMA = "goalzendo.g00f_g01_bridge_bundle_freeze"
STAGE_RECEIPT_SCHEMA = "goalzendo.g00f_g01_bridge_bundle_stage_receipt"
BRIDGE_SOURCE_DIGEST = "e0f3df053557cdf06c028a08f847db33b55dac3a7388a52e0fae0d94b07d284f"
CANONICAL_IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
CANONICAL_PYTHON = "/workspace/.venvs/goalzendo/bin/python"
CANONICAL_PYTHON_VERSION = (3, 12, 3, "final", 0)
CANONICAL_ZLIB_VERSION = "1.3"

_CANONICAL_PROGRAM_ROOT = Path("/workspace/status-goalzendo/g00f-executions")
# Tests can replace this private value only while pytest exposes its per-test
# marker.  No CLI argument, environment variable, or public API controls it.
_TEST_ONLY_PROGRAM_ROOT: Path | None = None
_TEST_ONLY_RUNTIME_IDENTITY: Mapping[str, Any] | None = None

PAYLOAD_SHA256: Mapping[str, str] = {
    "configs/goalzendo/g01_known_law.yaml": (
        "ee6d53556189a6b6e25cfbfb204325017a058c63a605653df4c175463e00ce18"
    ),
    "docs/goalzendo/protocols/g01-known-law.md": (
        "428243c3da271bc2d79e16c0fde20d47076eb3837c6740ab733e278cddcdbce9"
    ),
    "runs/goalzendo/run_g01_after_g00f_bridge.py": (
        "bc7703c48d47cc72f21e2f72ee3a4864ff5dd78759c5f692c59ce46ba03ee10b"
    ),
    "src/goalzendo_g00f_g01_bridge/__init__.py": (
        "94d447686b62177ed3f50742424c9a98a6f1c95d921422cf7bf3241689e24503"
    ),
    "src/goalzendo_g00f_g01_bridge/bridge.py": (
        "4ab7c684109b10c2d6ec1469a281fa342e0dac14c5d141a9d97f0cd8e4f575d2"
    ),
    "src/goalzendo_g00f_g01_bridge/cli.py": (
        "350cb69a2923b49aeb4957406b4d86319a2c75f7c7a32e14e53d479b04319801"
    ),
}
BRIDGE_SOURCE_PATHS = (
    "src/goalzendo_g00f_g01_bridge/__init__.py",
    "src/goalzendo_g00f_g01_bridge/bridge.py",
    "src/goalzendo_g00f_g01_bridge/cli.py",
    "runs/goalzendo/run_g01_after_g00f_bridge.py",
)

G01_IDENTITY: Mapping[str, Any] = {
    "config": {
        "path": "configs/goalzendo/g01_known_law.yaml",
        "file_sha256": "ee6d53556189a6b6e25cfbfb204325017a058c63a605653df4c175463e00ce18",
        "canonical_digest": "f9f91978a446a6e750172e0e377bc1b738ccef64b5238992555e1afb6e6bd110",
    },
    "protocol": {
        "path": "docs/goalzendo/protocols/g01-known-law.md",
        "file_sha256": "428243c3da271bc2d79e16c0fde20d47076eb3837c6740ab733e278cddcdbce9",
    },
    "runner": {
        "path": "src/goalzendo/runner.py",
        "file_sha256": "46b55ad4bdd08073e5f89ae101e0862372817b74b590f07ddb8f8c4331e5b9e1",
    },
    "source_fingerprint": "1a8146377b9a9620690025671614edb2dd20d214f4e195da3cf3528809b2c694",
    "guard": "G00_NOT_PASSED__LEARNING_RATES_NOT_FROZEN",
    "guard_signature": "9feaae82edf801aad4bd4a5b16f633be8dfa2dbd41b29601e7b61b564c763464",
    "target_binding_digest": "3315d20a6f9bdae3c5fdaf9567c5bce7b592890d0ebd26e10682002d816bf0c6",
    "plan": {
        "planned_runs": 120,
        "cell_count": 12,
        "seed_count": 10,
        "seeds": [1103, 1129, 1151, 1171, 1201, 1217, 1231, 1277, 1291, 1301],
        "rows_digest": "f51f6b6f574295433dacec8fafe20508526036ef15dce220a1d8c24e8cd6e55f",
        "plan_key_set_digest": ("7fb1bc870c3b93d1d6a5ae6148b83b460c9ee6d246b592f67f908e8080fa8b91"),
    },
    "model_identity": {
        "requested_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "requested_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    },
    "scope": {
        "eligible": "g01_primary_full_model_known_law_only",
        "excluded_experiment_ids": ["g01a", "g01l", "g02"],
    },
}

ROUTES: Mapping[str, Mapping[str, Any]] = {
    "h100": {
        "freeze": {
            "path": "reproducibility/goalzendo/g00f-execution-freeze-20260811/execution-freeze.json",
            "schema": "goalzendo.g00f_execution_freeze",
            "file_sha256": "b2e385488ea7eb7c6f7bfc834c32ff2707aa9f96b718ea62b7ae92c9a4b8df38",
            "freeze_digest": "2b04534b74873a88c73791a17a5c5a7b66a9ddd472b11168034f11517a8d5047",
        },
        "launcher": {
            "path": "runs/goalzendo/run_g00f_frozen_4h100.sh",
            "file_sha256": "576f7dc82f6c4842af872067f72bbe05b55667e4705b9255efbdb29b35493bbe",
        },
        "source_bundle": {
            "archive_path": (
                "reproducibility/goalzendo/g00f-execution-freeze-20260811/g00f-execution-source.tar.gz"
            ),
            "archive_sha256": "6b55d42bd2a75c9fc40f9c67b332c7ad5bd57024c26bd49447ef6bc831dece3a",
            "manifest_path": (
                "reproducibility/goalzendo/g00f-execution-freeze-20260811/g00f-source-bundle-manifest.json"
            ),
            "manifest_sha256": "482f57dc5e0b19a9c1b6ab18a1d63283ce559d288b8a00cc08e51af464d5f8f7",
            "manifest_digest": "02daee56a9f74166464c627a56371efb5a4922259d4fb8cd86869431657032bf",
            "member_count": 46,
        },
        "source_receipt": {
            "schema": "goalzendo.g00f_extracted_source_bundle_receipt",
            "schema_version": 1,
        },
        "controller_files": {
            "bootstrap": {
                "path": "runs/goalzendo/g00f_bundle_bootstrap.py",
                "sha256": "967aa6b68d831bbbfc9757cf28c62bcb4521c8c08bfe5fceb26434946b9ae286",
            },
            "launcher": {
                "path": "runs/goalzendo/run_g00f_frozen_4h100.sh",
                "sha256": "576f7dc82f6c4842af872067f72bbe05b55667e4705b9255efbdb29b35493bbe",
            },
            "watchdog": {
                "path": "runs/goalzendo/g00f_watchdog.py",
                "sha256": "290c4323975f9ba914a31260349b31f200a1f70cbcc17115306ef6b43019e1c5",
            },
        },
    },
    "h200": {
        "freeze": {
            "path": ("reproducibility/goalzendo/g00f-h200-execution-freeze-20260811/execution-freeze.json"),
            "schema": "goalzendo.g00f_h200_execution_freeze",
            "file_sha256": "fd9cf73d124ec566e0589ec0b2e5e4d48a172bca2d04aff51bbb75a58b76b0f9",
            "freeze_digest": "537282f70f7425855807db31681d13a7c07c9402c31170e4189f6fdf5889f52e",
        },
        "launcher": {
            "path": "runs/goalzendo/run_g00f_frozen_4h200.sh",
            "file_sha256": "5bc6dfcab92f9632a1fc3f4a29815d9c6298ee2fc86f47690ce6d1b371ad39f2",
        },
        "source_bundle": {
            "archive_path": (
                "reproducibility/goalzendo/g00f-h200-execution-freeze-20260811/"
                "g00f-h200-execution-source.tar.gz"
            ),
            "archive_sha256": "e08d81bf0345ad343baa02599add8714d9c9e2f180b936113a92f91b5e8cf8ea",
            "manifest_path": (
                "reproducibility/goalzendo/g00f-h200-execution-freeze-20260811/"
                "g00f-h200-source-bundle-manifest.json"
            ),
            "manifest_sha256": "a6b741366d7023a625e59cf3c14f7b49e78117fadb16869d8501ad5d396f746f",
            "manifest_digest": "27473dbfbcdb09e4163deffea2342383d087489d21ab3c2d09b0d4cce0068ec2",
            "member_count": 61,
        },
        "source_receipt": {
            "schema": "goalzendo.g00f_h200_extracted_source_bundle_receipt",
            "schema_version": 1,
        },
        "controller_files": {
            "bootstrap": {
                "path": "runs/goalzendo/g00f_h200_bundle_bootstrap.py",
                "sha256": "4f6b80b0c3e7a2a84f5ae351af9065b53088282112910beb5fb716443bef7ec4",
            },
            "detached_supervisor": {
                "path": "runs/goalzendo/g00f_h200_detached_supervisor.py",
                "sha256": "1565ba556736b6962e9a8691d3fa492ceda598993f9f3d1aec7b88f6aa74b0df",
            },
            "launcher": {
                "path": "runs/goalzendo/run_g00f_frozen_4h200.sh",
                "sha256": "5bc6dfcab92f9632a1fc3f4a29815d9c6298ee2fc86f47690ce6d1b371ad39f2",
            },
            "qualification_controller": {
                "path": "runs/goalzendo/g00f_h200_qualification_controller.py",
                "sha256": "fd835b7b5c7184c7a4ccc81e770ae1f6e6d400c43cf4bcd4d73def93e217f394",
            },
            "qualification_supervisor": {
                "path": "runs/goalzendo/g00f_h200_qualification_supervisor.py",
                "sha256": "93c0103ca2658578a2b90b1ed2ce723dd1c37c88c6f50888f2154b6f7da0dc2a",
            },
            "watchdog": {
                "path": "runs/goalzendo/g00f_h200_watchdog.py",
                "sha256": "533ef5b477723816c61b13ad616f47815be04949415a9db9cc52bca53ac02438",
            },
        },
    },
}

ROUTE_FREEZE_TOP_LEVEL_KEYS: Mapping[str, frozenset[str]] = {
    "h100": frozenset(
        {
            "additive_source",
            "authorization",
            "configurations",
            "controller_files",
            "execution_and_gate_contract",
            "freeze_digest",
            "legacy_goalzendo",
            "outcomes_seen",
            "prior_evidence",
            "runtime",
            "runtime_files",
            "schema",
            "schema_version",
            "source_bundle",
            "study_id",
        }
    ),
    "h200": frozenset(
        {
            "additive_source",
            "authorization",
            "configurations",
            "controller_files",
            "execution_and_gate_contract",
            "freeze_digest",
            "historical_h100_parent",
            "legacy_goalzendo",
            "models",
            "outcomes_seen",
            "prior_evidence",
            "profile_contract",
            "qualification_producer_contract",
            "runtime",
            "runtime_files",
            "schema",
            "schema_version",
            "source_bundle",
            "storage_preflight_contract",
            "study_id",
            "watchdog_supervision_contract",
        }
    ),
}

AUTHORIZATION: Mapping[str, bool] = {
    "g00f_outcomes_seen": False,
    "g01_scientifically_eligible": False,
    "direct_g01_launch_authorized": False,
    "dedicated_global_coordinator_required": True,
}


class StageError(RuntimeError):
    """The prebootstrap stage transaction is invalid or cannot continue."""


def _runtime_identity() -> dict[str, Any]:
    override = _TEST_ONLY_RUNTIME_IDENTITY
    if override is not None:
        if "PYTEST_CURRENT_TEST" not in os.environ:
            raise StageError("test-only runtime override is forbidden outside pytest")
        return dict(override)
    version = sys.version_info
    flags = {
        "isolated": bool(sys.flags.isolated),
        "ignore_environment": bool(sys.flags.ignore_environment),
        "no_user_site": bool(sys.flags.no_user_site),
        "safe_path": bool(getattr(sys.flags, "safe_path", False)),
    }
    executable = Path(sys.executable).absolute()
    resolved_executable = executable.resolve()
    resolved_executable_bytes = _snapshot_regular(
        resolved_executable,
        "resolved stager interpreter",
    )
    if (
        str(executable) != CANONICAL_PYTHON
        or (version.major, version.minor, version.micro, version.releaselevel, version.serial)
        != CANONICAL_PYTHON_VERSION
        or platform.python_implementation() != "CPython"
        or flags
        != {
            "isolated": True,
            "ignore_environment": True,
            "no_user_site": True,
            "safe_path": True,
        }
        or platform.system() != "Linux"
        or platform.machine() != "x86_64"
        or sys.byteorder != "little"
        or zlib.ZLIB_VERSION != CANONICAL_ZLIB_VERSION
        or zlib.ZLIB_RUNTIME_VERSION != CANONICAL_ZLIB_VERSION
    ):
        raise StageError("stager requires the exact isolated route-frozen Python/zlib platform")
    stdlib_modules: dict[str, dict[str, str]] = {}
    for name, module in (
        ("gzip", gzip),
        ("json", json),
        ("json.decoder", json.decoder),
        ("json.encoder", json.encoder),
        ("json.scanner", json.scanner),
        ("tarfile", tarfile),
    ):
        module_path = Path(str(module.__file__)).absolute()
        module_bytes = _snapshot_regular(module_path, f"loaded stdlib module {name}")
        if module_path.resolve() != module_path:
            raise StageError(f"loaded stdlib module is not a single-link regular file: {name}")
        stdlib_modules[name] = {
            "path": str(module_path),
            "sha256": hashlib.sha256(module_bytes).hexdigest(),
        }
    return {
        "trusted_image": CANONICAL_IMAGE,
        "trusted_image_claim_source": "fixed_route_freeze_contract_and_external_trust_assumption",
        "determinism_claim": "deterministic_given_frozen_build_runtime_trust_boundary",
        "python_executable": str(executable),
        "resolved_python_executable": str(resolved_executable),
        "python_executable_sha256": hashlib.sha256(resolved_executable_bytes).hexdigest(),
        "python_implementation": "CPython",
        "python_version": "3.12.3",
        "python_version_info": [3, 12, 3, "final", 0],
        "python_flags": flags,
        "pythonpath_environment_ignored": True,
        "platform": {"system": "Linux", "machine": "x86_64", "byteorder": "little"},
        "zlib_build_version": CANONICAL_ZLIB_VERSION,
        "zlib_runtime_version": CANONICAL_ZLIB_VERSION,
        "stdlib_modules": stdlib_modules,
    }


def _validate_runtime_identity(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    expected_keys = {
        "trusted_image",
        "trusted_image_claim_source",
        "determinism_claim",
        "python_executable",
        "resolved_python_executable",
        "python_executable_sha256",
        "python_implementation",
        "python_version",
        "python_version_info",
        "python_flags",
        "pythonpath_environment_ignored",
        "platform",
        "zlib_build_version",
        "zlib_runtime_version",
        "stdlib_modules",
    }
    resolved = Path(str(value.get("resolved_python_executable", "")))
    sha = str(value.get("python_executable_sha256", ""))
    if (
        set(value) != expected_keys
        or value.get("trusted_image") != CANONICAL_IMAGE
        or value.get("trusted_image_claim_source")
        != "fixed_route_freeze_contract_and_external_trust_assumption"
        or value.get("determinism_claim") != "deterministic_given_frozen_build_runtime_trust_boundary"
        or value.get("python_executable") != CANONICAL_PYTHON
        or not resolved.is_absolute()
        or ".." in resolved.parts
        or len(sha) != 64
        or any(character not in "0123456789abcdef" for character in sha)
        or value.get("python_implementation") != "CPython"
        or value.get("python_version") != "3.12.3"
        or value.get("python_version_info") != [3, 12, 3, "final", 0]
        or value.get("python_flags")
        != {
            "isolated": True,
            "ignore_environment": True,
            "no_user_site": True,
            "safe_path": True,
        }
        or value.get("pythonpath_environment_ignored") is not True
        or value.get("platform") != {"system": "Linux", "machine": "x86_64", "byteorder": "little"}
        or value.get("zlib_build_version") != CANONICAL_ZLIB_VERSION
        or value.get("zlib_runtime_version") != CANONICAL_ZLIB_VERSION
        or not isinstance(value.get("stdlib_modules"), Mapping)
        or set(value["stdlib_modules"])
        != {"gzip", "json", "json.decoder", "json.encoder", "json.scanner", "tarfile"}
    ):
        raise StageError(f"{label} is not the exact route-frozen runtime contract")
    modules = value["stdlib_modules"]
    for name in ("gzip", "json", "json.decoder", "json.encoder", "json.scanner", "tarfile"):
        row = modules[name]
        if (
            not isinstance(row, Mapping)
            or set(row) != {"path", "sha256"}
            or not Path(str(row.get("path", ""))).is_absolute()
            or len(str(row.get("sha256", ""))) != 64
            or any(character not in "0123456789abcdef" for character in str(row.get("sha256", "")))
        ):
            raise StageError(f"{label} stdlib module binding is malformed: {name}")
    return dict(value)


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def require_sha256(value: Any, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise StageError(f"{label} is not a lowercase SHA-256")
    return text


def strict_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    def reject(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise StageError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=reject)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise StageError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise StageError(f"{label} must contain one object")
    return value


def _regular_metadata_at(
    parent: int,
    relative: str,
    label: str,
    mode: int | None = None,
) -> tuple[os.stat_result, bytes]:
    """Open once beneath a held directory, then fstat and read that same FD."""

    safe = safe_relative(relative)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    file_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    current = os.dup(parent)
    try:
        parts = Path(safe).parts
        for part in parts[:-1]:
            child = os.open(part, directory_flags, dir_fd=current)
            os.close(current)
            current = child
        descriptor = os.open(parts[-1], file_flags, dir_fd=current)
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise StageError(f"{label} must be a single-link regular file")
            if mode is not None and stat.S_IMODE(metadata.st_mode) != mode:
                raise StageError(f"{label} mode must be {mode:04o}")
            with os.fdopen(os.dup(descriptor), "rb") as handle:
                payload = handle.read()
            if metadata.st_size != len(payload):
                raise StageError(f"{label} changed while it was read")
            return metadata, payload
        finally:
            os.close(descriptor)
    finally:
        os.close(current)


def _read_regular_at(parent: int, relative: str, label: str, mode: int | None = None) -> bytes:
    return _regular_metadata_at(parent, relative, label, mode)[1]


def _inventory_at(root: int) -> tuple[dict[str, tuple[os.stat_result, bytes]], set[str]]:
    """Return one fd-relative content snapshot while rejecting every non-regular leaf."""

    files: dict[str, tuple[os.stat_result, bytes]] = {}
    directories: set[str] = set()
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)

    def visit(descriptor: int, prefix: str) -> None:
        initial_names = set(os.listdir(descriptor))
        for name in sorted(initial_names):
            if name in {".", ".."} or "/" in name:
                raise StageError("directory inventory contains an unsafe leaf name")
            metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            relative = f"{prefix}/{name}" if prefix else name
            if stat.S_ISDIR(metadata.st_mode):
                child = os.open(name, directory_flags, dir_fd=descriptor)
                try:
                    held = os.fstat(child)
                    if (held.st_dev, held.st_ino) != (metadata.st_dev, metadata.st_ino):
                        raise StageError(f"directory changed during inventory: {relative}")
                    directories.add(relative)
                    visit(child, relative)
                finally:
                    os.close(child)
            elif stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1:
                observed, payload = _regular_metadata_at(
                    descriptor,
                    name,
                    f"inventory member {relative}",
                )
                if (observed.st_dev, observed.st_ino) != (metadata.st_dev, metadata.st_ino):
                    raise StageError(f"file changed during inventory: {relative}")
                files[relative] = (observed, payload)
            else:
                raise StageError(f"inventory contains linked/special/hardlinked entry: {relative}")
        if set(os.listdir(descriptor)) != initial_names:
            raise StageError("directory changed during inventory")

    visit(root, "")
    return files, directories


def _snapshot_regular(path: Path, label: str, mode: int | None = None) -> bytes:
    absolute = path.absolute()
    if not absolute.is_absolute() or not absolute.name:
        raise StageError(f"{label} path is unsafe")
    parent = _open_directory_chain(absolute.parent, f"{label} parent")
    try:
        metadata, payload = _regular_metadata_at(parent, absolute.name, label, mode)
        observed = os.stat(absolute.name, dir_fd=parent, follow_symlinks=False)
        if (metadata.st_dev, metadata.st_ino) != (observed.st_dev, observed.st_ino):
            raise StageError(f"{label} changed during authentication")
        _require_descriptor_matches_path(parent, absolute.parent, f"{label} parent")
        return payload
    finally:
        os.close(parent)


def _snapshot_bundle_artifacts(freeze_path: Path) -> tuple[bytes, bytes, bytes]:
    """Snapshot all external bundle artifacts through one retained parent FD."""

    absolute = freeze_path.absolute()
    if absolute.name != FREEZE_NAME:
        raise StageError(f"bridge-bundle freeze must be named exactly {FREEZE_NAME}")
    parent_path = absolute.parent
    parent = _open_directory_chain(parent_path, "bridge-bundle artifact parent")
    try:
        freeze_bytes = _read_regular_at(parent, FREEZE_NAME, "bridge-bundle freeze", 0o644)
        archive_bytes = _read_regular_at(parent, ARCHIVE_NAME, "bridge-bundle archive", 0o644)
        manifest_bytes = _read_regular_at(parent, MANIFEST_NAME, "bridge-bundle manifest", 0o644)
        _require_descriptor_matches_path(parent, parent_path, "bridge-bundle artifact parent")
        return freeze_bytes, archive_bytes, manifest_bytes
    finally:
        os.close(parent)


def _require_absent_at(parent: int, name: str, label: str) -> None:
    if not name or Path(name).name != name:
        raise StageError(f"{label} leaf name is unsafe")
    try:
        os.stat(name, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError as error:
        raise StageError(f"could not inspect {label}") from error
    raise StageError(f"{label} must be absent; a partial stage is permanent and cannot be retried")


def _require_relative_absent(parent: int, relative: str, label: str) -> None:
    safe = safe_relative(relative)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    current = os.dup(parent)
    try:
        parts = Path(safe).parts
        for part in parts[:-1]:
            try:
                child = os.open(part, directory_flags, dir_fd=current)
            except FileNotFoundError:
                return
            os.close(current)
            current = child
        _require_absent_at(current, parts[-1], label)
    finally:
        os.close(current)


def safe_relative(value: Any) -> str:
    relative = str(value)
    path = Path(relative)
    if (
        not relative
        or path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or path.as_posix() != relative
    ):
        raise StageError(f"unsafe relative path: {relative!r}")
    return relative


def _route(value: Any) -> Route:
    if value not in ROUTES:
        raise StageError("route must be exactly h100 or h200")
    return "h100" if value == "h100" else "h200"


def _uuid4(value: Any) -> str:
    text = str(value)
    try:
        parsed = uuid.UUID(text)
    except (ValueError, AttributeError) as error:
        raise StageError("execution UUID must be canonical UUID4") from error
    if parsed.version != 4 or str(parsed) != text:
        raise StageError("execution UUID must be canonical UUID4")
    return text


def _program_root() -> Path:
    override = _TEST_ONLY_PROGRAM_ROOT
    if override is not None:
        if "PYTEST_CURRENT_TEST" not in os.environ:
            raise StageError("test-only program-root override is forbidden outside pytest")
        return override.absolute()
    return _CANONICAL_PROGRAM_ROOT


def _open_directory_chain(path: Path, label: str) -> int:
    absolute = path.absolute()
    if not absolute.is_absolute() or ".." in absolute.parts:
        raise StageError(f"{label} is not a safe absolute directory path")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    current: int | None = None
    try:
        current = os.open(absolute.anchor, flags)
        for part in absolute.parts[1:]:
            child = os.open(part, flags, dir_fd=current)
            os.close(current)
            current = child
        return current
    except OSError as error:
        if current is not None:
            os.close(current)
        raise StageError(f"{label} cannot be opened component-wise without links") from error


def _require_descriptor_matches_path(descriptor: int, path: Path, label: str) -> None:
    held = os.fstat(descriptor)
    observed = path.stat(follow_symlinks=False)
    if not stat.S_ISDIR(observed.st_mode) or (held.st_dev, held.st_ino) != (
        observed.st_dev,
        observed.st_ino,
    ):
        raise StageError(f"{label} path no longer names the held directory")


def _open_child_directory(parent: int, name: str, label: str) -> int:
    if not name or Path(name).name != name:
        raise StageError(f"{label} leaf name is unsafe")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        return os.open(name, flags, dir_fd=parent)
    except OSError as error:
        raise StageError(f"{label} is not a real child directory") from error


def _open_stage_directories(
    program_root: Path,
    execution_uuid: str,
) -> tuple[int, int, int]:
    program = _open_directory_chain(program_root, "canonical G00-F program root")
    execution: int | None = None
    frozen: int | None = None
    try:
        execution = _open_child_directory(program, execution_uuid, "canonical execution root")
        frozen = _open_child_directory(execution, "frozen-source", "canonical frozen-source root")
        _require_descriptor_matches_path(program, program_root, "canonical program root")
        _require_descriptor_matches_path(
            execution,
            program_root / execution_uuid,
            "canonical execution root",
        )
        _require_descriptor_matches_path(
            frozen,
            program_root / execution_uuid / "frozen-source",
            "canonical frozen-source root",
        )
        return program, execution, frozen
    except BaseException:
        if frozen is not None:
            os.close(frozen)
        if execution is not None:
            os.close(execution)
        os.close(program)
        raise


def _recheck_stage_directories(
    *,
    program: int,
    execution: int,
    frozen: int,
    program_root: Path,
    execution_uuid: str,
) -> None:
    _require_descriptor_matches_path(program, program_root, "canonical program root")
    _require_descriptor_matches_path(
        execution,
        program_root / execution_uuid,
        "canonical execution root",
    )
    _require_descriptor_matches_path(
        frozen,
        program_root / execution_uuid / "frozen-source",
        "canonical frozen-source root",
    )


def _bridge_source() -> dict[str, Any]:
    files = {relative: PAYLOAD_SHA256[relative] for relative in BRIDGE_SOURCE_PATHS}
    if digest(files) != BRIDGE_SOURCE_DIGEST:
        raise StageError("hard-coded bridge source aggregate is internally inconsistent")
    return {"source_files": files, "source_digest": BRIDGE_SOURCE_DIGEST}


def _verify_route_freeze_payload(
    payload_bytes: bytes,
    route: Route,
    expected_sha256: str,
    label: str,
) -> dict[str, Any]:
    expected = ROUTES[route]
    freeze_row = expected["freeze"]
    if not isinstance(freeze_row, Mapping):
        raise StageError("selected-route freeze binding is malformed")
    required_sha = require_sha256(expected_sha256, "externally expected selected-route freeze SHA-256")
    if required_sha != freeze_row["file_sha256"]:
        raise StageError("selected-route freeze SHA differs from the fixed compatible route")
    if hashlib.sha256(payload_bytes).hexdigest() != required_sha:
        raise StageError(f"{label} bytes are unauthenticated")
    payload = strict_json_bytes(payload_bytes, label)
    body = {key: value for key, value in payload.items() if key != "freeze_digest"}
    source = expected["source_bundle"]
    expected_source = {key: value for key, value in source.items() if key != "member_count"}
    if (
        set(payload) != ROUTE_FREEZE_TOP_LEVEL_KEYS[route]
        or payload.get("schema") != freeze_row["schema"]
        or payload.get("schema_version") != 1
        or payload.get("freeze_digest") != freeze_row["freeze_digest"]
        or digest(body) != freeze_row["freeze_digest"]
        or payload.get("outcomes_seen") is not False
        or payload.get("authorization")
        != {
            "g00f_exact_execution_authorized": True,
            "g01_launch_authorized": False,
            "scope": "exact_g00f_frozen_worker_schedule_only",
        }
        or payload.get("source_bundle") != expected_source
        or payload.get("controller_files") != expected["controller_files"]
        or not isinstance(payload.get("runtime"), Mapping)
        or payload["runtime"].get("image") != CANONICAL_IMAGE
        or payload["runtime"].get("python") != "3.12.3"
    ):
        raise StageError("selected-route freeze schema or semantic identity changed")
    return payload


def _verify_route_freeze(path: Path, route: Route, expected_sha256: str) -> tuple[dict[str, Any], bytes]:
    payload = _snapshot_regular(path, "selected-route freeze", 0o644)
    return (
        _verify_route_freeze_payload(payload, route, expected_sha256, "selected-route freeze"),
        payload,
    )


def _verify_bundle(
    *,
    bundle_freeze_path: Path,
    expected_freeze_sha256: str,
    expected_archive_sha256: str,
    expected_manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, bytes], Path, Path]:
    freeze_path = bundle_freeze_path.absolute()
    freeze_bytes, archive_bytes, manifest_bytes = _snapshot_bundle_artifacts(freeze_path)
    freeze_sha = require_sha256(expected_freeze_sha256, "externally expected bundle-freeze SHA-256")
    if hashlib.sha256(freeze_bytes).hexdigest() != freeze_sha:
        raise StageError("bridge-bundle freeze bytes differ from the external SHA-256")
    freeze = strict_json_bytes(freeze_bytes, "bridge-bundle freeze")
    body = {key: value for key, value in freeze.items() if key != "freeze_digest"}
    expected_transaction = {
        "canonical_program_root": "/workspace/status-goalzendo/g00f-executions",
        "frozen_source_directory": "frozen-source",
        "bundle_targets_must_all_be_absent": True,
        "selected_route_freeze_target_must_be_absent": True,
        "exclusive_regular_writes": True,
        "receipt_written_last_outside_uuid_root": True,
        "partial_failure_is_permanent": True,
    }
    if (
        set(freeze)
        != {
            "schema",
            "schema_version",
            "study_id",
            "source_date_epoch",
            "outcomes_seen",
            "build_runtime",
            "bundle",
            "controller_files",
            "bridge_source",
            "g01_identity",
            "compatible_routes",
            "authorization",
            "transaction",
            "freeze_digest",
        }
        or freeze.get("schema") != FREEZE_SCHEMA
        or freeze.get("schema_version") != 1
        or freeze.get("study_id") != "g00f_to_g01_checkpoint_a"
        or freeze.get("source_date_epoch") != SOURCE_DATE_EPOCH
        or freeze.get("outcomes_seen") is not False
        or not isinstance(freeze.get("build_runtime"), Mapping)
        or freeze.get("freeze_digest") != digest(body)
        or freeze.get("bridge_source") != _bridge_source()
        or freeze.get("g01_identity") != G01_IDENTITY
        or freeze.get("compatible_routes") != ROUTES
        or freeze.get("authorization") != AUTHORIZATION
        or freeze.get("transaction") != expected_transaction
    ):
        raise StageError("bridge-bundle freeze schema, digest, or fixed identity changed")
    _validate_runtime_identity(freeze["build_runtime"], "bridge-bundle build runtime")
    bundle = freeze.get("bundle")
    controllers = freeze.get("controller_files")
    if (
        not isinstance(bundle, Mapping)
        or set(bundle)
        != {
            "archive_name",
            "archive_sha256",
            "manifest_name",
            "manifest_sha256",
            "manifest_digest",
            "payload_member_count",
            "archive_total_member_count",
            "selected_route_freeze_copy_count",
        }
        or bundle.get("archive_name") != ARCHIVE_NAME
        or bundle.get("manifest_name") != MANIFEST_NAME
        or bundle.get("payload_member_count") != 6
        or bundle.get("archive_total_member_count") != 6
        or bundle.get("selected_route_freeze_copy_count") != 1
        or not isinstance(controllers, Mapping)
        or set(controllers) != {"builder", "stager"}
    ):
        raise StageError("bridge-bundle freeze control bindings are malformed")
    stager = controllers["stager"]
    builder = controllers["builder"]
    if (
        not isinstance(builder, Mapping)
        or set(builder) != {"path", "sha256"}
        or builder.get("path") != "runs/goalzendo/build_g00f_g01_bridge_bundle.py"
        or require_sha256(builder.get("sha256"), "bridge-bundle builder SHA-256") != builder.get("sha256")
        or not isinstance(stager, Mapping)
        or set(stager) != {"path", "sha256"}
        or stager.get("path") != "runs/goalzendo/g00f_g01_bridge_bundle_stage.py"
        or require_sha256(stager.get("sha256"), "bridge-bundle stager SHA-256") != stager.get("sha256")
    ):
        raise StageError("bridge-bundle builder/stager binding is malformed")
    actual_stager = Path(__file__).absolute()
    actual_stager_bytes = _snapshot_regular(actual_stager, "running bridge-bundle stager")
    if actual_stager.resolve() != actual_stager or hashlib.sha256(
        actual_stager_bytes
    ).hexdigest() != stager.get("sha256"):
        raise StageError("running stager bytes differ from the externally frozen controller")
    archive_path = freeze_path.parent / ARCHIVE_NAME
    manifest_path = freeze_path.parent / MANIFEST_NAME
    archive_sha = require_sha256(expected_archive_sha256, "externally expected bundle-archive SHA-256")
    manifest_sha = require_sha256(
        expected_manifest_sha256,
        "externally expected bundle-manifest SHA-256",
    )
    if (
        archive_sha != bundle.get("archive_sha256")
        or manifest_sha != bundle.get("manifest_sha256")
        or hashlib.sha256(archive_bytes).hexdigest() != archive_sha
        or hashlib.sha256(manifest_bytes).hexdigest() != manifest_sha
    ):
        raise StageError("bridge-bundle archive/manifest differs from external/freeze bindings")
    manifest = strict_json_bytes(manifest_bytes, "bridge-bundle payload manifest")
    manifest_body = {key: value for key, value in manifest.items() if key != "manifest_digest"}
    if (
        set(manifest)
        != {
            "schema",
            "schema_version",
            "study_id",
            "source_date_epoch",
            "members",
            "bridge_source",
            "g01_identity",
            "compatible_routes",
            "authorization",
            "manifest_digest",
        }
        or manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("schema_version") != 1
        or manifest.get("study_id") != "g00f_to_g01_checkpoint_a"
        or manifest.get("source_date_epoch") != SOURCE_DATE_EPOCH
        or manifest.get("manifest_digest") != digest(manifest_body)
        or manifest.get("manifest_digest") != bundle.get("manifest_digest")
        or manifest.get("bridge_source") != _bridge_source()
        or manifest.get("g01_identity") != G01_IDENTITY
        or manifest.get("compatible_routes") != ROUTES
        or manifest.get("authorization") != AUTHORIZATION
    ):
        raise StageError("bridge-bundle payload manifest identity changed")
    rows = manifest.get("members")
    if not isinstance(rows, list) or len(rows) != 6:
        raise StageError("bridge-bundle manifest must contain exactly six payload members")
    expected_rows: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"path", "type", "mode", "bytes", "sha256"}:
            raise StageError("bridge-bundle manifest member is malformed")
        relative = safe_relative(row.get("path"))
        if relative in expected_rows:
            raise StageError("bridge-bundle manifest duplicates a payload member")
        if (
            relative not in PAYLOAD_SHA256
            or row.get("type") != "file"
            or row.get("mode") != 0o644
            or isinstance(row.get("bytes"), bool)
            or not isinstance(row.get("bytes"), int)
            or int(row["bytes"]) < 0
            or row.get("sha256") != PAYLOAD_SHA256[relative]
        ):
            raise StageError("bridge-bundle manifest member identity changed")
        expected_rows[relative] = row
    if set(expected_rows) != set(PAYLOAD_SHA256):
        raise StageError("bridge-bundle manifest member set changed")
    payloads: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
        infos = archive.getmembers()
        names = [safe_relative(info.name) for info in infos]
        if len(names) != 6 or len(names) != len(set(names)) or set(names) != set(PAYLOAD_SHA256):
            raise StageError("bridge-bundle archive has missing, duplicate, or extra members")
        for info in infos:
            if (
                not info.isreg()
                or info.issym()
                or info.islnk()
                or info.uid != 0
                or info.gid != 0
                or info.uname != ""
                or info.gname != ""
                or info.mtime != SOURCE_DATE_EPOCH
                or (info.mode & 0o777) != 0o644
                or bool(info.pax_headers)
            ):
                raise StageError(f"bridge-bundle member metadata changed: {info.name}")
            extracted = archive.extractfile(info)
            if extracted is None:
                raise StageError(f"bridge-bundle member payload is absent: {info.name}")
            row = expected_rows[info.name]
            expected_size = int(row["bytes"])
            if info.size != expected_size:
                raise StageError(f"bridge-bundle member size changed: {info.name}")
            payload = extracted.read(expected_size + 1)
            if len(payload) != expected_size or hashlib.sha256(payload).hexdigest() != row["sha256"]:
                raise StageError(f"bridge-bundle member bytes changed: {info.name}")
            payloads[info.name] = payload
    if set(payloads) != set(PAYLOAD_SHA256):
        raise StageError("bridge-bundle payload extraction was incomplete")
    return freeze, manifest, payloads, archive_path, manifest_path


def _verify_original_source(
    *,
    program_root: Path,
    execution_uuid: str,
    route: Route,
    expected_receipt_sha256: str,
    staged_payloads_present: bool = False,
    program_descriptor: int,
    execution_descriptor: int,
    frozen_descriptor: int,
) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    execution_root = program_root / execution_uuid
    frozen_source = execution_root / "frozen-source"
    execution_names = set(os.listdir(execution_descriptor))
    if execution_names != {
        "frozen-source",
        "source-bundle-receipt.json",
    }:
        raise StageError("execution root is not the pristine post-bootstrap two-entry layout")
    receipt_bytes = _read_regular_at(
        execution_descriptor,
        "source-bundle-receipt.json",
        "original source-bundle receipt",
        0o600,
    )
    receipt_sha = require_sha256(
        expected_receipt_sha256,
        "externally expected original source-bundle receipt SHA-256",
    )
    if hashlib.sha256(receipt_bytes).hexdigest() != receipt_sha:
        raise StageError("original source-bundle receipt differs from the external SHA-256")
    receipt = strict_json_bytes(receipt_bytes, "original source-bundle receipt")
    body = {key: value for key, value in receipt.items() if key != "receipt_digest"}
    expected = ROUTES[route]
    source = expected["source_bundle"]
    required_keys = {
        "schema",
        "schema_version",
        "freeze_file_sha256",
        "freeze_digest",
        "archive_sha256",
        "manifest_sha256",
        "manifest_digest",
        "authenticated_runtime_files",
        "member_count",
        "extracted_root",
        "tar_safety",
        "g01_launch_authorized",
        "receipt_digest",
    }
    if (
        set(receipt) != required_keys
        or receipt.get("schema") != expected["source_receipt"]["schema"]
        or receipt.get("schema_version") != 1
        or receipt.get("freeze_file_sha256") != expected["freeze"]["file_sha256"]
        or receipt.get("freeze_digest") != expected["freeze"]["freeze_digest"]
        or receipt.get("archive_sha256") != source["archive_sha256"]
        or receipt.get("manifest_sha256") != source["manifest_sha256"]
        or receipt.get("manifest_digest") != source["manifest_digest"]
        or receipt.get("member_count") != source["member_count"]
        or receipt.get("extracted_root") != str(frozen_source)
        or receipt.get("g01_launch_authorized") is not False
        or receipt.get("receipt_digest") != digest(body)
        or receipt.get("tar_safety")
        != {
            "only_regular_files": True,
            "no_absolute_or_parent_paths": True,
            "no_links": True,
            "exact_member_set": True,
            "exact_modes": True,
            "exact_bytes": True,
        }
    ):
        raise StageError("original source-bundle receipt route/root/identity changed")
    runtime = receipt.get("authenticated_runtime_files")
    controllers = expected["controller_files"]
    if not isinstance(runtime, Mapping) or set(runtime) != set(controllers):
        raise StageError("original source-bundle receipt controller set changed")
    for role, controller in controllers.items():
        row = runtime[role]
        if (
            not isinstance(row, Mapping)
            or set(row) != {"actual_path", "frozen_path", "sha256"}
            or row.get("frozen_path") != controller["path"]
            or row.get("sha256") != controller["sha256"]
        ):
            raise StageError(f"original source-bundle receipt {role} binding changed")
        actual = Path(str(row.get("actual_path", "")))
        if not actual.is_absolute():
            raise StageError(f"original source-bundle receipt {role} path is not absolute")
        actual_bytes = _snapshot_regular(actual, f"original authenticated {role}")
        if actual.resolve() != actual or hashlib.sha256(actual_bytes).hexdigest() != controller["sha256"]:
            raise StageError(f"original authenticated {role} bytes/path changed")
    observed_files, observed_directories = _inventory_at(frozen_descriptor)
    try:
        embedded_metadata, embedded_manifest_bytes = observed_files["G00F-BUNDLE-MANIFEST.json"]
    except KeyError as error:
        raise StageError("embedded original source manifest is absent") from error
    if stat.S_IMODE(embedded_metadata.st_mode) != 0o644:
        raise StageError("embedded original source manifest mode must be 0644")
    if hashlib.sha256(embedded_manifest_bytes).hexdigest() != source["manifest_sha256"]:
        raise StageError("embedded original source manifest bytes changed")
    manifest = strict_json_bytes(embedded_manifest_bytes, "embedded original source manifest")
    manifest_body = {key: value for key, value in manifest.items() if key != "manifest_digest"}
    rows = manifest.get("members")
    if (
        manifest.get("manifest_digest") != source["manifest_digest"]
        or digest(manifest_body) != source["manifest_digest"]
        or not isinstance(rows, list)
        or len(rows) != source["member_count"]
    ):
        raise StageError("embedded original source manifest identity changed")
    expected_files = {"G00F-BUNDLE-MANIFEST.json"}
    expected_directories: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"path", "type", "mode", "bytes", "sha256"}:
            raise StageError("embedded original source manifest member is malformed")
        relative = safe_relative(row.get("path"))
        if (
            relative in expected_files
            or row.get("type") != "file"
            or row.get("mode")
            not in {
                0o644,
                0o755,
            }
        ):
            raise StageError("embedded original source manifest member identity is invalid")
        expected_files.add(relative)
        parent = Path(relative).parent
        while parent != Path("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent
        try:
            metadata, target_bytes = observed_files[relative]
        except KeyError as error:
            raise StageError(f"original extracted member is absent: {relative}") from error
        if (
            isinstance(row.get("bytes"), bool)
            or not isinstance(row.get("bytes"), int)
            or stat.S_IMODE(metadata.st_mode) != row["mode"]
            or metadata.st_size != row["bytes"]
            or hashlib.sha256(target_bytes).hexdigest()
            != require_sha256(row.get("sha256"), f"source member {relative}")
        ):
            raise StageError(f"original extracted member bytes changed: {relative}")
    if staged_payloads_present:
        additions = dict(PAYLOAD_SHA256)
        selected_freeze = expected["freeze"]
        additions[str(selected_freeze["path"])] = str(selected_freeze["file_sha256"])
        for relative, expected_sha in additions.items():
            if relative in expected_files:
                raise StageError(f"staged path unexpectedly overlaps the original source: {relative}")
            expected_files.add(relative)
            parent = Path(relative).parent
            while parent != Path("."):
                expected_directories.add(parent.as_posix())
                parent = parent.parent
            try:
                metadata, target_bytes = observed_files[relative]
            except KeyError as error:
                raise StageError(f"staged source member is absent: {relative}") from error
            if (
                stat.S_IMODE(metadata.st_mode) != 0o644
                or hashlib.sha256(target_bytes).hexdigest() != expected_sha
            ):
                raise StageError(f"staged source member bytes changed: {relative}")
    if set(observed_files) != expected_files or observed_directories != expected_directories:
        raise StageError("original extracted source inventory is not exact")
    if set(os.listdir(execution_descriptor)) != execution_names:
        raise StageError("execution root changed during source authentication")
    return execution_root, frozen_source, receipt, manifest


def _write_relative(root_descriptor: int, relative: str, payload: bytes, mode: int) -> None:
    safe = safe_relative(relative)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    file_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptors: list[int] = []
    try:
        current = os.dup(root_descriptor)
        descriptors.append(current)
        parts = Path(safe).parts
        for part in parts[:-1]:
            try:
                child = os.open(part, directory_flags, dir_fd=current)
            except FileNotFoundError:
                os.mkdir(part, mode=0o755, dir_fd=current)
                os.fsync(current)
                child = os.open(part, directory_flags, dir_fd=current)
            descriptors.append(child)
            current = child
        try:
            descriptor = os.open(parts[-1], file_flags, 0o600, dir_fd=current)
        except OSError as error:
            if error.errno in {errno.EEXIST, errno.ELOOP}:
                raise StageError(
                    f"exclusive stage target became occupied: {relative}; partial failure is permanent"
                ) from error
            raise
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fchmod(handle.fileno(), mode)
            os.fsync(handle.fileno())
        os.fsync(current)
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _write_receipt(parent: int, name: str, payload: Mapping[str, Any]) -> None:
    if not name or Path(name).name != name:
        raise StageError("stage receipt leaf name is unsafe")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, 0o600, dir_fd=parent)
    except OSError as error:
        raise StageError("stage receipt could not be created exclusively") from error
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(json_bytes(payload))
        handle.flush()
        os.fchmod(handle.fileno(), 0o400)
        os.fsync(handle.fileno())
    os.fsync(parent)


def _parse_utc(value: Any, label: str) -> str:
    text = str(value)
    if not text.endswith("Z"):
        raise StageError(f"{label} must be an explicit UTC timestamp")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as error:
        raise StageError(f"{label} is malformed") from error
    if (
        parsed.tzinfo != timezone.utc
        or parsed.isoformat(timespec="microseconds").replace("+00:00", "Z") != text
    ):
        raise StageError(f"{label} is not canonical microsecond UTC")
    return text


def _verify_stage_with_directories(
    *,
    route: Route,
    execution_uuid: str,
    selected_route_freeze: str | Path,
    expected_selected_route_freeze_sha256: str,
    bundle_freeze: str | Path,
    expected_bundle_freeze_sha256: str,
    expected_bundle_archive_sha256: str,
    expected_bundle_manifest_sha256: str,
    expected_source_receipt_sha256: str,
    expected_stage_receipt_sha256: str,
    program_descriptor: int,
    execution_descriptor: int,
    frozen_descriptor: int,
) -> dict[str, Any]:
    """Freshly replay the exact completed 6+1 stage transaction."""

    runtime = _validate_runtime_identity(_runtime_identity(), "live verifier runtime")
    selected_route = _route(route)
    execution = _uuid4(execution_uuid)
    program_root = _program_root()
    _recheck_stage_directories(
        program=program_descriptor,
        execution=execution_descriptor,
        frozen=frozen_descriptor,
        program_root=program_root,
        execution_uuid=execution,
    )
    for forbidden_name in (
        "g00f-g01-route-lock.json",
        "g00f-g01-scientific-eligibility.json",
        "g00f-g01-coordinator-input.json",
    ):
        _require_absent_at(
            program_descriptor,
            forbidden_name,
            f"pre-lock verification file {forbidden_name}",
        )
    execution_root, frozen_source, source_receipt, source_manifest = _verify_original_source(
        program_root=program_root,
        execution_uuid=execution,
        route=selected_route,
        expected_receipt_sha256=expected_source_receipt_sha256,
        staged_payloads_present=True,
        program_descriptor=program_descriptor,
        execution_descriptor=execution_descriptor,
        frozen_descriptor=frozen_descriptor,
    )
    route_freeze_path = Path(selected_route_freeze)
    route_freeze, route_freeze_bytes = _verify_route_freeze(
        route_freeze_path,
        selected_route,
        expected_selected_route_freeze_sha256,
    )
    bundle, manifest, payloads, archive_path, manifest_path = _verify_bundle(
        bundle_freeze_path=Path(bundle_freeze),
        expected_freeze_sha256=expected_bundle_freeze_sha256,
        expected_archive_sha256=expected_bundle_archive_sha256,
        expected_manifest_sha256=expected_bundle_manifest_sha256,
    )
    selected_freeze_relative = str(ROUTES[selected_route]["freeze"]["path"])
    selected_copy = frozen_source / selected_freeze_relative
    copied_freeze_bytes = _read_regular_at(
        frozen_descriptor,
        selected_freeze_relative,
        "staged selected-route freeze",
        0o644,
    )
    copied_freeze = _verify_route_freeze_payload(
        copied_freeze_bytes,
        selected_route,
        expected_selected_route_freeze_sha256,
        "staged selected-route freeze",
    )
    if copied_freeze != route_freeze or copied_freeze_bytes != route_freeze_bytes:
        raise StageError("selected-route freeze copy no longer equals its authenticated source")
    receipt_path = program_root / f"g00f-g01-bridge-stage-{execution}.json"
    receipt_bytes = _read_regular_at(
        program_descriptor,
        receipt_path.name,
        "canonical bridge-bundle stage receipt",
        0o400,
    )
    stage_sha = require_sha256(
        expected_stage_receipt_sha256,
        "externally expected bridge-bundle stage receipt SHA-256",
    )
    if hashlib.sha256(receipt_bytes).hexdigest() != stage_sha:
        raise StageError("bridge-bundle stage receipt differs from the external SHA-256")
    receipt = strict_json_bytes(receipt_bytes, "bridge-bundle stage receipt")
    body = {key: value for key, value in receipt.items() if key != "receipt_digest"}
    if set(receipt) != {
        "schema",
        "schema_version",
        "study_id",
        "created_at_utc",
        "runtime",
        "route",
        "execution_uuid",
        "execution_root",
        "frozen_source",
        "original_source_receipt",
        "bundle",
        "selected_route_freeze_copy",
        "bridge_source",
        "g01_identity",
        "authorization",
        "transaction",
        "receipt_digest",
    }:
        raise StageError("bridge-bundle stage receipt top-level field set changed")
    if (
        receipt.get("schema") != STAGE_RECEIPT_SCHEMA
        or receipt.get("schema_version") != 1
        or receipt.get("study_id") != "g00f_to_g01_checkpoint_a"
        or receipt.get("route") != selected_route
        or receipt.get("execution_uuid") != execution
        or receipt.get("execution_root") != str(execution_root)
        or receipt.get("frozen_source") != str(frozen_source)
        or receipt.get("runtime") != runtime
        or receipt.get("receipt_digest") != digest(body)
        or receipt.get("bridge_source") != _bridge_source()
        or receipt.get("g01_identity") != G01_IDENTITY
        or receipt.get("authorization") != AUTHORIZATION
        or receipt.get("transaction")
        != {
            "bundle_members_written_exclusively": 6,
            "selected_route_freezes_written_exclusively": 1,
            "receipt_written_last": True,
            "receipt_outside_uuid_execution_root": True,
            "partial_failure_is_permanent": True,
        }
    ):
        raise StageError("bridge-bundle stage receipt fixed identity changed")
    _parse_utc(receipt.get("created_at_utc"), "stage receipt created_at_utc")
    original = receipt.get("original_source_receipt")
    if (
        not isinstance(original, Mapping)
        or set(original) != {"path", "file_sha256", "receipt_digest", "source_manifest_digest"}
        or original.get("path") != str(execution_root / "source-bundle-receipt.json")
        or original.get("file_sha256")
        != require_sha256(expected_source_receipt_sha256, "source receipt SHA-256")
        or original.get("receipt_digest") != source_receipt["receipt_digest"]
        or original.get("source_manifest_digest") != source_manifest["manifest_digest"]
    ):
        raise StageError("bridge-bundle stage receipt original-source binding changed")
    bundle_receipt = receipt.get("bundle")
    if not isinstance(bundle_receipt, Mapping) or set(bundle_receipt) != {
        "freeze_path",
        "freeze_file_sha256",
        "freeze_digest",
        "archive_path",
        "archive_sha256",
        "manifest_path",
        "manifest_sha256",
        "manifest_digest",
        "archive_entry_count",
        "staged_payload_count",
        "members",
    }:
        raise StageError("bridge-bundle stage receipt bundle binding is malformed")
    expected_members = []
    for relative in sorted(PAYLOAD_SHA256):
        payload = payloads[relative]
        expected_members.append(
            {
                "path": relative,
                "bytes": len(payload),
                "mode": 0o644,
                "sha256": PAYLOAD_SHA256[relative],
            }
        )
    if bundle_receipt != {
        "freeze_path": str(Path(bundle_freeze).absolute()),
        "freeze_file_sha256": require_sha256(
            expected_bundle_freeze_sha256,
            "bundle freeze SHA-256",
        ),
        "freeze_digest": bundle["freeze_digest"],
        "archive_path": str(archive_path),
        "archive_sha256": require_sha256(
            expected_bundle_archive_sha256,
            "bundle archive SHA-256",
        ),
        "manifest_path": str(manifest_path),
        "manifest_sha256": require_sha256(
            expected_bundle_manifest_sha256,
            "bundle manifest SHA-256",
        ),
        "manifest_digest": manifest["manifest_digest"],
        "archive_entry_count": 6,
        "staged_payload_count": 6,
        "members": expected_members,
    }:
        raise StageError("bridge-bundle stage receipt exact payload binding changed")
    copied = receipt.get("selected_route_freeze_copy")
    if copied != {
        "count": 1,
        "source_path": str(route_freeze_path.absolute()),
        "target_path": str(selected_copy),
        "file_sha256": ROUTES[selected_route]["freeze"]["file_sha256"],
        "freeze_digest": route_freeze["freeze_digest"],
    }:
        raise StageError("bridge-bundle stage receipt selected-freeze binding changed")
    for forbidden_name in (
        "g00f-g01-route-lock.json",
        "g00f-g01-scientific-eligibility.json",
        "g00f-g01-coordinator-input.json",
    ):
        _require_absent_at(
            program_descriptor,
            forbidden_name,
            f"pre-lock verification file {forbidden_name}",
        )
    _recheck_stage_directories(
        program=program_descriptor,
        execution=execution_descriptor,
        frozen=frozen_descriptor,
        program_root=program_root,
        execution_uuid=execution,
    )
    return {
        "path": str(receipt_path),
        "file_sha256": stage_sha,
        "receipt_digest": receipt["receipt_digest"],
        "verified": True,
        "route": selected_route,
        "execution_uuid": execution,
        "bridge_source_digest": BRIDGE_SOURCE_DIGEST,
        "archive_entry_count": 6,
        "staged_payload_count": 6,
        "selected_route_freeze_copy_count": 1,
        "exact_post_stage_inventory": True,
        "runtime": runtime,
        "g01_scientifically_eligible": False,
        "direct_g01_launch_authorized": False,
        "dedicated_global_coordinator_required": True,
    }


def verify_stage(
    *,
    route: Route,
    execution_uuid: str,
    selected_route_freeze: str | Path,
    expected_selected_route_freeze_sha256: str,
    bundle_freeze: str | Path,
    expected_bundle_freeze_sha256: str,
    expected_bundle_archive_sha256: str,
    expected_bundle_manifest_sha256: str,
    expected_source_receipt_sha256: str,
    expected_stage_receipt_sha256: str,
) -> dict[str, Any]:
    """Freshly replay the exact completed stage through held canonical directories."""

    execution = _uuid4(execution_uuid)
    program_root = _program_root()
    program, execution_root, frozen = _open_stage_directories(program_root, execution)
    try:
        return _verify_stage_with_directories(
            route=route,
            execution_uuid=execution,
            selected_route_freeze=selected_route_freeze,
            expected_selected_route_freeze_sha256=expected_selected_route_freeze_sha256,
            bundle_freeze=bundle_freeze,
            expected_bundle_freeze_sha256=expected_bundle_freeze_sha256,
            expected_bundle_archive_sha256=expected_bundle_archive_sha256,
            expected_bundle_manifest_sha256=expected_bundle_manifest_sha256,
            expected_source_receipt_sha256=expected_source_receipt_sha256,
            expected_stage_receipt_sha256=expected_stage_receipt_sha256,
            program_descriptor=program,
            execution_descriptor=execution_root,
            frozen_descriptor=frozen,
        )
    finally:
        os.close(frozen)
        os.close(execution_root)
        os.close(program)


def _stage_with_directories(
    *,
    route: Route,
    execution_uuid: str,
    selected_route_freeze: str | Path,
    expected_selected_route_freeze_sha256: str,
    bundle_freeze: str | Path,
    expected_bundle_freeze_sha256: str,
    expected_bundle_archive_sha256: str,
    expected_bundle_manifest_sha256: str,
    expected_source_receipt_sha256: str,
    program_descriptor: int,
    execution_descriptor: int,
    frozen_descriptor: int,
) -> dict[str, Any]:
    """Run the irreversible prebootstrap stage transaction."""

    runtime = _validate_runtime_identity(_runtime_identity(), "live stager runtime")
    selected_route = _route(route)
    execution = _uuid4(execution_uuid)
    program_root = _program_root()
    _recheck_stage_directories(
        program=program_descriptor,
        execution=execution_descriptor,
        frozen=frozen_descriptor,
        program_root=program_root,
        execution_uuid=execution,
    )
    execution_root, frozen_source, source_receipt, source_manifest = _verify_original_source(
        program_root=program_root,
        execution_uuid=execution,
        route=selected_route,
        expected_receipt_sha256=expected_source_receipt_sha256,
        program_descriptor=program_descriptor,
        execution_descriptor=execution_descriptor,
        frozen_descriptor=frozen_descriptor,
    )
    route_freeze_path = Path(selected_route_freeze)
    route_freeze, route_freeze_bytes = _verify_route_freeze(
        route_freeze_path,
        selected_route,
        expected_selected_route_freeze_sha256,
    )
    bundle, manifest, payloads, archive_path, manifest_path = _verify_bundle(
        bundle_freeze_path=Path(bundle_freeze),
        expected_freeze_sha256=expected_bundle_freeze_sha256,
        expected_archive_sha256=expected_bundle_archive_sha256,
        expected_manifest_sha256=expected_bundle_manifest_sha256,
    )
    receipt_path = program_root / f"g00f-g01-bridge-stage-{execution}.json"
    _require_absent_at(
        program_descriptor,
        receipt_path.name,
        "canonical bridge-bundle stage receipt",
    )
    for forbidden_name in (
        "g00f-g01-route-lock.json",
        "g00f-g01-scientific-eligibility.json",
        "g00f-g01-coordinator-input.json",
    ):
        _require_absent_at(
            program_descriptor,
            forbidden_name,
            f"pre-stage checkpoint file {forbidden_name}",
        )
    for relative in PAYLOAD_SHA256:
        _require_relative_absent(frozen_descriptor, relative, f"bundle target {relative}")
    selected_freeze_relative = str(ROUTES[selected_route]["freeze"]["path"])
    _require_relative_absent(
        frozen_descriptor,
        selected_freeze_relative,
        "selected-route freeze copy target",
    )
    # Every fallible authentication and absence check is complete before the
    # first write.  There is intentionally no rollback: an interrupted stage
    # leaves no receipt and at least one occupied target, making retry fail.
    _recheck_stage_directories(
        program=program_descriptor,
        execution=execution_descriptor,
        frozen=frozen_descriptor,
        program_root=program_root,
        execution_uuid=execution,
    )
    _write_relative(frozen_descriptor, selected_freeze_relative, route_freeze_bytes, 0o644)
    for relative in sorted(payloads):
        _write_relative(frozen_descriptor, relative, payloads[relative], 0o644)
    staged_members: list[dict[str, Any]] = []
    for relative in sorted(PAYLOAD_SHA256):
        metadata, target_bytes = _regular_metadata_at(
            frozen_descriptor,
            relative,
            f"staged bundle member {relative}",
            0o644,
        )
        if (
            metadata.st_size != len(payloads[relative])
            or hashlib.sha256(target_bytes).hexdigest() != PAYLOAD_SHA256[relative]
        ):
            raise StageError(f"staged bundle member failed post-write authentication: {relative}")
        staged_members.append(
            {
                "path": relative,
                "bytes": metadata.st_size,
                "mode": 0o644,
                "sha256": PAYLOAD_SHA256[relative],
            }
        )
    selected_copy = frozen_source / selected_freeze_relative
    selected_copy_bytes = _read_regular_at(
        frozen_descriptor,
        selected_freeze_relative,
        "staged selected-route freeze",
        0o644,
    )
    if (
        selected_copy_bytes != route_freeze_bytes
        or hashlib.sha256(selected_copy_bytes).hexdigest() != ROUTES[selected_route]["freeze"]["file_sha256"]
    ):
        raise StageError("staged selected-route freeze failed post-write authentication")
    now = datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
    body = {
        "schema": STAGE_RECEIPT_SCHEMA,
        "schema_version": 1,
        "study_id": "g00f_to_g01_checkpoint_a",
        "created_at_utc": now,
        "runtime": runtime,
        "route": selected_route,
        "execution_uuid": execution,
        "execution_root": str(execution_root),
        "frozen_source": str(frozen_source),
        "original_source_receipt": {
            "path": str(execution_root / "source-bundle-receipt.json"),
            "file_sha256": require_sha256(
                expected_source_receipt_sha256,
                "original source receipt SHA-256",
            ),
            "receipt_digest": source_receipt["receipt_digest"],
            "source_manifest_digest": source_manifest["manifest_digest"],
        },
        "bundle": {
            "freeze_path": str(Path(bundle_freeze).absolute()),
            "freeze_file_sha256": require_sha256(
                expected_bundle_freeze_sha256,
                "bundle freeze SHA-256",
            ),
            "freeze_digest": bundle["freeze_digest"],
            "archive_path": str(archive_path),
            "archive_sha256": require_sha256(
                expected_bundle_archive_sha256,
                "bundle archive SHA-256",
            ),
            "manifest_path": str(manifest_path),
            "manifest_sha256": require_sha256(
                expected_bundle_manifest_sha256,
                "bundle manifest SHA-256",
            ),
            "manifest_digest": manifest["manifest_digest"],
            "archive_entry_count": 6,
            "staged_payload_count": 6,
            "members": staged_members,
        },
        "selected_route_freeze_copy": {
            "count": 1,
            "source_path": str(route_freeze_path.absolute()),
            "target_path": str(selected_copy),
            "file_sha256": ROUTES[selected_route]["freeze"]["file_sha256"],
            "freeze_digest": route_freeze["freeze_digest"],
        },
        "bridge_source": _bridge_source(),
        "g01_identity": G01_IDENTITY,
        "authorization": AUTHORIZATION,
        "transaction": {
            "bundle_members_written_exclusively": 6,
            "selected_route_freezes_written_exclusively": 1,
            "receipt_written_last": True,
            "receipt_outside_uuid_execution_root": True,
            "partial_failure_is_permanent": True,
        },
    }
    receipt = {**body, "receipt_digest": digest(body)}
    _recheck_stage_directories(
        program=program_descriptor,
        execution=execution_descriptor,
        frozen=frozen_descriptor,
        program_root=program_root,
        execution_uuid=execution,
    )
    for forbidden_name in (
        "g00f-g01-route-lock.json",
        "g00f-g01-scientific-eligibility.json",
        "g00f-g01-coordinator-input.json",
    ):
        _require_absent_at(
            program_descriptor,
            forbidden_name,
            f"pre-receipt checkpoint file {forbidden_name}",
        )
    _write_receipt(program_descriptor, receipt_path.name, receipt)
    _recheck_stage_directories(
        program=program_descriptor,
        execution=execution_descriptor,
        frozen=frozen_descriptor,
        program_root=program_root,
        execution_uuid=execution,
    )
    receipt_file_sha256 = hashlib.sha256(json_bytes(receipt)).hexdigest()
    return _verify_stage_with_directories(
        route=selected_route,
        execution_uuid=execution,
        selected_route_freeze=route_freeze_path,
        expected_selected_route_freeze_sha256=expected_selected_route_freeze_sha256,
        bundle_freeze=bundle_freeze,
        expected_bundle_freeze_sha256=expected_bundle_freeze_sha256,
        expected_bundle_archive_sha256=expected_bundle_archive_sha256,
        expected_bundle_manifest_sha256=expected_bundle_manifest_sha256,
        expected_source_receipt_sha256=expected_source_receipt_sha256,
        expected_stage_receipt_sha256=receipt_file_sha256,
        program_descriptor=program_descriptor,
        execution_descriptor=execution_descriptor,
        frozen_descriptor=frozen_descriptor,
    )


def stage(
    *,
    route: Route,
    execution_uuid: str,
    selected_route_freeze: str | Path,
    expected_selected_route_freeze_sha256: str,
    bundle_freeze: str | Path,
    expected_bundle_freeze_sha256: str,
    expected_bundle_archive_sha256: str,
    expected_bundle_manifest_sha256: str,
    expected_source_receipt_sha256: str,
) -> dict[str, Any]:
    """Run the irreversible transaction through retained canonical directory FDs."""

    execution = _uuid4(execution_uuid)
    program_root = _program_root()
    program, execution_root, frozen = _open_stage_directories(program_root, execution)
    try:
        return _stage_with_directories(
            route=route,
            execution_uuid=execution,
            selected_route_freeze=selected_route_freeze,
            expected_selected_route_freeze_sha256=expected_selected_route_freeze_sha256,
            bundle_freeze=bundle_freeze,
            expected_bundle_freeze_sha256=expected_bundle_freeze_sha256,
            expected_bundle_archive_sha256=expected_bundle_archive_sha256,
            expected_bundle_manifest_sha256=expected_bundle_manifest_sha256,
            expected_source_receipt_sha256=expected_source_receipt_sha256,
            program_descriptor=program,
            execution_descriptor=execution_root,
            frozen_descriptor=frozen,
        )
    finally:
        os.close(frozen)
        os.close(execution_root)
        os.close(program)


def main() -> int:
    parser = argparse.ArgumentParser(prog="g00f_g01_bridge_bundle_stage")
    commands = parser.add_subparsers(dest="command", required=True)

    def add_common(target: argparse.ArgumentParser) -> None:
        target.add_argument("--route", choices=("h100", "h200"), required=True)
        target.add_argument("--execution-uuid", required=True)
        target.add_argument("--selected-route-freeze", type=Path, required=True)
        target.add_argument("--expected-selected-route-freeze-sha256", required=True)
        target.add_argument("--bundle-freeze", type=Path, required=True)
        target.add_argument("--expected-bundle-freeze-sha256", required=True)
        target.add_argument("--expected-bundle-archive-sha256", required=True)
        target.add_argument("--expected-bundle-manifest-sha256", required=True)
        target.add_argument("--expected-source-receipt-sha256", required=True)

    stage_parser = commands.add_parser("stage")
    add_common(stage_parser)
    verify_parser = commands.add_parser("verify-stage")
    add_common(verify_parser)
    verify_parser.add_argument("--expected-stage-receipt-sha256", required=True)
    arguments = parser.parse_args()
    try:
        common = {
            "route": arguments.route,
            "execution_uuid": arguments.execution_uuid,
            "selected_route_freeze": arguments.selected_route_freeze,
            "expected_selected_route_freeze_sha256": (arguments.expected_selected_route_freeze_sha256),
            "bundle_freeze": arguments.bundle_freeze,
            "expected_bundle_freeze_sha256": arguments.expected_bundle_freeze_sha256,
            "expected_bundle_archive_sha256": arguments.expected_bundle_archive_sha256,
            "expected_bundle_manifest_sha256": arguments.expected_bundle_manifest_sha256,
            "expected_source_receipt_sha256": arguments.expected_source_receipt_sha256,
        }
        if arguments.command == "stage":
            result = stage(**common)
        else:
            result = verify_stage(
                **common,
                expected_stage_receipt_sha256=arguments.expected_stage_receipt_sha256,
            )
    except (StageError, OSError, tarfile.TarError) as error:
        print(f"g00f_g01_bridge_bundle_stage: error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
