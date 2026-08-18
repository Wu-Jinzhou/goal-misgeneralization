#!/usr/bin/env python3
"""Build the runtime-bound checkpoint-A G00-F -> G01 bridge bundle.

This builder is deliberately stdlib-only and conditionally deterministic
under its frozen-build-runtime trust boundary.  It creates an exact six-file
payload archive, a separate exact external payload manifest, and an
acyclic external freeze that binds the archive, manifest, builder, and
stager.  The canonical output is prospective and must not be built until the
source has passed independent review.
"""

from __future__ import annotations

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
import zlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SOURCE_DATE_EPOCH = 1_786_492_800
DEFAULT_OUTPUT = "reproducibility/goalzendo/g00f-g01-bridge-prebootstrap-20260812"
ARCHIVE_NAME = "g00f-g01-bridge-bundle.tar.gz"
MANIFEST_NAME = "g00f-g01-bridge-bundle-manifest.json"
FREEZE_NAME = "g00f-g01-bridge-bundle-freeze.json"

MANIFEST_SCHEMA = "goalzendo.g00f_g01_bridge_bundle_payload_manifest"
FREEZE_SCHEMA = "goalzendo.g00f_g01_bridge_bundle_freeze"
BRIDGE_SOURCE_DIGEST = "e0f3df053557cdf06c028a08f847db33b55dac3a7388a52e0fae0d94b07d284f"
CANONICAL_IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
CANONICAL_PYTHON = "/workspace/.venvs/goalzendo/bin/python"
CANONICAL_PYTHON_VERSION = (3, 12, 3, "final", 0)
CANONICAL_ZLIB_VERSION = "1.3"
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


class BuildError(RuntimeError):
    """The prospective bundle cannot be constructed from the supplied tree."""


def _runtime_identity() -> dict[str, Any]:
    override = _TEST_ONLY_RUNTIME_IDENTITY
    if override is not None:
        if "PYTEST_CURRENT_TEST" not in os.environ:
            raise BuildError("test-only runtime override is forbidden outside pytest")
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
    _, resolved_executable_bytes = _snapshot_regular(
        resolved_executable,
        "resolved builder interpreter",
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
        raise BuildError("builder requires the exact isolated route-frozen Python/zlib platform")
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
        _, module_bytes = _snapshot_regular(module_path, f"loaded stdlib module {name}")
        if module_path.resolve() != module_path:
            raise BuildError(f"loaded stdlib module is not a single-link regular file: {name}")
        stdlib_modules[name] = {"path": str(module_path), "sha256": sha256_bytes(module_bytes)}
    return {
        "trusted_image": CANONICAL_IMAGE,
        "trusted_image_claim_source": "fixed_route_freeze_contract_and_external_trust_assumption",
        "determinism_claim": "deterministic_given_frozen_build_runtime_trust_boundary",
        "python_executable": str(executable),
        "resolved_python_executable": str(resolved_executable),
        "python_executable_sha256": sha256_bytes(resolved_executable_bytes),
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
    if (
        set(value) != expected_keys
        or value.get("trusted_image") != CANONICAL_IMAGE
        or value.get("trusted_image_claim_source")
        != "fixed_route_freeze_contract_and_external_trust_assumption"
        or value.get("determinism_claim") != "deterministic_given_frozen_build_runtime_trust_boundary"
        or value.get("python_executable") != CANONICAL_PYTHON
        or not resolved.is_absolute()
        or ".." in resolved.parts
        or len(str(value.get("python_executable_sha256", ""))) != 64
        or any(
            character not in "0123456789abcdef"
            for character in str(value.get("python_executable_sha256", ""))
        )
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
        raise BuildError(f"{label} is not the exact route-frozen build runtime contract")
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
            raise BuildError(f"{label} stdlib module binding is malformed: {name}")
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


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def strict_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    def reject(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise BuildError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=reject)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise BuildError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise BuildError(f"{label} must contain one object")
    return value


def _repo(start: Path) -> Path:
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (
            (candidate / "pyproject.toml").is_file()
            and (candidate / "src/goalzendo").is_dir()
            and (candidate / "src/goalzendo_g00f_g01_bridge").is_dir()
        ):
            return candidate
    raise BuildError("could not locate the GoalZendo repository")


def _safe_relative(value: str) -> tuple[str, ...]:
    path = Path(value)
    if not value or path.is_absolute() or "." in path.parts or ".." in path.parts or path.as_posix() != value:
        raise BuildError(f"unsafe repository-relative path: {value!r}")
    return path.parts


def _read_regular_at(
    parent: int,
    relative: str,
    label: str,
    *,
    expected_sha256: str | None = None,
    mode: int | None = None,
) -> tuple[os.stat_result, bytes]:
    """Open once beneath a held directory, then fstat and read that same FD."""

    parts = _safe_relative(relative)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    file_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    current = os.dup(parent)
    try:
        for part in parts[:-1]:
            child = os.open(part, directory_flags, dir_fd=current)
            os.close(current)
            current = child
        descriptor = os.open(parts[-1], file_flags, dir_fd=current)
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise BuildError(f"{label} is not a single-link regular file")
            if mode is not None and stat.S_IMODE(metadata.st_mode) != mode:
                raise BuildError(f"{label} mode must be {mode:04o}")
            with os.fdopen(os.dup(descriptor), "rb") as handle:
                payload = handle.read()
        finally:
            os.close(descriptor)
    except OSError as error:
        raise BuildError(f"{label} cannot be opened without link traversal") from error
    finally:
        os.close(current)
    if expected_sha256 is not None and sha256_bytes(payload) != expected_sha256:
        raise BuildError(f"{label} bytes changed")
    return metadata, payload


def _snapshot_regular(path: Path, label: str) -> tuple[os.stat_result, bytes]:
    absolute = path.absolute()
    if not absolute.is_absolute() or not absolute.name or ".." in absolute.parts:
        raise BuildError(f"{label} path is unsafe")
    parent = _open_directory_chain(absolute.parent, f"{label} parent")
    try:
        metadata, payload = _read_regular_at(parent, absolute.name, label)
        observed = os.stat(absolute.name, dir_fd=parent, follow_symlinks=False)
        if (metadata.st_dev, metadata.st_ino) != (observed.st_dev, observed.st_ino):
            raise BuildError(f"{label} changed during authentication")
        _require_descriptor_matches_path(parent, absolute.parent, f"{label} parent")
        return metadata, payload
    finally:
        os.close(parent)


def _require_descriptor_matches_path(descriptor: int, path: Path, label: str) -> None:
    held = os.fstat(descriptor)
    observed = path.stat(follow_symlinks=False)
    if not stat.S_ISDIR(observed.st_mode) or (held.st_dev, held.st_ino) != (
        observed.st_dev,
        observed.st_ino,
    ):
        raise BuildError(f"{label} path no longer names the held directory")


def _bridge_source() -> dict[str, Any]:
    files = {relative: PAYLOAD_SHA256[relative] for relative in BRIDGE_SOURCE_PATHS}
    if digest(files) != BRIDGE_SOURCE_DIGEST:
        raise BuildError("hard-coded bridge source aggregate is internally inconsistent")
    return {"source_files": files, "source_digest": BRIDGE_SOURCE_DIGEST}


def _verify_route(repo: int, route: str, expected: Mapping[str, Any]) -> None:
    freeze_row = expected["freeze"]
    if not isinstance(freeze_row, Mapping):
        raise BuildError(f"{route} freeze binding is malformed")
    _, freeze_bytes = _read_regular_at(
        repo,
        str(freeze_row["path"]),
        f"{route} execution freeze",
        expected_sha256=str(freeze_row["file_sha256"]),
    )
    freeze = strict_json_bytes(freeze_bytes, f"{route} execution freeze")
    freeze_body = {key: value for key, value in freeze.items() if key != "freeze_digest"}
    source = expected["source_bundle"]
    expected_source = {key: value for key, value in source.items() if key != "member_count"}
    if (
        set(freeze) != ROUTE_FREEZE_TOP_LEVEL_KEYS[route]
        or freeze.get("schema") != freeze_row["schema"]
        or freeze.get("schema_version") != 1
        or freeze.get("freeze_digest") != freeze_row["freeze_digest"]
        or digest(freeze_body) != freeze_row["freeze_digest"]
        or freeze.get("outcomes_seen") is not False
        or freeze.get("authorization")
        != {
            "g00f_exact_execution_authorized": True,
            "g01_launch_authorized": False,
            "scope": "exact_g00f_frozen_worker_schedule_only",
        }
        or freeze.get("source_bundle") != expected_source
    ):
        raise BuildError(f"{route} execution freeze identity changed")
    controllers = freeze.get("controller_files")
    if controllers != expected["controller_files"]:
        raise BuildError(f"{route} controller bindings changed")
    if freeze.get("source_bundle") != {
        key: value for key, value in expected["source_bundle"].items() if key != "member_count"
    }:
        raise BuildError(f"{route} source-bundle binding changed")
    for row in expected["controller_files"].values():
        _read_regular_at(
            repo,
            str(row["path"]),
            f"{route} controller {row['path']}",
            expected_sha256=str(row["sha256"]),
        )
    _read_regular_at(
        repo,
        str(source["archive_path"]),
        f"{route} source archive",
        expected_sha256=str(source["archive_sha256"]),
    )
    _, manifest_bytes = _read_regular_at(
        repo,
        str(source["manifest_path"]),
        f"{route} source payload manifest",
        expected_sha256=str(source["manifest_sha256"]),
    )
    manifest = strict_json_bytes(manifest_bytes, f"{route} source payload manifest")
    manifest_body = {key: value for key, value in manifest.items() if key != "manifest_digest"}
    members = manifest.get("members")
    if (
        manifest.get("manifest_digest") != source["manifest_digest"]
        or digest(manifest_body) != source["manifest_digest"]
        or not isinstance(members, list)
        or len(members) != source["member_count"]
    ):
        raise BuildError(f"{route} source payload manifest changed")


def _manifest(repo: int) -> tuple[dict[str, Any], dict[str, bytes]]:
    payloads: dict[str, bytes] = {}
    sizes: dict[str, int] = {}
    for relative, expected_sha in PAYLOAD_SHA256.items():
        metadata, payload = _read_regular_at(
            repo,
            relative,
            f"required payload source {relative}",
            expected_sha256=expected_sha,
        )
        payloads[relative] = payload
        sizes[relative] = metadata.st_size
    for route, route_binding in ROUTES.items():
        _verify_route(repo, route, route_binding)
    members = [
        {
            "path": relative,
            "type": "file",
            "mode": 0o644,
            "bytes": sizes[relative],
            "sha256": PAYLOAD_SHA256[relative],
        }
        for relative in sorted(PAYLOAD_SHA256)
    ]
    body = {
        "schema": MANIFEST_SCHEMA,
        "schema_version": 1,
        "study_id": "g00f_to_g01_checkpoint_a",
        "source_date_epoch": SOURCE_DATE_EPOCH,
        "members": members,
        "bridge_source": _bridge_source(),
        "g01_identity": G01_IDENTITY,
        "compatible_routes": ROUTES,
        "authorization": {
            "g00f_outcomes_seen": False,
            "g01_scientifically_eligible": False,
            "direct_g01_launch_authorized": False,
            "dedicated_global_coordinator_required": True,
        },
    }
    return {**body, "manifest_digest": digest(body)}, payloads


def _archive_bytes(payloads: Mapping[str, bytes], manifest: Mapping[str, Any]) -> bytes:
    output = io.BytesIO()
    with (
        gzip.GzipFile(filename="", mode="wb", fileobj=output, compresslevel=9, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive,
    ):
        members = manifest.get("members")
        if not isinstance(members, list):
            raise BuildError("payload manifest member list is malformed")
        for row in members:
            if not isinstance(row, Mapping):
                raise BuildError("payload manifest member is malformed")
            relative = str(row["path"])
            payload = payloads[relative]
            if len(payload) != row.get("bytes") or hashlib.sha256(payload).hexdigest() != row.get("sha256"):
                raise BuildError(f"payload changed during deterministic archive assembly: {relative}")
            info = tarfile.TarInfo(relative)
            info.size = len(payload)
            info.mode = 0o644
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = SOURCE_DATE_EPOCH
            archive.addfile(info, io.BytesIO(payload))
    return output.getvalue()


def _exclusive_write(parent: int, name: str, payload: bytes, mode: int = 0o644) -> None:
    if not name or Path(name).name != name:
        raise BuildError("build output leaf name is unsafe")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, 0o600, dir_fd=parent)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fchmod(handle.fileno(), mode)
        os.fsync(handle.fileno())
    os.fsync(parent)


def _open_directory_chain(path: Path, label: str) -> int:
    absolute = path.absolute()
    if not absolute.is_absolute() or ".." in absolute.parts:
        raise BuildError(f"{label} is not a safe absolute directory path")
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
        raise BuildError(f"{label} cannot be opened component-wise without links") from error


def _open_relative_directory(parent: int, relative: str, label: str) -> int:
    parts = _safe_relative(relative)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    current = os.dup(parent)
    try:
        for part in parts:
            child = os.open(part, flags, dir_fd=current)
            os.close(current)
            current = child
        return current
    except OSError as error:
        os.close(current)
        raise BuildError(f"{label} cannot be opened beneath the held repository") from error


def build(repo: str | Path, output: str | Path) -> dict[str, Any]:
    """Build into a new directory; used by tests and the canonical CLI."""

    build_runtime = _validate_runtime_identity(_runtime_identity(), "live builder runtime")
    resolved = _repo(Path(repo))
    repo_descriptor = _open_directory_chain(resolved, "repository root")
    target = Path(output).absolute()
    if not target.name or target.name in {".", ".."}:
        os.close(repo_descriptor)
        raise BuildError("bundle output must have one safe absent leaf name")
    try:
        manifest, payloads = _manifest(repo_descriptor)
        manifest_payload = json_bytes(manifest)
        archive_payload = _archive_bytes(payloads, manifest)
        builder_relative = "runs/goalzendo/build_g00f_g01_bridge_bundle.py"
        stager_relative = "runs/goalzendo/g00f_g01_bridge_bundle_stage.py"
        builder_metadata, builder_bytes = _read_regular_at(
            repo_descriptor,
            builder_relative,
            "canonical repository builder",
        )
        _, stager_bytes = _read_regular_at(
            repo_descriptor,
            stager_relative,
            "canonical repository stager",
        )
        running_builder = Path(__file__).absolute()
        running_metadata, running_bytes = _snapshot_regular(running_builder, "running builder")
        if (
            running_builder.resolve() != running_builder
            or (running_metadata.st_dev, running_metadata.st_ino)
            != (builder_metadata.st_dev, builder_metadata.st_ino)
            or running_bytes != builder_bytes
        ):
            raise BuildError("running builder is not the canonical repository builder")
        _require_descriptor_matches_path(repo_descriptor, resolved, "repository root")
        builder_sha = sha256_bytes(builder_bytes)
        stager_sha = sha256_bytes(stager_bytes)
        body = {
            "schema": FREEZE_SCHEMA,
            "schema_version": 1,
            "study_id": "g00f_to_g01_checkpoint_a",
            "source_date_epoch": SOURCE_DATE_EPOCH,
            "outcomes_seen": False,
            "build_runtime": build_runtime,
            "bundle": {
                "archive_name": ARCHIVE_NAME,
                "archive_sha256": sha256_bytes(archive_payload),
                "manifest_name": MANIFEST_NAME,
                "manifest_sha256": sha256_bytes(manifest_payload),
                "manifest_digest": manifest["manifest_digest"],
                "payload_member_count": len(PAYLOAD_SHA256),
                "archive_total_member_count": len(PAYLOAD_SHA256),
                "selected_route_freeze_copy_count": 1,
            },
            "controller_files": {
                "builder": {"path": builder_relative, "sha256": builder_sha},
                "stager": {"path": stager_relative, "sha256": stager_sha},
            },
            "bridge_source": manifest["bridge_source"],
            "g01_identity": G01_IDENTITY,
            "compatible_routes": ROUTES,
            "authorization": manifest["authorization"],
            "transaction": {
                "canonical_program_root": "/workspace/status-goalzendo/g00f-executions",
                "frozen_source_directory": "frozen-source",
                "bundle_targets_must_all_be_absent": True,
                "selected_route_freeze_target_must_be_absent": True,
                "exclusive_regular_writes": True,
                "receipt_written_last_outside_uuid_root": True,
                "partial_failure_is_permanent": True,
            },
        }
        freeze = {**body, "freeze_digest": digest(body)}
        freeze_payload = json_bytes(freeze)
        try:
            relative_parent = target.parent.relative_to(resolved).as_posix()
        except ValueError:
            output_parent = _open_directory_chain(target.parent, "bundle output parent")
        else:
            output_parent = (
                os.dup(repo_descriptor)
                if relative_parent == "."
                else _open_relative_directory(
                    repo_descriptor,
                    relative_parent,
                    "bundle output parent",
                )
            )
        try:
            try:
                os.mkdir(target.name, mode=0o755, dir_fd=output_parent)
            except FileExistsError as error:
                raise BuildError(f"prospective build refuses to reuse output directory: {target}") from error
            os.fsync(output_parent)
            target_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
            target_descriptor = os.open(target.name, target_flags, dir_fd=output_parent)
            try:
                _exclusive_write(target_descriptor, MANIFEST_NAME, manifest_payload)
                _exclusive_write(target_descriptor, ARCHIVE_NAME, archive_payload)
                _exclusive_write(target_descriptor, FREEZE_NAME, freeze_payload)
                for name, expected_payload in (
                    (MANIFEST_NAME, manifest_payload),
                    (ARCHIVE_NAME, archive_payload),
                    (FREEZE_NAME, freeze_payload),
                ):
                    _, observed_payload = _read_regular_at(
                        target_descriptor,
                        name,
                        f"written bundle artifact {name}",
                        mode=0o644,
                    )
                    if observed_payload != expected_payload:
                        raise BuildError(f"written bundle artifact changed: {name}")
                os.fsync(target_descriptor)
                _require_descriptor_matches_path(
                    repo_descriptor,
                    resolved,
                    "repository root",
                )
                _require_descriptor_matches_path(
                    output_parent,
                    target.parent,
                    "bundle output parent",
                )
                _require_descriptor_matches_path(
                    target_descriptor,
                    target,
                    "bundle output directory",
                )
            finally:
                os.close(target_descriptor)
            os.fsync(output_parent)
        finally:
            os.close(output_parent)
    finally:
        os.close(repo_descriptor)
    return {
        "output": str(target),
        "archive_sha256": sha256_bytes(archive_payload),
        "manifest_sha256": sha256_bytes(manifest_payload),
        "manifest_digest": manifest["manifest_digest"],
        "freeze_sha256": sha256_bytes(freeze_payload),
        "freeze_digest": freeze["freeze_digest"],
        "bridge_source_digest": BRIDGE_SOURCE_DIGEST,
        "g01_scientifically_eligible": False,
        "direct_g01_launch_authorized": False,
        "dedicated_global_coordinator_required": True,
    }


def main() -> int:
    try:
        _runtime_identity()
        repo = _repo(Path(__file__))
        result = build(repo, repo / DEFAULT_OUTPUT)
    except (BuildError, OSError, tarfile.TarError) as error:
        print(f"build_g00f_g01_bridge_bundle: error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
