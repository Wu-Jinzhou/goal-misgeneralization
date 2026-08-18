#!/usr/bin/env python3
"""Create and replay the non-authorizing G00-F -> G01 build-input transport.

The transport contains only the exact accepted inputs needed to run the
checkpoint-A canonical bundle builder on a disposable Runpod machine.  It is
not a G00-F execution bundle, contains no observations, and authorizes neither
G00-F nor G01.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import platform
import stat
import sys
import tarfile
import zlib
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Final

SOURCE_DATE_EPOCH: Final = 1_786_492_800
ARCHIVE_NAME: Final = "g00f-g01-build-input.tar.gz"
MANIFEST_NAME: Final = "g00f-g01-build-input-manifest.json"
INTENT_NAME: Final = "g00f-g01-build-input-intent.json"
CANONICAL_BUNDLE_RELATIVE: Final = "reproducibility/goalzendo/g00f-g01-bridge-prebootstrap-20260812"
CANONICAL_IMAGE: Final = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
CANONICAL_PYTHON: Final = "/workspace/.venvs/goalzendo/bin/python"
BRIDGE_SOURCE_DIGEST: Final = "e0f3df053557cdf06c028a08f847db33b55dac3a7388a52e0fae0d94b07d284f"
BUILDER_SHA256: Final = "29c906cdfc6c50f672a0819ece3d007138382099dea26e975715e7619e65b243"
STAGER_SHA256: Final = "58c0b75a4608a0b5cd6a85d2ae1b7b5a13af080fc8f9a6305151cca851963b49"

MANIFEST_SCHEMA: Final = "goalzendo.g00f_g01_build_input_transport_manifest"
INTENT_SCHEMA: Final = "goalzendo.g00f_g01_build_input_transport_intent"


class TransportError(RuntimeError):
    """Raised when the prospective transport contract is violated."""


# path -> (mode, byte count, SHA-256).  This is the exact accepted dependency
# closure of the separately audited canonical bundle builder.
FILES: Final[dict[str, tuple[int, int, str]]] = {
    "configs/goalzendo/g01_known_law.yaml": (
        0o644,
        1495,
        "ee6d53556189a6b6e25cfbfb204325017a058c63a605653df4c175463e00ce18",
    ),
    "docs/goalzendo/protocols/g01-known-law.md": (
        0o644,
        14174,
        "428243c3da271bc2d79e16c0fde20d47076eb3837c6740ab733e278cddcdbce9",
    ),
    "pyproject.toml": (
        0o644,
        1819,
        "aa6e5117128140343290abbed9bf2a3a0203f2662bd80f031748f2b371392092",
    ),
    "reproducibility/goalzendo/g00f-execution-freeze-20260811/execution-freeze.json": (
        0o644,
        16585,
        "b2e385488ea7eb7c6f7bfc834c32ff2707aa9f96b718ea62b7ae92c9a4b8df38",
    ),
    "reproducibility/goalzendo/g00f-execution-freeze-20260811/g00f-execution-source.tar.gz": (
        0o644,
        269430,
        "6b55d42bd2a75c9fc40f9c67b332c7ad5bd57024c26bd49447ef6bc831dece3a",
    ),
    "reproducibility/goalzendo/g00f-execution-freeze-20260811/g00f-source-bundle-manifest.json": (
        0o644,
        9985,
        "482f57dc5e0b19a9c1b6ab18a1d63283ce559d288b8a00cc08e51af464d5f8f7",
    ),
    "reproducibility/goalzendo/g00f-h200-execution-freeze-20260811/execution-freeze.json": (
        0o644,
        44636,
        "fd9cf73d124ec566e0589ec0b2e5e4d48a172bca2d04aff51bbb75a58b76b0f9",
    ),
    "reproducibility/goalzendo/g00f-h200-execution-freeze-20260811/g00f-h200-execution-source.tar.gz": (
        0o644,
        722902,
        "e08d81bf0345ad343baa02599add8714d9c9e2f180b936113a92f91b5e8cf8ea",
    ),
    "reproducibility/goalzendo/g00f-h200-execution-freeze-20260811/g00f-h200-source-bundle-manifest.json": (
        0o644,
        13501,
        "a6b741366d7023a625e59cf3c14f7b49e78117fadb16869d8501ad5d396f746f",
    ),
    "runs/goalzendo/build_g00f_g01_bridge_bundle.py": (
        0o644,
        36412,
        BUILDER_SHA256,
    ),
    "runs/goalzendo/g00f_bundle_bootstrap.py": (
        0o644,
        11604,
        "967aa6b68d831bbbfc9757cf28c62bcb4521c8c08bfe5fceb26434946b9ae286",
    ),
    "runs/goalzendo/g00f_g01_bridge_bundle_stage.py": (
        0o644,
        76131,
        STAGER_SHA256,
    ),
    "runs/goalzendo/g00f_h200_bundle_bootstrap.py": (
        0o644,
        12302,
        "4f6b80b0c3e7a2a84f5ae351af9065b53088282112910beb5fb716443bef7ec4",
    ),
    "runs/goalzendo/g00f_h200_detached_supervisor.py": (
        0o755,
        13603,
        "1565ba556736b6962e9a8691d3fa492ceda598993f9f3d1aec7b88f6aa74b0df",
    ),
    "runs/goalzendo/g00f_h200_qualification_controller.py": (
        0o644,
        253,
        "fd835b7b5c7184c7a4ccc81e770ae1f6e6d400c43cf4bcd4d73def93e217f394",
    ),
    "runs/goalzendo/g00f_h200_qualification_supervisor.py": (
        0o755,
        15778,
        "93c0103ca2658578a2b90b1ed2ce723dd1c37c88c6f50888f2154b6f7da0dc2a",
    ),
    "runs/goalzendo/g00f_h200_watchdog.py": (
        0o644,
        22653,
        "533ef5b477723816c61b13ad616f47815be04949415a9db9cc52bca53ac02438",
    ),
    "runs/goalzendo/g00f_watchdog.py": (
        0o644,
        6527,
        "290c4323975f9ba914a31260349b31f200a1f70cbcc17115306ef6b43019e1c5",
    ),
    "runs/goalzendo/run_g00f_frozen_4h100.sh": (
        0o644,
        13530,
        "576f7dc82f6c4842af872067f72bbe05b55667e4705b9255efbdb29b35493bbe",
    ),
    "runs/goalzendo/run_g00f_frozen_4h200.sh": (
        0o755,
        25375,
        "5bc6dfcab92f9632a1fc3f4a29815d9c6298ee2fc86f47690ce6d1b371ad39f2",
    ),
    "runs/goalzendo/run_g01_after_g00f_bridge.py": (
        0o644,
        2225,
        "bc7703c48d47cc72f21e2f72ee3a4864ff5dd78759c5f692c59ce46ba03ee10b",
    ),
    "src/goalzendo/__init__.py": (
        0o644,
        1777,
        "a6dd83fdd29b0f476e91cefc65d02794a161b7cbe11eeef91622736f5a36e977",
    ),
    "src/goalzendo_g00f_g01_bridge/__init__.py": (
        0o644,
        1312,
        "94d447686b62177ed3f50742424c9a98a6f1c95d921422cf7bf3241689e24503",
    ),
    "src/goalzendo_g00f_g01_bridge/bridge.py": (
        0o644,
        99568,
        "4ab7c684109b10c2d6ec1469a281fa342e0dac14c5d141a9d97f0cd8e4f575d2",
    ),
    "src/goalzendo_g00f_g01_bridge/cli.py": (
        0o644,
        9150,
        "350cb69a2923b49aeb4957406b4d86319a2c75f7c7a32e14e53d479b04319801",
    ),
}

DIRECTORIES: Final[tuple[str, ...]] = (
    "configs",
    "configs/goalzendo",
    "docs",
    "docs/goalzendo",
    "docs/goalzendo/protocols",
    "reproducibility",
    "reproducibility/goalzendo",
    "reproducibility/goalzendo/g00f-execution-freeze-20260811",
    "reproducibility/goalzendo/g00f-h200-execution-freeze-20260811",
    "runs",
    "runs/goalzendo",
    "src",
    "src/goalzendo",
    "src/goalzendo_g00f_g01_bridge",
)

AUTHORIZATION: Final[dict[str, Any]] = {
    "purpose": "disposable_build_input_transport_only",
    "g00f_execution_authorized": False,
    "g00f_outcomes_seen": False,
    "g01_scientifically_eligible": False,
    "g01_launch_authorized": False,
    "canonical_bundle_created": False,
}


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return _sha(_canonical(value))


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _strict_json(payload: bytes, label: str) -> dict[str, Any]:
    def reject(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise TransportError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=reject)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise TransportError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise TransportError(f"{label} must contain one object")
    return value


def _safe_relative(value: str) -> tuple[str, ...]:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or "." in path.parts or ".." in path.parts or path.as_posix() != value:
        raise TransportError(f"unsafe relative path: {value!r}")
    return path.parts


def _open_directory(path: Path, label: str) -> int:
    absolute = path.absolute()
    if not absolute.is_absolute() or ".." in absolute.parts:
        raise TransportError(f"{label} is not a safe absolute directory")
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
        raise TransportError(f"{label} cannot be opened without link traversal") from error


def _open_relative_directory(parent: int, relative: str, label: str) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    current = os.dup(parent)
    try:
        for part in _safe_relative(relative):
            child = os.open(part, flags, dir_fd=current)
            os.close(current)
            current = child
        return current
    except OSError as error:
        os.close(current)
        raise TransportError(f"{label} cannot be opened beneath held root") from error


def _read_at(parent: int, relative: str, label: str) -> tuple[os.stat_result, bytes]:
    parts = _safe_relative(relative)
    directory = os.dup(parent)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    file_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        for part in parts[:-1]:
            child = os.open(part, directory_flags, dir_fd=directory)
            os.close(directory)
            directory = child
        descriptor = os.open(parts[-1], file_flags, dir_fd=directory)
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise TransportError(f"{label} is not a single-link regular file")
            with os.fdopen(os.dup(descriptor), "rb") as handle:
                payload = handle.read()
        finally:
            os.close(descriptor)
    except OSError as error:
        raise TransportError(f"{label} cannot be opened without link traversal") from error
    finally:
        os.close(directory)
    return metadata, payload


def _descriptor_matches(descriptor: int, path: Path, label: str) -> None:
    held = os.fstat(descriptor)
    observed = path.stat(follow_symlinks=False)
    if not stat.S_ISDIR(observed.st_mode) or (held.st_dev, held.st_ino) != (
        observed.st_dev,
        observed.st_ino,
    ):
        raise TransportError(f"{label} no longer names the held directory")


def _snapshot_file(path: Path, label: str, *, mode: int | None = None) -> bytes:
    absolute = path.absolute()
    parent = _open_directory(absolute.parent, f"{label} parent")
    try:
        metadata, payload = _read_at(parent, absolute.name, label)
        if mode is not None and stat.S_IMODE(metadata.st_mode) != mode:
            raise TransportError(f"{label} mode must be {mode:04o}")
        _descriptor_matches(parent, absolute.parent, f"{label} parent")
        return payload
    finally:
        os.close(parent)


def _self_snapshot() -> tuple[str, bytes]:
    payload = _snapshot_file(Path(__file__), "transport controller", mode=0o644)
    return _sha(payload), payload


def _build_runtime() -> dict[str, Any]:
    return {
        "implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "byteorder": sys.byteorder,
        "zlib_build_version": zlib.ZLIB_VERSION,
        "zlib_runtime_version": zlib.ZLIB_RUNTIME_VERSION,
        "trust_boundary": (
            "transport bytes are deterministic only within this recorded local stdlib/runtime; "
            "their externally registered SHA-256 is authoritative"
        ),
    }


def _source_snapshot(repo: int) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    for relative in DIRECTORIES:
        descriptor = _open_relative_directory(repo, relative, f"required directory {relative}")
        try:
            metadata = os.fstat(descriptor)
            if stat.S_IMODE(metadata.st_mode) != 0o755:
                raise TransportError(f"required directory mode changed: {relative}")
        finally:
            os.close(descriptor)
    for relative, (mode, size, expected_sha) in FILES.items():
        metadata, payload = _read_at(repo, relative, f"required input {relative}")
        if (
            stat.S_IMODE(metadata.st_mode) != mode
            or metadata.st_size != size
            or _sha(payload) != expected_sha
        ):
            raise TransportError(f"accepted input changed: {relative}")
        payloads[relative] = payload
    return payloads


def _require_relative_absent(parent: int, relative: str, label: str) -> None:
    """Require one relative path to be absent beneath a held directory."""

    parts = _safe_relative(relative)
    current = os.dup(parent)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        for part in parts[:-1]:
            try:
                child = os.open(part, directory_flags, dir_fd=current)
            except FileNotFoundError:
                return
            os.close(current)
            current = child
        try:
            os.stat(parts[-1], dir_fd=current, follow_symlinks=False)
        except FileNotFoundError:
            return
        raise TransportError(f"{label} must be absent")
    finally:
        os.close(current)


def _manifest(controller_sha: str, build_runtime: Mapping[str, Any]) -> dict[str, Any]:
    body = {
        "schema": MANIFEST_SCHEMA,
        "schema_version": 1,
        "study_id": "g00f_to_g01_checkpoint_a",
        "source_date_epoch": SOURCE_DATE_EPOCH,
        "required_directories": [
            {"path": relative, "type": "directory", "mode": 0o755} for relative in DIRECTORIES
        ],
        "members": [
            {
                "path": relative,
                "type": "file",
                "mode": mode,
                "bytes": size,
                "sha256": sha256,
            }
            for relative, (mode, size, sha256) in FILES.items()
        ],
        "accepted_controllers": {
            "canonical_bundle_builder": {
                "path": "runs/goalzendo/build_g00f_g01_bridge_bundle.py",
                "sha256": BUILDER_SHA256,
            },
            "canonical_bundle_stager": {
                "path": "runs/goalzendo/g00f_g01_bridge_bundle_stage.py",
                "sha256": STAGER_SHA256,
            },
            "transport_controller": {
                "path": "runs/goalzendo/g00f_g01_build_input_transport.py",
                "sha256": controller_sha,
            },
        },
        "bridge_source_digest": BRIDGE_SOURCE_DIGEST,
        "build_runtime": dict(build_runtime),
        "authorization": AUTHORIZATION,
    }
    return {**body, "manifest_digest": _digest(body)}


def _archive(payloads: Mapping[str, bytes]) -> bytes:
    output = io.BytesIO()
    with (
        gzip.GzipFile(filename="", mode="wb", fileobj=output, compresslevel=9, mtime=0) as zipped,
        tarfile.open(fileobj=zipped, mode="w", format=tarfile.USTAR_FORMAT) as archive,
    ):
        for relative, (mode, size, expected_sha) in FILES.items():
            payload = payloads[relative]
            if len(payload) != size or _sha(payload) != expected_sha:
                raise TransportError(f"input changed during archive assembly: {relative}")
            info = tarfile.TarInfo(relative)
            info.type = tarfile.REGTYPE
            info.size = len(payload)
            info.mode = mode
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = SOURCE_DATE_EPOCH
            archive.addfile(info, io.BytesIO(payload))
    return output.getvalue()


def _intent(
    *,
    archive_sha: str,
    manifest_sha: str,
    manifest_digest: str,
    controller_sha: str,
) -> dict[str, Any]:
    body = {
        "schema": INTENT_SCHEMA,
        "schema_version": 1,
        "study_id": "g00f_to_g01_checkpoint_a",
        "purpose": "prospective_disposable_canonical_bundle_build_only",
        "transport": {
            "archive_name": ARCHIVE_NAME,
            "archive_sha256": archive_sha,
            "archive_member_count": len(FILES),
            "manifest_name": MANIFEST_NAME,
            "manifest_sha256": manifest_sha,
            "manifest_digest": manifest_digest,
            "required_directory_count": len(DIRECTORIES),
            "transport_controller_sha256": controller_sha,
        },
        "accepted_source": {
            "canonical_bundle_builder_sha256": BUILDER_SHA256,
            "canonical_bundle_stager_sha256": STAGER_SHA256,
            "bridge_source_digest": BRIDGE_SOURCE_DIGEST,
        },
        "canonical_build_contract": {
            "image": CANONICAL_IMAGE,
            "lexical_python": CANONICAL_PYTHON,
            "python_implementation": "CPython",
            "python_version": [3, 12, 3, "final", 0],
            "python_flags": {
                "isolated": True,
                "ignore_environment": True,
                "no_user_site": True,
                "safe_path": True,
            },
            "platform_system": "Linux",
            "platform_machine": "x86_64",
            "byteorder": "little",
            "zlib_build_version": "1.3",
            "zlib_runtime_version": "1.3",
            "canonical_output_relative": CANONICAL_BUNDLE_RELATIVE,
            "canonical_output_must_be_absent": True,
        },
        "registration": {
            "intent_whole_file_sha256_must_be_registered_before_pod_create": True,
            "disposable_build_witness": "timestamped_codex_transcript_external_to_runpod_and_worktree",
            "witness_limitations": (
                "the transcript fixes conversational order but is not claimed to be a cryptographic "
                "timestamp service or a durable independent append-only registry"
            ),
            "not_sufficient_for_scientific_itt": True,
            "durable_independent_registrar_required_before_g00f_itt": True,
        },
        "authorization": AUTHORIZATION,
    }
    return {**body, "intent_digest": _digest(body)}


def _exclusive_write(parent: int, name: str, payload: bytes) -> None:
    if not name or Path(name).name != name:
        raise TransportError("unsafe output leaf name")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, 0o600, dir_fd=parent)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fchmod(handle.fileno(), 0o644)
        os.fsync(handle.fileno())
    os.fsync(parent)


def _new_output_directory(path: Path, label: str) -> tuple[int, int]:
    absolute = path.absolute()
    if not absolute.name or absolute.name in {".", ".."}:
        raise TransportError(f"{label} must have one safe absent leaf")
    parent = _open_directory(absolute.parent, f"{label} parent")
    try:
        os.mkdir(absolute.name, mode=0o755, dir_fd=parent)
        os.fsync(parent)
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        target = os.open(absolute.name, flags, dir_fd=parent)
    except (FileExistsError, OSError) as error:
        os.close(parent)
        raise TransportError(f"{label} must be wholly absent") from error
    return parent, target


def build(repo: Path, output: Path) -> dict[str, Any]:
    """Create the prospective transport in one absent directory."""

    controller_sha, _ = _self_snapshot()
    repo_path = repo.absolute()
    output_path = output.absolute()
    canonical_path = (repo_path / CANONICAL_BUNDLE_RELATIVE).absolute()
    if output_path == canonical_path or canonical_path in output_path.parents:
        raise TransportError("transport output must be outside the canonical bridge bundle subtree")
    repo_descriptor = _open_directory(repo_path, "repository root")
    try:
        _require_relative_absent(
            repo_descriptor,
            CANONICAL_BUNDLE_RELATIVE,
            "canonical bridge bundle",
        )
        payloads = _source_snapshot(repo_descriptor)
        manifest = _manifest(controller_sha, _build_runtime())
        manifest_payload = _json_bytes(manifest)
        archive_payload = _archive(payloads)
        intent = _intent(
            archive_sha=_sha(archive_payload),
            manifest_sha=_sha(manifest_payload),
            manifest_digest=str(manifest["manifest_digest"]),
            controller_sha=controller_sha,
        )
        intent_payload = _json_bytes(intent)
        _descriptor_matches(repo_descriptor, repo_path, "repository root")
        parent, target = _new_output_directory(output_path, "transport output")
        try:
            _exclusive_write(target, ARCHIVE_NAME, archive_payload)
            _exclusive_write(target, MANIFEST_NAME, manifest_payload)
            _exclusive_write(target, INTENT_NAME, intent_payload)
            for name, expected in (
                (ARCHIVE_NAME, archive_payload),
                (MANIFEST_NAME, manifest_payload),
                (INTENT_NAME, intent_payload),
            ):
                metadata, observed = _read_at(target, name, f"written {name}")
                if stat.S_IMODE(metadata.st_mode) != 0o644 or observed != expected:
                    raise TransportError(f"written transport artifact changed: {name}")
            os.fsync(target)
            _require_relative_absent(
                repo_descriptor,
                CANONICAL_BUNDLE_RELATIVE,
                "canonical bridge bundle",
            )
            _descriptor_matches(repo_descriptor, repo_path, "repository root")
            _descriptor_matches(parent, output_path.parent, "transport output parent")
            _descriptor_matches(target, output_path, "transport output")
        finally:
            os.close(target)
            os.close(parent)
    finally:
        os.close(repo_descriptor)
    return {
        "archive_sha256": _sha(archive_payload),
        "manifest_sha256": _sha(manifest_payload),
        "manifest_digest": manifest["manifest_digest"],
        "intent_sha256": _sha(intent_payload),
        "intent_digest": intent["intent_digest"],
        "transport_controller_sha256": controller_sha,
        "member_count": len(FILES),
        "required_directory_count": len(DIRECTORIES),
        "authorization": AUTHORIZATION,
    }


def _expect_sha(value: str, label: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise TransportError(f"{label} must be a lowercase SHA-256")
    return value


def _artifact_snapshot(
    archive: Path,
    manifest: Path,
    intent: Path,
) -> tuple[bytes, bytes, bytes]:
    paths = [archive.absolute(), manifest.absolute(), intent.absolute()]
    if len({path.parent for path in paths}) != 1:
        raise TransportError("transport artifacts must share one directory")
    parent_path = paths[0].parent
    parent = _open_directory(parent_path, "transport artifact directory")
    try:
        rows = []
        for path, name in zip(paths, (ARCHIVE_NAME, MANIFEST_NAME, INTENT_NAME), strict=True):
            if path.name != name:
                raise TransportError(f"transport artifact must have canonical name {name}")
            metadata, payload = _read_at(parent, name, f"transport artifact {name}")
            if stat.S_IMODE(metadata.st_mode) != 0o644:
                raise TransportError(f"transport artifact mode must be 0644: {name}")
            rows.append(payload)
        _descriptor_matches(parent, parent_path, "transport artifact directory")
        return rows[0], rows[1], rows[2]
    finally:
        os.close(parent)


def _validate_manifest(value: Mapping[str, Any], controller_sha: str) -> None:
    keys = {
        "schema",
        "schema_version",
        "study_id",
        "source_date_epoch",
        "required_directories",
        "members",
        "accepted_controllers",
        "bridge_source_digest",
        "build_runtime",
        "authorization",
        "manifest_digest",
    }
    body = {key: item for key, item in value.items() if key != "manifest_digest"}
    runtime = value.get("build_runtime")
    if (
        not isinstance(runtime, Mapping)
        or set(runtime)
        != {
            "implementation",
            "python_version",
            "platform_system",
            "platform_machine",
            "byteorder",
            "zlib_build_version",
            "zlib_runtime_version",
            "trust_boundary",
        }
        or runtime.get("implementation") not in {"CPython", "PyPy"}
        or not all(isinstance(runtime.get(key), str) and runtime.get(key) for key in runtime)
    ):
        raise TransportError("transport manifest build runtime is malformed")
    expected = _manifest(controller_sha, runtime)
    if set(value) != keys or value != expected or value.get("manifest_digest") != _digest(body):
        raise TransportError("transport manifest is not the exact accepted manifest")


def _validate_intent(
    value: Mapping[str, Any],
    *,
    archive_sha: str,
    manifest_sha: str,
    manifest_digest: str,
    controller_sha: str,
) -> None:
    keys = {
        "schema",
        "schema_version",
        "study_id",
        "purpose",
        "transport",
        "accepted_source",
        "canonical_build_contract",
        "registration",
        "authorization",
        "intent_digest",
    }
    body = {key: item for key, item in value.items() if key != "intent_digest"}
    expected = _intent(
        archive_sha=archive_sha,
        manifest_sha=manifest_sha,
        manifest_digest=manifest_digest,
        controller_sha=controller_sha,
    )
    if set(value) != keys or value != expected or value.get("intent_digest") != _digest(body):
        raise TransportError("transport intent is not the exact prospective intent")


def _validate_archive(payload: bytes) -> dict[str, bytes]:
    observed: dict[str, bytes] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
            members = archive.getmembers()
            if [member.name for member in members] != list(FILES):
                raise TransportError("transport archive inventory or order changed")
            for member, (relative, (mode, size, expected_sha)) in zip(members, FILES.items(), strict=True):
                if (
                    member.name != relative
                    or member.type != tarfile.REGTYPE
                    or not member.isfile()
                    or member.mode != mode
                    or member.size != size
                    or member.uid != 0
                    or member.gid != 0
                    or member.uname != ""
                    or member.gname != ""
                    or member.mtime != SOURCE_DATE_EPOCH
                    or member.linkname != ""
                    or member.devmajor != 0
                    or member.devminor != 0
                    or member.pax_headers
                ):
                    raise TransportError(f"transport archive metadata changed: {relative}")
                handle = archive.extractfile(member)
                if handle is None:
                    raise TransportError(f"transport member cannot be read: {relative}")
                member_payload = handle.read()
                if len(member_payload) != size or _sha(member_payload) != expected_sha:
                    raise TransportError(f"transport member bytes changed: {relative}")
                observed[relative] = member_payload
    except (tarfile.TarError, OSError) as error:
        raise TransportError("transport archive cannot be parsed") from error
    return observed


def _verify_snapshots(
    *,
    archive: Path,
    manifest: Path,
    intent: Path,
    expected_archive_sha256: str,
    expected_manifest_sha256: str,
    expected_intent_sha256: str,
    expected_controller_sha256: str,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Authenticate each input once and return the exact verified member bytes."""

    controller_sha, _ = _self_snapshot()
    if controller_sha != _expect_sha(expected_controller_sha256, "expected controller SHA"):
        raise TransportError("running transport controller does not match external pin")
    archive_payload, manifest_payload, intent_payload = _artifact_snapshot(archive, manifest, intent)
    if _sha(archive_payload) != _expect_sha(expected_archive_sha256, "expected archive SHA"):
        raise TransportError("transport archive does not match external pin")
    if _sha(manifest_payload) != _expect_sha(expected_manifest_sha256, "expected manifest SHA"):
        raise TransportError("transport manifest does not match external pin")
    if _sha(intent_payload) != _expect_sha(expected_intent_sha256, "expected intent SHA"):
        raise TransportError("transport intent does not match external pin")
    manifest_value = _strict_json(manifest_payload, "transport manifest")
    intent_value = _strict_json(intent_payload, "transport intent")
    _validate_manifest(manifest_value, controller_sha)
    payloads = _validate_archive(archive_payload)
    _validate_intent(
        intent_value,
        archive_sha=_sha(archive_payload),
        manifest_sha=_sha(manifest_payload),
        manifest_digest=str(manifest_value["manifest_digest"]),
        controller_sha=controller_sha,
    )
    return (
        {
            "archive_sha256": _sha(archive_payload),
            "manifest_sha256": _sha(manifest_payload),
            "manifest_digest": manifest_value["manifest_digest"],
            "intent_sha256": _sha(intent_payload),
            "intent_digest": intent_value["intent_digest"],
            "transport_controller_sha256": controller_sha,
            "member_count": len(payloads),
            "authorization": AUTHORIZATION,
        },
        payloads,
    )


def verify(
    *,
    archive: Path,
    manifest: Path,
    intent: Path,
    expected_archive_sha256: str,
    expected_manifest_sha256: str,
    expected_intent_sha256: str,
    expected_controller_sha256: str,
) -> dict[str, Any]:
    """Replay the externally pinned prospective transport."""

    result, _ = _verify_snapshots(
        archive=archive,
        manifest=manifest,
        intent=intent,
        expected_archive_sha256=expected_archive_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_intent_sha256=expected_intent_sha256,
        expected_controller_sha256=expected_controller_sha256,
    )
    return result


def _mkdir_at(root: int, relative: str) -> None:
    parts = _safe_relative(relative)
    current = os.dup(root)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        for part in parts:
            try:
                os.mkdir(part, mode=0o755, dir_fd=current)
                os.fsync(current)
            except FileExistsError:
                pass
            child = os.open(part, flags, dir_fd=current)
            if stat.S_IMODE(os.fstat(child).st_mode) != 0o755:
                os.close(child)
                raise TransportError(f"extracted directory mode changed: {relative}")
            os.close(current)
            current = child
    finally:
        os.close(current)


def _write_relative(root: int, relative: str, payload: bytes, mode: int) -> None:
    parts = _safe_relative(relative)
    current = os.dup(root)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        for part in parts[:-1]:
            child = os.open(part, directory_flags, dir_fd=current)
            os.close(current)
            current = child
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(parts[-1], flags, 0o600, dir_fd=current)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fchmod(handle.fileno(), mode)
            os.fsync(handle.fileno())
        os.fsync(current)
    finally:
        os.close(current)


def _inventory(root: int) -> tuple[dict[str, int], dict[str, tuple[int, bytes]]]:
    directories: dict[str, int] = {}
    files: dict[str, tuple[int, bytes]] = {}

    def walk(descriptor: int, prefix: str) -> None:
        for name in sorted(os.listdir(descriptor)):
            relative = f"{prefix}/{name}" if prefix else name
            metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if stat.S_ISDIR(metadata.st_mode):
                flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
                child = os.open(name, flags, dir_fd=descriptor)
                try:
                    directories[relative] = stat.S_IMODE(os.fstat(child).st_mode)
                    walk(child, relative)
                finally:
                    os.close(child)
            elif stat.S_ISREG(metadata.st_mode):
                row, payload = _read_at(descriptor, name, f"extracted file {relative}")
                files[relative] = (stat.S_IMODE(row.st_mode), payload)
            else:
                raise TransportError(f"extracted tree contains special or linked entry: {relative}")

    walk(root, "")
    return directories, files


def extract(
    *,
    archive: Path,
    manifest: Path,
    intent: Path,
    expected_archive_sha256: str,
    expected_manifest_sha256: str,
    expected_intent_sha256: str,
    expected_controller_sha256: str,
    output: Path,
) -> dict[str, Any]:
    """Verify, then create one exact absent repository tree."""

    result, payloads = _verify_snapshots(
        archive=archive,
        manifest=manifest,
        intent=intent,
        expected_archive_sha256=expected_archive_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_intent_sha256=expected_intent_sha256,
        expected_controller_sha256=expected_controller_sha256,
    )
    output_path = output.absolute()
    parent, root = _new_output_directory(output_path, "extracted repository")
    try:
        for relative in DIRECTORIES:
            _mkdir_at(root, relative)
        for relative, (mode, _, _) in FILES.items():
            _write_relative(root, relative, payloads[relative], mode)
        directories, files = _inventory(root)
        if directories != {relative: 0o755 for relative in DIRECTORIES}:
            raise TransportError("extracted directory inventory changed")
        if set(files) != set(FILES):
            raise TransportError("extracted file inventory changed")
        for relative, (mode, size, sha256) in FILES.items():
            observed_mode, payload = files[relative]
            if observed_mode != mode or len(payload) != size or _sha(payload) != sha256:
                raise TransportError(f"extracted file changed: {relative}")
        if CANONICAL_BUNDLE_RELATIVE in directories or any(
            relative.startswith(CANONICAL_BUNDLE_RELATIVE + "/") for relative in files
        ):
            raise TransportError("canonical bundle unexpectedly exists in transport extraction")
        os.fsync(root)
        _descriptor_matches(parent, output_path.parent, "extraction parent")
        _descriptor_matches(root, output_path, "extracted repository")
    finally:
        os.close(root)
        os.close(parent)
    return {**result, "extracted_repository": str(output_path), "exact_inventory_verified": True}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--repo", required=True, type=Path)
    build_parser.add_argument("--output", required=True, type=Path)
    for name in ("verify", "extract"):
        child = subparsers.add_parser(name)
        child.add_argument("--archive", required=True, type=Path)
        child.add_argument("--manifest", required=True, type=Path)
        child.add_argument("--intent", required=True, type=Path)
        child.add_argument("--expected-archive-sha256", required=True)
        child.add_argument("--expected-manifest-sha256", required=True)
        child.add_argument("--expected-intent-sha256", required=True)
        child.add_argument("--expected-controller-sha256", required=True)
        if name == "extract":
            child.add_argument("--output", required=True, type=Path)
    return parser


def _require_isolated_cli() -> None:
    flags = sys.flags
    safe_path = bool(getattr(flags, "safe_path", False))
    if not (flags.isolated == 1 and flags.ignore_environment == 1 and flags.no_user_site == 1 and safe_path):
        raise TransportError(
            "production transport CLI requires isolated Python (-I): "
            "isolated=1, ignore_environment=1, no_user_site=1, safe_path=true"
        )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        _require_isolated_cli()
        args = _parser().parse_args(argv)
        if args.command == "build":
            result = build(args.repo, args.output)
        elif args.command == "verify":
            result = verify(
                archive=args.archive,
                manifest=args.manifest,
                intent=args.intent,
                expected_archive_sha256=args.expected_archive_sha256,
                expected_manifest_sha256=args.expected_manifest_sha256,
                expected_intent_sha256=args.expected_intent_sha256,
                expected_controller_sha256=args.expected_controller_sha256,
            )
        else:
            result = extract(
                archive=args.archive,
                manifest=args.manifest,
                intent=args.intent,
                expected_archive_sha256=args.expected_archive_sha256,
                expected_manifest_sha256=args.expected_manifest_sha256,
                expected_intent_sha256=args.expected_intent_sha256,
                expected_controller_sha256=args.expected_controller_sha256,
                output=args.output,
            )
    except (TransportError, OSError, tarfile.TarError) as error:
        print(f"g00f_g01_build_input_transport: error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
