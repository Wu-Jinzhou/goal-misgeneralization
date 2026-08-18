#!/usr/bin/env python3
"""Exclusively stage and freshly verify the nonauthorizing B source capsule.

This stdlib-only transaction never opens checkpoint-A state or its thin token.
It compares an operator-supplied eligibility execution UUID with a separately
prechosen G01 UUID, but does not authenticate the former.  Payload writes are
confined to the G01 input namespace; the mode-0400 receipt is written last in
the disjoint preexecution namespace.
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
import uuid
import zlib
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SOURCE_DATE_EPOCH = 1_786_492_800
ARCHIVE_NAME = "g01-b-source-capsule.tar.gz"
MANIFEST_NAME = "g01-b-source-capsule-manifest.json"
FREEZE_NAME = "g01-b-source-capsule-freeze.json"
RECEIPT_NAME = "source-capsule-stage-receipt.json"
MANIFEST_SCHEMA = "goalzendo.g01_b_source_capsule_manifest"
FREEZE_SCHEMA = "goalzendo.g01_b_source_capsule_freeze"
RECEIPT_SCHEMA = "goalzendo.g01_b_source_capsule_stage_receipt"
STUDY_ID = "g01_checkpoint_b_source_capsule"
GOALZENDO_FINGERPRINT = "1a8146377b9a9620690025671614edb2dd20d214f4e195da3cf3528809b2c694"
BRIDGE_SOURCE_DIGEST = "e0f3df053557cdf06c028a08f847db33b55dac3a7388a52e0fae0d94b07d284f"
COORDINATOR_SOURCE_DIGEST = "77d9ba4928fa29cc42a055201660304f2059d0ca6358371cc14618407bdc714b"
STAGER_PATH = "runs/goalzendo/g01_b_source_capsule_stage.py"
CANONICAL_IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
CANONICAL_PYTHON = "/workspace/.venvs/goalzendo/bin/python"

_CANONICAL_INPUT_ROOT = Path("/workspace/inputs-goalzendo/g01-executions")
_CANONICAL_PREEXECUTION_ROOT = Path("/workspace/status-goalzendo/g01-preexecution")
_CANONICAL_EXECUTION_STATUS_ROOT = Path("/workspace/status-goalzendo/g01-executions")
_CANONICAL_ARTIFACT_ROOT = Path("/workspace/artifacts-goalzendo/g01-known-law")
_CANONICAL_CHECKPOINT_A_ROOT = Path("/workspace/status-goalzendo/g00f-executions")
_TEST_ONLY_ROOTS: Mapping[str, Path] | None = None
_TEST_ONLY_ALLOW_RUNTIME = False

PAYLOAD_PATHS = (
    "configs/goalzendo/base.yaml",
    "configs/goalzendo/g01_known_law.yaml",
    "docs/goalzendo/protocols/g01-b-source-capsule.md",
    "docs/goalzendo/protocols/g01-global-coordinator-checkpoint-b.md",
    "docs/goalzendo/protocols/g01-known-law.md",
    "pyproject.toml",
    "runs/goalzendo/build_g01_b_source_capsule.py",
    "runs/goalzendo/g01_b_source_capsule_stage.py",
    "runs/goalzendo/run_g01_after_g00f_bridge.py",
    "runs/goalzendo/run_g01_global_coordinator.py",
    "src/goalzendo/__init__.py",
    "src/goalzendo/analysis.py",
    "src/goalzendo/artifacts.py",
    "src/goalzendo/cli.py",
    "src/goalzendo/config.py",
    "src/goalzendo/evaluation.py",
    "src/goalzendo/experiment.py",
    "src/goalzendo/generation.py",
    "src/goalzendo/interventions.py",
    "src/goalzendo/metrics.py",
    "src/goalzendo/modeling.py",
    "src/goalzendo/plotting.py",
    "src/goalzendo/py.typed",
    "src/goalzendo/rendering.py",
    "src/goalzendo/reproduction.py",
    "src/goalzendo/rules.py",
    "src/goalzendo/runner.py",
    "src/goalzendo/schema.py",
    "src/goalzendo/training.py",
    "src/goalzendo_g00f_g01_bridge/__init__.py",
    "src/goalzendo_g00f_g01_bridge/bridge.py",
    "src/goalzendo_g00f_g01_bridge/cli.py",
    "src/goalzendo_g01_coordinator/__init__.py",
    "src/goalzendo_g01_coordinator/coordinator.py",
)

KNOWN_EXACT_SHA256: Mapping[str, str] = {
    "configs/goalzendo/base.yaml": "34cb42eb2bf7ea275247ecf8baa0e5556565c2c86c30da0aec8e2a1a630e04fb",
    "configs/goalzendo/g01_known_law.yaml": "ee6d53556189a6b6e25cfbfb204325017a058c63a605653df4c175463e00ce18",
    "docs/goalzendo/protocols/g01-global-coordinator-checkpoint-b.md": "93acc6262758dd5478e342e2f89fd5cb1ab3b61a03bed34b1a6d5083cacaaeba",
    "docs/goalzendo/protocols/g01-known-law.md": "428243c3da271bc2d79e16c0fde20d47076eb3837c6740ab733e278cddcdbce9",
    "pyproject.toml": "aa6e5117128140343290abbed9bf2a3a0203f2662bd80f031748f2b371392092",
    "runs/goalzendo/run_g01_after_g00f_bridge.py": "bc7703c48d47cc72f21e2f72ee3a4864ff5dd78759c5f692c59ce46ba03ee10b",
    "runs/goalzendo/run_g01_global_coordinator.py": "69f5ec520d8c681d5e9cab1dd9687d7fd39c948f08ff397331bd5ce74f824a06",
    "src/goalzendo/runner.py": "46b55ad4bdd08073e5f89ae101e0862372817b74b590f07ddb8f8c4331e5b9e1",
}

AUTHORIZATION: Mapping[str, bool] = {
    "checkpoint_b_complete": False,
    "g01_launch_authorized": False,
    "scientific_execution_authorized": False,
    "runtime_overlay_frozen": False,
    "provision_authorized": False,
    "qualification_authorized": False,
    "outcomes_seen": False,
    "accepted_refusal_revision_contained": True,
    "supported_launch_entrypoint_activated": False,
}

TRUST_LIMITATIONS: Mapping[str, Any] = {
    "claim": "source_bytes_only",
    "rootfs_authenticated": False,
    "runtime_authenticated": False,
    "python_stdlib_authenticated": False,
    "native_dependency_closure_authenticated": False,
    "external_launcher_authenticated": False,
}


class StageError(RuntimeError):
    """The source-stage transaction cannot authenticate or continue."""


@dataclass(frozen=True)
class _Roots:
    input_root: Path
    preexecution_root: Path
    execution_status_root: Path
    artifact_root: Path


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def require_sha256(value: Any, label: str) -> str:
    if type(value) is not str:
        raise StageError(f"{label} must be one lowercase SHA-256 string")
    text = value
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise StageError(f"{label} must be one lowercase SHA-256")
    return text


def strict_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    def reject(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise StageError(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                StageError(f"{label} contains non-finite {constant}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise StageError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise StageError(f"{label} must contain one object")
    return value


def _reject_floats(value: Any, label: str) -> None:
    """The capsule schemas contain no JSON floating-point fields."""

    if isinstance(value, float):
        raise StageError(f"{label} contains a forbidden floating-point value")
    if isinstance(value, Mapping):
        for child in value.values():
            _reject_floats(child, label)
    elif isinstance(value, list):
        for child in value:
            _reject_floats(child, label)


def _exact_equal(observed: Any, expected: Any, label: str) -> None:
    if type(observed) is not type(expected):
        raise StageError(f"{label} JSON type changed")
    if isinstance(expected, Mapping):
        if set(observed) != set(expected):
            raise StageError(f"{label} JSON field set changed")
        for key in expected:
            _exact_equal(observed[key], expected[key], f"{label}.{key}")
    elif isinstance(expected, list):
        if len(observed) != len(expected):
            raise StageError(f"{label} JSON list length changed")
        for index, (child, expected_child) in enumerate(zip(observed, expected, strict=True)):
            _exact_equal(child, expected_child, f"{label}[{index}]")
    elif observed != expected:
        raise StageError(f"{label} JSON value changed")


def _exact_int(value: Any, expected: int, label: str) -> None:
    if type(value) is not int or value != expected:
        raise StageError(f"{label} must be exact integer {expected}")


def _require_runtime() -> None:
    if _TEST_ONLY_ALLOW_RUNTIME:
        if "PYTEST_CURRENT_TEST" not in os.environ:
            raise StageError("test-only runtime bypass is forbidden outside pytest")
        return
    flags = sys.flags
    if not (
        sys.implementation.name == "cpython"
        and sys.version_info[:5] == (3, 12, 3, "final", 0)
        and platform.system() == "Linux"
        and platform.machine() == "x86_64"
        and sys.byteorder == "little"
        and str(Path(sys.executable).absolute()) == CANONICAL_PYTHON
        and flags.isolated
        and flags.ignore_environment
        and flags.no_user_site
        and getattr(flags, "safe_path", False)
        and flags.no_site
        and zlib.ZLIB_VERSION == "1.3"
        and zlib.ZLIB_RUNTIME_VERSION == "1.3"
    ):
        raise StageError("stager requires exact CPython 3.12.3 Linux x86_64 -I -S and zlib 1.3")
    try:
        task_ids = os.listdir("/proc/self/task")
    except OSError as error:
        raise StageError("stager requires an observable single-task Linux process") from error
    if len(task_ids) != 1 or not task_ids[0].isdigit():
        raise StageError("stager requires one Linux task before any temporary umask change")


def _roots() -> _Roots:
    if _TEST_ONLY_ROOTS is not None:
        if "PYTEST_CURRENT_TEST" not in os.environ:
            raise StageError("test-only root override is forbidden outside pytest")
        if set(_TEST_ONLY_ROOTS) != {
            "input_root",
            "preexecution_root",
            "execution_status_root",
            "artifact_root",
        }:
            raise StageError("test-only root override has the wrong field set")
        return _Roots(**{key: Path(value).absolute() for key, value in _TEST_ONLY_ROOTS.items()})
    return _Roots(
        input_root=_CANONICAL_INPUT_ROOT,
        preexecution_root=_CANONICAL_PREEXECUTION_ROOT,
        execution_status_root=_CANONICAL_EXECUTION_STATUS_ROOT,
        artifact_root=_CANONICAL_ARTIFACT_ROOT,
    )


def _uuid4(value: Any, label: str) -> str:
    text = str(value)
    try:
        parsed = uuid.UUID(text)
    except (ValueError, AttributeError) as error:
        raise StageError(f"{label} must be one canonical UUID4") from error
    if parsed.version != 4 or str(parsed) != text:
        raise StageError(f"{label} must be one canonical UUID4")
    return text


def _safe_relative(value: Any) -> str:
    text = str(value)
    path = Path(text)
    if not text or path.is_absolute() or "." in path.parts or ".." in path.parts or path.as_posix() != text:
        raise StageError(f"unsafe relative path: {text!r}")
    return text


def _lexical_absolute(value: str | Path, label: str) -> Path:
    """Normalize dot components without resolving links or probing the filesystem."""

    try:
        raw = os.fspath(value)
    except (TypeError, ValueError) as error:
        raise StageError(f"{label} is not one lexical path") from error
    if type(raw) is not str or "\0" in raw:
        raise StageError(f"{label} must be a NUL-free text path")
    absolute = os.path.abspath(raw)
    if not absolute.startswith("/"):
        raise StageError(f"{label} is not one POSIX absolute path")
    lexical = Path("/" + absolute.lstrip("/"))
    if not lexical.is_absolute() or ".." in lexical.parts or "." in lexical.parts:
        raise StageError(f"{label} is not one normalized absolute path")
    return lexical


def _reject_checkpoint_a_path(path: Path, label: str) -> None:
    if path == _CANONICAL_CHECKPOINT_A_ROOT or _CANONICAL_CHECKPOINT_A_ROOT in path.parents:
        raise StageError(f"{label} is inside the forbidden canonical checkpoint-A namespace")


def _guard_source_stage_paths(capsule_freeze: str | Path) -> Path:
    running = _lexical_absolute(Path(__file__), "running stager")
    capsule = _lexical_absolute(capsule_freeze, "capsule freeze")
    _reject_checkpoint_a_path(running, "running stager")
    _reject_checkpoint_a_path(capsule, "capsule freeze")
    return capsule


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
    current = _open_directory_chain(path, f"{label} replay")
    try:
        observed = os.fstat(current)
        if not stat.S_ISDIR(observed.st_mode) or (held.st_dev, held.st_ino) != (
            observed.st_dev,
            observed.st_ino,
        ):
            raise StageError(f"{label} path no longer names the held directory")
    finally:
        os.close(current)


def _require_absent_at(parent: int, name: str, label: str) -> None:
    if not name or Path(name).name != name:
        raise StageError(f"{label} has an unsafe leaf name")
    try:
        os.stat(name, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError as error:
        raise StageError(f"could not inspect {label}") from error
    raise StageError(f"{label} must be absent; source staging is one-shot")


def _open_child(parent: int, name: str, label: str, mode: int | None = None) -> int:
    if not name or Path(name).name != name:
        raise StageError(f"{label} has an unsafe leaf name")
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent,
        )
    except OSError as error:
        raise StageError(f"{label} is not a real child directory") from error
    metadata = os.fstat(descriptor)
    if mode is not None and stat.S_IMODE(metadata.st_mode) != mode:
        os.close(descriptor)
        raise StageError(f"{label} mode must be {mode:04o}")
    return descriptor


def _regular_at(parent: int, name: str, label: str, mode: int | None = None) -> tuple[int, bytes]:
    if not name or Path(name).name != name:
        raise StageError(f"{label} has an unsafe leaf name")
    try:
        descriptor = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent)
    except OSError as error:
        raise StageError(f"{label} cannot be opened without links") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise StageError(f"{label} must be a single-link regular file")
        if mode is not None and stat.S_IMODE(metadata.st_mode) != mode:
            raise StageError(f"{label} mode must be {mode:04o}")
        payload = os.pread(descriptor, metadata.st_size, 0)
        if len(payload) != metadata.st_size:
            raise StageError(f"{label} changed while read")
        observed = os.stat(name, dir_fd=parent, follow_symlinks=False)
        if (metadata.st_dev, metadata.st_ino) != (observed.st_dev, observed.st_ino):
            raise StageError(f"{label} path changed while read")
        return descriptor, payload
    except BaseException:
        os.close(descriptor)
        raise


class _ArtifactSnapshot:
    def __init__(
        self,
        parent_path: Path,
        parent_fd: int,
        descriptors: dict[str, int],
        payloads: dict[str, bytes],
    ) -> None:
        self.parent_path = parent_path
        self.parent_fd = parent_fd
        self.descriptors = descriptors
        self.payloads = payloads

    @classmethod
    def capture(cls, freeze_path: Path) -> _ArtifactSnapshot:
        absolute = _lexical_absolute(freeze_path, "capsule freeze")
        _reject_checkpoint_a_path(absolute, "capsule freeze")
        if absolute.name != FREEZE_NAME:
            raise StageError(f"capsule freeze must be named exactly {FREEZE_NAME}")
        parent_fd = _open_directory_chain(absolute.parent, "capsule artifact parent")
        descriptors: dict[str, int] = {}
        payloads: dict[str, bytes] = {}
        try:
            parent_metadata = os.fstat(parent_fd)
            if not stat.S_ISDIR(parent_metadata.st_mode) or stat.S_IMODE(parent_metadata.st_mode) != 0o755:
                raise StageError("capsule artifact directory mode must be exactly 0755")
            if set(os.listdir(parent_fd)) != {FREEZE_NAME, MANIFEST_NAME, ARCHIVE_NAME}:
                raise StageError("capsule artifact directory must contain the exact three-file set")
            for name in (FREEZE_NAME, MANIFEST_NAME, ARCHIVE_NAME):
                descriptor, payload = _regular_at(parent_fd, name, f"capsule artifact {name}", 0o644)
                descriptors[name] = descriptor
                payloads[name] = payload
            return cls(absolute.parent, parent_fd, descriptors, payloads)
        except BaseException:
            for descriptor in descriptors.values():
                with suppress(OSError):
                    os.close(descriptor)
            os.close(parent_fd)
            raise

    def verify(self) -> None:
        _require_descriptor_matches_path(self.parent_fd, self.parent_path, "capsule artifact parent")
        parent_metadata = os.fstat(self.parent_fd)
        if not stat.S_ISDIR(parent_metadata.st_mode) or stat.S_IMODE(parent_metadata.st_mode) != 0o755:
            raise StageError("capsule artifact directory mode changed from exact 0755")
        if set(os.listdir(self.parent_fd)) != {FREEZE_NAME, MANIFEST_NAME, ARCHIVE_NAME}:
            raise StageError("capsule artifact directory inventory changed")
        for name, descriptor in self.descriptors.items():
            held = os.fstat(descriptor)
            current_fd, current_payload = _regular_at(
                self.parent_fd, name, f"capsule artifact replay {name}", 0o644
            )
            try:
                current = os.fstat(current_fd)
                if (
                    (held.st_dev, held.st_ino) != (current.st_dev, current.st_ino)
                    or os.pread(descriptor, held.st_size, 0) != self.payloads[name]
                    or current_payload != self.payloads[name]
                ):
                    raise StageError(f"capsule artifact changed after snapshot: {name}")
            finally:
                os.close(current_fd)

    def close(self) -> None:
        for descriptor in self.descriptors.values():
            with suppress(OSError):
                os.close(descriptor)
        with suppress(OSError):
            os.close(self.parent_fd)


def _directories() -> list[str]:
    values: set[str] = set()
    for relative in PAYLOAD_PATHS:
        parent = Path(relative).parent
        while parent.as_posix() != ".":
            values.add(parent.as_posix())
            parent = parent.parent
    return sorted(values)


def _source_digest(paths: Sequence[str], payloads: Mapping[str, bytes]) -> str:
    return digest({relative: sha256_bytes(payloads[relative]) for relative in paths})


def _goalzendo_fingerprint(payloads: Mapping[str, bytes]) -> str:
    paths = sorted(relative for relative in payloads if relative.startswith("src/goalzendo/"))
    accumulator = hashlib.sha256()
    accumulator.update(b"goalzendo-source-v1\0")
    for relative in paths:
        name = Path(relative).relative_to("src/goalzendo").as_posix().encode("utf-8")
        payload = payloads[relative]
        accumulator.update(len(name).to_bytes(8, "big"))
        accumulator.update(name)
        accumulator.update(len(payload).to_bytes(8, "big"))
        accumulator.update(payload)
    return accumulator.hexdigest()


def _expected_archive_bytes(payloads: Mapping[str, bytes]) -> bytes:
    output = io.BytesIO()
    with (
        gzip.GzipFile(filename="", mode="wb", fileobj=output, compresslevel=9, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT) as archive,
    ):
        for relative in PAYLOAD_PATHS:
            payload = payloads[relative]
            info = tarfile.TarInfo(relative)
            info.size = len(payload)
            info.mode = 0o444
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = SOURCE_DATE_EPOCH
            archive.addfile(info, io.BytesIO(payload))
    return output.getvalue()


def _verify_archive(archive_payload: bytes, members: Sequence[Mapping[str, Any]]) -> dict[str, bytes]:
    if (
        len(archive_payload) < 10
        or archive_payload[:4] != b"\x1f\x8b\x08\x00"
        or archive_payload[4:8] != b"\0\0\0\0"
    ):
        raise StageError("capsule gzip header is not exact and metadata-free")
    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    try:
        tar_payload = decompressor.decompress(archive_payload) + decompressor.flush()
    except zlib.error as error:
        raise StageError("capsule archive is not one valid gzip stream") from error
    if not decompressor.eof or decompressor.unused_data or decompressor.unconsumed_tail:
        raise StageError("capsule archive has concatenated or trailing gzip data")
    expected_rows = {str(row["path"]): row for row in members}
    payloads: dict[str, bytes] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(tar_payload), mode="r:") as archive:
            if archive.pax_headers:
                raise StageError("capsule archive has global PAX headers")
            infos = archive.getmembers()
            if [info.name for info in infos] != list(PAYLOAD_PATHS):
                raise StageError("capsule archive member order/inventory changed")
            for info in infos:
                relative = _safe_relative(info.name)
                row = expected_rows[relative]
                if (
                    info.type != tarfile.REGTYPE
                    or not info.isreg()
                    or info.linkname
                    or info.pax_headers
                    or stat.S_IMODE(info.mode) != 0o444
                    or info.uid != 0
                    or info.gid != 0
                    or info.uname != ""
                    or info.gname != ""
                    or info.mtime != SOURCE_DATE_EPOCH
                    or info.size != row["bytes"]
                ):
                    raise StageError(f"capsule archive metadata changed: {relative}")
                extracted = archive.extractfile(info)
                if extracted is None:
                    raise StageError(f"capsule archive member cannot be read: {relative}")
                payload = extracted.read()
                if len(payload) != row["bytes"] or sha256_bytes(payload) != row["sha256"]:
                    raise StageError(f"capsule archive member bytes changed: {relative}")
                payloads[relative] = payload
    except tarfile.TarError as error:
        raise StageError("capsule payload is not one strict tar archive") from error
    if _expected_archive_bytes(payloads) != archive_payload:
        raise StageError("capsule archive differs from exact deterministic USTAR/gzip encoding")
    return payloads


def _verify_capsule(
    snapshot: _ArtifactSnapshot,
    *,
    expected_freeze_sha256: str,
    expected_archive_sha256: str,
    expected_manifest_sha256: str,
    expected_stager_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, bytes]]:
    freeze_payload = snapshot.payloads[FREEZE_NAME]
    manifest_payload = snapshot.payloads[MANIFEST_NAME]
    archive_payload = snapshot.payloads[ARCHIVE_NAME]
    if sha256_bytes(freeze_payload) != require_sha256(expected_freeze_sha256, "capsule freeze SHA-256"):
        raise StageError("capsule freeze differs from external SHA-256")
    if sha256_bytes(archive_payload) != require_sha256(expected_archive_sha256, "capsule archive SHA-256"):
        raise StageError("capsule archive differs from external SHA-256")
    if sha256_bytes(manifest_payload) != require_sha256(expected_manifest_sha256, "capsule manifest SHA-256"):
        raise StageError("capsule manifest differs from external SHA-256")
    freeze = strict_json_bytes(freeze_payload, "capsule freeze")
    manifest = strict_json_bytes(manifest_payload, "capsule manifest")
    if freeze_payload != json_bytes(freeze):
        raise StageError("capsule freeze is not the exact canonical JSON encoding")
    if manifest_payload != json_bytes(manifest):
        raise StageError("capsule manifest is not the exact canonical JSON encoding")
    _reject_floats(freeze, "capsule freeze")
    _reject_floats(manifest, "capsule manifest")
    freeze_body = {key: value for key, value in freeze.items() if key != "freeze_digest"}
    manifest_body = {key: value for key, value in manifest.items() if key != "manifest_digest"}
    if set(manifest) != {
        "schema",
        "schema_version",
        "study_id",
        "source_date_epoch",
        "members",
        "directories",
        "identity",
        "authorization",
        "trust_limitations",
        "manifest_digest",
    }:
        raise StageError("capsule manifest field set changed")
    _exact_int(manifest.get("schema_version"), 1, "manifest schema_version")
    _exact_int(manifest.get("source_date_epoch"), SOURCE_DATE_EPOCH, "manifest source_date_epoch")
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("study_id") != STUDY_ID
        or manifest.get("authorization") != AUTHORIZATION
        or manifest.get("trust_limitations") != TRUST_LIMITATIONS
        or manifest.get("manifest_digest") != digest(manifest_body)
    ):
        raise StageError("capsule manifest fixed contract changed")
    _exact_equal(manifest.get("authorization"), dict(AUTHORIZATION), "manifest.authorization")
    _exact_equal(manifest.get("trust_limitations"), dict(TRUST_LIMITATIONS), "manifest.trust_limitations")
    rows = manifest.get("members")
    if not isinstance(rows, list) or len(rows) != len(PAYLOAD_PATHS):
        raise StageError("capsule manifest member count changed")
    normalized: list[Mapping[str, Any]] = []
    for expected_path, row in zip(PAYLOAD_PATHS, rows, strict=True):
        if (
            not isinstance(row, Mapping)
            or set(row) != {"path", "type", "mode", "bytes", "sha256"}
            or row.get("path") != expected_path
            or row.get("type") != "file"
            or row.get("mode") != 0o444
            or not isinstance(row.get("bytes"), int)
            or isinstance(row.get("bytes"), bool)
            or int(row["bytes"]) < 0
        ):
            raise StageError(f"capsule manifest member changed: {expected_path}")
        require_sha256(row.get("sha256"), f"manifest member {expected_path}")
        _exact_int(row.get("mode"), 0o444, f"manifest member mode {expected_path}")
        normalized.append(row)
    expected_directories = [
        {"path": relative, "type": "directory", "mode": 0o555} for relative in _directories()
    ]
    if manifest.get("directories") != expected_directories:
        raise StageError("capsule exact directory inventory changed")
    directories = manifest["directories"]
    for expected_path, row in zip(_directories(), directories, strict=True):
        if not isinstance(row, Mapping):
            raise StageError(f"capsule directory row is malformed: {expected_path}")
        _exact_int(row.get("mode"), 0o555, f"manifest directory mode {expected_path}")
    payloads = _verify_archive(archive_payload, normalized)
    for relative, expected in KNOWN_EXACT_SHA256.items():
        if sha256_bytes(payloads[relative]) != expected:
            raise StageError(f"known exact source changed: {relative}")
    bridge_paths = (
        "runs/goalzendo/run_g01_after_g00f_bridge.py",
        "src/goalzendo_g00f_g01_bridge/__init__.py",
        "src/goalzendo_g00f_g01_bridge/bridge.py",
        "src/goalzendo_g00f_g01_bridge/cli.py",
    )
    coordinator_paths = (
        "runs/goalzendo/run_g01_global_coordinator.py",
        "src/goalzendo_g01_coordinator/__init__.py",
        "src/goalzendo_g01_coordinator/coordinator.py",
    )
    identity = {
        "goalzendo_implementation_fingerprint": GOALZENDO_FINGERPRINT,
        "bridge_source_digest": BRIDGE_SOURCE_DIGEST,
        "coordinator_source_digest": COORDINATOR_SOURCE_DIGEST,
        "member_count": len(PAYLOAD_PATHS),
    }
    expected_transaction = {
        "canonical_input_root": "/workspace/inputs-goalzendo/g01-executions",
        "canonical_preexecution_root": "/workspace/status-goalzendo/g01-preexecution",
        "canonical_execution_status_root": "/workspace/status-goalzendo/g01-executions",
        "canonical_artifact_root": "/workspace/artifacts-goalzendo/g01-known-law",
        "frozen_source_directory": "frozen-source",
        "capsule_built_independent_of_execution_uuid": True,
        "canonical_checkpoint_a_namespace_never_read_or_written": True,
        "thin_token_consumed_later": True,
        "exclusive_one_shot_stage": True,
        "receipt_written_last": True,
    }
    expected_runtime = {
        "image": CANONICAL_IMAGE,
        "python": CANONICAL_PYTHON,
        "python_version": "3.12.3",
        "invocation": "-I -S",
        "zlib": "1.3",
        "trust_boundary": "externally_attested_rootfs_and_runtime_required",
        "python_lexical_path_checked_by_builder": True,
        "python_binary_authenticated_by_capsule": False,
        "image_authenticated_by_capsule": False,
        "rootfs_authenticated_by_capsule": False,
        "single_linux_task_before_temporary_umask_change": True,
    }
    if (
        _goalzendo_fingerprint(payloads) != GOALZENDO_FINGERPRINT
        or _source_digest(bridge_paths, payloads) != BRIDGE_SOURCE_DIGEST
        or _source_digest(coordinator_paths, payloads) != COORDINATOR_SOURCE_DIGEST
        or manifest.get("identity") != identity
    ):
        raise StageError("capsule source identities changed")
    if set(freeze) != {
        "schema",
        "schema_version",
        "study_id",
        "source_date_epoch",
        "generic_execution_uuid",
        "outcomes_seen",
        "bundle",
        "controller_files",
        "identity",
        "authorization",
        "transaction",
        "required_build_runtime_claim",
        "trust_limitations",
        "freeze_digest",
    }:
        raise StageError("capsule freeze field set changed")
    bundle = freeze.get("bundle")
    controllers = freeze.get("controller_files")
    transaction = freeze.get("transaction")
    runtime = freeze.get("required_build_runtime_claim")
    _exact_equal(freeze.get("authorization"), dict(AUTHORIZATION), "freeze.authorization")
    _exact_equal(freeze.get("trust_limitations"), dict(TRUST_LIMITATIONS), "freeze.trust_limitations")
    _exact_equal(transaction, expected_transaction, "freeze.transaction")
    _exact_equal(runtime, expected_runtime, "freeze.required_build_runtime_claim")
    stager_sha = require_sha256(expected_stager_sha256, "external stager SHA-256")
    manifest_identity = manifest.get("identity")
    if not isinstance(manifest_identity, Mapping):
        raise StageError("capsule manifest identity is malformed")
    _exact_int(manifest_identity.get("member_count"), len(PAYLOAD_PATHS), "identity member_count")
    _exact_int(freeze.get("schema_version"), 1, "freeze schema_version")
    _exact_int(freeze.get("source_date_epoch"), SOURCE_DATE_EPOCH, "freeze source_date_epoch")
    if isinstance(bundle, Mapping):
        _exact_int(bundle.get("file_member_count"), len(PAYLOAD_PATHS), "bundle file_member_count")
        _exact_int(bundle.get("directory_count"), len(expected_directories), "bundle directory_count")
    if (
        freeze.get("schema") != FREEZE_SCHEMA
        or freeze.get("study_id") != STUDY_ID
        or freeze.get("generic_execution_uuid") is not None
        or freeze.get("outcomes_seen") is not False
        or freeze.get("identity") != identity
        or freeze.get("authorization") != AUTHORIZATION
        or freeze.get("trust_limitations") != TRUST_LIMITATIONS
        or freeze.get("freeze_digest") != digest(freeze_body)
        or bundle
        != {
            "archive_name": ARCHIVE_NAME,
            "archive_sha256": sha256_bytes(archive_payload),
            "manifest_name": MANIFEST_NAME,
            "manifest_sha256": sha256_bytes(manifest_payload),
            "manifest_digest": manifest["manifest_digest"],
            "file_member_count": len(PAYLOAD_PATHS),
            "directory_count": len(expected_directories),
        }
        or not isinstance(controllers, Mapping)
        or set(controllers) != {"builder", "stager_and_fresh_verifier"}
        or controllers.get("stager_and_fresh_verifier") != {"path": STAGER_PATH, "sha256": stager_sha}
        or not isinstance(controllers.get("builder"), Mapping)
        or set(controllers["builder"]) != {"path", "sha256"}
        or controllers["builder"].get("path") != "runs/goalzendo/build_g01_b_source_capsule.py"
        or require_sha256(controllers["builder"].get("sha256"), "frozen builder SHA-256")
        != sha256_bytes(payloads["runs/goalzendo/build_g01_b_source_capsule.py"])
    ):
        raise StageError("capsule freeze fixed contract changed")
    if sha256_bytes(payloads[STAGER_PATH]) != stager_sha:
        raise StageError("archived stager differs from external stager SHA-256")
    snapshot.verify()
    return freeze, manifest, payloads


def _verify_running_stager(expected_sha256: str) -> None:
    expected = require_sha256(expected_sha256, "external stager SHA-256")
    path = _lexical_absolute(Path(__file__), "running stager")
    _reject_checkpoint_a_path(path, "running stager")
    parent = _open_directory_chain(path.parent, "running stager parent")
    try:
        descriptor, payload = _regular_at(parent, path.name, "running stager")
        try:
            if sha256_bytes(payload) != expected:
                raise StageError("running stager differs from external SHA-256")
        finally:
            os.close(descriptor)
        _require_descriptor_matches_path(parent, path.parent, "running stager parent")
    finally:
        os.close(parent)


def _mkdir_child(parent: int, name: str, mode: int, label: str) -> int:
    previous_umask = os.umask(0)
    changed_umask: int | None = None
    try:
        os.mkdir(name, mode, dir_fd=parent)
    except FileExistsError as error:
        raise StageError(f"{label} already exists; partial source stages are permanent") from error
    finally:
        changed_umask = os.umask(previous_umask)
    if changed_umask != 0:
        raise StageError("process umask changed concurrently during exact directory creation")
    os.fsync(parent)
    descriptor = _open_child(parent, name, label)
    os.fchmod(descriptor, mode)
    os.fsync(descriptor)
    os.fsync(parent)
    metadata = os.fstat(descriptor)
    if stat.S_IMODE(metadata.st_mode) != mode:
        os.close(descriptor)
        raise StageError(f"{label} initial mode changed")
    return descriptor


def _require_held_child(
    parent: int,
    name: str,
    held: int,
    label: str,
    mode: int,
) -> None:
    current = _open_child(parent, name, label, mode)
    try:
        expected = os.fstat(held)
        observed = os.fstat(current)
        if (
            not stat.S_ISDIR(expected.st_mode)
            or not stat.S_ISDIR(observed.st_mode)
            or stat.S_IMODE(expected.st_mode) != mode
            or (expected.st_dev, expected.st_ino) != (observed.st_dev, observed.st_ino)
        ):
            raise StageError(f"{label} path no longer names the held directory")
    finally:
        os.close(current)


def _recheck_transaction_directories(
    *,
    input_fd: int,
    preexecution_fd: int,
    execution_uuid: str,
    execution_fd: int,
    frozen_fd: int,
    preexecution_execution_fd: int | None,
) -> None:
    _require_held_child(input_fd, execution_uuid, execution_fd, "G01 input UUID root", 0o555)
    _require_held_child(execution_fd, "frozen-source", frozen_fd, "G01 frozen-source root", 0o555)
    if preexecution_execution_fd is not None:
        _require_held_child(
            preexecution_fd,
            execution_uuid,
            preexecution_execution_fd,
            "G01 preexecution UUID root",
            0o700,
        )


def _write_held_file(parent: int, name: str, payload: bytes, mode: int, label: str) -> int:
    try:
        writer = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent,
        )
    except FileExistsError as error:
        raise StageError(f"{label} already exists; partial source stages are permanent") from error
    try:
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(writer, view[written:])
            if count <= 0:
                raise StageError(f"{label} write made no progress")
            written += count
        os.fchmod(writer, mode)
        os.fsync(writer)
        written_metadata = os.fstat(writer)
        reader, observed = _regular_at(parent, name, label, mode)
        read_metadata = os.fstat(reader)
        if (written_metadata.st_dev, written_metadata.st_ino) != (
            read_metadata.st_dev,
            read_metadata.st_ino,
        ) or observed != payload:
            os.close(reader)
            raise StageError(f"{label} changed after exclusive write")
        os.fsync(parent)
        return reader
    finally:
        os.close(writer)


def _create_source_tree(
    frozen_fd: int,
    manifest: Mapping[str, Any],
    payloads: Mapping[str, bytes],
) -> tuple[dict[str, int], dict[str, int]]:
    directory_fds: dict[str, int] = {"": os.dup(frozen_fd)}
    file_fds: dict[str, int] = {}
    try:
        for relative in sorted(_directories(), key=lambda value: (len(Path(value).parts), value)):
            path = Path(relative)
            parent_name = path.parent.as_posix()
            if parent_name == ".":
                parent_name = ""
            directory_fds[relative] = _mkdir_child(
                directory_fds[parent_name], path.name, 0o700, f"source directory {relative}"
            )
        members = manifest["members"]
        member_rows = {str(row["path"]): row for row in members}
        for relative in PAYLOAD_PATHS:
            path = Path(relative)
            parent_name = path.parent.as_posix()
            if parent_name == ".":
                parent_name = ""
            row = member_rows[relative]
            payload = payloads[relative]
            if len(payload) != row["bytes"] or sha256_bytes(payload) != row["sha256"]:
                raise StageError(f"source payload changed before write: {relative}")
            file_fds[relative] = _write_held_file(
                directory_fds[parent_name], path.name, payload, 0o444, f"staged source {relative}"
            )
        for relative in sorted(directory_fds, key=lambda value: len(Path(value).parts), reverse=True):
            os.fchmod(directory_fds[relative], 0o555)
            os.fsync(directory_fds[relative])
        return directory_fds, file_fds
    except BaseException:
        for descriptor in file_fds.values():
            with suppress(OSError):
                os.close(descriptor)
        for descriptor in directory_fds.values():
            with suppress(OSError):
                os.close(descriptor)
        raise


def _verify_held_tree(
    frozen_fd: int,
    directory_fds: Mapping[str, int],
    file_fds: Mapping[str, int],
    payloads: Mapping[str, bytes],
) -> None:
    expected_children: dict[str, set[str]] = {relative: set() for relative in directory_fds}
    for relative in directory_fds:
        if not relative:
            continue
        path = Path(relative)
        parent = path.parent.as_posix()
        if parent == ".":
            parent = ""
        expected_children[parent].add(path.name)
    for relative in PAYLOAD_PATHS:
        path = Path(relative)
        parent = path.parent.as_posix()
        if parent == ".":
            parent = ""
        expected_children[parent].add(path.name)
    root_metadata = os.fstat(frozen_fd)
    held_root = os.fstat(directory_fds[""])
    if (root_metadata.st_dev, root_metadata.st_ino) != (held_root.st_dev, held_root.st_ino) or stat.S_IMODE(
        root_metadata.st_mode
    ) != 0o555:
        raise StageError("held frozen-source root changed")
    for relative, descriptor in directory_fds.items():
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o555:
            raise StageError(f"staged directory mode/type changed: {relative or '.'}")
        if set(os.listdir(descriptor)) != expected_children[relative]:
            raise StageError(f"staged directory inventory changed: {relative or '.'}")
        if relative:
            path = Path(relative)
            parent_name = path.parent.as_posix()
            if parent_name == ".":
                parent_name = ""
            current = _open_child(
                directory_fds[parent_name], path.name, f"directory replay {relative}", 0o555
            )
            try:
                observed = os.fstat(current)
                if (metadata.st_dev, metadata.st_ino) != (observed.st_dev, observed.st_ino):
                    raise StageError(f"staged directory path changed: {relative}")
            finally:
                os.close(current)
    for relative, descriptor in file_fds.items():
        held = os.fstat(descriptor)
        path = Path(relative)
        parent_name = path.parent.as_posix()
        if parent_name == ".":
            parent_name = ""
        current, observed_payload = _regular_at(
            directory_fds[parent_name], path.name, f"source replay {relative}", 0o444
        )
        try:
            observed = os.fstat(current)
            if (
                (held.st_dev, held.st_ino) != (observed.st_dev, observed.st_ino)
                or held.st_nlink != 1
                or observed.st_nlink != 1
                or os.pread(descriptor, held.st_size, 0) != payloads[relative]
                or observed_payload != payloads[relative]
            ):
                raise StageError(f"staged source changed: {relative}")
        finally:
            os.close(current)


def _close_descriptors(values: Mapping[str, int]) -> None:
    for descriptor in values.values():
        with suppress(OSError):
            os.close(descriptor)


def _open_root_set(roots: _Roots) -> tuple[int, int, int, int]:
    input_fd = _open_directory_chain(roots.input_root, "canonical G01 input root")
    preexecution_fd: int | None = None
    status_fd: int | None = None
    artifact_parent_fd: int | None = None
    try:
        preexecution_fd = _open_directory_chain(roots.preexecution_root, "canonical G01 preexecution root")
        status_fd = _open_directory_chain(roots.execution_status_root, "canonical G01 execution-status root")
        artifact_parent_fd = _open_directory_chain(
            roots.artifact_root.parent, "canonical G01 artifact parent"
        )
        identities = {
            (metadata.st_dev, metadata.st_ino)
            for metadata in (
                os.fstat(input_fd),
                os.fstat(preexecution_fd),
                os.fstat(status_fd),
                os.fstat(artifact_parent_fd),
            )
        }
        if len(identities) != 4:
            raise StageError("canonical G01 roots must be four distinct directories")
        return input_fd, preexecution_fd, status_fd, artifact_parent_fd
    except BaseException:
        if artifact_parent_fd is not None:
            os.close(artifact_parent_fd)
        if status_fd is not None:
            os.close(status_fd)
        if preexecution_fd is not None:
            os.close(preexecution_fd)
        os.close(input_fd)
        raise


def _recheck_roots(
    roots: _Roots,
    input_fd: int,
    preexecution_fd: int,
    status_fd: int,
    artifact_parent_fd: int,
) -> None:
    _require_descriptor_matches_path(input_fd, roots.input_root, "canonical G01 input root")
    _require_descriptor_matches_path(
        preexecution_fd, roots.preexecution_root, "canonical G01 preexecution root"
    )
    _require_descriptor_matches_path(
        status_fd, roots.execution_status_root, "canonical G01 execution-status root"
    )
    _require_descriptor_matches_path(
        artifact_parent_fd, roots.artifact_root.parent, "canonical G01 artifact parent"
    )


def _require_execution_absences(
    execution_uuid: str,
    roots: _Roots,
    status_fd: int,
    artifact_parent_fd: int,
) -> None:
    _require_absent_at(status_fd, execution_uuid, "G01 execution-status UUID root")
    _require_absent_at(artifact_parent_fd, roots.artifact_root.name, "G01 artifact root")


def _parse_utc(value: Any) -> str:
    text = str(value)
    if not text.endswith("Z"):
        raise StageError("receipt created_at_utc must be canonical UTC")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as error:
        raise StageError("receipt created_at_utc is invalid") from error
    if (
        parsed.tzinfo != timezone.utc
        or parsed.isoformat(timespec="microseconds").replace("+00:00", "Z") != text
    ):
        raise StageError("receipt created_at_utc is not canonical microsecond UTC")
    return text


def _receipt_body(
    *,
    execution_uuid: str,
    eligibility_execution_uuid: str,
    roots: _Roots,
    snapshot: _ArtifactSnapshot,
    freeze: Mapping[str, Any],
    manifest: Mapping[str, Any],
    expected_freeze_sha256: str,
    expected_archive_sha256: str,
    expected_manifest_sha256: str,
    expected_stager_sha256: str,
    created_at_utc: str,
) -> dict[str, Any]:
    frozen_source = roots.input_root / execution_uuid / "frozen-source"
    receipt_path = roots.preexecution_root / execution_uuid / RECEIPT_NAME
    return {
        "schema": RECEIPT_SCHEMA,
        "schema_version": 1,
        "study_id": STUDY_ID,
        "created_at_utc": created_at_utc,
        "g01_execution_uuid": execution_uuid,
        "eligibility_execution_uuid": eligibility_execution_uuid,
        "uuid_relation": {
            "distinct": True,
            "eligibility_uuid_source": "operator_supplied_compare_only",
            "eligibility_uuid_authenticated_by_source_stage": False,
            "thin_token_read_by_source_stage": False,
        },
        "paths": {
            "frozen_source": str(frozen_source),
            "preexecution_receipt": str(receipt_path),
            "execution_status_root_must_remain_absent": str(roots.execution_status_root / execution_uuid),
            "artifact_root_must_remain_absent": str(roots.artifact_root),
        },
        "capsule": {
            "freeze_path": str(snapshot.parent_path / FREEZE_NAME),
            "freeze_file_sha256": require_sha256(expected_freeze_sha256, "capsule freeze SHA-256"),
            "freeze_digest": freeze["freeze_digest"],
            "archive_path": str(snapshot.parent_path / ARCHIVE_NAME),
            "archive_sha256": require_sha256(expected_archive_sha256, "capsule archive SHA-256"),
            "manifest_path": str(snapshot.parent_path / MANIFEST_NAME),
            "manifest_sha256": require_sha256(expected_manifest_sha256, "capsule manifest SHA-256"),
            "manifest_digest": manifest["manifest_digest"],
            "file_member_count": len(PAYLOAD_PATHS),
            "directory_count": len(_directories()),
        },
        "stager": {"path": STAGER_PATH, "sha256": require_sha256(expected_stager_sha256, "stager SHA-256")},
        "identity": manifest["identity"],
        "authorization": dict(AUTHORIZATION),
        "transaction": {
            "payload_writes_confined_to_g01_input_frozen_source": True,
            "preexecution_parent_mode": 0o700,
            "receipt_mode": 0o400,
            "source_file_mode": 0o444,
            "source_directory_mode": 0o555,
            "single_link_regular_files": True,
            "receipt_written_last": True,
            "partial_failure_is_permanent": True,
            "canonical_checkpoint_a_path_or_token_opened": False,
        },
        "trust_limitations": dict(TRUST_LIMITATIONS),
    }


def _verify_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_body: Mapping[str, Any],
) -> None:
    _reject_floats(receipt, "source-stage receipt")
    _exact_int(receipt.get("schema_version"), 1, "receipt schema_version")
    if set(receipt) != {*expected_body, "receipt_digest"}:
        raise StageError("source-stage receipt field set changed")
    body = {key: value for key, value in receipt.items() if key != "receipt_digest"}
    _parse_utc(receipt.get("created_at_utc"))
    _exact_equal(body, expected_body, "source-stage receipt")
    if receipt.get("receipt_digest") != digest(body):
        raise StageError("source-stage receipt binding changed")


def _stage(
    *,
    g01_execution_uuid: str,
    eligibility_execution_uuid: str,
    capsule_freeze: str | Path,
    expected_capsule_freeze_sha256: str,
    expected_capsule_archive_sha256: str,
    expected_capsule_manifest_sha256: str,
    expected_stager_sha256: str,
) -> dict[str, Any]:
    _require_runtime()
    capsule_path = _guard_source_stage_paths(capsule_freeze)
    _verify_running_stager(expected_stager_sha256)
    execution = _uuid4(g01_execution_uuid, "G01 execution UUID")
    eligibility = _uuid4(eligibility_execution_uuid, "eligibility execution UUID")
    if execution == eligibility:
        raise StageError("G01 and eligibility execution UUIDs must be distinct")
    snapshot = _ArtifactSnapshot.capture(capsule_path)
    roots = _roots()
    input_fd = preexecution_fd = status_fd = artifact_parent_fd = -1
    execution_fd = frozen_fd = preexecution_execution_fd = -1
    directory_fds: dict[str, int] = {}
    file_fds: dict[str, int] = {}
    receipt_fd = -1
    try:
        freeze, manifest, payloads = _verify_capsule(
            snapshot,
            expected_freeze_sha256=expected_capsule_freeze_sha256,
            expected_archive_sha256=expected_capsule_archive_sha256,
            expected_manifest_sha256=expected_capsule_manifest_sha256,
            expected_stager_sha256=expected_stager_sha256,
        )
        input_fd, preexecution_fd, status_fd, artifact_parent_fd = _open_root_set(roots)
        _recheck_roots(roots, input_fd, preexecution_fd, status_fd, artifact_parent_fd)
        _require_absent_at(input_fd, execution, "G01 input UUID root")
        _require_absent_at(preexecution_fd, execution, "G01 preexecution UUID root")
        _require_execution_absences(execution, roots, status_fd, artifact_parent_fd)
        snapshot.verify()
        # First mutation.  Every fallible source/authentication/absence check is
        # complete.  Any interruption from this point is permanently occupied.
        execution_fd = _mkdir_child(input_fd, execution, 0o700, "G01 input UUID root")
        frozen_fd = _mkdir_child(execution_fd, "frozen-source", 0o700, "G01 frozen-source root")
        directory_fds, file_fds = _create_source_tree(frozen_fd, manifest, payloads)
        os.fchmod(frozen_fd, 0o555)
        os.fchmod(execution_fd, 0o555)
        os.fsync(frozen_fd)
        os.fsync(execution_fd)
        os.fsync(input_fd)
        _recheck_transaction_directories(
            input_fd=input_fd,
            preexecution_fd=preexecution_fd,
            execution_uuid=execution,
            execution_fd=execution_fd,
            frozen_fd=frozen_fd,
            preexecution_execution_fd=None,
        )
        _verify_held_tree(frozen_fd, directory_fds, file_fds, payloads)
        _require_execution_absences(execution, roots, status_fd, artifact_parent_fd)
        snapshot.verify()
        preexecution_execution_fd = _mkdir_child(
            preexecution_fd, execution, 0o700, "G01 preexecution UUID root"
        )
        _recheck_transaction_directories(
            input_fd=input_fd,
            preexecution_fd=preexecution_fd,
            execution_uuid=execution,
            execution_fd=execution_fd,
            frozen_fd=frozen_fd,
            preexecution_execution_fd=preexecution_execution_fd,
        )
        created_at = datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
        body = _receipt_body(
            execution_uuid=execution,
            eligibility_execution_uuid=eligibility,
            roots=roots,
            snapshot=snapshot,
            freeze=freeze,
            manifest=manifest,
            expected_freeze_sha256=expected_capsule_freeze_sha256,
            expected_archive_sha256=expected_capsule_archive_sha256,
            expected_manifest_sha256=expected_capsule_manifest_sha256,
            expected_stager_sha256=expected_stager_sha256,
            created_at_utc=created_at,
        )
        receipt = {**body, "receipt_digest": digest(body)}
        _recheck_transaction_directories(
            input_fd=input_fd,
            preexecution_fd=preexecution_fd,
            execution_uuid=execution,
            execution_fd=execution_fd,
            frozen_fd=frozen_fd,
            preexecution_execution_fd=preexecution_execution_fd,
        )
        _verify_held_tree(frozen_fd, directory_fds, file_fds, payloads)
        _require_execution_absences(execution, roots, status_fd, artifact_parent_fd)
        snapshot.verify()
        # The receipt is the final write in the transaction.
        receipt_fd = _write_held_file(
            preexecution_execution_fd,
            RECEIPT_NAME,
            json_bytes(receipt),
            0o400,
            "source-stage receipt",
        )
        _recheck_transaction_directories(
            input_fd=input_fd,
            preexecution_fd=preexecution_fd,
            execution_uuid=execution,
            execution_fd=execution_fd,
            frozen_fd=frozen_fd,
            preexecution_execution_fd=preexecution_execution_fd,
        )
        _verify_held_tree(frozen_fd, directory_fds, file_fds, payloads)
        _require_execution_absences(execution, roots, status_fd, artifact_parent_fd)
        snapshot.verify()
        _recheck_roots(roots, input_fd, preexecution_fd, status_fd, artifact_parent_fd)
        receipt_sha = sha256_bytes(json_bytes(receipt))
    finally:
        if receipt_fd >= 0:
            os.close(receipt_fd)
        _close_descriptors(file_fds)
        _close_descriptors(directory_fds)
        for descriptor in (
            preexecution_execution_fd,
            frozen_fd,
            execution_fd,
            artifact_parent_fd,
            status_fd,
            preexecution_fd,
            input_fd,
        ):
            if descriptor >= 0:
                with suppress(OSError):
                    os.close(descriptor)
        snapshot.close()
    return verify_stage(
        g01_execution_uuid=execution,
        eligibility_execution_uuid=eligibility,
        capsule_freeze=capsule_freeze,
        expected_capsule_freeze_sha256=expected_capsule_freeze_sha256,
        expected_capsule_archive_sha256=expected_capsule_archive_sha256,
        expected_capsule_manifest_sha256=expected_capsule_manifest_sha256,
        expected_stager_sha256=expected_stager_sha256,
        expected_stage_receipt_sha256=receipt_sha,
    )


def _open_existing_source_tree(
    input_fd: int, execution: str
) -> tuple[int, int, dict[str, int], dict[str, int], dict[str, bytes]]:
    execution_fd = _open_child(input_fd, execution, "G01 input UUID root", 0o555)
    frozen_fd = _open_child(execution_fd, "frozen-source", "G01 frozen-source root", 0o555)
    if set(os.listdir(execution_fd)) != {"frozen-source"}:
        os.close(frozen_fd)
        os.close(execution_fd)
        raise StageError("G01 input UUID root inventory changed")
    directory_fds: dict[str, int] = {"": os.dup(frozen_fd)}
    file_fds: dict[str, int] = {}
    payloads: dict[str, bytes] = {}
    try:
        for relative in sorted(_directories(), key=lambda value: (len(Path(value).parts), value)):
            path = Path(relative)
            parent_name = path.parent.as_posix()
            if parent_name == ".":
                parent_name = ""
            directory_fds[relative] = _open_child(
                directory_fds[parent_name], path.name, f"staged directory {relative}", 0o555
            )
        for relative in PAYLOAD_PATHS:
            path = Path(relative)
            parent_name = path.parent.as_posix()
            if parent_name == ".":
                parent_name = ""
            descriptor, payload = _regular_at(
                directory_fds[parent_name], path.name, f"staged source {relative}", 0o444
            )
            file_fds[relative] = descriptor
            payloads[relative] = payload
        _verify_held_tree(frozen_fd, directory_fds, file_fds, payloads)
        return execution_fd, frozen_fd, directory_fds, file_fds, payloads
    except BaseException:
        _close_descriptors(file_fds)
        _close_descriptors(directory_fds)
        os.close(frozen_fd)
        os.close(execution_fd)
        raise


def verify_stage(
    *,
    g01_execution_uuid: str,
    eligibility_execution_uuid: str,
    capsule_freeze: str | Path,
    expected_capsule_freeze_sha256: str,
    expected_capsule_archive_sha256: str,
    expected_capsule_manifest_sha256: str,
    expected_stager_sha256: str,
    expected_stage_receipt_sha256: str,
) -> dict[str, Any]:
    """Freshly reopen and replay every capsule, tree, receipt, and absence binding."""

    _require_runtime()
    capsule_path = _guard_source_stage_paths(capsule_freeze)
    _verify_running_stager(expected_stager_sha256)
    execution = _uuid4(g01_execution_uuid, "G01 execution UUID")
    eligibility = _uuid4(eligibility_execution_uuid, "eligibility execution UUID")
    if execution == eligibility:
        raise StageError("G01 and eligibility execution UUIDs must be distinct")
    snapshot = _ArtifactSnapshot.capture(capsule_path)
    roots = _roots()
    input_fd = preexecution_fd = status_fd = artifact_parent_fd = -1
    execution_fd = frozen_fd = preexecution_execution_fd = receipt_fd = -1
    directory_fds: dict[str, int] = {}
    file_fds: dict[str, int] = {}
    try:
        freeze, manifest, capsule_payloads = _verify_capsule(
            snapshot,
            expected_freeze_sha256=expected_capsule_freeze_sha256,
            expected_archive_sha256=expected_capsule_archive_sha256,
            expected_manifest_sha256=expected_capsule_manifest_sha256,
            expected_stager_sha256=expected_stager_sha256,
        )
        input_fd, preexecution_fd, status_fd, artifact_parent_fd = _open_root_set(roots)
        _recheck_roots(roots, input_fd, preexecution_fd, status_fd, artifact_parent_fd)
        _require_execution_absences(execution, roots, status_fd, artifact_parent_fd)
        execution_fd, frozen_fd, directory_fds, file_fds, staged_payloads = _open_existing_source_tree(
            input_fd, execution
        )
        if staged_payloads != capsule_payloads:
            raise StageError("staged source bytes differ from authenticated capsule")
        preexecution_execution_fd = _open_child(
            preexecution_fd, execution, "G01 preexecution UUID root", 0o700
        )
        if set(os.listdir(preexecution_execution_fd)) != {RECEIPT_NAME}:
            raise StageError("G01 preexecution UUID root inventory changed")
        receipt_fd, receipt_payload = _regular_at(
            preexecution_execution_fd, RECEIPT_NAME, "source-stage receipt", 0o400
        )
        if sha256_bytes(receipt_payload) != require_sha256(
            expected_stage_receipt_sha256, "source-stage receipt SHA-256"
        ):
            raise StageError("source-stage receipt differs from external SHA-256")
        receipt = strict_json_bytes(receipt_payload, "source-stage receipt")
        if receipt_payload != json_bytes(receipt):
            raise StageError("source-stage receipt is not the exact canonical JSON encoding")
        created_at = _parse_utc(receipt.get("created_at_utc"))
        expected_body = _receipt_body(
            execution_uuid=execution,
            eligibility_execution_uuid=eligibility,
            roots=roots,
            snapshot=snapshot,
            freeze=freeze,
            manifest=manifest,
            expected_freeze_sha256=expected_capsule_freeze_sha256,
            expected_archive_sha256=expected_capsule_archive_sha256,
            expected_manifest_sha256=expected_capsule_manifest_sha256,
            expected_stager_sha256=expected_stager_sha256,
            created_at_utc=created_at,
        )
        _verify_receipt(receipt, expected_body=expected_body)
        _recheck_transaction_directories(
            input_fd=input_fd,
            preexecution_fd=preexecution_fd,
            execution_uuid=execution,
            execution_fd=execution_fd,
            frozen_fd=frozen_fd,
            preexecution_execution_fd=preexecution_execution_fd,
        )
        _verify_held_tree(frozen_fd, directory_fds, file_fds, staged_payloads)
        held_receipt = os.fstat(receipt_fd)
        current_receipt_fd, current_receipt = _regular_at(
            preexecution_execution_fd, RECEIPT_NAME, "source-stage receipt replay", 0o400
        )
        try:
            observed_receipt = os.fstat(current_receipt_fd)
            if (
                (held_receipt.st_dev, held_receipt.st_ino)
                != (observed_receipt.st_dev, observed_receipt.st_ino)
                or held_receipt.st_nlink != 1
                or current_receipt != receipt_payload
                or os.pread(receipt_fd, held_receipt.st_size, 0) != receipt_payload
            ):
                raise StageError("held source-stage receipt changed")
        finally:
            os.close(current_receipt_fd)
        _require_execution_absences(execution, roots, status_fd, artifact_parent_fd)
        _recheck_transaction_directories(
            input_fd=input_fd,
            preexecution_fd=preexecution_fd,
            execution_uuid=execution,
            execution_fd=execution_fd,
            frozen_fd=frozen_fd,
            preexecution_execution_fd=preexecution_execution_fd,
        )
        snapshot.verify()
        _recheck_roots(roots, input_fd, preexecution_fd, status_fd, artifact_parent_fd)
    finally:
        _close_descriptors(file_fds)
        _close_descriptors(directory_fds)
        for descriptor in (
            receipt_fd,
            preexecution_execution_fd,
            frozen_fd,
            execution_fd,
            artifact_parent_fd,
            status_fd,
            preexecution_fd,
            input_fd,
        ):
            if descriptor >= 0:
                with suppress(OSError):
                    os.close(descriptor)
        snapshot.close()
    return {
        "verified": True,
        "g01_execution_uuid": execution,
        "eligibility_execution_uuid": eligibility,
        "receipt_path": str(roots.preexecution_root / execution / RECEIPT_NAME),
        "receipt_sha256": require_sha256(expected_stage_receipt_sha256, "source-stage receipt SHA-256"),
        "file_member_count": len(PAYLOAD_PATHS),
        "exact_source_inventory": True,
        "g01_launch_authorized": False,
        "runtime_overlay_frozen": False,
        "accepted_refusal_revision_contained": True,
        "supported_launch_entrypoint_activated": False,
    }


def stage(
    *,
    g01_execution_uuid: str,
    eligibility_execution_uuid: str,
    capsule_freeze: str | Path,
    expected_capsule_freeze_sha256: str,
    expected_capsule_archive_sha256: str,
    expected_capsule_manifest_sha256: str,
    expected_stager_sha256: str,
) -> dict[str, Any]:
    return _stage(
        g01_execution_uuid=g01_execution_uuid,
        eligibility_execution_uuid=eligibility_execution_uuid,
        capsule_freeze=capsule_freeze,
        expected_capsule_freeze_sha256=expected_capsule_freeze_sha256,
        expected_capsule_archive_sha256=expected_capsule_archive_sha256,
        expected_capsule_manifest_sha256=expected_capsule_manifest_sha256,
        expected_stager_sha256=expected_stager_sha256,
    )


def main() -> int:
    parser = argparse.ArgumentParser(prog="g01_b_source_capsule_stage")
    commands = parser.add_subparsers(dest="command", required=True)

    def common(target: argparse.ArgumentParser) -> None:
        target.add_argument("--g01-execution-uuid", required=True)
        target.add_argument("--eligibility-execution-uuid", required=True)
        target.add_argument("--capsule-freeze", type=Path, required=True)
        target.add_argument("--expected-capsule-freeze-sha256", required=True)
        target.add_argument("--expected-capsule-archive-sha256", required=True)
        target.add_argument("--expected-capsule-manifest-sha256", required=True)
        target.add_argument("--expected-stager-sha256", required=True)

    stage_parser = commands.add_parser("stage")
    common(stage_parser)
    verify_parser = commands.add_parser("verify-stage")
    common(verify_parser)
    verify_parser.add_argument("--expected-stage-receipt-sha256", required=True)
    arguments = parser.parse_args()
    kwargs = {
        "g01_execution_uuid": arguments.g01_execution_uuid,
        "eligibility_execution_uuid": arguments.eligibility_execution_uuid,
        "capsule_freeze": arguments.capsule_freeze,
        "expected_capsule_freeze_sha256": arguments.expected_capsule_freeze_sha256,
        "expected_capsule_archive_sha256": arguments.expected_capsule_archive_sha256,
        "expected_capsule_manifest_sha256": arguments.expected_capsule_manifest_sha256,
        "expected_stager_sha256": arguments.expected_stager_sha256,
    }
    try:
        if arguments.command == "stage":
            result = stage(**kwargs)
        else:
            result = verify_stage(
                **kwargs,
                expected_stage_receipt_sha256=arguments.expected_stage_receipt_sha256,
            )
    except (OSError, StageError, tarfile.TarError) as error:
        print(f"g01_b_source_capsule_stage: error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
