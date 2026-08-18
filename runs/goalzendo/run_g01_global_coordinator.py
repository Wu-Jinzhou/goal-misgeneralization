#!/usr/bin/env python3
"""Stdlib-only isolated bootstrap for the sole checkpoint-B G01 entrypoint.

This file performs no project or third-party import.  It captures and
authenticates the frozen inputs, replays their held identities, then refuses at
the explicit missing-B-overlay boundary.  A future authenticated module loader
is one of the separately reviewed overlay prerequisites.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from contextlib import suppress
from pathlib import Path
from types import MappingProxyType
from typing import Any

_RUNTIME_REFUSAL = "B_RUNTIME_OVERLAY_NOT_FROZEN"
_G01_SOURCE_FINGERPRINT = "1a8146377b9a9620690025671614edb2dd20d214f4e195da3cf3528809b2c694"
_G01_CONFIG_SHA256 = "ee6d53556189a6b6e25cfbfb204325017a058c63a605653df4c175463e00ce18"
_G01_BASE_CONFIG_SHA256 = "34cb42eb2bf7ea275247ecf8baa0e5556565c2c86c30da0aec8e2a1a630e04fb"
_G01_PROTOCOL_SHA256 = "428243c3da271bc2d79e16c0fde20d47076eb3837c6740ab733e278cddcdbce9"
_G01_RUNNER_SHA256 = "46b55ad4bdd08073e5f89ae101e0862372817b74b590f07ddb8f8c4331e5b9e1"
_BRIDGE_PATHS = (
    "src/goalzendo_g00f_g01_bridge/__init__.py",
    "src/goalzendo_g00f_g01_bridge/bridge.py",
    "src/goalzendo_g00f_g01_bridge/cli.py",
    "runs/goalzendo/run_g01_after_g00f_bridge.py",
)
_COORDINATOR_PATHS = (
    "src/goalzendo_g01_coordinator/__init__.py",
    "src/goalzendo_g01_coordinator/coordinator.py",
    "runs/goalzendo/run_g01_global_coordinator.py",
)
_GOALZENDO_PATHS = (
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
)
_BRIDGE_PACKAGE_PATHS = _BRIDGE_PATHS[:3]
_COORDINATOR_PACKAGE_PATHS = _COORDINATOR_PATHS[:2]


class _BootstrapError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _strict_json(payload: bytes, label: str) -> dict[str, Any]:
    def reject(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise _BootstrapError(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                _BootstrapError(f"{label} contains non-finite {constant}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise _BootstrapError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise _BootstrapError(f"{label} is not one JSON object")
    return value


def _exact_arguments(argv: list[str]) -> dict[str, str]:
    allowed = {
        "--repo",
        "--execution-uuid",
        "--expected-coordinator-token-sha256",
        "--expected-bridge-source-digest",
        "--expected-coordinator-source-digest",
    }
    if len(argv) != 2 * len(allowed):
        raise _BootstrapError("checkpoint-B bootstrap requires exactly five key/value arguments")
    result: dict[str, str] = {}
    for index in range(0, len(argv), 2):
        key, value = argv[index : index + 2]
        if key not in allowed or key in result or not value or value.startswith("--"):
            raise _BootstrapError("bootstrap arguments contain an unknown, duplicate, or missing value")
        result[key] = value
    if set(result) != allowed:
        raise _BootstrapError("bootstrap arguments omit a required exact key")
    return result


def _require_sha256(value: str, label: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise _BootstrapError(f"{label} must be one lowercase SHA-256 digest")
    return value


def _require_runtime() -> None:
    flags = sys.flags
    if not (
        flags.isolated
        and flags.ignore_environment
        and flags.no_user_site
        and getattr(flags, "safe_path", False)
        and flags.no_site
    ):
        raise _BootstrapError("checkpoint B requires CPython isolated/no-site mode (-I -S)")
    if sys.implementation.name != "cpython" or sys.version_info[:3] != (3, 12, 3):
        raise _BootstrapError("checkpoint B requires CPython 3.12.3 final")
    if sys.platform != "linux":
        raise _BootstrapError("checkpoint B requires Linux")


def _require_no_links(path: Path, label: str) -> None:
    if not path.is_absolute() or ".." in path.parts:
        raise _BootstrapError(f"{label} is not one safe absolute path")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise _BootstrapError(f"{label} contains symlink indirection: {current}")


class _Snapshot:
    """One-read source snapshot with held descriptors and final replay."""

    def __init__(
        self,
        repo: Path,
        repo_descriptor: int,
        payloads: dict[str, bytes],
        descriptors: dict[str, int],
        token_payload: bytes,
        token_descriptor: int,
    ) -> None:
        self.repo = repo
        self.repo_descriptor = repo_descriptor
        self.payloads = MappingProxyType(dict(payloads))
        self._descriptors = descriptors
        self.token_payload = token_payload
        self.token_descriptor = token_descriptor

    def bytes(self, relative: str) -> bytes:
        return self.payloads[relative]

    def verify_held(self) -> None:
        repo_now = _open_directory_chain(self.repo, "frozen source")
        try:
            held_repo = os.fstat(self.repo_descriptor)
            current_repo = os.fstat(repo_now)
            if (held_repo.st_dev, held_repo.st_ino) != (current_repo.st_dev, current_repo.st_ino):
                raise _BootstrapError("frozen source root changed")
        finally:
            os.close(repo_now)
        for relative, descriptor in self._descriptors.items():
            held = os.fstat(descriptor)
            observed = _open_relative(self.repo_descriptor, relative, directory=False)
            try:
                current = os.fstat(observed)
                if (
                    (held.st_dev, held.st_ino) != (current.st_dev, current.st_ino)
                    or not stat.S_ISREG(held.st_mode)
                    or not stat.S_ISREG(current.st_mode)
                    or held.st_nlink != 1
                    or current.st_nlink != 1
                ):
                    raise _BootstrapError(f"frozen source path changed: {relative}")
                if os.pread(descriptor, held.st_size, 0) != self.payloads[relative]:
                    raise _BootstrapError(f"held frozen source bytes changed: {relative}")
            finally:
                os.close(observed)
        token_parent = _open_directory_chain(self.repo.parent.parent, "thin-token canonical parent")
        try:
            observed_token = os.open(
                "g00f-g01-coordinator-input.json",
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=token_parent,
            )
            try:
                held = os.fstat(self.token_descriptor)
                current = os.fstat(observed_token)
                if (
                    (held.st_dev, held.st_ino) != (current.st_dev, current.st_ino)
                    or not stat.S_ISREG(held.st_mode)
                    or not stat.S_ISREG(current.st_mode)
                    or held.st_nlink != 1
                    or current.st_nlink != 1
                    or stat.S_IMODE(held.st_mode) != 0o400
                    or stat.S_IMODE(current.st_mode) != 0o400
                ):
                    raise _BootstrapError("canonical thin-token path changed")
                if os.pread(self.token_descriptor, held.st_size, 0) != self.token_payload:
                    raise _BootstrapError("held thin-token bytes changed")
            finally:
                os.close(observed_token)
        finally:
            os.close(token_parent)

    def close(self) -> None:
        for descriptor in (*self._descriptors.values(), self.token_descriptor, self.repo_descriptor):
            with suppress(OSError):
                os.close(descriptor)


def _open_directory_chain(path: Path, label: str) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path.anchor, flags)
    try:
        for component in path.parts[1:]:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except OSError as error:
        os.close(descriptor)
        raise _BootstrapError(f"{label} cannot be opened component-wise without links") from error


def _open_relative(root_descriptor: int, relative: str, *, directory: bool) -> int:
    parts = Path(relative).parts
    descriptor = os.dup(root_descriptor)
    try:
        for index, component in enumerate(parts):
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            if index < len(parts) - 1 or directory:
                flags |= getattr(os, "O_DIRECTORY", 0)
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except OSError as error:
        os.close(descriptor)
        raise _BootstrapError(f"snapshot relative member cannot be opened: {relative}") from error


def _capture(repo: Path, relatives: list[str], token: Path) -> _Snapshot:
    repo_descriptor = _open_directory_chain(repo, "frozen source")
    payloads: dict[str, bytes] = {}
    descriptors: dict[str, int] = {}
    token_descriptor: int | None = None
    try:
        for relative in sorted(set(relatives)):
            descriptor = _open_relative(repo_descriptor, relative, directory=False)
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                os.close(descriptor)
                raise _BootstrapError(f"snapshot member is not a single-link regular file: {relative}")
            payload = os.pread(descriptor, metadata.st_size, 0)
            if len(payload) != metadata.st_size:
                os.close(descriptor)
                raise _BootstrapError(f"snapshot member short read: {relative}")
            payloads[relative] = payload
            descriptors[relative] = descriptor
        token_descriptor = _open_directory_chain(token.parent, "thin token parent")
        token_leaf = os.open(token.name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=token_descriptor)
        os.close(token_descriptor)
        token_descriptor = token_leaf
        token_metadata = os.fstat(token_descriptor)
        if (
            not stat.S_ISREG(token_metadata.st_mode)
            or token_metadata.st_nlink != 1
            or stat.S_IMODE(token_metadata.st_mode) != 0o400
        ):
            raise _BootstrapError("thin token is not a single-link mode-0400 regular file")
        token_payload = os.pread(token_descriptor, token_metadata.st_size, 0)
    except BaseException:
        for descriptor in descriptors.values():
            os.close(descriptor)
        if token_descriptor is not None:
            os.close(token_descriptor)
        os.close(repo_descriptor)
        raise
    return _Snapshot(repo, repo_descriptor, payloads, descriptors, token_payload, token_descriptor)


def _implementation_fingerprint(snapshot: _Snapshot, files: list[str]) -> str:
    digest = hashlib.sha256()
    digest.update(b"goalzendo-source-v1\0")
    for relative in files:
        package_relative = Path(relative).relative_to("src/goalzendo").as_posix()
        name = package_relative.encode("utf-8")
        payload = snapshot.bytes(relative)
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _bootstrap() -> int:
    _require_runtime()
    arguments = _exact_arguments(sys.argv[1:])
    entrypoint = Path(__file__).absolute()
    _require_no_links(entrypoint, "checkpoint-B entrypoint")
    repo = entrypoint.parents[2]
    supplied_repo = Path(arguments["--repo"])
    if not supplied_repo.is_absolute() or supplied_repo != repo:
        raise _BootstrapError("--repo must be the entrypoint-derived frozen-source root")
    try:
        relative = repo.relative_to("/workspace/status-goalzendo/g00f-executions")
    except ValueError as error:
        raise _BootstrapError("entrypoint is outside the canonical checkpoint-A root") from error
    if len(relative.parts) != 2 or relative.parts[1] != "frozen-source":
        raise _BootstrapError("entrypoint is not inside one canonical frozen-source UUID root")
    expected_bridge = _require_sha256(arguments["--expected-bridge-source-digest"], "bridge source digest")
    expected_coordinator = _require_sha256(
        arguments["--expected-coordinator-source-digest"], "coordinator source digest"
    )
    goalzendo_files = sorted(
        path.relative_to(repo).as_posix()
        for path in (repo / "src/goalzendo").rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and (path.suffix == ".py" or path.name == "py.typed")
    )
    if tuple(goalzendo_files) != _GOALZENDO_PATHS:
        raise _BootstrapError("GoalZendo source inventory changed")
    project_python = sorted(
        path.relative_to(repo).as_posix()
        for package in ("goalzendo_g00f_g01_bridge", "goalzendo_g01_coordinator")
        for path in (repo / "src" / package).rglob("*.py")
    )
    expected_project_python = sorted((*_BRIDGE_PACKAGE_PATHS, *_COORDINATOR_PACKAGE_PATHS))
    if project_python != expected_project_python:
        raise _BootstrapError("bridge/coordinator package source inventory changed")
    extra = [
        *_BRIDGE_PATHS,
        *_COORDINATOR_PATHS,
        "configs/goalzendo/g01_known_law.yaml",
        "configs/goalzendo/base.yaml",
        "docs/goalzendo/protocols/g01-known-law.md",
    ]
    expected_token_sha = _require_sha256(
        arguments["--expected-coordinator-token-sha256"], "coordinator token SHA-256"
    )
    token_path = repo.parent.parent / "g00f-g01-coordinator-input.json"
    snapshot = _capture(repo, [*goalzendo_files, *project_python, *extra], token_path)
    try:
        if _sha(snapshot.token_payload) != expected_token_sha:
            raise _BootstrapError("thin token differs from its externally registered SHA-256")
        bridge_files = {relative: _sha(snapshot.bytes(relative)) for relative in _BRIDGE_PATHS}
        coordinator_files = {relative: _sha(snapshot.bytes(relative)) for relative in _COORDINATOR_PATHS}
        if _sha(_canonical(bridge_files)) != expected_bridge:
            raise _BootstrapError("checkpoint-A bridge differs from its external source digest")
        if _sha(_canonical(coordinator_files)) != expected_coordinator:
            raise _BootstrapError("checkpoint-B source differs from its external source digest")
        if _implementation_fingerprint(snapshot, goalzendo_files) != _G01_SOURCE_FINGERPRINT:
            raise _BootstrapError("GoalZendo implementation fingerprint changed")
        fixed = {
            "configs/goalzendo/g01_known_law.yaml": _G01_CONFIG_SHA256,
            "configs/goalzendo/base.yaml": _G01_BASE_CONFIG_SHA256,
            "docs/goalzendo/protocols/g01-known-law.md": _G01_PROTOCOL_SHA256,
            "src/goalzendo/runner.py": _G01_RUNNER_SHA256,
        }
        if any(_sha(snapshot.bytes(relative)) != digest for relative, digest in fixed.items()):
            raise _BootstrapError("exact G01 config/protocol/runner bytes changed")
        snapshot.verify_held()
        # Checkpoint A's route is evidence provenance, never a checkpoint-B
        # interpreter, dependency, GPU, image, or provision selection.  This
        # source checkpoint deliberately contains no way to mint execution
        # authority.  A separately audited, externally registered B overlay
        # transaction must replace this refusal in a later source revision.
        raise _BootstrapError(_RUNTIME_REFUSAL)
    finally:
        snapshot.close()


if __name__ == "__main__":
    try:
        raise SystemExit(_bootstrap())
    except (OSError, _BootstrapError) as error:
        print(f"run_g01_global_coordinator: bootstrap error: {error}", file=sys.stderr)
        raise SystemExit(2) from None
