#!/usr/bin/env python3
"""Build the generic, nonauthorizing checkpoint-B source capsule.

The canonical build is deliberately held.  Review tests may call :func:`build`
for a noncanonical absent output directory after supplying the externally
calculated hash of this builder.  The capsule contains source and documentary
inputs only.  It makes, binds, and authorizes no execution UUID, hardware
route, runtime overlay, deadline, provision receipt, or launch authority.
"""

from __future__ import annotations

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
from contextlib import suppress
from pathlib import Path
from types import MappingProxyType
from typing import Any

SOURCE_DATE_EPOCH = 1_786_492_800
DEFAULT_OUTPUT = "reproducibility/goalzendo/g01-b-source-capsule-20260812"
ARCHIVE_NAME = "g01-b-source-capsule.tar.gz"
MANIFEST_NAME = "g01-b-source-capsule-manifest.json"
FREEZE_NAME = "g01-b-source-capsule-freeze.json"
MANIFEST_SCHEMA = "goalzendo.g01_b_source_capsule_manifest"
FREEZE_SCHEMA = "goalzendo.g01_b_source_capsule_freeze"
STUDY_ID = "g01_checkpoint_b_source_capsule"
GOALZENDO_FINGERPRINT = "1a8146377b9a9620690025671614edb2dd20d214f4e195da3cf3528809b2c694"
BRIDGE_SOURCE_DIGEST = "e0f3df053557cdf06c028a08f847db33b55dac3a7388a52e0fae0d94b07d284f"
COORDINATOR_SOURCE_DIGEST = "77d9ba4928fa29cc42a055201660304f2059d0ca6358371cc14618407bdc714b"
CANONICAL_IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
CANONICAL_PYTHON = "/workspace/.venvs/goalzendo/bin/python"
CANONICAL_CHECKPOINT_A_ROOT = Path("/workspace/status-goalzendo/g00f-executions")
FORBIDDEN_OUTPUT_ROOTS = (
    CANONICAL_CHECKPOINT_A_ROOT,
    Path("/workspace/inputs-goalzendo/g01-executions"),
    Path("/workspace/status-goalzendo/g01-preexecution"),
    Path("/workspace/status-goalzendo/g01-executions"),
    Path("/workspace/artifacts-goalzendo/g01-known-law"),
)
_TEST_ONLY_ALLOW_RUNTIME = False

# These 31 already-reviewed files are byte-pinned.  The three additive files
# below are captured and sealed by the generated manifest/freeze instead of
# embedding impossible self-referential hashes in this builder.
LOCKED_PAYLOAD_SHA256: Mapping[str, str] = MappingProxyType(
    {
        "configs/goalzendo/base.yaml": "34cb42eb2bf7ea275247ecf8baa0e5556565c2c86c30da0aec8e2a1a630e04fb",
        "configs/goalzendo/g01_known_law.yaml": "ee6d53556189a6b6e25cfbfb204325017a058c63a605653df4c175463e00ce18",
        "docs/goalzendo/protocols/g01-global-coordinator-checkpoint-b.md": "93acc6262758dd5478e342e2f89fd5cb1ab3b61a03bed34b1a6d5083cacaaeba",
        "docs/goalzendo/protocols/g01-known-law.md": "428243c3da271bc2d79e16c0fde20d47076eb3837c6740ab733e278cddcdbce9",
        "pyproject.toml": "aa6e5117128140343290abbed9bf2a3a0203f2662bd80f031748f2b371392092",
        "runs/goalzendo/run_g01_after_g00f_bridge.py": "bc7703c48d47cc72f21e2f72ee3a4864ff5dd78759c5f692c59ce46ba03ee10b",
        "runs/goalzendo/run_g01_global_coordinator.py": "69f5ec520d8c681d5e9cab1dd9687d7fd39c948f08ff397331bd5ce74f824a06",
        "src/goalzendo/__init__.py": "a6dd83fdd29b0f476e91cefc65d02794a161b7cbe11eeef91622736f5a36e977",
        "src/goalzendo/analysis.py": "07e21725f76a4920e26eef8ccc6c26a9e6f3dca06292bdd5fbc7af203a93fbe1",
        "src/goalzendo/artifacts.py": "f47c60ce408ffd5f412a97fcc2befb123781831acd1ca2776df769d86298638b",
        "src/goalzendo/cli.py": "1816a5c7a6fef2e1949ab08dbef423c3c890af665aaef592afe3eddf1b98348a",
        "src/goalzendo/config.py": "88e255158d2cf7309732954754a580384d8dabea533144e1da536b24c03a3a36",
        "src/goalzendo/evaluation.py": "8a6d029c14a8a879b499f4b119202865a3ade3c2cb05a0bacad19913fbd37478",
        "src/goalzendo/experiment.py": "2b567d897f2ceb09682c390febc43b70ab3fac9b58c3d8dcec73cbd7fc24bf9e",
        "src/goalzendo/generation.py": "f8c91ee798b6c76453d8ad7cdd2125000231484d579c049b3690ec43912fab4a",
        "src/goalzendo/interventions.py": "e2402db054484c4a87a20ccbd22ca3e5f94baf04226fa91f61b8fbc44751feba",
        "src/goalzendo/metrics.py": "663c8d809de3a8fca883c2324a680eaa3974058941c15e0f3f1d024776365f9b",
        "src/goalzendo/modeling.py": "3cfde1cb82640f39b9a5b1841bc8b6fa65b31f6c8e6c09ec6e41e5ea3cd7ee62",
        "src/goalzendo/plotting.py": "736509fa2045cb860decf495957cb10636ba2e01bacfd47a5ec484c6d7edbdaa",
        "src/goalzendo/py.typed": "01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b",
        "src/goalzendo/rendering.py": "073ad8b446852fceb76a4f651df9085d19ae8d9f1d98378aee93a9f91ae37f25",
        "src/goalzendo/reproduction.py": "307f57f526cd9a5741a98c41971d44249c1393ac971e49e7e03b4fbc343ed660",
        "src/goalzendo/rules.py": "fc520a1df77d3d3dbd51cd5ca03be4f532a7d2adbe475c68d46bd52ce1a4a7a4",
        "src/goalzendo/runner.py": "46b55ad4bdd08073e5f89ae101e0862372817b74b590f07ddb8f8c4331e5b9e1",
        "src/goalzendo/schema.py": "f94a9f7213f853813f9690d3320ca74581704d5212aa3dfc8230e1d5d6702ab7",
        "src/goalzendo/training.py": "4b794824a812d16c93d8ea9b33d0e6eeff2359a86dc6ddd48e90bcfea6efd8f1",
        "src/goalzendo_g00f_g01_bridge/__init__.py": "94d447686b62177ed3f50742424c9a98a6f1c95d921422cf7bf3241689e24503",
        "src/goalzendo_g00f_g01_bridge/bridge.py": "4ab7c684109b10c2d6ec1469a281fa342e0dac14c5d141a9d97f0cd8e4f575d2",
        "src/goalzendo_g00f_g01_bridge/cli.py": "350cb69a2923b49aeb4957406b4d86319a2c75f7c7a32e14e53d479b04319801",
        "src/goalzendo_g01_coordinator/__init__.py": "85a9454fc21c2dc8c70e6ebf9eb605aa0208e6c13e73547ec0f4abdd08abad35",
        "src/goalzendo_g01_coordinator/coordinator.py": "a3390dad47ea1fd2aa7b1ca22e4475b9a510c47fc85f9b296ed925303663b302",
    }
)

BUILDER_PATH = "runs/goalzendo/build_g01_b_source_capsule.py"
STAGER_PATH = "runs/goalzendo/g01_b_source_capsule_stage.py"
CAPSULE_PROTOCOL_PATH = "docs/goalzendo/protocols/g01-b-source-capsule.md"
ADDITIVE_PATHS = (BUILDER_PATH, STAGER_PATH, CAPSULE_PROTOCOL_PATH)
PAYLOAD_PATHS = tuple(sorted((*LOCKED_PAYLOAD_SHA256, *ADDITIVE_PATHS)))

AUTHORIZATION: Mapping[str, bool] = MappingProxyType(
    {
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
)


class BuildError(RuntimeError):
    """The source capsule cannot be authenticated or constructed."""


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
        raise BuildError(f"{label} must be one lowercase SHA-256 string")
    text = value
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise BuildError(f"{label} must be one lowercase SHA-256")
    return text


def _require_runtime() -> None:
    if _TEST_ONLY_ALLOW_RUNTIME:
        if "PYTEST_CURRENT_TEST" not in os.environ:
            raise BuildError("test-only runtime bypass is forbidden outside pytest")
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
        raise BuildError("builder requires exact CPython 3.12.3 Linux x86_64 -I -S and zlib 1.3")
    try:
        task_ids = os.listdir("/proc/self/task")
    except OSError as error:
        raise BuildError("builder requires an observable single-task Linux process") from error
    if len(task_ids) != 1 or not task_ids[0].isdigit():
        raise BuildError("builder requires one Linux task before any temporary umask change")


def _safe_relative(value: str) -> tuple[str, ...]:
    path = Path(value)
    if not value or path.is_absolute() or "." in path.parts or ".." in path.parts or path.as_posix() != value:
        raise BuildError(f"unsafe repository-relative path: {value!r}")
    return path.parts


def _lexical_absolute(value: str | Path, label: str) -> Path:
    """Normalize dot components without resolving links or probing the filesystem."""

    try:
        raw = os.fspath(value)
    except (TypeError, ValueError) as error:
        raise BuildError(f"{label} is not one lexical path") from error
    if type(raw) is not str or "\0" in raw:
        raise BuildError(f"{label} must be a NUL-free text path")
    absolute = os.path.abspath(raw)
    if not absolute.startswith("/"):
        raise BuildError(f"{label} is not one POSIX absolute path")
    # POSIX kernels treat multiple leading slashes as the root for these local
    # paths, while pathlib preserves a special '//' anchor.  Collapse the run
    # before namespace comparisons so //workspace cannot bypass a guard.
    lexical = Path("/" + absolute.lstrip("/"))
    if not lexical.is_absolute() or ".." in lexical.parts or "." in lexical.parts:
        raise BuildError(f"{label} is not one normalized absolute path")
    return lexical


def _is_within(candidate: Path, root: Path) -> bool:
    return candidate == root or root in candidate.parents


def _reject_checkpoint_a_path(path: Path, label: str) -> None:
    if _is_within(path, CANONICAL_CHECKPOINT_A_ROOT):
        raise BuildError(f"{label} is inside the forbidden canonical checkpoint-A namespace")


def _reject_output_namespace(path: Path) -> None:
    if any(_is_within(path, root) for root in FORBIDDEN_OUTPUT_ROOTS):
        raise BuildError("output is inside a forbidden canonical scientific transaction namespace")


def _open_directory_chain(path: Path, label: str) -> int:
    absolute = path.absolute()
    if not absolute.is_absolute() or ".." in absolute.parts:
        raise BuildError(f"{label} is not a safe absolute path")
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


def _open_relative(root: int, relative: str, label: str) -> int:
    parts = _safe_relative(relative)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    current = os.dup(root)
    try:
        for part in parts[:-1]:
            child = os.open(part, directory_flags, dir_fd=current)
            os.close(current)
            current = child
        descriptor = os.open(parts[-1], os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=current)
        os.close(current)
        return descriptor
    except OSError as error:
        os.close(current)
        raise BuildError(f"{label} cannot be opened without link traversal") from error


def _require_descriptor_matches_path(descriptor: int, path: Path, label: str) -> None:
    held = os.fstat(descriptor)
    current = _open_directory_chain(path, f"{label} replay")
    try:
        observed = os.fstat(current)
        if not stat.S_ISDIR(observed.st_mode) or (held.st_dev, held.st_ino) != (
            observed.st_dev,
            observed.st_ino,
        ):
            raise BuildError(f"{label} path no longer names the held directory")
    finally:
        os.close(current)


def _same_directory_identity(left: int, right: int) -> bool:
    left_metadata = os.fstat(left)
    right_metadata = os.fstat(right)
    return (left_metadata.st_dev, left_metadata.st_ino) == (
        right_metadata.st_dev,
        right_metadata.st_ino,
    )


def _mkdir_exact_at(parent: int, name: str, mode: int, label: str) -> None:
    """Create one directory with an exact mode, independently of ambient umask."""

    previous_umask = os.umask(0)
    changed_umask: int | None = None
    try:
        os.mkdir(name, mode, dir_fd=parent)
    except FileExistsError as error:
        raise BuildError(f"{label} must be absent") from error
    finally:
        changed_umask = os.umask(previous_umask)
    if changed_umask != 0:
        raise BuildError("process umask changed concurrently during exact directory creation")
    os.fsync(parent)


class _Snapshot:
    def __init__(self, root: Path, root_fd: int, descriptors: dict[str, int], payloads: dict[str, bytes]):
        self.root = root
        self.root_fd = root_fd
        self.descriptors = descriptors
        self.payloads = payloads

    @classmethod
    def capture(cls, root: Path) -> _Snapshot:
        root_fd = _open_directory_chain(root, "repository root")
        descriptors: dict[str, int] = {}
        payloads: dict[str, bytes] = {}
        try:
            for relative in PAYLOAD_PATHS:
                descriptor = _open_relative(root_fd, relative, f"capsule source {relative}")
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                    os.close(descriptor)
                    raise BuildError(f"capsule source is not a single-link regular file: {relative}")
                payload = os.pread(descriptor, metadata.st_size, 0)
                if len(payload) != metadata.st_size:
                    os.close(descriptor)
                    raise BuildError(f"capsule source short read: {relative}")
                expected = LOCKED_PAYLOAD_SHA256.get(relative)
                if expected is not None and sha256_bytes(payload) != expected:
                    os.close(descriptor)
                    raise BuildError(f"locked capsule source changed: {relative}")
                descriptors[relative] = descriptor
                payloads[relative] = payload
            return cls(root, root_fd, descriptors, payloads)
        except BaseException:
            for descriptor in descriptors.values():
                with suppress(OSError):
                    os.close(descriptor)
            os.close(root_fd)
            raise

    def verify(self) -> None:
        _require_descriptor_matches_path(self.root_fd, self.root, "repository root")
        for relative, descriptor in self.descriptors.items():
            held = os.fstat(descriptor)
            current_fd = _open_relative(self.root_fd, relative, f"capsule source replay {relative}")
            try:
                current = os.fstat(current_fd)
                if (
                    not stat.S_ISREG(held.st_mode)
                    or not stat.S_ISREG(current.st_mode)
                    or held.st_nlink != 1
                    or current.st_nlink != 1
                    or (held.st_dev, held.st_ino) != (current.st_dev, current.st_ino)
                    or os.pread(descriptor, held.st_size, 0) != self.payloads[relative]
                ):
                    raise BuildError(f"capsule source changed after snapshot: {relative}")
            finally:
                os.close(current_fd)

    def close(self) -> None:
        for descriptor in self.descriptors.values():
            with suppress(OSError):
                os.close(descriptor)
        with suppress(OSError):
            os.close(self.root_fd)


def _repo(start: Path) -> Path:
    # Validation happens by component-wise O_NOFOLLOW opens in
    # _Snapshot.capture.  Never resolve or probe caller paths to discover a
    # parent repository.
    return _lexical_absolute(start, "repository root")


def _source_digest(paths: Sequence[str], payloads: Mapping[str, bytes]) -> str:
    return digest({relative: sha256_bytes(payloads[relative]) for relative in paths})


def _goalzendo_fingerprint(payloads: Mapping[str, bytes]) -> str:
    paths = sorted(relative for relative in payloads if relative.startswith("src/goalzendo/"))
    accumulator = hashlib.sha256()
    accumulator.update(b"goalzendo-source-v1\0")
    for relative in paths:
        package_relative = Path(relative).relative_to("src/goalzendo").as_posix().encode("utf-8")
        payload = payloads[relative]
        accumulator.update(len(package_relative).to_bytes(8, "big"))
        accumulator.update(package_relative)
        accumulator.update(len(payload).to_bytes(8, "big"))
        accumulator.update(payload)
    return accumulator.hexdigest()


def _directories() -> list[str]:
    values: set[str] = set()
    for relative in PAYLOAD_PATHS:
        parent = Path(relative).parent
        while parent.as_posix() != ".":
            values.add(parent.as_posix())
            parent = parent.parent
    return sorted(values)


def _manifest(payloads: Mapping[str, bytes]) -> dict[str, Any]:
    if _goalzendo_fingerprint(payloads) != GOALZENDO_FINGERPRINT:
        raise BuildError("GoalZendo implementation fingerprint changed")
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
    if _source_digest(bridge_paths, payloads) != BRIDGE_SOURCE_DIGEST:
        raise BuildError("accepted checkpoint-A bridge source digest changed")
    if _source_digest(coordinator_paths, payloads) != COORDINATOR_SOURCE_DIGEST:
        raise BuildError("accepted checkpoint-B coordinator source digest changed")
    members = [
        {
            "path": relative,
            "type": "file",
            "mode": 0o444,
            "bytes": len(payloads[relative]),
            "sha256": sha256_bytes(payloads[relative]),
        }
        for relative in PAYLOAD_PATHS
    ]
    body = {
        "schema": MANIFEST_SCHEMA,
        "schema_version": 1,
        "study_id": STUDY_ID,
        "source_date_epoch": SOURCE_DATE_EPOCH,
        "members": members,
        "directories": [
            {"path": relative, "type": "directory", "mode": 0o555} for relative in _directories()
        ],
        "identity": {
            "goalzendo_implementation_fingerprint": GOALZENDO_FINGERPRINT,
            "bridge_source_digest": BRIDGE_SOURCE_DIGEST,
            "coordinator_source_digest": COORDINATOR_SOURCE_DIGEST,
            "member_count": len(PAYLOAD_PATHS),
        },
        "authorization": dict(AUTHORIZATION),
        "trust_limitations": {
            "claim": "source_bytes_only",
            "rootfs_authenticated": False,
            "runtime_authenticated": False,
            "python_stdlib_authenticated": False,
            "native_dependency_closure_authenticated": False,
            "external_launcher_authenticated": False,
        },
    }
    return {**body, "manifest_digest": digest(body)}


def _archive_bytes(payloads: Mapping[str, bytes], manifest: Mapping[str, Any]) -> bytes:
    output = io.BytesIO()
    with (
        gzip.GzipFile(filename="", mode="wb", fileobj=output, compresslevel=9, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT) as archive,
    ):
        for row in manifest["members"]:
            relative = str(row["path"])
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


def _exclusive_write(parent: int, name: str, payload: bytes) -> int:
    writer = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=parent,
    )
    with os.fdopen(writer, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fchmod(handle.fileno(), 0o644)
        os.fsync(handle.fileno())
        written = os.fstat(handle.fileno())
    reader = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent)
    observed = os.fstat(reader)
    if (
        not stat.S_ISREG(observed.st_mode)
        or observed.st_nlink != 1
        or stat.S_IMODE(observed.st_mode) != 0o644
        or (written.st_dev, written.st_ino) != (observed.st_dev, observed.st_ino)
        or observed.st_size != len(payload)
        or os.pread(reader, observed.st_size, 0) != payload
    ):
        os.close(reader)
        raise BuildError(f"written capsule artifact changed: {name}")
    os.fsync(parent)
    return reader


def _verify_output_artifacts(
    target: int,
    descriptors: Mapping[str, int],
    payloads: Mapping[str, bytes],
) -> None:
    if set(os.listdir(target)) != set(payloads):
        raise BuildError("capsule output directory exact inventory changed")
    for name, payload in payloads.items():
        held = os.fstat(descriptors[name])
        current = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=target)
        try:
            observed = os.fstat(current)
            current_payload = os.pread(current, observed.st_size, 0)
            if (
                not stat.S_ISREG(held.st_mode)
                or not stat.S_ISREG(observed.st_mode)
                or held.st_nlink != 1
                or observed.st_nlink != 1
                or stat.S_IMODE(held.st_mode) != 0o644
                or stat.S_IMODE(observed.st_mode) != 0o644
                or (held.st_dev, held.st_ino) != (observed.st_dev, observed.st_ino)
                or held.st_size != len(payload)
                or observed.st_size != len(payload)
                or os.pread(descriptors[name], held.st_size, 0) != payload
                or current_payload != payload
                or sha256_bytes(current_payload) != sha256_bytes(payload)
            ):
                raise BuildError(f"capsule output artifact replay failed: {name}")
        finally:
            os.close(current)


def build(
    repo: str | Path,
    output: str | Path,
    *,
    expected_builder_sha256: str,
) -> dict[str, Any]:
    """Create one deterministic review artifact set in an absent noncanonical directory."""

    _require_runtime()
    raw_repo = _lexical_absolute(repo, "repository root")
    target = _lexical_absolute(output, "capsule output")
    running = _lexical_absolute(Path(__file__), "running builder")
    _reject_checkpoint_a_path(raw_repo, "repository root")
    _reject_checkpoint_a_path(running, "running builder")
    _reject_output_namespace(target)
    resolved = _repo(raw_repo)
    canonical_target = _lexical_absolute(resolved / DEFAULT_OUTPUT, "canonical capsule output")
    if target == canonical_target:
        raise BuildError("CANONICAL_G01_B_SOURCE_CAPSULE_BUILD_HELD")
    if not target.name or target.name in {".", ".."}:
        raise BuildError("output must have one absent leaf below an existing directory")
    canonical_parent_fd = _open_directory_chain(
        canonical_target.parent,
        "held canonical capsule-output parent",
    )
    try:
        snapshot = _Snapshot.capture(resolved)
        try:
            builder_sha = sha256_bytes(snapshot.payloads[BUILDER_PATH])
            if builder_sha != require_sha256(expected_builder_sha256, "external builder SHA-256"):
                raise BuildError("builder differs from its externally supplied SHA-256")
            running_parent = _open_directory_chain(running.parent, "running builder parent")
            try:
                running_descriptor = os.open(
                    running.name,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=running_parent,
                )
            except OSError as error:
                os.close(running_parent)
                raise BuildError("running builder cannot be opened without links") from error
            held_metadata = os.fstat(snapshot.descriptors[BUILDER_PATH])
            try:
                running_metadata = os.fstat(running_descriptor)
                if (
                    not stat.S_ISREG(running_metadata.st_mode)
                    or running_metadata.st_nlink != 1
                    or (running_metadata.st_dev, running_metadata.st_ino)
                    != (held_metadata.st_dev, held_metadata.st_ino)
                    or os.pread(running_descriptor, running_metadata.st_size, 0)
                    != snapshot.payloads[BUILDER_PATH]
                    or os.pread(
                        snapshot.descriptors[BUILDER_PATH],
                        held_metadata.st_size,
                        0,
                    )
                    != snapshot.payloads[BUILDER_PATH]
                ):
                    raise BuildError("running builder is not the snapshotted canonical builder")
                _require_descriptor_matches_path(
                    running_parent,
                    running.parent,
                    "running builder parent",
                )
            finally:
                os.close(running_descriptor)
                os.close(running_parent)
            manifest = _manifest(snapshot.payloads)
            manifest_payload = json_bytes(manifest)
            archive_payload = _archive_bytes(snapshot.payloads, manifest)
            stager_sha = sha256_bytes(snapshot.payloads[STAGER_PATH])
            body = {
                "schema": FREEZE_SCHEMA,
                "schema_version": 1,
                "study_id": STUDY_ID,
                "source_date_epoch": SOURCE_DATE_EPOCH,
                "generic_execution_uuid": None,
                "outcomes_seen": False,
                "bundle": {
                    "archive_name": ARCHIVE_NAME,
                    "archive_sha256": sha256_bytes(archive_payload),
                    "manifest_name": MANIFEST_NAME,
                    "manifest_sha256": sha256_bytes(manifest_payload),
                    "manifest_digest": manifest["manifest_digest"],
                    "file_member_count": len(PAYLOAD_PATHS),
                    "directory_count": len(manifest["directories"]),
                },
                "controller_files": {
                    "builder": {"path": BUILDER_PATH, "sha256": builder_sha},
                    "stager_and_fresh_verifier": {"path": STAGER_PATH, "sha256": stager_sha},
                },
                "identity": manifest["identity"],
                "authorization": dict(AUTHORIZATION),
                "transaction": {
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
                },
                "required_build_runtime_claim": {
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
                },
                "trust_limitations": manifest["trust_limitations"],
            }
            freeze = {**body, "freeze_digest": digest(body)}
            freeze_payload = json_bytes(freeze)
            snapshot.verify()
            parent_fd = _open_directory_chain(target.parent, "capsule output parent")
            artifact_fds: dict[str, int] = {}
            try:
                _require_descriptor_matches_path(
                    canonical_parent_fd,
                    canonical_target.parent,
                    "held canonical capsule-output parent",
                )
                if target.name == canonical_target.name and _same_directory_identity(
                    parent_fd,
                    canonical_parent_fd,
                ):
                    raise BuildError("CANONICAL_G01_B_SOURCE_CAPSULE_BUILD_HELD")
                snapshot.verify()
                _require_descriptor_matches_path(parent_fd, target.parent, "capsule output parent")
                _mkdir_exact_at(parent_fd, target.name, 0o755, "capsule output directory")
                target_fd = os.open(
                    target.name,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=parent_fd,
                )
                try:
                    os.fchmod(target_fd, 0o755)
                    os.fsync(target_fd)
                    artifact_payloads = {
                        MANIFEST_NAME: manifest_payload,
                        ARCHIVE_NAME: archive_payload,
                        FREEZE_NAME: freeze_payload,
                    }
                    for name, payload in artifact_payloads.items():
                        artifact_fds[name] = _exclusive_write(target_fd, name, payload)
                    _verify_output_artifacts(target_fd, artifact_fds, artifact_payloads)
                    snapshot.verify()
                    _require_descriptor_matches_path(parent_fd, target.parent, "capsule output parent")
                    _require_descriptor_matches_path(
                        canonical_parent_fd,
                        canonical_target.parent,
                        "held canonical capsule-output parent",
                    )
                    _require_descriptor_matches_path(target_fd, target, "capsule output directory")
                    if stat.S_IMODE(os.fstat(target_fd).st_mode) != 0o755:
                        raise BuildError("capsule output directory mode changed")
                    _verify_output_artifacts(target_fd, artifact_fds, artifact_payloads)
                finally:
                    for descriptor in artifact_fds.values():
                        with suppress(OSError):
                            os.close(descriptor)
                    os.close(target_fd)
            finally:
                os.close(parent_fd)
        finally:
            snapshot.close()
    finally:
        os.close(canonical_parent_fd)
    return {
        "output": str(target),
        "archive_sha256": sha256_bytes(archive_payload),
        "manifest_sha256": sha256_bytes(manifest_payload),
        "manifest_digest": manifest["manifest_digest"],
        "freeze_sha256": sha256_bytes(freeze_payload),
        "freeze_digest": freeze["freeze_digest"],
        "file_member_count": len(PAYLOAD_PATHS),
        "g01_launch_authorized": False,
        "runtime_overlay_frozen": False,
        "accepted_refusal_revision_contained": True,
        "supported_launch_entrypoint_activated": False,
    }


def main() -> int:
    # No CLI spelling can build the canonical artifact during this source-only
    # milestone.  Independent review invokes build() with an absent temporary
    # directory and a separately calculated builder digest.
    print("build_g01_b_source_capsule: error: CANONICAL_G01_B_SOURCE_CAPSULE_BUILD_HELD", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
