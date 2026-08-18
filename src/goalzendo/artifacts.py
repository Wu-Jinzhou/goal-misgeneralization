"""Reproducible, resume-safe artifacts for GoalZendo experiments.

This module is intentionally independent of ForkWorld's artifact contract.  A
GoalZendo run is identified by its scientific configuration, its focal seed,
and the byte-level implementation under ``src/goalzendo``.  Storage location,
restart policy, and other seeds launched at the same time are not scientific
identity inputs.

The metadata collector deliberately never inspects environment variables.
Package versions, accelerator properties, and a minimal Git fingerprint are
collected through explicit APIs, and metadata hooks reject secret-like keys.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from .config import canonical_config, validate_config

ARTIFACT_SCHEMA_VERSION = 2
SOURCE_FINGERPRINT_SCHEMA_VERSION = 1

_NON_IDENTITY_RUN_FIELDS = frozenset({"output_root", "resume", "seeds"})
_SOURCE_SUFFIXES = frozenset({".py"})
_SOURCE_NAMES = frozenset({"py.typed"})
_MANIFEST_KINDS = frozenset({"dataset", "model", "tokenizer"})
_COMPLETION_ATTESTED_FILES = (
    "identity.json",
    "resolved_config.yaml",
    "implementation.json",
    "environment.json",
    "manifests/dataset.json",
    "manifests/model.json",
    "manifests/tokenizer.json",
    "metrics.jsonl",
    "predictions.jsonl",
    "summary.json",
    "status.json",
)
_SECRET_KEYS = frozenset(
    {
        "api_key",
        "access_key",
        "private_key",
        "password",
        "passwd",
        "secret",
        "token",
        "auth_token",
        "access_token",
        "refresh_token",
        "bearer_token",
        "hf_token",
        "authorization",
        "cookie",
        "credential",
        "credentials",
    }
)
_SECRET_SUFFIXES = (
    "_api_key",
    "_access_key",
    "_private_key",
    "_password",
    "_passwd",
    "_secret",
    "_auth_token",
    "_access_token",
    "_refresh_token",
    "_bearer_token",
    "_hf_token",
)
class ArtifactError(RuntimeError):
    """Base class for artifact consistency failures."""


class ArtifactConflictError(ArtifactError):
    """Raised when an existing artifact disagrees with the requested run."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    """Convert common scientific values to strict JSON-compatible values."""

    if is_dataclass(value):
        return json_safe(asdict(value))  # type: ignore[arg-type]
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((json_safe(item) for item in value), key=lambda item: repr(item))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return json_safe(value.item())
    return value


def stable_hash(value: Any, length: int = 16) -> str:
    """Return a stable SHA-256 prefix for JSON-compatible content."""

    if not 1 <= int(length) <= 64:
        raise ValueError("length must lie in [1, 64]")
    raw = json.dumps(
        json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[: int(length)]


def atomic_text(path: str | Path, text: str) -> None:
    """Atomically replace a UTF-8 text file in its destination directory."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        # Persist the directory entry where the platform supports directory fsync.
        try:
            directory = os.open(target.parent, os.O_RDONLY)
        except OSError:
            directory = None
        if directory is not None:
            try:
                os.fsync(directory)
            except OSError:
                pass
            finally:
                os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: str | Path, value: Any) -> None:
    atomic_text(
        path,
        json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ArtifactError(f"expected a JSON object in {path}")
    return value


def _file_attestation(path: Path, *, count_jsonl: bool = False) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    rows = 0
    last_byte = b""
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
            rows += chunk.count(b"\n") if count_jsonl else 0
            last_byte = chunk[-1:]
    if count_jsonl and size and last_byte != b"\n":
        raise ArtifactConflictError(f"completed JSONL stream lacks a final newline: {path}")
    result: dict[str, Any] = {"sha256": digest.hexdigest(), "bytes": size}
    if count_jsonl:
        result["rows"] = rows
    return result


def verify_completion_attestation(path: str | Path) -> dict[str, Any]:
    """Verify the byte-level completion seal for artifact-schema-v2 runs."""

    run_path = Path(path)
    target = run_path / "completion.json"
    if not target.is_file():
        raise ArtifactConflictError(f"completed schema-v2 run lacks completion.json: {run_path}")
    payload = read_json(target)
    body = {key: value for key, value in payload.items() if key != "completion_digest"}
    if payload.get("schema") != "goalzendo.run_completion" or payload.get("schema_version") != 1:
        raise ArtifactConflictError(f"unrecognized completion attestation: {run_path}")
    if payload.get("completion_digest") != stable_hash(body, 64):
        raise ArtifactConflictError(f"completion attestation digest mismatch: {run_path}")
    if payload.get("run_id") != run_path.name:
        raise ArtifactConflictError(f"completion attestation run ID mismatch: {run_path}")
    files = payload.get("files")
    if not isinstance(files, Mapping) or set(files) != set(_COMPLETION_ATTESTED_FILES):
        raise ArtifactConflictError(f"completion attestation file set mismatch: {run_path}")
    for relative in _COMPLETION_ATTESTED_FILES:
        file_path = run_path / relative
        if not file_path.is_file():
            raise ArtifactConflictError(f"completion-attested file is missing: {file_path}")
        observed = _file_attestation(
            file_path,
            count_jsonl=relative in {"metrics.jsonl", "predictions.jsonl"},
        )
        if files.get(relative) != observed:
            raise ArtifactConflictError(f"completion-attested file changed: {file_path}")
    return payload


def _identity_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Remove operational fields that cannot change a seed-level experiment."""

    identity = copy.deepcopy(dict(config))
    run = dict(identity.get("run", {}))
    for field in _NON_IDENTITY_RUN_FIELDS:
        run.pop(field, None)
    identity["run"] = run
    return canonical_config(identity)


def implementation_provenance(repo: str | Path | None = None) -> dict[str, Any]:
    """Fingerprint every importable source file under ``src/goalzendo``."""

    installed = Path(__file__).resolve().parent
    candidate = Path(repo).resolve() / "src" / "goalzendo" if repo is not None else None
    source_root = candidate if candidate is not None and candidate.is_dir() else installed
    files = sorted(
        (
            path
            for path in source_root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and (path.suffix in _SOURCE_SUFFIXES or path.name in _SOURCE_NAMES)
        ),
        key=lambda path: path.relative_to(source_root).as_posix(),
    )
    if not files:
        raise ArtifactError(f"no GoalZendo source files found below {source_root}")

    digest = hashlib.sha256()
    digest.update(
        f"goalzendo-source-v{SOURCE_FINGERPRINT_SCHEMA_VERSION}\0".encode()
    )
    relative_files: list[str] = []
    for path in files:
        relative = path.relative_to(source_root).as_posix()
        relative_files.append(relative)
        relative_bytes = relative.encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative_bytes).to_bytes(8, "big"))
        digest.update(relative_bytes)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    result = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "source_fingerprint_schema_version": SOURCE_FINGERPRINT_SCHEMA_VERSION,
        "implementation_fingerprint": digest.hexdigest(),
        "source_file_count": len(relative_files),
        "source_files": relative_files,
    }
    return dict(result)


def _git_metadata(repo: Path) -> dict[str, Any]:
    """Return only non-sensitive Git state, never remotes, config, or diffs."""

    def call(*arguments: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *arguments],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip()

    sha = call("rev-parse", "HEAD")
    status = call("status", "--porcelain")
    return {"sha": sha, "dirty": bool(status) if status is not None else None}


def package_manifest() -> dict[str, str]:
    """Record installed distribution names and versions, without URLs or paths."""

    packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        try:
            name = distribution.metadata["Name"]
        except KeyError:
            name = None
        if name:
            packages[str(name).lower()] = str(distribution.version)
    return dict(sorted(packages.items()))


def accelerator_metadata() -> dict[str, Any]:
    """Collect accelerator capabilities through PyTorch, if it is importable."""

    try:
        import torch
    except ImportError:
        return {"torch_available": False, "cuda_available": False, "mps_available": False}

    cuda_available = bool(torch.cuda.is_available())
    devices: list[dict[str, Any]] = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            capability = torch.cuda.get_device_capability(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": int(properties.total_memory),
                    "compute_capability": [int(capability[0]), int(capability[1])],
                }
            )
    mps_backend = getattr(torch.backends, "mps", None)
    return {
        "torch_available": True,
        "torch_version": str(torch.__version__),
        "cuda_available": cuda_available,
        "cuda_runtime": str(torch.version.cuda) if torch.version.cuda is not None else None,
        "cudnn_version": torch.backends.cudnn.version() if cuda_available else None,
        "cuda_devices": devices,
        "mps_available": bool(mps_backend and mps_backend.is_available()),
        "mps_built": bool(mps_backend and mps_backend.is_built()),
    }


def environment_manifest(repo: str | Path) -> dict[str, Any]:
    """Create a secret-free execution manifest without reading ``os.environ``."""

    repository = Path(repo).resolve()
    return {
        "captured_at": utc_now(),
        "python": {
            "version": sys.version,
            "implementation": platform.python_implementation(),
        },
        "system": {
            "platform": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "accelerator": accelerator_metadata(),
        "packages": package_manifest(),
        "git": _git_metadata(repository),
    }


def _assert_secret_free(value: Any, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = str(key)
            normalized = name.lower().replace("-", "_")
            if normalized in _SECRET_KEYS or normalized.endswith(_SECRET_SUFFIXES):
                location = ".".join((*path, name))
                raise ArtifactError(f"secret-like metadata key is forbidden: {location}")
            _assert_secret_free(item, (*path, name))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_secret_free(item, (*path, str(index)))


def _append_json_lines(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    """Append a whole JSONL batch under an advisory lock and fsync it."""

    lines = [
        json.dumps(json_safe(dict(record)), sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
        for record in records
    ]
    if not lines:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        try:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_EX)
        except ImportError:  # pragma: no cover - GoalZendo targets Linux/macOS
            fcntl = None  # type: ignore[assignment]
        payload = "".join(lines).encode("utf-8")
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
        if fcntl is not None:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.is_file():
        return []
    records: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ArtifactError(f"invalid JSONL at {target}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ArtifactError(f"non-object JSONL record at {target}:{line_number}")
            records.append(value)
    return records


def _repair_incomplete_jsonl_tail(path: Path) -> bool:
    """Remove only a crash-truncated final line; reject all other corruption."""

    if not path.is_file() or path.stat().st_size == 0:
        return False
    payload = path.read_bytes()
    lines = payload.splitlines(keepends=True)
    valid_size = 0
    for index, line in enumerate(lines):
        if not line.strip():
            valid_size += len(line)
            continue
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise json.JSONDecodeError("JSONL record is not an object", line.decode(), 0)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            is_truncated_tail = index == len(lines) - 1 and not payload.endswith(b"\n")
            if not is_truncated_tail:
                raise ArtifactError(f"corrupt JSONL stream cannot be resumed: {path}") from exc
            atomic_text(path, payload[:valid_size].decode("utf-8"))
            return True
        valid_size += len(line)
    return False


def _slug(value: Any, fallback: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip()).strip(".-")
    return normalized or fallback


class RunStore:
    """Directory transaction and append-only streams for one training seed."""

    def __init__(
        self,
        root: str | Path,
        config: Mapping[str, Any],
        seed: int,
        repo: str | Path,
    ) -> None:
        self.root = Path(root).resolve()
        self.repo = Path(repo).resolve()
        self.config = copy.deepcopy(dict(config))
        self.seed = int(seed)
        self.implementation = implementation_provenance(self.repo)
        identity = {
            "config": _identity_config(self.config),
            "seed": self.seed,
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "source_fingerprint_schema_version": SOURCE_FINGERPRINT_SCHEMA_VERSION,
            "implementation_fingerprint": self.implementation["implementation_fingerprint"],
        }
        self.identity = identity
        self.run_id = stable_hash(identity, 20)
        experiment = self.config.get("experiment", {})
        experiment_id = _slug(experiment.get("id", "experiment"), "experiment")
        experiment_name = _slug(experiment.get("name", "unnamed"), "unnamed")
        self.path = self.root / experiment_id / experiment_name / self.run_id
        self.metrics_path = self.path / "metrics.jsonl"
        self.predictions_path = self.path / "predictions.jsonl"

    @property
    def complete(self) -> bool:
        return (self.path / "COMPLETE").is_file()

    @property
    def status(self) -> dict[str, Any] | None:
        path = self.path / "status.json"
        return read_json(path) if path.is_file() else None

    def initialize(self, *, resume: bool = True) -> str:
        """Start or resume the run, returning ``new``, ``resumed``, or ``complete``."""

        validate_config(self.config)
        self.path.mkdir(parents=True, exist_ok=True)
        identity_path = self.path / "identity.json"
        if identity_path.is_file():
            if read_json(identity_path) != json_safe(self.identity):
                raise ArtifactConflictError(f"artifact identity mismatch at {self.path}")
        else:
            write_json(identity_path, self.identity)

        if self.complete:
            if not resume:
                raise ArtifactConflictError(f"completed artifact already exists: {self.path}")
            verify_completion_attestation(self.path)
            return "complete"

        previous = self.status
        if previous is not None and not resume:
            raise ArtifactConflictError(
                f"incomplete artifact already exists; resume it or choose a new identity: {self.path}"
            )
        state = "resumed" if previous is not None else "new"
        attempt = int(previous.get("attempt", 0)) + 1 if previous else 1
        created_at = str(previous.get("created_at") or utc_now()) if previous else utc_now()

        resolved = copy.deepcopy(self.config)
        resolved["seed"] = self.seed
        resolved_text = yaml.safe_dump(json_safe(resolved), sort_keys=True)
        resolved_path = self.path / "resolved_config.yaml"
        if not resolved_path.is_file():
            atomic_text(resolved_path, resolved_text)
        if not (self.path / "implementation.json").is_file():
            write_json(self.path / "implementation.json", self.implementation)

        repaired_streams = [
            path.name
            for path in (self.metrics_path, self.predictions_path)
            if _repair_incomplete_jsonl_tail(path)
        ]

        attempt_manifest = environment_manifest(self.repo)
        attempt_manifest.update({"run_id": self.run_id, "seed": self.seed, "attempt": attempt})
        attempts = self.path / "attempts"
        write_json(attempts / f"attempt-{attempt:04d}.json", attempt_manifest)
        atomic_text(attempts / f"resolved-config-{attempt:04d}.yaml", resolved_text)
        if not (self.path / "environment.json").is_file():
            write_json(self.path / "environment.json", attempt_manifest)

        status = {
            "state": "running",
            "run_id": self.run_id,
            "seed": self.seed,
            "attempt": attempt,
            "created_at": created_at,
            "updated_at": utc_now(),
            "resumed": state == "resumed",
            "repaired_streams": repaired_streams,
        }
        if previous and "last_step" in previous:
            status["last_step"] = previous["last_step"]
        write_json(self.path / "status.json", status)
        return state

    def record_manifest(self, kind: str, metadata: Mapping[str, Any]) -> Path:
        """Record exact dataset/model/tokenizer metadata, rejecting secret fields."""

        normalized_kind = str(kind).strip().lower()
        if normalized_kind not in _MANIFEST_KINDS:
            raise ValueError(f"manifest kind must be one of {sorted(_MANIFEST_KINDS)}")
        normalized = json_safe(dict(metadata))
        _assert_secret_free(normalized)
        payload = {
            "schema_version": 1,
            "kind": normalized_kind,
            "digest": stable_hash(normalized, 64),
            "metadata": normalized,
        }
        target = self.path / "manifests" / f"{normalized_kind}.json"
        if target.is_file():
            if read_json(target) != payload:
                raise ArtifactConflictError(
                    f"{normalized_kind} manifest changed while resuming {self.run_id}"
                )
            return target
        write_json(target, payload)
        return target

    def record_dataset_metadata(self, metadata: Mapping[str, Any]) -> Path:
        return self.record_manifest("dataset", metadata)

    def record_model_metadata(self, metadata: Mapping[str, Any]) -> Path:
        return self.record_manifest("model", metadata)

    def record_tokenizer_metadata(self, metadata: Mapping[str, Any]) -> Path:
        return self.record_manifest("tokenizer", metadata)

    def append_metrics(
        self, records: Mapping[str, Any] | Iterable[Mapping[str, Any]]
    ) -> None:
        rows = [records] if isinstance(records, Mapping) else list(records)
        enriched = [{**dict(row), "run_id": self.run_id, "seed": self.seed} for row in rows]
        _append_json_lines(self.metrics_path, enriched)

    def append_predictions(
        self, records: Mapping[str, Any] | Iterable[Mapping[str, Any]]
    ) -> None:
        rows = [records] if isinstance(records, Mapping) else list(records)
        enriched = [{**dict(row), "run_id": self.run_id, "seed": self.seed} for row in rows]
        _append_json_lines(self.predictions_path, enriched)

    def update_status(self, **fields: Any) -> None:
        status = self.status or {"state": "running", "run_id": self.run_id, "seed": self.seed}
        safe_fields = json_safe(fields)
        _assert_secret_free(safe_fields)
        status.update(safe_fields)
        status.update({"run_id": self.run_id, "seed": self.seed, "updated_at": utc_now()})
        write_json(self.path / "status.json", status)

    def record_progress(self, step: int, **fields: Any) -> None:
        if isinstance(step, bool) or int(step) < 0:
            raise ValueError("step must be a non-negative integer")
        self.update_status(state="running", last_step=int(step), **fields)

    def finalize(self, summary: Mapping[str, Any]) -> None:
        """Commit a run.  ``COMPLETE`` is deliberately the final write."""

        normalized = json_safe(dict(summary))
        _assert_secret_free(normalized)
        target = self.path / "summary.json"
        if target.is_file() and read_json(target) != normalized:
            raise ArtifactConflictError(f"summary changed after it was written: {self.run_id}")
        write_json(target, normalized)
        previous = self.status or {}
        write_json(
            self.path / "status.json",
            {
                **previous,
                "state": "complete",
                "run_id": self.run_id,
                "seed": self.seed,
                "updated_at": utc_now(),
                "completed_at": utc_now(),
            },
        )
        # Empty streams are valid for small injected backends, but their
        # existence and byte identity are still part of the completion seal.
        for stream in (self.metrics_path, self.predictions_path):
            if not stream.is_file():
                atomic_text(stream, "")
        files = {
            relative: _file_attestation(
                self.path / relative,
                count_jsonl=relative in {"metrics.jsonl", "predictions.jsonl"},
            )
            for relative in _COMPLETION_ATTESTED_FILES
        }
        completion_body = {
            "schema": "goalzendo.run_completion",
            "schema_version": 1,
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "seed": self.seed,
            "files": files,
        }
        write_json(
            self.path / "completion.json",
            {**completion_body, "completion_digest": stable_hash(completion_body, 64)},
        )
        atomic_text(self.path / "COMPLETE", "complete\n")

    def fail(self, error: BaseException) -> None:
        """Record a safe failure type; exception text may contain credentials."""

        previous = self.status or {}
        write_json(
            self.path / "status.json",
            {
                **previous,
                "state": "failed",
                "run_id": self.run_id,
                "seed": self.seed,
                "error_type": type(error).__name__,
                "updated_at": utc_now(),
                "failed_at": utc_now(),
            },
        )


def discover_runs(root: str | Path, *, completed_only: bool = True) -> list[Path]:
    runs = sorted(path.parent for path in Path(root).glob("**/identity.json"))
    if completed_only:
        runs = [path for path in runs if (path / "COMPLETE").is_file()]
    return runs


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "SOURCE_FINGERPRINT_SCHEMA_VERSION",
    "ArtifactConflictError",
    "ArtifactError",
    "RunStore",
    "accelerator_metadata",
    "atomic_text",
    "discover_runs",
    "environment_manifest",
    "implementation_provenance",
    "json_safe",
    "package_manifest",
    "read_json",
    "read_jsonl",
    "stable_hash",
    "verify_completion_attestation",
    "write_json",
]
