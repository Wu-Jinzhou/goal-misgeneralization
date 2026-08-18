"""Small, resume-safe artifacts for the finite-choice hidden-law experiment."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
import yaml  # type: ignore[import-untyped]

from .config import HiddenLawCondition, canonical_digest


class HiddenLawArtifactError(RuntimeError):
    """Raised when a run directory conflicts with its registered identity."""


DEPENDENCY_SOURCE_PACKAGES = (
    "src/goalzendo",
    "src/goalzendo_interactive",
)
COMPLETION_FILES = (
    "identity.json",
    "resolved_config.yaml",
    "implementation.json",
    "environment.json",
    "bank-manifest.json",
    "model-manifest.json",
    "metrics.jsonl",
    "transcripts.jsonl",
    "predictions.jsonl",
    "summary.json",
    "status.json",
)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: Path, value: Any) -> None:
    _atomic_bytes(
        path,
        (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8"),
    )


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HiddenLawArtifactError(f"cannot read JSON object: {path}") from exc
    if type(value) is not dict:
        raise HiddenLawArtifactError(f"expected JSON object: {path}")
    return value


def append_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, allow_nan=False) + "\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    return count


def file_identity(path: Path, *, count_rows: bool = False) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    rows = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
            if count_rows:
                rows += chunk.count(b"\n")
    result: dict[str, Any] = {"sha256": digest.hexdigest(), "bytes": size}
    if count_rows:
        result["rows"] = rows
    return result


def implementation_provenance(repo: str | Path) -> dict[str, Any]:
    """Bind the new package and the exact low-level source it imports."""

    root = Path(repo).resolve()
    package = root / "src/goalzendo_hidden_law"
    local_files = sorted(
        path for path in package.glob("*.py") if path.is_file() and path.name != "__pycache__"
    )
    dependency_files = sorted(
        path
        for relative in DEPENDENCY_SOURCE_PACKAGES
        for path in (root / relative).glob("*.py")
        if path.is_file()
    )
    files = [*local_files, *dependency_files]
    if not local_files or any(not path.is_file() for path in files):
        raise HiddenLawArtifactError("hidden-law implementation source set is incomplete")
    relative_files = [path.relative_to(root).as_posix() for path in files]
    if len(relative_files) != len(set(relative_files)):
        raise HiddenLawArtifactError("implementation source set contains duplicates")
    digest = hashlib.sha256()
    digest.update(b"goalzendo-hidden-law-source-v1\0")
    identities: list[dict[str, Any]] = []
    for relative, path in sorted(zip(relative_files, files, strict=True)):
        payload = path.read_bytes()
        relative_bytes = relative.encode("utf-8")
        digest.update(len(relative_bytes).to_bytes(8, "big"))
        digest.update(relative_bytes)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        identities.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
        )
    return {
        "schema": "goalzendo.hidden_law_implementation",
        "schema_version": 1,
        "implementation_fingerprint": digest.hexdigest(),
        "source_file_count": len(identities),
        "source_files": identities,
    }


def environment_manifest() -> dict[str, Any]:
    return {
        "schema": "goalzendo.hidden_law_environment",
        "schema_version": 1,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_devices": [
            {
                "name": torch.cuda.get_device_name(index),
                "total_memory": int(torch.cuda.get_device_properties(index).total_memory),
            }
            for index in range(torch.cuda.device_count())
        ],
    }


class HiddenLawRunStore:
    """One condition directory with ordinary status, streams, and a final seal."""

    def __init__(
        self,
        output_root: str | Path,
        condition: HiddenLawCondition,
        config: Mapping[str, Any],
        implementation: Mapping[str, Any],
    ) -> None:
        self.condition = condition
        self.config = dict(config)
        self.implementation = dict(implementation)
        self.path = Path(output_root) / condition.run_id

    @property
    def complete(self) -> bool:
        return (self.path / "COMPLETE").is_file()

    def initialize(self, *, resume: bool = True) -> str:
        identity = {
            "schema": "goalzendo.hidden_law_run_identity",
            "schema_version": 1,
            "run_id": self.condition.run_id,
            "plan_key": self.condition.plan_key,
            "condition": self.condition.as_obj(),
            "implementation_fingerprint": self.implementation.get("implementation_fingerprint"),
        }
        if self.path.exists():
            observed = read_json(self.path / "identity.json")
            if observed != identity:
                raise HiddenLawArtifactError(f"run identity conflict: {self.path}")
            if self.complete:
                verify_completed_run(self.path)
                return "complete"
            if not resume:
                raise HiddenLawArtifactError(f"incomplete run exists and resume is disabled: {self.path}")
            return "resume"

        self.path.mkdir(parents=True)
        (self.path / "checkpoints").mkdir()
        write_json(self.path / "identity.json", identity)
        _atomic_bytes(
            self.path / "resolved_config.yaml",
            yaml.safe_dump(self.config, sort_keys=True).encode("utf-8"),
        )
        write_json(self.path / "implementation.json", self.implementation)
        write_json(self.path / "environment.json", environment_manifest())
        for name in ("metrics.jsonl", "transcripts.jsonl", "predictions.jsonl"):
            _atomic_bytes(self.path / name, b"")
        self.write_status(state="running", phase="initialized", last_step=0)
        return "new"

    def write_status(self, *, state: str, phase: str, last_step: int, error: str | None = None) -> None:
        write_json(
            self.path / "status.json",
            {
                "schema": "goalzendo.hidden_law_run_status",
                "schema_version": 1,
                "run_id": self.condition.run_id,
                "state": state,
                "phase": phase,
                "last_step": int(last_step),
                "error": error,
            },
        )

    def write_bank_manifest(self, value: Mapping[str, Any]) -> None:
        write_json(self.path / "bank-manifest.json", dict(value))

    def write_model_manifest(self, value: Mapping[str, Any]) -> None:
        write_json(self.path / "model-manifest.json", dict(value))

    def append_metrics(self, rows: Iterable[Mapping[str, Any]]) -> int:
        return append_jsonl(self.path / "metrics.jsonl", rows)

    def append_transcripts(self, rows: Iterable[Mapping[str, Any]]) -> int:
        return append_jsonl(self.path / "transcripts.jsonl", rows)

    def append_predictions(self, rows: Iterable[Mapping[str, Any]]) -> int:
        return append_jsonl(self.path / "predictions.jsonl", rows)

    def finish(self, summary: Mapping[str, Any]) -> dict[str, Any]:
        write_json(self.path / "summary.json", dict(summary))
        last_step = int(summary.get("last_step", 128))
        # This sealed status records scientific completion. Checkpoint cleanup
        # is a separate best-effort operation performed only after the seal.
        self.write_status(state="complete", phase="complete", last_step=last_step)
        missing = [relative for relative in COMPLETION_FILES if not (self.path / relative).is_file()]
        if missing:
            raise HiddenLawArtifactError(f"cannot complete run; missing files: {missing}")
        files = {
            relative: file_identity(
                self.path / relative,
                count_rows=relative.endswith(".jsonl"),
            )
            for relative in COMPLETION_FILES
        }
        body = {
            "schema": "goalzendo.hidden_law_completion",
            "schema_version": 1,
            "run_id": self.condition.run_id,
            "files": files,
        }
        completion_digest = canonical_digest(body)
        completion = {**body, "completion_digest": completion_digest}
        write_json(self.path / "completion.json", completion)
        _atomic_bytes(self.path / "COMPLETE", (completion_digest + "\n").encode("ascii"))
        return completion


def verify_completed_run(path: str | Path) -> dict[str, Any]:
    run = Path(path)
    completion = read_json(run / "completion.json")
    body = {key: value for key, value in completion.items() if key != "completion_digest"}
    if completion.get("completion_digest") != canonical_digest(body):
        raise HiddenLawArtifactError(f"completion digest mismatch: {run}")
    if (run / "COMPLETE").read_text(encoding="ascii") != completion["completion_digest"] + "\n":
        raise HiddenLawArtifactError(f"COMPLETE seal mismatch: {run}")
    files = completion.get("files")
    if type(files) is not dict or set(files) != set(COMPLETION_FILES):
        raise HiddenLawArtifactError(f"completion file inventory mismatch: {run}")
    for relative in COMPLETION_FILES:
        observed = file_identity(run / relative, count_rows=relative.endswith(".jsonl"))
        if files[relative] != observed:
            raise HiddenLawArtifactError(f"completed file changed: {run / relative}")
    return completion
