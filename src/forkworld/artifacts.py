"""Reproducible, resume-safe experiment artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from .config import canonical_config, validate_config


_NON_SCIENTIFIC_RUN_FIELDS = frozenset({"output_root", "resume", "seeds"})

# Increment this when the on-disk artifact contract changes incompatibly.  The
# source fingerprint below catches implementation changes; this explicit
# version catches semantic changes to the artifact format itself.
ARTIFACT_SCHEMA_VERSION = 1
SOURCE_FINGERPRINT_SCHEMA_VERSION = 1
_SOURCE_FILE_NAMES = frozenset({"py.typed"})
_SOURCE_FILE_SUFFIXES = frozenset({".py"})
_IMPLEMENTATION_PROVENANCE_CACHE: dict[
    tuple[str, int, int, tuple[tuple[str, int, int], ...]], dict[str, Any]
] = {}


def _run_identity_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the seed-level scientific configuration used for run hashes.

    Artifact location, restart policy, and the other seeds scheduled alongside
    this one do not change a seed-level experiment. Removing them prevents
    identical work from acquiring a new identity when results are moved or a
    sweep is extended.
    """

    identity = dict(config)
    run = dict(identity.get("run", {}))
    for field in _NON_SCIENTIFIC_RUN_FIELDS:
        run.pop(field, None)
    identity["run"] = run
    return identity


def stable_hash(value: Any, length: int = 16) -> str:
    """Hash JSON-compatible content with stable key ordering."""

    raw = json.dumps(
        json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]


def implementation_provenance(repo: Path | None = None) -> dict[str, Any]:
    """Fingerprint the package implementation that determines artifacts.

    Paths are normalized relative to ``src/forkworld`` and file timestamps are
    deliberately ignored.  In an installed wheel, where there is no repository
    source tree, the importable package directory is fingerprinted instead.
    """

    package_source = Path(__file__).resolve().parent
    candidate = Path(repo).resolve() / "src" / "forkworld" if repo is not None else None
    source_root = candidate if candidate is not None and candidate.is_dir() else package_source
    files = sorted(
        (
            path
            for path in source_root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and (path.suffix in _SOURCE_FILE_SUFFIXES or path.name in _SOURCE_FILE_NAMES)
        ),
        key=lambda path: path.relative_to(source_root).as_posix(),
    )
    if not files:
        raise RuntimeError(f"no ForkWorld source files found under {source_root}")
    signature_rows: list[tuple[str, int, int]] = []
    for path in files:
        stat = path.stat()
        signature_rows.append(
            (path.relative_to(source_root).as_posix(), stat.st_size, stat.st_mtime_ns)
        )
    signature = tuple(signature_rows)
    cache_key = (
        str(source_root),
        ARTIFACT_SCHEMA_VERSION,
        SOURCE_FINGERPRINT_SCHEMA_VERSION,
        signature,
    )
    cached = _IMPLEMENTATION_PROVENANCE_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)
    digest = hashlib.sha256()
    digest.update(f"forkworld-source-v{SOURCE_FINGERPRINT_SCHEMA_VERSION}\0".encode("utf-8"))
    for path in files:
        relative = path.relative_to(source_root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    provenance = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "source_fingerprint_schema_version": SOURCE_FINGERPRINT_SCHEMA_VERSION,
        "implementation_fingerprint": digest.hexdigest(),
        "source_file_count": len(files),
    }
    _IMPLEMENTATION_PROVENANCE_CACHE[cache_key] = provenance
    return dict(provenance)


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Cannot JSON serialize {type(value).__name__}")


def json_safe(value: Any) -> Any:
    """Recursively normalize values to strict JSON (RFC 8259 has no NaN)."""

    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return json_safe(value.item())
    return value


def atomic_text(path: Path, text: str) -> None:
    """Write a UTF-8 text file atomically within its destination directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: Path, value: Any) -> None:
    atomic_text(
        path,
        json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def git_provenance(repo: Path) -> dict[str, Any]:
    def call(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args], cwd=repo, check=True, capture_output=True, text=True, timeout=5
            )
            return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    sha = call("rev-parse", "HEAD")
    status = call("status", "--porcelain")
    return {"sha": sha, "dirty": bool(status), "status": status.splitlines() if status else []}


def environment_metadata(repo: Path) -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "torch", "scipy", "pandas", "matplotlib", "PyYAML"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "git": git_provenance(repo),
    }


class RunStore:
    """Directory-level transaction for one sweep cell and training seed."""

    def __init__(self, root: str | Path, config: Mapping[str, Any], seed: int, repo: Path):
        self.config = dict(config)
        self.seed = int(seed)
        self.implementation = implementation_provenance(repo)
        experiment = self.config["experiment"]
        identity = {
            "config": canonical_config(_run_identity_config(self.config)),
            "seed": self.seed,
            "artifact_schema_version": self.implementation["artifact_schema_version"],
            "source_fingerprint_schema_version": self.implementation[
                "source_fingerprint_schema_version"
            ],
            "implementation_fingerprint": self.implementation[
                "implementation_fingerprint"
            ],
        }
        self.run_id = stable_hash(identity, 20)
        self.path = Path(root) / str(experiment["hypothesis"]) / str(experiment["name"]) / self.run_id
        self.repo = repo
        self.metrics_path = self.path / "metrics.jsonl"
        self.predictions_path = self.path / "predictions.jsonl"

    @property
    def complete(self) -> bool:
        return (self.path / "COMPLETE").is_file()

    def initialize(self, *, reset: bool = False) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        if reset:
            for name in ("metrics.jsonl", "predictions.jsonl", "summary.json", "status.json", "COMPLETE"):
                target = self.path / name
                if target.is_file():
                    target.unlink()
            checkpoint_dir = self.path / "checkpoints"
            if checkpoint_dir.is_dir():
                for target in checkpoint_dir.glob("*.pt"):
                    target.unlink()
        resolved = dict(self.config)
        resolved["seed"] = self.seed
        atomic_text(self.path / "resolved_config.yaml", yaml.safe_dump(resolved, sort_keys=True))
        cautions = validate_config(self.config)
        metadata = environment_metadata(self.repo)
        metadata.update(
            {
                "run_id": self.run_id,
                "seed": self.seed,
                "cautions": cautions,
                "implementation": self.implementation,
            }
        )
        write_json(self.path / "metadata.json", metadata)
        write_json(self.path / "status.json", {"state": "running", "run_id": self.run_id})

    def append_metrics(self, records: Mapping[str, Any] | Iterable[Mapping[str, Any]]) -> None:
        if isinstance(records, Mapping):
            records = [records]
        self.path.mkdir(parents=True, exist_ok=True)
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            for record in records:
                enriched = {"run_id": self.run_id, "seed": self.seed, **dict(record)}
                handle.write(
                    json.dumps(json_safe(enriched), sort_keys=True, allow_nan=False) + "\n"
                )

    def append_predictions(self, records: Iterable[Mapping[str, Any]]) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        with self.predictions_path.open("a", encoding="utf-8") as handle:
            for record in records:
                enriched = {"run_id": self.run_id, "seed": self.seed, **dict(record)}
                handle.write(
                    json.dumps(json_safe(enriched), sort_keys=True, allow_nan=False) + "\n"
                )

    def save_checkpoint(self, model: Any, name: str, extra: Mapping[str, Any] | None = None) -> Path:
        import torch

        target = self.path / "checkpoints" / f"{name}.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"model": model.state_dict(), "extra": dict(extra or {})}
        temporary = target.with_suffix(".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, target)
        return target

    def save_checkpoint_payload(self, payload: Mapping[str, Any], name: str) -> Path:
        """Atomically persist an in-memory protocol checkpoint/state dictionary."""

        import torch

        target = self.path / "checkpoints" / f"{name}.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        torch.save(dict(payload), temporary)
        os.replace(temporary, target)
        return target

    def finalize(self, summary: Mapping[str, Any]) -> None:
        write_json(self.path / "summary.json", dict(summary))
        write_json(self.path / "status.json", {"state": "complete", "run_id": self.run_id})
        atomic_text(self.path / "COMPLETE", "complete\n")

    def fail(self, error: BaseException) -> None:
        write_json(
            self.path / "status.json",
            {"state": "failed", "run_id": self.run_id, "error_type": type(error).__name__, "error": str(error)},
        )


def discover_runs(root: str | Path, completed_only: bool = True) -> list[Path]:
    paths = sorted(Path(root).glob("**/resolved_config.yaml"))
    runs = [path.parent for path in paths]
    if completed_only:
        runs = [path for path in runs if (path / "COMPLETE").is_file()]
    return runs


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
