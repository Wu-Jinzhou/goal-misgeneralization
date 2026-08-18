"""Content-addressed source provenance for the separate G03 implementation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._json import json_digest

INTERACTIVE_SOURCE_PROVENANCE_SCHEMA_VERSION = 1
_IGNORED_NAMES = {".DS_Store"}
_IGNORED_SUFFIXES = {".pyc", ".pyo"}


class InteractiveSourceProvenanceError(ValueError):
    """Raised when source bytes do not match a recorded G03 implementation."""


@dataclass(frozen=True, slots=True)
class InteractiveSourceFile:
    path: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        if type(self.path) is not str or not self.path or self.path.startswith("/"):
            raise InteractiveSourceProvenanceError("source path must be nonempty and relative")
        parts = Path(self.path).parts
        if ".." in parts or "." in parts:
            raise InteractiveSourceProvenanceError("source path cannot contain dot traversal")
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size < 0:
            raise InteractiveSourceProvenanceError("source size must be a non-negative integer")
        if (
            type(self.sha256) is not str
            or len(self.sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.sha256)
        ):
            raise InteractiveSourceProvenanceError("source sha256 must be lowercase hexadecimal")

    def as_obj(self) -> dict[str, str | int]:
        return {"path": self.path, "size": self.size, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class InteractiveSourceProvenance:
    files: tuple[InteractiveSourceFile, ...]
    fingerprint: str

    def __post_init__(self) -> None:
        files = tuple(self.files)
        object.__setattr__(self, "files", files)
        if not files or any(type(record) is not InteractiveSourceFile for record in files):
            raise InteractiveSourceProvenanceError("source provenance requires file records")
        paths = tuple(record.path for record in files)
        if paths != tuple(sorted(paths)) or len(set(paths)) != len(paths):
            raise InteractiveSourceProvenanceError("source records must have unique sorted paths")
        expected = _fingerprint(files)
        if self.fingerprint != expected:
            raise InteractiveSourceProvenanceError("source fingerprint is inconsistent with records")

    def as_obj(self) -> dict[str, Any]:
        return {
            "schema_version": INTERACTIVE_SOURCE_PROVENANCE_SCHEMA_VERSION,
            "fingerprint": self.fingerprint,
            "file_count": len(self.files),
            "total_bytes": sum(record.size for record in self.files),
            "files": [record.as_obj() for record in self.files],
        }


def _fingerprint(files: tuple[InteractiveSourceFile, ...]) -> str:
    return json_digest(
        {
            "schema_version": INTERACTIVE_SOURCE_PROVENANCE_SCHEMA_VERSION,
            "files": [record.as_obj() for record in files],
        },
        domain="goalzendo-interactive-source-provenance-v1",
    )


def _package_root(package_root: Path | str | None) -> Path:
    selected = Path(__file__).resolve().parent if package_root is None else Path(package_root)
    selected = selected.resolve()
    if not selected.is_dir():
        raise InteractiveSourceProvenanceError(f"source package root is not a directory: {selected}")
    return selected


def _source_paths(root: Path) -> tuple[Path, ...]:
    paths: list[Path] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts:
            continue
        if path.name in _IGNORED_NAMES or path.suffix in _IGNORED_SUFFIXES:
            continue
        if path.is_symlink():
            raise InteractiveSourceProvenanceError(f"source tree contains a symlink: {relative}")
        if path.is_file():
            paths.append(path)
    if not paths:
        raise InteractiveSourceProvenanceError("source package contains no files")
    return tuple(sorted(paths, key=lambda path: path.relative_to(root).as_posix()))


def interactive_source_provenance(
    package_root: Path | str | None = None,
) -> InteractiveSourceProvenance:
    """Hash every persistent file in the actual or explicitly supplied package."""

    root = _package_root(package_root)
    records: list[InteractiveSourceFile] = []
    for path in _source_paths(root):
        payload = path.read_bytes()
        records.append(
            InteractiveSourceFile(
                path=path.relative_to(root).as_posix(),
                size=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
            )
        )
    files = tuple(records)
    return InteractiveSourceProvenance(files=files, fingerprint=_fingerprint(files))


def verify_interactive_source_provenance(
    expected: InteractiveSourceProvenance,
    package_root: Path | str | None = None,
) -> InteractiveSourceProvenance:
    if type(expected) is not InteractiveSourceProvenance:
        raise TypeError("expected must be InteractiveSourceProvenance")
    observed = interactive_source_provenance(package_root)
    if observed != expected:
        expected_by_path = {record.path: record for record in expected.files}
        observed_by_path = {record.path: record for record in observed.files}
        missing = sorted(set(expected_by_path) - set(observed_by_path))
        extra = sorted(set(observed_by_path) - set(expected_by_path))
        changed = sorted(
            path
            for path in set(expected_by_path) & set(observed_by_path)
            if expected_by_path[path] != observed_by_path[path]
        )
        raise InteractiveSourceProvenanceError(
            "interactive source provenance mismatch: "
            f"missing={missing!r}, extra={extra!r}, changed={changed!r}"
        )
    return observed
