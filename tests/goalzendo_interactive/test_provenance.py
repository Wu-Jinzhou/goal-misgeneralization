from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from goalzendo_interactive.provenance import (
    INTERACTIVE_SOURCE_PROVENANCE_SCHEMA_VERSION,
    InteractiveSourceProvenanceError,
    interactive_source_provenance,
    verify_interactive_source_provenance,
)


def test_interactive_source_provenance_is_content_based_and_path_independent(
    tmp_path: Path,
) -> None:
    package = Path(__file__).resolve().parents[2] / "src" / "goalzendo_interactive"
    copied = tmp_path / "elsewhere" / "goalzendo_interactive"
    shutil.copytree(package, copied, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    original = interactive_source_provenance(package)
    clone = interactive_source_provenance(copied)
    assert original == clone
    assert original.fingerprint == clone.fingerprint
    assert original.as_obj()["schema_version"] == INTERACTIVE_SOURCE_PROVENANCE_SCHEMA_VERSION
    assert original.as_obj()["file_count"] == len(original.files)
    assert verify_interactive_source_provenance(original, copied) == clone


def test_source_mutation_addition_and_removal_fail_closed(tmp_path: Path) -> None:
    package = Path(__file__).resolve().parents[2] / "src" / "goalzendo_interactive"
    copied = tmp_path / "goalzendo_interactive"
    shutil.copytree(package, copied, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    expected = interactive_source_provenance(copied)

    target = copied / "actions.py"
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(InteractiveSourceProvenanceError, match=r"changed=.*actions.py"):
        verify_interactive_source_provenance(expected, copied)

    shutil.copy2(package / "actions.py", target)
    (copied / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(InteractiveSourceProvenanceError, match=r"extra=.*unexpected.txt"):
        verify_interactive_source_provenance(expected, copied)

    (copied / "unexpected.txt").unlink()
    (copied / "query.py").unlink()
    with pytest.raises(InteractiveSourceProvenanceError, match=r"missing=.*query.py"):
        verify_interactive_source_provenance(expected, copied)


def test_source_symlink_is_rejected(tmp_path: Path) -> None:
    package = tmp_path / "goalzendo_interactive"
    package.mkdir()
    (package / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (package / "alias.py").symlink_to(package / "module.py")
    with pytest.raises(InteractiveSourceProvenanceError, match="symlink"):
        interactive_source_provenance(package)
