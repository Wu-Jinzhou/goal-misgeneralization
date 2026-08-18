#!/usr/bin/env python3
"""Stdlib-only verifier/extractor for the frozen G00-F source archive.

This file intentionally imports no project package.  The launcher runs it
before putting the extracted source tree on ``PYTHONPATH``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

FREEZE_SCHEMA = "goalzendo.g00f_h200_execution_freeze"
BUNDLE_MANIFEST_SCHEMA = "goalzendo.g00f_h200_source_bundle_payload_manifest"
RECEIPT_SCHEMA = "goalzendo.g00f_h200_extracted_source_bundle_receipt"


class BootstrapError(RuntimeError):
    pass


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


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def strict_json(path: Path, label: str) -> dict[str, Any]:
    def reject(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise BootstrapError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BootstrapError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise BootstrapError(f"{label} must contain one object")
    return value


def safe_relative(value: Any) -> str:
    relative = str(value)
    path = Path(relative)
    if not relative or path.is_absolute() or ".." in path.parts or path.as_posix() != relative:
        raise BootstrapError(f"unsafe archive member path: {relative!r}")
    return relative


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    freeze_path = args.freeze.resolve()
    if file_sha256(freeze_path) != args.expected_freeze_sha256:
        raise BootstrapError("freeze bytes differ from the externally expected SHA-256")
    freeze = strict_json(freeze_path, "G00-F execution freeze")
    freeze_body = {key: value for key, value in freeze.items() if key != "freeze_digest"}
    if (
        freeze.get("schema") != FREEZE_SCHEMA
        or freeze.get("schema_version") != 1
        or freeze.get("freeze_digest") != digest(freeze_body)
    ):
        raise BootstrapError("execution freeze schema/digest mismatch")
    controllers = freeze.get("controller_files")
    supplied_direct = {
        "bootstrap": args.actual_bootstrap,
        "detached_supervisor": args.actual_detached_supervisor,
        "launcher": args.actual_launcher,
        "qualification_controller": args.actual_qualification_controller,
        "qualification_supervisor": args.actual_qualification_supervisor,
        "watchdog": args.actual_watchdog,
    }
    if any(path.is_symlink() for path in supplied_direct.values()):
        raise BootstrapError("execution controller path must not be a symlink")
    supplied = {role: path.resolve() for role, path in supplied_direct.items()}
    if not isinstance(controllers, Mapping) or set(controllers) != set(supplied):
        raise BootstrapError("execution freeze controller-file binding is incomplete")
    if supplied["bootstrap"] != Path(__file__).resolve():
        raise BootstrapError("running bootstrap differs from --actual-bootstrap")
    authenticated_runtime_files: dict[str, dict[str, Any]] = {}
    for role, actual in supplied.items():
        binding_row = controllers.get(role)
        if (
            not isinstance(binding_row, Mapping)
            or not actual.is_file()
            or actual.is_symlink()
            or file_sha256(actual) != binding_row.get("sha256")
        ):
            raise BootstrapError(f"actual {role} bytes differ from the frozen controller")
        authenticated_runtime_files[role] = {
            "actual_path": str(actual),
            "frozen_path": safe_relative(binding_row.get("path")),
            "sha256": binding_row["sha256"],
        }
    binding = freeze.get("source_bundle")
    if not isinstance(binding, Mapping):
        raise BootstrapError("execution freeze omits source bundle binding")
    archive = args.archive.resolve()
    manifest_path = args.manifest.resolve()
    if file_sha256(archive) != binding.get("archive_sha256") or file_sha256(manifest_path) != binding.get(
        "manifest_sha256"
    ):
        raise BootstrapError("source archive or payload-manifest bytes changed")
    manifest = strict_json(manifest_path, "G00-F source payload manifest")
    manifest_body = {key: value for key, value in manifest.items() if key != "manifest_digest"}
    members = manifest.get("members")
    if (
        manifest.get("schema") != BUNDLE_MANIFEST_SCHEMA
        or manifest.get("schema_version") != 1
        or manifest.get("manifest_digest") != digest(manifest_body)
        or not isinstance(members, Sequence)
        or isinstance(members, (str, bytes))
    ):
        raise BootstrapError("source payload manifest schema/digest mismatch")
    expected: dict[str, Mapping[str, Any]] = {}
    for raw in members:
        if not isinstance(raw, Mapping):
            raise BootstrapError("source payload manifest member is malformed")
        name = safe_relative(raw.get("path"))
        if name in expected:
            raise BootstrapError("source payload manifest duplicates a member")
        if raw.get("type") != "file" or raw.get("mode") not in {0o644, 0o755}:
            raise BootstrapError("source payload manifest has an invalid type/mode")
        expected[name] = raw
    for role, authenticated in authenticated_runtime_files.items():
        frozen_path = str(authenticated["frozen_path"])
        member = expected.get(frozen_path)
        if (
            not isinstance(member, Mapping)
            or member.get("type") != "file"
            or member.get("mode") not in {0o644, 0o755}
            or member.get("sha256") != authenticated["sha256"]
        ):
            raise BootstrapError(f"authenticated outer {role} does not equal the extracted controller member")
    special = "G00F-BUNDLE-MANIFEST.json"
    with tarfile.open(archive, mode="r:gz") as handle:
        infos = handle.getmembers()
        names = [safe_relative(info.name) for info in infos]
        if len(names) != len(set(names)) or set(names) != {*expected, special}:
            raise BootstrapError("source archive member set differs from the exact manifest")
        output = args.output.resolve()
        if output.exists():
            raise BootstrapError("refusing to reuse an extracted source directory")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
        try:
            for info in infos:
                if not info.isreg() or info.issym() or info.islnk():
                    raise BootstrapError("source archive contains a non-regular member")
                extracted = handle.extractfile(info)
                if extracted is None:
                    raise BootstrapError("source archive member payload is absent")
                payload = extracted.read()
                if info.name == special:
                    if (
                        payload != manifest_path.read_bytes()
                        or info.mode & 0o777 != 0o644
                        or info.uid != 0
                        or info.gid != 0
                        or info.uname != ""
                        or info.gname != ""
                        or info.mtime != manifest.get("source_date_epoch")
                    ):
                        raise BootstrapError("in-archive payload manifest bytes changed")
                    mode = 0o644
                else:
                    row = expected[info.name]
                    if (
                        info.size != row.get("bytes")
                        or len(payload) != row.get("bytes")
                        or hashlib.sha256(payload).hexdigest() != row.get("sha256")
                        or (info.mode & 0o777) != row.get("mode")
                        or info.uid != 0
                        or info.gid != 0
                        or info.uname != ""
                        or info.gname != ""
                        or info.mtime != manifest.get("source_date_epoch")
                    ):
                        raise BootstrapError(f"source archive metadata/payload changed: {info.name}")
                    mode = int(row["mode"])
                target = temporary / info.name
                target.parent.mkdir(parents=True, exist_ok=True)
                descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
                with os.fdopen(descriptor, "wb") as output_handle:
                    output_handle.write(payload)
                    output_handle.flush()
                    os.fsync(output_handle.fileno())
                os.chmod(target, mode)
            os.replace(temporary, output)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    receipt_body = {
        "schema": RECEIPT_SCHEMA,
        "schema_version": 1,
        "freeze_file_sha256": args.expected_freeze_sha256,
        "freeze_digest": freeze["freeze_digest"],
        "archive_sha256": binding["archive_sha256"],
        "manifest_sha256": binding["manifest_sha256"],
        "manifest_digest": manifest["manifest_digest"],
        "authenticated_runtime_files": authenticated_runtime_files,
        "member_count": len(expected),
        "extracted_root": str(output),
        "tar_safety": {
            "only_regular_files": True,
            "no_absolute_or_parent_paths": True,
            "no_links": True,
            "exact_member_set": True,
            "exact_modes": True,
            "exact_bytes": True,
        },
        "g01_launch_authorized": False,
    }
    receipt = {**receipt_body, "receipt_digest": digest(receipt_body)}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    if args.receipt.exists():
        raise BootstrapError("refusing to overwrite source bundle receipt")
    descriptor = os.open(args.receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(receipt, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(prog="g00f_h200_bundle_bootstrap")
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--actual-bootstrap", type=Path, required=True)
    parser.add_argument("--actual-detached-supervisor", type=Path, required=True)
    parser.add_argument("--actual-launcher", type=Path, required=True)
    parser.add_argument("--actual-qualification-controller", type=Path, required=True)
    parser.add_argument("--actual-qualification-supervisor", type=Path, required=True)
    parser.add_argument("--actual-watchdog", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = prepare(args)
    except (BootstrapError, OSError, tarfile.TarError) as error:
        print(f"g00f_h200_bundle_bootstrap: error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
