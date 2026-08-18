#!/usr/bin/env python3
"""Build the acyclic, deterministic prospective G00-F execution freeze."""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import tarfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from goalzendo.artifacts import implementation_provenance, stable_hash
from goalzendo.config import canonical_config, load_config
from goalzendo_g00f_h200.evaluator import create_preexecution_gate
from goalzendo_g00f_h200.freeze import (
    EXECUTION_AND_GATE_CONTRACT,
    EXECUTION_PROFILES,
    FROZEN_GOALZENDO_IMPLEMENTATION_FINGERPRINT,
    FROZEN_MODEL_DEPENDENCIES,
    FROZEN_RUNTIME_PACKAGES,
    H200_FREEZE_OUTPUT_RELATIVE,
    H200_HOUR_CEILING,
    H200_MODULE_INVOCATION_CONTRACT,
    H200_SOURCE_ARCHIVE_NAME,
    H200_SOURCE_BUNDLE_MANIFEST_NAME,
    MODEL_LEAF_FILES,
    MODEL_RUNTIME_IDENTITIES,
    PROFILE_CONFIG_SPECS,
    PROFILE_CONTRACT,
    PROFILE_SELECTOR_CONTRACT,
    QUALIFICATION_PRODUCER_CONTRACT,
    RUNPOD_OPERATOR_HANDOFF_CONTRACT,
    RUNPOD_PROVISIONING_CONTRACT,
    RUNS_PER_PANEL,
    STORAGE_PREFLIGHT_CONTRACT,
    TOKENIZER_RUNTIME_IDENTITY,
    WALL_CEILING_SECONDS,
    WATCHDOG_SUPERVISION_CONTRACT,
    WORKER_COUNT,
    FreezeError,
    config_source_chain,
    historical_h100_parent_binding,
    semantic_digest,
    sha256_file,
    verify_freeze,
    write_plan_files,
)

SOURCE_DATE_EPOCH = 1_786_406_400
DEFAULT_OUTPUT = H200_FREEZE_OUTPUT_RELATIVE
ARCHIVE_NAME = H200_SOURCE_ARCHIVE_NAME
BUNDLE_MANIFEST_NAME = H200_SOURCE_BUNDLE_MANIFEST_NAME
FREEZE_NAME = "execution-freeze.json"


def _repo(start: Path) -> Path:
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src/goalzendo").is_dir():
            return candidate
    raise FreezeError("could not locate repository")


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode()


def _write(path: Path, payload: bytes, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FreezeError(f"prospective build refuses to overwrite: {path}")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, mode)


def _source_manifest(repo: Path) -> dict[str, Any]:
    package = repo / "src/goalzendo_g00f_h200"
    names = (
        "__init__.py",
        "cli.py",
        "evaluator.py",
        "freeze.py",
        "py.typed",
        "qualification.py",
        "qualification_producer.py",
    )
    files = {name: sha256_file(package / name) for name in names}
    body = {
        "schema": "goalzendo.g00f_h200_additive_source_manifest",
        "schema_version": 1,
        "source_files": files,
        "source_digest": semantic_digest(files),
    }
    return {**body, "manifest_digest": semantic_digest(body)}


def _bundle_paths(repo: Path) -> list[str]:
    paths = {
        path.relative_to(repo).as_posix()
        for path in (repo / "src/goalzendo").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and not path.is_symlink()
    }
    paths.update(
        path.relative_to(repo).as_posix()
        for path in (repo / "src/goalzendo_g00f_h200").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and not path.is_symlink()
    )
    for profile in EXECUTION_PROFILES:
        for specification in PROFILE_CONFIG_SPECS[profile].values():
            for row in config_source_chain(repo, str(specification["path"])):
                paths.add(str(row["path"]))
            paths.add(str(specification["plan_path"]))
    paths.update(
        {
            "constraints-goalzendo.txt",
            "docs/goalzendo/g00f-h200-execution-freeze.md",
            "docs/goalzendo/protocols/g00f-capability-repair.md",
            "docs/goalzendo/protocols/g00f-h200-execution-amendment.md",
            "pyproject.toml",
            ("reproducibility/goalzendo/g00d-gate-20260811/g00-gate-assessment-derived-pre-fix.json"),
            "reproducibility/goalzendo/g00d-gate-20260811/g00e-gate-v3.json",
            "runs/goalzendo/build_g00f_h200_execution_freeze.py",
            "runs/goalzendo/g00f_h200_bundle_bootstrap.py",
            "runs/goalzendo/g00f_h200_detached_supervisor.py",
            "runs/goalzendo/g00f_h200_qualification_controller.py",
            "runs/goalzendo/g00f_h200_qualification_supervisor.py",
            "runs/goalzendo/g00f_h200_watchdog.py",
            "runs/goalzendo/run_g00f_frozen_4h200.sh",
            "reproducibility/goalzendo/g00f-execution-freeze-20260811/execution-freeze.json",
            "reproducibility/goalzendo/g00f-execution-freeze-20260811/g00f-execution-source.tar.gz",
            ("reproducibility/goalzendo/g00f-execution-freeze-20260811/g00f-source-bundle-manifest.json"),
        }
    )
    for relative in paths:
        target = repo / relative
        if not target.is_file() or target.is_symlink():
            raise FreezeError(f"source bundle member is absent or linked: {relative}")
    return sorted(paths)


def _mode(relative: str) -> int:
    if relative.startswith("runs/goalzendo/"):
        return 0o755
    return 0o644


def _bundle_manifest(repo: Path, relatives: Sequence[str]) -> dict[str, Any]:
    members = [
        {
            "path": relative,
            "type": "file",
            "mode": _mode(relative),
            "bytes": (repo / relative).stat().st_size,
            "sha256": sha256_file(repo / relative),
        }
        for relative in relatives
    ]
    body = {
        "schema": "goalzendo.g00f_h200_source_bundle_payload_manifest",
        "schema_version": 1,
        "source_date_epoch": SOURCE_DATE_EPOCH,
        "members": members,
    }
    return {**body, "manifest_digest": semantic_digest(body)}


def _archive_bytes(repo: Path, manifest: Mapping[str, Any], manifest_bytes: bytes) -> bytes:
    output = io.BytesIO()
    with (
        gzip.GzipFile(filename="", mode="wb", fileobj=output, compresslevel=9, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive,
    ):
        for row in manifest["members"]:
            payload = (repo / str(row["path"])).read_bytes()
            info = tarfile.TarInfo(str(row["path"]))
            info.size = len(payload)
            info.mode = int(row["mode"])
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = SOURCE_DATE_EPOCH
            archive.addfile(info, io.BytesIO(payload))
        info = tarfile.TarInfo("G00F-BUNDLE-MANIFEST.json")
        info.size = len(manifest_bytes)
        info.mode = 0o644
        info.uid = 0
        info.gid = 0
        info.uname = ""
        info.gname = ""
        info.mtime = SOURCE_DATE_EPOCH
        archive.addfile(info, io.BytesIO(manifest_bytes))
    return output.getvalue()


def _binding(repo: Path, relative: str) -> dict[str, str]:
    return {"path": relative, "sha256": sha256_file(repo / relative)}


def _freeze_payload(
    repo: Path,
    output_relative: str,
    manifest: Mapping[str, Any],
    archive_sha256: str,
    manifest_sha256: str,
    additive: Mapping[str, Any],
) -> dict[str, Any]:
    configurations: dict[str, dict[str, Any]] = {}
    for profile in EXECUTION_PROFILES:
        configurations[profile] = {}
        for panel_id, specification in PROFILE_CONFIG_SPECS[profile].items():
            config_path = str(specification["path"])
            plan_path = str(specification["plan_path"])
            config = load_config(repo / config_path)
            plan_lines = (repo / plan_path).read_text(encoding="ascii").splitlines()
            rows = [json.loads(line) for line in plan_lines]
            configurations[profile][panel_id] = {
                "config_path": config_path,
                "config_sha256": sha256_file(repo / config_path),
                "config_source_files": config_source_chain(repo, config_path),
                "canonical_config_digest": stable_hash(canonical_config(config), 64),
                "plan_path": plan_path,
                "plan_sha256": sha256_file(repo / plan_path),
                "plan_key_digest": semantic_digest(sorted(str(row["plan_key"]) for row in rows)),
                "run_count": RUNS_PER_PANEL,
            }
    models: dict[str, Any] = {}
    for panel_id, specification in PROFILE_CONFIG_SPECS["baseline"].items():
        leaves = [dict(row) for row in MODEL_LEAF_FILES[panel_id]]
        models[panel_id] = {
            "model_snapshot": {
                "repo_id": specification["model"],
                "revision": specification["revision"],
                "materialization": "fresh_regular_files_no_links_exact_10_leaf_full_repository",
                "leaf_files": leaves,
                "leaf_manifest_digest": semantic_digest(leaves),
            },
            "model_runtime_identity": dict(MODEL_RUNTIME_IDENTITIES[panel_id]),
            "tokenizer_runtime_identity": dict(TOKENIZER_RUNTIME_IDENTITY),
        }
    runtime_files = [
        "constraints-goalzendo.txt",
        "docs/goalzendo/g00f-h200-execution-freeze.md",
        "docs/goalzendo/protocols/g00f-capability-repair.md",
        "docs/goalzendo/protocols/g00f-h200-execution-amendment.md",
        "pyproject.toml",
        "runs/goalzendo/build_g00f_h200_execution_freeze.py",
        "runs/goalzendo/g00f_h200_bundle_bootstrap.py",
        "runs/goalzendo/g00f_h200_detached_supervisor.py",
        "runs/goalzendo/g00f_h200_qualification_controller.py",
        "runs/goalzendo/g00f_h200_qualification_supervisor.py",
        "runs/goalzendo/g00f_h200_watchdog.py",
        "runs/goalzendo/run_g00f_frozen_4h200.sh",
    ]
    body = {
        "schema": "goalzendo.g00f_h200_execution_freeze",
        "schema_version": 1,
        "study_id": "g00f",
        "outcomes_seen": False,
        "authorization": {
            "g00f_exact_execution_authorized": True,
            "g01_launch_authorized": False,
            "scope": "exact_g00f_frozen_worker_schedule_only",
        },
        "legacy_goalzendo": {
            "implementation_fingerprint": FROZEN_GOALZENDO_IMPLEMENTATION_FINGERPRINT,
            "implementation_provenance": implementation_provenance(repo),
        },
        "historical_h100_parent": historical_h100_parent_binding(repo),
        "additive_source": {
            "manifest_path": "src/goalzendo_g00f_h200/source_manifest.json",
            "manifest_sha256": sha256_file(repo / "src/goalzendo_g00f_h200/source_manifest.json"),
            "source_digest": additive["source_digest"],
        },
        "prior_evidence": [
            _binding(
                repo,
                "reproducibility/goalzendo/g00d-gate-20260811/g00-gate-assessment-derived-pre-fix.json",
            ),
            _binding(repo, "reproducibility/goalzendo/g00d-gate-20260811/g00e-gate-v3.json"),
        ],
        "runtime_files": [_binding(repo, relative) for relative in runtime_files],
        "controller_files": {
            "bootstrap": _binding(repo, "runs/goalzendo/g00f_h200_bundle_bootstrap.py"),
            "detached_supervisor": _binding(repo, "runs/goalzendo/g00f_h200_detached_supervisor.py"),
            "launcher": _binding(repo, "runs/goalzendo/run_g00f_frozen_4h200.sh"),
            "qualification_controller": _binding(
                repo, "runs/goalzendo/g00f_h200_qualification_controller.py"
            ),
            "qualification_supervisor": _binding(
                repo, "runs/goalzendo/g00f_h200_qualification_supervisor.py"
            ),
            "watchdog": _binding(repo, "runs/goalzendo/g00f_h200_watchdog.py"),
        },
        "source_bundle": {
            "archive_path": f"{output_relative}/{ARCHIVE_NAME}",
            "archive_sha256": archive_sha256,
            "manifest_path": f"{output_relative}/{BUNDLE_MANIFEST_NAME}",
            "manifest_sha256": manifest_sha256,
            "manifest_digest": manifest["manifest_digest"],
        },
        "runtime": {
            "worker_count": WORKER_COUNT,
            "concurrent_runs_per_gpu": 1,
            "wall_ceiling_seconds": WALL_CEILING_SECONDS,
            "h200_hour_ceiling": H200_HOUR_CEILING,
            "selected_profile_projection_source": ("authenticated_pre_itt_h200_profile_qualification"),
            "maximum_selected_profile_projection_seconds": 12 * 60 * 60,
            "projection_safety_multiplier": PROFILE_SELECTOR_CONTRACT["projection"][
                "projection_safety_multiplier"
            ],
            "image": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
            "network_volume_id": "9mut3tpzwd",
            "network_volume_mount": "/workspace",
            "data_center": "US-CA-2",
            "python": "3.12.3",
            "python_packages": dict(FROZEN_RUNTIME_PACKAGES),
            "model_dependency_versions": dict(FROZEN_MODEL_DEPENDENCIES),
            "offline_model_loading": True,
            "runpodctl": "2.9.0-c094cac",
            "runpod_provisioning": dict(RUNPOD_PROVISIONING_CONTRACT),
            "runpod_operator_handoff": dict(RUNPOD_OPERATOR_HANDOFF_CONTRACT),
            "h200_module_invocation": dict(H200_MODULE_INVOCATION_CONTRACT),
            "live_price_and_stock_are_runtime_receipt_facts": True,
        },
        "execution_and_gate_contract": dict(EXECUTION_AND_GATE_CONTRACT),
        "profile_contract": {key: dict(value) for key, value in PROFILE_CONTRACT.items()},
        "qualification_producer_contract": dict(QUALIFICATION_PRODUCER_CONTRACT),
        "storage_preflight_contract": dict(STORAGE_PREFLIGHT_CONTRACT),
        "watchdog_supervision_contract": dict(WATCHDOG_SUPERVISION_CONTRACT),
        "configurations": configurations,
        "models": models,
    }
    return body


def build(repo: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FreezeError("prospective G00-F freeze output already exists; never overwrite it")
    output_relative = output.relative_to(repo).as_posix()
    plans = write_plan_files(repo)
    additive = _source_manifest(repo)
    source_manifest_path = repo / "src/goalzendo_g00f_h200/source_manifest.json"
    additive_bytes = _json_bytes(additive)
    if source_manifest_path.exists():
        if source_manifest_path.read_bytes() != additive_bytes:
            raise FreezeError("existing additive source manifest differs from recomputed bytes")
    else:
        _write(source_manifest_path, additive_bytes)
    relatives = _bundle_paths(repo)
    manifest = _bundle_manifest(repo, relatives)
    manifest_bytes = _json_bytes(manifest)
    first_archive = _archive_bytes(repo, manifest, manifest_bytes)
    second_archive = _archive_bytes(repo, manifest, manifest_bytes)
    if first_archive != second_archive:
        raise FreezeError("two in-process clean source archive builds are not byte-identical")
    output.mkdir(parents=True)
    _write(output / BUNDLE_MANIFEST_NAME, manifest_bytes)
    _write(output / ARCHIVE_NAME, first_archive)
    freeze_body = _freeze_payload(
        repo,
        output_relative,
        manifest,
        sha256_file(output / ARCHIVE_NAME),
        sha256_file(output / BUNDLE_MANIFEST_NAME),
        additive,
    )
    freeze = {**freeze_body, "freeze_digest": semantic_digest(freeze_body)}
    _write(output / FREEZE_NAME, _json_bytes(freeze))
    verified = verify_freeze(
        repo=repo,
        freeze_path=output / FREEZE_NAME,
        expected_freeze_sha256=sha256_file(output / FREEZE_NAME),
    )
    gate_result = create_preexecution_gate(
        verified=verified,
        assessment_output=output / "preexecution-assessment.json",
        gate_output=output / "preexecution-gate.json",
    )
    readme = (
        b"# G00-F H200 prospective execution freeze\n\n"
        b"No model run occurred while producing this directory. The canonical "
        b"pre-execution gate is false and never authorizes G01. Execute only with "
        b"the externally pinned whole-file SHA-256 of `execution-freeze.json`, a "
        b"fresh exact model materialization for both panels, an externally "
        b"pinned Runpod provision receipt/raw catalog, create, and get responses, "
        b"and an authenticated "
        b"pre-ITT H200 profile qualification and selection receipt.\n"
    )
    _write(output / "README.md", readme)
    checksum_names = sorted(path.name for path in output.iterdir() if path.name != "SHA256SUMS")
    checksums = "".join(f"{sha256_file(output / name)}  {name}\n" for name in checksum_names).encode()
    _write(output / "SHA256SUMS", checksums)
    return {
        "freeze_path": str(output / FREEZE_NAME),
        "freeze_file_sha256": sha256_file(output / FREEZE_NAME),
        "freeze_digest": verified.digest,
        "archive_file_sha256": sha256_file(output / ARCHIVE_NAME),
        "manifest_file_sha256": sha256_file(output / BUNDLE_MANIFEST_NAME),
        "planned_runs": len(verified.all_rows),
        "candidate_profile_runs": {
            profile: sum(len(rows) for rows in panels.values())
            for profile, panels in verified.candidate_plans.items()
        },
        "selected_profile": None,
        "preexecution_gate_file_sha256": gate_result["gate_file_sha256"],
        "preexecution_overall_passed": gate_result["gate"]["overall_passed"],
        "g01_launch_authorized": False,
        "plans": plans,
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="build_g00f_h200_execution_freeze")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()
    try:
        repo = _repo(arguments.repo)
        expected_output = repo / DEFAULT_OUTPUT
        output = arguments.output.resolve() if arguments.output else expected_output
        if output != expected_output:
            raise FreezeError("G00-F freeze output path is fixed by the prospective contract")
        if repo not in output.parents:
            raise FreezeError("G00-F freeze output must remain inside the repository")
        result = build(repo, output)
    except (FreezeError, OSError, ValueError, tarfile.TarError) as error:
        print(f"build_g00f_h200_execution_freeze: error: {error}")
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
