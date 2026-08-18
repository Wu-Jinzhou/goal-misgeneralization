#!/usr/bin/env python3
"""Postlaunch operational publisher for the frozen Qwen3.5 geometry panel.

This wrapper adds no analysis or estimands. It waits for exact completion,
runs the SHA-pinned analyzer from the source archive, and publishes its stdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

JOB_ID = "qwen35-geometry-20260815T131905Z"
NUM_SHARDS = 8
PLANNED_RUNS = 36
ARCHIVE_NAME = "goalzendo-qwen35-geometry-src.tgz"
ARCHIVE_ROOT_NAME = "goalzendo-qwen35-geometry-src"
ANALYSIS_NAME = "qwen35-evidence-geometry-analysis.json"
PANEL_RELATIVE = Path("qwen35_evidence_geometry") / "qwen35_evidence_geometry_panel"

ARCHIVE_SHA256 = "44598313d56aafd5faf525a5b08dbd4e686b151bd53c60503bc10e3969e6bbc2"
CONFIG_SHA256 = "be932b1b57f07a6a0cab0dc53eca2c8a6d60e3f2f01c9e479e2d6aa2e096ed39"
ANALYZER_SHA256 = "1c62ee239ae1e2aa5a89821db386731b3b4c2c85da6cc64e9fc89d2f4cdb29bc"
IMPLEMENTATION_FINGERPRINT = "f061e04b2cc461ab39c044ea3e8620ec0803e5ab7eca263255bc665fee1c3857"
CONFIG_IDENTITY_SHA256 = "5650ac4efb96557671ac87d69a1c93750c70c8ed0460f55dc205ad6ae909a456"
PLAN_KEY_SET_SHA256 = "8d897f9489ed3cc08c61dca85bf5dbcc1eba4e938b0e109d76df580ac4833c57"


class PostcompletionError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PostcompletionError(f"cannot read receipt: {path}") from exc
    if not isinstance(value, dict):
        raise PostcompletionError(f"receipt is not a JSON object: {path}")
    return value


def _verify_receipts(job_root: Path) -> int:
    status = job_root / "status"
    expected_names = {f"shard-{index:02d}.complete.json" for index in range(NUM_SHARDS)}
    observed_names = {path.name for path in status.glob("shard-*.complete.json")}
    if observed_names != expected_names:
        raise PostcompletionError("terminal shard receipt inventory is not exactly eight")
    bindings = {
        "archive_sha256": ARCHIVE_SHA256,
        "config_sha256": CONFIG_SHA256,
        "implementation_fingerprint": IMPLEMENTATION_FINGERPRINT,
    }
    for index in range(NUM_SHARDS):
        path = status / f"shard-{index:02d}.complete.json"
        receipt = _read_object(path)
        exact = {
            "schema": "goalzendo.qwen35_evidence_geometry_shard_status",
            "schema_version": 1,
            "job_id": JOB_ID,
            "shard_index": index,
            "num_shards": NUM_SHARDS,
            "state": "complete",
            **bindings,
        }
        if any(receipt.get(key) != value for key, value in exact.items()):
            raise PostcompletionError(f"{path.name} differs from the frozen terminal receipt")
        if type(receipt.get("runner_rc")) is not int or receipt["runner_rc"] != 0:
            raise PostcompletionError(f"{path.name} is not an rc0 terminal receipt")
        if not isinstance(receipt.get("finished_at"), str) or not receipt["finished_at"]:
            raise PostcompletionError(f"{path.name} lacks a terminal timestamp")
    return NUM_SHARDS


def _extract_frozen_source(archive: Path, destination: Path) -> Path:
    if not archive.is_file() or _sha256(archive) != ARCHIVE_SHA256:
        raise PostcompletionError("source archive differs from its frozen SHA-256")
    try:
        with tarfile.open(archive, mode="r:gz") as handle:
            handle.extractall(destination, filter="data")
    except (OSError, tarfile.TarError) as exc:
        raise PostcompletionError("cannot extract frozen source archive") from exc
    source = destination / ARCHIVE_ROOT_NAME
    config = source / "configs" / "goalzendo" / "qwen35_evidence_geometry.yaml"
    analyzer = source / "scripts" / "analyze_qwen35_evidence_geometry.py"
    if not source.is_dir() or _sha256(config) != CONFIG_SHA256 or _sha256(analyzer) != ANALYZER_SHA256:
        raise PostcompletionError("archived config or analyzer differs from registration")
    return source


_PANEL_PREFLIGHT = r"""
import hashlib, json, sys
from pathlib import Path
from goalzendo.artifacts import discover_runs, read_json, verify_completion_attestation
from goalzendo.artifacts import implementation_provenance
from goalzendo.config import load_config
from goalzendo.runner import build_plan

source, output = map(lambda value: Path(value).resolve(), sys.argv[1:3])
fingerprint, registered_plan_digest = sys.argv[3:5]
config = load_config(source / "configs/goalzendo/qwen35_evidence_geometry.yaml")
plan = build_plan(config)
expected_keys = {spec.plan_key for spec in plan}
plan_digest = hashlib.sha256(("\n".join(sorted(expected_keys)) + "\n").encode()).hexdigest()
if len(plan) != 36 or len(expected_keys) != 36 or plan_digest != registered_plan_digest:
    raise RuntimeError("archived plan differs from registration")
if implementation_provenance(source)["implementation_fingerprint"] != fingerprint:
    raise RuntimeError("archived implementation differs from registration")

panel = output / "qwen35_evidence_geometry/qwen35_evidence_geometry_panel"
all_runs = discover_runs(panel, completed_only=False)
complete_runs = discover_runs(panel, completed_only=True)
if len(all_runs) != 36 or set(all_runs) != set(complete_runs):
    raise RuntimeError("panel is not exactly 36/36 COMPLETE")
observed_keys = []
for path in all_runs:
    identity = read_json(path / "identity.json")
    status = read_json(path / "status.json")
    summary = read_json(path / "summary.json")
    if identity.get("implementation_fingerprint") != fingerprint:
        raise RuntimeError("run identity differs from frozen source")
    if status.get("state") != "complete" or status.get("run_id") != path.name:
        raise RuntimeError("run status differs from COMPLETE seal")
    if summary.get("run_id") != path.name:
        raise RuntimeError("summary identity differs")
    observed_keys.append(summary.get("plan_key"))
    verify_completion_attestation(path)
if len(set(observed_keys)) != 36 or set(observed_keys) != expected_keys:
    raise RuntimeError("completed plan rows are missing or unexpected")
print(json.dumps({"plan_key_set_sha256": plan_digest, "verified_complete_run_count": 36},
                 sort_keys=True, separators=(",", ":")))
"""


def _environment(source: Path) -> dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(source / "src")
    environment["PYTHONNOUSERSITE"] = "1"
    environment.pop("PYTHONOPTIMIZE", None)
    return environment


def _run_command(arguments: list[str], *, source: Path) -> bytes:
    result = subprocess.run(
        arguments,
        cwd=source,
        env=_environment(source),
        check=False,
        capture_output=True,
    )
    if result.returncode:
        lines = result.stderr.decode(errors="replace").strip().splitlines()
        detail = f": {lines[-1]}" if lines else ""
        raise PostcompletionError(f"frozen postcompletion command failed{detail}")
    return result.stdout


def _run_panel_preflight(python: str, source: Path, output_root: Path) -> None:
    payload = _run_command(
        [
            python,
            "-c",
            _PANEL_PREFLIGHT,
            str(source),
            str(output_root),
            IMPLEMENTATION_FINGERPRINT,
            PLAN_KEY_SET_SHA256,
        ],
        source=source,
    )
    expected = {"plan_key_set_sha256": PLAN_KEY_SET_SHA256, "verified_complete_run_count": 36}
    try:
        if json.loads(payload) != expected:
            raise PostcompletionError("artifact preflight receipt differs")
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PostcompletionError("artifact preflight did not return JSON") from exc


def _validate_analysis(payload: bytes) -> dict[str, Any]:
    if payload.count(b"\n") != 1 or not payload.endswith(b"\n"):
        raise PostcompletionError("analyzer output is not one-line JSON")
    try:
        report = json.loads(payload)
        if not isinstance(report, dict) or payload != _canonical_bytes(report):
            raise PostcompletionError("analyzer output is not canonical JSON")
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise PostcompletionError("analyzer output is not canonical JSON") from exc
    panel = report.get("panel")
    expected_panel = {
        "completed_only": True,
        "expected_run_count": PLANNED_RUNS,
        "observed_run_count": PLANNED_RUNS,
        "expected_config_sha256": CONFIG_SHA256,
        "registered_config_identity_sha256": CONFIG_IDENTITY_SHA256,
        "implementation_fingerprint": IMPLEMENTATION_FINGERPRINT,
    }
    if (
        report.get("schema") != "goalzendo.qwen35_evidence_geometry_analysis"
        or report.get("schema_version") != 1
    ):
        raise PostcompletionError("analyzer schema differs")
    if not isinstance(panel, dict) or any(panel.get(key) != value for key, value in expected_panel.items()):
        raise PostcompletionError("analyzer panel metadata differs")
    if not all(
        isinstance(report.get(key), list) and len(report[key]) == 36
        for key in (
            "seed_endpoints",
            "seed_trajectories",
        )
    ):
        raise PostcompletionError("analyzer run inventory is not exactly 36")
    expected_contrasts = {"primary_contrast", "registered_secondary_contrast"}
    if {key for key in report if key.endswith("_contrast")} != expected_contrasts or not all(
        isinstance(report[key], dict) for key in expected_contrasts
    ):
        raise PostcompletionError("analyzer contrast inventory differs")
    return report


def _run_analyzer(python: str, source: Path, panel_root: Path) -> bytes:
    payload = _run_command(
        [python, str(source / "scripts/analyze_qwen35_evidence_geometry.py"), str(panel_root)],
        source=source,
    )
    _validate_analysis(payload)
    return payload


def _publish_bundle(output_dir: Path, analysis: bytes, status: Mapping[str, Any]) -> str:
    if output_dir.exists():
        raise PostcompletionError(f"refusing to overwrite analysis directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    digest = hashlib.sha256(analysis).hexdigest()
    try:
        (staging / ANALYSIS_NAME).write_bytes(analysis)
        (staging / "SHA256SUMS").write_text(f"{digest}  {ANALYSIS_NAME}\n", encoding="utf-8")
        (staging / "status.json").write_bytes(_canonical_bytes(status))
        os.replace(staging, output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return digest


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_postcompletion(job_root: Path, *, python: str, output_dir: Path | None = None) -> dict[str, Any]:
    started_at = _utc_now()
    job_root = job_root.resolve()
    if not job_root.is_dir():
        raise PostcompletionError(f"job root does not exist: {job_root}")
    receipt_count = _verify_receipts(job_root)
    artifact_root = job_root / "artifacts/main"
    destination = (output_dir or job_root / "analysis").resolve()
    with tempfile.TemporaryDirectory(prefix=".geometry-postcomplete-", dir=job_root) as temporary:
        source = _extract_frozen_source(job_root / "input" / ARCHIVE_NAME, Path(temporary))
        _run_panel_preflight(python, source, artifact_root)
        analysis = _run_analyzer(python, source, artifact_root / PANEL_RELATIVE)

    analysis_sha256 = hashlib.sha256(analysis).hexdigest()
    status = {
        "schema": "goalzendo.qwen35_evidence_geometry_postcompletion_status",
        "schema_version": 1,
        "status_scope": "postlaunch_operational",
        "state": "complete",
        "runner_rc": 0,
        "job_id": JOB_ID,
        "terminal_shard_receipt_count": receipt_count,
        "verified_complete_run_count": PLANNED_RUNS,
        "archive_sha256": ARCHIVE_SHA256,
        "config_sha256": CONFIG_SHA256,
        "analyzer_sha256": ANALYZER_SHA256,
        "implementation_fingerprint": IMPLEMENTATION_FINGERPRINT,
        "analysis_sha256": analysis_sha256,
        "started_at": started_at,
        "finished_at": _utc_now(),
    }
    if _publish_bundle(destination, analysis, status) != analysis_sha256:
        raise PostcompletionError("published analysis digest differs")
    return {
        "analysis_dir": str(destination),
        "analysis_sha256": analysis_sha256,
        "state": "complete",
        "status_scope": "postlaunch_operational",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish the frozen analysis after exact 36/36 completion")
    parser.add_argument("job_root", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        result = run_postcompletion(args.job_root, python=args.python, output_dir=args.output_dir)
    except PostcompletionError as exc:
        print(f"postcompletion verification failed: {exc}", file=sys.stderr)
        return 1
    print(_canonical_bytes(result).decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
