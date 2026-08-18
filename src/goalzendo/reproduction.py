"""Fail-closed exact-reproduction audits for adaptive GoalZendo studies."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .artifacts import RunStore, read_json, read_jsonl, write_json
from .config import get_path
from .experiment import model_state_sha256
from .runner import RunSpec, build_plan


class ReproductionError(RuntimeError):
    """Raised when an exact-reproduction audit is structurally impossible."""


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _artifact_path(root: Path, spec: RunSpec, repo: Path) -> Path:
    path = RunStore(root, spec.config, spec.seed, repo).path
    required = (
        path / "COMPLETE",
        path / "identity.json",
        path / "implementation.json",
        path / "summary.json",
        path / "metrics.jsonl",
        path / "predictions.jsonl",
        path / "manifests" / "dataset.json",
        path / "manifests" / "model.json",
        path / "manifests" / "tokenizer.json",
        path / "snapshots" / "index.json",
    )
    missing = [str(item.relative_to(path)) for item in required if not item.is_file()]
    if missing:
        raise ReproductionError(f"incomplete reproduction artifact {path}: {', '.join(missing)}")
    return path


def _snapshot_state_digest(path: Path, step: int) -> str:
    index = read_json(path / "snapshots" / "index.json")
    entries = [
        entry
        for entry in index.get("snapshots", [])
        if isinstance(entry, Mapping) and int(entry.get("step", -1)) == int(step)
    ]
    if len(entries) != 1:
        raise ReproductionError(f"{path}: expected one retained snapshot at step {step}")
    entry = entries[0]
    digest = str(entry.get("model_state_sha256", ""))
    if len(digest) != 64:
        raise ReproductionError(f"{path}: snapshot at step {step} lacks model_state_sha256")
    snapshot = path / "snapshots" / str(entry.get("file", ""))
    if not snapshot.is_file() or _sha256(snapshot) != entry.get("sha256"):
        raise ReproductionError(f"{path}: snapshot at step {step} fails its file digest")
    try:
        import torch

        payload = torch.load(snapshot, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, TypeError) as error:
        raise ReproductionError(f"{path}: cannot load snapshot at step {step}: {error}") from error
    if not isinstance(payload, Mapping) or not isinstance(payload.get("model_state"), Mapping):
        raise ReproductionError(f"{path}: snapshot at step {step} has no model state")
    if payload.get("model_state_sha256") != digest:
        raise ReproductionError(f"{path}: snapshot payload/index model-state digests differ")
    if model_state_sha256(payload["model_state"]) != digest:
        raise ReproductionError(f"{path}: snapshot at step {step} fails raw model-state hashing")
    return digest


def _training_provenance(path: Path) -> dict[str, Any]:
    dataset = read_json(path / "manifests" / "dataset.json")["metadata"]
    rendering = dataset["rendering"]
    summary = read_json(path / "summary.json")
    return {
        "derived_seeds": summary.get("derived_seeds"),
        "training_prompt_digest": rendering.get("training_prompt_digest"),
        "training_action_digest": rendering.get("training_action_digest"),
        "training_renderer_digest": rendering.get("training_renderer_digest"),
        "model_manifest_sha256": _sha256(path / "manifests" / "model.json"),
        "implementation_sha256": _sha256(path / "implementation.json"),
    }


def _record_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = "\n".join(
        json.dumps(dict(row), sort_keys=True, separators=(",", ":"), allow_nan=False)
        for row in rows
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _identity_normalized_rows(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Remove only the artifact identity from otherwise exact stream rows.

    RunStore deliberately enriches every metric and prediction with the
    artifact-specific ``run_id``.  Two independently materialized replicas
    therefore cannot have byte-identical JSONL files even when every
    scientific value is identical.  Validate that the embedded identity is
    internally consistent before removing that one field; all other fields,
    row order, and floating-point values remain part of the exact comparison.
    """

    identity = read_json(path / "identity.json")
    expected_run_id = str(identity.get("run_id", path.name))
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        actual_run_id = row.get("run_id")
        if actual_run_id != expected_run_id:
            raise ReproductionError(
                f"{path}: stream row {index} has run_id {actual_run_id!r}; "
                f"expected {expected_run_id!r}"
            )
        value = dict(row)
        value.pop("run_id")
        normalized.append(value)
    return normalized


def _semantic_stream_digest(path: Path, name: str) -> tuple[str, int]:
    rows = read_jsonl(path / name)
    normalized = _identity_normalized_rows(path, rows)
    return _record_digest(normalized), len(normalized)


def _result(
    *,
    mode: str,
    paths: Sequence[Path],
    checks: Mapping[str, bool],
    details: Mapping[str, Any],
) -> dict[str, Any]:
    failures = sorted(name for name, passed in checks.items() if not passed)
    return {
        "schema_version": 2,
        "artifact_kind": "goalzendo_exact_reproduction_audit",
        "mode": mode,
        "passed": not failures,
        "failures": failures,
        "run_paths": [str(path.resolve()) for path in paths],
        "checks": dict(sorted(checks.items())),
        "details": dict(details),
    }


def audit_exact_replicas(
    artifacts: str | Path,
    config: Mapping[str, Any],
    *,
    repo: str | Path,
    snapshot_step: int = 128,
) -> dict[str, Any]:
    """Require two replica-tagged runs to be byte/execution exact."""

    plan = build_plan(config)
    if len(plan) != 2 or len({spec.seed for spec in plan}) != 1:
        raise ReproductionError("exact-replica mode requires exactly two runs with one shared seed")
    replicas = [get_path(spec.config, "experiment.reproducibility_replica") for spec in plan]
    if len(set(replicas)) != 2 or any(replica is None for replica in replicas):
        raise ReproductionError("exact-replica runs require two distinct reproduction labels")
    root = Path(artifacts)
    repository = Path(repo)
    paths = [_artifact_path(root, spec, repository) for spec in plan]
    provenance = [_training_provenance(path) for path in paths]
    state_digests = [_snapshot_state_digest(path, snapshot_step) for path in paths]
    raw_metrics_digests = [_sha256(path / "metrics.jsonl") for path in paths]
    raw_prediction_digests = [_sha256(path / "predictions.jsonl") for path in paths]
    metrics_streams = [_semantic_stream_digest(path, "metrics.jsonl") for path in paths]
    prediction_streams = [
        _semantic_stream_digest(path, "predictions.jsonl") for path in paths
    ]
    checks = {
        "training_provenance_exact": provenance[0] == provenance[1],
        "metrics_stream_exact": metrics_streams[0] == metrics_streams[1],
        "prediction_stream_exact": prediction_streams[0] == prediction_streams[1],
        "model_state_exact": state_digests[0] == state_digests[1],
    }
    return _result(
        mode="exact_replicas",
        paths=paths,
        checks=checks,
        details={
            "seed": plan[0].seed,
            "replicas": replicas,
            "snapshot_step": int(snapshot_step),
            "model_state_sha256": state_digests,
            "artifact_identity_fields_ignored": ["run_id"],
            "metrics_semantic_sha256": [digest for digest, _ in metrics_streams],
            "metrics_row_count": [count for _, count in metrics_streams],
            "predictions_semantic_sha256": [
                digest for digest, _ in prediction_streams
            ],
            "predictions_row_count": [count for _, count in prediction_streams],
            "raw_metrics_file_sha256": raw_metrics_digests,
            "raw_predictions_file_sha256": raw_prediction_digests,
            "raw_metrics_files_identical": raw_metrics_digests[0]
            == raw_metrics_digests[1],
            "raw_prediction_files_identical": raw_prediction_digests[0]
            == raw_prediction_digests[1],
            "training_provenance": provenance,
        },
    )


def _filtered_prefix_rows(
    path: Path,
    *,
    prefix_step: int,
    prompt_view: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metrics = read_jsonl(path / "metrics.jsonl")
    predictions = read_jsonl(path / "predictions.jsonl")
    optimization = [
        row
        for row in metrics
        if row.get("kind") == "optimization" and int(row.get("step", -1)) <= prefix_step
    ]
    iid = [
        row
        for row in predictions
        if row.get("split") == "iid_validation"
        and row.get("prompt_view") == prompt_view
        and int(row.get("step", -1)) <= prefix_step
    ]
    shared_diagnostic = [
        row
        for row in predictions
        if row.get("split") in {"diagnostic_factorial", "diagnostic_factorial_causal"}
        and row.get("prompt_view") == prompt_view
        and int(row.get("step", -1)) < prefix_step
    ]
    return optimization, iid, shared_diagnostic


def audit_paired_prefix(
    artifacts: str | Path,
    config: Mapping[str, Any],
    *,
    repo: str | Path,
    prefix_step: int = 128,
    prompt_view: str = "audit_law_matched",
) -> dict[str, Any]:
    """Compare baseline and extended runs on their genuinely shared prefix."""

    plan = build_plan(config)
    grouped: dict[int, list[RunSpec]] = defaultdict(list)
    for spec in plan:
        grouped[int(spec.seed)].append(spec)
    if not grouped or any(len(group) != 2 for group in grouped.values()):
        raise ReproductionError("paired-prefix mode requires exactly two arms for every seed")

    root = Path(artifacts)
    repository = Path(repo)
    all_paths: list[Path] = []
    per_seed: dict[str, Any] = {}
    global_checks: dict[str, bool] = {}
    for seed, group in sorted(grouped.items()):
        by_arm = {
            str(get_path(spec.config, "experiment.horizon_arm", "")): spec for spec in group
        }
        if set(by_arm) != {"baseline_128", "extended_512"}:
            raise ReproductionError(f"seed {seed} lacks the registered baseline/extended arms")
        baseline = _artifact_path(root, by_arm["baseline_128"], repository)
        extended = _artifact_path(root, by_arm["extended_512"], repository)
        all_paths.extend((baseline, extended))
        provenance = (_training_provenance(baseline), _training_provenance(extended))
        state = (
            _snapshot_state_digest(baseline, prefix_step),
            _snapshot_state_digest(extended, prefix_step),
        )
        base_rows = _filtered_prefix_rows(
            baseline,
            prefix_step=prefix_step,
            prompt_view=prompt_view,
        )
        extended_rows = _filtered_prefix_rows(
            extended,
            prefix_step=prefix_step,
            prompt_view=prompt_view,
        )
        base_rows = (
            _identity_normalized_rows(baseline, base_rows[0]),
            _identity_normalized_rows(baseline, base_rows[1]),
            _identity_normalized_rows(baseline, base_rows[2]),
        )
        extended_rows = (
            _identity_normalized_rows(extended, extended_rows[0]),
            _identity_normalized_rows(extended, extended_rows[1]),
            _identity_normalized_rows(extended, extended_rows[2]),
        )
        row_names = ("optimization", "iid_predictions", "shared_diagnostic_predictions")
        row_digests = {
            name: (_record_digest(left), _record_digest(right))
            for name, left, right in zip(row_names, base_rows, extended_rows, strict=True)
        }
        seed_checks = {
            "training_provenance_exact": provenance[0] == provenance[1],
            "model_state_exact": state[0] == state[1],
            **{
                f"{name}_exact": digests[0] == digests[1]
                for name, digests in row_digests.items()
            },
        }
        for name, passed in seed_checks.items():
            global_checks[f"seed_{seed}.{name}"] = passed
        per_seed[str(seed)] = {
            "checks": seed_checks,
            "paths": [str(baseline.resolve()), str(extended.resolve())],
            "model_state_sha256": list(state),
            "row_sha256": {name: list(values) for name, values in row_digests.items()},
            "training_provenance": list(provenance),
        }
    return _result(
        mode="paired_prefix",
        paths=all_paths,
        checks=global_checks,
        details={
            "prefix_step": int(prefix_step),
            "prompt_view": prompt_view,
            "seeds": per_seed,
        },
    )


def write_reproduction_audit(path: str | Path, result: Mapping[str, Any]) -> Path:
    destination = Path(path)
    write_json(destination, result)
    return destination


__all__ = [
    "ReproductionError",
    "audit_exact_replicas",
    "audit_paired_prefix",
    "write_reproduction_audit",
]
