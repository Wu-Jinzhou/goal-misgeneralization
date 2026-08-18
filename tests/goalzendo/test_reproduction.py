from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from goalzendo.artifacts import RunStore, write_json
from goalzendo.config import load_config
from goalzendo.experiment import model_state_sha256
from goalzendo.reproduction import audit_exact_replicas, audit_paired_prefix
from goalzendo.runner import RunSpec, build_plan

ROOT = Path(__file__).resolve().parents[2]


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _materialize_fake_run(
    root: Path,
    spec: RunSpec,
    *,
    repo: Path,
    snapshot_steps: tuple[int, ...],
    extra_view: bool = False,
) -> Path:
    path = RunStore(root, spec.config, spec.seed, repo).path
    (path / "manifests").mkdir(parents=True)
    (path / "snapshots").mkdir()
    (path / "COMPLETE").write_text("complete\n", encoding="utf-8")
    write_json(path / "identity.json", {"seed": spec.seed})
    write_json(path / "implementation.json", {"implementation": "same"})
    write_json(
        path / "summary.json",
        {"derived_seeds": dict(spec.seeds), "seed": spec.seed},
    )
    write_json(
        path / "manifests" / "dataset.json",
        {
            "metadata": {
                "rendering": {
                    "training_prompt_digest": "prompt",
                    "training_action_digest": "action",
                    "training_renderer_digest": "renderer",
                }
            }
        },
    )
    write_json(path / "manifests" / "model.json", {"metadata": {"model": "same"}})
    write_json(path / "manifests" / "tokenizer.json", {"metadata": {"tokenizer": "same"}})

    metrics = [
        {
            "kind": "optimization",
            "record_id": f"opt-{step}",
            "run_id": path.name,
            "step": step,
            "loss": step / 10,
        }
        for step in (1, 2, 64, 128)
    ]
    predictions = [
        {
            "record_id": f"iid-{step}",
            "run_id": path.name,
            "step": step,
            "split": "iid_validation",
            "prompt_view": "audit_law_matched",
            "sample_id": "shared-iid",
            "action": step % 2,
        }
        for step in (0, 64, 128)
    ]
    predictions.extend(
        {
            "record_id": f"diag-{step}",
            "run_id": path.name,
            "step": step,
            "split": "diagnostic_factorial",
            "prompt_view": "audit_law_matched",
            "sample_id": "shared-diagnostic",
            "action": step % 2,
        }
        for step in (0, 64)
    )
    if extra_view:
        predictions.append(
            {
                "record_id": "law-only-extra",
                "run_id": path.name,
                "step": 128,
                "split": "iid_validation",
                "prompt_view": "law_only",
                "sample_id": "shared-iid",
                "action": 0,
            }
        )
    _jsonl(path / "metrics.jsonl", metrics)
    _jsonl(path / "predictions.jsonl", predictions)

    entries: list[dict[str, object]] = []
    for step in snapshot_steps:
        model_state = {"weight": torch.tensor([float(spec.seed), float(step)])}
        state_digest = model_state_sha256(model_state)
        payload = {
            "artifact_kind": "weights_only_snapshot",
            "step": step,
            "model_state_sha256": state_digest,
            "model_state": model_state,
        }
        snapshot = path / "snapshots" / f"weights-step-{step:08d}.pt"
        torch.save(payload, snapshot)
        entries.append(
            {
                "step": step,
                "file": snapshot.name,
                "sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                "model_state_sha256": state_digest,
            }
        )
    write_json(path / "snapshots" / "index.json", {"snapshots": entries})
    return path


def test_exact_replica_audit_passes_and_detects_prediction_drift(tmp_path: Path) -> None:
    config = load_config(
        ROOT / "configs" / "goalzendo" / "g00a2_deterministic_reproduction_preflight.yaml"
    )
    plan = build_plan(config)
    paths = [
        _materialize_fake_run(tmp_path / "artifacts", spec, repo=ROOT, snapshot_steps=(128,))
        for spec in plan
    ]
    result = audit_exact_replicas(tmp_path / "artifacts", config, repo=ROOT)
    assert result["passed"] is True
    assert result["details"]["raw_metrics_files_identical"] is False
    assert result["details"]["raw_prediction_files_identical"] is False
    assert result["details"]["artifact_identity_fields_ignored"] == ["run_id"]

    with (paths[1] / "predictions.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps({"drift": True, "run_id": paths[1].name}, sort_keys=True) + "\n"
        )
    result = audit_exact_replicas(tmp_path / "artifacts", config, repo=ROOT)
    assert result["passed"] is False
    assert result["failures"] == ["prediction_stream_exact"]


def test_paired_prefix_ignores_extra_view_but_requires_exact_state(tmp_path: Path) -> None:
    config = load_config(
        ROOT / "configs" / "goalzendo" / "g00a2_deterministic_matched_horizon.yaml"
    )
    paths: dict[tuple[int, str], Path] = {}
    for spec in build_plan(config):
        arm = str(spec.config["experiment"]["horizon_arm"])
        paths[(spec.seed, arm)] = _materialize_fake_run(
            tmp_path / "artifacts",
            spec,
            repo=ROOT,
            snapshot_steps=(128,) if arm == "baseline_128" else (128, 256, 512),
            extra_view=arm == "extended_512",
        )
    result = audit_paired_prefix(tmp_path / "artifacts", config, repo=ROOT)
    assert result["passed"] is True

    index_path = paths[(9201, "extended_512")] / "snapshots" / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["snapshots"][0]["model_state_sha256"] = "0" * 64
    write_json(index_path, index)
    try:
        audit_paired_prefix(tmp_path / "artifacts", config, repo=ROOT)
    except Exception as error:
        assert "payload/index" in str(error)
    else:  # pragma: no cover - fail loudly if corruption is accepted
        raise AssertionError("corrupt snapshot digest was accepted")
