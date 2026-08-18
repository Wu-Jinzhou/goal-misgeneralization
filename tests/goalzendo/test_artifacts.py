from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from goalzendo.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    ArtifactConflictError,
    ArtifactError,
    RunStore,
    discover_runs,
    implementation_provenance,
    read_jsonl,
    verify_completion_attestation,
)
from goalzendo.config import DEFAULT_CONFIG, deep_merge


def _config(**run: object) -> dict[str, object]:
    return deep_merge(
        DEFAULT_CONFIG,
        {
            "experiment": {"id": "gtest", "name": "artifact_contract"},
            "run": {"seeds": [3, 5], **run},
            "data": {"n_train": 100, "q_p": 0.99, "q_q": 0.9},
        },
    )


def _repo(tmp_path: Path, content: str = "VALUE = 1\n") -> Path:
    source = tmp_path / "repo" / "src" / "goalzendo"
    source.mkdir(parents=True)
    (source / "implementation.py").write_text(content, encoding="utf-8")
    return tmp_path / "repo"


def test_identity_excludes_location_resume_and_coscheduled_seeds(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    first = RunStore(tmp_path / "one", _config(output_root="one", resume=True), 3, repo)
    changed = RunStore(
        tmp_path / "two",
        _config(output_root="elsewhere", resume=False, seeds=[3, 5, 8, 13]),
        3,
        repo,
    )
    assert first.run_id == changed.run_id

    device_changed = RunStore(
        tmp_path / "three", _config(output_root="one", resume=True, device="cuda:0"), 3, repo
    )
    assert device_changed.run_id != first.run_id


def test_source_fingerprint_is_content_based_and_goalzendo_scoped(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    source = repo / "src" / "goalzendo" / "implementation.py"
    first = implementation_provenance(repo)
    assert first["artifact_schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert first["source_files"] == ["implementation.py"]

    stat = source.stat()
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert implementation_provenance(repo)["implementation_fingerprint"] == first[
        "implementation_fingerprint"
    ]

    unrelated = repo / "src" / "forkworld"
    unrelated.mkdir()
    (unrelated / "ignored.py").write_text("SECRET = 2\n", encoding="utf-8")
    assert implementation_provenance(repo)["implementation_fingerprint"] == first[
        "implementation_fingerprint"
    ]

    source.write_text("VALUE = 2\n", encoding="utf-8")
    assert implementation_provenance(repo)["implementation_fingerprint"] != first[
        "implementation_fingerprint"
    ]


def test_initialize_manifests_streams_and_complete_are_resume_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo(tmp_path)
    marker = "DO_NOT_SERIALIZE_THIS_ENVIRONMENT_VALUE_92831"
    monkeypatch.setenv("GOALZENDO_TEST_SECRET", marker)
    store = RunStore(tmp_path / "artifacts", _config(), 3, repo)
    assert store.initialize() == "new"
    store.record_dataset_metadata({"digest": "dataset-1", "count": 100})
    store.record_model_metadata({"commit": "abc123", "parameters": 12})
    store.record_tokenizer_metadata({"commit": "def456", "eos_token_id": 7})
    store.append_metrics(
        [
            {"step": 0, "metric": "reward", "value": float("nan")},
            {"step": 1, "metric": "reward", "value": 1.0, "run_id": "spoof"},
        ]
    )
    store.append_predictions({"sample_id": "x", "prediction": "A"})
    store.record_progress(1, phase="train")
    store.finalize({"final_reward": 1.0})

    assert store.complete
    assert store.initialize(resume=True) == "complete"
    assert discover_runs(tmp_path / "artifacts") == [store.path]
    metrics = read_jsonl(store.metrics_path)
    assert metrics[0]["value"] is None
    assert metrics[1]["run_id"] == store.run_id
    assert len(read_jsonl(store.predictions_path)) == 1
    assert json.loads((store.path / "status.json").read_text())["state"] == "complete"
    completion = verify_completion_attestation(store.path)
    assert completion["files"]["metrics.jsonl"]["rows"] == 2
    assert completion["files"]["predictions.jsonl"]["rows"] == 1
    assert marker not in "".join(
        path.read_text(encoding="utf-8")
        for path in store.path.rglob("*")
        if path.is_file()
    )

    with pytest.raises(ArtifactConflictError, match="completed artifact"):
        store.initialize(resume=False)


def test_completed_stream_tampering_is_rejected_before_resume(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "artifacts", _config(), 3, _repo(tmp_path))
    store.initialize()
    store.record_dataset_metadata({"digest": "dataset"})
    store.record_model_metadata({"revision": "model"})
    store.record_tokenizer_metadata({"revision": "tokenizer"})
    store.append_metrics({"step": 1, "value": 1.0})
    store.append_predictions({"sample_id": "x", "prediction": "A"})
    store.finalize({"run_id": store.run_id, "seed": store.seed})
    with store.metrics_path.open("a", encoding="utf-8") as handle:
        handle.write('{"run_id":"forged","seed":3,"step":2,"value":999}\n')
    with pytest.raises(ArtifactConflictError, match="file changed"):
        store.initialize(resume=True)


def test_manifest_hooks_are_exact_and_reject_secret_fields(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "artifacts", _config(), 3, _repo(tmp_path))
    store.initialize()
    metadata = {"revision": "abc", "special_tokens_map": {"eos_token": "<eos>"}}
    first = store.record_tokenizer_metadata(metadata)
    assert store.record_tokenizer_metadata(metadata) == first
    with pytest.raises(ArtifactConflictError, match="changed"):
        store.record_tokenizer_metadata({"revision": "different"})
    with pytest.raises(ArtifactError, match="secret-like"):
        store.record_model_metadata({"revision": "abc", "access_token": "never"})


def test_failure_status_never_serializes_exception_text(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "artifacts", _config(), 3, _repo(tmp_path))
    store.initialize()
    secret = "credential-value-that-must-not-be-written"
    store.fail(RuntimeError(secret))
    raw = (store.path / "status.json").read_text(encoding="utf-8")
    assert secret not in raw
    assert json.loads(raw)["error_type"] == "RuntimeError"
    assert store.initialize(resume=True) == "resumed"


def test_resume_repairs_only_a_truncated_jsonl_tail(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "artifacts", _config(), 3, _repo(tmp_path))
    store.initialize()
    store.append_metrics({"step": 1, "value": 0.5})
    with store.metrics_path.open("ab") as handle:
        handle.write(b'{"step":2,"value"')
    store.fail(RuntimeError("simulated crash"))

    assert store.initialize(resume=True) == "resumed"
    assert len(read_jsonl(store.metrics_path)) == 1
    status = json.loads((store.path / "status.json").read_text(encoding="utf-8"))
    assert status["repaired_streams"] == ["metrics.jsonl"]
