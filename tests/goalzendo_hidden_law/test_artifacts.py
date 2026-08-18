from __future__ import annotations

from pathlib import Path

import pytest

from goalzendo_hidden_law.artifacts import (
    HiddenLawArtifactError,
    HiddenLawRunStore,
    implementation_provenance,
    read_json,
    verify_completed_run,
)
from goalzendo_hidden_law.config import build_hidden_law_plan, load_hidden_law_config

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/goalzendo/qwen35_hidden_law_finite_choice.yaml"


def test_implementation_provenance_binds_new_and_reused_source() -> None:
    provenance = implementation_provenance(ROOT)
    paths = {entry["path"] for entry in provenance["source_files"]}
    assert "src/goalzendo_hidden_law/config.py" in paths
    assert "src/goalzendo_hidden_law/artifacts.py" in paths
    assert "src/goalzendo/modeling.py" in paths
    assert "src/goalzendo/__init__.py" in paths
    assert "src/goalzendo_interactive/catalog.py" in paths
    assert "src/goalzendo_interactive/__init__.py" in paths
    assert "src/goalzendo_interactive/objectives.py" in paths
    assert provenance["source_file_count"] == len(paths)
    assert len(provenance["implementation_fingerprint"]) == 64


def test_run_store_resumes_and_seals_exact_files(tmp_path: Path) -> None:
    config = load_hidden_law_config(CONFIG)
    condition = build_hidden_law_plan(config)[0]
    provenance = implementation_provenance(ROOT)
    store = HiddenLawRunStore(tmp_path, condition, config, provenance)

    assert store.initialize() == "new"
    store.write_bank_manifest({"schema": "test.bank", "seed": condition.seed})
    store.write_model_manifest({"schema": "test.model", "revision": condition.model_revision})
    assert store.append_metrics([{"kind": "metric", "step": 0}]) == 1
    assert store.append_transcripts([{"kind": "transcript", "episode": 0}]) == 1
    assert store.append_predictions([{"kind": "prediction", "choice": "A"}]) == 1
    store.write_status(state="running", phase="evaluated", last_step=128)
    assert store.initialize() == "resume"

    completion = store.finish({"last_step": 128, "result": "synthetic"})
    assert store.complete
    assert read_json(store.path / "status.json")["phase"] == "complete"
    assert verify_completed_run(store.path) == completion
    assert store.initialize() == "complete"


def test_completed_mutation_is_rejected(tmp_path: Path) -> None:
    config = load_hidden_law_config(CONFIG)
    condition = build_hidden_law_plan(config)[0]
    provenance = implementation_provenance(ROOT)
    store = HiddenLawRunStore(tmp_path, condition, config, provenance)
    store.initialize()
    store.write_bank_manifest({"schema": "test.bank"})
    store.write_model_manifest({"schema": "test.model"})
    store.finish({"last_step": 128})

    with (store.path / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"late":true}\n')
    with pytest.raises(HiddenLawArtifactError, match="completed file changed"):
        verify_completed_run(store.path)
