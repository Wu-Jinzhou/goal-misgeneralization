"""Regression tests for the strict paper-scale follow-up analyzer."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from forkworld.artifacts import _run_identity_config, stable_hash
from forkworld.config import canonical_config, expand_sweep, load_config

REPO = Path(__file__).resolve().parents[1]


def _followup_module() -> ModuleType:
    name = "forkworld_followup_analysis_test"
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    path = REPO / "paper" / "forkworld-current-results" / "followup_analysis.py"
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def test_artifact_run_id_reconstruction_matches_runstore_contract() -> None:
    module = _followup_module()
    resolved = {
        "_case_index": 2,
        "experiment": {"hypothesis": "h5", "name": "example"},
        "run": {
            "device": "cpu",
            "output_root": "/movable/artifacts",
            "resume": True,
            "seeds": [11, 23],
        },
        "seed": 23,
        "train": {"steps": 8},
    }
    implementation = {
        "artifact_schema_version": 1,
        "source_fingerprint_schema_version": 1,
        "implementation_fingerprint": "a" * 64,
    }
    metadata = {"implementation": implementation}

    seed = int(resolved["seed"])
    original_config = dict(resolved)
    original_config.pop("seed")
    identity = {
        "config": canonical_config(_run_identity_config(original_config)),
        "seed": seed,
        **implementation,
    }

    assert module.expected_artifact_run_id(resolved, metadata) == stable_hash(identity, 20)


def test_all_registered_followup_cells_satisfy_fixed_design_contract() -> None:
    module = _followup_module()
    config_paths = {
        "e10": REPO / "configs" / "e10_rl_exploration.yaml",
        "e11": REPO / "configs" / "e11_rule_families.yaml",
        "e12": REPO / "configs" / "e12_competing_goals.yaml",
        "e13": REPO / "configs" / "e13_routeworld.yaml",
    }
    for experiment in module.SPECS:
        cells = expand_sweep(load_config(config_paths[experiment.family], ("run.device=cpu",)))
        assert all(not module.validate_fixed_design(experiment, cell) for cell in cells)

    corrupted = copy.deepcopy(expand_sweep(load_config(config_paths["e10"], ("run.device=cpu",)))[0])
    corrupted["data"]["n_train"] = 999
    errors = module.validate_fixed_design(module.SPECS[0], corrupted)
    assert "data.n_train=999, expected 10000" in errors


def test_e10_acquisition_reports_first_checkpoint_of_persistent_pair(tmp_path: Path) -> None:
    module = _followup_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    records = []
    for step, rho_y, intended_probability in ((1, 0.91, 0.2), (4, 0.95, 0.4), (16, 0.96, 0.8)):
        common = {
            "global_step": step,
            "split": "conflict_eval",
            "stage": "train",
        }
        records.extend(
            (
                {**common, "metric": "rho_y", "value": rho_y},
                {
                    **common,
                    "metric": "intended_probability",
                    "value": intended_probability,
                },
            )
        )
    (run_dir / "metrics.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    run = module.Run("e10", run_dir, {"seed": 11}, {})

    result = module.trajectory_metrics(run)

    assert result["acquisition_step"] == 1
    assert result["acquisition_observed"] == 1


def test_metrics_validation_checks_record_identity(tmp_path: Path) -> None:
    module = _followup_module()
    path = tmp_path / "metrics.jsonl"
    record = {
        "global_step": 1,
        "condition": "primary",
        "examples_seen": 1,
        "experiment": "h5",
        "intervention": "none",
        "level": "choice",
        "metric": "loss",
        "n": 1,
        "run_id": "expected-run",
        "seed": 11,
        "split": "train",
        "stage": "train",
        "stage_step": 1,
        "value": 0.25,
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    assert module.validate_metrics_file(
        path, run_id="expected-run", seed=11, hypothesis="h5"
    ) == 1

    record["run_id"] = "wrong-run"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    try:
        module.validate_metrics_file(
            path, run_id="expected-run", seed=11, hypothesis="h5"
        )
    except RuntimeError as error:
        assert "metrics identity mismatch" in str(error)
    else:
        raise AssertionError("wrong metrics identity was accepted")


def test_loader_rejects_incomplete_directories_and_bad_status(tmp_path: Path) -> None:
    module = _followup_module()
    config = expand_sweep(
        load_config(REPO / "configs" / "e10_rl_exploration.yaml", ("run.device=cpu",))
    )[0]
    config["seed"] = 11
    implementation = {
        "artifact_schema_version": 1,
        "source_fingerprint_schema_version": 1,
        "implementation_fingerprint": "b" * 64,
        "source_file_count": 23,
    }
    run_id = module.expected_artifact_run_id(
        config, {"implementation": implementation}
    )
    root = tmp_path / "artifacts"
    experiment_dir = root / "h5" / "rl_entropy_shortcut_boundary"
    run_dir = experiment_dir / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=True), encoding="utf-8"
    )
    (run_dir / "summary.json").write_text(
        json.dumps({"hypothesis": "h5", "seed": 11}), encoding="utf-8"
    )
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "implementation": implementation,
                "run_id": run_id,
                "seed": 11,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "status.json").write_text(
        json.dumps({"run_id": run_id, "state": "complete"}), encoding="utf-8"
    )
    (run_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")
    metric = {
        "condition": "primary",
        "examples_seen": 1,
        "experiment": "h5",
        "global_step": 1,
        "intervention": "none",
        "level": "choice",
        "metric": "loss",
        "n": 1,
        "run_id": run_id,
        "seed": 11,
        "split": "train",
        "stage": "train",
        "stage_step": 1,
        "value": 0.25,
    }
    (run_dir / "metrics.jsonl").write_text(json.dumps(metric) + "\n", encoding="utf-8")
    spec = module.ExperimentSpec(
        "e10",
        "h5",
        "rl_entropy_shortcut_boundary",
        1,
        lambda _config, _seed: ("only-cell",),
        lambda: {("only-cell",)},
    )

    runs, audit = module.load_experiment(root, spec, allow_incomplete=False)
    assert len(runs) == 1
    assert audit["complete"] is True
    assert audit["metric_records_validated"] == 1

    (experiment_dir / "interrupted-run").mkdir()
    with pytest.raises(RuntimeError, match="is incomplete"):
        module.load_experiment(root, spec, allow_incomplete=False)
    _, provisional = module.load_experiment(root, spec, allow_incomplete=True)
    assert provisional["complete"] is False
    assert provisional["incomplete_run_directories"] == 1

    (run_dir / "status.json").write_text(
        json.dumps({"run_id": run_id, "state": "failed"}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="status state is 'failed'"):
        module.load_experiment(root, spec, allow_incomplete=True)
