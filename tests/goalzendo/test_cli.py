from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from goalzendo.cli import _repo_root, main
from goalzendo.config import ConfigError
from goalzendo.runner import BackendPreparation, BackendResult

from .test_analysis import _write_panel


def _write_config(path: Path, *, guarded: bool = False) -> None:
    value = {
        "experiment": {"id": "gcli", "name": "cli_test", "status": "prospective"},
        "run": {
            "output_root": str(path.parent / "artifacts"),
            "seeds": [3],
            "launch_guard": "pilot not frozen" if guarded else None,
        },
        "data": {"n_train": 100, "n_validation": 100, "q_p": 0.99, "q_q": 0.9},
        "train": {"steps": 2, "eval_steps": [0, 1, 2]},
    }
    path.write_text(yaml.safe_dump(value), encoding="utf-8")


def test_repo_override_must_match_the_imported_goalzendo_source(tmp_path: Path) -> None:
    (tmp_path / "src" / "goalzendo").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fake'\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="actually imported"):
        _repo_root(tmp_path)


def test_validate_plan_and_dry_run_are_backend_free(tmp_path: Path, capsys: object) -> None:
    config = tmp_path / "config.yaml"
    _write_config(config)
    assert main(["validate", str(config), "--json"]) == 0
    validated = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert validated["valid"] is True

    assert main(["plan", str(config)]) == 0
    plan_output = capsys.readouterr().out  # type: ignore[attr-defined]
    assert json.loads(plan_output.strip())["seed"] == 3

    assert (
        main(
            [
                "run",
                str(config),
                "--dry-run",
                "--backend",
                "does.not.exist:backend",
            ]
        )
        == 0
    )
    dry_lines = capsys.readouterr().out.strip().splitlines()  # type: ignore[attr-defined]
    assert json.loads(dry_lines[0])["state"] == "planned"
    assert not (tmp_path / "artifacts").exists()


def test_generate_and_inspect_dataset_manifest(tmp_path: Path, capsys: object) -> None:
    config = tmp_path / "config.yaml"
    output = tmp_path / "dataset.json"
    _write_config(config)
    assert main(["generate", str(config), "--split", "factorial", "--output", str(output)]) == 0
    generated = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert generated["n"] == 8 * 64
    assert output.is_file()

    assert main(["inspect", str(output)]) == 0
    inspected = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert inspected["type"] == "dataset_manifest"
    assert inspected["decisions"] == 8 * 64


def test_guarded_run_fails_before_backend_or_artifact_creation(tmp_path: Path, capsys: object) -> None:
    config = tmp_path / "guarded.yaml"
    _write_config(config, guarded=True)
    assert main(["run", str(config), "--backend", "does.not.exist:backend"]) == 2
    error = capsys.readouterr().err  # type: ignore[attr-defined]
    assert "launch is locked" in error
    assert not (tmp_path / "artifacts").exists()

    assert (
        main(
            [
                "run",
                str(config),
                "--set",
                "run.protocol_unlocked=true",
                "--backend",
                "does.not.exist:backend",
            ]
        )
        == 2
    )
    bypass_error = capsys.readouterr().err  # type: ignore[attr-defined]
    assert "bound G00 gate artifact is required" in bypass_error
    assert not (tmp_path / "artifacts").exists()


def test_cli_run_uses_callback_backend_and_inspects_run(
    tmp_path: Path, capsys: object, monkeypatch: object
) -> None:
    config = tmp_path / "config.yaml"
    _write_config(config)

    class Backend:
        def prepare(self, _context: object) -> BackendPreparation:
            return BackendPreparation(
                dataset_metadata={"digest": "d1"},
                model_metadata={"revision": "m1"},
                tokenizer_metadata={"revision": "t1", "eos_token_id": 1},
            )

        def run(self, _context: object) -> BackendResult:
            return BackendResult(summary={"done": True})

    import goalzendo.runner as runner

    monkeypatch.setattr(runner, "load_backend", lambda _reference: Backend())  # type: ignore[attr-defined]
    assert main(["run", str(config), "--backend", "fake:backend"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()  # type: ignore[attr-defined]
    outcome = json.loads(lines[0])
    assert outcome["state"] == "complete"

    assert main(["inspect", outcome["path"]]) == 0
    inspected = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert inspected["complete"] is True
    assert all(value["present"] for value in inspected["manifests"].values())


def test_cli_analyze_exports_only_a_complete_expected_panel(tmp_path: Path, capsys: object) -> None:
    artifacts, config = _write_panel(tmp_path)
    config_path = tmp_path / "analysis-config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = tmp_path / "exported-analysis"
    assert (
        main(
            [
                "analyze",
                str(artifacts),
                "--config",
                str(config_path),
                "--output",
                str(output),
                "--no-figures",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert result["status"] == "complete"
    assert result["figures"] is None
    assert (output / "analysis-manifest.json").is_file()


def test_cli_confirmatory_invokes_frozen_driver_lazily(
    tmp_path: Path, capsys: object, monkeypatch: object
) -> None:
    config = tmp_path / "g01.yaml"
    _write_config(config)
    artifacts = tmp_path / "attempts"
    output = tmp_path / "confirmatory"
    observed: dict[str, object] = {}

    def fake_driver(
        root: Path,
        expected_config: object,
        output_dir: Path,
        *,
        bootstrap_draws: int,
    ) -> object:
        observed.update(
            {
                "root": root,
                "config": expected_config,
                "output": output_dir,
                "draws": bootstrap_draws,
            }
        )
        return SimpleNamespace(as_dict=lambda: {"output_dir": str(output_dir), "files": {}})

    import goalzendo.analysis as analysis

    monkeypatch.setattr(analysis, "run_g01_confirmatory_analysis", fake_driver)  # type: ignore[attr-defined]
    assert (
        main(
            [
                "confirmatory",
                str(artifacts),
                "--config",
                str(config),
                "--output",
                str(output),
                "--bootstrap-draws",
                "123",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert result["status"] == "complete"
    assert observed["root"] == artifacts
    assert observed["output"] == output
    assert observed["draws"] == 123
