"""Regression tests for the paper-scale, bounded-memory analyzer."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import yaml


def _streaming_module() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_streaming.py"
    specification = importlib.util.spec_from_file_location("analyze_streaming", path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _write_run(
    root: Path,
    hypothesis: str,
    run_id: str,
    metric_lines: list[str],
) -> Path:
    run = root / hypothesis / "experiment" / run_id
    run.mkdir(parents=True)
    config = {
        "experiment": {"hypothesis": hypothesis, "name": "experiment"},
        "run": {"seeds": [1]},
        "seed": 1,
    }
    (run / "resolved_config.yaml").write_text(
        yaml.safe_dump(config), encoding="utf-8"
    )
    (run / "summary.json").write_text(
        json.dumps({"hypothesis": hypothesis, "seed": 1}), encoding="utf-8"
    )
    (run / "metrics.jsonl").write_text("\n".join(metric_lines) + "\n", encoding="utf-8")
    (run / "COMPLETE").write_text("complete\n", encoding="utf-8")
    return run


def _metric(split: str, metric: str, *, level: str = "choice", intervention: str = "none") -> str:
    return json.dumps(
        {
            "experiment": "h3",
            "run_id": "h3-run",
            "seed": 1,
            "global_step": 4,
            "split": split,
            "metric": metric,
            "value": 0.75,
            "level": level,
            "intervention": intervention,
        }
    )


def test_compact_collection_skips_non_h3_metric_files(tmp_path: Path) -> None:
    module = _streaming_module()
    _write_run(tmp_path, "h7", "h7-run", ["this is deliberately not JSON"])
    h3_run = _write_run(
        tmp_path,
        "h3",
        "h3-run",
        [
            _metric("iid", "target_accuracy"),
            _metric("conflict", "target_accuracy"),
            _metric("conflict", "rho_y"),
            _metric("conflict", "rho_p"),
            _metric("iid", "rho_y"),
            _metric("conflict", "confidence"),
            _metric("conflict", "rho_y", level="navigation"),
            _metric("conflict", "rho_p", intervention="flip_P"),
        ],
    )

    summaries, metrics, manifest = module.collect_compact_results(
        tmp_path, progress_every=0
    )

    assert len(summaries) == 2
    assert len(metrics) == 4
    assert set(zip(metrics["split"], metrics["metric"], strict=True)) == {
        ("iid", "target_accuracy"),
        ("conflict", "target_accuracy"),
        ("conflict", "rho_y"),
        ("conflict", "rho_p"),
    }
    assert metrics["run_path"].eq(str(h3_run)).all()
    assert manifest["metric_files_scanned"] == 1
    assert manifest["non_h3_metric_files_skipped"] == 1
    assert manifest["metric_rows_scanned"] == 8
    assert manifest["metric_rows_retained"] == 4
    assert manifest["full_metrics_materialized"] is False
