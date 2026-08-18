#!/usr/bin/env python3
"""Run the final ForkWorld analysis without loading every metric row into RAM.

Run-level summaries contain all inputs for H1, H2, and H4--H9.  H3 additionally
needs three checkpoint trajectories.  This launcher therefore retains only the
H3 rows consumed by the statistical report (plus conflict accuracy for its
diagnostic figure) and leaves the complete metric history in each run's
``metrics.jsonl`` file.

Keeping this launcher outside ``src/forkworld`` is intentional: it can be fixed
while a sweep is running without changing the source fingerprint used in run
identities.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from forkworld.analysis import evaluate_hypothesis, flatten, render_figures
from forkworld.artifacts import discover_runs, json_safe, write_json

_H3_RETAINED_SERIES = frozenset(
    {
        ("iid", "target_accuracy"),
        ("conflict", "target_accuracy"),
        ("conflict", "rho_y"),
        ("conflict", "rho_p"),
    }
)


def _keep_h3_metric(record: dict[str, Any]) -> bool:
    """Return whether a row is required by the H3 report or diagnostic plot."""

    return bool(
        str(record.get("experiment", "")).lower() == "h3"
        and str(record.get("level", "")) == "choice"
        and str(record.get("intervention", "")) == "none"
        and (str(record.get("split", "")), str(record.get("metric", "")))
        in _H3_RETAINED_SERIES
    )


def collect_compact_results(
    root: str | Path,
    *,
    progress_every: int = 5_000,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Collect summaries and only the checkpoint rows needed downstream.

    The former analyzer expanded every JSONL row with a copy of its full config
    before constructing one giant DataFrame.  H7 alone produces roughly 42
    million rows at paper scale, so that representation can exceed laptop RAM
    by a wide margin.  Here non-H3 metric files are never opened, and H3 rows
    are filtered one line at a time.
    """

    source = Path(root).resolve()
    runs = discover_runs(source, completed_only=True)
    summaries: list[dict[str, Any]] = []
    retained_metrics: list[dict[str, Any]] = []
    metric_rows_scanned = 0
    metric_bytes_scanned = 0
    metric_files_scanned = 0
    metric_files_skipped = 0
    metric_bytes_skipped = 0

    for index, run in enumerate(runs, start=1):
        with (run / "resolved_config.yaml").open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        summary_path = run / "summary.json"
        summary = (
            json.loads(summary_path.read_text(encoding="utf-8"))
            if summary_path.is_file()
            else {}
        )
        run_path = str(run)
        summaries.append(
            {"run_path": run_path, **flatten(config, "config"), **flatten(summary)}
        )

        metrics_path = run / "metrics.jsonl"
        if not metrics_path.is_file():
            continue
        metric_size = metrics_path.stat().st_size
        hypothesis = str(config.get("experiment", {}).get("hypothesis", "")).lower()
        if hypothesis != "h3":
            metric_files_skipped += 1
            metric_bytes_skipped += metric_size
        else:
            metric_files_scanned += 1
            metric_bytes_scanned += metric_size
            with metrics_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    metric_rows_scanned += 1
                    record = json.loads(line)
                    if _keep_h3_metric(record):
                        retained_metrics.append({"run_path": run_path, **record})

        if progress_every > 0 and index % progress_every == 0:
            print(
                f"analysis: indexed {index:,}/{len(runs):,} completed runs",
                file=sys.stderr,
                flush=True,
            )

    collection = {
        "completed_runs": len(runs),
        "metric_strategy": "streamed_h3_analysis_subset",
        "metrics_csv_scope": (
            "H3 choice-level, no-intervention checkpoint rows for IID target "
            "accuracy and conflict target accuracy/rho_y/rho_p"
        ),
        "raw_metric_source": str(source / "<hypothesis>/<experiment>/<run-id>/metrics.jsonl"),
        "metric_files_scanned": metric_files_scanned,
        "metric_rows_scanned": metric_rows_scanned,
        "metric_rows_retained": len(retained_metrics),
        "metric_bytes_scanned": metric_bytes_scanned,
        "non_h3_metric_files_skipped": metric_files_skipped,
        "non_h3_metric_bytes_skipped": metric_bytes_skipped,
        "full_metrics_materialized": False,
    }
    return pd.DataFrame(summaries), pd.DataFrame(retained_metrics), collection


def analyze_streaming(root: str | Path, output: str | Path) -> dict[str, Any]:
    """Produce the standard reports with a bounded metric working set."""

    summaries, metrics, collection = collect_compact_results(root)
    destination = Path(output).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    reports: dict[str, Any] = {}
    hypothesis_column = (
        "config.experiment.hypothesis"
        if "config.experiment.hypothesis" in summaries
        else None
    )
    hypotheses = (
        sorted(str(value) for value in summaries[hypothesis_column].dropna().unique())
        if hypothesis_column
        else []
    )
    for hypothesis in hypotheses:
        reports[hypothesis] = evaluate_hypothesis(
            summaries,
            hypothesis,
            metrics=metrics,
        )

    figures = render_figures(summaries, metrics, destination / "figures")
    summaries.to_csv(destination / "run_summaries.csv", index=False)
    metrics.to_csv(destination / "metrics.csv", index=False)
    write_json(destination / "metric_manifest.json", collection)
    payload = {
        "hypotheses": reports,
        "figures": [str(path) for path in figures],
        "n_runs": len(summaries),
        "metric_collection": collection,
    }
    write_json(destination / "hypothesis_report.json", payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Memory-safe final analysis for paper-scale ForkWorld runs"
    )
    parser.add_argument("--input", required=True, help="artifact root")
    parser.add_argument("--output", required=True, help="analysis output directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = analyze_streaming(args.input, args.output)
    print(json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
