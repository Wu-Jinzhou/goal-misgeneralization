from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import pandas as pd
import pytest

from goalzendo.plotting import (
    PlotError,
    export_analysis_figures,
    plot_control_trajectories,
    plot_walsh_coefficients,
    register_myriad_pro,
)


def _trajectory() -> pd.DataFrame:
    rows = []
    for seed in (1, 2, 3, 4):
        for step in (0, 2, 4, 8):
            progress = step / 8
            rows.append(
                {
                    "run_id": f"run-{seed}",
                    "cell_id": "cell-one",
                    "seed": seed,
                    "step": step,
                    "prompt_view": "full",
                    "panel": "conflict",
                    "rho_y": 0.5 + 0.5 * progress,
                    "rho_p": 1.0 - 0.5 * progress,
                    "rho_q": 0.5,
                    "causal_y": progress,
                    "causal_p": 1.0 - progress,
                    "causal_q": 0.0,
                }
            )
    return pd.DataFrame(rows)


def test_control_figure_has_external_layout_and_exact_csv(tmp_path: Path) -> None:
    output = tmp_path / "control.png"
    artifacts = plot_control_trajectories(
        _trajectory(),
        output,
        bootstrap_draws=200,
        title="Synthetic control trajectories",
        dpi=100,
    )
    assert artifacts.figure == output
    assert output.is_file() and output.stat().st_size > 1000
    plotted = pd.read_csv(artifacts.plotted_csv)
    assert len(plotted) == 4 * 3 * 2
    assert set(plotted["candidate"]) == {"Y", "P", "Q"}
    assert set(plotted["measure"]) == {"behavior", "causal"}
    metadata = json.loads(artifacts.metadata_json.read_text(encoding="utf-8"))
    assert metadata["rows"] == len(plotted)
    assert metadata["plot_data_digest"]
    assert metadata["font"] == artifacts.font == register_myriad_pro()


def test_plot_refuses_to_silently_pool_cells_or_prompt_views(tmp_path: Path) -> None:
    frame = _trajectory()
    frame.loc[frame.index[-1], "cell_id"] = "different-cell"
    with pytest.raises(PlotError, match="multiple scientific cells"):
        plot_control_trajectories(frame, tmp_path / "bad.png", bootstrap_draws=20)

    frame = _trajectory()
    frame.loc[frame.index[-1], "prompt_view"] = "nonce"
    with pytest.raises(PlotError, match="multiple prompt_view"):
        plot_control_trajectories(frame, tmp_path / "bad-view.png", bootstrap_draws=20)


def test_walsh_figure_writes_the_exact_coefficients_it_displays(tmp_path: Path) -> None:
    rows = []
    for seed in (1, 2, 3, 4):
        row = {
            "run_id": f"run-{seed}",
            "cell_id": "cell",
            "seed": seed,
            "step": 8,
            "prompt_view": "full",
        }
        for term in ("y", "p", "q", "y_p", "y_q", "p_q", "y_p_q"):
            row[f"walsh_{term}"] = 1.0 if term == "y_p" else 0.0
        rows.append(row)
    output = tmp_path / "walsh.pdf"
    artifacts = plot_walsh_coefficients(
        pd.DataFrame(rows),
        output,
        bootstrap_draws=100,
    )
    assert output.is_file() and output.stat().st_size > 1000
    plotted = pd.read_csv(artifacts.plotted_csv)
    interaction = plotted[plotted["term"] == "y_p"].iloc[0]
    assert interaction["mean"] == pytest.approx(1.0)
    assert interaction["ci_low"] == pytest.approx(1.0)


def test_figure_export_separates_cells_and_writes_manifest(tmp_path: Path) -> None:
    rows = []
    for seed in (1, 2, 3, 4):
        row = {
            "run_id": f"run-{seed}",
            "cell_id": "cell-one",
            "seed": seed,
            "step": 8,
            "prompt_view": "full",
            "split": "final_factorial",
        }
        for term in ("y", "p", "q", "y_p", "y_q", "p_q", "y_p_q"):
            row[f"walsh_{term}"] = 1.0 if term == "y" else 0.0
        rows.append(row)
    exports = export_analysis_figures(
        _trajectory(),
        pd.DataFrame(rows),
        tmp_path / "figures",
        bootstrap_draws=40,
        dpi=80,
    )
    assert set(exports.figures) == {"cell-one"}
    assert len(exports.figures["cell-one"]) == 2
    assert all(item.figure.is_file() for item in exports.figures["cell-one"])
    manifest = json.loads(exports.manifest_json.read_text(encoding="utf-8"))
    assert manifest["replication_unit"] == "training_seed"
    assert manifest["cells"]["cell-one"]["seeds"] == [1, 2, 3, 4]
