"""Publication figures for GoalZendo with auditable plotted-data sidecars."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.legend import Legend

from .analysis import DEFAULT_BOOTSTRAP_DRAWS, bootstrap_mean_interval
from .artifacts import atomic_text, stable_hash, write_json


class PlotError(RuntimeError):
    """Raised when a figure would pool incompatible data or overlap panels."""


_MEASURE_STYLE: Mapping[str, Mapping[str, Any]] = {
    "behavior": {
        "label": "Behavioral agreement",
        "color": "#266FA5",
        "marker": "o",
        "linestyle": "-",
    },
    "causal": {
        "label": "Matched causal flip",
        "color": "#C45B35",
        "marker": "s",
        "linestyle": "--",
    },
}
_CANDIDATE_TITLES = {
    "y": "Intended Law (Y)",
    "p": "Herald proxy (P)",
    "q": "Sage rule (Q)",
}


@lru_cache(maxsize=1)
def register_myriad_pro() -> str:
    """Register Myriad Pro even when Matplotlib has not indexed the font yet."""

    candidates: list[str] = []
    with suppress(OSError):
        candidates.extend(font_manager.findSystemFonts())
    for path in sorted(set(candidates)):
        if "myriad" not in Path(path).stem.lower():
            continue
        try:
            font_manager.fontManager.addfont(path)
            name = font_manager.FontProperties(fname=path).get_name()
        except (OSError, RuntimeError):
            continue
        if "Myriad" in name:
            return name
    for name in ("Myriad Pro", "Myriad", "Arial", "DejaVu Sans"):
        try:
            font_manager.findfont(name, fallback_to_default=False)
        except ValueError:
            continue
        return name
    return "DejaVu Sans"


def publication_style() -> dict[str, Any]:
    """A compact style suitable for both a paper PDF and a LessWrong post."""

    font = register_myriad_pro()
    return {
        "font.family": "sans-serif",
        "font.sans-serif": [font, "Arial", "DejaVu Sans"],
        "font.size": 10.5,
        "axes.titlesize": 11.5,
        "axes.labelsize": 10.5,
        "axes.titleweight": "semibold",
        "axes.edgecolor": "#31343A",
        "axes.linewidth": 0.8,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "xtick.color": "#31343A",
        "ytick.color": "#31343A",
        "text.color": "#24272C",
        "axes.labelcolor": "#24272C",
        "grid.color": "#D9DDE3",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.65,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
        "lines.markersize": 5.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }


@contextmanager
def _publication_context() -> Iterator[None]:
    with mpl.rc_context():
        style: Any = publication_style()
        mpl.rcParams.update(style)
        yield


def label_panels(axes: Sequence[Axes], titles: Sequence[str]) -> None:
    """Put panel letters in aligned, left-justified titles rather than floating text."""

    if len(axes) != len(titles):
        raise ValueError("axes and titles must have the same length")
    letters = "abcdefghijklmnopqrstuvwxyz"
    if len(axes) > len(letters):
        raise ValueError("too many panels")
    for index, (axis, title) in enumerate(zip(axes, titles, strict=True)):
        axis.set_title(f"({letters[index]})  {title}", loc="left", pad=11)


def _bootstrap_label(parts: Sequence[Any]) -> int:
    return int(stable_hash({"plot_bootstrap": [str(part) for part in parts]}, 16), 16) % (2**32 - 1)


def control_trajectory_plot_data(
    trajectory: pd.DataFrame,
    *,
    group_columns: Sequence[str] = (),
    confidence: float = 0.95,
    bootstrap_draws: int = DEFAULT_BOOTSTRAP_DRAWS,
) -> pd.DataFrame:
    """Collapse measurements within seed, then bootstrap across trained seeds."""

    required = {
        "seed",
        "step",
        "rho_y",
        "rho_p",
        "rho_q",
        "causal_y",
        "causal_p",
        "causal_q",
        *group_columns,
    }
    missing = sorted(required - set(trajectory.columns))
    if missing:
        raise PlotError(f"trajectory table is missing columns: {missing}")
    if not group_columns and "cell_id" in trajectory and trajectory["cell_id"].nunique() > 1:
        raise PlotError("trajectory contains multiple scientific cells; filter it or provide group_columns")
    for column in ("prompt_view", "panel"):
        if column in trajectory and column not in group_columns and trajectory[column].nunique() > 1:
            raise PlotError(f"trajectory contains multiple {column} values; filter it or group explicitly")
    records: list[dict[str, Any]] = []
    group_keys = [*group_columns, "step"]
    for values, group in trajectory.groupby(group_keys, dropna=False, sort=True):
        values = values if isinstance(values, tuple) else (values,)
        common = dict(zip(group_keys, values, strict=True))
        for candidate in ("y", "p", "q"):
            for measure, column in (
                ("behavior", f"rho_{candidate}"),
                ("causal", f"causal_{candidate}"),
            ):
                per_seed = group.groupby("seed", as_index=False)[column].mean()[column]
                interval = bootstrap_mean_interval(
                    per_seed.to_numpy(dtype=float),
                    confidence=confidence,
                    draws=bootstrap_draws,
                    seed=_bootstrap_label([*values, candidate, measure]),
                )
                records.append(
                    {
                        **common,
                        "candidate": candidate.upper(),
                        "measure": measure,
                        "mean": interval["estimate"],
                        "ci_low": interval["ci_low"],
                        "ci_high": interval["ci_high"],
                        "n_seeds": interval["n_seeds"],
                        "confidence": confidence,
                    }
                )
    return (
        pd.DataFrame(records)
        .sort_values([*group_columns, "candidate", "measure", "step"])
        .reset_index(drop=True)
    )


def _series_label(row: Mapping[str, Any], group_columns: Sequence[str], measure: str) -> str:
    base = str(_MEASURE_STYLE[measure]["label"])
    if not group_columns:
        return base
    condition = ", ".join(f"{column}={row[column]}" for column in group_columns)
    return f"{base} · {condition}"


def _style_axis(axis: Axes, steps: Sequence[int]) -> None:
    axis.set_ylim(-0.025, 1.025)
    axis.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    axis.grid(axis="y")
    axis.spines[["top", "right"]].set_visible(False)
    unique_steps = sorted(set(int(step) for step in steps))
    if len(unique_steps) > 8:
        indices = np.linspace(0, len(unique_steps) - 1, 8).round().astype(int)
        ticks = [unique_steps[index] for index in sorted(set(indices))]
    else:
        ticks = unique_steps
    axis.set_xscale("symlog", linthresh=1.0, linscale=0.8, base=2)
    axis.set_xticks(ticks)
    axis.set_xticklabels([str(tick) for tick in ticks])
    axis.set_xlabel("Optimizer step")


def _external_legend(fig: Figure, handles: Sequence[Any], labels: Sequence[str]) -> Legend:
    unique: dict[str, Any] = {}
    for handle, label in zip(handles, labels, strict=True):
        unique.setdefault(label, handle)
    return fig.legend(
        list(unique.values()),
        list(unique),
        loc="outside lower center",
        ncols=min(4, max(1, len(unique))),
        borderaxespad=0.4,
        columnspacing=1.5,
        handletextpad=0.6,
    )


def _overlap_area(left: mpl.transforms.Bbox, right: mpl.transforms.Bbox) -> float:
    x = max(0.0, min(left.x1, right.x1) - max(left.x0, right.x0))
    y = max(0.0, min(left.y1, right.y1) - max(left.y0, right.y0))
    return x * y


def validate_figure_layout(fig: Figure, axes: Sequence[Axes], legend: Legend | None) -> None:
    """Render once and fail if an external legend covers any data panel."""

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]
    boxes = [axis.get_window_extent(renderer) for axis in axes]
    for left_index, left in enumerate(boxes):
        for right in boxes[left_index + 1 :]:
            if _overlap_area(left, right) > 0.5:
                raise PlotError("constrained layout produced overlapping axes")
    if legend is not None:
        legend_box = legend.get_window_extent(renderer)
        if any(_overlap_area(legend_box, axis_box) > 0.5 for axis_box in boxes):
            raise PlotError("shared legend overlaps a data panel")


def _atomic_savefig(fig: Figure, target: Path, *, dpi: int) -> None:
    suffix = target.suffix.lower() or ".pdf"
    if suffix not in {".pdf", ".png", ".svg"}:
        raise PlotError("figure extension must be .pdf, .png, or .svg")
    target = target if target.suffix else target.with_suffix(suffix)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.stem}.", suffix=suffix, dir=target.parent)
    os.close(descriptor)
    try:
        fig.savefig(temporary, dpi=dpi, bbox_inches="tight", format=suffix.removeprefix("."))
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@dataclass(frozen=True)
class FigureArtifacts:
    figure: Path
    plotted_csv: Path
    metadata_json: Path
    font: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "figure": str(self.figure),
            "plotted_csv": str(self.plotted_csv),
            "metadata_json": str(self.metadata_json),
            "font": self.font,
        }


@dataclass(frozen=True)
class FigureSetArtifacts:
    """Per-design-cell figure exports and their machine-readable manifest."""

    output_dir: Path
    figures: Mapping[str, tuple[FigureArtifacts, ...]]
    manifest_json: Path

    def as_dict(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "manifest_json": str(self.manifest_json),
            "figures": {
                cell_id: [figure.as_dict() for figure in figures]
                for cell_id, figures in sorted(self.figures.items())
            },
        }


def _save_figure_artifacts(
    fig: Figure,
    plotted_data: pd.DataFrame,
    output: str | Path,
    *,
    dpi: int,
    metadata: Mapping[str, Any],
) -> FigureArtifacts:
    target = Path(output)
    if not target.suffix:
        target = target.with_suffix(".pdf")
    csv_path = target.with_name(f"{target.stem}.plot-data.csv")
    metadata_path = target.with_name(f"{target.stem}.plot-metadata.json")
    atomic_text(csv_path, plotted_data.to_csv(index=False, lineterminator="\n"))
    font = register_myriad_pro()
    write_json(
        metadata_path,
        {
            **dict(metadata),
            "figure": target.name,
            "plotted_csv": csv_path.name,
            "font": font,
            "plot_data_digest": stable_hash(plotted_data.to_dict(orient="records"), 64),
            "rows": len(plotted_data),
        },
    )
    _atomic_savefig(fig, target, dpi=dpi)
    return FigureArtifacts(target, csv_path, metadata_path, font)


def plot_control_trajectories(
    trajectory: pd.DataFrame,
    output: str | Path,
    *,
    group_columns: Sequence[str] = (),
    confidence: float = 0.95,
    bootstrap_draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    title: str | None = None,
    dpi: int = 240,
) -> FigureArtifacts:
    """Plot behavior and causal control for Y, P, and Q in three aligned panels."""

    plotted = control_trajectory_plot_data(
        trajectory,
        group_columns=group_columns,
        confidence=confidence,
        bootstrap_draws=bootstrap_draws,
    )
    with _publication_context():
        fig, axes_array = plt.subplots(
            1,
            3,
            figsize=(11.4, 3.65),
            sharex=True,
            sharey=True,
            layout="constrained",
        )
        axes = list(axes_array)
        label_panels(axes, [_CANDIDATE_TITLES[name] for name in ("y", "p", "q")])
        handles: list[Any] = []
        labels: list[str] = []
        grouping = [*group_columns, "measure"]
        marker_cycle = ("o", "^", "s", "D", "v", "P", "X")
        if group_columns:
            conditions = sorted(
                {tuple(row[column] for column in group_columns) for _, row in plotted.iterrows()},
                key=repr,
            )
            condition_markers = {
                condition: marker_cycle[index % len(marker_cycle)]
                for index, condition in enumerate(conditions)
            }
        else:
            condition_markers = {}
        for axis, candidate in zip(axes, ("Y", "P", "Q"), strict=True):
            candidate_rows = plotted[plotted["candidate"] == candidate]
            for keys, group in candidate_rows.groupby(grouping, dropna=False, sort=True):
                keys = keys if isinstance(keys, tuple) else (keys,)
                key_values = dict(zip(grouping, keys, strict=True))
                measure = str(key_values["measure"])
                style = _MEASURE_STYLE[measure]
                condition = tuple(key_values[column] for column in group_columns)
                marker = condition_markers[condition] if group_columns else style["marker"]
                ordered = group.sort_values("step")
                line = axis.plot(
                    ordered["step"],
                    ordered["mean"],
                    color=style["color"],
                    marker=marker,
                    linestyle=style["linestyle"],
                    alpha=0.95,
                )[0]
                axis.fill_between(
                    ordered["step"].to_numpy(dtype=float),
                    ordered["ci_low"].to_numpy(dtype=float),
                    ordered["ci_high"].to_numpy(dtype=float),
                    color=style["color"],
                    alpha=0.13,
                    linewidth=0,
                )
                handles.append(line)
                labels.append(_series_label(key_values, group_columns, measure))
            _style_axis(axis, candidate_rows["step"].astype(int).tolist())
        axes[0].set_ylabel("Agreement or causal flip rate")
        if title:
            fig.suptitle(title, fontsize=13, fontweight="semibold")
        legend = _external_legend(fig, handles, labels)
        validate_figure_layout(fig, axes, legend)
        result = _save_figure_artifacts(
            fig,
            plotted,
            output,
            dpi=dpi,
            metadata={
                "kind": "goal_control_trajectories",
                "confidence": confidence,
                "bootstrap_draws": bootstrap_draws,
                "group_columns": list(group_columns),
                "replication_unit": "training_seed",
            },
        )
        plt.close(fig)
        return result


def walsh_plot_data(
    coefficients: pd.DataFrame,
    *,
    group_columns: Sequence[str] = (),
    confidence: float = 0.95,
    bootstrap_draws: int = DEFAULT_BOOTSTRAP_DRAWS,
) -> pd.DataFrame:
    terms = ("y", "p", "q", "y_p", "y_q", "p_q", "y_p_q")
    required = {"seed", *[f"walsh_{term}" for term in terms], *group_columns}
    missing = sorted(required - set(coefficients.columns))
    if missing:
        raise PlotError(f"Walsh coefficient table is missing: {missing}")
    for column in ("cell_id", "step", "prompt_view"):
        if column in coefficients and column not in group_columns and coefficients[column].nunique() > 1:
            raise PlotError(
                f"coefficient table contains multiple {column} values; filter or group explicitly"
            )
    records: list[dict[str, Any]] = []
    groups: Any = (
        coefficients.groupby(list(group_columns), dropna=False, sort=True)
        if group_columns
        else [((), coefficients)]
    )
    for keys, group in groups:
        keys = keys if isinstance(keys, tuple) else (keys,)
        common = dict(zip(group_columns, keys, strict=True))
        for term in terms:
            per_seed = group.groupby("seed")[f"walsh_{term}"].mean().to_numpy(dtype=float)
            interval = bootstrap_mean_interval(
                per_seed,
                confidence=confidence,
                draws=bootstrap_draws,
                seed=_bootstrap_label([*keys, term, "walsh"]),
            )
            records.append(
                {
                    **common,
                    "term": term,
                    "mean": interval["estimate"],
                    "ci_low": interval["ci_low"],
                    "ci_high": interval["ci_high"],
                    "n_seeds": interval["n_seeds"],
                    "confidence": confidence,
                }
            )
    return pd.DataFrame(records)


def plot_walsh_coefficients(
    coefficients: pd.DataFrame,
    output: str | Path,
    *,
    confidence: float = 0.95,
    bootstrap_draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    title: str = "Conditional structure of the learned policy",
    dpi: int = 240,
) -> FigureArtifacts:
    """Plot seed-level saturated truth-table coefficients without pooling cells."""

    plotted = walsh_plot_data(
        coefficients,
        confidence=confidence,
        bootstrap_draws=bootstrap_draws,
    )
    labels = {
        "y": "Y",
        "p": "P",
        "q": "Q",
        "y_p": r"Y$\mathsf{\times}$P",
        "y_q": r"Y$\mathsf{\times}$Q",
        "p_q": r"P$\mathsf{\times}$Q",
        "y_p_q": r"Y$\mathsf{\times}$P$\mathsf{\times}$Q",
    }
    colors = ["#266FA5", "#C45B35", "#369873", "#8C6BB1", "#8C6BB1", "#8C6BB1", "#6F4E8D"]
    with _publication_context():
        fig, axis = plt.subplots(figsize=(7.4, 4.0), layout="constrained")
        x = np.arange(len(plotted))
        means = plotted["mean"].to_numpy(dtype=float)
        errors = np.vstack(
            [
                means - plotted["ci_low"].to_numpy(dtype=float),
                plotted["ci_high"].to_numpy(dtype=float) - means,
            ]
        )
        axis.bar(x, means, color=colors, width=0.72, zorder=2)
        axis.errorbar(x, means, yerr=errors, fmt="none", ecolor="#24272C", capsize=3, lw=1)
        axis.axhline(0.0, color="#535860", linewidth=0.8)
        axis.set_xticks(x)
        axis.set_xticklabels([labels[str(term)] for term in plotted["term"]])
        axis.set_ylabel("Walsh coefficient")
        axis.set_title(title, loc="left", pad=10)
        axis.grid(axis="y", zorder=0)
        axis.spines[["top", "right"]].set_visible(False)
        validate_figure_layout(fig, [axis], None)
        result = _save_figure_artifacts(
            fig,
            plotted,
            output,
            dpi=dpi,
            metadata={
                "kind": "walsh_coefficients",
                "confidence": confidence,
                "bootstrap_draws": bootstrap_draws,
                "replication_unit": "training_seed",
            },
        )
        plt.close(fig)
        return result


def _cell_stem(cell_id: str) -> str:
    readable = "".join(
        character if character.isalnum() or character in {"-", "_"} else "-" for character in cell_id
    ).strip("-")
    readable = readable[:32] or "cell"
    return f"{readable}-{stable_hash(cell_id, 8)}"


def export_analysis_figures(
    trajectory: pd.DataFrame,
    walsh: pd.DataFrame,
    output_dir: str | Path,
    *,
    behavior_panel: str = "conflict",
    prompt_view: str = "full",
    confidence: float = 0.95,
    bootstrap_draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    dpi: int = 240,
) -> FigureSetArtifacts:
    """Write trajectory and final truth-table figures separately for every cell."""

    required_trajectory = {"cell_id", "run_id", "seed", "step", "panel", "prompt_view"}
    missing = sorted(required_trajectory - set(trajectory.columns))
    if missing:
        raise PlotError(f"trajectory table is missing columns: {missing}")
    required_walsh = {"cell_id", "run_id", "seed", "step", "prompt_view"}
    missing = sorted(required_walsh - set(walsh.columns))
    if missing:
        raise PlotError(f"Walsh table is missing columns: {missing}")

    selected = trajectory[
        (trajectory["panel"] == behavior_panel) & (trajectory["prompt_view"] == prompt_view)
    ].copy()
    if selected.empty:
        raise PlotError(f"no trajectory rows match panel={behavior_panel!r}, view={prompt_view!r}")
    target = Path(output_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    exported: dict[str, tuple[FigureArtifacts, ...]] = {}
    manifest_cells: dict[str, Any] = {}
    for cell_id in sorted(selected["cell_id"].astype(str).unique()):
        cell_trajectory = selected[selected["cell_id"].astype(str) == cell_id].copy()
        cell_walsh = walsh[
            (walsh["cell_id"].astype(str) == cell_id) & (walsh["prompt_view"] == prompt_view)
        ].copy()
        if cell_walsh.empty:
            raise PlotError(f"cell {cell_id} has no Walsh coefficients for view {prompt_view}")
        final_indices = cell_walsh.groupby("run_id")["step"].idxmax()
        final_walsh = cell_walsh.loc[final_indices].copy()
        trajectory_seeds = set(cell_trajectory["seed"].tolist())
        walsh_seeds = set(final_walsh["seed"].tolist())
        if trajectory_seeds != walsh_seeds:
            raise PlotError(f"cell {cell_id} has unmatched trajectory and final-factorial seeds")
        stem = _cell_stem(cell_id)
        control = plot_control_trajectories(
            cell_trajectory,
            target / f"{stem}-control-trajectories.pdf",
            confidence=confidence,
            bootstrap_draws=bootstrap_draws,
            dpi=dpi,
        )
        truth_table = plot_walsh_coefficients(
            final_walsh,
            target / f"{stem}-final-walsh.pdf",
            confidence=confidence,
            bootstrap_draws=bootstrap_draws,
            dpi=dpi,
        )
        exported[cell_id] = (control, truth_table)
        manifest_cells[cell_id] = {
            "seeds": sorted(trajectory_seeds),
            "final_steps": sorted(set(final_walsh["step"].astype(int).tolist())),
            "figures": [control.as_dict(), truth_table.as_dict()],
        }
    manifest_path = target / "figure-manifest.json"
    write_json(
        manifest_path,
        {
            "behavior_panel": behavior_panel,
            "prompt_view": prompt_view,
            "confidence": confidence,
            "bootstrap_draws": bootstrap_draws,
            "replication_unit": "training_seed",
            "cells": manifest_cells,
        },
    )
    return FigureSetArtifacts(target, exported, manifest_path)


__all__ = [
    "FigureArtifacts",
    "FigureSetArtifacts",
    "PlotError",
    "control_trajectory_plot_data",
    "export_analysis_figures",
    "label_panels",
    "plot_control_trajectories",
    "plot_walsh_coefficients",
    "publication_style",
    "register_myriad_pro",
    "validate_figure_layout",
    "walsh_plot_data",
]
