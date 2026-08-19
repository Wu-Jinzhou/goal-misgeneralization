#!/usr/bin/env python3
"""Reproduce the figures for the Forkworld results essay.

The script reads completed run summaries (and the resolved H9 configurations
needed to recover width/depth labels).  All nine experiment families covered
by the essay are complete.  In particular, the temporal-noise figure uses the
full 28,900-run H6 panel, including all three sample sizes and the
terminal-reward equivalence control.

Run from anywhere with::

    .venv/bin/python paper/forkworld-current-results/analysis_and_plots.py

Plots are written as vector PDF and inspection PNG files.  Myriad Pro is loaded
from ``FORKWORLD_FONT_DIR`` or the current user's macOS font directory when it
is available; otherwise plotting continues with Matplotlib's bundled DejaVu
Sans fallback.
"""

from __future__ import annotations

import csv
import json
import math
import os
import warnings
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Iterator, Mapping, Sequence

# Keep matplotlib/fontconfig caches out of the read-only user cache exposed by
# the managed execution environment.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/forkworld-matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/forkworld-xdg-cache")

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ARTIFACTS = REPO / "artifacts-cpu10"
FIGURES = HERE / "figures"
DERIVED = HERE / "derived"

BLUE = "#2673B8"
ORANGE = "#D55E00"
TEAL = "#009E73"
PURPLE = "#7A5195"
GOLD = "#D4A72C"
SKY = "#56B4E9"
GRAY = "#667085"
LIGHT_GRAY = "#E6E9EE"
INK = "#17212B"


def plot_font_family() -> str:
    """Register Myriad Pro from a portable location, or use a bundled fallback."""

    font_dirs: list[Path] = []
    configured = os.environ.get("FORKWORLD_FONT_DIR")
    if configured:
        font_dirs.append(Path(configured).expanduser())
    home_fonts = Path.home() / "Library" / "Fonts"
    if home_fonts not in font_dirs:
        font_dirs.append(home_fonts)

    filenames = (
        "MYRIADPRO-REGULAR.OTF",
        "MYRIADPRO-SEMIBOLD.OTF",
        "MYRIADPRO-BOLD.OTF",
        "MyriadPro-Light.otf",
    )
    registered: list[str] = []
    for font_dir in font_dirs:
        for filename in filenames:
            path = font_dir / filename
            if path.is_file():
                fm.fontManager.addfont(str(path))
                registered.append(fm.FontProperties(fname=str(path)).get_name())
    if not registered:
        warnings.warn(
            "Myriad Pro was not found in FORKWORLD_FONT_DIR or ~/Library/Fonts; "
            "falling back to DejaVu Sans. Figure geometry may differ slightly.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "DejaVu Sans"
    return registered[0]


def configure_style() -> None:
    font_family = plot_font_family()

    mpl.rcParams.update(
        {
            "font.family": font_family,
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.titleweight": 600,
            "axes.labelsize": 9.5,
            "axes.labelcolor": INK,
            "axes.edgecolor": "#AAB2BD",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": "#4B5563",
            "ytick.color": "#4B5563",
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def get(mapping: Mapping[str, Any], path: str, default: Any = None) -> Any:
    value: Any = mapping
    for key in path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def summaries(hypothesis: str) -> Iterator[dict[str, Any]]:
    root = ARTIFACTS / hypothesis
    for path in sorted(root.glob("*/*/summary.json")):
        if not (path.parent / "COMPLETE").is_file():
            continue
        with path.open("r", encoding="utf-8") as handle:
            yield json.load(handle)


def grouped_mean(
    records: Iterable[Mapping[str, Any]],
    keys: Sequence[str],
    value: str,
) -> dict[tuple[Any, ...], float]:
    groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for record in records:
        raw = get(record, value)
        if raw is None:
            continue
        groups[tuple(get(record, key) for key in keys)].append(float(raw))
    return {key: mean(values) for key, values in groups.items()}


def save(figure: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / f"{stem}.pdf")
    figure.savefig(FIGURES / f"{stem}.png", dpi=220)
    plt.close(figure)


def panel_label(axis: plt.Axes, label: str, *, y_offset: float = 8) -> None:
    """Place a panel letter a fixed physical distance from the axes corner.

    Axes-relative offsets made labels on wide panels sit much farther left than
    labels on narrow panels.  Point offsets keep the visual gutter identical
    regardless of a panel's width or height.
    """

    axis.annotate(
        label,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(-14, y_offset),
        textcoords="offset points",
        fontsize=11,
        fontweight=600,
        color=INK,
        ha="right",
        va="bottom",
        annotation_clip=False,
        zorder=10,
    )


def kaplan_meier_cdf(
    times: Sequence[int], observed: Sequence[bool], horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return a right-continuous Kaplan--Meier event CDF."""

    rows = sorted(zip(times, observed, strict=True), key=lambda item: item[0])
    risk = len(rows)
    survival = 1.0
    xs = [1.0]
    ys = [0.0]
    for time in sorted({int(t) for t, event in rows if event}):
        events = sum(int(t == time and event) for t, event in rows)
        censored = sum(int(t == time and not event) for t, event in rows)
        if risk > 0:
            survival *= 1.0 - events / risk
        xs.append(max(float(time), 1.0))
        ys.append(1.0 - survival)
        risk -= events + censored
    xs.append(float(horizon))
    ys.append(1.0 - survival)
    return np.asarray(xs), np.asarray(ys)


def write_rows(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_mean_interval(
    values: Sequence[float], *, seed: int, samples: int = 4_000
) -> dict[str, float | int]:
    """Percentile interval after planned cells have been collapsed by seed."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError("values must be a non-empty one-dimensional sequence")
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(samples, len(array)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "estimate": float(array.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "n_seeds": int(len(array)),
    }


def seed_collapsed_interval(
    records: Sequence[Mapping[str, Any]],
    value: str | Any,
    *,
    bootstrap_seed: int,
) -> dict[str, float | int]:
    """Average planned cells within seed, then bootstrap the ten seeds."""

    by_seed: dict[int, list[float]] = defaultdict(list)
    for record in records:
        raw = value(record) if callable(value) else get(record, value)
        if raw is not None and math.isfinite(float(raw)):
            by_seed[int(get(record, "seed"))].append(float(raw))
    seed_values = [mean(by_seed[seed]) for seed in sorted(by_seed) if by_seed[seed]]
    return bootstrap_mean_interval(seed_values, seed=bootstrap_seed)


def figure_phase_and_dynamics() -> dict[str, Any]:
    h1 = list(summaries("h1"))
    h3 = list(summaries("h3"))

    phase = grouped_mean(
        h1,
        ("model.total_parameters", "data.realized_q", "data.k"),
        "final.conflict.rho_y",
    )
    capacities = [209, 1601, 18689]
    q_values = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
    k_values = [1, 2, 3, 4, 5]

    figure = plt.figure(figsize=(7.35, 6.45))
    grid = figure.add_gridspec(2, 3, height_ratios=[1.0, 0.85], hspace=0.48, wspace=0.30)
    figure.subplots_adjust(bottom=0.15, top=0.94, left=0.10, right=0.89)
    axes = [figure.add_subplot(grid[0, index]) for index in range(3)]
    image = None
    phase_rows: list[dict[str, Any]] = []
    for index, (axis, capacity) in enumerate(zip(axes, capacities, strict=True)):
        matrix = np.asarray(
            [[phase.get((capacity, q, k), np.nan) for q in q_values] for k in k_values]
        )
        image = axis.imshow(
            matrix,
            origin="lower",
            aspect="auto",
            vmin=0,
            vmax=1,
            cmap=mpl.colors.LinearSegmentedColormap.from_list(
                "proxy_to_goal", [ORANGE, "#F5F2EA", BLUE]
            ),
        )
        axis.set_xticks(range(len(q_values)), [f"{q:g}" for q in q_values], rotation=45)
        axis.set_yticks(range(len(k_values)), [str(k) for k in k_values])
        axis.set_xlabel("Proxy accuracy $q$")
        if index == 0:
            axis.set_ylabel("Exact-rule degree $k$")
            panel_label(axis, "a", y_offset=22)
        else:
            axis.tick_params(labelleft=False)
        width = {209: 8, 1601: 32, 18689: 128}[capacity]
        axis.set_title(f"Width {width}\n{capacity:,} parameters", pad=7)
        for row_index, k in enumerate(k_values):
            for column_index, q in enumerate(q_values):
                value = matrix[row_index, column_index]
                phase_rows.append(
                    {"parameters": capacity, "width": width, "q": q, "k": k, "mean_rho_y": value}
                )
                if q >= 0.95 and k >= 3:
                    axis.text(
                        column_index,
                        row_index,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7.0,
                        color="white" if value < 0.18 or value > 0.82 else INK,
                    )
    assert image is not None
    color_axis = figure.add_axes([0.915, 0.575, 0.012, 0.30])
    colorbar = figure.colorbar(image, cax=color_axis, orientation="vertical")
    colorbar.set_label("Conflict-set intended reliance, $\\rho_Y$")
    colorbar.set_ticks([0, 0.5, 1], labels=["proxy", "split", "intended"])

    dynamics = figure.add_subplot(grid[1, :])
    proxy_times = [int(get(row, "events.proxy_acquisition_time")) for row in h3]
    proxy_observed = [bool(get(row, "events.proxy_acquisition_observed")) for row in h3]
    intended_times = [int(get(row, "events.intended_acquisition_time")) for row in h3]
    intended_observed = [bool(get(row, "events.intended_acquisition_observed")) for row in h3]
    for times, observed, label, color in (
        (proxy_times, proxy_observed, "Proxy acquired", ORANGE),
        (intended_times, intended_observed, "Intended rule acquired", BLUE),
    ):
        x, y = kaplan_meier_cdf(times, observed, horizon=8192)
        dynamics.step(x, y, where="post", color=color, lw=2.2, label=label)
    dynamics.axvline(14, color=ORANGE, lw=1, alpha=0.45, ls="--")
    dynamics.axvline(279, color=BLUE, lw=1, alpha=0.45, ls="--")
    median_label_box = {"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 0.4}
    dynamics.text(
        14,
        0.60,
        "median 14",
        color=ORANGE,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox=median_label_box,
    )
    dynamics.text(
        279,
        0.60,
        "median 279",
        color=BLUE,
        ha="left",
        va="bottom",
        fontsize=8,
        bbox=median_label_box,
    )
    dynamics.set_xscale("log", base=2)
    dynamics.set_xlim(1, 8192)
    dynamics.set_ylim(0, 1.02)
    dynamics.set_xlabel("Optimizer step")
    dynamics.set_ylabel("Fraction acquired\n(Kaplan–Meier estimate)")
    dynamics.grid(axis="y", color=LIGHT_GRAY, lw=0.7)
    dynamics.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)
    dynamics.set_title("The proxy turns on long before the exact rule")
    panel_label(dynamics, "b")

    write_rows(
        DERIVED / "h1_selected_phase_cells.csv",
        ("parameters", "width", "q", "k", "mean_rho_y"),
        phase_rows,
    )
    save(figure, "fig1_phase_and_dynamics")
    return {"h1_runs": len(h1), "h3_runs": len(h3)}


def figure_counterevidence() -> dict[str, Any]:
    h3 = list(summaries("h3"))
    h4 = list(summaries("h4"))
    figure = plt.figure(figsize=(7.35, 7.55))
    outer = figure.add_gridspec(2, 1, height_ratios=[1.0, 0.98], hspace=0.24)
    top = outer[0].subgridspec(
        2,
        2,
        height_ratios=[1.0, 0.22],
        hspace=0.52,
        wspace=0.34,
    )
    bottom = outer[1].subgridspec(
        2,
        1,
        height_ratios=[1.0, 0.20],
        hspace=0.45,
    )
    figure.subplots_adjust(left=0.12, right=0.985, bottom=0.045, top=0.96)
    axis_a = figure.add_subplot(top[0, 0])
    axis_b = figure.add_subplot(top[0, 1])
    legend_a = figure.add_subplot(top[1, 0])
    legend_b = figure.add_subplot(top[1, 1])
    axis_c = figure.add_subplot(bottom[0, 0])
    legend_c = figure.add_subplot(bottom[1, 0])
    for legend_axis in (legend_a, legend_b, legend_c):
        legend_axis.set_axis_off()

    acquisition: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in h3:
        k = int(get(row, "data.k"))
        if k == 1:
            continue
        count = int(get(row, "data.n_conflict_train"))
        acquisition[(k, count)].append(float(bool(get(row, "events.intended_acquisition_observed"))))
    k_colors = {2: SKY, 3: TEAL, 4: PURPLE, 5: ORANGE}
    acquisition_rows: list[dict[str, Any]] = []
    for k in [2, 3, 4, 5]:
        counts = sorted(count for kk, count in acquisition if kk == k)
        values = [mean(acquisition[(k, count)]) for count in counts]
        axis_a.plot(counts, values, marker="o", ms=4, lw=1.8, color=k_colors[k], label=f"$k={k}$")
        acquisition_rows.extend(
            {"k": k, "unique_conflicts": count, "acquisition_fraction": value}
            for count, value in zip(counts, values, strict=True)
        )
    axis_a.set_xscale("log", base=2)
    axis_a.set_ylim(-0.02, 1.02)
    axis_a.set_xlabel("Distinct conflict examples in training")
    axis_a.set_ylabel("Exact rule acquired by step 8,192")
    axis_a.grid(axis="y", color=LIGHT_GRAY, lw=0.7)
    handles_a, labels_a = axis_a.get_legend_handles_labels()
    legend_a.legend(
        handles_a,
        labels_a,
        ncol=4,
        loc="center",
        fontsize=8.0,
        handlelength=1.6,
        columnspacing=1.0,
    )
    axis_a.set_title("Acquisition by conflict count", loc="left", pad=8)
    panel_label(axis_a, "a")

    condition_style = {
        ("concentrated", 0.02): ("2% unique", ORANGE, "o"),
        ("concentrated", 0.1): ("10% unique", GOLD, "o"),
        ("diverse", 0.5): ("50% unique", SKY, "o"),
        ("diverse", 1.0): ("100% unique", BLUE, "o"),
        ("structured_holdout", 1.0): ("new failure types", TEAL, "s"),
    }
    diversity: dict[tuple[str, float, int], list[float]] = defaultdict(list)
    for row in h4:
        condition = str(get(row, "condition"))
        fraction = float(get(row, "data.configured_unique_fraction"))
        count = int(get(row, "data.n_conflict"))
        diversity[(condition, fraction, count)].append(float(get(row, "final.conflict_unseen.rho_y")))
    counts = [0, 8, 32, 128, 512, 2048]
    diversity_rows: list[dict[str, Any]] = []
    for key, (label, color, marker) in condition_style.items():
        values = [mean(diversity[(key[0], key[1], count)]) for count in counts]
        axis_b.plot(range(len(counts)), values, marker=marker, ms=4, lw=1.8, color=color, label=label)
        diversity_rows.extend(
            {"condition": label, "n_conflict": count, "mean_unseen_rho_y": value}
            for count, value in zip(counts, values, strict=True)
        )
    axis_b.axhline(0.9, color=GRAY, ls="--", lw=1)
    axis_b.set_xticks(range(len(counts)), [f"{count:,}" for count in counts], rotation=35)
    axis_b.set_ylim(-0.02, 1.02)
    axis_b.set_xlabel("Total conflict presentations")
    axis_b.set_ylabel("Reliance on $Y$ for unseen conflicts")
    axis_b.grid(axis="y", color=LIGHT_GRAY, lw=0.7)
    handles_b, labels_b = axis_b.get_legend_handles_labels()
    legend_b.legend(
        handles_b,
        labels_b,
        fontsize=7.1,
        ncol=3,
        loc="center",
        handlelength=1.5,
        columnspacing=0.8,
        labelspacing=0.45,
    )
    axis_b.set_title("Generalization by count and diversity", loc="left", pad=8)
    panel_label(axis_b, "b")

    selected = [
        row
        for row in h4
        if str(get(row, "condition")) == "concentrated"
        and math.isclose(float(get(row, "data.configured_unique_fraction")), 0.02)
        and int(get(row, "data.n_conflict")) == 2048
    ]
    seen: dict[int, list[float]] = defaultdict(list)
    unseen: dict[int, list[float]] = defaultdict(list)
    for row in selected:
        k = int(get(row, "data.k"))
        seen[k].append(float(get(row, "final.conflict_seen_repeated.rho_y")))
        unseen[k].append(float(get(row, "final.conflict_unseen.rho_y")))
    for y, k in enumerate([1, 2, 3, 4, 5]):
        left, right = mean(unseen[k]), mean(seen[k])
        axis_c.plot([left, right], [y, y], color="#C8CDD4", lw=2.2, zorder=1)
        offset = 0.07 if abs(left - right) < 0.04 else 0.0
        axis_c.scatter(left, y - offset, color=ORANGE, s=38, zorder=2)
        axis_c.scatter(right, y + offset, color=BLUE, s=38, zorder=2)
    axis_c.scatter([], [], color=ORANGE, s=38, label="Unseen conflicts")
    axis_c.scatter([], [], color=BLUE, s=38, label="Repeated training conflicts")
    axis_c.set_yticks(range(5), [f"$k={k}$" for k in [1, 2, 3, 4, 5]])
    axis_c.invert_yaxis()
    axis_c.set_xlim(-0.04, 1.04)
    axis_c.set_xlabel("Intended-goal reliance, $\\rho_Y$")
    axis_c.grid(axis="x", color=LIGHT_GRAY, lw=0.7)
    handles_c, labels_c = axis_c.get_legend_handles_labels()
    legend_c.legend(
        handles_c,
        labels_c,
        ncol=2,
        loc="center",
        handletextpad=0.65,
        columnspacing=1.6,
    )
    axis_c.set_title(
        "Reliance on repeated versus unseen conflicts\n"
        "2,048 presentations from only 41 contexts",
        loc="left",
        pad=7,
    )
    panel_label(axis_c, "c", y_offset=22)

    write_rows(
        DERIVED / "h3_acquisition_by_unique_conflicts.csv",
        ("k", "unique_conflicts", "acquisition_fraction"),
        acquisition_rows,
    )
    write_rows(
        DERIVED / "h4_diversity_curves.csv",
        ("condition", "n_conflict", "mean_unseen_rho_y"),
        diversity_rows,
    )
    save(figure, "fig2_counterevidence")
    return {"h4_runs": len(h4), "h4_memorization_subset_runs": len(selected)}


def figure_decoding_vs_control() -> dict[str, Any]:
    rows = []
    for summary in summaries("h2"):
        rows.append(
            {
                "mode": str(get(summary, "model.competition.update_mode")),
                "k": int(get(summary, "data.k")),
                "total": int(get(summary, "model.competition.total_parameters")),
                "trainable": int(get(summary, "model.competition.trainable_parameters")),
                "decoder": float(get(summary, "final.exact_decoder_accuracy")),
                "behavior": float(get(summary, "final.competition_conflict.rho_y")),
                "crossed": bool(get(summary, "final.decoder_threshold_reached")),
            }
        )

    figure = plt.figure(figsize=(7.35, 6.25))
    grid = figure.add_gridspec(
        3,
        2,
        height_ratios=[1.0, 0.16, 0.95],
        hspace=0.28,
        wspace=0.34,
    )
    axis_a = figure.add_subplot(grid[0, :])
    legend_axis = figure.add_subplot(grid[1, :])
    legend_axis.set_axis_off()
    axis_b = figure.add_subplot(grid[2, 0])
    axis_c = figure.add_subplot(grid[2, 1])

    subspace = [row for row in rows if row["mode"] == "subspace" and row["k"] >= 2]
    aggregates: dict[tuple[int, int], dict[str, list[float]]] = defaultdict(lambda: {"decoder": [], "behavior": []})
    for row in subspace:
        cell = aggregates[(row["k"], row["trainable"])]
        cell["decoder"].append(row["decoder"])
        cell["behavior"].append(row["behavior"])
    k_colors = {2: SKY, 3: TEAL, 4: PURPLE, 5: ORANGE}
    for k in [2, 3, 4, 5]:
        budgets = sorted(budget for kk, budget in aggregates if kk == k)
        decoder = [mean(aggregates[(k, budget)]["decoder"]) for budget in budgets]
        behavior = [mean(aggregates[(k, budget)]["behavior"]) for budget in budgets]
        axis_a.plot(budgets, behavior, color=k_colors[k], lw=2, marker="o", ms=4)
        axis_a.plot(budgets, decoder, color=k_colors[k], lw=1.4, ls="--", alpha=0.85)
    axis_a.set_xscale("log", base=2)
    axis_a.set_ylim(-0.03, 1.03)
    axis_a.axhline(0.95, color=GRAY, lw=0.9, ls=":")
    axis_a.set_xlabel("Trainable subspace parameters")
    axis_a.set_ylabel("Accuracy / intended-goal reliance")
    axis_a.grid(axis="y", color=LIGHT_GRAY, lw=0.7)
    axis_a.set_title("Exact decoding precedes behavioral control", loc="left", pad=8)
    legend_handles = [
        Line2D([0], [0], color=k_colors[k], lw=2, marker="o", ms=4, label=f"$k={k}$")
        for k in [2, 3, 4, 5]
    ]
    legend_handles.extend(
        [
            Line2D([0], [0], color=INK, lw=2, marker="o", ms=4, label="behavior"),
            Line2D([0], [0], color=INK, lw=1.5, ls="--", label="standalone decoder"),
        ]
    )
    legend_axis.legend(handles=legend_handles, ncol=6, fontsize=7.8, loc="center")
    panel_label(axis_a, "a")

    scatter_modes = [("head_only", "Head-only updates", PURPLE, axis_b), ("subspace", "Random-subspace updates", TEAL, axis_c)]
    rng = np.random.default_rng(20260728)
    for mode, title, color, axis in scatter_modes:
        part = [row for row in rows if row["mode"] == mode]
        x = np.asarray([row["decoder"] for row in part])
        y = np.asarray([row["behavior"] for row in part])
        # Deterministic micro-jitter makes coincident seed points visible without
        # changing any substantive coordinate.
        jitter_x = np.clip(x + rng.normal(0, 0.006, len(x)), 0, 1)
        jitter_y = np.clip(y + rng.normal(0, 0.006, len(y)), 0, 1)
        axis.scatter(jitter_x, jitter_y, s=7, color=color, alpha=0.12, linewidths=0, rasterized=True)
        axis.plot([0, 1], [0, 1], color=GRAY, lw=1, ls="--")
        axis.fill_between([0.95, 1], [0, 0], [0.5, 0.5], color=ORANGE, alpha=0.07)
        axis.set_xlim(-0.03, 1.03)
        axis.set_ylim(-0.03, 1.03)
        axis.set_xlabel("Standalone exact-decoder accuracy")
        axis.set_ylabel("Competition $\\rho_Y$")
        axis.grid(color=LIGHT_GRAY, lw=0.55)
        axis.set_title(title, loc="left", pad=6)
    panel_label(axis_b, "b")
    panel_label(axis_c, "c")

    write_rows(
        DERIVED / "h2_decoder_behavior_points.csv",
        ("mode", "k", "total", "trainable", "decoder", "behavior", "crossed"),
        rows,
    )
    save(figure, "fig3_decoding_vs_control")
    crossed_counts = {
        mode: {
            "crossed": sum(row["crossed"] for row in rows if row["mode"] == mode),
            "not_crossed": sum(not row["crossed"] for row in rows if row["mode"] == mode),
        }
        for mode in ["full", "head_only", "subspace"]
    }
    return {"h2_runs": len(rows), "decoder_threshold_counts": crossed_counts}


def budget_number(value: Any, full_parameters: int) -> int:
    return full_parameters if str(value) == "full" else int(value)


def figure_update_capacity() -> dict[str, Any]:
    rows = []
    for summary in summaries("h5"):
        full = int(get(summary, "model.total_parameters"))
        rows.append(
            {
                "algorithm": str(get(summary, "algorithm")),
                "entropy": int(get(summary, "nuisance.requested_total_entropy")),
                "budget": budget_number(get(summary, "model.requested_budget"), full),
                "requested_budget": get(summary, "model.requested_budget"),
                "rho_y": float(get(summary, "final.rho_y")),
                "train_rho_y": float(get(summary, "train.rho_y")),
            }
        )
    labels = {
        "clean_sft": "Clean SFT",
        "trajectory_sft": "Trajectory SFT",
        "on_policy_imitation": "On-policy imitation",
        "rl": "RL",
    }
    colors = {
        "clean_sft": BLUE,
        "trajectory_sft": TEAL,
        "on_policy_imitation": PURPLE,
        "rl": ORANGE,
    }

    figure = plt.figure(figsize=(7.35, 4.15))
    grid = figure.add_gridspec(2, 2, height_ratios=[1.0, 0.12], hspace=0.36, wspace=0.24)
    axes = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])]
    axes[1].sharex(axes[0])
    axes[1].sharey(axes[0])
    legend_axis = figure.add_subplot(grid[1, :])
    legend_axis.set_axis_off()
    rng = np.random.default_rng(5)
    for index, (axis, entropy) in enumerate(zip(axes, [0, 8], strict=True)):
        for algorithm in labels:
            part = [row for row in rows if row["algorithm"] == algorithm and row["entropy"] == entropy]
            groups: dict[int, list[float]] = defaultdict(list)
            for row in part:
                groups[row["budget"]].append(row["rho_y"])
            budgets = sorted(groups)
            values = [mean(groups[budget]) for budget in budgets]
            axis.plot(budgets, values, color=colors[algorithm], marker="o", ms=4.5, lw=2, label=labels[algorithm])
            for budget in budgets:
                points = groups[budget]
                jitter = 2 ** rng.normal(0, 0.025, len(points))
                axis.scatter(np.asarray([budget] * len(points)) * jitter, points, s=7, color=colors[algorithm], alpha=0.18, linewidth=0)
        axis.axhline(0.9, color=GRAY, lw=1, ls="--")
        axis.set_xscale("log", base=2)
        axis.set_ylim(-0.04, 1.04)
        axis.grid(axis="y", color=LIGHT_GRAY, lw=0.7)
        axis.set_xlabel("Trainable actor parameters")
        axis.set_title(f"Nuisance entropy $H={entropy}$ bits", loc="left", pad=7)
        panel_label(axis, "ab"[index])
    axes[0].set_ylabel("Conflict-set intended reliance, $\\rho_Y$")
    tick_values = [1, 4, 16, 64, 256, 1024, 5897]
    for axis in axes:
        axis.set_xticks(tick_values, ["1", "4", "16", "64", "256", "1,024", "5,897"], rotation=35)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    legend_axis.legend(handles, legend_labels, loc="center", ncol=4, fontsize=8.2)
    save(figure, "fig4_update_capacity")
    write_rows(
        DERIVED / "h5_update_capacity_points.csv",
        ("algorithm", "entropy", "budget", "requested_budget", "rho_y", "train_rho_y"),
        rows,
    )
    return {"h5_runs": len(rows)}


def h6_plot_rows() -> list[dict[str, Any]]:
    """Normalize the H6 summary fields needed for matched-seed plotting."""

    structure_names = {
        "step_resampled": "step",
        "episode_static": "episode",
        "state_static": "state",
    }
    rows: list[dict[str, Any]] = []
    for record in summaries("h6"):
        presentations = int(get(record, "training.realized_presentations"))
        epochs = int(get(record, "training.requested_epochs"))
        requested_structure = str(get(record, "noise.requested_structure"))
        rows.append(
            {
                "seed": int(get(record, "seed")),
                "algorithm": str(get(record, "algorithm")),
                "structure": structure_names.get(requested_structure, requested_structure),
                "location": str(get(record, "noise.location")),
                "scale": float(get(record, "noise.scale")),
                "visits": int(get(record, "recurrence.realized_episode_visits_per_state")),
                "n_train": int(presentations // epochs),
                "presentations": presentations,
                "reward_mode": str(get(record, "recurrence.reward_mode")),
                "objective_uses_location": bool(
                    get(record, "noise.objective_uses_location")
                ),
                "rho_y": float(get(record, "final.rho_y")),
            }
        )
    return rows


def h6_matched_structure_contrast(
    rows: Sequence[Mapping[str, Any]],
    left: str,
    right: str,
    *,
    location: str = "observation",
    algorithm: str | None = None,
    bootstrap_seed: int,
) -> dict[str, float | int]:
    """Match all planned cells, average contrasts within seed, then resample seeds."""

    matched: dict[tuple[Any, ...], dict[str, float]] = defaultdict(dict)
    for row in rows:
        if (
            row["reward_mode"] != "dense_fixed_horizon"
            or not row["objective_uses_location"]
            or row["location"] != location
            or float(row["scale"]) <= 0
            or (algorithm is not None and row["algorithm"] != algorithm)
        ):
            continue
        key = (
            row["seed"],
            row["algorithm"],
            row["scale"],
            row["presentations"],
            row["visits"],
        )
        matched[key][str(row["structure"])] = float(row["rho_y"])
    by_seed: dict[int, list[float]] = defaultdict(list)
    for key, values in matched.items():
        if left in values and right in values:
            by_seed[int(key[0])].append(values[left] - values[right])
    seed_values = [mean(by_seed[seed]) for seed in sorted(by_seed)]
    return bootstrap_mean_interval(seed_values, seed=bootstrap_seed)


def h6_matched_noise_gaps(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return positive-noise rho_Y minus its exact scale-zero matched control."""

    eligible = [
        row
        for row in rows
        if row["reward_mode"] == "dense_fixed_horizon"
        and row["location"] == "observation"
        and row["objective_uses_location"]
    ]
    match_fields = ("seed", "algorithm", "structure", "presentations", "visits")
    zero: dict[tuple[Any, ...], float] = {}
    for row in eligible:
        if math.isclose(float(row["scale"]), 0.0):
            zero[tuple(row[field] for field in match_fields)] = float(row["rho_y"])
    gaps: list[dict[str, Any]] = []
    for row in eligible:
        if float(row["scale"]) <= 0:
            continue
        key = tuple(row[field] for field in match_fields)
        if key not in zero:
            raise ValueError(f"H6 positive-noise cell lacks a scale-zero match: {key}")
        gaps.append({**row, "gap": float(row["rho_y"]) - zero[key]})
    return gaps


def h6_gap_curve(
    gaps: Sequence[Mapping[str, Any]], algorithm: str, structure: str, *, seed: int
) -> list[dict[str, Any]]:
    """Summarize the plotted gap at each presentation count using ten seeds."""

    cells: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in gaps:
        if row["algorithm"] == algorithm and row["structure"] == structure:
            cells[(int(row["presentations"]), int(row["seed"]))].append(
                float(row["gap"])
            )
    records: list[dict[str, Any]] = []
    for index, presentations in enumerate(sorted({key[0] for key in cells})):
        seed_values = [
            mean(cells[(presentations, training_seed)])
            for training_seed in sorted(
                key[1] for key in cells if key[0] == presentations
            )
        ]
        summary = bootstrap_mean_interval(seed_values, seed=seed + index)
        records.append(
            {
                "algorithm": algorithm,
                "structure": structure,
                "presentations": presentations,
                **summary,
            }
        )
    return records


def h6_sample_scaling_slopes(
    gaps: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Reproduce the registered seed-level slopes behind the scaling verdict."""

    cells: dict[tuple[Any, ...], list[tuple[float, float]]] = defaultdict(list)
    for row in gaps:
        key = (
            int(row["seed"]),
            str(row["algorithm"]),
            str(row["structure"]),
            float(row["scale"]),
            int(row["visits"]),
        )
        cells[key].append((float(row["presentations"]), float(row["gap"])))

    slopes: dict[tuple[Any, ...], dict[str, float]] = {}
    for key, points in cells.items():
        averaged: dict[float, list[float]] = defaultdict(list)
        for presentations, gap in points:
            averaged[presentations].append(gap)
        ordered = sorted(
            (presentations, mean(values))
            for presentations, values in averaged.items()
        )
        if len(ordered) != 3:
            raise ValueError(f"Expected three H6 presentation levels for {key}")
        x = np.log2(np.asarray([point[0] for point in ordered], dtype=float))
        signed = np.asarray([point[1] for point in ordered], dtype=float)
        centered = x - x.mean()
        denominator = float(np.dot(centered, centered))
        slopes[key] = {
            "signed_gap": float(np.dot(centered, signed - signed.mean()) / denominator),
            "gap_to_zero": float(
                np.dot(centered, -np.abs(signed) - (-np.abs(signed)).mean())
                / denominator
            ),
        }

    output: list[dict[str, Any]] = []
    algorithm_groups: list[str | None] = [
        None,
        "clean_sft",
        "trajectory_sft",
        "on_policy_imitation",
        "rl",
    ]
    for algorithm in algorithm_groups:
        selected = {
            key: value
            for key, value in slopes.items()
            if algorithm is None or key[1] == algorithm
        }
        for metric in ("signed_gap", "gap_to_zero"):
            for structure in ("step", "state"):
                by_seed: dict[int, list[float]] = defaultdict(list)
                for key, values in selected.items():
                    if key[2] == structure:
                        by_seed[int(key[0])].append(values[metric])
                summary = bootstrap_mean_interval(
                    [mean(by_seed[seed]) for seed in sorted(by_seed)],
                    seed=(
                        {
                            ("signed_gap", "step"): 1631,
                            ("signed_gap", "state"): 1637,
                            ("gap_to_zero", "step"): 1621,
                            ("gap_to_zero", "state"): 1627,
                        }[(metric, structure)]
                        if algorithm is None
                        else {
                            ("signed_gap", "step"): 1663,
                            ("signed_gap", "state"): 1693,
                            ("gap_to_zero", "step"): 1667,
                            ("gap_to_zero", "state"): 1697,
                        }[(metric, structure)]
                    ),
                )
                output.append(
                    {
                        "algorithm": algorithm or "all",
                        "comparison": structure,
                        "metric": metric,
                        **summary,
                    }
                )

            differences: dict[int, list[float]] = defaultdict(list)
            base_keys = {
                (key[0], key[1], key[3], key[4])
                for key in selected
                if key[2] in {"step", "state"}
            }
            for seed_value, algorithm_value, scale_value, visits_value in base_keys:
                step_key = (
                    seed_value,
                    algorithm_value,
                    "step",
                    scale_value,
                    visits_value,
                )
                state_key = (
                    seed_value,
                    algorithm_value,
                    "state",
                    scale_value,
                    visits_value,
                )
                if step_key in selected and state_key in selected:
                    differences[int(seed_value)].append(
                        selected[step_key][metric] - selected[state_key][metric]
                    )
            summary = bootstrap_mean_interval(
                [mean(differences[seed]) for seed in sorted(differences)],
                seed=(
                    {"signed_gap": 1657, "gap_to_zero": 1643}[metric]
                    if algorithm is None
                    else {"signed_gap": 1669, "gap_to_zero": 1679}[metric]
                ),
            )
            output.append(
                {
                    "algorithm": algorithm or "all",
                    "comparison": "step_minus_state",
                    "metric": metric,
                    **summary,
                }
            )
    return output


def h6_terminal_equivalence(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, float | int | bool]:
    """Average the five terminal-control scales within each training seed."""

    matched: dict[tuple[int, float], dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row["reward_mode"] != "terminal":
            continue
        key = (int(row["seed"]), float(row["scale"]))
        matched[key][str(row["structure"])] = float(row["rho_y"])
    by_seed: dict[int, list[float]] = defaultdict(list)
    for (training_seed, _), values in matched.items():
        if "step" in values and "episode" in values:
            by_seed[training_seed].append(values["step"] - values["episode"])
    seed_values = [mean(by_seed[seed]) for seed in sorted(by_seed)]
    result: dict[str, float | int | bool] = bootstrap_mean_interval(
        seed_values, seed=1607
    )
    result["equivalent_within_margin"] = bool(
        float(result["ci_low"]) >= -0.05 and float(result["ci_high"]) <= 0.05
    )
    return result


def figure_temporal_noise() -> dict[str, Any]:
    rows = h6_plot_rows()
    if len(rows) != 28_900:
        raise ValueError(f"Expected the complete 28,900-run H6 panel, found {len(rows)}")

    contrast_specs = [
        ("All · step − episode", "step", "episode", None, "overall"),
        ("All · episode − state", "episode", "state", None, "overall"),
        ("All · step − state", "step", "state", None, "overall"),
        ("Clean SFT · step − state", "step", "state", "clean_sft", "algorithm"),
        (
            "Trajectory SFT · step − state",
            "step",
            "state",
            "trajectory_sft",
            "algorithm",
        ),
        (
            "On-policy imitation · step − state",
            "step",
            "state",
            "on_policy_imitation",
            "algorithm",
        ),
        ("RL · step − state", "step", "state", "rl", "algorithm"),
    ]
    contrast_rows: list[dict[str, Any]] = []
    for label, left, right, algorithm, group in contrast_specs:
        summary = h6_matched_structure_contrast(
            rows,
            left,
            right,
            algorithm=algorithm,
            bootstrap_seed=1601 if algorithm is None else 1613,
        )
        contrast_rows.append(
            {
                "contrast": label,
                "left": left,
                "right": right,
                "algorithm": algorithm or "all",
                "group": group,
                **summary,
            }
        )

    gaps = h6_matched_noise_gaps(rows)
    slope_rows = h6_sample_scaling_slopes(gaps)
    curve_rows: list[dict[str, Any]] = []
    for algorithm_index, algorithm in enumerate(("clean_sft", "rl")):
        for structure_index, structure in enumerate(("step", "episode", "state")):
            curve_rows.extend(
                h6_gap_curve(
                    gaps,
                    algorithm,
                    structure,
                    seed=1701 + 20 * algorithm_index + 4 * structure_index,
                )
            )

    figure = plt.figure(figsize=(7.35, 6.35))
    grid = figure.add_gridspec(
        3,
        2,
        height_ratios=[1.25, 1.0, 0.12],
        hspace=0.54,
        wspace=0.26,
    )
    figure.subplots_adjust(left=0.235, right=0.985, bottom=0.08, top=0.95)

    forest = figure.add_subplot(grid[0, :])
    positions = np.asarray([7.0, 6.0, 5.0, 3.4, 2.4, 1.4, 0.4])
    forest.axvspan(-0.05, 0.05, color=GRAY, alpha=0.09)
    forest.axvline(0, color=INK, lw=1)
    for position, row in zip(positions, contrast_rows, strict=True):
        color = BLUE if row["group"] == "overall" else TEAL
        if row["algorithm"] == "rl":
            color = ORANGE
        forest.plot(
            [row["ci_low"], row["ci_high"]],
            [position, position],
            color=color,
            lw=2.1,
        )
        forest.scatter(row["estimate"], position, color=color, s=36, zorder=3)
        forest.text(
            float(row["ci_high"]) + 0.004,
            position,
            f"{float(row['estimate']):+.3f}",
            va="center",
            fontsize=7.8,
            color=color,
        )
    forest.axhline(4.2, color=LIGHT_GRAY, lw=1)
    forest.set_yticks(positions, [row["contrast"] for row in contrast_rows])
    forest.set_xlim(-0.057, 0.238)
    forest.set_ylim(-0.2, 7.7)
    forest.set_xlabel("Difference in intended-goal reliance, $\\Delta\\rho_Y$")
    forest.grid(axis="x", color=LIGHT_GRAY, lw=0.7)
    forest.set_title("Persistent observation noise preserves proxy control", loc="left")
    panel_label(forest, "a")

    axes = [figure.add_subplot(grid[1, 0]), figure.add_subplot(grid[1, 1])]
    structure_labels = {
        "step": "Step-resampled",
        "episode": "Episode-static",
        "state": "State-static",
    }
    structure_colors = {"step": BLUE, "episode": TEAL, "state": ORANGE}
    for axis_index, (axis, algorithm, title) in enumerate(
        zip(axes, ("clean_sft", "rl"), ("Clean SFT", "RL"), strict=True)
    ):
        for structure in ("step", "episode", "state"):
            selected = [
                row
                for row in curve_rows
                if row["algorithm"] == algorithm and row["structure"] == structure
            ]
            x = np.asarray([row["presentations"] for row in selected], dtype=float)
            estimate = np.asarray([row["estimate"] for row in selected], dtype=float)
            low = np.asarray([row["ci_low"] for row in selected], dtype=float)
            high = np.asarray([row["ci_high"] for row in selected], dtype=float)
            color = structure_colors[structure]
            axis.plot(
                x,
                estimate,
                color=color,
                marker="o",
                ms=4.5,
                lw=2,
                label=structure_labels[structure],
            )
            axis.fill_between(x, low, high, color=color, alpha=0.13, linewidth=0)
        axis.axhline(0, color=INK, lw=1)
        axis.set_xscale("log", base=2)
        axis.set_xlim(17_500, 383_000)
        axis.set_ylim(-0.035, 0.72)
        axis.set_xticks(
            [20_480, 81_920, 327_680],
            ["20,480", "81,920", "327,680"],
            rotation=24,
        )
        axis.set_xlabel("Training presentations")
        axis.grid(axis="y", color=LIGHT_GRAY, lw=0.7)
        axis.set_title(title, loc="left")
        panel_label(axis, "bc"[axis_index])
    axes[0].set_ylabel("Noisy − matched clean reliance, $\\Delta\\rho_Y$")
    axes[1].tick_params(labelleft=False)

    legend_axis = figure.add_subplot(grid[2, :])
    legend_axis.set_axis_off()
    handles, labels = axes[0].get_legend_handles_labels()
    legend_axis.legend(handles, labels, loc="center", ncol=3, fontsize=8.4)

    save(figure, "fig5_temporal_noise")
    write_rows(
        DERIVED / "h6_temporal_contrasts.csv",
        (
            "contrast",
            "left",
            "right",
            "algorithm",
            "group",
            "estimate",
            "ci_low",
            "ci_high",
            "n_seeds",
        ),
        contrast_rows,
    )
    write_rows(
        DERIVED / "h6_sample_scaling.csv",
        (
            "algorithm",
            "structure",
            "presentations",
            "estimate",
            "ci_low",
            "ci_high",
            "n_seeds",
        ),
        curve_rows,
    )
    write_rows(
        DERIVED / "h6_sample_scaling_slopes.csv",
        (
            "algorithm",
            "comparison",
            "metric",
            "estimate",
            "ci_low",
            "ci_high",
            "n_seeds",
        ),
        slope_rows,
    )

    location_rows: list[dict[str, Any]] = []
    for location, algorithms in (
        ("observation", None),
        ("label", None),
        ("reward", "rl"),
    ):
        for left, right in (("step", "episode"), ("episode", "state"), ("step", "state")):
            summary = h6_matched_structure_contrast(
                rows,
                left,
                right,
                location=location,
                algorithm=algorithms,
                bootstrap_seed=1801,
            )
            location_rows.append(
                {
                    "location": location,
                    "contrast": f"{left}_minus_{right}",
                    **summary,
                }
            )
    write_rows(
        DERIVED / "h6_active_location_contrasts.csv",
        ("location", "contrast", "estimate", "ci_low", "ci_high", "n_seeds"),
        location_rows,
    )

    terminal = h6_terminal_equivalence(rows)
    write_rows(
        DERIVED / "h6_terminal_reward_control.csv",
        ("contrast", "estimate", "ci_low", "ci_high", "n_seeds", "equivalent_within_margin"),
        ({"contrast": "step_minus_episode", **terminal},),
    )
    objective_active = sum(bool(row["objective_uses_location"]) for row in rows)
    return {
        "status": "complete",
        "runs": len(rows),
        "dense_factorial_runs": sum(
            row["reward_mode"] == "dense_fixed_horizon" for row in rows
        ),
        "terminal_control_runs": sum(row["reward_mode"] == "terminal" for row in rows),
        "objective_active_runs": objective_active,
        "objective_ignored_negative_controls": len(rows) - objective_active,
        "n_train_values": [2_560, 10_240, 40_960],
        "independent_seeds": 10,
        "terminal_reward_equivalence": terminal,
    }


def figure_unlearning() -> dict[str, Any]:
    """Summarize behavioral suppression and the causal inverse-proxy failure."""

    all_rows = list(summaries("h7"))
    primary = [
        row
        for row in all_rows
        if math.isclose(float(get(row, "training.weight_decay")), 0.0)
        and bool(get(row, "final.eligible_for_primary_analysis"))
    ]
    if len(all_rows) != 1_920 or len(primary) != 840:
        raise ValueError(
            f"Expected 1,920 H7 runs and 840 primary runs; found {len(all_rows)} and {len(primary)}"
        )

    half_life_specs = [
        ("Removed", GRAY, lambda row: get(row, "condition") == "removal"),
        ("Decorrelated", TEAL, lambda row: get(row, "condition") == "decorrelation"),
        (
            "Reversed\n$q_B=.40$",
            "#E9A17B",
            lambda row: get(row, "condition") == "reversal"
            and math.isclose(float(get(row, "data.requested_q_b")), 0.4),
        ),
        (
            "Reversed\n$q_B=.25$",
            "#DF7B52",
            lambda row: get(row, "condition") == "reversal"
            and math.isclose(float(get(row, "data.requested_q_b")), 0.25),
        ),
        (
            "Reversed\n$q_B=.10$",
            ORANGE,
            lambda row: get(row, "condition") == "reversal"
            and math.isclose(float(get(row, "data.requested_q_b")), 0.1),
        ),
        (
            "Perfectly\nreversed",
            "#9F3D14",
            lambda row: get(row, "condition") == "reversal"
            and math.isclose(float(get(row, "data.requested_q_b")), 0.0),
        ),
        ("Replaced", GOLD, lambda row: get(row, "condition") == "replacement"),
    ]
    half_life_rows: list[dict[str, Any]] = []
    for index, (label, color, selector) in enumerate(half_life_specs):
        selected = [row for row in primary if selector(row)]
        interval = seed_collapsed_interval(
            selected,
            "events.literal_half_life.time",
            bootstrap_seed=7100 + index,
        )
        half_life_rows.append(
            {
                "label": label.replace("\n", " ").replace("$", ""),
                "color": color,
                "n_runs": len(selected),
                "censored_runs": sum(
                    not bool(get(row, "events.literal_half_life.observed"))
                    for row in selected
                ),
                **interval,
            }
        )

    figure = plt.figure(figsize=(7.35, 6.05))
    grid = figure.add_gridspec(
        3,
        2,
        height_ratios=[1.0, 1.0, 0.12],
        hspace=0.64,
        wspace=0.36,
    )
    figure.subplots_adjust(left=0.105, right=0.985, bottom=0.055, top=0.955)
    axis_a = figure.add_subplot(grid[0, :])
    axis_b = figure.add_subplot(grid[1, 0])
    axis_c = figure.add_subplot(grid[1, 1])
    legend_axis = figure.add_subplot(grid[2, :])
    legend_axis.set_axis_off()

    x = np.arange(len(half_life_rows))
    for position, row in zip(x, half_life_rows, strict=True):
        estimate = float(row["estimate"])
        low = float(row["ci_low"])
        high = float(row["ci_high"])
        axis_a.plot([position, position], [low, high], color=row["color"], lw=2.2)
        axis_a.scatter(position, estimate, color=row["color"], s=48, zorder=3)
        axis_a.text(
            position,
            high * 1.055,
            f"{estimate:.1f}",
            color=row["color"],
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axis_a.set_yscale("log", base=2)
    axis_a.set_ylim(12, 180)
    axis_a.set_yticks([16, 32, 64, 128], ["16", "32", "64", "128"])
    axis_a.set_xticks(x, [spec[0] for spec in half_life_specs])
    axis_a.set_ylabel("Restricted half-life (updates)")
    axis_a.grid(axis="y", color=LIGHT_GRAY, lw=0.7)
    axis_a.set_title(
        "Negative evidence suppresses old proxy-following faster than absence",
        loc="left",
        pad=8,
    )
    panel_label(axis_a, "a")

    persistence_conditions = [
        ("Removed", "removal", GRAY),
        ("Decorrelated", "decorrelation", TEAL),
        ("Replaced", "replacement", GOLD),
    ]
    rng = np.random.default_rng(7199)
    persistence_rows: list[dict[str, Any]] = []
    for position, (_label, condition, color) in enumerate(persistence_conditions):
        selected = [row for row in primary if get(row, "condition") == condition]
        values = np.asarray(
            [float(get(row, "final.restoration_rho_p")) for row in selected]
        )
        jitter = rng.uniform(-0.18, 0.18, size=len(values))
        axis_b.scatter(
            position + jitter,
            values,
            s=10,
            color=color,
            alpha=0.22,
            linewidths=0,
            rasterized=True,
        )
        interval = seed_collapsed_interval(
            selected,
            "final.restoration_rho_p",
            bootstrap_seed=7200 + position,
        )
        axis_b.plot(
            [position, position],
            [interval["ci_low"], interval["ci_high"]],
            color=INK,
            lw=2.0,
        )
        axis_b.scatter(position, interval["estimate"], color=INK, s=31, zorder=4)
        persistence_rows.append(
            {"condition": condition, "n_runs": len(selected), **interval}
        )
    axis_b.set_xticks(range(3), [item[0] for item in persistence_conditions], rotation=18)
    axis_b.set_ylim(-0.015, 0.53)
    axis_b.set_ylabel("Old-proxy reliance when restored, $\\rho_P$")
    axis_b.grid(axis="y", color=LIGHT_GRAY, lw=0.7)
    axis_b.set_title("Residual failure after 1,024 updates", loc="left", pad=8)
    panel_label(axis_b, "b")

    causal_groups: list[tuple[str, list[dict[str, Any]]]] = [
        (
            ".50\nneutral",
            [row for row in primary if get(row, "condition") == "decorrelation"],
        )
    ]
    for q_b in (0.4, 0.25, 0.1, 0.0):
        causal_groups.append(
            (
                f"{q_b:g}" + ("\nperfect" if math.isclose(q_b, 0.0) else ""),
                [
                    row
                    for row in primary
                    if get(row, "condition") == "reversal"
                    and math.isclose(float(get(row, "data.requested_q_b")), q_b)
                ],
            )
        )
    causal_rows: list[dict[str, Any]] = []
    causal_styles = [
        ("Flip old proxy $P$", "final.interventions.flip_P.hard_flip_rate", ORANGE, "o"),
        (
            "Flip exact-code channel $R_i$",
            "final.interventions.flip_R_mean.hard_flip_rate",
            BLUE,
            "s",
        ),
    ]
    for style_index, (label, path, color, marker) in enumerate(causal_styles):
        estimates: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        for group_index, (group_label, selected) in enumerate(causal_groups):
            interval = seed_collapsed_interval(
                selected,
                path,
                bootstrap_seed=7300 + 20 * style_index + group_index,
            )
            estimates.append(float(interval["estimate"]))
            lows.append(float(interval["ci_low"]))
            highs.append(float(interval["ci_high"]))
            causal_rows.append(
                {
                    "q_b": group_label.replace("\n", " "),
                    "intervention": label,
                    **interval,
                }
            )
        positions = np.arange(len(causal_groups))
        axis_c.errorbar(
            positions,
            estimates,
            yerr=[np.asarray(estimates) - np.asarray(lows), np.asarray(highs) - np.asarray(estimates)],
            color=color,
            marker=marker,
            ms=5,
            lw=1.9,
            capsize=2.2,
            label=label,
        )
    axis_c.annotate(
        "inverse proxy\n(probability effect $=-1.00$)",
        xy=(4, 1),
        xytext=(3.0, 0.72),
        arrowprops={"arrowstyle": "-", "color": ORANGE, "lw": 0.9},
        color=ORANGE,
        fontsize=7.8,
        ha="center",
    )
    axis_c.set_xticks(range(len(causal_groups)), [item[0] for item in causal_groups])
    axis_c.set_ylim(-0.03, 1.04)
    axis_c.set_xlabel("Phase-B proxy accuracy $q_B$")
    axis_c.set_ylabel("Fraction of decisions changed by flip")
    axis_c.grid(axis="y", color=LIGHT_GRAY, lw=0.7)
    axis_c.set_title("Perfect reversal changes the proxy's sign", loc="left", pad=8)
    panel_label(axis_c, "c")
    handles, labels = axis_c.get_legend_handles_labels()
    legend_axis.legend(handles, labels, loc="center", ncol=2, fontsize=8.3)

    write_rows(
        DERIVED / "h7_unlearning_half_lives.csv",
        (
            "label",
            "color",
            "n_runs",
            "censored_runs",
            "estimate",
            "ci_low",
            "ci_high",
            "n_seeds",
        ),
        half_life_rows,
    )
    write_rows(
        DERIVED / "h7_restoration_persistence.csv",
        ("condition", "n_runs", "estimate", "ci_low", "ci_high", "n_seeds"),
        persistence_rows,
    )
    write_rows(
        DERIVED / "h7_causal_flip_rates.csv",
        ("q_b", "intervention", "estimate", "ci_low", "ci_high", "n_seeds"),
        causal_rows,
    )
    save(figure, "fig6_unlearning")
    return {
        "runs": len(all_rows),
        "primary_runs": len(primary),
        "weight_decay_ablation_runs": len(all_rows) - len(primary),
        "phase_a_gate_failures": sum(
            not bool(get(row, "final.phase_a_gate_passed")) for row in all_rows
        ),
    }


def h8_direct_pairs(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return eligible, cell-matched old-history minus compute-sham outcomes."""

    eligible = [row for row in rows if bool(get(row, "final.eligible_for_primary_analysis"))]

    def key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            int(get(row, "seed")),
            str(get(row, "data.stage1_mode")),
            str(get(row, "data.perturbation.arm")),
            int(get(row, "data.requested_n0")),
            int(get(row, "model.total_parameters")),
        )

    old = {key(row): row for row in eligible if get(row, "data.history") == "old_goal"}
    sham = {
        key(row): row
        for row in eligible
        if get(row, "data.history") == "compute_matched_sham"
    }
    pairs: list[dict[str, Any]] = []
    for pair_key in sorted(set(old) & set(sham)):
        left, right = old[pair_key], sham[pair_key]
        pairs.append(
            {
                "seed": pair_key[0],
                "mode": pair_key[1],
                "perturbation": pair_key[2],
                "n0": pair_key[3],
                "parameters": pair_key[4],
                "rebound_difference": float(get(left, "final.rebound_g0"))
                - float(get(right, "final.rebound_g0")),
                "reactivation_time_difference": float(
                    get(left, "events.reactivation.time")
                )
                - float(get(right, "events.reactivation.time")),
                "alignment_examples_difference": float(
                    get(left, "training.stage1_examples")
                )
                - float(get(right, "training.stage1_examples")),
            }
        )
    return pairs


def figure_hysteresis() -> dict[str, Any]:
    """Show why raw rebound is not, on its own, evidence of old-goal memory."""

    all_rows = list(summaries("h8"))
    if len(all_rows) != 2_880:
        raise ValueError(f"Expected the complete 2,880-run H8 panel, found {len(all_rows)}")
    eligible = [row for row in all_rows if bool(get(row, "final.eligible_for_primary_analysis"))]
    pairs = h8_direct_pairs(all_rows)

    perturbations = [
        ("Neutral", "neutral"),
        ("Intended signal\nremoved", "intended_removal"),
        ("75% old-goal\nlabels", "partial_reversal"),
        ("55% old-goal\nlabels", "weak_conflict"),
    ]
    histories = [
        ("Old-goal history", "old_goal", ORANGE, "o"),
        ("Compute-matched sham", "compute_matched_sham", TEAL, "s"),
        ("No-history control", "control", BLUE, "D"),
    ]

    figure = plt.figure(figsize=(7.35, 6.55))
    grid = figure.add_gridspec(
        4,
        2,
        height_ratios=[1.0, 0.12, 1.0, 0.16],
        hspace=0.54,
        wspace=0.33,
    )
    figure.subplots_adjust(left=0.11, right=0.985, bottom=0.045, top=0.96)
    axis_a = figure.add_subplot(grid[0, :])
    legend_a = figure.add_subplot(grid[1, :])
    axis_b = figure.add_subplot(grid[2, 0])
    axis_c = figure.add_subplot(grid[2, 1])
    legend_c = figure.add_subplot(grid[3, :])
    legend_a.set_axis_off()
    legend_c.set_axis_off()

    outcome_rows: list[dict[str, Any]] = []
    offsets = np.asarray([-0.18, 0.0, 0.18])
    for history_index, (history_label, history, color, marker) in enumerate(histories):
        estimates: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        for perturbation_index, (_, perturbation) in enumerate(perturbations):
            selected = [
                row
                for row in eligible
                if get(row, "data.stage1_mode") == "behavior_matched"
                and get(row, "data.history") == history
                and get(row, "data.perturbation.arm") == perturbation
            ]
            interval = seed_collapsed_interval(
                selected,
                "final.rho_g0_after_perturbation",
                bootstrap_seed=8100 + 20 * history_index + perturbation_index,
            )
            estimates.append(float(interval["estimate"]))
            lows.append(float(interval["ci_low"]))
            highs.append(float(interval["ci_high"]))
            outcome_rows.append(
                {
                    "history": history,
                    "perturbation": perturbation,
                    "n_runs": len(selected),
                    **interval,
                }
            )
        positions = np.arange(len(perturbations)) + offsets[history_index]
        axis_a.errorbar(
            positions,
            estimates,
            yerr=[np.asarray(estimates) - np.asarray(lows), np.asarray(highs) - np.asarray(estimates)],
            color=color,
            marker=marker,
            ms=5,
            lw=1.7,
            capsize=2.0,
            label=history_label,
        )
    axis_a.set_xticks(range(len(perturbations)), [item[0] for item in perturbations])
    axis_a.set_ylim(0.5, 1.025)
    axis_a.set_ylabel("Final old-goal reliance")
    axis_a.grid(axis="y", color=LIGHT_GRAY, lw=0.7)
    axis_a.set_title(
        "Ambiguous retraining returns to the simple proxy under every history",
        loc="left",
        pad=8,
    )
    panel_label(axis_a, "a")
    handles, labels = axis_a.get_legend_handles_labels()
    legend_a.legend(handles, labels, loc="center", ncol=3, fontsize=8.3)

    matched_pairs = [row for row in pairs if row["mode"] == "behavior_matched"]
    n0_values = [512, 2_048, 8_192, 32_768]
    alignment_rows: list[dict[str, Any]] = []
    estimates = []
    lows = []
    highs = []
    for index, n0 in enumerate(n0_values):
        selected = [row for row in matched_pairs if int(row["n0"]) == n0]
        interval = seed_collapsed_interval(
            selected,
            lambda row: row["alignment_examples_difference"],
            bootstrap_seed=8200 + index,
        )
        estimates.append(float(interval["estimate"]))
        lows.append(float(interval["ci_low"]))
        highs.append(float(interval["ci_high"]))
        unique_pairs = {
            (int(row["seed"]), int(row["n0"]), int(row["parameters"]))
            for row in selected
        }
        alignment_rows.append({"n0": n0, "n_pairs": len(unique_pairs), **interval})
    x_n0 = np.asarray(n0_values, dtype=float)
    axis_b.errorbar(
        x_n0,
        estimates,
        yerr=[np.asarray(estimates) - np.asarray(lows), np.asarray(highs) - np.asarray(estimates)],
        color=PURPLE,
        marker="o",
        ms=5,
        lw=2,
        capsize=2.2,
    )
    axis_b.axhline(0, color=INK, lw=1)
    axis_b.set_xscale("log", base=2)
    axis_b.set_xticks(x_n0, ["512", "2,048", "8,192", "32,768"], rotation=24)
    axis_b.set_xlabel("Old-goal training examples, $N_0$")
    axis_b.set_ylabel("Extra intended-goal examples\n(old history $-$ sham)")
    axis_b.grid(axis="y", color=LIGHT_GRAY, lw=0.7)
    axis_b.set_title("History remains visible in alignment cost", loc="left", pad=8)
    panel_label(axis_b, "b")

    perturbation_styles = {
        "neutral": ("Neutral", BLUE, "o"),
        "intended_removal": ("Intended signal removed", TEAL, "s"),
        "partial_reversal": ("75% old-goal labels", PURPLE, "D"),
        "weak_conflict": ("55% old-goal labels", ORANGE, "^") ,
    }
    rebound_rows: list[dict[str, Any]] = []
    for perturbation_index, (perturbation, (label, color, marker)) in enumerate(
        perturbation_styles.items()
    ):
        estimates = []
        lows = []
        highs = []
        for n0_index, n0 in enumerate(n0_values):
            selected = [
                row
                for row in matched_pairs
                if row["perturbation"] == perturbation and int(row["n0"]) == n0
            ]
            interval = seed_collapsed_interval(
                selected,
                lambda row: row["rebound_difference"],
                bootstrap_seed=8300 + 20 * perturbation_index + n0_index,
            )
            estimates.append(float(interval["estimate"]))
            lows.append(float(interval["ci_low"]))
            highs.append(float(interval["ci_high"]))
            rebound_rows.append(
                {
                    "perturbation": perturbation,
                    "n0": n0,
                    "n_pairs": len(selected),
                    **interval,
                }
            )
        axis_c.plot(x_n0, estimates, color=color, marker=marker, ms=4.5, lw=1.8, label=label)
        axis_c.fill_between(x_n0, lows, highs, color=color, alpha=0.10, linewidth=0)
    axis_c.axhspan(-0.05, 0.05, color=GRAY, alpha=0.08)
    axis_c.axhline(0, color=INK, lw=1)
    axis_c.set_xscale("log", base=2)
    axis_c.set_xticks(x_n0, ["512", "2,048", "8,192", "32,768"], rotation=24)
    axis_c.set_ylim(-0.018, 0.072)
    axis_c.set_xlabel("Old-goal training examples, $N_0$")
    axis_c.set_ylabel("Rebound difference\n(old history $-$ sham)")
    axis_c.grid(axis="y", color=LIGHT_GRAY, lw=0.7)
    axis_c.set_title("Average rebound remains small", loc="left", pad=8)
    panel_label(axis_c, "c")
    handles, labels = axis_c.get_legend_handles_labels()
    legend_c.legend(
        handles,
        labels,
        loc="center",
        ncol=4,
        fontsize=7.5,
        handlelength=1.5,
        columnspacing=1.0,
    )

    write_rows(
        DERIVED / "h8_behavior_matched_outcomes.csv",
        ("history", "perturbation", "n_runs", "estimate", "ci_low", "ci_high", "n_seeds"),
        outcome_rows,
    )
    write_rows(
        DERIVED / "h8_alignment_cost.csv",
        ("n0", "n_pairs", "estimate", "ci_low", "ci_high", "n_seeds"),
        alignment_rows,
    )
    write_rows(
        DERIVED / "h8_rebound_history_contrasts.csv",
        ("perturbation", "n0", "n_pairs", "estimate", "ci_low", "ci_high", "n_seeds"),
        rebound_rows,
    )
    save(figure, "fig7_hysteresis")
    return {
        "runs": len(all_rows),
        "eligible_runs": len(eligible),
        "excluded_preperturbation_cells": (len(all_rows) - len(eligible)) // 4,
        "matched_old_sham_pairs": len(pairs),
    }


def h9_plot_rows() -> list[dict[str, Any]]:
    """Load H9 summaries and attach architecture labels from resolved configs."""

    rows: list[dict[str, Any]] = []
    root = ARTIFACTS / "h9"
    for path in sorted(root.glob("*/*/summary.json")):
        if not (path.parent / "COMPLETE").is_file():
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        config = yaml.safe_load((path.parent / "resolved_config.yaml").read_text(encoding="utf-8"))
        rows.append(
            {
                "seed": int(get(summary, "seed")),
                "mode": str(get(summary, "model.update_mode")),
                "width": int(get(config, "model.width")),
                "depth": int(get(config, "model.depth")),
                "budget": budget_number(
                    get(summary, "model.requested_budget"),
                    int(get(summary, "model.total_parameters")),
                ),
                "total_parameters": int(get(summary, "model.total_parameters")),
                "calibration_passed": bool(get(summary, "final.single_rule_controls_passed")),
                "selector_passed": bool(get(summary, "final.selector_gate_passed")),
                "weaker_single_rule": min(
                    float(get(summary, "final.single_rule_P0.target_accuracy")),
                    float(get(summary, "final.single_rule_P1.target_accuracy")),
                ),
                "normal": float(get(summary, "final.normal.target_accuracy")),
                "removed": float(get(summary, "final.removed.target_accuracy")),
                "randomized": float(get(summary, "final.randomized.target_accuracy")),
                "mismatch": 0.5
                * (
                    float(get(summary, "final.mismatch_0_1.target_accuracy"))
                    + float(get(summary, "final.mismatch_1_0.target_accuracy"))
                ),
                "strict_switching": float(get(summary, "final.strict_context_switching")),
                "flip_c0_p0": float(
                    get(summary, "final.proxy_sensitivity_matrix.C0.P0.hard_flip_rate")
                ),
                "flip_c0_p1": float(
                    get(summary, "final.proxy_sensitivity_matrix.C0.P1.hard_flip_rate")
                ),
                "flip_c1_p0": float(
                    get(summary, "final.proxy_sensitivity_matrix.C1.P0.hard_flip_rate")
                ),
                "flip_c1_p1": float(
                    get(summary, "final.proxy_sensitivity_matrix.C1.P1.hard_flip_rate")
                ),
            }
        )
    return rows


def figure_multiplicity_capacity() -> dict[str, Any]:
    """Show architecture reliability and the cost of learning the selector."""

    rows = h9_plot_rows()
    if len(rows) != 270:
        raise ValueError(f"Expected the complete 270-run H9 panel, found {len(rows)}")
    architecture = [row for row in rows if row["mode"] == "full"]
    subspace = [row for row in rows if row["mode"] == "subspace"]
    widths = [2, 4, 8, 16, 32, 64, 128]
    depths = [1, 2, 4]

    calibration = np.zeros((len(depths), len(widths)))
    selector = np.zeros_like(calibration)
    architecture_rows: list[dict[str, Any]] = []
    for depth_index, depth in enumerate(depths):
        for width_index, width in enumerate(widths):
            selected = [
                row
                for row in architecture
                if int(row["depth"]) == depth and int(row["width"]) == width
            ]
            calibration[depth_index, width_index] = mean(
                float(row["calibration_passed"]) for row in selected
            )
            selector[depth_index, width_index] = mean(
                float(row["selector_passed"]) for row in selected
            )
            architecture_rows.append(
                {
                    "depth": depth,
                    "width": width,
                    "parameters": selected[0]["total_parameters"],
                    "calibration_pass_fraction": calibration[depth_index, width_index],
                    "selector_pass_fraction": selector[depth_index, width_index],
                    "mean_strict_switching": mean(row["strict_switching"] for row in selected),
                }
            )

    figure = plt.figure(figsize=(7.35, 6.1))
    grid = figure.add_gridspec(
        3,
        2,
        height_ratios=[0.82, 1.0, 0.13],
        hspace=0.55,
        wspace=0.28,
    )
    figure.subplots_adjust(left=0.10, right=0.985, bottom=0.055, top=0.955)
    axes = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "pass_rate", ["#F4F5F7", SKY, BLUE]
    )
    for axis, matrix, title, label in zip(
        axes,
        (calibration, selector),
        ("Can learn each rule alone", "Can switch rules on context"),
        ("a", "b"),
        strict=True,
    ):
        axis.imshow(matrix, vmin=0, vmax=1, cmap=cmap, aspect="auto")
        axis.set_xticks(range(len(widths)), [str(width) for width in widths])
        axis.set_yticks(range(len(depths)), [str(depth) for depth in depths])
        axis.set_xlabel("Width")
        axis.set_ylabel("Depth")
        axis.set_title(title, loc="left", pad=8)
        for row_index in range(len(depths)):
            for column_index in range(len(widths)):
                value = matrix[row_index, column_index]
                axis.text(
                    column_index,
                    row_index,
                    f"{round(10 * value):d}/10",
                    ha="center",
                    va="center",
                    fontsize=8.0,
                    color="white" if value >= 0.78 else INK,
                )
        panel_label(axis, label)

    axis_c = figure.add_subplot(grid[1, :])
    legend_axis = figure.add_subplot(grid[2, :])
    legend_axis.set_axis_off()
    budget_values = [1, 4, 16, 64, 256, 1024]
    composition_metrics = [
        ("Weaker single-rule calibration", "weaker_single_rule", GRAY, "^"),
        ("Normal-context accuracy", "normal", TEAL, "s"),
        ("Strict context switching", "strict_switching", BLUE, "o"),
    ]
    composition_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(9101)
    for metric_index, (label, key, color, marker) in enumerate(composition_metrics):
        estimates = []
        lows = []
        highs = []
        for budget_index, budget in enumerate(budget_values):
            selected = [row for row in subspace if int(row["budget"]) == budget]
            interval = seed_collapsed_interval(
                selected,
                lambda row, metric=key: row[metric],
                bootstrap_seed=9100 + 20 * metric_index + budget_index,
            )
            estimates.append(float(interval["estimate"]))
            lows.append(float(interval["ci_low"]))
            highs.append(float(interval["ci_high"]))
            composition_rows.append({"budget": budget, "metric": key, **interval})
            values = np.asarray([float(row[key]) for row in selected])
            jitter = 2 ** rng.uniform(-0.055, 0.055, len(values))
            axis_c.scatter(
                np.full(len(values), budget) * jitter,
                values,
                s=10,
                color=color,
                alpha=0.18,
                linewidths=0,
            )
        axis_c.plot(
            budget_values,
            estimates,
            color=color,
            marker=marker,
            ms=5,
            lw=2,
            label=label,
        )
        axis_c.fill_between(budget_values, lows, highs, color=color, alpha=0.10, linewidth=0)
    axis_c.axhline(0.9, color=INK, lw=1, ls="--")
    axis_c.axvspan(0.7, 32, color=GRAY, alpha=0.055)
    axis_c.text(4, 0.12, "single-rule controls fail", ha="center", color=GRAY, fontsize=8)
    axis_c.set_xscale("log", base=2)
    axis_c.set_xlim(0.7, 1_450)
    axis_c.set_ylim(-0.02, 1.025)
    axis_c.set_xticks(budget_values, ["1", "4", "16", "64", "256", "1,024"])
    axis_c.set_xlabel("Trainable subspace scalars")
    axis_c.set_ylabel("Accuracy / strict switching")
    axis_c.grid(axis="y", color=LIGHT_GRAY, lw=0.7)
    axis_c.set_title(
        "Learning either rule is cheaper than composing both with a gate",
        loc="left",
        pad=8,
    )
    panel_label(axis_c, "c")
    handles, labels = axis_c.get_legend_handles_labels()
    legend_axis.legend(handles, labels, loc="center", ncol=3, fontsize=8.1)

    write_rows(
        DERIVED / "h9_architecture_reliability.csv",
        (
            "depth",
            "width",
            "parameters",
            "calibration_pass_fraction",
            "selector_pass_fraction",
            "mean_strict_switching",
        ),
        architecture_rows,
    )
    write_rows(
        DERIVED / "h9_composition_gap.csv",
        ("budget", "metric", "estimate", "ci_low", "ci_high", "n_seeds"),
        composition_rows,
    )
    save(figure, "fig8_multiplicity_capacity")
    return {
        "runs": len(rows),
        "architecture_runs": len(architecture),
        "subspace_runs": len(subspace),
        "single_rule_control_failures": sum(
            not bool(row["calibration_passed"]) for row in rows
        ),
    }


def figure_context_control() -> dict[str, Any]:
    """Show OOD context obedience and the representation-free causal gate."""

    rows = h9_plot_rows()
    subspace = [row for row in rows if row["mode"] == "subspace"]
    budget_values = [1, 4, 16, 64, 256, 1024]
    context_metrics = [
        ("Normal", "normal", BLUE, "o"),
        ("Context removed", "removed", GRAY, "s"),
        ("Context randomized", "randomized", TEAL, "D"),
        ("Context forced wrong", "mismatch", ORANGE, "^"),
    ]

    figure = plt.figure(figsize=(7.35, 3.65))
    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 0.15],
        width_ratios=[1.55, 0.85],
        hspace=0.40,
        wspace=0.37,
    )
    figure.subplots_adjust(left=0.10, right=0.985, bottom=0.09, top=0.93)
    axis_a = figure.add_subplot(grid[0, 0])
    axis_b = figure.add_subplot(grid[0, 1])
    legend_axis = figure.add_subplot(grid[1, :])
    legend_axis.set_axis_off()

    context_rows: list[dict[str, Any]] = []
    for metric_index, (label, key, color, marker) in enumerate(context_metrics):
        estimates = []
        lows = []
        highs = []
        for budget_index, budget in enumerate(budget_values):
            selected = [row for row in subspace if int(row["budget"]) == budget]
            interval = seed_collapsed_interval(
                selected,
                lambda row, metric=key: row[metric],
                bootstrap_seed=9200 + 20 * metric_index + budget_index,
            )
            estimates.append(float(interval["estimate"]))
            lows.append(float(interval["ci_low"]))
            highs.append(float(interval["ci_high"]))
            context_rows.append({"budget": budget, "condition": key, **interval})
        axis_a.plot(
            budget_values,
            estimates,
            color=color,
            marker=marker,
            ms=4.5,
            lw=1.9,
            label=label,
        )
        axis_a.fill_between(budget_values, lows, highs, color=color, alpha=0.09, linewidth=0)
    axis_a.axhline(0.75, color=INK, lw=0.9, ls="--")
    axis_a.text(1.1, 0.765, "one-rule ceiling", fontsize=7.6, color=INK, va="bottom")
    axis_a.set_xscale("log", base=2)
    axis_a.set_xlim(0.7, 1_450)
    axis_a.set_ylim(0.45, 1.025)
    axis_a.set_xticks(budget_values, ["1", "4", "16", "64", "256", "1,024"])
    axis_a.set_xlabel("Trainable subspace scalars")
    axis_a.set_ylabel("Target accuracy")
    axis_a.grid(axis="y", color=LIGHT_GRAY, lw=0.7)
    axis_a.set_title("A reliable selector is vulnerable to a false context", loc="left", pad=8)
    panel_label(axis_a, "a")

    highest = [row for row in subspace if int(row["budget"]) == 1_024]
    matrix = np.asarray(
        [
            [mean(row["flip_c0_p0"] for row in highest), mean(row["flip_c0_p1"] for row in highest)],
            [mean(row["flip_c1_p0"] for row in highest), mean(row["flip_c1_p1"] for row in highest)],
        ]
    )
    axis_b.imshow(
        matrix,
        vmin=0,
        vmax=1,
        cmap=mpl.colors.LinearSegmentedColormap.from_list(
            "causal_gate", ["#F4F5F7", "#B8D8ED", BLUE]
        ),
        aspect="equal",
    )
    axis_b.set_xticks([0, 1], ["Flip $P_0$", "Flip $P_1$"])
    axis_b.set_yticks([0, 1], ["Clamp $C=0$", "Clamp $C=1$"])
    for row_index in range(2):
        for column_index in range(2):
            value = matrix[row_index, column_index]
            axis_b.text(
                column_index,
                row_index,
                f"{value:.2f}",
                color="white" if value > 0.65 else INK,
                ha="center",
                va="center",
                fontsize=10,
                fontweight=600,
            )
    axis_b.set_title("Context selects the causal proxy", loc="left", pad=8)
    axis_b.set_xlabel("Fraction of decisions changed")
    panel_label(axis_b, "b")

    handles, labels = axis_a.get_legend_handles_labels()
    legend_axis.legend(
        handles,
        labels,
        loc="center",
        ncol=4,
        fontsize=7.8,
        handlelength=1.5,
        columnspacing=1.1,
    )
    write_rows(
        DERIVED / "h9_context_conditions.csv",
        ("budget", "condition", "estimate", "ci_low", "ci_high", "n_seeds"),
        context_rows,
    )
    write_rows(
        DERIVED / "h9_causal_gate_matrix.csv",
        ("context", "flipped_proxy", "hard_flip_rate"),
        (
            {"context": 0, "flipped_proxy": 0, "hard_flip_rate": matrix[0, 0]},
            {"context": 0, "flipped_proxy": 1, "hard_flip_rate": matrix[0, 1]},
            {"context": 1, "flipped_proxy": 0, "hard_flip_rate": matrix[1, 0]},
            {"context": 1, "flipped_proxy": 1, "hard_flip_rate": matrix[1, 1]},
        ),
    )
    save(figure, "fig9_context_control")
    return {"highest_update_budget_runs": len(highest), "causal_gate_matrix": matrix.tolist()}


def main() -> None:
    configure_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    DERIVED.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "artifact_root": str(ARTIFACTS),
        "scope": "nine complete experiment families; 50,290 runs",
        "figures": {},
        "confirmatory_inference_unit": "independent training seed",
    }
    report["figures"]["phase_and_dynamics"] = figure_phase_and_dynamics()
    report["figures"]["counterevidence"] = figure_counterevidence()
    report["figures"]["decoding_vs_control"] = figure_decoding_vs_control()
    report["figures"]["update_capacity"] = figure_update_capacity()
    report["figures"]["temporal_noise"] = figure_temporal_noise()
    report["figures"]["unlearning"] = figure_unlearning()
    report["figures"]["hysteresis"] = figure_hysteresis()
    report["figures"]["multiplicity_capacity"] = figure_multiplicity_capacity()
    report["figures"]["context_control"] = figure_context_control()
    (DERIVED / "figure_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
