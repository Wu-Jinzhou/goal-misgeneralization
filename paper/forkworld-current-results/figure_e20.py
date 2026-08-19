#!/usr/bin/env python3
"""Plot the finalized E20 identical-evidence order experiment.

The figure reads only strict-analyzer derived tables.  It emphasizes the
registered washout estimands and uses the preregistered phase-transition
summary only as a secondary mechanistic description.
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/forkworld-e20-figure-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/forkworld-e20-figure-xdg")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import Patch, Rectangle

HERE = Path(__file__).resolve().parent
DEFAULT_DERIVED = HERE / "derived"
DEFAULT_FIGURES = HERE / "figures"

SEEDS = (
    577,
    587,
    593,
    599,
    601,
    607,
    613,
    617,
    619,
    631,
    641,
    643,
    647,
    653,
    659,
    661,
    673,
    677,
    683,
    691,
)
OFFSETS = (0, 1, 2, 3, 4, 5, 7, 9, 13, 17, 24, 33, 45, 62, 85, 117, 128, 161, 222, 256)
SCHEDULES = ("p_q_y", "p_y_q", "q_p_y", "q_y_p", "y_p_q", "y_q_p")
GOALS = ("P", "Q", "Y")

INK = "#17212B"
MUTED = "#667085"
GRID = "#D9DEE7"
PALE_BLUE = "#EAF2F8"
BLUE = "#2673B8"
TEAL = "#009E73"
ORANGE = "#D55E00"
GOLD = "#D4A72C"
COMPOSITE = "#AAB2BD"
GOAL_COLORS = {"P": BLUE, "Q": TEAL, "Y": ORANGE}


def register_myriad() -> str:
    """Register Myriad Pro using the same search convention as existing figures."""

    candidates: list[Path] = []
    configured = os.environ.get("FORKWORLD_FONT_DIR")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend((Path.home() / "Library" / "Fonts", Path("/Library/Fonts")))
    names: list[str] = []
    for directory in candidates:
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if (
                path.is_file()
                and "myriad" in path.name.lower()
                and path.suffix.lower() in {".otf", ".ttf", ".ttc"}
            ):
                try:
                    fm.fontManager.addfont(str(path))
                    names.append(fm.FontProperties(fname=str(path)).get_name())
                except RuntimeError:
                    continue
    exact = [name for name in names if name.lower() == "myriad pro"]
    if exact:
        return exact[0]
    discovered = sorted(
        {font.name for font in fm.fontManager.ttflist if "myriad" in font.name.lower()}
    )
    if discovered:
        return discovered[0]
    warnings.warn("Myriad Pro is unavailable; falling back to DejaVu Sans", stacklevel=2)
    return "DejaVu Sans"


def configure_style() -> str:
    font = register_myriad()
    mpl.rcParams.update(
        {
            "font.family": font,
            "font.size": 8.5,
            "axes.titlesize": 10.2,
            "axes.titleweight": 600,
            "axes.labelsize": 8.8,
            "axes.labelcolor": INK,
            "axes.edgecolor": "#AAB2BD",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.9,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return font


def panel_label(ax: Axes, label: str) -> None:
    ax.annotate(
        label,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(-14, 8),
        textcoords="offset points",
        fontsize=11.5,
        fontweight=600,
        color=INK,
        ha="right",
        va="bottom",
        annotation_clip=False,
        zorder=20,
    )


def finish_axis(ax: Axes, *, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.65, alpha=0.78, zorder=0)
    ax.tick_params(length=3, width=0.7)


def require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise RuntimeError(f"{name} lacks required columns: {sorted(missing)}")


def load_tables(derived: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    washout = pd.read_csv(derived / "e20_full_washout_time.csv")
    truth = pd.read_csv(derived / "e20_full_truth_trajectories.csv")
    transitions = pd.read_csv(derived / "e20_full_phase_transitions.csv")
    inference = pd.read_csv(derived / "e20_full_inference.csv")
    require_columns(
        washout,
        {"seed", "washout_offset", "hamming_dispersion", "signed_recency_margin"},
        "washout table",
    )
    require_columns(
        truth,
        {"seed", "schedule", "phase", "local_offset", "truth_table_class"},
        "truth-trajectory table",
    )
    require_columns(
        transitions,
        {
            "seed",
            "schedule",
            "block_position",
            "requested_goal",
            "goal",
            "probe_layer",
            "pure_observed",
            "pure_first_offset",
        },
        "phase-transition table",
    )
    require_columns(
        inference,
        {"endpoint", "estimate", "ci_low", "ci_high", "confidence", "decision"},
        "inference table",
    )
    if (
        len(washout) != len(SEEDS) * len(OFFSETS)
        or tuple(sorted(washout["seed"].unique())) != SEEDS
        or tuple(sorted(washout["washout_offset"].unique())) != OFFSETS
    ):
        raise RuntimeError("washout table is not the exact frozen 20-seed checkpoint grid")
    return washout, truth, transitions, inference


def inference_row(frame: pd.DataFrame, endpoint: str) -> pd.Series:
    selected = frame[frame["endpoint"] == endpoint]
    if len(selected) != 1:
        raise RuntimeError(f"expected exactly one inference row for {endpoint}")
    return selected.iloc[0]


def plot_washout_curve(
    ax: Axes,
    washout: pd.DataFrame,
    inference: pd.DataFrame,
    *,
    column: str,
    endpoint: str,
    title: str,
    ylabel: str,
    color: str,
    ylim: tuple[float, float],
    reference: float,
    reference_label: str,
) -> None:
    matrix = (
        washout.pivot(index="seed", columns="washout_offset", values=column)
        .loc[list(SEEDS), list(OFFSETS)]
        .to_numpy(dtype=float)
    )
    x = np.asarray(OFFSETS, dtype=float)
    mean = matrix.mean(axis=0)
    low = matrix.min(axis=0)
    high = matrix.max(axis=0)
    ax.axvspan(33, 128, color=PALE_BLUE, alpha=0.78, zorder=0)
    ax.fill_between(x, low, high, color=color, alpha=0.12, linewidth=0, zorder=1)
    ax.plot(x, mean, color=color, linewidth=2.15, zorder=4)
    ax.axhline(reference, color=MUTED, linewidth=0.8, linestyle=(0, (2.2, 2.2)), zorder=2)
    ax.text(
        252,
        reference,
        reference_label,
        color=MUTED,
        fontsize=7.0,
        ha="right",
        va="bottom",
    )
    row = inference_row(inference, endpoint)
    decision = str(row["decision"]).replace("_", " ")
    ax.text(
        0.04,
        0.08,
        f"AUC 33–128 = {float(row['estimate']):.3f}\n"  # noqa: RUF001
        f"95% CI [{float(row['ci_low']):.3f}, {float(row['ci_high']):.3f}]\n"
        f"{decision}",
        transform=ax.transAxes,
        fontsize=7.4,
        color=INK,
        ha="left",
        va="bottom",
        linespacing=1.22,
        bbox={"facecolor": "white", "edgecolor": GRID, "linewidth": 0.6, "pad": 3.2},
        zorder=8,
    )
    ax.text(
        80.5,
        ylim[1] - 0.035 * (ylim[1] - ylim[0]),
        "registered persistence window",
        color=BLUE,
        fontsize=6.9,
        ha="center",
        va="top",
    )
    ax.set_xlim(0, 256)
    ax.set_ylim(*ylim)
    ax.set_xticks((0, 33, 62, 128, 192, 256))
    ax.set_xlabel("Common-washout updates")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", pad=7)
    finish_axis(ax)


def schedule_label(schedule: str) -> str:
    # The installed Myriad Pro face lacks a right-arrow glyph. En dashes retain
    # the registered order notation without triggering a font substitution.
    return "–".join(part.upper() for part in schedule.split("_"))  # noqa: RUF001


def plot_endpoint_composition(ax: Axes, truth: pd.DataFrame) -> None:
    selected = truth[(truth["phase"] == "washout") & (truth["local_offset"] == 128)].copy()
    if len(selected) != len(SEEDS) * len(SCHEDULES):
        raise RuntimeError("truth table lacks the exact 120 washout-offset-128 endpoints")
    categories = (*GOALS, "raw_codeword_specific_composite")
    labels = {"P": "Exact P", "Q": "Exact Q", "Y": "Exact Y", "raw_codeword_specific_composite": "Composite"}
    colors = {**GOAL_COLORS, "raw_codeword_specific_composite": COMPOSITE}
    counts = (
        selected.groupby(["schedule", "truth_table_class"]).size().unstack(fill_value=0)
        .reindex(index=SCHEDULES, columns=categories, fill_value=0)
    )
    y = np.arange(len(SCHEDULES), dtype=float)
    left = np.zeros(len(SCHEDULES), dtype=float)
    for category in categories:
        values = counts[category].to_numpy(dtype=float) / len(SEEDS)
        ax.barh(y, values, left=left, height=0.64, color=colors[category], edgecolor="white", linewidth=0.6)
        for row_index, (start, value) in enumerate(zip(left, values, strict=True)):
            if value >= 0.12:
                text_color = "white" if category != "raw_codeword_specific_composite" else INK
                ax.text(start + value / 2, row_index, f"{100 * value:.0f}%", ha="center", va="center", color=text_color, fontsize=7.2, fontweight=600)
        left += values
    # Mark the segment corresponding to the last-presented goal without adding
    # another visual encoding to the bar colors.
    for row_index, schedule in enumerate(SCHEDULES):
        last_goal = schedule.split("_")[-1].upper()
        start = sum(counts.loc[schedule, category] for category in categories[: GOALS.index(last_goal)]) / len(SEEDS)
        width = counts.loc[schedule, last_goal] / len(SEEDS)
        ax.add_patch(Rectangle((start, row_index - 0.34), width, 0.68, fill=False, edgecolor=INK, linewidth=1.0, clip_on=False))
    ax.set_yticks(y, [schedule_label(value) for value in SCHEDULES])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks((0, 0.25, 0.5, 0.75, 1.0), ("0", "25", "50", "75", "100"))
    ax.set_xlabel("Seeds at washout update 128 (%)")
    ax.set_title("Endpoints follow the last-presented rule", loc="left", pad=27)
    ax.grid(axis="x", color=GRID, linewidth=0.65, alpha=0.78, zorder=0)
    ax.tick_params(axis="y", length=0)
    handles = [Patch(facecolor=colors[item], edgecolor="none", label=labels[item]) for item in categories]
    ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.005),
        ncol=4,
        handlelength=1.0,
        columnspacing=0.9,
        borderaxespad=0,
    )


def bootstrap_interval(values: np.ndarray, key: int) -> tuple[float, float, float]:
    if values.shape != (len(SEEDS),) or not np.all(np.isfinite(values)):
        raise RuntimeError("phase acquisition bootstrap requires one finite value per seed")
    generator = np.random.default_rng(key)
    indices = generator.integers(0, len(values), size=(4_000, len(values)))
    draws = values[indices].mean(axis=1)
    return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def plot_phase_acquisition(ax: Axes, transitions: pd.DataFrame) -> None:
    selected = transitions[
        (transitions["probe_layer"] == "first_hidden")
        & (transitions["goal"] == transitions["requested_goal"])
    ].copy()
    if len(selected) != len(SEEDS) * len(SCHEDULES) * 3 or not selected["pure_observed"].all():
        raise RuntimeError("phase table lacks complete requested-goal pure-control acquisitions")
    collapsed = (
        selected.groupby(["seed", "block_position", "requested_goal"], as_index=False)["pure_first_offset"]
        .mean()
    )
    if len(collapsed) != len(SEEDS) * 3 * 3:
        raise RuntimeError("phase table does not collapse to the exact seed-position-goal grid")
    offsets = {"P": -0.16, "Q": 0.0, "Y": 0.16}
    markers = {"P": "o", "Q": "s", "Y": "^"}
    for goal_index, goal in enumerate(GOALS):
        means: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        positions: list[float] = []
        for position in (1, 2, 3):
            values = (
                collapsed[(collapsed["block_position"] == position) & (collapsed["requested_goal"] == goal)]
                .set_index("seed")
                .loc[list(SEEDS), "pure_first_offset"]
                .to_numpy(dtype=float)
            )
            mean, low, high = bootstrap_interval(values, 20_000 + 100 * goal_index + position)
            x = position + offsets[goal]
            deterministic_jitter = np.linspace(-0.035, 0.035, len(SEEDS))
            ax.scatter(
                x + deterministic_jitter,
                values,
                s=9,
                color=GOAL_COLORS[goal],
                alpha=0.20,
                edgecolor="none",
                zorder=2,
            )
            positions.append(x)
            means.append(mean)
            lows.append(low)
            highs.append(high)
        x_values = np.asarray(positions)
        mean_values = np.asarray(means)
        ax.plot(x_values, mean_values, color=GOAL_COLORS[goal], linewidth=1.7, alpha=0.9, zorder=3)
        ax.errorbar(
            x_values,
            mean_values,
            yerr=[mean_values - np.asarray(lows), np.asarray(highs) - mean_values],
            color=GOAL_COLORS[goal],
            marker=markers[goal],
            markersize=5.2,
            markeredgecolor="white",
            markeredgewidth=0.65,
            linewidth=0,
            elinewidth=1.4,
            capsize=2.5,
            label=goal,
            zorder=5,
        )
    ax.set_xlim(0.62, 3.38)
    ax.set_ylim(-4, 126)
    ax.set_xticks((1, 2, 3), ("First block", "Second block", "Third block"))
    ax.set_yticks((0, 24, 48, 72, 96, 120))
    ax.set_ylabel("Updates to stable pure control")
    ax.set_title("Goal installation slows with complexity and history", loc="left", pad=7)
    ax.legend(title="Requested goal", loc="upper left", ncol=3, handlelength=1.2, columnspacing=1.0)
    finish_axis(ax)


def make_figure(derived: Path, figures: Path) -> tuple[Path, Path, str]:
    font = configure_style()
    washout, truth, transitions, inference = load_tables(derived)
    figure, axes = plt.subplots(2, 2, figsize=(7.35, 6.35))
    figure.subplots_adjust(left=0.105, right=0.985, bottom=0.09, top=0.94, wspace=0.35, hspace=0.48)
    ax_a, ax_b, ax_c, ax_d = axes.flat
    plot_washout_curve(
        ax_a,
        washout,
        inference,
        column="hamming_dispersion",
        endpoint="primary_hamming_dispersion_auc_33_128",
        title="Distinct policies survive identical washout",
        ylabel="Pairwise truth-table dispersion",
        color=BLUE,
        ylim=(0.35, 0.406),
        reference=0.4,
        reference_label="three-rule recency pattern",
    )
    plot_endpoint_composition(ax_b, truth)
    plot_washout_curve(
        ax_c,
        washout,
        inference,
        column="signed_recency_margin",
        endpoint="signed_recency_auc_33_128",
        title="Control remains with the most recent goal",
        ylabel="Last-minus-first control margin",
        color=GOLD,
        ylim=(0.84, 1.012),
        reference=1.0,
        reference_label="complete recency",
    )
    plot_phase_acquisition(ax_d, transitions)
    for ax, label in zip(axes.flat, "abcd", strict=True):
        panel_label(ax, label)
    figures.mkdir(parents=True, exist_ok=True)
    pdf = figures / "fig20_identical_evidence_order.pdf"
    png = figures / "fig20_identical_evidence_order.png"
    figure.savefig(pdf, dpi=300, bbox_inches="tight", pad_inches=0.045)
    figure.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.045)
    plt.close(figure)
    return pdf, png, font


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived", type=Path, default=DEFAULT_DERIVED)
    parser.add_argument("--figures", type=Path, default=DEFAULT_FIGURES)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pdf, png, font = make_figure(args.derived.resolve(), args.figures.resolve())
    print(f"font={font}")
    print(pdf)
    print(png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
