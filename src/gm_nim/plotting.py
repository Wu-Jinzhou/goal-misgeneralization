from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def read_metrics(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        return pd.DataFrame(rows)
    return pd.read_csv(path)


def plot_accuracy_curves(
    metrics_file: str,
    output_file: str,
    *,
    x: str = "step",
    y: str = "exact",
    hue: str = "condition",
    chance_column: str | None = None,
) -> None:
    df = read_metrics(metrics_file)
    plt.figure(figsize=(8, 4.8))
    sns.lineplot(data=df, x=x, y=y, hue=hue if hue in df.columns else None, estimator="median")
    if chance_column and chance_column in df.columns:
        for value in sorted(df[chance_column].dropna().unique()):
            plt.axhline(value, color="0.75", linestyle=":", linewidth=1)
    plt.tight_layout()
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=200)
    plt.close()


def plot_probe_heatmap(csv_file: str, output_file: str) -> None:
    df = pd.read_csv(csv_file)
    pivot = df.pivot(index="strategy", columns="layer", values="accuracy")
    plt.figure(figsize=(10, 4))
    sns.heatmap(pivot, vmin=0.5, vmax=1.0, cmap="mako", annot=False)
    plt.tight_layout()
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=200)
    plt.close()


def plot_causal_trace(csv_file: str, output_file: str) -> None:
    df = pd.read_csv(csv_file)
    plt.figure(figsize=(7, 4))
    sns.lineplot(
        data=df,
        x="layer",
        y="p_cheat",
        hue="intervention",
        style="position_mode" if "position_mode" in df.columns else None,
        estimator="median",
        errorbar=("pi", 50),
    )
    plt.tight_layout()
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=200)
    plt.close()


def plot_logit_lens(csv_file: str, output_file: str) -> None:
    df = pd.read_csv(csv_file)
    grid = sns.relplot(
        data=df,
        x="layer",
        y="logit",
        hue="action",
        col="component",
        kind="line",
        facet_kws={"sharey": False},
        height=3.2,
        aspect=1.1,
    )
    grid.tight_layout()
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    grid.savefig(output_file, dpi=200)
    plt.close("all")

