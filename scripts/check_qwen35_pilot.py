#!/usr/bin/env python3
"""Apply the prospective, outcome-blind Qwen3.5 pilot proceed rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any

from goalzendo.analysis import load_analysis_panel, seed_level_trajectories
from goalzendo.config import get_path, load_config


def _iid_accuracy(panel: Any, run_id: str, final_step: int) -> float:
    rows = panel.raw_metrics[
        (panel.raw_metrics["run_id"] == run_id)
        & (panel.raw_metrics["step"] == final_step)
        & (panel.raw_metrics["kind"] == "behavioral_agreement")
        & (panel.raw_metrics["split"] == "iid_validation")
        & (panel.raw_metrics["prompt_view"] == "full")
        & (panel.raw_metrics["panel"] == "all")
    ]
    values = {
        float(row.get("rho_y", row.get("agreement_y")))
        for row in rows.to_dict(orient="records")
        if row.get("rho_y", row.get("agreement_y")) is not None
    }
    if len(values) != 1:
        raise ValueError(f"{run_id} has {len(values)} final IID Law-accuracy values")
    return next(iter(values))


def check_pilot(artifacts: Path, config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    panel = load_analysis_panel(
        artifacts,
        expected_config=config,
        require_complete_metrics=True,
        required_prompt_views=("full", "audit_law_matched"),
    )
    trajectory = seed_level_trajectories(
        panel,
        prompt_view="audit_law_matched",
        require_causal=False,
    )
    checks: list[dict[str, Any]] = []
    for run in panel.runs.itertuples(index=False):
        run_config = run.config
        run_id = str(run.run_id)
        final_step = int(get_path(run_config, "train.steps"))
        algorithm = str(get_path(run_config, "train.algorithm"))
        model = str(get_path(run_config, "model.name"))
        iid = _iid_accuracy(panel, run_id, final_step)
        final_rows = trajectory[
            (trajectory["run_id"] == run_id) & (trajectory["step"] == final_step)
        ]
        if len(final_rows) != 1:
            raise ValueError(f"{run_id} lacks one final matched-layout row")
        matched_law = float(final_rows.iloc[0]["rho_y"])
        sampling: float | None = None
        if algorithm == "outcome_rl":
            early = panel.raw_metrics[
                (panel.raw_metrics["run_id"] == run_id)
                & (panel.raw_metrics["kind"] == "optimization")
                & panel.raw_metrics["step"].between(1, 32)
            ]
            observed_steps = set(int(value) for value in early["step"])
            if observed_steps != set(range(1, 33)):
                raise ValueError(f"{run_id} lacks exact RL updates 1--32")
            sampling = float(median(float(value) for value in early["both_actions_sampled_fraction"]))
        passed = iid >= 0.90 and (
            matched_law >= 0.90 if algorithm == "sft" else sampling is not None and sampling >= 0.10
        )
        checks.append(
            {
                "run_id": run_id,
                "model": model,
                "algorithm": algorithm,
                "final_iid_law_accuracy": iid,
                "final_matched_layout_law_agreement": matched_law,
                "early_both_actions_sampled_median": sampling,
                "passed": passed,
            }
        )
    return {
        "schema": "goalzendo.qwen35_pilot_decision",
        "proceed": len(checks) == 4 and all(row["passed"] for row in checks),
        "outcome_blind_rule": (
            "Proceed depends only on complete scoring/model integrity, SFT Law capability in the matched "
            "layout, RL IID learning, and noncollapsed early action sampling. Conflict controller outcomes "
            "and SFT--RL differences do not affect this decision."
        ),
        "runs": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/goalzendo/qwen35_known_law_pilot.yaml"),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = check_pilot(args.artifacts, args.config)
    serialized = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if result["proceed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
