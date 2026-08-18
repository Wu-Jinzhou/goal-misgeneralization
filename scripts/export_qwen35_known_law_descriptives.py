#!/usr/bin/env python3
"""Export prespecified descriptive diagnostics for the Qwen3.5 known-Law panel.

This companion does not alter or replace the prelaunch-pinned primary
analyzer.  It reuses that analyzer's exact 96-run validation, then exports the
IID, matched-audit, checkpoint, causal, controller, and acquisition records
from prelaunch-defined measurements at the registered checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import product
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd

from goalzendo.analysis import (
    AnalysisPanel,
    ControllerCriteria,
    acquisition_intervals,
    classify_controllers,
    walsh_coefficients,
)
from goalzendo.config import get_path, load_config

ROOT = Path(__file__).resolve().parents[1]
FROZEN_ANALYZER_PATH = ROOT / "scripts" / "analyze_qwen35_known_law.py"
MAIN_CONFIG = ROOT / "configs" / "goalzendo" / "qwen35_known_law_main.yaml"

FROZEN_ANALYZER_SHA256 = "0aca675d5600724d6a993373b5ebba560f4bcf4f01887a6f30dc8acd1b7f7c89"
MAIN_CONFIG_SHA256 = "2e1bbd650d466bb42f24b70d6081fd0f4f2490b14bfc7e02d17d59d9e251e7a6"
EXPECTED_IMPLEMENTATION_FINGERPRINT = "592dd4df02ceff4293ce6eebc6ea2a0975e7d427cb1298335bcd3319431f301b"
EXPECTED_STEPS = (0, 1, 4, 16, 64, 256, 512, 1000)
PROMPT_VIEWS = ("full", "audit_law_matched")
CAUSAL_TARGETS = ("y", "p", "q", "d")


class Qwen35DescriptiveError(ValueError):
    """Raised when a promised descriptive record is absent or malformed."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_frozen_analyzer() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "goalzendo_frozen_qwen35_known_law_analyzer",
        FROZEN_ANALYZER_PATH,
    )
    if specification is None or specification.loader is None:  # pragma: no cover
        raise RuntimeError("cannot load the frozen Qwen3.5 analyzer")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _verify_frozen_bindings() -> None:
    observed_analyzer = _sha256(FROZEN_ANALYZER_PATH)
    if observed_analyzer != FROZEN_ANALYZER_SHA256:
        raise Qwen35DescriptiveError(
            "prelaunch-pinned primary analyzer changed: "
            f"expected {FROZEN_ANALYZER_SHA256}, observed {observed_analyzer}"
        )
    observed_config = _sha256(MAIN_CONFIG)
    if observed_config != MAIN_CONFIG_SHA256:
        raise Qwen35DescriptiveError(
            f"prelaunch-pinned main config changed: expected {MAIN_CONFIG_SHA256}, observed {observed_config}"
        )


# Check the primary program before importing and executing any of its code.
# ``analyze_main`` repeats the check at invocation time so an intervening file
# replacement also fails closed.
_verify_frozen_bindings()
FROZEN = _load_frozen_analyzer()


def _required_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise Qwen35DescriptiveError(f"{label} lacks columns: {missing}")


def _unit_interval(value: Any, label: str) -> float:
    try:
        return float(FROZEN._finite_unit_interval(value, label))
    except (TypeError, ValueError) as exc:
        raise Qwen35DescriptiveError(str(exc)) from exc


def _preferred_split(step: int) -> str:
    return "final_factorial" if step == EXPECTED_STEPS[-1] else "diagnostic_factorial"


def _behavior_at(
    panel: AnalysisPanel,
    *,
    run_id: str,
    step: int,
    prompt_view: str,
) -> dict[str, float]:
    split = _preferred_split(step)
    rows = panel.trajectory[
        (panel.trajectory["run_id"] == run_id)
        & (panel.trajectory["step"] == step)
        & (panel.trajectory["prompt_view"] == prompt_view)
        & (panel.trajectory["split"] == split)
        & (panel.trajectory["panel"] == "conflict")
    ]
    if len(rows) != 1:
        raise Qwen35DescriptiveError(
            f"{run_id} needs exactly one {split} conflict row at step {step} "
            f"in view {prompt_view}; observed {len(rows)}"
        )
    row = rows.iloc[0]
    return {
        name: _unit_interval(row[name], f"{run_id} step {step} {prompt_view} {name}")
        for name in ("rho_y", "rho_p", "rho_q")
    }


def _causal_at(panel: AnalysisPanel, *, run_id: str, step: int) -> dict[str, float]:
    split = _preferred_split(step)
    rows = panel.causal_effects[
        (panel.causal_effects["run_id"] == run_id)
        & (panel.causal_effects["step"] == step)
        & (panel.causal_effects["prompt_view"] == "full")
        & (panel.causal_effects["split"] == split)
    ]
    counts = Counter(str(value) for value in rows["target"])
    expected = Counter({target: 1 for target in CAUSAL_TARGETS})
    if counts != expected:
        raise Qwen35DescriptiveError(
            f"{run_id} step {step} full-view causal targets differ from "
            f"{dict(expected)}: observed {dict(counts)}"
        )
    by_target = {str(row["target"]): row for row in rows.to_dict(orient="records")}
    return {
        f"flip_{target}": _unit_interval(
            by_target[target]["action_flip_rate"],
            f"{run_id} step {step} flip_{target}",
        )
        for target in CAUSAL_TARGETS
    }


def _final_iid_law_accuracy(panel: AnalysisPanel, *, run_id: str) -> float:
    rows = panel.trajectory[
        (panel.trajectory["run_id"] == run_id)
        & (panel.trajectory["step"] == EXPECTED_STEPS[-1])
        & (panel.trajectory["prompt_view"] == "full")
        & (panel.trajectory["split"] == "iid_validation")
        & (panel.trajectory["panel"] == "all")
    ]
    if len(rows) != 1:
        raise Qwen35DescriptiveError(
            f"{run_id} needs exactly one final full-view IID all-panel row; observed {len(rows)}"
        )
    return _unit_interval(rows.iloc[0]["rho_y"], f"{run_id} final IID rho_y")


def _factorial_at(
    panel: AnalysisPanel,
    *,
    run_id: str,
    step: int,
    prompt_view: str,
    expected_n: int,
) -> pd.DataFrame:
    split = _preferred_split(step)
    rows = panel.factorial[
        (panel.factorial["run_id"] == run_id)
        & (panel.factorial["step"] == step)
        & (panel.factorial["prompt_view"] == prompt_view)
        & (panel.factorial["split"] == split)
    ].copy()
    if len(rows) != 8:
        raise Qwen35DescriptiveError(
            f"{run_id} step {step} needs exactly eight {prompt_view} factorial cells; observed {len(rows)}"
        )
    cells = [
        (int(row["choice_y"]), int(row["choice_p"]), int(row["choice_q"]))
        for row in rows.to_dict(orient="records")
    ]
    expected_cells = set(product((0, 1), repeat=3))
    if set(cells) != expected_cells or len(set(cells)) != 8:
        raise Qwen35DescriptiveError(
            f"{run_id} step {step} does not contain the exact 2^3 {prompt_view} factorial"
        )
    for index, row in rows.iterrows():
        try:
            observed_n = float(row["n"])
        except (TypeError, ValueError) as exc:
            raise Qwen35DescriptiveError(
                f"{run_id} step {step} {prompt_view} factorial cell has an invalid count"
            ) from exc
        if not observed_n.is_integer() or int(observed_n) != expected_n:
            raise Qwen35DescriptiveError(
                f"{run_id} step {step} {prompt_view} factorial cell count differs from "
                f"the registered {expected_n}: observed {row['n']}"
            )
        rows.loc[index, "action_b_rate"] = _unit_interval(
            row["action_b_rate"],
            f"{run_id} step {step} {prompt_view} factorial action_b_rate",
        )
    return rows


def _optional_step(value: Any) -> int | None:
    return None if pd.isna(value) else int(value)


def _criteria_signature(criteria: ControllerCriteria) -> str:
    return json.dumps(criteria.as_dict(), sort_keys=True, separators=(",", ":"))


def _run_descriptives(
    panel: AnalysisPanel,
    run: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> tuple[dict[str, Any], ControllerCriteria]:
    run_id = str(run["run_id"])
    plan_key = str(run["plan_key"])
    config = run.get("config")
    if not isinstance(config, Mapping):
        raise Qwen35DescriptiveError(f"{run_id} lacks a resolved config")
    if plan_key not in expected:
        raise Qwen35DescriptiveError(f"{run_id} has an unexpected plan key")
    eval_steps = tuple(int(step) for step in get_path(config, "train.eval_steps"))
    if eval_steps != EXPECTED_STEPS:
        raise Qwen35DescriptiveError(
            f"{run_id} evaluation steps changed: expected {EXPECTED_STEPS}, observed {eval_steps}"
        )
    diagnostic_n = int(get_path(config, "data.n_eval_per_cell"))
    final_n = int(get_path(config, "evaluation.final_eval_per_cell"))

    conflict_rows: list[dict[str, Any]] = []
    causal_rows: list[dict[str, Any]] = []
    classifier_rows: list[dict[str, Any]] = []
    factorial_parts: list[pd.DataFrame] = []
    for step in EXPECTED_STEPS:
        views = {
            view: _behavior_at(panel, run_id=run_id, step=step, prompt_view=view) for view in PROMPT_VIEWS
        }
        causal = _causal_at(panel, run_id=run_id, step=step)
        conflict_rows.append({"step": step, **views})
        causal_rows.append({"step": step, **causal})
        classifier_rows.append(
            {
                "run_id": run_id,
                "cell_id": str(run["cell_id"]),
                "seed": int(run["seed"]),
                "step": step,
                "prompt_view": "full",
                "split": _preferred_split(step),
                "panel": "conflict",
                **views["full"],
                **{key.replace("flip_", "causal_"): value for key, value in causal.items()},
            }
        )
        expected_n = final_n if step == EXPECTED_STEPS[-1] else diagnostic_n
        for prompt_view in PROMPT_VIEWS:
            factorial = _factorial_at(
                panel,
                run_id=run_id,
                step=step,
                prompt_view=prompt_view,
                expected_n=expected_n,
            )
            if prompt_view == "full":
                factorial_parts.append(factorial)

    classifier_frame = pd.DataFrame(classifier_rows)
    factorial = pd.concat(factorial_parts, ignore_index=True)
    walsh = walsh_coefficients(
        factorial,
        group_columns=("run_id", "cell_id", "seed", "step", "prompt_view", "split"),
    )
    criteria = ControllerCriteria.from_config(config)
    classified = classify_controllers(classifier_frame, criteria=criteria, walsh=walsh)
    if len(classified) != len(EXPECTED_STEPS):
        raise Qwen35DescriptiveError(
            f"{run_id} controller trajectory has {len(classified)} rather than eight rows"
        )
    controller_rows = [
        {
            "step": int(row["step"]),
            "raw_controller": str(row["raw_controller"]),
            "persistent_controller": (None if pd.isna(row["controller"]) else str(row["controller"])),
        }
        for row in classified.sort_values("step").to_dict(orient="records")
    ]

    acquisitions: list[dict[str, Any]] = []
    group_columns = tuple(
        column
        for column in ("run_id", "cell_id", "seed", "prompt_view", "panel")
        if column in classified.columns
    )
    for candidate in ("Y", "P", "Q"):
        rows = acquisition_intervals(
            classified,
            candidate,
            persistence=criteria.persistence,
            group_columns=group_columns,
        )
        if len(rows) != 1:
            raise Qwen35DescriptiveError(
                f"{run_id} {candidate} acquisition has {len(rows)} rather than one row"
            )
        row = rows.iloc[0]
        acquisitions.append(
            {
                "candidate": candidate,
                "left_step": _optional_step(row["left_step"]),
                "right_step": _optional_step(row["right_step"]),
                "confirmation_step": _optional_step(row["confirmation_step"]),
                "censoring": str(row["censoring"]),
            }
        )

    seed, model, algorithm, law, q_p = FROZEN._condition(expected[plan_key])
    final_audit = conflict_rows[-1]["audit_law_matched"]
    return (
        {
            "seed": int(seed),
            "model": str(model),
            "algorithm": str(algorithm),
            "law": str(law),
            "q_p": float(q_p),
            "run_id": run_id,
            "plan_key": plan_key,
            "final_full_iid_law_accuracy": _final_iid_law_accuracy(panel, run_id=run_id),
            "final_audit_law_matched_conflict_rho_y": float(final_audit["rho_y"]),
            "conflict_agreement_trajectories": conflict_rows,
            "full_view_causal_action_flip_trajectories": causal_rows,
            "full_view_controller_trajectory": controller_rows,
            "full_view_acquisition_intervals": acquisitions,
        },
        criteria,
    )


def export_descriptives(
    panel: AnalysisPanel,
    config: Mapping[str, Any],
    *,
    config_sha256: str,
    analyzer_sha256: str,
    expected_implementation_fingerprint: str,
) -> dict[str, Any]:
    """Build the complete claim-neutral descriptive companion document."""

    expected = FROZEN._expected_plan(config)
    FROZEN._validate_run_rows(panel, expected)
    _required_columns(
        panel.trajectory,
        {"run_id", "step", "prompt_view", "split", "panel", "rho_y", "rho_p", "rho_q"},
        "trajectory table",
    )
    _required_columns(
        panel.causal_effects,
        {"run_id", "step", "prompt_view", "split", "target", "action_flip_rate"},
        "causal table",
    )
    _required_columns(
        panel.factorial,
        {
            "run_id",
            "cell_id",
            "seed",
            "step",
            "prompt_view",
            "split",
            "choice_y",
            "choice_p",
            "choice_q",
            "action_b_rate",
            "n",
        },
        "factorial table",
    )

    audit = dict(panel.audit)
    observed_fingerprint = audit.get("implementation_fingerprint")
    if observed_fingerprint != expected_implementation_fingerprint:
        raise Qwen35DescriptiveError(
            "panel implementation fingerprint differs from the launched source: "
            f"expected {expected_implementation_fingerprint}, observed {observed_fingerprint}"
        )

    runs: list[dict[str, Any]] = []
    criteria_by_signature: dict[str, ControllerCriteria] = {}
    for run in panel.runs.to_dict(orient="records"):
        record, criteria = _run_descriptives(panel, run, expected)
        runs.append(record)
        criteria_by_signature[_criteria_signature(criteria)] = criteria
    if len(criteria_by_signature) != 1:
        raise Qwen35DescriptiveError(
            "controller classification criteria differ across the frozen 96-run panel"
        )
    criteria = next(iter(criteria_by_signature.values()))
    runs.sort(
        key=lambda row: (
            int(row["seed"]),
            str(row["model"]),
            str(row["algorithm"]),
            str(row["law"]),
            float(row["q_p"]),
        )
    )
    return {
        "schema": "goalzendo.qwen35_known_law_descriptive_companion",
        "schema_version": 1,
        "epistemic_status": {
            "analysis_role": "descriptive_companion_to_prelaunch_frozen_primary_analysis",
            "materializes_prelaunch_defined_measurements_from_registered_checkpoints": True,
            "adds_inferential_hypotheses": False,
            "computes_p_values_or_confidence_intervals": False,
        },
        "panel": {
            "completed_only": True,
            "expected_run_count": 96,
            "observed_run_count": len(runs),
            "evaluation_steps": list(EXPECTED_STEPS),
            "prompt_views": list(PROMPT_VIEWS),
            "full_view_causal_targets": list(CAUSAL_TARGETS),
            "expected_config_sha256": config_sha256,
            "frozen_primary_analyzer_sha256": analyzer_sha256,
            "expected_implementation_fingerprint": expected_implementation_fingerprint,
            "implementation_fingerprint": observed_fingerprint,
        },
        "controller_classification": {
            "implementation": "goalzendo.analysis.classify_controllers",
            "acquisition_implementation": "goalzendo.analysis.acquisition_intervals",
            "criteria": criteria.as_dict(),
            "persistent_label_requires_consecutive_checkpoints": criteria.persistence,
            "acquisition_times_are_interval_censored": True,
        },
        "runs": runs,
    }


def analyze_main(artifacts: Path) -> dict[str, Any]:
    _verify_frozen_bindings()
    config = load_config(MAIN_CONFIG)
    expected = FROZEN._expected_plan(config)
    panel = FROZEN._load_exact_panel(artifacts, config)
    FROZEN._validate_run_rows(panel, expected)
    return export_descriptives(
        panel,
        config,
        config_sha256=MAIN_CONFIG_SHA256,
        analyzer_sha256=FROZEN_ANALYZER_SHA256,
        expected_implementation_fingerprint=EXPECTED_IMPLEMENTATION_FINGERPRINT,
    )


def canonical_json(value: Mapping[str, Any]) -> str:
    """Return deterministic strict JSON with exactly one terminal newline."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export the complete prespecified descriptive companion for the exact "
            "96-run Qwen3.5 known-Law panel"
        )
    )
    parser.add_argument("artifacts", type=Path, help="root containing the main run artifacts")
    args = parser.parse_args(argv)
    print(canonical_json(analyze_main(args.artifacts)), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
