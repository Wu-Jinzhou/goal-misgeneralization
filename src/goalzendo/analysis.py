"""Fail-closed, seed-level analysis for completed GoalZendo artifacts.

The independent unit of replication is a trained model seed.  Evaluation
examples and factorial cells are measurements within that seed; they are never
treated as independent samples for uncertainty intervals.  This module also
keeps acquisition times interval-censored because a controller can change at
any update between two saved checkpoints.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import yaml  # type: ignore[import-untyped]
from numpy.typing import NDArray

from .artifacts import (
    ArtifactError,
    _identity_config,
    atomic_text,
    discover_runs,
    read_json,
    read_jsonl,
    stable_hash,
    verify_completion_attestation,
    write_json,
)
from .config import get_path
from .runner import (
    G00_OPTIMIZER_STABILITY_POLICY,
    RunSpec,
    build_plan,
    derived_seeds,
    g00_evidence_binding,
)

MIN_INFERENTIAL_SEEDS = 3
DEFAULT_BOOTSTRAP_DRAWS = 10_000
DEFAULT_CONFIDENCE = 0.95
DEFAULT_EQUIVALENCE_MARGIN = 0.10
G00_ASSESSMENT_SCHEMA = "goalzendo.g00_assessment"
G00_ASSESSMENT_SCHEMA_VERSION = 2


class AnalysisError(RuntimeError):
    """Base class for analysis-contract violations."""


class ProvenanceError(AnalysisError):
    """Raised when completed artifacts do not share auditable provenance."""


class PanelCompletenessError(AnalysisError):
    """Raised when a planned seed, checkpoint, view, or diagnostic cell is absent."""


def flatten(mapping: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten scalar config fields while retaining lists as canonical JSON."""

    result: dict[str, Any] = {}
    for key, value in mapping.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            result.update(flatten(value, name))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            result[name] = value
        elif isinstance(value, (list, tuple)):
            result[name] = json.dumps(value, sort_keys=True)
    return result


def _clean_resolved_config(config: Mapping[str, Any]) -> dict[str, Any]:
    cleaned = copy.deepcopy(dict(config))
    cleaned.pop("seed", None)
    return cleaned


def _spec_for_seed(config: Mapping[str, Any], seed: int) -> RunSpec:
    matches = [spec for spec in build_plan(_clean_resolved_config(config)) if spec.seed == seed]
    if len(matches) != 1:
        raise ProvenanceError(
            f"resolved config must plan focal seed {seed} exactly once; found {len(matches)}"
        )
    return matches[0]


def _read_manifest(run: Path, kind: str) -> tuple[str, Mapping[str, Any]]:
    path = run / "manifests" / f"{kind}.json"
    if not path.is_file():
        raise ProvenanceError(f"completed run is missing {kind} manifest: {run}")
    payload = read_json(path)
    if payload.get("kind") != kind or not isinstance(payload.get("metadata"), Mapping):
        raise ProvenanceError(f"malformed {kind} manifest: {path}")
    observed = str(payload.get("digest", ""))
    expected = stable_hash(payload["metadata"], 64)
    if observed != expected:
        raise ProvenanceError(f"{kind} manifest digest mismatch: {path}")
    return observed, payload["metadata"]


@dataclass(frozen=True)
class _LoadedRun:
    path: Path
    run_id: str
    seed: int
    plan_key: str
    cell_id: str
    config: Mapping[str, Any]
    summary: Mapping[str, Any]
    identity: Mapping[str, Any]
    manifest_digests: Mapping[str, str]
    manifest_metadata: Mapping[str, Mapping[str, Any]]
    metrics: tuple[Mapping[str, Any], ...]
    predictions: tuple[Mapping[str, Any], ...]


def _load_run(path: Path) -> _LoadedRun:
    required = (
        "COMPLETE",
        "identity.json",
        "resolved_config.yaml",
        "summary.json",
        "status.json",
        "implementation.json",
        "environment.json",
    )
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise ProvenanceError(f"completed run {path} is missing: {', '.join(missing)}")

    identity = read_json(path / "identity.json")
    run_id = path.name
    if stable_hash(identity, 20) != run_id:
        raise ProvenanceError(f"run directory does not match identity hash: {path}")
    if int(identity.get("artifact_schema_version", -1)) >= 2:
        try:
            verify_completion_attestation(path)
        except ArtifactError as exc:
            raise ProvenanceError(f"completion attestation failed: {path}") from exc
    status = read_json(path / "status.json")
    if status.get("state") != "complete" or status.get("run_id") != run_id:
        raise ProvenanceError(f"COMPLETE marker and status disagree: {path}")

    loaded_config = yaml.safe_load((path / "resolved_config.yaml").read_text(encoding="utf-8"))
    if not isinstance(loaded_config, Mapping):
        raise ProvenanceError(f"resolved config is not a mapping: {path}")
    config = dict(loaded_config)
    if "seed" not in config:
        raise ProvenanceError(f"resolved config does not identify its focal seed: {path}")
    seed = int(config["seed"])
    if identity.get("seed") != seed:
        raise ProvenanceError(f"identity and resolved config seed disagree: {path}")
    if identity.get("config") != _identity_config(_clean_resolved_config(config)):
        raise ProvenanceError(f"identity and resolved scientific config disagree: {path}")

    implementation = read_json(path / "implementation.json")
    fingerprint = identity.get("implementation_fingerprint")
    if implementation.get("implementation_fingerprint") != fingerprint:
        raise ProvenanceError(f"implementation manifest and identity disagree: {path}")
    if identity.get("artifact_schema_version") != implementation.get("artifact_schema_version"):
        raise ProvenanceError(f"artifact schema provenance disagrees: {path}")

    summary = read_json(path / "summary.json")
    if summary.get("run_id") != run_id or int(summary.get("seed", seed)) != seed:
        raise ProvenanceError(f"summary identity disagrees: {path}")
    spec = _spec_for_seed(config, seed)
    plan_key = str(summary.get("plan_key", spec.plan_key))
    if plan_key != spec.plan_key:
        raise ProvenanceError(f"summary plan key disagrees with resolved config: {path}")

    manifest_payloads = {kind: _read_manifest(path, kind) for kind in ("dataset", "model", "tokenizer")}
    manifests = {kind: payload[0] for kind, payload in manifest_payloads.items()}
    manifest_metadata = {kind: payload[1] for kind, payload in manifest_payloads.items()}
    requested_model = str(get_path(config, "model.name", ""))
    requested_revision = str(get_path(config, "model.revision", "main"))
    for kind in ("model", "tokenizer"):
        metadata = manifest_metadata[kind]
        recorded_model = metadata.get("requested_model", metadata.get("name"))
        recorded_revision = metadata.get("requested_revision", metadata.get("revision"))
        if recorded_model is not None and str(recorded_model) != requested_model:
            raise ProvenanceError(f"{kind} manifest conflicts with configured model name: {path}")
        if recorded_revision is not None and str(recorded_revision) != requested_revision:
            raise ProvenanceError(f"{kind} manifest conflicts with configured revision: {path}")
    metrics = tuple(read_jsonl(path / "metrics.jsonl"))
    predictions = tuple(read_jsonl(path / "predictions.jsonl"))
    for stream_name, records in (("metrics", metrics), ("predictions", predictions)):
        for record in records:
            if record.get("run_id", run_id) != run_id or int(record.get("seed", seed)) != seed:
                raise ProvenanceError(f"{stream_name} record identity disagrees in {path}")
    return _LoadedRun(
        path=path,
        run_id=run_id,
        seed=seed,
        plan_key=plan_key,
        cell_id=spec.cell_id,
        config=config,
        summary=summary,
        identity=identity,
        manifest_digests=manifests,
        manifest_metadata=manifest_metadata,
        metrics=metrics,
        predictions=predictions,
    )


def _target_name(value: Any) -> str | None:
    normalized = str(value).strip().lower()
    aliases = {
        "y": "y",
        "law": "y",
        "official_law": "y",
        "p": "p",
        "herald": "p",
        "stamp": "p",
        "q": "q",
        "sage": "q",
        "d": "d",
        "distractor": "d",
        "control": "d",
        "flip_y": "y",
        "flip_p": "p",
        "flip_q": "q",
        "flip_d": "d",
    }
    return aliases.get(normalized)


def _choice(value: Any) -> int:
    if isinstance(value, bool):
        raise AnalysisError("Boolean values are not valid A/B choices")
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return int(value)
    normalized = str(value).strip().upper()
    if normalized in {"A", "0"}:
        return 0
    if normalized in {"B", "1"}:
        return 1
    raise AnalysisError(f"invalid A/B choice: {value!r}")


def _step(record: Mapping[str, Any], final_step: int) -> int:
    value = record.get("step", record.get("checkpoint_step", record.get("update", final_step)))
    if isinstance(value, bool):
        raise AnalysisError("checkpoint step cannot be Boolean")
    return int(value)


def _expand_metric_record(record: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    """Expand a serialized EvaluationResult while preserving checkpoint metadata."""

    nested_names = (
        ("behavioral_agreement", "behavioral_agreement"),
        ("factorial_cells", "factorial_cell"),
        ("intervention_summary", "intervention_summary"),
    )
    expanded = False
    parent = {
        key: value
        for key, value in record.items()
        if key not in {name for name, _kind in nested_names} and key != "records"
    }
    for name, kind in nested_names:
        nested = record.get(name)
        if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes)):
            expanded = True
            for item in nested:
                if not isinstance(item, Mapping):
                    raise AnalysisError(f"{name} must contain mappings")
                yield {**parent, **dict(item), "kind": kind}
    if not expanded:
        yield dict(record)


_BEHAVIOR_ALIASES = {
    "rho_y": "rho_y",
    "agreement_y": "rho_y",
    "rho_p": "rho_p",
    "agreement_p": "rho_p",
    "rho_q": "rho_q",
    "agreement_q": "rho_q",
}
_SCALAR_MEASURES = frozenset(
    {
        "iid_reward",
        "validation_reward",
        "train_reward",
        "reward",
        "loss",
        "rule_report_accuracy",
        "decoder_accuracy",
    }
)


def _record_common(run: _LoadedRun, record: Mapping[str, Any]) -> dict[str, Any]:
    source_split = str(record.get("split", "unspecified"))
    return {
        "run_path": str(run.path),
        "run_id": run.run_id,
        "plan_key": run.plan_key,
        "cell_id": run.cell_id,
        "seed": run.seed,
        "step": _step(record, int(get_path(run.config, "train.steps", 0))),
        "prompt_view": str(record.get("prompt_view", "full")),
        "split": source_split.removesuffix("_causal"),
        "source_split": source_split,
    }


def _metric_tables(
    runs: Sequence[_LoadedRun],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    values: list[dict[str, Any]] = []
    causal: list[dict[str, Any]] = []
    factorial: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []

    for run in runs:
        for original in run.metrics:
            for record in _expand_metric_record(original):
                common = _record_common(run, record)
                raw.append({**dict(record), **common})
                panel = str(record.get("panel", "all"))
                kind = str(record.get("kind", "")).strip().lower()

                if kind not in {"factorial", "factorial_cell", "prediction"}:
                    for key, measure in _BEHAVIOR_ALIASES.items():
                        if key in record and record[key] is not None:
                            values.append(
                                {
                                    **common,
                                    "panel": panel,
                                    "measure": measure,
                                    "value": float(record[key]),
                                    "priority": 0,
                                }
                            )
                        conflict_key = f"conflict_{key}"
                        if conflict_key in record and record[conflict_key] is not None:
                            values.append(
                                {
                                    **common,
                                    "panel": "conflict",
                                    "measure": measure,
                                    "value": float(record[conflict_key]),
                                    "priority": 0,
                                }
                            )

                for measure in _SCALAR_MEASURES:
                    if measure in record and record[measure] is not None:
                        values.append(
                            {
                                **common,
                                "panel": panel,
                                "measure": measure,
                                "value": float(record[measure]),
                                "priority": 0,
                            }
                        )

                metric_name = str(record.get("metric", "")).strip().lower()
                if metric_name and "value" in record and record["value"] is not None:
                    long_panel = panel
                    normalized_metric = metric_name
                    if metric_name.startswith("conflict_"):
                        long_panel = "conflict"
                        normalized_metric = metric_name.removeprefix("conflict_")
                    normalized_metric = _BEHAVIOR_ALIASES.get(normalized_metric, normalized_metric)
                    recognized = set(_BEHAVIOR_ALIASES.values()) | set(_SCALAR_MEASURES)
                    if normalized_metric in recognized:
                        values.append(
                            {
                                **common,
                                "panel": long_panel,
                                "measure": normalized_metric,
                                "value": float(record["value"]),
                                "priority": 0,
                            }
                        )

                target = _target_name(record.get("target", record.get("intervention_target", "")))
                if target is not None:
                    effect_fields = {
                        "action_flip_rate": record.get("action_flip_rate"),
                        "mean_absolute_delta_margin": record.get(
                            "mean_absolute_delta_margin",
                            record.get("mean_abs_delta_margin"),
                        ),
                        "mean_target_aligned_delta_margin": record.get("mean_target_aligned_delta_margin"),
                        "mean_delta_margin_b_minus_a": record.get("mean_delta_margin_b_minus_a"),
                        "n_pairs": record.get("n_pairs", record.get("n")),
                    }
                    direct = record.get(f"causal_{target}", record.get("causal_effect"))
                    if effect_fields["action_flip_rate"] is None and direct is not None:
                        effect_fields["action_flip_rate"] = direct
                    if any(value is not None for value in effect_fields.values()):
                        causal.append(
                            {
                                **common,
                                "target": target,
                                **{
                                    key: (float(value) if key != "n_pairs" and value is not None else value)
                                    for key, value in effect_fields.items()
                                },
                                "priority": 0,
                            }
                        )
                for target_name in ("y", "p", "q", "d"):
                    direct = record.get(f"causal_{target_name}")
                    if direct is not None and target is None:
                        flip_rate = record.get(f"causal_{target_name}_flip_rate")
                        effect: dict[str, Any] = {
                            **common,
                            "target": target_name,
                            "action_flip_rate": float(direct if flip_rate is None else flip_rate),
                            "priority": 0,
                        }
                        if flip_rate is not None:
                            margin_field = (
                                "mean_absolute_delta_margin"
                                if target_name == "d"
                                else "mean_target_aligned_delta_margin"
                            )
                            effect[margin_field] = float(direct)
                        causal.append(effect)

                required_factorial = {"choice_y", "choice_p", "choice_q", "action_b_rate"}
                if required_factorial.issubset(record):
                    factorial.append(
                        {
                            **common,
                            "choice_y": _choice(record["choice_y"]),
                            "choice_p": _choice(record["choice_p"]),
                            "choice_q": _choice(record["choice_q"]),
                            "action_b_rate": float(record["action_b_rate"]),
                            "n": int(record.get("n", 1)),
                            "priority": 0,
                        }
                    )

    raw_frame = pd.DataFrame(raw)
    values_frame = pd.DataFrame(values)
    causal_frame = pd.DataFrame(causal)
    factorial_frame = pd.DataFrame(factorial)
    return raw_frame, values_frame, causal_frame, factorial_frame


def _prediction_tables(
    runs: Sequence[_LoadedRun],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_rows: list[dict[str, Any]] = []
    behavior_values: list[dict[str, Any]] = []
    causal_rows: list[dict[str, Any]] = []
    factorial_rows: list[dict[str, Any]] = []

    for run in runs:
        final_step = int(get_path(run.config, "train.steps", 0))
        for record in run.predictions:
            common = _record_common(run, record)
            raw_rows.append({**dict(record), **common})

        frame = pd.DataFrame(raw_rows)
        if frame.empty:
            continue
        frame = frame[frame["run_id"] == run.run_id].copy()
        candidate_columns = {"choice_y", "choice_p", "choice_q"}
        if not candidate_columns.issubset(frame.columns):
            continue
        for column in (*candidate_columns, "predicted_action"):
            if column in frame:
                frame[column] = frame[column].map(_choice)
        if "step" not in frame:
            frame["step"] = final_step
        base = frame[frame.get("intervention_role", pd.Series(index=frame.index, dtype=object)).isna()].copy()
        group_keys = [
            "run_path",
            "run_id",
            "plan_key",
            "cell_id",
            "seed",
            "step",
            "prompt_view",
            "split",
        ]
        for keys, group in base.groupby(group_keys, dropna=False, sort=True):
            common_group = dict(zip(group_keys, keys, strict=True))
            conflict = group[["choice_y", "choice_p", "choice_q"]].nunique(axis=1) > 1
            for panel, selected in (("all", group), ("conflict", group[conflict])):
                if selected.empty:
                    continue
                for target in ("y", "p", "q"):
                    behavior_values.append(
                        {
                            **common_group,
                            "panel": panel,
                            "measure": f"rho_{target}",
                            "value": float(
                                (selected["predicted_action"] == selected[f"choice_{target}"]).mean()
                            ),
                            "priority": 1,
                        }
                    )
            for choices, cell in group.groupby(["choice_y", "choice_p", "choice_q"], sort=True):
                factorial_rows.append(
                    {
                        **common_group,
                        "choice_y": int(choices[0]),
                        "choice_p": int(choices[1]),
                        "choice_q": int(choices[2]),
                        "action_b_rate": float((cell["predicted_action"] == 1).mean()),
                        "n": len(cell),
                        "priority": 1,
                    }
                )

        pair_columns = {
            "intervention_pair_id",
            "intervention_role",
            "intervention_target",
            "margin_b_minus_a",
        }
        if pair_columns.issubset(frame.columns):
            paired = frame.dropna(subset=list(pair_columns)).copy()
            pair_keys = [
                "run_path",
                "run_id",
                "plan_key",
                "cell_id",
                "seed",
                "step",
                "prompt_view",
                "split",
                "intervention_target",
                "intervention_pair_id",
            ]
            effects: list[dict[str, Any]] = []
            for keys, group in paired.groupby(pair_keys, dropna=False, sort=True):
                base_rows = group[group["intervention_role"].astype(str).str.lower() == "base"]
                changed_rows = group[group["intervention_role"].astype(str).str.lower() == "intervention"]
                if len(base_rows) != 1 or len(changed_rows) != 1:
                    raise PanelCompletenessError(
                        f"causal pair {keys[-1]!r} needs one base and one intervention"
                    )
                left, right = base_rows.iloc[0], changed_rows.iloc[0]
                effects.append(
                    {
                        **dict(zip(pair_keys[:-2], keys[:-2], strict=True)),
                        "target": _target_name(keys[-2]),
                        "delta_margin": float(right["margin_b_minus_a"] - left["margin_b_minus_a"]),
                        "action_flipped": int(right["predicted_action"] != left["predicted_action"]),
                    }
                )
            effects_frame = pd.DataFrame(effects)
            if not effects_frame.empty:
                causal_keys = [
                    "run_path",
                    "run_id",
                    "plan_key",
                    "cell_id",
                    "seed",
                    "step",
                    "prompt_view",
                    "split",
                    "target",
                ]
                for keys, group in effects_frame.groupby(causal_keys, dropna=False, sort=True):
                    target_name = keys[-1]
                    if target_name is None:
                        raise AnalysisError("unknown intervention target in prediction stream")
                    causal_rows.append(
                        {
                            **dict(zip(causal_keys, keys, strict=True)),
                            "action_flip_rate": float(group["action_flipped"].mean()),
                            "mean_absolute_delta_margin": float(group["delta_margin"].abs().mean()),
                            "mean_delta_margin_b_minus_a": float(group["delta_margin"].mean()),
                            "n_pairs": len(group),
                            "priority": 1,
                        }
                    )

    return (
        pd.DataFrame(raw_rows),
        pd.DataFrame(behavior_values),
        pd.DataFrame(causal_rows),
        pd.DataFrame(factorial_rows),
    )


def _deduplicate_long(
    frame: pd.DataFrame,
    keys: Sequence[str],
    value_columns: Sequence[str],
) -> pd.DataFrame:
    if frame.empty:
        return frame
    rows: list[dict[str, Any]] = []
    for values, group in frame.groupby(list(keys), dropna=False, sort=True):
        values = values if isinstance(values, tuple) else (values,)
        minimum_priority = int(group["priority"].min()) if "priority" in group else 0
        selected = group[group["priority"] == minimum_priority] if "priority" in group else group
        row = dict(zip(keys, values, strict=True))
        for column in value_columns:
            if column not in selected:
                continue
            observed = pd.to_numeric(selected[column], errors="coerce").dropna().to_numpy(dtype=float)
            if observed.size == 0:
                row[column] = np.nan
            elif not np.allclose(observed, observed[0], atol=1e-10, rtol=1e-10):
                raise PanelCompletenessError(
                    f"conflicting duplicate {column} values for {row}: {observed.tolist()}"
                )
            else:
                row[column] = float(observed[0])
        rows.append(row)
    return pd.DataFrame(rows)


def _assemble_tables(
    metric_values: pd.DataFrame,
    metric_causal: pd.DataFrame,
    metric_factorial: pd.DataFrame,
    prediction_values: pd.DataFrame,
    prediction_causal: pd.DataFrame,
    prediction_factorial: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    values = pd.concat([metric_values, prediction_values], ignore_index=True)
    value_keys = [
        "run_path",
        "run_id",
        "plan_key",
        "cell_id",
        "seed",
        "step",
        "prompt_view",
        "split",
        "panel",
        "measure",
    ]
    values = _deduplicate_long(values, value_keys, ["value"])
    if values.empty:
        trajectory = pd.DataFrame(columns=[*value_keys[:-1]])
    else:
        trajectory = (
            values.pivot(index=value_keys[:-1], columns="measure", values="value")
            .reset_index()
            .rename_axis(columns=None)
        )

    causal = pd.concat([metric_causal, prediction_causal], ignore_index=True)
    causal_keys = [
        "run_path",
        "run_id",
        "plan_key",
        "cell_id",
        "seed",
        "step",
        "prompt_view",
        "split",
        "target",
    ]
    causal = _deduplicate_long(
        causal,
        causal_keys,
        [
            "action_flip_rate",
            "mean_absolute_delta_margin",
            "mean_target_aligned_delta_margin",
            "mean_delta_margin_b_minus_a",
            "n_pairs",
        ],
    )
    if not causal.empty:
        for column in (
            "action_flip_rate",
            "mean_absolute_delta_margin",
            "mean_target_aligned_delta_margin",
            "mean_delta_margin_b_minus_a",
        ):
            if column not in causal:
                causal[column] = np.nan
        effect = causal.pivot(index=causal_keys[:-1], columns="target", values="action_flip_rate").rename(
            columns=lambda target: f"causal_{target}"
        )
        margins = causal.pivot(
            index=causal_keys[:-1], columns="target", values="mean_absolute_delta_margin"
        ).rename(columns=lambda target: f"causal_margin_{target}")
        signed = causal.pivot(
            index=causal_keys[:-1],
            columns="target",
            values="mean_target_aligned_delta_margin",
        ).rename(columns=lambda target: f"causal_signed_margin_{target}")
        raw_delta = causal.pivot(
            index=causal_keys[:-1],
            columns="target",
            values="mean_delta_margin_b_minus_a",
        ).rename(columns=lambda target: f"causal_delta_margin_{target}")
        wide_causal = effect.join(margins, how="outer").join(signed, how="outer")
        wide_causal = wide_causal.join(raw_delta, how="outer").reset_index()
        trajectory = trajectory.merge(
            wide_causal,
            on=causal_keys[:-1],
            how="outer",
            validate="many_to_one",
        )

    factorial = pd.concat([metric_factorial, prediction_factorial], ignore_index=True)
    factorial_keys = [
        "run_path",
        "run_id",
        "plan_key",
        "cell_id",
        "seed",
        "step",
        "prompt_view",
        "split",
        "choice_y",
        "choice_p",
        "choice_q",
    ]
    factorial = _deduplicate_long(factorial, factorial_keys, ["action_b_rate", "n"])
    return trajectory.sort_values(value_keys[:-2]).reset_index(drop=True), causal, factorial


@dataclass(frozen=True)
class AnalysisPanel:
    """Validated completed runs and their normalized evaluation tables."""

    runs: pd.DataFrame
    raw_metrics: pd.DataFrame
    predictions: pd.DataFrame
    trajectory: pd.DataFrame
    causal_effects: pd.DataFrame
    factorial: pd.DataFrame
    audit: Mapping[str, Any]


def _planned_keys(config: Mapping[str, Any]) -> set[str]:
    return {spec.plan_key for spec in build_plan(config)}


def load_analysis_panel(
    root: str | Path,
    *,
    expected_config: Mapping[str, Any] | None = None,
    require_complete_metrics: bool = False,
    required_prompt_views: Sequence[str] | None = None,
    allow_incomplete_runs: bool = False,
) -> AnalysisPanel:
    """Load completed artifacts and reject mixed or incomplete scientific panels."""

    paths = discover_runs(root, completed_only=True)
    if not paths:
        raise PanelCompletenessError(f"no completed GoalZendo runs found under {root}")
    loaded = tuple(_load_run(path) for path in paths)
    run_ids = [run.run_id for run in loaded]
    plan_keys = [run.plan_key for run in loaded]
    if len(set(run_ids)) != len(run_ids) or len(set(plan_keys)) != len(plan_keys):
        raise ProvenanceError("artifact panel contains duplicate run or plan identities")
    fingerprints = {str(run.identity.get("implementation_fingerprint")) for run in loaded}
    schemas = {int(run.identity.get("artifact_schema_version", -1)) for run in loaded}
    if len(fingerprints) != 1 or len(schemas) != 1:
        raise ProvenanceError(
            "one analysis panel must use exactly one source fingerprint and artifact schema"
        )
    experiments = {
        (
            str(get_path(run.config, "experiment.id", "")),
            str(get_path(run.config, "experiment.name", "")),
        )
        for run in loaded
    }
    if len(experiments) != 1:
        raise ProvenanceError(f"analysis root mixes experiments: {sorted(experiments)}")
    stable_runtime_fields = (
        "requested_model",
        "requested_revision",
        "resolved_revision",
        "tokenizer_resolved_revision",
        "model_class",
        "tokenizer_class",
        "vocabulary_size",
        "chat_template_sha256",
        "action_token_ids",
    )
    for cell_id in {run.cell_id for run in loaded}:
        cell_runs = [run for run in loaded if run.cell_id == cell_id]
        for kind in ("model", "tokenizer"):
            for field_name in stable_runtime_fields:
                values = {
                    json.dumps(run.manifest_metadata[kind][field_name], sort_keys=True)
                    for run in cell_runs
                    if run.manifest_metadata[kind].get(field_name) is not None
                }
                if len(values) > 1:
                    raise ProvenanceError(
                        f"{kind} provenance field {field_name!r} varies across seeds in cell {cell_id}"
                    )

    if expected_config is not None:
        expected_keys = _planned_keys(expected_config)
    else:
        expected_keys = set()
        for run in loaded:
            expected_keys.update(_planned_keys(_clean_resolved_config(run.config)))
    observed_keys = set(plan_keys)
    missing_keys = sorted(expected_keys - observed_keys)
    unexpected_keys = sorted(observed_keys - expected_keys)
    if unexpected_keys or (missing_keys and not allow_incomplete_runs):
        raise PanelCompletenessError(
            f"seed-level artifact panel is incomplete: {len(missing_keys)} missing, "
            f"{len(unexpected_keys)} unexpected"
        )

    run_rows: list[dict[str, Any]] = []
    for run in loaded:
        row = {
            "run_path": str(run.path),
            "run_id": run.run_id,
            "plan_key": run.plan_key,
            "cell_id": run.cell_id,
            "seed": run.seed,
            "implementation_fingerprint": run.identity["implementation_fingerprint"],
            "dataset_manifest_digest": run.manifest_digests["dataset"],
            "model_manifest_digest": run.manifest_digests["model"],
            "tokenizer_manifest_digest": run.manifest_digests["tokenizer"],
            "config": run.config,
            "dataset_manifest_metadata": run.manifest_metadata["dataset"],
            "model_manifest_metadata": run.manifest_metadata["model"],
            "tokenizer_manifest_metadata": run.manifest_metadata["tokenizer"],
            **flatten(run.config, "config"),
            **flatten(run.summary, "summary"),
            **flatten(run.manifest_metadata["dataset"], "manifest.dataset"),
            **flatten(run.manifest_metadata["model"], "manifest.model"),
            **flatten(run.manifest_metadata["tokenizer"], "manifest.tokenizer"),
        }
        run_rows.append(row)
    runs_frame = pd.DataFrame(run_rows).sort_values(["cell_id", "seed"]).reset_index(drop=True)
    raw_metrics, metric_values, metric_causal, metric_factorial = _metric_tables(loaded)
    predictions, prediction_values, prediction_causal, prediction_factorial = _prediction_tables(loaded)
    trajectory, causal, factorial = _assemble_tables(
        metric_values,
        metric_causal,
        metric_factorial,
        prediction_values,
        prediction_causal,
        prediction_factorial,
    )
    audit = {
        "run_count": len(loaded),
        "cell_count": int(runs_frame["cell_id"].nunique()),
        "seed_count": int(runs_frame["seed"].nunique()),
        "implementation_fingerprint": next(iter(fingerprints)),
        "artifact_schema_version": next(iter(schemas)),
        "expected_plan_keys": len(expected_keys),
        "observed_plan_keys": len(observed_keys),
        "panel_complete": not missing_keys,
        "missing_planned_runs": len(missing_keys),
        "design_cell_completeness_basis": (
            "prospective_config" if expected_config is not None else "completed_configs_only"
        ),
    }
    panel = AnalysisPanel(
        runs=runs_frame,
        raw_metrics=raw_metrics,
        predictions=predictions,
        trajectory=trajectory,
        causal_effects=causal,
        factorial=factorial,
        audit=audit,
    )
    if require_complete_metrics:
        validate_trajectory_completeness(panel, prompt_views=required_prompt_views)
    return panel


def intention_to_train_ledger(
    root: str | Path,
    expected_config: Mapping[str, Any],
) -> pd.DataFrame:
    """Enumerate every planned seed, including failed, incomplete, and missing runs."""

    expected = {spec.plan_key: spec for spec in build_plan(expected_config)}
    attempts: dict[str, tuple[Path, Mapping[str, Any]]] = {}
    for discovered_path in discover_runs(root, completed_only=False):
        resolved_path = discovered_path / "resolved_config.yaml"
        if not resolved_path.is_file():
            continue
        loaded = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, Mapping) or "seed" not in loaded:
            continue
        resolved = dict(loaded)
        seed = int(resolved["seed"])
        try:
            spec = _spec_for_seed(resolved, seed)
        except (ProvenanceError, ValueError):
            continue
        if spec.plan_key not in expected:
            continue
        if spec.plan_key in attempts:
            raise ProvenanceError(f"multiple artifact attempts resolve to plan key {spec.plan_key}")
        discovered_status = (
            read_json(discovered_path / "status.json") if (discovered_path / "status.json").is_file() else {}
        )
        attempts[spec.plan_key] = (discovered_path, discovered_status)

    rows: list[dict[str, Any]] = []
    for plan_key, spec in sorted(expected.items(), key=lambda item: item[1].global_index):
        attempt = attempts.get(plan_key)
        if attempt is None:
            attempt_path: Path | None = None
            attempt_status: Mapping[str, Any] = {}
            state = "missing_not_attempted"
        else:
            attempt_path, attempt_status = attempt
            complete_marker = (attempt_path / "COMPLETE").is_file()
            recorded = str(attempt_status.get("state", "unknown"))
            state = "complete" if complete_marker and recorded == "complete" else recorded
            if complete_marker != (recorded == "complete"):
                state = "corrupt_completion_state"
        rows.append(
            {
                "plan_key": plan_key,
                "cell_id": spec.cell_id,
                "seed": spec.seed,
                "run_path": str(attempt_path) if attempt_path is not None else None,
                "run_id": attempt_path.name if attempt_path is not None else None,
                "state": state,
                "outcome_observed": state == "complete",
                "intention_to_train": True,
                "error_type": attempt_status.get("error_type"),
                "last_step": attempt_status.get("last_step"),
                "attempt": attempt_status.get("attempt"),
                "law_family": str(get_path(spec.config, "data.rule_family", "")),
                "q_p": float(get_path(spec.config, "data.q_p")),
                "q_q": float(get_path(spec.config, "data.q_q")),
                "algorithm": str(get_path(spec.config, "train.algorithm", "")),
                "planned_final_step": int(get_path(spec.config, "train.steps", 0)),
            }
        )
    return pd.DataFrame(rows)


def paired_algorithm_integrity(
    panel: AnalysisPanel,
    ledger: pd.DataFrame,
    *,
    pair_columns: Sequence[str] = ("seed", "law_family", "q_p", "q_q"),
) -> pd.DataFrame:
    """Audit the frozen SFT/RL pairing fields for every planned matched pair."""

    required_ledger = {*pair_columns, "algorithm", "state", "run_id"}
    if not required_ledger.issubset(ledger.columns):
        raise PanelCompletenessError(
            f"ITT ledger is missing pairing columns: {sorted(required_ledger - set(ledger.columns))}"
        )
    run_lookup = panel.runs.set_index("run_id", drop=False) if not panel.runs.empty else panel.runs
    comparisons = {
        "dataset_manifest": "dataset_manifest_digest",
        "training_prompt_order": "manifest.dataset.rendering.training_prompt_digest",
        "training_action_order": "manifest.dataset.rendering.training_action_digest",
        "training_renderer_order": "manifest.dataset.rendering.training_renderer_digest",
        "initial_adapter": "manifest.model.initial_trainable_parameter_digest",
        "trainable_parameter_count": "manifest.model.trainable_parameter_count",
        "model_revision": "manifest.model.resolved_revision",
        "tokenizer_revision": "manifest.tokenizer.resolved_revision",
        "action_token_ids": "manifest.tokenizer.action_token_ids",
    }
    rows: list[dict[str, Any]] = []
    for keys, group in ledger.groupby(list(pair_columns), dropna=False, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        common = dict(zip(pair_columns, keys, strict=True))
        by_algorithm = {
            algorithm: part for algorithm, part in group.groupby("algorithm", dropna=False, sort=True)
        }
        row: dict[str, Any] = {**common}
        pair_complete = set(by_algorithm) == {"sft", "outcome_rl"} and all(
            len(by_algorithm[algorithm]) == 1 and by_algorithm[algorithm].iloc[0]["state"] == "complete"
            for algorithm in ("sft", "outcome_rl")
        )
        row["pair_complete"] = pair_complete
        row["sft_state"] = by_algorithm["sft"].iloc[0]["state"] if "sft" in by_algorithm else "not_planned"
        row["outcome_rl_state"] = (
            by_algorithm["outcome_rl"].iloc[0]["state"] if "outcome_rl" in by_algorithm else "not_planned"
        )
        checks: list[bool] = []
        for label, column in comparisons.items():
            if not pair_complete or column not in panel.runs.columns:
                passed: bool | None = None
            else:
                left_id = str(by_algorithm["sft"].iloc[0]["run_id"])
                right_id = str(by_algorithm["outcome_rl"].iloc[0]["run_id"])
                left = run_lookup.loc[left_id, column]
                right = run_lookup.loc[right_id, column]
                passed = bool(left == right and not pd.isna(left))
                checks.append(passed)
            row[f"check_{label}"] = passed
        row["integrity_passed"] = bool(pair_complete and checks and all(checks))
        rows.append(row)
    return pd.DataFrame(rows)


def _completed_g00_runs(
    artifact_roots: Sequence[str | Path],
    configs: Sequence[Mapping[str, Any]],
) -> tuple[_LoadedRun, ...]:
    expected = {spec.plan_key for config in configs for spec in build_plan(config)}
    observed: dict[str, _LoadedRun] = {}
    for root in artifact_roots:
        for path in discover_runs(root, completed_only=True):
            run = _load_run(path)
            if run.plan_key not in expected:
                continue
            if run.plan_key in observed:
                raise ProvenanceError(f"duplicate G00 run for plan key {run.plan_key}")
            observed[run.plan_key] = run
    missing = sorted(expected - set(observed))
    if missing:
        raise PanelCompletenessError(
            f"G00 assessment cannot be derived: {len(missing)} planned runs are missing"
        )
    return tuple(observed[key] for key in sorted(observed))


def _analysis_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PanelCompletenessError(f"{name} must be an object")
    return value


def _final_base_predictions(run: _LoadedRun) -> pd.DataFrame:
    records = []
    final_step = int(get_path(run.config, "train.steps", 0))
    for record in run.predictions:
        if _step(record, final_step) != final_step:
            continue
        source_split = str(record.get("split", ""))
        if source_split.removesuffix("_causal") != "final_factorial":
            continue
        if record.get("intervention_role") is not None:
            continue
        required = {
            "choice_y",
            "choice_p",
            "choice_q",
            "predicted_action",
            "score_a",
            "score_b",
            "probability_b",
        }
        if not required.issubset(record):
            raise PanelCompletenessError(f"G00 final prediction stream omits scorer fields in {run.run_id}")
        records.append(
            {
                **dict(record),
                "run_id": run.run_id,
                "seed": run.seed,
                "training_view": str(get_path(run.config, "data.training_view", "full")),
            }
        )
    if not records:
        raise PanelCompletenessError(f"G00 run has no final factorial predictions: {run.run_id}")
    frame = pd.DataFrame(records)
    for column in ("choice_y", "choice_p", "choice_q", "predicted_action"):
        frame[column] = frame[column].map(_choice)
    return frame


def _dataset_integrity_measurements(runs: Sequence[_LoadedRun]) -> dict[str, int]:
    truth_mismatches = 0
    overlap_count = 0
    duplicate_pair_ids = 0
    regeneration_mismatches = 0

    def overlap_values(value: Any, path: str = "") -> Iterable[int]:
        if isinstance(value, Mapping):
            for key, item in value.items():
                child = f"{path}.{key}" if path else str(key)
                yield from overlap_values(item, child)
        elif "overlap" in path.lower() and isinstance(value, (int, float)):
            yield int(value)

    from .experiment import materialize_banks

    for run in runs:
        metadata = run.manifest_metadata["dataset"]
        banks = metadata.get("banks")
        if not isinstance(banks, Mapping):
            raise PanelCompletenessError(f"G00 dataset manifest lacks bank audit: {run.run_id}")
        for name in ("diagnostic_factorial", "final_factorial"):
            bank = _analysis_mapping(banks.get(name), f"dataset banks.{name}")
            counts = _analysis_mapping(bank.get("candidate_counts"), f"dataset banks.{name}.candidate_counts")
            numeric = [int(value) for value in counts.values()]
            if len(counts) != 8 or len(set(numeric)) != 1 or sum(numeric) != int(bank.get("count", -1)):
                truth_mismatches += 1
            requested_mirrors = bool(get_path(run.config, "evaluation.mirror_pairs", False))
            registered = int(bank.get("registered_mirror_pairs", 0))
            if requested_mirrors and registered * 2 != int(bank.get("count", -1)):
                duplicate_pair_ids += 1
        train_bank = _analysis_mapping(banks.get("train"), "dataset banks.train")
        train_counts = _analysis_mapping(train_bank.get("candidate_counts"), "dataset banks.train counts")
        total = sum(int(value) for value in train_counts.values())
        exact_p = sum(
            int(count)
            for cell, count in train_counts.items()
            if len(str(cell)) == 3 and str(cell)[0] == str(cell)[1]
        )
        exact_q = sum(
            int(count)
            for cell, count in train_counts.items()
            if len(str(cell)) == 3 and str(cell)[0] == str(cell)[2]
        )
        if (
            total != int(train_bank.get("count", -1))
            or exact_p != round(total * float(get_path(run.config, "data.q_p")))
            or exact_q != round(total * float(get_path(run.config, "data.q_q")))
        ):
            truth_mismatches += 1
        overlap_count += sum(overlap_values(metadata.get("semantic_scene_audit", {})))

        regenerated = materialize_banks(run.config, derived_seeds(run.seed)).metadata
        stored_symbolic = {key: metadata.get(key) for key in regenerated}
        if stable_hash(stored_symbolic, 64) != stable_hash(regenerated, 64):
            regeneration_mismatches += 1
    return {
        "truth_cell_count_mismatches": truth_mismatches,
        "scene_overlap_count": overlap_count,
        "duplicate_pair_ids": duplicate_pair_ids,
        "regeneration_mismatches": regeneration_mismatches,
    }


def _capability_measurement_panel(
    runs: Sequence[_LoadedRun],
    predictions: Mapping[str, pd.DataFrame],
) -> dict[str, Any]:
    adapter_targets = {
        "law_only": "y",
        "audit_law_matched": "y",
        "sage_only": "q",
        "herald_only": "p",
    }
    adapter_agreement: dict[str, dict[str, float]] = {}
    law_only_by_run: list[dict[str, Any]] = []
    matched_law_by_run: list[dict[str, Any]] = []
    for view, target in adapter_targets.items():
        pieces = [
            predictions[run.run_id]
            for run in runs
            if str(get_path(run.config, "data.training_view", "full")) == view
        ]
        if not pieces:
            raise PanelCompletenessError(f"G00 capability panel has no {view} predictions")
        frame = pd.concat(pieces, ignore_index=True)
        target_column = f"choice_{target}"
        adapter_agreement[view] = {}
        for position, choice in (("A", 0), ("B", 1)):
            selected = frame[frame[target_column] == choice]
            if selected.empty:
                raise PanelCompletenessError(f"G00 {view} has no target-{position} examples")
            adapter_agreement[view][position] = float(
                (selected["predicted_action"] == selected[target_column]).mean()
            )
        if view in {"law_only", "audit_law_matched"}:
            for run in runs:
                if str(get_path(run.config, "data.training_view", "full")) != view:
                    continue
                run_frame = predictions[run.run_id]
                record: dict[str, Any] = {
                    "run_id": run.run_id,
                    "seed": run.seed,
                    "law_family": str(get_path(run.config, "data.rule_family", "")),
                }
                for position, choice in (("A", 0), ("B", 1)):
                    side = run_frame[run_frame["choice_y"] == choice]
                    if side.empty:
                        raise PanelCompletenessError(
                            f"G00 law_only run {run.run_id} has no target-{position} examples"
                        )
                    record[position] = float((side["predicted_action"] == side["choice_y"]).mean())
                if view == "law_only":
                    law_only_by_run.append(record)
                else:
                    matched_law_by_run.append(record)

    chance_results: dict[str, dict[str, int]] = {}
    for view, output_name in (("no_signal", "no_signal"), ("surface_only", "surface_classifier")):
        pieces = [
            predictions[run.run_id]
            for run in runs
            if str(get_path(run.config, "data.training_view", "full")) == view
        ]
        if not pieces:
            raise PanelCompletenessError(f"G00 capability panel has no {view} predictions")
        frame = pd.concat(pieces, ignore_index=True)
        chance_results[output_name] = {
            "correct": int((frame["predicted_action"] == frame["choice_y"]).sum()),
            "trials": len(frame),
        }

    scorer_frames = [predictions[run.run_id] for run in runs]
    scorer = pd.concat(scorer_frames, ignore_index=True)
    numeric = scorer[["score_a", "score_b", "probability_b"]].to_numpy(dtype=float)
    finite = np.isfinite(numeric).all(axis=1)
    score_values = scorer[["score_a", "score_b"]].to_numpy(dtype=float)
    shifted = score_values - np.max(score_values, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probability_sums = (exp_scores / exp_scores.sum(axis=1, keepdims=True)).sum(axis=1)
    position_bias = max(abs(float((frame["predicted_action"] == 1).mean()) - 0.5) for frame in scorer_frames)
    return {
        "rule_adapter_position_agreement": adapter_agreement,
        "law_only_position_agreement_by_run": law_only_by_run,
        "matched_law_position_agreement_by_run": matched_law_by_run,
        "constrained_scorer": {
            "finite_probability_fraction": float(finite.mean()),
            "maximum_probability_sum_error": float(np.max(np.abs(probability_sums - 1.0))),
            "maximum_absolute_position_bias": position_bias,
        },
        **chance_results,
    }


def derive_g00_gate_assessment(
    artifact_roots: Sequence[str | Path],
    configs: Sequence[Mapping[str, Any]],
    *,
    repo: str | Path,
) -> dict[str, Any]:
    """Derive every G00 gate measurement from immutable run artifacts."""

    runs = _completed_g00_runs(artifact_roots, configs)
    evidence = g00_evidence_binding(artifact_roots, configs, repo)
    all_predictions = {run.run_id: _final_base_predictions(run) for run in runs}
    aggregate_capability = _capability_measurement_panel(runs, all_predictions)
    by_model_runs: dict[str, list[_LoadedRun]] = {}
    model_identities: dict[str, dict[str, str]] = {}
    for run in runs:
        metadata = run.manifest_metadata["model"]
        identity = {
            "requested_model": str(metadata.get("requested_model", metadata.get("name", ""))),
            "requested_revision": str(
                metadata.get(
                    "requested_revision",
                    metadata.get("resolved_revision", metadata.get("revision", "")),
                )
            ),
        }
        digest = stable_hash(identity, 64)
        by_model_runs.setdefault(digest, []).append(run)
        model_identities[digest] = identity
    capability_by_model = {
        digest: {
            "model_identity": model_identities[digest],
            **_capability_measurement_panel(group, all_predictions),
        }
        for digest, group in sorted(by_model_runs.items())
    }

    optimizer_candidates: dict[str, list[dict[str, Any]]] = {"sft": [], "outcome_rl": []}
    optimizer_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for run in runs:
        if str(get_path(run.config, "data.training_view", "full")) != "full":
            continue
        # Only the exact deployed-model full-context pilot can select G01
        # optimizer settings. The 0.5B grid proposes candidates but cannot
        # authorize the 1.5B confirmatory model.
        if str(get_path(run.config, "experiment.name", "")) != "deployed_model_full_context_pilot":
            continue
        series = [
            record
            for record in run.metrics
            if str(record.get("kind", "")).lower() == "behavioral_agreement"
            and str(record.get("split", "")) == "iid_validation"
            and str(record.get("panel", "")) == "all"
            and str(record.get("prompt_view", "full")) == "full"
            and record.get("rho_y", record.get("agreement_y")) is not None
        ]
        policy = G00_OPTIMIZER_STABILITY_POLICY
        required_iid_steps = (policy.baseline_step, *policy.terminal_steps)
        iid_by_step: dict[int, Mapping[str, Any]] = {}
        for record in series:
            step = _step(record, 0)
            if step not in required_iid_steps:
                continue
            if step in iid_by_step:
                raise PanelCompletenessError(
                    f"G00 optimizer run duplicates IID step {step}: {run.run_id}"
                )
            iid_by_step[step] = record
        if set(iid_by_step) != set(required_iid_steps):
            raise PanelCompletenessError(
                f"G00 optimizer run lacks exact IID steps {required_iid_steps}: {run.run_id}"
            )
        if any(iid_by_step[step].get("action_b_rate") is None for step in policy.terminal_steps):
            raise PanelCompletenessError(
                f"G00 optimizer run lacks terminal IID action frequencies: {run.run_id}"
            )
        baseline_accuracy = float(
            iid_by_step[policy.baseline_step].get(
                "rho_y", iid_by_step[policy.baseline_step].get("agreement_y")
            )
        )
        terminal_accuracies = {
            step: float(iid_by_step[step].get("rho_y", iid_by_step[step].get("agreement_y")))
            for step in policy.terminal_steps
        }
        terminal_action_rates = {
            step: float(iid_by_step[step]["action_b_rate"]) for step in policy.terminal_steps
        }
        model_metadata = run.manifest_metadata["model"]
        model_identity = {
            "requested_model": str(
                model_metadata.get("requested_model", model_metadata.get("name", ""))
            ),
            "requested_revision": str(
                model_metadata.get(
                    "requested_revision",
                    model_metadata.get("resolved_revision", model_metadata.get("revision", "")),
                )
            ),
        }
        update_method = str(get_path(run.config, "update.method", ""))
        effective_batch = int(get_path(run.config, "train.batch_size", 0)) * int(
            get_path(run.config, "train.gradient_accumulation_steps", 0)
        )
        key = (
            str(get_path(run.config, "train.algorithm", "")),
            float(get_path(run.config, "train.learning_rate")),
            float(get_path(run.config, "train.entropy_coefficient", 0.0)),
            stable_hash(model_identity, 64),
            update_method,
            effective_batch,
        )
        early_both: float | None = None
        early_zero: float | None = None
        early_steps: list[int] | None = None
        if key[0] == "outcome_rl":
            early = [
                record
                for record in run.metrics
                if str(record.get("kind", "")).lower() == "optimization"
                and _step(record, 0) in policy.rl_sampling_steps
            ]
            observed_steps = [_step(record, 0) for record in early]
            if sorted(observed_steps) != list(policy.rl_sampling_steps) or len(
                set(observed_steps)
            ) != len(policy.rl_sampling_steps):
                raise PanelCompletenessError(
                    f"G00 RL pilot lacks exactly one sampling row at steps 1--32: {run.run_id}"
                )
            early.sort(key=lambda record: _step(record, 0))
            if any(
                record.get("both_actions_sampled_fraction") is None
                or record.get("all_zero_loo_advantages_fraction") is None
                for record in early
            ):
                raise PanelCompletenessError(
                    f"G00 RL pilot lacks steps 1--32 sampling diagnostics: {run.run_id}"
                )
            early_both = float(
                np.median([float(record["both_actions_sampled_fraction"]) for record in early])
            )
            early_zero = float(
                np.median(
                    [float(record["all_zero_loo_advantages_fraction"]) for record in early]
                )
            )
            early_steps = list(policy.rl_sampling_steps)
        optimizer_groups.setdefault(key, []).append(
            {
                "run_id": run.run_id,
                "baseline_step": policy.baseline_step,
                "baseline_iid_accuracy": baseline_accuracy,
                "terminal_iid_accuracy": {
                    str(step): terminal_accuracies[step] for step in policy.terminal_steps
                },
                "terminal_action_b_rate": {
                    str(step): terminal_action_rates[step] for step in policy.terminal_steps
                },
                "early_sampling_steps": early_steps,
                "early_both_actions_sampled_median": early_both,
                "early_all_zero_advantages_median": early_zero,
                "law_family": str(get_path(run.config, "data.rule_family", "")),
                "q_p": float(get_path(run.config, "data.q_p", 0.0)),
                "seed": run.seed,
                "model_identity": model_identity,
            }
        )
    for (
        algorithm,
        learning_rate,
        entropy,
        _model_digest,
        update_method,
        effective_batch,
    ), values in sorted(optimizer_groups.items()):
        if algorithm not in optimizer_candidates:
            continue
        optimizer_candidates[algorithm].append(
            {
                "learning_rate": learning_rate,
                "entropy_coefficient": entropy,
                "stability_window_steps": list(
                    G00_OPTIMIZER_STABILITY_POLICY.terminal_steps
                ),
                "run_measurements": sorted(
                    values,
                    key=lambda value: (
                        str(value["law_family"]),
                        float(value["q_p"]),
                        int(value["seed"]),
                        str(value["run_id"]),
                    ),
                ),
                "sampling_window": (
                    "per-run median over exact optimizer steps 1--32"
                    if algorithm == "outcome_rl"
                    else None
                ),
                "model_identity": dict(values[0]["model_identity"]),
                "update_method": update_method,
                "effective_batch": effective_batch,
                "law_families": sorted({str(value["law_family"]) for value in values}),
                "proxy_accuracies": sorted({float(value["q_p"]) for value in values}),
                "seeds": sorted({int(value["seed"]) for value in values}),
                "n_runs": len(values),
            }
        )

    measurements: dict[str, Any] = {
        **aggregate_capability,
        "capability_by_model": capability_by_model,
        "optimizer_stability": optimizer_candidates,
        "dataset_integrity": _dataset_integrity_measurements(runs),
    }
    body = {
        "schema": G00_ASSESSMENT_SCHEMA,
        "schema_version": G00_ASSESSMENT_SCHEMA_VERSION,
        "evidence_run_binding_digest": evidence["run_binding_digest"],
        "evidence_source_fingerprint": evidence["source_fingerprint"],
        "evidence_config_digest": evidence["expected_config_digest"],
        "optimizer_stability_policy": G00_OPTIMIZER_STABILITY_POLICY.as_dict(),
        "optimizer_stability_policy_digest": stable_hash(
            G00_OPTIMIZER_STABILITY_POLICY.as_dict(), 64
        ),
        "measurements": measurements,
        "derivation": {
            "source": "completed_metrics_predictions_and_manifests",
            "run_count": len(runs),
            "optimizer_aggregation": (
                "exact per-run baseline step 0, terminal steps 128/256, and RL steps 1--32"
            ),
            "dataset_regeneration": True,
        },
    }
    return {**body, "assessment_digest": stable_hash(body, 64)}


def validate_trajectory_completeness(
    panel: AnalysisPanel,
    *,
    behavior_panel: str = "conflict",
    prompt_views: Sequence[str] | None = None,
    causal_targets: Sequence[str] = ("y", "p", "q"),
    require_causal: bool | None = None,
) -> None:
    """Require every planned checkpoint/view and all causal candidate flips."""

    missing: list[str] = []
    for run in panel.runs.itertuples(index=False):
        config = run.config
        expected_steps = [int(step) for step in get_path(config, "train.eval_steps", [])]
        views = (
            tuple(str(view) for view in prompt_views)
            if prompt_views is not None
            else tuple(str(view) for view in get_path(config, "evaluation.prompt_views", ["full"]))
        )
        causal_views = set(str(view) for view in get_path(config, "evaluation.causal_prompt_views", ["full"]))
        rows = panel.trajectory[panel.trajectory["run_id"] == run.run_id]
        for step, view in itertools.product(expected_steps, views):
            matched = rows[
                (rows["step"] == step) & (rows["prompt_view"] == view) & (rows["panel"] == behavior_panel)
            ]
            preferred_split = (
                "final_factorial"
                if step == int(get_path(config, "train.steps", -1))
                else "diagnostic_factorial"
            )
            if "split" in matched and preferred_split in set(matched["split"]):
                matched = matched[matched["split"] == preferred_split]
            elif "split" in matched and matched["split"].nunique() > 1:
                missing.append(f"{run.run_id}:step={step}:view={view}:ambiguous diagnostic split")
                continue
            if len(matched) != 1:
                missing.append(f"{run.run_id}:step={step}:view={view}:behavior")
                continue
            row = matched.iloc[0]
            required = ["rho_y", "rho_p", "rho_q"]
            needs_causal = view in causal_views if require_causal is None else require_causal
            if needs_causal:
                required.extend(f"causal_{name}" for name in causal_targets)
            absent = [name for name in required if name not in matched or pd.isna(row.get(name))]
            if absent:
                missing.append(f"{run.run_id}:step={step}:view={view}:missing={','.join(absent)}")
    if missing:
        preview = "; ".join(missing[:8])
        raise PanelCompletenessError(
            f"checkpoint evaluation panel is incomplete ({len(missing)} omissions): {preview}"
        )


def seed_level_trajectories(
    panel: AnalysisPanel,
    *,
    behavior_panel: str = "conflict",
    prompt_view: str = "full",
    validate: bool = True,
    require_causal: bool = True,
) -> pd.DataFrame:
    """Return one checkpoint row per trained seed for a diagnostic panel."""

    if validate:
        validate_trajectory_completeness(
            panel,
            behavior_panel=behavior_panel,
            prompt_views=(prompt_view,),
            require_causal=require_causal,
        )
    candidates = panel.trajectory[
        (panel.trajectory["panel"] == behavior_panel) & (panel.trajectory["prompt_view"] == prompt_view)
    ].copy()
    selected: list[pd.DataFrame] = []
    for run in panel.runs.itertuples(index=False):
        rows = candidates[candidates["run_id"] == run.run_id]
        final_step = int(get_path(run.config, "train.steps", -1))
        for step, at_step in rows.groupby("step", sort=True):
            preferred = "final_factorial" if int(step) == final_step else "diagnostic_factorial"
            if preferred in set(at_step["split"]):
                selected.append(at_step[at_step["split"] == preferred])
            elif at_step["split"].nunique() == 1:
                selected.append(at_step)
            else:
                raise PanelCompletenessError(
                    f"cannot identify diagnostic split for {run.run_id} at step {step}"
                )
    frame = pd.concat(selected, ignore_index=True) if selected else candidates.iloc[0:0].copy()
    config_columns = [
        column
        for column in panel.runs.columns
        if column.startswith("config.") and column not in frame.columns
    ]
    frame = frame.merge(
        panel.runs[["run_id", *config_columns]],
        on="run_id",
        how="left",
        validate="many_to_one",
    )
    return frame.sort_values(["cell_id", "seed", "step"]).reset_index(drop=True)


@dataclass(frozen=True)
class ControllerCriteria:
    behavioral_threshold: float = 0.90
    causal_threshold: float = 0.50
    behavioral_winner_margin: float = 0.10
    causal_winner_margin: float = 0.10
    persistence: int = 2
    conditionality_threshold: float = 0.25

    def __post_init__(self) -> None:
        for name in (
            "behavioral_threshold",
            "causal_threshold",
            "behavioral_winner_margin",
            "causal_winner_margin",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0,1]")
        if isinstance(self.persistence, bool) or self.persistence < 1:
            raise ValueError("persistence must be a positive integer")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> ControllerCriteria:
        """Resolve the preregistered controller rule from a run configuration."""

        return cls(
            behavioral_threshold=float(get_path(config, "evaluation.acquisition_threshold", 0.90)),
            causal_threshold=float(get_path(config, "evaluation.causal_flip_threshold", 0.50)),
            behavioral_winner_margin=float(get_path(config, "evaluation.behavioral_controller_margin", 0.10)),
            causal_winner_margin=float(get_path(config, "evaluation.causal_controller_margin", 0.10)),
            persistence=int(get_path(config, "evaluation.persistence", 2)),
            conditionality_threshold=float(get_path(config, "evaluation.conditionality_threshold", 0.25)),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "behavioral_threshold": self.behavioral_threshold,
            "causal_threshold": self.causal_threshold,
            "behavioral_winner_margin": self.behavioral_winner_margin,
            "causal_winner_margin": self.causal_winner_margin,
            "persistence": self.persistence,
            "conditionality_threshold": self.conditionality_threshold,
        }


def _persistent_labels(labels: Sequence[str], persistence: int) -> list[str | None]:
    result: list[str | None] = []
    previous: str | None = None
    run_length = 0
    for label in labels:
        if label in {"Y", "P", "Q"} and label == previous:
            run_length += 1
        elif label in {"Y", "P", "Q"}:
            previous = label
            run_length = 1
        else:
            previous = None
            run_length = 0
        result.append(label if run_length >= persistence else None)
    return result


def classify_controllers(
    trajectory: pd.DataFrame,
    *,
    criteria: ControllerCriteria | None = None,
    walsh: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Classify candidate rule control using behavior plus matched interventions.

    A label is not promoted to ``controller`` until it has met the criteria at
    two consecutive saved checkpoints by default.  The first row in a streak
    therefore remains unconfirmed rather than being back-filled.
    """

    criteria = criteria or ControllerCriteria()
    required = {"run_id", "seed", "step", "rho_y", "rho_p", "rho_q", "causal_y", "causal_p", "causal_q"}
    missing = sorted(required - set(trajectory.columns))
    if missing:
        raise PanelCompletenessError(f"controller classification is missing columns: {missing}")
    frame = trajectory.copy()
    if walsh is not None and not walsh.empty:
        join_keys = [
            key
            for key in ("run_id", "step", "prompt_view", "split")
            if key in frame.columns and key in walsh.columns
        ]
        frame = frame.merge(walsh, on=join_keys, how="left", validate="many_to_one")

    raw_labels: list[str] = []
    tolerance = 1e-12
    for index, row in frame.iterrows():
        # The distractor is retained as a negative control.  It is never
        # subtracted from a named rule's raw action-flip rate.
        frame.loc[index, "distractor_flip_rate"] = row.get("causal_d", np.nan)
        behavioral_values: dict[str, float] = {}
        causal_values: dict[str, float] = {}
        for upper, lower in (("Y", "y"), ("P", "p"), ("Q", "q")):
            rho = row.get(f"rho_{lower}")
            causal = row.get(f"causal_{lower}")
            if not pd.isna(rho):
                behavioral_values[upper] = float(rho)
            if not pd.isna(causal):
                causal_values[upper] = float(causal)

        eligible: list[str] = []
        for upper, lower in (("Y", "y"), ("P", "p"), ("Q", "q")):
            if upper not in behavioral_values or upper not in causal_values:
                continue
            behavioral_runner_up = max(
                (value for name, value in behavioral_values.items() if name != upper),
                default=-math.inf,
            )
            causal_runner_up = max(
                (value for name, value in causal_values.items() if name != upper),
                default=-math.inf,
            )
            behavioral_margin = behavioral_values[upper] - behavioral_runner_up
            causal_margin = causal_values[upper] - causal_runner_up
            frame.loc[index, f"behavioral_winner_margin_{lower}"] = behavioral_margin
            frame.loc[index, f"causal_winner_margin_{lower}"] = causal_margin
            if (
                behavioral_values[upper] + tolerance >= criteria.behavioral_threshold
                and causal_values[upper] + tolerance >= criteria.causal_threshold
                and behavioral_margin + tolerance >= criteria.behavioral_winner_margin
                and causal_margin + tolerance >= criteria.causal_winner_margin
            ):
                eligible.append(upper)

        conditionality = row.get("conditionality_index", np.nan)
        if len(eligible) == 1:
            raw_labels.append(eligible[0])
        elif (
            not pd.isna(conditionality) and float(conditionality) >= criteria.conditionality_threshold
        ) or len(eligible) > 1:
            raw_labels.append("conditional_or_mixed")
        elif not behavioral_values or not causal_values:
            raw_labels.append("unclassified")
        else:
            raw_labels.append("none")
    frame["raw_controller"] = raw_labels
    frame["controller"] = None
    group_columns = [column for column in ("run_id", "prompt_view", "panel") if column in frame.columns]
    for _keys, indices in frame.sort_values("step").groupby(group_columns, sort=False).groups.items():
        ordered = frame.loc[list(indices)].sort_values("step")
        labels = _persistent_labels(ordered["raw_controller"].tolist(), criteria.persistence)
        frame.loc[ordered.index, "controller"] = labels
    return frame.sort_values(["run_id", "step"]).reset_index(drop=True)


def acquisition_intervals(
    classified: pd.DataFrame,
    candidate: str,
    *,
    persistence: int = 2,
    group_columns: Sequence[str] = ("run_id", "prompt_view", "panel"),
) -> pd.DataFrame:
    """Return first-acquisition intervals without interpolating between checkpoints."""

    label = str(candidate).upper()
    if label not in {"Y", "P", "Q"}:
        raise ValueError("candidate must be Y, P, or Q")
    if isinstance(persistence, bool) or persistence < 1:
        raise ValueError("persistence must be a positive integer")
    required = {"step", "raw_controller", *group_columns}
    if not required.issubset(classified.columns):
        raise PanelCompletenessError(
            f"acquisition table is missing: {sorted(required - set(classified.columns))}"
        )
    rows: list[dict[str, Any]] = []
    for keys, group in classified.groupby(list(group_columns), dropna=False, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        ordered = group.sort_values("step")
        steps = ordered["step"].astype(int).tolist()
        labels = ordered["raw_controller"].tolist()
        onset = None
        for index in range(len(labels) - persistence + 1):
            if labels[index : index + persistence] == [label] * persistence:
                onset = index
                break
        common = dict(zip(group_columns, keys, strict=True))
        if onset is not None:
            rows.append(
                {
                    **common,
                    "candidate": label,
                    "left_step": np.nan if onset == 0 else steps[onset - 1],
                    "right_step": steps[onset],
                    "confirmation_step": steps[onset + persistence - 1],
                    "censoring": "left" if onset == 0 else "interval",
                }
            )
            continue
        trailing = 0
        for observed in reversed(labels):
            if observed == label:
                trailing += 1
            else:
                break
        lower_index = max(0, len(steps) - trailing - 1) if trailing else len(steps) - 1
        rows.append(
            {
                **common,
                "candidate": label,
                "left_step": steps[lower_index],
                "right_step": np.nan,
                "confirmation_step": np.nan,
                "censoring": "right_unconfirmed_tail" if trailing else "right",
            }
        )
    return pd.DataFrame(rows)


def _stable_rng_seed(label: str) -> int:
    return int(stable_hash({"analysis_rng": label}, 16), 16) % (2**32 - 1)


def bootstrap_mean_interval(
    values: Sequence[float] | np.ndarray,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = 0,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        raise ValueError("bootstrap values must contain at least one finite observation")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0,1)")
    if isinstance(draws, bool) or draws < 1:
        raise ValueError("draws must be a positive integer")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, array.size, size=(int(draws), array.size))
    estimates = array[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "estimate": float(array.mean()),
        "ci_low": float(np.quantile(estimates, alpha)),
        "ci_high": float(np.quantile(estimates, 1.0 - alpha)),
        "confidence": confidence,
        "bootstrap_draws": int(draws),
        "n_seeds": int(array.size),
        "replication_unit": "training_seed",
        "status": "estimated" if array.size >= MIN_INFERENTIAL_SEEDS else "descriptive_only",
    }


def seed_block_bootstrap(
    frame: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    *,
    seed_column: str = "seed",
    confidence: float = DEFAULT_CONFIDENCE,
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    random_seed: int = 0,
) -> dict[str, Any]:
    """Resample whole trained-seed blocks, retaining all within-seed rows."""

    if seed_column not in frame or frame.empty:
        raise ValueError(f"non-empty frame must contain {seed_column!r}")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0,1)")
    if isinstance(draws, bool) or draws < 1:
        raise ValueError("draws must be a positive integer")
    seeds = sorted(frame[seed_column].drop_duplicates().tolist(), key=repr)
    if not seeds:
        raise ValueError("no seeds available for block bootstrap")
    estimate = float(statistic(frame.copy()))
    rng = np.random.default_rng(int(random_seed))
    estimates: list[float] = []
    blocks = {seed: frame[frame[seed_column] == seed] for seed in seeds}
    for _draw in range(int(draws)):
        sampled = rng.integers(0, len(seeds), size=len(seeds))
        pieces: list[pd.DataFrame] = []
        for block_index, sampled_index in enumerate(sampled):
            piece = blocks[seeds[int(sampled_index)]].copy()
            piece["_bootstrap_seed_block"] = block_index
            pieces.append(piece)
        estimates.append(float(statistic(pd.concat(pieces, ignore_index=True))))
    alpha = (1.0 - confidence) / 2.0
    return {
        "estimate": estimate,
        "ci_low": float(np.quantile(estimates, alpha)),
        "ci_high": float(np.quantile(estimates, 1.0 - alpha)),
        "confidence": confidence,
        "bootstrap_draws": int(draws),
        "n_seeds": len(seeds),
        "replication_unit": "training_seed",
        "status": "estimated" if len(seeds) >= MIN_INFERENTIAL_SEEDS else "descriptive_only",
    }


def sign_flip_pvalue(
    differences: Sequence[float] | np.ndarray,
    *,
    draws: int = 100_000,
    seed: int = 0,
) -> float:
    """Two-sided paired randomization p-value under exchangeable signs."""

    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("differences must contain a finite value")
    observed = abs(float(values.mean()))
    tolerance = 1e-15
    if values.size <= 16:
        signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=values.size)))
        null = np.abs((signs * values).mean(axis=1))
        return float(np.mean(null >= observed - tolerance))
    rng = np.random.default_rng(int(seed))
    exceed = 0
    completed = 0
    batch_size = 10_000
    while completed < draws:
        count = min(batch_size, draws - completed)
        signs = rng.choice((-1.0, 1.0), size=(count, values.size))
        null = np.abs((signs * values).mean(axis=1))
        exceed += int(np.sum(null >= observed - tolerance))
        completed += count
    return float((exceed + 1) / (draws + 1))


def equivalence_interval_90(
    differences: Sequence[float] | np.ndarray,
    *,
    margin: float = DEFAULT_EQUIVALENCE_MARGIN,
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = 0,
) -> dict[str, Any]:
    """A 90% seed-bootstrap interval and a conservative equivalence decision."""

    if not 0.0 < margin < 1.0:
        raise ValueError("equivalence margin must lie in (0,1)")
    result = bootstrap_mean_interval(
        differences,
        confidence=0.90,
        draws=draws,
        seed=seed,
    )
    inferential = result["n_seeds"] >= MIN_INFERENTIAL_SEEDS
    result.update(
        {
            "equivalence_margin": margin,
            "equivalent": (
                bool(result["ci_low"] > -margin and result["ci_high"] < margin) if inferential else None
            ),
            "decision_rule": "entire 90% interval strictly inside [-margin,+margin]",
        }
    )
    return result


@dataclass(frozen=True)
class PairedContrast:
    per_seed: pd.DataFrame
    summary: Mapping[str, Any]


def paired_seed_contrast(
    frame: pd.DataFrame,
    *,
    value_column: str,
    condition_column: str,
    treatment: Any,
    control: Any,
    seed_column: str = "seed",
    pair_columns: Sequence[str] = (),
    strict: bool = True,
    bootstrap_draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    bootstrap_seed: int = 0,
    equivalence_margin: float = DEFAULT_EQUIVALENCE_MARGIN,
) -> PairedContrast:
    """Collapse matched cells within seed before any inferential calculation."""

    required = {value_column, condition_column, seed_column, *pair_columns}
    if not required.issubset(frame.columns):
        raise ValueError(f"contrast frame is missing: {sorted(required - set(frame.columns))}")
    selected = frame[frame[condition_column].isin([treatment, control])].copy()
    keys = [seed_column, *pair_columns, condition_column]
    collapsed = selected.groupby(keys, dropna=False, as_index=False)[value_column].mean()
    index = [seed_column, *pair_columns]
    wide = collapsed.pivot(index=index, columns=condition_column, values=value_column)
    missing_conditions = [value for value in (treatment, control) if value not in wide.columns]
    if missing_conditions:
        raise PanelCompletenessError(f"contrast lacks conditions: {missing_conditions}")
    incomplete = wide[[treatment, control]].isna().any(axis=1)
    if strict and incomplete.any():
        raise PanelCompletenessError(f"paired contrast has {int(incomplete.sum())} unmatched seed/cell rows")
    wide = wide.loc[~incomplete, [treatment, control]].copy()
    wide["difference"] = wide[treatment] - wide[control]
    per_seed = wide.groupby(level=seed_column)["difference"].mean().reset_index()
    differences = per_seed["difference"].to_numpy(dtype=float)
    interval = bootstrap_mean_interval(
        differences,
        confidence=DEFAULT_CONFIDENCE,
        draws=bootstrap_draws,
        seed=bootstrap_seed,
    )
    equivalence = equivalence_interval_90(
        differences,
        margin=equivalence_margin,
        draws=bootstrap_draws,
        seed=bootstrap_seed + 1,
    )
    summary = {
        **interval,
        "treatment": treatment,
        "control": control,
        "p_value_sign_flip": sign_flip_pvalue(differences, seed=bootstrap_seed + 2),
        "equivalence_90": equivalence,
        "pair_columns": list(pair_columns),
    }
    return PairedContrast(per_seed=per_seed, summary=summary)


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm family-wise correction with monotonic adjusted p-values."""

    finite: list[tuple[str, float]] = []
    result: dict[str, float] = {}
    for name, value in p_values.items():
        numeric = float(value)
        if math.isnan(numeric):
            result[str(name)] = float("nan")
        elif not 0.0 <= numeric <= 1.0:
            raise ValueError(f"p-value for {name!r} must lie in [0,1]")
        else:
            finite.append((str(name), numeric))
    finite.sort(key=lambda item: item[1])
    running = 0.0
    count = len(finite)
    for rank, (name, value) in enumerate(finite):
        adjusted = min(1.0, (count - rank) * value)
        running = max(running, adjusted)
        result[name] = running
    return {str(name): result[str(name)] for name in p_values}


_WALSH_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("intercept", ()),
    ("y", ("y",)),
    ("p", ("p",)),
    ("q", ("q",)),
    ("y_p", ("y", "p")),
    ("y_q", ("y", "q")),
    ("p_q", ("p", "q")),
    ("y_p_q", ("y", "p", "q")),
)


def walsh_coefficients(
    factorial: pd.DataFrame,
    *,
    group_columns: Sequence[str] = ("run_id", "seed", "step", "prompt_view"),
    value_column: str = "action_b_rate",
    purity_tolerance: float = 0.10,
) -> pd.DataFrame:
    """Fit the saturated, assumption-free 2^3 truth table on action tendency.

    Choices A/B are encoded as -1/+1 and action probability as ``2*p(B)-1``.
    The eight Walsh coefficients reconstruct all eight cell means exactly.  An
    interaction coefficient describes a conditional rule; it is not evidence
    for any particular internal representation.
    """

    required = {*group_columns, "choice_y", "choice_p", "choice_q", value_column}
    if not required.issubset(factorial.columns):
        raise PanelCompletenessError(
            f"factorial table is missing: {sorted(required - set(factorial.columns))}"
        )
    if "split" in factorial and "split" not in group_columns and factorial["split"].nunique() > 1:
        raise PanelCompletenessError(
            "factorial table contains multiple evaluation splits; filter it or include split in group_columns"
        )
    rows: list[dict[str, Any]] = []
    for keys, group in factorial.groupby(list(group_columns), dropna=False, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        cells = (
            group.groupby(["choice_y", "choice_p", "choice_q"], as_index=False)[value_column]
            .mean()
            .sort_values(["choice_y", "choice_p", "choice_q"])
        )
        observed_cells = {
            tuple(int(value) for value in row)
            for row in cells[["choice_y", "choice_p", "choice_q"]].to_numpy()
        }
        expected_cells = set(itertools.product((0, 1), repeat=3))
        if observed_cells != expected_cells or len(cells) != 8:
            raise PanelCompletenessError(f"factorial group {keys} lacks the complete 2^3 diagnostic panel")
        encoded = {
            name: 2.0 * cells[f"choice_{name}"].to_numpy(dtype=float) - 1.0 for name in ("y", "p", "q")
        }
        response = 2.0 * cells[value_column].to_numpy(dtype=float) - 1.0
        if np.any((response < -1.0 - 1e-9) | (response > 1.0 + 1e-9)):
            raise ValueError(f"{value_column} must lie in [0,1]")
        coefficients: dict[str, float] = {}
        reconstruction: NDArray[np.float64] = np.zeros(8, dtype=float)
        for name, factors in _WALSH_TERMS:
            basis: NDArray[np.float64] = np.ones(8, dtype=float)
            for factor in factors:
                basis *= encoded[factor]
            coefficient = float(np.mean(response * basis))
            coefficients[name] = coefficient
            reconstruction += coefficient * basis
        main_mass = sum(abs(coefficients[name]) for name in ("y", "p", "q"))
        interaction_mass = sum(abs(coefficients[name]) for name in ("y_p", "y_q", "p_q", "y_p_q"))
        nonconstant_mass = main_mass + interaction_mass
        conditionality = interaction_mass / nonconstant_mass if nonconstant_mass else 0.0
        nonintercept = {name: value for name, value in coefficients.items() if name != "intercept"}
        dominant = max(nonintercept, key=lambda name: abs(nonintercept[name]))
        other_max = max(abs(value) for name, value in nonintercept.items() if name != dominant)
        if (
            dominant in {"y", "p", "q"}
            and abs(coefficients[dominant]) >= 1 - purity_tolerance
            and other_max <= purity_tolerance
        ):
            direction = "pure" if coefficients[dominant] > 0 else "inverse"
            structure = f"{direction}_{dominant.upper()}"
        elif interaction_mass > purity_tolerance:
            structure = "conditional_or_interaction"
        elif nonconstant_mass <= purity_tolerance:
            structure = "constant_or_side_bias"
        else:
            structure = "mixed_additive"
        rows.append(
            {
                **dict(zip(group_columns, keys, strict=True)),
                **{f"walsh_{name}": value for name, value in coefficients.items()},
                "dominant_walsh_term": dominant,
                "main_effect_mass": main_mass,
                "interaction_mass": interaction_mass,
                "conditionality_index": conditionality,
                "truth_table_structure": structure,
                "max_reconstruction_error": float(np.max(np.abs(reconstruction - response))),
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class AnalysisExports:
    """Paths written by :func:`export_analysis_tables`."""

    output_dir: Path
    files: Mapping[str, Path]

    def as_dict(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "files": {name: str(path) for name, path in sorted(self.files.items())},
        }


def _csv_ready(frame: pd.DataFrame) -> pd.DataFrame:
    """Encode structured object columns deterministically for a portable CSV."""

    result = frame.copy()
    for column in result.select_dtypes(include=["object"]).columns:
        result[column] = result[column].map(
            lambda value: (
                json.dumps(value, sort_keys=True, separators=(",", ":"))
                if isinstance(value, (Mapping, list, tuple))
                else value
            )
        )
    return result


def _write_frame(path: Path, frame: pd.DataFrame) -> str:
    encoded = _csv_ready(frame)
    payload = encoded.to_csv(index=False, lineterminator="\n")
    atomic_text(path, payload)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _diagnostic_factorial(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise PanelCompletenessError("no factorial evaluation rows are available")
    if "split" not in frame:
        return frame.copy()
    names = frame["split"].astype(str).str.lower()
    selected = frame[names.str.contains("factorial", regex=False)].copy()
    if selected.empty:
        raise PanelCompletenessError("no diagnostic factorial split is available for truth-table analysis")
    return selected


def _criteria_for_cell(panel: AnalysisPanel, cell_id: str) -> ControllerCriteria:
    configs = panel.runs.loc[panel.runs["cell_id"] == cell_id, "config"].tolist()
    if not configs:
        raise PanelCompletenessError(f"analysis cell {cell_id!r} has no run configuration")
    criteria = [ControllerCriteria.from_config(config) for config in configs]
    signatures = {stable_hash(value.as_dict(), 64) for value in criteria}
    if len(signatures) != 1:
        raise ProvenanceError(f"controller classification thresholds vary across seeds in cell {cell_id}")
    return criteria[0]


def export_analysis_tables(
    panel: AnalysisPanel,
    output_dir: str | Path,
    *,
    behavior_panel: str = "conflict",
    prompt_view: str = "full",
) -> AnalysisExports:
    """Write claim-neutral, seed-level analysis tables with an audit manifest.

    The export fails if checkpoint, causal, or factorial panels are incomplete.
    It does not choose scientific contrasts automatically: paired contrasts and
    multiplicity families must be specified by the experiment protocol.
    """

    target = Path(output_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    trajectory = seed_level_trajectories(
        panel,
        behavior_panel=behavior_panel,
        prompt_view=prompt_view,
        validate=True,
    )
    diagnostic_factorial = _diagnostic_factorial(panel.factorial)
    walsh = walsh_coefficients(
        diagnostic_factorial,
        group_columns=(
            "run_id",
            "cell_id",
            "seed",
            "step",
            "prompt_view",
            "split",
        ),
    )

    classified_parts: list[pd.DataFrame] = []
    acquisition_parts: list[pd.DataFrame] = []
    criteria_by_cell: dict[str, Any] = {}
    for cell_id in sorted(trajectory["cell_id"].astype(str).unique()):
        criteria = _criteria_for_cell(panel, cell_id)
        criteria_by_cell[cell_id] = criteria.as_dict()
        cell_trajectory = trajectory[trajectory["cell_id"].astype(str) == cell_id]
        cell_walsh = walsh[walsh["cell_id"].astype(str) == cell_id]
        classified = classify_controllers(
            cell_trajectory,
            criteria=criteria,
            walsh=cell_walsh,
        )
        classified_parts.append(classified)
        group_columns = tuple(
            column for column in ("run_id", "cell_id", "seed", "prompt_view", "panel") if column in classified
        )
        for candidate in ("Y", "P", "Q"):
            acquisition_parts.append(
                acquisition_intervals(
                    classified,
                    candidate,
                    persistence=criteria.persistence,
                    group_columns=group_columns,
                )
            )
    controllers = pd.concat(classified_parts, ignore_index=True)
    acquisitions = pd.concat(acquisition_parts, ignore_index=True)

    tables: dict[str, pd.DataFrame] = {
        "runs": panel.runs,
        "seed_trajectories": trajectory,
        "causal_effects": panel.causal_effects,
        "factorial_cells": diagnostic_factorial,
        "walsh_coefficients": walsh,
        "controller_classifications": controllers,
        "acquisition_intervals": acquisitions,
    }
    files: dict[str, Path] = {}
    table_manifest: dict[str, Any] = {}
    for name, frame in tables.items():
        path = target / f"{name}.csv"
        digest = _write_frame(path, frame)
        files[name] = path
        table_manifest[name] = {
            "file": path.name,
            "rows": len(frame),
            "columns": list(frame.columns),
            "sha256": digest,
        }

    audit_path = target / "analysis-manifest.json"
    write_json(
        audit_path,
        {
            "analysis_contract": {
                "replication_unit": "training_seed",
                "behavior_panel": behavior_panel,
                "prompt_view": prompt_view,
                "acquisition_times": "interval_censored_between_saved_checkpoints",
                "controller_requires_persistence": True,
                "factorial_model": "saturated_2^3_Walsh_transform",
                "automatic_scientific_contrasts": False,
                "minimum_inferential_seeds": MIN_INFERENTIAL_SEEDS,
                "equivalence_interval": "90% seed bootstrap",
                "multiple_testing": "Holm correction available; family must be protocol-defined",
            },
            "artifact_panel": dict(panel.audit),
            "controller_criteria_by_cell": criteria_by_cell,
            "tables": table_manifest,
        },
    )
    files["manifest"] = audit_path
    return AnalysisExports(output_dir=target, files=files)


@dataclass(frozen=True)
class ConfirmatoryExports:
    output_dir: Path
    files: Mapping[str, Path]

    def as_dict(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "files": {name: str(path) for name, path in sorted(self.files.items())},
        }


def _final_prompt_behavior(panel: AnalysisPanel, prompt_view: str) -> pd.DataFrame:
    rows: list[pd.Series[Any]] = []
    for run in panel.runs.itertuples(index=False):
        final_step = int(get_path(run.config, "train.steps", 0))
        selected = panel.trajectory[
            (panel.trajectory["run_id"] == run.run_id)
            & (panel.trajectory["step"] == final_step)
            & (panel.trajectory["prompt_view"] == prompt_view)
            & (panel.trajectory["panel"] == "conflict")
        ]
        if "split" in selected and "final_factorial" in set(selected["split"]):
            selected = selected[selected["split"] == "final_factorial"]
        if len(selected) != 1:
            raise PanelCompletenessError(
                f"final {prompt_view} endpoint needs one conflict row for {run.run_id}"
            )
        rows.append(selected.iloc[0])
    if not rows:
        return pd.DataFrame(columns=["run_id", "rho_y", "rho_p", "rho_q"])
    result = pd.DataFrame(rows).reset_index(drop=True)
    metadata = panel.runs[
        [
            "run_id",
            "config.data.rule_family",
            "config.data.q_p",
            "config.data.q_q",
            "config.train.algorithm",
        ]
    ].rename(
        columns={
            "config.data.rule_family": "law_family",
            "config.data.q_p": "q_p",
            "config.data.q_q": "q_q",
            "config.train.algorithm": "algorithm",
        }
    )
    return result.merge(metadata, on="run_id", how="left", validate="one_to_one")


def _final_iid_accuracy(panel: AnalysisPanel) -> dict[str, float]:
    """Extract the predeclared final IID action-accuracy manipulation check."""

    result: dict[str, float] = {}
    for run in panel.runs.itertuples(index=False):
        final_step = int(get_path(run.config, "train.steps", 0))
        rows = panel.raw_metrics[
            (panel.raw_metrics["run_id"] == run.run_id)
            & (panel.raw_metrics["step"] == final_step)
            & (panel.raw_metrics["split"] == "iid_validation")
            & (panel.raw_metrics["prompt_view"] == "full")
            & (panel.raw_metrics["kind"] == "behavioral_agreement")
            & (panel.raw_metrics["panel"] == "all")
        ]
        values: list[float] = []
        for record in rows.to_dict(orient="records"):
            value = record.get("rho_y", record.get("agreement_y", record.get("accuracy")))
            if value is not None and not pd.isna(value):
                values.append(float(value))
        unique = sorted(set(values))
        if len(unique) != 1:
            raise PanelCompletenessError(
                f"final IID accuracy needs one value for {run.run_id}; observed {unique}"
            )
        result[str(run.run_id)] = unique[0]
    return result


def _confirmatory_acquisitions(panel: AnalysisPanel) -> pd.DataFrame:
    if panel.runs.empty:
        return pd.DataFrame()
    trajectory = seed_level_trajectories(panel, prompt_view="full", require_causal=True)
    factorial = _diagnostic_factorial(panel.factorial)
    walsh = walsh_coefficients(
        factorial,
        group_columns=("run_id", "cell_id", "seed", "step", "prompt_view", "split"),
    )
    pieces: list[pd.DataFrame] = []
    for cell_id in sorted(trajectory["cell_id"].astype(str).unique()):
        criteria = _criteria_for_cell(panel, cell_id)
        classified = classify_controllers(
            trajectory[trajectory["cell_id"].astype(str) == cell_id],
            criteria=criteria,
            walsh=walsh[walsh["cell_id"].astype(str) == cell_id],
        )
        groups = tuple(
            column for column in ("run_id", "cell_id", "seed", "prompt_view", "panel") if column in classified
        )
        for candidate in ("Y", "P", "Q"):
            pieces.append(
                acquisition_intervals(
                    classified,
                    candidate,
                    persistence=criteria.persistence,
                    group_columns=groups,
                )
            )
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def _endpoint_units(
    panel: AnalysisPanel,
    ledger: pd.DataFrame,
    acquisitions: pd.DataFrame,
) -> pd.DataFrame:
    full = _final_prompt_behavior(panel, "full")
    law_only = _final_prompt_behavior(panel, "law_only")
    full_lookup = full.set_index("run_id") if not full.empty else full
    law_lookup = law_only.set_index("run_id") if not law_only.empty else law_only
    iid_accuracy = _final_iid_accuracy(panel)
    acquisition_lookup = (
        acquisitions.set_index(["run_id", "candidate"]) if not acquisitions.empty else acquisitions
    )
    rows: list[dict[str, Any]] = []

    def common(record: Any, endpoint: str, observed: bool) -> dict[str, Any]:
        return {
            "endpoint": endpoint,
            "unit_id": str(record.plan_key),
            "plan_key": record.plan_key,
            "run_id": record.run_id,
            "seed": int(record.seed),
            "law_family": record.law_family,
            "q_p": float(record.q_p),
            "q_q": float(record.q_q),
            "algorithm": record.algorithm,
            "run_state": record.state,
            "outcome_observed": observed,
            "itt_imputed": not observed,
        }

    for record in ledger.itertuples(index=False):
        observed = bool(record.state == "complete")
        run_id = str(record.run_id)
        if math.isclose(float(record.q_p), 1.0, abs_tol=1e-12):
            if observed:
                final = full_lookup.loc[run_id]
                value = float(final["rho_p"] - final["rho_y"])
                secondary = float(final["causal_p"] - final["causal_y"])
                iid_value = iid_accuracy[run_id]
            else:
                value = secondary = 0.0
                iid_value = np.nan
            rows.append(
                {
                    **common(record, "dissociation", observed),
                    "value": value,
                    "secondary_value": secondary,
                    "null_value": 0.0,
                    "missing_lower": -1.0,
                    "missing_upper": 1.0,
                    "prompt_view": "full",
                    "iid_accuracy": iid_value,
                    "qualified_iid_performance": bool(iid_value > 0.98) if observed else False,
                }
            )

        if math.isclose(float(record.q_p), 0.95, abs_tol=1e-12):
            p_interval = None
            y_interval = None
            q_interval = None
            if observed and not acquisitions.empty:
                if (run_id, "P") in acquisition_lookup.index:
                    p_interval = acquisition_lookup.loc[(run_id, "P")]
                if (run_id, "Y") in acquisition_lookup.index:
                    y_interval = acquisition_lookup.loc[(run_id, "Y")]
                if (run_id, "Q") in acquisition_lookup.index:
                    q_interval = acquisition_lookup.loc[(run_id, "Q")]
            p_right = None if p_interval is None else p_interval.get("right_step")
            y_right = None if y_interval is None else y_interval.get("right_step")
            q_right = None if q_interval is None else q_interval.get("right_step")
            p_acquired = p_right is not None and pd.notna(p_right)
            y_acquired = y_right is not None and pd.notna(y_right)
            q_acquired = q_right is not None and pd.notna(q_right)
            competitor_steps: list[float] = []
            if y_acquired:
                assert y_right is not None
                competitor_steps.append(float(y_right))
            if q_acquired:
                assert q_right is not None
                competitor_steps.append(float(q_right))
            if p_acquired and competitor_steps:
                assert p_right is not None
                p_step = float(p_right)
                competitor_step = min(competitor_steps)
                value = 1.0 if p_step < competitor_step else (0.0 if p_step > competitor_step else 0.5)
                censoring = "herald_and_semantic_candidate_acquired"
            elif p_acquired:
                value, censoring = 1.0, "semantic_candidates_right_censored"
            elif competitor_steps:
                value, censoring = 0.0, "herald_right_censored"
            else:
                value, censoring = 0.5, "both_right_censored_or_failed"
            rows.append(
                {
                    **common(record, "acquisition_order", observed),
                    "value": value,
                    "secondary_value": np.nan,
                    "null_value": 0.5,
                    "missing_lower": 0.0,
                    "missing_upper": 1.0,
                    "prompt_view": "full",
                    "censoring": censoring,
                    "p_right_step": p_right,
                    "y_right_step": y_right,
                    "q_right_step": q_right,
                }
            )

        if observed:
            full_row = full_lookup.loc[run_id]
            law_row = law_lookup.loc[run_id]
            availability = float(law_row["rho_y"] - full_row["rho_y"])
        else:
            availability = 0.0
        rows.append(
            {
                **common(record, "availability_use", observed),
                "value": availability,
                "secondary_value": np.nan,
                "null_value": 0.0,
                "missing_lower": -1.0,
                "missing_upper": 1.0,
                "prompt_view": "law_only-minus-full",
            }
        )

    algorithm_keys = ["seed", "law_family", "q_p", "q_q"]
    for keys, group in ledger.groupby(algorithm_keys, dropna=False, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        metadata = dict(zip(algorithm_keys, keys, strict=True))
        algorithms = {row.algorithm: row for row in group.itertuples(index=False)}
        observed = (
            set(algorithms) == {"sft", "outcome_rl"}
            and algorithms["sft"].state == "complete"
            and algorithms["outcome_rl"].state == "complete"
        )
        if observed:
            sft = full_lookup.loc[str(algorithms["sft"].run_id)]
            rl = full_lookup.loc[str(algorithms["outcome_rl"].run_id)]
            value = float(rl["rho_y"] - sft["rho_y"])
        else:
            value = 0.0
        rows.append(
            {
                "endpoint": "algorithm_control",
                "unit_id": stable_hash({**metadata, "endpoint": "algorithm_control"}, 20),
                "plan_key": None,
                "run_id": None,
                **metadata,
                "algorithm": "outcome_rl-minus-sft",
                "run_state": "paired_complete" if observed else "paired_failure_or_missing",
                "outcome_observed": observed,
                "itt_imputed": not observed,
                "value": value,
                "secondary_value": np.nan,
                "null_value": 0.0,
                "missing_lower": -1.0,
                "missing_upper": 1.0,
                "prompt_view": "full",
            }
        )
    return pd.DataFrame(rows)


def _fixed_effect_value(frame: pd.DataFrame, value_column: str = "value") -> float:
    seed_column = "_bootstrap_seed_block" if "_bootstrap_seed_block" in frame else "seed"
    per_seed_law = (
        frame.groupby(["law_family", seed_column], as_index=False)[value_column]
        .mean()
        .rename(columns={value_column: "seed_value"})
    )
    estimates: list[tuple[float, float]] = []
    for _law, group in per_seed_law.groupby("law_family", sort=True):
        values = group["seed_value"].to_numpy(dtype=float)
        variance = float(np.var(values, ddof=1) / len(values)) if len(values) > 1 else math.inf
        weight = 1.0 / variance if math.isfinite(variance) and variance > 1e-15 else float(len(values))
        estimates.append((float(values.mean()), weight))
    return sum(estimate * weight for estimate, weight in estimates) / sum(
        weight for _estimate, weight in estimates
    )


def _confirmatory_summaries(
    units: pd.DataFrame,
    *,
    bootstrap_draws: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for endpoint, endpoint_rows in units.groupby("endpoint", sort=True):
        null_value = float(endpoint_rows["null_value"].iloc[0])
        for scope in (*sorted(endpoint_rows["law_family"].dropna().unique()), "fixed_effect_pooled"):
            selected = (
                endpoint_rows
                if scope == "fixed_effect_pooled"
                else endpoint_rows[endpoint_rows["law_family"] == scope]
            )
            if selected.empty:
                continue
            per_seed = selected.groupby("seed", as_index=False)["value"].mean()
            if scope == "fixed_effect_pooled":
                interval = seed_block_bootstrap(
                    selected,
                    _fixed_effect_value,
                    draws=bootstrap_draws,
                    random_seed=_stable_rng_seed(f"{endpoint}:{scope}"),
                )
            else:
                interval = bootstrap_mean_interval(
                    per_seed["value"].to_numpy(dtype=float),
                    draws=bootstrap_draws,
                    seed=_stable_rng_seed(f"{endpoint}:{scope}"),
                )
            centered = per_seed["value"].to_numpy(dtype=float) - null_value
            p_value = sign_flip_pvalue(
                centered,
                seed=_stable_rng_seed(f"{endpoint}:{scope}:sign"),
            )
            missing = ~selected["outcome_observed"].astype(bool)
            lower_bound = selected["value"].where(~missing, selected["missing_lower"]).mean()
            upper_bound = selected["value"].where(~missing, selected["missing_upper"]).mean()
            secondary = selected["secondary_value"].dropna()
            secondary_estimate = float(secondary.mean()) if not secondary.empty else np.nan
            equivalence: Mapping[str, Any] = {}
            if endpoint == "algorithm_control":
                equivalence = equivalence_interval_90(
                    centered,
                    margin=DEFAULT_EQUIVALENCE_MARGIN,
                    draws=bootstrap_draws,
                    seed=_stable_rng_seed(f"{endpoint}:{scope}:equivalence"),
                )
            rows.append(
                {
                    "endpoint": endpoint,
                    "law_scope": scope,
                    "estimate": interval["estimate"],
                    "ci_low": interval["ci_low"],
                    "ci_high": interval["ci_high"],
                    "confidence": interval["confidence"],
                    "n_seeds": interval["n_seeds"],
                    "inference_status": interval["status"],
                    "null_value": null_value,
                    "p_value_raw": p_value,
                    "itt_missing_units": int(missing.sum()),
                    "itt_total_units": len(selected),
                    "itt_sensitivity_lower": float(lower_bound),
                    "itt_sensitivity_upper": float(upper_bound),
                    "secondary_causal_estimate": secondary_estimate,
                    "equivalence_margin": equivalence.get("equivalence_margin"),
                    "equivalence_ci_low": equivalence.get("ci_low"),
                    "equivalence_ci_high": equivalence.get("ci_high"),
                    "equivalent": equivalence.get("equivalent"),
                }
            )
    result = pd.DataFrame(rows)
    result["p_value_holm"] = np.nan
    for scope, indices in result.groupby("law_scope", sort=True).groups.items():
        family = {
            str(result.loc[index, "endpoint"]): float(result.loc[index, "p_value_raw"]) for index in indices
        }
        if len(family) != 4:
            raise PanelCompletenessError(
                f"Holm family for {scope} has {len(family)} endpoints instead of four"
            )
        adjusted = holm_adjust(family)
        for index in indices:
            result.loc[index, "p_value_holm"] = adjusted[str(result.loc[index, "endpoint"])]
    return result.sort_values(["law_scope", "endpoint"]).reset_index(drop=True)


def run_g01_confirmatory_analysis(
    root: str | Path,
    expected_config: Mapping[str, Any],
    output_dir: str | Path,
    *,
    bootstrap_draws: int = DEFAULT_BOOTSTRAP_DRAWS,
) -> ConfirmatoryExports:
    """Execute the four frozen G01 endpoints with pairing and ITT audits."""

    if str(get_path(expected_config, "experiment.id", "")) != "g01":
        raise AnalysisError("the G01 confirmatory driver requires experiment.id=g01")
    cells = build_plan(expected_config)
    if {str(get_path(spec.config, "train.algorithm", "")) for spec in cells} != {
        "sft",
        "outcome_rl",
    }:
        raise PanelCompletenessError("G01 requires paired SFT and outcome_rl cells")
    if any(
        not {"full", "law_only"}.issubset(
            set(str(view) for view in get_path(spec.config, "evaluation.prompt_views", []))
        )
        for spec in cells
    ):
        raise PanelCompletenessError("G01 endpoints require full and law_only prompt views")

    ledger = intention_to_train_ledger(root, expected_config)
    target = Path(output_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    ledger_path = target / "intention_to_train_ledger.csv"
    _write_frame(ledger_path, ledger)
    complete_paths = discover_runs(root, completed_only=True)
    if not complete_paths:
        write_json(
            target / "confirmatory-analysis-manifest.json",
            {
                "schema": "goalzendo.g01_confirmatory_analysis",
                "schema_version": 1,
                "analysis_status": "no_completed_outcomes",
                "intention_to_train_ledger": ledger_path.name,
                "state_counts": {
                    str(state): int(count) for state, count in ledger["state"].value_counts().items()
                },
            },
        )
        raise PanelCompletenessError(
            "no completed runs are available; ITT ledger states: "
            + json.dumps(ledger["state"].value_counts().to_dict(), sort_keys=True)
        )
    panel = load_analysis_panel(
        root,
        expected_config=expected_config,
        require_complete_metrics=True,
        allow_incomplete_runs=True,
    )
    integrity = paired_algorithm_integrity(panel, ledger)
    corrupted_pairs = integrity[integrity["pair_complete"] & ~integrity["integrity_passed"]]
    if not corrupted_pairs.empty:
        raise ProvenanceError(f"{len(corrupted_pairs)} completed SFT/RL pairs violate the pairing contract")
    acquisitions = _confirmatory_acquisitions(panel)
    units = _endpoint_units(panel, ledger, acquisitions)
    summaries = _confirmatory_summaries(units, bootstrap_draws=bootstrap_draws)

    tables = {
        "intention_to_train_ledger": ledger,
        "paired_algorithm_integrity": integrity,
        "acquisition_intervals": acquisitions,
        "primary_endpoint_units": units,
        "primary_endpoint_summaries": summaries,
    }
    files: dict[str, Path] = {}
    table_manifest: dict[str, Any] = {}
    for name, frame in tables.items():
        path = target / f"{name}.csv"
        digest = _write_frame(path, frame)
        files[name] = path
        table_manifest[name] = {"file": path.name, "rows": len(frame), "sha256": digest}
    manifest = target / "confirmatory-analysis-manifest.json"
    write_json(
        manifest,
        {
            "schema": "goalzendo.g01_confirmatory_analysis",
            "schema_version": 1,
            "primary_endpoints": [
                "dissociation",
                "acquisition_order",
                "availability_use",
                "algorithm_control",
            ],
            "prompt_view_endpoint": "law_only rho_Y minus full rho_Y",
            "dissociation_qualification": "final IID rho_Y > 0.98; primary ITT value is never dropped",
            "causal_primary_metric": "raw matched action_flip_rate",
            "causal_signed_margin_role": "secondary",
            "replication_unit": "training_seed",
            "bootstrap_draws": bootstrap_draws,
            "holm_family": "four primary endpoints, separately within each Law scope",
            "law_scopes": sorted(summaries["law_scope"].unique()),
            "fixed_effect_pooling": True,
            "algorithm_equivalence_margin": DEFAULT_EQUIVALENCE_MARGIN,
            "itt_missing_policy": "neutral-null point contribution plus worst/best sensitivity bounds",
            "acquisition_censoring": (
                "at q_P=.95, saved-checkpoint interval for Herald versus the earliest "
                "Sage/Law controller; unresolved order contributes 0.5"
            ),
            "artifact_panel": dict(panel.audit),
            "tables": table_manifest,
        },
    )
    files["manifest"] = manifest
    return ConfirmatoryExports(target, files)


__all__ = [
    "DEFAULT_BOOTSTRAP_DRAWS",
    "DEFAULT_CONFIDENCE",
    "DEFAULT_EQUIVALENCE_MARGIN",
    "G00_ASSESSMENT_SCHEMA",
    "G00_ASSESSMENT_SCHEMA_VERSION",
    "MIN_INFERENTIAL_SEEDS",
    "AnalysisError",
    "AnalysisExports",
    "AnalysisPanel",
    "ConfirmatoryExports",
    "ControllerCriteria",
    "PairedContrast",
    "PanelCompletenessError",
    "ProvenanceError",
    "acquisition_intervals",
    "bootstrap_mean_interval",
    "classify_controllers",
    "derive_g00_gate_assessment",
    "equivalence_interval_90",
    "export_analysis_tables",
    "flatten",
    "holm_adjust",
    "intention_to_train_ledger",
    "load_analysis_panel",
    "paired_algorithm_integrity",
    "paired_seed_contrast",
    "run_g01_confirmatory_analysis",
    "seed_block_bootstrap",
    "seed_level_trajectories",
    "sign_flip_pvalue",
    "validate_trajectory_completeness",
    "walsh_coefficients",
]
