#!/usr/bin/env python3
"""Reconstruct and plot the frozen E19 active-Q input-pathway experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

os.environ.setdefault("MPLCONFIGDIR", "/tmp/forkworld-e19-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/forkworld-e19-xdg")
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
import yaml  # type: ignore[import-untyped]
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from numpy.typing import NDArray

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forkworld.handoff import normalized_control_auc, pure_control  # noqa: E402

DEFAULT_ARTIFACTS = REPO / "artifacts-e19"
DEFAULT_GATE = HERE / "derived" / "e19_pilot_gate.json"
DEFAULT_FIGURES = HERE / "figures"
DEFAULT_DERIVED = HERE / "derived"
EXPERIMENT = "active_q_input_pathway_intervention"
CONFIG_PATH = REPO / "configs" / "e19_q_pathway_mediation.yaml"

SEEDS = (
    409,
    419,
    421,
    431,
    433,
    439,
    443,
    449,
    457,
    461,
    463,
    467,
    479,
    487,
    491,
    499,
    503,
    509,
    521,
    523,
)
BRANCHES = (
    "independent_noop",
    "independent_q_restore",
    "independent_padding_sham",
    "nested_noop",
    "nested_q_transplant",
    "nested_padding_sham",
)
CHECKPOINTS = (
    0,
    1,
    2,
    3,
    4,
    5,
    7,
    9,
    13,
    17,
    24,
    33,
    45,
    62,
    85,
    117,
    128,
    161,
    222,
    304,
    418,
    575,
    790,
    1024,
)
PLOT_CHECKPOINTS = tuple(step for step in CHECKPOINTS if step <= 128)
FROZEN_SOURCE_FINGERPRINT = (
    "ab91250cbb8379c1e6afd0f960abe0754b08a745bd2d9fd8922be0f65a1ba7a9"
)
FROZEN_CONFIG_SHA256 = (
    "79e24a5e68b95e76e3bffd61d62abfdd86f6f6c439a45c76c376a2f548b74345"
)
BOOTSTRAP_DRAWS = 4_000
CONFIDENCE = 0.95
AUC_HORIZON = 128
ACTIVE_EFFECT_THRESHOLD = 0.02
SHAM_AUC_EQUIVALENCE = 0.01
MANIPULATION_MARGIN = 0.05
MINIMUM_POSITIVE_SEEDS = 15
MINIMUM_ELIGIBLE_SEEDS = 15

INK = "#18212B"
MUTED = "#667085"
GRID = "#D9DEE7"
ACTIVE = "#286FB4"
ACTIVE_DARK = "#174E7A"
SHAM = "#C68A16"
NOOP = "#606A78"
PALE_BLUE = "#E7F0F8"
PALE_GRAY = "#EFF1F4"

SummaryGrid = dict[tuple[int, str], Mapping[str, Any]]


def get(mapping: Mapping[str, Any], path: str, default: Any = None) -> Any:
    value: Any = mapping
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value


def load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read JSON object {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise RuntimeError(f"expected JSON object: {path}")
    return cast(Mapping[str, Any], value)


def load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise RuntimeError(f"cannot read YAML mapping {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise RuntimeError(f"expected YAML mapping: {path}")
    return cast(Mapping[str, Any], value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def is_digest(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(
        float(value)
    )


def raise_audit(errors: Sequence[str]) -> None:
    preview = "\n".join(f"  - {error}" for error in errors[:80])
    suffix = "" if len(errors) <= 80 else f"\n  ... and {len(errors) - 80} more"
    raise RuntimeError(f"E19 full reconstruction failed ({len(errors)} issues):\n{preview}{suffix}")


def snapshot_control(snapshot: Mapping[str, Any], goal: str = "Q") -> float:
    return 0.5 * (
        float(get(snapshot, f"behavior.{goal}"))
        + float(get(snapshot, f"causal.{goal}"))
    )


def reconstructed_auc(summary: Mapping[str, Any]) -> float:
    snapshots = cast(Mapping[str, Mapping[str, Any]], summary["phase_b_snapshots"])
    event_input = {
        int(step): {
            "behavior": cast(Mapping[str, float], snapshot["behavior"]),
            "causal": cast(Mapping[str, float], snapshot["causal"]),
        }
        for step, snapshot in snapshots.items()
    }
    return normalized_control_auc(event_input, "Q", horizon=AUC_HORIZON)


def audit_and_load(artifacts_root: Path, gate_path: Path) -> tuple[SummaryGrid, dict[str, Any]]:
    if sha256_file(CONFIG_PATH) != FROZEN_CONFIG_SHA256:
        raise RuntimeError("E19 configuration differs from its frozen SHA-256")
    gate = load_json(gate_path)
    if (
        gate.get("decision") != "PASS"
        or gate.get("pilot_gate_passed") is not True
        or get(gate, "audit.source_fingerprint") != FROZEN_SOURCE_FINGERPRINT
        or get(gate, "audit.config_source_sha256") != FROZEN_CONFIG_SHA256
        or get(gate, "audit.complete_artifacts") != 18
    ):
        raise RuntimeError("the frozen, outcome-blind E19 engineering pilot did not pass")

    directory = artifacts_root / "h16" / EXPERIMENT
    if not directory.is_dir():
        raise RuntimeError(f"missing E19 full artifact directory {directory}")
    run_dirs = sorted(path for path in directory.iterdir() if path.is_dir())
    errors: list[str] = []
    if len(run_dirs) != len(SEEDS) * len(BRANCHES):
        errors.append(f"materialized run directories={len(run_dirs)}, expected 120")
    grid: SummaryGrid = {}
    aucs: dict[tuple[int, str], float] = {}
    metric_records = 0
    for run_dir in run_dirs:
        required = (
            "COMPLETE",
            "summary.json",
            "metadata.json",
            "resolved_config.yaml",
            "status.json",
            "metrics.jsonl",
        )
        missing = [name for name in required if not (run_dir / name).is_file()]
        if missing:
            errors.append(f"{run_dir.name}: missing {', '.join(missing)}")
            continue
        if (run_dir / "COMPLETE").read_text(encoding="utf-8") != "complete\n":
            errors.append(f"{run_dir.name}: invalid COMPLETE marker")
        summary = load_json(run_dir / "summary.json")
        metadata = load_json(run_dir / "metadata.json")
        config = load_yaml(run_dir / "resolved_config.yaml")
        status = load_json(run_dir / "status.json")
        seed = int(summary.get("seed", -1))
        branch = str(summary.get("branch", ""))
        key = (seed, branch)
        if key in grid:
            errors.append(f"{run_dir.name}: duplicate full-panel key {key}")
        grid[key] = summary
        if seed not in SEEDS or branch not in BRANCHES:
            errors.append(f"{run_dir.name}: unregistered full-panel key {key}")
        if status != {"state": "complete", "run_id": run_dir.name}:
            errors.append(f"{run_dir.name}: invalid completion status")
        if metadata.get("run_id") != run_dir.name or metadata.get("seed") != seed:
            errors.append(f"{run_dir.name}: metadata identity mismatch")
        implementation = get(metadata, "implementation", {})
        if (
            get(implementation, "implementation_fingerprint")
            != FROZEN_SOURCE_FINGERPRINT
            or get(implementation, "artifact_schema_version") != 1
            or get(implementation, "source_fingerprint_schema_version") != 1
            or get(implementation, "source_file_count") != 31
        ):
            errors.append(f"{run_dir.name}: implementation fingerprint differs from freeze")
        if (
            int(config.get("seed", -1)) != seed
            or get(config, "h16.branch") != branch
            or get(config, "h16.pilot_only") is not False
            or get(config, "run.device") != "cpu"
            or get(config, "evaluation.bootstrap_samples") != BOOTSTRAP_DRAWS
        ):
            errors.append(f"{run_dir.name}: resolved configuration identity differs")
        fixed_summary = {
            "hypothesis": "h16",
            "condition": f"active_q_input_pathway:{branch}",
            "pilot_only": False,
            "data.phase_b_constructed": True,
            "data.phase_b_pairing_verified": True,
            "data.probe_splits_disjoint": True,
            "measurement.phase_b_checkpoints": list(CHECKPOINTS),
            "measurement.direct_auc_horizon": True,
            "training.phase_a_steps_per_history": 45,
            "training.phase_b_steps": 1_024,
            "training.phase_b_examples_seen": 256_000,
            "training.all_minibatches_full": True,
            "optimizer_transition_audit.source": "fresh_empty_after_edit",
            "optimizer_transition_audit.semantic_reset_implemented_by_fresh_object": True,
            "optimizer_transition_audit.state_entry_count": 0,
            "optimizer_transition_audit.adam_step_entry_count": 0,
        }
        for path, expected in fixed_summary.items():
            if get(summary, path) != expected:
                errors.append(f"{run_dir.name}: summary.{path} differs from {expected!r}")
        for path in (
            "data.phase_b_batch_digest",
            "data.phase_b_sampler_digest",
            "data.probe_train_digest",
            "data.probe_eval_digest",
            "hashes.initial_model",
            "hashes.independent_prefix_model",
            "hashes.nested_prefix_model",
            "hashes.selected_postedit_model",
            "hashes.phase_b_initial_model",
            "hashes.phase_b_initial_optimizer",
            "hashes.phase_b_final_model",
            "hashes.phase_b_final_optimizer",
        ):
            if not is_digest(get(summary, path)):
                errors.append(f"{run_dir.name}: malformed summary.{path}")
        if get(summary, "hashes.selected_postedit_model") != get(
            summary, "hashes.phase_b_initial_model"
        ):
            errors.append(f"{run_dir.name}: phase B did not start from selected edit")
        snapshots = get(summary, "phase_b_snapshots")
        if not isinstance(snapshots, Mapping) or set(snapshots) != {
            str(step) for step in CHECKPOINTS
        }:
            errors.append(f"{run_dir.name}: phase-B direct checkpoint lattice differs")
        else:
            if summary.get("phase_b_local_zero") != snapshots["0"]:
                errors.append(f"{run_dir.name}: phase-B local-zero snapshot mismatch")
            if summary.get("postedit_snapshot") != snapshots["0"]:
                errors.append(f"{run_dir.name}: post-edit/direct-zero snapshot mismatch")
            if summary.get("final") != snapshots["1024"]:
                errors.append(f"{run_dir.name}: final/direct-1024 snapshot mismatch")
            for step in CHECKPOINTS:
                snapshot = cast(Mapping[str, Any], snapshots[str(step)])
                if (
                    snapshot.get("local_step") != step
                    or snapshot.get("global_step") != 45 + step
                    or snapshot.get("examples_seen") != (45 + step) * 250
                ):
                    errors.append(f"{run_dir.name}: malformed checkpoint counters at {step}")
                for section in ("behavior", "causal"):
                    values = snapshot.get(section)
                    if not isinstance(values, Mapping) or set(values) != {"P", "Q", "Y"}:
                        errors.append(f"{run_dir.name}: malformed {section} at {step}")
                    elif not all(finite(value) for value in values.values()):
                        errors.append(f"{run_dir.name}: non-finite {section} at {step}")
            auc = reconstructed_auc(summary)
            reported = get(
                summary,
                "outcomes.normalized_control_auc_through_direct_checkpoint.Q",
            )
            if not finite(reported) or not math.isclose(
                auc, float(reported), rel_tol=0.0, abs_tol=1e-12
            ):
                errors.append(f"{run_dir.name}: Q AUC does not reconstruct from checkpoints")
            aucs[key] = auc
        postedit = get(summary, "postedit_snapshot")
        if isinstance(postedit, Mapping):
            recalculated_pure = pure_control(
                cast(Mapping[str, float], postedit["behavior"]),
                cast(Mapping[str, float], postedit["causal"]),
                "P",
                threshold=0.90,
                margin=0.10,
            )
            if summary.get("postedit_pure_p") != recalculated_pure:
                errors.append(f"{run_dir.name}: post-edit pure-P flag does not reconstruct")
        prediction = run_dir / "predictions.jsonl"
        if prediction.exists() and prediction.stat().st_size:
            errors.append(f"{run_dir.name}: unexpected saved predictions")
        checkpoint_dir = run_dir / "checkpoints"
        if checkpoint_dir.is_dir() and any(checkpoint_dir.iterdir()):
            errors.append(f"{run_dir.name}: unexpected saved checkpoints")
        try:
            with (run_dir / "metrics.jsonl").open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    metric_records += 1
                    row = json.loads(line)
                    if (
                        not isinstance(row, Mapping)
                        or row.get("run_id") != run_dir.name
                        or row.get("seed") != seed
                        or row.get("experiment") != "h16"
                    ):
                        errors.append(f"{run_dir.name}: malformed metric identity at {line_number}")
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"{run_dir.name}: malformed metrics.jsonl: {error}")

    expected_keys = {(seed, branch) for seed in SEEDS for branch in BRANCHES}
    if set(grid) != expected_keys:
        errors.append(
            f"full grid missing={sorted(expected_keys - set(grid))}, "
            f"unexpected={sorted(set(grid) - expected_keys)}"
        )
    pair_fields = ("prefix_snapshots", "eligibility", "edit", "replay", "data", "model")
    for seed in SEEDS:
        if any((seed, branch) not in grid for branch in BRANCHES):
            continue
        anchor = grid[seed, BRANCHES[0]]
        for branch in BRANCHES[1:]:
            summary = grid[seed, branch]
            for field in pair_fields:
                if summary.get(field) != anchor.get(field):
                    errors.append(f"seed {seed}: six-arm pairing differs at {field}")
            for field in ("initial_model", "independent_prefix_model", "nested_prefix_model"):
                if get(summary, f"hashes.{field}") != get(anchor, f"hashes.{field}"):
                    errors.append(f"seed {seed}: paired prefix hash differs at {field}")
        phase_b_hashes = {
            (
                get(grid[seed, branch], "data.phase_b_batch_digest"),
                get(grid[seed, branch], "data.phase_b_sampler_digest"),
                get(grid[seed, branch], "hashes.phase_b_initial_optimizer"),
            )
            for branch in BRANCHES
        }
        if len(phase_b_hashes) != 1:
            errors.append(f"seed {seed}: phase-B data/sampler/fresh optimizer are not paired")
    if errors:
        raise_audit(errors)

    eligibility = [
        seed
        for seed in SEEDS
        if get(grid[seed, BRANCHES[0]], "eligibility.paired_intersection_eligible")
        is True
    ]
    pure_p_seeds = [
        seed
        for seed in SEEDS
        if all(grid[seed, branch].get("postedit_pure_p") is True for branch in BRANCHES)
    ]
    return grid, {
        "artifacts_root": str(artifacts_root.resolve()),
        "complete_artifacts": len(grid),
        "metric_records_audited": metric_records,
        "source_fingerprint": FROZEN_SOURCE_FINGERPRINT,
        "config_source_sha256": FROZEN_CONFIG_SHA256,
        "direct_Q_AUCs_reconstructed": len(aucs),
        "paired_intersection_eligible_count": len(eligibility),
        "paired_intersection_eligible_seeds": eligibility,
        "all_six_postedit_pure_P_count": len(pure_p_seeds),
        "pilot_gate_decision": gate["decision"],
        "pilot_joint_seed_pass_count": gate["joint_seed_pass_count"],
        "strict_pairing_and_direct_checkpoint_audit_passed": True,
    }


class Bootstrap:
    """Frozen, label-stable seed bootstrap used by the canonical E19 analysis."""

    @staticmethod
    def _indices(key: str) -> NDArray[np.int64]:
        digest = hashlib.blake2b(key.encode(), digest_size=8, person=b"forke19")
        generator = np.random.default_rng(int.from_bytes(digest.digest(), "little"))
        return generator.integers(
            0, len(SEEDS), size=(BOOTSTRAP_DRAWS, len(SEEDS))
        )

    def interval(self, values: NDArray[np.float64], *, key: str) -> dict[str, Any]:
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (len(SEEDS),):
            raise ValueError(f"paired estimate must have one value per seed, got {array.shape}")
        resampled = array[self._indices(key)].mean(axis=1)
        alpha = (1.0 - CONFIDENCE) / 2.0
        low, high = np.quantile(resampled, [alpha, 1.0 - alpha])
        return {
            "mean": float(np.mean(array)),
            "ci_low": float(low),
            "ci_high": float(high),
            "positive_seed_count": int(np.count_nonzero(array > 0.0)),
            "negative_seed_count": int(np.count_nonzero(array < 0.0)),
            "zero_seed_count": int(np.count_nonzero(array == 0.0)),
            "n_seeds": len(SEEDS),
        }

    def band(
        self, values: NDArray[np.float64], *, key: str
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[0] != len(SEEDS):
            raise ValueError("trajectory matrix must be seed x checkpoint")
        resampled = array[self._indices(key), :].mean(axis=1)
        alpha = (1.0 - CONFIDENCE) / 2.0
        return (
            array.mean(axis=0),
            np.quantile(resampled, alpha, axis=0),
            np.quantile(resampled, 1.0 - alpha, axis=0),
        )


def postedit_value(summary: Mapping[str, Any], path: str) -> float:
    value = get(summary, f"postedit_snapshot.{path}")
    if not finite(value):
        raise RuntimeError(f"non-finite post-edit value at {path}")
    return float(value)


def paired_postedit_delta(
    grid: SummaryGrid, branch: str, noop: str, path: str
) -> NDArray[np.float64]:
    return np.asarray(
        [
            postedit_value(grid[seed, branch], path)
            - postedit_value(grid[seed, noop], path)
            for seed in SEEDS
        ],
        dtype=np.float64,
    )


def auc_vector(grid: SummaryGrid, branch: str) -> NDArray[np.float64]:
    return np.asarray(
        [reconstructed_auc(grid[seed, branch]) for seed in SEEDS], dtype=np.float64
    )


def trajectory_matrix(grid: SummaryGrid, branch: str) -> NDArray[np.float64]:
    return np.asarray(
        [
            [
                snapshot_control(
                    cast(
                        Mapping[str, Any],
                        get(grid[seed, branch], f"phase_b_snapshots.{step}"),
                    )
                )
                for step in PLOT_CHECKPOINTS
            ]
            for seed in SEEDS
        ],
        dtype=np.float64,
    )


def reconstruct_results(
    grid: SummaryGrid, audit: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, NDArray[np.float64]]]:
    bootstrap = Bootstrap()
    edits = {
        "independent_q_restore": ("independent_noop", "restore_active"),
        "independent_padding_sham": ("independent_noop", "restore_sham"),
        "nested_q_transplant": ("nested_noop", "transplant_active"),
        "nested_padding_sham": ("nested_noop", "transplant_sham"),
    }
    vectors: dict[str, NDArray[np.float64]] = {}
    manipulation: dict[str, Any] = {}
    for branch, (noop, label) in edits.items():
        q_delta = paired_postedit_delta(
            grid, branch, noop, "selective_final_hidden_probe.Q"
        )
        vectors[f"postedit_{label}_selective_Q"] = q_delta
        preservation: dict[str, Any] = {}
        for name, path in {
            "P_behavior": "behavior.P",
            "P_causal": "causal.P",
            "selective_P": "selective_final_hidden_probe.P",
            "selective_Y": "selective_final_hidden_probe.Y",
        }.items():
            value = paired_postedit_delta(grid, branch, noop, path)
            vectors[f"postedit_{label}_{name}"] = value
            result = bootstrap.interval(
                value, key=f"manipulation:{branch}:{name}"
            )
            result["equivalent_within_0.05"] = bool(
                result["ci_low"] >= -MANIPULATION_MARGIN
                and result["ci_high"] <= MANIPULATION_MARGIN
            )
            preservation[name] = result
        q_result = bootstrap.interval(
            q_delta, key=f"manipulation:{branch}:selective_Q"
        )
        q_result["registered_rule_passed"] = bool(
            (
                q_result["mean"] <= -MANIPULATION_MARGIN
                and q_result["ci_high"] < 0.0
            )
            if label == "restore_active"
            else (
                q_result["mean"] >= MANIPULATION_MARGIN
                and q_result["ci_low"] > 0.0
            )
            if label == "transplant_active"
            else (
                q_result["ci_low"] >= -MANIPULATION_MARGIN
                and q_result["ci_high"] <= MANIPULATION_MARGIN
            )
        )
        manipulation[label] = {
            "selective_Q": q_result,
            "preservation": preservation,
            "all_preservation_rules_passed": all(
                item["equivalent_within_0.05"] for item in preservation.values()
            ),
        }

    aucs = {branch: auc_vector(grid, branch) for branch in BRANCHES}
    for branch, value in aucs.items():
        vectors[f"auc_{branch}"] = value
    contrast_vectors = {
        "necessity_sham_minus_restore": aucs["independent_padding_sham"]
        - aucs["independent_q_restore"],
        "sufficiency_transplant_minus_sham": aucs["nested_q_transplant"]
        - aucs["nested_padding_sham"],
        "independent_sham_minus_noop": aucs["independent_padding_sham"]
        - aucs["independent_noop"],
        "nested_sham_minus_noop": aucs["nested_padding_sham"]
        - aucs["nested_noop"],
    }
    contrasts: dict[str, Any] = {}
    canonical_contrast_names = {
        "necessity_sham_minus_restore": "Delta_N",
        "sufficiency_transplant_minus_sham": "Delta_S",
        "independent_sham_minus_noop": "independent_sham_minus_noop",
        "nested_sham_minus_noop": "nested_sham_minus_noop",
    }
    for name, value in contrast_vectors.items():
        vectors[f"contrast_{name}"] = value
        result = bootstrap.interval(
            value,
            key=(
                "fixed_all_20:"
                f"{canonical_contrast_names[name]}:Q_control_auc_0_128"
            ),
        )
        if name.startswith(("necessity", "sufficiency")):
            result["registered_rule_passed"] = bool(
                result["mean"] >= ACTIVE_EFFECT_THRESHOLD
                and result["ci_low"] > 0.0
                and result["positive_seed_count"] >= MINIMUM_POSITIVE_SEEDS
            )
        else:
            result["registered_equivalence_passed"] = bool(
                result["ci_low"] >= -SHAM_AUC_EQUIVALENCE
                and result["ci_high"] <= SHAM_AUC_EQUIVALENCE
            )
        contrasts[name] = result

    active_rules = all(
        contrasts[name]["registered_rule_passed"]
        for name in (
            "necessity_sham_minus_restore",
            "sufficiency_transplant_minus_sham",
        )
    )
    sham_rules = all(
        contrasts[name]["registered_equivalence_passed"]
        for name in (
            "independent_sham_minus_noop",
            "nested_sham_minus_noop",
        )
    )
    manipulation_passed = all(
        value["selective_Q"]["registered_rule_passed"]
        and value["all_preservation_rules_passed"]
        for value in manipulation.values()
    )
    eligibility_passed = int(audit["paired_intersection_eligible_count"]) >= MINIMUM_ELIGIBLE_SEEDS
    classification = (
        "bidirectional_active_first_layer_Q_input_pathway_contribution"
        if active_rules and sham_rules and manipulation_passed and eligibility_passed
        else "registered_bidirectional_pathway_criterion_not_met"
    )
    report = {
        "experiment": "E19 active first-layer Q-input-pathway intervention",
        "inference_status": "adaptive_posthoc_frozen_fresh_seed_panel",
        "independent_unit": "training seed",
        "audit": dict(audit),
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed_derivation": (
                "BLAKE2b-64 of the registered contrast label with person=forke19"
            ),
            "method": "deterministic paired nonparametric percentile bootstrap of seed means",
            "confidence": CONFIDENCE,
            "all_20_seeds_used_without_filtering": True,
        },
        "manipulation_checks": manipulation,
        "registered_Q_control_AUC_contrasts": contrasts,
        "eligibility_rule_passed": eligibility_passed,
        "full_manipulation_check_passed": manipulation_passed,
        "both_active_AUC_rules_passed": active_rules,
        "both_sham_equivalence_rules_passed": sham_rules,
        "classification": classification,
        "claim_scope": (
            "The edited active first-layer Q input columns contribute in both registered "
            "directions to later Q-control readiness after P knockout; this is not a "
            "mediation or whole-representation claim."
        ),
    }
    return report, vectors


def register_myriad() -> str:
    candidates = [Path.home() / "Library" / "Fonts", Path("/Library/Fonts")]
    names: list[str] = []
    for directory in candidates:
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if path.is_file() and "myriad" in path.name.lower() and path.suffix.lower() in {
                ".otf",
                ".ttf",
                ".ttc",
            }:
                try:
                    fm.fontManager.addfont(str(path))
                    names.append(fm.FontProperties(fname=str(path)).get_name())
                except RuntimeError:
                    pass
    exact = [name for name in names if name.lower() == "myriad pro"]
    if exact:
        return exact[0]
    discovered = sorted(
        {font.name for font in fm.fontManager.ttflist if "myriad" in font.name.lower()}
    )
    return discovered[0] if discovered else "DejaVu Sans"


def configure_style() -> str:
    font = register_myriad()
    mpl.rcParams.update(
        {
            "font.family": font,
            "font.size": 8.4,
            "axes.titlesize": 10.1,
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
            "legend.fontsize": 8.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )
    return font


def panel_label(ax: Axes, label: str) -> None:
    ax.text(
        -0.15,
        1.075,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11.5,
        fontweight="bold",
        color=INK,
        clip_on=False,
    )


def finish_axis(ax: Axes, *, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.65, alpha=0.75, zorder=0)
    ax.tick_params(length=3, width=0.7)


def plot_postedit(
    ax: Axes,
    report: Mapping[str, Any],
    vectors: Mapping[str, NDArray[np.float64]],
) -> None:
    names = (
        "restore_active",
        "restore_sham",
        "transplant_active",
        "transplant_sham",
    )
    labels = ("Restore\nactive Q", "Restore\npadding", "Transplant\nactive Q", "Transplant\npadding")
    colors = (ACTIVE, SHAM, ACTIVE, SHAM)
    x = np.arange(len(names), dtype=float)
    offsets = np.linspace(-0.105, 0.105, len(SEEDS))
    manipulation = cast(Mapping[str, Any], report["manipulation_checks"])
    for index, (name, color) in enumerate(zip(names, colors, strict=True)):
        value = vectors[f"postedit_{name}_selective_Q"]
        result = cast(Mapping[str, Any], get(manipulation, f"{name}.selective_Q"))
        ax.scatter(
            index + offsets,
            value,
            s=10,
            facecolor=color,
            edgecolor="white",
            linewidth=0.25,
            alpha=0.34,
            zorder=2,
        )
        ax.errorbar(
            index,
            result["mean"],
            yerr=[
                [float(result["mean"]) - float(result["ci_low"])],
                [float(result["ci_high"]) - float(result["mean"])],
            ],
            fmt="o",
            markersize=5.8,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.7,
            elinewidth=1.6,
            capsize=3.0,
            zorder=4,
        )
    ax.axhline(0.0, color=INK, linewidth=0.9, zorder=1)
    for threshold in (-MANIPULATION_MARGIN, MANIPULATION_MARGIN):
        ax.axhline(threshold, color=MUTED, linewidth=0.75, linestyle=(0, (2, 2)), zorder=1)
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.48, 3.48)
    ax.set_ylim(-0.142, 0.105)
    ax.set_ylabel("Change in selective Q probe advantage")
    ax.set_title("Active edits move Q accessibility as intended", loc="left", pad=7)
    finish_axis(ax)


def plot_trajectory(
    ax: Axes,
    grid: SummaryGrid,
    bootstrap: Bootstrap,
    branches: Sequence[tuple[str, str, str, str]],
    title: str,
) -> None:
    x = np.log1p(np.asarray(PLOT_CHECKPOINTS, dtype=np.float64))
    for branch, _label, color, linestyle in branches:
        mean, low, high = bootstrap.band(
            trajectory_matrix(grid, branch), key=f"trajectory:{branch}"
        )
        ax.fill_between(x, low, high, color=color, alpha=0.11, linewidth=0, zorder=1)
        ax.plot(
            x,
            mean,
            color=color,
            linewidth=1.8,
            linestyle=linestyle,
            zorder=3,
        )
    ax.axhline(0.90, color=MUTED, linewidth=0.75, linestyle=(0, (2, 2)), zorder=1)
    ticks = (0, 1, 5, 17, 45, 128)
    ax.set_xticks(np.log1p(ticks), [str(value) for value in ticks])
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0.475, 1.012)
    ax.set_yticks((0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
    ax.set_xlabel("Phase-B updates (direct checkpoints)")
    ax.set_ylabel("Q control: mean of behavior and causal score")
    ax.set_title(title, loc="left", pad=7)
    finish_axis(ax)


def plot_auc_contrasts(
    ax: Axes,
    report: Mapping[str, Any],
    vectors: Mapping[str, NDArray[np.float64]],
) -> None:
    names = (
        "necessity_sham_minus_restore",
        "sufficiency_transplant_minus_sham",
        "independent_sham_minus_noop",
        "nested_sham_minus_noop",
    )
    labels = (
        "Delta N: sham - restore",
        "Delta S: transplant - sham",
        "I sham - no edit",
        "N sham - no edit",
    )
    positions = np.asarray((3.25, 2.25, 0.75, -0.25))
    colors = (ACTIVE_DARK, ACTIVE, NOOP, NOOP)
    contrasts = cast(Mapping[str, Mapping[str, Any]], report["registered_Q_control_AUC_contrasts"])
    seed_offsets = np.linspace(-0.13, 0.13, len(SEEDS))
    ax.axvspan(
        -SHAM_AUC_EQUIVALENCE,
        SHAM_AUC_EQUIVALENCE,
        color=PALE_GRAY,
        zorder=0,
    )
    ax.axvline(0.0, color=INK, linewidth=0.9, zorder=1)
    ax.axvline(
        ACTIVE_EFFECT_THRESHOLD,
        color=ACTIVE,
        linewidth=0.8,
        linestyle=(0, (2, 2)),
        zorder=1,
    )
    for name, y, color in zip(names, positions, colors, strict=True):
        value = vectors[f"contrast_{name}"]
        result = contrasts[name]
        ax.scatter(
            value,
            y + seed_offsets,
            s=9,
            facecolor=color,
            edgecolor="white",
            linewidth=0.2,
            alpha=0.30,
            zorder=2,
        )
        ax.errorbar(
            result["mean"],
            y,
            xerr=[
                [float(result["mean"]) - float(result["ci_low"])],
                [float(result["ci_high"]) - float(result["mean"])],
            ],
            fmt="o",
            color=color,
            markersize=5.8,
            markeredgecolor="white",
            markeredgewidth=0.7,
            elinewidth=1.6,
            capsize=3.0,
            zorder=4,
        )
    ax.text(
        ACTIVE_EFFECT_THRESHOLD + 0.001,
        3.68,
        "active threshold",
        color=ACTIVE_DARK,
        fontsize=7.0,
        ha="left",
        va="bottom",
    )
    ax.text(
        0.0,
        -0.64,
        "frozen sham-equivalence region",
        color=MUTED,
        fontsize=7.0,
        ha="center",
        va="top",
    )
    ax.set_yticks(positions, ["", "", "", ""])
    for label, y in zip(labels, positions, strict=True):
        ax.text(
            -0.0115,
            y + 0.19,
            label,
            color=MUTED,
            fontsize=7.7,
            ha="left",
            va="bottom",
            zorder=5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 0.5},
        )
    ax.set_xlim(-0.0125, 0.0505)
    ax.set_ylim(-0.92, 3.92)
    ax.set_xlabel("Paired change in Q-control AUC, updates 0 to 128")
    ax.set_title("Both frozen AUC contrasts pass", loc="left", pad=7)
    finish_axis(ax, grid_axis="x")
    ax.tick_params(axis="y", length=0)


def make_figure(
    grid: SummaryGrid,
    report: Mapping[str, Any],
    vectors: Mapping[str, NDArray[np.float64]],
    output_dir: Path,
) -> tuple[Path, Path]:
    configure_style()
    figure, axes = plt.subplots(2, 2, figsize=(7.35, 6.15))
    figure.subplots_adjust(left=0.105, right=0.985, bottom=0.105, top=0.895, wspace=0.34, hspace=0.48)
    ax_a, ax_b, ax_c, ax_d = axes.flat
    plot_postedit(ax_a, report, vectors)
    bootstrap = Bootstrap()
    independent = (
        ("independent_noop", "No edit", NOOP, "-"),
        ("independent_q_restore", "Active Q columns", ACTIVE, "-"),
        ("independent_padding_sham", "Padding sham", SHAM, (0, (3, 1.7))),
    )
    nested = (
        ("nested_noop", "No edit", NOOP, "-"),
        ("nested_q_transplant", "Active Q columns", ACTIVE, "-"),
        ("nested_padding_sham", "Padding sham", SHAM, (0, (3, 1.7))),
    )
    plot_trajectory(ax_b, grid, bootstrap, independent, "Active restoration delays Q handoff")
    plot_trajectory(ax_c, grid, bootstrap, nested, "Active transplant accelerates Q handoff")
    plot_auc_contrasts(ax_d, report, vectors)
    for ax, label in zip(axes.flat, "abcd", strict=True):
        panel_label(ax, label)
    handles = (
        Line2D([0], [0], color=NOOP, linewidth=2.0, label="No edit"),
        Line2D([0], [0], color=ACTIVE, linewidth=2.0, label="Active Q columns"),
        Line2D(
            [0],
            [0],
            color=SHAM,
            linewidth=2.0,
            linestyle=(0, (3, 1.7)),
            label="Padding sham",
        ),
    )
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.65, 0.986),
        ncol=3,
        frameon=False,
        handlelength=2.5,
        columnspacing=1.5,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = output_dir / "fig19_active_q_pathway.pdf"
    png = output_dir / "fig19_active_q_pathway.png"
    figure.savefig(pdf, dpi=300, bbox_inches="tight", pad_inches=0.035)
    figure.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.035)
    plt.close(figure)
    return pdf, png


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_derived(
    output_dir: Path,
    report: Mapping[str, Any],
    vectors: Mapping[str, NDArray[np.float64]],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "e19_figure_analysis.json"
    json_path.write_text(
        json.dumps(json_safe(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    rows: list[dict[str, Any]] = []
    for name, value in vectors.items():
        for seed, observation in zip(SEEDS, value, strict=True):
            rows.append({"quantity": name, "seed": seed, "value": float(observation)})
    csv_path = output_dir / "e19_figure_seed_level_values.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("quantity", "seed", "value"))
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--pilot-gate", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--figures", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--derived", type=Path, default=DEFAULT_DERIVED)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    grid, audit = audit_and_load(args.artifacts, args.pilot_gate)
    report, vectors = reconstruct_results(grid, audit)
    json_path, csv_path = write_derived(args.derived, report, vectors)
    pdf, png = make_figure(grid, report, vectors, args.figures)
    contrasts = cast(Mapping[str, Mapping[str, Any]], report["registered_Q_control_AUC_contrasts"])
    necessity = contrasts["necessity_sham_minus_restore"]
    sufficiency = contrasts["sufficiency_transplant_minus_sham"]
    print(
        "E19 reconstructed: "
        f"Delta_N={necessity['mean']:.4f} [{necessity['ci_low']:.4f}, {necessity['ci_high']:.4f}], "
        f"Delta_S={sufficiency['mean']:.4f} [{sufficiency['ci_low']:.4f}, {sufficiency['ci_high']:.4f}]"
    )
    print(f"  classification: {report['classification']}")
    print(f"  figure: {pdf} and {png}")
    print(f"  data: {json_path} and {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
