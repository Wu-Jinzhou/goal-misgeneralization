#!/usr/bin/env python3
"""Fail-closed, outcome-blind engineering-pilot gate for E19.

This program intentionally accepts only the frozen 3-seed x 6-branch pilot.
It rejects any artifact that contains phase-B construction or outcomes and
emits only the preregistered manipulation-gate quantities.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml  # type: ignore[import-untyped]

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forkworld.artifacts import implementation_provenance  # noqa: E402
from forkworld.config import expand_sweep, load_config  # noqa: E402
from forkworld.handoff import pure_control, stable_state_digest  # noqa: E402
from forkworld.q_pathway import (  # noqa: E402
    EXPECTED_FEATURE_NAMES,
    EXPECTED_STATE_SHAPES,
    PADDING_SHAM_COLUMN_INDICES,
    Q_COLUMN_INDICES,
    Q_PATHWAY_BRANCHES,
)

EXPERIMENT = "active_q_input_pathway_intervention"
CONFIG_PATH = REPO / "configs" / "e19_q_pathway_mediation.yaml"
DEFAULT_ARTIFACTS = REPO / "artifacts-e19-pilot"
DEFAULT_OUTPUT = HERE / "derived"

PILOT_SEEDS = (541, 547, 557)
BRANCHES = tuple(Q_PATHWAY_BRANCHES)
FROZEN_SOURCE_FINGERPRINT = (
    "ab91250cbb8379c1e6afd0f960abe0754b08a745bd2d9fd8922be0f65a1ba7a9"
)
FROZEN_SOURCE_FILE_COUNT = 31
FROZEN_CONFIG_SHA256 = (
    "79e24a5e68b95e76e3bffd61d62abfdd86f6f6c439a45c76c376a2f548b74345"
)
ARTIFACT_SCHEMA_VERSION = 1
SOURCE_FINGERPRINT_SCHEMA_VERSION = 1
SHIFT = 0.05
PRESERVATION = 0.05
MINIMUM_JOINT_SEEDS = 2
FLOAT32_REALIZED_ROUNDOFF_TOLERANCE = 1e-6
NON_SCIENTIFIC_RUN_FIELDS = frozenset({"output_root", "resume", "seeds"})

NOOP_FOR = {
    "independent_q_restore": "independent_noop",
    "independent_padding_sham": "independent_noop",
    "nested_q_transplant": "nested_noop",
    "nested_padding_sham": "nested_noop",
}
EDIT_BRANCHES = tuple(NOOP_FOR)
PRESERVATION_PATHS = {
    "behavior_P": "behavior.P",
    "causal_P": "causal.P",
    "selective_P": "selective_final_hidden_probe.P",
    "selective_Y": "selective_final_hidden_probe.Y",
}


def get(mapping: Mapping[str, Any], path: str, default: Any = None) -> Any:
    value: Any = mapping
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str)


def _scientific_config(config: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        str(key): copy.deepcopy(value)
        for key, value in config.items()
        if key != "seed" and not str(key).startswith("_")
    }
    run = dict(cast(Mapping[str, Any], result.get("run", {})))
    for field in NON_SCIENTIFIC_RUN_FIELDS:
        run.pop(field, None)
    result["run"] = run
    return result


def _config_fingerprint(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(_scientific_config(config)).encode()).hexdigest()


def _expected_run_id(config: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    implementation = get(metadata, "implementation", {})
    identity = {
        "config": _canonical(_scientific_config(config)),
        "seed": int(config["seed"]),
        "artifact_schema_version": get(implementation, "artifact_schema_version"),
        "source_fingerprint_schema_version": get(
            implementation, "source_fingerprint_schema_version"
        ),
        "implementation_fingerprint": get(
            implementation, "implementation_fingerprint"
        ),
    }
    raw = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def _hash_is_valid(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read JSON object {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise RuntimeError(f"expected JSON object: {path}")
    return cast(Mapping[str, Any], value)


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise RuntimeError(f"cannot read YAML mapping {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise RuntimeError(f"expected YAML mapping: {path}")
    return cast(Mapping[str, Any], value)


def _fail(name: str, errors: Sequence[str]) -> None:
    preview = "\n".join(f"  - {error}" for error in errors[:80])
    suffix = "" if len(errors) <= 80 else f"\n  ... and {len(errors) - 80} more"
    raise RuntimeError(f"{name} failed with {len(errors)} issue(s):\n{preview}{suffix}")


def _expect(errors: list[str], run_id: str, actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        errors.append(f"{run_id}: {label}={actual!r}, expected {expected!r}")


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(
        float(value)
    )


def _audit_sham_delta_roundoff(
    errors: list[str],
    run_id: str,
    edit_name: str,
    intended: Any,
    sham: Any,
) -> None:
    """Allow only the registered float32 add/sub reconstruction roundoff.

    The constructor applies the intended tensor by ``index_add_`` to the sham
    columns.  Reconstructing that displacement from final minus initial
    float32 columns need not be bit-identical to the intended tensor.  Shape,
    support, finiteness, and scalar counts remain exact; only the three norm
    summaries receive this small numerical tolerance.
    """

    if not isinstance(intended, Mapping) or not isinstance(sham, Mapping):
        errors.append(f"{run_id}: malformed {edit_name} intended/sham displacement")
        return
    for field in (
        "shape",
        "target_scalar_count",
        "nonzero_scalar_count",
        "finite",
    ):
        _expect(
            errors,
            run_id,
            sham.get(field),
            intended.get(field),
            f"{edit_name} sham delta {field}",
        )
    if not _hash_is_valid(sham.get("digest")):
        errors.append(f"{run_id}: malformed {edit_name} sham displacement digest")
    for field in ("l1_norm", "l2_norm", "linf_norm"):
        actual = sham.get(field)
        target = intended.get(field)
        if not _finite(actual) or not _finite(target):
            errors.append(f"{run_id}: non-finite {edit_name} sham delta {field}")
        elif abs(float(actual) - float(target)) > FLOAT32_REALIZED_ROUNDOFF_TOLERANCE:
            errors.append(
                f"{run_id}: {edit_name} sham delta {field} differs from intended "
                f"by {abs(float(actual) - float(target)):.9g}, exceeding "
                f"{FLOAT32_REALIZED_ROUNDOFF_TOLERANCE:g}"
            )


@dataclass(frozen=True)
class PilotRun:
    path: Path
    config: Mapping[str, Any]
    summary: Mapping[str, Any]
    metadata: Mapping[str, Any]
    metric_lines: int

    @property
    def seed(self) -> int:
        return int(self.summary["seed"])

    @property
    def branch(self) -> str:
        return str(self.summary["branch"])


def _expected_configs() -> dict[str, Mapping[str, Any]]:
    if _sha256(CONFIG_PATH) != FROZEN_CONFIG_SHA256:
        raise RuntimeError("the E19 source configuration no longer has its frozen SHA-256")
    loaded = load_config(CONFIG_PATH, ("h16.pilot_only=true",))
    cells = expand_sweep(loaded)
    result = {str(get(cell, "h16.branch")): _scientific_config(cell) for cell in cells}
    if tuple(result) != BRANCHES:
        raise RuntimeError(f"the expanded E19 branch order changed: {tuple(result)!r}")
    return result


def _artifact_directory(root: Path) -> Path:
    nested = root / "h16" / EXPERIMENT
    if nested.is_dir():
        return nested
    if root.name == EXPERIMENT and root.parent.name == "h16" and root.is_dir():
        return root
    raise RuntimeError(f"missing E19 pilot artifact directory {nested}")


def _audit_metric_file(path: Path, run_id: str, seed: int, branch: str, errors: list[str]) -> int:
    line_count = 0
    stages: set[str] = set()
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as error:
        errors.append(f"{run_id}: cannot read metrics.jsonl: {error}")
        return 0
    with handle:
        for line_number, line in enumerate(handle, 1):
            line_count += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                errors.append(f"{run_id}: malformed metric line {line_number}: {error}")
                continue
            if not isinstance(row, Mapping):
                errors.append(f"{run_id}: metric line {line_number} is not an object")
                continue
            stage = str(row.get("stage", ""))
            stages.add(stage)
            if row.get("run_id") != run_id or row.get("seed") != seed:
                errors.append(f"{run_id}: metric identity mismatch at line {line_number}")
            if row.get("experiment") != "h16" or row.get("level") != "choice":
                errors.append(f"{run_id}: metric design mismatch at line {line_number}")
            if "phase_b" in stage.lower():
                errors.append(f"{run_id}: phase-B metric leaked at line {line_number}")
            global_step = row.get("global_step")
            examples_seen = row.get("examples_seen")
            if not isinstance(global_step, int) or global_step > 45:
                errors.append(f"{run_id}: post-pilot global step at metric line {line_number}")
            if examples_seen is not None and (
                not isinstance(examples_seen, int) or examples_seen > 11_250
            ):
                errors.append(f"{run_id}: post-pilot examples_seen at metric line {line_number}")
            condition = str(row.get("condition", ""))
            allowed = {
                f"active_q_input_pathway:{branch}",
                f"active_q_input_pathway:{branch}:independent_prefix",
                f"active_q_input_pathway:{branch}:nested_prefix",
                "adaptive_q_pathway_intervention",
            }
            if condition not in allowed:
                errors.append(f"{run_id}: unexpected metric condition at line {line_number}")
    required_stages = {
        "independent_phase_a_behavior",
        "independent_phase_a_causal",
        "independent_phase_a_probe",
        "independent_phase_a_strength",
        "independent_phase_a_truth_table",
        "independent_phase_a_optimization",
        "nested_phase_a_behavior",
        "nested_phase_a_causal",
        "nested_phase_a_probe",
        "nested_phase_a_strength",
        "nested_phase_a_truth_table",
        "nested_phase_a_optimization",
        "postedit_behavior",
        "postedit_causal",
        "postedit_probe",
        "postedit_strength",
        "postedit_truth_table",
        "final",
    }
    if stages != required_stages:
        errors.append(
            f"{run_id}: metric stage set has missing={sorted(required_stages - stages)}, "
            f"unexpected={sorted(stages - required_stages)}"
        )
    return line_count


def _audit_snapshot(
    snapshot: Any,
    run_id: str,
    label: str,
    errors: list[str],
    *,
    local_step: int,
    global_step: int,
) -> None:
    if not isinstance(snapshot, Mapping):
        errors.append(f"{run_id}: missing {label} snapshot")
        return
    for field, expected in {
        "local_step": local_step,
        "global_step": global_step,
        "examples_seen": global_step * 250,
    }.items():
        _expect(errors, run_id, snapshot.get(field), expected, f"{label}.{field}")
    for section in ("behavior", "causal", "selective_final_hidden_probe"):
        values = snapshot.get(section)
        if not isinstance(values, Mapping) or set(values) != {"P", "Q", "Y"}:
            errors.append(f"{run_id}: malformed {label}.{section}")
            continue
        for goal, value in values.items():
            if not _finite(value):
                errors.append(f"{run_id}: non-finite {label}.{section}.{goal}")


def _audit_replay(run: PilotRun, errors: list[str]) -> None:
    for overlap in ("independent", "nested"):
        replay = get(run.summary, f"replay.{overlap}")
        if not isinstance(replay, Mapping):
            errors.append(f"{run.path.name}: missing {overlap} replay")
            continue
        for field, expected in {
            "overlap": overlap,
            "samples_seen": 11_250,
            "optimizer_steps": 45,
            "batch_size": 250,
            "all_minibatches_full": True,
        }.items():
            _expect(errors, run.path.name, replay.get(field), expected, f"replay.{overlap}.{field}")
        checks = replay.get("checks")
        expected_checks = {
            "initial_models_equal",
            "final_models_equal",
            "final_optimizers_equal",
            "samples_seen_equal",
            "optimizer_steps_equal",
        }
        if (
            not isinstance(checks, Mapping)
            or set(checks) != expected_checks
            or not all(value is True for value in checks.values())
        ):
            errors.append(f"{run.path.name}: {overlap} deterministic replay checks failed")
        hashes = replay.get("hashes")
        expected_hashes = {
            "initial_model",
            "replay_initial_model",
            "observed_final_model",
            "replay_final_model",
            "observed_final_optimizer",
            "replay_final_optimizer",
            "phase_a_batch",
            "phase_a_sampler",
        }
        if not isinstance(hashes, Mapping) or set(hashes) != expected_hashes:
            errors.append(f"{run.path.name}: malformed {overlap} replay hash set")
            continue
        if not all(_hash_is_valid(value) for value in hashes.values()):
            errors.append(f"{run.path.name}: malformed {overlap} replay digest")
        for left, right in (
            ("initial_model", "replay_initial_model"),
            ("observed_final_model", "replay_final_model"),
            ("observed_final_optimizer", "replay_final_optimizer"),
        ):
            _expect(errors, run.path.name, hashes.get(left), hashes.get(right), f"replay {left}")
        _expect(
            errors,
            run.path.name,
            hashes.get("initial_model"),
            get(run.summary, "hashes.initial_model"),
            f"{overlap} replay initial model",
        )
        _expect(
            errors,
            run.path.name,
            hashes.get("observed_final_model"),
            get(run.summary, f"hashes.{overlap}_prefix_model"),
            f"{overlap} replay final model",
        )


def _audit_edit(run: PilotRun, errors: list[str]) -> None:
    run_id = run.path.name
    edit = get(run.summary, "edit")
    if not isinstance(edit, Mapping):
        errors.append(f"{run_id}: missing edit audit")
        return
    audit = edit.get("audit")
    if not isinstance(audit, Mapping):
        errors.append(f"{run_id}: missing branch construction audit")
        return
    for field in (
        "donors_unchanged",
        "all_six_branches_verified",
        "all_unaffected_state_exact",
        "all_edits_target_exactly_128_scalars",
    ):
        _expect(errors, run_id, audit.get(field), True, f"edit.audit.{field}")
    contract = audit.get("contract")
    expected_contract = {
        "feature_names": list(EXPECTED_FEATURE_NAMES),
        "q_column_indices": list(Q_COLUMN_INDICES),
        "q_column_names": ["Q_1", "Q_2"],
        "padding_sham_column_indices": list(PADDING_SHAM_COLUMN_INDICES),
        "padding_sham_column_names": ["R_4", "R_5"],
        "state_shapes": {key: list(value) for key, value in EXPECTED_STATE_SHAPES.items()},
        "cpu_only": True,
        "contract_verified": True,
    }
    if not isinstance(contract, Mapping):
        errors.append(f"{run_id}: missing edit feature contract")
    else:
        for field, expected in expected_contract.items():
            _expect(errors, run_id, contract.get(field), expected, f"edit.contract.{field}")
        if not _hash_is_valid(contract.get("feature_names_digest")):
            errors.append(f"{run_id}: malformed feature-name digest")
        if contract.get("parameter_dtype") not in {"torch.float32", "torch.float64"}:
            errors.append(f"{run_id}: unexpected edit parameter dtype")
    donors = audit.get("donors")
    donor_copy = edit.get("donor_state_digests")
    if not isinstance(donors, Mapping) or set(donors) != {"initial", "independent", "nested"}:
        errors.append(f"{run_id}: malformed edit donor hashes")
    elif not all(_hash_is_valid(value) for value in donors.values()):
        errors.append(f"{run_id}: malformed edit donor digest")
    _expect(errors, run_id, donor_copy, donors, "duplicated donor digest audit")
    if isinstance(donors, Mapping):
        for donor, summary_name in {
            "initial": "initial_model",
            "independent": "independent_prefix_model",
            "nested": "nested_prefix_model",
        }.items():
            _expect(
                errors,
                run_id,
                donors.get(donor),
                get(run.summary, f"hashes.{summary_name}"),
                f"donor {donor} hash",
            )
    branches = audit.get("branches")
    branch_hashes = edit.get("branch_model_hashes")
    if not isinstance(branches, Mapping) or set(branches) != set(BRANCHES):
        errors.append(f"{run_id}: malformed audited branch set")
    if not isinstance(branch_hashes, Mapping) or set(branch_hashes) != set(BRANCHES):
        errors.append(f"{run_id}: malformed branch hash set")
    elif not all(_hash_is_valid(value) for value in branch_hashes.values()):
        errors.append(f"{run_id}: malformed branch model digest")
    if isinstance(branches, Mapping) and isinstance(branch_hashes, Mapping):
        for name in BRANCHES:
            branch_audit = branches.get(name)
            if not isinstance(branch_audit, Mapping):
                errors.append(f"{run_id}: missing audit for {name}")
                continue
            _expect(errors, run_id, branch_audit.get("unchanged_state_verified"), True, f"{name} unchanged")
            _expect(
                errors,
                run_id,
                branch_audit.get("designated_columns_verified"),
                True,
                f"{name} designated columns",
            )
            _expect(
                errors,
                run_id,
                branch_audit.get("branch_state_digest"),
                branch_hashes.get(name),
                f"{name} state digest",
            )
            edited = name not in {"independent_noop", "nested_noop"}
            expected_columns = (
                list(Q_COLUMN_INDICES)
                if name in {"independent_q_restore", "nested_q_transplant"}
                else list(PADDING_SHAM_COLUMN_INDICES)
                if edited
                else []
            )
            _expect(
                errors,
                run_id,
                branch_audit.get("edited_columns"),
                expected_columns,
                f"{name} designated column indices",
            )
            _expect(
                errors,
                run_id,
                branch_audit.get("target_scalar_count"),
                128 if edited else 0,
                f"{name} edited scalar count",
            )
        _expect(
            errors,
            run_id,
            get(run.summary, "hashes.selected_postedit_model"),
            branch_hashes.get(run.branch),
            "selected branch model hash",
        )
    edits = audit.get("edits")
    if not isinstance(edits, Mapping) or set(edits) != {"restore", "transplant"}:
        errors.append(f"{run_id}: malformed restore/transplant audit")
    else:
        for name in ("restore", "transplant"):
            item = edits.get(name)
            if not isinstance(item, Mapping):
                errors.append(f"{run_id}: missing {name} edit audit")
                continue
            _expect(errors, run_id, item.get("same_intended_delta_applied"), True, f"{name} paired delta")
            _expect(errors, run_id, item.get("active_target"), "Q_1,Q_2", f"{name} active target")
            _expect(errors, run_id, item.get("sham_target"), "R_4,R_5", f"{name} sham target")
            intended = item.get("intended_delta")
            _expect(errors, run_id, item.get("active_realized_delta"), intended, f"{name} active delta")
            _audit_sham_delta_roundoff(
                errors,
                run_id,
                name,
                intended,
                item.get("sham_realized_delta"),
            )
            if not isinstance(intended, Mapping) or intended.get("target_scalar_count") != 128:
                errors.append(f"{run_id}: malformed {name} displacement")
    audit_payload = {
        "contract": audit.get("contract"),
        "donors": audit.get("donors"),
        "branches": audit.get("branches"),
        "edits": audit.get("edits"),
    }
    _expect(
        errors,
        run_id,
        audit.get("audit_digest"),
        stable_state_digest(audit_payload),
        "edit audit digest",
    )
    for field in ("construction_digest", "audit_digest"):
        if not _hash_is_valid(audit.get(field)):
            errors.append(f"{run_id}: malformed edit {field}")
    if isinstance(branch_hashes, Mapping):
        _expect(
            errors,
            run_id,
            edit.get("branch_set_digest"),
            stable_state_digest(dict(branch_hashes)),
            "branch-set digest",
        )
    preactivation = edit.get("preactivation_effects")
    if not isinstance(preactivation, Mapping):
        errors.append(f"{run_id}: missing preactivation edit audit")
    else:
        _expect(errors, run_id, preactivation.get("panel_n"), 4_096, "preactivation panel n")
        _expect(
            errors,
            run_id,
            preactivation.get("panel_digest"),
            get(run.summary, "data.probe_eval_digest"),
            "preactivation panel digest",
        )
        _expect(errors, run_id, preactivation.get("all_values_finite"), True, "preactivation finite")
        comparisons = preactivation.get("comparisons")
        if not isinstance(comparisons, Mapping) or set(comparisons) != set(EDIT_BRANCHES):
            errors.append(f"{run_id}: malformed preactivation comparisons")


def _validate_run(run: PilotRun, expected: Mapping[str, Any], errors: list[str]) -> None:
    run_id = run.path.name
    if _scientific_config(run.config) != expected:
        errors.append(f"{run_id}: resolved scientific config differs from frozen pilot config")
    fixed = {
        "hypothesis": "h16",
        "seed": run.seed,
        "branch": run.branch,
        "condition": f"active_q_input_pathway:{run.branch}",
        "pilot_only": True,
        "design_status": "adaptive_posthoc_active_q_input_pathway_intervention_frozen_pre_outcome",
        "data.phase_b_constructed": False,
        "data.probe_splits_disjoint": True,
        "measurement.probe_train_n": 2_048,
        "measurement.probe_eval_n": 4_096,
        "measurement.probe_ridge": 0.001,
        "measurement.truth_table_control_seed": 1_500_450_271,
        "training.batch_size": 250,
        "training.phase_a_steps_per_history": 45,
        "training.phase_a_examples_seen_per_history": 11_250,
        "training.phase_b_steps": 0,
        "training.phase_b_examples_seen": 0,
        "training.all_minibatches_full": True,
        "model.input_dim": 19,
        "model.total_parameters": 5_505,
        "model.trainable_parameters": 5_505,
        "model.update_mode": "full",
        "model.requested_budget": "full",
    }
    for path, value in fixed.items():
        _expect(errors, run_id, get(run.summary, path), value, f"summary.{path}")
    forbidden = {
        "phase_b_snapshots",
        "phase_b_local_zero",
        "final",
        "outcomes",
        "optimizer_transition_audit",
    }
    leaked = forbidden & set(run.summary)
    if leaked:
        errors.append(f"{run_id}: forbidden pilot summary fields: {sorted(leaked)}")
    data = get(run.summary, "data")
    if not isinstance(data, Mapping) or set(data) != {
        "phase_b_constructed",
        "probe_train_digest",
        "probe_eval_digest",
        "probe_splits_disjoint",
    }:
        errors.append(f"{run_id}: pilot data section contains missing or phase-B fields")
    hashes = get(run.summary, "hashes")
    expected_hashes = {
        "initial_model",
        "independent_prefix_model",
        "nested_prefix_model",
        "selected_postedit_model",
    }
    if not isinstance(hashes, Mapping) or set(hashes) != expected_hashes:
        errors.append(f"{run_id}: pilot hash section contains missing or phase-B fields")
    elif not all(_hash_is_valid(value) for value in hashes.values()):
        errors.append(f"{run_id}: malformed pilot model digest")
    for digest in ("data.probe_train_digest", "data.probe_eval_digest"):
        if not _hash_is_valid(get(run.summary, digest)):
            errors.append(f"{run_id}: malformed {digest}")
    wall = get(run.summary, "training.wall_seconds")
    if not _finite(wall) or float(wall) <= 0:
        errors.append(f"{run_id}: invalid wall time")

    prefix = get(run.summary, "prefix_snapshots")
    if not isinstance(prefix, Mapping) or set(prefix) != {"independent", "nested"}:
        errors.append(f"{run_id}: malformed paired prefix snapshots")
    else:
        for overlap in ("independent", "nested"):
            snapshots = prefix.get(overlap)
            if not isinstance(snapshots, Mapping) or set(snapshots) != {"33", "45"}:
                errors.append(f"{run_id}: malformed {overlap} prefix lattice")
                continue
            for step in (33, 45):
                _audit_snapshot(
                    snapshots[str(step)],
                    run_id,
                    f"prefix.{overlap}.{step}",
                    errors,
                    local_step=step,
                    global_step=step,
                )
    postedit = get(run.summary, "postedit_snapshot")
    _audit_snapshot(postedit, run_id, "postedit", errors, local_step=0, global_step=45)
    if isinstance(postedit, Mapping):
        expected_pure = pure_control(
            cast(Mapping[str, float], postedit["behavior"]),
            cast(Mapping[str, float], postedit["causal"]),
            "P",
            threshold=0.90,
            margin=0.10,
        )
        _expect(errors, run_id, get(run.summary, "postedit_pure_p"), expected_pure, "postedit pure-P")

    eligibility = get(run.summary, "eligibility")
    if not isinstance(eligibility, Mapping):
        errors.append(f"{run_id}: missing paired eligibility")
    else:
        _expect(errors, run_id, eligibility.get("threshold"), 0.90, "eligibility threshold")
        _expect(errors, run_id, eligibility.get("margin"), 0.10, "eligibility margin")
        endpoints: list[bool] = []
        for overlap in ("independent", "nested"):
            item = eligibility.get(overlap)
            if not isinstance(item, Mapping) or set(item.get("per_checkpoint", {})) != {"33", "45"}:
                errors.append(f"{run_id}: malformed {overlap} eligibility")
                continue
            per_checkpoint = cast(Mapping[str, Any], item["per_checkpoint"])
            endpoint = all(value is True for value in per_checkpoint.values())
            endpoints.append(endpoint)
            _expect(
                errors,
                run_id,
                item.get("eligible_both_registered_checkpoints"),
                endpoint,
                f"{overlap} eligibility endpoint",
            )
        if len(endpoints) == 2:
            _expect(
                errors,
                run_id,
                eligibility.get("paired_intersection_eligible"),
                all(endpoints),
                "paired eligibility endpoint",
            )
    _audit_replay(run, errors)
    _audit_edit(run, errors)


def load_and_audit(root: Path) -> tuple[dict[tuple[int, str], PilotRun], dict[str, Any]]:
    expected_configs = _expected_configs()
    current = implementation_provenance(REPO)
    if current != {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "source_fingerprint_schema_version": SOURCE_FINGERPRINT_SCHEMA_VERSION,
        "implementation_fingerprint": FROZEN_SOURCE_FINGERPRINT,
        "source_file_count": FROZEN_SOURCE_FILE_COUNT,
    }:
        raise RuntimeError(f"current ForkWorld source differs from the E19 freeze: {current}")

    directory = _artifact_directory(root)
    children = sorted(path for path in directory.iterdir() if path.is_dir())
    errors: list[str] = []
    if len(children) != 18:
        errors.append(f"materialized run directories={len(children)}, expected exactly 18")
    runs: dict[tuple[int, str], PilotRun] = {}
    for run_dir in children:
        required = ("COMPLETE", "resolved_config.yaml", "summary.json", "metadata.json", "status.json", "metrics.jsonl")
        missing = [name for name in required if not (run_dir / name).is_file()]
        if missing:
            errors.append(f"{run_dir.name}: missing {', '.join(missing)}")
            continue
        if (run_dir / "COMPLETE").read_text(encoding="utf-8") != "complete\n":
            errors.append(f"{run_dir.name}: invalid COMPLETE marker")
        config = _load_yaml(run_dir / "resolved_config.yaml")
        summary = _load_json(run_dir / "summary.json")
        metadata = _load_json(run_dir / "metadata.json")
        status = _load_json(run_dir / "status.json")
        seed = int(summary.get("seed", -1))
        branch = str(summary.get("branch", ""))
        if status != {"state": "complete", "run_id": run_dir.name}:
            errors.append(f"{run_dir.name}: invalid completion status")
        if metadata.get("run_id") != run_dir.name or metadata.get("seed") != seed:
            errors.append(f"{run_dir.name}: metadata identity mismatch")
        implementation = get(metadata, "implementation")
        expected_implementation = {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "source_fingerprint_schema_version": SOURCE_FINGERPRINT_SCHEMA_VERSION,
            "implementation_fingerprint": FROZEN_SOURCE_FINGERPRINT,
            "source_file_count": FROZEN_SOURCE_FILE_COUNT,
        }
        if implementation != expected_implementation:
            errors.append(f"{run_dir.name}: artifact source fingerprint differs from freeze")
        if _expected_run_id(config, metadata) != run_dir.name:
            errors.append(f"{run_dir.name}: artifact run identity does not reconstruct")
        if seed not in PILOT_SEEDS or branch not in BRANCHES:
            errors.append(f"{run_dir.name}: unregistered pilot key {(seed, branch)!r}")
        if int(config.get("seed", -1)) != seed or get(config, "h16.branch") != branch:
            errors.append(f"{run_dir.name}: config/summary identity mismatch")
        if (run_dir / "predictions.jsonl").exists() and (run_dir / "predictions.jsonl").stat().st_size:
            errors.append(f"{run_dir.name}: pilot predictions are forbidden")
        checkpoint_dir = run_dir / "checkpoints"
        if checkpoint_dir.is_dir() and any(checkpoint_dir.iterdir()):
            errors.append(f"{run_dir.name}: pilot checkpoints are forbidden")
        metric_lines = _audit_metric_file(
            run_dir / "metrics.jsonl", run_dir.name, seed, branch, errors
        )
        run = PilotRun(run_dir, config, summary, metadata, metric_lines)
        if branch in expected_configs:
            _validate_run(run, expected_configs[branch], errors)
        key = (seed, branch)
        if key in runs:
            errors.append(f"duplicate pilot key {key!r}")
        runs[key] = run

    expected_keys = {(seed, branch) for seed in PILOT_SEEDS for branch in BRANCHES}
    if set(runs) != expected_keys:
        errors.append(
            f"pilot grid missing={sorted(expected_keys - set(runs))}, "
            f"unexpected={sorted(set(runs) - expected_keys)}"
        )

    pair_fields = (
        "prefix_snapshots",
        "eligibility",
        "edit",
        "replay",
        "data",
        "model",
        "measurement",
    )
    for seed in PILOT_SEEDS:
        arms = [runs.get((seed, branch)) for branch in BRANCHES]
        if any(run is None for run in arms):
            continue
        present = cast(list[PilotRun], arms)
        anchor = present[0]
        for run in present[1:]:
            for field in pair_fields:
                if run.summary.get(field) != anchor.summary.get(field):
                    errors.append(f"seed {seed}: cross-arm pairing differs at {field}")
            for field in ("initial_model", "independent_prefix_model", "nested_prefix_model"):
                if get(run.summary, f"hashes.{field}") != get(anchor.summary, f"hashes.{field}"):
                    errors.append(f"seed {seed}: cross-arm pairing differs at hashes.{field}")
        branch_hashes = get(anchor.summary, "edit.branch_model_hashes", {})
        if isinstance(branch_hashes, Mapping):
            for run in present:
                _expect(
                    errors,
                    run.path.name,
                    get(run.summary, "hashes.selected_postedit_model"),
                    branch_hashes.get(run.branch),
                    "paired selected branch hash",
                )

    if errors:
        _fail("E19 pilot artifact audit", errors)
    config_fingerprints = {
        branch: hashlib.sha256(_canonical(expected_configs[branch]).encode()).hexdigest()
        for branch in BRANCHES
    }
    return runs, {
        "artifacts_root": str(root.resolve()),
        "complete_artifacts": len(runs),
        "metric_records_audited": sum(run.metric_lines for run in runs.values()),
        "source_fingerprint": FROZEN_SOURCE_FINGERPRINT,
        "source_file_count": FROZEN_SOURCE_FILE_COUNT,
        "config_source_sha256": FROZEN_CONFIG_SHA256,
        "scientific_config_fingerprints_by_branch": config_fingerprints,
        "float32_realized_roundoff_tolerance": FLOAT32_REALIZED_ROUNDOFF_TOLERANCE,
        "exact_grid_pairing_replay_edit_and_no_phase_b_audit_passed": True,
    }


def _snapshot_delta(run: PilotRun, noop: PilotRun, path: str) -> float:
    edited = get(run.summary, f"postedit_snapshot.{path}")
    baseline = get(noop.summary, f"postedit_snapshot.{path}")
    if not _finite(edited) or not _finite(baseline):
        raise RuntimeError(f"non-finite gate metric {path} for seed {run.seed}")
    return float(edited) - float(baseline)


def evaluate_gate(runs: Mapping[tuple[int, str], PilotRun]) -> dict[str, Any]:
    """Evaluate only the frozen same-seed manipulation gate."""

    seed_results: list[dict[str, Any]] = []
    for seed in PILOT_SEEDS:
        arms = {branch: runs[(seed, branch)] for branch in BRANCHES}
        deltas: dict[str, dict[str, float]] = {}
        for branch in EDIT_BRANCHES:
            edited = arms[branch]
            noop = arms[NOOP_FOR[branch]]
            values = {
                "selective_Q": _snapshot_delta(
                    edited, noop, "selective_final_hidden_probe.Q"
                ),
                **{
                    name: _snapshot_delta(edited, noop, path)
                    for name, path in PRESERVATION_PATHS.items()
                },
            }
            deltas[branch] = values
        q_rules = {
            "independent_q_restore": deltas["independent_q_restore"]["selective_Q"] <= -SHIFT,
            "independent_padding_sham": abs(
                deltas["independent_padding_sham"]["selective_Q"]
            )
            <= SHIFT,
            "nested_q_transplant": deltas["nested_q_transplant"]["selective_Q"] >= SHIFT,
            "nested_padding_sham": abs(
                deltas["nested_padding_sham"]["selective_Q"]
            )
            <= SHIFT,
        }
        preservation = {
            branch: all(abs(values[name]) <= PRESERVATION for name in PRESERVATION_PATHS)
            for branch, values in deltas.items()
        }
        all_six_pure_p = all(
            get(arms[branch].summary, "postedit_pure_p") is True for branch in BRANCHES
        )
        joint = all_six_pure_p and all(q_rules.values()) and all(preservation.values())
        seed_results.append(
            {
                "seed": seed,
                "all_six_postedit_pure_P": all_six_pure_p,
                "Q_probe_shift_rules": q_rules,
                "preservation_rules": preservation,
                "deltas_vs_same_seed_noop": deltas,
                "joint_gate_passed": joint,
            }
        )
    joint_count = sum(bool(row["joint_gate_passed"]) for row in seed_results)
    return {
        "gate_contract": {
            "independent_q_restore_selective_Q_delta_at_most": -SHIFT,
            "independent_padding_sham_absolute_selective_Q_delta_at_most": SHIFT,
            "nested_q_transplant_selective_Q_delta_at_least": SHIFT,
            "nested_padding_sham_absolute_selective_Q_delta_at_most": SHIFT,
            "absolute_preservation_delta_at_most": PRESERVATION,
            "preservation_metrics": list(PRESERVATION_PATHS),
            "all_six_postedit_pure_P_required": True,
            "same_seed_joint_passes_required": MINIMUM_JOINT_SEEDS,
        },
        "seeds": seed_results,
        "joint_seed_pass_count": joint_count,
        "pilot_gate_passed": joint_count >= MINIMUM_JOINT_SEEDS,
        "decision": "PASS" if joint_count >= MINIMUM_JOINT_SEEDS else "STOP",
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_outputs(output_dir: Path, audit: Mapping[str, Any], gate: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": "E19 frozen active first-layer Q-input-pathway engineering pilot",
        "inference_status": "engineering_manipulation_gate_only_no_phase_b_outcomes",
        "audit": dict(audit),
        **dict(gate),
    }
    json_path = output_dir / "e19_pilot_gate.json"
    json_path.write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    rows: list[dict[str, Any]] = []
    for seed_row in cast(Sequence[Mapping[str, Any]], gate["seeds"]):
        deltas = cast(Mapping[str, Mapping[str, float]], seed_row["deltas_vs_same_seed_noop"])
        q_rules = cast(Mapping[str, bool], seed_row["Q_probe_shift_rules"])
        preservation = cast(Mapping[str, bool], seed_row["preservation_rules"])
        for branch in EDIT_BRANCHES:
            rows.append(
                {
                    "seed": seed_row["seed"],
                    "branch": branch,
                    **deltas[branch],
                    "Q_probe_shift_rule_passed": q_rules[branch],
                    "preservation_rule_passed": preservation[branch],
                    "all_six_postedit_pure_P": seed_row["all_six_postedit_pure_P"],
                    "same_seed_joint_gate_passed": seed_row["joint_gate_passed"],
                }
            )
    csv_path = output_dir / "e19_pilot_gate.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    runs, audit = load_and_audit(args.artifacts)
    gate = evaluate_gate(runs)
    write_outputs(args.output_dir, audit, gate)
    print(
        f"E19 PILOT {gate['decision']}: {gate['joint_seed_pass_count']}/3 "
        "same-seed joint gates passed"
    )
    return 0 if gate["pilot_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
