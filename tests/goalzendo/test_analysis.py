from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from goalzendo.analysis import (
    AnalysisPanel,
    ControllerCriteria,
    PanelCompletenessError,
    ProvenanceError,
    _confirmatory_summaries,
    _endpoint_units,
    acquisition_intervals,
    classify_controllers,
    equivalence_interval_90,
    export_analysis_tables,
    holm_adjust,
    intention_to_train_ledger,
    load_analysis_panel,
    paired_seed_contrast,
    seed_block_bootstrap,
    seed_level_trajectories,
    walsh_coefficients,
)
from goalzendo.artifacts import RunStore
from goalzendo.config import DEFAULT_CONFIG, deep_merge
from goalzendo.runner import build_plan


def _config(seeds: list[int] | None = None) -> dict[str, object]:
    return deep_merge(
        DEFAULT_CONFIG,
        {
            "experiment": {"id": "ganalysis", "name": "synthetic_panel"},
            "run": {"seeds": seeds or [11, 13, 17]},
            "data": {
                "n_train": 100,
                "n_validation": 100,
                "q_p": 0.9,
                "q_q": 0.9,
            },
            "train": {"steps": 4, "eval_steps": [0, 2, 4]},
            "evaluation": {"prompt_views": ["full"]},
        },
    )


def _repo(tmp_path: Path, version: int = 1) -> Path:
    repo = tmp_path / f"repo-{version}"
    source = repo / "src" / "goalzendo"
    source.mkdir(parents=True)
    (source / "implementation.py").write_text(f"VERSION = {version}\n", encoding="utf-8")
    return repo


def _write_run(
    root: Path,
    repo: Path,
    config: dict[str, object],
    seed: int,
    *,
    omit_target: str | None = None,
) -> Path:
    spec = next(item for item in build_plan(config) if item.seed == seed)
    store = RunStore(root, spec.config, seed, repo)
    store.initialize()
    store.record_dataset_metadata({"manifest_digest": f"dataset-{seed}", "count": 100})
    model_name = str(config["model"]["name"])  # type: ignore[index]
    revision = str(config["model"]["revision"])  # type: ignore[index]
    store.record_model_metadata({"requested_model": model_name, "requested_revision": revision})
    store.record_tokenizer_metadata(
        {
            "requested_model": model_name,
            "requested_revision": revision,
            "eos_token_id": 1,
        }
    )
    seed_offset = (seed - 13) / 1000
    for step in (0, 2, 4):
        diagnostic_split = "final_factorial" if step == 4 else "diagnostic_factorial"
        progress = step / 4
        store.append_metrics(
            {
                "step": step,
                "split": diagnostic_split,
                "kind": "behavioral_agreement",
                "prompt_view": "full",
                "panel": "conflict",
                "agreement_y": 0.5 + 0.5 * progress + seed_offset,
                "agreement_p": 1.0 - 0.5 * progress - seed_offset,
                "agreement_q": 0.5,
            }
        )
        for target in ("y", "p", "q", "d"):
            if target == omit_target:
                continue
            effect = {
                "y": progress,
                "p": 1.0 - progress,
                "q": 0.0,
                "d": 0.0,
            }[target]
            store.append_metrics(
                {
                    "step": step,
                    "split": f"{diagnostic_split}_causal",
                    "kind": "intervention_summary",
                    "prompt_view": "full",
                    "target": target,
                    "action_flip_rate": effect,
                    "mean_absolute_delta_margin": effect * 4,
                    "mean_target_aligned_delta_margin": (effect * 4 if target != "d" else None),
                    "n_pairs": 32,
                }
            )
        if omit_target is None:
            store.append_metrics(
                {
                    "step": step,
                    "split": diagnostic_split,
                    "kind": "checkpoint_summary",
                    "prompt_view": "full",
                    "panel": "conflict",
                    "rho_y": 0.5 + 0.5 * progress + seed_offset,
                    "rho_p": 1.0 - 0.5 * progress - seed_offset,
                    "rho_q": 0.5,
                    "causal_y": progress * 4,
                    "causal_p": (1.0 - progress) * 4,
                    "causal_q": 0.0,
                    "causal_d": 0.0,
                    "causal_y_flip_rate": progress,
                    "causal_p_flip_rate": 1.0 - progress,
                    "causal_q_flip_rate": 0.0,
                    "causal_d_flip_rate": 0.0,
                }
            )
        for choice_y, choice_p, choice_q in itertools.product((0, 1), repeat=3):
            store.append_metrics(
                {
                    "step": step,
                    "split": diagnostic_split,
                    "kind": "factorial_cell",
                    "prompt_view": "full",
                    "choice_y": choice_y,
                    "choice_p": choice_p,
                    "choice_q": choice_q,
                    "action_b_rate": float(choice_y if step == 4 else choice_p),
                    "n": 32,
                }
            )
    store.finalize(
        {
            "run_id": store.run_id,
            "seed": seed,
            "plan_key": spec.plan_key,
            "final_reward": 1.0,
        }
    )
    return store.path


def _write_panel(tmp_path: Path, *, omit_target: str | None = None) -> tuple[Path, dict[str, object]]:
    root = tmp_path / "artifacts"
    config = _config()
    repo = _repo(tmp_path)
    for seed in (11, 13, 17):
        _write_run(root, repo, config, seed, omit_target=omit_target)
    return root, config


def test_completed_artifacts_become_seed_level_trajectories(tmp_path: Path) -> None:
    root, config = _write_panel(tmp_path)
    panel = load_analysis_panel(
        root,
        expected_config=config,
        require_complete_metrics=True,
    )
    assert panel.audit["panel_complete"] is True
    assert len(panel.runs) == 3
    assert len(panel.factorial) == 3 * 3 * 8
    assert len(panel.causal_effects) == 3 * 3 * 4

    trajectories = seed_level_trajectories(panel)
    assert len(trajectories) == 9
    assert {"rho_y", "rho_p", "rho_q", "causal_y", "causal_p", "causal_q"}.issubset(trajectories)
    assert "causal_signed_margin_y" in trajectories
    assert trajectories.groupby("seed")["step"].apply(list).iloc[0] == [0, 2, 4]


def test_analysis_export_is_seed_level_audited_and_claim_neutral(tmp_path: Path) -> None:
    root, config = _write_panel(tmp_path)
    panel = load_analysis_panel(root, expected_config=config, require_complete_metrics=True)
    exports = export_analysis_tables(panel, tmp_path / "analysis")

    expected = {
        "runs",
        "seed_trajectories",
        "causal_effects",
        "factorial_cells",
        "walsh_coefficients",
        "controller_classifications",
        "acquisition_intervals",
        "manifest",
    }
    assert set(exports.files) == expected
    assert all(path.is_file() for path in exports.files.values())
    trajectory = pd.read_csv(exports.files["seed_trajectories"])
    assert len(trajectory) == 3 * 3
    assert trajectory.groupby(["run_id", "step"]).size().eq(1).all()
    walsh = pd.read_csv(exports.files["walsh_coefficients"])
    assert len(walsh) == 3 * 3
    assert walsh["max_reconstruction_error"].max() < 1e-12
    manifest = json.loads(exports.files["manifest"].read_text(encoding="utf-8"))
    assert manifest["analysis_contract"]["replication_unit"] == "training_seed"
    assert manifest["analysis_contract"]["automatic_scientific_contrasts"] is False
    assert manifest["tables"]["seed_trajectories"]["rows"] == 9
    for table in manifest["tables"].values():
        exported = exports.output_dir / table["file"]
        observed = hashlib.sha256(exported.read_bytes()).hexdigest()
        assert table["sha256"] == observed


def test_missing_seed_and_missing_causal_panel_fail_closed(tmp_path: Path) -> None:
    config = _config()
    root = tmp_path / "missing-seed"
    repo = _repo(tmp_path)
    for seed in (11, 13):
        _write_run(root, repo, config, seed)
    with pytest.raises(PanelCompletenessError, match="incomplete"):
        load_analysis_panel(root, expected_config=config)

    incomplete_root, complete_config = _write_panel(tmp_path / "missing-target", omit_target="q")
    with pytest.raises(PanelCompletenessError, match="causal_q"):
        load_analysis_panel(
            incomplete_root,
            expected_config=complete_config,
            require_complete_metrics=True,
        )


def test_mixed_source_fingerprints_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    config = _config([11, 13])
    _write_run(root, _repo(tmp_path, 1), config, 11)
    _write_run(root, _repo(tmp_path, 2), config, 13)
    with pytest.raises(ProvenanceError, match="one source fingerprint"):
        load_analysis_panel(root, expected_config=config)


def test_controller_labels_require_two_checkpoints_and_acquisition_is_censored() -> None:
    rows = []
    for step, controller in enumerate(("P", "P", "Y", "Y")):
        row = {
            "run_id": "run",
            "seed": 1,
            "step": step,
            "prompt_view": "full",
            "panel": "conflict",
            "causal_d": 0.0,
        }
        for candidate in ("y", "p", "q"):
            active = controller.lower() == candidate
            row[f"rho_{candidate}"] = 1.0 if active else 0.5
            row[f"causal_{candidate}"] = 1.0 if active else 0.0
        rows.append(row)
    classified = classify_controllers(
        pd.DataFrame(rows),
        criteria=ControllerCriteria(persistence=2),
    )
    assert classified["raw_controller"].tolist() == ["P", "P", "Y", "Y"]
    assert classified["controller"].tolist() == [None, "P", None, "Y"]

    proxy = acquisition_intervals(classified, "P")
    assert proxy.iloc[0]["censoring"] == "left"
    assert proxy.iloc[0]["right_step"] == 0
    assert proxy.iloc[0]["confirmation_step"] == 1
    intended = acquisition_intervals(classified, "Y")
    assert intended.iloc[0]["censoring"] == "interval"
    assert intended.iloc[0]["left_step"] == 1
    assert intended.iloc[0]["right_step"] == 2


@pytest.mark.parametrize(
    ("behavior_pass", "causal_pass", "behavior_margin_pass", "causal_margin_pass"),
    list(itertools.product((False, True), repeat=4)),
)
def test_controller_requires_all_four_raw_boundary_checks_independently(
    behavior_pass: bool,
    causal_pass: bool,
    behavior_margin_pass: bool,
    causal_margin_pass: bool,
) -> None:
    rho_y = 0.90 if behavior_pass else 0.899
    causal_y = 0.50 if causal_pass else 0.499
    frame = pd.DataFrame(
        [
            {
                "run_id": "r",
                "seed": 1,
                "step": 0,
                "prompt_view": "full",
                "panel": "conflict",
                "rho_y": rho_y,
                "rho_p": rho_y - (0.10 if behavior_margin_pass else 0.099),
                "rho_q": 0.0,
                "causal_y": causal_y,
                "causal_p": causal_y - (0.10 if causal_margin_pass else 0.099),
                "causal_q": 0.0,
                "causal_d": 1.0,
            }
        ]
    )
    classified = classify_controllers(frame, criteria=ControllerCriteria(persistence=1))
    expected = "Y" if all((behavior_pass, causal_pass, behavior_margin_pass, causal_margin_pass)) else "none"
    assert classified.iloc[0]["raw_controller"] == expected
    assert classified.iloc[0]["distractor_flip_rate"] == pytest.approx(1.0)
    assert not any(column.startswith("adjusted_causal") for column in classified)


def test_unconfirmed_final_controller_remains_right_censored() -> None:
    frame = pd.DataFrame(
        {
            "run_id": ["r"] * 3,
            "prompt_view": ["full"] * 3,
            "panel": ["conflict"] * 3,
            "step": [0, 4, 8],
            "raw_controller": ["P", "P", "Y"],
        }
    )
    interval = acquisition_intervals(frame, "Y", persistence=2).iloc[0]
    assert interval["censoring"] == "right_unconfirmed_tail"
    assert interval["left_step"] == 4
    assert np.isnan(interval["right_step"])


def test_paired_seed_inference_equivalence_holm_and_block_bootstrap() -> None:
    frame = pd.DataFrame(
        [
            {"seed": seed, "condition": condition, "value": base + effect}
            for seed, base, effect in (
                (1, 0.50, 0.01),
                (2, 0.55, 0.00),
                (3, 0.45, -0.01),
            )
            for condition in ("control", "treatment")
            for effect in ([0.0] if condition == "control" else [effect])
        ]
    )
    contrast = paired_seed_contrast(
        frame,
        value_column="value",
        condition_column="condition",
        treatment="treatment",
        control="control",
        bootstrap_draws=500,
    )
    assert len(contrast.per_seed) == 3
    assert contrast.summary["estimate"] == pytest.approx(0.0)
    assert contrast.summary["equivalence_90"]["equivalent"] is True
    assert equivalence_interval_90([0.0, 0.0], draws=100)["equivalent"] is None
    assert equivalence_interval_90([0.075, 0.075, 0.075], draws=100)["equivalent"] is True

    adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06})

    blocks = pd.DataFrame(
        {
            "seed": [1, 1, 2, 2, 3, 3],
            "value": [0.0, 2.0, 1.0, 3.0, 2.0, 4.0],
        }
    )
    result = seed_block_bootstrap(
        blocks,
        lambda sampled: float(
            sampled.groupby("_bootstrap_seed_block" if "_bootstrap_seed_block" in sampled else "seed")[
                "value"
            ]
            .mean()
            .mean()
        ),
        draws=200,
    )
    assert result["estimate"] == pytest.approx(2.0)
    assert result["n_seeds"] == 3


def _factorial_policy(kind: str) -> pd.DataFrame:
    rows = []
    for y, p, q in itertools.product((0, 1), repeat=3):
        if kind == "y":
            output = y
        elif kind == "y_xor_p":
            output = int((2 * y - 1) * (2 * p - 1) > 0)
        else:
            raise AssertionError(kind)
        rows.append(
            {
                "run_id": "run",
                "seed": 1,
                "step": 4,
                "prompt_view": "full",
                "choice_y": y,
                "choice_p": p,
                "choice_q": q,
                "action_b_rate": output,
            }
        )
    return pd.DataFrame(rows)


def test_saturated_walsh_coefficients_identify_pure_and_conditional_policies() -> None:
    pure = walsh_coefficients(_factorial_policy("y")).iloc[0]
    assert pure["walsh_y"] == pytest.approx(1.0)
    assert pure["truth_table_structure"] == "pure_Y"
    assert pure["max_reconstruction_error"] < 1e-12

    conditional = walsh_coefficients(_factorial_policy("y_xor_p")).iloc[0]
    assert conditional["walsh_y_p"] == pytest.approx(1.0)
    assert conditional["truth_table_structure"] == "conditional_or_interaction"
    assert conditional["conditionality_index"] == pytest.approx(1.0)

    with pytest.raises(PanelCompletenessError, match=r"2\^3"):
        walsh_coefficients(_factorial_policy("y").iloc[:-1])


def test_confirmatory_endpoint_units_include_iid_prompt_acquisition_and_pairing() -> None:
    run_rows: list[dict[str, object]] = []
    trajectory_rows: list[dict[str, object]] = []
    raw_rows: list[dict[str, object]] = []
    ledger_rows: list[dict[str, object]] = []
    acquisitions: list[dict[str, object]] = []
    run_index = 0
    for q_p in (0.95, 1.0):
        for algorithm, rho_y in (("sft", 0.35), ("outcome_rl", 0.45)):
            run_index += 1
            run_id = f"run-{run_index}"
            config = deep_merge(
                _config([11]),
                {
                    "experiment": {"id": "g01"},
                    "data": {"rule_family": "parity", "q_p": q_p},
                    "train": {"algorithm": algorithm, "steps": 4},
                    "evaluation": {"prompt_views": ["full", "law_only"]},
                },
            )
            run_rows.append(
                {
                    "run_id": run_id,
                    "config": config,
                    "config.data.rule_family": "parity",
                    "config.data.q_p": q_p,
                    "config.data.q_q": 0.9,
                    "config.train.algorithm": algorithm,
                }
            )
            for prompt_view, view_rho_y in (("full", rho_y), ("law_only", 0.85)):
                trajectory_rows.append(
                    {
                        "run_id": run_id,
                        "step": 4,
                        "split": "final_factorial",
                        "prompt_view": prompt_view,
                        "panel": "conflict",
                        "rho_y": view_rho_y,
                        "rho_p": 0.90 if prompt_view == "full" else 0.50,
                        "rho_q": 0.50,
                        "causal_y": 0.10,
                        "causal_p": 0.80,
                        "causal_q": 0.0,
                    }
                )
            raw_rows.append(
                {
                    "run_id": run_id,
                    "step": 4,
                    "split": "iid_validation",
                    "prompt_view": "full",
                    "kind": "behavioral_agreement",
                    "panel": "all",
                    "rho_y": 0.99,
                }
            )
            ledger_rows.append(
                {
                    "plan_key": f"plan-{run_index}",
                    "run_id": run_id,
                    "seed": 11,
                    "law_family": "parity",
                    "q_p": q_p,
                    "q_q": 0.9,
                    "algorithm": algorithm,
                    "state": "complete",
                }
            )
            if q_p == 0.95:
                p_step, y_step, q_step = (1, 3, 2) if algorithm == "sft" else (2, 3, 1)
                for candidate, step in (("P", p_step), ("Y", y_step), ("Q", q_step)):
                    acquisitions.append(
                        {
                            "run_id": run_id,
                            "candidate": candidate,
                            "right_step": step,
                        }
                    )
    panel = AnalysisPanel(
        runs=pd.DataFrame(run_rows),
        raw_metrics=pd.DataFrame(raw_rows),
        predictions=pd.DataFrame(),
        trajectory=pd.DataFrame(trajectory_rows),
        causal_effects=pd.DataFrame(),
        factorial=pd.DataFrame(),
        audit={},
    )
    units = _endpoint_units(panel, pd.DataFrame(ledger_rows), pd.DataFrame(acquisitions))
    assert units["endpoint"].value_counts().to_dict() == {
        "availability_use": 4,
        "algorithm_control": 2,
        "dissociation": 2,
        "acquisition_order": 2,
    }
    dissociation = units[units["endpoint"] == "dissociation"]
    assert dissociation["qualified_iid_performance"].all()
    assert set(dissociation["iid_accuracy"]) == {0.99}
    assert set(units[units["endpoint"] == "acquisition_order"]["value"]) == {0.0, 1.0}
    availability = units[units["endpoint"] == "availability_use"]
    assert set(np.round(availability["value"], 2)) == {0.4, 0.5}
    algorithm = units[units["endpoint"] == "algorithm_control"]
    assert set(np.round(algorithm["value"], 2)) == {0.1}


def test_confirmatory_four_endpoint_family_holm_itt_and_equivalence() -> None:
    rows: list[dict[str, object]] = []
    endpoint_null = {
        "dissociation": 0.0,
        "acquisition_order": 0.5,
        "availability_use": 0.0,
        "algorithm_control": 0.0,
    }
    for law_family, seed, endpoint in itertools.product(("parity", "majority"), (1, 2, 3), endpoint_null):
        observed = not (law_family == "majority" and seed == 3 and endpoint == "acquisition_order")
        rows.append(
            {
                "endpoint": endpoint,
                "seed": seed,
                "law_family": law_family,
                "value": endpoint_null[endpoint] + (0.02 if endpoint != "acquisition_order" else 0.1),
                "null_value": endpoint_null[endpoint],
                "outcome_observed": observed,
                "missing_lower": 0.0 if endpoint == "acquisition_order" else -1.0,
                "missing_upper": 1.0,
                "secondary_value": 0.05 if endpoint == "dissociation" else np.nan,
            }
        )
    summaries = _confirmatory_summaries(pd.DataFrame(rows), bootstrap_draws=200)
    assert len(summaries) == 12
    assert summaries.groupby("law_scope")["endpoint"].nunique().eq(4).all()
    assert summaries["p_value_holm"].between(0, 1).all()
    majority_acquisition = summaries[
        (summaries["law_scope"] == "majority") & (summaries["endpoint"] == "acquisition_order")
    ].iloc[0]
    assert majority_acquisition["itt_missing_units"] == 1
    algorithm = summaries[summaries["endpoint"] == "algorithm_control"]
    assert set(algorithm["equivalence_margin"]) == {0.10}


def test_intention_to_train_ledger_keeps_failed_and_unattempted_runs(tmp_path: Path) -> None:
    config = deep_merge(
        _config([11, 13]),
        {
            # Generic intention-to-train ledger fixture, not a G01 launch.
            "experiment": {"id": "gtest-ledger"},
            "cases": [
                {"train.algorithm": "sft"},
                {"train.algorithm": "outcome_rl"},
            ],
        },
    )
    first = build_plan(config)[0]
    store = RunStore(tmp_path / "attempts", first.config, first.seed, _repo(tmp_path))
    store.initialize()
    store.fail(RuntimeError("optimizer failed"))
    ledger = intention_to_train_ledger(tmp_path / "attempts", config)
    assert len(ledger) == 4
    assert ledger["intention_to_train"].all()
    assert ledger["state"].value_counts().to_dict() == {
        "missing_not_attempted": 3,
        "failed": 1,
    }
    failed = ledger[ledger["state"] == "failed"].iloc[0]
    assert failed["error_type"] == "RuntimeError"
    assert failed["outcome_observed"] == np.False_
