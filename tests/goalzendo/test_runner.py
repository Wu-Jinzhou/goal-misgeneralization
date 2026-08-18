from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

from goalzendo.artifacts import stable_hash
from goalzendo.config import DEFAULT_CONFIG, deep_merge, get_path
from goalzendo.runner import (
    G00_OPTIMIZER_STABILITY_POLICY,
    BackendContractError,
    BackendPreparation,
    BackendResult,
    LaunchGuardError,
    PlanError,
    assert_launch_unlocked,
    build_plan,
    create_g00_gate_artifact,
    derive_seed,
    evaluate_g00_gate,
    execute_plan,
    g00_evidence_binding,
    run_one,
    target_gate_binding,
    verify_g00_gate_artifact,
)


def _config(**run: object) -> dict[str, object]:
    return deep_merge(
        DEFAULT_CONFIG,
        {
            "experiment": {"id": "gtest", "name": "runner_contract"},
            "run": {"seeds": [7, 3], **run},
            "data": {"n_train": 100, "q_p": 0.99, "q_q": 0.9},
        },
    )


def _repo(tmp_path: Path) -> Path:
    source = tmp_path / "repo" / "src" / "goalzendo"
    source.mkdir(parents=True)
    (source / "runner_impl.py").write_text("VERSION = 1\n", encoding="utf-8")
    return tmp_path / "repo"


def _passing_gate_assessment() -> dict[str, object]:
    per_run = [
        {"run_id": "capability-run", "seed": 9001, "law_family": "parity", "A": 0.95, "B": 0.95}
    ]
    model_identity = {
        "requested_model": str(DEFAULT_CONFIG["model"]["name"]),
        "requested_revision": str(DEFAULT_CONFIG["model"]["revision"]),
    }

    def candidate(learning_rate: float, *, rl: bool) -> dict[str, object]:
        runs = [
            {
                "run_id": f"{family}-{q_p}-{seed}-{'rl' if rl else 'sft'}-{learning_rate}",
                "law_family": family,
                "q_p": q_p,
                "seed": seed,
                "baseline_step": 0,
                "baseline_iid_accuracy": 0.80,
                "terminal_iid_accuracy": {"128": 0.90, "256": 0.90},
                "terminal_action_b_rate": {"128": 0.50, "256": 0.50},
                "early_sampling_steps": list(range(1, 33)) if rl else None,
                "early_both_actions_sampled_median": 0.25 if rl else None,
                "early_all_zero_advantages_median": 0.75 if rl else None,
            }
            for family in ("majority", "parity")
            for q_p in (0.95, 1.0)
            for seed in (9001, 9002, 9003)
        ]
        return {
            "learning_rate": learning_rate,
            "entropy_coefficient": 0.0,
            "stability_window_steps": [128, 256],
            "run_measurements": runs,
            "model_identity": model_identity,
            "update_method": "full",
            "effective_batch": 50,
            "law_families": ["majority", "parity"],
            "proxy_accuracies": [0.95, 1.0],
            "seeds": [9001, 9002, 9003],
            "n_runs": 12,
        }

    return {
        "rule_adapter_position_agreement": {
            name: {"A": 0.95, "B": 0.95}
            for name in ("law_only", "audit_law_matched", "sage_only", "herald_only")
        },
        "law_only_position_agreement_by_run": per_run,
        "matched_law_position_agreement_by_run": [
            {**per_run[0], "run_id": "matched-capability-run"}
        ],
        "constrained_scorer": {
            "finite_probability_fraction": 1.0,
            "maximum_probability_sum_error": 1e-6,
            "maximum_absolute_position_bias": 0.02,
        },
        "no_signal": {"correct": 50, "trials": 100},
        "surface_classifier": {"correct": 50, "trials": 100},
        "optimizer_stability": {
            "sft": [candidate(1e-5, rl=False)],
            "outcome_rl": [candidate(3e-6, rl=True)],
        },
        "dataset_integrity": {
            "truth_cell_count_mismatches": 0,
            "scene_overlap_count": 0,
            "duplicate_pair_ids": 0,
            "regeneration_mismatches": 0,
        },
    }


def _g00_config(
    *,
    capability: bool = False,
    engineering_model: bool = False,
    target_pilot: bool = False,
) -> dict[str, object]:
    target_model = str(DEFAULT_CONFIG["model"]["name"])
    target_revision = str(DEFAULT_CONFIG["model"]["revision"])
    model_name = "engineering/model" if engineering_model else target_model
    model_revision = "engineering-commit" if engineering_model else target_revision
    views = (
        "law_only",
        "audit_law_matched",
        "sage_only",
        "herald_only",
        "no_signal",
        "surface_only",
    )
    return deep_merge(
        _config(seeds=[9001, 9002, 9003]),
        {
            "experiment": {
                "id": "g00",
                "name": (
                    f"capability-{'engineering' if engineering_model else 'target'}"
                    if capability
                    else (
                        "deployed_model_full_context_pilot" if target_pilot else "engineering"
                    )
                ),
                "status": "exploratory",
            },
            "run": {"launch_guard": None},
            "model": {
                "name": model_name,
                "revision": model_revision,
            },
            "data": {
                "law_features": [5, 6, 7],
                "sage_features": [8, 9],
            },
            "sweep": ({"data.q_p": [0.95, 1.0]} if target_pilot else {}),
            "cases": (
                [
                    *[{"data.rule_family": "parity", "data.training_view": view} for view in views],
                    *(
                        [
                            {"data.rule_family": "majority", "data.training_view": "law_only"},
                            {
                                "data.rule_family": "majority",
                                "data.training_view": "audit_law_matched",
                            },
                        ]
                        if not engineering_model
                        else []
                    ),
                ]
                if capability
                else (
                    [
                        {
                            "data.rule_family": family,
                            "train.algorithm": algorithm,
                            "train.learning_rate": learning_rate,
                        }
                        for family in ("parity", "majority")
                        for algorithm, learning_rate in (("sft", 1e-5), ("outcome_rl", 3e-6))
                    ]
                    if target_pilot
                    else []
                )
            ),
        },
    )


def _target_config(**run: object) -> dict[str, object]:
    return deep_merge(
        _config(seeds=[1103]),
        {
            "experiment": {"id": "g01", "name": "bound_target", "status": "prospective"},
            "run": {
                "launch_guard": "G00_NOT_PASSED__LEARNING_RATES_NOT_FROZEN",
                **run,
            },
            "cases": [
                {"train.algorithm": "sft", "train.learning_rate": 1e-5},
                {"train.algorithm": "outcome_rl", "train.learning_rate": 3e-6},
            ],
        },
    )


class _GateBackend:
    def prepare(self, context: object) -> BackendPreparation:
        config = context.config  # type: ignore[attr-defined]
        name = str(get_path(config, "model.name"))
        revision = str(get_path(config, "model.revision"))
        return BackendPreparation(
            dataset_metadata={"manifest_digest": f"data-{context.spec.plan_key}"},  # type: ignore[attr-defined]
            model_metadata={"requested_model": name, "requested_revision": revision},
            tokenizer_metadata={
                "requested_model": name,
                "requested_revision": revision,
                "eos_token_id": 1,
            },
        )

    def run(self, _context: object) -> BackendResult:
        return BackendResult(summary={"gate_evidence": True})


def test_seed_derivation_and_plan_are_deterministic_and_location_independent() -> None:
    assert derive_seed(3, "dataset") == derive_seed(3, "dataset")
    assert derive_seed(3, "dataset") != derive_seed(3, "training")
    config = _config(output_root="first", resume=True)
    changed = _config(output_root="second", resume=False, seeds=[3, 7])
    first = build_plan(config)
    second = build_plan(changed)
    assert [item.seed for item in first] == [3, 7]
    assert [item.plan_key for item in first] == [item.plan_key for item in second]
    assert [item.seeds for item in first] == [item.seeds for item in second]


def test_hash_shards_form_a_stable_partition() -> None:
    config = deep_merge(
        _config(),
        {"sweep": {"data.q_p": [0.9, 0.99], "data.rule_family": ["parity", "majority"]}},
    )
    whole = build_plan(config)
    shards = [build_plan(config, shard_index=index, num_shards=3) for index in range(3)]
    assert {item.plan_key for item in whole} == {item.plan_key for shard in shards for item in shard}
    assert sum(len(shard) for shard in shards) == len(whole)
    assert not (set(item.plan_key for item in shards[0]) & set(item.plan_key for item in shards[1]))


def test_duplicate_cells_fail_closed() -> None:
    config = deep_merge(
        _config(seeds=[3]),
        {"cases": [{"data.rule_family": "parity"}, {"data.rule_family": "parity"}]},
    )
    with pytest.raises(PlanError, match="duplicate"):
        build_plan(config)


class _Backend:
    def prepare(self, context: object) -> BackendPreparation:
        return BackendPreparation(
            dataset_metadata={"manifest_digest": "data-v1", "count": 100},
            model_metadata={"name": "tiny", "revision": "model-v1"},
            tokenizer_metadata={"name": "tiny", "revision": "tokenizer-v1", "eos_token_id": 1},
        )

    def run(self, context: object) -> BackendResult:
        return BackendResult(
            summary={"final_reward": 1.0},
            metrics=({"step": 0, "metric": "reward", "value": 0.5},),
            predictions=({"sample_id": "one", "prediction": "A"},),
        )


def test_callback_backend_writes_required_artifacts_and_skips_complete(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    spec = build_plan(_config(seeds=[3]))[0]
    first = run_one(spec, repo=repo, output_root=tmp_path / "artifacts", backend=_Backend())
    assert first.state == "complete"
    assert (Path(first.path) / "COMPLETE").is_file()
    assert json.loads((Path(first.path) / "summary.json").read_text())["plan_key"] == spec.plan_key

    second = run_one(spec, repo=repo, output_root=tmp_path / "artifacts", backend=_Backend())
    assert second.state == "skipped"


def test_failure_is_resumable_and_prior_metrics_are_visible(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    spec = build_plan(_config(seeds=[3]))[0]

    class Fails(_Backend):
        def run(self, context: object) -> BackendResult:
            context.append_metrics({"step": 1, "metric": "partial", "value": 1})  # type: ignore[attr-defined]
            raise RuntimeError("expected")

    with pytest.raises(RuntimeError, match="expected"):
        run_one(spec, repo=repo, output_root=tmp_path / "artifacts", backend=Fails())

    class Resumes(_Backend):
        def run(self, context: object) -> BackendResult:
            assert context.resumed is True  # type: ignore[attr-defined]
            assert len(context.prior_metrics) == 1  # type: ignore[attr-defined]
            return BackendResult(summary={"resumed": True})

    result = run_one(spec, repo=repo, output_root=tmp_path / "artifacts", backend=Resumes())
    assert result.state == "complete"


def test_backend_cannot_complete_without_exact_manifests(tmp_path: Path) -> None:
    spec = build_plan(_config(seeds=[3]))[0]
    with pytest.raises(BackendContractError, match="required exact metadata"):
        run_one(
            spec,
            repo=_repo(tmp_path),
            output_root=tmp_path / "artifacts",
            backend=lambda _context: {"summary": {"done": True}},
        )
    status = json.loads(next((tmp_path / "artifacts").glob("**/status.json")).read_text())
    assert status["state"] == "failed"


def test_launch_guard_allows_planning_but_refuses_execution_before_writes(tmp_path: Path) -> None:
    config = _config(
        seeds=[3],
        launch_guard="waiting for G00 pilot freeze",
        protocol_unlocked=False,
    )
    spec = build_plan(config)[0]
    dry = run_one(
        spec,
        repo=_repo(tmp_path),
        output_root=tmp_path / "artifacts",
        backend=None,
        dry_run=True,
    )
    assert dry.state == "planned"
    with pytest.raises(LaunchGuardError, match="waiting for G00"):
        run_one(
            spec,
            repo=tmp_path / "repo",
            output_root=tmp_path / "artifacts",
            backend=_Backend(),
        )
    assert not (tmp_path / "artifacts").exists()


def test_protocol_unlocked_override_never_bypasses_a_guard(tmp_path: Path) -> None:
    config = _target_config(protocol_unlocked=True)
    spec = build_plan(config)[0]
    with pytest.raises(LaunchGuardError, match="bound G00 gate artifact"):
        run_one(
            spec,
            repo=_repo(tmp_path),
            output_root=tmp_path / "target",
            backend=_Backend(),
        )
    assert not (tmp_path / "target").exists()


def test_missing_registered_guard_is_rejected_even_for_direct_config(tmp_path: Path) -> None:
    config = _target_config()
    config["run"]["launch_guard"] = None  # type: ignore[index]
    with pytest.raises(LaunchGuardError, match="missing or altered"):
        assert_launch_unlocked(config, repo=_repo(tmp_path))


def test_g00_gate_cannot_skip_the_g01_to_g02_selection_stage(tmp_path: Path) -> None:
    config = deep_merge(
        _target_config(),
        {
            "experiment": {"id": "g02", "name": "evidence_geometry"},
            "run": {
                "launch_guard": "G01_INFORMATIVE_CELL_AND_SETTINGS_NOT_FROZEN",
            },
        },
    )
    with pytest.raises(LaunchGuardError, match="separate G01 selection artifact"):
        target_gate_binding(config, _repo(tmp_path))


def test_gate_threshold_boundaries_are_numeric_and_all_six_are_required() -> None:
    assessment = _passing_gate_assessment()
    checks, selected = evaluate_g00_gate(assessment)
    assert len(checks) == 6
    assert all(check["passed"] for check in checks.values())
    assert selected["sft"]["learning_rate"] == pytest.approx(1e-5)

    mutations = (
        (
            "adapter",
            lambda value: value["rule_adapter_position_agreement"]["law_only"].__setitem__("A", 0.949),
        ),  # type: ignore[index,union-attr]
        (
            "finite",
            lambda value: value["constrained_scorer"].__setitem__("finite_probability_fraction", 0.999),
        ),  # type: ignore[union-attr]
        (
            "sum",
            lambda value: value["constrained_scorer"].__setitem__("maximum_probability_sum_error", 1.1e-6),
        ),  # type: ignore[union-attr]
        (
            "bias",
            lambda value: value["constrained_scorer"].__setitem__("maximum_absolute_position_bias", 0.021),
        ),  # type: ignore[union-attr]
        ("no_signal", lambda value: value["no_signal"].__setitem__("correct", 61)),  # type: ignore[union-attr]
        ("surface", lambda value: value["surface_classifier"].__setitem__("correct", 39)),  # type: ignore[union-attr]
        ("integrity", lambda value: value["dataset_integrity"].__setitem__("scene_overlap_count", 1)),  # type: ignore[union-attr]
    )
    for _name, mutate in mutations:
        changed = json.loads(json.dumps(assessment))
        mutate(changed)
        changed_checks, _selected = evaluate_g00_gate(changed)
        assert not all(check["passed"] for check in changed_checks.values())


@pytest.mark.parametrize(
    ("improvement", "regression", "frequency", "passes"),
    [
        (improvement, regression, frequency, improvement >= 0.10 and regression <= 0.05 and frequency <= 0.95)
        for improvement, regression, frequency in itertools.product(
            (0.10, 0.099), (0.05, 0.051), (0.95, 0.951)
        )
    ],
)
def test_optimizer_gate_boundaries(
    improvement: float,
    regression: float,
    frequency: float,
    passes: bool,
) -> None:
    assessment = _passing_gate_assessment()
    candidate = assessment["optimizer_stability"]["sft"][0]  # type: ignore[index]
    run = candidate["run_measurements"][0]
    terminal = 0.80 + improvement
    run["terminal_iid_accuracy"] = {"128": terminal + regression, "256": terminal}
    run["terminal_action_b_rate"] = {"128": frequency, "256": 0.50}
    if passes:
        checks, _selected = evaluate_g00_gate(assessment)
        assert checks["optimizer_stability"]["passed"] is True
    else:
        with pytest.raises(LaunchGuardError, match="no passing setting"):
            evaluate_g00_gate(assessment)


@pytest.mark.parametrize(
    ("initial", "terminal", "passes"),
    [
        (0.80, 0.90, True),
        (0.80, 0.899, False),
        (0.95, 0.93, True),
        (0.95, 0.929, False),
    ],
)
def test_optimizer_learning_gate_is_headroom_aware_per_run_class(
    initial: float,
    terminal: float,
    passes: bool,
) -> None:
    assessment = _passing_gate_assessment()
    candidate = assessment["optimizer_stability"]["sft"][0]  # type: ignore[index]
    run = candidate["run_measurements"][0]
    run["baseline_iid_accuracy"] = initial
    run["terminal_iid_accuracy"] = {"128": terminal, "256": terminal}
    if passes:
        checks, _ = evaluate_g00_gate(assessment)
        assert checks["optimizer_stability"]["passed"] is True
    else:
        with pytest.raises(LaunchGuardError, match="no passing setting"):
            evaluate_g00_gate(assessment)


@pytest.mark.parametrize(
    ("both", "zero", "passes"),
    [(0.25, 0.75, True), (0.249, 0.75, False), (0.25, 0.751, False)],
)
def test_rl_early_sampling_collapse_gate_boundaries(
    both: float,
    zero: float,
    passes: bool,
) -> None:
    assessment = _passing_gate_assessment()
    candidate = assessment["optimizer_stability"]["outcome_rl"][0]  # type: ignore[index]
    run = candidate["run_measurements"][0]
    run["early_both_actions_sampled_median"] = both
    run["early_all_zero_advantages_median"] = zero
    if passes:
        checks, _ = evaluate_g00_gate(assessment)
        assert checks["optimizer_stability"]["passed"] is True
    else:
        with pytest.raises(LaunchGuardError, match="no passing setting"):
            evaluate_g00_gate(assessment)


def test_optimizer_gate_rejects_support_holes_and_wrong_registered_steps() -> None:
    assessment = _passing_gate_assessment()
    candidate = assessment["optimizer_stability"]["sft"][0]  # type: ignore[index]
    candidate["run_measurements"].pop()
    with pytest.raises(LaunchGuardError, match="exactly 12 runs"):
        evaluate_g00_gate(assessment)

    assessment = _passing_gate_assessment()
    candidate = assessment["optimizer_stability"]["outcome_rl"][0]  # type: ignore[index]
    candidate["run_measurements"][0]["early_sampling_steps"] = list(range(2, 34))
    with pytest.raises(LaunchGuardError, match="exact RL updates"):
        evaluate_g00_gate(assessment)


def test_optimizer_gate_automatically_selects_the_smallest_passing_rate() -> None:
    assessment = _passing_gate_assessment()
    larger = json.loads(json.dumps(assessment["optimizer_stability"]["sft"][0]))
    larger["learning_rate"] = 3e-5
    for run in larger["run_measurements"]:
        run["run_id"] = f"larger-{run['run_id']}"
    assessment["optimizer_stability"]["sft"] = [larger, assessment["optimizer_stability"]["sft"][0]]
    checks, selected = evaluate_g00_gate(assessment)
    assert checks["optimizer_stability"]["passed"] is True
    assert selected["sft"]["learning_rate"] == pytest.approx(1e-5)


def test_matched_law_capability_is_checked_per_run() -> None:
    assessment = _passing_gate_assessment()
    assessment["matched_law_position_agreement_by_run"][0]["B"] = 0.94  # type: ignore[index]
    checks, _ = evaluate_g00_gate(assessment)
    assert checks["rule_adapters"]["passed"] is False


def test_capability_checks_fail_per_model_instead_of_averaging_commits() -> None:
    assessment = _passing_gate_assessment()
    first_identity = {"requested_model": "engineering", "requested_revision": "commit-a"}
    second_identity = {"requested_model": "target", "requested_revision": "commit-b"}
    first = {"model_identity": first_identity, **_passing_gate_assessment()}
    second = {"model_identity": second_identity, **_passing_gate_assessment()}
    second["rule_adapter_position_agreement"]["law_only"]["A"] = 0.0  # type: ignore[index]
    assessment["capability_by_model"] = {
        stable_hash(first_identity, 64): first,
        stable_hash(second_identity, 64): second,
    }
    checks, _selected = evaluate_g00_gate(assessment)
    assert checks["rule_adapters"]["passed"] is False
    assert set(checks["rule_adapters"]["capability_by_model"]) == {
        stable_hash(first_identity, 64),
        stable_hash(second_identity, 64),
    }


def test_passing_gate_is_bound_to_evidence_source_target_config_and_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _repo(tmp_path)
    g00_configs = [
        _g00_config(engineering_model=True),
        _g00_config(capability=True, engineering_model=True),
        _g00_config(capability=True),
        _g00_config(target_pilot=True),
    ]
    for config in g00_configs:
        for spec in build_plan(config):
            evidence = run_one(
                spec,
                repo=repo,
                output_root=tmp_path / "g00",
                backend=_GateBackend(),
            )
            assert evidence.state == "complete"
    target = _target_config()
    evidence_binding = g00_evidence_binding([tmp_path / "g00"], g00_configs, repo)
    measurements = _passing_gate_assessment()
    measurements["capability_by_model"] = {}
    for digest, identity in evidence_binding["model_identities"].items():
        model_panel = {"model_identity": identity, **_passing_gate_assessment()}
        model_panel["law_only_position_agreement_by_run"] = [
            {**record, "A": 0.95, "B": 0.95}
            for record in evidence_binding["law_only_model_coverage"].get(digest, [])
        ]
        model_panel["matched_law_position_agreement_by_run"] = [
            {**record, "A": 0.95, "B": 0.95}
            for record in evidence_binding["matched_law_model_coverage"].get(digest, [])
        ]
        measurements["capability_by_model"][digest] = model_panel
    assessment_body = {
        "schema": "goalzendo.g00_assessment",
        "schema_version": 2,
        "evidence_run_binding_digest": evidence_binding["run_binding_digest"],
        "evidence_source_fingerprint": evidence_binding["source_fingerprint"],
        "evidence_config_digest": evidence_binding["expected_config_digest"],
        "optimizer_stability_policy": G00_OPTIMIZER_STABILITY_POLICY.as_dict(),
        "optimizer_stability_policy_digest": stable_hash(
            G00_OPTIMIZER_STABILITY_POLICY.as_dict(), 64
        ),
        "measurements": measurements,
        "derivation": {"source": "completed_metrics_predictions_and_manifests"},
    }
    assessment = {
        **assessment_body,
        "assessment_digest": stable_hash(assessment_body, 64),
    }
    import goalzendo.analysis as analysis_module

    monkeypatch.setattr(
        analysis_module,
        "derive_g00_gate_assessment",
        lambda *_args, **_kwargs: assessment,
    )
    with pytest.raises(LaunchGuardError, match="artifact-derived"):
        create_g00_gate_artifact(
            _passing_gate_assessment(),
            artifact_roots=[tmp_path / "g00"],
            g00_configs=g00_configs,
            target_configs=[target],
            repo=repo,
            output=tmp_path / "unbound-gate.json",
        )
    edited = json.loads(json.dumps(assessment))
    edited["measurements"]["optimizer_stability"]["sft"][0]["run_measurements"][0][
        "terminal_iid_accuracy"
    ]["128"] = 0.91
    edited_body = {key: value for key, value in edited.items() if key != "assessment_digest"}
    edited["assessment_digest"] = stable_hash(edited_body, 64)
    with pytest.raises(LaunchGuardError, match="exactly match"):
        create_g00_gate_artifact(
            edited,
            artifact_roots=[tmp_path / "g00"],
            g00_configs=g00_configs,
            target_configs=[target],
            repo=repo,
            output=tmp_path / "edited-gate.json",
        )
    gate = create_g00_gate_artifact(
        assessment,
        artifact_roots=[tmp_path / "g00"],
        g00_configs=g00_configs,
        target_configs=[target],
        repo=repo,
        output=tmp_path / "g00-gate.json",
    )
    assert gate.passed is True
    result = run_one(
        build_plan(target)[0],
        repo=repo,
        output_root=tmp_path / "g01",
        backend=_Backend(),
        gate_artifact=gate.path,
    )
    assert result.state == "complete"

    tampered_payload = json.loads(gate.path.read_text(encoding="utf-8"))
    tampered_payload["optimizer_stability_policy"]["terminal_steps"] = [64, 256]
    tampered_body = {
        key: value for key, value in tampered_payload.items() if key != "gate_digest"
    }
    tampered_payload["gate_digest"] = stable_hash(tampered_body, 64)
    tampered_path = tmp_path / "tampered-policy-gate.json"
    tampered_path.write_text(json.dumps(tampered_payload), encoding="utf-8")
    with pytest.raises(LaunchGuardError, match="stability policy"):
        verify_g00_gate_artifact(tampered_path, config=target, repo=repo)

    evidence_free = json.loads(gate.path.read_text(encoding="utf-8"))
    evidence_free.pop("assessment")
    evidence_free.pop("selected_optimizer_settings")
    evidence_free_body = {
        key: value for key, value in evidence_free.items() if key != "gate_digest"
    }
    evidence_free["gate_digest"] = stable_hash(evidence_free_body, 64)
    evidence_free_path = tmp_path / "evidence-free-gate.json"
    evidence_free_path.write_text(json.dumps(evidence_free), encoding="utf-8")
    with pytest.raises(LaunchGuardError, match="assessment"):
        verify_g00_gate_artifact(evidence_free_path, config=target, repo=repo)

    forged_checks = json.loads(gate.path.read_text(encoding="utf-8"))
    forged_checks["checks"]["optimizer_stability"]["candidates"] = {}
    forged_checks_body = {
        key: value for key, value in forged_checks.items() if key != "gate_digest"
    }
    forged_checks["gate_digest"] = stable_hash(forged_checks_body, 64)
    forged_checks_path = tmp_path / "forged-checks-gate.json"
    forged_checks_path.write_text(json.dumps(forged_checks), encoding="utf-8")
    with pytest.raises(LaunchGuardError, match="checks do not exactly match"):
        verify_g00_gate_artifact(forged_checks_path, config=target, repo=repo)

    nonexistent_evidence = json.loads(gate.path.read_text(encoding="utf-8"))
    nonexistent_evidence["verification_inputs"]["artifact_roots"] = [
        str(tmp_path / "does-not-exist")
    ]
    nonexistent_evidence["verification_inputs_digest"] = stable_hash(
        nonexistent_evidence["verification_inputs"], 64
    )
    nonexistent_body = {
        key: value for key, value in nonexistent_evidence.items() if key != "gate_digest"
    }
    nonexistent_evidence["gate_digest"] = stable_hash(nonexistent_body, 64)
    nonexistent_path = tmp_path / "nonexistent-evidence-gate.json"
    nonexistent_path.write_text(json.dumps(nonexistent_evidence), encoding="utf-8")
    with pytest.raises(LaunchGuardError, match="evidence root is missing"):
        verify_g00_gate_artifact(nonexistent_path, config=target, repo=repo)

    changed_model = deep_merge(target, {"model": {"revision": "different-immutable-revision"}})
    with pytest.raises(LaunchGuardError, match="exact target"):
        run_one(
            build_plan(changed_model)[0],
            repo=repo,
            output_root=tmp_path / "changed-model",
            backend=_Backend(),
            gate_artifact=gate.path,
        )
    assert not (tmp_path / "changed-model").exists()


def test_execute_plan_can_isolate_ordinary_backend_failures(tmp_path: Path) -> None:
    plan = build_plan(_config(seeds=[3, 7]))

    class OneFails(_Backend):
        def run(self, context: object) -> BackendResult:
            if context.seed == 3:  # type: ignore[attr-defined]
                raise RuntimeError("one seed failed")
            return BackendResult(summary={"done": True})

    outcomes = execute_plan(
        plan,
        repo=_repo(tmp_path),
        output_root=tmp_path / "artifacts",
        backend=OneFails(),
        continue_on_error=True,
    )
    assert [outcome.state for outcome in outcomes] == ["failed", "complete"]


def test_dry_run_never_needs_or_imports_a_backend(tmp_path: Path) -> None:
    outcomes = execute_plan(
        build_plan(_config(seeds=[3])),
        repo=_repo(tmp_path),
        output_root=tmp_path / "artifacts",
        backend_reference="does.not.exist:backend",
        dry_run=True,
    )
    assert outcomes[0].state == "planned"
    assert not (tmp_path / "artifacts").exists()
