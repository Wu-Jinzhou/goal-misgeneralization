"""Command-line interface for planning and running GoalZendo experiments."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from .artifacts import ArtifactError, RunStore, read_json, read_jsonl, write_json
from .config import (
    ConfigError,
    canonical_config,
    get_path,
    load_config,
    smoke_config,
    validate_config,
)
from .runner import (
    DEFAULT_BACKEND,
    BackendContractError,
    LaunchGuardError,
    PlanError,
    build_plan,
    create_g00_gate_artifact,
    derived_seeds,
    evaluate_g00_gate,
    run_experiments,
)


def _repo_root(start: str | Path | None = None) -> Path:
    current = Path(start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "goalzendo").is_dir():
            imported_source = Path(__file__).resolve().parent
            requested_source = (candidate / "src" / "goalzendo").resolve()
            if requested_source != imported_source:
                raise ConfigError(
                    "--repo/source discovery does not match the GoalZendo package actually imported"
                )
            return candidate
    raise ConfigError("could not locate the repository containing the imported GoalZendo source")


def _print_json(value: Any) -> None:
    print(json.dumps(value, sort_keys=True, allow_nan=False))


def _add_common_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("config", type=Path, help="GoalZendo YAML experiment specification")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="override one dotted config field (repeatable)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="goalzendo")
    parser.add_argument("--repo", type=Path, default=None, help=argparse.SUPPRESS)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate a specification without running it")
    _add_common_config_arguments(validate)
    validate.add_argument("--json", action="store_true", help="emit machine-readable output")

    plan = subparsers.add_parser("plan", help="expand sweeps, seeds, and deterministic shards")
    _add_common_config_arguments(plan)
    plan.add_argument("--smoke", action="store_true", help="plan the minimal valid smoke run")
    plan.add_argument("--shard-index", type=int, default=0)
    plan.add_argument("--num-shards", type=int, default=1)
    plan.add_argument("--output-root", type=Path, default=None)
    plan.add_argument("--include-config", action="store_true")

    generate = subparsers.add_parser("generate", help="materialize or preview an exact symbolic dataset")
    _add_common_config_arguments(generate)
    generate.add_argument(
        "--split",
        choices=("train", "validation", "factorial"),
        default="train",
    )
    generate.add_argument("--seed", type=int, default=None, help="master seed (default: first run seed)")
    generate.add_argument("--smoke", action="store_true")
    generate.add_argument("--output", type=Path, default=None, help="write the complete manifest JSON")
    generate.add_argument("--show", type=int, default=0, help="print the first N rendered prompts")

    inspect = subparsers.add_parser(
        "inspect", help="summarize a run directory, manifest, JSONL stream, or config"
    )
    inspect.add_argument("target", type=Path)

    run = subparsers.add_parser("run", help="execute or resume an experiment plan")
    _add_common_config_arguments(run)
    run.add_argument("--smoke", action="store_true")
    run.add_argument("--dry-run", action="store_true", help="resolve paths and IDs without writes")
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument("--num-shards", type=int, default=1)
    run.add_argument("--output-root", type=Path, default=None)
    run.add_argument("--backend", default=None, help="lazy backend reference module:attribute")
    run.add_argument("--continue-on-error", action="store_true")
    run.add_argument(
        "--gate-artifact",
        type=Path,
        default=None,
        help="passing digest-bound G00 gate required by guarded protocols",
    )

    gate = subparsers.add_parser(
        "gate",
        help="evaluate the six frozen G00 checks and bind passing evidence to target configs",
    )
    gate.add_argument("--g00-artifacts", type=Path, action="append", required=True)
    gate.add_argument("--g00-config", type=Path, action="append", required=True)
    gate.add_argument("--target-config", type=Path, action="append", required=True)
    gate.add_argument("--select-sft-learning-rate", type=float, required=True)
    gate.add_argument("--select-rl-learning-rate", type=float, required=True)
    gate.add_argument("--select-rl-entropy", type=float, default=0.0)
    gate.add_argument("--assessment-output", type=Path, default=None)
    gate.add_argument("--output", type=Path, required=True)

    analyze = subparsers.add_parser(
        "analyze",
        help="validate completed artifacts and export seed-level tables and figures",
    )
    analyze.add_argument("artifacts", type=Path, help="root containing completed run directories")
    analyze.add_argument(
        "--config",
        type=Path,
        required=True,
        help="prospective config used to verify the complete planned seed panel",
    )
    analyze.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="repeat every scientific override used for the run",
    )
    analyze.add_argument("--output", type=Path, required=True)
    analyze.add_argument("--prompt-view", default="full")
    analyze.add_argument("--behavior-panel", default="conflict")
    analyze.add_argument("--bootstrap-draws", type=int, default=10_000)
    analyze.add_argument("--confidence", type=float, default=0.95)
    analyze.add_argument("--dpi", type=int, default=240)
    analyze.add_argument("--no-figures", action="store_true")

    confirmatory = subparsers.add_parser(
        "confirmatory",
        help="run the frozen G01 endpoint, pairing, Holm, and ITT analysis",
    )
    confirmatory.add_argument("artifacts", type=Path, help="root containing all G01 run attempts")
    confirmatory.add_argument("--config", type=Path, required=True)
    confirmatory.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="repeat every scientific override used for the run",
    )
    confirmatory.add_argument("--output", type=Path, required=True)
    confirmatory.add_argument("--bootstrap-draws", type=int, default=10_000)

    reproduction = subparsers.add_parser(
        "reproduction-audit",
        help="fail closed on non-exact deterministic replica or training-prefix comparisons",
    )
    reproduction.add_argument("artifacts", type=Path)
    reproduction.add_argument("--config", type=Path, required=True)
    reproduction.add_argument("--output", type=Path, required=True)
    reproduction.add_argument(
        "--mode",
        choices=("exact-replicas", "paired-prefix"),
        required=True,
    )
    reproduction.add_argument("--prefix-step", type=int, default=128)
    reproduction.add_argument("--prompt-view", default="audit_law_matched")
    return parser


def _load(args: argparse.Namespace) -> dict[str, Any]:
    return load_config(args.config, args.overrides)


def _validate_command(args: argparse.Namespace) -> int:
    config = _load(args)
    cautions = validate_config(config)
    guard = get_path(config, "run.launch_guard", None)
    result = {
        "valid": True,
        "experiment": canonical_config(config).get("experiment", {}),
        "cautions": cautions,
        "launch_guard": guard,
        "gate_artifact_required": bool(guard),
        "protocol_unlocked_override_ignored": (
            bool(guard) and get_path(config, "run.protocol_unlocked", False) is True
        ),
    }
    if args.json:
        _print_json(result)
    else:
        print("valid")
        if guard:
            print(f"launch locked: {guard}")
        for caution in cautions:
            print(f"caution: {caution}")
    return 0


def _plan_rows(
    config: Mapping[str, Any],
    *,
    repo: Path,
    smoke: bool,
    shard_index: int,
    num_shards: int,
    output_root: Path | None,
    include_config: bool,
) -> list[dict[str, Any]]:
    plan = build_plan(
        config,
        smoke=smoke,
        shard_index=shard_index,
        num_shards=num_shards,
    )
    rows: list[dict[str, Any]] = []
    for spec in plan:
        root = output_root or Path(str(get_path(spec.config, "run.output_root", "artifacts-goalzendo")))
        store = RunStore(root, spec.config, spec.seed, repo)
        row = spec.as_dict(include_config=include_config)
        row.update(
            {
                "run_id": store.run_id,
                "path": str(store.path),
                "state": "complete" if store.complete else "pending",
                "launch_guard": get_path(spec.config, "run.launch_guard", None),
                "gate_artifact_required": bool(get_path(spec.config, "run.launch_guard", None)),
            }
        )
        rows.append(row)
    return rows


def _plan_command(args: argparse.Namespace, repo: Path) -> int:
    config = _load(args)
    rows = _plan_rows(
        config,
        repo=repo,
        smoke=args.smoke,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        output_root=args.output_root,
        include_config=args.include_config,
    )
    for row in rows:
        _print_json(row)
    print(json.dumps({"planned_runs": len(rows)}, sort_keys=True), file=sys.stderr)
    return 0


def _rule_from_config(config: Mapping[str, Any], *, sage: bool = False) -> Any:
    from .schema import RuleSpec

    prefix = "sage_" if sage else ""
    family = str(get_path(config, f"data.{prefix}rule_family", "parity"))
    if family == "atomic":
        family = "literal"
    feature_key = "sage_features" if sage else "law_features"
    name = "sage_rule" if sage else "official_law"
    config_prefix = "sage" if sage else "law"
    features = tuple(int(item) for item in get_path(config, f"data.{feature_key}", ()))
    expected = get_path(config, f"data.{config_prefix}_expected_values", None)
    return RuleSpec(
        family=family,
        feature_indices=features,
        expected_values=tuple(bool(item) for item in expected) if expected is not None else (),
        output_negated=bool(get_path(config, f"data.{config_prefix}_output_negated", False)),
        name=str(get_path(config, f"data.{config_prefix}_name", name)),
    )


def _generated_dataset(config: Mapping[str, Any], split: str, master_seed: int) -> Any:
    from .generation import (
        ErrorGeometry,
        generate_factorial_evaluation,
        generate_known_law_dataset,
    )

    rule = _rule_from_config(config)
    sage_rule = _rule_from_config(config, sage=True)
    feature_count = int(get_path(config, "data.feature_count"))
    seeds = derived_seeds(master_seed)
    if split == "factorial":
        return generate_factorial_evaluation(
            repeats=int(get_path(config, "data.n_eval_per_cell")),
            seed=seeds["factorial_evaluation"],
            rule=rule,
            sage_rule=sage_rule,
            feature_count=feature_count,
            split="factorial_eval",
        )
    n_key = "n_train" if split == "train" else "n_validation"
    seed_key = "dataset" if split == "train" else "validation"
    geometry = str(get_path(config, "data.error_geometry", "independent"))
    joint_count = None
    if geometry == "specified":
        geometry = "custom"
        joint_count = round(
            int(get_path(config, f"data.{n_key}")) * float(get_path(config, "data.joint_error_rate"))
        )
    return generate_known_law_dataset(
        n=int(get_path(config, f"data.{n_key}")),
        seed=seeds[seed_key],
        rule=rule,
        sage_rule=sage_rule,
        q_p=float(get_path(config, "data.q_p")),
        q_q=float(get_path(config, "data.q_q")),
        error_geometry=cast("ErrorGeometry", geometry),
        joint_error_count=joint_count,
        feature_count=feature_count,
        split=split,
    )


def _generate_command(args: argparse.Namespace) -> int:
    config = _load(args)
    if args.smoke:
        config = smoke_config(config)
    master_seed = int(args.seed) if args.seed is not None else int(sorted(get_path(config, "run.seeds"))[0])
    dataset = _generated_dataset(config, args.split, master_seed)
    summary = {
        "split": dataset.split,
        "n": len(dataset),
        "master_seed": master_seed,
        "dataset_seed": dataset.seed,
        "manifest_digest": dataset.manifest_digest,
        "law": dataset.law.as_dict(),
        "requested_q_p": dataset.requested_q_p,
        "requested_q_q": dataset.requested_q_q,
        "realized_q_p": dataset.realized_q_p,
        "realized_q_q": dataset.realized_q_q,
        "joint_error_count": dataset.joint_error_count,
        "error_geometry": dataset.error_geometry,
    }
    if args.output is not None:
        write_json(
            args.output,
            {"manifest_digest": dataset.manifest_digest, "dataset": dataset.manifest()},
        )
        summary["output"] = str(args.output.resolve())
    _print_json(summary)

    if args.show:
        from .rendering import RenderStyle, render_known_law_decision

        style = str(get_path(config, "data.renderer", "natural"))
        for decision in dataset.decisions[: max(0, int(args.show))]:
            print(
                "\n"
                + render_known_law_decision(
                    decision,
                    dataset.feature_names,
                    style=cast("RenderStyle", style),
                )
            )
    return 0


def _artifact_summary(path: Path) -> dict[str, Any]:
    status = read_json(path / "status.json") if (path / "status.json").is_file() else None
    identity = read_json(path / "identity.json")
    manifests: dict[str, Any] = {}
    for kind in ("dataset", "model", "tokenizer"):
        target = path / "manifests" / f"{kind}.json"
        if target.is_file():
            payload = read_json(target)
            manifests[kind] = {"digest": payload.get("digest"), "present": True}
        else:
            manifests[kind] = {"present": False}
    return {
        "type": "run",
        "path": str(path.resolve()),
        "run_id": identity.get("implementation_fingerprint", "") and status.get("run_id") if status else None,
        "complete": (path / "COMPLETE").is_file(),
        "status": status,
        "metrics_records": len(read_jsonl(path / "metrics.jsonl")),
        "prediction_records": len(read_jsonl(path / "predictions.jsonl")),
        "manifests": manifests,
        "implementation_fingerprint": identity.get("implementation_fingerprint"),
    }


def _inspect_command(args: argparse.Namespace) -> int:
    target = args.target.resolve()
    if target.is_dir() and (target / "identity.json").is_file():
        _print_json(_artifact_summary(target))
        return 0
    if not target.is_file():
        raise ConfigError(f"inspect target does not exist: {target}")
    suffix = target.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        config = load_config(target)
        _print_json(
            {
                "type": "config",
                "valid": True,
                "experiment": canonical_config(config).get("experiment", {}),
                "cautions": validate_config(config),
            }
        )
        return 0
    if suffix == ".jsonl":
        records = read_jsonl(target)
        _print_json({"type": "jsonl", "path": str(target), "records": len(records)})
        return 0
    if suffix == ".json":
        value = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ConfigError("JSON inspect target must contain an object")
        dataset = value.get("dataset", value)
        decisions = dataset.get("decisions", []) if isinstance(dataset, Mapping) else []
        _print_json(
            {
                "type": "dataset_manifest" if decisions else "json",
                "path": str(target),
                "manifest_digest": value.get("manifest_digest"),
                "decisions": len(decisions),
                "keys": sorted(str(key) for key in value),
            }
        )
        return 0
    raise ConfigError(f"unsupported inspect target: {target}")


def _run_command(args: argparse.Namespace, repo: Path) -> int:
    config = _load(args)
    backend_reference = args.backend or str(get_path(config, "run.backend", DEFAULT_BACKEND))
    outcomes = run_experiments(
        config,
        repo=repo,
        output_root=args.output_root,
        backend_reference=backend_reference,
        smoke=args.smoke,
        dry_run=args.dry_run,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        continue_on_error=args.continue_on_error,
        gate_artifact=args.gate_artifact,
    )
    for outcome in outcomes:
        _print_json(outcome.as_dict())
    failed = sum(outcome.state == "failed" for outcome in outcomes)
    _print_json({"runs": len(outcomes), "failed": failed})
    return 1 if failed else 0


def _gate_command(args: argparse.Namespace, repo: Path) -> int:
    from .analysis import derive_g00_gate_assessment

    g00_configs = [load_config(path) for path in args.g00_config]
    target_configs = [load_config(path) for path in args.target_config]
    selected = {
        "sft": {
            "learning_rate": float(args.select_sft_learning_rate),
            "entropy_coefficient": 0.0,
        },
        "outcome_rl": {
            "learning_rate": float(args.select_rl_learning_rate),
            "entropy_coefficient": float(args.select_rl_entropy),
        },
    }
    assessment = derive_g00_gate_assessment(
        args.g00_artifacts,
        g00_configs,
        repo=repo,
    )
    _checks, automatically_selected = evaluate_g00_gate(assessment["measurements"])
    for algorithm, requested in selected.items():
        observed = automatically_selected[algorithm]
        if (
            abs(float(requested["learning_rate"]) - float(observed["learning_rate"])) > 1e-15
            or abs(
                float(requested["entropy_coefficient"])
                - float(observed["entropy_coefficient"])
            )
            > 1e-15
        ):
            raise LaunchGuardError(
                f"requested {algorithm} setting is not the automatically selected smallest passing rate"
            )
    assessment_output = args.assessment_output or args.output.with_name(f"{args.output.stem}-assessment.json")
    write_json(assessment_output, assessment)
    gate = create_g00_gate_artifact(
        assessment,
        artifact_roots=args.g00_artifacts,
        g00_configs=g00_configs,
        target_configs=target_configs,
        repo=repo,
        output=args.output,
    )
    _print_json({**gate.as_dict(), "assessment": str(assessment_output.resolve())})
    return 0 if gate.passed else 1


def _analyze_command(args: argparse.Namespace) -> int:
    # Keep pandas and Matplotlib out of validation/planning/training processes.
    from .analysis import (
        AnalysisError,
        export_analysis_tables,
        load_analysis_panel,
    )
    from .plotting import PlotError, export_analysis_figures

    try:
        expected_config = load_config(args.config, args.overrides)
        panel = load_analysis_panel(
            args.artifacts,
            expected_config=expected_config,
            require_complete_metrics=True,
        )
        tables = export_analysis_tables(
            panel,
            args.output,
            behavior_panel=args.behavior_panel,
            prompt_view=args.prompt_view,
        )
        result: dict[str, Any] = {
            "status": "complete",
            "artifacts": str(args.artifacts.resolve()),
            "tables": tables.as_dict(),
            "figures": None,
        }
        if not args.no_figures:
            import pandas as pd  # type: ignore[import-untyped]

            trajectory = pd.read_csv(tables.files["seed_trajectories"])
            walsh = pd.read_csv(tables.files["walsh_coefficients"])
            figures = export_analysis_figures(
                trajectory,
                walsh,
                Path(args.output) / "figures",
                behavior_panel=args.behavior_panel,
                prompt_view=args.prompt_view,
                confidence=float(args.confidence),
                bootstrap_draws=int(args.bootstrap_draws),
                dpi=int(args.dpi),
            )
            result["figures"] = figures.as_dict()
    except (AnalysisError, PlotError) as error:
        raise ConfigError(str(error)) from error
    _print_json(result)
    return 0


def _confirmatory_command(args: argparse.Namespace) -> int:
    # Keep the analysis stack out of validation/planning/training processes.
    from .analysis import AnalysisError, run_g01_confirmatory_analysis

    try:
        expected_config = load_config(args.config, args.overrides)
        exports = run_g01_confirmatory_analysis(
            args.artifacts,
            expected_config,
            args.output,
            bootstrap_draws=int(args.bootstrap_draws),
        )
    except AnalysisError as error:
        raise ConfigError(str(error)) from error
    _print_json({"status": "complete", **exports.as_dict()})
    return 0


def _reproduction_command(args: argparse.Namespace, repo: Path) -> int:
    from .reproduction import (
        ReproductionError,
        audit_exact_replicas,
        audit_paired_prefix,
        write_reproduction_audit,
    )

    config = load_config(args.config)
    try:
        if args.mode == "exact-replicas":
            result = audit_exact_replicas(
                args.artifacts,
                config,
                repo=repo,
                snapshot_step=int(args.prefix_step),
            )
        else:
            result = audit_paired_prefix(
                args.artifacts,
                config,
                repo=repo,
                prefix_step=int(args.prefix_step),
                prompt_view=str(args.prompt_view),
            )
    except ReproductionError as error:
        raise ConfigError(str(error)) from error
    write_reproduction_audit(args.output, result)
    _print_json(result)
    return 0 if bool(result["passed"]) else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        repo = _repo_root(args.repo)
        if args.command == "validate":
            return _validate_command(args)
        if args.command == "plan":
            return _plan_command(args, repo)
        if args.command == "generate":
            return _generate_command(args)
        if args.command == "inspect":
            return _inspect_command(args)
        if args.command == "run":
            return _run_command(args, repo)
        if args.command == "gate":
            return _gate_command(args, repo)
        if args.command == "analyze":
            return _analyze_command(args)
        if args.command == "confirmatory":
            return _confirmatory_command(args)
        if args.command == "reproduction-audit":
            return _reproduction_command(args, repo)
        parser.error(f"unknown command: {args.command}")
    except (
        ArtifactError,
        ConfigError,
        PlanError,
        LaunchGuardError,
        BackendContractError,
        ValueError,
    ) as error:
        print(f"goalzendo: error: {error}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
