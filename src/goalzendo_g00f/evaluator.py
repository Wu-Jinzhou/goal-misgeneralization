"""Canonical, fail-closed G00-F remediation evaluator.

Every result is reconstructed from completion-attested run artifacts and the
canonical regenerated final banks.  Missing, failed, incomplete, or corrupt
intention-to-train rows produce a persisted false artifact rather than only an
exception.  A passing result still does not authorize G01.
"""

from __future__ import annotations

import copy
import math
import stat
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from goalzendo.artifacts import (
    RunStore,
    discover_runs,
    read_json,
    read_jsonl,
    stable_hash,
    verify_completion_attestation,
)
from goalzendo.config import canonical_config, get_path, load_config
from goalzendo.experiment import (
    EXPERIMENT_BACKEND_VERSION,
    materialize_banks,
    render_experiment,
    render_prompt_view,
)
from goalzendo.runner import RunSpec, build_plan

from .freeze import (
    CONFIG_SPECS,
    EXPECTED_INFORMATIVE_VIEWS,
    FROZEN_GOALZENDO_IMPLEMENTATION_FINGERPRINT,
    FROZEN_MODEL_DEPENDENCIES,
    MODEL_RUNTIME_IDENTITIES,
    OPERATIONAL_FAILURE_TRIGGERS,
    RUN_BINDING_SCHEMA,
    RUN_BINDING_SCHEMA_VERSION,
    TOKENIZER_RUNTIME_IDENTITY,
    WALL_CEILING_SECONDS,
    WORKER_COUNT,
    VerifiedFreeze,
    atomic_json,
    semantic_digest,
    sha256_file,
    strict_json,
    verify_attempt_ledger,
    verify_launch_receipt,
    verify_model_snapshot_receipt,
)

ASSESSMENT_SCHEMA = "goalzendo.g00f_assessment"
ASSESSMENT_SCHEMA_VERSION = 1
GATE_SCHEMA = "goalzendo.g00f_gate"
GATE_SCHEMA_VERSION = 1
WORKER_RESULT_SCHEMA = "goalzendo.g00f_worker_result_receipt"
WORKER_RESULT_SCHEMA_VERSION = 1

FINAL_STEP = 1_000
FINAL_PREDICTIONS_PER_RUN = 512
MIRROR_PAIRS_PER_RUN = 256
MINIMUM_ADAPTER_NUMERATOR = 95
MINIMUM_ADAPTER_DENOMINATOR = 100
MAXIMUM_ORDER_ERROR_NUMERATOR = 2
MAXIMUM_ORDER_ERROR_DENOMINATOR = 100
MAXIMUM_ORDER_ERRORS = 5
MAXIMUM_PROBABILITY_SCORE_DISAGREEMENT = 0.005

ADAPTER_TARGET = {
    "law_only": "choice_y",
    "audit_law_matched": "choice_y",
    "sage_only": "choice_q",
    "herald_only": "choice_p",
}


class EvaluationError(RuntimeError):
    """Raised internally and converted to a persisted false gate."""


def exact_central_binomial_interval(
    trials: int,
    *,
    probability: float = 0.5,
    alpha: float = 0.05,
) -> tuple[int, int]:
    """Return the inclusive exact equal-tail interval using integer mass."""

    if type(trials) is not int or trials < 1:
        raise EvaluationError("binomial trials must be a positive integer")
    if probability != 0.5 or alpha != 0.05:
        raise EvaluationError("G00-F chance intervals are frozen to Binomial(n,.5), alpha=.05")
    denominator = 1 << trials
    coefficient = 1
    cumulative = 0
    lower: int | None = None
    for successes in range(trials + 1):
        cumulative += coefficient
        if lower is None and 40 * cumulative >= denominator:
            lower = successes
        if 40 * cumulative >= 39 * denominator:
            if lower is None:  # pragma: no cover
                raise AssertionError("upper binomial boundary preceded lower boundary")
            return lower, successes
        if successes < trials:
            coefficient = coefficient * (trials - successes) // (successes + 1)
    raise AssertionError("exact binomial cumulative mass did not reach one")


def _choice(value: Any, label: str) -> int:
    if isinstance(value, bool) or value not in (0, 1):
        raise EvaluationError(f"{label} must be the integer 0 or 1")
    return int(value)


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise EvaluationError(f"{label} must be finite numeric data")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise EvaluationError(f"{label} must be finite numeric data") from error
    if not math.isfinite(result):
        raise EvaluationError(f"{label} must be finite numeric data")
    return result


def _sigmoid_from_scores(score_a: float, score_b: float) -> float:
    difference = score_b - score_a
    if difference >= 0:
        tail = math.exp(-difference)
        return 1.0 / (1.0 + tail)
    head = math.exp(difference)
    return head / (1.0 + head)


def _specs(verified: VerifiedFreeze) -> dict[str, tuple[RunSpec, ...]]:
    result: dict[str, tuple[RunSpec, ...]] = {}
    for panel_id, static in CONFIG_SPECS.items():
        config = load_config(verified.repo / str(static["path"]))
        result[panel_id] = build_plan(config)
    return result


def _expected_paths(
    verified: VerifiedFreeze,
    roots: Mapping[str, str | Path],
) -> tuple[dict[str, tuple[str, RunSpec, Path]], list[dict[str, Any]]]:
    if set(roots) != set(CONFIG_SPECS):
        raise EvaluationError("evaluator requires exactly one artifact root for each model panel")
    expected: dict[str, tuple[str, RunSpec, Path]] = {}
    rows: list[dict[str, Any]] = []
    frozen_rows = {str(row["plan_key"]): row for row in verified.all_rows}
    for panel_id, panel_specs in _specs(verified).items():
        root = Path(roots[panel_id]).resolve()
        frozen_panel_roots = {
            Path(str(get_path(spec.config, "run.output_root"))).resolve() for spec in panel_specs
        }
        if len(frozen_panel_roots) != 1 or root != next(iter(frozen_panel_roots)):
            raise EvaluationError("caller artifact root differs from the frozen panel output root")
        for spec in panel_specs:
            store = RunStore(root, spec.config, spec.seed, verified.repo)
            frozen = frozen_rows[spec.plan_key]
            if (
                store.run_id != frozen["run_id"]
                or store.path.resolve() != Path(str(frozen["artifact_path"])).resolve()
            ):
                raise EvaluationError("evaluator RunStore identity/path differs from the frozen plan")
            expected[store.run_id] = (panel_id, spec, store.path)
            status_path = store.path / "status.json"
            status = read_json(status_path) if status_path.is_file() else {}
            marker = (store.path / "COMPLETE").is_file()
            recorded = str(status.get("state", "missing"))
            if marker and recorded == "complete":
                state = "complete"
            elif marker or recorded == "complete":
                state = "corrupt_completion_state"
            elif store.path.exists():
                state = recorded
            else:
                state = "missing_not_attempted"
            rows.append(
                {
                    "panel_id": panel_id,
                    "plan_key": spec.plan_key,
                    "run_id": store.run_id,
                    "seed": spec.seed,
                    "law_family": str(get_path(spec.config, "data.rule_family")),
                    "training_view": str(get_path(spec.config, "data.training_view")),
                    "worker_index": frozen["worker_index"],
                    "worker_order": frozen["worker_order"],
                    "state": state,
                    "error_type": status.get("error_type"),
                    "attempt_count": status.get("attempt", 0),
                    "intention_to_train": True,
                }
            )

        observed_paths = discover_runs(root, completed_only=False) if root.exists() else []
        expected_ids = {
            run_id
            for run_id, (expected_panel, _spec, _path) in expected.items()
            if expected_panel == panel_id
        }
        unexpected = sorted(path.name for path in observed_paths if path.name not in expected_ids)
        if unexpected:
            rows.append(
                {
                    "panel_id": panel_id,
                    "plan_key": None,
                    "run_id": None,
                    "seed": None,
                    "law_family": None,
                    "training_view": None,
                    "worker_index": None,
                    "worker_order": None,
                    "state": "unexpected_attempts",
                    "unexpected_run_ids": unexpected,
                    "intention_to_train": False,
                }
            )
        panel_run_paths = {
            path.resolve() for expected_panel, _spec, path in expected.values() if expected_panel == panel_id
        }
        if root.exists():
            for candidate in root.rglob("*"):
                resolved_candidate = candidate.resolve()
                inside_run = any(
                    run_path == resolved_candidate or run_path in resolved_candidate.parents
                    for run_path in panel_run_paths
                )
                ancestor_of_run = any(resolved_candidate in run_path.parents for run_path in panel_run_paths)
                if candidate.is_symlink() or (not inside_run and not ancestor_of_run):
                    raise EvaluationError(
                        f"artifact root contains a stray status/attempt/file path: {candidate}"
                    )
    expected_paths = [path.resolve() for _panel, _spec, path in expected.values()]
    if (
        len(expected) != 160
        or len({spec.plan_key for _panel, spec, _path in expected.values()}) != 160
        or len(set(expected_paths)) != 160
        or len(set(expected)) != 160
    ):
        raise EvaluationError("G00-F expected inventory lacks 160 unique run IDs/paths/keys")
    return expected, sorted(
        rows,
        key=lambda row: (
            str(row["panel_id"]),
            int(row["worker_index"]) if row["worker_index"] is not None else 999,
            int(row["worker_order"]) if row["worker_order"] is not None else 999,
            str(row["run_id"]),
        ),
    )


def _verify_worker_result_receipts(
    verified: VerifiedFreeze,
    paths: Mapping[int, str | Path],
    *,
    expected_provision_receipt_sha256: str,
    expected_pod_id: str,
) -> dict[str, Any]:
    if set(paths) != set(range(WORKER_COUNT)):
        raise EvaluationError("exactly four worker result receipts are required")
    rows: list[dict[str, Any]] = []
    total_seconds = 0.0
    execution_uuids: set[str] = set()
    gpu_uuids: set[str] = set()
    execution_roots: set[Path] = set()
    installed_distribution_digests: set[str] = set()
    accelerator_digests: set[str] = set()
    budget_start_file_sha256s: set[str] = set()
    budget_started_monotonic_values: set[int] = set()
    worker_completed_monotonic_values: list[int] = []
    worker_started_monotonic_values: list[int] = []
    for worker_index in range(WORKER_COUNT):
        path = Path(paths[worker_index]).resolve()
        execution_roots.add(path.parent)
        receipt = strict_json(path, f"worker {worker_index} result receipt")
        body = {key: value for key, value in receipt.items() if key != "receipt_digest"}
        wall_seconds = _finite(receipt.get("wall_seconds"), "worker wall_seconds")
        completed_monotonic_ns = receipt.get("completed_monotonic_ns")
        if (
            set(receipt)
            != {
                "completed_monotonic_ns",
                "completed_runs",
                "error_type",
                "execution_uuid",
                "exit_code",
                "failed_runs",
                "freeze_digest",
                "freeze_file_sha256",
                "g01_launch_authorized",
                "launch_receipt",
                "no_reassignment",
                "observed_outcome_rows",
                "outcome_metrics_read",
                "planned_runs",
                "predictions_read",
                "receipt_digest",
                "schema",
                "schema_version",
                "state",
                "wall_seconds",
                "worker_index",
            }
            or receipt.get("schema") != WORKER_RESULT_SCHEMA
            or receipt.get("schema_version") != WORKER_RESULT_SCHEMA_VERSION
            or receipt.get("worker_index") != worker_index
            or receipt.get("freeze_file_sha256") != verified.file_sha256
            or receipt.get("freeze_digest") != verified.digest
            or receipt.get("receipt_digest") != semantic_digest(body)
            or receipt.get("state") != "complete"
            or receipt.get("exit_code") != 0
            or receipt.get("planned_runs") != 40
            or receipt.get("completed_runs") != 40
            or receipt.get("failed_runs") != 0
            or receipt.get("observed_outcome_rows") != 40
            or receipt.get("no_reassignment") is not True
            or receipt.get("error_type") is not None
            or receipt.get("outcome_metrics_read") is not False
            or receipt.get("predictions_read") is not False
            or receipt.get("g01_launch_authorized") is not False
            or isinstance(completed_monotonic_ns, bool)
            or not isinstance(completed_monotonic_ns, int)
            or wall_seconds < 0
            or wall_seconds > WALL_CEILING_SECONDS
        ):
            raise EvaluationError(f"worker {worker_index} result receipt failed closed")
        launch = receipt.get("launch_receipt")
        if not isinstance(launch, Mapping):
            raise EvaluationError("worker result receipt omits its launch binding")
        verified_launch = verify_launch_receipt(
            verified=verified,
            worker_index=worker_index,
            receipt_path=str(launch.get("path", "")),
            expected_provision_receipt_sha256=expected_provision_receipt_sha256,
            expected_pod_id=expected_pod_id,
        )
        if verified_launch["file_sha256"] != launch.get("file_sha256") or verified_launch[
            "receipt_digest"
        ] != launch.get("receipt_digest"):
            raise EvaluationError("worker result/launch receipt binding mismatch")
        if receipt.get("execution_uuid") != verified_launch["execution_uuid"]:
            raise EvaluationError("worker result execution UUID differs from its launch receipt")
        launch_started_monotonic_ns = int(verified_launch["started_monotonic_ns"])
        ledger_budget = verified_launch["ledger"]["budget_start"]
        budget_started_monotonic_ns = int(ledger_budget["started_monotonic_ns"])
        deadline_monotonic_ns = budget_started_monotonic_ns + WALL_CEILING_SECONDS * 1_000_000_000
        if (
            completed_monotonic_ns < launch_started_monotonic_ns
            or completed_monotonic_ns > deadline_monotonic_ns
            or not math.isclose(
                wall_seconds,
                (completed_monotonic_ns - launch_started_monotonic_ns) / 1_000_000_000,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            raise EvaluationError("worker result did not complete within the monotonic 14-hour budget")
        total_seconds += wall_seconds
        execution_uuids.add(str(verified_launch["execution_uuid"]))
        gpu_uuids.add(str(verified_launch["gpu_uuid"]))
        runtime_environment = verified_launch["runtime_environment"]
        installed_distribution_digests.add(str(runtime_environment["installed_distributions_digest"]))
        accelerator_digests.add(str(runtime_environment["accelerator_digest"]))
        budget_start_file_sha256s.add(str(verified_launch["ledger"]["budget_start_file_sha256"]))
        budget_started_monotonic_values.add(budget_started_monotonic_ns)
        worker_started_monotonic_values.append(launch_started_monotonic_ns)
        worker_completed_monotonic_values.append(completed_monotonic_ns)
        rows.append(
            {
                "worker_index": worker_index,
                "file_sha256": sha256_file(path),
                "receipt_digest": receipt["receipt_digest"],
                "wall_seconds": wall_seconds,
                "completed_monotonic_ns": completed_monotonic_ns,
                "launch_receipt": verified_launch,
            }
        )
    total_hours = total_seconds / 3_600
    ceiling = float(verified.payload["runtime"]["h100_hour_ceiling"])
    if (
        total_hours > ceiling
        or len(execution_uuids) != 1
        or len(gpu_uuids) != WORKER_COUNT
        or len(execution_roots) != 1
        or len(installed_distribution_digests) != 1
        or len(accelerator_digests) != 1
        or len(budget_start_file_sha256s) != 1
        or len(budget_started_monotonic_values) != 1
    ):
        raise EvaluationError("worker runtime exceeded its ceiling or did not use one 4-GPU execution")
    execution_root = next(iter(execution_roots))
    expected_root_names = {
        "EXECUTION_UUID",
        "frozen-source",
        "itt-ledger",
        "model-receipt-0p5b.json",
        "model-receipt-1p5b.json",
        "model-integration-audit-0p5b.json",
        "model-integration-audit-1p5b.json",
        "runpod-api-response.json",
        "runpod-create-response.json",
        "runpod-provision-receipt.json",
        "source-bundle-receipt.json",
        "watchdog-normal-stop.json",
        "watchdog-started.json",
        "worker-pids.txt",
        *(f"worker-{index}-launch.json" for index in range(WORKER_COUNT)),
        *(f"worker-{index}-result.json" for index in range(WORKER_COUNT)),
        *(f"worker-{index}.log" for index in range(WORKER_COUNT)),
    }
    root_entries = set(execution_root.iterdir())
    if (
        {path.name for path in root_entries} != expected_root_names
        or any(path.is_symlink() for path in root_entries)
        or not (execution_root / "frozen-source").is_dir()
        or not (execution_root / "itt-ledger").is_dir()
        or any(
            not path.is_file() for path in root_entries if path.name not in {"frozen-source", "itt-ledger"}
        )
    ):
        raise EvaluationError("successful G00-F execution root inventory contains an extra/missing entry")
    execution_uuid = next(iter(execution_uuids))
    if (execution_root / "EXECUTION_UUID").read_text(encoding="ascii") != f"{execution_uuid}\n":
        raise EvaluationError("execution-root UUID marker differs from worker receipts")
    budget_started_monotonic_ns = next(iter(budget_started_monotonic_values))
    deadline_monotonic_ns = budget_started_monotonic_ns + WALL_CEILING_SECONDS * 1_000_000_000
    watchdog_started = strict_json(
        execution_root / "watchdog-started.json",
        "G00-F watchdog start receipt",
    )
    watchdog_stopped = strict_json(
        execution_root / "watchdog-normal-stop.json",
        "G00-F watchdog normal-stop receipt",
    )
    watchdog_started_body = {key: value for key, value in watchdog_started.items() if key != "receipt_digest"}
    watchdog_stopped_body = {key: value for key, value in watchdog_stopped.items() if key != "receipt_digest"}
    common_watchdog_keys = {
        "budget_start_file_sha256",
        "deadline_monotonic_ns",
        "execution_uuid",
        "freeze_digest",
        "freeze_file_sha256",
        "g01_launch_authorized",
        "outcome_metrics_read",
        "predictions_read",
        "receipt_digest",
        "schema",
        "schema_version",
        "watchdog_pid",
    }
    watchdog_started_ns = watchdog_started.get("started_monotonic_ns")
    watchdog_stopped_ns = watchdog_stopped.get("stopped_monotonic_ns")
    if (
        set(watchdog_started) != common_watchdog_keys | {"started_monotonic_ns"}
        or set(watchdog_stopped) != common_watchdog_keys | {"stopped_monotonic_ns"}
        or watchdog_started.get("schema") != "goalzendo.g00f_watchdog_started"
        or watchdog_stopped.get("schema") != "goalzendo.g00f_watchdog_normal_stop"
        or watchdog_started.get("schema_version") != 1
        or watchdog_stopped.get("schema_version") != 1
        or watchdog_started.get("execution_uuid") != execution_uuid
        or watchdog_stopped.get("execution_uuid") != execution_uuid
        or watchdog_started.get("freeze_file_sha256") != verified.file_sha256
        or watchdog_stopped.get("freeze_file_sha256") != verified.file_sha256
        or watchdog_started.get("freeze_digest") != verified.digest
        or watchdog_stopped.get("freeze_digest") != verified.digest
        or watchdog_started.get("budget_start_file_sha256") != next(iter(budget_start_file_sha256s))
        or watchdog_stopped.get("budget_start_file_sha256") != next(iter(budget_start_file_sha256s))
        or watchdog_started.get("deadline_monotonic_ns") != deadline_monotonic_ns
        or watchdog_stopped.get("deadline_monotonic_ns") != deadline_monotonic_ns
        or watchdog_started.get("watchdog_pid") != watchdog_stopped.get("watchdog_pid")
        or isinstance(watchdog_started.get("watchdog_pid"), bool)
        or not isinstance(watchdog_started.get("watchdog_pid"), int)
        or int(watchdog_started["watchdog_pid"]) <= 1
        or isinstance(watchdog_started_ns, bool)
        or not isinstance(watchdog_started_ns, int)
        or isinstance(watchdog_stopped_ns, bool)
        or not isinstance(watchdog_stopped_ns, int)
        or watchdog_started_ns < budget_started_monotonic_ns
        or watchdog_started_ns > min(worker_started_monotonic_values)
        or watchdog_stopped_ns < max(worker_completed_monotonic_values)
        or watchdog_stopped_ns > deadline_monotonic_ns
        or watchdog_started.get("outcome_metrics_read") is not False
        or watchdog_stopped.get("outcome_metrics_read") is not False
        or watchdog_started.get("predictions_read") is not False
        or watchdog_stopped.get("predictions_read") is not False
        or watchdog_started.get("g01_launch_authorized") is not False
        or watchdog_stopped.get("g01_launch_authorized") is not False
        or watchdog_started.get("receipt_digest") != semantic_digest(watchdog_started_body)
        or watchdog_stopped.get("receipt_digest") != semantic_digest(watchdog_stopped_body)
    ):
        raise EvaluationError("watchdog lifecycle does not prove continuous 14-hour supervision")
    return {
        "workers": rows,
        "execution_root": str(execution_root),
        "execution_uuid": execution_uuid,
        "installed_distributions_digest": next(iter(installed_distribution_digests)),
        "accelerator_digest": next(iter(accelerator_digests)),
        "watchdog": {
            "started_file_sha256": sha256_file(execution_root / "watchdog-started.json"),
            "normal_stop_file_sha256": sha256_file(execution_root / "watchdog-normal-stop.json"),
            "deadline_monotonic_ns": deadline_monotonic_ns,
            "stopped_monotonic_ns": watchdog_stopped_ns,
        },
        "total_h100_hours": total_hours,
        "h100_hour_ceiling": ceiling,
        "passed": True,
    }


def _require_same_execution(
    attempt_audit: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> None:
    """Prevent independently valid ledger/result trees from being mixed."""

    ledger = attempt_audit.get("ledger")
    if not isinstance(ledger, Mapping):
        raise EvaluationError("attempt audit omits its verified ledger binding")
    execution_uuid = runtime.get("execution_uuid")
    execution_root = Path(str(runtime.get("execution_root", ""))).resolve()
    expected_ledger_root = execution_root / "itt-ledger" / str(execution_uuid)
    if (
        not isinstance(execution_uuid, str)
        or not execution_uuid
        or ledger.get("execution_uuid") != execution_uuid
        or Path(str(ledger.get("ledger_root", ""))).resolve() != expected_ledger_root
    ):
        raise EvaluationError("attempt ledger and worker evidence come from different executions")


def _verify_attempt_receipt(
    *,
    verified: VerifiedFreeze,
    expected_row: Mapping[str, Any],
    path: Path,
    kind: str,
) -> dict[str, Any]:
    payload = strict_json(path, f"G00-F {kind} attempt receipt")
    body = {key: value for key, value in payload.items() if key != "receipt_digest"}
    if (
        payload.get("schema") != "goalzendo.g00f_attempt_receipt"
        or payload.get("schema_version") != 1
        or payload.get("kind") != kind
        or payload.get("freeze_file_sha256") != verified.file_sha256
        or payload.get("freeze_digest") != verified.digest
        or payload.get("panel_id") != expected_row["panel_id"]
        or payload.get("plan_key") != expected_row["plan_key"]
        or payload.get("run_id") != expected_row["run_id"]
        or payload.get("seed") != expected_row["seed"]
        or payload.get("worker_index") != expected_row["worker_index"]
        or payload.get("worker_order") != expected_row["worker_order"]
        or payload.get("outcome_metrics_read") is not False
        or payload.get("predictions_read") is not False
        or payload.get("g01_launch_authorized") is not False
        or payload.get("receipt_digest") != semantic_digest(body)
    ):
        raise EvaluationError(f"G00-F {kind} attempt receipt changed")
    return payload


def _audit_attempt_ledger(
    *,
    verified: VerifiedFreeze,
    ledger_root: str | Path,
) -> dict[str, Any]:
    ledger = verify_attempt_ledger(verified=verified, ledger_root=ledger_root)
    root = Path(str(ledger["ledger_root"]))
    execution_uuid = str(ledger["execution_uuid"])
    states: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    all_complete = True
    for frozen in verified.all_rows:
        plan_key = str(frozen["plan_key"])
        start_path = root / "starts" / f"{plan_key}.json"
        terminal_path = root / "terminals" / f"{plan_key}.json"
        lease_path = root / "leases" / f"{plan_key}.json"
        if not start_path.is_file() and not terminal_path.is_file():
            state = "preallocated_not_started"
            start = None
            terminal = None
        elif start_path.is_file() and not terminal_path.is_file():
            start = _verify_attempt_receipt(
                verified=verified,
                expected_row=frozen,
                path=start_path,
                kind="start",
            )
            terminal = None
            state = "started_without_terminal"
        elif not start_path.is_file() and terminal_path.is_file():
            start = None
            terminal = _verify_attempt_receipt(
                verified=verified,
                expected_row=frozen,
                path=terminal_path,
                kind="terminal",
            )
            state = str(terminal.get("state"))
            if state not in {"failed_before_start_receipt", "not_started_after_failure"}:
                raise EvaluationError("terminal-without-start has an unauthorized state")
        else:
            start = _verify_attempt_receipt(
                verified=verified,
                expected_row=frozen,
                path=start_path,
                kind="start",
            )
            terminal = _verify_attempt_receipt(
                verified=verified,
                expected_row=frozen,
                path=terminal_path,
                kind="terminal",
            )
            if start.get("state") != "started":
                raise EvaluationError("G00-F start receipt state changed")
            state = str(terminal.get("state"))
        for receipt in (start, terminal):
            if receipt is not None and receipt.get("execution_uuid") != execution_uuid:
                raise EvaluationError("attempt receipt execution UUID changed")
        if start is not None or state == "failed_before_start_receipt":
            lease = strict_json(lease_path, "G00-F run ownership lease")
            lease_body = {key: value for key, value in lease.items() if key != "lease_digest"}
            if (
                lease.get("schema") != "goalzendo.g00f_run_ownership_lease"
                or lease.get("schema_version") != 1
                or lease.get("execution_uuid") != execution_uuid
                or lease.get("freeze_file_sha256") != verified.file_sha256
                or lease.get("freeze_digest") != verified.digest
                or lease.get("plan_key") != plan_key
                or lease.get("run_id") != frozen["run_id"]
                or lease.get("worker_index") != frozen["worker_index"]
                or lease.get("worker_order") != frozen["worker_order"]
                or lease.get("g01_launch_authorized") is not False
                or lease.get("lease_digest") != semantic_digest(lease_body)
            ):
                raise EvaluationError("G00-F run ownership lease changed")
        elif lease_path.exists():
            raise EvaluationError("unstarted G00-F key has an unauthorized ownership lease")
        seals_for_row: dict[str, Any] | None = None
        if state == "complete":
            if start is None or terminal is None:
                raise EvaluationError("complete attempt lacks start and terminal receipts")
            seals = terminal.get("outcome_file_seals")
            if not isinstance(seals, Mapping) or set(seals) != {
                "metrics.jsonl",
                "predictions.jsonl",
                "summary.json",
            }:
                raise EvaluationError("complete attempt outcome seal inventory changed")
            for name, seal in seals.items():
                outcome_file = Path(str(frozen["artifact_path"])) / str(name)
                if (
                    not isinstance(seal, Mapping)
                    or set(seal) != {"bytes", "path", "sealed_mode", "sha256"}
                    or seal.get("path") != str(outcome_file)
                    or seal.get("sealed_mode") != 0
                    or not outcome_file.is_file()
                    or outcome_file.is_symlink()
                    or outcome_file.stat().st_size != seal.get("bytes")
                    or sha256_file(outcome_file) != seal.get("sha256")
                    or stat.S_IMODE(outcome_file.stat().st_mode) != 0o400
                ):
                    raise EvaluationError("complete attempt outcome-file seal does not replay")
            if (
                start.get("error_type") is not None
                or start.get("caused_by_plan_key") is not None
                or not isinstance(start.get("launch_receipt"), Mapping)
                or terminal.get("error_type") is not None
                or terminal.get("caused_by_plan_key") is not None
                or not isinstance(terminal.get("launch_receipt"), Mapping)
            ):
                raise EvaluationError("complete attempt contains failure or missing launch fields")
            seals_for_row = copy.deepcopy(dict(seals))
        else:
            all_complete = False
        states[plan_key] = state
        rows.append(
            {
                "plan_key": plan_key,
                "state": state,
                "start_file_sha256": sha256_file(start_path) if start_path.is_file() else None,
                "terminal_file_sha256": sha256_file(terminal_path) if terminal_path.is_file() else None,
                "terminal_error_type": terminal.get("error_type") if terminal else None,
                "lease_file_sha256": sha256_file(lease_path) if lease_path.is_file() else None,
                "outcome_file_seals": seals_for_row,
            }
        )
    stop_path = root / "global-stop.json"
    global_stop = strict_json(stop_path, "G00-F global stop") if stop_path.is_file() else None
    if global_stop is not None:
        all_complete = False
        stop_body = {key: value for key, value in global_stop.items() if key != "stop_digest"}
        if (
            global_stop.get("schema") != "goalzendo.g00f_global_execution_stop"
            or global_stop.get("schema_version") != 1
            or global_stop.get("freeze_file_sha256") != verified.file_sha256
            or global_stop.get("freeze_digest") != verified.digest
            or global_stop.get("execution_uuid") != execution_uuid
            or global_stop.get("trigger") not in OPERATIONAL_FAILURE_TRIGGERS
            or global_stop.get("outcome_metrics_read") is not False
            or global_stop.get("predictions_read") is not False
            or global_stop.get("g01_launch_authorized") is not False
            or global_stop.get("stop_digest") != semantic_digest(stop_body)
        ):
            raise EvaluationError("G00-F global stop receipt is malformed or outcome-dependent")
    timeout_path = root / "budget-timeout.json"
    if timeout_path.exists() and global_stop is None:
        raise EvaluationError("G00-F budget timeout exists without a bound global stop")
    unseal_path = root / "panel-unseal.json"
    unseal = strict_json(unseal_path, "G00-F panel unseal receipt") if unseal_path.is_file() else None
    if all_complete:
        if unseal is None:
            raise EvaluationError("all-complete G00-F panel lacks its atomic unseal receipt")
        unseal_body = {key: value for key, value in unseal.items() if key != "unseal_digest"}
        if (
            unseal.get("schema") != "goalzendo.g00f_panel_unseal_receipt"
            or unseal.get("schema_version") != 1
            or unseal.get("freeze_file_sha256") != verified.file_sha256
            or unseal.get("freeze_digest") != verified.digest
            or unseal.get("terminal_complete_count") != 160
            or unseal.get("outcomes_inspected_before_unseal") is not False
            or unseal.get("g01_launch_authorized") is not False
            or unseal.get("unseal_digest") != semantic_digest(unseal_body)
            or not isinstance(unseal.get("outcome_files"), Sequence)
            or len(unseal["outcome_files"]) != 480
        ):
            raise EvaluationError("G00-F panel unseal receipt changed")
        expected_unseal_rows = []
        for row in verified.all_rows:
            terminal = strict_json(
                root / "terminals" / f"{row['plan_key']}.json",
                "terminal receipt for unseal inventory",
            )
            for name in ("metrics.jsonl", "predictions.jsonl", "summary.json"):
                expected_unseal_rows.append(
                    {
                        "plan_key": row["plan_key"],
                        "run_id": row["run_id"],
                        "name": name,
                        "bytes": terminal["outcome_file_seals"][name]["bytes"],
                        "sha256": terminal["outcome_file_seals"][name]["sha256"],
                        "unsealed_mode": 0o400,
                    }
                )
        if unseal.get("outcome_files") != expected_unseal_rows:
            raise EvaluationError("G00-F panel unseal inventory differs from all 160 terminals")
    elif unseal is not None:
        raise EvaluationError("incomplete or failed G00-F panel was improperly unsealed")
    return {
        "ledger": ledger,
        "rows": rows,
        "states": states,
        "all_complete": all_complete,
        "global_stop": global_stop,
        "panel_unseal": (
            {
                "path": str(unseal_path),
                "file_sha256": sha256_file(unseal_path),
                "unseal_digest": unseal["unseal_digest"],
            }
            if unseal is not None
            else None
        ),
    }


def _verify_run_binding(
    *,
    verified: VerifiedFreeze,
    row: Mapping[str, Any],
    run_path: Path,
    receipt_cache: dict[tuple[str, str], Mapping[str, Any]],
    expected_provision_receipt_sha256: str | None = None,
    expected_pod_id: str | None = None,
) -> dict[str, Any]:
    target = run_path / "g00f-freeze-binding.json"
    payload = strict_json(target, "per-run G00-F freeze binding")
    body = {key: value for key, value in payload.items() if key != "binding_digest"}
    if (
        payload.get("schema") != RUN_BINDING_SCHEMA
        or payload.get("schema_version") != RUN_BINDING_SCHEMA_VERSION
        or payload.get("freeze_file_sha256") != verified.file_sha256
        or payload.get("freeze_digest") != verified.digest
        or payload.get("plan_key") != row["plan_key"]
        or payload.get("run_id") != row["run_id"]
        or payload.get("panel_id") != row["panel_id"]
        or payload.get("worker_index") != row["worker_index"]
        or payload.get("worker_order") != row["worker_order"]
        or not isinstance(payload.get("execution_uuid"), str)
        or not payload.get("execution_uuid")
        or payload.get("g01_launch_authorized") is not False
        or payload.get("binding_digest") != semantic_digest(body)
    ):
        raise EvaluationError("per-run G00-F freeze binding changed")
    launch = payload.get("launch_receipt")
    if not isinstance(launch, Mapping):
        raise EvaluationError("per-run G00-F binding omits the launch receipt")
    launch_key = ("launch", str(launch.get("path", "")))
    if launch_key not in receipt_cache:
        receipt_cache[launch_key] = verify_launch_receipt(
            verified=verified,
            worker_index=int(row["worker_index"]),
            receipt_path=str(launch.get("path", "")),
            expected_provision_receipt_sha256=expected_provision_receipt_sha256,
            expected_pod_id=expected_pod_id,
        )
    if any(receipt_cache[launch_key].get(key) != launch.get(key) for key in launch):
        raise EvaluationError("per-run launch receipt summary does not replay")
    if payload.get("execution_uuid") != receipt_cache[launch_key].get("execution_uuid"):
        raise EvaluationError("per-run execution UUID differs from its launch receipt")
    models = payload.get("model_receipts")
    if not isinstance(models, Mapping) or set(models) != set(CONFIG_SPECS):
        raise EvaluationError("per-run G00-F binding omits model receipts")
    for panel_id, raw in models.items():
        if not isinstance(raw, Mapping):
            raise EvaluationError("per-run model receipt summary is malformed")
        model_key = (str(panel_id), str(raw.get("path", "")))
        if model_key not in receipt_cache:
            receipt_cache[model_key] = verify_model_snapshot_receipt(
                verified=verified,
                panel_id=str(panel_id),
                receipt_path=str(raw.get("path", "")),
            )
        if any(receipt_cache[model_key].get(key) != raw.get(key) for key in raw):
            raise EvaluationError("per-run model receipt summary does not replay")
    attempt_start = payload.get("attempt_start")
    if not isinstance(attempt_start, Mapping):
        raise EvaluationError("per-run G00-F binding omits the exclusive attempt start")
    attempt_path = Path(str(attempt_start.get("path", "")))
    if (
        not attempt_path.is_file()
        or sha256_file(attempt_path) != attempt_start.get("file_sha256")
        or strict_json(attempt_path, "bound G00-F attempt start").get("receipt_digest")
        != attempt_start.get("receipt_digest")
    ):
        raise EvaluationError("per-run G00-F attempt-start binding does not replay")
    attempt_payload = strict_json(attempt_path, "bound G00-F attempt start")
    if (
        attempt_payload.get("execution_uuid") != payload.get("execution_uuid")
        or attempt_payload.get("plan_key") != row["plan_key"]
        or attempt_payload.get("worker_index") != row["worker_index"]
    ):
        raise EvaluationError("per-run G00-F attempt-start execution identity changed")
    lease = payload.get("ownership_lease")
    if not isinstance(lease, Mapping):
        raise EvaluationError("per-run G00-F binding omits its exclusive ownership lease")
    lease_path = Path(str(lease.get("path", "")))
    lease_payload = strict_json(lease_path, "bound G00-F ownership lease")
    if (
        sha256_file(lease_path) != lease.get("file_sha256")
        or lease_payload.get("lease_digest") != lease.get("lease_digest")
        or lease_payload.get("execution_uuid") != payload.get("execution_uuid")
        or lease_payload.get("plan_key") != row["plan_key"]
        or lease_payload.get("worker_index") != row["worker_index"]
    ):
        raise EvaluationError("per-run G00-F ownership-lease binding does not replay")
    return {
        "file_sha256": sha256_file(target),
        "binding_digest": payload["binding_digest"],
        "execution_uuid": payload["execution_uuid"],
        "launch_receipt": copy.deepcopy(dict(receipt_cache[launch_key])),
        "model_receipts": copy.deepcopy(dict(models)),
    }


def _resolved_config(path: Path) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise EvaluationError("run resolved config is unreadable") from error
    if not isinstance(value, Mapping):
        raise EvaluationError("run resolved config is not an object")
    return value


def _final_predictions(path: Path, spec: RunSpec) -> list[dict[str, Any]]:
    records = [
        row
        for row in read_jsonl(path / "predictions.jsonl")
        if row.get("kind") == "prediction"
        and row.get("step") == FINAL_STEP
        and row.get("split") == "final_factorial"
        and row.get("intervention_role") is None
    ]
    if len(records) != FINAL_PREDICTIONS_PER_RUN:
        raise EvaluationError("run lacks exactly 512 step-1000 final-factorial predictions")
    view = str(get_path(spec.config, "data.training_view"))
    if {str(row.get("prompt_view")) for row in records} != {view}:
        raise EvaluationError("final prediction prompt view differs from the registered case")
    sample_ids = [str(row.get("sample_id", "")) for row in records]
    record_ids = [str(row.get("record_id", "")) for row in records]
    if "" in sample_ids or "" in record_ids or len(set(sample_ids)) != 512 or len(set(record_ids)) != 512:
        raise EvaluationError("final predictions contain empty or duplicate sample/record identities")
    return records


def _regenerated_final_map(spec: RunSpec) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    banks = materialize_banks(spec.config, spec.seeds)
    decisions = banks.final_factorial.decisions
    if len(decisions) != FINAL_PREDICTIONS_PER_RUN:
        raise EvaluationError("canonical regenerated final bank is not 512 decisions")
    by_sample = {decision.sample_id: decision for decision in decisions}
    if len(by_sample) != len(decisions):
        raise EvaluationError("canonical regenerated final bank duplicates sample IDs")
    by_pair: dict[str, dict[str, Any]] = defaultdict(dict)
    for decision in decisions:
        if decision.mirror_pair_id is None or decision.mirror_role is None:
            raise EvaluationError("canonical final decision lacks mirror metadata")
        by_pair[decision.mirror_pair_id][decision.mirror_role] = decision
    if len(by_pair) != MIRROR_PAIRS_PER_RUN:
        raise EvaluationError("canonical final bank is not exactly 256 mirror pairs")
    for pair_id, roles in by_pair.items():
        if set(roles) != {"base", "mirror"}:
            raise EvaluationError(f"mirror pair {pair_id} lacks exact base/mirror roles")
        base = roles["base"]
        mirror = roles["mirror"]
        if (
            mirror.koans != (base.koans[1], base.koans[0])
            or tuple(int(value) for value in mirror.candidate_tuple)
            != tuple(1 - int(value) for value in base.candidate_tuple)
            or mirror.law != base.law
            or mirror.sage_rule != base.sage_rule
            or mirror.intervention != base.intervention
            or mirror.split != base.split
        ):
            raise EvaluationError(f"mirror pair {pair_id} is not only a complete A/B block swap")
        if str(get_path(spec.config, "data.training_view")) == "no_signal":
            renderers = tuple(str(value) for value in get_path(spec.config, "data.heldout_renderers"))
            for renderer_id in renderers:
                base_prompt = render_prompt_view(
                    base,
                    banks.final_factorial.feature_names,
                    renderer_id=renderer_id,
                    prompt_view="no_signal",
                ).encode("utf-8")
                mirror_prompt = render_prompt_view(
                    mirror,
                    banks.final_factorial.feature_names,
                    renderer_id=renderer_id,
                    prompt_view="no_signal",
                ).encode("utf-8")
                if base_prompt != mirror_prompt:
                    raise EvaluationError(
                        f"no_signal mirror pair {pair_id} is not byte-identical for {renderer_id}"
                    )
    return by_sample, by_pair


def _regenerated_dataset_metadata(
    spec: RunSpec,
    snapshot_root: Path,
    tokenizer_cache: dict[str, Any],
) -> Mapping[str, Any]:
    """Replay symbolic and rendered dataset identities from frozen inputs."""

    cache_key = str(snapshot_root.resolve())
    if cache_key not in tokenizer_cache:
        try:
            from transformers import AutoTokenizer  # type: ignore[import-not-found]
        except ImportError as error:  # pragma: no cover - frozen GPU runtime dependency
            raise EvaluationError("frozen transformers dependency is absent") from error
        tokenizer_cache[cache_key] = AutoTokenizer.from_pretrained(
            cache_key,
            local_files_only=True,
            trust_remote_code=False,
        )
    banks = materialize_banks(spec.config, spec.seeds)
    rendered = render_experiment(
        spec.config,
        banks,
        tokenizer_cache[cache_key],
        int(spec.seeds["rendering"]),
    )
    symbolic = dict(banks.metadata)
    rendering = dict(rendered.metadata)
    return {
        **symbolic,
        "rendering": rendering,
        "dataset_binding_digest": stable_hash(
            {"symbolic": symbolic, "rendering": rendering},
            64,
        ),
    }


def _validate_prediction_rows(
    records: Sequence[Mapping[str, Any]],
    by_sample: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], float]:
    observed = {str(record["sample_id"]): record for record in records}
    if set(observed) != set(by_sample):
        raise EvaluationError("prediction sample IDs do not exactly equal the regenerated bank")
    maximum_probability_difference = 0.0
    for sample_id, decision in by_sample.items():
        row = observed[sample_id]
        expected_choices = tuple(int(value) for value in decision.candidate_tuple)
        observed_choices = tuple(
            _choice(row.get(key), f"{sample_id}.{key}") for key in ("choice_y", "choice_p", "choice_q")
        )
        if observed_choices != expected_choices or row.get("factorial_cell") != list(expected_choices):
            raise EvaluationError("prediction truth tuple differs from the canonical regenerated bank")
        predicted = _choice(row.get("predicted_action"), "predicted_action")
        score_a = _finite(row.get("score_a"), "score_a")
        score_b = _finite(row.get("score_b"), "score_b")
        probability_b = _finite(row.get("probability_b"), "probability_b")
        margin = _finite(row.get("margin_b_minus_a"), "margin_b_minus_a")
        if not 0.0 <= probability_b <= 1.0:
            raise EvaluationError("saved action probability lies outside [0,1]")
        if predicted != (1 if score_b > score_a else 0):
            raise EvaluationError("saved greedy action does not equal score argmax with A tie-break")
        if not math.isclose(margin, score_b - score_a, rel_tol=0.0, abs_tol=1e-12):
            raise EvaluationError("saved action margin does not equal score difference")
        expected_probability = _sigmoid_from_scores(score_a, score_b)
        maximum_probability_difference = max(
            maximum_probability_difference,
            abs(probability_b - expected_probability),
        )
    if maximum_probability_difference > MAXIMUM_PROBABILITY_SCORE_DISAGREEMENT:
        raise EvaluationError("saved BF16 probabilities disagree with the two constrained scores")
    return observed, maximum_probability_difference


def _evaluate_run(
    *,
    verified: VerifiedFreeze,
    panel_id: str,
    spec: RunSpec,
    path: Path,
    row: Mapping[str, Any],
    outcome_file_seals: Mapping[str, Any],
    regenerated: Callable[[RunSpec], tuple[Mapping[str, Any], Mapping[str, Any]]],
    receipt_cache: dict[tuple[str, str], Mapping[str, Any]],
    regenerate_dataset: Callable[[RunSpec, Path, dict[str, Any]], Mapping[str, Any]] = (
        _regenerated_dataset_metadata
    ),
    tokenizer_cache: dict[str, Any] | None = None,
    expected_provision_receipt_sha256: str | None = None,
    expected_pod_id: str | None = None,
) -> dict[str, Any]:
    if path.resolve() != Path(str(row["artifact_path"])).resolve():
        raise EvaluationError("evaluated run path differs from the frozen artifact path")
    if set(outcome_file_seals) != {"metrics.jsonl", "predictions.jsonl", "summary.json"}:
        raise EvaluationError("evaluated run lacks the exact terminal outcome seal inventory")
    for name in ("metrics.jsonl", "predictions.jsonl", "summary.json"):
        outcome_path = path / name
        seal = outcome_file_seals[name]
        if (
            not isinstance(seal, Mapping)
            or set(seal) != {"bytes", "path", "sealed_mode", "sha256"}
            or seal.get("path") != str(outcome_path)
            or seal.get("sealed_mode") != 0
            or outcome_path.is_symlink()
            or not outcome_path.is_file()
            or stat.S_IMODE(outcome_path.stat().st_mode) != 0o400
            or outcome_path.stat().st_size != seal.get("bytes")
            or sha256_file(outcome_path) != seal.get("sha256")
        ):
            raise EvaluationError(f"evaluated {name} bytes differ from the terminal receipt seal")
    completion = verify_completion_attestation(path)
    identity = read_json(path / "identity.json")
    summary = read_json(path / "summary.json")
    status = read_json(path / "status.json")
    completion_files = {
        "environment.json",
        "identity.json",
        "implementation.json",
        "manifests/dataset.json",
        "manifests/model.json",
        "manifests/tokenizer.json",
        "metrics.jsonl",
        "predictions.jsonl",
        "resolved_config.yaml",
        "status.json",
        "summary.json",
    }
    if (
        set(completion)
        != {
            "artifact_schema_version",
            "completion_digest",
            "files",
            "run_id",
            "schema",
            "schema_version",
            "seed",
        }
        or completion.get("schema") != "goalzendo.run_completion"
        or completion.get("schema_version") != 1
        or completion.get("artifact_schema_version") != 2
        or set(completion.get("files", {})) != completion_files
        or identity.get("implementation_fingerprint") != FROZEN_GOALZENDO_IMPLEMENTATION_FINGERPRINT
        or summary.get("plan_key") != spec.plan_key
        or summary.get("seed") != spec.seed
        or summary.get("run_id") != path.name
        or completion.get("run_id") != path.name
        or completion.get("seed") != spec.seed
        or status.get("state") != "complete"
        or status.get("run_id") != path.name
        or status.get("seed") != spec.seed
        or status.get("attempt") != 1
        or status.get("resumed") is not False
        or status.get("repaired_streams") != []
        or status.get("last_step") != 1_000
    ):
        raise EvaluationError("run identity/summary/completion binding differs from the frozen plan")
    attempts = path / "attempts"
    if (
        not attempts.is_dir()
        or {item.name for item in attempts.iterdir()} != {"attempt-0001.json", "resolved-config-0001.yaml"}
        or any(item.is_symlink() or not item.is_file() for item in attempts.iterdir())
    ):
        raise EvaluationError("G00-F run was resumed, repaired, or attempted more than once")
    numerical = {
        "cublas_workspace_config": ":4096:8",
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
        "float32_matmul_precision": "highest",
    }
    if (
        summary.get("backend_version") != EXPERIMENT_BACKEND_VERSION
        or summary.get("final_step") != 1_000
        or summary.get("algorithm") != "sft"
        or summary.get("device") != "cuda"
        or summary.get("numerical_execution") != numerical
        or summary.get("derived_seeds") != dict(spec.seeds)
    ):
        raise EvaluationError("run summary differs from the exact backend/numerical endpoint")
    resolved = _resolved_config(path / "resolved_config.yaml")
    resolved_without_seed = dict(resolved)
    resolved_seed = resolved_without_seed.pop("seed", None)
    if (
        resolved_seed != spec.seed
        or stable_hash(canonical_config(resolved_without_seed), 64)
        != stable_hash(canonical_config(spec.config), 64)
        or stable_hash(canonical_config(spec.config), 64) != row["canonical_resolved_cell_digest"]
    ):
        raise EvaluationError("resolved config seed differs from the frozen plan")
    model_manifest = read_json(path / "manifests" / "model.json")
    tokenizer_manifest = read_json(path / "manifests" / "tokenizer.json")
    dataset_manifest = read_json(path / "manifests" / "dataset.json")
    for manifest, name in (
        (model_manifest, "model"),
        (tokenizer_manifest, "tokenizer"),
        (dataset_manifest, "dataset"),
    ):
        metadata = manifest.get("metadata")
        if (
            manifest.get("schema_version") != 1
            or manifest.get("kind") != name
            or not isinstance(metadata, Mapping)
            or manifest.get("digest") != stable_hash(metadata, 64)
        ):
            raise EvaluationError(f"run {name} manifest schema/digest mismatch")
    model_metadata = model_manifest["metadata"]
    static = CONFIG_SPECS[panel_id]
    expected_model = MODEL_RUNTIME_IDENTITIES[panel_id]
    if (
        model_metadata.get("requested_model") != static["model"]
        or model_metadata.get("requested_revision") != static["revision"]
        or model_metadata.get("resolved_revision") != static["revision"]
        or model_metadata.get("requested_dtype") != "bfloat16"
        or model_metadata.get("model_class") != expected_model["model_class"]
        or model_metadata.get("parameter_count") != expected_model["parameter_count"]
        or model_metadata.get("trainable_parameter_count") != expected_model["trainable_parameter_count"]
        or model_metadata.get("torch_version") != FROZEN_MODEL_DEPENDENCIES["torch"]
        or model_metadata.get("transformers_version") != FROZEN_MODEL_DEPENDENCIES["transformers"]
        or model_metadata.get("peft_version") != FROZEN_MODEL_DEPENDENCIES["peft"]
        or model_metadata.get("update") != {"method": "full"}
        or not isinstance(model_metadata.get("initial_trainable_parameter_digest"), str)
        or len(model_metadata["initial_trainable_parameter_digest"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in model_metadata["initial_trainable_parameter_digest"]
        )
    ):
        raise EvaluationError("runtime model/tokenizer snapshot differs from the frozen revision")
    frozen_row = row
    run_binding = _verify_run_binding(
        verified=verified,
        row=frozen_row,
        run_path=path,
        receipt_cache=receipt_cache,
        expected_provision_receipt_sha256=expected_provision_receipt_sha256,
        expected_pod_id=expected_pod_id,
    )
    environment = read_json(path / "environment.json")
    attempt_environment = read_json(attempts / "attempt-0001.json")
    launch_environment = run_binding["launch_receipt"]["runtime_environment"]
    python_environment = environment.get("python")
    if (
        environment != attempt_environment
        or environment.get("run_id") != path.name
        or environment.get("seed") != spec.seed
        or environment.get("attempt") != 1
        or environment.get("packages") != launch_environment["installed_distributions"]
        or semantic_digest(environment.get("packages"))
        != launch_environment["installed_distributions_digest"]
        or environment.get("accelerator") != launch_environment["accelerator"]
        or semantic_digest(environment.get("accelerator")) != launch_environment["accelerator_digest"]
        or not isinstance(python_environment, Mapping)
        or not str(python_environment.get("version", "")).startswith("3.12.3 ")
        or python_environment.get("implementation") != "CPython"
    ):
        raise EvaluationError("run environment differs from pre-outcome launch inventory")
    tokenizer_metadata = tokenizer_manifest["metadata"]
    expected_tokenizer = TOKENIZER_RUNTIME_IDENTITY
    model_receipt = run_binding["model_receipts"][panel_id]
    if (
        tokenizer_metadata.get("tokenizer_resolved_revision") != static["revision"]
        or tokenizer_metadata.get("tokenizer_class") != expected_tokenizer["tokenizer_class"]
        or Path(str(tokenizer_metadata.get("tokenizer_name_or_path", ""))).resolve()
        != Path(str(model_receipt["snapshot_root"])).resolve()
        or tokenizer_metadata.get("vocabulary_size") != expected_tokenizer["vocabulary_size"]
        or tokenizer_metadata.get("chat_template_sha256") != expected_tokenizer["chat_template_sha256"]
        or tokenizer_metadata.get("action_labels") != expected_tokenizer["action_labels"]
        or tokenizer_metadata.get("action_token_ids") != expected_tokenizer["action_token_ids"]
        or tokenizer_metadata.get("pad_token_id") != expected_tokenizer["pad_token_id"]
        or tokenizer_metadata.get("eos_token_id") != expected_tokenizer["eos_token_id"]
        or tokenizer_metadata.get("bos_token_id") != expected_tokenizer["bos_token_id"]
        or tokenizer_metadata.get("padding_side") != expected_tokenizer["padding_side"]
        or tokenizer_metadata.get("dependency_versions") != FROZEN_MODEL_DEPENDENCIES
    ):
        raise EvaluationError("runtime tokenizer/action/dependency identity changed")
    by_sample, by_pair = regenerated(spec)
    predictions = _final_predictions(path, spec)
    observed, probability_difference = _validate_prediction_rows(predictions, by_sample)

    dataset_metadata = dataset_manifest["metadata"]
    regenerated_metadata = regenerate_dataset(
        spec,
        Path(str(model_receipt["snapshot_root"])),
        tokenizer_cache if tokenizer_cache is not None else {},
    )
    regenerated_symbolic = {
        key: value
        for key, value in regenerated_metadata.items()
        if key not in {"dataset_binding_digest", "rendering"}
    }
    regenerated_rendering = regenerated_metadata.get("rendering")
    expected_binding = stable_hash(
        {"symbolic": regenerated_symbolic, "rendering": regenerated_rendering},
        64,
    )
    if (
        not isinstance(regenerated_rendering, Mapping)
        or dataset_metadata != regenerated_metadata
        or dataset_metadata.get("dataset_binding_digest") != expected_binding
        or summary.get("dataset_binding_digest") != expected_binding
    ):
        raise EvaluationError("run symbolic/rendered dataset binding does not replay canonically")

    view = str(get_path(spec.config, "data.training_view"))
    adapter: dict[str, Any] | None = None
    if view in ADAPTER_TARGET:
        target = ADAPTER_TARGET[view]
        side: dict[str, dict[str, Any]] = {}
        for label, choice in (("A", 0), ("B", 1)):
            rows = [record for record in predictions if int(record[target]) == choice]
            correct = sum(int(record["predicted_action"]) == choice for record in rows)
            passed = len(
                rows
            ) == 256 and correct * MINIMUM_ADAPTER_DENOMINATOR >= MINIMUM_ADAPTER_NUMERATOR * len(rows)
            side[label] = {
                "correct": correct,
                "trials": len(rows),
                "agreement": correct / len(rows) if rows else None,
                "passed": passed,
            }
        adapter = {
            "target": target,
            "positions": side,
            "passed": all(value["passed"] for value in side.values()),
        }

    order_errors = 0
    soft_deviations: list[float] = []
    for roles in by_pair.values():
        base = observed[roles["base"].sample_id]
        mirror = observed[roles["mirror"].sample_id]
        order_errors += int(base["predicted_action"] == mirror["predicted_action"])
        soft_deviations.append(abs(float(base["probability_b"]) + float(mirror["probability_b"]) - 1.0))
    order = {
        "errors": order_errors,
        "pairs": len(by_pair),
        "error_rate": order_errors / len(by_pair),
        "maximum_allowed_errors": MAXIMUM_ORDER_ERRORS,
        "passed": (
            view not in EXPECTED_INFORMATIVE_VIEWS
            or (
                len(by_pair) == MIRROR_PAIRS_PER_RUN
                and order_errors * MAXIMUM_ORDER_ERROR_DENOMINATOR
                <= MAXIMUM_ORDER_ERROR_NUMERATOR * len(by_pair)
            )
        ),
        "soft_deviation_mean": sum(soft_deviations) / len(soft_deviations),
        "soft_deviation_maximum": max(soft_deviations),
        "soft_deviation_quantiles": {
            str(percentile): sorted(soft_deviations)[round((len(soft_deviations) - 1) * percentile / 100)]
            for percentile in (0, 25, 50, 75, 90, 95, 99, 100)
        },
    }
    chance = None
    no_signal_pairing = None
    if view == "no_signal":
        identical_pairs = 0
        marginal_b = 0
        for roles in by_pair.values():
            base = observed[roles["base"].sample_id]
            mirror = observed[roles["mirror"].sample_id]
            identical = (
                base["predicted_action"] == mirror["predicted_action"]
                and base["score_a"] == mirror["score_a"]
                and base["score_b"] == mirror["score_b"]
                and base["probability_b"] == mirror["probability_b"]
                and base["margin_b_minus_a"] == mirror["margin_b_minus_a"]
            )
            identical_pairs += int(identical)
        marginal_b = sum(int(record["predicted_action"]) == 1 for record in predictions)
        structurally_forced_correct = sum(
            int(record["predicted_action"]) == int(record["choice_y"]) for record in predictions
        )
        no_signal_pairing = {
            "canonical_prompt_bytes_identical": True,
            "identical_base_mirror_scores_and_actions": identical_pairs,
            "pairs": len(by_pair),
            "structurally_forced_law_correct": structurally_forced_correct,
            "marginal_action_b_count": marginal_b,
            "marginal_action_b_rate": marginal_b / len(predictions),
            "passed": (
                identical_pairs == MIRROR_PAIRS_PER_RUN
                and structurally_forced_correct == MIRROR_PAIRS_PER_RUN
            ),
        }
    if view == "surface_only":
        correct = sum(int(record["predicted_action"]) == int(record["choice_y"]) for record in predictions)
        chance = {"correct": correct, "trials": len(predictions)}
    return {
        "panel_id": panel_id,
        "plan_key": spec.plan_key,
        "run_id": path.name,
        "seed": spec.seed,
        "law_family": str(get_path(spec.config, "data.rule_family")),
        "training_view": view,
        "adapter": adapter,
        "candidate_order": order,
        "chance": chance,
        "no_signal_pairing": no_signal_pairing,
        "maximum_probability_score_disagreement": probability_difference,
        "completion_digest": completion["completion_digest"],
        "dataset_manifest_digest": dataset_manifest["digest"],
        "model_manifest_digest": model_manifest["digest"],
        "tokenizer_manifest_digest": tokenizer_manifest["digest"],
        "run_freeze_binding": run_binding,
        "initial_trainable_parameter_digest": model_metadata["initial_trainable_parameter_digest"],
    }


def _aggregate_results(results: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    adapter_groups: dict[tuple[str, str, str], list[tuple[int, int]]] = defaultdict(list)
    per_run_adapter_passed = True
    order_passed = True
    scorer_maximum = 0.0
    chance_counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    no_signal_rows: list[dict[str, Any]] = []
    no_signal_passed = True
    initialization_digests: dict[str, set[str]] = defaultdict(set)
    for result in results:
        panel_id = str(result["panel_id"])
        view = str(result["training_view"])
        initialization_digests[panel_id].add(str(result["initial_trainable_parameter_digest"]))
        scorer_maximum = max(scorer_maximum, float(result["maximum_probability_score_disagreement"]))
        adapter = result.get("adapter")
        if isinstance(adapter, Mapping):
            per_run_adapter_passed = per_run_adapter_passed and adapter.get("passed") is True
            positions = adapter["positions"]
            for position in ("A", "B"):
                values = positions[position]
                adapter_groups[(panel_id, view, position)].append(
                    (int(values["correct"]), int(values["trials"]))
                )
                adapter_groups[("aggregate", view, position)].append(
                    (int(values["correct"]), int(values["trials"]))
                )
        if view in EXPECTED_INFORMATIVE_VIEWS:
            order_passed = order_passed and result["candidate_order"]["passed"] is True
        chance = result.get("chance")
        if isinstance(chance, Mapping):
            chance_counts[(panel_id, view)][0] += int(chance["correct"])
            chance_counts[(panel_id, view)][1] += int(chance["trials"])
            chance_counts[("aggregate", view)][0] += int(chance["correct"])
            chance_counts[("aggregate", view)][1] += int(chance["trials"])
        no_signal = result.get("no_signal_pairing")
        if isinstance(no_signal, Mapping):
            no_signal_passed = no_signal_passed and no_signal.get("passed") is True
            no_signal_rows.append(
                {
                    "panel_id": panel_id,
                    "plan_key": result["plan_key"],
                    "seed": result["seed"],
                    **dict(no_signal),
                }
            )

    aggregate_adapter: dict[str, Any] = {}
    aggregate_adapter_passed = True
    expected_adapter_runs = {
        "audit_law_matched": 20,
        "herald_only": 10,
        "law_only": 20,
        "sage_only": 10,
    }
    expected_adapter_keys = {
        (panel_id, view, position)
        for panel_id in (*CONFIG_SPECS, "aggregate")
        for view in EXPECTED_INFORMATIVE_VIEWS
        for position in ("A", "B")
    }
    if set(adapter_groups) != expected_adapter_keys:
        aggregate_adapter_passed = False
    for (panel_id, view, position), values in sorted(adapter_groups.items()):
        correct = sum(value[0] for value in values)
        trials = sum(value[1] for value in values)
        expected_runs = expected_adapter_runs[view] * (2 if panel_id == "aggregate" else 1)
        passed = (
            len(values) == expected_runs
            and trials == expected_runs * 256
            and all(value[1] == 256 for value in values)
            and correct * 100 >= 95 * trials
        )
        aggregate_adapter_passed = aggregate_adapter_passed and passed
        aggregate_adapter.setdefault(panel_id, {}).setdefault(view, {})[position] = {
            "correct": correct,
            "trials": trials,
            "agreement": correct / trials,
            "run_count": len(values),
            "expected_run_count": expected_runs,
            "passed": passed,
        }

    chance_results: dict[str, Any] = {}
    chance_passed = True
    for (panel_id, view), (correct, trials) in sorted(chance_counts.items()):
        expected_trials = 10_240 if panel_id == "aggregate" else 5_120
        interval = exact_central_binomial_interval(expected_trials)
        passed = trials == expected_trials and interval[0] <= correct <= interval[1]
        chance_passed = chance_passed and passed
        chance_results.setdefault(panel_id, {})[view] = {
            "correct": correct,
            "trials": trials,
            "acceptance_interval": list(interval),
            "passed": passed,
        }
    expected_chance_cells = {(panel_id, "surface_only") for panel_id in (*CONFIG_SPECS, "aggregate")}
    if set(chance_counts) != expected_chance_cells:
        chance_passed = False

    measurements = {
        "initial_trainable_parameter_digests": {
            panel_id: sorted(values) for panel_id, values in sorted(initialization_digests.items())
        },
        "adapter_aggregate": aggregate_adapter,
        "per_run_adapters": [
            {
                "panel_id": result["panel_id"],
                "plan_key": result["plan_key"],
                "seed": result["seed"],
                "law_family": result["law_family"],
                "training_view": result["training_view"],
                "adapter": result["adapter"],
            }
            for result in results
            if result.get("adapter") is not None
        ],
        "candidate_order_by_run": [
            {
                "panel_id": result["panel_id"],
                "plan_key": result["plan_key"],
                "seed": result["seed"],
                "law_family": result["law_family"],
                "training_view": result["training_view"],
                **dict(result["candidate_order"]),
            }
            for result in results
        ],
        "chance": chance_results,
        "surface_only_by_seed": [
            {
                "panel_id": result["panel_id"],
                "plan_key": result["plan_key"],
                "seed": result["seed"],
                "law_family": result["law_family"],
                **dict(result["chance"]),
            }
            for result in results
            if result.get("chance") is not None
        ],
        "no_signal_pair_determinism": no_signal_rows,
        "constrained_scorer": {
            "all_scores_and_probabilities_finite": True,
            "probability_range_valid": True,
            "maximum_probability_score_disagreement": scorer_maximum,
            "maximum_allowed_probability_score_disagreement": MAXIMUM_PROBABILITY_SCORE_DISAGREEMENT,
        },
    }
    checks = {
        "panel_completion": {"passed": True, "completed_runs": 160, "planned_runs": 160},
        "run_and_dataset_integrity": {
            "passed": True,
            "completion_attestations": 160,
            "regenerated_final_banks": 160,
            "exact_mirror_pairs": 160 * 256,
        },
        "model_initialization_identity": {
            "passed": set(initialization_digests) == set(CONFIG_SPECS)
            and all(len(values) == 1 for values in initialization_digests.values()),
            "requirement": "one exact initial full-model parameter digest within each panel",
        },
        "rule_adapters": {
            "passed": per_run_adapter_passed and aggregate_adapter_passed,
            "per_run_passed": per_run_adapter_passed,
            "aggregate_passed": aggregate_adapter_passed,
            "threshold": "correct/trials >= 95/100",
        },
        "g00f_paired_candidate_order_symmetry_v1": {
            "passed": order_passed,
            "per_informative_run_maximum": "5 errors among exactly 256 pairs",
        },
        "no_signal_pair_determinism": {
            "passed": no_signal_passed and len(no_signal_rows) == 20,
            "interpretation": (
                "exact deterministic-pair integrity; 256/512 Law correctness is structurally forced, "
                "not an IID label-leakage test"
            ),
        },
        "surface_leakage": {
            "passed": chance_passed
            and all(
                chance_results[panel]["surface_only"]["passed"] for panel in (*CONFIG_SPECS, "aggregate")
            ),
            "interpretation": (
                "fixed engineering qualification bands from a Binomial(n,.5) reference; "
                "not an IID scientific confidence interval"
            ),
        },
        "constrained_score_integrity": {
            "passed": scorer_maximum <= MAXIMUM_PROBABILITY_SCORE_DISAGREEMENT,
        },
    }
    return measurements, checks


def _false_assessment(
    *,
    verified: VerifiedFreeze,
    itt_rows: Sequence[Mapping[str, Any]],
    reason: str,
    runtime: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    state_counts = dict(sorted(Counter(str(row["state"]) for row in itt_rows).items()))
    checks = {
        "panel_completion": {
            "passed": False,
            "planned_runs": 160,
            "completed_runs": state_counts.get("complete", 0),
            "state_counts": state_counts,
            "reason": reason,
        },
        "run_and_dataset_integrity": {"passed": False, "status": "not_evaluable"},
        "model_initialization_identity": {"passed": False, "status": "not_evaluable"},
        "pretraining_model_boundary": {"passed": False, "status": "not_evaluable"},
        "rule_adapters": {"passed": False, "status": "not_evaluable"},
        "g00f_paired_candidate_order_symmetry_v1": {
            "passed": False,
            "status": "not_evaluable",
        },
        "no_signal_pair_determinism": {"passed": False, "status": "not_evaluable"},
        "surface_leakage": {"passed": False, "status": "not_evaluable"},
        "constrained_score_integrity": {"passed": False, "status": "not_evaluable"},
        "runtime_ceiling": {
            "passed": bool(runtime and runtime.get("passed") is True),
            "status": "verified" if runtime else "not_evaluable",
        },
    }
    body = {
        "schema": ASSESSMENT_SCHEMA,
        "schema_version": ASSESSMENT_SCHEMA_VERSION,
        "study_id": "g00f",
        "freeze_file_sha256": verified.file_sha256,
        "freeze_digest": verified.digest,
        "evidence_status": "incomplete_or_invalid",
        "intention_to_train": list(itt_rows),
        "runtime": dict(runtime) if runtime else None,
        "measurements": None,
        "checks": checks,
        "overall_passed": False,
        "failure_reason": reason,
    }
    return {**body, "assessment_digest": semantic_digest(body)}


def derive_assessment(
    *,
    verified: VerifiedFreeze,
    artifact_roots: Mapping[str, str | Path],
    ledger_root: str | Path,
    worker_result_receipts: Mapping[int, str | Path],
    expected_provision_receipt_sha256: str,
    expected_pod_id: str,
    regenerate: Callable[[RunSpec], tuple[Mapping[str, Any], Mapping[str, Any]]] = _regenerated_final_map,
) -> dict[str, Any]:
    """Derive a pass/fail assessment; never throw for incomplete ITT evidence."""

    try:
        expected, itt_rows = _expected_paths(verified, artifact_roots)
    except Exception as error:
        return _false_assessment(
            verified=verified,
            itt_rows=[],
            reason=f"artifact_inventory_error:{type(error).__name__}",
        )
    try:
        attempt_audit = _audit_attempt_ledger(verified=verified, ledger_root=ledger_root)
        for row in itt_rows:
            plan_key = row.get("plan_key")
            row["attempt_state"] = attempt_audit["states"].get(plan_key) if plan_key else None
    except Exception as error:
        return _false_assessment(
            verified=verified,
            itt_rows=itt_rows,
            reason=f"attempt_ledger_error:{type(error).__name__}",
        )
    incomplete = [
        row for row in itt_rows if row.get("state") != "complete" or row.get("attempt_state") != "complete"
    ]
    if incomplete:
        return _false_assessment(
            verified=verified,
            itt_rows=itt_rows,
            reason="intention_to_train_panel_not_exactly_complete",
        )
    try:
        runtime = _verify_worker_result_receipts(
            verified,
            worker_result_receipts,
            expected_provision_receipt_sha256=expected_provision_receipt_sha256,
            expected_pod_id=expected_pod_id,
        )
        _require_same_execution(attempt_audit, runtime)
        frozen_rows = {str(row["plan_key"]): row for row in verified.all_rows}
        receipt_cache: dict[tuple[str, str], Mapping[str, Any]] = {}
        tokenizer_cache: dict[str, Any] = {}
        attempt_rows_by_key = {
            str(row["plan_key"]): row for row in attempt_audit["rows"] if row.get("plan_key") is not None
        }
        results: list[dict[str, Any]] = []
        for _run_id, (panel_id, spec, path) in sorted(expected.items()):
            results.append(
                _evaluate_run(
                    verified=verified,
                    panel_id=panel_id,
                    spec=spec,
                    path=path,
                    row=frozen_rows[spec.plan_key],
                    outcome_file_seals=attempt_rows_by_key[spec.plan_key]["outcome_file_seals"],
                    regenerated=regenerate,
                    receipt_cache=receipt_cache,
                    tokenizer_cache=tokenizer_cache,
                    expected_provision_receipt_sha256=expected_provision_receipt_sha256,
                    expected_pod_id=expected_pod_id,
                )
            )
        measurements, checks = _aggregate_results(results)
        boundary_audits = attempt_audit["ledger"]["model_integration_audits"]
        measurements["pretraining_model_boundary"] = copy.deepcopy(dict(boundary_audits))
        checks["pretraining_model_boundary"] = {
            "passed": set(boundary_audits) == set(CONFIG_SPECS),
            "panels": sorted(boundary_audits),
            "swap_atol": 2e-3,
            "swap_rtol": 2e-3,
            "timing": "before_itt_ledger_and_weight_updates",
        }
        checks["runtime_ceiling"] = {"passed": True, **runtime}
        overall = all(value.get("passed") is True for value in checks.values())
        evidence_rows = [
            {
                "panel_id": result["panel_id"],
                "plan_key": result["plan_key"],
                "run_id": result["run_id"],
                "completion_digest": result["completion_digest"],
                "dataset_manifest_digest": result["dataset_manifest_digest"],
                "model_manifest_digest": result["model_manifest_digest"],
                "tokenizer_manifest_digest": result["tokenizer_manifest_digest"],
                "initial_trainable_parameter_digest": result["initial_trainable_parameter_digest"],
                "run_freeze_binding": result["run_freeze_binding"],
            }
            for result in results
        ]
        body = {
            "schema": ASSESSMENT_SCHEMA,
            "schema_version": ASSESSMENT_SCHEMA_VERSION,
            "study_id": "g00f",
            "freeze_file_sha256": verified.file_sha256,
            "freeze_digest": verified.digest,
            "evidence_status": "complete",
            "intention_to_train": itt_rows,
            "runtime": runtime,
            "attempt_ledger": attempt_audit,
            "evidence_rows": evidence_rows,
            "evidence_digest": semantic_digest(evidence_rows),
            "measurements": measurements,
            "checks": checks,
            "overall_passed": overall,
            "failure_reason": None if overall else "one_or_more_frozen_checks_failed",
        }
        return {**body, "assessment_digest": semantic_digest(body)}
    except Exception as error:
        return _false_assessment(
            verified=verified,
            itt_rows=itt_rows,
            reason=f"evidence_validation_error:{type(error).__name__}",
        )


def gate_from_assessment(verified: VerifiedFreeze, assessment: Mapping[str, Any]) -> dict[str, Any]:
    body = {
        "schema": GATE_SCHEMA,
        "schema_version": GATE_SCHEMA_VERSION,
        "study_id": "g00f",
        "freeze_file_sha256": verified.file_sha256,
        "freeze_digest": verified.digest,
        "assessment": copy.deepcopy(dict(assessment)),
        "checks": copy.deepcopy(dict(assessment.get("checks", {}))),
        "overall_passed": assessment.get("overall_passed") is True,
        "authorization": {
            "g00f_remediation_passed": assessment.get("overall_passed") is True,
            "g01_launch_authorized": False,
            "scope": "none",
            "reason": "a_separately_reviewed_digest_bound_g01_bridge_is_required",
        },
    }
    return {**body, "gate_digest": semantic_digest(body)}


def create_gate(
    *,
    verified: VerifiedFreeze,
    artifact_roots: Mapping[str, str | Path],
    ledger_root: str | Path,
    worker_result_receipts: Mapping[int, str | Path],
    expected_provision_receipt_sha256: str,
    expected_pod_id: str,
    assessment_output: str | Path,
    gate_output: str | Path,
) -> dict[str, Any]:
    """Always persist canonical assessment/gate bytes after freeze verification."""

    assessment = derive_assessment(
        verified=verified,
        artifact_roots=artifact_roots,
        ledger_root=ledger_root,
        worker_result_receipts=worker_result_receipts,
        expected_provision_receipt_sha256=expected_provision_receipt_sha256,
        expected_pod_id=expected_pod_id,
    )
    gate = gate_from_assessment(verified, assessment)
    atomic_json(assessment_output, assessment)
    atomic_json(gate_output, gate)
    return {
        "assessment": assessment,
        "assessment_file_sha256": sha256_file(assessment_output),
        "gate": gate,
        "gate_file_sha256": sha256_file(gate_output),
    }


def create_preexecution_gate(
    *,
    verified: VerifiedFreeze,
    assessment_output: str | Path,
    gate_output: str | Path,
) -> dict[str, Any]:
    rows = [
        {
            "panel_id": row["panel_id"],
            "plan_key": row["plan_key"],
            "run_id": row["run_id"],
            "seed": row["seed"],
            "law_family": row["law_family"],
            "training_view": row["training_view"],
            "worker_index": row["worker_index"],
            "worker_order": row["worker_order"],
            "state": "not_run",
            "error_type": None,
            "attempt_count": 0,
            "intention_to_train": True,
        }
        for row in verified.all_rows
    ]
    assessment = _false_assessment(
        verified=verified,
        itt_rows=rows,
        reason="prospective_freeze_no_model_execution",
    )
    gate = gate_from_assessment(verified, assessment)
    atomic_json(assessment_output, assessment)
    atomic_json(gate_output, gate)
    return {
        "assessment": assessment,
        "assessment_file_sha256": sha256_file(assessment_output),
        "gate": gate,
        "gate_file_sha256": sha256_file(gate_output),
    }


def verify_gate_artifact(
    path: str | Path,
    *,
    verified: VerifiedFreeze,
    expected_gate_sha256: str | None = None,
) -> dict[str, Any]:
    if (
        expected_gate_sha256 is None
        or len(expected_gate_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_gate_sha256)
        or sha256_file(path) != expected_gate_sha256
    ):
        raise EvaluationError("G00-F gate bytes lack the mandatory external SHA-256 binding")
    gate = strict_json(path, "G00-F gate artifact")
    body = {key: value for key, value in gate.items() if key != "gate_digest"}
    assessment = gate.get("assessment")
    if not isinstance(assessment, Mapping):
        raise EvaluationError("G00-F gate omits its assessment")
    assessment_body = {key: value for key, value in assessment.items() if key != "assessment_digest"}
    gate_keys = {
        "assessment",
        "authorization",
        "checks",
        "freeze_digest",
        "freeze_file_sha256",
        "gate_digest",
        "overall_passed",
        "schema",
        "schema_version",
        "study_id",
    }
    assessment_keys = {
        "assessment_digest",
        "checks",
        "evidence_status",
        "failure_reason",
        "freeze_digest",
        "freeze_file_sha256",
        "intention_to_train",
        "measurements",
        "overall_passed",
        "runtime",
        "schema",
        "schema_version",
        "study_id",
    }
    if assessment.get("evidence_status") == "complete":
        assessment_keys |= {"attempt_ledger", "evidence_digest", "evidence_rows"}
    expected_authorization = {
        "g00f_remediation_passed": gate.get("overall_passed") is True,
        "g01_launch_authorized": False,
        "scope": "none",
        "reason": "a_separately_reviewed_digest_bound_g01_bridge_is_required",
    }
    if (
        set(gate) != gate_keys
        or set(assessment) != assessment_keys
        or gate.get("schema") != GATE_SCHEMA
        or gate.get("schema_version") != GATE_SCHEMA_VERSION
        or gate.get("study_id") != "g00f"
        or gate.get("freeze_file_sha256") != verified.file_sha256
        or gate.get("freeze_digest") != verified.digest
        or gate.get("gate_digest") != semantic_digest(body)
        or assessment.get("schema") != ASSESSMENT_SCHEMA
        or assessment.get("schema_version") != ASSESSMENT_SCHEMA_VERSION
        or assessment.get("study_id") != "g00f"
        or assessment.get("freeze_file_sha256") != verified.file_sha256
        or assessment.get("freeze_digest") != verified.digest
        or assessment.get("assessment_digest") != semantic_digest(assessment_body)
        or gate.get("checks") != assessment.get("checks")
        or gate.get("overall_passed") != assessment.get("overall_passed")
        or not isinstance(gate.get("overall_passed"), bool)
        or gate.get("authorization") != expected_authorization
    ):
        raise EvaluationError("G00-F gate schema, digest, freeze, or nonauthorization changed")
    return {
        "path": str(Path(path).resolve()),
        "file_sha256": sha256_file(path),
        "gate_digest": gate["gate_digest"],
        "overall_passed": gate["overall_passed"],
        "g01_launch_authorized": False,
    }


__all__ = [
    "ASSESSMENT_SCHEMA",
    "GATE_SCHEMA",
    "MAXIMUM_ORDER_ERRORS",
    "WORKER_RESULT_SCHEMA",
    "EvaluationError",
    "create_gate",
    "create_preexecution_gate",
    "derive_assessment",
    "exact_central_binomial_interval",
    "gate_from_assessment",
    "verify_gate_artifact",
]
