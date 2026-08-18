"""Deterministic planning and backend-neutral execution for GoalZendo.

The runner owns experiment bookkeeping, not model training.  A training backend
receives a :class:`RunContext`, streams records through its :class:`RunStore`,
and returns a small :class:`BackendResult`.  Keeping this boundary narrow lets
the symbolic generator and the Hugging Face training loop evolve independently
without weakening artifact identity or resumption guarantees.
"""

from __future__ import annotations

import copy
import importlib
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import yaml  # type: ignore[import-untyped]

from .artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    ArtifactError,
    RunStore,
    _identity_config,
    discover_runs,
    implementation_provenance,
    json_safe,
    read_json,
    read_jsonl,
    stable_hash,
    utc_now,
    verify_completion_attestation,
    write_json,
)
from .config import (
    REQUIRED_LAUNCH_GUARDS,
    canonical_config,
    expand_sweep,
    get_path,
    smoke_config,
)

DEFAULT_BACKEND = "goalzendo.experiment:run_experiment"
DERIVED_SEED_NAMES = (
    "dataset",
    "validation",
    "factorial_evaluation",
    "rendering",
    "model_initialization",
    "training",
    "action_sampling",
)
G00_GATE_SCHEMA = "goalzendo.g00_gate"
G00_GATE_SCHEMA_VERSION = 2


class PlanError(ValueError):
    """Raised when an execution plan is ambiguous or invalid."""


class BackendContractError(RuntimeError):
    """Raised when a training backend omits required reproducibility data."""


class LaunchGuardError(RuntimeError):
    """Raised when a preregistered protocol is intentionally still locked."""


@dataclass(frozen=True)
class G00GateThresholds:
    """Numerical engineering criteria frozen before any confirmatory launch."""

    minimum_rule_adapter_position_agreement: float = 0.95
    minimum_finite_probability_fraction: float = 1.0
    maximum_probability_sum_error: float = 1e-6
    maximum_absolute_position_bias: float = 0.02
    binomial_alpha: float = 0.05
    chance_probability: float = 0.50
    minimum_iid_improvement: float = 0.10
    minimum_terminal_iid_accuracy: float = 0.90
    maximum_high_baseline_regression: float = 0.02
    maximum_terminal_accuracy_drop: float = 0.05
    maximum_terminal_action_frequency: float = 0.95
    minimum_rl_early_both_actions_fraction: float = 0.25
    maximum_rl_early_all_zero_advantages_fraction: float = 0.75
    maximum_integrity_mismatches: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "minimum_rule_adapter_position_agreement": self.minimum_rule_adapter_position_agreement,
            "minimum_finite_probability_fraction": self.minimum_finite_probability_fraction,
            "maximum_probability_sum_error": self.maximum_probability_sum_error,
            "maximum_absolute_position_bias": self.maximum_absolute_position_bias,
            "binomial_alpha": self.binomial_alpha,
            "chance_probability": self.chance_probability,
            "minimum_iid_improvement": self.minimum_iid_improvement,
            "minimum_terminal_iid_accuracy": self.minimum_terminal_iid_accuracy,
            "maximum_high_baseline_regression": self.maximum_high_baseline_regression,
            "maximum_terminal_accuracy_drop": self.maximum_terminal_accuracy_drop,
            "maximum_terminal_action_frequency": self.maximum_terminal_action_frequency,
            "minimum_rl_early_both_actions_fraction": (
                self.minimum_rl_early_both_actions_fraction
            ),
            "maximum_rl_early_all_zero_advantages_fraction": (
                self.maximum_rl_early_all_zero_advantages_fraction
            ),
            "maximum_integrity_mismatches": self.maximum_integrity_mismatches,
        }


G00_GATE_THRESHOLDS = G00GateThresholds()


@dataclass(frozen=True)
class G00OptimizerStabilityPolicy:
    """Exact evidence window and coverage used to select G01 optimizers."""

    baseline_step: int = 0
    terminal_steps: tuple[int, int] = (128, 256)
    rl_sampling_steps: tuple[int, ...] = tuple(range(1, 33))
    required_law_families: tuple[str, str] = ("majority", "parity")
    required_proxy_accuracies: tuple[float, float] = (0.95, 1.0)
    independent_seed_count: int = 3
    selection_rule: str = "minimum_learning_rate_then_entropy"

    def as_dict(self) -> dict[str, Any]:
        return {
            "baseline_step": self.baseline_step,
            "terminal_steps": list(self.terminal_steps),
            "rl_sampling_steps": list(self.rl_sampling_steps),
            "required_law_families": list(self.required_law_families),
            "required_proxy_accuracies": list(self.required_proxy_accuracies),
            "independent_seed_count": self.independent_seed_count,
            "selection_rule": self.selection_rule,
        }


G00_OPTIMIZER_STABILITY_POLICY = G00OptimizerStabilityPolicy()
_G00_EVIDENCE_VERIFICATION_CACHE: dict[tuple[str, str, str], dict[str, Any]] = {}


@dataclass(frozen=True)
class G00GateArtifact:
    payload: Mapping[str, Any]
    path: Path

    @property
    def passed(self) -> bool:
        return self.payload.get("overall_passed") is True

    @property
    def digest(self) -> str:
        return str(self.payload.get("gate_digest", ""))

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "gate_digest": self.digest,
            "overall_passed": self.passed,
            "checks": self.payload.get("checks", {}),
            "authorized_targets": sorted(str(key) for key in self.payload.get("authorized_targets", {})),
        }


def derive_seed(master_seed: int, namespace: str) -> int:
    """Derive a stable, nonzero 31-bit seed without touching global RNG state."""

    if isinstance(master_seed, bool) or not isinstance(master_seed, int):
        raise ValueError("master_seed must be an integer")
    if not namespace or not str(namespace).strip():
        raise ValueError("namespace must be non-empty")
    maximum = 2**31 - 1
    return int(stable_hash({"master_seed": master_seed, "namespace": namespace}, 16), 16) % (maximum - 1) + 1


def derived_seeds(master_seed: int) -> dict[str, int]:
    return {name: derive_seed(master_seed, name) for name in DERIVED_SEED_NAMES}


def _planning_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Remove scheduling/location fields from a logical plan cell."""

    value = copy.deepcopy(dict(config))
    value.pop("sweep", None)
    value.pop("cases", None)
    for key in tuple(value):
        if str(key).startswith("_"):
            value.pop(key, None)
    run = dict(value.get("run", {}))
    for field_name in ("output_root", "resume", "seeds"):
        run.pop(field_name, None)
    value["run"] = run
    return json_safe(value)


@dataclass(frozen=True)
class RunSpec:
    """One sweep cell, focal seed, and stable shard assignment."""

    config: Mapping[str, Any]
    seed: int
    cell_index: int
    global_index: int
    cell_id: str
    plan_key: str
    shard_index: int
    num_shards: int
    seeds: Mapping[str, int]

    def as_dict(self, *, include_config: bool = False) -> dict[str, Any]:
        value: dict[str, Any] = {
            "global_index": self.global_index,
            "cell_index": self.cell_index,
            "cell_id": self.cell_id,
            "plan_key": self.plan_key,
            "seed": self.seed,
            "derived_seeds": dict(self.seeds),
            "shard_index": self.shard_index,
            "num_shards": self.num_shards,
            "sweep_values": copy.deepcopy(dict(self.config.get("_sweep_values", {}))),
        }
        if include_config:
            value["config"] = json_safe(dict(self.config))
        return value


def build_plan(
    config: Mapping[str, Any],
    *,
    smoke: bool = False,
    shard_index: int = 0,
    num_shards: int = 1,
) -> tuple[RunSpec, ...]:
    """Expand cells and seeds with stable hash-based shard allocation.

    Sharding is based on each seed-level logical key rather than row number.
    Adding another seed or sweep cell therefore does not move existing work
    between shards.
    """

    if isinstance(num_shards, bool) or int(num_shards) < 1:
        raise PlanError("num_shards must be a positive integer")
    if isinstance(shard_index, bool) or not 0 <= int(shard_index) < int(num_shards):
        raise PlanError("shard_index must lie in [0, num_shards)")
    shard_index = int(shard_index)
    num_shards = int(num_shards)

    effective = smoke_config(config) if smoke else copy.deepcopy(dict(config))
    cells = expand_sweep(effective)
    all_specs: list[RunSpec] = []
    seen: set[str] = set()
    global_index = 0
    for cell_index, cell in enumerate(cells):
        raw_seeds = get_path(cell, "run.seeds", [])
        if not isinstance(raw_seeds, Sequence) or isinstance(raw_seeds, (str, bytes)):
            raise PlanError("run.seeds must be a sequence of integers")
        seeds = sorted(int(seed) for seed in raw_seeds)
        if any(isinstance(seed, bool) for seed in raw_seeds):
            raise PlanError("Boolean values are not valid seeds")
        cell_science = _planning_config(cell)
        cell_id = stable_hash(cell_science, 16)
        for seed in seeds:
            plan_key = stable_hash({"config": cell_science, "seed": seed}, 20)
            if plan_key in seen:
                raise PlanError(
                    f"sweep expansion produced duplicate seed-level work; first duplicate key is {plan_key}"
                )
            seen.add(plan_key)
            assigned_shard = int(plan_key[:16], 16) % num_shards
            spec = RunSpec(
                config=copy.deepcopy(cell),
                seed=seed,
                cell_index=cell_index,
                global_index=global_index,
                cell_id=cell_id,
                plan_key=plan_key,
                shard_index=assigned_shard,
                num_shards=num_shards,
                seeds=derived_seeds(seed),
            )
            global_index += 1
            if assigned_shard == shard_index:
                all_specs.append(spec)
    return tuple(all_specs)


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise LaunchGuardError(f"G00 gate measurement {name} must be numeric")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise LaunchGuardError(f"G00 gate measurement {name} must be numeric") from exc
    if not math.isfinite(numeric):
        raise LaunchGuardError(f"G00 gate measurement {name} must be finite")
    return numeric


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LaunchGuardError(f"G00 gate measurement {name} must be a non-negative integer")
    return int(value)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LaunchGuardError(f"G00 gate measurement {name} must be an object")
    return value


def _binomial_acceptance_interval(
    trials: int,
    *,
    probability: float,
    alpha: float,
) -> tuple[int, int]:
    """Return the central exact-binomial acceptance region for a fixed n."""

    if trials < 1:
        raise LaunchGuardError("binomial G00 checks require at least one trial")
    probabilities = [
        math.comb(trials, successes) * probability**successes * (1.0 - probability) ** (trials - successes)
        for successes in range(trials + 1)
    ]
    lower_tail = alpha / 2.0
    upper_tail = 1.0 - lower_tail
    cumulative = 0.0
    lower = 0
    upper = trials
    found_lower = False
    for successes, mass in enumerate(probabilities):
        cumulative += mass
        if not found_lower and cumulative >= lower_tail:
            lower = successes
            found_lower = True
        if cumulative >= upper_tail:
            upper = successes
            break
    return lower, upper


def _evaluate_capability_checks(
    assessment: Mapping[str, Any],
    thresholds: G00GateThresholds,
    *,
    prefix: str = "",
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    label_prefix = f"{prefix}." if prefix else ""
    adapters = _mapping(assessment.get("rule_adapter_position_agreement"), f"{label_prefix}rule adapters")
    adapter_values: list[float] = []
    required_adapters = ("law_only", "audit_law_matched", "sage_only", "herald_only")
    for adapter in required_adapters:
        positions = _mapping(adapters.get(adapter), f"{label_prefix}rule adapters.{adapter}")
        if set(positions) != {"A", "B"}:
            raise LaunchGuardError(f"{label_prefix}rule adapters.{adapter} must contain exactly A and B")
        adapter_values.extend(
            _finite_number(positions[position], f"{label_prefix}rule adapters.{adapter}.{position}")
            for position in ("A", "B")
        )
    adapters_minimum = min(adapter_values)
    checks["rule_adapters"] = {
        "observed_minimum": adapters_minimum,
        "required_minimum": thresholds.minimum_rule_adapter_position_agreement,
        "passed": adapters_minimum >= thresholds.minimum_rule_adapter_position_agreement,
    }
    for field_name, output_key in (
        ("law_only_position_agreement_by_run", "law_only_by_run"),
        ("matched_law_position_agreement_by_run", "matched_law_by_run"),
    ):
        by_run = assessment.get(field_name)
        if not isinstance(by_run, Sequence) or isinstance(by_run, (str, bytes)) or not by_run:
            raise LaunchGuardError(
                f"{label_prefix}{field_name} must be a nonempty list"
            )
        seen_runs: set[str] = set()
        run_checks: list[dict[str, Any]] = []
        for index, raw_record in enumerate(by_run):
            record = _mapping(raw_record, f"{label_prefix}{output_key}[{index}]")
            run_id = str(record.get("run_id", ""))
            law_family = str(record.get("law_family", ""))
            seed = _nonnegative_integer(record.get("seed"), f"{label_prefix}{output_key}.seed")
            if not run_id or run_id in seen_runs or law_family not in {"parity", "majority"}:
                raise LaunchGuardError(
                    f"{label_prefix}{output_key} records need unique run IDs and a known Law family"
                )
            seen_runs.add(run_id)
            agreement_a = _finite_number(record.get("A"), f"{label_prefix}{output_key}.A")
            agreement_b = _finite_number(record.get("B"), f"{label_prefix}{output_key}.B")
            passed = min(agreement_a, agreement_b) >= thresholds.minimum_rule_adapter_position_agreement
            run_checks.append(
                {
                    "run_id": run_id,
                    "seed": seed,
                    "law_family": law_family,
                    "A": agreement_a,
                    "B": agreement_b,
                    "passed": passed,
                }
            )
        checks["rule_adapters"][output_key] = run_checks
        checks["rule_adapters"]["passed"] = bool(
            checks["rule_adapters"]["passed"] and all(record["passed"] for record in run_checks)
        )

    scorer = _mapping(assessment.get("constrained_scorer"), f"{label_prefix}constrained scorer")
    finite_fraction = _finite_number(
        scorer.get("finite_probability_fraction"),
        f"{label_prefix}constrained scorer.finite_probability_fraction",
    )
    sum_error = _finite_number(
        scorer.get("maximum_probability_sum_error"),
        f"{label_prefix}constrained scorer.maximum_probability_sum_error",
    )
    position_bias = abs(
        _finite_number(
            scorer.get("maximum_absolute_position_bias"),
            f"{label_prefix}constrained scorer.maximum_absolute_position_bias",
        )
    )
    checks["constrained_scorer"] = {
        "finite_probability_fraction": finite_fraction,
        "required_finite_probability_fraction": thresholds.minimum_finite_probability_fraction,
        "maximum_probability_sum_error": sum_error,
        "allowed_probability_sum_error": thresholds.maximum_probability_sum_error,
        "maximum_absolute_position_bias": position_bias,
        "allowed_absolute_position_bias": thresholds.maximum_absolute_position_bias,
        "passed": (
            finite_fraction >= thresholds.minimum_finite_probability_fraction
            and sum_error <= thresholds.maximum_probability_sum_error
            and position_bias <= thresholds.maximum_absolute_position_bias
        ),
    }

    for source_name, check_name in (
        ("no_signal", "no_signal_chance"),
        ("surface_classifier", "surface_leakage"),
    ):
        observed = _mapping(assessment.get(source_name), f"{label_prefix}{source_name}")
        successes = _nonnegative_integer(observed.get("correct"), f"{label_prefix}{source_name}.correct")
        trials = _nonnegative_integer(observed.get("trials"), f"{label_prefix}{source_name}.trials")
        if successes > trials:
            raise LaunchGuardError(f"{source_name}.correct cannot exceed trials")
        lower, upper = _binomial_acceptance_interval(
            trials,
            probability=thresholds.chance_probability,
            alpha=thresholds.binomial_alpha,
        )
        checks[check_name] = {
            "correct": successes,
            "trials": trials,
            "accuracy": successes / trials,
            "chance_probability": thresholds.chance_probability,
            "alpha": thresholds.binomial_alpha,
            "accepted_correct_minimum": lower,
            "accepted_correct_maximum": upper,
            "passed": lower <= successes <= upper,
        }
    return checks


def _bounded_probability(value: Any, label: str) -> float:
    observed = _finite_number(value, label)
    if not 0.0 <= observed <= 1.0:
        raise LaunchGuardError(f"{label} must lie in [0, 1]")
    return observed


def _evaluate_optimizer_candidate(
    candidate: Mapping[str, Any],
    *,
    algorithm: str,
    candidate_index: int,
    thresholds: G00GateThresholds,
    policy: G00OptimizerStabilityPolicy,
) -> dict[str, Any]:
    label = f"optimizer stability.{algorithm}[{candidate_index}]"
    learning_rate = _finite_number(candidate.get("learning_rate"), f"{label}.learning_rate")
    entropy = _finite_number(
        candidate.get("entropy_coefficient", 0.0), f"{label}.entropy_coefficient"
    )
    if learning_rate <= 0.0 or entropy < 0.0:
        raise LaunchGuardError(f"{label} has an invalid learning rate or entropy coefficient")
    raw_steps = candidate.get("stability_window_steps")
    if not isinstance(raw_steps, Sequence) or isinstance(raw_steps, (str, bytes)):
        raise LaunchGuardError(f"{label}.stability_window_steps must be a list")
    steps = tuple(_nonnegative_integer(step, f"{label}.stability_window_steps") for step in raw_steps)
    if steps != policy.terminal_steps:
        raise LaunchGuardError(f"{label} does not use the frozen terminal stability window")

    raw_runs = candidate.get("run_measurements")
    if not isinstance(raw_runs, Sequence) or isinstance(raw_runs, (str, bytes)) or not raw_runs:
        raise LaunchGuardError(f"{label}.run_measurements must be a nonempty list")
    expected_count = (
        len(policy.required_law_families)
        * len(policy.required_proxy_accuracies)
        * policy.independent_seed_count
    )
    if len(raw_runs) != expected_count or candidate.get("n_runs") != expected_count:
        raise LaunchGuardError(f"{label} must contain exactly {expected_count} runs")

    seen_run_ids: set[str] = set()
    seen_cells: set[tuple[str, float, int]] = set()
    run_results: list[dict[str, Any]] = []
    for run_index, raw_run in enumerate(raw_runs):
        run = _mapping(raw_run, f"{label}.run_measurements[{run_index}]")
        run_id = str(run.get("run_id", ""))
        if not run_id or run_id in seen_run_ids:
            raise LaunchGuardError(f"{label} needs unique nonempty run IDs")
        seen_run_ids.add(run_id)
        law_family = str(run.get("law_family", ""))
        if law_family not in policy.required_law_families:
            raise LaunchGuardError(f"{label} has an unrecognized Law family")
        raw_q_p = _bounded_probability(run.get("q_p"), f"{label}.{run_id}.q_p")
        matching_q_p = [
            expected
            for expected in policy.required_proxy_accuracies
            if math.isclose(raw_q_p, expected, rel_tol=0.0, abs_tol=1e-12)
        ]
        if len(matching_q_p) != 1:
            raise LaunchGuardError(f"{label} has a proxy accuracy outside the frozen panel")
        q_p = matching_q_p[0]
        seed = _nonnegative_integer(run.get("seed"), f"{label}.{run_id}.seed")
        cell = (law_family, q_p, seed)
        if cell in seen_cells:
            raise LaunchGuardError(f"{label} contains a duplicate Law/proxy/seed cell")
        seen_cells.add(cell)

        baseline_step = _nonnegative_integer(
            run.get("baseline_step"), f"{label}.{run_id}.baseline_step"
        )
        if baseline_step != policy.baseline_step:
            raise LaunchGuardError(f"{label} does not use the frozen baseline step")
        baseline_accuracy = _bounded_probability(
            run.get("baseline_iid_accuracy"), f"{label}.{run_id}.baseline_iid_accuracy"
        )
        raw_accuracies = _mapping(
            run.get("terminal_iid_accuracy"), f"{label}.{run_id}.terminal_iid_accuracy"
        )
        raw_action_rates = _mapping(
            run.get("terminal_action_b_rate"), f"{label}.{run_id}.terminal_action_b_rate"
        )
        expected_step_keys = {str(step) for step in policy.terminal_steps}
        if set(raw_accuracies) != expected_step_keys or set(raw_action_rates) != expected_step_keys:
            raise LaunchGuardError(f"{label} lacks an exact terminal-step IID panel")
        accuracies = {
            step: _bounded_probability(
                raw_accuracies[str(step)], f"{label}.{run_id}.iid_accuracy.step_{step}"
            )
            for step in policy.terminal_steps
        }
        action_rates = {
            step: _bounded_probability(
                raw_action_rates[str(step)], f"{label}.{run_id}.action_b_rate.step_{step}"
            )
            for step in policy.terminal_steps
        }
        minimum_terminal = min(accuracies.values())
        terminal_drop = max(
            0.0,
            *(accuracies[left] - accuracies[right]
              for left, right in zip(policy.terminal_steps, policy.terminal_steps[1:], strict=False)),
        )
        maximum_action_frequency = max(
            max(rate, 1.0 - rate) for rate in action_rates.values()
        )
        low_baseline = baseline_accuracy < thresholds.minimum_terminal_iid_accuracy
        low_improvement = minimum_terminal - baseline_accuracy if low_baseline else None
        high_regression = baseline_accuracy - minimum_terminal if not low_baseline else None

        early_both: float | None = None
        early_zero: float | None = None
        raw_sampling_steps = run.get("early_sampling_steps")
        if algorithm == "outcome_rl":
            if not isinstance(raw_sampling_steps, Sequence) or isinstance(
                raw_sampling_steps, (str, bytes)
            ):
                raise LaunchGuardError(f"{label}.{run_id}.early_sampling_steps must be a list")
            sampling_steps = tuple(
                _nonnegative_integer(step, f"{label}.{run_id}.early_sampling_steps")
                for step in raw_sampling_steps
            )
            if sampling_steps != policy.rl_sampling_steps:
                raise LaunchGuardError(f"{label} lacks exact RL updates 1--32")
            early_both = _bounded_probability(
                run.get("early_both_actions_sampled_median"),
                f"{label}.{run_id}.early_both_actions_sampled_median",
            )
            early_zero = _bounded_probability(
                run.get("early_all_zero_advantages_median"),
                f"{label}.{run_id}.early_all_zero_advantages_median",
            )
        elif (
            raw_sampling_steps not in (None, [])
            or run.get("early_both_actions_sampled_median") is not None
            or run.get("early_all_zero_advantages_median") is not None
        ):
            raise LaunchGuardError(f"{label} has RL sampling rows for a non-RL algorithm")

        tolerance = 1e-12
        learning_passed = (
            minimum_terminal + tolerance >= thresholds.minimum_terminal_iid_accuracy
            and (
                low_improvement is None
                or low_improvement + tolerance >= thresholds.minimum_iid_improvement
            )
            and (
                high_regression is None
                or high_regression
                <= thresholds.maximum_high_baseline_regression + tolerance
            )
        )
        stability_passed = (
            terminal_drop <= thresholds.maximum_terminal_accuracy_drop + tolerance
            and maximum_action_frequency
            <= thresholds.maximum_terminal_action_frequency + tolerance
        )
        sampling_passed = (
            algorithm != "outcome_rl"
            or (
                early_both is not None
                and early_zero is not None
                and early_both + tolerance
                >= thresholds.minimum_rl_early_both_actions_fraction
                and early_zero
                <= thresholds.maximum_rl_early_all_zero_advantages_fraction + tolerance
            )
        )
        run_results.append(
            {
                "run_id": run_id,
                "law_family": law_family,
                "q_p": q_p,
                "seed": seed,
                "baseline_step": baseline_step,
                "baseline_iid_accuracy": baseline_accuracy,
                "terminal_iid_accuracy": {str(step): accuracies[step] for step in steps},
                "terminal_action_b_rate": {str(step): action_rates[step] for step in steps},
                "minimum_terminal_iid_accuracy": minimum_terminal,
                "terminal_accuracy_drop": terminal_drop,
                "maximum_terminal_action_frequency": maximum_action_frequency,
                "low_baseline_iid_improvement": low_improvement,
                "high_baseline_iid_regression": high_regression,
                "early_sampling_steps": (
                    list(policy.rl_sampling_steps) if algorithm == "outcome_rl" else None
                ),
                "early_both_actions_sampled_median": early_both,
                "early_all_zero_advantages_median": early_zero,
                "learning_passed": learning_passed,
                "stability_passed": stability_passed,
                "sampling_passed": sampling_passed,
                "passed": learning_passed and stability_passed and sampling_passed,
            }
        )

    seeds = sorted({seed for _, _, seed in seen_cells})
    expected_cells = {
        (family, q_p, seed)
        for family in policy.required_law_families
        for q_p in policy.required_proxy_accuracies
        for seed in seeds
    }
    if len(seeds) != policy.independent_seed_count or seen_cells != expected_cells:
        raise LaunchGuardError(f"{label} is not the exact Law x proxy-accuracy x seed panel")

    model_identity = dict(_mapping(candidate.get("model_identity"), f"{label}.model_identity"))
    update_method = str(candidate.get("update_method", ""))
    effective_batch = _nonnegative_integer(candidate.get("effective_batch"), f"{label}.effective_batch")
    all_runs_passed = all(run["passed"] for run in run_results)
    return {
        "learning_rate": learning_rate,
        "entropy_coefficient": entropy,
        "stability_window_steps": list(policy.terminal_steps),
        "run_measurements": run_results,
        "minimum_terminal_iid_accuracy": min(
            run["minimum_terminal_iid_accuracy"] for run in run_results
        ),
        "minimum_low_baseline_iid_improvement": min(
            (run["low_baseline_iid_improvement"] for run in run_results
             if run["low_baseline_iid_improvement"] is not None),
            default=None,
        ),
        "maximum_high_baseline_iid_regression": max(
            (run["high_baseline_iid_regression"] for run in run_results
             if run["high_baseline_iid_regression"] is not None),
            default=None,
        ),
        "maximum_terminal_accuracy_drop": max(
            run["terminal_accuracy_drop"] for run in run_results
        ),
        "maximum_terminal_action_frequency": max(
            run["maximum_terminal_action_frequency"] for run in run_results
        ),
        "minimum_early_both_actions_sampled_fraction": (
            min(run["early_both_actions_sampled_median"] for run in run_results)
            if algorithm == "outcome_rl"
            else None
        ),
        "maximum_early_all_zero_advantages_fraction": (
            max(run["early_all_zero_advantages_median"] for run in run_results)
            if algorithm == "outcome_rl"
            else None
        ),
        "model_identity": model_identity,
        "update_method": update_method,
        "effective_batch": effective_batch,
        "law_families": list(policy.required_law_families),
        "proxy_accuracies": list(policy.required_proxy_accuracies),
        "seeds": seeds,
        "n_runs": len(run_results),
        "passed": all_runs_passed and update_method == "full" and effective_batch > 0,
    }


def evaluate_g00_gate(
    assessment: Mapping[str, Any],
    *,
    thresholds: G00GateThresholds = G00_GATE_THRESHOLDS,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Evaluate all six G00 gates from raw, auditable measurements."""

    checks = _evaluate_capability_checks(assessment, thresholds)
    by_model = assessment.get("capability_by_model")
    if by_model is not None:
        panels = _mapping(by_model, "capability_by_model")
        if not panels:
            raise LaunchGuardError("capability_by_model cannot be empty")
        model_checks: dict[str, dict[str, Any]] = {}
        for model_digest, raw_panel in sorted(panels.items()):
            panel = _mapping(raw_panel, f"capability_by_model.{model_digest}")
            identity = _mapping(
                panel.get("model_identity"), f"capability_by_model.{model_digest}.model_identity"
            )
            if stable_hash(identity, 64) != str(model_digest):
                raise LaunchGuardError("capability model identity digest mismatch")
            panel_checks = _evaluate_capability_checks(
                panel,
                thresholds,
                prefix=f"capability_by_model.{model_digest}",
            )
            model_checks[str(model_digest)] = {
                "model_identity": dict(identity),
                "checks": panel_checks,
                "passed": all(check["passed"] for check in panel_checks.values()),
            }
        for check_name in (
            "rule_adapters",
            "constrained_scorer",
            "no_signal_chance",
            "surface_leakage",
        ):
            checks[check_name]["capability_by_model"] = {
                digest: details["checks"][check_name] for digest, details in model_checks.items()
            }
            checks[check_name]["passed"] = bool(
                checks[check_name]["passed"]
                and all(details["checks"][check_name]["passed"] for details in model_checks.values())
            )

    optimizer = _mapping(assessment.get("optimizer_stability"), "optimizer stability")
    selected_settings: dict[str, dict[str, Any]] = {}
    optimizer_results: dict[str, Any] = {}
    policy = G00_OPTIMIZER_STABILITY_POLICY
    for algorithm in ("sft", "outcome_rl"):
        raw_candidates = optimizer.get(algorithm)
        if (
            not isinstance(raw_candidates, Sequence)
            or isinstance(raw_candidates, (str, bytes))
            or not raw_candidates
        ):
            raise LaunchGuardError(f"optimizer stability.{algorithm} must be a nonempty list")
        candidates = [
            _evaluate_optimizer_candidate(
                _mapping(raw_candidate, f"optimizer stability.{algorithm}[{candidate_index}]"),
                algorithm=algorithm,
                candidate_index=candidate_index,
                thresholds=thresholds,
                policy=policy,
            )
            for candidate_index, raw_candidate in enumerate(raw_candidates)
        ]
        setting_keys = [
            (float(candidate["learning_rate"]), float(candidate["entropy_coefficient"]))
            for candidate in candidates
        ]
        if len(setting_keys) != len(set(setting_keys)):
            raise LaunchGuardError(f"optimizer stability.{algorithm} has duplicate settings")
        passing = [candidate for candidate in candidates if candidate["passed"]]
        if not passing:
            raise LaunchGuardError(f"optimizer stability.{algorithm} has no passing setting")
        selected = min(
            passing,
            key=lambda candidate: (
                float(candidate["learning_rate"]),
                float(candidate["entropy_coefficient"]),
            ),
        )
        for candidate in candidates:
            candidate["selected"] = candidate is selected
        selected_settings[algorithm] = {
            "learning_rate": float(selected["learning_rate"]),
            "entropy_coefficient": float(selected["entropy_coefficient"]),
            "model_identity": dict(selected["model_identity"]),
            "update_method": str(selected["update_method"]),
            "effective_batch": int(selected["effective_batch"]),
            "law_families": list(selected["law_families"]),
            "proxy_accuracies": list(selected["proxy_accuracies"]),
            "seeds": list(selected["seeds"]),
            "stability_window_steps": list(selected["stability_window_steps"]),
        }
        optimizer_results[algorithm] = candidates
    checks["optimizer_stability"] = {
        "policy": policy.as_dict(),
        "policy_digest": stable_hash(policy.as_dict(), 64),
        "minimum_iid_improvement": thresholds.minimum_iid_improvement,
        "minimum_terminal_iid_accuracy": thresholds.minimum_terminal_iid_accuracy,
        "maximum_high_baseline_regression": thresholds.maximum_high_baseline_regression,
        "maximum_terminal_accuracy_drop": thresholds.maximum_terminal_accuracy_drop,
        "maximum_terminal_action_frequency": thresholds.maximum_terminal_action_frequency,
        "minimum_rl_early_both_actions_fraction": (
            thresholds.minimum_rl_early_both_actions_fraction
        ),
        "maximum_rl_early_all_zero_advantages_fraction": (
            thresholds.maximum_rl_early_all_zero_advantages_fraction
        ),
        "candidates": optimizer_results,
        "selected": selected_settings,
        "passed": True,
    }

    integrity = _mapping(assessment.get("dataset_integrity"), "dataset integrity")
    integrity_fields = (
        "truth_cell_count_mismatches",
        "scene_overlap_count",
        "duplicate_pair_ids",
        "regeneration_mismatches",
    )
    integrity_counts = {
        name: _nonnegative_integer(integrity.get(name), f"dataset integrity.{name}")
        for name in integrity_fields
    }
    checks["dataset_integrity"] = {
        **integrity_counts,
        "allowed_mismatches": thresholds.maximum_integrity_mismatches,
        "passed": all(
            value <= thresholds.maximum_integrity_mismatches for value in integrity_counts.values()
        ),
    }
    if set(checks) != {
        "rule_adapters",
        "constrained_scorer",
        "no_signal_chance",
        "surface_leakage",
        "optimizer_stability",
        "dataset_integrity",
    }:
        raise AssertionError("internal G00 gate check set changed")
    return checks, selected_settings


def _gate_scientific_config(config: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(config))
    for key in tuple(value):
        if str(key).startswith("_"):
            value.pop(key, None)
    run = dict(value.get("run", {}))
    for name in (
        "output_root",
        "resume",
        "launch_guard",
        "protocol_unlocked",
        "gate_artifact",
    ):
        run.pop(name, None)
    value["run"] = run
    return canonical_config(value)


def _semantic_map_digest(config: Mapping[str, Any]) -> str:
    data = get_path(config, "data", {})
    if not isinstance(data, Mapping):
        raise LaunchGuardError("data config must be an object")
    fields = (
        "feature_count",
        "feature_names",
        "law_features",
        "law_expected_values",
        "law_output_negated",
        "rule_family",
        "sage_features",
        "sage_expected_values",
        "sage_output_negated",
        "sage_rule_family",
        "distractor_features",
        "renderer",
    )
    return stable_hash({field: data.get(field) for field in fields}, 64)


def target_gate_binding(config: Mapping[str, Any], repo: str | Path) -> dict[str, Any]:
    scientific = _gate_scientific_config(config)
    experiment_id = str(get_path(config, "experiment.id", ""))
    if experiment_id != "g01":
        raise LaunchGuardError(
            "G00 gates may authorize only G01; G02 requires a separate G01 selection artifact"
        )
    return {
        "experiment_id": experiment_id,
        "scientific_config_digest": stable_hash(scientific, 64),
        "authorized_cell_config_digests": sorted(
            {stable_hash(_gate_scientific_config(cell), 64) for cell in expand_sweep(config)}
        ),
        "model_spec_digest": stable_hash(
            {"model": scientific.get("model", {}), "update": scientific.get("update", {})},
            64,
        ),
        "model_identity": {
            "requested_model": str(get_path(config, "model.name", "")),
            "requested_revision": str(get_path(config, "model.revision", "")),
        },
        "semantic_map_digest": _semantic_map_digest(config),
        "source_fingerprint": implementation_provenance(repo)["implementation_fingerprint"],
        "planned_run_count": len(build_plan(config)),
    }


def g00_evidence_binding(
    artifact_roots: Sequence[str | Path],
    configs: Sequence[Mapping[str, Any]],
    repo: str | Path,
) -> dict[str, Any]:
    expected = {spec.plan_key: spec for config in configs for spec in build_plan(config)}
    if not expected:
        raise LaunchGuardError("G00 gate needs at least one planned engineering run")
    observed: dict[str, Path] = {}
    records: list[dict[str, Any]] = []
    for root in artifact_roots:
        for path in discover_runs(root, completed_only=True):
            summary_path = path / "summary.json"
            if not summary_path.is_file():
                continue
            summary = read_json(summary_path)
            plan_key = str(summary.get("plan_key", ""))
            if plan_key not in expected:
                continue
            if plan_key in observed:
                raise LaunchGuardError(f"duplicate completed G00 evidence for plan key {plan_key}")
            identity = read_json(path / "identity.json")
            if stable_hash(identity, 20) != path.name:
                raise LaunchGuardError(f"G00 evidence identity hash mismatch: {path}")
            if int(identity.get("artifact_schema_version", -1)) != ARTIFACT_SCHEMA_VERSION:
                raise LaunchGuardError(f"G00 evidence uses an obsolete artifact schema: {path}")
            try:
                completion = verify_completion_attestation(path)
            except ArtifactError as exc:
                raise LaunchGuardError(f"G00 evidence completion seal failed: {path}") from exc
            loaded = yaml.safe_load((path / "resolved_config.yaml").read_text(encoding="utf-8"))
            if not isinstance(loaded, Mapping):
                raise LaunchGuardError(f"G00 resolved config is malformed: {path}")
            resolved = dict(loaded)
            seed = int(resolved.pop("seed"))
            if seed != expected[plan_key].seed or identity.get("config") != _identity_config(resolved):
                raise LaunchGuardError(f"G00 evidence config identity mismatch: {path}")
            manifests: dict[str, str] = {}
            manifest_metadata: dict[str, Mapping[str, Any]] = {}
            for kind in ("dataset", "model", "tokenizer"):
                manifest = read_json(path / "manifests" / f"{kind}.json")
                if manifest.get("kind") != kind or not isinstance(manifest.get("metadata"), Mapping):
                    raise LaunchGuardError(f"malformed G00 {kind} manifest: {path}")
                if manifest.get("digest") != stable_hash(manifest["metadata"], 64):
                    raise LaunchGuardError(f"G00 {kind} manifest digest mismatch: {path}")
                manifests[kind] = str(manifest["digest"])
                manifest_metadata[kind] = manifest["metadata"]
            model_metadata = manifest_metadata["model"]
            model_identity = {
                "requested_model": str(model_metadata.get("requested_model", model_metadata.get("name", ""))),
                "requested_revision": str(
                    model_metadata.get(
                        "requested_revision",
                        model_metadata.get("resolved_revision", model_metadata.get("revision", "")),
                    )
                ),
            }
            observed[plan_key] = path
            records.append(
                {
                    "plan_key": plan_key,
                    "run_id": path.name,
                    "seed": seed,
                    "implementation_fingerprint": identity.get("implementation_fingerprint"),
                    "completion_attestation_digest": completion.get("completion_digest"),
                    "manifests": manifests,
                    "model_identity": model_identity,
                    "semantic_map_digest": _semantic_map_digest(resolved),
                    "training_view": str(get_path(resolved, "data.training_view", "full")),
                    "law_family": str(get_path(resolved, "data.rule_family", "")),
                    "q_p": float(get_path(resolved, "data.q_p", 0.0)),
                    "experiment_name": str(get_path(resolved, "experiment.name", "")),
                    "algorithm": str(get_path(resolved, "train.algorithm", "")),
                    "learning_rate": float(get_path(resolved, "train.learning_rate", 0.0)),
                    "entropy_coefficient": float(
                        get_path(resolved, "train.entropy_coefficient", 0.0)
                    ),
                    "update_method": str(get_path(resolved, "update.method", "")),
                    "effective_batch": int(get_path(resolved, "train.batch_size", 0))
                    * int(get_path(resolved, "train.gradient_accumulation_steps", 0)),
                }
            )
    missing = sorted(set(expected) - set(observed))
    if missing:
        raise LaunchGuardError(f"G00 evidence panel is incomplete: {len(missing)} planned runs missing")
    source_fingerprints = {str(record["implementation_fingerprint"]) for record in records}
    if len(source_fingerprints) != 1:
        raise LaunchGuardError("G00 evidence mixes source fingerprints")
    current_source = str(implementation_provenance(repo)["implementation_fingerprint"])
    if source_fingerprints != {current_source}:
        raise LaunchGuardError("G00 evidence source fingerprint differs from the launch source")
    records.sort(key=lambda record: str(record["plan_key"]))
    capability_views = {
        "law_only",
        "audit_law_matched",
        "sage_only",
        "herald_only",
        "no_signal",
        "surface_only",
    }
    model_identities = {
        stable_hash(record["model_identity"], 64): record["model_identity"] for record in records
    }
    capability_coverage: dict[str, set[str]] = {}
    law_only_coverage: dict[str, list[dict[str, Any]]] = {}
    matched_law_coverage: dict[str, list[dict[str, Any]]] = {}
    engineering_models: dict[str, Any] = {}
    for record in records:
        identity_digest = stable_hash(record["model_identity"], 64)
        view = str(record["training_view"])
        if view in capability_views:
            capability_coverage.setdefault(identity_digest, set()).add(view)
            if view == "law_only":
                law_only_coverage.setdefault(identity_digest, []).append(
                    {
                        "run_id": record["run_id"],
                        "seed": record["seed"],
                        "law_family": record["law_family"],
                    }
                )
            elif view == "audit_law_matched":
                matched_law_coverage.setdefault(identity_digest, []).append(
                    {
                        "run_id": record["run_id"],
                        "seed": record["seed"],
                        "law_family": record["law_family"],
                    }
                )
        else:
            engineering_models[identity_digest] = record["model_identity"]
    binding = {
        "source_fingerprint": current_source,
        "expected_config_digest": stable_hash([_gate_scientific_config(config) for config in configs], 64),
        "planned_run_keys_digest": stable_hash(sorted(expected), 64),
        "completed_run_count": len(records),
        "run_ids": [record["run_id"] for record in records],
        "run_completion_attestations": [
            {
                "run_id": record["run_id"],
                "completion_digest": record["completion_attestation_digest"],
            }
            for record in records
        ],
        "model_manifest_digests": sorted({record["manifests"]["model"] for record in records}),
        "model_identities": model_identities,
        "engineering_model_identities": engineering_models,
        "capability_model_coverage": {
            digest: sorted(views) for digest, views in sorted(capability_coverage.items())
        },
        "law_only_model_coverage": {
            digest: sorted(values, key=lambda value: (value["law_family"], value["seed"], value["run_id"]))
            for digest, values in sorted(law_only_coverage.items())
        },
        "matched_law_model_coverage": {
            digest: sorted(values, key=lambda value: (value["law_family"], value["seed"], value["run_id"]))
            for digest, values in sorted(matched_law_coverage.items())
        },
        "deployed_full_context_pilot_coverage": [
            {
                key: record[key]
                for key in (
                    "run_id",
                    "seed",
                    "model_identity",
                    "law_family",
                    "q_p",
                    "algorithm",
                    "learning_rate",
                    "entropy_coefficient",
                    "update_method",
                    "effective_batch",
                )
            }
            for record in records
            if record["experiment_name"] == "deployed_model_full_context_pilot"
            and record["training_view"] == "full"
        ],
        "semantic_map_digests": sorted({str(record["semantic_map_digest"]) for record in records}),
        "dataset_manifest_digests": sorted({record["manifests"]["dataset"] for record in records}),
        "tokenizer_manifest_digests": sorted({record["manifests"]["tokenizer"] for record in records}),
        "run_binding_digest": stable_hash(records, 64),
    }
    binding["run_completion_attestations_digest"] = stable_hash(
        binding["run_completion_attestations"], 64
    )
    return binding


def _assert_selected_optimizer_matches(
    config: Mapping[str, Any],
    selected: Mapping[str, Mapping[str, Any]],
) -> None:
    tolerance = 1e-15
    for cell in expand_sweep(config):
        algorithm = str(get_path(cell, "train.algorithm", ""))
        if algorithm not in selected:
            raise LaunchGuardError(f"target config uses optimizer {algorithm!r} absent from G00 gate")
        expected = selected[algorithm]
        learning_rate = float(get_path(cell, "train.learning_rate"))
        entropy = float(get_path(cell, "train.entropy_coefficient", 0.0))
        if abs(learning_rate - float(expected["learning_rate"])) > tolerance:
            raise LaunchGuardError(f"target {algorithm} learning rate does not match selected G00 setting")
        if abs(entropy - float(expected["entropy_coefficient"])) > tolerance:
            raise LaunchGuardError(f"target {algorithm} entropy does not match selected G00 setting")
        target_identity = {
            "requested_model": str(get_path(cell, "model.name", "")),
            "requested_revision": str(get_path(cell, "model.revision", "")),
        }
        if expected.get("model_identity") != target_identity:
            raise LaunchGuardError(
                f"selected {algorithm} pilot did not use the exact target model revision"
            )
        if str(expected.get("update_method", "")) != str(get_path(cell, "update.method", "")):
            raise LaunchGuardError(
                f"selected {algorithm} pilot did not use the target update method"
            )
        effective_batch = int(get_path(cell, "train.batch_size", 0)) * int(
            get_path(cell, "train.gradient_accumulation_steps", 0)
        )
        if int(expected.get("effective_batch", -1)) != effective_batch:
            raise LaunchGuardError(
                f"selected {algorithm} pilot did not use the target effective batch"
            )
        if set(expected.get("law_families", [])) != {"parity", "majority"}:
            raise LaunchGuardError(
                f"selected {algorithm} pilot did not cover both Law families"
            )
        if set(float(value) for value in expected.get("proxy_accuracies", [])) != set(
            G00_OPTIMIZER_STABILITY_POLICY.required_proxy_accuracies
        ):
            raise LaunchGuardError(
                f"selected {algorithm} pilot did not cover both frozen proxy accuracies"
            )
        if tuple(int(value) for value in expected.get("stability_window_steps", [])) != (
            G00_OPTIMIZER_STABILITY_POLICY.terminal_steps
        ):
            raise LaunchGuardError(
                f"selected {algorithm} pilot did not use the frozen terminal window"
            )
        pilot_seeds = [int(value) for value in expected.get("seeds", [])]
        if len(set(pilot_seeds)) != 3:
            raise LaunchGuardError(
                f"selected {algorithm} pilot did not cover exactly three independent seeds"
            )


def _validate_g00_target_eligibility(
    *,
    evidence: Mapping[str, Any],
    measurements: Mapping[str, Any],
    selected: Mapping[str, Mapping[str, Any]],
    target_configs: Sequence[Mapping[str, Any]],
    repo: str | Path,
) -> dict[str, Any]:
    """Apply the target-specific checks identically at creation and launch."""

    targets: dict[str, Any] = {}
    for config in target_configs:
        _assert_selected_optimizer_matches(config, selected)
        binding = target_gate_binding(config, repo)
        experiment_id = str(binding["experiment_id"])
        if experiment_id in targets:
            raise LaunchGuardError(f"duplicate target config for {experiment_id}")
        targets[experiment_id] = binding
    if not targets:
        raise LaunchGuardError("G00 gate must bind exactly one or more G01 target configs")

    required_capability_views = {
        "law_only",
        "audit_law_matched",
        "sage_only",
        "herald_only",
        "no_signal",
        "surface_only",
    }
    capability_measurements = measurements.get("capability_by_model")
    if not isinstance(capability_measurements, Mapping):
        raise LaunchGuardError("artifact-derived G00 assessment lacks per-model capability checks")
    engineering = _mapping(
        evidence.get("engineering_model_identities"),
        "G00 engineering model identities",
    )
    required_capability_models = set(engineering)
    for experiment_id, binding in targets.items():
        model_digest = stable_hash(binding["model_identity"], 64)
        required_capability_models.add(model_digest)
        pilot_coverage = evidence.get("deployed_full_context_pilot_coverage")
        if not isinstance(pilot_coverage, Sequence) or isinstance(
            pilot_coverage, (str, bytes)
        ):
            raise LaunchGuardError("G00 evidence lacks deployed-model pilot coverage")
        for algorithm, setting in selected.items():
            matching_pilots = [
                record
                for record in pilot_coverage
                if isinstance(record, Mapping)
                and record.get("model_identity") == binding["model_identity"]
                and str(record.get("algorithm", "")) == algorithm
                and abs(
                    float(record.get("learning_rate", -1.0))
                    - float(setting["learning_rate"])
                )
                <= 1e-15
                and abs(
                    float(record.get("entropy_coefficient", -1.0))
                    - float(setting["entropy_coefficient"])
                )
                <= 1e-15
                and str(record.get("update_method", "")) == str(setting["update_method"])
                and int(record.get("effective_batch", -1)) == int(setting["effective_batch"])
            ]
            pilot_cells = {
                (
                    str(record.get("law_family", "")),
                    float(record.get("q_p", -1.0)),
                    int(record.get("seed", -1)),
                )
                for record in matching_pilots
            }
            required_cells = {
                (family, q_p, seed)
                for family in ("parity", "majority")
                for q_p in G00_OPTIMIZER_STABILITY_POLICY.required_proxy_accuracies
                for seed in setting["seeds"]
            }
            if pilot_cells != required_cells or len(set(setting["seeds"])) != 3:
                raise LaunchGuardError(
                    f"selected {algorithm} setting lacks exact-model full-context pilot "
                    "evidence for both Law families, proxy accuracies, and all three seeds"
                )

        capability_coverage = _mapping(
            evidence.get("capability_model_coverage"),
            "G00 capability model coverage",
        )
        coverage = set(capability_coverage.get(model_digest, []))
        if coverage != required_capability_views:
            raise LaunchGuardError(
                f"G00 capability evidence for {experiment_id} does not cover all six views "
                "at the target model commit"
            )
        model_measurements = capability_measurements.get(model_digest)
        if not isinstance(model_measurements, Mapping):
            raise LaunchGuardError("target-model capability measurements are absent")
        for coverage_key, measurement_key, label in (
            (
                "law_only_model_coverage",
                "law_only_position_agreement_by_run",
                "law-only",
            ),
            (
                "matched_law_model_coverage",
                "matched_law_position_agreement_by_run",
                "matched-layout Law",
            ),
        ):
            coverage_by_model = _mapping(
                evidence.get(coverage_key), f"G00 evidence.{coverage_key}"
            )
            expected_runs = coverage_by_model.get(model_digest, [])
            if not isinstance(expected_runs, Sequence) or isinstance(
                expected_runs, (str, bytes)
            ):
                raise LaunchGuardError(f"G00 {label} coverage is malformed")
            law_families = {
                str(record.get("law_family", ""))
                for record in expected_runs
                if isinstance(record, Mapping)
            }
            seeds_by_family = {
                family: {
                    int(record["seed"])
                    for record in expected_runs
                    if isinstance(record, Mapping)
                    and str(record.get("law_family", "")) == family
                }
                for family in ("parity", "majority")
            }
            if law_families != {"parity", "majority"} or any(
                len(seeds) != 3 for seeds in seeds_by_family.values()
            ):
                raise LaunchGuardError(
                    f"G00 {label} evidence for {experiment_id} must test both Law families "
                    "at all three target-model seeds"
                )
            measured_runs = model_measurements.get(measurement_key)
            if not isinstance(measured_runs, Sequence) or isinstance(
                measured_runs, (str, bytes)
            ):
                raise LaunchGuardError(f"target-model {label} per-run measurements are absent")
            expected_run_keys = {
                (str(record["run_id"]), int(record["seed"]), str(record["law_family"]))
                for record in expected_runs
                if isinstance(record, Mapping)
            }
            measured_run_keys = {
                (
                    str(record.get("run_id", "")),
                    int(record.get("seed", -1)),
                    str(record.get("law_family", "")),
                )
                for record in measured_runs
                if isinstance(record, Mapping)
            }
            if measured_run_keys != expected_run_keys:
                raise LaunchGuardError(
                    f"target-model {label} measurements do not cover every planned seed and Law family"
                )
        if not any(digest != model_digest for digest in engineering):
            raise LaunchGuardError(
                "G00 evidence must include a distinct engineering-model calibration panel"
            )
        semantic_map_digests = evidence.get("semantic_map_digests")
        if not isinstance(semantic_map_digests, Sequence) or isinstance(
            semantic_map_digests, (str, bytes)
        ):
            raise LaunchGuardError("G00 semantic-map evidence is malformed")
        if binding["semantic_map_digest"] in semantic_map_digests:
            raise LaunchGuardError(
                f"G00 and {experiment_id} use the same semantic map; confirmatory mapping is contaminated"
            )

    capability_coverage = _mapping(
        evidence.get("capability_model_coverage"),
        "G00 capability model coverage",
    )
    for model_digest in sorted(required_capability_models):
        coverage = set(capability_coverage.get(model_digest, []))
        if coverage != required_capability_views:
            raise LaunchGuardError(
                "every engineering and target model commit needs all six G00 capability views"
            )
        if model_digest not in capability_measurements:
            raise LaunchGuardError(
                "artifact-derived G00 assessment omits a required model-specific capability panel"
            )
    return targets


def create_g00_gate_artifact(
    assessment: Mapping[str, Any],
    *,
    artifact_roots: Sequence[str | Path],
    g00_configs: Sequence[Mapping[str, Any]],
    target_configs: Sequence[Mapping[str, Any]],
    repo: str | Path,
    output: str | Path,
) -> G00GateArtifact:
    """Create an auditable pass/fail artifact; never mutate a target config."""

    evidence = g00_evidence_binding(artifact_roots, g00_configs, repo)
    assessment_body = {key: value for key, value in assessment.items() if key != "assessment_digest"}
    policy = G00_OPTIMIZER_STABILITY_POLICY.as_dict()
    if (
        assessment.get("schema") != "goalzendo.g00_assessment"
        or assessment.get("schema_version") != 2
        or assessment.get("assessment_digest") != stable_hash(assessment_body, 64)
        or assessment.get("evidence_run_binding_digest") != evidence["run_binding_digest"]
        or assessment.get("evidence_source_fingerprint") != evidence["source_fingerprint"]
        or assessment.get("evidence_config_digest") != evidence["expected_config_digest"]
        or assessment.get("optimizer_stability_policy") != policy
        or assessment.get("optimizer_stability_policy_digest") != stable_hash(policy, 64)
        or not isinstance(assessment.get("measurements"), Mapping)
        or get_path(assessment, "derivation.source", "") != "completed_metrics_predictions_and_manifests"
    ):
        raise LaunchGuardError(
            "G00 assessment is not an artifact-derived, digest-bound assessment for these runs"
        )
    # Re-derive from the immutable evidence rather than trusting a caller that
    # merely recomputed the outer assessment digest after editing measurements.
    from .analysis import derive_g00_gate_assessment

    independently_derived = derive_g00_gate_assessment(
        artifact_roots,
        g00_configs,
        repo=repo,
    )
    if dict(assessment) != independently_derived:
        raise LaunchGuardError("G00 assessment does not exactly match the bound run artifacts")
    checks, selected = evaluate_g00_gate(assessment["measurements"])
    targets = _validate_g00_target_eligibility(
        evidence=evidence,
        measurements=assessment["measurements"],
        selected=selected,
        target_configs=target_configs,
        repo=repo,
    )
    thresholds = G00_GATE_THRESHOLDS.as_dict()
    body = {
        "schema": G00_GATE_SCHEMA,
        "schema_version": G00_GATE_SCHEMA_VERSION,
        "created_at": utc_now(),
        "thresholds": thresholds,
        "threshold_digest": stable_hash(thresholds, 64),
        "optimizer_stability_policy": policy,
        "optimizer_stability_policy_digest": stable_hash(policy, 64),
        "verification_inputs": {
            "artifact_roots": [str(Path(root).resolve()) for root in artifact_roots],
            "g00_configs": [canonical_config(config) for config in g00_configs],
            "target_configs": [canonical_config(config) for config in target_configs],
        },
        "g00_evidence": evidence,
        "assessment": dict(assessment),
        "checks": checks,
        "selected_optimizer_settings": selected,
        "authorized_targets": targets,
        "overall_passed": all(check.get("passed") is True for check in checks.values()),
    }
    body["verification_inputs_digest"] = stable_hash(body["verification_inputs"], 64)
    payload = {**body, "gate_digest": stable_hash(body, 64)}
    target = Path(output).resolve()
    write_json(target, payload)
    return G00GateArtifact(payload, target)


def verify_g00_gate_artifact(
    path: str | Path,
    *,
    config: Mapping[str, Any],
    repo: str | Path,
) -> G00GateArtifact:
    target = Path(path).resolve()
    if not target.is_file():
        raise LaunchGuardError(f"G00 gate artifact does not exist: {target}")
    payload = read_json(target)
    body = {key: value for key, value in payload.items() if key != "gate_digest"}
    if payload.get("schema") != G00_GATE_SCHEMA or payload.get("schema_version") != G00_GATE_SCHEMA_VERSION:
        raise LaunchGuardError("G00 gate artifact schema is not recognized")
    if payload.get("gate_digest") != stable_hash(body, 64):
        raise LaunchGuardError("G00 gate artifact digest mismatch")
    thresholds = G00_GATE_THRESHOLDS.as_dict()
    policy = G00_OPTIMIZER_STABILITY_POLICY.as_dict()
    if payload.get("thresholds") != thresholds or payload.get("threshold_digest") != stable_hash(
        thresholds, 64
    ):
        raise LaunchGuardError("G00 gate thresholds differ from the frozen schema")
    if payload.get("optimizer_stability_policy") != policy or payload.get(
        "optimizer_stability_policy_digest"
    ) != stable_hash(policy, 64):
        raise LaunchGuardError("G00 optimizer stability policy differs from the frozen schema")
    evidence = _mapping(payload.get("g00_evidence"), "G00 gate evidence")
    completed_run_count = _nonnegative_integer(
        evidence.get("completed_run_count"), "G00 gate evidence.completed_run_count"
    )
    run_ids_raw = evidence.get("run_ids")
    attestations_raw = evidence.get("run_completion_attestations")
    if (
        completed_run_count == 0
        or not isinstance(run_ids_raw, Sequence)
        or isinstance(run_ids_raw, (str, bytes))
        or len(run_ids_raw) != completed_run_count
        or len({str(value) for value in run_ids_raw}) != completed_run_count
        or not isinstance(attestations_raw, Sequence)
        or isinstance(attestations_raw, (str, bytes))
        or len(attestations_raw) != completed_run_count
    ):
        raise LaunchGuardError("G00 evidence lacks a complete unique run-attestation panel")
    attestation_rows = [
        dict(_mapping(value, "G00 gate evidence.run_completion_attestations"))
        for value in attestations_raw
    ]
    if (
        [str(row.get("run_id", "")) for row in attestation_rows]
        != [str(value) for value in run_ids_raw]
        or any(
            len(str(row.get("completion_digest", ""))) != 64
            or any(character not in "0123456789abcdef" for character in str(row["completion_digest"]))
            for row in attestation_rows
        )
        or evidence.get("run_completion_attestations_digest")
        != stable_hash(attestation_rows, 64)
    ):
        raise LaunchGuardError("G00 evidence run-completion attestations are malformed")
    for digest_field in (
        "source_fingerprint",
        "expected_config_digest",
        "planned_run_keys_digest",
        "run_binding_digest",
    ):
        value = str(evidence.get(digest_field, ""))
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise LaunchGuardError(f"G00 evidence has an invalid {digest_field}")

    assessment = _mapping(payload.get("assessment"), "G00 gate assessment")
    assessment_body = {key: value for key, value in assessment.items() if key != "assessment_digest"}
    if (
        assessment.get("schema") != "goalzendo.g00_assessment"
        or assessment.get("schema_version") != 2
        or assessment.get("assessment_digest") != stable_hash(assessment_body, 64)
        or assessment.get("evidence_run_binding_digest") != evidence["run_binding_digest"]
        or assessment.get("evidence_source_fingerprint") != evidence["source_fingerprint"]
        or assessment.get("evidence_config_digest") != evidence["expected_config_digest"]
        or assessment.get("optimizer_stability_policy") != policy
        or assessment.get("optimizer_stability_policy_digest") != stable_hash(policy, 64)
        or not isinstance(assessment.get("measurements"), Mapping)
        or get_path(assessment, "derivation.source", "")
        != "completed_metrics_predictions_and_manifests"
    ):
        raise LaunchGuardError("G00 gate assessment is absent, malformed, or not evidence-bound")
    verification_inputs = _mapping(
        payload.get("verification_inputs"), "G00 gate verification inputs"
    )
    if payload.get("verification_inputs_digest") != stable_hash(verification_inputs, 64):
        raise LaunchGuardError("G00 gate verification-input digest mismatch")
    roots_raw = verification_inputs.get("artifact_roots")
    configs_raw = verification_inputs.get("g00_configs")
    target_configs_raw = verification_inputs.get("target_configs")
    if (
        not isinstance(roots_raw, Sequence)
        or isinstance(roots_raw, (str, bytes))
        or not roots_raw
        or not isinstance(configs_raw, Sequence)
        or isinstance(configs_raw, (str, bytes))
        or not configs_raw
        or any(not isinstance(value, Mapping) for value in configs_raw)
        or not isinstance(target_configs_raw, Sequence)
        or isinstance(target_configs_raw, (str, bytes))
        or not target_configs_raw
        or any(not isinstance(value, Mapping) for value in target_configs_raw)
    ):
        raise LaunchGuardError(
            "G00 gate lacks replayable evidence roots, G00 configs, or target configs"
        )
    roots = [Path(str(value)).resolve() for value in roots_raw]
    if any(not root.is_dir() for root in roots):
        raise LaunchGuardError("G00 gate evidence root is missing at launch verification")
    configs = [dict(value) for value in configs_raw if isinstance(value, Mapping)]
    target_configs = [
        dict(value) for value in target_configs_raw if isinstance(value, Mapping)
    ]
    source_fingerprint = implementation_provenance(repo)["implementation_fingerprint"]
    cache_key = (str(target), str(payload["gate_digest"]), str(source_fingerprint))
    cached_replay = _G00_EVIDENCE_VERIFICATION_CACHE.get(cache_key)
    if cached_replay is None:
        from .analysis import derive_g00_gate_assessment

        independently_derived = derive_g00_gate_assessment(roots, configs, repo=repo)
        independently_bound_evidence = g00_evidence_binding(roots, configs, repo)
        cached_replay = {
            "assessment": independently_derived,
            "evidence": independently_bound_evidence,
        }
        _G00_EVIDENCE_VERIFICATION_CACHE[cache_key] = cached_replay
    replayed_assessment = _mapping(
        cached_replay.get("assessment"), "cached G00 replay assessment"
    )
    replayed_evidence = _mapping(
        cached_replay.get("evidence"), "cached G00 replay evidence"
    )
    if dict(assessment) != replayed_assessment:
        raise LaunchGuardError("G00 gate assessment does not replay from its completed evidence")
    if dict(evidence) != replayed_evidence:
        raise LaunchGuardError("G00 gate evidence binding does not replay from completed runs")
    recomputed_checks, recomputed_selected = evaluate_g00_gate(assessment["measurements"])
    if payload.get("checks") != recomputed_checks:
        raise LaunchGuardError("G00 gate checks do not exactly match the embedded assessment")
    if payload.get("selected_optimizer_settings") != recomputed_selected:
        raise LaunchGuardError("G00 gate selected settings do not exactly match automatic selection")
    replayed_targets = _validate_g00_target_eligibility(
        evidence=evidence,
        measurements=assessment["measurements"],
        selected=recomputed_selected,
        target_configs=target_configs,
        repo=repo,
    )
    if payload.get("authorized_targets") != replayed_targets:
        raise LaunchGuardError("G00 gate target eligibility does not replay from bound evidence")
    _assert_selected_optimizer_matches(config, recomputed_selected)

    checks = recomputed_checks
    required_checks = {
        "rule_adapters",
        "constrained_scorer",
        "no_signal_chance",
        "surface_leakage",
        "optimizer_stability",
        "dataset_integrity",
    }
    if (
        payload.get("overall_passed") is not True
        or not isinstance(checks, Mapping)
        or set(checks) != required_checks
        or any(
            not isinstance(checks[name], Mapping) or checks[name].get("passed") is not True
            for name in required_checks
        )
    ):
        raise LaunchGuardError("G00 gate artifact has not passed all six checks")
    expected_binding = target_gate_binding(config, repo)
    targets = payload.get("authorized_targets")
    experiment_id = str(expected_binding["experiment_id"])
    bound_target = targets.get(experiment_id) if isinstance(targets, Mapping) else None
    current_cell_digest = stable_hash(_gate_scientific_config(config), 64)
    if (
        not isinstance(bound_target, Mapping)
        or bound_target.get("source_fingerprint") != expected_binding["source_fingerprint"]
        or bound_target.get("model_spec_digest") != expected_binding["model_spec_digest"]
        or current_cell_digest not in bound_target.get("authorized_cell_config_digests", [])
    ):
        raise LaunchGuardError("G00 gate is not bound to this exact target config/source/model")
    if (
        evidence.get("source_fingerprint") != expected_binding["source_fingerprint"]
        or not evidence.get("model_manifest_digests")
        or not evidence.get("expected_config_digest")
        or not evidence.get("run_binding_digest")
    ):
        raise LaunchGuardError("G00 evidence binding is incomplete or uses another source")
    return G00GateArtifact(payload, target)


@dataclass(frozen=True)
class BackendPreparation:
    """Exact identities captured after data/tokenizer/model materialization."""

    dataset_metadata: Mapping[str, Any] | None = None
    model_metadata: Mapping[str, Any] | None = None
    tokenizer_metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class BackendResult:
    """Small buffered result; large streams should be appended via ``context``."""

    summary: Mapping[str, Any] = field(default_factory=dict)
    metrics: tuple[Mapping[str, Any], ...] = ()
    predictions: tuple[Mapping[str, Any], ...] = ()
    dataset_metadata: Mapping[str, Any] | None = None
    model_metadata: Mapping[str, Any] | None = None
    tokenizer_metadata: Mapping[str, Any] | None = None


@dataclass
class RunContext:
    """The only operational surface required by a GoalZendo backend."""

    spec: RunSpec
    store: RunStore
    resumed: bool

    @property
    def config(self) -> Mapping[str, Any]:
        return self.spec.config

    @property
    def seed(self) -> int:
        return self.spec.seed

    @property
    def seeds(self) -> Mapping[str, int]:
        return self.spec.seeds

    @property
    def prior_metrics(self) -> tuple[dict[str, Any], ...]:
        return tuple(read_jsonl(self.store.metrics_path))

    @property
    def prior_predictions(self) -> tuple[dict[str, Any], ...]:
        return tuple(read_jsonl(self.store.predictions_path))

    def record_dataset_metadata(self, metadata: Mapping[str, Any]) -> Path:
        return self.store.record_dataset_metadata(metadata)

    def record_model_metadata(self, metadata: Mapping[str, Any]) -> Path:
        return self.store.record_model_metadata(metadata)

    def record_tokenizer_metadata(self, metadata: Mapping[str, Any]) -> Path:
        return self.store.record_tokenizer_metadata(metadata)

    def append_metrics(self, records: Mapping[str, Any] | Iterable[Mapping[str, Any]]) -> None:
        self.store.append_metrics(records)

    def append_predictions(self, records: Mapping[str, Any] | Iterable[Mapping[str, Any]]) -> None:
        self.store.append_predictions(records)

    def progress(self, step: int, **fields: Any) -> None:
        self.store.record_progress(step, **fields)


@runtime_checkable
class ExperimentBackend(Protocol):
    """Object-oriented backend contract (plain callables are also accepted)."""

    def run(self, context: RunContext) -> BackendResult | Mapping[str, Any] | None: ...


BackendCallable = Callable[[RunContext], BackendResult | Mapping[str, Any] | None]


def load_backend(reference: str = DEFAULT_BACKEND) -> Any:
    """Import ``module:attribute`` only when actual execution begins."""

    module_name, separator, attribute_name = str(reference).partition(":")
    if not separator or not module_name or not attribute_name:
        raise BackendContractError("backend must have module:attribute form")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise BackendContractError(f"could not import backend module {module_name!r}") from exc
    try:
        backend = getattr(module, attribute_name)
    except AttributeError as exc:
        raise BackendContractError(
            f"backend module {module_name!r} has no attribute {attribute_name!r}"
        ) from exc
    if not callable(backend) and not callable(getattr(backend, "run", None)):
        raise BackendContractError("backend attribute is neither callable nor a .run object")
    return backend


def _preparation(value: Any) -> BackendPreparation:
    if value is None:
        return BackendPreparation()
    if isinstance(value, BackendPreparation):
        return value
    if not isinstance(value, Mapping):
        raise BackendContractError("backend.prepare must return a mapping or BackendPreparation")
    return BackendPreparation(
        dataset_metadata=value.get("dataset_metadata"),
        model_metadata=value.get("model_metadata"),
        tokenizer_metadata=value.get("tokenizer_metadata"),
    )


def _result(value: Any) -> BackendResult:
    if value is None:
        return BackendResult()
    if isinstance(value, BackendResult):
        return value
    if not isinstance(value, Mapping):
        raise BackendContractError("backend must return a mapping, BackendResult, or None")
    reserved = {
        "summary",
        "metrics",
        "predictions",
        "dataset_metadata",
        "model_metadata",
        "tokenizer_metadata",
    }
    if not reserved.intersection(value):
        return BackendResult(summary=dict(value))
    summary = value.get("summary", {})
    metrics = value.get("metrics", ())
    predictions = value.get("predictions", ())
    if not isinstance(summary, Mapping):
        raise BackendContractError("backend summary must be a mapping")
    if isinstance(metrics, Mapping):
        metrics = (metrics,)
    if isinstance(predictions, Mapping):
        predictions = (predictions,)
    return BackendResult(
        summary=dict(summary),
        metrics=tuple(metrics),
        predictions=tuple(predictions),
        dataset_metadata=value.get("dataset_metadata"),
        model_metadata=value.get("model_metadata"),
        tokenizer_metadata=value.get("tokenizer_metadata"),
    )


def _record_preparation(context: RunContext, preparation: BackendPreparation) -> None:
    if preparation.dataset_metadata is not None:
        context.record_dataset_metadata(preparation.dataset_metadata)
    if preparation.model_metadata is not None:
        context.record_model_metadata(preparation.model_metadata)
    if preparation.tokenizer_metadata is not None:
        context.record_tokenizer_metadata(preparation.tokenizer_metadata)


def _prepare_backend(backend: Any, context: RunContext) -> None:
    prepare = getattr(backend, "prepare", None)
    if callable(prepare):
        _record_preparation(context, _preparation(prepare(context)))


def _execute_backend(backend: Any, context: RunContext) -> BackendResult:
    run_method = getattr(backend, "run", None)
    raw = run_method(context) if callable(run_method) else backend(context)
    return _result(raw)


def _record_result(context: RunContext, result: BackendResult) -> None:
    _record_preparation(
        context,
        BackendPreparation(
            dataset_metadata=result.dataset_metadata,
            model_metadata=result.model_metadata,
            tokenizer_metadata=result.tokenizer_metadata,
        ),
    )
    if result.metrics:
        context.append_metrics(result.metrics)
    if result.predictions:
        context.append_predictions(result.predictions)


def _require_manifests(context: RunContext) -> None:
    if not bool(get_path(context.config, "run.require_manifests", True)):
        return
    missing = [
        kind
        for kind in ("dataset", "model", "tokenizer")
        if not (context.store.path / "manifests" / f"{kind}.json").is_file()
    ]
    if missing:
        raise BackendContractError(
            "backend did not record required exact metadata manifests: " + ", ".join(missing)
        )


def assert_launch_unlocked(
    config: Mapping[str, Any],
    *,
    repo: str | Path,
    gate_artifact: str | Path | None = None,
) -> G00GateArtifact | None:
    """Require a passing, exact-binding G00 artifact for every guarded launch.

    ``run.protocol_unlocked`` is intentionally ignored.  A CLI/config override
    cannot replace the numerical engineering gates.
    """

    experiment_id = str(get_path(config, "experiment.id", ""))
    guard = get_path(config, "run.launch_guard", None)
    required_guard = REQUIRED_LAUNCH_GUARDS.get(experiment_id)
    if required_guard is not None and guard != required_guard:
        raise LaunchGuardError(
            f"registered experiment {experiment_id} has a missing or altered launch guard"
        )
    if not guard:
        return None
    if gate_artifact is None:
        raise LaunchGuardError(f"experiment launch is locked: {guard}; a bound G00 gate artifact is required")
    return verify_g00_gate_artifact(gate_artifact, config=config, repo=repo)


@dataclass(frozen=True)
class RunOutcome:
    plan_key: str
    run_id: str
    path: str
    seed: int
    state: str
    error_type: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "plan_key": self.plan_key,
            "run_id": self.run_id,
            "path": self.path,
            "seed": self.seed,
            "state": self.state,
            "error_type": self.error_type,
        }


def run_one(
    spec: RunSpec,
    *,
    repo: str | Path,
    output_root: str | Path | None = None,
    backend: Any,
    dry_run: bool = False,
    gate_artifact: str | Path | None = None,
) -> RunOutcome:
    root = Path(
        output_root
        if output_root is not None
        else get_path(spec.config, "run.output_root", "artifacts-goalzendo")
    )
    store = RunStore(root, spec.config, spec.seed, repo)
    if dry_run:
        return RunOutcome(spec.plan_key, store.run_id, str(store.path), spec.seed, "planned")

    assert_launch_unlocked(spec.config, repo=repo, gate_artifact=gate_artifact)
    resume = bool(get_path(spec.config, "run.resume", True))
    initialization = store.initialize(resume=resume)
    if initialization == "complete":
        return RunOutcome(spec.plan_key, store.run_id, str(store.path), spec.seed, "skipped")

    context = RunContext(spec=spec, store=store, resumed=initialization == "resumed")
    try:
        _prepare_backend(backend, context)
        result = _execute_backend(backend, context)
        _record_result(context, result)
        _require_manifests(context)
        summary = {
            **dict(result.summary),
            "run_id": store.run_id,
            "seed": spec.seed,
            "plan_key": spec.plan_key,
            "derived_seeds": dict(spec.seeds),
        }
        store.finalize(summary)
        return RunOutcome(spec.plan_key, store.run_id, str(store.path), spec.seed, "complete")
    except BaseException as error:
        store.fail(error)
        raise


def execute_plan(
    plan: Sequence[RunSpec],
    *,
    repo: str | Path,
    output_root: str | Path | None = None,
    backend: Any | None = None,
    backend_reference: str = DEFAULT_BACKEND,
    dry_run: bool = False,
    continue_on_error: bool = False,
    gate_artifact: str | Path | None = None,
) -> tuple[RunOutcome, ...]:
    """Execute a deterministic plan sequentially, with optional failure isolation."""

    resolved_backend = backend
    if not dry_run and resolved_backend is None:
        # Refuse every locked cell before importing heavyweight optional code.
        for spec in plan:
            assert_launch_unlocked(spec.config, repo=repo, gate_artifact=gate_artifact)
        resolved_backend = load_backend(backend_reference)

    outcomes: list[RunOutcome] = []
    for spec in plan:
        try:
            outcomes.append(
                run_one(
                    spec,
                    repo=repo,
                    output_root=output_root,
                    backend=resolved_backend,
                    dry_run=dry_run,
                    gate_artifact=gate_artifact,
                )
            )
        except Exception as error:
            if not continue_on_error:
                raise
            store = RunStore(
                output_root
                if output_root is not None
                else get_path(spec.config, "run.output_root", "artifacts-goalzendo"),
                spec.config,
                spec.seed,
                repo,
            )
            outcomes.append(
                RunOutcome(
                    spec.plan_key,
                    store.run_id,
                    str(store.path),
                    spec.seed,
                    "failed",
                    type(error).__name__,
                )
            )
    return tuple(outcomes)


def run_experiments(
    config: Mapping[str, Any],
    *,
    repo: str | Path,
    output_root: str | Path | None = None,
    backend: Any | None = None,
    backend_reference: str = DEFAULT_BACKEND,
    smoke: bool = False,
    dry_run: bool = False,
    shard_index: int = 0,
    num_shards: int = 1,
    continue_on_error: bool = False,
    gate_artifact: str | Path | None = None,
) -> tuple[RunOutcome, ...]:
    plan = build_plan(
        config,
        smoke=smoke,
        shard_index=shard_index,
        num_shards=num_shards,
    )
    return execute_plan(
        plan,
        repo=repo,
        output_root=output_root,
        backend=backend,
        backend_reference=backend_reference,
        dry_run=dry_run,
        continue_on_error=continue_on_error,
        gate_artifact=gate_artifact,
    )


__all__ = [
    "DEFAULT_BACKEND",
    "G00_GATE_SCHEMA",
    "G00_GATE_SCHEMA_VERSION",
    "G00_GATE_THRESHOLDS",
    "G00_OPTIMIZER_STABILITY_POLICY",
    "BackendContractError",
    "BackendPreparation",
    "BackendResult",
    "ExperimentBackend",
    "G00GateArtifact",
    "G00GateThresholds",
    "G00OptimizerStabilityPolicy",
    "LaunchGuardError",
    "PlanError",
    "RunContext",
    "RunOutcome",
    "RunSpec",
    "assert_launch_unlocked",
    "build_plan",
    "create_g00_gate_artifact",
    "derive_seed",
    "derived_seeds",
    "evaluate_g00_gate",
    "execute_plan",
    "g00_evidence_binding",
    "load_backend",
    "run_experiments",
    "run_one",
    "target_gate_binding",
    "verify_g00_gate_artifact",
]
