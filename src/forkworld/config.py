"""Configuration loading, sweep expansion, and confound-aware validation.

The project deliberately uses plain YAML dictionaries.  Experiment protocols evolve
quickly, and forcing every hypothesis into a single rigid dataclass tends to hide
scientifically meaningful fields.  Validation here focuses on shared invariants and
known confounds; hypothesis runners validate their own required fields as well.
"""

from __future__ import annotations

import copy
import itertools
import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

HYPOTHESES = tuple(f"h{i}" for i in range(1, 18))
TASK_LEVELS = ("choice", "fork", "navigation")

_H16_BRANCHES = (
    "independent_noop",
    "independent_q_restore",
    "independent_padding_sham",
    "nested_noop",
    "nested_q_transplant",
    "nested_padding_sham",
)
_H16_CONFIRMATORY_SEEDS = (
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
_H16_PILOT_SEEDS = (541, 547, 557)
_H16_PHASE_B_CHECKPOINTS = (
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
_H16_FEATURE_NAMES = (
    "P",
    "P_present",
    "R_1",
    "R_2",
    "R_3",
    "R_4",
    "R_5",
    "Q_present",
    "Q_1",
    "Q_2",
    "Q_3",
    "state_0",
    "state_1",
    "state_2",
    "state_3",
    "state_4",
    "state_5",
    "state_6",
    "state_7",
)

_H17_GOALS = ("P", "Q", "Y")
_H17_SCHEDULES = (
    "p_q_y",
    "p_y_q",
    "q_p_y",
    "q_y_p",
    "y_p_q",
    "y_q_p",
)
_H17_PILOT_SEEDS = (563, 569, 571)
_H17_FULL_SEEDS = (
    577,
    587,
    593,
    599,
    601,
    607,
    613,
    617,
    619,
    631,
    641,
    643,
    647,
    653,
    659,
    661,
    673,
    677,
    683,
    691,
)
_H17_CHECKPOINTS = (
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
    256,
)
_H17_FEATURE_NAMES = (
    "P",
    "P_present",
    "R_1",
    "R_2",
    "R_3",
    "Q_present",
    "Q_1",
    "Q_2",
)
_H17_FOLD_DIGEST = "25d5e9584a2ea74d42d48a673645a3b34c1b76e75e8230bf5722985e66968ec7"
_H17_CONTROL_DIGEST = "c7a186308102e807e594fd76b3fc5e7ef1678314a95de067b6fd347b390e53d0"
_H17_DESIGN_MEMO_SHA256 = "1386cb401aef8ed187389dc84aa210b09b564436bc16c9246bbd22fce257529a"
_H17_CONTROL_BITS = "0110001001011011111001011000100101011110100110000010011010110101"
_H17_DATA_SEED = 171_000_001
_H17_STREAM_SEED = 171_000_002
_H17_PHASE_SEEDS = {
    "component_P": 171_000_101,
    "component_Q": 171_000_102,
    "component_Y": 171_000_103,
    "washout": 171_000_104,
}


DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "experiment": {"hypothesis": "h1", "name": "unnamed", "mode": "primary"},
    "run": {
        "output_root": "artifacts",
        "seeds": [0],
        "device": "auto",
        "resume": True,
        "save_checkpoints": True,
        "task_levels": ["choice"],
    },
    "navigator": {
        "checkpoint_root": None,
        "auto_pretrain": True,
        "fork_maps": 1,
        "navigation_maps": 32,
        "epochs": 250,
        "target_accuracy": 0.99,
        "target_patience": 3,
        "evaluation_split": "support",
        "report_heldout_capability": True,
    },
    "data": {
        "n_train": 10000,
        "n_validation": 4000,
        "n_eval": 10000,
        "q": 0.9,
        "k": 3,
        "max_k": 5,
        "state_dim": 8,
    },
    "model": {
        "width": 64,
        "depth": 2,
        "activation": "relu",
        "residual": False,
        "bias": True,
        "nuisance_bits": 0,
    },
    "update": {"mode": "full", "budget": "full", "subspace_seed": 1729},
    "train": {
        "algorithm": "clean_sft",
        "steps": 1000,
        "batch_size": 128,
        "learning_rate": 0.003,
        "weight_decay": 0.0,
        "eval_steps": "log",
        "grad_clip": 1.0,
    },
    "evaluation": {
        "acquisition_threshold": 0.9,
        "persistence": 2,
        "equivalence_margin": 0.05,
        "bootstrap_samples": 2000,
        "confidence": 0.95,
        "minimum_inferential_seeds": 3,
        "save_predictions": True,
    },
    "cases": [],
    "sweep": {},
}


class ConfigError(ValueError):
    """Raised when a configuration would produce an invalid or confounded run."""


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings without mutating either input."""

    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def get_path(config: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def set_path(config: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    if not parts or any(not part for part in parts):
        raise ConfigError(f"Invalid dotted configuration path: {dotted!r}")
    current = config
    for part in parts[:-1]:
        child = current.setdefault(part, {})
        if not isinstance(child, dict):
            raise ConfigError(f"Cannot set {dotted!r}: {part!r} is not a mapping")
        current = child
    current[parts[-1]] = value


def parse_override(text: str) -> tuple[str, Any]:
    """Parse ``path=value`` using YAML scalar/list semantics."""

    if "=" not in text:
        raise ConfigError(f"Override must have path=value form: {text!r}")
    path, raw = text.split("=", 1)
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid override value in {text!r}: {exc}") from exc
    return path, value


def _load_yaml(source: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    source = source.resolve()
    if source in stack:
        raise ConfigError(f"Cyclic config extends chain: {' -> '.join(map(str, (*stack, source)))}")
    if not source.is_file():
        raise ConfigError(f"Configuration file does not exist: {source}")
    with source.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, Mapping):
        raise ConfigError(f"Top-level YAML object must be a mapping: {source}")
    loaded = dict(loaded)
    parent = loaded.pop("extends", None)
    if parent is None:
        return loaded
    parents = [parent] if isinstance(parent, str) else parent
    if not isinstance(parents, list) or not all(isinstance(item, str) for item in parents):
        raise ConfigError("extends must be a path or list of paths")
    merged: dict[str, Any] = {}
    for item in parents:
        merged = deep_merge(merged, _load_yaml(source.parent / item, (*stack, source)))
    return deep_merge(merged, loaded)


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> dict[str, Any]:
    """Load YAML (including ``extends``), apply defaults/overrides, and validate."""

    source = Path(path)
    loaded = _load_yaml(source)
    config = deep_merge(DEFAULT_CONFIG, loaded)
    for item in overrides:
        key, value = parse_override(item)
        set_path(config, key, value)
    config["_config_path"] = str(source.resolve())
    validate_config(config)
    return config


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def validate_config(config: Mapping[str, Any]) -> list[str]:
    """Validate shared invariants and return explicit scientific cautions.

    Cautions are persisted with each run.  Conditions that make the requested
    comparison uninterpretable raise :class:`ConfigError` instead.
    """

    cautions: list[str] = []
    hypothesis = str(get_path(config, "experiment.hypothesis", "")).lower()
    _require(hypothesis in HYPOTHESES, f"experiment.hypothesis must be one of {HYPOTHESES}")

    levels = get_path(config, "run.task_levels", [])
    _require(isinstance(levels, list) and levels, "run.task_levels must be a non-empty list")
    unknown_levels = sorted(set(levels) - set(TASK_LEVELS))
    _require(not unknown_levels, f"Unknown task levels: {unknown_levels}")

    seeds = get_path(config, "run.seeds", [])
    _require(isinstance(seeds, list) and seeds, "run.seeds must be a non-empty list")
    _require(len(set(seeds)) == len(seeds), "run.seeds must not contain duplicates")

    q = float(get_path(config, "data.q", 0.9))
    k = int(get_path(config, "data.k", 1))
    max_k = int(get_path(config, "data.max_k", 5))
    target_rule = str(get_path(config, "data.target_rule", "parity")).lower()
    _require(0.0 <= q <= 1.0, "data.q must be between 0 and 1")
    _require(1 <= k <= max_k, "data.k must satisfy 1 <= k <= data.max_k")
    _require(
        target_rule in {"parity", "majority", "conjunction", "multiplexer"},
        "data.target_rule must be parity, majority, conjunction, or multiplexer",
    )
    if target_rule == "majority":
        _require(k % 2 == 1, "majority target rules require odd data.k")
    if target_rule == "multiplexer":
        _require(
            k in {3, 6},
            "multiplexer target rules currently require data.k=3 or data.k=6",
        )
    for field in ("n_train", "n_validation", "n_eval"):
        n = int(get_path(config, f"data.{field}", 0))
        _require(n > 0 and n % 2 == 0, f"data.{field} must be a positive even integer")
    if hypothesis in {"h1", "h2", "h3", "h5", "h6"}:
        for field in ("n_train", "n_validation"):
            n = int(get_path(config, f"data.{field}"))
            conflicts = n * (1.0 - q)
            _require(
                abs(conflicts - round(conflicts)) < 1e-9,
                f"data.q={q:g} is not exactly realizable with data.{field}={n}",
            )

    mode = str(get_path(config, "update.mode", "full"))
    _require(mode in {"full", "head", "subspace"}, "update.mode must be full, head, or subspace")
    budget = get_path(config, "update.budget", "full")
    if mode == "subspace":
        _require(isinstance(budget, int) and budget > 0, "subspace update.budget must be a positive integer")

    sweep = get_path(config, "sweep", {})
    _require(isinstance(sweep, Mapping), "sweep must be a mapping from dotted paths to lists")
    for path, values in sweep.items():
        _require(isinstance(values, list) and values, f"Sweep {path!r} must be a non-empty list")
        _require(path != "run.seeds", "Use run.seeds rather than sweeping run.seeds")
    cases = get_path(config, "cases", [])
    _require(isinstance(cases, list), "cases must be a list of configuration overlays")
    _require(all(isinstance(case, Mapping) for case in cases), "every cases item must be a mapping")

    minimum_seeds = get_path(config, "evaluation.minimum_inferential_seeds", 3)
    _require(
        not isinstance(minimum_seeds, bool)
        and isinstance(minimum_seeds, int)
        and minimum_seeds >= 3,
        "evaluation.minimum_inferential_seeds must be an integer of at least 3",
    )
    equivalence_margin = get_path(config, "evaluation.equivalence_margin", 0.05)
    _require(
        not isinstance(equivalence_margin, bool)
        and isinstance(equivalence_margin, (int, float))
        and math.isfinite(float(equivalence_margin))
        and 0.0 <= float(equivalence_margin) < 1.0,
        "evaluation.equivalence_margin must be finite and lie in [0,1)",
    )
    bootstrap_samples = get_path(config, "evaluation.bootstrap_samples", 4000)
    _require(
        not isinstance(bootstrap_samples, bool)
        and isinstance(bootstrap_samples, int)
        and bootstrap_samples > 0,
        "evaluation.bootstrap_samples must be a positive integer",
    )
    confidence = get_path(config, "evaluation.confidence", 0.95)
    _require(
        not isinstance(confidence, bool)
        and isinstance(confidence, (int, float))
        and math.isfinite(float(confidence))
        and 0.0 < float(confidence) < 1.0,
        "evaluation.confidence must be finite and lie in (0,1)",
    )

    if hypothesis == "h2":
        parameter_tolerance = float(
            get_path(config, "h2.parameter_match_tolerance", 0.05)
        )
        _require(
            0.0 <= parameter_tolerance <= 1.0,
            "h2.parameter_match_tolerance must lie in [0,1]",
        )

    if hypothesis == "h4":
        n_conflict = get_path(config, "h4.n_conflict")
        if n_conflict is not None:
            n_train = int(get_path(config, "data.n_train"))
            implied = 1.0 - int(n_conflict) / n_train
            if "q" in get_path(config, "h4", {}) and abs(float(get_path(config, "h4.q")) - implied) > 1e-12:
                raise ConfigError(
                    "H4 cannot fix inconsistent N, q, and N_conflict; q must equal 1-N_conflict/N"
                )
            cautions.append(f"H4 proxy accuracy is implied by counts: q={implied:.8g}")

        ordered_mechanisms = (
            "location_reflection",
            "coordinate_exchange",
            "geometry_rotation",
            "nuisance_inversion",
        )
        known_mechanisms = set(ordered_mechanisms)

        def normalize_mechanisms(path: str) -> tuple[str, ...]:
            default = (
                list(ordered_mechanisms[:2])
                if path.endswith("structured_train_types")
                else list(ordered_mechanisms[2:])
            )
            raw = get_path(config, path, default)
            _require(
                isinstance(raw, (list, tuple)) and bool(raw),
                f"{path} must be a non-empty sequence",
            )
            normalized: list[str] = []
            for value in raw:
                if isinstance(value, bool):
                    normalized.append("<invalid>")
                elif isinstance(value, int) and 0 <= value < 4:
                    normalized.append(ordered_mechanisms[value])
                else:
                    normalized.append(str(value))
            _require(
                set(normalized) <= known_mechanisms,
                f"{path} contains an unknown failure mechanism",
            )
            _require(
                len(normalized) == len(set(normalized)),
                f"{path} must not contain duplicate mechanisms",
            )
            return tuple(normalized)

        train_mechanisms = normalize_mechanisms("h4.structured_train_types")
        test_mechanisms = normalize_mechanisms("h4.structured_test_types")
        _require(
            set(train_mechanisms).isdisjoint(test_mechanisms),
            "h4 structured train/test failure mechanisms must be disjoint",
        )

    if hypothesis == "h5":
        nuisance_bits = get_path(config, "h5.nuisance_bits", 0)
        nuisance_entropy = get_path(config, "h5.nuisance_entropy", 0)
        rollout_depth = get_path(config, "h5.on_policy_rollout_depth", 2)
        _require(
            isinstance(nuisance_bits, int)
            and not isinstance(nuisance_bits, bool)
            and nuisance_bits >= 0,
            "h5.nuisance_bits must be a non-negative integer",
        )
        _require(
            isinstance(nuisance_entropy, int)
            and not isinstance(nuisance_entropy, bool)
            and 0 <= nuisance_entropy <= nuisance_bits,
            "h5.nuisance_entropy must be an integer active-branch count between 0 and h5.nuisance_bits",
        )
        _require(
            isinstance(rollout_depth, int)
            and not isinstance(rollout_depth, bool)
            and 1 <= rollout_depth <= 16,
            "h5.on_policy_rollout_depth must be an integer in [1,16]",
        )
        if str(get_path(config, "experiment.mode", "primary")) == "entropy_timing":
            timing_schedules = {
                "zero_zero",
                "high_high",
                "early_only",
                "delayed_carry",
                "delayed_actor_reset",
                "delayed_critic_reset",
            }
            timing_schedule = str(get_path(config, "h5.timing_schedule", ""))
            phase_a_steps = get_path(config, "h5.phase_a_steps", 0)
            phase_b_steps = get_path(config, "h5.phase_b_steps", 0)
            train_steps = get_path(config, "train.steps", 0)
            _require(
                timing_schedule in timing_schedules,
                f"h5.timing_schedule must be one of {sorted(timing_schedules)}",
            )
            _require(
                all(
                    isinstance(value, int) and not isinstance(value, bool) and value > 0
                    for value in (phase_a_steps, phase_b_steps, train_steps)
                )
                and phase_a_steps + phase_b_steps == train_steps,
                "E14 phase_a_steps and phase_b_steps must be positive and sum to train.steps",
            )
            _require(
                str(get_path(config, "h5.algorithm", "")) == "rl"
                and str(get_path(config, "h5.rl_estimator", "")) == "actor_critic",
                "E14 requires h5.algorithm=rl and h5.rl_estimator=actor_critic",
            )
            _require(nuisance_entropy == 0, "E14 requires h5.nuisance_entropy=0")
            _require(
                str(get_path(config, "update.mode", "")) == "full"
                and get_path(config, "update.budget") == "full",
                "E14 requires the full actor update budget",
            )
            _require(
                str(get_path(config, "data.target_rule", "parity")) == "parity",
                "E14 requires the parity exact channel used by E10",
            )
            _require(
                (round(q, 12), k) in {(0.75, 4), (0.90, 3)},
                "E14 is fixed to the q=.75/k=4 responsive cell and q=.90/k=3 control",
            )
            if not bool(get_path(config, "_smoke", False)):
                _require(
                    (phase_a_steps, phase_b_steps, train_steps) == (64, 1_984, 2_048),
                    "full E14 runs require the fixed 64+1984=2048 update schedule",
                )
                _require(
                    get_path(config, "train.batch_size") == 250,
                    "full E14 runs require train.batch_size=250",
                )
                _require(
                    nuisance_bits == 8,
                    "full E14 runs require the fixed eight-head E10 actor interface",
                )
            cautions.append(
                "E14 entropy timing was selected after inspecting complete E10 results; "
                "all findings are exploratory/post-hoc and require independent replication."
            )

    if hypothesis == "h6":
        locations = set(get_path(config, "h6.locations", ["observation"]))
        structures = set(get_path(config, "h6.structures", []))
        _require(
            locations <= {"observation", "label", "reward"},
            "h6.locations contains an unknown intervention location",
        )
        _require(
            structures <= {"step", "step_resampled", "episode", "episode_static", "state", "state_static", "biased"},
            "h6.structures contains an unknown temporal regime",
        )
        horizon = int(get_path(config, "h6.fixed_horizon", 4))
        _require(horizon > 0, "h6.fixed_horizon must be positive")
        _require(
            int(get_path(config, "data.n_train")) % horizon == 0,
            "H6 data.n_train must be divisible by h6.fixed_horizon",
        )
        _require(
            int(get_path(config, "train.batch_size")) % horizon == 0,
            "H6 train.batch_size must be divisible by h6.fixed_horizon",
        )
        visits = get_path(config, "h6.visits_per_state", 1)
        _require(
            not isinstance(visits, bool) and isinstance(visits, int) and visits > 0,
            "h6.visits_per_state must be a positive integer",
        )
        training_epochs = get_path(config, "h6.training_epochs", 8)
        _require(
            not isinstance(training_epochs, bool)
            and isinstance(training_epochs, int)
            and training_epochs > 0,
            "h6.training_epochs must be a positive integer",
        )
        batch_size = int(get_path(config, "train.batch_size"))
        n_train = int(get_path(config, "data.n_train"))
        _require(
            n_train % batch_size == 0,
            "H6 fixed-epoch evidence requires data.n_train divisible by train.batch_size",
        )
        configured_algorithms = set(
            get_path(
                config,
                "sweep.h6.algorithm",
                [get_path(config, "h6.algorithm", "clean_sft")],
            )
        )
        if "on_policy_imitation" in configured_algorithms:
            collection_batch_size = get_path(
                config, "h6.collection_batch_size", batch_size
            )
            updates_per_collection = get_path(
                config, "h6.updates_per_collection", 1
            )
            _require(
                isinstance(collection_batch_size, int)
                and not isinstance(collection_batch_size, bool)
                and collection_batch_size == batch_size,
                "H6 fixed-epoch exposure requires on-policy collection_batch_size "
                "to equal train.batch_size",
            )
            _require(
                isinstance(updates_per_collection, int)
                and not isinstance(updates_per_collection, bool)
                and updates_per_collection == 1,
                "H6 fixed-epoch exposure requires on-policy updates_per_collection=1",
            )

        def validate_recurrence_cell(cell_n: int, cell_visits: int) -> int:
            _require(
                cell_n % horizon == 0,
                "Every H6 data.n_train sweep value must be divisible by h6.fixed_horizon",
            )
            episodes = cell_n // horizon
            _require(
                episodes % cell_visits == 0,
                "H6 recurrence requires n_train/horizon divisible by visits_per_state",
            )
            states = episodes // cell_visits
            _require(
                states > 0 and states % 2 == 0,
                "Every H6 recurrence cell must contain a positive even number of semantic states",
            )
            conflicts = states * (1.0 - q)
            _require(
                abs(conflicts - round(conflicts)) < 1e-9,
                "H6 q must be exactly realizable at the semantic-state level in every recurrence cell",
            )
            return states

        validate_recurrence_cell(n_train, int(visits))
        h6_sweep = get_path(config, "sweep", {})
        n_values = h6_sweep.get("data.n_train", [n_train])
        visit_values = h6_sweep.get("h6.visits_per_state", [int(visits)])
        _require(
            all(
                isinstance(value, int) and not isinstance(value, bool) and value > 0
                for value in visit_values
            ),
            "H6 visits_per_state sweep values must be positive integers",
        )
        _require(
            len(set(visit_values)) == len(visit_values),
            "H6 visits_per_state sweep levels must be distinct",
        )
        for cell_n in n_values:
            _require(
                isinstance(cell_n, int) and not isinstance(cell_n, bool) and cell_n > 0,
                "H6 data.n_train sweep values must be positive integers",
            )
            _require(
                cell_n % batch_size == 0,
                "Every H6 data.n_train sweep value must be divisible by train.batch_size",
            )
            realized_states = [
                validate_recurrence_cell(int(cell_n), int(cell_visits))
                for cell_visits in visit_values
            ]
            _require(
                len(set(realized_states)) == len(realized_states),
                "H6 visits_per_state levels alias the same realized recurrence",
            )
        reward_mode = get_path(config, "h6.reward_mode", "terminal")
        if "reward" in locations and reward_mode == "terminal" and {"step", "episode"} <= structures:
            cautions.append(
                "With one terminal reward, step- and episode-static reward noise are an equivalence control, not distinct treatments."
            )

    if hypothesis == "h7":
        weight_decay = float(get_path(config, "train.weight_decay", 0.0))
        ablation = bool(get_path(config, "h7.weight_decay_ablation", False))
        _require(weight_decay == 0.0 or ablation, "H7 primary comparison requires weight_decay=0")
        if not bool(get_path(config, "h7.reset_optimizer", True)):
            cautions.append("H7 carries optimizer state across phases; interpret this only as an ablation.")

    if hypothesis == "h8":
        stage1_mode = str(get_path(config, "h8.stage1_mode", "fixed")).lower()
        _require(
            stage1_mode in {"fixed", "matched", "behavior_matched"},
            "h8.stage1_mode must be fixed or behavior_matched",
        )
        if stage1_mode in {"matched", "behavior_matched"}:
            cautions.append(
                "Behavior-matched H8 uses variable realized N1 and must be reported separately from fixed-volume H8."
            )

    if hypothesis == "h9":
        for field in ("proxy0_degree", "proxy1_degree"):
            value = get_path(config, f"h9.{field}", 1)
            _require(
                not isinstance(value, bool) and isinstance(value, int) and value > 0,
                f"h9.{field} must be a positive integer",
            )

    if hypothesis == "h10":
        for field in ("q_p", "q_q"):
            value = float(get_path(config, f"h10.{field}", 0.9))
            _require(0.0 <= value <= 1.0, f"h10.{field} must lie in [0,1]")
            for count_field in ("n_train", "n_validation"):
                count = int(get_path(config, f"data.{count_field}"))
                conflicts = count * (1.0 - value)
                _require(
                    abs(conflicts - round(conflicts)) < 1e-9,
                    f"h10.{field}={value:g} is not exactly realizable with "
                    f"data.{count_field}={count}",
                )
        k_q = get_path(config, "h10.k_q", 2)
        k_y = get_path(config, "h10.k_y", 3)
        max_k_q = get_path(config, "h10.max_k_q", k_q)
        max_k_y = get_path(config, "h10.max_k_y", k_y)
        for value, name in ((k_q, "k_q"), (k_y, "k_y")):
            _require(
                isinstance(value, int) and not isinstance(value, bool) and value >= 1,
                f"h10.{name} must be a positive integer",
            )
        _require(int(max_k_q) >= int(k_q), "h10.max_k_q must be at least h10.k_q")
        _require(int(max_k_y) >= int(k_y), "h10.max_k_y must be at least h10.k_y")
        _require(
            str(get_path(config, "h10.error_structure", "independent"))
            in {"independent", "nested"},
            "h10.error_structure must be independent or nested",
        )

    if hypothesis == "h11":
        route_depth = get_path(config, "h11.route_depth", 1)
        max_depth = get_path(config, "h11.max_depth", route_depth)
        _require(
            isinstance(route_depth, int)
            and not isinstance(route_depth, bool)
            and route_depth >= 1,
            "h11.route_depth must be a positive integer",
        )
        _require(
            isinstance(max_depth, int)
            and not isinstance(max_depth, bool)
            and route_depth <= max_depth <= 4,
            "h11.max_depth must satisfy route_depth <= max_depth <= 4",
        )
        _require(
            str(get_path(config, "h11.evidence_regime", "fixed_total"))
            in {"fixed_total", "per_fork_matched"},
            "h11.evidence_regime must be fixed_total or per_fork_matched",
        )
        base_steps = get_path(
            config, "h11.base_steps", get_path(config, "train.steps", 1)
        )
        _require(
            isinstance(base_steps, int)
            and not isinstance(base_steps, bool)
            and base_steps > 0,
            "h11.base_steps must be a positive integer",
        )
        for count_field in ("n_train", "n_validation"):
            count = int(get_path(config, f"data.{count_field}"))
            per_sign_conflicts = (count // 2) * (1.0 - q)
            _require(
                abs(per_sign_conflicts - round(per_sign_conflicts)) < 1e-9,
                f"H11 q={q:g} must be exactly realizable within each label "
                f"stratum of data.{count_field}",
            )

    if hypothesis in {"h12", "h13"}:
        section = hypothesis
        for field in ("q_p", "q_q"):
            value = float(get_path(config, f"{section}.{field}", 0.9))
            _require(0.0 <= value <= 1.0, f"{section}.{field} must lie in [0,1]")
            for count_field in ("n_train", "n_validation"):
                count = int(get_path(config, f"data.{count_field}"))
                conflicts = count * (1.0 - value)
                _require(
                    abs(conflicts - round(conflicts)) < 1e-9,
                    f"{section}.{field}={value:g} is not exactly realizable with "
                    f"data.{count_field}={count}",
                )
        k_q = get_path(config, f"{section}.k_q", 2)
        k_y = get_path(config, f"{section}.k_y", 3)
        max_k_q = get_path(config, f"{section}.max_k_q", k_q)
        max_k_y = get_path(config, f"{section}.max_k_y", k_y)
        for value, name in ((k_q, "k_q"), (k_y, "k_y")):
            _require(
                isinstance(value, int) and not isinstance(value, bool) and value >= 1,
                f"{section}.{name} must be a positive integer",
            )
        _require(
            int(max_k_q) >= int(k_q),
            f"{section}.max_k_q must be at least {section}.k_q",
        )
        _require(
            int(max_k_y) >= int(k_y),
            f"{section}.max_k_y must be at least {section}.k_y",
        )
        _require(
            str(get_path(config, f"{section}.error_structure", "independent"))
            in {"independent", "nested"},
            f"{section}.error_structure must be independent or nested",
        )
        for field in ("calibration_steps", "competition_steps", "probe_train_n", "probe_eval_n"):
            value = get_path(config, f"{section}.{field}", 1)
            _require(
                isinstance(value, int) and not isinstance(value, bool) and value >= 1,
                f"{section}.{field} must be a positive integer",
            )
        calibration_steps = int(get_path(config, f"{section}.calibration_steps", 1))
        competition_steps = int(get_path(config, f"{section}.competition_steps", 1))
        bridge_step = get_path(config, f"{section}.bridge_step", calibration_steps)
        _require(
            isinstance(bridge_step, int)
            and not isinstance(bridge_step, bool)
            and 1 <= bridge_step <= competition_steps,
            f"{section}.bridge_step must be an integer in [1, {section}.competition_steps]",
        )
        eval_steps = get_path(config, "train.eval_steps", "log")
        if isinstance(eval_steps, list):
            _require(
                int(bridge_step) in {int(value) for value in eval_steps},
                f"{section}.bridge_step must be present in an explicit train.eval_steps list",
            )
        raw_codeword_count = 1 << (1 + int(k_q) + int(k_y))
        for field in ("probe_train_n", "probe_eval_n"):
            value = int(get_path(config, f"{section}.{field}", 1))
            _require(
                value % raw_codeword_count == 0,
                f"{section}.{field} must be divisible by the {raw_codeword_count} active raw codewords",
            )
        ridge = float(get_path(config, f"{section}.probe_ridge", 1e-3))
        _require(
            math.isfinite(ridge) and ridge >= 0.0,
            f"{section}.probe_ridge must be finite and non-negative",
        )
        _require(
            int(get_path(config, "data.max_k", max_k_y)) == int(max_k_y),
            f"{section.upper()} requires data.max_k == {section}.max_k_y so training and probe interfaces match",
        )
        _require(
            str(get_path(config, "update.mode", "full")) == "full",
            f"{section.upper()}'s hidden-representation bridge currently requires update.mode=full",
        )
        if hypothesis == "h13":
            q_only = get_path(config, "h13.q_only_error_count", 0)
            _require(
                isinstance(q_only, int) and not isinstance(q_only, bool) and q_only >= 0,
                "h13.q_only_error_count must be a non-negative integer",
            )
            n_train = int(get_path(config, "data.n_train"))
            p_errors = round(n_train * (1.0 - float(get_path(config, "h13.q_p"))))
            q_errors = round(n_train * (1.0 - float(get_path(config, "h13.q_q"))))
            minimum_overlap = max(0, p_errors + q_errors - n_train)
            maximum_overlap = min(p_errors, q_errors)
            _require(
                q_errors - maximum_overlap <= q_only <= q_errors - minimum_overlap,
                "h13.q_only_error_count is incompatible with the training error marginals",
            )
            _require(
                str(get_path(config, "h13.error_structure", "nested")) == "nested",
                "h13 support completion requires error_structure=nested",
            )

    if hypothesis == "h14":
        section = "h14"
        allowed_trajectories = {
            "independent_carry",
            "independent_reset",
            "nested_carry",
            "nested_reset",
            "scratch",
            "sham",
        }
        _require(
            str(get_path(config, "h14.trajectory", "")) in allowed_trajectories,
            "h14.trajectory must name one of the six frozen handoff trajectories",
        )

        n_train = int(get_path(config, "data.n_train"))
        for field in ("q_p", "q_q"):
            value = float(get_path(config, f"{section}.{field}", 0.9))
            _require(0.0 <= value <= 1.0, f"{section}.{field} must lie in [0,1]")
            phase_a_errors = n_train * (1.0 - value)
            _require(
                abs(phase_a_errors - round(phase_a_errors)) < 1e-9,
                f"{section}.{field}={value:g} is not exactly realizable with "
                f"data.n_train={n_train}",
            )
        q_q = float(get_path(config, "h14.q_q", 0.9))
        phase_b_base_errors = (n_train // 2) * (1.0 - q_q)
        _require(
            abs(phase_b_base_errors - round(phase_b_base_errors)) < 1e-9,
            "h14.q_q must be exactly realizable in the paired phase-B base batch",
        )

        k_q = get_path(config, "h14.k_q", 2)
        k_y = get_path(config, "h14.k_y", 3)
        max_k_q = get_path(config, "h14.max_k_q", k_q)
        max_k_y = get_path(config, "h14.max_k_y", k_y)
        for value, name in ((k_q, "k_q"), (k_y, "k_y")):
            _require(
                isinstance(value, int) and not isinstance(value, bool) and value >= 1,
                f"h14.{name} must be a positive integer",
            )
        _require(int(max_k_q) >= int(k_q), "h14.max_k_q must be at least h14.k_q")
        _require(int(max_k_y) >= int(k_y), "h14.max_k_y must be at least h14.k_y")
        _require(
            int(get_path(config, "data.max_k", max_k_y)) == int(max_k_y),
            "H14 requires data.max_k == h14.max_k_y",
        )
        _require(
            str(get_path(config, "update.mode", "full")) == "full",
            "H14 hidden-representation probes require update.mode=full",
        )

        phase_a_steps = get_path(config, "h14.phase_a_steps", 0)
        phase_b_steps = get_path(config, "h14.phase_b_steps", 0)
        for value, name in (
            (phase_a_steps, "phase_a_steps"),
            (phase_b_steps, "phase_b_steps"),
        ):
            _require(
                isinstance(value, int) and not isinstance(value, bool) and value >= 1,
                f"h14.{name} must be a positive integer",
            )
        _require(
            int(get_path(config, "train.steps", 0)) == int(phase_b_steps),
            "h14.phase_b_steps must equal train.steps",
        )
        batch_size = int(get_path(config, "train.batch_size", 0))
        _require(
            batch_size >= 1 and n_train % batch_size == 0,
            "H14 requires data.n_train to be divisible by train.batch_size",
        )

        eligibility_steps = get_path(config, "h14.eligibility_steps", [])
        _require(
            isinstance(eligibility_steps, list)
            and len(eligibility_steps) >= 2
            and all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and 1 <= value <= int(phase_a_steps)
                for value in eligibility_steps
            )
            and eligibility_steps == sorted(set(eligibility_steps))
            and eligibility_steps[-1] == int(phase_a_steps),
            "h14.eligibility_steps must be unique increasing phase-A steps ending at the boundary",
        )
        phase_b_checkpoints = get_path(config, "h14.phase_b_checkpoints", [])
        _require(
            isinstance(phase_b_checkpoints, list)
            and len(phase_b_checkpoints) >= 2
            and all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and 0 <= value <= int(phase_b_steps)
                for value in phase_b_checkpoints
            )
            and phase_b_checkpoints == sorted(set(phase_b_checkpoints))
            and phase_b_checkpoints[0] == 0
            and phase_b_checkpoints[-1] == int(phase_b_steps),
            "h14.phase_b_checkpoints must be unique increasing steps from zero to the phase-B horizon",
        )
        auc_horizon = get_path(config, "h14.auc_horizon", 0)
        _require(
            isinstance(auc_horizon, int)
            and not isinstance(auc_horizon, bool)
            and auc_horizon in phase_b_checkpoints
            and 1 <= auc_horizon <= int(phase_b_steps),
            "h14.auc_horizon must be an explicitly recorded positive phase-B checkpoint",
        )
        if int(phase_b_steps) >= 128:
            _require(
                auc_horizon == 128 and 128 in phase_b_checkpoints,
                "H14's full design requires a directly observed local checkpoint 128 AUC horizon",
            )

        raw_codeword_count = 1 << (1 + int(k_q) + int(k_y))
        for field in ("probe_train_n", "probe_eval_n"):
            value = get_path(config, f"h14.{field}", 0)
            _require(
                isinstance(value, int)
                and not isinstance(value, bool)
                and value >= 1
                and value % raw_codeword_count == 0,
                f"h14.{field} must be divisible by the {raw_codeword_count} active raw codewords",
            )
        ridge = float(get_path(config, "h14.probe_ridge", 1e-3))
        _require(
            math.isfinite(ridge) and ridge >= 0.0,
            "h14.probe_ridge must be finite and non-negative",
        )
        control_seed = get_path(config, "h14.truth_table_control_seed", None)
        _require(
            isinstance(control_seed, int) and not isinstance(control_seed, bool),
            "h14.truth_table_control_seed must be an integer",
        )
        sham_repeats = get_path(config, "h14.sham_source_repeats", 0)
        _require(
            isinstance(sham_repeats, int)
            and not isinstance(sham_repeats, bool)
            and sham_repeats >= 1
            and sham_repeats * raw_codeword_count >= n_train // 2,
            "h14.sham_source_repeats does not provide enough factorial rows for the compute sham",
        )
        for field in ("pure_threshold", "pure_margin"):
            value = float(get_path(config, f"h14.{field}", -1.0))
            _require(
                math.isfinite(value) and 0.0 <= value <= 1.0,
                f"h14.{field} must be finite and lie in [0,1]",
            )

    if hypothesis == "h15":
        section = "h15"
        allowed_schedules = {"b_then_d", "d_then_b", "interleave"}
        _require(
            str(get_path(config, "h15.schedule", "")) in allowed_schedules,
            "h15.schedule must be b_then_d, d_then_b, or interleave",
        )

        n_train = int(get_path(config, "data.n_train"))
        q_p = float(get_path(config, "h15.q_p", 0.90))
        q_q = float(get_path(config, "h15.q_q", 0.95))
        _require(
            math.isfinite(q_p)
            and math.isfinite(q_q)
            and 0.0 < q_p < q_q < 1.0,
            "H15 requires 0 < h15.q_p < h15.q_q < 1",
        )
        raw_stratum_counts = {
            "A": n_train * q_p,
            "B": n_train * (q_q - q_p),
            "D": n_train * (1.0 - q_q),
        }
        _require(
            all(abs(value - round(value)) < 1e-9 for value in raw_stratum_counts.values()),
            "H15 A/B/D evidence counts must be exactly realizable with data.n_train",
        )
        stratum_counts = {name: round(value) for name, value in raw_stratum_counts.items()}
        _require(
            sum(stratum_counts.values()) == n_train
            and all(value > 0 and value % 2 == 0 for value in stratum_counts.values()),
            "H15 A/B/D evidence strata must form positive label-balanced counts",
        )
        _require(
            stratum_counts["B"] == stratum_counts["D"],
            "H15 requires equal B and D counts for order-only diagnostic blocks",
        )

        k_q = get_path(config, "h15.k_q", 2)
        k_y = get_path(config, "h15.k_y", 3)
        max_k_q = get_path(config, "h15.max_k_q", k_q)
        max_k_y = get_path(config, "h15.max_k_y", k_y)
        for value, name in (
            (k_q, "k_q"),
            (k_y, "k_y"),
            (max_k_q, "max_k_q"),
            (max_k_y, "max_k_y"),
        ):
            _require(
                isinstance(value, int) and not isinstance(value, bool) and value >= 1,
                f"h15.{name} must be a positive integer",
            )
        _require(int(max_k_q) >= int(k_q), "h15.max_k_q must be at least h15.k_q")
        _require(int(max_k_y) >= int(k_y), "h15.max_k_y must be at least h15.k_y")
        _require(
            int(get_path(config, "data.max_k", max_k_y)) == int(max_k_y),
            "H15 requires data.max_k == h15.max_k_y",
        )
        _require(
            str(get_path(config, "update.mode", "full")) == "full",
            "H15 hidden-representation probes require update.mode=full",
        )
        _require(
            str(get_path(config, "train.optimizer", "adamw")).lower() == "adamw",
            "H15's frozen reset intervention requires train.optimizer=adamw",
        )
        _require(
            get_path(config, "train.shuffle", True) is False,
            "H15 requires train.shuffle=false so atomic-batch order is not randomized",
        )

        batch_size = get_path(config, "train.batch_size", 0)
        _require(
            isinstance(batch_size, int)
            and not isinstance(batch_size, bool)
            and batch_size >= 2
            and batch_size % 2 == 0,
            "H15 train.batch_size must be a positive even integer",
        )
        _require(
            all(count % int(batch_size) == 0 for count in stratum_counts.values()),
            "H15 each A/B/D stratum must fill label-balanced atomic batches exactly",
        )

        repetition_fields = (
            "prefix_a_repetitions",
            "washout_a_repetitions",
            "diagnostic_repetitions",
        )
        repetitions: dict[str, int] = {}
        for field in repetition_fields:
            value = get_path(config, f"h15.{field}", 0)
            _require(
                isinstance(value, int) and not isinstance(value, bool) and value >= 1,
                f"h15.{field} must be a positive integer",
            )
            repetitions[field] = int(value)
        _require(
            repetitions["prefix_a_repetitions"]
            + repetitions["washout_a_repetitions"]
            == repetitions["diagnostic_repetitions"],
            "H15 requires every A/B/D source row to receive the same total presentations",
        )

        derived_steps = {
            "prefix_steps": stratum_counts["A"]
            // int(batch_size)
            * repetitions["prefix_a_repetitions"],
            "block_steps": stratum_counts["B"]
            // int(batch_size)
            * repetitions["diagnostic_repetitions"],
            "washout_steps": stratum_counts["A"]
            // int(batch_size)
            * repetitions["washout_a_repetitions"],
        }
        derived_steps["total_steps"] = (
            derived_steps["prefix_steps"]
            + 2 * derived_steps["block_steps"]
            + derived_steps["washout_steps"]
        )
        for field, expected in derived_steps.items():
            value = get_path(config, f"h15.{field}", 0)
            _require(
                isinstance(value, int)
                and not isinstance(value, bool)
                and value == expected,
                f"h15.{field} must equal the exact derived value {expected}",
            )
        _require(
            int(get_path(config, "train.steps", 0)) == derived_steps["total_steps"],
            "H15 train.steps must equal h15.total_steps",
        )

        second_block_checkpoints = get_path(config, "h15.second_block_checkpoints", [])
        _require(
            isinstance(second_block_checkpoints, list)
            and len(second_block_checkpoints) >= 2
            and all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and 1 <= value <= derived_steps["block_steps"]
                for value in second_block_checkpoints
            )
            and second_block_checkpoints == sorted(set(second_block_checkpoints))
            and second_block_checkpoints[-1] == derived_steps["block_steps"],
            "h15.second_block_checkpoints must be unique increasing offsets ending at block_steps",
        )
        washout_checkpoints = get_path(config, "h15.washout_checkpoints", [])
        _require(
            isinstance(washout_checkpoints, list)
            and len(washout_checkpoints) >= 3
            and all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and 0 <= value <= derived_steps["washout_steps"]
                for value in washout_checkpoints
            )
            and washout_checkpoints == sorted(set(washout_checkpoints))
            and washout_checkpoints[0] == 0
            and washout_checkpoints[-1] == derived_steps["washout_steps"],
            "h15.washout_checkpoints must be unique increasing offsets from zero to washout_steps",
        )
        auc_horizon = get_path(config, "h15.auc_horizon", 0)
        _require(
            isinstance(auc_horizon, int)
            and not isinstance(auc_horizon, bool)
            and auc_horizon in washout_checkpoints
            and 1 <= auc_horizon <= derived_steps["washout_steps"],
            "h15.auc_horizon must be a directly observed positive washout checkpoint",
        )
        if derived_steps["washout_steps"] >= 128:
            _require(
                auc_horizon == 128 and 128 in washout_checkpoints,
                "H15's full design requires a directly observed washout checkpoint 128 AUC horizon",
            )

        late_stability_steps = get_path(config, "h15.late_stability_steps", [])
        _require(
            isinstance(late_stability_steps, list)
            and len(late_stability_steps) == 3
            and late_stability_steps == sorted(set(late_stability_steps))
            and all(value in washout_checkpoints for value in late_stability_steps)
            and late_stability_steps[-1] == derived_steps["washout_steps"],
            "h15.late_stability_steps must be three observed washout checkpoints ending at the horizon",
        )
        late_tolerance = float(get_path(config, "h15.late_stability_tolerance", -1.0))
        _require(
            math.isfinite(late_tolerance) and 0.0 <= late_tolerance <= 1.0,
            "h15.late_stability_tolerance must be finite and lie in [0,1]",
        )

        raw_codeword_count = 1 << (1 + int(k_q) + int(k_y))
        for field in ("probe_train_n", "probe_eval_n"):
            value = get_path(config, f"h15.{field}", 0)
            _require(
                isinstance(value, int)
                and not isinstance(value, bool)
                and value >= 1
                and value % raw_codeword_count == 0,
                f"h15.{field} must be divisible by the {raw_codeword_count} active raw codewords",
            )
        ridge = float(get_path(config, "h15.probe_ridge", 1e-3))
        _require(
            math.isfinite(ridge) and ridge >= 0.0,
            "h15.probe_ridge must be finite and non-negative",
        )
        control_seed = get_path(config, "h15.truth_table_control_seed", None)
        _require(
            isinstance(control_seed, int) and not isinstance(control_seed, bool),
            "h15.truth_table_control_seed must be an integer",
        )
        for field in ("pure_threshold", "pure_margin"):
            value = float(get_path(config, f"h15.{field}", -1.0))
            _require(
                math.isfinite(value) and 0.0 <= value <= 1.0,
                f"h15.{field} must be finite and lie in [0,1]",
            )
        for field in (
            "primary_effect_threshold",
            "equivalence_margin",
            "terminal_effect_threshold",
        ):
            value = float(get_path(config, f"h15.{field}", -1.0))
            _require(
                math.isfinite(value) and 0.0 <= value <= 2.0,
                f"h15.{field} must be finite and lie in [0,2]",
            )
        minimum_sign_count = get_path(config, "h15.minimum_sign_count", 0)
        _require(
            isinstance(minimum_sign_count, int)
            and not isinstance(minimum_sign_count, bool)
            and minimum_sign_count >= 1,
            "h15.minimum_sign_count must be a positive integer",
        )

        pilot_seeds = get_path(config, "h15.pilot_seeds", [])
        run_seeds = get_path(config, "run.seeds", [])
        _require(
            isinstance(pilot_seeds, list)
            and len(pilot_seeds) == 3
            and all(isinstance(seed, int) and not isinstance(seed, bool) for seed in pilot_seeds)
            and len(set(pilot_seeds)) == 3,
            "h15.pilot_seeds must contain exactly three unique integer seeds",
        )
        _require(
            not set(pilot_seeds) & set(run_seeds),
            "H15 engineering-pilot seeds must be disjoint from confirmatory run.seeds",
        )
        if not bool(get_path(config, "_smoke", False)):
            _require(
                int(minimum_sign_count) <= len(run_seeds),
                "h15.minimum_sign_count cannot exceed the confirmatory seed count",
            )

    if hypothesis == "h16":
        section = "h16"
        smoke = bool(get_path(config, "_smoke", False))
        branch = str(get_path(config, "h16.branch", ""))
        _require(
            branch in _H16_BRANCHES,
            "h16.branch must name one of the six frozen Q-pathway branches",
        )
        pilot_only = get_path(config, "h16.pilot_only", None)
        _require(isinstance(pilot_only, bool), "h16.pilot_only must be a boolean")

        run_seeds = get_path(config, "run.seeds", [])
        pilot_seeds = get_path(config, "h16.pilot_seeds", [])
        _require(
            isinstance(pilot_seeds, list)
            and len(pilot_seeds) == 3
            and all(isinstance(seed, int) and not isinstance(seed, bool) for seed in pilot_seeds)
            and len(set(pilot_seeds)) == 3,
            "h16.pilot_seeds must contain exactly three unique integer seeds",
        )
        _require(
            not set(pilot_seeds) & set(run_seeds),
            "H16 engineering-pilot seeds must be disjoint from confirmatory run.seeds",
        )

        n_train = get_path(config, "data.n_train", 0)
        _require(
            isinstance(n_train, int) and not isinstance(n_train, bool) and n_train >= 2,
            "H16 data.n_train must be an integer of at least two",
        )
        q_p = get_path(config, "h16.q_p", None)
        q_q = get_path(config, "h16.q_q", None)
        for field, value in (("q_p", q_p), ("q_q", q_q)):
            _require(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and 0.0 < float(value) < 1.0,
                f"h16.{field} must be finite and lie strictly between zero and one",
            )
            errors = int(n_train) * (1.0 - float(value))
            _require(
                abs(errors - round(errors)) < 1e-9,
                f"h16.{field} must be exactly realizable with data.n_train",
            )
        phase_b_base_errors = (int(n_train) // 2) * (1.0 - float(q_q))
        _require(
            int(n_train) % 2 == 0
            and abs(phase_b_base_errors - round(phase_b_base_errors)) < 1e-9,
            "h16.q_q must be exactly realizable in the paired phase-B base batch",
        )

        dimensions: dict[str, int] = {}
        for field in ("k_q", "k_y", "max_k_q", "max_k_y"):
            value = get_path(config, f"h16.{field}", 0)
            _require(
                isinstance(value, int) and not isinstance(value, bool) and value >= 1,
                f"h16.{field} must be a positive integer",
            )
            dimensions[field] = int(value)
        _require(
            dimensions["max_k_q"] >= dimensions["k_q"],
            "h16.max_k_q must be at least h16.k_q",
        )
        _require(
            dimensions["max_k_y"] >= dimensions["k_y"],
            "h16.max_k_y must be at least h16.k_y",
        )
        _require(
            int(get_path(config, "data.max_k", 0)) == dimensions["max_k_y"],
            "H16 requires data.max_k == h16.max_k_y",
        )
        _require(
            str(get_path(config, "update.mode", "")) == "full"
            and get_path(config, "update.budget", None) == "full",
            "H16 requires full-model updates with update.budget=full",
        )
        _require(
            str(get_path(config, "train.optimizer", "adamw")).lower() == "adamw",
            "H16 requires a fresh AdamW optimizer after surgery",
        )

        phase_a_steps = get_path(config, "h16.phase_a_steps", 0)
        phase_b_steps = get_path(config, "h16.phase_b_steps", 0)
        for field, value in (
            ("phase_a_steps", phase_a_steps),
            ("phase_b_steps", phase_b_steps),
        ):
            _require(
                isinstance(value, int) and not isinstance(value, bool) and value >= 1,
                f"h16.{field} must be a positive integer",
            )
        _require(
            get_path(config, "train.steps", 0) == phase_b_steps,
            "h16.phase_b_steps must equal train.steps",
        )
        batch_size = get_path(config, "train.batch_size", 0)
        _require(
            isinstance(batch_size, int)
            and not isinstance(batch_size, bool)
            and batch_size >= 1
            and int(n_train) % batch_size == 0,
            "H16 requires data.n_train to be divisible by train.batch_size",
        )

        eligibility_steps = get_path(config, "h16.eligibility_steps", [])
        _require(
            isinstance(eligibility_steps, list)
            and len(eligibility_steps) >= 2
            and all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and 1 <= value <= int(phase_a_steps)
                for value in eligibility_steps
            )
            and eligibility_steps == sorted(set(eligibility_steps))
            and eligibility_steps[-1] == int(phase_a_steps),
            "h16.eligibility_steps must be unique increasing phase-A steps ending at the boundary",
        )
        phase_b_checkpoints = get_path(config, "h16.phase_b_checkpoints", [])
        _require(
            isinstance(phase_b_checkpoints, list)
            and len(phase_b_checkpoints) >= 2
            and all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and 0 <= value <= int(phase_b_steps)
                for value in phase_b_checkpoints
            )
            and phase_b_checkpoints == sorted(set(phase_b_checkpoints))
            and phase_b_checkpoints[0] == 0
            and phase_b_checkpoints[-1] == int(phase_b_steps),
            "h16.phase_b_checkpoints must be unique increasing steps from zero to the phase-B horizon",
        )
        auc_horizon = get_path(config, "h16.auc_horizon", 0)
        _require(
            isinstance(auc_horizon, int)
            and not isinstance(auc_horizon, bool)
            and auc_horizon in phase_b_checkpoints
            and 1 <= auc_horizon <= int(phase_b_steps),
            "h16.auc_horizon must be an explicitly recorded positive phase-B checkpoint",
        )

        raw_codeword_count = 1 << (1 + dimensions["k_q"] + dimensions["k_y"])
        for field in ("probe_train_n", "probe_eval_n"):
            value = get_path(config, f"h16.{field}", 0)
            _require(
                isinstance(value, int)
                and not isinstance(value, bool)
                and value >= 1
                and value % raw_codeword_count == 0,
                f"h16.{field} must be divisible by the {raw_codeword_count} active raw codewords",
            )
        probe_ridge = get_path(config, "h16.probe_ridge", None)
        _require(
            isinstance(probe_ridge, (int, float))
            and not isinstance(probe_ridge, bool)
            and math.isfinite(float(probe_ridge))
            and float(probe_ridge) >= 0.0,
            "h16.probe_ridge must be finite and non-negative",
        )
        control_seed = get_path(config, "h16.truth_table_control_seed", None)
        _require(
            isinstance(control_seed, int) and not isinstance(control_seed, bool),
            "h16.truth_table_control_seed must be an integer",
        )

        feature_names = get_path(config, "h16.feature_names", [])
        active_columns = get_path(config, "h16.active_q_columns", [])
        sham_columns = get_path(config, "h16.sham_padding_columns", [])
        expected_input_dim = get_path(config, "h16.expected_input_dim", 0)
        expected_edited_scalars = get_path(config, "h16.expected_edited_scalars", 0)
        model_width = get_path(config, "model.width", 0)
        _require(
            get_path(config, "h16.parameter_name", "") == "input_projection.weight",
            "H16 may edit only input_projection.weight",
        )
        _require(
            feature_names == list(_H16_FEATURE_NAMES),
            "h16.feature_names must equal the frozen 19-coordinate feature interface",
        )
        _require(
            active_columns == [8, 9] and sham_columns == [5, 6],
            "H16 active Q and sham padding columns must be [8,9] and [5,6]",
        )
        _require(
            expected_input_dim == len(_H16_FEATURE_NAMES) == 19,
            "h16.expected_input_dim must equal the frozen feature width 19",
        )
        _require(
            isinstance(model_width, int)
            and not isinstance(model_width, bool)
            and expected_edited_scalars == model_width * len(active_columns) == 128,
            "h16.expected_edited_scalars must equal two columns times model.width (128)",
        )

        probability_thresholds = (
            "pure_threshold",
            "pure_margin",
            "probe_shift_threshold",
            "preservation_margin",
            "primary_effect_threshold",
            "sham_auc_equivalence_margin",
        )
        for field in probability_thresholds:
            value = get_path(config, f"h16.{field}", None)
            _require(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and 0.0 <= float(value) <= 1.0,
                f"h16.{field} must be finite and lie in [0,1]",
            )
        for field in (
            "pilot_minimum_joint_seeds",
            "minimum_sign_count",
            "minimum_eligible_seeds",
        ):
            value = get_path(config, f"h16.{field}", 0)
            _require(
                isinstance(value, int) and not isinstance(value, bool) and value >= 1,
                f"h16.{field} must be a positive integer",
            )
        _require(
            int(get_path(config, "h16.pilot_minimum_joint_seeds")) <= len(pilot_seeds),
            "h16.pilot_minimum_joint_seeds cannot exceed the pilot seed count",
        )

        if not smoke:
            _require(
                tuple(run_seeds) == _H16_CONFIRMATORY_SEEDS,
                "H16 run.seeds must equal the frozen 20-seed confirmatory panel",
            )
            _require(
                tuple(pilot_seeds) == _H16_PILOT_SEEDS,
                "h16.pilot_seeds must equal the frozen disjoint three-seed panel",
            )
            _require(
                int(get_path(config, "h16.minimum_sign_count")) <= len(run_seeds)
                and int(get_path(config, "h16.minimum_eligible_seeds")) <= len(run_seeds),
                "H16 full-panel count thresholds cannot exceed the confirmatory seed count",
            )

            h16_frozen_values: dict[str, Any] = {
                "run.device": "cpu",
                "run.task_levels": ["choice"],
                "run.save_checkpoints": False,
                "evaluation.bootstrap_samples": 4000,
                "evaluation.confidence": 0.95,
                "evaluation.save_predictions": False,
                "data.n_train": 10000,
                "data.n_validation": 4000,
                "data.n_eval": 10000,
                "data.q": 0.90,
                "data.k": 3,
                "data.max_k": 5,
                "data.state_dim": 8,
                "model.width": 64,
                "model.depth": 2,
                "model.activation": "relu",
                "model.residual": False,
                "model.bias": True,
                "update.mode": "full",
                "update.budget": "full",
                "train.algorithm": "clean_sft",
                "train.optimizer": "adamw",
                "train.steps": 1024,
                "train.batch_size": 250,
                "train.learning_rate": 0.003,
                "train.weight_decay": 0.0,
                "train.shuffle": True,
                "train.deterministic": True,
                "train.eval_steps": list(_H16_PHASE_B_CHECKPOINTS[1:]),
                "h16.q_p": 0.90,
                "h16.q_q": 0.90,
                "h16.k_q": 2,
                "h16.k_y": 3,
                "h16.max_k_q": 3,
                "h16.max_k_y": 5,
                "h16.phase_a_steps": 45,
                "h16.phase_b_steps": 1024,
                "h16.eligibility_steps": [33, 45],
                "h16.phase_b_checkpoints": list(_H16_PHASE_B_CHECKPOINTS),
                "h16.probe_train_n": 2048,
                "h16.probe_eval_n": 4096,
                "h16.probe_ridge": 0.001,
                "h16.truth_table_control_seed": 1500450271,
                "h16.auc_horizon": 128,
                "h16.pure_threshold": 0.90,
                "h16.pure_margin": 0.10,
                "h16.probe_shift_threshold": 0.05,
                "h16.preservation_margin": 0.05,
                "h16.pilot_minimum_joint_seeds": 2,
                "h16.primary_effect_threshold": 0.02,
                "h16.sham_auc_equivalence_margin": 0.01,
                "h16.minimum_sign_count": 15,
                "h16.minimum_eligible_seeds": 15,
            }
            for path, expected in h16_frozen_values.items():
                actual = get_path(config, path, None)
                _require(
                    type(actual) is type(expected) and actual == expected,
                    f"{path} must equal the frozen H16 value {expected!r}",
                )

            # Top-level configs must enumerate the complete six-arm grid. Once
            # expanded, `_case_index` identifies a single validated arm.
            if "_case_index" not in config:
                h16_cases = get_path(config, "cases", [])
                case_branches = [
                    get_path(case, "h16.branch", "")
                    for case in h16_cases
                    if isinstance(case, Mapping)
                ]
                exact_overlays = all(
                    set(case) == {"h16"}
                    and isinstance(case["h16"], Mapping)
                    and set(case["h16"]) == {"branch"}
                    for case in h16_cases
                    if isinstance(case, Mapping)
                )
                _require(
                    len(h16_cases) == len(_H16_BRANCHES)
                    and len(case_branches) == len(_H16_BRANCHES)
                    and tuple(case_branches) == _H16_BRANCHES
                    and exact_overlays,
                    "H16 cases must be the exact ordered six-branch intervention grid",
                )

    if hypothesis == "h17":
        smoke = bool(get_path(config, "_smoke", False))
        pilot_only = get_path(config, "h17.pilot_only", None)
        _require(isinstance(pilot_only, bool), "h17.pilot_only must be a boolean")

        pilot_seeds = get_path(config, "h17.pilot_seeds", [])
        full_seeds = get_path(config, "h17.full_seeds", [])
        run_seeds = get_path(config, "run.seeds", [])
        _require(
            isinstance(pilot_seeds, list)
            and tuple(pilot_seeds) == _H17_PILOT_SEEDS,
            "h17.pilot_seeds must equal the frozen three-seed pilot panel",
        )
        _require(
            isinstance(full_seeds, list)
            and tuple(full_seeds) == _H17_FULL_SEEDS,
            "h17.full_seeds must equal the frozen 20-seed full panel",
        )
        _require(
            not set(pilot_seeds) & set(full_seeds),
            "H17 pilot and full seed panels must be disjoint",
        )
        if not smoke:
            expected_seeds = _H17_PILOT_SEEDS if pilot_only else _H17_FULL_SEEDS
            _require(
                tuple(run_seeds) == expected_seeds,
                "H17 run.seeds must equal the frozen pilot or full panel",
            )

        schedule = get_path(config, "h17.schedule", None)
        isolated_goal = get_path(config, "h17.isolated_goal", None)
        if pilot_only:
            _require(
                isolated_goal in _H17_GOALS,
                "h17.isolated_goal must be P, Q, or Y for the isolated pilot",
            )
            _require(
                schedule is None,
                "H17 isolated pilot must not define or construct an order schedule",
            )
        else:
            _require(
                schedule in _H17_SCHEDULES,
                "h17.schedule must name one of the six frozen component orders",
            )
            _require(
                isolated_goal is None,
                "H17 full schedules must not define an isolated pilot goal",
            )

        feature_names = get_path(config, "h17.feature_names", [])
        _require(
            feature_names == list(_H17_FEATURE_NAMES),
            "h17.feature_names must equal the frozen eight-coordinate interface",
        )
        _require(
            get_path(config, "h17.presence_constants", None)
            == {"P_present": 1, "Q_present": 1},
            "h17.presence_constants must freeze both presence channels to +1",
        )
        _require(
            get_path(config, "h17.expected_input_dim", None) == 8,
            "h17.expected_input_dim must equal eight",
        )
        _require(
            get_path(config, "h17.k_q", None) == 2
            and get_path(config, "h17.k_y", None) == 3,
            "H17 requires direct P, parity-2 Q, and parity-3 Y",
        )
        _require(
            get_path(config, "data.k", None) == 3
            and get_path(config, "data.max_k", None) == 3
            and get_path(config, "data.state_dim", None) == 0,
            "H17 requires degree three with no padding or state features",
        )
        _require(
            get_path(config, "run.device", None) == "cpu",
            "H17 exact counterbalanced replay is CPU-only",
        )
        _require(
            get_path(config, "run.task_levels", None) == ["choice"],
            "H17 is frozen to the choice task level",
        )
        _require(
            get_path(config, "run.save_checkpoints", None) is False
            and get_path(config, "evaluation.save_predictions", None) is False,
            "H17 stores audited summaries only; checkpoints and predictions must be disabled",
        )
        _require(
            get_path(config, "model.width", None) == 64
            and get_path(config, "model.depth", None) == 2
            and get_path(config, "model.activation", None) == "relu"
            and get_path(config, "model.residual", None) is False
            and get_path(config, "model.bias", None) is True
            and get_path(config, "model.nuisance_bits", None) == 0,
            "H17 requires the frozen width-64 depth-2 nonresidual ReLU GoalMLP "
            "with bias and no auxiliary heads",
        )
        _require(
            get_path(config, "update.mode", None) == "full"
            and get_path(config, "update.budget", None) == "full",
            "H17 requires full-model updates with the full budget",
        )
        _require(
            get_path(config, "train.algorithm", None) == "clean_sft"
            and str(get_path(config, "train.optimizer", "")).lower() == "adamw"
            and get_path(config, "train.learning_rate", None) == 0.003
            and get_path(config, "train.weight_decay", None) == 0.0
            and get_path(config, "train.shuffle", None) is False
            and get_path(config, "train.deterministic", None) is True
            and type(get_path(config, "train.grad_clip", None)) is float
            and get_path(config, "train.grad_clip", None) == 1.0
            and type(get_path(config, "train.label_smoothing", None)) is float
            and get_path(config, "train.label_smoothing", None) == 0.0,
            "H17 requires deterministic unshuffled AdamW clean SFT at lr=.003, wd=0, "
            "gradient clipping at norm 1.0, and zero label smoothing",
        )
        _require(
            type(get_path(config, "h17.data_seed", None)) is int
            and get_path(config, "h17.data_seed", None) == _H17_DATA_SEED
            and type(get_path(config, "h17.stream_seed", None)) is int
            and get_path(config, "h17.stream_seed", None) == _H17_STREAM_SEED
            and get_path(config, "h17.phase_seeds", None) == _H17_PHASE_SEEDS,
            "H17 data, atomic stream, and phase RNG seeds must equal the frozen "
            "run-seed-independent constants",
        )

        integer_fields: dict[str, int] = {}
        for field in (
            "rows_per_weight_unit",
            "weighted_strata",
            "examples_per_weight_unit_per_batch",
            "batches_per_presentation",
            "presentations",
            "component_steps",
            "washout_steps",
            "total_steps",
        ):
            value = get_path(config, f"h17.{field}", 0)
            _require(
                isinstance(value, int) and not isinstance(value, bool) and value >= 1,
                f"h17.{field} must be a positive integer",
            )
            integer_fields[field] = int(value)
        batch_size = get_path(config, "train.batch_size", 0)
        n_train = get_path(config, "data.n_train", 0)
        _require(
            isinstance(batch_size, int) and not isinstance(batch_size, bool),
            "H17 train.batch_size must be an integer",
        )
        _require(
            integer_fields["component_steps"]
            == integer_fields["presentations"]
            * integer_fields["batches_per_presentation"],
            "h17.component_steps must equal presentations times batches_per_presentation",
        )
        _require(
            int(n_train)
            == integer_fields["rows_per_weight_unit"]
            * integer_fields["weighted_strata"],
            "H17 data.n_train must equal rows_per_weight_unit times weighted_strata",
        )
        _require(
            int(batch_size)
            == integer_fields["examples_per_weight_unit_per_batch"]
            * integer_fields["weighted_strata"],
            "H17 batch size must contain the registered count from every weighted stratum",
        )
        expected_total = (
            integer_fields["component_steps"]
            if pilot_only
            else 3 * integer_fields["component_steps"] + integer_fields["washout_steps"]
        )
        _require(
            integer_fields["total_steps"] == expected_total
            and get_path(config, "train.steps", None) == expected_total,
            "H17 total_steps and train.steps must equal the exact pilot/full block total",
        )

        component_checkpoints = get_path(config, "h17.component_checkpoints", [])
        washout_checkpoints = get_path(config, "h17.washout_checkpoints", [])
        for field, values, horizon in (
            ("component_checkpoints", component_checkpoints, integer_fields["component_steps"]),
            ("washout_checkpoints", washout_checkpoints, integer_fields["washout_steps"]),
        ):
            _require(
                isinstance(values, list)
                and len(values) >= 3
                and all(
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and 0 <= value <= horizon
                    for value in values
                )
                and values == sorted(set(values))
                and values[0] == 0
                and values[-1] == horizon,
                f"h17.{field} must be unique increasing direct offsets from zero to its horizon",
            )
        late_steps = get_path(config, "h17.pilot_late_steps", [])
        _require(
            isinstance(late_steps, list)
            and len(late_steps) == 3
            and late_steps == sorted(set(late_steps))
            and all(value in component_checkpoints for value in late_steps)
            and late_steps[-1] == integer_fields["component_steps"],
            "h17.pilot_late_steps must be three direct component checkpoints ending at the horizon",
        )
        primary_window = get_path(config, "h17.primary_auc_window", [])
        _require(
            isinstance(primary_window, list)
            and len(primary_window) == 2
            and primary_window == sorted(set(primary_window))
            and all(value in washout_checkpoints for value in primary_window),
            "h17.primary_auc_window endpoints must be directly observed washout checkpoints",
        )

        for field in (
            "probe_ridge",
            "pure_threshold",
            "pure_margin",
            "probe_threshold",
            "probe_advantage",
            "late_stability_tolerance",
            "primary_effect_threshold",
            "primary_ci_lower_boundary",
            "primary_equivalence_margin",
            "primary_prevalence_threshold",
            "directional_effect_threshold",
            "directional_equivalence_margin",
            "bonferroni_confidence",
        ):
            value = get_path(config, f"h17.{field}", None)
            _require(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and 0.0 <= float(value) <= 1.0,
                f"h17.{field} must be finite and lie in [0,1]",
            )
        for field in (
            "pilot_minimum_joint_seeds",
            "minimum_prevalence_count",
            "probe_folds",
        ):
            value = get_path(config, f"h17.{field}", 0)
            _require(
                isinstance(value, int) and not isinstance(value, bool) and value >= 1,
                f"h17.{field} must be a positive integer",
            )
        _require(
            get_path(config, "h17.probe_folds", None) == 8
            and get_path(config, "h17.fold_digest", None) == _H17_FOLD_DIGEST
            and get_path(config, "h17.truth_table_control_digest", None)
            == _H17_CONTROL_DIGEST
            and get_path(config, "h17.truth_table_control_bits", None)
            == _H17_CONTROL_BITS
            and get_path(config, "h17.design_memo_sha256", None)
            == _H17_DESIGN_MEMO_SHA256,
            "H17 fold/control contract differs from the frozen exhaustive panel",
        )

        if not smoke:
            expected_mode = (
                "isolated_component_pilot"
                if pilot_only
                else "frozen_adaptive_posthoc_full_panel"
            )
            h17_frozen_values: dict[str, Any] = {
                "experiment.name": (
                    "counterbalanced_identical_evidence_order_pilot"
                    if pilot_only
                    else "counterbalanced_identical_evidence_order"
                ),
                "experiment.mode": expected_mode,
                "evaluation.bootstrap_samples": 4000,
                "evaluation.confidence": 0.95,
                "data.n_train": 9216,
                "data.n_validation": 64,
                "data.n_eval": 64,
                "data.q": 0.5,
                "model.bias": True,
                "model.nuisance_bits": 0,
                "train.batch_size": 288,
                "train.grad_clip": 1.0,
                "train.label_smoothing": 0.0,
                "h17.data_seed": _H17_DATA_SEED,
                "h17.stream_seed": _H17_STREAM_SEED,
                "h17.phase_seeds": dict(_H17_PHASE_SEEDS),
                "h17.rows_per_weight_unit": 96,
                "h17.weighted_strata": 96,
                "h17.examples_per_weight_unit_per_batch": 3,
                "h17.batches_per_presentation": 32,
                "h17.presentations": 8,
                "h17.component_steps": 256,
                "h17.washout_steps": 256,
                "h17.total_steps": 256 if pilot_only else 1024,
                "h17.component_checkpoints": list(_H17_CHECKPOINTS),
                "h17.washout_checkpoints": list(_H17_CHECKPOINTS),
                "h17.pilot_late_steps": [161, 222, 256],
                "h17.probe_folds": 8,
                "h17.probe_ridge": 0.001,
                "h17.pure_threshold": 0.90,
                "h17.pure_margin": 0.10,
                "h17.probe_threshold": 0.90,
                "h17.probe_advantage": 0.20,
                "h17.late_stability_tolerance": 0.02,
                "h17.pilot_minimum_joint_seeds": 2,
                "h17.primary_auc_window": [33, 128],
                "h17.primary_effect_threshold": 0.10,
                "h17.primary_ci_lower_boundary": 0.05,
                "h17.primary_equivalence_margin": 0.05,
                "h17.primary_prevalence_threshold": 0.05,
                "h17.minimum_prevalence_count": 15,
                "h17.directional_effect_threshold": 0.10,
                "h17.directional_equivalence_margin": 0.05,
                "h17.bonferroni_confidence": 0.9833333333333333,
            }
            for path, expected in h17_frozen_values.items():
                actual = get_path(config, path, None)
                _require(
                    type(actual) is type(expected) and actual == expected,
                    f"{path} must equal the frozen H17 value {expected!r}",
                )

            if "_case_index" not in config:
                h17_cases = get_path(config, "cases", [])
                case_field = "isolated_goal" if pilot_only else "schedule"
                expected_values = _H17_GOALS if pilot_only else _H17_SCHEDULES
                case_values = [
                    get_path(case, f"h17.{case_field}", "")
                    for case in h17_cases
                    if isinstance(case, Mapping)
                ]
                exact_overlays = all(
                    set(case) == {"h17"}
                    and isinstance(case["h17"], Mapping)
                    and set(case["h17"]) == {case_field}
                    for case in h17_cases
                    if isinstance(case, Mapping)
                )
                _require(
                    len(h17_cases) == len(expected_values)
                    and tuple(case_values) == expected_values
                    and exact_overlays,
                    "H17 cases must be the exact ordered pilot-goal or six-schedule grid",
                )

    return cautions


def expand_sweep(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expand optional case overlays and the Cartesian product in ``sweep``.

    ``cases`` is useful for coupled settings such as ``(mode=full,budget=full)``
    alongside a scalar-budget subspace sweep.  Each case is deep-merged before
    applying the same Cartesian sweep.
    """

    base = copy.deepcopy(dict(config))
    cases = list(get_path(base, "cases", [])) or [{}]
    base["cases"] = []
    cells: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        case_base = deep_merge(base, case)
        sweep = dict(get_path(case_base, "sweep", {}))
        keys = sorted(sweep)
        products = itertools.product(*(sweep[key] for key in keys)) if keys else [()]
        for values in products:
            cell = copy.deepcopy(case_base)
            cell["sweep"] = {}
            chosen = dict(zip(keys, values, strict=True))
            for key, value in chosen.items():
                set_path(cell, key, value)
            cell["_case_index"] = case_index if len(cases) > 1 else None
            cell["_sweep_values"] = chosen
            validate_config(cell)
            cells.append(cell)
    return cells


def canonical_config(config: Mapping[str, Any]) -> str:
    """Canonical JSON form used to derive deterministic run identifiers."""

    clean = {key: value for key, value in config.items() if not str(key).startswith("_")}
    return json.dumps(clean, sort_keys=True, separators=(",", ":"), default=str)
