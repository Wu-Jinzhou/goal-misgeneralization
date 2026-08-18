"""Configuration loading and deterministic sweep expansion for GoalZendo.

GoalZendo deliberately has an independent configuration contract from ForkWorld.
Changing an LLM experiment must not change the identity of completed toy runs.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

REQUIRED_LAUNCH_GUARDS: dict[str, str] = {
    "g01": "G00_NOT_PASSED__LEARNING_RATES_NOT_FROZEN",
    "g02": "G01_INFORMATIVE_CELL_AND_SETTINGS_NOT_FROZEN",
    "g01l": "G01_FULL_FT_COMPLETE__LORA_SECONDARY_NOT_UNLOCKED",
    "g01a": "G01_PRIMARY_RESULT_NOT_SELECTED_FOR_LONGITUDINAL_FOLLOWUP",
}

# A guarded study remains guarded if its YAML is copied to another path and
# its human-facing experiment label is changed.  These source-registered
# signatures cover the scientific configuration while deliberately ignoring
# operational storage/checkpoint fields and the optimizer settings selected by
# a preceding gate.  The literal values are populated below after the
# normalization helper is defined.
PROTECTED_GUARD_SIGNATURES: dict[str, tuple[str, str]] = {
    "9feaae82edf801aad4bd4a5b16f633be8dfa2dbd41b29601e7b61b564c763464": (
        "g01",
        "G00_NOT_PASSED__LEARNING_RATES_NOT_FROZEN",
    ),
    "fc950b7eb414647caedec8f4dac7acf16fd0781ae681640e2d5005a892cf2c0e": (
        "g02",
        "G01_INFORMATIVE_CELL_AND_SETTINGS_NOT_FROZEN",
    ),
    "cdebd29552320e3cef3040c3d5a562f471d7fea6716ca9b79bb4cd568f689ca5": (
        "g01l",
        "G01_FULL_FT_COMPLETE__LORA_SECONDARY_NOT_UNLOCKED",
    ),
    "9c07833d93e57cd73d619d346c0f8da4e35f922c5c2700038add0206d9ac3f61": (
        "g01a",
        "G01_PRIMARY_RESULT_NOT_SELECTED_FOR_LONGITUDINAL_FOLLOWUP",
    ),
}

_CONFIG_SCHEMA_KEYS: dict[str, frozenset[str]] = {
    "experiment": frozenset(
        {"id", "name", "status", "horizon_arm", "reproducibility_replica"}
    ),
    "run": frozenset(
        {
            "output_root",
            "seeds",
            "device",
            "resume",
            "save_checkpoints",
            "checkpoint_steps",
            "snapshot_steps",
            "launch_guard",
            "protocol_unlocked",
        }
    ),
    "data": frozenset(
        {
            "n_train",
            "n_validation",
            "n_eval_per_cell",
            "feature_count",
            "feature_names",
            "law_features",
            "sage_features",
            "distractor_features",
            "rule_family",
            "sage_rule_family",
            "q_p",
            "q_q",
            "error_geometry",
            "joint_error_rate",
            "conflict_diversity",
            "concentrated_unique_conflicts_per_tuple",
            "training_view",
            "renderer",
            "train_renderers",
            "heldout_renderers",
            "counterbalance",
        }
    ),
    "model": frozenset(
        {"name", "revision", "dtype", "chat_template", "trust_remote_code", "action_labels"}
    ),
    "update": frozenset(
        {"method", "rank", "alpha", "dropout", "bias", "target_modules"}
    ),
    "train": frozenset(
        {
            "algorithm",
            "steps",
            "batch_size",
            "gradient_accumulation_steps",
            "learning_rate",
            "weight_decay",
            "warmup_ratio",
            "grad_clip",
            "eval_steps",
            "max_sequence_length",
            "gradient_checkpointing",
            "deterministic_algorithms",
            "allow_tf32",
            "cublas_workspace_config",
            "parameter_finite_check_interval",
            "kl_coefficient",
            "entropy_coefficient",
            "samples_per_prompt",
            "reward_gradient_exponent",
        }
    ),
    "evaluation": frozenset(
        {
            "acquisition_threshold",
            "persistence",
            "causal_flip_threshold",
            "controller_margin",
            "save_predictions",
            "save_activations",
            "prompt_views",
            "causal_prompt_views",
            "causal_per_cell",
            "final_causal_per_cell",
            "batch_size",
            "mirror_pairs",
            "final_eval_per_cell",
            "bootstrap_samples",
            "confidence",
        }
    ),
}
_CONFIG_TOP_LEVEL_KEYS = frozenset(
    {"schema_version", "experiment", "run", "data", "model", "update", "train", "evaluation", "cases", "sweep"}
)
_CONFIG_METADATA_KEYS = frozenset(
    {"_config_path", "_declared_experiment_id", "_declared_launch_guard", "_sweep_values", "_case_index"}
)
_CONFIG_EXPANSION_PATHS = frozenset(
    f"{section}.{key}"
    for section, keys in _CONFIG_SCHEMA_KEYS.items()
    for key in keys
)
_EFFECTIVE_TRAINING_DATA_KEYS = frozenset(
    {
        "n_train",
        "feature_count",
        "law_features",
        "sage_features",
        "rule_family",
        "sage_rule_family",
        "q_p",
        "q_q",
        "error_geometry",
        "joint_error_rate",
        "conflict_diversity",
        "concentrated_unique_conflicts_per_tuple",
        "training_view",
    }
)
_EFFECTIVE_TRAINING_TRAIN_KEYS = frozenset(
    {"algorithm", "steps"}
)
_BOOLEAN_CONFIG_PATHS = frozenset(
    {
        "run.resume",
        "run.save_checkpoints",
        "run.protocol_unlocked",
        "data.counterbalance",
        "model.chat_template",
        "model.trust_remote_code",
        "train.gradient_checkpointing",
        "train.deterministic_algorithms",
        "train.allow_tf32",
        "evaluation.save_predictions",
        "evaluation.save_activations",
        "evaluation.mirror_pairs",
    }
)
_INTEGER_CONFIG_PATHS = frozenset(
    {
        "data.n_train",
        "data.n_validation",
        "data.n_eval_per_cell",
        "data.feature_count",
        "data.concentrated_unique_conflicts_per_tuple",
        "update.rank",
        "update.alpha",
        "train.steps",
        "train.batch_size",
        "train.gradient_accumulation_steps",
        "train.max_sequence_length",
        "train.parameter_finite_check_interval",
        "train.samples_per_prompt",
        "evaluation.persistence",
        "evaluation.causal_per_cell",
        "evaluation.final_causal_per_cell",
        "evaluation.batch_size",
        "evaluation.final_eval_per_cell",
        "evaluation.bootstrap_samples",
    }
)
_NUMBER_CONFIG_PATHS = frozenset(
    {
        "data.q_p",
        "data.q_q",
        "data.joint_error_rate",
        "update.dropout",
        "train.learning_rate",
        "train.weight_decay",
        "train.warmup_ratio",
        "train.grad_clip",
        "train.kl_coefficient",
        "train.entropy_coefficient",
        "train.reward_gradient_exponent",
        "evaluation.acquisition_threshold",
        "evaluation.causal_flip_threshold",
        "evaluation.controller_margin",
        "evaluation.confidence",
    }
)
_INTEGER_LIST_CONFIG_PATHS = frozenset(
    {
        "run.seeds",
        "run.checkpoint_steps",
        "run.snapshot_steps",
        "data.law_features",
        "data.sage_features",
        "data.distractor_features",
        "train.eval_steps",
    }
)
_STRING_LIST_CONFIG_PATHS = frozenset(
    {
        "data.feature_names",
        "data.train_renderers",
        "data.heldout_renderers",
        "model.action_labels",
        "update.target_modules",
        "evaluation.prompt_views",
        "evaluation.causal_prompt_views",
    }
)
_STRING_CONFIG_PATHS = frozenset(
    {
        "experiment.id",
        "experiment.name",
        "experiment.status",
        "experiment.horizon_arm",
        "experiment.reproducibility_replica",
        "run.output_root",
        "run.device",
        "run.launch_guard",
        "data.rule_family",
        "data.sage_rule_family",
        "data.error_geometry",
        "data.conflict_diversity",
        "data.training_view",
        "data.renderer",
        "model.name",
        "model.revision",
        "model.dtype",
        "update.method",
        "update.bias",
        "train.algorithm",
        "train.cublas_workspace_config",
    }
)


class ConfigError(ValueError):
    """Raised when an experiment specification is invalid or confounded."""


DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "experiment": {
        "id": "g00",
        "name": "engineering",
        "status": "exploratory",
    },
    "run": {
        "output_root": "artifacts-goalzendo",
        "seeds": [9001, 9002, 9003],
        "device": "auto",
        "resume": True,
        "save_checkpoints": False,
        # Resumable checkpoints are a sparse subset of dense evaluation
        # boundaries. Production configs enable them explicitly and must
        # include the final update.
        "checkpoint_steps": [],
        # Dense evaluation is independent of durable model retention.  At every
        # evaluation boundary the backend atomically replaces the single
        # resumable checkpoint; only these explicitly selected, weights-only
        # snapshots are retained in addition to it.
        "snapshot_steps": [],
    },
    "data": {
        "n_train": 10000,
        "n_validation": 1000,
        "n_eval_per_cell": 64,
        "feature_count": 17,
        "law_features": [0, 1, 2],
        "sage_features": [3, 4],
        "distractor_features": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        "rule_family": "parity",
        "sage_rule_family": "parity",
        "q_p": 0.99,
        "q_q": 0.9,
        "error_geometry": "independent",
        "conflict_diversity": "diverse",
        "training_view": "full",
        "renderer": "natural",
        "train_renderers": ["natural_1", "natural_2", "natural_3", "natural_4"],
        "heldout_renderers": ["natural_5", "natural_6"],
        "counterbalance": True,
    },
    "model": {
        "name": "Qwen/Qwen2.5-0.5B-Instruct",
        "revision": "main",
        "dtype": "bfloat16",
        "chat_template": True,
        "trust_remote_code": False,
        "action_labels": ["A", "B"],
    },
    "update": {"method": "full"},
    "train": {
        "algorithm": "sft",
        "steps": 1000,
        "batch_size": 10,
        "gradient_accumulation_steps": 5,
        "learning_rate": 0.00001,
        "weight_decay": 0.0,
        "warmup_ratio": 0.05,
        "grad_clip": 1.0,
        "eval_steps": [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768, 1000],
        "max_sequence_length": 640,
        "gradient_checkpointing": True,
        # Full-model BF16 runs are numerically path dependent unless the CUDA
        # backend is explicitly constrained. Confirmatory and reproducibility
        # studies enable this; exploratory sweeps may leave it off for speed.
        "deterministic_algorithms": False,
        "allow_tf32": False,
        "cublas_workspace_config": ":4096:8",
        "parameter_finite_check_interval": 0,
        "kl_coefficient": 0.02,
        "entropy_coefficient": 0.0,
        "samples_per_prompt": 4,
    },
    "evaluation": {
        "acquisition_threshold": 0.9,
        "persistence": 2,
        "save_predictions": True,
        "save_activations": False,
        "prompt_views": [
            "full",
            "audit_law_full",
            "audit_law_matched",
            "no_herald",
            "no_sage",
            "law_only",
        ],
        "causal_prompt_views": ["full"],
        "causal_per_cell": 16,
        "final_causal_per_cell": 64,
        "batch_size": 32,
        "mirror_pairs": True,
        "final_eval_per_cell": 512,
        "bootstrap_samples": 4000,
        "confidence": 0.95,
    },
    "cases": [],
    "sweep": {},
}


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
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
        raise ConfigError(f"invalid dotted configuration path: {dotted!r}")
    current = config
    for part in parts[:-1]:
        child = current.setdefault(part, {})
        if not isinstance(child, dict):
            raise ConfigError(f"cannot set {dotted!r}: {part!r} is not a mapping")
        current = child
    current[parts[-1]] = copy.deepcopy(value)


def parse_override(text: str) -> tuple[str, Any]:
    if "=" not in text:
        raise ConfigError(f"override must have path=value form: {text!r}")
    path, raw = text.split("=", 1)
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid override value in {text!r}: {exc}") from exc
    return path, value


def _effective_training_cell_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    """Project one cell onto conservative guarded-study family identifiers."""

    data_source = dict(get_path(config, "data", {}) or {})
    data = {
        key: copy.deepcopy(value)
        for key, value in data_source.items()
        if key in _EFFECTIVE_TRAINING_DATA_KEYS
    }
    if str(data.get("rule_family", "")) in {"parity", "majority", "conjunction"}:
        data["law_features"] = sorted(set(data.get("law_features", [])))
    if str(data.get("sage_rule_family", "")) in {
        "parity",
        "majority",
        "conjunction",
    }:
        data["sage_features"] = sorted(set(data.get("sage_features", [])))
    if str(data.get("error_geometry", "")) != "specified":
        data.pop("joint_error_rate", None)

    update_source = dict(get_path(config, "update", {}) or {})
    update_method = str(update_source.get("method", "")).lower()
    update = {"method": update_method}

    train_source = dict(get_path(config, "train", {}) or {})
    train = {
        key: copy.deepcopy(value)
        for key, value in train_source.items()
        if key in _EFFECTIVE_TRAINING_TRAIN_KEYS
    }
    model_source = dict(get_path(config, "model", {}) or {})
    model = {
        key: copy.deepcopy(model_source.get(key))
        for key in ("name", "revision")
    }
    return _canonicalize_effective_numbers(
        {
        "data": data,
        "model": model,
        "update": update,
        "train": train,
        }
    )


def _canonicalize_effective_numbers(value: Any) -> Any:
    """Match the backend's numeric casts while preserving Boolean semantics."""

    if isinstance(value, Mapping):
        return {str(key): _canonicalize_effective_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_canonicalize_effective_numbers(item) for item in value]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return copy.deepcopy(value)
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ConfigError("effective training configuration contains a non-finite number")
    return 0.0 if normalized == 0.0 else normalized


def protected_guard_signature(config: Mapping[str, Any]) -> str:
    """Hash a conservative source-registered family for a guarded study.

    Defining data geometry, model revision, update family, algorithm, horizon,
    and seed set remain bound.  All other optimizer, rendering, evaluation,
    storage, or adapter details are conservatively kept behind the same guard,
    even when they would define a genuine variant.  Sweep/case cells are
    projected after overrides and deduplicated.  This fail-closed breadth
    avoids trying to prove semantic equivalence for every backend default.
    """

    sweep = get_path(config, "sweep", {}) or {}
    cases = get_path(config, "cases", []) or []
    if not isinstance(sweep, Mapping) or not isinstance(cases, list):
        raise ConfigError("sweep and cases must be structured before guard signature creation")
    axes: list[tuple[str, list[Any]]] = []
    for raw_path, raw_values in sorted(sweep.items(), key=lambda item: str(item[0])):
        if not isinstance(raw_values, list):
            raise ConfigError(f"sweep axis {raw_path!r} must be a list")
        axes.append((str(raw_path), raw_values))
    products: Iterable[tuple[Any, ...]] = (
        itertools.product(*(values for _, values in axes)) if axes else [()]
    )
    coupled = cases or [{}]
    semantic_cells: set[str] = set()
    for product_values in products:
        for case in coupled:
            if not isinstance(case, Mapping):
                raise ConfigError("each case must be a mapping")
            cell = copy.deepcopy(dict(config))
            for (path, _values), value in zip(axes, product_values, strict=True):
                set_path(cell, path, value)
            for path, value in case.items():
                set_path(cell, str(path), value)
            semantic_cells.add(
                json.dumps(
                    _effective_training_cell_payload(cell),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                )
            )
    payload = {
        "schema_version": get_path(config, "schema_version", None),
        "seeds": sorted(set(get_path(config, "run.seeds", []))),
        "training_cells": [json.loads(item) for item in sorted(semantic_cells)],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_yaml(source: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    source = source.resolve()
    if source in stack:
        chain = " -> ".join(str(path) for path in (*stack, source))
        raise ConfigError(f"cyclic config extends chain: {chain}")
    if not source.is_file():
        raise ConfigError(f"configuration file does not exist: {source}")
    loaded = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, Mapping):
        raise ConfigError(f"top-level YAML object must be a mapping: {source}")
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
    source = Path(path)
    config = deep_merge(DEFAULT_CONFIG, _load_yaml(source))
    declared_experiment_id = str(get_path(config, "experiment.id", ""))
    declared_launch_guard = get_path(config, "run.launch_guard", None)
    for item in overrides:
        dotted, value = parse_override(item)
        set_path(config, dotted, value)
    config["_config_path"] = str(source.resolve())
    config["_declared_experiment_id"] = declared_experiment_id
    config["_declared_launch_guard"] = declared_launch_guard
    validate_config(config)
    return config


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _validate_config_value_type(path: str, value: Any) -> None:
    if path in _BOOLEAN_CONFIG_PATHS:
        valid = isinstance(value, bool)
        expected = "a Boolean"
    elif path in _INTEGER_CONFIG_PATHS:
        valid = isinstance(value, int) and not isinstance(value, bool)
        expected = "an integer"
    elif path in _NUMBER_CONFIG_PATHS:
        valid = (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
        ) or (path == "train.reward_gradient_exponent" and value is None)
        expected = "a number or registered null"
    elif path in _INTEGER_LIST_CONFIG_PATHS:
        valid = isinstance(value, list) and all(
            isinstance(item, int) and not isinstance(item, bool) for item in value
        )
        expected = "a list that must contain integers"
    elif path in _STRING_LIST_CONFIG_PATHS:
        valid = isinstance(value, list) and all(isinstance(item, str) for item in value)
        expected = "a list of strings"
    elif path in _STRING_CONFIG_PATHS:
        valid = isinstance(value, str) or (path == "run.launch_guard" and value is None)
        expected = "a string"
    else:  # pragma: no cover - source registry completeness is tested directly.
        raise ConfigError(f"configuration path has no type contract: {path!r}")
    _require(valid, f"{path} must be {expected}")


def _validate_config_schema(config: Mapping[str, Any]) -> None:
    unknown_top = {
        str(key)
        for key in config
        if str(key) not in _CONFIG_TOP_LEVEL_KEYS and str(key) not in _CONFIG_METADATA_KEYS
    }
    _require(
        not unknown_top,
        f"unknown top-level configuration keys: {sorted(unknown_top)}",
    )
    for section, allowed_keys in _CONFIG_SCHEMA_KEYS.items():
        value = config.get(section, {})
        _require(isinstance(value, Mapping), f"{section} must be a mapping")
        unknown = {str(key) for key in value if str(key) not in allowed_keys}
        _require(
            not unknown,
            f"unknown {section} configuration keys: {sorted(unknown)}",
        )
        for key, item in value.items():
            _validate_config_value_type(f"{section}.{key}", item)

    cases = get_path(config, "cases", []) or []
    sweep = get_path(config, "sweep", {}) or {}
    _require(isinstance(cases, list), "cases must be a list")
    _require(isinstance(sweep, Mapping), "sweep must be a mapping")
    for case in cases:
        _require(isinstance(case, Mapping), "each case must be a mapping")
        for raw_path in case:
            path = str(raw_path)
            _require(
                isinstance(raw_path, str) and path in _CONFIG_EXPANSION_PATHS,
                f"unknown case configuration path: {path!r}",
            )
            _validate_config_value_type(path, case[raw_path])
    for raw_path, raw_values in sweep.items():
        path = str(raw_path)
        _require(
            isinstance(raw_path, str) and path in _CONFIG_EXPANSION_PATHS,
            f"unknown sweep configuration path: {path!r}",
        )
        _require(isinstance(raw_values, list), f"sweep axis {path!r} must be a list")
        for item in raw_values:
            _validate_config_value_type(path, item)


def _is_exact_rate(rate: float, count: int) -> bool:
    return abs(rate * count - round(rate * count)) < 1e-8


def validate_config(config: Mapping[str, Any]) -> list[str]:
    """Validate shared invariants and return cautions saved with each run."""

    _validate_config_schema(config)
    cautions: list[str] = []
    schema_version = get_path(config, "schema_version", None)
    _require(
        isinstance(schema_version, int)
        and not isinstance(schema_version, bool)
        and schema_version == 1,
        "schema_version must be the integer 1",
    )
    status = str(get_path(config, "experiment.status", ""))
    _require(status in {"prospective", "adaptive", "exploratory"}, "invalid experiment.status")
    experiment_id = str(get_path(config, "experiment.id", ""))
    declared_experiment_id = str(config.get("_declared_experiment_id", experiment_id))
    signature = protected_guard_signature(config)
    protected_registration = PROTECTED_GUARD_SIGNATURES.get(signature)
    if protected_registration is not None:
        protected_id, protected_guard = protected_registration
        _require(
            experiment_id == protected_id,
            "a source-registered guarded experiment cannot be copied or relabeled",
        )
        _require(
            get_path(config, "run.launch_guard", None) == protected_guard,
            "a source-registered guarded experiment requires its registered launch_guard",
        )
    if declared_experiment_id in REQUIRED_LAUNCH_GUARDS:
        required_guard = REQUIRED_LAUNCH_GUARDS[declared_experiment_id]
        _require(
            experiment_id == declared_experiment_id,
            "a registered guarded experiment cannot override experiment.id",
        )
        _require(
            get_path(config, "run.launch_guard", None) == required_guard,
            "a registered guarded experiment cannot remove or alter run.launch_guard",
        )
        _require(
            config.get("_declared_launch_guard", required_guard) == required_guard,
            "registered guarded experiment has an unexpected declared launch guard",
        )

    seeds = get_path(config, "run.seeds", [])
    _require(
        isinstance(seeds, list) and bool(seeds),
        "run.seeds must be a non-empty list",
    )
    _require(all(isinstance(seed, int) for seed in seeds), "run.seeds must contain integers")
    _require(len(set(seeds)) == len(seeds), "run.seeds must be unique")

    n_train = int(get_path(config, "data.n_train", 0))
    n_eval = int(get_path(config, "data.n_eval_per_cell", 0))
    _require(n_train >= 8, "data.n_train must be at least 8")
    _require(n_eval > 0, "data.n_eval_per_cell must be positive")
    for name in ("q_p", "q_q"):
        rate = float(get_path(config, f"data.{name}", -1))
        _require(0.5 <= rate <= 1.0, f"data.{name} must be in [0.5, 1]")
        _require(
            _is_exact_rate(rate, n_train),
            f"data.{name}={rate} cannot be represented exactly by n_train={n_train}",
        )

    feature_count = int(get_path(config, "data.feature_count", 0))
    law_features = [int(item) for item in get_path(config, "data.law_features", [])]
    sage_features = [int(item) for item in get_path(config, "data.sage_features", [])]
    _require(feature_count >= 5, "data.feature_count must be at least 5")
    _require(
        bool(law_features) and bool(sage_features),
        "Law and Sage feature sets must be non-empty",
    )
    _require(len(set(law_features)) == len(law_features), "Law features must be unique")
    _require(len(set(sage_features)) == len(sage_features), "Sage features must be unique")
    _require(set(law_features).isdisjoint(sage_features), "Law and Sage features must be disjoint")
    _require(
        all(0 <= item < feature_count for item in (*law_features, *sage_features)),
        "feature indices must lie within data.feature_count",
    )
    families = {"literal", "parity", "majority", "conjunction", "multiplexer"}
    _require(str(get_path(config, "data.rule_family")) in families, "invalid Law rule family")
    _require(str(get_path(config, "data.sage_rule_family")) in families, "invalid Sage rule family")

    geometry = str(get_path(config, "data.error_geometry", ""))
    _require(geometry in {"independent", "nested", "disjoint", "specified"}, "invalid error geometry")
    error_p = 1.0 - float(get_path(config, "data.q_p"))
    error_q = 1.0 - float(get_path(config, "data.q_q"))
    if geometry == "disjoint":
        _require(error_p + error_q <= 1.0 + 1e-9, "disjoint errors are infeasible")
    if geometry == "specified":
        joint = float(get_path(config, "data.joint_error_rate", -1.0))
        lower = max(0.0, error_p + error_q - 1.0)
        upper = min(error_p, error_q)
        _require(lower - 1e-9 <= joint <= upper + 1e-9, "specified joint error is infeasible")
        _require(_is_exact_rate(joint, n_train), "joint_error_rate is not exactly representable")

    algorithm = str(get_path(config, "train.algorithm", ""))
    _require(
        algorithm in {
            "sft",
            "trajectory_sft",
            "outcome_rl",
            "expected_outcome_rl",
            "tempered_outcome_control",
            "logprob_outcome_control",
        },
        "invalid train.algorithm",
    )
    reward_gradient_exponent = get_path(
        config,
        "train.reward_gradient_exponent",
        None,
    )
    exponent_is_number = isinstance(reward_gradient_exponent, (int, float)) and not isinstance(
        reward_gradient_exponent,
        bool,
    )
    if algorithm == "tempered_outcome_control":
        _require(
            exponent_is_number and 0.0 < float(reward_gradient_exponent) < 1.0,
            "tempered_outcome_control requires train.reward_gradient_exponent strictly in (0, 1)",
        )
    elif algorithm == "logprob_outcome_control":
        _require(
            exponent_is_number and float(reward_gradient_exponent) == 0.0,
            "logprob_outcome_control requires train.reward_gradient_exponent=0",
        )
    else:
        _require(
            reward_gradient_exponent is None,
            "train.reward_gradient_exponent is only valid for a power-gradient control algorithm",
        )
    steps = int(get_path(config, "train.steps", 0))
    _require(steps > 0, "train.steps must be positive")
    eval_steps = [int(item) for item in get_path(config, "train.eval_steps", [])]
    _require(bool(eval_steps), "train.eval_steps must be non-empty")
    _require(eval_steps == sorted(set(eval_steps)), "train.eval_steps must be sorted and unique")
    _require(eval_steps[0] == 0, "train.eval_steps must include the untrained checkpoint 0")
    _require(eval_steps[-1] <= steps, "train.eval_steps cannot exceed train.steps")
    finite_check_interval = get_path(config, "train.parameter_finite_check_interval", 0)
    _require(
        isinstance(finite_check_interval, int)
        and not isinstance(finite_check_interval, bool)
        and finite_check_interval >= 0,
        "train.parameter_finite_check_interval must be a non-negative integer",
    )
    deterministic_algorithms = get_path(config, "train.deterministic_algorithms", False)
    allow_tf32 = get_path(config, "train.allow_tf32", False)
    _require(
        isinstance(deterministic_algorithms, bool),
        "train.deterministic_algorithms must be Boolean",
    )
    _require(isinstance(allow_tf32, bool), "train.allow_tf32 must be Boolean")
    _require(
        not deterministic_algorithms or not allow_tf32,
        "deterministic algorithms require train.allow_tf32=false",
    )
    workspace_config = get_path(config, "train.cublas_workspace_config", ":4096:8")
    _require(
        workspace_config in {":4096:8", ":16:8"},
        "train.cublas_workspace_config must be :4096:8 or :16:8",
    )

    update_method = str(get_path(config, "update.method", "")).lower()
    _require(update_method in {"full", "lora", "frozen"}, "invalid update.method")
    _require(update_method != "frozen", "training runs cannot use update.method=frozen")

    save_checkpoints = get_path(config, "run.save_checkpoints", False)
    _require(isinstance(save_checkpoints, bool), "run.save_checkpoints must be Boolean")
    checkpoint_steps_raw = get_path(config, "run.checkpoint_steps", [])
    _require(isinstance(checkpoint_steps_raw, list), "run.checkpoint_steps must be a list")
    _require(
        all(isinstance(item, int) and not isinstance(item, bool) for item in checkpoint_steps_raw),
        "run.checkpoint_steps must contain integers",
    )
    checkpoint_steps = [int(item) for item in checkpoint_steps_raw]
    _require(
        checkpoint_steps == sorted(set(checkpoint_steps)),
        "run.checkpoint_steps must be sorted and unique",
    )
    _require(0 not in checkpoint_steps, "run.checkpoint_steps cannot include step zero")
    valid_durable_steps = set((*eval_steps, steps))
    _require(
        all(item in valid_durable_steps for item in checkpoint_steps),
        "run.checkpoint_steps must be scheduled evaluation steps",
    )
    _require(
        (not save_checkpoints and not checkpoint_steps)
        or (save_checkpoints and bool(checkpoint_steps) and checkpoint_steps[-1] == steps),
        "enabled resumable checkpoints must include the final training step",
    )
    snapshot_steps_raw = get_path(config, "run.snapshot_steps", [])
    _require(isinstance(snapshot_steps_raw, list), "run.snapshot_steps must be a list")
    _require(
        all(isinstance(item, int) and not isinstance(item, bool) for item in snapshot_steps_raw),
        "run.snapshot_steps must contain integers",
    )
    snapshot_steps = [int(item) for item in snapshot_steps_raw]
    _require(
        snapshot_steps == sorted(set(snapshot_steps)),
        "run.snapshot_steps must be sorted and unique",
    )
    _require(
        all(item in valid_durable_steps for item in snapshot_steps),
        "run.snapshot_steps must be scheduled evaluation steps",
    )
    _require(
        save_checkpoints or not snapshot_steps,
        "run.snapshot_steps requires run.save_checkpoints=true",
    )

    renderer = str(get_path(config, "data.renderer", ""))
    _require(renderer in {"natural", "nonce"}, "data.renderer must be natural or nonce")
    diversity = str(get_path(config, "data.conflict_diversity", ""))
    _require(diversity in {"diverse", "concentrated"}, "invalid conflict diversity")
    unique_conflicts_per_tuple = get_path(
        config,
        "data.concentrated_unique_conflicts_per_tuple",
        None,
    )
    if unique_conflicts_per_tuple is not None:
        _require(
            unique_conflicts_per_tuple > 0,
            "data.concentrated_unique_conflicts_per_tuple must be positive",
        )
    training_view = str(get_path(config, "data.training_view", ""))
    _require(
        training_view
        in {
            "full",
            "audit_law_full",
            "audit_law_matched",
            "law_only",
            "sage_only",
            "herald_only",
            "no_signal",
            "surface_only",
        },
        "invalid data.training_view",
    )
    if not bool(get_path(config, "data.counterbalance", False)):
        cautions.append("master names, side labels, and channel order are not counterbalanced")
    if len(seeds) < 3:
        cautions.append("fewer than three independent training seeds; inference is descriptive")
    return cautions


def canonical_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-normalized scientific configuration."""

    return json.loads(
        json.dumps(
            {key: value for key, value in config.items() if not str(key).startswith("_")},
            sort_keys=True,
        )
    )


def expand_sweep(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expand Cartesian ``sweep`` axes and coupled ``cases`` deterministically."""

    sweep = get_path(config, "sweep", {}) or {}
    cases = get_path(config, "cases", []) or []
    _require(isinstance(sweep, Mapping), "sweep must be a mapping")
    _require(isinstance(cases, list), "cases must be a list")
    axes = sorted((str(path), list(values)) for path, values in sweep.items())
    forbidden_expansion_paths = {
        "experiment.id",
        "experiment.name",
        "experiment.status",
        "run.launch_guard",
        "run.protocol_unlocked",
        "run.output_root",
        "run.seeds",
    }
    expansion_paths = [path for path, _values in axes]
    expansion_paths.extend(
        str(path)
        for case in cases
        if isinstance(case, Mapping)
        for path in case
    )
    for path in expansion_paths:
        parts = path.split(".")
        _require(
            path not in forbidden_expansion_paths
            and bool(parts)
            and all(part and not part.startswith("_") for part in parts),
            f"sweep/case path {path!r} may not alter experiment identity or launch controls",
        )
    for path, axis_values in axes:
        _require(bool(axis_values), f"sweep axis {path!r} must be non-empty")
    products: Iterable[tuple[Any, ...]] = (
        itertools.product(*(values for _, values in axes)) if axes else [()]
    )
    coupled = cases or [{}]
    cells: list[dict[str, Any]] = []
    for product_values in products:
        for case_index, case in enumerate(coupled):
            _require(isinstance(case, Mapping), "each case must be a mapping")
            cell = copy.deepcopy(dict(config))
            cell.pop("sweep", None)
            cell.pop("cases", None)
            assigned: dict[str, Any] = {}
            for (path, _), value in zip(axes, product_values, strict=True):
                set_path(cell, path, value)
                assigned[path] = copy.deepcopy(value)
            for path, value in case.items():
                set_path(cell, str(path), value)
                assigned[str(path)] = copy.deepcopy(value)
            cell["_sweep_values"] = assigned
            cell["_case_index"] = case_index if cases else None
            validate_config(cell)
            cells.append(cell)
    return cells


def smoke_config(config: Mapping[str, Any]) -> dict[str, Any]:
    smoke = copy.deepcopy(dict(config))
    set_path(smoke, "run.seeds", [991])
    set_path(smoke, "data.n_train", 40)
    set_path(smoke, "data.n_validation", 16)
    set_path(smoke, "data.n_eval_per_cell", 2)
    set_path(smoke, "data.q_p", 0.95)
    set_path(smoke, "data.q_q", 0.9)
    set_path(smoke, "train.steps", 2)
    set_path(smoke, "train.eval_steps", [0, 1, 2])
    set_path(smoke, "run.save_checkpoints", True)
    set_path(smoke, "run.checkpoint_steps", [1, 2])
    set_path(smoke, "run.snapshot_steps", [2])
    # Keep every evaluation subset within the deliberately tiny smoke panel.
    # The concrete backend fails closed instead of silently resampling a cell.
    set_path(smoke, "evaluation.causal_per_cell", 2)
    set_path(smoke, "evaluation.final_causal_per_cell", 2)
    set_path(smoke, "evaluation.final_eval_per_cell", 2)
    smoke["sweep"] = {}
    smoke["cases"] = []
    validate_config(smoke)
    return smoke
