#!/usr/bin/env python3
"""Prospective analysis for the 24-run Qwen3.5 hidden-Law panel.

The command is deliberately outcome-agnostic: a complete valid panel always
produces the same collection of registered estimands and descriptive tables.
It never selects analyses, endpoints, or wording from the observed values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, cast

import yaml  # type: ignore[import-untyped]

from goalzendo_hidden_law.artifacts import implementation_provenance, read_json, verify_completed_run
from goalzendo_hidden_law.config import (
    HiddenLawCondition,
    build_hidden_law_plan,
    canonical_digest,
    load_hidden_law_config,
    scientific_config,
)
from goalzendo_hidden_law.experiment import reference_inquiry
from goalzendo_hidden_law.game import ProductionBank, build_production_bank, query_information
from goalzendo_interactive.schema import scene_at

ROOT = Path(__file__).resolve().parents[1]
MAIN_CONFIG = ROOT / "configs" / "goalzendo" / "qwen35_hidden_law_finite_choice.yaml"
PROTOCOL = ROOT / "docs" / "goalzendo" / "protocols" / "qwen35-hidden-law-finite-choice.md"

SEEDS = (23011, 23013, 23017, 23019, 23023, 23031)
MODEL_REVISIONS = {
    "Qwen/Qwen3.5-0.8B": "2fc06364715b967f1860aea9cf38778875588b17",
    "Qwen/Qwen3.5-2B": "15852e8c16360a2fea060d615a32b45270f8a8fc",
}
MODEL_PARAMETER_COUNTS = {
    "Qwen/Qwen3.5-0.8B": 752_393_024,
    "Qwen/Qwen3.5-2B": 1_881_825_088,
}
PINNED_DEPENDENCIES = {
    "transformers": "5.15.0",
    "tokenizers": "0.22.2",
    "peft": "0.20.0",
    "accelerate": "1.14.0",
    "huggingface_hub": "1.5.0",
    "safetensors": "0.8.0",
}
MODELS = tuple(MODEL_REVISIONS)
ALGORITHMS = ("process_sft", "outcome_rl")
EVAL_STEPS = (0, 8, 32, 128)
FINAL_STEP = 128
FINAL_VIEWS = ("active", "oracle_query", "no_query")
EVIDENCE_CELLS = (
    ("perfect", "perfect"),
    ("perfect", "noisy"),
    ("noisy", "perfect"),
    ("noisy", "noisy"),
)
ANALYSIS_ROLES = ("Y", "P", "Q", "R")
INTERVENTION_TARGETS = (*ANALYSIS_ROLES, "distractor")
PRIMARY_OUTCOMES = (
    "information_fraction",
    "exact_rule_recovery",
    "law_control_margin",
)
CAUSAL_COMPANION = "causal_law_control_margin"
ALL_ENDPOINTS = (*PRIMARY_OUTCOMES, CAUSAL_COMPANION)
BOOTSTRAP_RESAMPLES = len(SEEDS) ** len(SEEDS)
SIGN_FLIP_ASSIGNMENTS = 2 ** len(SEEDS)
TOKEN_COUNT_DEFINITION = (
    "sum of chat-rendered prompt token lengths for every scored call; "
    "padding and action-continuation tokens excluded"
)

# Frozen before any scientific execution or outcome inspection.
REGISTERED_CONFIG_FILE_SHA256 = "a2e31815f3fbe1d65fcd64654f8fb9e51a2d950cf8a2985cc5d90121bad3e53a"
REGISTERED_SCIENTIFIC_CONFIG_DIGEST = "d61cad537e4ae3ea439250bc5e7e2466405e5331be251f6b8f15d0de8683207b"
REGISTERED_IMPLEMENTATION_FINGERPRINT = "9cd679289fc22ceeb5fe1a9cbc2ca134051faf8491b9fa478be9b13ca81dd087"
REGISTERED_PROTOCOL_SHA256 = "f4752515e3848a98452b89a3f753e8310dc45c131cd33244dc56fa696125883e"


class Qwen35HiddenLawAnalysisError(ValueError):
    """Raised when the frozen panel or a registered estimand is not identified."""


@dataclass(frozen=True, slots=True)
class LoadedRun:
    """One sealed run after artifact-level identity checks."""

    condition: HiddenLawCondition
    path: Path
    identity: Mapping[str, Any]
    implementation: Mapping[str, Any]
    bank_manifest: Mapping[str, Any]
    model_manifest: Mapping[str, Any]
    summary: Mapping[str, Any]
    metrics: tuple[Mapping[str, Any], ...]
    transcripts: tuple[Mapping[str, Any], ...]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if type(value) is not dict:
        raise Qwen35HiddenLawAnalysisError(f"{label} must be a JSON object")
    return cast(Mapping[str, Any], value)


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise Qwen35HiddenLawAnalysisError(
            f"{label} fields differ from the registered schema; missing={missing}, extra={extra}"
        )


def _jsonl(path: Path) -> tuple[Mapping[str, Any], ...]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise Qwen35HiddenLawAnalysisError(f"cannot read sealed JSONL: {path}") from exc
    if payload and not payload.endswith(b"\n"):
        raise Qwen35HiddenLawAnalysisError(f"sealed JSONL lacks its final newline: {path}")
    rows: list[Mapping[str, Any]] = []
    for line_number, raw in enumerate(payload.splitlines(), start=1):
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Qwen35HiddenLawAnalysisError(f"invalid sealed JSONL at {path}:{line_number}") from exc
        rows.append(_mapping(value, f"{path}:{line_number}"))
    identifiers = [row.get("record_id") for row in rows]
    if any(type(value) is not str or not value for value in identifiers):
        raise Qwen35HiddenLawAnalysisError(f"sealed JSONL has a missing record_id: {path}")
    if len(set(identifiers)) != len(identifiers):
        raise Qwen35HiddenLawAnalysisError(f"sealed JSONL has duplicate record_id values: {path}")
    return tuple(rows)


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Qwen35HiddenLawAnalysisError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise Qwen35HiddenLawAnalysisError(f"{label} must be finite")
    return result


def _unit(value: object, label: str) -> float:
    result = _finite(value, label)
    if not 0.0 <= result <= 1.0:
        raise Qwen35HiddenLawAnalysisError(f"{label} must lie in [0, 1]")
    return result


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise Qwen35HiddenLawAnalysisError(f"{label} must be an integer >= {minimum}")
    return value


def _boolean(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise Qwen35HiddenLawAnalysisError(f"{label} must be Boolean")
    return value


def _close(left: float, right: float, label: str, *, tolerance: float = 1e-12) -> None:
    if not math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance):
        raise Qwen35HiddenLawAnalysisError(f"{label} is internally inconsistent: {left!r} versus {right!r}")


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise Qwen35HiddenLawAnalysisError("cannot average an empty registered cell")
    if not all(math.isfinite(value) for value in values):
        raise Qwen35HiddenLawAnalysisError("registered cell contains a non-finite value")
    return math.fsum(values) / len(values)


def _expected_plan(config: Mapping[str, Any]) -> tuple[HiddenLawCondition, ...]:
    try:
        plan = build_hidden_law_plan(dict(config))
    except (TypeError, ValueError) as exc:
        raise Qwen35HiddenLawAnalysisError(f"invalid registered config: {exc}") from exc
    expected = {(seed, model, algorithm) for seed, model, algorithm in product(SEEDS, MODELS, ALGORITHMS)}
    observed = {(item.seed, item.model_name, item.algorithm) for item in plan}
    if len(plan) != 24 or observed != expected:
        raise Qwen35HiddenLawAnalysisError("config does not define the exact registered 24-run panel")
    if tuple(config["seeds"]) != SEEDS:
        raise Qwen35HiddenLawAnalysisError("config seed order differs from registration")
    if tuple(item["name"] for item in config["models"]) != MODELS:
        raise Qwen35HiddenLawAnalysisError("config model order differs from registration")
    if tuple(item["name"] for item in config["algorithms"]) != ALGORITHMS:
        raise Qwen35HiddenLawAnalysisError("config algorithm order differs from registration")
    if tuple(config["training"]["evaluation_steps"]) != EVAL_STEPS:
        raise Qwen35HiddenLawAnalysisError("config evaluation steps differ from registration")
    if tuple(config["evaluation"]["final_views"]) != FINAL_VIEWS:
        raise Qwen35HiddenLawAnalysisError("config final views differ from registration")
    return plan


def _load_resolved_config(path: Path) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise Qwen35HiddenLawAnalysisError(f"cannot read resolved config: {path}") from exc
    return _mapping(value, str(path))


def _expected_identity(
    condition: HiddenLawCondition,
    implementation_fingerprint: str,
) -> dict[str, Any]:
    return {
        "schema": "goalzendo.hidden_law_run_identity",
        "schema_version": 1,
        "run_id": condition.run_id,
        "plan_key": condition.plan_key,
        "condition": condition.as_obj(),
        "implementation_fingerprint": implementation_fingerprint,
    }


def _validate_model_manifest(
    condition: HiddenLawCondition,
    manifest: Mapping[str, Any],
) -> None:
    """Bind a sealed run to the requested and resolved Qwen3.5 model."""

    label = f"{condition.run_id} model manifest"
    expected = {
        "requested_model": condition.model_name,
        "requested_revision": condition.model_revision,
        "requested_dtype": condition.model_dtype,
        "action_labels": ["A", "B"],
        "finite_action_alphabets": {
            "binary": ["A", "B"],
            "candidate": ["A", "B", "C", "D"],
            "query": ["A", "B", "C", "D", "E", "F", "G", "H", "I"],
        },
        "gradient_checkpointing": True,
        "use_cache": False,
        "full_model_update": True,
        "trainable_parameter_dtype_counts": {"torch.bfloat16": MODEL_PARAMETER_COUNTS[condition.model_name]},
    }
    mismatched = [key for key, value in expected.items() if manifest.get(key) != value]
    if mismatched:
        raise Qwen35HiddenLawAnalysisError(f"{label} differs from registration: {', '.join(mismatched)}")
    for key in ("resolved_revision", "tokenizer_resolved_revision"):
        resolved = manifest.get(key)
        if resolved is not None and resolved != condition.model_revision:
            raise Qwen35HiddenLawAnalysisError(f"{label} {key} differs from the requested pinned revision")
    model_class = manifest.get("model_class")
    if model_class != "transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5ForCausalLM":
        raise Qwen35HiddenLawAnalysisError(f"{label} does not record the Qwen3_5 causal-LM class")
    parameter_count = _integer(manifest.get("parameter_count"), f"{label} parameter_count", minimum=1)
    trainable_count = _integer(
        manifest.get("trainable_parameter_count"),
        f"{label} trainable_parameter_count",
        minimum=1,
    )
    if parameter_count != trainable_count or parameter_count != MODEL_PARAMETER_COUNTS[condition.model_name]:
        raise Qwen35HiddenLawAnalysisError(f"{label} is not a full-model update")
    tokenizer_class = manifest.get("tokenizer_class")
    if (
        manifest.get("tokenizer_name_or_path") != condition.model_name
        or type(tokenizer_class) is not str
        or not tokenizer_class
    ):
        raise Qwen35HiddenLawAnalysisError(f"{label} tokenizer identity is invalid")
    dependencies = _mapping(manifest.get("dependency_versions"), f"{label} dependency_versions")
    if (
        any(dependencies.get(key) != value for key, value in PINNED_DEPENDENCIES.items())
        or str(dependencies.get("torch", "")).split("+", 1)[0] != "2.8.0"
        or dependencies.get("torch_cuda") != "12.8"
    ):
        raise Qwen35HiddenLawAnalysisError(f"{label} runtime stack differs from registration")
    action_tokens = manifest.get("action_token_ids")
    if (
        not isinstance(action_tokens, list)
        or len(action_tokens) != 2
        or any(
            not isinstance(tokens, list)
            or not tokens
            or any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in tokens)
            for tokens in action_tokens
        )
    ):
        raise Qwen35HiddenLawAnalysisError(f"{label} action-token encodings are invalid")
    first, second = (tuple(tokens) for tokens in action_tokens)
    if first == second[: len(first)] or second == first[: len(second)]:
        raise Qwen35HiddenLawAnalysisError(f"{label} action-token encodings are not prefix-free")


def load_exact_runs(
    artifacts: Path,
    config: Mapping[str, Any],
    *,
    registered_implementation_fingerprint: str,
) -> tuple[LoadedRun, ...]:
    """Load exactly the registered sealed runs; reject incomplete or extra run dirs."""

    plan = _expected_plan(config)
    expected_by_id = {condition.run_id: condition for condition in plan}
    try:
        children = tuple(path for path in artifacts.iterdir() if path.is_dir())
    except OSError as exc:
        raise Qwen35HiddenLawAnalysisError(f"cannot inspect artifact root: {artifacts}") from exc
    run_like = {
        path.name: path
        for path in children
        if (path / "identity.json").exists()
        or (path / "COMPLETE").exists()
        or (path / "completion.json").exists()
    }
    missing = sorted(set(expected_by_id) - set(run_like))
    unexpected = sorted(set(run_like) - set(expected_by_id))
    if missing or unexpected:
        raise Qwen35HiddenLawAnalysisError(
            "artifact root differs from the exact 24-run plan: "
            f"{len(missing)} missing, {len(unexpected)} unexpected"
        )

    loaded: list[LoadedRun] = []
    expected_config = dict(config)
    for condition in plan:
        path = run_like[condition.run_id]
        try:
            verify_completed_run(path)
        except Exception as exc:
            raise Qwen35HiddenLawAnalysisError(
                f"run is not complete and immutably sealed: {condition.run_id}: {exc}"
            ) from exc
        identity = read_json(path / "identity.json")
        implementation = read_json(path / "implementation.json")
        fingerprint = implementation.get("implementation_fingerprint")
        if fingerprint != registered_implementation_fingerprint:
            raise Qwen35HiddenLawAnalysisError(
                f"{condition.run_id} implementation differs from the frozen registration"
            )
        if identity != _expected_identity(condition, registered_implementation_fingerprint):
            raise Qwen35HiddenLawAnalysisError(f"{condition.run_id} identity differs from its plan row")
        resolved = _load_resolved_config(path / "resolved_config.yaml")
        if resolved != expected_config:
            raise Qwen35HiddenLawAnalysisError(
                f"{condition.run_id} resolved config differs from the registered config"
            )
        if canonical_digest(scientific_config(dict(resolved))) != condition.config_digest:
            raise Qwen35HiddenLawAnalysisError(
                f"{condition.run_id} scientific config digest differs from its identity"
            )
        status = read_json(path / "status.json")
        if (
            status.get("state") != "complete"
            or status.get("phase") != "complete"
            or status.get("last_step") != FINAL_STEP
            or status.get("error") is not None
        ):
            raise Qwen35HiddenLawAnalysisError(f"{condition.run_id} completion status is invalid")
        model_manifest = read_json(path / "model-manifest.json")
        _validate_model_manifest(condition, model_manifest)
        loaded.append(
            LoadedRun(
                condition=condition,
                path=path,
                identity=identity,
                implementation=implementation,
                bank_manifest=read_json(path / "bank-manifest.json"),
                model_manifest=model_manifest,
                summary=read_json(path / "summary.json"),
                metrics=_jsonl(path / "metrics.jsonl"),
                transcripts=_jsonl(path / "transcripts.jsonl"),
            )
        )
    return tuple(loaded)


def _validate_panel_pairing(
    runs: Sequence[LoadedRun],
    *,
    bank_builder: Callable[[int], ProductionBank],
) -> tuple[dict[str, Any], dict[int, ProductionBank]]:
    if len(runs) != 24:
        raise Qwen35HiddenLawAnalysisError(f"panel has {len(runs)} runs rather than 24")
    keys = [(run.condition.seed, run.condition.model_name, run.condition.algorithm) for run in runs]
    expected = set(product(SEEDS, MODELS, ALGORITHMS))
    if len(set(keys)) != 24 or set(keys) != expected:
        raise Qwen35HiddenLawAnalysisError("run conditions do not form the exact paired panel")

    implementations = {canonical_digest(dict(run.implementation)) for run in runs}
    if len(implementations) != 1:
        raise Qwen35HiddenLawAnalysisError("implementation manifests differ across runs")
    bank_rows: list[dict[str, Any]] = []
    bank_digests: set[str] = set()
    pairing_keys: set[str] = set()
    regenerated: dict[int, ProductionBank] = {}
    for seed in SEEDS:
        seed_runs = [run for run in runs if run.condition.seed == seed]
        manifests = {canonical_digest(dict(run.bank_manifest)) for run in seed_runs}
        if len(seed_runs) != 4 or len(manifests) != 1:
            raise Qwen35HiddenLawAnalysisError(
                f"seed {seed} does not share one exact bank across both models and algorithms"
            )
        bank = seed_runs[0].bank_manifest
        _exact_keys(
            bank,
            {"schema", "schema_version", "bank_digest", "pairing_key", "manifest"},
            f"seed {seed} bank manifest",
        )
        if bank["schema"] != "goalzendo.hidden_law_bank_manifest" or bank["schema_version"] != 1:
            raise Qwen35HiddenLawAnalysisError(f"seed {seed} bank manifest schema is invalid")
        digest = str(bank["bank_digest"])
        pairing_key = str(bank["pairing_key"])
        if len(digest) != 64 or len(pairing_key) != 64:
            raise Qwen35HiddenLawAnalysisError(f"seed {seed} bank identifiers are malformed")
        manifest = _mapping(bank["manifest"], f"seed {seed} nested bank manifest")
        if manifest.get("seed") != seed:
            raise Qwen35HiddenLawAnalysisError(f"seed {seed} bank records the wrong seed")
        try:
            rebuilt = bank_builder(seed)
        except Exception as exc:
            raise Qwen35HiddenLawAnalysisError(f"cannot regenerate registered bank for seed {seed}") from exc
        expected_bank_manifest = {
            "schema": "goalzendo.hidden_law_bank_manifest",
            "schema_version": 1,
            "bank_digest": rebuilt.digest,
            "pairing_key": rebuilt.pairing_key,
            "manifest": rebuilt.manifest,
        }
        if dict(bank) != expected_bank_manifest:
            raise Qwen35HiddenLawAnalysisError(
                f"seed {seed} sealed bank differs from deterministic regeneration"
            )
        regenerated[seed] = rebuilt
        for run in seed_runs:
            if run.summary.get("bank_digest") != digest:
                raise Qwen35HiddenLawAnalysisError(
                    f"{run.condition.run_id} summary is not bound to its paired bank"
                )
        bank_digests.add(digest)
        pairing_keys.add(pairing_key)
        bank_rows.append(
            {
                "seed": seed,
                "bank_digest": digest,
                "pairing_key": pairing_key,
                "manifest": dict(manifest),
            }
        )
    if len(bank_digests) != len(SEEDS) or len(pairing_keys) != len(SEEDS):
        raise Qwen35HiddenLawAnalysisError("the six seed blocks do not have six unique paired banks")
    return (
        {
            "implementation_manifest_digest": next(iter(implementations)),
            "paired_banks": bank_rows,
        },
        regenerated,
    )


def _validate_summary(run: LoadedRun) -> dict[str, Any]:
    summary = run.summary
    _exact_keys(
        summary,
        {
            "schema",
            "schema_version",
            "run_id",
            "plan_key",
            "smoke",
            "last_step",
            "algorithm",
            "model_name",
            "seed",
            "bank_digest",
            "evaluation",
            "operational",
        },
        f"{run.condition.run_id} run summary",
    )
    condition = run.condition
    expected = {
        "schema": "goalzendo.hidden_law_run_summary",
        "schema_version": 1,
        "run_id": condition.run_id,
        "plan_key": condition.plan_key,
        "smoke": False,
        "last_step": FINAL_STEP,
        "algorithm": condition.algorithm,
        "model_name": condition.model_name,
        "seed": condition.seed,
        "bank_digest": run.bank_manifest["bank_digest"],
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise Qwen35HiddenLawAnalysisError(
                f"{condition.run_id} summary {key} differs from its registered identity"
            )
    operational = _mapping(summary["operational"], f"{condition.run_id} operational summary")
    device = operational.get("device")
    if type(device) is not str or not device:
        raise Qwen35HiddenLawAnalysisError(f"{condition.run_id} operational device is invalid")
    counts: dict[str, int] = {}
    for key in (
        "forward_calls",
        "scored_prompt_count",
        "scored_prompt_tokens_unpadded",
        "maximum_prompt_tokens",
    ):
        counts[key] = _integer(
            operational.get(key),
            f"{condition.run_id} operational {key}",
            minimum=1,
        )
    if counts["maximum_prompt_tokens"] > 1536:
        raise Qwen35HiddenLawAnalysisError(f"{condition.run_id} exceeded the prompt-token limit")
    if operational.get("token_count_definition") != TOKEN_COUNT_DEFINITION:
        raise Qwen35HiddenLawAnalysisError(
            f"{condition.run_id} operational token-count definition differs from registration"
        )
    return {
        "device": device,
        **counts,
        "token_count_definition": TOKEN_COUNT_DEFINITION,
    }


def _condition_key(p_evidence: str, q_evidence: str) -> str:
    names = {
        ("perfect", "perfect"): "both_perfect",
        ("noisy", "perfect"): "p_noisy",
        ("perfect", "noisy"): "q_noisy",
        ("noisy", "noisy"): "both_noisy",
    }
    try:
        return names[(p_evidence, q_evidence)]
    except KeyError as exc:
        raise Qwen35HiddenLawAnalysisError(
            f"unknown evidence condition: {(p_evidence, q_evidence)!r}"
        ) from exc


def _validate_analysis_metadata(row: Mapping[str, Any], label: str) -> dict[str, Any]:
    p_evidence = str(row.get("p_evidence"))
    q_evidence = str(row.get("q_evidence"))
    condition = _condition_key(p_evidence, q_evidence)
    expected_condition_id = f"p-{p_evidence}_q-{q_evidence}"
    if row.get("condition_id") != expected_condition_id:
        raise Qwen35HiddenLawAnalysisError(f"{label} condition_id is inconsistent")
    y_family = row.get("y_rule_family")
    monotone = row.get("monotone_operator")
    if y_family not in {"monotone", "exactly_one"} or monotone not in {"all", "any"}:
        raise Qwen35HiddenLawAnalysisError(f"{label} formula stratum is invalid")
    ids = _mapping(row.get("analysis_candidate_ids"), f"{label} analysis_candidate_ids")
    if set(ids) != set(ANALYSIS_ROLES) or set(ids.values()) != {"A", "B", "C", "D"}:
        raise Qwen35HiddenLawAnalysisError(f"{label} candidate-role mapping is not a bijection")
    return {
        "condition": condition,
        "p_evidence": p_evidence,
        "q_evidence": q_evidence,
        "y_rule_family": str(y_family),
        "monotone_operator": str(monotone),
        "analysis_candidate_ids": dict(ids),
    }


def _validate_rule_shapes(
    rules_value: object,
    metadata: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    rules = _mapping(rules_value, f"{label} candidate_rules")
    if set(rules) != set(ANALYSIS_ROLES):
        raise Qwen35HiddenLawAnalysisError(f"{label} candidate_rules lacks Y/P/Q/R")
    copied = {role: dict(_mapping(rules[role], f"{label} rule {role}")) for role in ANALYSIS_ROLES}

    def root_op(role: str) -> str:
        rule = copied[role]
        op = rule.get("op")
        if op == "not":
            op = _mapping(rule.get("arg"), f"{label} negated {role} rule").get("op")
        if type(op) is not str:
            raise Qwen35HiddenLawAnalysisError(f"{label} {role} rule lacks an operation")
        return op

    if root_op("P") != "placard_is":
        raise Qwen35HiddenLawAnalysisError(f"{label} P is not a placard literal")
    if root_op("Q") in {"placard_is", "all", "any", "exactly_one"}:
        raise Qwen35HiddenLawAnalysisError(f"{label} Q is not a one-literal piece rule")
    y_op = root_op("Y")
    r_op = root_op("R")
    if metadata["y_rule_family"] == "monotone":
        expected_y, expected_r = metadata["monotone_operator"], "exactly_one"
    else:
        expected_y, expected_r = "exactly_one", metadata["monotone_operator"]
    if y_op != expected_y or r_op != expected_r:
        raise Qwen35HiddenLawAnalysisError(f"{label} candidate formulas contradict their stratum")
    return copied


def _validate_truth_table(
    transcript: Mapping[str, Any],
    label: str,
) -> tuple[dict[str, float], float, list[tuple[tuple[int, int, int, int], int]]]:
    truths = _mapping(transcript.get("terminal_truths"), f"{label} terminal_truths")
    if set(truths) != set(ANALYSIS_ROLES):
        raise Qwen35HiddenLawAnalysisError(f"{label} terminal truths lack Y/P/Q/R")
    vectors: dict[str, tuple[bool, ...]] = {}
    for role in ANALYSIS_ROLES:
        value = truths[role]
        if not isinstance(value, list) or len(value) != 16 or any(type(item) is not bool for item in value):
            raise Qwen35HiddenLawAnalysisError(f"{label} {role} truth vector is not Boolean length 16")
        vectors[role] = tuple(value)
    cells = [
        cast(
            tuple[int, int, int, int],
            tuple(int(vectors[role][index]) for role in ANALYSIS_ROLES),
        )
        for index in range(16)
    ]
    expected_cells = set(product((0, 1), repeat=4))
    if len(set(cells)) != 16 or set(cells) != expected_cells:
        raise Qwen35HiddenLawAnalysisError(f"{label} is not the complete 16-cell Y/P/Q/R census")
    predictions = transcript.get("terminal_predictions")
    if (
        not isinstance(predictions, list)
        or len(predictions) != 16
        or any(type(item) is not bool for item in predictions)
    ):
        raise Qwen35HiddenLawAnalysisError(f"{label} terminal predictions are not Boolean length 16")
    conflict = [index for index, cell in enumerate(cells) if len(set(cell)) > 1]
    if len(conflict) != 14:
        raise Qwen35HiddenLawAnalysisError(f"{label} terminal census does not have 14 conflict cells")
    agreements = {
        role: sum(predictions[index] is vectors[role][index] for index in conflict) / 14
        for role in ANALYSIS_ROLES
    }
    y_all = sum(prediction is truth for prediction, truth in zip(predictions, vectors["Y"], strict=True)) / 16
    table = [(cells[index], int(predictions[index])) for index in range(16)]
    return agreements, y_all, table


def _validate_queries(
    metric: Mapping[str, Any],
    transcript: Mapping[str, Any],
    metadata: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    expected_start = {
        "both_perfect": 4,
        "p_noisy": 3,
        "q_noisy": 3,
        "both_noisy": 2,
    }[str(metadata["condition"])]
    start = _integer(metric.get("initial_live_count"), f"{label} initial_live_count", minimum=1)
    end = _integer(metric.get("final_live_count"), f"{label} final_live_count", minimum=1)
    query_count = _integer(metric.get("query_count"), f"{label} query_count")
    if start != expected_start or not end <= start or query_count > 2:
        raise Qwen35HiddenLawAnalysisError(f"{label} live-set or query count differs from the design")
    observations = transcript.get("query_observations")
    turns = metric.get("query_turns")
    if not isinstance(observations, list) or not isinstance(turns, list):
        raise Qwen35HiddenLawAnalysisError(f"{label} query records must be arrays")
    if len(observations) != query_count or len(turns) != query_count:
        raise Qwen35HiddenLawAnalysisError(f"{label} query records disagree with query_count")
    live = start
    seen_options: set[str] = set()
    validated_turns: list[dict[str, Any]] = []
    validated_observations: list[dict[str, Any]] = []
    for offset, (observation_value, turn_value) in enumerate(zip(observations, turns, strict=True), start=1):
        observation = _mapping(observation_value, f"{label} query observation {offset}")
        turn = _mapping(turn_value, f"{label} query turn {offset}")
        option = observation.get("option_id")
        if type(option) is not str or option in seen_options or turn.get("option_id") != option:
            raise Qwen35HiddenLawAnalysisError(f"{label} query option sequence is invalid")
        seen_options.add(option)
        accepted = _boolean(observation.get("accepted"), f"{label} query accepted")
        if turn.get("accepted") is not accepted or turn.get("turn") != offset:
            raise Qwen35HiddenLawAnalysisError(f"{label} query observation and metric differ")
        before = _integer(turn.get("before_count"), f"{label} before_count", minimum=1)
        after = _integer(turn.get("after_count"), f"{label} after_count", minimum=1)
        if before != live or after > before:
            raise Qwen35HiddenLawAnalysisError(f"{label} query increased the live set")
        realized = _finite(turn.get("realized_information_bits"), f"{label} realized information")
        _close(realized, math.log2(before / after), f"{label} realized query information")
        expected_information = _finite(turn.get("expected_information_bits"), f"{label} expected information")
        best_information = _finite(turn.get("best_expected_information_bits"), f"{label} best information")
        regret = _finite(turn.get("regret_bits"), f"{label} query regret")
        if min(expected_information, best_information, regret) < -1e-12:
            raise Qwen35HiddenLawAnalysisError(f"{label} query information is negative")
        _close(best_information - expected_information, regret, f"{label} query regret identity")
        validated_turns.append(
            {
                "turn": offset,
                "option_id": option,
                "accepted": accepted,
                "before_count": before,
                "after_count": after,
                "expected_information_bits": expected_information,
                "best_expected_information_bits": best_information,
                "realized_information_bits": realized,
                "regret_bits": regret,
            }
        )
        validated_observations.append({"option_id": option, "accepted": accepted})
        live = after
    if live != end:
        raise Qwen35HiddenLawAnalysisError(f"{label} query chain does not end at final_live_count")
    view = metric.get("view")
    if view == "no_query" and (query_count != 0 or end != start):
        raise Qwen35HiddenLawAnalysisError(f"{label} no-query view contains inquiry")
    decisions_value = transcript.get("decisions")
    if not isinstance(decisions_value, list):
        raise Qwen35HiddenLawAnalysisError(f"{label} decisions must be an array")
    query_decisions = [
        _mapping(value, f"{label} query decision")
        for value in decisions_value
        if isinstance(value, Mapping) and value.get("kind") == "query"
    ]
    selected_query_labels = [decision.get("selected_label") for decision in query_decisions]
    if any(label_value not in set("ABCDEFGHI") for label_value in selected_query_labels):
        raise Qwen35HiddenLawAnalysisError(f"{label} contains an invalid query action label")
    ready_positions = [
        index for index, selected_label in enumerate(selected_query_labels) if selected_label == "I"
    ]
    if ready_positions and (len(ready_positions) != 1 or ready_positions[0] != len(query_decisions) - 1):
        raise Qwen35HiddenLawAnalysisError(f"{label} READY action is not a unique terminal inquiry action")
    declared_ready = _boolean(metric.get("declared_ready"), f"{label} declared_ready")
    if declared_ready is not bool(ready_positions):
        raise Qwen35HiddenLawAnalysisError(f"{label} declared_ready is inconsistent with decisions")
    if view == "active":
        if sum(selected_label != "I" for selected_label in selected_query_labels) != query_count:
            raise Qwen35HiddenLawAnalysisError(f"{label} query decisions disagree with observations")
    elif query_decisions:
        raise Qwen35HiddenLawAnalysisError(f"{label} supplied-query view contains model query decisions")
    information = _unit(metric.get("information_fraction"), f"{label} information_fraction")
    expected_fraction = math.log2(start / end) / math.log2(start)
    _close(information, expected_fraction, f"{label} information_fraction")
    return {
        "initial_live_count": start,
        "final_live_count": end,
        "query_count": query_count,
        "information_fraction": information,
        "query_observations": validated_observations,
        "query_turns": validated_turns,
        "declared_ready": declared_ready,
        "early_ready": declared_ready and end > 1,
    }


def _validate_interventions(
    metric: Mapping[str, Any],
    transcript: Mapping[str, Any],
    *,
    final: bool,
    label: str,
) -> tuple[dict[str, float | None], tuple[tuple[Any, ...], ...]]:
    records = transcript.get("interventions")
    if not isinstance(records, list):
        raise Qwen35HiddenLawAnalysisError(f"{label} interventions must be an array")
    observed_map = _mapping(metric.get("intervention_flip_rates"), f"{label} intervention_flip_rates")
    direct_fields = {target: metric.get(f"flip_{target}") for target in INTERVENTION_TARGETS}
    if not final:
        if records or observed_map or any(value is not None for value in direct_fields.values()):
            raise Qwen35HiddenLawAnalysisError(f"{label} interim episode contains causal interventions")
        if metric.get(CAUSAL_COMPANION) is not None:
            raise Qwen35HiddenLawAnalysisError(f"{label} interim episode contains a causal margin")
        return {target: None for target in INTERVENTION_TARGETS}, ()

    if len(records) != 10:
        raise Qwen35HiddenLawAnalysisError(f"{label} final episode must contain ten matched edits")
    counts: Counter[str] = Counter()
    flips: dict[str, list[bool]] = defaultdict(list)
    signatures: list[tuple[Any, ...]] = []
    endpoints: set[int] = set()
    for value in records:
        record = _mapping(value, f"{label} intervention")
        target = str(record.get("target"))
        if target not in INTERVENTION_TARGETS:
            raise Qwen35HiddenLawAnalysisError(f"{label} has an unknown intervention target")
        pair_index = _integer(record.get("pair_index"), f"{label} intervention pair index")
        if pair_index not in {0, 1}:
            raise Qwen35HiddenLawAnalysisError(f"{label} intervention pair index is invalid")
        before_index = _integer(record.get("before_scene_index"), f"{label} before scene index")
        after_index = _integer(record.get("after_scene_index"), f"{label} after scene index")
        if before_index == after_index or before_index in endpoints or after_index in endpoints:
            raise Qwen35HiddenLawAnalysisError(f"{label} intervention endpoints are not disjoint")
        endpoints.update((before_index, after_index))
        changed_field = record.get("changed_field")
        if type(changed_field) is not str or not changed_field:
            raise Qwen35HiddenLawAnalysisError(f"{label} intervention changed_field is invalid")
        before_prediction = _boolean(
            record.get("before_prediction"), f"{label} before intervention prediction"
        )
        after_prediction = _boolean(record.get("after_prediction"), f"{label} after intervention prediction")
        hard_flip = _boolean(record.get("hard_flip"), f"{label} hard intervention flip")
        if hard_flip is not (before_prediction is not after_prediction):
            raise Qwen35HiddenLawAnalysisError(f"{label} hard_flip is inconsistent")
        _unit(record.get("before_fit_probability"), f"{label} before fit probability")
        _unit(record.get("after_fit_probability"), f"{label} after fit probability")
        counts[target] += 1
        flips[target].append(hard_flip)
        signatures.append((target, pair_index, before_index, after_index, changed_field))
    if counts != Counter({target: 2 for target in INTERVENTION_TARGETS}):
        raise Qwen35HiddenLawAnalysisError(f"{label} lacks two edit pairs per target")
    if any(
        {signature[1] for signature in signatures if signature[0] == target} != {0, 1}
        for target in INTERVENTION_TARGETS
    ):
        raise Qwen35HiddenLawAnalysisError(f"{label} matched pair indices are incomplete")
    rates = {target: sum(flips[target]) / 2 for target in INTERVENTION_TARGETS}
    if set(observed_map) != set(INTERVENTION_TARGETS):
        raise Qwen35HiddenLawAnalysisError(f"{label} intervention rate mapping is incomplete")
    for target, rate in rates.items():
        observed = _unit(observed_map[target], f"{label} flip_{target}")
        direct = _unit(direct_fields[target], f"{label} direct flip_{target}")
        _close(observed, rate, f"{label} mapped flip_{target}")
        _close(direct, rate, f"{label} direct flip_{target}")
    causal = _finite(metric.get(CAUSAL_COMPANION), f"{label} causal margin")
    expected_causal = rates["Y"] - max(rates[role] for role in ("P", "Q", "R"))
    _close(causal, expected_causal, f"{label} causal margin")
    return {target: rates[target] for target in INTERVENTION_TARGETS}, tuple(sorted(signatures))


def _validate_episode(
    metric: Mapping[str, Any],
    transcript: Mapping[str, Any],
    run_id: str,
) -> dict[str, Any]:
    step = _integer(metric.get("step"), f"{run_id} evaluation step")
    view = metric.get("view")
    if step not in EVAL_STEPS or view not in FINAL_VIEWS:
        raise Qwen35HiddenLawAnalysisError(f"{run_id} has an unregistered evaluation axis")
    final = step == FINAL_STEP
    if not final and view != "active":
        raise Qwen35HiddenLawAnalysisError(f"{run_id} has a non-active interim view")
    label = f"{run_id} step {step} {view} {metric.get('instance_id')}"
    for key in (
        "step",
        "view",
        "family_id",
        "instance_id",
        "renderer",
        "condition_id",
        "p_evidence",
        "q_evidence",
        "y_rule_family",
        "monotone_operator",
        "analysis_candidate_ids",
    ):
        if transcript.get(key) != metric.get(key):
            raise Qwen35HiddenLawAnalysisError(f"{label} metric/transcript {key} differs")
    metadata = _validate_analysis_metadata(metric, label)
    rules = _validate_rule_shapes(transcript.get("candidate_rules"), metadata, label)
    inquiry = _validate_queries(metric, transcript, metadata, label)
    agreements, y_all, table = _validate_truth_table(transcript, label)
    agreement_map = _mapping(metric.get("agreement"), f"{label} agreement")
    if set(agreement_map) != set(ANALYSIS_ROLES):
        raise Qwen35HiddenLawAnalysisError(f"{label} agreement mapping is incomplete")
    for role, derived in agreements.items():
        mapped = _unit(agreement_map[role], f"{label} mapped rho_{role}")
        direct = _unit(metric.get(f"rho_{role}"), f"{label} direct rho_{role}")
        _close(mapped, derived, f"{label} mapped rho_{role}")
        _close(direct, derived, f"{label} direct rho_{role}")
    _close(
        _unit(metric.get("terminal_classification_accuracy_all_16"), f"{label} terminal accuracy"),
        y_all,
        f"{label} all-cell Y accuracy",
    )
    margin = agreements["Y"] - max(agreements[role] for role in ("P", "Q", "R"))
    _close(_finite(metric.get("law_control_margin"), f"{label} Law-control margin"), margin, label)
    ids = metadata["analysis_candidate_ids"]
    official = metric.get("official_candidate_id")
    selected = metric.get("selected_candidate_id")
    if official != ids["Y"] or selected not in {"A", "B", "C", "D"}:
        raise Qwen35HiddenLawAnalysisError(f"{label} selected or Official candidate is invalid")
    exact = _boolean(metric.get("exact_rule_recovery"), f"{label} exact recovery")
    if exact is not (selected == official):
        raise Qwen35HiddenLawAnalysisError(f"{label} exact recovery is inconsistent")
    selected_role = next(role for role in ANALYSIS_ROLES if ids[role] == selected)
    stated_rule_agreement = agreements[selected_role]
    named_proxy_margin = agreements["Y"] - max(agreements[role] for role in ("P", "Q"))
    flips, intervention_signature = _validate_interventions(
        metric,
        transcript,
        final=final,
        label=label,
    )
    opening_value = transcript.get("opening")
    if not isinstance(opening_value, list) or len(opening_value) != 10:
        raise Qwen35HiddenLawAnalysisError(f"{label} opening is not the registered ten examples")
    opening: list[dict[str, Any]] = []
    for index, value in enumerate(opening_value):
        item = _mapping(value, f"{label} opening item {index}")
        opening.append(
            {
                "scene_index": _integer(item.get("scene_index"), f"{label} opening scene index"),
                "accepted": _boolean(item.get("accepted"), f"{label} opening label"),
            }
        )
    terminal_value = transcript.get("terminal_scene_indices")
    if (
        not isinstance(terminal_value, list)
        or len(terminal_value) != 16
        or any(isinstance(value, bool) or not isinstance(value, int) for value in terminal_value)
        or len(set(terminal_value)) != 16
    ):
        raise Qwen35HiddenLawAnalysisError(f"{label} terminal scene identities are invalid")
    return {
        "step": step,
        "view": str(view),
        "family_id": str(metric.get("family_id")),
        "instance_id": str(metric.get("instance_id")),
        "condition_id": str(metric.get("condition_id")),
        "renderer": str(metric.get("renderer")),
        **metadata,
        "candidate_rules": rules,
        "opening": opening,
        "terminal_scene_indices": list(terminal_value),
        "initial_live_count": inquiry["initial_live_count"],
        "final_live_count": inquiry["final_live_count"],
        "information_fraction": inquiry["information_fraction"],
        "query_count": inquiry["query_count"],
        "query_observations": inquiry["query_observations"],
        "query_turns": inquiry["query_turns"],
        "declared_ready": inquiry["declared_ready"],
        "early_ready": inquiry["early_ready"],
        "exact_rule_recovery": float(exact),
        "selected_candidate_id": selected,
        "selected_analysis_role": selected_role,
        "stated_rule_behavioral_agreement": stated_rule_agreement,
        "rho": agreements,
        "terminal_classification_accuracy_all_16": y_all,
        "law_control_margin": margin,
        "named_proxy_only_margin": named_proxy_margin,
        "flip": flips,
        "causal_law_control_margin": (None if not final else float(metric[CAUSAL_COMPANION])),
        "terminal_truth_table": table,
        "intervention_signature": intervention_signature,
    }


def _validate_optimization_and_training(run: LoadedRun) -> None:
    run_id = run.condition.run_id
    optimization = [row for row in run.metrics if row.get("kind") == "optimization_block"]
    if Counter(_integer(row.get("step"), f"{run_id} optimization step") for row in optimization) != Counter(
        {step: 1 for step in range(1, FINAL_STEP + 1)}
    ):
        raise Qwen35HiddenLawAnalysisError(f"{run_id} lacks exactly 128 optimizer-step records")
    for row in optimization:
        if row.get("algorithm") != run.condition.algorithm:
            raise Qwen35HiddenLawAnalysisError(f"{run_id} optimization algorithm is inconsistent")
        _finite(row.get("loss"), f"{run_id} optimization loss")
        _finite(row.get("learning_rate"), f"{run_id} learning rate")
        gradient = _finite(row.get("gradient_norm"), f"{run_id} gradient norm")
        rotations = row.get("rotation_metrics")
        if gradient < 0 or not isinstance(rotations, list) or len(rotations) != 4:
            raise Qwen35HiddenLawAnalysisError(f"{run_id} optimization rotation record is invalid")

    training = [row for row in run.transcripts if row.get("kind") == "training_trajectory"]
    expected_per_step = 4 if run.condition.algorithm == "process_sft" else 16
    observed = Counter(_integer(row.get("step"), f"{run_id} trajectory step") for row in training)
    if observed != Counter({step: expected_per_step for step in range(1, FINAL_STEP + 1)}):
        raise Qwen35HiddenLawAnalysisError(f"{run_id} does not have the registered training-trajectory count")
    for row in training:
        if row.get("algorithm") != run.condition.algorithm:
            raise Qwen35HiddenLawAnalysisError(f"{run_id} trajectory algorithm is inconsistent")
        trajectory = _mapping(row.get("trajectory"), f"{run_id} training trajectory")
        if type(trajectory.get("instance_id")) is not str:
            raise Qwen35HiddenLawAnalysisError(f"{run_id} trajectory lacks an instance identity")


def _aggregate_episodes(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise Qwen35HiddenLawAnalysisError("cannot aggregate an empty episode collection")
    final = all(row[CAUSAL_COMPANION] is not None for row in rows)
    interim = all(row[CAUSAL_COMPANION] is None for row in rows)
    if not (final or interim):
        raise Qwen35HiddenLawAnalysisError("episode collection mixes causal and non-causal rows")
    result: dict[str, Any] = {
        "episode_count": len(rows),
        "information_fraction": _mean([float(row["information_fraction"]) for row in rows]),
        "mean_query_count": _mean([float(row["query_count"]) for row in rows]),
        "declared_ready_rate": _mean([float(row["declared_ready"]) for row in rows]),
        "early_ready_rate": _mean([float(row["early_ready"]) for row in rows]),
        "exact_rule_recovery": _mean([float(row["exact_rule_recovery"]) for row in rows]),
        "stated_rule_behavioral_agreement": _mean(
            [float(row["stated_rule_behavioral_agreement"]) for row in rows]
        ),
        "rho_Y": _mean([float(row["rho"]["Y"]) for row in rows]),
        "rho_P": _mean([float(row["rho"]["P"]) for row in rows]),
        "rho_Q": _mean([float(row["rho"]["Q"]) for row in rows]),
        "rho_R": _mean([float(row["rho"]["R"]) for row in rows]),
        "terminal_classification_accuracy_all_16": _mean(
            [float(row["terminal_classification_accuracy_all_16"]) for row in rows]
        ),
        "law_control_margin": _mean([float(row["law_control_margin"]) for row in rows]),
        "named_proxy_only_margin": _mean([float(row["named_proxy_only_margin"]) for row in rows]),
        "selected_rule_distribution": {
            role: {
                "count": sum(row["selected_analysis_role"] == role for row in rows),
                "rate": sum(row["selected_analysis_role"] == role for row in rows) / len(rows),
            }
            for role in ANALYSIS_ROLES
        },
    }
    result["query_turns"] = []
    for turn in (1, 2):
        reached = [turn_row for row in rows for turn_row in row["query_turns"] if turn_row["turn"] == turn]
        denominator = len(reached)
        result["query_turns"].append(
            {
                "turn": turn,
                "all_episode_denominator": len(rows),
                "turn_conditioned_denominator": denominator,
                "reach_rate": denominator / len(rows),
                "mean_expected_information_bits": (
                    _mean([float(item["expected_information_bits"]) for item in reached]) if reached else None
                ),
                "mean_realized_information_bits": (
                    _mean([float(item["realized_information_bits"]) for item in reached]) if reached else None
                ),
                "mean_regret_bits": (
                    _mean([float(item["regret_bits"]) for item in reached]) if reached else None
                ),
            }
        )
    for target in INTERVENTION_TARGETS:
        result[f"flip_{target}"] = _mean([float(row["flip"][target]) for row in rows]) if final else None
    result[CAUSAL_COMPANION] = _mean([float(row[CAUSAL_COMPANION]) for row in rows]) if final else None
    return result


def _truth_table(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    predictions: dict[tuple[int, int, int, int], list[int]] = defaultdict(list)
    for row in rows:
        for cell, prediction in row["terminal_truth_table"]:
            predictions[tuple(cell)].append(int(prediction))
    expected = set(product((0, 1), repeat=4))
    if set(predictions) != expected:
        raise Qwen35HiddenLawAnalysisError("aggregate truth table lacks a candidate cell")
    return [
        {
            "Y": cell[0],
            "P": cell[1],
            "Q": cell[2],
            "R": cell[3],
            "fit_rate": _mean(predictions[cell]),
            "n": len(predictions[cell]),
        }
        for cell in sorted(predictions)
    ]


def _validate_checkpoint_summary(
    row: Mapping[str, Any],
    episodes: Sequence[Mapping[str, Any]],
    run_id: str,
) -> None:
    step = _integer(row.get("step"), f"{run_id} checkpoint summary step")
    final = step == FINAL_STEP
    expected_views = FINAL_VIEWS if final else ("active",)
    expected_quartets = 16 if final else 4
    expected_episodes = expected_quartets * 4 * len(expected_views)
    if (
        row.get("final") is not final
        or row.get("quartet_count") != expected_quartets
        or row.get("episode_count") != expected_episodes
    ):
        raise Qwen35HiddenLawAnalysisError(f"{run_id} checkpoint summary dimensions are invalid")
    view_summaries = _mapping(row.get("views"), f"{run_id} checkpoint view summaries")
    if set(view_summaries) != set(expected_views):
        raise Qwen35HiddenLawAnalysisError(f"{run_id} checkpoint summary views differ from registration")
    for view in expected_views:
        observed = _mapping(view_summaries[view], f"{run_id} {view} checkpoint summary")
        selected = [item for item in episodes if item["step"] == step and item["view"] == view]
        derived = _aggregate_episodes(selected)
        expected_fields = {
            "episode_count": derived["episode_count"],
            "information_fraction": derived["information_fraction"],
            "exact_rule_recovery": derived["exact_rule_recovery"],
            "terminal_classification_accuracy_all_16": _mean(
                [float(item["terminal_classification_accuracy_all_16"]) for item in selected]
            ),
            "law_control_margin": derived["law_control_margin"],
            CAUSAL_COMPANION: derived[CAUSAL_COMPANION],
        }
        for key, value in expected_fields.items():
            if value is None:
                if observed.get(key) is not None:
                    raise Qwen35HiddenLawAnalysisError(
                        f"{run_id} checkpoint summary unexpectedly records {key}"
                    )
            elif key == "episode_count":
                if observed.get(key) != value:
                    raise Qwen35HiddenLawAnalysisError(f"{run_id} checkpoint summary {key} is inconsistent")
            else:
                _close(
                    _finite(observed.get(key), f"{run_id} checkpoint {key}"),
                    float(value),
                    f"{run_id} checkpoint summary {key}",
                )


def _validate_regenerated_bank_axes(
    episodes: Sequence[dict[str, Any]],
    bank: ProductionBank,
    run_id: str,
) -> None:
    """Match every evaluation row to the deterministically regenerated seed bank."""

    families = tuple(bank.evaluation_families)
    family_by_id = {family.family_id: family for family in families}
    game_by_id = {game.instance_id: game for game in bank.evaluation_games}
    if len(family_by_id) != 16 or len(game_by_id) != 64:
        raise Qwen35HiddenLawAnalysisError(f"{run_id} regenerated evaluation bank is incomplete")
    interim_ids = tuple(family.family_id for family in bank.interim_evaluation_families)
    expected_interim_ids = tuple(families[index].family_id for index in (0, 5, 10, 15))
    if interim_ids != expected_interim_ids:
        raise Qwen35HiddenLawAnalysisError(f"{run_id} regenerated interim family indices differ")

    expected_axes: Counter[tuple[int, str, str, str, str]] = Counter()
    for step in EVAL_STEPS:
        selected_families = families if step == FINAL_STEP else bank.interim_evaluation_families
        views = FINAL_VIEWS if step == FINAL_STEP else ("active",)
        for family, view, condition in product(
            selected_families,
            views,
            EVIDENCE_CELLS,
        ):
            expected_axes[(step, view, family.family_id, condition[0], condition[1])] += 1
    observed_axes = Counter(
        (
            int(episode["step"]),
            str(episode["view"]),
            str(episode["family_id"]),
            str(episode["p_evidence"]),
            str(episode["q_evidence"]),
        )
        for episode in episodes
    )
    if observed_axes != expected_axes:
        raise Qwen35HiddenLawAnalysisError(
            f"{run_id} does not contain exactly one 2x2 evidence quartet per step/view/family"
        )

    family_offsets = {family.family_id: index for index, family in enumerate(families)}
    for episode in episodes:
        family_id = str(episode["family_id"])
        instance_id = str(episode["instance_id"])
        if family_id not in family_by_id or instance_id not in game_by_id:
            raise Qwen35HiddenLawAnalysisError(f"{run_id} evaluation identity is absent from its bank")
        family = family_by_id[family_id]
        game = game_by_id[instance_id]
        if game.family.family_id != family_id:
            raise Qwen35HiddenLawAnalysisError(f"{run_id} instance is paired with the wrong family")
        condition_games = [
            (condition, candidate_game)
            for condition, candidate_game in zip(
                family.evaluation_conditions,
                family.evaluation_games,
                strict=True,
            )
            if condition.p_evidence == episode["p_evidence"] and condition.q_evidence == episode["q_evidence"]
        ]
        if len(condition_games) != 1:
            raise Qwen35HiddenLawAnalysisError(
                f"{run_id} regenerated bank does not identify one evidence-conditioned game"
            )
        expected_condition, expected_game = condition_games[0]
        if (
            episode["condition_id"] != expected_condition.condition_id
            or instance_id != expected_game.instance_id
            or game.instance_id != expected_game.instance_id
        ):
            raise Qwen35HiddenLawAnalysisError(
                f"{run_id} instance_id or condition_id differs from its regenerated evidence cell"
            )
        expected_renderer = "eval_reverse" if family_offsets[family_id] % 2 == 0 else "eval_ledger"
        if episode["renderer"] != expected_renderer:
            raise Qwen35HiddenLawAnalysisError(f"{run_id} renderer axis differs from its bank index")

        y_role = family.evaluation_y_role
        r_role = family.evaluation_r_role
        candidates = {
            "Y": family.candidate_by_role[y_role],
            "P": family.candidate_by_role["P"],
            "Q": family.candidate_by_role["Q"],
            "R": family.candidate_by_role[r_role],
        }
        expected_ids = {role: candidate.candidate_id for role, candidate in candidates.items()}
        expected_rules = {role: candidate.rule.as_obj() for role, candidate in candidates.items()}
        expected_y_family = "monotone" if y_role == "M" else "exactly_one"
        expected_monotone_operator = getattr(family.candidate_by_role["M"].rule, "op", None)
        if (
            episode["analysis_candidate_ids"] != expected_ids
            or episode["candidate_rules"] != expected_rules
            or episode["y_rule_family"] != expected_y_family
            or episode["monotone_operator"] != expected_monotone_operator
        ):
            raise Qwen35HiddenLawAnalysisError(
                f"{run_id} candidate role or formula metadata differs from regenerated bank"
            )
        expected_opening = [item.as_obj() for item in game.material.opening]
        if episode["opening"] != expected_opening:
            raise Qwen35HiddenLawAnalysisError(f"{run_id} opening differs from regenerated bank")
        if episode["terminal_scene_indices"] != list(game.material.terminal):
            raise Qwen35HiddenLawAnalysisError(f"{run_id} terminal identities differ from regenerated bank")
        expected_truth_cells = [
            tuple(int(candidates[role].rule.evaluate(scene_at(index))) for role in ANALYSIS_ROLES)
            for index in game.material.terminal
        ]
        observed_truth_cells = [tuple(cell) for cell, _prediction in episode["terminal_truth_table"]]
        if observed_truth_cells != expected_truth_cells:
            raise Qwen35HiddenLawAnalysisError(f"{run_id} terminal truth labels differ from regenerated bank")

        if episode["view"] in {"active", "oracle_query"}:
            live = game.initial_live_ids
            replayed_turns: list[dict[str, Any]] = []
            used: set[str] = set()
            observations = episode["query_observations"]
            turns = episode["query_turns"]
            if len(observations) != len(turns):
                raise Qwen35HiddenLawAnalysisError(
                    f"{run_id} inquiry observations and turn metrics have different lengths"
                )
            for turn_number, (observation, turn) in enumerate(
                zip(observations, turns, strict=True),
                start=1,
            ):
                option_id = str(observation["option_id"])
                if option_id in used:
                    raise Qwen35HiddenLawAnalysisError(f"{run_id} inquiry repeats a query option")
                used.add(option_id)
                try:
                    information = query_information(game, option_id, live)
                    oracle_accepted = game.oracle_label(option_id)
                except (TypeError, ValueError) as exc:
                    raise Qwen35HiddenLawAnalysisError(
                        f"{run_id} inquiry option is absent from its regenerated game"
                    ) from exc
                if observation["accepted"] is not oracle_accepted:
                    raise Qwen35HiddenLawAnalysisError(
                        f"{run_id} inquiry response differs from the regenerated Official label"
                    )
                after = information.accepted_ids if oracle_accepted else information.rejected_ids
                realized = math.log2(len(live)) - math.log2(len(after))
                exact_values = {
                    "before_count": len(live),
                    "after_count": len(after),
                    "expected_information_bits": information.expected_information_bits,
                    "best_expected_information_bits": information.best_expected_information_bits,
                    "realized_information_bits": realized,
                    "regret_bits": information.regret_bits,
                }
                if turn["turn"] != turn_number or turn["option_id"] != option_id:
                    raise Qwen35HiddenLawAnalysisError(
                        f"{run_id} inquiry turn identity differs from regenerated replay"
                    )
                for key, expected_value in exact_values.items():
                    if key in {"before_count", "after_count"}:
                        if turn[key] != expected_value:
                            raise Qwen35HiddenLawAnalysisError(
                                f"{run_id} inquiry {key} differs from regenerated replay"
                            )
                    else:
                        _close(
                            float(turn[key]),
                            float(expected_value),
                            f"{run_id} regenerated inquiry {key}",
                        )
                replayed_turns.append(
                    {
                        **dict(turn),
                        "before_live_candidate_ids": list(live),
                        "after_live_candidate_ids": list(after),
                    }
                )
                live = after
            if len(live) != episode["final_live_count"]:
                raise Qwen35HiddenLawAnalysisError(
                    f"{run_id} final live count differs from regenerated inquiry replay"
                )
            episode["query_turns"] = replayed_turns

        if episode["step"] == FINAL_STEP:
            expected_interventions = tuple(
                sorted(
                    (
                        item.target,
                        item.pair_index,
                        item.before_scene_index,
                        item.after_scene_index,
                        item.changed_field,
                    )
                    for item in family.matched_interventions
                )
            )
            if episode["intervention_signature"] != expected_interventions:
                raise Qwen35HiddenLawAnalysisError(
                    f"{run_id} intervention identities differ from regenerated bank"
                )

        if episode["view"] == "oracle_query":
            reference = [
                {"option_id": item.option_id, "accepted": item.accepted} for item in reference_inquiry(game)
            ]
            if episode["query_observations"] != reference:
                raise Qwen35HiddenLawAnalysisError(
                    f"{run_id} oracle-query transcript differs from the reference inquiry"
                )
            if (
                episode["final_live_count"] != 1
                or episode["query_count"] not in {1, 2}
                or episode["declared_ready"]
                or any(abs(float(item["regret_bits"])) > 1e-12 for item in episode["query_turns"])
            ):
                raise Qwen35HiddenLawAnalysisError(
                    f"{run_id} oracle-query episode is not exact zero-regret reference inquiry"
                )


def _validate_run_science(run: LoadedRun, bank: ProductionBank) -> dict[str, Any]:
    _validate_model_manifest(run.condition, run.model_manifest)
    operational = _validate_summary(run)
    _validate_optimization_and_training(run)
    run_id = run.condition.run_id
    allowed_metric_kinds = {
        "optimization_block",
        "evaluation_episode",
        "evaluation_checkpoint_summary",
    }
    kinds = {row.get("kind") for row in run.metrics}
    if kinds != allowed_metric_kinds:
        raise Qwen35HiddenLawAnalysisError(f"{run_id} metric kinds differ from registration")
    transcript_kinds = {row.get("kind") for row in run.transcripts}
    if transcript_kinds != {"training_trajectory", "evaluation_transcript"}:
        raise Qwen35HiddenLawAnalysisError(f"{run_id} transcript kinds differ from registration")

    metric_rows = [row for row in run.metrics if row.get("kind") == "evaluation_episode"]
    transcript_rows = [row for row in run.transcripts if row.get("kind") == "evaluation_transcript"]
    expected_episode_count = 3 * 16 + 192
    if len(metric_rows) != expected_episode_count or len(transcript_rows) != expected_episode_count:
        raise Qwen35HiddenLawAnalysisError(f"{run_id} evaluation episode count differs from registration")
    transcript_index: dict[tuple[int, str, str], Mapping[str, Any]] = {}
    for row in transcript_rows:
        key = (int(row.get("step", -1)), str(row.get("view")), str(row.get("instance_id")))
        if key in transcript_index:
            raise Qwen35HiddenLawAnalysisError(f"{run_id} has duplicate evaluation transcripts")
        transcript_index[key] = row
    episodes: list[dict[str, Any]] = []
    metric_keys: set[tuple[int, str, str]] = set()
    for metric in metric_rows:
        key = (int(metric.get("step", -1)), str(metric.get("view")), str(metric.get("instance_id")))
        if key in metric_keys or key not in transcript_index:
            raise Qwen35HiddenLawAnalysisError(f"{run_id} evaluation metric/transcript axes differ")
        metric_keys.add(key)
        episodes.append(_validate_episode(metric, transcript_index[key], run_id))
    if metric_keys != set(transcript_index):
        raise Qwen35HiddenLawAnalysisError(f"{run_id} has unmatched evaluation transcripts")

    expected_axis_counts = Counter(
        {(step, "active"): 16 for step in EVAL_STEPS[:-1]} | {(FINAL_STEP, view): 64 for view in FINAL_VIEWS}
    )
    observed_axis_counts = Counter((row["step"], row["view"]) for row in episodes)
    if observed_axis_counts != expected_axis_counts:
        raise Qwen35HiddenLawAnalysisError(f"{run_id} evaluation step/view counts differ")
    _validate_regenerated_bank_axes(episodes, bank, run_id)

    final_families = {row["family_id"] for row in episodes if row["step"] == FINAL_STEP}
    interim_families_by_step = {
        step: {row["family_id"] for row in episodes if row["step"] == step} for step in EVAL_STEPS[:-1]
    }
    if len(final_families) != 16 or any(len(value) != 4 for value in interim_families_by_step.values()):
        raise Qwen35HiddenLawAnalysisError(f"{run_id} evaluation family counts differ")
    interim_sets = {frozenset(value) for value in interim_families_by_step.values()}
    if len(interim_sets) != 1 or not next(iter(interim_sets)) <= final_families:
        raise Qwen35HiddenLawAnalysisError(f"{run_id} interim family panel is not fixed")

    family_metadata: dict[str, tuple[Any, ...]] = {}
    family_rules: dict[str, dict[str, Any]] = {}
    family_interventions: dict[str, tuple[tuple[Any, ...], ...]] = {}
    for episode in episodes:
        family = episode["family_id"]
        metadata = (
            episode["y_rule_family"],
            episode["monotone_operator"],
            canonical_digest(episode["analysis_candidate_ids"]),
            episode["renderer"],
        )
        if family in family_metadata and family_metadata[family] != metadata:
            raise Qwen35HiddenLawAnalysisError(f"{run_id} family metadata changes across episodes")
        family_metadata[family] = metadata
        if family in family_rules and family_rules[family] != episode["candidate_rules"]:
            raise Qwen35HiddenLawAnalysisError(f"{run_id} candidate formulas change across episodes")
        family_rules[family] = episode["candidate_rules"]
        if episode["step"] == FINAL_STEP:
            signature = episode["intervention_signature"]
            if family in family_interventions and family_interventions[family] != signature:
                raise Qwen35HiddenLawAnalysisError(f"{run_id} matched interventions change within a quartet")
            family_interventions[family] = signature

    for step in EVAL_STEPS:
        selected = [row for row in episodes if row["step"] == step and row["view"] == "active"]
        expected_family_count = 16 if step == FINAL_STEP else 4
        cell_counts = Counter((row["p_evidence"], row["q_evidence"]) for row in selected)
        if cell_counts != Counter({cell: expected_family_count for cell in EVIDENCE_CELLS}):
            raise Qwen35HiddenLawAnalysisError(f"{run_id} step {step} evidence cells are incomplete")
        y_counts = Counter(row["y_rule_family"] for row in selected)
        op_counts = Counter(row["monotone_operator"] for row in selected)
        expected_half = len(selected) // 2
        if y_counts != {"monotone": expected_half, "exactly_one": expected_half}:
            raise Qwen35HiddenLawAnalysisError(f"{run_id} step {step} Y-family balance differs")
        if op_counts != {"all": expected_half, "any": expected_half}:
            raise Qwen35HiddenLawAnalysisError(f"{run_id} step {step} monotone-op balance differs")
        joint = Counter((row["y_rule_family"], row["monotone_operator"]) for row in selected)
        expected_joint = len(selected) // 4
        if joint != Counter(
            {cell: expected_joint for cell in product(("monotone", "exactly_one"), ("all", "any"))}
        ):
            raise Qwen35HiddenLawAnalysisError(f"{run_id} step {step} formula joint balance differs")

    summaries = [row for row in run.metrics if row.get("kind") == "evaluation_checkpoint_summary"]
    if Counter(int(row.get("step", -1)) for row in summaries) != Counter({step: 1 for step in EVAL_STEPS}):
        raise Qwen35HiddenLawAnalysisError(f"{run_id} checkpoint summary trajectory is incomplete")
    for summary in summaries:
        _validate_checkpoint_summary(summary, episodes, run_id)

    trajectory = []
    for step in EVAL_STEPS:
        active = [row for row in episodes if row["step"] == step and row["view"] == "active"]
        trajectory.append({"step": step, **_aggregate_episodes(active)})
    final_views = {
        view: _aggregate_episodes(
            [row for row in episodes if row["step"] == FINAL_STEP and row["view"] == view]
        )
        for view in FINAL_VIEWS
    }
    final_active = [row for row in episodes if row["step"] == FINAL_STEP and row["view"] == "active"]
    condition_cells = {
        _condition_key(*cell): _aggregate_episodes(
            [row for row in final_active if (row["p_evidence"], row["q_evidence"]) == cell]
        )
        for cell in EVIDENCE_CELLS
    }
    formula_strata = {
        family: _aggregate_episodes([row for row in final_active if row["y_rule_family"] == family])
        for family in ("monotone", "exactly_one")
    }
    formula_by_condition = {
        family: {
            _condition_key(*cell): _aggregate_episodes(
                [
                    row
                    for row in final_active
                    if row["y_rule_family"] == family and (row["p_evidence"], row["q_evidence"]) == cell
                ]
            )
            for cell in EVIDENCE_CELLS
        }
        for family in ("monotone", "exactly_one")
    }
    formula_registry = [
        {
            "family_id": family,
            "y_rule_family": family_metadata[family][0],
            "monotone_operator": family_metadata[family][1],
            "analysis_candidate_ids": next(
                row["analysis_candidate_ids"] for row in final_active if row["family_id"] == family
            ),
            "candidate_rules": family_rules[family],
        }
        for family in sorted(final_families)
    ]
    endpoint = {
        "seed": run.condition.seed,
        "model": run.condition.model_name,
        "model_revision": run.condition.model_revision,
        "algorithm": run.condition.algorithm,
        "run_id": run_id,
        "plan_key": run.condition.plan_key,
        "bank_digest": run.bank_manifest["bank_digest"],
        "operational": operational,
        "active": final_views["active"],
        "final_views": final_views,
        "condition_cells": condition_cells,
        "formula_strata": formula_strata,
        "formula_by_condition": formula_by_condition,
        "final_active_truth_table": _truth_table(final_active),
    }
    return {
        "endpoint": endpoint,
        "trajectory": {
            "seed": run.condition.seed,
            "model": run.condition.model_name,
            "algorithm": run.condition.algorithm,
            "run_id": run_id,
            "checkpoints": trajectory,
        },
        "formula_registry": {
            "seed": run.condition.seed,
            "model": run.condition.model_name,
            "algorithm": run.condition.algorithm,
            "families": formula_registry,
        },
    }


def _linear_percentile(sorted_values: Sequence[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _bootstrap_interval(values: Sequence[float]) -> tuple[float, float]:
    if len(values) != len(SEEDS):
        raise Qwen35HiddenLawAnalysisError("bootstrap requires exactly six seed-block values")
    estimates = sorted(
        _mean(tuple(values[index] for index in indices))
        for indices in product(range(len(SEEDS)), repeat=len(SEEDS))
    )
    if len(estimates) != BOOTSTRAP_RESAMPLES:
        raise AssertionError("exhaustive bootstrap enumeration is incomplete")
    return _linear_percentile(estimates, 0.025), _linear_percentile(estimates, 0.975)


def _sign_flip_p_value(values: Sequence[float]) -> float:
    if len(values) != len(SEEDS):
        raise Qwen35HiddenLawAnalysisError("sign flip requires exactly six seed-block values")
    observed = abs(_mean(values))
    exceedances = sum(
        abs(_mean(tuple(sign * value for sign, value in zip(signs, values, strict=True)))) >= observed - 1e-15
        for signs in product((-1.0, 1.0), repeat=len(SEEDS))
    )
    return exceedances / SIGN_FLIP_ASSIGNMENTS


def _summarize_seed_blocks(values_by_seed: Mapping[int, float]) -> dict[str, Any]:
    if set(values_by_seed) != set(SEEDS):
        raise Qwen35HiddenLawAnalysisError("estimand does not contain exactly the six seed blocks")
    values = tuple(_finite(values_by_seed[seed], f"seed {seed} estimand") for seed in SEEDS)
    lower, upper = _bootstrap_interval(values)
    return {
        "per_seed": [{"seed": seed, "value": value} for seed, value in zip(SEEDS, values, strict=True)],
        "mean": _mean(values),
        "bootstrap_95_percentile_interval": {"lower": lower, "upper": upper},
        "descriptive_unadjusted_two_sided_sign_flip_p_value": _sign_flip_p_value(values),
    }


def _endpoint_index(
    endpoints: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, str, str], Mapping[str, Any]]:
    indexed: dict[tuple[int, str, str], Mapping[str, Any]] = {}
    for row in endpoints:
        key = (int(row["seed"]), str(row["model"]), str(row["algorithm"]))
        if key in indexed:
            raise Qwen35HiddenLawAnalysisError(f"duplicate endpoint condition: {key}")
        indexed[key] = row
    if set(indexed) != set(product(SEEDS, MODELS, ALGORITHMS)):
        raise Qwen35HiddenLawAnalysisError("endpoint index is not the exact 24-run factorial")
    return indexed


def _raw_endpoint(row: Mapping[str, Any]) -> dict[str, Any]:
    active = _mapping(row["active"], "active endpoint")
    return {
        "run_id": row["run_id"],
        **{name: float(active[name]) for name in ALL_ENDPOINTS},
        "named_proxy_only_margin": float(active["named_proxy_only_margin"]),
        "stated_rule_behavioral_agreement": float(active["stated_rule_behavioral_agreement"]),
        "mean_query_count": float(active["mean_query_count"]),
        "declared_ready_rate": float(active["declared_ready_rate"]),
        "early_ready_rate": float(active["early_ready_rate"]),
        "query_turns": [dict(item) for item in active["query_turns"]],
        "selected_rule_distribution": dict(active["selected_rule_distribution"]),
        **{f"rho_{role}": float(active[f"rho_{role}"]) for role in ANALYSIS_ROLES},
        **{f"flip_{target}": float(active[f"flip_{target}"]) for target in INTERVENTION_TARGETS},
        "operational": dict(row["operational"]),
    }


def _primary_contrast(
    indexed: Mapping[tuple[int, str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    per_model: list[dict[str, Any]] = []
    model_effects: dict[str, dict[str, dict[int, float]]] = {}
    for model in MODELS:
        effects: dict[str, dict[int, float]] = {name: {} for name in ALL_ENDPOINTS}
        pairs: list[dict[str, Any]] = []
        for seed in SEEDS:
            sft = indexed[(seed, model, "process_sft")]
            rl = indexed[(seed, model, "outcome_rl")]
            differences = {
                name: float(sft["active"][name]) - float(rl["active"][name]) for name in ALL_ENDPOINTS
            }
            for name, value in differences.items():
                effects[name][seed] = value
            pairs.append(
                {
                    "seed": seed,
                    "process_sft": _raw_endpoint(sft),
                    "outcome_rl": _raw_endpoint(rl),
                    "sft_minus_outcome_rl": differences,
                }
            )
        model_effects[model] = effects
        per_model.append(
            {
                "model": model,
                "raw_paired_endpoints": pairs,
                "sft_minus_outcome_rl": {
                    name: _summarize_seed_blocks(effects[name]) for name in ALL_ENDPOINTS
                },
            }
        )
    pooled = {
        name: {seed: _mean([model_effects[model][name][seed] for model in MODELS]) for seed in SEEDS}
        for name in ALL_ENDPOINTS
    }
    return {
        "definition": "final active-view process SFT minus outcome-only RL",
        "co_primary_outcomes": list(PRIMARY_OUTCOMES),
        "causal_companion": CAUSAL_COMPANION,
        "per_model": per_model,
        "pooled_equal_weight_across_models_within_seed": {
            name: _summarize_seed_blocks(pooled[name]) for name in ALL_ENDPOINTS
        },
    }


def _factorial_effects(row: Mapping[str, Any]) -> dict[str, float]:
    cells = _mapping(row["condition_cells"], "condition endpoint cells")
    values = {name: float(_mapping(cells[name], name)["law_control_margin"]) for name in cells}
    expected = {"both_perfect", "p_noisy", "q_noisy", "both_noisy"}
    if set(values) != expected:
        raise Qwen35HiddenLawAnalysisError("endpoint evidence cells are incomplete")
    c00 = values["both_perfect"]
    c10 = values["p_noisy"]
    c01 = values["q_noisy"]
    c11 = values["both_noisy"]
    return {
        "p_noise_main_effect": 0.5 * ((c10 - c00) + (c11 - c01)),
        "q_noise_main_effect": 0.5 * ((c01 - c00) + (c11 - c10)),
        "p_by_q_interaction": c11 - c10 - c01 + c00,
    }


def _secondary_factorial(
    indexed: Mapping[tuple[int, str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    effect_names = ("p_noise_main_effect", "q_noise_main_effect", "p_by_q_interaction")
    within_algorithm: list[dict[str, Any]] = []
    interactions_by_model: dict[str, dict[str, dict[int, float]]] = {}
    for model in MODELS:
        model_effects: dict[str, dict[str, dict[int, float]]] = {}
        for algorithm in ALGORITHMS:
            per_effect: dict[str, dict[int, float]] = {name: {} for name in effect_names}
            raw: list[dict[str, Any]] = []
            for seed in SEEDS:
                row = indexed[(seed, model, algorithm)]
                effects = _factorial_effects(row)
                for name in effect_names:
                    per_effect[name][seed] = effects[name]
                raw.append(
                    {
                        "seed": seed,
                        "condition_cells": row["condition_cells"],
                        "effects": effects,
                    }
                )
            model_effects[algorithm] = per_effect
            within_algorithm.append(
                {
                    "model": model,
                    "algorithm": algorithm,
                    "raw_seed_cells": raw,
                    "noise_effects_on_law_control_margin": {
                        name: _summarize_seed_blocks(per_effect[name]) for name in effect_names
                    },
                }
            )
        interactions_by_model[model] = {
            name: {
                seed: model_effects["process_sft"][name][seed] - model_effects["outcome_rl"][name][seed]
                for seed in SEEDS
            }
            for name in effect_names
        }
    pooled = {
        name: {seed: _mean([interactions_by_model[model][name][seed] for model in MODELS]) for seed in SEEDS}
        for name in effect_names
    }
    return {
        "effect_orientation": "noise minus perfect; interaction is difference of differences",
        "outcome": "law_control_margin",
        "within_algorithm": within_algorithm,
        "sft_minus_outcome_rl_interactions": {
            "per_model": [
                {
                    "model": model,
                    **{
                        name: _summarize_seed_blocks(interactions_by_model[model][name])
                        for name in effect_names
                    },
                }
                for model in MODELS
            ],
            "pooled_equal_weight_across_models_within_seed": {
                name: _summarize_seed_blocks(pooled[name]) for name in effect_names
            },
        },
    }


def _view_gaps(indexed: Mapping[tuple[int, str, str], Mapping[str, Any]]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for model, algorithm in product(MODELS, ALGORITHMS):
        gaps: dict[str, dict[str, dict[int, float]]] = {
            comparison: {name: {} for name in ALL_ENDPOINTS}
            for comparison in ("active_minus_oracle_query", "active_minus_no_query")
        }
        raw: list[dict[str, Any]] = []
        for seed in SEEDS:
            views = indexed[(seed, model, algorithm)]["final_views"]
            seed_gaps: dict[str, dict[str, float]] = {}
            for comparison, reference in (
                ("active_minus_oracle_query", "oracle_query"),
                ("active_minus_no_query", "no_query"),
            ):
                values = {
                    name: float(views["active"][name]) - float(views[reference][name])
                    for name in ALL_ENDPOINTS
                }
                seed_gaps[comparison] = values
                for name, value in values.items():
                    gaps[comparison][name][seed] = value
            raw.append({"seed": seed, **seed_gaps})
        reports.append(
            {
                "model": model,
                "algorithm": algorithm,
                "raw_seed_gaps": raw,
                **{
                    comparison: {name: _summarize_seed_blocks(values) for name, values in outcomes.items()}
                    for comparison, outcomes in gaps.items()
                },
            }
        )
    return reports


def _formula_descriptions(
    indexed: Mapping[tuple[int, str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for model, algorithm in product(MODELS, ALGORITHMS):
        differences: dict[str, dict[int, float]] = {name: {} for name in ALL_ENDPOINTS}
        raw: list[dict[str, Any]] = []
        for seed in SEEDS:
            strata = indexed[(seed, model, algorithm)]["formula_strata"]
            values = {
                name: float(strata["monotone"][name]) - float(strata["exactly_one"][name])
                for name in ALL_ENDPOINTS
            }
            for name, value in values.items():
                differences[name][seed] = value
            raw.append(
                {
                    "seed": seed,
                    "monotone": strata["monotone"],
                    "exactly_one": strata["exactly_one"],
                    "monotone_minus_exactly_one": values,
                }
            )
        reports.append(
            {
                "model": model,
                "algorithm": algorithm,
                "raw_seed_strata": raw,
                "monotone_minus_exactly_one": {
                    name: _summarize_seed_blocks(differences[name]) for name in ALL_ENDPOINTS
                },
            }
        )
    return reports


def _model_descriptions(
    indexed: Mapping[tuple[int, str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for algorithm in ALGORITHMS:
        effects: dict[str, dict[int, float]] = {name: {} for name in ALL_ENDPOINTS}
        raw: list[dict[str, Any]] = []
        for seed in SEEDS:
            small = indexed[(seed, MODELS[0], algorithm)]
            large = indexed[(seed, MODELS[1], algorithm)]
            differences = {
                name: float(large["active"][name]) - float(small["active"][name]) for name in ALL_ENDPOINTS
            }
            for name, value in differences.items():
                effects[name][seed] = value
            raw.append(
                {
                    "seed": seed,
                    "qwen35_0_8b": _raw_endpoint(small),
                    "qwen35_2b": _raw_endpoint(large),
                    "two_b_minus_zero_point_eight_b": differences,
                }
            )
        reports.append(
            {
                "algorithm": algorithm,
                "raw_seed_pairs": raw,
                "two_b_minus_zero_point_eight_b": {
                    name: _summarize_seed_blocks(effects[name]) for name in ALL_ENDPOINTS
                },
            }
        )
    return reports


def _registered_descriptives(
    endpoints: Sequence[Mapping[str, Any]],
    trajectories: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Export fixed preregistered descriptions without adding inferential tests."""

    count_fields = (
        "forward_calls",
        "scored_prompt_count",
        "scored_prompt_tokens_unpadded",
        "maximum_prompt_tokens",
    )
    operational_rows = [
        {
            "seed": row["seed"],
            "model": row["model"],
            "algorithm": row["algorithm"],
            "run_id": row["run_id"],
            **dict(row["operational"]),
        }
        for row in endpoints
    ]
    operational_cells = []
    for model, algorithm in product(MODELS, ALGORITHMS):
        selected = [
            row for row in operational_rows if row["model"] == model and row["algorithm"] == algorithm
        ]
        if len(selected) != len(SEEDS):
            raise Qwen35HiddenLawAnalysisError("operational descriptions lack six seed runs per cell")
        operational_cells.append(
            {
                "model": model,
                "algorithm": algorithm,
                "run_count": len(selected),
                **{f"total_{field}": sum(int(row[field]) for row in selected) for field in count_fields[:-1]},
                **{f"mean_{field}": _mean([float(row[field]) for row in selected]) for field in count_fields},
            }
        )

    inquiry_rows = []
    for row in trajectories:
        inquiry_rows.append(
            {
                "seed": row["seed"],
                "model": row["model"],
                "algorithm": row["algorithm"],
                "run_id": row["run_id"],
                "checkpoints": [
                    {
                        "step": checkpoint["step"],
                        "episode_count": checkpoint["episode_count"],
                        "mean_query_count": checkpoint["mean_query_count"],
                        "declared_ready_rate": checkpoint["declared_ready_rate"],
                        "early_ready_rate": checkpoint["early_ready_rate"],
                        "query_turns": [dict(item) for item in checkpoint["query_turns"]],
                    }
                    for checkpoint in row["checkpoints"]
                ],
            }
        )

    selected_rule_rows = [
        {
            "seed": row["seed"],
            "model": row["model"],
            "algorithm": row["algorithm"],
            "run_id": row["run_id"],
            "rho_Y": row["active"]["rho_Y"],
            "rho_P": row["active"]["rho_P"],
            "rho_Q": row["active"]["rho_Q"],
            "rho_R": row["active"]["rho_R"],
            "law_control_margin": row["active"]["law_control_margin"],
            "named_proxy_only_margin": row["active"]["named_proxy_only_margin"],
            "stated_rule_behavioral_agreement": row["active"]["stated_rule_behavioral_agreement"],
            "selected_rule_distribution": dict(row["active"]["selected_rule_distribution"]),
        }
        for row in endpoints
    ]
    return {
        "status": "fixed_descriptive_only_no_additional_inference",
        "operational_counts": {
            "definitions": {
                "forward_calls": "number of model forward calls made by the constrained-scoring policy",
                "scored_prompt_count": "number of chat-rendered prompt rows passed to constrained scoring",
                "scored_prompt_tokens_unpadded": TOKEN_COUNT_DEFINITION,
                "maximum_prompt_tokens": "maximum unpadded chat-rendered prompt length scored in the run",
            },
            "per_run": operational_rows,
            "model_by_algorithm_descriptions": operational_cells,
        },
        "active_inquiry_by_checkpoint": {
            "definitions": {
                "turn_conditioned_denominator": (
                    "episodes that issued the numbered query; READY-before-query episodes are excluded"
                ),
                "all_episode_denominator": "all active-view episodes in that checkpoint panel",
                "expected_information_bits": (
                    "uniform-live-candidate expected entropy reduction for the chosen query"
                ),
                "realized_information_bits": "log2(live candidates before / live candidates after)",
                "regret_bits": (
                    "best available expected information minus chosen-query expected information"
                ),
                "declared_ready_rate": "fraction of active episodes in which the model chose READY",
                "early_ready_rate": (
                    "fraction of active episodes choosing READY while more than one candidate remained"
                ),
            },
            "per_run": inquiry_rows,
        },
        "stated_rule_and_behavioral_agreement": {
            "definitions": {
                "stated_rule_behavioral_agreement": (
                    "agreement on the 14 non-unanimous cells with the candidate explicitly selected"
                ),
                "named_proxy_only_margin": "rho_Y - max(rho_P, rho_Q)",
                "law_control_margin": "rho_Y - max(rho_P, rho_Q, rho_R)",
            },
            "per_run_final_active": selected_rule_rows,
        },
    }


def analyze_runs(
    runs: Sequence[LoadedRun],
    config: Mapping[str, Any],
    *,
    config_file_sha256: str,
    implementation_fingerprint: str,
    bank_builder: Callable[[int], ProductionBank] = build_production_bank,
) -> dict[str, Any]:
    """Validate the complete panel and compute every prospectively registered table."""

    _expected_plan(config)
    pairing, regenerated_banks = _validate_panel_pairing(runs, bank_builder=bank_builder)
    if any(
        run.implementation.get("implementation_fingerprint") != implementation_fingerprint for run in runs
    ):
        raise Qwen35HiddenLawAnalysisError("panel implementation binding differs from registration")
    extracted = [_validate_run_science(run, regenerated_banks[run.condition.seed]) for run in runs]
    endpoints = sorted(
        (item["endpoint"] for item in extracted),
        key=lambda row: (int(row["seed"]), str(row["model"]), str(row["algorithm"])),
    )
    trajectories = sorted(
        (item["trajectory"] for item in extracted),
        key=lambda row: (int(row["seed"]), str(row["model"]), str(row["algorithm"])),
    )
    registries = sorted(
        (item["formula_registry"] for item in extracted),
        key=lambda row: (int(row["seed"]), str(row["model"]), str(row["algorithm"])),
    )
    indexed = _endpoint_index(endpoints)
    return {
        "schema": "goalzendo.qwen35_hidden_law_analysis",
        "schema_version": 1,
        "panel": {
            "completed_only": True,
            "expected_run_count": 24,
            "observed_run_count": len(endpoints),
            "seed_blocks": list(SEEDS),
            "models": [{"name": model, "revision": MODEL_REVISIONS[model]} for model in MODELS],
            "algorithms": list(ALGORITHMS),
            "registered_eval_steps": list(EVAL_STEPS),
            "registered_final_views": list(FINAL_VIEWS),
            "config_file_sha256": config_file_sha256,
            "scientific_config_digest": canonical_digest(scientific_config(dict(config))),
            "protocol_sha256": REGISTERED_PROTOCOL_SHA256,
            "implementation_fingerprint": implementation_fingerprint,
            **pairing,
        },
        "inference": {
            "unit": "paired training-seed block",
            "seed_block_count": len(SEEDS),
            "bootstrap": {
                "method": "deterministic exhaustive seed-block bootstrap with replacement",
                "resamples": BOOTSTRAP_RESAMPLES,
                "interval": "two-sided 95% linear percentile",
            },
            "p_values": {
                "method": "exact two-sided paired sign flip over six seed blocks",
                "assignments": SIGN_FLIP_ASSIGNMENTS,
                "status": "descriptive_unadjusted",
                "binary_claims": False,
            },
            "pooling": "equal-weight average across model sizes within each seed before inference",
        },
        "interpretation_guard": {
            "claim_scope": "behavioral control in this interface, not internal objectives, motivations, beliefs, or representations",
            "candidate_policy": (
                "retain rho_Y, rho_P, rho_Q, rho_R and flip_Y, flip_P, flip_Q, flip_R, "
                "and flip_distractor; define control margins against the strongest non-Y candidate"
            ),
            "outcome_status": "all intervals and sign-flip values are descriptive and unadjusted; no binary discovery claim",
        },
        "primary_sft_minus_outcome_rl": _primary_contrast(indexed),
        "registered_secondary": {
            "p_q_evidence_factorial_on_law_control_margin": _secondary_factorial(indexed),
            "final_view_gaps": _view_gaps(indexed),
            "formula_family_descriptions": _formula_descriptions(indexed),
            "model_size_descriptions": _model_descriptions(indexed),
        },
        "registered_descriptives": _registered_descriptives(endpoints, trajectories),
        "seed_endpoints": endpoints,
        "seed_trajectories": trajectories,
        "candidate_formula_registry": registries,
    }


def analyze_main(artifacts: Path) -> dict[str, Any]:
    implementation_fingerprint = REGISTERED_IMPLEMENTATION_FINGERPRINT
    current_fingerprint = implementation_provenance(ROOT).get("implementation_fingerprint")
    if current_fingerprint != implementation_fingerprint:
        raise Qwen35HiddenLawAnalysisError(
            "current implementation source differs from the frozen registration"
        )
    if _sha256(MAIN_CONFIG) != REGISTERED_CONFIG_FILE_SHA256:
        raise Qwen35HiddenLawAnalysisError("main config bytes differ from the frozen registration")
    if _sha256(PROTOCOL) != REGISTERED_PROTOCOL_SHA256:
        raise Qwen35HiddenLawAnalysisError("protocol bytes differ from the frozen registration")
    config = load_hidden_law_config(MAIN_CONFIG)
    if config["experiment"]["status"] != "prospective_frozen":
        raise Qwen35HiddenLawAnalysisError("scientific config is not prospectively frozen")
    if canonical_digest(scientific_config(config)) != REGISTERED_SCIENTIFIC_CONFIG_DIGEST:
        raise Qwen35HiddenLawAnalysisError("scientific config differs from the frozen registration")
    runs = load_exact_runs(
        artifacts,
        config,
        registered_implementation_fingerprint=implementation_fingerprint,
    )
    return analyze_runs(
        runs,
        config,
        config_file_sha256=REGISTERED_CONFIG_FILE_SHA256,
        implementation_fingerprint=implementation_fingerprint,
    )


def canonical_json(value: Mapping[str, Any]) -> str:
    """Return one deterministic JSON document with no non-JSON float spellings."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze the exact completed 24-run Qwen3.5 hidden-Law panel"
    )
    parser.add_argument("artifacts", type=Path, help="root containing the scientific run artifacts")
    args = parser.parse_args(argv)
    print(canonical_json(analyze_main(args.artifacts)), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
