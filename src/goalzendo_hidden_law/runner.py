"""Resume-safe execution for the 24 finite-choice hidden-law conditions."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from goalzendo.modeling import load_model_and_tokenizer, model_provenance

from .artifacts import HiddenLawRunStore, implementation_provenance, read_json, write_json
from .config import (
    HiddenLawCondition,
    build_hidden_law_plan,
    canonical_digest,
    load_hidden_law_config,
    scientific_config,
)
from .evaluation import evaluate_checkpoint
from .experiment import (
    BINARY_ACTION_LABELS,
    CANDIDATE_ACTION_LABELS,
    BlockUpdate,
    CausalLMActionPolicy,
    train_role_neutral_block,
)
from .game import ProductionBank, build_production_bank


class HiddenLawRunnerError(RuntimeError):
    """Raised when execution cannot preserve the registered run identity."""


ModelLoader = Callable[[Mapping[str, Any], Mapping[str, Any] | None], tuple[nn.Module, Any]]
BankBuilder = Callable[[int], ProductionBank]
PolicyFactory = Callable[..., Any]


def shard_plan(
    plan: Sequence[HiddenLawCondition],
    *,
    shard_index: int,
    num_shards: int,
) -> tuple[HiddenLawCondition, ...]:
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must lie in [0, num_shards)")
    return tuple(condition for condition in plan if int(condition.plan_key, 16) % num_shards == shard_index)


def select_condition(plan: Sequence[HiddenLawCondition], reference: str) -> HiddenLawCondition:
    matches = [condition for condition in plan if reference in {condition.run_id, condition.plan_key}]
    if len(matches) != 1:
        raise HiddenLawRunnerError(
            f"condition reference must match exactly one run_id or plan_key; matches={len(matches)}"
        )
    return matches[0]


def smoke_condition(condition: HiddenLawCondition, smoke_seed: int) -> HiddenLawCondition:
    digest = canonical_digest(
        {
            "kind": "qwen35-hidden-law-execution-smoke-v1",
            "source_condition": condition.as_obj(),
            "smoke_seed": int(smoke_seed),
            "blocks": 1,
        }
    )
    return replace(condition, seed=int(smoke_seed), config_digest=digest)


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    payload = path.read_bytes()
    if payload and not payload.endswith(b"\n"):
        last_newline = payload.rfind(b"\n")
        payload = payload[: last_newline + 1] if last_newline >= 0 else b""
        _atomic_bytes(path, payload)
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HiddenLawRunnerError(f"invalid UTF-8 in complete JSONL rows: {path}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise HiddenLawRunnerError(f"invalid JSONL at {path}:{line_number}") from exc
        if type(value) is not dict:
            raise HiddenLawRunnerError(f"non-object JSONL row at {path}:{line_number}")
        rows.append(value)
    return rows


def _atomic_bytes(path: Path, payload: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    payload = "".join(json.dumps(dict(row), sort_keys=True, allow_nan=False) + "\n" for row in rows).encode(
        "utf-8"
    )
    _atomic_bytes(path, payload)


class _RecordSink:
    def __init__(self, store: HiddenLawRunStore) -> None:
        self.store = store
        self.rows = {
            "metrics": _jsonl_rows(store.path / "metrics.jsonl"),
            "transcripts": _jsonl_rows(store.path / "transcripts.jsonl"),
            "predictions": _jsonl_rows(store.path / "predictions.jsonl"),
        }
        self.identifiers: dict[str, set[str]] = {}
        for stream, rows in self.rows.items():
            identifiers = [row.get("record_id") for row in rows]
            if any(not isinstance(value, str) or not value for value in identifiers):
                raise HiddenLawRunnerError(f"{stream} contains a row without a record_id")
            if len(set(identifiers)) != len(identifiers):
                raise HiddenLawRunnerError(f"{stream} contains duplicate record_ids")
            self.identifiers[stream] = {value for value in identifiers if isinstance(value, str)}

    def truncate_after_step(self, step: int) -> None:
        """Drop uncheckpointed suffix records before deterministic replay."""

        if step < 0:
            raise ValueError("checkpoint step cannot be negative")
        for stream, rows in self.rows.items():
            retained: list[dict[str, Any]] = []
            for row in rows:
                row_step = row.get("step")
                if isinstance(row_step, bool) or not isinstance(row_step, int) or row_step < 0:
                    raise HiddenLawRunnerError(f"{stream} row has no valid non-negative integer step")
                if row_step <= step:
                    retained.append(row)
            if len(retained) != len(rows):
                _atomic_jsonl(self.store.path / f"{stream}.jsonl", retained)
            self.rows[stream] = retained
            self.identifiers[stream] = {str(row["record_id"]) for row in retained}

    def append(self, stream: str, rows: Sequence[Mapping[str, Any]]) -> int:
        novel: list[Mapping[str, Any]] = []
        for row in rows:
            record_id = row.get("record_id")
            if not isinstance(record_id, str) or not record_id:
                raise HiddenLawRunnerError(f"new {stream} row lacks a record_id")
            if record_id not in self.identifiers[stream]:
                novel.append(row)
                self.identifiers[stream].add(record_id)
                self.rows[stream].append(dict(row))
        if not novel:
            return 0
        append = getattr(self.store, f"append_{stream}")
        return int(append(novel))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _atomic_torch_save(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(dict(value), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return _file_sha256(path)


def _checkpoint_binding(
    condition: HiddenLawCondition,
    implementation: Mapping[str, Any],
    bank: ProductionBank,
) -> dict[str, str]:
    return {
        "run_id": condition.run_id,
        "plan_key": condition.plan_key,
        "config_digest": condition.config_digest,
        "implementation_fingerprint": str(implementation["implementation_fingerprint"]),
        "bank_digest": bank.digest,
    }


_POLICY_COUNTER_FIELDS = (
    "forward_calls",
    "scored_prompt_count",
    "scored_prompt_tokens_unpadded",
    "maximum_prompt_tokens",
)


def _policy_counter_state(policy: Any) -> dict[str, int]:
    state: dict[str, int] = {}
    for field in _POLICY_COUNTER_FIELDS:
        value = getattr(policy, field, None)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise HiddenLawRunnerError(f"policy counter {field} is absent or invalid")
        state[field] = value
    return state


def _restore_policy_counters(policy: Any, value: object) -> None:
    if type(value) is not dict or set(value) != set(_POLICY_COUNTER_FIELDS):
        raise HiddenLawRunnerError("checkpoint policy counters are incomplete")
    counters = cast(dict[str, Any], value)
    for field in _POLICY_COUNTER_FIELDS:
        observed = counters[field]
        if isinstance(observed, bool) or not isinstance(observed, int) or observed < 0:
            raise HiddenLawRunnerError(f"checkpoint policy counter {field} is invalid")
        setattr(policy, field, observed)


def save_checkpoint(
    directory: Path,
    *,
    step: int,
    binding: Mapping[str, str],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    action_generator: torch.Generator,
    policy: Any,
) -> Path:
    target = directory / f"step-{step:08d}.pt"
    digest = _atomic_torch_save(
        target,
        {
            "schema": "goalzendo.hidden_law_checkpoint",
            "schema_version": 1,
            "step": int(step),
            "binding": dict(binding),
            "model_state": _cpu_tree(model.state_dict()),
            "optimizer_state": _cpu_tree(optimizer.state_dict()),
            "action_generator_state": action_generator.get_state(),
            "policy_counters": _policy_counter_state(policy),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                [state.cpu() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []
            ),
        },
    )
    previous: Path | None = None
    latest = directory / "latest.json"
    if latest.is_file():
        value = read_json(latest)
        name = value.get("file")
        if isinstance(name, str) and Path(name).name == name:
            previous = directory / name
    write_json(
        latest,
        {
            "schema": "goalzendo.hidden_law_checkpoint_pointer",
            "schema_version": 1,
            "step": int(step),
            "file": target.name,
            "sha256": digest,
            "binding": dict(binding),
        },
    )
    if previous is not None and previous != target:
        previous.unlink(missing_ok=True)
    return target


def load_latest_checkpoint(
    directory: Path,
    *,
    binding: Mapping[str, str],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    action_generator: torch.Generator,
    policy: Any,
    device: torch.device,
) -> int:
    latest = directory / "latest.json"
    if not latest.is_file():
        return 0
    pointer = read_json(latest)
    if pointer.get("binding") != dict(binding):
        raise HiddenLawRunnerError("checkpoint pointer binding mismatch")
    name = pointer.get("file")
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise HiddenLawRunnerError("checkpoint pointer contains an unsafe filename")
    path = directory / name
    if not path.is_file() or _file_sha256(path) != pointer.get("sha256"):
        raise HiddenLawRunnerError("checkpoint file is absent or changed")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover - older supported Torch
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, Mapping) or checkpoint.get("binding") != dict(binding):
        raise HiddenLawRunnerError("checkpoint payload binding mismatch")
    step = int(checkpoint.get("step", -1))
    if step != int(pointer.get("step", -2)) or step < 0:
        raise HiddenLawRunnerError("checkpoint step mismatch")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    action_generator.set_state(checkpoint["action_generator_state"].cpu())
    _restore_policy_counters(policy, checkpoint.get("policy_counters"))
    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all"):
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
    return step


def retire_checkpoints(directory: Path) -> bool:
    """Best-effort bounded cleanup after the immutable COMPLETE seal exists."""

    if not directory.exists():
        return True
    try:
        children = list(directory.iterdir())
        expected = all(
            child.is_file()
            and (
                child.name == "latest.json"
                or (
                    child.name.startswith("step-")
                    and child.name.endswith(".pt")
                    and len(child.name[5:-3]) == 8
                    and child.name[5:-3].isdigit()
                )
            )
            for child in children
        )
        if not expected:
            return False
        for child in children:
            child.unlink()
        directory.rmdir()
        return True
    except OSError:
        return False


def _model_config(config: Mapping[str, Any], condition: HiddenLawCondition) -> dict[str, Any]:
    matches = [item for item in config["models"] if item["name"] == condition.model_name]
    if len(matches) != 1:
        raise HiddenLawRunnerError("condition model is absent or duplicated in config")
    result = dict(matches[0])
    if result["revision"] != condition.model_revision or result["dtype"] != condition.model_dtype:
        raise HiddenLawRunnerError("condition model identity differs from config")
    return result


def _algorithm_config(config: Mapping[str, Any], condition: HiddenLawCondition) -> dict[str, Any]:
    matches = [item for item in config["algorithms"] if item["name"] == condition.algorithm]
    if len(matches) != 1:
        raise HiddenLawRunnerError("condition algorithm is absent or duplicated in config")
    result = dict(matches[0])
    if (
        float(result["learning_rate"]) != condition.learning_rate
        or float(result["entropy_coefficient"]) != condition.entropy_coefficient
    ):
        raise HiddenLawRunnerError("condition optimizer identity differs from config")
    return result


def _resolve_device(requested: str | torch.device) -> torch.device:
    if isinstance(requested, torch.device):
        return requested
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise HiddenLawRunnerError("CUDA was requested but is unavailable")
    return device


def _configure_model(model: nn.Module, *, strict: bool) -> None:
    model.requires_grad_(True)
    enable = getattr(model, "gradient_checkpointing_enable", None)
    if callable(enable):
        enable()
    elif strict:
        raise HiddenLawRunnerError("model does not support registered gradient checkpointing")
    runtime_config = getattr(model, "config", None)
    if runtime_config is not None and hasattr(runtime_config, "use_cache"):
        runtime_config.use_cache = False
    elif strict:
        raise HiddenLawRunnerError("model does not expose registered use_cache control")


def _validate_loaded_model_provenance(
    provenance: Mapping[str, Any],
    condition: HiddenLawCondition,
    *,
    strict_backend_identity: bool,
) -> None:
    expected = {
        "requested_model": condition.model_name,
        "requested_revision": condition.model_revision,
        "requested_dtype": condition.model_dtype,
        "action_labels": list(BINARY_ACTION_LABELS),
        "finite_action_alphabets": {
            "binary": list(BINARY_ACTION_LABELS),
            "candidate": list(CANDIDATE_ACTION_LABELS),
            "query": list("ABCDEFGHI"),
        },
        "gradient_checkpointing": True,
        "use_cache": False,
        "full_model_update": True,
    }
    mismatched = [key for key, value in expected.items() if provenance.get(key) != value]
    for key in ("resolved_revision", "tokenizer_resolved_revision"):
        if provenance.get(key) not in {None, condition.model_revision}:
            mismatched.append(key)
    parameter_count = provenance.get("parameter_count")
    trainable_count = provenance.get("trainable_parameter_count")
    if mismatched or type(parameter_count) is not int or parameter_count <= 0:
        raise HiddenLawRunnerError(
            "loaded model or tokenizer differs from the registered identity: "
            + ", ".join(mismatched or ["parameter_count"])
        )
    if trainable_count != parameter_count:
        raise HiddenLawRunnerError("loaded model is not an exact full-model update")
    if not strict_backend_identity:
        return
    expected_parameter_counts = {
        "Qwen/Qwen3.5-0.8B": 752_393_024,
        "Qwen/Qwen3.5-2B": 1_881_825_088,
    }
    expected_dependencies = {
        "transformers": "5.15.0",
        "tokenizers": "0.22.2",
        "peft": "0.20.0",
        "accelerate": "1.14.0",
        "huggingface_hub": "1.5.0",
        "safetensors": "0.8.0",
    }
    dependencies = provenance.get("dependency_versions")
    if (
        provenance.get("model_class") != "transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5ForCausalLM"
        or provenance.get("tokenizer_name_or_path") != condition.model_name
        or type(provenance.get("tokenizer_class")) is not str
        or not provenance["tokenizer_class"]
        or parameter_count != expected_parameter_counts.get(condition.model_name)
        or provenance.get("trainable_parameter_dtype_counts") != {"torch.bfloat16": parameter_count}
        or type(dependencies) is not dict
        or any(dependencies.get(key) != value for key, value in expected_dependencies.items())
        or str(dependencies.get("torch", "")).split("+", 1)[0] != "2.8.0"
        or dependencies.get("torch_cuda") != "12.8"
    ):
        raise HiddenLawRunnerError("loaded Qwen runtime differs from the pinned execution stack")


def _optimization_rows(update: BlockUpdate, *, smoke: bool) -> list[dict[str, Any]]:
    base: dict[str, Any] = {
        "record_id": f"train:{update.step:04d}:block",
        "kind": "smoke_update" if smoke else "optimization_block",
        "step": update.step,
        "algorithm": update.algorithm,
        "loss": update.loss,
        "learning_rate": update.learning_rate,
        "gradient_norm": update.gradient_norm,
    }
    if not smoke:
        base["rotation_metrics"] = [dict(row) for row in update.rotation_metrics]
    return [base]


def _trajectory_rows(update: BlockUpdate, *, smoke: bool) -> list[dict[str, Any]]:
    if smoke:
        return []
    return [
        {
            "record_id": f"train:{update.step:04d}:trajectory:{index:03d}",
            "kind": "training_trajectory",
            "step": update.step,
            "algorithm": update.algorithm,
            "trajectory": trajectory.as_obj(),
        }
        for index, trajectory in enumerate(update.trajectories)
    ]


def execute_condition(
    condition: HiddenLawCondition,
    config: Mapping[str, Any],
    *,
    repo_root: str | Path,
    output_root: str | Path | None = None,
    device: str | torch.device = "auto",
    smoke: bool = False,
    bank_builder: BankBuilder = build_production_bank,
    model_loader: ModelLoader = load_model_and_tokenizer,
    policy_factory: PolicyFactory = CausalLMActionPolicy,
    strict_runtime: bool = True,
) -> dict[str, Any]:
    """Execute or resume one exact condition, with no outcome-dependent branch."""

    root = Path(repo_root).resolve()
    observed_config_digest = canonical_digest(scientific_config(dict(config)))
    if not smoke and condition.config_digest != observed_config_digest:
        raise HiddenLawRunnerError("condition is not bound to the supplied scientific config")
    implementation = implementation_provenance(root)
    resolved_output = Path(output_root or config["run"]["output_root"])
    if smoke:
        resolved_output = resolved_output / "execution-smoke"
    store = HiddenLawRunStore(resolved_output, condition, config, implementation)
    mode = store.initialize(resume=bool(config["training"]["resume"]))
    if mode == "complete":
        retired = retire_checkpoints(store.path / "checkpoints")
        return {
            "run_id": condition.run_id,
            "status": "skipped_complete",
            "path": str(store.path),
            "checkpoint_retired": retired,
        }

    last_step = 0
    try:
        sink = _RecordSink(store)
        bank = bank_builder(condition.seed)
        store.write_bank_manifest(
            {
                "schema": "goalzendo.hidden_law_bank_manifest",
                "schema_version": 1,
                "bank_digest": bank.digest,
                "pairing_key": bank.pairing_key,
                "manifest": bank.manifest,
            }
        )

        target_device = _resolve_device(device)
        torch.manual_seed(condition.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(condition.seed)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        model_config = _model_config(config, condition)
        algorithm_config = _algorithm_config(config, condition)
        if Path(condition.model_name).exists():
            raise HiddenLawRunnerError("registered Hugging Face model name is shadowed by a local path")
        model, tokenizer = model_loader(model_config, {"method": "full"})
        model.to(target_device)
        _configure_model(model, strict=strict_runtime)
        provenance = model_provenance(model, tokenizer, model_config, BINARY_ACTION_LABELS)
        provenance.update(
            {
                "finite_action_alphabets": {
                    "binary": list(BINARY_ACTION_LABELS),
                    "candidate": list(CANDIDATE_ACTION_LABELS),
                    "query": list("ABCDEFGHI"),
                },
                "device": str(target_device),
                "gradient_checkpointing": True,
                "use_cache": False,
                "full_model_update": True,
                "trainable_parameter_dtype_counts": {
                    str(dtype): sum(
                        parameter.numel()
                        for parameter in model.parameters()
                        if parameter.requires_grad and parameter.dtype == dtype
                    )
                    for dtype in sorted(
                        {parameter.dtype for parameter in model.parameters() if parameter.requires_grad},
                        key=str,
                    )
                },
            }
        )
        _validate_loaded_model_provenance(
            provenance,
            condition,
            strict_backend_identity=model_loader is load_model_and_tokenizer,
        )
        store.write_model_manifest(provenance)

        training = config["training"]
        microbatch = int(training["scoring_microbatch"])
        policy = policy_factory(
            model,
            tokenizer,
            max_prompt_tokens=int(training["max_prompt_tokens"]),
            max_batch_size=microbatch,
        )
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=condition.learning_rate,
            weight_decay=float(training["weight_decay"]),
        )
        action_generator = torch.Generator(device="cpu")
        action_generator.manual_seed(condition.seed + 8_000_003)
        binding = _checkpoint_binding(condition, implementation, bank)
        checkpoint_directory = store.path / "checkpoints"
        start_step = load_latest_checkpoint(
            checkpoint_directory,
            binding=binding,
            model=model,
            optimizer=optimizer,
            action_generator=action_generator,
            policy=policy,
            device=target_device,
        )
        sink.truncate_after_step(start_step)
        last_step = start_step

        total_steps = 1 if smoke else int(training["steps"])
        evaluation_steps = set() if smoke else set(int(value) for value in training["evaluation_steps"])
        checkpoint_steps = {1} if smoke else set(int(value) for value in training["checkpoint_steps"])
        if start_step > total_steps:
            raise HiddenLawRunnerError("checkpoint is beyond the requested endpoint")

        latest_evaluation: Mapping[str, Any] | None = None
        if start_step == 0 and 0 in evaluation_steps:
            model.eval()
            result = evaluate_checkpoint(
                policy,
                bank,
                step=0,
                final=False,
                eval_renderers=tuple(config["game"]["eval_renderers"]),
            )
            sink.append("metrics", result.metric_rows)
            sink.append("transcripts", result.transcript_rows)
            sink.append("predictions", result.prediction_rows)
            latest_evaluation = result.summary
            store.write_status(state="running", phase="evaluated", last_step=0)

        model.train()
        for step in range(start_step + 1, total_steps + 1):
            family = bank.training_families[step - 1]
            renderer = config["game"]["train_renderers"][(step - 1) % 4]
            update = train_role_neutral_block(
                step=step,
                algorithm=condition.algorithm,
                policy=policy,
                model=model,
                optimizer=optimizer,
                games=family.training_games,
                renderer=renderer,
                generator=action_generator,
                base_learning_rate=condition.learning_rate,
                warmup=int(training["warmup_steps"]),
                gradient_clip_norm=float(training["gradient_clip_norm"]),
                entropy_coefficient=condition.entropy_coefficient,
                trajectories_per_official=int(algorithm_config.get("trajectories_per_official", 4)),
            )
            last_step = step
            sink.append("metrics", _optimization_rows(update, smoke=smoke))
            sink.append("transcripts", _trajectory_rows(update, smoke=smoke))
            store.write_status(state="running", phase="trained", last_step=step)

            if step in evaluation_steps:
                model.eval()
                result = evaluate_checkpoint(
                    policy,
                    bank,
                    step=step,
                    final=step == total_steps,
                    eval_renderers=tuple(config["game"]["eval_renderers"]),
                )
                sink.append("metrics", result.metric_rows)
                sink.append("transcripts", result.transcript_rows)
                sink.append("predictions", result.prediction_rows)
                latest_evaluation = result.summary
                store.write_status(state="running", phase="evaluated", last_step=step)
                model.train()

            if step in checkpoint_steps:
                save_checkpoint(
                    checkpoint_directory,
                    step=step,
                    binding=binding,
                    model=model,
                    optimizer=optimizer,
                    action_generator=action_generator,
                    policy=policy,
                )
                store.write_status(state="running", phase="checkpointed", last_step=step)

        if latest_evaluation is None and not smoke:
            summaries = [
                row
                for row in sink.rows["metrics"]
                if row.get("kind") == "evaluation_checkpoint_summary"
                and int(row.get("step", -1)) == total_steps
            ]
            if len(summaries) != 1:
                raise HiddenLawRunnerError("final evaluation summary is absent or duplicated")
            latest_evaluation = summaries[0]

        summary = {
            "schema": "goalzendo.hidden_law_run_summary",
            "schema_version": 1,
            "run_id": condition.run_id,
            "plan_key": condition.plan_key,
            "smoke": smoke,
            "last_step": total_steps,
            "algorithm": condition.algorithm,
            "model_name": condition.model_name,
            "seed": condition.seed,
            "bank_digest": bank.digest,
            "evaluation": None if smoke else latest_evaluation,
            "operational": {
                "device": str(target_device),
                "forward_calls": int(getattr(policy, "forward_calls", 0)),
                "scored_prompt_count": int(getattr(policy, "scored_prompt_count", 0)),
                "scored_prompt_tokens_unpadded": int(getattr(policy, "scored_prompt_tokens_unpadded", 0)),
                "token_count_definition": (
                    "sum of chat-rendered prompt token lengths for every scored call; "
                    "padding and action-continuation tokens excluded"
                ),
                "maximum_prompt_tokens": int(getattr(policy, "maximum_prompt_tokens", 0)),
            },
        }
        store.finish(summary)
        retired = retire_checkpoints(checkpoint_directory)
        return {
            "run_id": condition.run_id,
            "status": "complete",
            "path": str(store.path),
            "checkpoint_retired": retired,
        }
    except Exception as exc:
        store.write_status(
            state="failed",
            phase="exception",
            last_step=last_step,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise


def execute_plan(
    config_path: str | Path,
    *,
    repo_root: str | Path,
    condition_reference: str | None = None,
    shard_index: int | None = None,
    num_shards: int | None = None,
    output_root: str | Path | None = None,
    device: str | torch.device = "auto",
    smoke: bool = False,
) -> list[dict[str, Any]]:
    config = load_hidden_law_config(config_path)
    if not smoke and config["experiment"]["status"] != "prospective_frozen":
        raise HiddenLawRunnerError("scientific execution requires a prospectively frozen config")
    plan = build_hidden_law_plan(config)
    selected: tuple[HiddenLawCondition, ...]
    if condition_reference is not None:
        selected = (select_condition(plan, condition_reference),)
    else:
        if shard_index is None or num_shards is None:
            raise HiddenLawRunnerError("provide one exact --condition or both --shard-index and --num-shards")
        selected = shard_plan(plan, shard_index=shard_index, num_shards=num_shards)
    if smoke:
        if len(selected) != 1:
            raise HiddenLawRunnerError("execution smoke requires exactly one selected condition")
        selected = (smoke_condition(selected[0], int(config["run"]["smoke_seed"])),)

    results: list[dict[str, Any]] = []
    for condition in selected:
        results.append(
            execute_condition(
                condition,
                config,
                repo_root=repo_root,
                output_root=output_root,
                device=device,
                smoke=smoke,
            )
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results
