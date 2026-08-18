"""Frozen, nonauthorizing full-model smoke runner for G03-G.

The runner deliberately exposes only operational paths, the externally frozen
artifact digest, and one of the two preregistered cells.  All scientific and
optimizer coordinates are constants.  A successful call writes a complete,
content-addressed artifact root; it never authorizes capability or production
weight updates.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import stat
import sys
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal, cast

import torch

from ._json import CanonicalJSONError, dump_json, json_digest, load_json
from .action_tokenization_v2 import ExactDecodeTokenizerProtocol, FragmentActionTokenCompiler
from .actions import ReadyAction
from .authenticated_model_provider_v2 import (
    AuthenticatedIncrementalCacheProvider,
    IncrementalCacheComparison,
    build_model_artifact_manifest,
    compare_incremental_cache_trace,
)
from .authenticated_rollouts_v2 import collect_eight_authenticated_rollouts
from .authenticated_step_v1 import (
    AdamWOptimizerSpec,
    AuthenticatedStepEvidenceBundle,
    AuthenticatedUpdateCoordinate,
    execute_authenticated_rl_step,
    execute_authenticated_sft_step,
)
from .dialogue import DialogueMessage
from .episodes import HiddenEpisode
from .generation import EpisodeBank, parse_episode_bank
from .provenance import (
    InteractiveSourceProvenance,
    interactive_source_provenance,
    verify_interactive_source_provenance,
)
from .sealed_runtime_v1 import (
    G03_QWEN_MODEL_IDENTIFIER,
    G03_QWEN_REVISION,
    SealedQwenLoadConfig,
    SealedQwenRuntime,
    load_sealed_qwen_runtime,
)
from .streaming_objective_v3 import derive_verified_streaming_objective_plan
from .streaming_sft_v3 import ReferenceTrajectorySFTSource, build_streaming_sft_plan
from .trajectory_banks import ReferenceTrajectoryRecord, generate_reference_trajectory_bank
from .trajectory_encoding import render_generation_prefix

PINNED_QWEN_SMOKE_SCHEMA_VERSION = 1
PINNED_QWEN_SMOKE_REPORT_ID = "g03-g-pinned-qwen-model-pipeline-smoke-v1"
PINNED_QWEN_SMOKE_AUTHORIZES_EXECUTION = False

TOKENIZER_BINDING_DIGEST = "5acec12f5fb95d87c73d445f38aedc56e7b4991c78f51f243ba674206f6d0453"
COMPILER_MANIFEST_DIGEST = "d13f5c9325c031c8f4e965ee2ca6bb51542020edd81094ee4d82bb92595eb00c"
EPISODE_BANK_DIGEST = "54714717b307b142a123372f5bde9854fd42df6ba8dc4e8e9c79bb78fe231631"
EPISODE_FIXTURE_FILE_SHA256 = "2e95f7de1e4a6e859c2cd116d453e00c087d4fa85223d9d2fb7cc0cbdc6ff52d"
SMOKE_EPISODE_ID = "g03-engine-small-fixture-v1-p1-0-any-perfect-factorial-s03-ac92fe83ea-33a184e706"
SMOKE_EPISODE_DIGEST = "a8081c54afca80fb55221367c493d967ddfdec02c3985a142e5a1de441ad3cc6"
SMOKE_REFERENCE_TRANSCRIPT_DIGEST = "50ecef80f3f27621540b60451f1d3515c84ce7ade7c24b1b6a9771b156e25af7"
SMOKE_CACHE_TRACE_DIGEST = "ffbb9a7112feb1fc0e43fdb1a54f7b7c52f9ebd1039055f2e9c5e2bde029163e"

TRANSFORMERS_VERSION = "4.57.6"
TOKENIZERS_VERSION = "0.22.2"
TORCH_VERSION = "2.8.0"
ACCELERATE_VERSION = "1.14.0"
HUGGINGFACE_HUB_VERSION = "0.36.2"
PEFT_VERSION = "0.20.0"
SAFETENSORS_VERSION = "0.8.0"
PYTHON_VERSION = (3, 12, 3)
MAXIMUM_ACTION_TOKENS = 2_048
MAXIMUM_SEQUENCE_TOKENS = 8_192
MAXIMUM_TURNS = 6
CACHE_LOGIT_ABSOLUTE_TOLERANCE = 0.125
REPLAY_STATISTIC_ABSOLUTE_TOLERANCE = 0.02
RL_TEMPERATURE = 0.7
RL_ENTROPY_COEFFICIENT = Fraction(1, 100)
SFT_RUN_SEED = 30_201
RL_RUN_SEED = 30_202

_CACHE_TRACE_DOMAIN = "goalzendo-interactive-incremental-cache-trace-v1"
_PREFLIGHT_DOMAIN = "goalzendo-interactive-pinned-qwen-smoke-preflight-v1"
_REPORT_DOMAIN = "goalzendo-interactive-pinned-qwen-smoke-report-v1"
_COMPLETION_DOMAIN = "goalzendo-interactive-pinned-qwen-smoke-completion-v1"
_INTERACTIVE_PACKAGE_ROOT = Path(__file__).resolve().parent

_BASE_COMPLETION_ARTIFACT_PATHS = (
    "cache-comparison.json",
    "environment.json",
    "episode.json",
    "objective-plan.json",
    "reference-trajectory.json",
    "report.json",
    "runtime-post.json",
    "runtime-pre.json",
    "source-provenance.json",
    "step-evidence.json",
)
_COMPLETION_CONTROL_PATHS = ("COMPLETE", "completion-attestation.json")
_COMPLETION_KEYS = (
    "schema_version",
    "report_id",
    "cell_kind",
    "report_digest",
    "source_fingerprint",
    "files",
    "status",
    "authorization",
    "digest",
)
_REPORT_KEYS = (
    "schema_version",
    "report_id",
    "cell_kind",
    "run_id",
    "run_seed",
    "source_fingerprint",
    "artifact_manifest_digest",
    "episode_fixture_file_sha256",
    "episode_bank_digest",
    "episode_id",
    "episode_digest",
    "reference_transcript_digest",
    "tokenizer_binding_digest",
    "compiler_manifest_digest",
    "cache_comparison_digest",
    "cache_maximum_absolute_error_hex",
    "collection_cache_metrics",
    "rollout_digests",
    "rollout_count",
    "plan_digest",
    "step_evidence_bundle_digest",
    "step_record_digest",
    "pre_policy_state_digest",
    "post_policy_state_digest",
    "changed_parameter_count",
    "optimizer_step_call_count",
    "cuda_peak_allocated_bytes",
    "cuda_peak_reserved_bytes",
    "elapsed_seconds_hex",
    "status",
    "authorization",
    "digest",
)

CellKind = Literal["sft", "rl"]


class PinnedQwenSmokeError(RuntimeError):
    """Raised whenever the prospectively frozen smoke fails closed."""


@dataclass(frozen=True, slots=True)
class PinnedQwenSmokeCompletionFile:
    """One immutable file record from a verified v1 completion seal."""

    path: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.path) is not str
            or not self.path
            or self.path.startswith("/")
            or "\\" in self.path
            or any(part in {"", ".", ".."} for part in self.path.split("/"))
        ):
            raise PinnedQwenSmokeError("completion file path is not canonical and relative")
        try:
            self.path.encode("ascii")
        except UnicodeEncodeError as exc:
            raise PinnedQwenSmokeError("completion file path must be ASCII") from exc
        if type(self.size) is not int or self.size < 0:
            raise PinnedQwenSmokeError("completion file size must be a non-negative integer")
        _require_sha256(self.sha256, name="completion file sha256")

    def as_obj(self) -> dict[str, object]:
        return {"path": self.path, "size": self.size, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class VerifiedPinnedQwenSmokeCompletion:
    """Nominal result issued only after strict external-digest verification."""

    output_root: Path
    cell_kind: CellKind
    completion_digest: str
    report_digest: str
    source_fingerprint: str
    files: tuple[PinnedQwenSmokeCompletionFile, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.output_root, Path):
            raise TypeError("output_root must be a pathlib.Path")
        if self.cell_kind not in {"sft", "rl"}:
            raise PinnedQwenSmokeError("verified cell_kind must be exactly sft or rl")
        _require_sha256(self.completion_digest, name="verified completion digest")
        _require_sha256(self.report_digest, name="verified report digest")
        _require_sha256(self.source_fingerprint, name="verified source fingerprint")
        if type(self.files) is not tuple or any(
            type(record) is not PinnedQwenSmokeCompletionFile for record in self.files
        ):
            raise PinnedQwenSmokeError("verified completion files must be nominal records")
        paths = tuple(record.path for record in self.files)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise PinnedQwenSmokeError("verified completion files must have unique sorted paths")


def _require_sha256(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PinnedQwenSmokeError(f"{name} must be a lowercase SHA-256")
    return value


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_with_digest(value: dict[str, Any], *, domain: str) -> str:
    return dump_json({**value, "digest": json_digest(value, domain=domain)})


def _canonical_absolute_path(path: Path, *, name: str) -> Path:
    if not isinstance(path, Path):
        raise TypeError(f"{name} must be a pathlib.Path")
    if not path.is_absolute():
        raise PinnedQwenSmokeError(f"{name} must be absolute")
    try:
        return path.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise PinnedQwenSmokeError(f"{name} could not be canonically resolved") from exc


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _canonical_frozen_input_roots(
    artifact_root: Path,
    episode_fixture_path: Path,
) -> tuple[Path, Path, Path]:
    canonical_artifact = _canonical_absolute_path(artifact_root, name="artifact_root")
    canonical_fixture = _canonical_absolute_path(
        episode_fixture_path,
        name="episode_fixture_path",
    )
    canonical_fixture_root = canonical_fixture.parent
    protected_roots = (
        ("artifact root", canonical_artifact),
        ("interactive package root", _INTERACTIVE_PACKAGE_ROOT),
        ("episode fixture root", canonical_fixture_root),
    )
    for index, (first_name, first_path) in enumerate(protected_roots):
        for second_name, second_path in protected_roots[index + 1 :]:
            if _paths_overlap(first_path, second_path):
                raise PinnedQwenSmokeError(f"{first_name} and {second_name} must be canonically disjoint")
    return canonical_artifact, _INTERACTIVE_PACKAGE_ROOT, canonical_fixture_root


def validate_pinned_qwen_smoke_path_separation(
    artifact_root: Path,
    episode_fixture_path: Path,
    output_path: Path,
) -> None:
    """Reject every equality/ancestor/descendant alias among smoke coordinates."""

    protected_roots = _canonical_frozen_input_roots(artifact_root, episode_fixture_path)
    canonical_output = _canonical_absolute_path(output_path, name="output_path")
    protected_names = ("artifact root", "interactive package root", "episode fixture root")
    for protected_name, protected_root in zip(protected_names, protected_roots, strict=True):
        if _paths_overlap(canonical_output, protected_root):
            raise PinnedQwenSmokeError(f"output path and {protected_name} must be canonically disjoint")


def build_pinned_qwen_smoke_preflight(
    artifact_root: Path,
    episode_fixture_path: Path,
) -> str:
    """Hash frozen inputs in a separate process before any model load."""

    if not isinstance(artifact_root, Path) or not isinstance(episode_fixture_path, Path):
        raise TypeError("preflight paths must be pathlib.Path values")
    _canonical_frozen_input_roots(artifact_root, episode_fixture_path)
    artifact = build_model_artifact_manifest(
        artifact_root,
        model_identifier=G03_QWEN_MODEL_IDENTIFIER,
        revision=G03_QWEN_REVISION,
    )
    source = interactive_source_provenance()
    fixture_sha256 = _file_sha256(episode_fixture_path)
    if fixture_sha256 != EPISODE_FIXTURE_FILE_SHA256:
        raise PinnedQwenSmokeError("preflight episode fixture differs from the frozen file")
    value: dict[str, Any] = {
        "schema_version": PINNED_QWEN_SMOKE_SCHEMA_VERSION,
        "report_id": f"{PINNED_QWEN_SMOKE_REPORT_ID}-preflight",
        "model_identifier": G03_QWEN_MODEL_IDENTIFIER,
        "model_revision": G03_QWEN_REVISION,
        "artifact_manifest": {
            **artifact.as_obj(),
            "digest": artifact.digest,
        },
        "interactive_source_provenance": source.as_obj(),
        "episode_fixture_file_sha256": fixture_sha256,
        "episode_bank_digest": EPISODE_BANK_DIGEST,
        "episode_id": SMOKE_EPISODE_ID,
        "episode_digest": SMOKE_EPISODE_DIGEST,
        "reference_transcript_digest": SMOKE_REFERENCE_TRANSCRIPT_DIGEST,
        "tokenizer_binding_digest": TOKENIZER_BINDING_DIGEST,
        "compiler_manifest_digest": COMPILER_MANIFEST_DIGEST,
        "cache_trace_digest": SMOKE_CACHE_TRACE_DIGEST,
        "cells": ["sft", "rl"],
        "authorization": {
            "model_load": False,
            "weight_updates": False,
            "scientific_launch": False,
        },
    }
    return _canonical_with_digest(value, domain=_PREFLIGHT_DOMAIN)


def configure_frozen_cuda_runtime() -> None:
    """Configure the preregistered deterministic arithmetic before CUDA use."""

    if cast(Any, torch.cuda).is_initialized():
        raise PinnedQwenSmokeError("CUDA initialized before deterministic runtime configuration")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if os.environ["CUBLAS_WORKSPACE_CONFIG"] != ":4096:8":
        raise PinnedQwenSmokeError("CUBLAS_WORKSPACE_CONFIG differs from the frozen value")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False


def _require_frozen_stack() -> dict[str, object]:
    versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "tokenizers": importlib.metadata.version("tokenizers"),
        "accelerate": importlib.metadata.version("accelerate"),
        "huggingface_hub": importlib.metadata.version("huggingface-hub"),
        "peft": importlib.metadata.version("peft"),
        "safetensors": importlib.metadata.version("safetensors"),
    }
    if sys.version_info[:3] != PYTHON_VERSION:
        raise PinnedQwenSmokeError("Python differs from the prospectively frozen version")
    if torch.__version__.split("+", 1)[0] != TORCH_VERSION:
        raise PinnedQwenSmokeError("Torch differs from the prospectively frozen version")
    if versions["transformers"] != TRANSFORMERS_VERSION:
        raise PinnedQwenSmokeError("Transformers differs from the frozen version")
    if versions["tokenizers"] != TOKENIZERS_VERSION:
        raise PinnedQwenSmokeError("tokenizers differs from the frozen version")
    expected_packages = {
        "accelerate": ACCELERATE_VERSION,
        "huggingface_hub": HUGGINGFACE_HUB_VERSION,
        "peft": PEFT_VERSION,
        "safetensors": SAFETENSORS_VERSION,
    }
    if any(versions[name] != expected for name, expected in expected_packages.items()):
        raise PinnedQwenSmokeError("the auxiliary model stack differs from the freeze")
    if not torch.cuda.is_available():
        raise PinnedQwenSmokeError("the full-model smoke requires CUDA")
    return {
        "versions": versions,
        "platform": platform.platform(),
        "executable": sys.executable,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "allow_fp16_reduced_precision_reduction": (
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        ),
        "allow_bf16_reduced_precision_reduction": (
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        ),
        "allow_fp16_accumulation": torch.backends.cuda.matmul.allow_fp16_accumulation,
    }


@dataclass(frozen=True, slots=True)
class PinnedQwenSmokeCellConfig:
    """Only the operational coordinates not fixed by the v1 protocol."""

    cell_kind: CellKind
    artifact_root: Path
    episode_fixture_path: Path
    output_root: Path
    expected_artifact_manifest_digest: str
    expected_source_fingerprint: str
    device: str

    def __post_init__(self) -> None:
        if self.cell_kind not in {"sft", "rl"}:
            raise PinnedQwenSmokeError("cell_kind must be exactly sft or rl")
        for name in (
            "expected_artifact_manifest_digest",
            "expected_source_fingerprint",
        ):
            _require_sha256(getattr(self, name), name=name)
        for name in ("artifact_root", "episode_fixture_path", "output_root"):
            if not isinstance(getattr(self, name), Path):
                raise TypeError(f"{name} must be a pathlib.Path")
        validate_pinned_qwen_smoke_path_separation(
            self.artifact_root,
            self.episode_fixture_path,
            self.output_root,
        )
        try:
            device = torch.device(self.device)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise PinnedQwenSmokeError("device is invalid") from exc
        if device.type != "cuda" or device.index is None:
            raise PinnedQwenSmokeError("the full-model smoke requires an explicit CUDA device")

    @property
    def run_seed(self) -> int:
        return SFT_RUN_SEED if self.cell_kind == "sft" else RL_RUN_SEED

    @property
    def run_id(self) -> str:
        return f"{self.cell_kind}-seed{self.run_seed}"


def _load_frozen_episode(
    fixture_path: Path,
) -> tuple[EpisodeBank, HiddenEpisode, ReferenceTrajectoryRecord]:
    payload = fixture_path.read_bytes()
    if _sha256_bytes(payload) != EPISODE_FIXTURE_FILE_SHA256:
        raise PinnedQwenSmokeError("episode fixture bytes differ from the frozen file")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PinnedQwenSmokeError("episode fixture is not UTF-8") from exc
    bank = parse_episode_bank(text.removesuffix("\n"))
    if bank.digest != EPISODE_BANK_DIGEST:
        raise PinnedQwenSmokeError("episode-bank digest differs from the frozen bank")
    matches = tuple(episode for episode in bank.episodes if episode.episode_id == SMOKE_EPISODE_ID)
    if len(matches) != 1:
        raise PinnedQwenSmokeError("the exact frozen episode is absent or duplicated")
    episode = matches[0]
    if episode.digest != SMOKE_EPISODE_DIGEST:
        raise PinnedQwenSmokeError("the frozen episode digest changed")
    trajectory_bank = generate_reference_trajectory_bank(bank)
    records = tuple(record for record in trajectory_bank.records if record.episode_id == SMOKE_EPISODE_ID)
    if len(records) != 1:
        raise PinnedQwenSmokeError("the exact reference trajectory is absent or duplicated")
    reference = records[0]
    if reference.transcript_digest != SMOKE_REFERENCE_TRANSCRIPT_DIGEST:
        raise PinnedQwenSmokeError("the exact reference transcript digest changed")
    return bank, episode, reference


def reconstruct_frozen_cache_trace(
    tokenizer: ExactDecodeTokenizerProtocol,
    compiler: FragmentActionTokenCompiler,
) -> tuple[tuple[int, ...], ...]:
    """Reconstruct the exact preregistered nonvacuous prefix trace."""

    dialogue = (
        DialogueMessage(
            "system",
            "Play Hidden-Law Zendo. Return canonical JSON only.",
            "contract",
        ),
        DialogueMessage("user", "Choose the next legal action.", "opening"),
    )
    terminal_dialogue = (
        dialogue[0],
        DialogueMessage(
            "user",
            "Inquiry is over. Return one canonical answer with 12 labels.",
            "terminal",
        ),
    )
    _, opening = render_generation_prefix(tokenizer, dialogue)
    _, terminal = render_generation_prefix(tokenizer, terminal_dialogue)
    ready = compiler.trace_action(ReadyAction()).action_token_ids
    if len(opening) != 31 or len(terminal) != 40 or ready[:2] != (4_913, 3_397):
        raise PinnedQwenSmokeError("the live tokenizer changed the frozen cache trace")
    trace = (
        opening,
        opening + ready[:1],
        opening + ready[:2],
        terminal,
    )
    if tuple(len(prefix) for prefix in trace) != (31, 32, 33, 40):
        raise PinnedQwenSmokeError("cache trace prefix lengths changed")
    if json_digest([list(prefix) for prefix in trace], domain=_CACHE_TRACE_DOMAIN) != (
        SMOKE_CACHE_TRACE_DIGEST
    ):
        raise PinnedQwenSmokeError("cache trace digest differs from the prospective freeze")
    return trace


def _load_runtime(config: PinnedQwenSmokeCellConfig) -> SealedQwenRuntime:
    return load_sealed_qwen_runtime(
        config.artifact_root,
        expected_model_identifier=G03_QWEN_MODEL_IDENTIFIER,
        expected_revision=G03_QWEN_REVISION,
        expected_tokenizer_binding_digest=TOKENIZER_BINDING_DIGEST,
        expected_compiler_manifest_digest=COMPILER_MANIFEST_DIGEST,
        expected_artifact_manifest_digest=config.expected_artifact_manifest_digest,
        dtype=torch.bfloat16,
        device=config.device,
        load_config=SealedQwenLoadConfig(
            transformers_version=TRANSFORMERS_VERSION,
            tokenizers_version=TOKENIZERS_VERSION,
            attention_implementation="eager",
            low_cpu_mem_usage=True,
            use_safetensors=True,
            maximum_action_tokens=MAXIMUM_ACTION_TOKENS,
        ),
    )


def _cache_gate(
    runtime: SealedQwenRuntime,
) -> tuple[IncrementalCacheComparison, AuthenticatedIncrementalCacheProvider]:
    incremental = AuthenticatedIncrementalCacheProvider(runtime.provider)
    report = compare_incremental_cache_trace(
        incremental,
        runtime.compiler.manifest.tokenizer_manifest,
        reconstruct_frozen_cache_trace(runtime.tokenizer, runtime.compiler),
        absolute_tolerance=CACHE_LOGIT_ABSOLUTE_TOLERANCE,
    )
    if (
        report.trace_digest != SMOKE_CACHE_TRACE_DIGEST
        or report.prefix_lengths != (31, 32, 33, 40)
        or report.reuse_count != 2
        or report.reset_count != 1
        or not report.within_tolerance
    ):
        raise PinnedQwenSmokeError("the nonvacuous cache gate did not pass exactly")
    incremental.reauthenticate_policy_state()
    return report, incremental


def _optimizer_spec() -> AdamWOptimizerSpec:
    return AdamWOptimizerSpec(
        learning_rate=3e-6,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0.0,
    )


def _run_step(
    config: PinnedQwenSmokeCellConfig,
    runtime: SealedQwenRuntime,
    incremental: AuthenticatedIncrementalCacheProvider,
    episode: HiddenEpisode,
    reference: ReferenceTrajectoryRecord,
) -> tuple[
    AuthenticatedStepEvidenceBundle,
    str,
    tuple[str, ...],
    tuple[str, ...],
    dict[str, int],
]:
    coordinate = AuthenticatedUpdateCoordinate(
        study_id=PINNED_QWEN_SMOKE_REPORT_ID,
        run_id=config.run_id,
        update_index=0,
        objective_kind=config.cell_kind,
    )
    if config.cell_kind == "sft":
        source = ReferenceTrajectorySFTSource(episode, reference)
        sft_plan = build_streaming_sft_plan(
            (source,),
            runtime.tokenizer,
            runtime.compiler,
            runtime.provider,
            maximum_sequence_tokens=MAXIMUM_SEQUENCE_TOKENS,
        )
        bundle = execute_authenticated_sft_step(
            runtime,
            sft_plan,
            (source,),
            _optimizer_spec(),
            coordinate,
        )
        return (
            bundle,
            sft_plan.plan.to_json(),
            (),
            (),
            {
                "sampling_cache_reuse_count": 0,
                "sampling_cache_reset_count": 0,
            },
        )

    reuses_before = incremental.cache_reuse_count
    resets_before = incremental.reset_count
    try:
        group = collect_eight_authenticated_rollouts(
            incremental,
            runtime.tokenizer,
            runtime.compiler,
            episode,
            runtime.provider.model_provenance,
            run_seed=RL_RUN_SEED,
            temperature=RL_TEMPERATURE,
            absolute_tolerance=REPLAY_STATISTIC_ABSOLUTE_TOLERANCE,
            maximum_sequence_tokens=MAXIMUM_SEQUENCE_TOKENS,
            maximum_turns=MAXIMUM_TURNS,
        )
        cache_metrics = {
            "sampling_cache_reuse_count": incremental.cache_reuse_count - reuses_before,
            "sampling_cache_reset_count": incremental.reset_count - resets_before,
        }
    finally:
        # The rollout cache is derived acceleration only.  Drop every KV-tensor
        # reference before plan derivation, differentiable replay, or mutation.
        incremental.reset_cache()
    if len(group.rollouts) != 8:
        raise PinnedQwenSmokeError("the RL cell did not produce exactly eight rollouts")
    if cache_metrics["sampling_cache_reuse_count"] < 1:
        raise PinnedQwenSmokeError("RL collection did not exercise incremental cache reuse")
    records = tuple(rollout.record for rollout in group.rollouts)
    if sum(len(record.turns) for record in records) < 1:
        raise PinnedQwenSmokeError("the RL cell contains no authenticated action evidence")
    rl_plan = derive_verified_streaming_objective_plan(
        (group,),
        entropy_coefficient=RL_ENTROPY_COEFFICIENT,
    )
    bundle = execute_authenticated_rl_step(
        runtime,
        rl_plan,
        records,
        (episode,),
        _optimizer_spec(),
        coordinate,
    )
    return (
        bundle,
        rl_plan.plan.to_json(),
        tuple(record.to_json() for record in records),
        tuple(record.digest for record in records),
        cache_metrics,
    )


def _write_exclusive(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _artifact_records(root: Path) -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or (path.exists() and not path.is_file() and not path.is_dir()):
            raise PinnedQwenSmokeError("smoke output contains a link or special file")
        if not path.is_file() or path.name in {"completion-attestation.json", "COMPLETE"}:
            continue
        payload = path.read_bytes()
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": len(payload),
                "sha256": _sha256_bytes(payload),
            }
        )
    return tuple(records)


def _snapshot_output_tree(
    root: Path,
) -> tuple[tuple[str, str, int, int, int, int, int, int], ...]:
    """Return a non-following metadata snapshot while rejecting unsafe entries."""

    try:
        root_metadata = root.lstat()
    except OSError as exc:
        raise PinnedQwenSmokeError("smoke completion root is missing or unreadable") from exc
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise PinnedQwenSmokeError("smoke completion root must be a real directory")

    entries: list[tuple[str, str, int, int, int, int, int, int]] = [
        (
            ".",
            "directory",
            root_metadata.st_dev,
            root_metadata.st_ino,
            root_metadata.st_mode,
            root_metadata.st_size,
            root_metadata.st_mtime_ns,
            root_metadata.st_nlink,
        )
    ]
    try:
        candidates = sorted(root.rglob("*"), key=lambda path: path.relative_to(root).as_posix())
    except OSError as exc:
        raise PinnedQwenSmokeError("smoke completion tree could not be enumerated") from exc
    for candidate in candidates:
        relative_path = candidate.relative_to(root).as_posix()
        if (
            not relative_path
            or relative_path.startswith("/")
            or "\\" in relative_path
            or any(part in {"", ".", ".."} for part in relative_path.split("/"))
        ):
            raise PinnedQwenSmokeError("smoke completion tree has a noncanonical path")
        try:
            metadata = candidate.lstat()
        except OSError as exc:
            raise PinnedQwenSmokeError("smoke completion entry could not be inspected") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise PinnedQwenSmokeError("smoke completion tree may not contain symlinks")
        if stat.S_ISDIR(metadata.st_mode):
            kind = "directory"
        elif stat.S_ISREG(metadata.st_mode):
            if metadata.st_nlink != 1:
                raise PinnedQwenSmokeError("smoke completion files may not be hard-linked")
            kind = "file"
        else:
            raise PinnedQwenSmokeError("smoke completion tree may not contain special files")
        entries.append(
            (
                relative_path,
                kind,
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_mode,
                metadata.st_size,
                metadata.st_mtime_ns,
                metadata.st_nlink,
            )
        )
    return tuple(entries)


def _inspect_regular_output_file(
    path: Path,
    *,
    capture_bytes: bool,
) -> tuple[int, str, bytes | None]:
    """Hash one non-linked file and reject observable replacement races."""

    try:
        before_path = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise PinnedQwenSmokeError(f"smoke completion file is missing: {path.name}") from exc
    if not stat.S_ISREG(before_path.st_mode):
        raise PinnedQwenSmokeError("smoke completion tree may contain only regular files")
    if before_path.st_nlink != 1:
        raise PinnedQwenSmokeError("smoke completion files may not be hard-linked")

    digest = hashlib.sha256()
    captured = bytearray() if capture_bytes else None
    descriptor = -1
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        before_fd = os.fstat(descriptor)
        if not stat.S_ISREG(before_fd.st_mode) or before_fd.st_nlink != 1:
            raise PinnedQwenSmokeError("smoke completion file changed type while being read")
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            while block := handle.read(1024 * 1024):
                digest.update(block)
                if captured is not None:
                    captured.extend(block)
            after_fd = os.fstat(handle.fileno())
        after_path = path.stat(follow_symlinks=False)
    except PinnedQwenSmokeError:
        raise
    except OSError as exc:
        raise PinnedQwenSmokeError(f"smoke completion file could not be read: {path.name}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    stable_fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_nlink")
    if any(
        getattr(before_path, field) != getattr(before_fd, field)
        or getattr(before_fd, field) != getattr(after_fd, field)
        or getattr(after_fd, field) != getattr(after_path, field)
        for field in stable_fields
    ):
        raise PinnedQwenSmokeError("smoke completion file changed while being read")
    return before_fd.st_size, digest.hexdigest(), bytes(captured) if captured is not None else None


def _load_canonical_json_payload(payload: bytes, *, name: str) -> dict[str, Any]:
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise PinnedQwenSmokeError(f"{name} must be canonical ASCII JSON") from exc
    if not text.endswith("\n"):
        raise PinnedQwenSmokeError(f"{name} must end in exactly one newline")
    try:
        value = load_json(text[:-1])
        rendered = dump_json(value) + "\n"
    except CanonicalJSONError as exc:
        raise PinnedQwenSmokeError(f"{name} is not strict canonical JSON") from exc
    if rendered != text:
        raise PinnedQwenSmokeError(f"{name} is not in canonical compact form")
    if type(value) is not dict:
        raise PinnedQwenSmokeError(f"{name} must contain one JSON object")
    return cast(dict[str, Any], value)


def _expected_completion_artifact_paths(cell_kind: CellKind) -> tuple[str, ...]:
    paths = list(_BASE_COMPLETION_ARTIFACT_PATHS)
    if cell_kind == "rl":
        paths.extend(f"rollouts/{index:02d}.json" for index in range(8))
    return tuple(sorted(paths))


def _parse_completion_attestation(
    payload: bytes,
    *,
    expected_completion_digest: str,
) -> tuple[CellKind, str, str, tuple[PinnedQwenSmokeCompletionFile, ...]]:
    value = _load_canonical_json_payload(payload, name="completion-attestation.json")
    if tuple(value) != _COMPLETION_KEYS:
        raise PinnedQwenSmokeError("completion attestation has an unexpected structure")
    if type(value["schema_version"]) is not int or value["schema_version"] != (
        PINNED_QWEN_SMOKE_SCHEMA_VERSION
    ):
        raise PinnedQwenSmokeError("completion attestation schema version changed")
    if value["report_id"] != PINNED_QWEN_SMOKE_REPORT_ID:
        raise PinnedQwenSmokeError("completion attestation report identity changed")
    cell_kind_value = value["cell_kind"]
    if type(cell_kind_value) is not str or cell_kind_value not in {"sft", "rl"}:
        raise PinnedQwenSmokeError("completion attestation has an invalid cell kind")
    cell_kind = cast(CellKind, cell_kind_value)
    report_digest = _require_sha256(value["report_digest"], name="completion report digest")
    source_fingerprint = _require_sha256(
        value["source_fingerprint"],
        name="completion source fingerprint",
    )
    if value["status"] != "complete":
        raise PinnedQwenSmokeError("completion attestation is not complete")
    authorization = value["authorization"]
    if (
        type(authorization) is not dict
        or tuple(authorization) != ("weight_updates",)
        or authorization["weight_updates"] is not False
    ):
        raise PinnedQwenSmokeError("completion attestation authorization changed")

    raw_files = value["files"]
    if type(raw_files) is not list:
        raise PinnedQwenSmokeError("completion attestation files must be a JSON list")
    records: list[PinnedQwenSmokeCompletionFile] = []
    for raw_record in raw_files:
        if type(raw_record) is not dict or tuple(raw_record) != ("path", "size", "sha256"):
            raise PinnedQwenSmokeError("completion attestation has a malformed file record")
        records.append(
            PinnedQwenSmokeCompletionFile(
                path=raw_record["path"],
                size=raw_record["size"],
                sha256=raw_record["sha256"],
            )
        )
    files = tuple(records)
    paths = tuple(record.path for record in files)
    if paths != _expected_completion_artifact_paths(cell_kind):
        raise PinnedQwenSmokeError("completion attestation file inventory changed")

    digest = _require_sha256(value["digest"], name="completion attestation digest")
    body: dict[str, Any] = {
        "schema_version": PINNED_QWEN_SMOKE_SCHEMA_VERSION,
        "report_id": PINNED_QWEN_SMOKE_REPORT_ID,
        "cell_kind": cell_kind,
        "report_digest": report_digest,
        "source_fingerprint": source_fingerprint,
        "files": [record.as_obj() for record in files],
        "status": "complete",
        "authorization": {"weight_updates": False},
    }
    expected_json = _canonical_with_digest(body, domain=_COMPLETION_DOMAIN)
    computed_digest = cast(str, load_json(expected_json)["digest"])
    if digest != computed_digest or digest != expected_completion_digest:
        raise PinnedQwenSmokeError("completion attestation digest does not match its external anchor")
    if payload != (expected_json + "\n").encode("ascii"):
        raise PinnedQwenSmokeError("completion attestation is not the exact canonical v1 record")
    return cell_kind, report_digest, source_fingerprint, files


def _require_nonnegative_int(value: object, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise PinnedQwenSmokeError(f"{name} must be a non-negative integer")
    return value


def _verify_report_payload(
    payload: bytes,
    *,
    cell_kind: CellKind,
    report_digest: str,
    source_fingerprint: str,
) -> None:
    report = _load_canonical_json_payload(payload, name="report.json")
    if tuple(report) != _REPORT_KEYS:
        raise PinnedQwenSmokeError("smoke report has an unexpected structure")
    if type(report["schema_version"]) is not int or report["schema_version"] != (
        PINNED_QWEN_SMOKE_SCHEMA_VERSION
    ):
        raise PinnedQwenSmokeError("smoke report schema version changed")
    if (
        report["report_id"] != PINNED_QWEN_SMOKE_REPORT_ID
        or report["cell_kind"] != cell_kind
        or report["run_id"] != f"{cell_kind}-seed{SFT_RUN_SEED if cell_kind == 'sft' else RL_RUN_SEED}"
        or type(report["run_seed"]) is not int
        or report["run_seed"] != (SFT_RUN_SEED if cell_kind == "sft" else RL_RUN_SEED)
        or report["source_fingerprint"] != source_fingerprint
        or report["episode_fixture_file_sha256"] != EPISODE_FIXTURE_FILE_SHA256
        or report["episode_bank_digest"] != EPISODE_BANK_DIGEST
        or report["episode_id"] != SMOKE_EPISODE_ID
        or report["episode_digest"] != SMOKE_EPISODE_DIGEST
        or report["reference_transcript_digest"] != SMOKE_REFERENCE_TRANSCRIPT_DIGEST
        or report["tokenizer_binding_digest"] != TOKENIZER_BINDING_DIGEST
        or report["compiler_manifest_digest"] != COMPILER_MANIFEST_DIGEST
        or report["status"] != "complete"
    ):
        raise PinnedQwenSmokeError("smoke report differs from the frozen v1 identity")
    for name in (
        "source_fingerprint",
        "artifact_manifest_digest",
        "episode_fixture_file_sha256",
        "episode_bank_digest",
        "episode_digest",
        "reference_transcript_digest",
        "tokenizer_binding_digest",
        "compiler_manifest_digest",
        "cache_comparison_digest",
        "plan_digest",
        "step_evidence_bundle_digest",
        "step_record_digest",
        "pre_policy_state_digest",
        "post_policy_state_digest",
    ):
        _require_sha256(report[name], name=f"report {name}")
    metrics = report["collection_cache_metrics"]
    if type(metrics) is not dict or tuple(metrics) != (
        "sampling_cache_reuse_count",
        "sampling_cache_reset_count",
    ):
        raise PinnedQwenSmokeError("smoke report cache metrics have an unexpected structure")
    reuse_count = _require_nonnegative_int(
        metrics["sampling_cache_reuse_count"],
        name="sampling cache reuse count",
    )
    reset_count = _require_nonnegative_int(
        metrics["sampling_cache_reset_count"],
        name="sampling cache reset count",
    )
    rollout_digests = report["rollout_digests"]
    if type(rollout_digests) is not list:
        raise PinnedQwenSmokeError("smoke report rollout digests must be a JSON list")
    for index, digest in enumerate(rollout_digests):
        _require_sha256(digest, name=f"report rollout digest {index}")
    expected_rollout_count = 0 if cell_kind == "sft" else 8
    if (
        type(report["rollout_count"]) is not int
        or report["rollout_count"] != expected_rollout_count
        or len(rollout_digests) != expected_rollout_count
        or (cell_kind == "sft" and (reuse_count != 0 or reset_count != 0))
        or (cell_kind == "rl" and reuse_count < 1)
    ):
        raise PinnedQwenSmokeError("smoke report rollout/cache cardinality changed")
    if (
        _require_nonnegative_int(
            report["changed_parameter_count"],
            name="changed parameter count",
        )
        < 1
    ):
        raise PinnedQwenSmokeError("smoke report has no changed parameter")
    if report["optimizer_step_call_count"] != 1 or type(report["optimizer_step_call_count"]) is not int:
        raise PinnedQwenSmokeError("smoke report must attest exactly one optimizer step")
    _require_nonnegative_int(
        report["cuda_peak_allocated_bytes"],
        name="CUDA peak allocated bytes",
    )
    _require_nonnegative_int(
        report["cuda_peak_reserved_bytes"],
        name="CUDA peak reserved bytes",
    )
    for name in ("cache_maximum_absolute_error_hex", "elapsed_seconds_hex"):
        value = report[name]
        if type(value) is not str:
            raise PinnedQwenSmokeError(f"report {name} must be a hexadecimal float")
        try:
            numeric = float.fromhex(value)
        except ValueError as exc:
            raise PinnedQwenSmokeError(f"report {name} must be a hexadecimal float") from exc
        if not math.isfinite(numeric) or numeric < 0.0:
            raise PinnedQwenSmokeError(f"report {name} must be finite and non-negative")
    authorization = report["authorization"]
    if (
        type(authorization) is not dict
        or tuple(authorization) != ("capability_training", "production_weights", "scientific_launch")
        or any(value is not False for value in authorization.values())
    ):
        raise PinnedQwenSmokeError("smoke report authorization changed")
    digest = _require_sha256(report["digest"], name="smoke report digest")
    body = {key: report[key] for key in _REPORT_KEYS[:-1]}
    if digest != json_digest(body, domain=_REPORT_DOMAIN) or digest != report_digest:
        raise PinnedQwenSmokeError("smoke report digest differs from the completion seal")


def verify_pinned_qwen_smoke_completion(
    output_root: Path,
    *,
    expected_completion_digest: str,
) -> VerifiedPinnedQwenSmokeCompletion:
    """Verify an exact completed v1 tree against an independently held digest."""

    expected_digest = _require_sha256(
        expected_completion_digest,
        name="expected completion digest",
    )
    if not isinstance(output_root, Path):
        raise TypeError("output_root must be a pathlib.Path")
    before_tree = _snapshot_output_tree(output_root)

    _, _, complete_payload = _inspect_regular_output_file(
        output_root / "COMPLETE",
        capture_bytes=True,
    )
    if complete_payload != (expected_digest + "\n").encode("ascii"):
        raise PinnedQwenSmokeError("COMPLETE does not match the external completion digest")
    _, _, attestation_payload = _inspect_regular_output_file(
        output_root / "completion-attestation.json",
        capture_bytes=True,
    )
    if attestation_payload is None:  # pragma: no cover - capture_bytes is fixed above
        raise PinnedQwenSmokeError("completion attestation could not be captured")
    cell_kind, report_digest, source_fingerprint, files = _parse_completion_attestation(
        attestation_payload,
        expected_completion_digest=expected_digest,
    )

    expected_file_paths = tuple(
        sorted((*_expected_completion_artifact_paths(cell_kind), *_COMPLETION_CONTROL_PATHS))
    )
    expected_directory_paths = () if cell_kind == "sft" else ("rollouts",)
    observed_file_paths = tuple(entry[0] for entry in before_tree if entry[0] != "." and entry[1] == "file")
    observed_directory_paths = tuple(
        entry[0] for entry in before_tree if entry[0] != "." and entry[1] == "directory"
    )
    if observed_file_paths != expected_file_paths or observed_directory_paths != (expected_directory_paths):
        raise PinnedQwenSmokeError("smoke completion tree has missing or extra entries")

    report_payload: bytes | None = None
    for record in files:
        observed_size, observed_sha256, captured = _inspect_regular_output_file(
            output_root / record.path,
            capture_bytes=record.path == "report.json",
        )
        if observed_size != record.size or observed_sha256 != record.sha256:
            raise PinnedQwenSmokeError(f"completion-attested file changed: {record.path}")
        if record.path == "report.json":
            report_payload = captured
    if report_payload is None:
        raise PinnedQwenSmokeError("completion-attested report.json could not be captured")
    _verify_report_payload(
        report_payload,
        cell_kind=cell_kind,
        report_digest=report_digest,
        source_fingerprint=source_fingerprint,
    )

    after_tree = _snapshot_output_tree(output_root)
    if after_tree != before_tree:
        raise PinnedQwenSmokeError("smoke completion tree changed during verification")
    return VerifiedPinnedQwenSmokeCompletion(
        output_root=output_root,
        cell_kind=cell_kind,
        completion_digest=expected_digest,
        report_digest=report_digest,
        source_fingerprint=source_fingerprint,
        files=files,
    )


def _write_success_artifacts(
    config: PinnedQwenSmokeCellConfig,
    *,
    source: InteractiveSourceProvenance,
    environment: dict[str, object],
    bank: EpisodeBank,
    episode: HiddenEpisode,
    reference: ReferenceTrajectoryRecord,
    runtime_pre_json: str,
    runtime_post_json: str,
    cache: IncrementalCacheComparison,
    plan_json: str,
    rollout_json: tuple[str, ...],
    rollout_digests: tuple[str, ...],
    collection_cache_metrics: dict[str, int],
    bundle: AuthenticatedStepEvidenceBundle,
    elapsed_seconds: float,
) -> str:
    root = config.output_root
    _write_exclusive(root / "source-provenance.json", dump_json(source.as_obj()) + "\n")
    _write_exclusive(root / "environment.json", dump_json(environment) + "\n")
    _write_exclusive(root / "runtime-pre.json", runtime_pre_json + "\n")
    _write_exclusive(root / "runtime-post.json", runtime_post_json + "\n")
    _write_exclusive(root / "cache-comparison.json", cache.to_json() + "\n")
    _write_exclusive(root / "episode.json", dump_json(episode.as_obj()) + "\n")
    _write_exclusive(root / "reference-trajectory.json", dump_json(reference.as_obj()) + "\n")
    _write_exclusive(root / "objective-plan.json", plan_json + "\n")
    for index, payload in enumerate(rollout_json):
        _write_exclusive(root / "rollouts" / f"{index:02d}.json", payload + "\n")
    _write_exclusive(root / "step-evidence.json", bundle.to_json() + "\n")

    report: dict[str, Any] = {
        "schema_version": PINNED_QWEN_SMOKE_SCHEMA_VERSION,
        "report_id": PINNED_QWEN_SMOKE_REPORT_ID,
        "cell_kind": config.cell_kind,
        "run_id": config.run_id,
        "run_seed": config.run_seed,
        "source_fingerprint": source.fingerprint,
        "artifact_manifest_digest": config.expected_artifact_manifest_digest,
        "episode_fixture_file_sha256": EPISODE_FIXTURE_FILE_SHA256,
        "episode_bank_digest": bank.digest,
        "episode_id": episode.episode_id,
        "episode_digest": episode.digest,
        "reference_transcript_digest": reference.transcript_digest,
        "tokenizer_binding_digest": TOKENIZER_BINDING_DIGEST,
        "compiler_manifest_digest": COMPILER_MANIFEST_DIGEST,
        "cache_comparison_digest": cache.digest,
        "cache_maximum_absolute_error_hex": cache.maximum_absolute_error_hex,
        "collection_cache_metrics": collection_cache_metrics,
        "rollout_digests": list(rollout_digests),
        "rollout_count": len(rollout_digests),
        "plan_digest": bundle.record.plan_digest,
        "step_evidence_bundle_digest": bundle.digest,
        "step_record_digest": bundle.record.digest,
        "pre_policy_state_digest": bundle.record.pre_policy_state_digest,
        "post_policy_state_digest": bundle.record.post_policy_state_digest,
        "changed_parameter_count": bundle.record.changed_parameter_count,
        "optimizer_step_call_count": bundle.record.optimizer_step_call_count,
        "cuda_peak_allocated_bytes": bundle.record.cuda_peak_allocated_bytes,
        "cuda_peak_reserved_bytes": bundle.record.cuda_peak_reserved_bytes,
        "elapsed_seconds_hex": elapsed_seconds.hex(),
        "status": "complete",
        "authorization": {
            "capability_training": False,
            "production_weights": False,
            "scientific_launch": False,
        },
    }
    report_json = _canonical_with_digest(report, domain=_REPORT_DOMAIN)
    report_digest = cast(str, json.loads(report_json)["digest"])
    _write_exclusive(root / "report.json", report_json + "\n")

    completion: dict[str, Any] = {
        "schema_version": PINNED_QWEN_SMOKE_SCHEMA_VERSION,
        "report_id": PINNED_QWEN_SMOKE_REPORT_ID,
        "cell_kind": config.cell_kind,
        "report_digest": report_digest,
        "source_fingerprint": source.fingerprint,
        "files": list(_artifact_records(root)),
        "status": "complete",
        "authorization": {"weight_updates": False},
    }
    completion_json = _canonical_with_digest(completion, domain=_COMPLETION_DOMAIN)
    completion_digest = cast(str, json.loads(completion_json)["digest"])
    _write_exclusive(root / "completion-attestation.json", completion_json + "\n")
    _write_exclusive(root / "COMPLETE", completion_digest + "\n")
    return completion_digest


def run_pinned_qwen_smoke_cell(config: PinnedQwenSmokeCellConfig) -> str:
    """Run one fresh preregistered cell and return its completion digest."""

    if type(config) is not PinnedQwenSmokeCellConfig:
        raise TypeError("config must have exact type PinnedQwenSmokeCellConfig")
    validate_pinned_qwen_smoke_path_separation(
        config.artifact_root,
        config.episode_fixture_path,
        config.output_root,
    )
    if config.output_root.exists() or config.output_root.is_symlink():
        raise PinnedQwenSmokeError("the smoke output root must be fresh and absent")
    config.output_root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    configure_frozen_cuda_runtime()
    environment = _require_frozen_stack()
    source = interactive_source_provenance()
    if source.fingerprint != config.expected_source_fingerprint:
        raise PinnedQwenSmokeError("interactive source differs from the external freeze")
    bank, episode, reference = _load_frozen_episode(config.episode_fixture_path)
    runtime = _load_runtime(config)
    pre = runtime.reauthenticate()
    cache, incremental = _cache_gate(runtime)
    runtime.reauthenticate()
    bundle, plan_json, rollout_json, rollout_digests, collection_cache_metrics = _run_step(
        config,
        runtime,
        incremental,
        episode,
        reference,
    )
    post = runtime.reauthenticate()
    if bundle.record.post_runtime_manifest_digest != post.digest:
        raise PinnedQwenSmokeError("step evidence differs from the final sealed runtime")
    verify_interactive_source_provenance(source)
    if _file_sha256(config.episode_fixture_path) != EPISODE_FIXTURE_FILE_SHA256:
        raise PinnedQwenSmokeError("episode fixture changed during the smoke")
    elapsed = time.monotonic() - started
    return _write_success_artifacts(
        config,
        source=source,
        environment=environment,
        bank=bank,
        episode=episode,
        reference=reference,
        runtime_pre_json=pre.to_json(),
        runtime_post_json=post.to_json(),
        cache=cache,
        plan_json=plan_json,
        rollout_json=rollout_json,
        rollout_digests=rollout_digests,
        collection_cache_metrics=collection_cache_metrics,
        bundle=bundle,
        elapsed_seconds=elapsed,
    )


def write_failure_marker(output_root: Path, cell_kind: CellKind, exc: BaseException) -> None:
    """Best-effort diagnostic marker; never a nominal completion artifact."""

    if not isinstance(output_root, Path) or not output_root.is_dir():
        return
    value = {
        "schema_version": PINNED_QWEN_SMOKE_SCHEMA_VERSION,
        "report_id": PINNED_QWEN_SMOKE_REPORT_ID,
        "cell_kind": cell_kind,
        "status": "failed",
        "exception_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
        "exception_message": str(exc),
        "authorization": {"weight_updates": False},
    }
    try:
        _write_exclusive(output_root / "FAILURE.json", dump_json(value) + "\n")
    except (OSError, PinnedQwenSmokeError):
        return
