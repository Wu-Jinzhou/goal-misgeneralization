from __future__ import annotations

import copy
import hashlib
import json
import math
import uuid
from pathlib import Path
from typing import Any, cast

import pytest

from goalzendo_g00f_h200.qualification import (
    ACTION_LABELS,
    COMPARISON_SCHEMA,
    COMPARISON_SCHEMA_VERSION,
    ENGINEERING_ROOT_PREFIX,
    EVIDENCE_BOUNDARY,
    EVIDENCE_SCHEMA,
    EVIDENCE_SCHEMA_VERSION,
    LAW_FAMILIES,
    PANELS,
    PROCESS_SCHEMA,
    PROCESS_SCHEMA_VERSION,
    PROFILE_CONTRACT,
    PROFILES,
    QUALIFICATION_ENGINEERING_SEED,
    QUALIFICATION_SCHEMA,
    QUALIFICATION_SCHEMA_VERSION,
    REGISTERED_PARITY_UPDATE_CHECKPOINTS,
    TRAINABLE_NUMEL,
    TRAINING_VIEWS,
    TUNED_CAPACITY_DISQUALIFIERS,
    TUNED_CAPACITY_PROBE_SCHEMA,
    TUNED_CAPACITY_PROBE_SCHEMA_VERSION,
    UPDATES,
    VECTOR_KINDS,
    QualificationError,
    build_paired_execution_receipt,
    canonical_process_comparison_binding,
    create_qualification_report,
    replay_authenticated_evidence,
    seal_comparison_record,
    seal_evidence,
    seal_process_record,
    seal_tuned_capacity_probe,
    semantic_digest,
    streaming_metric_from_accumulator_rows,
    streaming_vector_metrics,
    validate_process_record,
    validate_qualification_evidence,
    validate_qualification_report,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


DEVICE_UUIDS = tuple(f"GPU-00000000-0000-0000-0000-{index:012d}" for index in range(1, 5))
MODULES = [
    {
        "name": "config.attention_dropout",
        "class_name": "ModelConfigProbability",
        "kind": "dropout",
        "active": False,
    },
    {
        "name": "model.dropout",
        "class_name": "Dropout",
        "kind": "dropout",
        "active": False,
    },
]
BASELINE_VECTOR = {"decoder.bias": [0.25], "decoder.weight": [1.0, -2.0, 3.0]}
TUNED_VECTOR = {"decoder.bias": [0.25], "decoder.weight": [1.0, -2.0, 3.0]}


def _capacity_probe(
    *,
    failed_device: str | None = None,
    passed_peak_reserved_bytes: int = 110 * 1024**3,
) -> dict[str, Any]:
    receipts = []
    for host_ordinal, device_uuid in enumerate(DEVICE_UUIDS):
        failed = device_uuid == failed_device
        cells = []
        for panel in PANELS:
            for law in LAW_FAMILIES:
                cell_failed = failed and panel == "g00f-1p5b" and law == "parity"
                argv = ["python", "controller.py", "--mode", "capacity-probe-cell", panel, law]
                cell = {
                    "panel_id": panel,
                    "law_family": law,
                    "device_uuid": device_uuid,
                    "actual_model_execution": True,
                    "isolated_subprocess": True,
                    "subprocess_argv": argv,
                    "subprocess_argv_sha256": semantic_digest(argv),
                    "subprocess_exit_code": 0,
                    "process_exit_resets_device": True,
                    "model_receipt_digest": _digest(f"capacity-model:{panel}"),
                    "resolved_config_sha256": _digest(f"capacity-config:{panel}"),
                    "workload_sha256": _digest(f"capacity-workload:{panel}:{law}"),
                    "numerical_execution_sha256": semantic_digest(
                        {
                            "deterministic_algorithms": True,
                            "deterministic_warn_only": False,
                            "cudnn_benchmark": False,
                            "cudnn_deterministic": True,
                            "cuda_matmul_allow_tf32": False,
                            "cudnn_allow_tf32": False,
                            "float32_matmul_precision": "highest",
                            "cublas_workspace_config": ":4096:8",
                        }
                    ),
                    "fresh_model_and_optimizer": True,
                    "train_batch_exercised": True,
                    "gradient_clip_exercised": True,
                    "adamw_step_exercised": True,
                    "optimizer_moments_resident_during_evaluation": True,
                    "contiguous_evaluation_example_count": 128,
                    "flattened_evaluation_prompt_count": 128 * len(TRAINING_VIEWS),
                    "cuda_synchronized_before_memory_read": True,
                    "finite_execution": True,
                    "peak_allocated_bytes": 100 * 1024**3,
                    "peak_reserved_bytes": (121 * 1024**3 if cell_failed else passed_peak_reserved_bytes),
                    "status": "recoverable_capacity_failure" if cell_failed else "passed",
                    "disqualifier": ("reserved_memory_ceiling_exceeded" if cell_failed else None),
                    "failure_type_sha256": _digest("capacity-type") if cell_failed else None,
                    "failure_message_sha256": _digest("capacity-message") if cell_failed else None,
                    "log_binding": {
                        "file_sha256": _digest(f"capacity-log:{device_uuid}:{panel}:{law}"),
                        "bytes": 1024,
                        "mode": 0o400,
                        "retention": "compact_digest_only_raw_log_deleted_after_hash",
                    },
                }
                cells.append(_reseal(cell, "cell_digest"))
        body = {
            "device_uuid": device_uuid,
            "device_name": "NVIDIA H200 NVL",
            "host_ordinal": host_ordinal,
            "probe_cells": cells,
            "peak_allocated_bytes": 100 * 1024**3,
            "peak_reserved_bytes": (121 * 1024**3 if failed else passed_peak_reserved_bytes),
            "status": "recoverable_capacity_failure" if failed else "passed",
            "disqualifiers": ["reserved_memory_ceiling_exceeded"] if failed else [],
        }
        receipts.append(_reseal(body, "receipt_digest"))
    branch = "tuned_capacity_fallback_baseline" if failed_device else "tuned_probe_passed_full"
    return seal_tuned_capacity_probe(
        {
            "schema": TUNED_CAPACITY_PROBE_SCHEMA,
            "schema_version": TUNED_CAPACITY_PROBE_SCHEMA_VERSION,
            "qualification_branch": branch,
            "status": "recoverable_capacity_failure" if failed_device else "passed",
            "actual_model_execution": True,
            "profile_contract": copy.deepcopy(dict(PROFILE_CONTRACT["tuned"])),
            "workload_contract": {
                "panels": list(PANELS),
                "train_batch_size": 50,
                "gradient_accumulation_steps": 1,
                "evaluation_batch_size_examples": 128,
                "registered_evaluation_example_count": 128,
                "production_evaluation_partition": "example_chunks_then_flatten_prompt_views",
                "all_six_views": True,
                "both_laws": True,
                "corpus_max_train_sample_injected": True,
                "exact_worst_shape_batch_exercised": True,
            },
            "numerical_execution": {
                "configure_numerical_execution_called": True,
                "receipt": {
                    "deterministic_algorithms": True,
                    "deterministic_warn_only": False,
                    "cudnn_benchmark": False,
                    "cudnn_deterministic": True,
                    "cuda_matmul_allow_tf32": False,
                    "cudnn_allow_tf32": False,
                    "float32_matmul_precision": "highest",
                    "cublas_workspace_config": ":4096:8",
                },
            },
            "gpu_uuids": list(DEVICE_UUIDS),
            "allowed_disqualifiers": list(TUNED_CAPACITY_DISQUALIFIERS),
            "device_receipts": receipts,
            "failed_device_uuids": [failed_device] if failed_device else [],
            "outcomes_seen": False,
            "itt_ledger_created": False,
            "g01_launch_authorized": False,
        }
    )


def _trainable_manifest(panel: str) -> dict[str, Any]:
    entries = [
        {
            "parameter_key": "decoder.bias",
            "dtype": "torch.bfloat16",
            "shape": [1],
            "numel": 1,
        },
        {
            "parameter_key": "decoder.weight",
            "dtype": "torch.bfloat16",
            "shape": [TRAINABLE_NUMEL[panel] - 1],
            "numel": TRAINABLE_NUMEL[panel] - 1,
        },
    ]
    parameter_binding = [
        {
            "order_index": index,
            "parameter_key": entry["parameter_key"],
            "element_count": entry["numel"],
        }
        for index, entry in enumerate(entries)
    ]
    return {
        "enumeration_api": "named_parameters(remove_duplicate=True)",
        "requires_grad_only": True,
        "order": "sorted_parameter_keys",
        "entries": entries,
        "trainable_numel": TRAINABLE_NUMEL[panel],
        "parameter_keys_sha256": semantic_digest(parameter_binding),
        "manifest_sha256": semantic_digest(entries),
    }


def _native_tensor_digest(*, dtype: str, shape: list[int], values: list[float]) -> str:
    return semantic_digest(
        {
            "dtype": dtype,
            "shape": shape,
            "contiguous_bytes_sha256": _digest(json.dumps(values, separators=(",", ":"))),
        }
    )


def _metric_for_panel(
    left: dict[str, list[float]],
    right: dict[str, list[float]],
    panel: str,
) -> dict[str, Any]:
    metric = streaming_vector_metrics(left, right)
    manifest = _trainable_manifest(panel)
    entries_by_key = {entry["parameter_key"]: entry for entry in manifest["entries"]}
    for row in metric["accumulator_rows"]:
        entry = entries_by_key[row["parameter_key"]]
        row["element_count"] = entry["numel"]
        row["left_native_tensor_sha256"] = _native_tensor_digest(
            dtype=entry["dtype"],
            shape=entry["shape"],
            values=left[row["parameter_key"]],
        )
        row["right_native_tensor_sha256"] = _native_tensor_digest(
            dtype=entry["dtype"],
            shape=entry["shape"],
            values=right[row["parameter_key"]],
        )
    parameter_binding = [
        {
            "order_index": row["order_index"],
            "parameter_key": row["parameter_key"],
            "element_count": row["element_count"],
        }
        for row in metric["accumulator_rows"]
    ]
    metric["parameter_keys_sha256"] = semantic_digest(parameter_binding)
    metric["left_native_chunk_manifest_sha256"] = semantic_digest(
        [
            {**binding, "native_tensor_sha256": row["left_native_tensor_sha256"]}
            for binding, row in zip(parameter_binding, metric["accumulator_rows"], strict=True)
        ]
    )
    metric["right_native_chunk_manifest_sha256"] = semantic_digest(
        [
            {**binding, "native_tensor_sha256": row["right_native_tensor_sha256"]}
            for binding, row in zip(parameter_binding, metric["accumulator_rows"], strict=True)
        ]
    )
    metric["element_count"] = TRAINABLE_NUMEL[panel]
    metric["accumulator_rows_sha256"] = semantic_digest(metric["accumulator_rows"])
    return metric


def _descriptor(vector: dict[str, list[float]], panel: str) -> dict[str, Any]:
    metric = _metric_for_panel(vector, vector, panel)
    manifest = _trainable_manifest(panel)
    return {
        "parameter_keys_sha256": metric["parameter_keys_sha256"],
        "native_chunk_manifest_sha256": metric["left_native_chunk_manifest_sha256"],
        "element_count": metric["element_count"],
        "float64_norm": metric["left_norm"],
        "chunk_count": len(metric["accumulator_rows"]),
        "trainable_parameter_manifest_sha256": manifest["manifest_sha256"],
    }


def _output(view: str, *, probability_a: float) -> dict[str, Any]:
    worst = view == "surface_only"
    return {
        "view": view,
        "sample_id": "worst-token-sample" if worst else f"{view}-sample",
        "token_length": 640 if worst else 128,
        "worst_case_token_sample": worst,
        "normalized_log_scores": [math.log(probability_a), math.log(1.0 - probability_a)],
        "probabilities": [probability_a, 1.0 - probability_a],
        "action_index": 0 if probability_a >= 0.5 else 1,
        "action_label": ACTION_LABELS[0 if probability_a >= 0.5 else 1],
    }


def _registered_parity_chunk(
    profile: str,
    law: str,
    *,
    state: str,
    probability_a: float = 0.6,
) -> dict[str, Any]:
    batch_size = int(PROFILE_CONTRACT[profile]["evaluation_batch_size"])
    example_ids = [f"{law}:registered-example-{index:03d}" for index in range(128)]
    token_shapes = [[128 + view_index for view_index in range(len(TRAINING_VIEWS))] for _ in range(128)]
    rows = [
        {
            "row_id": f"{example_id}:{view}",
            "normalized_log_scores": [math.log(probability_a), math.log(1.0 - probability_a)],
            "probabilities": [probability_a, 1.0 - probability_a],
            "action_index": 0 if probability_a >= 0.5 else 1,
            "action_label": ACTION_LABELS[0 if probability_a >= 0.5 else 1],
        }
        for example_id in example_ids
        for view in TRAINING_VIEWS
    ]
    return {
        "measurement_state": state,
        "after_optimizer_step": state != "initial",
        "example_count": 128,
        "prompt_count": 128 * len(TRAINING_VIEWS),
        "production_example_partition_count": 1,
        "batch_size": batch_size,
        "batch_count": math.ceil(128 / batch_size),
        "scorer_call_count": math.ceil(128 / batch_size),
        "prompt_sha256": _digest(f"{law}:registered-production-shaped-parity-prompts"),
        "normalized_outputs_sha256": semantic_digest([row["normalized_log_scores"] for row in rows]),
        "rows": rows,
        "rows_sha256": semantic_digest(rows),
        "contiguous_example_chunk": True,
        "prompt_views_per_example": len(TRAINING_VIEWS),
        "production_bank_name": "validation",
        "production_bank_start_index": 0,
        "example_ids": example_ids,
        "example_ids_sha256": semantic_digest(example_ids),
        "token_shapes": token_shapes,
        "token_shape_sha256": semantic_digest(token_shapes),
    }


def _execution_benchmark(profile: str, law: str, *, full_timing: bool) -> dict[str, Any]:
    batch_size = int(PROFILE_CONTRACT[profile]["evaluation_batch_size"])
    diagnostic_examples = [1000, 512, 1024]
    final_examples = [1000, 512, 4096]
    diagnostic_prompt_count = 10_096
    final_prompt_count = 13_168
    boundaries: list[dict[str, Any]] = []
    for ordinal in range(13):
        kind = "final" if ordinal == 12 else "diagnostic"
        example_counts = final_examples if kind == "final" else diagnostic_examples
        boundaries.append(
            {
                "ordinal": ordinal,
                "kind": kind,
                "example_count": sum(example_counts),
                "prompt_count": final_prompt_count if kind == "final" else diagnostic_prompt_count,
                "batch_size": batch_size,
                "batch_count": sum(math.ceil(value / batch_size) for value in example_counts),
                "scorer_call_count": sum(math.ceil(value / batch_size) for value in example_counts),
                "prompt_sha256": _digest(f"{law}:{kind}:prompts"),
                "normalized_outputs_sha256": _digest(f"{law}:{kind}:outputs"),
                "callback_artifact_sha256": _digest(f"{law}:{kind}:{ordinal}:callback"),
                "callback_artifact_bytes": 4096,
                "callback_artifact_count": 3,
                "callback_metrics_row_count": (
                    final_prompt_count if kind == "final" else diagnostic_prompt_count
                ),
                "callback_prediction_row_count": (
                    final_prompt_count if kind == "final" else diagnostic_prompt_count
                ),
                "callback_progress_record_count": 1,
                "callback_schema": "production_metrics_predictions_progress_envelope_v1",
            }
        )

    def call_shapes(max_seq_len: int) -> list[dict[str, Any]]:
        shapes: list[dict[str, Any]] = []
        for boundary_ordinal in range(13):
            calls = 20 if boundary_ordinal == 12 else 15
            for call_index in range(calls):
                shapes.append(
                    {
                        "boundary_ordinal": boundary_ordinal,
                        "kind": "final" if boundary_ordinal == 12 else "diagnostic",
                        "bank_index": call_index % 3,
                        "chunk_index": call_index // 3,
                        "example_count": 128,
                        "flattened_prompt_count": 768,
                        "max_seq_len": max_seq_len,
                        "padded_elements": 768 * max_seq_len,
                    }
                )
        return shapes

    parity_shapes = call_shapes(640)
    majority_shapes = call_shapes(600)
    record_shapes = parity_shapes if law == "parity" else majority_shapes
    other_shapes = majority_shapes if law == "parity" else parity_shapes
    record_workload = sum(item["padded_elements"] for item in record_shapes)
    other_workload = sum(item["padded_elements"] for item in other_shapes)
    body = {
        "tier": "full_production_timing_representative" if full_timing else "registered_numeric_only",
        "timing_representative": full_timing,
        "worst_law_proof": {
            "selected_law": "parity",
            "record_law": law,
            "record_is_selected_worst": law == "parity",
            "law_max_token_length": 640 if law == "parity" else 600,
            "other_law_max_token_length": 600 if law == "parity" else 640,
            "record_padded_token_elements": record_workload,
            "other_law_padded_token_elements": other_workload,
            "record_maximum_padded_token_elements_per_call": max(
                item["padded_elements"] for item in record_shapes
            ),
            "other_law_maximum_padded_token_elements_per_call": max(
                item["padded_elements"] for item in other_shapes
            ),
            "record_maximum_flattened_prompts_per_call": 768,
            "other_law_maximum_flattened_prompts_per_call": 768,
            "record_scorer_call_count": len(record_shapes),
            "other_law_scorer_call_count": len(other_shapes),
            "record_call_shapes": record_shapes,
            "other_law_call_shapes": other_shapes,
            "record_call_shapes_sha256": semantic_digest(record_shapes),
            "other_law_call_shapes_sha256": semantic_digest(other_shapes),
        },
        "data_bank_render_tokenization": {
            "corpus_size": 10_000,
            "law_family": law,
            "six_training_views_rendered": True,
            "diagnostic_prompt_count": diagnostic_prompt_count,
            "final_prompt_count": final_prompt_count,
            "diagnostic_bank_example_counts": diagnostic_examples,
            "final_bank_example_counts": final_examples,
            "diagnostic_prompt_sha256": _digest(f"{law}:diagnostic:prompts"),
            "final_prompt_sha256": _digest(f"{law}:final:prompts"),
        },
        "mode_transitions": {
            "scorer_train_for_every_forward_backward": True,
            "eval_only_at_boundaries_and_registered_parity_chunk": True,
            "train_mode_restored_after_every_evaluation": True,
            "dropout_effectively_zero_while_training": True,
        },
        "gradient_checkpointing": {
            "configured": profile == "baseline",
            "runtime_enabled": profile == "baseline",
            "training_graph_exercised": profile == "baseline",
        },
        "registered_production_shaped_parity_chunk": _registered_parity_chunk(
            profile,
            law,
            state="initial",
        ),
        "boundaries": boundaries if full_timing else [],
        "checkpoint_io": {
            "executed": full_timing,
            "artifact_count": 2 if full_timing else 0,
            "bytes": 8192 if full_timing else 0,
            "sha256": _digest(f"{profile}:{law}:checkpoint"),
        },
        "final_seal_io": {
            "executed": full_timing,
            "artifact_count": 15 if full_timing else 0,
            "bytes": 2048 if full_timing else 0,
            "sha256": _digest(f"{profile}:{law}:seal"),
            "file_names": (
                [
                    "metrics.jsonl",
                    "predictions.jsonl",
                    "summary.json",
                    "status.json",
                    "completion.json",
                    "COMPLETE",
                ]
                if full_timing
                else []
            ),
            "metrics_row_count": (
                sum(int(item["prompt_count"]) for item in boundaries) if full_timing else 0
            ),
            "prediction_row_count": (
                sum(int(item["prompt_count"]) for item in boundaries) if full_timing else 0
            ),
            "summary_schema": (
                "production_run_summary_envelope_v1"
                if full_timing
                else "not_executed_registered_numeric_only"
            ),
            "attested_file_count": 11 if full_timing else 0,
            "completion_attestation_verified": full_timing,
            "complete_marker_final_write": full_timing,
            "run_store_finalize_used": full_timing,
            "outcome_file_seal_hashes": (
                {
                    name: _digest(f"{profile}:{law}:sealed:{name}")
                    for name in ("metrics.jsonl", "predictions.jsonl", "summary.json")
                }
                if full_timing
                else {}
            ),
            "outcome_file_seal_modes": (
                {name: 0 for name in ("metrics.jsonl", "predictions.jsonl", "summary.json")}
                if full_timing
                else {}
            ),
            "outcome_file_seal_file_count": 3 if full_timing else 0,
            "outcome_file_seal_scan_and_chmod_completed": full_timing,
            "seal_restore_only_for_authenticated_transient_cleanup": full_timing,
        },
    }
    return {**body, "benchmark_digest": semantic_digest(body)}


def _standardized_worst_shape_training(
    profile: str,
    panel: str,
    *,
    host_envelopes_seconds: float = 1.0,
) -> dict[str, Any]:
    maximum_length = 512
    maximum_prompt_length = maximum_length - 1
    run_receipts = []
    for index in range(80):
        law = LAW_FAMILIES[index % len(LAW_FAMILIES)]
        view = TRAINING_VIEWS[index % len(TRAINING_VIEWS)]
        run_receipts.append(
            {
                "global_index": index,
                "baseline_plan_key": f"{index:020x}",
                "seed": 10_000 + index,
                "derived_seeds_sha256": _digest(f"derived:{panel}:{index}"),
                "law_family": law,
                "training_view": view,
                "train_renderers": ["natural_1", "natural_2", "natural_3", "natural_4"],
                "training_renderer_counts": {
                    "natural_1": 2_500,
                    "natural_2": 2_500,
                    "natural_3": 2_500,
                    "natural_4": 2_500,
                },
                "prompt_count": 10_000,
                "training_prompt_sha256": _digest(f"prompts:{panel}:{index}"),
                "token_length_stream_sha256": _digest(f"lengths:{panel}:{index}"),
                "contextual_continuation_stream_sha256": _digest(f"contextual-continuations:{panel}:{index}"),
                "maximum_prompt_token_length": maximum_prompt_length,
                "maximum_token_length": maximum_length,
                "maximum_contextual_token_work": 302,
                "maximum_utf8_bytes_work": 602,
            }
        )
    host_rows = []
    for index in range(10):
        prompt_tokens = 100 - index
        prompt_bytes = 200 - index
        host_rows.append(
            {
                "baseline_plan_key": run_receipts[index]["baseline_plan_key"],
                "global_index": index,
                "prompt_index": index,
                "prompt_sha256": _digest(f"host-envelope:{panel}:{index}"),
                "prompt_utf8_bytes": prompt_bytes,
                "prompt_token_length": prompt_tokens,
                "prompt_a_utf8_bytes": prompt_bytes + 1,
                "prompt_a_token_length": prompt_tokens + 1,
                "prompt_b_utf8_bytes": prompt_bytes + 1,
                "prompt_b_token_length": prompt_tokens + 1,
                "continuation_a_token_count": 1,
                "continuation_a_token_ids_sha256": semantic_digest([32]),
                "continuation_b_token_count": 1,
                "continuation_b_token_ids_sha256": semantic_digest([33]),
                "total_contextual_token_work": prompt_tokens * 3 + 2,
                "maximum_contextual_token_length": prompt_tokens + 1,
                "total_utf8_bytes": prompt_bytes * 3 + 2,
                "maximum_utf8_bytes": prompt_bytes + 1,
            }
        )

    def envelope(metric: str, *, role: str) -> dict[str, Any]:
        source_rows = host_rows if role == "variability_stress" else host_rows[:1]
        ordered_digests = [str(row["prompt_sha256"]) for row in source_rows]
        repetition_count = 5 if role == "variability_stress" else 50
        execution_digests = ordered_digests * repetition_count
        baseline_partitions = [execution_digests[start : start + 10] for start in range(0, 50, 10)]
        return {
            "role": role,
            "selection_algorithm": (
                f"global_top10_distinct_by_{metric}_desc_prompt_sha256_tiebreak_v1"
                if role == "variability_stress"
                else f"global_maximum_by_{metric}_desc_prompt_sha256_tiebreak_v1"
            ),
            "metric_field": metric,
            "source_row_count": len(source_rows),
            "rows": copy.deepcopy(source_rows),
            "rows_sha256": semantic_digest(source_rows),
            "selected_metric_sum": sum(cast(int, row[metric]) for row in source_rows),
            "selected_metric_maximum": max(cast(int, row[metric]) for row in source_rows),
            "repetition_count": repetition_count,
            "execution_row_count_per_update": 50,
            "ordered_execution_prompt_sha256": semantic_digest(execution_digests),
            "ordered_execution_rows_sha256": semantic_digest(source_rows * repetition_count),
            "baseline_call_partition_sizes": [10] * 5,
            "baseline_call_partitions_sha256": semantic_digest(baseline_partitions),
            "tuned_call_partition_sizes": [50],
            "tuned_call_partitions_sha256": semantic_digest([execution_digests]),
            "variability_stress_only": role == "variability_stress",
            "hard_dominance_tier": role == "hard_dominance",
            "each_baseline_call_dominates_any_registered_10_row_call": role == "hard_dominance",
            "tuned_ordered50_is_identical_to_concatenated_baseline_calls": True,
            "tuned_total_dominates_any_registered_50_row_batch": role == "hard_dominance",
            "global_metric_maximum_over_all_registered_occurrences": role == "hard_dominance",
        }

    host_envelope_order = [
        "contextual_token_work_variability",
        "utf8_byte_work_variability",
        "contextual_token_work_dominance",
        "utf8_byte_work_dominance",
    ]
    host_envelope_proofs = {
        "contextual_token_work_variability": envelope(
            "total_contextual_token_work",
            role="variability_stress",
        ),
        "utf8_byte_work_variability": envelope("total_utf8_bytes", role="variability_stress"),
        "contextual_token_work_dominance": envelope(
            "total_contextual_token_work",
            role="hard_dominance",
        ),
        "utf8_byte_work_dominance": envelope("total_utf8_bytes", role="hard_dominance"),
    }
    proof_body = {
        "algorithm": "exhaustive_registered_80_run_training_prompt_maximum_v1",
        "panel_id": panel,
        "profile_plan_bindings": {
            bound_profile: {
                "path": (
                    f"docs/goalzendo/plans/g00f-h200-{bound_profile}-{panel.removeprefix('g00f-')}.jsonl"
                ),
                "file_sha256": _digest(f"plan-file:{bound_profile}:{panel}"),
                "run_count": 80,
                "plan_key_sha256": _digest(f"plan-keys:{bound_profile}:{panel}"),
            }
            for bound_profile in PROFILES
        },
        "prompt_generation_inputs_identical_across_profiles": True,
        "registered_run_count": 80,
        "registered_training_prompt_count": 800_000,
        "law_families": list(LAW_FAMILIES),
        "training_views": list(TRAINING_VIEWS),
        "train_renderers": ["natural_1", "natural_2", "natural_3", "natural_4"],
        "run_receipts": run_receipts,
        "run_receipts_sha256": semantic_digest(run_receipts),
        "maximum_token_length": maximum_length,
        "maximum_prompt_token_length": maximum_prompt_length,
        "maximum_prompt_sha256": _digest(f"maximum-prompt:{panel}"),
        "maximum_prompt_identity": {
            "baseline_plan_key": run_receipts[0]["baseline_plan_key"],
            "global_index": 0,
            "prompt_index": 9_999,
            "seed": run_receipts[0]["seed"],
            "law_family": run_receipts[0]["law_family"],
            "training_view": run_receipts[0]["training_view"],
        },
        "tokenizer_host_envelopes": {
            "contextual_continuation_mode": (
                "goalzendo.modeling.encode_action_continuations:add_prompt_special_tokens=false"
            ),
            "action_labels": ["A", "B"],
            "standalone_action_token_ids": [[32], [33]],
            "standalone_action_token_ids_sha256": semantic_digest([[32], [33]]),
            "envelope_order": host_envelope_order,
            "envelopes": host_envelope_proofs,
            "measured_seconds_aggregation": "sum_all_four_envelopes",
        },
        "contextual_scorer_branch_invariant": {
            "contextual_continuation_mode": (
                "goalzendo.modeling.encode_action_continuations:add_prompt_special_tokens=false"
            ),
            "action_labels": ["A", "B"],
            "required_token_count_per_action_continuation": 1,
            "registered_prompt_count": 800_000,
            "per_run_stream_digest_field": "contextual_continuation_stream_sha256",
            "every_registered_a_and_b_continuation_exactly_one_token": True,
            "maximum_prompt_token_length": maximum_prompt_length,
            "maximum_scorer_branch_token_length": maximum_length,
            "maximum_scorer_branch_is_prompt_plus_one_token": True,
            "global_maximum_ranked_over_prompt_and_both_action_branches": True,
        },
        "global_maximum_dominates_every_registered_training_prompt": True,
        "stream_encoding": "uint64be_length_prefixed_canonical_json_rows_v1",
        "controller_scan_seconds": 12.0,
    }
    proof = {**proof_body, "proof_digest": semantic_digest(proof_body)}
    accumulation = int(PROFILE_CONTRACT[profile]["gradient_accumulation_steps"])
    batch_size = int(PROFILE_CONTRACT[profile]["train_batch_size"])
    call_shapes = [
        {
            "update": update,
            "micro_step": (update - 1) * accumulation + accumulation_index,
            "example_count": batch_size,
            "max_seq_len": maximum_prompt_length,
            "padded_elements": batch_size * maximum_prompt_length,
        }
        for update in UPDATES
        for accumulation_index in range(accumulation)
    ]
    partition_sizes = [batch_size] * accumulation
    host_contracts = {}
    for envelope_name, envelope_proof in host_envelope_proofs.items():
        source_rows = envelope_proof["rows"]
        execution_digests = [str(row["prompt_sha256"]) for row in source_rows] * int(
            envelope_proof["repetition_count"]
        )
        partitions = [
            execution_digests[start : start + batch_size]
            for start in range(0, len(execution_digests), batch_size)
        ]
        host_contracts[envelope_name] = {
            "metric_field": envelope_proof["metric_field"],
            "role": envelope_proof["role"],
            "source_row_count": envelope_proof["source_row_count"],
            "rows_sha256": envelope_proof["rows_sha256"],
            "selected_metric_sum": envelope_proof["selected_metric_sum"],
            "selected_metric_maximum": envelope_proof["selected_metric_maximum"],
            "repetition_count": envelope_proof["repetition_count"],
            "ordered_execution_prompt_sha256": envelope_proof["ordered_execution_prompt_sha256"],
            "timed_updates": len(UPDATES),
            "profile_batch_size": batch_size,
            "profile_call_partition_sizes": partition_sizes,
            "profile_call_partitions_sha256": semantic_digest(partitions),
            "profile_scorer_call_count": len(UPDATES) * accumulation,
            "total_examples_tokenized": len(UPDATES) * 50,
            "warmup_horizon_excluded": True,
            "worker_contextual_rows_replayed": True,
        }
    return {
        "global_maximum_proof": proof,
        "timed_corpus_size": 10_000,
        "all_timed_prompts_equal_global_maximum": True,
        "timed_call_shapes": call_shapes,
        "timed_call_shapes_sha256": semantic_digest(call_shapes),
        "timed_padded_token_elements": sum(row["padded_elements"] for row in call_shapes),
        "production_updates_per_run": 1_000,
        "runs_per_panel_per_worker": 20,
        "runs_per_worker": 40,
        "conservative_scaling_ratio": 1.0,
        "standardized_workload_dominates_every_registered_training_call": True,
        "tokenizer_host_envelope_contracts": {
            "contextual_continuation_mode": (
                "goalzendo.modeling.encode_action_continuations:add_prompt_special_tokens=false"
            ),
            "envelope_order": host_envelope_order,
            "measured_seconds_aggregation": "sum_all_four_envelopes",
            "measured_seconds_by_envelope": {
                "contextual_token_work_variability": host_envelopes_seconds * 0.2,
                "utf8_byte_work_variability": host_envelopes_seconds * 0.3,
                "contextual_token_work_dominance": host_envelopes_seconds * 0.2,
                "utf8_byte_work_dominance": host_envelopes_seconds * 0.3,
            },
            "measured_seconds_sum": host_envelopes_seconds,
            "envelopes": host_contracts,
        },
    }


def _process_body(
    *,
    counter: int,
    profile: str,
    replicate: str,
    panel: str,
    law: str,
    device: str,
    scope: dict[str, Any],
    tuned_vector: dict[str, list[float]],
    replay_probability_a: float,
    baseline_replay_probability_a: float,
    baseline_updates_seconds: float,
    tuned_updates_seconds: float,
    baseline_host_envelopes_seconds: float,
    tuned_host_envelopes_seconds: float,
    tuned_reserved_bytes: int,
    device_probability_offset: float,
) -> dict[str, Any]:
    if profile == "baseline" and replicate == "replay":
        probability_a = baseline_replay_probability_a
    elif profile == "tuned" and replicate == "replay":
        probability_a = replay_probability_a
    else:
        probability_a = 0.6
    if device == DEVICE_UUIDS[-1]:
        probability_a += device_probability_offset
    vector = BASELINE_VECTOR if profile == "baseline" else tuned_vector
    descriptor = _descriptor(vector, panel)
    steps: list[dict[str, Any]] = []
    for update in UPDATES:
        ordered_ids = [f"{panel}:{law}:{update}:sample-{index}" for index in range(49)] + [
            "worst-token-sample"
        ]
        outputs = [_output(view, probability_a=probability_a) for view in TRAINING_VIEWS]
        model_state = _digest(f"model-state:{panel}:{law}:{profile}:{replicate}:{update}")
        optimizer_state = _digest(f"optimizer-state:{panel}:{law}:{profile}:{replicate}:{update}")
        if replicate == "replay" and (profile == "baseline" or replay_probability_a == 0.6):
            model_state = _digest(f"model-state:{panel}:{law}:{profile}:primary:{update}")
            optimizer_state = _digest(f"optimizer-state:{panel}:{law}:{profile}:primary:{update}")
        outputs_sha256 = semantic_digest(outputs)
        steps.append(
            {
                "update": update,
                "ordered_example_count": 50,
                "ordered_example_ids": ordered_ids,
                "ordered_example_ids_sha256": semantic_digest(ordered_ids),
                "worst_sample_insertion": {
                    "algorithm": "deterministic_effective_50_replace_final_with_corpus_max_if_absent_v1",
                    "engineering_seed": QUALIFICATION_ENGINEERING_SEED,
                    "base_order_sha256": _digest(f"base-order:{panel}:{law}:{update}"),
                    "inserted": True,
                    "position": 49,
                    "worst_sample_id": "worst-token-sample",
                    "replaced_sample_id": f"{panel}:{law}:{update}:replaced",
                    "final_order_sha256": semantic_digest(ordered_ids),
                },
                "rng_receipt": {
                    "helper": "goalzendo.training._step_rng",
                    "engineering_seed": QUALIFICATION_ENGINEERING_SEED,
                    "micro_steps": list(
                        range(
                            (update - 1) * int(PROFILE_CONTRACT[profile]["gradient_accumulation_steps"]),
                            update * int(PROFILE_CONTRACT[profile]["gradient_accumulation_steps"]),
                        )
                    ),
                    "scorer_calls_guarded": True,
                },
                "outputs": outputs,
                "outputs_sha256": outputs_sha256,
                "registered_production_shaped_parity_chunk": (
                    _registered_parity_chunk(
                        profile,
                        law,
                        state=f"after_update_{update}",
                        probability_a=probability_a,
                    )
                    if update in REGISTERED_PARITY_UPDATE_CHECKPOINTS
                    else None
                ),
                "vectors": {kind: copy.deepcopy(descriptor) for kind in VECTOR_KINDS},
                "state_hashes": {
                    "model_state_sha256": model_state,
                    "optimizer_state_sha256": optimizer_state,
                    "combined_state_sha256": semantic_digest(
                        {
                            "model_state_sha256": model_state,
                            "optimizer_state_sha256": optimizer_state,
                        }
                    ),
                    "outputs_sha256": outputs_sha256,
                },
            }
        )
    full_timing = replicate == "primary" and law == "parity"
    updates_seconds = baseline_updates_seconds if profile == "baseline" else tuned_updates_seconds
    projected_update_seconds = updates_seconds if full_timing else 0.0
    host_envelopes_seconds = (
        baseline_host_envelopes_seconds if profile == "baseline" else tuned_host_envelopes_seconds
    )
    projected_host_envelopes_seconds = host_envelopes_seconds if full_timing else 0.0
    reserved = 96 * 1024**3 if profile == "baseline" else tuned_reserved_bytes
    return {
        "schema": PROCESS_SCHEMA,
        "schema_version": PROCESS_SCHEMA_VERSION,
        "identity": {
            "profile": profile,
            "replicate": replicate,
            "panel_id": panel,
            "model_name": PANELS[panel]["model_name"],
            "model_revision": PANELS[panel]["model_revision"],
            "law_family": law,
            "device_uuid": device,
            "device_name": "NVIDIA H200 NVL",
            "process_uuid": str(uuid.UUID(int=counter)),
        },
        "actual_model_execution": True,
        "profile_contract": copy.deepcopy(dict(PROFILE_CONTRACT[profile])),
        "engineering_scope": copy.deepcopy(scope),
        "trainable_parameters": _trainable_manifest(panel),
        "coverage": {
            "updates": list(UPDATES),
            "training_views": list(TRAINING_VIEWS),
            "action_labels": list(ACTION_LABELS),
            "dataset_sha256": _digest(f"dataset:{panel}:{law}"),
            "engineering_seed": QUALIFICATION_ENGINEERING_SEED,
            "ordered_corpus_sample_ids_sha256": _digest(f"corpus-order:{panel}:{law}"),
            "order_algorithm": "deterministic_effective_50_replace_final_with_corpus_max_if_absent_v1",
            "worst_case_token_sample": {
                "sample_id": "worst-token-sample",
                "view": "surface_only",
                "token_length": 640,
                "corpus_max_token_length": 640,
                "included_in_every_update": True,
            },
        },
        "initial_state": {
            "fresh_model_instance": True,
            "fresh_optimizer_instance": True,
            "model_snapshot_sha256": _digest(f"snapshot:{panel}"),
            "model_state_sha256": _digest(f"initial-model:{panel}"),
            "optimizer_state_sha256": _digest(f"initial-optimizer:{panel}"),
        },
        "stochastic_modules": {
            "enumeration_complete": True,
            "dropout_inactive": True,
            "all_stochastic_modules_inactive": True,
            "dropout_module_count": 2,
            "stochastic_module_count": 2,
            "functional_stochasticity_audit": {
                "runtime_training_mode": True,
                "model_config_dropout_fields_enumerated": True,
                "config_probability_count": 1,
                "attention_dropout_fields_enumerated": 1,
                "effective_probability_zero": True,
            },
            "modules": copy.deepcopy(MODULES),
            "modules_sha256": semantic_digest(MODULES),
        },
        "execution_benchmark": _execution_benchmark(profile, law, full_timing=full_timing),
        "steps": steps,
        "memory": {
            "cuda_synchronized_before_read": True,
            "peak_stats_reset_before_run": True,
            "peak_allocated_bytes": reserved - 8 * 1024**3,
            "peak_reserved_bytes": reserved,
        },
        "timing": {
            "tokenizer_load_initialization_seconds": 1.0 if full_timing else 0.0,
            "data_bank_render_tokenization_seconds": 2.0 if full_timing else 0.0,
            "model_load_seconds": 4.0,
            "updates_1_to_8_seconds": projected_update_seconds,
            "updates_1_to_8_cuda_event_seconds_diagnostic": (
                projected_update_seconds * 0.9 if full_timing else 0.0
            ),
            "tokenizer_host_envelopes_8_updates_seconds": projected_host_envelopes_seconds,
            "update_timing_contract": (
                {
                    "shared_helper": "goalzendo.training.train_steps",
                    "algorithm": "sft",
                    "total_steps": 8,
                    "batch_size": int(PROFILE_CONTRACT[profile]["train_batch_size"]),
                    "gradient_accumulation_steps": int(
                        PROFILE_CONTRACT[profile]["gradient_accumulation_steps"]
                    ),
                    "warmup_steps": 12,
                    "max_grad_norm": 1.0,
                    "parameter_finite_check_interval": 0,
                    "seed": QUALIFICATION_ENGINEERING_SEED,
                    "candidate_choice_geometry_included": True,
                    "hooks": None,
                    "fresh_state_final_global_step": 8,
                    "fresh_state_final_micro_step": 8
                    * int(PROFILE_CONTRACT[profile]["gradient_accumulation_steps"]),
                    "standardized_worst_shape_training": _standardized_worst_shape_training(
                        profile,
                        panel,
                        host_envelopes_seconds=host_envelopes_seconds,
                    ),
                }
                if full_timing
                else None
            ),
            "evaluation_13_boundaries_seconds": 20.0 if full_timing else 0.0,
            "evaluation_callback_io_seconds": 1.0 if full_timing else 0.0,
            "final_checkpoint_io_seconds": 2.0 if full_timing else 0.0,
            "outcome_seal_seconds": 1.0 if full_timing else 0.0,
            "update_timing_method": (
                "shared_train_steps_8_update_wall_clock_single_pre_post_cuda_sync_horizon"
                if full_timing
                else "not_measured_registered_numeric_only"
            ),
            "fresh_timing_only_model_optimizer": full_timing,
            "vector_state_capture_excluded_from_update_timing": full_timing,
            "warmup_model_discarded_before_measurement": full_timing,
            "capture_path_cuda_event_seconds_diagnostic_only": updates_seconds * 0.8,
            "warmup_excluded": True,
            "profile_timing_order": (
                ["baseline", "tuned"] if DEVICE_UUIDS.index(device) % 2 == 0 else ["tuned", "baseline"]
            ),
            "measured_end_to_end_seconds": (
                (31.0 + host_envelopes_seconds if full_timing else 4.0) + projected_update_seconds
            ),
            "evaluation_boundary_count": 13 if full_timing else 0,
            "projection_training_updates": 1_000,
            "worker_mix": {
                "runs_per_worker": 40,
                "panel_runs": {panel_id: 20 for panel_id in PANELS},
            },
            "projection_safety_multiplier": 1.20,
        },
    }


def _output_comparison(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "left_outputs_sha256": semantic_digest(left),
        "right_outputs_sha256": semantic_digest(right),
        "sample_identity_exact": all(
            (a["view"], a["sample_id"], a["token_length"]) == (b["view"], b["sample_id"], b["token_length"])
            for a, b in zip(left, right, strict=True)
        ),
        "maximum_normalized_score_difference": max(
            abs(float(a) - float(b))
            for left_output, right_output in zip(left, right, strict=True)
            for a, b in zip(
                left_output["normalized_log_scores"],
                right_output["normalized_log_scores"],
                strict=True,
            )
        ),
        "maximum_probability_difference": max(
            abs(float(a) - float(b))
            for left_output, right_output in zip(left, right, strict=True)
            for a, b in zip(left_output["probabilities"], right_output["probabilities"], strict=True)
        ),
        "actions_identical": all(
            (a["action_index"], a["action_label"]) == (b["action_index"], b["action_label"])
            for a, b in zip(left, right, strict=True)
        ),
    }


def _comparison_body(
    *,
    kind: str,
    panel: str,
    law: str,
    device: str,
    left: dict[str, Any],
    right: dict[str, Any],
    left_vector: dict[str, list[float]],
    right_vector: dict[str, list[float]],
) -> dict[str, Any]:
    vector_metric = _metric_for_panel(left_vector, right_vector, panel)
    left_binding = canonical_process_comparison_binding(left)
    right_binding = canonical_process_comparison_binding(right)
    replay_comparison = kind.endswith("primary_vs_replay")
    receipt = build_paired_execution_receipt(
        left_record=left,
        right_record=right,
        left_observed_binding=left_binding,
        right_observed_binding=right_binding,
        pair_capture_receipt_sha256=_digest(f"paired:{kind}:{panel}:{law}:{device}"),
        left_capture_mode="canonical_process",
        right_capture_mode=("exact_logical_identity_replay" if replay_comparison else "canonical_process"),
    )
    return {
        "schema": COMPARISON_SCHEMA,
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "kind": kind,
        "panel_id": panel,
        "law_family": law,
        "device_uuid": device,
        "left_record_digest": left["record_digest"],
        "right_record_digest": right["record_digest"],
        "paired_execution_receipt": receipt,
        "steps": [
            {
                "update": update,
                "outputs": _output_comparison(left_step["outputs"], right_step["outputs"]),
                "vectors": {kind: copy.deepcopy(vector_metric) for kind in VECTOR_KINDS},
            }
            for update, left_step, right_step in zip(
                UPDATES,
                left["steps"],
                right["steps"],
                strict=True,
            )
        ],
    }


def _evidence(
    *,
    root: Path | None = None,
    tuned_vector: dict[str, list[float]] | None = None,
    replay_probability_a: float = 0.6,
    baseline_replay_probability_a: float = 0.6,
    baseline_updates_seconds: float = 5.0,
    tuned_updates_seconds: float = 2.0,
    baseline_host_envelopes_seconds: float = 1.0,
    tuned_host_envelopes_seconds: float = 1.0,
    tuned_reserved_bytes: int = 110 * 1024**3,
    capacity_reserved_bytes: int = 110 * 1024**3,
    device_probability_offset: float = 0.0,
) -> dict[str, Any]:
    engineering_root = root or Path(f"/tmp/{ENGINEERING_ROOT_PREFIX}synthetic")
    scope = {
        "root": str(engineering_root.resolve()),
        "root_class": "engineering_only",
        "data_class": "synthetic_engineering_only",
        "outcomes_seen": False,
        "itt_ledger_created": False,
        "production_artifacts_read": False,
        "production_artifacts_written": False,
    }
    active_tuned_vector = tuned_vector or TUNED_VECTOR
    records: list[dict[str, Any]] = []
    records_by_key: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    counter = 1
    replicates = {
        "baseline": ("primary", "replay"),
        "tuned": ("primary", "replay"),
    }
    for profile in ("baseline", "tuned"):
        for replicate in replicates[profile]:
            for panel in PANELS:
                for law in LAW_FAMILIES:
                    for device in DEVICE_UUIDS:
                        record = seal_process_record(
                            _process_body(
                                counter=counter,
                                profile=profile,
                                replicate=replicate,
                                panel=panel,
                                law=law,
                                device=device,
                                scope=scope,
                                tuned_vector=active_tuned_vector,
                                replay_probability_a=replay_probability_a,
                                baseline_replay_probability_a=baseline_replay_probability_a,
                                baseline_updates_seconds=baseline_updates_seconds,
                                tuned_updates_seconds=tuned_updates_seconds,
                                baseline_host_envelopes_seconds=baseline_host_envelopes_seconds,
                                tuned_host_envelopes_seconds=tuned_host_envelopes_seconds,
                                tuned_reserved_bytes=tuned_reserved_bytes,
                                device_probability_offset=device_probability_offset,
                            )
                        )
                        key = (profile, replicate, panel, law, device)
                        records.append(record)
                        records_by_key[key] = record
                        counter += 1

    comparisons: list[dict[str, Any]] = []
    for panel in PANELS:
        for law in LAW_FAMILIES:
            device = DEVICE_UUIDS[0]
            baseline = records_by_key[("baseline", "primary", panel, law, device)]
            tuned = records_by_key[("tuned", "primary", panel, law, device)]
            comparisons.append(
                seal_comparison_record(
                    _comparison_body(
                        kind="baseline_vs_tuned",
                        panel=panel,
                        law=law,
                        device=device,
                        left=baseline,
                        right=tuned,
                        left_vector=BASELINE_VECTOR,
                        right_vector=active_tuned_vector,
                    )
                )
            )
    capacity_probe = _capacity_probe(passed_peak_reserved_bytes=capacity_reserved_bytes)
    record_manifest = semantic_digest(
        {
            "qualification_branch": "tuned_probe_passed_full",
            "tuned_capacity_probe_digest": capacity_probe["probe_digest"],
            "process_record_digests": sorted(item["record_digest"] for item in records),
            "comparison_record_digests": sorted(item["comparison_digest"] for item in comparisons),
        }
    )
    evidence_source = {
        **EVIDENCE_BOUNDARY,
        "capture_command_sha256": _digest("capture-command"),
        "capture_implementation_sha256": _digest("capture-implementation"),
        "records_manifest_sha256": record_manifest,
    }
    accumulator_row_count = sum(
        len(metric["accumulator_rows"])
        for comparison in comparisons
        for step in comparison["steps"]
        for metric in step["vectors"].values()
    )
    return seal_evidence(
        {
            "schema": EVIDENCE_SCHEMA,
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "actual_model_execution": True,
            "qualification_branch": "tuned_probe_passed_full",
            "tuned_capacity_probe": capacity_probe,
            "evidence_source": evidence_source,
            "engineering_scope": scope,
            "freeze_binding": {
                "freeze_file_sha256": _digest("freeze-file"),
                "freeze_digest": _digest("freeze-semantic"),
            },
            "provision_binding": {
                "file_sha256": _digest("provision-file"),
                "receipt_digest": _digest("provision-semantic"),
                "pod_id": "pod-synthetic",
            },
            "model_receipt_bindings": {
                panel: {
                    "panel_id": panel,
                    "file_sha256": _digest(f"model-receipt-file:{panel}"),
                    "receipt_digest": _digest(f"model-receipt:{panel}"),
                }
                for panel in PANELS
            },
            "model_integration_audit_bindings": {
                panel: {
                    "panel_id": panel,
                    "file_sha256": _digest(f"integration-file:{panel}"),
                    "audit_digest": _digest(f"integration-audit:{panel}"),
                    "report_digest": _digest(f"integration-report:{panel}"),
                }
                for panel in PANELS
            },
            "storage_contract": {
                "maximum_persisted_evidence_bytes": 512 * 1024**2,
                "observed_persisted_evidence_bytes": 128 * 1024**2,
                "accumulator_row_count": accumulator_row_count,
                "raw_vector_bytes_persisted": 0,
                "vector_persistence": "bounded_inline_per_parameter_accumulator_rows_only",
                "transient_raw_cleanup_before_itt_required": True,
                "transient_raw_cleanup_receipt_required": True,
                "transient_raw_cleanup_timing": "during_producer_before_compact_evidence_seal",
                "compact_evidence_retained_through_final_gate": True,
            },
            "expected_device_uuids": list(DEVICE_UUIDS),
            "process_records": records,
            "comparison_records": comparisons,
        }
    )


@pytest.fixture(scope="module")
def valid_evidence() -> dict[str, Any]:
    return _evidence()


def _reseal(payload: dict[str, Any], digest_field: str) -> dict[str, Any]:
    result = copy.deepcopy(payload)
    result.pop(digest_field, None)
    result[digest_field] = semantic_digest(result)
    return result


def _reseal_evidence_manifest(evidence: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(evidence)
    result["evidence_source"]["records_manifest_sha256"] = semantic_digest(
        {
            "qualification_branch": result["qualification_branch"],
            "tuned_capacity_probe_digest": result["tuned_capacity_probe"]["probe_digest"],
            "process_record_digests": sorted(item["record_digest"] for item in result["process_records"]),
            "comparison_record_digests": sorted(
                item["comparison_digest"] for item in result["comparison_records"]
            ),
        }
    )
    return _reseal(result, "evidence_digest")


def _fallback_evidence() -> dict[str, Any]:
    result = copy.deepcopy(_evidence())
    result["qualification_branch"] = "tuned_capacity_fallback_baseline"
    result["tuned_capacity_probe"] = _capacity_probe(failed_device=DEVICE_UUIDS[-1])
    baseline_records = [
        copy.deepcopy(item) for item in result["process_records"] if item["identity"]["profile"] == "baseline"
    ]
    for record in baseline_records:
        record["timing"]["profile_timing_order"] = ["baseline"]
    result["process_records"] = [_reseal(item, "record_digest") for item in baseline_records]
    result["comparison_records"] = []
    result["storage_contract"]["accumulator_row_count"] = 0
    return _reseal_evidence_manifest(result)


def test_streaming_vector_metrics_are_sorted_float64_and_handle_zero_norms() -> None:
    forward = streaming_vector_metrics(
        {"z": [3.0], "a": [1.0, 2.0]},
        {"z": [3.0], "a": [1.0, 2.0]},
    )
    reversed_keys = streaming_vector_metrics(
        {"a": [1.0, 2.0], "z": [3.0]},
        {"a": [1.0, 2.0], "z": [3.0]},
    )
    assert forward == reversed_keys
    assert forward["accumulator_dtype"] == "float64"
    assert forward["parameter_iteration_order"] == "sorted_parameter_keys"
    assert forward["element_count"] == 3
    assert [row["parameter_key"] for row in forward["accumulator_rows"]] == ["a", "z"]
    assert forward["accumulator_rows_sha256"] == semantic_digest(forward["accumulator_rows"])
    assert streaming_metric_from_accumulator_rows(forward["accumulator_rows"]) == forward
    assert forward["cosine"] == pytest.approx(1.0)
    assert forward["relative_l2"] == pytest.approx(0.0)

    both_zero = streaming_vector_metrics({"p": [0.0, 0.0]}, {"p": [0.0, 0.0]})
    assert both_zero["zero_norm_case"] == "both_zero"
    assert both_zero["cosine"] == 1.0
    assert both_zero["relative_l2"] == 0.0
    one_zero = streaming_vector_metrics({"p": [0.0, 0.0]}, {"p": [1.0, 0.0]})
    assert one_zero["zero_norm_case"] == "left_zero"
    assert one_zero["cosine"] == 0.0
    assert one_zero["relative_l2"] == 1.0

    with pytest.raises(QualificationError, match="parameter keys"):
        streaming_vector_metrics({"a": [1.0]}, {"b": [1.0]})
    with pytest.raises(QualificationError, match="non-finite"):
        streaming_vector_metrics({"a": [float("nan")]}, {"a": [0.0]})


def test_exact_64_process_evidence_qualifies_tuned_and_revalidates_report(
    valid_evidence: dict[str, Any],
) -> None:
    assert validate_qualification_evidence(valid_evidence) == valid_evidence
    assert (
        validate_process_record(valid_evidence["process_records"][0]) == valid_evidence["process_records"][0]
    )
    report = create_qualification_report(valid_evidence)

    assert report["schema"] == QUALIFICATION_SCHEMA
    assert report["schema_version"] == QUALIFICATION_SCHEMA_VERSION
    assert report["actual_model_execution"] is True
    assert report["execution_claim"]["models_executed_by_this_aggregator"] is False
    assert report["evidence_binding"]["process_record_count"] == 64
    assert report["evidence_binding"]["comparison_record_count"] == 4
    assert report["gpu_uuids"] == list(DEVICE_UUIDS)
    assert {
        panel: manifest["trainable_numel"]
        for panel, manifest in report["trainable_parameter_manifests"].items()
    } == dict(TRAINABLE_NUMEL)
    assert report["eligibility"] == {"baseline": True, "tuned": True}
    assert report["selection_candidate"] == "tuned"
    assert report["selection_authorized"] is False
    assert report["outcomes_seen"] is False
    assert report["weight_updates_scope"] == "engineering_qualification_only"
    assert report["g01_launch_authorized"] is False
    assert report["checks"]["ordered_data_exact"] is True
    assert report["checks"]["baseline_replay"]["passed"] is True
    assert report["checks"]["tuned_replay"]["passed"] is True
    assert report["checks"]["cross_gpu_equivalence"]["baseline"]["group_count"] == 64
    assert report["checks"]["cross_gpu_equivalence"]["baseline"]["passed"] is True
    assert report["checks"]["cross_gpu_equivalence"]["tuned"]["passed"] is True
    registered = report["checks"]["registered_evaluation_parity"]
    assert registered["states"] == [
        "initial",
        "after_update_1",
        "after_update_2",
        "after_update_4",
        "after_update_8",
    ]
    assert registered["group_count"] == 80
    assert registered["shape_authentication_group_count"] == 16
    assert registered["numeric_parity_group_count"] == 64
    assert registered["row_comparison_count"] == 64 * 128 * len(TRAINING_VIEWS)
    assert registered["single_registered_partition_exact"] is True
    assert registered["passed"] is True
    assert report["transient_raw_cleanup_before_itt_required"] is True
    assert report["compact_evidence_retained_through_final_gate"] is True
    assert report["storage_contract"]["raw_vector_bytes_persisted"] == 0
    assert report["checks"]["tuned_memory"]["maximum_peak_reserved_bytes"] == 110 * 1024**3
    assert report["checks"]["performance"]["throughput_ratio"] >= 1.20
    assert report["projections"]["baseline"]["projected_wall_seconds"] <= 12 * 60 * 60
    assert report["projections"]["tuned"]["projected_wall_seconds"] <= 12 * 60 * 60
    assert report["report_digest"] == semantic_digest(
        {key: value for key, value in report.items() if key != "report_digest"}
    )
    assert validate_qualification_report(report) == report


def test_recoverable_tuned_capacity_probe_seals_baseline_only_branch() -> None:
    evidence = validate_qualification_evidence(_fallback_evidence())
    assert evidence["qualification_branch"] == "tuned_capacity_fallback_baseline"
    assert len(evidence["process_records"]) == 32
    assert evidence["comparison_records"] == []

    report = validate_qualification_report(create_qualification_report(evidence))
    assert report["qualification_branch"] == "tuned_capacity_fallback_baseline"
    assert report["eligibility"] == {"baseline": True, "tuned": False}
    assert report["selection_candidate"] == "baseline"
    assert report["checks"]["tuned_replay"]["comparison_count"] == 0
    assert report["checks"]["cross_gpu_equivalence"]["tuned"]["group_count"] == 0
    assert report["checks"]["performance"]["tuned"]["available"] is False


def test_tuned_vector_failure_falls_back_to_eligible_baseline() -> None:
    evidence = _evidence(tuned_vector={"decoder.bias": [0.25], "decoder.weight": [-1.0, 2.0, -3.0]})
    report = create_qualification_report(evidence)
    assert report["checks"]["baseline_vs_tuned"]["vector_thresholds_passed"] is False
    assert report["eligibility"] == {"baseline": True, "tuned": False}
    assert report["selection_candidate"] == "baseline"
    assert validate_qualification_report(report) == report


def test_nonexact_tuned_replay_falls_back_without_fabricating_a_pass() -> None:
    evidence = _evidence(replay_probability_a=0.59)
    report = create_qualification_report(evidence)
    assert report["checks"]["tuned_replay"]["state_optimizer_output_and_vector_hashes_exact"] is False
    assert report["checks"]["tuned_replay"]["passed"] is False
    assert report["eligibility"] == {"baseline": True, "tuned": False}
    assert report["selection_candidate"] == "baseline"
    assert validate_qualification_report(report) == report


def test_baseline_replay_requires_exact_output_hashes() -> None:
    evidence = _evidence(baseline_replay_probability_a=0.59)
    report = create_qualification_report(evidence)
    assert report["checks"]["baseline_replay"]["state_optimizer_output_and_vector_hashes_exact"] is False
    assert report["checks"]["baseline_replay"]["passed"] is False
    assert report["eligibility"] == {"baseline": False, "tuned": True}
    assert report["selection_candidate"] == "tuned"
    assert validate_qualification_report(report) == report


def test_cross_gpu_output_mismatch_disqualifies_both_profiles() -> None:
    report = create_qualification_report(_evidence(device_probability_offset=0.01))
    assert report["checks"]["cross_gpu_equivalence"]["baseline"]["exact_output_hashes"] is False
    assert report["checks"]["cross_gpu_equivalence"]["tuned"]["exact_output_hashes"] is False
    assert report["eligibility"] == {"baseline": False, "tuned": False}
    assert report["selection_candidate"] is None
    assert validate_qualification_report(report) == report


def test_tuned_memory_ceiling_is_inclusive_and_fail_closed() -> None:
    inclusive = create_qualification_report(_evidence(capacity_reserved_bytes=120 * 1024**3))
    assert inclusive["checks"]["tuned_memory"]["passed"] is True
    diagnostic_only = create_qualification_report(_evidence(tuned_reserved_bytes=120 * 1024**3 + 1))
    assert diagnostic_only["checks"]["tuned_memory"]["passed"] is True
    fallback = create_qualification_report(_fallback_evidence())
    assert fallback["checks"]["tuned_memory"]["passed"] is False
    assert fallback["selection_candidate"] == "baseline"
    assert validate_qualification_report(fallback) == fallback


def test_registered_parity_checkpoint_schedule_is_fail_closed(
    valid_evidence: dict[str, Any],
) -> None:
    record = copy.deepcopy(valid_evidence["process_records"][0])
    record["steps"][0]["registered_production_shaped_parity_chunk"] = None
    record = _reseal(record, "record_digest")
    with pytest.raises(QualificationError, match="registered parity chunk"):
        validate_process_record(record)


def test_full_timing_requires_call_by_call_componentwise_law_dominance(
    valid_evidence: dict[str, Any],
) -> None:
    record = next(
        copy.deepcopy(item)
        for item in valid_evidence["process_records"]
        if item["execution_benchmark"]["timing_representative"] is True
    )
    proof = record["execution_benchmark"]["worst_law_proof"]
    shape = proof["record_call_shapes"][0]
    old_elements = shape["padded_elements"]
    shape["max_seq_len"] = proof["other_law_call_shapes"][0]["max_seq_len"] - 1
    shape["padded_elements"] = shape["flattened_prompt_count"] * shape["max_seq_len"]
    proof["record_padded_token_elements"] += shape["padded_elements"] - old_elements
    proof["record_call_shapes_sha256"] = semantic_digest(proof["record_call_shapes"])
    record["execution_benchmark"] = _reseal(record["execution_benchmark"], "benchmark_digest")
    record = _reseal(record, "record_digest")
    with pytest.raises(QualificationError, match="conservative tokenized production workload"):
        validate_process_record(record)


def test_host_envelope_partition_and_sum_are_replayed(
    valid_evidence: dict[str, Any],
) -> None:
    record = next(
        copy.deepcopy(item)
        for item in valid_evidence["process_records"]
        if item["execution_benchmark"]["timing_representative"] is True
    )
    host_contract = record["timing"]["update_timing_contract"]["standardized_worst_shape_training"][
        "tokenizer_host_envelope_contracts"
    ]
    host_contract["envelopes"]["contextual_token_work_dominance"]["profile_call_partition_sizes"][0] -= 1
    record = _reseal(record, "record_digest")
    with pytest.raises(QualificationError, match="host-envelope timing geometry"):
        validate_process_record(record)


def test_contextual_one_token_and_full_branch_maximum_proof_is_replayed(
    valid_evidence: dict[str, Any],
) -> None:
    record = next(
        copy.deepcopy(item)
        for item in valid_evidence["process_records"]
        if item["execution_benchmark"]["timing_representative"] is True
    )
    proof = record["timing"]["update_timing_contract"]["standardized_worst_shape_training"][
        "global_maximum_proof"
    ]
    proof["contextual_scorer_branch_invariant"]["required_token_count_per_action_continuation"] = 2
    proof["proof_digest"] = semantic_digest(
        {key: value for key, value in proof.items() if key != "proof_digest"}
    )
    record = _reseal(record, "record_digest")
    with pytest.raises(QualificationError, match="contextual A/B continuation"):
        validate_process_record(record)


def test_hard_host_tier_must_bind_the_exhaustive_per_run_metric_maximum(
    valid_evidence: dict[str, Any],
) -> None:
    record = next(
        copy.deepcopy(item)
        for item in valid_evidence["process_records"]
        if item["execution_benchmark"]["timing_representative"] is True
    )
    proof = record["timing"]["update_timing_contract"]["standardized_worst_shape_training"][
        "global_maximum_proof"
    ]
    proof["run_receipts"][0]["maximum_contextual_token_work"] += 1
    proof["run_receipts_sha256"] = semantic_digest(proof["run_receipts"])
    proof["proof_digest"] = semantic_digest(
        {key: value for key, value in proof.items() if key != "proof_digest"}
    )
    record = _reseal(record, "record_digest")
    with pytest.raises(QualificationError, match="host-envelope ordering/metric proof"):
        validate_process_record(record)


def test_final_seal_requires_hash_scan_and_chmod_zero(
    valid_evidence: dict[str, Any],
) -> None:
    record = next(
        copy.deepcopy(item)
        for item in valid_evidence["process_records"]
        if item["execution_benchmark"]["timing_representative"] is True
    )
    record["execution_benchmark"]["final_seal_io"]["outcome_file_seal_modes"]["summary.json"] = 0o600
    record["execution_benchmark"] = _reseal(
        record["execution_benchmark"],
        "benchmark_digest",
    )
    record = _reseal(record, "record_digest")
    with pytest.raises(QualificationError, match="chmod-000"):
        validate_process_record(record)


def test_slow_baseline_and_failed_tuned_yield_no_candidate() -> None:
    evidence = _evidence(
        tuned_vector={"decoder.bias": [0.25], "decoder.weight": [-1.0, 2.0, -3.0]},
        baseline_updates_seconds=10.0,
    )
    report = create_qualification_report(evidence)
    assert report["projections"]["baseline"]["projected_wall_seconds"] > 12 * 60 * 60
    assert report["eligibility"] == {"baseline": False, "tuned": False}
    assert report["selection_candidate"] is None
    assert report["overall_qualification_passed"] is False
    assert validate_qualification_report(report) == report


def test_synthetic_host_stress_cannot_fabricate_the_speedup_gate() -> None:
    report = create_qualification_report(
        _evidence(
            baseline_updates_seconds=2.0,
            tuned_updates_seconds=2.0,
            baseline_host_envelopes_seconds=4.0,
            tuned_host_envelopes_seconds=0.5,
        )
    )
    performance = report["checks"]["performance"]
    assert performance["throughput_ratio_basis"].endswith("excluding_additive_synthetic_host_stress")
    assert set(performance["per_device_throughput_ratio"].values()) == {1.0}
    assert set(performance["per_device_clean_train_steps_ratio"].values()) == {1.0}
    assert performance["clean_train_steps_throughput_passed"] is False
    assert performance["throughput_passed"] is False
    assert report["eligibility"]["tuned"] is False
    assert validate_qualification_report(report) == report


def test_incomplete_or_unauthenticated_evidence_never_constructs_a_report(
    valid_evidence: dict[str, Any],
) -> None:
    missing = copy.deepcopy(valid_evidence)
    missing["process_records"].pop()
    missing = _reseal(missing, "evidence_digest")
    with pytest.raises(QualificationError, match="exactly 64"):
        create_qualification_report(missing)

    not_actual = copy.deepcopy(valid_evidence)
    not_actual["actual_model_execution"] = False
    not_actual = _reseal(not_actual, "evidence_digest")
    with pytest.raises(QualificationError, match="actual-model"):
        create_qualification_report(not_actual)

    contaminated = copy.deepcopy(valid_evidence)
    contaminated["engineering_scope"]["outcomes_seen"] = True
    contaminated = _reseal(contaminated, "evidence_digest")
    with pytest.raises(QualificationError, match="saw outcomes"):
        create_qualification_report(contaminated)

    oversized = copy.deepcopy(valid_evidence)
    oversized["storage_contract"]["observed_persisted_evidence_bytes"] = 512 * 1024**2 + 1
    oversized = _reseal(oversized, "evidence_digest")
    with pytest.raises(QualificationError, match="storage ceiling"):
        create_qualification_report(oversized)


def test_accumulator_rows_are_replayed_instead_of_trusting_global_metrics(
    valid_evidence: dict[str, Any],
) -> None:
    forged = copy.deepcopy(valid_evidence)
    comparison = forged["comparison_records"][0]
    comparison["steps"][0]["vectors"]["raw_preclip_gradient"]["cosine"] = 0.5
    forged["comparison_records"][0] = _reseal(comparison, "comparison_digest")
    forged = _reseal_evidence_manifest(forged)
    with pytest.raises(QualificationError, match="cosine"):
        validate_qualification_evidence(forged)


def test_partial_vector_is_rejected_against_frozen_trainable_numel(
    valid_evidence: dict[str, Any],
) -> None:
    record = copy.deepcopy(valid_evidence["process_records"][0])
    record["steps"][0]["vectors"]["raw_preclip_gradient"]["element_count"] -= 1
    record = _reseal(record, "record_digest")
    with pytest.raises(QualificationError, match="exact frozen trainable parameter count"):
        validate_process_record(record)


def test_paired_execution_receipt_must_exactly_replay_canonical_process(
    valid_evidence: dict[str, Any],
) -> None:
    forged = copy.deepcopy(valid_evidence)
    comparison = forged["comparison_records"][0]
    receipt = comparison["paired_execution_receipt"]
    observed = receipt["left_observed_binding"]
    observed["steps"][0]["ordered_example_ids_sha256"] = _digest("wrong-ordered-data")
    observed = _reseal(observed, "binding_sha256")
    receipt["left_observed_binding"] = observed
    receipt["left_observed_binding_sha256"] = observed["binding_sha256"]
    receipt["exact_match_to_canonical_process_records"] = False
    forged["comparison_records"][0] = _reseal(comparison, "comparison_digest")
    forged = _reseal_evidence_manifest(forged)
    with pytest.raises(QualificationError, match=r"explicit order binding|does not exactly match"):
        validate_qualification_evidence(forged)


def test_report_validator_recomputes_eligibility_after_valid_self_digest(
    valid_evidence: dict[str, Any],
) -> None:
    report = create_qualification_report(valid_evidence)
    forged = copy.deepcopy(report)
    forged["eligibility"]["tuned"] = False
    forged = _reseal(forged, "report_digest")
    with pytest.raises(QualificationError, match="eligibility is inconsistent"):
        validate_qualification_report(forged)

    bad_count = copy.deepcopy(report)
    bad_count["evidence_binding"]["process_record_count"] = 47
    bad_count = _reseal(bad_count, "report_digest")
    with pytest.raises(QualificationError, match="process count"):
        validate_qualification_report(bad_count)

    forged_cross_gpu = copy.deepcopy(report)
    binding = forged_cross_gpu["checks"]["cross_gpu_equivalence"]["baseline"]["groups"][0]["device_bindings"][
        -1
    ]
    binding["outputs_sha256"] = _digest("forged-cross-gpu-output")
    binding["state_hashes"]["outputs_sha256"] = binding["outputs_sha256"]
    forged_cross_gpu = _reseal(forged_cross_gpu, "report_digest")
    with pytest.raises(QualificationError, match="output_hashes_exact is inconsistent"):
        validate_qualification_report(forged_cross_gpu)


def test_authenticated_evidence_replay_writes_once_inside_engineering_root(tmp_path: Path) -> None:
    root = tmp_path / f"{ENGINEERING_ROOT_PREFIX}authenticated-replay"
    root.mkdir()
    evidence = _evidence(root=root)
    evidence_path = root / "qualification-evidence.json"
    evidence_path.write_text(json.dumps(evidence, sort_keys=True), encoding="utf-8")
    output = root / "profile-qualification.json"

    report = replay_authenticated_evidence(evidence_path=evidence_path, output=output)
    assert output.is_file()
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert validate_qualification_report(report) == report
    with pytest.raises(QualificationError, match="already exists"):
        replay_authenticated_evidence(evidence_path=evidence_path, output=output)
    with pytest.raises(QualificationError, match="canonical engineering-only output path"):
        replay_authenticated_evidence(
            evidence_path=evidence_path,
            output=tmp_path / "profile-qualification.json",
        )
