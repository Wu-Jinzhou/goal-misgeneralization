"""Authenticated prospective execution freeze for GoalZendo G00-F.

This package is deliberately additive.  It authenticates the immutable G00-D
GoalZendo implementation, then permits only the exact 160 G00-F run
specifications enumerated by an externally digest-pinned freeze.  It never
changes the frozen source tree and never authorizes G01.
"""

from __future__ import annotations

import copy
import hashlib
import ipaddress
import json
import math
import os
import re
import shlex
import shutil
import socket
import stat
import subprocess
import tempfile
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from goalzendo.artifacts import RunStore, implementation_provenance, stable_hash
from goalzendo.config import canonical_config, get_path, load_config
from goalzendo.runner import RunSpec, build_plan, execute_plan

FREEZE_SCHEMA = "goalzendo.g00f_h200_execution_freeze"
FREEZE_SCHEMA_VERSION = 1
SOURCE_MANIFEST_SCHEMA = "goalzendo.g00f_h200_additive_source_manifest"
SOURCE_MANIFEST_SCHEMA_VERSION = 1
MODEL_RECEIPT_SCHEMA = "goalzendo.g00f_h200_model_snapshot_receipt"
MODEL_RECEIPT_SCHEMA_VERSION = 1
MODEL_INTEGRATION_AUDIT_SCHEMA = "goalzendo.g00f_h200_model_integration_audit"
MODEL_INTEGRATION_AUDIT_SCHEMA_VERSION = 1
RUNPOD_PROVISION_RECEIPT_SCHEMA = "goalzendo.g00f_h200_runpod_provision_receipt"
RUNPOD_PROVISION_RECEIPT_SCHEMA_VERSION = 1
RUNPOD_SSH_IDENTITY_RECEIPT_SCHEMA = "goalzendo.g00f_h200_runpod_ssh_identity_receipt"
RUNPOD_SSH_IDENTITY_RECEIPT_SCHEMA_VERSION = 1
LAUNCH_RECEIPT_SCHEMA = "goalzendo.g00f_h200_worker_launch_receipt"
LAUNCH_RECEIPT_SCHEMA_VERSION = 1
RUN_BINDING_SCHEMA = "goalzendo.g00f_h200_run_freeze_binding"
RUN_BINDING_SCHEMA_VERSION = 1
ITT_LEDGER_SCHEMA = "goalzendo.g00f_h200_itt_ledger"
ITT_LEDGER_SCHEMA_VERSION = 1
ATTEMPT_RECEIPT_SCHEMA = "goalzendo.g00f_h200_attempt_receipt"
ATTEMPT_RECEIPT_SCHEMA_VERSION = 1
PANEL_UNSEAL_SCHEMA = "goalzendo.g00f_h200_panel_unseal_receipt"
PANEL_UNSEAL_SCHEMA_VERSION = 1
PROFILE_QUALIFICATION_SCHEMA = "goalzendo.g00f_h200_profile_qualification"
PROFILE_QUALIFICATION_SCHEMA_VERSION = 1
PROFILE_QUALIFICATION_PRODUCER_SCHEMA = "goalzendo.g00f_h200_profile_qualification_producer_receipt"
PROFILE_QUALIFICATION_PRODUCER_SCHEMA_VERSION = 1
PROFILE_QUALIFICATION_HANDOFF_SCHEMA = "goalzendo.g00f_h200_profile_qualification_handoff"
PROFILE_QUALIFICATION_HANDOFF_SCHEMA_VERSION = 1
PROFILE_QUALIFICATION_CLEANUP_SCHEMA = "goalzendo.g00f_h200_profile_qualification_cleanup"
PROFILE_QUALIFICATION_CLEANUP_SCHEMA_VERSION = 1
STORAGE_PREFLIGHT_SCHEMA = "goalzendo.g00f_h200_storage_preflight"
STORAGE_PREFLIGHT_SCHEMA_VERSION = 1
PROFILE_SELECTION_SCHEMA = "goalzendo.g00f_h200_profile_selection"
PROFILE_SELECTION_SCHEMA_VERSION = 1

PROFILE_QUALIFICATION_COPY_NAMES: Mapping[str, str] = {
    "evidence": "h200-profile-qualification-evidence.json",
    "producer_receipt": "h200-profile-qualification-producer-receipt.json",
    "report": "h200-profile-qualification.json",
    "transient_inventory": "h200-profile-qualification-transient-inventory.json",
}

OPERATIONAL_FAILURE_TRIGGERS = frozenset(
    {
        "coordinator_exit_trap",
        "infrastructure_or_execution_exception",
        "launcher_partial_start",
        "launcher_signal",
        "monotonic_14h_deadline",
        "watchdog_process_exit",
        "worker_nonzero_exit",
    }
)

FROZEN_GOALZENDO_IMPLEMENTATION_FINGERPRINT = (
    "1a8146377b9a9620690025671614edb2dd20d214f4e195da3cf3528809b2c694"
)
G00F_GUARD = "G00F_H200_FROZEN_SELECTOR_AND_EXECUTION_MANIFEST_REQUIRED"
EXECUTION_PROFILES = ("baseline", "tuned")
H200_FREEZE_OUTPUT_RELATIVE = "reproducibility/goalzendo/g00f-h200-execution-freeze-20260811"
H200_SOURCE_ARCHIVE_NAME = "g00f-h200-execution-source.tar.gz"
H200_SOURCE_BUNDLE_MANIFEST_NAME = "g00f-h200-source-bundle-manifest.json"
HISTORICAL_H100_PARENT_FILES: Mapping[str, str] = {
    "reproducibility/goalzendo/g00f-execution-freeze-20260811/execution-freeze.json": (
        "b2e385488ea7eb7c6f7bfc834c32ff2707aa9f96b718ea62b7ae92c9a4b8df38"
    ),
    "reproducibility/goalzendo/g00f-execution-freeze-20260811/g00f-execution-source.tar.gz": (
        "6b55d42bd2a75c9fc40f9c67b332c7ad5bd57024c26bd49447ef6bc831dece3a"
    ),
    "reproducibility/goalzendo/g00f-execution-freeze-20260811/g00f-source-bundle-manifest.json": (
        "482f57dc5e0b19a9c1b6ab18a1d63283ce559d288b8a00cc08e51af464d5f8f7"
    ),
}
HISTORICAL_H100_FREEZE_DIGEST = "2b04534b74873a88c73791a17a5c5a7b66a9ddd472b11168034f11517a8d5047"
HISTORICAL_H100_PANEL_IDENTITIES: Mapping[str, Mapping[str, Any]] = {
    "g00f-0p5b": {
        "canonical_config_digest": ("dbbc10ce500089ddddfdd6a3c540d7e7825dcd1913df1c7ca90f7a02411190d8"),
        "config_path": "configs/goalzendo/g00f_capability_repair_0p5b.yaml",
        "config_sha256": "3c1ef3113be76373892b256b60ec5540419bd932d813a3db230d9a3f913cea30",
        "plan_key_digest": "1b7ffcd2b588ca69103be025e4e28a63f8877bbe04a53154937348be8407c473",
        "plan_path": "docs/goalzendo/plans/g00f-capability-repair-0p5b.jsonl",
        "plan_sha256": "2b12a2b75c564f8537245c1209d2cc2de38cf390f48f6f83d3bff329479a7998",
        "run_count": 80,
    },
    "g00f-1p5b": {
        "canonical_config_digest": ("aef2e90d5b8f9afccc93d1c93d05750a13d03e02170e7aa83f93c657770ef592"),
        "config_path": "configs/goalzendo/g00f_capability_repair_1p5b.yaml",
        "config_sha256": "d8fe04d5f6ea5041481f13e1c0fd93498520f40c7aa793c3bfe35d2e81bd2dc0",
        "plan_key_digest": "7f9619c3903cce5f94511bbaf4c4de684fc78969a6408d967b36f1edbbe96e6a",
        "plan_path": "docs/goalzendo/plans/g00f-capability-repair-1p5b.jsonl",
        "plan_sha256": "7c2e19f2753b49c42b1ac48f4f32d96cff43f779854304f07e98415f33d1bce0",
        "run_count": 80,
    },
}
WORKER_COUNT = 4
RUNS_PER_PANEL = 80
RUNS_PER_WORKER = 40
PANEL_RUNS_PER_WORKER = 20
WALL_CEILING_SECONDS = 14 * 60 * 60
H200_HOUR_CEILING = 56.0
PROVISION_WATCHDOG_GRACE_SECONDS = 60
RUNPOD_PROVISIONING_CONTRACT: Mapping[str, Any] = {
    "absolute_terminate_after_required": True,
    "cloud_type": "SECURE",
    "container_disk_in_gb": 50,
    "created_at_source_forms": [
        "runpodctl_go_json_utc_with_optional_1_to_9_digit_fraction",
        "rfc3339_utc_with_optional_1_to_9_digit_fraction",
    ],
    "gpu_count": 4,
    "gpu_display_name": "H200 SXM",
    "gpu_id": "NVIDIA H200",
    "gpu_memory_catalog_gb": 141,
    "gpu_catalog_capture_argv": [
        "runpodctl",
        "gpu",
        "list",
        "--include-unavailable",
        "-o",
        "json",
    ],
    "gpu_catalog_capture_file": "runpod-gpu-catalog-response.json",
    "gpu_catalog_capture_must_precede_create": True,
    "gpu_catalog_data_center_stock_rejected_values": ["none"],
    "maximum_secure_cost_usd": 312.12,
    "provision_ceiling_seconds": 17 * 60 * 60,
    "runpodctl_version": "runpodctl 2.9.0-c094cac",
    "runpodctl_version_capture_argv": ["runpodctl", "version"],
    "runpodctl_version_capture_file": "runpodctl-version.txt",
    "secure_price_ceiling_usd_per_gpu_hour": 4.59,
    "ssh": True,
    "terminate_after_source": "externally_chosen_absolute_utc",
    "create_wait": False,
    "readiness_poll_timeout_seconds": 900,
    "readiness_poll_interval_seconds": 5,
    "readiness_requires": [
        "provider_runtime_status_running",
        "runpodctl_ssh_info_same_pod_and_coordinates",
        "tcp_server_banner_prefix_SSH_dash",
        "batchmode_ssh_provider_environment_and_four_gpu_identity",
    ],
    "provider_output_limitations": {
        "network_volume_id": (
            "requested_by_authenticated_create_argv_not_emitted_by_runpodctl_get_"
            "then_cross_checked_by_authenticated_batchmode_ssh_RUNPOD_VOLUME_ID"
        ),
        "terminate_after_utc": "requested_by_authenticated_create_argv_not_emitted_by_runpodctl_get",
    },
    "pre_handoff_failure_cleanup": {
        "command_argv_prefix": ["runpodctl", "pod", "delete"],
        "delete_every_discovered_pod_id": True,
        "exit_trap_required": True,
        "failed_transaction_resume_allowed": False,
        "create_without_wait_then_poll": True,
        "preserve_create_poll_get_ssh_and_delete_responses": True,
    },
}
RUNPOD_OPERATOR_HANDOFF_CONTRACT: Mapping[str, Any] = {
    "schema": "goalzendo.g00f_h200_operator_handoff",
    "schema_version": 1,
    "whole_file_external_sha256_required": True,
    "parser": "jq_exact_key_validation_in_every_ssh_block",
    "stage_scope": "minimal_freeze_archive_manifest_and_authenticated_controllers",
    "manual_pod_fact_redeclaration": False,
    "raw_gpu_catalog_sha256_required": True,
    "catalog_price_and_stock_are_derived_not_operator_supplied": True,
    "execution_uuid_is_independently_pinned_before_launch": True,
    "detached_execution_handoff_required": True,
    "detached_supervisor": (
        "usr_bin_nohup_plus_util_linux_usr_bin_setsid_fork_wait_with_o_excl_log_pid_and_execution_handoff"
    ),
    "detached_supervisor_term_grace_seconds": 90,
    "detached_supervisor_inner_cleanup_minimum_margin_seconds": 45,
    "per_execution_venv": "/workspace/.venvs/goalzendo-h200-<pinned_execution_uuid>",
    "scientific_identity": False,
}
H200_MODULE_INVOCATION_CONTRACT: Mapping[str, Any] = {
    "immutable_h100_parent_pyproject_sha256": (
        "aa6e5117128140343290abbed9bf2a3a0203f2662bd80f031748f2b371392092"
    ),
    "h200_console_script_installed": False,
    "h200_package_data_installed": False,
    "runtime_project_wheel_or_editable_install_performed": False,
    "runtime_venv_install_scope": "exact_external_dependencies_only",
    "invocation": "authenticated_PYTHONPATH_frozen_source_then_python_m_goalzendo_g00f_h200_cli",
    "source_manifest_access": "authenticated_repository_path_not_importlib_package_data",
    "reason": "preserve_byte_exact_canonical_h100_pyproject_and_parent_verification",
}
STORAGE_PREFLIGHT_CONTRACT: Mapping[str, Any] = {
    "mount_root": "/workspace",
    "phases": ["before_qualification", "before_itt"],
    "receipt_names": {
        "before_qualification": "storage-preflight-before-qualification.json",
        "before_itt": "storage-preflight-before-itt.json",
    },
    "qualification_transient_ceiling_bytes": 1024**4,
    "operational_reserve_bytes": 256 * 1024**3,
    "minimum_free_bytes": 1024**4 + 256 * 1024**3,
    "minimum_free_inodes": 1_000_000,
    "same_filesystem_required_for_execution_engineering_and_mount": True,
    "mountpoint_required": True,
    "measurement_source": "python_os_stat_and_statvfs_on_exact_workspace_mount",
    "before_itt_recheck_required": True,
    "outcomes_seen": False,
    "g01_launch_authorized": False,
}
WATCHDOG_SUPERVISION_CONTRACT: Mapping[str, Any] = {
    "clock": "time.monotonic_ns",
    "wall_ceiling_seconds": WALL_CEILING_SECONDS,
    "watchdog_session": "independent_process_group",
    "guardian_session": "independent_process_group",
    "launcher_identity": "linux_pidfd_bound_before_guardian_fork",
    "readiness": "local_fifo_after_guardian_ready_before_any_worker",
    "network_volume_io_before_deadline_signal": False,
    "deadline_marker_lead_ns": 2_000_000_000,
    "term_grace_seconds": 30,
    "post_kill_receipt_margin_seconds": 5,
    "deadline_target": "exact_launcher_and_worker_process_group",
    "launcher_death_before_deadline": "pidfd_readable_fails_closed",
    "watchdog_death_before_deadline": "guardian_control_pipe_eof_fails_closed",
    "timeout_command_group": "watchdog_group_guardian_killable",
    "timeout_command_stdio": "devnull",
    "successful_lifecycle_receipts": [
        "watchdog-started.json",
        "watchdog-normal-stop.json",
        "watchdog-guardian-terminal.json",
    ],
    "g01_launch_authorized": False,
}
PROFILE_CONTRACT: Mapping[str, Any] = {
    "baseline": {
        "train_batch_size": 10,
        "gradient_accumulation_steps": 5,
        "effective_batch_size": 50,
        "gradient_checkpointing": True,
        "evaluation_batch_size": 16,
    },
    "tuned": {
        "train_batch_size": 50,
        "gradient_accumulation_steps": 1,
        "effective_batch_size": 50,
        "gradient_checkpointing": False,
        "evaluation_batch_size": 128,
    },
}
DATA_ORDER_EQUIVALENCE_CONTRACT: Mapping[str, Any] = {
    "algorithm": "goalzendo.training.deterministic_batch_indices",
    "baseline_microsteps_per_update": 5,
    "baseline_train_batch_size": 10,
    "dataset_size": 10_000,
    "digest": "81e2beacf42b75fce89742a17c55615c34f368d6f4f8960478c28231931dbeb0",
    "digest_domain": "goalzendo-g00f-h200-batch-order-v1\\0",
    "digest_encoding": (
        "sorted_panel_baseline_config_filename_nul_int64be_seed_then_"
        "1000_repetitions_of_uint32be_update_and_50_uint32be_indices"
    ),
    "registered_seed_count": 20,
    "tuned_microsteps_per_update": 1,
    "tuned_train_batch_size": 50,
    "updates": 1_000,
}
PROFILE_SELECTOR_CONTRACT: Mapping[str, Any] = {
    "profiles": list(EXECUTION_PROFILES),
    "scope": "pre_itt_runtime_only_no_scientific_outcomes",
    "qualification_branches": {
        "tuned_probe_passed_full": {
            "process_record_count": 64,
            "numeric_comparison_record_count": 4,
            "profiles": ["baseline", "tuned"],
        },
        "tuned_capacity_fallback_baseline": {
            "process_record_count": 32,
            "numeric_comparison_record_count": 0,
            "profiles": ["baseline"],
        },
    },
    "tuned_capacity_probe": {
        "isolated_process_per_gpu_panel_law": True,
        "cell_receipt_count": 16,
        "device_receipt_count": 4,
        "device_receipt_embeds_four_exact_cell_bindings": True,
        "cell_receipt_bindings": [
            "argv_and_exit_status",
            "config",
            "law_family",
            "memory_and_finite_result",
            "model_snapshot",
            "panel_id",
            "raw_log_sha256_and_mode",
            "workload",
        ],
        "fresh_model_and_optimizer": True,
        "tuned_batch_50_backward_clip_adamw_with_moments_resident": True,
        "then_contiguous_128_example_six_view_eval_with_optimizer_resident": True,
        "expected_single_eval_scorer_prompt_count": 768,
        "recoverable_disqualifiers": [
            "cuda_out_of_memory",
            "nonfinite_capacity_execution",
            "reserved_memory_ceiling_exceeded",
        ],
        "process_exit_resets_device": True,
        "all_pass_enters_full_branch": True,
        "any_recoverable_enters_baseline_only_branch": True,
        "later_tuned_execution_failure_is_not_recoverable": True,
        "no_partial_tuned_evidence": True,
    },
    "numeric_comparison_scope": ("baseline_vs_tuned_on_lowest_gpu_uuid_for_each_panel_and_law"),
    "replay_and_cross_gpu_comparison_scope": "digest_exact_process_evidence",
    "numeric_metric_inference": {
        "valid_only_after_exact_cross_gpu_process_equivalence": True,
        "required_exact_fields": [
            "actions",
            "combined_state_hashes",
            "model_state_hashes",
            "native_vector_chunk_manifests",
            "optimizer_state_hashes",
            "ordered_example_ids",
            "output_hashes",
        ],
        "canonical_gpu": "lexicographically_lowest_authenticated_gpu_uuid",
    },
    "qualification_coverage": {
        "production_training_views": [
            "audit_law_matched",
            "herald_only",
            "law_only",
            "no_signal",
            "sage_only",
            "surface_only",
        ],
        "law_families": ["majority", "parity"],
        "worst_case_token_length": True,
        "worst_case_train_probe": ("same_50_ordered_ids_then_replace_final_id_iff_corpus_max_is_absent"),
        "evaluation_workload": ("production_shaped_full_banks_with_multiple_batches_for_eval16_and_eval128"),
        "full_evaluation_timing_representatives": (
            "16_primary_profile_panel_gpu_conservative_both_law_tokenized_batch_envelopes"
        ),
        "other_process_evaluation_scope": (
            "registered_128_examples_via_production_evaluate_batches_up_to_768_prompts"
        ),
        "registered_parity_chunk": {
            "example_count": 128,
            "prompt_views_per_example": 6,
            "tuned_single_partition_prompt_count": 768,
            "states": ["initial", "after_1", "after_2", "after_4", "after_8"],
            "post_update_numeric_states": ["after_1", "after_2", "after_4", "after_8"],
            "after_optimizer_boundary": True,
            "excluded_from_update_timing": True,
        },
        "timing_envelope_proof": (
            "actual_tokenizer_realizes_componentwise_max_example_prompt_and_padded_length_"
            "per_production_bank_boundary_chunk_across_both_laws"
        ),
        "fresh_identical_snapshot_and_optimizer_state_per_process": True,
        "all_four_gpu_uuids": True,
        "gpu_mapping": ("separate_host_ordinal_and_uuid_sorted_logical_worker_with_isolated_uuid_replay"),
        "both_model_revisions": True,
        "production_numerical_configuration_helper_required": True,
        "numerical_configuration": {
            "deterministic_algorithms": True,
            "deterministic_warn_only": False,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "float32_matmul_precision": "highest",
            "tf32": "exact_production_contract",
        },
    },
    "trainable_parameter_contract": {
        "enumeration_api": "named_parameters(remove_duplicate=True)",
        "requires_grad_only": True,
        "optimizer_and_clip_order": "native_scorer_parameters_traversal_matching_production",
        "evidence_manifest_order": "sorted_parameter_keys_separate_from_execution_order",
        "trainable_numel": {
            "g00f-0p5b": 494_032_768,
            "g00f-1p5b": 1_543_714_304,
        },
        "every_vector_covers_exact_manifest": True,
        "native_tensor_digest_fields": ["dtype", "shape", "contiguous_bytes_sha256"],
        "numeric_accumulator_dtype": "float64",
    },
    "profile_replay_requirements": {
        "profiles": list(EXECUTION_PROFILES),
        "fresh_primary_and_replay": True,
        "exact_ordered_example_ids": True,
        "exact_actions": True,
        "exact_model_state_hashes": True,
        "exact_optimizer_state_hashes": True,
        "exact_combined_state_hashes": True,
        "exact_output_hashes": True,
        "exact_vector_native_chunk_manifests": True,
        "fixed_boundary_equivalence_on_all_four_gpu_uuids": True,
    },
    "projection": {
        "maximum_projected_wall_seconds": 12 * 60 * 60,
        "projection_includes": [
            "13_evaluation_boundaries",
            "dataset_and_bank_materialization",
            "evaluation_callback_metrics_predictions_progress_fsync",
            "exact_40_run_worker_mix",
            "final_resumable_model_optimizer_checkpoint_fsync_hash_pointer_prune",
            "model_load",
            "outcome_seal",
            "rendering_and_tokenization",
        ],
        "projection_safety_multiplier": 1.20,
        "projection_aggregation": "componentwise_maximum_per_profile_panel_and_device",
        "profile_independent_components": (
            "shared_cross_profile_conservative_maximum_for_prep_tokenizer_model_load_checkpoint_and_seal"
        ),
        "profile_dependent_timing_order": "counterbalanced_across_sorted_gpu_uuids",
        "timing_warmup": "synchronized_and_excluded",
        "training_timing_corpus": (
            "every_timed_example_repeats_authenticated_global_longest_prompt_over_all_"
            "registered_seeds_views_and_renderers_per_panel"
        ),
        "training_timing_call_shapes": (
            "exact_worst_padded_shape_every_microbatch_and_update_for_both_profiles"
        ),
        "training_host_tokenizer_envelopes": [
            "variability_top_10_distinct_by_total_contextual_token_work_repeated_five_times",
            "variability_top_10_distinct_by_total_utf8_bytes_repeated_five_times",
            "dominance_global_max_contextual_token_work_repeated_fifty_times",
            "dominance_global_max_utf8_bytes_repeated_fifty_times",
        ],
        "training_host_envelope_aggregation": (
            "add_all_four_exact_profile_call_geometry_times_only_to_absolute_wall_projection"
        ),
        "action_continuation_bound": (
            "scan_all_contextual_A_B_counts_require_global_one_token_invariant_rank_full_"
            "prompt_A_B_and_record_actual_prompt_only_fast_path_shape"
        ),
        "optimizer_update_timing": (
            "fresh_timing_only_goalzendo_training_train_steps_eight_updates_with_hooks_"
            "omitted_and_one_pre_post_cuda_synchronization_no_evidence_capture"
        ),
        "whole_update_wall_time_includes": [
            "cpu_prompt_format_and_tokenization",
            "host_device_scalar_synchronization",
            "python_training_loop",
        ],
        "timed_io_envelope": [
            "complete_metrics_jsonl_rows_and_bytes",
            "complete_predictions_jsonl_rows_and_bytes",
            "status_progress_atomic_rewrites",
            "summary_and_completion_and_COMPLETE_receipts",
            "three_outcome_file_hash_scan_and_chmod",
        ],
    },
    "tuned_requirements": {
        "all_models_and_laws": True,
        "compared_optimizer_steps": list(range(1, 9)),
        "data_order_exact": True,
        "data_order_equivalence": DATA_ORDER_EQUIVALENCE_CONTRACT,
        "deterministic_replay_exact": True,
        "dropout_and_stochastic_modules_inactive": True,
        "identical_actions": True,
        "maximum_score_difference": 0.002,
        "maximum_probability_difference": 0.005,
        "minimum_gradient_cosine": 0.99999,
        "maximum_gradient_relative_l2": 0.005,
        "minimum_parameter_update_cosine": 0.99999,
        "maximum_parameter_update_relative_l2": 0.005,
        "optimizer_moments_compared": True,
        "streaming_accumulator_dtype": "float64",
        "explicit_zero_norm_and_parameter_key_order": True,
        "exact_profile_state_optimizer_and_output_hashes": True,
        "maximum_peak_reserved_gib": 120.0,
        "memory_authority": "sixteen_clean_tuned_capacity_cells",
        "evidence_process_memory": "diagnostic_only",
        "record_peak_allocated_gib": True,
        "cuda_synchronize_before_memory_read": True,
        "minimum_throughput_ratio": 1.20,
        "minimum_throughput_ratio_required_on_every_gpu": True,
        "throughput_ratio_gates": [
            "production_shaped_end_to_end_projection_excluding_only_synthetic_host_stress",
            "clean_whole_goalzendo_train_steps_wall",
        ],
        "synthetic_host_stress_affects_throughput_ratio": False,
    },
    "baseline_requirements": {
        "deterministic_replay_exact": True,
        "all_four_gpu_boundary_equivalence_exact": True,
        "maximum_projected_wall_seconds": 12 * 60 * 60,
    },
    "storage_contract": {
        "maximum_persisted_evidence_bytes": 512 * 1024**2,
        "raw_vectors_persisted_after_pair_comparison": False,
        "bounded_per_parameter_accumulator_rows": True,
        "temporary_pairwise_vectors_deleted_before_handoff": True,
        "producer_deletion_complete_before_handoff": True,
        "post_selection_transient_absence_receipt_before_itt": True,
        "post_selection_receipt_performs_deletion": False,
        "compact_evidence_retained_read_only_through_final_gate": True,
    },
    "selection_rule": (
        "probe_recoverable_failure_runs_baseline_only_else_run_full;select_tuned_iff_every_"
        "tuned_requirement_passes_else_select_baseline_iff_every_baseline_replay_device_"
        "and_projection_requirement_passes;later_execution_failure_fails_closed"
    ),
    "no_third_profile": True,
    "no_threshold_relaxation": True,
    "receipt_timing": "o_excl_before_itt_ledger",
}
QUALIFICATION_PRODUCER_CONTRACT: Mapping[str, Any] = {
    "controller": "runs/goalzendo/g00f_h200_qualification_controller.py",
    "implementation": "src/goalzendo_g00f_h200/qualification_producer.py",
    "worker_count": WORKER_COUNT,
    "monotonic_wall_ceiling_seconds": 105 * 60,
    "external_process_group_supervisor": {
        "path": "runs/goalzendo/g00f_h200_qualification_supervisor.py",
        "controller_and_descendants_share_managed_process_group": True,
        "monotonic_ceiling_seconds": 6_300,
        "hard_kill_after_seconds": 30,
        "independent_pipe_guardian": True,
        "guardian_kills_controller_group_if_supervisor_dies_or_deadline_expires": True,
        "o_excl_started_term_kill_terminal_receipts": True,
    },
    "post_qualification_handoff_reserve_seconds": 10 * 60,
    "deadline_policy": "fail_closed_before_selection_and_itt",
    "minimum_provision_remaining_seconds_at_start": (
        105 * 60 + 10 * 60 + WALL_CEILING_SECONDS + PROVISION_WATCHDOG_GRACE_SECONDS
    ),
    "qualification_branches": copy.deepcopy(dict(PROFILE_SELECTOR_CONTRACT["qualification_branches"])),
    "tuned_capacity_probe": copy.deepcopy(dict(PROFILE_SELECTOR_CONTRACT["tuned_capacity_probe"])),
    "replay_and_cross_gpu_checks": "exact_process_digests_without_raw_pair_reruns",
    "temporary_pairwise_vector_ceiling_bytes": 1024**4,
    "persisted_evidence_ceiling_bytes": 512 * 1024**2,
    "raw_vector_retention": "delete_each_pair_before_next_pair_and_all_before_handoff",
    "raw_deletion_completion_timing": "before_handoff_and_profile_selection",
    "handoff": "authenticated_o_excl_copy_into_execution_root",
    "post_selection_absence_receipt_timing": "after_selection_before_itt_ledger",
    "compact_evidence_retention": "read_only_through_final_gate",
    "outcomes_seen": False,
    "g01_launch_authorized": False,
}
MODEL_INTEGRATION_ACTION_LABELS = ("A", "B")
MODEL_INTEGRATION_PROMPTS = (
    "Choose the koan that follows the Law.\nA: one red pyramid.\n"
    "B: one blue wedge.\nReply with exactly A or B.\nAnswer: ",
    "The Law accepts exactly one blue object touching a pyramid. The Herald says A; "
    "the Sage says B.\nA: a blue wedge touches a red pyramid.\n"
    "B: a red wedge is separate from a blue pyramid.\nReply with exactly A or B.\nAnswer: ",
)
MODEL_INTEGRATION_SYSTEM_PROMPT = (
    "Follow the official Law. Reply with exactly one answer label and no explanation."
)
MODEL_INTEGRATION_SWAP_ATOL = 2e-3
MODEL_INTEGRATION_SWAP_RTOL = 2e-3
MODEL_INTEGRATION_PROBABILITY_REPLAY_TOLERANCE = 5e-3
MODEL_INTEGRATION_PROMPTS_DIGEST = hashlib.sha256(
    json.dumps(
        MODEL_INTEGRATION_PROMPTS,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()
RUNPOD_PROVISION_EVIDENCE = {
    "catalog_capture_utc": "operator_wall_clock_not_runpod_api_attested",
    "created_at_utc": "derived_from_pinned_runpodctl_get_createdAt",
    "data_center": "derived_from_pinned_runpodctl_get_machine_dataCenterId",
    "gpu_count": "exact_match_in_pinned_runpodctl_create_and_get",
    "gpu_catalog": "derived_from_externally_pinned_raw_runpodctl_gpu_catalog_bytes",
    "operational_price_and_stock": (
        "derived_from_externally_pinned_raw_runpodctl_gpu_catalog_bytes_not_scientific_identity"
    ),
    "image": "exact_match_in_pinned_runpodctl_create_and_get",
    "network_volume_id": (
        "authenticated_create_argv_request_plus_authenticated_batchmode_ssh_"
        "RUNPOD_VOLUME_ID_runpodctl_get_omits_attachment_id"
    ),
    "network_volume_mount": "exact_match_in_pinned_runpodctl_create_and_get",
    "pod_id": "exact_match_in_pinned_runpodctl_create_get_and_ssh_info",
    "provider_cost": "derived_from_pinned_runpodctl_create_and_get",
    "provider_runtime": "derived_from_pinned_runpodctl_get_runtimeStatus_and_ssh",
    "runpodctl_version": "exact_pinned_raw_stdout_bytes",
    "ssh_identity": "authenticated_batchmode_ssh_allowlisted_provider_environment_and_gpu_receipt",
    "terminate_after_utc": "authenticated_create_argv_request_runpodctl_get_omits_deadline",
}
EXECUTION_AND_GATE_CONTRACT: Mapping[str, Any] = {
    "adapter": {
        "every_informative_run_and_mirror_position": True,
        "minimum_correct": 244,
        "trials": 256,
    },
    "candidate_order": {"maximum_noncomplementing_pairs": 5, "pairs": 256},
    "no_signal": {
        "canonical_prompt_bytes_identical": True,
        "correct_per_run": 256,
        "deterministic_pair_actions_identical": True,
        "trials_per_run": 512,
    },
    "pretraining_model_boundary": {
        "action_labels": list(MODEL_INTEGRATION_ACTION_LABELS),
        "legacy_function": "goalzendo.modeling.run_model_integration_check",
        "max_prompt_tokens": None,
        "panels": ["g00f-0p5b", "g00f-1p5b"],
        "prompt_count": len(MODEL_INTEGRATION_PROMPTS),
        "prompts_digest": MODEL_INTEGRATION_PROMPTS_DIGEST,
        "probability_replay_tolerance": MODEL_INTEGRATION_PROBABILITY_REPLAY_TOLERANCE,
        "strict_revision": True,
        "swap_atol": MODEL_INTEGRATION_SWAP_ATOL,
        "swap_rtol": MODEL_INTEGRATION_SWAP_RTOL,
        "system_prompt_sha256": hashlib.sha256(MODEL_INTEGRATION_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "timing": "after_snapshot_receipts_before_itt_ledger_and_weight_updates",
    },
    "outcome_file_sealing": {
        "files": ["metrics.jsonl", "predictions.jsonl", "summary.json"],
        "sealed_mode": 0,
        "unsealed_mode": 0o400,
    },
    "surface_only_engineering_bands": {
        "per_model": [2_490, 2_630],
        "pooled": [5_021, 5_219],
    },
    "h200_profile_selector": PROFILE_SELECTOR_CONTRACT,
}

FROZEN_RUNTIME_PACKAGES: Mapping[str, str] = {
    "accelerate": "1.14.0",
    "huggingface-hub": "0.36.2",
    "peft": "0.20.0",
    "safetensors": "0.8.0",
    "tokenizers": "0.22.2",
    "torch": "2.8.0",
    "transformers": "4.57.6",
}
FROZEN_MODEL_DEPENDENCIES: Mapping[str, str] = {
    "accelerate": "1.14.0",
    "huggingface_hub": "0.36.2",
    "peft": "0.20.0",
    "python": "3.12.3",
    "safetensors": "0.8.0",
    "tokenizers": "0.22.2",
    "torch": "2.8.0+cu128",
    "torch_cuda": "12.8",
    "transformers": "4.57.6",
}

_SHARED_MODEL_LEAVES: tuple[Mapping[str, Any], ...] = (
    {
        "path": ".gitattributes",
        "bytes": 1_519,
        "sha256": "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361",
    },
    {
        "path": "LICENSE",
        "bytes": 11_343,
        "sha256": "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e",
    },
    {
        "path": "generation_config.json",
        "bytes": 242,
        "sha256": "e558847a8b4402616f1273797b015104dc266fe4b520056fca88823ba8f8ebe6",
    },
    {
        "path": "merges.txt",
        "bytes": 1_671_839,
        "sha256": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
    },
    {
        "path": "tokenizer.json",
        "bytes": 7_031_645,
        "sha256": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    },
    {
        "path": "tokenizer_config.json",
        "bytes": 7_305,
        "sha256": "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
    },
    {
        "path": "vocab.json",
        "bytes": 2_776_833,
        "sha256": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    },
)

MODEL_LEAF_FILES: Mapping[str, tuple[Mapping[str, Any], ...]] = {
    "g00f-0p5b": tuple(
        sorted(
            (
                *_SHARED_MODEL_LEAVES,
                {
                    "path": "README.md",
                    "bytes": 4_917,
                    "sha256": "b19c806a904db6dc878a0462e70b551f6b7ac78dfbb88c2eb966ca2b9109ae15",
                },
                {
                    "path": "config.json",
                    "bytes": 659,
                    "sha256": "18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45",
                },
                {
                    "path": "model.safetensors",
                    "bytes": 988_097_824,
                    "sha256": "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
                },
            ),
            key=lambda row: str(row["path"]),
        )
    ),
    "g00f-1p5b": tuple(
        sorted(
            (
                *_SHARED_MODEL_LEAVES,
                {
                    "path": "README.md",
                    "bytes": 4_917,
                    "sha256": "2e1bcd8bd964728a820be709fa0f7b9dd54817a94fd2254c535df70c5e67fada",
                },
                {
                    "path": "config.json",
                    "bytes": 660,
                    "sha256": "98d2ff8cc47488d08a2b0b3acf4eb99ef210779b42bd48605f6b8e36acdbf670",
                },
                {
                    "path": "model.safetensors",
                    "bytes": 3_087_467_144,
                    "sha256": "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee",
                },
            ),
            key=lambda row: str(row["path"]),
        )
    ),
}

MODEL_RUNTIME_IDENTITIES: Mapping[str, Mapping[str, Any]] = {
    "g00f-0p5b": {
        "model_class": "transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM",
        "parameter_count": 494_032_768,
        "trainable_parameter_count": 494_032_768,
    },
    "g00f-1p5b": {
        "model_class": "transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM",
        "parameter_count": 1_543_714_304,
        "trainable_parameter_count": 1_543_714_304,
    },
}
TOKENIZER_RUNTIME_IDENTITY: Mapping[str, Any] = {
    "action_labels": ["A", "B"],
    "action_token_ids": [[32], [33]],
    "bos_token_id": None,
    "chat_template_sha256": "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f",
    "eos_token_id": 151_645,
    "pad_token_id": 151_643,
    "padding_side": "right",
    "tokenizer_class": ("transformers.models.qwen2.tokenization_qwen2_fast.Qwen2TokenizerFast"),
    "vocabulary_size": 151_643,
}

_PANEL_BASE: Mapping[str, Mapping[str, Any]] = {
    "g00f-0p5b": {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
        "seeds": (10007, 10009, 10037, 10039, 10061, 10067, 10069, 10079, 10091, 10093),
    },
    "g00f-1p5b": {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        "seeds": (10103, 10111, 10133, 10139, 10141, 10151, 10159, 10163, 10169, 10177),
    },
}

PROFILE_CONFIG_SPECS: Mapping[str, Mapping[str, Mapping[str, Any]]] = {
    profile: {
        panel_id: {
            **base,
            "profile": profile,
            "path": f"configs/goalzendo/g00f_h200_{profile}_{panel_id.removeprefix('g00f-')}.yaml",
            "plan_path": (f"docs/goalzendo/plans/g00f-h200-{profile}-{panel_id.removeprefix('g00f-')}.jsonl"),
        }
        for panel_id, base in _PANEL_BASE.items()
    }
    for profile in EXECUTION_PROFILES
}

# The mapping is mutated in place after an authenticated pre-ITT selection so
# modules that imported it retain the selected two-panel view.  Before
# selection it exposes BASELINE only; execution functions reject an unselected
# :class:`VerifiedFreeze`.
CONFIG_SPECS: dict[str, Mapping[str, Any]] = {
    key: dict(value) for key, value in PROFILE_CONFIG_SPECS["baseline"].items()
}

EXPECTED_INFORMATIVE_VIEWS = frozenset({"law_only", "audit_law_matched", "sage_only", "herald_only"})
EXPECTED_ALL_VIEWS = EXPECTED_INFORMATIVE_VIEWS | {"no_signal", "surface_only"}


class FreezeError(RuntimeError):
    """Raised before unauthenticated G00-F execution can occur."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def semantic_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def require_sha256(value: Any, label: str) -> str:
    normalized = str(value)
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise FreezeError(f"{label} must be a lowercase SHA-256 digest")
    return normalized


def _canonical_utc(value: str, label: str) -> datetime:
    if not isinstance(value, str) or len(value) != 20 or not value.endswith("Z") or value[10] != "T":
        raise FreezeError(f"{label} must be an absolute second-resolution UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise FreezeError(f"{label} is not a valid absolute UTC timestamp") from error
    if parsed.tzinfo != timezone.utc or parsed.microsecond != 0:
        raise FreezeError(f"{label} must be an absolute second-resolution UTC timestamp")
    return parsed


def _runpod_created_utc(value: str, label: str) -> datetime:
    """Parse the exact UTC forms emitted by runpodctl while preserving source bytes.

    Runpod's Go JSON encoder has emitted values such as
    ``2026-08-10 11:16:53.034 +0000 UTC``.  Some API surfaces instead use
    RFC3339.  Both forms may carry one through nine fractional digits.  The
    receipt retains the source string; this parser is used only for deadline
    arithmetic.
    """

    if not isinstance(value, str):
        raise FreezeError(f"{label} must be a Runpod UTC timestamp")
    match = re.fullmatch(
        r"(?P<date>\d{4}-\d{2}-\d{2})(?:T| )(?P<clock>\d{2}:\d{2}:\d{2})"
        r"(?:\.(?P<fraction>\d{1,9}))?(?P<zone>Z| \+0000 UTC)",
        value,
    )
    if match is None:
        raise FreezeError(f"{label} must be an exact Runpod UTC timestamp")
    try:
        parsed = datetime.strptime(
            f"{match.group('date')}T{match.group('clock')}",
            "%Y-%m-%dT%H:%M:%S",
        ).replace(tzinfo=timezone.utc)
    except ValueError as error:
        raise FreezeError(f"{label} is not a valid Runpod UTC timestamp") from error
    fraction = match.group("fraction") or ""
    return parsed.replace(microsecond=int((fraction + "000000")[:6]))


def canonical_runpod_create_command(
    *,
    execution_uuid: str,
    terminate_after_utc: str,
) -> tuple[str, ...]:
    """Return the one prospectively allowed four-H200 pod-create argv."""

    _canonical_utc(terminate_after_utc, "Runpod terminate-after")
    try:
        parsed_uuid = uuid.UUID(execution_uuid)
    except ValueError as error:
        raise FreezeError("Runpod execution UUID must be a canonical UUID4") from error
    if parsed_uuid.version != 4 or str(parsed_uuid) != execution_uuid:
        raise FreezeError("Runpod execution UUID must be a canonical UUID4")
    return (
        "runpodctl",
        "pod",
        "create",
        "--compute-type",
        "GPU",
        "--cloud-type",
        "SECURE",
        "--name",
        f"goalzendo-g00f-h200-{execution_uuid}",
        "--image",
        "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
        "--gpu-id",
        "NVIDIA H200",
        "--gpu-count",
        "4",
        "--data-center-ids",
        "US-CA-2",
        "--network-volume-id",
        "9mut3tpzwd",
        "--volume-mount-path",
        "/workspace",
        "--container-disk-in-gb",
        "50",
        "--ssh",
        "--terminate-after",
        terminate_after_utc,
        "-o",
        "json",
    )


def strict_json(path: str | Path, label: str) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise FreezeError(f"{label} does not exist: {target}")

    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise FreezeError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        parsed = json.loads(target.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FreezeError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(parsed, dict):
        raise FreezeError(f"{label} must contain one JSON object")
    return parsed


def _strict_json_array(path: str | Path, label: str) -> list[Any]:
    target = Path(path)
    if not target.is_file():
        raise FreezeError(f"{label} does not exist: {target}")

    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise FreezeError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        parsed = json.loads(target.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FreezeError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(parsed, list):
        raise FreezeError(f"{label} must contain one JSON array")
    return parsed


def verify_runpod_gpu_catalog_snapshot(
    *,
    catalog_path: str | Path,
    operator_capture_utc: str,
    require_bound_copy: bool = False,
) -> dict[str, Any]:
    """Replay the exact pre-create H200 catalog and reject unavailable US-CA-2 stock."""

    direct = Path(catalog_path)
    target = direct.resolve()
    expected_name = str(RUNPOD_PROVISIONING_CONTRACT["gpu_catalog_capture_file"])
    if (
        direct.is_symlink()
        or not target.is_file()
        or target.name != expected_name
        or (require_bound_copy and target.stat().st_nlink != 1)
        or (require_bound_copy and stat.S_IMODE(target.stat().st_mode) != 0o400)
    ):
        raise FreezeError("Runpod GPU catalog must be the direct canonical regular-file capture")
    _canonical_utc(operator_capture_utc, "Runpod GPU catalog operator capture time")
    catalog = _strict_json_array(target, "Runpod GPU catalog response")
    matching_products = [
        row
        for row in catalog
        if isinstance(row, Mapping) and row.get("gpuId") == RUNPOD_PROVISIONING_CONTRACT["gpu_id"]
    ]
    if len(matching_products) != 1:
        raise FreezeError("Runpod catalog must contain exactly one NVIDIA H200 product row")
    product = matching_products[0]
    price = product.get("securePricePerHr")
    if (
        product.get("displayName") != RUNPOD_PROVISIONING_CONTRACT["gpu_display_name"]
        or product.get("memoryInGb") != RUNPOD_PROVISIONING_CONTRACT["gpu_memory_catalog_gb"]
        or product.get("secureCloud") is not True
        or product.get("available") is not True
        or isinstance(price, bool)
        or not isinstance(price, (int, float))
        or not math.isfinite(float(price))
        or float(price) < 0
        or float(price) > float(RUNPOD_PROVISIONING_CONTRACT["secure_price_ceiling_usd_per_gpu_hour"])
    ):
        raise FreezeError("Runpod H200 identity, memory, secure availability, or price changed")
    availability = product.get("dataCenterAvailability")
    if not isinstance(availability, list):
        raise FreezeError("Runpod H200 catalog omits per-data-center availability")
    matching_data_centers = [
        row for row in availability if isinstance(row, Mapping) and row.get("dataCenterId") == "US-CA-2"
    ]
    if len(matching_data_centers) != 1:
        raise FreezeError("Runpod H200 catalog must contain exactly one US-CA-2 availability row")
    stock = matching_data_centers[0].get("stockStatus")
    rejected = {
        str(value).casefold()
        for value in RUNPOD_PROVISIONING_CONTRACT["gpu_catalog_data_center_stock_rejected_values"]
    }
    if not isinstance(stock, str) or not stock or stock.casefold() in rejected:
        raise FreezeError("Runpod H200 US-CA-2 stock is unavailable before pod creation")
    binding = {
        "command_argv": list(RUNPOD_PROVISIONING_CONTRACT["gpu_catalog_capture_argv"]),
        "file_name": expected_name,
        "operator_capture_utc": operator_capture_utc,
        "sha256": sha256_file(target),
    }
    return {
        "path": str(target),
        "binding": binding,
        "gpu_catalog": {
            "display_name": product["displayName"],
            "gpu_id": product["gpuId"],
            "memory_in_gb": product["memoryInGb"],
        },
        "operational_market_snapshot": {
            "observed_at_utc": operator_capture_utc,
            "secure_price_usd_per_gpu_hour": float(price),
            "stock_label": stock,
            "scientific_identity": False,
        },
    }


def verify_runpodctl_version_capture(
    *,
    version_path: str | Path,
    require_bound_copy: bool = False,
) -> dict[str, Any]:
    """Verify the exact CLI build whose JSON surfaces the freeze replays."""

    direct = Path(version_path)
    target = direct.resolve()
    expected_name = str(RUNPOD_PROVISIONING_CONTRACT["runpodctl_version_capture_file"])
    expected_bytes = (str(RUNPOD_PROVISIONING_CONTRACT["runpodctl_version"]) + "\n").encode()
    if (
        direct.is_symlink()
        or not target.is_file()
        or target.name != expected_name
        or target.read_bytes() != expected_bytes
        or (require_bound_copy and target.stat().st_nlink != 1)
        or (require_bound_copy and stat.S_IMODE(target.stat().st_mode) != 0o400)
    ):
        raise FreezeError("Runpod CLI version capture differs from runpodctl 2.9.0-c094cac")
    return {
        "command_argv": list(RUNPOD_PROVISIONING_CONTRACT["runpodctl_version_capture_argv"]),
        "file_name": expected_name,
        "sha256": sha256_file(target),
    }


def _provider_string(payload: Mapping[str, Any], key: str, label: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value or value != value.strip():
        raise FreezeError(f"{label} field {key!r} must be one nonempty exact string")
    return value


def _provider_integer(payload: Mapping[str, Any], key: str, label: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise FreezeError(f"{label} field {key!r} must be one integer")
    return value


def _provider_number(payload: Mapping[str, Any], key: str, label: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise FreezeError(f"{label} field {key!r} must be one finite number")
    return float(value)


def _runpod_ssh_connection(
    payload: Mapping[str, Any],
    *,
    expected_pod_id: str,
    expected_name: str,
    label: str,
) -> dict[str, Any]:
    allowed_keys = {"id", "ip", "name", "port", "ssh_command", "ssh_key"}
    if set(payload) != allowed_keys:
        raise FreezeError(f"{label} has an unexpected key set or is not connectable")
    if payload.get("id") != expected_pod_id or payload.get("name") != expected_name:
        raise FreezeError(f"{label} identifies a different pod")
    ip_text = _provider_string(payload, "ip", label)
    try:
        address = ipaddress.ip_address(ip_text)
    except ValueError as error:
        raise FreezeError(f"{label} has an invalid SSH IP address") from error
    if not address.is_global:
        raise FreezeError(f"{label} SSH IP address is not public")
    port = _provider_integer(payload, "port", label)
    if not 1 <= port <= 65_535:
        raise FreezeError(f"{label} SSH port is outside the TCP range")
    command = _provider_string(payload, "ssh_command", label)
    ssh_key = payload.get("ssh_key")
    if not isinstance(ssh_key, Mapping) or set(ssh_key) != {
        "exists",
        "fingerprint",
        "in_account",
        "path",
        "source",
    }:
        raise FreezeError(f"{label} SSH-key binding is malformed")
    if (
        ssh_key.get("exists") is not True
        or ssh_key.get("in_account") is not True
        or ssh_key.get("source") != "runpodctl doctor"
        or not isinstance(ssh_key.get("fingerprint"), str)
        or not ssh_key.get("fingerprint")
    ):
        raise FreezeError(f"{label} does not bind an available account SSH key")
    try:
        tokens = shlex.split(command)
    except ValueError as error:
        raise FreezeError(f"{label} SSH command is not shell-tokenizable") from error
    if len(tokens) == 4:
        expected_tokens = ["ssh", f"root@{ip_text}", "-p", str(port)]
        key_path: str | None = None
    elif len(tokens) == 6:
        expected_tokens = ["ssh", "-i", tokens[2], f"root@{ip_text}", "-p", str(port)]
        key_path = tokens[2]
        if not key_path.startswith("/") or any(character.isspace() for character in key_path):
            raise FreezeError(f"{label} SSH key path is not one absolute token")
    else:
        raise FreezeError(f"{label} SSH command has an unexpected argument shape")
    if tokens != expected_tokens:
        raise FreezeError(f"{label} SSH command does not match its IP and port")
    if key_path != ssh_key.get("path"):
        raise FreezeError(f"{label} SSH command and key binding disagree")
    return {
        "id": expected_pod_id,
        "ip": ip_text,
        "key_path": key_path,
        "name": expected_name,
        "port": port,
        "ssh_command": command,
    }


def _runpod_create_semantics(
    *,
    create_path: str | Path,
    runtime: Mapping[str, Any],
    catalog_price_per_gpu_hour: float,
) -> dict[str, Any]:
    payload = strict_json(create_path, "raw Runpod no-wait create response")
    if set(payload) != {
        "containerDiskInGb",
        "costPerHr",
        "desiredStatus",
        "env",
        "gpuCount",
        "id",
        "imageName",
        "lastStatusChange",
        "machine",
        "memoryInGb",
        "name",
        "ports",
        "vcpuCount",
        "volumeInGb",
        "volumeMountPath",
    }:
        raise FreezeError("Runpod create response differs from the pinned runpodctl JSON schema")
    pod_id = _provider_string(payload, "id", "Runpod create response")
    if any(character.isspace() for character in pod_id):
        raise FreezeError("Runpod create response pod id must be one token")
    name = _provider_string(payload, "name", "Runpod create response")
    machine = payload.get("machine")
    ports = _provider_string(payload, "ports", "Runpod create response")
    cost = _provider_number(payload, "costPerHr", "Runpod create response")
    maximum_cost = catalog_price_per_gpu_hour * WORKER_COUNT
    if (
        payload.get("imageName") != runtime["image"]
        or payload.get("desiredStatus") != "RUNNING"
        or payload.get("gpuCount") != WORKER_COUNT
        or payload.get("containerDiskInGb") != RUNPOD_PROVISIONING_CONTRACT["container_disk_in_gb"]
        or payload.get("volumeInGb") != 0
        or payload.get("volumeMountPath") != runtime["network_volume_mount"]
        or "22/tcp" not in {part.strip() for part in ports.split(",")}
        or not isinstance(machine, Mapping)
        or set(machine) != {"gpuDisplayName", "location"}
        or machine.get("gpuDisplayName") != RUNPOD_PROVISIONING_CONTRACT["gpu_display_name"]
        or not isinstance(machine.get("location"), str)
        or not machine.get("location")
        or cost <= 0
        or cost > maximum_cost + 1e-12
    ):
        raise FreezeError("Runpod create response differs from the requested four-H200 allocation")
    memory = _provider_integer(payload, "memoryInGb", "Runpod create response")
    vcpus = _provider_integer(payload, "vcpuCount", "Runpod create response")
    if memory <= 0 or vcpus <= 0:
        raise FreezeError("Runpod create response host memory or vCPU count is invalid")
    return {
        "container_disk_in_gb": int(payload["containerDiskInGb"]),
        "cost_per_hour_usd": cost,
        "desired_status": "RUNNING",
        "gpu_count": WORKER_COUNT,
        "gpu_display_name": machine["gpuDisplayName"],
        "host_memory_in_gb": memory,
        "host_vcpu_count": vcpus,
        "image": payload["imageName"],
        "location": machine["location"],
        "name": name,
        "pod_id": pod_id,
        "ports": ports,
        "volume_in_gb": 0,
        "volume_mount_path": payload["volumeMountPath"],
    }


def _runpod_get_semantics(
    *,
    get_path: str | Path,
    runtime: Mapping[str, Any],
    expected_create: Mapping[str, Any],
) -> dict[str, Any]:
    payload = strict_json(get_path, "raw ready Runpod get response")
    mandatory_keys = {
        "containerDiskInGb",
        "createdAt",
        "desiredStatus",
        "gpuCount",
        "id",
        "imageName",
        "machine",
        "name",
        "runtimeStatus",
        "ssh",
        "volumeInGb",
        "volumeMountPath",
    }
    allowed_keys = mandatory_keys | {
        "costPerHr",
        "dockerEntrypoint",
        "dockerStartCmd",
        "env",
        "gpuId",
        "lastStatusChange",
        "memoryInGb",
        "ports",
        "runtime",
        "uptimeSeconds",
        "vcpuCount",
    }
    if not mandatory_keys <= set(payload) or not set(payload) <= allowed_keys:
        raise FreezeError("Runpod get response differs from the pinned runpodctl JSON schema")
    machine = payload.get("machine")
    ssh = payload.get("ssh")
    created = _provider_string(payload, "createdAt", "Runpod get response")
    _runpod_created_utc(created, "Runpod get response creation time")
    cost = _provider_number(payload, "costPerHr", "Runpod get response")
    if (
        payload.get("id") != expected_create["pod_id"]
        or payload.get("name") != expected_create["name"]
        or payload.get("imageName") != runtime["image"]
        or payload.get("desiredStatus") != "RUNNING"
        or payload.get("runtimeStatus") != "running"
        or "runtimeStatusReason" in payload
        or payload.get("gpuCount") != WORKER_COUNT
        or payload.get("containerDiskInGb") != RUNPOD_PROVISIONING_CONTRACT["container_disk_in_gb"]
        or payload.get("volumeInGb") != 0
        or payload.get("volumeMountPath") != runtime["network_volume_mount"]
        or abs(cost - float(expected_create["cost_per_hour_usd"])) > 1e-12
        or not isinstance(machine, Mapping)
        or machine.get("gpuId") != RUNPOD_PROVISIONING_CONTRACT["gpu_id"]
        or machine.get("gpuDisplayName") != RUNPOD_PROVISIONING_CONTRACT["gpu_display_name"]
        or machine.get("dataCenterId") != runtime["data_center"]
        or machine.get("secureCloud") is not True
        or not isinstance(ssh, Mapping)
    ):
        raise FreezeError("Runpod get response does not prove the ready requested allocation")
    connection = _runpod_ssh_connection(
        ssh,
        expected_pod_id=str(expected_create["pod_id"]),
        expected_name=str(expected_create["name"]),
        label="Runpod get SSH block",
    )
    return {
        "container_disk_in_gb": int(payload["containerDiskInGb"]),
        "cost_per_hour_usd": cost,
        "created_at_utc": created,
        "data_center": machine["dataCenterId"],
        "desired_status": "RUNNING",
        "gpu_count": WORKER_COUNT,
        "gpu_id": machine["gpuId"],
        "image": payload["imageName"],
        "name": payload["name"],
        "pod_id": payload["id"],
        "runtime_status": "running",
        "secure_cloud": True,
        "ssh": connection,
        "volume_in_gb": 0,
        "volume_mount_path": payload["volumeMountPath"],
    }


def _runpod_ssh_info_semantics(
    *,
    ssh_info_path: str | Path,
    expected_get: Mapping[str, Any],
) -> dict[str, Any]:
    payload = strict_json(ssh_info_path, "raw Runpod SSH info response")
    return _runpod_ssh_connection(
        payload,
        expected_pod_id=str(expected_get["pod_id"]),
        expected_name=str(expected_get["name"]),
        label="Runpod SSH info response",
    )


def _provider_identity_remote_command() -> str:
    code = (
        "import json,os,subprocess;"
        "keys=('RUNPOD_POD_ID','RUNPOD_DC_ID','RUNPOD_POD_HOSTNAME','RUNPOD_GPU_COUNT',"
        "'RUNPOD_PUBLIC_IP','RUNPOD_TCP_PORT_22','RUNPOD_VOLUME_ID');"
        "gpu=subprocess.run(['nvidia-smi','--query-gpu=name,uuid,memory.total',"
        "'--format=csv,noheader,nounits'],check=True,capture_output=True,text=True);"
        "print(json.dumps({'provider_environment':{key:os.environ.get(key) for key in keys},"
        "'nvidia_smi_lines':gpu.stdout.splitlines()},sort_keys=True,separators=(',',':')))"
    )
    return f"python3 -c {shlex.quote(code)}"


def _provider_identity_ssh_argv(connection: Mapping[str, Any]) -> list[str]:
    key_path = connection.get("key_path")
    if not isinstance(key_path, str):
        raise FreezeError("authenticated SSH identity requires one exact private-key path")
    return [
        "ssh",
        "-i",
        key_path,
        "-p",
        str(connection["port"]),
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=10",
        f"root@{connection['ip']}",
        _provider_identity_remote_command(),
    ]


def _verify_runpod_ssh_identity_receipt(
    *,
    receipt_path: str | Path,
    expected_get_sha256: str,
    expected_ssh_info_sha256: str,
    expected_connection: Mapping[str, Any],
    require_bound_copy: bool,
) -> dict[str, Any]:
    direct = Path(receipt_path)
    target = direct.resolve()
    if (
        direct.is_symlink()
        or not target.is_file()
        or target.name != "runpod-ssh-identity-receipt.json"
        or (require_bound_copy and target.stat().st_nlink != 1)
        or (require_bound_copy and stat.S_IMODE(target.stat().st_mode) != 0o400)
    ):
        raise FreezeError("Runpod SSH identity receipt is not the direct canonical file")
    payload = strict_json(target, "Runpod SSH identity receipt")
    body = {key: value for key, value in payload.items() if key != "receipt_digest"}
    expected_keys = {
        "api_response_sha256",
        "authenticated_ssh",
        "accelerators",
        "banner_prefix",
        "connect_timeout_seconds",
        "g01_launch_authorized",
        "ip",
        "observed_at_utc",
        "outcomes_seen",
        "pod_id",
        "port",
        "probe_implementation",
        "provider_environment",
        "read_timeout_seconds",
        "ready",
        "schema",
        "schema_version",
        "ssh_info_sha256",
    }
    if (
        set(body) != expected_keys
        or payload.get("schema") != RUNPOD_SSH_IDENTITY_RECEIPT_SCHEMA
        or payload.get("schema_version") != RUNPOD_SSH_IDENTITY_RECEIPT_SCHEMA_VERSION
        or payload.get("api_response_sha256") != expected_get_sha256
        or payload.get("ssh_info_sha256") != expected_ssh_info_sha256
        or payload.get("pod_id") != expected_connection["id"]
        or payload.get("ip") != expected_connection["ip"]
        or payload.get("port") != expected_connection["port"]
        or payload.get("banner_prefix") != "SSH-"
        or payload.get("connect_timeout_seconds") != 5
        or payload.get("read_timeout_seconds") != 5
        or payload.get("probe_implementation")
        != "python_socket_banner_then_batchmode_ssh_allowlisted_provider_identity"
        or payload.get("ready") is not True
        or payload.get("outcomes_seen") is not False
        or payload.get("g01_launch_authorized") is not False
        or payload.get("receipt_digest") != semantic_digest(body)
    ):
        raise FreezeError("Runpod SSH identity receipt changed or does not bind the ready pod")
    _canonical_utc(str(payload.get("observed_at_utc", "")), "Runpod SSH identity observation time")
    authenticated_ssh = payload.get("authenticated_ssh")
    if (
        not isinstance(authenticated_ssh, Mapping)
        or set(authenticated_ssh) != {"argv", "exit_status", "stderr_sha256", "stdout_sha256"}
        or authenticated_ssh.get("argv") != _provider_identity_ssh_argv(expected_connection)
        or authenticated_ssh.get("exit_status") != 0
    ):
        raise FreezeError("Runpod authenticated SSH execution binding changed")
    require_sha256(authenticated_ssh.get("stdout_sha256"), "Runpod SSH identity stdout SHA-256")
    require_sha256(authenticated_ssh.get("stderr_sha256"), "Runpod SSH identity stderr SHA-256")
    provider_environment = payload.get("provider_environment")
    expected_provider_keys = {
        "RUNPOD_DC_ID",
        "RUNPOD_GPU_COUNT",
        "RUNPOD_POD_HOSTNAME",
        "RUNPOD_POD_ID",
        "RUNPOD_PUBLIC_IP",
        "RUNPOD_TCP_PORT_22",
        "RUNPOD_VOLUME_ID",
    }
    if (
        not isinstance(provider_environment, Mapping)
        or set(provider_environment) != expected_provider_keys
        or provider_environment.get("RUNPOD_POD_ID") != expected_connection["id"]
        or provider_environment.get("RUNPOD_DC_ID") != "US-CA-2"
        or provider_environment.get("RUNPOD_GPU_COUNT") != "4"
        or provider_environment.get("RUNPOD_VOLUME_ID") != "9mut3tpzwd"
        or provider_environment.get("RUNPOD_PUBLIC_IP") != expected_connection["ip"]
        or provider_environment.get("RUNPOD_TCP_PORT_22") != str(expected_connection["port"])
        or not isinstance(provider_environment.get("RUNPOD_POD_HOSTNAME"), str)
        or not provider_environment.get("RUNPOD_POD_HOSTNAME")
    ):
        raise FreezeError("Runpod provider-injected environment does not match the pinned pod")
    accelerators = payload.get("accelerators")
    if not isinstance(accelerators, list) or len(accelerators) != WORKER_COUNT:
        raise FreezeError("Runpod SSH identity receipt must contain exactly four accelerators")
    uuids: set[str] = set()
    for index, row in enumerate(accelerators):
        if (
            not isinstance(row, Mapping)
            or set(row) != {"host_ordinal", "memory_total_mib", "name", "uuid"}
            or row.get("host_ordinal") != index
            or not isinstance(row.get("name"), str)
            or not str(row["name"]).startswith("NVIDIA H200")
            or not isinstance(row.get("uuid"), str)
            or not str(row["uuid"]).startswith("GPU-")
            or isinstance(row.get("memory_total_mib"), bool)
            or not isinstance(row.get("memory_total_mib"), int)
            or not 140_000 <= int(row["memory_total_mib"]) <= 150_000
        ):
            raise FreezeError("Runpod SSH identity accelerator inventory changed")
        uuids.add(str(row["uuid"]))
    if len(uuids) != WORKER_COUNT:
        raise FreezeError("Runpod SSH identity accelerator UUIDs are not unique")
    return {
        "file_name": target.name,
        "sha256": sha256_file(target),
        "receipt_digest": payload["receipt_digest"],
    }


def create_runpod_ssh_identity_receipt(
    *,
    raw_api_response_path: str | Path,
    raw_ssh_info_path: str | Path,
    expected_pod_id: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Probe SSH, authenticate noninteractively, and bind provider identity only."""

    get_payload = strict_json(raw_api_response_path, "candidate Runpod get response")
    name = _provider_string(get_payload, "name", "candidate Runpod get response")
    if get_payload.get("id") != expected_pod_id or get_payload.get("runtimeStatus") != "running":
        raise FreezeError("candidate Runpod get response is not the expected running pod")
    ssh_payload = strict_json(raw_ssh_info_path, "candidate Runpod SSH info response")
    connection = _runpod_ssh_connection(
        ssh_payload,
        expected_pod_id=expected_pod_id,
        expected_name=name,
        label="candidate Runpod SSH info response",
    )
    get_ssh = get_payload.get("ssh")
    if (
        not isinstance(get_ssh, Mapping)
        or _runpod_ssh_connection(
            get_ssh,
            expected_pod_id=expected_pod_id,
            expected_name=name,
            label="candidate Runpod get SSH block",
        )
        != connection
    ):
        raise FreezeError("Runpod get and SSH-info coordinates differ")
    try:
        with socket.create_connection((str(connection["ip"]), int(connection["port"])), timeout=5) as stream:
            stream.settimeout(5)
            banner = b""
            while len(banner) < 4:
                chunk = stream.recv(4 - len(banner))
                if not chunk:
                    break
                banner += chunk
    except OSError as error:
        raise FreezeError("Runpod public SSH endpoint did not answer") from error
    if banner != b"SSH-":
        raise FreezeError("Runpod public endpoint did not emit an SSH server banner")
    ssh_argv = _provider_identity_ssh_argv(connection)
    try:
        completed = subprocess.run(
            ssh_argv,
            check=False,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise FreezeError("Runpod authenticated SSH identity command did not complete") from error
    if completed.returncode != 0:
        raise FreezeError("Runpod authenticated SSH identity command failed")
    try:
        remote_payload = json.loads(completed.stdout.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise FreezeError("Runpod authenticated SSH identity stdout is not JSON") from error
    if not isinstance(remote_payload, Mapping) or set(remote_payload) != {
        "nvidia_smi_lines",
        "provider_environment",
    }:
        raise FreezeError("Runpod authenticated SSH identity stdout has an unexpected schema")
    raw_lines = remote_payload.get("nvidia_smi_lines")
    if not isinstance(raw_lines, list) or len(raw_lines) != WORKER_COUNT:
        raise FreezeError("Runpod authenticated SSH identity did not report four GPUs")
    accelerators: list[dict[str, Any]] = []
    for index, raw_line in enumerate(raw_lines):
        if not isinstance(raw_line, str):
            raise FreezeError("Runpod nvidia-smi identity line is not text")
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) != 3:
            raise FreezeError("Runpod nvidia-smi identity line has an unexpected shape")
        try:
            memory_total_mib = int(parts[2])
        except ValueError as error:
            raise FreezeError("Runpod nvidia-smi memory value is not an integer") from error
        accelerators.append(
            {
                "host_ordinal": index,
                "memory_total_mib": memory_total_mib,
                "name": parts[0],
                "uuid": parts[1],
            }
        )
    observed = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    body = {
        "schema": RUNPOD_SSH_IDENTITY_RECEIPT_SCHEMA,
        "schema_version": RUNPOD_SSH_IDENTITY_RECEIPT_SCHEMA_VERSION,
        "api_response_sha256": sha256_file(raw_api_response_path),
        "ssh_info_sha256": sha256_file(raw_ssh_info_path),
        "pod_id": expected_pod_id,
        "ip": connection["ip"],
        "port": connection["port"],
        "banner_prefix": "SSH-",
        "connect_timeout_seconds": 5,
        "read_timeout_seconds": 5,
        "probe_implementation": "python_socket_banner_then_batchmode_ssh_allowlisted_provider_identity",
        "authenticated_ssh": {
            "argv": ssh_argv,
            "exit_status": completed.returncode,
            "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
            "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
        },
        "provider_environment": remote_payload["provider_environment"],
        "accelerators": accelerators,
        "observed_at_utc": observed,
        "ready": True,
        "outcomes_seen": False,
        "g01_launch_authorized": False,
    }
    payload = {**body, "receipt_digest": semantic_digest(body)}
    exclusive_json(output_path, payload)
    return _verify_runpod_ssh_identity_receipt(
        receipt_path=output_path,
        expected_get_sha256=body["api_response_sha256"],
        expected_ssh_info_sha256=body["ssh_info_sha256"],
        expected_connection=connection,
        require_bound_copy=False,
    )


def atomic_json(path: str | Path, value: Mapping[str, Any], *, overwrite: bool = False) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FreezeError(f"refusing to overwrite existing artifact: {target}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def exclusive_json(path: str | Path, value: Mapping[str, Any]) -> None:
    """Create one append-only receipt with O_EXCL and a durable directory entry."""

    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise FreezeError(f"append-only receipt already exists: {target}") from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        directory_descriptor = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        # A partially written exclusive receipt is itself durable evidence of
        # an interrupted attempt.  Never unlink it or silently retry.
        raise


def exclusive_copy(source: str | Path, destination: str | Path, *, mode: int = 0o400) -> None:
    """Copy authenticated bytes once without an overwrite or partial-file retry."""

    origin = Path(source).resolve()
    target = Path(destination).resolve()
    if not origin.is_file() or origin.is_symlink() or target == origin or mode not in {0o400, 0o600}:
        raise FreezeError("exclusive evidence copy has an invalid source, target, or mode")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    except FileExistsError as error:
        raise FreezeError(f"append-only evidence copy already exists: {target}") from error
    try:
        with origin.open("rb") as input_handle, os.fdopen(descriptor, "wb") as output_handle:
            shutil.copyfileobj(input_handle, output_handle, length=8 * 1024 * 1024)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        os.chmod(target, mode)
        directory_descriptor = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        # Like receipts, an interrupted exclusive copy is durable failure
        # evidence and is never removed or overwritten by this workflow.
        raise


def _repo_for_imported_core(repo: str | Path) -> Path:
    resolved = Path(repo).resolve()
    import goalzendo

    imported = Path(str(goalzendo.__file__)).resolve().parent
    expected = (resolved / "src" / "goalzendo").resolve()
    if imported != expected:
        raise FreezeError("requested repo is not the imported frozen GoalZendo source")
    return resolved


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FreezeError(f"{label} does not exist: {path}")
    payload = path.read_bytes()
    if payload and not payload.endswith(b"\n"):
        raise FreezeError(f"{label} lacks a final newline")
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(payload.splitlines(), start=1):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise FreezeError(f"{label} row {index} is invalid JSON") from error
        if not isinstance(value, dict):
            raise FreezeError(f"{label} row {index} is not an object")
        rows.append(value)
    return rows


def config_source_chain(repo: str | Path, leaf_relative: str) -> list[dict[str, Any]]:
    """Return the exact recursive YAML inheritance closure, parent first."""

    resolved = Path(repo).resolve()
    visiting: set[Path] = set()
    ordered: list[Path] = []

    def visit(path: Path) -> None:
        target = path.resolve()
        if target in visiting:
            raise FreezeError("G00-F config inheritance contains a cycle")
        if target in ordered:
            return
        if not target.is_file() or resolved not in target.parents:
            raise FreezeError("G00-F config inheritance leaves the repository")
        visiting.add(target)
        try:
            value = yaml.safe_load(target.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError) as error:
            raise FreezeError("G00-F inherited config is unreadable") from error
        if not isinstance(value, Mapping):
            raise FreezeError("G00-F inherited config is not a mapping")
        raw = value.get("extends")
        parents: Sequence[Any]
        if raw is None:
            parents = ()
        elif isinstance(raw, str):
            parents = (raw,)
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            parents = raw
        else:
            raise FreezeError("G00-F config extends field is malformed")
        for parent in parents:
            if not isinstance(parent, str):
                raise FreezeError("G00-F config parent path is not a string")
            visit(target.parent / parent)
        visiting.remove(target)
        ordered.append(target)

    visit(resolved / leaf_relative)
    return [
        {
            "path": path.relative_to(resolved).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in ordered
    ]


def _plan_schedule_rows(
    repo: Path,
    panel_id: str,
    config: Mapping[str, Any],
    specification: Mapping[str, Any],
) -> list[dict[str, Any]]:
    specs = list(build_plan(config))
    if len(specs) != RUNS_PER_PANEL:
        raise FreezeError(f"{panel_id} does not expand to exactly {RUNS_PER_PANEL} runs")
    ordered = sorted(specs, key=lambda item: item.global_index)
    panel_rank = {spec.plan_key: index for index, spec in enumerate(ordered)}
    seed_rank = {seed: index for index, seed in enumerate(sorted({spec.seed for spec in specs}))}
    worker_lists: dict[int, list[RunSpec]] = {index: [] for index in range(WORKER_COUNT)}
    for spec in ordered:
        worker_index = (seed_rank[spec.seed] + spec.cell_index) % WORKER_COUNT
        worker_lists[worker_index].append(spec)

    # Worker order is computed after both panels are available.  This helper
    # records the exact balanced assignment and panel-local order.
    result: list[dict[str, Any]] = []
    model = specification
    for worker_index, worker_specs in worker_lists.items():
        if len(worker_specs) != PANEL_RUNS_PER_WORKER:
            raise FreezeError("balanced worker assignment did not produce 20 panel runs")
        for panel_order, spec in enumerate(worker_specs):
            store = RunStore(get_path(spec.config, "run.output_root"), spec.config, spec.seed, repo)
            result.append(
                {
                    "panel_id": panel_id,
                    "panel_rank": panel_rank[spec.plan_key],
                    "panel_order_on_worker": panel_order,
                    "worker_index": worker_index,
                    "global_index": spec.global_index,
                    "cell_index": spec.cell_index,
                    "cell_id": spec.cell_id,
                    "plan_key": spec.plan_key,
                    "run_id": store.run_id,
                    "artifact_path": str(store.path),
                    "seed": spec.seed,
                    "derived_seeds": dict(spec.seeds),
                    "law_family": str(get_path(spec.config, "data.rule_family")),
                    "training_view": str(get_path(spec.config, "data.training_view")),
                    "prompt_views": list(get_path(spec.config, "evaluation.prompt_views")),
                    "model_name": str(model["model"]),
                    "model_revision": str(model["revision"]),
                    "canonical_resolved_cell_digest": stable_hash(canonical_config(spec.config), 64),
                }
            )
    return sorted(result, key=lambda row: int(row["global_index"]))


def _merge_worker_schedule(rows_by_panel: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for worker_index in range(WORKER_COUNT):
        by_panel = {
            panel_id: sorted(
                (dict(row) for row in rows if int(row["worker_index"]) == worker_index),
                key=lambda row: int(row["panel_order_on_worker"]),
            )
            for panel_id, rows in rows_by_panel.items()
        }
        if any(len(rows) != PANEL_RUNS_PER_WORKER for rows in by_panel.values()):
            raise FreezeError("each worker must receive exactly 20 runs from each model panel")
        first = "g00f-0p5b" if worker_index % 2 == 0 else "g00f-1p5b"
        second = "g00f-1p5b" if first == "g00f-0p5b" else "g00f-0p5b"
        worker_order = 0
        for panel_order in range(PANEL_RUNS_PER_WORKER):
            for panel_id in (first, second):
                row = dict(by_panel[panel_id][panel_order])
                row["worker_order"] = worker_order
                merged.append(row)
                worker_order += 1
    return sorted(merged, key=lambda row: (int(row["worker_index"]), int(row["worker_order"])))


def expected_plan_rows(
    repo: str | Path,
    *,
    profile: str = "baseline",
) -> dict[str, list[dict[str, Any]]]:
    resolved = Path(repo).resolve()
    if profile not in PROFILE_CONFIG_SPECS:
        raise FreezeError(f"unknown H200 execution profile: {profile}")
    specifications = PROFILE_CONFIG_SPECS[profile]
    panel_rows: dict[str, list[dict[str, Any]]] = {}
    for panel_id, spec in specifications.items():
        config = load_config(resolved / str(spec["path"]))
        panel_rows[panel_id] = _plan_schedule_rows(
            resolved,
            panel_id,
            config,
            spec,
        )
    merged = _merge_worker_schedule(panel_rows)
    by_key = {str(row["plan_key"]): row for row in merged}
    return {
        panel_id: [copy.deepcopy(by_key[str(row["plan_key"])]) for row in rows]
        for panel_id, rows in panel_rows.items()
    }


def write_plan_files(repo: str | Path) -> dict[str, dict[str, Any]]:
    """Generate all four prospective profile/panel plans deterministically."""

    resolved = Path(repo).resolve()
    result: dict[str, dict[str, Any]] = {}
    for profile in EXECUTION_PROFILES:
        generated = expected_plan_rows(resolved, profile=profile)
        for panel_id, rows in generated.items():
            target = resolved / str(PROFILE_CONFIG_SPECS[profile][panel_id]["plan_path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            text = "".join(canonical_json_bytes(row).decode("ascii") + "\n" for row in rows)
            payload = text.encode("ascii")
            if target.exists():
                if target.read_bytes() != payload:
                    raise FreezeError(f"existing H200 plan differs from regenerated bytes: {target}")
            else:
                descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
            result[f"{profile}:{panel_id}"] = {
                "profile": profile,
                "panel_id": panel_id,
                "path": str(target),
                "rows": len(rows),
                "sha256": sha256_file(target),
                "plan_key_digest": semantic_digest(sorted(str(row["plan_key"]) for row in rows)),
            }
    return result


@dataclass(frozen=True)
class VerifiedFreeze:
    repo: Path
    path: Path
    file_sha256: str
    payload: Mapping[str, Any]
    plans: Mapping[str, tuple[Mapping[str, Any], ...]]
    candidate_plans: Mapping[
        str,
        Mapping[str, tuple[Mapping[str, Any], ...]],
    ]
    selected_profile: str | None = None
    selection_receipt: Mapping[str, Any] | None = None

    @property
    def digest(self) -> str:
        return str(self.payload["freeze_digest"])

    @property
    def all_rows(self) -> tuple[Mapping[str, Any], ...]:
        rows = [row for panel in self.plans.values() for row in panel]
        return tuple(sorted(rows, key=lambda row: (int(row["worker_index"]), int(row["worker_order"]))))

    def worker_rows(self, worker_index: int) -> tuple[Mapping[str, Any], ...]:
        if self.selected_profile is None:
            raise FreezeError("H200 execution profile must be selected before worker access")
        if isinstance(worker_index, bool) or not 0 <= int(worker_index) < WORKER_COUNT:
            raise FreezeError("worker index must lie in [0, 4)")
        rows = [row for row in self.all_rows if int(row["worker_index"]) == int(worker_index)]
        rows.sort(key=lambda row: int(row["worker_order"]))
        if len(rows) != RUNS_PER_WORKER:
            raise FreezeError("frozen worker does not contain exactly 40 runs")
        return tuple(rows)


def _activate_profile(profile: str) -> None:
    if profile not in PROFILE_CONFIG_SPECS:
        raise FreezeError(f"unknown H200 execution profile: {profile}")
    CONFIG_SPECS.clear()
    CONFIG_SPECS.update({key: dict(value) for key, value in PROFILE_CONFIG_SPECS[profile].items()})


def _require_selected(verified: VerifiedFreeze) -> str:
    profile = verified.selected_profile
    if profile not in EXECUTION_PROFILES or verified.selection_receipt is None:
        raise FreezeError("authenticated H200 profile selection is required before ITT")
    _activate_profile(profile)
    return profile


def _model_binding(verified: VerifiedFreeze, panel_id: str) -> Mapping[str, Any]:
    models = verified.payload.get("models")
    if panel_id not in _PANEL_BASE or not isinstance(models, Mapping):
        raise FreezeError("unknown G00-F H200 model panel")
    binding = models.get(panel_id)
    if not isinstance(binding, Mapping) or not isinstance(binding.get("model_snapshot"), Mapping):
        raise FreezeError("freeze lacks H200 model snapshot binding")
    return binding


def _verify_additive_source_manifest(repo: Path, freeze: Mapping[str, Any]) -> None:
    binding = freeze.get("additive_source")
    if not isinstance(binding, Mapping):
        raise FreezeError("freeze omits additive source binding")
    relative = str(binding.get("manifest_path", ""))
    target = repo / relative
    expected_file = require_sha256(binding.get("manifest_sha256"), "source manifest SHA-256")
    if sha256_file(target) != expected_file:
        raise FreezeError("additive source manifest bytes changed")
    manifest = strict_json(target, "G00-F additive source manifest")
    body = {key: value for key, value in manifest.items() if key != "manifest_digest"}
    if (
        manifest.get("schema") != SOURCE_MANIFEST_SCHEMA
        or manifest.get("schema_version") != SOURCE_MANIFEST_SCHEMA_VERSION
        or manifest.get("manifest_digest") != semantic_digest(body)
    ):
        raise FreezeError("additive source manifest schema/digest mismatch")
    files = manifest.get("source_files")
    if not isinstance(files, Mapping) or set(files) != {
        "__init__.py",
        "cli.py",
        "evaluator.py",
        "freeze.py",
        "py.typed",
        "qualification.py",
        "qualification_producer.py",
    }:
        raise FreezeError("additive source manifest file set changed")
    package = repo / "src" / "goalzendo_g00f_h200"
    for name, digest in sorted(files.items()):
        source = package / str(name)
        if sha256_file(source) != require_sha256(digest, f"source digest for {name}"):
            raise FreezeError(f"additive source bytes changed: {name}")
    if binding.get("source_digest") != manifest.get("source_digest"):
        raise FreezeError("freeze/additive source aggregate digest mismatch")


def _verify_file_bindings(repo: Path, bindings: Any, label: str) -> None:
    if not isinstance(bindings, Sequence) or isinstance(bindings, (str, bytes)) or not bindings:
        raise FreezeError(f"freeze {label} bindings are absent")
    seen: set[str] = set()
    for raw in bindings:
        if not isinstance(raw, Mapping):
            raise FreezeError(f"freeze {label} binding is not an object")
        relative = str(raw.get("path", ""))
        if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts or relative in seen:
            raise FreezeError(f"freeze {label} path is unsafe or duplicated")
        seen.add(relative)
        target = repo / relative
        if not target.is_file() or sha256_file(target) != require_sha256(
            raw.get("sha256"), f"{label} digest for {relative}"
        ):
            raise FreezeError(f"freeze-bound {label} file changed: {relative}")


def historical_h100_parent_binding(repo: str | Path) -> dict[str, Any]:
    """Reconstruct the immutable, never-run H100 parent without mutable sources."""

    resolved = Path(repo).resolve()
    files = [
        {"path": relative, "sha256": digest}
        for relative, digest in sorted(HISTORICAL_H100_PARENT_FILES.items())
    ]
    for row in files:
        target = resolved / row["path"]
        if not target.is_file() or target.is_symlink() or sha256_file(target) != row["sha256"]:
            raise FreezeError(f"immutable H100 parent artifact changed: {row['path']}")
    parent_path = resolved / (
        "reproducibility/goalzendo/g00f-execution-freeze-20260811/execution-freeze.json"
    )
    parent = strict_json(parent_path, "historical H100 execution freeze")
    parent_body = {key: value for key, value in parent.items() if key != "freeze_digest"}
    panels = parent.get("configurations")
    projected: dict[str, Any] = {}
    if not isinstance(panels, Mapping) or set(panels) != set(HISTORICAL_H100_PANEL_IDENTITIES):
        raise FreezeError("historical H100 parent panel set changed")
    for panel_id, expected in HISTORICAL_H100_PANEL_IDENTITIES.items():
        panel_row = panels.get(panel_id)
        if not isinstance(panel_row, Mapping):
            raise FreezeError("historical H100 parent configuration is malformed")
        projected[panel_id] = {key: panel_row.get(key) for key in expected}
        if projected[panel_id] != expected:
            raise FreezeError(f"historical H100 {panel_id} config/plan identity changed")
    if (
        parent.get("schema") != "goalzendo.g00f_execution_freeze"
        or parent.get("schema_version") != 1
        or parent.get("study_id") != "g00f"
        or parent.get("outcomes_seen") is not False
        or parent.get("freeze_digest") != HISTORICAL_H100_FREEZE_DIGEST
        or parent.get("freeze_digest") != semantic_digest(parent_body)
    ):
        raise FreezeError("historical H100 parent freeze schema/digest changed")
    return {
        "role": "immutable_never_run_hardware_parent",
        "scientific_semantics_inherited": True,
        "hardware_execution_identity_inherited": False,
        "outcomes_seen": False,
        "freeze_digest": HISTORICAL_H100_FREEZE_DIGEST,
        "files": files,
        "panel_config_plan_identities": copy.deepcopy(
            {key: dict(value) for key, value in HISTORICAL_H100_PANEL_IDENTITIES.items()}
        ),
    }


def _verify_worker_schedule(
    plans: Mapping[str, tuple[Mapping[str, Any], ...]],
    *,
    profile: str,
) -> None:
    all_rows = [row for rows in plans.values() for row in rows]
    if len(all_rows) != 160 or len({str(row["plan_key"]) for row in all_rows}) != 160:
        raise FreezeError(f"G00-F H200 {profile} plan union is not exactly 160 unique run keys")
    for worker_index in range(WORKER_COUNT):
        worker = sorted(
            (row for row in all_rows if int(row["worker_index"]) == worker_index),
            key=lambda row: int(row["worker_order"]),
        )
        counts = {panel_id: sum(row["panel_id"] == panel_id for row in worker) for panel_id in _PANEL_BASE}
        if (
            len(worker) != RUNS_PER_WORKER
            or counts != {panel_id: PANEL_RUNS_PER_WORKER for panel_id in _PANEL_BASE}
            or [int(row["worker_order"]) for row in worker] != list(range(RUNS_PER_WORKER))
            or any(worker[index]["panel_id"] == worker[index + 1]["panel_id"] for index in range(39))
        ):
            raise FreezeError(
                f"G00-F H200 {profile} worker schedule is not exact, mixed, alternating, and balanced"
            )
        for panel_id in _PANEL_BASE:
            panel_worker = [row for row in worker if row["panel_id"] == panel_id]
            seed_counts: dict[int, int] = {}
            case_counts: dict[int, int] = {}
            for row in panel_worker:
                seed = int(row["seed"])
                case = int(row["cell_index"])
                seed_counts[seed] = seed_counts.get(seed, 0) + 1
                case_counts[case] = case_counts.get(case, 0) + 1
            if set(seed_counts.values()) != {2} or not set(case_counts.values()) <= {2, 3}:
                raise FreezeError(f"G00-F H200 {profile} Latin assignment lost seed/case balance")


def verify_freeze(
    *,
    repo: str | Path,
    freeze_path: str | Path,
    expected_freeze_sha256: str,
) -> VerifiedFreeze:
    """Authenticate every prospective G00-F identity before any model load."""

    _activate_profile("baseline")

    resolved = _repo_for_imported_core(repo)
    target = Path(freeze_path).resolve()
    expected = require_sha256(expected_freeze_sha256, "externally expected freeze SHA-256")
    observed = sha256_file(target)
    if observed != expected:
        raise FreezeError("G00-F freeze bytes differ from the externally expected SHA-256")
    payload = strict_json(target, "G00-F execution freeze")
    body = {key: value for key, value in payload.items() if key != "freeze_digest"}
    if (
        payload.get("schema") != FREEZE_SCHEMA
        or payload.get("schema_version") != FREEZE_SCHEMA_VERSION
        or payload.get("study_id") != "g00f"
        or payload.get("freeze_digest") != semantic_digest(body)
    ):
        raise FreezeError("G00-F freeze schema or semantic digest mismatch")
    if payload.get("outcomes_seen") is not False:
        raise FreezeError("G00-F freeze is not prospectively outcome-blind")
    authorization = payload.get("authorization")
    if not isinstance(authorization, Mapping) or authorization != {
        "g00f_exact_execution_authorized": True,
        "g01_launch_authorized": False,
        "scope": "exact_g00f_frozen_worker_schedule_only",
    }:
        raise FreezeError("G00-F prospective authorization scope changed")

    provenance = implementation_provenance(resolved)
    if provenance.get("implementation_fingerprint") != FROZEN_GOALZENDO_IMPLEMENTATION_FINGERPRINT:
        raise FreezeError("imported GoalZendo source is not the frozen G00-D implementation")
    legacy = payload.get("legacy_goalzendo")
    if (
        not isinstance(legacy, Mapping)
        or legacy.get("implementation_fingerprint") != FROZEN_GOALZENDO_IMPLEMENTATION_FINGERPRINT
    ):
        raise FreezeError("freeze does not bind the frozen G00-D implementation")

    if payload.get("historical_h100_parent") != historical_h100_parent_binding(resolved):
        raise FreezeError("G00-F H200 freeze does not bind the immutable H100 parent")

    _verify_additive_source_manifest(resolved, payload)
    _verify_file_bindings(resolved, payload.get("prior_evidence"), "prior evidence")
    _verify_file_bindings(resolved, payload.get("runtime_files"), "runtime")
    controllers = payload.get("controller_files")
    expected_controller_paths = {
        "bootstrap": "runs/goalzendo/g00f_h200_bundle_bootstrap.py",
        "detached_supervisor": "runs/goalzendo/g00f_h200_detached_supervisor.py",
        "launcher": "runs/goalzendo/run_g00f_frozen_4h200.sh",
        "qualification_controller": "runs/goalzendo/g00f_h200_qualification_controller.py",
        "qualification_supervisor": "runs/goalzendo/g00f_h200_qualification_supervisor.py",
        "watchdog": "runs/goalzendo/g00f_h200_watchdog.py",
    }
    if not isinstance(controllers, Mapping) or set(controllers) != set(expected_controller_paths):
        raise FreezeError("freeze controller-file role set changed")
    for role, relative in expected_controller_paths.items():
        binding = controllers.get(role)
        if (
            not isinstance(binding, Mapping)
            or binding.get("path") != relative
            or binding.get("sha256") != sha256_file(resolved / relative)
        ):
            raise FreezeError(f"freeze {role} controller binding changed")
    source_bundle = payload.get("source_bundle")
    if (
        not isinstance(source_bundle, Mapping)
        or source_bundle.get("archive_path") != f"{H200_FREEZE_OUTPUT_RELATIVE}/{H200_SOURCE_ARCHIVE_NAME}"
        or source_bundle.get("manifest_path")
        != f"{H200_FREEZE_OUTPUT_RELATIVE}/{H200_SOURCE_BUNDLE_MANIFEST_NAME}"
    ):
        raise FreezeError("freeze source-bundle locations changed")
    require_sha256(source_bundle.get("archive_sha256"), "source bundle archive SHA-256")
    require_sha256(source_bundle.get("manifest_sha256"), "source bundle manifest SHA-256")
    require_sha256(source_bundle.get("manifest_digest"), "source bundle manifest digest")

    runtime = payload.get("runtime")
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("worker_count") != WORKER_COUNT
        or runtime.get("concurrent_runs_per_gpu") != 1
        or runtime.get("wall_ceiling_seconds") != WALL_CEILING_SECONDS
        or float(runtime.get("h200_hour_ceiling", -1)) != H200_HOUR_CEILING
        or runtime.get("selected_profile_projection_source")
        != "authenticated_pre_itt_h200_profile_qualification"
        or runtime.get("maximum_selected_profile_projection_seconds") != 12 * 60 * 60
        or runtime.get("projection_safety_multiplier")
        != PROFILE_SELECTOR_CONTRACT["projection"]["projection_safety_multiplier"]
        or runtime.get("image") != "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
        or runtime.get("network_volume_id") != "9mut3tpzwd"
        or runtime.get("network_volume_mount") != "/workspace"
        or runtime.get("data_center") != "US-CA-2"
        or runtime.get("python") != "3.12.3"
        or runtime.get("python_packages") != FROZEN_RUNTIME_PACKAGES
        or runtime.get("model_dependency_versions") != FROZEN_MODEL_DEPENDENCIES
        or runtime.get("offline_model_loading") is not True
        or runtime.get("runpodctl") != "2.9.0-c094cac"
        or runtime.get("runpod_provisioning") != RUNPOD_PROVISIONING_CONTRACT
        or runtime.get("runpod_operator_handoff") != RUNPOD_OPERATOR_HANDOFF_CONTRACT
        or runtime.get("h200_module_invocation") != H200_MODULE_INVOCATION_CONTRACT
        or runtime.get("live_price_and_stock_are_runtime_receipt_facts") is not True
    ):
        raise FreezeError("G00-F runtime, balance, image, or ceiling binding changed")
    if payload.get("execution_and_gate_contract") != EXECUTION_AND_GATE_CONTRACT:
        raise FreezeError("G00-F execution/evaluator gate contract changed")

    if payload.get("profile_contract") != PROFILE_CONTRACT:
        raise FreezeError("G00-F H200 execution profile contract changed")
    if payload.get("qualification_producer_contract") != QUALIFICATION_PRODUCER_CONTRACT:
        raise FreezeError("G00-F H200 qualification producer contract changed")
    if payload.get("storage_preflight_contract") != STORAGE_PREFLIGHT_CONTRACT:
        raise FreezeError("G00-F H200 storage preflight contract changed")
    if payload.get("watchdog_supervision_contract") != WATCHDOG_SUPERVISION_CONTRACT:
        raise FreezeError("G00-F H200 watchdog supervision contract changed")

    frozen_configs = payload.get("configurations")
    if not isinstance(frozen_configs, Mapping) or set(frozen_configs) != set(EXECUTION_PROFILES):
        raise FreezeError("G00-F H200 frozen profile set changed")
    frozen_models = payload.get("models")
    if not isinstance(frozen_models, Mapping) or set(frozen_models) != set(_PANEL_BASE):
        raise FreezeError("G00-F H200 frozen model panel set changed")
    for panel_id, specification in _PANEL_BASE.items():
        binding = frozen_models.get(panel_id)
        if not isinstance(binding, Mapping):
            raise FreezeError(f"freeze omits {panel_id} model binding")
        model = binding.get("model_snapshot")
        expected_leaves = [dict(row) for row in MODEL_LEAF_FILES[panel_id]]
        if (
            not isinstance(model, Mapping)
            or model.get("repo_id") != specification["model"]
            or model.get("revision") != specification["revision"]
            or model.get("materialization") != "fresh_regular_files_no_links_exact_10_leaf_full_repository"
            or model.get("leaf_files") != expected_leaves
            or model.get("leaf_manifest_digest") != semantic_digest(expected_leaves)
            or binding.get("model_runtime_identity") != MODEL_RUNTIME_IDENTITIES[panel_id]
            or binding.get("tokenizer_runtime_identity") != TOKENIZER_RUNTIME_IDENTITY
        ):
            raise FreezeError(f"G00-F {panel_id} model leaf manifest changed")

    candidate_plans: dict[str, dict[str, tuple[Mapping[str, Any], ...]]] = {}
    all_candidate_keys: set[str] = set()
    all_candidate_run_ids: set[str] = set()
    for profile in EXECUTION_PROFILES:
        profile_bindings = frozen_configs.get(profile)
        if not isinstance(profile_bindings, Mapping) or set(profile_bindings) != set(_PANEL_BASE):
            raise FreezeError(f"G00-F H200 {profile} frozen panel set changed")
        expected_rows = expected_plan_rows(resolved, profile=profile)
        verified_profile: dict[str, tuple[Mapping[str, Any], ...]] = {}
        for panel_id, specification in PROFILE_CONFIG_SPECS[profile].items():
            binding = profile_bindings.get(panel_id)
            if not isinstance(binding, Mapping):
                raise FreezeError(f"freeze omits {profile}/{panel_id} configuration binding")
            config_path = resolved / str(specification["path"])
            plan_path = resolved / str(specification["plan_path"])
            config = load_config(config_path)
            profile_contract = PROFILE_CONTRACT[profile]
            if (
                binding.get("config_path") != specification["path"]
                or sha256_file(config_path) != require_sha256(binding.get("config_sha256"), "config SHA-256")
                or binding.get("canonical_config_digest") != stable_hash(canonical_config(config), 64)
                or get_path(config, "run.launch_guard") != G00F_GUARD
                or get_path(config, "run.protocol_unlocked") is not False
                or tuple(sorted(get_path(config, "run.seeds"))) != tuple(specification["seeds"])
                or get_path(config, "model.name") != specification["model"]
                or get_path(config, "model.revision") != specification["revision"]
                or get_path(config, "train.batch_size") != profile_contract["train_batch_size"]
                or get_path(config, "train.gradient_accumulation_steps")
                != profile_contract["gradient_accumulation_steps"]
                or get_path(config, "train.gradient_checkpointing")
                is not profile_contract["gradient_checkpointing"]
                or get_path(config, "evaluation.batch_size") != profile_contract["evaluation_batch_size"]
                or binding.get("config_source_files")
                != config_source_chain(resolved, str(specification["path"]))
            ):
                raise FreezeError(f"G00-F H200 {profile}/{panel_id} config identity changed")
            rows = _read_jsonl(plan_path, f"{profile}/{panel_id} exact plan")
            if (
                binding.get("plan_path") != specification["plan_path"]
                or sha256_file(plan_path) != require_sha256(binding.get("plan_sha256"), "plan SHA-256")
                or rows != expected_rows[panel_id]
                or binding.get("run_count") != RUNS_PER_PANEL
                or binding.get("plan_key_digest")
                != semantic_digest(sorted(str(row["plan_key"]) for row in rows))
            ):
                raise FreezeError(f"G00-F H200 {profile}/{panel_id} plan identity changed")
            verified_profile[panel_id] = tuple(rows)
        _verify_worker_schedule(verified_profile, profile=profile)
        profile_keys = {str(row["plan_key"]) for rows in verified_profile.values() for row in rows}
        profile_run_ids = {str(row["run_id"]) for rows in verified_profile.values() for row in rows}
        if profile_keys & all_candidate_keys or profile_run_ids & all_candidate_run_ids:
            raise FreezeError("G00-F H200 profile plan/run identities are not disjoint")
        all_candidate_keys.update(profile_keys)
        all_candidate_run_ids.update(profile_run_ids)
        candidate_plans[profile] = verified_profile

    baseline_plans = candidate_plans["baseline"]

    return VerifiedFreeze(
        repo=resolved,
        path=target,
        file_sha256=observed,
        payload=payload,
        plans=baseline_plans,
        candidate_plans=candidate_plans,
    )


def create_model_snapshot_receipt(
    *,
    verified: VerifiedFreeze,
    panel_id: str,
    snapshot_root: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Hash every actual snapshot leaf before a G00-F weight update."""

    binding = _model_binding(verified, panel_id)
    model = binding["model_snapshot"]
    root_direct = Path(snapshot_root)
    root = root_direct.resolve()
    if root_direct.is_symlink() or not root.is_dir() or stat.S_IMODE(root.stat().st_mode) != 0o555:
        raise FreezeError(f"model snapshot directory is absent: {root}")
    expected_rows = model.get("leaf_files")
    if not isinstance(expected_rows, Sequence) or isinstance(expected_rows, (str, bytes)):
        raise FreezeError("frozen model leaf manifest is malformed")
    tree = list(root.rglob("*"))
    leaves = [path for path in tree if not path.is_dir()]
    directories = [path for path in tree if path.is_dir()]
    if (
        any(path.is_symlink() for path in tree)
        or any(not path.is_file() for path in leaves)
        or any(stat.S_IMODE(path.stat().st_mode) != 0o555 for path in directories)
    ):
        raise FreezeError("materialized model root contains a link or special file")
    actual_relative = sorted(path.relative_to(root).as_posix() for path in leaves)
    expected_relative = sorted(str(row["path"]) for row in expected_rows if isinstance(row, Mapping))
    if actual_relative != expected_relative:
        raise FreezeError("local model snapshot leaf set differs from the frozen manifest")
    receipt_rows: list[dict[str, Any]] = []
    for raw in expected_rows:
        if not isinstance(raw, Mapping):
            raise FreezeError("frozen model leaf row is malformed")
        relative = str(raw["path"])
        target = root / relative
        resolved_target = target.resolve(strict=True)
        if (
            target.is_symlink()
            or not resolved_target.is_file()
            or root not in resolved_target.parents
            or target.stat().st_nlink != 1
            or stat.S_IMODE(target.stat().st_mode) != 0o444
        ):
            raise FreezeError(f"model snapshot leaf is not an immutable direct file: {relative}")
        observed_bytes = resolved_target.stat().st_size
        observed_sha = sha256_file(resolved_target)
        if observed_bytes != int(raw["bytes"]) or observed_sha != raw["sha256"]:
            raise FreezeError(f"model snapshot leaf bytes changed: {relative}")
        receipt_rows.append(
            {
                "path": relative,
                "bytes": observed_bytes,
                "sha256": observed_sha,
                "lstat_mode": stat.S_IMODE(target.lstat().st_mode),
                "symlink_target": None,
            }
        )
    body = {
        "schema": MODEL_RECEIPT_SCHEMA,
        "schema_version": MODEL_RECEIPT_SCHEMA_VERSION,
        "panel_id": panel_id,
        "repo_id": model["repo_id"],
        "revision": model["revision"],
        "snapshot_root": str(root),
        "freeze_file_sha256": verified.file_sha256,
        "freeze_digest": verified.digest,
        "leaf_manifest_digest": model["leaf_manifest_digest"],
        "leaf_files": receipt_rows,
    }
    payload = {**body, "receipt_digest": semantic_digest(body)}
    exclusive_json(output, payload)
    return {**payload, "file_sha256": sha256_file(output)}


def materialize_model_snapshot(
    *,
    verified: VerifiedFreeze,
    panel_id: str,
    output_root: str | Path,
) -> dict[str, Any]:
    """Download one pinned public revision and dereference its exact ten leaves."""

    if panel_id not in CONFIG_SPECS:
        raise FreezeError("unknown G00-F model panel")
    target = Path(output_root).resolve()
    if target.exists():
        raise FreezeError("model materialization output must be a fresh absent path")
    target.parent.mkdir(parents=True, exist_ok=True)
    specification = CONFIG_SPECS[panel_id]
    rows = MODEL_LEAF_FILES[panel_id]
    try:
        from huggingface_hub import snapshot_download  # type: ignore[import-not-found]
    except ImportError as error:  # pragma: no cover - frozen GPU runtime dependency
        raise FreezeError("frozen huggingface-hub dependency is absent") from error
    download_cache = Path(tempfile.mkdtemp(prefix=".g00f-hf-download-", dir=target.parent))
    materialized = Path(tempfile.mkdtemp(prefix=f".{target.name}.materializing-", dir=target.parent))
    try:
        downloaded = Path(
            snapshot_download(
                repo_id=str(specification["model"]),
                revision=str(specification["revision"]),
                allow_patterns=[str(row["path"]) for row in rows],
                cache_dir=download_cache,
                token=False,
                max_workers=4,
            )
        ).resolve()
        for raw in rows:
            relative = str(raw["path"])
            source = (downloaded / relative).resolve(strict=True)
            if (
                not source.is_file()
                or source.stat().st_size != int(raw["bytes"])
                or sha256_file(source) != raw["sha256"]
            ):
                raise FreezeError(f"downloaded frozen model leaf changed: {relative}")
            destination = materialized / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as input_handle:
                descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(descriptor, "wb") as output_handle:
                    shutil.copyfileobj(input_handle, output_handle, length=8 * 1024 * 1024)
                    output_handle.flush()
                    os.fsync(output_handle.fileno())
            if sha256_file(destination) != raw["sha256"]:
                raise FreezeError(f"materialized frozen model leaf changed: {relative}")
            os.chmod(destination, 0o444)
        for directory in sorted(
            (path for path in materialized.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            os.chmod(directory, 0o555)
        os.replace(materialized, target)
        os.chmod(target, 0o555)
        directory_descriptor = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if materialized.exists():
            os.chmod(materialized, 0o700)
            for directory in (path for path in materialized.rglob("*") if path.is_dir()):
                os.chmod(directory, 0o700)
            shutil.rmtree(materialized, ignore_errors=True)
        shutil.rmtree(download_cache, ignore_errors=True)
    return {
        "panel_id": panel_id,
        "repo_id": specification["model"],
        "revision": specification["revision"],
        "snapshot_root": str(target),
        "leaf_count": len(rows),
        "leaf_manifest_digest": semantic_digest([dict(row) for row in rows]),
        "fresh_regular_files_no_links": True,
        "g01_launch_authorized": False,
    }


def verify_model_snapshot_receipt(
    *, verified: VerifiedFreeze, panel_id: str, receipt_path: str | Path
) -> dict[str, Any]:
    receipt = strict_json(receipt_path, f"{panel_id} model receipt")
    body = {key: value for key, value in receipt.items() if key != "receipt_digest"}
    if (
        receipt.get("schema") != MODEL_RECEIPT_SCHEMA
        or receipt.get("schema_version") != MODEL_RECEIPT_SCHEMA_VERSION
        or receipt.get("panel_id") != panel_id
        or receipt.get("freeze_file_sha256") != verified.file_sha256
        or receipt.get("freeze_digest") != verified.digest
        or receipt.get("receipt_digest") != semantic_digest(body)
    ):
        raise FreezeError(f"{panel_id} model receipt is not bound to this freeze")
    # Re-hash every leaf at worker start.  The temporary re-creation is kept
    # in memory; only the already supplied receipt remains authoritative.
    binding = _model_binding(verified, panel_id)
    model = binding["model_snapshot"]
    if (
        receipt.get("repo_id") != model["repo_id"]
        or receipt.get("revision") != model["revision"]
        or receipt.get("leaf_manifest_digest") != model["leaf_manifest_digest"]
    ):
        raise FreezeError(f"{panel_id} model receipt identity changed")
    root = Path(str(receipt.get("snapshot_root", "")))
    actual_rows = receipt.get("leaf_files")
    if not isinstance(actual_rows, Sequence) or isinstance(actual_rows, (str, bytes)):
        raise FreezeError(f"{panel_id} model receipt leaf set is malformed")
    frozen_by_path = {str(row["path"]): row for row in model["leaf_files"]}
    if {str(row.get("path")) for row in actual_rows if isinstance(row, Mapping)} != set(frozen_by_path):
        raise FreezeError(f"{panel_id} model receipt leaf set changed")
    root_resolved = root.resolve(strict=True)
    tree_entries = list(root_resolved.rglob("*"))
    if (
        root.is_symlink()
        or not root_resolved.is_dir()
        or stat.S_IMODE(root_resolved.stat().st_mode) != 0o555
        or any(entry.is_symlink() or not entry.is_file() for entry in tree_entries)
        or {entry.relative_to(root_resolved).as_posix() for entry in tree_entries} != set(frozen_by_path)
    ):
        raise FreezeError(f"{panel_id} model snapshot tree inventory changed")
    for row in actual_rows:
        if not isinstance(row, Mapping):
            raise FreezeError(f"{panel_id} model receipt leaf row is malformed")
        relative = str(row["path"])
        direct = root / relative
        target = direct.resolve(strict=True)
        frozen = frozen_by_path[relative]
        if (
            direct.is_symlink()
            or not target.is_file()
            or root_resolved not in target.parents
            or direct.stat().st_nlink != 1
            or stat.S_IMODE(direct.stat().st_mode) != 0o444
            or target.stat().st_size != int(frozen["bytes"])
            or sha256_file(target) != frozen["sha256"]
            or row.get("bytes") != frozen["bytes"]
            or row.get("sha256") != frozen["sha256"]
            or row.get("lstat_mode") != 0o444
            or row.get("symlink_target") is not None
        ):
            raise FreezeError(f"{panel_id} model receipt no longer replays: {relative}")
    return {
        "path": str(Path(receipt_path).resolve()),
        "file_sha256": sha256_file(receipt_path),
        "receipt_digest": receipt["receipt_digest"],
        "snapshot_root": str(root.resolve()),
        "panel_id": panel_id,
    }


def _finite_integration_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise FreezeError(f"model integration {label} must be finite numeric data")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise FreezeError(f"model integration {label} must be finite numeric data") from error
    if not math.isfinite(result):
        raise FreezeError(f"model integration {label} must be finite numeric data")
    return result


def _validate_model_integration_report(
    *,
    panel_id: str,
    report: Mapping[str, Any],
    snapshot_root: str | Path,
) -> None:
    """Replay the exact success surface of the frozen legacy boundary audit."""

    if set(report) != {"action_boundary", "model", "passed", "prompt_tokens", "runtime", "scores"}:
        raise FreezeError("model integration report field inventory changed")
    model = report.get("model")
    runtime = report.get("runtime")
    boundary = report.get("action_boundary")
    prompt_tokens = report.get("prompt_tokens")
    scores = report.get("scores")
    if not all(isinstance(value, Mapping) for value in (model, runtime, boundary, prompt_tokens, scores)):
        raise FreezeError("model integration report sections are malformed")
    assert isinstance(model, Mapping)
    assert isinstance(runtime, Mapping)
    assert isinstance(boundary, Mapping)
    assert isinstance(prompt_tokens, Mapping)
    assert isinstance(scores, Mapping)
    specification = CONFIG_SPECS[panel_id]
    model_identity = MODEL_RUNTIME_IDENTITIES[panel_id]
    tokenizer_identity = TOKENIZER_RUNTIME_IDENTITY
    if (
        report.get("passed") is not True
        or set(model)
        != {
            "action_labels",
            "action_token_ids",
            "chat_template_sha256",
            "dependency_versions",
            "model_class",
            "parameter_count",
            "peft_version",
            "requested_dtype",
            "requested_model",
            "requested_revision",
            "resolved_revision",
            "tokenizer_class",
            "tokenizer_name_or_path",
            "tokenizer_resolved_revision",
            "torch_version",
            "trainable_parameter_count",
            "transformers_version",
            "vocabulary_size",
        }
        or model.get("requested_model") != specification["model"]
        or model.get("requested_revision") != specification["revision"]
        or model.get("resolved_revision") != specification["revision"]
        or model.get("tokenizer_resolved_revision") != specification["revision"]
        or model.get("requested_dtype") != "bfloat16"
        or model.get("model_class") != model_identity["model_class"]
        or model.get("parameter_count") != model_identity["parameter_count"]
        or model.get("trainable_parameter_count") != 0
        or model.get("tokenizer_class") != tokenizer_identity["tokenizer_class"]
        or Path(str(model.get("tokenizer_name_or_path", ""))).resolve() != Path(snapshot_root).resolve()
        or model.get("vocabulary_size") != tokenizer_identity["vocabulary_size"]
        or model.get("chat_template_sha256") != tokenizer_identity["chat_template_sha256"]
        or model.get("action_labels") != tokenizer_identity["action_labels"]
        or model.get("action_token_ids") != tokenizer_identity["action_token_ids"]
        or model.get("dependency_versions") != FROZEN_MODEL_DEPENDENCIES
        or model.get("torch_version") != FROZEN_MODEL_DEPENDENCIES["torch"]
        or model.get("transformers_version") != FROZEN_MODEL_DEPENDENCIES["transformers"]
        or model.get("peft_version") != FROZEN_MODEL_DEPENDENCIES["peft"]
    ):
        raise FreezeError(f"{panel_id} model integration provenance changed")
    if (
        set(runtime)
        != {
            "dependencies",
            "last_logit_parameter",
            "parameter_devices",
            "parameter_dtypes",
            "requested_device",
        }
        or runtime.get("requested_device") != "cuda"
        or runtime.get("parameter_devices") != ["cuda:0"]
        or runtime.get("parameter_dtypes") != ["torch.bfloat16"]
        or runtime.get("dependencies") != FROZEN_MODEL_DEPENDENCIES
    ):
        raise FreezeError(f"{panel_id} model integration runtime changed")
    continuations = boundary.get("continuation_token_ids_by_prompt")
    continuation_lengths = boundary.get("continuation_token_lengths_by_prompt")
    if (
        set(boundary)
        != {
            "continuation_token_ids_by_prompt",
            "continuation_token_lengths_by_prompt",
            "labels",
            "prefix_stable",
            "standalone_token_ids",
        }
        or boundary.get("labels") != list(MODEL_INTEGRATION_ACTION_LABELS)
        or boundary.get("standalone_token_ids") != tokenizer_identity["action_token_ids"]
        or boundary.get("prefix_stable") is not True
        or not isinstance(continuations, Sequence)
        or isinstance(continuations, (str, bytes))
        or len(continuations) != len(MODEL_INTEGRATION_PROMPTS)
        or not isinstance(continuation_lengths, Sequence)
        or isinstance(continuation_lengths, (str, bytes))
        or len(continuation_lengths) != len(MODEL_INTEGRATION_PROMPTS)
    ):
        raise FreezeError(f"{panel_id} continuation-token boundary changed")
    for token_pair, length_pair in zip(continuations, continuation_lengths, strict=True):
        if (
            not isinstance(token_pair, Sequence)
            or isinstance(token_pair, (str, bytes))
            or len(token_pair) != 2
            or not isinstance(length_pair, Sequence)
            or isinstance(length_pair, (str, bytes))
            or len(length_pair) != 2
            or any(
                not isinstance(tokens, Sequence) or isinstance(tokens, (str, bytes)) or len(tokens) < 1
                for tokens in token_pair
            )
            or [len(tokens) for tokens in token_pair] != list(length_pair)
        ):
            raise FreezeError(f"{panel_id} contextual A/B continuations are malformed")
    counts = prompt_tokens.get("counts")
    if (
        set(prompt_tokens) != {"configured_maximum", "counts", "maximum"}
        or prompt_tokens.get("configured_maximum") is not None
        or not isinstance(counts, Sequence)
        or isinstance(counts, (str, bytes))
        or len(counts) != len(MODEL_INTEGRATION_PROMPTS)
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in counts)
        or prompt_tokens.get("maximum") != max(counts)
    ):
        raise FreezeError(f"{panel_id} integration prompt-token counts changed")
    normalized = scores.get("normalized_log_scores")
    probabilities = scores.get("probabilities")
    if (
        set(scores)
        != {
            "finite",
            "maximum_normalization_error",
            "maximum_probability_swap_error",
            "maximum_score_swap_error",
            "normalized_log_scores",
            "probabilities",
            "swap_atol",
            "swap_invariant",
            "swap_rtol",
        }
        or scores.get("finite") is not True
        or scores.get("swap_invariant") is not True
        or scores.get("swap_atol") != MODEL_INTEGRATION_SWAP_ATOL
        or scores.get("swap_rtol") != MODEL_INTEGRATION_SWAP_RTOL
        or not isinstance(normalized, Sequence)
        or isinstance(normalized, (str, bytes))
        or not isinstance(probabilities, Sequence)
        or isinstance(probabilities, (str, bytes))
        or len(normalized) != len(MODEL_INTEGRATION_PROMPTS)
        or len(probabilities) != len(MODEL_INTEGRATION_PROMPTS)
    ):
        raise FreezeError(f"{panel_id} integration score contract changed")
    for label, rows in (("normalized scores", normalized), ("probabilities", probabilities)):
        for row in rows:
            if (
                not isinstance(row, Sequence)
                or isinstance(row, (str, bytes))
                or len(row) != 2
                or any(not math.isfinite(_finite_integration_number(value, label)) for value in row)
            ):
                raise FreezeError(f"{panel_id} integration {label} are malformed")
    probability_rows = [
        [_finite_integration_number(value, "probability") for value in row] for row in probabilities
    ]
    normalized_rows = [
        [_finite_integration_number(value, "normalized log score") for value in row] for row in normalized
    ]
    if any(value < 0 or value > 1 for row in probability_rows for value in row) or any(
        not math.isclose(
            sum(row),
            1.0,
            rel_tol=MODEL_INTEGRATION_PROBABILITY_REPLAY_TOLERANCE,
            abs_tol=MODEL_INTEGRATION_PROBABILITY_REPLAY_TOLERANCE,
        )
        for row in probability_rows
    ):
        raise FreezeError(f"{panel_id} integration probabilities are not normalized")
    if any(
        not math.isclose(
            math.exp(log_score),
            probability,
            rel_tol=MODEL_INTEGRATION_PROBABILITY_REPLAY_TOLERANCE,
            abs_tol=MODEL_INTEGRATION_PROBABILITY_REPLAY_TOLERANCE,
        )
        for log_row, probability_row in zip(normalized_rows, probability_rows, strict=True)
        for log_score, probability in zip(log_row, probability_row, strict=True)
    ):
        raise FreezeError(f"{panel_id} normalized scores disagree with probabilities")
    for key in (
        "maximum_normalization_error",
        "maximum_probability_swap_error",
        "maximum_score_swap_error",
    ):
        if _finite_integration_number(scores.get(key), key) < 0:
            raise FreezeError(f"{panel_id} integration {key} is negative")
    if (
        _finite_integration_number(scores.get("maximum_normalization_error"), "normalization")
        > MODEL_INTEGRATION_PROBABILITY_REPLAY_TOLERANCE
    ):
        raise FreezeError(f"{panel_id} integration normalization error is too large")
    if (
        _finite_integration_number(scores.get("maximum_score_swap_error"), "score swap error")
        > MODEL_INTEGRATION_SWAP_ATOL
        or _finite_integration_number(
            scores.get("maximum_probability_swap_error"),
            "probability swap error",
        )
        > MODEL_INTEGRATION_SWAP_ATOL
    ):
        raise FreezeError(f"{panel_id} integration A/B swap error exceeds 2e-3")


def create_model_integration_audit(
    *,
    verified: VerifiedFreeze,
    panel_id: str,
    model_receipt_path: str | Path,
    output: str | Path,
    integration_runner: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run and bind the legacy real-model boundary check before ITT creation."""

    if panel_id not in CONFIG_SPECS:
        raise FreezeError("unknown G00-F model integration panel")
    target = Path(output).resolve()
    execution_root = target.parent
    if target != execution_root / f"model-integration-audit-{panel_id.removeprefix('g00f-')}.json":
        raise FreezeError("model integration audit has a noncanonical execution-root path")
    if (execution_root / "itt-ledger").exists() or any(execution_root.glob("worker-*-launch.json")):
        raise FreezeError("model integration audit must precede the ITT ledger and worker launch")
    model_receipt = verify_model_snapshot_receipt(
        verified=verified,
        panel_id=panel_id,
        receipt_path=model_receipt_path,
    )
    expected_model_receipt = execution_root / f"model-receipt-{panel_id.removeprefix('g00f-')}.json"
    if Path(str(model_receipt["path"])) != expected_model_receipt:
        raise FreezeError("model integration audit receipt lies outside its execution root")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise FreezeError("model integration audit requires both offline-mode guards")
    if integration_runner is None:
        from goalzendo.modeling import run_model_integration_check

        integration_runner = run_model_integration_check
    model_receipts = {
        candidate: model_receipt if candidate == panel_id else {"snapshot_root": "unused"}
        for candidate in CONFIG_SPECS
    }
    specification = CONFIG_SPECS[panel_id]
    try:
        report = dict(
            integration_runner(
                {
                    "name": specification["model"],
                    "revision": specification["revision"],
                    "dtype": "bfloat16",
                    "trust_remote_code": False,
                },
                prompts=MODEL_INTEGRATION_PROMPTS,
                action_labels=MODEL_INTEGRATION_ACTION_LABELS,
                device="cuda",
                max_prompt_tokens=None,
                system_prompt=MODEL_INTEGRATION_SYSTEM_PROMPT,
                model_loader=_authenticated_local_model_loader(model_receipts),
                strict_revision=True,
                swap_atol=MODEL_INTEGRATION_SWAP_ATOL,
                swap_rtol=MODEL_INTEGRATION_SWAP_RTOL,
            )
        )
    except BaseException as error:
        raise FreezeError(f"{panel_id} pretraining model integration audit failed") from error
    _validate_model_integration_report(
        panel_id=panel_id,
        report=report,
        snapshot_root=model_receipt["snapshot_root"],
    )
    contract = copy.deepcopy(dict(EXECUTION_AND_GATE_CONTRACT["pretraining_model_boundary"]))
    body = {
        "schema": MODEL_INTEGRATION_AUDIT_SCHEMA,
        "schema_version": MODEL_INTEGRATION_AUDIT_SCHEMA_VERSION,
        "panel_id": panel_id,
        "freeze_file_sha256": verified.file_sha256,
        "freeze_digest": verified.digest,
        "model_snapshot_receipt": model_receipt,
        "contract": contract,
        "report": report,
        "report_digest": semantic_digest(report),
        "ledger_absent_at_creation": True,
        "weight_updates_performed": False,
        "outcomes_seen": False,
        "g01_launch_authorized": False,
    }
    payload = {**body, "audit_digest": semantic_digest(body)}
    exclusive_json(target, payload)
    return {
        "path": str(target),
        "file_sha256": sha256_file(target),
        "audit_digest": payload["audit_digest"],
        "report_digest": payload["report_digest"],
        "panel_id": panel_id,
        "model_snapshot_receipt": model_receipt,
    }


def verify_model_integration_audit(
    *,
    verified: VerifiedFreeze,
    panel_id: str,
    audit_path: str | Path,
    replay_model_snapshot: bool = True,
) -> dict[str, Any]:
    target = Path(audit_path).resolve()
    execution_root = target.parent
    if target != execution_root / f"model-integration-audit-{panel_id.removeprefix('g00f-')}.json":
        raise FreezeError("model integration audit has a noncanonical execution-root path")
    payload = strict_json(target, f"{panel_id} model integration audit")
    body = {key: value for key, value in payload.items() if key != "audit_digest"}
    exact_body_keys = {
        "contract",
        "freeze_digest",
        "freeze_file_sha256",
        "g01_launch_authorized",
        "ledger_absent_at_creation",
        "model_snapshot_receipt",
        "outcomes_seen",
        "panel_id",
        "report",
        "report_digest",
        "schema",
        "schema_version",
        "weight_updates_performed",
    }
    raw_model_receipt = payload.get("model_snapshot_receipt")
    report = payload.get("report")
    if (
        set(body) != exact_body_keys
        or payload.get("schema") != MODEL_INTEGRATION_AUDIT_SCHEMA
        or payload.get("schema_version") != MODEL_INTEGRATION_AUDIT_SCHEMA_VERSION
        or payload.get("panel_id") != panel_id
        or payload.get("freeze_file_sha256") != verified.file_sha256
        or payload.get("freeze_digest") != verified.digest
        or payload.get("contract") != EXECUTION_AND_GATE_CONTRACT["pretraining_model_boundary"]
        or not isinstance(raw_model_receipt, Mapping)
        or not isinstance(report, Mapping)
        or payload.get("report_digest") != semantic_digest(report)
        or payload.get("ledger_absent_at_creation") is not True
        or payload.get("weight_updates_performed") is not False
        or payload.get("outcomes_seen") is not False
        or payload.get("g01_launch_authorized") is not False
        or payload.get("audit_digest") != semantic_digest(body)
    ):
        raise FreezeError(f"{panel_id} model integration audit changed")
    receipt_path = execution_root / f"model-receipt-{panel_id.removeprefix('g00f-')}.json"
    if replay_model_snapshot:
        model_receipt = verify_model_snapshot_receipt(
            verified=verified,
            panel_id=panel_id,
            receipt_path=receipt_path,
        )
    else:
        receipt = strict_json(receipt_path, f"{panel_id} model receipt bound by integration audit")
        model_receipt = {
            "path": str(receipt_path),
            "file_sha256": sha256_file(receipt_path),
            "receipt_digest": receipt.get("receipt_digest"),
            "snapshot_root": str(Path(str(receipt.get("snapshot_root", ""))).resolve()),
            "panel_id": panel_id,
        }
    if raw_model_receipt != model_receipt:
        raise FreezeError(f"{panel_id} integration/model-snapshot receipt binding changed")
    _validate_model_integration_report(
        panel_id=panel_id,
        report=report,
        snapshot_root=model_receipt["snapshot_root"],
    )
    return {
        "path": str(target),
        "file_sha256": sha256_file(target),
        "audit_digest": payload["audit_digest"],
        "report_digest": payload["report_digest"],
        "panel_id": panel_id,
        "model_snapshot_receipt": model_receipt,
    }


def verify_runpod_provision_receipt(
    *,
    verified: VerifiedFreeze,
    receipt_path: str | Path,
    expected_receipt_sha256: str,
    expected_pod_id: str,
    require_bound_copy: bool = True,
) -> dict[str, Any]:
    """Replay every provider field exposed by the pinned Runpod CLI responses."""

    direct = Path(receipt_path)
    target = direct.resolve()
    expected_sha = require_sha256(
        expected_receipt_sha256,
        "externally expected Runpod provision receipt SHA-256",
    )
    if not expected_pod_id or any(character.isspace() for character in expected_pod_id):
        raise FreezeError("RUNPOD_POD_ID must be a nonempty token")
    if not target.is_file() or sha256_file(target) != expected_sha:
        raise FreezeError("Runpod provision receipt lacks its external SHA-256 binding")
    if direct.is_symlink() or (require_bound_copy and target.stat().st_nlink != 1):
        raise FreezeError("bound Runpod provision receipt must be one direct regular file")
    if require_bound_copy and stat.S_IMODE(target.stat().st_mode) != 0o400:
        raise FreezeError("bound Runpod provision receipt must be immutable mode 0400")
    receipt = strict_json(target, "externally pinned Runpod provision receipt")
    body = {key: value for key, value in receipt.items() if key != "receipt_digest"}
    exact_keys = {
        "api_response",
        "capture_tool",
        "created_at_utc",
        "data_center",
        "evidence_boundaries",
        "execution_uuid",
        "g01_launch_authorized",
        "gpu_count",
        "gpu_catalog",
        "gpu_catalog_snapshot",
        "image",
        "network_volume_id",
        "network_volume_mount",
        "operational_market_snapshot",
        "outcomes_seen",
        "pod_id",
        "pod_name",
        "provider_allocation",
        "provider_output_limitations",
        "provisioning",
        "provider",
        "readiness",
        "runpodctl_version_capture",
        "schema",
        "schema_version",
        "ssh_info_response",
    }
    runtime = verified.payload["runtime"]
    if (
        set(body) != exact_keys
        or receipt.get("schema") != RUNPOD_PROVISION_RECEIPT_SCHEMA
        or receipt.get("schema_version") != RUNPOD_PROVISION_RECEIPT_SCHEMA_VERSION
        or receipt.get("provider") != "runpod"
        or receipt.get("capture_tool") != "runpodctl 2.9.0-c094cac"
        or receipt.get("image") != runtime["image"]
        or receipt.get("gpu_count") != WORKER_COUNT
        or receipt.get("gpu_catalog")
        != {
            "display_name": "H200 SXM",
            "gpu_id": "NVIDIA H200",
            "memory_in_gb": 141,
        }
        or receipt.get("data_center") != runtime["data_center"]
        or receipt.get("network_volume_id") != runtime["network_volume_id"]
        or receipt.get("network_volume_mount") != runtime["network_volume_mount"]
        or receipt.get("provider_output_limitations")
        != RUNPOD_PROVISIONING_CONTRACT["provider_output_limitations"]
        or receipt.get("evidence_boundaries") != RUNPOD_PROVISION_EVIDENCE
        or receipt.get("outcomes_seen") is not False
        or receipt.get("g01_launch_authorized") is not False
        or receipt.get("receipt_digest") != semantic_digest(body)
    ):
        raise FreezeError("Runpod provision receipt differs from the frozen allocation contract")
    execution_uuid = str(receipt.get("execution_uuid", ""))
    expected_pod_name = f"goalzendo-g00f-h200-{execution_uuid}"
    try:
        parsed_execution_uuid = uuid.UUID(execution_uuid)
    except ValueError as error:
        raise FreezeError("Runpod provision execution UUID is not a canonical UUID4") from error
    if (
        parsed_execution_uuid.version != 4
        or str(parsed_execution_uuid) != execution_uuid
        or receipt.get("pod_name") != expected_pod_name
    ):
        raise FreezeError("Runpod provision execution UUID or unique pod name changed")

    version_binding = receipt.get("runpodctl_version_capture")
    if not isinstance(version_binding, Mapping) or set(version_binding) != {
        "command_argv",
        "file_name",
        "sha256",
    }:
        raise FreezeError("Runpod CLI version binding is malformed")
    version_direct = target.parent / str(RUNPOD_PROVISIONING_CONTRACT["runpodctl_version_capture_file"])
    version = verify_runpodctl_version_capture(
        version_path=version_direct,
        require_bound_copy=require_bound_copy,
    )
    if version != dict(version_binding):
        raise FreezeError("Runpod provision receipt does not derive from its CLI version bytes")

    catalog_snapshot = receipt.get("gpu_catalog_snapshot")
    if not isinstance(catalog_snapshot, Mapping) or set(catalog_snapshot) != {
        "command_argv",
        "file_name",
        "operator_capture_utc",
        "sha256",
    }:
        raise FreezeError("Runpod provision receipt GPU catalog binding is malformed")
    catalog_sha = require_sha256(
        catalog_snapshot.get("sha256"),
        "Runpod GPU catalog response SHA-256",
    )
    catalog_direct = target.parent / str(RUNPOD_PROVISIONING_CONTRACT["gpu_catalog_capture_file"])
    if (
        catalog_snapshot.get("command_argv") != list(RUNPOD_PROVISIONING_CONTRACT["gpu_catalog_capture_argv"])
        or catalog_snapshot.get("file_name") != RUNPOD_PROVISIONING_CONTRACT["gpu_catalog_capture_file"]
        or catalog_direct.is_symlink()
        or not catalog_direct.is_file()
        or sha256_file(catalog_direct) != catalog_sha
        or (require_bound_copy and catalog_direct.stat().st_nlink != 1)
        or (require_bound_copy and stat.S_IMODE(catalog_direct.stat().st_mode) != 0o400)
    ):
        raise FreezeError("bound raw Runpod GPU catalog response bytes do not replay")
    catalog = verify_runpod_gpu_catalog_snapshot(
        catalog_path=catalog_direct,
        operator_capture_utc=str(catalog_snapshot.get("operator_capture_utc", "")),
        require_bound_copy=require_bound_copy,
    )
    if (
        catalog["binding"] != dict(catalog_snapshot)
        or catalog["gpu_catalog"] != receipt.get("gpu_catalog")
        or catalog["operational_market_snapshot"] != receipt.get("operational_market_snapshot")
    ):
        raise FreezeError("Runpod provision receipt does not derive from its raw GPU catalog")

    api_response = receipt.get("api_response")
    if not isinstance(api_response, Mapping) or set(api_response) != {
        "capture_command",
        "file_name",
        "sha256",
    }:
        raise FreezeError("Runpod provision receipt raw API response binding is malformed")
    if (
        api_response.get("capture_command")
        != (f"runpodctl pod get {expected_pod_id} --include-machine --include-network-volume -o json")
        or api_response.get("file_name") != "runpod-api-response.json"
    ):
        raise FreezeError("Runpod raw API capture command or canonical file name changed")
    raw_sha = require_sha256(api_response.get("sha256"), "Runpod API response SHA-256")
    raw_direct = target.parent / "runpod-api-response.json"
    if (
        raw_direct.is_symlink()
        or not raw_direct.is_file()
        or sha256_file(raw_direct) != raw_sha
        or (require_bound_copy and raw_direct.stat().st_nlink != 1)
        or (require_bound_copy and stat.S_IMODE(raw_direct.stat().st_mode) != 0o400)
    ):
        raise FreezeError("bound raw Runpod API response bytes do not replay")

    ssh_info_response = receipt.get("ssh_info_response")
    expected_ssh_info_argv = ["runpodctl", "ssh", "info", expected_pod_id, "-o", "json"]
    if not isinstance(ssh_info_response, Mapping) or set(ssh_info_response) != {
        "command_argv",
        "file_name",
        "sha256",
    }:
        raise FreezeError("Runpod SSH-info response binding is malformed")
    ssh_info_sha = require_sha256(
        ssh_info_response.get("sha256"),
        "Runpod SSH-info response SHA-256",
    )
    ssh_info_direct = target.parent / "runpod-ssh-info-response.json"
    if (
        ssh_info_response.get("command_argv") != expected_ssh_info_argv
        or ssh_info_response.get("file_name") != ssh_info_direct.name
        or ssh_info_direct.is_symlink()
        or not ssh_info_direct.is_file()
        or sha256_file(ssh_info_direct) != ssh_info_sha
        or (require_bound_copy and ssh_info_direct.stat().st_nlink != 1)
        or (require_bound_copy and stat.S_IMODE(ssh_info_direct.stat().st_mode) != 0o400)
    ):
        raise FreezeError("bound raw Runpod SSH-info response bytes do not replay")

    provisioning = receipt.get("provisioning")
    if not isinstance(provisioning, Mapping) or set(provisioning) != {
        "capture_order",
        "catalog",
        "create",
        "get",
        "maximum_secure_cost_usd",
        "provision_ceiling_seconds",
        "ssh_info",
        "terminate_after_utc",
        "version",
    }:
        raise FreezeError("Runpod provisioning chronology or termination guard is malformed")
    create_binding = provisioning.get("create")
    get_binding = provisioning.get("get")
    provision_ssh_binding = provisioning.get("ssh_info")
    provision_version_binding = provisioning.get("version")
    provision_catalog_binding = provisioning.get("catalog")
    if (
        not isinstance(provision_catalog_binding, Mapping)
        or dict(provision_catalog_binding) != dict(catalog_snapshot)
        or not isinstance(create_binding, Mapping)
        or set(create_binding) != {"command_argv", "file_name", "sha256"}
        or not isinstance(get_binding, Mapping)
        or set(get_binding) != {"command_argv", "file_name", "sha256"}
        or not isinstance(provision_ssh_binding, Mapping)
        or dict(provision_ssh_binding) != dict(ssh_info_response)
        or not isinstance(provision_version_binding, Mapping)
        or dict(provision_version_binding) != dict(version_binding)
    ):
        raise FreezeError("Runpod create/get raw-response binding is malformed")
    terminate_after = str(provisioning.get("terminate_after_utc", ""))
    terminate_time = _canonical_utc(terminate_after, "Runpod absolute termination time")
    expected_get_argv = [
        "runpodctl",
        "pod",
        "get",
        expected_pod_id,
        "--include-machine",
        "--include-network-volume",
        "-o",
        "json",
    ]
    create_raw_sha = require_sha256(create_binding.get("sha256"), "Runpod create response SHA-256")
    create_raw = target.parent / "runpod-create-response.json"
    if (
        provisioning.get("capture_order")
        != ["version", "catalog", "create", "get", "ssh_info", "ssh_identity"]
        or create_binding.get("command_argv")
        != list(
            canonical_runpod_create_command(
                execution_uuid=execution_uuid,
                terminate_after_utc=terminate_after,
            )
        )
        or create_binding.get("file_name") != "runpod-create-response.json"
        or get_binding.get("command_argv") != expected_get_argv
        or get_binding.get("file_name") != "runpod-api-response.json"
        or get_binding.get("sha256") != raw_sha
        or provisioning.get("provision_ceiling_seconds")
        != RUNPOD_PROVISIONING_CONTRACT["provision_ceiling_seconds"]
        or provisioning.get("maximum_secure_cost_usd")
        != RUNPOD_PROVISIONING_CONTRACT["maximum_secure_cost_usd"]
        or create_raw.is_symlink()
        or not create_raw.is_file()
        or sha256_file(create_raw) != create_raw_sha
        or (require_bound_copy and create_raw.stat().st_nlink != 1)
        or (require_bound_copy and stat.S_IMODE(create_raw.stat().st_mode) != 0o400)
    ):
        raise FreezeError("Runpod termination guard exceeds or differs from the exact 17-hour ceiling")

    create_semantics = _runpod_create_semantics(
        create_path=create_raw,
        runtime=runtime,
        catalog_price_per_gpu_hour=float(
            catalog["operational_market_snapshot"]["secure_price_usd_per_gpu_hour"]
        ),
    )
    get_semantics = _runpod_get_semantics(
        get_path=raw_direct,
        runtime=runtime,
        expected_create=create_semantics,
    )
    ssh_semantics = _runpod_ssh_info_semantics(
        ssh_info_path=ssh_info_direct,
        expected_get=get_semantics,
    )
    if ssh_semantics != get_semantics["ssh"]:
        raise FreezeError("Runpod get and SSH-info responses disagree on pod coordinates")
    if (
        create_semantics["pod_id"] != expected_pod_id
        or create_semantics["name"] != expected_pod_name
        or get_semantics["pod_id"] != expected_pod_id
        or receipt.get("pod_id") != expected_pod_id
        or receipt.get("created_at_utc") != get_semantics["created_at_utc"]
        or receipt.get("data_center") != get_semantics["data_center"]
        or receipt.get("provider_allocation") != {"create": create_semantics, "get": get_semantics}
    ):
        raise FreezeError("Runpod provision receipt does not derive its allocation from provider bytes")
    created = str(get_semantics["created_at_utc"])
    created_time = _runpod_created_utc(created, "Runpod provision receipt creation time")
    provision_seconds = (terminate_time - created_time).total_seconds()
    if (
        provision_seconds <= 0
        or provision_seconds > RUNPOD_PROVISIONING_CONTRACT["provision_ceiling_seconds"]
    ):
        raise FreezeError("Runpod termination guard exceeds or differs from the exact 17-hour ceiling")

    readiness = receipt.get("readiness")
    if not isinstance(readiness, Mapping) or set(readiness) != {
        "final_get",
        "poll_interval_seconds",
        "ssh_identity",
        "ssh_info",
        "strategy",
        "timeout_seconds",
    }:
        raise FreezeError("Runpod readiness receipt is malformed")
    identity_binding = readiness.get("ssh_identity")
    if (
        readiness.get("strategy") != "no_wait_create_then_bounded_get_ssh_identity_poll"
        or readiness.get("timeout_seconds") != RUNPOD_PROVISIONING_CONTRACT["readiness_poll_timeout_seconds"]
        or readiness.get("poll_interval_seconds")
        != RUNPOD_PROVISIONING_CONTRACT["readiness_poll_interval_seconds"]
        or readiness.get("final_get") != dict(get_binding)
        or readiness.get("ssh_info") != dict(ssh_info_response)
        or not isinstance(identity_binding, Mapping)
        or set(identity_binding) != {"file_name", "receipt_digest", "sha256"}
    ):
        raise FreezeError("Runpod readiness policy or evidence bindings changed")
    identity_direct = target.parent / "runpod-ssh-identity-receipt.json"
    verified_identity = _verify_runpod_ssh_identity_receipt(
        receipt_path=identity_direct,
        expected_get_sha256=raw_sha,
        expected_ssh_info_sha256=ssh_info_sha,
        expected_connection=ssh_semantics,
        require_bound_copy=require_bound_copy,
    )
    if verified_identity != dict(identity_binding):
        raise FreezeError("Runpod readiness does not bind the authenticated SSH identity receipt")
    identity_payload = strict_json(identity_direct, "Runpod SSH identity receipt")
    identity_time = _canonical_utc(
        str(identity_payload["observed_at_utc"]),
        "Runpod SSH identity observation time",
    )
    readiness_seconds = (identity_time - created_time).total_seconds()
    if not 0 <= readiness_seconds <= RUNPOD_PROVISIONING_CONTRACT["readiness_poll_timeout_seconds"]:
        raise FreezeError("Runpod SSH readiness was not reached inside the frozen 900-second window")

    market = receipt.get("operational_market_snapshot")
    if not isinstance(market, Mapping) or set(market) != {
        "observed_at_utc",
        "scientific_identity",
        "secure_price_usd_per_gpu_hour",
        "stock_label",
    }:
        raise FreezeError("Runpod operational price/stock snapshot is malformed")
    price = market.get("secure_price_usd_per_gpu_hour")
    if (
        isinstance(price, bool)
        or not isinstance(price, (int, float))
        or not math.isfinite(float(price))
        or float(price) < 0
        or float(price) > float(RUNPOD_PROVISIONING_CONTRACT["secure_price_ceiling_usd_per_gpu_hour"])
        or not isinstance(market.get("stock_label"), str)
        or not market["stock_label"]
        or market.get("observed_at_utc") != catalog_snapshot.get("operator_capture_utc")
        or market.get("scientific_identity") is not False
    ):
        raise FreezeError("Runpod operational price/stock facts changed type or role")
    computed_cost = float(price) * WORKER_COUNT * provision_seconds / 3_600
    if computed_cost > float(RUNPOD_PROVISIONING_CONTRACT["maximum_secure_cost_usd"]) + 1e-12:
        raise FreezeError("Runpod provision duration and price exceed the frozen cost ceiling")
    return {
        "path": str(target),
        "file_sha256": expected_sha,
        "receipt_digest": receipt["receipt_digest"],
        "execution_uuid": execution_uuid,
        "pod_id": expected_pod_id,
        "network_volume_id": receipt["network_volume_id"],
        "created_at_utc": created,
        "provider_allocation": copy.deepcopy(dict(receipt["provider_allocation"])),
        "provisioning": copy.deepcopy(dict(provisioning)),
        "api_response": {
            "path": str(raw_direct.resolve()),
            "file_sha256": raw_sha,
            "capture_command": api_response["capture_command"],
        },
        "gpu_catalog_snapshot": {
            "path": str(catalog_direct.resolve()),
            "file_sha256": catalog_sha,
            "operator_capture_utc": catalog_snapshot["operator_capture_utc"],
            "command_argv": list(catalog_snapshot["command_argv"]),
        },
        "readiness": copy.deepcopy(dict(readiness)),
        "runpodctl_version_capture": {
            "path": str(version_direct.resolve()),
            "file_sha256": version_binding["sha256"],
            "command_argv": list(version_binding["command_argv"]),
        },
        "ssh_info_response": {
            "path": str(ssh_info_direct.resolve()),
            "file_sha256": ssh_info_sha,
            "command_argv": list(ssh_info_response["command_argv"]),
        },
        "evidence_boundaries": dict(RUNPOD_PROVISION_EVIDENCE),
        "operational_market_snapshot": copy.deepcopy(dict(market)),
    }


def create_runpod_provision_receipt(
    *,
    verified: VerifiedFreeze,
    raw_runpodctl_version_path: str | Path,
    raw_gpu_catalog_path: str | Path,
    catalog_operator_capture_utc: str,
    execution_uuid: str,
    raw_create_response_path: str | Path,
    raw_api_response_path: str | Path,
    raw_ssh_info_path: str | Path,
    ssh_identity_receipt_path: str | Path,
    terminate_after_utc: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Create the exact reviewer-facing receipt; an external party pins its SHA."""

    version_raw_direct = Path(raw_runpodctl_version_path)
    version_raw = version_raw_direct.resolve()
    catalog_raw_direct = Path(raw_gpu_catalog_path)
    catalog_raw = catalog_raw_direct.resolve()
    create_raw_direct = Path(raw_create_response_path)
    create_raw = create_raw_direct.resolve()
    raw_direct = Path(raw_api_response_path)
    raw = raw_direct.resolve()
    ssh_info_direct = Path(raw_ssh_info_path)
    ssh_info = ssh_info_direct.resolve()
    identity_direct = Path(ssh_identity_receipt_path)
    identity = identity_direct.resolve()
    output = Path(output_path).resolve()
    if (
        version_raw != output.parent / RUNPOD_PROVISIONING_CONTRACT["runpodctl_version_capture_file"]
        or version_raw_direct.is_symlink()
        or not version_raw.is_file()
        or catalog_raw != output.parent / RUNPOD_PROVISIONING_CONTRACT["gpu_catalog_capture_file"]
        or catalog_raw_direct.is_symlink()
        or not catalog_raw.is_file()
        or create_raw != output.parent / "runpod-create-response.json"
        or create_raw_direct.is_symlink()
        or not create_raw.is_file()
        or raw != output.parent / "runpod-api-response.json"
        or raw_direct.is_symlink()
        or not raw.is_file()
        or ssh_info != output.parent / "runpod-ssh-info-response.json"
        or ssh_info_direct.is_symlink()
        or not ssh_info.is_file()
        or identity != output.parent / "runpod-ssh-identity-receipt.json"
        or identity_direct.is_symlink()
        or not identity.is_file()
    ):
        raise FreezeError(
            "raw Runpod version/catalog/create/get/SSH evidence must be direct canonical siblings"
        )
    runtime = verified.payload["runtime"]
    canonical_create_argv = canonical_runpod_create_command(
        execution_uuid=execution_uuid,
        terminate_after_utc=terminate_after_utc,
    )
    expected_pod_name = f"goalzendo-g00f-h200-{execution_uuid}"
    version = verify_runpodctl_version_capture(version_path=version_raw_direct)
    catalog = verify_runpod_gpu_catalog_snapshot(
        catalog_path=catalog_raw_direct,
        operator_capture_utc=catalog_operator_capture_utc,
    )
    _canonical_utc(terminate_after_utc, "Runpod absolute termination time")
    create_semantics = _runpod_create_semantics(
        create_path=create_raw_direct,
        runtime=runtime,
        catalog_price_per_gpu_hour=float(
            catalog["operational_market_snapshot"]["secure_price_usd_per_gpu_hour"]
        ),
    )
    pod_id = str(create_semantics["pod_id"])
    if create_semantics["name"] != expected_pod_name:
        raise FreezeError("Runpod create response does not use the precommitted unique pod name")
    get_semantics = _runpod_get_semantics(
        get_path=raw_direct,
        runtime=runtime,
        expected_create=create_semantics,
    )
    ssh_semantics = _runpod_ssh_info_semantics(
        ssh_info_path=ssh_info_direct,
        expected_get=get_semantics,
    )
    if ssh_semantics != get_semantics["ssh"]:
        raise FreezeError("Runpod get and SSH-info responses disagree on pod coordinates")
    identity_binding = _verify_runpod_ssh_identity_receipt(
        receipt_path=identity_direct,
        expected_get_sha256=sha256_file(raw),
        expected_ssh_info_sha256=sha256_file(ssh_info),
        expected_connection=ssh_semantics,
        require_bound_copy=False,
    )
    created_at_utc = str(get_semantics["created_at_utc"])
    get_argv = [
        "runpodctl",
        "pod",
        "get",
        pod_id,
        "--include-machine",
        "--include-network-volume",
        "-o",
        "json",
    ]
    ssh_info_argv = ["runpodctl", "ssh", "info", pod_id, "-o", "json"]
    get_binding = {
        "command_argv": get_argv,
        "file_name": "runpod-api-response.json",
        "sha256": sha256_file(raw),
    }
    ssh_info_binding = {
        "command_argv": ssh_info_argv,
        "file_name": "runpod-ssh-info-response.json",
        "sha256": sha256_file(ssh_info),
    }
    body = {
        "schema": RUNPOD_PROVISION_RECEIPT_SCHEMA,
        "schema_version": RUNPOD_PROVISION_RECEIPT_SCHEMA_VERSION,
        "provider": "runpod",
        "capture_tool": "runpodctl 2.9.0-c094cac",
        "execution_uuid": execution_uuid,
        "pod_id": pod_id,
        "pod_name": expected_pod_name,
        "image": runtime["image"],
        "gpu_count": WORKER_COUNT,
        "gpu_catalog": {
            **catalog["gpu_catalog"],
        },
        "gpu_catalog_snapshot": catalog["binding"],
        "data_center": runtime["data_center"],
        "network_volume_id": runtime["network_volume_id"],
        "network_volume_mount": runtime["network_volume_mount"],
        "created_at_utc": created_at_utc,
        "provider_allocation": {"create": create_semantics, "get": get_semantics},
        "provider_output_limitations": copy.deepcopy(
            dict(RUNPOD_PROVISIONING_CONTRACT["provider_output_limitations"])
        ),
        "operational_market_snapshot": catalog["operational_market_snapshot"],
        "runpodctl_version_capture": version,
        "api_response": {
            "capture_command": " ".join(get_argv),
            "file_name": "runpod-api-response.json",
            "sha256": sha256_file(raw),
        },
        "ssh_info_response": ssh_info_binding,
        "readiness": {
            "strategy": "no_wait_create_then_bounded_get_ssh_identity_poll",
            "timeout_seconds": RUNPOD_PROVISIONING_CONTRACT["readiness_poll_timeout_seconds"],
            "poll_interval_seconds": RUNPOD_PROVISIONING_CONTRACT["readiness_poll_interval_seconds"],
            "final_get": get_binding,
            "ssh_info": ssh_info_binding,
            "ssh_identity": identity_binding,
        },
        "provisioning": {
            "capture_order": ["version", "catalog", "create", "get", "ssh_info", "ssh_identity"],
            "version": version,
            "catalog": catalog["binding"],
            "create": {
                "command_argv": list(canonical_create_argv),
                "file_name": "runpod-create-response.json",
                "sha256": sha256_file(create_raw),
            },
            "get": get_binding,
            "ssh_info": ssh_info_binding,
            "terminate_after_utc": terminate_after_utc,
            "provision_ceiling_seconds": RUNPOD_PROVISIONING_CONTRACT["provision_ceiling_seconds"],
            "maximum_secure_cost_usd": RUNPOD_PROVISIONING_CONTRACT["maximum_secure_cost_usd"],
        },
        "evidence_boundaries": dict(RUNPOD_PROVISION_EVIDENCE),
        "outcomes_seen": False,
        "g01_launch_authorized": False,
    }
    payload = {**body, "receipt_digest": semantic_digest(body)}
    atomic_json(output, payload)
    return verify_runpod_provision_receipt(
        verified=verified,
        receipt_path=output,
        expected_receipt_sha256=sha256_file(output),
        expected_pod_id=pod_id,
        require_bound_copy=False,
    )


def bind_runpod_provision_receipt(
    *,
    verified: VerifiedFreeze,
    input_path: str | Path,
    output_path: str | Path,
    expected_receipt_sha256: str,
    expected_pod_id: str,
) -> dict[str, Any]:
    """Copy externally authenticated provision bytes once into the execution root."""

    source_direct = Path(input_path)
    target_direct = Path(output_path)
    source = source_direct.resolve()
    target = target_direct.resolve()
    if source == target:
        raise FreezeError("external and execution-bound provision receipt paths must differ")
    verify_runpod_provision_receipt(
        verified=verified,
        receipt_path=source_direct,
        expected_receipt_sha256=expected_receipt_sha256,
        expected_pod_id=expected_pod_id,
        require_bound_copy=False,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    sibling_names = (
        str(RUNPOD_PROVISIONING_CONTRACT["runpodctl_version_capture_file"]),
        str(RUNPOD_PROVISIONING_CONTRACT["gpu_catalog_capture_file"]),
        "runpod-create-response.json",
        "runpod-api-response.json",
        "runpod-ssh-info-response.json",
        "runpod-ssh-identity-receipt.json",
    )
    for name in sibling_names:
        exclusive_copy(source.parent / name, target.parent / name)
    exclusive_copy(source, target)
    return verify_runpod_provision_receipt(
        verified=verified,
        receipt_path=target_direct,
        expected_receipt_sha256=expected_receipt_sha256,
        expected_pod_id=expected_pod_id,
    )


def _selected_family_binding(verified: VerifiedFreeze, profile: str) -> dict[str, Any]:
    configurations = verified.payload.get("configurations")
    if not isinstance(configurations, Mapping):
        raise FreezeError("freeze omits H200 candidate configurations")
    bindings = configurations.get(profile)
    if not isinstance(bindings, Mapping) or set(bindings) != set(_PANEL_BASE):
        raise FreezeError("freeze omits the selected H200 profile family")
    panels: dict[str, Any] = {}
    for panel_id in sorted(_PANEL_BASE):
        row = bindings.get(panel_id)
        if not isinstance(row, Mapping):
            raise FreezeError("selected H200 profile binding is malformed")
        panels[panel_id] = {
            "canonical_config_digest": row["canonical_config_digest"],
            "config_path": row["config_path"],
            "config_sha256": row["config_sha256"],
            "plan_key_digest": row["plan_key_digest"],
            "plan_path": row["plan_path"],
            "plan_sha256": row["plan_sha256"],
            "run_count": row["run_count"],
        }
    return {
        "profile": profile,
        "contract": copy.deepcopy(dict(PROFILE_CONTRACT[profile])),
        "panels": panels,
        "family_digest": semantic_digest(panels),
    }


def _qualification_expected_bindings(
    *,
    verified: VerifiedFreeze,
    execution_root: Path,
    expected_provision_receipt_sha256: str,
    expected_pod_id: str,
) -> dict[str, Any]:
    """Reconstruct every pre-ITT identity consumed by H200 qualification."""

    provision = verify_runpod_provision_receipt(
        verified=verified,
        receipt_path=execution_root / "runpod-provision-receipt.json",
        expected_receipt_sha256=expected_provision_receipt_sha256,
        expected_pod_id=expected_pod_id,
    )
    if (
        provision["execution_uuid"] != execution_root.name
        or provision["network_volume_id"] != verified.payload["runtime"]["network_volume_id"]
    ):
        raise FreezeError("qualification differs from its exact provision execution/volume")
    provision_binding = {
        "file_sha256": provision["file_sha256"],
        "pod_id": provision["pod_id"],
        "receipt_digest": provision["receipt_digest"],
    }
    model_receipt_bindings: dict[str, Any] = {}
    integration_bindings: dict[str, Any] = {}
    for panel_id in sorted(_PANEL_BASE):
        suffix = panel_id.removeprefix("g00f-")
        model = verify_model_snapshot_receipt(
            verified=verified,
            panel_id=panel_id,
            receipt_path=execution_root / f"model-receipt-{suffix}.json",
        )
        model_receipt_bindings[panel_id] = {
            "file_sha256": model["file_sha256"],
            "panel_id": panel_id,
            "receipt_digest": model["receipt_digest"],
        }
        audit = verify_model_integration_audit(
            verified=verified,
            panel_id=panel_id,
            audit_path=execution_root / f"model-integration-audit-{suffix}.json",
            replay_model_snapshot=False,
        )
        integration_bindings[panel_id] = {
            "audit_digest": audit["audit_digest"],
            "file_sha256": audit["file_sha256"],
            "panel_id": panel_id,
            "report_digest": audit["report_digest"],
        }
    return {
        "freeze_binding": {
            "freeze_digest": verified.digest,
            "freeze_file_sha256": verified.file_sha256,
        },
        "provision": provision,
        "provision_binding": provision_binding,
        "model_receipt_bindings": model_receipt_bindings,
        "model_integration_audit_bindings": integration_bindings,
    }


def _verify_profile_qualification_files(
    *,
    verified: VerifiedFreeze,
    report_path: str | Path,
    expected_report_sha256: str,
    evidence_path: str | Path,
    expected_evidence_sha256: str,
    expected_provision_receipt_sha256: str,
    expected_pod_id: str,
) -> dict[str, Any]:
    """Replay compact raw H200 evidence and its report without trusting flags."""

    target = Path(report_path).resolve()
    execution_root = target.parent
    if target != execution_root / "h200-profile-qualification.json":
        raise FreezeError("H200 qualification report has a noncanonical execution-root path")
    expected_sha = require_sha256(expected_report_sha256, "H200 qualification report SHA-256")
    if (
        not target.is_file()
        or target.is_symlink()
        or stat.S_IMODE(target.stat().st_mode) != 0o400
        or sha256_file(target) != expected_sha
    ):
        raise FreezeError("H200 qualification report bytes changed")
    report = strict_json(target, "H200 profile qualification report")
    evidence_path = Path(evidence_path).resolve()
    expected_evidence_sha256 = require_sha256(
        expected_evidence_sha256,
        "H200 qualification evidence SHA-256",
    )
    if (
        evidence_path != execution_root / "h200-profile-qualification-evidence.json"
        or not evidence_path.is_file()
        or evidence_path.is_symlink()
        or stat.S_IMODE(evidence_path.stat().st_mode) != 0o400
        or evidence_path.stat().st_size > 512 * 1024**2
        or sha256_file(evidence_path) != expected_evidence_sha256
    ):
        raise FreezeError("H200 compact qualification evidence is absent")
    evidence = strict_json(evidence_path, "H200 compact qualification evidence")
    try:
        from .qualification import (
            QualificationError,
            create_qualification_report,
            validate_qualification_evidence,
            validate_qualification_report,
        )
    except ImportError as error:
        raise FreezeError("H200 profile qualification implementation is unavailable") from error
    try:
        validated_evidence = validate_qualification_evidence(evidence)
        validated = validate_qualification_report(report)
        recomputed_report = create_qualification_report(validated_evidence)
    except QualificationError as error:
        raise FreezeError("H200 profile qualification failed strict replay") from error
    if validated_evidence != evidence or validated != report or recomputed_report != report:
        raise FreezeError("H200 qualification normalization changed persisted evidence")
    expected_bindings = _qualification_expected_bindings(
        verified=verified,
        execution_root=execution_root,
        expected_provision_receipt_sha256=expected_provision_receipt_sha256,
        expected_pod_id=expected_pod_id,
    )
    if report.get("freeze_binding") != expected_bindings["freeze_binding"]:
        raise FreezeError("H200 qualification is not bound to this execution freeze")
    expected_provision_binding = expected_bindings["provision_binding"]
    if report.get("provision_binding") != expected_provision_binding:
        raise FreezeError("H200 qualification/provision binding changed")
    model_receipt_bindings = report.get("model_receipt_bindings")
    integration_bindings = report.get("model_integration_audit_bindings")
    if (
        not isinstance(model_receipt_bindings, Mapping)
        or set(model_receipt_bindings) != set(_PANEL_BASE)
        or not isinstance(integration_bindings, Mapping)
        or set(integration_bindings) != set(_PANEL_BASE)
    ):
        raise FreezeError("H200 qualification omits model or boundary bindings")
    for panel_id in _PANEL_BASE:
        if model_receipt_bindings.get(panel_id) != expected_bindings["model_receipt_bindings"][panel_id]:
            raise FreezeError(f"H200 qualification {panel_id} model binding changed")
        if (
            integration_bindings.get(panel_id)
            != expected_bindings["model_integration_audit_bindings"][panel_id]
        ):
            raise FreezeError(f"H200 qualification {panel_id} boundary binding changed")
    gpu_uuids = report.get("gpu_uuids")
    if (
        not isinstance(gpu_uuids, Sequence)
        or isinstance(gpu_uuids, (str, bytes))
        or len(gpu_uuids) != WORKER_COUNT
        or list(gpu_uuids) != sorted(str(value) for value in gpu_uuids)
        or len(set(str(value) for value in gpu_uuids)) != WORKER_COUNT
        or any(not str(value).startswith("GPU-") for value in gpu_uuids)
    ):
        raise FreezeError("H200 qualification does not bind four distinct GPU UUIDs")
    return {
        "evidence_digest": evidence["evidence_digest"],
        "evidence_file_sha256": expected_evidence_sha256,
        "evidence_path": str(evidence_path),
        "path": str(target),
        "file_sha256": expected_sha,
        "report_digest": report["report_digest"],
        "qualification_branch": report["qualification_branch"],
        "tuned_capacity_probe": copy.deepcopy(dict(report["tuned_capacity_probe"])),
        "eligibility": copy.deepcopy(dict(report["eligibility"])),
        "projections": copy.deepcopy(dict(report["projections"])),
        "gpu_uuids": list(gpu_uuids),
        "provision_binding": expected_provision_binding,
        "model_receipt_bindings": copy.deepcopy(dict(model_receipt_bindings)),
        "model_integration_audit_bindings": copy.deepcopy(dict(integration_bindings)),
    }


def _qualification_engineering_root(execution_root: Path, execution_uuid: str) -> Path:
    return (
        execution_root.parent.parent
        / "g00f-h200-engineering"
        / f"g00f-h200-profile-qualification-{execution_uuid}"
    ).resolve()


def _qualification_controller_argv(
    *,
    verified: VerifiedFreeze,
    execution_root: Path,
    execution_uuid: str,
    expected_provision_receipt_sha256: str,
    expected_pod_id: str,
) -> list[str]:
    try:
        from .qualification_producer import canonical_controller_argv
    except ImportError as error:
        raise FreezeError("H200 qualification producer is unavailable") from error
    return canonical_controller_argv(
        repo=verified.repo,
        freeze_path=verified.path,
        expected_freeze_sha256=verified.file_sha256,
        execution_root=execution_root,
        engineering_root=_qualification_engineering_root(execution_root, execution_uuid),
        provision_receipt_path=execution_root / "runpod-provision-receipt.json",
        expected_provision_receipt_sha256=expected_provision_receipt_sha256,
        expected_pod_id=expected_pod_id,
        model_receipt_0p5b=execution_root / "model-receipt-0p5b.json",
        model_receipt_1p5b=execution_root / "model-receipt-1p5b.json",
        integration_audit_0p5b=execution_root / "model-integration-audit-0p5b.json",
        integration_audit_1p5b=execution_root / "model-integration-audit-1p5b.json",
        execution_uuid=execution_uuid,
    )


def _qualification_producer_hashes(verified: VerifiedFreeze) -> tuple[str, str]:
    controllers = verified.payload.get("controller_files")
    if not isinstance(controllers, Mapping):
        raise FreezeError("freeze omits its qualification controller binding")
    controller = controllers.get("qualification_controller")
    if not isinstance(controller, Mapping):
        raise FreezeError("freeze omits its qualification controller binding")
    controller_sha256 = require_sha256(
        controller.get("sha256"),
        "qualification controller SHA-256",
    )
    implementation_path = verified.repo / str(QUALIFICATION_PRODUCER_CONTRACT["implementation"])
    implementation_sha256 = sha256_file(implementation_path)
    return controller_sha256, implementation_sha256


def verify_qualification_supervisor_receipts(
    *,
    verified: VerifiedFreeze,
    execution_root: str | Path,
    expected_controller_argv: Sequence[str],
) -> dict[str, Any]:
    """Replay the hard monotonic qualification process-group boundary."""

    root = Path(execution_root).resolve()
    started_path = root / "qualification-supervisor-started.json"
    terminal_path = root / "qualification-supervisor-terminal.json"
    for path in (started_path, terminal_path):
        direct = Path(path)
        metadata = direct.lstat()
        if (
            direct.is_symlink()
            or not direct.is_file()
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o400
        ):
            raise FreezeError("qualification-supervisor receipt is not one direct mode-0400 file")
    if (root / "qualification-supervisor-term.json").exists() or (
        root / "qualification-supervisor-kill.json"
    ).exists():
        raise FreezeError("successful qualification unexpectedly required TERM or KILL")
    started = strict_json(started_path, "qualification-supervisor start receipt")
    terminal = strict_json(terminal_path, "qualification-supervisor terminal receipt")
    started_body = {key: value for key, value in started.items() if key != "receipt_digest"}
    terminal_body = {key: value for key, value in terminal.items() if key != "receipt_digest"}
    common_keys = {
        "ceiling_seconds",
        "controller_argv",
        "controller_path",
        "controller_pid",
        "controller_process_group_id",
        "controller_sha256",
        "deadline_monotonic_ns",
        "execution_root",
        "execution_uuid",
        "g01_launch_authorized",
        "guardian_pid",
        "guardian_protocol",
        "outcomes_seen",
        "schema_version",
        "started_at_utc",
        "started_monotonic_ns",
        "supervisor_path",
        "supervisor_pid",
        "supervisor_sha256",
        "term_grace_seconds",
    }
    terminal_only_keys = {
        "completed_at_utc",
        "completed_monotonic_ns",
        "controller_exit_code",
        "deadline_triggered",
        "descendants_clear",
        "elapsed_seconds",
        "guardian_clean_stop",
        "guardian_exit_code",
        "guardian_failed",
        "received_signal",
        "sigkill_sent",
        "sigterm_sent",
        "success",
    }
    controllers = verified.payload.get("controller_files")
    if not isinstance(controllers, Mapping):
        raise FreezeError("freeze omits qualification-supervisor controller bindings")
    controller_binding = controllers.get("qualification_controller")
    supervisor_binding = controllers.get("qualification_supervisor")
    actual_argv = started.get("controller_argv")
    expected_python = f"/workspace/.venvs/goalzendo-h200-{root.name}/bin/python"
    started_ns = started.get("started_monotonic_ns")
    deadline_ns = started.get("deadline_monotonic_ns")
    completed_ns = terminal.get("completed_monotonic_ns")
    elapsed = terminal.get("elapsed_seconds")
    controller_pid = started.get("controller_pid")
    supervisor_pid = started.get("supervisor_pid")
    guardian_pid = started.get("guardian_pid")
    controller_exit_code = terminal.get("controller_exit_code")
    if (
        not isinstance(controller_binding, Mapping)
        or not isinstance(supervisor_binding, Mapping)
        or set(started) != common_keys | {"schema", "receipt_digest"}
        or set(terminal) != common_keys | terminal_only_keys | {"schema", "receipt_digest"}
        or started.get("schema") != "goalzendo.g00f_h200_qualification_supervisor_started"
        or terminal.get("schema") != "goalzendo.g00f_h200_qualification_supervisor_terminal"
        or started.get("schema_version") != 1
        or started.get("execution_uuid") != root.name
        or started.get("execution_root") != str(root)
        or started.get("ceiling_seconds") != 6_300
        or started.get("term_grace_seconds") != 30
        or started.get("guardian_protocol")
        != "independent_session_pipe_eof_or_monotonic_deadline_group_cleanup_v1"
        or not isinstance(actual_argv, list)
        or actual_argv != [expected_python, *list(expected_controller_argv)]
        or started.get("controller_path") != str(root / "frozen-source" / str(controller_binding.get("path")))
        or started.get("supervisor_path") != str(root / "frozen-source" / str(supervisor_binding.get("path")))
        or started.get("controller_sha256") != controller_binding.get("sha256")
        or started.get("supervisor_sha256") != supervisor_binding.get("sha256")
        or isinstance(controller_pid, bool)
        or not isinstance(controller_pid, int)
        or controller_pid <= 1
        or started.get("controller_process_group_id") != controller_pid
        or isinstance(supervisor_pid, bool)
        or not isinstance(supervisor_pid, int)
        or supervisor_pid <= 1
        or isinstance(guardian_pid, bool)
        or not isinstance(guardian_pid, int)
        or guardian_pid <= 1
        or isinstance(started_ns, bool)
        or not isinstance(started_ns, int)
        or isinstance(deadline_ns, bool)
        or not isinstance(deadline_ns, int)
        or deadline_ns != started_ns + 6_300 * 1_000_000_000
        or isinstance(completed_ns, bool)
        or not isinstance(completed_ns, int)
        or not started_ns <= completed_ns <= deadline_ns
        or isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or abs(float(elapsed) - (completed_ns - started_ns) / 1_000_000_000) > 1e-9
        or any(terminal.get(key) != started.get(key) for key in common_keys)
        or isinstance(controller_exit_code, bool)
        or not isinstance(controller_exit_code, int)
        or controller_exit_code != 0
        or terminal.get("deadline_triggered") is not False
        or terminal.get("received_signal") is not None
        or terminal.get("sigterm_sent") is not False
        or terminal.get("sigkill_sent") is not False
        or terminal.get("descendants_clear") is not True
        or terminal.get("guardian_clean_stop") is not True
        or terminal.get("guardian_exit_code") is not None
        or terminal.get("guardian_failed") is not False
        or terminal.get("success") is not True
        or started.get("outcomes_seen") is not False
        or started.get("g01_launch_authorized") is not False
        or started.get("receipt_digest") != semantic_digest(started_body)
        or terminal.get("receipt_digest") != semantic_digest(terminal_body)
    ):
        raise FreezeError("qualification supervisor does not prove bounded clean completion")
    _canonical_utc(str(started.get("started_at_utc", "")), "qualification start UTC")
    _canonical_utc(str(terminal.get("completed_at_utc", "")), "qualification completion UTC")
    return {
        "started": {
            "path": str(started_path),
            "file_sha256": sha256_file(started_path),
            "receipt_digest": started["receipt_digest"],
        },
        "terminal": {
            "path": str(terminal_path),
            "file_sha256": sha256_file(terminal_path),
            "receipt_digest": terminal["receipt_digest"],
        },
        "completed_monotonic_ns": completed_ns,
        "started_monotonic_ns": started_ns,
        "deadline_monotonic_ns": deadline_ns,
        "success": True,
        "outcomes_seen": False,
        "g01_launch_authorized": False,
    }


def verify_detached_supervisor_receipts(
    *,
    verified: VerifiedFreeze,
    started_receipt_path: str | Path,
    terminal_receipt_path: str | Path,
    expected_execution_uuid: str,
    expected_execution_root: str | Path,
    expected_operator_handoff_sha256: str,
    expected_execution_handoff_sha256: str,
    expected_launcher_path: str | Path,
    expected_supervisor_path: str | Path,
) -> dict[str, Any]:
    """Replay the persistent launcher supervisor and require clean completion."""

    try:
        parsed_uuid = uuid.UUID(expected_execution_uuid)
    except ValueError as error:
        raise FreezeError("detached-supervisor execution UUID is malformed") from error
    if parsed_uuid.version != 4 or str(parsed_uuid) != expected_execution_uuid:
        raise FreezeError("detached-supervisor execution UUID is not canonical UUID4")
    execution_uuid = expected_execution_uuid
    execution_root = Path(expected_execution_root).resolve()
    expected_root = Path("/workspace/status-goalzendo/g00f-executions") / execution_uuid
    if execution_root != expected_root:
        raise FreezeError("detached-supervisor execution root differs from its UUID")
    operator_sha256 = require_sha256(
        expected_operator_handoff_sha256,
        "detached-supervisor operator-handoff SHA-256",
    )
    execution_sha256 = require_sha256(
        expected_execution_handoff_sha256,
        "detached-supervisor execution-handoff SHA-256",
    )
    started_path = Path(started_receipt_path)
    terminal_path = Path(terminal_receipt_path)
    for path in (started_path, terminal_path):
        if not path.exists() or path.is_symlink():
            raise FreezeError("detached-supervisor receipt is absent or indirect")
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o400
        ):
            raise FreezeError("detached-supervisor receipt is not one direct mode-0400 file")
    started = strict_json(started_path, "detached-supervisor start receipt")
    terminal = strict_json(terminal_path, "detached-supervisor terminal receipt")
    started_body = {key: value for key, value in started.items() if key != "receipt_digest"}
    terminal_body = {key: value for key, value in terminal.items() if key != "receipt_digest"}
    common_keys = {
        "execution_handoff_sha256",
        "execution_root",
        "execution_uuid",
        "g01_launch_authorized",
        "guardian_pid",
        "guardian_protocol",
        "launcher_path",
        "launcher_pid",
        "launcher_process_group_id",
        "launcher_sha256",
        "operator_handoff_sha256",
        "outcomes_seen",
        "schema_version",
        "started_at_utc",
        "started_monotonic_ns",
        "supervisor_path",
        "supervisor_pid",
        "supervisor_sha256",
        "term_grace_seconds",
    }
    terminal_only_keys = {
        "completed_at_utc",
        "completed_monotonic_ns",
        "descendants_clear",
        "elapsed_seconds",
        "guardian_clean_stop",
        "guardian_exit_code",
        "guardian_failed",
        "launcher_exit_code",
        "received_signal",
        "sigkill_sent",
        "sigterm_sent",
        "success",
    }
    controllers = verified.payload.get("controller_files")
    if not isinstance(controllers, Mapping):
        raise FreezeError("freeze omits detached-supervisor controller bindings")
    launcher_binding = controllers.get("launcher")
    supervisor_binding = controllers.get("detached_supervisor")
    launcher_direct = Path(expected_launcher_path)
    supervisor_direct = Path(expected_supervisor_path)
    if (
        launcher_direct.is_symlink()
        or supervisor_direct.is_symlink()
        or not launcher_direct.is_file()
        or not supervisor_direct.is_file()
    ):
        raise FreezeError("detached-supervisor runtime controller is absent or indirect")
    launcher_path = launcher_direct.resolve()
    supervisor_path = supervisor_direct.resolve()
    started_ns = started.get("started_monotonic_ns")
    completed_ns = terminal.get("completed_monotonic_ns")
    elapsed = terminal.get("elapsed_seconds")
    launcher_pid = started.get("launcher_pid")
    supervisor_pid = started.get("supervisor_pid")
    guardian_pid = started.get("guardian_pid")
    launcher_exit_code = terminal.get("launcher_exit_code")
    if (
        not isinstance(launcher_binding, Mapping)
        or not isinstance(supervisor_binding, Mapping)
        or set(started) != common_keys | {"schema", "receipt_digest"}
        or set(terminal) != common_keys | terminal_only_keys | {"schema", "receipt_digest"}
        or started.get("schema") != "goalzendo.g00f_h200_detached_supervisor_started"
        or terminal.get("schema") != "goalzendo.g00f_h200_detached_supervisor_terminal"
        or started.get("schema_version") != 1
        or started.get("execution_uuid") != execution_uuid
        or started.get("execution_root") != str(execution_root)
        or started.get("operator_handoff_sha256") != operator_sha256
        or started.get("execution_handoff_sha256") != execution_sha256
        or started.get("launcher_path") != str(launcher_path)
        or started.get("supervisor_path") != str(supervisor_path)
        or started.get("launcher_sha256") != launcher_binding.get("sha256")
        or started.get("supervisor_sha256") != supervisor_binding.get("sha256")
        or sha256_file(launcher_path) != launcher_binding.get("sha256")
        or sha256_file(supervisor_path) != supervisor_binding.get("sha256")
        or started.get("term_grace_seconds") != 90
        or started.get("guardian_protocol") != "ready_pipe_plus_parent_eof_launcher_group_cleanup_v1"
        or isinstance(launcher_pid, bool)
        or not isinstance(launcher_pid, int)
        or launcher_pid <= 1
        or started.get("launcher_process_group_id") != launcher_pid
        or isinstance(supervisor_pid, bool)
        or not isinstance(supervisor_pid, int)
        or supervisor_pid <= 1
        or isinstance(guardian_pid, bool)
        or not isinstance(guardian_pid, int)
        or guardian_pid <= 1
        or isinstance(started_ns, bool)
        or not isinstance(started_ns, int)
        or isinstance(completed_ns, bool)
        or not isinstance(completed_ns, int)
        or completed_ns < started_ns
        or isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or abs(float(elapsed) - (completed_ns - started_ns) / 1_000_000_000) > 1e-9
        or any(terminal.get(key) != started.get(key) for key in common_keys)
        or isinstance(launcher_exit_code, bool)
        or not isinstance(launcher_exit_code, int)
        or launcher_exit_code != 0
        or terminal.get("received_signal") is not None
        or terminal.get("sigterm_sent") is not False
        or terminal.get("sigkill_sent") is not False
        or terminal.get("descendants_clear") is not True
        or terminal.get("guardian_clean_stop") is not True
        or terminal.get("guardian_exit_code") is not None
        or terminal.get("guardian_failed") is not False
        or terminal.get("success") is not True
        or started.get("outcomes_seen") is not False
        or started.get("g01_launch_authorized") is not False
        or started.get("receipt_digest") != semantic_digest(started_body)
        or terminal.get("receipt_digest") != semantic_digest(terminal_body)
    ):
        raise FreezeError("detached supervisor does not prove a clean launcher completion")
    _canonical_utc(str(started.get("started_at_utc", "")), "detached supervisor start UTC")
    _canonical_utc(str(terminal.get("completed_at_utc", "")), "detached supervisor completion UTC")
    return {
        "started": {
            "path": str(started_path.resolve()),
            "file_sha256": sha256_file(started_path),
            "receipt_digest": started["receipt_digest"],
        },
        "terminal": {
            "path": str(terminal_path.resolve()),
            "file_sha256": sha256_file(terminal_path),
            "receipt_digest": terminal["receipt_digest"],
        },
        "launcher_exit_code": 0,
        "success": True,
        "g01_launch_authorized": False,
    }


def _storage_preflight_paths(
    *,
    execution_uuid: str,
    execution_root: Path,
    phase: str,
) -> tuple[Path, Path, Path, bool]:
    if phase not in STORAGE_PREFLIGHT_CONTRACT["phases"]:
        raise FreezeError("storage preflight phase is not frozen")
    try:
        parsed_uuid = uuid.UUID(execution_uuid)
    except (ValueError, AttributeError) as error:
        raise FreezeError("storage preflight execution UUID is invalid") from error
    expected_root = Path("/workspace/status-goalzendo/g00f-executions") / execution_uuid
    if parsed_uuid.version != 4 or str(parsed_uuid) != execution_uuid or execution_root != expected_root:
        raise FreezeError("storage preflight execution identity changed")
    engineering_root = _qualification_engineering_root(execution_root, execution_uuid)
    engineering_absent = phase == "before_qualification"
    observed_engineering_path = engineering_root.parent if engineering_absent else engineering_root
    receipt_name = str(STORAGE_PREFLIGHT_CONTRACT["receipt_names"][phase])
    return (
        execution_root / receipt_name,
        engineering_root,
        observed_engineering_path,
        engineering_absent,
    )


def _measure_storage_preflight(
    *,
    execution_root: Path,
    engineering_root: Path,
    observed_engineering_path: Path,
    engineering_absent: bool,
) -> dict[str, Any]:
    mount_direct = Path(str(STORAGE_PREFLIGHT_CONTRACT["mount_root"]))
    if mount_direct.is_symlink() or not mount_direct.is_dir() or not os.path.ismount(mount_direct):
        raise FreezeError("storage preflight mount root is absent, indirect, or not a mountpoint")
    if execution_root.is_symlink() or not execution_root.is_dir():
        raise FreezeError("storage preflight execution root is absent or indirect")
    if engineering_root.exists() != (not engineering_absent) or engineering_root.is_symlink():
        raise FreezeError("storage preflight engineering-root phase state changed")
    if observed_engineering_path.is_symlink() or not observed_engineering_path.is_dir():
        raise FreezeError("storage preflight engineering observation path is absent or indirect")
    mount_metadata = mount_direct.stat()
    execution_metadata = execution_root.stat()
    engineering_metadata = observed_engineering_path.stat()
    if not (mount_metadata.st_dev == execution_metadata.st_dev == engineering_metadata.st_dev):
        raise FreezeError("execution and qualification engineering roots are not on /workspace")
    filesystem = os.statvfs(mount_direct)
    filesystem_id = getattr(filesystem, "f_fsid", None)
    numeric_values = {
        "block_size_bytes": filesystem.f_bsize,
        "fragment_size_bytes": filesystem.f_frsize,
        "blocks_available": filesystem.f_bavail,
        "blocks_free": filesystem.f_bfree,
        "blocks_total": filesystem.f_blocks,
        "inodes_available": filesystem.f_favail,
        "inodes_free": filesystem.f_ffree,
        "inodes_total": filesystem.f_files,
        "filesystem_id": filesystem_id,
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in numeric_values.values()
    ):
        raise FreezeError("storage preflight statvfs result is incomplete")
    free_bytes = filesystem.f_bavail * filesystem.f_frsize
    if free_bytes < int(STORAGE_PREFLIGHT_CONTRACT["minimum_free_bytes"]) or filesystem.f_favail < int(
        STORAGE_PREFLIGHT_CONTRACT["minimum_free_inodes"]
    ):
        raise FreezeError("/workspace lacks the frozen qualification byte or inode reserve")
    return {
        "mount_root": str(mount_direct),
        "mount_is_mountpoint": True,
        "mount_device": mount_metadata.st_dev,
        "mount_device_major": os.major(mount_metadata.st_dev),
        "mount_device_minor": os.minor(mount_metadata.st_dev),
        "filesystem_id": filesystem_id,
        "block_size_bytes": filesystem.f_bsize,
        "fragment_size_bytes": filesystem.f_frsize,
        "blocks_available": filesystem.f_bavail,
        "blocks_free": filesystem.f_bfree,
        "blocks_total": filesystem.f_blocks,
        "free_bytes": free_bytes,
        "inodes_available": filesystem.f_favail,
        "inodes_free": filesystem.f_ffree,
        "inodes_total": filesystem.f_files,
        "execution_root": str(execution_root),
        "execution_root_device": execution_metadata.st_dev,
        "engineering_root": str(engineering_root),
        "engineering_observed_path": str(observed_engineering_path),
        "engineering_observed_device": engineering_metadata.st_dev,
        "engineering_root_absent": engineering_absent,
        "same_filesystem": True,
    }


def create_storage_preflight_receipt(
    *,
    verified: VerifiedFreeze,
    execution_uuid: str,
    execution_root: str | Path,
    phase: str,
    expected_provision_receipt_sha256: str,
    expected_pod_id: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Measure and seal the exact /workspace byte/inode reserve before GPU work or ITT."""

    root = Path(execution_root).resolve()
    target = Path(output_path).resolve()
    expected_target, engineering_root, observed_engineering_path, engineering_absent = (
        _storage_preflight_paths(
            execution_uuid=execution_uuid,
            execution_root=root,
            phase=phase,
        )
    )
    if target != expected_target or (root / "itt-ledger").exists():
        raise FreezeError("storage preflight receipt path or pre-ITT timing changed")
    provision = verify_runpod_provision_receipt(
        verified=verified,
        receipt_path=root / "runpod-provision-receipt.json",
        expected_receipt_sha256=expected_provision_receipt_sha256,
        expected_pod_id=expected_pod_id,
    )
    if (
        provision["execution_uuid"] != execution_uuid
        or provision["network_volume_id"] != verified.payload["runtime"]["network_volume_id"]
    ):
        raise FreezeError("storage preflight provision execution/volume binding changed")
    pre_itt_bindings: dict[str, Any] | None = None
    if phase == "before_itt":
        unselected_verified = replace(
            verified,
            plans={},
            selected_profile=None,
            selection_receipt=None,
        )
        selection_path = root / "h200-profile-selection.json"
        selected = verify_profile_selection_receipt(
            verified=unselected_verified,
            receipt_path=selection_path,
            expected_receipt_sha256=sha256_file(selection_path),
            expected_provision_receipt_sha256=expected_provision_receipt_sha256,
            expected_pod_id=expected_pod_id,
        )
        assert selected.selection_receipt is not None
        pre_itt_bindings = {
            "qualification_supervisor": copy.deepcopy(
                dict(selected.selection_receipt["qualification_handoff"]["qualification_supervisor"])
            ),
            "qualification_handoff": copy.deepcopy(dict(selected.selection_receipt["qualification_handoff"])),
            "profile_selection": copy.deepcopy(dict(selected.selection_receipt)),
        }
    observation = _measure_storage_preflight(
        execution_root=root,
        engineering_root=engineering_root,
        observed_engineering_path=observed_engineering_path,
        engineering_absent=engineering_absent,
    )
    previous_preflight: dict[str, Any] | None = None
    if phase == "before_itt":
        previous_path = root / str(STORAGE_PREFLIGHT_CONTRACT["receipt_names"]["before_qualification"])
        previous_preflight = verify_storage_preflight_receipt(
            verified=verified,
            execution_uuid=execution_uuid,
            execution_root=root,
            phase="before_qualification",
            receipt_path=previous_path,
            expected_receipt_sha256=sha256_file(previous_path),
            expected_provision_receipt_sha256=expected_provision_receipt_sha256,
            expected_pod_id=expected_pod_id,
        )
        if (
            previous_preflight["mount_device"] != observation["mount_device"]
            or previous_preflight["filesystem_id"] != observation["filesystem_id"]
            or previous_preflight["measured_monotonic_ns"] > time.monotonic_ns()
        ):
            raise FreezeError("storage preflight phases do not bind one ordered filesystem")
    body = {
        "schema": STORAGE_PREFLIGHT_SCHEMA,
        "schema_version": STORAGE_PREFLIGHT_SCHEMA_VERSION,
        "study_id": "g00f",
        "execution_uuid": execution_uuid,
        "execution_root": str(root),
        "phase": phase,
        "freeze_file_sha256": verified.file_sha256,
        "freeze_digest": verified.digest,
        "contract": copy.deepcopy(dict(STORAGE_PREFLIGHT_CONTRACT)),
        "measured_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "measured_monotonic_ns": time.monotonic_ns(),
        "observation": observation,
        "previous_preflight": previous_preflight,
        "runpod_provision": provision,
        "network_volume_id": verified.payload["runtime"]["network_volume_id"],
        "pre_itt_bindings": pre_itt_bindings,
        "thresholds_passed": True,
        "outcomes_seen": False,
        "g01_launch_authorized": False,
    }
    payload = {**body, "receipt_digest": semantic_digest(body)}
    exclusive_json(target, payload)
    os.chmod(target, 0o400)
    return verify_storage_preflight_receipt(
        verified=verified,
        execution_uuid=execution_uuid,
        execution_root=root,
        phase=phase,
        receipt_path=target,
        expected_receipt_sha256=sha256_file(target),
        expected_provision_receipt_sha256=expected_provision_receipt_sha256,
        expected_pod_id=expected_pod_id,
    )


def verify_storage_preflight_receipt(
    *,
    verified: VerifiedFreeze,
    execution_uuid: str,
    execution_root: str | Path,
    phase: str,
    receipt_path: str | Path,
    expected_receipt_sha256: str,
    expected_provision_receipt_sha256: str,
    expected_pod_id: str,
) -> dict[str, Any]:
    """Replay one immutable, time-specific storage measurement without reinterpreting it."""

    root = Path(execution_root).resolve()
    target = Path(receipt_path).resolve()
    expected_target, engineering_root, observed_engineering_path, engineering_absent = (
        _storage_preflight_paths(
            execution_uuid=execution_uuid,
            execution_root=root,
            phase=phase,
        )
    )
    expected_sha256 = require_sha256(expected_receipt_sha256, "storage preflight SHA-256")
    if (
        target != expected_target
        or not target.is_file()
        or target.is_symlink()
        or target.stat().st_nlink != 1
        or stat.S_IMODE(target.stat().st_mode) != 0o400
        or sha256_file(target) != expected_sha256
    ):
        raise FreezeError("storage preflight receipt bytes, path, or mode changed")
    payload = strict_json(target, "H200 storage preflight receipt")
    body = {key: value for key, value in payload.items() if key != "receipt_digest"}
    observation = payload.get("observation")
    previous_binding = payload.get("previous_preflight")
    pre_itt_binding = payload.get("pre_itt_bindings")
    observation_keys = {
        "block_size_bytes",
        "blocks_available",
        "blocks_free",
        "blocks_total",
        "engineering_observed_device",
        "engineering_observed_path",
        "engineering_root",
        "engineering_root_absent",
        "execution_root",
        "execution_root_device",
        "filesystem_id",
        "fragment_size_bytes",
        "free_bytes",
        "inodes_available",
        "inodes_free",
        "inodes_total",
        "mount_device",
        "mount_device_major",
        "mount_device_minor",
        "mount_is_mountpoint",
        "mount_root",
        "same_filesystem",
    }
    numeric_keys = observation_keys - {
        "engineering_observed_path",
        "engineering_root",
        "engineering_root_absent",
        "execution_root",
        "mount_is_mountpoint",
        "mount_root",
        "same_filesystem",
    }
    measured_ns = payload.get("measured_monotonic_ns")
    provision_binding = payload.get("runpod_provision")
    if not isinstance(provision_binding, Mapping):
        raise FreezeError("storage preflight omits its exact Runpod allocation")
    provision = verify_runpod_provision_receipt(
        verified=verified,
        receipt_path=root / "runpod-provision-receipt.json",
        expected_receipt_sha256=expected_provision_receipt_sha256,
        expected_pod_id=expected_pod_id,
    )
    if (
        provision["execution_uuid"] != execution_uuid
        or provision["network_volume_id"] != verified.payload["runtime"]["network_volume_id"]
    ):
        raise FreezeError("storage preflight provision execution/volume binding changed")
    if (
        set(payload)
        != {
            "contract",
            "execution_root",
            "execution_uuid",
            "freeze_digest",
            "freeze_file_sha256",
            "g01_launch_authorized",
            "measured_at_utc",
            "measured_monotonic_ns",
            "observation",
            "outcomes_seen",
            "phase",
            "previous_preflight",
            "runpod_provision",
            "network_volume_id",
            "pre_itt_bindings",
            "receipt_digest",
            "schema",
            "schema_version",
            "study_id",
            "thresholds_passed",
        }
        or payload.get("schema") != STORAGE_PREFLIGHT_SCHEMA
        or payload.get("schema_version") != STORAGE_PREFLIGHT_SCHEMA_VERSION
        or payload.get("study_id") != "g00f"
        or payload.get("execution_uuid") != execution_uuid
        or payload.get("execution_root") != str(root)
        or payload.get("phase") != phase
        or payload.get("freeze_file_sha256") != verified.file_sha256
        or payload.get("freeze_digest") != verified.digest
        or payload.get("contract") != STORAGE_PREFLIGHT_CONTRACT
        or provision_binding != provision
        or payload.get("network_volume_id") != verified.payload["runtime"]["network_volume_id"]
        or (phase == "before_qualification" and previous_binding is not None)
        or (phase == "before_itt" and not isinstance(previous_binding, Mapping))
        or (phase == "before_qualification" and pre_itt_binding is not None)
        or (phase == "before_itt" and not isinstance(pre_itt_binding, Mapping))
        or isinstance(measured_ns, bool)
        or not isinstance(measured_ns, int)
        or measured_ns <= 0
        or not isinstance(observation, Mapping)
        or set(observation) != observation_keys
        or any(
            isinstance(observation.get(key), bool)
            or not isinstance(observation.get(key), int)
            or int(observation[key]) < 0
            for key in numeric_keys
        )
        or observation.get("mount_root") != STORAGE_PREFLIGHT_CONTRACT["mount_root"]
        or observation.get("mount_is_mountpoint") is not True
        or observation.get("execution_root") != str(root)
        or observation.get("engineering_root") != str(engineering_root)
        or observation.get("engineering_observed_path") != str(observed_engineering_path)
        or observation.get("engineering_root_absent") is not engineering_absent
        or observation.get("same_filesystem") is not True
        or observation.get("mount_device") != observation.get("execution_root_device")
        or observation.get("mount_device") != observation.get("engineering_observed_device")
        or observation.get("free_bytes")
        != int(observation.get("blocks_available", -1)) * int(observation.get("fragment_size_bytes", -1))
        or int(observation.get("free_bytes", -1)) < int(STORAGE_PREFLIGHT_CONTRACT["minimum_free_bytes"])
        or int(observation.get("inodes_available", -1))
        < int(STORAGE_PREFLIGHT_CONTRACT["minimum_free_inodes"])
        or payload.get("thresholds_passed") is not True
        or payload.get("outcomes_seen") is not False
        or payload.get("g01_launch_authorized") is not False
        or payload.get("receipt_digest") != semantic_digest(body)
    ):
        raise FreezeError("storage preflight receipt violates the frozen byte/inode contract")
    _canonical_utc(str(payload.get("measured_at_utc", "")), "storage preflight measurement UTC")
    previous_preflight: dict[str, Any] | None = None
    if phase == "before_itt":
        assert isinstance(previous_binding, Mapping)
        previous_path = root / str(STORAGE_PREFLIGHT_CONTRACT["receipt_names"]["before_qualification"])
        previous_preflight = verify_storage_preflight_receipt(
            verified=verified,
            execution_uuid=execution_uuid,
            execution_root=root,
            phase="before_qualification",
            receipt_path=previous_path,
            expected_receipt_sha256=str(previous_binding.get("file_sha256", "")),
            expected_provision_receipt_sha256=expected_provision_receipt_sha256,
            expected_pod_id=expected_pod_id,
        )
        if (
            dict(previous_binding) != previous_preflight
            or previous_preflight["mount_device"] != observation["mount_device"]
            or previous_preflight["filesystem_id"] != observation["filesystem_id"]
            or previous_preflight["measured_monotonic_ns"] > measured_ns
        ):
            raise FreezeError("storage preflight phase cross-binding changed")
    pre_itt_bindings: dict[str, Any] | None = None
    if phase == "before_itt":
        assert isinstance(pre_itt_binding, Mapping)
        unselected_verified = replace(
            verified,
            plans={},
            selected_profile=None,
            selection_receipt=None,
        )
        selection_binding = pre_itt_binding.get("profile_selection")
        if not isinstance(selection_binding, Mapping):
            raise FreezeError("pre-ITT storage receipt omits profile selection")
        selected = verify_profile_selection_receipt(
            verified=unselected_verified,
            receipt_path=root / "h200-profile-selection.json",
            expected_receipt_sha256=str(selection_binding.get("file_sha256", "")),
            expected_provision_receipt_sha256=expected_provision_receipt_sha256,
            expected_pod_id=expected_pod_id,
        )
        assert selected.selection_receipt is not None
        pre_itt_bindings = {
            "qualification_supervisor": copy.deepcopy(
                dict(selected.selection_receipt["qualification_handoff"]["qualification_supervisor"])
            ),
            "qualification_handoff": copy.deepcopy(dict(selected.selection_receipt["qualification_handoff"])),
            "profile_selection": copy.deepcopy(dict(selected.selection_receipt)),
        }
        if (
            dict(pre_itt_binding) != pre_itt_bindings
            or selected.selection_receipt["selected_monotonic_ns"] > measured_ns
            or selected.selection_receipt["qualification_handoff"]["qualification_supervisor"][
                "completed_monotonic_ns"
            ]
            > selected.selection_receipt["selected_monotonic_ns"]
        ):
            raise FreezeError("pre-ITT storage receipt ordering or exact bindings changed")
    return {
        "path": str(target),
        "file_sha256": expected_sha256,
        "receipt_digest": payload["receipt_digest"],
        "execution_uuid": execution_uuid,
        "phase": phase,
        "measured_at_utc": payload["measured_at_utc"],
        "measured_monotonic_ns": measured_ns,
        "mount_device": observation["mount_device"],
        "filesystem_id": observation["filesystem_id"],
        "free_bytes": observation["free_bytes"],
        "inodes_available": observation["inodes_available"],
        "thresholds_passed": True,
        "previous_preflight": previous_preflight,
        "runpod_provision": provision,
        "network_volume_id": verified.payload["runtime"]["network_volume_id"],
        "pre_itt_bindings": pre_itt_bindings,
        "outcomes_seen": False,
        "g01_launch_authorized": False,
    }


def create_profile_qualification_handoff(
    *,
    verified: VerifiedFreeze,
    execution_uuid: str,
    producer_receipt_path: str | Path,
    expected_producer_receipt_sha256: str,
    expected_provision_receipt_sha256: str,
    expected_pod_id: str,
    observed_gpu_uuids: Sequence[str],
    output_path: str | Path,
) -> dict[str, Any]:
    """Authenticate producer output and copy compact evidence into the execution root."""

    if verified.selected_profile is not None or verified.selection_receipt is not None:
        raise FreezeError("qualification handoff must precede H200 profile selection")
    try:
        parsed_uuid = uuid.UUID(execution_uuid)
    except (ValueError, AttributeError) as error:
        raise FreezeError("qualification handoff execution UUID is invalid") from error
    if parsed_uuid.version != 4 or str(parsed_uuid) != execution_uuid:
        raise FreezeError("qualification handoff execution UUID must be a canonical UUID4")
    target = Path(output_path).resolve()
    execution_root = target.parent
    if target != execution_root / "h200-profile-qualification-handoff.json":
        raise FreezeError("qualification handoff has a noncanonical execution-root path")
    if execution_root.name != execution_uuid or (execution_root / "itt-ledger").exists():
        raise FreezeError("qualification handoff must use the fresh pre-ITT execution root")
    engineering_root = _qualification_engineering_root(execution_root, execution_uuid)
    source_receipt = Path(producer_receipt_path).resolve()
    if source_receipt != engineering_root / "qualification-producer-receipt.json":
        raise FreezeError("qualification producer receipt has a noncanonical engineering path")
    expected_producer_sha256 = require_sha256(
        expected_producer_receipt_sha256,
        "qualification producer receipt SHA-256",
    )
    if sha256_file(source_receipt) != expected_producer_sha256:
        raise FreezeError("qualification producer receipt differs from its pinned SHA-256")
    expected_bindings = _qualification_expected_bindings(
        verified=verified,
        execution_root=execution_root,
        expected_provision_receipt_sha256=expected_provision_receipt_sha256,
        expected_pod_id=expected_pod_id,
    )
    controller_argv = _qualification_controller_argv(
        verified=verified,
        execution_root=execution_root,
        execution_uuid=execution_uuid,
        expected_provision_receipt_sha256=expected_provision_receipt_sha256,
        expected_pod_id=expected_pod_id,
    )
    controller_sha256, implementation_sha256 = _qualification_producer_hashes(verified)
    qualification_supervisor = verify_qualification_supervisor_receipts(
        verified=verified,
        execution_root=execution_root,
        expected_controller_argv=controller_argv,
    )
    storage_before_qualification_path = execution_root / str(
        STORAGE_PREFLIGHT_CONTRACT["receipt_names"]["before_qualification"]
    )
    storage_before_qualification = verify_storage_preflight_receipt(
        verified=verified,
        execution_uuid=execution_uuid,
        execution_root=execution_root,
        phase="before_qualification",
        receipt_path=storage_before_qualification_path,
        expected_receipt_sha256=sha256_file(storage_before_qualification_path),
        expected_provision_receipt_sha256=expected_provision_receipt_sha256,
        expected_pod_id=expected_pod_id,
    )
    if (
        storage_before_qualification["measured_monotonic_ns"]
        > qualification_supervisor["started_monotonic_ns"]
    ):
        raise FreezeError("storage preflight did not precede qualification")
    canonical_gpu_uuids = sorted(str(value) for value in observed_gpu_uuids)
    if (
        len(canonical_gpu_uuids) != WORKER_COUNT
        or len(set(canonical_gpu_uuids)) != WORKER_COUNT
        or any(not value.startswith("GPU-") for value in canonical_gpu_uuids)
    ):
        raise FreezeError("qualification handoff requires four distinct observed GPU UUIDs")
    try:
        from .qualification_producer import ProducerError, verify_producer_receipt
    except ImportError as error:
        raise FreezeError("qualification producer implementation is unavailable") from error
    try:
        source = verify_producer_receipt(
            receipt_path=source_receipt,
            expected_controller_sha256=controller_sha256,
            expected_implementation_sha256=implementation_sha256,
            expected_controller_argv=controller_argv,
            expected_freeze_binding=expected_bindings["freeze_binding"],
            expected_provision_binding=expected_bindings["provision_binding"],
            expected_model_receipt_bindings=expected_bindings["model_receipt_bindings"],
            expected_integration_audit_bindings=expected_bindings["model_integration_audit_bindings"],
            expected_gpu_uuids=canonical_gpu_uuids,
        )
    except ProducerError as error:
        raise FreezeError("qualification producer receipt failed authenticated replay") from error
    if source.get("execution_uuid") != execution_uuid:
        raise FreezeError("qualification producer receipt belongs to a different execution")
    source_artifacts = source.get("artifacts")
    if not isinstance(source_artifacts, Mapping) or set(source_artifacts) != set(
        PROFILE_QUALIFICATION_COPY_NAMES
    ):
        raise FreezeError("qualification producer returned an incomplete compact artifact set")

    copied_paths = {kind: execution_root / name for kind, name in PROFILE_QUALIFICATION_COPY_NAMES.items()}
    for kind in ("evidence", "report", "transient_inventory", "producer_receipt"):
        binding = source_artifacts.get(kind)
        if not isinstance(binding, Mapping):
            raise FreezeError(f"qualification producer omitted its {kind} binding")
        exclusive_copy(str(binding.get("path", "")), copied_paths[kind], mode=0o400)
    try:
        from .qualification_producer import replay_producer_receipt
    except ImportError as error:
        raise FreezeError("qualification producer replay is unavailable") from error
    try:
        replayed = replay_producer_receipt(
            receipt_path=copied_paths["producer_receipt"],
            expected_controller_sha256=controller_sha256,
            expected_implementation_sha256=implementation_sha256,
            expected_controller_argv=controller_argv,
            expected_freeze_binding=expected_bindings["freeze_binding"],
            expected_provision_binding=expected_bindings["provision_binding"],
            expected_model_receipt_bindings=expected_bindings["model_receipt_bindings"],
            expected_integration_audit_bindings=expected_bindings["model_integration_audit_bindings"],
            expected_gpu_uuids=canonical_gpu_uuids,
            artifact_path_overrides={
                kind: copied_paths[kind] for kind in ("evidence", "report", "transient_inventory")
            },
        )
    except ProducerError as error:
        raise FreezeError("copied qualification artifacts failed independent replay") from error
    copied_artifacts = replayed.get("artifacts")
    if not isinstance(copied_artifacts, Mapping):
        raise FreezeError("qualification replay omitted copied artifact bindings")
    qualification = _verify_profile_qualification_files(
        verified=verified,
        report_path=copied_paths["report"],
        expected_report_sha256=str(copied_artifacts["report"]["file_sha256"]),
        evidence_path=copied_paths["evidence"],
        expected_evidence_sha256=str(copied_artifacts["evidence"]["file_sha256"]),
        expected_provision_receipt_sha256=expected_provision_receipt_sha256,
        expected_pod_id=expected_pod_id,
    )
    body = {
        "schema": PROFILE_QUALIFICATION_HANDOFF_SCHEMA,
        "schema_version": PROFILE_QUALIFICATION_HANDOFF_SCHEMA_VERSION,
        "study_id": "g00f",
        "execution_uuid": execution_uuid,
        "freeze_file_sha256": verified.file_sha256,
        "freeze_digest": verified.digest,
        "producer_contract": copy.deepcopy(dict(QUALIFICATION_PRODUCER_CONTRACT)),
        "controller_argv": controller_argv,
        "qualification_supervisor": qualification_supervisor,
        "storage_preflight_before_qualification": storage_before_qualification,
        "source_producer_receipt_file_sha256": expected_producer_sha256,
        "copied_artifacts": copy.deepcopy(dict(copied_artifacts)),
        "qualification": qualification,
        "gpu_uuids": canonical_gpu_uuids,
        "ledger_absent_at_creation": True,
        "compact_evidence_retained_through_final_gate": True,
        "outcomes_seen": False,
        "g01_launch_authorized": False,
    }
    payload = {**body, "handoff_digest": semantic_digest(body)}
    exclusive_json(target, payload)
    os.chmod(target, 0o400)
    return verify_profile_qualification_handoff(
        verified=verified,
        handoff_path=target,
        expected_handoff_sha256=sha256_file(target),
        expected_provision_receipt_sha256=expected_provision_receipt_sha256,
        expected_pod_id=expected_pod_id,
    )


def verify_profile_qualification_handoff(
    *,
    verified: VerifiedFreeze,
    handoff_path: str | Path,
    expected_handoff_sha256: str,
    expected_provision_receipt_sha256: str,
    expected_pod_id: str,
) -> dict[str, Any]:
    """Replay only the immutable execution-root copies made by the producer handoff."""

    target = Path(handoff_path).resolve()
    execution_root = target.parent
    if target != execution_root / "h200-profile-qualification-handoff.json":
        raise FreezeError("qualification handoff has a noncanonical execution-root path")
    expected_sha256 = require_sha256(expected_handoff_sha256, "qualification handoff SHA-256")
    if (
        not target.is_file()
        or target.is_symlink()
        or stat.S_IMODE(target.stat().st_mode) != 0o400
        or sha256_file(target) != expected_sha256
    ):
        raise FreezeError("qualification handoff bytes or mode changed")
    payload = strict_json(target, "H200 profile qualification handoff")
    body = {key: value for key, value in payload.items() if key != "handoff_digest"}
    execution_uuid = str(payload.get("execution_uuid", ""))
    try:
        parsed_uuid = uuid.UUID(execution_uuid)
    except (ValueError, AttributeError) as error:
        raise FreezeError("qualification handoff execution UUID is invalid") from error
    if (
        parsed_uuid.version != 4
        or str(parsed_uuid) != execution_uuid
        or execution_root.name != execution_uuid
    ):
        raise FreezeError("qualification handoff execution identity changed")
    expected_bindings = _qualification_expected_bindings(
        verified=verified,
        execution_root=execution_root,
        expected_provision_receipt_sha256=expected_provision_receipt_sha256,
        expected_pod_id=expected_pod_id,
    )
    controller_argv = _qualification_controller_argv(
        verified=verified,
        execution_root=execution_root,
        execution_uuid=execution_uuid,
        expected_provision_receipt_sha256=expected_provision_receipt_sha256,
        expected_pod_id=expected_pod_id,
    )
    controller_sha256, implementation_sha256 = _qualification_producer_hashes(verified)
    qualification_supervisor = verify_qualification_supervisor_receipts(
        verified=verified,
        execution_root=execution_root,
        expected_controller_argv=controller_argv,
    )
    storage_binding = payload.get("storage_preflight_before_qualification")
    if not isinstance(storage_binding, Mapping):
        raise FreezeError("qualification handoff omits its pre-qualification storage guard")
    storage_before_qualification = verify_storage_preflight_receipt(
        verified=verified,
        execution_uuid=execution_uuid,
        execution_root=execution_root,
        phase="before_qualification",
        receipt_path=execution_root
        / str(STORAGE_PREFLIGHT_CONTRACT["receipt_names"]["before_qualification"]),
        expected_receipt_sha256=str(storage_binding.get("file_sha256", "")),
        expected_provision_receipt_sha256=expected_provision_receipt_sha256,
        expected_pod_id=expected_pod_id,
    )
    if (
        storage_before_qualification["measured_monotonic_ns"]
        > qualification_supervisor["started_monotonic_ns"]
    ):
        raise FreezeError("qualification handoff storage guard was recorded after qualification")
    copied_paths = {kind: execution_root / name for kind, name in PROFILE_QUALIFICATION_COPY_NAMES.items()}
    gpu_uuids = payload.get("gpu_uuids")
    if not isinstance(gpu_uuids, list):
        raise FreezeError("qualification handoff omits GPU UUIDs")
    try:
        from .qualification_producer import ProducerError, replay_producer_receipt
    except ImportError as error:
        raise FreezeError("qualification producer replay is unavailable") from error
    try:
        replayed = replay_producer_receipt(
            receipt_path=copied_paths["producer_receipt"],
            expected_controller_sha256=controller_sha256,
            expected_implementation_sha256=implementation_sha256,
            expected_controller_argv=controller_argv,
            expected_freeze_binding=expected_bindings["freeze_binding"],
            expected_provision_binding=expected_bindings["provision_binding"],
            expected_model_receipt_bindings=expected_bindings["model_receipt_bindings"],
            expected_integration_audit_bindings=expected_bindings["model_integration_audit_bindings"],
            expected_gpu_uuids=gpu_uuids,
            artifact_path_overrides={
                kind: copied_paths[kind] for kind in ("evidence", "report", "transient_inventory")
            },
        )
    except ProducerError as error:
        raise FreezeError("qualification handoff copied artifacts failed replay") from error
    copied_artifacts = replayed.get("artifacts")
    if not isinstance(copied_artifacts, Mapping):
        raise FreezeError("qualification handoff replay omitted artifact bindings")
    qualification = _verify_profile_qualification_files(
        verified=verified,
        report_path=copied_paths["report"],
        expected_report_sha256=str(copied_artifacts["report"]["file_sha256"]),
        evidence_path=copied_paths["evidence"],
        expected_evidence_sha256=str(copied_artifacts["evidence"]["file_sha256"]),
        expected_provision_receipt_sha256=expected_provision_receipt_sha256,
        expected_pod_id=expected_pod_id,
    )
    expected_body = {
        "schema": PROFILE_QUALIFICATION_HANDOFF_SCHEMA,
        "schema_version": PROFILE_QUALIFICATION_HANDOFF_SCHEMA_VERSION,
        "study_id": "g00f",
        "execution_uuid": execution_uuid,
        "freeze_file_sha256": verified.file_sha256,
        "freeze_digest": verified.digest,
        "producer_contract": copy.deepcopy(dict(QUALIFICATION_PRODUCER_CONTRACT)),
        "controller_argv": controller_argv,
        "qualification_supervisor": qualification_supervisor,
        "storage_preflight_before_qualification": storage_before_qualification,
        "source_producer_receipt_file_sha256": copied_artifacts["producer_receipt"]["file_sha256"],
        "copied_artifacts": copy.deepcopy(dict(copied_artifacts)),
        "qualification": qualification,
        "gpu_uuids": list(gpu_uuids),
        "ledger_absent_at_creation": True,
        "compact_evidence_retained_through_final_gate": True,
        "outcomes_seen": False,
        "g01_launch_authorized": False,
    }
    if body != expected_body or payload.get("handoff_digest") != semantic_digest(expected_body):
        raise FreezeError("qualification handoff changed or violates the frozen contract")
    return {
        "path": str(target),
        "file_sha256": expected_sha256,
        "handoff_digest": payload["handoff_digest"],
        "execution_uuid": execution_uuid,
        "copied_artifacts": copy.deepcopy(dict(copied_artifacts)),
        "qualification": qualification,
        "qualification_supervisor": qualification_supervisor,
        "storage_preflight_before_qualification": storage_before_qualification,
        "gpu_uuids": list(gpu_uuids),
    }


def verify_profile_qualification(
    *,
    verified: VerifiedFreeze,
    handoff_path: str | Path,
    expected_handoff_sha256: str,
    expected_provision_receipt_sha256: str,
    expected_pod_id: str,
) -> dict[str, Any]:
    """Public qualification replay requires the authenticated copied-artifact handoff."""

    return verify_profile_qualification_handoff(
        verified=verified,
        handoff_path=handoff_path,
        expected_handoff_sha256=expected_handoff_sha256,
        expected_provision_receipt_sha256=expected_provision_receipt_sha256,
        expected_pod_id=expected_pod_id,
    )


def create_profile_selection_receipt(
    *,
    verified: VerifiedFreeze,
    execution_uuid: str,
    qualification_handoff_path: str | Path,
    expected_qualification_handoff_sha256: str,
    expected_provision_receipt_sha256: str,
    expected_pod_id: str,
    observed_gpu_uuids: Sequence[str],
    output_path: str | Path,
) -> VerifiedFreeze:
    """Apply the one frozen profile rule and persist its pre-ITT O_EXCL receipt."""

    if verified.selected_profile is not None or verified.selection_receipt is not None:
        raise FreezeError("H200 profile was already selected")
    try:
        parsed_uuid = uuid.UUID(execution_uuid)
    except (ValueError, AttributeError) as error:
        raise FreezeError("execution UUID must be a canonical UUID4") from error
    if parsed_uuid.version != 4 or str(parsed_uuid) != execution_uuid:
        raise FreezeError("execution UUID must be a canonical UUID4")
    target = Path(output_path).resolve()
    execution_root = target.parent
    if target != execution_root / "h200-profile-selection.json":
        raise FreezeError("H200 profile selection has a noncanonical execution-root path")
    if execution_root.name != execution_uuid:
        raise FreezeError("H200 profile selection execution root/UUID differ")
    if (execution_root / "itt-ledger").exists():
        raise FreezeError("H200 profile selection must precede ITT ledger allocation")
    qualification = verify_profile_qualification(
        verified=verified,
        handoff_path=qualification_handoff_path,
        expected_handoff_sha256=expected_qualification_handoff_sha256,
        expected_provision_receipt_sha256=expected_provision_receipt_sha256,
        expected_pod_id=expected_pod_id,
    )
    if qualification["execution_uuid"] != execution_uuid:
        raise FreezeError("H200 profile selection/qualification execution UUIDs differ")
    canonical_gpu_uuids = sorted(str(value) for value in observed_gpu_uuids)
    if canonical_gpu_uuids != qualification["gpu_uuids"]:
        raise FreezeError("selector GPU UUIDs differ from qualification evidence")
    eligibility = qualification["qualification"]["eligibility"]
    branch = qualification["qualification"]["qualification_branch"]
    if branch == "tuned_capacity_fallback_baseline" and eligibility.get("tuned") is True:
        raise FreezeError("recoverable tuned-capacity fallback cannot select tuned")
    if eligibility.get("tuned") is True:
        selected_profile = "tuned"
        reason = "every_tuned_requirement_passed"
    elif eligibility.get("baseline") is True:
        selected_profile = "baseline"
        reason = (
            "recoverable_tuned_capacity_failure_and_every_baseline_requirement_passed"
            if branch == "tuned_capacity_fallback_baseline"
            else "tuned_failed_and_every_baseline_replay_device_projection_requirement_passed"
        )
    else:
        raise FreezeError("neither frozen H200 profile qualifies for confirmatory execution")
    selected_family = _selected_family_binding(verified, selected_profile)
    selected_monotonic_ns = time.monotonic_ns()
    if selected_monotonic_ns < qualification["qualification_supervisor"]["completed_monotonic_ns"]:
        raise FreezeError("profile selection predates qualification completion")
    body = {
        "schema": PROFILE_SELECTION_SCHEMA,
        "schema_version": PROFILE_SELECTION_SCHEMA_VERSION,
        "study_id": "g00f",
        "execution_uuid": execution_uuid,
        "freeze_file_sha256": verified.file_sha256,
        "freeze_digest": verified.digest,
        "selector_contract": copy.deepcopy(dict(PROFILE_SELECTOR_CONTRACT)),
        "qualification_handoff": qualification,
        "selected_profile": selected_profile,
        "selection_reason": reason,
        "selected_monotonic_ns": selected_monotonic_ns,
        "selected_family": selected_family,
        "candidate_family_digests": {
            profile: _selected_family_binding(verified, profile)["family_digest"]
            for profile in EXECUTION_PROFILES
        },
        "gpu_uuids": canonical_gpu_uuids,
        "ledger_absent_at_creation": True,
        "one_profile_for_all_160_runs": True,
        "fallback_after_ledger": False,
        "outcomes_seen": False,
        "g01_launch_authorized": False,
    }
    payload = {**body, "selection_digest": semantic_digest(body)}
    exclusive_json(target, payload)
    os.chmod(target, 0o400)
    return verify_profile_selection_receipt(
        verified=verified,
        receipt_path=target,
        expected_receipt_sha256=sha256_file(target),
        expected_provision_receipt_sha256=expected_provision_receipt_sha256,
        expected_pod_id=expected_pod_id,
    )


def verify_profile_selection_receipt(
    *,
    verified: VerifiedFreeze,
    receipt_path: str | Path,
    expected_receipt_sha256: str,
    expected_provision_receipt_sha256: str,
    expected_pod_id: str,
) -> VerifiedFreeze:
    """Authenticate the deterministic profile decision for all later commands."""

    if verified.selected_profile is not None or verified.selection_receipt is not None:
        raise FreezeError("profile selection verifier requires an unselected freeze")
    target = Path(receipt_path).resolve()
    execution_root = target.parent
    if target != execution_root / "h200-profile-selection.json":
        raise FreezeError("H200 profile selection has a noncanonical execution-root path")
    expected_sha = require_sha256(expected_receipt_sha256, "H200 profile selection SHA-256")
    if (
        not target.is_file()
        or target.is_symlink()
        or stat.S_IMODE(target.stat().st_mode) != 0o400
        or sha256_file(target) != expected_sha
    ):
        raise FreezeError("H200 profile selection bytes changed")
    payload = strict_json(target, "H200 profile selection receipt")
    body = {key: value for key, value in payload.items() if key != "selection_digest"}
    selected_profile = str(payload.get("selected_profile", ""))
    selected_monotonic_ns = payload.get("selected_monotonic_ns")
    execution_uuid = str(payload.get("execution_uuid", ""))
    try:
        parsed_uuid = uuid.UUID(execution_uuid)
    except (ValueError, AttributeError) as error:
        raise FreezeError("H200 profile selection execution UUID is invalid") from error
    qualification_binding = payload.get("qualification_handoff")
    if not isinstance(qualification_binding, Mapping):
        raise FreezeError("H200 profile selection omits qualification evidence")
    qualification = verify_profile_qualification(
        verified=verified,
        handoff_path=execution_root / "h200-profile-qualification-handoff.json",
        expected_handoff_sha256=str(qualification_binding.get("file_sha256", "")),
        expected_provision_receipt_sha256=expected_provision_receipt_sha256,
        expected_pod_id=expected_pod_id,
    )
    eligibility = qualification["qualification"]["eligibility"]
    branch = qualification["qualification"]["qualification_branch"]
    deterministic_selection = (
        "tuned"
        if eligibility.get("tuned") is True
        else "baseline"
        if eligibility.get("baseline") is True
        else None
    )
    expected_reason = (
        "every_tuned_requirement_passed"
        if deterministic_selection == "tuned"
        else (
            "recoverable_tuned_capacity_failure_and_every_baseline_requirement_passed"
            if branch == "tuned_capacity_fallback_baseline"
            else "tuned_failed_and_every_baseline_replay_device_projection_requirement_passed"
        )
    )
    if (
        payload.get("schema") != PROFILE_SELECTION_SCHEMA
        or payload.get("schema_version") != PROFILE_SELECTION_SCHEMA_VERSION
        or payload.get("study_id") != "g00f"
        or parsed_uuid.version != 4
        or str(parsed_uuid) != execution_uuid
        or execution_root.name != execution_uuid
        or payload.get("freeze_file_sha256") != verified.file_sha256
        or payload.get("freeze_digest") != verified.digest
        or payload.get("selector_contract") != PROFILE_SELECTOR_CONTRACT
        or qualification_binding != qualification
        or qualification.get("execution_uuid") != execution_uuid
        or (branch == "tuned_capacity_fallback_baseline" and eligibility.get("tuned") is not False)
        or selected_profile != deterministic_selection
        or selected_profile not in EXECUTION_PROFILES
        or payload.get("selection_reason") != expected_reason
        or isinstance(selected_monotonic_ns, bool)
        or not isinstance(selected_monotonic_ns, int)
        or selected_monotonic_ns < qualification["qualification_supervisor"]["completed_monotonic_ns"]
        or payload.get("selected_family") != _selected_family_binding(verified, selected_profile)
        or payload.get("candidate_family_digests")
        != {
            profile: _selected_family_binding(verified, profile)["family_digest"]
            for profile in EXECUTION_PROFILES
        }
        or payload.get("gpu_uuids") != qualification["gpu_uuids"]
        or payload.get("ledger_absent_at_creation") is not True
        or payload.get("one_profile_for_all_160_runs") is not True
        or payload.get("fallback_after_ledger") is not False
        or payload.get("outcomes_seen") is not False
        or payload.get("g01_launch_authorized") is not False
        or payload.get("selection_digest") != semantic_digest(body)
    ):
        raise FreezeError("H200 profile selection receipt changed or violates the frozen rule")
    binding = {
        "path": str(target),
        "file_sha256": expected_sha,
        "selection_digest": payload["selection_digest"],
        "execution_uuid": execution_uuid,
        "selected_profile": selected_profile,
        "selected_monotonic_ns": selected_monotonic_ns,
        "qualification_handoff": qualification,
        "gpu_uuids": list(qualification["gpu_uuids"]),
    }
    _activate_profile(selected_profile)
    return replace(
        verified,
        plans=verified.candidate_plans[selected_profile],
        selected_profile=selected_profile,
        selection_receipt=binding,
    )


def _qualification_transient_cleanup_state(
    *,
    execution_root: Path,
    handoff: Mapping[str, Any],
) -> dict[str, Any]:
    copied = handoff.get("copied_artifacts")
    if not isinstance(copied, Mapping):
        raise FreezeError("qualification handoff omits copied compact artifacts")
    inventory_binding = copied.get("transient_inventory")
    if not isinstance(inventory_binding, Mapping):
        raise FreezeError("qualification handoff omits its transient inventory")
    inventory_path = execution_root / PROFILE_QUALIFICATION_COPY_NAMES["transient_inventory"]
    if (
        Path(str(inventory_binding.get("path", ""))).resolve() != inventory_path
        or not inventory_path.is_file()
        or inventory_path.is_symlink()
        or stat.S_IMODE(inventory_path.stat().st_mode) != 0o400
        or sha256_file(inventory_path) != inventory_binding.get("file_sha256")
    ):
        raise FreezeError("copied qualification transient inventory changed")
    inventory = strict_json(inventory_path, "H200 qualification transient inventory")
    transient_root = Path(str(inventory.get("transient_root", ""))).resolve()
    expected_root = (
        _qualification_engineering_root(execution_root, str(handoff["execution_uuid"])) / "transient-raw"
    )
    entries = inventory.get("entries")
    if (
        transient_root != expected_root
        or not isinstance(entries, list)
        or inventory.get("final_inventory") != []
        or inventory.get("final_bytes") != 0
        or inventory.get("all_listed_paths_absent") is not True
        or inventory.get("compact_manifest_only") is not True
        or inventory.get("raw_payloads_embedded") is not False
    ):
        raise FreezeError("qualification transient inventory violates the frozen cleanup contract")
    total_bytes = 0
    observed_paths: list[str] = []
    for raw_entry in entries:
        if not isinstance(raw_entry, Mapping):
            raise FreezeError("qualification transient inventory contains a malformed entry")
        path = Path(str(raw_entry.get("path", ""))).resolve()
        size = raw_entry.get("bytes")
        if (
            transient_root not in path.parents
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or raw_entry.get("deleted") is not True
            or path.exists()
        ):
            raise FreezeError("qualification transient artifact was retained or misreported")
        total_bytes += size
        observed_paths.append(str(path))
    if len(set(observed_paths)) != len(observed_paths):
        raise FreezeError("qualification transient inventory reuses a path")
    if transient_root.exists() and (
        transient_root.is_symlink() or not transient_root.is_dir() or any(transient_root.iterdir())
    ):
        raise FreezeError("qualification transient raw root is not exactly empty")
    compact_bindings: dict[str, Any] = {}
    for kind, name in PROFILE_QUALIFICATION_COPY_NAMES.items():
        path = execution_root / name
        binding = copied.get(kind)
        if (
            not isinstance(binding, Mapping)
            or Path(str(binding.get("path", ""))).resolve() != path
            or not path.is_file()
            or path.is_symlink()
            or stat.S_IMODE(path.stat().st_mode) != 0o400
            or sha256_file(path) != binding.get("file_sha256")
        ):
            raise FreezeError(f"qualification compact {kind} artifact changed before ITT")
        compact_bindings[kind] = copy.deepcopy(dict(binding))
    return {
        "transient_root": str(transient_root),
        "listed_entry_count": len(entries),
        "listed_total_bytes": total_bytes,
        "peak_bytes": inventory.get("peak_bytes"),
        "inventory": copy.deepcopy(dict(inventory_binding)),
        "compact_artifacts": compact_bindings,
        "all_listed_paths_absent": True,
        "transient_root_exactly_empty_or_absent": True,
    }


def create_profile_qualification_cleanup_receipt(
    *,
    verified: VerifiedFreeze,
    output_path: str | Path,
    expected_provision_receipt_sha256: str,
    expected_pod_id: str,
) -> dict[str, Any]:
    """Attest transient-only cleanup after selection and before ITT allocation."""

    selected_profile = _require_selected(verified)
    assert verified.selection_receipt is not None
    target = Path(output_path).resolve()
    execution_root = target.parent
    if target != execution_root / "h200-profile-qualification-cleanup.json":
        raise FreezeError("qualification cleanup has a noncanonical execution-root path")
    if (
        Path(str(verified.selection_receipt.get("path", ""))).parent != execution_root
        or verified.selection_receipt.get("execution_uuid") != execution_root.name
    ):
        raise FreezeError("qualification cleanup/profile selection execution roots differ")
    if (execution_root / "itt-ledger").exists():
        raise FreezeError("qualification transient cleanup must precede ITT ledger allocation")
    handoff_binding = verified.selection_receipt.get("qualification_handoff")
    if not isinstance(handoff_binding, Mapping):
        raise FreezeError("profile selection omits its qualification handoff")
    handoff = verify_profile_qualification_handoff(
        verified=verified,
        handoff_path=execution_root / "h200-profile-qualification-handoff.json",
        expected_handoff_sha256=str(handoff_binding.get("file_sha256", "")),
        expected_provision_receipt_sha256=expected_provision_receipt_sha256,
        expected_pod_id=expected_pod_id,
    )
    cleanup_state = _qualification_transient_cleanup_state(
        execution_root=execution_root,
        handoff=handoff,
    )
    storage_before_itt_path = execution_root / str(STORAGE_PREFLIGHT_CONTRACT["receipt_names"]["before_itt"])
    storage_before_itt = verify_storage_preflight_receipt(
        verified=verified,
        execution_uuid=execution_root.name,
        execution_root=execution_root,
        phase="before_itt",
        receipt_path=storage_before_itt_path,
        expected_receipt_sha256=sha256_file(storage_before_itt_path),
        expected_provision_receipt_sha256=expected_provision_receipt_sha256,
        expected_pod_id=expected_pod_id,
    )
    created_monotonic_ns = time.monotonic_ns()
    if created_monotonic_ns < storage_before_itt["measured_monotonic_ns"]:
        raise FreezeError("qualification cleanup predates the pre-ITT storage recheck")
    body = {
        "schema": PROFILE_QUALIFICATION_CLEANUP_SCHEMA,
        "schema_version": PROFILE_QUALIFICATION_CLEANUP_SCHEMA_VERSION,
        "study_id": "g00f",
        "execution_uuid": verified.selection_receipt["execution_uuid"],
        "freeze_file_sha256": verified.file_sha256,
        "freeze_digest": verified.digest,
        "selected_profile": selected_profile,
        "profile_selection": copy.deepcopy(dict(verified.selection_receipt)),
        "qualification_handoff": handoff,
        "cleanup_state": cleanup_state,
        "created_monotonic_ns": created_monotonic_ns,
        "storage_preflight_before_itt": storage_before_itt,
        "attestation_scope": "transient_raw_tensors_and_checkpoints_only",
        "raw_deletion_completed_by_producer_before_handoff": True,
        "deletion_performed_by_this_receipt": False,
        "compact_evidence_retained_through_final_gate": True,
        "ledger_absent_at_creation": True,
        "outcomes_seen": False,
        "g01_launch_authorized": False,
    }
    payload = {**body, "cleanup_digest": semantic_digest(body)}
    exclusive_json(target, payload)
    os.chmod(target, 0o400)
    return verify_profile_qualification_cleanup_receipt(
        verified=verified,
        receipt_path=target,
        expected_receipt_sha256=sha256_file(target),
        expected_provision_receipt_sha256=expected_provision_receipt_sha256,
        expected_pod_id=expected_pod_id,
    )


def verify_profile_qualification_cleanup_receipt(
    *,
    verified: VerifiedFreeze,
    receipt_path: str | Path,
    expected_receipt_sha256: str,
    expected_provision_receipt_sha256: str,
    expected_pod_id: str,
) -> dict[str, Any]:
    """Replay the transient-only cleanup and compact-evidence retention proof."""

    selected_profile = _require_selected(verified)
    assert verified.selection_receipt is not None
    target = Path(receipt_path).resolve()
    execution_root = target.parent
    if target != execution_root / "h200-profile-qualification-cleanup.json":
        raise FreezeError("qualification cleanup has a noncanonical execution-root path")
    if (
        Path(str(verified.selection_receipt.get("path", ""))).parent != execution_root
        or verified.selection_receipt.get("execution_uuid") != execution_root.name
    ):
        raise FreezeError("qualification cleanup/profile selection execution roots differ")
    expected_sha256 = require_sha256(expected_receipt_sha256, "qualification cleanup SHA-256")
    if (
        not target.is_file()
        or target.is_symlink()
        or stat.S_IMODE(target.stat().st_mode) != 0o400
        or sha256_file(target) != expected_sha256
    ):
        raise FreezeError("qualification cleanup bytes or mode changed")
    payload = strict_json(target, "H200 qualification cleanup receipt")
    body = {key: value for key, value in payload.items() if key != "cleanup_digest"}
    handoff_binding = verified.selection_receipt.get("qualification_handoff")
    if not isinstance(handoff_binding, Mapping):
        raise FreezeError("profile selection omits its qualification handoff")
    handoff = verify_profile_qualification_handoff(
        verified=verified,
        handoff_path=execution_root / "h200-profile-qualification-handoff.json",
        expected_handoff_sha256=str(handoff_binding.get("file_sha256", "")),
        expected_provision_receipt_sha256=expected_provision_receipt_sha256,
        expected_pod_id=expected_pod_id,
    )
    cleanup_state = _qualification_transient_cleanup_state(
        execution_root=execution_root,
        handoff=handoff,
    )
    created_monotonic_ns = payload.get("created_monotonic_ns")
    storage_binding = payload.get("storage_preflight_before_itt")
    if not isinstance(storage_binding, Mapping):
        raise FreezeError("qualification cleanup omits its pre-ITT storage recheck")
    storage_before_itt = verify_storage_preflight_receipt(
        verified=verified,
        execution_uuid=execution_root.name,
        execution_root=execution_root,
        phase="before_itt",
        receipt_path=execution_root / str(STORAGE_PREFLIGHT_CONTRACT["receipt_names"]["before_itt"]),
        expected_receipt_sha256=str(storage_binding.get("file_sha256", "")),
        expected_provision_receipt_sha256=expected_provision_receipt_sha256,
        expected_pod_id=expected_pod_id,
    )
    expected_body = {
        "schema": PROFILE_QUALIFICATION_CLEANUP_SCHEMA,
        "schema_version": PROFILE_QUALIFICATION_CLEANUP_SCHEMA_VERSION,
        "study_id": "g00f",
        "execution_uuid": verified.selection_receipt["execution_uuid"],
        "freeze_file_sha256": verified.file_sha256,
        "freeze_digest": verified.digest,
        "selected_profile": selected_profile,
        "profile_selection": copy.deepcopy(dict(verified.selection_receipt)),
        "qualification_handoff": handoff,
        "cleanup_state": cleanup_state,
        "created_monotonic_ns": created_monotonic_ns,
        "storage_preflight_before_itt": storage_before_itt,
        "attestation_scope": "transient_raw_tensors_and_checkpoints_only",
        "raw_deletion_completed_by_producer_before_handoff": True,
        "deletion_performed_by_this_receipt": False,
        "compact_evidence_retained_through_final_gate": True,
        "ledger_absent_at_creation": True,
        "outcomes_seen": False,
        "g01_launch_authorized": False,
    }
    if (
        isinstance(created_monotonic_ns, bool)
        or not isinstance(created_monotonic_ns, int)
        or created_monotonic_ns < storage_before_itt["measured_monotonic_ns"]
        or body != expected_body
        or payload.get("cleanup_digest") != semantic_digest(expected_body)
    ):
        raise FreezeError("qualification cleanup receipt changed or violates the frozen contract")
    return {
        "path": str(target),
        "file_sha256": expected_sha256,
        "cleanup_digest": payload["cleanup_digest"],
        "execution_uuid": payload["execution_uuid"],
        "selected_profile": selected_profile,
        "qualification_handoff": handoff,
        "cleanup_state": cleanup_state,
        "created_monotonic_ns": created_monotonic_ns,
        "storage_preflight_before_itt": storage_before_itt,
    }


def verify_launch_receipt(
    *,
    verified: VerifiedFreeze,
    worker_index: int,
    receipt_path: str | Path,
    expected_provision_receipt_sha256: str | None = None,
    expected_pod_id: str | None = None,
) -> dict[str, Any]:
    launch_path = Path(receipt_path).resolve()
    execution_root = launch_path.parent
    if launch_path != execution_root / f"worker-{worker_index}-launch.json":
        raise FreezeError("worker launch receipt has a noncanonical execution-root path")
    receipt = strict_json(launch_path, "G00-F worker launch receipt")
    body = {key: value for key, value in receipt.items() if key != "receipt_digest"}
    exact_keys = {
        "concurrent_runs",
        "cuda_visible_devices",
        "data_center",
        "execution_uuid",
        "freeze_digest",
        "freeze_file_sha256",
        "g01_launch_authorized",
        "gpu_family",
        "gpu_name",
        "gpu_uuid",
        "image",
        "ledger",
        "model_integration_audits",
        "network_volume_id",
        "network_volume_mount",
        "offline_environment",
        "outcomes_seen",
        "packages",
        "profile_selection",
        "profile_qualification_cleanup",
        "python",
        "runpod_provision",
        "runtime_environment",
        "schema",
        "schema_version",
        "source_bundle",
        "started_monotonic_ns",
        "started_unix_ns",
        "selected_profile",
        "torch_gpu_name",
        "visible_gpu_count",
        "worker_index",
    }
    required = {
        "schema": LAUNCH_RECEIPT_SCHEMA,
        "schema_version": LAUNCH_RECEIPT_SCHEMA_VERSION,
        "worker_index": worker_index,
        "freeze_file_sha256": verified.file_sha256,
        "freeze_digest": verified.digest,
        "image": verified.payload["runtime"]["image"],
        "network_volume_id": verified.payload["runtime"]["network_volume_id"],
        "network_volume_mount": verified.payload["runtime"]["network_volume_mount"],
        "data_center": verified.payload["runtime"]["data_center"],
        "visible_gpu_count": 1,
        "gpu_family": "NVIDIA H200",
        "concurrent_runs": 1,
        "selected_profile": verified.selected_profile,
    }
    if set(body) != exact_keys or any(receipt.get(key) != value for key, value in required.items()):
        raise FreezeError("worker launch receipt is not bound to the exact runtime/freeze")
    if receipt.get("receipt_digest") != semantic_digest(body):
        raise FreezeError("worker launch receipt semantic digest mismatch")
    packages = receipt.get("packages")
    frozen_packages = verified.payload["runtime"].get("python_packages")
    if packages != frozen_packages:
        raise FreezeError("worker package versions differ from the frozen runtime")
    runtime_environment = receipt.get("runtime_environment")
    if not isinstance(runtime_environment, Mapping) or set(runtime_environment) != {
        "accelerator",
        "accelerator_digest",
        "installed_distributions",
        "installed_distributions_digest",
        "lock_scope",
    }:
        raise FreezeError("worker launch omits the complete pre-outcome runtime inventory")
    accelerator = runtime_environment.get("accelerator")
    installed = runtime_environment.get("installed_distributions")
    if (
        not isinstance(accelerator, Mapping)
        or not isinstance(installed, Mapping)
        or runtime_environment.get("accelerator_digest") != semantic_digest(accelerator)
        or runtime_environment.get("installed_distributions_digest") != semantic_digest(installed)
        or runtime_environment.get("lock_scope")
        != "recorded_pre_outcome_runtime_evidence_not_complete_dependency_lock"
        or accelerator.get("torch_version") != "2.8.0+cu128"
        or accelerator.get("cuda_runtime") != "12.8"
        or accelerator.get("cuda_available") is not True
    ):
        raise FreezeError("worker launch full runtime inventory digest changed")
    if (
        receipt.get("python") != verified.payload["runtime"]["python"]
        or receipt.get("outcomes_seen") is not False
        or receipt.get("g01_launch_authorized") is not False
        or receipt.get("offline_environment") != {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        or not str(receipt.get("gpu_name", "")).startswith("NVIDIA H200")
        or "H200" not in str(receipt.get("torch_gpu_name", ""))
        or not str(receipt.get("gpu_uuid", "")).startswith("GPU-")
        or receipt.get("cuda_visible_devices") != receipt.get("gpu_uuid")
        or isinstance(receipt.get("started_unix_ns"), bool)
        or not isinstance(receipt.get("started_unix_ns"), int)
        or int(receipt["started_unix_ns"]) < 0
        or isinstance(receipt.get("started_monotonic_ns"), bool)
        or not isinstance(receipt.get("started_monotonic_ns"), int)
        or int(receipt["started_monotonic_ns"]) < 0
    ):
        raise FreezeError("worker launch runtime probe fields differ from the frozen contract")
    bundle = receipt.get("source_bundle")
    if not isinstance(bundle, Mapping):
        raise FreezeError("worker launch receipt omits its source bundle")
    bundle_receipt_path = Path(str(bundle.get("receipt_path", ""))).resolve()
    bundle_receipt = strict_json(bundle_receipt_path, "bound extracted source bundle receipt")
    bundle_body = {key: value for key, value in bundle_receipt.items() if key != "receipt_digest"}
    embedded_manifest = strict_json(
        verified.repo / "G00F-BUNDLE-MANIFEST.json",
        "embedded G00-F bundle manifest",
    )
    if (
        set(bundle)
        != {
            "archive_sha256",
            "authenticated_runtime_files",
            "extracted_root",
            "manifest_digest",
            "manifest_sha256",
            "receipt_digest",
            "receipt_file_sha256",
            "receipt_path",
        }
        or bundle.get("archive_sha256") != verified.payload["source_bundle"]["archive_sha256"]
        or bundle.get("manifest_sha256") != verified.payload["source_bundle"]["manifest_sha256"]
        or bundle.get("manifest_digest") != verified.payload["source_bundle"]["manifest_digest"]
        or bundle.get("extracted_root") != str(verified.repo)
        or bundle_receipt_path != execution_root / "source-bundle-receipt.json"
        or verified.repo != execution_root / "frozen-source"
        or bundle.get("receipt_file_sha256") != sha256_file(bundle_receipt_path)
        or bundle.get("receipt_digest") != bundle_receipt.get("receipt_digest")
        or bundle.get("authenticated_runtime_files") != bundle_receipt.get("authenticated_runtime_files")
        or bundle_receipt.get("receipt_digest") != semantic_digest(bundle_body)
        or bundle_receipt.get("freeze_file_sha256") != verified.file_sha256
        or bundle_receipt.get("freeze_digest") != verified.digest
        or set(bundle_body)
        != {
            "archive_sha256",
            "authenticated_runtime_files",
            "extracted_root",
            "freeze_digest",
            "freeze_file_sha256",
            "g01_launch_authorized",
            "manifest_digest",
            "manifest_sha256",
            "member_count",
            "schema",
            "schema_version",
            "tar_safety",
        }
        or bundle_receipt.get("schema") != "goalzendo.g00f_h200_extracted_source_bundle_receipt"
        or bundle_receipt.get("schema_version") != 1
        or bundle_receipt.get("member_count") != len(embedded_manifest.get("members", []))
        or bundle_receipt.get("tar_safety")
        != {
            "exact_bytes": True,
            "exact_member_set": True,
            "exact_modes": True,
            "no_absolute_or_parent_paths": True,
            "no_links": True,
            "only_regular_files": True,
        }
        or bundle_receipt.get("g01_launch_authorized") is not False
    ):
        raise FreezeError("worker launch receipt is not bound to the frozen source archive")
    ledger_binding = receipt.get("ledger")
    if not isinstance(ledger_binding, Mapping):
        raise FreezeError("worker launch receipt omits its ITT ledger")
    ledger = verify_attempt_ledger(
        verified=verified,
        ledger_root=str(ledger_binding.get("ledger_root", "")),
    )
    provision_binding = receipt.get("runpod_provision")
    if not isinstance(provision_binding, Mapping):
        raise FreezeError("worker launch receipt omits the external Runpod allocation")
    provision = verify_runpod_provision_receipt(
        verified=verified,
        receipt_path=str(provision_binding.get("path", "")),
        expected_receipt_sha256=str(provision_binding.get("file_sha256", "")),
        expected_pod_id=str(provision_binding.get("pod_id", "")),
    )
    if Path(str(provision["path"])) != execution_root / "runpod-provision-receipt.json":
        raise FreezeError("worker launch provision receipt is outside its execution root")
    if (expected_provision_receipt_sha256 is None) != (expected_pod_id is None):
        raise FreezeError("external provision SHA-256 and pod ID must be supplied together")
    if expected_provision_receipt_sha256 is not None and (
        provision["file_sha256"]
        != require_sha256(
            expected_provision_receipt_sha256,
            "externally expected Runpod provision receipt SHA-256",
        )
        or provision["pod_id"] != expected_pod_id
    ):
        raise FreezeError("worker launch differs from the externally pinned Runpod allocation")
    if (
        set(ledger_binding)
        != {
            "budget_start_file_sha256",
            "file_sha256",
            "ledger_digest",
            "ledger_root",
            "path",
        }
        or any(ledger.get(key) != ledger_binding.get(key) for key in ledger_binding)
        or provision_binding != provision
        or ledger.get("runpod_provision") != provision
        or receipt.get("model_integration_audits") != ledger.get("model_integration_audits")
        or receipt.get("profile_selection") != ledger.get("profile_selection")
        or receipt.get("profile_qualification_cleanup") != ledger.get("profile_qualification_cleanup")
        or receipt.get("selected_profile") != ledger.get("selected_profile")
        or receipt.get("profile_selection") != verified.selection_receipt
        or receipt.get("execution_uuid") != ledger["execution_uuid"]
        or Path(str(ledger["ledger_root"])) != execution_root / "itt-ledger" / str(ledger["execution_uuid"])
    ):
        raise FreezeError("worker launch receipt/ITT ledger binding changed")
    return {
        "path": str(launch_path),
        "file_sha256": sha256_file(launch_path),
        "receipt_digest": receipt["receipt_digest"],
        "worker_index": worker_index,
        "execution_uuid": receipt["execution_uuid"],
        "gpu_uuid": receipt["gpu_uuid"],
        "started_monotonic_ns": receipt["started_monotonic_ns"],
        "ledger": ledger,
        "runpod_provision": provision,
        "model_integration_audits": copy.deepcopy(dict(ledger["model_integration_audits"])),
        "profile_selection": copy.deepcopy(dict(ledger["profile_selection"])),
        "profile_qualification_cleanup": copy.deepcopy(dict(ledger["profile_qualification_cleanup"])),
        "selected_profile": ledger["selected_profile"],
        "runtime_environment": copy.deepcopy(dict(runtime_environment)),
    }


def initialize_attempt_ledger(
    *,
    verified: VerifiedFreeze,
    ledger_root: str | Path,
    execution_uuid: str,
    provision_receipt_path: str | Path,
    expected_provision_receipt_sha256: str,
    expected_pod_id: str,
    model_integration_audit_paths: Mapping[str, str | Path],
    profile_qualification_cleanup_path: str | Path,
    expected_profile_qualification_cleanup_sha256: str,
) -> dict[str, Any]:
    """Preallocate all 160 ITT keys after both pretraining boundary audits."""

    selected_profile = _require_selected(verified)
    assert verified.selection_receipt is not None
    profile_selection = copy.deepcopy(dict(verified.selection_receipt))

    try:
        parsed_uuid = uuid.UUID(execution_uuid)
    except (ValueError, AttributeError) as error:
        raise FreezeError("execution UUID must be a canonical UUID4") from error
    if parsed_uuid.version != 4 or str(parsed_uuid) != execution_uuid:
        raise FreezeError("execution UUID must be a canonical UUID4")
    root = Path(ledger_root).resolve()
    if root.name != execution_uuid:
        raise FreezeError("never-overwritten attempt directory must be named by execution UUID")
    provision = verify_runpod_provision_receipt(
        verified=verified,
        receipt_path=provision_receipt_path,
        expected_receipt_sha256=expected_provision_receipt_sha256,
        expected_pod_id=expected_pod_id,
    )
    expected_provision_path = root.parent.parent / "runpod-provision-receipt.json"
    if Path(str(provision["path"])) != expected_provision_path.resolve():
        raise FreezeError("bound Runpod provision receipt is outside the exact execution root")
    if set(model_integration_audit_paths) != set(CONFIG_SPECS):
        raise FreezeError("ITT ledger requires exactly both model integration audits")
    model_integration_audits = {
        panel_id: verify_model_integration_audit(
            verified=verified,
            panel_id=panel_id,
            audit_path=model_integration_audit_paths[panel_id],
        )
        for panel_id in CONFIG_SPECS
    }
    execution_root = root.parent.parent
    if (
        Path(str(profile_selection.get("path", ""))).parent != execution_root
        or profile_selection.get("execution_uuid") != execution_uuid
    ):
        raise FreezeError("H200 profile selection receipt lies outside the exact execution root")
    if any(Path(str(audit["path"])).parent != execution_root for audit in model_integration_audits.values()):
        raise FreezeError("model integration audits lie outside the exact execution root")
    profile_qualification_cleanup = verify_profile_qualification_cleanup_receipt(
        verified=verified,
        receipt_path=profile_qualification_cleanup_path,
        expected_receipt_sha256=expected_profile_qualification_cleanup_sha256,
        expected_provision_receipt_sha256=expected_provision_receipt_sha256,
        expected_pod_id=expected_pod_id,
    )
    if Path(str(profile_qualification_cleanup["path"])).parent != execution_root:
        raise FreezeError("qualification cleanup receipt lies outside the exact execution root")
    if profile_qualification_cleanup.get("execution_uuid") != execution_uuid:
        raise FreezeError("qualification cleanup receipt belongs to a different execution")
    started_unix_ns = time.time_ns()
    provisioning = provision.get("provisioning")
    if not isinstance(provisioning, Mapping):
        raise FreezeError("verified provision receipt omits its absolute termination guard")
    terminate_after = _canonical_utc(
        str(provisioning.get("terminate_after_utc", "")),
        "Runpod absolute termination time",
    )
    terminate_unix_ns = int(terminate_after.timestamp()) * 1_000_000_000
    provision_remaining_ns = terminate_unix_ns - started_unix_ns
    minimum_remaining_ns = (WALL_CEILING_SECONDS + PROVISION_WATCHDOG_GRACE_SECONDS) * 1_000_000_000
    if provision_remaining_ns < minimum_remaining_ns:
        raise FreezeError(
            "Runpod auto-termination leaves less than the frozen 14-hour budget plus watchdog grace"
        )
    root.parent.mkdir(parents=True, exist_ok=True)
    try:
        root.mkdir(mode=0o700)
    except FileExistsError as error:
        raise FreezeError("execution UUID attempt directory already exists and cannot be reused") from error
    (root / "starts").mkdir()
    (root / "terminals").mkdir()
    (root / "leases").mkdir()
    rows = [
        {
            "panel_id": row["panel_id"],
            "plan_key": row["plan_key"],
            "run_id": row["run_id"],
            "seed": row["seed"],
            "worker_index": row["worker_index"],
            "worker_order": row["worker_order"],
            "initial_state": "preallocated_not_started",
        }
        for row in verified.all_rows
    ]
    body = {
        "schema": ITT_LEDGER_SCHEMA,
        "schema_version": ITT_LEDGER_SCHEMA_VERSION,
        "study_id": "g00f",
        "execution_uuid": execution_uuid,
        "freeze_file_sha256": verified.file_sha256,
        "freeze_digest": verified.digest,
        "planned_runs": 160,
        "runpod_provision": provision,
        "model_integration_audits": model_integration_audits,
        "profile_selection": profile_selection,
        "profile_qualification_cleanup": profile_qualification_cleanup,
        "selected_profile": selected_profile,
        "rows": rows,
        "policy": {
            "attempts_per_plan_key": 1,
            "retry_after_start": False,
            "resume_after_start": False,
            "replacement_or_resampling": False,
            "recorded_failed_attempt_fails_study": True,
            "start_without_terminal_is_incomplete_and_fails_study": True,
            "global_fail_fast_trigger": "infrastructure_or_execution_exception_only",
            "metric_or_prediction_dependent_stop_forbidden": True,
            "remaining_state_after_global_failure": "not_started_after_failure",
            "outcome_file_visibility": (
                "metrics_predictions_summary_chmod_000_until_all_160_terminal_complete"
            ),
        },
        "g01_launch_authorized": False,
    }
    payload = {**body, "ledger_digest": semantic_digest(body)}
    target = root / "ledger.json"
    exclusive_json(target, payload)
    budget_body = {
        "schema": "goalzendo.g00f_h200_monotonic_budget_start",
        "schema_version": 1,
        "execution_uuid": execution_uuid,
        "freeze_file_sha256": verified.file_sha256,
        "freeze_digest": verified.digest,
        "profile_selection_file_sha256": profile_selection["file_sha256"],
        "profile_qualification_cleanup_file_sha256": profile_qualification_cleanup["file_sha256"],
        "selected_profile": selected_profile,
        "wall_ceiling_seconds": WALL_CEILING_SECONDS,
        "worker_count": WORKER_COUNT,
        "started_unix_ns": started_unix_ns,
        "started_monotonic_ns": time.monotonic_ns(),
        "provision_terminate_after_utc": provisioning["terminate_after_utc"],
        "provision_remaining_ns_at_start": provision_remaining_ns,
        "provision_required_remaining_ns": minimum_remaining_ns,
        "outcomes_seen": False,
        "g01_launch_authorized": False,
    }
    budget_payload = {**budget_body, "budget_digest": semantic_digest(budget_body)}
    exclusive_json(root / "budget-start.json", budget_payload)
    return {
        "path": str(target),
        "file_sha256": sha256_file(target),
        "ledger_digest": payload["ledger_digest"],
        "ledger_root": str(root),
        "execution_uuid": execution_uuid,
        "budget_start_file_sha256": sha256_file(root / "budget-start.json"),
        "budget_digest": budget_payload["budget_digest"],
        "runpod_provision": provision,
        "model_integration_audits": model_integration_audits,
        "profile_selection": profile_selection,
        "profile_qualification_cleanup": profile_qualification_cleanup,
        "selected_profile": selected_profile,
    }


def verify_attempt_ledger(
    *,
    verified: VerifiedFreeze,
    ledger_root: str | Path,
) -> dict[str, Any]:
    selected_profile = _require_selected(verified)
    assert verified.selection_receipt is not None
    expected_profile_selection = copy.deepcopy(dict(verified.selection_receipt))
    root = Path(ledger_root).resolve()
    payload = strict_json(root / "ledger.json", "G00-F ITT ledger")
    body = {key: value for key, value in payload.items() if key != "ledger_digest"}
    expected_rows = [
        {
            "panel_id": row["panel_id"],
            "plan_key": row["plan_key"],
            "run_id": row["run_id"],
            "seed": row["seed"],
            "worker_index": row["worker_index"],
            "worker_order": row["worker_order"],
            "initial_state": "preallocated_not_started",
        }
        for row in verified.all_rows
    ]
    expected_policy = {
        "attempts_per_plan_key": 1,
        "retry_after_start": False,
        "resume_after_start": False,
        "replacement_or_resampling": False,
        "recorded_failed_attempt_fails_study": True,
        "start_without_terminal_is_incomplete_and_fails_study": True,
        "global_fail_fast_trigger": "infrastructure_or_execution_exception_only",
        "metric_or_prediction_dependent_stop_forbidden": True,
        "remaining_state_after_global_failure": "not_started_after_failure",
        "outcome_file_visibility": ("metrics_predictions_summary_chmod_000_until_all_160_terminal_complete"),
    }
    execution_uuid = str(payload.get("execution_uuid", ""))
    try:
        parsed_uuid = uuid.UUID(execution_uuid)
    except (ValueError, AttributeError) as error:
        raise FreezeError("G00-F ITT ledger execution UUID is invalid") from error
    budget = strict_json(root / "budget-start.json", "G00-F monotonic budget start")
    budget_body = {key: value for key, value in budget.items() if key != "budget_digest"}
    allowed_root_names = {
        "budget-start.json",
        "budget-timeout.json",
        "global-stop.json",
        "leases",
        "ledger.json",
        "panel-unseal.json",
        "starts",
        "terminals",
    }
    if (
        any(path.is_symlink() for path in root.iterdir())
        or {path.name for path in root.iterdir()} - allowed_root_names
    ):
        raise FreezeError("G00-F ITT ledger root contains an unexpected entry")
    expected_receipt_names = {f"{row['plan_key']}.json" for row in verified.all_rows}
    for directory_name in ("starts", "terminals", "leases"):
        directory = root / directory_name
        if directory.is_dir():
            entries = set(directory.iterdir())
            if (
                any(path.is_symlink() or not path.is_file() for path in entries)
                or {path.name for path in entries} - expected_receipt_names
            ):
                raise FreezeError(f"G00-F {directory_name} inventory contains an extra receipt")
    provision_binding = payload.get("runpod_provision")
    if not isinstance(provision_binding, Mapping):
        raise FreezeError("G00-F ITT ledger omits the externally pinned pod allocation")
    provision_path = root.parent.parent / "runpod-provision-receipt.json"
    provision = verify_runpod_provision_receipt(
        verified=verified,
        receipt_path=provision_path,
        expected_receipt_sha256=str(provision_binding.get("file_sha256", "")),
        expected_pod_id=str(provision_binding.get("pod_id", "")),
    )
    audit_bindings = payload.get("model_integration_audits")
    if not isinstance(audit_bindings, Mapping) or set(audit_bindings) != set(CONFIG_SPECS):
        raise FreezeError("G00-F ITT ledger omits the model integration audits")
    model_integration_audits = {
        panel_id: verify_model_integration_audit(
            verified=verified,
            panel_id=panel_id,
            audit_path=root.parent.parent / f"model-integration-audit-{panel_id.removeprefix('g00f-')}.json",
            replay_model_snapshot=False,
        )
        for panel_id in CONFIG_SPECS
    }
    cleanup_binding = payload.get("profile_qualification_cleanup")
    if not isinstance(cleanup_binding, Mapping):
        raise FreezeError("G00-F ITT ledger omits qualification transient cleanup")
    profile_qualification_cleanup = verify_profile_qualification_cleanup_receipt(
        verified=verified,
        receipt_path=root.parent.parent / "h200-profile-qualification-cleanup.json",
        expected_receipt_sha256=str(cleanup_binding.get("file_sha256", "")),
        expected_provision_receipt_sha256=str(provision["file_sha256"]),
        expected_pod_id=str(provision["pod_id"]),
    )
    budget_started_unix_ns = budget.get("started_unix_ns")
    budget_started_monotonic_ns = budget.get("started_monotonic_ns")
    budget_provision_remaining_ns = budget.get("provision_remaining_ns_at_start")
    provision_terminate_unix_ns = (
        int(
            _canonical_utc(
                str(provision["provisioning"]["terminate_after_utc"]),
                "Runpod absolute termination time",
            ).timestamp()
        )
        * 1_000_000_000
    )
    if (
        payload.get("schema") != ITT_LEDGER_SCHEMA
        or payload.get("schema_version") != ITT_LEDGER_SCHEMA_VERSION
        or payload.get("freeze_file_sha256") != verified.file_sha256
        or payload.get("freeze_digest") != verified.digest
        or payload.get("planned_runs") != 160
        or payload.get("selected_profile") != selected_profile
        or payload.get("profile_selection") != expected_profile_selection
        or expected_profile_selection.get("execution_uuid") != execution_uuid
        or cleanup_binding != profile_qualification_cleanup
        or profile_qualification_cleanup.get("execution_uuid") != execution_uuid
        or provision_binding != provision
        or audit_bindings != model_integration_audits
        or parsed_uuid.version != 4
        or str(parsed_uuid) != execution_uuid
        or root.name != execution_uuid
        or payload.get("rows") != expected_rows
        or payload.get("policy") != expected_policy
        or payload.get("g01_launch_authorized") is not False
        or payload.get("ledger_digest") != semantic_digest(body)
        or not (root / "starts").is_dir()
        or not (root / "terminals").is_dir()
        or not (root / "leases").is_dir()
        or budget.get("schema") != "goalzendo.g00f_h200_monotonic_budget_start"
        or budget.get("schema_version") != 1
        or budget.get("execution_uuid") != execution_uuid
        or budget.get("freeze_file_sha256") != verified.file_sha256
        or budget.get("freeze_digest") != verified.digest
        or budget.get("selected_profile") != selected_profile
        or budget.get("profile_selection_file_sha256") != expected_profile_selection["file_sha256"]
        or budget.get("profile_qualification_cleanup_file_sha256")
        != profile_qualification_cleanup["file_sha256"]
        or budget.get("wall_ceiling_seconds") != WALL_CEILING_SECONDS
        or budget.get("worker_count") != WORKER_COUNT
        or budget.get("provision_terminate_after_utc") != provision["provisioning"]["terminate_after_utc"]
        or isinstance(budget_started_unix_ns, bool)
        or not isinstance(budget_started_unix_ns, int)
        or budget_started_unix_ns <= 0
        or isinstance(budget_started_monotonic_ns, bool)
        or not isinstance(budget_started_monotonic_ns, int)
        or budget_started_monotonic_ns <= 0
        or isinstance(budget_provision_remaining_ns, bool)
        or not isinstance(budget_provision_remaining_ns, int)
        or budget_provision_remaining_ns != provision_terminate_unix_ns - budget_started_unix_ns
        or budget_provision_remaining_ns
        < (WALL_CEILING_SECONDS + PROVISION_WATCHDOG_GRACE_SECONDS) * 1_000_000_000
        or budget.get("provision_required_remaining_ns")
        != (WALL_CEILING_SECONDS + PROVISION_WATCHDOG_GRACE_SECONDS) * 1_000_000_000
        or budget.get("outcomes_seen") is not False
        or budget.get("g01_launch_authorized") is not False
        or budget.get("budget_digest") != semantic_digest(budget_body)
    ):
        raise FreezeError("G00-F ITT ledger does not match the exact prospective plan/policy")
    return {
        "ledger_root": str(root),
        "path": str((root / "ledger.json").resolve()),
        "file_sha256": sha256_file(root / "ledger.json"),
        "ledger_digest": payload["ledger_digest"],
        "execution_uuid": execution_uuid,
        "budget_start": budget,
        "budget_start_file_sha256": sha256_file(root / "budget-start.json"),
        "runpod_provision": provision,
        "model_integration_audits": model_integration_audits,
        "profile_selection": expected_profile_selection,
        "profile_qualification_cleanup": profile_qualification_cleanup,
        "selected_profile": selected_profile,
    }


def enforce_monotonic_budget(ledger: Mapping[str, Any]) -> None:
    budget = ledger.get("budget_start")
    if not isinstance(budget, Mapping):
        raise FreezeError("verified ITT ledger lacks monotonic budget start")
    started = budget.get("started_monotonic_ns")
    if isinstance(started, bool) or not isinstance(started, int) or started < 0:
        raise FreezeError("monotonic budget start is malformed")
    elapsed = time.monotonic_ns() - started
    if elapsed < 0 or elapsed > WALL_CEILING_SECONDS * 1_000_000_000:
        raise FreezeError("G00-F monotonic 14-hour wall budget expired")


def _attempt_receipt(
    *,
    verified: VerifiedFreeze,
    row: Mapping[str, Any],
    kind: str,
    state: str,
    launch_receipt: Mapping[str, Any] | None = None,
    error_type: str | None = None,
    caused_by_plan_key: str | None = None,
    execution_uuid: str,
) -> dict[str, Any]:
    body = {
        "schema": ATTEMPT_RECEIPT_SCHEMA,
        "schema_version": ATTEMPT_RECEIPT_SCHEMA_VERSION,
        "kind": kind,
        "state": state,
        "freeze_file_sha256": verified.file_sha256,
        "freeze_digest": verified.digest,
        "execution_uuid": execution_uuid,
        "panel_id": row["panel_id"],
        "plan_key": row["plan_key"],
        "run_id": row["run_id"],
        "seed": row["seed"],
        "worker_index": row["worker_index"],
        "worker_order": row["worker_order"],
        "launch_receipt": dict(launch_receipt) if launch_receipt is not None else None,
        "error_type": error_type,
        "caused_by_plan_key": caused_by_plan_key,
        "outcome_metrics_read": False,
        "predictions_read": False,
        "g01_launch_authorized": False,
    }
    return {**body, "receipt_digest": semantic_digest(body)}


def _ledger_receipt_path(root: Path, kind: str, plan_key: str) -> Path:
    directory = "starts" if kind == "start" else "terminals"
    return root / directory / f"{plan_key}.json"


def _record_global_failure(
    *,
    verified: VerifiedFreeze,
    ledger_root: Path,
    failed_row: Mapping[str, Any],
    error_type: str,
    execution_uuid: str,
    trigger: str = "infrastructure_or_execution_exception",
) -> None:
    if trigger not in OPERATIONAL_FAILURE_TRIGGERS:
        raise FreezeError("G00-F global stop trigger is not a frozen operational failure")
    body = {
        "schema": "goalzendo.g00f_h200_global_execution_stop",
        "schema_version": 1,
        "freeze_file_sha256": verified.file_sha256,
        "freeze_digest": verified.digest,
        "execution_uuid": execution_uuid,
        "trigger": trigger,
        "failed_plan_key": failed_row["plan_key"],
        "error_type": error_type,
        "outcome_metrics_read": False,
        "predictions_read": False,
        "g01_launch_authorized": False,
    }
    payload = {**body, "stop_digest": semantic_digest(body)}
    stop = ledger_root / "global-stop.json"
    try:
        exclusive_json(stop, payload)
    except FreezeError:
        existing = strict_json(stop, "existing G00-F global stop")
        existing_body = {key: value for key, value in existing.items() if key != "stop_digest"}
        if (
            existing.get("schema") != "goalzendo.g00f_h200_global_execution_stop"
            or existing.get("schema_version") != 1
            or existing.get("freeze_file_sha256") != verified.file_sha256
            or existing.get("freeze_digest") != verified.digest
            or existing.get("execution_uuid") != execution_uuid
            or existing.get("trigger") not in OPERATIONAL_FAILURE_TRIGGERS
            or existing.get("outcome_metrics_read") is not False
            or existing.get("predictions_read") is not False
            or existing.get("g01_launch_authorized") is not False
            or existing.get("stop_digest") != semantic_digest(existing_body)
        ):
            raise FreezeError("existing G00-F global stop is not authentic") from None


def _terminalize_inflight_after_failure(
    *,
    verified: VerifiedFreeze,
    ledger_root: Path,
    error_type: str,
    caused_by_plan_key: str,
    execution_uuid: str,
) -> None:
    """Close every lease/start that lost its worker, without replacement."""

    for row in verified.all_rows:
        plan_key = str(row["plan_key"])
        start_path = _ledger_receipt_path(ledger_root, "start", plan_key)
        terminal_path = _ledger_receipt_path(ledger_root, "terminal", plan_key)
        lease_path = ledger_root / "leases" / f"{plan_key}.json"
        if terminal_path.exists() or (not start_path.exists() and not lease_path.exists()):
            continue
        start = strict_json(start_path, "in-flight G00-F attempt start") if start_path.is_file() else None
        launch = start.get("launch_receipt") if isinstance(start, Mapping) else None
        state = "failed" if start is not None else "failed_before_start_receipt"
        receipt = _attempt_receipt(
            verified=verified,
            row=row,
            kind="terminal",
            state=state,
            launch_receipt=launch if isinstance(launch, Mapping) else None,
            error_type=error_type,
            caused_by_plan_key=caused_by_plan_key,
            execution_uuid=execution_uuid,
        )
        with suppress(FreezeError):
            exclusive_json(terminal_path, receipt)


def reconcile_execution_failure(
    *,
    verified: VerifiedFreeze,
    ledger_root: str | Path,
    error_type: str,
    trigger: str,
    cancel_receipt: str | Path,
) -> dict[str, Any]:
    """Persist an outcome-blind coordinator stop after workers are signalled."""

    if trigger not in OPERATIONAL_FAILURE_TRIGGERS - {"monotonic_14h_deadline"}:
        raise FreezeError("reconcile-failure requires a registered non-metric operational trigger")
    if not error_type or len(error_type) > 128:
        raise FreezeError("reconcile-failure requires a short nonempty error type")
    ledger = verify_attempt_ledger(verified=verified, ledger_root=ledger_root)
    root = Path(str(ledger["ledger_root"]))
    execution_uuid = str(ledger["execution_uuid"])
    pending = [
        row
        for row in verified.all_rows
        if (root / "starts" / f"{row['plan_key']}.json").exists()
        and not (root / "terminals" / f"{row['plan_key']}.json").exists()
    ]
    failed_row: Mapping[str, Any] = (
        pending[0]
        if pending
        else {
            "panel_id": "runtime",
            "plan_key": "coordinator",
            "run_id": "none",
            "seed": 0,
            "worker_index": -1,
            "worker_order": -1,
        }
    )
    caused_by = str(failed_row["plan_key"])
    _record_global_failure(
        verified=verified,
        ledger_root=root,
        failed_row=failed_row,
        error_type=error_type,
        execution_uuid=execution_uuid,
        trigger=trigger,
    )
    _terminalize_inflight_after_failure(
        verified=verified,
        ledger_root=root,
        error_type=error_type,
        caused_by_plan_key=caused_by,
        execution_uuid=execution_uuid,
    )
    _mark_unstarted_after_failure(
        verified=verified,
        ledger_root=root,
        caused_by_plan_key=caused_by,
        execution_uuid=execution_uuid,
    )
    terminals = _terminal_receipts(verified, root)
    counts: dict[str, int] = {}
    for receipt in terminals.values():
        state = str(receipt.get("state", "unknown"))
        counts[state] = counts.get(state, 0) + 1
    body = {
        "schema": "goalzendo.g00f_h200_coordinator_cancel",
        "schema_version": 1,
        "execution_uuid": execution_uuid,
        "freeze_file_sha256": verified.file_sha256,
        "freeze_digest": verified.digest,
        "trigger": trigger,
        "error_type": error_type,
        "terminal_state_counts": dict(sorted(counts.items())),
        "outcome_metrics_read": False,
        "predictions_read": False,
        "g01_launch_authorized": False,
    }
    payload = {**body, "cancel_digest": semantic_digest(body)}
    target = Path(cancel_receipt).resolve()
    try:
        exclusive_json(target, payload)
    except FreezeError:
        existing = strict_json(target, "existing G00-F coordinator cancel receipt")
        if existing != payload:
            raise FreezeError("existing coordinator cancel receipt differs from this stop") from None
    return {
        "path": str(target),
        "file_sha256": sha256_file(target),
        "cancel_digest": payload["cancel_digest"],
    }


def record_budget_timeout(
    *,
    verified: VerifiedFreeze,
    ledger_root: str | Path,
) -> dict[str, Any]:
    """Record the monotonic deadline and terminalize every unstarted key."""

    ledger = verify_attempt_ledger(verified=verified, ledger_root=ledger_root)
    root = Path(str(ledger["ledger_root"]))
    budget = ledger["budget_start"]
    elapsed_ns = time.monotonic_ns() - int(budget["started_monotonic_ns"])
    if elapsed_ns < WALL_CEILING_SECONDS * 1_000_000_000:
        raise FreezeError("cannot record a budget timeout before the immutable deadline")
    body = {
        "schema": "goalzendo.g00f_h200_monotonic_budget_timeout",
        "schema_version": 1,
        "execution_uuid": ledger["execution_uuid"],
        "freeze_file_sha256": verified.file_sha256,
        "freeze_digest": verified.digest,
        "elapsed_monotonic_ns": elapsed_ns,
        "wall_ceiling_seconds": WALL_CEILING_SECONDS,
        "cancel_scope": "all_four_workers",
        "outcome_metrics_read": False,
        "predictions_read": False,
        "g01_launch_authorized": False,
    }
    payload = {**body, "timeout_digest": semantic_digest(body)}
    target = root / "budget-timeout.json"
    try:
        exclusive_json(target, payload)
    except FreezeError:
        existing = strict_json(target, "existing G00-F monotonic budget timeout")
        if (
            existing.get("schema") != payload["schema"]
            or existing.get("execution_uuid") != payload["execution_uuid"]
            or existing.get("freeze_digest") != payload["freeze_digest"]
            or existing.get("outcome_metrics_read") is not False
            or existing.get("predictions_read") is not False
        ):
            raise FreezeError("existing monotonic timeout receipt differs from this deadline") from None
        payload = existing
    sentinel = {
        "panel_id": "runtime",
        "plan_key": "monotonic-budget",
        "run_id": "none",
        "seed": 0,
        "worker_index": -1,
        "worker_order": -1,
    }
    _record_global_failure(
        verified=verified,
        ledger_root=root,
        failed_row=sentinel,
        error_type="MonotonicBudgetExpired",
        execution_uuid=str(ledger["execution_uuid"]),
        trigger="monotonic_14h_deadline",
    )
    _terminalize_inflight_after_failure(
        verified=verified,
        ledger_root=root,
        error_type="MonotonicBudgetExpired",
        caused_by_plan_key="monotonic-budget",
        execution_uuid=str(ledger["execution_uuid"]),
    )
    _mark_unstarted_after_failure(
        verified=verified,
        ledger_root=root,
        caused_by_plan_key="monotonic-budget",
        execution_uuid=str(ledger["execution_uuid"]),
    )
    return {
        "path": str(target),
        "file_sha256": sha256_file(target),
        "timeout_digest": payload["timeout_digest"],
    }


def _mark_unstarted_after_failure(
    *,
    verified: VerifiedFreeze,
    ledger_root: Path,
    caused_by_plan_key: str,
    execution_uuid: str,
) -> None:
    for row in verified.all_rows:
        plan_key = str(row["plan_key"])
        start = _ledger_receipt_path(ledger_root, "start", plan_key)
        terminal = _ledger_receipt_path(ledger_root, "terminal", plan_key)
        if start.exists() or terminal.exists():
            continue
        receipt = _attempt_receipt(
            verified=verified,
            row=row,
            kind="terminal",
            state="not_started_after_failure",
            caused_by_plan_key=caused_by_plan_key,
            execution_uuid=execution_uuid,
        )
        with suppress(FreezeError):
            exclusive_json(terminal, receipt)


def _seal_outcome_files(path: Path) -> dict[str, dict[str, Any]]:
    """Cooperatively hide every outcome-bearing core artifact until ITT close."""

    result: dict[str, dict[str, Any]] = {}
    for name in ("metrics.jsonl", "predictions.jsonl", "summary.json"):
        target = path / name
        if target.is_symlink() or not target.is_file():
            raise FreezeError(f"completed G00-F run lacks direct outcome file before sealing: {name}")
        digest = sha256_file(target)
        size = target.stat().st_size
        os.chmod(target, 0)
        result[name] = {"path": str(target), "bytes": size, "sha256": digest, "sealed_mode": 0}
    return result


def _terminal_receipts(verified: VerifiedFreeze, ledger_root: Path) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in verified.all_rows:
        plan_key = str(row["plan_key"])
        path = _ledger_receipt_path(ledger_root, "terminal", plan_key)
        if path.is_file():
            result[plan_key] = strict_json(path, f"terminal receipt {plan_key}")
    return result


def _unseal_if_complete(verified: VerifiedFreeze, ledger_root: Path) -> None:
    receipts = _terminal_receipts(verified, ledger_root)
    if len(receipts) != 160 or any(receipt.get("state") != "complete" for receipt in receipts.values()):
        return
    target = ledger_root / "panel-unseal.json"
    paths: list[dict[str, Any]] = []
    validated_files: list[tuple[Path, Mapping[str, Any], Mapping[str, Any], str]] = []
    for row in verified.all_rows:
        terminal = receipts[str(row["plan_key"])]
        terminal_body = {key: value for key, value in terminal.items() if key != "receipt_digest"}
        start_path = _ledger_receipt_path(ledger_root, "start", str(row["plan_key"]))
        start = strict_json(start_path, "start receipt before outcome unseal")
        start_body = {key: value for key, value in start.items() if key != "receipt_digest"}
        if (
            set(terminal_body)
            != {
                "caused_by_plan_key",
                "error_type",
                "execution_uuid",
                "freeze_digest",
                "freeze_file_sha256",
                "g01_launch_authorized",
                "kind",
                "launch_receipt",
                "outcome_file_seals",
                "outcome_metrics_read",
                "panel_id",
                "plan_key",
                "predictions_read",
                "run_id",
                "schema",
                "schema_version",
                "seed",
                "state",
                "worker_index",
                "worker_order",
            }
            or terminal.get("schema") != ATTEMPT_RECEIPT_SCHEMA
            or terminal.get("schema_version") != ATTEMPT_RECEIPT_SCHEMA_VERSION
            or terminal.get("kind") != "terminal"
            or terminal.get("state") != "complete"
            or terminal.get("freeze_file_sha256") != verified.file_sha256
            or terminal.get("freeze_digest") != verified.digest
            or terminal.get("execution_uuid") != ledger_root.name
            or terminal.get("panel_id") != row["panel_id"]
            or terminal.get("plan_key") != row["plan_key"]
            or terminal.get("run_id") != row["run_id"]
            or terminal.get("seed") != row["seed"]
            or terminal.get("worker_index") != row["worker_index"]
            or terminal.get("worker_order") != row["worker_order"]
            or not isinstance(terminal.get("launch_receipt"), Mapping)
            or terminal.get("error_type") is not None
            or terminal.get("caused_by_plan_key") is not None
            or terminal.get("outcome_metrics_read") is not False
            or terminal.get("predictions_read") is not False
            or terminal.get("g01_launch_authorized") is not False
            or terminal.get("receipt_digest") != semantic_digest(terminal_body)
            or start.get("schema") != ATTEMPT_RECEIPT_SCHEMA
            or start.get("schema_version") != ATTEMPT_RECEIPT_SCHEMA_VERSION
            or start.get("kind") != "start"
            or start.get("state") != "started"
            or start.get("plan_key") != row["plan_key"]
            or start.get("execution_uuid") != ledger_root.name
            or start.get("freeze_digest") != verified.digest
            or start.get("receipt_digest") != semantic_digest(start_body)
            or start.get("launch_receipt") != terminal.get("launch_receipt")
        ):
            raise FreezeError("complete terminal receipt changed before outcome unseal")
        expected_files = terminal.get("outcome_file_seals")
        if not isinstance(expected_files, Mapping) or set(expected_files) != {
            "metrics.jsonl",
            "predictions.jsonl",
            "summary.json",
        }:
            raise FreezeError("complete terminal omits the exact sealed outcome inventory")
        for name in ("metrics.jsonl", "predictions.jsonl", "summary.json"):
            outcome_file = Path(str(row["artifact_path"])) / name
            expected = expected_files[name]
            if (
                not isinstance(expected, Mapping)
                or set(expected) != {"bytes", "path", "sealed_mode", "sha256"}
                or expected.get("path") != str(outcome_file)
                or not outcome_file.is_file()
                or outcome_file.is_symlink()
                or outcome_file.stat().st_size != expected.get("bytes")
                or stat.S_IMODE(outcome_file.stat().st_mode) not in {0, 0o400}
                or sha256_file(outcome_file) != expected.get("sha256")
            ):
                raise FreezeError("sealed outcome bytes changed before panel completion")
            validated_files.append((outcome_file, expected, row, name))
    for outcome_file, expected, row, name in validated_files:
        os.chmod(outcome_file, 0o400)
        paths.append(
            {
                "plan_key": row["plan_key"],
                "run_id": row["run_id"],
                "name": name,
                "bytes": expected["bytes"],
                "sha256": expected["sha256"],
                "unsealed_mode": 0o400,
            }
        )
    body = {
        "schema": PANEL_UNSEAL_SCHEMA,
        "schema_version": PANEL_UNSEAL_SCHEMA_VERSION,
        "freeze_file_sha256": verified.file_sha256,
        "freeze_digest": verified.digest,
        "terminal_complete_count": 160,
        "outcome_files": paths,
        "outcomes_inspected_before_unseal": False,
        "g01_launch_authorized": False,
    }
    expected_payload = {**body, "unseal_digest": semantic_digest(body)}
    try:
        exclusive_json(target, expected_payload)
    except FreezeError:
        # The final workers can observe all 160 terminal receipts at the same
        # time.  O_EXCL chooses one winner; every loser must verify, not fail
        # or overwrite, the winner's byte-equivalent semantic payload.
        existing: Mapping[str, Any] | None = None
        for _attempt in range(100):
            try:
                existing = strict_json(target, "concurrent G00-F panel unseal winner")
                break
            except FreezeError:
                time.sleep(0.01)
        if existing != expected_payload:
            raise FreezeError("concurrent G00-F panel unseal winner differs from expectation") from None


def _spec_lookup(verified: VerifiedFreeze) -> dict[str, RunSpec]:
    result: dict[str, RunSpec] = {}
    for _panel_id, specification in CONFIG_SPECS.items():
        config = load_config(verified.repo / str(specification["path"]))
        for spec in build_plan(config):
            result[spec.plan_key] = spec
    if len(result) != 160:
        raise FreezeError("could not reconstruct the exact 160 frozen RunSpecs")
    return result


def _authenticated_local_model_loader(
    model_receipts: Mapping[str, Mapping[str, Any]],
) -> Callable[..., tuple[Any, Any]]:
    """Build the only loader allowed to consume G00-F model weights."""

    if set(model_receipts) != set(CONFIG_SPECS):
        raise FreezeError("authenticated local model roots are required for G00-F execution")

    def exact_local_loader(
        model_config: Mapping[str, Any],
        update_config: Mapping[str, Any] | None = None,
        *,
        device_map: str | Mapping[str, Any] | None = None,
    ) -> tuple[Any, Any]:
        if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
            raise FreezeError("G00-F local model loading requires both offline-mode guards")
        requested = str(model_config.get("name", ""))
        revision = str(model_config.get("revision", ""))
        matches = [
            panel_id
            for panel_id, static in CONFIG_SPECS.items()
            if static["model"] == requested and static["revision"] == revision
        ]
        if len(matches) != 1 or bool(model_config.get("trust_remote_code", False)):
            raise FreezeError("G00-F attempted to load an unregistered model or remote code")
        panel_id = matches[0]
        root = Path(str(model_receipts[panel_id]["snapshot_root"])).resolve()
        try:
            from transformers import (  # type: ignore[import-not-found]
                AutoModelForCausalLM,
                AutoTokenizer,
            )
        except ImportError as error:  # pragma: no cover - frozen GPU runtime dependency
            raise FreezeError("frozen transformers dependency is absent") from error
        from goalzendo.modeling import apply_update_method, resolve_torch_dtype

        tokenizer = AutoTokenizer.from_pretrained(
            root,
            local_files_only=True,
            trust_remote_code=False,
        )
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise FreezeError("authenticated tokenizer defines neither pad nor EOS")
            tokenizer.pad_token = tokenizer.eos_token
        kwargs: dict[str, Any] = {
            "local_files_only": True,
            "torch_dtype": resolve_torch_dtype(model_config.get("dtype", "bfloat16")),
            "trust_remote_code": False,
        }
        if device_map is not None:
            kwargs["device_map"] = device_map
        model = AutoModelForCausalLM.from_pretrained(root, **kwargs)
        # A regular-file materialization has no hub metadata.  Attach only the
        # revision that the direct-leaf receipt already authenticated.
        model.config._commit_hash = revision
        if isinstance(getattr(tokenizer, "init_kwargs", None), dict):
            tokenizer.init_kwargs["_commit_hash"] = revision
        if update_config is not None:
            model = apply_update_method(model, update_config)
        return model, tokenizer

    return exact_local_loader


@contextmanager
def _exact_g00f_launch_scope(
    verified: VerifiedFreeze,
    model_receipts: Mapping[str, Mapping[str, Any]] | None = None,
) -> Iterator[None]:
    """Narrowly replace only G00-F's design guard in this process."""

    import goalzendo.experiment as experiment_module
    import goalzendo.runner as runner_module

    original = runner_module.assert_launch_unlocked
    experiment_any: Any = experiment_module
    original_loader = experiment_any.load_model_and_tokenizer
    allowed_cells = {str(row["cell_id"]) for row in verified.all_rows}

    def exact_guard(
        config: Mapping[str, Any],
        *,
        repo: str | Path,
        gate_artifact: str | Path | None = None,
    ) -> Any:
        if get_path(config, "experiment.id") != "g00f":
            return original(config, repo=repo, gate_artifact=gate_artifact)
        if (
            get_path(config, "run.launch_guard") != G00F_GUARD
            or get_path(config, "run.protocol_unlocked") is not False
        ):
            raise FreezeError("G00-F launch guard/config authorization changed")
        # Rebuild the one-cell plan with one seed solely to recover its cell
        # identity.  Execution itself receives only RunSpecs from the exact
        # frozen lookup below.
        probe = copy.deepcopy(dict(config))
        seed_values = get_path(probe, "run.seeds")
        if not isinstance(seed_values, Sequence) or not seed_values:
            raise FreezeError("G00-F guarded cell has no registered seeds")
        probe["run"]["seeds"] = [int(seed_values[0])]
        cells = build_plan(probe)
        if len(cells) != 1 or cells[0].cell_id not in allowed_cells:
            raise FreezeError("G00-F guarded cell is outside the exact frozen plan")
        if (
            implementation_provenance(repo).get("implementation_fingerprint")
            != FROZEN_GOALZENDO_IMPLEMENTATION_FINGERPRINT
        ):
            raise FreezeError("GoalZendo source changed after G00-F authentication")
        return None

    runner_module.assert_launch_unlocked = exact_guard
    if model_receipts is not None:
        experiment_any.load_model_and_tokenizer = _authenticated_local_model_loader(model_receipts)
    try:
        yield
    finally:
        runner_module.assert_launch_unlocked = original
        experiment_any.load_model_and_tokenizer = original_loader


def _run_binding_payload(
    *,
    verified: VerifiedFreeze,
    row: Mapping[str, Any],
    launch_receipt: Mapping[str, Any],
    model_receipts: Mapping[str, Mapping[str, Any]],
    attempt_start: Mapping[str, Any],
    ownership_lease: Mapping[str, Any],
    execution_uuid: str,
) -> dict[str, Any]:
    body = {
        "schema": RUN_BINDING_SCHEMA,
        "schema_version": RUN_BINDING_SCHEMA_VERSION,
        "freeze_file_sha256": verified.file_sha256,
        "freeze_digest": verified.digest,
        "execution_uuid": execution_uuid,
        "plan_key": row["plan_key"],
        "run_id": row["run_id"],
        "panel_id": row["panel_id"],
        "worker_index": row["worker_index"],
        "worker_order": row["worker_order"],
        "launch_receipt": dict(launch_receipt),
        "model_receipts": {key: dict(value) for key, value in sorted(model_receipts.items())},
        "profile_selection": copy.deepcopy(dict(verified.selection_receipt or {})),
        "profile_qualification_cleanup": copy.deepcopy(
            dict(launch_receipt.get("profile_qualification_cleanup", {}))
        ),
        "selected_profile": verified.selected_profile,
        "attempt_start": dict(attempt_start),
        "ownership_lease": dict(ownership_lease),
        "g01_launch_authorized": False,
    }
    return {**body, "binding_digest": semantic_digest(body)}


def run_worker(
    *,
    verified: VerifiedFreeze,
    worker_index: int,
    launch_receipt_path: str | Path,
    model_receipt_paths: Mapping[str, str | Path],
    ledger_root: str | Path,
    dry_run: bool = False,
) -> tuple[Mapping[str, Any], ...]:
    """Run one exact mixed-model worker schedule, with no reassignment path."""

    launch = verify_launch_receipt(
        verified=verified,
        worker_index=worker_index,
        receipt_path=launch_receipt_path,
    )
    if set(model_receipt_paths) != set(CONFIG_SPECS):
        raise FreezeError("worker requires exactly both frozen model snapshot receipts")
    model_receipts = {
        panel_id: verify_model_snapshot_receipt(
            verified=verified,
            panel_id=panel_id,
            receipt_path=model_receipt_paths[panel_id],
        )
        for panel_id in CONFIG_SPECS
    }
    lookup = _spec_lookup(verified)
    rows = verified.worker_rows(worker_index)
    specs = [lookup[str(row["plan_key"])] for row in rows]
    ledger = verify_attempt_ledger(verified=verified, ledger_root=ledger_root)
    ledger_path = Path(str(ledger["ledger_root"]))
    execution_uuid = str(ledger["execution_uuid"])
    if dry_run:
        with _exact_g00f_launch_scope(verified, model_receipts):
            dry_outcomes = execute_plan(specs, repo=verified.repo, dry_run=True)
        return tuple(outcome.as_dict() for outcome in dry_outcomes)

    worker_outcomes: list[Mapping[str, Any]] = []
    for row, spec in zip(rows, specs, strict=True):
        global_stop = ledger_path / "global-stop.json"
        if global_stop.is_file():
            stop = strict_json(global_stop, "G00-F global stop")
            if stop.get("freeze_digest") != verified.digest:
                raise FreezeError("global stop is not bound to this freeze")
            _mark_unstarted_after_failure(
                verified=verified,
                ledger_root=ledger_path,
                caused_by_plan_key=str(stop.get("failed_plan_key", "")),
                execution_uuid=execution_uuid,
            )
            worker_outcomes.extend(
                {
                    "plan_key": remaining["plan_key"],
                    "run_id": remaining["run_id"],
                    "path": remaining["artifact_path"],
                    "seed": remaining["seed"],
                    "state": "not_started_after_failure",
                    "error_type": None,
                }
                for remaining in rows[len(worker_outcomes) :]
            )
            break
        start_path = _ledger_receipt_path(ledger_path, "start", str(row["plan_key"]))
        terminal_path = _ledger_receipt_path(ledger_path, "terminal", str(row["plan_key"]))
        lease_path = ledger_path / "leases" / f"{row['plan_key']}.json"
        try:
            store = RunStore(
                get_path(spec.config, "run.output_root"),
                spec.config,
                spec.seed,
                verified.repo,
            )
            if store.run_id != row["run_id"] or str(store.path) != row["artifact_path"]:
                raise FreezeError("runtime RunStore identity differs from the frozen plan")
            if store.path.exists():
                raise FreezeError("G00-F forbids pre-existing run artifacts or post-start resume")
            if start_path.exists() or terminal_path.exists() or lease_path.exists():
                raise FreezeError("G00-F permits exactly one append-only attempt per plan key")
            enforce_monotonic_budget(ledger)
            lease_body = {
                "schema": "goalzendo.g00f_h200_run_ownership_lease",
                "schema_version": 1,
                "execution_uuid": execution_uuid,
                "freeze_file_sha256": verified.file_sha256,
                "freeze_digest": verified.digest,
                "panel_id": row["panel_id"],
                "plan_key": row["plan_key"],
                "run_id": row["run_id"],
                "worker_index": row["worker_index"],
                "worker_order": row["worker_order"],
                "owner_pid": os.getpid(),
                "acquired_monotonic_ns": time.monotonic_ns(),
                "g01_launch_authorized": False,
            }
            lease_payload = {**lease_body, "lease_digest": semantic_digest(lease_body)}
            exclusive_json(lease_path, lease_payload)
            if global_stop.is_file():
                raise FreezeError("G00-F coordinator stopped execution after lease acquisition")
            lease_summary = {
                "path": str(lease_path),
                "file_sha256": sha256_file(lease_path),
                "lease_digest": lease_payload["lease_digest"],
            }
            start_payload = _attempt_receipt(
                verified=verified,
                row=row,
                kind="start",
                state="started",
                launch_receipt=launch,
                execution_uuid=execution_uuid,
            )
            exclusive_json(start_path, start_payload)
            start_summary = {
                "path": str(start_path),
                "file_sha256": sha256_file(start_path),
                "receipt_digest": start_payload["receipt_digest"],
            }
            binding_path = store.path / "g00f-freeze-binding.json"
            binding_payload = _run_binding_payload(
                verified=verified,
                row=row,
                launch_receipt=launch,
                model_receipts=model_receipts,
                attempt_start=start_summary,
                ownership_lease=lease_summary,
                execution_uuid=execution_uuid,
            )
            atomic_json(binding_path, binding_payload)
            with _exact_g00f_launch_scope(verified, model_receipts):
                (outcome,) = execute_plan([spec], repo=verified.repo)
            if outcome.state not in {"complete", "skipped"} or outcome.state == "skipped":
                raise FreezeError("fresh G00-F attempt did not complete exactly once")
            seals = _seal_outcome_files(store.path)
            terminal = _attempt_receipt(
                verified=verified,
                row=row,
                kind="terminal",
                state="complete",
                launch_receipt=launch,
                execution_uuid=execution_uuid,
            )
            terminal_body = {key: value for key, value in terminal.items() if key != "receipt_digest"}
            terminal_body["outcome_file_seals"] = seals
            terminal = {**terminal_body, "receipt_digest": semantic_digest(terminal_body)}
            exclusive_json(terminal_path, terminal)
            worker_outcomes.append(outcome.as_dict())
        except BaseException as error:
            if start_path.is_file() and not terminal_path.exists():
                terminal = _attempt_receipt(
                    verified=verified,
                    row=row,
                    kind="terminal",
                    state="failed",
                    launch_receipt=launch,
                    error_type=type(error).__name__,
                    execution_uuid=execution_uuid,
                )
                exclusive_json(terminal_path, terminal)
            elif lease_path.is_file() and not terminal_path.exists():
                terminal = _attempt_receipt(
                    verified=verified,
                    row=row,
                    kind="terminal",
                    state="failed_before_start_receipt",
                    error_type=type(error).__name__,
                    execution_uuid=execution_uuid,
                )
                exclusive_json(terminal_path, terminal)
            _record_global_failure(
                verified=verified,
                ledger_root=ledger_path,
                failed_row=row,
                error_type=type(error).__name__,
                execution_uuid=execution_uuid,
            )
            _mark_unstarted_after_failure(
                verified=verified,
                ledger_root=ledger_path,
                caused_by_plan_key=str(row["plan_key"]),
                execution_uuid=execution_uuid,
            )
            raise
        _unseal_if_complete(verified, ledger_path)
    return tuple(worker_outcomes)


__all__ = [
    "CONFIG_SPECS",
    "EXECUTION_AND_GATE_CONTRACT",
    "EXPECTED_ALL_VIEWS",
    "EXPECTED_INFORMATIVE_VIEWS",
    "FREEZE_SCHEMA",
    "FREEZE_SCHEMA_VERSION",
    "FROZEN_GOALZENDO_IMPLEMENTATION_FINGERPRINT",
    "G00F_GUARD",
    "H200_HOUR_CEILING",
    "H200_MODULE_INVOCATION_CONTRACT",
    "MODEL_INTEGRATION_AUDIT_SCHEMA",
    "MODEL_LEAF_FILES",
    "MODEL_RECEIPT_SCHEMA",
    "MODEL_RUNTIME_IDENTITIES",
    "PANEL_UNSEAL_SCHEMA",
    "RUNPOD_PROVISIONING_CONTRACT",
    "RUNPOD_PROVISION_EVIDENCE",
    "RUNS_PER_PANEL",
    "TOKENIZER_RUNTIME_IDENTITY",
    "WALL_CEILING_SECONDS",
    "WORKER_COUNT",
    "FreezeError",
    "VerifiedFreeze",
    "atomic_json",
    "bind_runpod_provision_receipt",
    "canonical_json_bytes",
    "canonical_runpod_create_command",
    "create_model_integration_audit",
    "create_model_snapshot_receipt",
    "create_runpod_provision_receipt",
    "create_runpod_ssh_identity_receipt",
    "exclusive_json",
    "expected_plan_rows",
    "initialize_attempt_ledger",
    "require_sha256",
    "run_worker",
    "semantic_digest",
    "sha256_file",
    "strict_json",
    "verify_attempt_ledger",
    "verify_detached_supervisor_receipts",
    "verify_freeze",
    "verify_launch_receipt",
    "verify_model_integration_audit",
    "verify_model_snapshot_receipt",
    "verify_runpod_gpu_catalog_snapshot",
    "verify_runpod_provision_receipt",
    "verify_runpodctl_version_capture",
    "write_plan_files",
]
