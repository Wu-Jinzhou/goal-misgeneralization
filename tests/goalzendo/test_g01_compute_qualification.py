from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest

import goalzendo_g01_qualification.qualification as qualification
from goalzendo_g01_qualification import (
    PRODUCTION_SHAPE_DIGEST,
    QualificationError,
    build_engineering_schedule,
    build_review_contract,
    canonical_json_bytes,
    execute_qualification,
    memory_gate,
    project_compute,
    project_storage,
    run_qualification,
    strict_canonical_json_bytes,
    validate_campaign,
    validate_report_bytes,
    validate_review_contract_bytes,
    validate_run_receipt,
    validate_run_receipt_bytes,
)

REPO = Path(__file__).resolve().parents[2]
ENTRYPOINT = REPO / "runs/goalzendo/run_g01_compute_qualification.py"
CAMPAIGN_UUID = "11111111-1111-4111-8111-111111111111"
QUALIFICATION_UUID = "22222222-2222-4222-8222-222222222222"
SHA = "a" * 64
GPU_UUIDS = tuple(f"GPU-00000000-0000-0000-0000-{index:012x}" for index in range(8))


def _receipt(
    worker_count: int,
    row_index: int,
    *,
    pair_start_seconds: int | None = None,
) -> dict[str, Any]:
    schedule = build_engineering_schedule(worker_count)
    row = schedule.rows[row_index]
    wave_index = row.pair_index // worker_count
    release_seconds = wave_index * 30_000
    start_seconds = release_seconds if pair_start_seconds is None else pair_start_seconds
    start_seconds += row.algorithm_order_index * 4_000
    body: dict[str, Any] = {
        "schema": qualification.RUN_RECEIPT_SCHEMA,
        "schema_version": 1,
        "study_id": qualification.STUDY_ID,
        "campaign_uuid": CAMPAIGN_UUID,
        "qualification_uuid": QUALIFICATION_UUID,
        "candidate_id": "candidate-0",
        "candidate_profile_id": "h200x8" if worker_count == 8 else "h200x4",
        "candidate_priority": 0 if worker_count == 8 else 2,
        "candidate_policy_digest": qualification.CANDIDATE_POLICY_DIGEST,
        "prior_candidate_dispositions_digest": (
            qualification.digest([])
            if worker_count == 8
            else qualification.digest(
                [
                    {
                        "profile_id": "h200x8",
                        "state": "consumed_failure",
                        "registrar_record_sha256": "b" * 64,
                    },
                    {
                        "profile_id": "h100-hbm3x8",
                        "state": "skipped_no_stock",
                        "registrar_record_sha256": "c" * 64,
                    },
                ]
            )
        ),
        "gpu_id": "NVIDIA H200",
        "cloud_type": "SECURE",
        "data_center_id": "US-CA-2",
        "network_volume_id": "volume_AbC123",
        "network_volume_type": "HIGH_PERFORMANCE",
        "network_volume_mount": "/workspace",
        "qualification_ceiling_seconds": 79_200 if worker_count == 8 else 151_200,
        "compute_freeze_sha256": SHA,
        "campaign_intent_sha256": SHA,
        "provision_receipt_sha256": SHA,
        "source_runtime_model_bindings_digest": SHA,
        "fixed_control_envelope_manifest_digest": SHA,
        "worker_count": worker_count,
        "schedule_digest": schedule.schedule_digest,
        "row_index": row.row_index,
        "pair_index": row.pair_index,
        "worker_index": row.worker_index,
        "algorithm_order_index": row.algorithm_order_index,
        "wave_index": wave_index,
        "wave_release_monotonic_ns": release_seconds * 1_000_000_000,
        "gpu_uuid": GPU_UUIDS[row.worker_index],
        "algorithm": row.algorithm,
        "rule_family": row.rule_family,
        "q_p_basis_points": row.q_p_basis_points,
        "engineering_seed": row.engineering_seed,
        "production_shape_digest": PRODUCTION_SHAPE_DIGEST,
        "started_monotonic_ns": start_seconds * 1_000_000_000,
        "completed_monotonic_ns": (start_seconds + 4_000) * 1_000_000_000,
        "wall_ns": 4_000 * 1_000_000_000,
        "phase_wall_ns": {
            "model_and_data_materialization": 100 * 1_000_000_000,
            "optimization": 3_000 * 1_000_000_000,
            "evaluation_and_serialization": 800 * 1_000_000_000,
            "cuda_sync_and_teardown": 100 * 1_000_000_000,
        },
        "maximum_reserved_gpu_bytes": 60 << 30,
        "total_gpu_bytes": 80 << 30,
        "maximum_peak_rss_bytes": 10 << 30,
        "maximum_live_bytes": 12_000_000_000,
        "final_bytes": 10_000_000_000,
        "maximum_live_inodes": 8_001,
        "final_inodes": 8_000,
        "bytes_written": 20_000_000_000,
        "completion_file_count": 12,
        "state": "complete",
        "retry": False,
        "resume": False,
        "scientific_values_persisted": False,
        "scientific_values_read_by_control": False,
        "scientific_value_branching": False,
        "metric_prediction_stream_padded_to_frozen_bytes": True,
        "model_weight_serialization_uncompressed": True,
        "control_log_fixed_schema": True,
        "raw_scientific_tree_absent": True,
        "forbidden_field_scan_passed": True,
    }
    return {**body, "receipt_digest": qualification.digest(body)}


def _campaign_kwargs(worker_count: int) -> dict[str, Any]:
    if worker_count == 8:
        return {
            "candidate_profile_id": "h200x8",
            "candidate_priority": 0,
            "prior_candidate_dispositions": [],
            "qualification_ceiling_seconds": 79_200,
        }
    return {
        "candidate_profile_id": "h200x4",
        "candidate_priority": 2,
        "prior_candidate_dispositions": [
            {
                "profile_id": "h200x8",
                "state": "consumed_failure",
                "registrar_record_sha256": "b" * 64,
            },
            {
                "profile_id": "h100-hbm3x8",
                "state": "skipped_no_stock",
                "registrar_record_sha256": "c" * 64,
            },
        ],
        "qualification_ceiling_seconds": 151_200,
    }


def _validate_campaign(receipts: list[dict[str, Any]], worker_count: int = 8) -> Any:
    return validate_campaign(
        receipts,
        worker_count=worker_count,
        sorted_gpu_uuids=GPU_UUIDS[:worker_count],
        expected_campaign_uuid=CAMPAIGN_UUID,
        expected_qualification_uuid=QUALIFICATION_UUID,
        expected_candidate_id="candidate-0",
        gpu_id="NVIDIA H200",
        cloud_type="SECURE",
        data_center_id="US-CA-2",
        network_volume_id="volume_AbC123",
        network_volume_type="HIGH_PERFORMANCE",
        network_volume_mount="/workspace",
        expected_compute_freeze_sha256=SHA,
        expected_campaign_intent_sha256=SHA,
        expected_provision_receipt_sha256=SHA,
        expected_source_runtime_model_bindings_digest=SHA,
        expected_fixed_control_envelope_manifest_digest=SHA,
        expected_final_bytes_per_run=10_000_000_000,
        expected_maximum_live_bytes_per_run=12_000_000_000,
        expected_final_inodes_per_run=8_000,
        expected_maximum_live_inodes_per_run=8_001,
        expected_bytes_written_per_run=20_000_000_000,
        expected_completion_file_count=12,
        price_micro_usd_per_gpu_hour=1_000_000,
        operator_maximum_wall_seconds=999_999,
        operator_maximum_gpu_seconds=9_999_999,
        operator_maximum_cost_micro_usd=9_999_999_999,
        candidate_maximum_wall_seconds=999_999,
        candidate_maximum_gpu_seconds=9_999_999,
        candidate_maximum_cost_micro_usd=9_999_999_999,
        fixed_bytes=100_000_000_000,
        fixed_inodes=40_003,
        available_free_bytes=2_000_000_000_000,
        available_free_inodes=2_000_000,
        **_campaign_kwargs(worker_count),
    )


def test_exact_engineering_schedules() -> None:
    expected_digests = {
        4: "e610df289b4d391ad8c25d6294cb45bf7ee630631cb95fc8bc9d94ba883c5887",
        8: "50d725e32ae8900af6a74ccb492a3f03dd9e8b64f367873b7238987efbbab426",
    }
    for worker_count in (4, 8):
        schedule = build_engineering_schedule(worker_count)
        assert schedule.schedule_digest == expected_digests[worker_count]
        assert len(schedule.rows) == 16
        assert {row.engineering_seed for row in schedule.rows} == set(range(8_611_107, 8_611_115))
        for pair_index in range(8):
            pair = [row for row in schedule.rows if row.pair_index == pair_index]
            assert pair[0].worker_index == pair_index % worker_count
            expected_order = ["sft", "outcome_rl"] if pair[0].worker_index % 2 == 0 else ["outcome_rl", "sft"]
            assert [row.algorithm for row in pair] == expected_order
    assert [row.pair_index for row in build_engineering_schedule(4).for_worker(0)][::2] == [0, 4]
    for invalid in (True, 4.0, "4", 0, 6):
        with pytest.raises(QualificationError):
            build_engineering_schedule(invalid)  # type: ignore[arg-type]


def test_compute_projection_exact_deadline_bands_and_type_rejection() -> None:
    vectors = (
        (4, 8_177, 172_800, 691_200, 192_000_000),
        (4, 8_178, 259_200, 1_036_800, 288_000_000),
        (4, 12_445, 345_600, 1_382_400, 384_000_000),
        (4, 16_712, 432_000, 1_728_000, 480_000_000),
        (4, 20_978, None, None, None),
        (8, 15_333, 172_800, 1_382_400, 384_000_000),
        (8, 15_334, 259_200, 2_073_600, 576_000_000),
        (8, 23_334, 345_600, 2_764_800, 768_000_000),
        (8, 31_334, 432_000, 3_456_000, 960_000_000),
        (8, 39_334, None, None, None),
    )
    for worker_count, qmax, deadline, gpu_seconds, cost in vectors:
        projection = project_compute(
            worker_count=worker_count,
            maximum_pair_wall_seconds=qmax,
            price_micro_usd_per_gpu_hour=1_000_000,
            operator_maximum_wall_seconds=999_999,
            operator_maximum_gpu_seconds=9_999_999,
            operator_maximum_cost_micro_usd=9_999_999_999,
            candidate_maximum_wall_seconds=999_999,
            candidate_maximum_gpu_seconds=9_999_999,
            candidate_maximum_cost_micro_usd=9_999_999_999,
        )
        assert projection.selected_deadline_seconds == deadline
        assert projection.projected_study_gpu_seconds == gpu_seconds
        assert projection.projected_cost_micro_usd == cost
        assert projection.g01_launch_authorized is False
    with pytest.raises(QualificationError):
        project_compute(
            worker_count=4,
            maximum_pair_wall_seconds=True,
            price_micro_usd_per_gpu_hour=1,
            operator_maximum_wall_seconds=1,
            operator_maximum_gpu_seconds=1,
            operator_maximum_cost_micro_usd=1,
            candidate_maximum_wall_seconds=1,
            candidate_maximum_gpu_seconds=1,
            candidate_maximum_cost_micro_usd=1,
        )


def test_memory_and_storage_boundary_vectors() -> None:
    assert not memory_gate(total_gpu_bytes=(8 << 30) - 1, maximum_reserved_gpu_bytes=0)
    assert memory_gate(total_gpu_bytes=8 << 30, maximum_reserved_gpu_bytes=0)
    assert not memory_gate(total_gpu_bytes=8 << 30, maximum_reserved_gpu_bytes=1)
    assert memory_gate(total_gpu_bytes=80 << 30, maximum_reserved_gpu_bytes=72 << 30)
    assert not memory_gate(total_gpu_bytes=80 << 30, maximum_reserved_gpu_bytes=(72 << 30) + 1)
    for worker_count, expected_bytes, expected_inodes in (
        (4, 1_635_000_000_000, 1_250_009),
        (8, 1_645_000_000_000, 1_250_014),
    ):
        projection = project_storage(
            worker_count=worker_count,
            final_bytes_per_run=10_000_000_000,
            maximum_live_bytes_per_run=12_000_000_000,
            fixed_bytes=100_000_000_000,
            final_inodes_per_run=8_000,
            maximum_live_inodes_per_run=8_001,
            fixed_inodes=40_003,
            maximum_bytes_written_per_run=20_000_000_000,
        )
        assert projection.required_free_bytes == expected_bytes
        assert projection.required_free_inodes == expected_inodes
        assert projection.projected_aggregate_write_bytes == 3_000_000_000_000
    for changes in (
        {"maximum_live_bytes_per_run": 9},
        {"maximum_live_inodes_per_run": 7},
        {"maximum_bytes_written_per_run": 9},
        {"final_bytes_per_run": True},
    ):
        arguments: dict[str, Any] = {
            "worker_count": 4,
            "final_bytes_per_run": 10,
            "maximum_live_bytes_per_run": 12,
            "fixed_bytes": 0,
            "final_inodes_per_run": 8,
            "maximum_live_inodes_per_run": 9,
            "fixed_inodes": 0,
            "maximum_bytes_written_per_run": 20,
            **changes,
        }
        with pytest.raises(QualificationError):
            project_storage(**arguments)


def test_contract_is_deeply_fresh_canonical_and_nonauthorizing() -> None:
    first = build_review_contract()
    second = build_review_contract()
    first["production_shape"]["eval_steps"].append(999)
    assert 999 not in second["production_shape"]["eval_steps"]
    payload = canonical_json_bytes(second)
    assert validate_review_contract_bytes(payload) == second
    assert strict_canonical_json_bytes(payload, "contract") == second
    for altered in (
        b'{"a":1,"a":1}',
        json.dumps(second, indent=2, sort_keys=True).encode(),
        canonical_json_bytes({**second, "schema_version": True}),
    ):
        with pytest.raises(QualificationError):
            validate_review_contract_bytes(altered)
    assert second["authorization"] == {
        "qualification_only": True,
        "g01_scientific_outcomes_seen": False,
        "g01_itt_created": False,
        "g01_training_authorized": False,
        "checkpoint_b_launch_authorized": False,
        "runtime_provision_frozen": False,
    }


def test_run_receipt_is_exact_schedule_shape_topology_and_outcome_bound() -> None:
    receipt = _receipt(8, 0)
    assert validate_run_receipt(receipt) == receipt
    assert validate_run_receipt_bytes(canonical_json_bytes(receipt)) == receipt
    with pytest.raises(QualificationError):
        validate_run_receipt_bytes(json.dumps(receipt, indent=2).encode())
    mutations: list[tuple[str, Any]] = [
        ("worker_index", 7),
        ("engineering_seed", 8_611_114),
        ("q_p_basis_points", 10_000),
        ("production_shape_digest", "0" * 64),
        ("gpu_uuid", "GPU-accuracy-reward-leak"),
        ("candidate_profile_id", "h200x4"),
        ("data_center_id", "us-ca-2"),
        ("maximum_reserved_gpu_bytes", 81 << 30),
        ("final_bytes", 13_000_000_000),
        ("model_weight_serialization_uncompressed", 1),
        ("raw_scientific_tree_absent", False),
    ]
    for field, replacement in mutations:
        changed = deepcopy(receipt)
        changed[field] = replacement
        body = {key: value for key, value in changed.items() if key != "receipt_digest"}
        changed["receipt_digest"] = qualification.digest(body)
        with pytest.raises(QualificationError):
            validate_run_receipt(changed)
    changed = deepcopy(receipt)
    changed["reward"] = 1
    with pytest.raises(QualificationError):
        validate_run_receipt(changed)


def test_aggregate_derives_metrics_but_cannot_qualify_or_authorize() -> None:
    receipts = [_receipt(8, index) for index in range(16)]
    report = _validate_campaign(receipts)
    assert report.maximum_pair_wall_seconds == 8_000
    assert report.supplied_evidence_gates_satisfied
    assert report.compute_route_qualified is False
    assert report.evidence_externally_authenticated is False
    assert report.same_final_pod_verified is False
    assert report.g01_launch_authorized is False
    assert report.as_dict()["compute_route_qualified"] is False
    assert report.as_dict()["g01_launch_authorized"] is False
    assert type(report.input_bindings["sorted_gpu_uuids"]) is tuple
    immutable_uuids = cast(Any, report.input_bindings["sorted_gpu_uuids"])
    with pytest.raises(AttributeError):
        immutable_uuids.append("GPU-deadbeef")
    assert (
        validate_report_bytes(canonical_json_bytes(report.as_dict()), expected_report=report)
        == report.as_dict()
    )
    forged_report = report.as_dict()
    forged_report["input_bindings"]["price_micro_usd_per_gpu_hour"] = 1
    forged_body = {key: value for key, value in forged_report.items() if key != "report_digest"}
    forged_report["report_digest"] = qualification.digest(forged_body)
    with pytest.raises(QualificationError):
        validate_report_bytes(canonical_json_bytes(forged_report), expected_report=report)
    assert "g01_launch_authorized" not in {field.name for field in dataclasses.fields(report)}
    assert "g01_launch_authorized" not in {
        field.name
        for field in dataclasses.fields(
            project_compute(
                worker_count=8,
                maximum_pair_wall_seconds=1,
                price_micro_usd_per_gpu_hour=1,
                operator_maximum_wall_seconds=999_999,
                operator_maximum_gpu_seconds=9_999_999,
                operator_maximum_cost_micro_usd=9_999_999,
                candidate_maximum_wall_seconds=999_999,
                candidate_maximum_gpu_seconds=9_999_999,
                candidate_maximum_cost_micro_usd=9_999_999,
            )
        )
    }


def test_aggregate_rejects_missing_swapped_crossbound_or_staggered_rows() -> None:
    receipts = [_receipt(8, index) for index in range(16)]
    for changed in (receipts[:-1], [*receipts[:1], *receipts[:0:-1]]):
        with pytest.raises(QualificationError):
            _validate_campaign(changed)
    changed = deepcopy(receipts)
    changed[0]["network_volume_id"] = "different"
    body = {key: value for key, value in changed[0].items() if key != "receipt_digest"}
    changed[0]["receipt_digest"] = qualification.digest(body)
    with pytest.raises(QualificationError, match="cross-binding"):
        _validate_campaign(changed)
    staggered = [_receipt(8, index) for index in range(16)]
    for row_index in range(2, 16):
        row = staggered[row_index]
        if row["wave_index"] == 0:
            row["started_monotonic_ns"] += 3_999 * 1_000_000_000
            row["completed_monotonic_ns"] += 3_999 * 1_000_000_000
            body = {key: value for key, value in row.items() if key != "receipt_digest"}
            row["receipt_digest"] = qualification.digest(body)
    with pytest.raises(QualificationError, match="common release"):
        _validate_campaign(staggered)
    idle_padded = [_receipt(8, index) for index in range(16)]
    idle_padded[1]["started_monotonic_ns"] += 1
    idle_padded[1]["completed_monotonic_ns"] += 1
    body = {key: value for key, value in idle_padded[1].items() if key != "receipt_digest"}
    idle_padded[1]["receipt_digest"] = qualification.digest(body)
    with pytest.raises(QualificationError, match="gapless"):
        _validate_campaign(idle_padded)
    case_aliases = list(GPU_UUIDS)
    case_aliases[0] = "GPU-ABCDEFAB-0000-0000-0000-000000000000"
    with pytest.raises(QualificationError, match="lowercase"):
        validate_campaign(
            receipts,
            worker_count=8,
            sorted_gpu_uuids=case_aliases,
            expected_campaign_uuid=CAMPAIGN_UUID,
            expected_qualification_uuid=QUALIFICATION_UUID,
            expected_candidate_id="candidate-0",
            candidate_profile_id="h200x8",
            candidate_priority=0,
            gpu_id="NVIDIA H200",
            cloud_type="SECURE",
            data_center_id="US-CA-2",
            network_volume_id="volume_AbC123",
            network_volume_type="HIGH_PERFORMANCE",
            network_volume_mount="/workspace",
            qualification_ceiling_seconds=79_200,
            prior_candidate_dispositions=[],
            expected_compute_freeze_sha256=SHA,
            expected_campaign_intent_sha256=SHA,
            expected_provision_receipt_sha256=SHA,
            expected_source_runtime_model_bindings_digest=SHA,
            expected_fixed_control_envelope_manifest_digest=SHA,
            expected_final_bytes_per_run=10_000_000_000,
            expected_maximum_live_bytes_per_run=12_000_000_000,
            expected_final_inodes_per_run=8_000,
            expected_maximum_live_inodes_per_run=8_001,
            expected_bytes_written_per_run=20_000_000_000,
            expected_completion_file_count=12,
            price_micro_usd_per_gpu_hour=1_000_000,
            operator_maximum_wall_seconds=999_999,
            operator_maximum_gpu_seconds=9_999_999,
            operator_maximum_cost_micro_usd=9_999_999_999,
            candidate_maximum_wall_seconds=999_999,
            candidate_maximum_gpu_seconds=9_999_999,
            candidate_maximum_cost_micro_usd=9_999_999_999,
            fixed_bytes=100_000_000_000,
            fixed_inodes=40_003,
            available_free_bytes=2_000_000_000_000,
            available_free_inodes=2_000_000,
        )


def test_w4_two_wave_aggregate_and_priority_dispositions() -> None:
    receipts = [_receipt(4, index) for index in range(16)]
    report = _validate_campaign(receipts, worker_count=4)
    assert report.candidate_profile_id == "h200x4"
    assert report.candidate_priority == 2
    assert report.qualification_ceiling_seconds == 151_200
    assert report.supplied_evidence_gates_satisfied
    assert report.compute_route_qualified is False

    barrier_violation = deepcopy(receipts)
    for index in range(8, 16):
        barrier_violation[index]["wave_release_monotonic_ns"] = 7_999 * 1_000_000_000
        barrier_violation[index]["started_monotonic_ns"] -= 22_001 * 1_000_000_000
        barrier_violation[index]["completed_monotonic_ns"] -= 22_001 * 1_000_000_000
        body = {key: value for key, value in barrier_violation[index].items() if key != "receipt_digest"}
        barrier_violation[index]["receipt_digest"] = qualification.digest(body)
    with pytest.raises(QualificationError, match=r"overlapping|rendezvous"):
        _validate_campaign(barrier_violation, worker_count=4)

    with pytest.raises(QualificationError, match="earlier candidate"):
        validate_campaign(
            receipts,
            worker_count=4,
            sorted_gpu_uuids=GPU_UUIDS[:4],
            expected_campaign_uuid=CAMPAIGN_UUID,
            expected_qualification_uuid=QUALIFICATION_UUID,
            expected_candidate_id="candidate-0",
            candidate_profile_id="h200x4",
            candidate_priority=2,
            gpu_id="NVIDIA H200",
            cloud_type="SECURE",
            data_center_id="US-CA-2",
            network_volume_id="volume_AbC123",
            network_volume_type="HIGH_PERFORMANCE",
            network_volume_mount="/workspace",
            qualification_ceiling_seconds=151_200,
            prior_candidate_dispositions=[],
            expected_compute_freeze_sha256=SHA,
            expected_campaign_intent_sha256=SHA,
            expected_provision_receipt_sha256=SHA,
            expected_source_runtime_model_bindings_digest=SHA,
            expected_fixed_control_envelope_manifest_digest=SHA,
            expected_final_bytes_per_run=10_000_000_000,
            expected_maximum_live_bytes_per_run=12_000_000_000,
            expected_final_inodes_per_run=8_000,
            expected_maximum_live_inodes_per_run=8_001,
            expected_bytes_written_per_run=20_000_000_000,
            expected_completion_file_count=12,
            price_micro_usd_per_gpu_hour=1_000_000,
            operator_maximum_wall_seconds=999_999,
            operator_maximum_gpu_seconds=9_999_999,
            operator_maximum_cost_micro_usd=9_999_999_999,
            candidate_maximum_wall_seconds=999_999,
            candidate_maximum_gpu_seconds=9_999_999,
            candidate_maximum_cost_micro_usd=9_999_999_999,
            fixed_bytes=100_000_000_000,
            fixed_inodes=40_003,
            available_free_bytes=2_000_000_000_000,
            available_free_inodes=2_000_000,
        )


def test_aggregate_uses_maximum_per_run_transient_storage_delta() -> None:
    receipts = [_receipt(8, index) for index in range(16)]
    receipts[0]["final_bytes"] = 1
    receipts[0]["maximum_live_bytes"] = 4_000_000_001
    receipts[0]["bytes_written"] = 4_000_000_001
    receipts[1]["final_bytes"] = 10_000_000_000
    receipts[1]["maximum_live_bytes"] = 10_000_000_001
    for index in (0, 1):
        body = {key: value for key, value in receipts[index].items() if key != "receipt_digest"}
        receipts[index]["receipt_digest"] = qualification.digest(body)
    with pytest.raises(QualificationError, match="fixed control envelope"):
        _validate_campaign(receipts)


def test_same_manifest_fixed_envelope_size_or_count_mutation_rejects() -> None:
    receipts = [_receipt(8, index) for index in range(16)]
    for field, replacement in (
        ("final_bytes", 9_999_999_999),
        ("final_inodes", 7_999),
        ("completion_file_count", 11),
    ):
        changed = deepcopy(receipts)
        changed[0][field] = replacement
        body = {key: value for key, value in changed[0].items() if key != "receipt_digest"}
        changed[0]["receipt_digest"] = qualification.digest(body)
        with pytest.raises(QualificationError, match="fixed control envelope"):
            _validate_campaign(changed)


def test_all_high_level_execution_surfaces_refuse_before_work(tmp_path: Path) -> None:
    for function in (run_qualification, execute_qualification):
        with pytest.raises(QualificationError, match=qualification.REFUSAL):
            function()
        source = ast.parse(Path(qualification.__file__).read_text(encoding="utf-8"))
        definition = next(
            node
            for node in source.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function.__name__
        )
        assert isinstance(definition.body[0], ast.Raise)
    result = subprocess.run(
        [sys.executable, str(ENTRYPOINT), "--ignored"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert qualification.REFUSAL in result.stderr
    assert list(tmp_path.iterdir()) == []
    tree = ast.parse(ENTRYPOINT.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert imported == {"annotations", "sys"}


def test_accepted_checkpoint_b_capsule_sources_remain_exact() -> None:
    expected = {
        "runs/goalzendo/build_g01_b_source_capsule.py": "c1255da5ed0bebe5c91bbcc724213e671e9b22b2b79dc52915426e51ea6aa593",
        "runs/goalzendo/g01_b_source_capsule_stage.py": "253685cefcf31a31324a8aa14ec854b9fef09babf38d24a5e930c68da80afa55",
        "tests/goalzendo/test_g01_b_source_capsule.py": "fe0919f58adc131c774ab5f7387a7ac0d3b2c2a4d83c7dd4503b9ba77fe6abd3",
        "docs/goalzendo/protocols/g01-b-source-capsule.md": "cf9d73edc279b6dacd1e64823ffff016a8d409b2cdb9cd434e5e244d17687251",
    }
    for relative, expected_sha in expected.items():
        assert hashlib.sha256((REPO / relative).read_bytes()).hexdigest() == expected_sha
