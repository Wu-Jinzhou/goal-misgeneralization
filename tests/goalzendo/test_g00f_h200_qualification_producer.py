from __future__ import annotations

import hashlib
import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from torch import nn

import goalzendo_g00f_h200.qualification_producer as producer
from goalzendo.training import build_optimizer
from goalzendo_g00f_h200.qualification import (
    LAW_FAMILIES,
    PANELS,
    PROFILE_CONTRACT,
    TUNED_CAPACITY_DISQUALIFIERS,
    semantic_digest,
)

REPO = Path(__file__).resolve().parents[2]
GPU_UUIDS = tuple(f"GPU-10000000-0000-0000-0000-{index:012d}" for index in range(4))


def _launch_request(*, index: int, controller: Path, repo: Path) -> dict[str, Any]:
    return {
        "request_digest": semantic_digest({"worker": index}),
        "request_kind": "qualification",
        "worker_index": index,
        "probe_panel_id": None,
        "probe_law_family": None,
        "device": {
            "host_ordinal": 9 - index,
            "uuid": GPU_UUIDS[index],
            "name": "NVIDIA H200 NVL",
        },
        "controller_binding": {"path": str(controller)},
        "controller_options": {"repo": str(repo)},
    }


def test_deadline_binding_freezes_105_minutes_and_full_post_reserve() -> None:
    started_wall = 1_800_000_000.0
    terminate = producer._canonical_utc_from_timestamp(
        started_wall + producer.MINIMUM_PROVISION_REMAINING_AT_START_SECONDS
    )
    binding = producer._deadline_binding(
        started_wall_seconds=started_wall,
        started_monotonic_seconds=100.0,
        completed_wall_seconds=started_wall + 6_299.0,
        completed_monotonic_seconds=6_399.0,
        provision_terminate_after_utc=terminate,
    )
    assert binding["producer_ceiling_seconds"] == 6_300
    assert binding["post_qualification_reserve_seconds"] == 600
    assert binding["ledger_ceiling_seconds"] == 50_400
    assert binding["grace_seconds"] == 60
    assert binding["minimum_provision_remaining_at_start_seconds"] == 57_360
    assert producer.PROVISION_START_GUARD_MESSAGE == (
        "provision leaves less than 57,360 seconds at producer start"
    )
    with pytest.raises(producer.ProducerError, match="105-minute"):
        producer._deadline_binding(
            started_wall_seconds=started_wall,
            started_monotonic_seconds=100.0,
            completed_wall_seconds=started_wall + 6_301.0,
            completed_monotonic_seconds=6_401.0,
            provision_terminate_after_utc=terminate,
        )


def test_isolated_worker_authentication_does_not_equate_host_and_logical_ordinals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = producer.GPUDevice(ordinal=7, uuid=GPU_UUIDS[0], name="NVIDIA H200 NVL")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", device.uuid)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda logical_ordinal: "NVIDIA H200 NVL")

    def run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert command[1] == f"--id={device.uuid}"
        return subprocess.CompletedProcess(command, 0, f"{device.uuid}, {device.name}\n", "")

    monkeypatch.setattr(subprocess, "run", run)
    producer._authenticate_isolated_worker_device(device)


def test_worker_supervisor_uses_files_and_cannot_deadlock_on_noisy_children(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = tmp_path / "noisy_controller.py"
    controller.write_text(
        "import sys\nsys.stdout.write('x' * 2_000_000)\nsys.stdout.flush()\n",
        encoding="utf-8",
    )
    requests = [tmp_path / f"request-{index}.json" for index in range(4)]
    by_path = {
        path.resolve(): _launch_request(index=index, controller=controller, repo=tmp_path)
        for index, path in enumerate(requests)
    }
    monkeypatch.setattr(producer, "_strict_json", lambda path, _label: by_path[Path(path).resolve()])
    monkeypatch.setattr(producer, "_validate_worker_request", lambda value: value)

    receipts = producer._launch_workers(
        requests,
        deadline_monotonic=time.monotonic() + 20.0,
    )
    assert len(receipts) == 4
    assert [receipt["worker_index"] for receipt in receipts] == list(range(4))
    assert all(receipt["exit_code"] == 0 for receipt in receipts)
    assert all(receipt["log_binding"]["bytes"] == 2_000_000 for receipt in receipts)
    assert all(receipt["log_binding"]["mode"] == 0o400 for receipt in receipts)


def test_worker_supervisor_reaps_started_children_if_later_launch_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = tmp_path / "controller.py"
    controller.write_text("pass\n", encoding="utf-8")
    requests = [tmp_path / f"request-{index}.json" for index in range(2)]
    by_path = {
        path.resolve(): _launch_request(index=index, controller=controller, repo=tmp_path)
        for index, path in enumerate(requests)
    }
    monkeypatch.setattr(producer, "_strict_json", lambda path, _label: by_path[Path(path).resolve()])
    monkeypatch.setattr(producer, "_validate_worker_request", lambda value: value)

    class FakeProcess:
        returncode: int | None = None
        terminated = False

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            self.terminated = True
            self.returncode = -15

        def kill(self) -> None:
            self.returncode = -9

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            assert self.returncode is not None
            return self.returncode

    child = FakeProcess()
    launch_count = 0

    def popen(*_args: Any, **_kwargs: Any) -> FakeProcess:
        nonlocal launch_count
        launch_count += 1
        if launch_count == 2:
            raise OSError("synthetic launch failure")
        return child

    monkeypatch.setattr(subprocess, "Popen", popen)
    with pytest.raises(producer.ProducerError, match="children were reaped"):
        producer._launch_workers(
            requests,
            deadline_monotonic=time.monotonic() + 20.0,
        )
    assert child.terminated is True
    assert (tmp_path / "request-0-launch-failure-receipt.json").is_file()
    assert os.stat(tmp_path / "request-0-worker.log").st_mode & 0o777 == 0o400


def test_clean_update_timer_delegates_exact_resolved_contract_to_train_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = object.__new__(producer.ActualModelWorkerEngine)
    engine.repo = REPO
    engine.device = torch.device("cpu")
    engine.tokenizers = {"g00f-0p5b": object()}
    calls: list[dict[str, Any]] = []

    def load_fresh(_panel: str, _profile: str) -> tuple[nn.Module, object, float]:
        return nn.Sequential(nn.Linear(2, 2, bias=False)), object(), 1.0

    class FakeOptimizer:
        def __init__(self, model: nn.Module) -> None:
            self.param_groups = [{"params": list(model.parameters())}]

    def fake_train_steps(
        scorer: nn.Module,
        optimizer: FakeOptimizer,
        prompts: tuple[str, ...],
        actions: tuple[int, ...],
        **kwargs: Any,
    ) -> SimpleNamespace:
        del scorer, optimizer, prompts, actions
        calls.append(kwargs)
        steps = int(kwargs["total_steps"])
        accumulation = int(kwargs["gradient_accumulation_steps"])
        return SimpleNamespace(
            state=SimpleNamespace(global_step=steps, micro_step=steps * accumulation),
            metrics=[object()] * steps,
        )

    class FakeEvent:
        def __init__(self, *, enable_timing: bool) -> None:
            assert enable_timing is True

        def record(self) -> None:
            return None

        def elapsed_time(self, _other: object) -> float:
            return 10.0

    monkeypatch.setattr(engine, "_load_fresh_model", load_fresh)
    monkeypatch.setattr(producer, "TwoActionScorer", lambda model, *_args, **_kwargs: model)
    monkeypatch.setattr(producer, "build_optimizer", lambda model, **_kwargs: FakeOptimizer(model))
    monkeypatch.setattr(producer, "train_steps", fake_train_steps)
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    def fake_encode_action_continuations(
        _tokenizer: object,
        prompts: tuple[str, ...],
        _labels: tuple[str, str],
        *,
        add_prompt_special_tokens: bool,
    ) -> SimpleNamespace:
        assert add_prompt_special_tokens is False
        return SimpleNamespace(
            prompt_tokens=tuple(tuple(range(len(prompt.encode("utf-8")))) for prompt in prompts),
            action_tokens=tuple(((32,), (33,)) for _prompt in prompts),
        )

    monkeypatch.setattr(producer, "encode_action_continuations", fake_encode_action_continuations)
    distinct_prompts = tuple(f"host-envelope-{index}" for index in range(10))
    host_rows = []
    for prompt in distinct_prompts:
        prompt_size = len(prompt.encode("utf-8"))
        host_rows.append(
            {
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "prompt_utf8_bytes": prompt_size,
                "prompt_token_length": prompt_size,
                "prompt_a_utf8_bytes": prompt_size + 1,
                "prompt_a_token_length": prompt_size + 1,
                "prompt_b_utf8_bytes": prompt_size + 1,
                "prompt_b_token_length": prompt_size + 1,
                "continuation_a_token_count": 1,
                "continuation_a_token_ids_sha256": semantic_digest([32]),
                "continuation_b_token_count": 1,
                "continuation_b_token_ids_sha256": semantic_digest([33]),
                "total_contextual_token_work": prompt_size * 3 + 2,
                "total_utf8_bytes": prompt_size * 3 + 2,
            }
        )

    def host_proof(metric: str, *, role: str) -> dict[str, Any]:
        source_rows = host_rows if role == "variability_stress" else host_rows[:1]
        repetition_count = 5 if role == "variability_stress" else 50
        ordered = [str(row["prompt_sha256"]) for row in source_rows] * repetition_count
        return {
            "role": role,
            "metric_field": metric,
            "source_row_count": len(source_rows),
            "rows": source_rows,
            "rows_sha256": semantic_digest(source_rows),
            "selected_metric_sum": sum(cast(int, row[metric]) for row in source_rows),
            "selected_metric_maximum": max(cast(int, row[metric]) for row in source_rows),
            "repetition_count": repetition_count,
            "execution_row_count_per_update": 50,
            "ordered_execution_prompt_sha256": semantic_digest(ordered),
            "baseline_call_partition_sizes": [10] * 5,
            "baseline_call_partitions_sha256": semantic_digest(
                [ordered[index : index + 10] for index in range(0, 50, 10)]
            ),
        }

    engine.training_maximums = {
        "g00f-0p5b": producer.GlobalTrainingMaximum(
            prompt="authenticated-global-longest-prompt",
            action=0,
            candidate_choices=(0, 0, 1),
            host_envelope_prompts={
                "contextual_token_work_variability": distinct_prompts,
                "utf8_byte_work_variability": distinct_prompts,
                "contextual_token_work_dominance": distinct_prompts[:1],
                "utf8_byte_work_dominance": distinct_prompts[:1],
            },
            proof={
                "maximum_prompt_token_length": 321,
                "maximum_token_length": 322,
                "tokenizer_host_envelopes": {
                    "contextual_continuation_mode": (
                        "goalzendo.modeling.encode_action_continuations:add_prompt_special_tokens=false"
                    ),
                    "envelope_order": list(producer.HOST_ENVELOPE_ORDER),
                    "measured_seconds_aggregation": "sum_all_four_envelopes",
                    "envelopes": {
                        "contextual_token_work_variability": host_proof(
                            "total_contextual_token_work",
                            role="variability_stress",
                        ),
                        "utf8_byte_work_variability": host_proof(
                            "total_utf8_bytes",
                            role="variability_stress",
                        ),
                        "contextual_token_work_dominance": host_proof(
                            "total_contextual_token_work",
                            role="hard_dominance",
                        ),
                        "utf8_byte_work_dominance": host_proof(
                            "total_utf8_bytes",
                            role="hard_dominance",
                        ),
                    },
                },
            },
        )
    }

    wall, cuda, host_seconds, contract = engine._clean_timing_only_updates(
        profile="baseline",
        panel_id="g00f-0p5b",
    )
    assert wall > 0.0 and cuda == pytest.approx(0.01) and host_seconds > 0.0
    assert [call["total_steps"] for call in calls] == [1, 8]
    assert calls[-1]["algorithm"] == "sft"
    assert calls[-1]["batch_size"] == PROFILE_CONTRACT["baseline"]["train_batch_size"]
    assert calls[-1]["gradient_accumulation_steps"] == 5
    assert calls[-1]["warmup_steps"] == 12
    assert calls[-1]["max_grad_norm"] == 1.0
    assert calls[-1]["parameter_finite_check_interval"] == 0
    assert calls[-1]["hooks"] is None
    assert contract["shared_helper"] == "goalzendo.training.train_steps"
    assert contract["candidate_choice_geometry_included"] is True
    standardized = contract["standardized_worst_shape_training"]
    assert len(standardized["timed_call_shapes"]) == 40
    assert {row["max_seq_len"] for row in standardized["timed_call_shapes"]} == {321}
    assert standardized["conservative_scaling_ratio"] == 1.0
    host_contract = standardized["tokenizer_host_envelope_contracts"]
    assert host_contract["measured_seconds_sum"] == pytest.approx(
        sum(host_contract["measured_seconds_by_envelope"].values())
    )
    assert all(
        envelope["profile_call_partition_sizes"] == [10] * 5
        for envelope in host_contract["envelopes"].values()
    )


def test_callback_and_final_seal_use_production_sized_runstore_envelopes(
    tmp_path: Path,
) -> None:
    engine = object.__new__(producer.ActualModelWorkerEngine)
    engine.inventory = producer.TransientInventory(tmp_path / "transient")
    engine.repo = REPO
    capture_root = engine.inventory.root / "capture"
    scores = [[-0.5108256237659907, -0.916290731874155] for _ in range(768)]
    callback, callback_seconds = engine._evaluation_callback_io(
        capture_root=capture_root,
        ordinal=0,
        kind="diagnostic",
        scored={
            "prompt_count": len(scores),
            "batch_size": 128,
            "normalized_outputs": scores,
        },
    )
    assert callback_seconds > 0.0
    assert callback == {
        **callback,
        "artifact_count": 3,
        "metrics_row_count": 768,
        "prediction_row_count": 768,
        "progress_record_count": 1,
        "schema": "production_metrics_predictions_progress_envelope_v1",
    }
    boundaries = [
        {
            "prompt_count": 768,
            "normalized_outputs_sha256": semantic_digest({"boundary": index}),
        }
        for index in range(13)
    ]
    seal, seal_seconds = engine._final_seal_io(
        capture_root=capture_root,
        benchmark_identity=semantic_digest("benchmark"),
        profile="baseline",
        panel_id="g00f-0p5b",
        steps=[{"update": update} for update in range(1, 9)],
        boundaries=boundaries,
    )
    assert seal_seconds > 0.0
    assert seal["artifact_count"] >= 15
    assert seal["file_names"] == [
        "metrics.jsonl",
        "predictions.jsonl",
        "summary.json",
        "status.json",
        "completion.json",
        "COMPLETE",
    ]
    assert seal["metrics_row_count"] == seal["prediction_row_count"] == 13 * 768
    assert seal["attested_file_count"] == 11
    assert seal["completion_attestation_verified"] is True
    assert seal["complete_marker_final_write"] is True
    assert seal["run_store_finalize_used"] is True
    assert seal["outcome_file_seal_file_count"] == 3
    assert set(seal["outcome_file_seal_hashes"]) == {
        "metrics.jsonl",
        "predictions.jsonl",
        "summary.json",
    }
    assert set(seal["outcome_file_seal_modes"].values()) == {0}
    assert seal["outcome_file_seal_scan_and_chmod_completed"] is True
    assert seal["seal_restore_only_for_authenticated_transient_cleanup"] is True
    assert not any((tmp_path / "transient").rglob("*.json*"))


def test_native_optimizer_order_is_distinct_from_sorted_evidence_order() -> None:
    class OrderedLayers(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer2 = nn.Linear(1, 1, bias=False)
            self.layer10 = nn.Linear(1, 1, bias=False)

    model = OrderedLayers()
    native = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    evidence = [name for name, _parameter in producer._named_trainable_parameters(model)]
    optimizer = build_optimizer(model, learning_rate=1e-5, weight_decay=0.0)
    optimizer_ids = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    assert native == ["layer2.weight", "layer10.weight"]
    assert evidence == ["layer10.weight", "layer2.weight"]
    assert optimizer_ids == [id(parameter) for parameter in model.parameters()]


def test_tuned_capacity_probe_contract_is_exactly_sixteen_cells() -> None:
    assert len(PANELS) * len(LAW_FAMILIES) * 4 == 16
    assert TUNED_CAPACITY_DISQUALIFIERS == (
        "cuda_out_of_memory",
        "nonfinite_capacity_execution",
        "reserved_memory_ceiling_exceeded",
    )
