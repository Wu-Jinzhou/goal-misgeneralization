from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "runs" / "goalzendo" / "runpod_runtime.py"
ENTRYPOINT = ROOT / "runs" / "goalzendo" / "runpod_entrypoint.sh"
DISPATCHER = ROOT / "runs" / "goalzendo" / "runpod_dispatch.sh"
BUNDLE_BUILDER = ROOT / "runs" / "goalzendo" / "build_runpod_bundle.sh"


def _helper(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(HELPER), *arguments],
        check=check,
        capture_output=True,
        text=True,
    )


def test_local_gpu_plan_uses_stable_inherited_device_order() -> None:
    result = _helper(
        "plan",
        "--requested-shards",
        "2",
        "--detected-gpus",
        "4",
        "--visible-devices",
        "GPU-third,GPU-first,GPU-spare",
    )
    payload = json.loads(result.stdout)
    assert payload["schema"] == "goalzendo.runpod_local_gpu_plan"
    assert payload["assignments"] == [
        {"shard_index": 0, "num_shards": 2, "cuda_visible_devices": "GPU-third"},
        {"shard_index": 1, "num_shards": 2, "cuda_visible_devices": "GPU-first"},
    ]


def test_local_gpu_plan_uses_cuda_ordinals_when_visibility_is_unset() -> None:
    result = _helper(
        "plan",
        "--requested-shards",
        "3",
        "--detected-gpus",
        "3",
        "--format",
        "tsv",
    )
    assert result.stdout.splitlines() == ["0\t3\t0", "1\t3\t1", "2\t3\t2"]


def test_local_gpu_plan_fails_closed_on_oversubscription_or_unsafe_visibility() -> None:
    too_many = _helper(
        "plan",
        "--requested-shards",
        "3",
        "--detected-gpus",
        "4",
        "--visible-devices",
        "0,1",
        check=False,
    )
    assert too_many.returncode == 2
    assert "only 2 CUDA-visible GPUs" in too_many.stderr

    duplicate = _helper(
        "plan",
        "--requested-shards",
        "1",
        "--detected-gpus",
        "2",
        "--visible-devices",
        "0,0",
        check=False,
    )
    assert duplicate.returncode == 2
    assert "duplicate" in duplicate.stderr

    unsafe = _helper(
        "plan",
        "--requested-shards",
        "1",
        "--detected-gpus",
        "2",
        "--visible-devices",
        "0,$bad",
        check=False,
    )
    assert unsafe.returncode == 2
    assert "unsafe" in unsafe.stderr


def test_runtime_json_writer_is_typed_and_atomic(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "status.json"
    _helper(
        "write-json",
        "--output",
        str(destination),
        "--string",
        "state=complete",
        "--integer",
        "exit_code=0",
        "--boolean",
        "analyzed=true",
    )
    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "analyzed": True,
        "exit_code": 0,
        "state": "complete",
    }
    assert not list(destination.parent.glob("*.tmp"))


def test_runpod_shells_are_syntactically_valid_and_executable() -> None:
    for script in (ENTRYPOINT, DISPATCHER, BUNDLE_BUILDER):
        subprocess.run(["bash", "-n", str(script)], check=True)
        assert os.access(script, os.X_OK)
    assert os.access(HELPER, os.X_OK)


def test_runpod_bundle_contains_every_file_required_by_bundled_tests(tmp_path: Path) -> None:
    bundle = tmp_path / "goalzendo.tar.gz"
    subprocess.run([str(BUNDLE_BUILDER), str(bundle)], check=True, cwd=ROOT)
    listing = subprocess.run(
        ["tar", "-tzf", str(bundle)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert "docs/goalzendo/plans/g00a4-reward-gradient-shape.jsonl" in listing
    assert "docs/goalzendo/plans/g00c-strict-engineering-0p5b.jsonl" in listing
    assert "docs/goalzendo/plans/g00d-fixed-window-engineering-0p5b.jsonl" in listing
    assert "docs/goalzendo/plans/g00d-fixed-window-optimizer-1p5b.jsonl" in listing
    assert "src/goalzendo_g00e/bridge.py" in listing
    assert "src/goalzendo_g00e/bridge_manifest.json" in listing


def test_g00_dispatch_dry_run_is_fixed_order_and_has_distinct_roots(tmp_path: Path) -> None:
    environment = {
        **os.environ,
        "GOALZENDO_WORK_ROOT": str(tmp_path / "network-volume"),
        "GOALZENDO_LOCAL_GPU_SHARDS": "4",
        "GOALZENDO_RUN_QWEN_INTEGRATION": "1",
    }
    result = subprocess.run(
        ["bash", str(DISPATCHER), "--dry-run", "g00_suite"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    stage_lines = [line for line in result.stdout.splitlines() if line.startswith("stage\t")]
    assert [line.split("\t")[2] for line in stage_lines] == [
        "g00-engineering",
        "g00-capability-0p5b",
        "g00-pilot-0p5b",
        "g00-capability-1p5b",
        "g00-pilot-1p5b",
    ]
    assert len({line.split("\t")[4] for line in stage_lines}) == 5
    assert result.stdout.count("GOALZENDO_LOCAL_GPU_SHARDS=4") == 5
    assert result.stdout.count("GOALZENDO_RUN_QWEN_INTEGRATION=1") == 1
    assert result.stdout.count("GOALZENDO_RUN_QWEN_INTEGRATION=0") == 4
    assert result.stdout.count("GOALZENDO_RUN_COMBINED_ANALYSIS=1") == 3
    assert result.stdout.count("GOALZENDO_RUN_COMBINED_ANALYSIS=0") == 2
    assert not (tmp_path / "network-volume").exists()


def test_g00d_dispatch_dry_run_is_fixed_order_and_has_distinct_roots(tmp_path: Path) -> None:
    environment = {
        **os.environ,
        "GOALZENDO_WORK_ROOT": str(tmp_path / "network-volume"),
        "GOALZENDO_LOCAL_GPU_SHARDS": "4",
        "GOALZENDO_RUN_QWEN_INTEGRATION": "1",
    }
    result = subprocess.run(
        ["bash", str(DISPATCHER), "--dry-run", "g00d_suite"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    stage_lines = [line for line in result.stdout.splitlines() if line.startswith("stage\t")]
    assert [line.split("\t")[2] for line in stage_lines] == [
        "g00d-engineering-0p5b",
        "g00d-capability-0p5b",
        "g00d-capability-1p5b",
        "g00d-optimizer-1p5b",
    ]
    assert len({line.split("\t")[4] for line in stage_lines}) == 4
    assert result.stdout.count("GOALZENDO_LOCAL_GPU_SHARDS=4") == 4
    assert result.stdout.count("GOALZENDO_RUN_QWEN_INTEGRATION=1") == 1
    assert result.stdout.count("GOALZENDO_RUN_QWEN_INTEGRATION=0") == 3
    assert result.stdout.count("GOALZENDO_RUN_COMBINED_ANALYSIS=1") == 2
    assert result.stdout.count("GOALZENDO_RUN_COMBINED_ANALYSIS=0") == 2
    assert not (tmp_path / "network-volume").exists()


def test_gate_and_g01_dispatch_plans_bind_all_roots_and_gate_artifact(tmp_path: Path) -> None:
    work_root = tmp_path / "network-volume"
    gate_environment = {
        **os.environ,
        "GOALZENDO_WORK_ROOT": str(work_root),
        "GOALZENDO_SELECTED_SFT_LR": "0.0001",
        "GOALZENDO_SELECTED_RL_LR": "0.00003",
    }
    gate = subprocess.run(
        ["bash", str(DISPATCHER), "gate", "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
        env=gate_environment,
    )
    assert gate.stdout.count("--g00-artifacts") == 4
    assert gate.stdout.count("--g00-config") == 4
    assert "g00d_fixed_window_engineering_0p5b.yaml" in gate.stdout
    assert "g00d_fixed_window_optimizer_1p5b.yaml" in gate.stdout
    assert "g01_known_law.yaml" in gate.stdout
    assert "goalzendo.cli" in gate.stdout
    assert "goalzendo_g00e.cli" not in gate.stdout
    assert not work_root.exists()

    bridged_gate = subprocess.run(
        ["bash", str(DISPATCHER), "g00e_gate", "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
        env=gate_environment,
    )
    assert bridged_gate.stdout.count("--g00-artifacts") == 4
    assert "goalzendo_g00e.cli" in bridged_gate.stdout
    assert "--g00e-sidecar-output" in bridged_gate.stdout
    assert "--g00e-manifest-sha256" in bridged_gate.stdout

    gate_path = work_root / "status-goalzendo" / "g00e-gate-v3.json"
    sidecar_path = work_root / "status-goalzendo" / "g00e-numerical-bridge-v3.json"
    sidecar_sha256 = "a" * 64
    g01_environment = {
        **os.environ,
        "GOALZENDO_WORK_ROOT": str(work_root),
        "GOALZENDO_GATE_ARTIFACT": str(gate_path),
        "GOALZENDO_G00E_V3_SIDECAR": str(sidecar_path),
        "GOALZENDO_G00E_V3_SIDECAR_SHA256": sidecar_sha256,
        "GOALZENDO_LOCAL_GPU_SHARDS": "2",
    }
    g01 = subprocess.run(
        ["bash", str(DISPATCHER), "--dry-run", "g01"],
        check=True,
        capture_output=True,
        text=True,
        env=g01_environment,
    )
    assert f"GOALZENDO_GATE_ARTIFACT={gate_path}" in g01.stdout
    assert f"GOALZENDO_G00E_V3_SIDECAR={sidecar_path}" in g01.stdout
    assert f"GOALZENDO_G00E_V3_SIDECAR_SHA256={sidecar_sha256}" in g01.stdout
    assert "GOALZENDO_CLI_MODULE=goalzendo_g00e.cli" in g01.stdout
    assert "GOALZENDO_LOCAL_GPU_SHARDS=2" in g01.stdout


def test_g01_dispatch_rejects_missing_external_v3_sidecar_identity(tmp_path: Path) -> None:
    environment = {
        **os.environ,
        "GOALZENDO_WORK_ROOT": str(tmp_path / "network-volume"),
        "GOALZENDO_GATE_ARTIFACT": str(tmp_path / "gate.json"),
    }
    result = subprocess.run(
        ["bash", str(DISPATCHER), "g01", "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 64
    assert "requires explicit" in result.stderr


def test_bridge_entrypoint_rejects_missing_sidecar_before_gpu_or_model_load(tmp_path: Path) -> None:
    environment = {
        **os.environ,
        "GOALZENDO_WORK_ROOT": str(tmp_path / "network-volume"),
        "GOALZENDO_CLI_MODULE": "goalzendo_g00e.cli",
        "GOALZENDO_GATE_ARTIFACT": str(tmp_path / "gate.json"),
    }
    result = subprocess.run(
        ["bash", str(ENTRYPOINT)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 66
    assert "require explicit gate, v3 sidecar" in result.stderr
    assert "nvidia-smi" not in result.stderr


def test_entrypoint_keeps_failures_external_and_base64_opt_in() -> None:
    source = ENTRYPOINT.read_text(encoding="utf-8")
    assert 'RUN_GATE_ARGUMENTS=(--gate-artifact "$GATE_ARTIFACT")' in source
    assert 'GOALZENDO_EMIT_ANALYSIS_B64:-0' in source
    assert 'GOALZENDO_RUN_COMBINED_ANALYSIS:-1' in source
    assert 'CLI_MODULE="${GOALZENDO_CLI_MODULE:-goalzendo.cli}"' in source
    assert 'GOALZENDO_G00E_V3_SIDECAR_SHA256:-' in source
    assert 'CURRENT_PHASE="g00e_worker_preflight"' in source
    assert "sleep infinity" not in source
    assert "while true" not in source
    assert "one or more local GPU shards failed; combined analysis was not started" in source
