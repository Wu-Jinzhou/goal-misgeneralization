from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest

import goalzendo_g01_coordinator.coordinator as coordinator
from goalzendo_g01_coordinator import CoordinatorError, build_balanced_schedule

REPO = Path(__file__).resolve().parents[2]
ENTRYPOINT = REPO / "runs/goalzendo/run_g01_global_coordinator.py"
UUID_A = "11111111-1111-4111-8111-111111111111"
SHA = "a" * 64


@dataclass(frozen=True)
class _Spec:
    plan_key: str
    seed: int
    config: dict[str, Any]


def _synthetic_plan() -> tuple[list[_Spec], dict[str, tuple[str, str]]]:
    plan: list[_Spec] = []
    paths: dict[str, tuple[str, str]] = {}
    index = 0
    for family in ("majority", "parity"):
        for q_p in (0.1, 0.2, 0.3):
            for seed in range(10):
                for algorithm in ("sft", "outcome_rl"):
                    key = f"{index:020x}"
                    spec = _Spec(
                        key,
                        seed,
                        {"data": {"rule_family": family, "q_p": q_p}, "train": {"algorithm": algorithm}},
                    )
                    plan.append(spec)
                    paths[key] = (f"run-{key}", f"/artifact/{family}/{q_p}/{seed}/{algorithm}")
                    index += 1
    return plan, paths


def _entrypoint_module() -> Any:
    spec = importlib.util.spec_from_file_location("_g01_checkpoint_b_bootstrap_test", ENTRYPOINT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_balanced_schedule_is_exact_four_by_thirty_and_keeps_original_specs() -> None:
    plan, paths = _synthetic_plan()
    schedule = build_balanced_schedule(
        plan,
        output_paths=paths,
        plan_rows_digest=SHA,
        plan_key_set_digest="b" * 64,
    )
    assert len(schedule.rows) == 120
    assert all(len(schedule.for_worker(index)) == 30 for index in range(4))
    assert all(len({row.pair_index for row in schedule.for_worker(index)}) == 15 for index in range(4))
    assert all(row.worker_index == row.pair_index % 4 for row in schedule.rows)
    assert {id(row.spec) for row in schedule.rows} == {id(spec) for spec in plan}
    for pair_index in range(60):
        pair = [row for row in schedule.rows if row.pair_index == pair_index]
        assert [row.algorithm for row in pair] == ["sft", "outcome_rl"]
        assert pair[0].pair_key == pair[1].pair_key


def test_schedule_rejects_a_mismatched_pair_and_extra_output() -> None:
    plan, paths = _synthetic_plan()
    broken = list(plan)
    original = broken[-1]
    broken[-1] = _Spec(
        original.plan_key,
        99,
        original.config,
    )
    with pytest.raises(CoordinatorError, match="60 paired"):
        build_balanced_schedule(
            broken,
            output_paths=paths,
            plan_rows_digest=SHA,
            plan_key_set_digest="b" * 64,
        )
    with pytest.raises(CoordinatorError, match="exact G01 membership"):
        build_balanced_schedule(
            plan,
            output_paths={**paths, "f" * 20: ("extra", "/extra")},
            plan_rows_digest=SHA,
            plan_key_set_digest="b" * 64,
        )


def test_structural_schedule_rejects_nonfinite_or_coerced_q_p() -> None:
    for bad_q_p in (float("nan"), float("inf"), "0.1", True):
        plan, paths = _synthetic_plan()
        original = plan[0]
        broken_config = {
            "data": {"rule_family": "majority", "q_p": bad_q_p},
            "train": {"algorithm": "sft"},
        }
        plan[0] = _Spec(original.plan_key, original.seed, broken_config)
        with pytest.raises(CoordinatorError, match="q_p"):
            build_balanced_schedule(
                plan,
                output_paths=paths,
                plan_rows_digest=SHA,
                plan_key_set_digest="b" * 64,
            )


def test_every_execution_entry_is_literal_source_refusal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Replacing an adjacent guard name cannot remove the literal first-statement
    # refusal from any prospective lifecycle entry.
    monkeypatch.setattr(coordinator, "_require_isolated_runtime", lambda: None)
    with pytest.raises(CoordinatorError, match=coordinator.RUNTIME_OVERLAY_REFUSAL):
        coordinator._coordinate(
            repo=REPO,
            execution_uuid=UUID_A,
            expected_token_sha256=SHA,
            expected_bridge_source_digest=SHA,
            source={},
            authority=object(),
        )
    with pytest.raises(CoordinatorError, match=coordinator.RUNTIME_OVERLAY_REFUSAL):
        coordinator._worker_main(
            0,
            (),
            repo=REPO,
            layout={},
            gate_read_fd=-1,
            ready_write_fd=-1,
            deadline_ns=0,
            reservations={},
            authority=object(),
        )
    with pytest.raises(CoordinatorError, match=coordinator.RUNTIME_OVERLAY_REFUSAL):
        coordinator._acquire_global_claim(
            layout={},
            execution_uuid=UUID_A,
            token=object(),
            source={},
            schedule=cast(Any, object()),
            expected_token_sha256=SHA,
        )
    row = cast(Any, object())
    with pytest.raises(CoordinatorError, match=coordinator.RUNTIME_OVERLAY_REFUSAL):
        coordinator._execute_row(
            row,
            repo=REPO,
            layout={},
            backend=object(),
            reservation=cast(Any, object()),
            authority=object(),
        )
    assert (
        coordinator.main_from_entrypoint(
            tmp_path / "forged.py", ["--repo", str(tmp_path)], _bootstrap_snapshot=object()
        )
        == 2
    )
    assert coordinator.RUNTIME_OVERLAY_REFUSAL in capsys.readouterr().err
    assert not tmp_path.exists() or not any(tmp_path.iterdir())


def test_bootstrap_has_exact_arguments_and_no_a_route_runtime_mapping() -> None:
    bootstrap = _entrypoint_module()
    pairs = {
        "--repo": "/x",
        "--execution-uuid": UUID_A,
        "--expected-coordinator-token-sha256": "a" * 64,
        "--expected-bridge-source-digest": "b" * 64,
        "--expected-coordinator-source-digest": "c" * 64,
    }
    argv = [value for key, value in pairs.items() for value in (key, value)]
    assert bootstrap._exact_arguments(argv) == pairs
    with pytest.raises(bootstrap._BootstrapError, match="exactly five"):
        bootstrap._exact_arguments([*argv, "--repo", "/decoy"])
    source = ENTRYPOINT.read_text(encoding="utf-8")
    assert "expected_python" not in source
    assert "site-packages" not in source
    assert "_H100_PYTHON" not in source
    tree = ast.parse(source)
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imported.update(node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom))
    assert not any(name.startswith("goalzendo") for name in imported)
    assert source.index("snapshot.verify_held()") < source.index("raise _BootstrapError(_RUNTIME_REFUSAL)")


def test_exact_cli_requires_isolated_no_site_before_any_capture() -> None:
    result = subprocess.run(
        [sys.executable, str(ENTRYPOINT)],
        cwd=REPO,
        env={**os.environ, "PYTHONPATH": str(REPO / "src")},
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "requires CPython isolated/no-site mode (-I -S)" in result.stderr


def test_bootstrap_refusal_precedes_project_or_third_party_import_syntax() -> None:
    tree = ast.parse(ENTRYPOINT.read_text(encoding="utf-8"))
    imports = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imports.update(node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom))
    assert not any(
        name == prefix or name.startswith(prefix + ".")
        for name in imports
        for prefix in ("goalzendo", "goalzendo_g00f_g01_bridge", "goalzendo_g01_coordinator", "yaml", "torch")
    )


def test_snapshot_final_replay_rejects_token_mode_or_link_change(tmp_path: Path) -> None:
    bootstrap = _entrypoint_module()
    repo = tmp_path / UUID_A / "frozen-source"
    repo.mkdir(parents=True)
    (repo / "member.py").write_bytes(b"x = 1\n")
    token = tmp_path / "g00f-g01-coordinator-input.json"
    token.write_bytes(b"{}\n")
    token.chmod(0o400)
    snapshot = bootstrap._capture(repo, ["member.py"], token)
    try:
        snapshot.verify_held()
        token.chmod(0o600)
        with pytest.raises(bootstrap._BootstrapError, match="thin-token path changed"):
            snapshot.verify_held()
        token.chmod(0o400)
        os.link(token, tmp_path / "token-hardlink.json")
        with pytest.raises(bootstrap._BootstrapError, match="thin-token path changed"):
            snapshot.verify_held()
    finally:
        snapshot.close()


def test_fd_relative_reader_rejects_symlink_and_ignores_procfd_path_traversal(tmp_path: Path) -> None:
    run = tmp_path / "run"
    manifests = run / "manifests"
    manifests.mkdir(parents=True)
    payload = b"identity\n"
    (manifests / "model.json").write_bytes(payload)
    run_fd = os.open(run, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        assert coordinator._require_nested_regular_at(run_fd, "manifests/model.json", "model") == payload
        (manifests / "model.json").unlink()
        (manifests / "model.json").symlink_to(tmp_path / "outside")
        with pytest.raises(CoordinatorError, match=r"without link traversal|fd-relative"):
            coordinator._require_nested_regular_at(run_fd, "manifests/model.json", "model")
    finally:
        os.close(run_fd)


def test_accounting_gate_requires_exact_complete_only_schema(tmp_path: Path) -> None:
    body = {
        "schema": "goalzendo.g01_checkpoint_b_final_accounting",
        "schema_version": 1,
        "study_id": "g01",
        "execution_uuid": UUID_A,
        "global_claim_sha256": "1" * 64,
        "global_claim_digest": "2" * 64,
        "schedule_digest": "3" * 64,
        "plan_rows_digest": coordinator.G01_PLAN_ROWS_DIGEST,
        "plan_key_set_digest": coordinator.G01_PLAN_KEY_SET_DIGEST,
        "planned_rows": 120,
        "terminal_rows": 120,
        "counts": {"complete": 120},
        "row_receipts_digest": "4" * 64,
        "worker_count": 4,
        "parent_accounting_outcome_independent": True,
        "parent_accounting_metrics_file_read": False,
        "parent_accounting_predictions_file_read": False,
        "parent_accounting_summary_file_read": False,
        "retry": False,
    }
    gate = {**body, "receipt_digest": coordinator._digest(body)}
    path = tmp_path / "gate.json"
    path.write_bytes(coordinator._pretty(gate))
    path.chmod(0o400)
    gate_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    verified = coordinator.verify_accounting_gate(
        path,
        expected_gate_sha256=gate_sha,
        expected_claim_sha256="1" * 64,
        expected_execution_uuid=UUID_A,
        expected_global_claim_digest="2" * 64,
        expected_schedule_digest="3" * 64,
        expected_row_receipts_digest="4" * 64,
    )
    assert verified["counts"] == {"complete": 120}

    fabricated = {**gate, "counts": {"complete": 119, "terminal_no_retry": 1}}
    fabricated_body = {key: value for key, value in fabricated.items() if key != "receipt_digest"}
    fabricated["receipt_digest"] = coordinator._digest(fabricated_body)
    path.chmod(0o600)
    path.write_bytes(coordinator._pretty(fabricated))
    path.chmod(0o400)
    with pytest.raises(CoordinatorError, match="schema or outcome boundary"):
        coordinator.verify_accounting_gate(
            path,
            expected_gate_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            expected_claim_sha256="1" * 64,
            expected_execution_uuid=UUID_A,
            expected_global_claim_digest="2" * 64,
            expected_schedule_digest="3" * 64,
            expected_row_receipts_digest="4" * 64,
        )


def test_accounting_gate_rejects_float_or_bool_integer_substitutes(tmp_path: Path) -> None:
    substitutions: tuple[tuple[str, object], ...] = (
        ("schema_version", 1.0),
        ("planned_rows", 120.0),
        ("terminal_rows", True),
        ("worker_count", 4.0),
        ("counts", {"complete": 120.0}),
    )
    for index, (field, bad_value) in enumerate(substitutions):
        body: dict[str, Any] = {
            "schema": "goalzendo.g01_checkpoint_b_final_accounting",
            "schema_version": 1,
            "study_id": "g01",
            "execution_uuid": UUID_A,
            "global_claim_sha256": "1" * 64,
            "global_claim_digest": "2" * 64,
            "schedule_digest": "3" * 64,
            "plan_rows_digest": coordinator.G01_PLAN_ROWS_DIGEST,
            "plan_key_set_digest": coordinator.G01_PLAN_KEY_SET_DIGEST,
            "planned_rows": 120,
            "terminal_rows": 120,
            "counts": {"complete": 120},
            "row_receipts_digest": "4" * 64,
            "worker_count": 4,
            "parent_accounting_outcome_independent": True,
            "parent_accounting_metrics_file_read": False,
            "parent_accounting_predictions_file_read": False,
            "parent_accounting_summary_file_read": False,
            "retry": False,
        }
        body[field] = bad_value
        gate = {**body, "receipt_digest": coordinator._digest(body)}
        path = tmp_path / f"bad-{index}-{field}.json"
        path.write_bytes(coordinator._pretty(gate))
        path.chmod(0o400)
        with pytest.raises(CoordinatorError, match="schema or outcome boundary"):
            coordinator.verify_accounting_gate(
                path,
                expected_gate_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                expected_claim_sha256="1" * 64,
                expected_execution_uuid=UUID_A,
                expected_global_claim_digest="2" * 64,
                expected_schedule_digest="3" * 64,
                expected_row_receipts_digest="4" * 64,
            )


def test_bootstrap_and_coordinator_contain_no_launch_override() -> None:
    combined = ENTRYPOINT.read_text(encoding="utf-8") + (REPO / coordinator.SOURCE_PATHS[1]).read_text(
        encoding="utf-8"
    )
    assert "B_RUNTIME_OVERLAY_FROZEN = True" not in combined
    assert "terminal_no_retry" not in combined
    assert "PYTEST_CURRENT_TEST" not in ENTRYPOINT.read_text(encoding="utf-8")
    assert json.loads(json.dumps({"refusal": coordinator.RUNTIME_OVERLAY_REFUSAL}))["refusal"] == (
        "B_RUNTIME_OVERLAY_NOT_FROZEN"
    )
