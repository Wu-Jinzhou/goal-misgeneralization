from __future__ import annotations

import hashlib
import json
import os
import select
import shutil
import signal
import stat
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import goalzendo_interactive_v2.evaluation_census_execution as execution_module
from goalzendo_interactive_v2.evaluation_census_execution import (
    BUILDER_GUARDIAN_TERMINAL_FILENAME,
    BUILDER_TERMINAL_FILENAME,
    EXECUTION_RECEIPT_FILENAME,
    FREEZE_REQUEST_FILENAME,
    PLAN_FILENAME,
    REPORT_FILENAME,
    STARTED_RECEIPT_FILENAME,
    VERIFIER_GUARDIAN_TERMINAL_FILENAME,
    VERIFIER_TERMINAL_FILENAME,
    EvaluationCensusExecutionV1Error,
    build_external_evaluation_census_registration_receipt_v1,
    canonical_evaluation_census_runner_path_v1,
    execute_engineering_evaluation_census_fixture_for_testing_v1,
    parse_evaluation_census_execution_receipt_v1,
    parse_evaluation_census_freeze_request_v1,
    parse_external_evaluation_census_registration_receipt_v1,
    prepare_engineering_evaluation_census_fixture_for_testing_v1,
    prepare_production_evaluation_census_v1,
    serialize_external_evaluation_census_registration_receipt_v1,
    verify_engineering_evaluation_census_execution_root_for_testing_v1,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _registration(
    prepared: Any,
    path: Path,
    *,
    execution_uuid: str,
    nonce: str,
    reference: str,
    deadline: int = 600,
) -> tuple[Any, str, str]:
    freeze_text = prepared.freeze_request_path.read_text(encoding="ascii")
    receipt = build_external_evaluation_census_registration_receipt_v1(
        prepared.freeze_request,
        freeze_text,
        registration_service="engineering-test-registrar",
        registration_reference=reference,
        registered_at_utc="2020-01-01T00:00:00Z",
        execution_uuid=execution_uuid,
        execution_nonce=nonce,
        execution_deadline_seconds=deadline,
    )
    text = serialize_external_evaluation_census_registration_receipt_v1(receipt)
    path.write_text(text, encoding="ascii")
    return receipt, text, hashlib.sha256(text.encode("ascii")).hexdigest()


def _execute_fixture(
    prepared: Any,
    registration_path: Path,
    registration: Any,
    registration_sha: str,
    output_root: Path,
    *,
    deadline: int = 600,
) -> Any:
    return execute_engineering_evaluation_census_fixture_for_testing_v1(
        plan_path=prepared.plan_path,
        freeze_request_path=prepared.freeze_request_path,
        registration_receipt_path=registration_path,
        expected_plan_digest=prepared.plan.digest,
        expected_plan_bytes_sha256=hashlib.sha256(prepared.plan_path.read_bytes()).hexdigest(),
        expected_registration_receipt_sha256=registration_sha,
        expected_registration_reference=registration.registration_reference,
        expected_execution_uuid=registration.execution_uuid,
        expected_execution_nonce=registration.execution_nonce,
        output_root=output_root,
        runner_source_path=canonical_evaluation_census_runner_path_v1(),
        deadline_seconds=deadline,
    )


@pytest.fixture(scope="module")
def executed_pair(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    base = tmp_path_factory.mktemp("g03-v2-census-execution")
    runner = canonical_evaluation_census_runner_path_v1()
    seed = _sha("g03-v2-execution-determinism-fixture")
    first_prepared = prepare_engineering_evaluation_census_fixture_for_testing_v1(
        seed, base / "prepared-a", runner_source_path=runner
    )
    second_prepared = prepare_engineering_evaluation_census_fixture_for_testing_v1(
        seed, base / "prepared-b", runner_source_path=runner
    )
    first_uuid = "11111111-1111-4111-8111-111111111111"
    second_uuid = "22222222-2222-4222-8222-222222222222"
    first_registration_path = base / "registration-a.json"
    second_registration_path = base / "registration-b.json"
    first_registration, first_registration_text, first_registration_sha = _registration(
        first_prepared,
        first_registration_path,
        execution_uuid=first_uuid,
        nonce=_sha("execution-nonce-a"),
        reference="engineering:test:execution-a",
    )
    second_registration, _, second_registration_sha = _registration(
        second_prepared,
        second_registration_path,
        execution_uuid=second_uuid,
        nonce=_sha("execution-nonce-b"),
        reference="engineering:test:execution-b",
    )
    first = _execute_fixture(
        first_prepared,
        first_registration_path,
        first_registration,
        first_registration_sha,
        base / first_uuid,
    )
    second = _execute_fixture(
        second_prepared,
        second_registration_path,
        second_registration,
        second_registration_sha,
        base / second_uuid,
    )
    receipt_sha = hashlib.sha256(first.execution_receipt_path.read_bytes()).hexdigest()
    verified = verify_engineering_evaluation_census_execution_root_for_testing_v1(
        output_root=first.output_root,
        plan_path=first_prepared.plan_path,
        freeze_request_path=first_prepared.freeze_request_path,
        registration_receipt_path=first_registration_path,
        expected_plan_digest=first_prepared.plan.digest,
        expected_plan_bytes_sha256=hashlib.sha256(first_prepared.plan_path.read_bytes()).hexdigest(),
        expected_registration_receipt_sha256=first_registration_sha,
        expected_registration_reference=first_registration.registration_reference,
        expected_execution_uuid=first_uuid,
        expected_execution_nonce=first_registration.execution_nonce,
        expected_execution_deadline_seconds=600,
        expected_execution_receipt_sha256=receipt_sha,
        expected_execution_receipt_digest=first.execution_receipt_digest,
        runner_source_path=runner,
        verification_deadline_seconds=600,
    )
    return {
        "base": base,
        "prepared": first_prepared,
        "registration": first_registration,
        "registration_text": first_registration_text,
        "registration_path": first_registration_path,
        "registration_sha": first_registration_sha,
        "first": first,
        "second": second,
        "verified": verified,
    }


def _all_keys(value: object) -> Iterator[str]:
    if type(value) is dict:
        for key, child in value.items():
            yield key
            yield from _all_keys(child)
    elif type(value) is list:
        for child in value:
            yield from _all_keys(child)


def test_production_prepare_is_exact_outcome_free_and_completion_last(tmp_path: Path) -> None:
    source_archive = tmp_path / "source.tar"
    constraints = tmp_path / "constraints.txt"
    environment_lock = tmp_path / "environment.lock"
    source_archive.write_bytes(b"opaque-source-archive-pin")
    constraints.write_text("package==1\n", encoding="utf-8")
    environment_lock.write_text("environment-lock-v1\n", encoding="utf-8")
    prepared = prepare_production_evaluation_census_v1(
        _sha("production-prepare-only-never-execute"),
        tmp_path / "prepared-production",
        runner_source_path=canonical_evaluation_census_runner_path_v1(),
        source_archive_path=source_archive,
        constraints_path=constraints,
        environment_lock_path=environment_lock,
    )
    assert prepared.plan.uses_production_attempt_budget
    assert len(prepared.plan.attempts) == 144
    assert len({m.composed_rule_id for a in prepared.plan.attempts for m in a.members}) == 288
    assert prepared.plan.as_obj()["fixed_budget"]["planned_draw_count"] == 1_152
    assert {item.name for item in prepared.output_root.iterdir()} == {
        PLAN_FILENAME,
        FREEZE_REQUEST_FILENAME,
    }
    assert stat.S_IMODE(prepared.output_root.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(item.stat().st_mode) == 0o400 for item in prepared.output_root.iterdir())
    request_obj = prepared.freeze_request.as_obj()
    assert request_obj["fixed_budget"] == {
        "formula_stratum_count": 9,
        "attempts_per_formula_stratum": 16,
        "mirror_attempt_count": 144,
        "draws_per_mirror_attempt": 8,
        "planned_draw_count": 1_152,
        "candidate_pool_size": 32,
        "distinct_evaluation_c_identity_count": 288,
        "early_stop_allowed": False,
    }
    assert not any(key.startswith("observed") for key in _all_keys(request_obj))
    assert "ranking_tables" not in set(_all_keys(request_obj))
    assert prepared.freeze_request_path.stat().st_mtime_ns >= prepared.plan_path.stat().st_mtime_ns


def test_reduced_execution_is_two_fresh_processes_deterministic_and_fully_verified(
    executed_pair: dict[str, Any],
) -> None:
    first = executed_pair["first"]
    second = executed_pair["second"]
    assert first.report_path.read_bytes() == second.report_path.read_bytes()
    assert first.report_digest == second.report_digest
    assert first.report_bytes_sha256 == second.report_bytes_sha256
    assert executed_pair["verified"].report_digest == first.report_digest
    expected_files = {
        STARTED_RECEIPT_FILENAME,
        BUILDER_GUARDIAN_TERMINAL_FILENAME,
        BUILDER_TERMINAL_FILENAME,
        VERIFIER_GUARDIAN_TERMINAL_FILENAME,
        VERIFIER_TERMINAL_FILENAME,
        REPORT_FILENAME,
        EXECUTION_RECEIPT_FILENAME,
    }
    assert {item.name for item in first.output_root.iterdir()} == expected_files
    assert all(stat.S_IMODE(item.stat().st_mode) == 0o400 for item in first.output_root.iterdir())
    receipt = parse_evaluation_census_execution_receipt_v1(
        first.execution_receipt_path.read_text(encoding="ascii")
    )
    assert receipt["accounting"]["mirror_attempt_count"] == 9
    assert receipt["accounting"]["opening_record_count"] == 72
    assert receipt["accounting"]["observed_cell_count"] == 36
    assert receipt["accounting"]["all_36_cells_preserved"]
    assert not receipt["accounting"]["production_attempt_budget_complete"]
    assert not receipt["external_registration_validation"][
        "registration_service_identity_independently_verified_by_repository"
    ]
    assert not receipt["authorization"]["g01_authorized"]
    assert not receipt["authorization"]["g03_scientific_launch_authorized"]


def test_freeze_and_registration_parsers_fail_closed_on_json_and_pin_tampering(
    executed_pair: dict[str, Any],
) -> None:
    prepared = executed_pair["prepared"]
    runner = canonical_evaluation_census_runner_path_v1()
    freeze_text = prepared.freeze_request_path.read_text(encoding="ascii")
    duplicate = freeze_text.replace('"request_kind":', '"request_kind":"x","request_kind":', 1)
    with pytest.raises(EvaluationCensusExecutionV1Error, match="duplicate"):
        parse_evaluation_census_freeze_request_v1(
            duplicate,
            plan=prepared.plan,
            plan_text=prepared.plan_path.read_text(encoding="ascii"),
            runner_source_path=runner,
        )
    nonfinite = freeze_text.replace('"formula_stratum_count":9', '"formula_stratum_count":NaN', 1)
    with pytest.raises(EvaluationCensusExecutionV1Error, match="non-finite"):
        parse_evaluation_census_freeze_request_v1(
            nonfinite,
            plan=prepared.plan,
            plan_text=prepared.plan_path.read_text(encoding="ascii"),
            runner_source_path=runner,
        )
    reordered = json.loads(freeze_text)
    digest = reordered.pop("freeze_request_digest")
    reordered_text = json.dumps({"freeze_request_digest": digest, **reordered}, separators=(",", ":")) + "\n"
    with pytest.raises(EvaluationCensusExecutionV1Error):
        parse_evaluation_census_freeze_request_v1(
            reordered_text,
            plan=prepared.plan,
            plan_text=prepared.plan_path.read_text(encoding="ascii"),
            runner_source_path=runner,
        )
    boolean = json.loads(freeze_text)
    boolean["fixed_budget"]["early_stop_allowed"] = 0
    with pytest.raises(EvaluationCensusExecutionV1Error):
        parse_evaluation_census_freeze_request_v1(
            json.dumps(boolean, separators=(",", ":")) + "\n",
            plan=prepared.plan,
            plan_text=prepared.plan_path.read_text(encoding="ascii"),
            runner_source_path=runner,
        )

    registration_text = executed_pair["registration_text"]
    registration = executed_pair["registration"]
    with pytest.raises(EvaluationCensusExecutionV1Error):
        parse_external_evaluation_census_registration_receipt_v1(
            registration_text,
            freeze_request=prepared.freeze_request,
            freeze_request_text=freeze_text,
            expected_bytes_sha256=executed_pair["registration_sha"],
            expected_registration_reference="engineering:test:wrong",
            expected_execution_uuid=registration.execution_uuid,
            expected_execution_nonce=registration.execution_nonce,
            expected_execution_deadline_seconds=600,
        )
    forged = json.loads(registration_text)
    forged["authorization"]["g01_authorized"] = True
    forged_text = json.dumps(forged, separators=(",", ":")) + "\n"
    with pytest.raises(EvaluationCensusExecutionV1Error):
        parse_external_evaluation_census_registration_receipt_v1(
            forged_text,
            freeze_request=prepared.freeze_request,
            freeze_request_text=freeze_text,
            expected_bytes_sha256=hashlib.sha256(forged_text.encode()).hexdigest(),
            expected_registration_reference=registration.registration_reference,
            expected_execution_uuid=registration.execution_uuid,
            expected_execution_nonce=registration.execution_nonce,
            expected_execution_deadline_seconds=600,
        )


def test_execute_preflight_rejects_wrong_pins_source_drift_overwrite_and_partial_root(
    executed_pair: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepared = executed_pair["prepared"]
    registration = executed_pair["registration"]
    registration_path = executed_pair["registration_path"]
    registration_sha = executed_pair["registration_sha"]
    existing_root = executed_pair["first"].output_root
    with pytest.raises(EvaluationCensusExecutionV1Error, match="expected SHA-256"):
        execute_engineering_evaluation_census_fixture_for_testing_v1(
            plan_path=prepared.plan_path,
            freeze_request_path=prepared.freeze_request_path,
            registration_receipt_path=registration_path,
            expected_plan_digest=prepared.plan.digest,
            expected_plan_bytes_sha256=_sha("wrong-plan-pin"),
            expected_registration_receipt_sha256=registration_sha,
            expected_registration_reference=registration.registration_reference,
            expected_execution_uuid=registration.execution_uuid,
            expected_execution_nonce=registration.execution_nonce,
            output_root=existing_root,
            runner_source_path=canonical_evaluation_census_runner_path_v1(),
        )
    with pytest.raises(EvaluationCensusExecutionV1Error, match="registration-receipt bytes"):
        execute_engineering_evaluation_census_fixture_for_testing_v1(
            plan_path=prepared.plan_path,
            freeze_request_path=prepared.freeze_request_path,
            registration_receipt_path=registration_path,
            expected_plan_digest=prepared.plan.digest,
            expected_plan_bytes_sha256=hashlib.sha256(prepared.plan_path.read_bytes()).hexdigest(),
            expected_registration_receipt_sha256=_sha("wrong-registration-pin"),
            expected_registration_reference=registration.registration_reference,
            expected_execution_uuid=registration.execution_uuid,
            expected_execution_nonce=registration.execution_nonce,
            output_root=existing_root,
            runner_source_path=canonical_evaluation_census_runner_path_v1(),
        )
    with pytest.raises(EvaluationCensusExecutionV1Error, match="reference differs"):
        execute_engineering_evaluation_census_fixture_for_testing_v1(
            plan_path=prepared.plan_path,
            freeze_request_path=prepared.freeze_request_path,
            registration_receipt_path=registration_path,
            expected_plan_digest=prepared.plan.digest,
            expected_plan_bytes_sha256=hashlib.sha256(prepared.plan_path.read_bytes()).hexdigest(),
            expected_registration_receipt_sha256=registration_sha,
            expected_registration_reference="engineering:test:wrong-reference",
            expected_execution_uuid=registration.execution_uuid,
            expected_execution_nonce=registration.execution_nonce,
            output_root=existing_root,
            runner_source_path=canonical_evaluation_census_runner_path_v1(),
        )
    with pytest.raises(EvaluationCensusExecutionV1Error, match="already exists"):
        _execute_fixture(
            prepared,
            registration_path,
            registration,
            registration_sha,
            existing_root,
        )

    original = execution_module._execution_code_binding

    def drifted(path: Path) -> dict[str, Any]:
        value = original(path)
        value["execution_module"]["sha256"] = _sha("source-drift")
        return value

    monkeypatch.setattr(execution_module, "_execution_code_binding", drifted)
    with pytest.raises(EvaluationCensusExecutionV1Error, match="rederivation"):
        _execute_fixture(
            prepared,
            registration_path,
            registration,
            registration_sha,
            existing_root,
        )
    monkeypatch.setattr(execution_module, "_execution_code_binding", original)

    opaque_source = tmp_path / "opaque-source"
    constraints = tmp_path / "constraints.txt"
    lock = tmp_path / "lock.txt"
    opaque_source.write_bytes(b"opaque")
    constraints.write_text("constraint\n", encoding="utf-8")
    lock.write_text("lock\n", encoding="utf-8")
    with pytest.raises(EvaluationCensusExecutionV1Error, match="production execution requires"):
        execution_module._execute(
            plan_path=prepared.plan_path,
            freeze_request_path=prepared.freeze_request_path,
            registration_receipt_path=registration_path,
            expected_plan_digest=prepared.plan.digest,
            expected_plan_bytes_sha256=hashlib.sha256(prepared.plan_path.read_bytes()).hexdigest(),
            expected_registration_receipt_sha256=registration_sha,
            expected_registration_reference=registration.registration_reference,
            expected_execution_uuid=registration.execution_uuid,
            expected_execution_nonce=registration.execution_nonce,
            output_root=existing_root,
            runner_source_path=canonical_evaluation_census_runner_path_v1(),
            deadline_seconds=600,
            controller_argv=("test",),
            require_production=True,
            source_archive_path=opaque_source,
            constraints_path=constraints,
            environment_lock_path=lock,
        )

    partial_uuid = "33333333-3333-4333-8333-333333333333"
    partial_registration_path = tmp_path / "partial-registration.json"
    partial_registration, _, partial_sha = _registration(
        prepared,
        partial_registration_path,
        execution_uuid=partial_uuid,
        nonce=_sha("partial-nonce"),
        reference="engineering:test:partial",
    )
    partial_root = tmp_path / partial_uuid
    partial_root.mkdir(mode=0o700)
    (partial_root / STARTED_RECEIPT_FILENAME).write_text("partial\n", encoding="ascii")
    with pytest.raises(EvaluationCensusExecutionV1Error, match="already exists"):
        _execute_fixture(
            prepared,
            partial_registration_path,
            partial_registration,
            partial_sha,
            partial_root,
        )


def _root_verify_kwargs(executed_pair: dict[str, Any], root: Path) -> dict[str, Any]:
    prepared = executed_pair["prepared"]
    registration = executed_pair["registration"]
    receipt = executed_pair["first"].execution_receipt_path.read_bytes()
    return {
        "output_root": root,
        "plan_path": prepared.plan_path,
        "freeze_request_path": prepared.freeze_request_path,
        "registration_receipt_path": executed_pair["registration_path"],
        "expected_plan_digest": prepared.plan.digest,
        "expected_plan_bytes_sha256": hashlib.sha256(prepared.plan_path.read_bytes()).hexdigest(),
        "expected_registration_receipt_sha256": executed_pair["registration_sha"],
        "expected_registration_reference": registration.registration_reference,
        "expected_execution_uuid": registration.execution_uuid,
        "expected_execution_nonce": registration.execution_nonce,
        "expected_execution_deadline_seconds": 600,
        "expected_execution_receipt_sha256": hashlib.sha256(receipt).hexdigest(),
        "expected_execution_receipt_digest": executed_pair["first"].execution_receipt_digest,
        "runner_source_path": canonical_evaluation_census_runner_path_v1(),
        "verification_deadline_seconds": 600,
    }


def test_root_verifier_rejects_wrong_external_pin_inventory_mode_and_nested_runtime(
    executed_pair: dict[str, Any], tmp_path: Path
) -> None:
    original = executed_pair["first"].output_root
    kwargs = _root_verify_kwargs(executed_pair, original)
    kwargs["expected_execution_receipt_sha256"] = _sha("wrong-completion-pin")
    with pytest.raises(EvaluationCensusExecutionV1Error, match="externally expected"):
        verify_engineering_evaluation_census_execution_root_for_testing_v1(**kwargs)

    extra_parent = tmp_path / "extra"
    extra_root = extra_parent / original.name
    shutil.copytree(original, extra_root)
    extra = extra_root / "unexpected.json"
    extra.write_text("{}\n", encoding="ascii")
    extra.chmod(0o400)
    with pytest.raises(EvaluationCensusExecutionV1Error, match="missing, extra"):
        verify_engineering_evaluation_census_execution_root_for_testing_v1(
            **_root_verify_kwargs(executed_pair, extra_root)
        )

    mode_parent = tmp_path / "mode"
    mode_root = mode_parent / original.name
    shutil.copytree(original, mode_root)
    (mode_root / REPORT_FILENAME).chmod(0o600)
    with pytest.raises(EvaluationCensusExecutionV1Error, match="0400"):
        verify_engineering_evaluation_census_execution_root_for_testing_v1(
            **_root_verify_kwargs(executed_pair, mode_root)
        )

    nested_parent = tmp_path / "nested"
    nested_root = nested_parent / original.name
    shutil.copytree(original, nested_root)
    receipt_path = nested_root / EXECUTION_RECEIPT_FILENAME
    receipt_obj = json.loads(receipt_path.read_text(encoding="ascii"))
    rss = receipt_obj["runtime_evidence"]["builder"]["maximum_resident_set_size"]
    rss["raw_value"] += 1
    rss["normalized_bytes"] += 1024 if rss["raw_unit"] == "kibibytes" else 1
    unsigned = {key: value for key, value in receipt_obj.items() if key != "execution_receipt_digest"}
    receipt_obj["execution_receipt_digest"] = execution_module._digest(
        unsigned, domain=execution_module._EXECUTION_RECEIPT_DOMAIN
    )
    receipt_text = json.dumps(receipt_obj, separators=(",", ":")) + "\n"
    receipt_path.chmod(0o600)
    receipt_path.write_text(receipt_text, encoding="ascii")
    receipt_path.chmod(0o400)
    nested_kwargs = _root_verify_kwargs(executed_pair, nested_root)
    nested_kwargs["expected_execution_receipt_sha256"] = hashlib.sha256(receipt_text.encode()).hexdigest()
    nested_kwargs["expected_execution_receipt_digest"] = receipt_obj["execution_receipt_digest"]
    with pytest.raises(EvaluationCensusExecutionV1Error, match="runtime differs"):
        verify_engineering_evaluation_census_execution_root_for_testing_v1(**nested_kwargs)


def test_receipt_parser_rejects_duplicate_reordered_nonfinite_boolean_and_forged_authorization(
    executed_pair: dict[str, Any],
) -> None:
    text = executed_pair["first"].execution_receipt_path.read_text(encoding="ascii")
    duplicate = text.replace('"receipt_kind":', '"receipt_kind":"x","receipt_kind":', 1)
    with pytest.raises(EvaluationCensusExecutionV1Error, match="duplicate"):
        parse_evaluation_census_execution_receipt_v1(duplicate)
    nonfinite = text.replace('"mirror_attempt_count":9', '"mirror_attempt_count":NaN', 1)
    with pytest.raises(EvaluationCensusExecutionV1Error, match="non-finite"):
        parse_evaluation_census_execution_receipt_v1(nonfinite)
    value = json.loads(text)
    digest = value.pop("execution_receipt_digest")
    reordered = json.dumps({"execution_receipt_digest": digest, **value}, separators=(",", ":")) + "\n"
    with pytest.raises(EvaluationCensusExecutionV1Error):
        parse_evaluation_census_execution_receipt_v1(reordered)
    for path, replacement in (
        (("accounting", "all_36_cells_preserved"), 1),
        (("authorization", "g01_authorized"), True),
        (("completion", "fresh_verifier_succeeded"), False),
    ):
        forged = json.loads(text)
        forged[path[0]][path[1]] = replacement
        unsigned = {key: item for key, item in forged.items() if key != "execution_receipt_digest"}
        forged["execution_receipt_digest"] = execution_module._digest(
            unsigned, domain=execution_module._EXECUTION_RECEIPT_DOMAIN
        )
        with pytest.raises(EvaluationCensusExecutionV1Error):
            parse_evaluation_census_execution_receipt_v1(json.dumps(forged, separators=(",", ":")) + "\n")


@pytest.mark.skipif(not hasattr(os, "killpg"), reason="requires POSIX process groups")
def test_watchdog_sigkill_triggers_worker_owned_parent_death_guard() -> None:
    channels = execution_module._ParentDeathPipe.create()
    started_ns = time.monotonic_ns()
    probe_self_deadline_ns = started_ns + 15_000_000_000
    watchdog_deadline_ns = started_ns + 30_000_000_000
    common = [
        str(Path(sys.executable).resolve(strict=True)),
        "-m",
        "goalzendo_interactive_v2.evaluation_census_execution",
    ]
    worker_argv = [
        *common,
        "__containment-probe-worker",
        "--parent-guard-fd",
        str(channels.worker_read_fd),
        "--guardian-ready-write-fd",
        str(channels.guardian_ready_write_fd),
        "--worker-lifetime-write-fd",
        str(channels.worker_lifetime_write_fd),
        "--self-deadline-monotonic-ns",
        str(probe_self_deadline_ns),
    ]
    watchdog_argv = [
        *common,
        "__watchdog-worker",
        "--parent-death-fd",
        str(channels.read_fd),
        "--worker-death-read-fd",
        str(channels.worker_read_fd),
        "--worker-death-write-fd",
        str(channels.worker_write_fd),
        "--worker-pid-report-fd",
        str(channels.pid_write_fd),
        "--guardian-ready-read-fd",
        str(channels.guardian_ready_read_fd),
        "--guardian-ready-write-fd",
        str(channels.guardian_ready_write_fd),
        "--worker-lifetime-read-fd",
        str(channels.worker_lifetime_read_fd),
        "--worker-lifetime-write-fd",
        str(channels.worker_lifetime_write_fd),
        "--deadline-monotonic-ns",
        str(watchdog_deadline_ns),
        "--worker-argv-json",
        json.dumps(worker_argv, separators=(",", ":")),
    ]
    inherited_signal_state_wrapper = (
        "import os,signal,sys;"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
        "signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGTERM});"
        "os.execv(sys.argv[1],sys.argv[1:])"
    )
    watchdog: subprocess.Popen[bytes] | None = None
    try:
        watchdog = subprocess.Popen(
            [
                str(Path(sys.executable).resolve(strict=True)),
                "-c",
                inherited_signal_state_wrapper,
                *watchdog_argv,
            ],
            cwd=execution_module._repository_root(),
            env=execution_module._safe_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            pass_fds=(
                channels.read_fd,
                channels.worker_read_fd,
                channels.worker_write_fd,
                channels.pid_write_fd,
                channels.guardian_ready_read_fd,
                channels.guardian_ready_write_fd,
                channels.worker_lifetime_read_fd,
                channels.worker_lifetime_write_fd,
            ),
        )
        for field in (
            "read_fd",
            "worker_read_fd",
            "worker_write_fd",
            "pid_write_fd",
            "guardian_ready_read_fd",
            "guardian_ready_write_fd",
            "worker_lifetime_write_fd",
        ):
            os.close(getattr(channels, field))
            setattr(channels, field, -1)
        readable, _, _ = select.select([channels.pid_read_fd], [], [], 10)
        assert readable
        reported_worker_pid = int(os.read(channels.pid_read_fd, 32).strip())
        assert reported_worker_pid > 0
        os.close(channels.pid_read_fd)
        channels.pid_read_fd = -1
        os.killpg(watchdog.pid, signal.SIGKILL)
        watchdog.wait(timeout=10)
        execution_module._require_worker_lifetime_eof(channels, timeout=10)
    finally:
        if watchdog is not None:
            execution_module._terminate_process_group(watchdog)
        if channels.worker_lifetime_read_fd >= 0:
            execution_module._require_worker_lifetime_eof(channels, timeout=10)
        channels.close()


def test_production_cli_has_no_engineering_fixture_or_combined_action() -> None:
    completed = subprocess.run(
        [str(Path(sys.executable).resolve()), str(canonical_evaluation_census_runner_path_v1()), "--help"],
        cwd=execution_module._repository_root(),
        env=execution_module._safe_environment(),
        check=True,
        capture_output=True,
        text=True,
    )
    assert "{prepare,execute,verify}" in completed.stdout
    assert "engineering" not in completed.stdout.lower()
    assert "combined" not in completed.stdout.lower()
