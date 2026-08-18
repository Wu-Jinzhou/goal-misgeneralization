from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
from argparse import Namespace
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import goalzendo_interactive_v2.nested_opening_feasibility_execution as execution_module
from goalzendo_interactive_v2.nested_opening_feasibility import (
    parse_nested_opening_construction_feasibility_report_for_testing_v1,
)
from goalzendo_interactive_v2.nested_opening_feasibility_execution import (
    BUILDER_GUARDIAN_TERMINAL_FILENAME,
    BUILDER_TERMINAL_FILENAME,
    EXECUTION_RECEIPT_FILENAME,
    FAILURE_RECEIPT_FILENAME,
    FREEZE_REQUEST_FILENAME,
    NESTED_PLAN_FILENAME,
    PRODUCTION_EXECUTION_REFUSAL_CODE,
    REPORT_FILENAME,
    STARTED_RECEIPT_FILENAME,
    UPSTREAM_CENSUS_PLAN_FILENAME,
    VERIFIER_GUARDIAN_TERMINAL_FILENAME,
    VERIFIER_TERMINAL_FILENAME,
    NestedOpeningFeasibilityExecutionV1Error,
    build_external_nested_opening_feasibility_registration_receipt_for_testing_v1,
    canonical_nested_opening_feasibility_runner_path_v1,
    execute_engineering_nested_opening_feasibility_fixture_for_testing_v1,
    execute_production_nested_opening_feasibility_v1,
    parse_external_nested_opening_feasibility_registration_receipt_v1,
    parse_nested_opening_feasibility_freeze_request_for_testing_v1,
    prepare_engineering_nested_opening_feasibility_fixture_for_testing_v1,
    prepare_production_nested_opening_feasibility_v1,
    serialize_external_nested_opening_feasibility_registration_receipt_for_testing_v1,
    verify_engineering_nested_opening_feasibility_execution_root_for_testing_v1,
    verify_production_nested_opening_feasibility_execution_root_v1,
)

_EXECUTION_UUID = "11111111-1111-4111-8111-111111111111"
_EXECUTION_NONCE_LABEL = "nested-opening-engineering-execution-nonce"
_REGISTRATION_REFERENCE = "engineering:test:nested-opening"
_DEADLINE_SECONDS = 600


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    ) + "\n"


def _all_keys(value: object) -> tuple[str, ...]:
    keys: list[str] = []
    if type(value) is dict:
        for key, child in value.items():
            keys.append(key)
            keys.extend(_all_keys(child))
    elif type(value) is list:
        for child in value:
            keys.extend(_all_keys(child))
    return tuple(keys)


def _freeze_parse_kwargs(prepared: Any, upstream_text: str, nested_text: str) -> dict[str, Any]:
    return {
        "upstream_census_plan": prepared.upstream_census_plan,
        "upstream_census_plan_text": upstream_text,
        "nested_plan": prepared.nested_plan,
        "nested_plan_text": nested_text,
        "expected_upstream_census_plan_digest": prepared.upstream_census_plan_digest,
        "expected_upstream_census_plan_bytes_sha256": (
            prepared.upstream_census_plan_bytes_sha256
        ),
        "expected_nested_plan_digest": prepared.nested_plan_digest,
        "expected_nested_plan_bytes_sha256": prepared.nested_plan_bytes_sha256,
        "runner_source_path": canonical_nested_opening_feasibility_runner_path_v1(),
    }


def _registration(
    prepared: Any,
    path: Path,
    *,
    execution_uuid: str,
    execution_nonce: str,
    registration_reference: str,
    deadline_seconds: int = _DEADLINE_SECONDS,
) -> tuple[Any, str, str]:
    upstream_text = prepared.upstream_census_plan_path.read_text(encoding="ascii")
    nested_text = prepared.nested_plan_path.read_text(encoding="ascii")
    freeze_text = prepared.freeze_request_path.read_text(encoding="ascii")
    replay = _freeze_parse_kwargs(prepared, upstream_text, nested_text)
    receipt = (
        build_external_nested_opening_feasibility_registration_receipt_for_testing_v1(
            prepared.freeze_request,
            freeze_text,
            **replay,
            registration_service="engineering-test-registrar",
            registration_reference=registration_reference,
            registered_at_utc="2020-01-01T00:00:00Z",
            execution_uuid=execution_uuid,
            execution_nonce=execution_nonce,
            execution_deadline_seconds=deadline_seconds,
        )
    )
    text = serialize_external_nested_opening_feasibility_registration_receipt_for_testing_v1(
        receipt,
        prepared.freeze_request,
        freeze_text,
        **replay,
    )
    path.write_text(text, encoding="ascii")
    path.chmod(0o400)
    return receipt, text, _sha_bytes(text.encode("ascii"))


def _execution_kwargs(
    prepared: Any,
    registration_path: Path,
    registration: Any,
    registration_sha: str,
    output_root: Path,
    *,
    deadline_seconds: int = _DEADLINE_SECONDS,
) -> dict[str, Any]:
    return {
        "upstream_census_plan_path": prepared.upstream_census_plan_path,
        "nested_plan_path": prepared.nested_plan_path,
        "freeze_request_path": prepared.freeze_request_path,
        "registration_receipt_path": registration_path,
        "expected_upstream_census_plan_digest": prepared.upstream_census_plan_digest,
        "expected_upstream_census_plan_bytes_sha256": (
            prepared.upstream_census_plan_bytes_sha256
        ),
        "expected_nested_plan_digest": prepared.nested_plan_digest,
        "expected_nested_plan_bytes_sha256": prepared.nested_plan_bytes_sha256,
        "expected_registration_receipt_sha256": registration_sha,
        "expected_registration_reference": registration.registration_reference,
        "expected_execution_uuid": registration.execution_uuid,
        "expected_execution_nonce": registration.execution_nonce,
        "output_root": output_root,
        "runner_source_path": canonical_nested_opening_feasibility_runner_path_v1(),
        "deadline_seconds": deadline_seconds,
    }


def _verification_kwargs(lifecycle: dict[str, Any], root: Path) -> dict[str, Any]:
    prepared = lifecycle["prepared"]
    registration = lifecycle["registration"]
    executed = lifecycle["executed"]
    receipt_bytes = executed.execution_receipt_path.read_bytes()
    return {
        "output_root": root,
        "upstream_census_plan_path": prepared.upstream_census_plan_path,
        "nested_plan_path": prepared.nested_plan_path,
        "freeze_request_path": prepared.freeze_request_path,
        "registration_receipt_path": lifecycle["registration_path"],
        "expected_upstream_census_plan_digest": prepared.upstream_census_plan_digest,
        "expected_upstream_census_plan_bytes_sha256": (
            prepared.upstream_census_plan_bytes_sha256
        ),
        "expected_nested_plan_digest": prepared.nested_plan_digest,
        "expected_nested_plan_bytes_sha256": prepared.nested_plan_bytes_sha256,
        "expected_registration_receipt_sha256": lifecycle["registration_sha"],
        "expected_registration_reference": registration.registration_reference,
        "expected_execution_uuid": registration.execution_uuid,
        "expected_execution_nonce": registration.execution_nonce,
        "expected_execution_deadline_seconds": registration.execution_deadline_seconds,
        "expected_execution_receipt_sha256": _sha_bytes(receipt_bytes),
        "expected_execution_receipt_digest": executed.execution_receipt_digest,
        "runner_source_path": canonical_nested_opening_feasibility_runner_path_v1(),
        "verification_deadline_seconds": _DEADLINE_SECONDS,
    }


@pytest.fixture(scope="module")
def lifecycle(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    base = tmp_path_factory.mktemp("g03-v2-nested-opening-execution")
    prepared = prepare_engineering_nested_opening_feasibility_fixture_for_testing_v1(
        _sha("nested-opening-upstream-engineering-seed"),
        _sha("nested-opening-independent-generator-seed"),
        base / "prepared",
        runner_source_path=canonical_nested_opening_feasibility_runner_path_v1(),
        attempts_per_formula_stratum=1,
        candidate_pool_size=1,
    )
    registration_path = base / "registration.json"
    registration, registration_text, registration_sha = _registration(
        prepared,
        registration_path,
        execution_uuid=_EXECUTION_UUID,
        execution_nonce=_sha(_EXECUTION_NONCE_LABEL),
        registration_reference=_REGISTRATION_REFERENCE,
    )
    executed = execute_engineering_nested_opening_feasibility_fixture_for_testing_v1(
        **_execution_kwargs(
            prepared,
            registration_path,
            registration,
            registration_sha,
            base / _EXECUTION_UUID,
        )
    )
    result: dict[str, Any] = {
        "base": base,
        "prepared": prepared,
        "registration": registration,
        "registration_text": registration_text,
        "registration_path": registration_path,
        "registration_sha": registration_sha,
        "executed": executed,
    }
    verified = verify_engineering_nested_opening_feasibility_execution_root_for_testing_v1(
        **_verification_kwargs(result, executed.output_root)
    )
    upstream_text = prepared.upstream_census_plan_path.read_text(encoding="ascii")
    nested_text = prepared.nested_plan_path.read_text(encoding="ascii")
    report_text = executed.report_path.read_text(encoding="ascii")
    report = parse_nested_opening_construction_feasibility_report_for_testing_v1(
        report_text,
        plan_text=nested_text,
        census_plan_text=upstream_text,
        expected_census_plan_digest=prepared.upstream_census_plan_digest,
        expected_census_plan_bytes_sha256=prepared.upstream_census_plan_bytes_sha256,
        expected_plan_digest=prepared.nested_plan_digest,
        expected_report_digest=executed.report_digest,
    )
    result.update({"verified": verified, "report": report})
    return result


def test_reduced_prepare_execute_and_third_replay_are_exactly_scoped(
    lifecycle: dict[str, Any],
) -> None:
    prepared = lifecycle["prepared"]
    executed = lifecycle["executed"]
    verified = lifecycle["verified"]
    report = lifecycle["report"]
    assert prepared.freeze_request.plan_class == "engineering_test_fixture"
    assert len(prepared.nested_plan.attempts) == 9
    assert len(report.attempts) == 9
    assert sum(len(attempt.members) for attempt in report.attempts) == 18
    assert executed.report_digest == verified.report_digest
    assert executed.report_bytes_sha256 == verified.report_bytes_sha256
    assert executed.execution_receipt_digest == verified.execution_receipt_digest
    assert verified.fresh_replay_terminal_digest
    assert verified.fresh_replay_watchdog_terminal_digest

    prepared_names = {item.name for item in prepared.output_root.iterdir()}
    assert prepared_names == {
        UPSTREAM_CENSUS_PLAN_FILENAME,
        NESTED_PLAN_FILENAME,
        FREEZE_REQUEST_FILENAME,
    }
    root_names = {item.name for item in executed.output_root.iterdir()}
    assert root_names == {
        STARTED_RECEIPT_FILENAME,
        BUILDER_GUARDIAN_TERMINAL_FILENAME,
        BUILDER_TERMINAL_FILENAME,
        VERIFIER_GUARDIAN_TERMINAL_FILENAME,
        VERIFIER_TERMINAL_FILENAME,
        REPORT_FILENAME,
        EXECUTION_RECEIPT_FILENAME,
    }
    assert stat.S_IMODE(prepared.output_root.stat().st_mode) == 0o700
    assert stat.S_IMODE(executed.output_root.stat().st_mode) == 0o700
    assert all(
        stat.S_IMODE(item.stat().st_mode) == 0o400
        for item in (*prepared.output_root.iterdir(), *executed.output_root.iterdir())
    )

    freeze_obj = json.loads(prepared.freeze_request_path.read_text(encoding="ascii"))
    receipt_obj = json.loads(executed.execution_receipt_path.read_text(encoding="ascii"))
    assert freeze_obj["fixed_budget"]["mirror_attempt_count"] == 9
    assert freeze_obj["fixed_budget"]["planned_member_construction_count"] == 18
    assert freeze_obj["fixed_budget"]["planned_opening_assessment_count"] == 72
    assert receipt_obj["accounting"]["mirror_attempt_count"] == 9
    assert receipt_obj["accounting"]["member_record_count"] == 18
    assert receipt_obj["accounting"]["planned_opening_assessment_count"] == 72
    assert receipt_obj["execution_scope"] == {
        "engineering_fixture_only": True,
        "production_evidence": False,
        "study_evidence": False,
    }
    assert verified.verification_receipt["execution_scope"] == receipt_obj["execution_scope"]
    assert all(
        value is False
        for value in receipt_obj["reconstruction_claim_boundary"].values()
    )
    assert all(value is False for key, value in receipt_obj["authorization"].items() if key != "scope")
    assert receipt_obj["scientific_claim_boundary"][
        "cpu_nested_opening_construction_feasibility_report_present"
    ] is True
    assert all(
        value is False
        for key, value in receipt_obj["scientific_claim_boundary"].items()
        if key != "cpu_nested_opening_construction_feasibility_report_present"
    )
    assert freeze_obj["scientific_claim_boundary"][
        "cpu_nested_opening_construction_feasibility_report_present"
    ] is False
    for watchdog_name in (
        BUILDER_GUARDIAN_TERMINAL_FILENAME,
        VERIFIER_GUARDIAN_TERMINAL_FILENAME,
    ):
        watchdog = json.loads(
            (executed.output_root / watchdog_name).read_text(encoding="ascii")
        )
        containment = watchdog["containment"]
        assert containment["guardian_ready_before_pid_report"] is True
        assert containment["worker_lifetime_pipe_eof_observed"] is True
        assert containment["worker_reaped_by_owning_watchdog"] is True
        assert containment[
            "forwarded_data_fd_copies_closed_immediately_after_spawn"
        ] is True
        assert containment[
            "same_group_cleanup_signals_attempted_before_direct_worker_reap"
        ] is True
        assert containment["post_reap_process_group_signal_attempted"] is False
        assert containment["detached_descendant_absence_os_proven"] is False
        assert not any("signals_sent" in key for key in containment)


def test_pair_witness_is_rederived_per_attempt_with_full_five_field_key() -> None:
    base_key = {
        "version_space_size": 8,
        "minimax_depth": 2,
        "target_formula_stratum": 1,
        "greedy_reference_query_count": 3,
        "best_first_query_branch_size_pair": [2, 5],
    }

    def member(identity: str, common_key: dict[str, Any]) -> SimpleNamespace:
        return SimpleNamespace(
            status="completed",
            openings=(),
            exact_match=SimpleNamespace(passed=True, common_key=common_key),
            member_feasibility_witness=True,
            member_plan=SimpleNamespace(composed_rule_id=identity),
        )

    mismatched = dict(base_key)
    mismatched["best_first_query_branch_size_pair"] = [2, 6]
    # The two attempts deliberately cross-compensate at the aggregate level:
    # one stored True is actually False and one stored False is actually True.
    attempts = (
        SimpleNamespace(
            members=(member("a", base_key), member("b", mismatched)),
            mirror_pair_disjoint=True,
            mirror_pair_feasibility_witness=True,
        ),
        SimpleNamespace(
            members=(member("c", base_key), member("d", dict(base_key))),
            mirror_pair_disjoint=True,
            mirror_pair_feasibility_witness=False,
        ),
    )
    report = SimpleNamespace(
        attempts=attempts,
        observed_m_q_summary=tuple(
            SimpleNamespace(m=m, q=q) for m in range(8, 17) for q in range(1, 5)
        ),
    )
    with pytest.raises(
        NestedOpeningFeasibilityExecutionV1Error,
        match="per-attempt full common-key replay",
    ):
        execution_module._report_accounting(report)


def _production_kwargs(root: Path) -> dict[str, Any]:
    return {
        "upstream_census_plan_path": root / "missing-upstream.json",
        "nested_plan_path": root / "missing-nested.json",
        "freeze_request_path": root / "missing-freeze.json",
        "registration_receipt_path": root / "missing-registration.json",
        "expected_upstream_census_plan_digest": _sha("upstream"),
        "expected_upstream_census_plan_bytes_sha256": _sha("upstream-bytes"),
        "expected_nested_plan_digest": _sha("nested"),
        "expected_nested_plan_bytes_sha256": _sha("nested-bytes"),
        "expected_registration_receipt_sha256": _sha("receipt"),
        "expected_registration_reference": "external:registration:test",
        "expected_execution_uuid": "22222222-2222-4222-8222-222222222222",
        "expected_execution_nonce": _sha("production-refusal-nonce"),
        "output_root": root / "22222222-2222-4222-8222-222222222222",
        "runner_source_path": canonical_nested_opening_feasibility_runner_path_v1(),
        "deadline_seconds": 600,
        "controller_argv": ("test", "production"),
        "source_archive_path": root / "missing-source.tar",
        "constraints_path": root / "missing-constraints.txt",
        "environment_lock_path": root / "missing-environment.lock",
    }


@pytest.mark.parametrize(
    "worker_action", ("__build-worker", "__verify-worker", "__watchdog-worker")
)
def test_hidden_worker_production_flag_refuses_before_argument_or_fd_parsing(
    worker_action: str,
) -> None:
    with pytest.raises(
        NestedOpeningFeasibilityExecutionV1Error,
        match=PRODUCTION_EXECUTION_REFUSAL_CODE,
    ):
        execution_module._module_worker_main(
            (worker_action, "--require-production")
        )


def test_all_production_execution_and_verification_entry_points_refuse_before_mutation(
    tmp_path: Path,
) -> None:
    kwargs = _production_kwargs(tmp_path)
    output_root = kwargs["output_root"]
    with pytest.raises(
        NestedOpeningFeasibilityExecutionV1Error,
        match=PRODUCTION_EXECUTION_REFUSAL_CODE,
    ):
        execute_production_nested_opening_feasibility_v1(**kwargs)
    assert not output_root.exists()

    with pytest.raises(
        NestedOpeningFeasibilityExecutionV1Error,
        match=PRODUCTION_EXECUTION_REFUSAL_CODE,
    ):
        execution_module._execute(
            **kwargs,
            require_production=True,
            entry_started_monotonic_ns=1,
            entry_started_wall_utc="2020-01-01T00:00:00.000000Z",
        )
    assert not output_root.exists()
    with pytest.raises(
        NestedOpeningFeasibilityExecutionV1Error,
        match=PRODUCTION_EXECUTION_REFUSAL_CODE,
    ):
        verify_production_nested_opening_feasibility_execution_root_v1()
    assert not output_root.exists()


@pytest.mark.parametrize("action", ("execute", "verify"))
def test_runner_refuses_production_before_any_project_import_or_output_mutation(
    action: str, tmp_path: Path
) -> None:
    marker = tmp_path / "imported-project-package"
    trap = tmp_path / "trap"
    package = trap / "goalzendo_interactive_v2"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('imported')\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "must-not-exist"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(trap)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            str(canonical_nested_opening_feasibility_runner_path_v1()),
            action,
            "--output-root",
            str(output_root),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 2
    assert PRODUCTION_EXECUTION_REFUSAL_CODE in result.stderr
    assert result.stdout == ""
    assert not marker.exists()
    assert not output_root.exists()


def test_distinct_seeds_and_exact_canonical_freeze_bytes_are_enforced(
    lifecycle: dict[str, Any], tmp_path: Path
) -> None:
    seed = _sha("same-seed-is-forbidden")
    same_seed_root = tmp_path / "same-seed"
    with pytest.raises(
        NestedOpeningFeasibilityExecutionV1Error,
        match="independently chosen",
    ):
        prepare_engineering_nested_opening_feasibility_fixture_for_testing_v1(
            seed,
            seed,
            same_seed_root,
            runner_source_path=canonical_nested_opening_feasibility_runner_path_v1(),
        )
    assert not same_seed_root.exists()

    prepared = lifecycle["prepared"]
    upstream_text = prepared.upstream_census_plan_path.read_text(encoding="ascii")
    nested_text = prepared.nested_plan_path.read_text(encoding="ascii")
    freeze_text = prepared.freeze_request_path.read_text(encoding="ascii")
    kwargs = _freeze_parse_kwargs(prepared, upstream_text, nested_text)
    assert (
        parse_nested_opening_feasibility_freeze_request_for_testing_v1(
            freeze_text, **kwargs
        )
        == prepared.freeze_request
    )
    duplicate = freeze_text.replace(
        '"request_kind":', '"request_kind":"duplicate","request_kind":', 1
    )
    with pytest.raises(NestedOpeningFeasibilityExecutionV1Error, match="duplicate"):
        parse_nested_opening_feasibility_freeze_request_for_testing_v1(
            duplicate, **kwargs
        )
    boolean_schema = freeze_text.replace('"schema_version":1', '"schema_version":true', 1)
    with pytest.raises(NestedOpeningFeasibilityExecutionV1Error):
        parse_nested_opening_feasibility_freeze_request_for_testing_v1(
            boolean_schema, **kwargs
        )
    value = json.loads(freeze_text)
    reordered = _canonical({"freeze_request_digest": value.pop("freeze_request_digest"), **value})
    with pytest.raises(NestedOpeningFeasibilityExecutionV1Error, match="canonical"):
        parse_nested_opening_feasibility_freeze_request_for_testing_v1(
            reordered, **kwargs
        )
    wrong_sha = dict(kwargs)
    wrong_sha["expected_nested_plan_bytes_sha256"] = _sha("wrong-nested-bytes")
    with pytest.raises(NestedOpeningFeasibilityExecutionV1Error, match="SHA-256"):
        parse_nested_opening_feasibility_freeze_request_for_testing_v1(
            freeze_text, **wrong_sha
        )


def test_python_identity_binds_exact_executable_mode_and_refuses_unsafe_modes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    executable = tmp_path / "python-identity-fixture"
    executable.write_bytes(b"engineering-interpreter-identity-fixture")
    monkeypatch.setattr(sys, "executable", str(executable))

    executable.chmod(0o775)
    identity = execution_module._python_identity()
    assert identity["executable_mode"] == "0775"
    assert identity["executable_sha256"] == _sha_bytes(executable.read_bytes())
    assert identity["executable_byte_count"] == executable.stat().st_size

    executable.chmod(0o777)
    with pytest.raises(NestedOpeningFeasibilityExecutionV1Error, match="world-writable"):
        execution_module._python_identity()

    executable.chmod(0o644)
    with pytest.raises(NestedOpeningFeasibilityExecutionV1Error, match="execute bit"):
        execution_module._python_identity()


def test_production_prepare_only_persists_exact_outcome_free_plan_pair(
    tmp_path: Path,
) -> None:
    source_archive = tmp_path / "sealed-source-archive.bin"
    constraints = tmp_path / "sealed-constraints.txt"
    environment_lock = tmp_path / "sealed-environment.lock"
    source_archive.write_bytes(b"opaque-source-archive-byte-pin-v1")
    constraints.write_text("example-dependency==1\n", encoding="utf-8")
    environment_lock.write_text("opaque-environment-lock-v1\n", encoding="utf-8")
    for path in (source_archive, constraints, environment_lock):
        path.chmod(0o400)
    prepared = prepare_production_nested_opening_feasibility_v1(
        _sha("production-upstream-census-seed"),
        _sha("production-independent-nested-generator-seed"),
        tmp_path / "prepared-production",
        runner_source_path=canonical_nested_opening_feasibility_runner_path_v1(),
        source_archive_path=source_archive,
        constraints_path=constraints,
        environment_lock_path=environment_lock,
    )
    assert prepared.freeze_request.plan_class == "production_9x16x2_nested_k32_floor8"
    assert prepared.upstream_census_plan_digest != prepared.nested_plan_digest
    assert prepared.upstream_census_plan_bytes_sha256 != prepared.nested_plan_bytes_sha256
    assert len(prepared.upstream_census_plan.attempts) == 144
    assert len(prepared.nested_plan.attempts) == 144
    identities = tuple(
        member.composed_rule_id
        for attempt in prepared.upstream_census_plan.attempts
        for member in attempt.members
    )
    assert len(identities) == len(set(identities)) == 288
    assert {item.name for item in prepared.output_root.iterdir()} == {
        UPSTREAM_CENSUS_PLAN_FILENAME,
        NESTED_PLAN_FILENAME,
        FREEZE_REQUEST_FILENAME,
    }
    assert all(
        stat.S_IMODE(item.stat().st_mode) == 0o400
        for item in prepared.output_root.iterdir()
    )
    request = json.loads(prepared.freeze_request_path.read_text(encoding="ascii"))
    expected_python_mode = f"{stat.S_IMODE(os.stat(os.path.realpath(sys.executable)).st_mode):04o}"
    assert request["execution_code_binding"]["python"]["executable_mode"] == (
        expected_python_mode
    )
    assert request["fixed_budget"]["mirror_attempt_count"] == 144
    assert request["fixed_budget"]["planned_member_construction_count"] == 288
    assert request["fixed_budget"]["planned_opening_assessment_count"] == 1_152
    assert request["fixed_budget"]["candidate_window_k"] == 32
    assert request["fixed_budget"]["supported_version_space_floor"] == 8
    assert request["fixed_budget"]["no_early_stop"] is True
    assert request["fixed_budget"]["no_identity_replacement"] is True
    assert request["environment_input_bindings"]["verification_boundary"] == {
        "source_archive_reconstructability_verified_by_repository": False,
        "constraints_installability_verified_by_repository": False,
        "environment_reconstruction_verified_by_repository": False,
        "reproducibility_or_reconstructability_claimed": False,
    }
    assert request["scientific_claim_boundary"][
        "cpu_nested_opening_construction_feasibility_report_present"
    ] is False
    assert all(
        value is False
        for key, value in request["authorization"].items()
        if key != "scope"
    )
    forbidden = {
        "observed_report_digest",
        "accounting",
        "runtime_evidence",
        "ranking_tables",
        "model_output",
        "execution_receipt",
    }
    assert forbidden.isdisjoint(_all_keys(request))


def test_external_receipt_requires_exact_pin_reference_identity_and_deadline(
    lifecycle: dict[str, Any]
) -> None:
    prepared = lifecycle["prepared"]
    registration = lifecycle["registration"]
    freeze_text = prepared.freeze_request_path.read_text(encoding="ascii")
    common = {
        "freeze_request": prepared.freeze_request,
        "freeze_request_text": freeze_text,
        "expected_bytes_sha256": lifecycle["registration_sha"],
        "expected_registration_reference": registration.registration_reference,
        "expected_execution_uuid": registration.execution_uuid,
        "expected_execution_nonce": registration.execution_nonce,
        "expected_execution_deadline_seconds": registration.execution_deadline_seconds,
    }
    assert (
        parse_external_nested_opening_feasibility_registration_receipt_v1(
            lifecycle["registration_text"], **common
        )
        == registration
    )
    for key, value in (
        ("expected_bytes_sha256", _sha("wrong-registration-bytes")),
        ("expected_registration_reference", "engineering:test:wrong-reference"),
        ("expected_execution_uuid", "99999999-9999-4999-8999-999999999999"),
        ("expected_execution_nonce", _sha("wrong-execution-nonce")),
        ("expected_execution_deadline_seconds", 601),
    ):
        forged = dict(common)
        forged[key] = value
        with pytest.raises(NestedOpeningFeasibilityExecutionV1Error):
            parse_external_nested_opening_feasibility_registration_receipt_v1(
                lifecycle["registration_text"], **forged
            )


def test_controller_preflights_threads_sigchld_signal_mask_and_deadline_before_root(
    lifecycle: dict[str, Any], tmp_path: Path
) -> None:
    prepared = lifecycle["prepared"]
    registration = lifecycle["registration"]
    registration_path = lifecycle["registration_path"]
    registration_sha = lifecycle["registration_sha"]

    thread_root = tmp_path / _EXECUTION_UUID
    release = threading.Event()
    extra = threading.Thread(target=release.wait, name="nested-preflight-test")
    extra.start()
    try:
        with pytest.raises(NestedOpeningFeasibilityExecutionV1Error, match="single-threaded"):
            execute_engineering_nested_opening_feasibility_fixture_for_testing_v1(
                **_execution_kwargs(
                    prepared,
                    registration_path,
                    registration,
                    registration_sha,
                    thread_root,
                )
            )
    finally:
        release.set()
        extra.join(timeout=5)
    assert not thread_root.exists()

    sigchld_root = tmp_path / "sigchld" / _EXECUTION_UUID
    sigchld_root.parent.mkdir(mode=0o700)
    previous_sigchld = signal.getsignal(signal.SIGCHLD)
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)
    try:
        with pytest.raises(NestedOpeningFeasibilityExecutionV1Error, match="SIGCHLD"):
            execute_engineering_nested_opening_feasibility_fixture_for_testing_v1(
                **_execution_kwargs(
                    prepared,
                    registration_path,
                    registration,
                    registration_sha,
                    sigchld_root,
                )
            )
    finally:
        signal.signal(signal.SIGCHLD, previous_sigchld)
    assert not sigchld_root.exists()

    if hasattr(signal, "pthread_sigmask"):
        masked_root = tmp_path / "masked" / _EXECUTION_UUID
        masked_root.parent.mkdir(mode=0o700)
        prior_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
        try:
            with pytest.raises(NestedOpeningFeasibilityExecutionV1Error, match="unblocked"):
                execute_engineering_nested_opening_feasibility_fixture_for_testing_v1(
                    **_execution_kwargs(
                        prepared,
                        registration_path,
                        registration,
                        registration_sha,
                        masked_root,
                    )
                )
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, prior_mask)
        assert not masked_root.exists()

    deadline_root = tmp_path / "deadline" / _EXECUTION_UUID
    deadline_root.parent.mkdir(mode=0o700)
    with pytest.raises(NestedOpeningFeasibilityExecutionV1Error, match="deadline"):
        execute_engineering_nested_opening_feasibility_fixture_for_testing_v1(
            **_execution_kwargs(
                prepared,
                registration_path,
                registration,
                registration_sha,
                deadline_root,
                deadline_seconds=0,
            )
        )
    assert not deadline_root.exists()


def test_controller_rejects_an_occupied_timer_without_changing_it_or_creating_root(
    lifecycle: dict[str, Any], tmp_path: Path
) -> None:
    if not hasattr(signal, "setitimer"):
        pytest.skip("requires POSIX interval timers")
    prepared = lifecycle["prepared"]
    registration = lifecycle["registration"]
    output_root = tmp_path / _EXECUTION_UUID
    old_timer = signal.setitimer(signal.ITIMER_REAL, 300.0)
    try:
        with pytest.raises(NestedOpeningFeasibilityExecutionV1Error, match="unused"):
            execute_engineering_nested_opening_feasibility_fixture_for_testing_v1(
                **_execution_kwargs(
                    prepared,
                    lifecycle["registration_path"],
                    registration,
                    lifecycle["registration_sha"],
                    output_root,
                )
            )
        remaining, interval = signal.getitimer(signal.ITIMER_REAL)
        assert remaining > 0
        assert interval == 0
    finally:
        signal.setitimer(signal.ITIMER_REAL, *old_timer)
    assert not output_root.exists()


def test_signal_handler_and_timer_setup_failures_restore_process_state(
    lifecycle: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepared = lifecycle["prepared"]
    registration = lifecycle["registration"]
    managed = execution_module._managed_controller_signals()
    handlers_before = {item: signal.getsignal(item) for item in managed}
    mask_before = signal.pthread_sigmask(signal.SIG_BLOCK, set())
    real_signal = signal.signal
    calls = 0

    def fail_second_install(signum: int, handler: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected partial handler installation failure")
        return real_signal(signum, handler)

    monkeypatch.setattr(signal, "signal", fail_second_install)
    handler_root = tmp_path / "handler" / _EXECUTION_UUID
    handler_root.parent.mkdir(mode=0o700)
    with pytest.raises(OSError, match="partial handler"):
        execute_engineering_nested_opening_feasibility_fixture_for_testing_v1(
            **_execution_kwargs(
                prepared,
                lifecycle["registration_path"],
                registration,
                lifecycle["registration_sha"],
                handler_root,
            )
        )
    monkeypatch.setattr(signal, "signal", real_signal)
    assert {item: signal.getsignal(item) for item in managed} == handlers_before
    assert signal.pthread_sigmask(signal.SIG_BLOCK, set()) == mask_before
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
    assert not handler_root.exists()

    real_setitimer = signal.setitimer

    def fail_nonzero_arm(which: int, seconds: float, interval: float = 0.0) -> Any:
        if seconds > 0:
            raise OSError("injected timer arm failure")
        return real_setitimer(which, seconds, interval)

    monkeypatch.setattr(signal, "setitimer", fail_nonzero_arm)
    timer_root = tmp_path / "timer" / _EXECUTION_UUID
    timer_root.parent.mkdir(mode=0o700)
    with pytest.raises(OSError, match="timer arm"):
        execute_engineering_nested_opening_feasibility_fixture_for_testing_v1(
            **_execution_kwargs(
                prepared,
                lifecycle["registration_path"],
                registration,
                lifecycle["registration_sha"],
                timer_root,
            )
        )
    monkeypatch.setattr(signal, "setitimer", real_setitimer)
    assert {item: signal.getsignal(item) for item in managed} == handlers_before
    assert signal.pthread_sigmask(signal.SIG_BLOCK, set()) == mask_before
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
    assert not timer_root.exists()


def test_root_capability_is_published_before_setup_unmask_failure(
    lifecycle: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepared = lifecycle["prepared"]
    registration = lifecycle["registration"]
    output_root = tmp_path / _EXECUTION_UUID
    real_sigmask = signal.pthread_sigmask
    injected = False

    def fail_first_restore(how: int, mask: Any) -> Any:
        nonlocal injected
        if how == signal.SIG_SETMASK and not injected:
            injected = True
            raise NestedOpeningFeasibilityExecutionV1Error(
                "injected setup-unmask delivery"
            )
        return real_sigmask(how, mask)

    monkeypatch.setattr(signal, "pthread_sigmask", fail_first_restore)
    with pytest.raises(NestedOpeningFeasibilityExecutionV1Error, match="setup-unmask"):
        execute_engineering_nested_opening_feasibility_fixture_for_testing_v1(
            **_execution_kwargs(
                prepared,
                lifecycle["registration_path"],
                registration,
                lifecycle["registration_sha"],
                output_root,
            )
        )
    assert injected
    assert (output_root / FAILURE_RECEIPT_FILENAME).is_file()
    assert not (output_root / EXECUTION_RECEIPT_FILENAME).exists()
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_monotonic_deadline_alarm_stamps_failure_and_restores_process_state(
    lifecycle: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepared = lifecycle["prepared"]
    execution_uuid = "88888888-8888-4888-8888-888888888888"
    registration_path = tmp_path / "deadline-registration.json"
    registration, _, registration_sha = _registration(
        prepared,
        registration_path,
        execution_uuid=execution_uuid,
        execution_nonce=_sha("deadline-alarm-nonce"),
        registration_reference="engineering:test:deadline-alarm",
        deadline_seconds=1,
    )
    output_root = tmp_path / execution_uuid
    managed = execution_module._managed_controller_signals()
    handlers_before = {item: signal.getsignal(item) for item in managed}
    mask_before = signal.pthread_sigmask(signal.SIG_BLOCK, set())

    def block_until_alarm(*_args: Any, **_kwargs: Any) -> Any:
        time.sleep(5)
        raise AssertionError("registered alarm did not interrupt blocked stage")

    monkeypatch.setattr(execution_module, "_controller_replay_held_chain", block_until_alarm)
    started = time.monotonic()
    with pytest.raises(NestedOpeningFeasibilityExecutionV1Error, match="SIGALRM"):
        execute_engineering_nested_opening_feasibility_fixture_for_testing_v1(
            **_execution_kwargs(
                prepared,
                registration_path,
                registration,
                registration_sha,
                output_root,
                deadline_seconds=1,
            )
        )
    assert time.monotonic() - started < 4
    assert (output_root / FAILURE_RECEIPT_FILENAME).is_file()
    assert not (output_root / EXECUTION_RECEIPT_FILENAME).exists()
    failure = json.loads(
        (output_root / FAILURE_RECEIPT_FILENAME).read_text(encoding="ascii")
    )
    assert failure["failed_stage"] == "held_input_preflight"
    assert failure["completion_receipt_present"] is False
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
    assert {item: signal.getsignal(item) for item in managed} == handlers_before
    assert signal.pthread_sigmask(signal.SIG_BLOCK, set()) == mask_before


def test_complete_root_verifier_preflights_threads_and_sigchld_before_acquisition(
    lifecycle: dict[str, Any]
) -> None:
    root = lifecycle["executed"].output_root
    inventory_before = tuple(sorted(item.name for item in root.iterdir()))
    release = threading.Event()
    extra = threading.Thread(target=release.wait, name="nested-verifier-preflight")
    extra.start()
    try:
        with pytest.raises(NestedOpeningFeasibilityExecutionV1Error, match="single-threaded"):
            verify_engineering_nested_opening_feasibility_execution_root_for_testing_v1(
                **_verification_kwargs(lifecycle, root)
            )
    finally:
        release.set()
        extra.join(timeout=5)
    assert tuple(sorted(item.name for item in root.iterdir())) == inventory_before

    previous_sigchld = signal.getsignal(signal.SIGCHLD)
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)
    try:
        with pytest.raises(NestedOpeningFeasibilityExecutionV1Error, match="SIGCHLD"):
            verify_engineering_nested_opening_feasibility_execution_root_for_testing_v1(
                **_verification_kwargs(lifecycle, root)
            )
    finally:
        signal.signal(signal.SIGCHLD, previous_sigchld)
    assert tuple(sorted(item.name for item in root.iterdir())) == inventory_before


def test_watchdog_spawn_boundary_rechecks_thread_and_sigchld_after_setup(
    lifecycle: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = lifecycle["executed"].output_root
    real_duplicate = execution_module._duplicate_worker_semantic_fds
    release = threading.Event()
    spawned: list[threading.Thread] = []

    def duplicate_then_add_thread(*args: Any, **kwargs: Any) -> Any:
        result = real_duplicate(*args, **kwargs)
        thread = threading.Thread(target=release.wait, name="late-verifier-reaper")
        thread.start()
        spawned.append(thread)
        return result

    monkeypatch.setattr(
        execution_module, "_duplicate_worker_semantic_fds", duplicate_then_add_thread
    )
    try:
        with pytest.raises(NestedOpeningFeasibilityExecutionV1Error, match="single-threaded"):
            verify_engineering_nested_opening_feasibility_execution_root_for_testing_v1(
                **_verification_kwargs(lifecycle, root)
            )
    finally:
        release.set()
        for thread in spawned:
            thread.join(timeout=5)
    monkeypatch.setattr(
        execution_module, "_duplicate_worker_semantic_fds", real_duplicate
    )

    previous_sigchld = signal.getsignal(signal.SIGCHLD)

    def duplicate_then_ignore_sigchld(*args: Any, **kwargs: Any) -> Any:
        result = real_duplicate(*args, **kwargs)
        signal.signal(signal.SIGCHLD, signal.SIG_IGN)
        return result

    monkeypatch.setattr(
        execution_module,
        "_duplicate_worker_semantic_fds",
        duplicate_then_ignore_sigchld,
    )
    try:
        with pytest.raises(NestedOpeningFeasibilityExecutionV1Error, match="SIGCHLD"):
            verify_engineering_nested_opening_feasibility_execution_root_for_testing_v1(
                **_verification_kwargs(lifecycle, root)
            )
    finally:
        signal.signal(signal.SIGCHLD, previous_sigchld)


def test_held_input_paths_modes_and_semantic_inode_aliases_fail_closed(
    lifecycle: dict[str, Any], tmp_path: Path
) -> None:
    prepared = lifecycle["prepared"]
    registration = lifecycle["registration"]
    registration_sha = lifecycle["registration_sha"]
    symlink = tmp_path / "registration-link.json"
    symlink.symlink_to(lifecycle["registration_path"])
    symlink_root = tmp_path / "symlink-root" / _EXECUTION_UUID
    symlink_root.parent.mkdir(mode=0o700)
    with pytest.raises(NestedOpeningFeasibilityExecutionV1Error):
        execute_engineering_nested_opening_feasibility_fixture_for_testing_v1(
            **_execution_kwargs(
                prepared,
                symlink,
                registration,
                registration_sha,
                symlink_root,
            )
        )
    assert (symlink_root / FAILURE_RECEIPT_FILENAME).is_file()
    assert not (symlink_root / EXECUTION_RECEIPT_FILENAME).exists()

    writable = tmp_path / "writable-registration.json"
    writable.write_bytes(lifecycle["registration_text"].encode("ascii"))
    writable.chmod(0o600)
    mode_root = tmp_path / "mode-root" / _EXECUTION_UUID
    mode_root.parent.mkdir(mode=0o700)
    with pytest.raises(NestedOpeningFeasibilityExecutionV1Error, match="mode"):
        execute_engineering_nested_opening_feasibility_fixture_for_testing_v1(
            **_execution_kwargs(
                prepared,
                writable,
                registration,
                registration_sha,
                mode_root,
            )
        )
    assert (mode_root / FAILURE_RECEIPT_FILENAME).is_file()

    sealed = tmp_path / "sealed-input"
    sealed.write_bytes(b"held bytes")
    sealed.chmod(0o400)
    first = execution_module._HeldOrdinaryFileV1.open(
        sealed, label="first", expected_modes=frozenset({0o400})
    )
    second = execution_module._HeldOrdinaryFileV1.open(
        sealed, label="second", expected_modes=frozenset({0o400})
    )
    try:
        with pytest.raises(NestedOpeningFeasibilityExecutionV1Error, match="alias one inode"):
            execution_module._require_distinct_held_inodes(
                {"first": first, "second": second}
            )
    finally:
        first.close()
        second.close()


def test_component_symlinks_hardlinks_and_held_basename_replacement_are_rejected(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir(mode=0o700)
    sealed = real_parent / "sealed.json"
    sealed.write_bytes(b"{}\n")
    sealed.chmod(0o400)
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(NestedOpeningFeasibilityExecutionV1Error):
        execution_module._HeldOrdinaryFileV1.open(
            linked_parent / sealed.name,
            label="symlinked-parent",
            expected_modes=frozenset({0o400}),
        )

    hardlink = real_parent / "sealed-hardlink.json"
    os.link(sealed, hardlink)
    with pytest.raises(NestedOpeningFeasibilityExecutionV1Error, match="hard link"):
        execution_module._HeldOrdinaryFileV1.open(
            sealed,
            label="hardlinked-input",
            expected_modes=frozenset({0o400}),
        )
    hardlink.unlink()

    held = execution_module._HeldOrdinaryFileV1.open(
        sealed,
        label="replacement-input",
        expected_modes=frozenset({0o400}),
    )
    moved = real_parent / "held-original.json"
    sealed.rename(moved)
    sealed.write_bytes(b"{}\n")
    sealed.chmod(0o400)
    try:
        with pytest.raises(NestedOpeningFeasibilityExecutionV1Error, match="changed"):
            held.replay_external_basename()
    finally:
        held.close()

    output_parent_link = tmp_path / "output-parent-link"
    output_parent_link.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(NestedOpeningFeasibilityExecutionV1Error):
        execution_module._secure_absent_root(output_parent_link / "new-root")
    assert not (real_parent / "new-root").exists()


def test_held_execution_root_snapshot_rejects_root_and_artifact_namespace_replacement(
    tmp_path: Path,
) -> None:
    def acquire(root: Path) -> Any:
        root.mkdir(mode=0o700)
        payloads = {"artifact.json": b"{\"exact\":true}\n"}
        artifact = root / "artifact.json"
        artifact.write_bytes(payloads["artifact.json"])
        artifact.chmod(0o400)
        directory_fd, _ = execution_module._open_directory_chain_nofollow(root)
        return execution_module._HeldExecutionRootSnapshotV1.acquire(
            root, directory_fd, payloads
        )

    root = tmp_path / "held-root"
    snapshot = acquire(root)
    moved = tmp_path / "held-root-original"
    root.rename(moved)
    root.mkdir(mode=0o700)
    replacement = root / "artifact.json"
    replacement.write_bytes(b"{\"exact\":true}\n")
    replacement.chmod(0o400)
    try:
        with pytest.raises(NestedOpeningFeasibilityExecutionV1Error):
            snapshot.replay()
    finally:
        snapshot.close()
        os.close(snapshot.directory_fd)

    root_two = tmp_path / "held-artifact-root"
    snapshot_two = acquire(root_two)
    artifact_two = root_two / "artifact.json"
    artifact_two.unlink()
    artifact_two.write_bytes(b"{\"exact\":true}\n")
    artifact_two.chmod(0o400)
    try:
        with pytest.raises(
            NestedOpeningFeasibilityExecutionV1Error, match=r"changed|replaced"
        ):
            snapshot_two.replay()
    finally:
        snapshot_two.close()
        os.close(snapshot_two.directory_fd)

    root_three = tmp_path / "held-artifact-symlink-root"
    snapshot_three = acquire(root_three)
    artifact_three = root_three / "artifact.json"
    symlink_target = tmp_path / "lookalike-artifact.json"
    symlink_target.write_bytes(b"{\"exact\":true}\n")
    symlink_target.chmod(0o400)
    artifact_three.unlink()
    artifact_three.symlink_to(symlink_target)
    try:
        with pytest.raises(NestedOpeningFeasibilityExecutionV1Error):
            snapshot_three.replay()
    finally:
        snapshot_three.close()
        os.close(snapshot_three.directory_fd)


def _replace_worker_option(argv: list[str], option: str, value: str) -> None:
    index = argv.index(option)
    argv[index + 1] = value


def test_watchdog_fd_grammar_sets_all_inherited_capabilities_cloexec_and_rejects_forgery(
    lifecycle: dict[str, Any]
) -> None:
    watchdog = json.loads(
        (
            lifecycle["executed"].output_root
            / BUILDER_GUARDIAN_TERMINAL_FILENAME
        ).read_text(encoding="ascii")
    )
    template = list(watchdog["worker_process"]["argv"])

    def capabilities(
        *, alias_held: bool = False
    ) -> tuple[Any, list[int], list[str], Namespace]:
        channels = execution_module._ParentDeathPipe.create()
        held_labels = tuple(
            json.loads(template[template.index("--held-input-fds-json") + 1])
        )
        dummy: list[int] = []
        for _ in range(len(held_labels) + 1):
            read_fd, write_fd = os.pipe()
            os.close(write_fd)
            dummy.append(read_fd)
        held_values = dummy[: len(held_labels)]
        if alias_held:
            held_values[1] = held_values[0]
        held_map = dict(zip(held_labels, held_values, strict=True))
        semantic_fd = dummy[-1]
        worker_argv = list(template)
        _replace_worker_option(
            worker_argv,
            "--held-input-fds-json",
            json.dumps(held_map, separators=(",", ":")),
        )
        _replace_worker_option(
            worker_argv, "--parent-guard-fd", str(channels.worker_read_fd)
        )
        _replace_worker_option(
            worker_argv,
            "--guardian-ready-write-fd",
            str(channels.guardian_ready_write_fd),
        )
        _replace_worker_option(
            worker_argv,
            "--worker-lifetime-write-fd",
            str(channels.worker_lifetime_write_fd),
        )
        _replace_worker_option(worker_argv, "--output-root-fd", str(semantic_fd))
        forward = [*held_values, semantic_fd]
        args = Namespace(
            parent_death_fd=channels.read_fd,
            worker_guard_read_fd=channels.worker_read_fd,
            worker_guard_write_fd=channels.worker_write_fd,
            worker_pid_report_fd=channels.pid_write_fd,
            guardian_ready_read_fd=channels.guardian_ready_read_fd,
            guardian_ready_write_fd=channels.guardian_ready_write_fd,
            worker_lifetime_read_fd=channels.worker_lifetime_read_fd,
            worker_lifetime_write_fd=channels.worker_lifetime_write_fd,
            deadline_monotonic_ns=time.monotonic_ns() + 10_000_000_000,
            forward_fds_json=json.dumps(forward, separators=(",", ":")),
            worker_argv_json=json.dumps(worker_argv, separators=(",", ":")),
        )
        return channels, dummy, worker_argv, args

    channels, dummy, worker_argv, args = capabilities()
    all_inherited = [
        channels.read_fd,
        channels.worker_read_fd,
        channels.worker_write_fd,
        channels.pid_write_fd,
        channels.guardian_ready_read_fd,
        channels.guardian_ready_write_fd,
        channels.worker_lifetime_read_fd,
        channels.worker_lifetime_write_fd,
        *dummy,
    ]
    for descriptor in all_inherited:
        os.set_inheritable(descriptor, True)
    try:
        execution_module._validate_watchdog_worker_launch_argv(
            worker_argv,
            worker_guard_read_fd=channels.worker_read_fd,
            ready_write_fd=channels.guardian_ready_write_fd,
            lifetime_write_fd=channels.worker_lifetime_write_fd,
            forward_fds=dummy,
        )
        execution_module._acquire_watchdog_worker_inputs(args)
        assert all(not os.get_inheritable(descriptor) for descriptor in all_inherited)
    finally:
        channels.close()
        for descriptor in dummy:
            with suppress(OSError):
                os.close(descriptor)

    forged_channels, forged_dummy, _, forged_args = capabilities(alias_held=True)
    try:
        with pytest.raises(
            NestedOpeningFeasibilityExecutionV1Error,
            match=r"canonical and unique|pairwise distinct",
        ):
            execution_module._acquire_watchdog_worker_inputs(forged_args)
    finally:
        forged_channels.close()
        for descriptor in forged_dummy:
            with suppress(OSError):
                os.close(descriptor)

    duplicate_option = list(template)
    duplicate_option.extend(("--parent-guard-fd", "0"))
    with pytest.raises(NestedOpeningFeasibilityExecutionV1Error, match="ordered action grammar"):
        execution_module._validate_watchdog_worker_launch_argv(
            duplicate_option,
            worker_guard_read_fd=0,
            ready_write_fd=1,
            lifetime_write_fd=2,
            forward_fds=(3, 4, 5, 6, 7),
        )


def test_failure_root_is_terminal_nonauthorizing_and_not_reusable(
    lifecycle: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepared = lifecycle["prepared"]
    execution_uuid = "33333333-3333-4333-8333-333333333333"
    registration_path = tmp_path / "failure-registration.json"
    registration, _, registration_sha = _registration(
        prepared,
        registration_path,
        execution_uuid=execution_uuid,
        execution_nonce=_sha("failure-root-nonce"),
        registration_reference="engineering:test:nested-failure",
    )
    output_root = tmp_path / execution_uuid

    def injected_failure(*_args: Any, **_kwargs: Any) -> Any:
        raise NestedOpeningFeasibilityExecutionV1Error("injected held-chain failure")

    monkeypatch.setattr(
        execution_module, "_controller_replay_held_chain", injected_failure
    )
    with pytest.raises(NestedOpeningFeasibilityExecutionV1Error, match="injected"):
        execute_engineering_nested_opening_feasibility_fixture_for_testing_v1(
            **_execution_kwargs(
                prepared,
                registration_path,
                registration,
                registration_sha,
                output_root,
            )
        )
    assert {item.name for item in output_root.iterdir()} == {FAILURE_RECEIPT_FILENAME}
    failure_path = output_root / FAILURE_RECEIPT_FILENAME
    assert stat.S_IMODE(failure_path.stat().st_mode) == 0o400
    failure = json.loads(failure_path.read_text(encoding="ascii"))
    assert failure["status"] == "failed_incomplete_root_do_not_reuse"
    assert failure["completion_receipt_present"] is False
    assert failure["retry_within_execution_uuid_allowed"] is False
    assert failure["execution_scope"] == {
        "engineering_fixture_only": True,
        "production_evidence": False,
        "study_evidence": False,
    }
    assert all(value is False for value in failure["reconstruction_claim_boundary"].values())
    assert all(value is False for key, value in failure["authorization"].items() if key != "scope")

    with pytest.raises(NestedOpeningFeasibilityExecutionV1Error, match="already exists"):
        execute_engineering_nested_opening_feasibility_fixture_for_testing_v1(
            **_execution_kwargs(
                prepared,
                registration_path,
                registration,
                registration_sha,
                output_root,
            )
        )


def test_post_mkdir_recovery_and_post_started_failure_leave_nonreusable_roots(
    lifecycle: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    real_secure_root = execution_module._secure_absent_root
    recovery_uuid = "44444444-4444-4444-8444-444444444444"
    recovery_root = tmp_path / "mkdir-recovery" / recovery_uuid
    recovery_root.parent.mkdir(mode=0o700)

    def fail_after_mkdir(path: Path) -> tuple[Path, int]:
        _root, descriptor = real_secure_root(path)
        os.close(descriptor)
        raise OSError("injected post-mkdir pre-return failure")

    monkeypatch.setattr(execution_module, "_secure_absent_root", fail_after_mkdir)
    with pytest.raises(OSError, match="post-mkdir"):
        execution_module._secure_absent_execution_root_with_failure_fallback(
            recovery_root, execution_uuid=recovery_uuid
        )
    assert {item.name for item in recovery_root.iterdir()} == {FAILURE_RECEIPT_FILENAME}
    recovered_failure = json.loads(
        (recovery_root / FAILURE_RECEIPT_FILENAME).read_text(encoding="ascii")
    )
    assert recovered_failure["failed_stage"] == "exclusive_root_creation_after_mkdir"
    assert recovered_failure["completion_receipt_present"] is False
    monkeypatch.setattr(execution_module, "_secure_absent_root", real_secure_root)

    prepared = lifecycle["prepared"]
    started_uuid = "55555555-5555-4555-8555-555555555555"
    registration_path = tmp_path / "started-failure-registration.json"
    registration, _, registration_sha = _registration(
        prepared,
        registration_path,
        execution_uuid=started_uuid,
        execution_nonce=_sha("post-started-failure-nonce"),
        registration_reference="engineering:test:post-started-failure",
    )
    started_root = tmp_path / "post-started" / started_uuid
    started_root.parent.mkdir(mode=0o700)

    def fail_watchdog(
        _watchdog_argv: Any,
        *,
        channels: Any,
        forward_fds: Any,
        **_kwargs: Any,
    ) -> Any:
        for descriptor in forward_fds:
            with suppress(OSError):
                os.close(descriptor)
        channels.close()
        raise NestedOpeningFeasibilityExecutionV1Error(
            "injected post-started watchdog failure"
        )

    monkeypatch.setattr(execution_module, "_run_watchdog", fail_watchdog)
    with pytest.raises(NestedOpeningFeasibilityExecutionV1Error, match="post-started"):
        execute_engineering_nested_opening_feasibility_fixture_for_testing_v1(
            **_execution_kwargs(
                prepared,
                registration_path,
                registration,
                registration_sha,
                started_root,
            )
        )
    assert {item.name for item in started_root.iterdir()} == {
        STARTED_RECEIPT_FILENAME,
        FAILURE_RECEIPT_FILENAME,
    }
    failure = json.loads(
        (started_root / FAILURE_RECEIPT_FILENAME).read_text(encoding="ascii")
    )
    assert failure["failed_stage"] == "builder"
    assert failure["started_receipt_basename_present"] is True
    assert failure["started_receipt_exact_and_canonical"] is True
    assert failure["completion_receipt_present"] is False


def test_partial_completion_is_removed_and_replaced_by_failure_receipt(
    lifecycle: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepared = lifecycle["prepared"]
    execution_uuid = "66666666-6666-4666-8666-666666666666"
    registration_path = tmp_path / "partial-completion-registration.json"
    registration, _, registration_sha = _registration(
        prepared,
        registration_path,
        execution_uuid=execution_uuid,
        execution_nonce=_sha("partial-completion-nonce"),
        registration_reference="engineering:test:partial-completion",
    )
    output_root = tmp_path / execution_uuid
    real_write = execution_module._write_exclusive_at
    injected = False

    def partial_terminal_write(
        directory_fd: int, name: str, payload: bytes
    ) -> None:
        nonlocal injected
        if name == EXECUTION_RECEIPT_FILENAME and not injected:
            injected = True
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
            descriptor = os.open(name, flags, 0o400, dir_fd=directory_fd)
            try:
                os.fchmod(descriptor, 0o400)
                os.write(descriptor, b"{\n")
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.fsync(directory_fd)
            raise OSError("injected partial completion write")
        real_write(directory_fd, name, payload)

    monkeypatch.setattr(execution_module, "_write_exclusive_at", partial_terminal_write)
    with pytest.raises(OSError, match="partial completion"):
        execute_engineering_nested_opening_feasibility_fixture_for_testing_v1(
            **_execution_kwargs(
                prepared,
                registration_path,
                registration,
                registration_sha,
                output_root,
            )
        )
    assert injected
    assert not (output_root / EXECUTION_RECEIPT_FILENAME).exists()
    assert (output_root / FAILURE_RECEIPT_FILENAME).is_file()
    assert (output_root / REPORT_FILENAME).is_file()
    failure = json.loads(
        (output_root / FAILURE_RECEIPT_FILENAME).read_text(encoding="ascii")
    )
    assert failure["failed_stage"] == "write_completion_receipt"
    assert failure["completion_receipt_present"] is False
    assert failure["retry_within_execution_uuid_allowed"] is False


def test_canonical_completion_remains_terminal_if_postcommit_restoration_fails(
    lifecycle: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepared = lifecycle["prepared"]
    execution_uuid = "77777777-7777-4777-8777-777777777777"
    registration_path = tmp_path / "completion-finality-registration.json"
    registration, _, registration_sha = _registration(
        prepared,
        registration_path,
        execution_uuid=execution_uuid,
        execution_nonce=_sha("completion-finality-nonce"),
        registration_reference="engineering:test:completion-finality",
    )
    output_root = tmp_path / execution_uuid
    real_block = execution_module._block_terminal_signals_and_disarm_alarm
    terminal_mask_observed = False

    def observe_terminal_mask(*, timer_armed: bool) -> Any:
        nonlocal terminal_mask_observed
        previous = real_block(timer_armed=timer_armed)
        current = signal.pthread_sigmask(signal.SIG_BLOCK, set())
        terminal_mask_observed = set(
            execution_module._managed_controller_signals()
        ).issubset(current)
        return previous

    real_restore = execution_module._restore_controller_signal_handlers

    def restore_then_fail(previous: Any) -> None:
        real_restore(previous)
        raise OSError("injected postcommit handler restoration failure")

    monkeypatch.setattr(
        execution_module,
        "_block_terminal_signals_and_disarm_alarm",
        observe_terminal_mask,
    )
    monkeypatch.setattr(
        execution_module, "_restore_controller_signal_handlers", restore_then_fail
    )
    with pytest.raises(OSError, match="postcommit handler restoration"):
        execute_engineering_nested_opening_feasibility_fixture_for_testing_v1(
            **_execution_kwargs(
                prepared,
                registration_path,
                registration,
                registration_sha,
                output_root,
            )
        )
    assert terminal_mask_observed
    assert (output_root / EXECUTION_RECEIPT_FILENAME).is_file()
    assert not (output_root / FAILURE_RECEIPT_FILENAME).exists()
    receipt_text = (output_root / EXECUTION_RECEIPT_FILENAME).read_text(encoding="ascii")
    assert _canonical(json.loads(receipt_text)) == receipt_text
    assert stat.S_IMODE((output_root / EXECUTION_RECEIPT_FILENAME).stat().st_mode) == 0o400


def _copy_complete_root(source: Path, parent: Path) -> Path:
    parent.mkdir(mode=0o700)
    destination = parent / source.name
    shutil.copytree(source, destination, copy_function=shutil.copy2)
    return destination


def test_complete_root_verifier_rejects_inventory_report_and_registration_crossbinding_tamper(
    lifecycle: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    original = lifecycle["executed"].output_root

    extra_root = _copy_complete_root(original, tmp_path / "extra")
    extra = extra_root / "unexpected.json"
    extra.write_text("{}\n", encoding="ascii")
    extra.chmod(0o400)
    with pytest.raises(NestedOpeningFeasibilityExecutionV1Error, match="exactly seven"):
        verify_engineering_nested_opening_feasibility_execution_root_for_testing_v1(
            **_verification_kwargs(lifecycle, extra_root)
        )

    report_root = _copy_complete_root(original, tmp_path / "report")
    report_path = report_root / REPORT_FILENAME
    report_path.chmod(0o600)
    report_path.write_bytes(report_path.read_bytes() + b" ")
    report_path.chmod(0o400)
    with pytest.raises(NestedOpeningFeasibilityExecutionV1Error):
        verify_engineering_nested_opening_feasibility_execution_root_for_testing_v1(
            **_verification_kwargs(lifecycle, report_root)
        )

    cross_root = _copy_complete_root(original, tmp_path / "cross")
    receipt_path = cross_root / EXECUTION_RECEIPT_FILENAME
    receipt = json.loads(receipt_path.read_text(encoding="ascii"))
    receipt["external_registration_validation"][
        "registered_execution_deadline_seconds"
    ] += 1
    unsigned = {
        key: value for key, value in receipt.items() if key != "execution_receipt_digest"
    }
    receipt["execution_receipt_digest"] = execution_module._digest(
        unsigned, domain=execution_module._EXECUTION_RECEIPT_DOMAIN
    )
    receipt_text = _canonical(receipt)
    receipt_path.chmod(0o600)
    receipt_path.write_text(receipt_text, encoding="ascii")
    receipt_path.chmod(0o400)
    cross_kwargs = _verification_kwargs(lifecycle, cross_root)
    cross_kwargs["expected_execution_receipt_sha256"] = _sha_bytes(
        receipt_text.encode("ascii")
    )
    cross_kwargs["expected_execution_receipt_digest"] = receipt[
        "execution_receipt_digest"
    ]
    with pytest.raises(NestedOpeningFeasibilityExecutionV1Error, match="registration"):
        verify_engineering_nested_opening_feasibility_execution_root_for_testing_v1(
            **cross_kwargs
        )

    deadline_kwargs = _verification_kwargs(lifecycle, original)
    deadline_kwargs["expected_execution_deadline_seconds"] = 601
    with pytest.raises(NestedOpeningFeasibilityExecutionV1Error, match="registered execution"):
        verify_engineering_nested_opening_feasibility_execution_root_for_testing_v1(
            **deadline_kwargs
        )

    mode_root = _copy_complete_root(original, tmp_path / "root-mode")
    mode_root.chmod(0o750)
    with pytest.raises(NestedOpeningFeasibilityExecutionV1Error):
        verify_engineering_nested_opening_feasibility_execution_root_for_testing_v1(
            **_verification_kwargs(lifecycle, mode_root)
        )

    target_root = _copy_complete_root(original, tmp_path / "root-target")
    symlink_parent = tmp_path / "root-symlink"
    symlink_parent.mkdir(mode=0o700)
    symlink_root = symlink_parent / original.name
    symlink_root.symlink_to(target_root, target_is_directory=True)
    with pytest.raises(NestedOpeningFeasibilityExecutionV1Error):
        verify_engineering_nested_opening_feasibility_execution_root_for_testing_v1(
            **_verification_kwargs(lifecycle, symlink_root)
        )

    alias_root = _copy_complete_root(original, tmp_path / "artifact-alias")
    alias_target = alias_root / VERIFIER_TERMINAL_FILENAME
    alias_target.unlink()
    os.link(alias_root / BUILDER_TERMINAL_FILENAME, alias_target)
    with pytest.raises(NestedOpeningFeasibilityExecutionV1Error, match="unsafe metadata"):
        verify_engineering_nested_opening_feasibility_execution_root_for_testing_v1(
            **_verification_kwargs(lifecycle, alias_root)
        )

    original_code_binding = execution_module._execution_code_binding

    def drifted_code_binding(path: Path) -> dict[str, Any]:
        value = json.loads(json.dumps(original_code_binding(path)))
        value["execution_module"]["sha256"] = _sha("post-freeze-code-drift")
        return value

    monkeypatch.setattr(execution_module, "_execution_code_binding", drifted_code_binding)
    with pytest.raises(NestedOpeningFeasibilityExecutionV1Error):
        verify_engineering_nested_opening_feasibility_execution_root_for_testing_v1(
            **_verification_kwargs(lifecycle, original)
        )


@pytest.mark.skipif(
    sys.platform != "darwin" or hasattr(os, "waitid"),
    reason="exercises the Darwin libc waitid and zombie-group EPERM path",
)
def test_darwin_libc_waitid_keeps_child_owned_until_group_cleanup_and_reap() -> None:
    child = subprocess.Popen(
        [sys.executable, "-S", "-c", "pass"],
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    deadline = time.monotonic() + 10
    try:
        while not execution_module._child_exited_without_reap(child.pid):
            if time.monotonic() >= deadline:
                pytest.fail("child did not become waitable without reap")
            time.sleep(0.01)
        stdout, stderr = execution_module._cleanup_worker_group_before_direct_reap(child)
        assert stdout == b""
        assert stderr == b""
        assert child.returncode == 0
    finally:
        if child.returncode is None:
            child.kill()
            child.wait(timeout=5)


@pytest.mark.skipif(not hasattr(os, "killpg"), reason="requires POSIX process groups")
def test_early_guardian_kills_worker_when_owning_watchdog_dies(
    lifecycle: dict[str, Any]
) -> None:
    persisted = json.loads(
        (
            lifecycle["executed"].output_root
            / BUILDER_GUARDIAN_TERMINAL_FILENAME
        ).read_text(encoding="ascii")
    )
    worker_template = list(persisted["worker_process"]["argv"])
    channels = execution_module._ParentDeathPipe.create()
    dummy: list[int] = []
    watchdog: subprocess.Popen[bytes] | None = None
    worker_pid: int | None = None
    worker_stopped = False
    try:
        held_labels = tuple(
            json.loads(
                worker_template[worker_template.index("--held-input-fds-json") + 1]
            )
        )
        for _ in range(len(held_labels) + 1):
            read_fd, write_fd = os.pipe()
            os.close(write_fd)
            dummy.append(read_fd)
        worker_argv = list(worker_template)
        held_map = dict(zip(held_labels, dummy[:-1], strict=True))
        _replace_worker_option(
            worker_argv,
            "--held-input-fds-json",
            json.dumps(held_map, separators=(",", ":")),
        )
        _replace_worker_option(
            worker_argv, "--parent-guard-fd", str(channels.worker_read_fd)
        )
        _replace_worker_option(
            worker_argv,
            "--guardian-ready-write-fd",
            str(channels.guardian_ready_write_fd),
        )
        _replace_worker_option(
            worker_argv,
            "--worker-lifetime-write-fd",
            str(channels.worker_lifetime_write_fd),
        )
        _replace_worker_option(worker_argv, "--output-root-fd", str(dummy[-1]))
        deadline_ns = time.monotonic_ns() + 30_000_000_000
        watchdog_argv = execution_module._watchdog_argv(
            channels=channels,
            deadline_monotonic_ns=deadline_ns,
            forward_fds=dummy,
            worker_argv=worker_argv,
        )
        watchdog = subprocess.Popen(
            watchdog_argv,
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
                *dummy,
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
        for descriptor in dummy:
            os.close(descriptor)
        dummy = []
        readable, _, _ = execution_module.select.select(
            [channels.pid_read_fd], [], [], 10
        )
        assert readable
        worker_pid = int(os.read(channels.pid_read_fd, 32).strip())
        assert worker_pid > 0
        os.close(channels.pid_read_fd)
        channels.pid_read_fd = -1
        # The PID report follows the bootstrap guardian-ready byte. Stop the
        # exact still-owned worker group and prove its lifetime writer remains
        # open; this excludes both an ordinary exit and a zombie before the
        # watchdog is killed.
        os.killpg(worker_pid, signal.SIGSTOP)
        worker_stopped = True
        readable_before, _, _ = execution_module.select.select(
            [channels.worker_lifetime_read_fd], [], [], 0
        )
        assert not readable_before
        os.killpg(watchdog.pid, signal.SIGKILL)
        watchdog.wait(timeout=10)
        os.killpg(worker_pid, signal.SIGCONT)
        execution_module._require_worker_lifetime_eof(channels, timeout=10)
        # Clear cleanup authority only after EOF proves the exact held lifetime
        # writer is gone. Until then the earlier stopped/non-EOF observation
        # justifies the finally-path signals even though the watchdog exited.
        worker_stopped = False
    finally:
        # The worker PID/PGID is reported only while its owning watchdog still
        # holds the direct-child identity. Resume first so no failed assertion
        # can strand a stopped worker; then kill only that exact still-owned
        # group before allowing watchdog cleanup/reap to finish.
        if worker_pid is not None and worker_stopped:
            with suppress(ProcessLookupError, PermissionError):
                os.killpg(worker_pid, signal.SIGCONT)
            with suppress(ProcessLookupError, PermissionError):
                os.killpg(worker_pid, signal.SIGKILL)
            worker_stopped = False
        if worker_pid is not None and watchdog is not None and watchdog.poll() is None:
            with suppress(ProcessLookupError, PermissionError):
                os.killpg(worker_pid, signal.SIGKILL)
        if watchdog is not None:
            execution_module._terminate_owned_process_group(watchdog)
        if channels.worker_lifetime_read_fd >= 0:
            with suppress(BaseException):
                execution_module._require_worker_lifetime_eof(channels, timeout=10)
        channels.close()
        for descriptor in dummy:
            with suppress(OSError):
                os.close(descriptor)
