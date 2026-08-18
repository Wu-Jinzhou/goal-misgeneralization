from __future__ import annotations

import ast
import builtins
import dataclasses
import hashlib
import importlib
import json
import os
import socket
import subprocess
import sys
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest

from goalzendo_g01_preprovision import contracts
from goalzendo_g01_preprovision.contracts import (
    PreprovisionError,
    build_policy_template,
    build_unregistered_campaign_intent_candidate,
    validate_policy_template_bytes,
    validate_unregistered_campaign_intent_candidate_bytes,
)

ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = ROOT / "runs" / "goalzendo" / "run_g01_preprovision.py"
SHA = "a" * 64
ELIGIBILITY_UUID = "11111111-1111-4111-8111-111111111111"
EXECUTION_UUID = "22222222-2222-4222-8222-222222222222"
QUALIFICATION_UUID = "33333333-3333-4333-8333-333333333333"


def _canonical(value: Any) -> bytes:
    def plain(child: Any) -> Any:
        if isinstance(child, Mapping):
            return {key: plain(item) for key, item in child.items()}
        if isinstance(child, (tuple, list)):
            return [plain(item) for item in child]
        return child

    return json.dumps(
        plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()


def _candidate_kwargs(priority: int = 0) -> dict[str, Any]:
    dispositions = [
        {
            "profile_id": profile_id,
            "state": "skipped_no_stock" if index % 2 == 0 else "consumed_failure",
            "registrar_record_sha256": chr(ord("b") + index) * 64,
        }
        for index, profile_id in enumerate(("h200x8", "h100-hbm3x8", "h200x4", "h100-hbm3x4")[:priority])
    ]
    return {
        "eligibility_execution_uuid": ELIGIBILITY_UUID,
        "g01_execution_uuid": EXECUTION_UUID,
        "campaign_uuid": EXECUTION_UUID,
        "qualification_uuid": QUALIFICATION_UUID,
        "checkpoint_a_token_file_sha256": SHA,
        "checkpoint_a_token_digest": "b" * 64,
        "checkpoint_a_bridge_source_digest": contracts.CHECKPOINT_A_BRIDGE_SOURCE_DIGEST,
        "checkpoint_a_token_portable_bytes": 1_024,
        "candidate_priority": priority,
        "data_center_id": "US-CA-2",
        "network_volume_id": "volume_AbC123",
        "prior_candidate_dispositions": dispositions,
        "prospective_transaction_source_digest": "c" * 64,
        "price_ceiling_micro_usd_per_gpu_hour": 4_000_000,
        "operator_maximum_wall_seconds": 432_000,
        "operator_maximum_gpu_seconds": 3_456_000,
        "operator_maximum_cost_micro_usd": 3_840_000_000,
        "candidate_maximum_wall_seconds": 172_800,
        "candidate_maximum_gpu_seconds": 1_382_400 if priority < 2 else 691_200,
        "candidate_maximum_cost_micro_usd": 1_600_000_000,
        "operator_maximum_total_provider_seconds": 1_000_000,
        "operator_maximum_total_provider_gpu_seconds": 8_000_000,
        "operator_maximum_total_provider_cost_micro_usd": 10_000_000_000,
        "operator_maximum_per_attempt_provider_seconds": 400_000,
        "operator_maximum_per_attempt_provider_gpu_seconds": 3_200_000,
        "operator_maximum_per_attempt_provider_cost_micro_usd": 3_000_000_000,
    }


def _candidate(priority: int = 0) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads(_canonical(build_unregistered_campaign_intent_candidate(**_candidate_kwargs(priority)))),
    )


def _redigest(value: dict[str, Any]) -> bytes:
    body = {key: child for key, child in value.items() if key != "candidate_digest"}
    value["candidate_digest"] = hashlib.sha256(_canonical(body)).hexdigest()
    return _canonical(value)


def test_policy_template_exact_schema_bindings_roots_and_future_order() -> None:
    template = build_policy_template()
    assert set(template) == {
        "schema",
        "schema_version",
        "study_id",
        "milestone",
        "accepted_bindings",
        "execution_identity_policy",
        "qualification_policy",
        "canonical_roots",
        "future_compute_freeze_schema",
        "registrar_order",
        "authorization",
        "template_digest",
    }
    assert template["schema"] == contracts.POLICY_TEMPLATE_SCHEMA
    accepted = template["accepted_bindings"]
    assert accepted["g01_target_binding_digest"] == (
        "3315d20a6f9bdae3c5fdaf9567c5bce7b592890d0ebd26e10682002d816bf0c6"
    )
    assert accepted["checkpoint_b_refusal_source_digest"] == (
        "77d9ba4928fa29cc42a055201660304f2059d0ca6358371cc14618407bdc714b"
    )
    assert accepted["checkpoint_b_source_capsule_source_digest"] == (
        "efb6b0427895f60e16794f4e20b87d620522673edffb94f9ce277089b5581782"
    )
    assert accepted["g01q_source_digest"] == (
        "efde59caf3830d021223940b2d6e1a8e9631c90f0f8e51262041008ea177c7b3"
    )
    roots = template["canonical_roots"]
    assert roots["frozen_source"] == ("/workspace/inputs-goalzendo/g01-executions/<B_UUID>/frozen-source")
    assert roots["preexecution"] == "/workspace/status-goalzendo/g01-preexecution/<B_UUID>"
    assert roots["execution_status_absent_through_qualification"] is True
    assert roots["scientific_artifacts_absent_through_qualification"] is True
    assert roots["qualification_root_not_defined_by_this_milestone"] is True
    steps = template["registrar_order"]["steps"]
    assert steps.index("external_registrar_record_r1") < steps.index("compute_freeze")
    assert steps.index("external_registrar_record_r2") < steps.index("provider_create")
    assert steps[-1] == "still_no_g01_launch"
    expected_fields = template["future_compute_freeze_schema"]["exact_fields"]
    assert expected_fields[0] == "schema"
    assert expected_fields[-1] == "freeze_digest"
    assert "provision_receipt" not in expected_fields
    assert "pod_id" not in expected_fields


def test_policy_template_is_fresh_immutable_and_strictly_canonical() -> None:
    first = build_policy_template()
    second = build_policy_template()
    assert first is not second
    with pytest.raises(TypeError):
        first["schema"] = "changed"  # type: ignore[index]
    profiles = first["qualification_policy"]["ordered_profiles"]
    with pytest.raises(AttributeError):
        profiles.append({})
    payload = _canonical(first)
    validated = validate_policy_template_bytes(payload)
    assert validated["template_digest"] == first["template_digest"]
    for bad in (
        payload + b"\n",
        b'{"schema":1,"schema":1}',
        b'{"x":1.0}',
        b'{"x":NaN}',
    ):
        with pytest.raises(PreprovisionError):
            validate_policy_template_bytes(bad)


def test_candidate_exact_schema_identity_token_and_unverified_boundary() -> None:
    candidate = _candidate()
    assert set(candidate) == {
        "schema",
        "schema_version",
        "study_id",
        "state",
        "policy_template_digest",
        "execution_identities",
        "checkpoint_a_token",
        "candidate_tuple",
        "operator_limits",
        "expected_source_binding",
        "expected_runtime_binding",
        "expected_model_binding",
        "future_compute_freeze_requirements",
        "external_trust",
        "authorization",
        "candidate_digest",
    }
    assert candidate["state"] == "UNREGISTERED"
    identities = candidate["execution_identities"]
    assert identities["campaign_uuid"] == identities["g01_execution_uuid"]
    assert identities["candidate_id"] == f"g01q-{QUALIFICATION_UUID}"
    assert (
        len(
            {
                identities["eligibility_execution_uuid"],
                identities["g01_execution_uuid"],
                identities["qualification_uuid"],
            }
        )
        == 3
    )
    token = candidate["checkpoint_a_token"]
    assert set(token) == {
        "file_sha256",
        "token_digest",
        "bridge_source_digest",
        "portability",
        "externally_authenticated",
    }
    assert token["externally_authenticated"] is False
    assert token["portability"]["representation"] == "PORTABLE_BYTES"
    assert candidate["external_trust"]["caller_supplied_fields_label"] == ("caller_supplied_unverified")
    assert all(value is False for value in candidate["authorization"].values())
    assert all(
        value is False
        for key, value in candidate["external_trust"].items()
        if key
        not in {
            "caller_supplied_fields_label",
            "excluded_charges_require_separate_external_acceptance_before_r1",
        }
    )
    assert (
        candidate["external_trust"]["excluded_charges_require_separate_external_acceptance_before_r1"] is True
    )
    assert candidate["external_trust"]["storage_and_egress_external_acceptance_present"] is False


def test_candidate_profile_order_dispositions_and_tuple_are_exact() -> None:
    expected = (
        ("h200x8", "NVIDIA H200", 8, 79_200),
        ("h100-hbm3x8", "NVIDIA H100 80GB HBM3", 8, 79_200),
        ("h200x4", "NVIDIA H200", 4, 151_200),
        ("h100-hbm3x4", "NVIDIA H100 80GB HBM3", 4, 151_200),
    )
    for priority, row in enumerate(expected):
        candidate_tuple = _candidate(priority)["candidate_tuple"]
        assert (
            candidate_tuple["candidate_profile_id"],
            candidate_tuple["gpu_id"],
            candidate_tuple["gpu_count"],
            candidate_tuple["qualification_ceiling_seconds"],
        ) == row
        assert len(candidate_tuple["prior_candidate_dispositions"]) == priority
        assert candidate_tuple["cloud_type"] == "SECURE"
        assert candidate_tuple["network_volume_type"] == "HIGH_PERFORMANCE"
        assert candidate_tuple["network_volume_mount"] == "/workspace"

    for changed in (
        {**_candidate_kwargs(2), "prior_candidate_dispositions": []},
        {
            **_candidate_kwargs(1),
            "prior_candidate_dispositions": [
                {"profile_id": "h100-hbm3x8", "state": "skipped_no_stock", "registrar_record_sha256": SHA}
            ],
        },
        {
            **_candidate_kwargs(1),
            "prior_candidate_dispositions": [
                {"profile_id": "h200x8", "state": "operator_choice", "registrar_record_sha256": SHA}
            ],
        },
    ):
        with pytest.raises(PreprovisionError):
            build_unregistered_campaign_intent_candidate(**changed)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("campaign_uuid", QUALIFICATION_UUID),
        ("eligibility_execution_uuid", EXECUTION_UUID),
        ("qualification_uuid", EXECUTION_UUID),
        ("qualification_uuid", "33333333-3333-5333-8333-333333333333"),
        ("data_center_id", "us-ca-2"),
        ("network_volume_id", "/volume/path"),
        ("checkpoint_a_bridge_source_digest", "f" * 64),
    ],
)
def test_identity_hash_and_provider_grammar_rejections(field: str, replacement: Any) -> None:
    kwargs = _candidate_kwargs()
    kwargs[field] = replacement
    with pytest.raises(PreprovisionError):
        build_unregistered_campaign_intent_candidate(**kwargs)


@pytest.mark.parametrize(
    "field",
    [
        "price_ceiling_micro_usd_per_gpu_hour",
        "operator_maximum_wall_seconds",
        "operator_maximum_gpu_seconds",
        "operator_maximum_cost_micro_usd",
        "candidate_maximum_wall_seconds",
        "candidate_maximum_gpu_seconds",
        "candidate_maximum_cost_micro_usd",
        "operator_maximum_total_provider_seconds",
        "operator_maximum_total_provider_gpu_seconds",
        "operator_maximum_total_provider_cost_micro_usd",
        "operator_maximum_per_attempt_provider_seconds",
        "operator_maximum_per_attempt_provider_gpu_seconds",
        "operator_maximum_per_attempt_provider_cost_micro_usd",
    ],
)
@pytest.mark.parametrize("replacement", [True, 0, -1, 1.5])
def test_every_cap_rejects_bool_nonpositive_and_float(field: str, replacement: Any) -> None:
    kwargs = _candidate_kwargs()
    kwargs[field] = replacement
    with pytest.raises(PreprovisionError):
        build_unregistered_campaign_intent_candidate(**kwargs)


def test_cap_componentwise_monotonicity() -> None:
    mutations = (
        ("candidate_maximum_wall_seconds", 79_201),
        ("candidate_maximum_gpu_seconds", 3_456_001),
        ("candidate_maximum_cost_micro_usd", 3_840_000_001),
        ("operator_maximum_per_attempt_provider_seconds", 1_000_001),
        ("operator_maximum_per_attempt_provider_gpu_seconds", 8_000_001),
        ("operator_maximum_per_attempt_provider_cost_micro_usd", 10_000_000_001),
    )
    for field, replacement in mutations:
        kwargs = _candidate_kwargs()
        kwargs[field] = replacement
        with pytest.raises(PreprovisionError):
            build_unregistered_campaign_intent_candidate(**kwargs)


def test_candidate_and_attempt_caps_cover_worst_case_deadline_at_price_ceiling() -> None:
    for priority in range(4):
        candidate = _candidate(priority)
        candidate_tuple = candidate["candidate_tuple"]
        limits = candidate["operator_limits"]
        gpu_count = candidate_tuple["gpu_count"]
        wall = limits["candidate_maximum_wall_seconds"]
        assert wall in {172_800, 259_200, 345_600, 432_000}
        assert limits["candidate_maximum_gpu_seconds"] >= gpu_count * wall
        minimum_cost = (limits["price_ceiling_micro_usd_per_gpu_hour"] * gpu_count * wall + 3_599) // 3_600
        assert limits["candidate_maximum_cost_micro_usd"] >= minimum_cost
        attempt_seconds = 7_200 + candidate_tuple["qualification_ceiling_seconds"] + 3_600 + wall + 3_600
        assert limits["operator_maximum_per_attempt_provider_seconds"] >= attempt_seconds
        assert limits["operator_maximum_per_attempt_provider_gpu_seconds"] >= gpu_count * attempt_seconds
        attempt_cost = (
            limits["price_ceiling_micro_usd_per_gpu_hour"] * gpu_count * attempt_seconds + 3_599
        ) // 3_600
        assert limits["operator_maximum_per_attempt_provider_cost_micro_usd"] >= attempt_cost


def test_each_deadline_and_provider_attempt_floor_rejects_one_below() -> None:
    price = _candidate_kwargs()["price_ceiling_micro_usd_per_gpu_hour"]
    wall = 172_800
    gpu_count = 8
    science_gpu = gpu_count * wall
    science_cost = (price * science_gpu + 3_599) // 3_600
    attempt_seconds = 7_200 + 79_200 + 3_600 + wall + 3_600
    attempt_gpu = gpu_count * attempt_seconds
    attempt_cost = (price * attempt_gpu + 3_599) // 3_600
    mutations = (
        ("candidate_maximum_wall_seconds", wall - 1),
        ("candidate_maximum_gpu_seconds", science_gpu - 1),
        ("candidate_maximum_cost_micro_usd", science_cost - 1),
        ("operator_maximum_per_attempt_provider_seconds", attempt_seconds - 1),
        ("operator_maximum_per_attempt_provider_gpu_seconds", attempt_gpu - 1),
        ("operator_maximum_per_attempt_provider_cost_micro_usd", attempt_cost - 1),
    )
    for field, replacement in mutations:
        kwargs = _candidate_kwargs()
        kwargs[field] = replacement
        with pytest.raises(PreprovisionError):
            build_unregistered_campaign_intent_candidate(**kwargs)


def test_policy_requires_price_ceiling_and_lifetime_debit_enforcement_at_r1() -> None:
    policy = build_policy_template()["qualification_policy"]
    assert policy["observed_provider_gpu_price_must_not_exceed_price_ceiling"] is True
    assert policy["r1_registrar_debits_all_prior_provider_attempts"] is True
    assert policy["r1_rejects_unless_remaining_lifetime_caps_cover_current_attempt"] is True


def test_runtime_and_model_are_expectations_never_receipts_or_authentication() -> None:
    candidate = _candidate()
    runtime = candidate["expected_runtime_binding"]
    model = candidate["expected_model_binding"]
    source = candidate["expected_source_binding"]
    assert runtime["image_reference"] == "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
    assert runtime["image_reference_authenticated"] is False
    assert runtime["runtime_authenticated"] is False
    assert "oci_digest" not in runtime
    assert model["expectation_only"] is True
    assert model["model_snapshot_authenticated"] is False
    assert "snapshot_receipt_sha256" not in model
    assert model["leaf_manifest_digest"] == (
        "ff6162df6d6d022e565f916bf04c8902d82c6a82ee176de77ec59209e78a5bbc"
    )
    assert len(model["leaf_files"]) == 10
    assert source["new_checkpoint_b_capsule_revision_required"] is True
    assert source["canonical_capsule_artifacts_present"] is False
    assert not any("artifact_sha256" in key for key in source)


def test_candidate_strict_validation_replays_and_rejects_forged_nested_keys() -> None:
    candidate = _candidate()
    payload = _canonical(candidate)
    assert (
        validate_unregistered_campaign_intent_candidate_bytes(payload)["candidate_digest"]
        == (candidate["candidate_digest"])
    )
    with pytest.raises(PreprovisionError):
        validate_unregistered_campaign_intent_candidate_bytes(payload + b"\n")
    with pytest.raises(PreprovisionError):
        validate_unregistered_campaign_intent_candidate_bytes(
            b'{"state":"UNREGISTERED","state":"UNREGISTERED"}'
        )
    with pytest.raises(PreprovisionError):
        validate_unregistered_campaign_intent_candidate_bytes(b'{"x":1.0}')

    for forbidden in (
        "token_path",
        "token_inode",
        "token_device",
        "token_route",
        "detailed_checkpoint_a_evidence",
        "scientific_outcome",
        "reward_value",
        "metric_summary",
        "prediction_hash",
    ):
        changed = deepcopy(candidate)
        changed["external_trust"]["nested"] = {forbidden: "x"}
        with pytest.raises(PreprovisionError, match="forbidden"):
            validate_unregistered_campaign_intent_candidate_bytes(_redigest(changed))


def test_candidate_rejects_semantic_mutation_even_after_redigest() -> None:
    candidate = _candidate()
    mutations = (
        ("state", "REGISTERED"),
        ("policy_template_digest", "f" * 64),
    )
    for field, replacement in mutations:
        changed = deepcopy(candidate)
        changed[field] = replacement
        with pytest.raises(PreprovisionError):
            validate_unregistered_campaign_intent_candidate_bytes(_redigest(changed))
    changed = deepcopy(candidate)
    changed["external_trust"]["caller_supplied_uuid_facts_verified"] = True
    with pytest.raises(PreprovisionError):
        validate_unregistered_campaign_intent_candidate_bytes(_redigest(changed))


def test_every_operational_function_literal_refuses_first() -> None:
    names = (
        "register_campaign",
        "create_compute_freeze",
        "build_capsule",
        "stage_capsule",
        "materialize_model",
        "provision",
        "run_qualification",
        "execute_preprovision",
    )
    tree = ast.parse(Path(contracts.__file__).read_text(encoding="utf-8"))
    definitions = {
        node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for name in names:
        function = getattr(contracts, name)
        with pytest.raises(PreprovisionError, match=contracts.REFUSAL):
            function("ignored", path="/tmp/forbidden")
        definition = definitions[name]
        assert len(definition.body) == 1
        assert isinstance(definition.body[0], ast.Raise)


def test_cli_every_spelling_refuses_without_import_or_io(tmp_path: Path) -> None:
    for arguments in ([], ["build"], ["register", "--output", "x"], ["--help"], ["--version"]):
        completed = subprocess.run(
            [sys.executable, str(ENTRYPOINT), *arguments],
            cwd=tmp_path,
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 2
        assert completed.stdout == ""
        assert completed.stderr == (f"run_g01_preprovision: error: {contracts.REFUSAL}\n")
        assert list(tmp_path.iterdir()) == []
    tree = ast.parse(ENTRYPOINT.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert imported == {"annotations", "sys"}


def test_no_io_provider_or_model_surface_in_pure_module() -> None:
    tree = ast.parse(Path(contracts.__file__).read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".")[0]
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert imported.isdisjoint(
        {
            "os",
            "pathlib",
            "subprocess",
            "socket",
            "requests",
            "runpod",
            "torch",
            "transformers",
            "time",
            "random",
            "secrets",
        }
    )
    assert "open(" not in Path(contracts.__file__).read_text(encoding="utf-8")


def test_operational_refusals_never_reach_monkeypatched_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def trap(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("an operational boundary was reached")

    monkeypatch.setattr(builtins, "open", trap)
    monkeypatch.setattr(os, "open", trap)
    monkeypatch.setattr(os, "mkdir", trap)
    monkeypatch.setattr(os, "makedirs", trap)
    monkeypatch.setattr(subprocess, "run", trap)
    monkeypatch.setattr(subprocess, "Popen", trap)
    monkeypatch.setattr(socket, "socket", trap)
    monkeypatch.setattr(importlib, "import_module", trap)
    for name in (
        "register_campaign",
        "create_compute_freeze",
        "build_capsule",
        "stage_capsule",
        "materialize_model",
        "provision",
        "run_qualification",
        "execute_preprovision",
    ):
        with pytest.raises(PreprovisionError, match=contracts.REFUSAL):
            getattr(contracts, name)()


def test_package_is_source_only_and_not_declared_as_wheel_cli_or_package_data() -> None:
    package = "goalzendo_g01_preprovision"
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert f"{package} = " not in pyproject
    assert f"{package} = [" not in pyproject
    policy = json.loads(
        (ROOT / "reproducibility" / "releases" / "2026-08-11" / "release-policy.json").read_text()
    )
    assert package not in policy["full_checks"]["wheel"]["packages"]
    assert package not in {
        row["package"] for row in policy["bindings"]["source_packages"]["component_manifests"]
    }
    assert not any(
        row["path"].startswith(f"{package}/") for row in policy["full_checks"]["wheel"]["package_data"]
    )
    assert package not in {
        row["entry_point"].split(".", 1)[0] for row in policy["full_checks"]["wheel"]["console_scripts"]
    }
    package_files = sorted(
        path.relative_to(ROOT / "src" / package).as_posix()
        for path in (ROOT / "src" / package).rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    )
    assert package_files == ["__init__.py", "contracts.py"]


def test_accepted_hash_locked_sources_remain_exact() -> None:
    expected = {
        "src/goalzendo_g01_coordinator/__init__.py": "85a9454fc21c2dc8c70e6ebf9eb605aa0208e6c13e73547ec0f4abdd08abad35",
        "src/goalzendo_g01_coordinator/coordinator.py": "a3390dad47ea1fd2aa7b1ca22e4475b9a510c47fc85f9b296ed925303663b302",
        "runs/goalzendo/run_g01_global_coordinator.py": "69f5ec520d8c681d5e9cab1dd9687d7fd39c948f08ff397331bd5ce74f824a06",
        "tests/goalzendo/test_g01_global_coordinator.py": "f0ec9d0c4815e6bd69e7dda2a72a8f9ba03bf4405dced5e12f05f99660be207b",
        "docs/goalzendo/protocols/g01-global-coordinator-checkpoint-b.md": "93acc6262758dd5478e342e2f89fd5cb1ab3b61a03bed34b1a6d5083cacaaeba",
        "src/goalzendo_g01_qualification/__init__.py": "075ea0981ba3fe5da1a7ce95b481f3ef988be8e0be62c9d185b30f0c0d699fab",
        "src/goalzendo_g01_qualification/qualification.py": "5edc9e9fb756081040c62da8965f462eb9baa1d26b276becd41872810e646355",
        "runs/goalzendo/run_g01_compute_qualification.py": "2531edd47bbe5883741e674ccf41b4ddf77e16530f6b67c92a1ed51a90cb7eb4",
        "tests/goalzendo/test_g01_compute_qualification.py": "a4acde81cf393673daf6ad49b3bfb6708a10b5b1f2b14f19e1a47b905f543c6e",
        "docs/goalzendo/protocols/g01-compute-qualification.md": "e4997c48f0611117c08ca6850d438f5f0fb54bcdc2dba155d23e0b77cbdec355",
        "runs/goalzendo/build_g01_b_source_capsule.py": "c1255da5ed0bebe5c91bbcc724213e671e9b22b2b79dc52915426e51ea6aa593",
        "runs/goalzendo/g01_b_source_capsule_stage.py": "253685cefcf31a31324a8aa14ec854b9fef09babf38d24a5e930c68da80afa55",
        "tests/goalzendo/test_g01_b_source_capsule.py": "fe0919f58adc131c774ab5f7387a7ac0d3b2c2a4d83c7dd4503b9ba77fe6abd3",
        "docs/goalzendo/protocols/g01-b-source-capsule.md": "cf9d73edc279b6dacd1e64823ffff016a8d409b2cdb9cd434e5e244d17687251",
        "runs/goalzendo/build_g00f_g01_bridge_bundle.py": "29c906cdfc6c50f672a0819ece3d007138382099dea26e975715e7619e65b243",
        "runs/goalzendo/g00f_g01_bridge_bundle_stage.py": "58c0b75a4608a0b5cd6a85d2ae1b7b5a13af080fc8f9a6305151cca851963b49",
        "reproducibility/goalzendo/g00f-g01-bridge-prebootstrap-20260812/g00f-g01-bridge-bundle.tar.gz": "db5b4533d1b5b5dc00afaaaf0f954ea91a570477e9837236f8e575401359630b",
        "reproducibility/goalzendo/g00f-g01-bridge-prebootstrap-20260812/g00f-g01-bridge-bundle-manifest.json": "c7bc759336ca374d81df6e10dd2338a27cb01127879f73321805b70aa7008258",
        "reproducibility/goalzendo/g00f-g01-bridge-prebootstrap-20260812/g00f-g01-bridge-bundle-freeze.json": "1b85565d571c5c52c5010ea5752ce172bf6d58356ac1824bf8ff5b80b6bcb5dc",
    }
    for relative, expected_sha in expected.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected_sha


def test_public_authority_flags_are_never_dataclass_fields_or_true() -> None:
    assert dataclasses.is_dataclass(build_policy_template()) is False
    candidate = _candidate()
    assert candidate["authorization"]["g01_launch_authorized"] is False
    assert candidate["authorization"]["qualification_execution_authorized"] is False
