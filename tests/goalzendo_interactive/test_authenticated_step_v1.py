from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path
from typing import NoReturn

import pytest
import torch

import goalzendo_interactive.authenticated_step_v1 as authenticated_step_module
from goalzendo_interactive.authenticated_model_provider_v2 import (
    TrainableParameterRegistry,
)
from goalzendo_interactive.authenticated_step_v1 import (
    AUTHENTICATED_STEP_AUTHORIZES_EXECUTION,
    AdamWOptimizerSpec,
    AuthenticatedStepError,
    AuthenticatedStepEvidenceBundle,
    AuthenticatedStepRecord,
    AuthenticatedUpdateCoordinate,
    ParameterByteRecord,
    _construct_optimizer,
    authenticated_step_manifest,
    execute_authenticated_rl_step,
    execute_authenticated_sft_step,
)
from goalzendo_interactive.generation import generate_episode_bank, small_fixture_bank_spec
from goalzendo_interactive.sealed_runtime_v1 import (
    SealedQwenRuntime,
    SealedQwenRuntimeError,
)
from goalzendo_interactive.streaming_sft_v3 import (
    ReferenceTrajectorySFTSource,
    build_streaming_sft_plan,
)
from goalzendo_interactive.trajectory_banks import generate_reference_trajectory_bank
from tests.goalzendo_interactive.test_sealed_runtime_v1 import (
    _deterministic_runtime,
    _load,
    _write_artifacts,
)


def _sources() -> tuple[ReferenceTrajectorySFTSource, ...]:
    episodes = generate_episode_bank(small_fixture_bank_spec())
    trajectories = generate_reference_trajectory_bank(episodes)
    return (
        ReferenceTrajectorySFTSource(
            episodes.episodes[0],
            trajectories.records[0],
        ),
    )


def _optimizer_spec() -> AdamWOptimizerSpec:
    return AdamWOptimizerSpec(
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0.01,
    )


def _coordinate(*, run_id: str) -> AuthenticatedUpdateCoordinate:
    return AuthenticatedUpdateCoordinate(
        study_id="g03-local-smoke",
        run_id=run_id,
        update_index=0,
        objective_kind="sft",
    )


def test_atomic_sft_step_refreshes_policy_and_attests_one_adamw_update(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _deterministic_runtime():
        runtime = _load(_write_artifacts(tmp_path), monkeypatch)
        sources = _sources()
        plan = build_streaming_sft_plan(
            sources,
            runtime.tokenizer,
            runtime.compiler,
            runtime.provider,
            maximum_sequence_tokens=65_536,
        )
        before = runtime.manifest
        bundle = execute_authenticated_sft_step(
            runtime,
            plan,
            sources,
            _optimizer_spec(),
            _coordinate(run_id="success"),
        )

        assert type(bundle) is AuthenticatedStepEvidenceBundle
        record = bundle.record
        assert type(record) is AuthenticatedStepRecord
        assert record.optimizer_step_call_count == 1
        assert record.changed_parameter_count >= 1
        assert record.pre_policy_state_digest == before.provider_policy_state_digest
        assert record.post_policy_state_digest != record.pre_policy_state_digest
        assert record.post_runtime_manifest_digest == runtime.manifest.digest
        assert record.post_policy_state_digest == runtime.provider.policy_state_digest
        assert record.authorizes_execution is False
        assert AUTHENTICATED_STEP_AUTHORIZES_EXECUTION is False
        assert record.digest in record.to_json()
        assert bundle.digest in bundle.to_json()
        assert bundle.post_optimizer_state.update_count == 1
        assert bundle.pre_optimizer_state.update_count == 0
        assert bundle.optimizer_spec == _optimizer_spec()
        assert bundle.optimizer_spec.digest == record.optimizer_spec_digest
        assert runtime.reauthenticate() == runtime.manifest

        forged_optimizer_binding = replace(record, optimizer_spec_digest="0" * 64)
        with pytest.raises(AuthenticatedStepError, match="optimizer specification differs"):
            replace(bundle, record=forged_optimizer_binding)
        forged_change_count = replace(
            record,
            changed_parameter_count=record.changed_parameter_count + 1,
        )
        with pytest.raises(AuthenticatedStepError, match="changed-parameter count differs"):
            replace(bundle, record=forged_change_count)

        post = runtime.manifest
        with pytest.raises(AuthenticatedStepError, match="failed closed"):
            execute_authenticated_sft_step(
                runtime,
                plan,
                sources,
                _optimizer_spec(),
                _coordinate(run_id="reused-stale-plan"),
            )
        assert runtime.reauthenticate() == post


def test_record_construction_baseexception_rolls_back_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _deterministic_runtime():
        runtime = _load(_write_artifacts(tmp_path), monkeypatch)
        sources = _sources()
        plan = build_streaming_sft_plan(
            sources,
            runtime.tokenizer,
            runtime.compiler,
            runtime.provider,
            maximum_sequence_tokens=65_536,
        )
        before = runtime.manifest

        def interrupt_record(_self: AuthenticatedStepRecord) -> str:
            raise KeyboardInterrupt("injected after prepared post-update manifest")

        monkeypatch.setattr(AuthenticatedStepRecord, "to_json", interrupt_record)
        with pytest.raises(KeyboardInterrupt, match="injected after prepared"):
            execute_authenticated_sft_step(
                runtime,
                plan,
                sources,
                _optimizer_spec(),
                _coordinate(run_id="rollback"),
            )
        assert runtime.reauthenticate() == before
        assert runtime.manifest.provider_policy_state_digest == before.provider_policy_state_digest
        assert runtime.manifest.model_provenance_digest == before.model_provenance_digest


def test_step_api_has_no_caller_optimizer_gradient_or_model_injection() -> None:
    for function in (execute_authenticated_sft_step, execute_authenticated_rl_step):
        parameters = inspect.signature(function).parameters
        assert {
            "model",
            "provider",
            "tokenizer",
            "compiler",
            "optimizer",
            "scheduler",
            "scaler",
            "gradients",
            "named_parameters",
            "step_callback",
        }.isdisjoint(parameters)
    manifest = authenticated_step_manifest()
    assert manifest["authorizes_execution"] is False
    assert manifest["optimizer_step_count"] == 1
    assert manifest["failure_semantics"] == (
        "byte-exact rollback or permanently corrupt sealed runtime"
    )


def test_adamw_step_monkeypatch_is_rejected_against_import_time_implementation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _deterministic_runtime():
        runtime = _load(_write_artifacts(tmp_path), monkeypatch)
        registry = runtime.trainable_parameter_registry
        original_step = torch.optim.AdamW.step

        def changed_step(_self: torch.optim.AdamW, closure: object = None) -> None:
            _ = closure

        monkeypatch.setattr(torch.optim.AdamW, "step", changed_step)
        with pytest.raises(AuthenticatedStepError, match="implementation changed"):
            _construct_optimizer(registry, _optimizer_spec())
        monkeypatch.setattr(torch.optim.AdamW, "step", original_step)

        closure = original_step.__closure__
        assert closure is not None and inspect.isfunction(closure[0].cell_contents)
        monkeypatch.setattr(closure[0], "cell_contents", changed_step)
        with pytest.raises(AuthenticatedStepError, match="implementation changed"):
            _construct_optimizer(registry, _optimizer_spec())


def test_partial_trusted_adamw_mutation_then_raise_rolls_back_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _deterministic_runtime():
        runtime = _load(_write_artifacts(tmp_path), monkeypatch)
        sources = _sources()
        plan = build_streaming_sft_plan(
            sources,
            runtime.tokenizer,
            runtime.compiler,
            runtime.provider,
            maximum_sequence_tokens=65_536,
        )
        before = runtime.manifest
        original_construct = authenticated_step_module._construct_optimizer
        original_require = (
            authenticated_step_module._require_trusted_optimizer_implementation
        )
        original_restore = authenticated_step_module._restore_parameters
        optimizers: list[torch.optim.AdamW] = []
        mutation_counts: list[int] = []
        guard_calls = 0

        def capture_optimizer(
            registry: TrainableParameterRegistry,
            spec: AdamWOptimizerSpec,
        ) -> torch.optim.AdamW:
            optimizer = original_construct(registry, spec)
            optimizers.append(optimizer)
            return optimizer

        def corrupt_second_state_immediately_before_step(
        ) -> authenticated_step_module._OptimizerImplementationGuard:
            nonlocal guard_calls
            implementation = original_require()
            guard_calls += 1
            if guard_calls == 3:
                optimizer = optimizers[0]
                victim = optimizer.param_groups[0]["params"][1]
                optimizer.state[victim] = {
                    "step": torch.tensor(0.0),
                    "exp_avg": torch.zeros((1,), dtype=victim.dtype),
                    "exp_avg_sq": torch.zeros((1,), dtype=victim.dtype),
                }
            return implementation

        def observe_partial_then_restore(
            registry: TrainableParameterRegistry,
            snapshots: tuple[torch.Tensor, ...],
        ) -> None:
            mutation_counts.append(
                sum(
                    not torch.equal(parameter.detach().to(device="cpu"), snapshot)
                    for parameter, snapshot in zip(
                        registry.parameters,
                        snapshots,
                        strict=True,
                    )
                )
            )
            original_restore(registry, snapshots)

        monkeypatch.setattr(
            authenticated_step_module,
            "_construct_optimizer",
            capture_optimizer,
        )
        monkeypatch.setattr(
            authenticated_step_module,
            "_require_trusted_optimizer_implementation",
            corrupt_second_state_immediately_before_step,
        )
        monkeypatch.setattr(
            authenticated_step_module,
            "_restore_parameters",
            observe_partial_then_restore,
        )
        with pytest.raises(AuthenticatedStepError, match="failed closed") as failure:
            execute_authenticated_sft_step(
                runtime,
                plan,
                sources,
                _optimizer_spec(),
                _coordinate(run_id="partial-adamw-raise"),
            )
        assert mutation_counts, repr(failure.value.__cause__)
        assert 0 < mutation_counts[0] < len(runtime.trainable_parameter_registry.parameters)
        assert runtime.reauthenticate() == before


def test_nonfinite_post_parameter_is_rejected_and_rolled_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _deterministic_runtime():
        runtime = _load(_write_artifacts(tmp_path), monkeypatch)
        sources = _sources()
        plan = build_streaming_sft_plan(
            sources,
            runtime.tokenizer,
            runtime.compiler,
            runtime.provider,
            maximum_sequence_tokens=65_536,
        )
        before = runtime.manifest
        original_records = authenticated_step_module._parameter_records
        calls = 0

        def inject_nonfinite_post_state(
            registry: TrainableParameterRegistry,
        ) -> tuple[tuple[ParameterByteRecord, ...], str]:
            nonlocal calls
            calls += 1
            if calls == 2:
                parameter = registry.parameters[0]
                with torch.no_grad():
                    parameter.reshape(-1)[0] = float("inf")
            return original_records(registry)

        monkeypatch.setattr(
            authenticated_step_module,
            "_parameter_records",
            inject_nonfinite_post_state,
        )
        with pytest.raises(AuthenticatedStepError, match="non-finite"):
            execute_authenticated_sft_step(
                runtime,
                plan,
                sources,
                _optimizer_spec(),
                _coordinate(run_id="nonfinite-post-parameter"),
            )
        assert calls == 2
        assert runtime.reauthenticate() == before


def test_rollback_failure_marks_runtime_permanently_corrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _deterministic_runtime():
        runtime = _load(_write_artifacts(tmp_path), monkeypatch)
        sources = _sources()
        plan = build_streaming_sft_plan(
            sources,
            runtime.tokenizer,
            runtime.compiler,
            runtime.provider,
            maximum_sequence_tokens=65_536,
        )

        def fail_record(_self: AuthenticatedStepRecord) -> NoReturn:
            raise RuntimeError("injected post-step record failure")

        def fail_rollback(
            _self: SealedQwenRuntime,
            _session: object,
        ) -> NoReturn:
            raise SealedQwenRuntimeError("injected rollback failure")

        monkeypatch.setattr(AuthenticatedStepRecord, "to_json", fail_record)
        monkeypatch.setattr(
            SealedQwenRuntime,
            "_rollback_authenticated_update",
            fail_rollback,
        )
        with pytest.raises(
            AuthenticatedStepError,
            match="recovery could not restore",
        ):
            execute_authenticated_sft_step(
                runtime,
                plan,
                sources,
                _optimizer_spec(),
                _coordinate(run_id="rollback-failure-corrupt"),
            )
        assert runtime._update_corrupt is True
        assert runtime._update_active is False
        with pytest.raises(SealedQwenRuntimeError, match="marked corrupt"):
            runtime.reauthenticate()
