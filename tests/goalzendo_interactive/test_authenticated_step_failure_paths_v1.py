from __future__ import annotations

from pathlib import Path

import pytest

from goalzendo_interactive.authenticated_step_v1 import (
    AuthenticatedStepError,
    execute_authenticated_sft_step,
)
from goalzendo_interactive.streaming_sft_v3 import build_streaming_sft_plan
from tests.goalzendo_interactive.test_authenticated_step_v1 import (
    _coordinate,
    _optimizer_spec,
    _sources,
)
from tests.goalzendo_interactive.test_sealed_runtime_v1 import (
    _deterministic_runtime,
    _load,
    _write_artifacts,
)


def test_foreign_policy_plan_fails_without_changing_the_target_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _deterministic_runtime():
        source_runtime = _load(_write_artifacts(tmp_path / "source"), monkeypatch)
        sources = _sources()
        foreign_plan = build_streaming_sft_plan(
            sources,
            source_runtime.tokenizer,
            source_runtime.compiler,
            source_runtime.provider,
            maximum_sequence_tokens=65_536,
        )

        target_runtime = _load(_write_artifacts(tmp_path / "target"), monkeypatch)
        before = target_runtime.manifest
        with pytest.raises(AuthenticatedStepError, match="failed closed"):
            execute_authenticated_sft_step(
                target_runtime,
                foreign_plan,
                sources,
                _optimizer_spec(),
                _coordinate(run_id="foreign-plan"),
            )
        assert target_runtime.reauthenticate() == before
