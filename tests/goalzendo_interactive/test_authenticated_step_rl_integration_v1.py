from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from goalzendo_interactive.authenticated_rollouts_v2 import (
    collect_eight_authenticated_rollouts,
)
from goalzendo_interactive.authenticated_step_v1 import (
    AdamWOptimizerSpec,
    AuthenticatedStepEvidenceBundle,
    AuthenticatedUpdateCoordinate,
    execute_authenticated_rl_step,
)
from goalzendo_interactive.generation import (
    generate_episode_bank,
    small_fixture_bank_spec,
)
from goalzendo_interactive.streaming_objective_v3 import (
    derive_verified_streaming_objective_plan,
)
from tests.goalzendo_interactive.test_sealed_runtime_v1 import (
    _deterministic_runtime,
    _load,
    _write_artifacts,
)

_SMOKE_EPISODE_ID = "g03-engine-small-fixture-v1-p1-0-any-perfect-factorial-s03-ac92fe83ea-33a184e706"
_SMOKE_EPISODE_DIGEST = "a8081c54afca80fb55221367c493d967ddfdec02c3985a142e5a1de441ad3cc6"


def test_atomic_rl_step_replays_eight_real_constrained_rollouts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the graph-free RL path without substituting fake diagnostics."""

    with _deterministic_runtime():
        runtime = _load(_write_artifacts(tmp_path), monkeypatch)
        episode = generate_episode_bank(small_fixture_bank_spec()).episodes[2]
        assert episode.episode_id == _SMOKE_EPISODE_ID
        assert episode.digest == _SMOKE_EPISODE_DIGEST

        group = collect_eight_authenticated_rollouts(
            runtime.provider,
            runtime.tokenizer,
            runtime.compiler,
            episode,
            runtime.provider.model_provenance,
            run_seed=30_202,
            temperature=0.7,
            absolute_tolerance=1e-6,
            maximum_sequence_tokens=65_536,
            maximum_turns=6,
        )
        assert len(group.rollouts) == 8
        assert sum(len(rollout.record.turns) for rollout in group.rollouts) > 0

        plan = derive_verified_streaming_objective_plan(
            (group,),
            entropy_coefficient=Fraction(1, 100),
        )
        records = tuple(rollout.record for rollout in group.rollouts)
        before = runtime.manifest
        bundle = execute_authenticated_rl_step(
            runtime,
            plan,
            records,
            (episode,),
            AdamWOptimizerSpec(
                learning_rate=3e-6,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-8,
                weight_decay=0.0,
            ),
            AuthenticatedUpdateCoordinate(
                study_id="g03-g-pinned-qwen-smoke-v1",
                run_id="rl-fake-model-integration",
                update_index=0,
                objective_kind="rl",
            ),
        )

        assert type(bundle) is AuthenticatedStepEvidenceBundle
        assert bundle.record.plan_digest == plan.plan.digest
        assert bundle.record.pre_runtime_manifest_digest == before.digest
        assert bundle.record.post_runtime_manifest_digest == runtime.manifest.digest
        assert bundle.record.pre_policy_state_digest != bundle.record.post_policy_state_digest
        assert bundle.record.changed_parameter_count > 0
        assert bundle.record.optimizer_step_call_count == 1
        assert bundle.pre_optimizer_state.update_count == 0
        assert bundle.post_optimizer_state.update_count == 1
        assert bundle.record.digest in bundle.to_json()
        assert runtime.reauthenticate() == runtime.manifest
