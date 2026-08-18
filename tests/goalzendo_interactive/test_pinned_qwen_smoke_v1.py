from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast

import pytest

import goalzendo_interactive.pinned_qwen_smoke_v1 as smoke_module
from goalzendo_interactive._json import json_digest
from goalzendo_interactive.action_tokenization_v2 import (
    ExactDecodeTokenizerProtocol,
    FragmentActionTokenCompiler,
)
from goalzendo_interactive.pinned_qwen_smoke_v1 import (
    PINNED_QWEN_SMOKE_AUTHORIZES_EXECUTION,
    PinnedQwenSmokeCellConfig,
    PinnedQwenSmokeError,
    VerifiedPinnedQwenSmokeCompletion,
    _artifact_records,
    build_pinned_qwen_smoke_preflight,
    reconstruct_frozen_cache_trace,
    run_pinned_qwen_smoke_cell,
    validate_pinned_qwen_smoke_path_separation,
    verify_pinned_qwen_smoke_completion,
    write_failure_marker,
)
from goalzendo_interactive.provenance import interactive_source_provenance


class _SyntheticCompiler:
    def __init__(self, token_ids: tuple[int, ...]) -> None:
        self.token_ids = token_ids

    def trace_action(self, _action: object) -> SimpleNamespace:
        return SimpleNamespace(action_token_ids=self.token_ids)


def _config(
    tmp_path: Path,
    *,
    output_root: Path | None = None,
    cell_kind: Literal["sft", "rl"] = "sft",
) -> PinnedQwenSmokeCellConfig:
    return PinnedQwenSmokeCellConfig(
        cell_kind=cell_kind,
        artifact_root=tmp_path / "model-root",
        episode_fixture_path=tmp_path / "fixture-root" / "episodes.json",
        output_root=tmp_path / "output-root" if output_root is None else output_root,
        expected_artifact_manifest_digest="a" * 64,
        expected_source_fingerprint="b" * 64,
        device="cuda:0",
    )


def _config_with_paths(
    artifact_root: Path,
    episode_fixture_path: Path,
    output_root: Path,
) -> PinnedQwenSmokeCellConfig:
    return PinnedQwenSmokeCellConfig(
        cell_kind="sft",
        artifact_root=artifact_root,
        episode_fixture_path=episode_fixture_path,
        output_root=output_root,
        expected_artifact_manifest_digest="a" * 64,
        expected_source_fingerprint="b" * 64,
        device="cuda:0",
    )


def _write_synthetic_completed_tree(
    tmp_path: Path,
    *,
    cell_kind: Literal["sft", "rl"] = "sft",
) -> tuple[Path, str]:
    output = tmp_path / "completed"
    output.mkdir(parents=True)
    source = SimpleNamespace(
        fingerprint="b" * 64,
        as_obj=lambda: {"fingerprint": "b" * 64},
    )
    bank = SimpleNamespace(digest=smoke_module.EPISODE_BANK_DIGEST)
    episode = SimpleNamespace(
        episode_id=smoke_module.SMOKE_EPISODE_ID,
        digest=smoke_module.SMOKE_EPISODE_DIGEST,
        as_obj=lambda: {"episode_id": smoke_module.SMOKE_EPISODE_ID},
    )
    reference = SimpleNamespace(
        transcript_digest=smoke_module.SMOKE_REFERENCE_TRANSCRIPT_DIGEST,
        as_obj=lambda: {
            "transcript_digest": smoke_module.SMOKE_REFERENCE_TRANSCRIPT_DIGEST,
        },
    )
    cache = SimpleNamespace(
        digest="3" * 64,
        maximum_absolute_error_hex=(0.0).hex(),
        to_json=lambda: json.dumps({"digest": "3" * 64}, separators=(",", ":")),
    )
    record = SimpleNamespace(
        plan_digest="4" * 64,
        digest="5" * 64,
        pre_policy_state_digest="6" * 64,
        post_policy_state_digest="7" * 64,
        changed_parameter_count=2,
        optimizer_step_call_count=1,
        cuda_peak_allocated_bytes=10,
        cuda_peak_reserved_bytes=20,
    )
    bundle = SimpleNamespace(
        record=record,
        digest="8" * 64,
        to_json=lambda: json.dumps({"digest": "8" * 64}, separators=(",", ":")),
    )
    rollout_digests = tuple(f"{index:064x}" for index in range(8)) if cell_kind == "rl" else ()
    rollout_json = tuple(json.dumps({"digest": digest}, separators=(",", ":")) for digest in rollout_digests)
    completion_digest = smoke_module._write_success_artifacts(
        _config(tmp_path, output_root=output, cell_kind=cell_kind),
        source=cast(Any, source),
        environment={"fake": True},
        bank=cast(Any, bank),
        episode=cast(Any, episode),
        reference=cast(Any, reference),
        runtime_pre_json="{}",
        runtime_post_json="{}",
        cache=cast(Any, cache),
        plan_json="{}",
        rollout_json=rollout_json,
        rollout_digests=rollout_digests,
        collection_cache_metrics={
            "sampling_cache_reuse_count": 0 if cell_kind == "sft" else 1,
            "sampling_cache_reset_count": 0 if cell_kind == "sft" else 2,
        },
        bundle=cast(Any, bundle),
        elapsed_seconds=1.0,
    )
    return output, completion_digest


def test_cache_trace_is_exactly_cold_two_extensions_then_divergence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opening = tuple(range(31))
    terminal = tuple(range(100, 140))

    def fake_render(
        _tokenizer: object,
        dialogue: tuple[object, ...],
    ) -> tuple[str, tuple[int, ...]]:
        phase = cast(Any, dialogue[-1]).phase
        return ("prompt", opening if phase == "opening" else terminal)

    monkeypatch.setattr(smoke_module, "render_generation_prefix", fake_render)
    expected = (
        opening,
        (*opening, 4_913),
        (*opening, 4_913, 3_397),
        terminal,
    )
    monkeypatch.setattr(
        smoke_module,
        "SMOKE_CACHE_TRACE_DIGEST",
        json_digest(
            [list(prefix) for prefix in expected],
            domain="goalzendo-interactive-incremental-cache-trace-v1",
        ),
    )
    result = reconstruct_frozen_cache_trace(
        cast(ExactDecodeTokenizerProtocol, object()),
        cast(FragmentActionTokenCompiler, _SyntheticCompiler((4_913, 3_397, 17))),
    )
    assert result == expected
    assert tuple(len(prefix) for prefix in result) == (31, 32, 33, 40)


def test_cache_trace_rejects_changed_ready_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_render(
        _tokenizer: object,
        dialogue: tuple[object, ...],
    ) -> tuple[str, tuple[int, ...]]:
        phase = cast(Any, dialogue[-1]).phase
        return ("prompt", tuple(range(31 if phase == "opening" else 40)))

    monkeypatch.setattr(smoke_module, "render_generation_prefix", fake_render)
    with pytest.raises(PinnedQwenSmokeError, match="changed the frozen cache trace"):
        reconstruct_frozen_cache_trace(
            cast(ExactDecodeTokenizerProtocol, object()),
            cast(FragmentActionTokenCompiler, _SyntheticCompiler((4_913, 9_999))),
        )


def test_rl_collection_releases_cache_before_plan_and_atomic_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeIncremental:
        def __init__(self) -> None:
            self.cache_reuse_count = 0
            self.reset_count = 0
            self.cache_is_live = False

        def reset_cache(self) -> None:
            events.append("reset")
            assert self.cache_is_live
            self.cache_is_live = False
            self.reset_count += 1

    incremental = FakeIncremental()
    record = SimpleNamespace(
        turns=(object(),),
        digest="9" * 64,
        to_json=lambda: "{}",
    )
    group = SimpleNamespace(
        rollouts=tuple(SimpleNamespace(record=record) for _ in range(8)),
    )
    plan = SimpleNamespace(plan=SimpleNamespace(to_json=lambda: "{}"))
    bundle = SimpleNamespace()

    def fake_collect(selected: FakeIncremental, *_args: object, **_kwargs: object) -> Any:
        events.append("collect")
        selected.cache_is_live = True
        selected.cache_reuse_count += 1
        return group

    def fake_derive(*_args: object, **_kwargs: object) -> Any:
        events.append("derive")
        assert not incremental.cache_is_live
        return plan

    def fake_execute(*_args: object, **_kwargs: object) -> Any:
        events.append("step")
        assert not incremental.cache_is_live
        return bundle

    monkeypatch.setattr(smoke_module, "collect_eight_authenticated_rollouts", fake_collect)
    monkeypatch.setattr(smoke_module, "derive_verified_streaming_objective_plan", fake_derive)
    monkeypatch.setattr(smoke_module, "execute_authenticated_rl_step", fake_execute)
    config = _config(tmp_path, cell_kind="rl")
    result = smoke_module._run_step(
        config,
        cast(
            Any,
            SimpleNamespace(
                tokenizer=object(),
                compiler=object(),
                provider=SimpleNamespace(model_provenance=object()),
            ),
        ),
        cast(Any, incremental),
        cast(Any, object()),
        cast(Any, object()),
    )
    assert events == ["collect", "reset", "derive", "step"]
    assert incremental.cache_is_live is False
    assert result[4] == {
        "sampling_cache_reuse_count": 1,
        "sampling_cache_reset_count": 0,
    }


def test_rl_collection_failure_still_releases_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeIncremental:
        cache_reuse_count = 0
        reset_count = 0
        cache_is_live = False

        def reset_cache(self) -> None:
            events.append("reset")
            self.cache_is_live = False

    incremental = FakeIncremental()

    def fail_collection(selected: FakeIncremental, *_args: object, **_kwargs: object) -> Any:
        events.append("collect")
        selected.cache_is_live = True
        raise RuntimeError("collection failed")

    monkeypatch.setattr(smoke_module, "collect_eight_authenticated_rollouts", fail_collection)
    config = _config(tmp_path, cell_kind="rl")
    with pytest.raises(RuntimeError, match="collection failed"):
        smoke_module._run_step(
            config,
            cast(
                Any,
                SimpleNamespace(
                    tokenizer=object(),
                    compiler=object(),
                    provider=SimpleNamespace(model_provenance=object()),
                ),
            ),
            cast(Any, incremental),
            cast(Any, object()),
            cast(Any, object()),
        )
    assert events == ["collect", "reset"]
    assert incremental.cache_is_live is False


def test_fresh_root_and_nonauthorizing_failure_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "out"
    output.mkdir()
    config = _config(tmp_path, output_root=output)
    with pytest.raises(PinnedQwenSmokeError, match="fresh and absent"):
        run_pinned_qwen_smoke_cell(config)

    error = RuntimeError("prospective failure")
    write_failure_marker(output, "sft", error)
    failure = json.loads((output / "FAILURE.json").read_text(encoding="utf-8"))
    assert failure["status"] == "failed"
    assert failure["authorization"] == {"weight_updates": False}
    assert PINNED_QWEN_SMOKE_AUTHORIZES_EXECUTION is False


def test_artifact_inventory_excludes_completion_pointers_and_rejects_links(
    tmp_path: Path,
) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "evidence.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "COMPLETE").write_text("c" * 64 + "\n", encoding="utf-8")
    (tmp_path / "completion-attestation.json").write_text("{}\n", encoding="utf-8")
    records = _artifact_records(tmp_path)
    assert tuple(record["path"] for record in records) == ("nested/evidence.json",)

    link = tmp_path / "nested" / "link"
    try:
        link.symlink_to(tmp_path / "nested" / "evidence.json")
    except OSError:
        pytest.skip("symlinks are unavailable on this test filesystem")
    with pytest.raises(PinnedQwenSmokeError, match="link or special"):
        _artifact_records(tmp_path)


def test_completion_verifier_accepts_only_externally_anchored_nominal_tree(
    tmp_path: Path,
) -> None:
    output, completion_digest = _write_synthetic_completed_tree(tmp_path)
    verified = verify_pinned_qwen_smoke_completion(
        output,
        expected_completion_digest=completion_digest,
    )
    assert type(verified) is VerifiedPinnedQwenSmokeCompletion
    assert verified.output_root == output
    assert verified.cell_kind == "sft"
    assert verified.completion_digest == completion_digest
    assert tuple(record.path for record in verified.files) == tuple(
        sorted(
            {
                "cache-comparison.json",
                "environment.json",
                "episode.json",
                "objective-plan.json",
                "reference-trajectory.json",
                "report.json",
                "runtime-post.json",
                "runtime-pre.json",
                "source-provenance.json",
                "step-evidence.json",
            }
        )
    )
    with pytest.raises(PinnedQwenSmokeError, match="COMPLETE"):
        verify_pinned_qwen_smoke_completion(
            output,
            expected_completion_digest="f" * 64,
        )


def test_completion_verifier_accepts_exact_eight_rollout_rl_inventory(tmp_path: Path) -> None:
    output, completion_digest = _write_synthetic_completed_tree(tmp_path, cell_kind="rl")
    verified = verify_pinned_qwen_smoke_completion(
        output,
        expected_completion_digest=completion_digest,
    )
    assert verified.cell_kind == "rl"
    assert tuple(record.path for record in verified.files if record.path.startswith("rollouts/")) == (
        "rollouts/00.json",
        "rollouts/01.json",
        "rollouts/02.json",
        "rollouts/03.json",
        "rollouts/04.json",
        "rollouts/05.json",
        "rollouts/06.json",
        "rollouts/07.json",
    )


@pytest.mark.parametrize("target_name", ["environment.json", "report.json"])
def test_completion_verifier_rejects_attested_byte_mutation(
    tmp_path: Path,
    target_name: str,
) -> None:
    output, completion_digest = _write_synthetic_completed_tree(tmp_path)
    target = output / target_name
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(PinnedQwenSmokeError, match="changed"):
        verify_pinned_qwen_smoke_completion(
            output,
            expected_completion_digest=completion_digest,
        )


def test_completion_verifier_rejects_missing_and_extra_entries(tmp_path: Path) -> None:
    missing_root, missing_digest = _write_synthetic_completed_tree(tmp_path / "missing")
    (missing_root / "episode.json").unlink()
    with pytest.raises(PinnedQwenSmokeError, match="missing or extra"):
        verify_pinned_qwen_smoke_completion(
            missing_root,
            expected_completion_digest=missing_digest,
        )

    extra_root, extra_digest = _write_synthetic_completed_tree(tmp_path / "extra")
    (extra_root / "unattested.json").write_text("{}\n", encoding="ascii")
    with pytest.raises(PinnedQwenSmokeError, match="missing or extra"):
        verify_pinned_qwen_smoke_completion(
            extra_root,
            expected_completion_digest=extra_digest,
        )


def test_completion_verifier_rejects_symlink_hardlink_and_special_file(
    tmp_path: Path,
) -> None:
    symlink_root, symlink_digest = _write_synthetic_completed_tree(tmp_path / "symlink")
    symlink_target = symlink_root / "environment.json"
    symlink_source = tmp_path / "symlink-source.json"
    symlink_source.write_bytes(symlink_target.read_bytes())
    symlink_target.unlink()
    try:
        symlink_target.symlink_to(symlink_source)
    except OSError:
        pytest.skip("symlinks are unavailable on this test filesystem")
    with pytest.raises(PinnedQwenSmokeError, match="symlink"):
        verify_pinned_qwen_smoke_completion(
            symlink_root,
            expected_completion_digest=symlink_digest,
        )

    hardlink_root, hardlink_digest = _write_synthetic_completed_tree(tmp_path / "hardlink")
    hardlink_target = hardlink_root / "environment.json"
    hardlink_source = tmp_path / "hardlink-source.json"
    hardlink_source.write_bytes(hardlink_target.read_bytes())
    hardlink_target.unlink()
    try:
        os.link(hardlink_source, hardlink_target)
    except OSError:
        pytest.skip("hard links are unavailable on this test filesystem")
    with pytest.raises(PinnedQwenSmokeError, match="hard-linked"):
        verify_pinned_qwen_smoke_completion(
            hardlink_root,
            expected_completion_digest=hardlink_digest,
        )

    special_root, special_digest = _write_synthetic_completed_tree(tmp_path / "special")
    try:
        os.mkfifo(special_root / "pipe")
    except (AttributeError, OSError):
        pytest.skip("special files are unavailable on this test filesystem")
    with pytest.raises(PinnedQwenSmokeError, match="special"):
        verify_pinned_qwen_smoke_completion(
            special_root,
            expected_completion_digest=special_digest,
        )


def test_completion_verifier_rejects_noncanonical_or_structurally_tampered_seal(
    tmp_path: Path,
) -> None:
    noncanonical_root, noncanonical_digest = _write_synthetic_completed_tree(tmp_path / "noncanonical")
    attestation_path = noncanonical_root / "completion-attestation.json"
    value = json.loads(attestation_path.read_text(encoding="ascii"))
    attestation_path.write_text(json.dumps(value, indent=2) + "\n", encoding="ascii")
    with pytest.raises(PinnedQwenSmokeError, match="canonical"):
        verify_pinned_qwen_smoke_completion(
            noncanonical_root,
            expected_completion_digest=noncanonical_digest,
        )

    structural_root, structural_digest = _write_synthetic_completed_tree(tmp_path / "structural")
    structural_path = structural_root / "completion-attestation.json"
    structural = json.loads(structural_path.read_text(encoding="ascii"))
    structural["unexpected"] = False
    structural_path.write_text(
        json.dumps(structural, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    with pytest.raises(PinnedQwenSmokeError, match="unexpected structure"):
        verify_pinned_qwen_smoke_completion(
            structural_root,
            expected_completion_digest=structural_digest,
        )


@pytest.mark.parametrize("relative_field", ["artifact", "fixture", "output"])
def test_smoke_config_rejects_every_relative_operational_path_before_creation(
    tmp_path: Path,
    relative_field: str,
) -> None:
    artifact = tmp_path / "model-root"
    fixture = tmp_path / "fixture-root" / "episodes.json"
    output = tmp_path / "output-root"
    if relative_field == "artifact":
        artifact = Path("relative-model")
    elif relative_field == "fixture":
        fixture = Path("relative-fixture.json")
    else:
        output = Path("relative-output")
    with pytest.raises(PinnedQwenSmokeError, match="must be absolute"):
        _config_with_paths(artifact, fixture, output)
    assert not (tmp_path / "output-root").exists()
    assert not (Path.cwd() / "relative-output").exists()


def test_smoke_config_rejects_equal_nested_and_ancestor_output_roots(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "model-root"
    fixture = tmp_path / "fixture-root" / "episodes.json"
    protected_package = Path(smoke_module.__file__).resolve().parent
    outputs = (
        artifact,
        artifact / "nested-output",
        tmp_path,
        fixture.parent,
        fixture.parent / "nested-output",
        protected_package / "never-create-g03-smoke-output",
    )
    for output in outputs:
        with pytest.raises(PinnedQwenSmokeError, match="canonically disjoint"):
            _config_with_paths(artifact, fixture, output)
        assert not (artifact / "nested-output").exists()
        assert not (fixture.parent / "nested-output").exists()
        assert not (protected_package / "never-create-g03-smoke-output").exists()


@pytest.mark.parametrize("relationship", ["equal", "artifact-ancestor", "fixture-ancestor"])
def test_smoke_config_rejects_overlapping_frozen_input_roots(
    tmp_path: Path,
    relationship: str,
) -> None:
    if relationship == "equal":
        artifact = tmp_path / "shared-root"
        fixture = artifact / "episodes.json"
    elif relationship == "artifact-ancestor":
        artifact = tmp_path / "model-root"
        fixture = artifact / "fixture-root" / "episodes.json"
    else:
        fixture_root = tmp_path / "fixture-root"
        artifact = fixture_root / "nested-model"
        fixture = fixture_root / "episodes.json"
    output = tmp_path / "separate-output"
    with pytest.raises(PinnedQwenSmokeError, match="canonically disjoint"):
        _config_with_paths(artifact, fixture, output)
    assert not output.exists()


def test_smoke_config_rejects_model_root_equal_to_actual_package_root(tmp_path: Path) -> None:
    package_root = Path(smoke_module.__file__).resolve().parent
    fixture = tmp_path / "fixture-root" / "episodes.json"
    output = tmp_path / "output-root"
    with pytest.raises(PinnedQwenSmokeError, match="interactive package root"):
        _config_with_paths(package_root, fixture, output)
    assert not output.exists()


def test_smoke_path_policy_resolves_symlinked_parent_aliases(tmp_path: Path) -> None:
    artifact = tmp_path / "model-root"
    artifact.mkdir()
    fixture = tmp_path / "fixture-root" / "episodes.json"
    fixture.parent.mkdir()
    alias = tmp_path / "model-alias"
    try:
        alias.symlink_to(artifact, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable on this test filesystem")
    aliased_output = alias / "nested-output"
    with pytest.raises(PinnedQwenSmokeError, match="canonically disjoint"):
        _config_with_paths(artifact, fixture, aliased_output)
    assert not (artifact / "nested-output").exists()

    aliased_fixture = alias / "episodes.json"
    with pytest.raises(PinnedQwenSmokeError, match="canonically disjoint"):
        _config_with_paths(artifact, aliased_fixture, tmp_path / "separate-output")


def test_smoke_path_policy_accepts_absolute_sibling_roots(tmp_path: Path) -> None:
    artifact = tmp_path / "model-root"
    fixture = tmp_path / "fixture-root" / "episodes.json"
    output = tmp_path / "output-root"
    validate_pinned_qwen_smoke_path_separation(artifact, fixture, output)
    config = _config_with_paths(artifact, fixture, output)
    assert config.output_root == output
    assert not output.exists()


def test_run_revalidates_path_separation_before_mkdir_or_input_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    unsafe_output = config.artifact_root / "nested-output"
    object.__setattr__(config, "output_root", unsafe_output)
    monkeypatch.setattr(
        smoke_module,
        "configure_frozen_cuda_runtime",
        lambda: pytest.fail("runtime configuration must not run after a path-policy failure"),
    )
    monkeypatch.setattr(
        smoke_module,
        "_load_frozen_episode",
        lambda _path: pytest.fail("fixture must not be read after a path-policy failure"),
    )
    monkeypatch.setattr(
        smoke_module,
        "_load_runtime",
        lambda _config: pytest.fail("model must not load after a path-policy failure"),
    )
    with pytest.raises(PinnedQwenSmokeError, match="canonically disjoint"):
        run_pinned_qwen_smoke_cell(config)
    assert not unsafe_output.exists()


def test_config_has_no_scientific_hyperparameter_surface(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assert config.run_seed == 30_201
    assert config.run_id == "sft-seed30201"
    fields = set(config.__dataclass_fields__)
    assert fields == {
        "cell_kind",
        "artifact_root",
        "episode_fixture_path",
        "output_root",
        "expected_artifact_manifest_digest",
        "expected_source_fingerprint",
        "device",
    }
    with pytest.raises(PinnedQwenSmokeError, match="cell_kind"):
        PinnedQwenSmokeCellConfig(
            cell_kind=cast(Any, "sft-tuned"),
            artifact_root=tmp_path / "model",
            episode_fixture_path=tmp_path / "episodes.json",
            output_root=tmp_path / "out",
            expected_artifact_manifest_digest="a" * 64,
            expected_source_fingerprint="b" * 64,
            device="cuda:0",
        )


def test_preflight_binds_artifact_source_and_frozen_fixture_without_model_load(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "config.json").write_text('{"model_type":"qwen2"}\n', encoding="utf-8")
    fixture = Path("tests/goalzendo_interactive/fixtures/g03-engine-small-fixture-v1.json").resolve()
    value = json.loads(build_pinned_qwen_smoke_preflight(artifact, fixture))
    assert len(value["artifact_manifest"]["files"]) == 1
    assert value["interactive_source_provenance"]["fingerprint"]
    assert value["episode_id"] == smoke_module.SMOKE_EPISODE_ID
    assert value["authorization"] == {
        "model_load": False,
        "weight_updates": False,
        "scientific_launch": False,
    }


def _runner_environment() -> dict[str, str]:
    repository_root = Path(__file__).resolve().parents[2]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repository_root / "src")
    return environment


def test_preflight_runner_rejects_overlap_before_hashing_or_writing_and_accepts_siblings(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    script = repository_root / "runs" / "goalzendo" / "04_g03_freeze_smoke_inputs.py"
    artifact = tmp_path / "model-root"
    artifact.mkdir()
    (artifact / "config.json").write_text('{"model_type":"qwen2"}\n', encoding="ascii")
    isolated_fixture = tmp_path / "fixture-root" / "episodes.json"
    isolated_fixture.parent.mkdir()
    isolated_fixture.write_text("not read on overlap\n", encoding="ascii")
    unsafe_output = artifact / "preflight.json"
    rejected = subprocess.run(
        (
            sys.executable,
            str(script),
            "--artifact-root",
            str(artifact),
            "--episode-fixture",
            str(isolated_fixture),
            "--output",
            str(unsafe_output),
        ),
        check=False,
        capture_output=True,
        text=True,
        env=_runner_environment(),
    )
    assert rejected.returncode != 0
    assert "canonically disjoint" in rejected.stderr
    assert not unsafe_output.exists()

    frozen_fixture = (
        repository_root / "tests" / "goalzendo_interactive" / "fixtures" / "g03-engine-small-fixture-v1.json"
    )
    valid_output = tmp_path / "preflight-output" / "preflight.json"
    accepted = subprocess.run(
        (
            sys.executable,
            str(script),
            "--artifact-root",
            str(artifact),
            "--episode-fixture",
            str(frozen_fixture),
            "--output",
            str(valid_output),
        ),
        check=False,
        capture_output=True,
        text=True,
        env=_runner_environment(),
    )
    assert accepted.returncode == 0, accepted.stderr
    assert valid_output.is_file()


def test_smoke_runner_rejects_overlap_before_output_creation_or_model_load(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    script = repository_root / "runs" / "goalzendo" / "04_g03_pinned_qwen_smoke.py"
    artifact = tmp_path / "model-root"
    artifact.mkdir()
    fixture = tmp_path / "fixture-root" / "episodes.json"
    fixture.parent.mkdir()
    fixture.write_text("must not be read\n", encoding="ascii")
    unsafe_output = artifact / "smoke-output"
    rejected = subprocess.run(
        (
            sys.executable,
            str(script),
            "--cell",
            "sft",
            "--artifact-root",
            str(artifact),
            "--episode-fixture",
            str(fixture),
            "--output-root",
            str(unsafe_output),
            "--artifact-manifest-digest",
            "a" * 64,
            "--source-fingerprint",
            "b" * 64,
            "--device",
            "cuda:0",
        ),
        check=False,
        capture_output=True,
        text=True,
        env=_runner_environment(),
    )
    assert rejected.returncode != 0
    assert "canonically disjoint" in rejected.stderr
    assert not unsafe_output.exists()


def _run_materializer(source: Path, destination: Path) -> subprocess.CompletedProcess[str]:
    repository_root = Path(__file__).resolve().parents[2]
    script = repository_root / "runs" / "goalzendo" / "04_g03_materialize_snapshot.sh"
    return subprocess.run(
        ("bash", str(script), str(source), str(destination)),
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("relationship", ["equal", "nested", "ancestor"])
def test_materializer_rejects_equal_nested_and_ancestor_destinations(
    tmp_path: Path,
    relationship: str,
) -> None:
    case_root = tmp_path / relationship
    source = case_root / "source"
    source.mkdir(parents=True)
    if relationship == "equal":
        destination = source
    elif relationship == "nested":
        destination = source / "nested-destination"
    else:
        destination = case_root
    rejected = _run_materializer(source, destination)
    assert rejected.returncode != 0
    assert "canonically disjoint" in rejected.stderr
    if relationship == "nested":
        assert not destination.exists()


def test_materializer_resolves_symlink_alias_and_accepts_valid_sibling(
    tmp_path: Path,
) -> None:
    source = tmp_path / "alias-case" / "source"
    source.mkdir(parents=True)
    alias = tmp_path / "source-alias"
    try:
        alias.symlink_to(source, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable on this test filesystem")
    aliased_destination = alias / "nested-destination"
    rejected = _run_materializer(source, aliased_destination)
    assert rejected.returncode != 0
    assert "canonically disjoint" in rejected.stderr
    assert not (source / "nested-destination").exists()

    valid_source = tmp_path / "valid-case" / "source"
    valid_source.mkdir(parents=True)
    for name in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors",
    ):
        (valid_source / name).write_text(f"{name}\n", encoding="ascii")
    valid_destination = tmp_path / "valid-case" / "destination"
    accepted = _run_materializer(valid_source, valid_destination)
    assert accepted.returncode == 0, accepted.stderr
    assert (valid_destination / "model.safetensors").read_text(encoding="ascii") == ("model.safetensors\n")


def test_success_orchestration_seals_every_emitted_file_without_live_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = interactive_source_provenance()

    class FakeManifest:
        def __init__(self, digest: str) -> None:
            self.digest = digest

        def to_json(self) -> str:
            return json.dumps({"digest": self.digest}, separators=(",", ":"))

    pre = FakeManifest("1" * 64)
    post = FakeManifest("2" * 64)

    class FakeRuntime:
        def __init__(self) -> None:
            self.calls = 0

        def reauthenticate(self) -> FakeManifest:
            self.calls += 1
            return pre if self.calls < 3 else post

    runtime = FakeRuntime()
    episode = SimpleNamespace(
        episode_id=smoke_module.SMOKE_EPISODE_ID,
        digest=smoke_module.SMOKE_EPISODE_DIGEST,
        as_obj=lambda: {"episode_id": smoke_module.SMOKE_EPISODE_ID},
    )
    reference = SimpleNamespace(
        transcript_digest=smoke_module.SMOKE_REFERENCE_TRANSCRIPT_DIGEST,
        as_obj=lambda: {"transcript_digest": smoke_module.SMOKE_REFERENCE_TRANSCRIPT_DIGEST},
    )
    bank = SimpleNamespace(digest=smoke_module.EPISODE_BANK_DIGEST)
    cache = SimpleNamespace(
        digest="3" * 64,
        maximum_absolute_error_hex=(0.0).hex(),
        to_json=lambda: json.dumps({"digest": "3" * 64}, separators=(",", ":")),
    )
    record = SimpleNamespace(
        plan_digest="4" * 64,
        digest="5" * 64,
        pre_policy_state_digest="6" * 64,
        post_policy_state_digest="7" * 64,
        changed_parameter_count=2,
        optimizer_step_call_count=1,
        cuda_peak_allocated_bytes=10,
        cuda_peak_reserved_bytes=20,
        post_runtime_manifest_digest=post.digest,
    )
    bundle = SimpleNamespace(
        record=record,
        digest="8" * 64,
        to_json=lambda: json.dumps({"digest": "8" * 64}, separators=(",", ":")),
    )

    def fake_stack() -> dict[str, object]:
        return {"fake": True}

    def fake_episode(_path: Path) -> tuple[Any, Any, Any]:
        return bank, episode, reference

    def fake_runtime(_config: PinnedQwenSmokeCellConfig) -> Any:
        return runtime

    def fake_cache(_runtime: Any) -> tuple[Any, Any]:
        return cache, object()

    def fake_step(
        _config: PinnedQwenSmokeCellConfig,
        _runtime: Any,
        _incremental: Any,
        _episode: Any,
        _reference: Any,
    ) -> tuple[Any, str, tuple[str, ...], tuple[str, ...], dict[str, int]]:
        return (
            bundle,
            "{}",
            (),
            (),
            {
                "sampling_cache_reuse_count": 0,
                "sampling_cache_reset_count": 0,
            },
        )

    monkeypatch.setattr(smoke_module, "configure_frozen_cuda_runtime", lambda: None)
    monkeypatch.setattr(smoke_module, "_require_frozen_stack", fake_stack)
    monkeypatch.setattr(smoke_module, "_load_frozen_episode", fake_episode)
    monkeypatch.setattr(smoke_module, "_load_runtime", fake_runtime)
    monkeypatch.setattr(smoke_module, "_cache_gate", fake_cache)
    monkeypatch.setattr(smoke_module, "_run_step", fake_step)

    output = tmp_path / "sealed-output"
    config = PinnedQwenSmokeCellConfig(
        cell_kind="sft",
        artifact_root=tmp_path / "model",
        episode_fixture_path=Path(
            "tests/goalzendo_interactive/fixtures/g03-engine-small-fixture-v1.json"
        ).resolve(),
        output_root=output,
        expected_artifact_manifest_digest="a" * 64,
        expected_source_fingerprint=source.fingerprint,
        device="cuda:0",
    )
    completion_digest = run_pinned_qwen_smoke_cell(config)
    assert (output / "COMPLETE").read_text(encoding="utf-8") == completion_digest + "\n"
    attestation = json.loads((output / "completion-attestation.json").read_text(encoding="utf-8"))
    assert attestation["digest"] == completion_digest
    assert {record["path"] for record in attestation["files"]} == {
        "cache-comparison.json",
        "environment.json",
        "episode.json",
        "objective-plan.json",
        "reference-trajectory.json",
        "report.json",
        "runtime-post.json",
        "runtime-pre.json",
        "source-provenance.json",
        "step-evidence.json",
    }
