from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest

ROOT = Path(__file__).resolve().parents[2]
BUILDER_PATH = ROOT / "runs/goalzendo/build_g01_b_source_capsule.py"
STAGER_PATH = ROOT / "runs/goalzendo/g01_b_source_capsule_stage.py"
G01_UUID = "22222222-2222-4222-8222-222222222222"
ELIGIBILITY_UUID = "11111111-1111-4111-8111-111111111111"


def _load(path: Path, name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__file__ = str(path.resolve())
    module.__package__ = ""
    sys.modules[name] = module
    source = path.read_bytes()
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


builder = _load(BUILDER_PATH, "goalzendo_g01_b_capsule_builder_test")
stager = _load(STAGER_PATH, "goalzendo_g01_b_capsule_stager_test")
builder.__dict__["_TEST_ONLY_ALLOW_RUNTIME"] = True
stager.__dict__["_TEST_ONLY_ALLOW_RUNTIME"] = True

LOCKED_ACCEPTED_SHA256 = {
    "src/goalzendo_g01_coordinator/__init__.py": (
        "85a9454fc21c2dc8c70e6ebf9eb605aa0208e6c13e73547ec0f4abdd08abad35"
    ),
    "src/goalzendo_g01_coordinator/coordinator.py": (
        "a3390dad47ea1fd2aa7b1ca22e4475b9a510c47fc85f9b296ed925303663b302"
    ),
    "runs/goalzendo/run_g01_global_coordinator.py": (
        "69f5ec520d8c681d5e9cab1dd9687d7fd39c948f08ff397331bd5ce74f824a06"
    ),
    "docs/goalzendo/protocols/g01-global-coordinator-checkpoint-b.md": (
        "93acc6262758dd5478e342e2f89fd5cb1ab3b61a03bed34b1a6d5083cacaaeba"
    ),
    "configs/goalzendo/g01_known_law.yaml": (
        "ee6d53556189a6b6e25cfbfb204325017a058c63a605653df4c175463e00ce18"
    ),
    "docs/goalzendo/protocols/g01-known-law.md": (
        "428243c3da271bc2d79e16c0fde20d47076eb3837c6740ab733e278cddcdbce9"
    ),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _builder_sha() -> str:
    return _sha(BUILDER_PATH)


def _stager_sha() -> str:
    return _sha(STAGER_PATH)


def _build(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    tmp_path.mkdir(parents=True)
    output = tmp_path.resolve() / "capsule"
    return output, builder.build(ROOT, output, expected_builder_sha256=_builder_sha())


def _roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    values = {
        "input_root": (tmp_path / "inputs").resolve(),
        "preexecution_root": (tmp_path / "preexecution").resolve(),
        "execution_status_root": (tmp_path / "status" / "g01-executions").resolve(),
        "artifact_root": (tmp_path / "artifacts" / "g01-known-law").resolve(),
    }
    for path in (
        values["input_root"],
        values["preexecution_root"],
        values["execution_status_root"],
        values["artifact_root"].parent,
    ):
        path.mkdir(parents=True)
    monkeypatch.setattr(stager, "_TEST_ONLY_ROOTS", values)
    return values


def _kwargs(output: Path, built: dict[str, Any]) -> dict[str, Any]:
    return {
        "g01_execution_uuid": G01_UUID,
        "eligibility_execution_uuid": ELIGIBILITY_UUID,
        "capsule_freeze": output / stager.FREEZE_NAME,
        "expected_capsule_freeze_sha256": built["freeze_sha256"],
        "expected_capsule_archive_sha256": built["archive_sha256"],
        "expected_capsule_manifest_sha256": built["manifest_sha256"],
        "expected_stager_sha256": _stager_sha(),
    }


def _rewrite_archive(
    output: Path,
    transform: Callable[[list[tuple[tarfile.TarInfo, bytes]]], list[tuple[tarfile.TarInfo, bytes]]],
) -> dict[str, Any]:
    archive_path = output / stager.ARCHIVE_NAME
    rows: list[tuple[tarfile.TarInfo, bytes]] = []
    with tarfile.open(archive_path, "r:gz") as archive:
        for original in archive.getmembers():
            info = copy.copy(original)
            extracted = archive.extractfile(original)
            assert extracted is not None
            rows.append((info, extracted.read()))
    buffer = io.BytesIO()
    with (
        gzip.GzipFile(filename="", mode="wb", fileobj=buffer, compresslevel=9, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT) as archive,
    ):
        for info, payload in transform(rows):
            archive.addfile(info, None if info.islnk() or info.issym() else io.BytesIO(payload))
    archive_path.write_bytes(buffer.getvalue())
    archive_path.chmod(0o644)
    freeze_path = output / stager.FREEZE_NAME
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze["bundle"]["archive_sha256"] = _sha(archive_path)
    body = {key: value for key, value in freeze.items() if key != "freeze_digest"}
    freeze["freeze_digest"] = stager.digest(body)
    freeze_path.write_text(json.dumps(freeze, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    freeze_path.chmod(0o644)
    return {"archive_sha256": _sha(archive_path), "freeze_sha256": _sha(freeze_path)}


def _rewrite_freeze(output: Path, mutate: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    freeze_path = output / stager.FREEZE_NAME
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    mutate(freeze)
    body = {key: value for key, value in freeze.items() if key != "freeze_digest"}
    freeze["freeze_digest"] = stager.digest(body)
    freeze_path.write_text(json.dumps(freeze, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    freeze_path.chmod(0o644)
    return {"freeze_sha256": _sha(freeze_path)}


def _rewrite_manifest_and_freeze(output: Path, mutate: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    manifest_path = output / stager.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(manifest)
    manifest_body = {key: value for key, value in manifest.items() if key != "manifest_digest"}
    manifest["manifest_digest"] = stager.digest(manifest_body)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    manifest_path.chmod(0o644)
    freeze_path = output / stager.FREEZE_NAME
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze["bundle"]["manifest_sha256"] = _sha(manifest_path)
    freeze["bundle"]["manifest_digest"] = manifest["manifest_digest"]
    freeze_body = {key: value for key, value in freeze.items() if key != "freeze_digest"}
    freeze["freeze_digest"] = stager.digest(freeze_body)
    freeze_path.write_text(json.dumps(freeze, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    freeze_path.chmod(0o644)
    return {"manifest_sha256": _sha(manifest_path), "freeze_sha256": _sha(freeze_path)}


def _compact_reversed_json(value: dict[str, Any]) -> str:
    reordered = {key: value[key] for key in reversed(tuple(value))}
    return json.dumps(reordered, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def test_accepted_inputs_are_unchanged_and_no_generated_cache() -> None:
    assert {relative: _sha(ROOT / relative) for relative in LOCKED_ACCEPTED_SHA256} == (
        LOCKED_ACCEPTED_SHA256
    )
    cache = ROOT / "runs/goalzendo/__pycache__"
    assert not list(cache.glob("build_g01_b_source_capsule.*.pyc"))
    assert not list(cache.glob("g01_b_source_capsule_stage.*.pyc"))


def test_builder_is_deterministic_exact_34_and_canonical_cli_is_held(tmp_path: Path) -> None:
    first, result1 = _build(tmp_path / "first")
    second, result2 = _build(tmp_path / "second")
    for name in (builder.ARCHIVE_NAME, builder.MANIFEST_NAME, builder.FREEZE_NAME):
        assert (first / name).read_bytes() == (second / name).read_bytes()
        assert (first / name).stat().st_mode & 0o777 == 0o644
    assert first.stat().st_mode & 0o777 == 0o755
    assert result1 == {**result2, "output": str(first)}
    with tarfile.open(first / builder.ARCHIVE_NAME, "r:gz") as archive:
        infos = archive.getmembers()
    assert len(infos) == 34
    assert [info.name for info in infos] == list(stager.PAYLOAD_PATHS)
    assert all(info.isreg() and info.mode & 0o777 == 0o444 and not info.pax_headers for info in infos)
    freeze = json.loads((first / builder.FREEZE_NAME).read_text(encoding="utf-8"))
    assert freeze["generic_execution_uuid"] is None
    assert freeze["authorization"] == stager.AUTHORIZATION
    assert freeze["trust_limitations"] == stager.TRUST_LIMITATIONS
    with pytest.raises(builder.BuildError, match="BUILD_HELD"):
        builder.build(
            ROOT,
            ROOT / builder.DEFAULT_OUTPUT,
            expected_builder_sha256=_builder_sha(),
        )
    process = subprocess.run([sys.executable, str(BUILDER_PATH)], capture_output=True, text=True)
    assert process.returncode == 2
    assert "CANONICAL_G01_B_SOURCE_CAPSULE_BUILD_HELD" in process.stderr


def test_builder_holds_canonical_parent_identity_against_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = os.open(tmp_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    second = os.dup(first)
    other_path = tmp_path / "other"
    other_path.mkdir()
    other = os.open(other_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        assert builder._same_directory_identity(first, second) is True
        assert builder._same_directory_identity(first, other) is False
    finally:
        os.close(other)
        os.close(second)
        os.close(first)

    alias_parent = tmp_path / "alias-parent"
    alias_parent.mkdir()
    target = alias_parent / Path(builder.DEFAULT_OUTPUT).name
    monkeypatch.setattr(builder, "_same_directory_identity", lambda _left, _right: True)
    with pytest.raises(builder.BuildError, match="BUILD_HELD"):
        builder.build(ROOT, target, expected_builder_sha256=_builder_sha())
    assert not target.exists()


def test_builder_and_stage_are_independent_of_restrictive_umask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build_parent = tmp_path / "build"
    build_parent.mkdir()
    output = build_parent.resolve() / "capsule"
    roots = _roots(tmp_path / "remote", monkeypatch)
    previous_umask = os.umask(0o777)
    observed_umask: int | None = None
    try:
        built = builder.build(ROOT, output, expected_builder_sha256=_builder_sha())
        result = stager.stage(**_kwargs(output, built))
    finally:
        observed_umask = os.umask(previous_umask)
    assert observed_umask == 0o777
    assert output.stat().st_mode & 0o777 == 0o755
    assert all(path.stat().st_mode & 0o777 == 0o644 for path in output.iterdir())
    input_execution = roots["input_root"] / G01_UUID
    frozen = input_execution / "frozen-source"
    receipt_parent = roots["preexecution_root"] / G01_UUID
    receipt = receipt_parent / stager.RECEIPT_NAME
    assert input_execution.stat().st_mode & 0o777 == 0o555
    assert frozen.stat().st_mode & 0o777 == 0o555
    assert receipt_parent.stat().st_mode & 0o777 == 0o700
    assert receipt.stat().st_mode & 0o777 == 0o400
    assert result["verified"] is True


def test_stage_and_fresh_verify_exact_tree_and_nonauthority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, built = _build(tmp_path / "build")
    roots = _roots(tmp_path / "remote", monkeypatch)
    result = stager.stage(**_kwargs(output, built))
    assert result["verified"] is True
    assert result["g01_launch_authorized"] is False
    assert result["supported_launch_entrypoint_activated"] is False
    frozen = roots["input_root"] / G01_UUID / "frozen-source"
    actual_files = sorted(path.relative_to(frozen).as_posix() for path in frozen.rglob("*") if path.is_file())
    assert actual_files == list(stager.PAYLOAD_PATHS)
    assert all(path.stat().st_mode & 0o777 == 0o444 for path in frozen.rglob("*") if path.is_file())
    assert all(path.stat().st_mode & 0o777 == 0o555 for path in frozen.rglob("*") if path.is_dir())
    receipt = Path(result["receipt_path"])
    assert receipt.stat().st_mode & 0o777 == 0o400
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["uuid_relation"] == {
        "distinct": True,
        "eligibility_uuid_source": "operator_supplied_compare_only",
        "eligibility_uuid_authenticated_by_source_stage": False,
        "thin_token_read_by_source_stage": False,
    }
    assert payload["transaction"]["canonical_checkpoint_a_path_or_token_opened"] is False
    assert not (roots["execution_status_root"] / G01_UUID).exists()
    assert not roots["artifact_root"].exists()
    verified = stager.verify_stage(
        **_kwargs(output, built), expected_stage_receipt_sha256=result["receipt_sha256"]
    )
    assert verified == result
    with pytest.raises(stager.StageError, match="one-shot"):
        stager.stage(**_kwargs(output, built))


@pytest.mark.parametrize(
    "spelling",
    [
        "/workspace/status-goalzendo/g00f-executions/source",
        "/workspace/status-goalzendo/g01-preexecution/../g00f-executions/source",
        "//workspace/status-goalzendo/g00f-executions/source",
    ],
)
def test_checkpoint_a_caller_paths_reject_before_application_fs_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    spelling: str,
) -> None:
    probes: list[str] = []
    monkeypatch.setattr(builder, "_repo", lambda _path: probes.append("builder_repo"))
    monkeypatch.setattr(stager, "_verify_running_stager", lambda _sha: probes.append("stager_self"))
    with pytest.raises(builder.BuildError, match="checkpoint-A"):
        builder.build(spelling, tmp_path / "out", expected_builder_sha256=_builder_sha())
    with pytest.raises(builder.BuildError, match="forbidden canonical scientific"):
        builder.build(ROOT, spelling, expected_builder_sha256=_builder_sha())
    kwargs = _kwargs(
        tmp_path, {"freeze_sha256": "a" * 64, "archive_sha256": "b" * 64, "manifest_sha256": "c" * 64}
    )
    kwargs["capsule_freeze"] = spelling
    with pytest.raises(stager.StageError, match="checkpoint-A"):
        stager.stage(**kwargs)
    with pytest.raises(stager.StageError, match="checkpoint-A"):
        stager.verify_stage(**kwargs, expected_stage_receipt_sha256="d" * 64)
    assert probes == []


@pytest.mark.parametrize(
    "root",
    [
        "/workspace/inputs-goalzendo/g01-executions",
        "/workspace/status-goalzendo/g01-preexecution",
        "/workspace/status-goalzendo/g01-executions",
        "/workspace/artifacts-goalzendo/g01-known-law",
    ],
)
@pytest.mark.parametrize("prefix", ["", "/"])
def test_builder_output_rejects_every_g01_transaction_namespace_before_repo_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    root: str,
    prefix: str,
) -> None:
    probes: list[str] = []
    monkeypatch.setattr(builder, "_repo", lambda _path: probes.append("repo"))
    spelling = f"{prefix}{root}/descendant"
    with pytest.raises(builder.BuildError, match="forbidden canonical scientific"):
        builder.build(ROOT, spelling, expected_builder_sha256=_builder_sha())
    assert probes == []


def test_running_program_checkpoint_a_guard_precedes_self_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, built = _build(tmp_path / "build")
    probes: list[str] = []
    original_lexical = stager._lexical_absolute

    def fake_lexical(value: str | Path, label: str) -> Path:
        if label == "running stager":
            return Path("/workspace/status-goalzendo/g00f-executions/fake/stager.py")
        return cast(Path, original_lexical(value, label))

    monkeypatch.setattr(stager, "_lexical_absolute", fake_lexical)
    monkeypatch.setattr(stager, "_verify_running_stager", lambda _sha: probes.append("self-open"))
    with pytest.raises(stager.StageError, match="checkpoint-A"):
        stager.stage(**_kwargs(output, built))
    assert probes == []

    builder_probes: list[str] = []
    original_builder_lexical = builder._lexical_absolute

    def fake_builder_lexical(value: str | Path, label: str) -> Path:
        if label == "running builder":
            return Path("/workspace/status-goalzendo/g00f-executions/fake/builder.py")
        return cast(Path, original_builder_lexical(value, label))

    monkeypatch.setattr(builder, "_lexical_absolute", fake_builder_lexical)
    monkeypatch.setattr(builder, "_repo", lambda _path: builder_probes.append("repo"))
    with pytest.raises(builder.BuildError, match="checkpoint-A"):
        builder.build(ROOT, tmp_path / "other", expected_builder_sha256=_builder_sha())
    assert builder_probes == []


def test_nul_paths_reject_before_application_fs_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probes: list[str] = []
    monkeypatch.setattr(builder, "_repo", lambda _path: probes.append("builder_repo"))
    monkeypatch.setattr(stager, "_verify_running_stager", lambda _sha: probes.append("stager_self"))
    with pytest.raises(builder.BuildError, match="NUL-free"):
        builder.build("bad\0repo", tmp_path / "out", expected_builder_sha256=_builder_sha())
    with pytest.raises(builder.BuildError, match="NUL-free"):
        builder.build(ROOT, "bad\0output", expected_builder_sha256=_builder_sha())
    kwargs = _kwargs(
        tmp_path, {"freeze_sha256": "a" * 64, "archive_sha256": "b" * 64, "manifest_sha256": "c" * 64}
    )
    kwargs["capsule_freeze"] = "bad\0capsule"
    with pytest.raises(stager.StageError, match="NUL-free"):
        stager.stage(**kwargs)
    assert probes == []


@pytest.mark.parametrize("kind", ["extra", "link", "traversal", "duplicate", "mode"])
def test_archive_rejects_nonexact_or_unsafe_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    output, built = _build(tmp_path / "build")
    _roots(tmp_path / "remote", monkeypatch)

    def alter(rows: list[tuple[tarfile.TarInfo, bytes]]) -> list[tuple[tarfile.TarInfo, bytes]]:
        if kind == "extra":
            info = tarfile.TarInfo("extra")
            info.size = 1
            info.mode = 0o444
            info.mtime = stager.SOURCE_DATE_EPOCH
            rows.append((info, b"x"))
        elif kind == "link":
            rows[0][0].type = tarfile.SYMTYPE
            rows[0][0].linkname = "pyproject.toml"
        elif kind == "traversal":
            rows[0][0].name = "../escape"
        elif kind == "duplicate":
            rows.append((copy.copy(rows[0][0]), rows[0][1]))
        elif kind == "mode":
            rows[0][0].mode = 0o644
        return rows

    changed = _rewrite_archive(output, alter)
    built.update(changed)
    with pytest.raises(stager.StageError):
        stager.stage(**_kwargs(output, built))


def test_stage_rejects_extra_capsule_artifact_and_equal_uuid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, built = _build(tmp_path / "build")
    _roots(tmp_path / "remote", monkeypatch)
    (output / "extra").write_bytes(b"x")
    with pytest.raises(stager.StageError, match="exact three-file"):
        stager.stage(**_kwargs(output, built))
    (output / "extra").unlink()
    equal = _kwargs(output, built)
    equal["eligibility_execution_uuid"] = G01_UUID
    with pytest.raises(stager.StageError, match="must be distinct"):
        stager.stage(**equal)


def test_capsule_artifact_parent_requires_exact_0755_at_capture_and_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rejected_output, rejected_built = _build(tmp_path / "rejected")
    rejected_roots = _roots(tmp_path / "remote-rejected", monkeypatch)
    rejected_output.chmod(0o777)
    with pytest.raises(stager.StageError, match="mode must be exactly 0755"):
        stager.stage(**_kwargs(rejected_output, rejected_built))
    assert not (rejected_roots["input_root"] / G01_UUID).exists()

    output, built = _build(tmp_path / "replay")
    _roots(tmp_path / "remote-replay", monkeypatch)
    result = stager.stage(**_kwargs(output, built))
    snapshot = stager._ArtifactSnapshot.capture(output / stager.FREEZE_NAME)
    try:
        output.chmod(0o777)
        with pytest.raises(stager.StageError, match="changed from exact 0755"):
            snapshot.verify()
    finally:
        snapshot.close()
    with pytest.raises(stager.StageError, match="mode must be exactly 0755"):
        stager.verify_stage(
            **_kwargs(output, built),
            expected_stage_receipt_sha256=result["receipt_sha256"],
        )


def test_stage_rejects_status_or_artifact_presence_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, built = _build(tmp_path / "build")
    roots = _roots(tmp_path / "remote", monkeypatch)
    (roots["execution_status_root"] / G01_UUID).mkdir()
    with pytest.raises(stager.StageError, match="must be absent"):
        stager.stage(**_kwargs(output, built))
    assert not (roots["input_root"] / G01_UUID).exists()
    (roots["execution_status_root"] / G01_UUID).rmdir()
    roots["artifact_root"].mkdir()
    with pytest.raises(stager.StageError, match="must be absent"):
        stager.stage(**_kwargs(output, built))
    assert not (roots["input_root"] / G01_UUID).exists()


def test_stage_rejects_aliased_canonical_roots_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, built = _build(tmp_path / "build")
    roots = _roots(tmp_path / "remote", monkeypatch)
    roots["preexecution_root"] = roots["input_root"]
    monkeypatch.setattr(stager, "_TEST_ONLY_ROOTS", roots)
    with pytest.raises(stager.StageError, match="four distinct"):
        stager.stage(**_kwargs(output, built))
    assert not (roots["input_root"] / G01_UUID).exists()


def test_exact_json_types_reject_bool_for_integer_and_numeric_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, built = _build(tmp_path / "bool")
    _roots(tmp_path / "remote-bool", monkeypatch)
    built.update(_rewrite_freeze(output, lambda value: value.__setitem__("schema_version", True)))
    with pytest.raises(stager.StageError, match="exact integer"):
        stager.stage(**_kwargs(output, built))

    output2, built2 = _build(tmp_path / "hash")
    _roots(tmp_path / "remote-hash", monkeypatch)
    built2.update(
        _rewrite_freeze(
            output2,
            lambda value: value["controller_files"]["builder"].__setitem__("sha256", 123),
        )
    )
    with pytest.raises(stager.StageError, match="SHA-256 string"):
        stager.stage(**_kwargs(output2, built2))
    with pytest.raises(builder.BuildError, match="SHA-256 string"):
        builder.build(ROOT, tmp_path / "builder-hash", expected_builder_sha256=123)


def test_exact_nested_authorization_types_reject_false_to_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, built = _build(tmp_path / "manifest")
    _roots(tmp_path / "remote-manifest", monkeypatch)
    built.update(
        _rewrite_manifest_and_freeze(
            output,
            lambda value: value["authorization"].__setitem__("g01_launch_authorized", 0),
        )
    )
    with pytest.raises(stager.StageError, match="JSON type changed"):
        stager.stage(**_kwargs(output, built))

    output2, built2 = _build(tmp_path / "freeze")
    _roots(tmp_path / "remote-freeze", monkeypatch)
    built2.update(
        _rewrite_freeze(
            output2,
            lambda value: value["authorization"].__setitem__("g01_launch_authorized", 0),
        )
    )
    with pytest.raises(stager.StageError, match="JSON type changed"):
        stager.stage(**_kwargs(output2, built2))


def test_exact_receipt_authorization_type_rejects_false_to_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, built = _build(tmp_path / "build")
    roots = _roots(tmp_path / "remote", monkeypatch)
    result = stager.stage(**_kwargs(output, built))
    receipt = Path(result["receipt_path"])
    receipt.chmod(0o600)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["authorization"]["g01_launch_authorized"] = 0
    body = {key: value for key, value in payload.items() if key != "receipt_digest"}
    payload["receipt_digest"] = stager.digest(body)
    receipt.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    receipt.chmod(0o400)
    receipt_sha = _sha(receipt)
    with pytest.raises(stager.StageError, match="JSON type changed"):
        stager.verify_stage(**_kwargs(output, built), expected_stage_receipt_sha256=receipt_sha)
    assert not roots["artifact_root"].exists()


def test_manifest_and_freeze_require_exact_canonical_json_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_output, manifest_built = _build(tmp_path / "manifest")
    _roots(tmp_path / "remote-manifest", monkeypatch)
    manifest_path = manifest_output / stager.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_path.write_text(_compact_reversed_json(manifest), encoding="utf-8")
    manifest_path.chmod(0o644)
    manifest_built.update(
        _rewrite_freeze(
            manifest_output,
            lambda value: value["bundle"].__setitem__("manifest_sha256", _sha(manifest_path)),
        )
    )
    manifest_built["manifest_sha256"] = _sha(manifest_path)
    with pytest.raises(stager.StageError, match="manifest is not the exact canonical"):
        stager.stage(**_kwargs(manifest_output, manifest_built))

    freeze_output, freeze_built = _build(tmp_path / "freeze")
    _roots(tmp_path / "remote-freeze", monkeypatch)
    freeze_path = freeze_output / stager.FREEZE_NAME
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze_path.write_text(_compact_reversed_json(freeze), encoding="utf-8")
    freeze_path.chmod(0o644)
    freeze_built["freeze_sha256"] = _sha(freeze_path)
    with pytest.raises(stager.StageError, match="freeze is not the exact canonical"):
        stager.stage(**_kwargs(freeze_output, freeze_built))


def test_receipt_requires_exact_canonical_json_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output, built = _build(tmp_path / "build")
    _roots(tmp_path / "remote", monkeypatch)
    result = stager.stage(**_kwargs(output, built))
    receipt = Path(result["receipt_path"])
    receipt.chmod(0o600)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    receipt.write_text(_compact_reversed_json(payload), encoding="utf-8")
    receipt.chmod(0o400)
    with pytest.raises(stager.StageError, match="receipt is not the exact canonical"):
        stager.verify_stage(**_kwargs(output, built), expected_stage_receipt_sha256=_sha(receipt))


def test_fresh_verify_rejects_extra_source_or_receipt_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, built = _build(tmp_path / "build")
    roots = _roots(tmp_path / "remote", monkeypatch)
    result = stager.stage(**_kwargs(output, built))
    frozen = roots["input_root"] / G01_UUID / "frozen-source"
    source_dir = frozen / "src/goalzendo"
    source_dir.chmod(0o755)
    (source_dir / "extra.py").write_bytes(b"x")
    source_dir.chmod(0o555)
    with pytest.raises(stager.StageError, match="inventory"):
        stager.verify_stage(**_kwargs(output, built), expected_stage_receipt_sha256=result["receipt_sha256"])


def test_cli_surface_has_no_root_hardware_runtime_deadline_or_authority_options() -> None:
    source = STAGER_PATH.read_text(encoding="utf-8")
    tree = __import__("ast").parse(source)
    options = {
        node.args[0].value
        for node in __import__("ast").walk(tree)
        if isinstance(node, __import__("ast").Call)
        and isinstance(node.func, __import__("ast").Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], __import__("ast").Constant)
        and isinstance(node.args[0].value, str)
    }
    assert options == {
        "--g01-execution-uuid",
        "--eligibility-execution-uuid",
        "--capsule-freeze",
        "--expected-capsule-freeze-sha256",
        "--expected-capsule-archive-sha256",
        "--expected-capsule-manifest-sha256",
        "--expected-stager-sha256",
        "--expected-stage-receipt-sha256",
    }
    assert not options & {
        "--root",
        "--output",
        "--gpu",
        "--hardware",
        "--runtime",
        "--deadline",
        "--worker-count",
        "--authorize",
    }


def test_runtime_bypass_and_root_override_refuse_outside_pytest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    with pytest.raises(builder.BuildError, match="outside pytest"):
        builder._require_runtime()
    with pytest.raises(stager.StageError, match="outside pytest"):
        stager._require_runtime()
    monkeypatch.setattr(stager, "_TEST_ONLY_ROOTS", {"x": ROOT})
    with pytest.raises(stager.StageError, match="outside pytest"):
        stager._roots()


def test_cli_positive_dispatch_with_exact_runtime_mock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    output, built = _build(tmp_path / "build")
    _roots(tmp_path / "remote", monkeypatch)
    arguments = argparse.Namespace(
        command="stage",
        g01_execution_uuid=G01_UUID,
        eligibility_execution_uuid=ELIGIBILITY_UUID,
        capsule_freeze=output / stager.FREEZE_NAME,
        expected_capsule_freeze_sha256=built["freeze_sha256"],
        expected_capsule_archive_sha256=built["archive_sha256"],
        expected_capsule_manifest_sha256=built["manifest_sha256"],
        expected_stager_sha256=_stager_sha(),
        expected_stage_receipt_sha256=None,
    )
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", lambda _self: arguments)
    assert stager.main() == 0
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["verified"] is True
