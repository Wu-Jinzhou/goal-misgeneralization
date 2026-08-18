from __future__ import annotations

import copy
import gzip
import hashlib
import io
import json
import os
import shutil
import stat
import subprocess
import sys
import tarfile
import types
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
CONTROLLER = ROOT / "runs/goalzendo/g00f_g01_build_input_transport.py"


def _load() -> types.ModuleType:
    module = types.ModuleType("g00f_g01_build_input_transport_test")
    module.__file__ = str(CONTROLLER)
    source = CONTROLLER.read_bytes()
    exec(compile(source, str(CONTROLLER), "exec"), module.__dict__)
    return module


transport = _load()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "source-repo"
    repo.mkdir(mode=0o755, parents=True)
    for relative in transport.DIRECTORIES:
        target = repo / relative
        target.mkdir(mode=0o755)
        target.chmod(0o755)
    for relative, (mode, _, _) in transport.FILES.items():
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
        target.chmod(mode)
    return repo


def _built(
    tmp_path: Path,
    name: str = "transport",
    *,
    repo: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    source_repo = repo if repo is not None else _source_repo(tmp_path / f"{name}-source")
    output = tmp_path / name
    return output, transport.build(source_repo, output)


def _verify_kwargs(output: Path, built: dict[str, Any]) -> dict[str, Any]:
    return {
        "archive": output / transport.ARCHIVE_NAME,
        "manifest": output / transport.MANIFEST_NAME,
        "intent": output / transport.INTENT_NAME,
        "expected_archive_sha256": built["archive_sha256"],
        "expected_manifest_sha256": built["manifest_sha256"],
        "expected_intent_sha256": built["intent_sha256"],
        "expected_controller_sha256": built["transport_controller_sha256"],
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o644)


def _rewrite_archive(
    path: Path,
    transform: Any,
) -> None:
    rows: list[tuple[tarfile.TarInfo, bytes]] = []
    with tarfile.open(path, "r:gz") as archive:
        for original in archive.getmembers():
            info = copy.copy(original)
            handle = archive.extractfile(original)
            assert handle is not None
            rows.append((info, handle.read()))
    rows = transform(rows)
    output = io.BytesIO()
    with (
        gzip.GzipFile(filename="", mode="wb", fileobj=output, compresslevel=9, mtime=0) as zipped,
        tarfile.open(fileobj=zipped, mode="w", format=tarfile.USTAR_FORMAT) as archive,
    ):
        for info, payload in rows:
            archive.addfile(info, io.BytesIO(payload))
    path.write_bytes(output.getvalue())
    path.chmod(0o644)


def test_byte_equal_build_exact_inventory_and_nonauthorization(tmp_path: Path) -> None:
    first, first_result = _built(tmp_path, "first")
    second, second_result = _built(tmp_path, "second")
    assert first_result == second_result
    for name in (transport.ARCHIVE_NAME, transport.MANIFEST_NAME, transport.INTENT_NAME):
        assert (first / name).read_bytes() == (second / name).read_bytes()
        assert stat.S_IMODE((first / name).stat().st_mode) == 0o644
    with tarfile.open(first / transport.ARCHIVE_NAME, "r:gz") as archive:
        members = archive.getmembers()
    assert len(members) == 25
    assert [member.name for member in members] == list(transport.FILES)
    assert all(member.type == tarfile.REGTYPE and member.isfile() for member in members)
    manifest = json.loads((first / transport.MANIFEST_NAME).read_text(encoding="utf-8"))
    intent = json.loads((first / transport.INTENT_NAME).read_text(encoding="utf-8"))
    assert len(manifest["members"]) == 25
    assert len(manifest["required_directories"]) == 14
    assert manifest["authorization"] == transport.AUTHORIZATION
    assert intent["authorization"] == transport.AUTHORIZATION
    assert intent["registration"]["not_sufficient_for_scientific_itt"] is True
    assert intent["registration"]["durable_independent_registrar_required_before_g00f_itt"] is True
    assert not ((tmp_path / "first-source/source-repo") / transport.CANONICAL_BUNDLE_RELATIVE).exists()


def test_verify_and_extract_replay_exact_registered_bytes(tmp_path: Path) -> None:
    output, built = _built(tmp_path)
    verified = transport.verify(**_verify_kwargs(output, built))
    assert verified["member_count"] == 25
    extracted = tmp_path / "repo"
    result = transport.extract(**_verify_kwargs(output, built), output=extracted)
    assert result["exact_inventory_verified"] is True
    assert sorted(
        path.relative_to(extracted).as_posix() for path in extracted.rglob("*") if path.is_file()
    ) == list(transport.FILES)
    assert not (extracted / transport.CANONICAL_BUNDLE_RELATIVE).exists()
    for relative, (mode, size, expected_sha) in transport.FILES.items():
        target = extracted / relative
        assert stat.S_IMODE(target.stat().st_mode) == mode
        assert target.stat().st_size == size
        assert _sha(target) == expected_sha


@pytest.mark.parametrize("field", ["archive", "manifest", "intent"])
def test_external_sha_mismatch_fails(field: str, tmp_path: Path) -> None:
    output, built = _built(tmp_path)
    kwargs = _verify_kwargs(output, built)
    kwargs[f"expected_{field}_sha256"] = "0" * 64
    with pytest.raises(transport.TransportError, match="external pin"):
        transport.verify(**kwargs)


def test_controller_external_sha_mismatch_fails(tmp_path: Path) -> None:
    output, built = _built(tmp_path)
    kwargs = _verify_kwargs(output, built)
    kwargs["expected_controller_sha256"] = "0" * 64
    with pytest.raises(transport.TransportError, match="running transport controller"):
        transport.verify(**kwargs)


def test_manifest_duplicate_key_and_semantic_reseal_fail(tmp_path: Path) -> None:
    output, built = _built(tmp_path)
    manifest = output / transport.MANIFEST_NAME
    manifest.write_bytes(manifest.read_bytes().replace(b'{\n  "', b'{\n  "schema":"shadow",\n  "', 1))
    kwargs = _verify_kwargs(output, built)
    kwargs["expected_manifest_sha256"] = _sha(manifest)
    with pytest.raises(transport.TransportError, match="duplicate key"):
        transport.verify(**kwargs)

    shutil.rmtree(output)
    output, built = _built(tmp_path, "reseal")
    intent_path = output / transport.INTENT_NAME
    intent = json.loads(intent_path.read_text(encoding="utf-8"))
    intent["authorization"]["g01_launch_authorized"] = True
    body = {key: value for key, value in intent.items() if key != "intent_digest"}
    intent["intent_digest"] = transport._digest(body)
    _write_json(intent_path, intent)
    kwargs = _verify_kwargs(output, built)
    kwargs["expected_intent_sha256"] = _sha(intent_path)
    with pytest.raises(transport.TransportError, match="exact prospective intent"):
        transport.verify(**kwargs)


@pytest.mark.parametrize("mutation", ["duplicate", "traversal", "link", "mode", "extra"])
def test_archive_structural_attacks_fail(mutation: str, tmp_path: Path) -> None:
    output, built = _built(tmp_path)
    archive_path = output / transport.ARCHIVE_NAME

    def mutate(rows: list[tuple[tarfile.TarInfo, bytes]]) -> list[tuple[tarfile.TarInfo, bytes]]:
        if mutation == "duplicate":
            return [*rows, copy.deepcopy(rows[0])]
        if mutation == "extra":
            info = tarfile.TarInfo("extra.txt")
            info.type = tarfile.REGTYPE
            info.mode = 0o644
            info.size = 1
            info.uid = info.gid = 0
            info.mtime = transport.SOURCE_DATE_EPOCH
            return [*rows, (info, b"x")]
        info, payload = rows[0]
        info = copy.copy(info)
        if mutation == "traversal":
            info.name = "../escape"
        elif mutation == "link":
            info.type = tarfile.SYMTYPE
            info.linkname = "target"
        elif mutation == "mode":
            info.mode = 0o600
        return [(info, payload), *rows[1:]]

    _rewrite_archive(archive_path, mutate)
    kwargs = _verify_kwargs(output, built)
    kwargs["expected_archive_sha256"] = _sha(archive_path)
    with pytest.raises(transport.TransportError):
        transport.verify(**kwargs)


def test_source_mode_drift_and_output_reuse_fail(tmp_path: Path) -> None:
    source_transport, source_built = _built(tmp_path, "source-transport")
    repo = tmp_path / "repo"
    transport.extract(**_verify_kwargs(source_transport, source_built), output=repo)
    source = repo / "configs/goalzendo/g01_known_law.yaml"
    source.chmod(0o600)
    with pytest.raises(transport.TransportError, match="accepted input changed"):
        transport.build(repo, tmp_path / "bad")
    output, _ = _built(tmp_path, "first")
    source_repo = _source_repo(tmp_path / "reuse-source")
    with pytest.raises(transport.TransportError, match="wholly absent"):
        transport.build(source_repo, output)


def test_transport_cannot_claim_canonical_bundle_subtree(tmp_path: Path) -> None:
    source_transport, source_built = _built(tmp_path, "source-transport")
    repo = tmp_path / "repo"
    transport.extract(**_verify_kwargs(source_transport, source_built), output=repo)
    canonical = repo / transport.CANONICAL_BUNDLE_RELATIVE
    with pytest.raises(transport.TransportError, match="outside the canonical bridge bundle"):
        transport.build(repo, canonical)
    with pytest.raises(transport.TransportError, match="outside the canonical bridge bundle"):
        transport.build(repo, canonical / "nested")
    assert not canonical.exists()


def test_extract_target_reuse_and_partial_tree_fail(tmp_path: Path) -> None:
    output, built = _built(tmp_path)
    target = tmp_path / "existing"
    target.mkdir()
    with pytest.raises(transport.TransportError, match="wholly absent"):
        transport.extract(**_verify_kwargs(output, built), output=target)


def test_locked_inputs_remain_exact() -> None:
    assert len(transport.FILES) == 25
    assert transport.FILES["runs/goalzendo/build_g00f_g01_bridge_bundle.py"][2] == (
        "29c906cdfc6c50f672a0819ece3d007138382099dea26e975715e7619e65b243"
    )
    assert transport.FILES["runs/goalzendo/g00f_g01_bridge_bundle_stage.py"][2] == (
        "58c0b75a4608a0b5cd6a85d2ae1b7b5a13af080fc8f9a6305151cca851963b49"
    )
    for relative, (mode, size, expected_sha) in transport.FILES.items():
        path = ROOT / relative
        assert not path.is_symlink()
        assert path.stat().st_nlink == 1
        assert stat.S_IMODE(path.stat().st_mode) == mode
        assert path.stat().st_size == size
        assert _sha(path) == expected_sha


def test_extracted_repo_has_no_links_or_special_files(tmp_path: Path) -> None:
    output, built = _built(tmp_path)
    target = tmp_path / "repo"
    transport.extract(**_verify_kwargs(output, built), output=target)
    for path in target.rglob("*"):
        metadata = path.lstat()
        assert stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode)
        if stat.S_ISREG(metadata.st_mode):
            assert metadata.st_nlink == 1
    assert os.path.commonpath([target.resolve(), (target / "src").resolve()]) == str(target.resolve())


def test_cli_requires_isolated_python() -> None:
    refused = subprocess.run(
        [sys.executable, str(CONTROLLER), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode == 2
    assert "requires isolated Python (-I)" in refused.stderr

    accepted = subprocess.run(
        [sys.executable, "-I", str(CONTROLLER), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert accepted.returncode == 0
    assert "{build,verify,extract}" in accepted.stdout


def test_verifier_accepts_historical_producer_runtime(tmp_path: Path, monkeypatch: Any) -> None:
    output, built = _built(tmp_path)
    monkeypatch.setattr(
        transport,
        "_build_runtime",
        lambda: {
            "implementation": "CPython",
            "python_version": "3.12.3",
            "platform_system": "Linux",
            "platform_machine": "x86_64",
            "byteorder": "little",
            "zlib_build_version": "1.3",
            "zlib_runtime_version": "1.3",
            "trust_boundary": "different verifier runtime must not replace producer history",
        },
    )
    assert transport.verify(**_verify_kwargs(output, built))["member_count"] == 25
