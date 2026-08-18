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
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
BUILDER_PATH = ROOT / "runs/goalzendo/build_g00f_g01_bridge_bundle.py"
STAGER_PATH = ROOT / "runs/goalzendo/g00f_g01_bridge_bundle_stage.py"


def _load_source(path: Path, name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__file__ = str(path.resolve())
    module.__package__ = ""
    source = path.read_bytes()
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


builder = _load_source(BUILDER_PATH, "goalzendo_bridge_bundle_builder_test")
stager = _load_source(STAGER_PATH, "goalzendo_bridge_bundle_stager_test")

TEST_RUNTIME_IDENTITY = {
    "trusted_image": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
    "trusted_image_claim_source": "fixed_route_freeze_contract_and_external_trust_assumption",
    "determinism_claim": "deterministic_given_frozen_build_runtime_trust_boundary",
    "python_executable": "/workspace/.venvs/goalzendo/bin/python",
    "resolved_python_executable": "/usr/bin/python3.12",
    "python_executable_sha256": "a" * 64,
    "python_implementation": "CPython",
    "python_version": "3.12.3",
    "python_version_info": [3, 12, 3, "final", 0],
    "python_flags": {
        "isolated": True,
        "ignore_environment": True,
        "no_user_site": True,
        "safe_path": True,
    },
    "pythonpath_environment_ignored": True,
    "platform": {"system": "Linux", "machine": "x86_64", "byteorder": "little"},
    "zlib_build_version": "1.3",
    "zlib_runtime_version": "1.3",
    "stdlib_modules": {
        "gzip": {"path": "/usr/lib/python3.12/gzip.py", "sha256": "1" * 64},
        "json": {"path": "/usr/lib/python3.12/json/__init__.py", "sha256": "2" * 64},
        "json.decoder": {"path": "/usr/lib/python3.12/json/decoder.py", "sha256": "3" * 64},
        "json.encoder": {"path": "/usr/lib/python3.12/json/encoder.py", "sha256": "4" * 64},
        "json.scanner": {"path": "/usr/lib/python3.12/json/scanner.py", "sha256": "5" * 64},
        "tarfile": {"path": "/usr/lib/python3.12/tarfile.py", "sha256": "6" * 64},
    },
}
builder.__dict__["_TEST_ONLY_RUNTIME_IDENTITY"] = TEST_RUNTIME_IDENTITY
stager.__dict__["_TEST_ONLY_RUNTIME_IDENTITY"] = TEST_RUNTIME_IDENTITY

LOCKED_EXISTING_SHA256 = {
    "src/goalzendo_g00f_g01_bridge/__init__.py": (
        "94d447686b62177ed3f50742424c9a98a6f1c95d921422cf7bf3241689e24503"
    ),
    "src/goalzendo_g00f_g01_bridge/bridge.py": (
        "4ab7c684109b10c2d6ec1469a281fa342e0dac14c5d141a9d97f0cd8e4f575d2"
    ),
    "src/goalzendo_g00f_g01_bridge/cli.py": (
        "350cb69a2923b49aeb4957406b4d86319a2c75f7c7a32e14e53d479b04319801"
    ),
    "runs/goalzendo/run_g01_after_g00f_bridge.py": (
        "bc7703c48d47cc72f21e2f72ee3a4864ff5dd78759c5f692c59ce46ba03ee10b"
    ),
    "tests/goalzendo/test_g00f_g01_bridge.py": (
        "f4c04c6c04ba1a7904c7135e0e8479a15eeefc7a97fa1b380fa4a076d9fec98e"
    ),
    "docs/goalzendo/protocols/g00f-g01-authorization-bridge.md": (
        "3aeb28b0cd8df3aace1b16fb9e25a512e9b3e0632a3ee16cfa143e9587cd4661"
    ),
    "reproducibility/goalzendo/g00f-execution-freeze-20260811/execution-freeze.json": (
        "b2e385488ea7eb7c6f7bfc834c32ff2707aa9f96b718ea62b7ae92c9a4b8df38"
    ),
    "reproducibility/goalzendo/g00f-execution-freeze-20260811/g00f-execution-source.tar.gz": (
        "6b55d42bd2a75c9fc40f9c67b332c7ad5bd57024c26bd49447ef6bc831dece3a"
    ),
    "reproducibility/goalzendo/g00f-execution-freeze-20260811/g00f-source-bundle-manifest.json": (
        "482f57dc5e0b19a9c1b6ab18a1d63283ce559d288b8a00cc08e51af464d5f8f7"
    ),
    "reproducibility/goalzendo/g00f-h200-execution-freeze-20260811/execution-freeze.json": (
        "fd9cf73d124ec566e0589ec0b2e5e4d48a172bca2d04aff51bbb75a58b76b0f9"
    ),
    "reproducibility/goalzendo/g00f-h200-execution-freeze-20260811/g00f-h200-execution-source.tar.gz": (
        "e08d81bf0345ad343baa02599add8714d9c9e2f180b936113a92f91b5e8cf8ea"
    ),
    (
        "reproducibility/goalzendo/g00f-h200-execution-freeze-20260811/g00f-h200-source-bundle-manifest.json"
    ): "a6b741366d7023a625e59cf3c14f7b49e78117fadb16869d8501ad5d396f746f",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_write(path: Path, value: dict[str, Any], mode: int = 0o644) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    path.chmod(mode)


def _bundle(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    tmp_path.mkdir(parents=True)
    output = tmp_path / "bridge-bundle"
    return output, builder.build(ROOT, output)


def _bootstrap(tmp_path: Path, route: str, execution_uuid: str) -> dict[str, Any]:
    program_root = (tmp_path / route / "g00f-executions").resolve()
    program_root.mkdir(parents=True)
    execution_root = program_root / execution_uuid
    if route == "h100":
        bootstrap = _load_source(
            ROOT / "runs/goalzendo/g00f_bundle_bootstrap.py",
            f"g00f_bundle_bootstrap_{execution_uuid}",
        )
        source_dir = ROOT / "reproducibility/goalzendo/g00f-execution-freeze-20260811"
        freeze = source_dir / "execution-freeze.json"
        arguments = argparse.Namespace(
            freeze=freeze,
            expected_freeze_sha256=stager.ROUTES[route]["freeze"]["file_sha256"],
            archive=source_dir / "g00f-execution-source.tar.gz",
            manifest=source_dir / "g00f-source-bundle-manifest.json",
            output=execution_root / "frozen-source",
            receipt=execution_root / "source-bundle-receipt.json",
            actual_bootstrap=ROOT / "runs/goalzendo/g00f_bundle_bootstrap.py",
            actual_launcher=ROOT / "runs/goalzendo/run_g00f_frozen_4h100.sh",
            actual_watchdog=ROOT / "runs/goalzendo/g00f_watchdog.py",
        )
    else:
        bootstrap = _load_source(
            ROOT / "runs/goalzendo/g00f_h200_bundle_bootstrap.py",
            f"g00f_h200_bundle_bootstrap_{execution_uuid}",
        )
        source_dir = ROOT / "reproducibility/goalzendo/g00f-h200-execution-freeze-20260811"
        freeze = source_dir / "execution-freeze.json"
        arguments = argparse.Namespace(
            freeze=freeze,
            expected_freeze_sha256=stager.ROUTES[route]["freeze"]["file_sha256"],
            archive=source_dir / "g00f-h200-execution-source.tar.gz",
            manifest=source_dir / "g00f-h200-source-bundle-manifest.json",
            output=execution_root / "frozen-source",
            receipt=execution_root / "source-bundle-receipt.json",
            actual_bootstrap=ROOT / "runs/goalzendo/g00f_h200_bundle_bootstrap.py",
            actual_detached_supervisor=ROOT / "runs/goalzendo/g00f_h200_detached_supervisor.py",
            actual_launcher=ROOT / "runs/goalzendo/run_g00f_frozen_4h200.sh",
            actual_qualification_controller=(ROOT / "runs/goalzendo/g00f_h200_qualification_controller.py"),
            actual_qualification_supervisor=(ROOT / "runs/goalzendo/g00f_h200_qualification_supervisor.py"),
            actual_watchdog=ROOT / "runs/goalzendo/g00f_h200_watchdog.py",
        )
    bootstrap.prepare(arguments)
    receipt = execution_root / "source-bundle-receipt.json"
    return {
        "route": route,
        "execution_uuid": execution_uuid,
        "program_root": program_root,
        "execution_root": execution_root,
        "frozen_source": execution_root / "frozen-source",
        "source_receipt": receipt,
        "source_receipt_sha256": _sha256(receipt),
        "selected_route_freeze": freeze,
    }


def _stage_kwargs(context: dict[str, Any], output: Path, built: dict[str, Any]) -> dict[str, Any]:
    route = str(context["route"])
    return {
        "route": route,
        "execution_uuid": context["execution_uuid"],
        "selected_route_freeze": context["selected_route_freeze"],
        "expected_selected_route_freeze_sha256": stager.ROUTES[route]["freeze"]["file_sha256"],
        "bundle_freeze": output / stager.FREEZE_NAME,
        "expected_bundle_freeze_sha256": built["freeze_sha256"],
        "expected_bundle_archive_sha256": built["archive_sha256"],
        "expected_bundle_manifest_sha256": built["manifest_sha256"],
        "expected_source_receipt_sha256": context["source_receipt_sha256"],
    }


def _set_root(monkeypatch: pytest.MonkeyPatch, context: dict[str, Any]) -> None:
    monkeypatch.setattr(stager, "_TEST_ONLY_PROGRAM_ROOT", context["program_root"])


def _rewrite_archive(
    output: Path,
    transform: Callable[[list[tuple[tarfile.TarInfo, bytes]]], list[tuple[tarfile.TarInfo, bytes]]],
) -> dict[str, str]:
    archive_path = output / stager.ARCHIVE_NAME
    rows: list[tuple[tarfile.TarInfo, bytes]] = []
    with tarfile.open(archive_path, "r:gz") as archive:
        for original in archive.getmembers():
            info = copy.copy(original)
            extracted = archive.extractfile(original)
            assert extracted is not None
            rows.append((info, extracted.read()))
    altered = transform(rows)
    buffer = io.BytesIO()
    with (
        gzip.GzipFile(filename="", mode="wb", fileobj=buffer, compresslevel=9, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive,
    ):
        for info, payload in altered:
            archive.addfile(info, None if info.issym() or info.islnk() else io.BytesIO(payload))
    archive_path.write_bytes(buffer.getvalue())
    archive_path.chmod(0o644)
    freeze_path = output / stager.FREEZE_NAME
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze["bundle"]["archive_sha256"] = _sha256(archive_path)
    body = {key: value for key, value in freeze.items() if key != "freeze_digest"}
    freeze["freeze_digest"] = stager.digest(body)
    _json_write(freeze_path, freeze)
    return {"archive_sha256": _sha256(archive_path), "freeze_sha256": _sha256(freeze_path)}


def test_locked_inputs_and_generated_cache_are_unchanged() -> None:
    assert {relative: _sha256(ROOT / relative) for relative in LOCKED_EXISTING_SHA256} == (
        LOCKED_EXISTING_SHA256
    )
    cache = ROOT / "runs/goalzendo/__pycache__"
    assert not list(cache.glob("build_g00f_g01_bridge_bundle.*.pyc"))
    assert not list(cache.glob("g00f_g01_bridge_bundle_stage.*.pyc"))


def test_builder_is_byte_equal_under_bound_runtime_exact_six_and_non_authorizing(
    tmp_path: Path,
) -> None:
    first, first_result = _bundle(tmp_path / "first")
    second, second_result = _bundle(tmp_path / "second")
    for name in (builder.ARCHIVE_NAME, builder.MANIFEST_NAME, builder.FREEZE_NAME):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    assert first_result["bridge_source_digest"] == stager.BRIDGE_SOURCE_DIGEST
    assert first_result["archive_sha256"] == second_result["archive_sha256"]
    with tarfile.open(first / builder.ARCHIVE_NAME, "r:gz") as archive:
        infos = archive.getmembers()
    assert len(infos) == 6
    assert {info.name for info in infos} == set(stager.PAYLOAD_SHA256)
    assert all(info.isreg() and info.mode & 0o777 == 0o644 for info in infos)
    assert all("__pycache__" not in Path(info.name).parts for info in infos)
    freeze = json.loads((first / builder.FREEZE_NAME).read_text(encoding="utf-8"))
    assert freeze["bundle"]["archive_total_member_count"] == 6
    assert freeze["bundle"]["payload_member_count"] == 6
    assert freeze["bundle"]["selected_route_freeze_copy_count"] == 1
    assert freeze["authorization"] == stager.AUTHORIZATION


@pytest.mark.parametrize(
    ("route", "execution_uuid"),
    [
        ("h100", "00000000-0000-4000-8000-000000000101"),
        ("h200", "00000000-0000-4000-8000-000000000102"),
    ],
)
def test_real_route_bootstrap_stage_verify_freeze_and_exact_g01(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route: str,
    execution_uuid: str,
) -> None:
    output, built = _bundle(tmp_path / "built")
    context = _bootstrap(tmp_path, route, execution_uuid)
    _set_root(monkeypatch, context)
    kwargs = _stage_kwargs(context, output, built)
    result = stager.stage(**kwargs)
    assert result["verified"] is True
    assert result["archive_entry_count"] == result["staged_payload_count"] == 6
    assert result["selected_route_freeze_copy_count"] == 1
    receipt_path = Path(result["path"])
    assert receipt_path.parent == context["program_root"]
    assert context["execution_root"] not in receipt_path.parents
    assert receipt_path.stat().st_mode & 0o777 == 0o400
    assert (
        stager.verify_stage(
            **kwargs,
            expected_stage_receipt_sha256=result["file_sha256"],
        )["exact_post_stage_inventory"]
        is True
    )

    frozen_source = context["frozen_source"]
    internal_freeze = frozen_source / stager.ROUTES[route]["freeze"]["path"]
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": str(frozen_source / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    package = "goalzendo_g00f" if route == "h100" else "goalzendo_g00f_h200"
    verified = subprocess.run(
        [
            sys.executable,
            "-P",
            "-m",
            f"{package}.cli",
            "verify-freeze",
            "--repo",
            str(frozen_source),
            "--freeze",
            str(internal_freeze),
            "--expected-freeze-sha256",
            stager.ROUTES[route]["freeze"]["file_sha256"],
        ],
        cwd=frozen_source,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert (verified.returncode, verified.stderr) == (0, "")
    replay = subprocess.run(
        [
            sys.executable,
            "-P",
            "-c",
            (
                "from pathlib import Path; "
                "from goalzendo_g00f_g01_bridge.bridge import "
                "calculate_bridge_source_binding,g01_binding; "
                "import json; print(json.dumps({'source':"
                "calculate_bridge_source_binding(Path.cwd()),'g01':g01_binding(Path.cwd())},"
                "sort_keys=True))"
            ),
        ],
        cwd=frozen_source,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert (replay.returncode, replay.stderr) == (0, "")
    replayed = json.loads(replay.stdout)
    assert replayed["source"]["source_digest"] == stager.BRIDGE_SOURCE_DIGEST
    assert replayed["g01"]["plan"] == stager.G01_IDENTITY["plan"]
    assert replayed["g01"]["target_binding_digest"] == stager.G01_IDENTITY["target_binding_digest"]


def _traversal(rows: list[tuple[tarfile.TarInfo, bytes]]) -> list[tuple[tarfile.TarInfo, bytes]]:
    rows[0][0].name = "../escape.py"
    return rows


def _duplicate(rows: list[tuple[tarfile.TarInfo, bytes]]) -> list[tuple[tarfile.TarInfo, bytes]]:
    rows.append((copy.copy(rows[0][0]), rows[0][1]))
    return rows


def _link(rows: list[tuple[tarfile.TarInfo, bytes]]) -> list[tuple[tarfile.TarInfo, bytes]]:
    rows[0][0].type = tarfile.SYMTYPE
    rows[0][0].linkname = "target"
    rows[0][0].size = 0
    rows[0] = (rows[0][0], b"")
    return rows


def _mode(rows: list[tuple[tarfile.TarInfo, bytes]]) -> list[tuple[tarfile.TarInfo, bytes]]:
    rows[0][0].mode = 0o600
    return rows


def _tamper(rows: list[tuple[tarfile.TarInfo, bytes]]) -> list[tuple[tarfile.TarInfo, bytes]]:
    rows[0] = (rows[0][0], rows[0][1] + b"x")
    rows[0][0].size += 1
    return rows


@pytest.mark.parametrize(
    "transform",
    [_traversal, _duplicate, _link, _mode, _tamper],
    ids=["traversal", "duplicate", "link", "mode", "tamper"],
)
def test_stager_rejects_malformed_exact_member_archive_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    transform: Callable[[list[tuple[tarfile.TarInfo, bytes]]], list[tuple[tarfile.TarInfo, bytes]]],
) -> None:
    output, built = _bundle(tmp_path / "built")
    changed = _rewrite_archive(output, transform)
    built.update(changed)
    context = _bootstrap(tmp_path, "h100", "00000000-0000-4000-8000-000000000201")
    _set_root(monkeypatch, context)
    with pytest.raises(stager.StageError):
        stager.stage(**_stage_kwargs(context, output, built))
    for relative in stager.PAYLOAD_SHA256:
        assert not (context["frozen_source"] / relative).exists()
    assert not (context["frozen_source"] / stager.ROUTES["h100"]["freeze"]["path"]).exists()


def test_stager_rejects_overwrite_and_verify_stage_detects_post_stage_extra(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, built = _bundle(tmp_path / "built")
    context = _bootstrap(tmp_path, "h100", "00000000-0000-4000-8000-000000000301")
    _set_root(monkeypatch, context)
    kwargs = _stage_kwargs(context, output, built)
    result = stager.stage(**kwargs)
    with pytest.raises(stager.StageError, match=r"inventory|permanent"):
        stager.stage(**kwargs)
    (context["frozen_source"] / "unexpected-shadow.py").write_text("x", encoding="utf-8")
    with pytest.raises(stager.StageError, match="inventory"):
        stager.verify_stage(
            **kwargs,
            expected_stage_receipt_sha256=result["file_sha256"],
        )


def test_stager_rejects_symlink_shadow_in_pristine_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, built = _bundle(tmp_path / "built")
    context = _bootstrap(tmp_path, "h100", "00000000-0000-4000-8000-000000000302")
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    (context["frozen_source"] / "src/goalzendo_g00f_g01_bridge").symlink_to(
        shadow,
        target_is_directory=True,
    )
    _set_root(monkeypatch, context)
    with pytest.raises(stager.StageError, match=r"linked|inventory"):
        stager.stage(**_stage_kwargs(context, output, built))


@pytest.mark.parametrize("case", ["receipt", "route", "root", "selected-freeze", "stager"])
def test_stager_rejects_wrong_receipt_route_root_freeze_or_stager(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    output, built = _bundle(tmp_path / "built")
    context = _bootstrap(tmp_path, "h100", "00000000-0000-4000-8000-000000000401")
    _set_root(monkeypatch, context)
    kwargs = _stage_kwargs(context, output, built)
    if case == "receipt":
        kwargs["expected_source_receipt_sha256"] = "0" * 64
    elif case == "route":
        kwargs["route"] = "h200"
        kwargs["selected_route_freeze"] = (
            ROOT / "reproducibility/goalzendo/g00f-h200-execution-freeze-20260811/execution-freeze.json"
        )
        kwargs["expected_selected_route_freeze_sha256"] = stager.ROUTES["h200"]["freeze"]["file_sha256"]
    elif case == "root":
        receipt_path = context["source_receipt"]
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["extracted_root"] = str(tmp_path / "wrong-root")
        body = {key: value for key, value in receipt.items() if key != "receipt_digest"}
        receipt["receipt_digest"] = stager.digest(body)
        _json_write(receipt_path, receipt, 0o600)
        kwargs["expected_source_receipt_sha256"] = _sha256(receipt_path)
    elif case == "selected-freeze":
        kwargs["selected_route_freeze"] = (
            ROOT / "reproducibility/goalzendo/g00f-h200-execution-freeze-20260811/execution-freeze.json"
        )
    else:
        freeze_path = output / stager.FREEZE_NAME
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        freeze["controller_files"]["stager"]["sha256"] = "0" * 64
        body = {key: value for key, value in freeze.items() if key != "freeze_digest"}
        freeze["freeze_digest"] = stager.digest(body)
        _json_write(freeze_path, freeze)
        kwargs["expected_bundle_freeze_sha256"] = _sha256(freeze_path)
    with pytest.raises(stager.StageError):
        stager.stage(**kwargs)


def test_builder_rejects_noncanonical_running_copy(tmp_path: Path) -> None:
    copied = tmp_path / "copied-builder.py"
    copied.write_bytes(BUILDER_PATH.read_bytes())
    alternate = _load_source(copied, "copied_bridge_bundle_builder_test")
    alternate.__dict__["_TEST_ONLY_RUNTIME_IDENTITY"] = TEST_RUNTIME_IDENTITY
    with pytest.raises(alternate.BuildError, match="running builder"):
        alternate.build(ROOT, tmp_path / "output")


@pytest.mark.parametrize("controller", ["builder", "stager"])
def test_stager_rejects_malformed_frozen_controller_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    controller: str,
) -> None:
    output, built = _bundle(tmp_path / "built")
    context = _bootstrap(tmp_path, "h100", "00000000-0000-4000-8000-000000000402")
    _set_root(monkeypatch, context)
    freeze_path = output / stager.FREEZE_NAME
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze["controller_files"][controller]["path"] = "wrong.py"
    body = {key: value for key, value in freeze.items() if key != "freeze_digest"}
    freeze["freeze_digest"] = stager.digest(body)
    _json_write(freeze_path, freeze)
    built["freeze_sha256"] = _sha256(freeze_path)
    with pytest.raises(stager.StageError, match="builder/stager"):
        stager.stage(**_stage_kwargs(context, output, built))


@pytest.mark.parametrize("artifact", ["bundle-freeze", "manifest", "route-freeze"])
def test_stager_rejects_extra_schema_fields_or_wrong_freeze_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
) -> None:
    output, built = _bundle(tmp_path / "built")
    context = _bootstrap(tmp_path, "h100", "00000000-0000-4000-8000-000000000403")
    _set_root(monkeypatch, context)
    kwargs = _stage_kwargs(context, output, built)
    if artifact == "bundle-freeze":
        freeze_path = output / stager.FREEZE_NAME
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        freeze["unknown_claim"] = True
        body = {key: value for key, value in freeze.items() if key != "freeze_digest"}
        freeze["freeze_digest"] = stager.digest(body)
        _json_write(freeze_path, freeze)
        kwargs["expected_bundle_freeze_sha256"] = _sha256(freeze_path)
    elif artifact == "manifest":
        manifest_path = output / stager.MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["unknown_claim"] = True
        body = {key: value for key, value in manifest.items() if key != "manifest_digest"}
        manifest["manifest_digest"] = stager.digest(body)
        _json_write(manifest_path, manifest)
        freeze_path = output / stager.FREEZE_NAME
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        freeze["bundle"]["manifest_sha256"] = _sha256(manifest_path)
        freeze["bundle"]["manifest_digest"] = manifest["manifest_digest"]
        freeze_body = {key: value for key, value in freeze.items() if key != "freeze_digest"}
        freeze["freeze_digest"] = stager.digest(freeze_body)
        _json_write(freeze_path, freeze)
        kwargs["expected_bundle_manifest_sha256"] = _sha256(manifest_path)
        kwargs["expected_bundle_freeze_sha256"] = _sha256(freeze_path)
    else:
        copied_freeze = tmp_path / "selected-freeze.json"
        copied_freeze.write_bytes(Path(context["selected_route_freeze"]).read_bytes())
        copied_freeze.chmod(0o600)
        kwargs["selected_route_freeze"] = copied_freeze
    with pytest.raises(stager.StageError):
        stager.stage(**kwargs)


@pytest.mark.parametrize(
    "occupied",
    [
        "g00f-g01-route-lock.json",
        "g00f-g01-scientific-eligibility.json",
        "g00f-g01-coordinator-input.json",
    ],
)
def test_verify_stage_requires_fresh_pre_lock_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    occupied: str,
) -> None:
    output, built = _bundle(tmp_path / "built")
    context = _bootstrap(tmp_path, "h100", "00000000-0000-4000-8000-000000000501")
    _set_root(monkeypatch, context)
    kwargs = _stage_kwargs(context, output, built)
    result = stager.stage(**kwargs)
    (context["program_root"] / occupied).write_text("occupied\n", encoding="utf-8")
    with pytest.raises(stager.StageError, match="pre-lock"):
        stager.verify_stage(
            **kwargs,
            expected_stage_receipt_sha256=result["file_sha256"],
        )


@pytest.mark.parametrize("case", ["mode", "bytes", "hash"])
def test_verify_stage_rejects_receipt_mode_bytes_or_external_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    output, built = _bundle(tmp_path / "built")
    context = _bootstrap(tmp_path, "h100", "00000000-0000-4000-8000-000000000502")
    _set_root(monkeypatch, context)
    kwargs = _stage_kwargs(context, output, built)
    result = stager.stage(**kwargs)
    receipt_path = Path(result["path"])
    expected_sha = result["file_sha256"]
    if case == "mode":
        receipt_path.chmod(0o600)
    elif case == "bytes":
        receipt_path.chmod(0o600)
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["route"] = "h200"
        body = {key: value for key, value in receipt.items() if key != "receipt_digest"}
        receipt["receipt_digest"] = stager.digest(body)
        _json_write(receipt_path, receipt, 0o400)
        expected_sha = _sha256(receipt_path)
    else:
        expected_sha = "0" * 64
    with pytest.raises(stager.StageError):
        stager.verify_stage(
            **kwargs,
            expected_stage_receipt_sha256=expected_sha,
        )


def test_cli_exposes_no_root_output_or_member_override() -> None:
    source = STAGER_PATH.read_text(encoding="utf-8")
    assert 'add_parser("stage")' in source
    assert 'add_parser("verify-stage")' in source
    assert "--expected-stage-receipt-sha256" in source
    for forbidden in ("--root", "--repo", "--output", "--receipt", "--member"):
        assert f'add_argument("{forbidden}"' not in source
    protocol = (ROOT / "docs/goalzendo/protocols/g00f-g01-prebootstrap-bundle.md").read_text(encoding="utf-8")
    assert protocol.count("/workspace/.venvs/goalzendo/bin/python -I") == 3
    assert protocol.index("verify-stage") < protocol.index("real `verify-freeze`")
    assert protocol.index("real `verify-freeze`") < protocol.index("create and externally pin")


def test_production_runtime_rejects_local_nonisolated_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(stager, "_TEST_ONLY_RUNTIME_IDENTITY", None)
    with pytest.raises(stager.StageError, match="exact isolated"):
        stager._runtime_identity()
    monkeypatch.setattr(builder, "_TEST_ONLY_RUNTIME_IDENTITY", None)
    with pytest.raises(builder.BuildError, match="exact isolated"):
        builder._runtime_identity()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("python_version", "3.12.4"),
        ("zlib_runtime_version", "1.2.13"),
        ("python_executable_sha256", "not-a-sha"),
        (
            "python_flags",
            {
                "isolated": False,
                "ignore_environment": True,
                "no_user_site": True,
                "safe_path": True,
            },
        ),
    ],
)
def test_stager_rejects_malformed_build_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: Any,
) -> None:
    output, built = _bundle(tmp_path / "built")
    context = _bootstrap(tmp_path, "h100", "00000000-0000-4000-8000-000000000601")
    _set_root(monkeypatch, context)
    freeze_path = output / stager.FREEZE_NAME
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze["build_runtime"][field] = value
    body = {key: item for key, item in freeze.items() if key != "freeze_digest"}
    freeze["freeze_digest"] = stager.digest(body)
    _json_write(freeze_path, freeze)
    built["freeze_sha256"] = _sha256(freeze_path)
    with pytest.raises(stager.StageError, match="build runtime"):
        stager.stage(**_stage_kwargs(context, output, built))


def test_portable_build_runtime_may_have_different_executable_sha_but_verify_is_same_pod(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_runtime = {**TEST_RUNTIME_IDENTITY, "python_executable_sha256": "b" * 64}
    monkeypatch.setattr(builder, "_TEST_ONLY_RUNTIME_IDENTITY", build_runtime)
    output, built = _bundle(tmp_path / "built")
    context = _bootstrap(tmp_path, "h100", "00000000-0000-4000-8000-000000000602")
    _set_root(monkeypatch, context)
    monkeypatch.setattr(stager, "_TEST_ONLY_RUNTIME_IDENTITY", TEST_RUNTIME_IDENTITY)
    kwargs = _stage_kwargs(context, output, built)
    result = stager.stage(**kwargs)
    receipt = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
    assert receipt["runtime"]["python_executable_sha256"] == "a" * 64
    assert (
        json.loads((output / stager.FREEZE_NAME).read_text(encoding="utf-8"))["build_runtime"][
            "python_executable_sha256"
        ]
        == "b" * 64
    )
    changed_live_runtime = {**TEST_RUNTIME_IDENTITY, "python_executable_sha256": "c" * 64}
    monkeypatch.setattr(stager, "_TEST_ONLY_RUNTIME_IDENTITY", changed_live_runtime)
    with pytest.raises(stager.StageError, match="receipt"):
        stager.verify_stage(
            **kwargs,
            expected_stage_receipt_sha256=result["file_sha256"],
        )


def test_receipt_writer_uses_held_parent_descriptor() -> None:
    source = STAGER_PATH.read_text(encoding="utf-8")
    function = source[source.index("def _write_receipt") : source.index("def _parse_utc")]
    assert "os.open(name, flags, 0o600, dir_fd=parent)" in function
    assert "os.open(path.parent" not in function
    assert function.index("os.fchmod") < function.index("os.fsync(handle.fileno())")


def test_builder_rejects_output_directory_ancestor_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "build-parent"
    parent.mkdir()
    output = parent / "bridge-bundle"
    moved = parent / "moved-held-bundle"
    original_write = builder._exclusive_write
    swapped = False

    def swap_then_write(parent_fd: int, name: str, payload: bytes, mode: int = 0o644) -> None:
        nonlocal swapped
        if not swapped:
            swapped = True
            output.rename(moved)
            output.mkdir()
        original_write(parent_fd, name, payload, mode)

    monkeypatch.setattr(builder, "_exclusive_write", swap_then_write)
    with pytest.raises(builder.BuildError, match="output directory path no longer"):
        builder.build(ROOT, output)
    assert not list(output.iterdir())
    assert (moved / builder.MANIFEST_NAME).is_file()


def test_stager_keeps_writes_on_held_tree_and_rejects_program_root_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, built = _bundle(tmp_path / "built")
    execution_uuid = "00000000-0000-4000-8000-000000000701"
    context = _bootstrap(tmp_path, "h100", execution_uuid)
    _set_root(monkeypatch, context)
    program_root = context["program_root"]
    moved_root = program_root.with_name("moved-held-g00f-executions")
    decoy_frozen = program_root / execution_uuid / "frozen-source"
    original_write = stager._write_relative
    swapped = False

    def swap_then_write(root_fd: int, relative: str, payload: bytes, mode: int) -> None:
        nonlocal swapped
        if not swapped:
            swapped = True
            program_root.rename(moved_root)
            decoy_frozen.mkdir(parents=True)
        original_write(root_fd, relative, payload, mode)

    monkeypatch.setattr(stager, "_write_relative", swap_then_write)
    with pytest.raises(stager.StageError, match="program root path no longer"):
        stager.stage(**_stage_kwargs(context, output, built))
    assert not any((decoy_frozen / relative).exists() for relative in stager.PAYLOAD_SHA256)
    selected_relative = stager.ROUTES["h100"]["freeze"]["path"]
    assert (moved_root / execution_uuid / "frozen-source" / selected_relative).is_file()
