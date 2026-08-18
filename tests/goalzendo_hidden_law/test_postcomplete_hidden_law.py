from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "runs" / "goalzendo" / "postcomplete_qwen35_hidden_law.sh"


def _embedded_python_blocks() -> list[str]:
    source = SCRIPT.read_text(encoding="utf-8")
    return re.findall(r"<<'PY'\n(.*?)\nPY$", source, flags=re.MULTILINE | re.DOTALL)


def test_postcompletion_wrapper_has_valid_shell_and_embedded_python() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    blocks = _embedded_python_blocks()
    assert len(blocks) == 4
    for index, block in enumerate(blocks):
        compile(block, f"<hidden-law-postcompletion-{index}>", "exec")


def test_receipt_preflight_accepts_only_six_exact_rc0_receipts(tmp_path: Path) -> None:
    fixed = {
        "schema": "goalzendo.hidden_law_shard_status",
        "schema_version": 1,
        "job_id": "qwen35-hidden-law-20260815T153604Z",
        "num_shards": 6,
        "state": "complete",
        "runner_rc": 0,
        "archive_sha256": "56700896a4f62f6e92a495cc0af109ff1f8982c8e9be57b920de6737a3b919f8",
        "config_sha256": "a2e31815f3fbe1d65fcd64654f8fb9e51a2d950cf8a2985cc5d90121bad3e53a",
        "scientific_config_digest": "d61cad537e4ae3ea439250bc5e7e2466405e5331be251f6b8f15d0de8683207b",
        "implementation_fingerprint": "9cd679289fc22ceeb5fe1a9cbc2ca134051faf8491b9fa478be9b13ca81dd087",
        "started_at": "2026-08-15T16:00:00Z",
        "finished_at": "2026-08-16T00:00:00Z",
    }
    for index in range(6):
        receipt = {**fixed, "shard_index": index}
        payload = json.dumps(receipt) + "\n"
        (tmp_path / f"shard-{index:02d}.complete.json").write_text(payload, encoding="utf-8")
        (tmp_path / f"shard-{index:02d}.status.json").write_text(payload, encoding="utf-8")

    receipt_preflight = _embedded_python_blocks()[1]
    passed = subprocess.run(
        [sys.executable, "-", str(tmp_path)],
        input=receipt_preflight,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(passed.stdout) == {"terminal_shard_receipt_count": 6}

    bad_path = tmp_path / "shard-03.complete.json"
    bad = json.loads(bad_path.read_text(encoding="utf-8"))
    bad["runner_rc"] = 1
    bad_path.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    failed = subprocess.run(
        [sys.executable, "-", str(tmp_path)],
        input=receipt_preflight,
        check=False,
        capture_output=True,
        text=True,
    )
    assert failed.returncode != 0
    assert "frozen bindings" in failed.stderr


def test_postcompletion_wrapper_is_exact_and_outcome_agnostic() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    required = (
        "qwen35-hidden-law-20260815T153604Z",
        "56700896a4f62f6e92a495cc0af109ff1f8982c8e9be57b920de6737a3b919f8",
        "a2e31815f3fbe1d65fcd64654f8fb9e51a2d950cf8a2985cc5d90121bad3e53a",
        "9cd679289fc22ceeb5fe1a9cbc2ca134051faf8491b9fa478be9b13ca81dd087",
        "8bedacc9b5935e67224676101e12692bb0a5b913f1e9de180a033ac20f0a2d8c",
        'expected_names = {f"shard-{index:02d}.complete.json" for index in range(num_shards)}',
        "verify_completed_run(path)",
        "if set(children) != set(expected):",
        'scripts/analyze_qwen35_hidden_law.py "$ARTIFACT_ROOT"',
        'sha256sum "$ANALYSIS_NAME" status.json > SHA256SUMS',
        'mv "$STAGING" "$ANALYSIS_DIR"',
        "sleep infinity",
    )
    for item in required:
        assert item in source

    # The wrapper may validate fixed panel metadata, but it must not select or
    # gate publication on any scientific estimate emitted by the analyzer.
    forbidden = (
        "information_fraction",
        "exact_rule_recovery",
        "law_control_margin",
        "causal_law_control_margin",
        "primary_sft_minus_outcome_rl",
        "registered_secondary",
    )
    for item in forbidden:
        assert item not in source
