from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "runs/goalzendo/run_qwen35_hidden_law.sh"


def test_hidden_law_launcher_exposes_exact_outcome_blind_plan() -> None:
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
    environment = dict(os.environ)
    environment["GOALZENDO_PYTHON"] = sys.executable
    result = subprocess.run(
        ["bash", str(LAUNCHER), "plan"],
        check=True,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    plan = json.loads(result.stdout)

    assert plan["schema"] == "goalzendo.hidden_law_launch_plan"
    assert {
        (item["model_selector"], item["algorithm"], item["source_seed"], item["smoke_seed"])
        for item in plan["smokes"]
    } == {
        (model, algorithm, 23011, 23999)
        for model in ("0.8B", "2B")
        for algorithm in ("process_sft", "outcome_rl")
    }
    assert len({item["smoke_run_id"] for item in plan["smokes"]}) == 4

    production = plan["production"]
    assert production["num_shards"] == 6
    assert production["run_count"] == 24
    assert [item["run_count"] for item in production["shards"]] == [4, 5, 3, 4, 5, 3]
    run_ids = [run_id for item in production["shards"] for run_id in item["run_ids"]]
    assert len(run_ids) == len(set(run_ids)) == 24
