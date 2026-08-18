from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "paper" / "forkworld-current-results" / "e19_pilot_gate.py"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("e19_pilot_gate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves postponed annotations through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _runs(module: ModuleType, failing_seeds: set[int] | None = None) -> dict[Any, Any]:
    failing_seeds = failing_seeds or set()
    runs: dict[Any, Any] = {}
    for seed in module.PILOT_SEEDS:
        snapshots = {
            "independent_noop": (0.50, 0.95, 0.95, 0.20, 0.10),
            "independent_q_restore": (
                0.46 if seed in failing_seeds else 0.44,
                0.95,
                0.95,
                0.20,
                0.10,
            ),
            "independent_padding_sham": (0.51, 0.95, 0.95, 0.20, 0.10),
            "nested_noop": (0.40, 0.95, 0.95, 0.20, 0.10),
            "nested_q_transplant": (0.46, 0.95, 0.95, 0.20, 0.10),
            "nested_padding_sham": (0.39, 0.95, 0.95, 0.20, 0.10),
        }
        for branch, (q_probe, p_behavior, p_causal, p_probe, y_probe) in snapshots.items():
            summary = {
                "seed": seed,
                "branch": branch,
                "postedit_pure_p": True,
                "postedit_snapshot": {
                    "behavior": {"P": p_behavior},
                    "causal": {"P": p_causal},
                    "selective_final_hidden_probe": {
                        "P": p_probe,
                        "Q": q_probe,
                        "Y": y_probe,
                    },
                },
            }
            runs[(seed, branch)] = module.PilotRun(
                Path(branch), {}, summary, {}, 0
            )
    return runs


def test_pilot_gate_requires_two_same_seed_joint_passes() -> None:
    module = _module()
    passed = module.evaluate_gate(_runs(module))
    assert passed["pilot_gate_passed"] is True
    assert passed["joint_seed_pass_count"] == 3
    assert passed["decision"] == "PASS"

    two_pass = module.evaluate_gate(_runs(module, {541}))
    assert two_pass["pilot_gate_passed"] is True
    assert two_pass["joint_seed_pass_count"] == 2

    stopped = module.evaluate_gate(_runs(module, {541, 547}))
    assert stopped["pilot_gate_passed"] is False
    assert stopped["joint_seed_pass_count"] == 1
    assert stopped["decision"] == "STOP"


def test_sham_delta_roundoff_fails_above_frozen_tolerance() -> None:
    module = _module()
    intended = {
        "shape": [64, 2],
        "target_scalar_count": 128,
        "nonzero_scalar_count": 128,
        "digest": "a" * 64,
        "l1_norm": 1.0,
        "l2_norm": 0.2,
        "linf_norm": 0.03,
        "finite": True,
    }
    within = dict(intended)
    within["digest"] = "b" * 64
    within["l1_norm"] += 0.9e-6
    errors: list[str] = []
    module._audit_sham_delta_roundoff(errors, "run", "restore", intended, within)
    assert errors == []

    above = dict(within)
    above["l1_norm"] = intended["l1_norm"] + 1.1e-6
    module._audit_sham_delta_roundoff(errors, "run", "restore", intended, above)
    assert len(errors) == 1
    assert "exceeding 1e-06" in errors[0]
