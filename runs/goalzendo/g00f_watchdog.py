#!/usr/bin/env python3
"""Monotonic G00-F deadline watchdog with signal-before-receipt ordering."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any


def _digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _receipt(
    schema: str,
    budget: dict[str, Any],
    deadline: int,
    **fields: Any,
) -> dict[str, Any]:
    body = {
        "schema": schema,
        "schema_version": 1,
        "execution_uuid": budget["execution_uuid"],
        "freeze_file_sha256": budget["freeze_file_sha256"],
        "freeze_digest": budget["freeze_digest"],
        "deadline_monotonic_ns": deadline,
        **fields,
        "outcome_metrics_read": False,
        "predictions_read": False,
        "g01_launch_authorized": False,
    }
    return {**body, "receipt_digest": _digest(body)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget-start", type=Path, required=True)
    parser.add_argument("--pid-file", type=Path, required=True)
    parser.add_argument("--started-receipt", type=Path, required=True)
    parser.add_argument("--normal-stop-receipt", type=Path, required=True)
    parser.add_argument("--fired-receipt", type=Path, required=True)
    parser.add_argument("--cancel-receipt", type=Path, required=True)
    parser.add_argument("--kill-receipt", type=Path, required=True)
    parser.add_argument("timeout_command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    budget = json.loads(args.budget_start.read_text(encoding="utf-8"))
    budget_body = {key: value for key, value in budget.items() if key != "budget_digest"}
    if (
        budget.get("schema") != "goalzendo.g00f_monotonic_budget_start"
        or budget.get("budget_digest") != _digest(budget_body)
        or budget.get("wall_ceiling_seconds") != 14 * 60 * 60
    ):
        raise RuntimeError("watchdog budget-start receipt is not authentic")
    start = int(budget["started_monotonic_ns"])
    ceiling = int(budget["wall_ceiling_seconds"]) * 1_000_000_000
    deadline = start + ceiling
    normal_stop_requested = False

    def request_normal_stop(_signal_number: int, _frame: Any) -> None:
        nonlocal normal_stop_requested
        normal_stop_requested = True

    signal.signal(signal.SIGTERM, request_normal_stop)
    started_monotonic_ns = time.monotonic_ns()
    _exclusive(
        args.started_receipt,
        _receipt(
            "goalzendo.g00f_watchdog_started",
            budget,
            deadline,
            budget_start_file_sha256=_sha256_file(args.budget_start),
            started_monotonic_ns=started_monotonic_ns,
            watchdog_pid=os.getpid(),
        ),
    )
    while True:
        if normal_stop_requested:
            stopped_monotonic_ns = time.monotonic_ns()
            if stopped_monotonic_ns > deadline:
                raise RuntimeError("watchdog normal stop arrived after the monotonic deadline")
            _exclusive(
                args.normal_stop_receipt,
                _receipt(
                    "goalzendo.g00f_watchdog_normal_stop",
                    budget,
                    deadline,
                    budget_start_file_sha256=_sha256_file(args.budget_start),
                    stopped_monotonic_ns=stopped_monotonic_ns,
                    watchdog_pid=os.getpid(),
                ),
            )
            return 0
        remaining = deadline - time.monotonic_ns()
        if remaining <= 0:
            break
        time.sleep(min(30.0, remaining / 1_000_000_000))
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    pids = [int(value) for value in args.pid_file.read_text(encoding="ascii").split()]
    if len(pids) != len(set(pids)) or any(pid <= 1 for pid in pids):
        raise RuntimeError("watchdog PID inventory is duplicated or unsafe")

    # Signal first.  No evaluator, prediction, or metric read occurs before or
    # after this operational decision.  Only then may the coordinator write
    # the 160 timeout/failure receipts.
    terminated: list[int] = []
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
            terminated.append(pid)
        except ProcessLookupError:
            pass
    signalled_ns = time.monotonic_ns()
    _exclusive(
        args.fired_receipt,
        _receipt(
            "goalzendo.g00f_watchdog_deadline_fired",
            budget,
            deadline,
            signalled_monotonic_ns=signalled_ns,
            sigterm_pids=terminated,
        ),
    )
    _exclusive(
        args.cancel_receipt,
        _receipt(
            "goalzendo.g00f_coordinated_timeout_cancel",
            budget,
            deadline,
            cancel_monotonic_ns=signalled_ns,
            sigterm_pids=terminated,
        ),
    )

    command = list(args.timeout_command)
    if command and command[0] == "--":
        command = command[1:]
    timeout_exit = subprocess.run(command, check=False).returncode if command else 64
    time.sleep(30)
    killed: list[int] = []
    for pid in terminated:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        os.kill(pid, signal.SIGKILL)
        killed.append(pid)
    _exclusive(
        args.kill_receipt,
        _receipt(
            "goalzendo.g00f_coordinated_timeout_kill",
            budget,
            deadline,
            completed_monotonic_ns=time.monotonic_ns(),
            sigkill_pids=killed,
            timeout_command_exit_code=timeout_exit,
        ),
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
