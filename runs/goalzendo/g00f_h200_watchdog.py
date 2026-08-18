#!/usr/bin/env python3
"""Monotonic G00-F deadline watchdog with signal-before-receipt ordering."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import select
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

TERM_GRACE_SECONDS = 30
WATCHDOG_RECEIPT_MARGIN_SECONDS = 5
DEADLINE_MARKER_LEAD_NS = 2_000_000_000
GUARDIAN_PROTOCOL = "ready_pipe_pidfd_launcher_liveness_and_monotonic_group_cutoff_v1"


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
    os.chmod(path, 0o400)


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


def _group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return sys.platform != "darwin"
    return True


def _signal_group(pgid: int, signal_number: int) -> bool:
    try:
        os.killpg(pgid, signal_number)
    except (PermissionError, ProcessLookupError):
        return False
    return True


def _guardian_main(
    *,
    control_fd: int,
    launcher_liveness_fd: int,
    watchdog_pid: int,
    watchdog_pgid: int,
    launcher_pid: int,
    launcher_pgid: int,
    deadline_ns: int,
    term_grace_seconds: float = TERM_GRACE_SECONDS,
    receipt_margin_seconds: float = WATCHDOG_RECEIPT_MARGIN_SECONDS,
    marker_lead_ns: int = DEADLINE_MARKER_LEAD_NS,
) -> int:
    marker_acknowledged = False

    def acknowledge_marker(_signal_number: int, _frame: Any) -> None:
        nonlocal marker_acknowledged
        marker_acknowledged = True

    signal.signal(signal.SIGUSR1, acknowledge_marker)
    for signal_number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signal_number, signal.SIG_IGN)

    marker_ns = deadline_ns - marker_lead_ns
    marker_sent = False
    while True:
        now_ns = time.monotonic_ns()
        if not marker_sent and now_ns >= marker_ns:
            with suppress(ProcessLookupError):
                os.kill(launcher_pid, signal.SIGUSR2)
            marker_sent = True
        if now_ns >= deadline_ns:
            break
        next_boundary_ns = marker_ns if not marker_sent else deadline_ns
        timeout = min(0.2, max(0.0, (next_boundary_ns - now_ns) / 1_000_000_000))
        readable, _, _ = select.select(
            [control_fd, launcher_liveness_fd],
            [],
            [],
            timeout,
        )
        if launcher_liveness_fd in readable:
            # The pidfd is tied to the exact launcher process, so this cannot
            # be confused by PID reuse.  Clear both groups without touching
            # the network volume; the detached supervisor will preserve the
            # operational failure evidence.
            _signal_group(launcher_pgid, signal.SIGTERM)
            time.sleep(term_grace_seconds)
            _signal_group(launcher_pgid, signal.SIGKILL)
            _signal_group(watchdog_pgid, signal.SIGKILL)
            return 1
        if control_fd in readable:
            message = os.read(control_fd, 1)
            if message == b"N":
                return 0
            _signal_group(launcher_pgid, signal.SIGTERM)
            time.sleep(term_grace_seconds)
            _signal_group(launcher_pgid, signal.SIGKILL)
            _signal_group(watchdog_pgid, signal.SIGKILL)
            return 1

    # No filesystem operation is allowed before this hard monotonic signal.
    _signal_group(launcher_pgid, signal.SIGTERM)
    with suppress(ProcessLookupError):
        os.kill(watchdog_pid, signal.SIGALRM)
    kill_deadline_ns = deadline_ns + int(term_grace_seconds * 1_000_000_000)
    while time.monotonic_ns() < kill_deadline_ns:
        remaining = max(0.0, (kill_deadline_ns - time.monotonic_ns()) / 1_000_000_000)
        # Launcher exit is expected after the deadline TERM/KILL.  The pidfd
        # is therefore intentionally not a failure source in this phase; the
        # guardian stays alive solely to enforce the worker-group KILL and the
        # bounded post-signal receipt margin.
        readable, _, _ = select.select([control_fd], [], [], min(0.1, remaining))
        if control_fd in readable:
            message = os.read(control_fd, 1)
            if message == b"N":
                return 0 if marker_acknowledged else 125
            break
    _signal_group(launcher_pgid, signal.SIGKILL)
    with suppress(ProcessLookupError):
        os.kill(watchdog_pid, signal.SIGWINCH)
    fallback_deadline_ns = kill_deadline_ns + int(receipt_margin_seconds * 1_000_000_000)
    while time.monotonic_ns() < fallback_deadline_ns:
        remaining = max(0.0, (fallback_deadline_ns - time.monotonic_ns()) / 1_000_000_000)
        readable, _, _ = select.select([control_fd], [], [], min(0.1, remaining))
        if control_fd in readable and os.read(control_fd, 1) == b"N":
            return 0 if marker_acknowledged else 125
    _signal_group(watchdog_pgid, signal.SIGKILL)
    return 124 if marker_acknowledged else 125


def _launch_guardian(
    *,
    launcher_ready_fd: int,
    launcher_liveness_fd: int,
    watchdog_pid: int,
    watchdog_pgid: int,
    launcher_pid: int,
    launcher_pgid: int,
    deadline_ns: int,
) -> tuple[int, int]:
    control_read_fd, control_write_fd = os.pipe()
    ready_read_fd, ready_write_fd = os.pipe()
    try:
        guardian_pid = os.fork()
    except BaseException:
        for descriptor in (control_read_fd, control_write_fd, ready_read_fd, ready_write_fd):
            os.close(descriptor)
        raise
    if guardian_pid == 0:  # pragma: no cover - covered by subprocess tests
        os.close(control_write_fd)
        os.close(ready_read_fd)
        os.close(launcher_ready_fd)
        try:
            os.setsid()
            os.write(ready_write_fd, b"R")
            os.close(ready_write_fd)
            exit_code = _guardian_main(
                control_fd=control_read_fd,
                launcher_liveness_fd=launcher_liveness_fd,
                watchdog_pid=watchdog_pid,
                watchdog_pgid=watchdog_pgid,
                launcher_pid=launcher_pid,
                launcher_pgid=launcher_pgid,
                deadline_ns=deadline_ns,
            )
        except BaseException:
            exit_code = 125
        finally:
            with suppress(OSError):
                os.close(ready_write_fd)
            os.close(launcher_liveness_fd)
            os.close(control_read_fd)
        os._exit(exit_code)
    os.close(control_read_fd)
    os.close(ready_write_fd)
    try:
        readable, _, _ = select.select([ready_read_fd], [], [], 5.0)
        ready = os.read(ready_read_fd, 1) if readable else b""
    finally:
        os.close(ready_read_fd)
    if ready != b"R":
        os.close(control_write_fd)
        with suppress(ProcessLookupError):
            os.kill(guardian_pid, signal.SIGKILL)
        with suppress(ChildProcessError):
            os.waitpid(guardian_pid, 0)
        raise RuntimeError("watchdog guardian did not acknowledge readiness")
    return guardian_pid, control_write_fd


def _stop_guardian(guardian_pid: int, control_write_fd: int) -> tuple[bool, int | None]:
    with suppress(BrokenPipeError):
        os.write(control_write_fd, b"N")
    os.close(control_write_fd)
    deadline_ns = time.monotonic_ns() + 5_000_000_000
    while time.monotonic_ns() < deadline_ns:
        observed_pid, status = os.waitpid(guardian_pid, os.WNOHANG)
        if observed_pid == guardian_pid:
            exit_code = os.waitstatus_to_exitcode(status)
            return exit_code == 0, exit_code
        time.sleep(0.05)
    with suppress(ProcessLookupError):
        os.kill(guardian_pid, signal.SIGKILL)
    try:
        observed_pid, status = os.waitpid(guardian_pid, 0)
    except ChildProcessError:
        return False, None
    exit_code = os.waitstatus_to_exitcode(status)
    return observed_pid == guardian_pid and exit_code == 0, exit_code


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget-start", type=Path, required=True)
    parser.add_argument("--pid-file", type=Path, required=True)
    parser.add_argument("--started-receipt", type=Path, required=True)
    parser.add_argument("--normal-stop-receipt", type=Path, required=True)
    parser.add_argument("--owner-lock", type=Path, required=True)
    parser.add_argument("--claim-receipt", type=Path, required=True)
    parser.add_argument("--fired-receipt", type=Path, required=True)
    parser.add_argument("--cancel-receipt", type=Path, required=True)
    parser.add_argument("--kill-receipt", type=Path, required=True)
    parser.add_argument("--guardian-terminal-receipt", type=Path, required=True)
    parser.add_argument("--ready-fd", type=int, required=True)
    parser.add_argument("--launcher-pid", type=int, required=True)
    parser.add_argument("--launcher-process-group-id", type=int, required=True)
    parser.add_argument("timeout_command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    budget = json.loads(args.budget_start.read_text(encoding="utf-8"))
    budget_body = {key: value for key, value in budget.items() if key != "budget_digest"}
    if (
        budget.get("schema") != "goalzendo.g00f_h200_monotonic_budget_start"
        or budget.get("budget_digest") != _digest(budget_body)
        or budget.get("wall_ceiling_seconds") != 14 * 60 * 60
    ):
        raise RuntimeError("watchdog budget-start receipt is not authentic")
    start = int(budget["started_monotonic_ns"])
    ceiling = int(budget["wall_ceiling_seconds"]) * 1_000_000_000
    deadline = start + ceiling
    watchdog_pid = os.getpid()
    if (
        args.ready_fd < 0
        or args.launcher_pid <= 1
        or args.launcher_process_group_id <= 1
        or os.getppid() != args.launcher_pid
        or os.getpgid(args.launcher_pid) != args.launcher_process_group_id
    ):
        raise RuntimeError("watchdog launcher/ready-pipe identity is invalid")
    if sys.platform != "linux" or not hasattr(os, "pidfd_open"):
        raise RuntimeError("watchdog requires Linux pidfd launcher supervision")
    try:
        launcher_liveness_fd = os.pidfd_open(args.launcher_pid, 0)
    except OSError as error:
        raise RuntimeError("watchdog could not bind the exact launcher pidfd") from error
    try:
        os.setsid()
    except PermissionError as error:
        os.close(launcher_liveness_fd)
        raise RuntimeError("watchdog could not enter its independent session") from error
    watchdog_pgid = os.getpgrp()
    if watchdog_pgid != watchdog_pid:
        raise RuntimeError("watchdog did not become its independent process-group leader")
    normal_stop_requested = False
    guardian_term_observed_ns: int | None = None
    guardian_kill_observed_ns: int | None = None

    def request_normal_stop(_signal_number: int, _frame: Any) -> None:
        nonlocal normal_stop_requested
        normal_stop_requested = True

    def observe_guardian_term(_signal_number: int, _frame: Any) -> None:
        nonlocal guardian_term_observed_ns
        if guardian_term_observed_ns is None:
            guardian_term_observed_ns = time.monotonic_ns()

    def observe_guardian_kill(_signal_number: int, _frame: Any) -> None:
        nonlocal guardian_kill_observed_ns
        if guardian_kill_observed_ns is None:
            guardian_kill_observed_ns = time.monotonic_ns()

    signal.signal(signal.SIGTERM, request_normal_stop)
    signal.signal(signal.SIGALRM, observe_guardian_term)
    signal.signal(signal.SIGWINCH, observe_guardian_kill)

    guardian_pid, guardian_control_fd = _launch_guardian(
        launcher_ready_fd=args.ready_fd,
        launcher_liveness_fd=launcher_liveness_fd,
        watchdog_pid=watchdog_pid,
        watchdog_pgid=watchdog_pgid,
        launcher_pid=args.launcher_pid,
        launcher_pgid=args.launcher_process_group_id,
        deadline_ns=deadline,
    )
    os.close(launcher_liveness_fd)
    guardian_reaped = False

    def require_guardian_alive() -> None:
        nonlocal guardian_reaped
        if guardian_reaped:
            raise RuntimeError("watchdog guardian was already reaped")
        observed_pid, status = os.waitpid(guardian_pid, os.WNOHANG)
        if observed_pid == guardian_pid:
            guardian_reaped = True
            # The guardian is the I/O-independent hard boundary.  Its loss is
            # therefore an immediate fail-closed kill, not a receipt-writing
            # opportunity that could itself block on the network volume.
            _signal_group(args.launcher_process_group_id, signal.SIGKILL)
            raise RuntimeError(f"watchdog guardian exited unexpectedly: {os.waitstatus_to_exitcode(status)}")

    budget_start_file_sha256 = _sha256_file(args.budget_start)
    common = {
        "budget_start_file_sha256": budget_start_file_sha256,
        "watchdog_pid": watchdog_pid,
        "watchdog_process_group_id": watchdog_pgid,
        "launcher_pid": args.launcher_pid,
        "launcher_process_group_id": args.launcher_process_group_id,
        "guardian_pid": guardian_pid,
        "guardian_protocol": GUARDIAN_PROTOCOL,
        "term_grace_seconds": TERM_GRACE_SECONDS,
        "watchdog_receipt_margin_seconds": WATCHDOG_RECEIPT_MARGIN_SECONDS,
        "deadline_marker_lead_ns": DEADLINE_MARKER_LEAD_NS,
    }
    try:
        # This anonymous-pipe acknowledgement is deliberately before any
        # network-volume receipt I/O.  The launcher may wait for the durable
        # receipt too, but the guardian is already enforcing the deadline.
        os.write(args.ready_fd, f"R {guardian_pid}\n".encode("ascii"))
        os.close(args.ready_fd)
        started_monotonic_ns = time.monotonic_ns()
        _exclusive(
            args.started_receipt,
            _receipt(
                "goalzendo.g00f_h200_watchdog_started",
                budget,
                deadline,
                **common,
                started_monotonic_ns=started_monotonic_ns,
            ),
        )
    except BaseException:
        with suppress(OSError):
            os.close(args.ready_fd)
        # Closing without the normal-stop byte makes guardian EOF fail closed.
        with suppress(OSError):
            os.close(guardian_control_fd)
        raise

    def write_guardian_terminal(
        *,
        deadline_triggered: bool,
        clean_stop: bool,
        exit_code: int | None,
    ) -> None:
        _exclusive(
            args.guardian_terminal_receipt,
            _receipt(
                "goalzendo.g00f_h200_watchdog_guardian_terminal",
                budget,
                deadline,
                **common,
                completed_monotonic_ns=time.monotonic_ns(),
                deadline_triggered=deadline_triggered,
                guardian_clean_stop=clean_stop,
                guardian_exit_code=exit_code,
            ),
        )

    while True:
        require_guardian_alive()
        if normal_stop_requested:
            stopped_monotonic_ns = time.monotonic_ns()
            if stopped_monotonic_ns > deadline:
                raise RuntimeError("watchdog normal stop arrived after the monotonic deadline")
            _exclusive(
                args.normal_stop_receipt,
                _receipt(
                    "goalzendo.g00f_h200_watchdog_normal_stop",
                    budget,
                    deadline,
                    **common,
                    stopped_monotonic_ns=stopped_monotonic_ns,
                ),
            )
            clean_stop, guardian_exit_code = _stop_guardian(
                guardian_pid,
                guardian_control_fd,
            )
            guardian_reaped = True
            write_guardian_terminal(
                deadline_triggered=False,
                clean_stop=clean_stop,
                exit_code=guardian_exit_code,
            )
            return 0 if clean_stop else 70
        if guardian_term_observed_ns is not None:
            break
        remaining = deadline - time.monotonic_ns()
        if remaining <= 0:
            # The guardian owns the exact deadline signal.  Never perform a
            # filesystem operation while waiting for that in-memory notice.
            time.sleep(0.001)
            continue
        time.sleep(min(0.2, remaining / 1_000_000_000))
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signalled_ns = guardian_term_observed_ns
    if signalled_ns is None:
        raise RuntimeError("guardian deadline signal was not observed")

    # Everything below is post-signal evidence/reconciliation.  A blocked
    # read, lock, hash, fsync, or record-timeout command cannot postpone TERM
    # or KILL because the guardian remains independent of this process and the
    # network-volume filesystem.
    pids = [int(value) for value in args.pid_file.read_text(encoding="ascii").split()]
    if len(pids) != len(set(pids)) or any(pid <= 1 for pid in pids):
        raise RuntimeError("watchdog PID inventory is duplicated or unsafe")

    try:
        args.owner_lock.mkdir(mode=0o700)
    except FileExistsError:
        # TERM has already happened.  Keep the guardian armed through its KILL
        # boundary so an operational reconciler cannot leave a stuck worker.
        while guardian_kill_observed_ns is None:
            require_guardian_alive()
            time.sleep(0.05)
        clean_stop, guardian_exit_code = _stop_guardian(guardian_pid, guardian_control_fd)
        guardian_reaped = True
        write_guardian_terminal(
            deadline_triggered=True,
            clean_stop=clean_stop,
            exit_code=guardian_exit_code,
        )
        return 2

    # Claim timeout coordination before signalling so the launcher cannot race
    # the watchdog and become a second reconciler.  This operational IPC marker
    # is not the scientific timeout receipt: signal-before-timeout-receipt
    # ordering remains unchanged, and no evaluator/prediction/metric is read.
    claimed_ns = time.monotonic_ns()
    _exclusive(
        args.claim_receipt,
        _receipt(
            "goalzendo.g00f_h200_watchdog_deadline_claimed",
            budget,
            deadline,
            claimed_monotonic_ns=claimed_ns,
            pid_inventory_digest=_digest(pids),
            **common,
        ),
    )
    terminated = pids
    _exclusive(
        args.fired_receipt,
        _receipt(
            "goalzendo.g00f_h200_watchdog_deadline_fired",
            budget,
            deadline,
            **common,
            signalled_monotonic_ns=signalled_ns,
            sigterm_pids=terminated,
            signal_authority="in_memory_guardian_launcher_process_group",
        ),
    )
    _exclusive(
        args.cancel_receipt,
        _receipt(
            "goalzendo.g00f_h200_coordinated_timeout_cancel",
            budget,
            deadline,
            **common,
            cancel_monotonic_ns=signalled_ns,
            sigterm_pids=terminated,
            signal_authority="in_memory_guardian_launcher_process_group",
        ),
    )

    command = list(args.timeout_command)
    if command and command[0] == "--":
        command = command[1:]
    timeout_command_start_failed = False
    try:
        # Deliberately inherit the watchdog group: the guardian's independent
        # fallback KILL owns this reconciliation child too.
        timeout_process = (
            subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if command
            else None
        )
    except OSError:
        timeout_process = None
        timeout_command_start_failed = True
    while guardian_kill_observed_ns is None:
        require_guardian_alive()
        time.sleep(0.05)
    killed = pids
    timeout_command_killed = False
    if timeout_process is None:
        timeout_exit = 66 if timeout_command_start_failed else 64
    elif timeout_process.poll() is None:
        timeout_command_killed = True
        with suppress(ProcessLookupError):
            os.kill(timeout_process.pid, signal.SIGKILL)
        try:
            timeout_exit = timeout_process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            timeout_exit = 124
    else:
        timeout_exit = int(timeout_process.returncode)
    _exclusive(
        args.kill_receipt,
        _receipt(
            "goalzendo.g00f_h200_coordinated_timeout_kill",
            budget,
            deadline,
            **common,
            completed_monotonic_ns=time.monotonic_ns(),
            sigkill_pids=killed,
            guardian_sigkill_observed_monotonic_ns=guardian_kill_observed_ns,
            timeout_command_exit_code=timeout_exit,
            timeout_command_killed=timeout_command_killed,
            timeout_command_start_failed=timeout_command_start_failed,
        ),
    )
    clean_stop, guardian_exit_code = _stop_guardian(guardian_pid, guardian_control_fd)
    guardian_reaped = True
    write_guardian_terminal(
        deadline_triggered=True,
        clean_stop=clean_stop,
        exit_code=guardian_exit_code,
    )
    return 1 if clean_stop else 70


if __name__ == "__main__":
    raise SystemExit(main())
