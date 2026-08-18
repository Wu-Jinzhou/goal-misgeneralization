#!/usr/bin/env python3
"""Hard monotonic process-group supervisor for H200 engineering qualification."""

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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CEILING_SECONDS = 6_300
TERM_GRACE_SECONDS = 30


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _exclusive(path: Path, body: dict[str, Any]) -> None:
    payload = {**body, "receipt_digest": _digest(body)}
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o400)


def _group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return sys.platform != "darwin"
    return True


def _gate_child(read_fd: int, command: list[str]) -> int:
    try:
        release = os.read(read_fd, 1)
    finally:
        os.close(read_fd)
    if release != b"G" or not command:
        return 125
    os.execvp(command[0], command)
    return 125


def _terminate_controller_group(
    pgid: int,
    leader: subprocess.Popen[bytes] | None = None,
) -> tuple[bool, bool, bool]:
    term_sent = False
    kill_sent = False
    if _group_exists(pgid):
        with suppress(ProcessLookupError):
            os.killpg(pgid, signal.SIGTERM)
            term_sent = True
    kill_deadline_ns = time.monotonic_ns() + TERM_GRACE_SECONDS * 1_000_000_000
    while _group_exists(pgid):
        if leader is not None:
            leader.poll()
        remaining_ns = kill_deadline_ns - time.monotonic_ns()
        if remaining_ns <= 0:
            break
        time.sleep(min(0.1, remaining_ns / 1_000_000_000))
    if _group_exists(pgid):
        with suppress(ProcessLookupError):
            os.killpg(pgid, signal.SIGKILL)
            kill_sent = True
    clear_deadline_ns = time.monotonic_ns() + 5 * 1_000_000_000
    while _group_exists(pgid) and time.monotonic_ns() < clear_deadline_ns:
        if leader is not None:
            leader.poll()
        time.sleep(0.05)
    return term_sent, kill_sent, not _group_exists(pgid)


def _guardian_main(
    *,
    read_fd: int,
    supervisor_pid: int,
    controller_pgid: int,
    deadline_ns: int,
) -> int:
    for signal_number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signal_number, signal.SIG_IGN)
    while True:
        remaining_seconds = max(0.0, (deadline_ns - time.monotonic_ns()) / 1_000_000_000)
        readable, _, _ = select.select([read_fd], [], [], min(0.2, remaining_seconds))
        if readable:
            message = os.read(read_fd, 1)
            if message == b"N":
                return 0
            _terminate_controller_group(controller_pgid)
            return 1
        if time.monotonic_ns() >= deadline_ns:
            _terminate_controller_group(controller_pgid)
            with suppress(ProcessLookupError):
                os.kill(supervisor_pid, signal.SIGTERM)
            parent_deadline_ns = time.monotonic_ns() + 5 * 1_000_000_000
            while os.getppid() == supervisor_pid and time.monotonic_ns() < parent_deadline_ns:
                time.sleep(0.05)
            if os.getppid() == supervisor_pid:
                with suppress(ProcessLookupError):
                    os.kill(supervisor_pid, signal.SIGKILL)
            return 124


def _launch_guardian(
    *,
    supervisor_pid: int,
    controller_pgid: int,
    deadline_ns: int,
    inherited_gate_write_fd: int,
) -> tuple[int, int]:
    read_fd, write_fd = os.pipe()
    ready_read_fd, ready_write_fd = os.pipe()
    try:
        guardian_pid = os.fork()
    except BaseException:
        os.close(read_fd)
        os.close(write_fd)
        os.close(ready_read_fd)
        os.close(ready_write_fd)
        raise
    if guardian_pid == 0:  # pragma: no cover - exercised in subprocess tests
        os.close(write_fd)
        os.close(ready_read_fd)
        os.close(inherited_gate_write_fd)
        try:
            os.setsid()
            os.write(ready_write_fd, b"R")
            os.close(ready_write_fd)
            exit_code = _guardian_main(
                read_fd=read_fd,
                supervisor_pid=supervisor_pid,
                controller_pgid=controller_pgid,
                deadline_ns=deadline_ns,
            )
        except BaseException:
            exit_code = 125
        finally:
            with suppress(OSError):
                os.close(ready_write_fd)
            os.close(read_fd)
        os._exit(exit_code)
    os.close(read_fd)
    os.close(ready_write_fd)
    try:
        readable, _, _ = select.select([ready_read_fd], [], [], 5.0)
        ready = os.read(ready_read_fd, 1) if readable else b""
    finally:
        os.close(ready_read_fd)
    if ready != b"R":
        os.close(write_fd)
        with suppress(ProcessLookupError):
            os.kill(guardian_pid, signal.SIGKILL)
        with suppress(ChildProcessError):
            os.waitpid(guardian_pid, 0)
        raise RuntimeError("qualification guardian did not acknowledge readiness")
    return guardian_pid, write_fd


def _stop_guardian(guardian_pid: int, write_fd: int) -> bool:
    with suppress(BrokenPipeError):
        os.write(write_fd, b"N")
    os.close(write_fd)
    deadline_ns = time.monotonic_ns() + 5 * 1_000_000_000
    while time.monotonic_ns() < deadline_ns:
        observed_pid, status = os.waitpid(guardian_pid, os.WNOHANG)
        if observed_pid == guardian_pid:
            return os.waitstatus_to_exitcode(status) == 0
        time.sleep(0.05)
    with suppress(ProcessLookupError):
        os.kill(guardian_pid, signal.SIGKILL)
    observed_pid, status = os.waitpid(guardian_pid, 0)
    return observed_pid == guardian_pid and os.waitstatus_to_exitcode(status) == 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="g00f_h200_qualification_supervisor")
    parser.add_argument("--execution-uuid", required=True)
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument("--started-receipt", type=Path, required=True)
    parser.add_argument("--term-receipt", type=Path, required=True)
    parser.add_argument("--kill-receipt", type=Path, required=True)
    parser.add_argument("--terminal-receipt", type=Path, required=True)
    parser.add_argument("controller_command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = list(args.controller_command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("qualification controller command is absent")
    controller_direct = Path(command[0]) if len(command) == 1 else Path(command[1])
    controller = controller_direct.resolve()
    if controller_direct.is_symlink() or not controller.is_file():
        raise SystemExit("qualification controller is not one direct regular file")
    execution_root = args.execution_root.resolve()
    expected_root = Path("/workspace/status-goalzendo/g00f-executions") / args.execution_uuid
    if execution_root != expected_root:
        raise SystemExit("qualification execution root differs from its UUID")
    receipt_paths = (args.started_receipt, args.term_receipt, args.kill_receipt, args.terminal_receipt)
    if any(path.exists() or path.is_symlink() for path in receipt_paths):
        raise SystemExit("qualification-supervisor receipt path already exists")

    supervisor = Path(__file__).resolve()
    controller_sha256 = _sha256_file(controller)
    supervisor_sha256 = _sha256_file(supervisor)
    received_signal: int | None = None

    def request_stop(signal_number: int, _frame: Any) -> None:
        nonlocal received_signal
        received_signal = signal_number

    for signal_number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signal_number, request_stop)
    started_ns = time.monotonic_ns()
    deadline_ns = started_ns + CEILING_SECONDS * 1_000_000_000
    gate_read_fd, gate_write_fd = os.pipe()
    guarded_command = [
        sys.executable,
        str(supervisor),
        "--gate-child",
        str(gate_read_fd),
        "--",
        *command,
    ]
    process = subprocess.Popen(
        guarded_command,
        start_new_session=True,
        pass_fds=(gate_read_fd,),
    )
    os.close(gate_read_fd)
    pgid = process.pid
    try:
        guardian_pid, guardian_write_fd = _launch_guardian(
            supervisor_pid=os.getpid(),
            controller_pgid=pgid,
            deadline_ns=deadline_ns,
            inherited_gate_write_fd=gate_write_fd,
        )
    except BaseException:
        os.close(gate_write_fd)
        _terminate_controller_group(pgid, process)
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=5)
        raise
    common = {
        "schema_version": 1,
        "execution_uuid": args.execution_uuid,
        "execution_root": str(execution_root),
        "controller_argv": command,
        "controller_path": str(controller),
        "controller_sha256": controller_sha256,
        "supervisor_path": str(supervisor),
        "supervisor_sha256": supervisor_sha256,
        "supervisor_pid": os.getpid(),
        "controller_pid": process.pid,
        "controller_process_group_id": pgid,
        "guardian_pid": guardian_pid,
        "guardian_protocol": "independent_session_pipe_eof_or_monotonic_deadline_group_cleanup_v1",
        "ceiling_seconds": CEILING_SECONDS,
        "term_grace_seconds": TERM_GRACE_SECONDS,
        "started_at_utc": _utc_now(),
        "started_monotonic_ns": started_ns,
        "deadline_monotonic_ns": deadline_ns,
        "outcomes_seen": False,
        "g01_launch_authorized": False,
    }
    try:
        _exclusive(
            args.started_receipt,
            {"schema": "goalzendo.g00f_h200_qualification_supervisor_started", **common},
        )
        os.write(gate_write_fd, b"G")
        os.close(gate_write_fd)
    except BaseException:
        with suppress(OSError):
            os.close(gate_write_fd)
        _terminate_controller_group(pgid, process)
        _stop_guardian(guardian_pid, guardian_write_fd)
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=5)
        raise

    timed_out = False
    guardian_failed = False
    guardian_exit_code: int | None = None
    while process.poll() is None:
        observed_guardian, guardian_status = os.waitpid(guardian_pid, os.WNOHANG)
        if observed_guardian == guardian_pid:
            guardian_exit_code = os.waitstatus_to_exitcode(guardian_status)
            guardian_failed = True
            _terminate_controller_group(pgid, process)
            break
        if received_signal is not None:
            break
        remaining_ns = deadline_ns - time.monotonic_ns()
        if remaining_ns <= 0:
            timed_out = True
            break
        time.sleep(min(0.2, remaining_ns / 1_000_000_000))

    observed_completion_ns = time.monotonic_ns()
    if observed_completion_ns > deadline_ns:
        timed_out = True
    leader_exit = process.poll()
    lingering_group = _group_exists(pgid)
    term_sent = False
    kill_sent = False
    interrupted = received_signal is not None
    receipt_error: BaseException | None = None
    if timed_out or lingering_group or interrupted:
        term_ns = time.monotonic_ns()
        term_sent, kill_sent, _ = _terminate_controller_group(pgid, process)
        reason = (
            "deadline"
            if timed_out
            else ("supervisor_signal" if interrupted else "controller_exit_with_live_descendants")
        )
        try:
            _exclusive(
                args.term_receipt,
                {
                    "schema": "goalzendo.g00f_h200_qualification_supervisor_term",
                    **common,
                    "reason": reason,
                    "received_signal": received_signal,
                    "term_monotonic_ns": term_ns,
                },
            )
        except BaseException as error:
            receipt_error = error
        try:
            _exclusive(
                args.kill_receipt,
                {
                    "schema": "goalzendo.g00f_h200_qualification_supervisor_kill",
                    **common,
                    "kill_monotonic_ns": time.monotonic_ns(),
                    "sigterm_sent": term_sent,
                    "sigkill_sent": kill_sent,
                },
            )
        except BaseException as error:
            if receipt_error is None:
                receipt_error = error

    if process.poll() is None:
        try:
            leader_exit = process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            leader_exit = None
    else:
        leader_exit = process.returncode
    descendants_clear = not _group_exists(pgid)
    if guardian_exit_code is None:
        guardian_clean_stop = _stop_guardian(guardian_pid, guardian_write_fd)
    else:
        os.close(guardian_write_fd)
        guardian_clean_stop = False
    if receipt_error is not None:
        raise receipt_error
    completed_ns = time.monotonic_ns()
    success = (
        leader_exit == 0
        and not timed_out
        and not interrupted
        and completed_ns <= deadline_ns
        and descendants_clear
        and not lingering_group
        and guardian_clean_stop
        and not guardian_failed
    )
    terminal_exit = (
        0
        if success
        else (
            124
            if timed_out
            else (
                128 + int(received_signal)
                if received_signal is not None
                else (int(leader_exit) if leader_exit else 125)
            )
        )
    )
    _exclusive(
        args.terminal_receipt,
        {
            "schema": "goalzendo.g00f_h200_qualification_supervisor_terminal",
            **common,
            "completed_at_utc": _utc_now(),
            "completed_monotonic_ns": completed_ns,
            "elapsed_seconds": (completed_ns - started_ns) / 1_000_000_000,
            "controller_exit_code": leader_exit,
            "deadline_triggered": timed_out,
            "received_signal": received_signal,
            "sigterm_sent": term_sent,
            "sigkill_sent": kill_sent,
            "descendants_clear": descendants_clear,
            "guardian_clean_stop": guardian_clean_stop,
            "guardian_exit_code": guardian_exit_code,
            "guardian_failed": guardian_failed,
            "success": success,
        },
    )
    return terminal_exit


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--gate-child":
        separator = sys.argv.index("--", 3)
        raise SystemExit(_gate_child(int(sys.argv[2]), sys.argv[separator + 1 :]))
    raise SystemExit(main())
