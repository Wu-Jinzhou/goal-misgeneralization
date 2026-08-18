#!/usr/bin/env python3
"""Persistent, receipt-producing supervisor for the frozen H200 launcher."""

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

TERM_GRACE_SECONDS = 90


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


def _exclusive_json(path: Path, body: dict[str, Any], *, mode: int = 0o400) -> None:
    payload = {**body, "receipt_digest": _digest(body)}
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, mode)


def _exclusive_pid(path: Path, pid: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="ascii") as handle:
        handle.write(f"{pid}\n")
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


def _gate_child(read_fd: int, launcher: str) -> int:
    try:
        release = os.read(read_fd, 1)
    finally:
        os.close(read_fd)
    if release != b"G":
        return 125
    os.execv(launcher, [launcher])
    return 125


def _stop_launcher_group(
    process: subprocess.Popen[bytes] | None,
    pgid: int,
    *,
    grace_seconds: int,
) -> tuple[bool, bool, int | None, bool]:
    term_sent = False
    kill_sent = False
    if _group_exists(pgid):
        try:
            os.killpg(pgid, signal.SIGTERM)
            term_sent = True
        except (PermissionError, ProcessLookupError):
            pass
    deadline_ns = time.monotonic_ns() + grace_seconds * 1_000_000_000
    while _group_exists(pgid):
        if process is not None:
            process.poll()
        remaining_ns = deadline_ns - time.monotonic_ns()
        if remaining_ns <= 0:
            break
        time.sleep(min(0.1, remaining_ns / 1_000_000_000))
    if _group_exists(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
            kill_sent = True
        except (PermissionError, ProcessLookupError):
            pass
    if process is None:
        exit_code = None
    else:
        try:
            exit_code = process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            exit_code = None
    clear_deadline_ns = time.monotonic_ns() + 5 * 1_000_000_000
    while _group_exists(pgid) and time.monotonic_ns() < clear_deadline_ns:
        if process is not None:
            process.poll()
        time.sleep(0.05)
    return term_sent, kill_sent, exit_code, not _group_exists(pgid)


def _guardian_main(read_fd: int, launcher_pgid: int) -> int:
    for signal_number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signal_number, signal.SIG_IGN)
    while True:
        readable, _, _ = select.select([read_fd], [], [], 0.2)
        if not readable:
            continue
        message = os.read(read_fd, 1)
        if message == b"N":
            return 0
        _stop_launcher_group(None, launcher_pgid, grace_seconds=TERM_GRACE_SECONDS)
        return 1


def _launch_guardian(launcher_pgid: int, inherited_gate_write_fd: int) -> tuple[int, int]:
    read_fd, write_fd = os.pipe()
    ready_read_fd, ready_write_fd = os.pipe()
    try:
        guardian_pid = os.fork()
    except BaseException:
        for descriptor in (read_fd, write_fd, ready_read_fd, ready_write_fd):
            os.close(descriptor)
        raise
    if guardian_pid == 0:  # pragma: no cover - exercised in subprocess tests
        os.close(write_fd)
        os.close(ready_read_fd)
        os.close(inherited_gate_write_fd)
        try:
            os.setsid()
            os.write(ready_write_fd, b"R")
            os.close(ready_write_fd)
            exit_code = _guardian_main(read_fd, launcher_pgid)
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
        raise RuntimeError("detached guardian did not acknowledge readiness")
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
    parser = argparse.ArgumentParser(prog="g00f_h200_detached_supervisor")
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--execution-uuid", required=True)
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument("--operator-handoff-sha256", required=True)
    parser.add_argument("--execution-handoff-sha256", required=True)
    parser.add_argument("--pid-file", type=Path, required=True)
    parser.add_argument("--started-receipt", type=Path, required=True)
    parser.add_argument("--terminal-receipt", type=Path, required=True)
    args = parser.parse_args()

    launcher_direct = args.launcher
    launcher = launcher_direct.resolve()
    execution_root = args.execution_root.resolve()
    supervisor = Path(__file__).resolve()
    for label, value in (
        ("operator handoff", args.operator_handoff_sha256),
        ("execution handoff", args.execution_handoff_sha256),
    ):
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise SystemExit(f"{label} SHA-256 is malformed")
    if launcher_direct.is_symlink() or not launcher.is_file() or not os.access(launcher, os.X_OK):
        raise SystemExit("launcher is not one direct executable regular file")
    expected_root = Path("/workspace/status-goalzendo/g00f-executions") / args.execution_uuid
    if execution_root != expected_root:
        raise SystemExit("execution root does not match the pinned execution UUID")
    if any(
        path.exists() or path.is_symlink()
        for path in (args.pid_file, args.started_receipt, args.terminal_receipt)
    ):
        raise SystemExit("detached-supervisor receipt path already exists")

    supervisor_pid = os.getpid()
    started_monotonic_ns = time.monotonic_ns()
    _exclusive_pid(args.pid_file, supervisor_pid)
    launcher_sha256 = _sha256_file(launcher)
    supervisor_sha256 = _sha256_file(supervisor)
    started_at_utc = _utc_now()
    received_signal: int | None = None

    def request_stop(signal_number: int, _frame: Any) -> None:
        nonlocal received_signal
        received_signal = signal_number

    for signal_number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signal_number, request_stop)

    common_without_launcher_process = {
        "schema_version": 1,
        "execution_uuid": args.execution_uuid,
        "execution_root": str(execution_root),
        "operator_handoff_sha256": args.operator_handoff_sha256,
        "execution_handoff_sha256": args.execution_handoff_sha256,
        "launcher_path": str(launcher),
        "launcher_sha256": launcher_sha256,
        "supervisor_path": str(supervisor),
        "supervisor_sha256": supervisor_sha256,
        "supervisor_pid": supervisor_pid,
        "term_grace_seconds": TERM_GRACE_SECONDS,
        "started_at_utc": started_at_utc,
        "started_monotonic_ns": started_monotonic_ns,
        "outcomes_seen": False,
        "g01_launch_authorized": False,
    }
    gate_read_fd, gate_write_fd = os.pipe()
    process = subprocess.Popen(
        [sys.executable, str(supervisor), "--gate-child", str(gate_read_fd), str(launcher)],
        start_new_session=True,
        pass_fds=(gate_read_fd,),
    )
    os.close(gate_read_fd)
    launcher_pgid = process.pid
    try:
        guardian_pid, guardian_write_fd = _launch_guardian(launcher_pgid, gate_write_fd)
        common = {
            **common_without_launcher_process,
            "launcher_pid": process.pid,
            "launcher_process_group_id": launcher_pgid,
            "guardian_pid": guardian_pid,
            "guardian_protocol": "ready_pipe_plus_parent_eof_launcher_group_cleanup_v1",
        }
        _exclusive_json(
            args.started_receipt,
            {"schema": "goalzendo.g00f_h200_detached_supervisor_started", **common},
        )
        os.write(gate_write_fd, b"G")
        os.close(gate_write_fd)
    except BaseException:
        with suppress(OSError):
            os.close(gate_write_fd)
        _stop_launcher_group(process, launcher_pgid, grace_seconds=0)
        if "guardian_pid" in locals():
            _stop_guardian(guardian_pid, guardian_write_fd)
        raise

    guardian_failed = False
    guardian_exit_code: int | None = None
    while process.poll() is None and received_signal is None:
        observed_guardian, guardian_status = os.waitpid(guardian_pid, os.WNOHANG)
        if observed_guardian == guardian_pid:
            guardian_exit_code = os.waitstatus_to_exitcode(guardian_status)
            guardian_failed = True
            _stop_launcher_group(process, launcher_pgid, grace_seconds=0)
            break
        time.sleep(0.2)
    launcher_exit = process.poll()
    lingering_group = _group_exists(launcher_pgid)
    term_sent = False
    kill_sent = False
    descendants_clear = not lingering_group
    if received_signal is not None or lingering_group:
        term_sent, kill_sent, stopped_exit, descendants_clear = _stop_launcher_group(
            process,
            launcher_pgid,
            grace_seconds=TERM_GRACE_SECONDS,
        )
        if launcher_exit is None:
            launcher_exit = stopped_exit
    elif launcher_exit is None:
        launcher_exit = process.wait()
    if guardian_exit_code is None:
        guardian_clean_stop = _stop_guardian(guardian_pid, guardian_write_fd)
    else:
        os.close(guardian_write_fd)
        guardian_clean_stop = False
    completed_monotonic_ns = time.monotonic_ns()
    success = (
        launcher_exit == 0
        and received_signal is None
        and descendants_clear
        and not lingering_group
        and guardian_clean_stop
        and not guardian_failed
    )
    try:
        _exclusive_json(
            args.terminal_receipt,
            {
                "schema": "goalzendo.g00f_h200_detached_supervisor_terminal",
                **common,
                "completed_at_utc": _utc_now(),
                "completed_monotonic_ns": completed_monotonic_ns,
                "elapsed_seconds": (completed_monotonic_ns - started_monotonic_ns) / 1_000_000_000,
                "launcher_exit_code": launcher_exit,
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
    except BaseException:
        _stop_launcher_group(process, launcher_pgid, grace_seconds=0)
        raise
    if success:
        return 0
    if received_signal is not None:
        return 128 + received_signal
    if launcher_exit is not None and launcher_exit != 0:
        return launcher_exit
    return 125


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--gate-child":
        raise SystemExit(_gate_child(int(sys.argv[2]), sys.argv[3]))
    raise SystemExit(main())
