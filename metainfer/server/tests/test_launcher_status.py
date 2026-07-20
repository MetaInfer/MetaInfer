"""Direct tests for :meth:`LocalLauncher.status`.

The status() method reads ``orchestrator.pid`` and decides whether the
orchestrator is running. These tests pin its three branches so the
multi-node bug (``finished_at`` ignored when ``pid`` is present) doesn't
regress.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from metainfer.server.launcher import LocalLauncher


def _write_pid_file(home: Path, task_id: str, payload: dict) -> Path:
    """Drop a fake orchestrator.pid under <home>/tasks/<task_id>/."""
    task_dir = home / "tasks" / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    pf = task_dir / "orchestrator.pid"
    pf.write_text(json.dumps(payload), encoding="utf-8")
    return pf


def test_status_finished_at_short_circuits_alive_pid(isolated_env):
    """Regression: when ``finished_at`` is stamped in the pid file, the
    orchestrator is dead by definition — even if some unrelated process
    on this host happens to have recycled the same PID number. The old
    code did a liveness check here and could return running=True on a
    different machine via shared NFS storage."""
    home = isolated_env["home"]
    # Use THIS process's pid + started_at so validate_pid_started_at would
    # return True if status() were to call it. The bug is that it must NOT
    # get that far once finished_at is set.
    me = Path("/proc/self/stat").read_text()
    rparen = me.rfind(")")
    start_ticks = int(me[rparen + 2:].split()[19])
    boot_time = float(Path("/proc/stat").read_text().split("btime ")[1].split()[0])
    clk_tck = 100
    my_started_at = boot_time + (start_ticks / clk_tck)

    _write_pid_file(home, "t-finished", {
        "pid": Path("/proc/self").resolve().name,  # this Python process
        "task_id": "t-finished",
        "started_at": my_started_at,
        "finished_at": time.time(),
        "exit_hint": "reaped-by-kill-on-dead-pid",
    })

    launcher = LocalLauncher()
    status = launcher.status("t-finished")
    assert status.running is False
    assert status.finished_at is not None
    assert status.exit_hint == "reaped-by-kill-on-dead-pid"


def test_status_no_pid_file(isolated_env):
    """No pid file at all → not-running with the canonical exit_hint."""
    launcher = LocalLauncher()
    status = launcher.status("never-spawned")
    assert status.running is False
    assert status.pid is None
    assert status.finished_at is None
    assert status.exit_hint == "no-pid-file"


def test_status_pid_dead_no_finished_at(isolated_env):
    """pid present, finished_at absent, pid is genuinely dead (PID 0 is
    reserved by the kernel and never resolves to a live process). Should
    return running=False via the liveness check."""
    home = isolated_env["home"]
    _write_pid_file(home, "t-crashed", {
        "pid": 999_999_989,  # almost certainly unused; validate_pid_started_at will return False
        "task_id": "t-crashed",
        "started_at": time.time(),
    })

    launcher = LocalLauncher()
    status = launcher.status("t-crashed")
    assert status.running is False
    assert status.exit_hint == "pid-dead"


def test_status_pid_alive_no_finished_at(isolated_env):
    """Golden-path: orchestrator is THIS Python process (we know it's
    alive). finished_at is absent, so status() should walk the liveness
    branch and report running=True."""
    home = isolated_env["home"]
    me = Path("/proc/self/stat").read_text()
    rparen = me.rfind(")")
    start_ticks = int(me[rparen + 2:].split()[19])
    boot_time = float(Path("/proc/stat").read_text().split("btime ")[1].split()[0])
    clk_tck = 100
    my_started_at = boot_time + (start_ticks / clk_tck)
    my_pid = int(Path("/proc/self").resolve().name)

    _write_pid_file(home, "t-alive", {
        "pid": my_pid,
        "task_id": "t-alive",
        "started_at": my_started_at,
    })

    launcher = LocalLauncher()
    status = launcher.status("t-alive")
    assert status.running is True
    assert status.exit_hint == "pid-alive"
    assert status.finished_at is None
