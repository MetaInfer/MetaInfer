"""Tests for :mod:`metainfer.server.liveness`.

Verifies that the periodic scan detects orchestrator processes that
died ungracefully mid-run and invokes the existing reaper so the UI
doesn't freeze on a stale "running" snapshot.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from metainfer.server import liveness as _liveness
from metainfer.server import tasks as _tasks
from metainfer.server.launcher import ProcStatus


def _make_entry(tid):
    """Minimal TaskEntry-like object for the liveness scan.

    Identity only — process state (pid/started_at/finished_at) is NOT
    stored in the registry; liveness calls launcher.status() per task.
    """
    return MagicMock(
        id=tid,
        state_dir=f"/tmp/{tid}", workspace_dir=f"/tmp/{tid}-ws",
    )


def test_scan_calls_status_on_every_task(monkeypatch):
    """No pre-filter: every task in the registry gets probed, because
    process state lives in orchestrator.pid (not the registry) and we
    can't know which tasks are running without checking."""
    entries = [_make_entry("a"), _make_entry("b"), _make_entry("c")]
    monkeypatch.setattr(_tasks, "list_tasks", lambda: entries)

    launcher = MagicMock()
    launcher.status.return_value = ProcStatus(
        running=False, pid=None, started_at=None,
        finished_at=1.0, exit_hint="pid-file-cleared",
    )
    checker = _liveness.LivenessChecker(launcher=launcher)
    checker._scan_once()

    probed = [call.args[0] for call in launcher.status.call_args_list]
    assert probed == ["a", "b", "c"]


def test_scan_ignores_alive_orchestrator(monkeypatch):
    """A live orchestrator (status.running=True) is left alone."""
    monkeypatch.setattr(_tasks, "list_tasks",
                        lambda: [_make_entry("alive")])
    launcher = MagicMock()
    launcher.status.return_value = ProcStatus(
        running=True, pid=123, started_at=1.0,
        finished_at=None, exit_hint="pid-alive",
    )
    # _reap_dead_pid_file must not be called for live processes
    launcher._reap_dead_pid_file = MagicMock()
    # But LocalLauncher check uses isinstance, so make it look like one
    from metainfer.server.launcher import LocalLauncher
    launcher.__class__ = LocalLauncher

    checker = _liveness.LivenessChecker(launcher=launcher)
    checker._scan_once()

    launcher._reap_dead_pid_file.assert_not_called()


def test_scan_reaps_dead_orchestrator(monkeypatch):
    """The bug scenario: orchestrator pid file claims running, /proc
    says dead. Scanner must invoke the reaper so the UI flips to stopped."""
    monkeypatch.setattr(_tasks, "list_tasks",
                        lambda: [_make_entry("dead")])
    launcher = MagicMock()
    launcher.status.return_value = ProcStatus(
        running=False, pid=999, started_at=1.0,
        finished_at=None, exit_hint="pid-dead",
    )
    from metainfer.server.launcher import LocalLauncher
    launcher.__class__ = LocalLauncher
    launcher._reap_dead_pid_file = MagicMock()

    checker = _liveness.LivenessChecker(launcher=launcher)
    checker._scan_once()

    launcher._reap_dead_pid_file.assert_called_once_with("dead", 999, 1.0)


def test_scan_ignores_already_cleared_pid_file(monkeypatch):
    """exit_hint 'no-pid-file' / 'pid-file-cleared' are bookkeeping
    states — the orchestrator already wrote finished_at itself, so we
    must not double-write a reap event."""
    monkeypatch.setattr(_tasks, "list_tasks",
                        lambda: [_make_entry("clean")])
    launcher = MagicMock()
    launcher.status.return_value = ProcStatus(
        running=False, pid=None, started_at=None,
        finished_at=1234.0, exit_hint="pid-file-cleared",
    )
    from metainfer.server.launcher import LocalLauncher
    launcher.__class__ = LocalLauncher
    launcher._reap_dead_pid_file = MagicMock()

    checker = _liveness.LivenessChecker(launcher=launcher)
    checker._scan_once()

    launcher._reap_dead_pid_file.assert_not_called()


def test_scan_skips_when_started_at_missing(monkeypatch):
    """PID-reuse safety: without started_at we can't validate. Don't reap."""
    monkeypatch.setattr(_tasks, "list_tasks",
                        lambda: [_make_entry("risky")])
    launcher = MagicMock()
    launcher.status.return_value = ProcStatus(
        running=False, pid=123, started_at=None,  # ← missing
        finished_at=None, exit_hint="pid-dead",
    )
    from metainfer.server.launcher import LocalLauncher
    launcher.__class__ = LocalLauncher
    launcher._reap_dead_pid_file = MagicMock()

    checker = _liveness.LivenessChecker(launcher=launcher)
    checker._scan_once()

    launcher._reap_dead_pid_file.assert_not_called()


def test_scan_survives_per_task_exception(monkeypatch):
    """A buggy entry must not kill the watcher — other tasks still get checked."""
    entries = [_make_entry("bad"), _make_entry("good")]
    monkeypatch.setattr(_tasks, "list_tasks", lambda: entries)

    call_log = []

    def fake_status(tid):
        call_log.append(tid)
        if tid == "bad":
            raise RuntimeError("simulated registry corruption")
        return ProcStatus(
            running=False, pid=2, started_at=2.0,
            finished_at=None, exit_hint="pid-dead",
        )

    launcher = MagicMock()
    launcher.status = fake_status
    from metainfer.server.launcher import LocalLauncher
    launcher.__class__ = LocalLauncher
    launcher._reap_dead_pid_file = MagicMock()

    checker = _liveness.LivenessChecker(launcher=launcher)
    checker._scan_once()  # must not raise

    # Both tasks were attempted
    assert call_log == ["bad", "good"]
    # "good" was reaped, "bad" was skipped (exception swallowed)
    launcher._reap_dead_pid_file.assert_called_once_with("good", 2, 2.0)


def test_scan_survives_registry_read_failure(monkeypatch):
    """If the registry itself can't be read, the scan logs and exits
    cleanly — the next interval will retry."""
    def boom():
        raise RuntimeError("disk gone")
    monkeypatch.setattr(_tasks, "list_tasks", boom)

    launcher = MagicMock()
    checker = _liveness.LivenessChecker(launcher=launcher)
    checker._scan_once()  # must not raise

    launcher.status.assert_not_called()


def test_scan_survives_per_task_exception(monkeypatch):
    """A buggy entry must not kill the watcher — other tasks still get checked."""
    entries = [_make_entry("bad"), _make_entry("good")]
    monkeypatch.setattr(_tasks, "list_tasks", lambda: entries)

    call_log = []

    def fake_status(tid):
        call_log.append(tid)
        if tid == "bad":
            raise RuntimeError("simulated registry corruption")
        return ProcStatus(
            running=False, pid=2, started_at=2.0,
            finished_at=None, exit_hint="pid-dead",
        )

    launcher = MagicMock()
    launcher.status = fake_status
    from metainfer.server.launcher import LocalLauncher
    launcher.__class__ = LocalLauncher
    launcher._reap_dead_pid_file = MagicMock()

    checker = _liveness.LivenessChecker(launcher=launcher)
    checker._scan_once()  # must not raise

    # Both tasks were attempted
    assert call_log == ["bad", "good"]
    # "good" was reaped, "bad" was skipped (exception swallowed)
    launcher._reap_dead_pid_file.assert_called_once_with("good", 2, 2.0)


def test_scan_survives_registry_read_failure(monkeypatch):
    """If the registry itself can't be read, the scan logs and exits
    cleanly — the next interval will retry."""
    def boom():
        raise RuntimeError("disk gone")
    monkeypatch.setattr(_tasks, "list_tasks", boom)

    launcher = MagicMock()
    checker = _liveness.LivenessChecker(launcher=launcher)
    checker._scan_once()  # must not raise

    launcher.status.assert_not_called()
