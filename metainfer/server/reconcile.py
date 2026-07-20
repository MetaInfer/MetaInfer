"""Startup reconciliation: bring runtime.json + registry.json in sync
with the actual process table.

Called once at WebUI startup (see :func:`metainfer.server.app.create_app`).
Does three things:

1. **Stamp** this WebUI session into ``runtime.json`` (pid, boot_id,
   started_at). See :func:`metainfer.server.runtime.record_webui_start`.

2. **Scan** ``/proc`` for every ``metainfer-orchestrator`` process on
   the host (see :func:`metainfer.server.proc.list_orchestrator_processes`).

3. **Reconcile**:

   - **Adopt** — a running orchestrator whose task_id is in the
     registry but missing from runtime.json (orphaned by a previous
     WebUI session that crashed). Adopt by recording it under the new
     boot_id.
   - **Mark dead** — a task in the registry + runtime whose orchestrator
     is no longer running. Update the task's pid file to reflect
     ``finished_at`` (if not already set) so the UI shows "not running".
   - **Detect PID reuse** — a runtime entry whose recorded PID now
     belongs to a different process (start time mismatch). Treat as
     dead.
   - **Detect ghosts** — a process in the table that doesn't correspond
     to any task in the registry. Logged but left alone (could be a
     task from a different METAINFER_HOME that happens to share this
     host).

The goal is: after this runs, ``runtime.json`` is a faithful picture of
which orchestrators are *actually* running, and the launcher's
``status()`` answers will all be correct.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import paths as _paths
from . import proc as _proc
from . import runtime as _runtime
from . import tasks as _tasks


def _log(msg: str) -> None:
    """Reconciliation runs at startup, before logging is fully
    configured. Print to stderr so it shows in the WebUI's console."""
    print(f"[metainfer-reconcile] {msg}", file=sys.stderr, flush=True)


def _read_pid_file(state_dir: Path) -> Dict[str, Any]:
    p = state_dir / "orchestrator.pid"
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (ValueError, OSError):
        return {}


# NOTE: reconcile used to have its own _write_pid_file_finished that only
# touched orchestrator.pid. That diverged from launcher._reap_dead_pid_file
# (which also updates run.json + writes a timeline event) — two reap
# paths with different effects, classic SSOT violation. Now both
# reconcile and liveness funnel through the same launcher reaper so
# cleanup is identical regardless of who detects the death.


def reconcile(silent: bool = False) -> Dict[str, Any]:
    """Run the startup reconciliation. Idempotent; safe to call multiple
    times.

    Returns a summary dict::

        {
          "boot_id": "<this session>",
          "adopted": [<task_id>, ...],          # was running, now claimed
          "marked_dead": [<task_id>, ...],      # was running, now gone
          "ghosts": [<pid>, ...],               # unknown orchestrator processes
          "scanned": <count>,                   # total proc table entries
        }
    """
    # 1. Stamp this WebUI session.
    boot_id = _runtime.record_webui_start()
    if not silent:
        _log(f"WebUI session boot_id={boot_id} pid={_proc.pid_alive and __import__('os').getpid()}")

    # 2. Scan the process table.
    live_procs = _proc.list_orchestrator_processes()
    live_by_task: Dict[str, List[Dict[str, Any]]] = {}
    for p in live_procs:
        tid = p.get("task_id")
        if tid:
            live_by_task.setdefault(tid, []).append(p)
    if not silent:
        _log(f"found {len(live_procs)} orchestrator process(es) on host")

    # 3. Reconcile per task.
    state = _runtime.read_state()
    runtime_tasks = state.get("tasks", {})
    registry_entries = {e.id: e for e in _tasks.list_tasks()}

    adopted: List[str] = []
    marked_dead: List[str] = []
    ghosts: List[int] = []

    # 3a. Walk every orchestrator process we found.
    for tid, procs in live_by_task.items():
        if tid not in registry_entries:
            # Unknown task. Either a stale process from a purged task,
            # or belongs to a different METAINFER_HOME on the same host.
            for p in procs:
                ghosts.append(p["pid"])
            continue
        entry = registry_entries[tid]
        sd = Path(entry.state_dir)
        # Pick the newest matching process if multiple exist (shouldn't
        # happen in normal operation; if it does, the older one is
        # probably a zombie).
        chosen = max(procs, key=lambda p: p.get("started_at") or 0)
        # Adopt: write into runtime.json under this boot_id and refresh
        # the pid file's started_at to match the actual kernel value.
        # Process state lives ONLY in orchestrator.pid + runtime.json;
        # registry.json holds identity only (see TaskEntry).
        _runtime.record_task_spawn(
            tid, chosen["pid"], sd, boot_id,
            started_at=chosen.get("started_at"),
        )
        adopted.append(tid)
        if not silent:
            _log(f"adopted task {tid}: pid={chosen['pid']}")

    # 3b. Walk every runtime entry that we did NOT see in the proc table.
    seen = set(live_by_task.keys())
    for tid, entry in list(runtime_tasks.items()):
        if tid in seen:
            continue
        if tid not in registry_entries:
            # Task removed from registry while we were down — drop from
            # runtime too.
            _runtime.clear_task(tid)
            continue
        # The recorded PID isn't running (or was recycled). Reap via the
        # SAME path as the user-Kill and liveness paths — single source
        # of cleanup truth. _reap_dead_pid_file stamps orchestrator.pid,
        # updates run.json::finished/final_status, and writes a timeline
        # event.
        registry_entry = registry_entries[tid]
        sd = Path(registry_entry.state_dir)
        pidfile = _read_pid_file(sd)
        recorded_pid = pidfile.get("pid")
        if recorded_pid is None:
            # Already cleared by a prior reap. Just drop the runtime entry.
            _runtime.clear_task(tid)
            continue
        recorded_started = pidfile.get("started_at")
        # Double-check the recorded pid isn't actually alive — it's
        # possible we missed it in the scan (race with fork).
        if _proc.validate_pid_started_at(recorded_pid, recorded_started):
            # Race: the process is alive but our scan missed it.
            # Don't mark dead; we'll pick it up next reconcile.
            continue
        from .launcher import get_default_launcher
        try:
            get_default_launcher()._reap_dead_pid_file(
                tid, recorded_pid, recorded_started,
            )
            marked_dead.append(tid)
            if not silent:
                _log(f"marked task {tid} dead (pid {recorded_pid} no longer alive)")
        except Exception:  # noqa: BLE001 — reaper is best-effort
            if not silent:
                _log(f"reaper FAILED for task {tid} (pid {recorded_pid}); "
                     "leaving runtime entry in place for next pass")
            continue
        _runtime.clear_task(tid)

    if ghosts and not silent:
        _log(f"{len(ghosts)} unknown orchestrator process(es) (PIDs: {ghosts}); "
             "left alone — may belong to another METAINFER_HOME")

    return {
        "boot_id": boot_id,
        "adopted": adopted,
        "marked_dead": marked_dead,
        "ghosts": ghosts,
        "scanned": len(live_procs),
    }
