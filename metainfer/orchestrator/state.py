"""Persistent state store for one MetaInfer task.

All state lives under ``<cwd>/.metainfer/state/<task_id>/``:

* ``requirements.json``    — frozen requirements captured by the entry skill
* ``run.json``             — mutable run status (current phase, iteration, ...)
* ``iterations/<n>.json``  — one record per iteration (schema is TASK-DEFINED;
  the shell no longer ships an :class:`IterationRecord` dataclass)
* ``timeline.jsonl``       — append-only event log (phase transitions, agent events)

The store is intentionally file-based so the WebUI can run in a separate
process and observe state without IPC. Iteration records are treated as
opaque JSON dicts here — each task package defines its own iteration
schema in whatever shape fits (dataclass, TypedDict, plain dict...).
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# --------------------------------------------------------------------------- #
# Shared phase / outcome types
# --------------------------------------------------------------------------- #
#
# ``Phase`` and ``Outcome`` are plain ``str`` aliases here — the shell
# stores them opaquely. Each task package's ``orchestrator/phases.py``
# defines its OWN Phase Literal for internal type safety.
#
# The shell no longer ships an ``IterationRecord`` dataclass. Each task
# that runs an iteration loop defines its own record type and
# (de)serializes through :class:`StateStore`'s dict-based API.
Phase = str
Outcome = Union[str, None]


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #


@dataclass
class RunStatus:
    task_id: str
    current_iteration: int = 0
    current_phase: Phase = "idle"
    last_update: float = 0.0
    finished: bool = False
    # Terminal status — written exactly once when the run ends. The
    # orchestrator NEVER produces "failed"; the only values are:
    #   "success"  — exited on a real success transition
    #   "stopped"  — iteration cap hit, or infra issue halted the run
    #                 without producing a success (the run is NOT marked
    #                 failed — it explored until it couldn't continue)
    #   "aborted"  — externally interrupted (Ctrl-C / SIGTERM)
    final_status: Optional[str] = None
    # last transition the orchestrator took — used by the WebUI to highlight
    # the active edge in the state graph.
    last_outcome: Optional[Outcome] = None
    last_transition_label: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    # ``task_type`` and ``created_at`` are deliberately NOT fields here.
    # Authoritative sources:
    #   - task_type  → requirements.json (immutable after task creation)
    #   - created_at → registry.json::created_at (task spawn time)
    # Persisting either in run.json created stale-copy hazards (reset
    # used to overwrite created_at, losing the original). See CLAUDE.md
    # "数据一致性" section.


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #


class StateStore:
    """File-based state for a single task. Thread-safe via a single RLock."""

    def __init__(self, task_dir: Path) -> None:
        self.task_dir = task_dir
        self.task_dir.mkdir(parents=True, exist_ok=True)
        (self.task_dir / "iterations").mkdir(exist_ok=True)
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ #
    # Paths
    # ------------------------------------------------------------------ #

    @property
    def requirements_path(self) -> Path:
        return self.task_dir / "requirements.json"

    @property
    def run_path(self) -> Path:
        return self.task_dir / "run.json"

    @property
    def timeline_path(self) -> Path:
        return self.task_dir / "timeline.jsonl"

    def iter_path(self, n: int) -> Path:
        return self.task_dir / "iterations" / f"{n:03d}.json"

    def interrupted_iter_path(self, n: int) -> Path:
        """Where to archive an iteration's record when the orchestrator was
        interrupted mid-flight. Sibling to :meth:`iter_path`; same glob
        pattern picks it up for display."""
        return self.task_dir / "iterations" / f"{n:03d}.interrupted.json"

    # ------------------------------------------------------------------ #
    # Requirements
    # ------------------------------------------------------------------ #

    def load_requirements(self) -> Dict[str, Any]:
        return json.loads(self.requirements_path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------ #
    # Run status
    # ------------------------------------------------------------------ #

    def init_run(self, task_id: str) -> RunStatus:
        rs = RunStatus(
            task_id=task_id,
            last_update=time.time(),
        )
        self._write_run(rs)
        return rs

    def init_or_resume(self, task_id: str) -> tuple[RunStatus, bool]:
        """Either initialize a fresh run.json or load the existing one.

        Returns ``(run_status, is_resume)``. ``is_resume`` is True iff a
        ``run.json`` already existed on disk (i.e. the orchestrator has
        run before for this task).
        """
        with self._lock:
            if self.run_path.exists():
                return self.load_run(), True
            return self.init_run(task_id), False

    def load_run(self) -> RunStatus:
        if not self.run_path.exists():
            raise FileNotFoundError(f"no run.json at {self.run_path}")
        data = json.loads(self.run_path.read_text(encoding="utf-8"))
        # Filter to known fields so old run.json files (which used to
        # persist task_type) load cleanly. Extra keys are silently
        # dropped — single source of truth lives in requirements.json.
        if not isinstance(data, dict):
            raise ValueError(f"run.json at {self.run_path} is not a JSON object")
        known = {f.name for f in fields(RunStatus)}
        filtered = {k: v for k, v in data.items() if k in known}
        return RunStatus(**filtered)

    def update_run(self, **kwargs: Any) -> RunStatus:
        with self._lock:
            rs = self.load_run()
            for k, v in kwargs.items():
                if hasattr(rs, k):
                    setattr(rs, k, v)
            rs.last_update = time.time()
            self._write_run(rs)
            return rs

    def _write_run(self, rs: RunStatus) -> None:
        tmp = self.run_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(rs), indent=2), encoding="utf-8")
        os.replace(tmp, self.run_path)

    # ------------------------------------------------------------------ #
    # Iteration records
    # ------------------------------------------------------------------ #

    def write_iteration(self, n: int, data: Dict[str, Any]) -> None:
        """Persist one iteration record as JSON. ``data`` is the task's
        own schema — the shell treats it as an opaque dict. Callers
        that keep an in-memory dataclass typically pass
        ``asdict(rec)`` here."""
        with self._lock:
            path = self.iter_path(n)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            os.replace(tmp, path)

    def load_iteration(self, n: int) -> Optional[Dict[str, Any]]:
        """Return iteration ``n``'s record as a plain dict, or None."""
        path = self.iter_path(n)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def load_all_iterations(self) -> List[Dict[str, Any]]:
        """All iteration records (as dicts) sorted by filename (= number)."""
        recs = []
        for p in sorted((self.task_dir / "iterations").glob("*.json")):
            recs.append(json.loads(p.read_text(encoding="utf-8")))
        return recs

    def delete_iteration(self, n: int) -> bool:
        """Remove the iteration record JSON for ``n``. Used during resume
        to discard an iteration whose folder was incomplete. Returns True
        if a file was removed."""
        with self._lock:
            p = self.iter_path(n)
            if p.exists():
                p.unlink()
                return True
            return False

    def archive_interrupted_iteration(
        self,
        n: int,
        reason: str = "interrupted: orchestrator process exited unexpectedly",
    ) -> bool:
        """Finalize iteration ``n``'s record as failed/interrupted and move
        it aside so the retry can reuse the slot.

        Loads the existing record (if any), stamps ``status="failed"`` with
        ``failure_reason=reason`` and ``ended_at=now``, writes it to
        :meth:`interrupted_iter_path`, then deletes the live record. The
        archived file is still discovered by :meth:`load_all_iterations`
        (same ``*.json`` glob), so the WebUI shows the interrupted attempt
        in the history with its fail reason — instead of silently
        disappearing or, worse, showing as "running".

        Returns True if a record existed and was archived.
        """
        with self._lock:
            src = self.iter_path(n)
            if not src.exists():
                return False
            data = json.loads(src.read_text(encoding="utf-8"))
            now = time.time()
            data.setdefault("status", "failed")
            data["status"] = "failed"
            data["failure_reason"] = reason
            data["ended_at"] = data.get("ended_at") or now
            data["duration_s"] = max(0.0, data["ended_at"] - data.get("started_at", now))
            # marker so the UI / downstream readers can tell interrupted
            # apart from a "real" failed C step. Optional; not enforced.
            data["interrupted"] = True
            dst = self.interrupted_iter_path(n)
            tmp = dst.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            tmp.replace(dst)
            src.unlink()
            return True

    # ------------------------------------------------------------------ #
    # Timeline (append-only)
    # ------------------------------------------------------------------ #

    def append_timeline(self, event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
        with self._lock:
            entry = {
                "ts": time.time(),
                "type": event_type,
                "payload": payload or {},
            }
            from metainfer.server.filelock import lock_file
            with lock_file(self.timeline_path):
                with open(self.timeline_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")

    def load_timeline(self, since: float = 0.0) -> List[Dict[str, Any]]:
        if not self.timeline_path.exists():
            return []
        out = []
        for ln in self.timeline_path.read_text(encoding="utf-8").splitlines():
            try:
                ev = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if ev.get("ts", 0) >= since:
                out.append(ev)
        return out
