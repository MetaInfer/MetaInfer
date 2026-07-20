"""Task registry: durable list of all tasks the WebUI knows about.

Stored at ``<node_dir>/.metainfer/registry.json`` (see :mod:`metainfer.server.paths`
for the node-rooted layout). Each entry pins a task to its ``state_dir``
(metadata + logs), its ``workspace_dir`` (generated artifacts), the task
type, the launcher used to spawn it, and the last-known PID + status. The
registry is the source of truth for the task list view; everything else
(current phase, iterations, agents, perf, etc.) is read on demand from
the task's ``state_dir``.

Atomic updates via :func:`fcntl.flock` on a sibling lock file so multiple
WebUI processes (or the orchestrator subprocess writing its PID) can
safely read-modify-write.
"""

from __future__ import annotations

import fcntl
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import paths as _paths


@dataclass
class TaskEntry:
    """One row in the registry — IDENTITY ONLY.

    The registry is the durable list of tasks the WebUI knows about.
    It pins each task to its ``state_dir`` (metadata + logs) and
    ``workspace_dir`` (artifacts), plus the task type and display
    label. That's it.

    **Process state (pid / started_at / finished_at) is NOT stored
    here.** It lives only in the per-task ``orchestrator.pid`` file,
    read on demand via :meth:`Launcher.status`. Caching it here used
    to cause SSOT violations: every cleanup path (reconcile,
    _reap_dead_pid_file, kill) had to remember to mirror its write
    into the registry cache, and the ``if v is None: continue`` rule
    in :func:`update_task` silently swallowed ``pid=None`` clearings,
    so the cache stuck at stale values forever. If you need to know
    whether a task is running, call ``launcher.status(task_id)`` —
    do NOT add a pid field back to this dataclass.
    """
    id: str                                 # user-visible task id, unique
    type: str                               # task type id (matches WebPlugin.type)
    label: str                              # short display name
    state_dir: str                          # absolute path to task metadata dir
    created_at: float
    # Absolute path to task's generated-artifacts dir (parallel to
    # state_dir but under <node>/workspaces/). Empty for legacy entries
    # created before the split — callers should fall back to deriving
    # from id via paths.workspace_dir(id) if they need a Path.
    workspace_dir: str = ""
    # Launcher that owns this task: "local" or "remote:<node_id>" (future).
    # "remote:<node_id>" semantics will eventually resolve to a path under
    # <root>/nodes/<node_id>/ on the shared filesystem.
    launcher: str = "local"


# Legacy fields that used to live on TaskEntry but were removed when
# process state was consolidated into orchestrator.pid. Silently
# stripped on read so old registry.json files keep working.
_LEGACY_FIELDS = frozenset({"pid", "started_at", "finished_at"})


def _read_registry_locked() -> Dict[str, Any]:
    """Read the raw registry dict. Call this while holding the flock."""
    p = _paths.registry_path()
    if not p.exists():
        return {"tasks": []}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {"tasks": []}
    if "tasks" not in data or not isinstance(data["tasks"], list):
        data["tasks"] = []
    return data


def _write_registry_locked(data: Dict[str, Any]) -> None:
    p = _paths.registry_path()
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(p)


def _lock():
    """Context manager that acquires an exclusive flock on the registry
    lock file. Safe across processes. Usage:

        with _lock():
            ...
    """
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        lock_path = _paths.registry_lock_path()
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        f = open(lock_path, "a+", encoding="utf-8")
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        finally:
            f.close()

    return _ctx()


def list_tasks() -> List[TaskEntry]:
    """Return all tasks in creation order."""
    with _lock():
        data = _read_registry_locked()
    return [TaskEntry(**_strip_legacy(t)) for t in data["tasks"]]


def get_task(task_id: str) -> Optional[TaskEntry]:
    """Return one task by id, or None."""
    with _lock():
        data = _read_registry_locked()
    for t in data["tasks"]:
        if t.get("id") == task_id:
            return TaskEntry(**_strip_legacy(t))
    return None


def _strip_legacy(d: Dict[str, Any]) -> Dict[str, Any]:
    """Remove legacy fields (pid/started_at/finished_at) from a registry
    entry dict. Old registry.json files may still contain them; new code
    must not consume them. See :class:`TaskEntry` for why."""
    return {k: v for k, v in d.items() if k not in _LEGACY_FIELDS}


def add_task(entry: TaskEntry) -> None:
    """Insert a new task. Raises ValueError if id collides."""
    with _lock():
        data = _read_registry_locked()
        for t in data["tasks"]:
            if t.get("id") == entry.id:
                raise ValueError(f"task id {entry.id!r} already exists")
        data["tasks"].append(asdict(entry))
        _write_registry_locked(data)


def update_task(task_id: str, **patch: Any) -> Optional[TaskEntry]:
    """Patch identity fields on an existing task (e.g. relabel). Returns
    the updated entry, or None if no such task.

    Process state (pid/started_at/finished_at) is NOT accepted here —
    those live only in ``orchestrator.pid``. Passing them will be
    silently dropped (see :class:`TaskEntry` for why).
    """
    patch = {k: v for k, v in patch.items() if k not in _LEGACY_FIELDS}
    if not patch:
        return get_task(task_id)
    with _lock():
        data = _read_registry_locked()
        for i, t in enumerate(data["tasks"]):
            if t.get("id") == task_id:
                t.update(patch)
                data["tasks"][i] = t
                _write_registry_locked(data)
                return TaskEntry(**_strip_legacy(t))
    return None


def remove_task(task_id: str) -> bool:
    """Remove a task from the registry. Returns True if it was present.
    Does NOT delete the state_dir on disk — that's a separate operation
    so it can be confirmed by the user."""
    with _lock():
        data = _read_registry_locked()
        before = len(data["tasks"])
        data["tasks"] = [t for t in data["tasks"] if t.get("id") != task_id]
        if len(data["tasks"]) == before:
            return False
        _write_registry_locked(data)
    return True


def gen_task_id(task_type: str, label: Optional[str] = None) -> str:
    """Generate a unique, readable task id of the form
    ``<slug>-<short-uuid>``. ``task_type`` is the prefix; ``label`` (if
    given) is slugified and prepended for readability."""
    import re
    import uuid
    short = uuid.uuid4().hex[:8]
    if label:
        slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")[:32]
        if slug:
            return f"{slug}-{short}"
    # Fall back to type-based prefix
    type_slug = re.sub(r"[^a-z0-9]+", "-", task_type.lower()).strip("-")[:24]
    return f"{type_slug}-{short}"
