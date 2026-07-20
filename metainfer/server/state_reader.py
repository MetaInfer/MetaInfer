"""Read-only access to a task's on-disk observable state.

The WebUI never imports the orchestrator package (no in-memory coupling).
Instead it reads the files the orchestrator writes under each task's
``state_dir``:

    <state_dir>/
    ├── requirements.json     # frozen inputs
    ├── run.json              # RunStatus
    ├── timeline.jsonl        # append-only events
    ├── iterations/<n>.json   # per-iteration records
    └── agents.json           # SubAgentManager snapshot

All reads are defensive: missing files return None / empty defaults
rather than raising. This is what lets the WebUI render a half-spawned
task whose orchestrator hasn't written anything yet.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return default


def read_requirements(state_dir: Path) -> Optional[Dict[str, Any]]:
    return _load_json(state_dir / "requirements.json", None)


def read_run(state_dir: Path) -> Dict[str, Any]:
    """Return RunStatus dict, or a default 'idle' sentinel if missing.

    ``task_type`` is intentionally absent — its authoritative source is
    ``requirements.json::task_type`` (read via :func:`read_requirements`),
    not run.json. The frontend gets type from the registry entry.
    """
    default = {
        "task_id": None,
        "current_iteration": 0, "current_phase": "idle",
        "last_update": 0, "finished": False, "final_status": None,
        "last_outcome": None, "last_transition_label": None, "notes": [],
    }
    data = _load_json(state_dir / "run.json", None)
    if data is None:
        return default
    # Drop legacy fields if an old run.json still has them — single
    # source of truth is requirements.json (task_type) and registry.json
    # (created_at). Merge with defaults so missing keys don't crash the
    # frontend.
    for _legacy in ("task_type", "created_at"):
        data.pop(_legacy, None)
    return {**default, **data}


def read_timeline(state_dir: Path, since: float = 0.0) -> List[Dict[str, Any]]:
    path = state_dir / "timeline.jsonl"
    if not path.exists():
        return []
    out = []
    for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            ev = json.loads(ln)
        except ValueError:
            continue
        if ev.get("ts", 0) >= since:
            out.append(ev)
    return out


def read_agents(state_dir: Path) -> Dict[str, Any]:
    """SubAgentManager snapshot. Returns ``{ts: 0, agents: []}`` when
    no orchestrator has written anything yet (e.g. orchestrator hasn't
    started, or hasn't spawned any sub-agents yet)."""
    default = {"ts": 0, "agents": []}
    return _load_json(state_dir / "agents.json", default)


# --------------------------------------------------------------------------- #
# Write helpers (very limited)
# --------------------------------------------------------------------------- #
# The WebUI is read-only by design — but the restart flow needs to stamp
# explicit audit events into timeline.jsonl so it's visible WHY each
# orchestrator was killed + respawned. We don't touch run.json, agents.json,
# iterations/, code/, logs/ — those belong to the orchestrator. timeline.jsonl
# is append-only JSONL, safe for the WebUI to append to without coordination.

def append_timeline_event(
    state_dir: Path, event_type: str, payload: Optional[Dict[str, Any]] = None,
) -> None:
    """Append one event to ``<state_dir>/timeline.jsonl``.

    Format matches what the orchestrator's StateStore writes (see
    ``state.py:append_timeline``): ``{"ts": float, "type": str, "payload": dict}``.
    Used by the WebUI to record lifecycle events it initiated (e.g.
    ``restart_initiated``), so the timeline gives a full audit trail
    spanning both orchestrator and WebUI actions.
    """
    state_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": time.time(),
        "type": event_type,
        "payload": payload or {},
    }
    from metainfer.server.filelock import lock_file
    timeline_path = state_dir / "timeline.jsonl"
    with lock_file(timeline_path):
        with open(timeline_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


def reset_state_dir(
    state_dir: Path, workspace_dir: Path, task_id: str,
) -> Dict[str, Any]:
    """Wipe everything in ``state_dir`` except ``requirements.json``,
    and wipe the entire ``workspace_dir``.

    For state_dir: removes run.json, timeline.jsonl, orchestrator.log,
    orchestrator.pid, agents.json, and all subdirectories (iterations/,
    logs/, etc.). For workspace_dir: removes the whole tree (iteration
    code, step outputs) and recreates an empty dir. Then writes a fresh
    ``run.json`` matching the RunStatus defaults so the WebUI shows a
    clean idle state immediately, and stamps a single ``task_reset``
    timeline event so the reset itself is auditable.

    ``task_type`` is intentionally NOT a parameter — it lives in
    requirements.json (which is preserved across reset) and the registry
    entry (which the caller already has).

    Caller MUST ensure the orchestrator is not running — this function
    does not check.
    """
    import shutil
    state_dir = Path(state_dir)
    workspace_dir = Path(workspace_dir)
    keep = {"requirements.json"}
    removed: List[str] = []
    if state_dir.exists():
        for p in state_dir.iterdir():
            if p.name in keep:
                continue
            is_dir = p.is_dir() and not p.is_symlink()
            try:
                if is_dir:
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    p.unlink()
                removed.append(p.name + ("/" if is_dir else ""))
            except OSError:
                pass
    if workspace_dir.exists():
        shutil.rmtree(workspace_dir, ignore_errors=True)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    now = time.time()
    fresh_run = {
        "task_id": task_id,
        "current_iteration": 0,
        "current_phase": "idle",
        "last_update": now,
        "finished": False,
        "final_status": None,
        "last_outcome": None,
        "last_transition_label": None,
        "notes": [],
    }
    state_dir.mkdir(parents=True, exist_ok=True)
    run_path = state_dir / "run.json"
    tmp = run_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(fresh_run, indent=2), encoding="utf-8")
    tmp.replace(run_path)
    append_timeline_event(state_dir, "task_reset", {
        "task_id": task_id, "reset_at": now, "removed_count": len(removed),
        "workspace_reset": True,
    })
    return {"removed": removed, "workspace_reset": True, "run": fresh_run}
