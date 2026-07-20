"""Shell router — type-agnostic endpoints for task CRUD, lifecycle, monitoring.

All routes are mounted at ``/api/sys-shell`` by ``create_app``. Each is
a relative-path endpoint (no ``/api/sys-shell`` prefix); the shell's
mount prefix carries it.

Moved here from ``metainfer/server/app.py`` so the shell is a true task
package alongside calc_value / gen_infer_framework.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Request
from starlette.responses import StreamingResponse

from metainfer.server import launcher as _launcher
from metainfer.server import forms as _forms
from metainfer.server import paths as _paths
from metainfer.server import sse as _sse
from metainfer.server import state_reader as _sr
from metainfer.server import tasks as _tasks
from metainfer.server._helpers import (
    state_dir_for as _state_dir_for,
    task_or_404 as _task_or_404,
    workspace_dir_for as _workspace_dir_for,
)
from metainfer.server.registry import get as _get_web_plugin


def _plugin_view_hint(task_type: str) -> Dict[str, Any]:
    """Return the frontend detail-view hint fields for a task type.

    Plugins set ``detail_view_module`` (an importmap key) on their
    WebPlugin; the frontend uses it to dynamically dispatch the task
    detail view. Returns an empty dict when no plugin is registered for
    this task type (the frontend then renders its default view).
    """
    plugin = _get_web_plugin(task_type)
    if plugin is None or not plugin.detail_view_module:
        return {}
    return {
        "detail_view_module": plugin.detail_view_module,
        "detail_view_export": plugin.detail_view_export,
    }


def build_router(plugin):
    """Build and return the shell APIRouter with all type-agnostic routes.

    Called by ``create_app`` once at startup; the returned router is
    mounted at ``/api/sys-shell``.
    """
    router = APIRouter()

    # -- task types / forms ---------------------------------------------------
    @router.get("/task-types")
    def task_types() -> List[Dict[str, str]]:
        return _forms.list_task_types()

    @router.get("/task-types/{task_type}/schema")
    def task_type_schema(task_type: str) -> Dict[str, Any]:
        schema = _forms.load_form_schema(task_type)
        if schema is None:
            raise HTTPException(404, f"unknown task type: {task_type}")
        return schema

    # -- task listing / creation (no {task_id} in path) ------------------------
    @router.get("/tasks")
    def list_tasks() -> Dict[str, Any]:
        launcher = _launcher.get_default_launcher()
        out = []
        for e in _tasks.list_tasks():
            status = launcher.status(e.id).to_dict()
            out.append({
                "id": e.id, "type": e.type, "label": e.label,
                "state_dir": e.state_dir, "created_at": e.created_at,
                "launcher": e.launcher,
                "status": status,
                **_plugin_view_hint(e.type),
            })
        return {"tasks": out}

    @router.get("/{task_id}")
    def get_task(task_id: str) -> Dict[str, Any]:
        entry = _task_or_404(task_id)
        sd = _state_dir_for(entry)
        launcher = _launcher.get_default_launcher()
        return {
            "id": entry.id, "type": entry.type, "label": entry.label,
            "state_dir": entry.state_dir, "created_at": entry.created_at,
            "launcher": entry.launcher,
            "status": launcher.status(task_id).to_dict(),
            "requirements": _sr.read_requirements(sd),
            "run": _sr.read_run(sd),
            **_plugin_view_hint(entry.type),
        }

    @router.delete("/{task_id}")
    def delete_task(task_id: str, purge: bool = False) -> Dict[str, Any]:
        entry = _task_or_404(task_id)
        launcher = _launcher.get_default_launcher()
        if launcher.status(task_id).running:
            launcher.kill(task_id, force=True)
        removed_files = False
        if purge:
            import shutil
            sd = Path(entry.state_dir)
            if sd.exists():
                shutil.rmtree(sd, ignore_errors=True)
                removed_files = True
            wd = _workspace_dir_for(entry)
            if wd.exists():
                shutil.rmtree(wd, ignore_errors=True)
                removed_files = True
        _tasks.remove_task(task_id)
        return {"removed_from_registry": True, "purged_files": removed_files}

    @router.get("/{task_id}/run")
    def task_run(task_id: str) -> Dict[str, Any]:
        """RunStatus only — convenience for clients that don't need the
        full task envelope. Same file as ``get_task()['run']``."""
        entry = _task_or_404(task_id)
        return _sr.read_run(_state_dir_for(entry))

    @router.post("/tasks")
    async def create_task(request: Request) -> Dict[str, Any]:
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001 — malformed JSON
            raise HTTPException(400, "request body must be valid JSON")
        if not isinstance(body, dict):
            raise HTTPException(400, "request body must be a JSON object")
        task_type = body.get("type")
        label = body.get("label") or ""
        answers = body.get("answers") or {}
        extra_args = body.get("extra_args") or []
        if not task_type:
            raise HTTPException(400, "missing 'type'")
        # Validate against schema before doing anything destructive.
        v = _forms.validate_submission(task_type, answers)
        if not v["ok"]:
            raise HTTPException(400, detail={"errors": v["errors"]})
        # Generate a unique task id.
        task_id = _tasks.gen_task_id(task_type, label)
        sd = _paths.task_dir(task_id)
        sd.mkdir(parents=True, exist_ok=True)
        # Workspace dir holds task-package-generated artifacts (each task
        # type owns its own workspace subtree); state_dir holds metadata + logs.
        wd = _paths.workspace_dir(task_id)
        wd.mkdir(parents=True, exist_ok=True)
        # Build the requirements.json the orchestrator will read.
        _plugin = _get_web_plugin(task_type)
        _default_label = (_plugin.label if _plugin else "") or task_type
        requirements = {
            "task_id": task_id,
            "task_type": task_type,
            "raw_request": body.get("raw_request") or "",
            "label": label or _default_label,
            **answers,
        }
        # Register first (so list view picks it up even if spawn fails).
        entry = _tasks.TaskEntry(
            id=task_id, type=task_type, label=label or _default_label,
            state_dir=str(sd), workspace_dir=str(wd),
            created_at=time.time(), launcher="local",
        )
        _tasks.add_task(entry)
        # Spawn the orchestrator.
        launcher = _launcher.get_default_launcher()
        try:
            pid = launcher.start(task_id, requirements, sd, wd, extra_args=extra_args)
        except Exception as e:  # noqa: BLE001
            # Spawn failed — stamp orchestrator.pid with finished_at so
            # launcher.status() reports not-running. Process state lives
            # only in orchestrator.pid (SSOT); registry holds identity.
            pf = sd / "orchestrator.pid"
            try:
                import json as _json
                data = _json.dumps({
                    "pid": None,
                    "task_id": task_id,
                    "finished_at": time.time(),
                    "exit_hint": "spawn-failed",
                }, indent=2)
                tmp = pf.with_suffix(".tmp")
                tmp.write_text(data, encoding="utf-8")
                tmp.replace(pf)
            except OSError:
                pass
            raise HTTPException(500, detail={"error": f"spawn failed: {e!r}"})
        return {"task_id": task_id, "pid": pid,
                "state_dir": str(sd), "workspace_dir": str(wd)}

    @router.post("/{task_id}/control")
    async def control_task(task_id: str, request: Request) -> Dict[str, Any]:
        entry = _task_or_404(task_id)
        body = await request.json()
        action = body.get("action")
        launcher = _launcher.get_default_launcher()
        if action == "kill":
            force = bool(body.get("force", False))
            ok = launcher.kill(task_id, force=force)
            return {"ok": ok, "action": "kill", "force": force}
        if action == "restart":
            sd = _state_dir_for(entry)
            req = _sr.read_requirements(sd)
            if req is None:
                raise HTTPException(400, "no requirements.json to restart from")
            prior_status = launcher.status(task_id).to_dict()
            prior_run = _sr.read_run(sd) or {}
            _sr.append_timeline_event(sd, "restart_initiated", {
                "task_id": task_id,
                "prior_pid": prior_status.get("pid"),
                "prior_running": prior_status.get("running"),
                "prior_phase": prior_run.get("current"),
                "prior_iteration": prior_run.get("iteration_count")
                                or prior_run.get("iteration"),
                "prior_outcome": prior_run.get("outcome"),
                "resume_mode": "preserve_state",
            })
            if prior_status.get("running"):
                launcher.kill(task_id, force=True)
                await asyncio.sleep(0.5)
            wd = _workspace_dir_for(entry)
            pid = launcher.start(task_id, req, sd, wd)
            return {
                "ok": True, "action": "restart", "pid": pid,
                "prior_status": prior_status,
            }
        if action == "reset":
            if launcher.status(task_id).running:
                raise HTTPException(
                    409, "task is still running; kill it before resetting",
                )
            sd = _state_dir_for(entry)
            prior_run = _sr.read_run(sd) or {}
            tid = prior_run.get("task_id") or task_id
            wd = _workspace_dir_for(entry)
            summary = _sr.reset_state_dir(sd, wd, tid)
            return {"ok": True, "action": "reset", **summary}
        raise HTTPException(400, f"unknown action: {action}")

    # -- per-task shell data ---------------------------------------------------

    @router.get("/{task_id}/timeline")
    def task_timeline(task_id: str, since: float = 0.0) -> Dict[str, Any]:
        entry = _task_or_404(task_id)
        return {"events": _sr.read_timeline(_state_dir_for(entry), since=since)}

    @router.get("/{task_id}/agents")
    def task_agents(task_id: str) -> Dict[str, Any]:
        entry = _task_or_404(task_id)
        return _sr.read_agents(_state_dir_for(entry))

    @router.get("/{task_id}/token-budget")
    def task_token_budget(task_id: str) -> Dict[str, Any]:
        entry = _task_or_404(task_id)
        budget_path = _state_dir_for(entry) / "token_budget.json"
        if not budget_path.exists():
            return {"configured": False}
        try:
            data = json.loads(budget_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return {"configured": False, "error": "budget file unreadable"}
        cfg = data.get("config") or {}
        totals = data.get("totals") or {}
        used = float(totals.get("total_cost_usd", 0.0))
        limit = cfg.get("max_cost_usd")
        hard = cfg.get("max_cost_usd_hard")
        exhausted = bool(totals.get("exhausted"))
        hard_exhausted = bool(totals.get("hard_exhausted"))
        pct = None
        if isinstance(limit, (int, float)) and limit > 0:
            pct = round(min(100.0, (used / float(limit)) * 100.0), 2)
        return {
            "configured": True,
            "used_cost_usd": used,
            "limit_cost_usd": limit,
            "hard_limit_cost_usd": hard,
            "exhausted": exhausted,
            "hard_exhausted": hard_exhausted,
            "used_pct": pct,
            "agent_count": int(totals.get("agent_count", 0)),
            "total_input_tokens": int(totals.get("total_input_tokens", 0)),
            "total_output_tokens": int(totals.get("total_output_tokens", 0)),
            "total_cache_read_input_tokens": int(
                totals.get("total_cache_read_input_tokens", 0)),
            "per_source": data.get("per_source") or {},
            "per_phase": data.get("per_phase") or {},
        }

    @router.post("/{task_id}/token-budget")
    def task_token_budget_update(task_id: str, body: dict) -> Dict[str, Any]:
        entry = _task_or_404(task_id)
        sd = _state_dir_for(entry)
        budget_path = sd / "token_budget.json"
        if budget_path.exists():
            try:
                data = json.loads(budget_path.read_text(encoding="utf-8"))
            except (ValueError, OSError) as exc:
                raise HTTPException(500, f"cannot read budget file: {exc}")
        else:
            data = {
                "schema_version": 1,
                "config": {},
                "totals": {
                    "total_cost_usd": 0.0,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "total_cache_read_input_tokens": 0,
                    "agent_count": 0,
                    "exhausted": False,
                    "hard_exhausted": False,
                },
                "per_source": {},
                "per_phase": {},
                "records": [],
            }
        if not isinstance(data, dict):
            data = {}
        cfg = data.setdefault("config", {})
        if not isinstance(cfg, dict):
            cfg = {}
            data["config"] = cfg
        body = body or {}
        if "max_cost_usd" in body:
            v = body["max_cost_usd"]
            cfg["max_cost_usd"] = (float(v) if v is not None else None)
        if "max_cost_usd_hard" in body:
            v = body["max_cost_usd_hard"]
            cfg["max_cost_usd_hard"] = (float(v) if v is not None else None)
        totals = data.setdefault("totals", {})
        if not isinstance(totals, dict):
            totals = {}
            data["totals"] = totals
        totals.setdefault("total_cost_usd", 0.0)
        used = float(totals.get("total_cost_usd", 0.0))
        soft = cfg.get("max_cost_usd")
        hard = cfg.get("max_cost_usd_hard")
        totals["exhausted"] = bool(
            isinstance(soft, (int, float)) and used >= soft)
        totals["hard_exhausted"] = bool(
            isinstance(hard, (int, float)) and used >= hard)
        tmp = budget_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        from metainfer.server.filelock import lock_file
        with lock_file(budget_path):
            tmp.replace(budget_path)
        os.utime(budget_path)
        return task_token_budget(task_id)

    @router.get("/{task_id}/log")
    def task_log(task_id: str, tail_bytes: int = 65536) -> Dict[str, Any]:
        """Tail of the orchestrator's stdout+stderr log."""
        entry = _task_or_404(task_id)
        p = _state_dir_for(entry) / "orchestrator.log"
        if not p.exists():
            return {"content": "", "truncated": False}
        try:
            data = p.read_bytes()
        except OSError:
            return {"content": "", "truncated": False}
        truncated = len(data) > tail_bytes
        if truncated:
            data = data[-tail_bytes:]
        return {
            "content": data.decode("utf-8", errors="replace"),
            "truncated": truncated,
        }

    # -- SSE stream ------------------------------------------------------------
    @router.get("/events")
    async def events(request: Request) -> StreamingResponse:
        await _sse.watcher.start()
        q = await _sse.watcher.subscribe()

        async def gen():
            try:
                # Initial hello so the client knows the stream is alive.
                yield "event: hello\ndata: {}\n\n"
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        ev = await asyncio.wait_for(q.get(), timeout=15.0)
                        yield f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n"
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
            finally:
                await _sse.watcher.unsubscribe(q)

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return router
