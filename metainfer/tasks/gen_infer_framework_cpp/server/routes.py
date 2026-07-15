"""FastAPI router for the gen-infer-framework-cpp task type.

Builds a single :class:`fastapi.APIRouter` that the shell mounts under
``/api/{type}/{task_id}``. Routes provided:

    /api/{type}/{task_id}/iterations[/{n}[/retrospective]]
    /api/{type}/{task_id}/charts
    /api/{type}/{task_id}/state-graph
    /api/{type}/{task_id}/qa[/start|/<sid>]

These used to be shell endpoints. They've been pulled into the task
package because the iteration record schema (perf / goal /
start_phase / retrospective_path / failure_reason) is fundamentally
calc-shaped — gf ships the same schema since it was the second task
type and inherited calc's shape; future tasks that don't fit will
ship a different ``_state_readers``.
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from metainfer.server._helpers import (
    require_task_type,
    state_dir_for,
    task_or_404,
)
from metainfer.server.qa_routes import register_qa_routes

from . import _state_readers

PLUGIN_TYPE = "gen-infer-framework-cpp"


def build_router(plugin) -> APIRouter:
    """Build the gf router. ``plugin`` is the WebPlugin itself, passed
    in by :func:`metainfer.server.app.create_app` so we can hand it to the
    generic QA helper without a circular import."""
    router = APIRouter()

    @router.get("/iterations")
    def gf_iterations(task_id: str) -> list:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        return _state_readers.read_iterations(state_dir_for(entry))

    @router.get("/iterations/{n}")
    def gf_iteration_detail(task_id: str, n: int) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        rec = _state_readers.read_iteration(state_dir_for(entry), n)
        if rec is None:
            raise HTTPException(404, f"no iteration {n} for task {task_id}")
        return rec

    @router.get("/iterations/{n}/retrospective")
    def gf_retrospective(task_id: str, n: int) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        return _state_readers.read_retrospective(state_dir_for(entry), n)

    @router.get("/charts")
    def gf_charts(task_id: str) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        return _state_readers.read_charts(state_dir_for(entry))

    @router.get("/state-graph")
    def gf_state_graph(task_id: str) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        return _state_readers.read_state_graph(state_dir_for(entry))

    register_qa_routes(router, plugin, prefix="/qa")

    return router
