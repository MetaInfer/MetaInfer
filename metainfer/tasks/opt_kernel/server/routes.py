"""FastAPI router for opt-kernel.

Borrowed from gen-infer-framework — same iteration/charts/state-graph/QA endpoints.
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

PLUGIN_TYPE = "opt-kernel"


def build_router(plugin) -> APIRouter:
    router = APIRouter()

    @router.get("/iterations")
    def ok_iterations(task_id: str) -> list:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        return _state_readers.read_iterations(state_dir_for(entry))

    @router.get("/iterations/{n}")
    def ok_iteration_detail(task_id: str, n: int) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        rec = _state_readers.read_iteration(state_dir_for(entry), n)
        if rec is None:
            raise HTTPException(404, f"no iteration {n} for task {task_id}")
        return rec

    @router.get("/iterations/{n}/retrospective")
    def ok_retrospective(task_id: str, n: int) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        return _state_readers.read_retrospective(state_dir_for(entry), n)

    @router.get("/charts")
    def ok_charts(task_id: str) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        return _state_readers.read_charts(state_dir_for(entry))

    @router.get("/state-graph")
    def ok_state_graph(task_id: str) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        return _state_readers.read_state_graph(state_dir_for(entry))

    register_qa_routes(router, plugin, prefix="/qa")

    return router
