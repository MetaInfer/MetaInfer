"""FastAPI router for evolve-kernel .

Serves: iterations, charts, state-graph, kernel-library, harnesses,
retrospective, and QA endpoints.
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from metainfer.server._helpers import (
    require_task_type,
    state_dir_for,
    task_or_404,
    workspace_dir_for,
)
from metainfer.server.qa_routes import register_qa_routes

from . import _state_readers

PLUGIN_TYPE = "evolve-kernel"


def build_router(plugin) -> APIRouter:
    router = APIRouter()

    # ---- Iterations ----

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

    # ---- Charts ----

    @router.get("/charts")
    def ok_charts(task_id: str) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        return _state_readers.read_charts(state_dir_for(entry))

    # ---- State Graph ----

    @router.get("/state-graph")
    def ok_state_graph(task_id: str) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        return _state_readers.read_state_graph(state_dir_for(entry))

    # ---- Kernel Library ----

    @router.get("/kernel-library")
    def ok_kernel_library(task_id: str) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        return _state_readers.read_kernel_library(workspace_dir_for(entry))

    # ---- Harnesses ----

    @router.get("/harnesses/correctness")
    def ok_correctness_harness(task_id: str) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        return _state_readers.read_harness(workspace_dir_for(entry), "correctness")

    @router.get("/harnesses/perf")
    def ok_perf_harness(task_id: str) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        return _state_readers.read_harness(workspace_dir_for(entry), "perf")

    # ---- Reference Kernel ----

    @router.get("/reference-kernel")
    def ok_reference_kernel(task_id: str) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        return _state_readers.read_reference_kernel(workspace_dir_for(entry))

    # ---- QA ----

    register_qa_routes(router, plugin, prefix="/qa")

    return router
