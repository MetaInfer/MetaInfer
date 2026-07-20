"""FastAPI router for find-low-hanging-kernel.

Endpoints (all relative to the shell mount ``/api/find-low-hanging-kernel/{task_id}``):

  GET /iterations                 — list iteration records
  GET /iterations/{n}             — single iteration record
  GET /state-graph                — phase state graph payload
  GET /flow-graph                 — validated flow_graph.json
  GET /trace-parsed               — deterministic parser output
  GET /memory/{step}              — step memory markdown (step1_code_analysis,
                                    step2_tracing_analysis, validation_warnings)
  GET /visualization              — standalone flow_graph.html (text/html)
  GET /workspace-file/{name}      — download flow_graph.html / flow_graph.json
  /qa/*                           — generic QA routes
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse

from metainfer.server._helpers import (
    require_task_type,
    state_dir_for,
    task_or_404,
    workspace_dir_for,
)
from metainfer.server.qa_routes import register_qa_routes

from . import _state_readers

PLUGIN_TYPE = "find-low-hanging-kernel"

_MEM_ALLOWED_STEPS = {
    "step1_code_analysis",
    "step2_tracing_analysis",
    "validation_warnings",
}


def build_router(plugin) -> APIRouter:
    router = APIRouter()

    @router.get("/iterations")
    def flhk_iterations(task_id: str) -> list:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        return _state_readers.read_iterations(state_dir_for(entry))

    @router.get("/iterations/{n}")
    def flhk_iteration_detail(task_id: str, n: int) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        rec = _state_readers.read_iteration(state_dir_for(entry), n)
        if rec is None:
            raise HTTPException(404, f"no iteration {n} for task {task_id}")
        return rec

    @router.get("/state-graph")
    def flhk_state_graph(task_id: str) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        return _state_readers.read_state_graph(state_dir_for(entry))

    @router.get("/flow-graph")
    def flhk_flow_graph(task_id: str) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        return _state_readers.read_flow_graph(workspace_dir_for(entry))

    @router.get("/trace-parsed")
    def flhk_trace_parsed(task_id: str) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        return _state_readers.read_trace_summary(workspace_dir_for(entry))

    @router.get("/memory/{step}")
    def flhk_memory(task_id: str, step: str) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        if step not in _MEM_ALLOWED_STEPS:
            raise HTTPException(
                400, f"unknown step {step!r}; expected one of {sorted(_MEM_ALLOWED_STEPS)}"
            )
        return _state_readers.read_memory_markdown(workspace_dir_for(entry), step)

    @router.get("/visualization", response_class=HTMLResponse)
    def flhk_visualization(task_id: str):
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        html_path = workspace_dir_for(entry) / "flow_graph.html"
        if not html_path.is_file():
            raise HTTPException(404, "visualization not ready yet (Step 4 not complete)")
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))

    @router.get("/workspace-file/{name}")
    def flhk_workspace_file(task_id: str, name: str):
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        # Only allow downloading the two canonical artifacts (defensive
        # path-traversal guard — name is a single segment).
        allowed = {
            "flow_graph.html": "text/html",
            "flow_graph.json": "application/json",
            "trace_parsed.json": "application/json",
        }
        if name not in allowed:
            raise HTTPException(400, f"unknown workspace file {name!r}")
        p = workspace_dir_for(entry) / name
        if not p.is_file():
            raise HTTPException(404, f"{name} not written yet")
        return PlainTextResponse(
            content=p.read_text(encoding="utf-8"),
            media_type=allowed[name],
        )

    register_qa_routes(router, plugin, prefix="/qa")
    return router
