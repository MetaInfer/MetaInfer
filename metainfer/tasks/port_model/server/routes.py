"""API routes for port-model, mounted at /api/port-model/{task_id}."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse, HTMLResponse

from metainfer.server._helpers import task_or_404, state_dir_for, workspace_dir_for
from metainfer.server.qa_routes import register_qa_routes

from ._state_readers import (
    read_iterations,
    read_iteration,
    read_state_graph,
    read_memory_markdown,
    read_diff,
    read_test_results,
)


def build_router(plugin) -> APIRouter:
    router = APIRouter()

    @router.get("/iterations")
    async def iterations(task_id: str) -> list:
        entry = task_or_404(task_id)
        return read_iterations(state_dir_for(entry))

    @router.get("/iterations/{n}")
    async def iteration(task_id: str, n: int) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        rec = read_iteration(state_dir_for(entry), n)
        if rec is None:
            raise HTTPException(404, f"iteration {n} not found")
        return rec

    @router.get("/state-graph")
    async def state_graph(task_id: str) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        return read_state_graph(state_dir_for(entry))

    @router.get("/memory/{step}")
    async def memory(task_id: str, step: str) -> PlainTextResponse:
        """Serve a memory markdown file.

        Allowed steps: p1_model_analysis, p2_source_analysis, p3_target_analysis.
        """
        allowed = {"p1_model_analysis", "p2_source_analysis", "p3_target_analysis"}
        if step not in allowed:
            raise HTTPException(400, f"unknown memory step: {step}")
        entry = task_or_404(task_id)
        content = read_memory_markdown(workspace_dir_for(entry), step)
        if content is None:
            raise HTTPException(404, f"memory/{step}.md not found")
        return PlainTextResponse(content, media_type="text/markdown")

    @router.get("/diff")
    async def diff(task_id: str) -> PlainTextResponse:
        entry = task_or_404(task_id)
        content = read_diff(workspace_dir_for(entry))
        if content is None:
            raise HTTPException(404, "diff/model_port.patch not found")
        return PlainTextResponse(content, media_type="text/plain")

    @router.get("/test-results")
    async def test_results(task_id: str) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        results = read_test_results(workspace_dir_for(entry))
        if results is None:
            return {"configured": False}
        return results

    register_qa_routes(router, plugin, prefix="/qa")
    return router
