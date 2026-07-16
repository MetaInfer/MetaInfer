"""Task-specific HTTP routes for knowledge-evolution.

Mounted by the shell at ``/api/knowledge-evolution/{task_id}``.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from metainfer.server._helpers import state_dir_for, task_or_404
from metainfer.server.qa_routes import register_qa_routes

from . import _state_readers


def build_router(plugin) -> APIRouter:
    """Build and return the task-specific FastAPI router.

    Called by ``create_app()``; the shell mounts the result at
    ``/api/{plugin.type}/{task_id}``.
    """
    router = APIRouter()

    @router.get("/iterations")
    async def get_iterations(task_id: str):
        """Return all iteration records for this task."""
        entry = task_or_404(task_id)
        sd = state_dir_for(entry)
        return _state_readers.read_iterations(sd)

    @router.get("/state-graph")
    async def get_state_graph(task_id: str):
        """Return the state-machine graph for the current run."""
        entry = task_or_404(task_id)
        sd = state_dir_for(entry)
        return _state_readers.read_state_graph(sd)

    @router.get("/knowledge-gained")
    async def get_knowledge_gained(task_id: str):
        """Return knowledge gained across all completed iterations."""
        entry = task_or_404(task_id)
        sd = state_dir_for(entry)
        return _state_readers.read_knowledge_gained(sd)

    @router.get("/iterations/{iteration}/oracle-report")
    async def get_oracle_report(task_id: str, iteration: int):
        """Return the oracle report markdown for a given iteration."""
        entry = task_or_404(task_id)
        sd = state_dir_for(entry)
        return _state_readers.read_oracle_report(sd, iteration)

    @router.get("/iterations/{iteration}/retrospective")
    async def get_retrospective(task_id: str, iteration: int):
        """Return the retrospective markdown for a given iteration."""
        entry = task_or_404(task_id)
        sd = state_dir_for(entry)
        return _state_readers.read_retrospective(sd, iteration)

    @router.get("/knowledge-diff")
    async def get_knowledge_diff(
        task_id: str,
        iteration: int = Query(...),
        file: str = Query(...),
    ):
        """Return the content of a notebook file from a given iteration."""
        entry = task_or_404(task_id)
        sd = state_dir_for(entry)
        return _state_readers.read_knowledge_diff(sd, iteration, file)

    @router.get("/log")
    async def get_log(task_id: str):
        """Return the orchestrator log text."""
        entry = task_or_404(task_id)
        sd = state_dir_for(entry)
        return _state_readers.read_log(sd)

    @router.get("/charts")
    async def get_charts(task_id: str):
        """Return chart series data (durations + perf metrics)."""
        entry = task_or_404(task_id)
        sd = state_dir_for(entry)
        return _state_readers.read_charts(sd)

    @router.get("/agent-status")
    async def get_agent_status(task_id: str):
        """Return the current agent activity string (or null)."""
        entry = task_or_404(task_id)
        sd = state_dir_for(entry)
        return {"status": _state_readers.read_agent_status(sd)}

    # Wire QA routes
    register_qa_routes(router, plugin, prefix="/qa")

    return router
