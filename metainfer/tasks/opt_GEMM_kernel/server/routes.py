"""Task-local API routes for GEMM optimization evidence."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from metainfer.server._helpers import require_task_type, state_dir_for, task_or_404
from metainfer.server.qa_routes import register_qa_routes

from . import _state_readers
from ..orchestrator.guidance import GuidanceError, GuidanceStore


PLUGIN_TYPE = "opt_GEMM_kernel"


def build_router(plugin) -> APIRouter:
    router = APIRouter()

    def state(task_id: str):
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        return state_dir_for(entry)

    @router.get("/iterations")
    def iterations(task_id: str) -> list:
        return _state_readers.read_iterations(state(task_id))

    @router.get("/iterations/{n}")
    def iteration(task_id: str, n: int) -> Dict[str, Any]:
        result = _state_readers.read_iteration(state(task_id), n)
        if result is None:
            raise HTTPException(404, f"no iteration {n}")
        return result

    @router.get("/iterations/{n}/retrospective")
    def retrospective(task_id: str, n: int) -> Dict[str, Any]:
        return _state_readers.read_retrospective(state(task_id), n)

    @router.get("/charts")
    def charts(task_id: str) -> Dict[str, Any]:
        return _state_readers.read_charts(state(task_id))

    @router.get("/champion")
    def champion(task_id: str) -> Dict[str, Any]:
        return _state_readers.read_champion(state(task_id))

    @router.get("/baseline")
    def baseline(task_id: str) -> Dict[str, Any]:
        return _state_readers.read_baseline(state(task_id))

    @router.get("/state-graph")
    def state_graph(task_id: str) -> Dict[str, Any]:
        return _state_readers.read_state_graph(state(task_id))

    @router.get("/guidance")
    def guidance(task_id: str) -> Dict[str, Any]:
        return GuidanceStore(state(task_id) / "guidance").snapshot()

    @router.post("/guidance")
    def submit_guidance(task_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            item = GuidanceStore(state(task_id) / "guidance").submit(
                str(payload.get("text") or "")
            )
        except GuidanceError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {
            "accepted": True,
            "delivery": "next_planner_or_implementer_boundary",
            "item": item,
        }

    register_qa_routes(router, plugin, prefix="/qa")
    return router
