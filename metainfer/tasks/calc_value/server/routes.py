"""FastAPI router for the calc-theoretical-value task type.

This module builds a single :class:`fastapi.APIRouter` carrying every
HTTP route calc_value exposes. The shell mounts it under
``/api/{type}/{task_id}`` so every route below lands at:

    /api/{type}/{task_id}/calc/graph
    /api/{type}/{task_id}/calc/compute
    /api/{type}/{task_id}/calc/viz
    /api/{type}/{task_id}/calc/summary
    /api/{type}/{task_id}/calc/iterations
    /api/{type}/{task_id}/calc/rough
    /api/{type}/{task_id}/calc/cells
    /api/{type}/{task_id}/calc/cell/{compound}/{angle}/{round_idx}
    /api/{type}/{task_id}/calc/qa[/start|/<sid>]
    /api/{type}/{task_id}/iterations[/{n}[/retrospective]]
    /api/{type}/{task_id}/charts
    /api/{type}/{task_id}/state-graph

The first block (``/calc/...``) reads calc-specific artifacts from the
task's ``workspace_dir`` via :mod:`._readers`. The second block
(``/iterations``, ``/charts``, ``/state-graph``, ``/retrospective``)
reads the orchestrator's iteration records / phases from ``state_dir``
— these used to live in the shell but are task-shaped, so they now
live with the task package.

Type guard is enforced at the route layer via
:func:`metainfer.server._helpers.require_task_type` for safety.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

from metainfer.server._helpers import (
    require_task_type,
    state_dir_for,
    task_or_404,
    workspace_dir_for,
)
from metainfer.server.qa_routes import register_qa_routes

from . import _readers, _state_readers

PLUGIN_TYPE = "calc-theoretical-value"


def build_router(plugin) -> APIRouter:
    """Build the calc_value router. ``plugin`` is the WebPlugin itself,
    passed in by :func:`metainfer.server.app.create_app` so we can hand it
    to the generic QA helper without a circular import."""
    router = APIRouter()

    # ----------------------------------------------------------------- #
    # calc-specific workspace reads
    # ----------------------------------------------------------------- #
    @router.get("/calc/graph")
    def calc_graph(task_id: str) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        wd = workspace_dir_for(entry)
        try:
            return _readers.read_graph(wd)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc))

    @router.get("/calc/compute")
    def calc_compute(task_id: str, batch_size: int = 1, seq_len: int = 1) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        wd = workspace_dir_for(entry)
        try:
            return _readers.compute(wd, batch_size, seq_len)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc))
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @router.get("/calc/viz")
    def calc_viz(task_id: str) -> HTMLResponse:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        wd = workspace_dir_for(entry)
        try:
            return HTMLResponse(_readers.read_viz(wd, task_id))
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc))

    @router.get("/calc/summary")
    def calc_summary(task_id: str) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        return _readers.read_summary(workspace_dir_for(entry))

    @router.get("/calc/iterations")
    def calc_iterations(task_id: str) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        return _readers.read_iterations(workspace_dir_for(entry))

    @router.get("/calc/rough")
    def calc_rough(task_id: str, request: Request) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        wd = workspace_dir_for(entry)
        qp = request.query_params
        bs_str = qp.get("batch_size")
        sl_str = qp.get("seq_len")
        bs = None
        sl = None
        if bs_str is not None:
            try:
                bs = int(bs_str)
            except ValueError:
                raise HTTPException(400, "batch_size / seq_len must be integers")
        if sl_str is not None:
            try:
                sl = int(sl_str)
            except ValueError:
                raise HTTPException(400, "batch_size / seq_len must be integers")
        try:
            return _readers.read_rough(wd, bs, sl)
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @router.get("/calc/cells")
    def calc_cells(task_id: str, request: Request) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        wd = workspace_dir_for(entry)
        qp = request.query_params
        bs_str = qp.get("batch_size")
        sl_str = qp.get("seq_len")
        bs = None
        sl = None
        if bs_str is not None:
            try:
                bs = int(bs_str)
            except ValueError:
                raise HTTPException(400, "batch_size / seq_len must be integers")
        if sl_str is not None:
            try:
                sl = int(sl_str)
            except ValueError:
                raise HTTPException(400, "batch_size / seq_len must be integers")
        try:
            return _readers.read_cells(wd, bs, sl)
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @router.get("/calc/cell/{compound}/{angle}/{round_idx}")
    def calc_cell_detail(
        task_id: str, compound: str, angle: str, round_idx: int,
    ) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        wd = workspace_dir_for(entry)
        try:
            return _readers.read_cell_detail(wd, compound, angle, round_idx)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc))
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    # ----------------------------------------------------------------- #
    # Orchestrator iteration records / charts / state-graph / retro
    # (used to be shell endpoints; now task-owned because the record
    # schema is task-specific)
    # ----------------------------------------------------------------- #
    @router.get("/iterations")
    def calc_orch_iterations(task_id: str) -> list:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        return _state_readers.read_iterations(state_dir_for(entry))

    @router.get("/iterations/{n}")
    def calc_orch_iteration_detail(task_id: str, n: int) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        rec = _state_readers.read_iteration(state_dir_for(entry), n)
        if rec is None:
            raise HTTPException(404, f"no iteration {n} for task {task_id}")
        return rec

    @router.get("/iterations/{n}/retrospective")
    def calc_orch_retrospective(task_id: str, n: int) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        return _state_readers.read_retrospective(state_dir_for(entry), n)

    @router.get("/charts")
    def calc_orch_charts(task_id: str) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        return _state_readers.read_charts(state_dir_for(entry))

    @router.get("/state-graph")
    def calc_orch_state_graph(task_id: str) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        return _state_readers.read_state_graph(state_dir_for(entry))

    # ----------------------------------------------------------------- #
    # Offline QA over agent conversation history
    # ----------------------------------------------------------------- #
    register_qa_routes(router, plugin, prefix="/calc/qa")

    return router
