"""FastAPI router for sglang_trace_analyze.

Routes mounted under ``/api/sglang_trace_analyze/{task_id}``:

    GET /summary               → summary.json
    GET /mapping               → mapping.json
    GET /hints                 → hints.json
    GET /batch/{bs}/{stage}    → {kernel_table, overlap, fuse}
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from metainfer.server._helpers import (
    require_task_type,
    state_dir_for,
    task_or_404,
)
from . import _state_readers

PLUGIN_TYPE = "sglang_trace_analyze"


def build_router(plugin) -> APIRouter:
    router = APIRouter()

    @router.get("/summary")
    def get_summary(task_id: str):
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        data = _state_readers.read_summary(state_dir_for(entry))
        if data is None:
            raise HTTPException(404, "summary not yet available")
        return data

    @router.get("/mapping")
    def get_mapping(task_id: str):
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        data = _state_readers.read_mapping(state_dir_for(entry))
        if data is None:
            raise HTTPException(404, "mapping not yet available")
        return data

    @router.get("/hints")
    def get_hints(task_id: str):
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        data = _state_readers.read_hints(state_dir_for(entry))
        if data is None:
            raise HTTPException(404, "hints not yet available")
        return data

    @router.get("/batch/{bs}/{stage}")
    def get_batch_detail(task_id: str, bs: int, stage: str):
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        data = _state_readers.read_batch_detail(
            state_dir_for(entry), bs, stage
        )
        if data is None:
            raise HTTPException(
                404, f"no analysis data for batch {bs}/{stage}"
            )
        return data

    return router
