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
        result = _state_readers.read_kernel_library(workspace_dir_for(entry))
        # Attach optimizer_mode from requirements for smart code display
        result["optimizer_mode"] = _state_readers.read_optimizer_mode(state_dir_for(entry))
        return result

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

    # ---- Kernel Lineage ----

    @router.get("/kernel-library/{kernel_id}/lineage")
    def ok_kernel_lineage(task_id: str, kernel_id: str) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        result = _state_readers.read_kernel_lineage(
            workspace_dir_for(entry), state_dir_for(entry), kernel_id,
        )
        if "error" in result:
            raise HTTPException(404, result["error"])
        return result

    # ---- Failures ----

    @router.get("/failures")
    def ok_failures(task_id: str) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        return _state_readers.read_failures(state_dir_for(entry))

    # ---- Shape Benchmark ----

    @router.get("/shape-benchmark")
    def ok_shape_benchmark(task_id: str) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        return _state_readers.read_shape_benchmark(
            state_dir_for(entry), workspace_dir_for(entry),
        )

    @router.post("/shape-benchmark/refresh")
    def ok_shape_benchmark_refresh(task_id: str) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        return _state_readers.refresh_shape_benchmark(
            state_dir_for(entry), workspace_dir_for(entry),
        )

    # ---- Multi-GPU ----

    @router.get("/gpu-status")
    def ok_gpu_status(task_id: str) -> Dict[str, Any]:
        """Return live status of all GPU workers for a multi-GPU task."""
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        return _state_readers.read_gpu_status(
            state_dir_for(entry), workspace_dir_for(entry),
        )

    @router.get("/aggregate-bench")
    def ok_aggregate_bench(task_id: str) -> Dict[str, Any]:
        """Aggregated shape benchmarks across all GPUs."""
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        return _state_readers.read_aggregate_bench(
            state_dir_for(entry), workspace_dir_for(entry),
        )

    @router.get("/combined-timeline")
    def ok_combined_timeline(task_id: str, since: float = 0.0) -> Dict[str, Any]:
        """Timeline events from parent + all GPU workers."""
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        events = _state_readers.read_combined_timeline(
            state_dir_for(entry), since=since,
        )
        return {"events": events}

    # ---- Per-GPU Detail ----

    @router.get("/gpu/{gpu_idx}/detail")
    def ok_gpu_detail(task_id: str, gpu_idx: int) -> Dict[str, Any]:
        """Aggregated detail for one GPU worker: state graph + kernel library + agents."""
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        return _state_readers.read_gpu_detail(
            state_dir_for(entry), workspace_dir_for(entry), gpu_idx,
        )

    @router.get("/gpu/{gpu_idx}/harnesses/{harness_type}")
    def ok_gpu_harness(task_id: str, gpu_idx: int, harness_type: str) -> Dict[str, Any]:
        """Read correctness or perf harness for one GPU worker."""
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        return _state_readers.read_gpu_harness(
            workspace_dir_for(entry), gpu_idx, harness_type,
        )

    @router.get("/gpu/{gpu_idx}/kernel-library/{kernel_id}/lineage")
    def ok_gpu_kernel_lineage(task_id: str, gpu_idx: int, kernel_id: str) -> Dict[str, Any]:
        """Kernel lineage for one GPU worker."""
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        result = _state_readers.read_gpu_kernel_lineage(
            workspace_dir_for(entry), state_dir_for(entry), gpu_idx, kernel_id,
        )
        if "error" in result:
            raise HTTPException(404, result["error"])
        return result

    # ---- Summary Report ----

    @router.get("/summary")
    def ok_summary(task_id: str) -> Dict[str, Any]:
        """Return the end-of-task summary report (Markdown text)."""
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        return _state_readers.read_summary(state_dir_for(entry), workspace_dir_for(entry))

    # ---- QA ----

    return router
