"""Independent mock worker loop with durable experiment records."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict

from .adapters.base import KernelAdapter
from .config import (
    ROUND_ACCEPTANCE_IMPROVEMENT_PERCENT,
    OptimizerConfig,
    WorkerAssignment,
)
from .guidance import claim_next_guidance
from .result_store import SCHEMA_VERSION, append_jsonl, write_json


def _status(
    path: Path,
    *,
    assignment: WorkerAssignment,
    state: str,
    iteration: int,
    shape_id: str | None,
    best: Dict[str, Any] | None,
) -> None:
    write_json(path, {
        "schema_version": SCHEMA_VERSION,
        "worker_id": assignment.worker_id,
        "state": state,
        "iteration": iteration,
        "shape_id": shape_id,
        "pid": None,
        "physical_gpu": assignment.gpu,
        "logical_gpu": 0,
        "gpu_binding": {
            "HIP_VISIBLE_DEVICES": str(assignment.gpu),
            "ROCR_VISIBLE_DEVICES": None,
            "strategy": "HIP_VISIBLE_DEVICES-only",
            "enforced": False,
            "reason": "mock mode does not launch a GPU subprocess",
        },
        "best": best,
        "last_update": time.time(),
    })


def run_mock_worker(
    *,
    assignment: WorkerAssignment,
    config: OptimizerConfig,
    baseline: Dict[str, Dict[str, float]],
    worker_root: Path,
    guidance_root: Path,
    adapter_factory: Callable[[], KernelAdapter],
) -> Dict[str, Any]:
    """Run one deterministic worker. Each caller owns a disjoint root."""
    adapter = adapter_factory()
    for name in ("source", "build", "cache", "logs", "runs", "artifacts"):
        (worker_root / name).mkdir(parents=True, exist_ok=True)
    status_path = worker_root / "status.json"
    _status(
        status_path, assignment=assignment, state="starting",
        iteration=0, shape_id=None, best=None,
    )

    worker_result: Dict[str, Any] = {
        "worker_id": assignment.worker_id,
        "physical_gpu": assignment.gpu,
        "branch": f"metainfer/mock/{assignment.worker_id}",
        "worktree_created": False,
        "mode": "mock",
        "shapes": {},
    }

    for shape_id in assignment.shape_ids:
        shape = config.shapes[shape_id]
        run_dir = worker_root / "runs" / shape_id
        experiments_path = run_dir / "experiments.jsonl"
        baseline_metrics = baseline[shape_id]
        baseline_us = baseline_metrics["median_us"]
        best: Dict[str, Any] = {
            "shape_id": shape_id,
            "iteration": 0,
            "median_us": baseline_us,
            "p90_us": baseline_metrics["p90_us"],
            "speedup": 1.0,
            "commit": None,
            "mock_candidate": "baseline",
        }
        write_json(run_dir / "best.json", best)

        for iteration in range(1, config.mock_iterations + 1):
            guidance = claim_next_guidance(
                guidance_root, assignment.worker_id, iteration
            )
            _status(
                status_path, assignment=assignment, state="profiling",
                iteration=iteration, shape_id=shape_id, best=best,
            )
            profile = adapter.profile(worker_root, shape)
            build = adapter.build(worker_root)
            correct = adapter.correctness(worker_root, shape)
            bench = (
                adapter.benchmark(worker_root, shape, iteration=iteration)
                if build.success and correct.success else None
            )
            median_us = (
                bench.metrics.get("median_us") if bench is not None else None
            )
            speedup = (
                baseline_us / median_us
                if median_us is not None and median_us > 0 else 0.0
            )
            improvement = (
                best["median_us"] / median_us - 1.0
                if median_us is not None and median_us > 0 else 0.0
            ) * 100.0
            accepted = bool(
                build.success
                and correct.success
                and bench is not None
                and bench.success
                and median_us < best["median_us"]
                and improvement >= ROUND_ACCEPTANCE_IMPROVEMENT_PERCENT
            )
            experiment = {
                "schema_version": SCHEMA_VERSION,
                "worker_id": assignment.worker_id,
                "iteration": iteration,
                "shape_id": shape_id,
                "shape": shape.params,
                "hypothesis": (
                    f"Apply manual guidance: {guidance['text']}"
                    if guidance else f"mock-hypothesis-{iteration}"
                ),
                "changes": [
                    (
                        f"manual plan: {guidance['text']}"
                        if guidance else f"synthetic candidate {iteration}"
                    )
                ],
                "manual_guidance": guidance["text"] if guidance else None,
                "guidance_id": guidance["id"] if guidance else None,
                "profile_evidence": profile.evidence,
                "build_success": build.success,
                "correctness_passed": correct.success,
                "metrics": bench.metrics if bench else {},
                "baseline_us": baseline_us,
                "speedup": round(speedup, 6),
                "accepted": accepted,
                "commit": None,
                "failure_reason": None,
                "timestamp": time.time(),
            }
            append_jsonl(experiments_path, experiment)
            if accepted:
                best = {
                    "shape_id": shape_id,
                    "iteration": iteration,
                    "median_us": median_us,
                    "p90_us": bench.metrics["p90_us"],
                    "speedup": round(speedup, 6),
                    "commit": None,
                    "mock_candidate": f"candidate-{iteration}",
                }
                write_json(run_dir / "best.json", best)
            _status(
                status_path, assignment=assignment, state="benchmarking",
                iteration=iteration, shape_id=shape_id, best=best,
            )
        worker_result["shapes"][shape_id] = best

    _status(
        status_path, assignment=assignment, state="completed",
        iteration=config.mock_iterations, shape_id=None, best=None,
    )
    write_json(worker_root / "result.json", worker_result)
    return worker_result
