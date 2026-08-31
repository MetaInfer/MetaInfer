"""Deterministic no-GPU adapter used to validate the orchestration MVP."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .base import AdapterResult, KernelAdapter
from ..config import ShapeSpec


class MockKernelAdapter(KernelAdapter):
    requires_gpu = False

    def describe_environment(self) -> Dict[str, Any]:
        return {
            "adapter": "mock",
            "requires_gpu": False,
            "gpu_runtime_loaded": False,
        }

    def prepare(self, workspace: Path) -> AdapterResult:
        workspace.mkdir(parents=True, exist_ok=True)
        return AdapterResult(True, evidence={"workspace": str(workspace)})

    def build(self, workspace: Path) -> AdapterResult:
        return AdapterResult(True, evidence={"build": "mock-success"})

    def correctness(self, workspace: Path, shape: ShapeSpec) -> AdapterResult:
        return AdapterResult(
            True,
            evidence={
                "reference": "mock-trusted-reference",
                "seed": 20260724,
                "shape": {"id": shape.id, **shape.params},
            },
        )

    def benchmark(
        self, workspace: Path, shape: ShapeSpec, *, iteration: int = 0
    ) -> AdapterResult:
        seed = sum(ord(ch) for ch in shape.id)
        baseline_us = 80.0 + float(seed % 70)
        factor = max(0.70, 1.0 - 0.025 * max(0, iteration))
        median_us = round(baseline_us * factor, 4)
        try:
            m = float(shape.params.get("M", 0))
            n = float(shape.params.get("N", 0))
            k = float(shape.params.get("K", 0))
        except (TypeError, ValueError):
            m = n = k = 0.0
        elapsed_s = median_us * 1e-6
        tflops = (
            (2.0 * m * n * k) / elapsed_s / 1e12
            if elapsed_s > 0 and m > 0 and n > 0 and k > 0 else 0.0
        )
        # Mock W8A8 traffic model: int8 A/B plus int32 output.
        bytes_moved = m * k + k * n + 4.0 * m * n
        bandwidth_gb_s = (
            bytes_moved / elapsed_s / 1e9
            if elapsed_s > 0 and bytes_moved > 0 else 0.0
        )
        return AdapterResult(
            True,
            metrics={
                "median_us": median_us,
                "p90_us": round(median_us * 1.015, 4),
                "min_us": round(median_us * 0.99, 4),
                "max_us": round(median_us * 1.03, 4),
                "tflops": round(tflops, 4),
                "bandwidth_gb_s": round(bandwidth_gb_s, 4),
            },
            evidence={"samples": 30, "warmup": 10, "mock": True},
        )

    def profile(self, workspace: Path, shape: ShapeSpec) -> AdapterResult:
        return AdapterResult(
            True,
            evidence={
                "tool": "mock-profiler",
                "bottleneck": "synthetic-dispatch-overhead",
                "profile_is_not_benchmark": True,
            },
        )
