"""Kernel library for the evolve-kernel optimization loop.

Maintains a ranked pool of kernels (max 10). Each kernel has:
  - exec_time_ms: measured execution time (lower is better)
  - complexity_score: 0-1 agent-assessed complexity (lower = simpler = better)
  - combined_score: weighted combination, higher is better

Selection from the library is weighted-random by combined_score, giving
preference to kernels that are both fast AND simple (easier to further
optimize).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


MAX_LIBRARY_SIZE = 10


@dataclass
class KernelEntry:
    """One kernel in the optimization library."""
    id: str
    code: str
    exec_time_ms: float = 0.0
    complexity_score: float = 0.5  # 0=simplest, 1=most complex
    combined_score: float = 0.0
    iteration_added: int = 0
    parent_id: Optional[str] = None

    # Headroom analysis (populated during Phase H after perf measurement)
    headroom_bottleneck: Optional[str] = None        # memory_bound | compute_bound | inefficient | near_optimal
    headroom_roofline_efficiency_pct: float = 0.0    # P_achieved / P_max × 100 — primary metric
    headroom_pct: float = 0.0                        # 100 - roofline_efficiency (for backward compat)
    headroom_p_max_tflops: float = 0.0               # min(P_compute, BW_peak × AI)
    headroom_p_bw_roof_tflops: float = 0.0           # BW_peak × AI
    headroom_ai_ridge: float = 0.0                   # P_compute / BW_hbm ridge point
    headroom_suggestions_json: Optional[str] = None  # JSON-encoded list of suggestion strings
    headroom_advice: Optional[str] = None            # human-readable optimization advice paragraph
    headroom_achieved_bw_gbps: float = 0.0           # achieved HBM bandwidth (GB/s) — from theoretical bytes
    headroom_achieved_tflops: float = 0.0            # achieved compute throughput (TFLOPS) — from theoretical FLOPs
    headroom_peak_bw_gbps: float = 0.0               # peak HBM bandwidth of the GPU
    headroom_peak_tflops: float = 0.0                # peak TFLOPS for the kernel's compute dtype
    headroom_arithmetic_intensity: float = 0.0       # FLOP / byte (HBM-level, theoretical)
    headroom_measured_ai: float = 0.0                # FLOP / byte from profiler-measured HBM bytes (0=none)
    headroom_shape_label: str = ""                   # shape used for roofline analysis (e.g. "M×2048 (K)×4096")
    headroom_M: int = 0
    headroom_N: int = 0
    headroom_K: int = 0
    # Deprecated fields kept for backward compat with old kernel_library.json
    headroom_bw_util_pct: float = 0.0
    headroom_compute_util_pct: float = 0.0

    # HIP C++ kernel source (from scratch mode — the real .cpp file content)
    cpp_code: Optional[str] = None

    # hipprof profiling (populated during Phase H when enable_profiling=True)
    profiled: bool = False
    profiling_kernel_duration_us: float = 0.0
    profiling_achieved_bw_gbps: float = 0.0
    profiling_occupancy_pct: float = 0.0
    profiling_l2_cache_hit_pct: float = 0.0
    profiling_advice: Optional[str] = None             # profiling-based optimization advice

    def recompute_combined(self) -> float:
        """Recompute combined_score from exec_time and complexity.

        Higher combined_score = better. Formula:
          perf_score = 1.0 / max(exec_time_ms, 1e-6)  (normalized by 10ms baseline)
          simplicity_bonus = 1.0 - complexity_score
          combined = perf_score * 0.7 + simplicity_bonus * 0.3
        """
        perf = 10.0 / max(self.exec_time_ms, 1e-6)  # 10ms baseline → 1.0
        simplicity = 1.0 - self.complexity_score
        self.combined_score = 0.7 * perf + 0.3 * simplicity
        return self.combined_score


class KernelLibrary:
    """Ranked pool of kernels driving the optimization loop.

    Thread-safe for single-orchestrator use (no concurrent access).
    Persisted to ``kernel_library.json`` in the workspace directory.
    """

    def __init__(self, kernels: Optional[List[KernelEntry]] = None) -> None:
        self._kernels: List[KernelEntry] = list(kernels or [])
        self._sort()

    def _sort(self) -> None:
        self._kernels.sort(key=lambda k: k.combined_score, reverse=True)
        # Trim to max size
        if len(self._kernels) > MAX_LIBRARY_SIZE:
            self._kernels = self._kernels[:MAX_LIBRARY_SIZE]

    @property
    def kernels(self) -> List[KernelEntry]:
        return list(self._kernels)

    @property
    def size(self) -> int:
        return len(self._kernels)

    @property
    def best(self) -> Optional[KernelEntry]:
        return self._kernels[0] if self._kernels else None

    def add(self, entry: KernelEntry) -> bool:
        """Add a kernel. Returns True if it was added (beats or fills a slot).

        If library is at capacity, the new kernel only enters if its
        combined_score beats the lowest-ranked kernel's score.
        """
        entry.recompute_combined()

        if self.size < MAX_LIBRARY_SIZE:
            self._kernels.append(entry)
            self._sort()
            return True

        # Full — only add if it beats the current last place
        if entry.combined_score > self._kernels[-1].combined_score:
            self._kernels.append(entry)
            self._sort()
            return True

        return False

    def select(self) -> Optional[KernelEntry]:
        """Weighted-random selection. Returns None if library is empty.

        Weight = combined_score (higher = more likely to be selected).
        If all scores are zero or negative, uniform random.
        """
        if not self._kernels:
            return None
        if self.size == 1:
            return self._kernels[0]

        weights = [max(k.combined_score, 1e-9) for k in self._kernels]
        total = sum(weights)
        if total <= 0:
            return random.choice(self._kernels)

        # Weighted random selection
        r = random.random() * total
        cumulative = 0.0
        for k, w in zip(self._kernels, weights):
            cumulative += w
            if r <= cumulative:
                return k

        return self._kernels[-1]  # fallback

    def get_by_id(self, kernel_id: str) -> Optional[KernelEntry]:
        for k in self._kernels:
            if k.id == kernel_id:
                return k
        return None

    def top_n(self, n: int = 5) -> List[KernelEntry]:
        return self._kernels[:n]

    def to_list(self) -> List[Dict[str, Any]]:
        return [asdict(k) for k in self._kernels]

    @classmethod
    def from_list(cls, data: List[Dict[str, Any]]) -> "KernelLibrary":
        kernels = [
            KernelEntry(
                id=d["id"],
                code=d["code"],
                exec_time_ms=d.get("exec_time_ms", 0.0),
                complexity_score=d.get("complexity_score", 0.5),
                combined_score=d.get("combined_score", 0.0),
                iteration_added=d.get("iteration_added", 0),
                parent_id=d.get("parent_id"),
                headroom_bottleneck=d.get("headroom_bottleneck"),
                headroom_roofline_efficiency_pct=(
                    d.get("headroom_roofline_efficiency_pct", 0.0)
                    or max(d.get("headroom_bw_util_pct", 0), d.get("headroom_compute_util_pct", 0))
                ),
                headroom_pct=d.get("headroom_pct", 0.0),
                headroom_p_max_tflops=d.get("headroom_p_max_tflops", 0.0),
                headroom_p_bw_roof_tflops=d.get("headroom_p_bw_roof_tflops", 0.0),
                headroom_ai_ridge=d.get("headroom_ai_ridge", 0.0),
                headroom_suggestions_json=d.get("headroom_suggestions_json"),
                headroom_advice=d.get("headroom_advice"),
                headroom_achieved_bw_gbps=d.get("headroom_achieved_bw_gbps", 0.0),
                headroom_achieved_tflops=d.get("headroom_achieved_tflops", 0.0),
                headroom_peak_bw_gbps=d.get("headroom_peak_bw_gbps", 0.0),
                headroom_peak_tflops=d.get("headroom_peak_tflops", 0.0),
                headroom_arithmetic_intensity=d.get("headroom_arithmetic_intensity", 0.0),
                headroom_measured_ai=d.get("headroom_measured_ai", 0.0),
                headroom_shape_label=d.get("headroom_shape_label", ""),
                headroom_M=d.get("headroom_M", 0),
                headroom_N=d.get("headroom_N", 0),
                headroom_K=d.get("headroom_K", 0),
                headroom_bw_util_pct=d.get("headroom_bw_util_pct", 0.0),
                headroom_compute_util_pct=d.get("headroom_compute_util_pct", 0.0),
                cpp_code=d.get("cpp_code"),
                profiled=d.get("profiled", False),
                profiling_kernel_duration_us=d.get("profiling_kernel_duration_us", 0.0),
                profiling_achieved_bw_gbps=d.get("profiling_achieved_bw_gbps", 0.0),
                profiling_occupancy_pct=d.get("profiling_occupancy_pct", 0.0),
                profiling_l2_cache_hit_pct=d.get("profiling_l2_cache_hit_pct", 0.0),
                profiling_advice=d.get("profiling_advice"),
            )
            for d in data
        ]
        return cls(kernels)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_list(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "KernelLibrary":
        if not path.is_file():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                return cls()
            return cls.from_list(data)
        except (json.JSONDecodeError, OSError, KeyError):
            return cls()

    def last_added(self) -> Optional[KernelEntry]:
        """Return the most recently added kernel (highest iteration_added)."""
        if not self._kernels:
            return None
        return max(self._kernels, key=lambda k: k.iteration_added)
