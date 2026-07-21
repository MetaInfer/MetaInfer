"""evolve-kernel iteration record schema .

Tracks:
  - Iteration (consumed on each H→E loop-back transition)
  - Phase outcomes per iteration
  - Current selected kernel, exec time, complexity
  - Harness paths
  - Performance metrics
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from metainfer.orchestrator.state import Phase


@dataclass
class IterationRecord:
    iteration: int
    goal: str = ""
    start_phase: Phase = ""
    started_at: float = 0.0
    ended_at: float = 0.0
    duration_s: float = 0.0
    status: str = "running"  # running | success | failed
    failure_reason: Optional[str] = None
    outcome: Optional[str] = None
    phases: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    perf: Dict[str, float] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    interrupted: bool = False
    retrospective_path: Optional[str] = None

    # fields
    correctness_harness_path: Optional[str] = None
    perf_harness_path: Optional[str] = None
    selected_kernel_id: Optional[str] = None
    selected_kernel_parent_id: Optional[str] = None
    optimized_kernel_path: Optional[str] = None
    exec_time_ms: float = 0.0
    complexity_score: float = 0.5
    combined_score: float = 0.0
    speedup_vs_original: float = 1.0
    kernel_library_size: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IterationRecord":
        names = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in names})
