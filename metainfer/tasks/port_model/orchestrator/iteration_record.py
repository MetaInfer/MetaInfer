"""Iteration record schema for port-model.

An "iteration" in this task is one P4→P5 implement-test-repair cycle.
P1-P3 are single-shot analysis within iteration 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class IterationRecord:
    iteration: int
    goal: str = ""
    started_at: float = 0.0
    ended_at: float = 0.0
    duration_s: float = 0.0
    status: str = "running"  # "running" | "success" | "failed"
    phase: Optional[str] = None
    outcome: Optional[str] = None
    test_results: Optional[Dict[str, Any]] = None
    artifacts: List[str] = field(default_factory=list)
    interrupted: bool = False
