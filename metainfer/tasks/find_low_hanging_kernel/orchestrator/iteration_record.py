"""Iteration record schema for find-low-hanging-kernel.

An "iteration" in this task is one graph-validation round (Step 3b).
Steps 1, 2, 3a (build), and 4 are single-shot and live inside iteration 1's
record before the first validation round opens iteration 2.
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
    # The validation round number that this iteration represents (1-indexed).
    # Round 0 == pre-validation build.
    round: int = 0
    # Issues the integrity check found + fixed deterministically.
    integrity_fixes: List[Dict[str, Any]] = field(default_factory=list)
    # Issues the 5-worker pool reported (per-group, flattened).
    semantic_issues: List[Dict[str, Any]] = field(default_factory=list)
    # Outcome of the round: "clean" | "needs_fix" | "failed".
    outcome: Optional[str] = None
    artifacts: List[str] = field(default_factory=list)
    interrupted: bool = False
