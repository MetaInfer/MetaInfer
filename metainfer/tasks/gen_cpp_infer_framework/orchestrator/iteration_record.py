"""Iteration-record schema owned by the C++ inference task.

The shared :class:`metainfer.orchestrator.state.StateStore` persists task
records as opaque dictionaries.  This task keeps its ABCDEF-specific shape
locally and converts at the store boundary.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from metainfer.orchestrator.state import Phase


@dataclass
class IterationRecord:
    """One iteration of the C++ task's ABCDEF pipeline."""

    iteration: int
    goal: str = ""
    start_phase: Phase = ""
    started_at: float = 0.0
    ended_at: float = 0.0
    duration_s: float = 0.0
    status: str = "running"
    failure_reason: Optional[str] = None
    outcome: Optional[str] = None
    phases: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    perf: Dict[str, float] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    interrupted: bool = False
    retrospective_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IterationRecord":
        """Load older records while ignoring unknown future fields."""
        names = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in data.items() if key in names})
