"""Iteration schema owned exclusively by the GEMM kernel task."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class IterationRecord:
    iteration: int
    started_at: float
    start_phase: str = "A_plan"
    ended_at: float = 0.0
    duration_s: float = 0.0
    status: str = "running"
    outcome: Optional[str] = None
    failure_reason: Optional[str] = None
    phases: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    score: Dict[str, Any] = field(default_factory=dict)
    measurement_report: Dict[str, str] = field(default_factory=dict)
    profile_report: Dict[str, str] = field(default_factory=dict)
    incumbent_measurement_report: Dict[str, str] = field(default_factory=dict)
    incumbent_profile_report: Dict[str, str] = field(default_factory=dict)
    promoted: bool = False
    champion_iteration: int = 0
    artifacts: List[str] = field(default_factory=list)
    retrospective_path: Optional[str] = None
    interrupted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IterationRecord":
        names = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in data.items() if key in names})
