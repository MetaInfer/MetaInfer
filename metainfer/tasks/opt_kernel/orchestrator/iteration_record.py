"""opt-kernel iteration record schema.

Borrowed from gen-infer-framework — same ABCDEF shape (perf dict,
retrospective_path, failure_reason, outcome).
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
        names = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in names})
