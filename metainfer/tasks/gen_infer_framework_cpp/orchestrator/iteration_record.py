"""gen-infer-framework-cpp's iteration record schema.

Used to live in ``metainfer/orchestrator/state.py`` as the canonical
shell-level dataclass. It's been pulled into the gf task package
because the schema is fundamentally gf/calc-shaped (ABCD phase model,
perf dict, retrospective_path, failure_reason). Tasks with different
iteration shapes are free to define their own record type — the shell's
:class:`metainfer.orchestrator.state.StateStore` treats iteration
records as opaque JSON dicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from metainfer.orchestrator.state import Phase


@dataclass
class IterationRecord:
    """One iteration of the C++ task's ABCDEGF pipeline.

    Fields are gf-specific: ``goal`` is what plan.md promises this iter,
    ``start_phase`` is which ABCDEGF phase the iter resumes from, ``perf``
    holds the E-step's metrics (tokens/s, ms/req, memory MB, etc.),
    ``outcome`` is the terminal transition of the C step, and
    ``retrospective_path`` is the absolute path to the post-review retro .md.
    """
    iteration: int
    goal: str = ""
    start_phase: Phase = ""   # ABCDEGF phase to resume from (default set by pipeline)
    started_at: float = 0.0
    ended_at: float = 0.0
    duration_s: float = 0.0
    status: str = "running"   # running | success | failed
    failure_reason: Optional[str] = None
    # outcome of the iteration's terminating C step (None until C has run)
    outcome: Optional[str] = None
    phases: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # perf metrics from E step (tokens/s, ms/req, memory MB, etc.)
    perf: Dict[str, float] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    # True iff the orchestrator process died mid-flight and this record was
    # finalized retroactively on the next resume. Distinguishes a "real"
    # failed C step from an externally-interrupted attempt.
    interrupted: bool = False
    # Absolute path to this iteration's retrospective.md (written by
    # the retrospective agent at the end of an iteration). None if no
    # retrospective was ever produced.
    retrospective_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IterationRecord":
        # Tolerate missing keys (older records from before new fields
        # were added) by filtering to known dataclass fields.
        names = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in names})
