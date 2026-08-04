"""Phase-specific iteration records for sglang_trace_analyze.

Each phase gets its own dataclass so the schema stays clean — no
``None``-filled optional fields bleeding across phases.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, fields
from typing import Any, Dict


def _base_dict(rec, **overrides) -> Dict[str, Any]:
    """Serialize *any* iteration record to a dict the WebUI can read.

    Keys: phase (str), status, started_at, ended_at, plus phase-specific
    fields from the dataclass.
    """
    out: Dict[str, Any] = {
        "phase": getattr(rec, "phase", ""),
        "status": rec.status,
        "started_at": rec.started_at,
        "ended_at": rec.ended_at,
    }
    for f in fields(rec):
        if f.name in ("phase", "status", "started_at", "ended_at"):
            continue
        val = getattr(rec, f.name)
        if val is not None:
            out[f.name] = val
    out.update(overrides)
    return out


# ------------------------------------------------------------------ #
#  MAPPING phase
# ------------------------------------------------------------------ #

@dataclass
class MappingRecord:
    phase: str = "mapping"
    status: str = "running"
    started_at: float = 0.0
    ended_at: float = 0.0
    batch_size: int | None = None
    trace_dir: str | None = None
    duration_s: float | None = None
    kernel_count: int | None = None
    confidence_issues: int = 0  # entries with low confidence after LLM check
    error: str | None = None

    def start(self):
        self.started_at = time.time()
        self.status = "running"

    def done(self, **kw):
        self.status = "success"
        self.ended_at = time.time()
        for k, v in kw.items():
            setattr(self, k, v)

    def fail(self, error: str):
        self.status = "failed"
        self.ended_at = time.time()
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        return _base_dict(self)


# ------------------------------------------------------------------ #
#  BENCHMARK phase
# ------------------------------------------------------------------ #

@dataclass
class BenchmarkRecord:
    phase: str = "benchmark"
    status: str = "running"
    started_at: float = 0.0
    ended_at: float = 0.0
    batch_size: int | None = None
    trace_dir: str | None = None
    duration_s: float | None = None
    throughput: float | None = None
    latency_p50: float | None = None
    error: str | None = None

    def start(self):
        self.started_at = time.time()
        self.status = "running"

    def done(self, **kw):
        self.status = "success"
        self.ended_at = time.time()
        for k, v in kw.items():
            setattr(self, k, v)

    def fail(self, error: str):
        self.status = "failed"
        self.ended_at = time.time()
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        return _base_dict(self)


# ------------------------------------------------------------------ #
#  ANALYZE phase
# ------------------------------------------------------------------ #

@dataclass
class AnalyzeRecord:
    phase: str = "analyze"
    status: str = "running"
    started_at: float = 0.0
    ended_at: float = 0.0
    batch_size: int | None = None
    stage: str | None = None  # "prefill" | "decode"
    kernel_count: int | None = None
    top_kernel: str | None = None
    top_kernel_pct: float | None = None
    mfu_avg: float | None = None
    fuse_hits: int = 0
    error: str | None = None

    def start(self):
        self.started_at = time.time()
        self.status = "running"

    def done(self, **kw):
        self.status = "success"
        self.ended_at = time.time()
        for k, v in kw.items():
            setattr(self, k, v)

    def fail(self, error: str):
        self.status = "failed"
        self.ended_at = time.time()
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        return _base_dict(self)


# ------------------------------------------------------------------ #
#  HINTS phase
# ------------------------------------------------------------------ #

@dataclass
class HintsRecord:
    phase: str = "hints"
    status: str = "running"
    started_at: float = 0.0
    ended_at: float = 0.0
    model_used: str | None = None
    batch_count: int = 0
    error: str | None = None

    def start(self):
        self.started_at = time.time()
        self.status = "running"

    def done(self, **kw):
        self.status = "success"
        self.ended_at = time.time()
        for k, v in kw.items():
            setattr(self, k, v)

    def fail(self, error: str):
        self.status = "failed"
        self.ended_at = time.time()
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        return _base_dict(self)


# ------------------------------------------------------------------ #
#  SUMMARIZE phase
# ------------------------------------------------------------------ #

@dataclass
class SummarizeRecord:
    phase: str = "summarize"
    status: str = "running"
    started_at: float = 0.0
    ended_at: float = 0.0
    batch_count: int = 0
    best_batch: int | None = None
    best_mfu: float | None = None
    error: str | None = None

    def start(self):
        self.started_at = time.time()
        self.status = "running"

    def done(self, **kw):
        self.status = "success"
        self.ended_at = time.time()
        for k, v in kw.items():
            setattr(self, k, v)

    def fail(self, error: str):
        self.status = "failed"
        self.ended_at = time.time()
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        return _base_dict(self)
