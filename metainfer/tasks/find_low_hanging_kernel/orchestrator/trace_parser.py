"""Deterministic Chrome tracing parser (Step 2a).

This module is the **only** code that reads the raw trace file. Analysis agents
read its output (`trace_parsed.json`) instead — per the task spec, agents must
not be fed the raw (potentially huge) trace.

Supports:
- ``.json`` and ``.json.gz`` transparently
- Standard Chrome trace event format: ``{"traceEvents": [...]}`` or a bare list
- Complete events (``ph == "X"``) and begin/end event pairs (``B``/``E``)
- Per-``(name, cat)`` aggregation: count, mean, std, min, max, percentiles,
  total, histogram, and the set of (pid, tid) pairs observed
- Flow events (``ph == f``/``s``/``t``) capture for CPU↔GPU correlation
- Heuristic detection of CUDA Graph capture and CPU-stack presence

Output schema is documented in :func:`parse_trace`.
"""

from __future__ import annotations

import gzip
import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Fixed histogram buckets in microseconds (log10-spaced). Coarse on purpose:
# the goal is to let the analyst spot bimodal distributions, not to be a
# precise density estimator.
_HIST_EDGES_US: Tuple[float, ...] = (
    0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0,
    1_000.0, 2_000.0, 5_000.0, 10_000.0, 20_000.0, 50_000.0,
    100_000.0, math.inf,
)

_HIST_LABELS: Tuple[str, ...] = (
    "<1us", "1-2us", "2-5us", "5-10us", "10-20us", "20-50us",
    "50-100us", "100-200us", "200-500us", "0.5-1ms", "1-2ms",
    "2-5ms", "5-10ms", "10-20ms", "20-50ms", "50-100ms", ">100ms",
)

# Cats we treat as GPU/kernel work for the "GPU summary" block.
_GPU_CATS = {"kernel", "gpu", "cuda", "cuda_runtime", "cuda_meta", "runtime"}
# Cats that indicate CPU-side call-stack events.
_CPU_CATS = {"cpu", "python", "python.function", "C++", "ftrace"}
# Markers that hint at CUDA Graph capture vs replay.
_GRAPH_NAME_HINTS = ("cuda_graph", "cudagraph", "graph_replay", "graph capture")


@dataclass
class _Agg:
    name: str
    cat: str
    durs: List[float] = field(default_factory=list)
    pids: set = field(default_factory=set)
    tids: set = field(default_factory=set)
    args_examples: List[Dict[str, Any]] = field(default_factory=list)


def _open_trace(path: Path):
    """Open .json or .json.gz transparently, returning a text stream."""
    p = Path(path)
    if str(p).endswith(".gz"):
        return gzip.open(p, "rt", encoding="utf-8")
    return open(p, "r", encoding="utf-8")


def _load_events(path: Path) -> List[Dict[str, Any]]:
    """Load trace events from a Chrome trace file. Accepts either a bare
    list or ``{"traceEvents": [...], ...}``."""
    with _open_trace(path) as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        evts = raw.get("traceEvents")
        if isinstance(evts, list):
            return evts
    raise ValueError(
        f"unrecognized trace format in {path}: "
        f"expected list or {{'traceEvents': [...]}}, got {type(raw).__name__}"
    )


def _bucketize(durs: List[float]) -> List[Dict[str, Any]]:
    counts = [0] * (len(_HIST_EDGES_US) - 1)
    for d in durs:
        for i in range(len(_HIST_EDGES_US) - 1):
            if _HIST_EDGES_US[i] <= d < _HIST_EDGES_US[i + 1]:
                counts[i] += 1
                break
    return [{"label": _HIST_LABELS[i], "count": counts[i]} for i in range(len(counts))]


def _percentile(sorted_durs: List[float], p: float) -> Optional[float]:
    if not sorted_durs:
        return None
    if len(sorted_durs) == 1:
        return sorted_durs[0]
    k = (len(sorted_durs) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_durs[int(k)]
    d0 = sorted_durs[f] * (c - k)
    d1 = sorted_durs[c] * (k - f)
    return d0 + d1


def _finalize_agg(agg: _Agg) -> Dict[str, Any]:
    durs = agg.durs
    n = len(durs)
    if n == 0:
        return {}
    sorted_durs = sorted(durs)
    total = sum(durs)
    mean = total / n
    std = statistics.pstdev(durs) if n > 1 else 0.0
    # Bimodal flag: high coefficient of variation OR a large max/min ratio
    # (a single kernel name with very different per-call durations is a
    # strong signal that one name is invoked from multiple call sites with
    # different shapes — a key disambiguation target for Step 2).
    cv = (std / mean) if mean > 0 else 0.0
    max_min_ratio = (
        (sorted_durs[-1] / sorted_durs[0])
        if sorted_durs[0] > 0 else 0.0
    )
    bimodal = (cv > 0.5) or (max_min_ratio > 3.0 and n >= 2)
    return {
        "name": agg.name,
        "cat": agg.cat,
        "count": n,
        "total_us": round(total, 2),
        "mean_us": round(mean, 2),
        "std_us": round(std, 2),
        "min_us": round(sorted_durs[0], 2),
        "max_us": round(sorted_durs[-1], 2),
        "p50_us": round(_percentile(sorted_durs, 0.50) or 0.0, 2),
        "p95_us": round(_percentile(sorted_durs, 0.95) or 0.0, 2),
        "p99_us": round(_percentile(sorted_durs, 0.99) or 0.0, 2),
        "histogram": _bucketize(durs),
        "pids": sorted(str(p) for p in agg.pids),
        "tids": sorted(str(t) for t in agg.tids),
        "args_examples": agg.args_examples[:3],
        "bimodal_suspect": bimodal,
    }


def _looks_like_cuda_graph(events: List[Dict[str, Any]]) -> bool:
    """Heuristic: do we see events whose name or cat matches CUDA-graph
    patterns? Triggers the "trace may undercount individual kernel launches
    because they were captured" warning in the synthesis."""
    sample = events[:5000] if len(events) > 5000 else events
    for e in sample:
        cat = str(e.get("cat", "")).lower()
        name = str(e.get("name", "")).lower()
        if any(h in cat for h in _GRAPH_NAME_HINTS):
            return True
        if any(h in name for h in _GRAPH_NAME_HINTS):
            return True
        args = e.get("args") or {}
        if isinstance(args, dict):
            for k in args:
                if any(h in str(k).lower() for h in _GRAPH_NAME_HINTS):
                    return True
    return False


def _has_cpu_stack(events: List[Dict[str, Any]]) -> bool:
    sample = events[:5000] if len(events) > 5000 else events
    for e in sample:
        cat = str(e.get("cat", "")).lower()
        if cat in _CPU_CATEGORIES or cat in _CPU_CATS:
            return True
    return False


# Add a few more aliases seen in the wild (pytorch profiler, rocm).
_CPU_CATEGORIES = {
    "cpu", "python", "python_function", "python.function",
    "c++", "ftrace", "user_annotation", "function",
}


def parse_trace(path: Path) -> Dict[str, Any]:
    """Parse a Chrome trace file and return the aggregated summary.

    Returns a dict with the following structure::

        {
          "source": str,                    # filename of the trace
          "event_count": int,               # total events seen
          "has_cpu_stack": bool,            # whether CPU-side stack was captured
          "cuda_graph_detected": bool,      # heuristic CUDA-graph marker
          "cats_present": [str, ...],       # distinct categories observed
          "by_name_cat": [                  # one entry per (name, cat)
            {"name": ..., "cat": ..., "count": ..., "mean_us": ...,
             "std_us": ..., "min_us": ..., "max_us": ...,
             "p50_us": ..., "p95_us": ..., "p99_us": ...,
             "total_us": ..., "histogram": [...],
             "pids": [...], "tids": [...], "args_examples": [...],
             "bimodal_suspect": bool},
            ...
          ],
          "flow_events": [                  # CPU↔GPU correlation events
            {"name": ..., "cat": ..., "ph": ..., "pid": ..., "tid": ..., "ts": ...},
            ...
          ],
        }

    Entries in ``by_name_cat`` are sorted by ``total_us`` descending so the
    top entries are the biggest optimization targets.
    """
    path = Path(path)
    events = _load_events(path)

    # Aggregate per (name, cat). Begin/end events (B/E) need pairing by tid.
    aggs: Dict[Tuple[str, str], _Agg] = {}
    open_be: Dict[Tuple[Any, Any, str, str], float] = {}  # (pid,tid,name,cat) -> ts
    flow_events: List[Dict[str, Any]] = []
    cats_present: set = set()

    for e in events:
        ph = e.get("ph")
        name = e.get("name", "")
        cat = e.get("cat", "")
        if cat:
            cats_present.add(str(cat))

        # Flow events for correlation
        if ph in ("f", "s", "t"):
            flow_events.append({
                "name": str(name), "cat": str(cat), "ph": str(ph),
                "pid": e.get("pid"), "tid": e.get("tid"), "ts": e.get("ts"),
            })
            continue

        # Complete events with dur
        if ph == "X":
            dur_us = e.get("dur")
            if dur_us is None or not isinstance(dur_us, (int, float)):
                continue
            if not name or not cat:
                continue
            key = (str(name), str(cat))
            agg = aggs.setdefault(key, _Agg(name=str(name), cat=str(cat)))
            agg.durs.append(float(dur_us))
            if "pid" in e:
                agg.pids.add(e.get("pid"))
            if "tid" in e:
                agg.tids.add(e.get("tid"))
            args = e.get("args")
            if isinstance(args, dict) and args and len(agg.args_examples) < 3:
                agg.args_examples.append(args)
            continue

        # Begin/end pairing
        if ph == "B":
            if "pid" in e and "tid" in e and "ts" in e:
                be_key = (e.get("pid"), e.get("tid"), str(name), str(cat))
                open_be[be_key] = float(e.get("ts"))
        elif ph == "E":
            if "pid" in e and "tid" in e and "ts" in e:
                be_key = (e.get("pid"), e.get("tid"), str(name), str(cat))
                start = open_be.pop(be_key, None)
                if start is None:
                    continue
                dur_us = float(e.get("ts")) - start
                if dur_us < 0 or not name or not cat:
                    continue
                key = (str(name), str(cat))
                agg = aggs.setdefault(key, _Agg(name=str(name), cat=str(cat)))
                agg.durs.append(dur_us)
                agg.pids.add(e.get("pid"))
                agg.tids.add(e.get("tid"))
                args = e.get("args")
                if isinstance(args, dict) and args and len(agg.args_examples) < 3:
                    agg.args_examples.append(args)

    by_name_cat: List[Dict[str, Any]] = []
    for agg in aggs.values():
        if not agg.durs:
            continue
        finalized = _finalize_agg(agg)
        if finalized:
            by_name_cat.append(finalized)
    by_name_cat.sort(key=lambda r: r.get("total_us", 0), reverse=True)

    return {
        "source": path.name,
        "event_count": len(events),
        "has_cpu_stack": _has_cpu_stack(events),
        "cuda_graph_detected": _looks_like_cuda_graph(events),
        "cats_present": sorted(cats_present),
        "by_name_cat": by_name_cat,
        "flow_events": flow_events[:200],  # cap; agents don't need millions
    }


def write_summary(path: Path, out_path: Path) -> Dict[str, Any]:
    """Parse ``path`` and write the summary to ``out_path``. Returns the
    summary dict as well so callers can use it in-process."""
    summary = parse_trace(path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
