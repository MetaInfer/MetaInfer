"""Chrome trace JSON parser + kernel aggregation.

Loads a ``torch.profiler`` Chrome trace (``.json`` or ``.json.gz``) and
produces an aggregated kernel table: one row per unique kernel name,
sorted by total GPU duration descending.

In the MAPPING phase this also extracts call-stack information for
structure mapping. In the ANALYZE phase it aggregates CUDA Graph replay
events into per-kernel durations.
"""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def _open_trace(trace_path: Path):
    """Open a trace file — transparently handles .gz compression."""
    if trace_path.suffix == ".gz":
        return gzip.open(trace_path, "rt", encoding="utf-8")
    return open(trace_path, "r", encoding="utf-8")


def parse_trace(trace_path: Path) -> Dict[str, Any]:
    """Load a Chrome trace JSON and return the top-level document.

    Returns:
        Dict with keys: ``traceEvents``, ``displayTimeUnit``, etc.
    """
    with _open_trace(trace_path) as f:
        data = json.load(f)
    return data


def aggregate_kernels(
    trace_data: Dict[str, Any],
    *,
    include_call_stack: bool = False,
) -> List[Dict[str, Any]]:
    """Aggregate GPU kernel events by kernel name.

    Args:
        trace_data: Parsed Chrome trace JSON.
        include_call_stack: If True, preserve ``call_stack`` from the first
            occurrence of each unique kernel name.

    Returns:
        List of kernel dicts sorted by ``total_dur_us`` descending. Each dict:
        ``kernel_name``, ``total_dur_us``, ``count``, ``call_stack`` (optional).
    """
    trace_events = trace_data.get("traceEvents", [])
    if not trace_events:
        # sglang sometimes wraps in a list directly
        if isinstance(trace_data, list):
            trace_events = trace_data
        else:
            return []

    # Filter GPU kernel events
    kernels: Dict[str, Dict[str, Any]] = {}
    for evt in trace_events:
        cat = evt.get("cat", "")
        name = evt.get("name", "")
        dur = evt.get("dur", 0)

        # Torch profiler GPU kernel events: cat="kernel", name like
        # "triton_fused_moe_kernel" or "void at::native::..."
        if cat != "kernel" or dur <= 0:
            continue

        if name not in kernels:
            entry: Dict[str, Any] = {
                "kernel_name": name,
                "total_dur_us": 0,
                "count": 0,
            }
            if include_call_stack:
                args = evt.get("args", {}) or {}
                call_stack = args.get("call stack", "")
                if call_stack:
                    entry["call_stack"] = call_stack
            kernels[name] = entry

        kernels[name]["total_dur_us"] += dur
        kernels[name]["count"] += 1

    # Sort by total duration descending
    result = sorted(
        kernels.values(), key=lambda k: k["total_dur_us"], reverse=True
    )
    return result


def aggregate_kernels_with_dims(
    trace_data: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Like :func:`aggregate_kernels`, but also collects Input Dims from
    ``args["Input Dims"]`` for shape-aware kernels (GEMM, attention).

    This is only meaningful when the trace was captured WITHOUT CUDA Graph
    (i.e. during the MAPPING phase), because CUDA Graph replay hides
    individual kernel dims.
    """
    trace_events = trace_data.get("traceEvents", [])
    if isinstance(trace_data, list):
        trace_events = trace_data

    kernels: Dict[str, Dict[str, Any]] = {}
    for evt in trace_events:
        cat = evt.get("cat", "")
        name = evt.get("name", "")
        dur = evt.get("dur", 0)
        if cat != "kernel" or dur <= 0:
            continue

        if name not in kernels:
            args = evt.get("args", {}) or {}
            entry: Dict[str, Any] = {
                "kernel_name": name,
                "total_dur_us": 0,
                "count": 0,
                "input_dims": [],
                "call_stack": args.get("call stack", ""),
            }
            kernels[name] = entry

        kernels[name]["total_dur_us"] += dur
        kernels[name]["count"] += 1
        args = evt.get("args", {}) or {}
        dims = args.get("Input Dims", [])
        if dims and dims not in kernels[name]["input_dims"]:
            kernels[name]["input_dims"].append(dims)

    return sorted(
        kernels.values(), key=lambda k: k["total_dur_us"], reverse=True
    )
