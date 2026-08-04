"""Detect communication-computation overlap gaps in a torch profiler trace.

Scans GPU kernel timeline for gaps between consecutive events where the
GPU is idle. On K100, this is lower priority — the detector is kept
simple.
"""

from __future__ import annotations

from typing import Any, Dict, List


def detect_gaps(
    trace_data: Dict[str, Any],
    *,
    gap_threshold_us: float = 10.0,
) -> List[Dict[str, Any]]:
    """Find GPU-idle gaps in the kernel timeline.

    Args:
        trace_data: Parsed Chrome trace JSON.
        gap_threshold_us: Minimum gap duration (us) to report.

    Returns:
        List of gap dicts with ``gap_id``, ``description``, ``gap_us``,
        ``affected_kernels``, ``severity``.
    """
    trace_events = trace_data.get("traceEvents", [])
    if isinstance(trace_data, list):
        trace_events = trace_data

    # Collect GPU kernel events with their timestamps
    events = []
    for evt in trace_events:
        cat = evt.get("cat", "")
        dur = evt.get("dur", 0)
        ts = evt.get("ts", 0)
        if cat == "kernel" and dur > 0:
            events.append({
                "name": evt.get("name", ""),
                "ts": ts,
                "end": ts + dur,
            })

    events.sort(key=lambda e: e["ts"])

    gaps = []
    gap_id = 0
    for i in range(1, len(events)):
        prev_end = events[i - 1]["end"]
        curr_start = events[i]["ts"]
        gap = curr_start - prev_end
        if gap > gap_threshold_us:
            gap_id += 1
            severity = "low"
            if gap > 100:
                severity = "high"
            elif gap > 50:
                severity = "medium"

            gaps.append({
                "gap_id": gap_id,
                "description": (
                    f"{events[i - 1]['name']} → {events[i]['name']}: "
                    f"{gap:.1f}us idle"
                ),
                "gap_us": round(gap, 1),
                "cumulative_gap_us": 0,  # filled in by caller
                "pct_of_total": 0,       # filled in by caller
                "affected_kernels": [
                    events[i - 1]["name"],
                    events[i]["name"],
                ],
                "severity": severity,
            })

    # Compute cumulative stats
    total_gap = sum(g["gap_us"] for g in gaps)
    total_dur = sum(
        (e["end"] - events[0]["ts"]) for e in events[-1:]
    ) if events else 0

    for g in gaps:
        g["cumulative_gap_us"] = round(total_gap, 1)
        g["pct_of_total"] = round(g["gap_us"] / total_dur * 100, 2) if total_dur > 0 else 0

    return gaps


def build_overlap_report(
    trace_data: Dict[str, Any],
    batch_size: int,
    stage: str,
    *,
    gap_threshold_us: float = 10.0,
) -> Dict[str, Any]:
    """Produce the full overlap.json payload."""
    gaps = detect_gaps(trace_data, gap_threshold_us=gap_threshold_us)
    total_gap = sum(g["gap_us"] for g in gaps)
    total_dur = sum(
        evt.get("dur", 0) for evt in
        (trace_data.get("traceEvents", []) or [])
        if evt.get("cat") == "kernel"
    )

    return {
        "batch_size": batch_size,
        "stage": stage,
        "gaps": gaps,
        "summary": {
            "total_gap_us": round(total_gap, 1),
            "total_gap_pct": round(total_gap / total_dur * 100, 2) if total_dur > 0 else 0,
            "cuda_graph_effective": len(gaps) < 5,
        },
    }
