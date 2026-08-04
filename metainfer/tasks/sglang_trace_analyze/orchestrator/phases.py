"""Phase graph for sglang_trace_analyze.

Linear pipeline: MAPPING -> BENCHMARK -> ANALYZE -> HINTS -> SUMMARIZE -> done.

The WebUI state-graph endpoint reads ``terminal_phases`` and
``graph_payload()`` from this module.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# Ordered list of phases in the pipeline.
PHASES: List[str] = ["mapping", "benchmark", "analyze", "hints", "summarize"]

# Phases that signal the task is done (whichever is current at exit).
TERMINAL: set[str] = {"done", "failed"}


def terminal_phases() -> set[str]:
    return TERMINAL


def next_phase(current: str) -> str:
    """Linear advance. Returns "done" at the end."""
    try:
        idx = PHASES.index(current)
        if idx + 1 < len(PHASES):
            return PHASES[idx + 1]
        return "done"
    except ValueError:
        return "done"


def graph_payload(
    current: str = "idle",
    last_outcome: Optional[str] = None,
    last_label: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a mermaid-friendly description of the phase graph."""
    nodes = []
    edges = []
    for i, p in enumerate(PHASES):
        nodes.append({"id": p, "label": p.upper()})
        if i > 0:
            edges.append({"from": PHASES[i - 1], "to": p})
    edges.append({"from": PHASES[-1], "to": "done"})
    nodes.append({"id": "done", "label": "DONE"})
    return {
        "nodes": nodes,
        "edges": edges,
        "current": current,
        "last_outcome": last_outcome,
        "last_transition_label": last_label,
    }
