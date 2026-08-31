"""Small top-level state machine for the multi-worker optimizer."""

from __future__ import annotations

from typing import Any, Dict, List


PREPARE = "prepare"
GENERATE = "generate_kernel_repo"
BASELINE = "baseline"
EXPLORE = "parallel_explore"
SYNTHESIZE = "skill_synthesis"
VALIDATE = "serial_validate"
REPORT = "report"
FINISHED = "finished"

_ORDER = [
    PREPARE, GENERATE, BASELINE, EXPLORE, SYNTHESIZE, VALIDATE, REPORT, FINISHED
]
_LABELS = {
    PREPARE: "Prepare",
    GENERATE: "Generate kernel repo",
    BASELINE: "Baseline",
    EXPLORE: "Parallel explore",
    SYNTHESIZE: "Skill synthesis",
    VALIDATE: "Serial validate",
    REPORT: "Report",
    FINISHED: "Finished",
}


def graph_payload(
    current: str,
    last_outcome: str | None = None,
    last_label: str | None = None,
    *,
    include_baseline: bool = True,
) -> Dict[str, Any]:
    order = (
        _ORDER
        if include_baseline
        else [phase for phase in _ORDER if phase != BASELINE]
    )
    nodes: List[Dict[str, Any]] = [
        {
            "id": phase,
            "label": _LABELS[phase],
            "description": _LABELS[phase],
            "is_terminal": phase == FINISHED,
        }
        for phase in order
    ]
    edges = [
        {
            "from": order[i],
            "to": order[i + 1],
            "label": "ok",
            "outcomes": ["ok"],
        }
        for i in range(len(order) - 1)
    ]
    active_edge = None
    if current in order and current != PREPARE:
        idx = order.index(current)
        active_edge = f"{order[idx - 1]} / {current}"
    return {
        "current": current,
        "nodes": nodes,
        "edges": edges,
        "active_edge": active_edge,
        "last_outcome": last_outcome,
        "last_transition_label": last_label,
        "terminal_nodes": [FINISHED],
        "outcome_legend": [{"id": "ok", "label": "OK"}],
    }
