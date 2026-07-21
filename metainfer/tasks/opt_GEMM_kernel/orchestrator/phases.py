"""State machine for the independent GEMM kernel optimization loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple


Phase = Literal[
    "idle",
    "S_baseline",
    "A_plan",
    "B_implement",
    "C_test",
    "D_review",
    "E_perf_test",
    "F_perf_plan",
    "finished",
]
Outcome = Literal["ok", "logic_fail", "infra_fail", "perf_regression", "aborted"]

OK: Outcome = "ok"
LOGIC_FAIL: Outcome = "logic_fail"
INFRA_FAIL: Outcome = "infra_fail"
PERF_REGRESSION: Outcome = "perf_regression"
ABORTED: Outcome = "aborted"


@dataclass(frozen=True)
class PhaseMeta:
    id: Phase
    label: str
    description: str = ""
    is_terminal: bool = False


PHASES: List[PhaseMeta] = [
    PhaseMeta("idle", "idle", "not started"),
    # S is a one-time preflight status, not one of the six iteration phases.
    PhaseMeta("S_baseline", "Baseline", "one-time system build, correctness, and benchmark certification"),
    PhaseMeta("A_plan", "A: Plan", "agent proposes one measurable GEMM optimization"),
    PhaseMeta("B_implement", "B: Implement", "agent edits submission/ only"),
    PhaseMeta("C_test", "C: Correctness Test", "system build followed by the frozen harness correctness gate"),
    PhaseMeta("D_review", "D: Review + Retro", "review compile/correctness evidence before performance or replanning"),
    PhaseMeta("E_perf_test", "E: Perf Test", "frozen harness multi-shape benchmark and champion decision"),
    PhaseMeta("F_perf_plan", "F: Perf Plan", "analyze performance evidence and prepare the next iteration"),
    PhaseMeta("finished", "finished", "iteration budget exhausted or interrupted", True),
]
PHASE_ORDER: List[Phase] = [
    "A_plan", "B_implement", "C_test", "D_review", "E_perf_test", "F_perf_plan",
]


def is_terminal(phase: Phase) -> bool:
    return phase == "finished"


def edges_for_graph() -> List[Dict[str, str]]:
    return [
        {"from": "A_plan", "to": "B_implement", "outcome": "ok", "label": "plan ready"},
        {"from": "B_implement", "to": "C_test", "outcome": "ok", "label": "candidate ready"},
        {"from": "C_test", "to": "D_review", "outcome": "ok", "label": "correct"},
        {"from": "C_test", "to": "D_review", "outcome": "logic_fail", "label": "compile/correctness fail"},
        {"from": "D_review", "to": "E_perf_test", "outcome": "ok", "label": "C passed"},
        {"from": "D_review", "to": "A_plan", "outcome": "logic_fail", "label": "C failed; replan"},
        {"from": "E_perf_test", "to": "F_perf_plan", "outcome": "ok", "label": "promoted"},
        {"from": "E_perf_test", "to": "F_perf_plan", "outcome": "perf_regression", "label": "not promoted"},
        {"from": "F_perf_plan", "to": "A_plan", "outcome": "ok", "label": "next iteration"},
    ]


def graph_payload(
    current: str,
    last_outcome: Optional[str] = None,
    last_label: Optional[str] = None,
) -> Dict[str, object]:
    nodes = [
        {
            "id": item.id,
            "label": item.label,
            "description": item.description,
            "active": item.id == current,
            "terminal": item.is_terminal,
        }
        for item in PHASES
        if item.id in PHASE_ORDER
    ]
    return {
        "nodes": nodes,
        "edges": edges_for_graph(),
        "order": list(PHASE_ORDER),
        "current": current,
        "last_outcome": last_outcome,
        "last_label": last_label,
        "terminal": current == "finished",
    }
