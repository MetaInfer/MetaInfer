"""Phase / Outcome / Transition definitions for opt-kernel.

Same ABCDEF state machine as gen-infer-framework. The only difference is
C_test uses agent-written test.sh instead of an immutable oracle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple


Phase = Literal[
    "idle",
    "A_plan",
    "B_implement",
    "C_test",
    "D_review",
    "E_perf_test",
    "F_perf_plan",
    "finished",
]

Outcome = Literal[
    "ok",
    "logic_fail",
    "infra_fail",
    "perf_regression",
    "aborted",
]

OK = "ok"
LOGIC_FAIL = "logic_fail"
INFRA_FAIL = "infra_fail"
PERF_REGRESSION = "perf_regression"
ABORTED = "aborted"

ALL_OUTCOMES: List[Outcome] = [OK, LOGIC_FAIL, INFRA_FAIL, PERF_REGRESSION, ABORTED]


@dataclass(frozen=True)
class PhaseMeta:
    id: Phase
    label: str
    description: str = ""
    is_terminal: bool = False


@dataclass(frozen=True)
class Transition:
    from_phase: Phase
    on: Outcome
    to_phase: Phase
    label: str = ""
    carry_failure: bool = True
    carry_perf: bool = False
    consume_iteration: bool = True


PHASES: List[PhaseMeta] = [
    PhaseMeta("idle", "idle", "not started"),
    PhaseMeta("A_plan", "A: Plan", "planner writes plan.md + test_spec.md"),
    PhaseMeta("B_implement", "B: Implement",
              "implementer writes kernel code + test.sh"),
    PhaseMeta("C_test", "C: Correctness Test",
              "run test.sh for correctness"),
    PhaseMeta("D_review", "D: Review + Retro",
              "post-test reviewer writes review.md; advisory only. "
              "Routes to E on C-pass, back to B on C-fail"),
    PhaseMeta("E_perf_test", "E: Perf Test",
              "agent writes + runs perf.sh → perf_report.json"),
    PhaseMeta("F_perf_plan", "F: Perf Plan",
              "agent reads perf_report.json + review.md, writes perf_plan.md"),
    PhaseMeta("finished", "finished",
              "run ended", is_terminal=True),
]

PHASE_ORDER: List[Phase] = [
    "A_plan", "B_implement", "C_test", "D_review", "E_perf_test", "F_perf_plan",
]

TRANSITIONS: Dict[Tuple[Phase, Outcome], Transition] = {
    # intra-iteration forward
    ("A_plan", OK): Transition("A_plan", OK, "B_implement",
                                label="ok", carry_failure=False, consume_iteration=False),
    ("B_implement", OK): Transition("B_implement", OK, "C_test",
                                     label="ok", carry_failure=False, consume_iteration=False),
    ("C_test", OK): Transition("C_test", OK, "D_review",
                                label="pass", carry_failure=False, consume_iteration=False),
    ("C_test", LOGIC_FAIL): Transition("C_test", LOGIC_FAIL, "D_review",
                                        label="fail", carry_failure=False, consume_iteration=False),
    ("C_test", INFRA_FAIL): Transition("C_test", INFRA_FAIL, "D_review",
                                        label="infra", carry_failure=False, consume_iteration=False),
    ("C_test", PERF_REGRESSION): Transition("C_test", PERF_REGRESSION, "D_review",
                                              label="regress", carry_failure=False, consume_iteration=False),
    ("D_review", OK): Transition("D_review", OK, "E_perf_test",
                                  label="C ok → perf", carry_failure=False, consume_iteration=False),
    ("D_review", LOGIC_FAIL): Transition("D_review", LOGIC_FAIL, "B_implement",
                                          label="C fail → redo", consume_iteration=True),
    ("E_perf_test", OK): Transition("E_perf_test", OK, "F_perf_plan",
                                     label="ok", carry_failure=False, carry_perf=True, consume_iteration=False),
    ("F_perf_plan", OK): Transition("F_perf_plan", OK, "A_plan",
                                     label="new iter", carry_failure=False, consume_iteration=True),

    # infra failures: retry in place
    ("A_plan", INFRA_FAIL): Transition("A_plan", INFRA_FAIL, "A_plan",
                                        label="retry", carry_failure=False, consume_iteration=False),
    ("E_perf_test", INFRA_FAIL): Transition("E_perf_test", INFRA_FAIL, "E_perf_test",
                                              label="retry", carry_failure=False, consume_iteration=False),
    ("F_perf_plan", INFRA_FAIL): Transition("F_perf_plan", INFRA_FAIL, "F_perf_plan",
                                              label="retry", carry_failure=False, consume_iteration=False),

    # B_implement failures → new iteration at A_plan
    ("B_implement", INFRA_FAIL): Transition("B_implement", INFRA_FAIL, "A_plan",
                                             label="B fail → replan", carry_failure=True, consume_iteration=True),
    ("B_implement", LOGIC_FAIL): Transition("B_implement", LOGIC_FAIL, "A_plan",
                                             label="B fail → replan", carry_failure=True, consume_iteration=True),

    # logic failures at A/E/F: redo in place
    ("A_plan", LOGIC_FAIL): Transition("A_plan", LOGIC_FAIL, "A_plan",
                                        label="replan", carry_failure=False, consume_iteration=False),
    ("E_perf_test", LOGIC_FAIL): Transition("E_perf_test", LOGIC_FAIL, "E_perf_test",
                                              label="redo", carry_failure=False, consume_iteration=False),
    ("F_perf_plan", LOGIC_FAIL): Transition("F_perf_plan", LOGIC_FAIL, "F_perf_plan",
                                              label="redo", carry_failure=False, consume_iteration=False),
}


def next_transition(from_phase: Phase, outcome: Outcome) -> Optional[Transition]:
    return TRANSITIONS.get((from_phase, outcome))


def phase_label(p: Phase) -> str:
    for m in PHASES:
        if m.id == p:
            return m.label
    return str(p)


def phase_meta(p: Phase) -> Optional[PhaseMeta]:
    for m in PHASES:
        if m.id == p:
            return m
    return None


def is_terminal(p: Phase) -> bool:
    m = phase_meta(p)
    return bool(m and m.is_terminal)


def nodes_for_graph() -> List[Dict[str, str]]:
    return [
        {"id": m.id, "label": m.label, "description": m.description}
        for m in PHASES if m.id in PHASE_ORDER
    ]


def edges_for_graph() -> List[Dict[str, str]]:
    merged: Dict[Tuple[Phase, Phase], List[str]] = {}
    for (frm, _outc), t in TRANSITIONS.items():
        merged.setdefault((frm, t.to_phase), []).append(t.label or _outc)
    out: List[Dict[str, str]] = []
    for (frm, to), labels in merged.items():
        out.append({
            "from": frm,
            "to": to,
            "label": " / ".join(sorted(set(labels))),
        })
    return out


def outcome_label(o: Outcome) -> str:
    return {
        OK: "ok",
        LOGIC_FAIL: "logic fail",
        INFRA_FAIL: "infra fail",
        PERF_REGRESSION: "perf regression",
        ABORTED: "aborted",
    }.get(o, str(o))


def graph_payload(current, last_outcome, last_label) -> Dict[str, Any]:
    """Build the state-graph render payload for the WebUI."""
    nodes = nodes_for_graph()
    edges = edges_for_graph()
    active_edge = None
    if last_label:
        for e in edges:
            if e["to"] == current and last_label in e["label"].split(" / "):
                active_edge = {"from": e["from"], "to": e["to"], "label": last_label}
                break
    terminal_nodes = [
        {"id": m.id, "label": m.label, "description": m.description}
        for m in PHASES if m.is_terminal
    ]
    outcome_legend = [{"id": o, "label": outcome_label(o)} for o in ALL_OUTCOMES]
    return {
        "current": current,
        "nodes": nodes,
        "edges": edges,
        "active_edge": active_edge,
        "last_outcome": last_outcome,
        "terminal_nodes": terminal_nodes,
        "outcome_legend": outcome_legend,
    }
