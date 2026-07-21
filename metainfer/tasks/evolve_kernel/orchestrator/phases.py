"""Phase / Outcome / Transition definitions for evolve-kernel .

8-phase state machine for LLM-guided iterative kernel optimization:

  Phases 1-4 (bootstrap — runs once):
    A: Generate Correctness Harness
    B: Adversarial Review of Correctness Harness
    C: Generate Performance Harness
    D: Adversarial Review of Performance Harness

  Phases 5-8 (optimization loop):
    E: Select Kernel from Library
    F: Optimize Selected Kernel
    G: Verify Correctness (run harness)
    H: Measure Performance + Complexity → Update Library
    → Loop back to E
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple


Phase = Literal[
    "idle",
    "A_gen_correctness_harness",
    "B_review_correctness_harness",
    "C_gen_perf_harness",
    "D_review_perf_harness",
    "E_select_kernel",
    "F_optimize",
    "G_verify_correctness",
    "H_measure_perf",
    "finished",
]

Outcome = Literal[
    "ok",
    "logic_fail",
    "infra_fail",
    "aborted",
]

OK = "ok"
LOGIC_FAIL = "logic_fail"
INFRA_FAIL = "infra_fail"
ABORTED = "aborted"

ALL_OUTCOMES: List[Outcome] = [OK, LOGIC_FAIL, INFRA_FAIL, ABORTED]


# --------------------------------------------------------------------------- #
# Phase metadata
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PhaseMeta:
    id: Phase
    label: str
    description: str = ""
    is_terminal: bool = False


PHASES: List[PhaseMeta] = [
    PhaseMeta("idle", "idle", "not started"),
    PhaseMeta(
        "A_gen_correctness_harness",
        "A: Gen Correctness Harness",
        "Agent generates a correctness test harness that covers all kernel code paths",
    ),
    PhaseMeta(
        "B_review_correctness_harness",
        "B: Review Correctness Harness",
        "Adversarial review — is the harness complete? Edge cases covered?",
    ),
    PhaseMeta(
        "C_gen_perf_harness",
        "C: Gen Perf Harness",
        "Agent generates a performance measurement harness with interleaved timing",
    ),
    PhaseMeta(
        "D_review_perf_harness",
        "D: Review Perf Harness",
        "Adversarial review — is timing methodology sound? No confounding factors?",
    ),
    PhaseMeta(
        "E_select_kernel",
        "E: Select Kernel",
        "Weighted-random selection from kernel library by exec_time + complexity",
    ),
    PhaseMeta(
        "F_optimize",
        "F: Optimize",
        "Agent optimizes the selected kernel for better GPU performance",
    ),
    PhaseMeta(
        "G_verify_correctness",
        "G: Verify Correctness",
        "Run correctness harness — compare optimized vs original kernel output",
    ),
    PhaseMeta(
        "H_measure_perf",
        "H: Measure Perf + Update Library",
        "Run perf harness, evaluate complexity, update kernel library rankings",
    ),
    PhaseMeta("finished", "finished", "run ended", is_terminal=True),
]

PHASE_ORDER: List[Phase] = [
    "A_gen_correctness_harness",
    "B_review_correctness_harness",
    "C_gen_perf_harness",
    "D_review_perf_harness",
    "E_select_kernel",
    "F_optimize",
    "G_verify_correctness",
    "H_measure_perf",
]


# --------------------------------------------------------------------------- #
# Transitions
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Transition:
    from_phase: Phase
    on: Outcome
    to_phase: Phase
    label: str = ""
    carry_failure: bool = True
    carry_perf: bool = False
    consume_iteration: bool = True


# Shorthand constants to keep the table readable
A = "A_gen_correctness_harness"
B = "B_review_correctness_harness"
C = "C_gen_perf_harness"
D = "D_review_perf_harness"
E = "E_select_kernel"
F = "F_optimize"
G = "G_verify_correctness"
H = "H_measure_perf"


TRANSITIONS: Dict[Tuple[Phase, Outcome], Transition] = {
    # ---- Phase A: Generate Correctness Harness ----
    (A, OK): Transition(A, OK, B,
        label="harness generated", carry_failure=False, consume_iteration=False),
    (A, LOGIC_FAIL): Transition(A, LOGIC_FAIL, A,
        label="retry", carry_failure=False, consume_iteration=False),
    (A, INFRA_FAIL): Transition(A, INFRA_FAIL, A,
        label="infra retry", carry_failure=False, consume_iteration=False),

    # ---- Phase B: Review Correctness Harness ----
    # Pass → move on to perf harness generation
    (B, OK): Transition(B, OK, C,
        label="harness approved", carry_failure=False, consume_iteration=False),
    # Fail → send feedback back to A to regenerate
    (B, LOGIC_FAIL): Transition(B, LOGIC_FAIL, A,
        label="harness rejected → regen", carry_failure=True, consume_iteration=False),
    (B, INFRA_FAIL): Transition(B, INFRA_FAIL, B,
        label="infra retry", carry_failure=False, consume_iteration=False),

    # ---- Phase C: Generate Performance Harness ----
    (C, OK): Transition(C, OK, D,
        label="perf harness generated", carry_failure=False, consume_iteration=False),
    (C, LOGIC_FAIL): Transition(C, LOGIC_FAIL, C,
        label="retry", carry_failure=False, consume_iteration=False),
    (C, INFRA_FAIL): Transition(C, INFRA_FAIL, C,
        label="infra retry", carry_failure=False, consume_iteration=False),

    # ---- Phase D: Review Performance Harness ----
    (D, OK): Transition(D, OK, E,
        label="perf harness approved", carry_failure=False, consume_iteration=False),
    (D, LOGIC_FAIL): Transition(D, LOGIC_FAIL, C,
        label="perf harness rejected → regen", carry_failure=True, consume_iteration=False),
    (D, INFRA_FAIL): Transition(D, INFRA_FAIL, D,
        label="infra retry", carry_failure=False, consume_iteration=False),

    # ---- Phase E: Select Kernel ----
    (E, OK): Transition(E, OK, F,
        label="kernel selected", carry_failure=False, consume_iteration=False),
    (E, LOGIC_FAIL): Transition(E, LOGIC_FAIL, E,
        label="retry select", carry_failure=False, consume_iteration=False),

    # ---- Phase F: Optimize ----
    (F, OK): Transition(F, OK, G,
        label="optimization done", carry_failure=False, consume_iteration=False),
    (F, LOGIC_FAIL): Transition(F, LOGIC_FAIL, F,
        label="retry optimize", carry_failure=False, consume_iteration=False),
    (F, INFRA_FAIL): Transition(F, INFRA_FAIL, F,
        label="infra retry", carry_failure=False, consume_iteration=False),

    # ---- Phase G: Verify Correctness ----
    (G, OK): Transition(G, OK, H,
        label="correctness passed", carry_failure=False, consume_iteration=False),
    (G, LOGIC_FAIL): Transition(G, LOGIC_FAIL, F,
        label="correctness failed → re-optimize", carry_failure=True, consume_iteration=False),

    # ---- Phase H: Measure Perf + Update Library ----
    (H, OK): Transition(H, OK, E,
        label="kernel stored → next", carry_failure=False, consume_iteration=True),
    (H, LOGIC_FAIL): Transition(H, LOGIC_FAIL, E,
        label="perf fail → next kernel", carry_failure=False, consume_iteration=True),
    (H, INFRA_FAIL): Transition(H, INFRA_FAIL, H,
        label="infra retry", carry_failure=False, consume_iteration=False),
}


# --------------------------------------------------------------------------- #
# Public helpers
# --------------------------------------------------------------------------- #


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


def outcome_label(o: Outcome) -> str:
    return {
        OK: "ok",
        LOGIC_FAIL: "logic fail",
        INFRA_FAIL: "infra fail",
        ABORTED: "aborted",
    }.get(o, str(o))


# --------------------------------------------------------------------------- #
# Frontend graph payload
# --------------------------------------------------------------------------- #


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


def graph_payload(current: Phase, last_outcome: Optional[str],
                  last_label: Optional[str]) -> Dict[str, Any]:
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
    return {
        "current": current,
        "nodes": nodes,
        "edges": edges,
        "active_edge": active_edge,
        "last_outcome": last_outcome,
        "terminal_nodes": terminal_nodes,
        "outcome_legend": [{"id": o, "label": outcome_label(o)} for o in ALL_OUTCOMES],
    }


# --------------------------------------------------------------------------- #
# Phase groups (for bootstrap vs optimization loop)
# --------------------------------------------------------------------------- #

BOOTSTRAP_PHASES: set = {A, B, C, D}
OPTIMIZATION_PHASES: set = {E, F, G, H}

def is_bootstrap(p: Phase) -> bool:
    return p in BOOTSTRAP_PHASES

def is_optimization(p: Phase) -> bool:
    return p in OPTIMIZATION_PHASES
