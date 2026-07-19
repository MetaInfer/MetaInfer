"""Phase / Outcome / Transition definitions for find-low-hanging-kernel.

Five-phase linear pipeline with one self-loop at P3_graph_validate (when a
validation round applies fixes, the loop re-runs). Validation rounds are
also bounded by ``max_validator_rounds`` in the orchestrator config, so the
self-loop cannot spin forever.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple


Phase = Literal[
    "idle",
    "P1_code_analysis",
    "P2_tracing_analysis",
    "P3_graph_build",
    "P3_graph_validate",
    "P4_visualize",
    "finished",
]

Outcome = Literal[
    "ok",
    "logic_fail",
    "infra_fail",
    "needs_fix",   # graph validator found issues and applied patches
    "clean",       # graph validator found no issues
    "aborted",
]

OK = "ok"
LOGIC_FAIL = "logic_fail"
INFRA_FAIL = "infra_fail"
NEEDS_FIX = "needs_fix"
CLEAN = "clean"
ABORTED = "aborted"

ALL_OUTCOMES: List[Outcome] = [OK, LOGIC_FAIL, INFRA_FAIL, NEEDS_FIX, CLEAN, ABORTED]


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
    consume_iteration: bool = False


PHASES: List[PhaseMeta] = [
    PhaseMeta("idle", "idle", "not started"),
    PhaseMeta("P1_code_analysis", "1: Code + quant analysis",
              "3 independent agents trace architecture, quantization loading, "
              "and runtime-resolved code paths; synthesizer cross-validates."),
    PhaseMeta("P2_tracing_analysis", "2: Tracing analysis",
              "Deterministic chrome-trace parser produces stats; 3 agents "
              "cross-validate kernel↔source mapping and shape disambiguation."),
    PhaseMeta("P3_graph_build", "3a: Build flow graph",
              "Fresh agent reads step-1+2 memory and emits flow_graph.json."),
    PhaseMeta("P3_graph_validate", "3b: Validate flow graph",
              "Deterministic driver: integrity check + 5-worker pool validates "
              "3-node groups against memory + framework source. Loops on fixes."),
    PhaseMeta("P4_visualize", "4: Render visualization",
              "Substitute validated graph into the ELK + SVG HTML template."),
    PhaseMeta("finished", "finished", "run ended", is_terminal=True),
]

PHASE_ORDER: List[Phase] = [
    "P1_code_analysis",
    "P2_tracing_analysis",
    "P3_graph_build",
    "P3_graph_validate",
    "P4_visualize",
]

TRANSITIONS: Dict[Tuple[Phase, Outcome], Transition] = {
    # Linear forward path.
    ("P1_code_analysis", OK): Transition(
        "P1_code_analysis", OK, "P2_tracing_analysis", label="ok"),
    ("P2_tracing_analysis", OK): Transition(
        "P2_tracing_analysis", OK, "P3_graph_build", label="ok"),
    ("P3_graph_build", OK): Transition(
        "P3_graph_build", OK, "P3_graph_validate", label="ok"),

    # Step 3 validation: loop on needs_fix, advance on clean.
    ("P3_graph_validate", NEEDS_FIX): Transition(
        "P3_graph_validate", NEEDS_FIX, "P3_graph_validate",
        label="fix → revalidate", consume_iteration=True),
    ("P3_graph_validate", CLEAN): Transition(
        "P3_graph_validate", CLEAN, "P4_visualize", label="clean"),

    ("P4_visualize", OK): Transition(
        "P4_visualize", OK, "finished", label="done"),

    # Retry-in-place on infra failures (deterministic re-run).
    ("P1_code_analysis", INFRA_FAIL): Transition(
        "P1_code_analysis", INFRA_FAIL, "P1_code_analysis", label="retry"),
    ("P2_tracing_analysis", INFRA_FAIL): Transition(
        "P2_tracing_analysis", INFRA_FAIL, "P2_tracing_analysis", label="retry"),
    ("P3_graph_build", INFRA_FAIL): Transition(
        "P3_graph_build", INFRA_FAIL, "P3_graph_build", label="retry"),
    ("P3_graph_validate", INFRA_FAIL): Transition(
        "P3_graph_validate", INFRA_FAIL, "P3_graph_validate", label="retry"),
    ("P4_visualize", INFRA_FAIL): Transition(
        "P4_visualize", INFRA_FAIL, "P4_visualize", label="retry"),

    # Logic failures: stop the run (we don't auto-retry analysis logic).
    ("P1_code_analysis", LOGIC_FAIL): Transition(
        "P1_code_analysis", LOGIC_FAIL, "finished", label="fail"),
    ("P2_tracing_analysis", LOGIC_FAIL): Transition(
        "P2_tracing_analysis", LOGIC_FAIL, "finished", label="fail"),
    ("P3_graph_build", LOGIC_FAIL): Transition(
        "P3_graph_build", LOGIC_FAIL, "finished", label="fail"),
    ("P3_graph_validate", LOGIC_FAIL): Transition(
        "P3_graph_validate", LOGIC_FAIL, "finished", label="fail"),
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
        NEEDS_FIX: "needs fix",
        CLEAN: "clean",
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
