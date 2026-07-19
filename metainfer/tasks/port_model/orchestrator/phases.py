"""Phase / Outcome / Transition definitions for port-model.

Five-phase linear pipeline with a self-loop at P5_test (when a test fails,
the implementer repairs and we re-test; capped at 3 retries).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

Phase = Literal[
    "idle",
    "P1_model_analysis",
    "P2_source_analysis",
    "P3_target_analysis",
    "P4_implement",
    "P5_test",
    "finished",
]

Outcome = Literal[
    "ok",
    "logic_fail",
    "infra_fail",
    "test_fail",   # P5: model loaded but results don't match reference
    "aborted",
]

OK = "ok"
LOGIC_FAIL = "logic_fail"
INFRA_FAIL = "infra_fail"
TEST_FAIL = "test_fail"
ABORTED = "aborted"

ALL_OUTCOMES: List[Outcome] = [OK, LOGIC_FAIL, INFRA_FAIL, TEST_FAIL, ABORTED]


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
    PhaseMeta("P1_model_analysis", "1: Model analysis",
              "Agent reads model config + weights to extract architecture, "
              "quantization config, special layers (MoE, VLM encoder, …)."),
    PhaseMeta("P2_source_analysis", "2: Source framework analysis",
              "Agent studies how the model is already registered in the "
              "source (reference) framework — entry points, custom layers, "
              "weight loading, forward call chain."),
    PhaseMeta("P3_target_analysis", "3: Target framework analysis",
              "Agent studies registration patterns in the target framework: "
              "which files to change, which existing model to use as a "
              "template, how custom ops are wired."),
    PhaseMeta("P4_implement", "4: Implement",
              "Agent writes the model adapter code into the target "
              "framework directory — registration, custom layers, weight "
              "mapping, import smoke test."),
    PhaseMeta("P5_test", "5: Test",
              "Deterministic boot of both source and target frameworks; "
              "collect outputs; LLM-judge comparison; repair loop on fail."),
    PhaseMeta("finished", "finished", "run ended", is_terminal=True),
]

TRANSITIONS: Dict[Tuple[Phase, Outcome], Transition] = {
    # Linear forward path.
    ("P1_model_analysis", OK): Transition(
        "P1_model_analysis", OK, "P2_source_analysis", label="ok"),
    ("P2_source_analysis", OK): Transition(
        "P2_source_analysis", OK, "P3_target_analysis", label="ok"),
    ("P3_target_analysis", OK): Transition(
        "P3_target_analysis", OK, "P4_implement", label="ok"),
    ("P4_implement", OK): Transition(
        "P4_implement", OK, "P5_test", label="ok"),

    # P5: pass → done, fail → back to implement (repair).
    ("P5_test", OK): Transition(
        "P5_test", OK, "finished", label="done"),
    ("P5_test", TEST_FAIL): Transition(
        "P5_test", TEST_FAIL, "P4_implement", label="repair"),
    ("P5_test", INFRA_FAIL): Transition(
        "P5_test", INFRA_FAIL, "P4_implement", label="repair"),

    # Logic failures on analysis phases: stop (don't auto-retry analysis).
    ("P1_model_analysis", LOGIC_FAIL): Transition(
        "P1_model_analysis", LOGIC_FAIL, "finished", label="fail"),
    ("P2_source_analysis", LOGIC_FAIL): Transition(
        "P2_source_analysis", LOGIC_FAIL, "finished", label="fail"),
    ("P3_target_analysis", LOGIC_FAIL): Transition(
        "P3_target_analysis", LOGIC_FAIL, "finished", label="fail"),
    ("P4_implement", LOGIC_FAIL): Transition(
        "P4_implement", LOGIC_FAIL, "finished", label="fail"),
    ("P5_test", LOGIC_FAIL): Transition(
        "P5_test", LOGIC_FAIL, "finished", label="fail"),

    # Infra retry on analysis phases.
    ("P1_model_analysis", INFRA_FAIL): Transition(
        "P1_model_analysis", INFRA_FAIL, "P1_model_analysis", label="retry"),
    ("P2_source_analysis", INFRA_FAIL): Transition(
        "P2_source_analysis", INFRA_FAIL, "P2_source_analysis", label="retry"),
    ("P3_target_analysis", INFRA_FAIL): Transition(
        "P3_target_analysis", INFRA_FAIL, "P3_target_analysis", label="retry"),
    ("P4_implement", INFRA_FAIL): Transition(
        "P4_implement", INFRA_FAIL, "P4_implement", label="retry"),
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
        for m in PHASES if m.id not in ("idle", "finished")
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
        TEST_FAIL: "test fail",
        ABORTED: "aborted",
    }.get(o, str(o))


def graph_payload(current, last_outcome, last_label) -> Dict[str, Any]:
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
