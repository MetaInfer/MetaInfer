"""Phase / Outcome / Transition definitions for the knowledge-evolution
state machine.

Flow::

    A_attempt_pure -- pass -> DONE (knowledge base already sufficient)
         |
         +-- fail -> B_enrich -- pass -> C_consolidate -> D_verify_final
                        |                                    |
                        +-- fail -> retry (*3) or HALT    +-- pass -> DONE
                                                         +-- fail -> retry in place (*N)
                                                                     +-- exhausted -> B_enrich

Phases:
  A_attempt_pure   - generate inference framework from notebooks/ only (no open source).
  B_enrich         - explore open-source code, supplement knowledge, re-generate.
  C_consolidate    - write validated knowledge back into notebooks/.
  D_verify_final   - re-generate WITHOUT open source using updated notebooks/.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

# ---- Phase & Outcome types ----

Phase = Literal["idle", "A_attempt_pure", "B_enrich", "C_consolidate", "D_verify_final", "finished"]

Outcome = Literal["ok", "logic_fail", "infra_fail", "aborted"]

# ---- Outcome constants ----

OK: Outcome = "ok"
LOGIC_FAIL: Outcome = "logic_fail"
INFRA_FAIL: Outcome = "infra_fail"
ABORTED: Outcome = "aborted"

ALL_OUTCOMES: List[Outcome] = [OK, LOGIC_FAIL, INFRA_FAIL, ABORTED]


# ---- PhaseMeta ----

@dataclass(frozen=True)
class PhaseMeta:
    """Metadata for one phase in the state machine."""

    id: Phase
    label: str
    description: str = ""
    is_terminal: bool = False


# ---- Transition ----

@dataclass(frozen=True)
class Transition:
    """A directed edge in the state machine."""

    from_phase: Phase
    on: Outcome
    to_phase: Phase
    label: str = ""
    carry_failure: bool = True
    carry_perf: bool = False
    consume_iteration: bool = True


# ---- Phase definitions ----

PHASES: List[PhaseMeta] = [
    PhaseMeta(id="idle", label="idle", description="not started"),
    PhaseMeta(
        id="A_attempt_pure",
        label="A: Pure KB",
        description="generate framework from notebooks/ only, no open-source code",
    ),
    PhaseMeta(
        id="B_enrich",
        label="B: Enrich",
        description="explore open-source code, supplement knowledge, re-generate",
    ),
    PhaseMeta(
        id="C_consolidate",
        label="C: Consolidate",
        description="write validated knowledge into notebooks/",
    ),
    PhaseMeta(
        id="D_verify_final",
        label="D: Verify",
        description="re-generate WITHOUT open source using updated notebooks/",
    ),
    PhaseMeta(
        id="finished",
        label="finished",
        description="evolution complete or halted",
        is_terminal=True,
    ),
]

PHASE_ORDER: List[Phase] = ["A_attempt_pure", "B_enrich", "C_consolidate", "D_verify_final"]


# ---- Transitions ----

TRANSITIONS: Dict[Tuple[Phase, Outcome], Transition] = {
    # A_attempt_pure
    ("A_attempt_pure", OK): Transition(
        from_phase="A_attempt_pure",
        on=OK,
        to_phase="finished",
        label="KB sufficient -> done",
        carry_failure=False,
        consume_iteration=True,
    ),
    ("A_attempt_pure", LOGIC_FAIL): Transition(
        from_phase="A_attempt_pure",
        on=LOGIC_FAIL,
        to_phase="B_enrich",
        label="KB insufficient -> enrich",
        carry_failure=True,
        consume_iteration=False,
    ),
    ("A_attempt_pure", INFRA_FAIL): Transition(
        from_phase="A_attempt_pure",
        on=INFRA_FAIL,
        to_phase="A_attempt_pure",
        label="retry",
        carry_failure=False,
        consume_iteration=False,
    ),
    # B_enrich
    ("B_enrich", OK): Transition(
        from_phase="B_enrich",
        on=OK,
        to_phase="C_consolidate",
        label="enrich ok -> consolidate",
        carry_failure=False,
        consume_iteration=False,
    ),
    ("B_enrich", LOGIC_FAIL): Transition(
        from_phase="B_enrich",
        on=LOGIC_FAIL,
        to_phase="B_enrich",
        label="retry enrich",
        carry_failure=True,
        consume_iteration=False,
    ),
    ("B_enrich", INFRA_FAIL): Transition(
        from_phase="B_enrich",
        on=INFRA_FAIL,
        to_phase="B_enrich",
        label="retry",
        carry_failure=False,
        consume_iteration=False,
    ),
    # C_consolidate
    ("C_consolidate", OK): Transition(
        from_phase="C_consolidate",
        on=OK,
        to_phase="D_verify_final",
        label="consolidated -> verify",
        carry_failure=False,
        consume_iteration=False,
    ),
    ("C_consolidate", LOGIC_FAIL): Transition(
        from_phase="C_consolidate",
        on=LOGIC_FAIL,
        to_phase="C_consolidate",
        label="retry consolidate",
        carry_failure=True,
        consume_iteration=False,
    ),
    ("C_consolidate", INFRA_FAIL): Transition(
        from_phase="C_consolidate",
        on=INFRA_FAIL,
        to_phase="C_consolidate",
        label="retry",
        carry_failure=False,
        consume_iteration=False,
    ),
    # D_verify_final
    ("D_verify_final", OK): Transition(
        from_phase="D_verify_final",
        on=OK,
        to_phase="finished",
        label="verified without open source -> done",
        carry_failure=False,
        consume_iteration=True,
    ),
    ("D_verify_final", LOGIC_FAIL): Transition(
        from_phase="D_verify_final",
        on=LOGIC_FAIL,
        to_phase="D_verify_final",
        label="re-verify failed -> retry in place",
        carry_failure=True,
        consume_iteration=False,
    ),
    ("D_verify_final", INFRA_FAIL): Transition(
        from_phase="D_verify_final",
        on=INFRA_FAIL,
        to_phase="D_verify_final",
        label="retry",
        carry_failure=False,
        consume_iteration=False,
    ),
}


# ---- Helpers ----

def next_transition(from_phase: Phase, outcome: Outcome) -> Optional[Transition]:
    """Look up the transition for a given phase and outcome."""
    return TRANSITIONS.get((from_phase, outcome))


def phase_label(p: Phase) -> str:
    """Return the human-readable label for a phase."""
    for pm in PHASES:
        if pm.id == p:
            return pm.label
    return str(p)


def phase_meta(p: Phase) -> Optional[PhaseMeta]:
    """Return PhaseMeta for a phase, or None."""
    for pm in PHASES:
        if pm.id == p:
            return pm
    return None


def is_terminal(p: Phase) -> bool:
    """Check whether a phase is terminal."""
    pm = phase_meta(p)
    return bool(pm.is_terminal) if pm else False


def nodes_for_graph() -> List[Dict[str, str]]:
    """Build node list for the WebUI state graph."""
    nodes: List[Dict[str, str]] = []
    for pm in PHASES:
        if pm.id == "idle":
            continue
        nodes.append({
            "id": pm.id,
            "label": pm.label,
            "description": pm.description,
        })
    return nodes


def edges_for_graph() -> List[Dict[str, str]]:
    """Build edge list for the WebUI state graph.

    Merges duplicate edges (same from/to) by joining labels with " / ".
    Adds a synthetic edge from D_verify_final to B_enrich representing
    retries-exhausted fallback.
    """
    seen: Dict[Tuple[str, str], str] = {}
    for (from_p, outcome), t in TRANSITIONS.items():
        if outcome == ABORTED:
            continue
        key = (t.from_phase, t.to_phase)
        if key in seen:
            seen[key] = seen[key] + " / " + t.label
        else:
            seen[key] = t.label

    edges: List[Dict[str, str]] = []
    for (frm, to), label in seen.items():
        edges.append({"from": frm, "to": to, "label": label})

    # Synthetic edge: retries exhausted -> re-enrich
    edges.append({
        "from": "D_verify_final",
        "to": "B_enrich",
        "label": "retries exhausted -> re-enrich",
    })

    return edges


def outcome_label(o: Outcome) -> str:
    """Return a human-readable label for an outcome."""
    _labels: Dict[Outcome, str] = {
        OK: "ok",
        LOGIC_FAIL: "logic fail",
        INFRA_FAIL: "infra fail",
        ABORTED: "aborted",
    }
    return _labels.get(o, str(o))


# ---- graph_payload (WebUI API) ----

def graph_payload(
    current: str,
    last_outcome: str = "",
    last_label: str = "",
) -> dict:
    """Build the full state-graph payload for the WebUI.

    Called by ``_state_readers.read_state_graph()``.
    """
    nodes = nodes_for_graph()
    edges = edges_for_graph()

    # Determine active edge from last_outcome + last_label
    active_edge = None
    if last_outcome and last_label:
        for edge in edges:
            if last_label in edge.get("label", ""):
                active_edge = edge
                break
        if active_edge is None:
            # Try exact label match from TRANSITIONS
            for (from_p, outcome), t in TRANSITIONS.items():
                if outcome == last_outcome and t.label == last_label:
                    active_edge = {
                        "from": t.from_phase,
                        "to": t.to_phase,
                        "label": t.label,
                    }
                    break

    return {
        "current": current,
        "nodes": nodes,
        "edges": edges,
        "active_edge": active_edge,
        "last_outcome": last_outcome,
        "terminal_nodes": ["finished"],
        "outcome_legend": {o: outcome_label(o) for o in ALL_OUTCOMES},
    }
