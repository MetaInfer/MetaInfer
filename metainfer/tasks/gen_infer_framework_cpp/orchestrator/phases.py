"""Phase / Outcome / Transition definitions — the single source of truth for
the orchestrator's state machine.

This module is the **only** place where phase names, their display labels, and
the legal transitions live. The pipeline consults :data:`TRANSITIONS` to decide
"what runs next"; the WebUI's ``/api/state-graph`` reads :data:`PHASES` and
:data:`PHASE_ORDER` plus :func:`edges_for_graph` to render the flow diagram.

**Adding or changing behavior is a one-file edit here:**

1. add the phase to :data:`PHASES` (and :data:`PHASE_ORDER` if it should appear
   in the graph),
2. add the relevant :data:`TRANSITIONS` entries,
3. register a ``_do_<phase>`` handler in :mod:`metainfer.pipeline`.

The WebUI picks up the new nodes/edges automatically — no frontend edit
required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple


# Type aliases (kept as Literals so IDEs / mypy catch typos).
# NOTE: there is deliberately NO ``"failed"`` phase. The orchestrator never
# gives up mid-run — every failure path either retries in place, consumes
# the iteration and starts a fresh one, or routes back to A_plan. The only
# terminal phase is ``"finished"`` (success / stopped at iteration cap /
# externally interrupted). Iteration-level records still carry a per-iter
# ``status`` field where ``"failed"`` records "this attempt didn't succeed";
# that's a historical marker, not a system state.
Phase = Literal[
    "idle",
    "A_plan",
    "B_implement",
    "C_test",
    "D_review",
    "E_perf_test",
    "G_perf_review",
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


# Runtime constants — useful where you need a value rather than a type.
OK              : Outcome = "ok"
LOGIC_FAIL      : Outcome = "logic_fail"
INFRA_FAIL      : Outcome = "infra_fail"
PERF_REGRESSION : Outcome = "perf_regression"
ABORTED         : Outcome = "aborted"

ALL_OUTCOMES: List[Outcome] = [OK, LOGIC_FAIL, INFRA_FAIL, PERF_REGRESSION, ABORTED]


@dataclass(frozen=True)
class PhaseMeta:
    """Display metadata for one phase."""

    id: Phase
    label: str
    description: str = ""
    is_terminal: bool = False


@dataclass(frozen=True)
class Transition:
    """One edge of the state machine.

    Attributes
    ----------
    carry_failure
        If True, propagate ``ctx.failure`` to the next phase. If False and the
        outcome was OK, the failure is cleared.
    carry_perf
        If True, update ``ctx.last_perf`` from this step's measured perf
        (typically only on C-pass).
    consume_iteration
        If True, close the current iteration folder and open a fresh one for
        the next phase. If False, the next phase runs in the same folder
        (used for in-place infra retries and intra-iteration forward steps
        like A→B→C→D→E→F).
    """

    from_phase: Phase
    on: Outcome
    to_phase: Phase
    label: str = ""
    carry_failure: bool = True
    carry_perf: bool = False
    consume_iteration: bool = True


# --------------------------------------------------------------------------- #
# Phases — canonical list + display order
# --------------------------------------------------------------------------- #
#
# Flow:
#   A → B → C → D correctness review ──┬─ C ok → E perf → G perf review → F plan → A
#                                       └─ C fail → A (new iter, failure carried)
#   B_implement fail → A_plan (new iter, replan with failure carried forward)
#
# D_review ALWAYS runs after C (regardless of C outcome). Its egress routing
# (→ E vs → A) is encoded in D's outcome, which the orchestrator derives from
# C's outcome (see _do_review in pipeline.py). E and F only run on the happy
# path; if C failed, D closes the iteration and requests a fresh plan.
#
# B_implement failures (either INFRA or LOGIC) do NOT retry in place —
# they consume the iteration and route back to A_plan for a fresh plan.
# Rationale: SubAgentManager already retries the agent internally; piling
# pipeline-level in-place retries on top traps the agent in an ever-deeper
# --resume context that fixates on the same failing path. A fresh iteration
# is the cleanest escape (see the B-fail block in TRANSITIONS below).
#
PHASES: List[PhaseMeta] = [
    PhaseMeta("idle",        "idle",     "not started"),
    PhaseMeta("A_plan",      "A: Plan + Gate",
              "planner writes full architecture + minimum E2E plan; "
              "deterministic gate must pass"),
    PhaseMeta("B_implement", "B: Implement",
              "implementer writes code + smoke-tests serve.sh"),
    PhaseMeta("C_test",      "C: Correctness Test",
              "run immutable oracle (or test.sh) for correctness only"),
    PhaseMeta("D_review",    "D: Correctness Review",
              "correctness/code reviewer writes review.md; advisory, does NOT "
              "make performance claims before E. "
              "Routes to E on C-pass, fresh A plan on C-fail"),
    PhaseMeta("E_perf_test", "E: Perf Test",
              "agent writes + runs perf.sh (heavier load) → perf_report.json"),
    PhaseMeta("G_perf_review", "G: Perf Review",
              "review measured E results and code changes; writes "
              "perf-review.md before the next perf plan"),
    PhaseMeta("F_perf_plan", "F: Perf Plan",
              "agent reads perf_report.json + review.md, writes perf_plan.md; "
              "no code changes; next iteration's A executes the plan"),
    PhaseMeta("finished",    "finished",
              "run ended (success, stopped at iteration cap, or externally "
              "interrupted)", is_terminal=True),
]

# Left-to-right display order for the graph. Phases not in this list
# (idle / finished / failed) are rendered as status badges, not graph nodes.
PHASE_ORDER: List[Phase] = [
    "A_plan", "B_implement", "C_test", "D_review", "E_perf_test",
    "G_perf_review", "F_perf_plan",
]


# --------------------------------------------------------------------------- #
# Transition table
#
# Keys are (from_phase, outcome). A missing key for a runtime (phase, outcome)
# pair is treated as "close this iteration and start a fresh one at A_plan"
# (see pipeline._loop — undefined transitions no longer abort the run).
#
# D_review's outcome is set by the orchestrator based on what C's outcome was
# (NOT on whether the reviewer agent itself succeeded — D is advisory). So:
#   (D_review, OK)         → E_perf_test   [meaning: C had passed]
#   (D_review, LOGIC_FAIL) → A_plan        [meaning: C had failed]
# --------------------------------------------------------------------------- #

TRANSITIONS: Dict[Tuple[Phase, Outcome], Transition] = {
    # ---- intra-iteration forward (do NOT consume the iteration) ----------- #
    ("A_plan",       OK): Transition("A_plan",       OK, "B_implement",
                                    label="ok",   carry_failure=False, consume_iteration=False),
    ("B_implement",  OK): Transition("B_implement",  OK, "C_test",
                                    label="ok",   carry_failure=False, consume_iteration=False),
    ("C_test",       OK):              Transition("C_test",  OK, "D_review",
                                                  label="pass",   carry_failure=False, consume_iteration=False),
    ("C_test",       LOGIC_FAIL):      Transition("C_test",  LOGIC_FAIL, "D_review",
                                                  label="fail",   carry_failure=True, consume_iteration=False),
    ("C_test",       INFRA_FAIL):      Transition("C_test",  INFRA_FAIL, "D_review",
                                                  label="infra",  carry_failure=True, consume_iteration=False),
    ("C_test",       PERF_REGRESSION): Transition("C_test",  PERF_REGRESSION, "D_review",
                                                  label="regress", carry_failure=True, consume_iteration=False),
    ("D_review",     OK):              Transition("D_review", OK, "E_perf_test",
                                                  label="C ok → perf",
                                                  carry_failure=False, consume_iteration=False),
    ("D_review",     LOGIC_FAIL):      Transition("D_review", LOGIC_FAIL, "A_plan",
                                                  label="C fail → replan",
                                                  carry_failure=True,
                                                  consume_iteration=True),
    ("E_perf_test",  OK):              Transition("E_perf_test", OK, "G_perf_review",
                                                  label="ok", carry_failure=False,
                                                  carry_perf=True, consume_iteration=False),
    ("G_perf_review", OK):              Transition("G_perf_review", OK, "F_perf_plan",
                                                  label="reviewed", carry_failure=False,
                                                  carry_perf=True, consume_iteration=False),
    ("F_perf_plan",  OK):              Transition("F_perf_plan", OK, "A_plan",
                                                  label="new iter",
                                                  carry_failure=False, consume_iteration=True),

    # ---- infra failures --------------------------------------------------- #
    # B is intentionally excluded — see the dedicated B-fail block below.
    #
    # A_plan: rare (no GPU work, just agent). Retry in place is fine.
    ("A_plan",       INFRA_FAIL): Transition("A_plan",       INFRA_FAIL, "A_plan",
                                             label="retry", carry_failure=False, consume_iteration=False),
    # E_perf_test: GPU OOM / serve.sh hang / ccb crash. Retrying in place
    # traps the loop in a 2-hour spiral (PerfOracle waits on the same hung
    # server, OOMs again). Close the iteration and replan: A_plan in the
    # next iteration sees the carried failure ("E_perf_test crashed: HIP
    # OOM") and can adjust the plan (smaller batch / KV cache / etc.).
    # SubAgentManager already retried the agent internally, so in-place
    # retry at the pipeline level adds no value here.
    ("E_perf_test",  INFRA_FAIL): Transition("E_perf_test",  INFRA_FAIL, "A_plan",
                                             label="E infra → replan",
                                             carry_failure=True, consume_iteration=True),
    # F_perf_plan: short (no GPU). Retry in place is fine.
    ("F_perf_plan",  INFRA_FAIL): Transition("F_perf_plan",  INFRA_FAIL, "F_perf_plan",
                                             label="retry", carry_failure=False, consume_iteration=False),
    ("G_perf_review", INFRA_FAIL): Transition("G_perf_review", INFRA_FAIL, "F_perf_plan",
                                             label="advisory fail", carry_failure=False,
                                             consume_iteration=False),

    # ---- B_implement failures: advance to a fresh iteration --------------- #
    # Design choice (Plan A): a B failure — whether INFRA_FAIL (timeout, GPU
    # OOM, ccb crash) or LOGIC_FAIL (agent produced non-JSON / empty / malformed
    # deliverable) — closes the current iteration folder and starts a new one
    # back at A_plan. The previous failure reason is carried forward via
    # ctx.failure so the new planner sees what went wrong and can re-plan
    # around it.
    #
    # Why not retry in place (the old behavior):
    #   1. SubAgentManager already retries internally (max_retries=2 default),
    #      so by the time the pipeline sees a failure, the agent has already
    #      had 2-3 shots at it in the same ccb session.
    #   2. --resume across in-place retries accumulates context: the agent
    #      keeps staring at the same failing test, digging a deeper debugging
    #      hole (real example: 60-minute OOM spiral across 3 attempts, then
    #      the whole run aborted). A fresh iteration forces a fresh session
    #      and a fresh plan, which is the only reliable way to escape that
    #      hole.
    #   3. Outer iteration cap (cfg.max_iterations, default 20) prevents
    #      runaway loops, so advancing is safe.
    ("B_implement",  INFRA_FAIL): Transition("B_implement",  INFRA_FAIL, "A_plan",
                                             label="B fail → replan",
                                             carry_failure=True, consume_iteration=True),
    ("B_implement",  LOGIC_FAIL): Transition("B_implement",  LOGIC_FAIL, "A_plan",
                                             label="B fail → replan",
                                             carry_failure=True, consume_iteration=True),

    # ---- logic failures at A/E/F: redo in place, same folder -------------- #
    # (SubAgentManager already retried 3× internally; one more redo here with
    #  a fresh prompt before burning a new iteration folder.)
    ("A_plan",       LOGIC_FAIL): Transition("A_plan",       LOGIC_FAIL, "A_plan",
                                             label="replan", carry_failure=False, consume_iteration=False),
    ("E_perf_test",  LOGIC_FAIL): Transition("E_perf_test",  LOGIC_FAIL, "E_perf_test",
                                             label="redo",   carry_failure=False, consume_iteration=False),
    ("F_perf_plan",  LOGIC_FAIL): Transition("F_perf_plan",  LOGIC_FAIL, "F_perf_plan",
                                             label="redo",   carry_failure=False, consume_iteration=False),

    # NOTE: there are intentionally NO ``(phase, ABORTED)`` edges. The
    # pipeline loop never produces an ABORTED outcome on its own — when
    # ``MAX_PHASE_ATTEMPTS`` is exceeded, the orchestrator closes the
    # iteration and starts a fresh one rather than aborting. External
    # interrupts (Ctrl-C, SIGTERM) bypass the transition table entirely
    # and write ``final_status="aborted"`` directly. The system never
    # enters a ``"failed"`` terminal state from a transition.
}


# --------------------------------------------------------------------------- #
# Lookup helpers
# --------------------------------------------------------------------------- #


def next_transition(from_phase: Phase, outcome: Outcome) -> Optional[Transition]:
    """Return the transition for ``(from_phase, outcome)`` or ``None`` if no
    edge is defined (the orchestrator treats this as an unrecoverable abort)."""
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


# --------------------------------------------------------------------------- #
# Graph export — consumed by the WebUI
# --------------------------------------------------------------------------- #


def nodes_for_graph() -> List[Dict[str, str]]:
    """Return node metadata for every phase in :data:`PHASE_ORDER`."""
    return [
        {"id": m.id, "label": m.label, "description": m.description}
        for m in PHASES
        if m.id in PHASE_ORDER
    ]


def edges_for_graph() -> List[Dict[str, str]]:
    """Return deduped ``{from, to, label}`` edges.

    Multiple outcomes on the same ``(from, to)`` pair get their labels merged
    (sorted, de-duplicated, ``" / "``-joined) so the graph stays readable.
    """
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


# --------------------------------------------------------------------------- #
# graph_payload — the ONLY function the WebUI's state-graph endpoint calls.
#
# Protocol (task-type-agnostic): every task plugin's phases_module MUST
# expose ``graph_payload(current, last_outcome, last_label) -> dict`` with
# the keys below. The WebUI doesn't know whether the graph is linear,
# multi-edge, or something else — it just renders what we return.
#
# Return shape::
#
#     {
#       "current":         str | None,           # active phase id
#       "nodes":           [{"id","label","description"}],
#       "edges":           [{"from","to","label"}],
#       "active_edge":     {"from","to","label"} | None,
#       "last_outcome":    str | None,           # advisory
#       "terminal_nodes":  [{"id","label","description"}],
#       "outcome_legend":  [{"id","label"}],
#     }
# --------------------------------------------------------------------------- #


def graph_payload(current, last_outcome, last_label) -> Dict[str, any]:
    """Build the state-graph render payload for the WebUI."""
    nodes = nodes_for_graph()
    edges = edges_for_graph()
    active_edge = None
    if last_label:
        for e in edges:
            if e["to"] == current and last_label in e["label"].split(" / "):
                active_edge = {
                    "from": e["from"], "to": e["to"], "label": last_label,
                }
                break
    terminal_nodes = [
        {"id": m.id, "label": m.label, "description": m.description}
        for m in PHASES if m.is_terminal
    ]
    outcome_legend = [
        {"id": o, "label": outcome_label(o)} for o in ALL_OUTCOMES
    ]
    return {
        "current": current,
        "nodes": nodes,
        "edges": edges,
        "active_edge": active_edge,
        "last_outcome": last_outcome,
        "terminal_nodes": terminal_nodes,
        "outcome_legend": outcome_legend,
    }
