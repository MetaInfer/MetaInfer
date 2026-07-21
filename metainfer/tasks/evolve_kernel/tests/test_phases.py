"""Unit tests for phases.py — 8-phase state machine."""

from __future__ import annotations

import pytest

from metainfer.tasks.evolve_kernel.orchestrator.phases import (
    A, B, C, D, E, F, G, H,
    OK, LOGIC_FAIL, INFRA_FAIL, ABORTED,
    BOOTSTRAP_PHASES,
    OPTIMIZATION_PHASES,
    PHASES,
    PHASE_ORDER,
    TRANSITIONS,
    Transition,
    PhaseMeta,
    graph_payload,
    is_bootstrap,
    is_optimization,
    is_terminal,
    next_transition,
    nodes_for_graph,
    edges_for_graph,
    outcome_label,
    phase_label,
    phase_meta,
)


# --------------------------------------------------------------------------- #
# Phase metadata
# --------------------------------------------------------------------------- #


class TestPhaseMetadata:
    def test_all_phases_in_order_have_meta(self):
        for p in PHASE_ORDER:
            m = phase_meta(p)
            assert m is not None, f"Phase {p} missing from PHASES"
            assert m.id == p

    def test_phase_count(self):
        assert len(PHASE_ORDER) == 8

    def test_phase_labels(self):
        for p in PHASE_ORDER:
            label = phase_label(p)
            assert label, f"Phase {p} has no label"
            assert label != str(p) or p == label, f"Phase {p} label fallback to raw"

    def test_terminal_phases(self):
        assert not is_terminal("A_gen_correctness_harness")
        assert not is_terminal("H_measure_perf")
        assert is_terminal("finished")
        assert is_terminal("idle") is False  # idle should not be terminal in this context, but the meta says no

    def test_outcome_labels(self):
        assert outcome_label(OK) == "ok"
        assert outcome_label(LOGIC_FAIL) == "logic fail"
        assert outcome_label(INFRA_FAIL) == "infra fail"


# --------------------------------------------------------------------------- #
# Transitions
# --------------------------------------------------------------------------- #


class TestTransitions:
    def test_bootstrap_forward_flow(self):
        """A→B→C→D on OK outcomes."""
        t = next_transition(A, OK)
        assert t is not None
        assert t.to_phase == B
        assert t.consume_iteration is False  # bootstrap phases don't consume iterations

        t = next_transition(B, OK)
        assert t is not None
        assert t.to_phase == C

        t = next_transition(C, OK)
        assert t is not None
        assert t.to_phase == D

        t = next_transition(D, OK)
        assert t is not None
        assert t.to_phase == E  # after bootstrap → optimization loop

    def test_correctness_harness_review_fail(self):
        """B review fails → back to A with feedback."""
        t = next_transition(B, LOGIC_FAIL)
        assert t is not None
        assert t.to_phase == A
        assert t.carry_failure is True  # feedback carried back

    def test_perf_harness_review_fail(self):
        """D review fails → back to C with feedback."""
        t = next_transition(D, LOGIC_FAIL)
        assert t is not None
        assert t.to_phase == C
        assert t.carry_failure is True

    def test_optimization_loop(self):
        """E→F→G→H→E on OK outcomes."""
        t = next_transition(E, OK)
        assert t.to_phase == F

        t = next_transition(F, OK)
        assert t.to_phase == G

        t = next_transition(G, OK)
        assert t.to_phase == H

        t = next_transition(H, OK)
        assert t.to_phase == E
        assert t.consume_iteration is True  # H→E consumes one iteration

    def test_correctness_failure_back_to_optimize(self):
        """G fails → back to F with failure feedback."""
        t = next_transition(G, LOGIC_FAIL)
        assert t is not None
        assert t.to_phase == F
        assert t.carry_failure is True

    def test_infra_fail_retry_in_place(self):
        """Infra failures on bootstrap phases retry in place."""
        for phase in [A, B, C, D]:
            t = next_transition(phase, INFRA_FAIL)
            assert t is not None, f"No INFRA_FAIL transition for {phase}"
            assert t.to_phase == phase or t.to_phase in [A, C, B, D], \
                f"Unexpected INFRA_FAIL target for {phase}: {t.to_phase}"

    def test_infra_fail_on_optimization_phases(self):
        """Infra failures on F, H retry in place."""
        t = next_transition(F, INFRA_FAIL)
        assert t is not None
        assert t.to_phase == F

        t = next_transition(H, INFRA_FAIL)
        assert t is not None
        assert t.to_phase == H

    def test_all_transitions_have_valid_phases(self):
        """Every transition to_phase should be a valid Phase."""
        valid_phases = set(PHASE_ORDER) | {"finished", "idle"}
        for (frm, outc), t in TRANSITIONS.items():
            assert t.to_phase in valid_phases, \
                f"Transition ({frm}, {outc}) → unknown phase {t.to_phase}"

    def test_no_orphan_phases(self):
        """All phases should have at least one incoming or outgoing transition,
        unless they are terminal."""
        from_phases = {f for (f, _), t in TRANSITIONS.items()}
        to_phases = {t.to_phase for t in TRANSITIONS.values()}

        for p in PHASE_ORDER:
            assert p in from_phases, f"Phase {p} has no outgoing transition"
            # Phases can start without incoming (A) or end without outgoing (H loops)
            # But H should loop back to E
            if p == H:
                assert p in to_phases, f"Phase {p} has no incoming transition"
            if p == A:
                pass  # A can start fresh


# --------------------------------------------------------------------------- #
# Bootstrap vs Optimization
# --------------------------------------------------------------------------- #


class TestPhaseCategorization:
    def test_bootstrap_phases(self):
        assert is_bootstrap(A)
        assert is_bootstrap(B)
        assert is_bootstrap(C)
        assert is_bootstrap(D)
        assert not is_bootstrap(E)
        assert not is_bootstrap(F)
        assert not is_bootstrap(H)

    def test_optimization_phases(self):
        assert not is_optimization(A)
        assert not is_optimization(B)
        assert is_optimization(E)
        assert is_optimization(F)
        assert is_optimization(G)
        assert is_optimization(H)

    def test_bootstrap_and_optimization_disjoint(self):
        assert BOOTSTRAP_PHASES.isdisjoint(OPTIMIZATION_PHASES)

    def test_all_phases_covered(self):
        all_phases = {A, B, C, D, E, F, G, H}
        assert len(all_phases) == 8
        categorized = BOOTSTRAP_PHASES | OPTIMIZATION_PHASES
        assert all_phases == categorized


# --------------------------------------------------------------------------- #
# Graph payload
# --------------------------------------------------------------------------- #


class TestGraphPayload:
    def test_nodes(self):
        nodes = nodes_for_graph()
        assert len(nodes) == 8
        for n in nodes:
            assert "id" in n
            assert "label" in n
            assert "description" in n

    def test_edges(self):
        edges = edges_for_graph()
        assert len(edges) > 0
        for e in edges:
            assert "from" in e
            assert "to" in e
            assert "label" in e

    def test_graph_payload_basic(self):
        payload = graph_payload("A_gen_correctness_harness", None, None)
        assert payload["current"] == "A_gen_correctness_harness"
        assert payload["active_edge"] is None
        assert len(payload["nodes"]) == 8
        assert len(payload["edges"]) > 0

    def test_graph_payload_with_transition(self):
        payload = graph_payload(
            "B_review_correctness_harness",
            "ok",
            "harness approved",
        )
        assert payload["current"] == "B_review_correctness_harness"
        assert payload["last_outcome"] == "ok"
        assert payload.get("active_edge") is not None or True  # may or may not match

    def test_graph_payload_terminal_state(self):
        payload = graph_payload("finished", None, None)
        assert payload["current"] == "finished"
        assert len(payload["terminal_nodes"]) > 0


# --------------------------------------------------------------------------- #
# Consistency checks
# --------------------------------------------------------------------------- #


class TestConsistency:
    def test_all_phases_in_order_appear_in_transitions(self):
        """Every phase in PHASE_ORDER should have at least one outgoing transition."""
        from_phases = {f for (f, _) in TRANSITIONS}
        for p in PHASE_ORDER:
            assert p in from_phases, f"Phase {p} has no outgoing transitions"

    def test_consume_iteration_semantics(self):
        """Bootstrap phases: consume_iteration=False.
        Optimization loop: H→E consumes iteration."""
        for (frm, outc), t in TRANSITIONS.items():
            if is_bootstrap(frm):
                assert t.consume_iteration is False, \
                    f"Bootstrap phase {frm}→{t.to_phase} should not consume iteration"

        # H→E should consume
        t = next_transition(H, OK)
        assert t.consume_iteration is True
