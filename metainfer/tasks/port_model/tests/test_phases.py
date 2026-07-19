"""Tests for the port-model phase state machine."""

from __future__ import annotations

from metainfer.tasks.port_model.orchestrator.phases import (
    PHASES,
    Phase,
    TRANSITIONS,
    graph_payload,
    is_terminal,
    next_transition,
    phase_label,
)


def test_phase_count():
    phase_ids = {m.id for m in PHASES}
    assert "P1_model_analysis" in phase_ids
    assert "P2_source_analysis" in phase_ids
    assert "P3_target_analysis" in phase_ids
    assert "P4_implement" in phase_ids
    assert "P5_test" in phase_ids
    assert "finished" in phase_ids


def test_linear_forward_path():
    path: list[Phase] = [
        "P1_model_analysis", "P2_source_analysis", "P3_target_analysis",
        "P4_implement", "P5_test",
    ]
    for i in range(len(path) - 1):
        t = next_transition(path[i], "ok")
        assert t is not None, f"no transition from {path[i]} ok"
        assert t.to_phase == path[i + 1], f"{path[i]} ok → {t.to_phase}, expected {path[i + 1]}"


def test_p5_pass_to_done():
    t = next_transition("P5_test", "ok")
    assert t is not None
    assert t.to_phase == "finished"


def test_p5_test_fail_goes_back_to_implement():
    t = next_transition("P5_test", "test_fail")
    assert t is not None
    assert t.to_phase == "P4_implement"


def test_p5_infra_fail_goes_back_to_implement():
    t = next_transition("P5_test", "infra_fail")
    assert t is not None
    assert t.to_phase == "P4_implement"


def test_logic_fail_stops():
    for phase in (
        "P1_model_analysis", "P2_source_analysis",
        "P3_target_analysis", "P4_implement",
    ):
        t = next_transition(phase, "logic_fail")
        assert t is not None, f"no transition from {phase} logic_fail"
        assert t.to_phase == "finished", f"{phase} logic_fail → {t.to_phase}"


def test_infra_fail_retries_analysis():
    for phase in ("P1_model_analysis", "P2_source_analysis", "P3_target_analysis"):
        t = next_transition(phase, "infra_fail")
        assert t is not None, f"no transition from {phase} infra_fail"
        assert t.to_phase == phase  # self-loop


def test_terminal_only_finished():
    assert is_terminal("finished") is True
    assert is_terminal("P1_model_analysis") is False
    assert is_terminal("P5_test") is False


def test_graph_payload_has_all_nodes():
    payload = graph_payload("P1_model_analysis", "ok", "ok")
    assert len(payload["nodes"]) == 5  # P1-P5
    assert payload["current"] == "P1_model_analysis"
    assert isinstance(payload["edges"], list)
    assert len(payload["edges"]) > 0
    assert len(payload["outcome_legend"]) > 0
