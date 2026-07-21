from __future__ import annotations

import pytest

from ..orchestrator.guidance import GuidanceError, GuidanceStore


def test_guidance_waits_for_an_action_agent_boundary(tmp_path):
    store = GuidanceStore(tmp_path / "guidance")
    item = store.submit("Try a wider N tile, but keep the public ABI.")
    assert item["status"] == "pending"
    assert store.consume(iteration=1, phase="D_review", role="reviewer") == []
    assert store.snapshot()["pending_count"] == 1

    delivered = store.consume(iteration=2, phase="A_plan", role="planner")
    assert [entry["id"] for entry in delivered] == [item["id"]]
    snapshot = store.snapshot()
    assert snapshot["pending_count"] == 0
    assert snapshot["items"][0]["applied_iteration"] == 2
    assert snapshot["items"][0]["applied_role"] == "planner"


def test_guidance_rejects_empty_and_oversized_text(tmp_path):
    store = GuidanceStore(tmp_path / "guidance")
    with pytest.raises(GuidanceError, match="required"):
        store.submit("  ")
    with pytest.raises(GuidanceError, match="8000"):
        store.submit("x" * 8_001)
