from __future__ import annotations

from ..orchestrator.guidance import (
    add_guidance,
    claim_next_guidance,
    list_guidance,
)


def test_guidance_is_worker_isolated_and_claimed_once(tmp_path):
    first = add_guidance(tmp_path, "worker_0", "Use a smaller tile")
    add_guidance(tmp_path, "worker_1", "Try split-K")

    claimed = claim_next_guidance(tmp_path, "worker_0", 4)
    assert claimed is not None
    assert claimed["id"] == first["id"]
    assert claimed["consumed_iteration"] == 4
    assert claim_next_guidance(tmp_path, "worker_0", 5) is None

    worker_0 = list_guidance(tmp_path, "worker_0")
    worker_1 = list_guidance(tmp_path, "worker_1")
    assert worker_0[0]["status"] == "consumed"
    assert worker_1[0]["status"] == "pending"


def test_guidance_rejects_empty_text(tmp_path):
    try:
        add_guidance(tmp_path, "worker_0", "  ")
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("empty guidance should be rejected")
