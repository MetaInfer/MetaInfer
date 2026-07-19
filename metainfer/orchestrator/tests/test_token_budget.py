"""Unit tests for :mod:`metainfer.orchestrator.token_budget`.

Run directly:

    python tests/test_token_budget.py

Or via pytest once the project adopts it. Assertions raise on failure.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path

from metainfer.orchestrator.token_budget import (
    BudgetSnapshot,
    TokenBudget,
    UsageRecord,
    resolve_budget_limits,
    usage_from_result_event,
)


def _rec(agent: str, cost: float, *, source: str = "orchestrator",
         phase: str = "phase_a") -> UsageRecord:
    return UsageRecord(
        agent=agent, source=source, phase=phase, ended_at=time.time(),
        input_tokens=100, output_tokens=50,
        cache_read_input_tokens=200,
        total_cost_usd=cost,
    )


def test_basic_accumulation():
    with tempfile.TemporaryDirectory() as td:
        b = TokenBudget(td, max_cost_usd=10.0)
        b.record(_rec("a", 1.0))
        b.record(_rec("b", 2.5))
        snap = b.snapshot()
        assert snap.total_cost_usd == 3.5
        assert snap.agent_count == 2
        assert snap.total_input_tokens == 200
        assert snap.total_output_tokens == 100
        assert not snap.exhausted
        assert snap.remaining_cost_usd == 6.5


def test_soft_threshold_flips():
    with tempfile.TemporaryDirectory() as td:
        b = TokenBudget(td, max_cost_usd=1.0)
        snap = b.record(_rec("a", 0.5))
        assert not snap.exhausted
        assert snap.limit_kind is None
        snap = b.record(_rec("b", 0.6))  # total = 1.1, over
        assert snap.exhausted
        assert snap.limit_kind == "soft"
        # Refusal message includes both numbers
        refusal = b.check_launch_allowed("next")
        assert refusal is not None
        assert "1.1000" in refusal
        assert "1.0000" in refusal


def test_check_launch_allowed_under_limit():
    with tempfile.TemporaryDirectory() as td:
        b = TokenBudget(td, max_cost_usd=10.0)
        b.record(_rec("a", 1.0))
        assert b.check_launch_allowed("b") is None


def test_persistence_round_trip():
    with tempfile.TemporaryDirectory() as td:
        b = TokenBudget(td, max_cost_usd=5.0)
        b.record(_rec("a", 1.0, phase="phase_a"))
        b.record(_rec("b", 2.0, phase="phase_b"))
        # File exists and is valid JSON
        assert (Path(td) / "token_budget.json").exists()
        # Reload — totals must match
        b2 = TokenBudget(td)  # no limits passed; should pick up from disk
        snap = b2.snapshot()
        assert snap.total_cost_usd == 3.0
        assert snap.agent_count == 2
        assert snap.per_phase == {"phase_a": 1.0, "phase_b": 2.0}
        assert snap.per_source == {"orchestrator": 3.0}
        # Limit was loaded from disk too
        assert snap.limit_cost_usd == 5.0


def test_persistence_atomic():
    """Parallel recorders must not corrupt the JSON. Single-process
    simulation via threads."""
    import threading
    with tempfile.TemporaryDirectory() as td:
        b = TokenBudget(td, max_cost_usd=10000.0)

        def worker(n: int):
            for i in range(20):
                b.record(_rec(f"agent-{n}-{i}", 0.1))

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        snap = b.snapshot()
        # 8 threads * 20 records * 0.1 = 16.0
        assert abs(snap.total_cost_usd - 16.0) < 1e-6, snap.total_cost_usd
        assert snap.agent_count == 160
        # JSON still parseable
        data = json.loads((Path(td) / "token_budget.json").read_text())
        assert len(data["records"]) == 160


def test_hard_threshold_callback():
    """Hard threshold fires callback exactly once."""
    fired = []
    with tempfile.TemporaryDirectory() as td:
        b = TokenBudget(
            td, max_cost_usd=1.0, max_cost_usd_hard=2.0,
            on_hard_exhausted=lambda: fired.append(time.time()),
        )
        b.record(_rec("a", 0.5))   # neither
        b.record(_rec("b", 0.7))   # soft crossed (1.2 > 1.0)
        assert len(fired) == 0
        b.record(_rec("c", 1.0))   # hard crossed (2.2 > 2.0)
        assert len(fired) == 1
        snap = b.snapshot()
        assert snap.hard_exhausted
        assert snap.limit_kind == "hard"
        # More records do NOT refire
        b.record(_rec("d", 0.1))
        assert len(fired) == 1


def test_per_source_phase_buckets():
    with tempfile.TemporaryDirectory() as td:
        b = TokenBudget(td, max_cost_usd=100.0)
        b.record(_rec("a", 1.0, source="orchestrator", phase="phase_a"))
        b.record(_rec("b", 2.0, source="orchestrator", phase="phase_b"))
        b.record(_rec("c", 3.0, source="web_qa", phase=None))
        snap = b.snapshot()
        assert snap.per_source == {"orchestrator": 3.0, "web_qa": 3.0}
        # None phase gets bucketed under "(unknown)"
        assert snap.per_phase == {"phase_a": 1.0, "phase_b": 2.0,
                                  "(unknown)": 3.0}


def test_usage_from_result_event():
    ev = {
        "type": "result",
        "session_id": "abc-123",
        "total_cost_usd": 1.25,
        "usage": {
            "input_tokens": 1000,
            "output_tokens": 500,
            "cache_read_input_tokens": 5000,
            "cache_creation_input_tokens": 100,
        },
    }
    rec = usage_from_result_event(ev, agent="x", source="orchestrator",
                                  phase="phase_b")
    assert rec.agent == "x"
    assert rec.input_tokens == 1000
    assert rec.output_tokens == 500
    assert rec.cache_read_input_tokens == 5000
    assert rec.total_cost_usd == 1.25
    assert rec.session_id == "abc-123"
    # Tolerant of missing usage block
    rec2 = usage_from_result_event({"type": "result"}, agent="y",
                                   source="orchestrator")
    assert rec2.input_tokens == 0
    assert rec2.total_cost_usd == 0.0


def test_reset_clears_everything():
    with tempfile.TemporaryDirectory() as td:
        b = TokenBudget(td, max_cost_usd=1.0)
        b.record(_rec("a", 2.0))  # over
        assert b.snapshot().exhausted
        b.reset()
        snap = b.snapshot()
        assert snap.total_cost_usd == 0.0
        assert snap.agent_count == 0
        assert not snap.exhausted


def test_update_limit_unblocks():
    """Raising the limit past the current total clears the exhausted flag."""
    with tempfile.TemporaryDirectory() as td:
        b = TokenBudget(td, max_cost_usd=1.0)
        b.record(_rec("a", 2.0))  # over
        assert b.snapshot().exhausted
        # Raise the limit
        snap = b.update_limit(max_cost_usd=10.0)
        assert not snap.exhausted
        assert snap.limit_cost_usd == 10.0
        assert snap.remaining_cost_usd == 8.0
        # check_launch_allowed now passes
        assert b.check_launch_allowed("b") is None
        # Persistence — re-read from disk sees the new limit too
        b2 = TokenBudget(td)
        snap2 = b2.snapshot()
        assert snap2.limit_cost_usd == 10.0
        assert not snap2.exhausted


def test_external_edit_is_hot_reloaded():
    """If another process edits token_budget.json, an in-life TokenBudget
    sees the new config on the next snapshot() call."""
    with tempfile.TemporaryDirectory() as td:
        b = TokenBudget(td, max_cost_usd=1.0)
        b.record(_rec("a", 0.5))
        assert not b.snapshot().exhausted
        # Simulate the WebUI writing a lower limit
        data = json.loads((Path(td) / "token_budget.json").read_text())
        data["config"]["max_cost_usd"] = 0.1  # below current total
        # Bump mtime by writing with a small delay
        time.sleep(0.05)
        (Path(td) / "token_budget.json").write_text(json.dumps(data))
        # Next snapshot should reflect the new limit + exhausted flag
        snap = b.snapshot()
        assert snap.limit_cost_usd == 0.1
        assert snap.exhausted


def test_resolve_budget_limits_runtime_file_overrides_seed():
    """Regression: WebUI raises the budget mid-task → orchestrator
    restarts → the new (runtime) limit must win over the stale seed in
    requirements.json. Before resolve_budget_limits existed, the
    orchestrator read the seed and silently ignored the runtime file,
    so user budget bumps were lost on restart."""
    with tempfile.TemporaryDirectory() as td:
        # requirements.json has the original $50 seed from task creation
        req = {"token_budget_max_cost_usd": 50}
        # WebUI later bumped to $100 — runtime file is authoritative
        (Path(td) / "token_budget.json").write_text(json.dumps({
            "schema_version": 1,
            "config": {"max_cost_usd": 100.0, "max_cost_usd_hard": None},
            "totals": {"total_cost_usd": 50.28},
        }))
        soft, hard = resolve_budget_limits(td, req)
        assert soft == 100.0
        assert hard is None


def test_resolve_budget_limits_falls_back_to_seed_on_first_boot():
    """First boot: no token_budget.json yet → seed from requirements.json."""
    with tempfile.TemporaryDirectory() as td:
        req = {"token_budget_max_cost_usd": 50}
        soft, hard = resolve_budget_limits(td, req)
        assert soft == 50.0
        assert hard is None


def test_resolve_budget_limits_env_var_wins():
    """Env var is the ops escape hatch — overrides both file + seed."""
    with tempfile.TemporaryDirectory() as td:
        req = {"token_budget_max_cost_usd": 50}
        (Path(td) / "token_budget.json").write_text(json.dumps({
            "config": {"max_cost_usd": 100.0},
        }))
        old = os.environ.get("METAINFER_TOKEN_BUDGET_COST_USD")
        os.environ["METAINFER_TOKEN_BUDGET_COST_USD"] = "200"
        try:
            soft, _ = resolve_budget_limits(td, req)
            assert soft == 200.0
        finally:
            if old is None:
                del os.environ["METAINFER_TOKEN_BUDGET_COST_USD"]
            else:
                os.environ["METAINFER_TOKEN_BUDGET_COST_USD"] = old


def test_resolve_budget_limits_nested_token_budget_object():
    """requirements.json::token_budget.max_cost_usd (nested object form)
    is honored on first boot too."""
    with tempfile.TemporaryDirectory() as td:
        req = {"token_budget": {"max_cost_usd": 75.0, "max_cost_usd_hard": 90.0}}
        soft, hard = resolve_budget_limits(td, req)
        assert soft == 75.0
        assert hard == 90.0


def test_resolve_budget_limits_disabled_when_nothing_set():
    """No env, no runtime file, no seed → (None, None) → budget disabled."""
    with tempfile.TemporaryDirectory() as td:
        soft, hard = resolve_budget_limits(td, {})
        assert soft is None
        assert hard is None


def _main() -> None:
    tests = [
        ("test_basic_accumulation", test_basic_accumulation),
        ("test_soft_threshold_flips", test_soft_threshold_flips),
        ("test_check_launch_allowed_under_limit", test_check_launch_allowed_under_limit),
        ("test_persistence_round_trip", test_persistence_round_trip),
        ("test_persistence_atomic", test_persistence_atomic),
        ("test_hard_threshold_callback", test_hard_threshold_callback),
        ("test_per_source_phase_buckets", test_per_source_phase_buckets),
        ("test_usage_from_result_event", test_usage_from_result_event),
        ("test_reset_clears_everything", test_reset_clears_everything),
        ("test_update_limit_unblocks", test_update_limit_unblocks),
        ("test_external_edit_is_hot_reloaded", test_external_edit_is_hot_reloaded),
        ("test_resolve_budget_limits_runtime_file_overrides_seed",
         test_resolve_budget_limits_runtime_file_overrides_seed),
        ("test_resolve_budget_limits_falls_back_to_seed_on_first_boot",
         test_resolve_budget_limits_falls_back_to_seed_on_first_boot),
        ("test_resolve_budget_limits_env_var_wins",
         test_resolve_budget_limits_env_var_wins),
        ("test_resolve_budget_limits_nested_token_budget_object",
         test_resolve_budget_limits_nested_token_budget_object),
        ("test_resolve_budget_limits_disabled_when_nothing_set",
         test_resolve_budget_limits_disabled_when_nothing_set),
    ]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    _main()
