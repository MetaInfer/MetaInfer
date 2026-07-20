"""Tests for graph_validator pool + loop driver.

Uses MockAgentManager so we can drive validation rounds without spawning
real Claude Code subprocesses.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Dict

from metainfer.testing.mock_agent import MockAgentManager

from metainfer.tasks.find_low_hanging_kernel.orchestrator import graph_validator
from metainfer.tasks.find_low_hanging_kernel.orchestrator.graph_schema import (
    check_integrity,
)
from metainfer.tasks.find_low_hanging_kernel.tests._helpers import (
    make_minimal_valid_graph,
)


def _stub_step_files(tmp_path: Path) -> Path:
    """Lay out a fake step1.md + step2.md + framework dir."""
    mem = tmp_path / "memory"
    mem.mkdir()
    (mem / "step1_code_analysis.md").write_text("# step1 stub\n", encoding="utf-8")
    (mem / "step2_tracing_analysis.md").write_text("# step2 stub\n", encoding="utf-8")
    fw = tmp_path / "framework"
    fw.mkdir()
    (fw / "norm.py").write_text("# stub framework file\n", encoding="utf-8")
    return fw


def test_split_into_groups_preserves_order():
    nodes = [{"id": f"n{i:02d}"} for i in range(7)]
    groups = graph_validator._split_into_groups(nodes)
    assert len(groups) == 3
    assert [g[0]["id"] for g in groups] == ["n00", "n03", "n06"]
    assert [len(g) for g in groups] == [3, 3, 1]


def test_extract_json_object_handles_fenced_block():
    text = '''Here is my verdict:

```json
{"n01": {"ok": true}}
```

Done.'''
    parsed = graph_validator._extract_json_object(text)
    assert parsed == {"n01": {"ok": True}}


def test_extract_json_object_returns_none_on_garbage():
    assert graph_validator._extract_json_object("no json here") is None


def test_apply_patches_skips_id_changes():
    graph = copy.deepcopy(make_minimal_valid_graph())
    verdicts = {
        "n01": {"ok": False, "suggested_patch": {"id": "nEVIL", "role": "renamed"}},
    }
    applied, notes = graph_validator.apply_patches(graph, verdicts)
    assert applied == 0
    assert any("refusing to change id" in n for n in notes)
    # Original id preserved.
    assert graph["nodes"][0]["id"] == "n01"


def test_apply_patches_applies_safe_fields():
    graph = copy.deepcopy(make_minimal_valid_graph())
    verdicts = {
        "n02": {
            "ok": False,
            "suggested_patch": {
                "role": "BetterRMSNorm",
                "confidence": "high",
                "source_ref": {"file": "norm.py", "line": 99, "symbol": "Better.forward"},
            },
        },
    }
    applied, _ = graph_validator.apply_patches(graph, verdicts)
    assert applied == 3
    n02 = graph["nodes"][1]
    assert n02["role"] == "BetterRMSNorm"
    assert n02["confidence"] == "high"
    assert n02["source_ref"]["line"] == 99


def test_validation_loop_converges_in_two_rounds(tmp_path: Path):
    """Round 1: every worker reports 1 issue → needs_fix. Round 2: clean."""
    fw_dir = _stub_step_files(tmp_path)
    step1 = tmp_path / "memory" / "step1_code_analysis.md"
    step2 = tmp_path / "memory" / "step2_tracing_analysis.md"
    validation_root = tmp_path / "validation"
    logs_root = tmp_path / "logs"

    def response_fn(spec) -> str:
        # spec.name like "validator_r1_g00_<ids>" — detect round number.
        m = re.match(r"validator_r(\d+)_g", spec.name)
        round_num = int(m.group(1)) if m else 1
        # Extract the node ids from the prompt to emit verdicts per node.
        # The prompt contains a JSON block with the node group; we just emit
        # a verdict for "n01" / "n02" / "n03" — any id works since the validator
        # only requires the JSON shape.
        if round_num == 1:
            return json.dumps({
                "n01": {"ok": True},
                "n02": {"ok": False, "issues": ["wrong shape"], "suggested_patch": {"confidence": "low"}},
                "n03": {"ok": True},
            })
        return json.dumps({
            "n01": {"ok": True},
            "n02": {"ok": True},
            "n03": {"ok": True},
        })

    manager = MockAgentManager(response_fn=response_fn)
    graph = copy.deepcopy(make_minimal_valid_graph())

    rounds, exhausted = graph_validator.run_validation_loop(
        graph=graph,
        manager=manager,
        step1_path=step1,
        step2_path=step2,
        framework_dir=fw_dir,
        validation_root=validation_root,
        logs_root=logs_root,
        max_rounds=5,
        timeout_s=30,
    )

    assert len(rounds) == 2
    assert rounds[0].outcome == "needs_fix"
    assert rounds[1].outcome == "clean"
    assert exhausted is False
    # Round directories were written.
    assert (validation_root / "round_01" / "integrity_fixes.json").is_file()
    group_outputs = list((validation_root / "round_01").glob("group_*.json"))
    assert group_outputs


def test_validation_loop_exhausts_at_cap(tmp_path: Path):
    fw_dir = _stub_step_files(tmp_path)
    step1 = tmp_path / "memory" / "step1_code_analysis.md"
    step2 = tmp_path / "memory" / "step2_tracing_analysis.md"

    def always_issues(spec) -> str:
        return json.dumps({
            "n02": {"ok": False, "issues": ["perpetually wrong"],
                    "suggested_patch": {"confidence": "low"}}
        })

    manager = MockAgentManager(response_fn=always_issues)
    graph = copy.deepcopy(make_minimal_valid_graph())
    rounds, exhausted = graph_validator.run_validation_loop(
        graph=graph,
        manager=manager,
        step1_path=step1,
        step2_path=step2,
        framework_dir=fw_dir,
        validation_root=tmp_path / "validation",
        logs_root=tmp_path / "logs",
        max_rounds=3,
        timeout_s=30,
    )
    assert len(rounds) == 3
    assert exhausted is True
    # All rounds reported needs_fix.
    assert all(r.outcome == "needs_fix" for r in rounds)


def test_integrity_errors_short_circuit_pool(tmp_path: Path):
    """If integrity check fails, we skip the pool entirely."""
    fw_dir = _stub_step_files(tmp_path)
    step1 = tmp_path / "memory" / "step1_code_analysis.md"
    step2 = tmp_path / "memory" / "step2_tracing_analysis.md"

    manager = MockAgentManager(response_fn=lambda spec: '{"n01": {"ok": true}}')
    # Duplicate node id → integrity error.
    graph = copy.deepcopy(make_minimal_valid_graph())
    graph["nodes"].append(copy.deepcopy(graph["nodes"][1]))

    result = graph_validator.run_validation_round(
        round_num=1,
        graph=graph,
        manager=manager,
        step1_path=step1,
        step2_path=step2,
        framework_dir=fw_dir,
        round_dir=tmp_path / "round_01",
        pool_log_dir=tmp_path / "pool_logs",
        timeout_s=30,
    )
    assert result.outcome == "needs_fix"
    assert result.group_result_paths == []  # pool was skipped
    assert any("duplicate" in e for e in result.integrity.errors)
    # Manager never launched any agents.
    assert manager.launched_specs == []
