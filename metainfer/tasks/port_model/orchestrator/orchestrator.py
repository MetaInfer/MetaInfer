"""Bootstrap entry point for the port-model orchestrator.

Called from the CLI (``cli.py``) with ``run_with_requirements()``.
Sets up directories, creates the SubAgentManager, and launches the
Pipeline state machine.
"""

from __future__ import annotations

import json
import signal
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from metainfer.orchestrator._bootstrap import setup_orchestrator
from metainfer.orchestrator.requirements import req_field, req_field_int, req_field_float
from metainfer.orchestrator.state import StateStore
from metainfer.orchestrator.token_budget import TokenBudget

from .pipeline import OrchestratorConfig, Pipeline


def run_with_requirements(
    *,
    req_path: Path,
    state_dir: Path,
    workspace_dir: Path,
    claude_bin: str = "ccb",
    permission_mode: str = "bypassPermissions",
    model: Optional[str] = None,
    effort: str = "max",
    extra_claude_args: Optional[List[str]] = None,
) -> None:
    """Parse requirements.json, wire up the shared infra, and run the pipeline."""
    req = json.loads(Path(req_path).read_text(encoding="utf-8"))
    form = req.get("form") or {}

    model_dir = Path(form.get("model_dir") or "")
    source_fw = Path(form.get("source_framework_dir") or "")
    target_fw = Path(form.get("target_framework_dir") or "")

    # Workspace subdirs.
    memory_dir = workspace_dir / "memory"
    diff_dir = workspace_dir / "diff"
    test_dir = workspace_dir / "test"
    inputs_snapshot_dir = workspace_dir / "inputs_snapshot"
    logs_root = state_dir / "logs"
    for p in (
        state_dir, workspace_dir, memory_dir, diff_dir, test_dir,
        inputs_snapshot_dir, logs_root,
    ):
        p.mkdir(parents=True, exist_ok=True)

    cfg = OrchestratorConfig(
        workspace_dir=workspace_dir,
        memory_dir=memory_dir,
        diff_dir=diff_dir,
        test_dir=test_dir,
        inputs_snapshot_dir=inputs_snapshot_dir,
        repo_root=workspace_dir.parent.parent.parent,
        state_dir=state_dir,
        logs_root=logs_root,
        source_framework_dir=source_fw,
        target_framework_dir=target_fw,
        model_dir=model_dir,
        claude_bin=claude_bin,
        model=model,
        permission_mode=permission_mode,
        extra_claude_args=list(extra_claude_args or []),
        effort=effort,
        user_paths=[model_dir, source_fw, target_fw],
    )

    store = StateStore(state_dir)

    # Token budget (if max_cost_usd is set).
    max_cost = req_field_float(req, "token_budget_max_cost_usd")
    budget = TokenBudget(task_id=req.get("task_id", ""), max_cost_usd=max_cost) if max_cost else None

    manager = setup_orchestrator(
        task_id=req.get("task_id", ""),
        state_dir=state_dir,
        store=store,
        token_budget=budget,
        claude_bin=claude_bin,
        permission_mode=permission_mode,
        model=model,
        extra_add_dirs=cfg.user_paths,
        extra_claude_args=cfg.extra_claude_args,
    )

    pipeline = Pipeline(req=req, store=store, cfg=cfg, manager=manager)
    pipeline.run()
