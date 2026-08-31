"""Lifecycle wrapper for the mock optimization pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from metainfer.orchestrator._bootstrap import (
    clear_pid_file,
    install_subagent_shutdown_handlers,
    make_subagent_manager,
    set_process_name,
    write_pid_file,
)
from metainfer.orchestrator.state import StateStore

from .config import (
    GEN_AND_OPT_MODE,
    LEGACY_SMOKE_MODE,
    SMOKE_MODE,
    W8A8_MODE,
    load_config,
)
from .gen_and_opt_pipeline import GenAndOptPipeline
from .pipeline import MockOptimizationPipeline
from .real_pipeline import RealSmokeOptimizationPipeline
from .w8a8_pipeline import RealW8A8OptimizationPipeline


def run_with_requirements(
    requirements_path: Path,
    *,
    state_dir: Optional[Path] = None,
    workspace_dir: Optional[Path] = None,
    dry_run: bool = False,
    claude_bin: str = "ccb",
) -> int:
    if not requirements_path.is_file():
        raise FileNotFoundError(requirements_path)
    req: Dict[str, Any] = json.loads(
        requirements_path.read_text(encoding="utf-8")
    )
    task_id = str(req.get("task_id", "task"))
    # The container-side bridge client forwards this ownership tag to the
    # host bridge so task deletion can terminate only this task's agents.
    os.environ["METAINFER_TASK_ID"] = task_id
    if state_dir is None or workspace_dir is None:
        from metainfer.server import paths as web_paths
        state_dir = state_dir or web_paths.task_dir(task_id)
        workspace_dir = workspace_dir or web_paths.workspace_dir(task_id)
    state_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    target_req = state_dir / "requirements.json"
    if requirements_path.resolve() != target_req.resolve():
        target_req.write_text(
            requirements_path.read_text(encoding="utf-8"), encoding="utf-8"
        )

    set_process_name("metainfer-dkao")
    pid_file = state_dir / "orchestrator.pid"
    write_pid_file(pid_file, task_id)
    answers = req.get("answers")
    mode_source = answers if isinstance(answers, dict) else req
    mode = str(mode_source.get("execution_mode", "Mock (no GPU)"))
    config = load_config(req)
    manager = make_subagent_manager(
        claude_bin=claude_bin,
        model=config.claude_model,
        permission_mode="bypassPermissions",
        effort=(
            "low"
            if mode in {LEGACY_SMOKE_MODE, SMOKE_MODE}
            else "max"
        ),
        extra_add_dirs=[workspace_dir],
        snapshot_file=state_dir / "agents.json",
        max_concurrent=4,
    )
    restore = install_subagent_shutdown_handlers(manager, pid_file=pid_file)
    try:
        common = {
            "req": req,
            "state_dir": state_dir,
            "workspace_dir": workspace_dir,
            "store": StateStore(state_dir),
        }
        if mode in {LEGACY_SMOKE_MODE, SMOKE_MODE}:
            pipeline = RealSmokeOptimizationPipeline(
                manager=manager, **common
            )
        elif mode == W8A8_MODE:
            pipeline = RealW8A8OptimizationPipeline(
                manager=manager, **common
            )
        elif mode == GEN_AND_OPT_MODE:
            pipeline = GenAndOptPipeline(
                manager=manager, **common
            )
        else:
            pipeline = MockOptimizationPipeline(**common)
        pipeline.run(dry_run=dry_run)
        return 0
    finally:
        manager.shutdown()
        restore()
        clear_pid_file(pid_file)
