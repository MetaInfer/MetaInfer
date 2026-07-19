"""Bootstrap + entry point for the find-low-hanging-kernel orchestrator.

Reads requirements.json, wires up StateStore + SubAgentManager + IterationWorkspace,
constructs the :class:`Pipeline` and runs it. Most of the heavy lifting lives in
``pipeline.py``; this module is the glue.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from metainfer.orchestrator._bootstrap import (
    clear_pid_file,
    install_subagent_shutdown_handlers,
    make_subagent_manager,
    set_process_name,
    write_pid_file,
)
from metainfer.orchestrator.paths import repo_root as _repo_root
from metainfer.orchestrator.state import StateStore

from .pipeline import OrchestratorConfig, Pipeline


def _task_subdirs(state_dir: Path, workspace_dir: Path) -> Dict[str, Path]:
    state_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir.mkdir(parents=True, exist_ok=True)

    memory = workspace_dir / "memory"
    validation = workspace_dir / "validation"
    inputs_snapshot = workspace_dir / "inputs_snapshot"
    for p in (memory, validation, inputs_snapshot):
        p.mkdir(parents=True, exist_ok=True)

    logs = state_dir / "logs"
    iterations_state = state_dir / "iterations"
    for p in (logs, iterations_state):
        p.mkdir(parents=True, exist_ok=True)

    return {
        "state_dir": state_dir,
        "workspace_dir": workspace_dir,
        "memory_dir": memory,
        "validation_dir": validation,
        "inputs_snapshot_dir": inputs_snapshot,
        "logs_root": logs,
        "iterations_state": iterations_state,
        "requirements": state_dir / "requirements.json",
        "pid_file": state_dir / "orchestrator.pid",
        "log_file": state_dir / "orchestrator.log",
        "run_file": state_dir / "run.json",
        "timeline_file": state_dir / "timeline.jsonl",
        "agents_file": state_dir / "agents.json",
    }


def _parse_max_validator_rounds(req: Dict[str, Any], default: int = 5) -> int:
    from metainfer.orchestrator.requirements import req_field_int
    return req_field_int(req, "max_validator_rounds", default)


def run_with_requirements(
    requirements_path: Path,
    *,
    state_dir: Optional[Path] = None,
    workspace_dir: Optional[Path] = None,
    claude_bin: str = "ccb",
    model: Optional[str] = None,
    permission_mode: str = "bypassPermissions",
    extra_claude_args: Optional[List[str]] = None,
    effort: str = "max",
) -> int:
    if not requirements_path.exists():
        raise FileNotFoundError(f"requirements file not found: {requirements_path}")

    req: Dict[str, Any] = json.loads(requirements_path.read_text(encoding="utf-8"))
    task_id = req.get("task_id", "task")

    set_process_name("metainfer-orch")

    if state_dir is None:
        state_dir = Path.cwd() / "nodes" / "localhost" / ".metainfer" / "tasks" / task_id
    if workspace_dir is None:
        workspace_dir = Path.cwd() / "nodes" / "localhost" / "workspaces" / task_id
    paths = _task_subdirs(state_dir, workspace_dir)

    target_req = paths["requirements"]
    if requirements_path.resolve() != target_req.resolve():
        target_req.write_text(
            requirements_path.read_text(encoding="utf-8"), encoding="utf-8"
        )

    write_pid_file(paths["pid_file"], task_id)

    repo_root = _repo_root()
    logs_root = paths["logs_root"]

    store = StateStore(state_dir)

    # Resolve user-provided paths once; passed to every agent via add-dir.
    form = req.get("form") or {}
    user_paths: List[Path] = []
    for key in ("trace_file", "model_dir", "framework_source_dir", "startup_log"):
        v = form.get(key)
        if v:
            user_paths.append(Path(v))

    cfg = OrchestratorConfig(
        workspace_dir=workspace_dir,
        memory_dir=paths["memory_dir"],
        validation_dir=paths["validation_dir"],
        inputs_snapshot_dir=paths["inputs_snapshot_dir"],
        repo_root=repo_root,
        state_dir=state_dir,
        logs_root=logs_root,
        max_validator_rounds=_parse_max_validator_rounds(req, default=5),
        claude_bin=claude_bin,
        model=model,
        permission_mode=permission_mode,
        extra_claude_args=list(extra_claude_args or []),
        effort=effort,
        user_paths=user_paths,
    )

    manager = make_subagent_manager(
        claude_bin=claude_bin,
        model=model,
        permission_mode=permission_mode,
        effort=effort,
        extra_add_dirs=[repo_root, logs_root, workspace_dir, *user_paths],
        snapshot_file=paths["agents_file"],
    )
    pipeline = Pipeline(req=req, store=store, cfg=cfg, manager=manager)

    print(f"[metainfer] task_id        = {task_id}")
    print(f"[metainfer] state dir      = {state_dir}")
    print(f"[metainfer] workspace dir  = {workspace_dir}")
    print(f"[metainfer] memory dir     = {paths['memory_dir']}")
    print(f"[metainfer] logs dir       = {logs_root}")
    print(f"[metainfer] user paths     = {user_paths}")

    restore_signals = install_subagent_shutdown_handlers(manager, pid_file=paths["pid_file"])

    try:
        pipeline.run()
    finally:
        restore_signals()
        clear_pid_file(paths["pid_file"])
    return 0
