"""Bootstrap + entry point for the knowledge-evolution orchestrator.

This is the per-task subprocess the WebUI spawns when
``requirements.json.task_type == "knowledge-evolution"``. It reads the
requirements, sets up the state directory, boots a SubAgentManager, and
hands control to the 4-phase evolution loop in :mod:`.pipeline`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from metainfer.orchestrator._bootstrap import (
    clear_pid_file,
    install_subagent_shutdown_handlers,
    make_subagent_manager,
    set_process_name,
    write_pid_file,
)
from metainfer.orchestrator.paths import repo_root as _repo_root
from metainfer.orchestrator.state import StateStore

from .pipeline import EvolutionConfig, EvolutionOrchestrator


def _task_subdirs(state_dir: Path) -> Dict[str, Path]:
    """Create and return the standard subdirectory layout for a task.

    All directories are created with ``parents=True, exist_ok=True``.
    """
    dirs = {
        "state_dir": state_dir,
        "code_root": state_dir / "code",
        "logs_root": state_dir / "logs",
        "iterations_state": state_dir / "iterations",
        "requirements": state_dir / "requirements.json",
        "pid_file": state_dir / "orchestrator.pid",
        "log_file": state_dir / "orchestrator.log",
        "run_file": state_dir / "run.json",
        "timeline_file": state_dir / "timeline.jsonl",
        "agents_file": state_dir / "agents.json",
    }
    for p in dirs.values():
        if p.suffix:  # file
            p.parent.mkdir(parents=True, exist_ok=True)
        else:  # directory
            p.mkdir(parents=True, exist_ok=True)
    return dirs


def _extract_max_iter(req: Dict[str, Any], default: int = 20) -> int:
    """Extract max_iterations from requirements, with fallbacks."""
    try:
        val = req.get("max_iterations")
        if val is None:
            val = req.get("form", {}).get("max_iterations")
        if val is None:
            return default
        return int(val)
    except (TypeError, ValueError):
        return default


def _extract_max_verify_attempts(req: Dict[str, Any]) -> int:
    """Extract max_verify_attempts from requirements, with fallbacks."""
    try:
        val = req.get("max_verify_attempts")
        if val is None:
            val = req.get("form", {}).get("max_verify_attempts")
        if val is None:
            return 3
        return int(val)
    except (TypeError, ValueError):
        return 3


def run_with_requirements(
    requirements_path: Path,
    state_dir: Optional[Path] = None,
    workspace_dir: Optional[Path] = None,
    claude_bin: str = "ccb",
    model: Optional[str] = None,
    permission_mode: str = "bypassPermissions",
    max_iterations: Optional[int] = None,
    max_verify_attempts: Optional[int] = None,
    extra_claude_args: Optional[list] = None,
    effort: str = "max",
) -> int:
    """Bootstrap and run the knowledge-evolution orchestrator.

    This is the main entry point called by the CLI ``run`` subcommand.
    It reads ``requirements.json``, sets up directories, creates the
    EvolutionOrchestrator, and executes the state machine loop.

    Returns:
        Exit code: 0 on success, non-zero on failure.
    """
    # 1. Read requirements
    req: Dict[str, Any] = {}
    try:
        req = json.loads(requirements_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"ERROR: failed to read requirements.json: {exc}", flush=True)
        return 1

    task_id = req.get("task_id", "task")

    # 2. Set process name
    set_process_name("metainfer-orch")

    # 3. Resolve state_dir
    if state_dir is None:
        state_dir = Path.cwd() / ".metainfer" / "tasks" / task_id
    state_dir.mkdir(parents=True, exist_ok=True)

    # 4. Build subdirectories
    paths = _task_subdirs(state_dir)

    # 5. Copy requirements.json into state_dir if paths differ
    req_dest = paths["requirements"]
    if requirements_path.resolve() != req_dest.resolve():
        req_dest.parent.mkdir(parents=True, exist_ok=True)
        req_dest.write_text(json.dumps(req, indent=2, ensure_ascii=False))

    # 6. Write PID file
    write_pid_file(paths["pid_file"], task_id)

    # 7. Resolve paths for EvolutionConfig
    repo_root = _repo_root()
    notebooks_dir = repo_root / "notebooks"
    logs_root = paths["logs_root"]
    iterations_root = paths["code_root"]

    # 8. Resolve config overrides
    resolved_max_iter = max_iterations
    if resolved_max_iter is None:
        resolved_max_iter = _extract_max_iter(req)
    resolved_max_verify = max_verify_attempts
    if resolved_max_verify is None:
        resolved_max_verify = _extract_max_verify_attempts(req)
    if extra_claude_args is None:
        extra_claude_args = []

    # 9. Create StateStore
    store = StateStore(state_dir)

    # 10. Create config
    cfg = EvolutionConfig(
        workdir=workspace_dir or (state_dir / "workspace"),
        repo_root=repo_root,
        notebooks_dir=notebooks_dir,
        iterations_root=iterations_root,
        state_dir=state_dir,
        logs_root=logs_root,
        max_iterations=resolved_max_iter,
        max_verify_attempts=resolved_max_verify,
        claude_bin=claude_bin,
        model=model,
        permission_mode=permission_mode,
        extra_claude_args=extra_claude_args,
    )

    # 11. Bootstrap SubAgentManager
    manager = make_subagent_manager(
        claude_bin=claude_bin,
        model=model,
        permission_mode=permission_mode,
        effort=effort,
        extra_add_dirs=[notebooks_dir, cfg.workdir, logs_root],
        snapshot_file=paths["agents_file"],
    )

    # 12. Create orchestrator
    orch = EvolutionOrchestrator(req=req, store=store, cfg=cfg, manager=manager)

    print(f"[knowledge-evolution] Task: {task_id}", flush=True)
    print(f"[knowledge-evolution] State dir: {state_dir}", flush=True)
    print(f"[knowledge-evolution] Workspace dir: {cfg.workdir}", flush=True)
    print(f"[knowledge-evolution] Notebooks dir: {notebooks_dir}", flush=True)
    print(f"[knowledge-evolution] Max iterations: {resolved_max_iter}", flush=True)
    print(f"[knowledge-evolution] Max verify attempts: {resolved_max_verify}", flush=True)
    print(f"[knowledge-evolution] Claude binary: {claude_bin}", flush=True)
    print(f"[knowledge-evolution] Permission mode: {permission_mode}", flush=True)
    print(f"[knowledge-evolution] Effort: {effort}", flush=True)
    print("[knowledge-evolution] Starting 4-phase evolution loop...", flush=True)

    # 13. Run the state machine
    restore_signals = install_subagent_shutdown_handlers(
        manager, pid_file=paths["pid_file"]
    )
    try:
        orch.run()
    except KeyboardInterrupt:
        print("[knowledge-evolution] Interrupted by user.", flush=True)
        store.update_run(
            current_phase="finished",
            final_status="aborted",
            final_message="interrupted by user",
        )
        return 130
    except Exception as exc:
        print(f"[knowledge-evolution] FATAL: {exc}", flush=True, file=__import__("sys").stderr)
        store.update_run(
            current_phase="finished",
            final_status="failed",
            final_message=f"unhandled exception: {exc}",
        )
        return 1
    finally:
        restore_signals()
        clear_pid_file(paths["pid_file"])

    print("[knowledge-evolution] Evolution loop complete.", flush=True)
    return 0
