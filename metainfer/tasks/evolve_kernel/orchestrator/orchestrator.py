"""Bootstrap + entry point for the evolve-kernel orchestrator .

8-phase LLM-guided iterative kernel optimization.
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
from metainfer.orchestrator.requirements import req_field_int
from metainfer.orchestrator.state import StateStore
from .pipeline import Orchestrator, OrchestratorConfig


def _task_subdirs(state_dir: Path, workspace_dir: Path) -> Dict[str, Path]:
    state_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    logs = state_dir / "logs"
    iterations_state = state_dir / "iterations"
    for p in (logs, iterations_state):
        p.mkdir(parents=True, exist_ok=True)
    return {
        "state_dir": state_dir,
        "workspace_dir": workspace_dir,
        "code_root": workspace_dir,
        "iterations_root": workspace_dir,
        "logs_root": logs,
        "iterations_state": iterations_state,
        "requirements": state_dir / "requirements.json",
        "pid_file": state_dir / "orchestrator.pid",
        "log_file": state_dir / "orchestrator.log",
        "run_file": state_dir / "run.json",
        "timeline_file": state_dir / "timeline.jsonl",
        "agents_file": state_dir / "agents.json",
    }


def run_with_requirements(
    requirements_path: Path,
    *,
    state_dir: Optional[Path] = None,
    workspace_dir: Optional[Path] = None,
    claude_bin: str = "ccb",
    model: Optional[str] = None,
    permission_mode: str = "bypassPermissions",
    max_iterations: Optional[int] = None,
    extra_claude_args: Optional[list] = None,
    effort: str = "max",
) -> int:
    if not requirements_path.exists():
        raise FileNotFoundError(f"requirements file not found: {requirements_path}")

    req: Dict[str, Any] = json.loads(requirements_path.read_text(encoding="utf-8"))
    task_id = req.get("task_id", "task")

    # Resolve dirs early — needed by both single-GPU and multi-GPU paths
    if state_dir is None:
        state_dir = Path.cwd() / "nodes" / "localhost" / ".metainfer" / "tasks" / task_id
    if workspace_dir is None:
        workspace_dir = Path.cwd() / "nodes" / "localhost" / "workspaces" / task_id
    paths = _task_subdirs(state_dir, workspace_dir)

    # Copy requirements if needed
    target_req = paths["requirements"]
    if requirements_path.resolve() != target_req.resolve():
        target_req.write_text(requirements_path.read_text(encoding="utf-8"), encoding="utf-8")

    # Also copy reference kernel from kernel_file_path to workspace
    kernel_path_str = req.get("kernel_file_path", "")
    if kernel_path_str:
        kernel_path = Path(kernel_path_str)
        if kernel_path.is_file():
            ref_dir = workspace_dir / "reference"
            ref_dir.mkdir(parents=True, exist_ok=True)
            ref_path = ref_dir / "original_kernel.py"
            if not ref_path.is_file():
                ref_path.write_text(kernel_path.read_text(encoding="utf-8"), encoding="utf-8")

    # Multi-GPU dispatch: one orchestrator spawns N GPU workers internally
    multi_gpu = req.get("multi_gpu", "no")
    if multi_gpu in ("All GPUs (auto)", "yes", "true", "1"):
        set_process_name("metainfer-orch-evolve-kernel-multi")
        print(f"[metainfer-evolve-kernel] MULTI-GPU mode")
        print(f"[metainfer-evolve-kernel] task_id        = {task_id}")
        print(f"[metainfer-evolve-kernel] state dir      = {state_dir}")
        print(f"[metainfer-evolve-kernel] workspace dir  = {workspace_dir}")

        from ._parallel import MultiGpuOrchestrator
        write_pid_file(paths["pid_file"], task_id)
        orch_multi = MultiGpuOrchestrator(
            req=req,
            state_dir=state_dir,
            workspace_dir=workspace_dir,
            claude_bin=claude_bin,
            model=model,
            permission_mode=permission_mode,
            effort=effort,
        )
        try:
            orch_multi.run()
        finally:
            clear_pid_file(paths["pid_file"])
        return 0

    set_process_name("metainfer-orch-evolve-kernel")

    write_pid_file(paths["pid_file"], task_id)

    repo_root = _repo_root()
    logs_root = paths["logs_root"]
    iterations_root = paths["code_root"]

    store = StateStore(state_dir)
    # Optional perf-test worker nodes (comma-separated string in req)
    from metainfer.orchestrator.requirements import req_field
    raw_workers = req_field(req, "worker_nodes", "") or ""
    perf_workers = [w.strip() for w in str(raw_workers).split(",") if w.strip()]

    cfg = OrchestratorConfig(
        workdir=state_dir,
        state_dir=state_dir,
        workspace_dir=workspace_dir,
        repo_root=repo_root,
        iterations_root=iterations_root,
        logs_root=logs_root,
        max_iterations=max_iterations or _extract_max_iter(req, default=20),
        claude_bin=claude_bin,
        model=model,
        permission_mode=permission_mode,
        extra_claude_args=list(extra_claude_args or []),
        perf_worker_nodes=perf_workers,
    )

    manager = make_subagent_manager(
        claude_bin=claude_bin,
        model=model,
        permission_mode=permission_mode,
        effort=effort,
        extra_add_dirs=[repo_root, logs_root],
        snapshot_file=paths["agents_file"],
    )
    orch = Orchestrator(req=req, store=store, cfg=cfg, manager=manager)

    print(f"[metainfer-evolve-kernel] task_id        = {task_id}")
    print(f"[metainfer-evolve-kernel] state dir      = {state_dir}")
    print(f"[metainfer-evolve-kernel] workspace dir  = {workspace_dir}")
    print(f"[metainfer-evolve-kernel] code dir       = {iterations_root}")
    print(f"[metainfer-evolve-kernel] logs dir       = {logs_root}")
    print(f"[metainfer-evolve-kernel] max iterations = {cfg.max_iterations}")

    restore_signals = install_subagent_shutdown_handlers(manager, pid_file=paths["pid_file"])

    try:
        orch.run()
    finally:
        restore_signals()
        clear_pid_file(paths["pid_file"])
    return 0


def _extract_max_iter(req: Dict[str, Any], default: int = 20) -> int:
    return req_field_int(req, "max_iterations", default=default)
