"""Bootstrap + entry point for the gen-cpp-infer-framework orchestrator.

This is the per-task subprocess the WebUI spawns when
``requirements.json.task_type == "gen-cpp-infer-framework"``. It reads the
requirements, sets up the state directory, boots a SubAgentManager, and
hands control to the ABCDEF iteration loop in :mod:`.pipeline`.

The shared PID / signal / process-name / SubAgentManager machinery lives
in :mod:`metainfer.orchestrator._bootstrap` — every orchestrator package
uses the same lifecycle. This file holds only the gen-cpp-infer-framework-
specific bits: the ``code/`` + ``logs/`` + ``iterations/`` subdir
schema, the OrchestratorConfig wiring, and the iteration-loop entry.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from ..._bootstrap import (
    clear_pid_file,
    install_subagent_shutdown_handlers,
    make_subagent_manager,
    set_process_name,
    write_pid_file,
)
from ...paths import notebooks_dir as _notebooks_dir
from ...paths import repo_root as _repo_root
from ...state import StateStore
from .pipeline import Orchestrator, OrchestratorConfig


# --------------------------------------------------------------------------- #
# Per-task state_dir layout for the gen-cpp-infer-framework task type
# --------------------------------------------------------------------------- #
#
#   <state_dir>/
#   ├── requirements.json       # frozen inputs (read at start)
#   ├── orchestrator.pid        # PID of the running orchestrator (or last)
#   ├── orchestrator.log        # stdout+stderr, for debugging
#   ├── run.json                # RunStatus (live phase / iteration / outcome)
#   ├── timeline.jsonl          # append-only event log
#   ├── iterations/<n>.json     # per-iteration records
#   ├── agents.json             # SubAgentManager snapshot (live agents)
#   ├── code/                   # iteration CODE root (visible to user)
#   │   ├── 001/
#   │   └── 002/
#   └── logs/                   # per-iteration agent/oracle/server logs
#       ├── 001/
#       └── 002/


def _task_subdirs(state_dir: Path) -> Dict[str, Path]:
    """Return the canonical sub-paths under ``state_dir``. All directories
    are created on first call."""
    state_dir.mkdir(parents=True, exist_ok=True)
    code = state_dir / "code"
    logs = state_dir / "logs"
    iterations_state = state_dir / "iterations"
    for p in (code, logs, iterations_state):
        p.mkdir(parents=True, exist_ok=True)
    return {
        "state_dir": state_dir,
        "code_root": code,
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
    claude_bin: str = "ccb",
    model: Optional[str] = None,
    permission_mode: str = "bypassPermissions",
    max_iterations: Optional[int] = None,
    extra_claude_args: Optional[list] = None,
    effort: str = "max",
) -> int:
    """Per-task orchestrator entry point.

    Reads ``requirements.json`` from ``requirements_path`` (or from
    ``<state_dir>/requirements.json`` if ``state_dir`` is given and
    ``requirements_path`` is the same file), runs the ABCDEF loop to
    completion, and exits.

    All artifacts go under ``state_dir`` (or a default location derived
    from CWD if not provided — kept for ad-hoc CLI debugging).
    """
    if not requirements_path.exists():
        raise FileNotFoundError(f"requirements file not found: {requirements_path}")

    req: Dict[str, Any] = json.loads(
        requirements_path.read_text(encoding="utf-8")
    )
    task_id = req.get("task_id", "task")

    # Stamp the kernel task name ASAP so process scans pick us up even
    # if we hang during initialization. Format: "metainfer-orch" (kernel
    # truncates to 15 chars anyway; this is already 14).
    set_process_name("metainfer-orch")

    # Resolve state_dir. Default: <cwd>/.metainfer/tasks/<task_id>/ — keeps
    # ad-hoc CLI usage working without the WebUI. The WebUI always passes
    # an explicit state_dir under ~/.metainfer/tasks/<id>/.
    if state_dir is None:
        state_dir = Path.cwd() / ".metainfer" / "tasks" / task_id
    paths = _task_subdirs(state_dir)

    # Copy requirements into state_dir if invoked from elsewhere so the
    # task is fully self-contained (WebUI re-reads it from there).
    target_req = paths["requirements"]
    if requirements_path.resolve() != target_req.resolve():
        target_req.write_text(
            requirements_path.read_text(encoding="utf-8"), encoding="utf-8"
        )

    # Stamp PID file BEFORE doing anything heavy so the WebUI sees us
    # alive immediately.
    write_pid_file(paths["pid_file"], task_id)

    repo_root = _repo_root()
    notebooks_dir = _notebooks_dir()
    logs_root = paths["logs_root"]
    iterations_root = paths["code_root"]

    store = StateStore(state_dir)
    cfg = OrchestratorConfig(
        workdir=state_dir,
        repo_root=repo_root,
        notebooks_dir=notebooks_dir,
        iterations_root=iterations_root,
        logs_root=logs_root,
        state_dir=state_dir,
        max_iterations=max_iterations or _extract_max_iter(req, default=20),
        claude_bin=claude_bin,
        model=model,
        permission_mode=permission_mode,
        extra_claude_args=list(extra_claude_args or []),
    )

    manager = make_subagent_manager(
        claude_bin=claude_bin,
        model=model,
        permission_mode=permission_mode,
        effort=effort,
        # Sub-agent prompts reference these paths outside the iteration dir:
        #   - notebooks/: read-only knowledge base every prompt tells agents
        #     to consult
        #   - repo_root: so prompts can reference paths under the install
        #   - logs_root: where reviewer writes review.md and where the
        #     prev-iter diagnostic snapshot lives
        extra_add_dirs=[notebooks_dir, repo_root, logs_root],
        snapshot_file=paths["agents_file"],
    )
    orch = Orchestrator(req=req, store=store, cfg=cfg, manager=manager)

    print(f"[metainfer] task_id        = {task_id}")
    print(f"[metainfer] state dir      = {state_dir}")
    print(f"[metainfer] code dir       = {iterations_root}")
    print(f"[metainfer] logs dir       = {logs_root}")
    print(f"[metainfer] notebooks      = {notebooks_dir}")
    print(f"[metainfer] orchestrator starting; WebUI is in a separate process.")

    restore_signals = install_subagent_shutdown_handlers(
        manager, pid_file=paths["pid_file"]
    )

    try:
        orch.run()
    finally:
        restore_signals()
        clear_pid_file(paths["pid_file"])
    return 0


def _extract_max_iter(req: Dict[str, Any], default: int = 20) -> int:
    """Read max_iterations from requirements, preferring top-level.

    The interview writes ``max_iterations`` as a TOP-LEVEL field on
    requirements.json (alongside ``target_model``, ``target_hardware``,
    etc.). Top-level takes precedence; ``answers.`` is checked as a
    back-compat fallback for older requirements files.
    """
    v = req.get("max_iterations")
    if v is None:
        v = req.get("answers", {}).get("max_iterations")
    if v is None:
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default
