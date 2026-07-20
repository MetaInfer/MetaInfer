"""Orchestrator launcher — reads requirements, constructs the pipeline, runs it.

This is the "glue" module that:

1. Resolves ``state_dir`` and ``workspace_dir`` (accepting overrides from CLI).
2. Sets up sub-directories (e.g. ``step0/``, ``iterations/`` in workspace).
3. Initialises or resumes the ``StateStore`` (run.json, timeline, iterations).
4. Constructs a :class:`metainfer.orchestrator.subagent_manager.SubAgentManager`.
5. Instantiates the pipeline and calls ``run()``.

It imports shared infrastructure from :mod:`metainfer.orchestrator`:

- :func:`metainfer.orchestrator._bootstrap.setup_orchestrator` — PID file,
  signal handlers, SubAgentManager construction.
- :class:`metainfer.orchestrator.state.StateStore` — file-based run state.
"""

from __future__ import annotations

from pathlib import Path


def _task_subdirs(state_dir: Path, workspace_dir: Path) -> dict[str, Path]:
    """Create and return the task-specific sub-directories.

    ``state_dir`` holds metadata (.metainfer/tasks/<id>/).
    ``workspace_dir`` holds generated artefacts that the user interacts with.
    """
    # Example: create per-step dirs under workspace.
    # In a real task, replace these with your own structure.
    subdirs: dict[str, Path] = {}
    for name in ("step1", "step2"):
        p = workspace_dir / name
        p.mkdir(parents=True, exist_ok=True)
        subdirs[name] = p
    return subdirs


def run_with_requirements(
    req: dict,
    *,
    state_dir: Path,
    workspace_dir: Path,
    iter_limit: int = 10,
    dry_run: bool = False,
) -> int:
    """Main entry point. Returns exit code (0 = success, 1 = error)."""
    task_id = req["task_id"]
    task_type = req["task_type"]

    # Resolve sub-directories.
    paths = _task_subdirs(state_dir, workspace_dir)

    # In a real task, bootstrap here:
    #
    #   from metainfer.orchestrator._bootstrap import setup_orchestrator
    #   from metainfer.orchestrator.state import StateStore
    #   from .pipeline import Pipeline
    #
    #   store = StateStore(state_dir)
    #   run, is_resume = store.init_or_resume(task_id)
    #   agent_manager = setup_orchestrator(state_dir, task_id, [workspace_dir])
    #   pipeline = Pipeline(store, agent_manager, paths, req)
    #   try:
    #       pipeline.run(iter_limit=iter_limit, is_resume=is_resume)
    #   finally:
    #       agent_manager.shutdown()

    print(f"[example] task {task_id} ({task_type}) ready; "
          f"state_dir={state_dir} workspace_dir={workspace_dir} "
          f"iter_limit={iter_limit} dry_run={dry_run}")
    return 0
