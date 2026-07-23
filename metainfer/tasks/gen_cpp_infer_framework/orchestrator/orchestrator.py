"""Bootstrap + entry point for the gen-cpp-infer-framework orchestrator.

This is the per-task subprocess the WebUI spawns when
``requirements.json.task_type == "gen-cpp-infer-framework"``. It reads the
requirements, sets up the state directory, boots a SubAgentManager, and
hands control to the ABCDEF iteration loop in :mod:`.pipeline`.

The shared PID / signal / process-name / SubAgentManager machinery lives
in :mod:`metainfer.orchestrator._bootstrap` — every orchestrator package
uses the same lifecycle. This file holds only the gen-cpp-infer-framework-
specific bits: the separate generated-code workspace and metadata/log
schema, the OrchestratorConfig wiring, and the iteration-loop entry.
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
from metainfer.orchestrator.token_budget import TokenBudget, resolve_budget_limits
from .capabilities import CapabilityResolutionError, freeze_resolved_requirements
from .pipeline import Orchestrator, OrchestratorConfig


# Each task owns its knowledge base.  The C++ task starts with a small
# placeholder and can grow its own hardware, build, and profiling contracts
# without mixing them into the Python-oriented inference task's notebooks.
_NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"


# --------------------------------------------------------------------------- #
# Per-task state_dir layout for the gen-cpp-infer-framework task type
# --------------------------------------------------------------------------- #
#
#   <state_dir>/               # hidden metadata owned by the server
#   ├── requirements.json
#   ├── resolved_requirements.json  # frozen capability/parameter contract
#   ├── orchestrator.{pid,log}
#   ├── run.json / timeline.jsonl / agents.json
#   ├── iterations/<n>.json     # per-iteration records
#   └── logs/<NNN>/             # prompts, transcripts, oracle logs
#
#   <workspace_dir>/           # visible generated artifacts
#   ├── 001/
#   └── 002/


def _task_subdirs(state_dir: Path, workspace_dir: Path) -> Dict[str, Path]:
    """Return the canonical metadata and generated-artifact paths."""
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
        "logs_root": logs,
        "iterations_state": iterations_state,
        "requirements": state_dir / "requirements.json",
        "resolved_requirements": state_dir / "resolved_requirements.json",
        "stable_candidate": state_dir / "stable_candidate.json",
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
    """Per-task orchestrator entry point.

    Reads ``requirements.json`` from ``requirements_path`` (or from
    ``<state_dir>/requirements.json`` if ``state_dir`` is given and
    ``requirements_path`` is the same file), runs the ABCDEF loop to
    completion, and exits.

    Metadata and logs go under ``state_dir``; generated iteration trees go
    under the parallel ``workspace_dir``. The WebUI passes both explicitly.
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

    if state_dir is None or workspace_dir is None:
        from metainfer.server import paths as _server_paths
        if state_dir is None:
            state_dir = _server_paths.task_dir(task_id)
        if workspace_dir is None:
            workspace_dir = _server_paths.workspace_dir(task_id)
    paths = _task_subdirs(state_dir, workspace_dir)

    # Copy requirements into state_dir if invoked from elsewhere so the
    # task is fully self-contained (WebUI re-reads it from there).
    target_req = paths["requirements"]
    if requirements_path.resolve() != target_req.resolve():
        target_req.write_text(
            requirements_path.read_text(encoding="utf-8"), encoding="utf-8"
        )

    store = StateStore(state_dir)

    # Compile the user-facing form fields into a deterministic capability
    # contract before A can run. Restarts reuse the frozen snapshot; changing
    # requirements.json underneath a live task is rejected by the compiler.
    try:
        resolved_req = freeze_resolved_requirements(req, state_dir)
    except CapabilityResolutionError as exc:
        run_status, _is_resume = store.init_or_resume(task_id)
        note = f"requirements rejected ({exc.field}): {exc}"
        store.update_run(
            current_phase="finished",
            finished=True,
            final_status="stopped",
            notes=[*run_status.notes, note],
        )
        store.append_timeline(
            "requirements_rejected",
            {"field": exc.field, "reason": str(exc)},
        )
        print(f"[metainfer] {note}")
        return 2
    runtime_req = dict(req)
    runtime_req["resolved_requirements"] = resolved_req

    # Stamp PID file BEFORE doing anything heavy so the WebUI sees us
    # alive immediately.
    write_pid_file(paths["pid_file"], task_id)

    repo_root = _repo_root()
    notebooks_dir = _NOTEBOOKS_DIR
    logs_root = paths["logs_root"]
    iterations_root = paths["code_root"]

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

    # Build the per-task token/cost budget. Reads ``token_budget`` block
    # from requirements.json (e.g. {"max_cost_usd": 50.0}); env override
    # METAINFER_TOKEN_BUDGET_COST_USD wins over both. When no limit is
    # set, budget is None and the circuit breaker is inert.
    budget = _build_budget(state_dir, req)
    if budget is not None:
        # Mirror every record into timeline.jsonl so the WebUI can draw
        # a live cost-over-time graph without re-parsing events.jsonl.
        # We attach this AFTER construction (the budget may already have
        # loaded historical records from disk; those don't fire the
        # callback, only new records do).
        store_for_cb = store
        budget._on_recorded = lambda rec, snap: store_for_cb.append_timeline(
            "token_usage",
            {
                "agent": rec.agent,
                "source": rec.source,
                "phase": rec.phase,
                "input_tokens": rec.input_tokens,
                "output_tokens": rec.output_tokens,
                "cache_read_input_tokens": rec.cache_read_input_tokens,
                "cost_usd": rec.total_cost_usd,
                "running_total_cost_usd": snap.total_cost_usd,
                "running_total_input_tokens": snap.total_input_tokens,
                "running_total_output_tokens": snap.total_output_tokens,
                "agent_count": snap.agent_count,
                "exhausted": snap.exhausted,
            },
        )

    manager = make_subagent_manager(
        claude_bin=claude_bin,
        model=model,
        permission_mode=permission_mode,
        effort=effort,
        # Sub-agent prompts reference these paths outside the iteration dir:
        #   - notebooks_dir: the shared inference-framework knowledge base,
        #     which every prompt tells agents to consult
        #   - repo_root: so prompts can reference paths under the install
        #   - workspace_dir: where generated iteration source trees live
        #   - logs_root: where reviewer writes review.md and where the
        #     prev-iter diagnostic snapshot lives
        extra_add_dirs=[notebooks_dir, repo_root, workspace_dir, logs_root, state_dir],
        snapshot_file=paths["agents_file"],
        budget=budget,
    )
    # Wire the hard-exhausted callback NOW that the manager exists.
    # When the hard threshold is crossed, every in-flight agent gets
    # SIGTERM'd via the manager's process-group kill (soft-abort policy
    # for the soft threshold; hard threshold = "stop bleeding right
    # now"). The callback fires exactly once per task lifetime.
    if budget is not None and budget.max_cost_usd_hard is not None:
        budget._on_hard = lambda: manager.shutdown()
    orch = Orchestrator(req=runtime_req, store=store, cfg=cfg, manager=manager,
                        budget=budget)

    print(f"[metainfer] task_id        = {task_id}")
    print(f"[metainfer] state dir      = {state_dir}")
    print(f"[metainfer] workspace dir  = {workspace_dir}")
    print(f"[metainfer] code dir       = {iterations_root}")
    print(f"[metainfer] logs dir       = {logs_root}")
    print(f"[metainfer] notebooks      = {notebooks_dir}")
    print(f"[metainfer] resolved req   = {paths['resolved_requirements']}")
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
    """Read ``max_iterations`` via the shared helper. See
    :mod:`metainfer.orchestrator.requirements`."""
    return req_field_int(req, "max_iterations", default=default)


def _build_budget(state_dir: Path, req: Dict[str, Any]) -> Optional[TokenBudget]:
    """Construct the per-task :class:`TokenBudget`.

    Limit resolution is delegated to :func:`token_budget.resolve_budget_limits`
    so all task types share one source of truth: ``token_budget.json::config``
    wins over the ``requirements.json`` seed (which is only consulted on
    first boot before the runtime file exists). See that helper's docstring
    for the full cascade + the bug this prevents.
    """
    soft, hard = resolve_budget_limits(state_dir, req)
    if soft is None and hard is None:
        return None
    return TokenBudget(
        state_dir,
        max_cost_usd=soft,
        max_cost_usd_hard=hard,
    )
