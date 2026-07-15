"""Bootstrap + entry point for the gen-infer-framework-cpp orchestrator.

This is the per-task subprocess the WebUI spawns when
``requirements.json.task_type == "gen-infer-framework-cpp"``. It reads the
requirements, sets up the state directory, boots a SubAgentManager, and
hands control to the ABCDEF iteration loop in :mod:`.pipeline`.

The shared PID / signal / process-name / SubAgentManager machinery lives
in :mod:`metainfer.orchestrator._bootstrap` — every orchestrator package
uses the same lifecycle. This file holds only the C++ generator-specific
workspace schema, hardware discovery, OrchestratorConfig wiring, and
iteration-loop entry.
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
from metainfer.orchestrator.token_budget import TokenBudget
from .hardware_discovery import (
    configure_assigned_devices,
    discover_hardware_profile,
    prompt_hardware_summary,
    write_hardware_profile,
)
from .pipeline import Orchestrator, OrchestratorConfig
from .task_types import is_cpp_framework_task


# Knowledge base for this task type. Lives inside the task package so the
# whole thing is self-contained — every prompt that tells an agent "consult
# the notebooks/ knowledge base" resolves to this path.
_NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"


# --------------------------------------------------------------------------- #
# Per-task state_dir layout for the gen-infer-framework-cpp task type
# --------------------------------------------------------------------------- #
#
#   <state_dir>/
#   ├── requirements.json       # frozen inputs (read at start)
#   ├── hardware_profile.json   # read-only host facts (C++ task only)
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


def _task_subdirs(state_dir: Path, workspace_dir: Path) -> Dict[str, Path]:
    """Return the canonical sub-paths for this task.

    ``state_dir`` holds metadata + logs (run.json, timeline.jsonl,
    agents.json, requirements.json, orchestrator.pid/log, iterations/*.json,
    logs/<NNN>/). ``workspace_dir`` holds generated iteration code
    (001/, 002/, ...). Both are created on first call."""
    state_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    logs = state_dir / "logs"
    iterations_state = state_dir / "iterations"
    for p in (logs, iterations_state):
        p.mkdir(parents=True, exist_ok=True)
    return {
        "state_dir": state_dir,
        "workspace_dir": workspace_dir,
        "code_root": workspace_dir,  # iteration code goes directly under workspace/
        "logs_root": logs,
        "iterations_state": iterations_state,
        "requirements": state_dir / "requirements.json",
        "hardware_profile": state_dir / "hardware_profile.json",
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

    Artifacts split between ``state_dir`` (metadata + logs) and
    ``workspace_dir`` (generated iteration code). Defaults are derived
    under ``<cwd>/nodes/<host>/`` so ad-hoc CLI usage works without
    the WebUI; the WebUI always passes both explicitly.
    """
    if not requirements_path.exists():
        raise FileNotFoundError(f"requirements file not found: {requirements_path}")

    req: Dict[str, Any] = json.loads(
        requirements_path.read_text(encoding="utf-8")
    )
    task_type = str(req.get("task_type") or "")
    if not is_cpp_framework_task(task_type):
        raise ValueError(
            "gen_infer_framework_cpp can only run task_type "
            f"'gen-infer-framework-cpp', got {task_type!r}"
        )
    task_id = req.get("task_id", "task")

    # Stamp the kernel task name ASAP so process scans pick us up even
    # if we hang during initialization. Format: "metainfer-orch" (kernel
    # truncates to 15 chars anyway; this is already 14).
    set_process_name("metainfer-orch")

    # Resolve state_dir + workspace_dir. Defaults derive from the WebUI's
    # path module so an ad-hoc CLI invocation lands in the same place
    # the WebUI would have put it.
    if state_dir is None or workspace_dir is None:
        from metainfer.server import paths as _web_paths
        if state_dir is None:
            state_dir = _web_paths.task_dir(task_id)
        if workspace_dir is None:
            workspace_dir = _web_paths.workspace_dir(task_id)
    paths = _task_subdirs(state_dir, workspace_dir)

    # Copy requirements into state_dir if invoked from elsewhere so the
    # task is fully self-contained (WebUI re-reads it from there).
    target_req = paths["requirements"]
    if requirements_path.resolve() != target_req.resolve():
        target_req.write_text(
            requirements_path.read_text(encoding="utf-8"), encoding="utf-8"
        )

    if is_cpp_framework_task(req.get("task_type", "")):
        # This is pure environment validation and must happen before the PID
        # stamp so an invalid direct-CLI value cannot leave stale task state.
        configure_assigned_devices(req)

    # Stamp PID before the bounded hardware commands so the WebUI sees the
    # orchestrator as alive while discovery runs.
    write_pid_file(paths["pid_file"], task_id)

    # The C++ generator needs concrete compiler/device facts, not just a
    # marketing-model selection.  Probe here so the detector inherits the
    # exact SSH login, scheduler visibility and device permissions that the
    # generated server will inherit.  The full evidence stays beside task
    # state; prompts receive a bounded summary through requirements.json.
    if is_cpp_framework_task(req.get("task_type", "")):
        hardware_profile = discover_hardware_profile(req)
        write_hardware_profile(paths["hardware_profile"], hardware_profile)
        req["hardware_profile_path"] = str(paths["hardware_profile"])
        req["hardware_profile"] = prompt_hardware_summary(hardware_profile)
        target_req.write_text(
            json.dumps(req, indent=2, sort_keys=True), encoding="utf-8"
        )

    repo_root = _repo_root()
    notebooks_dir = _NOTEBOOKS_DIR
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
        #   - notebooks_dir: this task's read-only knowledge base
        #     (metainfer/tasks/gen_infer_framework_cpp/notebooks/), which every
        #     prompt tells agents to consult
        #   - repo_root: so prompts can reference paths under the install
        #   - workspace_dir: where iteration code (001/, 002/, ...) lives —
        #     agents write here, the sandbox needs to allow it explicitly
        #   - logs_root: where reviewer writes review.md and where the
        #     prev-iter diagnostic snapshot lives
        extra_add_dirs=[
            notebooks_dir,
            repo_root,
            workspace_dir,
            logs_root,
            state_dir,
        ],
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
    orch = Orchestrator(req=req, store=store, cfg=cfg, manager=manager,
                        budget=budget)

    print(f"[metainfer] task_id        = {task_id}")
    print(f"[metainfer] state dir      = {state_dir}")
    print(f"[metainfer] workspace dir  = {workspace_dir}")
    print(f"[metainfer] code dir       = {iterations_root}")
    print(f"[metainfer] logs dir       = {logs_root}")
    print(f"[metainfer] notebooks      = {notebooks_dir}")
    if is_cpp_framework_task(req.get("task_type", "")):
        validation = req["hardware_profile"].get("validation", {})
        print(f"[metainfer] hardware       = {paths['hardware_profile']}")
        print(f"[metainfer] hw validation  = {validation.get('status', 'unknown')}")
        for warning in validation.get("warnings", []):
            print(f"[metainfer] hardware warning: {warning}")
        for blocker in validation.get("blockers", []):
            print(f"[metainfer] hardware blocker: {blocker}")
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


def _build_budget(state_dir: Path, req: Dict[str, Any]) -> Optional[TokenBudget]:
    """Construct the per-task :class:`TokenBudget` from req + env.

    Resolution order for the soft cost limit (first match wins):
      1. ``METAINFER_TOKEN_BUDGET_COST_USD`` env var
      2. ``requirements.json::token_budget.max_cost_usd`` (nested object —
         direct requirements.json edit)
      3. ``requirements.json::token_budget_max_cost_usd`` (flat scalar —
         what the WebUI new-task form writes when the user fills the
         "Cost cap" field)
      4. None — budget circuit breaker disabled

    Hard limit follows the same cascade with the ``_hard`` suffix.
    """
    import os

    tb_cfg = (req.get("token_budget")
              or req.get("answers", {}).get("token_budget")
              or {})
    if not isinstance(tb_cfg, dict):
        tb_cfg = {}

    def _resolve_float(env_key: str, conf_key: str,
                       flat_key: Optional[str] = None) -> Optional[float]:
        env_v = os.environ.get(env_key)
        if env_v:
            try:
                return float(env_v)
            except ValueError:
                pass
        # nested object first
        v = tb_cfg.get(conf_key)
        if v is None and flat_key:
            v = req.get(flat_key)
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    soft = _resolve_float("METAINFER_TOKEN_BUDGET_COST_USD", "max_cost_usd",
                          flat_key="token_budget_max_cost_usd")
    hard = _resolve_float("METAINFER_TOKEN_BUDGET_COST_USD_HARD",
                          "max_cost_usd_hard",
                          flat_key="token_budget_max_cost_usd_hard")
    if soft is None and hard is None:
        return None
    return TokenBudget(
        state_dir,
        max_cost_usd=soft,
        max_cost_usd_hard=hard,
    )
