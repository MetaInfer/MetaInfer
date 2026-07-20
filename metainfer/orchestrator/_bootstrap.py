"""Shared bootstrap helpers for per-task-type orchestrators.

Each task package lives in its own subpackage under
``metainfer/tasks/<task_pkg>/orchestrator/`` and owns its pipeline
logic. But every orchestrator subprocess has the same lifecycle:

  1. Read ``requirements.json``
  2. Resolve + mkdir the ``state_dir`` layout (task-type-specific subdirs)
  3. Stamp the PID file
  4. Install SIGTERM/SIGINT handlers that tear down SubAgentManager
  5. Construct a SubAgentManager
  6. Run the pipeline
  7. Clear the PID file on exit

This module factors out steps 3, 4, 5 (and the kernel process-name helper
used by 1). The orchestrator-specific pieces (subdir schema, pipeline
entry, extra add-dirs) stay in each orchestrator package.

Keeping these helpers out of ``state.py`` (the data layer) keeps that
module pure-storage — no subprocess / signal concerns mixed in.
"""

from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .subagent_manager import SubAgentManager


# --------------------------------------------------------------------------- #
# Process naming
# --------------------------------------------------------------------------- #
#
# The orchestrator is launched with a friendly argv[0] (see launcher.py —
# ``args[0]`` is set to e.g. "metainfer-orchestrator" while the actual
# binary executed is Python). That alone makes `ps aux` show the friendly
# name.
#
# We additionally use prctl(PR_SET_NAME) to set the *kernel* task name
# (visible in /proc/<pid>/comm and the default `ps`, `top`, `htop` views).
# The kernel truncates comm to 15 chars (TASK_COMM_LEN), so callers should
# pick names that fit (e.g. "metainfer-orch" rather than
# "metainfer-orchestrator").
#
# Why both? Different process-table tools look at different fields:
#   - `ps aux` reads /proc/<pid>/cmdline → sees argv[0] = friendly name
#   - `ps`, `top`, `htop` default view reads /proc/<pid>/comm → sees 15-char comm
#   - kill/pkill -f matches against either, depending on flags
# Setting both means a `ps -e | grep metainfer` from anywhere works.

def set_process_name(name: str) -> None:
    """Set the kernel task name (``/proc/self/comm``) via prctl. Also
    truncates to 15 chars (TASK_COMM_LEN). No-op on non-Linux."""
    name_b = name.encode("utf-8", errors="replace")[:15]
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        # prctl(int option, ...) — PR_SET_NAME = 15 on Linux.
        PR_SET_NAME = 15
        rc = libc.prctl(PR_SET_NAME, ctypes.c_char_p(name_b), 0, 0, 0)
        if rc != 0:  # pragma: no cover - best-effort
            pass
    except (OSError, ValueError):
        # Non-Linux or libc not found. Not fatal — argv[0] still works.
        pass


# --------------------------------------------------------------------------- #
# PID file management
# --------------------------------------------------------------------------- #

def write_pid_file(pid_file: Path, task_id: str) -> None:
    """Stamp the current PID + task_id so the WebUI can detect a live
    orchestrator and offer kill/restart controls."""
    payload = {
        "pid": os.getpid(),
        "task_id": task_id,
        "started_at": time.time(),
    }
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = pid_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(pid_file)


def clear_pid_file(pid_file: Path) -> None:
    """Mark the PID file as 'exited' rather than deleting it — the WebUI
    wants the last-known PID + finished_at for status display, and a
    clean exit should leave a clear signal."""
    try:
        data = json.loads(pid_file.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        data = {}
    data["pid"] = None
    data["finished_at"] = time.time()
    tmp = pid_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(pid_file)


# --------------------------------------------------------------------------- #
# Signal handler installation
# --------------------------------------------------------------------------- #

def install_subagent_shutdown_handlers(
    manager: SubAgentManager,
    pid_file: Optional[Path] = None,
) -> Callable[[], None]:
    """Install SIGTERM/SIGINT handlers that kill sub-agents before we die.

    Without this, ccb sub-agent children escape (each is its own process
    group leader via start_new_session=True) and leak as orphans.

    Returns a callable that restores the previous handlers (call from
    a ``finally:`` block).

    If ``pid_file`` is given, the handler also clears it before exiting
    so the WebUI sees the orchestrator as stopped even if it was killed
    mid-run.
    """
    def _on_signal(signum, _frame):
        try:
            manager.shutdown()
        except Exception:  # noqa: BLE001 — best-effort cleanup
            pass
        if pid_file is not None:
            try:
                clear_pid_file(pid_file)
            except Exception:  # noqa: BLE001
                pass
        # Use _exit so we don't run atexit hooks twice (we already
        # cleaned up). 143 = 128 + SIGTERM(15); 130 = 128 + SIGINT(2).
        os._exit(143 if signum == signal.SIGTERM else 130)

    previous_term = signal.signal(signal.SIGTERM, _on_signal)
    previous_int = signal.signal(signal.SIGINT, _on_signal)

    def restore() -> None:
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)

    return restore


# --------------------------------------------------------------------------- #
# SubAgentManager factory
# --------------------------------------------------------------------------- #

def make_subagent_manager(
    *,
    claude_bin: str,
    codex_bin: Optional[str] = None,
    agent_backend: Optional[str] = None,
    model: Optional[str],
    permission_mode: str,
    effort: str,
    extra_add_dirs: List[Path],
    snapshot_file: Path,
    max_concurrent: int = 4,
    budget: Any = None,
) -> SubAgentManager:
    """Build a SubAgentManager with the standard settings shared by every
    orchestrator. Per-orchestrator customization happens via
    ``extra_add_dirs`` (paths agents may Read) and ``max_concurrent``
    (how many sub-agents may run in parallel).

    ``budget`` (optional) wires the per-task :class:`TokenBudget` so
    every agent launch is gated + every result's cost is recorded.
    """
    resolved_agent_backend = (
        agent_backend
        or os.environ.get("METAINFER_AGENT_BACKEND")
        or "claude"
    )
    resolved_codex_bin = (
        codex_bin
        or os.environ.get("METAINFER_CODEX_BIN")
        or "codex"
    )
    return SubAgentManager(
        claude_bin=claude_bin,
        codex_bin=resolved_codex_bin,
        agent_backend=resolved_agent_backend,
        default_model=model,
        permission_mode=permission_mode,
        effort=effort,
        extra_add_dirs=list(extra_add_dirs),
        snapshot_file=snapshot_file,
        max_concurrent=max_concurrent,
        budget=budget,
    )


__all__ = [
    "set_process_name",
    "write_pid_file",
    "clear_pid_file",
    "install_subagent_shutdown_handlers",
    "make_subagent_manager",
]
