"""Background liveness checker for orchestrator processes.

Problem
-------

``reconcile()`` only runs at WebUI startup. If an orchestrator dies
mid-run while the WebUI is up (crash, OOM kill, agent SIGKILLing its
own parent process tree, etc.), nothing notices — ``agents.json`` is
frozen at the last snapshot, ``run.json`` still shows the dead phase,
and the user stares at a "running" pill with ``last_output_age_s``
growing without bound. The next WebUI restart eventually picks it up,
but that can be hours away.

Fix
---

A single asyncio background task scans the registry every
``interval`` seconds. For each task the registry considers "running"
(``pid is not None and finished_at is None``), it calls the existing
``launcher.status()`` validated-liveness check. If that returns
``running=False`` AND the pid file still claims the pid is alive
(i.e. orchestrator exited ungracefully), we invoke the existing
``launcher._reap_dead_pid_file`` — the same cleanup used by the
user-Kill path — to flip the UI to "stopped" and stamp an audit
event.

The check is intentionally cheap (one file read + one ``stat`` per
task) so a 10-second cadence costs single-digit milliseconds even at
hundreds of tasks.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Optional

from . import launcher as _launcher
from . import tasks as _tasks

if TYPE_CHECKING:
    from fastapi import FastAPI

_log = logging.getLogger(__name__)


class LivenessChecker:
    """Periodic background liveness probe.

    Construct once per WebUI process; ``start()`` spawns the loop,
    ``stop()`` cancels it. Idempotent.
    """

    def __init__(
        self,
        interval: float = 10.0,
        launcher: Optional[_launcher.Launcher] = None,
    ) -> None:
        self.interval = interval
        # Use the default launcher if none supplied. The default
        # resolves to LocalLauncher bound to this WebUI session's
        # boot_id; that's what we want for the liveness check.
        self._launcher = launcher or _launcher.get_default_launcher()
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="metainfer-liveness")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def _run(self) -> None:
        """Loop forever, reaping dead orchestrators as we find them.

        Each iteration is bounded and swallows exceptions — a buggy
        task entry or transient filesystem error must not kill the
        watcher (or the WebUI loses liveness detection until restart).
        """
        while True:
            try:
                await asyncio.sleep(self.interval)
                # Run the synchronous scan in a worker so we never
                # block the event loop on /proc reads under heavy load.
                await asyncio.to_thread(self._scan_once)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — watcher must survive
                _log.exception("liveness scan iteration failed; will retry")

    def _scan_once(self) -> None:
        """One pass: probe every task in the registry.

        Visible for tests — call directly to avoid the asyncio sleep loop.

        Process state (pid/started_at/finished_at) is NOT cached in the
        registry — it lives only in each task's orchestrator.pid. So we
        must call status() on every task; we cannot pre-filter. The check
        is intentionally cheap (one file read + one /proc stat per task).
        """
        try:
            entries = _tasks.list_tasks()
        except Exception:  # noqa: BLE001 — registry read must not crash watcher
            _log.exception("liveness: could not read task registry")
            return

        for entry in entries:
            self._check_one(entry.id)

    def _check_one(self, task_id: str) -> None:
        """Check one task. On detected death, invoke the existing reaper
        so the cleanup path is identical to the user-Kill flow."""
        try:
            st = self._launcher.status(task_id)
        except Exception:  # noqa: BLE001 — per-task error must not stop loop
            _log.exception("liveness: status() failed for %s", task_id)
            return

        if st.running:
            return

        # ``exit_hint == "pid-dead"`` means the pid file claimed the
        # process was alive but /proc says otherwise — orchestrator
        # died ungracefully. Other hints (``no-pid-file``,
        # ``pid-file-cleared``) are bookkeeping states we don't touch.
        if st.exit_hint != "pid-dead":
            return

        # Only reap if we had a real pid + started_at to validate
        # against. Without started_at, we cannot tell PID reuse apart
        # from a genuine death — too risky to fire cleanup.
        if st.pid is None or st.started_at is None:
            return

        _log.warning(
            "liveness: orchestrator for %s died (pid=%s started_at=%s) — "
            "reaping stale state",
            task_id, st.pid, st.started_at,
        )
        try:
            # LocalLauncher._reap_dead_pid_file already does the full
            # cleanup: stamps orchestrator.pid::finished_at, clears
            # registry pid, updates run.json::finished/final_status,
            # appends kill_reaped_dead_pid timeline event. Same code
            # path as user-initiated Kill on a dead pid.
            if isinstance(self._launcher, _launcher.LocalLauncher):
                self._launcher._reap_dead_pid_file(
                    task_id, st.pid, st.started_at,
                )
            else:
                _log.warning(
                    "liveness: non-LocalLauncher (%s); cannot reap %s",
                    type(self._launcher).__name__, task_id,
                )
        except Exception:  # noqa: BLE001 — reaper is best-effort
            _log.exception("liveness: reaper failed for %s", task_id)


def attach(app: "FastAPI", interval: float = 10.0) -> LivenessChecker:
    """Wire a :class:`LivenessChecker` into ``app``'s startup/shutdown.

    Idempotent — stores the checker on ``app.state.liveness`` so
    repeated calls return the same instance.
    """
    if getattr(app.state, "liveness", None) is not None:
        return app.state.liveness  # type: ignore[no-any-return]

    checker = LivenessChecker(interval=interval)

    @app.on_event("startup")
    async def _start_liveness() -> None:
        await checker.start()

    @app.on_event("shutdown")
    async def _stop_liveness() -> None:
        await checker.stop()

    app.state.liveness = checker
    return checker
