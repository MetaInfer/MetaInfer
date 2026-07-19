"""SubAgentManager: deterministic lifecycle management for Claude Code subprocesses.

Each "sub-agent" is a `claude -p` invocation run as a child process. The
manager:

* spawns the process with a prompt file piped via stdin
* streams `--output-format stream-json` events to a per-agent log file
* records lifecycle events (start / events / done / killed) into a JSON
  sidecar file so the WebUI can render progress without scraping logs
* detects deadlocks (no new output for ``stuck_timeout_s``)
* kills stuck / timed-out processes
* retries failures up to ``max_retries``
* exposes :meth:`snapshot` for the WebUI to poll

This module is deliberately free of any LLM-driven control flow. The
orchestrator decides *what* to run; the manager owns *how* to keep the
process alive and observable.
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

# Exit codes / signals that indicate infrastructure rather than logic failure.
# 124 = timeout (coreutils convention), 137 = SIGKILL (128+9), 143 = SIGTERM (128+15).
_INFRA_EXIT_CODES = {124, 137, 143}


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass
class AgentSpec:
    """Declarative description of one sub-agent invocation."""
    name: str
    role: str
    prompt_file: Path
    workdir: Path
    log_dir: Path
    timeout_s: int = 1800
    stuck_timeout_s: int = 600
    max_retries: int = 2
    extra_args: List[str] = field(default_factory=list)
    env_overrides: Dict[str, str] = field(default_factory=dict)
    # Session continuation. When ``resume_session_id`` is set the agent is
    # launched with ``ccb --resume <id>`` so it inherits the prior turn's
    # full conversation (loaded files, prior tool results, in-flight
    # diagnoses). ``cache_read_input_tokens`` typically takes ~95% of the
    # context, so subsequent turns are ~10x cheaper than re-seeding from
    # scratch. ``session_id`` (when set on the FIRST turn only) pins the
    # session UUID so the caller can later resume by the same id. If
    # neither is set, ccb mints a fresh session and exposes its id via
    # ``AgentResult.session_id`` so the caller can resume from it.
    session_id: Optional[str] = None
    resume_session_id: Optional[str] = None
    # Per-spec model override. When set, takes precedence over the
    # manager's ``default_model``. Lets cost-conscious roles (e.g. node
    # validators that only emit pass/reject) use a cheaper model while
    # reasoning-heavy roles keep the strong
    # default. None → fall back to manager default.
    model: Optional[str] = None

    def log_file(self, attempt: int) -> Path:
        return self.log_dir / f"{self.name}.attempt{attempt}.log"

    def events_file(self, attempt: int) -> Path:
        return self.log_dir / f"{self.name}.attempt{attempt}.events.jsonl"

    def status_file(self) -> Path:
        return self.log_dir / f"{self.name}.status.json"


@dataclass
class AgentHandle:
    """Runtime handle for a launched (or being-retried) agent."""

    spec: AgentSpec
    attempt: int
    process: Optional[subprocess.Popen] = None
    started_at: float = 0.0
    last_output_at: float = 0.0
    killed: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def elapsed(self) -> float:
        if not self.started_at:
            return 0.0
        return time.time() - self.started_at


@dataclass
class AgentResult:
    name: str
    role: str
    success: bool
    returncode: int
    duration_s: float
    events: List[Dict[str, Any]] = field(default_factory=list)
    final_text: str = ""
    error: Optional[str] = None
    attempts: int = 0
    # ccb session UUID this agent ran under. Set on every successful launch
    # (extracted from the first ``system`` event in the stream). The
    # caller can pass this back via ``AgentSpec.resume_session_id`` to
    # continue the same conversation in a later invocation — re-reading
    # files / re-doing analysis then hits the cache instead of paying
    # full input-token cost again.
    session_id: Optional[str] = None
    # Why the agent failed, if it did:
    #   "infra"    — killed (timeout / stuck / signal) → orchestrator retries
    #                the same phase in place without consuming an iteration
    #   "logic"    — nonzero exit with a "real" error → orchestrator's
    #                transition table decides what to do
    #   "budget"   — refused to launch because the task's token-cost
    #                budget was exhausted. Not retriable; the orchestrator
    #                should treat the whole task as aborted.
    failure_mode: Optional[Literal["infra", "logic", "budget"]] = None
    # Cost / usage pulled from the final ``result`` event of the stream.
    # None when the agent never produced a result event (killed before
    # completion, budget-refused, or stream-json parse failure).
    usage: Optional[Dict[str, Any]] = None


# --------------------------------------------------------------------------- #
# Manager
# --------------------------------------------------------------------------- #


class SubAgentManager:
    """Owns the lifecycle of every Claude Code subprocess in a task.

    The manager is single-threaded *for control* (launch / wait / kill calls
    are serialized by an internal lock) but each agent's stdout is drained on
    its own background thread so a chatty agent never blocks the orchestrator.
    """

    def __init__(
        self,
        claude_bin: str = "ccb",
        default_model: Optional[str] = None,
        max_concurrent: int = 4,
        permission_mode: str = "bypassPermissions",
        extra_add_dirs: Optional[List[Path]] = None,
        effort: str = "max",
        snapshot_file: Optional[Path] = None,
        budget: Any = None,
        budget_source: str = "orchestrator",
    ) -> None:
        self.claude_bin = claude_bin
        self.default_model = default_model
        self.max_concurrent = max_concurrent
        # Per-task token / cost budget (TokenBudget instance or None to
        # disable the circuit breaker). When set, every launch is gated
        # by ``budget.check_launch_allowed`` (refuses new spawns once the
        # soft threshold is crossed) and every successful run records its
        # cost via ``budget.record``. The orchestrator loop separately
        # polls ``budget.snapshot().exhausted`` to abort gracefully.
        self.budget = budget
        # Tag stamped onto every UsageRecord emitted from this manager.
        # Lets the budget bucket distinguish orchestrator-driven agents
        # from web-qa-driven analysts when both share a state_dir.
        self.budget_source = budget_source
        # Claude Code "effort" level — controls how much extended thinking
        # the model is allowed to do per turn. Choices: low / medium / high
        # / max. Iteration logs show reviewers writing only ~550 tokens of
        # text after 13k tokens of thinking, which is fine in principle but
        # was being silently throttled by the default effort setting, so
        # the thinking got cut off mid-analysis. Default "max" lets the
        # model finish its reasoning. Override per-invocation via the
        # METAINFER_EFFORT env var or the CLI --effort flag.
        self.effort = effort
        # Directories every sub-agent is allowed to read from, in addition to
        # the per-invocation workdir. Each task package passes its own
        # knowledge base (e.g. its bundled ``notebooks/`` dir) via this list;
        # without --add-dir the Claude Code sandbox blocks those reads and the
        # agent loops forever against the sandbox. Resolve to absolute, real
        # paths so the flag stays valid even when invoked via a symlink.
        self.extra_add_dirs: List[Path] = [
            Path(p).resolve() for p in (extra_add_dirs or []) if p
        ]
        # Claude Code permission mode for sub-agents. Sub-agents run non-
        # interactively (`-p` with stdin), so `default` mode is unusable:
        # every Edit/Write hangs on a permission prompt nobody can answer.
        # `bypassPermissions` (default) = skip ALL prompts AND the LLM-based
        # Bash safety classifier (which has caused false-denial storms on
        # trusted scripts like perf.sh). `_build_env` sets IS_SANDBOX=1
        # when this mode is active so ccb doesn't refuse under EUID=0.
        # `acceptEdits` / `auto` = weaker alternatives kept for cases where
        # you specifically want the classifier gating sub-agent bash.
        self.permission_mode = permission_mode
        # If set, every agent state change (launch/finish) atomically writes
        # a snapshot of all live + recently-finished agents here. This is
        # the cross-process channel that lets the WebUI (in a different
        # process) render the Live Agents panel without any in-memory
        # coupling to this manager.
        self.snapshot_file: Optional[Path] = (
            Path(snapshot_file) if snapshot_file else None
        )
        self._handles: Dict[str, AgentHandle] = {}
        self._results: Dict[str, AgentResult] = {}
        self._ctrl_lock = threading.Lock()
        self._semaphore = threading.Semaphore(max_concurrent)
        self._stop = threading.Event()
        # Periodic snapshot dumper. dump_snapshot() is otherwise only
        # called on state transitions (launch/finish/kill), which means
        # the WebUI's "Last output" column would go stale for the
        # entire lifetime of a long-running agent — last_output_age_s
        # updates in memory per stdout line but never reaches disk.
        # This thread refreshes the snapshot file every few seconds so
        # the Live Agents panel reflects real-time progress.
        self._snapshot_stop = threading.Event()
        self._snapshot_thread: Optional[threading.Thread] = None
        if self.snapshot_file is not None:
            self._start_snapshot_thread()

    # ------------------------------------------------------------------ #
    # Launch / wait
    # ------------------------------------------------------------------ #

    def launch(self, spec: AgentSpec) -> AgentHandle:
        """Launch ``spec`` (with retries handled internally). Blocking call.

        Returns once the agent has either succeeded or exhausted its retries.
        The returned handle carries the final process; see :meth:`result`.
        """
        # Budget pre-check: refuse to spawn a new subprocess once the
        # task's cost budget is exhausted. Synthetic failure result so
        # the orchestrator's failure-mode handling can route accordingly
        # (failure_mode="budget" → abort the whole task, not retry).
        if self.budget is not None:
            refusal = self.budget.check_launch_allowed(spec.name)
            if refusal:
                synth = AgentResult(
                    name=spec.name, role=spec.role,
                    success=False, returncode=-1, duration_s=0.0,
                    error=refusal, attempts=0,
                    failure_mode="budget",
                )
                with self._ctrl_lock:
                    self._results[spec.name] = synth
                self._write_status(spec, AgentHandle(spec=spec, attempt=0),
                                   synth, 0, phase="budget_refused")
                self.dump_snapshot()
                return AgentHandle(spec=spec, attempt=0)

        spec.log_dir.mkdir(parents=True, exist_ok=True)
        spec.prompt_file.parent.mkdir(parents=True, exist_ok=True)

        attempt = 0
        last_result: Optional[AgentResult] = None
        while attempt <= spec.max_retries:
            attempt += 1
            handle = self._run_once(spec, attempt)
            result = self._materialize_result(handle, spec, attempt)
            last_result = result
            # Record cost into the budget as soon as we have a result,
            # BEFORE deciding to retry. Even failed attempts cost money,
            # and we want the running total to reflect that so the next
            # launch's pre-check can refuse if we just blew past the
            # limit on this attempt.
            self._record_budget(spec, result)
            self._write_status(spec, handle, result, attempt, phase="completed")
            # Surface completion to the WebUI's Live Agents panel.
            self.dump_snapshot()
            if result.success:
                with self._ctrl_lock:
                    self._results[spec.name] = result
                return handle
            # If the budget just got exhausted by this attempt, no
            # point retrying — refuse fast so the orchestrator can
            # surface the abort instead of grinding through max_retries.
            if result.failure_mode == "budget" or (
                self.budget is not None
                and self.budget.snapshot().exhausted
            ):
                with self._ctrl_lock:
                    self._results[spec.name] = result
                return AgentHandle(spec=spec, attempt=attempt - 1)
            # If the failure was caused by a stale session resume (ccb
            # sessions are in-memory, not persisted across processes),
            # drop resume_session_id so the retry starts fresh instead
            # of repeating the same "No conversation found" error.
            if (
                spec.resume_session_id
                and result.error
                and "No conversation found" in result.error
            ):
                spec.resume_session_id = None
            # failed -> retry
            self._write_status(spec, handle, result, attempt, phase="retrying")
            time.sleep(2.0)  # brief backoff

        # exhausted retries
        assert last_result is not None
        with self._ctrl_lock:
            self._results[spec.name] = last_result
        # Return a synthetic handle so caller has something to inspect
        return AgentHandle(spec=spec, attempt=attempt - 1)

    def _record_budget(self, spec: AgentSpec, result: AgentResult) -> None:
        """Fold this attempt's cost into the budget, if any is configured."""
        if self.budget is None or result.usage is None:
            return
        # Local import to avoid a circular dependency at module load time
        # (token_budget.py is in the same package but doesn't depend on
        # this module — still, keeping the import local makes the
        # contract obvious and lets tests stub it more easily).
        from .token_budget import usage_from_result_event
        # phase is not stamped on AgentSpec — the caller knows it. We
        # could add a phase attr to AgentSpec, but the snapshot's
        # per_phase bucket is best-effort; "(unknown)" is fine here and
        # the orchestrator can supply a richer record separately if it
        # cares about per-phase breakdown.
        rec = usage_from_result_event(
            result.usage,
            agent=spec.name,
            source=self.budget_source,
            phase=None,
        )
        self.budget.record(rec)

    def launch_async(self, spec: AgentSpec, on_done: Optional[callable] = None) -> threading.Thread:
        """Run :meth:`launch` on a background thread."""
        t = threading.Thread(
            target=self._async_wrapper, args=(spec, on_done), name=f"agent-{spec.name}", daemon=True
        )
        t.start()
        return t

    def _async_wrapper(self, spec: AgentSpec, on_done: Optional[callable]) -> None:
        try:
            self.launch(spec)
        except Exception as exc:  # noqa: BLE001
            self._results[spec.name] = AgentResult(
                name=spec.name, role=spec.role, success=False, returncode=-1,
                duration_s=0.0, error=f"manager exception: {exc!r}",
            )
        finally:
            self._semaphore.release()
            if on_done:
                try:
                    on_done(spec.name)
                except Exception:  # noqa: BLE001
                    pass

    def _run_once(self, spec: AgentSpec, attempt: int) -> AgentHandle:
        with self._semaphore:
            handle = AgentHandle(spec=spec, attempt=attempt)
            with self._ctrl_lock:
                self._handles[spec.name] = handle
            self._write_status(spec, handle, None, attempt, phase="starting")
            # Surface the new agent to the WebUI immediately so the
            # Live Agents panel reflects the start without waiting for
            # the next periodic dump.
            self.dump_snapshot()

            cmd = self._build_command(spec)
            env = self._build_env(spec)
            log_fp = open(spec.log_file(attempt), "wb")
            events_fp = open(spec.events_file(attempt), "w", encoding="utf-8")

            try:
                handle.started_at = time.time()
                handle.last_output_at = handle.started_at
                with open(spec.prompt_file, "rb") as prompt_fp:
                    proc = subprocess.Popen(
                        cmd,
                        stdin=prompt_fp,
                        stdout=subprocess.PIPE,
                        stderr=log_fp,
                        cwd=str(spec.workdir),
                        env=env,
                        text=False,
                        start_new_session=True,
                    )
                handle.process = proc
                self._write_status(spec, handle, None, attempt, phase="running")

                # drain stdout on a thread (stream-json is line-delimited JSON)
                stop_evt = threading.Event()

                def drain() -> None:
                    assert proc.stdout is not None
                    for raw in proc.stdout:
                        line = raw.decode("utf-8", errors="replace")
                        events_fp.write(line)
                        events_fp.flush()
                        handle.last_output_at = time.time()
                        log_fp.write(raw)
                        log_fp.flush()
                    stop_evt.set()

                reader = threading.Thread(target=drain, name=f"drain-{spec.name}", daemon=True)
                reader.start()

                # watchdog: timeout / stuck detection
                while proc.poll() is None:
                    if self._stop.is_set():
                        self._terminate(handle)
                        break
                    now = time.time()
                    if now - handle.started_at > spec.timeout_s:
                        self._terminate(handle, reason="timeout")
                        break
                    if now - handle.last_output_at > spec.stuck_timeout_s:
                        self._terminate(handle, reason="stuck")
                        break
                    time.sleep(2.0)

                proc.wait(timeout=30)
                reader.join(timeout=10)
            finally:
                log_fp.close()
                events_fp.close()
            return handle

    # ------------------------------------------------------------------ #
    # Process control
    # ------------------------------------------------------------------ #

    def _terminate(self, handle: AgentHandle, reason: str = "manual") -> None:
        with handle.lock:
            if handle.killed or handle.process is None:
                return
            handle.killed = True
            proc = handle.process
        try:
            # try graceful on the process group first
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                return
            except Exception:  # noqa: BLE001
                proc.terminate()
            # hard kill after grace period
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:  # noqa: BLE001
                    proc.kill()
        finally:
            # write a marker into the events file for forensics
            try:
                ef = handle.spec.events_file(handle.attempt)
                with open(ef, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"type": "__meta__", "killed": reason}) + "\n")
            except Exception:  # noqa: BLE001
                pass
            # Reflect the kill in the cross-process snapshot.
            self.dump_snapshot()

    def kill(self, name: str) -> bool:
        with self._ctrl_lock:
            handle = self._handles.get(name)
        if handle is None:
            return False
        self._terminate(handle, reason="kill-requested")
        return True

    def kill_all(self) -> None:
        with self._ctrl_lock:
            handles = list(self._handles.values())
        for h in handles:
            self._terminate(h, reason="shutdown")

    def shutdown(self) -> None:
        self._stop.set()
        self.kill_all()
        # Stop the periodic snapshot dumper last so the final kill_all
        # state still gets one fresh snapshot out the door.
        self._stop_snapshot_thread()

    # ------------------------------------------------------------------ #
    # Periodic snapshot dumper
    # ------------------------------------------------------------------ #

    # Refresh interval for the cross-process snapshot file. 3s trades a
    # little I/O for keeping the WebUI's Live Agents / Last output column
    # honest. Without this, the snapshot is only written on agent
    # state transitions and goes stale for the entire duration of a
    # long-running agent.
    SNAPSHOT_REFRESH_S = 3.0

    def _start_snapshot_thread(self) -> None:
        if self._snapshot_thread is not None and self._snapshot_thread.is_alive():
            return
        self._snapshot_stop.clear()
        t = threading.Thread(
            target=self._snapshot_loop,
            name="subagent-snapshot",
            daemon=True,
        )
        t.start()
        self._snapshot_thread = t

    def _stop_snapshot_thread(self) -> None:
        self._snapshot_stop.set()
        t = self._snapshot_thread
        if t is not None and t.is_alive():
            t.join(timeout=self.SNAPSHOT_REFRESH_S * 2 + 1.0)
        self._snapshot_thread = None
        # Final flush so the file reflects whatever state we ended in.
        self.dump_snapshot()

    def _snapshot_loop(self) -> None:
        while not self._snapshot_stop.is_set():
            try:
                # Only pay the serialization cost when something is
                # actually running. Idle managers skip the write entirely.
                with self._ctrl_lock:
                    active = any(h.process is not None and h.process.poll() is None
                                 for h in self._handles.values())
                if active:
                    self.dump_snapshot()
            except Exception:  # noqa: BLE001 — observability must never break the loop
                pass
            self._snapshot_stop.wait(self.SNAPSHOT_REFRESH_S)

    # ------------------------------------------------------------------ #
    # Health / snapshot
    # ------------------------------------------------------------------ #

    def health(self, name: str) -> Literal["running", "stuck", "done", "failed", "unknown"]:
        with self._ctrl_lock:
            handle = self._handles.get(name)
            result = self._results.get(name)
        if result is not None:
            return "done" if result.success else "failed"
        if handle is None or handle.process is None:
            return "unknown"
        if handle.process.poll() is not None:
            return "done" if handle.process.returncode == 0 and not handle.killed else "failed"
        if time.time() - handle.last_output_at > handle.spec.stuck_timeout_s:
            return "stuck"
        return "running"

    def snapshot(self) -> List[Dict[str, Any]]:
        """Return a list of agent status dicts for the WebUI."""
        out: List[Dict[str, Any]] = []
        with self._ctrl_lock:
            names = list(self._handles.keys())
        for name in names:
            with self._ctrl_lock:
                handle = self._handles.get(name)
                result = self._results.get(name)
            if handle is None:
                continue
            out.append(
                {
                    "name": name,
                    "role": handle.spec.role,
                    "attempt": handle.attempt,
                    "phase": self.health(name),
                    "elapsed_s": round(handle.elapsed, 1),
                    "last_output_age_s": round(time.time() - handle.last_output_at, 1)
                    if handle.last_output_at
                    else None,
                    "killed": handle.killed,
                    "success": result.success if result else None,
                    "error": result.error if result else None,
                    "log_file": str(handle.spec.log_file(handle.attempt)),
                }
            )
        return out

    def result(self, name: str) -> Optional[AgentResult]:
        with self._ctrl_lock:
            return self._results.get(name)

    def results(self) -> Dict[str, AgentResult]:
        with self._ctrl_lock:
            return dict(self._results)

    # ------------------------------------------------------------------ #
    # Cross-process snapshot
    # ------------------------------------------------------------------ #

    def dump_snapshot(self) -> None:
        """Atomically write the current snapshot to ``self.snapshot_file``.

        The WebUI lives in a separate process and reads this file to
        render the Live Agents panel. Called by the orchestrator at
        every meaningful state change (agent start / finish / kill).

        Swallows all errors — snapshotting is observability-only and
        must never break the orchestrator loop. If ``snapshot_file`` is
        None, this is a no-op (legacy callers that don't need it).
        """
        if self.snapshot_file is None:
            return
        try:
            snap = self.snapshot()
            payload = {"ts": time.time(), "agents": snap}
            tmp = self.snapshot_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(self.snapshot_file)
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _build_command(self, spec: AgentSpec) -> List[str]:
        cmd = [
            self.claude_bin,
            "-p",  # print (non-interactive) mode; prompt read from stdin
            "--output-format", "stream-json",
            "--input-format", "text",
            "--verbose",
            # Sub-agents can't answer permission prompts (non-interactive).
            # Default to a mode that lets them write code without hanging.
            # The iteration dir is the agent's CWD, so file ops stay scoped
            # to that folder.
            "--permission-mode", self.permission_mode,
            # Explicitly add the iteration dir as an allowed working dir.
            # Belt-and-suspenders: cwd is already set to spec.workdir below,
            # but --add-dir guarantees the agent treats it as writable.
            "--add-dir", str(spec.workdir),
        ]
        # Read-only knowledge sources (per-task notebooks/, repo root, etc.).
        # The manager-level list applies to every sub-agent so individual
        # phase code doesn't have to remember to opt in.
        for d in self.extra_add_dirs:
            cmd += ["--add-dir", str(d)]
        if spec.model:
            cmd += ["--model", spec.model]
        elif self.default_model:
            cmd += ["--model", self.default_model]
        # Effort level controls extended-thinking budget. "max" lets the
        # model finish long reasoning chains instead of getting cut off
        # mid-analysis (which iteration logs showed happening to reviewers).
        if self.effort:
            cmd += ["--effort", self.effort]
        # Session continuation. ``--resume`` takes precedence (explicitly
        # continuing an existing conversation); otherwise ``--session-id``
        # pins the UUID for the first turn so the caller knows what to
        # resume later. If neither is set, ccb mints a fresh session and
        # the manager captures its id from the stream.
        if spec.resume_session_id:
            cmd += ["--resume", spec.resume_session_id]
        elif spec.session_id:
            cmd += ["--session-id", spec.session_id]
        cmd += list(spec.extra_args)
        return cmd

    def _build_env(self, spec: AgentSpec) -> Dict[str, str]:
        env = dict(os.environ)
        env.update(spec.env_overrides)
        # Keep the agent from going interactive
        env.setdefault("DISABLE_INTERACTIVITY", "1")
        # bypassPermissions under EUID=0 normally trips a hard exit
        # ("--dangerously-skip-permissions cannot be used with root/sudo
        # privileges"). ccb skips that check when IS_SANDBOX=1, which is
        # the official escape hatch for trusted container/CI contexts.
        # The orchestrator's sub-agents are non-interactive (-p + piped
        # stdin) and run trusted code, so this is the correct signal —
        # it does NOT change the actual sandboxing of the process.
        # See setup-C2V4elOv.js in claude-code: the root check fires
        # only when IS_SANDBOX !== "1" && !CLAUDE_CODE_BUBBLEWRAP.
        if self.permission_mode == "bypassPermissions":
            env["IS_SANDBOX"] = "1"
        return env

    def _materialize_result(
        self, handle: AgentHandle, spec: AgentSpec, attempt: int
    ) -> AgentResult:
        proc = handle.process
        rc = proc.returncode if proc is not None else -1
        events: List[Dict[str, Any]] = []
        ef = spec.events_file(attempt)
        if ef.exists():
            for ln in ef.read_text(encoding="utf-8", errors="replace").splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    events.append(json.loads(ln))
                except json.JSONDecodeError:
                    continue
        final_text = ""
        for ev in reversed(events):
            if ev.get("type") == "assistant" and isinstance(ev.get("message"), dict):
                content = ev["message"].get("content")
                if isinstance(content, list):
                    for blk in content:
                        if isinstance(blk, dict) and blk.get("type") == "text":
                            final_text = blk.get("text", "")
                            break
                if final_text:
                    break
            if ev.get("type") == "result":
                final_text = ev.get("result", "") or final_text
                break
        # Session id: emitted on the very first ``system`` event of the
        # stream and again on every ``result`` event. Prefer the result's
        # value (it's the final, post-turn session id; for ``--resume``
        # invocations this matches the resumed-from id and confirms the
        # continuation actually happened).
        session_id = None
        for ev in events:
            sid = ev.get("session_id")
            if sid:
                session_id = sid
                if ev.get("type") == "result":
                    break
        # Pull the cost / usage block from the final ``result`` event.
        # This is what the token budget circuit breaker keys off. None
        # when the agent was killed / crashed before emitting result.
        usage: Optional[Dict[str, Any]] = None
        for ev in reversed(events):
            if ev.get("type") == "result" and isinstance(ev, dict):
                if "usage" in ev or "total_cost_usd" in ev:
                    usage = ev
                    break
        error = None
        failure_mode: Optional[Literal["infra", "logic", "budget"]] = None
        success = (rc == 0) and not handle.killed
        if handle.killed:
            error = "killed (see events log tail)"
            failure_mode = "infra"
        elif rc != 0:
            error = f"nonzero exit {rc}"
            # negative rc = killed by signal; specific positive codes = timeout/etc.
            if rc < 0 or rc in _INFRA_EXIT_CODES:
                failure_mode = "infra"
            else:
                failure_mode = "logic"
        return AgentResult(
            name=spec.name,
            role=spec.role,
            success=success,
            returncode=rc,
            duration_s=handle.elapsed,
            events=events,
            final_text=final_text,
            error=error,
            attempts=attempt,
            session_id=session_id,
            failure_mode=failure_mode,
            usage=usage,
        )

    def _write_status(
        self,
        spec: AgentSpec,
        handle: AgentHandle,
        result: Optional[AgentResult],
        attempt: int,
        phase: str,
    ) -> None:
        status = {
            "name": spec.name,
            "role": spec.role,
            "attempt": attempt,
            "phase": phase,
            "elapsed_s": round(handle.elapsed, 1),
            "success": result.success if result else None,
            "error": result.error if result else None,
            "updated_at": time.time(),
            "log_file": str(spec.log_file(attempt)),
            "events_file": str(spec.events_file(attempt)),
        }
        tmp = spec.status_file().with_suffix(".tmp")
        tmp.write_text(json.dumps(status, indent=2), encoding="utf-8")
        tmp.replace(spec.status_file())
