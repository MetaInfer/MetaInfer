"""Deterministic state-machine-driven orchestrator for MetaInfer.

The control flow is fully owned by this module and is driven by the
**transition table** in :mod:`metainfer.phases`. Adding a new phase or
changing when it fires is a one-line edit there — no changes here.

## Loop shape

The orchestrator runs a phase-at-a-time loop::

    phase = "A_plan"
    while not terminal(phase):
        if no open iteration folder: open one
        outcome = run_phase(phase, ...)
        t = TRANSITIONS[(phase, outcome)]   # or abort if missing
        update ctx (carry_failure / carry_perf)
        record into iteration record + timeline
        if t.consume_iteration: close the folder (next phase gets a fresh one)
        phase = t.to_phase

A/D/E/F infra failures retry in place (same folder,
``consume_iteration=False``) up to :data:`MAX_PHASE_ATTEMPTS`, after which
they close the current iteration as failed and start a fresh one back at
``A_plan`` (carrying the failure reason forward so the new planner can
re-plan around it). The run stops at the iteration cap, task wall-clock cap,
the correctness-only target, or a measured throughput target. B_implement
failures are special: they consume
the iteration immediately and route to A_plan for a fresh plan
(SubAgentManager already did internal retries; piling more in-place
retries on top traps the agent in a --resume spiral on the same failing
approach). C always continues to D in the same iteration so review can use
the actual oracle result. D then routes a C-pass to E, or closes a C-failed
iteration and starts a fresh A plan with the failure and review carried.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import phases as P
from .plugin import PLUGIN
from .task_types import is_http_framework_task
from metainfer.orchestrator.iteration import IterationWorkspace
from .prompts import (
    c_repair_followup_prompt,
    c_repair_prompt,
    failure_retrospective_prompt,
    implement_prompt,
    implement_redo_prompt,
    perf_plan_prompt,
    perf_review_prompt,
    perf_test_prompt,
    plan_prompt,
    retrospective_prompt,
    review_prompt,
    write_test_harness_prompt,
)
from metainfer.orchestrator.state import StateStore
from metainfer.orchestrator.subagent_manager import AgentSpec, SubAgentManager
from metainfer.orchestrator.token_budget import TokenBudget
from .iteration_record import IterationRecord
from .implementation_gate import (
    execute_native_build_gate,
    source_fingerprint,
    validate_implementation,
)
from .plan_gate import resolve_plan_gate_context, validate_plan


# StateStore treats iteration records as opaque dicts; gf wraps each
# call to convert between dict and its own IterationRecord dataclass.
def _load_iter(store: StateStore, n: int) -> Optional[IterationRecord]:
    d = store.load_iteration(n)
    return IterationRecord.from_dict(d) if d else None


def _write_iter(store: StateStore, rec: IterationRecord) -> None:
    store.write_iteration(rec.iteration, rec.to_dict())


JSON_LINE_RE = re.compile(r"\{.*\}")

# How many times a single phase may be retried in place (within one
# iteration folder) before the orchestrator closes the iteration as
# failed and routes back to A_plan to start a fresh iteration.
MAX_PHASE_ATTEMPTS = 3

# Relative drop in the primary perf metric that counts as a regression.
PERF_REGRESSION_THRESHOLD = 0.20


# --------------------------------------------------------------------------- #
# IterationContext — cross-phase / cross-iteration mutable state
# --------------------------------------------------------------------------- #


@dataclass
class IterationContext:
    """Mutable state threaded through the orchestrator loop.

    Replaces the previous bundle of ad-hoc locals (``prev_failed``,
    ``last_perf``). New cross-iteration state goes here, not as a new
    ``_run_iteration`` parameter.
    """

    failure: Optional[str] = None
    last_perf: Optional[Dict[str, float]] = None
    best_perf: Optional[Dict[str, float]] = None
    # Perf produced by THIS iteration's own phases only. Reset on every
    # iteration-folder open. Used at close time to stamp the iteration
    # record's `perf` field — WITHOUT it, _close_iteration would stamp
    # `ctx.last_perf`, which carries across iterations and makes a
    # perf-less iteration (e.g. one where E failed) inherit the most
    # recent successful iteration's numbers. That fooled both the perf
    # chart and the retro endpoint's "this vs previous" comparison into
    # showing identical values for iterations that never measured
    # anything. Keeping this scoped per-iteration means an iteration
    # only has perf data if IT actually produced some.
    this_iter_perf: Optional[Dict[str, float]] = None
    last_outcome: Optional[P.Outcome] = None
    # Most recent post-test reviewer feedback (code + test-result review).
    # Set by D_review (and historically after C). Fed into the next
    # iteration's implementer / planner / optimizer prompt so review
    # suggestions compound across iterations instead of being discarded.
    review_feedback: Optional[str] = None
    # Most recent perf plan written by F_perf_plan. Fed into the NEXT
    # iteration's A_plan / B_implement so the next cycle executes the plan.
    # Carries the optimization targets that drive the next iteration's work.
    perf_plan: Optional[str] = None
    # Machine-readable diagnostics from the most recent rejected A output.
    # Kept only within the current iteration so an in-place A retry can fix
    # the artifacts without polluting later successful planning cycles.
    plan_gate_feedback: Optional[str] = None
    # phase-id -> attempts in the current iteration folder; reset on folder open
    phase_attempts: Dict[str, int] = field(default_factory=dict)
    # ccb session UUID of the most recent B_implement turn in the current
    # iteration. When B LOGIC_FAILs and the transition table routes back to
    # B (in-place redo, same iter_dir), _do_implement passes this back via
    # --resume so the implementer keeps the prior turn's full context
    # (loaded plan, loaded source files, the diagnosis it was about to
    # act on) instead of starting cold. Reset to None on each new
    # iteration folder — a fresh iter gets a fresh session.
    b_session_id: Optional[str] = None


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass
class OrchestratorConfig:
    workdir: Path
    repo_root: Path
    notebooks_dir: Path
    iterations_root: Path
    state_dir: Path
    # Where per-iteration agent/oracle logs land. New layout (since the
    # refactor) puts logs OUTSIDE the iteration code dir, under
    # ``<cwd>/.metainfer/logs/<task_id>/<NNN>/``. The orchestrator writes
    # prompts, agent stdout/stderr, oracle reports, server logs, and the
    # prev-iter diagnostic snapshot here. Iteration CODE stays clean under
    # ``<cwd>/<task_id>/<NNN>/``.
    logs_root: Optional[Path] = None
    max_iterations: int = 20
    max_wall_time_s: Optional[int] = None
    execution_mode: str = "optimize"
    # agent tuning
    plan_timeout_s: int = 1800
    impl_timeout_s: int = 3600
    # C repair is deliberately shorter than a full implementation turn.
    # The debugger edits one diagnosed root cause; architectural work must
    # return to A rather than occupy three one-hour internal retries.
    c_repair_timeout_s: int = 1200
    review_timeout_s: int = 1800
    optimize_timeout_s: int = 3600
    stuck_timeout_s: int = 600
    # Post-E retrospective writer budget. The retro is a single short
    # Markdown summary; 10 minutes is generous but bounded. Failure to
    # produce it is non-fatal (E's outcome is unchanged).
    retro_timeout_s: int = 600
    # C-step in-place repair budget. When the correctness test (oracle or
    # agent-written test.sh) fails with a LOGIC_FAIL, the orchestrator lets
    # a debugger sub-agent try to fix the code in the SAME iteration folder
    # and re-runs the test, up to this many times. Only after the budget is
    # exhausted does C route forward to D_review with LOGIC_FAIL (which then
    # routes to a fresh A plan). Infra failures bypass this loop and
    # surface immediately. 3 = "try twice as hard before giving up an
    # iteration" — empirically most fixable bugs are caught in 1-2 retries;
    # deeper architectural issues need a fresh plan with reviewer context.
    max_c_retries: int = 3
    claude_bin: str = "ccb"
    model: Optional[str] = None
    # Claude Code permission mode for sub-agents. Default
    # "bypassPermissions" skips ALL permission checks AND the LLM-based
    # Bash safety classifier (the one that denied `bash perf.sh` 60+
    # times in iter3 of the qwen3-8b task). Sub-agents are non-interactive
    # (-p + piped stdin) running trusted code in a controlled working
    # dir, so the classifier adds false positives without real safety
    # value. SubAgentManager sets IS_SANDBOX=1 to bypass ccb's
    # root/bypassPermissions hard-exit.
    permission_mode: str = "bypassPermissions"
    extra_claude_args: List[str] = field(default_factory=list)
    # If set, perf regression is detected against this metric key (e.g.
    # "tokens_per_sec"). If None, regression detection is disabled.
    primary_perf_metric: Optional[str] = "tokens_per_sec"


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #


class Orchestrator:
    def __init__(
        self,
        req: Dict[str, Any],
        store: StateStore,
        cfg: OrchestratorConfig,
        manager: Optional[SubAgentManager] = None,
        budget: Optional[TokenBudget] = None,
    ) -> None:
        self.req = req
        self.store = store
        self.cfg = cfg
        self.budget = budget
        self.manager = manager or SubAgentManager(
            claude_bin=cfg.claude_bin,
            default_model=cfg.model,
            permission_mode=cfg.permission_mode,
            # Fall-through path: when no manager is injected, still grant
            # access to the knowledge base and the logs root (reviewer writes
            # review.md there, prev-iter snapshot lives there) so prompts
            # that point at those paths don't loop against the sandbox.
            extra_add_dirs=[
                cfg.notebooks_dir,
                *([cfg.logs_root] if cfg.logs_root else []),
            ],
        )
        self.workspace = IterationWorkspace(
            cfg.iterations_root, logs_root=cfg.logs_root,
            diagnostic_globs=PLUGIN.diagnostic_globs,
        )
        self._stop = False
        self._wall_deadline_epoch: Optional[float] = None
        # Set when run() no-ops because the task was already finished.
        # Caller (orchestrator.py) reads this to decide whether keepalive
        # makes sense — keeping the WebUI up for a run that did nothing
        # just traps the user in a frozen-looking process.
        self.nooped = False

    def _logs_dir_for(self, n: int) -> Path:
        """Per-iteration logs dir (prompts, agent stdout/stderr, oracle
        reports, server logs, prev-iter snapshot).

        Delegates to :class:`IterationWorkspace` so the path stays in sync
        with where the prev-iter copy-forward logic also writes.
        """
        return self.workspace.logs_dir_for(n)

    # ------------------------------------------------------------------ #
    # Public entry
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        task_id = self.req.get("task_id", "task")
        run_status, is_resume = self.store.init_or_resume(
            task_id=task_id, task_type=self.req.get("task_type", "unknown"),
        )

        if self.cfg.max_wall_time_s is not None:
            # A live resume keeps the original task deadline. An explicit
            # restart of a finished task starts a new bounded run window.
            base = (
                time.time()
                if is_resume and run_status.finished
                else float(run_status.created_at or time.time())
            )
            self._wall_deadline_epoch = base + self.cfg.max_wall_time_s

        resume_from: Optional[Dict[str, Any]] = None
        if is_resume:
            rs = self.store.load_run()
            if rs.finished:
                # The run already terminated (success / stopped / aborted).
                # The orchestrator only gets re-invoked on a finished task
                # when the user clicked Restart in the WebUI. Treat that
                # as "clear the terminal flag and keep exploring" — pick
                # up where the previous run left off and start a fresh
                # iteration. The historical iterations / timeline are
                # preserved; we only reset the run-level terminal stamp.
                self.store.append_timeline(
                    "orchestrator_restart",
                    {"prior_final_status": rs.final_status,
                     "prior_phase": rs.current_phase},
                )
                print(f"[metainfer] task {task_id!r} was finished "
                      f"(status={rs.final_status}); restarting on user request")
                self.store.update_run(
                    finished=False,
                    final_status=None,
                    current_phase="idle",
                    last_outcome=None,
                    last_transition_label=None,
                )
            self.store.append_timeline("orchestrator_resume", {"task_id": task_id})
            print(f"[metainfer] resuming task {task_id!r} from existing state")
            resume_from = self._prepare_resume()
        else:
            self.store.append_timeline("orchestrator_start", {"task_id": task_id})

        try:
            self._loop(resume_from=resume_from)
        except KeyboardInterrupt:
            self.store.append_timeline("orchestrator_abort", {"reason": "keyboard-interrupt"})
            # External interrupt — never mark the run as "failed". The
            # system was exploring and was externally stopped; that is
            # distinct from giving up. final_status="aborted" records
            # "user/dev interrupted this", and current_phase lands on the
            # generic terminal "finished" so the WebUI doesn't show a
            # Fail badge.
            self.store.update_run(finished=True, final_status="aborted",
                                  current_phase="finished")
        finally:
            self.manager.shutdown()
            self.store.append_timeline("orchestrator_end", {"task_id": task_id})

    def _prepare_resume(self) -> Dict[str, Any]:
        """Inspect existing iteration state and figure out where to restart.

        Returns a dict with ``iter_num``, ``start_phase``, and any context
        we can recover from the prior iteration (carried failure, last
        outcome). The folder for an incomplete top iteration is deleted,
        along with its state-store record (after we read its start_phase).
        """
        # 1. If the top iteration folder lacks the completion sentinel,
        #    discard it and archive its record as failed/interrupted —
        #    it was abandoned mid-flight. The archived record stays
        #    visible in the WebUI history (status=failed, with an
        #    "interrupted" marker) so the user can see what happened.
        discarded = self.workspace.discard_latest_incomplete()
        if discarded is not None:
            old_rec = _load_iter(self.store, discarded)
            # Capture the phase the iteration was in when the orchestrator
            # died, if we can tell from the partial phases dict.
            interrupted_in = None
            if old_rec is not None and old_rec.phases:
                interrupted_in = max(
                    old_rec.phases.keys(),
                    key=lambda k: old_rec.phases[k].get("started_at", 0),
                )
            reason = (
                f"user interrupted (orchestrator process exited mid-"
                f"{interrupted_in or 'flight'})"
                if interrupted_in
                else "user interrupted (orchestrator process exited unexpectedly)"
            )
            self.store.archive_interrupted_iteration(discarded, reason=reason)
            self.store.append_timeline(
                "iteration_interrupted",
                {"iteration": discarded, "reason": reason,
                 "restart_from": (old_rec.start_phase if old_rec else "A_plan")},
            )
            print(f"[metainfer] iteration {discarded:03d} marked interrupted "
                  f"({reason}); will restart it from "
                  f"{old_rec.start_phase if old_rec else 'A_plan'}")
            start_phase = (old_rec.start_phase if old_rec else "A_plan") or "A_plan"
            iter_num = discarded
            carried_failure = None
            last_outcome: Optional[P.Outcome] = None
            # Look at the iteration just before the discarded one to recover
            # context (failure reason, outcome) so the retry has the same
            # starting point as the original attempt.
            prev_rec = _load_iter(self.store, iter_num - 1) if iter_num > 1 else None
            if prev_rec is not None:
                last_outcome = prev_rec.outcome
                if prev_rec.outcome != P.OK:
                    carried_failure = prev_rec.failure_reason
        else:
            # All existing iterations complete → start a fresh next one
            # whose phase is determined by the last iteration's outcome.
            last_complete = self.workspace.latest_complete_number()
            prev_rec = _load_iter(self.store, last_complete) if last_complete else None
            iter_num = last_complete + 1
            start_phase = self._phase_after(prev_rec)
            carried_failure = (
                prev_rec.failure_reason
                if prev_rec is not None and prev_rec.outcome != P.OK
                else None
            )
            last_outcome = prev_rec.outcome if prev_rec is not None else None

        return {
            "iter_num": iter_num,
            "start_phase": start_phase,
            "carried_failure": carried_failure,
            "last_outcome": last_outcome,
        }

    @staticmethod
    def _phase_after(rec: Optional[IterationRecord]) -> P.Phase:
        """Given the last completed iteration's record, decide which phase
        the *next* iteration should start at.

        In the 6-phase flow:
        - C passed → D ran → E ran → F ran → next iter starts at A (the
          cycle always re-plans from scratch using F's perf_plan.md).
        - C failed → D ran → next iter starts at A so the planner can
          redesign around the concrete failure and D's review.

        The iteration's recorded ``outcome`` is C's outcome (the iteration
        terminates after F on the happy path or after D on the fail path).
        """
        if rec is None:
            return "A_plan"
        # Both successful optimization cycles and failed correctness cycles
        # start from a fresh plan. Failure details and review.md are carried
        # into A so architectural mistakes do not get trapped in patch-only
        # B turns.
        return "A_plan"

    def stop(self) -> None:
        self._stop = True

    def _remaining_wall_time_s(self) -> Optional[int]:
        deadline = self._wall_deadline_epoch
        if deadline is None:
            return None
        return max(0, int(deadline - time.time()))

    def _bounded_timeout(self, requested_s: int) -> int:
        remaining = self._remaining_wall_time_s()
        if remaining is None:
            return max(1, int(requested_s))
        return max(1, min(int(requested_s), remaining))

    def _wall_time_exhausted(self) -> bool:
        remaining = self._remaining_wall_time_s()
        return remaining is not None and remaining <= 0

    def _wall_time_reason(self) -> str:
        return (
            "task wall-clock budget exhausted: "
            f"limit={self.cfg.max_wall_time_s}s"
        )

    def _throughput_target_met(
        self, perf: Optional[Dict[str, float]],
    ) -> Optional[Dict[str, float]]:
        raw = self.req.get("target_tokens_per_sec")
        if raw in (None, "") or not perf:
            return None
        try:
            target = float(raw)
            measured = float(perf.get("tokens_per_sec", 0.0))
        except (TypeError, ValueError):
            return None
        if target <= 0 or measured < target:
            return None
        return {"target_tokens_per_sec": target, "tokens_per_sec": measured}

    # ------------------------------------------------------------------ #
    # Main loop — phase at a time, transition-table driven
    # ------------------------------------------------------------------ #

    def _loop(self, resume_from: Optional[Dict[str, Any]] = None) -> None:
        max_iters = self._resolve_max_iterations()
        ctx = IterationContext()

        if resume_from is not None:
            phase: P.Phase = resume_from["start_phase"]
            # iter_num is set to one less than the target so the first
            # pass through the loop's "open a fresh folder" branch
            # increments to the right number.
            iter_num = resume_from["iter_num"] - 1
            ctx.failure = resume_from.get("carried_failure")
            ctx.last_outcome = resume_from.get("last_outcome")
        else:
            phase = "A_plan"
            iter_num = 0

        iter_dir: Optional[Path] = None
        iter_rec: Optional[IterationRecord] = None
        final_status: Optional[str] = None

        while not self._stop and not P.is_terminal(phase):
            if self._wall_time_exhausted():
                reason = self._wall_time_reason()
                if iter_rec is not None:
                    self._close_iteration(
                        iter_rec, status="failed", failure=reason,
                        perf=ctx.this_iter_perf, outcome=P.ABORTED,
                        write_retrospective=False,
                    )
                self.store.append_timeline(
                    "wall_time_exhausted",
                    {"iteration": iter_num, "phase": phase, "reason": reason},
                )
                final_status = "stopped"
                phase = "finished"
                break
            # ---- open a fresh iteration folder if needed ------------------ #
            if iter_dir is None:
                if iter_num >= max_iters:
                    # Iteration budget exhausted. The system explored until
                    # the configured cap. Record as "stopped" if
                    # the last attempt didn't happen to land on OK, but
                    # never as "failed" — there is no Fail state.
                    final_status = "success" if ctx.last_outcome == P.OK else "stopped"
                    phase = "finished"
                    break
                iter_num += 1
                iter_dir = self.workspace.open_iteration(iter_num)
                iter_rec = IterationRecord(
                    iteration=iter_num,
                    started_at=time.time(),
                    start_phase=phase,
                )
                _write_iter(self.store, iter_rec)
                # Reset per-iteration session state — the new iteration's
                # code tree is a fresh starting point, so resuming from
                # the prior iter's B/C sessions would only confuse the
                # agent with stale file contents.
                ctx.b_session_id = None
                ctx.plan_gate_feedback = None
                # Reset the per-iteration perf accumulator so we don't
                # inherit the prior iteration's perf when this iteration
                # never produces any of its own (see this_iter_perf docs).
                ctx.this_iter_perf = None
                self.store.update_run(
                    current_iteration=iter_num,
                    current_phase=phase,
                    last_outcome=ctx.last_outcome,
                )
                self.store.append_timeline(
                    "iteration_start",
                    {"iteration": iter_num, "start_phase": phase,
                     "carried_failure": ctx.failure is not None},
                )
                ctx.phase_attempts.clear()

            # ---- run the phase ------------------------------------------- #
            assert iter_rec is not None
            self._set_phase(iter_num, iter_dir, phase)
            outcome, perf, failure = self._run_phase(phase, iter_num, iter_dir, iter_rec, ctx)

            ctx.last_outcome = outcome
            ctx.phase_attempts[phase] = ctx.phase_attempts.get(phase, 0) + 1

            # record phase outcome into the iteration record
            phase_rec = iter_rec.phases.setdefault(phase, {})
            phase_rec["outcome"] = outcome
            phase_rec["attempts"] = ctx.phase_attempts[phase]
            phase_rec["ended_at"] = time.time()
            if failure:
                phase_rec["failure"] = failure
            if perf:
                phase_rec["perf"] = perf

            if self._wall_time_exhausted():
                reason = self._wall_time_reason()
                self._close_iteration(
                    iter_rec, status="failed", failure=reason,
                    perf=ctx.this_iter_perf, outcome=P.ABORTED,
                    write_retrospective=False,
                )
                self.store.append_timeline(
                    "wall_time_exhausted",
                    {"iteration": iter_num, "phase": phase, "reason": reason},
                )
                final_status = "stopped"
                phase = "finished"
                break

            # ---- token-budget circuit breaker ------------------------------ #
            # After every phase, check whether the task's cost budget has
            # been exhausted. If so: close the current iteration (with
            # status=aborted so the WebUI distinguishes "we ran out of
            # money" from "the agent gave up"), write a timeline event,
            # set final_status, and break. Subsequent launch() calls
            # would refuse anyway via the budget's own pre-check; this
            # loop-level check is what makes the exit prompt + clean.
            if self.budget is not None and self.budget.snapshot().exhausted:
                snap = self.budget.snapshot()
                budget_reason = (
                    f"token budget exhausted: used ${snap.total_cost_usd:.4f} "
                    f">= limit ${snap.limit_cost_usd:.4f} "
                    f"(agents={snap.agent_count}, "
                    f"in_tokens={snap.total_input_tokens}, "
                    f"out_tokens={snap.total_output_tokens})"
                )
                # Close the iteration as failed (the only closed state
                # besides success) but stamp outcome=ABORTED so the
                # record distinguishes "ran out of budget" from "agent
                # gave up". The TASK-level run.json::final_status is
                # where we surface "aborted" to the WebUI.
                self._close_iteration(
                    iter_rec, status="failed",
                    failure=budget_reason, perf=ctx.this_iter_perf,
                    outcome=P.ABORTED,
                )
                self.store.append_timeline(
                    "token_budget_exhausted",
                    {
                        "phase": phase, "iteration": iter_num,
                        "used_cost_usd": snap.total_cost_usd,
                        "limit_cost_usd": snap.limit_cost_usd,
                        "agent_count": snap.agent_count,
                        "per_source": snap.per_source,
                        "per_phase": snap.per_phase,
                    },
                )
                final_status = "aborted"
                phase = "finished"
                break

            if (
                phase == "D_review"
                and outcome == P.OK
                and self.cfg.execution_mode == "correctness_only"
            ):
                self.store.append_timeline(
                    "correctness_target_reached",
                    {"iteration": iter_num, "mode": self.cfg.execution_mode},
                )
                self._write_retrospective(
                    n=iter_num,
                    iter_dir=iter_dir,
                    ctx=ctx,
                    rec=iter_rec,
                    this_perf=ctx.this_iter_perf,
                    e_ok=True,
                    e_error="performance phases skipped by correctness-only mode",
                )
                self._close_iteration(
                    iter_rec, status="success", failure=None,
                    perf=ctx.this_iter_perf, outcome=P.OK,
                )
                final_status = "success"
                phase = "finished"
                break

            if phase == "G_perf_review" and outcome == P.OK:
                target_result = self._throughput_target_met(
                    perf or ctx.last_perf or ctx.this_iter_perf,
                )
                if target_result is not None:
                    self.store.append_timeline(
                        "performance_target_reached",
                        {"iteration": iter_num, **target_result},
                    )
                    self._close_iteration(
                        iter_rec, status="success", failure=None,
                        perf=perf or ctx.last_perf or ctx.this_iter_perf,
                        outcome=P.OK,
                    )
                    final_status = "success"
                    phase = "finished"
                    break

            # ---- escalation: too many attempts at this phase in this folder #
            # Instead of aborting the whole run, close this iteration as
            # failed and start a fresh one back at A_plan, carrying the
            # failure reason forward so the new planner can re-plan around
            # it. The run only stops legitimately at the iteration cap
            # (max_iterations). This satisfies the invariant: a task never
            # enters the `failed` terminal state while iterations remain.
            if outcome != P.OK and ctx.phase_attempts[phase] >= MAX_PHASE_ATTEMPTS:
                self.store.append_timeline(
                    "phase_escalation",
                    {"phase": phase, "attempts": ctx.phase_attempts[phase]},
                )
                forced_failure = (
                    failure
                    or f"{phase} exceeded {MAX_PHASE_ATTEMPTS} in-place attempts "
                       f"(last outcome={outcome})"
                )
                self._close_iteration(
                    iter_rec, status="failed",
                    failure=forced_failure, perf=ctx.this_iter_perf,
                    outcome=outcome,
                )
                self.store.append_timeline(
                    "iteration_escalation",
                    {"phase": phase, "attempts": ctx.phase_attempts[phase],
                     "iteration": iter_num, "last_outcome": str(outcome)},
                )
                ctx.failure = forced_failure
                ctx.last_outcome = outcome
                iter_dir = None
                iter_rec = None
                phase = "A_plan"
                continue

            # ---- consult the transition table ---------------------------- #
            t = P.next_transition(phase, outcome)
            if t is None:
                # Undefined (phase, outcome) — don't abort the whole run.
                # Close this iteration and start a fresh one. The undefined
                # transition is treated as a soft skip so the run keeps
                # trying until the iteration cap.
                err = f"no transition defined for ({phase}, {outcome})"
                forced_failure = failure or err
                self._close_iteration(
                    iter_rec, status="failed",
                    failure=err, perf=ctx.this_iter_perf,
                    outcome=outcome,
                )
                self.store.append_timeline(
                    "orchestrator_skip_iteration",
                    {"reason": err, "phase": phase, "outcome": str(outcome),
                     "iteration": iter_num},
                )
                ctx.failure = forced_failure
                ctx.last_outcome = outcome
                iter_dir = None
                iter_rec = None
                phase = "A_plan"
                continue

            # ---- update ctx ------------------------------------------------ #
            if t.carry_failure:
                ctx.failure = failure
            elif outcome == P.OK:
                ctx.failure = None
            # else: keep the existing ctx.failure (carry-over for in-place retries)

            if perf:
                if t.carry_perf:
                    ctx.last_perf = perf
                ctx.best_perf = _merge_best(ctx.best_perf, perf)
                # Always also remember it as this iteration's own perf.
                # This is what gets stamped onto the iteration record at
                # close time; scoping it per-iteration prevents stale
                # carry-over from prior iterations.
                ctx.this_iter_perf = perf

            # ---- record + advance ----------------------------------------- #
            self.store.append_timeline(
                "transition",
                {"from": phase, "outcome": outcome, "to": t.to_phase,
                 "label": t.label, "iteration": iter_num,
                 "consume_iteration": t.consume_iteration},
            )
            self.store.update_run(
                current_phase=t.to_phase,
                last_outcome=outcome,
                last_transition_label=t.label,
                current_iteration=iter_num,
            )

            if t.consume_iteration:
                iter_status = "success" if outcome == P.OK else "failed"
                self._close_iteration(
                    iter_rec,
                    status=iter_status,
                    failure=(failure if outcome != P.OK else None),
                    perf=ctx.this_iter_perf,
                    outcome=outcome,
                )
                iter_dir = None
                iter_rec = None

            phase = t.to_phase

        # ---- loop done ----------------------------------------------------- #
        if final_status is None:
            # Loop exited because phase reached "finished" (shouldn't happen
            # in practice — no transition goes to finished) or self._stop
            # was set externally. Either way: never "failed".
            final_status = "success" if ctx.last_outcome == P.OK else "stopped"
        self.store.update_run(
            finished=True,
            final_status=final_status,
            current_phase="finished",
            last_outcome=ctx.last_outcome,
        )

    # ------------------------------------------------------------------ #
    # Phase dispatcher
    # ------------------------------------------------------------------ #

    def _run_phase(
        self,
        phase: P.Phase,
        iter_num: int,
        iter_dir: Path,
        rec: IterationRecord,
        ctx: IterationContext,
    ) -> Tuple[P.Outcome, Optional[Dict[str, float]], Optional[str]]:
        if phase == "A_plan":
            return self._do_plan(iter_num, iter_dir, ctx)
        if phase == "B_implement":
            return self._do_implement(iter_num, iter_dir, ctx)
        if phase == "C_test":
            return self._do_test(iter_num, iter_dir, ctx)
        if phase == "D_review":
            return self._do_review(iter_num, iter_dir, ctx)
        if phase == "E_perf_test":
            return self._do_perf_test(iter_num, iter_dir, ctx, rec)
        if phase == "G_perf_review":
            return self._do_perf_review(iter_num, iter_dir, ctx, rec)
        if phase == "F_perf_plan":
            return self._do_perf_plan(iter_num, iter_dir, ctx)
        raise ValueError(f"no handler for phase {phase!r}")

    # ---- A: plan --------------------------------------------------------- #

    def _do_plan(
        self, n: int, iter_dir: Path, ctx: IterationContext,
    ) -> Tuple[P.Outcome, Optional[Dict[str, float]], Optional[str]]:
        previous = _load_iter(self.store, n - 1) if n > 1 else None
        gate_context = resolve_plan_gate_context(
            iter_dir,
            self.req,
            n,
            previous.to_dict() if previous is not None else None,
        )
        ok, err, mode, _sid = self._run_agent(
            name=f"iter{n}-planner", role="planner", iteration=n, iter_dir=iter_dir,
            prompt=plan_prompt(
                req=self.req, iter_dir=iter_dir,
                notebooks_dir=self.cfg.notebooks_dir,
                iteration=n, prev_failures=ctx.failure,
                review_feedback=ctx.review_feedback,
                perf_plan=ctx.perf_plan,
                logs_dir=self._logs_dir_for(n),
                plan_gate_feedback=ctx.plan_gate_feedback,
                plan_gate_context=gate_context,
            ),
            timeout=self.cfg.plan_timeout_s,
        )
        if not ok:
            return _failure_outcome(mode), None, err

        gate = validate_plan(iter_dir, self.req, n, gate_context)
        attempt = ctx.phase_attempts.get("A_plan", 0) + 1
        report = {
            **gate.to_dict(),
            "iteration": n,
            "attempt": attempt,
        }
        report_path = self._logs_dir_for(n) / "plan-gate-report.json"
        try:
            report_path.write_text(
                json.dumps(report, indent=2, sort_keys=True), encoding="utf-8",
            )
        except OSError as exc:
            return P.INFRA_FAIL, None, f"cannot write Plan Gate report: {exc}"
        event = "plan_gate_passed" if gate.passed else "plan_gate_rejected"
        self.store.append_timeline(event, {
            "iteration": n,
            "attempt": attempt,
            "errors": list(gate.errors),
            "report": str(report_path),
            "gate_mode": gate.context.mode,
            "baseline_iteration": gate.context.baseline_iteration,
        })
        if gate.passed:
            ctx.plan_gate_feedback = None
            return P.OK, None, None

        ctx.plan_gate_feedback = gate.diagnostics()
        return P.LOGIC_FAIL, None, (
            "Plan Gate rejected the A-stage artifacts:\n" + gate.diagnostics()
        )

    # ---- B: implement --------------------------------------------------- #
    # The reviewer no longer runs inside B. It runs AFTER C (see
    # _run_post_test_review below) so it can comment on the actual test
    # outcome, not just on a guess at what would pass. Its suggestions
    # land in ctx.review_feedback and are threaded into the next iteration's
    # planner / implementer / optimizer prompts.

    def _do_implement(
        self, n: int, iter_dir: Path, ctx: IterationContext,
    ) -> Tuple[P.Outcome, Optional[Dict[str, float]], Optional[str]]:
        # When redoing B in-place (transition table routes B LOGIC_FAIL → B,
        # same iter_dir), resume the prior turn's ccb session. The
        # implementer then keeps its loaded plan + source files + the
        # diagnosis it was working through, and only pays cache-read rates
        # for the bulk of that context. ctx.b_session_id is reset to None
        # whenever a new iteration folder opens, so cross-iteration B
        # sessions never leak into each other.
        is_redo = ctx.b_session_id is not None
        if is_redo:
            prompt = implement_redo_prompt(
                req=self.req, iter_dir=iter_dir,
                notebooks_dir=self.cfg.notebooks_dir,
                iteration=n, prev_failure=ctx.failure,
                logs_dir=self._logs_dir_for(n),
            )
        else:
            prompt = implement_prompt(
                req=self.req, iter_dir=iter_dir,
                notebooks_dir=self.cfg.notebooks_dir,
                iteration=n, prev_failure=ctx.failure,
                review_feedback=ctx.review_feedback,
                perf_plan=ctx.perf_plan,
                logs_dir=self._logs_dir_for(n),
            )
        ok, err, mode, sid = self._run_agent(
            name=f"iter{n}-implementer", role="implementer",
            iteration=n, iter_dir=iter_dir,
            prompt=prompt,
            timeout=self.cfg.impl_timeout_s,
            resume_session_id=ctx.b_session_id,
        )
        # Track the session for the next redo in the same iteration. If
        # the agent returned a different id (e.g. ccb had to fork on
        # resume), prefer the new one.
        if sid:
            ctx.b_session_id = sid
        if not ok:
            return _failure_outcome(mode), None, f"B (implement) failed: {err}"
        gate = validate_implementation(iter_dir, n)
        gate_path = self._logs_dir_for(n) / "implementation-gate-report.json"
        try:
            gate_path.write_text(gate.diagnostics(), encoding="utf-8")
        except OSError as exc:
            return P.INFRA_FAIL, None, (
                f"cannot write Implementation Gate report: {exc}"
            )
        self.store.append_timeline(
            "implementation_gate_passed" if gate.passed
            else "implementation_gate_rejected",
            {
                "iteration": n,
                "errors": list(gate.errors),
                "expected_files": list(gate.expected_files),
                "changed_files": list(gate.changed_files),
                "report": str(gate_path),
            },
        )
        if not gate.passed:
            return P.LOGIC_FAIL, None, (
                "Implementation Gate rejected the B-stage delivery:\n"
                + gate.diagnostics()
            )
        native_gate = execute_native_build_gate(
            iter_dir,
            self._logs_dir_for(n),
            build_type=str(self.req.get("build_type") or "Release"),
            timeout_s=self._bounded_timeout(min(self.cfg.impl_timeout_s, 900)),
        )
        self.store.append_timeline(
            "implementation_native_gate_passed" if native_gate.passed
            else "implementation_native_gate_rejected",
            {
                "iteration": n,
                "errors": list(native_gate.errors),
                "steps": list(native_gate.steps),
                "report": native_gate.report_path,
            },
        )
        if not native_gate.passed:
            return P.LOGIC_FAIL, None, (
                "Independent native build gate rejected the B-stage tree:\n"
                + native_gate.diagnostics()
            )
        # On clean pass, drop the session — the next B (if any) will be a
        # new iteration's fresh start, not a redo of this one.
        ctx.b_session_id = None
        return P.OK, None, None

    # ---- D: review + retro ---------------------------------------------- #
    #
    # D runs AFTER every C (any outcome). It is advisory — the reviewer's
    # verdict never gates anything. Its job is to write review.md with
    # concrete improvement suggestions that go into ctx.review_feedback for
    # the next iteration.
    #
    # D's "outcome" is DERIVED from C's outcome (not from whether the reviewer
    # agent itself succeeded). The transition table maps:
    #   (D_review, OK)         → E_perf_test   [meaning: C had passed]
    #   (D_review, LOGIC_FAIL) → A_plan        [meaning: C had failed]
    # So we set D's outcome to OK iff ctx.last_outcome (C's outcome) was OK.

    def _do_review(
        self, n: int, iter_dir: Path, ctx: IterationContext,
    ) -> Tuple[P.Outcome, Optional[Dict[str, float]], Optional[str]]:
        # Pull the failure + perf from C's outcome (already in ctx).
        c_outcome = ctx.last_outcome
        c_failure = ctx.failure
        c_perf = ctx.last_perf
        ok, _err, _mode, _sid = self._run_agent(
            name=f"iter{n}-reviewer", role="reviewer", iteration=n, iter_dir=iter_dir,
            prompt=review_prompt(
                req=self.req, iter_dir=iter_dir,
                notebooks_dir=self.cfg.notebooks_dir, iteration=n,
                outcome=c_outcome, failure=c_failure, perf=c_perf,
                logs_dir=self._logs_dir_for(n),
            ),
            timeout=self.cfg.review_timeout_s,
        )
        # Capture review.md as next-iteration feedback regardless of whether
        # the reviewer agent succeeded — the file may exist even on agent
        # failure if it crashed after writing it.
        review_path = self._logs_dir_for(n) / "review.md"
        feedback: Optional[str] = None
        if review_path.is_file():
            try:
                text = review_path.read_text(encoding="utf-8", errors="replace")
                if len(text) > 8192:
                    text = text[:8160] + "\n...[truncated]..."
                feedback = text
            except OSError:
                feedback = None
        ctx.review_feedback = feedback
        self.store.append_timeline(
            "review_done",
            {"iteration": n, "c_outcome": c_outcome,
             "feedback_captured": feedback is not None,
             "reviewer_agent_ok": ok},
        )
        # D's outcome drives the next-phase routing. C passed → OK → E.
        # C failed (any flavor) → LOGIC_FAIL → A (new iter). Reviewer agent
        # failure does NOT change the routing — D is advisory.
        if c_outcome == P.OK:
            return P.OK, None, None
        return P.LOGIC_FAIL, None, (
            c_failure or f"C_test ended with {c_outcome or P.LOGIC_FAIL}"
        )

    # ---- E: perf test (only on C-pass) ---------------------------------- #

    def _do_perf_test(
        self, n: int, iter_dir: Path, ctx: IterationContext, rec: IterationRecord,
    ) -> Tuple[P.Outcome, Optional[Dict[str, float]], Optional[str]]:
        # For HTTP-served framework generators, bypass the
        # LLM perf-tester agent and run the immutable perf oracle directly.
        # The agent-written perf.sh was unreliable across iterations
        # (different prompts, different concurrency, different warmup),
        # making iter(N) vs iter(N-1) comparisons meaningless. The oracle
        # owns the benchmark → deterministic and comparable.
        if is_http_framework_task(self.req.get("task_type", "")):
            return self._do_perf_test_oracle(n, iter_dir, ctx, rec)

        ok, err, mode, _sid = self._run_agent(
            name=f"iter{n}-perf-tester", role="perf_tester",
            iteration=n, iter_dir=iter_dir,
            prompt=perf_test_prompt(
                req=self.req, iter_dir=iter_dir,
                notebooks_dir=self.cfg.notebooks_dir, iteration=n,
                review_feedback=ctx.review_feedback,
                logs_dir=self._logs_dir_for(n),
            ),
            timeout=self.cfg.optimize_timeout_s,
        )
        perf: Optional[Dict[str, float]] = None
        e_error: Optional[str] = None
        if not ok:
            e_error = err
            # Fall through to the retro writer — we still want a summary
            # explaining that E failed, even without fresh perf data.
        else:
            # Parse perf_report.json if the agent wrote one — surface its
            # numbers as this step's perf so F can use them.
            perf = self._read_perf_report(n, iter_dir)

        if not ok:
            return _failure_outcome(mode), None, f"E (perf test) failed: {err}"
        return P.OK, perf, None

    def _do_perf_test_oracle(
        self, n: int, iter_dir: Path, ctx: IterationContext, rec: IterationRecord,
    ) -> Tuple[P.Outcome, Optional[Dict[str, float]], Optional[str]]:
        """E-step using the immutable PerfOracle instead of an LLM agent.

        Boots the iteration's serve.sh, sweeps the fixed concurrency ladder
        against the fixed prompt set, writes perf_report.json. No agent
        involved → numbers are directly comparable across iterations.
        """
        from .oracles.perf import PerfOracle

        logs_dir = self._logs_dir_for(n)
        logs_dir.mkdir(parents=True, exist_ok=True)
        prev_perf = dict(ctx.last_perf) if ctx.last_perf else None

        oracle = PerfOracle()
        try:
            report = oracle.run(
                iter_dir=iter_dir,
                req=self.req,
                report_dir=logs_dir,
                iteration=n,
                prev_perf=prev_perf,
                timeout_s=self._bounded_timeout(self.cfg.optimize_timeout_s),
            )
        except Exception as e:  # noqa: BLE001 — oracle crash shouldn't kill the run
            err = f"perf oracle crashed: {e!r}"
            return P.INFRA_FAIL, None, f"E (perf oracle) crashed: {err}"

        # The oracle writes perf_report.json; reuse the existing parser.
        perf = self._read_perf_report(n, iter_dir)
        e_ok = bool(perf) and perf.get("tokens_per_sec", 0) > 0
        accelerator_validation = report.hardware.get("accelerator_validation") or {}
        if accelerator_validation and not accelerator_validation.get("passed", False):
            e_ok = False
        e_error = None if e_ok else (report.notes or "perf oracle produced no usable data")

        if not e_ok:
            # Treat as INFRA_FAIL rather than a logic failure — a perf
            # oracle that can't get numbers usually means the server
            # wouldn't boot or answer, which is the same class of problem
            # as a crashed C step. Don't burn a C-repair slot on it.
            return P.INFRA_FAIL, perf, f"E (perf oracle): {e_error}"
        return P.OK, perf, None

    # ---- G: measured performance review -------------------------------- #

    def _do_perf_review(
        self, n: int, iter_dir: Path, ctx: IterationContext, rec: IterationRecord,
    ) -> Tuple[P.Outcome, Optional[Dict[str, float]], Optional[str]]:
        """Review E's measured report before F proposes another optimization.

        G is advisory.  A reviewer failure never discards valid benchmark
        data or blocks F, but any produced report is carried into F and the
        next iteration's A/B prompts.
        """
        logs_dir = self._logs_dir_for(n)
        ok, err, mode, _sid = self._run_agent(
            name=f"iter{n}-perf-reviewer", role="perf_reviewer",
            iteration=n, iter_dir=iter_dir,
            prompt=perf_review_prompt(
                req=self.req,
                iter_dir=iter_dir,
                notebooks_dir=self.cfg.notebooks_dir,
                iteration=n,
                measured_perf=ctx.last_perf or ctx.this_iter_perf,
                correctness_review=ctx.review_feedback,
                logs_dir=logs_dir,
            ),
            timeout=self.cfg.review_timeout_s,
            max_retries=0,
        )
        review_path = logs_dir / "perf-review.md"
        feedback: Optional[str] = None
        if review_path.is_file():
            try:
                feedback = review_path.read_text(
                    encoding="utf-8", errors="replace",
                )
                if len(feedback) > 8192:
                    feedback = feedback[:8160] + "\n...[truncated]..."
            except OSError:
                feedback = None
        if feedback:
            prefix = ctx.review_feedback or ""
            ctx.review_feedback = (
                prefix + "\n\n# Measured performance review\n" + feedback
            ).strip()
        self.store.append_timeline("perf_review_done", {
            "iteration": n,
            "agent_ok": ok,
            "feedback_captured": feedback is not None,
            "measured_perf": ctx.last_perf or ctx.this_iter_perf,
            "error": err,
            "failure_mode": mode,
        })
        self._write_retrospective(
            n=n,
            iter_dir=iter_dir,
            ctx=ctx,
            rec=rec,
            this_perf=ctx.last_perf or ctx.this_iter_perf,
            e_ok=True,
            e_error=None,
        )
        return P.OK, ctx.last_perf or ctx.this_iter_perf, None

    def _write_retrospective(
        self,
        n: int,
        iter_dir: Path,
        ctx: IterationContext,
        rec: IterationRecord,
        *,
        this_perf: Optional[Dict[str, float]],
        e_ok: bool,
        e_error: Optional[str],
    ) -> None:
        """Spawn a short-lived agent that writes ``retrospective.md`` into
        this iteration's logs dir, then record its path on the iteration
        record so the WebUI can find it. Non-gating — any failure is
        logged and swallowed.

        ``rec`` is the SAME in-memory IterationRecord the main loop holds
        and will hand to ``_close_iteration``. Mutate it directly so the
        retro path survives the close write — loading a fresh copy from
        the store here would be a lost update (close would overwrite it).
        """
        prev_rec = _load_iter(self.store, n - 1) if n > 1 else None
        prev_perf = dict(prev_rec.perf) if prev_rec and prev_rec.perf else None
        # Goal: prefer the most recent plan-level goal we can find. We
        # don't currently store a one-line goal separately on the record,
        # so fall back to the iteration's failure_reason (when C failed)
        # or a placeholder. The retro agent reads plan.md anyway.
        goal: Optional[str] = rec.failure_reason or rec.goal or None

        logs_dir = self._logs_dir_for(n)
        try:
            ok, _err, _mode, _sid = self._run_agent(
                name=f"iter{n}-retro", role="retro_writer",
                iteration=n, iter_dir=iter_dir,
                prompt=retrospective_prompt(
                    req=self.req, iter_dir=iter_dir,
                    notebooks_dir=self.cfg.notebooks_dir, iteration=n,
                    this_perf=this_perf, prev_perf=prev_perf,
                    review_feedback=ctx.review_feedback,
                    e_ok=e_ok, e_error=e_error,
                    logs_dir=logs_dir, goal=goal,
                ),
                timeout=self.cfg.retro_timeout_s,
            )
        except Exception as exc:  # noqa: BLE001 — never let retro break E
            self.store.append_timeline(
                "retrospective_error",
                {"iteration": n, "error": f"agent raised: {exc!r}"},
            )
            return

        retro_path = logs_dir / "retrospective.md"
        if not retro_path.is_file():
            self.store.append_timeline(
                "retrospective_missing",
                {"iteration": n, "agent_ok": ok,
                 "expected_path": str(retro_path)},
            )
            return

        # Stamp the path on the IN-MEMORY record the main loop is holding.
        # _close_iteration will persist it (along with everything else
        # the loop has accumulated on `rec` this turn). We also write the
        # record now so the path is durable even if the orchestrator dies
        # between here and the close write.
        rec.retrospective_path = str(retro_path)
        _write_iter(self.store, rec)
        self.store.append_timeline(
            "retrospective_written",
            {"iteration": n, "path": str(retro_path),
             "bytes": retro_path.stat().st_size if retro_path.exists() else 0},
        )

    def _write_failure_retrospective(
        self,
        n: int,
        iter_dir: Path,
        rec: IterationRecord,
    ) -> None:
        """Spawn a postmortem agent for a FAILED iteration.

        Failed iterations never reach E (perf test), so the perf-centric
        :meth:`_write_retrospective` doesn't apply. This method writes a
        failure-focused ``retrospective.md`` at the same path the WebUI
        already serves — no UI changes needed.

        Recovers the failing phase + attempt count from the iteration
        record's ``phases`` map. Non-gating — any agent/file failure is
        logged and swallowed (a failure-within-a-failure must not break
        the close path).

        ``rec`` is mutated in place (``retrospective_path`` set) so the
        caller — typically :meth:`_close_iteration` — can persist it as
        part of the same final write.
        """
        # Find the failing phase: prefer the last non-OK phase recorded on
        # the iteration. The record's phases dict is keyed by phase id.
        failed_phase: Optional[str] = None
        phase_attempts: Optional[int] = None
        if rec.phases:
            # Walk in phase order so we pick the FURTHEST reached phase
            # that has a non-OK outcome.
            for phase_id in ("A_plan", "B_implement", "C_test",
                              "D_review", "E_perf_test", "G_perf_review",
                              "F_perf_plan"):
                info = rec.phases.get(phase_id)
                if not info:
                    continue
                if info.get("outcome") != P.OK:
                    failed_phase = phase_id
                    phase_attempts = info.get("attempts")
                    # keep going — a later phase might also be marked failed
                    # if e.g. C failed and the loop ran E with an error flag

        logs_dir = self._logs_dir_for(n)
        goal: Optional[str] = rec.goal or rec.failure_reason or None
        try:
            ok, _err, _mode, _sid = self._run_agent(
                name=f"iter{n}-fail-retro", role="retro_writer",
                iteration=n, iter_dir=iter_dir,
                prompt=failure_retrospective_prompt(
                    req=self.req, iter_dir=iter_dir,
                    notebooks_dir=self.cfg.notebooks_dir, iteration=n,
                    failure_reason=rec.failure_reason,
                    failed_phase=failed_phase,
                    phase_attempts=phase_attempts,
                    logs_dir=logs_dir, goal=goal,
                ),
                timeout=self.cfg.retro_timeout_s,
            )
        except Exception as exc:  # noqa: BLE001 — never let retro break close
            self.store.append_timeline(
                "failure_retrospective_error",
                {"iteration": n, "error": f"agent raised: {exc!r}"},
            )
            return

        retro_path = logs_dir / "retrospective.md"
        if not retro_path.is_file():
            self.store.append_timeline(
                "failure_retrospective_missing",
                {"iteration": n, "agent_ok": ok,
                 "expected_path": str(retro_path)},
            )
            return

        rec.retrospective_path = str(retro_path)
        self.store.append_timeline(
            "failure_retrospective_written",
            {"iteration": n, "path": str(retro_path),
             "failed_phase": failed_phase,
             "bytes": retro_path.stat().st_size if retro_path.exists() else 0},
        )

    # ---- F: perf plan (writes plan, no code changes) -------------------- #

    def _do_perf_plan(
        self, n: int, iter_dir: Path, ctx: IterationContext,
    ) -> Tuple[P.Outcome, Optional[Dict[str, float]], Optional[str]]:
        ok, err, mode, _sid = self._run_agent(
            name=f"iter{n}-perf-planner", role="perf_planner",
            iteration=n, iter_dir=iter_dir,
            prompt=perf_plan_prompt(
                req=self.req, iter_dir=iter_dir,
                notebooks_dir=self.cfg.notebooks_dir, iteration=n,
                last_perf=ctx.last_perf or ctx.best_perf,
                review_feedback=ctx.review_feedback,
                logs_dir=self._logs_dir_for(n),
            ),
            timeout=self.cfg.review_timeout_s,
        )
        # Capture perf_plan.md as ctx.perf_plan regardless of agent success —
        # the file may exist even on agent failure. The next iteration's A
        # will read it via ctx.
        plan_path = iter_dir / "perf_plan.md"
        plan_text: Optional[str] = None
        if plan_path.is_file():
            try:
                text = plan_path.read_text(encoding="utf-8", errors="replace")
                if len(text) > 12288:  # 12 KiB cap — plans can be longer than reviews
                    text = text[:12256] + "\n...[truncated]..."
                plan_text = text
            except OSError:
                plan_text = None
        ctx.perf_plan = plan_text
        self.store.append_timeline(
            "perf_plan_done",
            {"iteration": n, "plan_captured": plan_text is not None,
             "plan_bytes": len(plan_text) if plan_text else 0,
             "agent_ok": ok},
        )
        if ok:
            return P.OK, ctx.last_perf, None
        return _failure_outcome(mode), None, f"F (perf plan) failed: {err}"

    def _read_perf_report(
        self, n: int, iter_dir: Path,
    ) -> Optional[Dict[str, float]]:
        """Parse perf_report.json (written by E) into a flat {metric: float}
        dict. Best-effort — returns None on any failure."""
        path = iter_dir / "perf_report.json"
        if not path.is_file():
            return None
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(obj, dict):
            return None
        return {k: float(v) for k, v in obj.items() if _is_num(v)}

    # ---- C: test -------------------------------------------------------- #

    def _do_test(
        self, n: int, iter_dir: Path, ctx: IterationContext,
    ) -> Tuple[P.Outcome, Optional[Dict[str, float]], Optional[str]]:
        # task_type with a registered oracle → run the immutable oracle instead
        # of any agent-written test.sh. The oracle lives outside iter_dir and
        # cannot be modified by the implementer.
        #
        # C no longer runs the reviewer inline — that's D's job now. C just
        # produces an outcome + perf, which D will use.
        #
        # C has an in-place repair budget (cfg.max_c_retries). On a LOGIC_FAIL,
        # a debugger sub-agent is dispatched in the SAME iteration folder to
        # fix the code, and the test is re-run. After the budget is exhausted
        # the final LOGIC_FAIL is returned and the transition table routes
        # to D_review → A_plan (new iter). INFRA_FAIL and PERF_REGRESSION
        # surface immediately without consuming repair attempts (the former
        # is an environment issue, the latter is a perf-only signal where the
        # code is already correct).
        #
        # The correctness oracle is THIS task's own oracle — imported directly
        # from the plugin's oracles/ subpackage. No global registry, no
        # task_type lookup: the pipeline and its oracle live in the same
        # package by design (high cohesion).
        from .oracles.correctness import InferFrameworkOracle
        oracle = InferFrameworkOracle()

        # ---- default path bootstrap: ensure test.sh exists ---------------- #
        # Dead path today (oracle import can't fail in practice), but kept
        # as a defensive fallback: if the local oracle module is somehow
        # unimportable, fall back to letting the agent write test.sh.
        if oracle is None:
            test_sh = iter_dir / "test.sh"
            if not test_sh.exists():
                ok, err, mode, _sid = self._run_agent(
                    name=f"iter{n}-testwriter", role="testwriter",
                    iteration=n, iter_dir=iter_dir,
                    prompt=write_test_harness_prompt(
                        req=self.req, iter_dir=iter_dir,
                        notebooks_dir=self.cfg.notebooks_dir, iteration=n,
                    ),
                    timeout=self.cfg.review_timeout_s,
                )
                if not ok or not test_sh.exists():
                    return _failure_outcome(mode), None, f"C (test harness) missing: {err}"

        # ---- repair loop --------------------------------------------------- #
        max_attempts = max(1, int(self.cfg.max_c_retries))
        last_outcome: Optional[P.Outcome] = None
        last_perf: Optional[Dict[str, float]] = None
        last_failure: Optional[str] = None
        # ccb conversation UUID shared across all debugger turns in this C
        # step. Attempt 1 mints a fresh session and we capture its id;
        # attempts 2..N pass it back via --resume so the debugger keeps its
        # prior diagnosis / file reads / edits in context. The bulk of the
        # re-loaded context hits cache_read_input_tokens (~10x cheaper than
        # re-seeding), and the agent doesn't have to re-discover what it
        # already figured out about the failing iteration's code.
        c_session_id: Optional[str] = None
        repair_log_path = self._logs_dir_for(n) / "c-repairs.jsonl"
        # Truncate at loop entry so each iteration's C-step log starts clean
        # (resume / retry-in-place scenarios would otherwise append onto a
        # stale file from a previous attempt at the same iteration number).
        try:
            repair_log_path.write_text("", encoding="utf-8")
        except OSError:
            pass

        for attempt in range(1, max_attempts + 1):
            self.store.append_timeline("c_test_attempt", {
                "iteration": n, "attempt": attempt, "max": max_attempts,
                "mode": "oracle" if oracle is not None else "test.sh",
            })

            t_attempt_start = time.time()
            if oracle is not None:
                outcome, perf, failure = self._run_oracle_once(n, iter_dir, ctx, oracle)
            else:
                outcome, perf, failure = self._run_test_once(n, iter_dir, ctx)
            last_outcome, last_perf, last_failure = outcome, perf, failure
            attempt_duration = time.time() - t_attempt_start

            # Happy path: C passed. Check perf regression vs running best.
            if outcome == P.OK:
                if perf and ctx.best_perf and self.cfg.primary_perf_metric:
                    if _is_regression(ctx.best_perf, perf,
                                      metric=self.cfg.primary_perf_metric,
                                      threshold=PERF_REGRESSION_THRESHOLD):
                        self._append_repair_record(
                            repair_log_path, n, attempt,
                            input_failure=failure, repair_md=None,
                            debugger_ok=True, debugger_err=None,
                            test_outcome=P.PERF_REGRESSION, test_perf=perf,
                            test_failure=None, duration_s=attempt_duration,
                            note="passed but perf regression",
                        )
                        return P.PERF_REGRESSION, perf, (
                            f"perf regression on {self.cfg.primary_perf_metric}: "
                            f"best={ctx.best_perf.get(self.cfg.primary_perf_metric)} "
                            f"new={perf.get(self.cfg.primary_perf_metric)}"
                        )
                self._append_repair_record(
                    repair_log_path, n, attempt,
                    input_failure=failure, repair_md=None,
                    debugger_ok=True, debugger_err=None,
                    test_outcome=P.OK, test_perf=perf,
                    test_failure=None, duration_s=attempt_duration,
                    note="passed (no repair needed)" if attempt == 1
                         else f"passed after {attempt-1} repair(s)",
                )
                return P.OK, perf, None

            # PERF_REGRESSION / INFRA_FAIL surface immediately (no repair).
            if outcome != P.LOGIC_FAIL:
                self._append_repair_record(
                    repair_log_path, n, attempt,
                    input_failure=failure, repair_md=None,
                    debugger_ok=False, debugger_err=None,
                    test_outcome=outcome, test_perf=perf,
                    test_failure=failure, duration_s=attempt_duration,
                    note=f"{outcome} (no repair attempted)",
                )
                return outcome, perf, failure

            # LOGIC_FAIL: dispatch debugger and re-run, unless budget exhausted.
            if attempt >= max_attempts:
                self.store.append_timeline("c_test_budget_exhausted", {
                    "iteration": n, "attempts": attempt,
                    "final_failure": (failure or "")[:500],
                })
                self._append_repair_record(
                    repair_log_path, n, attempt,
                    input_failure=failure, repair_md=None,
                    debugger_ok=False, debugger_err="budget exhausted",
                    test_outcome=outcome, test_perf=perf,
                    test_failure=failure, duration_s=attempt_duration,
                    note="repair budget exhausted, surfacing to D",
                )
                break

            self.store.append_timeline("c_test_repair_start", {
                "iteration": n, "attempt": attempt,
                "reason": (failure or "")[:500],
                "resuming_session": c_session_id is not None,
            })
            dbg_name = f"iter{n}-c-debugger.attempt{attempt}"
            t_repair_start = time.time()
            source_before = source_fingerprint(iter_dir)
            # First repair turn: full bootstrap prompt (knowledge base
            # hints, framework rules, deliverable contract, etc.) +
            # fresh ccb session. Subsequent turns: short follow-up that
            # assumes the prior context (already loaded files, already
            # made diagnosis) and just reports the new failure + asks
            # for the next minimal fix. The follow-up runs under
            # --resume so the bulk of context is served from cache.
            if c_session_id is None:
                prompt = c_repair_prompt(
                    req=self.req, iter_dir=iter_dir,
                    notebooks_dir=self.cfg.notebooks_dir,
                    iteration=n, attempt=attempt, max_attempts=max_attempts,
                    failure=failure, logs_dir=self._logs_dir_for(n),
                )
            else:
                prompt = c_repair_followup_prompt(
                    iteration=n, attempt=attempt, max_attempts=max_attempts,
                    new_failure=failure, logs_dir=self._logs_dir_for(n),
                )
            ok, err, mode, sid = self._run_agent(
                name=dbg_name,
                role="c_debugger",
                iteration=n, iter_dir=iter_dir,
                prompt=prompt,
                timeout=getattr(self.cfg, "c_repair_timeout_s", 1200),
                resume_session_id=c_session_id,
                max_retries=0,
            )
            # Capture the session id from the first turn so later turns
            # can resume. If the turn somehow returned a different id
            # (e.g. ccb had to fork), prefer the new one — resuming by
            # a stale id would fail with "No conversation found".
            if sid:
                c_session_id = sid
            repair_duration = time.time() - t_repair_start
            repair_md_path = self._logs_dir_for(n) / f"c-repair-attempt{attempt}.md"
            repair_md = None
            if repair_md_path.is_file():
                try:
                    repair_md = repair_md_path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    repair_md = None
            dbg_final = ""
            try:
                r = self.manager.result(dbg_name)
                if r is not None:
                    dbg_final = r.final_text or ""
            except Exception:  # noqa: BLE001
                pass

            source_after = source_fingerprint(iter_dir)
            changed_paths = sorted(
                path for path in set(source_before) | set(source_after)
                if source_before.get(path) != source_after.get(path)
            )

            if not ok:
                self.store.append_timeline("c_test_repair_agent_fail", {
                    "iteration": n, "attempt": attempt,
                    "error": err, "mode": mode,
                })
            # Record this repair attempt even if the debugger crashed — the
            # forensics log is for understanding what was tried, not just
            # what succeeded.
            self._append_repair_record(
                repair_log_path, n, attempt,
                input_failure=failure, repair_md=repair_md,
                debugger_ok=ok, debugger_err=err,
                debugger_mode=mode, debugger_final=dbg_final,
                test_outcome=None, test_perf=None, test_failure=None,
                duration_s=repair_duration,
                changed_files=changed_paths,
                note="debugger crashed" if not ok else "debugger done, re-run pending",
            )

            requested_replan = bool(
                repair_md and "REPLAN_REQUIRED:" in repair_md
            )
            if requested_replan:
                last_failure = (
                    (failure or "C correctness failure")
                    + "\nC debugger classified this as an architecture gap and "
                      "requested a fresh A plan."
                )
                self.store.append_timeline("c_test_replan_requested", {
                    "iteration": n, "attempt": attempt,
                    "changed_files": changed_paths,
                })
                break
            if not changed_paths:
                last_failure = (
                    (failure or "C correctness failure")
                    + "\nC debugger produced no source diff; fail-fast to a "
                      "fresh A plan instead of repeating the same diagnosis."
                )
                self.store.append_timeline("c_test_no_source_diff", {
                    "iteration": n, "attempt": attempt,
                })
                break

        # Budget exhausted: return the last LOGIC_FAIL. Transition table
        # routes (C_test, LOGIC_FAIL) → D_review → A_plan (new iter).
        return last_outcome or P.LOGIC_FAIL, last_perf, last_failure

    def _append_repair_record(
        self,
        path: Path,
        iteration: int,
        attempt: int,
        *,
        input_failure: Optional[str],
        repair_md: Optional[str],
        debugger_ok: Optional[bool],
        debugger_err: Optional[str],
        debugger_mode: Optional[str] = None,
        debugger_final: str = "",
        test_outcome: Optional[P.Outcome],
        test_perf: Optional[Dict[str, float]],
        test_failure: Optional[str],
        duration_s: float,
        changed_files: Optional[List[str]] = None,
        note: str = "",
    ) -> None:
        """Append one structured record to ``<logs_dir>/c-repairs.jsonl``.

        Each line is a self-contained JSON object capturing what the C-step
        repair loop did at one attempt: the failure that triggered it, the
        debugger's diagnosis + diff (from the agent-authored
        ``c-repair-attempt{N}.md``), whether the debugger itself succeeded,
        and the test outcome after the fix (if the re-run has happened).
        Post-run, ``jq`` over this file answers questions like "which
        iteration needed the most repairs" or "what root cause recurred".
        """
        rec = {
            "iteration": iteration,
            "attempt": attempt,
            "timestamp": time.time(),
            "input_failure": (input_failure or "")[:2000],
            "repair": None,
            "debugger": {
                "ok": debugger_ok,
                "error": debugger_err,
                "mode": debugger_mode,
                "duration_s": round(duration_s, 1) if duration_s else None,
                "final_text_head": (debugger_final or "")[:500],
            } if debugger_ok is not None else None,
            "test": {
                "outcome": test_outcome,
                "perf": test_perf,
                "failure": (test_failure or "")[:2000] if test_failure else None,
            } if test_outcome is not None else None,
            "changed_files": sorted(set(changed_files or [])),
            "note": note,
        }
        # Embed the agent-written repair markdown verbatim (truncated) so
        # the jsonl is self-contained for forensics even if the .md files
        # get cleaned up.
        if repair_md:
            rec["repair"] = repair_md[:8000]
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError:
            pass

    # ---- C (oracle path): one-shot oracle run --------------------------- #

    def _run_oracle_once(
        self,
        n: int,
        iter_dir: Path,
        ctx: IterationContext,
        oracle,
    ) -> Tuple[P.Outcome, Optional[Dict[str, float]], Optional[str]]:
        report_dir = self._logs_dir_for(n)
        report_dir.mkdir(parents=True, exist_ok=True)
        self.store.append_timeline("oracle_start",
                                   {"iteration": n, "oracle": oracle.task_type})
        try:
            result = oracle.run(
                iter_dir=iter_dir, req=self.req, report_dir=report_dir,
                timeout_s=self._bounded_timeout(self.cfg.impl_timeout_s),
                manager=self.manager,
            )
        except Exception as exc:  # noqa: BLE001
            err = f"oracle exception: {exc!r}"
            self.store.append_timeline("oracle_end",
                                       {"iteration": n, "passed": False, "error": err})
            return P.INFRA_FAIL, None, err

        self.store.append_timeline("oracle_end", {
            "iteration": n, "passed": result.passed,
            "judge_mode": result.judge_mode,
            "cases_total": len(result.cases),
            "cases_passed": sum(1 for c in result.cases if c.judge_verdict == "pass"),
            "failure_reason": result.failure_reason,
        })

        if not result.passed:
            # Map to LOGIC_FAIL; the repair loop in _do_test will decide
            # whether to retry in-place or surface.
            return P.LOGIC_FAIL, result.perf or None, self._render_oracle_failure(n, result)

        return P.OK, result.perf or None, None

    # ---- C (test.sh path): one-shot test.sh run ------------------------- #

    def _run_test_once(
        self, n: int, iter_dir: Path, ctx: IterationContext,
    ) -> Tuple[P.Outcome, Optional[Dict[str, float]], Optional[str]]:
        test_sh = iter_dir / "test.sh"
        if not test_sh.exists():
            return P.INFRA_FAIL, None, f"test.sh missing in {iter_dir}"
        success, perf, failure = self._run_test(test_sh, iter_dir, n)
        if success:
            return P.OK, perf, None
        if failure and _looks_like_infra(failure):
            return P.INFRA_FAIL, None, failure
        return P.LOGIC_FAIL, perf, failure

    def _render_oracle_failure(self, iter_num: int, result: Any) -> str:
        """Compose a structured failure description from an OracleResult.

        The string is what gets carried through ``ctx.failure`` into the next
        iteration's implementer/optimizer prompt. We surface:

        * the oracle's headline failure_reason
        * the failing case ids + their per-case judge reasons (truncated)
        * explicit ABSOLUTE pointers to the prev-iter diagnostic files
          (the snapshot copied forward into the next iteration's logs dir).
          Absolute paths are required because logs now live OUTSIDE the
          iteration code dir (under ``<cwd>/.metainfer/logs/<task_id>/``),
          so a relative ``.metainfer-logs/...`` path no longer resolves
          from the agent's CWD.

        The next iteration's logs dir is ``logs_dir_for(iter_num + 1)``,
        which is where :meth:`IterationWorkspace._copy_prev_diagnostics`
        will land the snapshot during ``open_iteration(iter_num + 1)``.
        """
        from metainfer.orchestrator.iteration import PREV_ITER_LOGS_SUBDIR

        next_logs = self._logs_dir_for(iter_num + 1)
        snap = next_logs / PREV_ITER_LOGS_SUBDIR

        lines: List[str] = []
        head = result.failure_reason or "oracle did not pass"
        lines.append(f"[iter {iter_num:03d} C_test LOGIC_FAIL] {head}")

        failed_cases = [c for c in result.cases if c.judge_verdict == "fail"]
        if failed_cases:
            lines.append("")
            lines.append(f"Failing test cases ({len(failed_cases)}):")
            for c in failed_cases[:8]:  # cap to keep the prompt bounded
                reason = (c.judge_reason or c.error or "").strip()
                if len(reason) > 240:
                    reason = reason[:237] + "..."
                prompt_head = (c.prompt or "").strip().replace("\n", " ")
                if len(prompt_head) > 80:
                    prompt_head = prompt_head[:77] + "..."
                lines.append(f"  - case {c.case_id!r} [{c.judge_verdict}] "
                             f"prompt={prompt_head!r} reason={reason!r}")
            if len(failed_cases) > 8:
                lines.append(f"  ... ({len(failed_cases) - 8} more, see oracle-report.json)")

        if result.judge_mode and result.judge_mode != "llm":
            lines.append(f"(judge_mode={result.judge_mode})")

        lines.append("")
        lines.append(
            "Diagnostic files from this failed iteration will be copied into "
            f"the next iteration's logs snapshot at `{snap}/`. READ them "
            "before writing code:"
        )
        lines.append(f"  - {snap / 'oracle-report.json'}  "
                     "(full structured verdict + per-case responses)")
        lines.append(f"  - {snap / 'server.stderr.log'}   "
                     "(server error output — first place to look for crashes)")
        lines.append(f"  - {snap / 'server.stdout.log'}   "
                     "(server stdout — startup banner, model load info)")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Agent runner (returns (ok, error, failure_mode))
    # ------------------------------------------------------------------ #

    def _run_agent(
        self,
        name: str,
        role: str,
        iteration: int,
        iter_dir: Path,
        prompt: str,
        timeout: int,
        *,
        resume_session_id: Optional[str] = None,
        session_id: Optional[str] = None,
        max_retries: int = 2,
    ) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
        """Launch one sub-agent.

        Returns ``(ok, error, failure_mode, session_id)``. The ``session_id``
        is the ccb conversation UUID this agent ran under — pass it back via
        ``resume_session_id`` on a later call to continue the same
        conversation (loaded files, prior tool results, diagnoses all stay
        in context, with the bulk of tokens served from cache).

        ``session_id`` (when set on the FIRST turn of a fresh conversation)
        pins the UUID so the caller knows what to resume from. Both are
        optional; if neither is set, ccb mints a fresh session.
        """
        logs_dir = self._logs_dir_for(iteration)
        logs_dir.mkdir(parents=True, exist_ok=True)
        prompt_file = logs_dir / f"{name}.prompt.txt"
        prompt_file.write_text(prompt, encoding="utf-8")
        effective_timeout = self._bounded_timeout(timeout)
        if self._wall_time_exhausted():
            return False, self._wall_time_reason(), "infra", None
        spec = AgentSpec(
            name=name,
            role=role,
            prompt_file=prompt_file,
            workdir=iter_dir,
            log_dir=logs_dir,
            timeout_s=effective_timeout,
            stuck_timeout_s=min(self.cfg.stuck_timeout_s, effective_timeout),
            max_retries=max_retries,
            extra_args=list(self.cfg.extra_claude_args),
            session_id=session_id,
            resume_session_id=resume_session_id,
        )
        self.store.append_timeline(
            "agent_launch", {"name": name, "role": role, "iteration": iteration,
                             "resume_from": resume_session_id,
                             "session_id_pinned": session_id})
        self.manager.launch(spec)
        result = self.manager.result(name)
        if result is None:
            return False, "no result recorded", "infra", None

        # record into the iteration record's phases dict (best-effort)
        try:
            rec = _load_iter(self.store, iteration)
            if rec is not None:
                rec.phases.setdefault(role, {}).update({
                    "duration_s": result.duration_s,
                    "success": result.success,
                    "error": result.error,
                    "failure_mode": result.failure_mode,
                    "attempts": result.attempts,
                    "final_text_head": result.final_text[:1000],
                    "log_file": str(spec.log_file(result.attempts)),
                    "session_id": result.session_id,
                })
                _write_iter(self.store, rec)
        except Exception:  # noqa: BLE001
            pass

        self.store.append_timeline(
            "agent_end",
            {"name": name, "role": role, "iteration": iteration,
             "success": result.success, "error": result.error,
             "failure_mode": result.failure_mode,
             "duration_s": result.duration_s, "attempts": result.attempts,
             "session_id": result.session_id},
        )
        return result.success, result.error, result.failure_mode, result.session_id

    # ------------------------------------------------------------------ #
    # Phase / iteration bookkeeping
    # ------------------------------------------------------------------ #

    def _set_phase(self, n: int, iter_dir: Path, phase: P.Phase) -> None:
        self.store.update_run(current_iteration=n, current_phase=phase)
        self.store.append_timeline("phase_start", {"iteration": n, "phase": phase})
        rec = _load_iter(self.store, n)
        if rec is not None:
            rec.phases.setdefault(phase, {"started_at": time.time()})
            _write_iter(self.store, rec)

    def _close_iteration(
        self,
        rec: IterationRecord,
        *,
        status: str,
        failure: Optional[str],
        perf: Optional[Dict[str, float]],
        outcome: Optional[P.Outcome],
        write_retrospective: bool = True,
    ) -> None:
        rec.ended_at = time.time()
        rec.duration_s = rec.ended_at - rec.started_at
        rec.status = status
        rec.failure_reason = failure
        rec.outcome = outcome
        if perf:
            rec.perf = perf
        # For FAILED iterations that never reached E (so no perf-centric
        # retro was written), spawn a failure postmortem BEFORE persisting.
        # Skip when a retro path is already set (covers the rare case where
        # E ran, failed, and already produced a retro).
        if status == "failed" and write_retrospective and not rec.retrospective_path:
            try:
                iter_dir = self.workspace.iter_dir(rec.iteration)
                self._write_failure_retrospective(rec.iteration, iter_dir, rec)
            except Exception as exc:  # noqa: BLE001
                self.store.append_timeline(
                    "failure_retrospective_error",
                    {"iteration": rec.iteration,
                     "error": f"close-path raised: {exc!r}"},
                )
        _write_iter(self.store, rec)
        # Mark the folder complete *after* the record write so the sentinel
        # is a durable signal that the close path ran end-to-end. An
        # iteration that lacks this sentinel on resume was killed mid-flight.
        self.workspace.mark_complete(rec.iteration)
        self.store.append_timeline(
            "iteration_end",
            {"iteration": rec.iteration, "status": status, "outcome": outcome,
             "perf": perf, "failure_reason": failure, "duration_s": rec.duration_s},
        )

    def _resolve_max_iterations(self) -> int:
        """Resolve the iteration cap. cfg.max_iterations is itself seeded
        from requirements by ``_extract_max_iter`` (orchestrator.py), which
        already reads top-level ``max_iterations`` — so by the time we get
        here ``self.cfg.max_iterations`` carries the user's intent. But
        double-check the top-level field too in case the cfg was built by
        a code path that bypassed ``_extract_max_iter`` (e.g. tests,
        hand-rolled configs). Top-level wins; ``answers.`` is a back-compat
        fallback.
        """
        v = self.req.get("max_iterations")
        if v is None:
            v = self.req.get("answers", {}).get("max_iterations")
        if v is None:
            return self.cfg.max_iterations
        try:
            return int(v)
        except (TypeError, ValueError):
            return self.cfg.max_iterations

    # ------------------------------------------------------------------ #
    # Test runner
    # ------------------------------------------------------------------ #

    def _run_test(
        self, test_sh: Path, iter_dir: Path, iteration: int
    ) -> Tuple[bool, Optional[Dict[str, float]], Optional[str]]:
        log_path = self._logs_dir_for(iteration) / f"iter{iteration}-test.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env["METAINFER_ITER_DIR"] = str(iter_dir)
        env["METAINFER_ITERATION"] = str(iteration)
        try:
            with open(log_path, "w", encoding="utf-8") as logf:
                proc = subprocess.run(
                    ["bash", str(test_sh)],
                    cwd=str(iter_dir),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=logf,
                    timeout=self._bounded_timeout(self.cfg.impl_timeout_s),
                    text=True,
                )
        except subprocess.TimeoutExpired:
            return False, None, "test timed out"
        except Exception as exc:  # noqa: BLE001
            return False, None, f"test runner exception: {exc!r}"

        stdout = proc.stdout or ""
        parsed = self._parse_test_json(stdout)
        if parsed is None:
            return (
                False, None,
                f"test did not emit parseable JSON. stdout tail: {stdout[-1000:]!r}",
            )
        passed = bool(parsed.get("passed", False))
        perf = parsed.get("perf") if isinstance(parsed.get("perf"), dict) else {}
        perf = {k: float(v) for k, v in perf.items() if _is_num(v)}
        if passed:
            return True, perf, None
        err = parsed.get("error") or parsed.get("traceback") or "test failed (no error field)"
        return False, perf, str(err)[:4000]

    @staticmethod
    def _parse_test_json(stdout: str) -> Optional[Dict[str, Any]]:
        """Grab the LAST JSON object printed on its own line that has ``passed``."""
        candidates: List[str] = []
        for ln in reversed(stdout.splitlines()):
            ln = ln.strip()
            if ln.startswith("{") and ln.endswith("}"):
                candidates.append(ln)
                if len(candidates) >= 5:
                    break
        for c in candidates:
            try:
                obj = json.loads(c)
                if isinstance(obj, dict) and "passed" in obj:
                    return obj
            except json.JSONDecodeError:
                continue
        m = list(JSON_LINE_RE.finditer(stdout))
        for mt in reversed(m):
            try:
                obj = json.loads(mt.group(0))
                if isinstance(obj, dict) and "passed" in obj:
                    return obj
            except json.JSONDecodeError:
                continue
        return None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _failure_outcome(mode: Optional[str]) -> P.Outcome:
    """Map a SubAgentManager failure_mode to an orchestrator Outcome."""
    if mode == "infra":
        return P.INFRA_FAIL
    return P.LOGIC_FAIL


def _looks_like_infra(failure: str) -> bool:
    """Heuristic: test-runner-level infra failures (vs logic bugs in the code)."""
    f = failure.lower()
    return any(s in f for s in ("timed out", "timeout", "exception", "traceback",
                                "no such file", "permission denied", "killed"))


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _merge_best(
    best: Optional[Dict[str, float]], new: Dict[str, float]
) -> Dict[str, float]:
    """Keep the max of each metric (higher is better; safe default for
    throughput / tokens-per-second / etc.)."""
    out = dict(best or {})
    for k, v in new.items():
        if k not in out or v > out[k]:
            out[k] = v
    return out


def _is_regression(
    best: Dict[str, float],
    new: Dict[str, float],
    metric: str,
    threshold: float,
) -> bool:
    if metric not in best or metric not in new:
        return False
    b = best[metric]
    if not b:
        return False
    return (new[metric] / b) < (1.0 - threshold)
