"""Deterministic state-machine-driven orchestrator for knowledge evolution.

Each high-level phase (A/B/D) runs an internal multi-agent sub-pipeline
mirroring gen-infer-framework's planner -> implementer -> oracle -> reviewer
pattern. Phase C is unique to knowledge-evolution (consolidator).

Loop shape::

    phase = initial_phase()  # A_attempt_pure or B_enrich depending on config
    while not terminal(phase):
        if no open iteration folder: open one
        outcome = run_phase(phase, ...)
          +-- internal: plan -> implement -> oracle -> review
        t = TRANSITIONS[(phase, outcome)]
        update ctx
        record iteration + timeline
        if t.consume_iteration: close folder
        phase = t.to_phase
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from metainfer.orchestrator.iteration import IterationWorkspace
from metainfer.orchestrator.state import Phase, StateStore
from metainfer.orchestrator.subagent_manager import AgentSpec, SubAgentManager
from metainfer.tasks.gen_infer_framework.orchestrator.oracles.correctness import (
    InferFrameworkOracle,
)

from . import phases as P
from . import prompts

JSON_LINE_RE = re.compile(r"\{.*\}")

MAX_PHASE_ATTEMPTS = 3
MAX_CONSECUTIVE_ENRICH_FAILURES = 3
MAX_C_RETRIES = 3
PERF_REGRESSION_THRESHOLD = 0.2


# ---- IterationRecord (KE-specific schema) ----

@dataclass
class IterationRecord:
    """One iteration of the KE 4-phase pipeline (A→B→C→D).

    Uses plain dict fields so the shell's StateStore can persist it as JSON.
    """
    iteration: int = 0
    goal: str = ""
    start_phase: Phase = ""
    started_at: float = 0.0
    ended_at: float = 0.0
    duration_s: float = 0.0
    status: str = "running"
    failure_reason: Optional[str] = None
    outcome: Optional[str] = None
    phases: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    perf: Dict[str, float] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    interrupted: bool = False
    retrospective_path: Optional[str] = None
    # Non-dataclass runtime-only field:
    closed_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("closed_at", None)  # runtime-only, never persisted
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IterationRecord":
        names = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in names})


# ---- Helpers ----

def _failure_outcome(mode: Optional[str] = None) -> P.Outcome:
    """Classify a failure mode into an outcome.

    If the sub-agent errored out due to infrastructure (timeout, GPU OOM, etc.),
    return INFRA_FAIL. Otherwise return LOGIC_FAIL.
    """
    if mode and mode.lower() in ("error", "timeout", "infra", "infra_fail"):
        return P.INFRA_FAIL
    return P.LOGIC_FAIL


# ---- Data classes ----


@dataclass
class IterationContext:
    """Mutable context carried across iterations within a single run."""

    failure: Optional[str] = None
    this_iter_perf: Optional[Dict[str, float]] = None
    last_outcome: Optional[P.Outcome] = None
    phase_attempts: Dict[str, int] = field(default_factory=dict)
    consecutive_enrich_failures: int = 0
    verify_failures: int = 0
    open_source_used: bool = False
    exploration_report: Optional[str] = None
    review_feedback: Optional[str] = None
    b_session_id: Optional[str] = None
    c_session_id: Optional[str] = None
    goal: Optional[str] = None


@dataclass
class EvolutionConfig:
    """Immutable configuration for an evolution run."""

    workdir: Path
    repo_root: Path
    notebooks_dir: Path
    iterations_root: Path
    state_dir: Path
    logs_root: Optional[Path] = None
    max_iterations: int = 20
    max_verify_attempts: int = 3
    plan_timeout_s: int = 1800
    impl_timeout_s: int = 3600
    impl_timeout_s_phase_a: int = 2400
    review_timeout_s: int = 1800
    retro_timeout_s: int = 900
    stuck_timeout_s: int = 600
    max_c_retries: int = MAX_C_RETRIES
    enable_phase_a: bool = True
    claude_bin: str = "ccb"
    model: Optional[str] = None
    permission_mode: str = "bypassPermissions"
    extra_claude_args: List[str] = field(default_factory=list)


# ---- Evolution Orchestrator ----


class EvolutionOrchestrator:
    """Top-level orchestrator for the 4-phase knowledge-evolution loop.

    ``run()`` executes the state machine: read current phase, execute it,
    record results, transition, repeat until a terminal phase is reached.
    """

    def __init__(
        self,
        req: Dict[str, Any],
        store: StateStore,
        cfg: EvolutionConfig,
        manager: Optional[SubAgentManager] = None,
    ) -> None:
        self.req = req
        self.store = store
        self.cfg = cfg
        self.manager = manager or SubAgentManager(
            claude_bin=cfg.claude_bin,
            default_model=cfg.model,
            permission_mode=cfg.permission_mode,
            extra_add_dirs=[
                cfg.notebooks_dir,
                *([cfg.logs_root] if cfg.logs_root else []),
            ],
        )
        self.workspace = IterationWorkspace(
            cfg.iterations_root, logs_root=cfg.logs_root,
        )
        self._stop = False
        self.nooped = False

    # ---- agent_status file (KE-private, avoids touching shared RunStatus) ----

    def _set_agent_status(self, status: Optional[str]) -> None:
        """Persist current agent activity so the WebUI can show a live pill."""
        path = self.store.task_dir / "agent_status"
        if status:
            path.write_text(status.strip(), encoding="utf-8")
        elif path.exists():
            path.unlink()

    def _initial_phase(self) -> P.Phase:
        """Return the starting phase: A_attempt_pure if enabled, else B_enrich."""
        if self.cfg.enable_phase_a:
            return "A_attempt_pure"
        return "B_enrich"

    def _impl_timeout_for(self, phase: P.Phase) -> int:
        """Return the implementer/oracle timeout for *phase*.

        Phase A has a shorter timeout because it works without open-source
        reference code — expected to fail fast and move on to B_enrich.
        """
        if phase == "A_attempt_pure":
            return self.cfg.impl_timeout_s_phase_a
        return self.cfg.impl_timeout_s

    # ---- Main entry ----

    def run(self) -> None:
        """Execute the state machine to completion (or until stopped)."""
        task_id = self.req.get("task_id", "task")
        _, is_resume = self.store.init_or_resume(task_id=task_id)

        resume_from: Optional[Dict[str, Any]] = None
        if is_resume:
            rs = self.store.load_run()
            if rs.finished:
                self.store.append_timeline(
                    "orchestrator_restart",
                    {"prior_final_status": rs.final_status,
                     "prior_phase": rs.current_phase},
                )
                self.store.update_run(
                    finished=False, final_status=None,
                    current_phase="idle", last_outcome=None,
                    last_transition_label=None,
                )
            self.store.append_timeline("orchestrator_resume", {"task_id": task_id})
            resume_from = self._prepare_resume()
        else:
            self.store.append_timeline("orchestrator_start", {"task_id": task_id})

        try:
            self._loop(resume_from=resume_from)
        except KeyboardInterrupt:
            self.store.append_timeline(
                "orchestrator_abort", {"reason": "keyboard-interrupt"},
            )
            self.store.update_run(
                finished=True, final_status="aborted",
                current_phase="finished",
            )
        finally:
            self.manager.shutdown()
            self.store.append_timeline("orchestrator_end", {"task_id": task_id})

    def stop(self) -> None:
        """Signal the orchestrator to stop at the next safe point."""
        self._stop = True

    # ---- State machine loop ----

    def _prepare_resume(self) -> Dict[str, Any]:
        """Check for a previous run and return resume state."""
        discarded = self.workspace.discard_latest_incomplete()
        if discarded is not None:
            old_dict = self.store.load_iteration(discarded)
            old_rec = IterationRecord.from_dict(old_dict) if old_dict else None
            interrupted_in = None
            if old_rec is not None and old_rec.phases:
                interrupted_in = max(
                    old_rec.phases.keys(),
                    key=lambda k: old_rec.phases[k].get("started_at", 0),
                )
            reason = (
                f"interrupted mid-{interrupted_in}" if interrupted_in
                else "interrupted unexpectedly"
            )
            self.store.archive_interrupted_iteration(discarded, reason=reason)
            self.store.append_timeline(
                "iteration_interrupted",
                {"iteration": discarded, "reason": reason},
            )
            start_phase = (old_rec.start_phase if old_rec else self._initial_phase()) or self._initial_phase()
            iter_num = discarded
            carried_failure = None
            last_outcome: Optional[P.Outcome] = None
            prev_dict = self.store.load_iteration(iter_num - 1) if iter_num > 1 else None
            if prev_dict is not None:
                prev_rec = IterationRecord.from_dict(prev_dict)
                last_outcome = prev_rec.outcome
                if prev_rec.outcome != P.OK:
                    carried_failure = prev_rec.failure_reason
        else:
            last_complete = self.workspace.latest_complete_number()
            prev_dict = self.store.load_iteration(last_complete) if last_complete else None
            iter_num = last_complete + 1
            if prev_dict is not None:
                prev_rec = IterationRecord.from_dict(prev_dict)
                if prev_rec.outcome != P.OK:
                    start_phase = self._phase_after(prev_rec)
                    carried_failure = prev_rec.failure_reason
                    last_outcome = prev_rec.outcome
                else:
                    start_phase = self._initial_phase()
                    carried_failure = None
                    last_outcome = prev_rec.outcome
            else:
                start_phase = self._initial_phase()
                carried_failure = None
                last_outcome = None

        return {
            "iter_num": iter_num,
            "start_phase": start_phase,
            "carried_failure": carried_failure,
            "last_outcome": last_outcome,
        }

    def _phase_after(self, rec: IterationRecord) -> P.Phase:
        """Determine the starting phase after a previous iteration record."""
        if rec.outcome == P.OK:
            return self._initial_phase()
        return "B_enrich"

    def _loop(self, resume_from: Optional[Dict[str, Any]] = None) -> None:
        """Main state-machine loop."""
        max_iters = self._resolve_max_iterations()
        ctx = IterationContext()

        if resume_from is not None:
            phase: P.Phase = resume_from["start_phase"]
            iter_num = resume_from["iter_num"] - 1
            ctx.failure = resume_from.get("carried_failure")
            ctx.last_outcome = resume_from.get("last_outcome")
        else:
            phase: P.Phase = self._initial_phase()
            iter_num = 0

        iter_dir: Optional[Path] = None
        iter_rec: Optional[IterationRecord] = None
        final_status: Optional[str] = None

        while not self._stop and not P.is_terminal(phase):
            # ---- open iteration folder ---------------------------------- #
            if iter_dir is None:
                if iter_num >= max_iters:
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
                self.store.write_iteration(iter_num, iter_rec.to_dict())
                ctx.this_iter_perf = None
                ctx.b_session_id = None
                ctx.c_session_id = None
                ctx.goal = None
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

            # ---- run the phase ------------------------------------------ #
            assert iter_rec is not None
            self._set_phase(iter_num, iter_dir, phase)
            outcome, perf, failure = self._run_phase(
                phase, iter_num, iter_dir, iter_rec, ctx,
            )

            ctx.last_outcome = outcome
            ctx.phase_attempts[phase] = ctx.phase_attempts.get(phase, 0) + 1

            phase_rec = iter_rec.phases.setdefault(phase, {})
            phase_rec["outcome"] = outcome
            phase_rec["attempts"] = ctx.phase_attempts[phase]
            phase_rec["ended_at"] = time.time()
            if failure:
                phase_rec["failure"] = failure
            if perf:
                phase_rec["perf"] = perf

            # ---- escalation check --------------------------------------- #
            effective_max = MAX_PHASE_ATTEMPTS
            if phase == "D_verify_final":
                max_va = self._resolve_max_verify_attempts()
                effective_max = max_va
            if outcome != P.OK and ctx.phase_attempts[phase] >= effective_max:
                self.store.append_timeline(
                    "phase_escalation",
                    {"phase": phase, "attempts": ctx.phase_attempts[phase],
                     "effective_max": effective_max},
                )
                forced_failure = (
                    failure or f"{phase} exceeded {effective_max} in-place attempts"
                )

                if phase == "B_enrich":
                    ctx.consecutive_enrich_failures += 1
                    if ctx.consecutive_enrich_failures >= MAX_CONSECUTIVE_ENRICH_FAILURES:
                        final_status = "halted"
                        self.store.append_timeline(
                            "evolution_halted",
                            {"reason": "consecutive enrich failures exceeded",
                             "count": ctx.consecutive_enrich_failures},
                        )
                        self._close_iteration(
                            iter_rec, status="failed",
                            failure=forced_failure, perf=ctx.this_iter_perf,
                            outcome=outcome,
                        )
                        phase = "finished"
                        break

                if phase == "D_verify_final":
                    if self._halt_if_verify_exhausted(
                        ctx, iter_rec,
                        timeline_event="verify_escalation",
                        outcome=outcome,
                    ):
                        final_status = "halted"
                        phase = "finished"
                        break

                self._close_iteration(
                    iter_rec, status="failed",
                    failure=forced_failure, perf=ctx.this_iter_perf,
                    outcome=outcome,
                )
                ctx.failure = forced_failure
                ctx.last_outcome = outcome
                iter_dir = None
                iter_rec = None
                if phase == "B_enrich":
                    phase = "B_enrich"
                elif phase == "D_verify_final":
                    phase = "B_enrich"
                else:
                    phase = "B_enrich"
                self._set_agent_status(f"retrying: {phase}")
                continue

            # ---- transition table --------------------------------------- #
            t = P.next_transition(phase, outcome)
            if t is None:
                forced_failure = failure or f"no transition for ({phase}, {outcome})"
                self._close_iteration(
                    iter_rec, status="failed",
                    failure=forced_failure, perf=ctx.this_iter_perf,
                    outcome=outcome,
                )
                ctx.failure = forced_failure
                ctx.last_outcome = outcome
                iter_dir = None
                iter_rec = None
                phase = "B_enrich"
                self._set_agent_status(f"retrying: {phase}")
                continue

            # ---- update ctx --------------------------------------------- #
            if t.carry_failure:
                ctx.failure = failure
            elif outcome == P.OK:
                ctx.failure = None

            if perf:
                ctx.this_iter_perf = perf

            # ---- record + advance --------------------------------------- #
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
            self._set_agent_status(None)

            # ---- spawn Failure Analyst on failed oracle phases ---------- #
            # Any phase whose oracle found a failure gets distilled into
            # notebooks/06_experience and 08_issues/ so the knowledge
            # gained (even failed knowledge) persists across iterations.
            if failure and outcome in (P.LOGIC_FAIL, P.INFRA_FAIL):
                self._spawn_failure_analyst(
                    iter_num, iter_dir, failure,
                    source_open=(phase == "B_enrich"),
                )

            # ---- verify failure counter (D_verify_final → D_verify_final) - #
            if phase == "D_verify_final" and outcome == P.LOGIC_FAIL:
                if self._halt_if_verify_exhausted(
                    ctx, iter_rec,
                    timeline_event="verify_failure",
                    outcome=outcome,
                ):
                    final_status = "halted"
                    iter_dir = None
                    iter_rec = None
                    phase = "finished"
                    break

            if t.consume_iteration:
                iter_status = "success" if outcome == P.OK else "failed"
                self._close_iteration(
                    iter_rec, status=iter_status,
                    failure=(failure if outcome != P.OK else None),
                    perf=ctx.this_iter_perf,
                    outcome=outcome,
                )
                iter_dir = None
                iter_rec = None

            phase = t.to_phase

        # ---- loop done ------------------------------------------------- #
        if final_status is None:
            final_status = "success" if ctx.last_outcome == P.OK else "stopped"
        self.store.update_run(
            finished=True, final_status=final_status,
            current_phase="finished", last_outcome=ctx.last_outcome,
        )
        self._set_agent_status(None)

    # ---- Phase execution ----

    def _run_phase(
        self,
        phase: P.Phase,
        iter_num: int,
        iter_dir: Path,
        iter_rec: IterationRecord,
        ctx: IterationContext,
    ) -> Tuple[P.Outcome, Optional[Dict[str, float]], Optional[str]]:
        """Dispatch to phase handler.

        A/B/D phases run the multi-agent sub-pipeline (plan → implement →
        oracle → review). C runs the consolidator agent.
        """
        if phase == "C_consolidate":
            return self._do_C_consolidate(iter_num, iter_dir, iter_rec, ctx)
        source_open = (phase == "B_enrich")
        return self._do_attempt(phase, iter_num, iter_dir, iter_rec, ctx, source_open)

    # ---- Multi-agent sub-pipeline (shared by A, B, D) ----

    def _do_attempt(
        self,
        phase: P.Phase,
        iter_num: int,
        iter_dir: Path,
        iter_rec: IterationRecord,
        ctx: IterationContext,
        source_open: bool,
    ) -> Tuple[P.Outcome, Optional[Dict[str, float]], Optional[str]]:
        """Run the multi-agent pipeline: plan → implement → oracle → review.

        Used by A_attempt_pure (source_open=False), B_enrich (source_open=True),
        and D_verify_final (source_open=False).
        """
        phase_label = P.phase_label(phase)
        logs_dir = self._logs_dir_for(iter_num)
        is_redo = ctx.b_session_id is not None

        if not is_redo:
            # ---- Step 1: Plan (skip on redo — plan.md is already in place) #
            self._set_agent_status("running: planner")
            ok, err, mode, _sid = self._run_agent(
                name=f"iter{iter_num}-planner",
                role="planner",
                iteration=iter_num,
                iter_dir=iter_dir,
                prompt=prompts.plan_prompt(
                    req=self.req,
                    iter_dir=iter_dir,
                    notebooks_dir=self.cfg.notebooks_dir,
                    iteration=iter_num,
                    source_open=source_open,
                    prev_failures=ctx.failure,
                    review_feedback=ctx.review_feedback,
                    logs_dir=logs_dir,
                ),
                timeout=self.cfg.plan_timeout_s,
            )
            if not ok:
                self._set_agent_status(f"failed: planner ({mode or 'unknown'})")
                return _failure_outcome(mode), None, f"{phase_label} plan failed: {err}"

            # Capture goal from plan.md for retro writer
            self._capture_goal(iter_dir, ctx)

        # ---- Step 2: Implement ----------------------------------------- #
        self._set_agent_status("running: implementer")
        if is_redo:
            impl_prompt = prompts.implement_redo_prompt(
                req=self.req,
                iter_dir=iter_dir,
                notebooks_dir=self.cfg.notebooks_dir,
                iteration=iter_num,
                prev_failure=ctx.failure,
                logs_dir=logs_dir,
            )
        else:
            impl_prompt = prompts.implement_prompt(
                req=self.req,
                iter_dir=iter_dir,
                notebooks_dir=self.cfg.notebooks_dir,
                iteration=iter_num,
                source_open=source_open,
                prev_failure=ctx.failure,
                review_feedback=ctx.review_feedback,
                logs_dir=logs_dir,
            )
        ok, err, mode, sid = self._run_agent(
            name=f"iter{iter_num}-implementer",
            role="implementer",
            iteration=iter_num,
            iter_dir=iter_dir,
            prompt=impl_prompt,
            timeout=self._impl_timeout_for(phase),
            resume_session_id=ctx.b_session_id,
        )
        if sid:
            ctx.b_session_id = sid
        if not ok:
            self._set_agent_status(f"failed: implementer ({mode or 'unknown'})")
            return _failure_outcome(mode), None, f"{phase_label} implement failed: {err}"
        ctx.b_session_id = None

        # ---- Step 3: Oracle (with c_debugger repair loop) -------------- #
        self._set_agent_status("running: oracle")
        c_outcome, c_perf, c_failure = self._run_oracle_step(
            iter_num, iter_dir, ctx, phase, source_open,
        )

        # ---- Step 4: Review (post-oracle, advisory only) --------------- #
        self._set_agent_status("running: reviewer")
        self._run_review_step(iter_num, iter_dir, ctx, c_outcome, c_failure, source_open)

        # ---- Result ---------------------------------------------------- #
        self._set_agent_status(None)
        if c_outcome == P.OK:
            return P.OK, c_perf, None
        return c_outcome, c_perf, c_failure

    # ---- C_consolidate (unique to knowledge-evolution) ----

    def _do_C_consolidate(
        self,
        iter_num: int,
        iter_dir: Path,
        iter_rec: IterationRecord,
        ctx: IterationContext,
    ) -> Tuple[P.Outcome, Optional[Dict[str, float]], Optional[str]]:
        """Consolidate validated knowledge back into notebooks/."""
        log_dir = self._logs_dir_for(iter_num)
        prompt = prompts.consolidate_prompt(
            req=self.req,
            notebooks_dir=self.cfg.notebooks_dir,
            iter_dir=iter_dir,
            log_dir=log_dir,
        )
        self._set_agent_status("running: consolidator")
        ok, err, mode, _sid = self._run_agent(
            name=f"iter{iter_num}-consolidator",
            role="consolidator",
            iteration=iter_num,
            iter_dir=iter_dir,
            prompt=prompt,
            timeout=self.cfg.plan_timeout_s,
        )
        if ok:
            self._set_agent_status(None)
            return P.OK, None, None
        self._set_agent_status(f"failed: consolidator ({mode or 'unknown'})")
        return _failure_outcome(mode), None, f"C (consolidate) failed: {err}"

    # ---- Oracle + c_debugger repair loop ----

    def _run_oracle_step(
        self,
        iter_num: int,
        iter_dir: Path,
        ctx: IterationContext,
        phase: P.Phase,
        source_open: bool,
    ) -> Tuple[P.Outcome, Optional[Dict[str, float]], Optional[str]]:
        """Run the correctness oracle, with a c_debugger repair loop on failure."""
        oracle = InferFrameworkOracle()
        logs_dir = self._logs_dir_for(iter_num)
        phase_label = P.phase_label(phase)
        impl_timeout = self._impl_timeout_for(phase)
        max_attempts = max(1, int(self.cfg.max_c_retries))
        last_outcome: Optional[P.Outcome] = None
        last_perf: Optional[Dict[str, float]] = None
        last_failure: Optional[str] = None
        c_session_id: Optional[str] = None
        repair_log_path = logs_dir / "c-repairs.jsonl"
        try:
            repair_log_path.write_text("", encoding="utf-8")
        except OSError:
            pass

        for attempt in range(1, max_attempts + 1):
            self.store.append_timeline("c_test_attempt", {
                "iteration": iter_num, "attempt": attempt, "max": max_attempts,
                "phase": phase_label,
            })

            t_start = time.time()
            outcome, perf, failure = self._run_oracle_once(
                iter_num, iter_dir, ctx, oracle, impl_timeout,
            )
            last_outcome, last_perf, last_failure = outcome, perf, failure
            attempt_duration = time.time() - t_start

            if outcome == P.OK:
                self._append_repair_record(
                    repair_log_path, iter_num, attempt,
                    input_failure=failure, repair_md=None,
                    debugger_ok=True, debugger_err=None,
                    test_outcome=P.OK, test_perf=perf,
                    test_failure=None, duration_s=attempt_duration,
                    note="passed (no repair needed)" if attempt == 1
                         else f"passed after {attempt - 1} repair(s)",
                )
                return P.OK, perf, None

            if outcome != P.LOGIC_FAIL:
                self._append_repair_record(
                    repair_log_path, iter_num, attempt,
                    input_failure=failure, repair_md=None,
                    debugger_ok=False, debugger_err=None,
                    test_outcome=outcome, test_perf=perf,
                    test_failure=failure, duration_s=attempt_duration,
                    note=f"{outcome} (no repair attempted)",
                )
                return outcome, perf, failure

            if attempt >= max_attempts:
                self.store.append_timeline("c_test_budget_exhausted", {
                    "iteration": iter_num, "attempts": attempt,
                    "final_failure": (failure or "")[:500],
                })
                self._append_repair_record(
                    repair_log_path, iter_num, attempt,
                    input_failure=failure, repair_md=None,
                    debugger_ok=False, debugger_err="budget exhausted",
                    test_outcome=outcome, test_perf=perf,
                    test_failure=failure, duration_s=attempt_duration,
                    note="repair budget exhausted",
                )
                break

            # ---- dispatch c_debugger ----------------------------------- #
            self.store.append_timeline("c_test_repair_start", {
                "iteration": iter_num, "attempt": attempt,
                "reason": (failure or "")[:500],
                "resuming_session": c_session_id is not None,
            })
            dbg_name = f"iter{iter_num}-c-debugger.attempt{attempt}"
            t_repair = time.time()

            if c_session_id is None:
                prompt = prompts.c_repair_prompt(
                    req=self.req,
                    iter_dir=iter_dir,
                    notebooks_dir=self.cfg.notebooks_dir,
                    iteration=iter_num,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    failure=failure,
                    logs_dir=logs_dir,
                )
            else:
                prompt = prompts.c_repair_followup_prompt(
                    iteration=iter_num,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    new_failure=failure,
                    logs_dir=logs_dir,
                )
            self._set_agent_status(f"running: oracle repair {attempt}/{max_attempts}")
            ok, err, mode, sid = self._run_agent(
                name=dbg_name,
                role="c_debugger",
                iteration=iter_num,
                iter_dir=iter_dir,
                prompt=prompt,
                timeout=impl_timeout,
                resume_session_id=c_session_id,
            )
            if sid:
                c_session_id = sid
            repair_duration = time.time() - t_repair
            repair_md_path = logs_dir / f"c-repair-attempt{attempt}.md"
            repair_md = None
            if repair_md_path.is_file():
                try:
                    repair_md = repair_md_path.read_text(
                        encoding="utf-8", errors="replace",
                    )
                except OSError:
                    repair_md = None

            dbg_final = ""
            try:
                r = self.manager.result(dbg_name)
                if r is not None:
                    dbg_final = r.final_text or ""
            except Exception:
                pass

            if not ok:
                self.store.append_timeline("c_test_repair_agent_fail", {
                    "iteration": iter_num, "attempt": attempt,
                    "error": err, "mode": mode,
                })

            self._append_repair_record(
                repair_log_path, iter_num, attempt,
                input_failure=failure, repair_md=repair_md,
                debugger_ok=ok, debugger_err=err, debugger_mode=mode,
                debugger_final=dbg_final,
                test_outcome=None, test_perf=None, test_failure=None,
                duration_s=repair_duration,
                note="repair applied; re-test pending",
            )

        return last_outcome or P.LOGIC_FAIL, last_perf, last_failure

    # ---- Review ----

    def _run_review_step(
        self,
        iter_num: int,
        iter_dir: Path,
        ctx: IterationContext,
        c_outcome: P.Outcome,
        c_failure: Optional[str],
        source_open: bool,
    ) -> None:
        """Run the post-oracle reviewer and capture its feedback."""
        logs_dir = self._logs_dir_for(iter_num)
        outcome_str = "ok" if c_outcome == P.OK else "logic_fail"

        ok, _err, _mode, _sid = self._run_agent(
            name=f"iter{iter_num}-reviewer",
            role="reviewer",
            iteration=iter_num,
            iter_dir=iter_dir,
            prompt=prompts.review_prompt(
                req=self.req,
                iter_dir=iter_dir,
                notebooks_dir=self.cfg.notebooks_dir,
                iteration=iter_num,
                outcome=outcome_str,
                failure=c_failure,
                logs_dir=logs_dir,
            ),
            timeout=self.cfg.review_timeout_s,
        )

        # Capture review.md as next-iteration feedback
        review_path = logs_dir / "review.md"
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
            {"iteration": iter_num, "c_outcome": outcome_str,
             "feedback_captured": feedback is not None,
             "reviewer_agent_ok": ok},
        )

    # ---- Oracle single run ----

    def _run_oracle_once(
        self,
        n: int,
        iter_dir: Path,
        ctx: IterationContext,
        oracle: InferFrameworkOracle,
        timeout_s: int,
    ) -> Tuple[P.Outcome, Optional[Dict[str, float]], Optional[str]]:
        """Run the correctness oracle once against the code in iter_dir."""
        report_dir = self._logs_dir_for(n)
        report_dir.mkdir(parents=True, exist_ok=True)
        self.store.append_timeline(
            "oracle_start",
            {"iteration": n, "oracle": oracle.task_type},
        )
        try:
            result = oracle.run(
                iter_dir=iter_dir, req=self.req, report_dir=report_dir,
                timeout_s=timeout_s, manager=self.manager,
            )
        except Exception as exc:
            err = f"oracle exception: {exc!r}"
            self.store.append_timeline(
                "oracle_end",
                {"iteration": n, "passed": False, "error": err},
            )
            return P.INFRA_FAIL, None, err

        self.store.append_timeline("oracle_end", {
            "iteration": n, "passed": result.passed,
            "judge_mode": result.judge_mode,
            "cases_total": len(result.cases),
            "cases_passed": sum(1 for c in result.cases if c.judge_verdict == "pass"),
            "failure_reason": result.failure_reason,
        })

        if not result.passed:
            return (
                P.LOGIC_FAIL,
                result.perf or None,
                result.failure_reason or "oracle reported failure",
            )

        return P.OK, result.perf or None, None

    # ---- Failure Analyst ----

    def _spawn_failure_analyst(
        self,
        iter_num: int,
        iter_dir: Path,
        failure_reason: str,
        source_open: bool = False,
    ) -> None:
        """Spawn a Failure Analyst agent to distill error knowledge into notebooks/.

        Runs after Phase A or D failures. Non-gating — failures are logged
        but never change the phase outcome.
        """
        log_dir = self._logs_dir_for(iter_num)
        prompt = prompts.failure_analyst_prompt(
            req=self.req,
            notebooks_dir=self.cfg.notebooks_dir,
            iter_dir=iter_dir,
            log_dir=log_dir,
            failure_reason=failure_reason,
            source_open=source_open,
        )
        try:
            self._set_agent_status("running: failure analyst")
            ok, err, _mode, _sid = self._run_agent(
                name=f"iter{iter_num}-failure-analyst",
                role="failure_analyst",
                iteration=iter_num,
                iter_dir=iter_dir,
                prompt=prompt,
                timeout=self.cfg.review_timeout_s,
            )
            self.store.append_timeline(
                "failure_analyst_done",
                {"iteration": iter_num, "ok": ok,
                 "error": err if not ok else None},
            )
        except Exception as exc:  # noqa: BLE001
            self.store.append_timeline(
                "failure_analyst_error",
                {"iteration": iter_num, "error": f"agent raised: {exc!r}"},
            )
        finally:
            self._set_agent_status(None)

    # ---- Agent lifecycle ----

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
    ) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
        """Launch one sub-agent and wait for completion.

        Returns ``(ok, error, failure_mode, session_id)``.
        """
        logs_dir = self._logs_dir_for(iteration)
        logs_dir.mkdir(parents=True, exist_ok=True)
        prompt_file = logs_dir / f"{name}.prompt.txt"
        prompt_file.write_text(prompt, encoding="utf-8")
        spec = AgentSpec(
            name=name,
            role=role,
            prompt_file=prompt_file,
            workdir=iter_dir,
            log_dir=logs_dir,
            timeout_s=timeout,
            stuck_timeout_s=self.cfg.stuck_timeout_s,
            extra_args=list(self.cfg.extra_claude_args),
            session_id=session_id,
            resume_session_id=resume_session_id,
        )
        self.store.append_timeline(
            "agent_launch",
            {"name": name, "role": role, "iteration": iteration,
             "resume_from": resume_session_id,
             "session_id_pinned": session_id},
        )
        self.manager.launch(spec)
        result = self.manager.result(name)
        if result is None:
            return False, "no result recorded", "infra", None

        try:
            rec_dict = self.store.load_iteration(iteration)
            if rec_dict is not None:
                rec = IterationRecord.from_dict(rec_dict)
                rec.phases.setdefault(role, {}).update({
                    "duration_s": result.duration_s,
                    "success": result.success,
                    "error": result.error,
                    "failure_mode": result.failure_mode,
                    "attempts": result.attempts,
                    "final_text_head": (result.final_text or "")[:1000],
                    "log_file": str(spec.log_file(result.attempts)),
                    "session_id": result.session_id,
                })
                self.store.write_iteration(iteration, rec.to_dict())
        except Exception:
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

    # ---- Bookkeeping helpers ----

    def _capture_goal(self, iter_dir: Path, ctx: IterationContext) -> None:
        """Capture the plan goal from plan.md if available."""
        plan_path = iter_dir / "plan.md"
        if not plan_path.is_file():
            return
        try:
            text = plan_path.read_text(encoding="utf-8", errors="replace")
            # First pass: explicit goal lines
            for line in text.split("\n"):
                stripped = line.strip()
                if stripped.lower().startswith("**goal") or stripped.lower().startswith("goal"):
                    ctx.goal = stripped.split(":", 1)[-1].strip().strip("*").strip()
                    return
            # Fallback: first non-empty, non-heading line
            for line in text.split("\n"):
                s = line.strip()
                if s and not s.startswith("#"):
                    ctx.goal = s[:200]
                    return
        except OSError:
            pass

    def _set_phase(self, iter_num: int, iter_dir: Path, phase: P.Phase) -> None:
        """Record a phase change in the iteration record."""
        self.store.update_run(current_phase=phase)
        self._set_agent_status(None)
        self.store.append_timeline(
            "phase_start",
            {"iteration": iter_num, "phase": phase, "label": P.phase_label(phase)},
        )

    def _halt_if_verify_exhausted(
        self,
        ctx: IterationContext,
        iter_rec: IterationRecord,
        *,
        timeline_event: str,
        outcome: P.Outcome,
    ) -> bool:
        """Increment verify_failures and halt evolution if limit exceeded.

        Returns True when evolution was halted (caller must ``break`` out
        of the main loop), False otherwise.
        """
        ctx.verify_failures += 1
        max_va = self._resolve_max_verify_attempts()
        self.store.append_timeline(
            timeline_event,
            {"iteration": iter_rec.iteration,
             "verify_failures": ctx.verify_failures},
        )
        if ctx.verify_failures >= max_va:
            self.store.append_timeline(
                "evolution_halted",
                {"reason": "max_verify_attempts exceeded",
                 "verify_failures": ctx.verify_failures,
                 "limit": max_va},
            )
            self._close_iteration(
                iter_rec, status="failed",
                failure=f"D_verify_final failed {ctx.verify_failures} times "
                        f"(limit {max_va}); halting evolution.",
                perf=ctx.this_iter_perf,
                outcome=outcome,
            )
            return True
        return False

    def _close_iteration(
        self,
        rec: IterationRecord,
        *,
        status: str,
        failure: Optional[str] = None,
        perf: Optional[Dict[str, float]] = None,
        outcome: Optional[P.Outcome] = None,
    ) -> None:
        """Finalize an iteration record."""
        rec.closed_at = time.time()
        rec.ended_at = rec.closed_at
        rec.duration_s = rec.closed_at - rec.started_at
        rec.status = status
        rec.failure_reason = failure
        rec.perf = perf or {}
        if outcome is not None:
            rec.outcome = outcome
        # Spawn retro writer (failure → postmortem; success → summary)
        if status == "failed" and not rec.retrospective_path:
            try:
                iter_dir = self.workspace.iter_dir(rec.iteration)
                self._write_failure_retrospective(rec.iteration, iter_dir, rec)
            except Exception:
                pass
        elif status == "success" and not rec.retrospective_path:
            try:
                iter_dir = self.workspace.iter_dir(rec.iteration)
                self._write_retrospective(rec.iteration, iter_dir)
            except Exception:
                pass
        self.store.write_iteration(rec.iteration, rec.to_dict())
        self.workspace.mark_complete(rec.iteration)
        self.store.append_timeline(
            "iteration_end",
            {"iteration": rec.iteration, "status": status,
             "outcome": str(outcome)},
        )

    # ---- Retrospective writers ----

    def _write_retrospective(self, n: int, iter_dir: Path) -> None:
        """Spawn a retro_writer agent for a SUCCESSFUL iteration."""
        logs_dir = self._logs_dir_for(n)
        try:
            ok, _err, _mode, _sid = self._run_agent(
                name=f"iter{n}-retro",
                role="retro_writer",
                iteration=n,
                iter_dir=iter_dir,
                prompt=prompts.retrospective_prompt(
                    req=self.req,
                    iter_dir=iter_dir,
                    notebooks_dir=self.cfg.notebooks_dir,
                    iteration=n,
                    review_feedback=None,
                    logs_dir=logs_dir,
                    goal=None,
                ),
                timeout=self.cfg.retro_timeout_s,
            )
        except Exception:
            return
        # Record path
        retro_path = logs_dir / "retrospective.md"
        if retro_path.is_file():
            rec_dict = self.store.load_iteration(n)
            if rec_dict:
                rec = IterationRecord.from_dict(rec_dict)
                rec.retrospective_path = str(retro_path)
                self.store.write_iteration(n, rec.to_dict())

    def _write_failure_retrospective(
        self, n: int, iter_dir: Path, rec: IterationRecord,
    ) -> None:
        """Spawn a postmortem agent for a FAILED iteration."""
        logs_dir = self._logs_dir_for(n)
        failed_phase: Optional[str] = None
        phase_attempts: Optional[int] = None
        if rec.phases:
            for pid in ("A_attempt_pure", "B_enrich", "C_consolidate", "D_verify_final"):
                info = rec.phases.get(pid)
                if not info:
                    continue
                if info.get("outcome") != P.OK:
                    failed_phase = pid
                    phase_attempts = info.get("attempts")
        goal: Optional[str] = rec.goal or rec.failure_reason or None
        try:
            ok, _err, _mode, _sid = self._run_agent(
                name=f"iter{n}-fail-retro",
                role="retro_writer",
                iteration=n,
                iter_dir=iter_dir,
                prompt=prompts.failure_retrospective_prompt(
                    req=self.req,
                    iter_dir=iter_dir,
                    notebooks_dir=self.cfg.notebooks_dir,
                    iteration=n,
                    failure_reason=rec.failure_reason,
                    failed_phase=failed_phase,
                    phase_attempts=phase_attempts,
                    logs_dir=logs_dir,
                    goal=goal,
                ),
                timeout=self.cfg.retro_timeout_s,
            )
        except Exception:
            return
        retro_path = logs_dir / "retrospective.md"
        if retro_path.is_file():
            rec.retrospective_path = str(retro_path)

    # ---- Repair record (C debugger audit log) ----

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
        note: str = "",
    ) -> None:
        """Write one structured repair record to the audit log (JSONL)."""
        rec_entry = {
            "iteration": iteration,
            "attempt": attempt,
            "timestamp": time.time(),
            "input_failure": (input_failure or "")[:2000],
            "repair": repair_md[:8000] if repair_md else None,
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
            "note": note,
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec_entry, ensure_ascii=False) + "\n")
        except OSError:
            pass

    # ---- Utility ----

    def _resolve_max_iterations(self) -> int:
        """Resolve max iterations from requirements or config."""
        v = self.req.get("max_iterations")
        if v is None:
            v = self.req.get("form", {}).get("max_iterations")
        if v is None:
            return self.cfg.max_iterations
        try:
            return int(v)
        except (TypeError, ValueError):
            return self.cfg.max_iterations

    def _resolve_max_verify_attempts(self) -> int:
        """Resolve max verify attempts from requirements or config."""
        v = self.req.get("max_verify_attempts")
        if v is None:
            v = self.req.get("form", {}).get("max_verify_attempts")
        if v is None:
            return self.cfg.max_verify_attempts
        try:
            return int(v)
        except (TypeError, ValueError):
            return self.cfg.max_verify_attempts

    def _logs_dir_for(self, n: int) -> Path:
        """Return the logs directory for a given iteration."""
        return self.workspace.logs_dir_for(n)
