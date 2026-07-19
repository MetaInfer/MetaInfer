"""Deterministic ABCDEF orchestrator for GPU kernel optimization.

Control flow is driven by the transition table in .phases. The orchestrator
runs phase-at-a-time: Plan → Implement → Test → Review → Perf Test → Perf Plan.

C_test uses agent-written test.sh (no immutable oracle — the test compares
kernel output against a reference implementation).
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import phases as P
from metainfer.orchestrator.iteration import IterationWorkspace
from .prompts import (
    c_repair_followup_prompt,
    c_repair_prompt,
    failure_retrospective_prompt,
    implement_prompt,
    implement_redo_prompt,
    perf_plan_prompt,
    perf_test_prompt,
    plan_prompt,
    retrospective_prompt,
    review_prompt,
    write_test_harness_prompt,
)
from metainfer.orchestrator.state import StateStore
from metainfer.orchestrator.subagent_manager import AgentSpec, SubAgentManager
from .iteration_record import IterationRecord


MAX_PHASE_ATTEMPTS = 3
PERF_REGRESSION_THRESHOLD = 0.20


# --------------------------------------------------------------------------- #
# IterationContext
# --------------------------------------------------------------------------- #


@dataclass
class IterationContext:
    failure: Optional[str] = None
    last_perf: Optional[Dict[str, float]] = None
    best_perf: Optional[Dict[str, float]] = None
    this_iter_perf: Optional[Dict[str, float]] = None
    last_outcome: Optional[P.Outcome] = None
    review_feedback: Optional[str] = None
    perf_plan: Optional[str] = None
    phase_attempts: Dict[str, int] = field(default_factory=dict)
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
    logs_root: Optional[Path] = None
    max_iterations: int = 20
    plan_timeout_s: int = 1800
    impl_timeout_s: int = 3600
    review_timeout_s: int = 1800
    optimize_timeout_s: int = 3600
    stuck_timeout_s: int = 600
    retro_timeout_s: int = 600
    max_c_retries: int = 3
    claude_bin: str = "ccb"
    model: Optional[str] = None
    permission_mode: str = "bypassPermissions"
    extra_claude_args: List[str] = field(default_factory=list)
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
        self.workspace = IterationWorkspace(cfg.iterations_root, logs_root=cfg.logs_root)
        self._stop = False
        self.nooped = False

    def _logs_dir_for(self, n: int) -> Path:
        return self.workspace.logs_dir_for(n)

    # ------------------------------------------------------------------ #
    # Public entry
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        task_id = self.req.get("task_id", "task")
        _, is_resume = self.store.init_or_resume(task_id=task_id)

        resume_from: Optional[Dict[str, Any]] = None
        if is_resume:
            rs = self.store.load_run()
            if rs.finished:
                self.store.append_timeline(
                    "orchestrator_restart",
                    {"prior_final_status": rs.final_status, "prior_phase": rs.current_phase},
                )
                self.store.update_run(finished=False, final_status=None,
                                      current_phase="idle", last_outcome=None,
                                      last_transition_label=None)
            self.store.append_timeline("orchestrator_resume", {"task_id": task_id})
            resume_from = self._prepare_resume()
        else:
            self.store.append_timeline("orchestrator_start", {"task_id": task_id})

        try:
            self._loop(resume_from=resume_from)
        except KeyboardInterrupt:
            self.store.append_timeline("orchestrator_abort", {"reason": "keyboard-interrupt"})
            self.store.update_run(finished=True, final_status="aborted", current_phase="finished")
        finally:
            self.manager.shutdown()
            self.store.append_timeline("orchestrator_end", {"task_id": task_id})

    def _prepare_resume(self) -> Dict[str, Any]:
        discarded = self.workspace.discard_latest_incomplete()
        if discarded is not None:
            old_rec = self.store.load_iteration(discarded)
            interrupted_in = None
            if old_rec is not None and old_rec.phases:
                interrupted_in = max(old_rec.phases.keys(),
                                     key=lambda k: old_rec.phases[k].get("started_at", 0))
            reason = (f"user interrupted mid-{interrupted_in}" if interrupted_in
                      else "user interrupted (orchestrator process exited unexpectedly)")
            self.store.archive_interrupted_iteration(discarded, reason=reason)
            self.store.append_timeline("iteration_interrupted",
                                       {"iteration": discarded, "reason": reason})
            start_phase = (old_rec.start_phase if old_rec else "A_plan") or "A_plan"
            carried_failure = None
            prev_rec = self.store.load_iteration(discarded - 1) if discarded > 1 else None
            last_outcome = prev_rec.outcome if prev_rec else None
            if prev_rec and start_phase == "B_implement" and prev_rec.outcome != P.OK:
                carried_failure = prev_rec.failure_reason
        else:
            last_complete = self.workspace.latest_complete_number()
            prev_rec = self.store.load_iteration(last_complete) if last_complete else None
            discarded = last_complete + 1
            start_phase = self._phase_after(prev_rec)
            carried_failure = None
            if prev_rec and start_phase == "B_implement" and prev_rec.outcome != P.OK:
                carried_failure = prev_rec.failure_reason
            last_outcome = prev_rec.outcome if prev_rec else None

        return {
            "iter_num": discarded,
            "start_phase": start_phase,
            "carried_failure": carried_failure,
            "last_outcome": last_outcome,
        }

    @staticmethod
    def _phase_after(rec: Optional[IterationRecord]) -> P.Phase:
        if rec is None:
            return "A_plan"
        if rec.outcome == P.OK:
            return "A_plan"
        return "B_implement"

    def stop(self) -> None:
        self._stop = True

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #

    def _loop(self, resume_from: Optional[Dict[str, Any]] = None) -> None:
        max_iters = self._resolve_max_iterations()
        ctx = IterationContext()

        if resume_from is not None:
            phase: P.Phase = resume_from["start_phase"]
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
            if iter_dir is None:
                if iter_num >= max_iters:
                    final_status = "success" if ctx.last_outcome == P.OK else "stopped"
                    phase = "finished"
                    break
                iter_num += 1
                iter_dir = self.workspace.open_iteration(iter_num)
                iter_rec = IterationRecord(
                    iteration=iter_num, started_at=time.time(), start_phase=phase,
                )
                self.store.write_iteration(iter_rec)
                ctx.b_session_id = None
                ctx.this_iter_perf = None
                self.store.update_run(current_iteration=iter_num, current_phase=phase,
                                      last_outcome=ctx.last_outcome)
                self.store.append_timeline("iteration_start",
                                           {"iteration": iter_num, "start_phase": phase})
                ctx.phase_attempts.clear()

            assert iter_rec is not None
            self._set_phase(iter_num, iter_dir, phase)
            outcome, perf, failure = self._run_phase(phase, iter_num, iter_dir, iter_rec, ctx)

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

            # Escalation: too many attempts → close iteration, start fresh at A
            if outcome != P.OK and ctx.phase_attempts[phase] >= MAX_PHASE_ATTEMPTS:
                forced_failure = failure or f"{phase} exceeded {MAX_PHASE_ATTEMPTS} attempts"
                self._close_iteration(iter_rec, status="failed", failure=forced_failure,
                                      perf=ctx.this_iter_perf, outcome=outcome)
                ctx.failure = forced_failure
                ctx.last_outcome = outcome
                iter_dir = None
                iter_rec = None
                phase = "A_plan"
                continue

            t = P.next_transition(phase, outcome)
            if t is None:
                err = f"no transition for ({phase}, {outcome})"
                self._close_iteration(iter_rec, status="failed", failure=err,
                                      perf=ctx.this_iter_perf, outcome=outcome)
                ctx.failure = err
                ctx.last_outcome = outcome
                iter_dir = None
                iter_rec = None
                phase = "A_plan"
                continue

            if t.carry_failure:
                ctx.failure = failure
            elif outcome == P.OK:
                ctx.failure = None

            if perf:
                if t.carry_perf:
                    ctx.last_perf = perf
                ctx.best_perf = _merge_best(ctx.best_perf, perf)
                ctx.this_iter_perf = perf

            self.store.append_timeline("transition", {
                "from": phase, "outcome": outcome, "to": t.to_phase,
                "label": t.label, "iteration": iter_num,
                "consume_iteration": t.consume_iteration,
            })
            self.store.update_run(current_phase=t.to_phase, last_outcome=outcome,
                                  last_transition_label=t.label, current_iteration=iter_num)

            if t.consume_iteration:
                iter_status = "success" if outcome == P.OK else "failed"
                self._close_iteration(iter_rec, status=iter_status,
                                      failure=(failure if outcome != P.OK else None),
                                      perf=ctx.this_iter_perf, outcome=outcome)
                iter_dir = None
                iter_rec = None

            phase = t.to_phase

        if final_status is None:
            final_status = "success" if ctx.last_outcome == P.OK else "stopped"
        self.store.update_run(finished=True, final_status=final_status,
                              current_phase="finished", last_outcome=ctx.last_outcome)

    # ------------------------------------------------------------------ #
    # Phase dispatcher
    # ------------------------------------------------------------------ #

    def _run_phase(
        self, phase: P.Phase, iter_num: int, iter_dir: Path,
        rec: IterationRecord, ctx: IterationContext,
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
        if phase == "F_perf_plan":
            return self._do_perf_plan(iter_num, iter_dir, ctx)
        raise ValueError(f"no handler for phase {phase!r}")

    # ---- A: plan --------------------------------------------------------- #

    def _do_plan(self, n: int, iter_dir: Path, ctx: IterationContext,
                 ) -> Tuple[P.Outcome, Optional[Dict[str, float]], Optional[str]]:
        ok, err, mode, _sid = self._run_agent(
            name=f"iter{n}-planner", role="planner", iteration=n, iter_dir=iter_dir,
            prompt=plan_prompt(
                req=self.req, iter_dir=iter_dir, notebooks_dir=self.cfg.notebooks_dir,
                iteration=n, prev_failures=ctx.failure,
                review_feedback=ctx.review_feedback, perf_plan=ctx.perf_plan,
                logs_dir=self._logs_dir_for(n),
            ),
            timeout=self.cfg.plan_timeout_s,
        )
        return (P.OK, None, None) if ok else (_failure_outcome(mode), None, err)

    # ---- B: implement --------------------------------------------------- #

    def _do_implement(self, n: int, iter_dir: Path, ctx: IterationContext,
                      ) -> Tuple[P.Outcome, Optional[Dict[str, float]], Optional[str]]:
        is_redo = ctx.b_session_id is not None
        if is_redo:
            prompt = implement_redo_prompt(
                req=self.req, iter_dir=iter_dir, notebooks_dir=self.cfg.notebooks_dir,
                iteration=n, prev_failure=ctx.failure, logs_dir=self._logs_dir_for(n),
            )
        else:
            prompt = implement_prompt(
                req=self.req, iter_dir=iter_dir, notebooks_dir=self.cfg.notebooks_dir,
                iteration=n, prev_failure=ctx.failure,
                review_feedback=ctx.review_feedback, perf_plan=ctx.perf_plan,
                logs_dir=self._logs_dir_for(n),
            )
        ok, err, mode, sid = self._run_agent(
            name=f"iter{n}-implementer", role="implementer", iteration=n, iter_dir=iter_dir,
            prompt=prompt, timeout=self.cfg.impl_timeout_s,
            resume_session_id=ctx.b_session_id,
        )
        if sid:
            ctx.b_session_id = sid
        if not ok:
            return _failure_outcome(mode), None, f"B (implement) failed: {err}"
        ctx.b_session_id = None
        return P.OK, None, None

    # ---- C: test -------------------------------------------------------- #
    # Uses agent-written test.sh. No immutable oracle.

    def _do_test(self, n: int, iter_dir: Path, ctx: IterationContext,
                 ) -> Tuple[P.Outcome, Optional[Dict[str, float]], Optional[str]]:
        test_sh = iter_dir / "test.sh"
        if not test_sh.exists():
            ok, err, mode, _sid = self._run_agent(
                name=f"iter{n}-testwriter", role="testwriter", iteration=n, iter_dir=iter_dir,
                prompt=write_test_harness_prompt(
                    req=self.req, iter_dir=iter_dir, notebooks_dir=self.cfg.notebooks_dir,
                    iteration=n,
                ),
                timeout=self.cfg.review_timeout_s,
            )
            if not ok or not test_sh.exists():
                return _failure_outcome(mode), None, f"C (test harness) missing: {err}"

        max_attempts = max(1, int(self.cfg.max_c_retries))
        last_outcome: Optional[P.Outcome] = None
        last_perf: Optional[Dict[str, float]] = None
        last_failure: Optional[str] = None
        c_session_id: Optional[str] = None

        for attempt in range(1, max_attempts + 1):
            self.store.append_timeline("c_test_attempt", {
                "iteration": n, "attempt": attempt, "max": max_attempts,
                "mode": "test.sh",
            })
            success, perf, failure = self._run_test(test_sh, iter_dir, n)
            if success:
                ctx.last_perf = perf
                return P.OK, perf, None
            outcome = P.LOGIC_FAIL
            if failure and _looks_like_infra(failure):
                outcome = P.INFRA_FAIL
                self.store.append_timeline("c_test_infra_fail", {
                    "iteration": n, "attempt": attempt, "error": failure,
                })
                return outcome, perf, failure
            last_outcome, last_perf, last_failure = outcome, perf, failure

            if attempt >= max_attempts:
                break

            # Dispatch debugger
            dbg_name = f"iter{n}-c-debugger.attempt{attempt}"
            if c_session_id is None:
                prompt = c_repair_prompt(
                    req=self.req, iter_dir=iter_dir, notebooks_dir=self.cfg.notebooks_dir,
                    iteration=n, attempt=attempt, max_attempts=max_attempts,
                    failure=failure, logs_dir=self._logs_dir_for(n),
                )
            else:
                prompt = c_repair_followup_prompt(
                    iteration=n, attempt=attempt, max_attempts=max_attempts,
                    new_failure=failure, logs_dir=self._logs_dir_for(n),
                )
            ok, err, mode, sid = self._run_agent(
                name=dbg_name, role="c_debugger", iteration=n, iter_dir=iter_dir,
                prompt=prompt, timeout=self.cfg.impl_timeout_s,
                resume_session_id=c_session_id,
            )
            if sid:
                c_session_id = sid

        return last_outcome or P.LOGIC_FAIL, last_perf, last_failure

    # ---- D: review ------------------------------------------------------ #

    def _do_review(self, n: int, iter_dir: Path, ctx: IterationContext,
                   ) -> Tuple[P.Outcome, Optional[Dict[str, float]], Optional[str]]:
        c_outcome = ctx.last_outcome
        c_failure = ctx.failure
        c_perf = ctx.last_perf
        ok, _err, _mode, _sid = self._run_agent(
            name=f"iter{n}-reviewer", role="reviewer", iteration=n, iter_dir=iter_dir,
            prompt=review_prompt(
                req=self.req, iter_dir=iter_dir, notebooks_dir=self.cfg.notebooks_dir,
                iteration=n, outcome=c_outcome, failure=c_failure, perf=c_perf,
                logs_dir=self._logs_dir_for(n),
            ),
            timeout=self.cfg.review_timeout_s,
        )
        # Capture review.md for next iteration
        review_path = self._logs_dir_for(n) / "review.md"
        feedback: Optional[str] = None
        if review_path.is_file():
            try:
                text = review_path.read_text(encoding="utf-8", errors="replace")
                feedback = text[:8192] if len(text) > 8192 else text
            except OSError:
                feedback = None
        ctx.review_feedback = feedback
        self.store.append_timeline("review_done", {
            "iteration": n, "c_outcome": c_outcome,
            "feedback_captured": feedback is not None,
        })
        if c_outcome == P.OK:
            return P.OK, None, None
        return P.LOGIC_FAIL, None, None

    # ---- E: perf test --------------------------------------------------- #

    def _do_perf_test(self, n: int, iter_dir: Path, ctx: IterationContext, rec: IterationRecord,
                      ) -> Tuple[P.Outcome, Optional[Dict[str, float]], Optional[str]]:
        ok, err, mode, _sid = self._run_agent(
            name=f"iter{n}-perf-tester", role="perf_tester", iteration=n, iter_dir=iter_dir,
            prompt=perf_test_prompt(
                req=self.req, iter_dir=iter_dir, notebooks_dir=self.cfg.notebooks_dir,
                iteration=n, review_feedback=ctx.review_feedback,
                logs_dir=self._logs_dir_for(n),
            ),
            timeout=self.cfg.optimize_timeout_s,
        )
        perf: Optional[Dict[str, float]] = None
        e_error: Optional[str] = None
        if not ok:
            e_error = err
        else:
            perf = self._read_perf_report(n, iter_dir)

        self._write_retrospective(n=n, iter_dir=iter_dir, ctx=ctx, rec=rec,
                                  this_perf=perf, e_ok=ok, e_error=e_error)
        if not ok:
            return _failure_outcome(mode), None, f"E (perf test) failed: {err}"
        return P.OK, perf, None

    def _write_retrospective(self, n: int, iter_dir: Path, ctx: IterationContext,
                             rec: IterationRecord, *, this_perf: Optional[Dict[str, float]],
                             e_ok: bool, e_error: Optional[str]) -> None:
        prev_rec = self.store.load_iteration(n - 1) if n > 1 else None
        prev_perf = dict(prev_rec.perf) if prev_rec and prev_rec.perf else None
        goal: Optional[str] = rec.failure_reason or rec.goal or None
        logs_dir = self._logs_dir_for(n)
        try:
            self._run_agent(
                name=f"iter{n}-retro", role="retro_writer", iteration=n, iter_dir=iter_dir,
                prompt=retrospective_prompt(
                    req=self.req, iter_dir=iter_dir, notebooks_dir=self.cfg.notebooks_dir,
                    iteration=n, this_perf=this_perf, prev_perf=prev_perf,
                    review_feedback=ctx.review_feedback, e_ok=e_ok, e_error=e_error,
                    logs_dir=logs_dir, goal=goal,
                ),
                timeout=self.cfg.retro_timeout_s,
            )
        except Exception:
            pass
        retro_path = logs_dir / "retrospective.md"
        if retro_path.is_file():
            rec.retrospective_path = str(retro_path)
            self.store.write_iteration(rec)

    # ---- F: perf plan --------------------------------------------------- #

    def _do_perf_plan(self, n: int, iter_dir: Path, ctx: IterationContext,
                      ) -> Tuple[P.Outcome, Optional[Dict[str, float]], Optional[str]]:
        ok, err, mode, _sid = self._run_agent(
            name=f"iter{n}-perf-planner", role="perf_planner", iteration=n, iter_dir=iter_dir,
            prompt=perf_plan_prompt(
                req=self.req, iter_dir=iter_dir, notebooks_dir=self.cfg.notebooks_dir,
                iteration=n, last_perf=ctx.last_perf or ctx.best_perf,
                review_feedback=ctx.review_feedback, logs_dir=self._logs_dir_for(n),
            ),
            timeout=self.cfg.review_timeout_s,
        )
        plan_path = iter_dir / "perf_plan.md"
        plan_text: Optional[str] = None
        if plan_path.is_file():
            try:
                text = plan_path.read_text(encoding="utf-8", errors="replace")
                plan_text = text[:12288] if len(text) > 12288 else text
            except OSError:
                plan_text = None
        ctx.perf_plan = plan_text
        if ok:
            return P.OK, ctx.last_perf, None
        return _failure_outcome(mode), None, f"F (perf plan) failed: {err}"

    # ------------------------------------------------------------------ #
    # Agent runner
    # ------------------------------------------------------------------ #

    def _run_agent(self, name: str, role: str, iteration: int, iter_dir: Path,
                   prompt: str, timeout: int, *,
                   resume_session_id: Optional[str] = None,
                   session_id: Optional[str] = None,
                   ) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
        logs_dir = self._logs_dir_for(iteration)
        logs_dir.mkdir(parents=True, exist_ok=True)
        prompt_file = logs_dir / f"{name}.prompt.txt"
        prompt_file.write_text(prompt, encoding="utf-8")
        spec = AgentSpec(
            name=name, role=role, prompt_file=prompt_file, workdir=iter_dir,
            log_dir=logs_dir, timeout_s=timeout, stuck_timeout_s=self.cfg.stuck_timeout_s,
            extra_args=list(self.cfg.extra_claude_args), session_id=session_id,
            resume_session_id=resume_session_id,
        )
        self.store.append_timeline("agent_launch", {
            "name": name, "role": role, "iteration": iteration,
        })
        self.manager.launch(spec)
        result = self.manager.result(name)
        if result is None:
            return False, "no result recorded", "infra", None
        return result.success, result.error, result.failure_mode, result.session_id

    # ------------------------------------------------------------------ #
    # Bookkeeping
    # ------------------------------------------------------------------ #

    def _set_phase(self, n: int, iter_dir: Path, phase: P.Phase) -> None:
        self.store.update_run(current_iteration=n, current_phase=phase)
        self.store.append_timeline("phase_start", {"iteration": n, "phase": phase})

    def _close_iteration(self, rec: IterationRecord, *, status: str,
                         failure: Optional[str], perf: Optional[Dict[str, float]],
                         outcome: Optional[P.Outcome]) -> None:
        rec.ended_at = time.time()
        rec.duration_s = rec.ended_at - rec.started_at
        rec.status = status
        rec.failure_reason = failure
        rec.outcome = outcome
        if perf:
            rec.perf = perf
        if status == "failed" and not rec.retrospective_path:
            try:
                iter_dir = self.workspace.iter_dir(rec.iteration)
                self._write_failure_retrospective(rec.iteration, iter_dir, rec)
            except Exception:
                pass
        self.store.write_iteration(rec)
        self.workspace.mark_complete(rec.iteration)
        self.store.append_timeline("iteration_end", {
            "iteration": rec.iteration, "status": status, "outcome": outcome,
            "perf": perf, "failure_reason": failure, "duration_s": rec.duration_s,
        })

    def _write_failure_retrospective(self, n: int, iter_dir: Path, rec: IterationRecord) -> None:
        failed_phase: Optional[str] = None
        phase_attempts: Optional[int] = None
        if rec.phases:
            for phase_id in ("A_plan", "B_implement", "C_test", "D_review", "E_perf_test", "F_perf_plan"):
                info = rec.phases.get(phase_id)
                if info and info.get("outcome") != P.OK:
                    failed_phase = phase_id
                    phase_attempts = info.get("attempts")
        logs_dir = self._logs_dir_for(n)
        goal: Optional[str] = rec.goal or rec.failure_reason or None
        try:
            self._run_agent(
                name=f"iter{n}-fail-retro", role="retro_writer", iteration=n, iter_dir=iter_dir,
                prompt=failure_retrospective_prompt(
                    req=self.req, iter_dir=iter_dir, notebooks_dir=self.cfg.notebooks_dir,
                    iteration=n, failure_reason=rec.failure_reason,
                    failed_phase=failed_phase, phase_attempts=phase_attempts,
                    logs_dir=logs_dir, goal=goal,
                ),
                timeout=self.cfg.retro_timeout_s,
            )
        except Exception:
            return
        retro_path = logs_dir / "retrospective.md"
        if retro_path.is_file():
            rec.retrospective_path = str(retro_path)

    # ------------------------------------------------------------------ #
    # Test runner
    # ------------------------------------------------------------------ #

    def _run_test(self, test_sh: Path, iter_dir: Path, iteration: int,
                  ) -> Tuple[bool, Optional[Dict[str, float]], Optional[str]]:
        log_path = self._logs_dir_for(iteration) / f"iter{iteration}-test.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env["METAINFER_ITER_DIR"] = str(iter_dir)
        env["METAINFER_ITERATION"] = str(iteration)
        try:
            with open(log_path, "w", encoding="utf-8") as logf:
                proc = subprocess.run(
                    ["bash", str(test_sh)], cwd=str(iter_dir), env=env,
                    stdout=subprocess.PIPE, stderr=logf,
                    timeout=self.cfg.impl_timeout_s, text=True,
                )
        except subprocess.TimeoutExpired:
            return False, None, "test timed out"
        except Exception as exc:
            return False, None, f"test runner exception: {exc!r}"

        stdout = proc.stdout or ""
        parsed = self._parse_test_json(stdout)
        if parsed is None:
            return False, None, f"test did not emit parseable JSON. stdout tail: {stdout[-1000:]!r}"
        passed = bool(parsed.get("passed", False))
        perf = parsed.get("perf") if isinstance(parsed.get("perf"), dict) else {}
        perf = {k: float(v) for k, v in perf.items() if _is_num(v)}
        if passed:
            return True, perf, None
        err = parsed.get("error") or parsed.get("traceback") or "test failed"
        return False, perf, str(err)[:4000]

    def _read_perf_report(self, n: int, iter_dir: Path) -> Optional[Dict[str, float]]:
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

    @staticmethod
    def _parse_test_json(stdout: str) -> Optional[Dict[str, Any]]:
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
        return None

    def _resolve_max_iterations(self) -> int:
        from metainfer.orchestrator.requirements import req_field_int
        return req_field_int(self.req, "max_iterations", self.cfg.max_iterations)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _failure_outcome(mode: Optional[str]) -> P.Outcome:
    return P.INFRA_FAIL if mode == "infra" else P.LOGIC_FAIL


def _looks_like_infra(failure: str) -> bool:
    f = failure.lower()
    return any(s in f for s in ("timed out", "timeout", "exception", "traceback",
                                "no such file", "permission denied", "killed"))


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _merge_best(best: Optional[Dict[str, float]], new: Dict[str, float]) -> Dict[str, float]:
    out = dict(best or {})
    for k, v in new.items():
        if k not in out or v > out[k]:
            out[k] = v
    return out
