"""LLM-guided iterative GPU kernel optimization orchestrator.

8-phase flow:
  Phases 1-4 (bootstrap — runs once per task):
    A: Generate Correctness Harness    (agent writes correctness_harness.py)
    B: Adversarial Review of Correctness Harness  (agent reviews → PASS or FAIL with feedback)
    C: Generate Performance Harness    (agent writes perf_harness.py)
    D: Adversarial Review of Performance Harness  (agent reviews → PASS or FAIL)

  Phases 5-8 (optimization loop — repeats until max_iterations):
    E: Select Kernel from Library     (weighted random by exec_time + complexity)
    F: Optimize Selected Kernel       (agent modifies kernel for better perf)
    G: Verify Correctness             (run correctness_harness.py against optimized kernel)
    H: Measure Perf + Complexity      (run perf_harness.py, agent evaluates complexity)
       → Update library, loop back to E

The optimization loop (E→F→G→H→E) consumes one "iteration" per cycle.
Max optimization iterations is configurable (default 20).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import phases as P
from .kernel_library import KernelEntry, KernelLibrary, MAX_LIBRARY_SIZE
from .harness import (
    build_correctness_harness_template,
    build_perf_harness_template,
    run_correctness_test,
    run_perf_test,
)
from .prompts import (
    gen_correctness_harness_prompt,
    review_correctness_harness_prompt,
    gen_perf_harness_prompt,
    review_perf_harness_prompt,
    optimize_kernel_prompt,
    evaluate_complexity_prompt,
    retrospective_prompt,
    failure_retrospective_prompt,
    kernel_fn_name_from_code,
)
from .iteration_record import IterationRecord
from metainfer.orchestrator.iteration import IterationWorkspace
from metainfer.orchestrator.state import StateStore
from metainfer.orchestrator.subagent_manager import AgentSpec, SubAgentManager


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DEFAULT_MAX_ITERATIONS = 20
CONVERGENCE_WINDOW = 6  # stop if no improvement in last N iterations (must be > max_iterations/3 to avoid premature stop)


# --------------------------------------------------------------------------- #
# IterationContext
# --------------------------------------------------------------------------- #


@dataclass
class IterationContext:
    """Mutable state carried across the optimization loop."""

    # Harness state
    correctness_harness_path: Optional[Path] = None
    perf_harness_path: Optional[Path] = None
    correctness_harness_code: Optional[str] = None
    perf_harness_code: Optional[str] = None

    # Review feedback (for rejected harnesses)
    correctness_review_feedback: Optional[str] = None
    perf_review_feedback: Optional[str] = None

    # Kernel library
    library: KernelLibrary = field(default_factory=KernelLibrary)

    # Current optimization state
    selected_kernel: Optional[KernelEntry] = None
    current_exec_time_ms: float = 0.0
    current_complexity: float = 0.5
    best_exec_time_ms: float = float("inf")

    # Optimization history
    optimization_history: List[str] = field(default_factory=list)

    # Failure tracking
    failure: Optional[str] = None
    last_outcome: Optional[P.Outcome] = None
    phase_attempts: Dict[str, int] = field(default_factory=dict)

    # Convergence tracking
    no_improvement_count: int = 0

    # Session tracking for resume
    session_id: Optional[str] = None

    # Original kernel
    kernel_code: str = ""
    kernel_fn_name: str = ""
    ref_kernel_path: str = ""


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass
class OrchestratorConfig:
    workdir: Path
    state_dir: Path
    workspace_dir: Path
    repo_root: Path
    iterations_root: Path
    logs_root: Optional[Path] = None
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    claude_bin: str = "ccb"
    model: Optional[str] = None
    permission_mode: str = "bypassPermissions"
    extra_claude_args: List[str] = field(default_factory=list)
    agent_timeout_s: int = 1800  # 30 min default
    harness_timeout_s: int = 300  # 5 min for correctness
    perf_timeout_s: int = 600    # 10 min for performance
    stuck_timeout_s: int = 600


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #


class Orchestrator:
    """Drive the 8-phase kernel optimization flow."""

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
            extra_add_dirs=[cfg.repo_root, *([cfg.logs_root] if cfg.logs_root else [])],
        )
        self.workspace = IterationWorkspace(cfg.iterations_root, logs_root=cfg.logs_root)
        self._stop = False
        self._agent_call_index = 0

    def _logs_dir_for(self, n: int) -> Path:
        return self.workspace.logs_dir_for(n)

    def _harnesses_dir(self) -> Path:
        d = self.cfg.workspace_dir / "harnesses"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def _kernel_library_path(self) -> Path:
        return self.cfg.workspace_dir / "kernel_library.json"

    # ------------------------------------------------------------------ #
    # Public entry
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        task_id = self.req.get("task_id", "task")
        _, is_resume = self.store.init_or_resume(task_id=task_id)

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
        else:
            self.store.append_timeline("orchestrator_start", {"task_id": task_id})

        try:
            self._loop(is_resume=is_resume)
        except KeyboardInterrupt:
            self.store.append_timeline("orchestrator_abort", {"reason": "keyboard-interrupt"})
            self.store.update_run(finished=True, final_status="aborted", current_phase="finished")
        finally:
            self.manager.shutdown()
            self.store.append_timeline("orchestrator_end", {"task_id": task_id})

    def stop(self) -> None:
        self._stop = True

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #

    def _loop(self, is_resume: bool = False) -> None:
        ctx = IterationContext()
        self._load_kernel_from_req(ctx)
        ctx.library = KernelLibrary.load(self._kernel_library_path)

        # Add the original kernel to the library as seed
        if ctx.library.size == 0:
            seed = KernelEntry(
                id=str(uuid.uuid4()),
                code=ctx.kernel_code,
                exec_time_ms=0.0,  # Will be measured in first H phase
                complexity_score=0.5,  # Will be evaluated
                combined_score=0.0,
                iteration_added=0,
                parent_id=None,
            )
            ctx.library.add(seed)
            ctx.library.save(self._kernel_library_path)

        # Determine starting phase
        if is_resume:
            rs = self.store.load_run()
            start_phase: P.Phase = rs.current_phase if rs.current_phase != "finished" else "A_gen_correctness_harness"
            # Try to load existing harnesses
            hdir = self._harnesses_dir()
            ch = hdir / "correctness_harness.py"
            ph = hdir / "perf_harness.py"
            if ch.exists():
                ctx.correctness_harness_path = ch
                ctx.correctness_harness_code = ch.read_text()
            if ph.exists():
                ctx.perf_harness_path = ph
                ctx.perf_harness_code = ph.read_text()
        else:
            start_phase = "A_gen_correctness_harness"

        phase = start_phase
        iter_num = 0
        iter_dir: Optional[Path] = None
        iter_rec: Optional[IterationRecord] = None
        final_status: Optional[str] = None
        optimization_iter: int = 0  # count of E→F→G→H→E cycles

        self.store.update_run(current_phase=phase)
        self.store.append_timeline("loop_start", {"start_phase": phase, "is_resume": is_resume})

        while not self._stop and not P.is_terminal(phase):
            # Open a new iteration directory if needed
            if iter_dir is None:
                iter_num += 1
                self._agent_call_index = 0  # reset for each iteration
                iter_dir = self.workspace.open_iteration(iter_num)
                iter_rec = IterationRecord(
                    iteration=iter_num, started_at=time.time(), start_phase=phase,
                )
                self.store.write_iteration(iter_num, iter_rec.to_dict())
                self.store.update_run(current_iteration=iter_num, current_phase=phase)
                self.store.append_timeline("iteration_start",
                                           {"iteration": iter_num, "start_phase": phase})
                ctx.phase_attempts.clear()

            assert iter_rec is not None

            # Check max iterations (only for optimization phases)
            if P.is_optimization(phase):
                if optimization_iter >= self._resolve_max_iterations():
                    final_status = "success"
                    phase = "finished"
                    self.store.append_timeline("max_iterations_reached",
                                               {"iterations": optimization_iter})
                    break
                # Check convergence
                if ctx.no_improvement_count >= CONVERGENCE_WINDOW:
                    final_status = "success"
                    phase = "finished"
                    self.store.append_timeline("convergence_detected",
                                               {"no_improvement_count": ctx.no_improvement_count})
                    break

            self._set_phase(iter_num, iter_dir, phase)
            outcome, perf, failure = self._run_phase(phase, iter_num, iter_dir, iter_rec, ctx)

            ctx.last_outcome = outcome
            ctx.phase_attempts[phase] = ctx.phase_attempts.get(phase, 0) + 1

            # Record phase result
            phase_rec = iter_rec.phases.setdefault(phase, {})
            phase_rec["outcome"] = outcome
            phase_rec["attempts"] = ctx.phase_attempts[phase]
            phase_rec["ended_at"] = time.time()
            if failure:
                phase_rec["failure"] = failure
            if perf:
                phase_rec["perf"] = perf

            # Get next transition
            t = P.next_transition(phase, outcome)
            if t is None:
                err = f"no transition for ({phase}, {outcome})"
                self.store.append_timeline("transition_error", {"error": err})
                self._close_iteration(iter_rec, status="failed", failure=err,
                                      perf=perf, outcome=outcome)
                phase = "finished"
                final_status = "error"
                continue

            # Carry state
            if t.carry_failure:
                ctx.failure = failure
            elif outcome == P.OK:
                ctx.failure = None

            if perf and t.carry_perf:
                if iter_rec.perf:
                    iter_rec.perf.update(perf)
                else:
                    iter_rec.perf = perf

            self.store.append_timeline("transition", {
                "from": phase, "outcome": outcome, "to": t.to_phase,
                "label": t.label, "iteration": iter_num,
            })
            self.store.update_run(current_phase=t.to_phase, last_outcome=outcome,
                                  last_transition_label=t.label, current_iteration=iter_num)

            # Count optimization iterations
            if t.consume_iteration and P.is_optimization(phase) and t.to_phase == P.E:
                optimization_iter += 1

            if t.consume_iteration:
                iter_status = "success" if outcome == P.OK else "failed"
                self._close_iteration(iter_rec, status=iter_status,
                                      failure=(failure if outcome != P.OK else None),
                                      perf=perf, outcome=outcome)
                iter_dir = None
                iter_rec = None

            phase = t.to_phase

        if final_status is None:
            final_status = "success" if ctx.last_outcome == P.OK else "stopped"
        self.store.update_run(finished=True, final_status=final_status,
                              current_phase="finished", last_outcome=ctx.last_outcome)
        # Save final library state
        ctx.library.save(self._kernel_library_path)

    # ------------------------------------------------------------------ #
    # Phase dispatcher
    # ------------------------------------------------------------------ #

    def _run_phase(
        self, phase: P.Phase, iter_num: int, iter_dir: Path,
        rec: IterationRecord, ctx: IterationContext,
    ) -> Tuple[P.Outcome, Optional[Dict[str, float]], Optional[str]]:
        dispatch = {
            "A_gen_correctness_harness": self._do_gen_correctness_harness,
            "B_review_correctness_harness": self._do_review_correctness_harness,
            "C_gen_perf_harness": self._do_gen_perf_harness,
            "D_review_perf_harness": self._do_review_perf_harness,
            "E_select_kernel": self._do_select_kernel,
            "F_optimize": self._do_optimize,
            "G_verify_correctness": self._do_verify_correctness,
            "H_measure_perf": self._do_measure_perf,
        }
        handler = dispatch.get(phase)
        if handler is None:
            raise ValueError(f"no handler for phase {phase!r}")
        return handler(iter_num, iter_dir, ctx)

    # ---- A: Generate Correctness Harness --------------------------------- #

    def _do_gen_correctness_harness(
        self, n: int, iter_dir: Path, ctx: IterationContext,
    ) -> Tuple[P.Outcome, Optional[Dict[str, float]], Optional[str]]:
        ok, err, mode, _sid = self._run_agent(
            name=f"iter{n}-gen-correctness-harness",
            role="correctness_harness_generator",
            iteration=n, iter_dir=iter_dir,
            prompt=gen_correctness_harness_prompt(
                req=self.req, iter_dir=iter_dir,
                kernel_code=ctx.kernel_code,
                kernel_fn_name=ctx.kernel_fn_name,
                ref_kernel_path=ctx.ref_kernel_path,
                iteration=n,
                review_feedback=ctx.correctness_review_feedback,
                logs_dir=self._logs_dir_for(n),
            ),
            timeout=self.cfg.agent_timeout_s,
        )

        if not ok:
            return _failure_outcome(mode), None, f"A harness gen failed: {err}"

        # Read the generated harness
        harness_path = iter_dir / "correctness_harness.py"
        if not harness_path.exists():
            return P.LOGIC_FAIL, None, "Agent did not produce correctness_harness.py"

        harness_code = harness_path.read_text(encoding="utf-8")
        ctx.correctness_harness_path = harness_path
        ctx.correctness_harness_code = harness_code

        # Save to shared harnesses dir
        shared = self._harnesses_dir() / "correctness_harness.py"
        shared.write_text(harness_code, encoding="utf-8")

        return P.OK, None, None

    # ---- B: Review Correctness Harness ----------------------------------- #

    def _do_review_correctness_harness(
        self, n: int, iter_dir: Path, ctx: IterationContext,
    ) -> Tuple[P.Outcome, Optional[Dict[str, float]], Optional[str]]:
        if ctx.correctness_harness_code is None:
            return P.LOGIC_FAIL, None, "No correctness harness to review"

        ok, err, mode, _sid = self._run_agent(
            name=f"iter{n}-review-correctness-harness",
            role="correctness_harness_reviewer",
            iteration=n, iter_dir=iter_dir,
            prompt=review_correctness_harness_prompt(
                req=self.req, iter_dir=iter_dir,
                kernel_code=ctx.kernel_code,
                harness_code=ctx.correctness_harness_code,
                iteration=n,
                logs_dir=self._logs_dir_for(n),
            ),
            timeout=self.cfg.agent_timeout_s,
        )

        if not ok:
            return _failure_outcome(mode), None, f"B review failed: {err}"

        # Parse the review verdict
        review_path = iter_dir / "correctness_review.md"
        review_text = ""
        if review_path.exists():
            review_text = review_path.read_text(encoding="utf-8", errors="replace")

        verdict = self._parse_verdict(review_text)
        if verdict == "PASS":
            ctx.correctness_review_feedback = None
            return P.OK, None, None
        else:
            ctx.correctness_review_feedback = review_text[:4000]
            return P.LOGIC_FAIL, None, "Correctness harness rejected by reviewer"

    # ---- C: Generate Performance Harness --------------------------------- #

    def _do_gen_perf_harness(
        self, n: int, iter_dir: Path, ctx: IterationContext,
    ) -> Tuple[P.Outcome, Optional[Dict[str, float]], Optional[str]]:
        ok, err, mode, _sid = self._run_agent(
            name=f"iter{n}-gen-perf-harness",
            role="perf_harness_generator",
            iteration=n, iter_dir=iter_dir,
            prompt=gen_perf_harness_prompt(
                req=self.req, iter_dir=iter_dir,
                kernel_code=ctx.kernel_code,
                kernel_fn_name=ctx.kernel_fn_name,
                ref_kernel_path=ctx.ref_kernel_path,
                iteration=n,
                review_feedback=ctx.perf_review_feedback,
                logs_dir=self._logs_dir_for(n),
            ),
            timeout=self.cfg.agent_timeout_s,
        )

        if not ok:
            return _failure_outcome(mode), None, f"C perf harness gen failed: {err}"

        harness_path = iter_dir / "perf_harness.py"
        if not harness_path.exists():
            return P.LOGIC_FAIL, None, "Agent did not produce perf_harness.py"

        harness_code = harness_path.read_text(encoding="utf-8")
        ctx.perf_harness_path = harness_path
        ctx.perf_harness_code = harness_code

        # Save to shared harnesses dir
        shared = self._harnesses_dir() / "perf_harness.py"
        shared.write_text(harness_code, encoding="utf-8")

        return P.OK, None, None

    # ---- D: Review Performance Harness ----------------------------------- #

    def _do_review_perf_harness(
        self, n: int, iter_dir: Path, ctx: IterationContext,
    ) -> Tuple[P.Outcome, Optional[Dict[str, float]], Optional[str]]:
        if ctx.perf_harness_code is None:
            return P.LOGIC_FAIL, None, "No performance harness to review"

        ok, err, mode, _sid = self._run_agent(
            name=f"iter{n}-review-perf-harness",
            role="perf_harness_reviewer",
            iteration=n, iter_dir=iter_dir,
            prompt=review_perf_harness_prompt(
                req=self.req, iter_dir=iter_dir,
                kernel_code=ctx.kernel_code,
                harness_code=ctx.perf_harness_code,
                iteration=n,
                logs_dir=self._logs_dir_for(n),
            ),
            timeout=self.cfg.agent_timeout_s,
        )

        if not ok:
            return _failure_outcome(mode), None, f"D perf review failed: {err}"

        review_path = iter_dir / "perf_review.md"
        review_text = ""
        if review_path.exists():
            review_text = review_path.read_text(encoding="utf-8", errors="replace")

        verdict = self._parse_verdict(review_text)
        if verdict == "PASS":
            ctx.perf_review_feedback = None
            return P.OK, None, None
        else:
            ctx.perf_review_feedback = review_text[:4000]
            return P.LOGIC_FAIL, None, "Performance harness rejected by reviewer"

    # ---- E: Select Kernel from Library ----------------------------------- #

    def _do_select_kernel(
        self, n: int, iter_dir: Path, ctx: IterationContext,
    ) -> Tuple[P.Outcome, Optional[Dict[str, float]], Optional[str]]:
        selected = ctx.library.select()
        if selected is None:
            # Library empty — use original kernel
            selected = KernelEntry(
                id=str(uuid.uuid4()),
                code=ctx.kernel_code,
                exec_time_ms=0.0,
                complexity_score=0.5,
                combined_score=0.0,
                iteration_added=0,
            )
            ctx.library.add(selected)
            ctx.library.save(self._kernel_library_path)

        ctx.selected_kernel = selected

        # Write the selected kernel to iter_dir
        kernel_path = iter_dir / "selected_kernel.py"
        kernel_path.write_text(selected.code, encoding="utf-8")

        self.store.append_timeline("kernel_selected", {
            "kernel_id": selected.id,
            "exec_time_ms": selected.exec_time_ms,
            "complexity": selected.complexity_score,
            "combined_score": selected.combined_score,
            "iteration": n,
        })

        return P.OK, None, None

    # ---- F: Optimize Kernel ---------------------------------------------- #

    def _do_optimize(
        self, n: int, iter_dir: Path, ctx: IterationContext,
    ) -> Tuple[P.Outcome, Optional[Dict[str, float]], Optional[str]]:
        if ctx.selected_kernel is None:
            return P.LOGIC_FAIL, None, "No kernel selected for optimization"

        history_text = "\n".join(ctx.optimization_history[-10:]) if ctx.optimization_history else "(first optimization attempt)"

        ok, err, mode, sid = self._run_agent(
            name=f"iter{n}-optimizer",
            role="kernel_optimizer",
            iteration=n, iter_dir=iter_dir,
            prompt=optimize_kernel_prompt(
                req=self.req, iter_dir=iter_dir,
                original_kernel_code=ctx.kernel_code,
                current_kernel_code=ctx.selected_kernel.code,
                kernel_fn_name=ctx.kernel_fn_name,
                current_exec_time_ms=ctx.selected_kernel.exec_time_ms,
                current_complexity=ctx.selected_kernel.complexity_score,
                best_exec_time_ms=ctx.best_exec_time_ms,
                best_kernel_code=ctx.library.best.code if ctx.library.best else ctx.kernel_code,
                optimization_history=history_text,
                failure_feedback=ctx.failure if ctx.failure and "correctness" in (ctx.failure or "").lower() else None,
                iteration=n,
                logs_dir=self._logs_dir_for(n),
            ),
            timeout=self.cfg.agent_timeout_s,
            resume_session_id=ctx.session_id,
        )

        if sid:
            ctx.session_id = sid

        if not ok:
            return _failure_outcome(mode), None, f"F optimize failed: {err}"

        # Check for the optimized kernel file
        opt_path = iter_dir / "optimized_kernel.py"
        if not opt_path.exists():
            return P.LOGIC_FAIL, None, "Agent did not produce optimized_kernel.py"

        ctx.session_id = None
        return P.OK, None, None

    # ---- G: Verify Correctness ------------------------------------------- #

    def _do_verify_correctness(
        self, n: int, iter_dir: Path, ctx: IterationContext,
    ) -> Tuple[P.Outcome, Optional[Dict[str, float]], Optional[str]]:
        harness_path = self._harnesses_dir() / "correctness_harness.py"
        if not harness_path.exists():
            # Fall back to iter_dir copy
            harness_path = iter_dir / "correctness_harness.py"
            if not harness_path.exists() and ctx.correctness_harness_path:
                harness_path = ctx.correctness_harness_path
            if not harness_path.exists():
                return P.LOGIC_FAIL, None, "No correctness harness available"

        evolved_path = iter_dir / "optimized_kernel.py"
        if not evolved_path.exists():
            return P.LOGIC_FAIL, None, "No optimized kernel to verify"

        self.store.append_timeline("correctness_test_start", {
            "iteration": n,
            "harness": str(harness_path),
            "kernel": str(evolved_path),
        })

        passed, result = run_correctness_test(
            harness_path, evolved_path,
            timeout_s=self.cfg.harness_timeout_s,
        )

        self.store.append_timeline("correctness_test_end", {
            "iteration": n,
            "passed": passed,
            "summary": str(result.get("error", ""))[:500] if not passed else "all passed",
        })

        if passed:
            ctx.failure = None
            return P.OK, None, None

        # Return failure to feed back to F
        error_msg = result.get("error", "unknown error")
        detail = result.get("results", [])
        failed_cases = [r for r in detail if not r.get("passed", False)]
        failure_report = f"Correctness FAILED: {error_msg}\nFailed cases: {json.dumps(failed_cases[:5])}"
        ctx.failure = failure_report

        # Save failure detail
        detail_path = self._logs_dir_for(n) / "correctness_failure.json"
        detail_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

        return P.LOGIC_FAIL, None, failure_report[:2000]

    # ---- H: Measure Performance + Complexity → Update Library ------------ #

    def _do_measure_perf(
        self, n: int, iter_dir: Path, ctx: IterationContext,
    ) -> Tuple[P.Outcome, Optional[Dict[str, float]], Optional[str]]:
        # 1. Performance measurement
        harness_path = self._harnesses_dir() / "perf_harness.py"
        if not harness_path.exists():
            harness_path = iter_dir / "perf_harness.py"
            if not harness_path.exists() and ctx.perf_harness_path:
                harness_path = ctx.perf_harness_path
            if not harness_path.exists():
                # No perf harness — measure original kernel first
                return self._measure_original_perf(n, iter_dir, ctx)

        evolved_path = iter_dir / "optimized_kernel.py"
        if not evolved_path.exists():
            return P.LOGIC_FAIL, None, "No optimized kernel to measure"

        # Ensure original kernel exec_time is measured
        if ctx.library.best and ctx.library.best.exec_time_ms == 0.0:
            # Measure the original kernel first
            self._measure_original_perf(n, iter_dir, ctx)

        self.store.append_timeline("perf_test_start", {
            "iteration": n,
            "harness": str(harness_path),
            "kernel": str(evolved_path),
        })

        perf_ok, perf_result = run_perf_test(
            harness_path, evolved_path,
            timeout_s=self.cfg.perf_timeout_s,
        )

        exec_time_ms = perf_result.get("evo_median_ms", 0.0)
        speedup = perf_result.get("overall_speedup", 1.0)

        self.store.append_timeline("perf_test_end", {
            "iteration": n,
            "success": perf_ok,
            "exec_time_ms": exec_time_ms,
            "speedup": speedup,
        })

        if not perf_ok:
            ctx.no_improvement_count += 1
            return P.LOGIC_FAIL, None, f"Perf measurement failed: {perf_result.get('error', 'unknown')}"

        # 2. Complexity evaluation
        opt_code = evolved_path.read_text(encoding="utf-8")
        complexity = self._evaluate_complexity(opt_code, n, iter_dir)

        # 3. Create kernel entry and try to add to library
        entry = KernelEntry(
            id=str(uuid.uuid4()),
            code=opt_code,
            exec_time_ms=exec_time_ms,
            complexity_score=complexity,
            combined_score=0.0,
            iteration_added=n,
            parent_id=ctx.selected_kernel.id if ctx.selected_kernel else None,
        )
        entry.recompute_combined()

        added = ctx.library.add(entry)
        ctx.library.save(self._kernel_library_path)

        ctx.current_exec_time_ms = exec_time_ms
        ctx.current_complexity = complexity

        # Track optimization history
        ctx.optimization_history.append(
            f"Iter {n}: exec={exec_time_ms:.4f}ms, complexity={complexity:.2f}, "
            f"combined={entry.combined_score:.4f}, added={'YES' if added else 'NO'}"
        )

        # Update best
        if exec_time_ms < ctx.best_exec_time_ms:
            ctx.best_exec_time_ms = exec_time_ms
            ctx.no_improvement_count = 0
        else:
            ctx.no_improvement_count += 1

        # 4. Write retrospective
        self._write_retrospective(n, iter_dir, ctx, exec_time_ms, complexity, speedup, added)

        # 5. Update iteration record
        kernel_path = iter_dir / "optimized_kernel.py"
        if kernel_path.exists():
            # Copy to shared location
            shared_kernel = self.cfg.workspace_dir / "optimized_kernels" / f"{entry.id}.py"
            shared_kernel.parent.mkdir(parents=True, exist_ok=True)
            shared_kernel.write_text(opt_code, encoding="utf-8")

        perf_dict = {
            "exec_time_ms": exec_time_ms,
            "speedup": speedup,
            "complexity": complexity,
            "combined_score": entry.combined_score,
            "added_to_library": added,
            "library_size": ctx.library.size,
        }

        return P.OK, perf_dict, None

    def _measure_original_perf(
        self, n: int, iter_dir: Path, ctx: IterationContext,
    ) -> Tuple[P.Outcome, Optional[Dict[str, float]], Optional[str]]:
        """Measure the original kernel's execution time using the perf harness."""
        harness_path = self._harnesses_dir() / "perf_harness.py"
        if not harness_path.exists() and ctx.perf_harness_path:
            harness_path = ctx.perf_harness_path
        if not harness_path.exists():
            return P.LOGIC_FAIL, None, "No perf harness to measure original kernel"

        perf_ok, perf_result = run_perf_test(
            harness_path, ctx.ref_kernel_path,
            timeout_s=self.cfg.perf_timeout_s,
        )

        if perf_ok:
            exec_time = perf_result.get("evo_median_ms", 0.0)
            if ctx.library.best:
                ctx.library.best.exec_time_ms = exec_time
                ctx.library.best.recompute_combined()
                ctx.library.save(self._kernel_library_path)
                ctx.best_exec_time_ms = exec_time
                ctx.current_exec_time_ms = exec_time

                # Complexity
                complexity = self._evaluate_complexity(ctx.kernel_code, n, iter_dir)
                ctx.library.best.complexity_score = complexity
                ctx.library.best.recompute_combined()
                ctx.library.save(self._kernel_library_path)
                ctx.current_complexity = complexity

                ctx.optimization_history.append(
                    f"Iter {n} (seed): exec={exec_time:.4f}ms, complexity={complexity:.2f}, "
                    f"combined={ctx.library.best.combined_score:.4f}"
                )

            return P.OK, {"exec_time_ms": exec_time}, None

        return P.LOGIC_FAIL, None, f"Failed to measure original kernel: {perf_result.get('error', 'unknown')}"

    def _evaluate_complexity(self, kernel_code: str, n: int, iter_dir: Path) -> float:
        """Agent evaluates kernel complexity. Returns 0.0-1.0.

        The agent is asked to output a JSON object like
        ``{"overall_complexity": 0.45, "code_length": 0.5, ...}``.
        We parse the agent's final text to extract ``overall_complexity``.
        """
        agent_name = f"iter{n}-complexity"
        ok, err, mode, _sid = self._run_agent(
            name=agent_name,
            role="complexity_evaluator",
            iteration=n, iter_dir=iter_dir,
            prompt=evaluate_complexity_prompt(
                kernel_code=kernel_code,
                iteration=n,
                logs_dir=self._logs_dir_for(n),
            ),
            timeout=300,
        )

        if not ok:
            return 0.5  # default fallback on infra/logic failure

        # Extract the agent's final text output from the SubAgentManager result
        result = self.manager.result(agent_name)
        if result is None:
            return 0.5

        final_text = result.final_text or ""
        score = _parse_complexity_from_text(final_text)
        if score is not None:
            return score

        # Fallback: scan the agent's events file for any JSON with overall_complexity
        log_dir = self._logs_dir_for(n)
        events_file = log_dir / f"{agent_name}.attempt1.events.jsonl"
        score = _parse_complexity_from_events(events_file)
        if score is not None:
            return score

        return 0.5  # default — complexity from agent is advisory

    # ------------------------------------------------------------------ #
    # Agent runner
    # ------------------------------------------------------------------ #

    def _run_agent(self, name: str, role: str, iteration: int, iter_dir: Path,
                   prompt: str, timeout: int, *,
                   resume_session_id: Optional[str] = None,
                   session_id: Optional[str] = None,
                   ) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
        # Ensure unique agent name: append call index within this iteration
        self._agent_call_index += 1
        unique_name = f"{name}#{self._agent_call_index}"

        logs_dir = self._logs_dir_for(iteration)
        logs_dir.mkdir(parents=True, exist_ok=True)
        prompt_file = logs_dir / f"{unique_name}.prompt.txt"
        prompt_file.write_text(prompt, encoding="utf-8")
        spec = AgentSpec(
            name=unique_name, role=role, prompt_file=prompt_file, workdir=iter_dir,
            log_dir=logs_dir, timeout_s=timeout, stuck_timeout_s=self.cfg.stuck_timeout_s,
            extra_args=list(self.cfg.extra_claude_args), session_id=session_id,
            resume_session_id=resume_session_id,
        )
        self.store.append_timeline("agent_launch", {
            "name": unique_name, "role": role, "iteration": iteration,
        })
        self.manager.launch(spec)
        result = self.manager.result(unique_name)
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
                logs_dir = self._logs_dir_for(rec.iteration)
                self._run_agent(
                    name=f"iter{rec.iteration}-fail-retro",
                    role="retro_writer",
                    iteration=rec.iteration, iter_dir=iter_dir,
                    prompt=failure_retrospective_prompt(
                        req=self.req, iter_dir=iter_dir,
                        iteration=rec.iteration,
                        failure_reason=failure,
                        failed_phase=list(rec.phases.keys())[-1] if rec.phases else None,
                        logs_dir=logs_dir,
                    ),
                    timeout=300,
                )
            except Exception:
                pass
            retro_path = logs_dir / "retrospective.md"
            if retro_path.is_file():
                rec.retrospective_path = str(retro_path)
        self.store.write_iteration(rec.iteration, rec.to_dict())
        self.workspace.mark_complete(rec.iteration)
        self.store.append_timeline("iteration_end", {
            "iteration": rec.iteration, "status": status, "outcome": outcome,
            "perf": perf, "failure_reason": failure, "duration_s": rec.duration_s,
        })

    def _write_retrospective(self, n: int, iter_dir: Path, ctx: IterationContext,
                             exec_time_ms: float, complexity: float, speedup: float,
                             added: bool) -> None:
        logs_dir = self._logs_dir_for(n)
        prev = ctx.optimization_history[-2] if len(ctx.optimization_history) >= 2 else ""
        prev_time = 0.0
        if prev:
            m = re.search(r'exec=([\d.]+)ms', prev)
            if m:
                prev_time = float(m.group(1))
        try:
            self._run_agent(
                name=f"iter{n}-retro",
                role="retro_writer",
                iteration=n, iter_dir=iter_dir,
                prompt=retrospective_prompt(
                    req=self.req, iter_dir=iter_dir,
                    iteration=n,
                    exec_time_ms=exec_time_ms,
                    prev_exec_time_ms=prev_time,
                    complexity=complexity,
                    speedup=speedup,
                    library_size=ctx.library.size,
                    added_to_library=added,
                    logs_dir=logs_dir,
                ),
                timeout=300,
            )
        except Exception:
            pass

    def _parse_verdict(self, review_text: str) -> str:
        """Parse PASS/FAIL verdict from a review markdown file."""
        if not review_text:
            return "FAIL"
        text_upper = review_text.upper()
        # Look for explicit verdict
        for line in review_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("**Verdict**") or stripped.startswith("- **Verdict**"):
                if "PASS" in stripped.upper():
                    return "PASS"
                return "FAIL"
        # Heuristic: if text contains "PASS" more prominently than "FAIL"
        pass_count = text_upper.count("PASS")
        # Only count "FAIL" that aren't part of "NOT FAIL"
        fail_count = len(re.findall(r'\bFAIL\b', text_upper))
        if pass_count > fail_count:
            return "PASS"
        return "FAIL"

    def _load_kernel_from_req(self, ctx: IterationContext) -> None:
        """Load the user-provided kernel file from the task requirements.

        Accepts either ``kernel_file_path`` (path to a .py file) or
        ``kernel_code`` (inline source).
        """
        from metainfer.orchestrator.requirements import req_field

        kernel_path_str = req_field(self.req, "kernel_file_path", "")
        if kernel_path_str:
            kernel_path = Path(kernel_path_str)
            if kernel_path.is_file():
                ctx.kernel_code = kernel_path.read_text(encoding="utf-8")
                ctx.kernel_fn_name = req_field(self.req, "kernel_function_name", "") or kernel_fn_name_from_code(ctx.kernel_code)
                ref_dir = self.cfg.workspace_dir / "reference"
                ref_dir.mkdir(parents=True, exist_ok=True)
                ref_path = ref_dir / "original_kernel.py"
                ref_path.write_text(ctx.kernel_code, encoding="utf-8")
                ctx.ref_kernel_path = str(ref_path)
                return
            else:
                raise FileNotFoundError(
                    f"Kernel file not found: {kernel_path_str}. "
                    f"Please check the path and try again."
                )

        # Fallback: inline kernel_code in requirements
        inline_code = req_field(self.req, "kernel_code", "")
        if inline_code:
            ctx.kernel_code = inline_code
            ctx.kernel_fn_name = req_field(self.req, "kernel_function_name", "") or kernel_fn_name_from_code(inline_code)
            ref_dir = self.cfg.workspace_dir / "reference"
            ref_dir.mkdir(parents=True, exist_ok=True)
            ref_path = ref_dir / "original_kernel.py"
            ref_path.write_text(inline_code, encoding="utf-8")
            ctx.ref_kernel_path = str(ref_path)
            return

        # Check if old form v1 fields are present
        old_fields = []
        for key in ("operator", "kernel_language", "target_hardware", "input_shapes"):
            val = req_field(self.req, key, "")
            if val:
                old_fields.append(f"{key}={val}")

        if old_fields:
            raise ValueError(
                "This task uses unsupported form fields. "
                "The evolve-kernel task requires a Triton kernel .py file.\n\n"
                "To fix: delete this task and create a new one. "
                "Provide the path to your Triton kernel .py file.\n\n"
                f"Old form fields found: {', '.join(old_fields)}"
            )

        raise ValueError(
            "No kernel code provided in requirements. "
            "Need either 'kernel_file_path' (path to a Triton .py file) or 'kernel_code' (inline source). "
            "Create a new task using the current form to provide a kernel file."
        )

    def _resolve_max_iterations(self) -> int:
        from metainfer.orchestrator.requirements import req_field_int
        return req_field_int(self.req, "max_iterations", self.cfg.max_iterations)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _failure_outcome(mode: Optional[str]) -> P.Outcome:
    return P.INFRA_FAIL if mode == "infra" else P.LOGIC_FAIL


# --------------------------------------------------------------------------- #
# Complexity parsing helpers
# --------------------------------------------------------------------------- #


def _parse_complexity_from_text(text: str) -> Optional[float]:
    """Extract ``overall_complexity`` from an agent's final text output.

    The agent is prompted to return a JSON object. We scan the text for
    JSON blocks (```json ... ```) and bare JSON objects.
    """
    if not text:
        return None

    # Strategy 1: ```json ... ``` code block
    pattern = r'```(?:json)?\s*\n?([\s\S]*?)```'
    for match in re.finditer(pattern, text):
        score = _try_parse_complexity_json(match.group(1))
        if score is not None:
            return score

    # Strategy 2: find bare JSON objects
    for match in re.finditer(r'\{[^{}]*"overall_complexity"[^{}]*\}', text):
        score = _try_parse_complexity_json(match.group(0))
        if score is not None:
            return score

    # Strategy 3: try to parse the entire text as JSON
    score = _try_parse_complexity_json(text)
    if score is not None:
        return score

    return None


def _try_parse_complexity_json(json_str: str) -> Optional[float]:
    """Try to parse a JSON string and extract ``overall_complexity``."""
    try:
        obj = json.loads(json_str) if isinstance(json_str, str) else json_str
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    score = obj.get("overall_complexity")
    if score is not None and isinstance(score, (int, float)):
        return float(max(0.0, min(1.0, score)))
    return None


def _parse_complexity_from_events(events_file: Path) -> Optional[float]:
    """Scan an agent's events file for JSON responses with overall_complexity."""
    if not events_file.is_file():
        return None
    try:
        for line in events_file.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            score = _parse_complexity_from_text(line)
            if score is not None:
                return score
    except OSError:
        pass
    return None
