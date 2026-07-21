"""Arena-style GEMM optimization pipeline with a frozen external judge."""

from __future__ import annotations

import json
import hashlib
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from metainfer.orchestrator.iteration import IterationWorkspace
from metainfer.orchestrator.state import StateStore
from metainfer.orchestrator.subagent_manager import AgentSpec, SubAgentManager

from . import phases as P
from .build import BuildResult, SystemBuilder
from .evaluator import (
    ChampionStore,
    EvaluationResult,
    EvaluatorRunner,
    FrozenEvaluatorBundle,
    FrozenWeightBundle,
)
from .guidance import GuidanceStore
from .iteration_record import IterationRecord
from .plugin import PLUGIN
from .profiler import ProfilerRunner
from .prompts import (
    implement_prompt,
    perf_plan_prompt,
    plan_prompt,
    review_prompt,
    with_human_guidance,
)


@dataclass
class OrchestratorConfig:
    state_dir: Path
    iterations_root: Path
    logs_root: Path
    notebooks_dir: Path
    evaluator_bundle: FrozenEvaluatorBundle
    system_builder: SystemBuilder
    profiler: Optional[ProfilerRunner] = None
    weight_bundle: Optional[FrozenWeightBundle] = None
    initial_submission: Optional[Path] = None
    max_iterations: int = 20
    agent_timeout_s: int = 3600
    stuck_timeout_s: int = 600
    extra_claude_args: List[str] = field(default_factory=list)


class Orchestrator:
    def __init__(
        self,
        req: Dict[str, Any],
        store: StateStore,
        cfg: OrchestratorConfig,
        manager: SubAgentManager,
    ) -> None:
        self.req = req
        self.agent_req = {
            **req,
            "public_contract": cfg.evaluator_bundle.spec.agent_contract(),
        }
        self.store = store
        self.cfg = cfg
        self.manager = manager
        self.workspace = IterationWorkspace(
            cfg.iterations_root,
            logs_root=cfg.logs_root,
            diagnostic_globs=PLUGIN.diagnostic_globs,
        )
        private_env: Dict[str, str] = {}
        if cfg.weight_bundle is not None:
            private_env = {
                "METAINFER_WEIGHT_BUNDLE": str(cfg.weight_bundle.root.resolve()),
                "METAINFER_WEIGHT_SHA256": cfg.weight_bundle.digest,
            }
        self.evaluator = EvaluatorRunner(
            cfg.evaluator_bundle,
            private_env=private_env,
            private_verifier=cfg.weight_bundle.verify if cfg.weight_bundle is not None else None,
        )
        self.builder = cfg.system_builder
        self.profiler = cfg.profiler
        self.guidance = GuidanceStore(cfg.state_dir / "guidance")
        self.champions = ChampionStore(
            cfg.state_dir / "champion",
            cfg.evaluator_bundle.spec.acceptance.noise_threshold,
        )

    def run(self) -> None:
        task_id = str(self.req.get("task_id") or "task")
        _, is_resume = self.store.init_or_resume(task_id)
        if is_resume:
            run = self.store.load_run()
            if run.finished:
                self.store.update_run(
                    finished=False, final_status=None, current_phase="idle",
                    last_outcome=None, last_transition_label=None,
                )
            discarded = self.workspace.discard_latest_incomplete()
            if discarded is not None:
                self.store.archive_interrupted_iteration(discarded)
            self.store.append_timeline("orchestrator_resume", {"task_id": task_id})
        else:
            self.store.append_timeline("orchestrator_start", {"task_id": task_id})

        try:
            self.store.update_run(current_iteration=0, current_phase="S_baseline")
            self.store.append_timeline("phase_start", {"iteration": 0, "phase": "S_baseline"})
            self.baseline = self._ensure_baseline()
            self.store.append_timeline(
                "phase_end", {"iteration": 0, "phase": "S_baseline", "outcome": P.OK}
            )
        except Exception as exc:  # noqa: BLE001
            failure = f"baseline certification failed: {exc}"
            self.store.append_timeline("baseline_failed", {"failure": failure})
            self.store.update_run(
                finished=True, final_status="stopped", current_phase="finished",
                last_outcome=P.LOGIC_FAIL, last_transition_label=failure,
            )
            self.manager.shutdown()
            return
        self.champions.initialize(self.cfg.state_dir / "baseline" / "submission")

        start = self.workspace.latest_complete_number() + 1
        any_success = False
        try:
            for iteration in range(start, self.cfg.max_iterations + 1):
                outcome = self._run_iteration(iteration)
                any_success = any_success or outcome == P.OK
        except KeyboardInterrupt:
            self.store.update_run(finished=True, final_status="aborted", current_phase="finished")
            self.store.append_timeline("orchestrator_abort", {"reason": "keyboard-interrupt"})
            return
        finally:
            self.manager.shutdown()

        champion = self.champions.load()
        any_success = any_success or int(champion.get("iteration", 0)) > 0
        self.store.update_run(
            finished=True,
            final_status="success" if any_success else "stopped",
            current_phase="finished",
            last_transition_label="iteration limit reached",
        )
        self.store.append_timeline("orchestrator_end", {"task_id": task_id, "champion": champion})

    def _run_iteration(self, n: int) -> P.Outcome:
        iter_dir = self.workspace.open_iteration(n)
        self._seed_from_champion(iter_dir)
        logs_dir = self.workspace.logs_dir_for(n)
        logs_dir.mkdir(parents=True, exist_ok=True)
        rec = IterationRecord(iteration=n, started_at=time.time())
        self._write(rec)
        self.store.update_run(current_iteration=n, current_phase="A_plan")
        self.store.append_timeline("iteration_start", {"iteration": n})

        feedback = self._load_prior_feedback(n)
        champion = self.champions.load()
        ok, failure = self._agent_phase(
            rec,
            "A_plan",
            role="planner",
            workdir=iter_dir,
            prompt=plan_prompt(
                self.agent_req, iter_dir, self.cfg.notebooks_dir, n, champion, feedback
            ),
        )
        if not ok:
            return self._finish_failed(rec, P.INFRA_FAIL, failure or "planner failed")

        submission_dir = iter_dir / "submission"
        ok, failure = self._agent_phase(
            rec,
            "B_implement",
            role="implementer",
            workdir=submission_dir,
            prompt=implement_prompt(
                self.agent_req, submission_dir, self.cfg.notebooks_dir, n
            ),
        )
        if not ok:
            return self._finish_failed(rec, P.INFRA_FAIL, failure or "implementer failed")

        # C is one correctness-test phase. Compilation is its first internal
        # gate; only a compiled artifact is handed to the frozen harness.
        build_result, compile_result, correctness = self._test_phase(
            rec, submission_dir, logs_dir
        )
        test_feedback = self._write_feedback(
            logs_dir,
            compile_result=compile_result,
            correctness_result=correctness,
        )
        self._review(
            rec,
            iter_dir,
            test_feedback,
            test_passed=bool(
                compile_result.passed and correctness is not None and correctness.passed
            ),
        )

        if not compile_result.passed:
            return self._finish_failed(
                rec,
                P.INFRA_FAIL if compile_result.infra_failure else P.LOGIC_FAIL,
                compile_result.failure or "compile failed",
            )
        if correctness is None or not correctness.passed:
            return self._finish_failed(
                rec,
                P.INFRA_FAIL if correctness and correctness.infra_failure else P.LOGIC_FAIL,
                (correctness.failure if correctness else None) or "correctness failed",
            )

        benchmark = self._evaluation_phase(
            rec, "E_perf_test", "benchmark", submission_dir,
            build_result.artifact_dir, logs_dir,
        )
        score = dict(benchmark.report.get("score") or {})
        rec.score = score
        rec.hardware_profile = dict(benchmark.report.get("hardware_profile") or {})
        promoted = False
        reason = benchmark.failure or "benchmark failed"
        champion = self.champions.load()
        if benchmark.passed:
            promoted, reason, champion = self.champions.consider(n, submission_dir, score)
        rec.promoted = promoted
        rec.champion_iteration = int(champion.get("iteration", 0))
        self._write(rec)
        promotion = {"promoted": promoted, "reason": reason, "champion": champion}
        perf_feedback = self._write_feedback(
            logs_dir,
            compile_result=compile_result,
            correctness_result=correctness,
            benchmark_result=benchmark,
            promotion=promotion,
        )
        self._perf_plan(rec, iter_dir, perf_feedback, promotion)

        if benchmark.infra_failure:
            return self._finish_failed(rec, P.INFRA_FAIL, benchmark.failure or "benchmark infrastructure failure")
        outcome = P.OK if promoted else P.PERF_REGRESSION
        return self._finish(rec, "success" if promoted else "not_promoted", outcome, reason if not promoted else None)

    def _seed_from_champion(self, iter_dir: Path) -> None:
        self.champions.load()  # verifies the persisted source tree digest
        submission = iter_dir / "submission"
        if submission.exists():
            shutil.rmtree(submission)
        champion_submission = self.champions.submission_dir
        if champion_submission.is_dir():
            shutil.copytree(champion_submission, submission)
        else:
            submission.mkdir(parents=True)

    def _agent_phase(
        self,
        rec: IterationRecord,
        phase: P.Phase,
        *,
        role: str,
        workdir: Path,
        prompt: str,
        success_outcome: P.Outcome = P.OK,
    ) -> Tuple[bool, Optional[str]]:
        self._start_phase(rec, phase)
        live_guidance = self.guidance.consume(
            iteration=rec.iteration, phase=phase, role=role,
        )
        if live_guidance:
            prompt = with_human_guidance(prompt, live_guidance)
            self.store.append_timeline(
                "human_guidance_applied",
                {
                    "iteration": rec.iteration,
                    "phase": phase,
                    "role": role,
                    "guidance_ids": [item["id"] for item in live_guidance],
                },
            )
        logs_dir = self.workspace.logs_dir_for(rec.iteration)
        name = f"iter{rec.iteration}-{role}"
        prompt_file = logs_dir / f"{name}.prompt.txt"
        prompt_file.write_text(prompt, encoding="utf-8")
        spec = AgentSpec(
            name=name,
            role=role,
            prompt_file=prompt_file,
            workdir=workdir,
            log_dir=logs_dir,
            timeout_s=self.cfg.agent_timeout_s,
            stuck_timeout_s=self.cfg.stuck_timeout_s,
            extra_args=list(self.cfg.extra_claude_args),
        )
        self.store.append_timeline("agent_launch", {"iteration": rec.iteration, "name": name, "role": role})
        self.manager.launch(spec)
        result = self.manager.result(name)
        ok = bool(result and result.success)
        failure = None if ok else (result.error if result else "agent produced no result")
        self._end_phase(rec, phase, success_outcome if ok else P.INFRA_FAIL, failure)
        self.store.append_timeline(
            "agent_end",
            {"iteration": rec.iteration, "name": name, "success": ok, "error": failure},
        )
        return ok, failure

    def _evaluation_phase(
        self,
        rec: IterationRecord,
        phase: P.Phase,
        evaluator_phase: str,
        submission_dir: Path,
        artifact_dir: Path,
        logs_dir: Path,
    ) -> EvaluationResult:
        self._start_phase(rec, phase)
        try:
            result = self.evaluator.run(
                evaluator_phase,
                submission_dir,
                artifact_dir,
                logs_dir,
                role="candidate",
                build_fingerprint=self.builder.profile.fingerprint,
                baseline_report=(
                    self.baseline.get("benchmark") if evaluator_phase == "benchmark" else None
                ),
            )
            if (
                evaluator_phase == "benchmark" and result.passed
                and self.profiler is not None
            ):
                profile_result = self.profiler.run(
                    artifact_dir, logs_dir, role="candidate"
                )
                result.report["hardware_profile"] = profile_result.report
                if not profile_result.passed and self.profiler.profile.required:
                    result = EvaluationResult(
                        evaluator_phase, False, result.report,
                        profile_result.failure or "required hardware profile failed", True,
                    )
        except Exception as exc:  # noqa: BLE001
            result = EvaluationResult(
                evaluator_phase, False, {}, f"evaluator crashed: {exc!r}", True
            )
        outcome = P.OK if result.passed else (
            P.INFRA_FAIL if result.infra_failure else (
                P.PERF_REGRESSION if evaluator_phase == "benchmark" else P.LOGIC_FAIL
            )
        )
        summary: Dict[str, Any] = {
            "report": str(logs_dir / f"candidate-{evaluator_phase}-report.json"),
            "build_fingerprint": self.builder.profile.fingerprint,
        }
        if evaluator_phase == "benchmark":
            summary["score"] = result.report.get("score")
            summary["hardware_profile"] = str(
                logs_dir / "candidate-hardware-profile.json"
            )
        if evaluator_phase == "correctness":
            summary["summary"] = result.report.get("summary")
        self._end_phase(rec, phase, outcome, result.failure, summary)
        return result

    def _test_phase(
        self, rec: IterationRecord, submission_dir: Path, logs_dir: Path,
    ) -> tuple[BuildResult, EvaluationResult, Optional[EvaluationResult]]:
        """C phase: fixed system compilation followed by harness correctness."""
        self._start_phase(rec, "C_test")
        build_dir = logs_dir / "build"
        build_result = self.builder.build(submission_dir, build_dir)
        compile_result = EvaluationResult(
            "compile", build_result.passed, build_result.report,
            build_result.failure, build_result.infra_failure,
        )
        summary: Dict[str, Any] = {
            "compile_report": str(build_dir / "compile-report.json"),
            "build_fingerprint": self.builder.profile.fingerprint,
        }
        if not build_result.passed:
            outcome = P.INFRA_FAIL if build_result.infra_failure else P.LOGIC_FAIL
            self._end_phase(rec, "C_test", outcome, build_result.failure, summary)
            return build_result, compile_result, None

        try:
            correctness = self.evaluator.run(
                "correctness",
                submission_dir,
                build_result.artifact_dir,
                logs_dir,
                role="candidate",
                build_fingerprint=self.builder.profile.fingerprint,
            )
        except Exception as exc:  # noqa: BLE001
            correctness = EvaluationResult(
                "correctness", False, {}, f"evaluator crashed: {exc!r}", True
            )
        summary.update({
            "correctness_report": str(logs_dir / "candidate-correctness-report.json"),
            "correctness_summary": correctness.report.get("summary"),
        })
        outcome = P.OK if correctness.passed else (
            P.INFRA_FAIL if correctness.infra_failure else P.LOGIC_FAIL
        )
        self._end_phase(rec, "C_test", outcome, correctness.failure, summary)
        return build_result, compile_result, correctness

    def _ensure_baseline(self) -> Dict[str, Any]:
        """Compile, validate and measure the original submission before agents run."""
        baseline_dir = self.cfg.state_dir / "baseline"
        submission = baseline_dir / "submission"
        manifest_path = baseline_dir / "baseline-manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            digest = manifest.pop("manifest_sha256", None)
            actual = _canonical_digest(manifest)
            if digest != actual:
                raise RuntimeError("frozen baseline manifest changed")
            if manifest.get("build_fingerprint") != self.builder.profile.fingerprint:
                raise RuntimeError("baseline BuildProfile differs from active BuildProfile")
            if manifest.get("evaluator_digest") != self.cfg.evaluator_bundle.digest:
                raise RuntimeError("baseline evaluator differs from active evaluator")
            if self.profiler is not None:
                expected_profile = self.profiler.profile.fingerprint
                actual_profile = (
                    manifest.get("hardware_profile") or {}
                ).get("profile_fingerprint")
                if actual_profile != expected_profile:
                    raise RuntimeError("baseline profiler differs from active hardware profile")
            if not submission.is_dir() or manifest.get("submission_digest") != _tree_digest(submission):
                raise RuntimeError("frozen baseline submission changed")
            self.builder.verify()
            self.store.append_timeline(
                "baseline_reused",
                {"build_fingerprint": self.builder.profile.fingerprint},
            )
            return manifest

        initial = self.cfg.initial_submission
        if initial is None or not initial.is_dir():
            raise RuntimeError("initial_submission is required for baseline certification")
        baseline_dir.mkdir(parents=True, exist_ok=True)
        if submission.exists():
            shutil.rmtree(submission)
        shutil.copytree(initial, submission)
        build_result = self.builder.build(submission, baseline_dir / "build")
        if not build_result.passed:
            raise RuntimeError(build_result.failure or "baseline did not compile")
        correctness = self.evaluator.run(
            "correctness",
            submission,
            build_result.artifact_dir,
            baseline_dir,
            role="baseline",
            build_fingerprint=self.builder.profile.fingerprint,
        )
        if not correctness.passed:
            raise RuntimeError(correctness.failure or "baseline failed correctness")
        benchmark = self.evaluator.run(
            "benchmark",
            submission,
            build_result.artifact_dir,
            baseline_dir,
            role="baseline",
            build_fingerprint=self.builder.profile.fingerprint,
        )
        if not benchmark.passed:
            raise RuntimeError(benchmark.failure or "baseline benchmark failed")
        hardware_profile: Dict[str, Any] = {}
        if self.profiler is not None:
            profiled = self.profiler.run(
                build_result.artifact_dir, baseline_dir, role="baseline"
            )
            hardware_profile = profiled.report
            if not profiled.passed and self.profiler.profile.required:
                raise RuntimeError(profiled.failure or "baseline hardware profile failed")
        payload = {
            "schema_version": 1,
            "certified_at": time.time(),
            "build_fingerprint": self.builder.profile.fingerprint,
            "evaluator_digest": self.cfg.evaluator_bundle.digest,
            "submission_digest": _tree_digest(submission),
            "compile": build_result.report,
            "correctness": correctness.report,
            "benchmark": benchmark.report,
            "hardware_profile": hardware_profile,
        }
        payload["manifest_sha256"] = _canonical_digest(payload)
        manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.store.append_timeline(
            "baseline_certified",
            {
                "build_fingerprint": self.builder.profile.fingerprint,
                "benchmark_cases": len(benchmark.report.get("cases") or []),
            },
        )
        return payload

    def _review(
        self,
        rec: IterationRecord,
        iter_dir: Path,
        feedback: Dict[str, Any],
        *,
        test_passed: bool,
    ) -> None:
        ok, _ = self._agent_phase(
            rec,
            "D_review",
            role="reviewer",
            workdir=iter_dir,
            prompt=review_prompt(iter_dir, self.cfg.notebooks_dir, rec.iteration, feedback),
            # As in the C++/Python outer loop, D is advisory but its egress
            # reflects C: C-pass advances to E; C-fail closes for replanning.
            success_outcome=P.OK if test_passed else P.LOGIC_FAIL,
        )
        review = iter_dir / "review.md"
        if review.is_file():
            rec.retrospective_path = str(review)
            rec.artifacts.append(str(review))
            self._write(rec)
        if not ok:
            self.store.append_timeline("review_warning", {"iteration": rec.iteration})

    def _perf_plan(
        self,
        rec: IterationRecord,
        iter_dir: Path,
        feedback: Dict[str, Any],
        promotion: Dict[str, Any],
    ) -> None:
        ok, _ = self._agent_phase(
            rec,
            "F_perf_plan",
            role="perf_planner",
            workdir=iter_dir,
            prompt=perf_plan_prompt(
                iter_dir,
                self.cfg.notebooks_dir,
                rec.iteration,
                feedback,
                promotion,
            ),
        )
        plan = iter_dir / "perf_plan.md"
        if plan.is_file():
            rec.artifacts.append(str(plan))
            self._write(rec)
        if not ok:
            self.store.append_timeline(
                "perf_plan_warning", {"iteration": rec.iteration}
            )

    def _write_feedback(
        self,
        logs_dir: Path,
        *,
        compile_result: Optional[EvaluationResult] = None,
        correctness_result: Optional[EvaluationResult] = None,
        benchmark_result: Optional[EvaluationResult] = None,
        promotion: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        private = set(self.cfg.evaluator_bundle.spec.private_case_ids)
        feedback: Dict[str, Any] = {}
        if compile_result:
            feedback["compile"] = {
                "passed": compile_result.passed,
                "failure": compile_result.failure,
                "build_fingerprint": compile_result.report.get("build_fingerprint"),
                "stdout_tail": compile_result.report.get("stdout_tail", ""),
                "stderr_tail": compile_result.report.get("stderr_tail", ""),
            }
        if correctness_result:
            all_cases = [
                case for case in correctness_result.report.get("cases", [])
                if isinstance(case, dict)
            ]
            public_cases = [
                case for case in correctness_result.report.get("cases", [])
                if isinstance(case, dict) and str(case.get("id")) not in private
            ]
            private_cases = [case for case in all_cases if str(case.get("id")) in private]
            public_failed = [
                str(case.get("id")) for case in public_cases if case.get("passed") is not True
            ]
            heldout_passed = (
                len(private_cases) == len(private)
                and all(case.get("passed") is True for case in private_cases)
            )
            feedback["correctness"] = {
                "passed": correctness_result.passed,
                "failure": (
                    None if correctness_result.passed else
                    f"public_failed={public_failed}; held_out_passed={heldout_passed}"
                ),
                "public_cases": public_cases,
                "held_out": {
                    "count": len(private),
                    "passed": heldout_passed,
                },
            }
        if benchmark_result:
            score = dict(benchmark_result.report.get("score") or {})
            score["cases"] = [
                case for case in score.get("cases", [])
                if str(case.get("id")) not in private
            ]
            score["reasons"] = [
                _redact_private(str(reason), private)
                for reason in score.get("reasons", [])
            ]
            feedback["benchmark"] = {
                "passed": benchmark_result.passed,
                "failure": _redact_private(benchmark_result.failure, private),
                "score": score,
                "methodology": benchmark_result.report.get("methodology") or {},
                "hardware_profile": _agent_profile_feedback(
                    benchmark_result.report.get("hardware_profile") or {}
                ),
            }
        if promotion:
            feedback["promotion"] = promotion
        path = logs_dir / "feedback.json"
        path.write_text(json.dumps(feedback, indent=2), encoding="utf-8")
        return feedback

    def _load_prior_feedback(self, n: int) -> Optional[Dict[str, Any]]:
        if n <= 1:
            return None
        path = self.workspace.logs_dir_for(n) / "prev-iter" / "feedback.json"
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
        except (OSError, json.JSONDecodeError):
            return None

    def _start_phase(self, rec: IterationRecord, phase: P.Phase) -> None:
        rec.phases[phase] = {"started_at": time.time(), "status": "running"}
        self._write(rec)
        self.store.update_run(current_iteration=rec.iteration, current_phase=phase)
        self.store.append_timeline("phase_start", {"iteration": rec.iteration, "phase": phase})

    def _end_phase(
        self,
        rec: IterationRecord,
        phase: P.Phase,
        outcome: P.Outcome,
        failure: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        item = rec.phases.setdefault(phase, {})
        item.update({"ended_at": time.time(), "status": "done", "outcome": outcome})
        if failure:
            item["failure"] = failure
        if extra:
            item.update(extra)
        self._write(rec)
        self.store.update_run(last_outcome=outcome, last_transition_label=f"{phase}: {outcome}")
        self.store.append_timeline(
            "phase_end", {"iteration": rec.iteration, "phase": phase, "outcome": outcome}
        )

    def _finish_failed(self, rec: IterationRecord, outcome: P.Outcome, failure: str) -> P.Outcome:
        return self._finish(rec, "failed", outcome, failure)

    def _finish(
        self,
        rec: IterationRecord,
        status: str,
        outcome: P.Outcome,
        failure: Optional[str],
    ) -> P.Outcome:
        rec.ended_at = time.time()
        rec.duration_s = rec.ended_at - rec.started_at
        rec.status = status
        rec.outcome = outcome
        rec.failure_reason = failure
        self._write(rec)
        self.workspace.mark_complete(rec.iteration)
        self.store.append_timeline(
            "iteration_end",
            {
                "iteration": rec.iteration,
                "status": status,
                "outcome": outcome,
                "promoted": rec.promoted,
                "score": rec.score,
            },
        )
        return outcome

    def _write(self, rec: IterationRecord) -> None:
        self.store.write_iteration(rec.iteration, rec.to_dict())


def _redact_private(value: Optional[str], private_ids: set[str]) -> Optional[str]:
    if value is None:
        return None
    redacted = value
    for case_id in sorted(private_ids, key=len, reverse=True):
        redacted = redacted.replace(case_id, "<held-out>")
    return redacted


def _agent_profile_feedback(report: Dict[str, Any]) -> Dict[str, Any]:
    """Expose measurements to F without leaking system paths/launch commands."""
    return {
        "passed": report.get("passed"),
        "profile_id": report.get("profile_id"),
        "gpu_arch": report.get("gpu_arch"),
        "tool": report.get("tool"),
        "counter_groups": report.get("counter_groups") or [],
        "cases": report.get("cases") or [],
    }


def _canonical_digest(data: Dict[str, Any]) -> str:
    payload = dict(data)
    payload.pop("manifest_sha256", None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"baseline submission contains symlink: {path}")
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()
