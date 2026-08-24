"""Arena-style GEMM optimization pipeline with a frozen external judge."""

from __future__ import annotations

import json
import hashlib
import shutil
import statistics
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
from .evaluator.champion import (
    ReportReference,
    load_report_reference,
    make_report_reference,
    write_json_atomic,
)
from .guidance import GuidanceStore
from .iteration_record import IterationRecord
from .plugin import PLUGIN
from .profiler import ProfilerRunner
from .prompts import (
    implement_prompt,
    perf_plan_prompt,
    plan_prompt,
    repair_prompt,
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
            cfg.evaluator_bundle.spec.benchmark_case_ids,
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
            self.baseline = self._ensure_triton_baseline()
            self.champions.initialize_triton(self.baseline["benchmark_report"])
            self.initial_hip = self._ensure_initial_hip()
            initial_score = dict(self.initial_hip["benchmark"].get("score") or {})
            if initial_score.get("passed"):
                promoted, reason, champion = self.champions.consider(
                    0,
                    self.cfg.state_dir / "certified" / "initial-hip" / "submission",
                    self.initial_hip["benchmark_report"],
                    self.baseline["benchmark_report"],
                )
                self.store.append_timeline("initial_hip_challenged", {
                    "promoted": promoted, "reason": reason, "champion": champion,
                })
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

        if not (
            compile_result.passed and correctness is not None and correctness.passed
        ):
            ok, repair_failure = self._agent_phase(
                rec, "B_implement", role="repair", workdir=submission_dir,
                prompt=repair_prompt(self.agent_req, submission_dir, n, test_feedback),
            )
            if ok:
                build_result, compile_result, correctness = self._test_phase(
                    rec, submission_dir, logs_dir
                )
                test_feedback = self._write_feedback(
                    logs_dir, compile_result=compile_result,
                    correctness_result=correctness,
                )
                rec.phases["C_test"]["repair_attempted"] = True
                self._write(rec)
            else:
                rec.phases.setdefault("C_test", {})["repair_failure"] = repair_failure
                self._write(rec)

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
            rec, "E_perf_test", submission_dir,
            build_result.artifact_dir, logs_dir,
        )
        score = dict(benchmark.report.get("score") or {})
        rec.score = score
        if benchmark.infra_failure:
            perf_feedback = self._write_feedback(
                logs_dir, compile_result=compile_result,
                correctness_result=correctness, benchmark_result=benchmark,
            )
            del perf_feedback
            return self._finish_failed(
                rec, P.INFRA_FAIL,
                benchmark.failure or "benchmark infrastructure failure",
            )

        diagnostic_ids = [] if benchmark.passed else list(
            score.get("failed_case_ids") or []
        )

        if self.profiler is not None and (benchmark.passed or diagnostic_ids):
            diagnostic = self.profiler.run(
                build_result.artifact_dir, logs_dir, role="candidate",
                collection_mode="full",
                case_ids=None if benchmark.passed else diagnostic_ids,
                run_label="candidate-diagnostic",
                implementation="candidate",
            )
            if diagnostic.passed:
                diagnostic_path = logs_dir / "candidate-diagnostic-hardware-profile.json"
                rec.profile_report = make_report_reference(
                    self.cfg.state_dir, diagnostic_path
                )
                benchmark.report["_profile_report"] = diagnostic.report
            elif benchmark.passed:
                benchmark.infra_failure = True
                benchmark.passed = False
                benchmark.failure = (
                    diagnostic.failure or "promotable candidate full PMC archive failed"
                )
                self._write_feedback(
                    logs_dir, compile_result=compile_result,
                    correctness_result=correctness, benchmark_result=benchmark,
                )
                return self._finish_failed(
                    rec, P.INFRA_FAIL, benchmark.failure
                )
        promoted = False
        reason = benchmark.failure or "benchmark failed"
        champion = self.champions.load()
        if benchmark.passed:
            promoted, reason, champion = self.champions.consider(
                n,
                submission_dir,
                rec.measurement_report,
                rec.incumbent_measurement_report,
            )
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

        outcome = P.OK if promoted else P.PERF_REGRESSION
        return self._finish(rec, "success" if promoted else "not_promoted", outcome, reason if not promoted else None)

    def _seed_from_champion(self, iter_dir: Path) -> None:
        champion = self.champions.load()  # verifies persisted HIP source when present
        submission = iter_dir / "submission"
        if submission.exists():
            shutil.rmtree(submission)
        seed = (
            self.champions.submission_dir
            if champion.get("kind") == "hip" else
            self.cfg.state_dir / "certified" / "initial-hip" / "submission"
        )
        if seed.is_dir():
            shutil.copytree(seed, submission)
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
        submission_dir: Path,
        artifact_dir: Path,
        logs_dir: Path,
    ) -> EvaluationResult:
        self._start_phase(rec, phase)
        try:
            incumbent = self.champions.load()
            if incumbent.get("kind") == "hip":
                incumbent_build = self.builder.build(
                    self.champions.submission_dir, logs_dir / "incumbent-build"
                )
                if not incumbent_build.passed:
                    raise RuntimeError(
                        incumbent_build.failure or "current Champion did not rebuild"
                    )
                incumbent_artifact = incumbent_build.artifact_dir
                incumbent_impl = "candidate"
                incumbent_fingerprint = self.builder.profile.fingerprint
            else:
                incumbent_artifact = self.cfg.state_dir / "baseline" / "runtime-artifacts"
                incumbent_impl = "triton"
                incumbent_fingerprint = "triton-jit"
            incumbent_result, incumbent_ref, incumbent_profile_ref, _ = self._profile_benchmark(
                incumbent_artifact, logs_dir, role="baseline",
                build_fingerprint=incumbent_fingerprint,
                report_label="incumbent", collection_mode="trace",
                implementation=incumbent_impl,
            )
            if not incumbent_result.passed or incumbent_ref is None:
                raise RuntimeError(
                    incumbent_result.failure or "same-round Champion trace failed"
                )
            rec.incumbent_measurement_report = incumbent_ref
            if incumbent_profile_ref:
                rec.incumbent_profile_report = incumbent_profile_ref
            result, measurement_ref, profile_ref, profile_report = (
                self._profile_benchmark(
                    artifact_dir,
                    logs_dir,
                    role="candidate",
                    build_fingerprint=self.builder.profile.fingerprint,
                    baseline_report=incumbent_result.report,
                    collection_mode="trace",
                )
            )
            if result.passed and _near_promotion_boundary(
                incumbent_result.report, result.report,
                self.cfg.evaluator_bundle.spec.acceptance.noise_threshold,
            ):
                incumbent_retry, _, _, _ = self._profile_benchmark(
                    incumbent_artifact, logs_dir, role="baseline",
                    build_fingerprint=incumbent_fingerprint,
                    report_label="incumbent-retest", collection_mode="trace",
                    implementation=incumbent_impl,
                )
                candidate_retry, _, _, _ = self._profile_benchmark(
                    artifact_dir, logs_dir, role="candidate",
                    build_fingerprint=self.builder.profile.fingerprint,
                    baseline_report=incumbent_retry.report,
                    report_label="candidate-retest", collection_mode="trace",
                )
                if not incumbent_retry.passed or candidate_retry.infra_failure:
                    raise RuntimeError("boundary retest hipprof trace failed")
                incumbent_combined = _combine_hipprof_reports(
                    incumbent_result.report, incumbent_retry.report
                )
                candidate_combined = _combine_hipprof_reports(
                    result.report, candidate_retry.report
                )
                incumbent_path = logs_dir / "incumbent-combined-benchmark-report.json"
                candidate_path = logs_dir / "candidate-combined-benchmark-report.json"
                write_json_atomic(incumbent_path, incumbent_combined)
                incumbent_ref = make_report_reference(self.cfg.state_dir, incumbent_path)
                rec.incumbent_measurement_report = incumbent_ref
                result = self.evaluator.validate_benchmark_report(
                    candidate_combined, role="candidate",
                    build_fingerprint=self.builder.profile.fingerprint,
                    baseline_report=incumbent_combined,
                )
                result.report["boundary_retested"] = True
                write_json_atomic(candidate_path, result.report)
                measurement_ref = make_report_reference(self.cfg.state_dir, candidate_path)
            if measurement_ref:
                rec.measurement_report = measurement_ref
            if profile_ref:
                rec.profile_report = profile_ref
            if profile_report:
                result.report["_profile_report"] = profile_report
        except Exception as exc:  # noqa: BLE001
            result = EvaluationResult(
                "benchmark", False, {}, f"hipprof evaluation crashed: {exc!r}", True
            )
        outcome = P.OK if result.passed else (
            P.INFRA_FAIL if result.infra_failure else P.PERF_REGRESSION
        )
        summary: Dict[str, Any] = {
            "report": str(logs_dir / "candidate-benchmark-report.json"),
            "build_fingerprint": self.builder.profile.fingerprint,
            "score": result.report.get("score"),
            "measurement_report": dict(rec.measurement_report),
            "profile_report": dict(rec.profile_report),
            "incumbent_measurement_report": dict(rec.incumbent_measurement_report),
        }
        self._end_phase(rec, phase, outcome, result.failure, summary)
        return result

    def _profile_benchmark(
        self,
        artifact_dir: Path,
        report_dir: Path,
        *,
        role: str,
        build_fingerprint: str,
        baseline_report: Optional[Dict[str, Any]] = None,
        report_label: Optional[str] = None,
        collection_mode: str = "trace",
        implementation: Optional[str] = None,
    ) -> tuple[
        EvaluationResult,
        Optional[ReportReference],
        Optional[ReportReference],
        Dict[str, Any],
    ]:
        report_dir.mkdir(parents=True, exist_ok=True)
        label = report_label or role
        benchmark_path = report_dir / f"{label}-benchmark-report.json"
        if self.profiler is None:
            result = EvaluationResult(
                "benchmark", False, {}, "required hipprof profiler is unavailable", True
            )
            return result, None, None, {}

        profiled = self.profiler.run(
            artifact_dir, report_dir, role=role,
            collection_mode=collection_mode, run_label=label,
            implementation=implementation,
        )

        profile_report = dict(profiled.report)
        profile_path = report_dir / f"{label}-hardware-profile.json"
        try:
            profile_ref = make_report_reference(self.cfg.state_dir, profile_path)
        except RuntimeError as exc:
            result = EvaluationResult(
                "benchmark", False, {}, f"hipprof report is unavailable: {exc}", True
            )
            return result, None, None, profile_report

        report: Dict[str, Any] = {
            "schema_version": 2,
            "passed": bool(profiled.passed),
            "methodology": dict(self.cfg.evaluator_bundle.spec.benchmark_protocol),
            "timing_source": "hipprof GPU kernel DurationNs",
            "timed_scope": "operator_gpu_dispatches_only",
            "profile_report": dict(profile_ref),
            "cases": [],
        }
        if not profiled.passed:
            write_json_atomic(benchmark_path, report)
            measurement_ref = make_report_reference(self.cfg.state_dir, benchmark_path)
            result = EvaluationResult(
                "benchmark",
                False,
                report,
                profiled.failure or "required hipprof profile failed",
                True,
            )
            return result, measurement_ref, profile_ref, profile_report

        specs = {
            spec.id: spec for spec in self.cfg.evaluator_bundle.spec.benchmark_cases
        }
        timing_cases = profile_report.get("timing_cases") or []
        ids = [
            str(case.get("id") or "")
            for case in timing_cases
            if isinstance(case, dict)
        ]
        duplicates = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
        expected = self.cfg.evaluator_bundle.spec.benchmark_case_ids
        missing = sorted(set(expected) - set(ids))
        unexpected = sorted(set(ids) - set(expected))
        invalid = len(ids) != len(timing_cases) or "" in ids
        if missing or unexpected or duplicates or invalid:
            report["passed"] = False
            report["profile_case_errors"] = {
                "missing": missing,
                "unexpected": unexpected,
                "duplicate": duplicates,
                "invalid": invalid,
            }
            write_json_atomic(benchmark_path, report)
            measurement_ref = make_report_reference(self.cfg.state_dir, benchmark_path)
            failure = (
                "hipprof timing cases are incomplete: "
                f"missing={missing}, unexpected={unexpected}, "
                f"duplicate={duplicates}, invalid={invalid}"
            )
            result = EvaluationResult("benchmark", False, report, failure, True)
            return result, measurement_ref, profile_ref, profile_report

        cases: List[Dict[str, Any]] = []
        for raw in timing_cases:
            case_id = str(raw["id"])
            spec = specs[case_id]
            case = dict(raw)
            case["timing_source"] = "hipprof GPU kernel DurationNs"
            if spec.shape is not None:
                case["shape"] = dict(spec.shape)
            if spec.flops is not None:
                case["flops"] = spec.flops
            if spec.bytes is not None:
                case["bytes"] = spec.bytes
            cases.append(case)
        report["cases"] = cases
        result = self.evaluator.validate_benchmark_report(
            report,
            role=role,
            build_fingerprint=build_fingerprint,
            baseline_report=baseline_report,
        )
        write_json_atomic(benchmark_path, result.report)
        measurement_ref = make_report_reference(self.cfg.state_dir, benchmark_path)
        return result, measurement_ref, profile_ref, profile_report

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

    def _ensure_triton_baseline(self) -> Dict[str, Any]:
        """Certify frozen Triton as the initial arena Champion."""
        baseline_dir = self.cfg.state_dir / "baseline"
        submission = baseline_dir / "submission"
        artifact_dir = baseline_dir / "runtime-artifacts"
        manifest_path = baseline_dir / "baseline-manifest.json"
        if manifest_path.is_file():
            stored = json.loads(manifest_path.read_text(encoding="utf-8"))
            digest = stored.get("manifest_sha256")
            manifest = {
                key: value for key, value in stored.items()
                if key != "manifest_sha256"
            }
            if digest != _canonical_digest(manifest):
                raise RuntimeError("frozen baseline manifest changed")
            if manifest.get("evaluator_digest") != self.cfg.evaluator_bundle.digest:
                raise RuntimeError("baseline evaluator differs from active evaluator")
            if manifest.get("implementation") != "triton":
                raise RuntimeError("baseline is not the certified Triton implementation")
            benchmark_ref = manifest.get("benchmark_report")
            profile_ref = manifest.get("profile_report")
            if not isinstance(benchmark_ref, dict):
                benchmark_ref = make_report_reference(
                    self.cfg.state_dir,
                    baseline_dir / "baseline-benchmark-report.json",
                )
            if not isinstance(profile_ref, dict):
                profile_ref = make_report_reference(
                    self.cfg.state_dir,
                    baseline_dir / "baseline-hardware-profile.json",
                )
            benchmark = load_report_reference(self.cfg.state_dir, benchmark_ref)
            profile = load_report_reference(self.cfg.state_dir, profile_ref)
            validated = self.evaluator.validate_benchmark_report(
                benchmark,
                role="baseline",
                build_fingerprint="triton-jit",
            )
            if not validated.passed:
                raise RuntimeError(validated.failure or "frozen Triton benchmark is invalid")
            if self.profiler is None:
                raise RuntimeError("required hipprof profiler is unavailable")
            if profile.get("profile_fingerprint") != self.profiler.profile.fingerprint:
                raise RuntimeError(
                    "baseline profiler differs from active hardware profile"
                )
            self.store.append_timeline(
                "baseline_reused", {"implementation": "triton"}
            )
            return {
                **manifest,
                "manifest_sha256": digest,
                "benchmark_report": dict(benchmark_ref),
                "profile_report": dict(profile_ref),
                "benchmark": validated.report,
                "hardware_profile": profile,
            }

        baseline_dir.mkdir(parents=True, exist_ok=True)
        submission.mkdir(parents=True, exist_ok=True)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (submission / "IMPLEMENTATION.json").write_text(
            json.dumps({"kind": "triton", "frozen_evaluator": True}, indent=2),
            encoding="utf-8",
        )
        correctness = self.evaluator.run(
            "correctness", submission, artifact_dir, baseline_dir,
            role="baseline", build_fingerprint="triton-jit",
        )
        if not correctness.passed:
            raise RuntimeError(correctness.failure or "Triton baseline failed correctness")
        benchmark, benchmark_ref, profile_ref, profile = self._profile_benchmark(
            artifact_dir,
            baseline_dir,
            role="baseline",
            build_fingerprint="triton-jit",
            collection_mode="full",
        )
        if not benchmark.passed or benchmark_ref is None or profile_ref is None:
            raise RuntimeError(
                benchmark.failure or "Triton baseline hipprof benchmark failed"
            )
        payload = {
            "schema_version": 3,
            "implementation": "triton",
            "certified_at": time.time(),
            "build_fingerprint": "triton-jit",
            "evaluator_digest": self.cfg.evaluator_bundle.digest,
            "correctness": correctness.report,
            "benchmark_report": dict(benchmark_ref),
            "profile_report": dict(profile_ref),
        }
        payload["manifest_sha256"] = _canonical_digest(payload)
        write_json_atomic(manifest_path, payload)
        self.store.append_timeline(
            "baseline_certified",
            {
                "implementation": "triton",
                "benchmark_cases": len(benchmark.report.get("cases") or []),
                "measurement_report": benchmark_ref,
            },
        )
        return {
            **payload,
            "benchmark": benchmark.report,
            "hardware_profile": profile,
        }

    def _ensure_initial_hip(self) -> Dict[str, Any]:
        """Independently certify the user-provided HIP optimization seed."""
        root = self.cfg.state_dir / "certified" / "initial-hip"
        submission = root / "submission"
        manifest_path = root / "initial-hip-manifest.json"
        if manifest_path.is_file():
            stored = json.loads(manifest_path.read_text(encoding="utf-8"))
            digest = stored.get("manifest_sha256")
            manifest = {
                key: value for key, value in stored.items()
                if key != "manifest_sha256"
            }
            if digest != _canonical_digest(manifest):
                raise RuntimeError("frozen Initial HIP manifest changed")
            if manifest.get("submission_digest") != _tree_digest(submission):
                raise RuntimeError("frozen Initial HIP submission changed")
            if manifest.get("evaluator_digest") != self.cfg.evaluator_bundle.digest:
                raise RuntimeError("Initial HIP evaluator differs from active evaluator")
            benchmark_ref = manifest.get("benchmark_report")
            profile_ref = manifest.get("profile_report")
            if not isinstance(benchmark_ref, dict):
                benchmark_ref = make_report_reference(
                    self.cfg.state_dir,
                    root / "candidate-benchmark-report.json",
                )
            if not isinstance(profile_ref, dict):
                profile_ref = make_report_reference(
                    self.cfg.state_dir,
                    root / "candidate-hardware-profile.json",
                )
            benchmark = load_report_reference(self.cfg.state_dir, benchmark_ref)
            profile = load_report_reference(self.cfg.state_dir, profile_ref)
            validated = self.evaluator.validate_benchmark_report(
                benchmark,
                role="candidate",
                build_fingerprint=self.builder.profile.fingerprint,
                baseline_report=self.baseline["benchmark"],
            )
            if validated.infra_failure:
                raise RuntimeError(validated.failure or "Initial HIP benchmark is invalid")
            if self.profiler is None:
                raise RuntimeError("required hipprof profiler is unavailable")
            if profile.get("profile_fingerprint") != self.profiler.profile.fingerprint:
                raise RuntimeError(
                    "Initial HIP profiler differs from active hardware profile"
                )
            return {
                **manifest,
                "manifest_sha256": digest,
                "benchmark_report": dict(benchmark_ref),
                "profile_report": dict(profile_ref),
                "benchmark": validated.report,
                "hardware_profile": profile,
            }

        initial = self.cfg.initial_submission
        if initial is None or not initial.is_dir():
            raise RuntimeError("initial_submission is required")
        root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(initial, submission)
        build_result = self.builder.build(submission, root / "build")
        if not build_result.passed:
            raise RuntimeError(build_result.failure or "Initial HIP did not compile")
        correctness = self.evaluator.run(
            "correctness", submission, build_result.artifact_dir, root,
            role="candidate", build_fingerprint=self.builder.profile.fingerprint,
        )
        if not correctness.passed:
            raise RuntimeError(correctness.failure or "Initial HIP failed correctness")
        benchmark, benchmark_ref, profile_ref, profile = self._profile_benchmark(
            build_result.artifact_dir,
            root,
            role="candidate",
            build_fingerprint=self.builder.profile.fingerprint,
            baseline_report=self.baseline["benchmark"],
            collection_mode="full",
        )
        if benchmark.infra_failure or benchmark_ref is None or profile_ref is None:
            raise RuntimeError(
                benchmark.failure or "Initial HIP hipprof benchmark failed"
            )
        payload = {
            "schema_version": 2,
            "implementation": "initial-hip",
            "certified_at": time.time(),
            "build_fingerprint": self.builder.profile.fingerprint,
            "evaluator_digest": self.cfg.evaluator_bundle.digest,
            "submission_digest": _tree_digest(submission),
            "compile": build_result.report,
            "correctness": correctness.report,
            "benchmark_report": dict(benchmark_ref),
            "profile_report": dict(profile_ref),
        }
        payload["manifest_sha256"] = _canonical_digest(payload)
        write_json_atomic(manifest_path, payload)
        self.store.append_timeline("initial_hip_certified", {
            "benchmark_cases": len(benchmark.report.get("cases") or []),
            "passed_gates": bool((benchmark.report.get("score") or {}).get("passed")),
            "measurement_report": benchmark_ref,
        })
        return {
            **payload,
            "benchmark": benchmark.report,
            "hardware_profile": profile,
        }

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
                    benchmark_result.report.get("_profile_report") or {}
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
        "cases": [
            {key: value for key, value in case.items()
             if key != "operator_samples_us"}
            for case in report.get("cases") or [] if isinstance(case, dict)
        ],
    }


def _canonical_digest(data: Dict[str, Any]) -> str:
    payload = dict(data)
    payload.pop("manifest_sha256", None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _near_promotion_boundary(
    incumbent: Dict[str, Any], candidate: Dict[str, Any], threshold: float,
) -> bool:
    incumbent_by_id = {
        str(case.get("id")): case for case in incumbent.get("cases") or []
        if isinstance(case, dict) and case.get("id")
    }
    for case in candidate.get("cases") or []:
        case_id = str(case.get("id") or "")
        if case_id not in incumbent_by_id:
            continue
        old = float(incumbent_by_id[case_id]["latency_ms"])
        new = float(case["latency_ms"])
        improvement = 1.0 - new / old
        if abs(improvement - threshold) <= threshold:
            return True
    return False


def _combine_hipprof_reports(
    first: Dict[str, Any], second: Dict[str, Any],
) -> Dict[str, Any]:
    """Combine equal-sized hipprof batches without shape weighting."""
    other = {
        str(case.get("id")): case for case in second.get("cases") or []
        if isinstance(case, dict) and case.get("id")
    }
    combined_cases = []
    for raw in first.get("cases") or []:
        case = dict(raw)
        peer = other.get(str(case.get("id") or ""))
        if peer is None:
            raise RuntimeError("boundary retest cases differ")
        samples = [
            float(value) for value in case.get("operator_samples_ms") or []
        ] + [
            float(value) for value in peer.get("operator_samples_ms") or []
        ]
        if not samples:
            samples = [float(case["latency_ms"]), float(peer["latency_ms"])]
        mean = statistics.fmean(samples)
        case.update({
            "latency_ms": mean,
            "latency_mean_ms": mean,
            "latency_median_ms": statistics.median(samples),
            "latency_stddev_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
            "latency_cv": (
                statistics.stdev(samples) / mean if len(samples) > 1 and mean > 0 else 0.0
            ),
            "latency_min_ms": min(samples),
            "latency_max_ms": max(samples),
            "operator_samples_ms": samples,
            "sample_count": len(samples),
            "measurement_batches": 2,
        })
        combined_cases.append(case)
    result = dict(first)
    result["cases"] = combined_cases
    result["measurement_batches"] = 2
    result["aggregation"] = "arithmetic mean of all raw hipprof DurationNs operator samples"
    result.pop("score", None)
    return result


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
