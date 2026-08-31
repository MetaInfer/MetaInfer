"""MVP control plane: baseline, parallel mock workers, serial validation."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict

from metainfer.orchestrator.state import StateStore

from . import phases
from .adapters.mock import MockKernelAdapter
from .config import (
    ROUND_ACCEPTANCE_IMPROVEMENT_PERCENT,
    OptimizerConfig,
    load_config,
)
from .result_store import SCHEMA_VERSION, write_json
from .skill_store import generate_merged_skill, generate_worker_skill
from .worker import run_mock_worker


class MockOptimizationPipeline:
    def __init__(
        self,
        *,
        req: Dict[str, Any],
        state_dir: Path,
        workspace_dir: Path,
        store: StateStore,
    ) -> None:
        self.req = req
        self.state_dir = state_dir
        self.workspace_dir = workspace_dir
        self.store = store

    def _phase(self, phase: str, **payload: Any) -> None:
        self.store.update_run(current_phase=phase)
        self.store.append_timeline(
            "phase_start", {"phase": phase, **payload}
        )

    def run(self, *, dry_run: bool = False) -> Dict[str, Any]:
        task_id = str(self.req.get("task_id", "task"))
        self.store.init_or_resume(task_id, "dcu-kernel-auto-opt")
        self.store.update_run(finished=False, final_status=None, notes=[])
        started = time.time()
        try:
            self._phase(phases.PREPARE)
            config = load_config(self.req)
            self._create_layout(config)
            plan = self._plan_payload(config)
            write_json(self.workspace_dir / "plan.json", plan)
            if dry_run:
                result = {**plan, "dry_run": True, "status": "success"}
                write_json(self.workspace_dir / "final_report.json", result)
                self.store.update_run(
                    current_phase=phases.FINISHED,
                    finished=True,
                    final_status="success",
                    last_outcome="ok",
                )
                return result

            self._phase(phases.BASELINE)
            baseline = self._baseline(config)

            self._phase(phases.EXPLORE, workers=len(config.assignments))
            worker_results = self._parallel_workers(config, baseline)

            self._phase(phases.SYNTHESIZE)
            merged_skill = generate_merged_skill(
                config=config,
                assignments=config.assignments,
                workspace_dir=self.workspace_dir,
            )

            self._phase(phases.VALIDATE)
            validation = self._serial_validate(config, worker_results)

            self._phase(phases.REPORT)
            report = {
                "schema_version": SCHEMA_VERSION,
                "task_id": task_id,
                "task_type": "dcu-kernel-auto-opt",
                "mode": "mock",
                "started_at": started,
                "finished_at": time.time(),
                "duration_s": round(time.time() - started, 4),
                "config": plan,
                "baseline": baseline,
                "workers": worker_results,
                "merged_skill": merged_skill,
                "final_validation": validation,
                "real_gpu_used": False,
                "target_repo_modified": False,
                "status": "success",
            }
            write_json(self.workspace_dir / "final_report.json", report)
            (self.workspace_dir / "report.md").write_text(
                self._markdown_report(report), encoding="utf-8"
            )
            self.store.write_iteration(1, {
                "iteration": 1,
                "status": "success",
                "goal": "validate multi-worker mock orchestration",
                "started_at": started,
                "ended_at": report["finished_at"],
                "duration_s": report["duration_s"],
                "perf": {},
                "workers": worker_results,
            })
            self.store.update_run(
                current_iteration=1,
                current_phase=phases.FINISHED,
                finished=True,
                final_status="success",
                last_outcome="ok",
                last_transition_label="report complete",
            )
            self.store.append_timeline(
                "orchestrator_success",
                {"workers": len(worker_results), "real_gpu_used": False},
            )
            return report
        except Exception as exc:
            self.store.append_timeline(
                "orchestrator_error", {"error": repr(exc)}
            )
            self.store.update_run(
                current_phase=phases.FINISHED,
                finished=True,
                final_status="stopped",
                last_outcome="infra_fail",
                notes=[str(exc)],
            )
            raise

    def _create_layout(self, config: OptimizerConfig) -> None:
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        for name in ("main", "shared_baseline", "final_validation", "workers"):
            (self.workspace_dir / name).mkdir(parents=True, exist_ok=True)
        for assignment in config.assignments:
            (self.workspace_dir / "workers" / assignment.worker_id).mkdir(
                parents=True, exist_ok=True
            )

    @staticmethod
    def _plan_payload(config: OptimizerConfig) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "execution_mode": config.execution_mode,
            "operator": config.operator,
            "dtype": config.dtype,
            "hardware": config.hardware,
            "kernel_language": config.kernel_language,
            "claude_model": config.claude_model,
            "target_repo_path": (
                str(config.target_repo_path) if config.target_repo_path else None
            ),
            "mock_iterations": config.mock_iterations,
            "minimum_improvement_percent": config.minimum_improvement_percent,
            "minimum_improvement_semantics": (
                "final validated result versus fixed baseline"
            ),
            "round_acceptance_improvement_percent": (
                ROUND_ACCEPTANCE_IMPROVEMENT_PERCENT
            ),
            "shapes": [
                {"id": shape.id, **shape.params}
                for shape in config.shapes.values()
            ],
            "assignments": [
                {
                    "worker_id": a.worker_id,
                    "gpu": a.gpu,
                    "shapes": a.shape_ids,
                }
                for a in config.assignments
            ],
            "real_gpu_used": False,
        }

    def _baseline(
        self, config: OptimizerConfig
    ) -> Dict[str, Dict[str, float]]:
        adapter = MockKernelAdapter()
        adapter.prepare(self.workspace_dir / "shared_baseline")
        baseline: Dict[str, Dict[str, float]] = {}
        for shape in config.shapes.values():
            correct = adapter.correctness(
                self.workspace_dir / "shared_baseline", shape
            )
            if not correct.success:
                raise RuntimeError(f"baseline correctness failed: {shape.id}")
            result = adapter.benchmark(
                self.workspace_dir / "shared_baseline", shape, iteration=0
            )
            if not result.success:
                raise RuntimeError(f"baseline benchmark failed: {shape.id}")
            baseline[shape.id] = result.metrics
        write_json(
            self.workspace_dir / "shared_baseline" / "results.json",
            {"schema_version": SCHEMA_VERSION, "shapes": baseline},
        )
        return baseline

    def _parallel_workers(
        self,
        config: OptimizerConfig,
        baseline: Dict[str, Dict[str, float]],
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        with ThreadPoolExecutor(
            max_workers=len(config.assignments),
            thread_name_prefix="mock-gpu-worker",
        ) as pool:
            futures = {
                pool.submit(
                    run_mock_worker,
                    assignment=assignment,
                    config=config,
                    baseline=baseline,
                    worker_root=self.workspace_dir / "workers" / assignment.worker_id,
                    guidance_root=self.state_dir / "guidance",
                    adapter_factory=MockKernelAdapter,
                ): assignment.worker_id
                for assignment in config.assignments
            }
            for future in as_completed(futures):
                worker_id = futures[future]
                out[worker_id] = future.result()
                assignment = next(
                    item for item in config.assignments
                    if item.worker_id == worker_id
                )
                out[worker_id]["skill"] = generate_worker_skill(
                    config=config,
                    assignment=assignment,
                    workspace_dir=self.workspace_dir,
                )
                self.store.append_timeline(
                    "worker_complete", {
                        "worker_id": worker_id,
                        "skill": out[worker_id]["skill"]["name"],
                    }
                )
        return dict(sorted(out.items()))

    def _serial_validate(
        self, config: OptimizerConfig, workers: Dict[str, Any]
    ) -> Dict[str, Any]:
        adapter = MockKernelAdapter()
        root = self.workspace_dir / "final_validation"
        results: Dict[str, Any] = {}
        for assignment in config.assignments:
            for shape_id in assignment.shape_ids:
                shape = config.shapes[shape_id]
                best = workers[assignment.worker_id]["shapes"][shape_id]
                correctness = adapter.correctness(root, shape)
                benchmark = adapter.benchmark(
                    root, shape, iteration=int(best["iteration"])
                )
                passed = correctness.success and benchmark.success
                results[shape_id] = {
                    "passed": passed,
                    "worker_id": assignment.worker_id,
                    "physical_gpu": assignment.gpu,
                    "candidate": best["mock_candidate"],
                    "metrics": benchmark.metrics,
                    "serial": True,
                }
                if not passed:
                    raise RuntimeError(f"final validation failed: {shape_id}")
        write_json(root / "results.json", {
            "schema_version": SCHEMA_VERSION,
            "shapes": results,
            "real_gpu_used": False,
        })
        return results

    @staticmethod
    def _markdown_report(report: Dict[str, Any]) -> str:
        lines = [
            "# DCU Kernel Auto-Optimization — Mock MVP",
            "",
            "No GPU was used and no target repository was modified.",
            "",
            "| Shape | Worker | Candidate | Median (us) |",
            "|---|---|---|---:|",
        ]
        for shape_id, result in report["final_validation"].items():
            lines.append(
                f"| {shape_id} | {result['worker_id']} | "
                f"{result['candidate']} | {result['metrics']['median_us']} |"
            )
        return "\n".join(lines) + "\n"
