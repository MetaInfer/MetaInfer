"""Real Claude-agent + real DCU smoke pipeline.

This validates the production isolation and lifecycle without pretending that
the built-in vector kernel is the user's future operator adapter.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict

from metainfer.orchestrator.state import StateStore
from metainfer.orchestrator.subagent_manager import AgentSpec, SubAgentManager

from . import phases
from .config import (
    ROUND_ACCEPTANCE_IMPROVEMENT_PERCENT,
    OptimizerConfig,
    WorkerAssignment,
    load_config,
)
from .gpu_binding import bind_worker_gpu
from .guidance import claim_next_guidance
from .result_store import SCHEMA_VERSION, append_jsonl, write_json
from .skill_store import generate_merged_skill, generate_worker_skill


ASSET = Path(__file__).resolve().parent.parent / "assets" / "smoke_harness.cpp"


def _run(
    command: list[str],
    *,
    cwd: Path,
    env: Dict[str, str] | None = None,
    timeout: int = 180,
) -> subprocess.CompletedProcess[str]:
    effective_command = list(command)
    if effective_command and effective_command[0] == "git":
        # Repositories mounted from the host are host-user-owned, while the
        # orchestrator runs as root. Trust only this exact repository (and its
        # main worktree when cwd is a linked worktree), never a global wildcard.
        safe_directories = [cwd.resolve()]
        git_marker = cwd / ".git"
        if git_marker.is_file():
            marker = git_marker.read_text(encoding="utf-8").strip()
            if marker.startswith("gitdir:"):
                git_dir = Path(marker.removeprefix("gitdir:").strip())
                common_repo, separator, _ = str(git_dir).partition("/.git/worktrees/")
                if separator:
                    safe_directories.append(Path(common_repo).resolve())

        effective_command = ["git"]
        for safe_directory in dict.fromkeys(safe_directories):
            effective_command.extend(["-c", f"safe.directory={safe_directory}"])
        effective_command.extend(command[1:])

    result = subprocess.run(
        effective_command, cwd=cwd, env=env, text=True, capture_output=True,
        timeout=timeout, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{' '.join(command)} failed ({result.returncode}): "
            f"{result.stderr[-2000:] or result.stdout[-2000:]}"
        )
    return result


def _last_json(text: str) -> Dict[str, Any]:
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError(f"no JSON object in output: {text[-1000:]}")


def _elements(shape: Dict[str, Any]) -> int:
    try:
        value = int(shape.get("M", 1)) * int(shape.get("N", 1))
    except (TypeError, ValueError):
        value = 1 << 20
    value = max(1 << 20, min(value, 16 << 20))
    return (value + 3) // 4 * 4


class SmokeRunner:
    def __init__(self, worker_root: Path, gpu: int) -> None:
        self.worker_root = worker_root
        self.source = worker_root / "source"
        self.binary = worker_root / "build" / "smoke_harness"
        self.env = dict(os.environ)
        bind_worker_gpu(self.env, gpu)
        self.env.update({
            "TORCH_EXTENSIONS_DIR": str(worker_root / "cache" / "torch"),
            "TRITON_CACHE_DIR": str(worker_root / "cache" / "triton"),
            "XDG_CACHE_HOME": str(worker_root / "cache" / "xdg"),
            "TMPDIR": str(worker_root / "cache" / "tmp"),
        })
        for key in (
            "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR", "XDG_CACHE_HOME",
            "TMPDIR",
        ):
            Path(self.env[key]).mkdir(parents=True, exist_ok=True)

    def build(self) -> None:
        self.binary.parent.mkdir(parents=True, exist_ok=True)
        _run([
            "/opt/dtk/bin/hipcc", "-O3", "--offload-arch=gfx928",
            str(self.source / "smoke_harness.cpp"), "-o", str(self.binary),
        ], cwd=self.source, env=self.env, timeout=240)

    def probe(self) -> Dict[str, Any]:
        return _last_json(
            _run(
                [str(self.binary), "--probe"], cwd=self.source, env=self.env
            ).stdout
        )

    def benchmark(self, shape: Dict[str, Any], variant: str) -> Dict[str, Any]:
        if variant not in {"scalar", "vector4"}:
            raise ValueError(f"unsupported smoke variant: {variant}")
        result = _run(
            [str(self.binary), str(_elements(shape)), variant],
            cwd=self.source, env=self.env, timeout=180,
        )
        return _last_json(result.stdout)


def _status(
    worker_root: Path,
    assignment: WorkerAssignment,
    *,
    state: str,
    iteration: int,
    shape_id: str | None,
    probe: Dict[str, Any] | None = None,
    **details: Any,
) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "worker_id": assignment.worker_id,
        "state": state,
        "iteration": iteration,
        "shape_id": shape_id,
        "pid": os.getpid(),
        "physical_gpu": assignment.gpu,
        "logical_gpu": 0,
        "gpu_binding": {
            "HIP_VISIBLE_DEVICES": str(assignment.gpu),
            "strategy": "HIP_VISIBLE_DEVICES-only",
            "enforced": True,
            "visible_devices": (probe or {}).get("visible_devices"),
            "device_name": (probe or {}).get("device_name"),
        },
        "last_update": time.time(),
    }
    payload.update(details)
    write_json(worker_root / "status.json", payload)


class RealSmokeOptimizationPipeline:
    def __init__(
        self,
        *,
        req: Dict[str, Any],
        state_dir: Path,
        workspace_dir: Path,
        store: StateStore,
        manager: SubAgentManager,
    ) -> None:
        self.req = req
        self.state_dir = state_dir
        self.workspace_dir = workspace_dir
        self.store = store
        self.manager = manager
        self._current_phase = phases.PREPARE
        self._progress_lock = threading.Lock()
        self._reported_iteration = 0

    def _phase(self, phase: str, **payload: Any) -> None:
        self._current_phase = phase
        self.store.update_run(current_phase=phase)
        self.store.append_timeline("phase_start", {"phase": phase, **payload})

    def run(self, *, dry_run: bool = False) -> Dict[str, Any]:
        task_id = str(self.req.get("task_id", "task"))
        self.store.init_or_resume(task_id, "dcu-kernel-auto-opt")
        self.store.update_run(
            finished=False,
            final_status=None,
            last_outcome=None,
            last_transition_label=None,
            notes=[],
        )
        started = time.time()
        try:
            self._phase(phases.PREPARE)
            config = load_config(self.req)
            self._prepare_worktrees(config, task_id)
            plan = self._plan(config)
            write_json(self.workspace_dir / "plan.json", plan)
            if dry_run:
                return plan

            self._phase(phases.BASELINE)
            baseline = self._parallel_baseline(config)

            self._phase(phases.EXPLORE, workers=len(config.assignments))
            workers = self._parallel_agents(config, baseline)

            self._phase(phases.SYNTHESIZE)
            merged_skill = generate_merged_skill(
                config=config, assignments=config.assignments,
                workspace_dir=self.workspace_dir,
            )

            self._phase(phases.VALIDATE)
            validation = self._serial_validate(config, workers)

            self._phase(phases.REPORT)
            report = {
                "schema_version": SCHEMA_VERSION,
                "task_id": task_id,
                "task_type": "dcu-kernel-auto-opt",
                "mode": "real-agent-dcu-smoke",
                "started_at": started,
                "finished_at": time.time(),
                "duration_s": round(time.time() - started, 4),
                "config": plan,
                "baseline": baseline,
                "workers": workers,
                "merged_skill": merged_skill,
                "final_validation": validation,
                "real_gpu_used": True,
                "target_repo_modified": False,
                "status": "success",
            }
            write_json(self.workspace_dir / "final_report.json", report)
            self.store.update_run(
                current_iteration=config.mock_iterations,
                current_phase=phases.FINISHED,
                finished=True,
                final_status="success",
                last_outcome="ok",
                last_transition_label="real smoke complete",
            )
            self.store.append_timeline(
                "orchestrator_success",
                {"workers": len(workers), "real_gpu_used": True},
            )
            return report
        except Exception as exc:
            self.store.append_timeline(
                "orchestrator_error", {"error": repr(exc)}
            )
            self.store.update_run(
                # Keep the state machine on the phase that failed.  A stopped
                # task must not look like a successfully completed workflow.
                current_phase=self._current_phase, finished=True,
                final_status="stopped", last_outcome="infra_fail",
                notes=[str(exc)],
            )
            raise

    def _prepare_worktrees(
        self, config: OptimizerConfig, task_id: str
    ) -> None:
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        seed = self.workspace_dir / "main"
        if not (seed / ".git").exists():
            seed.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ASSET, seed / "smoke_harness.cpp")
            _run(["git", "init"], cwd=seed)
            _run(["git", "config", "user.name", "MetaInfer Agent"], cwd=seed)
            _run([
                "git", "config", "user.email", "metainfer@localhost"
            ], cwd=seed)
            _run(["git", "add", "smoke_harness.cpp"], cwd=seed)
            _run(["git", "commit", "-m", "seed trusted DCU smoke harness"], cwd=seed)
        for assignment in config.assignments:
            root = self.workspace_dir / "workers" / assignment.worker_id
            for name in ("build", "cache", "logs", "runs", "artifacts"):
                (root / name).mkdir(parents=True, exist_ok=True)
            source = root / "source"
            if not source.exists():
                branch = f"agent/{_safe(task_id)}/{assignment.worker_id}"
                _run([
                    "git", "worktree", "add", "-b", branch,
                    str(source), "HEAD",
                ], cwd=seed)
        for name in ("shared_baseline", "final_validation", "skills"):
            (self.workspace_dir / name).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _plan(config: OptimizerConfig) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "execution_mode": config.execution_mode,
            "operator": config.operator,
            "dtype": config.dtype,
            "hardware": config.hardware,
            "kernel_language": config.kernel_language,
            "claude_model": config.claude_model,
            "mock_iterations": config.mock_iterations,
            "minimum_improvement_percent": config.minimum_improvement_percent,
            "minimum_improvement_semantics": (
                "final validated result versus fixed baseline"
            ),
            "round_acceptance_improvement_percent": (
                ROUND_ACCEPTANCE_IMPROVEMENT_PERCENT
            ),
            "harness": "trusted built-in DCU vector smoke harness",
            "shapes": [
                {"id": shape.id, **shape.params}
                for shape in config.shapes.values()
            ],
            "assignments": [
                {"worker_id": item.worker_id, "gpu": item.gpu,
                 "shapes": item.shape_ids}
                for item in config.assignments
            ],
            "real_gpu_used": True,
        }

    def _parallel_baseline(
        self, config: OptimizerConfig
    ) -> Dict[str, Dict[str, Any]]:
        output: Dict[str, Dict[str, Any]] = {}

        def run_one(assignment: WorkerAssignment) -> Dict[str, Dict[str, Any]]:
            root = self.workspace_dir / "workers" / assignment.worker_id
            runner = SmokeRunner(root, assignment.gpu)
            _status(root, assignment, state="building", iteration=0, shape_id=None)
            runner.build()
            probe = runner.probe()
            if probe.get("visible_devices") != 1:
                raise RuntimeError(
                    f"{assignment.worker_id} sees "
                    f"{probe.get('visible_devices')} GPUs"
                )
            _status(
                root, assignment, state="baseline", iteration=0,
                shape_id=None, probe=probe,
            )
            return {
                shape_id: runner.benchmark(
                    config.shapes[shape_id].params, "scalar"
                )
                for shape_id in assignment.shape_ids
            }

        with ThreadPoolExecutor(max_workers=len(config.assignments)) as pool:
            futures = {
                pool.submit(run_one, item): item for item in config.assignments
            }
            for future in as_completed(futures):
                output.update(future.result())
        write_json(
            self.workspace_dir / "shared_baseline" / "results.json",
            {"schema_version": SCHEMA_VERSION, "shapes": output},
        )
        return output

    def _parallel_agents(
        self,
        config: OptimizerConfig,
        baseline: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        output: Dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=len(config.assignments)) as pool:
            futures = {
                pool.submit(
                    self._run_worker, config, assignment, baseline
                ): assignment
                for assignment in config.assignments
            }
            for future in as_completed(futures):
                assignment = futures[future]
                result = future.result()
                result["skill"] = generate_worker_skill(
                    config=config, assignment=assignment,
                    workspace_dir=self.workspace_dir,
                )
                output[assignment.worker_id] = result
                self.store.append_timeline(
                    "worker_complete",
                    {"worker_id": assignment.worker_id,
                     "skill": result["skill"]["name"]},
                )
        return dict(sorted(output.items()))

    def _run_worker(
        self,
        config: OptimizerConfig,
        assignment: WorkerAssignment,
        baseline: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        root = self.workspace_dir / "workers" / assignment.worker_id
        runner = SmokeRunner(root, assignment.gpu)
        probe = runner.probe()
        result: Dict[str, Any] = {
            "worker_id": assignment.worker_id,
            "physical_gpu": assignment.gpu,
            "branch": f"agent/{_safe(self.req.get('task_id', 'task'))}/"
                      f"{assignment.worker_id}",
            "worktree_created": True,
            "mode": "real-agent-dcu-smoke",
            "gpu_probe": probe,
            "shapes": {},
        }
        for shape_id in assignment.shape_ids:
            shape = config.shapes[shape_id]
            best_variant = "scalar"
            best_metrics = baseline[shape_id]
            experiments_path = root / "runs" / shape_id / "experiments.jsonl"
            for iteration in range(1, config.mock_iterations + 1):
                guidance = claim_next_guidance(
                    self.state_dir / "guidance",
                    assignment.worker_id,
                    iteration,
                )
                _status(
                    root, assignment, state="agent_running",
                    iteration=iteration, shape_id=shape_id, probe=probe,
                )
                proposal_path = root / "source" / "proposal.json"
                try:
                    proposal_path.unlink()
                except FileNotFoundError:
                    pass
                prompt = self._worker_prompt(
                    assignment, shape_id, shape.params, baseline[shape_id],
                    root, iteration, guidance,
                )
                prompt_file = root / "logs" / (
                    f"{shape_id}-iteration-{iteration}.prompt.txt"
                )
                prompt_file.write_text(prompt, encoding="utf-8")
                agent_name = (
                    f"{assignment.worker_id}-{shape_id}-iter{iteration}"
                )
                spec = AgentSpec(
                    name=agent_name,
                    role="dcu_kernel_worker",
                    prompt_file=prompt_file,
                    workdir=root / "source",
                    log_dir=root / "logs",
                    timeout_s=600,
                    stuck_timeout_s=240,
                    max_retries=0,
                    env_overrides=runner.env,
                )
                self.store.append_timeline(
                    "agent_launch",
                    {"name": agent_name, "worker_id": assignment.worker_id,
                     "physical_gpu": assignment.gpu},
                )
                self.manager.launch(spec)
                agent_result = self.manager.result(agent_name)
                if agent_result is None or not agent_result.success:
                    raise RuntimeError(
                        f"{agent_name} failed: "
                        f"{agent_result.error if agent_result else 'no result'}"
                    )
                proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
                variant = str(proposal.get("variant"))
                metrics = runner.benchmark(shape.params, variant)
                speedup = (
                    float(baseline[shape_id]["median_us"])
                    / float(metrics["median_us"])
                )
                round_improvement = (
                    float(best_metrics["median_us"])
                    / float(metrics["median_us"])
                    - 1.0
                ) * 100.0
                accepted = (
                    bool(metrics.get("passed"))
                    and float(metrics["median_us"])
                    < float(best_metrics["median_us"])
                    and round_improvement
                    >= ROUND_ACCEPTANCE_IMPROVEMENT_PERCENT
                )
                experiment = {
                    "schema_version": SCHEMA_VERSION,
                    "worker_id": assignment.worker_id,
                    "iteration": iteration,
                    "shape_id": shape_id,
                    "shape": shape.params,
                    "hypothesis": proposal.get("hypothesis"),
                    "changes": [f"select trusted smoke variant: {variant}"],
                    "profile_evidence": proposal.get("profile_evidence") or {},
                    "build_success": True,
                    "correctness_passed": bool(metrics.get("passed")),
                    "metrics": {
                        key: metrics[key] for key in (
                            "median_us", "p90_us", "min_us", "max_us",
                            "tflops", "bandwidth_gb_s",
                        )
                    },
                    "baseline_us": baseline[shape_id]["median_us"],
                    "speedup": round(speedup, 6),
                    "round_improvement_percent": round(
                        round_improvement, 6
                    ),
                    "accepted": accepted,
                    "commit": None,
                    "failure_reason": None,
                    "manual_guidance": (
                        guidance["text"] if guidance else None
                    ),
                    "guidance_id": guidance["id"] if guidance else None,
                    "agent_session_id": agent_result.session_id,
                    "timestamp": time.time(),
                }
                if accepted:
                    _run(["git", "add", "proposal.json"], cwd=root / "source")
                    _run([
                        "git", "commit", "-m",
                        f"{shape_id}: select {variant} in iteration {iteration}",
                    ], cwd=root / "source")
                    commit = _run(
                        ["git", "rev-parse", "HEAD"], cwd=root / "source"
                    ).stdout.strip()
                    experiment["commit"] = commit
                    best_variant = variant
                    best_metrics = metrics
                else:
                    tracked = _run(
                        ["git", "ls-files", "proposal.json"],
                        cwd=root / "source",
                    ).stdout.strip()
                    if tracked:
                        _run(
                            ["git", "restore", "proposal.json"],
                            cwd=root / "source",
                        )
                    else:
                        proposal_path.unlink(missing_ok=True)
                append_jsonl(experiments_path, experiment)
                # Surface live progress in the task header while workers run
                # concurrently.  StateStore serializes updates within this
                # orchestrator process; never move the displayed round back.
                with self._progress_lock:
                    if iteration > self._reported_iteration:
                        self.store.update_run(current_iteration=iteration)
                        self._reported_iteration = iteration
            result["shapes"][shape_id] = {
                "shape_id": shape_id,
                "variant": best_variant,
                "metrics": best_metrics,
            }
        _status(
            root, assignment, state="completed",
            iteration=config.mock_iterations, shape_id=None, probe=probe,
        )
        write_json(root / "result.json", result)
        return result

    @staticmethod
    def _worker_prompt(
        assignment: WorkerAssignment,
        shape_id: str,
        shape: Dict[str, Any],
        baseline: Dict[str, Any],
        root: Path,
        iteration: int,
        guidance: Dict[str, Any] | None,
    ) -> str:
        guidance_text = (
            guidance["text"] if guidance else "(none; decide independently)"
        )
        return f"""You are {assignment.worker_id}, an autonomous DCU smoke-tuning worker.
You are bound to physical GPU {assignment.gpu}; inside this process it must be
the only visible GPU and is logical device 0.

This is a real infrastructure smoke run, not the final operator integration.
Do not edit smoke_harness.cpp. Inspect it and make an evidence-based choice
between its scalar and vector4 candidates for shape {shape_id}: {json.dumps(shape)}.
Baseline: {json.dumps(baseline)}.
Human guidance for this round: {guidance_text}

First run `{root / 'build' / 'smoke_harness'} --probe`; stop if visible_devices
is not exactly 1. Then benchmark BOTH candidates with:
`{root / 'build' / 'smoke_harness'} {_elements(shape)} scalar`
`{root / 'build' / 'smoke_harness'} {_elements(shape)} vector4`

Write `{root / 'source' / 'proposal.json'}` as strict JSON:
{{
  "iteration": {iteration},
  "variant": "scalar or vector4",
  "hypothesis": "short evidence-based explanation",
  "profile_evidence": {{
    "scalar_median_us": 0.0,
    "vector4_median_us": 0.0,
    "visible_devices": 1,
    "device_name": "actual device"
  }}
}}
Do not choose before measuring both. Do not change correctness coverage or work.
"""

    def _serial_validate(
        self, config: OptimizerConfig, workers: Dict[str, Any]
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        for assignment in config.assignments:
            runner = SmokeRunner(
                self.workspace_dir / "workers" / assignment.worker_id,
                assignment.gpu,
            )
            for shape_id in assignment.shape_ids:
                winner = workers[assignment.worker_id]["shapes"][shape_id]
                metrics = runner.benchmark(
                    config.shapes[shape_id].params, winner["variant"]
                )
                if not metrics.get("passed"):
                    raise RuntimeError(
                        f"serial validation failed: {shape_id}"
                    )
                results[shape_id] = {
                    "passed": True,
                    "worker_id": assignment.worker_id,
                    "physical_gpu": assignment.gpu,
                    "candidate": winner["variant"],
                    "metrics": metrics,
                    "serial": True,
                }
        write_json(
            self.workspace_dir / "final_validation" / "results.json",
            {"schema_version": SCHEMA_VERSION, "shapes": results,
             "real_gpu_used": True},
        )
        return results


def _safe(value: Any) -> str:
    return "".join(
        char if char.isalnum() or char in "-_" else "-"
        for char in str(value)
    )[:48]
