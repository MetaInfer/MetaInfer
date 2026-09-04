"""Unified multi-GPU orchestrator — ONE task, N GPU workers.

The MultiGpuOrchestrator spawns one standard evolve-kernel orchestrator
subprocess per GPU, each pinned to ``CUDA_VISIBLE_DEVICES=N`` and
assigned a subset of target shapes. All state lives under a single
``state_dir`` / ``workspace_dir`` pair so the WebUI sees ONE task with
a unified multi-GPU dashboard.

Lifecycle:
  1. Parse shapes → split across GPUs
  2. Write per-GPU requirements under ``state_dir/gpu_N/``
  3. Spawn subprocesses (``python -m ...cli run ... --gpu-device N``)
  4. Poll per-GPU ``run.json`` → update parent ``run.json``
  5. When all workers finish: merge kernel_library.json + shape_bench.json
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from metainfer.tasks.evolve_kernel.server._multi_gpu import (
    detect_gpus,
    split_shapes_for_gpus,
    shapes_to_notes,
)


# =========================================================================== #
# Config
# =========================================================================== #


@dataclass
class GpuWorkerConfig:
    gpu_idx: int
    label: str  # "GPU 0"
    shapes: list  # List[ShapeSpec]
    state_dir: Path
    workspace_dir: Path
    requirements_path: Path


# =========================================================================== #
# MultiGpuOrchestrator
# =========================================================================== #


class MultiGpuOrchestrator:
    """Spawns and monitors N parallel GPU kernel-optimization workers."""

    def __init__(
        self,
        req: Dict[str, Any],
        state_dir: Path,
        workspace_dir: Path,
        claude_bin: str = "ccb",
        model: Optional[str] = None,
        permission_mode: str = "bypassPermissions",
        effort: str = "max",
    ) -> None:
        self.req = req
        self.state_dir = state_dir
        self.workspace_dir = workspace_dir
        self.claude_bin = claude_bin
        self.model = model
        self.permission_mode = permission_mode
        self.effort = effort

        num_gpus_str = req.get("gpu_count", "")
        try:
            self.num_gpus = int(num_gpus_str) if num_gpus_str else detect_gpus()
        except (ValueError, TypeError):
            self.num_gpus = detect_gpus()
        self.num_gpus = max(1, min(self.num_gpus, 8))

        self.worker_procs: list[Tuple[GpuWorkerConfig, subprocess.Popen]] = []
        self._start_ts = time.time()

    # ------------------------------------------------------------------ #
    # Public entry
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        """Main orchestration: setup → spawn → monitor → aggregate → done."""
        # 1. Setup
        workers = self._prepare_workers()

        # Write parent run.json with multi-gpu metadata
        self._write_parent_run({"gpu_count": len(workers), "phase": "starting"})

        # 2. Spawn
        self._spawn_all(workers)

        # Refresh PID immediately after spawn — closes the gap between
        # the initial write and the first monitor loop iteration.
        self._refresh_pid_file()

        # 3. Monitor
        self._monitor_loop()

        # 4. Check results
        self._check_failures()

        # 5. Aggregate
        self._merge_results()

        # 6. Summary report
        summary = self._generate_summary()

        # 7. Done
        self._write_parent_run({
            "phase": "finished",
            "finished": True,
            "final_status": "success",
            "summary": summary,
        })

        # Print summary to orchestrator log for quick inspection
        print("\n" + "=" * 60)
        print("TASK COMPLETE — Summary Report")
        print("=" * 60)
        print(summary)
        print("=" * 60)

    # ------------------------------------------------------------------ #
    # Worker setup
    # ------------------------------------------------------------------ #

    def _prepare_workers(self) -> List[GpuWorkerConfig]:
        """Split shapes and create per-GPU worker configs."""
        extra_notes = self.req.get("extra_notes", "")
        groups = split_shapes_for_gpus(extra_notes, self.num_gpus)

        workers: List[GpuWorkerConfig] = []
        for gpu_label, shapes in groups:
            gpu_idx = int(gpu_label.replace("GPU ", ""))
            gpu_state = self.state_dir / f"gpu_{gpu_idx}"
            gpu_workspace = self.workspace_dir / f"gpu_{gpu_idx}"

            gpu_state.mkdir(parents=True, exist_ok=True)
            gpu_workspace.mkdir(parents=True, exist_ok=True)

            # Build per-GPU requirements (copy parent, override)
            gpu_req = dict(self.req)
            gpu_req.pop("multi_gpu", None)
            gpu_req.pop("gpu_count", None)
            gpu_req["gpu_device"] = str(gpu_idx)
            gpu_req["extra_notes"] = (
                "{}\nGPU={} shapes:\n{}".format(
                    extra_notes.split("\n")[0] if extra_notes else "",
                    gpu_label,
                    shapes_to_notes(shapes),
                )
            )

            req_path = gpu_state / "requirements.json"
            req_path.write_text(json.dumps(gpu_req, indent=2), encoding="utf-8")

            # Copy reference kernel to per-GPU workspace
            ref_src = self.workspace_dir / "reference" / "original_kernel.py"
            if ref_src.is_file():
                ref_dst = gpu_workspace / "reference"
                ref_dst.mkdir(parents=True, exist_ok=True)
                (ref_dst / "original_kernel.py").write_text(
                    ref_src.read_text(encoding="utf-8"),
                    encoding="utf-8",
                )

            workers.append(GpuWorkerConfig(
                gpu_idx=gpu_idx,
                label=gpu_label,
                shapes=shapes,
                state_dir=gpu_state,
                workspace_dir=gpu_workspace,
                requirements_path=req_path,
            ))

        return workers

    # ------------------------------------------------------------------ #
    # Spawn
    # ------------------------------------------------------------------ #

    def _spawn_all(self, workers: List[GpuWorkerConfig]) -> None:
        """Launch one orchestrator subprocess per GPU."""
        for w in workers:
            log_path = w.state_dir / "orchestrator.log"
            log_fp = open(str(log_path), "ab", buffering=0)

            # Resolve python executable — avoid the pip entry-point wrapper
            # which points to the wrong module path.
            python_bin = os.environ.get("METAINFER_PYTHON", sys.executable)
            # If sys.executable is the metainfer-orchestrator entry point,
            # fall back to the standard python3 binary.
            if "orchestrator" in python_bin.lower():
                import shutil
                fallback = shutil.which("python3") or "/usr/bin/python3"
                python_bin = fallback

            cmd = [
                python_bin,
                "-m", "metainfer.tasks.evolve_kernel.orchestrator.cli",
                "run", str(w.requirements_path),
                "--state-dir", str(w.state_dir),
                "--workspace-dir", str(w.workspace_dir),
                "--gpu-device", str(w.gpu_idx),
                "--claude-bin", self.claude_bin,
                "--permission-mode", self.permission_mode,
                "--effort", self.effort,
            ]
            if self.model:
                cmd += ["--model", self.model]

            # Set PYTHONPATH so the worker can import metainfer
            worker_env = dict(os.environ)
            python_path = os.pathsep.join(
                p for p in sys.path
                if p and p not in worker_env.get("PYTHONPATH", "").split(os.pathsep)
            )
            if python_path:
                existing = worker_env.get("PYTHONPATH", "")
                worker_env["PYTHONPATH"] = (
                    f"{python_path}{os.pathsep}{existing}".rstrip(os.pathsep)
                    if existing else python_path
                )

            proc = subprocess.Popen(
                cmd,
                stdout=log_fp,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                cwd=str(w.state_dir),
                env=worker_env,
                start_new_session=True,
            )
            log_fp.close()

            # Write PID file for status monitoring
            pid_data = {
                "pid": proc.pid,
                "task_id": self.req.get("task_id", ""),
                "started_at": time.time(),
            }
            (w.state_dir / "orchestrator.pid").write_text(
                json.dumps(pid_data, indent=2), encoding="utf-8",
            )

            self.worker_procs.append((w, proc))
            print(f"[multi-gpu] spawned GPU {w.gpu_idx} worker (PID={proc.pid})")

    # ------------------------------------------------------------------ #
    # Monitor loop
    # ------------------------------------------------------------------ #

    def _monitor_loop(self) -> None:
        """Poll per-GPU run.json and update parent state until all finish."""
        poll_s = 3.0
        _crashed_handled: set = set()
        while True:
            all_done = True
            gpu_statuses: List[Dict[str, Any]] = []

            for w, proc in self.worker_procs:
                # Detect crashed/zombie workers: if process exited but
                # run.json was never updated to "finished", mark as crashed.
                if w.gpu_idx not in _crashed_handled and proc.poll() is not None:
                    _crashed_handled.add(w.gpu_idx)
                    run_path = w.state_dir / "run.json"
                    if run_path.is_file():
                        try:
                            run = json.loads(run_path.read_text(encoding="utf-8"))
                            if run.get("current_phase") not in ("finished", "crashed"):
                                exit_code = proc.poll()
                                print(f"[multi-gpu] GPU {w.gpu_idx} worker exited with code {exit_code} "
                                      f"at phase {run.get('current_phase')} — marking as crashed")
                                run["finished"] = True
                                run["final_status"] = "crashed"
                                run["crash_exit_code"] = exit_code
                                run_path.write_text(json.dumps(run, indent=2), encoding="utf-8")
                        except Exception:
                            pass

                status = self._read_gpu_status(w)
                gpu_statuses.append(status)

                if status["phase"] not in ("finished", "crashed", ""):
                    all_done = False

            # Update parent run.json with live status
            self._write_parent_run({
                "phase": "running",
                "gpu_status": gpu_statuses,
                "gpu_count": len(self.worker_procs),
            })

            # Refresh parent orchestrator.pid every cycle so the liveness
            # scanner never sees a stale PID file. The liveness checker
            # runs every 10s in the WebUI and will stamp finished_at on any
            # PID file it misdiagnoses as dead.
            self._refresh_pid_file()

            if all_done:
                print("[multi-gpu] all workers finished")
                break

            time.sleep(poll_s)

        # Wait for processes to actually exit
        for w, proc in self.worker_procs:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    def _refresh_pid_file(self) -> None:
        """Rewrite parent orchestrator.pid to keep liveness scanner happy.

        Preserves the original started_at from the launcher's placeholder
        (which matches /proc/<pid>/stat) — overwriting it causes the
        liveness scanner to falsely detect PID reuse.
        """
        import json as _json
        pid_path = self.state_dir / "orchestrator.pid"

        # Read existing started_at (written by launcher at spawn time)
        started_at = None
        my_pid = os.getpid()
        if pid_path.is_file():
            try:
                prev = _json.loads(pid_path.read_text(encoding="utf-8"))
                if prev.get("pid") == my_pid and prev.get("started_at"):
                    started_at = prev["started_at"]
            except Exception:
                pass

        payload = {
            "pid": my_pid,
            "task_id": self.req.get("task_id", ""),
            "started_at": started_at or self._start_ts,
        }
        try:
            tmp = pid_path.with_suffix(".tmp")
            tmp.write_text(_json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(pid_path)
        except OSError:
            pass

    def _read_gpu_status(self, w: GpuWorkerConfig) -> Dict[str, Any]:
        """Read one GPU worker's current state from disk."""
        run_path = w.state_dir / "run.json"
        if not run_path.is_file():
            return {
                "gpu_idx": w.gpu_idx,
                "label": w.label,
                "phase": "starting",
                "iteration": 0,
                "exec_time_ms": 0,
                "speedup": 0,
                "running": True,
                "pid": 0,
            }

        try:
            run = json.loads(run_path.read_text(encoding="utf-8"))
        except Exception:
            return {"gpu_idx": w.gpu_idx, "label": w.label, "phase": "error"}

        # Check if process is alive
        pid_file = w.state_dir / "orchestrator.pid"
        running = False
        pid = 0
        if pid_file.is_file():
            try:
                pid_data = json.loads(pid_file.read_text(encoding="utf-8"))
                pid = pid_data.get("pid", 0)
                if pid and pid_data.get("finished_at") is None:
                    # Verify process is alive
                    try:
                        os.kill(pid, 0)
                        running = True
                    except OSError:
                        running = False
            except Exception:
                pass

        # Read phase/iteration from run.json
        phase = run.get("current_phase", "idle")
        iteration = run.get("current_iteration", 0)

        # Read best kernel from per-GPU library
        exec_time_ms = 0.0
        speedup = 0.0
        lib_path = w.workspace_dir / "kernel_library.json"
        if lib_path.is_file():
            try:
                lib = json.loads(lib_path.read_text(encoding="utf-8"))
                if lib:
                    lib.sort(key=lambda k: k.get("exec_time_ms", float("inf")))
                    best = lib[0]
                    exec_time_ms = best.get("exec_time_ms", 0)
                    # Speedup vs seed (exec_time of iteration_added=0)
                    for k in lib:
                        if k.get("iteration_added") == 0:
                            seed_time = k.get("exec_time_ms", 0)
                            if seed_time > 0 and exec_time_ms > 0:
                                speedup = seed_time / exec_time_ms
                            break
            except Exception:
                pass

        return {
            "gpu_idx": w.gpu_idx,
            "label": w.label,
            "phase": phase,
            "iteration": iteration,
            "exec_time_ms": round(exec_time_ms, 4) if exec_time_ms else 0,
            "speedup": round(speedup, 2) if speedup else 0,
            "running": running,
            "pid": pid,
        }

    # ------------------------------------------------------------------ #
    # Result aggregation
    # ------------------------------------------------------------------ #

    def _check_failures(self) -> None:
        """Log which workers crashed."""
        for w, _proc in self.worker_procs:
            run_path = w.state_dir / "run.json"
            if run_path.is_file():
                try:
                    run = json.loads(run_path.read_text(encoding="utf-8"))
                    if run.get("final_status") == "crashed":
                        print(f"[multi-gpu] GPU {w.gpu_idx} ended with crash")
                except Exception:
                    pass

    def _merge_results(self) -> None:
        """Merge all per-GPU kernel libraries and shape benchmarks into parent."""
        # Merge kernel libraries
        all_kernels: List[Dict[str, Any]] = []
        for w, _proc in self.worker_procs:
            lib_path = w.workspace_dir / "kernel_library.json"
            if lib_path.is_file():
                try:
                    lib = json.loads(lib_path.read_text(encoding="utf-8"))
                    for k in lib:
                        k["gpu_id"] = w.gpu_idx
                        k["gpu_label"] = w.label
                    all_kernels.extend(lib)
                except Exception:
                    pass

        all_kernels.sort(key=lambda k: k.get("exec_time_ms", float("inf")))
        parent_lib = self.workspace_dir / "kernel_library.json"
        parent_lib.write_text(json.dumps(all_kernels, indent=2, ensure_ascii=False),
                             encoding="utf-8")

        # Merge shape benchmarks
        all_bench: List[Dict[str, Any]] = []
        for w, _proc in self.worker_procs:
            bench_path = w.workspace_dir / "shape_bench.json"
            if bench_path.is_file():
                try:
                    data = json.loads(bench_path.read_text(encoding="utf-8"))
                    gpu_source = f"GPU{w.gpu_idx}"
                    for r in data.get("results", []):
                        r["gpu_source"] = gpu_source
                    all_bench.extend(data.get("results", []))
                except Exception:
                    pass

        if all_bench:
            parent_bench = self.workspace_dir / "shape_bench.json"
            parent_bench.write_text(json.dumps({
                "results": all_bench,
                "best_kernel_id": "merged",
                "cached": False,
            }, indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"[multi-gpu] merged {len(all_kernels)} kernels, {len(all_bench)} benchmarks")

    # ------------------------------------------------------------------ #
    # Parent run.json
    # ------------------------------------------------------------------ #

    def _generate_summary(self) -> str:
        """Generate a human-readable summary report after all workers finish.

        Writes the summary to ``summary.txt`` in the state dir and returns
        the markdown-formatted report string.
        """
        import time as _time

        elapsed_s = _time.time() - self._start_ts
        elapsed_str = f"{elapsed_s:.0f}s"
        if elapsed_s > 3600:
            elapsed_str = f"{elapsed_s / 3600:.1f}h"
        elif elapsed_s > 60:
            elapsed_str = f"{elapsed_s / 60:.1f}min"

        lines = []
        lines.append("")
        lines.append(f"## Task: {self.req.get('label', self.req.get('task_id', '?'))}")
        lines.append(f"**Elapsed:** {elapsed_str}  |  **Mode:** {self.req.get('optimizer_mode', 'Triton')}  |  **GPUs:** {self.num_gpus}")
        lines.append(f"**Max iterations:** {self.req.get('max_iterations', '?')}  |  **Profiling:** {self.req.get('enable_profiling', 'No')}")
        lines.append("")
        lines.append("| GPU | Shape Task | Seed (ms) | Best (ms) | Speedup | Kernels | Stop Reason |")
        lines.append("|-----|-----------|----------|----------|---------|---------|-------------|")

        all_ok = True
        for w, _proc in self.worker_procs:
            run_path = w.state_dir / "run.json"
            lib_path = w.workspace_dir / "kernel_library.json"

            stop_reason = "unknown"
            final_status = "?"
            if run_path.is_file():
                try:
                    run = json.loads(run_path.read_text(encoding="utf-8"))
                    final_status = run.get("final_status", "?")
                    if run.get("crash_exit_code"):
                        stop_reason = f"crashed (exit {run['crash_exit_code']})"
                        all_ok = False
                    elif final_status == "success":
                        stop_reason = "converged / max iters"
                    elif final_status == "crashed":
                        stop_reason = f"crashed: {run.get('crash_reason', '?')[:60]}"
                        all_ok = False
                except Exception:
                    pass

            seed_ms = 0.0
            best_ms = 0.0
            speedup = 0.0
            kernel_count = 0
            if lib_path.is_file():
                try:
                    lib = json.loads(lib_path.read_text(encoding="utf-8"))
                    kernel_count = len(lib)
                    if lib:
                        best = min(lib, key=lambda k: k.get("exec_time_ms", float("inf")))
                        best_ms = best.get("exec_time_ms", 0)
                        for k in lib:
                            if k.get("iteration_added") == 0:
                                seed_ms = k.get("exec_time_ms", 0)
                                break
                        if seed_ms > 0 and best_ms > 0:
                            speedup = seed_ms / best_ms
                except Exception:
                    pass

            # Extract shape task summary from requirements
            shapes_notes = ""
            req_path = w.state_dir / "requirements.json"
            if req_path.is_file():
                try:
                    gpu_req = json.loads(req_path.read_text(encoding="utf-8"))
                    extra = gpu_req.get("extra_notes", "")
                    # Extract first line that looks like a shape description
                    for line in extra.split("\n"):
                        if "(" in line and "@" in line and ("M=" in line or "TP=" in line):
                            shapes_notes = line.strip()[:60]
                            break
                except Exception:
                    pass

            su_str = f"{speedup:.2f}×" if speedup > 0 else "—"
            seed_str = f"{seed_ms:.4f}" if seed_ms > 0 else "—"
            best_str = f"{best_ms:.4f}" if best_ms > 0 else "—"

            lines.append(
                f"| GPU {w.gpu_idx} | {shapes_notes} | {seed_str}ms | {best_str}ms | "
                f"{su_str} | {kernel_count} | {stop_reason} |"
            )

        lines.append("")

        # Warnings
        if not all_ok:
            lines.append("### ⚠ Warnings")
            lines.append("Some workers did not finish cleanly. Check per-GPU orchestrator logs for details.")
            lines.append("")

        # Export paths
        exp_dir = self.workspace_dir / "optimized_kernels"
        if exp_dir.is_dir():
            cpp_count = len(list(exp_dir.glob("*.cpp")))
            py_count = len(list(exp_dir.glob("*.py")))
            lines.append(f"### Exported Kernels")
            lines.append(f"`{exp_dir}` — {py_count} `.py` wrappers + {cpp_count} `.cpp` sources")
            lines.append("")

        report = "\n".join(lines)

        # Write to summary.txt
        summary_path = self.state_dir / "summary.txt"
        try:
            summary_path.write_text(report, encoding="utf-8")
        except Exception:
            pass

        return report

    def _write_parent_run(self, updates: Dict[str, Any]) -> None:
        """Update parent run.json with current multi-GPU state.

        Always ensures ``finished``, ``final_status`` reflect the real state.
        Stale ``finished: true`` from a prior crash/restart is reset.
        """
        run_path = self.state_dir / "run.json"
        current: Dict[str, Any] = {
            "task_id": self.req.get("task_id", ""),
            "current_iteration": 0,
            "current_phase": "running",
            "finished": False,
            "final_status": None,
            "multi_gpu": True,
            "last_update": time.time(),
        }
        if run_path.is_file():
            try:
                loaded = json.loads(run_path.read_text(encoding="utf-8"))
                # Merge but NEVER carry over stale finished/stopped
                for k, v in loaded.items():
                    if k in ("finished", "final_status"):
                        continue
                    if k not in current:
                        current[k] = v
            except Exception:
                pass
        # Apply latest updates (including override of phase, gpu_status, etc.)
        current.update(updates)
        current["last_update"] = time.time()
        # Only set finished if explicitly passed True
        if updates.get("finished") is not True:
            current["finished"] = False
        run_path.write_text(json.dumps(current, indent=2), encoding="utf-8")
