"""Read-only task-specific Web routes."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request

from metainfer.server._helpers import (
    require_task_type,
    state_dir_for,
    task_or_404,
    workspace_dir_for,
)
from metainfer.server.state_reader import read_requirements, read_run

from ..orchestrator import phases
from ..orchestrator.config import (
    GEN_AND_OPT_MODE,
    LEGACY_SMOKE_MODE,
    SMOKE_MODE,
)
from ..orchestrator.guidance import add_guidance, list_guidance
from ..orchestrator.skill_store import list_skill_library, publish_skill


PLUGIN_TYPE = "dcu-kernel-auto-opt"


def _load(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _read_jsonl(path: Path) -> list[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _running_pid(path: Path) -> int | None:
    record = _load(path, {}) or {}
    try:
        pid = int(record.get("pid") or 0)
    except (TypeError, ValueError):
        return None
    if pid <= 0:
        return None
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return None
    return pid


def _read_bootstrap_attempts(
    worker_root: Path,
    state_dir: Path | None,
    worker_id: str,
    assigned_shapes: list[str],
) -> list[Dict[str, Any]]:
    if state_dir is None:
        return []
    snapshot = _load(state_dir / "agents.json", {}) or {}
    agents = snapshot.get("agents") or []
    snapshot_ts = float(snapshot.get("ts") or 0)
    snapshot_stale = snapshot_ts > 0 and time.time() - snapshot_ts > 30
    pattern = re.compile(
        rf"^{re.escape(worker_id)}-bootstrap-attempt(\d+)$"
    )
    matched: list[tuple[int, Dict[str, Any]]] = []
    for agent in agents:
        if not isinstance(agent, dict):
            continue
        match = pattern.match(str(agent.get("name") or ""))
        if match:
            matched.append((int(match.group(1)), agent))
    bootstrap_progress = _load(
        worker_root / "bootstrap_progress.json", {}
    ) or {}
    progress_attempt = int(bootstrap_progress.get("attempt") or 0)
    if not matched and progress_attempt:
        progress_status = str(
            bootstrap_progress.get("status") or "pending"
        )
        matched.append((progress_attempt, {
            "name": f"{worker_id}-bootstrap-attempt{progress_attempt}",
            "status": (
                "running"
                if progress_status in {"agent_running", "validating"}
                else progress_status
            ),
            "success": (
                True if progress_status == "passed"
                else False if progress_status == "failed"
                else None
            ),
        }))
    matched.sort(key=lambda item: item[0])
    if not matched:
        return []

    successful_workers = {
        str(event.get("payload", {}).get("worker_id"))
        for event in _read_jsonl(state_dir / "timeline.jsonl")
        if event.get("type") == "worker_bootstrap_success"
    }
    source = worker_root / "source"
    generated_files = [
        relative
        for relative in (
            "w8a8_backend.py",
            "setup.py",
            "csrc/bindings.cpp",
            "csrc/w8a8_gemm_hip.hip",
        )
        if (source / relative).is_file()
    ]
    proposal = _load(source / "proposal.json", {}) or {}
    bootstrap_result = _load(worker_root / "bootstrap_result.json", {}) or {}
    latest_attempt = matched[-1][0]
    attempts = []
    for attempt, agent in matched:
        raw_status = str(
            agent.get("status") or agent.get("phase") or "pending"
        )
        success = agent.get("success")
        error = agent.get("error")
        persisted_success = (
            int(bootstrap_result.get("attempt") or 0) == attempt
            and bool(bootstrap_result.get("passed"))
        )
        progress_matches = progress_attempt == attempt
        iteration_record = _load(
            worker_root / "iterations" / "bootstrap"
            / f"iteration{attempt}" / "iteration.json",
            {},
        ) or {}
        attempt_record = (
            iteration_record
            if iteration_record else
            bootstrap_progress
            if progress_matches else {}
        )
        progress_status = str(attempt_record.get("status") or "")
        if persisted_success:
            status = "passed"
        elif progress_status == "failed":
            status = "failed"
            error = attempt_record.get("error") or error
        elif progress_status == "passed":
            status = "passed"
        elif progress_status == "validating":
            status = "validating"
        elif progress_status == "agent_running" and raw_status == "running":
            status = "running"
        elif (
            raw_status == "running"
            and snapshot_stale
        ):
            status = "orphaned"
            error = error or (
                "The orchestrator stopped updating this agent; its last "
                "recorded process state is stale."
            )
        elif agent.get("killed") or raw_status == "failed" or success is False:
            status = "failed"
        elif (
            attempt == latest_attempt
            and worker_id in successful_workers
        ):
            status = "passed"
        elif raw_status == "running":
            status = "running"
        elif attempt < latest_attempt:
            status = "retrying"
            if not error:
                error = (
                    "Agent finished, but trusted validation requested "
                    "another bootstrap attempt."
                )
        elif success is True:
            status = "validating"
        else:
            status = raw_status
        hypothesis = (
            bootstrap_result.get("hypothesis")
            if persisted_success else proposal.get("hypothesis")
            if attempt == latest_attempt and isinstance(proposal, dict)
            else None
        )
        if (
            not persisted_success
            and attempt_record.get("hypothesis")
        ):
            hypothesis = attempt_record["hypothesis"]
        attempt_metrics = (
            bootstrap_result.get("metrics") or {}
            if persisted_success else
            attempt_record.get("metrics") or {}
        )
        attempts.append({
            "kind": "bootstrap",
            "attempt": attempt,
            "status": status,
            "hypothesis": hypothesis or (
                "Create and validate the initial HIP implementation for "
                f"{', '.join(assigned_shapes) or 'assigned shapes'}."
            ),
            "generated_files": generated_files,
            "metrics": attempt_metrics,
            "artifact_dir": attempt_record.get("artifact_dir"),
            "candidate_files": attempt_record.get("candidate_files") or [],
            "error": error,
            "elapsed_s": agent.get("elapsed_s"),
            "last_output_age_s": agent.get("last_output_age_s"),
            "started_at": agent.get("started_at"),
        })
    return attempts


def read_worker_lanes(
    workspace_dir: Path, state_dir: Path | None = None
) -> Dict[str, Any]:
    """Return exactly four worker lanes with their full iteration history."""
    plan = _load(workspace_dir / "plan.json", {}) or {}
    max_iterations = int(plan.get("max_iterations") or 0)
    assignment_by_worker = {
        str(item.get("worker_id")): item
        for item in (plan.get("assignments") or [])
        if isinstance(item, dict) and item.get("worker_id")
    }
    agent_by_worker: Dict[str, Dict[str, Any]] = {}
    if state_dir is not None:
        snapshot = _load(state_dir / "agents.json", {}) or {}
        for agent in snapshot.get("agents") or []:
            if not isinstance(agent, dict):
                continue
            name = str(agent.get("name") or "")
            match = re.match(
                r"^(worker_[0-3])-(?:.+-iter\d+|skill|bootstrap-attempt\d+)$",
                name,
            )
            if not match:
                continue
            worker_id = match.group(1)
            current = agent_by_worker.get(worker_id)
            if (
                current is None
                or float(agent.get("started_at") or 0)
                >= float(current.get("started_at") or 0)
            ):
                agent_by_worker[worker_id] = agent
    lanes = []
    for index in range(4):
        worker_id = f"worker_{index}"
        worker_root = workspace_dir / "workers" / worker_id
        assignment = assignment_by_worker.get(worker_id, {})
        assigned_shapes = assignment.get("shapes") or []
        status = _load(worker_root / "status.json", {}) or {}
        bootstrap_attempts = _read_bootstrap_attempts(
            worker_root, state_dir, worker_id, assigned_shapes
        )
        experiments = []
        runs_root = worker_root / "runs"
        if runs_root.exists():
            for path in sorted(runs_root.glob("*/experiments.jsonl")):
                experiments.extend(_read_jsonl(path))
        experiments.sort(
            key=lambda item: (
                float(item.get("timestamp") or 0),
                int(item.get("iteration") or 0),
            )
        )
        lane_state = status.get("state")
        if not lane_state and bootstrap_attempts:
            lane_state = f"bootstrap_{bootstrap_attempts[-1]['status']}"
        current_agent = agent_by_worker.get(worker_id, {})
        agent_status = str(current_agent.get("status") or "")
        last_output_age = current_agent.get("last_output_age_s")
        if lane_state == "building":
            step = "Building trusted baseline"
        elif lane_state == "baseline":
            step = "Benchmarking trusted baseline"
        elif str(lane_state).startswith("bootstrap_"):
            bootstrap_state = str(lane_state).removeprefix("bootstrap_")
            step = {
                "running": "Bootstrap Agent generating initial kernel",
                "agent_running": "Child Agent generating initial HIP kernel",
                "validating": "Validating initial kernel",
                "retrying": "Retrying initial kernel generation",
                "passed": "Initial kernel validated",
                "failed": "Initial kernel unavailable; lane will be skipped",
                "orphaned": "Bootstrap Agent stopped responding",
            }.get(bootstrap_state, "Preparing initial kernel")
        elif lane_state == "agent_running":
            step = "Agent planning and editing kernel"
        elif lane_state == "profiling_current_best":
            step = "Profiling current best kernel with PMC"
        elif lane_state == "repairing_candidate":
            repair = int(status.get("repair") or 0)
            max_repairs = int(status.get("max_repairs") or 4)
            step = (
                "Repairing compile/correctness failure "
                f"({repair}/{max_repairs})"
            )
        elif lane_state == "validating_candidate":
            step = "Compiling and validating candidate"
        elif lane_state == "recording_result":
            step = "Recording round result"
        elif lane_state == "optimization_complete":
            step = "Optimization rounds complete"
        elif lane_state == "skill_writing":
            step = "Writing worker optimization skill"
        elif lane_state == "completed":
            step = "Worker skill ready"
        elif lane_state in {"failed", "timed_out", "skipped"}:
            step = "Lane unavailable; main Agent will skip it"
        elif not assignment:
            step = "No shapes assigned"
        else:
            step = "Waiting to start"
        if agent_status in {"failed", "orphaned"} and lane_state == "agent_running":
            step = "Agent failed; waiting for lane fallback"
        completed_rounds = len(experiments)
        target_rounds = max_iterations * max(1, len(assigned_shapes))
        current_iteration = int(status.get("iteration") or 0)
        current_shape = status.get("shape_id")
        active_states = {
            "profiling_current_best",
            "agent_running",
            "validating_candidate",
            "repairing_candidate",
            "recording_result",
        }
        already_recorded = any(
            int(item.get("iteration") or 0) == current_iteration
            and item.get("shape_id") == current_shape
            for item in experiments
        )
        active_iteration = None
        if (
            current_iteration > 0
            and lane_state in active_states
            and not already_recorded
        ):
            active_iteration = {
                "iteration": current_iteration,
                "shape_id": current_shape,
                "state": lane_state,
                "step": step,
                "agent_name": current_agent.get("name"),
                "agent_status": current_agent.get("status"),
                "elapsed_s": current_agent.get("elapsed_s"),
                "last_output_age_s": last_output_age,
            }
            if lane_state == "repairing_candidate":
                active_iteration["repair"] = int(
                    status.get("repair") or 0
                )
                active_iteration["max_repairs"] = int(
                    status.get("max_repairs") or 4
                )
        lanes.append({
            "worker_id": worker_id,
            "gpu": assignment.get("gpu", index),
            "assigned": bool(assignment),
            "assigned_shapes": assigned_shapes,
            "state": lane_state or (
                "not_assigned" if not assignment else "pending"
            ),
            "current_iteration": current_iteration,
            "current_shape": current_shape,
            "step": step,
            "completed_rounds": completed_rounds,
            "target_rounds": target_rounds,
            "agent": {
                "name": current_agent.get("name"),
                "status": current_agent.get("status"),
                "elapsed_s": current_agent.get("elapsed_s"),
                "last_output_age_s": last_output_age,
                "error": current_agent.get("error"),
            } if current_agent else None,
            "long_running": (
                isinstance(last_output_age, (int, float))
                and last_output_age >= 180
                and agent_status == "running"
            ),
            "bootstrap_attempts": bootstrap_attempts,
            "experiments": experiments,
            "active_iteration": active_iteration,
            "latest": experiments[-1] if experiments else None,
            "guidance": (
                list_guidance(state_dir / "guidance", worker_id)
                if state_dir is not None else []
            ),
        })
    return {"workers": lanes}


def build_router(plugin) -> APIRouter:
    router = APIRouter()

    @router.get("/summary")
    def summary(task_id: str) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        state_dir = state_dir_for(entry)
        workspace_dir = workspace_dir_for(entry)
        workers = []
        workers_root = workspace_dir / "workers"
        if workers_root.exists():
            for status_path in sorted(workers_root.glob("*/status.json")):
                workers.append(_load(status_path, {}))
        return {
            "run": read_run(state_dir),
            "plan": _load(workspace_dir / "plan.json", None),
            "workers": workers,
            "report": _load(workspace_dir / "final_report.json", None),
        }

    @router.get("/iterations")
    def iterations(task_id: str) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        return read_worker_lanes(workspace_dir_for(entry), state_dir_for(entry))

    @router.post("/workers/{worker_id}/guidance")
    async def submit_guidance(
        task_id: str, worker_id: str, request: Request
    ) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        plan = _load(workspace_dir_for(entry) / "plan.json", {}) or {}
        assigned = {
            str(item.get("worker_id"))
            for item in (plan.get("assignments") or [])
            if isinstance(item, dict)
        }
        if worker_id not in assigned:
            raise HTTPException(status_code=400, detail="worker is not assigned")
        try:
            body = await request.json()
            text = body.get("text", "") if isinstance(body, dict) else ""
            guidance = add_guidance(
                state_dir_for(entry) / "guidance", worker_id, str(text)
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"guidance": guidance}

    @router.post("/workers/{worker_id}/restart")
    def restart_worker(task_id: str, worker_id: str) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        if not re.fullmatch(r"worker_[0-3]", worker_id):
            raise HTTPException(status_code=400, detail="invalid worker id")
        state_dir = state_dir_for(entry)
        workspace_dir = workspace_dir_for(entry)
        plan = _load(workspace_dir / "plan.json", {}) or {}
        assignment = next((
            item for item in (plan.get("assignments") or [])
            if isinstance(item, dict) and item.get("worker_id") == worker_id
        ), None)
        if assignment is None:
            raise HTTPException(status_code=400, detail="worker is not assigned")
        worker_root = workspace_dir / "workers" / worker_id
        status = _load(worker_root / "status.json", {}) or {}
        if status.get("state") not in {"failed", "timed_out"}:
            raise HTTPException(
                status_code=409,
                detail="only a failed or timed-out worker can be restarted",
            )
        pid_path = worker_root / "restart.pid.json"
        active_pid = _running_pid(pid_path)
        if active_pid is not None:
            raise HTTPException(
                status_code=409,
                detail=f"worker restart is already running as pid {active_pid}",
            )
        requirements = state_dir / "requirements.json"
        if not requirements.is_file():
            raise HTTPException(status_code=400, detail="requirements.json is missing")
        bridge = os.environ.get("METAINFER_CLAUDE_BIN") or str(
            Path(__file__).resolve().parent.parent
            / "bridge" / "agent_bridge_client.py"
        )
        log_path = state_dir / f"{worker_id}_restart.log"
        command = [
            sys.executable,
            "-m",
            "metainfer.tasks.dcu_kernel_auto_opt.orchestrator.restart_worker",
            str(requirements),
            "--state-dir", str(state_dir),
            "--workspace-dir", str(workspace_dir),
            "--worker-id", worker_id,
            "--claude-bin", bridge,
        ]
        integration_command = [
            sys.executable,
            "-m",
            (
                "metainfer.tasks.dcu_kernel_auto_opt.orchestrator."
                "integrate_restarted_worker"
            ),
            str(requirements),
            "--state-dir", str(state_dir),
            "--workspace-dir", str(workspace_dir),
            "--worker-id", worker_id,
            "--claude-bin", bridge,
        ]
        try:
            with log_path.open("ab") as log_file:
                process = subprocess.Popen(
                    command,
                    cwd=Path(__file__).resolve().parents[4],
                    stdin=subprocess.DEVNULL,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    close_fds=True,
                )
                integration_process = subprocess.Popen(
                    integration_command,
                    cwd=Path(__file__).resolve().parents[4],
                    stdin=subprocess.DEVNULL,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    close_fds=True,
                )
        except OSError as exc:
            raise HTTPException(
                status_code=500, detail=f"failed to start worker: {exc}"
            ) from exc
        pid_path.write_text(json.dumps({
            "pid": process.pid,
            "task_id": task_id,
            "worker_id": worker_id,
            "physical_gpu": assignment.get("gpu"),
            "started_at": time.time(),
            "integration_pid": integration_process.pid,
        }, indent=2), encoding="utf-8")
        return {
            "ok": True,
            "worker_id": worker_id,
            "physical_gpu": assignment.get("gpu"),
            "pid": process.pid,
            "integration_pid": integration_process.pid,
            "isolated": True,
        }

    @router.get("/state-graph")
    def state_graph(task_id: str) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        state_dir = state_dir_for(entry)
        run = read_run(state_dir)
        requirements = read_requirements(state_dir) or {}
        answers = requirements.get("answers")
        source = answers if isinstance(answers, dict) else requirements
        return phases.graph_payload(
            run.get("current_phase", phases.PREPARE),
            run.get("last_outcome"),
            run.get("last_transition_label"),
            include_baseline=(
                source.get("execution_mode") != GEN_AND_OPT_MODE
            ),
        )

    @router.get("/skills")
    def skills(task_id: str) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        workspace_dir = workspace_dir_for(entry)
        library = list_skill_library(workspace_dir)
        plan = _load(workspace_dir / "plan.json", {}) or {}
        if plan.get("execution_mode") in {LEGACY_SMOKE_MODE, SMOKE_MODE}:
            quarantined = library.get("pending") or []
            library["pending"] = []
            library["quarantined_count"] = len(quarantined)
            library["publish_disabled_reason"] = (
                "Infrastructure-smoke findings are unrelated to the selected "
                "operator and cannot be published as optimization skills."
            )
        return library

    @router.post("/skills/{skill_name}/publish")
    def publish(task_id: str, skill_name: str) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        workspace_dir = workspace_dir_for(entry)
        plan = _load(workspace_dir / "plan.json", {}) or {}
        if plan.get("execution_mode") in {LEGACY_SMOKE_MODE, SMOKE_MODE}:
            raise HTTPException(
                status_code=400,
                detail=(
                    "infrastructure-smoke skills are quarantined and cannot "
                    "be published"
                ),
            )
        try:
            return {
                "skill": publish_skill(workspace_dir, skill_name)
            }
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except FileExistsError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    return router
