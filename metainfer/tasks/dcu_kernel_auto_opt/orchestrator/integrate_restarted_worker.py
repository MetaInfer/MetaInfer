"""Integrate a completed restarted lane after the original orchestrator exits."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from metainfer.orchestrator._bootstrap import make_subagent_manager
from metainfer.orchestrator.state import StateStore

from .config import load_config
from .gen_and_opt_pipeline import GenAndOptPipeline
from .result_store import write_json
from .w8a8_pipeline import evaluate_final_target


def _load(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _orchestrator_running(state_dir: Path) -> bool:
    record = _load(state_dir / "orchestrator.pid")
    try:
        pid = int(record.get("pid") or 0)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("requirements", type=Path)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--workspace-dir", type=Path, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--claude-bin", required=True)
    parser.add_argument("--timeout", type=int, default=43200)
    args = parser.parse_args()

    req = _load(args.requirements)
    config = load_config(req)
    assignment = next(
        item for item in config.assignments
        if item.worker_id == args.worker_id
    )
    worker_result_path = (
        args.workspace_dir / "workers" / args.worker_id
        / "restart_result.json"
    )
    final_report_path = args.workspace_dir / "final_report.json"
    deadline = time.time() + args.timeout
    while time.time() < deadline:
        worker_result = _load(worker_result_path)
        report = _load(final_report_path)
        if (
            worker_result.get("status") == "completed"
            and report
            and not _orchestrator_running(args.state_dir)
        ):
            break
        time.sleep(5)
    else:
        raise TimeoutError("timed out waiting for worker and original task")

    workers = dict(report.get("workers") or {})
    workers.update(worker_result.get("workers") or {})
    initial_metrics = dict(report.get("initial_metrics") or {})
    initial_metrics.update(worker_result.get("baseline") or {})
    failures = dict(report.get("worker_failures") or {})
    failures.pop(args.worker_id, None)

    os.environ["METAINFER_TASK_ID"] = str(req.get("task_id") or "task")
    os.environ["METAINFER_SERIAL_VALIDATE_GPU"] = str(assignment.gpu)
    manager = make_subagent_manager(
        claude_bin=args.claude_bin,
        model=config.claude_model,
        permission_mode="bypassPermissions",
        effort="max",
        extra_add_dirs=[args.workspace_dir],
        snapshot_file=(
            args.workspace_dir / "workers" / args.worker_id
            / "restart_integration_agents.json"
        ),
        max_concurrent=1,
    )
    store = StateStore(args.state_dir)
    pipeline = GenAndOptPipeline(
        req=req,
        state_dir=args.state_dir,
        workspace_dir=args.workspace_dir,
        store=store,
        manager=manager,
    )
    pipeline._worker_failures = failures
    try:
        completed_assignments = [
            item for item in config.assignments
            if item.worker_id in workers
        ]
        merged_skill = pipeline._author_merged_skill(
            config, completed_assignments
        )
        synthesis = pipeline._synthesize_final_candidate(
            config,
            workers,
            initial_metrics,
            str(req.get("task_id") or "task"),
        )
        validation = synthesis["validation"]
        worker_validation = dict(report.get("worker_validation") or {})
        for shape_id, shape_result in workers[args.worker_id].get(
            "shapes", {}
        ).items():
            worker_validation[shape_id] = {
                "passed": bool(shape_result.get("metrics", {}).get("passed")),
                "worker_id": args.worker_id,
                "physical_gpu": assignment.gpu,
                "candidate": shape_result.get("candidate"),
                "metrics": shape_result.get("metrics") or {},
                "artifact": shape_result.get("artifact") or {},
                "source": "restarted_child_accepted_compiled_artifact",
                "rerun_by_main": False,
            }
        report.update({
            "workers": workers,
            "worker_failures": failures,
            "initial_metrics": initial_metrics,
            "worker_validation": worker_validation,
            "synthesis": synthesis,
            "merged_skill": merged_skill,
            "final_validation": validation,
            "final_target": evaluate_final_target(
                baseline=initial_metrics,
                validation=validation,
                target_improvement_percent=config.minimum_improvement_percent,
            ),
            "status": "partial_success" if failures else "success",
            "restart_integrated_worker": args.worker_id,
            "restart_integrated_at": time.time(),
        })
        write_json(final_report_path, report)
        store.append_timeline("worker_restart_integrated", {
            "worker_id": args.worker_id,
            "physical_gpu": assignment.gpu,
            "serial_validation_gpu": assignment.gpu,
            "workers": sorted(workers),
        })
        return 0
    finally:
        manager.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
