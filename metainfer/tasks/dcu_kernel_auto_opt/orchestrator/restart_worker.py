"""Restart one failed DCU auto-opt lane without stopping sibling lanes."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from metainfer.orchestrator._bootstrap import make_subagent_manager
from metainfer.orchestrator.state import StateStore

from .config import load_config, replace_assignments
from .gen_and_opt_pipeline import GenAndOptPipeline
from .result_store import write_json


def main() -> int:
    parser = argparse.ArgumentParser(prog="dcu-kernel-auto-opt-restart-worker")
    parser.add_argument("requirements", type=Path)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--workspace-dir", type=Path, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--claude-bin", default="ccb")
    args = parser.parse_args()

    req = json.loads(args.requirements.read_text(encoding="utf-8"))
    config = load_config(req)
    matches = [
        assignment
        for assignment in config.assignments
        if assignment.worker_id == args.worker_id
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one assignment for {args.worker_id}, got {len(matches)}"
        )
    assignment = matches[0]
    lane_config = replace_assignments(config, [assignment])
    task_id = str(req.get("task_id", "task"))
    os.environ["METAINFER_TASK_ID"] = task_id

    worker_root = args.workspace_dir / "workers" / args.worker_id
    restart_root = worker_root / "restarts" / str(int(time.time()))
    restart_root.mkdir(parents=True, exist_ok=False)
    prior_failure = worker_root / "failure.json"
    if prior_failure.is_file():
        prior_failure.replace(restart_root / "prior_failure.json")

    store = StateStore(args.state_dir)
    store.append_timeline(
        "worker_restart_started",
        {
            "worker_id": assignment.worker_id,
            "physical_gpu": assignment.gpu,
            "shape_ids": assignment.shape_ids,
            "mode": "isolated_lane_sidecar",
            "restart_dir": str(restart_root),
        },
    )
    manager = make_subagent_manager(
        claude_bin=args.claude_bin,
        model=config.claude_model,
        permission_mode="bypassPermissions",
        effort="max",
        extra_add_dirs=[args.workspace_dir],
        snapshot_file=restart_root / "agents.json",
        max_concurrent=1,
    )
    pipeline = GenAndOptPipeline(
        req=req,
        state_dir=args.state_dir,
        workspace_dir=args.workspace_dir,
        store=store,
        manager=manager,
    )
    try:
        baseline = pipeline._bootstrap_worker_repos(lane_config)
        workers = pipeline._parallel_agents(lane_config, baseline)
        result = {
            "schema_version": 1,
            "task_id": task_id,
            "worker_id": assignment.worker_id,
            "physical_gpu": assignment.gpu,
            "shape_ids": assignment.shape_ids,
            "status": "completed",
            "baseline": baseline,
            "workers": workers,
            "finished_at": time.time(),
        }
        write_json(restart_root / "result.json", result)
        write_json(worker_root / "restart_result.json", result)
        store.append_timeline(
            "worker_restart_complete",
            {
                "worker_id": assignment.worker_id,
                "physical_gpu": assignment.gpu,
                "shape_ids": assignment.shape_ids,
                "restart_dir": str(restart_root),
            },
        )
        return 0
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "task_id": task_id,
            "worker_id": assignment.worker_id,
            "physical_gpu": assignment.gpu,
            "shape_ids": assignment.shape_ids,
            "status": "failed",
            "error": repr(exc),
            "finished_at": time.time(),
        }
        write_json(restart_root / "result.json", failure)
        store.append_timeline("worker_restart_failed", failure)
        raise
    finally:
        manager.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
