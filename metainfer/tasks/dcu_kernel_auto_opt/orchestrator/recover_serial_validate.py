"""Recovery driver: re-run final synthesis + serial validation + report.

Use when a generate-and-optimize task stopped in ``serial_validate`` (e.g. an
infrastructure failure inside the validation bench) after the parallel lanes
already produced accepted artifacts. The driver reconstructs the per-worker
results from ``workers/<w>/accepted/<shape>/manifest.json``, re-runs
``_synthesize_final_candidate`` (which rebuilds ``final/source`` and performs
the full serial validation against every API shape), writes ``final_report.json``
and marks the task finished/success.

The accepted artifacts are the single source of truth; the in-memory
``workers``/``initial_metrics`` of the crashed orchestrator are rebuilt from
disk (accepted manifests + ``shared_baseline/results.json``).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from metainfer.orchestrator._bootstrap import make_subagent_manager
from metainfer.orchestrator.state import StateStore

from . import phases
from .config import load_config, resolve_claude_bin
from .gen_and_opt_pipeline import GenAndOptPipeline
from .result_store import SCHEMA_VERSION, write_json
from .w8a8_pipeline import evaluate_final_target


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def reconstruct_workers(
    workspace_dir: Path, config
) -> dict[str, dict]:
    """Rebuild ``workers[worker_id] = {"shapes": {shape_id: {...}}}`` from the
    accepted artifact manifests written by the (possibly restarted) lanes."""
    workspace_dir = Path(workspace_dir)
    workers: dict[str, dict] = {}
    for assignment in config.assignments:
        root = workspace_dir / "workers" / assignment.worker_id
        lane_shapes: dict[str, dict] = {}
        for shape_id in assignment.shape_ids:
            manifest_path = root / "accepted" / shape_id / "manifest.json"
            if not manifest_path.is_file():
                continue
            manifest = _load_json(manifest_path)
            if not manifest.get("source") or not manifest.get("object"):
                continue
            lane_shapes[shape_id] = {
                "metrics": manifest.get("metrics") or {},
                "artifact": {
                    "source": manifest["source"],
                    "object": manifest["object"],
                    "source_sha256": manifest.get("source_sha256"),
                    "object_sha256": manifest.get("object_sha256"),
                    "commit": manifest.get("commit"),
                },
                "candidate": manifest.get("commit"),
            }
        if lane_shapes:
            workers[assignment.worker_id] = {"shapes": lane_shapes}
    return workers


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="dcu-kernel-auto-opt-recover-serial-validate"
    )
    parser.add_argument("requirements", type=Path)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--workspace-dir", type=Path, required=True)
    parser.add_argument(
        "--claude-bin",
        default=None,
        help=(
            "Agent binary override for the skill-synthesis agent; defaults "
            "resolved from agent_framework."
        ),
    )
    args = parser.parse_args()

    req = _load_json(args.requirements)
    config = load_config(req)
    task_id = str(req.get("task_id") or "task")
    claude_bin = resolve_claude_bin(
        config.agent_framework, explicit=args.claude_bin
    )
    os.environ["METAINFER_TASK_ID"] = task_id
    # Serial validation is GPU-contention sensitive (µs-scale decode kernels;
    # see bridge/dsh/README.md). Use an env override so recovery can target an
    # idle GPU instead of always hammering GPU 0; default to a non-zero card.
    os.environ.setdefault("METAINFER_SERIAL_VALIDATE_GPU", "3")

    workspace = args.workspace_dir
    baseline = _load_json(workspace / "shared_baseline" / "results.json")
    initial_metrics = dict(baseline.get("shapes") or {})
    workers = reconstruct_workers(workspace, config)
    if not workers:
        raise RuntimeError(
            "no accepted artifacts found under workers/*/accepted; "
            "cannot recover serial validation"
        )
    completed_assignments = [
        assignment
        for assignment in config.assignments
        if assignment.worker_id in workers
    ]
    print(
        f"recovered workers: {sorted(workers)} "
        f"({sum(len(w['shapes']) for w in workers.values())} shapes)",
        flush=True,
    )

    manager = make_subagent_manager(
        claude_bin=claude_bin,
        model=config.claude_model,
        permission_mode="bypassPermissions",
        effort="max",
        extra_add_dirs=[workspace],
        snapshot_file=workspace / "recovery_agents.json",
        max_concurrent=1,
    )
    store = StateStore(args.state_dir)
    pipeline = GenAndOptPipeline(
        req=req,
        state_dir=args.state_dir,
        workspace_dir=workspace,
        store=store,
        manager=manager,
    )
    pipeline._worker_failures = {}
    try:
        pipeline._phase(
            phases.VALIDATE,
            action="merge_and_validate_final_kernel",
        )
        try:
            merged_skill = pipeline._author_merged_skill(
                config, completed_assignments
            )
        except Exception as exc:  # noqa: BLE001 - skill is documentation only
            print(
                f"skill synthesis agent failed ({exc!r}); "
                "continuing with an empty merged skill",
                flush=True,
            )
            merged_skill = {
                "schema_version": 1,
                "task_id": task_id,
                "status": "skipped",
                "reason": "recovery: skill synthesis agent unavailable",
            }
        synthesis = pipeline._synthesize_final_candidate(
            config, workers, initial_metrics, task_id
        )
        validation = synthesis["validation"]
        worker_validation = {
            shape_id: {
                "passed": bool(shape_result.get("metrics", {}).get("passed")),
                "worker_id": assignment.worker_id,
                "physical_gpu": assignment.gpu,
                "candidate": shape_result.get("candidate"),
                "metrics": shape_result.get("metrics") or {},
                "artifact": shape_result.get("artifact") or {},
                "source": "accepted_compiled_artifact",
                "rerun_by_main": False,
            }
            for assignment in completed_assignments
            for shape_id, shape_result in workers[
                assignment.worker_id
            ].get("shapes", {}).items()
        }
        pipeline._phase(phases.REPORT)
        report = {
            "schema_version": SCHEMA_VERSION,
            "task_id": task_id,
            "task_type": "dcu-kernel-auto-opt",
            "mode": "generate-and-optimize",
            "started_at": _load_json(
                workspace / "plan.json"
            ).get("created_at", time.time()),
            "finished_at": time.time(),
            "duration_s": None,
            "config": _load_json(workspace / "plan.json"),
            "initial_metrics": initial_metrics,
            "workers": workers,
            "worker_failures": {},
            "worker_validation": worker_validation,
            "synthesis": synthesis,
            "merged_skill": merged_skill,
            "final_validation": validation,
            "final_target": evaluate_final_target(
                baseline=initial_metrics,
                validation=validation,
                target_improvement_percent=config.minimum_improvement_percent,
            ),
            "real_gpu_used": True,
            "kernel_generated": True,
            "gpu_assignment_agent_decided": False,
            "target_repo_modified": True,
            "status": "success",
            "recovered_serial_validate": True,
            "recovered_at": time.time(),
        }
        write_json(workspace / "final_report.json", report)
        store.update_run(
            current_iteration=config.mock_iterations,
            current_phase=phases.FINISHED,
            finished=True,
            final_status="success",
            last_outcome="ok",
            last_transition_label="recovered serial validate complete",
        )
        store.append_timeline(
            "orchestrator_success",
            {
                "task_id": task_id,
                "status": "success",
                "recovered_serial_validate": True,
            },
        )
        print(f"recovery complete: {workspace / 'final_report.json'}", flush=True)
        return 0
    finally:
        manager.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
