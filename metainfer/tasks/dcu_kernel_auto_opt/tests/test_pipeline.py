from __future__ import annotations

import json

from metainfer.orchestrator.state import StateStore

from ..orchestrator.guidance import add_guidance
from ..orchestrator.pipeline import MockOptimizationPipeline


def _requirements():
    return {
        "task_id": "mock-task",
        "task_type": "dcu-kernel-auto-opt",
        "answers": {
            "execution_mode": "Mock (no GPU)",
            "mock_iterations": "2",
            "minimum_improvement_percent": 1.0,
            "shape_config": """
shapes:
  - {id: m2, M: 2, N: 1536, K: 4096}
  - {id: m16, M: 16, N: 1536, K: 4096}
assignments:
  worker_0: {gpu: 0, shapes: [m2]}
  worker_1: {gpu: 1, shapes: [m16]}
""",
        },
    }


def test_mock_pipeline_is_parallel_and_gpu_free(tmp_path):
    state_dir = tmp_path / "state"
    workspace_dir = tmp_path / "workspace"
    add_guidance(
        state_dir / "guidance", "worker_0", "prefer a smaller LDS tile"
    )
    report = MockOptimizationPipeline(
        req=_requirements(),
        state_dir=state_dir,
        workspace_dir=workspace_dir,
        store=StateStore(state_dir),
    ).run()
    assert report["status"] == "success"
    assert report["real_gpu_used"] is False
    assert set(report["workers"]) == {"worker_0", "worker_1"}
    assert all(v["passed"] for v in report["final_validation"].values())
    for result in report["final_validation"].values():
        assert result["metrics"]["tflops"] > 0
        assert result["metrics"]["bandwidth_gb_s"] > 0
    for worker in ("worker_0", "worker_1"):
        status = json.loads(
            (workspace_dir / "workers" / worker / "status.json").read_text()
        )
        assert status["state"] == "completed"
        assert status["gpu_binding"]["enforced"] is False
    worker_0_log = (
        workspace_dir / "workers" / "worker_0" / "runs" / "m2"
        / "experiments.jsonl"
    )
    first = json.loads(worker_0_log.read_text().splitlines()[0])
    assert first["manual_guidance"] == "prefer a smaller LDS tile"
    pending = workspace_dir / "skills" / "pending"
    skills = sorted(pending.glob("*/SKILL.md"))
    assert len(skills) == 3
    contents = [path.read_text() for path in skills]
    manifests = [
        json.loads((path.parent / "manifest.json").read_text())
        for path in skills
    ]
    assert sum(item["kind"] == "merged" for item in manifests) == 1
    assert sum("## Measured results" in text for text in contents) == 2


def test_dry_run_only_writes_plan(tmp_path):
    state_dir = tmp_path / "state"
    workspace_dir = tmp_path / "workspace"
    report = MockOptimizationPipeline(
        req=_requirements(),
        state_dir=state_dir,
        workspace_dir=workspace_dir,
        store=StateStore(state_dir),
    ).run(dry_run=True)
    assert report["dry_run"] is True
    assert not (workspace_dir / "shared_baseline" / "results.json").exists()
