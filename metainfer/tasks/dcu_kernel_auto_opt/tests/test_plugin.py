from __future__ import annotations

import json

import metainfer.tasks  # noqa: F401
from metainfer.orchestrator.registry import get_orchestrator
from metainfer.server.forms import load_form_schema
from metainfer.server.registry import get
from ..server.routes import read_worker_lanes


def test_task_and_web_plugins_registered():
    assert get_orchestrator("dcu-kernel-auto-opt").task_type == "dcu-kernel-auto-opt"
    plugin = get("dcu-kernel-auto-opt")
    assert plugin is not None
    assert plugin.detail_view_module == "app/dkao-detail"
    assert "app/form-overrides/dcu-kernel-auto-opt" in plugin.importmap_entries


def test_form_schema_is_available():
    schema = load_form_schema("dcu-kernel-auto-opt")
    assert schema is not None
    assert schema["type"] == "dcu-kernel-auto-opt"
    keys = {field["key"] for field in schema["fields"]}
    assert {
        "operator", "kernel_language", "target_hardware", "shape_config",
        "dtype", "correctness_ref", "perf_target", "max_iterations",
        "execution_mode", "target_repo_path", "claude_model",
    } <= keys
    shape = next(
        field for field in schema["fields"]
        if field["key"] == "shape_config"
    )
    assert shape["required"] is False
    assert shape["default"] == ""
    assert shape["override_component"] == "shape-input"
    model = next(
        field for field in schema["fields"]
        if field["key"] == "claude_model"
    )
    assert model["default"] == "Opus"
    assert [option["label"] for option in model["options"]] == [
        "Opus", "Sonnet"
    ]


def test_worker_lanes_always_has_four_rows(tmp_path):
    lanes = read_worker_lanes(tmp_path)
    assert [row["worker_id"] for row in lanes["workers"]] == [
        "worker_0", "worker_1", "worker_2", "worker_3",
    ]


def test_worker_lanes_surface_live_optimization_step(tmp_path):
    workspace = tmp_path / "workspace"
    state = tmp_path / "state"
    worker = workspace / "workers" / "worker_0"
    runs = worker / "runs" / "m2"
    runs.mkdir(parents=True)
    state.mkdir()
    workspace.joinpath("plan.json").write_text(json.dumps({
        "max_iterations": 5,
        "assignments": [{
            "worker_id": "worker_0",
            "gpu": 0,
            "shapes": ["m2"],
        }],
    }), encoding="utf-8")
    worker.joinpath("status.json").write_text(json.dumps({
        "state": "validating_candidate",
        "iteration": 2,
        "shape_id": "m2",
    }), encoding="utf-8")
    runs.joinpath("experiments.jsonl").write_text(
        json.dumps({"iteration": 1, "shape_id": "m2"}) + "\n",
        encoding="utf-8",
    )
    state.joinpath("agents.json").write_text(json.dumps({
        "agents": [{
            "name": "worker_0-m2-iter2",
            "status": "running",
            "started_at": 10,
            "last_output_age_s": 181,
        }],
    }), encoding="utf-8")

    lane = read_worker_lanes(workspace, state)["workers"][0]

    assert lane["step"] == "Compiling and validating candidate"
    assert lane["completed_rounds"] == 1
    assert lane["target_rounds"] == 5
    assert lane["long_running"] is True
    assert lane["active_iteration"] == {
        "iteration": 2,
        "shape_id": "m2",
        "state": "validating_candidate",
        "step": "Compiling and validating candidate",
        "agent_name": "worker_0-m2-iter2",
        "agent_status": "running",
        "elapsed_s": None,
        "last_output_age_s": 181,
    }


def test_worker_lanes_surface_pmc_and_repair_steps(tmp_path):
    workspace = tmp_path / "workspace"
    state = tmp_path / "state"
    worker = workspace / "workers" / "worker_0"
    worker.mkdir(parents=True)
    state.mkdir()
    workspace.joinpath("plan.json").write_text(json.dumps({
        "max_iterations": 5,
        "assignments": [{
            "worker_id": "worker_0",
            "gpu": 0,
            "shapes": ["m2"],
        }],
    }), encoding="utf-8")

    worker.joinpath("status.json").write_text(json.dumps({
        "state": "profiling_current_best",
        "iteration": 2,
        "shape_id": "m2",
    }), encoding="utf-8")
    lane = read_worker_lanes(workspace, state)["workers"][0]
    assert lane["active_iteration"]["step"] == (
        "Profiling current best kernel with PMC"
    )

    worker.joinpath("status.json").write_text(json.dumps({
        "state": "repairing_candidate",
        "iteration": 2,
        "shape_id": "m2",
        "repair": 3,
        "max_repairs": 4,
    }), encoding="utf-8")
    lane = read_worker_lanes(workspace, state)["workers"][0]
    assert lane["active_iteration"]["step"] == (
        "Repairing compile/correctness failure (3/4)"
    )
    assert lane["active_iteration"]["repair"] == 3
    assert lane["active_iteration"]["max_repairs"] == 4


def test_worker_lanes_surface_bootstrap_attempts(tmp_path):
    workspace = tmp_path / "workspace"
    state = tmp_path / "state"
    source = workspace / "workers" / "worker_0" / "source"
    source.joinpath("csrc").mkdir(parents=True)
    state.mkdir()
    workspace.joinpath("plan.json").write_text(json.dumps({
        "assignments": [{
            "worker_id": "worker_0",
            "gpu": 0,
            "shapes": ["m2_wqkv_a"],
        }],
    }), encoding="utf-8")
    source.joinpath("csrc", "w8a8_gemm_hip.hip").write_text(
        "// hip", encoding="utf-8"
    )
    state.joinpath("agents.json").write_text(json.dumps({
        "agents": [{
            "name": "worker_0-bootstrap-attempt1",
            "status": "running",
            "success": None,
            "elapsed_s": 12.5,
            "last_output_age_s": 0.5,
        }],
    }), encoding="utf-8")

    lane = read_worker_lanes(workspace, state)["workers"][0]

    assert lane["state"] == "bootstrap_running"
    assert lane["bootstrap_attempts"] == [{
        "kind": "bootstrap",
        "attempt": 1,
        "status": "running",
        "hypothesis": (
            "Create and validate the initial HIP implementation for "
            "m2_wqkv_a."
        ),
        "generated_files": ["csrc/w8a8_gemm_hip.hip"],
        "metrics": {},
        "artifact_dir": None,
        "candidate_files": [],
        "error": None,
        "elapsed_s": 12.5,
        "last_output_age_s": 0.5,
        "started_at": None,
    }]


def test_worker_lanes_surface_live_bootstrap_performance_metrics(tmp_path):
    workspace = tmp_path / "workspace"
    state = tmp_path / "state"
    worker = workspace / "workers" / "worker_0"
    source = worker / "source"
    source.mkdir(parents=True)
    state.mkdir()
    workspace.joinpath("plan.json").write_text(json.dumps({
        "assignments": [{
            "worker_id": "worker_0",
            "gpu": 0,
            "shapes": ["m2_wqkv_a"],
        }],
    }), encoding="utf-8")
    state.joinpath("agents.json").write_text(json.dumps({
        "agents": [{
            "name": "worker_0-bootstrap-attempt1",
            "status": "completed",
            "success": True,
        }],
    }), encoding="utf-8")
    metrics = {
        "passed": True,
        "median_us": 8.25,
        "p90_us": 8.75,
        "tflops": 19.5,
        "bandwidth_gb_s": 812.0,
    }
    worker.joinpath("bootstrap_progress.json").write_text(json.dumps({
        "attempt": 1,
        "status": "validating",
        "hypothesis": "Use a DUMMA tiled kernel.",
        "metrics": {"m2_wqkv_a": metrics},
    }), encoding="utf-8")

    attempt = read_worker_lanes(
        workspace, state
    )["workers"][0]["bootstrap_attempts"][0]

    assert attempt["status"] == "validating"
    assert attempt["hypothesis"] == "Use a DUMMA tiled kernel."
    assert attempt["metrics"]["m2_wqkv_a"] == metrics


def test_worker_lanes_surface_bootstrap_iteration_snapshot(tmp_path):
    workspace = tmp_path / "workspace"
    state = tmp_path / "state"
    worker = workspace / "workers" / "worker_0"
    record = (
        worker / "iterations" / "bootstrap"
        / "iteration1" / "iteration.json"
    )
    record.parent.mkdir(parents=True)
    state.mkdir()
    workspace.joinpath("plan.json").write_text(json.dumps({
        "assignments": [{
            "worker_id": "worker_0",
            "gpu": 0,
            "shapes": ["m2"],
        }],
    }), encoding="utf-8")
    state.joinpath("agents.json").write_text(json.dumps({
        "agents": [{
            "name": "worker_0-bootstrap-attempt1",
            "status": "done",
            "success": True,
        }],
    }), encoding="utf-8")
    record.write_text(json.dumps({
        "attempt": 1,
        "status": "failed",
        "error": "trusted compile failed",
        "artifact_dir": "iterations/bootstrap/iteration1",
        "candidate_files": ["csrc/w8a8_gemm_hip.hip"],
        "metrics": {},
    }), encoding="utf-8")

    attempt = read_worker_lanes(
        workspace, state
    )["workers"][0]["bootstrap_attempts"][0]

    assert attempt["status"] == "failed"
    assert attempt["error"] == "trusted compile failed"
    assert attempt["artifact_dir"] == "iterations/bootstrap/iteration1"
    assert attempt["candidate_files"] == ["csrc/w8a8_gemm_hip.hip"]
