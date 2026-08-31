from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from metainfer.orchestrator.state import StateStore
from metainfer.orchestrator.subagent_manager import SubAgentManager

from ..orchestrator.config import (
    GEN_AND_OPT_MODE,
    load_config,
    replace_assignments,
    validate_gpu_assignment,
    ShapeSpec,
    WorkerAssignment,
)
from ..orchestrator.gen_and_opt_pipeline import (
    GenAndOptPipeline,
    _COORDINATOR_AGENT_ARGS,
    _PERF_GATE_MAX_RETRIES,
    _PERF_GATE_RETRY_INTERVAL_S,
    _artifact_symbol_prefix,
    _final_performance_gate,
    _final_synthesis_prompt,
    _is_control_plane_artifact,
    _render_prebuilt_dispatch,
    _require_valid_child_assignments,
)
from ..orchestrator import phases
from ..orchestrator import w8a8_pipeline as pipeline_module
from ..orchestrator.w8a8_pipeline import (
    RealW8A8OptimizationPipeline,
    W8A8Runner,
    _BENCHMARK_TIMEOUT_S,
    _check_required_files,
    _REFERENCE_PREPARE_TIMEOUT_S,
    archive_iteration_candidate,
    candidate_iteration_destination,
    publish_iteration_candidate,
    snapshot_accepted_kernel_artifact,
)


def test_generated_repo_is_valid_existing_repo_seed(tmp_path):
    repo = tmp_path / "generated-repo"
    for name in (
        "int8_w8a8_gemm_api.py",
        "w8a8_backend.py",
        "w8a8_bench.py",
        "setup.py",
        "csrc/bindings.cpp",
        "csrc/w8a8_gemm_hip.hip",
    ):
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# seed\n", encoding="utf-8")

    assert _check_required_files(repo)


def _gen_req(shape_config: str = ""):
    if not shape_config:
        shape_config = """
shapes:
  - {id: m2, M: 2, N: 1536, K: 4096}
  - {id: m16, M: 16, N: 1536, K: 4096}
assignments:
  worker_0: {gpu: 0, shapes: [m2]}
  worker_1: {gpu: 1, shapes: [m16]}
"""
    return {
        "task_id": "gen-test",
        "task_type": "dcu-kernel-auto-opt",
        "answers": {
            "execution_mode": GEN_AND_OPT_MODE,
            "shape_config": shape_config,
            "mock_iterations": "3",
            "minimum_improvement_percent": 1.0,
            "operator": "Quantized GEMM",
            "dtype": "INT8 W8A8",
            "target_hardware": "K500SM_AI / gfx928",
            "kernel_language": "HIP C++",
        },
    }


def _shapes_only_req():
    """Request with shapes but no assignments — the generate agent decides."""
    return {
        "task_id": "gen-test-auto",
        "task_type": "dcu-kernel-auto-opt",
        "answers": {
            "execution_mode": GEN_AND_OPT_MODE,
            "shape_config": """
shapes:
  - {id: m2, M: 2, N: 1536, K: 4096}
  - {id: m16, M: 16, N: 1536, K: 4096}
  - {id: m64, M: 64, N: 1536, K: 4096}
""",
            "mock_iterations": "3",
            "minimum_improvement_percent": 1.0,
            "operator": "Quantized GEMM",
            "dtype": "INT8 W8A8",
            "target_hardware": "K500SM_AI / gfx928",
            "kernel_language": "HIP C++",
        },
    }


def _four_worker_req():
    return _gen_req("""
shapes:
  - {id: m2, M: 2, N: 1536, K: 4096}
  - {id: m4, M: 4, N: 1536, K: 4096}
  - {id: m16, M: 16, N: 1536, K: 4096}
  - {id: m4096, M: 4096, N: 1536, K: 4096}
assignments:
  worker_0: {gpu: 0, shapes: [m2]}
  worker_1: {gpu: 1, shapes: [m4]}
  worker_2: {gpu: 2, shapes: [m16]}
  worker_3: {gpu: 3, shapes: [m4096]}
""")


def _make_pipeline(req, tmp_path):
    """Create a minimal GenAndOptPipeline for contract validation testing."""
    state_dir = tmp_path / "state"
    workspace_dir = tmp_path / "workspace"
    manager = SubAgentManager(claude_bin="ccb")
    return GenAndOptPipeline(
        req=req,
        state_dir=state_dir,
        workspace_dir=workspace_dir,
        store=StateStore(state_dir),
        manager=manager,
    )


def test_real_continuation_keeps_fixed_triton_comparison_baseline(
    tmp_path, monkeypatch
):
    from ..orchestrator import w8a8_pipeline as pipeline_module

    req = _gen_req("""
shapes:
  - {id: m16_wo_b, M: 16, N: 4096, K: 2048}
assignments:
  worker_2: {gpu: 2, shapes: [m16_wo_b]}
""")
    req["answers"]["execution_mode"] = "Real INT8 W8A8 GEMM"
    config = load_config(req)
    workspace = tmp_path / "workspace"
    (workspace / "workers" / "worker_2").mkdir(parents=True)
    (workspace / "shared_baseline").mkdir()
    pipeline = RealW8A8OptimizationPipeline(
        req=req,
        state_dir=tmp_path / "state",
        workspace_dir=workspace,
        store=StateStore(tmp_path / "state"),
        manager=SubAgentManager(claude_bin="ccb"),
    )

    class FakeRunner:
        def __init__(self, worker_root, gpu):
            pass

        def probe(self):
            return {"visible_devices": 1}

        def benchmark(self, shape):
            return {
                "passed": True,
                "median_us": 45.0,
                "p90_us": 46.0,
            }

    monkeypatch.setattr(pipeline_module, "W8A8Runner", FakeRunner)

    baseline = pipeline._parallel_baseline(config)

    assert baseline["m16_wo_b"]["median_us"] == 54.617
    assert baseline["m16_wo_b"]["baseline_kind"] == "triton_graph"
    assert baseline["m16_wo_b"]["bootstrap_metrics"]["median_us"] == 45.0


def test_parallel_explore_continues_with_two_of_four_workers(
    tmp_path, monkeypatch
):
    pipeline = _make_pipeline(_four_worker_req(), tmp_path)
    config = load_config(_four_worker_req())
    for assignment in config.assignments:
        (
            pipeline.workspace_dir / "workers" / assignment.worker_id
        ).mkdir(parents=True)

    def run_worker(_config, assignment, _baseline):
        if assignment.worker_id in {"worker_2", "worker_3"}:
            raise RuntimeError("agent stuck timeout")
        return {"worker_id": assignment.worker_id, "shapes": {}}

    monkeypatch.setattr(pipeline, "_run_worker", run_worker)
    monkeypatch.setattr(
        pipeline,
        "_author_worker_skill",
        lambda _config, assignment: {
            "name": f"{assignment.worker_id}-skill"
        },
    )

    workers = pipeline._parallel_agents(config, {})

    assert set(workers) == {"worker_0", "worker_1"}
    assert set(pipeline._worker_failures) == {"worker_2", "worker_3"}
    assert all(
        item["state"] == "timed_out"
        for item in pipeline._worker_failures.values()
    )


def test_parallel_explore_continues_with_only_one_of_four_workers(
    tmp_path, monkeypatch
):
    pipeline = _make_pipeline(_four_worker_req(), tmp_path)
    config = load_config(_four_worker_req())
    for assignment in config.assignments:
        (
            pipeline.workspace_dir / "workers" / assignment.worker_id
        ).mkdir(parents=True)

    def run_worker(_config, assignment, _baseline):
        if assignment.worker_id != "worker_0":
            raise RuntimeError("worker failed")
        return {"worker_id": assignment.worker_id, "shapes": {}}

    monkeypatch.setattr(pipeline, "_run_worker", run_worker)
    monkeypatch.setattr(
        pipeline,
        "_author_worker_skill",
        lambda _config, assignment: {
            "name": f"{assignment.worker_id}-skill"
        },
    )

    workers = pipeline._parallel_agents(config, {})

    assert set(workers) == {"worker_0"}
    assert set(pipeline._worker_failures) == {
        "worker_1", "worker_2", "worker_3",
    }


def test_one_successful_lane_is_enough_for_main_synthesis(
    tmp_path, monkeypatch
):
    pipeline = _make_pipeline(_four_worker_req(), tmp_path)
    config = load_config(_four_worker_req())
    optimized = []

    def bootstrap(lane_config):
        assignment = lane_config.assignments[0]
        if assignment.worker_id != "worker_0":
            raise RuntimeError("bootstrap failed")
        return {"s0": {"passed": True, "median_us": 1.0}}

    def optimize(lane_config, baseline):
        assignment = lane_config.assignments[0]
        optimized.append((assignment.worker_id, baseline))
        return {
            assignment.worker_id: {
                "worker_id": assignment.worker_id,
                "shapes": {},
            }
        }

    monkeypatch.setattr(pipeline, "_bootstrap_worker_repos", bootstrap)
    monkeypatch.setattr(pipeline, "_parallel_agents", optimize)

    baseline, workers = pipeline._parallel_lane_lifecycles(config)

    assert optimized == [
        ("worker_0", {"s0": {"passed": True, "median_us": 1.0}})
    ]
    assert baseline == {"s0": {"passed": True, "median_us": 1.0}}
    assert set(workers) == {"worker_0"}


def test_iteration_agent_timeout_is_recorded_and_next_round_runs(
    tmp_path, monkeypatch
):
    from ..orchestrator import w8a8_pipeline as pipeline_module

    req = _gen_req("""
assignment_mode: manual
shapes:
  - {id: m2, M: 2, N: 1536, K: 4096}
assignments:
  worker_0: {gpu: 0, shapes: [m2]}
""")
    req["answers"]["mock_iterations"] = "2"
    config = load_config(req)
    pipeline = _make_pipeline(req, tmp_path)
    pipeline.store.init_or_resume("iteration-timeout-test", "dcu-kernel-auto-opt")
    root = pipeline.workspace_dir / "workers" / "worker_0"
    source = root / "source"
    source.joinpath("csrc").mkdir(parents=True)
    root.joinpath("logs").mkdir(parents=True)
    source.joinpath("int8_w8a8_gemm_api.py").write_text(
        "# immutable\n", encoding="utf-8"
    )
    source.joinpath("csrc", "w8a8_gemm_hip.hip").write_text(
        "// best kernel\n", encoding="utf-8"
    )
    subprocess.run(
        ["git", "init"], cwd=source, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.name", "test"], cwd=source, check=True
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=source,
        check=True,
    )
    subprocess.run(["git", "add", "."], cwd=source, check=True)
    subprocess.run(
        ["git", "commit", "-m", "best"],
        cwd=source,
        check=True,
        capture_output=True,
    )

    class FakeRunner:
        def __init__(self, worker_root, gpu):
            self.env = {}
            object_path = (
                worker_root / "cache" / "torch"
                / "metainfer_w8a8_backend"
                / "w8a8_gemm_hip.cuda.o"
            )
            object_path.parent.mkdir(parents=True, exist_ok=True)
            object_path.write_bytes(b"fake accepted object")

        def probe(self):
            return {"visible_devices": 1}

    launched = []
    monkeypatch.setattr(pipeline_module, "W8A8Runner", FakeRunner)
    monkeypatch.setattr(
        pipeline.manager, "launch", lambda spec: launched.append(spec.name)
    )
    monkeypatch.setattr(
        pipeline.manager,
        "result",
        lambda name: SimpleNamespace(
            success=False,
            error="killed after timeout",
            session_id=None,
        ),
    )

    baseline = {"m2": {"passed": True, "median_us": 10.0}}
    result = pipeline._run_worker(
        config, config.assignments[0], baseline
    )

    records = [
        json.loads(line)
        for line in (
            root / "runs" / "m2" / "experiments.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert launched == [
        f"worker_0-m2-iter{iteration}" for iteration in range(1, 8)
    ]
    assert [item["iteration"] for item in records] == list(range(1, 8))
    assert all(not item["accepted"] for item in records)
    assert all(
        "killed after timeout" in item["failure_reason"]
        for item in records
    )
    assert result["shapes"]["m2"]["metrics"] == baseline["m2"]
    assert source.joinpath("csrc", "w8a8_gemm_hip.hip").read_text(
        encoding="utf-8"
    ) == "// best kernel\n"


def test_control_plane_artifacts_are_not_attributed_to_agent():
    assert _is_control_plane_artifact(
        "__pycache__/w8a8_backend.cpython-310.pyc"
    )
    assert _is_control_plane_artifact("csrc/bindings_hip.cpp")
    assert not _is_control_plane_artifact("csrc/w8a8_gemm_hip.hip")


def test_trusted_harness_pytorch_reference_self_test():
    harness = (
        Path(__file__).resolve().parent.parent
        / "assets" / "w8a8_bench.py"
    )
    result = subprocess.run(
        ["python3", str(harness), "--self-test"],
        check=True,
        capture_output=True,
        text=True,
    )
    evidence = json.loads(result.stdout.strip().splitlines()[-1])
    assert evidence["self_test"] == "exact_w8a8_reference"
    assert evidence["passed"] is True
    assert evidence["actual"] == evidence["expected"]


def test_generate_stages_scaffold_and_assignment_without_hip(
    tmp_path, monkeypatch
):
    from ..orchestrator import gen_and_opt_pipeline as pipeline_module

    kernel_root = tmp_path / "kernel repos"
    monkeypatch.setenv("METAINFER_KERNEL_REPOS", str(kernel_root))
    req = _gen_req("""
assignment_mode: manual
shapes:
  - {id: m2, M: 2, N: 1536, K: 4096}
assignments:
  worker_0: {gpu: 0, shapes: [m2]}
""")
    req["answers"]["target_repo_path"] = "int8 task with spaces"
    pipeline = _make_pipeline(req, tmp_path)
    config = load_config(req)
    pipeline._validate_contract(config)
    monkeypatch.setattr(
        pipeline_module,
        "_validate_generate_scaffold",
        lambda *args, **kwargs: {
            "status": "passed",
            "implementation_present": False,
        },
    )
    launched_specs = []

    def launch(spec):
        launched_specs.append(spec)
        spec.workdir.joinpath("proposal.json").write_text(
            json.dumps({
                "gpu_assignment": {
                    "worker_0": {"gpu": 0, "shapes": ["m2"]}
                },
                "scaffold_review": {
                    "preflight_file": "generation_preflight.json",
                    "preflight_status": "passed",
                    "harness_reference_self_test_passed": True,
                    "gpu_probe_passed": True,
                    "cudagraph_available": True,
                    "python_graph_wrapper_staged": True,
                    "pmc_script_checked": True,
                    "no_hip_implementation": True,
                },
            }),
            encoding="utf-8",
        )

    monkeypatch.setattr(pipeline.manager, "launch", launch)
    monkeypatch.setattr(
        pipeline.manager,
        "result",
        lambda name: SimpleNamespace(success=True, error=None),
    )

    pipeline._prepare_worktrees(config, "gen-test")
    assignments = pipeline._generate_kernel_repo(config)

    repo = config.target_repo_path
    assert repo is not None
    assert assignments == config.assignments
    assert launched_specs[0].role == "kernel_coordinator"
    assert repo.joinpath("profile_pmc.sh").is_file()
    assert "--profile-only" in repo.joinpath(
        "profile_pmc.sh"
    ).read_text(encoding="utf-8")
    assert repo.joinpath("int8_w8a8_gemm_api.py").is_file()
    assert repo.joinpath("w8a8_bench.py").is_file()
    assert "--reference-cache-dir" in repo.joinpath(
        "w8a8_bench.py"
    ).read_text(encoding="utf-8")
    assert repo.joinpath("w8a8_graph.py").is_file()
    assert not repo.joinpath("csrc", "w8a8_gemm_hip.hip").exists()
    assert repo.joinpath("generation_preflight.json").is_file()
    assert repo.joinpath("generation_review.json").is_file()
    manifest = json.loads(
        repo.joinpath("scaffold_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert (
        manifest["initial_kernel"]
        == "pending_parallel_explore_child_generation"
    )
    timeline = pipeline.store.load_timeline()
    success = [
        item for item in timeline
        if item.get("type") == "generate_success"
    ][-1]
    assert success["payload"]["role"] == "main_coordinator"
    assert success["payload"]["kernel_source_created"] is False
    assert (
        success["payload"]["kernel_source"]
        == "pending_child_generation"
    )


def test_parallel_explore_child_generates_initial_hip(
    tmp_path, monkeypatch
):
    from ..orchestrator import gen_and_opt_pipeline as pipeline_module

    kernel_root = tmp_path / "kernel repos"
    monkeypatch.setenv("METAINFER_KERNEL_REPOS", str(kernel_root))
    req = _gen_req("""
assignment_mode: manual
shapes:
  - {id: m2, M: 2, N: 1536, K: 4096}
assignments:
  worker_0: {gpu: 0, shapes: [m2]}
""")
    req["answers"]["target_repo_path"] = "int8 task with spaces"
    pipeline = _make_pipeline(req, tmp_path)
    config = load_config(req)
    pipeline._validate_contract(config)
    pipeline._prepare_worktrees(config, "gen-test")
    monkeypatch.setattr(
        pipeline_module,
        "_validate_generate_scaffold",
        lambda *args, **kwargs: {
            "status": "passed",
            "implementation_present": False,
        },
    )

    def coordinate(spec):
        spec.workdir.joinpath("proposal.json").write_text(
            json.dumps({
                "gpu_assignment": {
                    "worker_0": {"gpu": 0, "shapes": ["m2"]}
                },
                "scaffold_review": {
                    "preflight_file": "generation_preflight.json",
                    "preflight_status": "passed",
                    "harness_reference_self_test_passed": True,
                    "gpu_probe_passed": True,
                    "cudagraph_available": True,
                    "python_graph_wrapper_staged": True,
                    "pmc_script_checked": True,
                    "no_hip_implementation": True,
                },
            }),
            encoding="utf-8",
        )

    monkeypatch.setattr(pipeline.manager, "launch", coordinate)
    monkeypatch.setattr(
        pipeline.manager,
        "result",
        lambda name: SimpleNamespace(success=True, error=None),
    )
    assignments = pipeline._generate_kernel_repo(config)
    config = replace_assignments(config, assignments)
    pipeline._create_worker_worktrees(config, "gen-test")

    class FakeRunner:
        def __init__(self, root, gpu):
            self.env = {}
            object_path = (
                root / "cache" / "torch"
                / "metainfer_w8a8_backend"
                / "w8a8_gemm_hip.cuda.o"
            )
            object_path.parent.mkdir(parents=True, exist_ok=True)
            object_path.write_bytes(b"fake accepted object")

        def benchmark(self, params):
            return {
                "passed": True,
                "graph_capture_passed": True,
                "timing_mode": "cuda_graph_replay",
                "median_us": 1.0,
                "p90_us": 1.1,
            }

    launched_specs = []

    def launch(spec):
        launched_specs.append(spec)
        spec.workdir.joinpath("csrc").mkdir(exist_ok=True)
        spec.workdir.joinpath(
            "csrc", "w8a8_gemm_hip.hip"
        ).write_text(
            "// generated by child Agent\n", encoding="utf-8"
        )
        spec.workdir.joinpath("proposal.json").write_text(
            '{"hypothesis":"fresh child kernel"}',
            encoding="utf-8",
        )

    monkeypatch.setattr(pipeline_module, "W8A8Runner", FakeRunner)
    monkeypatch.setattr(pipeline.manager, "launch", launch)
    monkeypatch.setattr(
        pipeline.manager,
        "result",
        lambda name: SimpleNamespace(success=True, error=None),
    )

    metrics = pipeline._bootstrap_worker_repos(config)

    repo = config.target_repo_path
    assert repo is not None
    worker_source = (
        pipeline.workspace_dir / "workers" / "worker_0" / "source"
    )
    assert metrics["m2"]["baseline_kind"] == "triton_graph"
    assert metrics["m2"]["median_us"] == 66.924
    assert metrics["m2"]["bootstrap_metrics"]["passed"] is True
    assert launched_specs[0].role == "dcu_w8a8_bootstrap_generator"
    assert launched_specs[0].workdir == worker_source
    assert not any(
        char.isspace() for char in str(worker_source.resolve())
    )
    assert not repo.joinpath("csrc", "w8a8_gemm_hip.hip").exists()
    assert worker_source.joinpath(
        "csrc", "w8a8_gemm_hip.hip"
    ).read_text(encoding="utf-8") == "// generated by child Agent\n"
    assert b"\r\n" not in worker_source.joinpath(
        "profile_pmc.sh"
    ).read_bytes()
    result = json.loads(
        (
            pipeline.workspace_dir / "workers" / "worker_0"
            / "bootstrap_result.json"
        ).read_text(encoding="utf-8")
    )
    assert result["source"] == "child_agent_generated"
    assert result["generated_files"] == ["csrc/w8a8_gemm_hip.hip"]
    assert (
        result["comparison_baselines"]["m2"]["median_us"] == 66.924
    )


def test_fresh_hip_generation_completes_before_parallel_explore(
    tmp_path, monkeypatch
):
    from ..orchestrator import gen_and_opt_pipeline as pipeline_module

    req = _gen_req("""
assignment_mode: manual
shapes:
  - {id: m2, M: 2, N: 16, K: 32}
assignments:
  worker_0: {gpu: 0, shapes: [m2]}
""")
    config = load_config(req)
    pipeline = _make_pipeline(req, tmp_path)
    observed_phases = []
    synthesis_phases = []

    monkeypatch.setattr(pipeline_module, "load_config", lambda request: config)
    monkeypatch.setattr(
        pipeline_module,
        "replace_assignments",
        lambda current, assignments: current,
    )
    monkeypatch.setattr(pipeline, "_validate_contract", lambda current: None)
    monkeypatch.setattr(
        pipeline, "_prepare_worktrees", lambda current, task_id: None
    )
    monkeypatch.setattr(pipeline, "_plan", lambda current: {})
    def generate(current):
        observed_phases.append(("generated", pipeline._current_phase))
        return current.assignments

    monkeypatch.setattr(pipeline, "_generate_kernel_repo", generate)
    monkeypatch.setattr(
        pipeline,
        "_create_worker_worktrees",
        lambda current, task_id: None,
    )

    def lane_lifecycles(current):
        observed_phases.append(("explored", pipeline._current_phase))
        return (
            {"m2": {"passed": True, "median_us": 1.0}},
            {"worker_0": {"worker_id": "worker_0", "shapes": {}}},
        )

    monkeypatch.setattr(
        pipeline,
        "_parallel_lane_lifecycles",
        lane_lifecycles,
    )
    def synthesize_final(current, workers, baseline, task_id):
        synthesis_phases.append(("final_candidate", pipeline._current_phase))
        return {"validation": {}}

    def author_skill(current, assignments):
        synthesis_phases.append(("merged_skill", pipeline._current_phase))
        return {"name": "merged"}

    monkeypatch.setattr(
        pipeline, "_synthesize_final_candidate", synthesize_final
    )
    monkeypatch.setattr(pipeline, "_author_merged_skill", author_skill)

    pipeline.run()

    assert observed_phases == [
        ("generated", phases.GENERATE),
        ("explored", phases.EXPLORE),
    ]
    assert synthesis_phases == [
        ("merged_skill", phases.SYNTHESIZE),
        ("final_candidate", phases.VALIDATE),
    ]


# --- Config parsing tests ----------------------------------------------- #

def test_gen_mode_config_parses():
    """Blank repo name deterministically falls back to the task id."""
    cfg = load_config(_gen_req())
    assert cfg.execution_mode == GEN_AND_OPT_MODE
    assert cfg.target_repo_path is not None
    assert cfg.target_repo_path.name == "gen-test"
    assert cfg.operator == "Quantized GEMM"
    assert cfg.dtype == "INT8 W8A8"
    assert set(cfg.shapes) == {"m2", "m16"}
    assert [a.gpu for a in cfg.assignments] == [0, 1]


def test_gen_mode_shapes_only_auto_assigns():
    """When no assignments given in generate mode, auto-assign to worker_0."""
    cfg = load_config(_shapes_only_req())
    assert cfg.execution_mode == GEN_AND_OPT_MODE
    assert set(cfg.shapes) == {"m2", "m16", "m64"}
    # Auto-assigned: all shapes → worker_0, gpu 0
    assert len(cfg.assignments) == 1
    assert cfg.assignments[0].worker_id == "worker_0"
    assert cfg.assignments[0].gpu == 0
    assert set(cfg.assignments[0].shape_ids) == {"m2", "m16", "m64"}


def test_gen_mode_requires_quantized_gemm_op(tmp_path):
    """Generate mode validates operator=Quantized GEMM in contract check."""
    req = _gen_req()
    req["answers"]["operator"] = "Custom operator"
    pipeline = _make_pipeline(req, tmp_path)
    cfg = load_config(req)
    with pytest.raises(ValueError, match="Generate mode requires operator"):
        pipeline._validate_contract(cfg)


def test_gen_mode_requires_w8a8_dtype(tmp_path):
    """Generate mode validates dtype=INT8 W8A8 in contract check."""
    req = _gen_req()
    req["answers"]["dtype"] = "FP16 / BF16"
    pipeline = _make_pipeline(req, tmp_path)
    cfg = load_config(req)
    with pytest.raises(ValueError, match="Generate mode requires dtype"):
        pipeline._validate_contract(cfg)


def test_gen_mode_shapes_need_mnk(tmp_path):
    """Every shape must have M, N, K dimensions."""
    req = _gen_req("""
shapes:
  - {id: bad, N: 16, K: 32}
assignments:
  worker_0: {gpu: 0, shapes: [bad]}
""")
    pipeline = _make_pipeline(req, tmp_path)
    cfg = load_config(req)
    with pytest.raises(ValueError, match="is missing"):
        pipeline._validate_contract(cfg)


def test_gen_mode_resolves_kernel_repo_name_with_spaces(
    tmp_path, monkeypatch
):
    """The user-facing name resolves below sibling kernel-repos."""
    kernel_root = tmp_path / "kernel-repos"
    monkeypatch.setenv("METAINFER_KERNEL_REPOS", str(kernel_root))
    req = _gen_req()
    req["answers"]["target_repo_path"] = "int8 test2"
    cfg = load_config(req)
    assert cfg.target_repo_path == kernel_root / "int8 test2"
    assert not cfg.target_repo_path.exists()


@pytest.mark.parametrize(
    "bad_name", ["../escape", "nested/repo", "/tmp/absolute", ".", ".."]
)
def test_gen_mode_rejects_unsafe_kernel_repo_name(bad_name):
    req = _gen_req()
    req["answers"]["target_repo_path"] = bad_name
    with pytest.raises(ValueError, match="single folder name"):
        load_config(req)


# --- GPU assignment validation ------------------------------------------ #

def test_validate_gpu_assignment_accepts_valid():
    """Valid assignment dict parses correctly."""
    shapes = {
        "m2": ShapeSpec("m2", {"M": 2, "N": 1536, "K": 4096}),
        "m16": ShapeSpec("m16", {"M": 16, "N": 1536, "K": 4096}),
        "m64": ShapeSpec("m64", {"M": 64, "N": 1536, "K": 4096}),
    }
    raw = {
        "worker_0": {"gpu": 0, "shapes": ["m2", "m16"]},
        "worker_1": {"gpu": 1, "shapes": ["m64"]},
    }
    result = validate_gpu_assignment(shapes, raw)
    assert len(result) == 2
    assert result[0].worker_id == "worker_0"
    assert result[0].gpu == 0
    assert set(result[0].shape_ids) == {"m2", "m16"}
    assert result[1].worker_id == "worker_1"
    assert result[1].gpu == 1
    assert result[1].shape_ids == ["m64"]


def test_validate_gpu_assignment_rejects_missing_shapes():
    """All shapes must be assigned."""
    shapes = {"m2": ShapeSpec("m2", {"M": 2}), "m16": ShapeSpec("m16", {"M": 16})}
    raw = {"worker_0": {"gpu": 0, "shapes": ["m2"]}}
    with pytest.raises(ValueError, match="unassigned shapes"):
        validate_gpu_assignment(shapes, raw)


def test_validate_gpu_assignment_rejects_duplicate_gpu():
    """Same GPU can't be used by two workers."""
    shapes = {"m2": ShapeSpec("m2", {"M": 2}), "m16": ShapeSpec("m16", {"M": 16})}
    raw = {
        "worker_0": {"gpu": 0, "shapes": ["m2"]},
        "worker_1": {"gpu": 0, "shapes": ["m16"]},
    }
    with pytest.raises(ValueError, match="assigned to more than one"):
        validate_gpu_assignment(shapes, raw)


def test_validate_gpu_assignment_rejects_duplicate_shapes():
    """Same shape can't be assigned twice."""
    shapes = {"m2": ShapeSpec("m2", {"M": 2}), "m16": ShapeSpec("m16", {"M": 16})}
    raw = {
        "worker_0": {"gpu": 0, "shapes": ["m2", "m16"]},
        "worker_1": {"gpu": 1, "shapes": ["m2"]},
    }
    with pytest.raises(ValueError, match="assigned more than once"):
        validate_gpu_assignment(shapes, raw)


def test_validate_gpu_assignment_rejects_empty():
    """Empty assignment dict raises."""
    with pytest.raises(ValueError, match="non-empty"):
        validate_gpu_assignment({"m2": ShapeSpec("m2", {"M": 2})}, {})


# --- replace_assignments ----------------------------------------------- #

def test_replace_assignments_preserves_shapes():
    """replace_assignments only changes the assignment list."""
    cfg = load_config(_gen_req())
    new_assignments = [
        WorkerAssignment("worker_0", 0, ["m2", "m16"]),
    ]
    new_cfg = replace_assignments(cfg, new_assignments)
    assert new_cfg.shapes == cfg.shapes
    assert new_cfg.operator == cfg.operator
    assert new_cfg.dtype == cfg.dtype
    assert new_cfg.assignment_mode == cfg.assignment_mode
    assert len(new_cfg.assignments) == 1
    assert set(new_cfg.assignments[0].shape_ids) == {"m2", "m16"}


# --- Phase ordering ---------------------------------------------------- #

def test_generate_phase_is_in_order():
    """Generate mode transitions directly from GENERATE to EXPLORE."""
    graph = phases.graph_payload(phases.PREPARE, include_baseline=False)
    node_ids = [n["id"] for n in graph["nodes"]]
    gen_idx = node_ids.index(phases.GENERATE)
    prep_idx = node_ids.index(phases.PREPARE)
    explore_idx = node_ids.index(phases.EXPLORE)
    assert phases.BASELINE not in node_ids
    assert prep_idx < gen_idx < explore_idx
    assert explore_idx == gen_idx + 1


def test_generate_accepts_sparse_one_to_four_worker_gpu_mapping():
    assignments = [
        WorkerAssignment(f"worker_{gpu}", gpu, [f"shape_{gpu}"])
        for gpu in range(4)
    ]
    _require_valid_child_assignments(assignments)
    _require_valid_child_assignments(assignments[:1])
    _require_valid_child_assignments(assignments[:3])
    _require_valid_child_assignments(assignments[1:3])
    with pytest.raises(ValueError, match="map each worker_N"):
        _require_valid_child_assignments([
            WorkerAssignment("worker_0", 2, ["shape_2"])
        ])
    with pytest.raises(ValueError, match="between one and four"):
        _require_valid_child_assignments([])


def test_generate_accepts_m_variants_on_different_workers():
    shapes = {
        "m1_wqkv_a": ShapeSpec(
            "m1_wqkv_a", {"M": 1, "N": 1536, "K": 4096}
        ),
        "m4_wqkv_a": ShapeSpec(
            "m4_wqkv_a", {"M": 4, "N": 1536, "K": 4096}
        ),
        "m16_wqkv_a": ShapeSpec(
            "m16_wqkv_a", {"M": 16, "N": 1536, "K": 4096}
        ),
    }
    raw = {
        "worker_0": {
            "gpu": 0, "shapes": ["m1_wqkv_a", "m4_wqkv_a"]
        },
        "worker_2": {"gpu": 2, "shapes": ["m16_wqkv_a"]},
    }

    assignments = validate_gpu_assignment(shapes, raw)
    _require_valid_child_assignments(assignments)

    assert assignments[0].shape_ids == ["m1_wqkv_a", "m4_wqkv_a"]
    assert assignments[1].shape_ids == ["m16_wqkv_a"]


def test_generate_phase_has_label():
    """GENERATE phase has a human-readable label."""
    graph = phases.graph_payload(phases.GENERATE)
    labels = {n["id"]: n["label"] for n in graph["nodes"]}
    assert labels[phases.GENERATE] == "Generate kernel repo"


def test_worker_cache_dirs_exist_before_owner_match(tmp_path, monkeypatch):
    """Host-agent cache directories must be included in the ownership pass."""
    from ..orchestrator import gen_and_opt_pipeline as pipeline_module

    pipeline = _make_pipeline(_gen_req(), tmp_path)
    (pipeline.workspace_dir / "main").mkdir(parents=True)
    config = load_config(_gen_req())
    observed = []

    def fake_run(command, *, cwd, **kwargs):
        if command[:3] == ["git", "worktree", "add"]:
            # The source path is the penultimate argument.
            Path(command[-2]).mkdir(parents=True)

        class Result:
            stdout = ""

        return Result()

    def fake_match_tree_owner(path, owner_source):
        observed.append(path)
        for name in ("torch", "triton", "xdg", "tmp"):
            assert (path / "cache" / name).is_dir()

    monkeypatch.setattr(pipeline_module, "_run", fake_run)
    monkeypatch.setattr(
        pipeline_module, "_match_tree_owner", fake_match_tree_owner
    )

    pipeline._create_worker_worktrees(config, "gen-test")

    assert len(observed) == len(config.assignments)
    for assignment in config.assignments:
        candidate = (
            pipeline.workspace_dir / "main" / "candidates"
            / assignment.worker_id
        )
        assert candidate.is_dir()
        assert not candidate.is_symlink()
        assert (candidate / "source").is_symlink()
        assert (candidate / "source").resolve() == (
            pipeline.workspace_dir / "workers"
            / assignment.worker_id / "source"
        ).resolve()
        assert (candidate / "csrc").is_symlink()
    assert "candidates/" in (
        pipeline.workspace_dir / "main" / ".git" / "info" / "exclude"
    ).read_text(encoding="utf-8").splitlines()


# --- Prompt content ---------------------------------------------------- #

def test_prompt_renders_shapes_table():
    """generate_kernel_prompt includes shape dims in a readable table."""
    from ..orchestrator.prompts import generate_kernel_prompt
    from pathlib import Path

    prompt = generate_kernel_prompt(
        operator="Quantized GEMM",
        dtype="INT8 W8A8",
        shapes={"m2": {"M": 2, "N": 1536, "K": 4096}},
        hardware="K500SM_AI / gfx928",
        kernel_language="HIP C++",
        source_dir=Path("/tmp/test"),
        harness_path=Path("/tmp/harness.py"),
    )
    assert "m2" in prompt
    assert "1536" in prompt
    assert "4096" in prompt
    assert "w8a8_gemm_out" in prompt
    assert "zth_w8a8.gemm_out" in prompt
    assert "immutable" in prompt.lower()
    assert "gfx928" in prompt
    assert "main coordinator" in prompt.lower()
    assert "do not create `.hip`" in prompt.lower()
    assert "do not compile" in prompt.lower()
    assert "generation_preflight.json" in prompt
    assert "w8a8_bench.py" in prompt
    assert "pytorch-reference self-test" in prompt.lower()
    assert "parallel explore" in prompt.lower()
    assert "child implementation agents" in prompt.lower()
    assert "initial hip kernels from scratch" in prompt.lower()
    assert "simple scalar" not in prompt.lower()


def test_prompt_includes_prev_failure():
    """When prev_failure is set, it appears in the prompt."""
    from ..orchestrator.prompts import generate_kernel_prompt
    from pathlib import Path

    prompt = generate_kernel_prompt(
        operator="Quantized GEMM",
        dtype="INT8 W8A8",
        shapes={"m2": {"M": 2, "N": 1536, "K": 4096}},
        hardware="K500SM_AI / gfx928",
        kernel_language="HIP C++",
        source_dir=Path("/tmp/test"),
        harness_path=Path("/tmp/harness.py"),
        prev_failure="Harness check failed: passed was false",
    )
    assert "Previous attempt failed" in prompt
    assert "Harness check failed" in prompt


def test_prompt_includes_gpu_assignment_instructions():
    """The prompt makes exact-shape assignment authoritative."""
    from ..orchestrator.prompts import generate_kernel_prompt
    from pathlib import Path

    prompt = generate_kernel_prompt(
        operator="Quantized GEMM",
        dtype="INT8 W8A8",
        shapes={"m2": {"M": 2, "N": 1536, "K": 4096}},
        hardware="K500SM_AI / gfx928",
        kernel_language="HIP C++",
        source_dir=Path("/tmp/test"),
        harness_path=Path("/tmp/harness.py"),
        fixed_assignment={
            "worker_0": {"gpu": 0, "shapes": ["m2"]}
        },
    )
    assert "GPU assignment" in prompt
    assert "gpu_assignment" in prompt
    assert "worker_0" in prompt
    assert "preserve it exactly" in prompt.lower()
    assert "different m variants" in prompt.lower()
    assert "may be assigned to different workers" in prompt.lower()


def test_control_plane_assignment_balances_exact_shapes():
    from ..orchestrator.prompts import shape_balanced_assignment

    shapes = {
        "m2_wqkv_a": {"M": 2, "N": 1536, "K": 4096},
        "m16_wqkv_a": {"M": 16, "N": 1536, "K": 4096},
        "m2_wq_b": {"M": 2, "N": 8192, "K": 1024},
        "m16_wq_b": {"M": 16, "N": 8192, "K": 1024},
        "m2_wo_b": {"M": 2, "N": 4096, "K": 2048},
        "m16_wo_b": {"M": 16, "N": 4096, "K": 2048},
        "m2_shared_gate_up": {"M": 2, "N": 1024, "K": 4096},
        "m16_shared_gate_up": {"M": 16, "N": 1024, "K": 4096},
        "m2_shared_down": {"M": 2, "N": 4096, "K": 512},
        "m16_shared_down": {"M": 16, "N": 4096, "K": 512},
    }
    assignment = shape_balanced_assignment(shapes)
    owners = {
        shape: worker
        for worker, payload in assignment.items()
        for shape in payload["shapes"]
    }
    assert set(owners) == set(shapes)
    loads = []
    for payload in assignment.values():
        loads.append(sum(
            2 * shapes[shape]["M"] * shapes[shape]["N"] * shapes[shape]["K"]
            for shape in payload["shapes"]
        ))
    assert max(loads) / min(loads) < 1.5


def test_prompt_assignment_example_omits_empty_subset_workers():
    from ..orchestrator.prompts import shape_balanced_assignment

    assignment = shape_balanced_assignment({
        "m2_wqkv_a": {"M": 2, "N": 1536, "K": 4096},
    })

    assert assignment == {
        "worker_0": {"gpu": 0, "shapes": ["m2_wqkv_a"]}
    }


def test_parallel_child_bootstrap_prompt_owns_hip_implementation():
    """Only the Parallel explore child owns fresh HIP source."""
    from ..orchestrator.prompts import bootstrap_worker_prompt
    from pathlib import Path

    prompt = bootstrap_worker_prompt(
        worker_id="worker_2",
        gpu=2,
        shapes={"m2": {"M": 2, "N": 1536, "K": 4096}},
        hardware="K500SM_AI / gfx928",
        kernel_language="HIP C++",
        source_dir=Path("/tmp/worker/source"),
        harness_path=Path("/tmp/harness.py"),
        api_contract_path=Path("/tmp/worker/source/int8_w8a8_gemm_api.py"),
        attempt=1,
    )
    assert "child kernel implementation agent" in prompt.lower()
    assert "parallel explore" in prompt.lower()
    assert "assigned shapes" in prompt.lower()
    assert "do not copy another task repo" in prompt.lower()
    assert "generate kernel repo phase" not in prompt.lower()
    assert "csrc/w8a8_gemm_hip.hip" in prompt
    assert "w8a8_backend.py" in prompt
    assert "w8a8_graph.py" in prompt
    assert "torch.cuda.CUDAGraph" in prompt
    assert "non-default stream" in prompt
    assert "Correctness-first bootstrap strategy" in prompt
    assert "Do not run the benchmark or correctness harness" in prompt
    assert "trusted control plane" in prompt
    assert "blockDim 128 or 256" in prompt
    assert "preserve it byte-for-byte" in prompt
    assert "load_extension()" in prompt
    assert "is_python_module=False" in prompt
    assert "You may change only" in prompt
    assert "`csrc/w8a8_gemm_hip.hip` plus `proposal.json`" in prompt
    assert "launch_pack_w8a8_weight" in prompt
    assert "void* workspace" in prompt
    assert "main coordinator and\nGenerate phase never write the HIP" in prompt


def test_parallel_child_initial_prompt_uses_scalar_correctness_for_m16():
    from ..orchestrator.prompts import bootstrap_worker_prompt

    prompt = bootstrap_worker_prompt(
        worker_id="worker_3",
        gpu=3,
        shapes={"m16": {"M": 16, "N": 8192, "K": 1024}},
        hardware="K500SM_AI / gfx928",
        kernel_language="HIP C++",
        source_dir=Path("/tmp/worker/source"),
        harness_path=Path("/tmp/harness.py"),
        api_contract_path=Path("/tmp/worker/source/int8_w8a8_gemm_api.py"),
        attempt=1,
    )

    assert "For every assigned shape,\nincluding M=16" in prompt
    assert "Do not use DUMMA" in prompt
    assert '"path": "scalar_correctness"' in prompt
    assert "wavefront size: 64" in prompt
    assert "Never use NVIDIA" in prompt
    assert "validation_owner" in prompt
    assert "scalar generic fallback" in prompt
    assert "paired M=2 API shape" in prompt


def test_parallel_child_large_prefill_bootstrap_starts_with_dumma():
    from ..orchestrator.prompts import bootstrap_worker_prompt

    prompt = bootstrap_worker_prompt(
        worker_id="worker_0",
        gpu=0,
        shapes={"prefill": {"M": 3072, "N": 1536, "K": 4096}},
        hardware="K500SM_AI / gfx928",
        kernel_language="HIP C++",
        source_dir=Path("/tmp/worker/source"),
        harness_path=Path("/tmp/harness.py"),
        api_contract_path=Path(
            "/tmp/worker/source/int8_w8a8_gemm_api.py"
        ),
        attempt=1,
    )

    assert "large-Prefill\nscalar K loop is not a usable" in prompt
    assert "INT8 DUMMA m16n16k32" in prompt
    assert "references/int8w8a8-gemm/hy3/TP4/M4096/o_proj.hip" in prompt
    assert "references/w8a8_gemm_variants.hip" in prompt
    assert "neither a whitelist nor a restriction" in prompt
    assert '"path": "dumma_prefill_with_scalar_fallback"' in prompt
    assert "Keep one simple scalar int8/int32 fallback" in prompt


def test_optimization_prompt_enforces_one_evidence_driven_change():
    prompt = RealW8A8OptimizationPipeline._worker_prompt(
        WorkerAssignment("worker_0", 0, ["m2"]),
        "m2",
        {"M": 2, "N": 1536, "K": 4096},
        {"median_us": 12.0, "p90_us": 13.0},
        Path("/tmp/worker"),
        2,
        None,
    )

    assert "one falsifiable bottleneck hypothesis" in prompt
    assert "already rejected change" in prompt
    assert "Do not choose split-K merely because K is large" in prompt
    assert "torch.cuda.CUDAGraph.replay()" in prompt
    assert "Do not run the harness" in prompt
    assert "namespace `du::dumma`" in prompt
    assert "filesystem-wide searches" in prompt
    assert "hygon-gfx928-memory-isa" not in prompt
    assert "hygon-gfx928-compute-isa" not in prompt
    assert "Skill tool is disabled" in prompt
    assert '"isa_optimization"' not in prompt
    assert '"observed_best": {"median_us": 12.0' in prompt
    assert "Mandatory decision for this round" in prompt
    assert "avoid per-K-tile LDS barriers" in prompt


def test_optimization_prompt_repairs_faster_incorrect_candidate():
    prompt = RealW8A8OptimizationPipeline._worker_prompt(
        WorkerAssignment("worker_0", 0, ["m2"]),
        "m2",
        {"M": 2, "N": 1536, "K": 4096},
        {"median_us": 129.0},
        Path("/tmp/worker"),
        2,
        None,
        [{
            "iteration": 1,
            "correctness_passed": False,
            "build_success": True,
            "speedup": 1.77,
            "artifact_dir": "iterations/m2/iteration1",
        }],
    )

    assert "repair the faster but incorrect candidate" in prompt
    assert "iterations/m2/iteration1" in prompt
    assert "Do not replace it with an unrelated architecture" in prompt


def test_m16_prompt_uses_exact_installed_dumma_api():
    prompt = RealW8A8OptimizationPipeline._worker_prompt(
        WorkerAssignment("worker_0", 0, ["m16"]),
        "m16",
        {"M": 16, "N": 1536, "K": 4096},
        {"median_us": 200.0},
        Path("/tmp/worker"),
        1,
        None,
    )

    assert "du::dumma::du_load_matrix_sync" in prompt
    assert "du::dumma::du_mma_sync" in prompt
    assert "du::dumma::mem_row_major" in prompt


def test_final_synthesis_prompt_requires_one_all_shape_extension():
    prompt = _final_synthesis_prompt(
        worker_inputs=[
            {
                "worker_id": "worker_0",
                "shapes": ["m2_wo_b", "m16_wo_b"],
                "source": "/tmp/worker_0/source",
                "results": {},
            }
        ],
        shapes=[
            {"id": "m2_wo_b", "M": 2, "N": 4096, "K": 2048},
            {"id": "m16_wo_b", "M": 16, "N": 4096, "K": 2048},
        ],
        source=Path("/tmp/final/source"),
        proposal_path=Path("/tmp/final/source/proposal.json"),
        previous_failure="correctness failed for m16_wo_b",
        fallback_shapes=[
            {"id": "m2_wq_b", "M": 2, "N": 8192, "K": 1024}
        ],
    )
    assert "one compiled extension" in prompt
    assert "generic scalar path as a correctness fallback" in prompt
    assert "edit only `csrc/w8a8_gemm_hip.hip`" in prompt
    assert "within 5%" in prompt
    assert "m2_wo_b" in prompt and "m16_wo_b" in prompt
    assert "outside this task's optimization scope" in prompt
    assert "m2_wq_b" in prompt
    assert "Previous trusted synthesis failure" in prompt


def test_prebuilt_dispatch_routes_shape_family_without_hip_code():
    dispatch = _render_prebuilt_dispatch([
        {
            "shape_id": "m16_wq_b",
            "shape": {"M": 16, "N": 8192, "K": 1024},
            "launch_symbol": "mi_m16_wq_b_launch_w8a8_gemm",
            "pack_symbol": "mi_m16_wq_b_launch_pack_w8a8_weight",
        },
        {
            "shape_id": "m16_wo_b",
            "shape": {"M": 16, "N": 4096, "K": 2048},
            "launch_symbol": "mi_m16_wo_b_launch_w8a8_gemm",
            "pack_symbol": "mi_m16_wo_b_launch_pack_w8a8_weight",
        },
    ])

    assert "if (n == 8192 && k == 1024)" in dispatch
    assert "if (n == 4096 && k == 2048)" in dispatch
    assert "mi_m16_wq_b_launch_w8a8_gemm" in dispatch
    assert "launch_w8a8_gemm(" in dispatch
    assert "__global__" not in dispatch
    assert "hipLaunchKernelGGL" not in dispatch


def test_artifact_symbol_prefix_is_valid_c_identifier():
    assert _artifact_symbol_prefix("m16-wq/b") == "mi_m16_wq_b_"


def test_snapshot_accepted_kernel_artifact_keeps_exact_object(tmp_path):
    worker = tmp_path / "worker"
    source = worker / "source" / "csrc"
    cache = (
        worker / "cache" / "torch" / "metainfer_w8a8_backend"
    )
    source.mkdir(parents=True)
    cache.mkdir(parents=True)
    source.joinpath("w8a8_gemm_hip.hip").write_bytes(b"accepted hip")
    cache.joinpath("w8a8_gemm_hip.cuda.o").write_bytes(b"accepted object")

    manifest = snapshot_accepted_kernel_artifact(
        worker_root=worker,
        shape_id="m16_wq_b",
        shape={"M": 16, "N": 8192, "K": 1024},
        metrics={
            "median_us": 30.1,
            "p90_us": 30.2,
            "graph_capture_passed": True,
        },
        commit="abc123",
    )

    assert (worker / manifest["source"]).read_bytes() == b"accepted hip"
    assert (worker / manifest["object"]).read_bytes() == b"accepted object"
    assert manifest["compile_target"] == "gfx928"
    assert manifest["commit"] == "abc123"


def test_compile_cache_is_content_addressed(tmp_path):
    worker = tmp_path / "worker"
    csrc = worker / "source" / "csrc"
    csrc.mkdir(parents=True)
    csrc.joinpath("bindings.cpp").write_text("// binding\n")
    hip = csrc / "w8a8_gemm_hip.hip"
    hip.write_text("// candidate one\n")

    runner = W8A8Runner(worker, 0)
    first = runner._prepare_compile_cache()
    repeated = runner._prepare_compile_cache()
    hip.write_text("// candidate two\n")
    second = runner._prepare_compile_cache()

    assert first["build_key"] == repeated["build_key"]
    assert first["compile_source_dir"] == repeated["compile_source_dir"]
    assert first["build_key"] != second["build_key"]
    assert Path(first["compile_source_dir"]).joinpath(
        "csrc/w8a8_gemm_hip.hip"
    ).read_text() == "// candidate one\n"


def _runner_with_source(tmp_path: Path) -> tuple[W8A8Runner, Path]:
    worker = tmp_path / "worker"
    csrc = worker / "source" / "csrc"
    csrc.mkdir(parents=True)
    csrc.joinpath("bindings.cpp").write_text("// binding\n")
    csrc.joinpath("w8a8_gemm_hip.hip").write_text("// kernel\n")
    return W8A8Runner(worker, 0), worker


def _fake_run_for_reference(records: list):
    """Return a _run stand-in that seeds the reference cache on demand."""

    def fake_run(command, *, cwd, env=None, timeout=None):
        records.append((list(command), timeout))
        if "--prepare-reference" in command:
            def arg(name):
                return command[command.index(name) + 1]
            cache_dir = Path(arg("--reference-cache-dir"))
            cache_dir.mkdir(parents=True, exist_ok=True)
            m, n, k = int(arg("--m")), int(arg("--n")), int(arg("--k"))
            (cache_dir / f"exact-int64-v1-m{m}-n{n}-k{k}.pt").write_bytes(
                b"ref"
            )
            stdout = (
                '{"reference_prepared": true, '
                '"reference_cache_hit": false}\n'
            )
        else:
            stdout = (
                '{"passed": true, "graph_capture_passed": true, '
                '"median_us": 1.0, "mismatch_count": 0}\n'
            )
        return SimpleNamespace(stdout=stdout, returncode=0, stderr="")

    return fake_run


def test_benchmark_prepares_reference_cache_when_missing(
    tmp_path, monkeypatch
):
    runner, worker = _runner_with_source(tmp_path)
    records: list = []
    monkeypatch.setattr(pipeline_module, "_run", _fake_run_for_reference(records))

    metrics = runner.benchmark(
        {"M": 4096, "N": 2304, "K": 6144}, check_correctness=True
    )

    # The reference is prepared first, outside the benchmark budget.
    assert len(records) == 2
    prepare_command, prepare_timeout = records[0]
    assert "--prepare-reference" in prepare_command
    assert prepare_timeout == _REFERENCE_PREPARE_TIMEOUT_S
    bench_command, bench_timeout = records[1]
    assert "--prepare-reference" not in bench_command
    assert bench_timeout == _BENCHMARK_TIMEOUT_S
    assert runner._reference_cache_path(4096, 2304, 6144).is_file()
    assert metrics["median_us"] == 1.0


def test_benchmark_reuses_existing_reference_cache(tmp_path, monkeypatch):
    runner, worker = _runner_with_source(tmp_path)
    path = runner._reference_cache_path(4096, 2304, 6144)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"ref")
    records: list = []
    monkeypatch.setattr(pipeline_module, "_run", _fake_run_for_reference(records))

    runner.benchmark(
        {"M": 4096, "N": 2304, "K": 6144}, check_correctness=True
    )

    assert len(records) == 1  # no prepare-reference call
    assert "--prepare-reference" not in records[0][0]


def test_benchmark_skips_reference_prep_when_correctness_disabled(
    tmp_path, monkeypatch
):
    runner, worker = _runner_with_source(tmp_path)
    records: list = []
    monkeypatch.setattr(pipeline_module, "_run", _fake_run_for_reference(records))

    runner.benchmark(
        {"M": 4096, "N": 2304, "K": 6144}, check_correctness=False
    )

    assert len(records) == 1
    command, _ = records[0]
    assert "--prepare-reference" not in command
    assert "--skip-correctness" in command


def test_reference_prep_failure_raises(tmp_path, monkeypatch):
    runner, worker = _runner_with_source(tmp_path)
    records: list = []

    def failing_run(command, *, cwd, env=None, timeout=None):
        records.append((list(command), timeout))
        return SimpleNamespace(
            stdout='{"reference_prepared": false}\n',
            returncode=0,
            stderr="",
        )

    monkeypatch.setattr(pipeline_module, "_run", failing_run)

    with pytest.raises(RuntimeError, match="reference cache preparation failed"):
        runner.benchmark(
            {"M": 4096, "N": 2304, "K": 6144}, check_correctness=True
        )


class _FakeTimelineStore:
    def __init__(self):
        self.events = []

    def append_timeline(self, type_: str, payload: dict):
        self.events.append((type_, payload))


def _perf_gate(benchmark, median_us, best=100.0, store=None, **kw):
    return _final_performance_gate(
        shape_id="minimax_tp8_qkv_proj_m16",
        best_median=best,
        metrics={"passed": True, "median_us": median_us},
        benchmark=benchmark,
        max_retries=kw.pop("max_retries", 3),
        retry_interval_s=kw.pop("retry_interval_s", 300),
        store=store or _FakeTimelineStore(),
    )


def test_perf_gate_accepts_without_retry():
    store = _FakeTimelineStore()
    result = _perf_gate(lambda: None, median_us=90.0, store=store)
    assert result["median_us"] == 90.0
    assert store.events == []


def test_perf_gate_retries_until_best_passes(monkeypatch):
    sleeps: list = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
    store = _FakeTimelineStore()
    calls = {"n": 0}

    def benchmark():
        calls["n"] += 1
        return {"passed": True, "median_us": 120.0 if calls["n"] == 1 else 95.0}

    result = _perf_gate(benchmark, median_us=130.0, store=store)
    # 130 (>105) -> retry1 120 (>105) -> retry2 95 (<=105) -> accept min 95
    assert calls["n"] == 2
    assert result["median_us"] == 95.0
    assert sleeps == [300, 300]
    assert [t for t, _ in store.events] == [
        "final_perf_gate_retry", "final_perf_gate_retry",
    ]
    assert store.events[0][1]["attempt"] == 1
    assert store.events[1][1]["attempt"] == 2


def test_perf_gate_fails_only_after_all_retries(monkeypatch):
    sleeps: list = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
    with pytest.raises(RuntimeError, match="after 3 re-measures"):
        _perf_gate(lambda: {"passed": True, "median_us": 200.0}, median_us=200.0)
    assert len(sleeps) == 3


def test_perf_gate_retry_correctness_failure_raises(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="correctness failed"):
        _perf_gate(
            lambda: {"passed": False, "median_us": 999.0},
            median_us=200.0,
        )


def test_iteration_archive_keeps_kernel_but_shares_api(tmp_path):
    source = tmp_path / "source"
    source.joinpath("csrc").mkdir(parents=True)
    source.joinpath("csrc", "w8a8_gemm_hip.hip").write_text(
        "// iteration kernel", encoding="utf-8"
    )
    source.joinpath("csrc", "bindings.cpp").write_text(
        "// bindings", encoding="utf-8"
    )
    source.joinpath("int8_w8a8_gemm_api.py").write_text(
        "# immutable shared API", encoding="utf-8"
    )
    destination = tmp_path / "iterations" / "m2" / "iteration1"

    archived = archive_iteration_candidate(source, destination)

    assert "csrc/w8a8_gemm_hip.hip" in archived
    assert (
        destination / "csrc" / "w8a8_gemm_hip.hip"
    ).read_text() == "// iteration kernel"
    assert (destination / "csrc" / "bindings.cpp").is_file()
    assert not (destination / "int8_w8a8_gemm_api.py").exists()


def test_iteration_archive_is_published_to_candidate_repo(tmp_path):
    assignment = WorkerAssignment("worker_0", 0, ["m2"])
    candidate = tmp_path / "main" / "candidates" / "worker_0"
    candidate.mkdir(parents=True)
    archived = tmp_path / "workers" / "worker_0" / "iterations" / (
        "m2/iteration1"
    )
    archived.joinpath("csrc").mkdir(parents=True)
    archived.joinpath("csrc", "w8a8_gemm_hip.hip").write_text(
        "// round one", encoding="utf-8"
    )

    destination = candidate_iteration_destination(
        tmp_path, assignment, "m2", 1
    )
    assert destination == candidate / "iteration1"
    assert destination is not None
    publish_iteration_candidate(archived, destination)

    saved = destination / "csrc" / "w8a8_gemm_hip.hip"
    assert saved.read_text(encoding="utf-8") == "// round one"
    assert not saved.is_symlink()


def test_coordinator_uses_minimal_file_tool_allowlist():
    assert _COORDINATOR_AGENT_ARGS == [
        "--tools", "Read,Glob,Grep,Write",
    ]


def test_bridge_path_translation_does_not_rewrite_workspaces_segment():
    """The /workspaces directory name must not be translated a second time."""
    from ..bridge.agent_bridge_server import (
        HOST_WORKSPACE_ROOT,
        _translate_prompt_paths,
    )

    raw = (
        b"/workspace/MetaInfer/nodes/worker29/workspaces/task/workers/"
        b"worker_0/source"
    )
    translated = _translate_prompt_paths(raw).decode()
    assert translated == (
        f"{HOST_WORKSPACE_ROOT}/MetaInfer/nodes/worker29/workspaces/"
        "task/workers/worker_0/source"
    )


def test_bridge_accepts_source_only_agent_tool_restrictions():
    from ..bridge.agent_bridge_server import _validated_args

    args = [
        "-p",
        "--setting-sources",
        "project,local",
        "--tools",
        "Read,Glob,Grep,Write,Edit",
    ]
    assert _validated_args(args) == args


def test_bridge_enforces_source_only_tools_for_older_orchestrators():
    from ..bridge.agent_bridge_server import _validated_args

    assert _validated_args(["-p"]) == [
        "-p",
        "--tools",
        "Read,Glob,Grep,Write,Edit",
    ]


def _write_test_api_contract(api_root):
    contract_dir = api_root / "int8w8a8gemm"
    contract_dir.mkdir(parents=True)
    contract = contract_dir / "int8_w8a8_gemm_api.py"
    contract.write_text(
        "\n".join([
            "def prepare_weight(*args): pass",
            "def allocate_workspace(*args): pass",
            "def validate_gemm_out_inputs(*args): pass",
            "def w8a8_gemm_out(*args): pass",
            "DEFAULT_OPTIMIZATION_SHAPES = (",
            "    {'id': 'm2', 'M': 2, 'N': 1536, 'K': 4096},",
            "    {'id': 'm16', 'M': 16, 'N': 1536, 'K': 4096},",
            ")",
            "def _check_target_shape(m, n, k):",
            "    if m > 16: raise ValueError('decode M must be <= 16')",
        ]),
        encoding="utf-8",
    )
    return contract


def test_fixed_api_rejects_shape_outside_decode_contract(
    tmp_path, monkeypatch
):
    api_root = tmp_path / "API"
    _write_test_api_contract(api_root)
    monkeypatch.setenv("METAINFER_OPERATOR_API_ROOT", str(api_root))
    pipeline = _make_pipeline(_shapes_only_req(), tmp_path)
    cfg = load_config(_shapes_only_req())
    with pytest.raises(ValueError, match="m64.*outside"):
        pipeline._validate_contract(cfg)


def test_prepare_uses_named_kernel_repo_and_stages_fixed_api(
    tmp_path, monkeypatch
):
    api_root = tmp_path / "API"
    kernel_root = tmp_path / "kernel-repos"
    contract = _write_test_api_contract(api_root)
    monkeypatch.setenv("METAINFER_OPERATOR_API_ROOT", str(api_root))
    monkeypatch.setenv("METAINFER_KERNEL_REPOS", str(kernel_root))
    req = _gen_req()
    req["answers"]["target_repo_path"] = "int8 test2"
    pipeline = _make_pipeline(req, tmp_path)
    cfg = load_config(req)
    pipeline._validate_contract(cfg)
    pipeline._prepare_worktrees(cfg, "gen-test")
    repo = kernel_root / "int8 test2"
    main = tmp_path / "workspace" / "main"
    assert cfg.target_repo_path == repo
    assert main.is_symlink()
    assert not main.readlink().is_absolute()
    assert main.resolve() == repo
    assert (repo / ".git").is_dir()
    staged = (
        main / "int8_w8a8_gemm_api.py"
    )
    assert staged.read_bytes() == contract.read_bytes()
    assert staged.stat().st_mode & 0o222 == 0
    backend = (repo / "w8a8_backend.py").read_text(encoding="utf-8")
    assert "def load_extension()" in backend
    assert "is_python_module=False" in backend
    assert (repo / "setup.py").is_file()
    assert (repo / "csrc" / "bindings.cpp").is_file()
    assert (repo / "w8a8_bench.py").is_file()
    assert (repo / "w8a8_graph.py").is_file()
    assert not (repo / "csrc" / "w8a8_gemm_hip.hip").exists()
    manifest = json.loads(
        (repo / "scaffold_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["task_id"] == "gen-test"
    assert manifest["fresh_repository"] is True
    assert manifest["implementation_inherited"] is False
    assert (
        manifest["initial_kernel"]
        == "pending_parallel_explore_child_generation"
    )


def test_new_task_rejects_existing_kernel_repo_without_continuation(
    tmp_path, monkeypatch
):
    api_root = tmp_path / "API"
    kernel_root = tmp_path / "kernel-repos"
    _write_test_api_contract(api_root)
    monkeypatch.setenv("METAINFER_OPERATOR_API_ROOT", str(api_root))
    monkeypatch.setenv("METAINFER_KERNEL_REPOS", str(kernel_root))
    existing = kernel_root / "int8 existing"
    existing.mkdir(parents=True)
    subprocess.run(
        ["git", "init"], cwd=existing, check=True, capture_output=True
    )
    (existing / "csrc").mkdir()
    (existing / "csrc" / "w8a8_gemm_hip.hip").write_text(
        "// existing kernel", encoding="utf-8"
    )
    req = _gen_req()
    req["answers"]["target_repo_path"] = "int8 existing"
    pipeline = _make_pipeline(req, tmp_path)
    cfg = load_config(req)
    pipeline._validate_contract(cfg)

    with pytest.raises(RuntimeError, match="explicit continuation"):
        pipeline._prepare_worktrees(cfg, "gen-test")


def test_gen_mode_uses_api_default_shapes_when_omitted(
    tmp_path, monkeypatch
):
    api_root = tmp_path / "API"
    _write_test_api_contract(api_root)
    monkeypatch.setenv("METAINFER_OPERATOR_API_ROOT", str(api_root))
    req = _gen_req()
    req["answers"].pop("shape_config")
    cfg = load_config(req)
    assert set(cfg.shapes) == {"m2", "m16"}
    assert len(cfg.assignments) == 1
    assert cfg.assignments[0].shape_ids == ["m2", "m16"]


def test_mock_mode_can_also_use_api_default_shapes(
    tmp_path, monkeypatch
):
    api_root = tmp_path / "API"
    _write_test_api_contract(api_root)
    monkeypatch.setenv("METAINFER_OPERATOR_API_ROOT", str(api_root))
    req = _gen_req()
    req["answers"]["execution_mode"] = "Mock (no GPU)"
    req["answers"].pop("shape_config")
    cfg = load_config(req)
    assert set(cfg.shapes) == {"m2", "m16"}
    assert cfg.assignments[0].worker_id == "worker_0"
