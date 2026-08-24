import json

import pytest

from metainfer.orchestrator.state import StateStore

from ..orchestrator.evaluator.spec import FrozenEvaluatorBundle
from ..orchestrator.evaluator.champion import ChampionStore, make_report_reference
from ..orchestrator.guidance import GuidanceStore
from ..orchestrator.phases import graph_payload
from ..orchestrator.pipeline import (
    Orchestrator, OrchestratorConfig, _combine_hipprof_reports,
    _near_promotion_boundary,
)
from ._helpers import FakeBuilder, FakeManager, FakeProfiler, make_bundle


def test_state_graph_is_the_standard_six_phase_outer_loop():
    graph = graph_payload("S_baseline")
    expected = [
        "A_plan", "B_implement", "C_test", "D_review",
        "E_perf_test", "F_perf_plan",
    ]
    assert graph["order"] == expected
    assert [node["id"] for node in graph["nodes"]] == expected
    assert "S_baseline" not in graph["order"]


def test_one_iteration_promotes_challenger_without_old_task_dependencies(tmp_path):
    state = tmp_path / "state"
    workspace = tmp_path / "workspace"
    initial = tmp_path / "initial"
    initial.mkdir()
    (initial / "kernel.cpp").write_text("// baseline\n", encoding="utf-8")
    (initial / "submission.yaml").write_text(
        "schema_version: 1\nsources: [kernel.cpp]\n", encoding="utf-8"
    )
    bundle = FrozenEvaluatorBundle.materialize(make_bundle(tmp_path / "source"), state / "system_evaluator")
    manager = FakeManager()
    cfg = OrchestratorConfig(
        state_dir=state,
        iterations_root=workspace,
        logs_root=state / "logs",
        notebooks_dir=tmp_path,
        evaluator_bundle=bundle,
        system_builder=FakeBuilder(),
        profiler=FakeProfiler(),
        initial_submission=initial,
        max_iterations=1,
    )
    req = {"task_id": "gemm-test", "task_type": "opt_GEMM_kernel"}
    GuidanceStore(state / "guidance").submit(
        "Try a 128x128 tile, but preserve correctness on irregular K."
    )
    orch = Orchestrator(req, StateStore(state), cfg, manager)
    orch.run()

    record = json.loads((state / "iterations" / "001.json").read_text(encoding="utf-8"))
    champion = json.loads((state / "champion" / "champion.json").read_text(encoding="utf-8"))
    feedback = json.loads((state / "logs" / "001" / "feedback.json").read_text(encoding="utf-8"))
    assert record["promoted"] is True
    assert champion["iteration"] == 1
    assert "heldout" not in json.dumps(feedback)
    assert (state / "champion" / "submission" / "kernel.cpp").is_file()
    assert manager.shutdown_called
    assert list(record["phases"]) == [
        "A_plan", "B_implement", "C_test", "D_review",
        "E_perf_test", "F_perf_plan",
    ]
    assert (workspace / "iter_001" / "perf_plan.md").is_file() or any(
        path.name == "perf_plan.md" for path in workspace.rglob("perf_plan.md")
    )
    timeline = [
        json.loads(line)
        for line in (state / "timeline.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    baseline_index = next(i for i, event in enumerate(timeline) if event["type"] == "baseline_certified")
    agent_index = next(i for i, event in enumerate(timeline) if event["type"] == "agent_launch")
    assert baseline_index < agent_index
    baseline = json.loads(
        (state / "baseline" / "baseline-manifest.json").read_text(encoding="utf-8")
    )
    baseline_report = json.loads(
        (state / baseline["benchmark_report"]["path"]).read_text(encoding="utf-8")
    )
    baseline_profile = json.loads(
        (state / baseline["profile_report"]["path"]).read_text(encoding="utf-8")
    )
    assert baseline_report["evaluation_role"] == "baseline"
    assert baseline["implementation"] == "triton"
    assert baseline["build_fingerprint"] == "triton-jit"
    assert baseline_profile["profile_id"] == "hygon-k100-gfx928"
    initial_hip = json.loads(
        (state / "certified" / "initial-hip" / "initial-hip-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    initial_profile = json.loads(
        (state / initial_hip["profile_report"]["path"]).read_text(encoding="utf-8")
    )
    assert initial_hip["implementation"] == "initial-hip"
    assert initial_hip["correctness"]["summary"]["expected"] == 2
    assert initial_profile["profile_id"] == "hygon-k100-gfx928"
    record_profile = json.loads(
        (state / record["profile_report"]["path"]).read_text(encoding="utf-8")
    )
    assert record_profile["cases"][0]["vgpr_count"] == 32
    assert record["measurement_report"]["path"] == (
        "logs/001/candidate-benchmark-report.json"
    )
    assert feedback["benchmark"]["hardware_profile"]["gpu_arch"] == "gfx928"
    assert "Try a 128x128 tile" in manager.prompts["planner"]
    assert '"entrypoint": "launch_gemm"' in manager.prompts["planner"]
    assert '"benchmark_shapes"' in manager.prompts["planner"]
    guidance = GuidanceStore(state / "guidance").snapshot()
    assert guidance["pending_count"] == 0
    assert guidance["items"][0]["applied_role"] == "planner"
    assert any(event["type"] == "human_guidance_applied" for event in timeline)

    reloaded = ChampionStore(
        state / "champion",
        noise_threshold=0.01,
        expected_case_ids=["small", "large"],
    ).load()
    assert reloaded["measurement_report"] == champion["measurement_report"]
    champion_report_path = state / champion["measurement_report"]["path"]
    original_report = champion_report_path.read_bytes()
    champion_report_path.write_text('{"passed": true, "cases": []}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="performance report changed"):
        ChampionStore(
            state / "champion",
            noise_threshold=0.01,
            expected_case_ids=["small", "large"],
        ).load()
    champion_report_path.write_bytes(original_report)

    (state / "champion" / "submission" / "kernel.cpp").write_text(
        "// tampered\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="changed outside promotion"):
        ChampionStore(
            state / "champion",
            noise_threshold=0.01,
            expected_case_ids=["small", "large"],
        ).load()


def test_champion_rejects_candidate_at_exact_noise_boundary(tmp_path):
    state = tmp_path / "state"
    baseline_path = state / "baseline" / "baseline-benchmark-report.json"
    baseline_path.parent.mkdir(parents=True)
    baseline_path.write_text(
        json.dumps({"cases": [{"id": "shape", "latency_ms": 1.0}]}),
        encoding="utf-8",
    )
    baseline_ref = make_report_reference(state, baseline_path)
    store = ChampionStore(
        state / "champion",
        noise_threshold=0.01,
        expected_case_ids=["shape"],
    )
    store.initialize_triton(baseline_ref)

    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()
    (candidate_dir / "kernel.cpp").write_text("// candidate\n", encoding="utf-8")
    candidate_path = state / "logs" / "001" / "candidate-benchmark-report.json"
    candidate_path.parent.mkdir(parents=True)
    candidate_path.write_text(
        json.dumps({"cases": [{"id": "shape", "latency_ms": 0.99}]}),
        encoding="utf-8",
    )
    candidate_ref = make_report_reference(state, candidate_path)

    promoted, reason, champion = store.consider(
        1, candidate_dir, candidate_ref, baseline_ref,
    )

    assert promoted is False
    assert "beyond noise threshold" in reason
    assert champion["kind"] == "triton"
    assert not store.submission_dir.exists()


def test_boundary_retest_combines_raw_hipprof_samples_without_shape_weighting():
    methodology = {"timer": "hipprof_gpu_kernel_duration_ns"}
    first = {
        "methodology": methodology,
        "cases": [{
            "id": "shape", "latency_ms": 1.0,
            "operator_samples_ms": [0.9, 1.1],
        }],
    }
    second = {
        "methodology": methodology,
        "cases": [{
            "id": "shape", "latency_ms": 0.9,
            "operator_samples_ms": [0.8, 1.0],
        }],
    }
    combined = _combine_hipprof_reports(first, second)
    case = combined["cases"][0]
    assert case["latency_ms"] == pytest.approx(0.95)
    assert case["sample_count"] == 4
    assert case["measurement_batches"] == 2


def test_one_percent_boundary_triggers_retest():
    incumbent = {"cases": [{"id": "shape", "latency_ms": 1.0}]}
    assert _near_promotion_boundary(
        incumbent, {"cases": [{"id": "shape", "latency_ms": 0.99}]}, 0.01
    )
    assert not _near_promotion_boundary(
        incumbent, {"cases": [{"id": "shape", "latency_ms": 0.8}]}, 0.01
    )
