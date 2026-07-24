import json

import pytest

from metainfer.orchestrator.state import StateStore

from ..orchestrator.evaluator.spec import FrozenEvaluatorBundle
from ..orchestrator.evaluator.champion import ChampionStore
from ..orchestrator.guidance import GuidanceStore
from ..orchestrator.phases import graph_payload
from ..orchestrator.pipeline import Orchestrator, OrchestratorConfig
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
    assert baseline["benchmark"]["evaluation_role"] == "baseline"
    assert baseline["implementation"] == "triton"
    assert baseline["build_fingerprint"] == "triton-jit"
    assert baseline["hardware_profile"]["profile_id"] == "hygon-k100-gfx928"
    initial_hip = json.loads(
        (state / "certified" / "initial-hip" / "initial-hip-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert initial_hip["implementation"] == "initial-hip"
    assert initial_hip["correctness"]["summary"]["expected"] == 2
    assert initial_hip["hardware_profile"]["profile_id"] == "hygon-k100-gfx928"
    assert record["hardware_profile"]["cases"][0]["vgpr_count"] == 32
    assert feedback["benchmark"]["hardware_profile"]["gpu_arch"] == "gfx928"
    assert "Try a 128x128 tile" in manager.prompts["planner"]
    assert '"entrypoint": "launch_gemm"' in manager.prompts["planner"]
    assert '"benchmark_shapes"' in manager.prompts["planner"]
    guidance = GuidanceStore(state / "guidance").snapshot()
    assert guidance["pending_count"] == 0
    assert guidance["items"][0]["applied_role"] == "planner"
    assert any(event["type"] == "human_guidance_applied" for event in timeline)

    (state / "champion" / "submission" / "kernel.cpp").write_text(
        "// tampered\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="changed outside promotion"):
        ChampionStore(state / "champion", noise_threshold=0.01).load()
