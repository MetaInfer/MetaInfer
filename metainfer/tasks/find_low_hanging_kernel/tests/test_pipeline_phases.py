"""End-to-end pipeline test using MockAgentManager.

Drives a full Pipeline.run() with mocked sub-agents that write canned
artifacts to disk (memory files + flow_graph.json). Asserts the pipeline
transitions all the way to P4_visualize and produces the expected outputs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict

from metainfer.orchestrator.state import StateStore
from metainfer.testing.mock_agent import MockAgentManager

from metainfer.tasks.find_low_hanging_kernel.orchestrator.pipeline import (
    OrchestratorConfig,
    Pipeline,
)
from metainfer.tasks.find_low_hanging_kernel.tests._helpers import (
    make_minimal_valid_graph,
    make_requirements,
    make_small_trace_events,
    write_trace,
)


def _mock_response_fn_factory(tmp_path: Path):
    """Build a MockAgentManager response_fn that writes the right canned
    artifacts based on the agent's role."""

    def response_fn(spec) -> str:
        role = getattr(spec, "role", "")
        wd = Path(spec.workdir)

        if role in ("step1_analyst", "step2_analyst"):
            # Cross-validation pool worker — write report.md into the workdir.
            (wd / "report.md").write_text(
                f"# Report from {spec.name}\n\nstub findings\n", encoding="utf-8"
            )
            return f"report written by {spec.name}"

        if role == "step1_synthesizer":
            # Synthesizer is supposed to write step1_code_analysis.md to the
            # path called out in the prompt. Extract that path from the prompt
            # text (the prompt file lives at workdir/{name}.prompt.txt).
            return "ok"

        if role == "step2_synthesizer":
            return "ok"

        if role == "graph_builder":
            return "ok"

        if role == "node_validator":
            # Validator pool worker: always return clean.
            return json.dumps({"n01": {"ok": True}, "n02": {"ok": True}, "n03": {"ok": True}})

        return "ok"

    return response_fn


def _patch_synthesizers_to_write_outputs(tmp_path: Path, manager):
    """The synthesizer prompts tell the agent to write to specific paths.
    Since the mock doesn't parse prompts, we monkey-patch `manager.launch`
    to intercept synthesizer / builder specs and write the canned output."""

    real_launch = manager.launch

    def patched_launch(spec):
        # Read the prompt to find target output paths.
        prompt_path = Path(spec.prompt_file)
        prompt_text = prompt_path.read_text(encoding="utf-8") if prompt_path.is_file() else ""

        role = getattr(spec, "role", "")
        wd = Path(spec.workdir)

        if role == "step1_synthesizer":
            # The synthesizer is instructed to write to step1_code_analysis.md
            # under workspace_dir/memory/. We can derive it from workdir:
            # workdir = memory/build/step1/synthesizer → memory is 3 levels up.
            memory_dir = wd.parent.parent.parent
            target = memory_dir / "step1_code_analysis.md"
            target.write_text("# Step 1 (mock)\n\nstub\n", encoding="utf-8")
        elif role == "step2_synthesizer":
            memory_dir = wd.parent.parent.parent
            target = memory_dir / "step2_tracing_analysis.md"
            target.write_text("# Step 2 (mock)\n\nstub\n", encoding="utf-8")
        elif role == "graph_builder":
            # Builder writes to workspace/flow_graph.json (prompt contains the
            # full path).
            m = re.search(r"write to `([^`]+flow_graph\.json)`", prompt_text)
            if m:
                target = Path(m.group(1))
            else:
                target = wd.parent.parent.parent.parent / "flow_graph.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(make_minimal_valid_graph()), encoding="utf-8"
            )
        return real_launch(spec)

    manager.launch = patched_launch


def test_pipeline_runs_end_to_end(tmp_path: Path):
    # --- Lay out inputs ---
    trace_path = tmp_path / "inputs" / "trace.json"
    write_trace(trace_path, make_small_trace_events())
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"arch": "stub"}), encoding="utf-8"
    )
    fw_dir = tmp_path / "framework"
    fw_dir.mkdir()
    (fw_dir / "norm.py").write_text("# stub\n", encoding="utf-8")

    state_dir = tmp_path / "state"
    workspace_dir = tmp_path / "workspace"
    memory_dir = workspace_dir / "memory"
    validation_dir = workspace_dir / "validation"
    inputs_snapshot_dir = workspace_dir / "inputs_snapshot"
    logs_root = state_dir / "logs"
    for p in (state_dir, workspace_dir, memory_dir, validation_dir, inputs_snapshot_dir, logs_root):
        p.mkdir(parents=True, exist_ok=True)

    # --- Build request + config ---
    req = make_requirements(form={
        "trace_file": str(trace_path),
        "model_dir": str(model_dir),
        "framework_source_dir": str(fw_dir),
        "startup_log": "",
        "cli_args_and_env": "--tp 1",
        "max_validator_rounds": 3,
    })
    cfg = OrchestratorConfig(
        workspace_dir=workspace_dir,
        memory_dir=memory_dir,
        validation_dir=validation_dir,
        inputs_snapshot_dir=inputs_snapshot_dir,
        repo_root=tmp_path,
        state_dir=state_dir,
        logs_root=logs_root,
        max_validator_rounds=3,
        user_paths=[trace_path, model_dir, fw_dir],
    )

    manager = MockAgentManager(response_fn=_mock_response_fn_factory(tmp_path))
    _patch_synthesizers_to_write_outputs(tmp_path, manager)

    store = StateStore(state_dir)
    pipeline = Pipeline(req=req, store=store, cfg=cfg, manager=manager)
    pipeline.run()

    # --- Assertions ---
    run = json.loads((state_dir / "run.json").read_text(encoding="utf-8"))
    assert run["finished"] is True
    assert run["final_status"] == "success", run
    assert run["current_phase"] == "finished"

    # All four major artifacts exist.
    assert (memory_dir / "step1_code_analysis.md").is_file()
    assert (memory_dir / "step2_tracing_analysis.md").is_file()
    assert (workspace_dir / "flow_graph.json").is_file()
    assert (workspace_dir / "flow_graph.html").is_file()
    assert (workspace_dir / "trace_parsed.json").is_file()

    # At least one validation round record.
    iters = list((state_dir / "iterations").glob("*.json"))
    assert iters

    # The rendered HTML embeds our graph.
    html = (workspace_dir / "flow_graph.html").read_text(encoding="utf-8")
    assert "rms_norm_kernel" in html
    assert "ELK" in html


def test_pipeline_resume_skips_completed_phases(tmp_path: Path):
    """Re-running a pipeline whose outputs already exist should be a no-op
    that immediately transitions to finished."""
    state_dir = tmp_path / "state"
    workspace_dir = tmp_path / "workspace"
    memory_dir = workspace_dir / "memory"
    validation_dir = workspace_dir / "validation"
    inputs_snapshot_dir = workspace_dir / "inputs_snapshot"
    logs_root = state_dir / "logs"
    for p in (state_dir, workspace_dir, memory_dir, validation_dir, inputs_snapshot_dir, logs_root):
        p.mkdir(parents=True, exist_ok=True)

    # Pre-populate all the outputs.
    (memory_dir / "step1_code_analysis.md").write_text("# stub", encoding="utf-8")
    (memory_dir / "step2_tracing_analysis.md").write_text("# stub", encoding="utf-8")
    (workspace_dir / "flow_graph.json").write_text(
        json.dumps(make_minimal_valid_graph()), encoding="utf-8"
    )
    # Render the HTML so _resume_phase returns "finished".
    from metainfer.tasks.find_low_hanging_kernel.orchestrator.visualizer import (
        render_html,
    )
    render_html(
        make_minimal_valid_graph(),
        out_path=workspace_dir / "flow_graph.html",
    )

    req = make_requirements()
    cfg = OrchestratorConfig(
        workspace_dir=workspace_dir, memory_dir=memory_dir,
        validation_dir=validation_dir, inputs_snapshot_dir=inputs_snapshot_dir,
        repo_root=tmp_path, state_dir=state_dir, logs_root=logs_root,
    )
    manager = MockAgentManager(response_fn=lambda spec: "ok")
    store = StateStore(state_dir)
    pipeline = Pipeline(req=req, store=store, cfg=cfg, manager=manager)
    pipeline.run()

    run = json.loads((state_dir / "run.json").read_text(encoding="utf-8"))
    assert run["finished"] is True
    # No agents should have been launched for analysis — only the no-op path.
    # (shutdown + initial init_or_resume don't launch anything.)
    assert manager.launched_specs == []
