"""gen-cpp-infer-framework plugin tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from metainfer.orchestrator.tasks import get_task
from metainfer.tasks.gen_cpp_infer_framework.server._qa import (
    GenCppInferQAConfig,
    _resolve_events_file,
)
from metainfer.tasks.gen_cpp_infer_framework.orchestrator.orchestrator import (
    _task_subdirs,
)
from metainfer.tasks.gen_cpp_infer_framework.orchestrator.iteration_record import (
    IterationRecord,
)
from metainfer.tasks.gen_cpp_infer_framework.orchestrator.pipeline import (
    _load_iter,
    _write_iter,
)
from metainfer.orchestrator.state import StateStore
from metainfer.tasks.gen_cpp_infer_framework.orchestrator.hardware import (
    execution_environment,
    materialize_hardware_binding,
    profiler_launch_command,
    render_hardware_profile,
    resolve_hardware_profile,
)
from metainfer.server.registry import get as _get_web_plugin


def test_feature_picker_only_contains_optional_runtime_capabilities():
    form_path = Path(__file__).parents[1] / "form.yaml"
    fields = yaml.safe_load(form_path.read_text(encoding="utf-8"))
    features = next(field for field in fields if field["key"] == "features")
    labels = {option["label"] for option in features["options"]}

    assert "Native C++ HTTP server" not in labels
    assert "CMake build" not in labels
    assert labels == {
        "Paged KV cache",
        "Continuous batching",
        "Tensor parallelism",
        "Speculative decoding",
    }


def test_task_plugin_registered():
    plugin = get_task("gen-cpp-infer-framework")
    assert plugin.cli_module == (
        "metainfer.tasks.gen_cpp_infer_framework.orchestrator.cli"
    )
    assert plugin.phases_module == (
        "metainfer.tasks.gen_cpp_infer_framework.orchestrator.phases"
    )


def test_web_plugin_registered():
    plugin = _get_web_plugin("gen-cpp-infer-framework")
    assert plugin is not None
    assert plugin.detail_view_module == "app/cpp-gf-detail"
    assert plugin.frontend_dir is not None and plugin.frontend_dir.exists()
    stem = plugin.detail_view_module.split("/", 1)[-1]
    assert (plugin.frontend_dir / f"{stem}.js").exists()
    assert plugin.build_router is not None
    route_paths = {route.path for route in plugin.build_router(plugin).routes}
    assert "/iterations" in route_paths
    assert "/charts" in route_paths
    assert "/state-graph" in route_paths
    assert plugin.qa_config is not None


def test_state_and_generated_workspace_are_separate(tmp_path: Path):
    state_dir = tmp_path / "metadata"
    workspace_dir = tmp_path / "generated"
    paths = _task_subdirs(state_dir, workspace_dir)

    assert paths["state_dir"] == state_dir
    assert paths["code_root"] == workspace_dir
    assert paths["logs_root"] == state_dir / "logs"
    assert paths["iterations_state"] == state_dir / "iterations"
    assert workspace_dir.is_dir()
    assert not (state_dir / "code").exists()


def test_cpp_iteration_record_round_trips_through_shared_store(tmp_path: Path):
    store = StateStore(tmp_path / "state")
    rec = IterationRecord(iteration=1, goal="compile on Z200")
    _write_iter(store, rec)

    raw = store.load_iteration(1)
    assert isinstance(raw, dict)
    loaded = _load_iter(store, 1)
    assert isinstance(loaded, IterationRecord)
    assert loaded.goal == "compile on Z200"


def test_cli_forwards_state_and_workspace_dirs(tmp_path: Path, monkeypatch):
    from metainfer.tasks.gen_cpp_infer_framework.orchestrator import cli
    from metainfer.tasks.gen_cpp_infer_framework.orchestrator import (
        orchestrator as orch,
    )

    requirements = tmp_path / "requirements.json"
    requirements.write_text("{}", encoding="utf-8")
    state_dir = tmp_path / "state"
    workspace_dir = tmp_path / "workspace"
    captured = {}

    def fake_run_with_requirements(requirements_path, **kwargs):
        captured["requirements_path"] = requirements_path
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(orch, "run_with_requirements", fake_run_with_requirements)
    assert cli.main([
        "run", str(requirements),
        "--state-dir", str(state_dir),
        "--workspace-dir", str(workspace_dir),
    ]) == 0
    assert captured["requirements_path"] == requirements
    assert captured["state_dir"] == state_dir
    assert captured["workspace_dir"] == workspace_dir


def test_cpp_task_uses_its_own_notebooks():
    from metainfer.tasks.gen_cpp_infer_framework.orchestrator import (
        orchestrator as orch,
    )

    notebooks_dir = orch._NOTEBOOKS_DIR
    assert notebooks_dir.name == "notebooks"
    assert "metainfer" in notebooks_dir.parts
    assert "tasks" in notebooks_dir.parts
    assert "gen_cpp_infer_framework" in notebooks_dir.parts
    assert notebooks_dir.is_dir()
    assert (notebooks_dir / "README.md").is_file()


def test_z200_hardware_profile_binds_system_build_and_profiler(tmp_path: Path):
    req = {"target_hardware": "Hygon Z200"}
    selected, profile = resolve_hardware_profile(req)
    assert selected == "Hygon Z200"
    assert profile is not None
    assert profile["build"]["cmake_cache"]["CMAKE_HIP_ARCHITECTURES"] == "gfx906"
    assert profile["profiling"]["profiler"] == "rocprof"
    snapshot = materialize_hardware_binding(req, tmp_path)
    assert snapshot.is_file()
    build_sh = tmp_path / "build.sh"
    assert "SYSTEM-OWNED FILE" in build_sh.read_text(encoding="utf-8")
    assert "-DCMAKE_HIP_ARCHITECTURES=gfx906" in build_sh.read_text(encoding="utf-8")
    # Re-materializing restores the system command path after an agent edit.
    build_sh.write_text("agent override", encoding="utf-8")
    materialize_hardware_binding(req, tmp_path)
    assert "agent override" not in build_sh.read_text(encoding="utf-8")
    env = execution_environment(req, tmp_path)
    assert env["METAINFER_HIPCC"] == "hipcc"
    assert profiler_launch_command(req) == ["rocprof", "--stats"]
    rendered = render_hardware_profile(req)
    assert "hipcc" in rendered
    assert "gfx906" in rendered
    assert "do not edit build.sh" in rendered


def test_qa_explicit_events_file(tmp_path: Path):
    cfg = GenCppInferQAConfig()
    events_file = tmp_path / "agent.events.jsonl"
    events_file.write_text("{}", encoding="utf-8")
    out = cfg.resolve_target(tmp_path, {"events_file": str(events_file)})
    assert out["events_file"] == events_file
    assert out["target_workdir"] is None
    assert "events_file=" in out["target_label"]


def test_qa_tuple_lookup_finds_events_file(tmp_path: Path):
    log_dir = tmp_path / "logs" / "001"
    log_dir.mkdir(parents=True)
    events_file = log_dir / "foo.attempt0.events.jsonl"
    events_file.write_text(json.dumps({"ok": True}), encoding="utf-8")
    cfg = GenCppInferQAConfig()
    out = cfg.resolve_target(tmp_path, {"iteration": 1, "agent": "foo"})
    assert out["events_file"] == events_file
    assert "iter=1" in out["target_label"]


def test_qa_tuple_lookup_falls_back_to_glob(tmp_path: Path):
    log_dir = tmp_path / "logs" / "002" / "nested" / "bar"
    log_dir.mkdir(parents=True)
    events_file = log_dir / "bar.attempt0.events.jsonl"
    events_file.write_text("{}", encoding="utf-8")
    assert _resolve_events_file(tmp_path, 2, "bar") == events_file


def test_qa_rejects_empty_payload(tmp_path: Path):
    cfg = GenCppInferQAConfig()
    with pytest.raises(ValueError):
        cfg.resolve_target(tmp_path, {})
