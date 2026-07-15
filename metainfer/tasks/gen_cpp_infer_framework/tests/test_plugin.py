"""gen-cpp-infer-framework plugin tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from metainfer.orchestrator.tasks import get_task
from metainfer.tasks.gen_cpp_infer_framework.web_server_handler._qa import (
    GenCppInferQAConfig,
    _resolve_events_file,
)
from metainfer.web.registry import get as _get_web_plugin


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
    assert "app/cpp-gf-detail" in plugin.importmap_entries
    assert plugin.qa_config is not None


def test_cpp_task_reuses_gen_infer_notebooks():
    from metainfer.tasks.gen_cpp_infer_framework.orchestrator import (
        orchestrator as orch,
    )

    notebooks_dir = orch._NOTEBOOKS_DIR
    assert notebooks_dir.name == "notebooks"
    assert "metainfer" in notebooks_dir.parts
    assert "tasks" in notebooks_dir.parts
    assert "gen_infer_framework" in notebooks_dir.parts
    assert notebooks_dir.is_dir()
    assert (notebooks_dir / "00_contracts").is_dir()


def test_qa_explicit_events_file(tmp_path: Path):
    cfg = GenCppInferQAConfig()
    events_file = tmp_path / "agent.events.jsonl"
    events_file.write_text("{}", encoding="utf-8")
    out = cfg.resolve_target(tmp_path, {"events_file": str(events_file)})
    assert out["events_file"] == events_file
    assert out["target_workdir"] is None
    assert "events_file=" in out["target_label"]


def test_qa_tuple_lookup_finds_events_file(tmp_path: Path):
    log_dir = tmp_path / "iterations" / "001" / ".metainfer-logs" / "foo"
    log_dir.mkdir(parents=True)
    events_file = log_dir / "foo.attempt0.events.jsonl"
    events_file.write_text(json.dumps({"ok": True}), encoding="utf-8")
    cfg = GenCppInferQAConfig()
    out = cfg.resolve_target(tmp_path, {"iteration": 1, "agent": "foo"})
    assert out["events_file"] == events_file
    assert "iter=1" in out["target_label"]


def test_qa_tuple_lookup_falls_back_to_glob(tmp_path: Path):
    log_dir = tmp_path / "iterations" / "002" / "nested" / "bar"
    log_dir.mkdir(parents=True)
    events_file = log_dir / "bar.attempt0.events.jsonl"
    events_file.write_text("{}", encoding="utf-8")
    assert _resolve_events_file(tmp_path, 2, "bar") == events_file


def test_qa_rejects_empty_payload(tmp_path: Path):
    cfg = GenCppInferQAConfig()
    with pytest.raises(ValueError):
        cfg.resolve_target(tmp_path, {})
