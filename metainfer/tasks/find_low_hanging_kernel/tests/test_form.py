"""Tests for form.yaml + plugin registration."""

from __future__ import annotations

from pathlib import Path

import yaml

from metainfer.orchestrator.tasks import all_tasks, get_task
from metainfer.server.forms import load_form_schema, validate_submission
from metainfer.server.registry import all_plugins, get


PLUGIN_TYPE = "find-low-hanging-kernel"


def test_plugin_registered():
    types = [p.task_type for p in all_tasks()]
    assert PLUGIN_TYPE in types


def test_web_plugin_registered():
    types = [p.type for p in all_plugins()]
    assert PLUGIN_TYPE in types
    plugin = get(PLUGIN_TYPE)
    assert plugin.label
    assert plugin.detail_view_module == "app/flhk-detail"
    assert plugin.build_router is not None


def test_task_plugin_descriptor_fields():
    p = get_task(PLUGIN_TYPE)
    assert p.cli_module.endswith(".cli")
    assert p.phases_module.endswith(".phases")
    assert isinstance(p.diagnostic_globs, tuple)


def test_form_yaml_exists_and_loads():
    schema = load_form_schema(PLUGIN_TYPE)
    assert schema is not None
    keys = {f["key"] for f in schema["fields"]}
    assert {
        "trace_file", "model_dir", "framework_source_dir",
        "cli_args_and_env", "startup_log", "max_validator_rounds",
        "token_budget_max_cost_usd",
    } <= keys


def test_form_required_fields_marked():
    schema = load_form_schema(PLUGIN_TYPE)
    required = {f["key"] for f in schema["fields"] if f["required"]}
    assert {"trace_file", "model_dir", "framework_source_dir"} <= required


def test_validate_submission_rejects_missing_required():
    result = validate_submission(PLUGIN_TYPE, {})
    assert result["ok"] is False
    for k in ("trace_file", "model_dir", "framework_source_dir"):
        assert k in result["errors"]


def test_validate_submission_accepts_full_form():
    schema = load_form_schema(PLUGIN_TYPE)
    answers = {f["key"]: "x" for f in schema["fields"] if f["required"]}
    answers["max_validator_rounds"] = 5
    answers["cli_args_and_env"] = ""
    result = validate_submission(PLUGIN_TYPE, answers)
    assert result["ok"] is True, result["errors"]


def test_form_yaml_is_valid_yaml():
    # Sanity: form.yaml must parse.
    p = Path(__file__).resolve().parent.parent / "form.yaml"
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) >= 5
