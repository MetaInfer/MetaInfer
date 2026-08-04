"""Validate plugin registration and import sanity."""

from metainfer.server.registry import all_plugins
from metainfer.orchestrator.tasks import all_tasks


def test_all_plugins_includes_sglang_trace_analyze():
    types = [p.type for p in all_plugins()]
    assert "sglang_trace_analyze" in types, (
        f"sglang_trace_analyze not found in registered plugins: {types}"
    )


def test_all_tasks_includes_sglang_trace_analyze():
    task_types = [p.task_type for p in all_tasks()]
    assert "sglang_trace_analyze" in task_types, (
        f"sglang_trace_analyze not found in registered tasks: {task_types}"
    )
