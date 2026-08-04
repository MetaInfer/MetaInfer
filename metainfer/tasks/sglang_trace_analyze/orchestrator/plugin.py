"""TaskPlugin descriptor for sglang_trace_analyze."""

from metainfer.orchestrator.tasks.base import TaskPlugin

PLUGIN = TaskPlugin(
    task_type="sglang_trace_analyze",
    cli_module="metainfer.tasks.sglang_trace_analyze.orchestrator.cli",
    phases_module="metainfer.tasks.sglang_trace_analyze.orchestrator.phases",
    diagnostic_globs=("*",),
)
