"""TaskPlugin descriptor for port-model."""

from metainfer.orchestrator.tasks.base import TaskPlugin

DIAGNOSTIC_GLOBS = ("*.md", "*.json", "*.prompt.txt", "*.patch")

PLUGIN = TaskPlugin(
    task_type="port-model",
    cli_module="metainfer.tasks.port_model.orchestrator.cli",
    phases_module="metainfer.tasks.port_model.orchestrator.phases",
    diagnostic_globs=DIAGNOSTIC_GLOBS,
)
