"""Task plugin descriptor."""

from metainfer.orchestrator.tasks.base import TaskPlugin


PLUGIN = TaskPlugin(
    task_type="dcu-kernel-auto-opt",
    cli_module="metainfer.tasks.dcu_kernel_auto_opt.orchestrator.cli",
    phases_module="metainfer.tasks.dcu_kernel_auto_opt.orchestrator.phases",
    diagnostic_globs=("*.json", "*.jsonl", "*.log"),
)
