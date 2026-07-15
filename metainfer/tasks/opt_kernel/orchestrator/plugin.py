"""TaskPlugin — static metadata for opt-kernel task type.

The launcher reads ``cli_module`` and runs::

    python -m <cli_module> run <requirements.json> --state-dir … --workspace-dir …
"""

from metainfer.orchestrator.tasks.base import TaskPlugin


PLUGIN = TaskPlugin(
    task_type="opt-kernel",
    cli_module="metainfer.tasks.opt_kernel.orchestrator.cli",
    phases_module="metainfer.tasks.opt_kernel.orchestrator.phases",
    diagnostic_globs=(
        "*-test.log",
        "test.log",
        "*.prompt.txt",
    ),
)
