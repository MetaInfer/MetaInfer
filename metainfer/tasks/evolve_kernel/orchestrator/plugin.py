"""TaskPlugin — static metadata for evolve-kernel task type.

The launcher reads ``cli_module`` and runs::

    python -m <cli_module> run <requirements.json> --state-dir … --workspace-dir …
"""

from metainfer.orchestrator.tasks.base import TaskPlugin


PLUGIN = TaskPlugin(
    task_type="evolve-kernel",
    cli_module="metainfer.tasks.evolve_kernel.orchestrator.cli",
    phases_module="metainfer.tasks.evolve_kernel.orchestrator.phases",
    diagnostic_globs=(
        "*-test.log",
        "test.log",
        "*.prompt.txt",
    ),
)
