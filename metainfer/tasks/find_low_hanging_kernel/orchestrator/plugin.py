"""TaskPlugin descriptor for find-low-hanging-kernel.

The launcher reads ``cli_module`` and runs::

    python -m <cli_module> run <requirements.json> --state-dir … --workspace-dir …
"""

from metainfer.orchestrator.tasks.base import TaskPlugin


PLUGIN = TaskPlugin(
    task_type="find-low-hanging-kernel",
    cli_module="metainfer.tasks.find_low_hanging_kernel.orchestrator.cli",
    phases_module="metainfer.tasks.find_low_hanging_kernel.orchestrator.phases",
    # Per-iteration diagnostic files we want copied forward into the next
    # round's prev-iter/ when Step 3 graph-validation loops.
    diagnostic_globs=(
        "*.md",
        "flow_graph.json",
        "*.prompt.txt",
        "integrity_fixes.json",
        "group_*.json",
    ),
)
