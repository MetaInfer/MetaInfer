"""TaskPlugin — static metadata for the knowledge-evolution orchestrator.

The launcher reads ``cli_module`` and runs::

    python -m <cli_module> run <requirements.json> --state-dir … --workspace-dir …

``phases_module`` points to the state-machine definition — used by the WebUI's
state-graph endpoint.
"""

from metainfer.orchestrator.tasks.base import TaskPlugin

PLUGIN = TaskPlugin(
    task_type="knowledge-evolution",
    cli_module="metainfer.tasks.knowledge_evolution.orchestrator.cli",
    phases_module="metainfer.tasks.knowledge_evolution.orchestrator.phases",
    diagnostic_globs=("*.jsonl", "*.log"),
)
