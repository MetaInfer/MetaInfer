"""port-model orchestrator package."""

from metainfer.orchestrator.tasks import register

from .plugin import PLUGIN

register(PLUGIN)
