"""Orchestrator-side registration for knowledge-evolution.

Importing this module registers the TaskPlugin with the framework,
making the launcher aware of the ``knowledge-evolution`` task type.
"""

from metainfer.orchestrator.tasks import register
from .plugin import PLUGIN

register(PLUGIN)
