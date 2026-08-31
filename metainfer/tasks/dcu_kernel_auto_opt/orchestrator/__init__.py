"""Orchestrator registration for dcu-kernel-auto-opt."""

from metainfer.orchestrator.tasks import register

from .plugin import PLUGIN

register(PLUGIN)

__version__ = "0.1.0"
