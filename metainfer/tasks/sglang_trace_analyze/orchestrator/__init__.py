"""Orchestrator (worker subprocess) for sglang_trace_analyze."""

from metainfer.orchestrator.tasks import register
from .plugin import PLUGIN

register(PLUGIN)
