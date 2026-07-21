"""Orchestrator registration for :mod:`opt_GEMM_kernel`."""

from metainfer.orchestrator.tasks import register

from .plugin import PLUGIN

register(PLUGIN)

__version__ = "0.1.0"

