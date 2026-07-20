"""port-model server package."""

from metainfer.server.registry import register

from .plugin import plugin

register(plugin)
