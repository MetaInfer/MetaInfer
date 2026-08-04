"""WebPlugin for sglang_trace_analyze — registers routes + detail view."""

from __future__ import annotations

from pathlib import Path

from metainfer.server.registry import WebPlugin, register

from .routes import build_router

PLUGIN_TYPE = "sglang_trace_analyze"
_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "static"
_STATIC_PREFIX = f"/static/plugins/{PLUGIN_TYPE}"

_IMPORTMAP_ENTRIES: dict = {}

plugin = WebPlugin(
    type=PLUGIN_TYPE,
    label="SGLang Trace Analyze",
    description=(
        "Profile a model with SGLang's torch profiler across multiple batch "
        "sizes, then analyze kernel hotspots, TFLOPS/MFU, operator-to-model-"
        "structure mapping, fuse opportunities, and generate LLM-powered "
        "optimization hints."
    ),
    build_router=build_router,
    frontend_dir=_FRONTEND_DIR,
    importmap_entries=_IMPORTMAP_ENTRIES,
)

register(plugin)
