"""Server plugin for the gen-cpp-infer-framework task."""

from __future__ import annotations

from pathlib import Path

from metainfer.server.registry import WebPlugin, register

from ._qa import CONFIG as _QA_CONFIG
from .routes import build_router

PLUGIN_TYPE = "gen-cpp-infer-framework"
_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "static"

plugin = WebPlugin(
    type=PLUGIN_TYPE,
    label="Build C++ inference framework",
    description=(
        "Build a model-specific native C++ inference server for the selected "
        "hardware profile and an OpenAI-compatible HTTP API."
    ),
    build_router=build_router,
    detail_view_module="app/cpp-gf-detail",
    detail_view_export="default",
    qa_config=_QA_CONFIG,
    frontend_dir=_FRONTEND_DIR,
    importmap_entries={},
    extra_stylesheets=["cpp-gf.css"],
)

register(plugin)
