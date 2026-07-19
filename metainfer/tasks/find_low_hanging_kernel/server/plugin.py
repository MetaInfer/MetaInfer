"""WebPlugin registration for find-low-hanging-kernel."""

from pathlib import Path

from metainfer.server.registry import WebPlugin, register

from ._qa import CONFIG as _QA_CONFIG
from .routes import build_router

PLUGIN_TYPE = "find-low-hanging-kernel"

plugin = WebPlugin(
    type=PLUGIN_TYPE,
    label="Find Low-Hanging Kernel",
    description=(
        "Analyze a Chrome trace + model directory + inference framework source "
        "to build an auditable execution-flow graph and identify the kernels "
        "with the most optimization headroom."
    ),
    detail_view_module="app/flhk-detail",
    qa_config=_QA_CONFIG,
    build_router=build_router,
    frontend_dir=Path(__file__).resolve().parent.parent / "static",
    importmap_entries={},
    extra_stylesheets=["flhk.css"],
)

register(plugin)
