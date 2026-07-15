"""WebPlugin registration for opt-kernel."""

from pathlib import Path

from metainfer.server.registry import WebPlugin, register

from ._qa import CONFIG as _QA_CONFIG
from .routes import build_router

PLUGIN_TYPE = "opt-kernel"

plugin = WebPlugin(
    type=PLUGIN_TYPE,
    label="Optimize GPU kernel",
    description=(
        "Optimize an existing GPU kernel (attention, GEMM, norm, RoPE) "
        "for a specific shape and platform."
    ),
    detail_view_module="app/ok-detail",
    qa_config=_QA_CONFIG,
    build_router=build_router,
    frontend_dir=Path(__file__).resolve().parent.parent / "static",
    importmap_entries={},
)

register(plugin)
