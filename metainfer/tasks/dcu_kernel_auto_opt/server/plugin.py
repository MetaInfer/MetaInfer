"""Web plugin registration."""

from pathlib import Path

from metainfer.server.registry import WebPlugin, register

from .routes import build_router


PLUGIN_TYPE = "dcu-kernel-auto-opt"
_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "static"

plugin = WebPlugin(
    type=PLUGIN_TYPE,
    label="DCU kernel auto-opt",
    description=(
        "Run isolated multi-agent, multi-GPU kernel optimization with "
        "trusted per-operator harnesses, including real gfx928 INT8 W8A8 GEMM."
    ),
    build_router=build_router,
    detail_view_module="app/dkao-detail",
    frontend_dir=_FRONTEND_DIR,
    extra_stylesheets=["dkao.css"],
)

register(plugin)
