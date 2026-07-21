"""Web plugin registration for ``opt_GEMM_kernel``."""

from pathlib import Path

from metainfer.server.registry import WebPlugin, register

from ._qa import CONFIG
from .routes import build_router


PLUGIN_TYPE = "opt_GEMM_kernel"
_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "static"

plugin = WebPlugin(
    type=PLUGIN_TYPE,
    label="Optimize GEMM kernel",
    description=(
        "Select a kernel path and GPU profile, then optimize GEMM with frozen "
        "correctness tests and latency/TFLOPS/bandwidth profiling."
    ),
    build_router=build_router,
    detail_view_module="app/gemm-arena-detail",
    qa_config=CONFIG,
    frontend_dir=_FRONTEND_DIR,
    extra_stylesheets=["gemm-arena.css"],
)

register(plugin)
