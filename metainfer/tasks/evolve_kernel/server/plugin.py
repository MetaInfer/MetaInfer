"""WebPlugin registration for evolve-kernel .

LLM-guided iterative GPU kernel optimization.
"""

from pathlib import Path

from metainfer.server.registry import WebPlugin, register

from ._qa import CONFIG as _QA_CONFIG
from .routes import build_router

PLUGIN_TYPE = "evolve-kernel"

plugin = WebPlugin(
    type=PLUGIN_TYPE,
    label="Optimize GPU Kernel (LLM-guided)",
    description=(
        "Feed in a Triton GEMM kernel — the LLM generates test harnesses "
        "and iteratively optimizes it for GPU performance."
    ),
    detail_view_module="app/ok-detail",
    qa_config=_QA_CONFIG,
    build_router=build_router,
    frontend_dir=Path(__file__).resolve().parent.parent / "static",
    importmap_entries={},
    extra_stylesheets=["ok.css"],
)

register(plugin)
