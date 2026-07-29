"""WebPlugin registration for the knowledge-evolution task type."""

from pathlib import Path

from metainfer.server.registry import WebPlugin, register

from ._qa import KEEvolutionQAConfig
from .routes import build_router

_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "static"

plugin = WebPlugin(
    type="knowledge-evolution",
    label="Knowledge Evolution",
    description=(
        "4-phase evolution loop (attempt_pure->enrich->consolidate->verify_final). "
        "Evolves the knowledge base so a target model's inference framework can be "
        "generated from notebooks/ alone, without referencing open-source framework "
        "source code."
    ),
    detail_view_module="app/ke-detail",
    frontend_dir=_FRONTEND_DIR,
    qa_config=KEEvolutionQAConfig(),
    build_router=build_router,
    extra_stylesheets=["ke.css"],
)
register(plugin)
