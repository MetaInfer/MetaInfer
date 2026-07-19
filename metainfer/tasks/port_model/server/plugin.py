"""WebPlugin descriptor for port-model."""

from pathlib import Path

from metainfer.server.registry import WebPlugin

from ._qa import QA_CONFIG
from .routes import build_router

plugin = WebPlugin(
    type="port-model",
    label="Port Model",
    description="Port a model to a target inference framework (vLLM, SGLang, …)",
    detail_view_module="app/pm-detail",
    qa_config=QA_CONFIG,
    build_router=build_router,
    frontend_dir=Path(__file__).resolve().parent.parent / "static",
    extra_stylesheets=["pm.css"],
)
