"""gen-infer-framework-cpp web plugin definition.

Registers the detail view + QA pathsolver for the multi-iteration
ABCDEF pipeline. Peer of
:mod:`metainfer.tasks.calc_value.server.plugin`.
"""

from __future__ import annotations

from pathlib import Path

from metainfer.server.registry import WebPlugin, register

from ._qa import CONFIG as _QA_CONFIG
from .routes import build_router

PLUGIN_TYPE = "gen-infer-framework-cpp"
# Frontend assets live in the task package's static/ dir (sibling of this
# package). Resolve via Path(__file__) so the package is relocatable.
_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "static"
_STATIC_PREFIX = f"/static/plugins/{PLUGIN_TYPE}"

# Importmap entries are auto-discovered: every ``*.js`` directly under
# ``_FRONTEND_DIR`` is registered under ``app/<stem>`` by ``create_app``.
# That covers ``app/cpp-gf-detail`` for free. We only need to populate this
# dict to OVERRIDE shell entries (e.g. ship a divergent ``app/state-graph``
# for this task type). The shell's default widgets (in
# ``metainfer/tasks/sys_shell/static/components/``) are inherited as-is.
_IMPORTMAP_ENTRIES: dict = {}

plugin = WebPlugin(
    type=PLUGIN_TYPE,
    label="构建 C++ 推理框架",
    description=(
        "以 C++ 为主构建面向 Qwen3 的原生推理框架，支持海光 DTK/HIP "
        "硬件探测、OpenAI-compatible HTTP 服务与持续性能优化。"
    ),
    build_router=build_router,
    detail_view_module="app/cpp-gf-detail",
    detail_view_export="default",
    qa_config=_QA_CONFIG,
    frontend_dir=_FRONTEND_DIR,
    importmap_entries=_IMPORTMAP_ENTRIES,
    extra_stylesheets=["cpp-gf.css"],
)

register(plugin)
