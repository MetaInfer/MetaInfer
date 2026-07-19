"""Deterministic visualizer (Step 4).

Reads the validated ``flow_graph.json`` and substitutes it into the standalone
``flow_graph_template.html`` (which inlines ELK.js + the client-side renderer)
to produce a self-contained ``flow_graph.html`` that works via ``file://``.

The template lives at ``static/flow_graph_template.html`` and contains the
literal placeholder ``/*__FLOW_GRAPH_JSON__*/`` where the graph JSON is
substituted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


PLACEHOLDER_GRAPH = "/*__FLOW_GRAPH_JSON__*/"
PLACEHOLDER_ELK = "/*__ELK_BUNDLE__*/"


def _static_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "static"


def render_html(
    graph: Dict[str, Any],
    *,
    out_path: Path,
    template_path: Path | None = None,
    elk_bundle_path: Path | None = None,
) -> Path:
    """Substitute ``graph`` + the ELK bundle into the template and write the
    result to ``out_path``. The output is fully self-contained — it works via
    ``file://``. Returns ``out_path``."""
    if template_path is None:
        template_path = _static_dir() / "flow_graph_template.html"
    if elk_bundle_path is None:
        elk_bundle_path = _static_dir() / "vendor" / "elk.bundled.js"
    template = template_path.read_text(encoding="utf-8")
    if PLACEHOLDER_GRAPH not in template:
        raise ValueError(
            f"template {template_path} is missing the {PLACEHOLDER_GRAPH!r} placeholder"
        )
    if PLACEHOLDER_ELK not in template:
        raise ValueError(
            f"template {template_path} is missing the {PLACEHOLDER_ELK!r} placeholder"
        )

    # Inline the ELK bundle as a JS blob.
    elk_js = elk_bundle_path.read_text(encoding="utf-8")
    html = template.replace(PLACEHOLDER_ELK, elk_js)

    graph_json = json.dumps(graph, indent=2, ensure_ascii=False)
    # Embed as a JSON island that the renderer reads via document.getElementById.
    replacement = (
        '<script type="application/json" id="flow-graph-data">\n'
        f"{graph_json}\n"
        "</script>"
    )
    html = html.replace(PLACEHOLDER_GRAPH, replacement)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def render_from_files(
    *,
    graph_path: Path,
    out_html_path: Path,
    template_path: Path | None = None,
) -> Path:
    """Convenience: read ``graph_path`` and render to ``out_html_path``."""
    graph = json.loads(Path(graph_path).read_text(encoding="utf-8"))
    return render_html(graph, out_path=out_html_path, template_path=template_path)


def write_graph_json(graph: Dict[str, Any], *, out_path: Path) -> Path:
    """Write the validated graph as flow_graph.json (auditable artifact)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out_path
