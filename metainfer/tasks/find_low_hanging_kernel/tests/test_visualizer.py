"""Tests for the visualizer."""

from __future__ import annotations

import json
from pathlib import Path

from metainfer.tasks.find_low_hanging_kernel.orchestrator import visualizer
from metainfer.tasks.find_low_hanging_kernel.tests._helpers import (
    make_minimal_valid_graph,
)


def test_render_html_writes_self_contained_file(tmp_path: Path):
    graph = make_minimal_valid_graph()
    out = tmp_path / "flow_graph.html"
    result = visualizer.render_html(graph, out_path=out)
    assert result == out
    assert out.is_file()
    text = out.read_text(encoding="utf-8")

    # The embedded JSON data block is present.
    assert 'id="flow-graph-data"' in text
    # ELK bundle was inlined (look for the ELK global constructor pattern).
    assert "ELK" in text
    # All three node roles appear in the embedded JSON.
    assert "embedding_kernel" in text
    assert "rms_norm_kernel" in text
    assert "logits_kernel" in text
    # Embedded JSON parses.
    start = text.find('<script type="application/json" id="flow-graph-data">')
    end = text.find("</script>", start)
    blob = text[start + len('<script type="application/json" id="flow-graph-data">'):end].strip()
    parsed = json.loads(blob)
    assert parsed["metadata"]["model"] == "TestModel"


def test_render_from_files_roundtrip(tmp_path: Path):
    graph = make_minimal_valid_graph()
    gpath = tmp_path / "flow_graph.json"
    gpath.write_text(json.dumps(graph), encoding="utf-8")
    out = tmp_path / "flow_graph.html"
    visualizer.render_from_files(graph_path=gpath, out_html_path=out)
    assert out.is_file()


def test_render_raises_on_missing_placeholder(tmp_path: Path):
    bad_template = tmp_path / "bad.html"
    bad_template.write_text("<html></html>", encoding="utf-8")
    try:
        visualizer.render_html(
            make_minimal_valid_graph(),
            out_path=tmp_path / "out.html",
            template_path=bad_template,
        )
    except ValueError as e:
        assert "placeholder" in str(e)
    else:
        raise AssertionError("expected ValueError on missing placeholder")


def test_write_graph_json(tmp_path: Path):
    graph = make_minimal_valid_graph()
    out = tmp_path / "flow_graph.json"
    visualizer.write_graph_json(graph, out_path=out)
    parsed = json.loads(out.read_text(encoding="utf-8"))
    assert parsed["schema_version"] == 1
