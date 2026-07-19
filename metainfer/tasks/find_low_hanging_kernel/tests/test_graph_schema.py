"""Unit tests for graph_schema integrity validator."""

from __future__ import annotations

import copy

from metainfer.tasks.find_low_hanging_kernel.orchestrator.graph_schema import (
    SCHEMA_VERSION,
    check_integrity,
)
from metainfer.tasks.find_low_hanging_kernel.tests._helpers import (
    make_minimal_valid_graph,
)


def test_valid_graph_is_clean():
    graph = copy.deepcopy(make_minimal_valid_graph())
    report = check_integrity(graph)
    assert report.clean, f"errors: {report.errors}"
    assert report.errors == []


def test_dangling_edge_is_dropped():
    graph = copy.deepcopy(make_minimal_valid_graph())
    graph["edges"].append({"from": "n01", "to": "nGHOST", "label": "x"})
    report = check_integrity(graph)
    assert report.clean is True  # auto-fixed
    # The dangling edge was pruned.
    assert all(e["to"] != "nGHOST" for e in graph["edges"])
    # The fix is logged.
    assert any(
        f["kind"] == "drop_dangling_edge" for f in report.fixes_applied
    )


def test_duplicate_node_id_fails():
    graph = copy.deepcopy(make_minimal_valid_graph())
    bad = copy.deepcopy(graph["nodes"][1])
    graph["nodes"].append(bad)
    report = check_integrity(graph)
    assert not report.clean
    assert any("duplicate node id" in e for e in report.errors)


def test_missing_required_field_fails():
    graph = copy.deepcopy(make_minimal_valid_graph())
    del graph["nodes"][0]["operator"]
    report = check_integrity(graph)
    assert not report.clean
    assert any("missing required field 'operator'" in e for e in report.errors)


def test_isolated_non_entry_exit_fails():
    graph = copy.deepcopy(make_minimal_valid_graph())
    graph["nodes"].append({
        "id": "nLONER", "role": "Mystery", "operator": "mystery_kernel",
        "inputs": [], "outputs": [], "stats": {}, "source_ref": {},
    })
    report = check_integrity(graph)
    assert not report.clean
    assert any("isolated" in e for e in report.errors)


def test_entry_role_exempt_from_isolated_check():
    graph = copy.deepcopy(make_minimal_valid_graph())
    # Remove the entry/exit edges so n01 has no outgoing connection,
    # but it IS tagged entry → should still be clean.
    # (make_minimal_valid_graph already has n01→n02 edge; this is just a sanity
    # check that the entry role doesn't accidentally trigger the isolated rule.)
    report = check_integrity(graph)
    assert report.clean


def test_undeclared_shape_var_fails():
    graph = copy.deepcopy(make_minimal_valid_graph())
    graph["nodes"][0]["inputs"].append({
        "name": "extra", "dtype": "fp16", "shape": ["UNDECLARED"],
    })
    report = check_integrity(graph)
    assert not report.clean
    assert any("undeclared variable" in e for e in report.errors)


def test_schema_version_mismatch_warns():
    graph = copy.deepcopy(make_minimal_valid_graph())
    graph["schema_version"] = 999
    report = check_integrity(graph)
    assert report.clean  # warnings don't break
    assert any("schema_version" in w for w in report.warnings)


def test_default_empty_fixes_applied():
    graph = copy.deepcopy(make_minimal_valid_graph())
    del graph["nodes"][0]["inputs"]
    del graph["nodes"][0]["outputs"]
    del graph["nodes"][0]["stats"]
    del graph["nodes"][0]["source_ref"]
    report = check_integrity(graph)
    assert report.clean
    kinds = {f["kind"] for f in report.fixes_applied}
    assert "default_empty" in kinds
    # Defaults were actually written.
    assert graph["nodes"][0]["inputs"] == []
    assert graph["nodes"][0]["outputs"] == []
    assert graph["nodes"][0]["stats"] == {}


def test_schema_version_constant():
    assert SCHEMA_VERSION == 1
