"""Graph schema + integrity validator (Step 3b-i).

The flow-graph JSON schema is intentionally narrow so the deterministic
integrity check can verify it without LLM help. Semantic correctness (does
this node's role/operator/source_ref match reality?) is the job of the
5-worker pool in :mod:`graph_validator`.

Schema
------

::

    {
      "schema_version": 1,
      "metadata": {
        "task_id": "...",
        "model": "...",
        "tp_size": 1,
        "vars": {"B": "batch", "M": "seq_len", ...}
      },
      "nodes": [
        {
          "id": "n01",                    # required, unique
          "role": "RMSNorm",              # required, human-readable step purpose
          "operator": "rms_norm_kernel",  # required, kernel name from trace
          "source_ref": {                 # required; may be null if unknown
            "file": "layers.py", "line": 142, "symbol": "RMSNorm.forward"
          },
          "inputs":  [{"name":"x","dtype":"fp16","shape":["B","M","4096"]}],
          "outputs": [{"name":"y","dtype":"fp16","shape":["B","M","4096"]}],
          "stats": {                       # required; from trace_parsed.json
            "count": 12, "mean_us": 34.2, "std_us": 1.1,
            "total_us": 410.4, "p99_us": 36.0
          },
          "confidence": "high"             # "high" | "medium" | "low"
        }, ...
      ],
      "edges": [{"from": "n01", "to": "n02", "label": "y"}, ...]
    }

Special node roles:
- ``role == "entry"``  — graph source; no incoming edges required.
- ``role == "exit"``   — graph sink;   no outgoing edges required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


SCHEMA_VERSION = 1

NODE_REQUIRED_FIELDS: Tuple[str, ...] = ("id", "role", "operator")
EDGE_REQUIRED_FIELDS: Tuple[str, ...] = ("from", "to")

# Fields that may be present but must have the right type when they are.
NODE_TYPED_FIELDS: Dict[str, type] = {
    "id": str,
    "role": str,
    "operator": str,
    "confidence": str,
    "inputs": list,
    "outputs": list,
    "stats": dict,
    "source_ref": dict,
}

_VALID_CONFIDENCE = {"high", "medium", "low", "unknown", ""}
_VALID_SPECIAL_ROLES = {"entry", "exit"}


@dataclass
class IntegrityReport:
    """Result of :func:`check_integrity`."""

    clean: bool
    fixes_applied: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def _is_str_or_int(v: Any) -> bool:
    return isinstance(v, (str, int))


def _validate_tensor(t: Any, *, path: str, problems: List[str]) -> None:
    if not isinstance(t, dict):
        problems.append(f"{path}: tensor must be an object, got {type(t).__name__}")
        return
    if "name" not in t or not isinstance(t["name"], str):
        problems.append(f"{path}: tensor.name missing or not str")
    if "dtype" not in t or not isinstance(t["dtype"], str):
        problems.append(f"{path}: tensor.dtype missing or not str")
    shape = t.get("shape")
    if not isinstance(shape, list) or not all(
        isinstance(s, (str, int)) for s in shape
    ):
        problems.append(
            f"{path}: tensor.shape must be a list of str|int (variable dims are str)"
        )


def _coerce_str_int(v: Any) -> Any:
    """Allow ints to masquerade as shape dims; leave strings as variables."""
    return v


def _check_shape_var_refs(
    nodes: List[Dict[str, Any]],
    declared_vars: Set[str],
    problems: List[str],
) -> None:
    """All string entries in tensor.shape must be either a declared variable
    OR an arithmetic expression involving declared variables (e.g. ``"hidden_size/tp_size"``).
    We only enforce that the *first identifier* of any compound expression is
    a declared variable, to keep the check cheap and predictable."""
    import re

    ident_re = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
    for node in nodes:
        for side in ("inputs", "outputs"):
            tensors = node.get(side) or []
            if not isinstance(tensors, list):
                continue
            for i, t in enumerate(tensors):
                if not isinstance(t, dict):
                    continue
                shape = t.get("shape")
                if not isinstance(shape, list):
                    continue
                for dim in shape:
                    if not isinstance(dim, str):
                        continue
                    idents = ident_re.findall(dim)
                    unknown = [tok for tok in idents if tok not in declared_vars]
                    if unknown:
                        problems.append(
                            f"node {node.get('id')!r} {side}[{i}].shape={dim!r} "
                            f"references undeclared variable(s): {unknown}"
                        )


def _check_and_fix_integrity(graph: Dict[str, Any]) -> IntegrityReport:
    """Pure-Python integrity check + safe auto-fixes.

    Auto-fixes applied (each logged in ``fixes_applied``):
    - Drop edges whose endpoint isn't a known node id.
    - Add an ``inputs``/``outputs``/``stats`` default if missing.

    Non-fixable problems (recorded in ``errors``):
    - duplicate node ids
    - missing required node fields
    - malformed tensors
    - undeclared shape variables
    """
    fixes: List[Dict[str, Any]] = []
    errors: List[str] = []
    warnings: List[str] = []

    if not isinstance(graph, dict):
        return IntegrityReport(False, errors=["graph root must be an object"])
    if graph.get("schema_version") != SCHEMA_VERSION:
        warnings.append(
            f"schema_version is {graph.get('schema_version')!r}, expected {SCHEMA_VERSION}"
        )

    nodes = graph.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        errors.append("graph.nodes missing or empty")
        nodes = []
    edges = graph.get("edges")
    if not isinstance(edges, list):
        warnings.append("graph.edges missing or not a list — assuming []")
        edges = []
        graph["edges"] = edges

    # 1. Node id uniqueness + required fields.
    seen_ids: Set[str] = set()
    for i, node in enumerate(nodes):
        if not isinstance(node, dict):
            errors.append(f"node[{i}] is not an object")
            continue
        for f_ in NODE_REQUIRED_FIELDS:
            v = node.get(f_)
            if v in (None, ""):
                errors.append(f"node[{i}] missing required field {f_!r}")
        nid = node.get("id")
        if not isinstance(nid, str):
            continue
        if nid in seen_ids:
            errors.append(f"duplicate node id {nid!r}")
        seen_ids.add(nid)

        # Type check optional-but-present fields.
        for fname, expected in NODE_TYPED_FIELDS.items():
            if fname in node and node[fname] is not None:
                if not isinstance(node[fname], expected):
                    errors.append(
                        f"node {nid!r}.{fname} must be {expected.__name__}, "
                        f"got {type(node[fname]).__name__}"
                    )

        # Confidence vocabulary.
        conf = node.get("confidence")
        if isinstance(conf, str) and conf and conf not in _VALID_CONFIDENCE:
            warnings.append(
                f"node {nid!r}.confidence={conf!r} not in {sorted(_VALID_CONFIDENCE)}"
            )

        # Auto-fix: default missing optional fields to empty containers so
        # downstream code doesn't need to None-check.
        for empty_field in ("inputs", "outputs"):
            if node.get(empty_field) is None:
                node[empty_field] = []
                fixes.append({
                    "kind": "default_empty",
                    "node": nid,
                    "field": empty_field,
                })
        if node.get("stats") is None:
            node["stats"] = {}
            fixes.append({"kind": "default_empty", "node": nid, "field": "stats"})
        if node.get("source_ref") is None:
            node["source_ref"] = {}
            fixes.append({
                "kind": "default_empty", "node": nid, "field": "source_ref",
            })

        # Validate tensors.
        for side in ("inputs", "outputs"):
            tensors = node.get(side) or []
            for j, t in enumerate(tensors):
                _validate_tensor(
                    t, path=f"node {nid!r}.{side}[{j}]", problems=errors,
                )

    # 2. Edges — drop dangling ones.
    pruned_edges: List[Dict[str, Any]] = []
    for i, e in enumerate(edges):
        if not isinstance(e, dict):
            errors.append(f"edge[{i}] is not an object")
            continue
        for f_ in EDGE_REQUIRED_FIELDS:
            if e.get(f_) in (None, ""):
                errors.append(f"edge[{i}] missing required field {f_!r}")
        frm, to = e.get("from"), e.get("to")
        if frm in seen_ids and to in seen_ids:
            pruned_edges.append(e)
        else:
            missing = []
            if frm not in seen_ids:
                missing.append(f"from={frm!r}")
            if to not in seen_ids:
                missing.append(f"to={to!r}")
            fixes.append({
                "kind": "drop_dangling_edge",
                "edge_index": i,
                "missing": missing,
                "edge": e,
            })
    if len(pruned_edges) != len(edges):
        graph["edges"] = pruned_edges

    # 3. Isolated-node check (entry/exit roles are exempt).
    touched: Set[str] = set()
    for e in pruned_edges:
        touched.add(e.get("from"))
        touched.add(e.get("to"))
    for node in nodes:
        nid = node.get("id") if isinstance(node, dict) else None
        if not isinstance(nid, str):
            continue
        if node.get("role") in _VALID_SPECIAL_ROLES:
            continue
        if nid not in touched:
            errors.append(
                f"node {nid!r} is isolated (no edges). "
                f"If it's a graph entry/exit, set role to 'entry' or 'exit'."
            )

    # 4. Shape-variable references.
    metadata = graph.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
        graph["metadata"] = metadata
    declared = metadata.get("vars") or {}
    if not isinstance(declared, dict):
        warnings.append("metadata.vars is not a dict — treating as empty")
        declared = {}
    declared_set: Set[str] = set(declared.keys())
    # Numeric literals are also always valid shape dims, but we don't need to
    # add them — only string dims are treated as variable references.
    _check_shape_var_refs(nodes, declared_set, errors)

    clean = not errors
    return IntegrityReport(
        clean=clean, fixes_applied=fixes, errors=errors, warnings=warnings,
    )


def check_integrity(graph: Dict[str, Any]) -> IntegrityReport:
    """Run :func:`_check_and_fix_integrity` and return the report. The graph
    dict is mutated in-place to apply safe auto-fixes."""
    return _check_and_fix_integrity(graph)


def is_isolated_exempt(node: Dict[str, Any]) -> bool:
    return node.get("role") in _VALID_SPECIAL_ROLES
