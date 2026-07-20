"""Disk readers for the calc-theoretical-value task type.

Every read-only endpoint in ``routes.py`` delegates here. Separating
the readers from the FastAPI handlers makes them unit-testable without
spinning up an ASGI client.

The functions read artifacts produced by the calc_value orchestrator's
4-step pipeline. All step0..step4 outputs live under the task's
``workspace_dir`` (NOT the metadata-only ``state_dir``)::

    <workspace>/step0/agent_rough/per_node/<compound>.py
    <workspace>/step0/rough_results.json
    <workspace>/step1/round_NN/agent_X/memory.json
    <workspace>/step1/memory.round_NN.json
    <workspace>/step2/graph.json
    <workspace>/step2/rounds/<NN>_<build|validate|fix>/...
    <workspace>/step3/final/<compound>.py
    <workspace>/step3/final/<compound>.meta.json
    <workspace>/step3/cells/_state.json
    <workspace>/step3/cells/<compound>/<angle>/round_NN/writer/calc.py
    <workspace>/step4/viz.html
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from metainfer.server._helpers import find_events_file

# Lazy import — the deterministic helpers live in the orchestrator
# subpackage; importing them eagerly at module load would slow the
# WebUI for non-calc tasks.
def _det():
    from metainfer.tasks.calc_value.orchestrator import deterministic as _d
    return _d


# --------------------------------------------------------------------------- #
# Step-level readers
# --------------------------------------------------------------------------- #

def read_graph(workspace_dir: Path) -> Dict[str, Any]:
    """Return step2/graph.json verbatim. Raises FileNotFoundError if
    not built yet."""
    graph_path = workspace_dir / "step2" / "graph.json"
    if not graph_path.exists():
        raise FileNotFoundError("graph.json not built yet")
    return json.loads(graph_path.read_text(encoding="utf-8"))


def read_viz(workspace_dir: Path, task_id: str) -> str:
    """Return step4/viz.html with TASK_ID and COMPUTE_URL injected as a
    server-side header script.  This guarantees correctness even if the
    S4 agent ignored the prompt instruction to hardcode them (the agent
    may try to parse window.location which always fails inside the
    WebUI's path-routed iframe)."""
    viz_path = workspace_dir / "step4" / "viz.html"
    if not viz_path.exists():
        raise FileNotFoundError("viz.html not built yet")
    html = viz_path.read_text(encoding="utf-8")
    header = (
        "<script>"
        "window.METAINFER_TASK_ID = " + json.dumps(task_id) + ";"
        "window.METAINFER_COMPUTE_URL = " + json.dumps(
            f"/api/calc-theoretical-value/{task_id}/calc/compute"
        ) + ";"
        "</script>"
    )
    # Insert right after <head> or <html>, before any agent-written
    # <script> that tries to parse window.location for task_id.
    import re
    if re.search(r"<head[^>]*>", html, re.IGNORECASE):
        html = re.sub(r"(<head[^>]*>)", r"\1\n" + header, html, count=1, flags=re.IGNORECASE)
    elif re.search(r"<html[^>]*>", html, re.IGNORECASE):
        html = re.sub(r"(<html[^>]*>)", r"\1\n" + header, html, count=1, flags=re.IGNORECASE)
    else:
        html = header + html
    return html


def read_summary(workspace_dir: Path) -> Dict[str, Any]:
    """Step-by-step pipeline progress for the calc-value task."""
    out: Dict[str, Any] = {"steps": {}}
    # Step 1
    s1 = workspace_dir / "step1" / "memory.json"
    out["steps"]["s1_analyze"] = {
        "done": s1.exists(),
        "memory_path": str(s1) if s1.exists() else None,
    }
    # Step 2
    s2 = workspace_dir / "step2" / "graph.json"
    graph_node_count = 0          # template nodes (per-section)
    aggregated_node_count = 0     # × repeat_count
    section_count = 0
    sections_summary: List[Dict[str, Any]] = []
    if s2.exists():
        try:
            _d = _det()
            g = _d.normalize_graph(json.loads(s2.read_text(encoding="utf-8")))
            graph_node_count = _d.section_node_count(g)
            aggregated_node_count = _d.aggregated_node_count(g)
            section_count = len(g.get("sections") or [])
            for sec in g.get("sections") or []:
                if not isinstance(sec, dict):
                    continue
                rc = (sec.get("repeat_count") if sec.get("kind") == "layer_template"
                      else 1)
                sections_summary.append({
                    "id": sec.get("id"),
                    "kind": sec.get("kind"),
                    "repeat_count": rc,
                    "node_count": len(((sec.get("graph") or {}).get("nodes")) or []),
                    "edge_count": len(((sec.get("graph") or {}).get("edges")) or []),
                })
        except ValueError:
            pass
    out["steps"]["s2_graph"] = {
        "done": s2.exists(),
        "graph_path": str(s2) if s2.exists() else None,
        "node_count": graph_node_count,
        "aggregated_node_count": aggregated_node_count,
        "section_count": section_count,
        "sections": sections_summary,
    }
    # Step 3
    final_dir = workspace_dir / "step3" / "final"
    calc_scripts = list(final_dir.glob("*.py")) if final_dir.exists() else []
    out["steps"]["s3_calculate"] = {
        "done": len(calc_scripts) > 0,
        "final_dir": str(final_dir),
        "node_count": len(calc_scripts),
    }
    # Step 4
    s4 = workspace_dir / "step4" / "viz.html"
    out["steps"]["s4_visualize"] = {
        "done": s4.exists(),
        "viz_path": str(s4) if s4.exists() else None,
    }
    return out


def read_iterations(workspace_dir: Path) -> Dict[str, Any]:
    """Per-round, per-agent analysis results for every step.

    Surfaces each agent's individual output (including disagreements)
    so the user can audit the convergence process, not just the final
    consensus.
    """
    out: Dict[str, Any] = {"s1_analyze": [], "s2_graph": [], "s3_calculate": []}

    # ---- Step 1: round_NN/agent_X/memory.json ----
    s1 = workspace_dir / "step1"
    if s1.exists():
        rounds = sorted(d for d in s1.iterdir() if d.is_dir()
                        and d.name.startswith("round_"))
        for r in rounds:
            round_idx = int(r.name.split("_")[1])
            agents = []
            for a in sorted(x for x in r.iterdir() if x.is_dir()
                            and x.name.startswith("agent_")):
                mem_p = a / "memory.json"
                memory = None
                if mem_p.exists():
                    try:
                        memory = json.loads(mem_p.read_text(encoding="utf-8"))
                    except ValueError:
                        memory = None
                resp_p = a / "response.txt"
                response_excerpt = None
                if resp_p.exists():
                    try:
                        response_excerpt = resp_p.read_text(encoding="utf-8")[:8000]
                    except OSError:
                        response_excerpt = None
                err_p = a / "parse_error.txt"
                parse_error = err_p.read_text(encoding="utf-8")[:2000] \
                    if err_p.exists() else None
                # Locate events.jsonl + workdir so QA can target this
                # agent.
                events_file = find_events_file(r / "logs" / a.name)
                agents.append({
                    "name": a.name,
                    "has_memory": memory is not None,
                    "memory": memory,
                    "response_excerpt": response_excerpt,
                    "parse_error": parse_error,
                    "events_file": str(events_file) if events_file else None,
                    "target_workdir": str(a),
                })
            # Disputes for this round (re-derived from the per-agent
            # memories by running the deterministic merge again).
            disputes = []
            agent_mems = [a.get("memory") for a in agents
                          if a.get("memory")]
            if len(agent_mems) >= 2:
                try:
                    _d = _det()
                    _, disp = _d.merge_memories(agent_mems)
                    disputes = disp
                except Exception:  # noqa: BLE001
                    pass
            out["s1_analyze"].append({
                "round": round_idx,
                "agents": agents,
                "disputes": disputes,
                "converged": len(disputes) == 0,
            })

    # ---- Step 2: rounds/<NN>_(validate|fix|build)/ ----
    s2 = workspace_dir / "step2"
    if s2.exists():
        rroot = s2 / "rounds"
        if rroot.exists():
            rounds = sorted(rroot.iterdir(), key=lambda p: p.name)
            for r in rounds:
                if not r.is_dir():
                    continue
                label = r.name  # e.g. "00_build", "01_validate", "02_fix"
                kind = "build" if "_build" in label else (
                    "validate" if "_validate" in label else (
                        "fix" if "_fix" in label else "other"))
                entry_rec: Dict[str, Any] = {
                    "dir": label, "kind": kind,
                }
                gj = r / "graph.json"
                if gj.exists():
                    try:
                        _d = _det()
                        g = _d.normalize_graph(
                            json.loads(gj.read_text(encoding="utf-8"))
                        )
                        entry_rec["node_count"] = _d.section_node_count(g)
                        entry_rec["edge_count"] = _d.section_edge_count(g)
                        entry_rec["aggregated_node_count"] = (
                            _d.aggregated_node_count(g)
                        )
                        entry_rec["section_count"] = (
                            len(g.get("sections") or [])
                        )
                        entry_rec["sections"] = [
                            {
                                "id": sec.get("id"),
                                "kind": sec.get("kind"),
                                "repeat_count": (
                                    sec.get("repeat_count")
                                    if sec.get("kind") == "layer_template"
                                    else 1
                                ),
                                "node_count": len(
                                    ((sec.get("graph") or {}).get("nodes")) or []
                                ),
                                "edge_count": len(
                                    ((sec.get("graph") or {}).get("edges")) or []
                                ),
                            }
                            for sec in (g.get("sections") or [])
                            if isinstance(sec, dict)
                        ]
                    except ValueError:
                        entry_rec["node_count"] = None
                vj = r / "verdicts.json"
                if vj.exists():
                    try:
                        verdicts = json.loads(vj.read_text(encoding="utf-8"))
                        entry_rec["verdicts"] = verdicts
                        entry_rec["pass"] = sum(
                            1 for v in verdicts
                            if isinstance(v, dict) and v.get("verdict") == "pass")
                        entry_rec["reject"] = sum(
                            1 for v in verdicts
                            if isinstance(v, dict) and v.get("verdict") == "reject")
                    except ValueError:
                        entry_rec["verdicts"] = []
                # Per-validator raw responses (one per node).
                validators = []
                for vdir in sorted(r.iterdir()):
                    if not vdir.is_dir() or not vdir.name.startswith("validator_"):
                        continue
                    resp_p = vdir / "response.txt"
                    v_ef = find_events_file(r / "logs" / vdir.name)
                    validators.append({
                        "name": vdir.name,
                        "response_excerpt": (
                            resp_p.read_text(encoding="utf-8")[:1500]
                            if resp_p.exists() else None),
                        "events_file": str(v_ef) if v_ef else None,
                        "target_workdir": str(vdir),
                    })
                if validators:
                    entry_rec["validators"] = validators
                out["s2_graph"].append(entry_rec)

    # ---- Step 3: rounds/<node>/round_NN/writer_X/ ----
    s3 = workspace_dir / "step3" / "rounds"
    if s3.exists():
        nodes = []
        for ndir in sorted(s3.iterdir()):
            if not ndir.is_dir():
                continue
            node_rec: Dict[str, Any] = {
                "node_id": ndir.name, "rounds": [],
                "compound_id": ndir.name,
            }
            # Compound id is ``<section_id>__<node_id>`` (sanitized).
            # Resolve the bare node_id + section context from the
            # sibling final/<compound>.meta.json if present.
            meta_p = workspace_dir / "step3" / "final" / f"{ndir.name}.meta.json"
            if meta_p.exists():
                try:
                    m = json.loads(meta_p.read_text(encoding="utf-8"))
                    node_rec["node_id"] = m.get("node_id") or ndir.name
                    node_rec["section_id"] = m.get("section_id")
                    node_rec["section_kind"] = m.get("section_kind")
                    node_rec["section_repeat_count"] = (
                        m.get("section_repeat_count")
                    )
                except (ValueError, OSError):
                    pass
            for rdir in sorted(ndir.iterdir()):
                if not rdir.is_dir() or not rdir.name.startswith("round_"):
                    continue
                round_idx = int(rdir.name.split("_")[1])
                writers = []
                for wdir in sorted(rdir.iterdir()):
                    if not wdir.is_dir() or not wdir.name.startswith("writer_"):
                        continue
                    calc_p = wdir / "calc.py"
                    resp_p = wdir / "response.txt"
                    err_p = wdir / "error.txt"
                    w_ef = find_events_file(rdir / "logs" / wdir.name)
                    writers.append({
                        "name": wdir.name,
                        "has_script": calc_p.exists(),
                        "script_excerpt": (
                            calc_p.read_text(encoding="utf-8")[:2500]
                            if calc_p.exists() else None),
                        "response_excerpt": (
                            resp_p.read_text(encoding="utf-8")[:1500]
                            if resp_p.exists() else None),
                        "events_file": str(w_ef) if w_ef else None,
                        "target_workdir": str(wdir),
                        "error": (
                            err_p.read_text(encoding="utf-8")[:500]
                            if err_p.exists() else None),
                    })
                rec: Dict[str, Any] = {"round": round_idx, "writers": writers}
                comp_p = rdir / "comparison.json"
                if comp_p.exists():
                    try:
                        comp = json.loads(comp_p.read_text(encoding="utf-8"))
                        rec["ok"] = bool(comp.get("ok"))
                        rec["mismatch_count"] = len(comp.get("mismatches") or [])
                        rec["mismatches_excerpt"] = (
                            comp.get("mismatches") or [])[:5]
                    except ValueError:
                        pass
                med_p = rdir / "median_fallback.json"
                if med_p.exists():
                    rec["median_fallback"] = True
                node_rec["rounds"].append(rec)
            nodes.append(node_rec)
        out["s3_calculate"] = nodes

    return out


# --------------------------------------------------------------------------- #
# Compute (deterministic, on-the-fly)
# --------------------------------------------------------------------------- #

def compute(workspace_dir: Path, batch_size: int, seq_len: int) -> Dict[str, Any]:
    """Run every per-compound calc.py at the given shape and return
    per-instance numbers keyed by compound_id, split into prefill and
    decode phases. Totals are aggregated as
    ``Σ per_compound.{phase} * section.repeat_count``.

    Deterministic — no LLM in the loop.
    """
    if batch_size <= 0 or seq_len <= 0:
        raise ValueError("batch_size and seq_len must be positive")
    final_dir = workspace_dir / "step3" / "final"
    if not final_dir.exists():
        raise FileNotFoundError("calc scripts not built yet")
    _d = _det()
    per_compound: Dict[str, Dict[str, Any]] = {}
    per_compound_meta: Dict[str, Dict[str, Any]] = {}
    total_pre_tflops = 0.0
    total_pre_gb = 0.0
    total_dec_tflops = 0.0
    total_dec_gb = 0.0
    errors: Dict[str, str] = {}
    # Walk *.py scripts; resolve each script's repeat_count from its
    # sibling .meta.json (defaults to 1 if meta missing or unset).
    for script in sorted(final_dir.glob("*.py")):
        compound_id = script.stem
        meta_path = final_dir / f"{compound_id}.meta.json"
        repeat = 1
        section_id: Optional[str] = None
        section_kind: Optional[str] = None
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                repeat = int(meta.get("section_repeat_count") or 1)
                section_id = meta.get("section_id")
                section_kind = meta.get("section_kind")
                per_compound_meta[compound_id] = {
                    "section_id": section_id,
                    "section_kind": section_kind,
                    "repeat_count": repeat,
                    "node_id": meta.get("node_id"),
                }
            except (ValueError, OSError):
                pass
        try:
            mod = _d.load_calc_module(
                script, module_name=f"_calc_web_{compound_id}",
            )
            phases = _d.call_calc(mod, batch_size, seq_len)
            pre = phases["prefill"]
            dec = phases["decode"]
            per_compound[compound_id] = {
                "prefill": pre,
                "decode": dec,
                # Legacy aliases (prefill-derived) for older consumers.
                "tflops": pre["tflops"],
                "access_gb": pre["access_gb"],
            }
            total_pre_tflops += pre["tflops"] * repeat
            total_pre_gb += pre["access_gb"] * repeat
            total_dec_tflops += dec["tflops"] * repeat
            total_dec_gb += dec["access_gb"] * repeat
        except Exception as exc:  # noqa: BLE001
            errors[compound_id] = f"{type(exc).__name__}: {exc}"
            per_compound[compound_id] = {
                "prefill": {"tflops": 0.0, "access_gb": 0.0},
                "decode":  {"tflops": 0.0, "access_gb": 0.0},
                "tflops": 0.0, "access_gb": 0.0,
            }
    # approximate_compounds: read flags from each meta file.
    approximate_compounds: List[str] = []
    for compound_id, _meta in per_compound_meta.items():
        # Re-read approximate flag (we only kept a subset above).
        mp = final_dir / f"{compound_id}.meta.json"
        try:
            full = json.loads(mp.read_text(encoding="utf-8"))
            if full.get("approximate"):
                approximate_compounds.append(compound_id)
        except (ValueError, OSError):
            pass
    pre_ai = (total_pre_tflops / total_pre_gb) if total_pre_gb > 0 else 0.0
    dec_ai = (total_dec_tflops / total_dec_gb) if total_dec_gb > 0 else 0.0
    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "per_compound": per_compound,
        # Legacy alias for older frontends; same data as per_compound.
        "per_node": per_compound,
        "compound_meta": per_compound_meta,
        "totals": {
            "prefill": {
                "tflops": total_pre_tflops,
                "access_gb": total_pre_gb,
                "arithmetic_intensity": pre_ai,
            },
            "decode": {
                "tflops": total_dec_tflops,
                "access_gb": total_dec_gb,
                "arithmetic_intensity": dec_ai,
            },
            # Legacy aliases (prefill-derived).
            "tflops": total_pre_tflops,
            "access_gb": total_pre_gb,
            "arithmetic_intensity": pre_ai,
        },
        "approximate_compounds": approximate_compounds,
        "approximate_nodes": approximate_compounds,
        "errors": errors,
    }


# --------------------------------------------------------------------------- #
# Step 0 rough + step 3 streaming cells (with optional shape recompute)
# --------------------------------------------------------------------------- #

def read_rough(workspace_dir: Path, batch_size: Optional[int], seq_len: Optional[int]) -> Dict[str, Any]:
    """Return the S0 rough-pass estimate.

    Reads ``step0/rough_results.json``. Returns an empty placeholder
    shape if S0 hasn't run yet.

    When ``batch_size`` or ``seq_len`` is non-None, re-runs each
    ``step0/agent_rough/per_node/<compound>.py`` at the requested shape
    on the fly and overrides the ``prefill`` / ``decode`` (and legacy
    ``tflops_picked`` / ``gb_picked``) fields. The on-disk file is
    never modified.
    """
    rough = workspace_dir / "step0" / "rough_results.json"
    if not rough.exists():
        return {"ok": False, "pending": True, "results": [], "graph": {"sections": []}}
    try:
        data = json.loads(rough.read_text(encoding="utf-8"))
    except ValueError:
        return {"ok": False, "pending": False, "error": "corrupt rough_results.json",
                "results": [], "graph": {"sections": []}}

    if batch_size is None and seq_len is None:
        return data  # canonical-shape values baked into the JSON

    bs = batch_size if batch_size is not None else 1
    sl = seq_len if seq_len is not None else 512
    if bs < 1 or sl < 1:
        raise ValueError("batch_size / seq_len must be >= 1")

    # On-demand recompute at the requested shape.
    _d = _det()
    per_node_dir = workspace_dir / "step0" / "agent_rough" / "per_node"
    for row in data.get("results") or []:
        if not isinstance(row, dict) or not row.get("ok"):
            continue
        compound = row.get("compound")
        if not compound:
            continue
        script = per_node_dir / f"{compound}.py"
        if not script.exists():
            continue
        try:
            mod = _d.load_calc_module(
                script, module_name=f"_calc_rough_{compound}",
            )
            phases = _d.call_calc(mod, bs, sl)
            pre = phases["prefill"]
            dec = phases["decode"]
            row["prefill"] = pre
            row["decode"] = dec
            # Legacy aliases.
            row["tflops_picked"] = pre["tflops"]
            row["gb_picked"] = pre["access_gb"]
        except Exception:  # noqa: BLE001
            # Leave canonical values in place if recompute fails.
            continue
    data["combo"] = {"batch_size": bs, "seq_len": sl}
    return data


def read_cells(workspace_dir: Path, batch_size: Optional[int], seq_len: Optional[int]) -> Dict[str, Any]:
    """Return the S3 streaming cell-state grid.

    Reads ``step3/cells/_state.json``. Returns an empty shell if S3
    hasn't started. Optional shape override recomputes each cell on
    the fly; the on-disk file is never modified.
    """
    state = workspace_dir / "step3" / "cells" / "_state.json"
    if not state.exists():
        return {"round": 0, "nodes": {}, "pending": True}
    try:
        data = json.loads(state.read_text(encoding="utf-8"))
    except ValueError:
        return {"round": 0, "nodes": {}, "error": "corrupt _state.json"}

    if batch_size is None and seq_len is None:
        return data  # canonical-shape values baked into _state.json

    bs = batch_size if batch_size is not None else 1
    sl = seq_len if seq_len is not None else 512
    if bs < 1 or sl < 1:
        raise ValueError("batch_size / seq_len must be >= 1")

    _d = _det()
    cells_root = workspace_dir / "step3" / "cells"
    nodes = data.get("nodes") or {}
    for compound, node in nodes.items():
        cells = node.get("cells") or {}
        for angle, c in cells.items():
            if not isinstance(c, dict):
                continue
            # Need a usable calc.py to recompute.
            if c.get("status") not in ("ok", "approximate"):
                continue
            round_idx = c.get("round")
            if round_idx is None:
                continue
            calc_path = (
                cells_root / compound / angle
                / f"round_{int(round_idx):02d}" / "writer" / "calc.py"
            )
            if not calc_path.exists():
                continue
            try:
                mod = _d.load_calc_module(
                    calc_path, module_name=f"_calc_cell_{compound}_{angle}",
                )
                phases = _d.call_calc(mod, bs, sl)
                pre = phases["prefill"]
                dec = phases["decode"]
                c["prefill"] = pre
                c["decode"] = dec
                # Legacy aliases stay in sync with prefill.
                c["tflops"] = pre["tflops"]
                c["gb"] = pre["access_gb"]
                c["picked_combo"] = {"batch_size": bs, "seq_len": sl}
            except Exception:  # noqa: BLE001
                continue
    data["combo"] = {"batch_size": bs, "seq_len": sl}
    return data


def read_cell_detail(
    workspace_dir: Path, compound: str, angle: str, round_idx: int,
) -> Dict[str, Any]:
    """Return one cell's full detail: calc.py source, response.txt
    (agent thinking), result.json (single canonical-shape record), and
    the writer's events.jsonl path so the UI can launch a QA session
    against this cell.
    """
    if angle not in ("a", "b"):
        raise ValueError("angle must be one of a / b")
    cell_dir = workspace_dir / "step3" / "cells" / compound / angle / f"round_{round_idx:02d}"
    if not cell_dir.exists():
        raise FileNotFoundError(
            f"cell not found: {compound}/{angle}/round_{round_idx:02d}"
        )

    calc_path = cell_dir / "writer" / "calc.py"
    response_path = cell_dir / "writer" / "response.txt"
    result_path = cell_dir / "result.json"

    calc_py = calc_path.read_text(encoding="utf-8") if calc_path.exists() else ""
    response = response_path.read_text(encoding="utf-8") if response_path.exists() else ""
    result: Any = None
    if result_path.exists():
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except ValueError:
            result = None

    # Locate the writer's events.jsonl (attempt-agnostic — pick the
    # highest attempt number that exists).
    events_file: Optional[str] = None
    logs_dir = cell_dir / "logs"
    if logs_dir.is_dir():
        candidates = sorted(logs_dir.rglob("*.events.jsonl"))
        if candidates:
            events_file = str(candidates[-1])

    # Read sibling cells (other angle at same round) for mismatch
    # context if both exist.
    _d = _det()
    siblings: Dict[str, Any] = {}
    for a in ("a", "b"):
        sib_result_path = (
            workspace_dir / "step3" / "cells" / compound / a
            / f"round_{round_idx:02d}" / "result.json"
        )
        if sib_result_path.exists():
            try:
                siblings[a] = json.loads(sib_result_path.read_text(encoding="utf-8"))
            except ValueError:
                pass
    mismatches: List[Dict[str, Any]] = []
    if len(siblings) == 2:
        cmp = _d.compare_calc_results([siblings[a] for a in ("a", "b")])
        mismatches = cmp.get("mismatches") or []

    return {
        "compound": compound,
        "angle": angle,
        "round": round_idx,
        "calc_py": calc_py,
        "response": response,
        "result": result,
        "events_file": events_file,
        "workdir": str(cell_dir / "writer"),
        "siblings_present": sorted(siblings.keys()),
        "mismatches": mismatches,
    }
