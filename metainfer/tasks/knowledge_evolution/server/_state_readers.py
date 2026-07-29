"""Task-specific on-disk state readers for knowledge-evolution.

These functions read the task-private iteration schema, state graph,
knowledge-gained records, oracle reports, and log files.
Kept separate from the shell's generic ``state_reader.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from metainfer.server.state_reader import read_run


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load a JSON file, returning None if missing or broken."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


# ---- Iterations ----

def read_iterations(state_dir: Path) -> List[Dict[str, Any]]:
    """Read all iteration records from ``<state_dir>/iterations/``."""
    iterations_dir = state_dir / "iterations"
    if not iterations_dir.is_dir():
        return []

    records: list[Dict[str, Any]] = []
    for fpath in sorted(iterations_dir.glob("*.json")):
        data = _load_json(fpath)
        if data is not None:
            records.append(data)

    records.sort(key=lambda r: r.get("n", r.get("iteration", 0)))

    # Enrich the running iteration with current agent status from agents.json
    agents_data = _load_json(state_dir / "agents.json")
    running_agents: list[str] = []
    if agents_data:
        for a in agents_data.get("agents", []):
            if a.get("status", a.get("phase", "")) in ("running", "starting"):
                running_agents.append(a.get("role", a.get("name", "")))
    if running_agents:
        for r in records:
            if r.get("status") == "running":
                r["agent_status"] = "running: " + ", ".join(running_agents)
                break

    return records


# ---- State graph ----

def read_state_graph(state_dir: Path) -> Dict[str, Any]:
    """Build the state-graph payload for the current run.

    Reads ``run.json`` to get the current phase and last transition,
    then delegates to ``phases.graph_payload()`` from the orchestrator.
    """
    run = read_run(state_dir)
    current = run.get("current_phase", "idle")
    last_outcome = run.get("last_outcome", "")
    last_label = run.get("last_transition_label", "")

    from metainfer.tasks.knowledge_evolution.orchestrator.phases import graph_payload

    return graph_payload(current, last_outcome, last_label)


# ---- Knowledge gained ----

def read_knowledge_gained(state_dir: Path) -> Dict[str, Any]:
    """Summarize what knowledge the consolidator wrote across iterations.

    Reads ``<state_dir>/logs/<nnn>/consolidation.json`` (canonical format).
    Falls back to legacy filenames (knowledge_delta.json, result.json) for
    older task directories, and to ``retrospective.md`` for failed iterations
    that never reached C_consolidate.
    """
    logs_dir = state_dir / "logs"
    if not logs_dir.is_dir():
        return {"iterations": []}

    entries = []
    for iter_dir in sorted(logs_dir.iterdir()):
        if not iter_dir.is_dir():
            continue
        try:
            n = int(iter_dir.name)
        except ValueError:
            continue

        # Canonical: consolidation.json
        data = _load_json(iter_dir / "consolidation.json")
        # Legacy fallbacks
        if data is None:
            for alt in ("knowledge_delta.json", "result.json"):
                data = _load_json(iter_dir / alt)
                if data is not None:
                    break
        if data is None:
            retro = iter_dir / "retrospective.md"
            if retro.exists():
                entries.append({
                    "iteration": n,
                    "status": "failed",
                    "summary": "Iteration failed; see failure analysis and retrospective.",
                    "files": [],
                })
            continue

        if "iteration" not in data:
            data["iteration"] = n
        if "status" not in data:
            data["status"] = "success"
        # Normalize files list across legacy formats
        if "files" not in data:
            files = data.get("files_written", [])
            if not files:
                files = (data.get("created") or []) + (data.get("modified") or [])
            data["files"] = files
        entries.append(data)

    entries.sort(key=lambda e: e.get("iteration", 0))
    return {"iterations": entries, "total_files": sum(len(e.get("files", [])) for e in entries)}


# ---- Oracle report ----

def read_oracle_report(state_dir: Path, iteration: int) -> Dict[str, Any]:
    """Read the oracle report for a specific iteration.

    Looks for ``<state_dir>/logs/<nnn>/oracle_report.md`` or ``c_repair.md``.
    """
    candidates = [
        state_dir / "logs" / f"{iteration:03d}" / "oracle_report.md",
        state_dir / "logs" / f"{iteration:03d}" / "c_repair.md",
    ]
    for path in candidates:
        if path.exists():
            return {
                "found": True,
                "iteration": iteration,
                "markdown": path.read_text(encoding="utf-8", errors="replace"),
                "file": path.name,
            }
    return {"found": False, "iteration": iteration}


# ---- Model knowledge (plan.md + notebooks) ----

def _extract_section(text: str, heading: str) -> Optional[str]:
    """Extract a markdown section by heading (e.g. '## Architecture Summary').

    Returns everything from the heading line up to the next heading of the
    same or higher level, or end of text.
    """
    prefix = heading.split(" ")[0]  # e.g. "##"
    level = prefix.count("#")
    lines = text.split("\n")
    start = None
    for i, line in enumerate(lines):
        if line.strip() == heading or line.strip().startswith(heading):
            start = i
            break
    if start is None:
        return None
    # collect until next heading of same or higher level
    collected = [lines[start]]
    for j in range(start + 1, len(lines)):
        stripped = lines[j].strip()
        if stripped.startswith("#") and not stripped.startswith("####"):
            h_level = len(stripped) - len(stripped.lstrip("#"))
            if h_level <= level:
                break
        collected.append(lines[j])
    return "\n".join(collected).strip()


def _extract_bullet_points(text: str) -> list[str]:
    """Extract bullet-point lines from markdown text."""
    bullets = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("- **") or stripped.startswith("- "):
            # keep it concise
            clean = stripped.lstrip("- ").strip()
            if len(clean) > 200:
                clean = clean[:197] + "..."
            bullets.append(clean)
    return bullets[:20]  # limit


def _read_notebook_entries(notebooks_dir: Path) -> list[dict]:
    """Read model-specific notebook entries from a notebooks/ directory.

    Only reads md files under 02_model_specifics/.
    """
    entries = []
    model_dir = notebooks_dir / "02_model_specifics"
    if not model_dir.is_dir():
        return entries
    for fpath in sorted(model_dir.rglob("*.md")):
        try:
            text = fpath.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        # Get first heading as title
        title = fpath.stem
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped.startswith("# "):
                title = stripped[2:].strip()
                break
        # Relative path for display
        try:
            rel = fpath.relative_to(notebooks_dir)
        except ValueError:
            rel = fpath
        entries.append({
            "title": title,
            "path": str(rel),
            "text": text,
            "size": len(text),
        })
    return entries


def read_model_knowledge(state_dir: Path) -> Dict[str, Any]:
    """Read model architecture knowledge from plan.md and notebooks.

    Returns structured knowledge about the target model:
    - plan_architecture from each iteration's plan.md
    - notebook entries from existing knowledge base
    """
    code_dir = state_dir / "code"
    iterations = []

    if code_dir.is_dir():
        for iter_dir in sorted(code_dir.iterdir()):
            if not iter_dir.is_dir():
                continue
            try:
                n = int(iter_dir.name)
            except ValueError:
                continue

            plan_path = iter_dir / "plan.md"
            if not plan_path.exists():
                continue

            try:
                plan_text = plan_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            # Try multiple possible headings
            arch = (
                _extract_section(plan_text, "## Architecture Summary")
                or _extract_section(plan_text, "## Architecture")
                or _extract_section(plan_text, "## Model Architecture")
            )
            if arch is None:
                # Use first 2000 chars as fallback
                arch = plan_text[:2000]

            bullets = _extract_bullet_points(arch)

            iterations.append({
                "iteration": n,
                "architecture": arch,
                "key_points": bullets,
            })

    # Try to find existing notebook knowledge relevant to this model
    notebook_entries: list[dict] = []
    try:
        # Determine target model name from requirements.json
        req_path = state_dir / "requirements.json"
        model_keywords: list[str] = []
        if req_path.is_file():
            try:
                req = json.loads(req_path.read_text(encoding="utf-8"))
                target = (req.get("target_model") or "")
                # /data/model/Qwen3.5-35B-A3B → "qwen3.5-35b-a3b"
                # Extract model family name from the path basename
                name = Path(target).name.lower()  # "qwen3.5-35b-a3b"
                # Build keywords: ["qwen3.5", "qwen3", "qwen"]
                for part in name.split("-"):
                    part = part.strip()
                    if part:
                        model_keywords.append(part)
                # Add progressively shorter prefixes for fuzzy matching
                # e.g. "qwen3.5" → also "qwen3", "qwen"
                for kw in list(model_keywords):
                    base = kw.rstrip("0123456789. ")
                    if base and base not in model_keywords:
                        model_keywords.append(base)
                    # Also add with stripped version: "qwen3.5" in notebooks is "qwen3"
            except Exception:
                pass

        # Walk up from state_dir to find the repo root
        p = state_dir
        for _ in range(8):
            p = p.parent
            if (p / "CLAUDE.md").is_file():
                break
        candidates = [
            p / "metainfer" / "tasks" / "gen_infer_framework" / "notebooks",
            p / "notebooks",
        ]
        for nb_dir in candidates:
            if nb_dir.is_dir():
                all_entries = _read_notebook_entries(nb_dir)
                for e in all_entries:
                    path_lower = e["path"].lower()
                    title_lower = e["title"].lower()
                    relevant = not model_keywords or any(
                        kw in path_lower or kw in title_lower
                        for kw in model_keywords
                    )
                    if relevant:
                        notebook_entries.append(e)
                break
    except Exception:
        pass

    return {
        "iterations": iterations,
        "notebook_entries": notebook_entries,
    }


def read_model_code_file(
    state_dir: Path, iteration: int, file: str
) -> Dict[str, Any]:
    """Read a single generated code file from a given iteration.

    Looks in ``<state_dir>/code/<nnn>/<file>``.
    """
    path = state_dir / "code" / f"{iteration:03d}" / file
    if path.exists():
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return {"found": False, "iteration": iteration, "file": file}
        return {
            "found": True,
            "iteration": iteration,
            "file": file,
            "text": text,
            "size": len(text),
        }
    return {"found": False, "iteration": iteration, "file": file}


def read_knowledge_diff(state_dir: Path, iteration: int, file: str) -> Dict[str, Any]:
    """Read a notebook file's content from a given iteration's code/logs dir.

    Looks in ``<state_dir>/code/<nnn>/`` or ``<state_dir>/logs/<nnn>/``
    for the requested file.
    """
    candidates = [
        state_dir / "code" / f"{iteration:03d}" / file,
        state_dir / "logs" / f"{iteration:03d}" / file,
    ]
    for path in candidates:
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="replace")
            return {
                "found": True,
                "iteration": iteration,
                "file": file,
                "text": text,
            }
    return {"found": False, "iteration": iteration, "file": file}


# ---- Retrospective ----

def read_retrospective(state_dir: Path, iteration: int) -> Dict[str, Any]:
    """Read the retrospective markdown for a given iteration.

    Looks for ``<state_dir>/logs/<nnn>/retrospective.md``.
    """
    path = state_dir / "logs" / f"{iteration:03d}" / "retrospective.md"
    if path.exists():
        return {
            "found": True,
            "iteration": iteration,
            "markdown": path.read_text(encoding="utf-8", errors="replace"),
        }
    return {"found": False, "iteration": iteration}


# ---- Log ----

def read_log(state_dir: Path) -> Dict[str, Any]:
    """Read the orchestrator log file."""
    log_path = state_dir / "orchestrator.log"
    if log_path.exists():
        return {
            "found": True,
            "text": log_path.read_text(encoding="utf-8", errors="replace"),
        }
    return {"found": False, "text": "(no log file yet)"}


# ---- Charts ----

def read_charts(state_dir: Path) -> Dict[str, Any]:
    """Aggregate perf-per-iteration + durations for the charts panel.

    Reads iteration records and reshapes them into series for Chart.js.
    For knowledge-evolution, the only perf metric surfaced is
    ``oracle_cases_passed``.
    """
    recs = read_iterations(state_dir)
    durations = [
        {"x": r.get("iteration", r.get("n", 0)), "y": round(r.get("duration_s", 0) or 0, 1)}
        for r in recs if r.get("duration_s")
    ]
    perf_keys: list[str] = []
    for r in recs:
        for k in (r.get("perf") or {}):
            if k not in perf_keys:
                perf_keys.append(k)
    perf_series = []
    for k in perf_keys:
        series = [
            {"x": r.get("iteration", r.get("n", 0)), "y": (r.get("perf") or {}).get(k)}
            for r in recs if r.get("perf") and k in r["perf"]
        ]
        perf_series.append({"metric": k, "points": series})
    return {
        "durations": durations,
        "perf_series": perf_series,
    }


# ---- Iteration detail (sub-agent progress + oracle cases + c-repairs) ----

def read_iteration_detail(state_dir: Path, iteration: int) -> Dict[str, Any]:
    """Return detailed progress for a single iteration.

    Includes sub-agent phase summaries (planner/implementer/c_debugger),
    structured oracle per-case results, and c-repair-attempt content.
    """
    # 1. Iteration record
    iter_path = state_dir / "iterations" / f"{iteration:03d}.json"
    iter_rec = _load_json(iter_path)
    if iter_rec is None:
        return {"found": False, "iteration": iteration}

    # 2. Structured oracle report (oracle-report.json)
    oracle_json = _load_json(state_dir / "logs" / f"{iteration:03d}" / "oracle-report.json")

    # 3. c-repair-attempt markdown files
    logs_dir = state_dir / "logs" / f"{iteration:03d}"
    repairs: list[dict] = []
    if logs_dir.is_dir():
        for rp in sorted(logs_dir.glob("c-repair-attempt*.md")):
            try:
                repairs.append({
                    "file": rp.name,
                    "markdown": rp.read_text(encoding="utf-8", errors="replace"),
                })
            except OSError:
                continue

    # 4. c-repairs.jsonl entries
    repair_entries: list[dict] = []
    rj_path = logs_dir / "c-repairs.jsonl"
    if rj_path.exists():
        for line in rj_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry.get("iteration") == iteration:
                    repair_entries.append(entry)
            except (json.JSONDecodeError, TypeError):
                continue

    # 5. Sub-agent phase summaries (planner / implementer / c_debugger)
    phases = iter_rec.get("phases") or {}
    sub_agents: dict[str, dict] = {}
    for role in ("planner", "implementer", "c_debugger"):
        if role in phases:
            entry = dict(phases[role])
            if entry.get("final_text_head"):
                # truncate for display
                entry["final_text_head"] = str(entry["final_text_head"])[:2000]
            sub_agents[role] = entry

    # 5b. State-machine phase outcomes (A_attempt_pure / B_enrich / C_consolidate / D_verify_final)
    phase_outcomes: dict[str, dict] = {}
    for pkey in ("A_attempt_pure", "B_enrich", "C_consolidate", "D_verify_final"):
        if pkey in phases:
            phase_outcomes[pkey] = dict(phases[pkey])

    # 6. Oracle report markdown (oracle_report.md or c_repair.md)
    oracle_md = None
    if logs_dir.is_dir():
        for cand in ("oracle_report.md", "c_repair.md"):
            p = logs_dir / cand
            if p.is_file():
                try:
                    oracle_md = p.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    pass
                break

    # 7. Retrospective
    retro_md = None
    retro_path = logs_dir / "retrospective.md"
    if retro_path.is_file():
        try:
            retro_md = retro_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            pass

    return {
        "found": True,
        "iteration": iteration,
        "status": iter_rec.get("status"),
        "start_phase": iter_rec.get("start_phase"),
        "outcome": iter_rec.get("outcome"),
        "failure_reason": iter_rec.get("failure_reason"),
        "sub_agents": sub_agents,
        "phase_outcomes": phase_outcomes,
        "oracle": oracle_json,
        "oracle_md": oracle_md,
        "retro_md": retro_md,
        "repairs": repairs,
        "repair_entries": repair_entries,
    }


def read_agent_status(state_dir: Path) -> Optional[str]:
    """Read the current agent activity string written by the pipeline."""
    path = state_dir / "agent_status"
    if path.is_file():
        try:
            return path.read_text(encoding="utf-8").strip() or None
        except OSError:
            return None
    return None
