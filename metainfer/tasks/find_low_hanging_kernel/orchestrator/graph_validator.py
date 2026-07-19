"""Graph semantic validator (Step 3b-ii) — deterministic driver.

For each round:
1. Run :func:`graph_schema.check_integrity` (auto-fixes).
2. Split nodes into consecutive 3-node groups (deterministic ordering by id).
3. Build a :class:`PoolTask` per group; prompt asks the agent to verify each
   node's role/operator/source_ref/shapes/stats against the step-1/step-2
   memory + framework source.
4. Collect results, apply suggested patches, write per-group JSON to disk.
5. Round decision: 0 issues → CLEAN; otherwise NEEDS_FIX (loop).

The 5 workers form a *persistent* pool (one ``AgentPool`` instance for the
whole round) — we do not spawn a fresh pool per group, per the spec. Token
accounting flows automatically through SubAgentManager → TokenBudget.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from metainfer.orchestrator.agent_pool import AgentPool, PoolTask

from .graph_schema import IntegrityReport, check_integrity


POOL_SIZE = 5
GROUP_SIZE = 3


@dataclass
class RoundResult:
    round_num: int
    integrity: IntegrityReport
    group_result_paths: List[Path]
    issue_count: int
    outcome: str  # "clean" | "needs_fix" | "failed"
    notes: List[str]


def _split_into_groups(nodes: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Deterministic consecutive grouping by node position (NOT id-sorted) —
    the builder already emits nodes in a meaningful order (forward-pass order
    where possible), so preserving input order gives meaningful neighbors."""
    out: List[List[Dict[str, Any]]] = []
    for i in range(0, len(nodes), GROUP_SIZE):
        out.append(nodes[i : i + GROUP_SIZE])
    return out


def _neighbor_context(nodes: List[Dict[str, Any]], graph: Dict[str, Any]) -> str:
    """Return a compact description of how the group connects to the rest of
    the graph, so the validator can reason about edge correctness."""
    ids = {n.get("id") for n in nodes}
    edges = graph.get("edges") or []
    in_edges = [e for e in edges if e.get("to") in ids]
    out_edges = [e for e in edges if e.get("from") in ids]
    lines = ["Inbound edges:"]
    for e in in_edges:
        lines.append(f"  {e.get('from')} -> {e.get('to')} (label={e.get('label','')!r})")
    lines.append("Outbound edges:")
    for e in out_edges:
        lines.append(f"  {e.get('from')} -> {e.get('to')} (label={e.get('label','')!r})")
    return "\n".join(lines)


def _build_validator_prompt(
    group: List[Dict[str, Any]],
    graph: Dict[str, Any],
    *,
    step1_path: Path,
    step2_path: Path,
    framework_dir: Path,
) -> str:
    import json as _json

    neighbors = _neighbor_context(group, graph)
    nodes_json = _json.dumps(group, indent=2, ensure_ascii=False)

    return f"""You are a validation agent for an inference-framework execution-flow graph.

# Your task
For each of the {len(group)} node(s) below, verify its declared fields against
the ground-truth evidence:

  1. Read the Step 1 code-analysis memory at: `{step1_path}`
  2. Read the Step 2 tracing-analysis memory at: `{step2_path}`
  3. Cross-reference kernel names, source-file paths, line numbers, and tensor
     shapes against the framework source at: `{framework_dir}` (READ-ONLY — do
     not write to anything under that directory).

For each node, check:
  - role:           Is the human-readable step description accurate?
  - operator:       Does this kernel name appear in the trace summary? Does it
                    match the source-level operator that should produce it?
  - source_ref:     Does the file:line:symbol actually exist in the framework
                    source? Is it the code path that ACTUALLY runs under the
                    user's CLI args + env vars?
  - inputs/outputs: Are tensor shapes correct? Pay special attention to
                    tensor-parallel (TP) splits — a weight dim like
                    `hidden_size` may become `hidden_size/tp_size` on a single
                    rank. Shape variables (e.g. "B", "M") must be consistent
                    across the graph.
  - stats:          Do mean_us / count / total_us plausibly match the trace
                    summary for this kernel name? Note that one kernel name
                    may be invoked from multiple call sites with very different
                    shapes — flag when the stats look like a blend.

# Output format
Emit a single JSON object mapping each node id to a verdict:

```json
{{
  "<node_id>": {{
    "ok": true|false,
    "issues": ["... concise description of each problem ..."],
    "suggested_patch": {{
      "role": "...", "operator": "...",
      "source_ref": {{"file": "...", "line": 0, "symbol": "..."}},
      "inputs": [...], "outputs": [...],
      "stats": {{...}}, "confidence": "..."
    }}
  }},
  ...
}}
```

Rules:
- Only populate `suggested_patch` fields you actually want to change. An empty
  patch means "this node is OK as-is" (still set ok=true).
- Do NOT propose patches that change the node id or remove the node.
- Output ONLY the JSON object (no preamble, no markdown fences around the JSON
  payload other than the one shown above if you need it for clarity).

# Node group to validate
{nodes_json}

# Neighbors (for edge correctness context)
{neighbors}
"""


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Pull the first balanced {...} block out of an agent response."""
    if not text:
        return None
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    blob = text[start : i + 1]
                    try:
                        parsed = json.loads(blob)
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        pass
                    break
        start = text.find("{", start + 1)
    return None


def apply_patches(graph: Dict[str, Any], verdicts: Dict[str, Any]) -> Tuple[int, List[str]]:
    """Apply non-destructive patches from the validator verdicts.

    Refuses to change node id or remove nodes. Returns (count_applied, notes)."""
    applied = 0
    notes: List[str] = []
    nodes_by_id = {n.get("id"): n for n in (graph.get("nodes") or []) if isinstance(n, dict)}

    for nid, verdict in verdicts.items():
        if not isinstance(verdict, dict):
            continue
        patch = verdict.get("suggested_patch")
        if not isinstance(patch, dict):
            continue
        target = nodes_by_id.get(nid)
        if target is None:
            notes.append(f"verdict references unknown node id {nid!r} — skipped")
            continue

        # Never change id; refuse destructive ops.
        if "id" in patch and patch["id"] != target.get("id"):
            notes.append(f"node {nid!r}: refusing to change id — skipped")
            continue

        for k, v in patch.items():
            if k == "id":
                continue
            # Light type guard.
            if k in ("inputs", "outputs") and not isinstance(v, list):
                notes.append(f"node {nid!r}.{k}: expected list — skipped")
                continue
            if k == "source_ref" and not isinstance(v, dict):
                notes.append(f"node {nid!r}.source_ref: expected dict — skipped")
                continue
            if k == "stats" and not isinstance(v, dict):
                notes.append(f"node {nid!r}.stats: expected dict — skipped")
                continue
            target[k] = v
            applied += 1
    return applied, notes


def _count_issues(verdicts: Dict[str, Any]) -> int:
    n = 0
    for v in verdicts.values():
        if isinstance(v, dict) and v.get("ok") is False:
            n += 1
    return n


def run_validation_round(
    *,
    round_num: int,
    graph: Dict[str, Any],
    manager,  # SubAgentManager | MockAgentManager
    step1_path: Path,
    step2_path: Path,
    framework_dir: Path,
    round_dir: Path,
    pool_log_dir: Path,
    timeout_s: int = 600,
) -> RoundResult:
    """Execute one integrity + semantic validation round.

    The ``graph`` dict is mutated in-place (both by integrity auto-fixes and
    by patch application). All artifacts land in ``round_dir``.
    """
    round_dir.mkdir(parents=True, exist_ok=True)
    pool_log_dir.mkdir(parents=True, exist_ok=True)

    # 1. Integrity check + auto-fix.
    integrity = check_integrity(graph)
    (round_dir / "integrity_fixes.json").write_text(
        json.dumps(
            {
                "clean": integrity.clean,
                "errors": integrity.errors,
                "warnings": integrity.warnings,
                "fixes_applied": integrity.fixes_applied,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if not integrity.clean:
        # Don't even run the pool — surface errors and let the caller decide.
        return RoundResult(
            round_num=round_num,
            integrity=integrity,
            group_result_paths=[],
            issue_count=len(integrity.errors),
            outcome="needs_fix",
            notes=["integrity errors block semantic validation"],
        )

    nodes = graph.get("nodes") or []
    groups = _split_into_groups(nodes)

    # 2. Build pool tasks.
    pool = AgentPool(
        manager,
        n_workers=POOL_SIZE,
        log_dir=pool_log_dir,
        role="node_validator",
        name_prefix=f"validator_r{round_num}",
        timeout_s=timeout_s,
        stuck_timeout_s=max(60, timeout_s // 2),
        max_retries=2,
    )

    tasks: List[PoolTask] = []
    for i, group in enumerate(groups):
        prompt = _build_validator_prompt(
            group, graph,
            step1_path=step1_path, step2_path=step2_path,
            framework_dir=framework_dir,
        )
        workdir = round_dir / f"group_{i:02d}"
        workdir.mkdir(parents=True, exist_ok=True)
        ids_in_group = "_".join(str(n.get("id", "?")) for n in group)
        safe_ids = re.sub(r"[^A-Za-z0-9_.-]", "_", ids_in_group)[:80] or f"g{i}"
        tasks.append(PoolTask(
            key=f"r{round_num}_g{i:02d}_{safe_ids}",
            prompt=prompt,
            workdir=workdir,
            name=f"validator_r{round_num}_g{i:02d}",
        ))

    # 3. Run the pool (in input order).
    pool_results = pool.run(tasks)

    # 4. Persist + accumulate patches.
    group_result_paths: List[Path] = []
    total_issues = 0
    merged_verdicts: Dict[str, Any] = {}
    all_notes: List[str] = []

    for pr, task in zip(pool_results, tasks):
        out_path = round_dir / f"{task.workdir.name}.json"
        verdicts = _extract_json_object(pr.final_text) or {}
        applied, notes = apply_patches(graph, verdicts)
        total_issues += _count_issues(verdicts)
        payload = {
            "group_key": pr.key,
            "worker_id": pr.worker_id,
            "success": pr.success,
            "duration_s": pr.duration_s,
            "error": pr.error,
            "verdicts": verdicts,
            "issues_in_group": _count_issues(verdicts),
            "patches_applied": applied,
            "patch_notes": notes,
            "raw_final_text_head": (pr.final_text or "")[:500],
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        group_result_paths.append(out_path)
        merged_verdicts.update(verdicts)
        all_notes.extend(notes)

    outcome = "clean" if total_issues == 0 else "needs_fix"
    return RoundResult(
        round_num=round_num,
        integrity=integrity,
        group_result_paths=group_result_paths,
        issue_count=total_issues,
        outcome=outcome,
        notes=all_notes,
    )


def run_validation_loop(
    *,
    graph: Dict[str, Any],
    manager,
    step1_path: Path,
    step2_path: Path,
    framework_dir: Path,
    validation_root: Path,
    logs_root: Path,
    max_rounds: int = 5,
    timeout_s: int = 600,
) -> Tuple[List[RoundResult], bool]:
    """Iterate validation rounds until CLEAN or until ``max_rounds`` is hit.

    Returns ``(rounds, exhausted)``. ``exhausted=True`` means we hit the cap
    and the caller should write a best-effort warning before visualization.
    """
    rounds: List[RoundResult] = []
    for r in range(1, max_rounds + 1):
        round_dir = validation_root / f"round_{r:02d}"
        pool_log_dir = logs_root / f"validate_round_{r:02d}" / "pool"
        result = run_validation_round(
            round_num=r,
            graph=graph,
            manager=manager,
            step1_path=step1_path,
            step2_path=step2_path,
            framework_dir=framework_dir,
            round_dir=round_dir,
            pool_log_dir=pool_log_dir,
            timeout_s=timeout_s,
        )
        rounds.append(result)
        if result.outcome == "clean":
            return rounds, False

    return rounds, True
