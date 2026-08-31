"""Generate, inspect, and explicitly publish optimization skills."""

from __future__ import annotations

import difflib
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable

from .config import OptimizerConfig, WorkerAssignment, resolve_claude_bin


def _slug(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return value[:63] or "kernel-optimization"


def _frontmatter(name: str, description: str) -> str:
    return (
        "---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        "---\n\n"
    )


def _write_candidate(
    pending_root: Path,
    *,
    name: str,
    kind: str,
    source: str,
    content: str,
) -> Dict[str, Any]:
    target = pending_root / name
    target.mkdir(parents=True, exist_ok=True)
    (target / "SKILL.md").write_text(content, encoding="utf-8")
    manifest = {
        "name": name,
        "kind": kind,
        "source": source,
        "status": "pending",
        "created_at": time.time(),
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _read_experiments(worker_root: Path) -> list[Dict[str, Any]]:
    experiments: list[Dict[str, Any]] = []
    for path in sorted((worker_root / "runs").glob("*/experiments.jsonl")):
        for line in path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            try:
                item = json.loads(line)
            except ValueError:
                continue
            if isinstance(item, dict):
                experiments.append(item)
    return experiments


def generate_worker_skill(
    *,
    config: OptimizerConfig,
    assignment: WorkerAssignment,
    workspace_dir: Path,
    agent_draft: str | None = None,
) -> Dict[str, Any]:
    """Write one skill after a worker has finished all assigned shapes."""
    worker_root = workspace_dir / "workers" / assignment.worker_id
    experiments = _read_experiments(worker_root)
    shape_label = "-".join(assignment.shape_ids)
    name = _slug(
        f"{config.dtype}-{config.operator}-{assignment.worker_id}-{shape_label}"
    )
    description = (
        f"Optimize {config.dtype} {config.operator} on {config.hardware} for "
        f"the shape group {', '.join(assignment.shape_ids)}. Use when tuning "
        "a matching kernel shape or transferring measured worker findings."
    )
    if agent_draft and agent_draft.strip():
        content = _frontmatter(name, description) + agent_draft.strip() + "\n"
        return _write_candidate(
            workspace_dir / "skills" / "pending",
            name=name,
            kind="worker",
            source=assignment.worker_id,
            content=content,
        )

    lines = [
        _frontmatter(name, description),
        f"# {config.dtype} {config.operator}: {assignment.worker_id}",
        "",
        "## Scope",
        "",
        f"- Hardware: {config.hardware}",
        f"- Stack: {config.kernel_language}",
        f"- GPU worker: {assignment.worker_id} / physical GPU {assignment.gpu}",
        f"- Shapes: {', '.join(assignment.shape_ids)}",
        "",
        "## Measured results",
        "",
        "| Shape | Round | Plan | Median (us) | INT8 TOPS | Algorithmic BW (GB/s) | Accepted |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    for item in experiments:
        metrics = item.get("metrics") or {}
        plan = str(item.get("hypothesis") or "").replace("|", "/")
        lines.append(
            f"| {item.get('shape_id', '—')} | {item.get('iteration', '—')} | "
            f"{plan} | {metrics.get('median_us', '—')} | "
            f"{metrics.get('logical_tops', metrics.get('tflops', '—'))} | "
            f"{metrics.get('algorithmic_bandwidth_gb_s', metrics.get('bandwidth_gb_s', '—'))} | "
            f"{'yes' if item.get('accepted') else 'no'} |"
        )
    manual = [
        str(item["manual_guidance"])
        for item in experiments if item.get("manual_guidance")
    ]
    lines.extend([
        "",
        "## Optimization procedure",
        "",
        "1. Match the incoming dimensions to the measured shape group.",
        "2. Start from the best accepted candidate and preserve the trusted reference.",
        "3. Change one optimization variable per round and record the hypothesis.",
        "4. Reject correctness failures and unstable median or P90 regressions.",
        "5. Re-run the winner in serial validation before integration.",
    ])
    if manual:
        lines.extend([
            "",
            "## Human guidance applied",
            "",
            *[f"- {text}" for text in dict.fromkeys(manual)],
        ])
    lines.extend([
        "",
        "Treat these results as shape-specific evidence; remeasure before "
        "applying them outside the listed dimensions.",
        "",
    ])
    return _write_candidate(
        workspace_dir / "skills" / "pending",
        name=name,
        kind="worker",
        source=assignment.worker_id,
        content="\n".join(lines),
    )


def generate_merged_skill(
    *,
    config: OptimizerConfig,
    assignments: Iterable[WorkerAssignment],
    workspace_dir: Path,
    agent_draft: str | None = None,
    failed_workers: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Create the main-agent synthesis after every worker skill exists."""
    assignments = list(assignments)
    name = _slug(f"{config.dtype}-{config.operator}-optimization")
    description = (
        f"Optimize {config.dtype} {config.operator} across multiple shape "
        f"regimes on {config.hardware}. Use when selecting, tuning, validating, "
        "or integrating a kernel covered by the measured shape families."
    )
    if agent_draft and agent_draft.strip():
        content = _frontmatter(name, description) + agent_draft.strip() + "\n"
        return _write_candidate(
            workspace_dir / "skills" / "pending",
            name=name,
            kind="merged",
            source="main_agent",
            content=content,
        )

    lines = [
        _frontmatter(name, description),
        f"# {config.dtype} {config.operator} optimization",
        "",
        "## Shape routing",
        "",
        "| Worker evidence | Physical GPU | Shapes |",
        "|---|---:|---|",
    ]
    for assignment in assignments:
        lines.append(
            f"| {assignment.worker_id} | {assignment.gpu} | "
            f"{', '.join(assignment.shape_ids)} |"
        )
    if failed_workers:
        lines.extend([
            "",
            "## Unavailable worker evidence",
            "",
            *[
                f"- `{worker_id}`: {failure.get('error', 'worker failed')}"
                for worker_id, failure in sorted(failed_workers.items())
            ],
        ])
    lines.extend([
        "",
        "## Workflow",
        "",
        "1. Route the target shape to the closest measured shape family.",
        "2. Read that worker skill before choosing the first candidate.",
        "3. Preserve quantization, scale, reference, build, and call contracts.",
        "4. Compare stable median, P90, TFLOPS, and bandwidth every round.",
        "5. Feed human guidance only to the selected worker's next round.",
        "6. Validate winners serially and keep per-shape fallbacks.",
        "",
        "## Evidence files",
        "",
    ])
    for assignment in assignments:
        worker_name = _slug(
            f"{config.dtype}-{config.operator}-{assignment.worker_id}-"
            f"{'-'.join(assignment.shape_ids)}"
        )
        lines.append(f"- `{worker_name}/SKILL.md`")
    lines.extend([
        "",
        "Do not infer an unmeasured shape rule solely from a nearby result. "
        "Benchmark the new shape and append evidence before generalizing.",
        "",
    ])
    return _write_candidate(
        workspace_dir / "skills" / "pending",
        name=name,
        kind="merged",
        source="main_agent",
        content="\n".join(lines),
    )


def dsh_skills_root() -> Path:
    """Authoritative skill library (DeepSeek Harness).

    dsh is the canonical library: the fusion agent scans it, and new/updated
    skills land here first. ``sync_skill_libraries()`` then mirrors them into
    the ccb (Claude Code) library so both agent frameworks share one set of
    skills.
    """
    for env in ("DSH_SKILLS_DIR", "METAINFER_SKILLS_DIR"):
        configured = os.environ.get(env)
        if configured:
            return Path(configured).expanduser()
    return Path.home() / ".dsh" / "skills"


def ccb_skills_root() -> Path:
    """Claude Code skill library — the mirror target for dsh skills."""
    configured = os.environ.get("METAINFER_CLAUDE_SKILLS_DIR")
    return (
        Path(configured).expanduser()
        if configured else Path.home() / ".claude" / "skills"
    )


def existing_skills_root() -> Path:
    """The library publish/fuse write into (dsh, the authoritative library)."""
    return dsh_skills_root()


def _skill_entries(root: Path, *, status: str) -> list[Dict[str, Any]]:
    entries = []
    if not root.exists():
        return entries
    for path in sorted(root.glob("*/SKILL.md")):
        manifest_path = path.parent / "manifest.json"
        manifest: Dict[str, Any] = {}
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        entries.append({
            "name": path.parent.name,
            "status": status,
            "kind": manifest.get("kind", "existing"),
            "source": manifest.get("source", "claude"),
            "path": str(path),
            "content": path.read_text(encoding="utf-8", errors="replace"),
        })
    return entries


def _frontmatter_description(content: str) -> str:
    """Pull the one-line description out of a SKILL.md frontmatter block."""
    match = re.match(r"^---\n(.*?)\n---\n", content, flags=re.DOTALL)
    if not match:
        return ""
    desc = re.search(r"(?m)^description:\s*(.+)$", match.group(1))
    if not desc:
        return ""
    return desc.group(1).strip().strip('"').strip("'")


def _copy_skill_dir(src: Path, dst: Path) -> None:
    """Copy one skill directory, skipping manifests and Windows Zone junk."""
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.name == "manifest.json" or "Zone.Identifier" in item.name:
            continue
        if item.is_dir():
            shutil.copytree(item, dst / item.name, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst / item.name)


def _backup_skill(skill_dir: Path) -> Path | None:
    """Snapshot SKILL.md as ``SKILL.md.bak-<ms>`` for rollback."""
    skill = skill_dir / "SKILL.md"
    if not skill.is_file():
        return None
    backup = skill_dir / f"SKILL.md.bak-{int(time.time() * 1000)}"
    shutil.copy2(skill, backup)
    return backup


def _skill_name_set(root: Path) -> set[str]:
    if not root.exists():
        return set()
    return {path.parent.name for path in root.glob("*/SKILL.md")}


# --------------------------------------------------------------------------- #
# dsh -> ccb one-way mirror
# --------------------------------------------------------------------------- #

def _sync_summary_path(workspace_dir: Path) -> Path:
    return workspace_dir / "skills" / "sync.json"


def read_sync_summary(workspace_dir: Path) -> Dict[str, Any]:
    try:
        return json.loads(_sync_summary_path(workspace_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_sync_summary(workspace_dir: Path, summary: Dict[str, Any]) -> None:
    _sync_summary_path(workspace_dir).parent.mkdir(parents=True, exist_ok=True)
    _sync_summary_path(workspace_dir).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sync_skill_libraries(*, workspace_dir: Path | None = None) -> Dict[str, Any]:
    """Mirror the authoritative dsh library into the ccb library (one-way).

    Copies skills that are new or changed in dsh into ccb; never deletes
    ccb-only skills. A ccb SKILL.md is backed up before being overwritten so
    a bad mirror can be rolled back. Idempotent and safe to call repeatedly.
    """
    source = dsh_skills_root()
    target = ccb_skills_root()
    added: list[str] = []
    updated: list[str] = []
    skipped: list[str] = []
    if source.exists():
        target.mkdir(parents=True, exist_ok=True)
        for src_dir in sorted(source.glob("*/")):
            src_skill = src_dir / "SKILL.md"
            if not src_skill.is_file():
                continue
            name = src_dir.name
            dst = target / name
            dst_skill = dst / "SKILL.md"
            if not dst_skill.is_file():
                _copy_skill_dir(src_dir, dst)
                added.append(name)
            elif dst_skill.read_bytes() != src_skill.read_bytes():
                _backup_skill(dst)
                _copy_skill_dir(src_dir, dst)
                updated.append(name)
            else:
                skipped.append(name)
    summary = {
        "source": str(source),
        "target": str(target),
        "added": added,
        "updated": updated,
        "skipped": skipped,
        "ccb_only": sorted(_skill_name_set(target) - _skill_name_set(source)),
        "ts": time.time(),
    }
    if workspace_dir is not None:
        _write_sync_summary(workspace_dir, summary)
    return summary


def rollback_skill(name: str, *, workspace_dir: Path) -> Dict[str, Any]:
    """Restore the latest SKILL.md backup in the dsh library, then re-mirror."""
    skill_dir = dsh_skills_root() / name
    if not skill_dir.is_dir():
        raise FileNotFoundError(f"skill not found in dsh library: {name}")
    backups = sorted(skill_dir.glob("SKILL.md.bak-*"))
    if not backups:
        raise FileNotFoundError(f"no backup available for skill {name}")
    latest = backups[-1]
    shutil.copy2(latest, skill_dir / "SKILL.md")
    latest.unlink()
    sync = sync_skill_libraries(workspace_dir=workspace_dir)
    return {
        "name": name,
        "restored_from": str(latest),
        "synced_to_ccb": sync,
    }


# --------------------------------------------------------------------------- #
# Main-agent skill fusion (UI-triggered)
# --------------------------------------------------------------------------- #

def _fuse_status_path(workspace_dir: Path) -> Path:
    return workspace_dir / "skills" / "fuse" / "status.json"


def read_fuse_status(workspace_dir: Path) -> Dict[str, Any]:
    try:
        return json.loads(_fuse_status_path(workspace_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"status": "idle"}


def mark_fuse_running(workspace_dir: Path, skill_name: str) -> None:
    _fuse_status_path(workspace_dir).parent.mkdir(parents=True, exist_ok=True)
    _fuse_status_path(workspace_dir).write_text(
        json.dumps({
            "status": "running",
            "skill_name": skill_name,
            "ts": time.time(),
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_fuse_status(workspace_dir: Path, payload: Dict[str, Any]) -> None:
    _fuse_status_path(workspace_dir).parent.mkdir(parents=True, exist_ok=True)
    _fuse_status_path(workspace_dir).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _build_fuse_prompt(
    existing: list[Dict[str, Any]],
    pending_content: str,
    decision_path: Path,
) -> str:
    lines = [
        "You are the skill-fusion agent for DCU kernel optimization.",
        "",
        "Fold the task's newly written optimization skill into the existing",
        "DeepSeek Harness skill library so future agents reuse it. The library",
        "is mirrored to Claude Code automatically, so this one decision applies",
        "to both frameworks.",
        "",
        "## Existing skills in the library (name, description, body excerpt)",
        "",
    ]
    if not existing:
        lines.append("(the library is empty — add a new skill)")
    for item in existing:
        lines.append(f"### {item.get('name')}")
        lines.append(f"description: {item.get('description') or ''}")
        body = item.get("content") or ""
        if body.count("---\n") >= 2:
            first = body.find("---\n")
            body = body[body.find("---\n", first + 1) + 4:]
        body = body.strip()
        if len(body) > 1200:
            body = body[:1200] + "\n... (truncated; read the full file with tools if needed)"
        lines.append(body)
        lines.append("")
    lines.extend([
        "## New skill to fuse",
        "",
        pending_content,
        "",
        "## Decision contract",
        "",
        "Decide whether the new skill should be ADDED as a brand-new skill file or",
        "MERGED into one of the existing skills above (only when the new material is",
        "a natural extension of an existing skill's scope).",
        "",
        f"Write ONE JSON object to `{decision_path}`. No markdown fences, no prose",
        "around it. Schema:",
        '{"action": "new" | "merge", "name": "<skill-name>", "description": "<one-line>", "content": "<complete markdown body, no YAML frontmatter>"}',
        "",
        '- action "new": name must be a short kebab-case id (<=63 chars) that does',
        "  not collide with an existing skill; content is the complete body.",
        '- action "merge": name must be EXACTLY an existing skill\'s name; content is',
        "  the complete UPDATED body after folding the new material in.",
        "- Preserve all existing working knowledge; add the new evidence/rules where",
        "  they fit.",
        "- The body must be concise, actionable Markdown for a kernel-optimization",
        "  agent that runs on DCU (gfx928).",
    ])
    return "\n".join(lines) + "\n"


def _parse_fuse_decision(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        decision = json.loads(cleaned)
    except ValueError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(
                "skill fusion agent did not return a JSON decision"
            ) from None
        try:
            decision = json.loads(cleaned[start:end + 1])
        except ValueError as exc:
            raise ValueError(
                "skill fusion agent returned unparseable JSON"
            ) from exc
    if not isinstance(decision, dict):
        raise ValueError("skill fusion decision must be a JSON object")
    action = str(decision.get("action") or "").strip().lower()
    if action not in {"new", "merge"}:
        raise ValueError(
            f"skill fusion decision action must be new or merge, got {action!r}"
        )
    name = _slug(str(decision.get("name") or ""))
    if not name:
        raise ValueError("skill fusion decision missing name")
    content = str(decision.get("content") or "").strip()
    if not content:
        raise ValueError("skill fusion decision missing content")
    description = str(decision.get("description") or "").strip()
    return {
        "action": action,
        "name": name,
        "description": description,
        "content": content,
    }


def _apply_fuse_decision(decision: Dict[str, Any]) -> Dict[str, Any]:
    """Apply a parsed fusion decision into the dsh library."""
    root = dsh_skills_root()
    root.mkdir(parents=True, exist_ok=True)
    action = decision["action"]
    name = decision["name"]
    content = decision["content"]
    description = decision["description"]

    if action == "new":
        target = root / name
        if target.exists():
            raise FileExistsError(
                f"skill {name} already exists in the dsh library; "
                "choose merge or a new name"
            )
        target.mkdir(parents=True, exist_ok=True)
        body = _frontmatter(
            name, description or f"{name} optimization skill"
        ) + content + "\n"
        (target / "SKILL.md").write_text(body, encoding="utf-8")
        return {"action": "new", "name": name, "path": str(target / "SKILL.md")}

    target = root / name
    if not (target / "SKILL.md").is_file():
        raise FileNotFoundError(
            f"merge target skill {name} does not exist in the dsh library"
        )
    previous = (target / "SKILL.md").read_text(encoding="utf-8", errors="replace")
    backup = _backup_skill(target)
    # Keep the existing frontmatter (the dir name is the skill identity);
    # only the body is replaced.
    fm = re.match(r"^(---\n.*?\n---\n)", previous, flags=re.DOTALL)
    body = (fm.group(1) if fm else _frontmatter(name, description or f"{name} optimization skill")) + content + "\n"
    (target / "SKILL.md").write_text(body, encoding="utf-8")
    diff = "".join(difflib.unified_diff(
        previous.splitlines(keepends=True),
        body.splitlines(keepends=True),
        fromfile=f"{name}/SKILL.md (before)",
        tofile=f"{name}/SKILL.md (after)",
    ))
    return {
        "action": "merge",
        "name": name,
        "path": str(target / "SKILL.md"),
        "backup": str(backup) if backup else None,
        "diff": diff,
    }


def fuse_skill(
    *,
    config: OptimizerConfig,
    workspace_dir: Path,
    state_dir: Path,
    skill_name: str,
) -> Dict[str, Any]:
    """Fuse one pending skill into the dsh library via the task's main agent.

    Runs the agent with the same framework/model as the task (ccb or dsh).
    The agent scans the existing dsh library, writes a JSON decision
    (new file vs merge into an existing skill), the decision is applied to
    the dsh library, and the result is mirrored to the ccb library.
    """
    pending = workspace_dir / "skills" / "pending" / skill_name
    if not (pending / "SKILL.md").is_file():
        raise FileNotFoundError(f"pending skill not found: {skill_name}")
    pending_content = (pending / "SKILL.md").read_text(
        encoding="utf-8", errors="replace"
    )

    existing = [
        {
            "name": item["name"],
            "description": _frontmatter_description(item["content"]),
            "content": item["content"],
        }
        for item in _skill_entries(dsh_skills_root(), status="existing")
    ]

    fuse_dir = workspace_dir / "skills" / "fuse"
    fuse_dir.mkdir(parents=True, exist_ok=True)
    decision_path = fuse_dir / "decision.json"
    decision_path.unlink(missing_ok=True)
    prompt_file = fuse_dir / "prompt.txt"
    prompt_file.write_text(
        _build_fuse_prompt(existing, pending_content, decision_path),
        encoding="utf-8",
    )

    from metainfer.orchestrator._bootstrap import make_subagent_manager
    from metainfer.orchestrator.subagent_manager import AgentSpec

    manager = make_subagent_manager(
        claude_bin=resolve_claude_bin(config.agent_framework),
        model=config.claude_model,
        permission_mode="bypassPermissions",
        effort="max",
        extra_add_dirs=[workspace_dir, dsh_skills_root()],
        snapshot_file=state_dir / "agents.json",
        max_concurrent=1,
    )
    try:
        manager.launch(AgentSpec(
            name="skill-fusion",
            role="dcu_skill_fusion",
            prompt_file=prompt_file,
            workdir=fuse_dir,
            log_dir=fuse_dir / "logs",
            timeout_s=600,
            stuck_timeout_s=300,
            max_retries=1,
        ))
        agent_result = manager.result("skill-fusion")
        if agent_result is None or not agent_result.success:
            raise RuntimeError(
                agent_result.error if agent_result and agent_result.error
                else "skill fusion agent failed"
            )
        if not decision_path.is_file():
            raise RuntimeError(
                "skill fusion agent did not write a decision file"
            )
        decision = _parse_fuse_decision(
            decision_path.read_text(encoding="utf-8", errors="replace")
        )
        applied = _apply_fuse_decision(decision)
        sync = sync_skill_libraries(workspace_dir=workspace_dir)
        outcome = {
            **applied,
            "synced_to_ccb": {
                "added": sync["added"],
                "updated": sync["updated"],
            },
        }
        _write_fuse_status(workspace_dir, {
            "status": "done", "ok": True, **outcome, "ts": time.time(),
        })
        return outcome
    except Exception as exc:  # noqa: BLE001 - surfaced via the status file
        _write_fuse_status(workspace_dir, {
            "status": "error", "ok": False,
            "error": str(exc), "ts": time.time(),
        })
        raise
    finally:
        manager.close()


def list_skill_library(workspace_dir: Path) -> Dict[str, Any]:
    pending_root = workspace_dir / "skills" / "pending"
    return {
        "existing_root": str(dsh_skills_root()),
        "ccb_mirror_root": str(ccb_skills_root()),
        "existing": _skill_entries(dsh_skills_root(), status="existing"),
        "pending": _skill_entries(pending_root, status="pending"),
        "fuse_status": read_fuse_status(workspace_dir),
        "sync": read_sync_summary(workspace_dir),
    }


def publish_skill(workspace_dir: Path, name: str) -> Dict[str, Any]:
    """Publish one exact pending directory into the dsh library, then mirror
    the result to the ccb library. Never overwrites an existing skill."""
    if _slug(name) != name:
        raise ValueError("invalid skill name")
    source = workspace_dir / "skills" / "pending" / name
    if not (source / "SKILL.md").is_file():
        raise FileNotFoundError(f"pending skill not found: {name}")
    root = dsh_skills_root()
    root.mkdir(parents=True, exist_ok=True)
    target = root / name
    if target.exists():
        raise FileExistsError(f"skill already exists: {name}")
    shutil.copytree(
        source, target, ignore=shutil.ignore_patterns("manifest.json")
    )
    published = workspace_dir / "skills" / "published" / name
    published.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(published))
    sync = sync_skill_libraries(workspace_dir=workspace_dir)
    return {
        "name": name,
        "path": str(target),
        "status": "existing",
        "synced_to_ccb": {"added": sync["added"], "updated": sync["updated"]},
    }
