"""Generate, inspect, and explicitly publish optimization skills."""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable

from .config import OptimizerConfig, WorkerAssignment


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


def existing_skills_root() -> Path:
    configured = os.environ.get("METAINFER_CLAUDE_SKILLS_DIR")
    return (
        Path(configured).expanduser()
        if configured else Path.home() / ".claude" / "skills"
    )


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


def list_skill_library(workspace_dir: Path) -> Dict[str, Any]:
    existing_root = existing_skills_root()
    pending_root = workspace_dir / "skills" / "pending"
    return {
        "existing_root": str(existing_root),
        "existing": _skill_entries(existing_root, status="existing"),
        "pending": _skill_entries(pending_root, status="pending"),
    }


def publish_skill(workspace_dir: Path, name: str) -> Dict[str, Any]:
    """Publish one exact pending directory, never overwriting an existing skill."""
    if _slug(name) != name:
        raise ValueError("invalid skill name")
    source = workspace_dir / "skills" / "pending" / name
    if not (source / "SKILL.md").is_file():
        raise FileNotFoundError(f"pending skill not found: {name}")
    root = existing_skills_root()
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
    return {"name": name, "path": str(target), "status": "existing"}
