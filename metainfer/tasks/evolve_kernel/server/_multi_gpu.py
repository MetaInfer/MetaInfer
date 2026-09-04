"""Multi-GPU parallel task launcher for evolve-kernel.

When the user selects "All GPUs (auto)" in the form, this module:
  1. Detects available GPUs
  2. Splits target shapes from extra_notes across GPUs
  3. Creates N child tasks (one per GPU), each optimizing a subset of shapes
  4. Creates a virtual "parent" task that aggregates results

Child tasks are normal evolve-kernel tasks, each with:
  - gpu_device: "0" / "1" / "2" / "3"
  - multi_gpu_parent: parent task ID
  - extra_notes: subset of shapes for this GPU

The parent task has a requirements.json with:
  - multi_gpu_children: ["child-id-1", "child-id-2", ...]
  - multi_gpu_mode: true
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from metainfer.server import launcher as _launcher
from metainfer.server import paths as _paths
from metainfer.server.registry import get as _get_web_plugin
from ._shape_bench import parse_shapes_from_extra_notes, ShapeSpec


# =========================================================================== #
# GPU detection
# =========================================================================== #


def detect_gpus() -> int:
    """Detect number of available GPUs. Tries torch, then rocm-smi, then env."""
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        pass
    try:
        import subprocess
        result = subprocess.run(
            ["rocm-smi", "--showid", "--csv"],
            capture_output=True, text=True, timeout=5,
        )
        lines = [l for l in result.stdout.strip().split("\n") if l.strip() and "GPU" not in l]
        return len(lines) if lines else 1
    except Exception:
        pass
    cu = _os_environ("CUDA_VISIBLE_DEVICES")
    if cu:
        return max(1, cu.count(",") + 1)
    return 1


def _os_environ(key: str) -> str:
    import os
    return os.environ.get(key, "")


# =========================================================================== #
# Shape splitting
# =========================================================================== #


def split_shapes_for_gpus(
    extra_notes: str,
    num_gpus: int,
) -> List[Tuple[str, List[ShapeSpec]]]:
    """Split parsed shapes across GPUs.

    Strategy:
      1. Parse shapes from extra_notes
      2. If multiple distinct shapes (different label), assign roughly evenly
      3. If only one shape, each GPU gets the same shape (competing optimization)
      4. Returns list of (gpu_label, shapes) tuples
    """
    shapes = parse_shapes_from_extra_notes(extra_notes)
    if not shapes:
        return []

    # If only one shape, replicate across all GPUs
    if len(shapes) == 1:
        return [
            ("GPU {}".format(i), list(shapes))
            for i in range(num_gpus)
        ]

    # Multiple shapes: distribute evenly
    groups: List[List[ShapeSpec]] = [[] for _ in range(num_gpus)]
    for i, s in enumerate(shapes):
        groups[i % num_gpus].append(s)

    return [
        ("GPU {}".format(i), grp)
        for i, grp in enumerate(groups)
        if grp  # skip empty groups
    ]


def shapes_to_notes(shapes: List[ShapeSpec]) -> str:
    """Convert ShapeSpec list back to extra_notes text."""
    lines = []
    for s in shapes:
        m_str = ",".join(str(m) for m in s.M_values)
        lines.append(
            "{}: M={} (M, {}) @ ({}, {})".format(
                s.label, m_str, s.K, s.K, s.N,
            )
        )
    return "\n".join(lines)


# =========================================================================== #
# Child task creation
# =========================================================================== #


def create_child_task(
    parent_id: str,
    gpu_index: int,
    child_shapes: List[ShapeSpec],
    base_requirements: Dict[str, Any],
    label_suffix: str,
) -> str:
    """Create one child task pinned to a specific GPU.

    Args:
        parent_id: The parent task ID for linkage.
        gpu_index: Which GPU index to bind to (0, 1, 2, 3).
        child_shapes: Subset of shapes for this child.
        base_requirements: The user's original form answers.
        label_suffix: Display label suffix, e.g. "GPU 0".

    Returns:
        child task ID.
    """
    import metainfer.server.tasks as _tasks

    child_id = _tasks.gen_task_id("evolve-kernel", "{}-gpu{}".format(parent_id, gpu_index))

    sd = _paths.task_dir(child_id)
    sd.mkdir(parents=True, exist_ok=True)
    wd = _paths.workspace_dir(child_id)
    wd.mkdir(parents=True, exist_ok=True)

    # Build child requirements: copy base but override shapes and GPU.
    # Strip multi_gpu so child tasks don't recursively trigger more launches.
    child_req = dict(base_requirements)
    child_req.pop("multi_gpu", None)
    child_req.pop("gpu_count", None)
    child_req["task_id"] = child_id
    child_req["gpu_device"] = str(gpu_index)
    child_req["multi_gpu_parent"] = parent_id
    child_req["extra_notes"] = shapes_to_notes(child_shapes)

    plugin = _get_web_plugin("evolve-kernel")
    default_label = (plugin.label if plugin else "") or "evolve-kernel"
    label = "{} — {}".format(default_label, label_suffix)

    # Register child task
    entry = _tasks.TaskEntry(
        id=child_id, type="evolve-kernel", label=label,
        state_dir=str(sd), workspace_dir=str(wd),
        created_at=time.time(), launcher="local",
    )
    _tasks.add_task(entry)

    # Spawn child orchestrator
    child_launcher = _launcher.get_default_launcher()
    child_launcher.start(child_id, child_req, sd, wd)

    return child_id


def launch_multi_gpu(
    base_requirements: Dict[str, Any],
    num_gpus: Optional[int] = None,
    label: str = "",
) -> Dict[str, Any]:
    """Create a multi-GPU optimization batch.

    Args:
        base_requirements: The user's form answers (as if creating one task).
        num_gpus: Number of GPUs to use. Auto-detected if None.
        label: User-provided label for the parent task.

    Returns:
        {"parent_id": "...", "children": ["child1", "child2", ...]}
    """
    import metainfer.server.tasks as _tasks

    if num_gpus is None:
        num_gpus = detect_gpus()
    num_gpus = max(1, min(num_gpus, 8))  # clamp 1-8

    extra_notes = base_requirements.get("extra_notes", "")

    # Split shapes
    groups = split_shapes_for_gpus(extra_notes, num_gpus)
    if not groups:
        raise ValueError(
            "No shapes found in extra_notes. Cannot split across GPUs. "
            "Add shapes using the format: shape_name: (M, K) @ (K, N)"
        )

    # Create parent task ID
    parent_id = _tasks.gen_task_id("evolve-kernel", "{}-multi".format(label or "kernel"))

    # Create parent task entries (a virtual task — no orchestrator)
    plugin = _get_web_plugin("evolve-kernel")
    default_label = (plugin.label if plugin else "") or "evolve-kernel"
    parent_label = "{} [{} GPUs]".format(label or default_label, num_gpus)

    sd = _paths.task_dir(parent_id)
    sd.mkdir(parents=True, exist_ok=True)
    wd = _paths.workspace_dir(parent_id)
    wd.mkdir(parents=True, exist_ok=True)

    # Write parent requirements
    parent_req = dict(base_requirements)
    parent_req["task_id"] = parent_id
    parent_req["multi_gpu_mode"] = True
    parent_req["multi_gpu_children"] = []
    req_path = sd / "requirements.json"
    req_path.write_text(json.dumps(parent_req, indent=2), encoding="utf-8")

    # Write parent run.json (virtual task, marked finished immediately)
    run_data = {
        "task_id": parent_id,
        "current_phase": "idle",
        "finished": False,
        "final_status": None,
    }
    (sd / "run.json").write_text(json.dumps(run_data, indent=2), encoding="utf-8")

    # Register parent
    entry = _tasks.TaskEntry(
        id=parent_id, type="evolve-kernel", label=parent_label,
        state_dir=str(sd), workspace_dir=str(wd),
        created_at=time.time(), launcher="local",
    )
    _tasks.add_task(entry)

    # Create children
    children: List[str] = []
    for gpu_label, child_shapes in groups:
        gpu_idx = int(gpu_label.replace("GPU ", ""))
        try:
            child_id = create_child_task(
                parent_id=parent_id,
                gpu_index=gpu_idx,
                child_shapes=child_shapes,
                base_requirements=base_requirements,
                label_suffix=gpu_label,
            )
            children.append(child_id)
        except Exception:
            # Continue creating other children
            import traceback
            traceback.print_exc()

    # Update parent with child IDs
    parent_req["multi_gpu_children"] = children
    req_path.write_text(json.dumps(parent_req, indent=2), encoding="utf-8")

    return {"parent_id": parent_id, "children": children, "num_gpus": num_gpus}


# =========================================================================== #
# Aggregated results
# =========================================================================== #


def get_child_task_ids(state_dir: Path) -> List[str]:
    """Read the child task IDs from a parent task's requirements."""
    req_path = state_dir / "requirements.json"
    if not req_path.is_file():
        return []
    try:
        req = json.loads(req_path.read_text(encoding="utf-8"))
        if req.get("multi_gpu_mode"):
            return req.get("multi_gpu_children", [])
    except Exception:
        pass
    return []


def aggregate_status(task_ids: List[str]) -> List[Dict[str, Any]]:
    """Collect status for each child task."""
    launcher = _launcher.get_default_launcher()
    results = []
    for tid in task_ids:
        entry_data = None
        from metainfer.server import tasks as _tasks
        entry = _tasks.get_task(tid)
        if entry:
            entry_data = {
                "id": entry.id,
                "label": entry.label,
                "type": entry.type,
                "state_dir": entry.state_dir,
                "workspace_dir": entry.workspace_dir,
            }
        status = launcher.status(tid)
        results.append({
            "task_id": tid,
            "entry": entry_data,
            "running": status.running,
            "pid": status.pid,
        })
    return results


def aggregate_shape_benchmarks(workspace_dirs: List[str]) -> List[Dict[str, Any]]:
    """Aggregate shape benchmark results from all child tasks.

    Reads shape_bench.json from each child's workspace, merges into one table
    with a 'source' column indicating which task produced each row.
    """
    all_results: List[Dict[str, Any]] = []
    for wd_str in workspace_dirs:
        bench_path = Path(wd_str) / "shape_bench.json"
        if not bench_path.is_file():
            continue
        try:
            data = json.loads(bench_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        task_id = data.get("best_kernel_id", "?")[:8]
        for r in data.get("results", []):
            r["source"] = task_id
            all_results.append(r)

    # Sort: by shape_label, then by M
    all_results.sort(key=lambda r: (r.get("shape_label", ""), r.get("M", 0)))
    return all_results


def aggregate_best_kernels(workspace_dirs: List[str]) -> List[Dict[str, Any]]:
    """Collect the best kernel from each child's library."""
    kernels = []
    for wd_str in workspace_dirs:
        lib_path = Path(wd_str) / "kernel_library.json"
        if not lib_path.is_file():
            continue
        try:
            lib = json.loads(lib_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not lib:
            continue
        # Sort by exec_time_ms
        lib.sort(key=lambda k: k.get("exec_time_ms", float("inf")))
        best = lib[0]
        best["source_workspace"] = str(Path(wd_str).parent.name)
        kernels.append(best)
    return kernels
