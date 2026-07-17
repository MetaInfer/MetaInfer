"""Immutable native-runtime checks for the C++ framework task."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set

from ..capabilities import normalize_features


_PYTHON_EXE_RE = re.compile(r"^python(?:\d+(?:\.\d+)*)?(?:\.exe)?$", re.IGNORECASE)
_ACCELERATOR_LIBRARY_MARKERS = (
    "libamdhip",
    "libhip",
    "libhsa",
    "librocblas",
    "libhipblas",
    "librccl",
    "libcuda",
    "libcudart",
    "libcublas",
    "libnccl",
)


def validate_cpp_artifacts(iter_dir: Path) -> List[str]:
    """Return native delivery-contract violations before server startup."""
    errors: List[str] = []
    for filename in ("CMakeLists.txt", "serve.sh", "LANGUAGE_BOUNDARY.md"):
        if not (iter_dir / filename).is_file():
            errors.append(f"missing required native artifact: {filename}")
    for dirname in ("include", "src", "tests"):
        if not (iter_dir / dirname).is_dir():
            errors.append(f"missing required native directory: {dirname}/")

    native_sources = [
        path
        for suffix in ("*.cpp", "*.cc", "*.cxx", "*.hip", "*.cu")
        for path in (iter_dir / "src").rglob(suffix)
    ] if (iter_dir / "src").is_dir() else []
    if not native_sources:
        errors.append("src/ contains no C++/HIP translation unit")

    serve_sh = iter_dir / "serve.sh"
    if serve_sh.is_file():
        try:
            text = serve_sh.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            errors.append(f"cannot read serve.sh: {exc}")
        else:
            executable_lines = [
                line for line in text.splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
            executable_text = "\n".join(executable_lines)
            if re.search(r"\b(?:python(?:3)?|uvicorn|gunicorn)\b", executable_text):
                errors.append("serve.sh starts a Python process; the C++ service must be native")
    return errors


def hardware_validation_errors(req: Mapping[str, Any]) -> List[str]:
    profile = req.get("hardware_profile") or {}
    validation = profile.get("validation") or {}
    return [str(item) for item in validation.get("blockers") or []]


def native_accelerator_errors(
    req: Mapping[str, Any], evidence: Mapping[str, Any],
) -> List[str]:
    """Require concrete accelerator ownership in the native server process.

    Merely mapping ``libhipblas``/``libcuda`` is not execution evidence: a
    CPU-only server can link those libraries and never initialize a device.
    Conversely, an open render node without the selected runtime mapped does
    not prove that model operators use the requested backend. GPU tasks must
    show both signals, preferably on the same native process.
    """
    target = str(req.get("target_hardware") or "").lower()
    backend = str(req.get("accelerator_backend") or "").lower()
    requires_accelerator = any(
        token in f"{target} {backend}"
        for token in (
            "hygon", "nvidia", "amd", "cuda", "hip", "dtk",
            "h100", "a100", "mi300",
        )
    )
    if not requires_accelerator:
        return []

    libraries = list(evidence.get("loaded_accelerator_libraries") or [])
    device_fds = list(evidence.get("gpu_device_fds") or [])
    errors: List[str] = []
    if not libraries:
        errors.append(
            "native server process mapped no HIP/HSA/BLAS/CUDA runtime library"
        )
    if not device_fds:
        errors.append(
            "native server process opened no accelerator device FD "
            "(/dev/kfd, render node, or /dev/nvidia*)"
        )

    processes = list(evidence.get("processes") or [])
    if libraries and device_fds and processes:
        active_native = [
            proc for proc in processes
            if proc.get("exe_in_iteration")
            and not proc.get("is_python")
            and proc.get("accelerator_libraries")
            and proc.get("gpu_device_fds")
        ]
        if not active_native:
            errors.append(
                "accelerator library and device-FD evidence did not belong to "
                "the same native server process"
            )
        features = set(normalize_features(req.get("features")))
        if "tensor parallelism" in features:
            raw_tp = str(req.get("tensor_parallel_size") or "Auto").strip()
            if raw_tp.isdigit():
                expected_ranks = max(2, int(raw_tp))
            else:
                assigned = str(req.get("assigned_devices") or "")
                expected_ranks = max(
                    2, len([item for item in assigned.split(",") if item.strip()]),
                )
            if len(active_native) < expected_ranks:
                errors.append(
                    "Tensor parallelism requested "
                    f"{expected_ranks} native ranks but process evidence found "
                    f"{len(active_native)}"
                )
    return errors


def collect_native_process_evidence(root_pid: int, iter_dir: Path) -> Dict[str, Any]:
    """Inspect a Linux process tree after the HTTP server becomes healthy."""
    process_ids = _descendants_including(root_pid)
    processes: List[Dict[str, Any]] = []
    python_pids: List[int] = []
    native_pids: List[int] = []
    loaded_accelerator_libraries: Set[str] = set()
    gpu_device_fds: Set[str] = set()
    resolved_iter_dir = iter_dir.resolve()

    for pid in sorted(process_ids):
        proc_dir = Path("/proc") / str(pid)
        exe = _readlink(proc_dir / "exe")
        cwd = _readlink(proc_dir / "cwd")
        cmdline = _read_cmdline(proc_dir / "cmdline")
        exe_name = Path(exe).name if exe else ""
        is_python = bool(_PYTHON_EXE_RE.match(exe_name))
        if is_python:
            python_pids.append(pid)

        exe_path = _safe_resolve(exe)
        cwd_path = _safe_resolve(cwd)
        exe_in_iteration = bool(
            exe_path and _is_relative_to(exe_path, resolved_iter_dir)
        )
        cwd_in_iteration = bool(
            cwd_path and _is_relative_to(cwd_path, resolved_iter_dir)
        )
        is_shell = exe_name in {"bash", "sh", "dash", "zsh"}
        if exe_in_iteration and exe and not is_python and not is_shell:
            native_pids.append(pid)

        libraries = _accelerator_libraries(proc_dir / "maps")
        loaded_accelerator_libraries.update(libraries)
        process_gpu_fds = _gpu_device_fds(proc_dir / "fd")
        gpu_device_fds.update(process_gpu_fds)
        processes.append({
            "pid": pid,
            "exe": exe,
            "cwd": cwd,
            "cmdline": cmdline,
            "exe_in_iteration": exe_in_iteration,
            "cwd_in_iteration": cwd_in_iteration,
            "is_python": is_python,
            "accelerator_libraries": libraries,
            "gpu_device_fds": process_gpu_fds,
        })

    # On non-Linux systems /proc is unavailable. The generated server contract
    # targets Linux accelerators, so record that state explicitly.
    proc_available = Path("/proc").is_dir()
    errors: List[str] = []
    if proc_available and python_pids:
        errors.append(f"Python process(es) present in native server tree: {python_pids}")
    if proc_available and not native_pids:
        errors.append("no native executable from the iteration tree is serving HTTP")
    return {
        "root_pid": root_pid,
        "proc_available": proc_available,
        "processes": processes,
        "native_pids": native_pids,
        "python_pids": python_pids,
        "loaded_accelerator_libraries": sorted(loaded_accelerator_libraries),
        "gpu_device_fds": sorted(gpu_device_fds),
        "errors": errors,
    }


def write_native_evidence(path: Path, evidence: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")


def _descendants_including(root_pid: int) -> Set[int]:
    if not Path("/proc").is_dir():
        return {root_pid}
    parents: Dict[int, int] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        status = entry / "status"
        try:
            text = status.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        match = re.search(r"(?m)^PPid:\s*(\d+)\s*$", text)
        if match:
            parents[int(entry.name)] = int(match.group(1))
    result = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in result and pid not in result:
                result.add(pid)
                changed = True
    return result


def _readlink(path: Path) -> Optional[str]:
    try:
        return os.readlink(path)
    except OSError:
        return None


def _read_cmdline(path: Path) -> List[str]:
    try:
        raw = path.read_bytes()
    except OSError:
        return []
    return [part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part]


def _safe_resolve(value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    try:
        return Path(value).resolve()
    except OSError:
        return None


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _accelerator_libraries(maps_path: Path) -> List[str]:
    try:
        text = maps_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    libraries: Set[str] = set()
    for line in text.splitlines():
        lowered = line.lower()
        if not any(marker in lowered for marker in _ACCELERATOR_LIBRARY_MARKERS):
            continue
        path = line.rsplit(maxsplit=1)[-1]
        libraries.add(path)
    return sorted(libraries)


def _gpu_device_fds(fd_dir: Path) -> List[str]:
    devices: Set[str] = set()
    try:
        entries = list(fd_dir.iterdir())
    except OSError:
        return []
    for entry in entries:
        target = _readlink(entry)
        if not target:
            continue
        if (
            target == "/dev/kfd"
            or target.startswith("/dev/dri/renderD")
            or target.startswith("/dev/nvidia")
        ):
            devices.add(target)
    return sorted(devices)


__all__ = [
    "collect_native_process_evidence",
    "hardware_validation_errors",
    "native_accelerator_errors",
    "validate_cpp_artifacts",
    "write_native_evidence",
]
