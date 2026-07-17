"""Read-only accelerator discovery for inference-framework tasks.

The detector runs in the orchestrator process, on the same host and with the
same environment as generated code.  It intentionally uses a small command
allow-list and never invokes a shell, sudo, package managers, or mutating SMI
operations.  Product labels are evidence, not compiler targets: PCI, SMI and
HIP runtime names are preserved independently so vendor aliases do not hide
useful disagreements (for example PCI ``Z200SM`` vs SMI ``Z100SM``).
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import socket
import stat
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


SCHEMA_VERSION = 1
COMMAND_TIMEOUT_S = 10
_DTK_BIN_DIRS = (Path("/opt/dtk/bin"), Path("/opt/dtk/llvm/bin"))
_VISIBLE_DEVICE_VARS = (
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
)


def _find_tool(name: str, extra_dirs: Iterable[Path] = _DTK_BIN_DIRS) -> Optional[str]:
    found = shutil.which(name)
    if found:
        return found
    for directory in extra_dirs:
        candidate = directory / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _run_read_only(command: Sequence[str], timeout_s: int = COMMAND_TIMEOUT_S) -> Dict[str, Any]:
    """Run one allow-listed command without a shell and return bounded evidence."""
    started = time.monotonic()
    try:
        proc = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout_s,
            check=False,
        )
        return {
            "available": True,
            "returncode": proc.returncode,
            "stdout": proc.stdout[:65536],
            "stderr": proc.stderr[:8192],
            "duration_ms": round((time.monotonic() - started) * 1000, 1),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "available": True,
            "returncode": None,
            "stdout": _coerce_output(exc.stdout)[:65536],
            "stderr": "command timed out",
            "duration_ms": round((time.monotonic() - started) * 1000, 1),
        }
    except OSError as exc:
        return {
            "available": False,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
            "duration_ms": round((time.monotonic() - started) * 1000, 1),
        }


def _coerce_output(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def parse_lspci(output: str) -> List[Dict[str, Any]]:
    """Parse ``lspci -D -nn`` display/3D controllers."""
    devices: List[Dict[str, Any]] = []
    for line in output.splitlines():
        raw = line.strip()
        lowered = raw.lower()
        if not raw or not any(
            marker in lowered
            for marker in ("display controller", "vga compatible controller", "3d controller")
        ):
            continue
        parts = raw.split(maxsplit=1)
        if len(parts) != 2:
            continue
        bdf, description = parts
        id_matches = re.findall(r"\[([0-9a-fA-F]{4}):([0-9a-fA-F]{4})\]", description)
        vendor_id, device_id = id_matches[-1] if id_matches else (None, None)
        revision_match = re.search(r"\(rev\s+([0-9a-fA-F]+)\)\s*$", description)
        revision = revision_match.group(1) if revision_match else None
        product = re.sub(r"\s+\[[0-9a-fA-F]{4}:[0-9a-fA-F]{4}\]", "", description)
        product = re.sub(r"\s+\(rev\s+[0-9a-fA-F]+\)\s*$", "", product)
        if ": " in product:
            product = product.split(": ", 1)[1]
        devices.append({
            "bdf": bdf,
            "product_name": product.strip(),
            "vendor_id": vendor_id.lower() if vendor_id else None,
            "device_id": device_id.lower() if device_id else None,
            "revision": revision,
            "vendor_family": _vendor_from_text(" ".join((description, vendor_id or ""))),
        })
    return devices


def parse_rocm_smi(output: str) -> Dict[str, Any]:
    """Parse common AMD and Hygon DTK ``rocm-smi`` text variants."""
    names: Dict[int, str] = {}
    memory: Dict[int, int] = {}
    driver_version: Optional[str] = None
    for line in output.splitlines():
        name_match = re.search(
            r"(?:GPU|DCU)\[(\d+)\].*?(?:Card Series|Card Model|Product Name)\s*:\s*(.+?)\s*$",
            line,
            flags=re.IGNORECASE,
        )
        if name_match:
            names[int(name_match.group(1))] = name_match.group(2).strip()
        mem_match = re.search(
            r"(?:GPU|DCU)\[(\d+)\].*?vram Total Memory \(MiB\)\s*:\s*(\d+)",
            line,
            flags=re.IGNORECASE,
        )
        if mem_match:
            memory[int(mem_match.group(1))] = int(mem_match.group(2))
        driver_match = re.search(r"Driver Version\s*:\s*(\S+)", line, flags=re.IGNORECASE)
        if driver_match:
            driver_version = driver_match.group(1)
    indices = sorted(set(names) | set(memory))
    return {
        "devices": [
            {
                "index": index,
                "product_name": names.get(index),
                "vram_total_mib": memory.get(index),
            }
            for index in indices
        ],
        "driver_version": driver_version,
    }


def parse_rocminfo(output: str) -> Dict[str, Any]:
    """Extract GPU agents and compiler architecture names from ``rocminfo``."""
    agents: List[Dict[str, Any]] = []
    blocks = re.split(r"(?m)^\s*Agent\s+\d+\s*$", output)
    for block in blocks[1:]:
        device_type = _field(block, "Device Type")
        architecture_match = re.search(r"(?m)^\s*Name:\s*(gfx[0-9a-zA-Z_-]+)\s*$", block)
        architecture = architecture_match.group(1) if architecture_match else None
        if (device_type or "").upper() != "GPU" and not architecture:
            continue
        agents.append({
            "name": _first_non_arch_name(block),
            "marketing_name": _field(block, "Marketing Name"),
            "vendor_name": _field(block, "Vendor Name"),
            "architecture": architecture,
            "uuid": _field(block, "Uuid"),
        })
    if not agents:
        # Some vendor builds omit Agent headings. Preserve architecture
        # evidence rather than treating the runtime as empty.
        for architecture in sorted(set(re.findall(r"\b(gfx[0-9a-zA-Z_-]+)\b", output))):
            agents.append({
                "name": None,
                "marketing_name": None,
                "vendor_name": None,
                "architecture": architecture,
                "uuid": None,
            })
    return {
        "agents": agents,
        "architectures": sorted({a["architecture"] for a in agents if a["architecture"]}),
    }


def _field(block: str, name: str) -> Optional[str]:
    match = re.search(rf"(?m)^\s*{re.escape(name)}:\s*(.+?)\s*$", block)
    return match.group(1).strip() if match else None


def _first_non_arch_name(block: str) -> Optional[str]:
    for value in re.findall(r"(?m)^\s*Name:\s*(.+?)\s*$", block):
        value = value.strip()
        if not value.startswith("gfx"):
            return value
    return None


def parse_nvidia_smi(output: str) -> List[Dict[str, Any]]:
    devices: List[Dict[str, Any]] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            index = int(parts[0])
            memory = int(float(parts[2]))
        except ValueError:
            continue
        devices.append({
            "index": index,
            "product_name": parts[1],
            "vram_total_mib": memory,
            "pci_bus_id": parts[3],
            "driver_version": parts[4],
        })
    return devices


def _vendor_from_text(value: str) -> Optional[str]:
    lowered = value.lower()
    if any(token in lowered for token in ("hygon", "haiguang", "chengdu haiguang", "1d94")):
        return "hygon"
    if "nvidia" in lowered or "10de" in lowered:
        return "nvidia"
    if any(token in lowered for token in ("advanced micro devices", "amd/ati", " amd ", "1002")):
        return "amd"
    return None


def _device_access(path: Path) -> Dict[str, Any]:
    exists = path.exists()
    result: Dict[str, Any] = {
        "path": str(path),
        "exists": exists,
        "readable": bool(exists and os.access(path, os.R_OK)),
        "writable": bool(exists and os.access(path, os.W_OK)),
    }
    if not exists:
        return result
    try:
        info = path.stat()
        result.update({
            "mode": stat.filemode(info.st_mode),
            "uid": info.st_uid,
            "gid": info.st_gid,
        })
    except OSError as exc:
        result["stat_error"] = str(exc)
    return result


def _version_head(output: str, max_lines: int = 3) -> Optional[str]:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return " | ".join(lines[:max_lines]) if lines else None


def _library_inventory(output: str) -> List[str]:
    wanted = ("hip", "hsa", "rocblas", "hipblas", "rccl", "nccl")
    names = set()
    for line in output.splitlines():
        lowered = line.lower()
        if not any(token in lowered for token in wanted):
            continue
        match = re.match(r"\s*(\S+)\s+", line)
        if match:
            names.add(match.group(1))
    return sorted(names)


def normalize_assigned_devices(value: Any) -> str:
    """Return a safe, canonical comma-separated device allocation."""
    raw = str(value or "").strip().translate(
        str.maketrans({"\uff0c": ",", "\u3001": ","})
    )
    if not raw:
        return ""
    if not re.fullmatch(r"\d+(?:\s*,\s*\d+)*", raw):
        raise ValueError("assigned_devices must be a comma-separated list of non-negative integers")
    return ",".join(part.strip() for part in raw.split(","))


def configure_assigned_devices(
    req: Mapping[str, Any], environ: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Apply an explicit task device allocation before probing/launching.

    A blank value preserves scheduler-provided visibility variables.  The
    accepted grammar is deliberately narrow because this value becomes a
    process environment variable inherited by generated code.
    """
    env = environ if environ is not None else os.environ
    normalized = normalize_assigned_devices(req.get("assigned_devices"))
    if not normalized:
        return {name: env[name] for name in _VISIBLE_DEVICE_VARS if name in env}
    target = str(req.get("target_hardware") or "").lower()
    backend = str(req.get("accelerator_backend") or "").lower()
    if "nvidia" in target or "cuda" in backend:
        env["CUDA_VISIBLE_DEVICES"] = normalized
        return {"CUDA_VISIBLE_DEVICES": normalized}
    env["HIP_VISIBLE_DEVICES"] = normalized
    env["ROCR_VISIBLE_DEVICES"] = normalized
    return {
        "HIP_VISIBLE_DEVICES": normalized,
        "ROCR_VISIBLE_DEVICES": normalized,
    }


def discover_hardware_profile(req: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Collect a JSON-serializable, read-only hardware capability profile."""
    req = req or {}
    evidence: Dict[str, Any] = {}

    lspci_devices: List[Dict[str, Any]] = []
    lspci = _find_tool("lspci", extra_dirs=())
    if lspci:
        result = _run_read_only([lspci, "-D", "-nn"])
        evidence["lspci"] = _evidence_record([lspci, "-D", "-nn"], result)
        if result["returncode"] == 0:
            lspci_devices = parse_lspci(result["stdout"])

    rocm_smi_data: Dict[str, Any] = {"devices": [], "driver_version": None}
    rocm_smi = _find_tool("rocm-smi")
    if rocm_smi:
        outputs: List[str] = []
        for args in (
            ("--showproductname",),
            ("--showmeminfo", "vram"),
            ("--showdriverversion",),
            ("--showtopo",),
            ("--showuse",),
            ("--showmemuse",),
        ):
            result = _run_read_only([rocm_smi, *args])
            evidence[f"rocm_smi_{args[0].lstrip('-')}"] = _evidence_record(
                [rocm_smi, *args], result,
            )
            if result["returncode"] == 0:
                outputs.append(result["stdout"])
        rocm_smi_data = parse_rocm_smi("\n".join(outputs))

    rocminfo_data: Dict[str, Any] = {"agents": [], "architectures": []}
    rocminfo = _find_tool("rocminfo")
    if rocminfo:
        result = _run_read_only([rocminfo])
        evidence["rocminfo"] = _evidence_record([rocminfo], result)
        if result["returncode"] == 0:
            rocminfo_data = parse_rocminfo(result["stdout"])

    nvidia_devices: List[Dict[str, Any]] = []
    nvidia_smi = _find_tool("nvidia-smi", extra_dirs=())
    if nvidia_smi:
        query = "index,name,memory.total,pci.bus_id,driver_version"
        command = [nvidia_smi, f"--query-gpu={query}", "--format=csv,noheader,nounits"]
        result = _run_read_only(command)
        evidence["nvidia_smi"] = _evidence_record(command, result)
        if result["returncode"] == 0:
            nvidia_devices = parse_nvidia_smi(result["stdout"])

    toolchain: Dict[str, Any] = {}
    tool_specs = (
        ("hipconfig", "hipconfig", ("--full",)),
        ("hipcc", "hipcc", ("--version",)),
        ("clang", "clang++", ("--version",)),
        ("cmake", "cmake", ("--version",)),
    )
    for key, tool_name, args in tool_specs:
        tool = _find_tool(tool_name, extra_dirs=() if key == "cmake" else _DTK_BIN_DIRS)
        if not tool:
            toolchain[key] = {"available": False, "path": None, "version": None}
            continue
        result = _run_read_only([tool, *args])
        evidence[key] = _evidence_record([tool, *args], result)
        toolchain[key] = {
            "available": result["returncode"] == 0,
            "path": tool,
            "version": _version_head(result["stdout"] or result["stderr"]),
        }

    libraries: List[str] = []
    ldconfig = _find_tool("ldconfig", extra_dirs=(Path("/sbin"), Path("/usr/sbin")))
    if ldconfig:
        result = _run_read_only([ldconfig, "-p"])
        evidence["ldconfig"] = _evidence_record([ldconfig, "-p"], result, include_stdout=False)
        if result["returncode"] == 0:
            libraries = _library_inventory(result["stdout"])

    kfd = _device_access(Path("/dev/kfd"))
    render_nodes = [_device_access(path) for path in sorted(Path("/dev/dri").glob("renderD*"))]
    visible_env = {name: os.environ[name] for name in _VISIBLE_DEVICE_VARS if name in os.environ}

    preferred_vendor = (
        _expected_vendor(str(req.get("target_hardware") or ""), "")
        or _expected_backend_vendor(str(req.get("accelerator_backend") or ""))
    )
    vendor_family = _detect_vendor(
        lspci_devices,
        rocm_smi_data,
        rocminfo_data,
        nvidia_devices,
        preferred_vendor=preferred_vendor,
    )
    vendor_pci_count = len([
        device for device in lspci_devices
        if vendor_family in (None, "mixed") or device.get("vendor_family") == vendor_family
    ])
    identified_counts = [vendor_pci_count]
    if vendor_family in ("hygon", "amd"):
        identified_counts.append(len(rocm_smi_data["devices"]))
    if vendor_family == "nvidia":
        identified_counts.append(len(nvidia_devices))
    physical_device_count = max(identified_counts)
    if physical_device_count == 0:
        physical_device_count = max(len(rocminfo_data["agents"]), len(render_nodes))
    visibility_limit = _visible_device_count(visible_env, vendor_family)
    detected_count = (
        min(physical_device_count, visibility_limit)
        if visibility_limit is not None
        else physical_device_count
    )
    dtk_root = Path("/opt/dtk")
    sdk = {
        "dtk_root": str(dtk_root),
        "exists": dtk_root.exists(),
        "is_symlink": dtk_root.is_symlink(),
        "symlink_target": _readlink_text(dtk_root),
        "resolved_path": str(dtk_root.resolve()) if dtk_root.exists() else None,
    }
    profile: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "host": {
            "hostname": socket.gethostname(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "requested": {
            "target_hardware": req.get("target_hardware"),
            "accelerator_backend": req.get("accelerator_backend"),
            "assigned_devices": req.get("assigned_devices"),
        },
        "visibility": visible_env,
        "detected": {
            "vendor_family": vendor_family,
            "device_count": detected_count,
            "physical_device_count": physical_device_count,
            "pci_devices": lspci_devices,
            "smi_devices": rocm_smi_data["devices"],
            "runtime_agents": rocminfo_data["agents"],
            "nvidia_devices": nvidia_devices,
            "hip_architectures": rocminfo_data["architectures"],
            "driver_version": rocm_smi_data["driver_version"] or _first_driver(nvidia_devices),
        },
        "toolchain": toolchain,
        "sdk": sdk,
        "libraries": libraries,
        "permissions": {
            "kfd": kfd,
            "render_nodes": render_nodes,
        },
        "evidence": evidence,
    }
    profile["validation"] = validate_hardware_selection(profile)
    return profile


def _evidence_record(
    command: Sequence[str], result: Mapping[str, Any], *, include_stdout: bool = True,
) -> Dict[str, Any]:
    record = {
        "command": [str(part) for part in command],
        "available": result.get("available"),
        "returncode": result.get("returncode"),
        "duration_ms": result.get("duration_ms"),
        "stderr": str(result.get("stderr") or "")[:2048],
    }
    if include_stdout:
        record["stdout"] = str(result.get("stdout") or "")[:16384]
    return record


def _detect_vendor(
    pci: Sequence[Mapping[str, Any]],
    rocm: Mapping[str, Any],
    rocminfo: Mapping[str, Any],
    nvidia: Sequence[Mapping[str, Any]],
    *,
    preferred_vendor: Optional[str] = None,
) -> Optional[str]:
    candidates = {
        str(item.get("vendor_family"))
        for item in pci
        if item.get("vendor_family")
    }
    text = json.dumps({"pci": pci, "rocm": rocm, "rocminfo": rocminfo}).lower()
    runtime_vendor = _vendor_from_text(f" {text} ")
    if runtime_vendor:
        candidates.add(runtime_vendor)
    if nvidia:
        candidates.add("nvidia")
    if preferred_vendor in candidates:
        return preferred_vendor
    if len(candidates) == 1:
        return next(iter(candidates))
    if len(candidates) > 1:
        return "mixed"
    return None


def _visible_device_count(
    visibility: Mapping[str, str], vendor_family: Optional[str],
) -> Optional[int]:
    if vendor_family in ("hygon", "amd"):
        order = ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
    elif vendor_family == "nvidia":
        order = ("CUDA_VISIBLE_DEVICES",)
    else:
        order = _VISIBLE_DEVICE_VARS
    for name in order:
        if name not in visibility:
            continue
        raw = visibility[name].strip()
        if not raw or raw == "-1":
            return 0
        return len([part for part in raw.split(",") if part.strip()])
    return None


def _readlink_text(path: Path) -> Optional[str]:
    try:
        return os.readlink(path)
    except OSError:
        return None


def _first_driver(devices: Sequence[Mapping[str, Any]]) -> Optional[str]:
    for device in devices:
        if device.get("driver_version"):
            return str(device["driver_version"])
    return None


def validate_hardware_selection(profile: Mapping[str, Any]) -> Dict[str, Any]:
    """Compare user intent with discovered evidence without hiding aliases."""
    requested = profile.get("requested") or {}
    detected = profile.get("detected") or {}
    target = str(requested.get("target_hardware") or "").strip()
    backend = str(requested.get("accelerator_backend") or "").strip()
    actual_vendor = detected.get("vendor_family")
    target_vendor = _expected_vendor(target, "")
    backend_vendor = _expected_backend_vendor(backend)
    expected_vendor = target_vendor or backend_vendor
    warnings: List[str] = []
    blockers: List[str] = []

    device_count = int(detected.get("device_count") or 0)
    if device_count == 0:
        if profile.get("visibility"):
            blockers.append(
                "The inherited device-visibility allocation exposes no accessible accelerator."
            )
        else:
            warnings.append("No accelerator was detected in the orchestrator environment.")
    if target_vendor and backend_vendor and target_vendor != backend_vendor:
        blockers.append(
            f"Target {target!r} is incompatible with backend {backend!r}."
        )
    if expected_vendor and actual_vendor and expected_vendor != actual_vendor:
        blockers.append(
            f"Requested {expected_vendor} hardware/backend but detected {actual_vendor}."
        )

    pci_names = [
        str(device.get("product_name") or "")
        for device in detected.get("pci_devices") or []
    ]
    secondary_names = [
        str(device.get("product_name") or "")
        for key in ("smi_devices", "nvidia_devices")
        for device in detected.get(key) or []
    ]
    model_token = _target_model_token(target)
    if model_token and pci_names:
        if not any(model_token in _normalize_model(name) for name in pci_names):
            blockers.append(
                f"Requested model {target!r} does not match PCI devices: {pci_names}."
            )
    elif model_token and secondary_names and not any(
        model_token in _normalize_model(name) for name in secondary_names
    ):
        warnings.append(
            f"Requested model {target!r} was not confirmed by runtime/SMI names: {secondary_names}."
        )

    if pci_names and secondary_names:
        pci_models = {_normalize_model(name) for name in pci_names}
        secondary_models = {_normalize_model(name) for name in secondary_names}
        if pci_models.isdisjoint(secondary_models):
            warnings.append(
                "PCI and SMI/runtime product names differ; PCI identifies the board, "
                "while rocminfo architecture remains authoritative for HIP compilation."
            )

    permissions = profile.get("permissions") or {}
    if actual_vendor in ("hygon", "amd"):
        kfd = permissions.get("kfd") or {}
        if not kfd.get("exists"):
            blockers.append("/dev/kfd is missing from the task environment.")
        elif not (kfd.get("readable") and kfd.get("writable")):
            blockers.append("/dev/kfd is not readable and writable by the current user.")
        render_nodes = permissions.get("render_nodes") or []
        if not render_nodes:
            blockers.append("No GPU render nodes are visible under /dev/dri.")
        inaccessible = [
            node.get("path") for node in permissions.get("render_nodes") or []
            if not (node.get("readable") and node.get("writable"))
        ]
        if inaccessible:
            blockers.append(f"GPU render nodes are inaccessible: {inaccessible}.")

    toolchain = profile.get("toolchain")
    if isinstance(toolchain, Mapping):
        if not (toolchain.get("cmake") or {}).get("available"):
            blockers.append("CMake is not available in the orchestrator environment.")
        requires_hip = any(token in backend.lower() for token in ("hip", "dtk"))
        if requires_hip and not (toolchain.get("hipcc") or {}).get("available"):
            blockers.append("hipcc is not available for the selected HIP/DTK backend.")
        if requires_hip and not (detected.get("hip_architectures") or []):
            blockers.append("rocminfo did not report a HIP compiler architecture.")

    if blockers:
        status = "mismatch"
    elif device_count == 0:
        status = "unknown"
    elif warnings:
        status = "compatible_with_warnings"
    else:
        status = "matched"
    return {
        "status": status,
        "runnable": not blockers and device_count > 0,
        "expected_vendor": expected_vendor,
        "detected_vendor": actual_vendor,
        "warnings": warnings,
        "blockers": blockers,
    }


def _expected_vendor(target: str, backend: str) -> Optional[str]:
    value = f"{target} {backend}".lower()
    if any(token in value for token in ("hygon", "k100ai", "bw1000", "z100", "z200", "dtk")):
        return "hygon"
    if any(token in value for token in ("nvidia", "h100", "a100", "cuda")):
        return "nvidia"
    if any(token in value for token in ("amd", "mi300")):
        return "amd"
    return None


def _expected_backend_vendor(backend: str) -> Optional[str]:
    value = backend.lower()
    if "dtk" in value:
        return "hygon"
    if "cuda" in value:
        return "nvidia"
    # Generic HIP/ROCm and LibTorch APIs can target more than one vendor.
    return None


def _target_model_token(target: str) -> Optional[str]:
    normalized = _normalize_model(target)
    for token in ("k100ai", "bw1000", "z200sm", "z200", "z100sm", "h100", "a100", "mi300"):
        if token in normalized:
            return token
    return None


def _normalize_model(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def prompt_hardware_summary(profile: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the stable, compact subset injected into every agent prompt."""
    detected = profile.get("detected") or {}
    toolchain = profile.get("toolchain") or {}
    permissions = profile.get("permissions") or {}
    return {
        "profile_schema_version": profile.get("schema_version"),
        "generated_at": profile.get("generated_at"),
        "host": profile.get("host"),
        "requested": profile.get("requested"),
        "visibility": profile.get("visibility"),
        "vendor_family": detected.get("vendor_family"),
        "device_count": detected.get("device_count"),
        "physical_device_count": detected.get("physical_device_count"),
        "pci_devices": detected.get("pci_devices"),
        "smi_devices": detected.get("smi_devices"),
        "runtime_agents": detected.get("runtime_agents"),
        "nvidia_devices": detected.get("nvidia_devices"),
        "hip_architectures": detected.get("hip_architectures"),
        "driver_version": detected.get("driver_version"),
        "toolchain": toolchain,
        "sdk": profile.get("sdk"),
        "libraries": profile.get("libraries"),
        "kfd_access": {
            key: (permissions.get("kfd") or {}).get(key)
            for key in ("path", "exists", "readable", "writable", "mode")
        },
        "render_nodes": permissions.get("render_nodes"),
        "validation": profile.get("validation"),
    }


def write_hardware_profile(path: Path, profile: Mapping[str, Any]) -> None:
    """Atomically persist a profile so the WebUI never observes partial JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


__all__ = [
    "SCHEMA_VERSION",
    "configure_assigned_devices",
    "discover_hardware_profile",
    "normalize_assigned_devices",
    "parse_lspci",
    "parse_nvidia_smi",
    "parse_rocm_smi",
    "parse_rocminfo",
    "prompt_hardware_summary",
    "validate_hardware_selection",
    "write_hardware_profile",
]
