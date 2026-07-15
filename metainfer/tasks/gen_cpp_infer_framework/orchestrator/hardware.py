"""Resolve hardware profiles into system-owned iteration bindings.

The WebUI selection is the sole input.  This module reads the matching YAML
profile, snapshots it into the iteration, and generates the ``build.sh`` that
the correctness oracle executes.  Agents write CMake targets and C++ source;
they neither choose a compiler/backend nor author profiling commands.
"""

from __future__ import annotations

import json
import shlex
import stat
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


_PROFILES_FILE = Path(__file__).with_name("hardware_profiles.yaml")
_BINDING_DIR = ".metainfer"
_PROFILE_SNAPSHOT = "hardware-profile.json"


class HardwareProfileError(ValueError):
    """Raised when a task selects no usable hardware execution profile."""


@lru_cache(maxsize=1)
def load_hardware_profiles() -> Dict[str, Dict[str, Any]]:
    """Return form-label to hardware-profile mappings from package data."""
    raw = yaml.safe_load(_PROFILES_FILE.read_text(encoding="utf-8")) or {}
    profiles = raw.get("profiles")
    if not isinstance(profiles, dict):
        raise ValueError(f"invalid hardware profiles file: {_PROFILES_FILE}")
    return {
        str(label): profile
        for label, profile in profiles.items()
        if isinstance(profile, dict)
    }


def selected_hardware(req: Dict[str, Any]) -> Optional[str]:
    """Read the current hardware selection, including legacy nested inputs."""
    value = req.get("target_hardware")
    if value is None:
        value = (req.get("answers") or {}).get("target_hardware")
    return str(value) if value else None


def resolve_hardware_profile(
    req: Dict[str, Any],
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Return ``(selected_label, profile)``; profile is None if not mapped."""
    selected = selected_hardware(req)
    if selected is None:
        return None, None
    return selected, load_hardware_profiles().get(selected)


def require_hardware_profile(req: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Return the selected profile or fail before any agent can guess."""
    selected, profile = resolve_hardware_profile(req)
    if selected is None:
        raise HardwareProfileError("target_hardware is required for a C++ inference task")
    if profile is None:
        raise HardwareProfileError(
            f"no verified hardware execution profile is registered for {selected!r}"
        )
    return selected, profile


def materialize_hardware_binding(req: Dict[str, Any], iter_dir: Path) -> Path:
    """Write the immutable-on-execution profile snapshot and ``build.sh``.

    ``build.sh`` is deliberately regenerated on every call.  A sub-agent may
    inspect it, but an attempted edit is discarded before correctness or perf
    runs, so the selected hardware remains the only source of build flags.
    """
    selected, profile = require_hardware_profile(req)
    binding_dir = iter_dir / _BINDING_DIR
    binding_dir.mkdir(parents=True, exist_ok=True)
    snapshot = binding_dir / _PROFILE_SNAPSHOT
    snapshot.write_text(
        json.dumps(
            {"selected_hardware": selected, "profile": profile},
            ensure_ascii=False, indent=2, sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    build_script = iter_dir / "build.sh"
    build_script.write_text(_render_build_script(selected, profile), encoding="utf-8")
    build_script.chmod(build_script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
    return snapshot


def execution_environment(req: Dict[str, Any], iter_dir: Path) -> Dict[str, str]:
    """Return the environment shared by B self-tests and C/E oracles."""
    selected, profile = require_hardware_profile(req)
    compiler = str(profile["build"]["compiler"]["command"])
    return {
        "METAINFER_TARGET_HARDWARE": selected,
        "METAINFER_HARDWARE_PROFILE": str(profile["id"]),
        "METAINFER_HARDWARE_PROFILE_FILE": str(
            iter_dir / _BINDING_DIR / _PROFILE_SNAPSHOT
        ),
        "METAINFER_HIPCC": compiler,
    }


def profiler_launch_command(req: Dict[str, Any]) -> List[str]:
    """Return the externally-owned profiler prefix for the selected profile."""
    _, profile = require_hardware_profile(req)
    profiling = profile.get("profiling") or {}
    profiler = profiling.get("profiler")
    if not profiler:
        return []
    return [str(profiler), *(str(v) for v in profiling.get("launch_args") or [])]


def profiler_artifact_globs(req: Dict[str, Any]) -> List[str]:
    """Return the external profiler artifacts expected in an iteration dir."""
    _, profile = require_hardware_profile(req)
    profiling = profile.get("profiling") or {}
    return [str(v) for v in profiling.get("artifact_globs") or []]


def _render_build_script(selected: str, profile: Dict[str, Any]) -> str:
    """Render the only build command path used by a selected platform."""
    build = profile["build"]
    cache = build["cmake_cache"]
    compiler = str(build["compiler"]["command"])
    cache_args = [
        "-DCMAKE_BUILD_TYPE=Release",
        *(f"-D{k}={v}" for k, v in cache.items()),
    ]
    quoted_args = " ".join(shlex.quote(arg) for arg in cache_args)
    quoted_selected = shlex.quote(selected)
    quoted_compiler = shlex.quote(compiler)
    return f'''#!/usr/bin/env bash
# SYSTEM-OWNED FILE. Generated from hardware_profiles.yaml for {quoted_selected}.
# Do not edit: the orchestrator regenerates this file before C/E execution.
set -euo pipefail

ROOT="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
HIPCC_BIN="${{METAINFER_HIPCC:-{quoted_compiler}}}"
if ! command -v "$HIPCC_BIN" >/dev/null 2>&1; then
  echo "required HIP compiler not found: $HIPCC_BIN" >&2
  exit 127
fi
if ! command -v cmake >/dev/null 2>&1; then
  echo "required build tool not found: cmake" >&2
  exit 127
fi
if [[ ! -f "$ROOT/CMakeLists.txt" ]]; then
  echo "missing CMakeLists.txt in $ROOT" >&2
  exit 2
fi

export HIPCXX="$(command -v "$HIPCC_BIN")"
cmake -S "$ROOT" -B "$ROOT/build" {quoted_args}
cmake --build "$ROOT/build" --parallel "${{METAINFER_BUILD_JOBS:-$(nproc)}}"
'''


def render_hardware_profile(req: Dict[str, Any]) -> str:
    """Render the selected executable profile for an agent prompt."""
    selected, profile = resolve_hardware_profile(req)
    if selected is None:
        return ""
    if profile is None:
        return (
            "\n# Hardware execution profile\n"
            f"No verified execution profile is registered for {selected!r}. "
            "Do not guess compiler, operator, or profiler commands.\n"
        )
    rendered = json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True)
    return (
        "\n# Hardware execution profile (mandatory)\n"
        f"The selected platform is {selected!r}. This profile is SYSTEM-OWNED: "
        "do not edit build.sh, invoke a compiler directly, or launch a profiler. "
        "Write only CMake targets and C++ code that conform to its runtime and "
        "operator constraints.\n"
        "```json\n"
        f"{rendered}\n"
        "```\n"
    )
