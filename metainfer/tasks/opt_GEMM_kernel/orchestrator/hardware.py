"""Resolve the WebUI selection to one task-local, system-owned profile."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import yaml

from metainfer.orchestrator.requirements import req_field


_PROFILES_FILE = Path(__file__).with_name("hardware_profiles.yaml")


class HardwareProfileError(ValueError):
    pass


@lru_cache(maxsize=1)
def load_hardware_profiles() -> Dict[str, Dict[str, Any]]:
    raw = yaml.safe_load(_PROFILES_FILE.read_text(encoding="utf-8")) or {}
    if raw.get("schema_version") != 1 or not isinstance(raw.get("profiles"), dict):
        raise HardwareProfileError(f"invalid hardware profile file: {_PROFILES_FILE}")
    return {
        str(label): dict(profile)
        for label, profile in raw["profiles"].items()
        if isinstance(profile, Mapping)
    }


def require_hardware_profile(req: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    selected = str(req_field(req, "target_hardware", "") or "").strip()
    if not selected:
        raise HardwareProfileError("target_hardware is required")
    profile = load_hardware_profiles().get(selected)
    if profile is None:
        raise HardwareProfileError(
            f"no opt_GEMM_kernel execution profile is registered for {selected!r}"
        )
    requested_arch = str(req_field(req, "gpu_arch", "") or "").lower().strip()
    profile_arch = str(profile.get("gpu_arch") or "").lower()
    if requested_arch and requested_arch != profile_arch:
        raise HardwareProfileError(
            f"{selected} requires gpu_arch={profile_arch}, got {requested_arch}"
        )
    return selected, profile

