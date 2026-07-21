"""Frozen, system-owned hardware profiling for Hygon K100 / gfx928."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


class ProfilerError(RuntimeError):
    pass


def _version(executable: str) -> str:
    for option in ("--version", "-v"):
        try:
            proc = subprocess.run(
                [executable, option], text=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, timeout=15, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if proc.returncode == 0 and (proc.stdout or "").strip():
            return (proc.stdout or "").strip()[:2000]
    raise ProfilerError(f"cannot query profiler version: {executable}")


def _find_tool(candidates: Iterable[str]) -> Optional[Path]:
    for value in candidates:
        if "/" in value:
            path = Path(value).expanduser()
            if path.is_file() and os.access(path, os.X_OK):
                return path.resolve()
        else:
            found = shutil.which(value)
            if found:
                return Path(found).resolve()
    return None


def _available_counters(executable: Path, kind: str) -> set[str]:
    commands: List[List[str]] = []
    if kind == "rocprofv3":
        companion = executable.with_name("rocprofv3-avail")
        if companion.is_file():
            commands.append([str(companion), "list", "--pmc"])
        commands.append([str(executable), "--list-avail"])
    else:
        commands.append([str(executable), "--list-basic"])
    for command in commands:
        try:
            proc = subprocess.run(
                command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                timeout=60, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if proc.returncode == 0:
            return set(re.findall(r"\b[A-Za-z][A-Za-z0-9_]{2,}\b", proc.stdout or ""))
    return set()


@dataclass(frozen=True)
class FrozenProfilerProfile:
    id: str
    label: str
    backend: str
    gpu_arch: str
    executable: str
    executable_version: str
    tool_kind: str
    representative_cases: List[str]
    counter_groups: List[List[str]]
    kernel_name_contains: str
    required: bool
    fingerprint: str
    schema_version: int = 1

    @classmethod
    def resolve(
        cls,
        requirements: Mapping[str, Any],
        hardware_profile: Mapping[str, Any],
        state_dir: Path,
    ) -> Optional["FrozenProfilerProfile"]:
        raw = dict(hardware_profile.get("profiling") or {})
        profile_arch = str(hardware_profile.get("gpu_arch") or "").lower()
        arch = str(_req(requirements, "gpu_arch", "")).lower().strip()
        target = str(_req(requirements, "target_hardware", "")).lower()
        if arch != profile_arch:
            return None
        if not any(token in target for token in ("k100", "海光", "hygon")):
            return None
        manifest_path = state_dir / "profiler_profile.json"
        if manifest_path.is_file():
            profile = cls(**json.loads(manifest_path.read_text(encoding="utf-8")))
            profile.verify()
            return profile

        executable = _find_tool(raw.get("tool_candidates") or [])
        if executable is None:
            raise ProfilerError(
                "Hygon K100/gfx928 requires rocprofv3 or rocprof on the target node"
            )
        kind = "rocprofv3" if "rocprofv3" in executable.name else "rocprof"
        available = _available_counters(executable, kind)
        configured = [list(map(str, group)) for group in raw.get("counter_groups") or []]
        groups = [
            [counter for counter in group if not available or counter in available]
            for group in configured
        ]
        groups = [group for group in groups if group]
        if not groups:
            raise ProfilerError("K100 profiler exposed none of the frozen counter whitelist")
        data = {
            "id": str(hardware_profile["id"]),
            "label": str(_req(requirements, "target_hardware", "Hygon K100")),
            "backend": str(hardware_profile["backend"]),
            "gpu_arch": str(hardware_profile["gpu_arch"]),
            "executable": str(executable),
            "executable_version": _version(str(executable)),
            "tool_kind": kind,
            "representative_cases": list(map(str, raw.get("representative_cases") or [])),
            "counter_groups": groups,
            "kernel_name_contains": str(raw.get("kernel_name_contains") or "w8a8_scaled_"),
            "required": bool(raw.get("required", True)),
            "fingerprint": "",
            "schema_version": 1,
        }
        data["fingerprint"] = _fingerprint(data)
        state_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return cls(**data)

    def verify(self) -> None:
        data = asdict(self)
        expected = _fingerprint({**data, "fingerprint": ""})
        if expected != self.fingerprint:
            raise ProfilerError("frozen profiler profile fingerprint changed")
        path = Path(self.executable)
        if not path.is_file() or _version(self.executable) != self.executable_version:
            raise ProfilerError("frozen profiler executable or version changed")


@dataclass
class ProfileResult:
    passed: bool
    report: Dict[str, Any]
    failure: Optional[str] = None


class ProfilerRunner:
    def __init__(self, profile: FrozenProfilerProfile, *, private_env: Mapping[str, str]) -> None:
        self.profile = profile
        self.private_env = dict(private_env)

    def run(
        self,
        artifact_dir: Path,
        output_dir: Path,
        *,
        role: str,
    ) -> ProfileResult:
        try:
            self.profile.verify()
        except Exception as exc:  # noqa: BLE001
            return ProfileResult(False, {}, str(exc))
        harness = artifact_dir / "metainfer_gemm_harness"
        if not harness.is_file():
            return ProfileResult(False, {}, f"native harness is missing: {harness}")
        root = output_dir / f"{role}-hardware-profile"
        root.mkdir(parents=True, exist_ok=True)
        cases: List[Dict[str, Any]] = []
        commands: List[List[str]] = []
        for case_id in self.profile.representative_cases:
            case_root = root / case_id
            case_root.mkdir(parents=True, exist_ok=True)
            for group_index, counters in enumerate(self.profile.counter_groups, 1):
                pass_dir = case_root / f"pass_{group_index}"
                pass_dir.mkdir(parents=True, exist_ok=True)
                command = self._command(harness, case_id, counters, pass_dir)
                commands.append(command)
                env = dict(os.environ)
                env.update(self.private_env)
                env.update({
                    "METAINFER_EVALUATION_PHASE": "profile",
                    "METAINFER_EVALUATION_ROLE": role,
                    "METAINFER_BUILD_ARTIFACT_DIR": str(artifact_dir.resolve()),
                    "METAINFER_REPORT_PATH": str((pass_dir / "harness-profile.json").resolve()),
                    "PYTHONDONTWRITEBYTECODE": "1",
                })
                proc = subprocess.run(
                    command, cwd=str(artifact_dir), env=env, text=True,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    timeout=1800, check=False,
                )
                (pass_dir / "profiler.stdout.log").write_text(proc.stdout or "", encoding="utf-8")
                (pass_dir / "profiler.stderr.log").write_text(proc.stderr or "", encoding="utf-8")
                if proc.returncode != 0:
                    report = self._report(cases, commands)
                    _write_json(output_dir / f"{role}-hardware-profile.json", report)
                    return ProfileResult(
                        False, report,
                        f"{self.profile.tool_kind} failed for {case_id} pass {group_index}",
                    )
            try:
                cases.append(
                    _parse_case(case_id, case_root, self.profile.kernel_name_contains)
                )
            except (OSError, ValueError, ProfilerError) as exc:
                report = self._report(cases, commands)
                _write_json(output_dir / f"{role}-hardware-profile.json", report)
                return ProfileResult(False, report, str(exc))
        report = self._report(cases, commands)
        _write_json(output_dir / f"{role}-hardware-profile.json", report)
        return ProfileResult(True, report)

    def _command(
        self, harness: Path, case_id: str, counters: List[str], output: Path,
    ) -> List[str]:
        if self.profile.tool_kind == "rocprofv3":
            return [
                self.profile.executable,
                "--pmc", ",".join(counters),
                "--output-format", "csv", "json",
                "--output-directory", str(output.resolve()),
                "--kernel-include-regex", self.profile.kernel_name_contains,
                "--", str(harness.resolve()), "profile", case_id,
            ]
        input_path = output / "counters.txt"
        input_path.write_text("pmc: " + " ".join(counters) + "\n", encoding="utf-8")
        return [
            self.profile.executable, "-i", str(input_path.resolve()),
            "-o", str((output / "counter_collection.csv").resolve()), "--timestamp", "on",
            str(harness.resolve()), "profile", case_id,
        ]

    def _report(self, cases: List[Dict[str, Any]], commands: List[List[str]]) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "passed": len(cases) == len(self.profile.representative_cases),
            "profile_id": self.profile.id,
            "label": self.profile.label,
            "gpu_arch": self.profile.gpu_arch,
            "tool": self.profile.tool_kind,
            "tool_version": self.profile.executable_version,
            "profile_fingerprint": self.profile.fingerprint,
            "counter_groups": self.profile.counter_groups,
            "commands": commands,
            "cases": cases,
        }


def _parse_case(case_id: str, root: Path, kernel_token: str) -> Dict[str, Any]:
    rows: List[Dict[str, str]] = []
    for path in root.rglob("*.csv"):
        try:
            with path.open(newline="", encoding="utf-8", errors="replace") as stream:
                rows.extend(dict(row) for row in csv.DictReader(stream))
        except OSError:
            continue
    selected = [row for row in rows if kernel_token in _text(row, "Kernel_Name", "KernelName", "Name")]
    if not selected:
        raise ProfilerError(f"no target-kernel rows found in profiler output for {case_id}")
    counters: Dict[str, float] = {}
    for row in selected:
        name = _text(row, "Counter_Name", "CounterName")
        value = _number(_text(row, "Counter_Value", "CounterValue"))
        if name and value is not None:
            counters[_normalize_counter(name)] = value
        for key, raw in row.items():
            normalized = _normalize_counter(key or "")
            if normalized in _KNOWN_COUNTERS:
                parsed = _number(raw)
                if parsed is not None:
                    counters[normalized] = parsed
    last = selected[-1]
    begin = _value(last, "Start_Timestamp", "BeginNs", "Begin_Ns")
    end = _value(last, "End_Timestamp", "EndNs", "End_Ns")
    duration = end - begin if begin is not None and end is not None and end >= begin else None
    tcc_hit, tcc_miss = counters.get("TCC_HIT"), counters.get("TCC_MISS")
    l2_hit = counters.get("L2_CACHE_HIT")
    if l2_hit is None:
        l2_hit = (
            100.0 * tcc_hit / (tcc_hit + tcc_miss)
            if tcc_hit is not None and tcc_miss is not None
            and tcc_hit + tcc_miss > 0 else None
        )
    traffic_kib = sum(counters.get(name, 0.0) for name in ("FETCH_SIZE", "WRITE_SIZE"))
    measured_bw = (
        traffic_kib * 1024.0 / duration
        if traffic_kib > 0 and duration and duration > 0 else None
    )
    grbm_count, grbm_active = counters.get("GRBM_COUNT"), counters.get("GRBM_GUI_ACTIVE")
    compute_busy = counters.get("GPU_BUSY")
    if compute_busy is None:
        compute_busy = (
            100.0 * grbm_active / grbm_count
            if grbm_count and grbm_active is not None else None
        )
    wave_cycles, waves = counters.get("SQ_WAVE_CYCLES"), counters.get("SQ_WAVES")
    return {
        "id": case_id,
        "kernel_name": _text(last, "Kernel_Name", "KernelName", "Name"),
        "duration_ns": duration,
        "grid_size": _integer(last, "Grid_Size", "GridSize", "grd"),
        "workgroup_size": _integer(last, "Workgroup_Size", "WorkgroupSize", "wgr"),
        "vgpr_count": _integer(last, "VGPR_Count", "Arch_VGPR", "VGPRCount", "vgpr", "arch_vgpr"),
        "agpr_count": _integer(last, "Accum_VGPR_Count", "Accum_VGPR", "AGPR_Count", "accum_vgpr"),
        "sgpr_count": _integer(last, "SGPR_Count", "SGPR", "sgpr"),
        "lds_bytes": _integer(last, "LDS_Block_Size", "LDS_Per_Workgroup", "LDSBytes", "lds"),
        "scratch_bytes": _integer(last, "Scratch_Size", "Scratch_Per_Workitem", "ScratchBytes", "scr"),
        "waves": waves,
        "wave_cycles_per_wave": wave_cycles / waves if wave_cycles and waves else None,
        "l2_hit_pct": l2_hit,
        "measured_bandwidth_gbps": measured_bw,
        "compute_busy_pct": compute_busy,
        "matrix_instructions": _first_counter(counters, "SQ_INSTS_MFMA", "SQ_INSTS_MMAC"),
        "valu_instructions": counters.get("SQ_INSTS_VALU"),
        "counters": counters,
    }


_KNOWN_COUNTERS = {
    "SQ_WAVES", "SQ_WAVE_CYCLES", "SQ_INSTS_VALU", "SQ_INSTS_SALU",
    "SQ_INSTS_MFMA", "SQ_INSTS_MMAC", "TCC_HIT", "TCC_MISS",
    "FETCH_SIZE", "WRITE_SIZE", "GRBM_COUNT", "GRBM_GUI_ACTIVE",
    "L2_CACHE_HIT", "GPU_BUSY",
}


def _normalize_counter(value: str) -> str:
    token = re.sub(r"\[[0-9]+\]$", "", value.strip())
    aliases = {
        "FetchSize": "FETCH_SIZE", "WriteSize": "WRITE_SIZE",
        "L2CacheHit": "L2_CACHE_HIT", "GPUBusy": "GPU_BUSY",
    }
    return aliases.get(token, token.upper())


def _req(requirements: Mapping[str, Any], key: str, default: Any = None) -> Any:
    fields = requirements.get("fields")
    if isinstance(fields, dict) and key in fields:
        value = fields[key]
        return value.get("value", default) if isinstance(value, dict) else value
    return requirements.get(key, default)


def _fingerprint(data: Mapping[str, Any]) -> str:
    payload = dict(data)
    payload["fingerprint"] = ""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_json(path: Path, value: Dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _text(row: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _number(value: Any) -> Optional[float]:
    try:
        number = float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _value(row: Mapping[str, Any], *keys: str) -> Optional[float]:
    return _number(_text(row, *keys))


def _integer(row: Mapping[str, Any], *keys: str) -> Optional[int]:
    value = _value(row, *keys)
    return int(value) if value is not None else None


def _first_counter(counters: Mapping[str, float], *keys: str) -> Optional[float]:
    for key in keys:
        if key in counters:
            return counters[key]
    return None
