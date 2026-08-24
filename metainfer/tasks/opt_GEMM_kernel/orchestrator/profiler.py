"""Frozen, system-owned hardware profiling for Hygon K100 / gfx928."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import resource
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


class ProfilerError(RuntimeError):
    pass


def _disable_core_dump() -> None:
    """Prevent a profiler crash from writing a multi-gigabyte core file."""
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _python_executable() -> str:
    """Return the real interpreter even when a launcher rewrites argv[0]."""
    candidate = Path(sys.executable) if sys.executable else None
    if candidate is not None and candidate.name.lower().startswith("python"):
        return str(candidate.resolve())
    try:
        executable = Path("/proc/self/exe").resolve(strict=True)
    except OSError as exc:
        raise ProfilerError("cannot resolve the Python executable") from exc
    if not executable.name.lower().startswith("python"):
        raise ProfilerError(
            f"resolved process executable is not Python: {executable}"
        )
    return str(executable)


def _version(executable: str) -> str:
    # rocprofv3 and newer rocprof releases expose a conventional version
    # option.  The legacy RPL rocprof shipped by DTK only exposes ``-h``;
    # its first help line contains a per-invocation timestamp, so remove that
    # line before using the output as the frozen executable identity.
    for option in ("--version", "-v", "-h"):
        try:
            proc = subprocess.run(
                [executable, option], text=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, timeout=15, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        output = (proc.stdout or "").strip()
        if option == "-h":
            output = "\n".join(
                line for line in output.splitlines()
                if not line.lstrip().startswith("RPL:")
            ).strip()
        legacy_help = (
            option == "-h"
            and "ROCm Profiling Library" in output
            and "Usage:" in output
        )
        if output and (proc.returncode == 0 or legacy_help):
            return output[:2000]
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
    if kind == "hipprof":
        # hipprof owns its full PMC set; unlike rocprof, it does not accept a
        # caller-provided list of individual counters.
        return set()
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
            if raw.get("required", True):
                raise ProfilerError(
                    "Hygon K100/gfx928 requires hipprof on the target node"
                )
            return None
        if "hipprof" not in executable.name:
            raise ProfilerError(
                f"Hygon K100/gfx928 timing requires hipprof, got {executable}"
            )
        kind = "hipprof"
        available = _available_counters(executable, kind)
        configured = [list(map(str, group)) for group in raw.get("counter_groups") or []]
        groups = (
            [["HIPPROF_PMC_FULL"]]
            if kind == "hipprof"
            else [
                [counter for counter in group if not available or counter in available]
                for group in configured
            ]
        )
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
            # Preserve an explicit empty filter: an empty token means that
            # _parse_case accepts every kernel row for the representative case.
            "kernel_name_contains": str(raw.get("kernel_name_contains", "")),
            "required": bool(raw.get("required", True)),
            "fingerprint": "",
            "schema_version": 1,
        }
        data["fingerprint"] = _fingerprint(data)
        state_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return cls(**data)

    def verify(self) -> None:
        if self.tool_kind != "hipprof":
            raise ProfilerError(
                f"Hygon K100/gfx928 timing requires hipprof, got {self.tool_kind}"
            )
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
    def __init__(
        self,
        profile: FrozenProfilerProfile,
        *,
        private_env: Mapping[str, str],
        harness_argv: Optional[List[str]] = None,
        benchmark_protocol: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.profile = profile
        self.private_env = dict(private_env)
        self.harness_argv = harness_argv
        self.benchmark_protocol = dict(benchmark_protocol or {})

    def run(
        self,
        artifact_dir: Path,
        output_dir: Path,
        *,
        role: str,
        collection_mode: str = "trace",
        case_ids: Optional[List[str]] = None,
        run_label: Optional[str] = None,
        implementation: Optional[str] = None,
    ) -> ProfileResult:
        try:
            self.profile.verify()
        except Exception as exc:  # noqa: BLE001
            return ProfileResult(False, {}, str(exc))

        if self.profile.tool_kind != "hipprof":
            return ProfileResult(
                False, {},
                f"K100 performance evaluation requires hipprof, got {self.profile.tool_kind}",
            )
        if self.harness_argv is None or len(self.harness_argv) < 2:
            return ProfileResult(False, {}, "frozen hipprof workload driver is missing")
        harness_cmd = list(self.harness_argv)

        if collection_mode not in {"trace", "full"}:
            return ProfileResult(False, {}, f"invalid collection mode: {collection_mode}")
        root = output_dir / (run_label or f"{role}-hardware-profile")
        root.mkdir(parents=True, exist_ok=True)
        evaluator = Path(harness_cmd[1]).resolve()
        suite = evaluator.with_name("run_hipprof_suite.py")
        analyzer = evaluator.with_name("analyze_hipprof_suite.py")
        if not suite.is_file() or not analyzer.is_file():
            return ProfileResult(False, {}, "task-local hipprof suite or analyzer is missing")
        return self._run_hipprof_suite(
            artifact_dir, output_dir, root, role, suite, analyzer,
            collection_mode=collection_mode, case_ids=case_ids,
            report_label=run_label or role, implementation=implementation)

    def _run_legacy_profile_route(
        self,
        artifact_dir: Path,
        output_dir: Path,
        root: Path,
        role: str,
        harness_cmd: List[str],
    ) -> ProfileResult:
        """Retained only for parsing historical profiler fixtures."""
        cases: List[Dict[str, Any]] = []
        commands: List[List[str]] = []
        for case_id in self.profile.representative_cases:
            case_root = root / case_id
            case_root.mkdir(parents=True, exist_ok=True)
            for group_index, counters in enumerate(self.profile.counter_groups, 1):
                pass_dir = case_root / f"pass_{group_index}"
                pass_dir.mkdir(parents=True, exist_ok=True)
                command = self._command(harness_cmd, case_id, counters, pass_dir)
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
                try:
                    proc = subprocess.run(
                        command, cwd=str(artifact_dir), env=env, text=True,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        timeout=120, check=False, preexec_fn=_disable_core_dump,
                    )
                except subprocess.TimeoutExpired as exc:
                    stdout = exc.stdout or ""
                    stderr = exc.stderr or ""
                    if isinstance(stdout, bytes):
                        stdout = stdout.decode(errors="replace")
                    if isinstance(stderr, bytes):
                        stderr = stderr.decode(errors="replace")
                    (pass_dir / "profiler.stdout.log").write_text(stdout, encoding="utf-8")
                    (pass_dir / "profiler.stderr.log").write_text(stderr, encoding="utf-8")
                    report = self._report(cases, commands)
                    _write_json(output_dir / f"{role}-hardware-profile.json", report)
                    return ProfileResult(
                        False, report,
                        f"{self.profile.tool_kind} timed out for {case_id} "
                        f"pass {group_index} after 120 seconds",
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
                    _validate_harness_profile(pass_dir, case_id)
                except ProfilerError as exc:
                    report = self._report(cases, commands)
                    _write_json(output_dir / f"{role}-hardware-profile.json", report)
                    return ProfileResult(False, report, str(exc))
            try:
                kernel_token = (
                    "matmul_kernel" if role == "baseline"
                    else self.profile.kernel_name_contains
                )
                cases.append(
                    _parse_case(case_id, case_root, kernel_token)
                )
            except (OSError, ValueError, ProfilerError) as exc:
                report = self._report(cases, commands)
                _write_json(output_dir / f"{role}-hardware-profile.json", report)
                return ProfileResult(False, report, str(exc))
        report = self._report(cases, commands)
        _write_json(output_dir / f"{role}-hardware-profile.json", report)
        return ProfileResult(True, report)

    def _run_hipprof_suite(
        self,
        artifact_dir: Path,
        output_dir: Path,
        root: Path,
        role: str,
        suite: Path,
        analyzer: Path,
        *,
        collection_mode: str,
        case_ids: Optional[List[str]],
        report_label: str,
        implementation: Optional[str],
    ) -> ProfileResult:
        """Run task-local full trace + PMC/read/write collection.

        The frozen harness owns case enumeration and analysis. The system still
        owns the resolved hipprof executable, candidate artifact, weights, and
        output directory; no external benchmark checkout is consulted.
        """
        implementation = implementation or (
            "triton" if role == "baseline" else "candidate"
        )
        if implementation not in {"triton", "candidate"}:
            return ProfileResult(False, {}, f"invalid implementation: {implementation}")
        python_executable = _python_executable()
        command = [
            python_executable, str(suite),
            "--hipprof", self.profile.executable,
            "--output-dir", str(root.resolve()),
            "--implementations", implementation,
            "--passes", collection_mode,
        ]
        if case_ids:
            command.extend(["--case-ids", ",".join(case_ids)])
        env = dict(os.environ)
        env.update(self.private_env)
        env.update({
            "METAINFER_EVALUATION_PHASE": "profile-batch",
            "METAINFER_EVALUATION_ROLE": role,
            "METAINFER_BUILD_ARTIFACT_DIR": str(artifact_dir.resolve()),
            "METAINFER_REPORT_PATH": str((root / "suite-report.json").resolve()),
            "METAINFER_BENCHMARK_PROTOCOL": json.dumps(
                self.benchmark_protocol, sort_keys=True
            ),
            "PYTHONDONTWRITEBYTECODE": "1",
        })
        try:
            proc = subprocess.run(
                command, cwd=str(suite.parent), env=env, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=2400 if collection_mode == "full" else 900,
                check=False, preexec_fn=_disable_core_dump,
            )
        except subprocess.TimeoutExpired as exc:
            report = self._report([], [command])
            _write_json(output_dir / f"{report_label}-hardware-profile.json", report)
            return ProfileResult(False, report, f"hipprof suite timed out: {exc}")
        (root / "suite.stdout.log").write_text(proc.stdout or "", encoding="utf-8")
        (root / "suite.stderr.log").write_text(proc.stderr or "", encoding="utf-8")
        if proc.returncode:
            report = self._report([], [command])
            _write_json(output_dir / f"{report_label}-hardware-profile.json", report)
            return ProfileResult(False, report, "task-local hipprof suite failed")

        analyze_command = [python_executable, str(analyzer), str(root.resolve())]
        analyzed = subprocess.run(
            analyze_command, cwd=str(analyzer.parent), env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=300, check=False, preexec_fn=_disable_core_dump,
        )
        (root / "analyze.stdout.log").write_text(
            analyzed.stdout or "", encoding="utf-8")
        (root / "analyze.stderr.log").write_text(
            analyzed.stderr or "", encoding="utf-8")
        if analyzed.returncode:
            report = self._report([], [command, analyze_command])
            _write_json(output_dir / f"{report_label}-hardware-profile.json", report)
            return ProfileResult(False, report, "task-local hipprof analysis failed")

        metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
        cases = []
        for row in metrics.get("rows", []):
            read_bytes = float(row.get("hbm_read_bytes") or 0)
            write_bytes = float(row.get("hbm_write_bytes") or 0)
            try:
                kernel_breakdown = json.loads(
                    row.get("trace_kernel_means_json") or "{}"
                )
            except (TypeError, json.JSONDecodeError):
                kernel_breakdown = {}
            cases.append({
                "id": row["case_id"],
                "duration_ns": float(row["operator_mean_us"]) * 1000.0,
                "latency_mean_us": float(row["operator_mean_us"]),
                "latency_median_us": float(row.get("operator_median_us") or 0),
                "latency_stddev_us": float(row.get("operator_stddev_us") or 0),
                "latency_cv": float(row.get("operator_cv") or 0),
                "latency_min_us": float(row.get("operator_min_us") or 0),
                "latency_max_us": float(row.get("operator_max_us") or 0),
                "operator_samples_us": list(row.get("operator_samples_us") or []),
                "dispatch_count": int(row.get("trace_dispatches_per_call") or 0),
                "pmc_dispatch_count": int(row.get("pmc_dispatch_count") or 0),
                "kernel_name": row.get("main_kernel", ""),
                "kernel_breakdown_us": kernel_breakdown,
                "vgpr_count": int(row.get("vgpr") or 0),
                "agpr_count": int(row.get("agpr") or 0),
                "sgpr_count": int(row.get("sgpr") or 0),
                "lds_bytes": int(row.get("lds_bytes") or 0),
                "scratch_bytes": int(row.get("scratch_bytes") or 0),
                "grid_size": int(row.get("grid_size") or 0),
                "workgroup_size": int(row.get("workgroup_size") or 0),
                "wave_size": int(row.get("wave_size") or 0),
                "waves_per_workgroup": row.get("waves_per_workgroup"),
                "occupancy_pct": row.get("occupancy_pct"),
                "l2_hit_pct": row.get("l2_hit_pct"),
                "hbm_read_bytes": read_bytes,
                "hbm_write_bytes": write_bytes,
                "hbm_read_gbps": row.get("hbm_read_gbs"),
                "hbm_write_gbps": row.get("hbm_write_gbs"),
                "measured_bandwidth_gbps": row.get("hbm_total_gbs"),
                "compute_busy_pct": None,
                "matrix_instructions": None,
                "valu_instructions": None,
                "counters": {
                    "HBM_READ_BYTES": read_bytes,
                    "HBM_WRITE_BYTES": write_bytes,
                },
            })
        report = self._report(cases, [command, analyze_command])
        # The task-local analyzer has already validated the exact requested
        # case set. Representative-case coverage applies only to the legacy
        # route, not to scoped diagnostic collections.
        report["passed"] = bool(cases)
        report["collection_mode"] = collection_mode
        report["collection_method"] = (
            "task-local hipprof trace plus pmc/pmc-read/pmc-write"
            if collection_mode == "full" else "task-local hipprof trace"
        )
        report["pass_records"] = metrics.get("pass_records") or []
        report["full_metrics_csv"] = str((root / "metrics.csv").resolve())
        report["timing_source"] = "hipprof kernel DurationNs"
        report["timing_cases"] = [
            {
                "id": row["case_id"],
                "latency_ms": float(row["operator_mean_us"]) / 1000.0,
                "latency_mean_ms": float(row["operator_mean_us"]) / 1000.0,
                "latency_median_ms": float(row.get("operator_median_us") or 0) / 1000.0,
                "latency_stddev_ms": float(row.get("operator_stddev_us") or 0) / 1000.0,
                "latency_cv": float(row.get("operator_cv") or 0),
                "latency_min_ms": float(row.get("operator_min_us") or 0) / 1000.0,
                "latency_max_ms": float(row.get("operator_max_us") or 0) / 1000.0,
                "operator_samples_ms": [
                    float(value) / 1000.0
                    for value in row.get("operator_samples_us") or []
                ],
                "sample_count": len(row.get("operator_samples_us") or []),
                "kernel_name": row.get("main_kernel") or row.get("timed_kernel", ""),
                "dispatch_count": int(row.get("trace_dispatches_per_call") or 0),
                "kernel_breakdown_us": json.loads(
                    row.get("trace_kernel_means_json") or "{}"
                ),
            }
            for row in metrics.get("rows", [])
        ]
        _write_json(output_dir / f"{report_label}-hardware-profile.json", report)
        return ProfileResult(report["passed"], report,
                             None if report["passed"] else "profile cases missing")

    def _command(
        self, harness_cmd: List[str], case_id: str, counters: List[str], output: Path,
    ) -> List[str]:
        if self.profile.tool_kind == "hipprof":
            # --pmc-type 3 is the stable CSV table form. hipprof appends the
            # .csv suffix to this output base.
            output_base = output / "counter_collection"
            return [
                self.profile.executable, "--pmc", "--pmc-type", "3",
                "-o", str(output_base.resolve()),
                *harness_cmd, "profile", case_id,
            ]
        if self.profile.tool_kind == "rocprofv3":
            return [
                self.profile.executable,
                "--pmc", ",".join(counters),
                "--output-format", "csv", "json",
                "--output-directory", str(output.resolve()),
                "--kernel-include-regex", self.profile.kernel_name_contains,
                "--", *harness_cmd, "profile", case_id,
            ]
        input_path = output / "counters.txt"
        input_path.write_text("pmc: " + " ".join(counters) + "\n", encoding="utf-8")
        return [
            self.profile.executable, "-i", str(input_path.resolve()),
            "-o", str((output / "counter_collection.csv").resolve()), "--timestamp", "on",
            *harness_cmd, "profile", case_id,
        ]

    def _report(self, cases: List[Dict[str, Any]], commands: List[List[str]]) -> Dict[str, Any]:
        observed = {
            str(case.get("id") or case.get("case_id") or "") for case in cases
        }
        expected = set(self.profile.representative_cases)
        return {
            "schema_version": 1,
            "passed": bool(cases) and expected.issubset(observed),
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
    dispatches: List[Dict[str, Any]] = []
    seen_dispatches: set[tuple[str, Optional[float], Optional[float]]] = set()
    for row in selected:
        name = _text(row, "Counter_Name", "CounterName")
        value = _number(_text(row, "Counter_Value", "CounterValue"))
        if name and value is not None:
            normalized = _normalize_counter(name)
            counters[normalized] = counters.get(normalized, 0.0) + value
        row_counters: Dict[str, float] = {}
        for key, raw in row.items():
            normalized = _normalize_counter(key or "")
            if _is_counter(normalized):
                parsed = _number(raw)
                if parsed is not None:
                    # hipprof expands per-instance counters as TCC_HIT[0..31].
                    # Normalize and sum them before combining dispatches.
                    row_counters[normalized] = row_counters.get(normalized, 0.0) + parsed
        for normalized, parsed in row_counters.items():
            counters[normalized] = counters.get(normalized, 0.0) + parsed

        begin = _value(row, "Start_Timestamp", "BeginNs", "Begin_Ns")
        end = _value(row, "End_Timestamp", "EndNs", "End_Ns")
        kernel_name = _text(row, "Kernel_Name", "KernelName", "Name")
        dispatch_key = (kernel_name, begin, end)
        if dispatch_key not in seen_dispatches:
            seen_dispatches.add(dispatch_key)
            dispatches.append({
                "kernel_name": kernel_name,
                "duration_ns": (
                    end - begin
                    if begin is not None and end is not None and end >= begin else None
                ),
                "grid_size": _integer(row, "Grid_Size", "GridSize", "grd"),
                "workgroup_size": _integer(row, "Workgroup_Size", "WorkgroupSize", "wgr"),
            })
    last = selected[-1]
    durations = [item["duration_ns"] for item in dispatches if item["duration_ns"] is not None]
    duration = sum(durations) if durations else None
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
        "dispatch_count": len(dispatches),
        "dispatches": dispatches,
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


def _validate_harness_profile(pass_dir: Path, case_id: str) -> None:
    """Tie a profiler CSV to the exact successful harness case invocation."""
    path = pass_dir / "harness-profile.json"
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProfilerError(
            f"missing or invalid harness profile report for {case_id}"
        ) from exc
    if report.get("passed") is not True or report.get("case_id") != case_id:
        raise ProfilerError(
            f"harness profile report does not match requested case {case_id}"
        )


_KNOWN_COUNTERS = {
    # Compute / instruction counters
    "SQ_WAVES", "SQ_WAVE_CYCLES", "SQ_INSTS_VALU", "SQ_INSTS_SALU",
    "SQ_INSTS", "SQ_BUSY_CYCLES", "SQ_CYCLES",
    "SQ_ACTIVE_INST_VALU", "SQ_INSTS_FLAT_LDS_ONLY", "SQ_INSTS_LDS",
    "SQ_INSTS_VMEM_RD", "SQ_INSTS_VMEM_WR", "SQ_LDS_BANK_CONFLICT",
    "SQ_WAIT_INST_LDS",
    # Legacy matrix counters (not available on gfx928; kept for compatibility)
    "SQ_INSTS_MFMA", "SQ_INSTS_MMAC",
    # TCC / cache counters
    "TCC_HIT", "TCC_MISS", "TCC_READ", "TCC_WRITE", "TCC_REQ",
    "TCC_BUSY",
    # ATCL2 (L2) counters (gfx928)
    "ATCL2_NUMBER_OF_BANK0_HITS", "ATCL2_NUMBER_OF_BANK0_MISSES",
    "ATCL2_NUMBER_OF_BANK0_REQUESTS",
    # Legacy L2/bandwidth counters (not available on gfx928; kept for compatibility)
    "FETCH_SIZE", "WRITE_SIZE", "L2_CACHE_HIT",
    # GRBM / GPU utilization counters
    "GRBM_COUNT", "GRBM_GUI_ACTIVE",
    "GRBM_CPC_BUSY", "GRBM_CPF_BUSY", "GRBM_SPI_BUSY",
    # Legacy GPU busy counter (not available on gfx928; kept for compatibility)
    "GPU_BUSY",
}


def _is_counter(normalized: str) -> bool:
    """Recognize the wide, architecture-specific PMC columns from hipprof."""
    return normalized in _KNOWN_COUNTERS or normalized.startswith(
        ("SQ_", "TCC_", "TA_", "TCP_", "GRBM_", "ATCL2_")
    )


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
