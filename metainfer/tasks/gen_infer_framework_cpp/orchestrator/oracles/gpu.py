"""Best-effort NVIDIA and DTK/ROCm telemetry for performance runs.

The sampler is observational. It never changes clocks, resets devices or
terminates processes. Missing tools/metrics produce ``collected: false`` or
partial fields without affecting the benchmark process.
"""

from __future__ import annotations

import os
import re
import shutil
import statistics
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GpuSamples:
    """Collected GPU telemetry samples over one benchmark window."""

    samples: List[Dict[str, float]] = field(default_factory=list)
    gpu_name: Optional[str] = None
    device_count: int = 0
    backend: Optional[str] = None
    device_peaks: Dict[int, Dict[str, float]] = field(default_factory=dict)

    def aggregate(self) -> Dict[str, Any]:
        if not self.samples:
            return {
                "collected": False,
                "backend": self.backend,
                "gpu_name": self.gpu_name,
                "device_count": self.device_count,
            }
        keys = (
            "utilization_gpu",
            "utilization_memory",
            "memory_used_mib",
            "power_draw_w",
        )
        aggregate: Dict[str, Any] = {
            "collected": True,
            "sample_count": len(self.samples),
            "gpu_name": self.gpu_name,
            "device_count": self.device_count,
            "backend": self.backend,
        }
        for key in keys:
            values = [sample[key] for sample in self.samples if key in sample]
            if not values:
                continue
            aggregate[f"{key}_mean"] = round(statistics.mean(values), 2)
            aggregate[f"{key}_max"] = round(max(values), 2)
            if len(values) >= 2:
                aggregate[f"{key}_stdev"] = round(statistics.stdev(values), 2)
            ordered = sorted(values)
            aggregate[f"{key}_p50"] = round(_percentile(ordered, 0.50), 2)
            if len(values) >= 10:
                aggregate[f"{key}_p99"] = round(_percentile(ordered, 0.99), 2)

        per_device = [
            {
                "index": index,
                **{key: round(value, 2) for key, value in values.items()},
            }
            for index, values in sorted(self.device_peaks.items())
        ]
        aggregate["per_device_peaks"] = per_device
        aggregate["active_device_count"] = sum(
            1
            for device in per_device
            if device.get("memory_used_mib", 0.0) >= 128.0
            or device.get("utilization_gpu", 0.0) > 0.0
        )
        return aggregate


def _percentile(sorted_values: List[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = q * (len(sorted_values) - 1)
    low = int(index)
    high = min(low + 1, len(sorted_values) - 1)
    fraction = index - low
    return sorted_values[low] * (1 - fraction) + sorted_values[high] * fraction


class GpuTelemetry:
    """Poll an available GPU management tool in a background thread."""

    def __init__(
        self,
        poll_interval_s: float = 1.0,
        preferred_backend: Optional[str] = None,
    ) -> None:
        self.poll_interval_s = poll_interval_s
        self._samples = GpuSamples()
        self._stop_evt: Optional[threading.Event] = None
        self._thread: Optional[threading.Thread] = None
        self._nvidia_smi: Optional[str] = shutil.which("nvidia-smi")
        self._rocm_smi: Optional[str] = _find_rocm_smi()
        if preferred_backend == "rocm-smi":
            self._nvidia_smi = None
        elif preferred_backend == "nvidia-smi":
            self._rocm_smi = None

    def __enter__(self) -> "GpuTelemetry":
        if self._nvidia_smi:
            if self._probe_nvidia():
                self._samples.backend = "nvidia-smi"
            else:
                self._nvidia_smi = None
        if not self._nvidia_smi and self._rocm_smi:
            if self._probe_rocm():
                self._samples.backend = "rocm-smi"
            else:
                self._rocm_smi = None

        if not self._nvidia_smi and not self._rocm_smi:
            return self
        self._stop_evt = threading.Event()
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="gpu-telemetry",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._stop_evt and self._thread:
            self._stop_evt.set()
            self._thread.join(timeout=self.poll_interval_s * 2)

    def _probe_nvidia(self) -> bool:
        assert self._nvidia_smi is not None
        try:
            process = subprocess.run(
                [
                    self._nvidia_smi,
                    "--query-gpu=name",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (subprocess.SubprocessError, OSError):
            return False
        if process.returncode != 0:
            return False
        names = [line.strip() for line in process.stdout.splitlines() if line.strip()]
        self._samples.device_count = len(names)
        self._samples.gpu_name = names[0] if names else None
        return bool(names)

    def _probe_rocm(self) -> bool:
        assert self._rocm_smi is not None
        try:
            process = subprocess.run(
                [self._rocm_smi, "--showproductname"],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (subprocess.SubprocessError, OSError):
            return False
        if process.returncode != 0:
            return False
        names = _parse_rocm_names(process.stdout)
        self._samples.device_count = len(names)
        self._samples.gpu_name = names[0] if names else None
        return bool(names)

    def _poll_loop(self) -> None:
        if self._nvidia_smi:
            self._poll_nvidia()
        elif self._rocm_smi:
            self._poll_rocm()

    def _poll_nvidia(self) -> None:
        assert self._nvidia_smi is not None
        command = [
            self._nvidia_smi,
            "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,power.draw",
            "--format=csv,noheader,nounits",
        ]
        while self._stop_evt and not self._stop_evt.is_set():
            try:
                process = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=max(2.0, self.poll_interval_s * 2),
                )
                if process.returncode == 0:
                    self._record_devices(_parse_nvidia_samples(process.stdout))
            except (subprocess.SubprocessError, OSError):
                pass
            self._stop_evt.wait(self.poll_interval_s)

    def _poll_rocm(self) -> None:
        assert self._rocm_smi is not None
        command = [
            self._rocm_smi,
            "--showuse",
            "--showmemuse",
            "--showmeminfo",
            "vram",
        ]
        while self._stop_evt and not self._stop_evt.is_set():
            try:
                process = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=max(2.0, self.poll_interval_s * 2),
                )
                if process.returncode == 0:
                    self._record_devices(_parse_rocm_samples(process.stdout))
                else:
                    # Older vendor SMI builds accept only one show flag per
                    # invocation. Fall back to separate read-only queries.
                    outputs: List[str] = []
                    for args in (("--showuse",), ("--showmemuse",), ("--showmeminfo", "vram")):
                        fallback = subprocess.run(
                            [self._rocm_smi, *args],
                            capture_output=True,
                            text=True,
                            timeout=max(2.0, self.poll_interval_s * 2),
                        )
                        if fallback.returncode == 0:
                            outputs.append(fallback.stdout)
                    self._record_devices(_parse_rocm_samples("\n".join(outputs)))
            except (subprocess.SubprocessError, OSError):
                pass
            self._stop_evt.wait(self.poll_interval_s)

    def _record_devices(self, devices: Dict[int, Dict[str, float]]) -> None:
        if not devices:
            return
        keys = set().union(*(values.keys() for values in devices.values()))
        self._samples.samples.append({
            key: statistics.mean([
                values[key] for values in devices.values() if key in values
            ])
            for key in keys
        })
        for index, values in devices.items():
            peaks = self._samples.device_peaks.setdefault(index, {})
            for key, value in values.items():
                peaks[key] = max(peaks.get(key, value), value)

    def aggregate(self) -> Dict[str, Any]:
        return self._samples.aggregate()


def _find_rocm_smi() -> Optional[str]:
    found = shutil.which("rocm-smi")
    if found:
        return found
    for path in ("/opt/dtk/bin/rocm-smi", "/usr/bin/rocm-smi"):
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def _parse_rocm_names(output: str) -> List[str]:
    indexed: Dict[int, str] = {}
    for line in output.splitlines():
        match = re.search(
            r"(?:GPU|DCU)\[(\d+)\].*?(?:Card Series|Card Model|Product Name)\s*:\s*(.+?)\s*$",
            line,
            flags=re.IGNORECASE,
        )
        if match:
            indexed[int(match.group(1))] = match.group(2).strip()
    return [indexed[index] for index in sorted(indexed)]


def _parse_rocm_samples(output: str) -> Dict[int, Dict[str, float]]:
    devices: Dict[int, Dict[str, float]] = {}
    patterns = (
        ("utilization_gpu", r"(?:GPU|DCU) use(?: \(%\))?\s*:\s*([0-9.]+)"),
        ("utilization_memory", r"GPU Memory Allocated(?: \(VRAM%\))?\s*:\s*([0-9.]+)"),
        ("memory_used_mib", r"vram Total Used Memory \(MiB\)\s*:\s*([0-9.]+)"),
    )
    for line in output.splitlines():
        index_match = re.search(r"(?:GPU|DCU)\[(\d+)\]", line, flags=re.IGNORECASE)
        if not index_match:
            continue
        index = int(index_match.group(1))
        values = devices.setdefault(index, {})
        for key, pattern in patterns:
            match = re.search(pattern, line, flags=re.IGNORECASE)
            if match:
                values[key] = float(match.group(1))
    return {index: values for index, values in devices.items() if values}


def _parse_nvidia_samples(output: str) -> Dict[int, Dict[str, float]]:
    devices: Dict[int, Dict[str, float]] = {}
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            index = int(parts[0])
            power = 0.0 if parts[4].upper() in ("[N/A]", "N/A") else float(parts[4])
            devices[index] = {
                "utilization_gpu": float(parts[1]),
                "utilization_memory": float(parts[2]),
                "memory_used_mib": float(parts[3]),
                "power_draw_w": power,
            }
        except ValueError:
            continue
    return devices


__all__ = [
    "GpuSamples",
    "GpuTelemetry",
    "_parse_nvidia_samples",
    "_parse_rocm_names",
    "_parse_rocm_samples",
]
