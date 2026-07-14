"""Best-effort GPU telemetry during perf benchmark runs.

Spawns a background thread that polls ``nvidia-smi`` at 1 Hz while the
benchmark runs, capturing:

  - utilization.gpu (percent)
  - utilization.memory (percent)
  - memory.used (MiB)
  - power.draw (W)

Aggregates to mean / max / p50 / p99 across all samples collected
during the benchmark window. If ``nvidia-smi`` isn't on PATH or the
GPU is unreachable, every call silently returns empty results — the
benchmark itself is unaffected.

Why background polling (not ``--query-gpu`` against a daemon): we want
the telemetry window to exactly match the benchmark window, including
the ramp-up as the server warms its KV cache. Daemon-mode telemetry
misses the ramp.
"""

from __future__ import annotations

import shutil
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class GpuSamples:
    """Collected GPU telemetry samples over a benchmark window."""
    samples: List[Dict[str, float]] = field(default_factory=list)
    gpu_name: Optional[str] = None
    device_count: int = 0

    def aggregate(self) -> Dict[str, Any]:
        """Return a JSON-serializable summary. Empty if no samples."""
        if not self.samples:
            return {"collected": False}
        keys = ("utilization_gpu", "utilization_memory",
                "memory_used_mib", "power_draw_w")
        agg: Dict[str, Any] = {"collected": True,
                               "sample_count": len(self.samples),
                               "gpu_name": self.gpu_name,
                               "device_count": self.device_count}
        for k in keys:
            vals = [s[k] for s in self.samples if k in s]
            if not vals:
                continue
            agg[f"{k}_mean"] = round(statistics.mean(vals), 2)
            agg[f"{k}_max"] = round(max(vals), 2)
            if len(vals) >= 2:
                agg[f"{k}_stdev"] = round(statistics.stdev(vals), 2)
            if len(vals) >= 1:
                sorted_v = sorted(vals)
                agg[f"{k}_p50"] = round(_percentile(sorted_v, 0.50), 2)
                if len(vals) >= 10:
                    agg[f"{k}_p99"] = round(_percentile(sorted_v, 0.99), 2)
        return agg


def _percentile(sorted_vals: List[float], q: float) -> float:
    """Linear-interpolation percentile on an already-sorted list."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = q * (len(sorted_vals) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


class GpuTelemetry:
    """Context manager that polls nvidia-smi in a background thread.

    Usage::

        with GpuTelemetry() as tel:
            run_benchmark()
        summary = tel.aggregate()  # add to perf report
    """

    def __init__(self, poll_interval_s: float = 1.0) -> None:
        self.poll_interval_s = poll_interval_s
        self._samples = GpuSamples()
        self._stop_evt: Optional[threading.Event] = None
        self._thread: Optional[threading.Thread] = None
        self._nvidia_smi: Optional[str] = shutil.which("nvidia-smi")

    def __enter__(self) -> "GpuTelemetry":
        if not self._nvidia_smi:
            # No nvidia-smi → silent no-op. The benchmark is unaffected.
            return self
        # Probe GPU name + count up front (single shot).
        try:
            name_proc = subprocess.run(
                [self._nvidia_smi,
                 "--query-gpu=name",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            if name_proc.returncode == 0:
                lines = [ln.strip() for ln in name_proc.stdout.splitlines() if ln.strip()]
                self._samples.device_count = len(lines)
                self._samples.gpu_name = lines[0] if lines else None
        except (subprocess.SubprocessError, OSError):
            # nvidia-smi present but errored — treat as unavailable.
            self._nvidia_smi = None
            return self

        self._stop_evt = threading.Event()
        self._thread = threading.Thread(
            target=self._poll_loop, name="gpu-telemetry", daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._stop_evt and self._thread:
            self._stop_evt.set()
            self._thread.join(timeout=self.poll_interval_s * 2)

    def _poll_loop(self) -> None:
        assert self._nvidia_smi is not None
        cmd = [
            self._nvidia_smi,
            "--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop_evt.is_set():
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=self.poll_interval_s * 2,
                )
                if proc.returncode == 0:
                    # Multi-GPU: average across devices.
                    vals: Dict[str, List[float]] = {
                        "utilization_gpu": [],
                        "utilization_memory": [],
                        "memory_used_mib": [],
                        "power_draw_w": [],
                    }
                    for line in proc.stdout.splitlines():
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) < 4:
                            continue
                        try:
                            u_gpu = float(parts[0])
                            u_mem = float(parts[1])
                            mem_used = float(parts[2])
                            power = float(parts[3])
                        except ValueError:
                            continue
                        # Some nvidia-smi builds emit "[N/A]" for power on idle GPUs.
                        if parts[3].upper() in ("[N/A]", "N/A"):
                            power = 0.0
                        vals["utilization_gpu"].append(u_gpu)
                        vals["utilization_memory"].append(u_mem)
                        vals["memory_used_mib"].append(mem_used)
                        vals["power_draw_w"].append(power)
                    if vals["utilization_gpu"]:
                        self._samples.samples.append({
                            k: statistics.mean(v) for k, v in vals.items() if v
                        })
            except subprocess.SubprocessError:
                pass
            self._stop_evt.wait(self.poll_interval_s)

    def aggregate(self) -> Dict[str, Any]:
        return self._samples.aggregate()
