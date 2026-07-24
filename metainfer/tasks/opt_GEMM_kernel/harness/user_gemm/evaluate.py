#!/usr/bin/env python3
"""Python evaluator for opt_GEMM_kernel — Triton as reference and baseline.

Replaces evaluate_native.cpp.  Uses Triton matmul_int8 as the correctness
reference AND as the performance baseline, so the MetaInfer optimization
loop chases Triton-level (MFMA) throughput.

Phases:
  correctness  – candidate vs Triton, per-element comparison
  benchmark    – GPU-event timed measurement (Triton for baseline role,
                 candidate .so for candidate role)
  profile      – single candidate launch (wrapped by rocprof)
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import struct
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# ── Triton import ──
sys.path.insert(0, "/usr/local/lib/python3.10/dist-packages/lmslim/layers/gemm")
from int8_utils import matmul_int8, per_token_quant_int8  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"missing environment variable: {name}")
    return value


def _float_to_bf16(value: float) -> int:
    """Round float32 -> bfloat16, return as uint16 bits."""
    arr = np.array([value], dtype=np.float32)
    # reinterpret as uint32, round to nearest even for bf16
    bits = arr.view(np.uint32)[0]
    bits += 0x7FFF + ((bits >> 16) & 1)
    return int((bits >> 16) & 0xFFFF)


def _bf16_to_float(bits: int) -> float:
    arr = np.array([np.uint32(bits << 16)], dtype=np.uint32)
    return float(arr.view(np.float32)[0])


# ═══════════════════════════════════════════════════════════════════════════════
# weight store
# ═══════════════════════════════════════════════════════════════════════════════


class WeightStore:
    """Load and derive weights, mirroring evaluate_native.cpp WeightStore."""

    def __init__(self, root: Path) -> None:
        info_path = root / "info.json"
        if not info_path.is_file():
            raise RuntimeError("weight directory has no info.json")
        with open(info_path, "rb") as f:
            self._info = json.loads(f.read().decode("utf-8"))
        self._validate_meta("q_proj_a", [4096, 1024], "int8", 4194304)
        self._validate_meta("q_proj_a_scale", [1024], "float32", 4096)
        self._validate_meta("q_proj_b", [1024, 32768], "int8", 33554432)
        self._validate_meta("q_proj_b_scale", [32768], "float32", 131072)
        self._validate_meta("kv_proj", [4096, 512], "int8", 2097152)
        self._validate_meta("kv_proj_scale", [512], "float32", 2048)
        self._validate_meta("o_proj", [8192, 4096], "int8", 33554432)
        self._validate_meta("o_proj_scale", [4096], "float32", 16384)
        self._validate_meta("moe_w1", [4096, 2048], "int8", 8388608)
        self._validate_meta("moe_w1_scale", [2048], "float32", 8192)
        self._validate_meta("moe_w2", [2048, 4096], "int8", 8388608)
        self._validate_meta("moe_w2_scale", [4096], "float32", 16384)
        self._validate_meta("moe_w3", [4096, 2048], "int8", 8388608)
        self._validate_meta("moe_w3_scale", [2048], "float32", 8192)

        self.qa = self._load_bin(root / "q_proj_a.bin", np.int8, 4096 * 1024)
        self.qas = self._load_bin(root / "q_proj_a_scale.bin", np.float32, 1024)
        self.qb = self._load_bin(root / "q_proj_b.bin", np.int8, 1024 * 32768)
        self.qbs = self._load_bin(root / "q_proj_b_scale.bin", np.float32, 32768)
        self.kv = self._load_bin(root / "kv_proj.bin", np.int8, 4096 * 512)
        self.kvs = self._load_bin(root / "kv_proj_scale.bin", np.float32, 512)
        self.o = self._load_bin(root / "o_proj.bin", np.int8, 8192 * 4096)
        self.os = self._load_bin(root / "o_proj_scale.bin", np.float32, 4096)
        self.w1 = self._load_bin(root / "moe_w1.bin", np.int8, 4096 * 2048)
        self.w1s = self._load_bin(root / "moe_w1_scale.bin", np.float32, 2048)
        self.w2 = self._load_bin(root / "moe_w2.bin", np.int8, 2048 * 4096)
        self.w2s = self._load_bin(root / "moe_w2_scale.bin", np.float32, 4096)
        self.w3 = self._load_bin(root / "moe_w3.bin", np.int8, 4096 * 2048)
        self.w3s = self._load_bin(root / "moe_w3_scale.bin", np.float32, 2048)

    def _validate_meta(
        self, name: str, shape: List[int], dtype: str, nbytes: int,
    ) -> None:
        meta = self._info.get(name)
        if meta is None:
            raise RuntimeError(f"info.json is missing {name}")
        if meta.get("shape") != shape:
            raise RuntimeError(f"shape mismatch for {name}")
        if meta.get("dtype") != dtype:
            raise RuntimeError(f"dtype mismatch for {name}")
        if meta.get("nbytes") != nbytes:
            raise RuntimeError(f"nbytes mismatch for {name}")

    @staticmethod
    def _load_bin(path: Path, dtype: np.dtype, count: int) -> np.ndarray:
        itemsize = np.dtype(dtype).itemsize
        if not path.is_file() or path.stat().st_size != count * itemsize:
            raise RuntimeError(f"binary size mismatch: {path}")
        data = np.fromfile(path, dtype=dtype)
        if data.size != count:
            raise RuntimeError(f"cannot load {path}")
        return data.copy()

    def derive(self, case: Case) -> Tuple[np.ndarray, np.ndarray]:
        """Return (weight_int8 [K,N], weight_scale [N])."""
        if case.op == "wqkv_a":
            w = np.concatenate(
                [self.qa.reshape(4096, 1024), self.kv.reshape(4096, 512)], axis=1
            )
            s = np.concatenate([self.qas, self.kvs])
        elif case.op == "wq_b":
            width = 32768 // case.tp
            w = self.qb.reshape(1024, 32768)[:, :width].copy()
            s = self.qbs[:width].copy()
        elif case.op == "wo_b":
            depth = 8192 // case.tp
            w = self.o.reshape(8192, 4096)[:depth, :].copy()
            s = self.os.copy()
        elif case.op == "shared_gate_up_proj":
            width = 2048 // case.tp
            w = np.concatenate(
                [
                    self.w1.reshape(4096, 2048)[:, :width],
                    self.w3.reshape(4096, 2048)[:, :width],
                ],
                axis=1,
            )
            s = np.concatenate([self.w1s[:width], self.w3s[:width]])
        elif case.op == "shared_down_proj":
            depth = 2048 // case.tp
            w = self.w2.reshape(2048, 4096)[:depth, :].copy()
            s = self.w2s.copy()
        else:
            raise RuntimeError(f"unknown workload {case.op}")

        if w.shape[0] != case.k or w.shape[1] != case.n or s.shape[0] != case.n:
            raise RuntimeError(f"derived tensor shape mismatch for {case.id}")
        return np.ascontiguousarray(w), np.ascontiguousarray(s)


# ═══════════════════════════════════════════════════════════════════════════════
# case generation
# ═══════════════════════════════════════════════════════════════════════════════


class Case:
    __slots__ = ("id", "op", "tp", "m", "n", "k")

    def __init__(self, id: str, op: str, tp: int, m: int, n: int, k: int) -> None:
        self.id = id
        self.op = op
        self.tp = tp
        self.m = m
        self.n = n
        self.k = k


def _public_cases() -> List[Case]:
    workloads = [
        ("wqkv-a-tp4", "wqkv_a", 4, 4096, 1536),
        ("wq-b-tp4", "wq_b", 4, 1024, 8192),
        ("wo-b-tp4", "wo_b", 4, 2048, 4096),
        ("shared-gate-up-proj-tp4", "shared_gate_up_proj", 4, 4096, 1024),
        ("shared-down-proj-tp4", "shared_down_proj", 4, 512, 4096),
        ("wqkv-a-tp8", "wqkv_a", 8, 4096, 1536),
        ("wq-b-tp8", "wq_b", 8, 1024, 4096),
        ("wo-b-tp8", "wo_b", 8, 1024, 4096),
        ("shared-gate-up-proj-tp8", "shared_gate_up_proj", 8, 4096, 512),
        ("shared-down-proj-tp8", "shared_down_proj", 8, 256, 4096),
    ]
    ms = [1, 2, 4, 8, 16, 4096]
    result = []
    for wl_id, op, tp, k, n in workloads:
        for m in ms:
            result.append(Case(f"{wl_id}-m{m}", op, tp, m, n, k))
    return result


def _correctness_cases() -> List[Case]:
    result = _public_cases()
    result.append(Case("heldout-wq-b-tp4-m7", "wq_b", 4, 7, 8192, 1024))
    result.append(Case("heldout-wo-b-tp8-m13", "wo_b", 8, 13, 4096, 1024))
    result.append(
        Case("heldout-shared-gate-up-proj-tp4-m3", "shared_gate_up_proj", 4, 3, 1024, 4096)
    )
    result.append(
        Case("heldout-shared-down-proj-tp8-m7", "shared_down_proj", 8, 7, 4096, 256)
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# activation generation
# ═══════════════════════════════════════════════════════════════════════════════


def _case_seed(case_id: str) -> int:
    h = hashlib.sha256(case_id.encode()).digest()
    return int.from_bytes(h[:8], "little") & 0x7FFFFFFF


def _generate_activation(case: Case) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate deterministic BF16 activations [M,K] and quantize them.

    Returns (a_bf16, a_int8, a_scale) on CPU.
    """
    seed = _case_seed(case.id)
    g = torch.Generator()
    g.manual_seed(seed)
    # Generate float32, round-trip through BF16 (matching C++ behaviour)
    x_fp32 = torch.randn((case.m, case.k), generator=g, dtype=torch.float32) * 1.5
    x_bf16 = x_fp32.to(torch.bfloat16).to(torch.float32)  # round-trip
    # Per-row symmetric INT8 quantize
    absmax = x_bf16.abs().amax(dim=1, keepdim=True)  # [M, 1]
    scale = absmax / 127.0  # [M, 1]
    x_int8 = (x_bf16 / scale.clamp(min=1e-12)).round().clamp(-127, 127).to(torch.int8)
    return x_bf16.to(torch.bfloat16), x_int8, scale.squeeze(1)


# ═══════════════════════════════════════════════════════════════════════════════
# candidate loading
# ═══════════════════════════════════════════════════════════════════════════════


class Candidate:
    """dlopen the candidate shared library from the artifact directory."""

    def __init__(self, artifact_dir: Path) -> None:
        libs = list(artifact_dir.rglob("libmetainfer_gemm_candidate*.so"))
        if not libs:
            raise RuntimeError("candidate shared library is missing")
        lib_path = str(libs[0])
        self._handle = ctypes.CDLL(lib_path, mode=ctypes.RTLD_LOCAL)
        self._launch = self._handle.launch_w8a8_gemm
        self._launch.argtypes = [
            ctypes.c_void_p,  # a (int8)
            ctypes.c_void_p,  # w (int8)
            ctypes.c_void_p,  # a_scale (float32)
            ctypes.c_void_p,  # w_scale (float32)
            ctypes.c_void_p,  # y (bf16)
            ctypes.c_int,     # M
            ctypes.c_int,     # N
            ctypes.c_int,     # K
            ctypes.c_void_p,  # stream
        ]
        self._launch.restype = ctypes.c_int

    def launch(
        self,
        a: torch.Tensor,
        w: torch.Tensor,
        a_scale: torch.Tensor,
        w_scale: torch.Tensor,
        y: torch.Tensor,
        stream: int = 0,
    ) -> int:
        M, N, K = a.shape[0], w.shape[1], a.shape[1]
        return self._launch(
            a.data_ptr(), w.data_ptr(), a_scale.data_ptr(), w_scale.data_ptr(),
            y.data_ptr(), M, N, K, stream,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# correctness
# ═══════════════════════════════════════════════════════════════════════════════


def _run_correctness_case(
    candidate: Candidate,
    weights: WeightStore,
    case: Case,
    device: torch.device,
) -> Dict[str, Any]:
    a_bf16, a_int8, a_scale = _generate_activation(case)
    w_int8_np, w_scale_np = weights.derive(case)

    # Move to device
    a_int8_dev = a_int8.to(device)
    a_scale_dev = a_scale.to(device)
    w_int8_dev = torch.from_numpy(w_int8_np).to(device)
    w_scale_dev = torch.from_numpy(w_scale_np).to(device)

    total = case.m * case.n

    # ── Candidate ──
    y_cand = torch.empty((case.m, case.n), dtype=torch.bfloat16, device=device)
    ret = candidate.launch(a_int8_dev, w_int8_dev, a_scale_dev, w_scale_dev, y_cand)
    if ret != 0:
        raise RuntimeError(f"candidate returned non-zero for {case.id}")
    torch.cuda.synchronize()

    # ── Triton reference ──
    y_ref = matmul_int8(
        a_int8_dev, a_scale_dev, w_int8_dev, w_scale_dev, torch.bfloat16, None,
    )
    torch.cuda.synchronize()

    # ── Compare ──
    got = y_cand.float().cpu()
    expected = y_ref.float().cpu()

    diff = (got - expected).abs()
    max_abs = float(diff.max().item())
    mismatches = int((diff > 1e-3).sum().item())
    passed = mismatches == 0

    return {
        "id": case.id,
        "passed": passed,
        "mismatches": mismatches,
        "elements": total,
        "max_abs_error": max_abs,
    }


def _run_triton_correctness_case(
    weights: WeightStore,
    case: Case,
    device: torch.device,
) -> Dict[str, Any]:
    """Certify that the frozen Triton reference executes and is finite."""
    _, a_int8, a_scale = _generate_activation(case)
    w_int8_np, w_scale_np = weights.derive(case)
    output = matmul_int8(
        a_int8.to(device), a_scale.to(device),
        torch.from_numpy(w_int8_np).to(device),
        torch.from_numpy(w_scale_np).to(device),
        torch.bfloat16, None,
    )
    torch.cuda.synchronize()
    shape_ok = tuple(output.shape) == (case.m, case.n)
    finite = bool(torch.isfinite(output.float()).all().item())
    return {
        "id": case.id,
        "passed": shape_ok and finite,
        "shape_ok": shape_ok,
        "finite": finite,
        "elements": case.m * case.n,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# benchmark
# ═══════════════════════════════════════════════════════════════════════════════


def _benchmark_case_triton(
    weights: WeightStore,
    case: Case,
    device: torch.device,
    warmup: int,
    samples: int,
) -> Dict[str, Any]:
    """Benchmark Triton matmul_int8 with GPU events."""
    a_bf16, a_int8, a_scale = _generate_activation(case)
    w_int8_np, w_scale_np = weights.derive(case)

    a_int8_dev = a_int8.to(device)
    a_scale_dev = a_scale.to(device)
    w_int8_dev = torch.from_numpy(w_int8_np).to(device)
    w_scale_dev = torch.from_numpy(w_scale_np).to(device)

    # Warmup
    for _ in range(warmup):
        matmul_int8(a_int8_dev, a_scale_dev, w_int8_dev, w_scale_dev, torch.bfloat16, None)
    torch.cuda.synchronize()

    values = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        matmul_int8(a_int8_dev, a_scale_dev, w_int8_dev, w_scale_dev, torch.bfloat16, None)
        end.record()
        torch.cuda.synchronize()
        values.append(start.elapsed_time(end))

    values.sort()
    latency = values[len(values) // 2]
    flops = 2.0 * case.m * case.n * case.k
    # rough byte count: A(bf16)+W(int8)+scales+output(bf16)
    nbytes = (case.m * case.k * 2 + case.k * case.n * 1
              + case.m * 4 + case.n * 4 + case.m * case.n * 2)

    return {
        "id": case.id,
        "latency_ms": latency,
        "min_ms": values[0],
        "max_ms": values[-1],
        "tops": flops / (latency * 1e9),
        "bandwidth_gbps": nbytes / (latency * 1e6),
    }


def _benchmark_case_candidate(
    candidate: Candidate,
    weights: WeightStore,
    case: Case,
    device: torch.device,
    warmup: int,
    samples: int,
) -> Dict[str, Any]:
    """Benchmark candidate .so with GPU events."""
    a_bf16, a_int8, a_scale = _generate_activation(case)
    w_int8_np, w_scale_np = weights.derive(case)

    a_int8_dev = a_int8.to(device)
    a_scale_dev = a_scale.to(device)
    w_int8_dev = torch.from_numpy(w_int8_np).to(device)
    w_scale_dev = torch.from_numpy(w_scale_np).to(device)
    y = torch.empty((case.m, case.n), dtype=torch.bfloat16, device=device)

    # Warmup
    for _ in range(warmup):
        ret = candidate.launch(a_int8_dev, w_int8_dev, a_scale_dev, w_scale_dev, y)
        if ret != 0:
            raise RuntimeError(f"candidate returned non-zero for {case.id}")
    torch.cuda.synchronize()

    values = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        ret = candidate.launch(a_int8_dev, w_int8_dev, a_scale_dev, w_scale_dev, y)
        if ret != 0:
            raise RuntimeError(f"candidate returned non-zero for {case.id}")
        end.record()
        torch.cuda.synchronize()
        values.append(start.elapsed_time(end))

    values.sort()
    latency = values[len(values) // 2]
    flops = 2.0 * case.m * case.n * case.k
    nbytes = (case.m * case.k * 2 + case.k * case.n * 1
              + case.m * 4 + case.n * 4 + case.m * case.n * 2)

    return {
        "id": case.id,
        "latency_ms": latency,
        "min_ms": values[0],
        "max_ms": values[-1],
        "tops": flops / (latency * 1e9),
        "bandwidth_gbps": nbytes / (latency * 1e6),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# profile
# ═══════════════════════════════════════════════════════════════════════════════


def _profile_case(
    candidate: Candidate,
    weights: WeightStore,
    case: Case,
    device: torch.device,
) -> None:
    """Single candidate launch for rocprof capture."""
    a_bf16, a_int8, a_scale = _generate_activation(case)
    w_int8_np, w_scale_np = weights.derive(case)

    a_int8_dev = a_int8.to(device)
    a_scale_dev = a_scale.to(device)
    w_int8_dev = torch.from_numpy(w_int8_np).to(device)
    w_scale_dev = torch.from_numpy(w_scale_np).to(device)
    y = torch.empty((case.m, case.n), dtype=torch.bfloat16, device=device)

    torch.cuda.synchronize()
    ret = candidate.launch(a_int8_dev, w_int8_dev, a_scale_dev, w_scale_dev, y)
    if ret != 0:
        raise RuntimeError(f"candidate returned non-zero for {case.id}")
    torch.cuda.synchronize()


def _profile_case_triton(
    weights: WeightStore,
    case: Case,
    device: torch.device,
) -> None:
    """Warm up Triton JIT, then launch exactly one profiled invocation."""
    _, a_int8, a_scale = _generate_activation(case)
    w_int8_np, w_scale_np = weights.derive(case)
    a_int8_dev = a_int8.to(device)
    a_scale_dev = a_scale.to(device)
    w_int8_dev = torch.from_numpy(w_int8_np).to(device)
    w_scale_dev = torch.from_numpy(w_scale_np).to(device)
    # Triton's disk cache is populated by certification benchmark. This is
    # the single matmul invocation observed by hipprof in this process.
    matmul_int8(
        a_int8_dev, a_scale_dev, w_int8_dev, w_scale_dev,
        torch.bfloat16, None,
    )
    torch.cuda.synchronize()


# ═══════════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    phase = sys.argv[1]
    is_profile = phase == "profile"
    is_eval = phase in ("correctness", "benchmark")
    if not is_profile and not is_eval:
        raise RuntimeError(
            "usage: evaluate.py correctness|benchmark|profile CASE_ID"
        )

    report_path = Path(_env("METAINFER_REPORT_PATH"))
    weight_root = Path(_env("METAINFER_WEIGHT_BUNDLE")).resolve()
    artifact_dir = Path(_env("METAINFER_BUILD_ARTIFACT_DIR")).resolve()
    role = _env("METAINFER_EVALUATION_ROLE")

    if phase != _env("METAINFER_EVALUATION_PHASE"):
        raise RuntimeError("phase mismatch")

    device = torch.device("cuda:0")

    weights = WeightStore(weight_root)
    candidate = None if role == "baseline" else Candidate(artifact_dir)

    if is_profile:
        case_id = sys.argv[2]
        cases = _public_cases()
        found = next((c for c in cases if c.id == case_id), None)
        if found is None:
            raise RuntimeError(f"unknown public profile case: {case_id}")
        if role == "baseline":
            _profile_case_triton(weights, found, device)
        else:
            assert candidate is not None
            _profile_case(candidate, weights, found, device)
        write_json(report_path, {
            "passed": True,
            "case_id": found.id,
            "implementation": "triton" if role == "baseline" else "candidate",
            "timed_scope": "launch_w8a8_gemm_only",
        })
        return

    if phase == "correctness":
        all_cases = _correctness_cases()
        report: Dict[str, Any] = {
            "passed": True,
            "reference": "Triton matmul_int8 (MFMA hardware)",
            "activation_quantization_timed": False,
            "cases": [],
        }
        for c in all_cases:
            try:
                case_result = (
                    _run_triton_correctness_case(weights, c, device)
                    if role == "baseline" else
                    _run_correctness_case(candidate, weights, c, device)
                )
            except Exception as exc:
                write_json(
                    report_path,
                    {"passed": False, "reason": str(exc), "cases": []},
                )
                sys.exit(2)
            if not case_result["passed"]:
                report["passed"] = False
            report["cases"].append(case_result)
        write_json(report_path, report)
        return

    if phase == "benchmark":
        protocol = json.loads(_env("METAINFER_BENCHMARK_PROTOCOL"))
        warmup = int(protocol["warmup"])
        samples = int(protocol["samples"])
        all_cases = _public_cases()
        cases_out = []
        for c in all_cases:
            try:
                if role == "baseline":
                    item = _benchmark_case_triton(weights, c, device, warmup, samples)
                else:
                    assert candidate is not None
                    item = _benchmark_case_candidate(
                        candidate, weights, c, device, warmup, samples
                    )
            except Exception as exc:
                write_json(
                    report_path,
                    {"passed": False, "reason": str(exc), "cases": []},
                )
                sys.exit(2)
            cases_out.append(item)
        write_json(report_path, {
            "passed": True,
            "methodology": protocol,
            "timed_scope": "launch_w8a8_gemm_only",
            "activation_quantization_timed": False,
            "weight_loading_or_preprocessing_timed": False,
            "cases": cases_out,
        })


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        report_path = Path(os.environ.get("METAINFER_REPORT_PATH", "/dev/null"))
        try:
            write_json(
                report_path,
                {"passed": False, "reason": str(exc), "cases": []},
            )
        except Exception:
            pass
        print(f"GEMM harness failed: {exc}", file=sys.stderr)
        sys.exit(2)
