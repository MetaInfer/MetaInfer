#!/usr/bin/env python3
"""Frozen correctness harness and hipprof workload driver.

Triton is the independent correctness reference and the frozen performance
baseline. Performance measurements are produced only by task-local hipprof
trace collection around ``profile-batch`` steady-state GPU dispatches.

Phases:
  correctness   – candidate vs Triton, per-element comparison
  profile-batch – all public cases, repeated steady-state calls for hipprof
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
# hipprof workload
# ═══════════════════════════════════════════════════════════════════════════════


def _profile_batch_case_candidate(
    candidate: Candidate,
    weights: WeightStore,
    case: Case,
    device: torch.device,
    calls: int,
) -> Tuple[int, int, int, int]:
    """Prepare once, then enqueue exactly ``calls`` candidate invocations."""
    _, a_int8, a_scale = _generate_activation(case)
    w_int8_np, w_scale_np = weights.derive(case)
    a_dev = a_int8.to(device)
    as_dev = a_scale.to(device)
    w_dev = torch.from_numpy(w_int8_np).to(device)
    ws_dev = torch.from_numpy(w_scale_np).to(device)
    y = torch.empty((case.m, case.n), dtype=torch.bfloat16, device=device)
    # Complete one-time weight packing, workspace allocation, and lazy runtime
    # setup before the profiler's marked steady-state interval.
    ret = candidate.launch(a_dev, w_dev, as_dev, ws_dev, y)
    if ret != 0:
        raise RuntimeError(f"candidate returned non-zero for {case.id}")
    torch.cuda.synchronize()
    begin_ns = time.monotonic_ns()
    begin_epoch_ns = time.time_ns()
    print(
        f"PROFILE_GROUP,candidate,{case.id},{case.m},{case.n},{case.k},"
        f"calls={calls}", flush=True,
    )
    for _ in range(calls):
        ret = candidate.launch(a_dev, w_dev, as_dev, ws_dev, y)
        if ret != 0:
            raise RuntimeError(f"candidate returned non-zero for {case.id}")
    torch.cuda.synchronize()
    return begin_ns, time.monotonic_ns(), begin_epoch_ns, time.time_ns()


def _profile_batch_case_triton(
    weights: WeightStore,
    case: Case,
    device: torch.device,
    calls: int,
) -> Tuple[int, int, int, int]:
    """JIT before the marked group, then enqueue fixed Triton invocations."""
    _, a_int8, a_scale = _generate_activation(case)
    w_int8_np, w_scale_np = weights.derive(case)
    a_dev = a_int8.to(device)
    as_dev = a_scale.to(device)
    w_dev = torch.from_numpy(w_int8_np).to(device)
    ws_dev = torch.from_numpy(w_scale_np).to(device)
    # Force JIT/allocation before the group marker. The analyzer uses the
    # manifest and final repeated core launches, never this preparation call.
    matmul_int8(a_dev, as_dev, w_dev, ws_dev, torch.bfloat16, None)
    torch.cuda.synchronize()
    begin_ns = time.monotonic_ns()
    begin_epoch_ns = time.time_ns()
    print(
        f"PROFILE_GROUP,triton,{case.id},{case.m},{case.n},{case.k},"
        f"calls={calls}", flush=True,
    )
    for _ in range(calls):
        matmul_int8(a_dev, as_dev, w_dev, ws_dev, torch.bfloat16, None)
    torch.cuda.synchronize()
    return begin_ns, time.monotonic_ns(), begin_epoch_ns, time.time_ns()


# ═══════════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    phase = sys.argv[1]
    is_profile_batch = phase == "profile-batch"
    is_correctness = phase == "correctness"
    if not is_profile_batch and not is_correctness:
        raise RuntimeError(
            "usage: evaluate.py correctness|profile-batch candidate|triton CALLS"
        )

    report_path = Path(_env("METAINFER_REPORT_PATH"))
    weight_root = Path(_env("METAINFER_WEIGHT_BUNDLE")).resolve()
    artifact_dir = Path(_env("METAINFER_BUILD_ARTIFACT_DIR")).resolve()
    role = _env("METAINFER_EVALUATION_ROLE")

    if phase != _env("METAINFER_EVALUATION_PHASE"):
        raise RuntimeError("phase mismatch")

    device = torch.device("cuda:0")

    weights = WeightStore(weight_root)
    batch_impl = sys.argv[2] if is_profile_batch and len(sys.argv) > 2 else ""
    needs_candidate = role != "baseline" and (
        not is_profile_batch or batch_impl == "candidate")
    candidate = Candidate(artifact_dir) if needs_candidate else None

    if is_profile_batch:
        if batch_impl not in ("candidate", "triton"):
            raise RuntimeError("profile-batch implementation must be candidate or triton")
        calls = int(sys.argv[3]) if len(sys.argv) > 3 else 120
        if calls <= 0:
            raise RuntimeError("profile-batch calls must be positive")
        if batch_impl == "candidate" and candidate is None:
            raise RuntimeError("candidate profile requested without candidate artifact")
        cases = _public_cases()
        selected = {
            token.strip() for token in os.environ.get(
                "METAINFER_PROFILE_CASE_IDS", ""
            ).split(",") if token.strip()
        }
        if selected:
            known = {case.id for case in cases}
            unknown = sorted(selected - known)
            if unknown:
                raise RuntimeError(f"unknown profile case ids: {unknown}")
            cases = [case for case in cases if case.id in selected]
        profiled_cases = []
        for found in cases:
            if batch_impl == "triton":
                begin_ns, end_ns, begin_epoch_ns, end_epoch_ns = _profile_batch_case_triton(
                    weights, found, device, calls)
            else:
                assert candidate is not None
                begin_ns, end_ns, begin_epoch_ns, end_epoch_ns = _profile_batch_case_candidate(
                    candidate, weights, found, device, calls)
            profiled_cases.append({
                "id": found.id, "m": found.m, "n": found.n, "k": found.k,
                "host_monotonic_begin_ns": begin_ns,
                "host_monotonic_end_ns": end_ns,
                "host_epoch_begin_ns": begin_epoch_ns,
                "host_epoch_end_ns": end_epoch_ns,
            })
        write_json(report_path, {
            "passed": True,
            "implementation": batch_impl,
            "calls_per_case": calls,
            "case_ids": [case.id for case in cases],
            "cases": profiled_cases,
            "timed_scope": "core implementation launches only",
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
