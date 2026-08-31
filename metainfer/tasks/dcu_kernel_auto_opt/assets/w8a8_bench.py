#!/usr/bin/env python3
"""Trusted correctness and performance harness for the gfx928 W8A8 adapter."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import statistics
import sys
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:  # Allow CPU-only CI to import pure helpers.
    torch = None  # type: ignore[assignment]


def load_module(module_path: Path, name: str):
    spec = importlib.util.spec_from_file_location(
        name, module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(module_path.parent))
    spec.loader.exec_module(module)
    return module


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def validate_profile_protocol(
    profile_only: bool,
    warmups: int,
    samples: int,
    replays_per_sample: int,
) -> None:
    """Keep one unambiguous operator replay after the PMC marker."""
    if profile_only and (warmups, samples, replays_per_sample) != (0, 1, 1):
        raise ValueError(
            "--profile-only requires --warmups 0 --samples 1 "
            "--replays-per-sample 1"
        )


class CapturedGraphRunner:
    """Small internal runner used by the trusted benchmark."""

    def __init__(
        self,
        graph: torch.cuda.CUDAGraph,
        output: torch.Tensor,
        stream: torch.cuda.Stream,
    ) -> None:
        self.graph = graph
        self.output = output
        self.stream = stream

    def replay(self) -> torch.Tensor:
        self.graph.replay()
        return self.output


def capture_candidate_graph(
    candidate, output: torch.Tensor
) -> CapturedGraphRunner:
    """Capture a zero-argument candidate on a non-default HIP stream."""
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        candidate()
    capture_stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        candidate()
    runner = CapturedGraphRunner(graph, output, capture_stream)
    runner.replay()
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()
    return runner


def exact_w8a8_reference(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
) -> torch.Tensor:
    """Compute the contract's integer dot exactly before float scaling.

    CPU int64 avoids treating a float32 GEMM's accumulation order as the
    W8A8 contract. Large-prefill callers should run this once per candidate,
    then use ``--skip-correctness`` only for repeated timing/profiling of the
    exact same source and deterministic inputs.
    """
    device = a.device
    dot = torch.mm(
        a.to(device="cpu", dtype=torch.int64),
        b.to(device="cpu", dtype=torch.int64),
    )
    scaled = (
        dot.to(torch.float32)
        * a_scale.to(device="cpu", dtype=torch.float32)
        * b_scale.to(device="cpu", dtype=torch.float32).T
    )
    return scaled.to(torch.bfloat16).to(device)


def main() -> int:
    if torch is None:
        raise RuntimeError(
            "w8a8_bench.py requires PyTorch in the DCU benchmark environment"
        )
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--m", type=int)
    parser.add_argument("--n", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--warmups", type=int, default=100)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--replays-per-sample", type=int, default=100)
    parser.add_argument("--reference-cache-dir", type=Path)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--skip-correctness",
        action="store_true",
        help=(
            "Skip the CPU int64 reference when the exact same source and "
            "deterministic inputs already passed trusted correctness."
        ),
    )
    parser.add_argument(
        "--profile-only",
        action="store_true",
        help="Alias for --skip-correctness used by trusted PMC profiling.",
    )
    args = parser.parse_args()

    validate_profile_protocol(
        args.profile_only,
        args.warmups,
        args.samples,
        args.replays_per_sample,
    )

    if args.self_test:
        a = torch.tensor(
            [[1, -2, 3], [-4, 5, -6]], dtype=torch.int8
        )
        b = torch.tensor(
            [[7, -8], [9, 10], [-11, 12]], dtype=torch.int8
        )
        a_scale = torch.tensor(
            [[0.5], [0.25]], dtype=torch.float32
        )
        b_scale = torch.tensor(
            [[2.0], [4.0]], dtype=torch.float32
        )
        actual = exact_w8a8_reference(a, b, a_scale, b_scale)
        expected = torch.tensor(
            [[-44.0, 16.0], [41.5, 10.0]], dtype=torch.bfloat16
        )
        passed = bool(torch.equal(actual, expected))
        print(json.dumps({
            "self_test": "exact_w8a8_reference",
            "passed": passed,
            "device": "cpu",
            "torch_version": torch.__version__,
            "actual": actual.float().tolist(),
            "expected": expected.float().tolist(),
        }, sort_keys=True))
        return 0 if passed else 4

    missing = [
        name for name in ("source", "m", "n", "k")
        if getattr(args, name) is None
    ]
    if missing:
        parser.error(
            "the following arguments are required unless --self-test is "
            f"used: {', '.join('--' + name for name in missing)}"
        )

    count = torch.cuda.device_count()
    props = torch.cuda.get_device_properties(0) if count else None
    if args.probe:
        print(json.dumps({
            "visible_devices": count,
            "logical_device": 0,
            "device_name": props.name if props else "",
            "multi_processor_count": (
                props.multi_processor_count if props else 0
            ),
            "cudagraph_available": hasattr(torch.cuda, "CUDAGraph"),
            "python_graph_api": "torch.cuda.CUDAGraph",
        }))
        return (
            0
            if count == 1 and hasattr(torch.cuda, "CUDAGraph")
            else 3
        )

    if min(args.m, args.n, args.k) <= 0:
        raise ValueError("M, N and K must be positive")
    source = args.source.resolve()
    fixed_contract = source / "int8_w8a8_gemm_api.py"
    if fixed_contract.is_file():
        backend = load_module(
            source / "w8a8_backend.py", "metainfer_w8a8_backend"
        )
        backend.load_extension()
        api = load_module(fixed_contract, "metainfer_w8a8_contract")
        fixed_api = True
    else:
        # Backward compatibility for pre-contract extracted repositories.
        api = load_module(
            source / "w8a8_gemm.py", "metainfer_w8a8_candidate"
        )
        api.load_extension()
        fixed_api = False

    torch.manual_seed(20260724 + args.m + args.n + args.k)
    a = torch.randint(
        -127, 128, (args.m, args.k), dtype=torch.int8, device="cuda"
    )
    b = torch.randint(
        -127, 128, (args.k, args.n), dtype=torch.int8, device="cuda"
    )
    a_scale = torch.rand(
        (args.m, 1), dtype=torch.float32, device="cuda"
    ) * 0.01
    b_scale = torch.rand(
        (args.n, 1), dtype=torch.float32, device="cuda"
    ) * 0.01
    out = torch.empty(
        (args.m, args.n), dtype=torch.bfloat16, device="cuda"
    )

    if fixed_api:
        packed_weight, packed_weight_scale = api.prepare_weight(b, b_scale)
        workspace = api.allocate_workspace(
            args.m, args.n, args.k, a.device
        )

        def candidate() -> None:
            api.w8a8_gemm_out(
                a,
                packed_weight,
                a_scale,
                packed_weight_scale,
                out,
                workspace,
            )
        path = "w8a8_gemm_out"
    elif args.m <= 16:
        workspace = api.empty_optimized_workspace(a, b)

        def candidate() -> None:
            api.gemm_out_optimized(a, b, a_scale, b_scale, out, workspace)
        path = "gemm_out_optimized"
    else:
        def candidate() -> None:
            api.gemm_out_prefill(a, b, a_scale, b_scale, out)
        path = "gemm_out_prefill"

    try:
        graph_runner = capture_candidate_graph(candidate, out)
    except Exception as exc:
        print(json.dumps({
            "passed": False,
            "operator": "int8_w8a8_gemm",
            "path": path,
            "shape": {"M": args.m, "N": args.n, "K": args.k},
            "visible_devices": count,
            "device_name": props.name if props else "",
            "graph_capture_passed": False,
            "timing_mode": "cuda_graph_replay",
            "python_callable": True,
            "graph_error": f"{type(exc).__name__}: {exc}",
            "mismatch_count": None,
            "first_mismatch": None,
        }, sort_keys=True))
        return 0

    correctness_checked = not (
        args.skip_correctness or args.profile_only
    )
    reference_cache_hit = False
    mismatch_count = None
    passed = True
    max_abs_error = None
    first_mismatch = None
    if correctness_checked:
        reference_path = None
        if args.reference_cache_dir is not None:
            args.reference_cache_dir.mkdir(parents=True, exist_ok=True)
            reference_path = args.reference_cache_dir / (
                f"exact-int64-v1-m{args.m}-n{args.n}-k{args.k}.pt"
            )
        if reference_path is not None and reference_path.is_file():
            cached_reference = torch.load(
                reference_path, map_location="cpu", weights_only=True
            )
            if (
                cached_reference.shape != out.shape
                or cached_reference.dtype != torch.bfloat16
            ):
                raise RuntimeError(
                    f"invalid cached W8A8 reference: {reference_path}"
                )
            reference = cached_reference.to(a.device)
            reference_cache_hit = True
        else:
            reference = exact_w8a8_reference(a, b, a_scale, b_scale)
            if reference_path is not None:
                temporary = reference_path.with_name(
                    f"{reference_path.name}.tmp-{os.getpid()}"
                )
                torch.save(reference.to("cpu"), temporary)
                temporary.replace(reference_path)
        mismatch_mask = out != reference
        mismatch_count = int(mismatch_mask.sum().item())
        passed = mismatch_count == 0
        absolute_error = (out.float() - reference.float()).abs()
        max_abs_error = float(absolute_error.max().item())
        if mismatch_count:
            first_flat = int(
                mismatch_mask.reshape(-1).nonzero()[0].item()
            )
            first_m = first_flat // args.n
            first_n = first_flat % args.n
            first_mismatch = {
                "flat_index": first_flat,
                "m": first_m,
                "n": first_n,
                "actual": float(out[first_m, first_n].float().item()),
                "expected": float(reference[first_m, first_n].float().item()),
                "abs_error": float(absolute_error[first_m, first_n].item()),
            }

    if args.replays_per_sample <= 0:
        raise ValueError("replays-per-sample must be positive")
    for _ in range(args.warmups):
        graph_runner.replay()
    graph_runner.stream.synchronize()
    profile_marker_emitted = False
    if args.profile_only:
        # Graph creation performs an eager warmup and a validation replay.
        # Emit a non-W8A8 dispatch after both so the PMC parser can identify
        # the single timed operator replay that follows. The marker is outside
        # the timing events and does not change candidate inputs or workspace.
        with torch.cuda.stream(graph_runner.stream):
            out.zero_()
        graph_runner.stream.synchronize()
        profile_marker_emitted = True
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples_us: list[float] = []
    for _ in range(args.samples):
        with torch.cuda.stream(graph_runner.stream):
            begin.record()
            for _ in range(args.replays_per_sample):
                graph_runner.replay()
            end.record()
        end.synchronize()
        samples_us.append(
            float(begin.elapsed_time(end))
            * 1000.0
            / args.replays_per_sample
        )

    median_us = statistics.median(samples_us)
    seconds = median_us * 1.0e-6
    logical_ops = 2.0 * args.m * args.n * args.k
    algorithmic_bytes = (
        args.m * args.k
        + args.k * args.n
        + 4 * (args.m + args.n)
        + 2 * args.m * args.n
    )
    print(json.dumps({
        "passed": passed,
        "operator": "int8_w8a8_gemm",
        "path": path,
        "shape": {"M": args.m, "N": args.n, "K": args.k},
        "visible_devices": count,
        "device_name": props.name if props else "",
        "graph_capture_passed": True,
        "timing_mode": "cuda_graph_replay",
        "python_callable": True,
        "python_graph_api": "torch.cuda.CUDAGraph",
        "median_us": median_us,
        "p90_us": percentile(samples_us, 0.9),
        "min_us": min(samples_us),
        "max_us": max(samples_us),
        "latency_samples_us": samples_us,
        "logical_ops": logical_ops,
        "logical_tops": logical_ops / seconds / 1.0e12,
        "algorithmic_bytes": algorithmic_bytes,
        "algorithmic_bandwidth_gb_s": (
            algorithmic_bytes / seconds / 1.0e9
        ),
        "metric_semantics": {
            "logical_tops": (
                "INT8 GEMM logical operation rate; one multiply and one add "
                "count as two operations."
            ),
            "algorithmic_bandwidth_gb_s": (
                "Algorithmic minimum bytes divided by unprofiled median "
                "latency; this is not measured HBM traffic."
            ),
        },
        "max_abs_error": max_abs_error,
        "mismatch_count": mismatch_count,
        "first_mismatch": first_mismatch,
        "correctness_checked": correctness_checked,
        "profile_only": args.profile_only,
        "profile_replay_marker_emitted": profile_marker_emitted,
        "reference_cache_hit": reference_cache_hit,
        "warmup": args.warmups,
        "samples": args.samples,
        "replays_per_sample": args.replays_per_sample,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
