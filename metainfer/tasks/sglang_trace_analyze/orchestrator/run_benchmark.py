#!/usr/bin/env python3
"""Wrapper script for sglang.bench_one_batch_server.

Called by the orchestrator pipeline in two modes:

    # Mapping run — one batch size, --disable-cuda-graph
    python run_benchmark.py --config bench_config.json --mapping-only

    # Formal runs — one or all batch sizes, CUDA Graph ON
    python run_benchmark.py --config bench_config.json --formal-only [--single-batch N]

Environment variables required for K100/HIP are set inside this script
so callers don't need to source them externally.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

# ── K100 / HIP environment — must match /workspace/sglang/scripts/run_traces.sh ──

_K100_ENV = {
    "SGL_CHUNKED_PREFIX_CACHE_THRESHOLD": "0",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "1200",
    "GLIBC_TUNABLES": "glibc.rtld.optional_static_tls=0x40000",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "HIP_KERNEL_BATCH_CEILING": "100",
    "GPU_MAX_HW_QUEUES": "3",
    # "HIP_GRAPH_ACCUMULATE_DISPATCH": "0",  # torchprof needs this
    "HIP_H2D_DISABLE_COPY_BUFFER": "0",
    "HIP_D2H_DISABLE_COPY_BUFFER": "0",
    "HIP_H2D_DIRECT_COPY_THRESHOLD": "32768",
    "HIP_H2D_HSAAPI_COPY_THRESHOLD": "32768",
    "HIP_D2H_DIRECT_COPY_THRESHOLD": "512",
    "HIP_D2H_HSAAPI_COPY_THRESHOLD": "512",
    "USE_DCU_CUSTOM_ALLREDUCE": "1",
    "HIP_KERNEL_EVENT_SYSTENFENCE": "1",
    "SGLANG_USE_FP8_W8A8_MOE": "0",
    "SGLANG_USE_LIGHTOP": "0",
    "SGLANG_ROCM_USE_AITER_MOE": "0",
    "SGLANG_OPT_USE_FUSED_HASH_TOPK": "false",
    "SGLANG_OPT_SWIGLU_CLAMP_FUSION": "false",
    "SGLANG_TOPK_TRANSFORM_512_TORCH": "false",
    "SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK": "false",
    "SGLANG_NSA_FUSE_TOPK": "false",
    "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "0",
    "SGLANG_APPLY_CONFIG_BACKUP": "none",
    "SGLANG_DSV4_MODE": "2604",
    "SGLANG_OPT_BF16_FP32_GEMM_ALGO": "torch",
    "SGLANG_OPT_USE_HIP_PAGED_MQA_LOGITS": "1",
    "SGLANG_OPT_USE_HIP_MHC_PRE": "1",
    "SGLANG_OPT_USE_HIP_MHC_POST": "1",
    "SGLANG_OPT_USE_HIP_INT8_SCALED_MM": "0",
    "SGLANG_OPT_USE_LMSLIM_INT8_QUANT": "1",
    "SGLANG_OPT_USE_W8A8_MARLIN_GEMM": "1",
}

_PYTHONPATH_EXTRA = "/workspace/sglang/sglang-v0.5.15_k100/python"


def _setup_env():
    """Apply K100 env vars and PYTHONPATH once per process."""
    for k, v in _K100_ENV.items():
        if k not in os.environ:
            os.environ[k] = v
    pp = os.environ.get("PYTHONPATH", "")
    if _PYTHONPATH_EXTRA not in pp:
        os.environ["PYTHONPATH"] = f"{_PYTHONPATH_EXTRA}:{pp}" if pp else _PYTHONPATH_EXTRA


def build_dir_name(args: Dict[str, Any], disable_cuda_graph: bool = False) -> str:
    """Build sglang-style directory name from config."""
    parts = [args["version"], f"tp{args['tp_size']}", f"pp{args['pp_size']}"]
    parts.append("nograph" if disable_cuda_graph else "graph")
    return "_".join(parts)


def run_benchmark(
    args: Dict[str, Any],
    dir_name: str,
    batch_size: int,
    *,
    disable_cuda_graph: bool = False,
) -> bool:
    """Run a single bench_one_batch_server invocation."""
    output_dir = os.path.join(
        args["output_dir"], "mapping" if disable_cuda_graph else f"bs_{batch_size}"
    )
    profile_prefix = f"{dir_name}_"

    cmd = [
        sys.executable, "-m", "sglang.bench_one_batch_server",
        "--model-path", args["model_path"],
        "--tp-size", str(args["tp_size"]),
        "--pp-size", str(args["pp_size"]),
        "--batch-size", str(batch_size),
        "--input-len", str(args["input_len"]),
        "--output-len", str(args["output_len"]),
        "--run-name", dir_name,
        "--show-report",
        "--dataset-name", "random-ids",
        "--fake-prefill",
        "--profile",
        "--profile-start-step", str(args.get("profile_start_step", 5)),
        "--profile-steps", str(args.get("profile_steps", 5)),
        "--profile-by-stage",
        "--profile-prefix", profile_prefix,
        "--profile-output-dir", output_dir,
        "--disable-radix-cache",
        "--chunked-prefill-size", "4096",
        "--kv-cache-dtype", "auto",
        "--disable-flashinfer-autotune",
        "--reasoning-parser", "deepseek-v4",
        "--tool-call-parser", "deepseekv4",
        "--enable-metrics",
    ]

    if disable_cuda_graph:
        cmd.append("--disable-cuda-graph")
    else:
        cmd.extend(["--cuda-graph-bs", str(batch_size)])

    print(f"\n{'='*80}")
    print(f"Running batch_size={batch_size}"
          f"{' (CUDA Graph OFF)' if disable_cuda_graph else ''}")
    print(f"  profile-output-dir: {output_dir}")
    print(f"  profile-prefix:      {profile_prefix}")
    print(f"{'='*80}\n")

    try:
        subprocess.run(cmd, check=True, timeout=3600)
    except subprocess.TimeoutExpired:
        print(f"\n[FAILED] batch_size={batch_size}: timed out after 1 hour\n")
        return False
    except subprocess.CalledProcessError as e:
        print(f"\n[FAILED] batch_size={batch_size}: exit code {e.returncode}\n")
        return False
    return True


def main():
    _setup_env()
    parser = argparse.ArgumentParser(
        description="Run sglang bench_one_batch_server with torch profiler"
    )
    parser.add_argument("--config", required=True,
                       help="Path to JSON benchmark config")
    parser.add_argument("--mapping-only", action="store_true",
                       help="Run only the mapping benchmark (--disable-cuda-graph, one batch)")
    parser.add_argument("--formal-only", action="store_true",
                       help="Run formal benchmarks (CUDA Graph ON, one or all batches)")
    parser.add_argument("--single-batch", type=int, default=None,
                       help="When --formal-only, run only this batch size")

    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: config file not found: {args.config}")
        return 1

    with open(config_path) as f:
        cfg = json.load(f)

    if args.mapping_only:
        dir_name = build_dir_name(cfg, disable_cuda_graph=True)
        bs = cfg.get("mapping_batch_size", 8)
        ok = run_benchmark(cfg, dir_name, bs, disable_cuda_graph=True)
        return 0 if ok else 1

    if args.formal_only:
        dir_name = build_dir_name(cfg, disable_cuda_graph=False)
        batch_sizes: List[int] = cfg.get("batch_sizes", [1])
        if args.single_batch is not None:
            if args.single_batch in batch_sizes:
                batch_sizes = [args.single_batch]
            else:
                print(f"ERROR: --single-batch {args.single_batch} not in "
                      f"configured batch_sizes {batch_sizes}")
                return 1

        succeeded, failed = [], []
        for bs in batch_sizes:
            ok = run_benchmark(cfg, dir_name, bs)
            (succeeded if ok else failed).append(bs)

        print(f"\n{'='*80}")
        print(f"Completed: {len(succeeded)} succeeded, {len(failed)} failed")
        if succeeded:
            print(f"  Succeeded batches: {succeeded}")
        if failed:
            print(f"  Failed batches: {failed}")
        return 0 if not failed else 1

    print("ERROR: must specify --mapping-only or --formal-only")
    return 1


if __name__ == "__main__":
    sys.exit(main())
