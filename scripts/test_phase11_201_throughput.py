"""
Phase 11 Stage 2 — Throughput baseline & gate.

Measures end-to-end throughput with warmup, validates correctness against
the Phase 10 greedy-decode baseline, and enforces a throughput floor gate.

Gates:
  T201-THRESHOLD : Single GPU throughput >= 3.0 tok/s
  T201-CORRECT   : Output matches Phase 10 greedy baseline exactly

Usage:
  MODEL_DIR=/path/to/model python scripts/test_phase11_201_throughput.py
  MODEL_DIR=/path/to/model torchrun --nproc_per_node=4 scripts/test_phase11_201_throughput.py

Trace Source: physical_trace_tp4_rank0.json [runtime] Phase 10 baseline ~3.9 tok/s
"""

import os
import sys
import time
from pathlib import Path

# --- env ---
_root = os.environ.get(
    "AGENT_INFER_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
sys.path.insert(0, _root)

os.environ.setdefault("META_INFER_LOG_RANK0_ONLY", "1")
os.environ.setdefault("META_INFER_CUDA_GRAPH", "0")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
os.environ.setdefault("NCCL_DEBUG", "WARN")

import torch
from llm_engine import LLMEngine

# --- test params (pinned to Phase 10 baseline) ---
PROMPT = "苏州园林的特点是"
EXPECTED = "（ ） A：建筑与园林结合 B：建筑与自然结合 C：建筑与山水结合 D：建筑"
MAX_TOKENS = 24
WARMUP_TOKENS = 8
MIN_THROUGHPUT = 3.0       # tok/s floor (baseline ~3.9)
TRACE_SRC = "physical_trace_tp4_rank0.json [runtime] Phase 10 baseline"

# --- helpers ---
_RANK = int(os.environ.get("RANK", 0))
_WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
_IS_TP = _WORLD_SIZE > 1


def _log(msg: str) -> None:
    if _RANK == 0:
        print(msg, flush=True)


def _fail(tag: str, detail: str) -> int:
    _log(f"[{tag}] FAIL: {detail}")
    return 1


# --- main ---
def main() -> int:
    model_dir = os.environ.get("MODEL_DIR")
    if not model_dir:
        return _fail("T201-THRESHOLD", "MODEL_DIR not set")

    _log("=== Phase 11 Stage 2: Throughput Baseline ===")
    _log(f"TP:      {'TP=' + str(_WORLD_SIZE) if _IS_TP else 'single GPU'}")
    _log(f"Model:   {model_dir}")
    _log(f"Prompt:  {PROMPT}")

    engine = LLMEngine(
        model_dir=Path(model_dir),
        inference_backend="qwen_tp",
        max_num_seqs=4,
    )

    # Warmup — absorb one-time allocs + kernel JIT
    _log("[WARMUP] 8 tokens...")
    _ = engine.generate(PROMPT, max_new_tokens=WARMUP_TOKENS, temperature=0.0)
    torch.cuda.synchronize()

    # Measured run
    _log(f"[MEASURE] {MAX_TOKENS} tokens...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = engine.generate(PROMPT, max_new_tokens=MAX_TOKENS, temperature=0.0)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    tps = MAX_TOKENS / elapsed if elapsed > 0 else 0.0

    _log(f"  Elapsed:    {elapsed:.3f}s")
    _log(f"  Throughput: {tps:.1f} tok/s")

    # Correctness gate
    if _RANK == 0 and out != EXPECTED:
        return _fail(
            "T201-CORRECT",
            f"Output mismatch.\n  Got:      {out!r}\n  Expected: {EXPECTED!r}",
        )
    _log("[T201-CORRECT] PASS: output matches Phase 10 greedy baseline exactly")

    # Throughput floor
    if tps < MIN_THROUGHPUT:
        return _fail(
            "T201-THRESHOLD",
            f"{tps:.1f} tok/s < {MIN_THROUGHPUT} tok/s minimum",
        )
    _log(f"[T201-THRESHOLD] PASS: {tps:.1f} tok/s >= {MIN_THROUGHPUT}")

    if _IS_TP:
        _log(f"[T201-TP] TP={_WORLD_SIZE} throughput: {tps:.1f} tok/s")

    _log("\nPHASE11_201_THROUGHPUT: ALL TESTS PASSED")
    _log(f"Source: {TRACE_SRC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
